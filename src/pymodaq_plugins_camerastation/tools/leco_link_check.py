"""Check the LECO link between computer B (Camera Station relay) and computer A, without any hardware.

On B (sender): a CameraStation relay viewer, with its plot, connected to the coordinator, publishes simulated
camera results (10 values per camera, like a beam camera, plus the Stats heartbeat)::

    python -m pymodaq_plugins_camerastation.tools.leco_link_check sender --host 192.168.0.100 --seconds 120

On A (receiver): a LECODirector 0D viewer, with its plot, asks the relay to grab and counts what it receives.
It does not need Camera Station installed (run this file by its path from a clone)::

    python leco_link_check.py receiver --host localhost --seconds 30

Start the coordinator first (on A), then the sender, then the receiver. Both print a short report; the receiver
exits with code 0 only when publications arrived with growing frame counters.
"""
from __future__ import annotations

import argparse
import inspect
import logging
import math
import sys
import time

DEFAULT_NAME = 'CameraStation_linkcheck'
BEAM = ('x', 'y', 'sx', 'sy', 'sum', 'peak', 'sat', 'valid', 'age', 'frame')


def _app():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        try:
            from pymodaq_gui.qt_utils import mkQApp
        except ImportError:
            from pymodaq_gui.utils.utils import mkQApp
        app = mkQApp('leco_link_check')
    return app


def _wait(pred=lambda: False, timeout=1.0):
    app = _app()
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    app.processEvents()
    return pred()


def _find(root, name):
    for c in root.children():
        if c.name() == name:
            return c
        found = _find(c, name)
        if found is not None:
            return found
    return None


class _Errors(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _viewer(title, plugin):
    """A 0D DAQ_Viewer with its user interface (plot), like in a dashboard"""
    from qtpy import QtWidgets
    from pymodaq_gui.utils.dock import DockArea
    from pymodaq.control_modules.daq_viewer import DAQ_Viewer
    win = QtWidgets.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.setWindowTitle(f'leco_link_check: {title}')
    win.show()
    det = DAQ_Viewer(area, title=title)
    det.daq_type = 'DAQ0D'
    det.detector = plugin
    det._link_check_window = win
    return det


def _init(det, timeout=30):
    inits = []
    det.init_signal.connect(inits.append)
    det.init_hardware(True)
    return _wait(lambda: inits, timeout) and inits[-1]


def sender(host: str, port: int, name: str, seconds: float, rate_hz: float) -> int:
    _app()
    errors = _Errors()
    logging.getLogger().addHandler(errors)
    from pymodaq_plugins_camerastation.relay import bus, MAX_PLOTTED_CHANNELS
    relay = _viewer(name, 'CameraStation')
    leco = relay.settings.child('main_settings', 'leco')
    for key, value in (('host', host), ('port', port), ('leco_name', name)):
        _find(leco, key).setValue(value)
    if not _init(relay):
        print('SENDER: the CameraStation relay plugin did not initialize')
        return 2
    _find(leco, 'connect_leco_server').setValue(True)
    if not _wait(lambda: relay.settings['main_settings', 'leco', 'leco_connected'], 15):
        print(f'SENDER: not connected to the coordinator {host}:{port} (coordinator started? firewall? address?)')
        return 3
    print(f'SENDER: connected to {host}:{port} as "{name}"; plot colors limit: {MAX_PLOTTED_CHANNELS}', flush=True)
    key = next(k for k in ('send_to_leco', 'send_to_tcpip') if k in inspect.signature(relay.grab_data).parameters)
    relay.grab(**{key: True})
    sent = restarts = 0
    t0 = time.time()
    next_report = t0 + 10
    while time.time() - t0 < seconds:
        if not relay.grab_state:              # a Grab of A toggled the running grab off: restart it, as the station
            relay.grab(**{key: True})
            restarts += 1
        sent += 1
        cam = {lab: float(sent) for lab in BEAM}
        cam['valid'] = 1.0
        bus.publish({'Cam1': cam, 'Stats': {'frame_counter': float(sent), 't_wall': time.time()}})
        _wait(timeout=1.0 / rate_hz)
        if time.time() >= next_report:
            next_report += 10
            print(f'SENDER: {sent} publications, grab restarted {restarts} times, errors {len(errors.messages)}',
                  flush=True)
    relay.stop_grab()
    _wait(timeout=0.5)
    relay.quit_fun()
    _wait(timeout=1.0)
    plot_errors = [m for m in errors.messages if 'index out of range' in m]
    print(f'SENDER done: {sent} publications, grab restarted {restarts} times, '
          f'{len(errors.messages)} errors ({len(plot_errors)} "index out of range")')
    for m in errors.messages[:5]:
        print('   ', m)
    return 0 if not plot_errors else 1


def receiver(host: str, port: int, actor: str, seconds: float, title: str) -> int:
    _app()
    errors = _Errors()
    logging.getLogger().addHandler(errors)
    director = _viewer(title, 'LECODirector')
    ds = director.settings.child('detector_settings')
    for key, value in (('host', host), ('port', port), ('actor_name', actor)):
        p = _find(ds, key)
        if p is not None:
            p.setValue(value)
    if not _init(director):
        print(f'RECEIVER: the LECODirector viewer did not initialize ({host}:{port}, actor "{actor}")')
        return 2
    received = []
    director.grab_done_signal.connect(received.append)
    _wait(timeout=1.0)
    director.grab()
    t0 = time.time()
    next_report = t0 + 5
    while time.time() - t0 < seconds:
        _wait(timeout=0.2)
        if time.time() >= next_report:
            next_report += 5
            print(f'RECEIVER: {len(received)} publications received', flush=True)
    director.stop_grab()
    _wait(timeout=0.5)
    counters = []
    keys = []
    for dte in received:
        for dwa in dte:
            if dwa.name == 'Stats' and 'frame_counter' in dwa.labels:
                counters.append(float(dwa.data[list(dwa.labels).index('frame_counter')][0]))
    if received:
        keys = [f'{title}/{dwa.name}/{lab}' for dwa in received[-1] for lab in dwa.labels]
    director.quit_fun()
    _wait(timeout=1.0)
    ok = len(counters) >= 2 and counters == sorted(counters) and not any(math.isnan(c) for c in counters)
    print(f'RECEIVER done: {len(received)} publications, frame counters {counters[:1]}...{counters[-1:]}, '
          f'{len(errors.messages)} errors -> {"OK" if ok else "FAILED"}')
    print('   keys as seen by Laser Control:', ', '.join(keys) if keys else '-')
    for m in errors.messages[:5]:
        print('   ', m)
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='role', required=True)
    s = sub.add_parser('sender', help='computer B: simulated Camera Station relay')
    s.add_argument('--host', required=True, help='address of the LECO coordinator (computer A)')
    s.add_argument('--port', type=int, default=12300)
    s.add_argument('--name', default=DEFAULT_NAME, help='LECO name of the relay')
    s.add_argument('--seconds', type=float, default=120.0)
    s.add_argument('--rate', type=float, default=2.0, help='publications per second')
    r = sub.add_parser('receiver', help='computer A: LECODirector viewer')
    r.add_argument('--host', default='localhost', help='address of the LECO coordinator')
    r.add_argument('--port', type=int, default=12300)
    r.add_argument('--actor', default=DEFAULT_NAME, help='LECO name of the relay of computer B')
    r.add_argument('--seconds', type=float, default=30.0)
    r.add_argument('--title', default='LECO_B', help='title of the viewer (first part of the channel keys)')
    a = parser.parse_args(argv)
    if a.role == 'sender':
        return sender(a.host, a.port, a.name, a.seconds, a.rate)
    return receiver(a.host, a.port, a.actor, a.seconds, a.title)


if __name__ == '__main__':
    import os
    try:
        code = main()
    finally:
        sys.stdout.flush()
    os._exit(code)
