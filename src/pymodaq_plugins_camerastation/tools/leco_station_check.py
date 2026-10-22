"""Run, on computer B, the whole Camera Station chain with simulated cameras and send it to computer A through LECO.

Two CamSim cameras (a disk and a beam, 10 values) -> the Camera Station analysis -> the CameraStation relay viewer
(with its plot) connected to the coordinator of computer A. Every file of the station (settings, references,
recordings) goes to a temporary folder: the station settings of this machine are not touched. No dashboard preset
is written, no real camera is opened.

    python -m pymodaq_plugins_camerastation.tools.leco_station_check --host 192.168.0.100 --seconds 300

On computer A: ``leco_link_check.py receiver --actor CameraStation_sim`` or Laser Control's ``leco_lab_check``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
import types
from pathlib import Path

from .leco_link_check import _Errors, _app, _find, _init, _viewer, _wait

DEFAULT_NAME = 'CameraStation_sim'


def run(host: str, port: int, name: str, seconds: float) -> int:
    _app()
    errors = _Errors()
    logging.getLogger().addHandler(errors)
    from qtpy import QtWidgets
    from pymodaq_gui.utils.dock import DockArea
    from pymodaq_plugins_camerastation.relay import bus
    from pymodaq_plugins_camerastation.ui.station import CameraStationApp
    from pymodaq_plugins_camerastation.utils import Config

    cams = []
    for title, scene in (('Disk cam', 'disk'), ('Output cam', 'beam')):
        det = _viewer(title, 'CamSim', 'DAQ2D')
        if not _init(det):
            print(f'STATION: {title} (CamSim) did not initialize')
            return 2
        for key, value in (('scene', scene), ('fps', 10.0)):
            _find(det.settings.child('detector_settings'), key).setValue(value)
        cams.append(det)
    relay = _viewer(name, 'CameraStation')
    leco = relay.settings.child('main_settings', 'leco')
    for key, value in (('host', host), ('port', port), ('leco_name', name)):
        _find(leco, key).setValue(value)
    if not _init(relay):
        print('STATION: the CameraStation relay did not initialize')
        return 2
    _find(leco, 'connect_leco_server').setValue(True)
    if not _wait(lambda: relay.settings['main_settings', 'leco', 'leco_connected'], 15):
        print(f'STATION: relay not connected to the coordinator {host}:{port}')
        return 3
    print(f'STATION: relay "{name}" connected to {host}:{port}', flush=True)

    data = Path(tempfile.mkdtemp(prefix='leco_station_check_'))
    win = QtWidgets.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.setWindowTitle('leco_station_check: Camera Station')
    win.show()
    mm = types.SimpleNamespace(detectors=cams + [relay])
    st = CameraStationApp(mm, Config(), window=None, data_folder=data)
    st.build(area)
    disk = st.add_camera()
    out = st.add_camera()
    st.params.child('cameras', disk.id, 'kind').setValue('disk')
    st.params.child('cameras', disk.id, 'name').setValue('Disk')
    st.params.child('cameras', out.id, 'name').setValue('Output')
    st.grab_all(True)
    n0 = bus.n_published
    t0 = time.time()
    next_report = t0 + 10
    while time.time() - t0 < seconds:
        _wait(timeout=0.2)
        if time.time() >= next_report:
            next_report += 10
            res = st.engine.snapshot(out.id).get('result', {})
            print(f'STATION: {bus.n_published - n0} publications, Output valid={res.get("valid")} '
                  f'x={res.get("x", float("nan")):.1f}, relay grabbing={relay.grab_state}, '
                  f'errors {len(errors.messages)}', flush=True)
    published = bus.n_published - n0
    st.grab_all(False)
    _wait(timeout=0.5)
    st.shutdown()
    for det in cams + [relay]:
        try:
            det.quit_fun()
        except Exception:
            pass
    _wait(timeout=1.0)
    plot_errors = [m for m in errors.messages if 'index out of range' in m]
    print(f'STATION done: {published} publications, {len(errors.messages)} errors '
          f'({len(plot_errors)} "index out of range"); station files in {data}')
    for m in errors.messages[:5]:
        print('   ', m)
    return 0 if published and not plot_errors else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--host', required=True, help='address of the LECO coordinator (computer A)')
    parser.add_argument('--port', type=int, default=12300)
    parser.add_argument('--name', default=DEFAULT_NAME, help='LECO name of the relay')
    parser.add_argument('--seconds', type=float, default=300.0)
    a = parser.parse_args(argv)
    return run(a.host, a.port, a.name, a.seconds)


if __name__ == '__main__':
    try:
        code = main()
    finally:
        sys.stdout.flush()
    os._exit(code)
