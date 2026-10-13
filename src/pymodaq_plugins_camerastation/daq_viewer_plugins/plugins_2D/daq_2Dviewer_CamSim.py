"""Simulated camera for the office: a drifting beam, or the disk fluorescence depleted by two passes.

The frame rate is paced by the plugin (like a real camera waiting for its exposure), so the viewer
does not spin at 100 % CPU with a wait time of 0. "Laser on" off gives the disk reference image.
"""
import time

import numpy as np
from qtpy.QtCore import QThread

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.data import DataFromPlugins, DataToExport

from pymodaq_plugins_camerastation.core.sim import beam_frame, disk_frame


class DAQ_2DViewer_CamSim(DAQ_Viewer_base):
    params = comon_parameters + [
        {'title': 'Scene:', 'name': 'scene', 'type': 'list', 'limits': ['beam', 'disk'], 'value': 'beam'},
        {'title': 'Width (px):', 'name': 'width', 'type': 'int', 'value': 640, 'min': 16, 'max': 4096},
        {'title': 'Height (px):', 'name': 'height', 'type': 'int', 'value': 480, 'min': 16, 'max': 4096},
        {'title': 'Frame rate (Hz):', 'name': 'fps', 'type': 'float', 'value': 5.0, 'min': 0.1, 'max': 100.0},
        {'title': 'Noise (counts):', 'name': 'noise', 'type': 'float', 'value': 8.0, 'min': 0.0},
        {'title': 'Laser on:', 'name': 'laser_on', 'type': 'bool', 'value': True},
        {'title': 'Drift amplitude (px):', 'name': 'drift', 'type': 'float', 'value': 15.0, 'min': 0.0},
        {'title': 'Drift period (s):', 'name': 'period', 'type': 'float', 'value': 60.0, 'min': 1.0},
        {'title': 'Pass separation (px):', 'name': 'separation', 'type': 'float', 'value': 30.0, 'min': 0.0},
    ]

    def ini_attributes(self):
        self.controller = None
        self._rng = np.random.default_rng()
        self._t0 = time.monotonic()
        self._next = 0.0

    def ini_detector(self, controller=None):
        self.controller = 'CamSim'
        self._t0 = time.monotonic()
        return 'Simulated camera ready', True

    def commit_settings(self, param):
        pass

    def _frame(self) -> np.ndarray:
        s = self.settings
        shape = (int(s['height']), int(s['width']))
        t = time.monotonic() - self._t0
        ph = 2 * np.pi * t / float(s['period'])
        dx = float(s['drift']) * np.sin(ph)
        dy = 0.5 * float(s['drift']) * np.sin(0.7 * ph)
        cx, cy = shape[1] / 2 + dx, shape[0] / 2 + dy
        if s['scene'] == 'disk':
            half = float(s['separation']) / 2
            r = 0.38 * min(shape)
            return disk_frame(shape, spots=((cx - half, cy), (cx + half, cy)), sigma=0.03 * min(shape),
                              laser_on=bool(s['laser_on']), noise=float(s['noise']), rng=self._rng,
                              cx=shape[1] / 2, cy=shape[0] / 2, radius=r)
        amp = 2000.0 if s['laser_on'] else 0.0
        return beam_frame(shape, x=cx, y=cy, sx=0.04 * shape[1], sy=0.035 * shape[0], amp=amp,
                          noise=float(s['noise']), rng=self._rng)

    def grab_data(self, Naverage=1, **kwargs):
        wait = self._next - time.monotonic()
        if wait > 0:
            QThread.msleep(int(wait * 1000))
        self._next = time.monotonic() + 1.0 / float(self.settings['fps'])
        img = self._frame()
        self.dte_signal.emit(DataToExport('CamSim', data=[
            DataFromPlugins(name='CamSim', data=[img], dim='Data2D', labels=['frame'])]))

    def stop(self):
        return ''

    def close(self):
        pass


if __name__ == '__main__':
    main(__file__)
