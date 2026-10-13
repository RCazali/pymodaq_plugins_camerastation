"""0D relay of the Camera Station: what computer A receives through LECO.

No hardware. The Camera Station extension publishes its 0D values at a fixed rate (Station > Publish
rate); this plugin emits each publication **once**, while the viewer grabs:

* ``live_mode_available = True``: PyMoDAQ calls ``grab_data`` once and the plugin emits on its own when
  a publication arrives. (The former BeamAnalyzer relay re-emitted its cached value each time PyMoDAQ
  asked again, i.e. continuously: CPU for nothing, and a frozen extension looked alive to computer A.)
* a *snap* returns the last publication, if any, once;
* the heartbeat ``Stats/frame_counter`` only advances when the extension really publishes.

Use: add a DAQ_Viewer 0D "CameraStation" to the dashboard of the camera computer, connect it to LECO
(as for any LECO viewer), and open the Camera Station extension.
"""
import math

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq_utils.utils import ThreadCommand

from pymodaq_plugins_camerastation.relay import bus, payload_to_dte


class DAQ_0DViewer_CameraStation(DAQ_Viewer_base):
    live_mode_available = True
    params = comon_parameters + [
        {'title': 'Publications received:', 'name': 'n_received', 'type': 'int', 'value': 0, 'readonly': True},
    ]

    def ini_attributes(self):
        self.controller = None
        self._live = False
        self._n = 0

    def ini_detector(self, controller=None):
        self.controller = 'Camera Station relay (no hardware)'
        bus.data_ready.connect(self._on_data_ready)
        return 'Camera Station relay ready: open the Camera Station extension to publish', True

    def _on_data_ready(self, dte):
        self._n += 1
        if self._live:
            self.dte_signal.emit(dte)
        if self._n % 20 == 1:          # cheap: the settings tree is not updated at every publication
            self.emit_status(ThreadCommand('update_settings', [['n_received'], self._n, 'value']))

    def commit_settings(self, param):
        pass

    def grab_data(self, Naverage=1, **kwargs):
        if kwargs.get('live', False):
            self._live = True              # the next publications are emitted as they arrive
        elif bus.last is not None:
            self.dte_signal.emit(bus.last)
        else:
            # a snap must always answer (PyMoDAQ waits for it); NaN = no valid value, never a fake 0
            self.dte_signal.emit(payload_to_dte({'Stats': {'frame_counter': math.nan}}))
            self.emit_status(ThreadCommand('Update_Status', ['Camera Station: nothing published yet']))

    def stop(self):
        self._live = False
        return ''

    def close(self):
        self._live = False
        try:
            bus.data_ready.disconnect(self._on_data_ready)
        except (TypeError, RuntimeError):
            pass


if __name__ == '__main__':
    main(__file__, init=False)
