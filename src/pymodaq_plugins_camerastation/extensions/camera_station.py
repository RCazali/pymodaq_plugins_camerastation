# -*- coding: utf-8 -*-
"""
Camera Station - PyMoDAQ dashboard extension of the camera computer (computer B).

Analyses the cameras of the dashboard (beam centroid / size, disk depletion, ROI intensity) at a low,
bounded rate, publishes the results as 0D values to computer A (Laser Control) through the
"CameraStation" 0D viewer connected to LECO, with a heartbeat, and records a light HDF5 log
(0D values + reduced images + events) synchronised with the Laser Control recordings.

Works with PyMoDAQ 5.1.x (developed on 5.1.10) and 5.2.x.
"""
from pathlib import Path

from qtpy import QtCore, QtWidgets

from pymodaq_gui import utils as gutils
from pymodaq_utils.logger import set_logger, get_module_name

from pymodaq.extensions.utils import CustomExt

from pymodaq_plugins_camerastation.utils import Config as PluginConfig
from pymodaq_plugins_camerastation.ui.icons import shape_icon as _icon, std_icon as _std_icon
from pymodaq_plugins_camerastation.ui.station import CameraStationApp

logger = set_logger(get_module_name(__file__))
plugin_config = PluginConfig()

EXTENSION_NAME = 'Camera Station'
CLASS_NAME = 'CameraStation'


class CameraStation(CustomExt):
    params = []
    app = None          # class level: CustomExt.__init__ may call hooks before the app exists (5.2)

    def __init__(self, parent: gutils.DockArea, dashboard, data_folder: Path = None):
        super().__init__(parent, dashboard)
        self.app = CameraStationApp(self.modules_manager, plugin_config, window=self.mainwindow,
                                    data_folder=data_folder)
        self.setup_ui()
        self._title = f'Camera Station — machine {self.app.machine}'
        if self.mainwindow is not None:
            self.mainwindow.installEventFilter(self)
            self._decorate_window()
            QtCore.QTimer.singleShot(0, self._decorate_window)    # PyMoDAQ 5.2 retitles the window afterwards

    def _decorate_window(self):
        if self.mainwindow is not None:
            self.mainwindow.setWindowTitle(self._title)

    # ---------------------------------------------------------------- CustomApp API (5.1 and 5.2)
    def setup_docks_and_widgets(self):
        self.app.build(self.dockarea)

    def setup_docks(self):        # pymodaq_gui 5.0.x calls setup_docks from setup_ui
        self.setup_docks_and_widgets()

    def setup_menus_and_toolbars(self, menubar=None):
        # PyMoDAQ 5.3: only the "Show dashboard" button (the dashboard checks it when it opens the
        # extension). Not the experiment / state managers: they act on the dashboard, not on the station,
        # which has its own settings profiles (Settings > Profiles).
        create = getattr(self, 'create_dashboard_toolbar', None)
        if create is not None and getattr(self.dashboard, 'experiment_manager', None) is not None:
            try:
                create(add_dashboard=True, add_experiment=False, add_state=False, add_break=False)
            except Exception as e:  # pragma: no cover - never block the extension for a toolbar
                logger.warning(f'dashboard toolbar not created: {e}')

    def setup_menu(self, menubar=None):
        pass

    def setup_actions(self):
        self.add_action('record', 'Recording', _icon('#e74c3c'), 'Start / stop the HDF5 recording', checkable=True)
        self.add_action('note', 'Note', _std_icon('SP_FileDialogDetailedView'), 'Write a note in the recording')
        self.add_action('snapshot', 'Snapshot', _std_icon('SP_DialogSaveButton'),
                        'Record the current image of every camera')
        self.add_action('grab', 'Grab all', _icon('#2ecc71', 'triangle'),
                        'Grab all the cameras of the station and the CameraStation relay '
                        '(one camera: ▶ in its panel)')
        self.add_action('stop_grab', 'Stop all', _icon('#95a5a6', 'square'), 'Stop all the cameras and the relay')
        self.add_action('refresh_hardware', 'Reload modules', _std_icon('SP_BrowserReload'),
                        'Re-attach the dashboard viewers (after adding / changing modules)')

    def connect_things(self):
        self.connect_action('record', lambda: self.app.toggle_recording(self.is_action_checked('record')))
        self.connect_action('note', self._ask_note)
        self.connect_action('snapshot', lambda: self.app.snapshot())
        self.connect_action('grab', lambda: self.app.grab_all(True))
        self.connect_action('stop_grab', lambda: self.app.grab_all(False))
        self.connect_action('refresh_hardware', self.app.refresh_hardware)
        self.app.ui_timer.timeout.connect(self._sync_actions)

    def _ask_note(self):
        text, ok = QtWidgets.QInputDialog.getText(self.mainwindow, 'Note', 'Note for the recording:')
        if ok:
            self.app.add_note(text.strip())

    def _sync_actions(self):
        try:
            act = self.get_action('record')
            if act.isChecked() != self.app.recording:
                act.blockSignals(True)
                act.setChecked(self.app.recording)
                act.blockSignals(False)
        except Exception:
            pass

    # ---------------------------------------------------------------- PyMoDAQ hooks
    def stop(self):
        pass

    def do_things_after_experiment_set(self, experiment_name: str):
        parent = getattr(super(), 'do_things_after_experiment_set', None)
        if parent is not None:
            parent(experiment_name)
        if getattr(self, 'app', None) is not None:
            self.app.refresh_hardware()

    def eventFilter(self, obj, event):
        if obj is self.mainwindow and event.type() == QtCore.QEvent.Close and self.app is not None \
                and not self.app._closing:
            if self.app.recording:
                answer = QtWidgets.QMessageBox.question(
                    self.mainwindow, 'Close Camera Station',
                    'The recording is running and will be stopped. Close?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
                if answer != QtWidgets.QMessageBox.Yes:
                    event.ignore()
                    return True
            self.app.shutdown()
        return super().eventFilter(obj, event)

    def _quit_fun(self) -> bool:
        return True       # PyMoDAQ 5.3: CustomExt.quit_fun does nothing unless this hook returns True

    def quit_fun(self) -> bool:
        if self.app is not None:
            self.app.shutdown()
        try:
            super().quit_fun()
        except Exception:
            pass
        if self.mainwindow is not None:
            self.mainwindow.close()
        return True       # 5.3: the "Quit" button of the window closes it only if this is True


def main():
    """Standalone launch: ``python -m pymodaq_plugins_camerastation.extensions.camera_station <preset>``."""
    import sys
    try:
        from pymodaq_gui.qt_utils import mkQApp          # pymodaq_gui 5.2
    except ImportError:
        from pymodaq_gui.utils.utils import mkQApp       # pymodaq_gui 5.0 / 5.1
    app = mkQApp('Camera Station')
    name = sys.argv[1] if len(sys.argv) > 1 else 'camera_sim'
    try:  # PyMoDAQ 5.2: extensions are designated by the upper-cased class name (ExtensionEnum)
        from pymodaq.dashboard import load_dashboard_with_experiment
        dashboard, extension, win = load_dashboard_with_experiment(name, CLASS_NAME.upper())
    except ImportError:  # PyMoDAQ 5.1
        from pymodaq.utils.gui_utils.loader_utils import load_dashboard_with_preset
        dashboard, extension, win = load_dashboard_with_preset(name, EXTENSION_NAME)
    fn = getattr(app, 'exec', None) or app.exec_
    sys.exit(fn())


if __name__ == '__main__':
    main()
