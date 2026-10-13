"""The real PyMoDAQ dashboard loads the extension through the ``pymodaq.extensions`` entry point,
with the office simulator configuration (tools/sim_preset.py). The preset / experiment file written in
the PyMoDAQ folder is removed at the end; the station data go to a temporary folder."""
import os
import time

import pytest

pytest.importorskip('pymodaq')

from test_station_pymodaq import app, wait  # noqa: E402
from qtpy import QtWidgets  # noqa: E402

pytestmark = pytest.mark.pymodaq
NAME = 'cs_pytest_dashboard'


@pytest.fixture(scope='module')
def loaded(tmp_path_factory):
    import pymodaq_plugins_camerastation.ui.station as station_mod
    from pymodaq.dashboard import DashBoard
    from pymodaq_plugins_camerastation.tools import sim_preset
    app()
    data = tmp_path_factory.mktemp('cs_dash')
    cwd = os.getcwd()
    os.chdir(data)
    orig_resolve = station_mod.resolve_paths
    station_mod.resolve_paths = lambda config, override=None: orig_resolve(config, data)
    orig_check = getattr(DashBoard, 'check_update', None)
    DashBoard.check_update = lambda self, show=True: False          # no network in tests
    path = sim_preset.write_sim_config(NAME, overwrite=True)
    sim_preset.write_sim_station(data)
    try:
        if sim_preset.is_pymodaq_52():
            from pymodaq.dashboard import load_dashboard_with_experiment
            from pymodaq_plugins_camerastation.extensions.camera_station import CLASS_NAME
            dashboard, extension, win = load_dashboard_with_experiment(NAME, CLASS_NAME.upper())
        else:
            from pymodaq_gui.utils.dock import DockArea
            win = QtWidgets.QMainWindow()
            area = DockArea()
            win.setCentralWidget(area)
            dashboard = DashBoard(area)
            dashboard.set_preset_mode(path)
            extension = dashboard.load_extension_from_name('Camera Station')
        # PyMoDAQ 5.3 creates the modules while the event loop runs: wait for the station to see them
        wait(lambda: len(extension.app._links) == 2 and len(extension.app.relay_viewers()) == 1, 60)
        wait(lambda: False, 1.0)
        yield dashboard, extension
        try:
            extension.app.grab_all(False)
            wait(lambda: False, 0.5)
            dashboard.quit_fun()
            wait(lambda: False, 1.0)
        except Exception:
            pass
    finally:
        os.chdir(cwd)
        station_mod.resolve_paths = orig_resolve
        if orig_check is not None:
            DashBoard.check_update = orig_check
        for p in path.parent.glob(f'{NAME}*'):
            try:
                p.unlink()
            except OSError:
                pass


def test_extension_is_discovered():
    from pymodaq.extensions.utils import get_extensions
    extensions = get_extensions()
    if isinstance(extensions, dict):          # PyMoDAQ 5.3: {ExtensionEnum: Extension(name, class_name, klass)}
        names = [e.name for e in extensions.values()]
    else:                                     # 5.1 / 5.2: list of dicts
        names = [e['name'] for e in extensions]
    assert 'Camera Station' in names


def test_dashboard_runs_the_station(loaded):
    dashboard, ext = loaded
    from pymodaq_plugins_camerastation.extensions.camera_station import CameraStation
    assert isinstance(ext, CameraStation)
    st = ext.app
    assert [c.detector for c in st.station.cameras] == ['Disk cam', 'Output cam']
    assert sorted(st._links) == ['disk', 'out']
    assert [d.title for d in st.relay_viewers()] == ['CameraStation']
    for action in ('record', 'note', 'snapshot', 'grab', 'refresh_hardware'):
        assert ext.has_action(action)
    st.grab_all(True)
    assert wait(lambda: st.engine.snapshot('out')['result'].get('valid') == 1.0, 30)
    assert wait(lambda: st.publisher.counter > 3, 10)
    relay = st.relay_viewers()[0]
    assert relay.grab_state
    t0 = time.time()
    st.toggle_recording(True)
    assert wait(lambda: st.writer is not None and st.writer.n_rows > 0, 10)
    st.toggle_recording(False)
    assert time.time() - t0 < 15
