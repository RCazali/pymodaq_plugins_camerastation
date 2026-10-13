"""The station with real PyMoDAQ viewers (CamSim 2D cameras + CameraStation 0D relay), headless.

Checks the whole chain: frames -> engine -> publication -> relay viewer (what LECO sends to computer A),
reference capture of the disk camera, recording with a trigger event of Laser Control, shutdown."""
import json
import math
import os
import time
import types

import pytest

pytest.importorskip('pymodaq')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy import QtWidgets  # noqa: E402

pytestmark = pytest.mark.pymodaq
_app = None


def app():
    global _app
    _app = QtWidgets.QApplication.instance()
    if _app is None:
        try:
            from pymodaq_gui.qt_utils import mkQApp
        except ImportError:
            from pymodaq_gui.utils.utils import mkQApp
        _app = mkQApp('pytest')
    return _app


def wait(pred, timeout=10.0):
    a = app()
    t0 = time.time()
    while time.time() - t0 < timeout:
        a.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    a.processEvents()
    return pred()


def make_viewer(title, daq_type, plugin, **settings):
    from pymodaq.control_modules.daq_viewer import DAQ_Viewer
    det = DAQ_Viewer(None, title=title)
    det.daq_type = daq_type
    det.detector = plugin
    inits = []
    det.init_signal.connect(lambda s: inits.append(s))
    det.init_hardware(True)
    assert wait(lambda: inits, 20) and inits[-1], f'{title} not initialized'
    for k, v in settings.items():
        det.settings.child('detector_settings', k).setValue(v)
    wait(lambda: False, 0.2)
    return det


@pytest.fixture(scope='module')
def station(tmp_path_factory):
    app()
    from pymodaq_gui.utils.dock import DockArea
    from pymodaq_plugins_camerastation.ui.station import CameraStationApp
    from pymodaq_plugins_camerastation.utils import Config
    data = tmp_path_factory.mktemp('station')
    disk = make_viewer('Disk cam', 'DAQ2D', 'CamSim', scene='disk', fps=10.0, drift=0.0)
    out = make_viewer('Output cam', 'DAQ2D', 'CamSim', scene='beam', fps=10.0, drift=0.0)
    relay = make_viewer('CameraStation', 'DAQ0D', 'CameraStation')
    mm = types.SimpleNamespace(detectors=[disk, out, relay])
    win = QtWidgets.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    st = CameraStationApp(mm, Config(), window=None, data_folder=data)
    st.build(area)
    yield st, disk, out, relay, data
    st.shutdown()
    for d in (disk, out, relay):
        try:
            d.stop_grab()
            d.quit_fun()
        except Exception:
            pass
    wait(lambda: False, 0.5)


def test_relay_excluded_from_cameras(station):
    st, *_ = station
    assert st.available_detectors() == ['Disk cam', 'Output cam']


def test_frames_are_analysed_and_published_to_the_relay(station):
    st, disk, out, relay, data = station
    c_disk = st.add_camera()
    c_out = st.add_camera()
    assert (c_disk.detector, c_out.detector) == ('Disk cam', 'Output cam')
    st.params.child('cameras', c_disk.id, 'kind').setValue('disk')
    st.params.child('cameras', c_out.id, 'name').setValue('Output')
    received = []
    relay.grab_done_signal.connect(lambda dte: received.append(dte))
    st.grab_all(True)
    assert wait(lambda: st.engine.snapshot(c_out.id).get('result', {}).get('valid') == 1.0, 20)
    r = st.engine.snapshot(c_out.id)['result']
    assert r['x'] == pytest.approx(320, abs=3) and r['y'] == pytest.approx(240, abs=3)
    assert wait(lambda: len(received) >= 3, 10)
    dte = received[-1]
    names = {dwa.name: list(dwa.labels) for dwa in dte}
    assert 'Output' in names and 'x' in names['Output'] and 'Stats' in names
    assert 'frame_counter' in names['Stats']
    counters = [float(next(d for d in x if d.name == 'Stats').data[names['Stats'].index('frame_counter')][0])
                for x in received[-3:]]
    assert counters == sorted(counters) and len(set(counters)) == 3     # each publication once, heartbeat advances
    # rate gate: the camera runs at 10 Hz, the analysis at 2 Hz
    snap = st.engine.snapshot(c_out.id)
    assert snap['frames_in'] > snap['frames_done']
    # the disk camera has no reference yet: published, but invalid (NaN), with the reason
    assert st.engine.snapshot(c_disk.id)['reason'] == 'no reference image'


def test_disk_reference_capture(station):
    st, disk, out, relay, data = station
    cam = next(c for c in st.station.cameras if c.kind == 'disk')
    cam.mask_cx, cam.mask_cy, cam.mask_radius = 320, 240, 175
    disk.settings.child('detector_settings', 'laser_on').setValue(False)
    wait(lambda: False, 0.5)
    st.params.child('cameras', cam.id, 'disk', 'n_capture').setValue(3)
    st.capture(cam.id, 'ref')
    assert wait(lambda: bool(cam.reference_file), 10)
    assert (st.paths['references'] / cam.reference_file).exists()
    disk.settings.child('detector_settings', 'laser_on').setValue(True)
    assert wait(lambda: st.engine.snapshot(cam.id)['result'].get('valid') == 1.0, 15)
    r = st.engine.snapshot(cam.id)['result']
    assert r['x'] == pytest.approx(320, abs=4) and r['sx'] > r['sy']


def test_recording_with_trigger_event(station, tmp_path):
    st, disk, out, relay, data = station
    from pymodaq_plugins_camerastation.core.loader import load_session
    trig = tmp_path / 'trig'
    trig.mkdir()
    st.params.child('station', 'trigger_folder').setValue(str(trig))
    st.params.child('station', 'recording', 'burst_frames').setValue(3)
    wait(lambda: False, 2.0)                                   # first listing of the folder
    assert st.start_recording()
    wait(lambda: False, 1.5)
    (trig / 'event_20261010_120000_000_7_shake.json').write_text(json.dumps({'id': 7, 'kind': 'shake',
                                                                             'message': 'shake burst'}))
    assert wait(lambda: st.watcher.n_events >= 1, 10)
    wait(lambda: False, 2.5)                                   # burst at the analysis rate
    st.add_note('test note')
    path = st.writer.current_path
    st.stop_recording()
    s = load_session(path)
    assert len(s['t']) >= 3 and 'Output/x' in s['scalars'] and 'Stats/frame_counter' in s['scalars']
    assert list(s['frame_counter']) == list(s['scalars']['Stats/frame_counter'].astype(int))
    kinds = [e['kind'] for e in s['events']]
    assert 'trigger' in kinds and 'note' in kinds and 'recording' in kinds
    groups = s['images']
    ev_imgs = sum(int((g['event_id'] == 7).sum()) for g in groups.values())
    assert ev_imgs >= 2 * 2                                    # both cameras, event image + burst
    periodic = sum(int((g['kind'] == 0).sum()) for g in groups.values())
    assert periodic >= 2                                       # first periodic images at the start
    img = s['image'](next(iter(groups)), 0)
    assert img.ndim == 2 and math.isfinite(float(img.mean()))


def test_stale_camera_publishes_nan(station):
    st, disk, out, relay, data = station
    st.params.child('station', 'stale_s').setValue(1.0)
    out.stop_grab()
    cam = next(c for c in st.station.cameras if c.detector == 'Output cam')
    assert wait(lambda: st.engine.state(cam.id, 1.0) == 'no_frames', 10)
    payload = st.publisher.build([(cam.name, cam.labels(), st.engine.snapshot(cam.id))], time.time(), 1.0)
    assert payload[cam.name]['valid'] == 0.0 and math.isnan(payload[cam.name]['x'])
    st.grab_all(True)                       # restarts only what is stopped (out.grab() needs a UI)
    assert wait(lambda: st.engine.state(cam.id, 1.0) == 'ok', 10)
    st.params.child('station', 'stale_s').setValue(3.0)


LIVE = '''
LABELS = ("mean",)
PARAMS = {"gain": 1.0}
def analyze(frame, ctx):
    w = ctx.crop(frame)
    ctx.show(w)
    return {"mean": float(w.mean()) * ctx.params["gain"]}
'''


def test_scripts_live_edit_publish_and_preview(station):
    st, disk, out, relay, data = station
    folder = st.paths['scripts']
    assert (folder / '_template.py').exists() and 'spot_shape' in st.library.names()   # examples installed
    cam = next(c for c in st.station.cameras if c.detector == 'Output cam')
    # tick the example on the camera from the settings tree
    st.params.child('cameras', cam.id, 'scripts', 'spot_shape').setValue(True)
    assert cam.scripts == ['spot_shape']
    assert wait(lambda: st.runner.snapshot(cam.id, 'spot_shape').get('result', {}).get('valid') == 1.0, 15)
    # a new script dropped in the folder, then edited, without restarting anything
    (folder / 'live.py').write_text(LIVE, encoding='utf-8')
    st._scan_scripts()
    st.params.child('cameras', cam.id, 'scripts', 'live').setValue(True)
    assert wait(lambda: st.runner.snapshot(cam.id, 'live').get('result', {}).get('valid') == 1.0, 15)
    m1 = st.runner.snapshot(cam.id, 'live')['result']['mean']
    st.params.child('scripts', 'script__live', 'params', 'gain').setValue(2.0)      # live parameter
    assert wait(lambda: st.runner.snapshot(cam.id, 'live')['result']['mean'] > 1.5 * m1, 10)
    received = []
    relay.grab_done_signal.connect(lambda dte: received.append({d.name for d in dte}))
    wait(lambda: len(received) >= 2, 5)
    pub = f'{cam.name}_live'
    assert pub not in received[-1]                                       # testing: not sent to A
    st.params.child('scripts', 'script__live', 'publish').setValue(True)
    assert wait(lambda: received and pub in received[-1], 5)             # now sent to A
    # broken new version: the previous one keeps running
    time.sleep(1.1)
    (folder / 'live.py').write_text(LIVE + '\n  oops(', encoding='utf-8')
    st._scan_scripts()
    assert 'still running' in st.library.scripts['live'].error
    runs = st.runner.snapshot(cam.id, 'live')['runs']
    assert wait(lambda: st.runner.snapshot(cam.id, 'live')['runs'] > runs, 10)
    # preview of the script output
    st.station.preview = True
    st.params.child('station', 'preview_group', 'preview_camera').setValue(cam.id)
    st.params.child('station', 'preview_group', 'preview_view').setValue('script output')
    st.docks['preview'].window().show()                 # the preview only refreshes a visible dock
    st.docks['preview'].raiseDock()
    wait(lambda: st.docks['preview'].isVisible(), 5)
    st._refresh_preview()
    assert 'script' in st.preview.info.text()
    st._refresh_ui()
    assert st.scripts_table.rowCount() == 2
    st.params.child('scripts', 'script__live', 'publish').setValue(False)


def test_settings_persisted(station):
    st, disk, out, relay, data = station
    st._save_settings()
    from pymodaq_plugins_camerastation.core.settings import SettingsStore
    back = SettingsStore(st.store.path).load()
    assert [c.name for c in back.cameras] == [c.name for c in st.station.cameras]
    assert any(c.reference_file for c in back.cameras)
