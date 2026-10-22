import math
import time

import numpy as np
import pytest

from pymodaq_plugins_camerastation.core.engine import (AnalysisEngine, ST_ERROR, ST_INVALID, ST_NO_FRAMES, ST_OFF,
                                                       ST_OK, ST_WAITING)
from pymodaq_plugins_camerastation.core.publisher import Publisher, STATS, flatten
from pymodaq_plugins_camerastation.core.settings import CameraConfig, SettingsStore, StationConfig
from pymodaq_plugins_camerastation.core.sim import beam_frame

RNG = np.random.default_rng(2)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(rate=2.0, **kw):
    clock = Clock()
    eng = AnalysisEngine(clock=clock, wall=lambda: 1.7e9 + clock.t)
    cam = CameraConfig(id='cam1', name='Cam1', rate_hz=rate, binning=2, **kw).validate()
    eng.set_cameras([cam])
    return eng, clock, cam


def drain(eng):
    """Run the pending work synchronously (no thread in the unit tests)."""
    while eng._pending:
        cam_id, item = eng._pending.popitem(last=False)
        eng.process(eng.cameras[cam_id], item)


def test_rate_gate_drops_extra_frames_without_work():
    eng, clock, cam = make(rate=2.0)
    img = beam_frame(rng=RNG)
    accepted = 0
    for _ in range(50):                    # 10 Hz camera during 5 s
        accepted += eng.submit('cam1', img)
        drain(eng)
        clock.t += 0.1
    assert eng.cameras['cam1'].frames_in == 50
    assert 9 <= accepted <= 11              # ~2 Hz
    assert eng.cameras['cam1'].frames_done == accepted


@pytest.mark.parametrize('fps', [5.0, 8.0, 10.0, 12.4, 15.0, 18.0, 30.0])
@pytest.mark.parametrize('acquired', [True, False])
def test_rate_gate_reaches_the_set_rate(fps, acquired):
    """10 Hz gate, 10 s. A minimum interval would give fps / 2 for a 15 Hz camera (7.5 Hz). With the
    acquisition time, frames delivered in bursts by a busy GUI thread keep every frame of a slower
    camera and the set rate of a faster one; never more than the set rate."""
    eng, clock, cam = make(rate=10.0)
    img = np.zeros((8, 8))
    rng = np.random.default_rng(0)
    n = int(10 * fps)
    t_acq = 1.7e9 + (np.arange(n) + rng.uniform(-0.1, 0.1, n)) / fps
    delivery = np.maximum.accumulate(t_acq + rng.exponential(0.06, n))    # late, sometimes in bursts
    accepted = 0
    for ta, td in zip(t_acq, delivery):
        clock.t = td
        accepted += eng.submit('cam1', img, t_wall=float(ta), acquired=acquired)
        eng._pending.clear()
    expected = min(fps, 10.0) * 10
    assert accepted <= 10 * 10 + 2
    if acquired:
        assert accepted >= 0.97 * expected
    else:
        assert accepted >= 0.75 * expected       # arrival times only: bursts cost frames, bounded


def test_snap_is_analysed_whatever_the_gate():
    eng, clock, cam = make(rate=1.0)
    img = np.zeros((8, 8))
    assert eng.submit('cam1', img)
    assert not eng.submit('cam1', img)      # same instant: gated
    eng.force_next('cam1')
    assert eng.submit('cam1', img)          # snap
    assert not eng.submit('cam1', img)


def test_latest_wins_when_the_worker_is_late():
    eng, clock, cam = make(rate=10.0)
    for i in range(5):
        eng.submit('cam1', beam_frame(x=100 + 10 * i, noise=0, rng=RNG))
        clock.t += 0.2
    assert len(eng._pending) == 1          # never a queue
    assert eng.cameras['cam1'].dropped_pending == 4
    drain(eng)
    assert eng.snapshot('cam1')['result']['x'] == pytest.approx(140, abs=0.5)


def test_states_and_staleness():
    eng, clock, cam = make()
    assert eng.state('cam1', 3.0) == ST_WAITING
    eng.submit('cam1', beam_frame(rng=RNG))
    drain(eng)
    assert eng.state('cam1', 3.0) == ST_OK
    clock.t += 5.0                          # camera silent
    assert eng.state('cam1', 3.0) == ST_NO_FRAMES
    eng.submit('cam1', np.zeros((100, 100), np.uint16))
    drain(eng)
    assert eng.state('cam1', 3.0) == ST_INVALID
    cam.enabled = False
    assert eng.state('cam1', 3.0) == ST_OFF


def test_exception_is_isolated_per_camera():
    eng, clock, cam = make()
    errors = []
    eng.on_error = lambda cid, msg: errors.append((cid, msg))
    eng.submit('cam1', np.zeros((2, 3, 4, 5)))          # 4D frame: the analysis raises
    drain(eng)
    snap = eng.snapshot('cam1')
    assert snap['error'] and snap['result']['valid'] == 0.0 and math.isnan(snap['result']['x'])
    assert eng.state('cam1', 3.0) == ST_ERROR
    assert errors and errors[0][0] == 'cam1'
    clock.t += 1.0
    eng.submit('cam1', beam_frame(rng=RNG))             # recovers by itself on the next good frame
    drain(eng)
    assert eng.state('cam1', 3.0) == ST_OK


def test_disabled_camera_still_keeps_latest_frame_but_no_analysis():
    eng, clock, cam = make(enabled=False)
    assert not eng.submit('cam1', beam_frame(rng=RNG))
    assert eng.cameras['cam1'].latest is not None and eng.cameras['cam1'].frames_done == 0


def test_capture_averages_n_frames():
    eng, clock, cam = make()
    got = {}
    eng.start_capture('cam1', 'ref', 4, lambda cid, img: got.setdefault(cid, img))
    for v in (10, 20, 30, 40, 50):
        eng.submit('cam1', np.full((10, 10), v, np.uint16))
        clock.t += 1
    assert got['cam1'].shape == (10, 10) and got['cam1'][0, 0] == pytest.approx(25.0)


def test_worker_thread_end_to_end():
    eng = AnalysisEngine()
    eng.set_cameras([CameraConfig(id='a', name='A', rate_hz=30).validate(),
                     CameraConfig(id='b', name='B', rate_hz=30).validate()])
    eng.start()
    try:
        for _ in range(5):
            eng.submit('a', beam_frame(x=100, rng=RNG))
            eng.submit('b', beam_frame(x=200, rng=RNG))
            time.sleep(0.05)
        t0 = time.time()
        while time.time() - t0 < 5 and (eng.snapshot('a')['frames_done'] == 0 or eng.snapshot('b')['frames_done'] == 0):
            time.sleep(0.02)
        assert eng.snapshot('a')['result']['x'] == pytest.approx(100, abs=1)
        assert eng.snapshot('b')['result']['x'] == pytest.approx(200, abs=1)
    finally:
        eng.stop()
    assert not eng.running


def test_publisher_payload_heartbeat_and_stale_nan():
    eng, clock, cam = make()
    eng.submit('cam1', beam_frame(rng=RNG))
    drain(eng)
    pub = Publisher()
    p1 = pub.build([('Cam1', cam.labels(), eng.snapshot('cam1'))], t_wall=123.0, stale_s=3.0)
    assert p1['Cam1']['valid'] == 1.0 and math.isfinite(p1['Cam1']['x'])
    assert p1[STATS] == {'frame_counter': 1.0, 't_wall': 123.0, 'cams_ok': 1.0, 'recording': 0.0}
    clock.t += 10
    p2 = pub.build([('Cam1', cam.labels(), eng.snapshot('cam1'))], t_wall=133.0, stale_s=3.0)
    assert p2[STATS]['frame_counter'] == 2.0
    assert p2['Cam1']['valid'] == 0.0 and math.isnan(p2['Cam1']['x'])   # frozen camera: never a stale value
    assert p2['Cam1']['age'] == pytest.approx(10.0)
    assert 'Cam1/x' in flatten(p2) and 'Stats/frame_counter' in flatten(p2)


def test_channel_keys_seen_by_laser_control():
    """The keys that Laser Control (computer A) will build from our DataToExport."""
    pytest.importorskip('pymodaq_data')
    lc = pytest.importorskip('pymodaq_plugins_lasercontrol.hardware.channels')
    from pymodaq_plugins_camerastation.relay import payload_to_dte
    payload = {'Cam1': {'x': 1.0, 'y': 2.0, 'valid': 1.0}, STATS: {'frame_counter': 7.0, 't_wall': 1.0}}
    dte = payload_to_dte(payload)
    for dwa in dte:
        dwa.origin = 'LECO_B'              # what the DAQ_Viewer of computer A does
    keys = lc.extract_scalars(dte, 'LECO_B')
    assert keys['LECO_B/Cam1/x'] == 1.0
    assert keys['LECO_B/Stats/frame_counter'] == 7.0


def test_groups_too_wide_for_a_5_1_viewer_are_not_plotted_whatever_the_local_version():
    """A station on PyMoDAQ 5.3 feeds a viewer of computer A that may run 5.1 (6-7 pens, IndexError beyond)."""
    pytest.importorskip('pymodaq_data')
    from pymodaq_plugins_camerastation.relay import MAX_PLOTTED_CHANNELS, payload_to_dte
    assert MAX_PLOTTED_CHANNELS is not None and MAX_PLOTTED_CHANNELS <= 6
    dte = payload_to_dte({'Cam1': {k: 1.0 for k in 'abcdefghij'}, STATS: {'frame_counter': 1.0, 't_wall': 0.0}})
    plot = {dwa.name: getattr(dwa, 'do_plot', True) for dwa in dte}
    assert plot == {'Cam1': False, STATS: True}


def test_settings_roundtrip_validation_and_corrupt_file(tmp_path):
    st = StationConfig(publish_hz=1000, cameras=[CameraConfig(id='c', name='Disk cam/1', roi=[1, 2]),
                                                  CameraConfig(id='c', name='Stats', kind='nope')])
    st.validate()
    assert st.publish_hz == 20.0
    assert [c.id for c in st.cameras] == ['c', 'c_2']
    assert st.cameras[0].name == 'Disk_cam_1' and st.cameras[1].name == 'Stats_2'   # 'Stats' is reserved
    assert st.cameras[0].roi == [1, 2, 0, 0] and st.cameras[1].kind == 'beam'
    store = SettingsStore(tmp_path / 'station.json')
    store.save(st)
    store.save(st)
    assert (tmp_path / 'station.json.bak').exists()
    back = store.load()
    assert back.to_dict() == st.to_dict() and not store.load_error
    (tmp_path / 'station.json').write_text('{ broken', encoding='utf-8')
    d = store.load()
    assert d.cameras == [] and 'unreadable' in store.load_error
    assert list(tmp_path.glob('station.json.corrupt-*'))
