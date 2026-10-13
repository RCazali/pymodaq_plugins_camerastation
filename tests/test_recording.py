import json
import math
import time

import numpy as np
import pytest

h5py = pytest.importorskip('h5py')

from pymodaq_plugins_camerastation.core.loader import align_with_laser_control, load_session  # noqa: E402
from pymodaq_plugins_camerastation.core.recorder import (H5StationWriter, IMG_EVENT, IMG_PERIODIC,  # noqa: E402
                                                         reduce_image)
from pymodaq_plugins_camerastation.core.sim import beam_frame  # noqa: E402
from pymodaq_plugins_camerastation.core.triggers import TriggerWatcher  # noqa: E402


def test_reduce_image_8_and_16_bits():
    img = beam_frame(shape=(100, 120), x=60, y=50, noise=0, rng=np.random.default_rng(0))
    q, off, scale = reduce_image(img, roi=(10, 0, 100, 100), binning=2, bits=8)
    assert q.dtype == np.uint8 and q.shape == (50, 50)
    approx = off + q.astype(np.float32) * scale
    ref = img[0:100, 10:110].reshape(50, 2, 50, 2).mean(axis=(1, 3))
    assert np.max(np.abs(approx - ref)) <= scale / 2 + 1e-3
    q16, off16, s16 = reduce_image(img, binning=1, bits=16)
    assert q16.dtype == np.uint16 and off16 == 0.0 and s16 == 1.0 and np.array_equal(q16, img)


def write_session(folder, n=20, counter0=1, **kw):
    w = H5StationWriter(folder, 'session_test', metadata={'machine': 'test'}, flush_s=0.1, **kw)
    w.start()
    for i in range(n):
        vals = {'Cam1/x': float(i), 'Cam1/valid': 1.0, 'Stats/frame_counter': float(counter0 + i),
                'Stats/t_wall': 1000.0 + i}
        if i >= 10:
            vals['Cam2/x'] = 5.0                       # column appearing mid-session
        w.write_row(1000.0 + i, counter0 + i, vals)
        if i % 5 == 0:
            q, off, sc = reduce_image(beam_frame(shape=(64, 80), x=40, y=30, rng=np.random.default_rng(i)))
            assert w.write_image('cam1', 'Cam1', q, off, sc, 1000.0 + i, i, counter0 + i, IMG_PERIODIC,
                                 roi=(0, 0, 80, 64), binning=2)
    w.write_event(1005.0, 'trigger', 'shake', source='LaserControl', event_id=42, data={'k': 1})
    q, off, sc = reduce_image(beam_frame(shape=(64, 64), rng=np.random.default_rng(9)))
    w.write_image('cam1', 'Cam1', q, off, sc, 1006.0, 99, counter0 + 6, IMG_EVENT, event_id=42,
                  roi=(0, 0, 64, 64), binning=2)                     # new shape -> new group
    w.close()
    return w


def test_writer_and_loader(tmp_path):
    w = write_session(tmp_path)
    assert not w.error and w.n_rows == 20 and w.n_images == 5 and w.n_events == 1
    s = load_session(w.paths[0])
    assert s['attrs']['format'] == 'camera_station_session' and s['attrs']['machine'] == 'test'
    assert list(s['frame_counter']) == list(range(1, 21))
    assert np.isnan(s['scalars']['Cam2/x'][:10]).all() and (s['scalars']['Cam2/x'][10:] == 5.0).all()
    assert set(s['images']) == {'cam1', 'cam1~2'}
    assert s['images']['cam1']['n'] == 4 and s['images']['cam1~2']['event_id'][0] == 42
    img = s['image']('cam1', 0)
    assert img.shape == (32, 40) and img.max() > 500
    ev = s['events'][0]
    assert ev['kind'] == 'trigger' and ev['source'] == 'LaserControl' and ev['data'] == {'k': 1}


def test_image_queue_is_bounded_in_bytes(tmp_path):
    w = H5StationWriter(tmp_path, 'bounded', max_queue_mb=1.0)        # writer thread not started
    big = np.zeros((600, 1000), np.uint8)                            # 0.6 MB
    assert w.write_image('c', 'C', big, 0, 1, 0, 0, 0, IMG_PERIODIC)
    assert not w.write_image('c', 'C', big, 0, 1, 0, 0, 0, IMG_PERIODIC)
    assert w.n_dropped_images == 1


def test_disk_error_does_not_kill_the_writer(tmp_path):
    blocker = tmp_path / 'not_a_folder'
    blocker.write_text('x')                                          # mkdir fails: a file is in the way
    w = H5StationWriter(blocker, 's', flush_s=0.05)
    w.start()
    w.write_row(1.0, 1, {'a': 1.0})
    t0 = time.time()
    while not w.error and time.time() - t0 < 5:
        time.sleep(0.02)
    assert w.error and w.running
    w.close()
    assert w.n_dropped_rows == 1


def test_rotation(tmp_path):
    w = H5StationWriter(tmp_path, 'rot', max_mb=0.0001, flush_s=0.05)
    w.start()
    for i in range(3):
        w.write_row(float(i), i, {'a': 1.0})
        time.sleep(0.3)
    w.close()
    assert len(w.paths) >= 2


def test_align_with_laser_control(tmp_path):
    lc = pytest.importorskip('pymodaq_plugins_lasercontrol.recording.h5writer')
    b = write_session(tmp_path / 'b', counter0=100)
    a = lc.H5SessionWriter(tmp_path / 'a', 'lc', flush_s=0.05)
    a.start()
    # column names as written by Laser Control's build_snapshot (recording/snapshot.py)
    for k, c in enumerate((101.0, 105.0, 500.0, math.nan)):
        a.write_snapshot(2000.0 + k, 0, -1, {'raw/LECO_B/Stats/frame_counter': c, 'raw/LECO_B/Stats/t_wall': 1000.0 + k,
                                             'det/LECO_B/heartbeat': c, 'act/OC_H/pos': 1.0}, {})
    a.close()
    a_path = next((tmp_path / 'a').glob('*.h5'))
    m = align_with_laser_control(a_path, b.paths[0])
    assert m['viewer'] == 'LECO_B'
    assert list(m['b_index']) == [1, 5, -1, -1]
    assert m['clock_offset_median'] == pytest.approx(1000.0)
    st = m['station']
    assert st['frame_counter'][m['b_index'][1]] == 105
    assert list(align_with_laser_control(a_path, b.paths[0], viewer='LECO_B')['b_index']) == [1, 5, -1, -1]


def test_align_with_heartbeat_only(tmp_path):
    """Laser Control with 'record raw channels' off: only the declared heartbeat is in the snapshots."""
    lc = pytest.importorskip('pymodaq_plugins_lasercontrol.recording.h5writer')
    b = write_session(tmp_path / 'b', counter0=1)
    a = lc.H5SessionWriter(tmp_path / 'a', 'lc', flush_s=0.05)
    a.start()
    a.write_snapshot(10.0, 0, -1, {'det/LECO_B/heartbeat': 3.0}, {})
    a.close()
    m = align_with_laser_control(next((tmp_path / 'a').glob('*.h5')), b.paths[0])
    assert m['viewer'] == 'LECO_B' and list(m['b_index']) == [2]


def test_trigger_watcher(tmp_path):
    folder = tmp_path / 'triggers'
    w = TriggerWatcher(str(folder))
    assert w.poll() == [] and 'not reachable' in w.error            # share not mounted yet
    folder.mkdir()
    (folder / 'event_old_1_shake.json').write_text(json.dumps({'id': 1, 'kind': 'shake'}))
    assert w.poll() == [] and not w.error                            # old events are not replayed
    (folder / 'event_new_2_shake.json').write_text(json.dumps({'id': 2, 'kind': 'shake', 't': 5.0}))
    (folder / 'event_new_3_x.tmp').write_text('partial')             # being written by A: ignored
    (folder / 'event_new_4_bad.json').write_text('{ bad')
    evs = w.poll()
    assert [e.get('id') for e in evs] == [2, None]
    assert evs[1]['kind'] == 'unreadable' and 'unreadable' in w.error
    assert w.poll() == []                                            # each file once
    assert TriggerWatcher('').poll() == [] and not TriggerWatcher('').enabled
