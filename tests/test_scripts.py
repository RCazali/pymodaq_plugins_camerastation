import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from pymodaq_plugins_camerastation.core.analysis import CameraAnalyzer
from pymodaq_plugins_camerastation.core.scripts import (ScriptLibrary, ScriptRunner, load_script, run_offline,
                                                        run_on_session)
from pymodaq_plugins_camerastation.core.settings import CameraConfig
from pymodaq_plugins_camerastation.core.sim import beam_frame, disk_frame

RNG = np.random.default_rng(5)
EXAMPLES = Path(__file__).resolve().parents[1] / 'src' / 'pymodaq_plugins_camerastation' / 'resources' / 'scripts'

GOOD = '''
LABELS = ("mean", "p")
PARAMS = {"gain": 2.0}
def analyze(frame, ctx):
    ctx.state["n"] = ctx.state.get("n", 0) + 1
    return {"mean": float(frame.mean()) * ctx.params["gain"], "p": ctx.state["n"]}
'''


def write(folder, name, text):
    p = folder / f'{name}.py'
    p.write_text(text, encoding='utf-8')
    t = time.time() + getattr(write, 'bump', 0)          # distinct mtimes even on coarse file systems
    write.bump = getattr(write, 'bump', 0) + 2
    os.utime(p, (t, t))
    return p


def test_library_load_reload_and_broken_new_version(tmp_path):
    lib = ScriptLibrary(tmp_path)
    write(tmp_path, 'good', GOOD)
    write(tmp_path, '_ignored', 'this is not python')
    msgs = lib.scan()
    assert lib.names() == ['good'] and lib.scripts['good'].ok and lib.scripts['good'].version == 1
    assert msgs[0][:2] == ('loaded', 'good')
    assert lib.scan() == []                                         # unchanged: nothing reloaded
    assert lib.scan(force=True)[0][:2] == ('loaded', 'good') and lib.scripts['good'].version == 2
    write(tmp_path, 'good', GOOD + '\n  oops(')                     # syntax error in the new version
    lvl, _, msg = lib.scan()[0]
    s = lib.scripts['good']
    assert lvl == 'warning' and s.ok and s.version == 2 and 'still running' in s.error and 'line' in s.error
    write(tmp_path, 'good', GOOD.replace('2.0', '3.0'))
    lib.scan()
    assert lib.scripts['good'].version == 3 and not lib.scripts['good'].error
    assert lib.scripts['good'].defaults == {'gain': 3.0}
    write(tmp_path, 'nolabels', 'def analyze(f, c): return {}')
    lib.scan()
    assert not lib.scripts['nolabels'].ok and 'LABELS' in lib.scripts['nolabels'].error
    (tmp_path / 'nolabels.py').unlink()
    lib.scan()
    assert 'nolabels' not in lib.names()


def make_runner(tmp_path, text, name='s', **cam_kw):
    lib = ScriptLibrary(tmp_path)
    write(tmp_path, name, text)
    lib.scan()
    runner = ScriptRunner(lib)
    cam = CameraConfig(id='c1', name='Cam1', scripts=[name], **cam_kw).validate()
    return lib, runner, cam, CameraAnalyzer(cam)


def run_once(runner, cam, an, img, frame_no=1):
    runner.submit(cam, an, img, frame_no, 0.0, {'x': 1.0})
    _, item = runner._pending.popitem(last=False)
    runner._process(runner._gen, item)


def test_runner_params_state_and_errors(tmp_path):
    lib, runner, cam, an = make_runner(tmp_path, GOOD)
    img = np.full((10, 10), 5, np.uint16)
    run_once(runner, cam, an, img)
    r = runner.snapshot('c1', 's')
    assert r['result'] == {'mean': 10.0, 'p': 1.0, 'valid': 1.0} and not r['error']
    runner.params['s'] = {'gain': 1.0}                               # live parameter change
    run_once(runner, cam, an, img, 2)
    assert runner.snapshot('c1', 's')['result'] == {'mean': 5.0, 'p': 2.0, 'valid': 1.0}
    runner.script_reloaded('s')                                      # reload: fresh state
    run_once(runner, cam, an, img, 3)
    assert runner.snapshot('c1', 's')['result']['p'] == 1.0
    write(tmp_path, 's', 'LABELS=("a",)\ndef analyze(f, c):\n    return {"a": 1/0}\n')
    lib.scan()
    run_once(runner, cam, an, img, 4)
    r = runner.snapshot('c1', 's')
    assert 'ZeroDivisionError' in r['error'] and r['result']['valid'] == 0.0 and math.isnan(r['result']['a'])


def test_frame_is_read_only_for_scripts(tmp_path):
    lib, runner, cam, an = make_runner(tmp_path, 'LABELS=("a",)\ndef analyze(f, c):\n    f[0, 0] = 9\n    return {"a": 1}\n')
    img = np.zeros((4, 4), np.uint16)
    run_once(runner, cam, an, img)
    assert 'read-only' in runner.snapshot('c1', 's')['error'] and img[0, 0] == 0


def test_watchdog_disables_a_stuck_script(tmp_path):
    lib, runner, cam, an = make_runner(tmp_path, 'import time\nLABELS=("a",)\ndef analyze(f, c):\n'
                                                 '    time.sleep(c.params.get("t", 3))\n    return {"a": 1}\n')
    runner.start()
    try:
        runner.submit(cam, an, np.zeros((4, 4)), 1, 0.0, {})
        time.sleep(0.5)
        msg = runner.watchdog(0.2)
        assert msg and 'disabled' in msg and 's' in runner.disabled
        assert runner._thread is not None and runner._thread.is_alive()        # a new worker took over
        runner.script_reloaded('s')
        assert 's' not in runner.disabled
    finally:
        runner.stop(0.1)


def test_example_scripts_load_and_work():
    tpl = load_script(EXAMPLES / '_template.py')
    shape = load_script(EXAMPLES / 'spot_shape.py')
    assert tpl.ok and shape.ok
    ref = disk_frame(laser_on=False, noise=0, rng=RNG).astype(np.float32)
    tilted = [disk_frame(spots=((300, 230), (340, 250)), noise=3, rng=RNG) for _ in range(2)]
    r = run_offline(EXAMPLES / 'spot_shape.py', tilted, kind='disk', reference=ref, binning=2)
    assert r['valid'].all() and (r['angle'] > 10).all() and (r['ellipticity'] < 0.8).all()
    beams = [beam_frame(x=200, y=150, rng=RNG)]
    r2 = run_offline(EXAMPLES / '_template.py', beams, binning=2, params={'use_depletion': False})
    assert r2['peak_x'][0] == pytest.approx(200, abs=3) and r2['peak_y'][0] == pytest.approx(150, abs=3)


def test_run_on_recorded_images(tmp_path):
    from pymodaq_plugins_camerastation.core.recorder import H5StationWriter, IMG_PERIODIC, reduce_image
    w = H5StationWriter(tmp_path, 'rec', flush_s=0.05)
    w.start()
    for i in range(3):
        img = beam_frame(shape=(200, 300), x=150 + 10 * i, y=90, noise=2, rng=RNG)
        roi = (50, 20, 200, 160)
        q, off, sc = reduce_image(img, roi=roi, binning=2)
        w.write_image('cam1', 'Cam1', q, off, sc, float(i), i, i + 1, IMG_PERIODIC, roi=roi, binning=2)
    w.close()
    r = run_on_session(EXAMPLES / '_template.py', w.paths[0], 'cam1', params={'use_depletion': False})
    assert list(r['counter']) == [1, 2, 3]
    assert r['peak_x'] == pytest.approx([150, 160, 170], abs=3)        # full-frame pixels again
    assert r['peak_y'] == pytest.approx([90, 90, 90], abs=3)
