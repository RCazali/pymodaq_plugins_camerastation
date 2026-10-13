"""User analysis scripts: try new analyses in the lab without restarting PyMoDAQ.

A script is a ``.py`` file in the scripts folder (``~/.pymodaq/camera_station/scripts``)::

    LABELS = ('angle', 'ellipticity')        # required: the values it returns (stable channel labels)
    PARAMS = {'threshold': 0.1}              # optional: numbers / bools / strings, editable live in the UI

    def analyze(frame, ctx):                 # required
        w = ctx.crop(frame)                  # ROI + binning of the camera, float32 (a copy)
        ...
        return {'angle': a, 'ellipticity': e}

* **hot reload**: the folder is checked every few seconds; a saved file is reloaded, its state reset.
  A new version that does not load (syntax error, missing LABELS...) is reported and **the previous
  version keeps running**. Files starting with ``_`` are ignored (``_template.py``).
* **isolation**: scripts run in their own thread, after the built-in analysis of the frame, so a slow or
  broken script never delays the built-in values sent to computer A. An exception gives NaN /
  ``valid = 0`` for that script only. A script stuck longer than the timeout is disabled and a new
  thread takes over (the stuck one is abandoned; restart PyMoDAQ when convenient to free it).
* **the frame is read-only** (it is shared with the built-in analysis and the recorder): work on
  ``ctx.crop(frame)`` or ``frame.astype(...)``.
* **offline**: ``run_offline`` runs the same file on the images of a recording or on any frames.

``ctx`` gives: ``camera`` (name), ``kind``, ``params`` (current values), ``state`` (dict kept between
frames), ``builtin`` (results of the built-in analysis of this frame), ``reference`` / ``dark`` (full
frames or None), ``crop(frame, background=False)``, ``depletion(frame)`` (reference − frame in the ROI),
``moments(w)`` (projection moments), ``to_frame(xb, yb)`` (binned ROI coordinates → full-frame pixels)
and ``show(array)`` (displayed in the preview, view "script output").

No Qt, no PyMoDAQ here.
"""
from __future__ import annotations

import collections
import importlib.util
import itertools
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .analysis import bin_sum, border_level, clip_roi, projection_moments

_ids = itertools.count(1)
PARAM_TYPES = (bool, int, float, str)


class Script:
    """One loaded version of a script file (immutable once built; a reload creates a new object)."""

    def __init__(self, name: str, path: Path, mtime: float, module=None, error: str = '', version: int = 0):
        self.name = name
        self.path = path
        self.mtime = mtime
        self.module = module
        self.error = error
        self.version = version
        self.fn = getattr(module, 'analyze', None) if module is not None else None
        self.labels: Tuple[str, ...] = tuple(getattr(module, 'LABELS', ())) if module is not None else ()
        self.defaults: Dict = dict(getattr(module, 'PARAMS', {}) or {}) if module is not None else {}
        rate = getattr(module, 'RATE_HZ', None) if module is not None else None
        self.rate_hz: Optional[float] = float(rate) if rate else None
        self.doc = (getattr(module, '__doc__', '') or '').strip() if module is not None else ''

    @property
    def ok(self) -> bool:
        return self.fn is not None


def _short_error(e: BaseException) -> str:
    tb = traceback.extract_tb(e.__traceback__)
    line = ''
    for fr in reversed(tb):
        if fr.filename and Path(fr.filename).suffix == '.py' and 'importlib' not in fr.filename:
            line = f' (line {fr.lineno})'
            break
    if isinstance(e, SyntaxError):              # its message already gives the file and the line
        line = ''
    return f'{type(e).__name__}: {e}{line}'


def load_script(path: Path, name: Optional[str] = None, version: int = 1) -> Script:
    """Loads a script file; raises with a readable message when it is not usable."""
    path = Path(path)
    name = name or path.stem
    mtime = path.stat().st_mtime
    spec = importlib.util.spec_from_file_location(f'camera_station_script_{name}_{next(_ids)}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)            # not registered in sys.modules: old versions can be freed
    if not callable(getattr(module, 'analyze', None)):
        raise ValueError('no analyze(frame, ctx) function')
    labels = getattr(module, 'LABELS', None)
    if not labels or isinstance(labels, str) or not all(isinstance(x, str) and x for x in labels):
        raise ValueError('LABELS must be a tuple of names, e.g. LABELS = ("x", "y")')
    params = getattr(module, 'PARAMS', {}) or {}
    if not isinstance(params, dict) or not all(isinstance(v, PARAM_TYPES) for v in params.values()):
        raise ValueError('PARAMS must be a dict of numbers, booleans or strings')
    return Script(name, path, mtime, module, version=version)


class ScriptLibrary:
    """The scripts of a folder, reloaded when their file changes (``scan`` is called by a timer)."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.scripts: Dict[str, Script] = {}
        self.folder_error = ''
        folder_s = str(self.folder)
        if folder_s not in sys.path:            # scripts may share helper modules of the folder
            sys.path.append(folder_s)

    def names(self) -> List[str]:
        return sorted(self.scripts)

    def scan(self, force: bool = False) -> List[Tuple[str, str, str]]:
        """Reloads new / changed files (all of them with ``force``), forgets deleted ones.
        Returns (level, script name, message) to log; level 'info' + 'loaded' = a new version runs."""
        msgs: List[Tuple[str, str, str]] = []
        try:
            files = {}
            with os.scandir(self.folder) as it:
                for e in it:
                    if e.is_file() and e.name.endswith('.py') and not e.name.startswith('_'):
                        files[e.name[:-3]] = (Path(e.path), e.stat().st_mtime)
            self.folder_error = ''
        except OSError as e:
            self.folder_error = f'scripts folder not readable: {e}'
            return msgs
        for name in list(self.scripts):
            if name not in files:
                del self.scripts[name]
                msgs.append(('removed', name, f'script {name} removed'))
        for name, (path, mtime) in sorted(files.items()):
            old = self.scripts.get(name)
            if old is not None and old.mtime == mtime and not force:
                continue
            version = (old.version + 1) if old is not None else 1
            try:
                new = load_script(path, name, version)
                self.scripts[name] = new
                msgs.append(('loaded', name, f'script {name} loaded (version {version}): {", ".join(new.labels)}'))
            except Exception as e:
                err = _short_error(e)
                if old is not None and old.ok:        # keep the working version running
                    kept = Script(name, path, mtime, old.module, error=f'new version not loaded: {err}; '
                                                                       f'version {old.version} still running',
                                  version=old.version)
                    self.scripts[name] = kept
                    msgs.append(('warning', name, f'script {name}: {kept.error}'))
                else:
                    self.scripts[name] = Script(name, path, mtime, None, error=err, version=version)
                    msgs.append(('error', name, f'script {name} not loaded: {err}'))
        return msgs


class ScriptContext:
    def __init__(self, camera: str = '', kind: str = 'beam', params: Optional[Dict] = None,
                 state: Optional[Dict] = None, builtin: Optional[Dict] = None, reference=None, dark=None,
                 roi=(0, 0, 0, 0), binning: int = 1, origin=(0.0, 0.0), scale: float = 1.0):
        self.camera = camera
        self.kind = kind
        self.params = params if params is not None else {}
        self.state = state if state is not None else {}
        self.builtin = builtin or {}
        self.reference = reference
        self.dark = dark
        self.roi = tuple(roi)
        self.binning = max(int(binning), 1)
        self.origin = origin                    # offline: position / binning of the saved image in the frame
        self.scale = float(scale)
        self.shown: Optional[np.ndarray] = None
        self._roi_used = (0, 0, 0, 0)
        self.moments = projection_moments

    def _cut(self, image, binning):
        r = clip_roi(self.roi, np.asarray(image).shape[:2])
        self._roi_used = r
        x0, y0, w, h = r
        b = self.binning if binning is None else max(int(binning), 1)
        a = np.asarray(image)
        if a.ndim == 3:
            a = a.mean(axis=2)
        return bin_sum(a[y0:y0 + h, x0:x0 + w], b), b

    def crop(self, frame, background: bool = False, binning: Optional[int] = None) -> np.ndarray:
        """ROI of the camera, binned (sum), float32 copy; ``background=True`` subtracts the border median."""
        w, self._b = self._cut(frame, binning)
        if background:
            w -= border_level(w)
        return w

    def depletion(self, frame, binning: Optional[int] = None) -> Optional[np.ndarray]:
        """reference − frame in the ROI (None without a reference of the same shape)."""
        if self.reference is None or np.asarray(self.reference).shape[:2] != np.asarray(frame).shape[:2]:
            return None
        ref, _ = self._cut(self.reference, binning)
        img, self._b = self._cut(frame, binning)
        return ref - img

    def to_frame(self, xb: float, yb: float) -> Tuple[float, float]:
        """Coordinates in the last ``crop`` / ``depletion`` array → full-frame camera pixels."""
        b = getattr(self, '_b', self.binning)
        x0, y0 = self._roi_used[0], self._roi_used[1]
        xi, yi = x0 + (xb + 0.5) * b - 0.5, y0 + (yb + 0.5) * b - 0.5
        return self.origin[0] + (xi + 0.5) * self.scale - 0.5, self.origin[1] + (yi + 0.5) * self.scale - 0.5

    def show(self, array):
        self.shown = np.asarray(array)


def run_script(script: Script, frame, ctx: ScriptContext) -> Tuple[Dict[str, float], str]:
    """Calls the script; returns ({label: value, 'valid'}, error). Never raises."""
    res = {lab: math.nan for lab in script.labels}
    res['valid'] = 0.0
    view = np.asarray(frame).view()
    view.flags.writeable = False               # the frame is shared: a script must never modify it
    try:
        out = script.fn(view, ctx)
    except Exception as e:
        return res, _short_error(e)
    if not isinstance(out, dict):
        return res, f'analyze returned {type(out).__name__}, expected a dict'
    for lab in script.labels:
        try:
            v = float(out.get(lab, math.nan))
        except (TypeError, ValueError):
            v = math.nan
        res[lab] = v if math.isfinite(v) else math.nan
    if 'valid' in out:
        try:
            res['valid'] = 1.0 if float(out['valid']) == 1.0 else 0.0
        except (TypeError, ValueError):
            res['valid'] = 0.0
    else:
        res['valid'] = 1.0 if any(math.isfinite(res[lab]) for lab in script.labels) else 0.0
    return res, ''


class ScriptRunner:
    """Runs the scripts enabled on each camera, in its own thread, latest frame wins."""

    def __init__(self, library: ScriptLibrary, clock: Callable[[], float] = time.monotonic):
        self.library = library
        self.clock = clock
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending: 'collections.OrderedDict[str, tuple]' = collections.OrderedDict()
        self.params: Dict[str, Dict] = {}               # script -> current parameter values (GUI thread)
        self.disabled: Dict[str, str] = {}              # script -> reason (watchdog)
        self.results: Dict[Tuple[str, str], Dict] = {}
        self.states: Dict[Tuple[str, str], Dict] = {}
        self.shown: Dict[str, Tuple[str, np.ndarray]] = {}   # camera id -> (script, array)
        self._last_run: Dict[Tuple[str, str], float] = {}
        self._busy: Optional[tuple] = None              # (generation, cam id, script, start)
        self._gen = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self.busy_s = 0.0

    # ------------------------------------------------------------------ thread
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        gen = self._gen
        self._thread = threading.Thread(target=self._run, args=(gen,), name=f'CameraStationScripts{gen}',
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        with self._cv:
            self._stop = True
            self._pending.clear()
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    # ------------------------------------------------------------------ input (engine worker thread)
    def submit(self, cam_cfg, analyzer, image, frame_no: int, t_wall: float, builtin: Dict):
        if not getattr(cam_cfg, 'scripts', None):
            return
        with self._cv:
            self._pending.pop(cam_cfg.id, None)
            self._pending[cam_cfg.id] = (cam_cfg, analyzer, image, frame_no, t_wall, dict(builtin))
            self._cv.notify()

    def _run(self, gen: int):
        while True:
            with self._cv:
                while not self._pending and not self._stop and gen == self._gen:
                    self._cv.wait(0.5)
                if self._stop or gen != self._gen:
                    return
                cam_id, item = self._pending.popitem(last=False)
            self._process(gen, item)

    def _process(self, gen: int, item):
        cfg, analyzer, image, frame_no, t_wall, builtin = item
        for name in list(cfg.scripts):
            script = self.library.scripts.get(name)
            if script is None or not script.ok or name in self.disabled:
                continue
            key = (cfg.id, name)
            now = self.clock()
            if script.rate_hz and now - self._last_run.get(key, -math.inf) < 0.9 / script.rate_hz:
                continue
            self._last_run[key] = now
            state = self.states.setdefault(key, {})
            params = dict(script.defaults)
            params.update(self.params.get(name, {}))
            ctx = ScriptContext(cfg.name, cfg.kind, params, state, builtin, analyzer.reference, analyzer.dark,
                                roi=cfg.roi, binning=cfg.binning)
            self._busy = (gen, cfg.id, name, now)
            t0 = time.perf_counter()
            res, err = run_script(script, image, ctx)
            dt = time.perf_counter() - t0
            if gen != self._gen:                      # abandoned by the watchdog meanwhile
                return
            self._busy = None
            with self._lock:
                old = self.results.get(key, {})
                self.results[key] = {'result': res, 'frame': frame_no, 't_wall': t_wall, 'mono': self.clock(),
                                     'ms': dt * 1e3, 'error': err, 'version': script.version,
                                     'runs': old.get('runs', 0) + 1, 'n_errors': old.get('n_errors', 0) + bool(err)}
                self.busy_s += dt
                if ctx.shown is not None:
                    self.shown[cfg.id] = (name, ctx.shown)

    # ------------------------------------------------------------------ GUI thread
    def watchdog(self, timeout_s: float) -> Optional[str]:
        """A script running for longer than ``timeout_s`` is disabled and a new thread takes over."""
        busy = self._busy
        if busy is None or self.clock() - busy[3] <= timeout_s:
            return None
        gen, cam_id, name, t0 = busy
        with self._cv:
            self._gen += 1
            self._busy = None
            self._cv.notify_all()
        self.disabled[name] = f'stuck for more than {timeout_s:g} s: disabled (save the file again to retry)'
        self._thread = None
        self.start()
        return f'script {name} on {cam_id}: {self.disabled[name]}; its thread is abandoned ' \
               f'(restart PyMoDAQ when convenient to free it)'

    def script_reloaded(self, name: str):
        """New version of a script: fresh state, enabled again."""
        self.disabled.pop(name, None)
        for key in [k for k in self.states if k[1] == name]:
            del self.states[key]
        for key in [k for k in self._last_run if k[1] == name]:
            del self._last_run[key]

    def snapshot(self, cam_id: str, name: str) -> Dict:
        with self._lock:
            r = self.results.get((cam_id, name))
            if r is None:
                return {}
            out = dict(r)
            out['result'] = dict(r['result'])
        out['age'] = self.clock() - out['mono']
        return out


# ---------------------------------------------------------------------- offline
def run_offline(script_path, frames: Iterable, camera: str = 'offline', kind: str = 'beam',
                params: Optional[Dict] = None, reference=None, dark=None, roi=(0, 0, 0, 0), binning: int = 1,
                origin=(0.0, 0.0), scale: float = 1.0) -> Dict[str, np.ndarray]:
    """Runs a script file on frames (e.g. at the office, on recorded images): {label: array, 'error': list}."""
    script = load_script(Path(script_path))
    p = dict(script.defaults)
    p.update(params or {})
    state: Dict = {}
    cols: Dict[str, list] = {lab: [] for lab in script.labels + ('valid',)}
    errors = []
    for frame in frames:
        ctx = ScriptContext(camera, kind, p, state, {}, reference, dark, roi=roi, binning=binning,
                            origin=origin, scale=scale)
        res, err = run_script(script, frame, ctx)
        for k in cols:
            cols[k].append(res[k])
        errors.append(err)
    out = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    out['error'] = errors
    return out


def run_on_session(script_path, session_file, camera_id: str, params: Optional[Dict] = None, kind: str = 'beam',
                   reference=None) -> Dict[str, np.ndarray]:
    """Runs a script on the images of camera ``camera_id`` saved in a Camera Station recording.

    The saved images are cropped / binned / quantised (see the recorder): ``ctx.to_frame`` still returns
    full-frame pixels. ``reference`` (full frame, or a .npy path) is cut and binned like the images."""
    from .loader import load_session
    s = load_session(session_file)
    if isinstance(reference, (str, Path)):
        reference = np.load(reference)
    out: Dict[str, list] = {}
    groups = sorted(g for g, info in s['images'].items() if info['camera_id'] == camera_id or g == camera_id)
    if not groups:
        raise KeyError(f'no images of camera "{camera_id}" in {session_file}: {sorted(s["images"])}')
    for g in groups:
        info = s['images'][g]
        x0, y0, w, h = info['roi']
        b = info['binning']
        ref = None
        if reference is not None:
            r = np.asarray(reference, dtype=np.float32)[y0:y0 + h, x0:x0 + w]
            ref = bin_sum(r, b) / float(b * b)
        frames = (s['image'](camera_id, i, group=g) for i in range(info['n']))
        res = run_offline(script_path, frames, camera=camera_id, kind=kind, params=params, reference=ref,
                          origin=(x0, y0), scale=b)
        for k, v in res.items():
            out.setdefault(k, []).extend(list(v))
        for k in ('t', 'counter', 'frame', 'kind', 'event_id'):
            out.setdefault(k, []).extend(list(info[k]))
    return {k: (v if k == 'error' else np.asarray(v)) for k, v in out.items()}
