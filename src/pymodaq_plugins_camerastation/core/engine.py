"""Analysis engine: frames in (GUI thread), results out, one worker thread for all the cameras.

Load and robustness rules:

* **rate gate**: a camera is analysed at most ``rate_hz`` times per second; the other frames are only
  counted (no copy, no work). The camera itself may run faster without costing anything here.
* **latest wins**: one pending frame per camera at most. If the worker is late, the older pending
  frame is replaced, never queued: memory and latency are bounded, whatever happens.
* **one worker**: the analyses never use more than one core, the GUI thread never computes.
* **isolation**: an exception in the analysis of one camera gives NaN / ``valid = 0`` and an error
  text for that camera only; the worker keeps running.
* **staleness** is computed from the time of the last result, so a camera that stops sending frames
  is detected by the publisher (values NaN after ``stale_s``).

No Qt, no PyMoDAQ: tested with plain numpy frames.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from .analysis import CameraAnalyzer, nan_result

ST_OFF, ST_WAITING, ST_OK, ST_INVALID, ST_NO_FRAMES, ST_ERROR = 'off', 'waiting', 'ok', 'invalid', 'no_frames', 'error'

# shape / symbol + text, never colour alone (laser goggles)
STATE_TEXT = {
    ST_OFF: '■ off',
    ST_WAITING: '… waiting',
    ST_OK: '▶ ok',
    ST_INVALID: '⚠ no beam',
    ST_NO_FRAMES: '⚠ no frames',
    ST_ERROR: '✖ error',
}


class RateMeter:
    """Events per second over the last few seconds (cheap, for the status display)."""

    def __init__(self, window_s: float = 5.0):
        self.window_s = window_s
        self._t: 'collections.deque[float]' = collections.deque(maxlen=128)

    def tick(self, now: float):
        self._t.append(now)

    def rate(self, now: float) -> float:
        while self._t and now - self._t[0] > self.window_s:
            self._t.popleft()
        if len(self._t) < 2:
            return 0.0
        span = now - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0


class Capture:
    """Average of the next ``n`` frames of a camera (reference or dark frame)."""

    def __init__(self, what: str, n: int, done: Callable[[str, np.ndarray], None]):
        self.what = what
        self.n = max(int(n), 1)
        self.done = done
        self.count = 0
        self.acc: Optional[np.ndarray] = None

    def add(self, image) -> bool:
        a = np.asarray(image)
        if a.ndim == 3:
            a = a.mean(axis=2)
        if self.acc is None:
            self.acc = np.zeros(a.shape, dtype=np.float64)
        elif self.acc.shape != a.shape:            # the camera changed its format: start again
            self.acc = np.zeros(a.shape, dtype=np.float64)
            self.count = 0
        self.acc += a
        self.count += 1
        return self.count >= self.n

    def result(self) -> np.ndarray:
        return (self.acc / max(self.count, 1)).astype(np.float32)


class CameraRuntime:
    def __init__(self, cfg):
        self.cfg = cfg
        self.analyzer = CameraAnalyzer(cfg)
        # intake (GUI thread)
        self.frames_in = 0
        self.dropped_pending = 0
        self.last_frame_mono: Optional[float] = None
        self.last_submit_mono = -math.inf
        self.latest = None                      # (image, frame number, wall time): latest frame received
        self.in_rate = RateMeter()
        self.capture: Optional[Capture] = None
        # results (written by the worker, under the engine lock)
        self.result: Dict[str, float] = nan_result(cfg.kind)
        self.result_frame = -1
        self.result_wall = math.nan
        self.result_mono: Optional[float] = None
        self.frames_done = 0
        self.calc_ms = math.nan
        self.error = ''
        self.reason = ''
        self.n_errors = 0
        self.done_rate = RateMeter()


class AnalysisEngine:
    def __init__(self, clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time):
        self.clock = clock
        self.wall = wall
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending: 'collections.OrderedDict[str, tuple]' = collections.OrderedDict()
        self.cameras: Dict[str, CameraRuntime] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self.busy_s = 0.0                       # total time spent analysing (CPU budget display)
        self.on_error: Optional[Callable[[str, str], None]] = None   # (camera id, message), worker thread
        # (runtime, (image, frame, t_wall), results) after each analysis, worker thread: feeds the user scripts
        self.after_analysis: Optional[Callable] = None

    # ------------------------------------------------------------------ configuration (GUI thread)
    def set_cameras(self, cfgs: Iterable):
        cfgs = list(cfgs)
        with self._lock:
            keep = {}
            for cfg in cfgs:
                rt = self.cameras.get(cfg.id)
                if rt is None:
                    rt = CameraRuntime(cfg)
                else:
                    rt.cfg = cfg
                    rt.analyzer.settings = cfg
                keep[cfg.id] = rt
            self.cameras = keep
            for cam_id in list(self._pending):
                if cam_id not in keep:
                    del self._pending[cam_id]

    def runtime(self, cam_id: str) -> Optional[CameraRuntime]:
        return self.cameras.get(cam_id)

    # ------------------------------------------------------------------ thread
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, name='CameraStationAnalysis', daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0):
        with self._cv:
            self._stop = True
            self._pending.clear()
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ intake (GUI thread)
    def submit(self, cam_id: str, image, t_wall: Optional[float] = None) -> bool:
        """New frame of a camera. Returns True when it was handed to the worker."""
        rt = self.cameras.get(cam_id)
        if rt is None or image is None:
            return False
        now = self.clock()
        rt.frames_in += 1
        rt.last_frame_mono = now
        rt.in_rate.tick(now)
        rt.latest = (image, rt.frames_in, self.wall() if t_wall is None else t_wall)
        if rt.capture is not None:
            try:
                if rt.capture.add(image):
                    cap, rt.capture = rt.capture, None
                    cap.done(cam_id, cap.result())
            except Exception as e:      # never break the intake for a capture
                rt.capture = None
                rt.error = f'capture failed: {e}'
        if not rt.cfg.enabled:
            return False
        period = 1.0 / max(rt.cfg.rate_hz, 1e-3)
        if now - rt.last_submit_mono < 0.9 * period:      # 10 % tolerance for the camera jitter
            return False
        rt.last_submit_mono = now
        with self._cv:
            if cam_id in self._pending:
                rt.dropped_pending += 1
                del self._pending[cam_id]
            self._pending[cam_id] = (image, rt.frames_in, rt.latest[2])
            self._cv.notify()
        return True

    def start_capture(self, cam_id: str, what: str, n: int, done: Callable[[str, np.ndarray], None]) -> bool:
        rt = self.cameras.get(cam_id)
        if rt is None:
            return False
        rt.capture = Capture(what, n, done)
        return True

    # ------------------------------------------------------------------ worker
    def _run(self):
        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                cam_id, item = self._pending.popitem(last=False)
                rt = self.cameras.get(cam_id)
            if rt is None:
                continue
            self.process(rt, item)

    def process(self, rt: CameraRuntime, item: tuple):
        """Analyse one frame (worker thread; also called directly by the tests)."""
        image, frame_no, t_wall = item
        t0 = time.perf_counter()
        error = ''
        try:
            res = rt.analyzer.analyze(image)
        except Exception as e:
            res = nan_result(rt.cfg.kind)
            error = f'{type(e).__name__}: {e}'
        dt = time.perf_counter() - t0
        now = self.clock()
        with self._lock:
            rt.result = res
            rt.result_frame = frame_no
            rt.result_wall = t_wall
            rt.result_mono = now
            rt.frames_done += 1
            rt.calc_ms = dt * 1e3
            rt.reason = rt.analyzer.last_reason if not error else ''
            if error:
                rt.n_errors += 1
            new_error = error and error != rt.error
            rt.error = error
            rt.done_rate.tick(now)
            self.busy_s += dt
        if new_error and self.on_error is not None:
            try:
                self.on_error(rt.cfg.id, error)
            except Exception:
                pass
        if self.after_analysis is not None:
            try:
                self.after_analysis(rt, item, res)
            except Exception:
                pass

    # ------------------------------------------------------------------ reading (GUI thread)
    def state(self, cam_id: str, stale_s: float) -> str:
        rt = self.cameras.get(cam_id)
        if rt is None or not rt.cfg.enabled:
            return ST_OFF
        now = self.clock()
        if rt.last_frame_mono is None:
            return ST_WAITING
        if now - rt.last_frame_mono > stale_s:
            return ST_NO_FRAMES
        with self._lock:
            if rt.error:
                return ST_ERROR
            if rt.result_mono is None:
                return ST_WAITING
            if now - rt.result_mono > stale_s:
                return ST_NO_FRAMES
            return ST_OK if rt.result.get('valid', 0.0) == 1.0 else ST_INVALID

    def snapshot(self, cam_id: str) -> Dict:
        """Copy of the last result of a camera with its age (s); ``age`` is inf without a result."""
        rt = self.cameras.get(cam_id)
        if rt is None:
            return {}
        now = self.clock()
        with self._lock:
            return {
                'result': dict(rt.result),
                'frame': rt.result_frame,
                't_wall': rt.result_wall,
                'age': (now - rt.result_mono) if rt.result_mono is not None else math.inf,
                'calc_ms': rt.calc_ms,
                'error': rt.error,
                'reason': rt.reason,
                'frames_in': rt.frames_in,
                'frames_done': rt.frames_done,
                'in_rate': rt.in_rate.rate(now),
                'done_rate': rt.done_rate.rate(now),
            }

    def latest_images(self) -> Dict[str, tuple]:
        return {cid: rt.latest for cid, rt in self.cameras.items() if rt.latest is not None}

    def ids(self) -> List[str]:
        return list(self.cameras)
