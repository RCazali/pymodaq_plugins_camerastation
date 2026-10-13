"""Settings of the station and of each camera, and their JSON file (pure Python).

Robustness rules:

* every value is validated and clamped when loaded (a hand-edited file cannot crash the extension);
* the file is written atomically (temporary file, then replace) and the previous version is kept as
  ``.bak``; an unreadable file is renamed ``.corrupt-<date>`` and the station starts with defaults
  (the problem is reported, never hidden);
* camera names (the published channel names) are unique and never contain ``/``.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .analysis import BACKGROUNDS, KIND_BEAM, LABELS

SCHEMA_VERSION = 1
RESERVED_NAMES = {'Stats'}


def _num(v, default, lo=-math.inf, hi=math.inf, cast=float):
    try:
        x = cast(v)
        if isinstance(x, float) and not math.isfinite(x):
            return default
    except (TypeError, ValueError):
        return default
    return min(max(x, lo), hi)


def clean_name(name: str, default: str = 'Cam') -> str:
    """Channel-safe camera name: letters, digits, '_' and '-' only."""
    n = re.sub(r'[^A-Za-z0-9_\-]+', '_', str(name or '')).strip('_')
    return n or default


@dataclass
class CameraConfig:
    id: str = 'cam1'                  # stable identifier (files, HDF5 groups); never changes
    name: str = 'Cam1'                # published channel name: "<viewer on A>/<name>/<label>"
    detector: str = ''                # title of the 2D DAQ_Viewer of this dashboard
    enabled: bool = True
    kind: str = KIND_BEAM
    rate_hz: float = 2.0              # maximum analysis rate (frames above it are dropped, not queued)
    roi: List[int] = field(default_factory=lambda: [0, 0, 0, 0])   # x0, y0, w, h (0 = up to the edge)
    binning: int = 2
    background: str = 'border'        # none / border / dark
    threshold: float = 0.05           # fraction of the maximum below which pixels are ignored
    min_signal: float = 0.0           # counts per pixel above background, below = no beam (valid = 0)
    saturation: float = 0.0           # saturation level in counts; 0 = maximum of the image type
    mask_cx: float = 0.0              # disk: circular mask in full-frame pixels (radius 0 = no mask)
    mask_cy: float = 0.0
    mask_radius: float = 0.0
    reference_file: str = ''          # disk: reference image (.npy) in the references folder
    dark_file: str = ''
    record_images: bool = True
    scripts: List[str] = field(default_factory=list)   # user scripts run on this camera (core.scripts)

    def labels(self) -> Tuple[str, ...]:
        return LABELS.get(self.kind, LABELS[KIND_BEAM])

    def validate(self) -> 'CameraConfig':
        self.id = clean_name(self.id, 'cam')
        self.name = clean_name(self.name, self.id)
        self.detector = str(self.detector or '')
        self.enabled = bool(self.enabled)
        if self.kind not in LABELS:
            self.kind = KIND_BEAM
        self.rate_hz = _num(self.rate_hz, 2.0, 0.05, 30.0)
        roi = list(self.roi) if isinstance(self.roi, (list, tuple)) else []
        self.roi = [int(_num(v, 0, 0, 1e6, int)) for v in (roi + [0, 0, 0, 0])[:4]]
        self.binning = int(_num(self.binning, 2, 1, 16, int))
        if self.background not in BACKGROUNDS:
            self.background = 'border'
        self.threshold = _num(self.threshold, 0.05, 0.0, 0.95)
        self.min_signal = _num(self.min_signal, 0.0, 0.0)
        self.saturation = _num(self.saturation, 0.0, 0.0)
        self.mask_cx = _num(self.mask_cx, 0.0, 0.0)
        self.mask_cy = _num(self.mask_cy, 0.0, 0.0)
        self.mask_radius = _num(self.mask_radius, 0.0, 0.0)
        self.reference_file = Path(str(self.reference_file or '')).name
        self.dark_file = Path(str(self.dark_file or '')).name
        self.record_images = bool(self.record_images)
        scripts = self.scripts if isinstance(self.scripts, (list, tuple)) else []
        self.scripts = list(dict.fromkeys(str(s) for s in scripts if s))
        return self


@dataclass
class StationConfig:
    schema_version: int = SCHEMA_VERSION
    publish_hz: float = 2.0           # 0D values sent to computer A (and logged) per second
    stale_s: float = 3.0              # a camera without a new result for this long publishes NaN
    image_period_s: float = 30.0      # periodic reduced images in the recording (0 = none)
    image_binning: int = 2
    image_bits: int = 8               # 8 (per-image offset / scale) or 16
    image_full_frame: bool = True     # False: only the analysis ROI of each camera
    burst_frames: int = 5             # images saved after a trigger event (per camera)
    trigger_folder: str = ''          # folder where Laser Control writes its event files (O8)
    preview: bool = False             # preview image (costs a little CPU, off by default)
    script_timeout_s: float = 10.0    # a user script running longer is disabled (core.scripts)
    script_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # name -> {publish, params}
    cameras: List[CameraConfig] = field(default_factory=list)

    def script_setting(self, name: str) -> Dict[str, Any]:
        s = self.script_settings.setdefault(name, {})
        s.setdefault('publish', False)
        s.setdefault('params', {})
        return s

    def validate(self) -> 'StationConfig':
        self.schema_version = SCHEMA_VERSION
        self.publish_hz = _num(self.publish_hz, 2.0, 0.1, 20.0)
        self.stale_s = _num(self.stale_s, 3.0, 0.5, 600.0)
        self.image_period_s = _num(self.image_period_s, 30.0, 0.0, 86400.0)
        self.image_binning = int(_num(self.image_binning, 2, 1, 16, int))
        self.image_bits = 16 if int(_num(self.image_bits, 8, 8, 16, int)) > 8 else 8
        self.image_full_frame = bool(self.image_full_frame)
        self.burst_frames = int(_num(self.burst_frames, 5, 0, 100, int))
        self.trigger_folder = str(self.trigger_folder or '')
        self.preview = bool(self.preview)
        self.script_timeout_s = _num(self.script_timeout_s, 10.0, 1.0, 600.0)
        ss = self.script_settings if isinstance(self.script_settings, dict) else {}
        self.script_settings = {}
        for name, s in ss.items():
            if isinstance(s, dict):
                params = s.get('params', {})
                self.script_settings[str(name)] = {
                    'publish': bool(s.get('publish', False)),
                    'params': {str(k): v for k, v in (params.items() if isinstance(params, dict) else [])
                               if isinstance(v, (bool, int, float, str))}}
        seen_ids, seen_names = set(), set(RESERVED_NAMES)
        for cam in self.cameras:
            cam.validate()
            cam.id = _unique(cam.id, seen_ids)
            cam.name = _unique(cam.name, seen_names)
        return self

    def camera(self, cam_id: str) -> Optional[CameraConfig]:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        return None

    def new_camera(self, detector: str = '') -> CameraConfig:
        n = 1
        ids = {c.id for c in self.cameras}
        names = {c.name for c in self.cameras}
        while f'cam{n}' in ids or f'Cam{n}' in names:
            n += 1
        cam = CameraConfig(id=f'cam{n}', name=f'Cam{n}', detector=detector)
        self.cameras.append(cam)
        return cam

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StationConfig':
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        cam_known = {f.name for f in dataclasses.fields(CameraConfig)}
        cams = []
        for c in d.get('cameras', []) or []:
            if isinstance(c, dict):
                cams.append(CameraConfig(**{k: v for k, v in c.items() if k in cam_known}))
        st = cls(**{k: v for k, v in d.items() if k in known and k != 'cameras'})
        st.cameras = cams
        return st.validate()


def _unique(name: str, seen: set) -> str:
    base, n, out = name, 2, name
    while out in seen:
        out = f'{base}_{n}'
        n += 1
    seen.add(out)
    return out


class SettingsStore:
    """Reads / writes ``StationConfig`` as JSON. ``load_error`` tells why defaults were used."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.load_error = ''

    def load(self) -> StationConfig:
        self.load_error = ''
        if not self.path.exists():
            return StationConfig().validate()
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('not a JSON object')
            return StationConfig.from_dict(data)
        except Exception as e:
            stamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            bad = self.path.with_name(f'{self.path.name}.corrupt-{stamp}')
            try:
                self.path.replace(bad)
                self.load_error = f'settings file unreadable ({e}); kept as {bad.name}, defaults used'
            except OSError:
                self.load_error = f'settings file unreadable ({e}); defaults used'
            return StationConfig().validate()

    def save(self, station: StationConfig):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(station.to_dict(), indent=2, ensure_ascii=False)
        tmp = self.path.with_name(self.path.name + '.tmp')
        tmp.write_text(text, encoding='utf-8')
        if self.path.exists():
            try:      # copy, not move: the settings file exists at every instant
                shutil.copy2(self.path, self.path.with_name(self.path.name + '.bak'))
            except OSError:
                pass
        tmp.replace(self.path)
