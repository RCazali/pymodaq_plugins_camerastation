"""HDF5 logger of computer B (stage 0.5), synchronised with the Laser Control recordings of computer A.

What is written (one file per session, rotation on size / duration)::

    /                       attrs: format, format_version, machine, host, started, station_json, ...
    /scalars/t              float64  wall clock of B (s since epoch)
    /scalars/frame_counter  int64    heartbeat value of the publication (same value received by A)
    /scalars/values/<name>  float64  'Cam1|x', 'Stats|t_wall'... ('/' written as '|'), NaN padded
    /images/<cam id>[~n]/data       (n, h, w) uint8 or uint16, gzip + shuffle, one chunk per image
                        /t, /frame, /counter, /kind, /event_id, /offset, /scale
                        attrs: name, roi (x0, y0, w, h of the saved area), binning, bits
    /events/{t, id, level, kind, source, message, data}

Synchronisation with computer A (no shared clock needed):

* every publication of B carries ``Stats/frame_counter``; Laser Control records it in each of its
  snapshots. The same counter is the row key of ``/scalars`` here, and each image stores the counter
  of the last publication -> exact join between the two files (``loader.align_with_laser_control``);
* ``Stats/t_wall`` (clock of B) is published too, so A's file contains the clock offset between the
  two computers at every snapshot.

Images are reduced before writing (binning, 8 bits with a per-image offset and scale: counts ~
offset + value * scale), so a 1 Mpx 12-bit camera costs ~30-80 kB per image instead of 2 MB.

Robustness: the GUI thread only puts items in a queue bounded **in bytes**; when it is full, images are
dropped and counted (never the GUI blocked, never the memory exhausted). A disk error is reported in
``error``, the batch is counted as lost and the writer goes on in a new file (``_002``...); the file is
flushed every few seconds (a crash loses seconds).
"""
from __future__ import annotations

import json
import math
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .analysis import bin_sum

FORMAT_NAME = 'camera_station_session'
FORMAT_VERSION = 1

IMG_PERIODIC, IMG_EVENT, IMG_MANUAL = 0, 1, 2


def escape(name: str) -> str:
    return str(name).replace('/', '|')


def unescape(name: str) -> str:
    return str(name).replace('|', '/')


def reduce_image(image, roi=None, binning: int = 2, bits: int = 8) -> Tuple[np.ndarray, float, float]:
    """Crop, bin (mean) and quantise a frame. Returns (array, offset, scale): counts ~ offset + a * scale.

    8 bits: offset = minimum, scale = (max - min) / 255 for each image (full dynamic for dim images).
    16 bits: counts kept as they are (offset 0, scale 1) when they fit, otherwise rescaled likewise."""
    a = np.asarray(image)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if roi is not None:
        x0, y0, w, h = roi
        a = a[y0:y0 + h, x0:x0 + w]
    b = max(int(binning), 1)
    a = bin_sum(a, b)
    if b > 1:
        a /= float(b * b)                       # mean: values stay in counts per pixel
    lo = float(a.min()) if a.size else 0.0
    hi = float(a.max()) if a.size else 0.0
    if bits == 16:
        if lo >= 0 and hi <= 65535 and np.all(a == np.round(a)):
            return a.astype(np.uint16), 0.0, 1.0
        top, dtype = 65535.0, np.uint16
    else:
        top, dtype = 255.0, np.uint8
    scale = (hi - lo) / top if hi > lo else 1.0
    q = np.rint((a - lo) / scale)
    np.clip(q, 0, top, out=q)
    return q.astype(dtype), lo, scale


class H5StationWriter:
    def __init__(self, folder: Path, base_name: str, metadata: Optional[Dict[str, Any]] = None,
                 max_mb: float = 1000.0, max_hours: float = 12.0, flush_s: float = 3.0,
                 max_queue_mb: float = 200.0, compression: int = 4):
        self.folder = Path(folder)
        self.base_name = base_name
        self.metadata = dict(metadata or {})
        self.metadata.setdefault('host', socket.gethostname())
        self.metadata.setdefault('started', time.strftime('%Y-%m-%d %H:%M:%S'))
        self.max_bytes = max(float(max_mb) * 1e6, 1.0)
        self.max_s = max(float(max_hours), 0.01) * 3600.0
        self.flush_s = flush_s
        self.max_queue_bytes = max(float(max_queue_mb), 1.0) * 1e6
        self.compression = int(compression)
        self._q: 'queue.Queue' = queue.Queue()
        self._queued_bytes = 0
        self._qlock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._file = None
        self._file_index = 0
        self._opened_at = 0.0
        self.current_path: Optional[Path] = None
        self.paths: List[Path] = []
        self.n_rows = 0
        self.n_images = 0
        self.n_events = 0
        self.n_dropped_images = 0
        self.n_dropped_rows = 0
        self.error = ''
        self._img_groups: Dict[str, str] = {}        # cam id -> current group name (changes with the shape)

    # ------------------------------------------------------------------ API (GUI thread)
    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name='CameraStationH5Writer', daemon=True)
        self._thread.start()

    def write_row(self, t: float, counter: int, values: Dict[str, float]):
        self._q.put(('row', (float(t), int(counter), dict(values))))

    def write_image(self, cam_id: str, name: str, image: np.ndarray, offset: float, scale: float,
                    t: float, frame: int, counter: int, kind: int, event_id: int = -1,
                    roi=(0, 0, 0, 0), binning: int = 1) -> bool:
        nbytes = int(getattr(image, 'nbytes', 0))
        with self._qlock:
            if self._queued_bytes + nbytes > self.max_queue_bytes:
                self.n_dropped_images += 1
                return False
            self._queued_bytes += nbytes
        self._q.put(('image', (cam_id, name, image, float(offset), float(scale), float(t), int(frame),
                               int(counter), int(kind), int(event_id), tuple(int(v) for v in roi), int(binning))))
        return True

    def write_event(self, t: float, kind: str, message: str, source: str = 'station', level: int = 20,
                    event_id: int = -1, data: Optional[Dict[str, Any]] = None):
        self._q.put(('event', {'t': float(t), 'id': int(event_id), 'level': int(level), 'kind': str(kind),
                               'source': str(source), 'message': str(message), 'data': data or {}}))

    def update_metadata(self, metadata: Dict[str, Any]):
        self._q.put(('meta', dict(metadata)))

    def close(self, timeout: float = 15.0):
        if self._thread is None:
            return
        self._q.put(('close', None))
        self._thread.join(timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def queued_mb(self) -> float:
        return self._queued_bytes / 1e6

    # ------------------------------------------------------------------ writer thread
    def _run(self):
        rows: List[tuple] = []
        images: List[tuple] = []
        events: List[dict] = []
        last_flush = time.monotonic()
        closing = False
        while not closing:
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                item = None
            if item is not None:
                kind, payload = item
                if kind == 'row':
                    rows.append(payload)
                elif kind == 'image':
                    images.append(payload)
                elif kind == 'event':
                    events.append(payload)
                elif kind == 'meta':
                    self.metadata.update(payload)
                    self._safe(self._write_attrs)
                elif kind == 'close':
                    closing = True
            if closing or len(rows) + len(events) > 500 or len(images) >= 8 \
                    or time.monotonic() - last_flush > self.flush_s:
                if rows or images or events:
                    if not self._safe(lambda: self._flush(rows, images, events)):
                        # a half-written batch is never retried (it would misalign the columns): it is
                        # counted as lost and the next batch goes into a new file
                        self.n_dropped_rows += len(rows)
                        self.n_dropped_images += len(images)
                        self._file_index += 1
                    self._release(images)
                    rows, images, events = [], [], []
                last_flush = time.monotonic()
        self._safe(self._close_file)

    def _release(self, images: List[tuple]):
        with self._qlock:
            self._queued_bytes -= sum(int(getattr(i[2], 'nbytes', 0)) for i in images)
            self._queued_bytes = max(self._queued_bytes, 0)

    def _safe(self, fn) -> bool:
        try:
            fn()
            self.error = ''
            return True
        except Exception as e:  # a disk problem must never kill the extension
            self.error = f'{type(e).__name__}: {e}'
            try:
                self._close_file()     # reopen cleanly at the next flush
            except Exception:
                self._file = None
            return False

    def _flush(self, rows, images, events):
        self._ensure_file()
        self._write_rows(rows)
        for img in images:
            self._write_image(*img)
        self._write_events(events)
        self._file.flush()
        self._rotate_if_needed()

    def _new_path(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        suffix = '' if self._file_index == 0 else f'_{self._file_index + 1:03d}'
        return self.folder / f'{self.base_name}{suffix}.h5'

    def _ensure_file(self):
        if self._file is not None:
            return
        import h5py
        path = self._new_path()
        self._file = h5py.File(path, 'a')
        self._opened_at = time.monotonic()
        if path not in self.paths:
            self.paths.append(path)
        self.current_path = path
        self._img_groups = {}
        f = self._file
        s = f.require_group('scalars')
        for name, dt in (('t', 'f8'), ('frame_counter', 'i8')):
            if name not in s:
                s.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt, chunks=(1024,))
        s.require_group('values')
        f.require_group('images')
        str_dt = h5py.string_dtype('utf-8')
        e = f.require_group('events')
        for name, dt in (('t', 'f8'), ('id', 'i8'), ('level', 'i2'), ('kind', str_dt), ('source', str_dt),
                         ('message', str_dt), ('data', str_dt)):
            if name not in e:
                e.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt, chunks=(256,))
        self._write_attrs()

    def _write_attrs(self):
        if self._file is None:
            return
        attrs = self._file.attrs
        attrs['format'] = FORMAT_NAME
        attrs['format_version'] = FORMAT_VERSION
        attrs['file_index'] = self._file_index + 1
        for k, v in self.metadata.items():
            try:
                if isinstance(v, (dict, list)):
                    attrs[k] = json.dumps(v, ensure_ascii=False, default=str)
                elif v is None:
                    attrs[k] = ''
                else:
                    attrs[k] = v
            except Exception:
                attrs[k] = str(v)

    @staticmethod
    def _append(ds, values):
        n0 = ds.shape[0]
        ds.resize((n0 + len(values),) + ds.shape[1:])
        ds[n0:] = values

    def _write_rows(self, rows):
        if not rows:
            return
        s = self._file['scalars']
        n0 = s['t'].shape[0]
        self._append(s['t'], [r[0] for r in rows])
        self._append(s['frame_counter'], [r[1] for r in rows])
        vg = s['values']
        names = set()
        for r in rows:
            names.update(r[2].keys())
        for name in names:
            if escape(name) not in vg:           # a column appearing mid-session: NaN before
                ds = vg.create_dataset(escape(name), shape=(n0,), maxshape=(None,), dtype='f8',
                                       chunks=(1024,), fillvalue=np.nan)
                ds.attrs['name'] = name
        for esc in vg.keys():
            name = unescape(esc)
            self._append(vg[esc], np.array([r[2].get(name, math.nan) for r in rows], dtype=float))
        self.n_rows += len(rows)

    def _image_group(self, cam_id: str, shape, dtype, attrs):
        import h5py  # noqa: F401
        root = self._file['images']
        gname = self._img_groups.get(cam_id, cam_id)
        n = 1
        while True:
            if gname not in root:
                g = root.create_group(gname)
                g.create_dataset('data', shape=(0,) + tuple(shape), maxshape=(None,) + tuple(shape), dtype=dtype,
                                 chunks=(1,) + tuple(shape), compression='gzip',
                                 compression_opts=self.compression, shuffle=True)
                for name, dt in (('t', 'f8'), ('frame', 'i8'), ('counter', 'i8'), ('kind', 'i1'),
                                 ('event_id', 'i8'), ('offset', 'f8'), ('scale', 'f8')):
                    g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt, chunks=(256,))
                for k, v in attrs.items():
                    g.attrs[k] = v
                break
            g = root[gname]
            if g['data'].shape[1:] == tuple(shape) and g['data'].dtype == dtype \
                    and tuple(g.attrs.get('roi', ())) == tuple(attrs['roi']) \
                    and int(g.attrs.get('binning', 0)) == attrs['binning']:
                break
            n += 1                                 # new shape / area: new group, never a reshaped one
            gname = f'{cam_id}~{n}'
        self._img_groups[cam_id] = gname
        return g

    def _write_image(self, cam_id, name, image, offset, scale, t, frame, counter, kind, event_id, roi, binning):
        attrs = {'name': name, 'camera_id': cam_id, 'roi': np.asarray(roi, dtype=np.int64),
                 'binning': int(binning), 'bits': int(image.dtype.itemsize * 8),
                 'counts': 'offset + data * scale'}
        g = self._image_group(cam_id, image.shape, image.dtype, attrs)
        self._append(g['data'], image[None, ...])
        for key, v in (('t', t), ('frame', frame), ('counter', counter), ('kind', kind), ('event_id', event_id),
                       ('offset', offset), ('scale', scale)):
            self._append(g[key], [v])
        self.n_images += 1

    def _write_events(self, events):
        if not events:
            return
        e = self._file['events']
        cols = {
            't': [ev['t'] for ev in events],
            'id': [ev['id'] for ev in events],
            'level': [ev['level'] for ev in events],
            'kind': [ev['kind'] for ev in events],
            'source': [ev['source'] for ev in events],
            'message': [ev['message'] for ev in events],
            'data': [json.dumps(ev['data'], ensure_ascii=False, default=str) for ev in events],
        }
        for name, col in cols.items():
            self._append(e[name], col)
        self.n_events += len(events)

    def _rotate_if_needed(self):
        if self._file is None or self.current_path is None:
            return
        size = self.current_path.stat().st_size if self.current_path.exists() else 0
        if size > self.max_bytes or time.monotonic() - self._opened_at > self.max_s:
            self._close_file()
            self._file_index += 1

    def _close_file(self):
        if self._file is not None:
            f, self._file = self._file, None
            f.flush()
            f.close()
