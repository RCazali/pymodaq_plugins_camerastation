"""Reading the Camera Station recordings, and joining them with the Laser Control recordings.

    >>> from pymodaq_plugins_camerastation.core.loader import load_session, align_with_laser_control
    >>> b = load_session('session_20261010_101500.h5')
    >>> b['scalars']['Cam1/x'], b['t'], b['frame_counter']
    >>> img = b['image'](cam_id='cam1', index=-1)     # counts (float32) of the last saved image
    >>> m = align_with_laser_control('lab_A.h5', 'session_B.h5')

Only numpy and h5py are needed (no PyMoDAQ, no Qt): the files can be analysed on any computer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .recorder import unescape


def load_session(path) -> Dict:
    """Scalars and events in memory; images read on demand with ``out['image'](cam_id, index)``."""
    import h5py
    path = Path(path)
    out: Dict = {'path': path, 'attrs': {}, 'scalars': {}, 'events': [], 'images': {}}
    with h5py.File(path, 'r') as f:
        out['attrs'] = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        s = f['scalars']
        n = min([s['t'].shape[0], s['frame_counter'].shape[0]] +
                [ds.shape[0] for ds in s['values'].values()])          # a failed batch may leave a ragged end
        out['t'] = s['t'][:n]
        out['frame_counter'] = s['frame_counter'][:n]
        for esc, ds in s['values'].items():
            out['scalars'][unescape(esc)] = ds[:n]
        e = f['events']
        for i in range(e['t'].shape[0]):
            def _s(name):
                v = e[name][i]
                return v.decode() if isinstance(v, bytes) else str(v)
            try:
                data = json.loads(_s('data'))
            except Exception:
                data = {}
            out['events'].append({'t': float(e['t'][i]), 'id': int(e['id'][i]), 'level': int(e['level'][i]),
                                  'kind': _s('kind'), 'source': _s('source'), 'message': _s('message'),
                                  'data': data})
        for gname, g in f['images'].items():
            out['images'][gname] = {
                'name': g.attrs.get('name', gname), 'camera_id': g.attrs.get('camera_id', gname),
                'roi': tuple(int(v) for v in g.attrs.get('roi', (0, 0, 0, 0))),
                'binning': int(g.attrs.get('binning', 1)), 'n': int(g['data'].shape[0]),
                'shape': tuple(g['data'].shape[1:]),
                **{k: g[k][:] for k in ('t', 'frame', 'counter', 'kind', 'event_id', 'offset', 'scale')},
            }

    def image(cam_id: str, index: int = -1, group: Optional[str] = None) -> np.ndarray:
        """Counts of one saved image (offset + data * scale), float32. ``group`` overrides ``cam_id``
        when a camera has several groups (shape changed during the session: 'cam1~2'...)."""
        gname = group or cam_id
        with h5py.File(path, 'r') as f:
            g = f['images'][gname]
            raw = g['data'][index].astype(np.float32)
            return float(g['offset'][index]) + raw * float(g['scale'][index])

    out['image'] = image
    return out


def align_with_laser_control(laser_control_file, station_file, viewer: Optional[str] = None) -> Dict:
    """Joins a Laser Control session (computer A) with a Camera Station session (computer B).

    In A's snapshots the counter is ``raw/<viewer>/Stats/frame_counter`` (every received channel,
    the default) or ``det/<viewer>/heartbeat`` (when it is declared as the heartbeat of the viewer). For
    every A snapshot, returns the index of the B row with the same counter (-1 if absent), and the clock
    offset ``t_A - t_B`` measured from ``raw/<viewer>/Stats/t_wall`` (includes the LECO latency, ~ms).

    Note: the counter restarts at 1 when the station restarts; the join is made within one station file
    and the clock offset is used to remove the ambiguity (closest time wins).
    """
    import h5py
    with h5py.File(laser_control_file, 'r') as f:
        snaps = f['snapshots']
        t_a = snaps['t'][:]
        values = snaps['values']
        names = {unescape(k): k for k in values.keys()}

        def viewer_of(n: str) -> str:
            parts = n.split('/')
            return parts[1] if n.startswith('det/') else '/'.join(parts[1:-2] if n.startswith('raw/') else parts[:-2])

        fc_keys = [n for n in names if n.endswith('/Stats/frame_counter') or
                   (n.startswith('det/') and n.endswith('/heartbeat'))]
        fc_keys = [n for n in fc_keys if viewer is None or viewer_of(n) == viewer]
        fc_keys.sort(key=lambda n: not n.endswith('/Stats/frame_counter'))   # the raw channel first
        if not fc_keys:
            raise KeyError('no "raw/<viewer>/Stats/frame_counter" nor "det/<viewer>/heartbeat" column in the '
                           'Laser Control file: is the Camera Station viewer recorded there?')
        fc_key = fc_keys[0]
        prefix = viewer_of(fc_key)
        counter_a = values[names[fc_key]][:]
        tw = [n for n in names if n.endswith(f'{prefix}/Stats/t_wall')]
        t_wall_b = values[names[tw[0]]][:] if tw else np.full_like(t_a, np.nan)
    b = load_session(station_file)
    t_b = b['t']
    counter_b = b['frame_counter']
    offset = t_a - t_wall_b
    med = float(np.nanmedian(offset)) if np.any(np.isfinite(offset)) else 0.0
    index = np.full(t_a.shape, -1, dtype=np.int64)
    by_counter: Dict[int, list] = {}
    for i, c in enumerate(counter_b):
        by_counter.setdefault(int(c), []).append(i)
    for k, c in enumerate(counter_a):
        if not np.isfinite(c):
            continue
        rows = by_counter.get(int(c))
        if not rows:
            continue
        if len(rows) == 1:
            index[k] = rows[0]
        else:                                    # station restarted inside the file: closest in time
            index[k] = min(rows, key=lambda i: abs((t_b[i] + med) - t_a[k]))
    return {'viewer': prefix, 't_a': t_a, 'counter': counter_a, 'b_index': index, 'clock_offset': offset,
            'clock_offset_median': med, 'station': b}
