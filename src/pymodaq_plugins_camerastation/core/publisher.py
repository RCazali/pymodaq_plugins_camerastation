"""0D values sent to computer A at a fixed rate (pure Python).

One publication = ``{data name: {label: value}}``:

* one entry per enabled camera, named after the camera (``Cam1``), with the labels of its analysis
  (core.analysis.LABELS) plus ``age`` (s since the analysed frame) and ``frame`` (frame number of the
  camera, frozen = camera frozen);
* ``Stats``: ``frame_counter`` (+1 per publication: the heartbeat of option O9, also the
  synchronisation key with the recordings of both computers), ``t_wall`` (clock of computer B, so
  that computer A records the clock offset between the two computers), ``cams_ok``, ``recording``.

On computer A the keys become ``<LECO viewer title>/<data name>/<label>``, e.g. ``LECO_B/Cam1/x``.

A camera without a recent result (``stale_s``) publishes NaN and ``valid = 0``: computer A must never
act on a frozen value. Publishing at a fixed rate keeps the network load constant (a few hundred
bytes per publication) whatever the camera frame rates.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

STATS = 'Stats'
STATS_LABELS = ('frame_counter', 't_wall', 'cams_ok', 'recording')
EXTRA_LABELS = ('age', 'frame')

Payload = Dict[str, Dict[str, float]]


class Publisher:
    def __init__(self):
        self.counter = 0

    def build(self, cameras: Iterable[Tuple[str, Tuple[str, ...], Dict]], t_wall: float, stale_s: float,
              recording: bool = False, extra: Iterable[Tuple[str, Tuple[str, ...], Dict]] = ()) -> Payload:
        """``cameras``: (name, labels, engine snapshot) of each enabled camera; ``extra``: same for the
        user scripts (``<camera>_<script>``), same staleness rule, not counted in ``cams_ok``."""
        self.counter += 1
        out: Payload = {}
        n_ok = 0
        cameras = [(n, lab, s, True) for n, lab, s in cameras] + [(n, lab, s, False) for n, lab, s in extra]
        for name, labels, snap, is_camera in cameras:
            res = snap.get('result', {}) if snap else {}
            age = float(snap.get('age', math.inf)) if snap else math.inf
            fresh = age <= stale_s
            values = {}
            for lab in labels:
                v = res.get(lab, math.nan) if fresh else math.nan
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = math.nan
                values[lab] = v
            if 'valid' in values:
                values['valid'] = 1.0 if fresh and values['valid'] == 1.0 else 0.0
                n_ok += is_camera and values['valid'] == 1.0
            values['age'] = age if math.isfinite(age) else math.nan
            values['frame'] = float(snap.get('frame', -1)) if snap else -1.0
            out[name] = values
        out[STATS] = {
            'frame_counter': float(self.counter),
            't_wall': float(t_wall),
            'cams_ok': float(n_ok),
            'recording': 1.0 if recording else 0.0,
        }
        return out


def flatten(payload: Payload) -> Dict[str, float]:
    """``{'Cam1/x': ..., 'Stats/frame_counter': ...}`` (recording columns)."""
    return {f'{name}/{lab}': v for name, values in payload.items() for lab, v in values.items()}
