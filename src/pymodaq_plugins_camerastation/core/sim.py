"""Synthetic camera frames (numpy only): used by the tests and by the CamSim 2D plugin (office).

* ``beam``: one Gaussian spot on an offset with noise;
* ``disk``: the fluorescence of the disk (a broad disc) depleted by the two laser passes (two
  Gaussian holes), i.e. what the disk camera shows: reference (no laser) - frame = the two spots.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def gaussian(shape, x, y, sx, sy, amp) -> np.ndarray:
    h, w = shape
    gx = np.exp(-0.5 * ((np.arange(w, dtype=np.float32) - x) / sx) ** 2)
    gy = np.exp(-0.5 * ((np.arange(h, dtype=np.float32) - y) / sy) ** 2)
    return amp * gy[:, None] * gx[None, :]          # separable: cheap even for large frames


def beam_frame(shape=(480, 640), x=320.0, y=240.0, sx=25.0, sy=18.0, amp=2000.0, offset=100.0,
               noise=10.0, bits=12, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    img = gaussian(shape, x, y, sx, sy, amp) + offset
    if noise > 0:
        img = img + rng.normal(0.0, noise, shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 2 ** bits - 1).astype(np.uint16)


def disk_fluorescence(shape=(480, 640), cx=320.0, cy=240.0, radius=180.0, level=1500.0, offset=100.0):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (radius * radius)
    return (offset + level * np.clip(1.0 - r2, 0.0, None) ** 0.5).astype(np.float32)


def disk_frame(shape=(480, 640), spots=((300.0, 240.0), (340.0, 240.0)), sigma=15.0, depth=0.3,
               laser_on=True, noise=8.0, bits=12, rng: Optional[np.random.Generator] = None,
               **disk_kw) -> np.ndarray:
    rng = rng or np.random.default_rng()
    fl = disk_fluorescence(shape, **disk_kw)
    if laser_on:
        hole = np.zeros(shape, dtype=np.float32)
        for (x, y) in spots:
            hole += gaussian(shape, x, y, sigma, sigma, 1.0)
        fl = fl * (1.0 - depth * np.clip(hole, 0.0, 1.0))
    if noise > 0:
        fl = fl + rng.normal(0.0, noise, shape).astype(np.float32)
    return np.clip(np.rint(fl), 0, 2 ** bits - 1).astype(np.uint16)
