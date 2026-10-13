"""Shape of the spot: orientation, ellipticity and asymmetry (2D moments).

Disk camera (with a reference): computed on the depletion, so it describes the two overlapping passes
together: a growing ellipticity / a tilt / an asymmetry shows the passes separating even when they
cannot be resolved. Other cameras: on the ROI with the border median subtracted.

angle: orientation of the major axis (degrees, 0 = along x, + = towards +y)
ellipticity: minor / major rms size (1 = round)
skew_x, skew_y: asymmetry of the x / y profiles (0 = symmetric)
"""
import math

import numpy as np

LABELS = ('angle', 'ellipticity', 'skew_x', 'skew_y')
PARAMS = {'threshold': 0.1}


def analyze(frame, ctx):
    w = ctx.depletion(frame) if ctx.kind == 'disk' else None
    if w is None:
        w = ctx.crop(frame, background=True)
    np.maximum(w, 0.0, out=w)
    top = float(w.max())
    if top <= 0:
        return {'valid': 0}
    w[w < ctx.params['threshold'] * top] = 0.0
    s = float(w.sum())
    y = np.arange(w.shape[0], dtype=np.float64)[:, None]
    x = np.arange(w.shape[1], dtype=np.float64)[None, :]
    px, py = w.sum(axis=0), w.sum(axis=1)
    mx, my = float(px @ x[0]) / s, float(py @ y[:, 0]) / s
    dx, dy = x[0] - mx, y[:, 0] - my
    mu20, mu02 = float(px @ dx ** 2) / s, float(py @ dy ** 2) / s
    mu11 = float(((x - mx) * (y - my) * w).sum()) / s          # the only 2D sum
    mu30, mu03 = float(px @ dx ** 3) / s, float(py @ dy ** 3) / s
    common = math.sqrt(max(((mu20 - mu02) / 2) ** 2 + mu11 ** 2, 0.0))
    l1, l2 = (mu20 + mu02) / 2 + common, (mu20 + mu02) / 2 - common
    ctx.show(w)
    return {
        'angle': math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02)),
        'ellipticity': math.sqrt(l2 / l1) if l1 > 0 and l2 > 0 else math.nan,
        'skew_x': mu30 / mu20 ** 1.5 if mu20 > 0 else math.nan,
        'skew_y': mu03 / mu02 ** 1.5 if mu02 > 0 else math.nan,
    }
