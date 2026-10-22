"""Image analyses of one camera (pure numpy, no Qt, no PyMoDAQ).

Cost model (the camera computer is slow): every analysis works on the ROI only, after an optional
binning, in float32, and the moments are computed from the two 1D projections (row / column sums)
instead of 2D coordinate grids. A 1 Mpx frame with binning 2 costs a few ms.

Coordinates and sizes are always returned in **full-frame pixels** of the camera, whatever the ROI
and the binning, so changing the ROI or the binning never shifts a published centroid.

Published values never fake a measurement: an image without signal, a missing reference... gives
NaN with ``valid = 0`` (same rule as S12 of Laser Control: NaN = invalid, never 0.0).

Kinds of analysis (one per camera):

* ``beam``  centroid, rms sizes (sigma), integrated signal, peak, saturated fraction;
* ``disk``  fluorescence depletion of the disk camera: ``reference - image`` (positive part) inside an
  optional circular mask, then the same moments plus the marginal kurtosis ``kx``, ``ky``;
* ``sum``   ROI intensity only (plasma / gas jet camera...).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

KIND_BEAM = 'beam'
KIND_DISK = 'disk'
KIND_SUM = 'sum'

KINDS = {
    KIND_BEAM: 'Beam: centroid and size',
    KIND_DISK: 'Disk: depletion (reference - image)',
    KIND_SUM: 'ROI intensity only',
}

# Published labels per kind. They never depend on the other settings: computer A refers to them in its
# roles ("<viewer>/<camera name>/<label>") and a key that appears / disappears breaks a role.
LABELS: Dict[str, Tuple[str, ...]] = {
    KIND_BEAM: ('x', 'y', 'sx', 'sy', 'sum', 'peak', 'sat', 'valid'),
    KIND_DISK: ('x', 'y', 'sx', 'sy', 'sum', 'kx', 'ky', 'peak', 'sat', 'valid'),
    KIND_SUM: ('sum', 'mean', 'peak', 'sat', 'valid'),
}

UNITS = {'x': 'px', 'y': 'px', 'sx': 'px', 'sy': 'px', 'sat': 'fraction'}

BACKGROUNDS = ('none', 'border', 'dark')

# A beam / depletion is accepted when the mean of one block of NOISE_BLOCK x NOISE_BLOCK (binned)
# pixels stands out of the others by NOISE_FACTOR times the noise of such a mean. Noise alone reaches
# about 4 over a few thousand blocks; a spot of the size of a block is seen from ~1 sigma per pixel.
NOISE_BLOCK = 8
NOISE_FACTOR = 6.0
# A real camera without beam is not white noise: vignetting (a dome) or bands of the sensor vary the block
# means 4 to 8 times more than the pixel noise says (an 8-bit sensor at 5 counts even gives a pixel noise of
# ~0, see computer B, 2026-10-22). The brightest block must also stand out of the spread of the background
# blocks (robust sigma of the blocks below half the contrast) by NOISE_FACTOR. In a small ROI / mask
# (fewer than SMALL_ROI_BLOCKS blocks) the spot covers most blocks and leaves no background to measure, while a
# dome or bands hardly vary over it: there the pixel noise test alone applies, as before.
SMALL_ROI_BLOCKS = 100


def nan_result(kind: str) -> Dict[str, float]:
    out = {k: math.nan for k in LABELS.get(kind, LABELS[KIND_BEAM])}
    out['valid'] = 0.0
    return out


def as_2d(image) -> np.ndarray:
    """Camera frame -> 2D array without copying when possible (colour frames are averaged)."""
    a = np.asarray(image)
    if a.ndim == 3:
        a = a.mean(axis=2, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f'image must be 2D, got shape {a.shape}')
    return a


def clip_roi(roi: Sequence[int], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """(x0, y0, w, h) clipped inside the frame; w or h <= 0 means "up to the edge"."""
    H, W = int(shape[0]), int(shape[1])
    x0, y0, w, h = (int(v) for v in (list(roi) + [0, 0, 0, 0])[:4])
    x0 = min(max(x0, 0), max(W - 1, 0))
    y0 = min(max(y0, 0), max(H - 1, 0))
    w = W - x0 if w <= 0 else min(w, W - x0)
    h = H - y0 if h <= 0 else min(h, H - y0)
    return x0, y0, max(w, 1), max(h, 1)


def bin_sum(a: np.ndarray, b: int) -> np.ndarray:
    """b x b binning (sum) to float32; the incomplete last row / column is dropped.

    Sum of the b*b strided sub-images, in place: ~4x faster than ``reshape(...).sum(axis=(1, 3))``
    (3 ms instead of 14 ms for a 1.3 Mpx frame at b = 2), which matters on the lab computers."""
    if b <= 1:
        return a.astype(np.float32, copy=True)
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    if h == 0 or w == 0:
        return a.astype(np.float32, copy=True)
    a = a[:h, :w]
    acc = a[0::b, 0::b].astype(np.float32)
    for i in range(b):
        for j in range(b):
            if i or j:
                np.add(acc, a[i::b, j::b], out=acc, casting='unsafe')
    return acc


def default_saturation(dtype) -> float:
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        return float(np.iinfo(dt).max)
    return math.inf


def border_level(a: np.ndarray) -> float:
    """Median of the 1-pixel border: a cheap, robust background estimate when the beam is inside."""
    if a.shape[0] < 3 or a.shape[1] < 3:
        return float(np.median(a))
    edge = np.concatenate((a[0, :], a[-1, :], a[1:-1, 0], a[1:-1, -1]))
    return float(np.median(edge))


def projection_moments(w: np.ndarray, kurtosis: bool = False) -> Optional[Dict[str, float]]:
    """Centroid, variances (and marginal kurtosis) of a non-negative weight image, in binned pixels.

    Uses the row / column projections only: O(N) and no coordinate grid. The marginal kurtosis
    mu40 / mu20^2 depends on the x projection only, so it is exact."""
    px = w.sum(axis=0, dtype=np.float64)
    py = w.sum(axis=1, dtype=np.float64)
    s = float(px.sum())
    if not (s > 0.0) or not math.isfinite(s):
        return None
    xs = np.arange(px.size, dtype=np.float64)
    ys = np.arange(py.size, dtype=np.float64)
    mx = float(px @ xs) / s
    my = float(py @ ys) / s
    dx = xs - mx
    dy = ys - my
    dx2 = dx * dx
    dy2 = dy * dy
    vx = float(px @ dx2) / s
    vy = float(py @ dy2) / s
    out = {'mx': mx, 'my': my, 'vx': max(vx, 0.0), 'vy': max(vy, 0.0), 'sum': s}
    if kurtosis:
        out['kx'] = float(px @ (dx2 * dx2)) / s / (vx * vx) if vx > 0 else math.nan
        out['ky'] = float(py @ (dy2 * dy2)) / s / (vy * vy) if vy > 0 else math.nan
    return out


def apply_threshold(w: np.ndarray, fraction: float) -> float:
    """In place: negative values and values below ``fraction`` of the maximum are set to 0.
    Returns the maximum (before thresholding)."""
    np.maximum(w, 0.0, out=w)
    peak = float(w.max()) if w.size else 0.0
    if peak > 0 and fraction > 0:
        w[w < fraction * peak] = 0.0
    return peak


def noise_level(a: np.ndarray, max_rows: int = 64, max_cols: int = 512) -> float:
    """Pixel noise (sigma) from the differences of neighbouring pixels, on a subset of the image.

    A beam is smooth at the pixel scale, so it hardly changes the median of the differences: the
    estimate stays right when the beam fills the ROI, which a spread of the values would not."""
    if a.shape[0] < 1 or a.shape[1] < 2:
        return 0.0
    rows = a[::max(a.shape[0] // max_rows, 1)]
    d = np.abs(rows[:, 1:] - rows[:, :-1])[:, ::max(a.shape[1] // max_cols, 1)]
    return 1.0484 * float(np.median(d))             # 1.4826 (MAD -> sigma) / sqrt(2) (difference)


def block_mean(a: np.ndarray, n: int) -> np.ndarray:
    """Mean of the n x n blocks (the incomplete last row / column is dropped)."""
    if n <= 1:
        return a
    return bin_sum(a, n) / np.float32(n * n)


class CameraAnalyzer:
    """Analysis of one camera with its caches (prepared dark / reference, mask).

    ``settings`` is a ``CameraConfig`` (core.settings) or any object with the same attributes.
    ``analyze`` never keeps a reference to the frame and never modifies it.
    """

    def __init__(self, settings):
        self.settings = settings
        self.reference: Optional[np.ndarray] = None    # full frame, as captured
        self.dark: Optional[np.ndarray] = None         # full frame, as captured
        self._ref_cache: Tuple[tuple, Optional[np.ndarray]] = ((), None)
        self._dark_cache: Tuple[tuple, Optional[np.ndarray]] = ((), None)
        self._mask_cache: Tuple[tuple, Optional[np.ndarray]] = ((), None)
        self._inside_cache: Tuple[tuple, Optional[np.ndarray]] = ((), None)
        self.last_reason = ''
        self.last_depletion: Optional[np.ndarray] = None   # binned ROI, for the preview (disk only)

    # ------------------------------------------------------------------ references
    def set_reference(self, image: Optional[np.ndarray]):
        self.reference = None if image is None else np.asarray(image, dtype=np.float32)
        self._ref_cache = ((), None)

    def set_dark(self, image: Optional[np.ndarray]):
        self.dark = None if image is None else np.asarray(image, dtype=np.float32)
        self._dark_cache = ((), None)

    def _prepared(self, which: str, roi, b) -> Optional[np.ndarray]:
        full = self.reference if which == 'ref' else self.dark
        if full is None:
            return None
        key = (id(full), roi, b)
        cache = self._ref_cache if which == 'ref' else self._dark_cache
        if cache[0] == key:
            return cache[1]
        x0, y0, w, h = roi
        arr = bin_sum(full[y0:y0 + h, x0:x0 + w], b)
        if which == 'ref':
            self._ref_cache = (key, arr)
        else:
            self._dark_cache = (key, arr)
        return arr

    def _mask(self, shape, roi, b) -> Optional[np.ndarray]:
        s = self.settings
        r = float(getattr(s, 'mask_radius', 0) or 0)
        if r <= 0:
            return None
        cx, cy = float(s.mask_cx), float(s.mask_cy)
        key = (shape, roi, b, cx, cy, r)
        if self._mask_cache[0] == key:
            return self._mask_cache[1]
        x0, y0 = roi[0], roi[1]
        # centres of the binned pixels in full-frame coordinates
        xs = x0 + (np.arange(shape[1]) + 0.5) * b - 0.5
        ys = y0 + (np.arange(shape[0]) + 0.5) * b - 0.5
        m = ((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2) <= r * r
        self._mask_cache = (key, m)
        return m

    def _contrast(self, wgt: np.ndarray, mask: Optional[np.ndarray], n: int) -> Tuple[float, float, int]:
        """(contrast, background spread, number of blocks). Contrast: highest block mean above the median block
        (blocks inside the mask), a local signal. Spread: robust sigma (1.4826 MAD) of the blocks below half the
        contrast, i.e. without the spot; inf when fewer than 8 such blocks.

        The median removes a uniform offset. A fluorescence that changed since the reference keeps
        the shape of the disk, so it still gives a contrast: that case is for ``min_signal``."""
        blocks = block_mean(wgt, n)
        if mask is not None:
            key = (self._mask_cache[0], n)
            if self._inside_cache[0] != key:
                cover = block_mean(mask.astype(np.float32), n)
                inside = cover >= 0.999
                self._inside_cache = (key, inside if inside.any() else cover > 0)
            blocks = blocks[self._inside_cache[1]]
        if blocks.size == 0:
            return 0.0, math.inf, 0
        med = float(np.median(blocks))
        contrast = float(blocks.max()) - med
        rest = blocks[blocks <= med + 0.5 * contrast]
        spread = 1.4826 * float(np.median(np.abs(rest - np.median(rest)))) if rest.size >= 8 else math.inf
        return contrast, spread, int(blocks.size)

    # ------------------------------------------------------------------ analysis
    def analyze(self, image) -> Dict[str, float]:
        s = self.settings
        kind = s.kind if s.kind in LABELS else KIND_BEAM
        raw = as_2d(image)
        roi = clip_roi(s.roi, raw.shape)
        x0, y0, w, h = roi
        b = max(int(s.binning), 1)
        sub = raw[y0:y0 + h, x0:x0 + w]

        sat_level = float(s.saturation) if s.saturation and s.saturation > 0 else default_saturation(raw.dtype)
        peak_raw = float(sub.max())
        sat = 0.0                                   # counting pass only when something is saturated
        if math.isfinite(sat_level) and peak_raw >= sat_level:
            sat = float(np.count_nonzero(sub >= sat_level)) / sub.size

        img = bin_sum(sub, b)
        out = nan_result(kind)
        out['peak'] = peak_raw
        out['sat'] = sat
        self.last_depletion = None

        if kind == KIND_DISK:
            ref = self._prepared('ref', roi, b)
            if ref is None:
                self.last_reason = 'no reference image'
                return out
            if ref.shape != img.shape:
                self.last_reason = f'reference {self.reference.shape} does not match the frame {raw.shape}'
                return out
            wgt = ref - img
        else:
            wgt = img
            if s.background == 'dark':
                dark = self._prepared('dark', roi, b)
                if dark is None or dark.shape != img.shape:
                    self.last_reason = 'no dark frame' if dark is None else 'dark frame does not match the frame'
                    return out
                wgt -= dark
            elif s.background == 'border':
                wgt -= border_level(wgt)

        mask = self._mask(wgt.shape, roi, b) if kind == KIND_DISK else None
        noise = 0.0
        if kind != KIND_SUM:            # before the mask: its edge and its zeros are not noise
            noise = noise_level(wgt)
            if np.issubdtype(raw.dtype, np.integer):
                noise = max(noise, 0.5 * b)         # digitisation: never trust a noise of 0
        if mask is not None:
            wgt *= mask

        if kind == KIND_SUM:            # no clipping: the noise around the background averages out
            total = float(wgt.sum(dtype=np.float64))
            out['sum'] = total
            out['mean'] = total / float(sub.size)
            out['valid'] = 1.0 if math.isfinite(total) else 0.0
            self.last_reason = ''
            return out

        # Without this test an image of noise alone (laser off) gives a "valid" centroid in the
        # middle of the ROI: its positive half survives the threshold below.
        n = max(1, min(NOISE_BLOCK, min(wgt.shape) // 4))
        contrast, spread, n_blocks = self._contrast(wgt, mask, n)
        found = contrast >= NOISE_FACTOR * noise / n and (
            n_blocks < SMALL_ROI_BLOCKS or contrast >= NOISE_FACTOR * spread)
        top = apply_threshold(wgt, float(s.threshold))
        if kind == KIND_DISK:
            self.last_depletion = wgt
        per_pixel = top / float(b * b)
        if not (top > 0) or not found or per_pixel < float(s.min_signal):
            self.last_reason = 'no signal' if not (top > 0) else \
                'no signal above the noise' if not found else \
                f'signal {per_pixel:.3g} below the minimum {float(s.min_signal):.3g}'
            return out
        m = projection_moments(wgt, kurtosis=(kind == KIND_DISK))
        if m is None:
            self.last_reason = 'no signal'
            return out
        out['x'] = x0 + (m['mx'] + 0.5) * b - 0.5
        out['y'] = y0 + (m['my'] + 0.5) * b - 0.5
        out['sx'] = math.sqrt(m['vx']) * b
        out['sy'] = math.sqrt(m['vy']) * b
        out['sum'] = m['sum']
        if kind == KIND_DISK:
            out['kx'] = m['kx']
            out['ky'] = m['ky']
        out['valid'] = 1.0
        self.last_reason = ''
        return out
