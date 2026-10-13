import math

import numpy as np
import pytest

from pymodaq_plugins_camerastation.core.analysis import (CameraAnalyzer, LABELS, bin_sum, clip_roi,
                                                         projection_moments)
from pymodaq_plugins_camerastation.core.settings import CameraConfig
from pymodaq_plugins_camerastation.core.sim import beam_frame, disk_frame

RNG = np.random.default_rng(1)


def cfg(**kw):
    return CameraConfig(**kw).validate()


@pytest.mark.parametrize('binning', [1, 2, 4])
def test_beam_centroid_and_size_in_full_frame_pixels(binning):
    img = beam_frame(x=301.3, y=211.7, sx=20.0, sy=12.0, noise=5.0, rng=RNG)
    a = CameraAnalyzer(cfg(kind='beam', binning=binning, background='border', threshold=0.02))
    r = a.analyze(img)
    assert r['valid'] == 1.0
    assert r['x'] == pytest.approx(301.3, abs=0.3)
    assert r['y'] == pytest.approx(211.7, abs=0.3)
    # binning adds (b^2 - 1) / 12 to the variance; the threshold removes a bit of the wings
    assert r['sx'] == pytest.approx(20.0, rel=0.08)
    assert r['sy'] == pytest.approx(12.0, rel=0.1)
    assert set(r) == set(LABELS['beam'])


def test_roi_does_not_shift_the_centroid():
    img = beam_frame(x=400.0, y=300.0, noise=0.0, rng=RNG)
    full = CameraAnalyzer(cfg(binning=1)).analyze(img)
    roi = CameraAnalyzer(cfg(binning=2, roi=[250, 150, 300, 300])).analyze(img)
    assert roi['x'] == pytest.approx(full['x'], abs=0.3)
    assert roi['y'] == pytest.approx(full['y'], abs=0.3)


def test_no_beam_is_nan_and_invalid_never_zero():
    img = np.full((200, 300), 100, dtype=np.uint16)
    a = CameraAnalyzer(cfg())
    r = a.analyze(img)
    assert r['valid'] == 0.0
    assert math.isnan(r['x']) and math.isnan(r['sum'])
    assert a.last_reason


def test_min_signal_rejects_noise():
    img = beam_frame(amp=30.0, noise=10.0, rng=RNG)
    r = CameraAnalyzer(cfg(min_signal=200.0)).analyze(img)
    assert r['valid'] == 0.0 and math.isnan(r['x'])
    assert CameraAnalyzer(cfg(min_signal=0.0)).analyze(beam_frame(rng=RNG))['valid'] == 1.0


@pytest.mark.parametrize('binning', [1, 2])
@pytest.mark.parametrize('background', ['border', 'dark'])
def test_beam_off_is_invalid_without_any_setting(binning, background):
    a = CameraAnalyzer(cfg(binning=binning, background=background))
    a.set_dark(np.full((480, 640), 100.0, dtype=np.float32))
    for _ in range(20):                                  # noise alone: never a centroid
        r = a.analyze(beam_frame(amp=0.0, noise=10.0, rng=RNG))
        assert r['valid'] == 0.0 and math.isnan(r['x'])
        assert a.last_reason == 'no signal above the noise'
    faint = a.analyze(beam_frame(x=200.0, y=150.0, amp=30.0, noise=10.0, rng=RNG))
    assert faint['valid'] == 1.0                         # 3 sigma per pixel, but 25 x 18 px wide


def test_beam_filling_the_roi_and_noiseless_frames_stay_valid():
    img = beam_frame(x=320.0, y=240.0, sx=25.0, sy=18.0, noise=10.0, rng=RNG)
    assert CameraAnalyzer(cfg(roi=[295, 222, 50, 36], binning=1)).analyze(img)['valid'] == 1.0
    assert CameraAnalyzer(cfg(roi=[300, 230, 12, 10], binning=1)).analyze(img)['valid'] == 1.0   # tiny ROI
    assert CameraAnalyzer(cfg()).analyze(beam_frame(amp=20.0, noise=0.0, rng=RNG))['valid'] == 1.0


@pytest.mark.parametrize('mask_radius', [0, 170, 60])
def test_disk_laser_off_is_invalid_without_any_setting(mask_radius):
    """Reference - frame without laser is noise: no spot (a fluorescence that changed since the
    reference is a real difference: that case needs ``min_signal``)."""
    ref = np.mean([disk_frame(laser_on=False, rng=RNG) for _ in range(3)], axis=0)
    a = CameraAnalyzer(cfg(kind='disk', binning=2, mask_cx=320, mask_cy=240, mask_radius=mask_radius))
    a.set_reference(ref)
    for _ in range(20):
        r = a.analyze(disk_frame(laser_on=False, rng=RNG))
        assert r['valid'] == 0.0 and math.isnan(r['x']) and math.isnan(r['sx'])
        assert a.last_reason == 'no signal above the noise'
    r = a.analyze(disk_frame(laser_on=True, depth=0.02, rng=RNG))    # 2 % depletion: 30 counts, noise 8
    assert r['valid'] == 1.0 and r['x'] == pytest.approx(320.0, abs=6)


def test_saturation_fraction_and_peak():
    img = beam_frame(amp=6000.0, noise=0.0, bits=12, rng=RNG)
    r = CameraAnalyzer(cfg(saturation=4095)).analyze(img)
    assert r['peak'] == 4095
    assert 0.0 < r['sat'] < 0.05
    r2 = CameraAnalyzer(cfg()).analyze(beam_frame(amp=1000.0, noise=0.0, rng=RNG))   # uint16 max = 65535
    assert r2['sat'] == 0.0


def test_dark_background():
    dark = np.full((480, 640), 300.0, dtype=np.float32)
    img = beam_frame(x=200.0, y=100.0, offset=300.0, noise=0.0, rng=RNG)
    a = CameraAnalyzer(cfg(background='dark', binning=1))
    assert a.analyze(img)['valid'] == 0.0 and a.last_reason == 'no dark frame'
    a.set_dark(dark)
    r = a.analyze(img)
    assert r['x'] == pytest.approx(200.0, abs=0.2) and r['y'] == pytest.approx(100.0, abs=0.2)


def test_disk_depletion_of_two_passes():
    ref = disk_frame(laser_on=False, noise=0.0, rng=RNG).astype(np.float32)
    img = disk_frame(spots=((290.0, 245.0), (350.0, 245.0)), laser_on=True, noise=4.0, rng=RNG)
    a = CameraAnalyzer(cfg(kind='disk', binning=2, threshold=0.1, mask_cx=320, mask_cy=240, mask_radius=170))
    r0 = a.analyze(img)
    assert r0['valid'] == 0.0 and a.last_reason == 'no reference image'
    a.set_reference(ref)
    r = a.analyze(img)
    assert r['valid'] == 1.0
    assert r['x'] == pytest.approx(320.0, abs=1.5)      # mean of the two passes
    assert r['y'] == pytest.approx(245.0, abs=1.5)
    assert r['sx'] > r['sy'] * 1.5                      # two spots side by side: wide in x
    # bimodal in x: ~1.7 for these two passes; single spot in y: < 3 only because the threshold cuts the wings
    assert r['kx'] < 2.0 < r['ky']
    assert a.last_depletion is not None


def test_disk_reference_shape_mismatch_is_reported():
    a = CameraAnalyzer(cfg(kind='disk'))
    a.set_reference(np.zeros((100, 100), np.float32))
    r = a.analyze(disk_frame(rng=RNG))
    assert r['valid'] == 0.0 and 'does not match' in a.last_reason


def test_sum_kind():
    img = np.full((100, 100), 10, dtype=np.uint16)
    img[40:60, 40:60] += 100
    r = CameraAnalyzer(cfg(kind='sum', background='none', binning=1)).analyze(img)
    assert r['valid'] == 1.0
    assert r['sum'] == pytest.approx(100 * 100 * 10 + 400 * 100)
    assert set(r) == set(LABELS['sum'])


def test_frame_is_not_modified():
    img = beam_frame(rng=RNG)
    before = img.copy()
    CameraAnalyzer(cfg(binning=1)).analyze(img)
    CameraAnalyzer(cfg(kind='sum', binning=1)).analyze(img)
    assert np.array_equal(img, before)


def test_helpers():
    assert clip_roi([0, 0, 0, 0], (10, 20)) == (0, 0, 20, 10)
    assert clip_roi([15, 5, 100, 100], (10, 20)) == (15, 5, 5, 5)
    a = np.arange(25, dtype=np.float32).reshape(5, 5)
    assert bin_sum(a, 2).shape == (2, 2)
    assert projection_moments(np.zeros((5, 5), np.float32)) is None


@pytest.mark.parametrize('b', [2, 3, 4])
def test_fast_binning_equals_reshape_sum(b):
    a = RNG.integers(0, 4096, size=(101, 203), dtype=np.uint16)
    h, w = (101 // b) * b, (203 // b) * b
    ref = a[:h, :w].reshape(h // b, b, w // b, b).sum(axis=(1, 3), dtype=np.float64)
    assert np.array_equal(bin_sum(a, b), ref.astype(np.float32))
    roi_view = a[10:90, 5:200]                          # non contiguous view (ROI)
    assert bin_sum(roi_view, b).shape == (80 // b, 195 // b)


def test_color_frame_is_accepted():
    img = np.dstack([beam_frame(noise=0.0, rng=RNG)] * 3)
    assert CameraAnalyzer(cfg()).analyze(img)['valid'] == 1.0


def test_cost_is_small_on_a_megapixel_frame():
    """Order of magnitude only: the lab computers are slower, the budget is ~10 ms at binning 2."""
    import time
    img = beam_frame(shape=(1024, 1280), x=600, y=500, rng=RNG)
    a = CameraAnalyzer(cfg(binning=2))
    a.analyze(img)
    t0 = time.perf_counter()
    for _ in range(10):
        a.analyze(img)
    dt = (time.perf_counter() - t0) / 10
    assert dt < 0.05, f'{dt * 1e3:.1f} ms per frame'
