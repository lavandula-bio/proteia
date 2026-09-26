# SPDX-License-Identifier: Apache-2.0
"""Tests for region-growing a box from a seed (GUI-independent)."""

import itertools
import math

import numpy as np
import pytest

from proteia.core.grow import grow_box, grow_region

BG = 200.0


def _light_image_with_band() -> np.ndarray:
    """Light membrane (200) with one darker band (100) at x[2:5], y[3:7]."""
    img = np.full((10, 10), BG)
    img[3:7, 2:5] = 100.0
    return img


def test_grow_fits_the_band():
    img = _light_image_with_band()
    assert grow_box(img, (3, 4), BG) == (2, 3, 5, 7)


def test_seed_on_background_returns_none():
    img = _light_image_with_band()
    assert grow_box(img, (8, 1), BG) is None  # blank membrane, no signal


def test_grow_stops_at_background_between_two_bands():
    # Two bands separated by a background gap; growing one must not reach the other.
    img = np.full((10, 12), BG)
    img[3:6, 1:4] = 100.0  # band A
    img[3:6, 8:11] = 100.0  # band B
    box = grow_box(img, (2, 4), BG)  # seed in A
    assert box == (1, 3, 4, 6)  # only A, not B


def test_max_width_caps_a_wide_blob():
    img = np.full((10, 20), BG)
    img[3:6, 1:19] = 100.0  # very wide smear
    box = grow_box(img, (10, 4), BG, max_width=6)
    x0, _, x1, _ = box
    assert x1 - x0 == 6  # capped


def test_light_on_dark_grows_bright_band():
    img = np.full((10, 10), 30.0)  # dark membrane
    img[3:7, 2:5] = 220.0  # bright band
    assert grow_box(img, (3, 4), 30.0, dark_on_light=False) == (2, 3, 5, 7)


def _noisy_membrane() -> np.ndarray:
    """Membrane at 200 with a deterministic ±4 checkerboard texture."""
    img = np.full((12, 12), 200.0)
    img[::2, ::2] = 196.0
    img[1::2, 1::2] = 204.0
    return img


def test_noise_floor_prevents_flood_into_texture():
    # A strong band on a textured membrane fits the band, not the whole frame.
    img = _noisy_membrane()
    img[4:8, 3:6] = 100.0
    assert grow_box(img, (4, 5), 200.0) == (3, 4, 6, 8)


def test_seed_on_textured_membrane_returns_none():
    # Texture alone is below the noise floor -> no band to grow.
    assert grow_box(_noisy_membrane(), (2, 2), 200.0) is None


# --- The growth rule on a caller's own signal (row-box detection) ---


def _two_blocks() -> np.ndarray:
    signal = np.zeros((8, 10))
    signal[2:5, 1:4] = 5.0
    signal[2:6, 6:9] = 3.0
    return signal


def test_grow_region_bounds_the_seed_component():
    assert grow_region(_two_blocks(), (2, 3), 1.0) == (1, 2, 4, 5)
    assert grow_region(_two_blocks(), (7, 5), 1.0) == (6, 2, 9, 6)
    # A higher threshold drops the weaker block, not the seed's.
    assert grow_region(_two_blocks(), (2, 3), 4.0) == (1, 2, 4, 5)


def test_grow_region_seed_not_above_threshold_is_none():
    assert grow_region(_two_blocks(), (2, 3), 5.0) is None  # equal is not above
    assert grow_region(_two_blocks(), (0, 0), 0.0) is None  # background


def test_a_nan_seed_grows_nothing():
    # A NaN seed has no signal above any threshold: no box, never the frame.
    img = _light_image_with_band()
    img[4, 3] = np.nan
    assert grow_box(img, (3, 4), BG) is None
    assert grow_box(img, (3, 4), BG, max_width=2, max_height=2) is None
    assert grow_region(np.full((5, 5), np.nan), (2, 2), 1.0) is None
    assert grow_region(_two_blocks(), (2, 3), np.nan) is None  # nothing is above NaN
    # A NaN elsewhere only drops out of the region.
    assert grow_box(img, (2, 3), BG) == (2, 3, 5, 7)


def test_grow_region_is_4_connected_with_plain_ints():
    signal = np.zeros((5, 5))
    signal[1, 1] = signal[2, 2] = 3.0  # diagonal neighbours only
    rect = grow_region(signal, (1, 1), 1.0)
    assert rect == (1, 1, 2, 2)
    assert all(type(v) is int for v in rect)


def test_mad_sigma_is_the_one_robust_noise_estimator():
    # 1.4826 times the median absolute deviation, from the median or from a
    # given centre: grow_box's membrane noise and rowdetect's pixel noise and
    # one-sided spreads all use it.
    from proteia.core import rowdetect
    from proteia.core.grow import mad_sigma

    rng = np.random.default_rng(58)
    values = rng.normal(0.0, 2.0, 5001)
    assert mad_sigma(values) == float(1.4826 * np.median(np.abs(values - np.median(values))))
    assert mad_sigma(np.abs(values), 0.0) == float(1.4826 * np.median(np.abs(values)))
    assert mad_sigma(values) == pytest.approx(2.0, rel=0.05)
    crop = np.round(rng.normal(1000.0, 50.0, (20, 60)))
    diffs = np.diff(crop, axis=1).ravel()
    assert rowdetect._pixel_noise(crop) == mad_sigma(diffs) / math.sqrt(2.0)


def test_grow_box_is_grow_region_on_its_signal_and_threshold():
    rng = np.random.default_rng(51)
    img = BG + rng.normal(0.0, 3.0, (30, 40))
    img[10:18, 5:15] -= 80.0
    img[12:20, 22:35] -= 50.0
    signal = np.maximum(BG - img, 0.0)
    noise = 1.4826 * np.median(np.abs(img - np.median(img)))
    for sy, sx in itertools.product(range(0, 30, 3), range(0, 40, 3)):
        threshold = max(signal[sy, sx] * 0.3, 3.0 * noise)
        assert grow_box(img, (sx, sy), BG) == grow_region(signal, (sx, sy), threshold)
