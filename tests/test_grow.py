# SPDX-License-Identifier: Apache-2.0
"""Tests for region-growing a box from a seed (GUI-independent)."""

import numpy as np

from proteia.core.grow import grow_box

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
