# SPDX-License-Identifier: Apache-2.0
"""Tests for pixel-level box integration."""

import numpy as np
import pytest

from proteia.core.model import Box, BoxSize
from proteia.core.quantify import (
    clipped_pixels,
    detector_limit,
    estimate_background,
    is_clipped,
    net_signal,
    to_grayscale,
)


def test_to_grayscale_reduces_rgb():
    rgb = np.ones((4, 4, 3))
    gray = to_grayscale(rgb)
    assert gray.shape == (4, 4)
    assert gray[0, 0] == 1.0


def test_to_grayscale_ignores_alpha_and_keeps_gray_of_gray_alpha():
    rgba = np.zeros((2, 2, 4))
    rgba[..., :3] = [30.0, 60.0, 90.0]
    rgba[..., 3] = 255.0
    assert to_grayscale(rgba)[0, 0] == 60.0  # (30 + 60 + 90) / 3; alpha ignored
    la = np.zeros((2, 2, 2))
    la[..., 0], la[..., 1] = 42.0, 255.0
    assert to_grayscale(la)[0, 0] == 42.0


def test_to_grayscale_refuses_other_layouts():
    with pytest.raises(ValueError, match="unsupported image layout"):
        to_grayscale(np.zeros((2, 2, 5)))


def test_estimate_background_is_membrane_median():
    img = np.full((10, 10), 200.0)  # uniform "membrane"
    img[4:6, 4:6] = 50.0  # a small dark band
    assert estimate_background(img) == 200.0  # band is a minority -> median = membrane


def test_net_signal_dark_band_above_background():
    img = np.full((10, 10), 200.0)
    img[2:6, 1:4] = 150.0  # 4x3 darker band, 50 below background
    net = net_signal(img, Box(x=1, y=2), BoxSize(width=3, height=4), background=200.0)
    assert net == 600.0  # 12 pixels * (200 - 150)


def test_net_signal_blank_box_is_zero():
    img = np.full((10, 10), 200.0)
    net = net_signal(img, Box(x=0, y=0), BoxSize(width=5, height=5), background=200.0)
    assert net == 0.0  # nothing darker than background


def test_net_signal_darker_band_reads_larger():
    img = np.full((12, 12), 220.0)
    img[2:5, 1:4] = 180.0  # faint band
    img[2:5, 6:9] = 100.0  # strong band
    size = BoxSize(width=3, height=3)
    bg = 220.0
    faint = net_signal(img, Box(x=1, y=2), size, bg)
    strong = net_signal(img, Box(x=6, y=2), size, bg)
    assert strong > faint > 0


def test_net_signal_light_on_dark_flips_direction():
    img = np.full((10, 10), 30.0)  # dark membrane
    img[2:5, 1:4] = 90.0  # bright band, 60 above background
    net = net_signal(
        img, Box(x=1, y=2), BoxSize(width=3, height=3), background=30.0, dark_on_light=False
    )
    assert net == 540.0  # 9 pixels * (90 - 30)


def test_net_signal_out_of_bounds_raises():
    img = np.ones((10, 10))
    with pytest.raises(ValueError, match="bounds"):
        net_signal(img, Box(x=8, y=0), BoxSize(width=5, height=2), background=0.5)


# --- over-exposure (clipping) ---


def test_detector_limit_follows_bit_depth_and_polarity():
    assert detector_limit(16, dark_on_light=False) == 65535
    assert detector_limit(8, dark_on_light=False) == 255
    assert detector_limit(16, dark_on_light=True) == 0


def test_a_band_at_the_detector_limit_is_clipped():
    # Light on dark, 16-bit: two pixels of the band hit 65535.
    img = np.full((10, 10), 1000.0)
    img[4:6, 4:7] = 40000.0
    img[5, 5] = img[4, 5] = 65535.0
    box, size = Box(x=3, y=3), BoxSize(width=5, height=4)
    assert clipped_pixels(img, box, size, bit_depth=16, dark_on_light=False) == 2
    assert is_clipped(img, box, size, bit_depth=16, dark_on_light=False)
    img[5, 5] = img[4, 5] = 65534.0  # one level below the limit: not clipped
    assert not is_clipped(img, box, size, bit_depth=16, dark_on_light=False)


def test_a_dark_band_pinned_at_zero_is_clipped():
    # Dark on light, 8-bit: saturation shows as black.
    img = np.full((10, 10), 200.0)
    img[4:6, 4:7] = 0.0
    box, size = Box(x=3, y=3), BoxSize(width=5, height=4)
    assert is_clipped(img, box, size, bit_depth=8, dark_on_light=True)
    assert not is_clipped(img, box, size, bit_depth=8, dark_on_light=False)  # 255 never reached
    assert not is_clipped(
        img, Box(x=0, y=0), BoxSize(width=3, height=3), bit_depth=8, dark_on_light=True
    )


def test_is_clipped_without_a_trusted_limit_is_not_checked():
    img = np.zeros((4, 4))
    assert (
        is_clipped(
            img, Box(x=0, y=0), BoxSize(width=2, height=2), bit_depth=None, dark_on_light=True
        )
        is None
    )
