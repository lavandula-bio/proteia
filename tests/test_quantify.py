# SPDX-License-Identifier: Apache-2.0
"""Tests for pixel-level box integration."""

import numpy as np
import pytest

from proteia.core.model import Box, BoxSize
from proteia.core.quantify import (
    estimate_background,
    integrate_box,
    net_signal,
    to_grayscale,
)


def test_integrate_sums_pixels_in_box():
    img = np.zeros((10, 10))
    img[2:6, 1:4] = 5.0  # 4 rows x 3 cols of value 5 -> sum 60
    raw = integrate_box(img, Box(x=1, y=2), BoxSize(width=3, height=4))
    assert raw == 60.0


def test_integrate_uniform_image():
    img = np.ones((20, 20))
    raw = integrate_box(img, Box(x=0, y=0), BoxSize(width=5, height=4))
    assert raw == 20.0  # 5 * 4 pixels of value 1


def test_to_grayscale_reduces_rgb():
    rgb = np.ones((4, 4, 3))
    gray = to_grayscale(rgb)
    assert gray.shape == (4, 4)
    assert gray[0, 0] == 1.0


def test_integrate_out_of_bounds_raises():
    img = np.ones((10, 10))
    with pytest.raises(ValueError, match="bounds"):
        integrate_box(img, Box(x=8, y=0), BoxSize(width=5, height=2))


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
