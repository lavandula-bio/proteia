# SPDX-License-Identifier: Apache-2.0
"""Tests for pixel-level box integration."""

import numpy as np
import pytest

from proteia.core.model import Box, BoxSize
from proteia.core.quantify import integrate_box, to_grayscale


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
