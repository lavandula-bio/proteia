# SPDX-License-Identifier: Apache-2.0
"""Tests for the GUI-independent box constraints: locked size, equal area,
no overlap (the validate-and-correct rules)."""

from proteia.core.boxes import (
    normalize_corners,
    resize_all,
)
from proteia.core.model import BoxSize

SIZE = BoxSize(width=10, height=4)


def test_normalize_corners_orders_and_rounds():
    # napari vertices are (y, x) and may be in any order.
    corners = [(5.4, 20.6), (5.4, 10.2), (8.7, 10.2), (8.7, 20.6)]
    assert normalize_corners(corners) == (10, 5, 21, 9)


def test_resize_all_grows_around_each_center():
    # Boxes centred at (15, 12) and (55, 12); growing keeps the centres.
    rects = [(10, 10, 20, 14), (50, 10, 60, 14)]
    bigger = BoxSize(width=20, height=8)
    assert resize_all(rects, bigger) == [(5, 8, 25, 16), (45, 8, 65, 16)]


def test_resize_all_can_shrink_around_center():
    rects = [(10, 10, 20, 14)]  # centre (15, 12)
    smaller = BoxSize(width=4, height=2)
    assert resize_all(rects, smaller) == [(13, 11, 17, 13)]


def test_resize_all_clamps_to_image_bounds():
    rects = [(0, 0, 10, 4)]  # centre (5, 2); growing would go negative
    bigger = BoxSize(width=20, height=8)
    assert resize_all(rects, bigger, width=100, height=100) == [(0, 0, 20, 8)]


def test_resize_all_rejects_when_it_forces_overlap():
    # Two boxes 15px apart on x; growing width to 20 would overlap them.
    rects = [(0, 0, 10, 4), (15, 0, 25, 4)]
    bigger = BoxSize(width=20, height=4)
    assert resize_all(rects, bigger) is None
