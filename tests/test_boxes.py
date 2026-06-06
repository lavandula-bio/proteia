# SPDX-License-Identifier: Apache-2.0
"""Tests for the GUI-independent box constraints: locked size, equal area,
no overlap (the validate-and-correct rules)."""

from proteia.core.boxes import (
    normalize_corners,
    reconcile,
    resize_all,
    snap_to_size,
)
from proteia.core.model import BoxSize

SIZE = BoxSize(width=10, height=4)


def test_normalize_corners_orders_and_rounds():
    # napari vertices are (y, x) and may be in any order.
    corners = [(5.4, 20.6), (5.4, 10.2), (8.7, 10.2), (8.7, 20.6)]
    assert normalize_corners(corners) == (10, 5, 21, 9)


def test_snap_keeps_top_left_and_forces_size():
    # A box the user dragged to 30x30; snap back to the locked 10x4.
    assert snap_to_size((5, 5, 35, 35), SIZE) == (5, 5, 15, 9)


def test_reconcile_snaps_resized_box_to_locked_size():
    prev = [(0, 0, 10, 4)]
    new = [(0, 0, 40, 40)]  # user resized it
    assert reconcile(prev, new, SIZE) == [(0, 0, 10, 4)]


def test_reconcile_keeps_a_valid_move():
    prev = [(0, 0, 10, 4)]
    new = [(50, 50, 60, 54)]  # moved, still locked size, no neighbour
    assert reconcile(prev, new, SIZE) == [(50, 50, 60, 54)]


def test_reconcile_reverts_move_that_overlaps():
    prev = [(0, 0, 10, 4), (50, 0, 60, 4)]
    new = [(0, 0, 10, 4), (5, 0, 15, 4)]  # box 1 moved on top of box 0
    # Box 1 is reverted to its previous position; box 0 untouched.
    assert reconcile(prev, new, SIZE) == [(0, 0, 10, 4), (50, 0, 60, 4)]


def test_reconcile_drops_new_box_that_overlaps():
    prev = [(0, 0, 10, 4)]
    new = [(0, 0, 10, 4), (5, 0, 15, 4)]  # newly drawn box overlaps the first
    assert reconcile(prev, new, SIZE) == [(0, 0, 10, 4)]


def test_reconcile_accepts_new_box_in_free_space():
    prev = [(0, 0, 10, 4)]
    new = [(0, 0, 10, 4), (50, 0, 70, 9)]  # new box, will be snapped to size
    assert reconcile(prev, new, SIZE) == [(0, 0, 10, 4), (50, 0, 60, 4)]


def test_reconcile_accepts_deletion():
    prev = [(0, 0, 10, 4), (50, 0, 60, 4)]
    new = [(0, 0, 10, 4)]  # one box deleted
    assert reconcile(prev, new, SIZE) == [(0, 0, 10, 4)]


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