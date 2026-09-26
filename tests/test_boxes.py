# SPDX-License-Identifier: Apache-2.0
"""Tests for the GUI-independent box constraints: locked size, equal area,
no overlap (the validate-and-correct rules)."""

import itertools

import numpy as np
import pytest

from proteia.core.boxes import (
    BoxRuleError,
    center_snap,
    centered_rect,
    grow_to_fit,
    initial_box_size,
    normalize_corners,
    overlaps_any,
    place_in_row,
    resize_all,
)
from proteia.core.model import BoxSize, overlaps

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


# --- placement rules lifted from the napari app ---

W, H = 100, 60  # image size


def test_centered_rect_centres_the_box():
    assert centered_rect(50, 30, SIZE, W, H) == (45, 28, 55, 32)
    odd = BoxSize(width=5, height=3)
    assert centered_rect(50, 30, odd, W, H) == (48, 29, 53, 32)


@pytest.mark.parametrize(
    ("cx", "cy", "rect"),
    [
        (1, 30, (0, 28, 10, 32)),  # left edge
        (99, 30, (90, 28, 100, 32)),  # right edge
        (50, 0, (45, 0, 55, 4)),  # top edge
        (50, 59, (45, 56, 55, 60)),  # bottom edge
        (0, 0, (0, 0, 10, 4)),  # corner
    ],
)
def test_centered_rect_clamps_at_every_edge(cx, cy, rect):
    assert centered_rect(cx, cy, SIZE, W, H) == rect


def test_center_snap_maps_a_same_size_rect_to_itself():
    for rect in [(0, 0, 10, 4), (45, 28, 55, 32), (90, 56, 100, 60), (13, 7, 23, 11)]:
        assert center_snap(rect, SIZE, W, H) == rect


def test_center_snap_recentres_a_resized_rect():
    # A dragged corner: the rect (40, 20)-(70, 40) is read by its centre (55, 30).
    assert center_snap((40, 20, 70, 40), SIZE, W, H) == (50, 28, 60, 32)
    # Past the edge, the box is clamped back inside.
    assert center_snap((95, 50, 110, 70), SIZE, W, H) == (90, 56, 100, 60)


def test_overlaps_any():
    rect = (10, 10, 20, 14)
    assert overlaps_any(rect, [(0, 0, 5, 5), (19, 13, 30, 20)])
    assert not overlaps_any(rect, [(20, 10, 30, 14), (10, 14, 20, 18)])  # edges only touch
    assert not overlaps_any(rect, [])


def test_initial_box_size_is_clamped_to_the_image():
    assert initial_box_size(340, 150) == BoxSize(width=42, height=12)
    assert initial_box_size(20, 20) == BoxSize(width=4, height=4)  # the 4 px floor
    assert initial_box_size(3, 2) == BoxSize(width=3, height=2)  # but never past the image


def test_box_rules_match_the_napari_app():
    # The app module imports headlessly (napari and Qt are imported inside launch).
    # #57 removes the app's private helpers, and these parity cases with them.
    from proteia.gui import app

    dims = [(4, 4), (7, 5), (40, 30), (340, 150), (1000, 800)]
    for iw, ih in dims:
        expected = app._initial_size(np.zeros((ih, iw)))
        assert initial_box_size(iw, ih) == expected, (iw, ih)

    sizes = [BoxSize(width=w, height=h) for w, h in [(1, 1), (4, 3), (10, 4), (25, 18)]]
    corners = [-5, 0, 3, 17, 48, 95, 120]
    for size, (x0, x1), (y0, y1) in itertools.product(
        sizes, itertools.combinations(corners, 2), itertools.combinations(corners, 2)
    ):
        rect = (x0, y0, x1, y1)
        assert center_snap(rect, size, W, H) == app._center_snap(rect, size, W, H), rect


# --- grow_to_fit: napari's seed-grow sizing ---


def test_grow_to_fit_first_box_takes_the_grown_size():
    size, resized, rect = grow_to_fit([], SIZE, (40, 20, 51, 27), width=W, height=H)
    assert size == BoxSize(width=11, height=7)  # the given size is replaced
    assert resized == []
    assert rect == (40, 20, 51, 27)


def test_grow_to_fit_size_is_at_least_two_pixels_and_fits_the_image():
    size, _, rect = grow_to_fit([], SIZE, (50, 30, 51, 31), width=W, height=H)
    assert size == BoxSize(width=2, height=2)
    assert rect == (49, 29, 51, 31)
    size, _, rect = grow_to_fit([], SIZE, (0, 0, 1, 1), width=1, height=1)
    assert size == BoxSize(width=1, height=1)
    assert rect == (0, 0, 1, 1)


def test_grow_to_fit_later_boxes_only_grow_the_size():
    rects = [(10, 10, 20, 14), (30, 40, 40, 44)]  # centres (15, 12) and (35, 42)
    # Narrower but taller than the shared size: width kept, height grown.
    size, resized, rect = grow_to_fit(rects, SIZE, (60, 9, 64, 17), width=W, height=H)
    assert size == BoxSize(width=10, height=8)
    assert resized == [(10, 8, 20, 16), (30, 38, 40, 46)]  # same centres, in input order
    assert rect == (57, 9, 67, 17)
    # A smaller band keeps the size and moves nothing.
    size, resized, rect = grow_to_fit(rects, SIZE, (70, 30, 72, 31), width=W, height=H)
    assert size == SIZE
    assert resized == rects
    assert rect == (66, 28, 76, 32)


def test_grow_to_fit_refuses_a_size_that_forces_an_overlap():
    rects = [(0, 0, 10, 4), (15, 0, 25, 4)]  # 5 px apart
    with pytest.raises(BoxRuleError) as info:
        grow_to_fit(rects, SIZE, (60, 30, 80, 34), width=W, height=H)  # 20 px wide
    assert info.value.code == "size_would_overlap"


def test_grow_to_fit_refuses_a_box_on_an_existing_one():
    rects = [(10, 10, 20, 14)]
    with pytest.raises(BoxRuleError) as info:
        grow_to_fit(rects, SIZE, (12, 10, 18, 14), width=W, height=H)
    assert info.value.code == "overlap"
    assert isinstance(info.value, ValueError)


# --- One row of same-size boxes (row-box detection) ---


def test_place_in_row_keeps_ordered_targets():
    assert place_in_row([0, 20, 45], 10, 0, 100) == [0, 20, 45]
    assert place_in_row([], 10, 0, 5) == []


def test_place_in_row_pushes_boxes_apart_about_their_mean():
    assert place_in_row([20, 24], 10, 0, 100) == [17, 27]  # each moves 3 px
    assert place_in_row([50, 50, 50], 10, 0, 200) == [40, 50, 60]
    # Only the colliding pair moves; the first box stays on its target.
    assert place_in_row([0, 40, 44], 10, 0, 100) == [0, 37, 47]


def test_place_in_row_rounds_a_pooled_half_up():
    assert place_in_row([21, 30], 10, 0, 100) == [21, 31]  # offsets 21, 20 pool to 20.5
    assert place_in_row([-3, 4], 10, -100, 100) == [-4, 6]  # -4.5 rounds up to -4


def test_place_in_row_stays_inside_the_bounds():
    assert place_in_row([-5, 3], 10, 0, 100) == [0, 10]
    assert place_in_row([95], 10, 0, 100) == [90]
    assert place_in_row([0, 0, 0], 10, 0, 30) == [0, 10, 20]  # exactly fits


def test_place_in_row_refuses_boxes_that_do_not_fit():
    with pytest.raises(ValueError, match="do not fit"):
        place_in_row([0, 0, 0], 10, 0, 29)


def test_place_in_row_takes_ints_only():
    assert place_in_row([np.int64(3)], 4, 0, 10) == [3]
    with pytest.raises(TypeError):
        place_in_row([1.5], 10, 0, 100)


def test_place_in_row_properties():
    rng = np.random.default_rng(51)
    for _ in range(300):
        m, w = int(rng.integers(1, 9)), int(rng.integers(1, 21))
        lo = int(rng.integers(-50, 51))
        hi = lo + m * w + int(rng.integers(0, 60))
        targets = [int(t) for t in rng.integers(lo - 30, hi + 30, m)]
        lefts = place_in_row(targets, w, lo, hi)
        assert all(type(x) is int for x in lefts)
        assert lo <= lefts[0] and lefts[-1] + w <= hi
        assert all(b - a >= w for a, b in itertools.pairwise(lefts))
        rects = [(x, int(rng.integers(0, 5)), x + w, 10) for x in lefts]  # any y
        assert not any(overlaps(a, b) for a, b in itertools.combinations(rects, 2))
        # Exact translation: shifting targets and bounds shifts the result.
        d = int(rng.integers(-40, 41))
        assert place_in_row([t + d for t in targets], w, lo + d, hi + d) == [x + d for x in lefts]
        # Targets that already fit come back unchanged.
        assert place_in_row(lefts, w, lo, hi) == lefts
