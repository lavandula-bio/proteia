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
    grow_to_fit_all,
    initial_box_size,
    overlaps_any,
    padding_words,
    place_in_row,
    resize_all,
    resize_checked,
    size_words,
)
from proteia.core.model import BoxPadding, BoxSize, overlaps

SIZE = BoxSize(width=10, height=4)


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


# --- placement rules ---

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


# --- grow_to_fit: seed-grow sizing ---


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


# --- grow_to_fit_all: several grown bands at once (row-box detection) ---


@pytest.mark.parametrize(
    ("rects", "grown"),
    [
        ([], (40, 20, 51, 27)),
        ([], (50, 30, 51, 31)),
        ([(10, 10, 20, 14), (30, 40, 40, 44)], (60, 9, 64, 17)),
        ([(10, 10, 20, 14), (30, 40, 40, 44)], (70, 30, 72, 31)),
    ],
)
def test_grow_to_fit_all_of_one_band_is_grow_to_fit(rects, grown):
    size, resized, rect = grow_to_fit(rects, SIZE, grown, width=W, height=H)
    assert grow_to_fit_all(rects, SIZE, [grown], width=W, height=H) == (size, resized, [rect])


def test_grow_to_fit_all_first_boxes_take_the_largest_grown_size():
    # 11x5 and 7x9 grown: the first boxes set the size to the larger of each,
    # and each box is centred on its own band, in input order.
    grown = [(10, 20, 21, 25), (40, 18, 47, 27)]
    size, resized, placed = grow_to_fit_all([], SIZE, grown, width=W, height=H)
    assert size == BoxSize(width=11, height=9)
    assert resized == []
    assert placed == [(10, 18, 21, 27), (38, 18, 49, 27)]


def test_grow_to_fit_all_takes_a_size_fitted_elsewhere():
    # A size the bands share (a row's detector fits one) stands in for their
    # own extents; it grows the existing boxes even with no band to place.
    rects = [(10, 10, 20, 14), (30, 40, 40, 44)]  # centres (15, 12) and (35, 42)
    need = BoxSize(width=8, height=8)
    size, resized, placed = grow_to_fit_all(rects, SIZE, [], need=need, width=W, height=H)
    assert size == BoxSize(width=10, height=8)
    assert resized == [(10, 8, 20, 16), (30, 38, 40, 46)]
    assert placed == []
    need = BoxSize(width=12, height=6)
    size, _, placed = grow_to_fit_all([], SIZE, [(60, 10, 64, 12)], need=need, width=W, height=H)
    assert (size, placed) == (need, [(56, 8, 68, 14)])


def test_grow_to_fit_all_needs_a_band_or_a_size():
    with pytest.raises(ValueError, match="grown band or a size") as info:
        grow_to_fit_all([(10, 10, 20, 14)], SIZE, [], width=W, height=H)
    assert not isinstance(info.value, BoxRuleError)


def test_grow_to_fit_all_refuses_existing_boxes_the_size_makes_overlap():
    rects = [(0, 0, 10, 4), (15, 0, 25, 4), (60, 30, 70, 34)]  # the first two 5 px apart
    with pytest.raises(BoxRuleError) as info:
        grow_to_fit_all(rects, SIZE, [(80, 10, 100, 14)], width=W, height=H)  # 20 px wide
    error = info.value
    assert (error.code, error.size, error.hits) == (
        "size_would_overlap",
        BoxSize(width=20, height=4),
        (0, 1),
    )


def test_grow_to_fit_all_refuses_new_boxes_the_size_makes_overlap():
    # The existing box sets no overlap; the two new ones overlap each other at
    # the grown size: no existing box is named.
    grown = [(10, 10, 30, 14), (25, 10, 45, 14)]
    with pytest.raises(BoxRuleError) as info:
        grow_to_fit_all([(60, 30, 70, 34)], SIZE, grown, width=W, height=H)
    error = info.value
    assert (error.code, error.size, error.hits) == (
        "size_would_overlap",
        BoxSize(width=20, height=4),
        (),
    )


def test_grow_to_fit_all_refuses_new_boxes_on_existing_ones():
    rects = [(10, 10, 20, 14), (40, 10, 50, 14), (70, 10, 80, 14)]
    grown = [(12, 20, 18, 24), (72, 10, 78, 14), (41, 10, 49, 14)]  # over the last two
    with pytest.raises(BoxRuleError) as info:
        grow_to_fit_all(rects, SIZE, grown, width=W, height=H)
    error = info.value
    assert (error.code, error.size, error.hits) == ("overlap", SIZE, (1, 2))  # input order
    assert str(error) == "a new box would overlap another box of this protein"


# --- A protein's padding (#57): every fit keeps it ---

PAD = BoxPadding(across=2, along=3)


def test_grow_to_fit_all_pad_adds_twice_the_pad_to_need():
    # The first box: the band's own extent, 11x7, plus the padding on each side,
    # centred on the band's integer centre (45, 23): the band's box grown evenly.
    size, _, [rect] = grow_to_fit_all([], SIZE, [(40, 20, 51, 27)], width=W, height=H, pad=PAD)
    assert (size, rect) == (BoxSize(width=15, height=13), (38, 17, 53, 30))
    assert grow_to_fit([], SIZE, (40, 20, 51, 27), width=W, height=H, pad=PAD) == (size, [], rect)
    # A size fitted elsewhere (a row's detector) is padded the same way.
    need = BoxSize(width=12, height=6)
    size, _, placed = grow_to_fit_all(
        [], SIZE, [(60, 10, 72, 16)], need=need, width=W, height=H, pad=PAD
    )
    assert (size, placed) == (BoxSize(width=16, height=12), [(58, 7, 74, 19)])
    # The 2 px floor applies to the band, before the padding.
    size, _, _ = grow_to_fit_all([], SIZE, [(50, 30, 51, 31)], width=W, height=H, pad=PAD)
    assert size == BoxSize(width=6, height=8)
    # No padding: as before.
    assert grow_to_fit_all([], SIZE, [(40, 20, 51, 27)], width=W, height=H) == (
        BoxSize(width=11, height=7),
        [],
        [(40, 20, 51, 27)],
    )


def test_grow_to_fit_all_pad_only_grows_the_fitted_size():
    # The existing boxes are 14x10 under the padding: fitted 10x4.
    padded = BoxSize(width=14, height=10)
    rects = [(8, 7, 22, 17), (28, 37, 42, 47)]  # centres (15, 12) and (35, 42)
    # A band smaller than the fitted size keeps the size and moves nothing.
    size, resized, [rect] = grow_to_fit_all(
        rects, padded, [(70, 30, 72, 33)], width=W, height=H, pad=PAD
    )
    assert (size, resized, rect) == (padded, rects, (64, 26, 78, 36))
    # A taller band (8 px) grows the fitted height to it, and the padding stays:
    # max(fitted, need) + 2 * pad.
    size, resized, _ = grow_to_fit_all(rects, padded, [(60, 9, 64, 17)], width=W, height=H, pad=PAD)
    assert size == BoxSize(width=14, height=14)
    assert resized == [(8, 5, 22, 19), (28, 35, 42, 49)]  # same centres


def test_grow_to_fit_all_pad_clamps_to_the_image():
    # A 30 px band under 6 px above and below on a 40 px image: the box takes the
    # image's full height, and the fitted size it leaves is 28, less than the band.
    pad = BoxPadding(along=6)
    size, _, [rect] = grow_to_fit_all([], SIZE, [(10, 5, 20, 35)], width=W, height=40, pad=pad)
    assert size == BoxSize(width=10, height=40)
    assert size.height - 2 * pad.along == 28
    assert rect == (10, 0, 20, 40)


def test_padded_new_boxes_that_overlap_are_refused_without_hits():
    # Two 8 px bands 12 px apart fit unpadded, but not 3 px wider on each side.
    grown = [(10, 10, 18, 14), (22, 10, 30, 14)]
    assert grow_to_fit_all([], SIZE, grown, width=W, height=H)[2] == grown
    with pytest.raises(BoxRuleError) as info:
        grow_to_fit_all([], SIZE, grown, width=W, height=H, pad=BoxPadding(across=3))
    error = info.value
    assert (error.code, error.size, error.hits) == (
        "size_would_overlap",
        BoxSize(width=14, height=4),
        (),
    )
    assert str(error) == (
        "the box size 14x4 (fitted 8x4 plus 3 px left and right) would make the new boxes"
        " overlap each other"
    )


def test_padded_refusals_name_the_padding():
    rects = [(0, 0, 14, 10), (15, 0, 29, 10)]  # 1 px apart, fitted 10x4 under PAD
    padded = BoxSize(width=14, height=10)
    with pytest.raises(BoxRuleError) as info:
        grow_to_fit(rects, padded, (60, 30, 72, 34), width=W, height=H, pad=PAD)
    assert str(info.value) == (
        "growing the box size to 16x10 (fitted 12x4 plus 2 px left and right and 3 px above"
        " and below) would make boxes overlap"
    )
    with pytest.raises(BoxRuleError) as info:
        grow_to_fit(rects[:1], padded, (2, 2, 8, 6), width=W, height=H, pad=PAD)
    assert (info.value.code, str(info.value)) == (
        "overlap",
        "the new box would overlap another box of this protein (boxes are padded 2 px left"
        " and right and 3 px above and below)",
    )


def test_size_and_padding_words():
    assert padding_words(BoxPadding()) == ""
    assert padding_words(BoxPadding(along=5)) == "5 px above and below"
    assert padding_words(BoxPadding(across=6)) == "6 px left and right"
    size = BoxSize(width=98, height=25)
    assert size_words(size) == size_words(size, BoxPadding()) == "98x25"
    assert size_words(size, BoxPadding(along=5)) == "98x25 (fitted 98x15 plus 5 px above and below)"


def test_resize_checked_names_the_boxes_that_would_overlap():
    rects = [(0, 0, 10, 4), (15, 0, 25, 4), (60, 30, 70, 34)]
    bigger = BoxSize(width=20, height=4)
    resized, hits = resize_checked(rects, bigger, width=W, height=H)
    assert (resized, hits) == (
        [(0, 0, 20, 4), (10, 0, 30, 4), (55, 30, 75, 34)],  # the first shifted inside
        (0, 1),
    )
    assert resize_all(rects, bigger, width=W, height=H) is None
    assert resize_checked(rects, SIZE, width=W, height=H) == (rects, ())


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
