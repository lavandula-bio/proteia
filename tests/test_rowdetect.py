# SPDX-License-Identifier: Apache-2.0
"""Tests for row-box band detection (#51): one slot per declared lane, empty
lanes that shift nothing, the flags, and the invariants of every result (one
shared size, no overlap, inside the row, plain ints, exact translation,
determinism, polarity symmetry). The rows come from :mod:`rowcases`."""

import dataclasses
import itertools
import json
import math
import re
import time
from functools import cache

import numpy as np
import pytest
from scipy.ndimage import median_filter, uniform_filter

from proteia.core import rowdetect
from proteia.core.evaluate import iou
from proteia.core.grow import grow_box
from proteia.core.model import BoxSize, overlaps
from proteia.core.quantify import estimate_background
from proteia.core.rowdetect import (
    AMBIGUITY_MARGIN,
    DETECT_K,
    FIT_MAX_PIXELS,
    REFUSING_FLAGS,
    SIZE_GUARD,
    SMOOTH,
    WARNING_FLAGS,
    RowDetectError,
    RowDetection,
    detect_row,
    settings,
)
from rowcases import (
    FULL_SCALE,
    MEMBRANE,
    RowCase,
    adversarial,
    adversarial_row,
    bench_cases,
    blob,
    fuzz_row,
)

BENCH = {case.name: case for case in bench_cases()}
# Adversarial rows with every kind of lane result, for the invariant checks.
ADVERSARIAL_KEYS = [
    ("nbr_above_miss", 1000),
    ("blotch_empty", 1000),
    ("vstreak_empty", 1000),
    ("box_omits_empty_first", 1000),
    ("doublet_two", 1001),
    ("twenty_lanes", 1000),
    ("blob_gap", 1000),  # more pieces than lanes: a merge, noted in image x
    ("bubble_band", 1000),  # a band split in two along x
    ("tall_band", 1000),  # size_outlier
    ("doublet_deep", 1000),  # multiple_components
    ("smile_tall_tight", 1000),  # the vertical placement clamp binds
    ("wide_row", 1000),  # above FIT_MAX_PIXELS: every fit subsamples
]


@cache
def _adversarial(key: str, seed: int) -> RowCase:
    return adversarial(key, seed)


def _case(name: str) -> RowCase:
    if name in BENCH:
        return BENCH[name]
    key, seed = name.rsplit("/", 1)
    return _adversarial(key, int(seed))


INVARIANT_CASES = list(BENCH) + [f"{key}/{seed}" for key, seed in ADVERSARIAL_KEYS]


def detect(case: RowCase, **kwargs) -> RowDetection:
    return detect_row(
        case.image,
        case.row,
        case.n_lanes,
        background=estimate_background(case.image),
        dark_on_light=case.dark_on_light,
        **kwargs,
    )


@cache
def _detected(name: str) -> RowDetection:
    return detect(_case(name))


def assert_hits_own_lanes(case: RowCase, found: RowDetection) -> None:
    """Every reference band's own lane got a box over it; no other lane got one."""
    for lane, slot in enumerate(found.slots):
        if lane in case.reference:
            assert slot is not None, f"lane {lane} got no box"
            assert iou(slot, case.reference[lane]) >= 0.5, f"lane {lane}: {slot}"
        else:
            assert slot is None, f"empty lane {lane} got {slot}"


def components(found: RowDetection) -> list[int]:
    return [lane.components for lane in found.lanes]


# --- Lanes present, missing and touching ---


def test_all_lanes_present():
    case = BENCH["all_present"]
    found = _detected("all_present")
    assert_hits_own_lanes(case, found)
    assert found.flags == ()
    assert [lane.reason for lane in found.lanes] == ["band"] * 6
    assert all(lane.components == 1 and lane.snr >= DETECT_K for lane in found.lanes)
    assert found.pitch == pytest.approx(70.0, rel=0.05)
    assert found.size == BoxSize(width=44, height=12)  # the bands' 30% extent


@pytest.mark.parametrize(
    ("name", "missing"),
    [
        ("missing_first", [0]),
        ("missing_middle", [2]),
        ("missing_last", [5]),
        ("missing_two", [0, 3]),
    ],
)
def test_missing_lane_is_an_empty_slot_and_shifts_nothing(name, missing):
    case = BENCH[name]
    found = _detected(name)
    assert [i for i, slot in enumerate(found.slots) if slot is None] == missing
    assert_hits_own_lanes(case, found)  # every other lane keeps its own index
    assert found.flags == ()
    for i in missing:
        lane = found.lanes[i]
        assert (lane.reason, lane.rect, lane.extent, lane.bg_offset) == (
            "no_band",
            None,
            None,
            None,
        )
        assert lane.components == 0
        assert lane.snr < DETECT_K
        # Its expected centre is where the lane is, a tenth of the pitch at most
        # away (also past the present lanes, extrapolated with the pitch).
        assert abs(lane.expected_x - case.lane_cx[i]) < 0.1 * 70


def test_touching_bands_get_one_box_per_lane():
    case = BENCH["touching"]  # pitch 48, bands 60 wide: no valley between them
    found = _detected("touching")
    assert_hits_own_lanes(case, found)
    for rect, cx in zip(found.slots, case.lane_cx, strict=True):
        assert abs((rect[0] + rect[2]) / 2 - cx) < 48 / 4  # centred on its own band
    assert found.pitch == pytest.approx(48.0, rel=0.05)


@pytest.mark.parametrize(("key", "seed"), [("twenty_lanes", 1000), ("gauss_tails_miss", 1000)])
def test_many_lanes_and_long_tails(key, seed):
    # Twenty lanes with empty ends and two adjacent empty lanes; Gaussian tails.
    case = _adversarial(key, seed)
    found = detect(case)
    assert_hits_own_lanes(case, found)
    assert not found.refused


def test_blank_row_has_no_boxes_and_even_expected_centres():
    case = adversarial_row("blank", 1000, missing=range(6))
    found = detect(case)
    assert found.slots == (None,) * 6
    assert (found.size, found.pitch, found.cost, found.margin) == (None, None, None, None)
    assert found.flags == ()
    x0, _, x1, _ = case.row
    for i, lane in enumerate(found.lanes):
        assert lane.reason == "no_band"
        assert lane.expected_x == pytest.approx(x0 + (i + 0.5) * (x1 - x0) / 6)


def test_single_band_among_empty_lanes():
    case = adversarial_row("one", 1000, missing=[0, 1, 2, 4, 5])
    found = detect(case)
    assert_hits_own_lanes(case, found)
    assert found.flags == ()


def test_more_pieces_than_lanes_merge_or_drop_the_weak_one():
    # Dust between lanes 2 and 3 makes a seventh piece. Strong enough beside
    # lane 3's band, the closest pair merges; weaker, it is dropped. The notes
    # give image x (the translation test checks that they move with the row).
    merged = _detected("blob_gap/1000")
    assert merged.notes[0] == "merged pieces at x=257..321"
    dropped = detect(_adversarial("blob_gap", 1003))
    assert dropped.notes == ("dropped a weak piece at x=257..277",)
    assert dropped.flags == ()
    for found in (merged, dropped):
        assert_hits_own_lanes(_adversarial("blob_gap", 1000), found)


# --- Empty-lane reasons ---


@pytest.mark.parametrize(
    ("key", "seed", "lane", "reason"),
    [
        ("nbr_above_miss", 1000, 2, "edge_signal"),  # only the neighbouring row's rows
        ("blotch_empty", 1000, 4, "artefact"),  # a broad stain, rejected as flat
        ("vstreak_empty", 1001, 4, "artefact"),  # a streak down the empty lane
        ("vstreak_empty", 1000, 4, "unassigned"),  # the streak kept, but in no piece
    ],
)
def test_empty_lane_reasons(key, seed, lane, reason):
    case = _adversarial(key, seed)
    found = detect(case)
    assert_hits_own_lanes(case, found)
    empty = found.lanes[lane]
    assert (empty.rect, empty.reason, empty.components) == (None, reason, 0)
    if reason == "unassigned":
        assert empty.snr >= DETECT_K  # signal in the lane's rows, in no piece
    elif reason == "edge_signal":
        assert empty.snr < DETECT_K  # the signal is only in the rows left out


# --- Flags ---

# Every bench row's exact flags and notes: only the ramped membrane warns, and
# only the rows with too few band-free pixels skip stage 2.
BENCH_FLAGS = {"uneven_background": ("background_mismatch",)}
STAGE_2_SKIPPED = ("stage 2 skipped: too few band-free pixels",)
BENCH_NOTES = {"touching": STAGE_2_SKIPPED, "tight_box": STAGE_2_SKIPPED}


@pytest.mark.parametrize("name", list(BENCH))
def test_bench_flags_notes_and_one_component_per_band(name):
    found = _detected(name)
    assert found.flags == BENCH_FLAGS.get(name, ())
    assert found.notes == BENCH_NOTES.get(name, ())
    assert components(found) == [int(lane.rect is not None) for lane in found.lanes]


def test_ambiguous_lanes_refuses():
    # Lane 0 is empty and the box stops short of it: five bands, six lanes.
    found = detect(_adversarial("box_omits_empty_first", 1000))
    assert "ambiguous_lanes" in found.flags
    assert found.margin is not None and found.margin < AMBIGUITY_MARGIN
    assert found.refused


def test_lanes_outside_row_refuses():
    # The box leaves out most of empty lane 0: its expected centre is outside.
    case = adversarial_row("outside", 1000, missing=[0], box_adjust=(50, 0, 0, 0))
    found = detect(case)
    assert "lanes_outside_row" in found.flags
    assert found.slots[0] is None and found.lanes[0].expected_x < case.row[0]
    assert found.refused
    assert found.flags[: len(REFUSING_FLAGS)] == REFUSING_FLAGS  # refusing flags first


@pytest.mark.parametrize(("lane", "adjust"), [(0, (40, 0, 0, 0)), (5, (0, 0, -40, 0))])
def test_lanes_outside_row_alone_refuses_on_either_side(lane, adjust):
    # The box leaves out most of the empty first or last lane, and nothing else
    # is unclear: the one refusing flag, from either end.
    case = adversarial_row("outside", 1000, missing=[lane], box_adjust=adjust)
    found = detect(case)
    assert found.flags == ("lanes_outside_row",)
    assert found.refused and found.slots[lane] is None
    expected = found.lanes[lane].expected_x
    assert expected < case.row[0] if lane == 0 else expected > case.row[2]


def test_background_mismatch_warns_without_refusing():
    case = BENCH["uneven_background"]  # membrane -6000..+6000 across the row
    found = _detected("uneven_background")
    assert found.flags == ("background_mismatch",)
    assert not found.refused
    assert_hits_own_lanes(case, found)
    offsets = [lane.bg_offset for lane in found.lanes]
    assert offsets[0] < -3 and offsets[-1] > 3  # the band side of a dark-on-light row
    assert all(abs(lane.bg_offset) < 1 for lane in _detected("all_present").lanes)


@pytest.mark.parametrize(("k", "warns"), [(-5, True), (-2, False), (2, False), (5, True)])
def test_bg_offset_counts_pixel_sigmas_to_the_band_side(k, warns):
    # The stored background k pixel sigmas lighter (to the membrane side of a
    # dark-on-light row) moves every offset by -k; beyond BG_WARN_K it warns.
    # (Near the membrane the stored level may become the stage-1 surface, which
    # moves the offsets by a few hundredths.)
    case = BENCH["all_present"]
    base = _detected("all_present")
    found = detect_row(
        case.image,
        case.row,
        6,
        background=estimate_background(case.image) + k * base.pixel_noise,
    )
    for moved, lane in zip(found.lanes, base.lanes, strict=True):
        assert moved.bg_offset == pytest.approx(lane.bg_offset - k, abs=0.05)
    assert ("background_mismatch" in found.flags) is warns


def test_size_outlier_does_not_set_the_shared_size():
    # Lane 3's band is 30 px tall, the others about 12: above 2x the median.
    case = _adversarial("tall_band", 1000)
    found = detect(case)
    assert found.flags == ("size_outlier",)
    assert any("lane 4:" in note for note in found.notes)  # notes count lanes from 1
    heights = [lane.extent[3] - lane.extent[1] for lane in found.lanes]
    median = float(np.median(heights))
    assert heights[3] > SIZE_GUARD * median
    assert found.size.height == max(h for h in heights if h <= SIZE_GUARD * median)
    for lane in (0, 1, 2, 4, 5):  # the tall band's own box is too short to hit it
        assert iou(found.slots[lane], case.reference[lane]) >= 0.5
    # The plain maximum lets the outlier set the size, and flags nothing.
    by_max = detect(case, size_rule="max")
    assert by_max.size.height == max(heights)
    assert "size_outlier" not in by_max.flags


def test_doublet_boxes_the_strongest_component_and_flags_the_lane():
    # Lane 2 holds two components 14 px apart, the lower one 0.7x as deep.
    case = _adversarial("doublet_deep", 1000)
    found = detect(case)
    assert found.flags == ("multiple_components",)
    assert components(found) == [1, 1, 2, 1, 1, 1]
    assert any(note.startswith("lane 3:") for note in found.notes)
    rect = found.slots[2]
    upper = case.lane_cy[2] - 7  # the stronger component's centre
    assert abs((rect[1] + rect[3]) / 2 - upper) <= 2
    assert not found.refused


@pytest.mark.parametrize(("dy", "frac"), [(14, 0.2), (14, 0.25), (20, 0.2), (20, 0.3)])
def test_a_weaker_second_component_is_flagged_and_counted_once(dy, frac):
    # Lane 2's second band is dy px below the first and a fifth to 0.3 as deep:
    # under the extent level of the box, yet tens of sigmas above its saddle.
    case = adversarial_row("doublet", 1000, doublet={2: (dy, frac)}, my=12)
    found = detect(case)
    assert found.flags == ("multiple_components",)
    assert components(found) == [1, 1, 2, 1, 1, 1]


def test_a_band_split_by_a_bubble_is_flagged():
    # A hole through lane 1's band leaves two halves side by side.
    found = _detected("bubble_band/1000")
    assert found.flags == ("multiple_components",)
    assert components(found) == [1, 2, 1, 1, 1, 1]


@pytest.mark.parametrize(("depth", "split"), [(-2500.0, False), (-5000.0, False), (-20000.0, True)])
def test_a_light_spot_splits_a_band_only_when_deep(depth, split):
    # A light spot of radius 7 px on lane 1's band. 2500 or 5000 deep, it
    # dents the band's top by 4-10%: two lobes 7-25 sigma apart, one band.
    # 20000 deep, it cuts the band in two. As between pieces along x, two
    # peaks are separate only if the saddle is DETECT_K sigma below the lower
    # one and at most VALLEY_FRAC of it.
    found = detect(adversarial_row("spot", 1002, artefacts=[blob(1, 7.0, depth)]))
    assert components(found)[1] == (2 if split else 1)
    assert ("multiple_components" in found.flags) is split


# Single-band rows where noise alone splits a band's 30% contour: faint, tall
# Gaussian bands; wide ones at SNR 20-40 (seed 1005 has two equally high peaks
# 0.9 sigma above their saddle); faint rows at SNR 11-14 and 7-8, where noise
# dips of 3-6 sigma part a band's top by over a quarter.
SINGLE_BAND_ROWS = {
    "gauss_tall": {"depth_range": (3000.0, 3600.0), "shape": "gauss", "h": 24.0, "my": 8},
    "gauss_wide": {
        "shape": "gauss",
        "pitch": 150.0,
        "w": 100.0,
        "h": 24.0,
        "img_h": 200,
        "depth_range": (2500.0, 4000.0),
    },
    "faint": {"depth_range": (1100.0, 1300.0)},
    "faint_2s": {"depth_range": (750.0, 850.0)},
}


@pytest.mark.parametrize(
    ("recipe", "seed"),
    [("gauss_tall", seed) for seed in (1001, 1003, 1006, 1013, 1015)]
    + [("gauss_wide", seed) for seed in (1000, 1001, 1005)]
    + [("faint", seed) for seed in (1000, 1005, 1012)]
    + [("faint_2s", seed) for seed in (1001, 1002)],
)
def test_single_bands_are_one_component(recipe, seed):
    case = adversarial_row(recipe, seed, **SINGLE_BAND_ROWS[recipe])
    found = detect(case)
    assert "multiple_components" not in found.flags
    assert found.slots.count(None) <= (2 if recipe == "faint_2s" else 0)
    assert components(found) == [int(slot is not None) for slot in found.slots]


def test_flag_vocabulary():
    assert set(REFUSING_FLAGS).isdisjoint(WARNING_FLAGS)
    seen = set()
    for name in INVARIANT_CASES:
        seen.update(_detected(name).flags)
    assert seen <= set(REFUSING_FLAGS) | set(WARNING_FLAGS)


# --- What the result reports ---


def test_noise_is_measured_on_the_band_free_pixels():
    # Stage 2 refits the plane and the noise on the band-free pixels of the
    # row; a stage-1 estimate or plane is 1-6% off these values.
    found = _detected("all_present")
    assert found.noise == pytest.approx(121.46200102625718, rel=1e-4)
    assert found.pixel_noise == pytest.approx(453.93837046984686, rel=1e-4)
    # White noise: the detection noise is the pixel noise over the 3x5 kernel.
    assert found.noise == pytest.approx(found.pixel_noise / math.sqrt(15), rel=0.1)


def test_snr_is_the_smoothed_band_peak_over_the_noise():
    case = BENCH["all_present"]
    found = _detected("all_present")
    darkness = MEMBRANE - uniform_filter(case.image, size=SMOOTH, mode="nearest")
    for lane in found.lanes:
        x0, y0, x1, y1 = case.reference[lane.lane]
        assert lane.snr * found.noise == pytest.approx(darkness[y0:y1, x0:x1].max(), rel=0.02)


# Rows whose bands stand alone on a flat membrane: not touching, not faint
# beside strong ones, not on a ramp, not cut by the box.
CLICK_ROWS = [
    name
    for name in BENCH
    if name not in ("touching", "faint_band", "uneven_background", "tight_box")
]


@pytest.mark.parametrize("name", CLICK_ROWS)
def test_extent_is_what_a_click_grows(name):
    # EXTENT_LEVEL is the click's REL_THRESHOLD: a click at the extent's centre
    # grows the same extent, to a pixel.
    case = BENCH[name]
    background = estimate_background(case.image)
    for lane in _detected(name).lanes:
        if lane.extent is None:
            continue
        x0, y0, x1, y1 = lane.extent
        seed = ((x0 + x1) // 2, (y0 + y1) // 2)
        click = grow_box(case.image, seed, background, dark_on_light=case.dark_on_light)
        assert click is not None
        assert max(abs(a - b) for a, b in zip(click, lane.extent, strict=True)) <= 1, lane


# --- Invariants of every result ---


def check_invariants(case: RowCase, found: RowDetection) -> None:
    """The invariants of the module contract, for any input."""
    height, width = case.image.shape
    rx0, ry0, rx1, ry1 = case.row
    rx0, ry0, rx1, ry1 = max(0, rx0), max(0, ry0), min(width, rx1), min(height, ry1)
    assert len(found.lanes) == case.n_lanes
    assert [lane.lane for lane in found.lanes] == list(range(case.n_lanes))
    rects = [slot for slot in found.slots if slot is not None]
    assert (found.size is None) == (not rects)
    for rect in rects:
        assert all(type(v) is int for v in rect)  # plain ints, not numpy
        assert (rect[2] - rect[0], rect[3] - rect[1]) == (found.size.width, found.size.height)
        assert rx0 <= rect[0] and rect[2] <= rx1 and ry0 <= rect[1] and rect[3] <= ry1
    for a, b in itertools.combinations(rects, 2):
        assert not overlaps(a, b)
    assert rects == sorted(rects)  # lanes left to right
    if found.size is not None:
        assert type(found.size.width) is int and type(found.size.height) is int
    for lane in found.lanes:
        assert (lane.rect is None) == (lane.reason != "band") == (lane.extent is None)
        assert (lane.rect is None) == (lane.bg_offset is None) == (lane.components == 0)
        if lane.extent is not None:  # image coordinates, inside the clipped row
            assert all(type(v) is int for v in lane.extent)
            ex0, ey0, ex1, ey1 = lane.extent
            assert rx0 <= ex0 < ex1 <= rx1 and ry0 <= ey0 < ey1 <= ry1
        assert math.isfinite(lane.snr) and math.isfinite(lane.expected_x)


@pytest.mark.parametrize("name", INVARIANT_CASES)
def test_result_invariants(name):
    check_invariants(_case(name), _detected(name))


@pytest.mark.parametrize("seed", range(2000, 2030))
def test_result_invariants_on_random_rows(seed):
    # Random box edges, lane counts off by one, a tall band, a smile: whatever
    # is proposed, the invariants hold.
    case = fuzz_row(seed)
    check_invariants(case, detect(case))


def _shift_notes(notes: tuple[str, ...], dx: int) -> tuple[str, ...]:
    def move(m: re.Match[str]) -> str:
        return f"x={int(m[1]) + dx}..{int(m[2]) + dx}"

    return tuple(re.sub(r"x=(\d+)\.\.(\d+)", move, note) for note in notes)


@pytest.mark.parametrize("name", INVARIANT_CASES)
def test_translation_is_exact(name):
    # Shift the image and the row: every rect, extent and note x shifts by
    # exactly the same, and nothing else changes.
    case = _case(name)
    found = _detected(name)
    dx, dy = 13, 7
    height, width = case.image.shape
    shifted = np.full((height + dy + 5, width + dx + 11), 12345.0)
    shifted[dy : dy + height, dx : dx + width] = case.image
    x0, y0, x1, y1 = case.row
    moved = detect_row(
        shifted,
        (x0 + dx, y0 + dy, x1 + dx, y1 + dy),
        case.n_lanes,
        background=estimate_background(case.image),
        dark_on_light=case.dark_on_light,
    )

    def move(r):
        return None if r is None else (r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy)

    for after, before in zip(moved.lanes, found.lanes, strict=True):
        assert after.expected_x - dx == pytest.approx(before.expected_x)
        assert dataclasses.replace(after, expected_x=0.0) == dataclasses.replace(
            before, rect=move(before.rect), extent=move(before.extent), expected_x=0.0
        )
    assert dataclasses.replace(moved, lanes=(), notes=()) == dataclasses.replace(
        found, lanes=(), notes=()
    )
    assert moved.notes == _shift_notes(found.notes, dx)


@pytest.mark.parametrize("name", INVARIANT_CASES)
def test_deterministic_and_read_only(name):
    case = _case(name)
    image = case.image.copy()
    image.setflags(write=False)  # detection must never write its input
    again = detect_row(
        image,
        case.row,
        case.n_lanes,
        background=estimate_background(case.image),
        dark_on_light=case.dark_on_light,
    )
    assert again == _detected(name)
    assert np.array_equal(image, case.image)


@pytest.mark.parametrize("name", INVARIANT_CASES)
def test_polarity_symmetry(name):
    # The photometric inverse, read light-on-dark, gives the same boxes and the
    # same band-side background offsets.
    case = _case(name)
    found = _detected(name)
    inverse = detect_row(
        FULL_SCALE - case.image,
        case.row,
        case.n_lanes,
        background=FULL_SCALE - estimate_background(case.image),
        dark_on_light=not case.dark_on_light,
    )
    assert inverse.slots == found.slots
    assert inverse.flags == found.flags
    assert [lane.reason for lane in inverse.lanes] == [lane.reason for lane in found.lanes]
    assert components(inverse) == components(found)
    for a, b in zip(inverse.lanes, found.lanes, strict=True):
        assert (a.bg_offset is None) == (b.bg_offset is None)
        if b.bg_offset is not None:
            assert a.bg_offset == pytest.approx(b.bg_offset, abs=1e-6)


def test_integer_input_and_a_row_beyond_the_image():
    case = BENCH["all_present"]
    height, width = case.image.shape
    x0, y0, x1, y1 = case.row
    found = detect_row(
        case.image.astype(np.uint16),  # integer pixels, as tifffile gives them
        (x0 - 1000, y0, width + 1000, y1),  # clipped to the image first
        6,
        background=estimate_background(case.image),
    )
    clipped = detect_row(
        case.image, (0, y0, width, y1), 6, background=estimate_background(case.image)
    )
    assert found == clipped
    assert all(0 <= r[0] and r[2] <= width for r in found.slots if r is not None)


def test_numpy_integer_arguments_are_ints():
    case = BENCH["all_present"]
    row = tuple(np.int64(v) for v in case.row)
    found = detect_row(case.image, row, np.int32(6), background=estimate_background(case.image))
    assert found == _detected("all_present")


def test_fits_subsample_at_a_fixed_stride_across_every_index():
    # Determinism and an unbiased fit rest on it: every stride-th index from
    # the first, never a random or leading subset. The wide row takes this path.
    idx = np.arange(3 * FIT_MAX_PIXELS + 7) * 2
    sub = rowdetect._subsample(idx)
    assert sub.size <= FIT_MAX_PIXELS
    assert np.array_equal(sub, idx[:: sub[1] // 2])
    assert sub[-1] >= idx[-1] - 2 * (sub[1] // 2)
    few = np.arange(FIT_MAX_PIXELS)
    assert np.array_equal(rowdetect._subsample(few), few)
    x0, y0, x1, y1 = _case("wide_row/1000").row
    assert (x1 - x0) * (y1 - y0) > 4 * FIT_MAX_PIXELS


def test_despeckle_is_a_running_median_along_x_at_a_cost_flat_in_its_width():
    rng = np.random.default_rng(51)
    small = np.round(rng.normal(1000.0, 50.0, (12, 300)))
    for k in (3, 25, 151, 301, 601):
        expected = median_filter(small, size=(1, k), mode="nearest")
        assert np.array_equal(rowdetect._despeckle(small, k), expected)
    # A few wide lanes make the window wide; the cost must not grow with it.
    big = np.round(rng.normal(1000.0, 50.0, (200, 2000)))

    def cost(k: int) -> float:
        best = math.inf
        for _ in range(3):
            start = time.perf_counter()
            rowdetect._despeckle(big, k)
            best = min(best, time.perf_counter() - start)
        return best

    assert cost(201) < 4 * cost(3)


# --- Errors ---

IMAGE = np.full((40, 60), 1000.0)


@pytest.mark.parametrize(
    ("gray", "row", "n_lanes", "code"),
    [
        (np.zeros((40, 60, 3)), (0, 0, 60, 40), 2, "invalid_row"),  # not 2-D
        (IMAGE, (0, 0, 60, 40), 0, "invalid_row"),
        (IMAGE, (0, 0, 60, 40), -1, "invalid_row"),
        (IMAGE, (0, 0, 60, 40), True, "invalid_row"),  # a bool is not a lane count
        (IMAGE, (0, 0, 60, 40), 2.0, "invalid_row"),
        (IMAGE, (0, 0, 60), 2, "invalid_row"),
        (IMAGE, (0, 0, 60.0, 40), 2, "invalid_row"),
        (IMAGE, (0, 0, True, 40), 2, "invalid_row"),
        (IMAGE, None, 2, "invalid_row"),
        (IMAGE, (10, 0, 10, 40), 2, "invalid_row"),  # empty
        (IMAGE, (0, 30, 60, 20), 2, "invalid_row"),  # inverted
        (IMAGE, (60, 0, 90, 40), 2, "row_outside_image"),
        (IMAGE, (-30, 0, 0, 40), 2, "row_outside_image"),
        (IMAGE, (0, 40, 60, 50), 2, "row_outside_image"),
        (IMAGE, (0, 0, 7, 40), 4, "row_too_small"),  # under MIN_BOX px per lane
        (IMAGE, (55, 0, 70, 40), 3, "row_too_small"),  # 5 px left after clipping
        (IMAGE, (0, 38, 60, 45), 2, "row_too_small"),  # 2 rows after clipping
    ],
)
def test_row_errors_have_stable_codes(gray, row, n_lanes, code):
    with pytest.raises(RowDetectError) as err:
        detect_row(gray, row, n_lanes, background=1000.0)
    assert err.value.code == code
    assert isinstance(err.value, ValueError)


def test_non_finite_pixels_are_an_invalid_row():
    image = IMAGE.copy()
    image[20, 30] = np.nan
    with pytest.raises(RowDetectError) as err:
        detect_row(image, (0, 0, 60, 40), 2, background=1000.0)
    assert err.value.code == "invalid_row"
    # A NaN outside the row does not matter.
    assert detect_row(image, (40, 0, 60, 40), 2, background=1000.0).slots == (None, None)


@pytest.mark.parametrize("background", [math.nan, math.inf, -math.inf, None, "1000", True])
def test_a_background_that_is_not_a_finite_number_is_an_invalid_row(background):
    image = IMAGE.copy()
    image[1, 1] = 0.0
    for row in ((0, 0, 60, 40), (0, 0, 4, 3)):
        with pytest.raises(RowDetectError) as err:
            detect_row(image, row, 2, background=background)
        assert err.value.code == "invalid_row"
    # Any finite real number is a background, a numpy one included.
    for background in (1000, np.float32(1000.0), np.int64(1000)):
        assert len(detect_row(image, (0, 0, 4, 3), 2, background=background).lanes) == 2


def test_unknown_size_rule_is_a_value_error():
    with pytest.raises(ValueError, match="size rule") as err:
        detect_row(IMAGE, (0, 0, 60, 40), 2, background=1000.0, size_rule="p75")
    assert not isinstance(err.value, RowDetectError)


def test_flat_membrane_finds_nothing():
    found = detect_row(IMAGE, (0, 0, 60, 40), 2, background=1000.0)
    assert found.slots == (None, None) and found.flags == ()


# --- Settings ---


def _constants() -> dict[str, object]:
    """rowdetect's public settings: its upper-case constants, except the
    vocabularies (tuples of names)."""
    return {
        name: value
        for name, value in vars(rowdetect).items()
        if name.isupper()
        and not name.startswith("_")
        and not (isinstance(value, tuple) and all(isinstance(v, str) for v in value))
    }


def test_settings_are_json_plain_and_list_every_constant():
    found = settings()
    assert json.loads(json.dumps(found, allow_nan=False)) == found
    constants = _constants()
    assert set(found) == {name.lower() for name in constants}
    for name, value in constants.items():
        assert found[name.lower()] == (list(value) if isinstance(value, tuple) else value)
    assert found["size_rule"] == "max_guarded" and found["size_guard"] == 2.0
    assert found["detect_k"] == 6.0 and found["extent_level"] == found["rel_threshold"] == 0.3


# Tuning values once written inline: named, reported, and each one used.
NAMED_SETTINGS = {
    "row_walk_tol": 0.02,
    "row_min_rows": 5,
    "row_min_keep": 3,
    "min_fit": 16,
    "q_window": 2,
    "gap_search": 1,
    "despeckle_min": 3,
    "envelope_min": 2,
    "bg_guard_min": 2,
    "cell_seed": 0.25,
}


def test_settings_name_every_tuning_value():
    found = settings()
    assert {name: found.get(name) for name in NAMED_SETTINGS} == NAMED_SETTINGS


@pytest.mark.parametrize(
    ("name", "value", "row"),
    [
        ("row_walk_tol", 0.3, "nbr_above_miss/1000"),
        ("row_min_rows", 1000, "nbr_above_miss/1000"),
        ("row_min_keep", 1000, "nbr_above_miss/1000"),
        ("min_fit", 2000, "touching"),
        ("q_window", -1, "touching"),
        ("gap_search", 0, "missing_middle"),
        ("despeckle_min", 99, "all_present"),
        ("envelope_min", 60, "touching"),
        ("bg_guard_min", 20, "all_present"),
        ("cell_seed", 0.0, "touching"),
    ],
)
def test_each_named_setting_changes_detection(monkeypatch, name, value, row):
    base = _detected(row)  # cached before the setting changes
    monkeypatch.setattr(rowdetect, name.upper(), value)
    assert settings()[name] == value
    assert detect(_case(row)) != base


def test_settings_are_a_fresh_copy():
    found = settings()
    found["smooth"].append(99)
    assert settings()["smooth"] == [3, 5]


def test_box_size_is_the_model_type():
    assert isinstance(_detected("all_present").size, BoxSize)
