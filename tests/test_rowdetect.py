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
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import (
    find_objects,
    gaussian_filter,
    label,
    maximum_position,
    median_filter,
    uniform_filter,
)
from skimage.morphology import reconstruction

from proteia.core import rowdetect
from proteia.core.evaluate import hit_rate, iou
from proteia.core.grow import NOISE_K, grow_box
from proteia.core.model import BoxSize, lanes_phrase, overlaps
from proteia.core.quantify import estimate_background
from proteia.core.rowdetect import (
    AMBIGUITY_MARGIN,
    APART_DOUBT,
    BG_GUARD,
    BG_GUARD_MIN,
    CUT_LEVEL,
    DETECT_K,
    EMPTY_WINDOW,
    END_DOUBT,
    EXTENT_LEVEL,
    FIT_MAX_PIXELS,
    HOLLOW_PIXELS,
    LANES_DOUBT,
    MEMBRANE_SHIFT_K,
    PITCH_DOUBT,
    REFUSING_FLAGS,
    ROW_LINE_K,
    ROW_LINE_MIN,
    ROW_LINE_TOL,
    ROW_LINE_TOL_PX,
    ROW_SMILE,
    SECOND_SHARE,
    SIZE_GUARD,
    SMOOTH,
    WARNING_FLAGS,
    RowDetectError,
    RowDetection,
    detect_row,
    settings,
)
from rowcases import (
    ADVERSARIAL,
    FULL_SCALE,
    MEMBRANE,
    NOISE_SIGMA,
    RowCase,
    adversarial,
    adversarial_row,
    band_between,
    bench_cases,
    beside,
    blob,
    bottom_strip,
    dark_edge,
    frame,
    framed,
    fuzz_row,
    hstripe,
    image_cut,
    jpeg,
    mw_slot_row,
    shade_above,
    synthetic_row,
    two_rows,
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
    ("dumbbell_band", 1000),  # a dip inside one grown extent
    ("hollow_band", 1000),  # the same, clipped flat at 0 around it
    ("hollow_ring", 1000),  # a ring: saturated all around a lighter centre
    ("notched_band", 1000),  # a saturated band's ends rising above its middle
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
        # The slot its snr was read in: EMPTY_WINDOW pitch either side of that
        # centre, over the row's rows (no neighbouring row here: all of them).
        x0, y0, x1, y1 = case.row
        half = EMPTY_WINDOW * found.pitch
        assert lane.window == (
            max(x0, math.floor(lane.expected_x - half)),
            y0,
            min(x1, math.ceil(lane.expected_x + half)),
            y1,
        )
    assert all(lane.window is None for lane in found.lanes if lane.rect is not None)


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
        # The streak peaks on an edge row of the box: signal at the box's edge,
        # never kept (#123: its SNR leaves it out, so it is not unassigned).
        ("vstreak_empty", 1000, 4, "edge_signal"),
    ],
)
def test_empty_lane_reasons(key, seed, lane, reason):
    case = _adversarial(key, seed)
    found = detect(case)
    assert_hits_own_lanes(case, found)
    empty = found.lanes[lane]
    assert (empty.rect, empty.reason, empty.components) == (None, reason, 0)
    assert empty.window is not None  # measured, whatever the reason
    if key == "nbr_above_miss":  # the neighbouring row's rows are left out of the slot
        assert empty.window[1] > case.row[1]
    if reason == "edge_signal":
        assert empty.snr < DETECT_K  # the signal is only at the box's edge
    assert not empty.cut  # a neighbouring row, a streak: no band the box cuts through


def test_a_kept_band_in_no_assigned_piece_leaves_its_lane_unassigned():
    # Lane 2 spreads to 1.4 pitches over empty lane 3: the piece is lane 2's,
    # and the kept signal it spills into lane 3 reaches the detection level.
    found = _detected("wide_next_empty/1000")
    empty = found.lanes[3]
    assert (empty.rect, empty.reason) == (None, "unassigned")
    assert empty.snr >= DETECT_K  # a kept candidate in the lane's rows, in no piece


def darkness(case: RowCase, window) -> float:
    """The strongest box-smoothed darkening of the case's membrane in
    ``window``, as the detector smooths its signal."""
    x0, y0, x1, y1 = window
    smoothed = uniform_filter(case.image, size=SMOOTH, mode="nearest")
    return float((MEMBRANE - smoothed[y0:y1, x0:x1]).max())


@pytest.mark.parametrize(
    "speck",
    [
        # 3x3 px: gone under the running median along x.
        pytest.param(blob(5, 1.2, 30000), id="under-the-running-median"),
        # 5x7 px: taller, it outlasts the running median, but its core there
        # is narrower than the dust floor.
        pytest.param(blob(5, 2.0, 30000, ry=3.0), id="below-the-width-floor"),
    ],
)
def test_a_speck_in_an_empty_lane_leaves_it_no_band(speck):
    # A speck in empty end lane 5 reaches the detection level, but the dust
    # rules reject it: the lane holds no band (#123), so it gets a not-detected
    # record instead of reading as signal that fits no lane or at the edge.
    case = adversarial_row("speck", 1000, missing=[5], artefacts=[speck])
    found = detect(case)
    assert_hits_own_lanes(case, found)
    lane = found.lanes[5]
    assert darkness(case, lane.window) >= DETECT_K * found.noise  # it is there
    assert (lane.reason, lane.components, lane.cut) == ("no_band", 0, False)
    assert lane.snr < DETECT_K
    assert found.flags == ()


def test_a_speck_on_the_row_box_edge_is_dust_not_a_cut_band():
    # A 3x3 px speck centred on the box's top row, in empty end lane 5: dust
    # wherever it peaks, so the lane holds no band (#123) and no band is cut
    # (#115); the bands below stay whole.
    case = adversarial_row(
        "edge speck", 1000, missing=[5], artefacts=[blob(5, 1.2, 30000, dy=-10.0)]
    )
    x0, _, x1, y1 = case.row
    top = round(case.lane_cy[5] - 10.0)  # the speck's centre row
    found = detect(dataclasses.replace(case, row=(x0, top, x1, y1)))
    assert_hits_own_lanes(case, found)
    lane = found.lanes[5]
    assert darkness(case, (lane.window[0], top, lane.window[2], top + 1)) >= DETECT_K * found.noise
    assert (lane.reason, lane.cut) == ("no_band", False)
    assert found.flags == ()


def test_a_band_below_the_detection_level_keeps_its_snr():
    # The empty lane's SNR still tells how close a faint band came.
    case = adversarial_row("faint", 1000, depths={5: 500.0})
    lane = detect(case).lanes[5]
    assert lane.reason == "no_band"
    assert NOISE_K < lane.snr < DETECT_K


def test_a_band_peaked_on_the_row_box_edge_is_signal_at_the_edge():
    # The box's top edge runs through the bands' centres: each band's peak
    # lies on its top row, so no band is kept, and each lane holds signal at
    # the box's edge only (not signal that fits no lane).
    case = CUT_BASE
    x0, _, x1, y1 = case.row
    found = detect(dataclasses.replace(case, row=(x0, _centre_row(case), x1, y1)))
    assert found.size is None
    assert [lane.reason for lane in found.lanes] == ["edge_signal"] * 6
    assert all(lane.snr < DETECT_K for lane in found.lanes)
    # Each is a band the box cuts through, kept or not (#115, #117).
    assert [lane.cut for lane in found.lanes] == [True] * 6
    assert (found.flags, found.notes) == (("cut_by_row_box",), (CUT_NOTE,))


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
    assert not any(lane.cut for lane in found.lanes)  # no bench box cuts a band (#115)
    assert found.membrane_shift < MEMBRANE_SHIFT_K  # membrane around every row's bands


# --- #115: bands the row box cuts through ---

CUT_BASE = adversarial_row("cut", 1000)  # six bands, about 12 px high
CUT_NOTE = (
    "lanes 1, 2, 3, 4, 5, 6: the row box's top or bottom edge cuts through the band"
    " (the box's edge row holds at least 30% of its peak)"
)


def _centre_row(case: RowCase) -> int:
    """The row through the bands' mean centre."""
    return round(float(np.mean(case.lane_cy)))


def cut_row(case: RowCase, edge: str, into: int) -> tuple[int, int, int, int]:
    """The case's row box with its top or bottom edge moved ``into`` px short of
    the bands' mean centre row."""
    x0, y0, x1, y1 = case.row
    centre = _centre_row(case)
    return (x0, centre - into, x1, y1) if edge == "top" else (x0, y0, x1, centre + into)


@pytest.mark.parametrize("edge", ["top", "bottom"])
def test_a_row_box_edge_through_the_bands_flags_them_cut(monkeypatch, edge):
    row = cut_row(CUT_BASE, edge, 4)
    found = detect(dataclasses.replace(CUT_BASE, row=row))
    assert found.flags == ("cut_by_row_box",)
    assert found.notes == (CUT_NOTE,)
    assert [lane.cut for lane in found.lanes] == [True] * 6
    edge_y = row[1] if edge == "top" else row[3]
    assert all(edge_y in (lane.extent[1], lane.extent[3]) for lane in found.lanes)
    assert not found.refused  # a warning: the boxes are placed as without it
    monkeypatch.setattr(rowdetect, "CUT_LEVEL", 1.0)  # no edge row holds the whole peak
    uncut = detect(dataclasses.replace(CUT_BASE, row=row))
    assert (uncut.flags, uncut.notes) == ((), ())
    assert not any(lane.cut for lane in uncut.lanes)
    assert (uncut.slots, uncut.size) == (found.slots, found.size)


def test_a_band_the_box_edge_only_grazes_is_not_cut():
    # The edge row a little past the band's own extent holds less than
    # CUT_LEVEL of its peak: the box takes the whole band.
    found = detect(dataclasses.replace(CUT_BASE, row=cut_row(CUT_BASE, "top", 8)))
    assert found.flags == ()
    assert not any(lane.cut for lane in found.lanes)


def test_only_the_lanes_the_box_cuts_are_named():
    # A smile: the outer bands sit higher, so a low top edge cuts only them.
    case = adversarial_row("smile_cut", 1000, smile=8.0)
    x0, y0, x1, y1 = case.row
    middle = min(case.lane_cy)
    found = detect(dataclasses.replace(case, row=(x0, round(middle) - 2, x1, y1)))
    cut = [lane.lane for lane in found.lanes if lane.cut]
    assert cut and len(cut) < 6
    assert found.flags == ("cut_by_row_box",)
    assert found.notes[-1].startswith(f"lanes {', '.join(str(i + 1) for i in cut)}: ")


def test_a_band_left_out_for_peaking_on_the_box_edge_is_named_cut():
    # A lower top edge still: the smile's outer bands peak on its row and are
    # left out (edge_signal), the next ones reach it as kept extents. Both are
    # bands the box cuts through (#115), placed or not, and the row is placed.
    case = adversarial_row("smile_cut", 1000, smile=8.0)
    x0, _, x1, y1 = case.row
    found = detect(dataclasses.replace(case, row=(x0, math.floor(min(case.lane_cy)) + 2, x1, y1)))
    assert [lane.reason for lane in found.lanes] == ["edge_signal", *["band"] * 4, "edge_signal"]
    assert [lane.cut for lane in found.lanes] == [True, True, False, False, True, True]
    assert found.flags == ("cut_by_row_box",)
    assert found.notes[-1].startswith("lanes 1, 2, 5, 6: ")


def test_the_cut_flag_is_a_warning_listed_with_its_setting():
    assert "cut_by_row_box" in WARNING_FLAGS
    assert settings()["cut_level"] == CUT_LEVEL == EXTENT_LEVEL


# --- #117: a box with too little membrane around its bands ---

BLANK = adversarial_row("blank", 1000, missing=range(6))


@pytest.mark.parametrize(
    ("case", "background", "too_little"),
    [
        # Saturated touching bands fill a snug box: detection takes their level
        # for the membrane and finds nothing.
        pytest.param(
            adversarial_row(
                "thick", 1000, h=30.0, pitch=48.0, w=60.0, depth_range=(75000.0, 80000.0), my=0
            ),
            None,
            True,
            id="filled-by-bands",
        ),
        # A box just inside six saturated bands' 20% extents: the bands take in
        # the noise estimate, so nothing reaches the detection level.
        pytest.param(
            adversarial_row("snug", 1000, w=56.0, h=12.0, depth_range=(75000.0, 80000.0), my=-1),
            None,
            True,
            id="snug",
        ),
        pytest.param(BLANK, None, False, id="blank"),
        # The membrane under the box 7.5 noise sigmas darker than the stored
        # background (a darker stretch of membrane): membrane all the same.
        pytest.param(BLANK, MEMBRANE + 3000.0, False, id="darker-than-stored"),
        # Bands too faint to detect, with membrane around them.
        pytest.param(
            adversarial_row("faint", 1000, depth_range=(500.0, 500.0)), None, False, id="faint"
        ),
    ],
)
def test_membrane_shift_tells_a_box_with_too_little_membrane(case, background, too_little):
    background = estimate_background(case.image) if background is None else background
    found = detect_row(case.image, case.row, case.n_lanes, background=background)
    assert found.size is None
    assert (found.membrane_shift >= MEMBRANE_SHIFT_K) == too_little


def test_the_membrane_shift_setting_is_listed():
    assert settings()["membrane_shift_k"] == MEMBRANE_SHIFT_K == DETECT_K


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
    # Its slot is clipped to the row box: only the part inside was measured.
    assert found.lanes[0].window[0] == case.row[0]
    assert found.refused
    # Refusing flags first, in their order.
    assert found.flags[:2] == ("lanes_outside_row", "ambiguous_lanes") == REFUSING_FLAGS[:2]


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
    # The plane on the membrane stays the detection surface whatever the
    # stored level, so nothing else moves.
    case = BENCH["all_present"]
    base = _detected("all_present")
    found = detect_row(
        case.image,
        case.row,
        6,
        background=estimate_background(case.image) + k * base.pixel_noise,
    )
    assert found.slots == base.slots and found.noise == base.noise
    for moved, lane in zip(found.lanes, base.lanes, strict=True):
        assert moved.bg_offset == pytest.approx(lane.bg_offset - k, abs=1e-6)
    assert ("background_mismatch" in found.flags) is warns


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("dark_on_light", [True, False])
@pytest.mark.parametrize("k", [5.0, -4.0])
def test_background_mismatch_measures_the_membrane_under_the_boxes(
    monkeypatch, flat, dark_on_light, k
):
    # The stored background k pixel sigmas off the membrane, to the band side
    # (k > 0) or the membrane side, in either polarity. With ``flat`` every
    # plane fit fails, so the stored level is the detection surface in both
    # stages: the offset of the membrane from it (the signal's own centre)
    # still measures the mismatch.
    sigma = _detected("all_present").pixel_noise  # cached before any patch
    if flat:
        monkeypatch.setattr(rowdetect, "_fit_plane", lambda *args: None)
    case = BENCH["all_present"]
    sign = 1.0 if dark_on_light else -1.0
    image = case.image if dark_on_light else FULL_SCALE - case.image
    membrane = MEMBRANE if dark_on_light else FULL_SCALE - MEMBRANE
    found = detect_row(
        image, case.row, 6, background=membrane - sign * k * sigma, dark_on_light=dark_on_light
    )
    assert found.flags == ("background_mismatch",)
    assert_hits_own_lanes(case, found)
    for lane in found.lanes:
        assert lane.bg_offset == pytest.approx(k, abs=0.5)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("dark_on_light", [True, False])
@pytest.mark.parametrize("k", [0.8, 1.5, 2.0])
def test_a_stored_level_beyond_the_membrane_still_reads_the_membrane_as_zero(
    monkeypatch, flat, dark_on_light, k
):
    # The stored background k pixel sigmas to the membrane side. With ``flat``
    # every plane fit fails, so it is the stage-1 surface, yet next to no pixel
    # lies beyond it: the median, about which the values spread as the white
    # noise, is the membrane's level. Either way the membrane is not read as
    # signal, stage 2 has its band-free pixels, and the offsets are -k.
    sigma = _detected("all_present").pixel_noise  # cached before any patch
    if flat:
        monkeypatch.setattr(rowdetect, "_fit_plane", lambda *args: None)
    case = BENCH["all_present"]
    sign = 1.0 if dark_on_light else -1.0
    image = case.image if dark_on_light else FULL_SCALE - case.image
    membrane = MEMBRANE if dark_on_light else FULL_SCALE - MEMBRANE
    found = detect_row(
        image, case.row, 6, background=membrane + sign * k * sigma, dark_on_light=dark_on_light
    )
    assert found.flags == () and found.notes == ()  # stage 2 ran
    assert_hits_own_lanes(case, found)
    for lane in found.lanes:
        assert lane.bg_offset == pytest.approx(-k, abs=0.15)


def _spy_surfaces(monkeypatch) -> list[np.ndarray]:
    """The detection surface of each stage, in order, as detection runs."""
    surfaces: list[np.ndarray] = []
    real = rowdetect._signal

    def spy(crop, crop_ds, plane, *args):
        surfaces.append(plane)
        return real(crop, crop_ds, plane, *args)

    monkeypatch.setattr(rowdetect, "_signal", spy)
    return surfaces


@pytest.mark.parametrize("dark_on_light", [True, False])
@pytest.mark.parametrize("k", [-2.0, -1.0, 1.0, 2.0])
def test_a_plane_on_the_membrane_stands_whatever_the_stored_level(monkeypatch, dark_on_light, k):
    # The stored background k pixel sigmas off the membrane, beyond it (k > 0)
    # or into the band side. Beyond it, only the membrane's tail lies past it,
    # so the membrane side spreads less about it than about the plane; the
    # plane's spreads as the pixel noise and stands, in both stages.
    sigma = _detected("all_present").pixel_noise  # cached before the spy
    surfaces = _spy_surfaces(monkeypatch)
    case = BENCH["all_present"]
    sign = 1.0 if dark_on_light else -1.0
    image = case.image if dark_on_light else FULL_SCALE - case.image
    membrane = MEMBRANE if dark_on_light else FULL_SCALE - MEMBRANE
    background = membrane + sign * k * sigma
    found = detect_row(image, case.row, 6, background=background, dark_on_light=dark_on_light)
    assert_hits_own_lanes(case, found)
    assert len(surfaces) == 2
    assert not any(np.all(surface == background) for surface in surfaces)


def _bright_strip(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
    """The judge's light strip: +3000 over the 6 rows from 15 to 9 px above the
    highest band centre, across the image."""
    top = float(np.min(lcy))
    return -3000.0 * ((Y < top - 9) & (Y >= top - 15)) + 0.0 * X


def test_stage_2_keeps_the_stored_level_where_the_plane_spreads_wide(monkeypatch):
    # A light strip across the top rows widens the membrane side about the
    # plane to over MEMBRANE_SPREAD_K pixel sigmas, and less about the stored
    # level: both stages detect over the stored level, stage 2 judging on the
    # band-free pixels as stage 1 on all.
    case = adversarial_row("bright_strip", 1009, depths={3: 1500.0}, artefacts=[_bright_strip])
    background = estimate_background(case.image)
    surfaces = _spy_surfaces(monkeypatch)
    found = detect(case)
    assert_hits_own_lanes(case, found)
    assert "stage 2 skipped: too few band-free pixels" not in found.notes
    assert len(surfaces) == 2
    assert all(np.all(surface == background) for surface in surfaces)


@pytest.mark.parametrize("dark_on_light", [True, False])
def test_a_plane_pulled_into_faint_close_bands_gives_way_to_the_stored_level(
    monkeypatch, dark_on_light
):
    # Faint bands (1500 to 3000 deep) 40 px apart in a tight box pull the
    # fitted plane about a pixel sigma into them: its membrane side spreads
    # over MEMBRANE_SPREAD_K pixel sigmas. Over it the noise would be read
    # four times too high and every band missed (stage 2 is skipped); over
    # the stored level every band is found.
    case = adversarial_row(
        "tight_faint",
        1002,
        pitch=40.0,
        mx=2,
        my=2,
        depth_range=(1500.0, 3000.0),
        light_on_dark=not dark_on_light,
    )
    surfaces = _spy_surfaces(monkeypatch)
    found = detect(case)
    assert found.flags == ()
    assert "stage 2 skipped: too few band-free pixels" in found.notes
    assert_hits_own_lanes(case, found)
    assert np.all(surfaces[0] == estimate_background(case.image))


@pytest.mark.parametrize("dark_on_light", [True, False])
def test_a_box_filled_by_bands_is_read_over_the_stored_level(monkeypatch, dark_on_light):
    # Twelve touching bands fill a tight box: the fitted plane is pulled into
    # them, and next to no pixel lies beyond the stored level. Stage 1 detects
    # over the stored level with the white noise (the median is a band's
    # level, far wider spread), finds every band, and leaves too few band-free
    # pixels for stage 2.
    case = adversarial_row(
        "touch12",
        1036,
        n=12,
        pitch=40.0,
        w=44.0,
        h=12.0,
        mx=4,
        my=3,
        light_on_dark=not dark_on_light,
    )
    surfaces = _spy_surfaces(monkeypatch)
    found = detect(case)
    assert found.flags == ()
    assert found.notes == ("stage 2 skipped: too few band-free pixels",)
    assert_hits_own_lanes(case, found)
    assert len(surfaces) == 1 and np.all(surfaces[0] == estimate_background(case.image))
    assert found.noise == pytest.approx(found.pixel_noise / math.sqrt(15))


def _ramp_offsets(case: RowCase, found: RowDetection, gradient: float, background: float):
    """Each box's true membrane offset from ``background``, band side positive,
    in intensity units: the generator's noise-free ramp averaged under the box."""
    cx = np.array(case.lane_cx)
    xc, half = 0.5 * (cx[0] + cx[-1]), max(0.5 * (cx[-1] - cx[0]), 1.0)
    sign = 1.0 if case.dark_on_light else -1.0
    out = []
    for lane in found.lanes:
        xs = np.arange(lane.rect[0], lane.rect[2], dtype=float)
        level = MEMBRANE + gradient * float(np.mean((xs - xc) / half))
        level = level if case.dark_on_light else FULL_SCALE - level
        out.append(sign * (level - background))
    return out


@pytest.mark.parametrize("dark_on_light", [True, False])
@pytest.mark.parametrize(
    ("seed", "q", "depths"),
    [(57, 90.0, (18000.0, 30000.0)), (57, 99.5, (18000.0, 30000.0)), (80, 98.0, (5000.0, 9000.0))],
)
def test_a_ramp_is_detected_over_its_plane_whatever_the_stored_level(
    monkeypatch, dark_on_light, seed, q, depths
):
    # The membrane ramps by 3000 across the row; the stored background is the
    # q-th percentile of its outer rows, towards the light end, so most of the
    # ramp lies to its band side and none of the stored level's membrane side
    # widens. The plane follows the ramp in both stages: every band is found,
    # stage 2 runs, and each box's bg_offset is the ramp's own offset there.
    case = synthetic_row(
        "ramp",
        "",
        seed,
        gradient=(1500.0, 0.0),
        depth_range=depths,
        light_on_dark=not dark_on_light,
    )
    x0, y0, x1, y1 = case.row
    crop = case.image[y0:y1, x0:x1]
    outer = np.concatenate([crop[:3].ravel(), crop[-3:].ravel()])
    background = float(np.percentile(outer, q if dark_on_light else 100.0 - q))
    surfaces = _spy_surfaces(monkeypatch)
    found = detect_row(case.image, case.row, 6, background=background, dark_on_light=dark_on_light)
    assert_hits_own_lanes(case, found)
    assert found.notes == ()  # stage 2 ran
    assert len(surfaces) == 2 and all(np.ptp(surface) > 2000 for surface in surfaces)
    true = _ramp_offsets(case, found, 1500.0, background)
    for lane, offset in zip(found.lanes, true, strict=True):
        assert lane.bg_offset * found.pixel_noise == pytest.approx(offset, abs=0.5 * NOISE_SIGMA)
    assert found.flags == ("background_mismatch",)


@pytest.mark.parametrize("dark_on_light", [True, False])
@pytest.mark.parametrize(("k", "warns"), [(0.0, False), (6.0, True)])
def test_bg_offset_is_measured_when_stage_2_is_skipped(monkeypatch, dark_on_light, k, warns):
    # A tight box over close bands, the stored level k noise sigmas to the
    # band side: the band-free pixels are too few for stage 2 but enough to
    # read the membrane's level on. Stage 1 detects over the stored level (k =
    # 0) or over a plane the bands pulled towards them (k = 6); either way the
    # membrane lies about 0 in its signal, and the offsets show the stored
    # level where it is (a little short: those pixels lie by the bands' tails).
    case = adversarial_row(
        "tight", 1001, pitch=40.0, w=44.0, mx=2, my=2, light_on_dark=not dark_on_light
    )
    sign = 1.0 if dark_on_light else -1.0
    background = estimate_background(case.image) - sign * k * NOISE_SIGMA
    surfaces = _spy_surfaces(monkeypatch)
    found = detect_row(case.image, case.row, 6, background=background, dark_on_light=dark_on_light)
    assert_hits_own_lanes(case, found)
    assert "stage 2 skipped: too few band-free pixels" in found.notes
    assert bool(np.all(surfaces[0] == background)) is (k == 0.0)
    for lane in found.lanes:
        assert lane.bg_offset == pytest.approx(k * NOISE_SIGMA / found.pixel_noise, abs=1.5)
    assert ("background_mismatch" in found.flags) is warns


def test_an_extent_the_row_box_cuts_is_no_reference_for_the_others():
    # A band cut to 22 px by the box edge beside a complete 60 px band: the cut
    # one's true width is at least 22, so it cannot make the complete one an outlier.
    assert rowdetect._shared_size([22, 60], [10, 10], "max_guarded") == (22, 10, (1,))
    assert rowdetect._shared_size(
        [22, 60], [10, 10], "max_guarded", cut_w=[True, False], cut_h=[False, False]
    ) == (60, 10, ())
    # A complete extent above twice the complete others is still an outlier.
    assert rowdetect._shared_size(
        [22, 60, 23, 21], [10, 10, 10, 10], "max_guarded", cut_w=[False, False, True, False]
    ) == (23, 10, (1,))
    # Of three, the reference is the upper of the two others: one cut extent
    # there leaves the wide one unjudged (the cut band may be as wide).
    assert rowdetect._shared_size(
        [22, 60, 23], [10, 10, 10], "max_guarded", cut_w=[False, False, True]
    ) == (60, 10, ())


@pytest.mark.parametrize("seed", [322, 356])
def test_a_band_the_box_cuts_does_not_shrink_the_boxes_of_complete_bands(seed, monkeypatch):
    # Fuzz rows whose row box cuts one outer band: with the cut extent as a
    # reference, the complete band was flagged and its box shrunk to a miss.
    case = fuzz_row(seed)
    found = detect(case)
    hits = hit_rate(list(found.slots), case.reference).hits
    plain = rowdetect._shared_size
    monkeypatch.setattr(rowdetect, "_shared_size", lambda ws, hs, rule, **_: plain(ws, hs, rule))
    assert hits > hit_rate(list(detect(case).slots), case.reference).hits


def test_the_margin_counts_the_runner_up_at_every_pitch_of_the_best_reading(monkeypatch):
    # Every pitch reads the same lanes; only the second pitch has a close
    # runner-up. Its cost, not the first pitch's distant one, is the margin's.
    seen = []

    def one_reading(pieces, n, box_w, pitch):
        seen.append(pitch)
        k = len(seen)
        second = 2.0 if k == 2 else 10.0 + k
        return rowdetect._Assignment(1.0 + 0.01 * k, second, pitch, ((0, 1), (1, 1)))

    monkeypatch.setattr(rowdetect, "_dp", one_reading)
    monkeypatch.setattr(rowdetect, "PITCH_PRIOR_TOL", 1e12)  # no pitch prior
    best, alt = rowdetect._assign([], 2, 100.0)
    assert best.cost == pytest.approx(1.01)
    assert alt == pytest.approx(2.0)


def test_size_outlier_does_not_set_the_shared_size():
    # Lane 3's band is 30 px tall, the others about 12: above 2x their median.
    case = _adversarial("tall_band", 1000)
    found = detect(case)
    assert found.flags == ("size_outlier",)
    assert any("lane 4:" in note for note in found.notes)  # notes count lanes from 1
    heights = [lane.extent[3] - lane.extent[1] for lane in found.lanes]
    others = heights[:3] + heights[4:]
    assert heights[3] > SIZE_GUARD * np.median(others)
    assert found.size.height == max(others)
    for lane in (0, 1, 2, 4, 5):  # the tall band's own box is too short to hit it
        assert iou(found.slots[lane], case.reference[lane]) >= 0.5
    # The plain maximum lets the outlier set the size, and flags nothing.
    by_max = detect(case, size_rule="max")
    assert by_max.size.height == max(heights)
    assert "size_outlier" not in by_max.flags


@pytest.mark.parametrize(
    ("widths", "heights", "expected"),
    [
        ([22, 82], [12, 12], (22, 12, (1,))),  # the median of two is their mean: 82 < 2 x 52
        ([20, 22, 40, 60], [12] * 4, (40, 12, (3,))),  # 60 > 2 x 22, though not 2 x 31
        ([40, 41, 42, 43], [11, 12, 18, 26], (43, 18, (3,))),  # per dimension
        ([12, 12, 30], [10] * 3, (12, 10, (2,))),  # an odd count, as before
        # An odd count keeps the whole row's median: 49 <= 2 x 32 (not 2 x 23.5),
        # 40 <= 2 x 25, 23 <= 2 x 12, and 25 <= 2 x 13 of five.
        ([15, 32, 49], [12, 11, 10], (49, 12, ())),
        ([10, 25, 40], [10] * 3, (40, 10, ())),
        ([10, 12, 23], [8, 11, 20], (23, 20, ())),
        ([40] * 5, [10, 10, 13, 14, 25], (40, 25, ())),
        ([20, 22, 40, 44], [12] * 4, (44, 12, ())),  # at most twice the others' median
        ([30], [10], (30, 10, ())),  # one extent: nothing to compare it with
    ],
)
def test_size_guard_compares_each_extent_with_the_median_of_the_others(widths, heights, expected):
    assert rowdetect._shared_size(widths, heights, "max_guarded") == expected
    assert rowdetect._shared_size(widths, heights, "max") == (max(widths), max(heights), ())


def test_size_guard_keeps_the_row_median_for_an_odd_count():
    # Of an odd count, an extent is left out exactly when it exceeds twice the
    # median of all (the rule before the others' median); of an even count,
    # exactly when it exceeds twice the median of the others, an odd number.
    rng = np.random.default_rng(51)
    for _ in range(3000):
        v = rng.integers(4, 60, int(rng.integers(2, 9)))
        _, _, outliers = rowdetect._shared_size(v.tolist(), [10] * v.size, "max_guarded")
        if v.size % 2:
            expected = np.flatnonzero(v > SIZE_GUARD * np.median(v))
        else:
            others = [np.median(np.delete(v, k)) for k in range(v.size)]
            expected = np.flatnonzero(v > SIZE_GUARD * np.array(others))
        assert outliers == tuple(int(k) for k in expected), v


def test_an_extent_within_twice_the_row_median_of_three_sets_the_size():
    # Extents 15, 32 and 49 px wide: 49 is within twice the median (32), so it
    # sets the size and nothing is flagged, as before the others' median.
    case = adversarial_row("odd", 1000, n=3, pitch=80.0, widths={0: 16.0, 1: 34.0, 2: 52.0})
    found = detect(case)
    widths = sorted(lane.extent[2] - lane.extent[0] for lane in found.lanes)
    assert widths[2] <= SIZE_GUARD * widths[1] and widths[2] > widths[0] + widths[1]
    assert found.flags == ()
    assert found.size.width == widths[2]


@pytest.mark.parametrize(
    ("recipe", "outlier", "dim"),
    [
        ({"n": 2, "pitch": 130.0, "widths": {0: 22.0, 1: 82.0}}, 1, 0),  # a band beside a smear
        ({"n": 4, "heights": {2: 20.0, 3: 30.0}}, 3, 1),  # 18 px sets the size, 26 does not
    ],
)
def test_an_outlier_among_two_or_four_extents_does_not_set_the_size(recipe, outlier, dim):
    case = adversarial_row("guard", 1000, **recipe)
    found = detect(case)
    assert found.flags == ("size_outlier",)
    assert found.notes == (
        f"lane {outlier + 1}: extent above 2x the median of the other extents, "
        "left out of the shared size",
    )
    sizes = [(e[2] - e[0], e[3] - e[1]) for e in (lane.extent for lane in found.lanes)]
    others = [size[dim] for k, size in enumerate(sizes) if k != outlier]
    assert sizes[outlier][dim] > SIZE_GUARD * np.median(others)
    assert (found.size.width, found.size.height)[dim] == max(others)
    for lane, slot in enumerate(found.slots):
        if lane != outlier:
            assert iou(slot, case.reference[lane]) >= 0.5


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


@pytest.mark.parametrize(("frac", "counted"), [(0.2, False), (0.25, True), (0.3, True)])
def test_a_doublet_s_weaker_band_counts_from_a_quarter_of_the_peak(frac, counted):
    # #121: lane 2's second band is 14 px below the first and 0.2 to 0.3 as
    # deep, joined to it by signal above the noise (a doublet not fully
    # apart): under the extent level of the box, tens of sigmas above its
    # saddle, and wide as a band. It counts only from SECOND_SHARE of the
    # lane's peak.
    case = adversarial_row("doublet", 1000, doublet={2: (14, frac)}, my=12)
    found = detect(case)
    assert found.flags == (("multiple_components",) if counted else ())
    assert components(found) == [1, 1, 1 + counted, 1, 1, 1]
    assert (frac >= SECOND_SHARE) is counted
    if counted:
        assert found.notes == (
            "lane 3: a second separate component reaches 25% of the lane's peak;"
            " the box covers the one with the lane's strongest pixel",
        )


@pytest.mark.parametrize("frac", [0.3, 0.7])
def test_a_band_apart_from_the_band_s_rows_is_no_second_component(frac):
    # #121: 20 px apart, with membrane between them, the second band lies
    # below the band's rows (its signal above the noise): no part of the
    # band, however deep, as JPEG blocks and specks at the box's edge are not.
    # The pair is centred on the row, so the box on the first lies 10 px above
    # the other boxes' line, more than ROW_LINE_K of the 12 px box (#114).
    case = adversarial_row("doublet", 1000, doublet={2: (20, frac)}, my=12)
    found = detect(case)
    assert 10 > ROW_LINE_K * found.size.height
    assert found.flags == ("off_row_line",)
    assert components(found) == [1] * 6


@pytest.mark.parametrize("seed", [1000, 1002, 1003, 1006, 1007])
def test_jpeg_block_noise_is_no_second_component(seed):
    # #121: an 8-bit JPEG export of a row (quality 75, membrane 200, bands 70
    # to 120 levels deep) on a smooth membrane: its 8x8 block artefacts, a
    # level or so deep, reach DETECT_K sigma of the smooth membrane beside
    # the bands and at the box's edges, and were counted as second bands in
    # lanes of each of these rows. They reach about 1% of the lane's peak.
    case = jpeg(adversarial_row("jpeg", seed, noise=100.0, my=10))
    found = detect(case)
    assert_hits_own_lanes(case, found)
    assert found.flags == ()
    assert components(found) == [1] * 6


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_a_dumbbell_band_is_one_band(seed):
    # #121: lane 1's band pale across its middle, its ends about 1.7x as dark:
    # two peaks, but the dip between them stays inside the band's grown
    # extent. One band, boxed whole, and not hollow: nothing is saturated.
    case = _adversarial("dumbbell_band", seed)
    found = detect(case, saturated_at=0.0)
    assert_hits_own_lanes(case, found)
    assert found.flags == ()
    assert components(found) == [1] * 6
    assert not any(lane.hollow for lane in found.lanes)
    x0, _, x1, _ = found.lanes[1].extent
    ref = case.reference[1]
    assert x0 <= ref[0] + 2 and x1 >= ref[2] - 2  # both ends


HOLLOW_NOTE = (
    ": a hollow band, lighter in its centre than the saturated pixels on either side of it:"
    " a sign of over-exposure"
)


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
@pytest.mark.parametrize("light_on_dark", [False, True])
def test_a_hollow_saturated_band_is_over_exposed_not_two_bands(seed, light_on_dark):
    # #121: lane 1's band over-exposed, clipped flat at the limit around its
    # lighter centre (a burnt-out band): one band, reported as hollow, a sign
    # of over-exposure, where the saturation level is known.
    case = _adversarial("hollow_band", seed)
    limit = 0.0
    if light_on_dark:
        case = dataclasses.replace(case, image=FULL_SCALE - case.image, dark_on_light=False)
        limit = FULL_SCALE
    found = detect(case, saturated_at=limit)
    assert_hits_own_lanes(case, found)
    assert found.flags == ("hollow_band",)
    assert components(found) == [1] * 6
    assert [lane.hollow for lane in found.lanes] == [False, True, False, False, False, False]
    # Its halves, split along x by the lighter centre, are merged into one piece.
    assert len(found.notes) == 2 and found.notes[0].startswith("merged pieces at x=")
    assert found.notes[1] == f"lane 2{HOLLOW_NOTE}"
    # With no known saturation level it is one band, not hollow.
    unknown = detect(case)
    assert unknown.flags == ()
    assert not any(lane.hollow for lane in unknown.lanes)
    assert unknown.slots == found.slots


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
@pytest.mark.parametrize("stored", ["16-bit", "light_on_dark", "jpeg"])
def test_a_band_burnt_out_in_its_middle_is_hollow(seed, stored):
    # #121: lane 1's band over-exposed all around a lighter centre (a ring),
    # clipped at the limit on every side of it: one connected peak, no dip
    # between two peaks, yet hollow. An 8-bit JPEG export's saturated pixels
    # lie within 2 levels of 0 (quantify.saturation_level).
    case = _adversarial("hollow_ring", seed)
    limit = 0.0
    if stored == "light_on_dark":
        case = dataclasses.replace(case, image=FULL_SCALE - case.image, dark_on_light=False)
        limit = FULL_SCALE
    elif stored == "jpeg":
        case, limit = jpeg(case), 2.0
    found = detect(case, saturated_at=limit)
    assert_hits_own_lanes(case, found)
    assert found.flags == ("hollow_band",)
    assert components(found) == [1] * 6
    assert [lane.hollow for lane in found.lanes] == [False, True, False, False, False, False]
    assert found.notes[-1] == f"lane 2{HOLLOW_NOTE}"
    unknown = detect(case)
    assert unknown.flags == ()
    assert unknown.slots == found.slots


def burnt_out(depth: float, light: float, seed: int) -> RowCase:
    """Lane 1's band ``depth`` times as deep as the membrane (clipped at 0),
    its centre lightened by ``light`` through its whole height and more: the
    band of hollow_band, lightened 45000 there."""
    blobs = [blob(1, 6.0, -light, ry=12.0)]
    return adversarial_row("burnt_out", seed, depths={1: depth * MEMBRANE}, artefacts=blobs)


# #121: 1.5x as deep lightened by 45000 to 120000, 2x by 60000 to 120000. From
# 60000 (1.5x) or 90000 (2x) on, the centre falls below the extent level and
# splits the band along x. Left out: 2x lightened by 45000, its centre still
# at the limit, and seed 1001's band 1.5x deep lightened by 90000 or more,
# which reaches the limit at fewer than HOLLOW_PIXELS pixels (both below).
BURNT_OUT = [
    (depth, light, seed)
    for depth, lights in ((1.5, range(45000, 120001, 15000)), (2.0, range(60000, 120001, 15000)))
    for light in lights
    for seed in (1000, 1001)
    if not (depth == 1.5 and seed == 1001 and light >= 90000)
]


@pytest.mark.parametrize("light_on_dark", [False, True])
@pytest.mark.parametrize(("depth", "light", "seed"), BURNT_OUT)
def test_a_band_split_by_its_burnt_out_centre_is_one_hollow_band(depth, light, seed, light_on_dark):
    # #121: lane 1's band clipped at the limit on both sides of a centre
    # lighter through its whole height, however light: split along x where
    # the centre falls below the extent level, its halves in the same rows,
    # saturated on either side of the centre. One band, hollow, not two, and
    # its extent and box over both halves: a box over one of them, as over
    # the stronger of two bands, would measure half the band.
    case = burnt_out(depth, light, seed)
    limit = 0.0
    if light_on_dark:
        case = dataclasses.replace(case, image=FULL_SCALE - case.image, dark_on_light=False)
        limit = FULL_SCALE
    found = detect(case, saturated_at=limit)
    assert_hits_own_lanes(case, found)
    assert found.flags == ("hollow_band",)
    assert components(found) == [1] * 6
    assert [lane.hollow for lane in found.lanes] == [False, True, False, False, False, False]
    assert found.notes[-1] == f"lane 2{HOLLOW_NOTE}"
    ref = case.reference[1]
    for x0, _, x1, _ in (found.lanes[1].extent, found.slots[1]):
        assert x0 <= ref[0] + 2 and x1 >= ref[2] - 2  # both halves


def test_a_stroke_across_a_burnt_out_band_s_rows_does_not_join_it():
    # #121: the halves of a saturated band split by a lighter centre (rows 15
    # to 24), and a stroke drawn through the centre from row 2 to 37, at the
    # limit too, as a pen line on a blot. The band's other half shares its
    # rows and joins it; the stroke shares fewer than JOIN_ROWS of the rows it
    # and the band span, and does not. (On a real drag, strokes joined in a
    # chain gave one band a 77 x 37 px extent.)
    s = np.zeros((40, 60))
    saturated = np.zeros(s.shape, bool)
    for c0, c1 in ((5, 25), (36, 56)):
        s[15:25, c0:c1] = 100.0
        saturated[17:23, c0 + 3 : c1 - 3] = True
    s[15:25, 25:36] = 10.0  # the lighter centre
    s[2:38, 30:32] = 100.0
    saturated[2:38, 30:32] = True
    pieces, count = label(s > 30.0)
    assert count == 3
    grown = pieces == pieces[20, 10]
    joined = rowdetect._join_burnt_out(pieces, grown, s, saturated, 6.0, (0, 60))
    assert joined is not None and joined[20, 45]  # the other half
    assert np.flatnonzero(joined.any(axis=1)).tolist() == list(range(15, 25))
    assert np.flatnonzero(joined.any(axis=0)).tolist() == list(range(5, 56))


@pytest.mark.parametrize("seed", [1000, 1001])
def test_a_band_at_the_limit_through_its_centre_is_not_hollow(seed):
    # 2x as deep, lightened by 45000, lane 1's centre still reaches the limit:
    # saturated across, no lighter centre. One band, boxed whole.
    case = burnt_out(2.0, 45000.0, seed)
    found = detect(case, saturated_at=0.0)
    assert_hits_own_lanes(case, found)
    assert found.flags == ()
    assert components(found) == [1] * 6
    ref = case.reference[1]
    x0, _, x1, _ = found.slots[1]
    assert x0 <= ref[0] + 2 and x1 >= ref[2] - 2


@pytest.mark.parametrize("light", [90000.0, 105000.0, 120000.0])
def test_a_band_split_where_it_hardly_reaches_the_limit_is_two_peaks(light):
    # Seed 1001's band, 1.5x as deep, lightened by 90000 or more, reaches the
    # limit at fewer than HOLLOW_PIXELS pixels: not over-exposed by the
    # quantification's count (#112), no saturated pixels to enclose a centre.
    # Its halves are two peaks, as a band's split by a bubble.
    case = burnt_out(1.5, light, 1001)
    assert np.count_nonzero(case.image <= 0.0) < HOLLOW_PIXELS
    found = detect(case, saturated_at=0.0)
    assert found.flags == ("multiple_components",)
    assert components(found) == [1, 2, 1, 1, 1, 1]
    assert not any(lane.hollow for lane in found.lanes)


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
@pytest.mark.parametrize(
    ("left", "right"), [(24000.0, 24000.0), (1.5 * MEMBRANE, 24000.0), (24000.0, 1.5 * MEMBRANE)]
)
def test_two_bands_side_by_side_not_both_saturated_count_as_two(seed, left, right):
    # #121: lane 1 holds two bands side by side, 28 px apart, membrane between
    # them: neither saturated, or one only. No centre with saturated pixels on
    # either side of it: two bands where the saturation level is known too.
    blobs = [blob(1, 5.0, left, ry=3.5, dx=-14.0), blob(1, 5.0, right, ry=3.5, dx=14.0)]
    case = adversarial_row("pair", seed, depths={1: 0.0}, artefacts=blobs)
    found = detect(case, saturated_at=0.0)
    assert found.flags == ("multiple_components",)
    assert components(found) == [1, 2, 1, 1, 1, 1]
    assert not any(lane.hollow for lane in found.lanes)


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
@pytest.mark.parametrize(("frac", "depth"), [(0.5, 1.3 * MEMBRANE), (1.0, 1.6 * MEMBRANE)])
def test_an_over_exposed_close_doublet_is_not_a_hollow_band(seed, frac, depth):
    # #121: lane 5 holds two bands 9 px apart, the upper clipped at 0, the
    # lower half as deep (not saturated) or as deep (saturated too). The
    # lighter rows between them hold no saturated pixel on either side: two
    # bands stacked, not a band lighter in its centre, whatever one grown
    # extent holds (R5). The saturation level changes nothing.
    case = adversarial_row("doublet", seed, doublet={4: (9, frac)}, depths={4: depth})
    assert np.count_nonzero(case.image <= 0.0) >= HOLLOW_PIXELS  # over-exposed
    found = detect(case, saturated_at=0.0)
    assert not any(lane.hollow for lane in found.lanes)
    unknown = detect(case)
    assert (found.flags, found.notes, found.slots) == (unknown.flags, unknown.notes, unknown.slots)
    assert components(found) == components(unknown)


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_a_notch_in_a_saturated_band_s_edge_is_not_hollow(seed):
    # Lane 2's band clipped flat at 0, its two ends rising 6 px above its
    # middle: along the rows above the middle, its lighter edge lies between
    # saturated pixels, but the band is saturated across below it only. A
    # notch in its top edge, not a lighter centre.
    found = detect(_adversarial("notched_band", seed), saturated_at=0.0)
    assert found.flags == ()
    assert not any(lane.hollow for lane in found.lanes)


def test_a_saturated_band_without_a_dip_is_not_hollow():
    # Lane 2 of the bench row is clipped flat at 0: saturated, one peak, no
    # lighter centre. Over-exposure is the quantification's to flag.
    found = detect(BENCH["overexposed"], saturated_at=0.0)
    assert found.flags == ()
    assert not any(lane.hollow for lane in found.lanes)


@pytest.mark.parametrize("value", [math.nan, math.inf, "0"])
def test_a_saturation_level_that_is_not_a_finite_number_is_refused(value):
    with pytest.raises(ValueError, match="saturated_at"):
        detect(BENCH["all_present"], saturated_at=value)


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


# --- #116: lines, strips and edges that are not bands ---


def _rules_off(monkeypatch) -> None:
    """Detection without #116's rules: no line, no piece rising into a side."""
    monkeypatch.setattr(rowdetect, "_lines", lambda s, *_: np.zeros(s.shape, bool))
    monkeypatch.setattr(rowdetect, "_rises_to_side", lambda *_: False)


@pytest.mark.parametrize(
    ("dy", "px"),
    [(16, 2), (16, 1), (16, 4), (10, 2)],
    ids=["frame", "hairline", "thick-line", "close-to-the-bands"],
)
def test_frame_lines_across_the_row_are_no_bands(monkeypatch, dy, px):
    # The lines cross every gap between the lanes: taken out, they leave each
    # band its box and the empty lane unmeasured (a band may lie under a
    # line), as the same row without the frame is boxed.
    case = framed(dy, px)
    found = detect(case)
    assert_hits_own_lanes(case, found)
    assert found.lanes[2].reason == "line"
    unframed = adversarial_row("unframed", 1000, missing=[2])
    assert found.size == detect(unframed).size
    assert "multiple_components" not in found.flags
    # Without the rule the frame takes boxes: the empty lane's, on a line.
    _rules_off(monkeypatch)
    off = detect(case)
    assert off.lanes[2].rect is not None and off.size != found.size


def test_a_frame_around_empty_lanes_is_no_band():
    # Each of the frame's lines lies beside the other and ends where it ends
    # (neither holds its level past the other's ends, as a frame's line does
    # past a row of thin bands): a box over a frame and no band finds nothing.
    case = adversarial_row(
        "empty frame",
        1000,
        missing=range(6),
        artefacts=[frame(16, 2, 20000.0)],
        box_adjust=(-30, -12, 30, 12),
    )
    found = detect(case)
    assert found.size is None
    assert [lane.reason for lane in found.lanes] == ["line"] * 6


@pytest.mark.parametrize("degrees", [0.75, 1.5, 2.0])
def test_a_frame_turned_a_little_is_no_band(degrees):
    # A scan turned by a degree or two: the frame's lines drift 2 to 6 px
    # across the rows over two pitches, and still hold their level within
    # half a line's height up or down.
    case = adversarial_row(
        "tilted frame",
        1000,
        missing=[2],
        artefacts=[frame(16, 2, 20000.0, slope=math.tan(math.radians(degrees)))],
        box_adjust=(-30, -12, 30, 12),
    )
    found = detect(case)
    assert_hits_own_lanes(case, found)
    assert found.lanes[2].reason == "line"


@pytest.mark.parametrize(
    ("dy", "missing"), [(10, [0, 5]), (9, [2])], ids=["end-lanes-empty", "closer"]
)
def test_a_hairline_just_past_the_bands_is_no_band(dy, missing):
    # A 1 px frame line 3-4 px past the bands' 20% extents: where a band's
    # tail crosses it, the tail lifts its edge row and breaks it over the
    # band's width, yet it runs on across the lanes. The empty lanes are not
    # boxed on it (the piece under the bands' tails stays with them).
    case = adversarial_row(
        "hairline",
        1000,
        missing=missing,
        artefacts=[frame(dy, 1, 20000.0)],
        box_adjust=(-30, -(dy - 4), 30, dy - 4),
    )
    found = detect(case)
    assert_hits_own_lanes(case, found)
    assert [found.lanes[lane].reason for lane in missing] == ["line"] * len(missing)
    unframed = adversarial_row("unframed", 1000, missing=missing)
    assert found.size == detect(unframed).size


def test_a_dark_strip_along_the_image_edge_is_no_row_of_bands(monkeypatch):
    # A screenshot's toolbar along the image's bottom, darkest at its top: a
    # box over it finds nothing, where without the rule the strip was cut
    # into one box per lane.
    case = adversarial_row(
        "strip",
        1000,
        missing=range(6),
        artefacts=[bottom_strip(14, 20000.0)],
        box_adjust=(0, 52, 0, 80),
    )
    assert case.row[3] == case.image.shape[0]  # the box reaches the image's bottom row
    found = detect(case)
    assert found.size is None
    assert [lane.reason for lane in found.lanes] == ["line"] * 6
    _rules_off(monkeypatch)
    assert detect(case).slots.count(None) == 0


def test_bands_above_a_dark_strip_keep_their_boxes():
    case = adversarial_row(
        "strip row",
        1000,
        missing=[2],
        artefacts=[bottom_strip(14, 20000.0)],
        box_adjust=(0, 0, 0, 80),
    )
    found = detect(case)
    assert_hits_own_lanes(case, found)
    assert found.lanes[2].reason == "line"


def test_a_dark_image_edge_on_a_faint_row_is_no_band(monkeypatch):
    # Bands too faint to detect, and the image darkening into its right edge,
    # which the box reaches: nothing is boxed, where without the rule the edge
    # was boxed as the last lane's band (and the other lanes' slots rested on
    # that one box).
    case = adversarial_row(
        "edge",
        1000,
        depth_range=(300.0, 300.0),
        artefacts=[dark_edge(30, 1500.0)],
        box_adjust=(0, 0, 200, 0),
    )
    assert case.row[2] == case.image.shape[1]  # the box reaches the image's right edge
    found = detect(case)
    assert found.size is None and found.lanes[-1].reason != "band"
    _rules_off(monkeypatch)
    off = detect(case)
    assert off.slots[:-1] == (None,) * 5 and off.slots[-1][2] == case.row[2]


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_band_the_box_side_cuts_past_its_centre_is_read_but_not_boxed(side):
    # The box's side edge runs 8 px beyond the end band's centre: its piece
    # rises into the edge, so its lane is read (the others keep their lanes)
    # but gets no box, and the box leaves out most of that lane.
    base = adversarial_row("side", 1000)
    x0, y0, x1, y1 = base.row
    if side == "left":
        lane, row = 0, (math.ceil(base.lane_cx[0]) + 8, y0, x1, y1)
    else:
        lane, row = 5, (x0, y0, math.floor(base.lane_cx[-1]) - 8, y1)
    found = detect(dataclasses.replace(base, row=row))
    assert (found.lanes[lane].reason, found.lanes[lane].rect) == ("side_signal", None)
    assert found.flags == ("lanes_outside_row",)
    for other in set(range(6)) - {lane}:
        assert iou(found.lanes[other].rect, base.reference[other]) >= 0.5


@pytest.mark.parametrize(
    ("depth", "dx"),
    [(None, 5), (None, 0), (80000.0, 0)],
    ids=["peak-inside", "edge-at-centre", "saturated-top-to-the-edge"],
)
def test_a_band_the_box_side_only_trims_keeps_its_box(depth, dx):
    # A band whose top lies inside the box, or runs on inside it (flat-topped,
    # or clipped flat by saturation), however the edge trims it.
    base = adversarial_row("side", 1000, depths=None if depth is None else {5: depth})
    x0, y0, _, y1 = base.row
    found = detect(dataclasses.replace(base, row=(x0, y0, math.floor(base.lane_cx[-1]) + dx, y1)))
    assert [lane.reason for lane in found.lanes] == ["band"] * 6


def test_rising_into_a_side_is_read_on_the_profile_at_the_edge():
    ramp = np.linspace(0.0, 10.0, 40)
    assert rowdetect._rises_to_side(ramp, False, 10)
    assert rowdetect._rises_to_side(ramp[::-1], True, 10)
    assert not rowdetect._rises_to_side(ramp, True, 10)  # it peaks at the far end
    plateau = np.concatenate([np.linspace(0.0, 10.0, 10), np.full(30, 10.0)])
    assert not rowdetect._rises_to_side(plateau, False, 10)  # a flat top runs on inside
    hump = np.concatenate([np.linspace(0.0, 10.0, 20), np.linspace(10.0, 9.0, 5)])
    assert not rowdetect._rises_to_side(hump, False, 10)  # it peaks inside


_TOUCHING = {"pitch": 48.0, "w": 60.0, "depth_range": (22000.0, 26000.0)}
_SATURATED_TOUCHING = {"pitch": 48.0, "w": 60.0, "depth_range": (75000.0, 80000.0)}

# Rows whose bands run together along x, flat or nearly (no dip between them,
# or saturated), thick or thin, alone in the box or with the image's top or
# bottom edge a few px past them or through them: #116's rules leave them
# exactly as they were.
UNCHANGED_ROWS = {
    "touching": BENCH["touching"],
    "overexposed": BENCH["overexposed"],
    "touch12": adversarial_row("touch12", 1036, n=12, pitch=40.0, w=44.0, h=12.0, mx=4, my=3),
    "touching_weak_end": adversarial("touching_weak_end", 1000),
    "saturated_touching": adversarial_row(
        "saturated touching", 1000, pitch=48.0, w=60.0, depth_range=(75000.0, 80000.0)
    ),
    "saturated_touching_thin": adversarial_row(
        "thin saturated touching", 1000, pitch=48.0, w=60.0, h=6.0, depth_range=(75000.0, 80000.0)
    ),
    "saturated_filling_the_box": adversarial_row(
        "thick", 1000, h=30.0, pitch=48.0, w=60.0, depth_range=(75000.0, 80000.0), my=0
    ),
    "bloom": adversarial_row("bloom", 1000, depths={2: 80000.0}, widths={2: 62.0}, missing=[3]),
    "thin_touching": adversarial_row(
        "thin touching", 1000, n=9, pitch=15.0, w=15.0, h=4.0, mx=3, my=4
    ),
    "scratch_through_the_bands": adversarial_row("scratch", 1000, artefacts=[hstripe(0.5, 6000.0)]),
    # Thin touching bands (a 20% extent of 4-5 px) with no dip between them,
    # flat along x and no higher than a drawn line: alone in the box, they are
    # the row, not a line beside it.
    "thin_touching_no_dip": adversarial_row("thin", 1000, h=5.0, mx=4, **_TOUCHING),
    "thin_touching_twelve": adversarial_row(
        "thin", 1000, n=12, pitch=30.0, w=38.0, h=5.0, mx=4, my=6
    ),
    "thin_saturated_touching_no_dip": adversarial_row(
        "thin", 1000, h=4.0, mx=4, **_SATURATED_TOUCHING
    ),
    # The image cropped 3 to 10 px past the bands' centres, the box dragged to
    # its edge: the bands' tails, flat along that edge, or the bands
    # themselves, cut there, are no strip along the image's edge.
    "cropped_below_saturated_touching": image_cut(
        adversarial_row("crop", 1002, **_SATURATED_TOUCHING), bottom=90
    ),
    "cropped_above_saturated_touching": image_cut(
        adversarial_row("crop", 1001, **_SATURATED_TOUCHING), top=72
    ),
    "cropped_below_touching": image_cut(adversarial_row("crop", 1001, **_TOUCHING), bottom=87),
    "cut_through_by_the_image": image_cut(adversarial_row("crop", 1000, **_TOUCHING), bottom=83),
}


@pytest.mark.parametrize("name", UNCHANGED_ROWS)
def test_touching_and_saturated_rows_are_unchanged(monkeypatch, name):
    case = UNCHANGED_ROWS[name]
    found = detect(case)
    _rules_off(monkeypatch)
    assert detect(case) == found


@pytest.mark.parametrize(
    ("up", "down"), [(10, 0), (0, 10), (10, 10)], ids=["line-above", "line-below", "both-lines"]
)
def test_thin_touching_bands_beside_a_frame_line_keep_their_boxes(up, down):
    # Thin touching bands (flat along x, no higher than a drawn line) in a
    # panel's frame, the box inside its sides reaching up or down to its
    # lines: each lies beside the other, but a line runs on across the box
    # past the row's end shoulders, where the row ends inside it. The lines
    # are taken out; the bands keep their boxes, as without the frame.
    case = adversarial_row(
        "thin framed",
        1000,
        h=5.0,
        mx=4,
        artefacts=[frame(12, 2, 20000.0)],
        box_adjust=(0, -up, 0, down),
        **_TOUCHING,
    )
    found = detect(case)
    assert_hits_own_lanes(case, found)
    assert [lane.reason for lane in found.lanes] == ["band"] * 6
    alone = detect(UNCHANGED_ROWS["thin_touching_no_dip"])
    assert found.size == alone.size


@pytest.mark.parametrize(
    ("seed", "shade", "bands"),
    [
        (1000, (105, 30, 6000.0), {}),
        (1001, (105, 30, 6000.0), _TOUCHING),
        (1002, (110, 40, 10000.0), {}),
    ],
    ids=["separate", "touching", "deeper"],
)
def test_bands_above_an_image_edge_darkening_into_it_keep_their_boxes(
    monkeypatch, seed, shade, bands
):
    # The membrane darkens into the image's bottom edge, 25-30 px below the
    # bands, and the box runs to it: the shade, flat along that edge, is a
    # strip there, but the bands above its valley are no part of it.
    case = image_cut(
        adversarial_row("vignette", seed, artefacts=[shade_above(*shade)], **bands), bottom=shade[0]
    )
    found = detect(case)
    assert [lane.reason for lane in found.lanes] == ["band"] * 6
    _rules_off(monkeypatch)
    off = detect(case)
    assert (found.slots, found.flags) == (off.slots, off.flags)


def _side_cut() -> RowCase:
    base = adversarial_row("side", 1000)
    x0, y0, _, y1 = base.row
    return dataclasses.replace(base, row=(x0, y0, math.floor(base.lane_cx[-1]) - 8, y1))


# The rows #116's rules act on: lines, strips, the image's edge, a side cut.
RULE_ROWS = {
    "frame": lambda: framed(16, 2),
    "strip": lambda: adversarial_row(
        "strip",
        1000,
        missing=range(6),
        artefacts=[bottom_strip(14, 20000.0)],
        box_adjust=(0, 52, 0, 80),
    ),
    "strip_row": lambda: adversarial_row(
        "strip row",
        1000,
        missing=[2],
        artefacts=[bottom_strip(14, 20000.0)],
        box_adjust=(0, 0, 0, 80),
    ),
    "dark_edge": lambda: adversarial_row(
        "edge",
        1000,
        depth_range=(300.0, 300.0),
        artefacts=[dark_edge(30, 1500.0)],
        box_adjust=(0, 0, 200, 0),
    ),
    "side_cut": _side_cut,
}


@pytest.mark.parametrize("name", RULE_ROWS)
def test_rule_rows_keep_the_invariants_and_polarity_symmetry(name):
    case = RULE_ROWS[name]()
    found = detect(case)
    check_invariants(case, found)
    inverse = detect_row(
        FULL_SCALE - case.image,
        case.row,
        case.n_lanes,
        background=FULL_SCALE - estimate_background(case.image),
        dark_on_light=not case.dark_on_light,
    )
    assert (inverse.slots, inverse.flags) == (found.slots, found.flags)
    assert [lane.reason for lane in inverse.lanes] == [lane.reason for lane in found.lanes]


@pytest.mark.parametrize(
    ("name", "value"), [("LINE_SPAN", 100.0), ("LINE_FLAT", 1.01), ("LINE_PX", 0)]
)
def test_each_line_setting_takes_part(monkeypatch, name, value):
    case = framed(16, 2)
    found = detect(case)
    monkeypatch.setattr(rowdetect, name, value)
    assert settings()[name.lower()] == value
    assert detect(case) != found


# --- What the result reports ---


def test_noise_is_measured_on_the_band_free_pixels():
    # Stage 2 refits the plane and the noise on the band-free pixels of the
    # row; a stage-1 estimate or plane is 1-6% off these values.
    found = _detected("all_present")
    assert found.noise == pytest.approx(121.46200102625718, rel=1e-4)
    assert found.pixel_noise == pytest.approx(453.93837046984686, rel=1e-4)
    # White noise: the detection noise is the pixel noise over the 3x5 kernel.
    assert found.noise == pytest.approx(found.pixel_noise / math.sqrt(15), rel=0.1)


def _guard(x0: int, y0: int, x1: int, y1: int, fy: float, fx: float) -> tuple[slice, slice]:
    """A rect dilated by (fy, fx) of its own size, at least BG_GUARD_MIN px."""
    gy = max(BG_GUARD_MIN, math.ceil(fy * (y1 - y0)))
    gx = max(BG_GUARD_MIN, math.ceil(fx * (x1 - x0)))
    return slice(max(0, y0 - gy), y1 + gy), slice(max(0, x0 - gx), x1 + gx)


@pytest.mark.parametrize(
    ("case", "note", "leaked"),
    [
        # A weak extra band between lanes 3 and 4, dropped for want of a lane:
        # a component of its own, whose 266 pixels stage 2 used to fit.
        (
            adversarial_row(
                "weak_band", 1000, pitch=100.0, artefacts=[band_between(2, 20.0, 12.0, 3000.0)]
            ),
            "dropped a weak piece at x=332..352",
            266,
        ),
        # Dust there, dropped too, joins those lanes' bands into one component
        # (259 pixels used to be fitted).
        (_adversarial("blob_gap", 1003), "dropped a weak piece at x=256..277", 259),
    ],
    ids=["weak_band", "dust"],
)
def test_stage_2_background_leaves_out_every_kept_component(monkeypatch, case, note, leaked):
    # Every kept pixel stays out of the stage-2 plane and noise: inside a
    # band's extent guarded by BG_GUARD of its size, or in a part outside those
    # guarded by BG_GUARD_MIN px (a kept component already reaches down to the
    # noise, tails and all).
    seen = []
    real = rowdetect._stage2_free

    def spy(res, shape):
        free = real(res, shape)
        seen.append((res, free))
        return free

    monkeypatch.setattr(rowdetect, "_stage2_free", spy)
    found = detect(case)
    ((res, free),) = seen
    assert res.notes == [note]  # stage 1 dropped the piece
    assert "stage 2 skipped: too few band-free pixels" not in found.notes
    kept = res.cand.kept
    assert not (free & kept).any()
    bands = np.zeros(free.shape, bool)
    for lane in res.lanes:
        if lane.rect is not None:
            bands[_guard(*lane.rect, *BG_GUARD)] = True
    assert not (free & bands).any()
    rest, _ = label(kept & ~bands)
    assert np.bincount(rest.ravel())[1:].max() >= leaked // 2  # the dropped piece's part
    for rows, cols in find_objects(rest):
        assert not free[_guard(cols.start, rows.start, cols.stop, rows.stop, 0.0, 0.0)].any()


@pytest.mark.parametrize("shift", [0, -6, 5])  # the same centre, out of order, closer than a band
def test_extents_closer_than_a_band_refuse_the_row(monkeypatch, shift):
    # Whatever the growth gives, two lanes' extents at one centre, crossed, or
    # closer than the narrowest band do not show which lane each band is in:
    # the row is refused, and the boxes are not shrunk to slivers between them.
    real = rowdetect._measure

    def crowd(lanes, *args):
        real(lanes, *args)
        a, b = lanes[2].rect, lanes[3].rect
        x = (a[0] + a[2]) // 2 + shift - (b[2] - b[0]) // 2
        lanes[3].rect = (x, b[1], x + b[2] - b[0], b[3])

    monkeypatch.setattr(rowdetect, "_measure", crowd)
    case = BENCH["all_present"]
    found = detect(case)
    assert found.flags == ("ambiguous_lanes",)
    assert found.refused
    assert any(note.startswith("lanes 3, 4: extents closer than") for note in found.notes)
    assert found.size.width >= rowdetect.MIN_WIDTH_PX
    check_invariants(case, found)


def _peaks_reference(ks: np.ndarray, h: float) -> list[tuple[int, int]]:
    """The rule of :func:`rowdetect._peaks` as first written: one labelling of
    the box per candidate peak."""
    rows, cols = np.flatnonzero(ks.any(axis=1)), np.flatnonzero(ks.any(axis=0))
    if rows.size == 0:
        return []
    y0, x0 = int(rows[0]), int(cols[0])
    box = ks[y0 : int(rows[-1]) + 1, x0 : int(cols[-1]) + 1]
    if float(box.max()) < h:
        return []
    eps = h * 1e-9
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    rec = reconstruction(box - h, box, method="dilation", footprint=cross)
    lab, count = label(box - rec >= h - eps)
    tops = [(int(y), int(x)) for y, x in maximum_position(box, lab, np.arange(1, count + 1))]
    tops.sort(key=lambda p: (-float(box[p]), p))
    found = tops[:1]
    for top in tops[1:]:
        height = float(box[top])
        region, _ = label(box > min(height - h, rowdetect.VALLEY_FRAC * height) + eps)
        joined = region == region[top]
        if float(box[joined].max()) <= height + eps and not any(joined[p] for p in found):
            found.append(top)
    return [(y + y0, x + x0) for y, x in found]


def _peak_field(seed: int) -> tuple[np.ndarray, float]:
    """A random field >= 0 with many peaks: smooth, box-smoothed, quantised
    (plateaus and exact ties), bumps on a plateau, or integer noise."""
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(3, 40)), int(rng.integers(3, 90)))
    noise = rng.normal(0.0, 1.0, shape)
    kind = seed % 5
    if kind == 0:
        field = gaussian_filter(noise, float(rng.uniform(0.5, 4.0)))
    elif kind == 1:
        field = uniform_filter(noise, size=SMOOTH)
    elif kind == 2:
        field = np.round(gaussian_filter(noise, float(rng.uniform(0.7, 3.0))) * 4.0)
    elif kind == 3:
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        field = 0.05 * noise
        for _ in range(int(rng.integers(1, 12))):
            cy, cx, r = rng.uniform(0, shape[0]), rng.uniform(0, shape[1]), rng.uniform(1.0, 6.0)
            field = field + rng.uniform(0.5, 5.0) * np.exp(
                -0.5 * (((yy - cy) / r) ** 2 + ((xx - cx) / r) ** 2)
            )
    else:
        field = rng.integers(0, 6, shape).astype(float)
    ks = np.pad(np.maximum(field - np.quantile(field, rng.uniform(0.0, 0.8)), 0.0), (2, 1))
    return ks, max(float(ks.max()) * float(rng.uniform(0.02, 0.6)), 1e-6)


@pytest.mark.parametrize("seed", range(60))
def test_peaks_are_the_per_peak_labelling_rule(seed):
    ks, h = _peak_field(seed)
    assert rowdetect._peaks(ks, h) == _peaks_reference(ks, h)


def test_peaks_cost_does_not_grow_with_the_number_of_peaks():
    # A plateau carrying about 10 bumps or 50 times as many: labelling the box
    # once per peak costs 15x more for the second; the saddle graph of the
    # peaks under 2x.
    yy, xx = np.mgrid[0:200, 0:600].astype(float)
    rng = np.random.default_rng(85)

    def bumps(spacing: float) -> np.ndarray:
        field = np.full(yy.shape, 100.0)
        for cy in np.arange(spacing / 2, 200, spacing):
            for cx in np.arange(spacing / 2, 600, spacing):
                depth = rng.uniform(30.0, 60.0)
                field += depth * np.exp(-0.5 * (((yy - cy) / 2.5) ** 2 + ((xx - cx) / 2.5) ** 2))
        return field

    few, many = bumps(100.0), bumps(12.0)
    assert len(rowdetect._peaks(few, 10.0)) <= 12 and len(rowdetect._peaks(many, 10.0)) > 500

    def cost(field: np.ndarray) -> float:
        best = math.inf
        for _ in range(3):
            start = time.perf_counter()
            rowdetect._peaks(field, 10.0)
            best = min(best, time.perf_counter() - start)
        return best

    assert cost(many) < 5 * cost(few)


def test_pitch_search_evaluates_each_pitch_once(monkeypatch):
    # The fine pass skips the coarse fraction it is centred on.
    fractions: list[float] = []
    real = rowdetect._dp

    def spy(pieces, n, box_w, pitch):
        fractions.append(pitch / (box_w / n))
        return real(pieces, n, box_w, pitch)

    monkeypatch.setattr(rowdetect, "_dp", spy)
    pieces = [rowdetect._Piece(10.0 + 70.0 * i, 54.0 + 70.0 * i, 1.0, 44.0) for i in range(6)]
    best, _ = rowdetect._assign(pieces, 6, 430.0)
    assert best is not None and best.lanes == tuple((i, 1) for i in range(6))
    assert len(fractions) > len(np.arange(0.55, 1.30 + 1e-9, 0.05))  # a fine pass ran
    assert len({round(f, 9) for f in fractions}) == len(fractions)


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
        if lane.rect is not None:
            assert lane.window is None  # only an empty lane's slot is reported
        elif lane.window is None:
            assert lane.snr == 0.0  # not measured
        else:  # a non-empty slot inside the clipped row
            assert all(type(v) is int for v in lane.window)
            wx0, wy0, wx1, wy1 = lane.window
            assert rx0 <= wx0 < wx1 <= rx1 and ry0 <= wy0 < wy1 <= ry1
        if lane.cut and lane.extent is not None:  # it reaches the top or bottom edge
            assert lane.extent[1] == ry0 or lane.extent[3] == ry1
        elif lane.cut:  # an empty lane: its band peaks on the edge row
            assert lane.reason == "edge_signal"
        assert math.isfinite(lane.snr) and math.isfinite(lane.expected_x)
    assert ("cut_by_row_box" in found.flags) == any(lane.cut for lane in found.lanes)
    assert ("multiple_components" in found.flags) == any(
        lane.components > 1 for lane in found.lanes
    )
    assert ("hollow_band" in found.flags) == any(lane.hollow for lane in found.lanes)
    assert all(lane.rect is not None for lane in found.lanes if lane.hollow)
    assert math.isfinite(found.membrane_shift)
    # Lanes recorded as not detected off the row's line: read on it, below
    # the detection limit, with their window, the row placed.
    recorded = [lane for lane in found.lanes if lane.reason == "off_expected_row"]
    assert ("off_expected_row" in found.flags) == bool(recorded)
    for lane in recorded:
        assert lane.window is not None and lane.line_reason == "no_band"
        assert lane.snr == lane.line_snr < DETECT_K and abs(lane.line_offset) > ROW_LINE_K
        assert abs(lane.expected_offset) > ROW_LINE_K
    if recorded:
        assert not found.refused and found.off_cause is None
        assert 2 * len(recorded) < len(recorded) + len(rects)
        assert all(abs(lane.expected_offset) <= ROW_LINE_K for lane in found.lanes if lane.rect)
    assert (found.off_cause == "again") == bool(found.again)


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
        assert dataclasses.replace(after, expected_x=0.0, peaks=()) == dataclasses.replace(
            before,
            rect=move(before.rect),
            extent=move(before.extent),
            window=move(before.window),
            expected_x=0.0,
            peaks=(),
        )
        # The peaks' continuous positions move by the same, up to the last bit
        # of a float that holds a larger coordinate (#58).
        assert len(after.peaks) == len(before.peaks)
        for a, b in zip(after.peaks, before.peaks, strict=True):
            assert (a.y - dy, a.x - dx) == pytest.approx((b.y, b.x), rel=0, abs=1e-9)
            assert a[2:] == b[2:]
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
    assert [lane.window for lane in inverse.lanes] == [lane.window for lane in found.lanes]
    assert components(inverse) == components(found)
    assert [lane.cut for lane in inverse.lanes] == [lane.cut for lane in found.lanes]
    assert inverse.membrane_shift == pytest.approx(found.membrane_shift, abs=1e-6)
    for a, b in zip(inverse.lanes, found.lanes, strict=True):
        assert (a.bg_offset is None) == (b.bg_offset is None)
        if b.bg_offset is not None:
            assert a.bg_offset == pytest.approx(b.bg_offset, abs=1e-6)


@pytest.mark.parametrize(
    ("name", "renumbered"),
    [
        ("missing_first", {}),
        ("tall_band/1000", {"lane 4:": "lane 3:"}),  # size_outlier
        ("doublet_deep/1000", {"lane 3:": "lane 4:"}),  # multiple_components
        ("tilt_cut/1000", {"lanes 1, 2:": "lanes 5, 6:"}),  # cut_by_row_box
    ],
)
def test_lanes_numbered_right_to_left_are_the_same_reading_reversed(name, renumbered):
    # Only the numbering changes: lane 0 is the one at the box's right end, in
    # the lanes and in the notes.
    found = _detected(name)
    back = detect(_case(name), right_to_left=True)
    n = len(found.lanes)
    assert back.lanes == tuple(
        dataclasses.replace(lane, lane=n - 1 - lane.lane) for lane in reversed(found.lanes)
    )
    assert dataclasses.replace(back, lanes=(), notes=()) == dataclasses.replace(
        found, lanes=(), notes=()
    )
    for old in renumbered:
        assert any(note.startswith(old) for note in found.notes)
    assert back.notes == tuple(
        next(
            (new + note[len(old) :] for old, new in renumbered.items() if note.startswith(old)),
            note,
        )
        for note in found.notes
    )


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
        (np.zeros((40, 60, 3)), (0, 0, 60, 40), 2, "invalid_image"),  # not 2-D
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


def test_non_finite_pixels_are_an_invalid_image():
    image = IMAGE.copy()
    image[20, 30] = np.nan
    with pytest.raises(RowDetectError) as err:
        detect_row(image, (0, 0, 60, 40), 2, background=1000.0)
    assert err.value.code == "invalid_image"
    # A NaN outside the row does not matter.
    assert detect_row(image, (40, 0, 60, 40), 2, background=1000.0).slots == (None, None)


@pytest.mark.parametrize("background", [math.nan, math.inf, -math.inf, None, "1000", True])
def test_a_background_that_is_not_a_finite_number_is_an_invalid_image(background):
    image = IMAGE.copy()
    image[1, 1] = 0.0
    for row in ((0, 0, 60, 40), (0, 0, 4, 3)):
        with pytest.raises(RowDetectError) as err:
            detect_row(image, row, 2, background=background)
        assert err.value.code == "invalid_image"
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
        ("min_fit", 2000, "missing_last"),
        ("q_window", -1, "touching"),
        ("gap_search", 0, "missing_middle"),
        ("despeckle_min", 99, "all_present"),
        ("envelope_min", 60, "touching_weak_end/1000"),
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


# --- #114: the boxes of a row lie on one line ---


def off_line(found: RowDetection) -> list[int]:
    """The lanes whose box lies more than ROW_LINE_K box heights off the row's line."""
    return [
        lane.lane
        for lane in found.lanes
        if lane.line_offset is not None and abs(lane.line_offset) > ROW_LINE_K
    ]


def in_row_above(found: RowDetection, case: RowCase) -> list[int]:
    """The lanes whose box lies on the row 40 px above the case's."""
    return [
        lane.lane
        for lane in found.lanes
        if lane.rect is not None
        and (lane.rect[1] + lane.rect[3]) / 2 < case.lane_cy[lane.lane] - 20
    ]


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_a_row_box_over_two_rows_is_refused(seed):
    # The strongest band of lanes 0-2 lies in the row above, of lanes 3-5 in
    # the row's own: the boxes lie on two rows, a row's height apart, as a
    # loading control's row box dragged over its target's row too put them.
    case = two_rows(seed, {0: 2.0, 1: 2.0, 2: 2.0}, rel=0.5)
    found = detect(case)
    assert in_row_above(found, case) == [0, 1, 2]
    assert found.refused and "off_row_line" in found.flags
    # The lanes named are one of the two rows' (three and three: either).
    off = off_line(found)
    assert off in ([0, 1, 2], [3, 4, 5])
    assert found.flags[0] == "off_row_line"  # a refusing flag: first
    [note] = [note for note in found.notes if "row's line" in note]
    assert note.startswith(f"{lanes_phrase(off)}: box centre more than {ROW_LINE_K:g} box height")


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_a_row_box_over_two_rows_names_the_lanes_on_the_other_row(seed):
    # Lanes 4 and 5 hold their strongest band in the row above: those two
    # boxes lie off the line of the other four.
    case = two_rows(seed, {4: 2.0, 5: 2.0}, rel=0.5)
    found = detect(case)
    assert in_row_above(found, case) == off_line(found) == [4, 5]
    assert found.refused
    assert all(found.lanes[lane].line_offset < -3 for lane in (4, 5))  # above: negative


@pytest.mark.parametrize(
    ("lane", "shift"),
    [(2, -14.0), (2, 14.0), (3, -14.0), (0, -18.0), (5, 18.0)],
)
def test_a_lane_whose_band_lies_off_the_row_is_refused(lane, shift):
    # A montage's panel (or a mark beside the row) puts one lane's band
    # 14-18 px (1.2-1.5 box heights) above or below the others'.
    case = adversarial_row("montage", 1000, shifts={lane: shift})
    found = detect(case)
    assert found.size.height == 12
    assert found.refused and found.flags == ("off_row_line",)
    assert off_line(found) == [lane]
    assert np.sign(found.lanes[lane].line_offset) == np.sign(shift)
    assert found.notes == (
        f"lane {lane + 1}: box centre more than {ROW_LINE_K:g} box height (9 px) off the"
        " row's line through the other boxes",
    )
    # The others lie on the line through them.
    assert all(abs(ld.line_offset) < 0.1 for ld in found.lanes if ld.lane != lane)


def test_a_lane_half_a_box_height_off_the_row_is_placed():
    case = adversarial_row("montage", 1000, shifts={2: -6.0})
    found = detect(case)
    assert not found.refused and found.flags == ()
    assert 0.3 < -found.lanes[2].line_offset < ROW_LINE_K


def test_the_lanes_off_the_line_are_numbered_as_the_caller_numbers_them():
    # Read right to left, lane 1 (from the left) is lane 4 (index 4) of six.
    case = adversarial_row("montage", 1000, shifts={1: -14.0})
    found = detect(case, right_to_left=True)
    assert off_line(found) == [4]
    assert found.notes[0].startswith("lane 5: box centre")
    ltr = detect(case)
    assert found.lanes[4].line_offset == ltr.lanes[1].line_offset


SMILES_AND_TILTS = {
    "bench smile (8 px)": BENCH["smile"],
    **{f"smile_tall_tight/{seed}": _adversarial("smile_tall_tight", seed) for seed in (1000, 1002)},
    **{f"tilt_cut/{seed}": _adversarial("tilt_cut", seed) for seed in (1000, 1001)},
    # The accuracy judge's strongest smile: 20 px (1.8 box heights) across, of
    # which four boxes show it.
    **{
        f"strong smile/{seed}": adversarial_row("strong smile", seed, smile=20.0, missing=[1, 4])
        for seed in (1000, 1001, 1002)
    },
    "frown (20 px)": adversarial_row("frown", 1000, smile=-20.0),
    "tilt (14 px)": adversarial_row("tilted", 1000, tilt=14.0, my=8),
    "steep tilt (40 px)": adversarial_row("steep tilt", 1000, tilt=40.0, my=25),
}


@pytest.mark.parametrize("name", SMILES_AND_TILTS)
def test_smiles_and_tilts_lie_on_the_row_line(name):
    # The row's line bends with a smile and tilts with the row: no box of
    # these lies near ROW_LINE_K off it (the bench's largest: 0.19).
    case = SMILES_AND_TILTS[name]
    found = detect(case)
    assert "off_row_line" not in found.flags and not found.refused
    offsets = [abs(lane.line_offset) for lane in found.lanes if lane.line_offset is not None]
    assert len(offsets) >= ROW_LINE_MIN
    assert max(offsets) < ROW_LINE_K / 3


@pytest.mark.parametrize("seed", range(0, 300, 6))
def test_fuzz_rows_lie_on_their_row_line(seed):
    # Random geometry (smiles up to 12 px, a tall band, tight edges, a lane
    # count off by one): the bands lie on one row, and no box off it.
    found = detect(fuzz_row(seed))
    assert "off_row_line" not in found.flags


def test_three_boxes_are_not_checked():
    # Three centres lie on some smile: a row of three boxes shows no line.
    case = adversarial_row("montage", 1000, shifts={2: -14.0}, missing=[1, 4, 5])
    found = detect(case)
    assert [lane.lane for lane in found.lanes if lane.rect is not None] == [0, 2, 3]
    assert "off_row_line" not in found.flags
    assert all(lane.line_offset is None for lane in found.lanes)


def test_an_empty_lane_has_no_line_offset():
    found = detect(adversarial_row("montage", 1000, shifts={2: -14.0}, missing=[4]))
    assert found.lanes[4].rect is None and found.lanes[4].line_offset is None
    assert off_line(found) == [2]


def test_the_row_line_fits_through_the_other_boxes():
    # Straight or bent by a smile, the line runs through the boxes; one box
    # off it does not drag it there, nor two of seven.
    xs = [70.0 * i for i in range(7)]
    tilted = [(x, 50.0 + 0.1 * x) for x in xs]
    assert np.allclose(rowdetect._row_line(tilted, 12), 0.0)
    smiled = [(x, 50.0 + 24.0 * ((x - 210.0) / 420.0) ** 2) for x in xs]  # 6 px of sag
    assert np.allclose(rowdetect._row_line(smiled, 12), 0.0, atol=1e-9)
    off = [(x, y + (15.0 if i == 3 else 0.0)) for i, (x, y) in enumerate(tilted)]
    assert np.allclose(rowdetect._row_line(off, 12), [0, 0, 0, 1.25, 0, 0, 0])
    two = [(x, y - (40.0 if i in (5, 6) else 0.0)) for i, (x, y) in enumerate(smiled)]
    offsets = rowdetect._row_line(two, 12)
    assert np.allclose(offsets[:5], 0.0, atol=1e-9) and np.allclose(offsets[5:], -40 / 12)
    assert rowdetect._row_line(tilted[:3], 12) is None  # fewer than ROW_LINE_MIN
    # Over two rows (heights more than a smile apart), the smaller group lies
    # off whole, the line through the larger: 45 px (6.4 box heights) apart.
    rows = [(70.0 * i, 75.5 if i < 2 else 30.5) for i in range(5)]
    assert np.allclose(rowdetect._row_line(rows, 7), [45 / 7, 45 / 7, 0, 0, 0])


# Honest flat rows of thin bands and few lanes, a pixel of jitter (the review
# of #114): the half of the boxes nearest a fit alone let four boxes a pixel
# apart pick a smile that fits them exactly and passes the others by box
# heights. Main placed every one of these rows with no flag.
THIN = {"pitch": 50.0, "w": 32.0}


@pytest.mark.parametrize(
    ("seed", "n", "h", "jitter"),
    [(7, 5, 4.0, 1.0), (9, 6, 4.0, 1.0), (11, 6, 6.0, 1.5), (81, 7, 5.0, 1.0), (17, 8, 5.0, 1.0)],
)
def test_a_flat_row_of_thin_bands_lies_on_its_line(seed, n, h, jitter):
    found = detect(adversarial_row("flat", seed, n=n, h=h, y_jitter=jitter, **THIN))
    assert found.size.height <= 8  # a pixel is an eighth of a box height or more
    assert not found.refused and "off_row_line" not in found.flags
    assert max(abs(lane.line_offset) for lane in found.lanes) < 0.4


@pytest.mark.parametrize(("n", "h", "jitter"), [(5, 4.0, 1.0), (5, 6.0, 1.5), (6, 6.0, 1.5)])
def test_flat_rows_of_thin_bands_are_placed(n, h, jitter):
    for seed in range(20):
        found = detect(adversarial_row("flat", seed, n=n, h=h, y_jitter=jitter, **THIN))
        assert "off_row_line" not in found.flags, seed


def test_the_row_line_is_level_through_boxes_a_pixel_apart():
    # Seven 5 px boxes whose centres lie within 2 px of each other: the last
    # four lie exactly on a smile of nearly two box heights, which passes the
    # first three by up to 2.2 box heights. The line is the level one that
    # every box lies near.
    ys = [8.5, 9.5, 8.5, 10.5, 9.5, 9.5, 10.5]
    offsets = rowdetect._row_line([(50.0 * i, y) for i, y in enumerate(ys)], 5)
    assert np.max(np.abs(offsets)) < 0.3


@pytest.mark.parametrize(("n", "above"), [(2, 0), (3, 0), (3, 1), (4, 0), (4, 3)])
def test_a_row_box_over_two_rows_with_few_boxes_is_refused(n, above):
    # Two to four lanes, one lane's strongest band in the row 40 px (3.3 box
    # heights) above: no line through so few boxes shows it (any three lie on
    # some smile, and four bend to one), but no smile or tilt steps a row that
    # far from one lane to the next. Of two boxes, neither is the row's.
    case = two_rows(1000, {above: 2.0}, rel=0.5, n=n)
    found = detect(case)
    assert in_row_above(found, case) == [above]
    assert found.refused and found.flags[0] == "off_row_line"
    assert off_line(found) == ([0, 1] if n == 2 else [above])
    if n == 2:
        assert found.notes[0] == (
            "lanes 1, 2: box centres more than 2 box heights (24 px) apart, on two rows"
        )


@pytest.mark.parametrize("missing", [[1, 2, 3, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5]])
def test_a_sparse_tilted_row_of_few_thin_boxes_is_placed(missing):
    # Two to four of eight lanes, thin bands on a row tilted by 30 px across
    # (5 degrees): boxes more than two box heights apart, but none of them in
    # neighbouring lanes, on one line.
    case = adversarial_row("sparse tilt", 1000, n=8, h=5.0, tilt=30.0, missing=missing, **THIN)
    found = detect(case)
    assert not found.refused and "off_row_line" not in found.flags
    ys = [(lane.rect[1] + lane.rect[3]) / 2 for lane in found.lanes if lane.rect is not None]
    assert max(ys) - min(ys) > ROW_SMILE * found.size.height


def test_the_row_line_is_flagged_as_refusing_with_its_settings():
    assert "off_row_line" in REFUSING_FLAGS
    found = settings()
    keys = ("row_line_k", "row_smile", "row_line_min", "row_line_tol", "row_line_tol_px")
    assert (
        tuple(found[key] for key in keys)
        == (ROW_LINE_K, ROW_SMILE, ROW_LINE_MIN, ROW_LINE_TOL, ROW_LINE_TOL_PX)
        == (0.75, 2.0, 4, 0.25, 3.0)
    )


@pytest.mark.parametrize(
    ("name", "value", "case"),
    [
        ("ROW_LINE_K", 1.5, adversarial_row("montage", 1000, shifts={2: -14.0})),
        ("ROW_SMILE", 0.5, adversarial_row("strong smile", 1000, smile=20.0, missing=[1, 4])),
        ("ROW_LINE_MIN", 7, adversarial_row("montage", 1000, shifts={2: -14.0})),
        # An end lane 13 px (1.1 box heights) off: a tolerance of a box height
        # lets the line bend to it.
        ("ROW_LINE_TOL", 1.0, adversarial_row("montage", 1000, shifts={0: -13.0})),
        # Five 4 px boxes a pixel apart: without the pixels, four of them pick
        # a smile again.
        ("ROW_LINE_TOL_PX", 0.0, adversarial_row("flat", 7, n=5, h=4.0, **THIN)),
    ],
)
def test_each_row_line_setting_takes_part(monkeypatch, name, value, case):
    found = detect(case)
    monkeypatch.setattr(rowdetect, name, value)
    assert settings()[name.lower()] == value
    assert detect(case) != found


def test_box_size_is_the_model_type():
    assert isinstance(_detected("all_present").size, BoxSize)


# --- #111: a first row box over a ladder, a label or a neighbouring panel ---

# The rows whose box also covers what lies beside the row (rowcases).
BESIDE = ("ladder_beside", "ladder_before", "label_beside", "panel_beside")
DOUBT_NOTE = "lane numbers doubtful: "


def lanes_off(case: RowCase, found: RowDetection) -> list[int]:
    """The lanes whose band's extent is centred more than half a lane step
    off the lane's own centre: read into another lane."""
    step = float(np.median(np.diff(case.lane_cx)))
    return [
        lane.lane
        for lane in found.lanes
        if lane.extent is not None
        and abs((lane.extent[0] + lane.extent[2]) / 2 - case.lane_cx[lane.lane]) > step / 2
    ]


def doubts(found: RowDetection) -> str:
    """The doubtful_lanes note ("" without the flag), flag and note checked
    together."""
    note = found.doubt_note
    assert ("doubtful_lanes" in found.flags) == (note is not None)
    assert [n for n in found.notes if n.startswith(DOUBT_NOTE)] == ([note] if note else [])
    return note or ""


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
@pytest.mark.parametrize("key", BESIDE)
def test_a_row_box_over_what_lies_beside_the_row_is_placed_with_its_lanes_doubtful(key, seed):
    # With no lanes on the image to check it, the reading takes the ladder's
    # band, the label or the neighbouring panel for a lane and numbers the
    # bands a lane or more off, with no refusing flag. The row is placed, its
    # lane numbers doubtful (the maintainer's decision on #111: flag first).
    case = _adversarial(key, seed)
    found = detect(case)
    assert not found.refused and found.size is not None
    assert lanes_off(case, found)  # what the flag is there for
    assert doubts(found).startswith(DOUBT_NOTE)


@pytest.mark.parametrize("key", BESIDE)
def test_the_same_rows_boxed_over_their_lanes_only_are_not_doubtful(key):
    recipe = {name: value for name, value in ADVERSARIAL[key].items() if name != "box_adjust"}
    for seed in (1000, 1001, 1002):
        case = adversarial_row(key, seed, **recipe)
        found = detect(case)
        assert_hits_own_lanes(case, found)
        assert doubts(found) == ""


# The accuracy judge's loose boxes and uneven spacings (acc_adv), which
# stretch the fitted pitch, or the steps between the bands, the most while the
# reading fits.
JUDGE_HONEST = {
    "very_loose_box": {"mx": 70},
    "very_loose_miss0": {"mx": 70, "missing": [0]},
    "very_loose_miss5": {"mx": 70, "missing": [5]},
    "uneven35_miss": {
        "pitches": [46, 95, 52, 92, 49],
        "w": 36.0,
        "x_jitter": 0.0,
        "missing": [3],
    },
    "drift_uneven_miss": {
        "n": 12,
        "pitch": 50.0,
        "w": 32.0,
        "h": 10.0,
        "pitches": [60, 60, 60, 60, 60, 50, 40, 40, 40, 40, 40],
        "x_jitter": 0.0,
        "missing": [3, 9],
    },
}


@pytest.mark.parametrize(
    "name",
    [
        *BENCH,
        *(
            f"{key}/{seed}"
            for key in ADVERSARIAL
            if key not in (*BESIDE, "box_omits_empty_first")  # that one is refused
            for seed in (1000, 1001, 1002)
        ),
    ],
)
def test_honest_rows_are_not_doubtful(name):
    found = _detected(name)
    assert not found.refused
    assert doubts(found) == ""


@pytest.mark.parametrize("key", JUDGE_HONEST)
def test_loose_boxes_and_uneven_spacing_are_not_doubtful(key):
    for seed in range(1000, 1010):
        case = adversarial_row(key, seed, **JUDGE_HONEST[key])
        found = detect(case)
        assert not lanes_off(case, found)
        assert doubts(found) == "", seed


def test_pieces_merged_or_dropped_within_a_lane_are_no_doubt():
    # Dust midway between two lanes, merged with a band or dropped, and a
    # band's halves split by a bubble, merged again: pieces half a lane apart
    # at most, not two bands.
    for key, seeds in (("blob_gap", (1000, 1003, 1006)), ("bubble_band", (1000, 1001, 1002))):
        for seed in seeds:
            found = _detected(f"{key}/{seed}")
            assert any(note.startswith(("merged", "dropped")) for note in found.notes)
            assert doubts(found) == "", (key, seed)


# A weak speck 0.8 lane steps past the last band.
SPECK = beside(0.8, 12, 10, 10000)


@pytest.mark.parametrize(
    "recipe",
    [
        {"mx": 40},  # in a loose box's margin
        {"mx": 60},
        {"margin_right": 120, "box_adjust": (0, 0, 50, 0)},  # a box dragged past the row
    ],
)
def test_a_weak_speck_dropped_past_the_end_lane_is_no_doubt(recipe):
    # The weaker of the closest pair, the speck is dropped: past the pieces
    # read, it leaves each of their lanes as it was, and the reading is right.
    dropped = 0
    for seed in range(1000, 1010):
        case = adversarial_row("speck", seed, artefacts=[SPECK], **recipe)
        found = detect(case)
        assert not found.refused and not lanes_off(case, found)
        dropped += any(note.startswith("dropped") for note in found.notes)
        assert doubts(found) == "", seed
    assert dropped >= 5


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_a_weak_band_dropped_between_the_pieces_read_is_doubtful(seed):
    # Seven bands, six lanes declared, the weak fourth band 60 px from the
    # fifth: dropped between the others to fit, the only doubt.
    case = adversarial_row("seven", seed, n=7, pitches=[70, 70, 70, 60, 70, 70], depths={3: 4000})
    note = doubts(detect(dataclasses.replace(case, n_lanes=6)))
    dropped = (
        r"two pieces 0\.7 lanes apart, one dropped at x=2\d\d\.\.3\d\d to fit the declared lanes"
    )
    assert re.fullmatch(DOUBT_NOTE + dropped, note)


@pytest.mark.parametrize("seed", [1000, 1004, 1006])
def test_more_bands_than_declared_lanes_are_doubtful(seed):
    # Six bands, five lanes declared: two bands a lane apart are merged to fit.
    case = dataclasses.replace(adversarial_row("six", seed), n_lanes=5)
    note = doubts(detect(case))
    assert re.search(r"two pieces \d\.\d lanes apart, merged at x=\d+\.\.\d+ to fit", note)


def test_a_box_with_room_for_another_lane_is_doubtful():
    # Two lane steps of bare membrane past the last lane and lane 0 empty:
    # the empty lane is read at the box's right end, every band a lane early,
    # the pitch stretched by less than PITCH_DOUBT.
    case = adversarial_row("room", 1000, missing=[0], margin_right=200, box_adjust=(0, 0, 150, 0))
    found = detect(case)
    assert lanes_off(case, found) == [0, 1, 2, 3, 4]
    assert abs(found.pitch / 70.0 - 1.0) < PITCH_DOUBT
    assert "the row box reaches 1.6 lanes past lane 6's centre" in doubts(found)


# Eight lanes 48 px apart, the first three bands 64 px wide and touching: one
# run the reading cuts into cells.
TOUCHING_RUN = {
    "n": 8,
    "pitch": 48.0,
    "w": 30.0,
    "widths": {0: 64.0, 1: 64.0, 2: 64.0},
    "x_jitter": 0.0,
}


def test_a_touching_run_cut_into_too_few_cells_is_doubtful():
    # The box reaches past the lanes on both sides: the run of three is read
    # as two, the lanes after it numbered a lane late.
    case = adversarial_row(
        "run", 1000, margin_left=150, margin_right=150, box_adjust=(-60, 0, 60, 0), **TOUCHING_RUN
    )
    found = detect(case)
    assert lanes_off(case, found)
    assert "the touching bands read as lanes 2 to 3 span 3.2 lanes" in doubts(found)
    honest = detect(adversarial_row("run", 1000, **TOUCHING_RUN))
    assert doubts(honest) == ""  # three wide bands in three cells


def test_the_doubtful_lanes_are_numbered_as_the_caller_numbers_them():
    case = _adversarial("panel_beside", 1000)
    note = doubts(detect(case))
    back = doubts(detect(case, right_to_left=True))
    assert "past lane 1's centre" in note
    assert back == note.replace("past lane 1's centre", "past lane 5's centre")


def test_the_doubt_is_a_warning_listed_with_its_settings():
    assert "doubtful_lanes" in WARNING_FLAGS and "doubtful_lanes" not in REFUSING_FLAGS
    found = settings()
    keys = ("pitch_doubt", "lanes_doubt", "end_doubt", "apart_doubt")
    assert (
        tuple(found[key] for key in keys)
        == (PITCH_DOUBT, LANES_DOUBT, END_DOUBT, APART_DOUBT)
        == (0.3, 0.5, 1.5, 0.6)
    )


@pytest.mark.parametrize(
    ("name", "value", "row"),
    [
        ("PITCH_DOUBT", 0.0, "all_present"),
        ("LANES_DOUBT", 0.0, "uneven_spacing"),
        ("END_DOUBT", 0.0, "all_present"),
        ("APART_DOUBT", 0.0, "blob_gap/1000"),
    ],
)
def test_each_doubt_setting_takes_part(monkeypatch, name, value, row):
    assert doubts(_detected(row)) == ""  # cached before the setting changes
    monkeypatch.setattr(rowdetect, name, value)
    assert settings()[name.lower()] == value
    assert doubts(detect(_case(row))) != ""


# --- #58: what the detector finds stays what it found ---

GOLDEN = Path(__file__).parent / "data" / "rowdetect_golden.json"
# The saturated rows are also read with their saturation level (#121), and one
# row from its right end.
_GOLDEN_SATURATED = ("dumbbell_band", "hollow_band", "hollow_ring", "notched_band")


def _golden_cases() -> dict[str, tuple[RowCase, dict]]:
    """The rows the golden file pins, by name: every bench row, every
    adversarial recipe (at the seed the invariant checks use, else 1000), the
    saturated rows with their saturation level, a JPEG export and a row read
    from its right end."""
    seeds = dict(ADVERSARIAL_KEYS)
    cases: dict[str, tuple[RowCase, dict]] = {f"bench/{name}": (BENCH[name], {}) for name in BENCH}
    for key in ADVERSARIAL:
        seed = seeds.get(key, 1000)
        cases[f"adversarial/{key}/{seed}"] = (_adversarial(key, seed), {})
    cases["bench/overexposed saturated_at=0"] = (BENCH["overexposed"], {"saturated_at": 0.0})
    for key in _GOLDEN_SATURATED:
        case = _adversarial(key, 1000)
        cases[f"adversarial/{key}/1000 saturated_at=0"] = (case, {"saturated_at": 0.0})
    cases["jpeg/1000"] = (jpeg(adversarial_row("jpeg", 1000, noise=100.0, my=10)), {})
    cases["bench/all_present right_to_left"] = (BENCH["all_present"], {"right_to_left": True})
    return cases


def _golden_entry(found: RowDetection) -> dict:
    """What the golden file keeps of a result: its flags and notes, and per
    lane its rect, reason, components and SNR."""
    return {
        "flags": list(found.flags),
        "notes": list(found.notes),
        "lanes": [
            {
                "rect": None if lane.rect is None else list(lane.rect),
                "reason": lane.reason,
                "components": lane.components,
                "snr": lane.snr,
            }
            for lane in found.lanes
        ],
    }


@cache
def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_golden_file_covers_every_golden_row():
    assert sorted(_golden()["cases"]) == sorted(_golden_cases())


@pytest.mark.parametrize("name", list(_golden_cases()))
def test_rowdetect_matches_golden(name):
    # The file was written by the detector as it was before #58 added peaks:
    # flags, notes, rects, reasons and components exactly, the SNR within
    # rel_tol (the last bits of numpy and scipy may differ between platforms).
    case, kwargs = _golden_cases()[name]
    expected = _golden()["cases"][name]
    found = _golden_entry(detect(case, **kwargs))
    assert found["flags"] == expected["flags"]
    assert found["notes"] == expected["notes"]
    assert len(found["lanes"]) == len(expected["lanes"])
    for got, want in zip(found["lanes"], expected["lanes"], strict=True):
        assert {k: got[k] for k in ("rect", "reason", "components")} == {
            k: want[k] for k in ("rect", "reason", "components")
        }
        assert math.isclose(got["snr"], want["snr"], rel_tol=1e-9, abs_tol=1e-12)


# --- #58: the lane's peaks, for a count of its bands ---


@pytest.mark.parametrize("name", [*INVARIANT_CASES, "hollow_band/1000 saturated"])
def test_peaks_are_the_component_tops(name, monkeypatch):
    # Every top _count_components reads, each in the lane whose span holds it,
    # top to bottom, at its pixel's centre (y moved at most half a row).
    if name.endswith(" saturated"):
        case, kwargs = _case(name.removesuffix(" saturated")), {"saturated_at": 0.0}
    else:
        case, kwargs = _case(name), {}
    tops: list[list[tuple[int, int]]] = []
    real = rowdetect._peaks

    def spy(ks, h):
        found = real(ks, h)
        tops.append(found)
        return found

    monkeypatch.setattr(rowdetect, "_peaks", spy)
    found = detect(case, **kwargs)
    assert len(tops) <= 1  # read once, on the pass that gives the result
    x0, y0 = max(0, case.row[0]), max(0, case.row[1])
    listed = []
    for lane in found.lanes:
        if lane.rect is None:
            assert lane.peaks == ()
            continue
        # Top to bottom by their pixel rows; a refined y may cross a row's by less than one.
        assert all(b.y > a.y - 1.0 for a, b in itertools.pairwise(lane.peaks))
        for peak in lane.peaks:
            assert isinstance(peak, rowdetect.Peak) and peak[:3] == (peak.y, peak.x, peak.snr)
            column, row = peak.x - 0.5 - x0, math.floor(peak.y - y0)
            assert column == int(column)
            assert (row, int(column)) in tops[0] or (row - 1, int(column)) in tops[0], peak
            assert peak.snr >= DETECT_K
            listed.append((round(peak.y, 6), peak.x))
            if peak.own:  # the band's own lies in its grown extent
                ex0, ey0, ex1, ey1 = lane.extent
                assert ex0 <= peak.x <= ex1 and ey0 <= peak.y <= ey1
        # A second component is a peak of another band; not every such peak is one.
        assert lane.components - 1 <= sum(peak.other_band for peak in lane.peaks)
        assert not any(peak.own and peak.other_band for peak in lane.peaks)
        assert rowdetect.bands_in(lane, -math.inf, math.inf) >= lane.components
    assert len(listed) == len(set(listed))  # no top in two lanes


def test_a_band_whose_top_is_no_separate_peak_still_counts_itself():
    # Touching bands: a lane's highest pixel may rise from its neighbour's hill
    # without a saddle deep enough to be a peak of its own.
    found = _detected("touching")
    assert any(lane.rect is not None and lane.peaks == () for lane in found.lanes)
    assert [rowdetect.bands_in(lane, -math.inf, math.inf) for lane in found.lanes] == [1] * 6


def test_a_doublet_s_peaks_hold_both_bands():
    case = _adversarial("doublet_deep", 1000)
    lane = detect(case).lanes[2]
    assert lane.components == 2
    own, other = lane.peaks
    assert (own.own, own.other_band, other.own, other.other_band) == (True, False, False, True)
    assert abs((other.y - own.y) - 14.0) < 1.5  # 14 px apart
    assert other.snr >= SECOND_SHARE * own.snr
    # A dumbbell and a hollow band are one band: both their tops are their own.
    for key in ("dumbbell_band", "hollow_band"):
        lane = detect(_adversarial(key, 1000), saturated_at=0.0).lanes[1]
        assert len(lane.peaks) == 2 and all(peak.own for peak in lane.peaks), key
        assert rowdetect.bands_in(lane, -math.inf, math.inf) == 1, key


_JPEG_SEEDS = [1000, 1002, 1003, 1006, 1007]


def _jpeg_detected(seed: int) -> RowDetection:
    return detect(jpeg(adversarial_row("jpeg", seed, noise=100.0, my=10)))


@pytest.mark.parametrize("seed", _JPEG_SEEDS)
def test_jpeg_block_noise_is_no_other_band(seed):
    # Block artefacts reach DETECT_K beside the bands, but only about 1% of the
    # lane's peak: peaks of no band.
    found = _jpeg_detected(seed)
    assert [rowdetect.bands_in(lane, -math.inf, math.inf) for lane in found.lanes] == [1] * 6


def test_jpeg_block_noise_reaches_detect_k():
    # The test above is not vacuous: its artefacts do make extra peaks. Which
    # seeds do depends on the platform's JPEG encoder, so the seeds are pooled.
    assert any(len(lane.peaks) > 1 for seed in _JPEG_SEEDS for lane in _jpeg_detected(seed).lanes)


@pytest.mark.parametrize(("frac", "counted"), [(0.2, False), (0.3, True), (0.7, True)])
def test_bands_found_uses_second_share(frac, counted, monkeypatch):
    # #58, D10: a band 20 px below lane 3's, with membrane between, is no
    # second component (#121), but it is another band in the lane: a count
    # reads it from SECOND_SHARE of the lane's peak, where it lies in the window.
    case = adversarial_row("doublet", 1000, doublet={2: (20, frac)}, my=12)
    lane = detect(case).lanes[2]
    assert lane.components == 1
    [own] = [peak for peak in lane.peaks if peak.own]
    others = [peak for peak in lane.peaks if not peak.own]
    assert len(others) == 1 and abs(others[0].y - own.y - 20.0) < 1.5
    assert others[0].other_band is counted
    assert (others[0].snr >= SECOND_SHARE * own.snr) is counted
    assert rowdetect.bands_in(lane, -math.inf, math.inf) == 1 + counted
    # A window around the band that stops short of the other holds one band.
    assert rowdetect.bands_in(lane, own.y - 12.0, own.y + 12.0) == 1
    assert rowdetect.bands_in(lane, own.y - 12.0, others[0].y) == 1 + counted
    if counted:  # the share decides it
        monkeypatch.setattr(rowdetect, "SECOND_SHARE", frac + 0.05)
        lane = detect(case).lanes[2]
        assert rowdetect.bands_in(lane, -math.inf, math.inf) == 1
    # An empty lane holds no band.
    empty = detect(BENCH["missing_middle"]).lanes[2]
    assert (empty.peaks, rowdetect.bands_in(empty, -math.inf, math.inf)) == ((), 0)


# Lane 3's band with another 20 px below it, 0.8 as deep and 10 px to the right.
_PARTNER = {"doublet": {2: (20, 0.8)}, "shifts": {2: 10}, "my": 12, "box_adjust": (0, 0, 0, 14)}


def two_topped_partner(seed: int, *, saturated: bool = False) -> RowCase:
    """The row of :data:`_PARTNER` with the lower band lighter across its
    middle than at its ends: a dumbbell, or over-exposed (clipped flat at 0)
    around a lighter centre, a hollow band."""
    if saturated:
        light = blob(2, 6.0, -45000.0, ry=12.0, dy=10)
        return adversarial_row(
            "doublet", seed, **_PARTNER, depths={2: 1.5 * MEMBRANE}, artefacts=[light]
        )
    light = blob(2, 6.0, -11000.0, ry=12.0, dy=10)
    return adversarial_row("doublet", seed, **_PARTNER, artefacts=[light])


@pytest.mark.parametrize("seed", [1000, 1001, 1002, 1003])
@pytest.mark.parametrize("saturated", [False, True])
def test_another_band_with_two_tops_side_by_side_is_one_band(seed, saturated):
    # #58: the band below lane 3's has two tops side by side, one per end:
    # one band, the count's second, as its own tops are the band's one (a
    # dumbbell or a hollow band is one band).
    plain = detect(adversarial_row("doublet", seed, **_PARTNER)).lanes[2]
    assert rowdetect.bands_in(plain, -math.inf, math.inf) == 2
    kwargs = {"saturated_at": 0.0} if saturated else {}
    lane = detect(two_topped_partner(seed, saturated=saturated), **kwargs).lanes[2]
    assert lane.components == 1
    tops = [peak for peak in lane.peaks if not peak.own]
    assert len(tops) == 2, lane.peaks
    left, right = sorted(tops, key=lambda peak: peak.x)
    assert abs(left.y - right.y) < 1.0 and right.x - left.x > 15.0  # side by side
    # The higher of the two stands for the band.
    [counted] = [peak for peak in tops if peak.other_band]
    assert counted.snr == max(peak.snr for peak in tops)
    assert rowdetect.bands_in(lane, -math.inf, math.inf) == 2
    # Two bands stacked in a lane stay two (a doublet's).
    doublet = detect(_adversarial("doublet_deep", 1000)).lanes[2]
    assert rowdetect.bands_in(doublet, -math.inf, math.inf) == 2


# --- #58: a row along a sloping line, grown at an expected row ---


@pytest.mark.parametrize("name", INVARIANT_CASES)
def test_detect_row_along_zero_shift_is_identical(name):
    # A level line: detect_row's result, bit for bit, preferred row or not.
    case = _case(name)
    x0, _, x1, _ = case.row
    level = [0] * (x1 - x0)
    background = estimate_background(case.image)
    kwargs = {"background": background, "dark_on_light": case.dark_on_light}
    along = rowdetect.detect_row_along(case.image, case.row, case.n_lanes, level, **kwargs)
    assert along == detect(case)
    prefer = 0.5 * (case.row[1] + case.row[3])
    along = rowdetect.detect_row_along(
        case.image, case.row, case.n_lanes, level, prefer_y=prefer, **kwargs
    )
    assert along == detect(case, prefer_y=prefer)


def _moved(found: RowDetection, dy: Callable[[int], int]) -> RowDetection:
    """``found`` with each rect, extent and window moved down by ``dy`` of its
    centre column, and each peak by ``dy`` of its own column."""

    def back(rect):
        return (
            None
            if rect is None
            else (
                rect[0],
                rect[1] + dy((rect[0] + rect[2]) // 2),
                rect[2],
                rect[3] + dy((rect[0] + rect[2]) // 2),
            )
        )

    lanes = tuple(
        dataclasses.replace(
            lane,
            rect=back(lane.rect),
            extent=back(lane.extent),
            window=back(lane.window),
            peaks=tuple(p._replace(y=p.y + dy(math.floor(p.x))) for p in lane.peaks),
        )
        for lane in found.lanes
    )
    return dataclasses.replace(found, lanes=lanes)


def _without_peak_ys(found: RowDetection) -> RowDetection:
    lanes = tuple(
        dataclasses.replace(lane, peaks=tuple(p._replace(y=0.0) for p in lane.peaks))
        for lane in found.lanes
    )
    return dataclasses.replace(found, lanes=lanes)


def _peak_ys(found: RowDetection) -> list[float]:
    return [p.y for lane in found.lanes for p in lane.peaks]


_PAD = 20  # rows of membrane above and below a row turned along a line


def _along_a_line(case: RowCase, shift: Callable[[int], int]) -> np.ndarray:
    """The case's image with each column x moved ``_PAD + shift(x)`` rows down,
    on a canvas ``2 * _PAD`` rows higher: its bands along a line."""
    height, width = case.image.shape
    level = FULL_SCALE - MEMBRANE if not case.dark_on_light else MEMBRANE
    canvas = np.full((height + 2 * _PAD, width), level)
    for x in range(width):
        top = _PAD + shift(x)
        canvas[top : top + height, x] = case.image[:, x]
    return canvas


_LINES = {
    "ramp": lambda x: round(-12.0 + 24.0 * x / 500.0),  # a straight line, 24 rows over 500 px
    "steps": lambda x: 7 if (x // 37) % 2 else -5,  # an uneven one: the columns are free
}


@pytest.mark.parametrize("name", ["all_present", "missing_middle", "smile", "light_on_dark"])
@pytest.mark.parametrize("line", list(_LINES))
def test_detect_row_along_known_shift(name, line):
    # A row turned along a known line is found where the level row is, each
    # box moved by its centre column's shift: the levelled pixels are the
    # level row's own.
    case = BENCH[name]
    shift = _LINES[line]
    x0, y0, x1, y1 = case.row
    background = estimate_background(case.image)
    level = detect(case)
    along = rowdetect.detect_row_along(
        _along_a_line(case, shift),
        (x0, y0 + _PAD, x1, y1 + _PAD),
        case.n_lanes,
        [shift(x) for x in range(x0, x1)],
        background=background,
        dark_on_light=case.dark_on_light,
    )
    expected = _moved(level, lambda x: _PAD + shift(x))
    # Exact but for a peak's y: its offset is summed in another order, which
    # may round the last bit differently from one platform to another.
    assert _without_peak_ys(along) == _without_peak_ys(expected)
    for got, want in zip(_peak_ys(along), _peak_ys(expected), strict=True):
        assert math.isclose(got, want, rel_tol=0.0, abs_tol=1e-9)
    moved = [lane.rect for lane in along.lanes if lane.rect is not None]
    assert len({(r[2] - r[0], r[3] - r[1]) for r in moved}) == 1  # one shared size
    assert not any(overlaps(a, b) for a, b in itertools.combinations(moved, 2))


def test_detect_row_along_reads_a_row_past_the_image_s_right_side_as_cut_to_it():
    # A row along the ramp from x0 to 15 columns past the image's right
    # side: its columns past the side are never read, and their shifts not
    # used; found as the row cut to the image, with its shifts.
    case = BENCH["all_present"]
    x0, y0, x1, y1 = case.row
    image = _along_a_line(case, _LINES["ramp"])
    width = image.shape[1]
    assert x0 > 0 and x1 <= width  # not vacuous: the row starts inside
    shifts = [_LINES["ramp"](x) for x in range(x0, width)]
    kwargs = {
        "background": estimate_background(case.image),
        "dark_on_light": case.dark_on_light,
    }
    cut = rowdetect.detect_row_along(
        image, (x0, y0 + _PAD, width, y1 + _PAD), case.n_lanes, shifts, **kwargs
    )
    wide = rowdetect.detect_row_along(
        image, (x0, y0 + _PAD, width + 15, y1 + _PAD), case.n_lanes, shifts + [99] * 15, **kwargs
    )
    assert cut.size is not None and wide == cut


def test_detect_row_along_refuses_rows_the_line_takes_off_the_image():
    case = BENCH["all_present"]
    x0, y0, x1, y1 = case.row
    height = case.image.shape[0]
    background = estimate_background(case.image)

    def along(row, shifts):
        return rowdetect.detect_row_along(
            case.image, row, case.n_lanes, shifts, background=background
        )

    # Up past the top at the row's first column, or down past the bottom at its last.
    for shifts in ([-y0 - 1] + [0] * (x1 - x0 - 1), [0] * (x1 - x0 - 1) + [height - y1 + 1]):
        with pytest.raises(RowDetectError) as refused:
            along(case.row, shifts)
        assert refused.value.code == "row_outside_image"
    # Just inside: found.
    assert along(case.row, [-y0] + [0] * (x1 - x0 - 2) + [height - y1]).size is not None
    # Columns past the image's side are not checked: they are never read.
    wide = (-5, y0, x1, y1)
    assert along(wide, [-y0 - 50] * 5 + [0] * x1).size is not None
    for shifts in ([0] * (x1 - x0 - 1), [0.0] * (x1 - x0), "0" * (x1 - x0)):
        with pytest.raises(RowDetectError) as refused:
            along(case.row, shifts)
        assert refused.value.code == "invalid_row"


def _stronger_row_above(seed: int) -> RowCase:
    """A row with another row 20 px above it, twice as deep, the row box over both."""
    return adversarial_row(
        "stronger_above", seed, neighbour_dy=-20.0, neighbour_rel=2.0, box_adjust=(0, -30, 0, 0)
    )


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_prefer_y_grows_each_lane_at_the_expected_row(seed):
    # #58: without a preferred row each lane grows from its strongest pixel,
    # the deeper row above; with the row's own y, from its band there.
    case = _stronger_row_above(seed)
    below = [cy + 0.5 for cy in case.lane_cy]  # continuous rows
    above = [cy - 20.0 for cy in below]

    def rows_held(found: RowDetection, ys: Sequence[float]) -> list[bool]:
        return [
            lane.rect is not None and lane.rect[1] <= y <= lane.rect[3]
            for lane, y in zip(found.lanes, ys, strict=True)
        ]

    strongest = detect(case)
    assert all(rows_held(strongest, above)), strongest
    preferred = detect(case, prefer_y=float(np.mean(below)))
    assert all(rows_held(preferred, below)), preferred
    check_invariants(case, preferred)
    assert all(
        p.snr < s.snr for p, s in zip(preferred.lanes, strongest.lanes, strict=True)
    )  # the band's own peak, not the deeper row's
    # Nearest the preferred row wins whichever way: the row above, preferred.
    assert all(rows_held(detect(case, prefer_y=float(np.mean(above))), above))


@pytest.mark.parametrize("value", [math.nan, math.inf, "80", True])
def test_a_preferred_row_that_is_not_a_finite_number_is_refused(value):
    with pytest.raises(ValueError, match="prefer_y"):
        detect(BENCH["all_present"], prefer_y=value)


def _row_beside(dy: float, rel: float, seed: int = 1000, **kwargs) -> RowCase:
    """A row with another row ``dy`` px below it (above if negative), ``rel``
    times as deep, the row box over both."""
    adjust = (0, -30, 0, 0) if dy < 0 else (0, 0, 0, 30)
    return adversarial_row(
        "beside", seed, neighbour_dy=dy, neighbour_rel=rel, box_adjust=adjust, **kwargs
    )


def _target_row(case: RowCase) -> float:
    """The row's own bands' mean row, continuous: where a caller expects them."""
    return float(np.mean(case.lane_cy)) + 0.5


@cache
def _lone_height(seed: int) -> int:
    """The shared height of the row alone, with nothing beside it."""
    return detect(adversarial_row("lone", seed)).size.height


def _holds(rect, y: float) -> bool:
    return rect[1] <= y <= rect[3]


def test_prefer_y_stops_at_the_valley_to_a_deeper_band():
    # #58: a row three times as deep 14 px above the expected one. Grown from
    # the band's own peak at EXTENT_LEVEL of it, growth used to climb the dip
    # (above that level, below VALLEY_FRAC of the peak) into the deeper band:
    # one extent over both, a 52 x 27 box in every lane, the deeper band's
    # peak read as the band's own, no flag. It stops at the valley: each band
    # boxed alone at a lone band's size, the deeper band another component,
    # named 14 px or so above the box centre, outside it.
    case = _row_beside(-14.0, 3.0)
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    assert found.flags == ("multiple_components",)
    [note] = found.notes
    lanes, lo, hi, way = _another_band(note)
    assert (lanes, way) == ("lanes 1, 2, 3, 4, 5, 6", "above"), note
    assert 12 <= lo <= hi <= 16, note
    assert found.size.height <= _lone_height(1000) + 1
    for lane, cy in zip(found.lanes, case.lane_cy, strict=True):
        y, deeper_y = cy + 0.5, cy + 0.5 - 14.0
        assert _holds(lane.extent, y) and not _holds(lane.extent, deeper_y)
        assert _holds(lane.rect, y) and not _holds(lane.rect, deeper_y)
        assert lane.components == 2
        [deeper] = [peak for peak in lane.peaks if abs(peak.y - deeper_y) < 3.0]
        assert (deeper.own, deeper.other_band) == (False, True)
        assert rowdetect.bands_in(lane, -math.inf, math.inf) == 2


@pytest.mark.parametrize("rel", [1.5, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_prefer_y_never_boxes_a_deeper_band_with_the_target(side, rel):
    # A row rel times as deep 10 to 24 px above or below the expected one.
    # Where a lane's band is a separate peak, it is boxed alone, the deeper
    # band reported (a second component, or another band apart from it), and
    # the shared size is a lone band's; or the row is refused. A band that is
    # no separate peak of its lane (a shoulder on the deeper band: 10 px, and
    # 12 px from three times as deep) is not told from it here, with or
    # without prefer_y.
    lone = _lone_height(1000)
    for dy in range(10, 26, 2):
        case = _row_beside(side * dy, rel)
        found = detect(case, prefer_y=_target_row(case))
        check_invariants(case, found)
        ys = [cy + 0.5 for cy in case.lane_cy]
        separate = [
            any(abs(peak.y - y) <= 2.5 for peak in lane.peaks)
            for lane, y in zip(found.lanes, ys, strict=True)
        ]
        if dy >= 14:  # not vacuous: from 14 px on, every lane's band is a peak
            assert all(separate), (dy, found)
        if found.refused:
            continue
        for lane, y, alone in zip(found.lanes, ys, separate, strict=True):
            if not alone:
                continue
            deeper_y = y + side * dy
            assert _holds(lane.extent, y) and not _holds(lane.extent, deeper_y), (dy, lane)
            assert _holds(lane.rect, y) and not _holds(lane.rect, deeper_y), (dy, lane)
            assert lane.components > 1 or any(peak.other_band for peak in lane.peaks), (dy, lane)
        if all(separate):
            assert found.size.height <= lone + 1, (dy, found.size)


_DUMBBELL = {"depths": {1: 24000.0}, "artefacts": [blob(1, 6.0, -16000.0, ry=5.0)]}


def test_prefer_y_keeps_two_tops_side_by_side_one_band_beside_a_deeper_one():
    # Lane 2's band lighter in its middle (a dumbbell: two tops side by side)
    # with a row three times as deep 13 px above, the valley to it higher than
    # the dumbbell's dip. Growth from the nearer top stops at the valley, not
    # at the dip: both tops are the band's own, the deeper band another.
    case = _row_beside(-13.0, 3.0, **_DUMBBELL)
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    lane = found.lanes[1]
    y = case.lane_cy[1] + 0.5
    tops = [peak for peak in lane.peaks if abs(peak.y - y) <= 2.5]
    assert len(tops) == 2 and all(peak.own for peak in tops), lane.peaks
    assert abs(tops[0].x - tops[1].x) > 15.0  # side by side, both in the extent
    [deeper] = [peak for peak in lane.peaks if peak not in tops]
    assert (deeper.own, deeper.other_band) == (False, True)
    assert not _holds(lane.extent, deeper.y)
    # And the other way: a deeper band with two tops of its own stays out whole.
    light = blob(1, 6.0, -18000.0, ry=5.0, dy=14.0)  # half as deep across its middle
    case = _row_beside(14.0, 3.0, depths={1: 12000.0}, artefacts=[light])
    lane = detect(case, prefer_y=_target_row(case)).lanes[1]
    y = case.lane_cy[1] + 0.5
    deeper = [peak for peak in lane.peaks if abs(peak.y - y - 14.0) <= 2.5]
    assert len(deeper) == 2 and not any(peak.own for peak in deeper), lane.peaks
    assert _holds(lane.extent, y) and not _holds(lane.extent, y + 14.0)


@pytest.mark.parametrize("key", ["dumbbell_band", "hollow_band"])
def test_prefer_y_keeps_a_dumbbell_and_a_hollow_band_one_band(key):
    # With nothing deeper beside them, a band's tops side by side stay its own
    # wherever the preferred row lies about them.
    case = _adversarial(key, 1000)
    plain = detect(case, saturated_at=0.0)
    for dy in (-1.5, 0.5, 2.5):
        found = detect(case, saturated_at=0.0, prefer_y=case.lane_cy[1] + dy)
        assert found.flags == plain.flags
        lane = found.lanes[1]
        assert len(lane.peaks) == 2 and all(peak.own for peak in lane.peaks), key
        assert lane.extent == plain.lanes[1].extent
        assert rowdetect.bands_in(lane, -math.inf, math.inf) == 1


def _sloping_dumbbell(lane: int, h: float, slope: float):
    """A band 60 x ``h`` px at 20% on ``lane``'s centre, 26000 deep, sloping
    ``slope`` px down for each px right, half as deep in its middle as at its
    ends (a dumbbell), its right end a tenth deeper than its left."""
    w, ux4, ux2 = 60.0, (2.0 * math.log(5.0)) ** 0.25, math.sqrt(2.0 * math.log(5.0))

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        u = X - lcx[lane]
        across = np.exp(-0.5 * np.abs(u / (w / (2.0 * ux4))) ** 4)
        down = np.exp(-0.5 * ((Y - lcy[lane] - slope * u) / (h / (2.0 * ux2))) ** 2)
        dip = 1.0 - 0.5 * np.exp(-0.5 * (u / (w / 6.0)) ** 2)
        ends = 1.0 + 0.05 * np.clip(u / (w / 2.0), -1.0, 1.0)
        return 26000.0 * across * down * dip * ends

    return f


@pytest.mark.parametrize(("h", "slope"), [(8.0, 0.08), (8.0, -0.08), (12.0, 0.12)])
def test_prefer_y_keeps_a_sloping_dumbbell_one_band(h, slope):
    # Lane 2's band half as deep in its middle as at its ends, sloping 4.6 to
    # 6.8 degrees: on so thin a band its ends' rows lie 2.5 to 3.5 px apart,
    # each outside the other's hill's rows. Grown from its weaker end, the other
    # end was read as a band above or below it and the band cut in two at its
    # middle: half of it boxed, the other half a second component. It is one
    # band: grown whole, both ends its own, as without prefer_y.
    artefacts = [_sloping_dumbbell(1, h, slope)]
    case = adversarial_row(
        "sloping", 1000, w=60.0, h=h, depths={1: 0.0}, artefacts=artefacts, my=10
    )
    plain = detect(case)
    whole = plain.lanes[1].extent
    ends = plain.lanes[1].peaks
    assert len(ends) == 2 and all(peak.own for peak in ends)
    weaker, stronger = sorted(ends, key=lambda peak: peak.snr)
    assert abs(weaker.y - stronger.y) >= 1.5 and abs(weaker.x - stronger.x) >= 20.0
    away = math.copysign(1.0, weaker.y - stronger.y)
    for dy in (0.0, 1.0, 2.0):
        found = detect(case, prefer_y=weaker.y + away * dy)
        check_invariants(case, found)
        assert found.flags == plain.flags
        lane = found.lanes[1]
        assert len(lane.peaks) == 2 and all(peak.own for peak in lane.peaks), lane
        assert lane.snr == pytest.approx(min(peak.snr for peak in lane.peaks))  # the weaker end
        assert lane.components == 1
        assert rowdetect.bands_in(lane, -math.inf, math.inf) == 1
        assert abs(lane.extent[0] - whole[0]) <= 1 and abs(lane.extent[2] - whole[2]) <= 1


def _smiling_beside(dy: float, rel: float, seed: int = 1000) -> RowCase:
    """A row smiling 8 px (its middle lanes lower than its end ones) with
    another row ``dy`` px below it (above if negative), ``rel`` times as deep,
    the row box over both."""
    adjust = (0, -30, 0, 0) if dy < 0 else (0, 0, 0, 30)
    return adversarial_row(
        "smiling", seed, smile=8.0, neighbour_dy=dy, neighbour_rel=rel, box_adjust=adjust
    )


@pytest.mark.parametrize(("dy", "rel"), [(-16.0, 2.0), (-16.0, 3.0), (16.0, 2.0)])
def test_prefer_y_refuses_lanes_grown_from_bands_on_two_rows(dy, rel):
    # #58: the expected row level, 3.5 px off the row's bands in its end
    # lanes towards the other row (above), or in its middle lanes (below).
    # The smile puts the other row's peak nearer it in the middle lanes (or
    # the end lanes): those lanes grew from the other row's bands, the rest
    # from the row's own, and the boxes passed for a smile, only a second
    # component flagged. Lanes 1 and 2, 5 and 6 grew from bands on two rows,
    # each holding a band on both: refused.
    case = _smiling_beside(dy, rel)
    ys = [cy + 0.5 for cy in case.lane_cy]
    prefer = min(ys) - 3.5 if dy < 0 else max(ys) + 3.5
    found = detect(case, prefer_y=prefer)
    check_invariants(case, found)
    assert found.refused and "off_row_line" in found.flags
    assert found.crossed == (0, 1, 4, 5)
    assert any(
        note.startswith("lanes 1, 2, 5, 6: grown from bands on two rows") for note in found.notes
    )
    # The row's own y: every lane grown from the row's band, not refused.
    own = detect(case, prefer_y=float(np.mean(ys)))
    assert not own.refused and own.crossed == ()
    assert all(_holds(lane.rect, y) for lane, y in zip(own.lanes, ys, strict=True))


@pytest.mark.parametrize("rel", [1.5, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_prefer_y_never_boxes_the_other_row(side, rel):
    # A row smiling 8 px with another rel times as deep 12 to 24 px above or
    # below it, the expected row level: at the row's bands in its end lanes
    # (above) or its middle lanes (below), or up to a third of the way from
    # them to the other row. Where every lane's band is a separate peak, the
    # row is refused, or every lane boxed on its own band alone: never a box
    # over both rows' bands, nor a lane boxed on the other row's.
    refused = placed = 0
    for dy in (12, 16, 20, 24):
        case = _smiling_beside(side * dy, rel)
        ys = [cy + 0.5 for cy in case.lane_cy]
        start = min(ys) if side < 0 else max(ys)
        for share in (0.0, 0.2, 0.35):
            found = detect(case, prefer_y=start + share * side * dy)
            check_invariants(case, found)
            if found.refused:
                refused += 1
                continue
            lanes = list(zip(found.lanes, ys, strict=True))
            if not all(any(abs(peak.y - y) <= 2.5 for peak in lane.peaks) for lane, y in lanes):
                continue  # a lane's band is no separate peak: a shoulder on the other row's
            placed += 1
            for lane, y in lanes:
                assert lane.extent is not None, (dy, share, lane)
                assert _holds(lane.extent, y), (dy, share, lane)
                assert not _holds(lane.extent, y + side * dy), (dy, share, lane)
    assert refused and placed  # not vacuous


# --- #58: which peaks are one band, and which band is higher ---


@pytest.mark.parametrize("light_on_dark", [False, True])
@pytest.mark.parametrize(("depth", "light", "seed"), BURNT_OUT)
def test_prefer_y_boxes_a_burnt_out_band_whole(depth, light, seed, light_on_dark):
    # #121: lane 2's band split along x by its burnt-out centre, which falls
    # below the noise: no signal joins its halves, whose tops lie on one row
    # or a row apart, a hair apart in height. Grown from the half nearer the
    # expected row, the other half, a hair higher, was read as a band above
    # or below it and taken out of the growth: half the band boxed (42 x 12
    # for 48 x 12), a second component flagged, and the over-exposure warning
    # lost. The halves share their rows, as a band's halves do: one band,
    # joined and boxed whole and hollow, as without an expected row.
    case = burnt_out(depth, light, seed)
    limit = 0.0
    if light_on_dark:
        case = dataclasses.replace(case, image=FULL_SCALE - case.image, dark_on_light=False)
        limit = FULL_SCALE
    plain = detect(case, saturated_at=limit)
    for dy in (-3.0, 0.5, 3.0):
        found = detect(case, saturated_at=limit, prefer_y=case.lane_cy[1] + dy)
        assert (found.flags, found.slots) == (plain.flags, plain.slots), dy
        lane = found.lanes[1]
        assert lane.extent == plain.lanes[1].extent and lane.hollow, dy
        assert lane.components == 1 and all(peak.own for peak in lane.peaks), dy
        assert rowdetect.bands_in(lane, -math.inf, math.inf) == 1


def _sloping_band(lane: int, depth: float, slope: float, w: float = 44.0, h: float = 12.0):
    """A flat-topped band ``w`` x ``h`` px at 20% on ``lane``'s centre, sloping
    ``slope`` px down for each px right."""
    ux4, ux2 = (2.0 * math.log(5.0)) ** 0.25, math.sqrt(2.0 * math.log(5.0))

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        u = X - lcx[lane]
        across = np.exp(-0.5 * np.abs(u / (w / (2.0 * ux4))) ** 4)
        down = np.exp(-0.5 * ((Y - lcy[lane] - slope * u) / (h / (2.0 * ux2))) ** 2)
        return depth * across * down

    return f


@pytest.mark.parametrize(
    ("slope", "light", "seed"),
    [
        (0.06, 75000.0, 1000),
        (-0.06, 75000.0, 1000),
        (0.09, 75000.0, 1002),
        (-0.09, 75000.0, 1003),
        (0.12, 50000.0, 1000),
        (-0.12, 60000.0, 1002),
    ],
)
def test_prefer_y_boxes_a_sloping_burnt_out_band_whole(slope, light, seed):
    # The band of burnt_out, 1.5 times as deep as the membrane, sloping 3.4
    # to 6.8 degrees: its halves' tops lie one to three rows apart. From 3.4
    # degrees on, growth from the half nearer the expected row left the
    # other half out as a band above or below, at one expected row or
    # another. One band, as without an expected row: boxed whole, hollow.
    artefacts = [_sloping_band(1, 1.5 * MEMBRANE, slope), blob(1, 6.0, -light, ry=12.0)]
    case = adversarial_row("sloping", seed, depths={1: 0.0}, artefacts=artefacts, my=10)
    plain = detect(case, saturated_at=0.0)
    whole = plain.lanes[1].extent
    assert plain.lanes[1].hollow and plain.lanes[1].components == 1  # not vacuous
    for dy in (-2.0, 0.5, 2.0):
        found = detect(case, saturated_at=0.0, prefer_y=case.lane_cy[1] + dy)
        lane = found.lanes[1]
        assert found.flags == plain.flags, dy
        assert lane.hollow and lane.components == 1, dy
        assert abs(lane.extent[0] - whole[0]) <= 1 and abs(lane.extent[2] - whole[2]) <= 1, dy
        assert rowdetect.bands_in(lane, -math.inf, math.inf) == 1


def _smear(lane: int, dx: tuple[float, float], depth: float, length: float, up: bool = False):
    """A smear under part of ``lane``'s band, from ``dx[0]`` to ``dx[1]`` px
    of its centre along x: ``depth`` deep from the band's centre down to 6 px
    below it (up to 6 px above it if ``up``), fading over ``length`` px past
    that."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        u = X - lcx[lane]
        across = 1.0 / (1.0 + np.exp(-(u - dx[0]) / 1.5)) / (1.0 + np.exp((u - dx[1]) / 1.5))
        d = lcy[lane] - Y if up else Y - lcy[lane]
        down = np.where(d < 0.0, 0.0, np.where(d < 6.0, 1.0, np.exp(-(d - 6.0) / length)))
        return depth * across * down

    return f


@pytest.mark.parametrize("light_on_dark", [False, True])
@pytest.mark.parametrize(
    "artefact",
    [
        _smear(1, (-22.0, -8.0), 6000.0, 15.0),
        _smear(1, (8.0, 22.0), 6000.0, 15.0),
        _smear(1, (-22.0, -8.0), 6000.0, 15.0, up=True),
        blob(1, 2.5, 60000.0, dx=-15.0, dy=10.0),
    ],
    ids=["smear below the left half", "... the right half", "above the left half", "speck"],
)
def test_prefer_y_boxes_a_burnt_out_band_with_a_smear_under_one_half_whole(artefact, light_on_dark):
    # The band of burnt_out(1.5, 75000, 1000), split along x by its burnt-out
    # centre, and a faint smear (6000 deep, fading over 15 px) hanging below
    # or above one half only, or a clipped speck touching the left half's
    # bottom edge. The smear drains into that half's basin, which then
    # spanned 47 rows against the other half's 21: read by their basins, the
    # halves shared too few rows, and the other half, a hair higher, was
    # taken for a band above or below and left out of the growth. Half the
    # band boxed (43 x 12 for 48 x 12), a second component flagged, the
    # over-exposure warning lost, at every expected row from -8 to +8 px. The
    # halves are read by the rows their pieces span at the growth level, as
    # _join_burnt_out reads them (the smear lies below it): one band, boxed
    # whole and hollow, as without an expected row.
    case = adversarial_row(
        "burnt_out",
        1000,
        depths={1: 1.5 * MEMBRANE},
        artefacts=[blob(1, 6.0, -75000.0, ry=12.0), artefact],
        my=10,
        box_adjust=(0, -20, 0, 20),
        img_h=200,
    )
    limit = 0.0
    if light_on_dark:
        case = dataclasses.replace(case, image=FULL_SCALE - case.image, dark_on_light=False)
        limit = FULL_SCALE
    plain = detect(case, saturated_at=limit)
    assert plain.lanes[1].hollow and plain.lanes[1].components == 1  # not vacuous
    for dy in (-6.0, 0.5, 6.0):
        found = detect(case, saturated_at=limit, prefer_y=case.lane_cy[1] + dy)
        assert (found.flags, found.slots) == (plain.flags, plain.slots), dy
        lane = found.lanes[1]
        assert lane.extent == plain.lanes[1].extent and lane.hollow, dy
        assert lane.components == 1 and rowdetect.bands_in(lane, -math.inf, math.inf) == 1, dy


@pytest.mark.parametrize("saturated_at", [0.0, None])
@pytest.mark.parametrize(
    ("depth", "rel", "dy"),
    [(2.0, 2.0, 16.0), (2.0, 2.0, -16.0), (1.6, 3.0, 16.0), (2.0, 1.0, 14.0)],
)
def test_prefer_y_stops_at_a_band_as_saturated_as_the_target(depth, rel, dy, saturated_at):
    # Every lane's band clipped at 0, and another row, rel times as deep and
    # as clipped, dy px below it (above if negative). The two clipped tops
    # differ in the signal by the background plane alone, a hair: the other
    # band, a hair lower, was not higher than the target's, and the growth
    # climbed into it, one 54 x 31 box over both rows in every lane and no
    # flag. Growth stops at the valley to any other band, whether or not
    # the saturation level is known: each lane is boxed on its own band
    # alone, the other band a second component.
    case = _row_beside(dy, rel, depths={i: depth * MEMBRANE for i in range(6)})
    kwargs = {} if saturated_at is None else {"saturated_at": saturated_at}
    found = detect(case, prefer_y=_target_row(case), **kwargs)
    check_invariants(case, found)
    assert found.flags == ("multiple_components",)
    for lane, slot, cy in zip(found.lanes, found.slots, case.lane_cy, strict=True):
        y = cy + 0.5
        assert _holds(lane.extent, y) and not _holds(lane.extent, y + dy), lane
        assert _holds(slot, y) and not _holds(slot, y + dy), slot
        assert lane.components == 2


@pytest.mark.parametrize("light_on_dark", [False, True])
@pytest.mark.parametrize(("depth", "rel", "dy"), [(2.5, 2.0, 14.0), (1.3, 3.0, -12.0)])
def test_prefer_y_stops_at_a_clipped_band_across_a_shallow_valley(depth, rel, dy, light_on_dark):
    # Every lane's band clipped, and another row rel times as deep, clipped
    # too, dy px below it (above if negative): the two clipped cores lie
    # apart, but the valley between them in the smoothed signal falls only
    # to some 85% of their clipped height, above VALLEY_FRAC, so each lane
    # held one peak, the growth took in both rows (55 x 30, 54 x 26), and
    # nothing was flagged. Clipped, a peak's height is not its band's, and
    # the valley is deeper than it shows: with the saturation level known,
    # two clipped cores with a valley of DETECT_K sigma between them are two
    # peaks, two bands. Each lane is boxed on its own band alone, the other a
    # second component and another band along the lane.
    case = _row_beside(dy, rel, depths={i: depth * MEMBRANE for i in range(6)})
    limit = 0.0
    if light_on_dark:
        case = dataclasses.replace(case, image=FULL_SCALE - case.image, dark_on_light=False)
        limit = FULL_SCALE
    found = detect(case, prefer_y=_target_row(case), saturated_at=limit)
    check_invariants(case, found)
    assert found.flags == ("multiple_components",)
    for lane, slot, cy in zip(found.lanes, found.slots, case.lane_cy, strict=True):
        y = cy + 0.5
        assert _holds(lane.extent, y) and not _holds(lane.extent, y + dy), lane
        assert _holds(slot, y) and not _holds(slot, y + dy), slot
        assert lane.components == 2
        assert rowdetect.bands_in(lane, -math.inf, math.inf) == 2
    # Without the saturation level, or without an expected row, as before.
    assert detect(case, prefer_y=_target_row(case)).flags == ()
    assert detect(case, saturated_at=limit).flags == ()


_ANOTHER_BAND = re.compile(
    r"(?P<lanes>lanes? [\d, ]+): another band lies (?P<lo>\d+)(?: to (?P<hi>\d+))? px"
    r" (?P<way>above|below) the box centre; the box does not include it"
)


def _another_band(note: str) -> tuple[str, int, int, str]:
    """The lanes, the px range and the way of the note naming another band
    outside the boxes (the whole note)."""
    match = _ANOTHER_BAND.fullmatch(note)
    assert match, note
    lo = int(match["lo"])
    return match["lanes"], lo, int(match["hi"] or lo), match["way"]


def test_prefer_y_stops_at_a_saturated_band_on_a_shaded_membrane():
    # The bands of lanes 2 to 6 clipped at 0 (lane 1's 0.9 times as deep as
    # the membrane, not clipped), another row 1.5 times as deep and clipped
    # 16 px below, on a membrane darkening 60 levels a row downward: in the
    # signal the other row's clipped tops lie some 960 levels, about eight
    # noise sigmas, below the target's. Without the saturation level the
    # other band read as clearly lower, and the growth took it in where the
    # valley to it was shallow. Growth stops at the valley to any other band
    # (#58), whichever is higher, with or without the saturation
    # level: each lane boxed on its own band alone, the other band a second
    # component named below the box centre: 12 to 14 px, its peak where the
    # shading puts it, at its clipped core's top row, some 3 px above its
    # middle.
    def shade(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        return 60.0 * (Y - float(np.mean(lcy))) + 0.0 * X

    depths = {i: 2.0 * MEMBRANE if i else 0.9 * MEMBRANE for i in range(6)}
    case = _row_beside(16.0, 1.5, depths=depths, artefacts=[shade])
    for kwargs in ({"saturated_at": 0.0}, {}):
        found = detect(case, prefer_y=_target_row(case), **kwargs)
        check_invariants(case, found)
        assert found.flags == ("multiple_components",), kwargs
        for lane, slot, cy in zip(found.lanes, found.slots, case.lane_cy, strict=True):
            assert _holds(slot, cy + 0.5) and not _holds(slot, cy + 16.5), (kwargs, slot)
            assert _holds(lane.extent, cy + 0.5) and not _holds(lane.extent, cy + 16.5), kwargs
            assert lane.components == 2
        [note] = found.notes
        lanes, lo, hi, way = _another_band(note)
        assert (lanes, way) == ("lanes 1, 2, 3, 4, 5, 6", "below"), note
        assert 11 <= lo <= hi <= 17, note
        for lane in found.lanes:
            [other] = [peak for peak in lane.peaks if not peak.own]
            centre = (lane.rect[1] + lane.rect[3]) / 2
            assert lo <= math.floor(other.y - centre + 0.5) <= hi, (note, other)


def test_prefer_y_stops_at_another_band_stronger_or_weaker_than_the_target():
    # #58: the targets 8000 to 48000 deep, another band 24000 deep 12
    # px below each: stronger than the target in lanes 1 to 3, weaker in lanes
    # 4 to 6. Growth used to stop only at a band not clearly lower than the
    # target's, so lanes 4 to 6 boxed both and boxes across the row held
    # different bands (fold changes off). It stops at the valley to any other
    # band: every box holds its target alone, the other band another band
    # along the lane, named in one note with how far below the box centre.
    targets = (8000.0, 12000.0, 18000.0, 26000.0, 36000.0, 48000.0)
    rels = {i: 24000.0 / d for i, d in enumerate(targets)}
    case = _row_beside(12.0, 1.0, depths=dict(enumerate(targets)), neighbour_rels=rels)
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    assert found.flags == ("multiple_components",)
    [note] = found.notes
    lanes, lo, hi, way = _another_band(note)
    assert (lanes, way) == ("lanes 1, 2, 3, 4, 5, 6", "below"), note
    assert 10 <= lo <= hi <= 14, note
    for lane, cy in zip(found.lanes, case.lane_cy, strict=True):
        y, other_y = cy + 0.5, cy + 12.5
        assert lane.components == 2
        assert _holds(lane.extent, y) and not _holds(lane.extent, other_y), lane
        assert _holds(lane.rect, y) and not _holds(lane.rect, other_y), lane
        [other] = [peak for peak in lane.peaks if abs(peak.y - other_y) < 2.5]
        assert (other.own, other.other_band) == (False, True)
        assert rowdetect.bands_in(lane, -math.inf, math.inf) == 2


def _mirrored(case: RowCase) -> RowCase:
    """The row flipped left to right: lane i becomes lane n - 1 - i."""
    width = case.image.shape[1]
    n = case.n_lanes
    x0, y0, x1, y1 = case.row
    return dataclasses.replace(
        case,
        image=np.ascontiguousarray(case.image[:, ::-1]),
        row=(width - x1, y0, width - x0, y1),
        reference={
            n - 1 - i: (width - r[2], r[1], width - r[0], r[3]) for i, r in case.reference.items()
        },
        lane_cx=tuple(width - x for x in reversed(case.lane_cx)),
        lane_cy=tuple(reversed(case.lane_cy)),
    )


def _inverted(case: RowCase) -> RowCase:
    """The row as light bands on a dark membrane."""
    return dataclasses.replace(case, image=FULL_SCALE - case.image, dark_on_light=False)


_ANY_BAND = [
    *[(rel, dy, "plain") for rel in (0.45, 0.8, 1.5, 3.0) for dy in (-18, -14, -12, 12, 14, 18)],
    *[(rel, dy, way) for rel in (0.45, 3.0) for dy in (-12, 14) for way in ("mirrored", "light")],
]


@pytest.mark.parametrize(("rel", "dy", "way"), _ANY_BAND)
def test_prefer_y_stops_at_any_other_band(rel, dy, way):
    # #58: another row 0.45 to 3 times as deep 12 to 18 px above or
    # below the expected one; also flipped left to right, and as light bands
    # on a dark membrane. Where a lane's band is a separate peak, its extent
    # and box hold it and not the other band, whatever its depth, and the
    # other band is reported (a second component, or another band along the
    # lane). Growth used to take in a band clearly weaker than the target's.
    case = _row_beside(float(dy), rel)
    if way == "mirrored":
        case = _mirrored(case)
    elif way == "light":
        case = _inverted(case)
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    ys = [cy + 0.5 for cy in case.lane_cy]
    separate = [
        any(abs(peak.y - y) <= 2.5 for peak in lane.peaks)
        for lane, y in zip(found.lanes, ys, strict=True)
    ]
    if abs(dy) >= 14:  # not vacuous: from 14 px on, every lane's band is a peak
        assert all(separate), found
    if found.refused:
        assert rel > 1.0, found  # a weaker band never refuses the row
        return
    for lane, y, alone in zip(found.lanes, ys, separate, strict=True):
        if not alone:
            continue
        other_y = y + dy
        assert _holds(lane.extent, y) and not _holds(lane.extent, other_y), lane
        assert _holds(lane.rect, y) and not _holds(lane.rect, other_y), lane
        assert lane.components > 1 or any(peak.other_band for peak in lane.peaks), lane
        assert not any(peak.own and abs(peak.y - other_y) < 2.5 for peak in lane.peaks), lane


def _band_at(lane: int, dy: float, depth: float, w: float, h: float):
    """A flat-topped band ``w`` x ``h`` px at 20%, ``depth`` deep, ``dy`` px
    below ``lane``'s band centre (above if negative)."""
    ux4, ux2 = (2.0 * math.log(5.0)) ** 0.25, math.sqrt(2.0 * math.log(5.0))

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        across = np.exp(-0.5 * np.abs((X - lcx[lane]) / (w / (2.0 * ux4))) ** 4)
        down = np.exp(-0.5 * ((Y - lcy[lane] - dy) / (h / (2.0 * ux2))) ** 2)
        return depth * across * down

    return f


@pytest.mark.parametrize(
    ("pitch", "w", "dy", "seed"), [(20.0, 22.0, -15.0, 1001), (20.0, 22.0, 15.0, 1000)]
)
def test_prefer_y_seeds_a_lane_whose_band_touches_a_stronger_neighbour(pitch, w, dy, seed):
    # Eleven touching lanes 20 px apart, bands 22 px wide; lane 3's band 6 px
    # high and half as deep as its neighbours' (24 px high), a faint band 15
    # px above or below it in lane 3 alone. The shallow valley from lane 3's
    # band to its neighbours' leaves it no separate peak of the row, so lane
    # 3 grew from the faint band, the only peak in its seed range, its box 15
    # px off with no flag (the boxes 24 px high pass for a row). Its own
    # separate peaks seed it too: lane 3 is boxed on its band.
    depths = {i: 15000.0 if i == 2 else 30000.0 for i in range(11)}
    case = adversarial_row(
        "touching",
        seed,
        n=11,
        pitch=pitch,
        w=w,
        h=6.0,
        heights={i: 24.0 for i in range(11) if i != 2},
        depths=depths,
        artefacts=[_band_at(2, dy, 9000.0, w, 6.0)],
        img_h=200,
        margin_left=30,
        margin_right=30,
        x_jitter=0.5,
        mx=4,
    )
    cy = _target_row(case)
    x0, _, x1, _ = case.row
    case = dataclasses.replace(case, row=(x0, round(cy - 36), x1, round(cy + 36)))
    found = detect(case, prefer_y=cy, saturated_at=0.0)
    check_invariants(case, found)
    assert not found.refused
    for lane, lane_cy in zip(found.lanes, case.lane_cy, strict=True):
        assert _holds(lane.rect, lane_cy + 0.5), lane
    third = found.lanes[2]
    assert not _holds(third.rect, case.lane_cy[2] + 0.5 + dy), third


_NOT_DETECTED = re.compile(
    r"(?P<lanes>lanes? [\d, ]+): the band found lies off the row's line through the other"
    r" boxes \((?P<offsets>[-\d., and]+) box heights, limit 0\.75\) and off the expected MW's"
    r" row, and no other band reaches the detection limit on that line \(SNR"
    r" (?P<snrs>[-\d., and]+), limit 6\): recorded as not detected \(n\.d\.\) at the"
    r" expected MW"
)


def _numbers(text: str) -> list[float]:
    """The numbers of "a", "a and b" or "a, b and c"."""
    return [float(v) for v in re.split(r", | and ", text)]


def _not_detected_note(found: RowDetection) -> tuple[str, list[float], list[float]]:
    """The lanes, offsets and SNRs the off_expected_row note names."""
    [note] = [note for note in found.notes if "recorded as not detected" in note]
    match = _NOT_DETECTED.fullmatch(note)
    assert match, note
    return match["lanes"], _numbers(match["offsets"]), _numbers(match["snrs"])


def _assert_not_detected(found: RowDetection, lanes: Sequence[int]) -> None:
    """``lanes`` (0-based) recorded as not detected, the row placed: the
    first placement's off_row_line flag and note replaced by the record's."""
    assert not found.refused and found.crossed == ()
    assert "off_expected_row" in found.flags and "off_row_line" not in found.flags
    assert not any("box centre more than" in note for note in found.notes), found.notes
    for i in lanes:
        lane = found.lanes[i]
        assert (lane.rect, lane.extent, lane.reason) == (None, None, "off_expected_row")
        assert abs(lane.line_offset) > ROW_LINE_K and lane.snr < DETECT_K
        assert lane.window is not None and not lane.cut and lane.components == 0
    named, offsets, snrs = _not_detected_note(found)
    assert named == lanes_phrase(list(lanes))
    for i, offset, snr in zip(lanes, offsets, snrs, strict=True):
        assert offset == pytest.approx(abs(found.lanes[i].line_offset), abs=0.051)
        assert snr == pytest.approx(found.lanes[i].snr, abs=0.051)


@pytest.mark.parametrize("speck", [None, 32.0])
@pytest.mark.parametrize("dy", [-16.0, 16.0])
def test_prefer_y_marks_a_knocked_out_lane_not_detected(dy, speck):
    # Lane 3 holds no band at the expected row (a knockout), another row
    # twice as deep lies dy px away: lane 3 grows from that row's band, its
    # only one, off the row's line through the other boxes and off the
    # expected row, and no other band reaches the detection limit on the
    # line there. It is recorded as not detected at the expected MW, with no
    # box, and the row is placed (it used to be refused). Its neighbours hold
    # a band on both rows, but lane 3 none on theirs: no lanes cross. A speck
    # of dust at the row's height 32 px beside lane 3's centre, outside its
    # band's x-range and its slot, is no band of lane 3: not detected all
    # the same.
    artefacts = [] if speck is None else [blob(2, 4.0, 15000.0, dx=speck)]
    case = _row_beside(dy, 2.0, missing=(2,), artefacts=artefacts)
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    _assert_not_detected(found, [2])
    for i, (lane, cy) in enumerate(zip(found.lanes, case.lane_cy, strict=True)):
        if i != 2:
            assert _holds(lane.rect, cy + 0.5) and not _holds(lane.rect, cy + 0.5 + dy), lane
    assert found.size.height <= _lone_height(1000) + 1


@pytest.mark.parametrize("dy", [-16.0, 16.0])
def test_a_speck_in_a_knocked_out_lane_is_no_seed_and_keeps_the_row_refused(dy):
    # The knockout above, the speck 24 px beside lane 3's centre, at the
    # expected row: past the side of the lane's band's x-range, in its margin.
    # Read in the x-range alone, the speck's flank rose to its side, a peak
    # nearest the expected row, and lane 3 was boxed on the speck, a 4 x 1
    # extent. Its row holds nothing at the lane's middle: no peak of the
    # lane, which grows from the other row's band. Within the slot an n.d.
    # record would cover on the row's line,
    # the speck reaches the detection limit: the row stays refused.
    case = _row_beside(dy, 2.0, missing=(2,), artefacts=[blob(2, 4.0, 15000.0, dx=24.0)])
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    assert found.refused and found.off_cause == "line_signal"
    lane = found.lanes[2]
    assert _holds(lane.extent, case.lane_cy[2] + 0.5 + dy), lane
    assert lane.extent[3] - lane.extent[1] > 6 and lane.extent[2] - lane.extent[0] > 30, lane
    assert (lane.line_reason, lane.line_snr > DETECT_K) == ("unassigned", True)


def _slot_row(case: RowCase, half: int = 36) -> RowCase:
    """The row box over the lanes' span and ``half`` px above and below the
    expected row, as a slot placed by an expected MW searches it."""
    cy = round(_target_row(case))
    x0, _, x1, _ = case.row
    return dataclasses.replace(case, row=(x0, cy - half, x1, cy + half))


def _knocked_out(
    depth: float, dy: float, missing: Sequence[int], noise: float = 400.0, seed: int = 1000
) -> tuple[RowCase, float]:
    """A row in its MW slot (:func:`mw_slot_row`) and the expected row: a
    non-specific band ``depth`` deep ``dy`` px from every lane's band, the
    bands of ``missing`` knocked out, membrane noise ``noise``."""
    return mw_slot_row(
        "knocked out",
        seed,
        neighbour_dy=dy,
        neighbour_depths=[depth] * 6,
        missing=missing,
        noise=noise,
    )


@pytest.mark.parametrize("noise", [0.0, 25.0, 400.0])
@pytest.mark.parametrize(
    ("depth", "dy", "missing"),
    [(24000.0, 12.0, (1,)), (12000.0, -12.0, (1,)), (48000.0, 12.0, (1, 2))],
)
def test_a_knocked_out_lane_is_not_detected_at_low_noise(depth, dy, missing, noise):
    # Lanes knocked out, a non-specific band 12 px from every lane's band:
    # the knocked-out lanes grow from it. With no noise or little, the
    # non-specific band's own tail on the row's line reaches the detection
    # limit there (8 sigma or so at noise 25); it is its own band's, not
    # another band's, and is left out (by its reflection about the band's
    # centre): the knocked-out lanes are recorded as not detected at any
    # noise level, and the row is placed.
    case, expected = _knocked_out(depth, dy, missing, noise)
    found = detect(case, prefer_y=expected, saturated_at=0.0)
    check_invariants(case, found)
    _assert_not_detected(found, list(missing))
    for i, (lane, cy) in enumerate(zip(found.lanes, case.lane_cy, strict=True)):
        if i not in missing:
            assert _holds(lane.rect, cy + 0.5) and not _holds(lane.rect, cy + 0.5 + dy), lane


def test_an_off_lane_does_not_size_the_other_boxes():
    # Lane 3 knocked out, a band 20 px high 18 px above its row in lane 3
    # alone: lane 3 grows from it, off the row's line. Recorded as not
    # detected, its extent sizes no box: every other box is a lone band's
    # height, as if lane 3 held nothing.
    case = _slot_row(
        adversarial_row(
            "tall off", 1000, missing=(2,), artefacts=[_band_at(2, -18.0, 24000.0, 44.0, 20.0)]
        )
    )
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    _assert_not_detected(found, [2])
    assert found.size.height <= _lone_height(1000) + 1
    assert "size_outlier" not in found.flags


def test_a_lane_recorded_not_detected_is_never_named_cut_or_outlier(monkeypatch):
    # Lane 3 knocked out, a band 46 px high 18 px below its row in lane 3
    # alone, reaching the row box's bottom edge: lane 3 grows from it, off
    # the row's line, its extent cut by the box and over twice the others'
    # height (refused, cut, a size outlier). Its tail on the row's line,
    # whose reflection the box's edge cuts off, keeps the row refused; read
    # as nothing there instead, lane 3 is recorded as not detected, and the
    # row is placed again without it: named neither cut nor a size outlier,
    # the boxes placed the other lanes', none cut, a lone band's size.
    case = _slot_row(
        adversarial_row(
            "cut off", 1000, missing=(2,), artefacts=[_band_at(2, 18.0, 24000.0, 44.0, 46.0)]
        )
    )
    refused = detect(case, prefer_y=_target_row(case))
    assert refused.refused and refused.off_cause == "line_signal"
    assert {"size_outlier", "cut_by_row_box"} <= set(refused.flags)  # not vacuous
    assert refused.lanes[2].cut
    real = rowdetect._on_the_line

    def nothing(*args, **kwargs):
        _, _, window = real(*args, **kwargs)
        return "no_band", 1.0, window

    monkeypatch.setattr(rowdetect, "_on_the_line", nothing)
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    _assert_not_detected(found, [2])
    assert not any(lane.cut for lane in found.lanes)
    assert "cut_by_row_box" not in found.flags and "size_outlier" not in found.flags
    assert not any(
        "cuts through" in note or "left out of the shared size" in note for note in found.notes
    )
    assert found.size.height <= _lone_height(1000) + 1


@pytest.mark.parametrize(
    ("offsets", "expected", "on_line", "outcome"),
    [
        # 5 boxes of 7 lanes (2 empty), 3 off the row's line: half or more.
        (
            {0: 0.0, 1: 1.2, 3: -1.1, 4: 0.1, 6: 1.3},
            {0: 0.0, 1: 1.2, 3: -1.1, 4: 0.1, 6: 1.3},
            {},
            ("half", [1, 3, 6]),
        ),
        # 7 boxes, the same 3 off: fewer than half, recorded.
        (
            {0: 0.0, 1: 1.2, 2: 0.2, 3: -1.1, 4: 0.1, 5: -0.3, 6: 1.3},
            {0: 0.0, 1: 1.2, 2: 0.2, 3: -1.1, 4: 0.1, 5: -0.3, 6: 1.3},
            {1: "no_band", 3: "no_band", 6: "no_band"},
            (None, [1, 3, 6]),
        ),
        # 6 boxes, 3 off: half.
        (
            {i: (1.0 if i < 3 else 0.0) for i in range(6)},
            {i: (1.0 if i < 3 else 0.0) for i in range(6)},
            {},
            ("half", [0, 1, 2]),
        ),
        # Off the line but on the expected row (0.75 is on it): the line runs elsewhere.
        (
            {0: 0.0, 1: 0.9, 2: 0.0, 3: 0.0, 4: 0.0},
            {0: 0.9, 1: 0.75, 2: 0.9, 3: 0.9, 4: 0.9},
            {1: "no_band"},
            ("expected_row", [1]),
        ),
        # Off both, the other lane off the line on it (above): named, first.
        (
            {0: 0.0, 1: -0.9, 2: 0.0, 3: 0.0, 4: 1.2, 5: 0.0, 6: 0.0},
            {0: 0.0, 1: -0.76, 2: 0.0, 3: 0.0, 4: -0.7, 5: 0.0, 6: 0.0},
            {1: "unassigned", 4: "no_band"},
            ("expected_row", [4]),
        ),
        # Another band on the line in lane 2 (and a line in lane 4): kept refused.
        (
            {0: 0.0, 1: 0.9, 2: 0.0, 3: -0.8, 4: 0.0},
            {0: 0.0, 1: 0.9, 2: 0.0, 3: -0.8, 4: 0.0},
            {1: "unassigned", 3: "line"},
            ("line_signal", [1, 3]),
        ),
        # The line out of the rows searched in lane 2.
        (
            {0: 0.0, 1: 0.9, 2: 0.0, 3: 0.0, 4: 0.0},
            {0: 0.0, 1: 0.9, 2: 0.0, 3: 0.0, 4: 0.0},
            {1: "outside"},
            ("line_signal", [1]),
        ),
        # A box that stays lies off the expected row (just past 0.75, above).
        (
            {0: 0.0, 1: 0.9, 2: 0.0, 3: 0.0, 4: 0.0},
            {0: 0.0, 1: 0.9, 2: 0.0, 3: -0.76, 4: 0.0},
            {1: "no_band"},
            ("other_box", [3]),
        ),
        # The same 0.75 below: on the row, recorded.
        (
            {0: 0.0, 1: 0.9, 2: 0.0, 3: 0.0, 4: 0.0},
            {0: 0.0, 1: 0.9, 2: 0.0, 3: 0.75, 4: 0.0},
            {1: "no_band"},
            (None, [1]),
        ),
        # A box exactly 0.75 off the line (on the expected row) is on the line:
        # not off it, so not one whose box on the expected row says the line
        # runs elsewhere.
        (
            {0: 0.0, 1: 0.9, 2: 0.0, 3: -0.75, 4: 0.0},
            {0: 0.0, 1: 0.9, 2: 0.0, 3: 0.0, 4: 0.0},
            {1: "no_band"},
            (None, [1]),
        ),
    ],
    ids=[
        "half-of-boxed",
        "fewer-than-half",
        "half",
        "on-row",
        "on-row-first",
        "line-signal",
        "line-outside",
        "off-row",
        "off-row-limit",
        "off-line-limit",
    ],
)
def test_off_lanes_are_not_detected_only_when_every_condition_holds(
    offsets, expected, on_line, outcome
):
    # The lanes off the row's line are recorded as not detected only when
    # fewer than half of the boxes are off (of the boxes, not the declared
    # lanes), each lies off the expected row too, no other band reaches the
    # detection limit on the line in it, and every other box lies on the
    # expected row; else the first condition that fails says why not.
    assert rowdetect._not_detected(offsets, expected, on_line) == outcome


@pytest.mark.parametrize(
    ("missing", "off"), [((1, 2, 3), [0, 4, 5]), ((0, 2, 4), [0, 1, 2, 3, 4, 5])]
)
def test_half_the_lanes_knocked_out_refuses_the_row(missing, off):
    # Three lanes of six knocked out, a non-specific band 48000 deep 26 px
    # below every lane's: half of the boxes, or all, lie off the row's line
    # (here the protein's: the line runs through the other band's boxes), so
    # the line cannot tell the protein's row from the other. Refused, none
    # recorded. (No row of the synthetic sets reaches this refusal; 16 px
    # apart, the line bends through both rows' boxes and leaves a box on the
    # expected row off it.)
    case, expected = _knocked_out(48000.0, 26.0, missing)
    found = detect(case, prefer_y=expected, saturated_at=0.0)
    check_invariants(case, found)
    assert found.refused and "off_row_line" in found.flags and "off_expected_row" not in found.flags
    assert found.off_cause == "half" and found.crossed == ()
    assert not any(lane.reason == "off_expected_row" for lane in found.lanes)
    assert [lane.lane for lane in found.lanes if abs(lane.line_offset) > ROW_LINE_K] == off
    assert sum(lane.rect is not None for lane in found.lanes) == 6


def test_the_not_detected_lane_s_reading_is_its_region_s():
    # Lane 2 knocked out, a non-specific band 24000 deep 12 px below every
    # lane's, little noise: on the row's line in lane 2 the non-specific
    # band's own tail reaches some 8 noise sigmas, which its reflection about
    # the band's centre takes out. The lane's SNR is read on the line, in the
    # four rows its region spans: a speck of about 5 sigmas there counts, the
    # same speck 10 px below the line, outside the region, does not.
    case, expected = _knocked_out(24000.0, 12.0, (1,), 25.0)
    found = detect(case, prefer_y=expected, saturated_at=0.0)
    _assert_not_detected(found, [1])
    lane = found.lanes[1]
    assert lane.snr < 3.0
    wx0, wy0, wx1, wy1 = lane.window
    assert wy1 - wy0 == 4  # the rows the reading interpolates between, smoothed 3 high
    line = float(np.mean([(r[1] + r[3]) / 2 for r in found.slots if r is not None]))
    assert wy0 + 1.0 <= line <= wy0 + 3.0
    assert wx1 - wx0 == pytest.approx(2 * EMPTY_WINDOW * found.pitch, abs=2.0)
    # A speck 2 px across: 3 x 5 smoothing keeps 0.73 of its depth.
    depth = 5.0 * found.noise / 0.733
    ys, xs = np.mgrid[0 : case.image.shape[0], 0 : case.image.shape[1]]
    for dy, low, high in ((0.0, 4.0, 6.0), (10.0, -math.inf, 3.0)):
        speck = depth * np.exp(
            -0.5 * (((xs - 0.5 * (wx0 + wx1)) / 2.0) ** 2 + ((ys - line - dy) / 2.0) ** 2)
        )
        specked = dataclasses.replace(case, image=np.round(case.image - speck))
        again = detect(specked, prefer_y=expected, saturated_at=0.0)
        _assert_not_detected(again, [1])
        assert low <= again.lanes[1].snr < high, (dy, again.lanes[1].snr)
        assert again.lanes[1].window == lane.window


def test_an_off_lane_s_own_band_is_the_one_it_grew_from_of_two_in_its_window():
    # Lane 2 knocked out, a non-specific band 12 px above every lane's, and
    # another band as deep 20 px below the row in lane 2 alone, little
    # noise: lane 2's window holds two bands, and it grows from the one
    # above, nearer the expected row. Its own band is the pixels draining to
    # that one: on the row's line its tail is taken out by its reflection
    # about it, and lane 2 is recorded as not detected. Read with the other
    # band as its own, the tail would stay (some 26 noise sigmas).
    case, expected = _knocked_out(24000.0, -12.0, (1,), 25.0)
    ys, xs = np.mgrid[0 : case.image.shape[0], 0 : case.image.shape[1]]
    below = _band_at(1, 20.0, 24000.0, 44.0, 12.0)
    image = case.image - below(xs, ys, np.array(case.lane_cx), np.array(case.lane_cy))
    case = dataclasses.replace(case, image=np.round(image))
    found = detect(case, prefer_y=expected, saturated_at=0.0)
    check_invariants(case, found)
    _assert_not_detected(found, [1])
    assert found.lanes[1].snr < 3.0


def test_a_row_still_off_its_line_placed_again_is_refused(monkeypatch):
    # Lane 3 off the row's line and recorded as not detected, the row is
    # placed again without it; were a box still off the line then (the
    # second fit made to leave lane 6 1.1 box heights off it), the row is
    # refused: no lane recorded, the first fit's boxes reported, and why.
    # Lane 1, exactly 0.75 box heights off the second line, is on it: not
    # named.
    case = _row_beside(16.0, 2.0, missing=(2,))
    fits = []
    real = rowdetect._row_line

    def row_line(centres, height, lanes=()):
        offsets = real(centres, height, lanes)
        fits.append(list(lanes))
        if len(fits) == 2:
            offsets = offsets.copy()
            offsets[-1] = 1.1
            offsets[0] = -0.75
        return offsets

    monkeypatch.setattr(rowdetect, "_row_line", row_line)
    found = detect(case, prefer_y=_target_row(case))
    assert fits == [[0, 1, 2, 3, 4, 5], [0, 1, 3, 4, 5]]
    assert found.refused and "off_row_line" in found.flags and "off_expected_row" not in found.flags
    assert found.lanes[2].rect is not None and found.lanes[2].reason == "band"
    assert abs(found.lanes[2].line_offset) > ROW_LINE_K
    assert abs(found.lanes[5].line_offset) <= ROW_LINE_K  # the first fit's
    assert (found.off_cause, found.again) == ("again", ((5, 1.1),))
    assert found.lanes[2].line_reason == "no_band"  # it would have been recorded
    assert [note for note in found.notes if "recorded as not detected" in note] == []


def _line_with(monkeypatch, at: dict[int, dict[int, float]]) -> list[list[int]]:
    """Make the row's line fit number k (from 1) leave lane i ``at[k][i]`` box
    heights off it, the other lanes as fitted; returns the lanes of each fit."""
    fits: list[list[int]] = []
    real = rowdetect._row_line

    def row_line(centres, height, lanes=()):
        offsets = real(centres, height, lanes)
        fits.append(list(lanes))
        if len(fits) in at:
            offsets = offsets.copy()
            for i, v in at[len(fits)].items():
                offsets[list(lanes).index(i)] = v
        return offsets

    monkeypatch.setattr(rowdetect, "_row_line", row_line)
    return fits


@pytest.mark.parametrize(
    ("value", "limit", "places", "shown"),
    [
        (1.234, 0.75, 2, "1.23"),
        (0.75, 0.75, 2, "0.75"),
        (0.751, 0.75, 2, "0.751"),  # past the limit: never shown as it
        (0.7504, 0.75, 2, "0.7504"),
        (0.7496, 0.75, 2, "0.7496"),  # short of it
        (0.7449, 0.75, 2, "0.74"),
        (5.94, 6.0, 1, "5.9"),
        (5.96, 6.0, 1, "5.96"),
        (6.0, 6.0, 1, "6.0"),
        (6.04, 6.0, 1, "6.04"),
        (61.25, 6.0, 1, "61.2"),
    ],
)
def test_shown_against_keeps_a_number_on_its_side_of_the_limit(value, limit, places, shown):
    # A number printed with the limit it was compared with lies on the
    # side of the limit the value does: to more decimals where fewer would
    # round it onto the limit.
    assert rowdetect.shown_against(value, limit, places) == shown


@pytest.mark.parametrize(
    ("offset", "snr", "words"),
    [
        (0.7504, 5.96, "(0.7504 box heights, limit 0.75) and off the expected MW's row, and no"),
        (1.2345, 2.345, "(1.23 box heights, limit 0.75) and off the expected MW's row, and no"),
    ],
)
def test_a_not_detected_lane_s_numbers_just_past_their_limits_are_not_shown_as_them(
    monkeypatch, offset, snr, words
):
    # Lane 3 knocked out, its box 0.7504 box heights off the row's line,
    # its reading on the line 5.96 noise sigmas: recorded as not detected,
    # the note says 0.7504 (limit 0.75) and 5.96 (limit 6), not 0.75 and 6.0;
    # farther from the limits, 2 decimals and 1.
    case = _row_beside(16.0, 2.0, missing=(2,))
    _line_with(monkeypatch, {1: {2: offset}})
    real = rowdetect._on_the_line

    def reading(*args, **kwargs):
        _, _, window = real(*args, **kwargs)
        return "no_band", snr, window

    monkeypatch.setattr(rowdetect, "_on_the_line", reading)
    found = detect(case, prefer_y=_target_row(case))
    _assert_not_detected(found, [2])
    [note] = [note for note in found.notes if "recorded as not detected" in note]
    shown = {5.96: "5.96", 2.345: "2.3"}[snr]
    assert words in note and f"(SNR {shown}, limit 6)" in note, note


def test_a_box_exactly_the_limit_off_the_row_s_line_is_on_it(monkeypatch):
    # ROW_LINE_K (0.75 box heights) off the row's line is on it, in every
    # fit: a row whose one box lies that far off is placed with no
    # off_row_line; with lane 3 knocked out and off the line, a box that far
    # off in the first fit is neither read on the line nor recorded as not
    # detected with lane 3, and that far off in the second it keeps the row
    # placed.
    plain = adversarial_row("on the limit", 1000)
    fits = _line_with(monkeypatch, {1: {3: 0.75}})
    found = detect(plain, prefer_y=_target_row(plain))
    assert len(fits) == 1 and not found.refused and found.flags == (), found.flags
    case = _row_beside(16.0, 2.0, missing=(2,))
    for at in ({1: {4: -0.75}}, {2: {4: 0.75}}):
        fits = _line_with(monkeypatch, at)
        found = detect(case, prefer_y=_target_row(case))
        assert fits == [[0, 1, 2, 3, 4, 5], [0, 1, 3, 4, 5]], (at, fits)
        _assert_not_detected(found, [2])
        assert found.lanes[4].rect is not None and found.lanes[4].line_reason is None, at


_A_BAND_LEFT_OUT = re.compile(
    r"(?P<lanes>lanes? [\d, ]+): another band lies (?P<lo>\d+)(?: to (?P<hi>\d+))? px"
    r" (?P<way>above|below) the box centre(?P<inside>, inside the box: the box holds part of"
    r" it|; the box does not include it)"
)


def test_prefer_y_names_another_band_the_box_leaves_out():
    # #58: the note on a second component says where the other band
    # lies, from the box's centre, and whether the box holds part of it.
    # Above and below: another row 0.6 times as deep 14 px above every
    # lane's band, and a band 16 px below lanes 4 to 6 too: one clause per
    # way, above first, lanes in order, the px as a range when they differ.
    below = [_band_at(i, 16.0, 12000.0, 44.0, 12.0) for i in (3, 4, 5)]
    case = adversarial_row(
        "above and below",
        1000,
        neighbour_dy=-14.0,
        neighbour_rel=0.6,
        artefacts=below,
        box_adjust=(0, -30, 0, 30),
        img_h=200,
    )
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    assert found.flags == ("multiple_components",)
    [note] = found.notes
    parts = re.findall(
        r"lanes? [\d, ]+: another band lies [^;]+; the box does not include it", note
    )
    assert "; ".join(parts) == note, note
    named = [_another_band(part) for part in parts]
    assert [(lanes, way) for lanes, _, _, way in named] == [
        ("lanes 1, 2, 3, 4, 5, 6", "above"),
        ("lanes 4, 5, 6", "below"),
    ], note
    (_, lo_up, hi_up, _), (_, lo_down, hi_down, _) = named
    assert 12 <= lo_up < hi_up <= 16 and 14 <= lo_down < hi_down <= 18, note  # ranges
    for lane, cy in zip(found.lanes, case.lane_cy, strict=True):
        assert not _holds(lane.rect, cy + 0.5 - 14.0) and not _holds(lane.rect, cy + 16.5)


def test_prefer_y_names_another_band_inside_or_beside_the_box():
    # The box holds part of another band: lanes 1 and 6 hold bands 18 px
    # high, which set the box's height (16 px); lanes 2 to 5 hold bands 8 px
    # high, another band 8 px below each. Where that band's peak falls in the
    # box's rows (lanes 2, 3), the box holds part of it; else (lanes 4, 5)
    # the box does not include it: a clause for each, the box's own first.
    case = adversarial_row(
        "inside",
        1000,
        h=8.0,
        heights={0: 18.0, 5: 18.0},
        artefacts=[_band_at(i, 8.0, 18000.0, 44.0, 8.0) for i in range(1, 5)],
        box_adjust=(0, -20, 0, 20),
    )
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    assert found.flags == ("multiple_components",)
    [note] = found.notes
    clauses = note.split("; the box does not include it; ")
    assert len(clauses) == 2, note
    outside, inside = (
        _A_BAND_LEFT_OUT.fullmatch(clauses[0] + "; the box does not include it"),
        (_A_BAND_LEFT_OUT.fullmatch(clauses[1])),
    )
    assert outside and inside and inside["inside"].startswith(", inside"), note
    assert (outside["lanes"], inside["lanes"]) == ("lanes 4, 5", "lanes 2, 3"), note
    assert outside["way"] == inside["way"] == "below"
    for match, held in ((outside, False), (inside, True)):
        assert 7 <= int(match["lo"]) <= int(match["hi"] or match["lo"]) <= 9, note
        for i in (1, 2) if held else (3, 4):
            lane = found.lanes[i]
            [other] = [peak for peak in lane.peaks if not peak.own]
            assert (lane.rect[1] <= other.y < lane.rect[3]) == held, (lane, other)
    # Beside it: lanes 140 px apart, a stain in lane 4 46 px right of its
    # centre, past the box's side, in its rows.
    wide = adversarial_row(
        "beside",
        1000,
        pitch=140.0,
        artefacts=[blob(3, 6.0, 20000.0, dx=46.0)],
        box_adjust=(0, -20, 0, 20),
    )
    found = detect(wide, prefer_y=_target_row(wide))
    check_invariants(wide, found)
    assert found.lanes[3].components == 2
    [note] = [note for note in found.notes if "another band" in note]
    assert note == (
        "lane 4: another band lies beside the box, 0 px below its centre; the box does not"
        " include it"
    )
    [stain] = [peak for peak in found.lanes[3].peaks if not peak.own]
    rect = found.lanes[3].rect
    assert rect[1] <= stain.y < rect[3] and stain.x >= rect[2]


@pytest.mark.parametrize(
    ("dx", "edge", "words"),
    [
        # peaking on the box's first column: in the box
        (-33.0, 0, "lies 0 px below the box centre, inside the box: the box holds part of it"),
        # on the column just past its last: beside it
        (33.25, 2, "lies beside the box, 0 px below its centre; the box does not include it"),
    ],
)
def test_prefer_y_reads_another_band_on_the_box_s_side_column_by_its_pixel(dx, edge, words):
    # Lanes 140 px apart, bands 70 px wide but lane 4's, 30 px: its box, as
    # wide as the others, reaches 20 px past its band either side. A stain
    # beside the band, in its rows, peaks on the box's first column, or on
    # the column just past its last: a pixel's column is in the box where
    # the box's columns hold it.
    widths = dict.fromkeys(range(6), 70.0)
    widths[3] = 30.0
    case = adversarial_row(
        "side column",
        1000,
        pitch=140.0,
        widths=widths,
        artefacts=[blob(3, 5.0, 20000.0, dx=dx)],
        box_adjust=(0, -20, 0, 20),
    )
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    lane = found.lanes[3]
    [stain] = [peak for peak in lane.peaks if not peak.own]
    assert stain.x == lane.rect[edge] + 0.5 and lane.rect[1] <= stain.y < lane.rect[3]
    assert lane.components == 2
    [note] = [note for note in found.notes if "another band" in note]
    assert note == f"lane 4: another band {words}"


def test_other_bands_note_words_each_kind_and_way():
    # The note from the boxes and second components alone (crop at (100,
    # 50)): every box 40 x 12 at x 100..140, rows 50..62, centre row 56.
    # Above them, 12.5 px (to the nearest px, half up: 13) and 10 px; below,
    # past the box's bottom row, 15.4 px; inside one, 3.4 px below; in their
    # rows beside one, right of it 2.5 px above and left of it 0.5 px below.
    box = (100, 50, 140, 62)
    note = rowdetect._other_bands_note(
        {
            0: (box, [(-6.5, 20.0)]),  # image y 43.5
            2: (box, [(-4.0, 20.0), (21.4, 20.0)]),  # y 46, and y 71.4
            3: (box, [(9.4, 30.0)]),  # y 59.4, x 130: inside
            5: (box, [(3.5, 40.0), (6.5, -0.5)]),  # x 140 and 99.5: beside
        },
        100,
        50,
    )
    assert note == (
        "lanes 1, 3: another band lies 10 to 13 px above the box centre; the box does not"
        " include it; lane 3: another band lies 15 px below the box centre; the box does not"
        " include it; lane 4: another band lies 3 px below the box centre, inside the box:"
        " the box holds part of it; lane 6: another band lies beside the box, 3 px above its"
        " centre; the box does not include it; lane 6: another band lies beside the box,"
        " 1 px below its centre; the box does not include it"
    )
    # The box's last row (61.x) is in it; its bottom edge (62) is not.
    inside = rowdetect._other_bands_note({0: (box, [(11.9, 20.0)])}, 100, 50)
    edge = rowdetect._other_bands_note({0: (box, [(12.0, 20.0)])}, 100, 50)
    assert inside.endswith("6 px below the box centre, inside the box: the box holds part of it")
    assert edge.endswith("6 px below the box centre; the box does not include it")
    # The top row (50) is in it.
    assert ", inside the box" in rowdetect._other_bands_note({0: (box, [(0.0, 20.0)])}, 100, 50)
    assert "inside" not in rowdetect._other_bands_note({0: (box, [(-0.1, 20.0)])}, 100, 50)
    # Level with the box's centre (56), beside it: 0 px, above (not below).
    assert rowdetect._other_bands_note({0: (box, [(6.0, -0.5)])}, 100, 50) == (
        "lane 1: another band lies beside the box, 0 px above its centre; the box does not"
        " include it"
    )


@pytest.mark.parametrize("seed", [1001, 1003])
def test_a_lane_on_the_expected_row_but_off_the_bent_line_refuses(seed):
    # Lanes 2 and 3 knocked out, a non-specific band 48000 deep 12 px below
    # every lane's: lanes 2 and 3 are boxed on it, and the row's line, bent
    # through those boxes, leaves the box of a lane that holds the protein,
    # on the expected row, off it (lane 6; at seed 1003 lanes 5 and 6). That
    # box on the expected row says the line runs through another row: none
    # is recorded as not detected, and the row is refused.
    case, expected = _knocked_out(48000.0, 12.0, (1, 2), seed=seed)
    found = detect(case, prefer_y=expected, saturated_at=0.0)
    check_invariants(case, found)
    assert found.refused and found.off_cause == "expected_row"
    on_row = [
        lane.lane
        for lane in found.lanes
        if abs(lane.line_offset) > ROW_LINE_K and abs(lane.expected_offset) <= ROW_LINE_K
    ]
    assert on_row == {1001: [5], 1003: [4, 5]}[seed]
    for i in on_row:
        assert _holds(found.lanes[i].rect, case.lane_cy[i] + 0.5)  # the protein's band


@pytest.mark.parametrize("noise", [0.0, 25.0, 400.0])
@pytest.mark.parametrize("dy", [-12.0, 12.0])
def test_a_band_on_the_line_keeps_the_row_refused(dy, noise):
    # Targets 8000 to 48000 deep, another band 24000 deep 12 px from each:
    # lane 1's target, a shoulder on the deeper band with no peak of its
    # own, is no seed, and lane 1 is boxed on the other band, off the row's
    # line. On the line there lies the target, read past the other band's
    # own tail by some 60 noise sigmas at noise 400 (more with less noise):
    # it is not recorded as not detected, and the row is refused.
    targets = [8000.0, 12000.0, 18000.0, 26000.0, 36000.0, 48000.0]
    case, expected = mw_slot_row(
        "shoulder",
        1000,
        depths=targets,
        neighbour_dy=dy,
        neighbour_depths=[24000.0] * 6,
        noise=noise,
    )
    found = detect(case, prefer_y=expected, saturated_at=0.0)
    check_invariants(case, found)
    assert found.refused and found.off_cause == "line_signal"
    lane = found.lanes[0]
    assert _holds(lane.rect, case.lane_cy[0] + 0.5 + dy) and abs(lane.line_offset) > ROW_LINE_K
    assert lane.line_reason == "unassigned" and lane.line_snr > 40.0
    assert [ld.lane for ld in found.lanes if ld.line_reason is not None] == [0]


_ENDS = [
    (dy, noise, seed)
    for dy in (-22.0, -18.0, 18.0, 22.0)
    for noise in (400.0, 25.0)
    for seed in (1000, 1001, 1002, 1003, 1004)
]


@pytest.mark.parametrize(("dy", "noise", "seed"), _ENDS)
def test_a_box_off_the_expected_row_keeps_the_row_refused(dy, noise, seed):
    # Five lanes, lanes 1 to 3 holding the protein at the expected row,
    # lanes 4 and 5 none, a band 24000 deep 18 to 22 px above or below in
    # each of those two. The row's line passes lane 5's box for a smile and
    # leaves lane 4's off it: lane 4 alone would be recorded as not
    # detected, beside lane 5's box on that other band. A box that stays off
    # the expected row says the line is not the expected row's: refused.
    # Nearer, 12 to 16 px, the line may take both boxes in as a smile (#201,
    # the next test).
    case, expected = mw_slot_row(
        "ends",
        seed,
        n=5,
        missing=(3, 4),
        neighbour_dy=dy,
        neighbour_depths=[0.0, 0.0, 0.0, 24000.0, 24000.0],
        noise=noise,
    )
    found = detect(case, prefer_y=expected, saturated_at=0.0)
    check_invariants(case, found)
    assert found.refused and found.off_cause == "other_box"
    assert [lane.lane for lane in found.lanes if abs(lane.line_offset) > ROW_LINE_K] == [3]
    fifth = found.lanes[4]
    assert abs(fifth.line_offset) <= ROW_LINE_K < abs(fifth.expected_offset)
    assert _holds(fifth.rect, case.lane_cy[4] + 0.5 + dy)  # on the other band


@pytest.mark.xfail(
    strict=True,
    reason="known limit (#201): two knocked-out end lanes beside a band 12 to 16 px off,"
    " within the MW tolerance, pass for a smile and are boxed on it with no flag",
)
@pytest.mark.parametrize(
    ("dy", "noise", "seed"),
    [
        (-12.0, 400.0, 1000),
        (12.0, 400.0, 1000),
        (-14.0, 400.0, 1001),
        (14.0, 400.0, 1000),
        (-14.0, 25.0, 1003),
        (14.0, 25.0, 1004),
        (-16.0, 400.0, 1002),
        (16.0, 400.0, 1000),
        (16.0, 25.0, 1000),
    ],
)
def test_two_knocked_out_end_lanes_beside_a_band_within_the_tolerance(dy, noise, seed):
    # As above, the band 12 to 16 px off (in 2 of 10 rows at 16 px, 6 or 8
    # of 10 at 14, nearly every row at 12): the row's line takes both end
    # lanes' boxes in as a smile, and both are boxed on that band with no
    # flag. A knocked-out lane should be recorded as not detected, or the
    # row refused.
    case, expected = mw_slot_row(
        "ends",
        seed,
        n=5,
        missing=(3, 4),
        neighbour_dy=dy,
        neighbour_depths=[0.0, 0.0, 0.0, 24000.0, 24000.0],
        noise=noise,
    )
    found = detect(case, prefer_y=expected, saturated_at=0.0)
    assert found.refused or all(found.lanes[i].rect is None for i in (3, 4)), found.flags


def _line_pass(s: np.ndarray) -> SimpleNamespace:
    """A detection pass over the smoothed signal ``s`` (noise sigma 1), its
    rows all the row's, nothing rejected, no line or side signal."""
    none = np.zeros(s.shape, bool)
    cand = SimpleNamespace(
        rows=(0, s.shape[0]), dropped=none, lines=none, side=none, dust=none, rejected=[]
    )
    return SimpleNamespace(sig=SimpleNamespace(s_sm=s, sigma_sm=1.0), cand=cand)


def _column_band(rows: int, cols: int, centre: float, depth: float, sigma: float) -> np.ndarray:
    """A band ``depth`` high at row ``centre`` (index units) in every column."""
    y = np.arange(rows, dtype=float)
    return np.outer(depth * np.exp(-0.5 * ((y - centre) / sigma) ** 2), np.ones(cols))


def test_on_the_line_leaves_out_the_own_band_s_tail_by_its_reflection():
    # The lane's own band at row 10 (100 high, sigma 3) in every column; the
    # row's line at row 16 (continuous 16.5), where the band's tail is 13.5
    # high. Its reflection about the band's centre (row 4) is as high: the
    # line holds nothing else, SNR 0, no band, though the tail in the four
    # rows read (15 to 18) reaches DETECT_K. The slot: within 0.3 pitch of
    # the box's centre (x 5, pitch 10: columns 2 to 7).
    s = _column_band(40, 10, 10.0, 100.0, 3.0)
    own = s > 1e-6
    res = _line_pass(s)
    surfaces = rowdetect._slot_surfaces(res)
    reason, snr, window = rowdetect._on_the_line(res, surfaces, own, 10, (5.0, 10.5), 16.5, 10.0)
    assert (reason, window) == ("no_band", (2, 15, 8, 19))
    assert snr == pytest.approx(0.0, abs=1e-9)
    # Another band on the line, 20 high, apart from the own band: it reads.
    s2 = s + _column_band(40, 10, 16.0, 20.0, 1.0)
    res2 = _line_pass(s2)
    reason, snr, _ = rowdetect._on_the_line(
        res2,
        rowdetect._slot_surfaces(res2),
        own & (s2 > 0) & (np.arange(40) < 14)[:, None],
        10,
        (5.0, 10.5),
        16.5,
        10.0,
    )
    assert reason == "unassigned" and snr == pytest.approx(20.0 + 100.0 * math.exp(-2.0), abs=0.01)


def test_on_the_line_finds_the_own_band_s_centre_on_the_seed_s_side():
    # The lane grew from a band 50 high at row 10; joined to it, with no peak
    # of its own there, a band 100 high lies on the row's line (row 22). The
    # own band's centre is read on the seed's side of row 16, midway: at row
    # 10, so the band on the line reads whole. Read over the whole column,
    # the centre would be the band on the line's, and its reflection would
    # take it all out: recorded as not detected over the protein.
    s = _column_band(40, 6, 10.0, 50.0, 3.0) + _column_band(40, 6, 22.0, 100.0, 3.0)
    own = s > 1e-6
    res = _line_pass(s)
    reason, snr, _ = rowdetect._on_the_line(
        res, rowdetect._slot_surfaces(res), own, 10, (3.0, 10.5), 22.5, 10.0
    )
    assert reason == "unassigned" and snr > 95.0
    # The seed below the line: its side is below row 16.
    flipped = s[::-1].copy()
    res = _line_pass(flipped)
    reason, snr, _ = rowdetect._on_the_line(
        res, rowdetect._slot_surfaces(res), own[::-1], 29, (3.0, 30.5), 17.5, 10.0
    )
    assert reason == "unassigned" and snr > 95.0


def test_on_the_line_outside_the_rows_reads_nothing():
    # The line at a lane off the row's line can run past the row box's rows:
    # nothing on it is read, and nothing recorded (reason outside).
    s = _column_band(20, 10, 10.0, 100.0, 3.0)
    own = s > 1e-6
    res = _line_pass(s)
    surfaces = rowdetect._slot_surfaces(res)

    def read(line: float) -> tuple[str, float, tuple[int, int, int, int] | None]:
        return rowdetect._on_the_line(res, surfaces, own, 10, (5.0, 10.5), line, 10.0)

    assert read(0.4) == ("outside", 0.0, None)  # index -0.1
    assert read(20.6) == ("outside", 0.0, None)  # index 20.1, past row 19
    assert read(0.5)[2] == (2, 0, 8, 3)  # row 0: the rows read, clipped
    assert read(19.5)[2] == (2, 18, 8, 20)  # row 19
    # The columns read stop at the crop's sides: from column 0, to column 10.
    assert rowdetect._on_the_line(res, surfaces, own, 10, (1.0, 10.5), 16.5, 10.0)[2] == (
        0,
        15,
        4,
        19,
    )
    assert rowdetect._on_the_line(res, surfaces, own, 10, (8.0, 10.5), 16.5, 10.0)[2] == (
        5,
        15,
        10,
        19,
    )


@pytest.mark.parametrize("flip", [False, True])
def test_on_the_line_reads_the_own_band_s_centre_short_of_the_row_midway(flip):
    # One column, its own band rising from the seed (row 10) to the row's
    # line (row 22) and falling past it: no peak of its own on the seed's
    # side. Its centre is read on the rows short of the row midway (16):
    # the top of rows 0 to 15 (15, at row 15), moved to the parabola's
    # vertex through rows 14 to 16 (row 16 read as 0): row 14.5625. The
    # reflection of row 22 about it, row 7.125, holds 7.125: the line reads
    # 22 - 7.125 = 14.875. Mirrored (the seed below the line), the same.
    s = (22.0 - np.abs(np.arange(40.0) - 22.0))[:, None]
    seed, line = 10, 22.5
    if flip:
        s, seed, line = s[::-1].copy(), 29, 17.5
    res = _line_pass(s)
    reason, snr, _ = rowdetect._on_the_line(
        res, rowdetect._slot_surfaces(res), s > 0.0, seed, (0.5, 10.5), line, 1.0
    )
    assert (reason, snr) == ("unassigned", pytest.approx(14.875, abs=1e-9))


def test_on_the_line_reads_all_of_a_column_whose_own_band_has_no_centre_on_the_seed_s_side():
    # Two columns: in column 0 the own band peaks at the seed's row (10),
    # holding nothing on the row's line (row 22); in column 1 it lies only
    # past the row midway (16), peaking at row 27.5, 9 high on the line. No
    # centre on the seed's side in column 1: nothing is taken out there, and
    # the line reads 9.
    y = np.arange(40.0)
    s = np.stack(
        [
            np.maximum(0.0, 10.0 - 2.0 * np.abs(y - 10.0)),
            np.maximum(0.0, 20.0 - 2.0 * np.abs(y - 27.5)),
        ],
        axis=1,
    )
    res = _line_pass(s)
    reason, snr, _ = rowdetect._on_the_line(
        res, rowdetect._slot_surfaces(res), s > 0.0, 10, (1.0, 10.5), 22.5, 10.0
    )
    assert (reason, snr) == ("unassigned", pytest.approx(9.0, abs=1e-9))


def test_centre_row_reads_a_band_s_top_plateau_by_its_middle():
    # The middle of the top plateau: the pixels within a relative 1e-9 of
    # the top (equal up to rounding); a lone top moved to the parabola's
    # vertex; None for a column holding nothing above 0.
    centre = rowdetect._centre_row
    assert centre(np.array([0.0, 5.0, 10.0, 10.0, 10.0, 5.0, 0.0])) == 3.0
    assert centre(np.array([7.0, 7.0, 3.0])) == 0.5  # at the column's first row
    assert centre(np.array([0.0, 4.0, 10.0, 6.0, 0.0])) == pytest.approx(2.1)  # toward the 6
    top = 10.0
    assert centre(np.array([0.0, 5.0, top, top, top - 1e-9 * top, 5.0, 0.0])) == 3.0
    assert centre(np.array([0.0, 5.0, top, top, top - 1.2e-8, 5.0, 0.0])) == 2.5
    assert centre(np.zeros(5)) is None
    assert centre(np.array([0.0, 0.5, 0.0])) == 1.0


def test_read_at_interpolates_within_the_column_and_reads_0_outside():
    column = np.array([5.0, 6.0, 7.0])
    assert rowdetect._read_at(column, 0.0) == 5.0
    assert rowdetect._read_at(column, 1.5) == 6.5
    assert rowdetect._read_at(column, 2.0) == 7.0
    assert rowdetect._read_at(column, -0.1) == 0.0
    assert rowdetect._read_at(column, 2.1) == 0.0


_SLOT = (40, 10)  # rows, columns
_STAIN = ("artefact", (slice(10, 20), slice(2, 5)))  # a streak's rows and columns


@pytest.mark.parametrize(
    ("region", "snr", "line", "reason"),
    [
        ((slice(19, 23), slice(4, 6)), 0.0, 0.0, "artefact"),  # its last row and column
        ((slice(0, None), slice(0, 10)), 0.0, 0.0, "artefact"),  # every row (the crop's 40)
        ((slice(20, 24), slice(0, 10)), 0.0, 0.0, "no_band"),  # the rows below it
        ((slice(6, 10), slice(0, 10)), 0.0, 0.0, "no_band"),  # the rows above it
        ((slice(10, 14), slice(5, 8)), 0.0, 0.0, "no_band"),  # the columns right of it
        ((slice(10, 14), slice(0, 2)), 0.0, 0.0, "no_band"),  # the columns left of it
        ((slice(20, 24), slice(0, 10)), 6.0, 0.0, "unassigned"),  # DETECT_K reached
        ((slice(20, 24), slice(0, 10)), 0.0, 6.0, "line"),
        ((slice(20, 24), slice(0, 10)), 0.0, 5.9, "no_band"),
    ],
)
def test_slot_reason_reads_a_streak_by_the_rows_and_columns_it_shares(region, snr, line, reason):
    # Why a slot (rows, columns of the crop) leaves its lane empty: a streak
    # or stain over any of its rows and columns; else its SNR reaching
    # DETECT_K (6); else a line reaching it in the slot.
    s = np.zeros(_SLOT)
    lines = np.zeros(_SLOT)
    lines[21, :] = line
    res = SimpleNamespace(
        sig=SimpleNamespace(s_sm=s, sigma_sm=1.0), cand=SimpleNamespace(rejected=[_STAIN])
    )
    surfaces = rowdetect._Surfaces(row=s, all=s, line=lines, side=s)
    assert rowdetect._slot_reason(res, surfaces, snr, region) == reason


def test_an_empty_lane_s_slot_reads_the_box_s_edge_rows_too():
    # Two empty lanes, the row's rows 1 to 19 (row 0 left out): signal 7
    # sigmas high on row 0 alone in lane 1's slot makes lane 1 edge_signal
    # (signal at the box's edge), lane 2 no band.
    s = np.zeros((20, 20))
    s[0, 3:6] = 7.0
    none = np.zeros(s.shape, bool)
    lanes = [rowdetect._Lane(), rowdetect._Lane()]
    cand = SimpleNamespace(
        rows=(1, 20), dropped=none, lines=none, side=none, dust=none, cut=none, rejected=[]
    )
    res = SimpleNamespace(
        sig=SimpleNamespace(s_sm=s, sigma_sm=1.0), cand=cand, lanes=lanes, assign=None
    )
    rowdetect._empty_lanes(res, 2, 20, 10.0)
    assert [(ln.reason, ln.snr, ln.window) for ln in lanes] == [
        ("edge_signal", 0.0, (2, 1, 8, 20)),
        ("no_band", 0.0, (12, 1, 18, 20)),
    ]


def test_beside_the_lane_tells_a_speck_beside_it_from_a_band_over_it():
    # A seed range [10, 20), its middle column 15; a peak of the range alone
    # on its last or first column whose row, at the middle, holds under
    # EXTENT_LEVEL (30%) of it lies beside the lane (a speck's flank past
    # the side); 30% or more there, the band reaches over the lane's middle.
    ks = np.zeros((3, 40))
    ks[1, 19] = 100.0
    ks[1, 15] = 29.0
    assert rowdetect._beside_the_lane(ks, 1, 19, 10, 20)
    ks[1, 15] = 30.0
    assert not rowdetect._beside_the_lane(ks, 1, 19, 10, 20)
    ks[1, 10], ks[1, 15] = 100.0, 29.9
    assert rowdetect._beside_the_lane(ks, 1, 10, 10, 20)
    ks[1, 11] = ks[1, 18] = 100.0  # as high, the middle as low: an inner column, never
    assert not rowdetect._beside_the_lane(ks, 1, 11, 10, 20)
    assert not rowdetect._beside_the_lane(ks, 1, 18, 10, 20)
    ks[0, 15] = 0.0  # the middle read in the peak's own row, not another
    ks[2, 15] = 100.0
    assert rowdetect._beside_the_lane(ks, 1, 10, 10, 20)
    # An odd range [10, 21): middle column 15.
    ks[1, 20], ks[1, 15], ks[1, 16] = 100.0, 31.0, 0.0
    assert not rowdetect._beside_the_lane(ks, 1, 20, 10, 21)


def _basin_pairs(basin: np.ndarray) -> tuple[int, int]:
    """The pixel pairs across two basins' boundary: side by side, stacked."""
    pairs = []
    for a, b in ((basin[:, :-1], basin[:, 1:]), (basin[:-1, :], basin[1:, :])):
        pairs.append(int(np.count_nonzero((a != b) & (a > 0) & (b > 0))))
    return pairs[0], pairs[1]


def test_band_basins_joins_tops_whose_valley_runs_as_much_down_the_rows_as_along_them():
    # A band sloping 45 degrees, lighter in its middle (a dumbbell): its two
    # tops' basins meet along a staircase, as many pixel pairs side by side
    # as stacked. One band.
    window = np.array(
        [
            [97.1, 80.1, 45.1, 16.9, 0.0, 0.0, 0.0],
            [80.1, 90.3, 67.9, 33.7, 11.5, 0.0, 0.0],
            [45.1, 67.9, 67.5, 46.3, 25.0, 11.6, 0.0],
            [16.9, 33.7, 46.3, 50.0, 46.6, 34.2, 17.3],
            [0.0, 11.5, 25.0, 46.6, 68.4, 69.3, 46.3],
            [0.0, 0.0, 11.6, 34.2, 69.3, 92.8, 82.8],
            [0.0, 0.0, 0.0, 17.3, 46.3, 82.8, 101.1],
        ]
    )
    basin, band = rowdetect._band_basins(window, [(0, 0), (6, 6)], 30.0)
    assert _basin_pairs(basin) == (3, 3)
    assert band == [0, 0]


def test_band_basins_are_four_connected():
    # A band above and to the left touching the one below only at a corner:
    # another band. The basins are 4-connected, as the pixel pairs counted
    # across them and the components that tell tops apart.
    window = np.array([[9.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 8.0]])
    basin, band = rowdetect._band_basins(window, [(0, 0), (2, 2)], 2.7)
    assert basin.tolist() == [[1, 0, 0], [0, 2, 2], [0, 0, 2]]
    assert band == [0, 1]


def _hump(rows: int, cols: int) -> np.ndarray:
    """A small hump ``rows`` x ``cols`` px, highest at its centre."""
    y = 1.0 - np.abs(np.linspace(-0.8, 0.8, rows))
    x = 1.0 - np.abs(np.linspace(-0.8, 0.8, cols))
    return 10.0 * np.outer(y, x)


@pytest.mark.parametrize(
    ("second", "one"),
    [
        ((0, 4), True),  # a band's other half, split off by a centre below the noise
        ((1, 4), True),  # ... a row lower: 2 of the 4 rows the two span (JOIN_ROWS)
        ((2, 4), False),  # ... two rows lower: 1 of 5, a band beside it
        ((4, 0), False),  # a band below it, apart
    ],
)
def test_band_basins_reads_tops_apart_by_the_rows_they_share(second, one):
    # Two humps 3 x 3 px with no signal between them: one band when they
    # share JOIN_ROWS of the rows the two span, as _join_burnt_out joins a
    # band's halves; else two.
    window = np.zeros((7, 7))
    window[0:3, 0:3] = _hump(3, 3)
    dy, dx = second
    window[dy : dy + 3, dx : dx + 3] = _hump(3, 3) * 1.01
    _, band = rowdetect._band_basins(window, [(1, 1), (dy + 1, dx + 1)], 0.3)
    assert band == ([0, 0] if one else [0, 1])


def test_band_basins_does_not_join_a_stroke_across_a_band_s_rows():
    # A stroke 9 rows high beside a band 3 rows high, apart: it shares 3 of
    # the 9 rows the two span. Two bands.
    window = np.zeros((9, 7))
    window[3:6, 0:3] = _hump(3, 3)
    window[:, 5:6] = _hump(9, 1) * 1.01
    _, band = rowdetect._band_basins(window, [(4, 1), (4, 5)], 0.3)
    assert band == [0, 1]


def test_band_basins_reads_tops_apart_by_the_rows_their_pieces_span_at_the_growth_level():
    # Two humps side by side with no signal between them, a faint tail (1.0)
    # hanging 9 rows below the left one. Its basin, and the part of the
    # signal it lies in, span the tail's rows too, 12 against the right
    # hump's 3. Above the growth threshold (3.0) each hump's piece is its
    # centre, on one row: one band. Were the tail above it (a threshold of
    # 0.5), the left piece would span 12 rows: two.
    window = np.zeros((12, 7))
    window[0:3, 0:3] = _hump(3, 3)
    window[0:3, 4:7] = _hump(3, 3) * 1.01
    window[3:12, 1] = 1.0
    tops = [(1, 1), (1, 5)]
    basin, band = rowdetect._band_basins(window, tops, 3.0)
    assert [(r.start, r.stop) for r, _ in find_objects(basin)] == [(0, 12), (0, 3)]
    assert band == [0, 0]
    _, band = rowdetect._band_basins(window, tops, 0.5)
    assert band == [0, 1]
    # A faint hump (2.0 high) apart on the same rows: below the growth
    # threshold it has no piece, and no growth reaches it: not joined.
    window = np.pad(window, ((0, 0), (0, 4)))
    window[0:3, 8:11] = _hump(3, 3) * 0.2
    _, band = rowdetect._band_basins(window, [*tops, (1, 9)], 3.0)
    assert band == [0, 0, 2]


def test_band_basins_reads_tops_touching_at_a_corner_by_their_rows():
    # Two humps side by side on the same rows whose signal touches only at a
    # corner (no 4-connected path between them): read by their rows, as
    # tops with no signal between them, one band. Read as joined by signal,
    # their basins would not meet at all, and they would be two.
    window = np.zeros((3, 7))
    window[0:3, 0:3] = _hump(3, 3)
    window[0:3, 4:7] = _hump(3, 3) * 1.01
    window[0:2, 2] = 0.0
    window[2, 3] = 0.0
    window[1, 3] = 0.3  # touches (2, 2) at a corner only
    _, band = rowdetect._band_basins(window, [(1, 1), (1, 5)], 0.2)
    assert band == [0, 0]


def test_without_other_bands_takes_out_every_band_but_the_seed_s():
    # One pixel per top, each its own basin, the values their heights. The
    # seed (top 2) is of band 1, with top 5 beside it; bands 0, 3 and 4 are
    # others, higher or lower, whatever their heights: taken out, each
    # pixel draining to them set to 0. The seed's band's other top stays,
    # however high, and so do pixels draining to no top (basin 0).
    window = np.array([[50.0], [140.0], [100.0], [99.0], [20.0], [130.0], [7.0]])
    basin = np.array([[1], [2], [3], [4], [5], [6], [0]])
    band = [0, 1, 1, 3, 4, 1]  # top k's band: tops 1, 2 and 5 are one band
    kept = rowdetect._without_other_bands(window, basin, band, 2)
    assert kept[:, 0].tolist() == [0.0, 140.0, 100.0, 0.0, 0.0, 130.0, 7.0]
    assert window[:, 0].tolist() == [50.0, 140.0, 100.0, 99.0, 20.0, 130.0, 7.0]  # not written
    # Seeded from top 0, band 0's alone: every other top's basin goes.
    kept = rowdetect._without_other_bands(window, basin, band, 0)
    assert kept[:, 0].tolist() == [50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]
    # Every top of the seed's band: the window itself.
    assert rowdetect._without_other_bands(window, basin, [0] * 6, 3) is window
    # One top: no basins (_band_basins), the window itself.
    assert rowdetect._without_other_bands(window, None, [0], 0) is window


def test_other_row_takes_the_own_band_of_two_tops_as_near():
    tops = [(10.5, False), (12.5, True)]
    assert not rowdetect._other_row(tops, 11.5)  # as near: the band it grew from
    assert not rowdetect._other_row(tops[::-1], 11.5)
    assert rowdetect._other_row(tops, 11.0)
    assert not rowdetect._other_row(tops, 12.0)


def test_band_basins_reads_tops_joined_by_signal_by_their_valley():
    # Two humps side by side, joined only by a loop of signal below them (a
    # smear), whose lowest point lies on a stretch running down the rows: the
    # valley between their basins runs along the rows, one pixel pair
    # stacked. Tops joined by signal are read by their valley, two bands,
    # although they share their rows: the rows decide only for tops with no
    # signal between them.
    window = np.zeros((6, 9))
    window[0:3, 0:3] = _hump(3, 3)
    window[0:3, 6:9] = _hump(3, 3) * 1.01
    window[3:6, 1] = [9.0, 8.0, 7.0]  # down from the left hump
    window[5, 2:7] = [5.5, 5.0, 4.5, 4.0, 3.5]  # along the bottom
    window[3:6, 7] = [9.0, 1.0, 3.2]  # up to the right hump, lowest at row 4
    basin, band = rowdetect._band_basins(window, [(1, 1), (1, 7)], 3.0)
    assert _basin_pairs(basin) == (0, 1)
    assert band == [0, 1]


def test_band_basins_joins_tops_whose_basins_meet_at_one_pixel_pair_side_by_side():
    # One row: two tops, the pixel between them in one basin, so the basins
    # meet at a single pair of pixels side by side, none stacked: one band.
    basin, band = rowdetect._band_basins(np.array([[9.0, 5.0, 8.0]]), [(0, 0), (0, 2)], 1.0)
    assert _basin_pairs(basin) == (1, 0)
    assert band == [0, 0]


def test_band_basins_reads_pieces_above_the_threshold_only():
    # Two humps apart on rows 0 to 2 (their centres 10 high, the rest 2 or
    # less), a tail exactly at the growth threshold (3.0) hanging from the
    # left one's lower edge. Above the threshold, each hump's piece is its
    # centre on row 1: one band. Were the tail in it, the left piece would
    # span rows 1 to 11 against the right's one: two.
    window = np.zeros((12, 7))
    window[0:3, 0:3] = _hump(3, 3)
    window[0:3, 4:7] = _hump(3, 3) * 1.01
    window[2:12, 1] = 3.0
    _, band = rowdetect._band_basins(window, [(1, 1), (1, 5)], 3.0)
    assert band == [0, 0]


def test_band_basins_reads_each_top_by_its_own_piece_s_rows():
    # Three humps apart: two side by side on rows 0 to 2, one under the left
    # on rows 6 to 8. The two side by side are one band, the one under it
    # another: each top read by the rows of the piece that holds it.
    window = np.zeros((9, 7))
    window[0:3, 0:3] = _hump(3, 3)
    window[0:3, 4:7] = _hump(3, 3) * 1.01
    window[6:9, 0:3] = _hump(3, 3) * 1.02
    _, band = rowdetect._band_basins(window, [(1, 1), (1, 5), (7, 1)], 0.3)
    assert band == [0, 0, 2]


@pytest.mark.parametrize(
    ("a", "b", "side_by_side"),
    [
        ((0, 4), (1, 5), True),  # 3 of the 5 rows the two span
        ((0, 4), (2, 6), False),  # 2 of 6
        ((0, 10), (3, 7), False),  # 4 of 10
        ((0, 4), (0, 2), True),  # 2 of 4: JOIN_ROWS exactly
        ((2, 6), (0, 6), True),  # 4 of 6
        ((5, 9), (0, 6), False),  # 1 of 9
        ((0, 2), (1, 2), True),  # 1 of 2
        ((0, 3), (1, 4), True),  # 2 of 4
    ],
)
def test_share_rows_reads_the_rows_two_things_share_of_those_they_span(a, b, side_by_side):
    assert rowdetect._share_rows(a, b) is side_by_side
    assert rowdetect._share_rows(b, a) is side_by_side


def test_step_is_the_shift_that_best_matches_the_rows():
    # A band one row lower in the second lane: 1; one row higher: -1; the
    # same rows, two bands each: 0, though a shift of 2 matches one band.
    assert rowdetect._step(np.array([0.0, 0, 1, 0, 0, 0]), np.array([0.0, 0, 0, 1, 0, 0])) == 1
    assert rowdetect._step(np.array([0.0, 0, 0, 1, 0, 0]), np.array([0.0, 0, 1, 0, 0, 0])) == -1
    assert rowdetect._step(np.array([0.0, 1, 0, 1, 0, 0]), np.array([0.0, 1, 0, 1, 0, 0])) == 0


def _row_under(dy: float, depth: float):
    """A flat-topped band 44 x 12 px at 20% ``depth`` deep ``dy`` px below each
    lane's band."""
    ux4, ux2 = (2.0 * math.log(5.0)) ** 0.25, math.sqrt(2.0 * math.log(5.0))

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        out = np.zeros((Y.shape[0], X.shape[1]))
        for cx, cy in zip(lcx, lcy, strict=True):
            across = np.exp(-0.5 * np.abs((X - cx) / (44.0 / (2.0 * ux4))) ** 4)
            down = np.exp(-0.5 * ((Y - cy - dy) / (12.0 / (2.0 * ux2))) ** 2)
            out += depth * across * down
        return out

    return f


def test_prefer_y_counts_another_band_as_a_component_from_a_share_of_the_lane_s_peak():
    # Every band 12000 deep, a row three times as deep 16 px above it, and a
    # band 7200 deep 12 px below it. The growth stops at the valley to each
    # (it used to take in the one below, clearly weaker): the extent holds
    # neither. The band above is a second component; the one below reaches a
    # fifth of the lane's peak, the band above's: under SECOND_SHARE, not one.
    case = adversarial_row(
        "three rows",
        1000,
        depths=dict.fromkeys(range(6), 12000.0),
        neighbour_dy=-16.0,
        neighbour_rel=3.0,
        box_adjust=(0, -30, 0, 20),
        artefacts=[_row_under(12.0, 7200.0)],
        img_h=200,
    )
    found = detect(case, prefer_y=_target_row(case))
    check_invariants(case, found)
    assert found.flags == ("multiple_components",)
    for lane, cy in zip(found.lanes, case.lane_cy, strict=True):
        assert _holds(lane.extent, cy + 0.5), lane
        assert not _holds(lane.extent, cy + 12.5) and not _holds(lane.extent, cy - 15.5), lane
        [under] = [peak for peak in lane.peaks if abs(peak.y - cy - 12.5) < 2.5]
        assert not under.own
        assert under.snr < 0.25 * max(peak.snr for peak in lane.peaks)
        assert lane.components == 2


def _two_bands(below: float, flip: bool) -> np.ndarray:
    """One lane 14 px wide, noise sigma 1: a band 100 high on rows 8 to 12 and
    another ``below`` lower on rows 20 to 24, the valley between them 50 high
    (rows upside down with ``flip``)."""
    column = np.zeros(40)
    column[4:9] = np.linspace(0.0, 100.0, 5)
    column[8:13] = 100.0
    column[12:17] = np.linspace(100.0, 50.0, 5)
    column[16:21] = np.linspace(50.0, 100.0 - below, 5)
    column[20:25] = 100.0 - below
    column[24:29] = np.linspace(100.0 - below, 0.0, 5)
    if flip:
        column = column[::-1]
    return np.outer(column, np.r_[np.zeros(3), np.ones(14), np.zeros(3)])


@pytest.mark.parametrize("saturated", [False, True])
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("below", [-20.0, 4.5, 6.5, 25.0])
def test_prefer_y_stops_at_the_valley_to_a_band_however_much_lower(below, flip, saturated):
    # The valley between the two bands lies above the growth threshold (30).
    # Grown from the first (the expected row's), growth stops at the valley
    # to the other, below or above it, higher (20 more), within DETECT_K
    # sigma of it, or lower by more (6.5, 25), where it used to be taken in;
    # with the bands' tops at the saturation level, too.
    s = _two_bands(below, flip)
    kept = s > 0.0
    tops = rowdetect._peaks(np.where(kept, s, 0.0), DETECT_K)
    rows = sorted(y for y, _ in tops)
    assert len(rows) == 2 and rows[1] - rows[0] >= 8  # two peaks: not vacuous
    lane = rowdetect._Lane(present=True, x_range=(0.0, 20.0), centre=10.0)
    limit = s >= min(100.0, 100.0 - below) if saturated else None
    prefer = 29.5 if flip else 10.5
    rowdetect._measure([lane], s, kept, 1.0, limit, tops, prefer)
    if flip:  # rows 27 to 31, the valley at row 23
        assert lane.rect[3] == 34 and lane.rect[1] in (23, 24), lane.rect
    else:  # rows 8 to 12, the valley at row 16
        assert lane.rect[1] == 6 and lane.rect[3] in (16, 17), lane.rect


@pytest.mark.parametrize(
    ("top", "speck", "prefer", "seed_row"),
    [
        ((19, 10), False, 20.5, 19),  # its own top a row from the crop's: that top
        ((18, 10), False, 20.5, 20),  # two rows: its own as well, nearer the expected row
        ((20, 10), True, 31.5, 20),  # a speck 8 high (4 sigmas): no peak, though nearer
    ],
)
def test_prefer_y_seeds_from_the_lane_s_own_peaks_apart_from_the_crop_s(
    top, speck, prefer, seed_row
):
    # One lane, its band peaking 100 high at row 20, noise sigma 2: its seed
    # range's peaks are the crop's top given there and its own peaks
    # (DETECT_K sigmas, 12, at the range alone), each of these more than a
    # row from the crop's; the seed is the one nearest the expected row. A
    # top of its own a row from the crop's is that top; two rows away it is
    # another. A speck 8 high (row 32) is no peak of the lane's: under 12.
    y = np.arange(40.0)[:, None]
    s = (
        100.0
        * np.exp(-0.5 * ((y - 20.0) / 3.0) ** 2)
        * np.r_[np.zeros(3), np.ones(14), np.zeros(3)]
    )
    if speck:
        s[31:34, 8:12] += 8.0
    lane = rowdetect._Lane(present=True, x_range=(0.0, 20.0), centre=10.0)
    rowdetect._measure([lane], s, s > 0.0, 2.0, None, [top], prefer)
    assert lane.seed_row == seed_row and lane.rect[1] < 20 < lane.rect[3], lane


def _measured(
    s: np.ndarray,
    ranges: Sequence[tuple[float, float]],
    prefer: float | None,
    sigma: float = 1.0,
) -> list:
    """Lanes over ``ranges`` measured on ``s`` (noise ``sigma``, all of it
    kept), seeded nearest ``prefer`` from the peaks at DETECT_K sigma, or
    from their strongest pixel without it: each lane's present, rect, seed
    row and crossed."""
    lanes = [rowdetect._Lane(present=True, x_range=r, centre=0.5 * (r[0] + r[1])) for r in ranges]
    tops = None if prefer is None else rowdetect._peaks(s, 6.0 * sigma)
    rowdetect._measure(lanes, s, s > 0.0, sigma, None, tops, prefer)
    return [(ln.present, ln.rect, ln.seed_row, ln.crossed) for ln in lanes]


@pytest.mark.parametrize("prefer", [None, 11.5])
def test_a_lane_grows_from_its_walls_and_seeds_only(prefer):
    # Lane 1 over columns 0 to 9, its band (rows 10 to 12) 60 high from
    # column 0, 100 at columns 4 to 6: grown from column 4, its extent
    # reaches column 0, the box's edge, its wall.
    s = np.zeros((20, 10))
    s[10:13, 0:10] = 60.0
    s[10:13, 4:7] = 100.0
    [(present, rect, _, _)] = _measured(s, [(0.0, 10.0)], prefer)
    assert present and rect == (0, 10, 10, 13)
    # A lane 0 to 13 beside a narrow one (14 to 17): the wall between them
    # at their centres' midpoint, 11.5; lane 1's band (6 to 13) peaks at
    # column 12, past it. Its growth window reaches past the wall to its
    # seed's column and no further: the band's column 13 stays out.
    s = np.zeros((20, 20))
    s[10:13, 6:14] = 60.0
    s[10:13, 12] = 100.0
    s[10:13, 15:18] = 100.0
    first, second = _measured(s, [(0.0, 14.0), (14.0, 18.0)], prefer)
    assert first[1] == (6, 10, 13, 13) and second[1] == (15, 10, 18, 13)


def test_prefer_y_seeds_from_a_peak_on_its_range_s_first_column():
    # Two lanes, 0 to 9 and 10 to 19; lane 2's band (rows 10 to 12) peaks at
    # (11, 10): its seed range's first column, and lane 1's wall. That peak
    # is lane 2's, not lane 1's, though nearer lane 1's expected row than
    # its own band (rows 4 to 6); each lane boxes its own band.
    s = np.zeros((20, 20))
    s[4:7, 2:8] = 90.0
    s[5, 5] = 100.0
    s[10:13, 10:18] = 90.0
    s[11, 10] = 100.0
    first, second = _measured(s, [(0.0, 10.0), (10.0, 20.0)], 11.5)
    assert first == (True, (2, 4, 8, 7), 5, False)
    assert second == (True, (10, 10, 18, 13), 11, False)
    # One lane whose band peaks on column 0, the box's edge.
    s = np.zeros((20, 10))
    s[10:13, 0:8] = 90.0
    s[11, 0] = 100.0
    assert _measured(s, [(0.0, 10.0)], 11.5) == [(True, (0, 10, 8, 13), 11, False)]


def test_prefer_y_seeds_from_the_peak_nearest_the_row_by_its_centre():
    # Bands on rows 10 and 13 (their centres 10.5 and 13.5): the expected
    # row 12.1 lies nearer 13.5; at 12.0, as near both, the higher (row 10).
    s = np.zeros((20, 10))
    s[10, 2:8] = s[13, 2:8] = 100.0
    assert _measured(s, [(0.0, 10.0)], 12.1)[0][2] == 13
    assert _measured(s, [(0.0, 10.0)], 12.0)[0][2] == 10


@pytest.mark.parametrize("prefer", [None, 11.5])
def test_a_lane_with_no_signal_in_its_range_is_empty_and_a_faint_band_is_not(prefer):
    # Lane 2's seed range holds no signal: no band, not present. A band 0.8
    # high over a noise of 0.1 (8 sigmas) is one.
    s = np.zeros((20, 20))
    s[10:13, 2:8] = 100.0
    first, second = _measured(s, [(0.0, 10.0), (10.0, 20.0)], prefer)
    assert first[0] and second[:2] == (False, None)
    s = np.zeros((20, 10))
    s[10:13, 2:8] = 0.8
    [(present, rect, _, _)] = _measured(s, [(0.0, 10.0)], prefer, sigma=0.1)
    assert present and rect == (2, 10, 8, 13)


def test_prefer_y_crosses_lanes_by_their_seeds_rows_and_tops_rows_alike():
    # Lane 1's bands on rows 8 and 10, lane 2's on rows 9 and 11 (one row
    # lower: the step). Expected at row 10.0, lane 1 grows from row 10, lane
    # 2 from row 9: moved by the step, each seed lies on the other lane's
    # other band, row for row. Crossed.
    s = np.zeros((20, 20))
    s[8, 2:8] = s[10, 2:8] = 100.0
    s[9, 12:18] = s[11, 12:18] = 100.0
    first, second = _measured(s, [(0.0, 10.0), (10.0, 20.0)], 10.0)
    assert (first[2], second[2]) == (10, 9) and first[3] and second[3]


@pytest.mark.parametrize(
    ("first_rows", "second_rows", "prefer", "seeds"),
    [
        # lane 1 grows from row 6 (6.5), midway between lane 2's bands (4.5, 8.5)
        ((4, 6), (4, 8), 6.0, (6, 4)),
        # lane 2 grows from row 6, midway between lane 1's (4.5, 8.5)
        ((4, 8), (6, 8), 7.0, (8, 6)),
    ],
)
def test_prefer_y_reads_a_seed_midway_between_two_bands_as_on_the_lane_s_own(
    first_rows, second_rows, prefer, seeds
):
    # Bands one row high over 6 columns, the step between the two lanes'
    # rows 0. A lane's seed row lying midway between the other lane's two
    # bands is read as on that lane's own band (of two as near, its own):
    # the lanes are not crossed.
    s = np.zeros((20, 20))
    for row in first_rows:
        s[row, 2:8] = 100.0
    for row in second_rows:
        s[row, 12:18] = 100.0
    first, second = _measured(s, [(0.0, 10.0), (10.0, 20.0)], prefer)
    assert (first[2], second[2]) == seeds
    assert not first[3] and not second[3]


def test_prefer_y_reads_a_lane_s_bands_in_its_seed_range_only():
    # Lane 1's seed range 0 to 7, its walls 0 and 10; lane 2's 12 to 19.
    # Lane 1 grows from row 6 of its bands (4, 6), lane 2 from row 8 of its
    # (5, 8); the step between their rows is -1. A speck on row 10 at
    # columns 8 and 9, in lane 1's growth window but past its seed range, is
    # none of lane 1's bands: read with them, it would lie nearer lane 2's
    # seed row moved by the step (9.5) than lane 1's own band, and cross the
    # lanes.
    s = np.zeros((20, 20))
    s[4, 1:7] = s[6, 1:7] = 100.0
    s[5, 13:19] = s[8, 13:19] = 100.0
    s[10, 8:10] = 100.0
    first, second = _measured(s, [(0.0, 8.0), (12.0, 20.0)], 8.0)
    assert (first[2], second[2]) == (6, 8)
    assert not first[3] and not second[3]


def test_prefer_y_marks_a_lone_band_s_whole_signal_its_own():
    # One band (rows 10 to 12, 100 high) in the lane's window, a faint tail
    # under it (rows 13 to 15, half a grey level): with one peak there are no
    # basins, and the lane's own pixels, read on the row's line, are the
    # band's part of the signal, its faint tail too.
    s = np.zeros((20, 10))
    s[10:13, 2:8] = 100.0
    s[13:16, 2:8] = 0.5
    lane = rowdetect._Lane(present=True, x_range=(0.0, 10.0), centre=5.0)
    rowdetect._measure([lane], s, s > 0.0, 1.0, None, rowdetect._peaks(s, 6.0), 11.5)
    assert lane.rect == (2, 10, 8, 13)
    assert np.array_equal(lane.own, s > 0.0)


@pytest.mark.parametrize("seed", [1000, 1002])
def test_prefer_y_reads_a_weak_band_s_peaks_as_without_it(seed):
    # Weak bands (3000 to 5000 deep) under heavy noise (sigma 1600): a dip
    # the noise leaves on a band's top, shallower than DETECT_K noise sigmas,
    # makes no second peak, with an expected row as without one.
    case = adversarial_row("weak", seed, depth_range=(3000.0, 5000.0), noise=1600.0)
    plain = detect(case)
    found = detect(case, prefer_y=_target_row(case))
    assert [len(lane.peaks) for lane in found.lanes] == [1] * 6
    assert [lane.peaks for lane in found.lanes] == [lane.peaks for lane in plain.lanes]


@pytest.mark.parametrize("slope", [0.06, -0.06])
@pytest.mark.parametrize("seed", [1001, 1002])
def test_prefer_y_keeps_a_tilted_clipped_ring_one_band(slope, seed):
    # Lane 2's band four times as deep as the membrane, clipped, a hole along
    # much of its length (24 px, 4 px high) lightened below the limit: a ring
    # of clipped pixels. Tilted 3.4 degrees, the ring's clipped pixels fall
    # in two pieces, each a rim with one end, the valley between their tops
    # shallower than VALLEY_FRAC. The pieces lie side by side, sharing their
    # rows: one band's, one peak, as without an expected row (hollow).
    # Stacked clipped pieces with a valley of DETECT_K sigma between them are
    # two bands (test_prefer_y_stops_at_a_clipped_band_across_a_shallow_valley).
    artefacts = [
        _sloping_band(1, 4.0 * MEMBRANE, slope),
        blob(1, 12.0, -4.0 * MEMBRANE, ry=2.0),
    ]
    case = adversarial_row("ring", seed, depths={1: 0.0}, artefacts=artefacts, my=10)
    plain = detect(case, saturated_at=0.0)
    assert plain.lanes[1].hollow and plain.lanes[1].components == 1  # not vacuous
    for dy in (-4.0, 0.5, 4.0):
        found = detect(case, saturated_at=0.0, prefer_y=case.lane_cy[1] + dy)
        assert found.flags == plain.flags, dy
        lane = found.lanes[1]
        assert lane.hollow and lane.components == 1, dy
        whole = plain.lanes[1].extent
        assert max(abs(a - b) for a, b in zip(lane.extent, whole, strict=True)) <= 1, dy


def _plateaus(*parts: tuple[slice, slice, float, bool]) -> tuple[np.ndarray, np.ndarray]:
    """A signal 24 x 11 px of flat parts ``(rows, cols, level, clipped)``, and
    its clipped pixels; a clipped part 98 high far below the rest (rows 20 to
    22), a band of its own."""
    s, saturated = np.zeros((24, 11)), np.zeros((24, 11), bool)
    for rows, cols, level, clipped in (*parts, (slice(20, 23), slice(0, 11), 98.0, True)):
        s[rows, cols] = level
        saturated[rows, cols] = clipped
    return s, saturated


ALL = slice(0, 11)
TOP, VALLEY, UNDER = slice(2, 5), slice(5, 8), slice(8, 11)


@pytest.mark.parametrize(
    ("parts", "extra"),
    [
        # a clipped band (99) under another (100), a valley of 80 between:
        # stacked, the second a peak of its own
        ([(TOP, ALL, 100.0, True), (VALLEY, ALL, 80.0, False), (UNDER, ALL, 99.0, True)], [ALL]),
        # the same band not clipped (95): the valley rule holds for it
        ([(TOP, ALL, 100.0, True), (VALLEY, ALL, 80.0, False), (UNDER, ALL, 95.0, False)], []),
        # a clipped band under one that is not (101): as _peaks reads it
        ([(TOP, ALL, 101.0, False), (VALLEY, ALL, 85.0, False), (UNDER, ALL, 100.0, True)], []),
        # a clipped dumbbell, its ends side by side across a middle of 80
        (
            [
                (TOP, slice(0, 4), 100.0, True),
                (TOP, slice(4, 7), 80.0, False),
                (TOP, slice(7, 11), 99.0, True),
            ],
            [],
        ),
        # a clipped dumbbell (99.5, 99) under a clipped band: its higher end
        # a peak, its other end, beside that one, the same band's
        (
            [
                (TOP, ALL, 100.0, True),
                (VALLEY, ALL, 80.0, False),
                (UNDER, slice(0, 5), 99.5, True),
                (UNDER, slice(5, 6), 90.0, False),
                (UNDER, slice(6, 11), 99.0, True),
            ],
            [slice(0, 5)],
        ),
    ],
    ids=["stacked", "not clipped", "under unclipped", "dumbbell", "dumbbell under"],
)
def test_clipped_peaks_are_clipped_bands_stacked_across_a_shallow_valley(parts, extra):
    # h = 6: every valley here lies h below the tops beside it, and above
    # VALLEY_FRAC of them, so _peaks reads one peak above the far band. The
    # peaks added lie under it, in the columns ``extra`` gives.
    s, saturated = _plateaus(*parts)
    kept = s > 0.0
    tops = rowdetect._peaks(s, 6.0)
    assert sorted(y // 10 for y, _ in tops) == [0, 2]  # rows 2 to 4, and 20 to 22
    found = rowdetect._clipped_peaks(s, kept, 6.0, saturated, tops)
    assert len(found) == len(extra)
    for (y, x), cols in zip(found, extra, strict=True):
        assert UNDER.start <= y < UNDER.stop and cols.start <= x < cols.stop


def test_clipped_peaks_read_a_hollow_band_by_the_extent_a_click_grows():
    # A clipped band (96) under another (100), a valley of 78 between, and a
    # lighter centre (32) through the under band's height and the valley
    # (columns 3 and 4), a tail (45) below it: both above EXTENT_LEVEL of
    # the under band's height (28.8), so the extent a click on it grows holds
    # them, and the centre lies between clipped pixels along its rows and
    # well below the signal above and below it: a hollow band, its halves one
    # band, no peak added. Grown to half its height instead (48), the extent
    # would leave the centre and the tail out, read no hollow band, and add
    # a peak (8, 2).
    s, saturated = _plateaus(
        (TOP, ALL, 100.0, True),
        (VALLEY, ALL, 78.0, False),
        (UNDER, ALL, 96.0, True),
        (slice(6, 11), slice(3, 5), 32.0, False),
        (slice(11, 14), slice(3, 5), 45.0, False),
    )
    tops = rowdetect._peaks(s, 6.0)
    assert not any(UNDER.start <= y < UNDER.stop for y, _ in tops)  # no peak of _peaks
    assert EXTENT_LEVEL * 96.0 < 32.0 < 45.0 < 0.5 * 96.0  # not vacuous
    assert rowdetect._clipped_peaks(s, s > 0.0, 6.0, saturated, tops) == []


def test_clipped_peaks_grow_a_click_s_extent_above_extent_level_only():
    # As above, with the lighter centre and the tail exactly at EXTENT_LEVEL
    # of the under band's height (0.3 x 96), and a patch 60 high under the
    # tail: the extent a click grows (above that level) leaves the centre
    # out, open below through the tail: no hollow band, a peak added. Were
    # pixels at the level in it, the centre would lie in the band between
    # clipped pixels, well below the signal above and below it: hollow.
    s, saturated = _plateaus(
        (TOP, ALL, 100.0, True),
        (VALLEY, ALL, 78.0, False),
        (UNDER, ALL, 96.0, True),
        (slice(6, 11), slice(3, 5), 0.3 * 96.0, False),
        (slice(11, 14), slice(3, 5), 0.3 * 96.0, False),
        (slice(14, 16), slice(3, 5), 60.0, False),
    )
    tops = rowdetect._peaks(s, 6.0)
    [(y, _)] = rowdetect._clipped_peaks(s, s > 0.0, 6.0, saturated, tops)
    assert UNDER.start <= y < UNDER.stop


def test_clipped_peaks_count_only_the_kept_signal_s_clipped_pixels():
    # Two clipped bands stacked across a valley of 80 (a peak added under
    # the top one), and clipped pixels the detector did not keep (a speck it
    # rejected) in one column of the valley, joining them: they are no
    # band's clipped pixels, and the two bands stay stacked.
    s, saturated = _plateaus(
        (TOP, ALL, 100.0, True), (VALLEY, ALL, 80.0, False), (UNDER, ALL, 99.0, True)
    )
    kept = s > 0.0
    saturated[VALLEY, 5] = True
    kept[VALLEY, 5] = False
    tops = rowdetect._peaks(np.where(kept, s, 0.0), 6.0)
    [(y, _)] = rowdetect._clipped_peaks(s, kept, 6.0, saturated, tops)
    assert UNDER.start <= y < UNDER.stop


def test_clipped_peaks_read_a_saddle_at_the_level_as_below_it():
    # The clipped bands stacked across a valley of 80, and under the lower
    # (99), through a bridge exactly at VALLEY_FRAC of it (0.75 x 99), a band
    # higher than both and not clipped. At the level it reads (0.75 x 99),
    # the lower band's region stops at the bridge: it holds the top band
    # alone, clipped and stacked, and a peak is added. Were the bridge above
    # the level, the region would hold the band not clipped: none.
    s, saturated = _plateaus(
        (TOP, ALL, 100.0, True),
        (VALLEY, ALL, 80.0, False),
        (UNDER, ALL, 99.0, True),
        (slice(11, 13), slice(4, 7), 0.75 * 99.0, False),
        (slice(13, 16), ALL, 101.0, False),
    )
    tops = rowdetect._peaks(s, 6.0)
    assert not any(UNDER.start <= y < UNDER.stop for y, _ in tops)  # not vacuous
    [(y, _)] = rowdetect._clipped_peaks(s, s > 0.0, 6.0, saturated, tops)
    assert UNDER.start <= y < UNDER.stop
