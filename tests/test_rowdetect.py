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
from proteia.core.grow import grow_box
from proteia.core.model import BoxSize, overlaps
from proteia.core.quantify import estimate_background
from proteia.core.rowdetect import (
    AMBIGUITY_MARGIN,
    BG_GUARD,
    BG_GUARD_MIN,
    DETECT_K,
    EMPTY_WINDOW,
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
    NOISE_SIGMA,
    RowCase,
    adversarial,
    adversarial_row,
    band_between,
    bench_cases,
    blob,
    fuzz_row,
    synthetic_row,
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
        ("vstreak_empty", 1000, 4, "unassigned"),  # the streak kept, but in no piece
    ],
)
def test_empty_lane_reasons(key, seed, lane, reason):
    case = _adversarial(key, seed)
    found = detect(case)
    assert_hits_own_lanes(case, found)
    empty = found.lanes[lane]
    assert (empty.rect, empty.reason, empty.components) == (None, reason, 0)
    assert empty.window is not None  # measured, whatever the reason
    if reason == "edge_signal":  # the neighbouring row's rows are left out of the slot
        assert empty.window[1] > case.row[1]
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
    # Its slot is clipped to the row box: only the part inside was measured.
    assert found.lanes[0].window[0] == case.row[0]
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
            before,
            rect=move(before.rect),
            extent=move(before.extent),
            window=move(before.window),
            expected_x=0.0,
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
    assert [lane.window for lane in inverse.lanes] == [lane.window for lane in found.lanes]
    assert components(inverse) == components(found)
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


def test_box_size_is_the_model_type():
    assert isinstance(_detected("all_present").size, BoxSize)
