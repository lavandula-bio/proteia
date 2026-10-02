# SPDX-License-Identifier: Apache-2.0
"""Adversarial rows for row-box detection (#180).

Each recipe makes one known kind of error show: a box cut through a band, a
mark beside the row read as a lane, heavy membrane noise, dust on an empty
lane, a box dragged over two rows, a neighbouring band above or below the
row's own. Every row is scored by :func:`rowcases.score_row` against the
recipe's own truth: the lanes it gets wrong, and those of them the page shows
no flag for (silent). Today's detector has floors written as numbers; the
rows it reads wrong today are expected failures (``xfail(strict=True)``)
citing their issue, so the fix must remove the mark.

The rows of every neighbouring-band family run only with ``PROTEIA_SLOW=1``;
one row of each runs always. The ring-stain and lane-anchor recipes run
through the operations, in ``test_operations``.
"""

import dataclasses
import functools
import math
import os

import numpy as np
import pytest

from proteia.core import evaluate, rowdetect
from proteia.core.quantify import estimate_background
from proteia.core.rowdetect import Peak, RowDetection
from rowcases import (
    ADVERSARIAL,
    BENCH_RECIPES,
    MEMBRANE,
    NOISE_SIGMA,
    STRESS,
    LaneTruth,
    Recipe,
    RowCase,
    RowScore,
    adversarial_row,
    bench_cases,
    bench_row,
    blob,
    lane_truth,
    neighbour_grid_row,
    own_bands,
    ring_stain,
    score_row,
    stress,
)

SLOW = os.environ.get("PROTEIA_SLOW") == "1"
slow = pytest.mark.skipif(not SLOW, reason="the full set runs with PROTEIA_SLOW=1")
SEEDS = (1000, 1001, 1002)


def _noisy(noise: float) -> str:
    return "" if noise == NOISE_SIGMA else f"noise{noise:g}/"


def bench(name: str, seed: int | None = None, *, noise: float = NOISE_SIGMA) -> Recipe:
    """The bench row ``name`` at its own seed (or ``seed``), as a recipe to
    score."""
    _, own_seed, params = BENCH_RECIPES[name]
    label = f"{_noisy(noise)}bench/{name}" + ("" if seed is None else f"/{seed}")
    return Recipe(label, "bench", own_seed if seed is None else seed, params, noise)


def adv(key: str, seed: int, *, noise: float = NOISE_SIGMA) -> Recipe:
    """The :data:`ADVERSARIAL` or :data:`STRESS` recipe ``key`` at ``seed``."""
    params = ADVERSARIAL[key] if key in ADVERSARIAL else STRESS[key]
    return Recipe(f"{_noisy(noise)}{key}/{seed}", "adversarial", seed, params, noise)


def detect(recipe: Recipe, case: RowCase) -> RowDetection:
    return rowdetect.detect_row(
        case.image,
        case.row,
        case.n_lanes,
        background=estimate_background(case.image),
        dark_on_light=case.dark_on_light,
        **recipe.detect,
    )


_SCORED: dict[str, tuple[RowScore, RowDetection, dict[int, LaneTruth]]] = {}


def scored(recipe: Recipe) -> tuple[RowScore, RowDetection, dict[int, LaneTruth]]:
    """The recipe's score, today's detection and its lanes' truth, kept by
    label: the floors and the recipe tests share them (no image is kept)."""
    if recipe.label not in _SCORED:
        case = recipe.build()
        found = detect(recipe, case)
        truth = lane_truth(recipe, case)
        _SCORED[recipe.label] = (score_row(recipe, case, found, truth), found, truth)
    return _SCORED[recipe.label]


def ids(recipes: list[Recipe]) -> list[str]:
    return [recipe.label for recipe in recipes]


# --- The rows the floors score ---

# The bench at its own seeds and every ADVERSARIAL recipe at three seeds.
HONEST = [
    *(bench(name) for name in BENCH_RECIPES),
    *(adv(key, seed) for key in ADVERSARIAL for seed in SEEDS),
]
# The ADVERSARIAL recipes read honestly under heavy noise (the rows whose box
# covers a mark beside the row, or omits an empty end lane, are doubtful or
# refused by design).
NOISY_KEYS = (
    "nbr_above_miss",
    "blotch_empty",
    "vstreak_empty",
    "doublet_two",
    "gauss_tails_miss",
    "twenty_lanes",
    "blob_gap",
    "bubble_band",
    "tall_band",
    "doublet_deep",
    "smile_tall_tight",
    "wide_row",
    "touching_weak_end",
    "wide_next_empty",
    "tilt_cut",
    "dumbbell_band",
    "hollow_band",
    "hollow_ring",
    "notched_band",
)
# The new rows every lane of which today's detector reads right or flags,
# and the bench and the recipes above at noise 3200 and 6400 (a depth of 3
# to 9 noise sigmas, where detection starts to tell good from bad; the faint
# bench band is below the detection limit at 6400).
STRESSED = [
    *(
        adv(key, seed)
        for key in (
            "dust",
            "dust_2px",
            "bottom_cut",
            "tight_sides",
            "two_rows",
            "two_rows_near_empty",
        )
        for seed in SEEDS
    ),
    *(adv("empty_three", seed, noise=noise) for noise in (3200.0, 6400.0) for seed in SEEDS),
    *(bench(name, noise=3200.0) for name in BENCH_RECIPES),
    *(bench(name, noise=6400.0) for name in BENCH_RECIPES if name != "faint_band"),
    *(adv(key, 1000, noise=3200.0) for key in NOISY_KEYS),
]
FLOOR_ROWS = HONEST + STRESSED
# The rows at seed 1000 and the bench rows (at their own seeds).
FLOOR_SUBSET = [recipe for recipe in FLOOR_ROWS if recipe.seed == 1000 or recipe.kind == "bench"]


def totals(recipes: list[Recipe]) -> tuple[int, int, str, str]:
    """(silent lanes, wrong lanes, the rows with a silent lane, the rows with
    a wrong lane) of the rows."""
    scores = [scored(recipe)[0] for recipe in recipes]
    silent = "\n".join(str(score) for score in scores if score.silent)
    wrong = "\n".join(str(score) for score in scores if score.wrong)
    return sum(s.silent for s in scores), sum(s.wrong for s in scores), silent, wrong


def test_the_floor_rows_are_the_measured_set():
    assert (len(HONEST), len(STRESSED), len(FLOOR_SUBSET)) == (90, 72, 96)
    assert len({recipe.label for recipe in FLOOR_ROWS}) == 162


def test_todays_detector_holds_its_floor():
    # Measured on main (f9bb182): 2 silent lanes, both lane 4 of notched_band
    # at seed 1000 (IoU 0.493), at noise 400 and 3200; 144 wrong lanes, every
    # other one flagged or in a refused row; one honest n.d. (the faint bench
    # band at noise 3200, expected SNR 2.8) not counted. A change may lower
    # these numbers (then lower them here), never raise them.
    silent, wrong, silent_rows, wrong_rows = totals(FLOOR_ROWS)
    assert silent <= 2, silent_rows
    assert wrong <= 144, wrong_rows


def test_todays_detector_holds_its_floor_on_the_seed_1000_rows():
    # The rows at seed 1000 and the bench rows: the subset the self-check of
    # degraded detectors compares against (#181).
    silent, wrong, silent_rows, wrong_rows = totals(FLOOR_SUBSET)
    assert silent <= 2, silent_rows
    assert wrong <= 51, wrong_rows


# --- Row boxes cut through bands ---


def unflagged_cuts(found: RowDetection, truth: dict[int, LaneTruth]) -> str:
    """The boxed lanes that lose 15% or more of their band to the row box
    with no cut_by_row_box, in words ("" for none)."""
    return "; ".join(
        f"lane {lane + 1}: {t.cut_share:.1%} of its band outside the row box, boxed"
        " without cut_by_row_box"
        for lane, t in sorted(truth.items())
        if t.cut_share >= 0.15 and found.lanes[lane].rect is not None and not found.lanes[lane].cut
    )


@pytest.mark.parametrize(
    ("key", "cut"), [("side_cut", [0, 5]), ("right_side_cut", [5]), ("tight_sides", [])]
)
@pytest.mark.parametrize("seed", SEEDS)
def test_the_side_cut_recipes_cut_the_end_bands(key, cut, seed):
    # The recipes themselves (no detection): 12 px inside an end band's
    # extent leaves a fifth of it outside the box; 4 px, a twentieth.
    truth = scored(adv(key, seed))[2]
    shares = {lane: t.cut_share for lane, t in truth.items()}
    assert [lane for lane, share in shares.items() if share >= 0.15] == cut
    assert all(0.19 < shares[lane] < 0.25 for lane in cut)
    assert all(share < 0.07 for lane, share in shares.items() if lane not in cut)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#178 a row box whose side edge cuts an end band boxes it without a flag",
)
@pytest.mark.parametrize("key", ["side_cut", "right_side_cut"])
@pytest.mark.parametrize("seed", SEEDS)
def test_a_band_the_box_side_cuts_is_flagged(key, seed):
    _, found, truth = scored(adv(key, seed))
    assert not unflagged_cuts(found, truth), unflagged_cuts(found, truth)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_box_a_little_inside_the_end_bands_reads_them_right(seed):
    score, found, _ = scored(adv("tight_sides", seed))
    assert found.flags == ()
    assert (score.hits, score.lanes) == (6, ()), str(score)


@pytest.mark.parametrize(
    ("key", "seed"),
    [
        *(("bottom_cut", seed) for seed in range(1000, 1005)),
        *(("tilt_cut", seed) for seed in range(1001, 1005)),
    ],
)
def test_bands_the_box_top_or_bottom_edge_cuts_are_flagged(key, seed):
    # The bottom edge 10 px up cuts 2 to 6 bands of each row by 15% or more,
    # the tilted row's top edge lane 1's band.
    _, found, truth = scored(adv(key, seed))
    assert any(t.cut_share >= 0.15 for t in truth.values())
    assert not unflagged_cuts(found, truth), unflagged_cuts(found, truth)


# --- Heavy membrane noise ---


@pytest.mark.parametrize("noise", [3200.0, 6400.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_empty_lanes_under_heavy_noise_get_no_box(noise, seed):
    score, found, _ = scored(adv("empty_three", seed, noise=noise))
    assert [found.slots[lane] for lane in (1, 3, 4)] == [None] * 3
    assert (score.hits, score.lanes) == (3, ()), str(score)


@pytest.mark.parametrize("seed", [1000, 1017])
def test_a_label_beside_a_noisy_row_is_bands_a_lane_apart(seed):
    # The recipe: lane 0 empty, the label's strokes past the last lane, every
    # band well above the noise (expected SNR over 20).
    truth = scored(adv("label_beside", seed, noise=3200.0))[2]
    assert sorted(truth) == [1, 2, 3, 4, 5]
    assert min(t.snr for t in truth.values()) > 20.0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#111 a label beside a very noisy row is read as a lane and every band a lane"
    " early, without a flag",
)
@pytest.mark.parametrize("seed", [1000, 1017])
def test_a_label_beside_a_noisy_row_makes_its_lanes_doubtful(seed):
    score, found, _ = scored(adv("label_beside", seed, noise=3200.0))
    assert found.refused or "doubtful_lanes" in found.flags, str(score)


@pytest.mark.parametrize("seed", [1000, 1008])
def test_a_noisy_tight_row_holds_bands_well_above_the_noise(seed):
    # The recipe: the bench's tight_box at noise 6400; every band's expected
    # SNR is over 10, against a limit of 6.
    truth = scored(bench("tight_box", seed, noise=6400.0))[2]
    assert min(t.snr for t in truth.values()) > 10.0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#179 very noisy tightly cropped rows lose bands to an inflated noise estimate",
)
@pytest.mark.parametrize("seed", [1000, 1008])
def test_a_noisy_tight_row_loses_no_band_silently(seed):
    score, found, _ = scored(bench("tight_box", seed, noise=6400.0))
    lost = [lane for lane in score.lanes if lane.kind == "miss" and lane.status == "silent"]
    assert not lost, (
        "; ".join(
            f"lane {lane.lane + 1}: no box, no flag, expected SNR {lane.value:.1f}" for lane in lost
        )
        + f" (the row's smoothed noise read as {found.noise:.0f})"
    )


# --- Lines, dust and a box over two rows ---


@pytest.mark.parametrize("seed", SEEDS)
def test_the_frame_lines_are_no_band(seed):
    score, found, _ = scored(adv("frame", seed))
    assert found.lanes[2].reason == "line"
    assert (score.refused, score.hits, score.lanes, found.flags) == (None, 5, (), ()), str(score)


@pytest.mark.parametrize("key", ["dust", "dust_2px"])
@pytest.mark.parametrize("seed", SEEDS)
def test_dust_on_an_empty_lane_gets_no_box(key, seed):
    score, found, _ = scored(adv(key, seed))
    assert found.slots[3] is None
    assert (score.hits, score.lanes) == (5, ()), str(score)


@pytest.mark.parametrize("key", ["two_rows", "two_rows_near_empty"])
@pytest.mark.parametrize("seed", SEEDS)
def test_a_box_over_two_rows_is_refused(key, seed):
    score = scored(adv(key, seed))[0]
    assert score.refused == "off_row_line"
    assert score.silent == 0 and score.wrong == score.n_ref


# --- The neighbouring-band grid ---


def _grid_cells() -> list[tuple[str, str, dict]]:
    """``(family, cell, params)`` of every cell of the neighbouring-band grid
    that today's detector must read right or flag, as a user drags the row
    box snugly (:func:`neighbour_grid_row`)."""
    cells: list[tuple[str, str, dict]] = []
    # Weak bands (8000-12000 deep), a band 1.5 to 4 times as deep 10 to 24 px
    # above or below.
    for dy in (-24.0, -18.0, -14.0, -10.0, 10.0, 14.0, 18.0, 24.0):
        for rel in (1.5, 2.0, 3.0, 4.0):
            params = {"depth_range": (8000.0, 12000.0), "neighbour_dy": dy, "neighbour_rel": rel}
            cells.append(("strong_neighbour", f"dy{dy:+g}_x{rel:g}", params))
    # Bands 4000 to 16000 deep across the row, a 32000-deep band beside each.
    for dy in (-24.0, -18.0, -14.0, 14.0, 18.0, 24.0):
        params = {
            "depths": [4000.0, 6000.0, 8000.0, 10000.0, 12000.0, 16000.0],
            "neighbour_dy": dy,
            "neighbour_depths": [32000.0] * 6,
        }
        cells.append(("weak_bands", f"dy{dy:+g}", params))
    # Bands 8000 to 48000 deep, rising or falling across the row, a
    # 24000-deep band 10 to 18 px above or below each.
    graded = [8000.0, 12000.0, 18000.0, 26000.0, 36000.0, 48000.0]
    for dy in (-18.0, -14.0, -12.0, -10.0, 10.0, 12.0, 14.0, 18.0):
        for order, depths in (("rising", graded), ("falling", graded[::-1])):
            params = {"depths": depths, "neighbour_dy": dy, "neighbour_depths": [24000.0] * 6}
            cells.append(("graded_bands", f"dy{dy:+g}_{order}", params))
    # Over-exposed bands (1.5 times the membrane) burnt out in their centres
    # in lanes 1 and 4: alone, with a smear down from lane 1's band, with a
    # weak band 16 px below every band, or with a dark speck beside lane 1's.
    for light in (60000.0, 90000.0):
        burnt = [blob(lane, 6.0, -light, ry=12.0) for lane in (1, 4)]
        base = {"depths": [1.5 * MEMBRANE] * 6, "artefacts": burnt, "img_h": 220}
        cells.append(("burnt_out", f"light{light:g}", base))
        cells.append(
            ("burnt_out", f"light{light:g}_smear", {**base, "smears": [(1, 15000.0, 12.0)]})
        )
        weak = {"neighbour_dy": 16.0, "neighbour_depths": [0.4 * 1.5 * MEMBRANE] * 6}
        cells.append(("burnt_out", f"light{light:g}_weak_neighbour", {**base, **weak}))
        for r in (3.0, 4.0):
            for depth in (40000.0, 60000.0):
                for dx, dy in ((-18.0, -10.0), (-12.0, 9.0), (12.0, -9.0), (18.0, 10.0)):
                    speck = blob(1, r, depth, dx=dx, dy=dy)
                    cell = f"light{light:g}_speck{r:g}_{depth:g}_{dx:+g}{dy:+g}"
                    cells.append(("burnt_out", cell, {**base, "artefacts": [*burnt, speck]}))
    # Over-exposed bands (1.3 to 3 times the membrane), a band as deep or up
    # to twice as deep 12 to 20 px above or below each.
    for over in (1.3, 2.0, 3.0):
        for rel in (1.0, 1.5, 2.0):
            for dy in (-20.0, -16.0, -12.0, 12.0, 16.0, 20.0):
                params = {"depths": [over * MEMBRANE] * 6, "neighbour_dy": dy, "neighbour_rel": rel}
                cells.append(("saturated_neighbour", f"x{over:g}_x{rel:g}_dy{dy:+g}", params))
    # A pale line across every band's middle (30 to 70% of its depth), bands
    # 12 or 18 px high, over-exposed or not.
    for h in (12.0, 18.0):
        for frac in (0.3, 0.5, 0.7):
            line = {"h": h, "slit": (frac, 1.2)}
            cells.append(("slit", f"h{h:g}_{frac:g}", {**line, "depth_range": (20000.0, 28000.0)}))
            cells.append(
                ("slit", f"h{h:g}_{frac:g}_saturated", {**line, "depths": [1.5 * MEMBRANE] * 6})
            )
    # A tilted or smiling row, a band half or twice as deep 14 px above or
    # below each.
    for tilt, smile in (
        (-12.0, 0.0),
        (12.0, 0.0),
        (0.0, 4.0),
        (0.0, 6.0),
        (12.0, 6.0),
        (-12.0, 4.0),
    ):
        for rel, depth_range in ((0.5, (18000.0, 30000.0)), (2.0, (8000.0, 12000.0))):
            for dy in (-14.0, 14.0):
                params = {
                    "tilt": tilt,
                    "smile": smile,
                    "neighbour_dy": dy,
                    "neighbour_rel": rel,
                    "depth_range": depth_range,
                }
                cells.append(("tilt_smile", f"t{tilt:+g}_s{smile:g}_x{rel:g}_dy{dy:+g}", params))
    # No band beside: plain bands, lane 2 a tenth as deep, lane 2 empty,
    # doublets 8 px apart (the lower 0.6 or 1.6 times as deep).
    cells += [
        ("plain", "plain", {}),
        ("plain", "faint_lane", {"faint": 2}),
        ("plain", "empty_lane", {"missing": [2]}),
        ("plain", "doublet_x0.6", {"doublet": (8.0, 0.6)}),
        ("plain", "doublet_x1.6", {"doublet": (8.0, 1.6)}),
    ]
    return cells


def grid(family: str, cell: str, params: dict) -> Recipe:
    """A grid cell at seed 1000, read as the operation reads a 16-bit image
    (pixels at 0 saturated)."""
    return Recipe(f"{family}/{cell}", "grid", 1000, params, detect={"saturated_at": 0.0})


GRID = [grid(*cell) for cell in _grid_cells()]
# One row of each family always runs, with today's wrong lanes (all flagged):
# the row with the most wrong lanes, else the one whose right lanes come
# closest to partial (a box holding 0.92 of the band, the burnt-out family's).
GRID_ALWAYS = {
    "strong_neighbour/dy-10_x2": 6,
    "weak_bands/dy-14": 0,
    "graded_bands/dy+10_falling": 0,
    "burnt_out/light90000_speck4_60000_-18-10": 0,
    "saturated_neighbour/x3_x2_dy+12": 6,
    "slit/h12_0.3": 0,
    "tilt_smile/t-12_s0_x2_dy-14": 6,
    "plain/doublet_x0.6": 0,
}


def test_the_grid_is_the_measured_set():
    families = {}
    for recipe in GRID:
        family = recipe.label.split("/")[0]
        families[family] = families.get(family, 0) + 1
    assert families == {
        "strong_neighbour": 32,
        "weak_bands": 6,
        "graded_bands": 16,
        "burnt_out": 38,
        "saturated_neighbour": 54,
        "slit": 12,
        "tilt_smile": 24,
        "plain": 5,
    }
    labels = {recipe.label for recipe in GRID}
    assert len(labels) == 187 and set(GRID_ALWAYS) <= labels


@pytest.mark.parametrize(
    "recipe", [r for r in GRID if r.label in GRID_ALWAYS], ids=lambda r: r.label
)
def test_a_neighbouring_band_is_never_boxed_silently(recipe):
    score = scored(recipe)[0]
    assert score.silent == 0, str(score)
    assert score.wrong <= GRID_ALWAYS[recipe.label], str(score)


@slow
def test_no_grid_row_is_read_wrong_silently():
    # Measured on main (f9bb182): 87 wrong lanes of 1122, every one flagged.
    scores = [scored(recipe)[0] for recipe in GRID]
    assert sum(s.silent for s in scores) == 0, "\n".join(str(s) for s in scores if s.silent)
    assert sum(s.wrong for s in scores) <= 87, "\n".join(str(s) for s in scores if s.wrong)


# --- The score itself: every exit and every flag source ---

BASE = bench("all_present")
WINDOW = (0, 40, 30, 80)  # an empty lane's slot (its place plays no part)
EMPTY = {"rect": None, "reason": "no_band", "window": WINDOW, "components": 0, "peaks": ()}


@functools.cache
def _base_case() -> RowCase:
    return BASE.build()


def base() -> tuple[RowCase, RowDetection, dict[int, LaneTruth]]:
    """The bench's all_present row, today's detection of it (every lane right,
    no flag) and its truth."""
    score, found, truth = scored(BASE)
    assert (score.hits, score.lanes, found.flags) == (6, (), ())
    return _base_case(), found, truth


def changed(found: RowDetection, lanes: dict[int, dict], **fields) -> RowDetection:
    """``found`` with the given lanes' fields changed, and its own ``fields``."""
    new = list(found.lanes)
    for lane, changes in lanes.items():
        new[lane] = dataclasses.replace(new[lane], **changes)
    return dataclasses.replace(found, lanes=tuple(new), **fields)


def score_of(found: RowDetection, *, reference=None, truth=None) -> RowScore:
    """The score of ``found`` on the base row, its reference and truth changed
    if given."""
    case, _, base_truth = base()
    if reference is not None:
        case = dataclasses.replace(case, reference=reference)
    return score_row(BASE, case, found, base_truth if truth is None else truth)


def kinds(score: RowScore) -> list[tuple[int, str, str, str | None]]:
    return [(lane.lane, lane.kind, lane.status, lane.shown_by) for lane in score.lanes]


def test_a_row_read_right_has_no_wrong_lane():
    _, found, _ = base()
    score = score_of(found)
    assert (score.refused, score.n_ref, score.hits, score.lanes) == (None, 6, 6, ())
    assert (score.wrong, score.silent) == (0, 0)


def test_a_band_without_a_box_and_no_flag_is_a_silent_miss():
    _, found, truth = base()
    score = score_of(changed(found, {2: EMPTY}))
    assert kinds(score) == [(2, "miss", "silent", None)]
    assert score.lanes[0].value == truth[2].snr > 100
    assert (score.hits, score.wrong, score.silent) == (5, 1, 1)
    assert str(score) == f"bench/all_present: 5/6 right; lane 3 miss {truth[2].snr:.3f} silent"


def test_a_warning_about_the_row_only_is_no_flag_on_the_lane():
    _, found, _ = base()
    score = score_of(changed(found, {2: EMPTY}, flags=("background_mismatch",)))
    assert kinds(score) == [(2, "miss", "row_warned", None)]
    assert (score.wrong, score.silent) == (1, 0)


@pytest.mark.parametrize(
    ("lanes", "shown_by"),
    [
        ({}, "doubtful_lanes"),
        ({4: {**EMPTY, "reason": "unassigned"}}, "unassigned piece"),
    ],
)
def test_a_doubt_about_the_row_flags_every_wrong_lane(lanes, shown_by):
    _, found, _ = base()
    flags = ("doubtful_lanes",) if shown_by == "doubtful_lanes" else ()
    score = score_of(changed(found, {2: EMPTY, **lanes}, flags=flags))
    wrong = [2, *lanes]
    assert kinds(score) == [(lane, "miss", "flagged", shown_by) for lane in sorted(wrong)]


@pytest.mark.parametrize("reason", ["artefact", "line", "edge_signal", "side_signal"])
def test_an_empty_lane_not_measured_is_flagged(reason):
    _, found, _ = base()
    score = score_of(changed(found, {2: {**EMPTY, "reason": reason}}))
    assert kinds(score) == [(2, "miss", "flagged", f"not measured: {reason}")]


def test_an_empty_lane_outside_the_row_box_is_flagged():
    _, found, _ = base()
    score = score_of(changed(found, {2: {**EMPTY, "window": None}}))
    assert kinds(score) == [(2, "miss", "flagged", "not measured: outside the row box")]


def test_misses_beside_fewer_than_two_boxes_are_flagged():
    # With one box the lanes are not placed (no n.d. is recorded); with two
    # they are, and the misses are silent.
    _, found, _ = base()
    one = score_of(changed(found, dict.fromkeys(range(1, 6), EMPTY)))
    shown = "not recorded: fewer than two bands"
    assert kinds(one) == [(lane, "miss", "flagged", shown) for lane in range(1, 6)]
    two = score_of(changed(found, dict.fromkeys(range(2, 6), EMPTY)))
    assert kinds(two) == [(lane, "miss", "silent", None) for lane in range(2, 6)]


def test_a_silent_miss_below_the_detection_limit_is_an_honest_nd():
    _, found, truth = base()
    missing = changed(found, {2: EMPTY})

    def at(snr: float) -> dict[int, LaneTruth]:
        return {**truth, 2: dataclasses.replace(truth[2], snr=snr)}

    below = score_of(missing, truth=at(5.99))
    assert kinds(below) == [(2, "miss", "limit", None)]
    assert (below.hits, below.wrong, below.silent) == (5, 0, 0)
    assert kinds(score_of(missing, truth=at(6.0))) == [(2, "miss", "silent", None)]
    # Only a silent miss: one with a warning on the row stays wrong.
    warned = score_of(changed(missing, {}, flags=("background_mismatch",)), truth=at(5.99))
    assert kinds(warned) == [(2, "miss", "row_warned", None)]
    assert warned.wrong == 1


@pytest.mark.parametrize(
    ("change", "shown_by"),
    [
        ({}, None),
        ({"components": 2}, "multiple_components"),
        ({"hollow": True}, "hollow_band"),
        ({"cut": True}, "cut_by_row_box"),
    ],
)
def test_a_box_in_a_lane_without_a_band_is_a_false_positive(change, shown_by):
    _, found, truth = base()
    reference = {lane: rect for lane, rect in base()[0].reference.items() if lane != 4}
    score = score_of(changed(found, {4: change}), reference=reference, truth=truth)
    assert kinds(score) == [(4, "fp", "silent" if shown_by is None else "flagged", shown_by)]
    assert (score.n_ref, score.hits) == (5, 5)


@pytest.mark.parametrize(
    ("note", "status"),
    [
        (
            "lane 5: extent above 2x the median of the other extents, left out of the shared size",
            "flagged",
        ),
        (
            "lanes 3, 5: extents above 2x the median of the others, left out of the shared size",
            "flagged",
        ),
        (
            "lane 3: extent above 2x the median of the other extents, left out of the shared size",
            "row_warned",
        ),
        ("lane 5: a second separate component reaches 0.4 of the peak", "row_warned"),
    ],
)
def test_a_box_the_size_note_names_is_flagged(note, status):
    # The note counts lanes from 1: lane 5 is index 4.
    _, found, truth = base()
    reference = {lane: rect for lane, rect in base()[0].reference.items() if lane != 4}
    flagged = changed(found, {}, flags=("size_outlier",), notes=(note,))
    score = score_of(flagged, reference=reference, truth=truth)
    shown_by = "size_outlier" if status == "flagged" else None
    assert kinds(score) == [(4, "fp", status, shown_by)]


@pytest.mark.parametrize(
    ("where", "other", "status"),
    [
        ("top", True, "flagged"),
        ("bottom", True, "flagged"),
        ("below", True, "silent"),
        ("top", False, "silent"),
    ],
)
def test_a_box_whose_lane_holds_another_band_in_the_row_is_flagged(where, other, status):
    # The results count a lane's bands in the row box's rows (both edges in)
    # and notice a count above the one expected.
    case, found, truth = base()
    y0, y1 = case.row[1], case.row[3]
    y = {"top": y0 + 0.5, "bottom": y1 - 0.5, "below": y1 + 0.5}[where]
    x = case.lane_cx[4]
    lane = {"peaks": (Peak(y, x, 30.0, False, other),)}
    reference = {k: rect for k, rect in case.reference.items() if k != 4}
    score = score_of(changed(found, {4: lane}), reference=reference, truth=truth)
    assert kinds(score) == [(4, "fp", status, "band count" if status == "flagged" else None)]


def test_a_noisy_doublet_is_flagged_by_its_band_count_alone():
    # Lane 2's doublet at noise 1600: the box takes the upper band only (IoU
    # 0.41 with the pair) and no flag is raised, but the lane's peaks count
    # two bands in the row box's rows, which the results notice.
    score, found, _ = scored(adv("doublet_deep", 1001, noise=1600.0))
    assert found.flags == ()
    assert kinds(score) == [(2, "wrong_box", "flagged", "band count")]


@pytest.mark.parametrize(
    ("rect", "kind"),
    [
        ((10, 0, 40, 10), None),  # IoU exactly 0.5: a hit
        ((11, 0, 41, 10), "wrong_box"),  # IoU 0.463
        ((100, 0, 130, 10), "wrong_box"),  # beside the band: IoU 0
    ],
)
def test_a_box_off_its_band_is_a_wrong_box(rect, kind):
    _, found, _ = base()
    others = dict.fromkeys([0, 2, 3, 4, 5], EMPTY)
    two = changed(found, {**others, 0: {}, 1: {"rect": rect, "peaks": ()}})
    reference = {1: (0, 0, 30, 10)}
    truth = {1: LaneTruth(cut_share=0.0, snr=50.0)}
    score = score_of(two, reference=reference, truth=truth)
    # Lane 0's box has no band in this reference: a false positive either way.
    assert [(s.lane, s.kind) for s in score.lanes if s.lane == 1] == (
        [] if kind is None else [(1, kind)]
    )
    if kind is not None:
        [lane] = [s for s in score.lanes if s.lane == 1]
        assert lane.value == pytest.approx({11: 190 / 410, 100: 0.0}[rect[0]])


@pytest.mark.parametrize(
    ("share", "cut", "expected"),
    [
        (0.1499, False, []),
        (0.15, False, [(3, "cut", "silent", None)]),
        (0.15, True, [(3, "cut", "flagged", "cut_by_row_box")]),
    ],
)
def test_a_box_on_a_band_the_row_box_cuts_is_wrong(share, cut, expected):
    _, found, truth = base()
    cut_truth = {**truth, 3: dataclasses.replace(truth[3], cut_share=share)}
    score = score_of(changed(found, {3: {"cut": cut}}), truth=cut_truth)
    assert kinds(score) == expected
    assert [lane.value for lane in score.lanes] == [share] * len(expected)


def test_a_refused_row_counts_every_band_wrong_and_none_silent():
    # Its lanes are not read: lane 4's box, on no band, is no false positive.
    case, found, truth = base()
    reference = {lane: rect for lane, rect in case.reference.items() if lane != 4}
    refused = changed(found, {}, flags=("off_row_line", "background_mismatch"))
    score = score_of(refused, reference=reference, truth=truth)
    assert (score.refused, score.n_ref, score.hits) == ("off_row_line", 5, 0)
    assert kinds(score) == [(lane, "miss", "refused", "off_row_line") for lane in (0, 1, 2, 3, 5)]
    assert (score.wrong, score.silent) == (5, 0)
    assert str(score).startswith("bench/all_present: refused (off_row_line), 0/5 right; lane 1")


def test_a_row_without_any_band_found_is_refused():
    _, found, _ = base()
    nothing = changed(found, dict.fromkeys(range(6), EMPTY), size=None)
    score = score_of(nothing)
    assert (score.refused, score.hits, score.wrong, score.silent) == ("no_band_found", 0, 6, 0)


def test_a_row_without_reference_bands_counts_its_boxes_only():
    _, found, _ = base()
    score = score_of(found, reference={}, truth={})
    assert (score.n_ref, score.hits) == (0, 0)
    assert kinds(score) == [(lane, "fp", "silent", None) for lane in range(6)]


def test_a_row_whose_lanes_disagree_is_not_scored():
    case, found, truth = base()
    with pytest.raises(ValueError, match="5 lanes declared, 6 drawn, 6 read"):
        score_row(BASE, dataclasses.replace(case, n_lanes=5), found, truth)
    with pytest.raises(ValueError, match="6 lanes declared, 6 drawn, 5 read"):
        score_row(BASE, case, dataclasses.replace(found, lanes=found.lanes[:5]), truth)


@pytest.mark.parametrize(
    ("recipe", "hits", "fp"),
    [
        (adv("doublet_two", 1001), 6, ()),
        (adv("ladder_beside", 1001), 0, (0,)),
        (adv("box_omits_empty_first", 1000), 0, ()),
        (bench("touching"), 6, ()),
    ],
    ids=lambda value: value.label if isinstance(value, Recipe) else None,
)
def test_the_score_counts_hits_as_the_benchmark_does(recipe, hits, fp):
    # Rows with no cut band: the score's hits and false positives are the
    # benchmark's (evaluate.hit_rate); a refused row proposes nothing.
    score, found, _ = scored(recipe)
    slots = (None,) * len(found.lanes) if found.refused else found.slots
    measured = evaluate.hit_rate(slots, recipe.build().reference)
    assert (score.hits, tuple(s.lane for s in score.lanes if s.kind == "fp")) == (hits, fp)
    assert (measured.hits, measured.false_positive_lanes) == (hits, fp)


# --- Grid rows: which centres a box holds ---

GRID_ROW = Recipe(
    "a band 18 px below",
    "grid",
    1000,
    {"neighbour_dy": 18.0, "missing": [4, 5]},
    detect={"saturated_at": 0.0},
)


def centred(case: RowCase, lane: int, *, y: float | None = None, dx: int = 0, dy: int = 0):
    """A 44 x 12 box centred on the lane's band (or on row ``y``), moved."""
    x0 = math.floor(case.lane_cx[lane] + 0.5 - 22)
    y0 = math.floor((case.lane_cy[lane] if y is None else y) + 0.5 - 6)
    return (x0 + dx, y0 + dy, x0 + dx + 44, y0 + dy + 12)


def grid_score(rects: dict[int, tuple | None]) -> RowScore:
    """The grid row's score with the given lanes' boxes in place of today's."""
    case = GRID_ROW.build()
    _, found, truth = scored(GRID_ROW)
    lanes = {
        lane: {"rect": rect, "reason": "band", "peaks": (), "cut": False, "components": 1}
        if rect is not None
        else EMPTY
        for lane, rect in rects.items()
    }
    return score_row(GRID_ROW, case, changed(found, lanes, flags=()), truth)


def test_grid_boxes_are_scored_by_the_centres_they_hold():
    case = GRID_ROW.build()
    neighbour = case.neighbour_cy
    assert neighbour == pytest.approx([cy + 18.0 for cy in case.lane_cy])
    both = centred(case, 3)[:3] + (centred(case, 3, y=neighbour[3])[3],)
    score = grid_score(
        {
            0: centred(case, 0),
            1: centred(case, 1, dy=3),  # still 0.88 of the band
            2: centred(case, 2, y=neighbour[2]),
            3: both,
            4: centred(case, 4),  # an empty lane
            5: None,  # an empty lane left empty
        }
    )
    assert [(s.lane, s.kind, s.status) for s in score.lanes] == [
        (2, "neighbour", "silent"),
        (3, "both", "silent"),
        (4, "false_box", "silent"),
    ]
    assert (score.n_ref, score.hits) == (4, 2)


@pytest.mark.parametrize("change", [{"dx": 11}, {"dy": 3}])
def test_a_grid_box_holding_most_of_its_band_is_right(change):
    # 11 px along the row, the box holds 0.851 of what the centred box does
    # (1 px further, 0.825: below); 3 px down, 0.876 (4 px, 0.784).
    case = GRID_ROW.build()
    assert grid_score({1: centred(case, 1, **change)}).lanes == ()


@pytest.mark.parametrize(
    ("change", "kind", "value"),
    [
        ({"dx": 12}, "partial", 0.8253),  # on the band's centre, a little short
        ({"dy": 4}, "partial", 0.7836),
        ({"dx": 27}, "partial", 0.4147),  # past the band's centre
        ({"wide": True}, "partial", 0.7821),  # lane 2's band in it counts for nothing
    ],
)
def test_a_grid_box_holding_too_little_of_its_band_is_partial(change, kind, value):
    # The capture compares the box with the box of its size centred on the
    # band, both within the lane's column (one pitch wide).
    case = GRID_ROW.build()
    if "wide" in change:
        x0 = math.floor(case.lane_cx[1] + 0.5) - 10  # 120 px, over lane 2's band
        rect = (x0, centred(case, 1)[1], x0 + 120, centred(case, 1)[3])
    else:
        rect = centred(case, 1, **change)
    [lane] = grid_score({1: rect}).lanes
    assert (lane.lane, lane.kind) == (1, kind)
    assert lane.value == pytest.approx(value, abs=1e-4)


def test_a_grid_band_missed_is_scored_by_its_expected_snr():
    [lane] = grid_score({1: None}).lanes
    assert (lane.lane, lane.kind, lane.status) == (1, "missed", "silent")
    assert lane.value == scored(GRID_ROW)[2][1].snr > 100.0


def test_a_grid_box_off_its_band_centre_is_partial_however_much_it_holds():
    # A doublet whose lower part is a tenth as deep, 12 px below: a box on
    # the upper part holds more of the band than one on the band's centre
    # between them, but not the centre.
    recipe = Recipe("weak doublet", "grid", 1000, {"doublet": (12.0, 0.1)})
    case = recipe.build()
    _, found, truth = scored(recipe)
    rect = centred(case, 1, y=case.lane_cy[1] - 6.0)
    rect = (rect[0], rect[1] + 2, rect[2], rect[3] - 2)  # 8 px high, clear of the centre
    lanes = {1: {"rect": rect, "peaks": (), "cut": False, "components": 1}}
    score = score_row(recipe, case, changed(found, lanes, flags=()), truth)
    [lane] = [lane for lane in score.lanes if lane.lane == 1]
    assert (lane.lane, lane.kind) == (1, "partial")
    assert lane.value > 1.0


@pytest.mark.parametrize(("below", "kind"), [(0.0, "both"), (0.5, None)])
def test_a_centre_on_a_grid_box_edge_is_held(below, kind):
    # Box edges hold a centre that lies on them: the neighbouring band's
    # centre on the box's bottom edge makes the box hold both.
    case = GRID_ROW.build()
    _, found, truth = scored(GRID_ROW)
    rect = centred(case, 2)
    neighbour = list(case.neighbour_cy)
    neighbour[2] = rect[3] - 0.5 + below  # its pixel centre (+ 0.5) on the edge, or past it
    moved = dataclasses.replace(case, neighbour_cy=tuple(neighbour))
    lanes = {2: {"rect": rect, "peaks": (), "cut": False, "components": 1}}
    score = score_row(GRID_ROW, moved, changed(found, lanes, flags=()), truth)
    assert [s.kind for s in score.lanes if s.lane == 2] == ([] if kind is None else [kind])


# --- What the score is measured against ---


def hand_made_row(*, mirrored: bool = False) -> tuple[RowCase, np.ndarray]:
    """A hand-made row and its bands: two lanes 20 px apart, centred at x 15
    and 35; a band of 1000 over 10 x 10 px in the first, one of 5 over 2 x 20
    px (x 30 to 32) in the second; the row box over the top 12 of 20 rows.
    Mirrored, the lanes are numbered right to left (lane 0 at x 35)."""
    own = np.zeros((20, 40))
    own[5:15, 10:20] = 1000.0
    own[0:20, 30:32] = 5.0  # the second band: outside the first lane's column
    left, right = (10, 5, 20, 15), (30, 0, 32, 20)
    case = dataclasses.replace(
        _base_case(),
        image=np.zeros((20, 40)),
        row=(0, 0, 40, 12),
        n_lanes=2,
        reference={0: right, 1: left} if mirrored else {0: left, 1: right},
        lane_cx=(35.0, 15.0) if mirrored else (15.0, 35.0),
        lane_cy=(10.0, 10.0),
    )
    return case, own


def test_the_truth_is_read_from_the_bands_alone():
    # A hand-made band of 1000 over 10 x 10 px in lane 0's column (lanes 20
    # px apart), the row box over its top 7 rows: 30% cut off; its smoothed
    # peak 1000 over noise 400 / sqrt(15).
    case, own = hand_made_row()
    truth = lane_truth(Recipe("hand-made", "adversarial", 0, noise=400.0), case, own)
    assert truth[0].cut_share == pytest.approx(0.3)
    assert truth[0].snr == pytest.approx(1000.0 * math.sqrt(15) / 400.0)
    assert truth[1].cut_share == pytest.approx(0.4)
    # The expected SNR reads the smoothed signal: a band 2 px wide keeps
    # 2/5 of its depth over the 5 columns.
    assert truth[1].snr == pytest.approx(5.0 * 2 / 5 * math.sqrt(15) / 400.0)
    quiet = lane_truth(Recipe("hand-made", "adversarial", 0, noise=0.0), case, own)
    assert quiet[0].snr == math.inf
    # One lane: its column is the whole row, lane 1's band included.
    single = dataclasses.replace(case, n_lanes=1, reference={0: (10, 5, 20, 15)}, lane_cx=(15.0,))
    alone = lane_truth(Recipe("hand-made", "adversarial", 0), single, own)
    assert alone[0].cut_share == pytest.approx(1.0 - (70000 + 120) / (100000 + 200))


def test_the_truth_reads_a_row_whose_lanes_run_right_to_left():
    # The same row numbered right to left: the centres step by -20 px, and
    # each lane's column is still one pitch wide about its centre.
    case, own = hand_made_row(mirrored=True)
    truth = lane_truth(Recipe("hand-made", "adversarial", 0, noise=400.0), case, own)
    assert truth[1].cut_share == pytest.approx(0.3)
    assert truth[0].cut_share == pytest.approx(0.4)
    assert truth[1].snr == pytest.approx(1000.0 * math.sqrt(15) / 400.0)


def test_the_own_bands_leave_out_noise_artefacts_and_the_row_beside():
    plain = Recipe("plain", "adversarial", 1000)
    busy = Recipe(
        "busy",
        "adversarial",
        1000,
        {"artefacts": [blob(2, 6.0, 9000.0)], "neighbour_dy": -22.0},
        noise=3200.0,
    )
    assert np.array_equal(own_bands(busy), own_bands(plain))
    plain_grid = Recipe("plain", "grid", 1000)
    busy_grid = Recipe(
        "busy",
        "grid",
        1000,
        {
            "artefacts": [blob(2, 6.0, 9000.0)],
            "smears": [(1, 15000.0, 12.0)],
            "neighbour_dy": 16.0,
        },
        noise=3200.0,
    )
    assert np.array_equal(own_bands(busy_grid), own_bands(plain_grid))
    # A band's depth (its centre drawn off the pixel grid by up to half a
    # pixel), and none of it in a lane without a band.
    deep = Recipe("deep", "adversarial", 1000, {"depths": {2: 12345.0}, "missing": [4]})
    case, own = deep.build(), own_bands(deep)
    x0, y0, x1, y1 = case.reference[2]
    assert 0.98 * 12345.0 < own[y0:y1, x0:x1].max() <= 12345.0
    x = round(case.lane_cx[4])
    assert own[:, x - 10 : x + 10].max() == 0.0
    # A light-on-dark row's bands count positive too.
    light = bench("light_on_dark")
    assert own_bands(light).min() >= 0.0 and own_bands(light).max() > 18000.0


# --- The recipes ---


def test_the_bench_table_draws_the_bench():
    cases = bench_cases()
    assert [case.name for case in cases] == list(BENCH_RECIPES)
    assert sum(len(case.reference) for case in cases) == 91
    for case in cases:
        again = bench_row(case.name)
        assert np.array_equal(again.image, case.image) and again.row == case.row
    other = bench_row("all_present", 1000)
    assert not np.array_equal(other.image, cases[0].image)


@pytest.mark.parametrize(
    "make",
    [
        lambda noise: bench_row("all_present", 1000, noise=noise),
        lambda noise: adversarial_row("blob gap", 1001, noise=noise, **ADVERSARIAL["blob_gap"]),
        lambda noise: neighbour_grid_row("grid", 1000, noise=noise, neighbour_dy=18.0),
    ],
    ids=["bench", "adversarial", "grid"],
)
def test_more_noise_scales_the_same_draws(make):
    # Away from clipping, the noise at 800 is twice the noise at 400, pixel
    # for pixel (each image rounded): the bands, box and pattern stay.
    quiet, once, twice = (make(noise) for noise in (0.0, 400.0, 800.0))
    assert quiet.row == once.row == twice.row
    assert np.max(np.abs((twice.image - quiet.image) - 2 * (once.image - quiet.image))) <= 2.0
    assert np.std(once.image - quiet.image) == pytest.approx(400.0, rel=0.05)


@pytest.mark.parametrize("key", STRESS)
def test_every_stress_recipe_draws_a_row(key):
    case = stress(key, 1000)
    x0, y0, x1, y1 = case.row
    assert 0 <= x0 < x1 <= case.image.shape[1] and 0 <= y0 < y1 <= case.image.shape[0]
    assert len(case.lane_cx) == case.n_lanes and case.reference


@pytest.mark.parametrize(("side", "rows"), [("top", (40, 68)), ("bottom", (93, 121))])
def test_a_ring_stain_is_flat_on_one_side_of_the_band(side, rows):
    # On a row without bands, noise or jitter (lane 2's centre at x 232,
    # y 80): 1000 deep from 13 to 40 px from it on that side, both ends in,
    # and 40 px either side of it along the row.
    case = adversarial_row(
        "stain",
        1000,
        missing=range(6),
        noise=0.0,
        x_jitter=0.0,
        y_jitter=0.0,
        artefacts=[ring_stain(2, side, 1000.0)],
    )
    assert (case.lane_cx[2], case.lane_cy[2]) == (232.0, 80.0)
    expected = np.zeros(case.image.shape)
    expected[rows[0] : rows[1], 192:273] = 1000.0
    assert np.array_equal(MEMBRANE - case.image, expected)


def test_the_grid_row_draws_its_geometry():
    # Seed 1000's draws: lanes about 70 px apart from x 92, the box from half
    # a pitch before the first lane's nominal centre to half a pitch past the
    # last's, 6 px beyond the bands' extents; a pitch of membrane more on the
    # right. A tilt and a smile are measured from the nominal centre (x 267).
    case = neighbour_grid_row("grid", 1000)
    assert (case.row, case.image.shape, case.neighbour_cy) == ((57, 87, 477, 114), (200, 604), None)
    assert case.lane_cx == pytest.approx(
        [92.08554295, 162.41536739, 231.88376719, 300.81299177, 372.11503610, 440.76414512]
    )
    tilted = neighbour_grid_row("grid", 1000, tilt=12.0, smile=4.0)
    assert tilted.lane_cy == pytest.approx(
        [91.56993368, 97.49297467, 100.73829396, 103.73741534, 104.77153153, 103.51066989]
    )


def test_the_grid_row_draws_each_part_where_it_says():
    def drawn(**params) -> tuple[RowCase, np.ndarray]:
        case = neighbour_grid_row("parts", 1000, noise=0.0, **params)
        return case, MEMBRANE - case.image

    flat = {"depths": [12000.0] * 6}
    # A faint lane: a tenth of the others' mean, its peak up to half a pixel
    # off its drawn centre.
    case, dark = drawn(faint=2, **flat)
    x0, y0, x1, y1 = case.reference[2]
    assert 0.98 * 1200.0 < dark[y0:y1, x0:x1].max() <= 1200.5
    # A band 30 px below each, half as deep, or 7000 deep.
    for params, depth in (
        ({"neighbour_rel": 0.5}, 6000.0),
        ({"neighbour_depths": [7000.0] * 6}, 7000.0),
    ):
        case, dark = drawn(neighbour_dy=30.0, **flat, **params)
        y, x = round(case.neighbour_cy[3]), round(case.lane_cx[3])
        assert case.neighbour_cy[3] == case.lane_cy[3] + 30.0
        assert 0.98 * depth < dark[y, x] <= depth + 0.5
    # A pale line across each band's middle, half its depth.
    _, plain = drawn(**flat)
    case, dark = drawn(slit=(0.5, 1.2), **flat)
    y, x = round(case.lane_cy[3]), round(case.lane_cx[3])
    assert 0.49 < dark[y, x] / plain[y, x] < 0.55
    # A doublet: two parts 12 px apart, a dip between them.
    case, dark = drawn(doublet=(12.0, 0.6), **flat)
    cy, x = case.lane_cy[3], round(case.lane_cx[3])
    upper, middle, lower = (dark[round(cy + dy), x] for dy in (-6.0, 0.0, 6.0))
    assert middle < 0.2 * lower and lower < upper
    # A smear down from the left half of lane 1's band, fading over 12 px.
    case, dark = drawn(smears=[(1, 15000.0, 12.0)], **flat)
    cx, cy = case.lane_cx[1], case.lane_cy[1]
    left, right = round(cx - 11.0), round(cx + 11.0)
    y = round(cy + 15.0)
    assert dark[y, left] > 3000.0 and dark[y, right] < 50.0
    assert dark[round(cy - 15.0), left] < 50.0


def test_a_ring_stain_has_two_sides_only():
    with pytest.raises(ValueError, match="side must be 'top' or 'bottom', not 'left'"):
        ring_stain(2, "left", 1000.0)
