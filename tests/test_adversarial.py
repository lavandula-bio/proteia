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

The rows prove their own power (#181): detectors degraded on purpose (lanes
numbered by equal slots, the doubt or the cut flag left out, a duller
threshold, and so on) must score worse on them than today's, in the same
run. Five run always, on the rows at seed 1000; all of them run on every row
with ``PROTEIA_SLOW=1``.
"""

import dataclasses
import functools
import math
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from proteia.core import evaluate, rowdetect
from proteia.core.quantify import estimate_background
from proteia.core.rowdetect import LaneDetection, Peak, RowDetection
from rowcases import (
    ADVERSARIAL,
    BENCH_RECIPES,
    MEMBRANE,
    NOISE_SIGMA,
    STRESS,
    LaneScore,
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


Scored = tuple[RowScore, RowDetection, dict[int, LaneTruth]]
_SCORED: dict[str, tuple[Recipe, Scored]] = {}


def _drawn_alike(a: Recipe, b: Recipe) -> bool:
    """Whether two recipes draw and read the same row."""
    return (a.kind, a.seed, a.noise, dict(a.params), dict(a.detect)) == (
        b.kind,
        b.seed,
        b.noise,
        dict(b.params),
        dict(b.detect),
    )


def scored(recipe: Recipe) -> Scored:
    """The recipe's score, today's detection and its lanes' truth, kept by
    label: the floors and the recipe tests share them (no image is kept).
    Raises ValueError for a recipe whose label another recipe was scored
    under: it would get that recipe's score."""
    if recipe.label in _SCORED:
        kept, result = _SCORED[recipe.label]
        if not _drawn_alike(kept, recipe):
            raise ValueError(
                f"{recipe.label!r} already names another recipe: scores are kept by label"
            )
        return result
    case = recipe.build()
    found = detect(recipe, case)
    truth = lane_truth(recipe, case)
    result = (score_row(recipe, case, found, truth), found, truth)
    _SCORED[recipe.label] = (recipe, result)
    return result


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


def test_two_recipes_under_one_label_are_not_scored_as_one():
    # The scores are kept by label, so the floors and the recipe tests share
    # one detection of each row: the same recipe built again gets the kept
    # score; another recipe under that label is an error, never that score.
    first = scored(Recipe("one label", "adversarial", 1000))
    assert scored(Recipe("one label", "adversarial", 1000)) is first
    with pytest.raises(ValueError, match="'one label' already names another recipe"):
        scored(Recipe("one label", "adversarial", 1001))
    with pytest.raises(ValueError, match="'one label' already names another recipe"):
        scored(Recipe("one label", "adversarial", 1000, {"missing": [2]}))


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


def test_a_grid_box_is_compared_with_the_box_on_the_nearest_pixel():
    # Lane 4's centre lies past a pixel's middle (x 300.81): the box of the
    # same size centred on its band starts at x 279 (300.81 + 0.5 - 22,
    # rounded down), not 278. A box 12 px to the right of that holds 0.8136
    # of what it holds (0.8156 against the box at 278).
    case = GRID_ROW.build()
    assert case.lane_cx[3] == pytest.approx(300.81, abs=0.01)
    [lane] = grid_score({3: centred(case, 3, dx=12)}).lanes
    assert (lane.lane, lane.kind) == (3, "partial")
    assert lane.value == pytest.approx(0.8136, abs=1e-4)


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


# --- #181: a degraded detector must lose score ---

# A degraded detector has at least this many more silent lanes than today's
# on the same rows, or at least this many more wrong lanes and no fewer
# silent ones. Every variant below loses 6 or more on every row; a detector
# that makes no row worse loses nothing.
LOSS = 3


def _nth(labels: Sequence[str], k: int) -> str:
    return repr(labels[k]) if k < len(labels) else "none"


def _lanes(score: RowScore) -> str:
    """The row's score in words, without its label."""
    return str(score).removeprefix(f"{score.label}: ")


def not_worse(name: str, today: Sequence[RowScore], degraded: Sequence[RowScore]) -> str:
    """Why the degraded detector ``name`` does not score worse than today's on
    the same rows, "" when it does (:data:`LOSS`): both differences, the
    threshold, and the rows whose silent or wrong lanes changed (the first
    10), each with both scores. Raises ValueError when the two score other
    rows, naming the first that differs."""
    labels = [score.label for score in today]
    theirs = [score.label for score in degraded]
    if theirs != labels:
        k = next(
            k
            for k in range(max(len(labels), len(theirs)))
            if labels[k : k + 1] != theirs[k : k + 1]
        )
        raise ValueError(
            f"{name}: {len(degraded)} rows scored against today's {len(today)}, not the same"
            f" rows; row {k + 1}: {_nth(theirs, k)} against today's {_nth(labels, k)}"
        )
    silent = sum(score.silent for score in degraded) - sum(score.silent for score in today)
    wrong = sum(score.wrong for score in degraded) - sum(score.wrong for score in today)
    if silent >= LOSS or (wrong >= LOSS and silent >= 0):
        return ""
    changed = [
        (old, new)
        for old, new in zip(today, degraded, strict=True)
        if (old.silent, old.wrong) != (new.silent, new.wrong)
    ]
    rows = "".join(
        f"\n  {new.label}: silent {old.silent} -> {new.silent}, wrong {old.wrong} -> {new.wrong}"
        f"\n    today:    {_lanes(old)}"
        f"\n    degraded: {_lanes(new)}"
        for old, new in changed[:10]
    )
    return (
        f"{name}: silent {silent:+d}, wrong {wrong:+d} against today's detector on"
        f" {len(labels)} rows; it must have at least {LOSS} more silent lanes, or at least"
        f" {LOSS} more wrong lanes and no fewer silent ones. Rows changed: {len(changed)}"
        f"{' (the first 10)' if len(changed) > 10 else ''}{rows}"
    )


@dataclasses.dataclass(frozen=True)
class Degraded:
    """A detector degraded on purpose: ``patches`` sets these
    :mod:`~proteia.core.rowdetect` attributes while each row is detected
    again (monkeypatch, raising: a renamed one is an error, never a detector
    left whole), or ``rewrite`` changes today's result of the row (kept, never
    changed in place). ``quiet``: the message of a RuntimeWarning the setting
    raises, ignored."""

    id: str
    how: str
    patches: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    rewrite: Callable[[RowDetection, RowCase], RowDetection] | None = None
    quiet: str | None = None

    def detect(self, recipe: Recipe, case: RowCase, today: RowDetection) -> RowDetection:
        """The degraded detection of the recipe's row ``case``, ``today`` today's."""
        if self.rewrite is not None:
            return self.rewrite(today, case)
        with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
            if self.quiet is not None:
                warnings.filterwarnings("ignore", self.quiet, RuntimeWarning)
            for attribute, value in self.patches.items():
                patch.setattr(rowdetect, attribute, value)
            return detect(recipe, case)


# What a lane reading is checked by: its doubt and its two refusals.
_READING_FLAGS = ("doubtful_lanes", "ambiguous_lanes", "lanes_outside_row")


def equal_slots(found: RowDetection, case: RowCase) -> RowDetection:
    """``found`` numbered as a plain gel tool numbers lanes: the row box split
    into equal slots, one per lane, each box in the slot its centre falls in
    (the stronger of two), the reading never checked (its doubt and
    refusals dropped)."""
    x0, y0, x1, y1 = case.row
    n = case.n_lanes
    width = (x1 - x0) / n
    best: list[LaneDetection | None] = [None] * n
    for lane in found.lanes:
        if lane.rect is None:
            continue
        slot = min(n - 1, max(0, int(((lane.rect[0] + lane.rect[2]) / 2 - x0) // width)))
        held = best[slot]
        if held is None or lane.snr > held.snr:
            best[slot] = lane
    lanes = []
    for slot, lane in enumerate(best):
        if lane is not None:
            lanes.append(dataclasses.replace(lane, lane=slot))
            continue
        empty = dataclasses.replace(
            found.lanes[slot],
            rect=None,
            reason="no_band",
            extent=None,
            bg_offset=None,
            components=0,
            hollow=False,
            window=(int(x0 + slot * width), y0, int(x0 + (slot + 1) * width), y1),
            cut=False,
            line_offset=None,
            peaks=(),
        )
        lanes.append(empty)
    return dataclasses.replace(
        found,
        lanes=tuple(lanes),
        flags=tuple(flag for flag in found.flags if flag not in _READING_FLAGS),
        notes=tuple(note for note in found.notes if note != found.doubt_note),
    )


def half_pitch_shift(found: RowDetection, case: RowCase) -> RowDetection:
    """``found`` with every box moved right by half the row's pitch (to the
    nearest px), as a coordinate off by half a lane; a row without a pitch as
    it was."""
    if found.pitch is None:
        return found
    dx = round(found.pitch / 2)
    return dataclasses.replace(
        found,
        lanes=tuple(
            lane
            if lane.rect is None
            else dataclasses.replace(
                lane, rect=(lane.rect[0] + dx, lane.rect[1], lane.rect[2] + dx, lane.rect[3])
            )
            for lane in found.lanes
        ),
    )


def no_doubt(found: RowDetection, case: RowCase) -> RowDetection:
    """``found`` without the doubtful_lanes flag and its note: what detection
    returns when it never doubts a reading (it only adds them)."""
    return dataclasses.replace(
        found,
        flags=tuple(flag for flag in found.flags if flag != "doubtful_lanes"),
        notes=tuple(note for note in found.notes if note != found.doubt_note),
    )


def no_lines(s: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """No line or strip across the lanes, anywhere (for ``rowdetect._lines``)."""
    return np.zeros(s.shape, bool)


DEGRADED = (
    # A plain gel tool's lane numbers: wrong when the box holds a ladder, a
    # label or empty lanes, or its margins differ.
    Degraded("equal_slots", "lanes numbered by equal slots of the box", rewrite=equal_slots),
    # A coordinate off by half a lane (a mirror, a crop shift); also a check
    # of the score itself.
    Degraded("half_pitch", "every box moved right by half the pitch", rewrite=half_pitch_shift),
    # #111: a first row box over a ladder or a label reads every lane off.
    Degraded("no_doubt", "no doubtful_lanes flag", rewrite=no_doubt),
    # A box edge through a band, its net low and nothing said.
    Degraded("no_cut", "CUT_LEVEL=inf: no cut_by_row_box", {"CUT_LEVEL": math.inf}),
    # A threshold tuned on clean rows misses weak bands under real noise.
    Degraded("detect_k_12", "DETECT_K=12", {"DETECT_K": 12.0}),
    # A box dragged over two rows takes the other row's bands for its own.
    Degraded("no_off_row_line", "ROW_LINE_K=inf: no off_row_line", {"ROW_LINE_K": math.inf}),
    # One very tall band stretches the shared box.
    Degraded("no_size_guard", "SIZE_GUARD=inf", {"SIZE_GUARD": math.inf}),
    # A panel's frame line or a strip along the image's edge boxed (#116).
    Degraded("no_line_filter", "no line across the lanes", {"_lines": no_lines}),
    # Dust boxed as a band.
    Degraded(
        "weak_speck_filter",
        "MIN_WIDTH_PX=1, MIN_WIDTH_PITCH=0",
        {"MIN_WIDTH_PX": 1, "MIN_WIDTH_PITCH": 0.0},
    ),
    # Two bands in one box, no hint of the second.
    Degraded("no_second_peak", "SECOND_SHARE=inf", {"SECOND_SHARE": math.inf}),
    # Touching bands not cut apart.
    Degraded(
        "no_valley_split",
        "VALLEY_FRAC=0",
        {"VALLEY_FRAC": 0.0},
        quiet="invalid value encountered in multiply",
    ),
)
IN_CI = DEGRADED[:5]
# Each attribute the variants patch, as the detector holds it.
_UNPATCHED = {name: getattr(rowdetect, name) for variant in DEGRADED for name in variant.patches}


def self_check_rows(every: bool) -> list[Recipe]:
    """The rows the variants are scored on: every floor row and grid row, or
    the floor rows at seed 1000 and the bench's."""
    return FLOOR_ROWS + GRID if every else FLOOR_SUBSET


@functools.cache
def degraded_scores(every: bool) -> dict[str, list[RowScore]]:
    """Each variant's score of each row (:func:`self_check_rows`), by its id:
    all variants on every row, those in :data:`IN_CI` on the subset. Each row
    is drawn once for all of them and dropped before the next."""
    variants = DEGRADED if every else IN_CI
    table: dict[str, list[RowScore]] = {variant.id: [] for variant in variants}
    for recipe in self_check_rows(every):
        _, today, truth = scored(recipe)
        case = recipe.build()
        for variant in variants:
            found = variant.detect(recipe, case, today)
            table[variant.id].append(score_row(recipe, case, found, truth))
    return table


def lost_score(variant: Degraded, every: bool) -> str:
    """What :func:`not_worse` says of the variant against today's detector."""
    today = [scored(recipe)[0] for recipe in self_check_rows(every)]
    name = f"{variant.id} ({variant.how})"
    return not_worse(name, today, degraded_scores(every)[variant.id])


# Measured on main (fbc2afb), silent / wrong lanes against today's (2 / 51
# on the 96 rows at seed 1000, 2 / 231 on all 349):
#                     seed 1000      every row
#   equal_slots        +22 / -6      +58 / -22
#   half_pitch        +467 / +521  +1394 / +1838
#   no_doubt           +23 / 0       +63 / 0
#   no_cut              +8 / 0       +56 / 0
#   detect_k_12        +16 / +48     +16 / +51
#   no_off_row_line                   +9 / -9
#   no_size_guard                    +20 / +14
#   no_line_filter                    +0 / +15
#   weak_speck_filter                 +6 / +18
#   no_second_peak                    +8 / 0
#   no_valley_split                  +26 / +22
# Three settings make no row worse and are not tested (every row): no
# ambiguity margin (6 wrong lanes fewer, as many silent), DETECT_K=3 (no
# change), no second background stage (7 wrong lanes fewer).


@pytest.mark.parametrize("variant", IN_CI, ids=lambda variant: variant.id)
def test_a_degraded_detector_loses_score_on_the_seed_1000_rows(variant):
    problem = lost_score(variant, every=False)
    assert not problem, problem


@slow
@pytest.mark.parametrize("variant", DEGRADED, ids=lambda variant: variant.id)
def test_a_degraded_detector_loses_score_on_every_row(variant):
    problem = lost_score(variant, every=True)
    assert not problem, problem


def test_the_self_check_runs_the_measured_variants_on_the_measured_rows():
    # The variants and rows the deltas above were measured on: five always,
    # on the 96 rows; all eleven on the 162 floor rows and the 187 grid rows.
    assert [variant.id for variant in DEGRADED] == [
        "equal_slots",
        "half_pitch",
        "no_doubt",
        "no_cut",
        "detect_k_12",
        "no_off_row_line",
        "no_size_guard",
        "no_line_filter",
        "weak_speck_filter",
        "no_second_peak",
        "no_valley_split",
    ]
    assert [variant.id for variant in IN_CI] == [
        "equal_slots",
        "half_pitch",
        "no_doubt",
        "no_cut",
        "detect_k_12",
    ]
    every = self_check_rows(True)
    assert (len(self_check_rows(False)), len(every)) == (96, 349)
    assert len({recipe.label for recipe in every}) == 349


def test_the_valley_split_variant_ignores_its_warning():
    # VALLEY_FRAC=0 on a burnt-out band raises a RuntimeWarning; the variant
    # ignores it (the grid row that always runs).
    variant = {variant.id: variant for variant in DEGRADED}["no_valley_split"]
    label = "burnt_out/light90000_speck4_60000_-18-10"
    recipe = next(recipe for recipe in GRID if recipe.label == label)
    case, today = recipe.build(), scored(recipe)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="invalid value encountered in multiply"):
            dataclasses.replace(variant, quiet=None).detect(recipe, case, today)
        variant.detect(recipe, case, today)


def test_a_degraded_detector_leaves_the_detector_as_it_was():
    degraded_scores(False)
    assert [
        name for name, value in _UNPATCHED.items() if getattr(rowdetect, name) is not value
    ] == []


def test_a_setting_the_detector_does_not_have_is_an_error():
    # The setting patched before it is set back too.
    variant = Degraded("renamed", "a setting gone", {"CUT_LEVEL": 0.0, "NO_SUCH_SETTING": 1.0})
    with pytest.raises(AttributeError, match="NO_SUCH_SETTING"):
        variant.detect(BASE, _base_case(), scored(BASE)[1])
    assert rowdetect.CUT_LEVEL is _UNPATCHED["CUT_LEVEL"]


def counted(label: str, silent: int, wrong: int) -> RowScore:
    """A 20-lane row's score with ``silent`` silent misses and ``wrong`` -
    ``silent`` flagged ones."""
    lanes = [LaneScore(lane, "miss", "silent") for lane in range(silent)]
    lanes += [
        LaneScore(lane, "miss", "flagged", shown_by="doubtful_lanes")
        for lane in range(silent, wrong)
    ]
    return RowScore(label, None, 20, 20 - wrong, tuple(lanes))


@pytest.mark.parametrize(
    ("first", "second", "loses"),
    [
        ((3, 3), (1, 1), True),  # 3 more silent, 2 fewer wrong
        ((2, 7), (1, 1), False),  # 2 more of each
        ((0, 8), (1, 1), True),  # 3 more wrong, as many silent
        ((0, 15), (0, 1), False),  # 10 more wrong, 1 fewer silent
        ((0, 7), (1, 1), False),  # 2 more wrong
        ((0, 5), (1, 1), False),  # today's scores
    ],
)
def test_a_degraded_detector_loses_three_silent_or_three_wrong_lanes(first, second, loses):
    # Today: 1 silent and 6 wrong lanes on the two rows.
    today = [counted("a", 0, 5), counted("b", 1, 1)]
    degraded = [counted("a", *first), counted("b", *second)]
    assert (not_worse("variant", today, degraded) == "") is loses


def test_a_detector_that_does_not_lose_is_shown_with_the_rows_it_changed():
    # 13 rows, each with one wrong lane, flagged; the variant reads 12 of
    # them right: 12 fewer wrong lanes, as many silent.
    today = [counted(f"row {k}", 0, 1) for k in range(13)]
    degraded = [counted(f"row {k}", 0, 1 if k == 6 else 0) for k in range(13)]
    message = not_worse("variant (how)", today, degraded)
    assert message.startswith(
        "variant (how): silent +0, wrong -12 against today's detector on 13 rows; it must"
        " have at least 3 more silent lanes, or at least 3 more wrong lanes and no fewer"
        " silent ones. Rows changed: 12 (the first 10)\n"
        "  row 0: silent 0 -> 0, wrong 1 -> 0\n"
        "    today:    19/20 right; lane 1 miss flagged (doubtful_lanes)\n"
        "    degraded: 20/20 right\n"
        "  row 1: silent 0 -> 0, wrong 1 -> 0\n"
    )
    lines = message.splitlines()
    assert len(lines) == 31
    shown = [line.split(":")[0] for line in lines[1::3]]
    assert shown == [f"  row {k}" for k in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10)]
    # One row more silent: lane 1, wrong today, read right; lane 2 missed
    # silently. Both lanes are named.
    lane_2_silent = RowScore("row 0", None, 20, 19, (LaneScore(1, "miss", "silent"),))
    one = not_worse("variant", today[:2], [lane_2_silent, today[1]])
    assert one.endswith(
        "silent +1, wrong +0 against today's detector on 2 rows; it must have at least 3 more"
        " silent lanes, or at least 3 more wrong lanes and no fewer silent ones. Rows changed:"
        " 1\n  row 0: silent 0 -> 1, wrong 1 -> 1\n"
        "    today:    19/20 right; lane 1 miss flagged (doubtful_lanes)\n"
        "    degraded: 19/20 right; lane 2 miss silent"
    )


def test_a_degraded_detector_scored_on_other_rows_is_an_error():
    # The message names the first row that differs, counted from 1.
    today = [counted("a", 0, 1), counted("b", 0, 1)]
    with pytest.raises(
        ValueError,
        match="^variant: 1 rows scored against today's 2, not the same rows; row 2: none"
        " against today's 'b'$",
    ):
        not_worse("variant", today, today[:1])
    with pytest.raises(ValueError, match="; row 3: 'c' against today's none$"):
        not_worse("variant", today, [*today, counted("c", 0, 1)])
    with pytest.raises(
        ValueError,
        match="^variant: 2 rows scored against today's 2, not the same rows; row 2: 'c'"
        " against today's 'b'$",
    ):
        not_worse("variant", today, [today[0], counted("c", 0, 1)])
    with pytest.raises(ValueError, match="; row 1: 'b' against today's 'a'$"):
        not_worse("variant", today, today[::-1])


def test_equal_slots_number_the_boxes_by_where_they_lie():
    # The bench row with lane 2's box moved onto lane 1's band, stronger
    # than lane 1's box: slot 1 keeps it, slot 2 is empty, none of lane 2's
    # band left in it. The doubt and the reading's refusals are dropped; any
    # other flag and note stay.
    case, found, _ = base()
    x0, y0, x1, y1 = case.row
    width = (x1 - x0) / 6
    moved = {
        2: {
            "rect": found.lanes[1].rect,
            "snr": found.lanes[1].snr + 1.0,
            "extent": (1, 2, 3, 4),
            "bg_offset": 2.0,
            "hollow": True,
            "cut": True,
            "line_offset": 3.0,
        }
    }
    doubt = "lane numbers doubtful: lane 3 lies a lane early"
    flags = ("ambiguous_lanes", "lanes_outside_row", "doubtful_lanes", "background_mismatch")
    notes = ("a note", doubt)
    numbered = equal_slots(changed(found, moved, flags=flags, notes=notes), case)
    assert numbered.slots[:2] == found.slots[:2] and numbered.slots[3:] == found.slots[3:]
    assert (numbered.lanes[1].lane, numbered.lanes[1].snr) == (1, found.lanes[1].snr + 1.0)
    empty = numbered.lanes[2]
    assert (empty.rect, empty.reason, empty.components, empty.peaks) == (None, "no_band", 0, ())
    kept = (empty.extent, empty.bg_offset, empty.hollow, empty.cut, empty.line_offset)
    assert kept == (None, None, False, False, None)
    assert empty.window == (int(x0 + 2 * width), y0, int(x0 + 3 * width), y1)
    assert (numbered.flags, numbered.notes) == (("background_mismatch",), ("a note",))
    # The weaker of two boxes in a slot gives way: lane 1's own box, now.
    weaker = {2: {"rect": found.lanes[1].rect, "snr": found.lanes[1].snr - 1.0}}
    assert equal_slots(changed(found, weaker), case).lanes[1] == found.lanes[1]


def test_equal_slots_hold_a_box_beyond_the_row_box_in_the_end_slot():
    # Lane 1's box moved left of the row box and lane 6's right of it, their
    # centres 25 px outside: each stays in the slot at its own end.
    case, found, _ = base()
    x0, _, x1, _ = case.row
    first, last = found.lanes[0].rect, found.lanes[5].rect
    left = (x0 - 40, first[1], x0 - 10, first[3])
    right = (x1 + 10, last[1], x1 + 40, last[3])
    numbered = equal_slots(changed(found, {0: {"rect": left}, 5: {"rect": right}}), case)
    assert numbered.slots == (left, *found.slots[1:5], right)
    assert [lane.lane for lane in numbered.lanes] == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(("pitch", "dx"), [(68.6, 34), (69.8, 35)])
def test_a_half_pitch_shift_moves_every_box_right(pitch, dx):
    # Half the pitch to the nearest px: 34.3 is 34, 34.9 is 35.
    case, found, _ = base()
    empty = changed(found, {3: EMPTY}, pitch=pitch)
    shifted = half_pitch_shift(empty, case)
    assert shifted.lanes[3] == empty.lanes[3]
    for before, after in zip(empty.slots, shifted.slots, strict=True):
        if before is not None:
            assert after == (before[0] + dx, before[1], before[2] + dx, before[3])
    no_pitch = dataclasses.replace(found, pitch=None)
    assert half_pitch_shift(no_pitch, case) is no_pitch


def test_no_doubt_drops_the_doubt_alone():
    case, found, _ = base()
    doubt = "lane numbers doubtful: lane 3 lies a lane early"
    flags = ("doubtful_lanes", "background_mismatch")
    doubtful = changed(found, {}, flags=flags, notes=("a note", doubt))
    assert no_doubt(doubtful, case) == changed(
        found, {}, flags=("background_mismatch",), notes=("a note",)
    )
