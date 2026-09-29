# SPDX-License-Identifier: Apache-2.0
"""Tests for the single compute step, on the stored model only (no files, no pixels).

Most tests edit the ``conftest`` sample project through ``apply_change``, so every
variant is a valid model: β-catenin (``prot-7``) is the target, α-tubulin
(``prot-8``) and GAPDH (``prot-9``) are loading controls, lane 2 has no β-catenin
band, lane 3 is excluded and ``vehicle`` is the reference. The last tests pin the
compute step to the golden numbers of ``test_regression_baseline``.
"""

import json
import math
from collections.abc import Callable

import pytest

from conftest import as_legacy, assert_strict_json, make_project
from proteia.core import imaging, model, results
from proteia.core.analyze import ReduceMethod, StatisticsSetting, Tier, compare
from proteia.core.model import (
    Band,
    Box,
    ImageKind,
    ImageRef,
    Lane,
    Membrane,
    Polarity,
    Project,
    ProposalSource,
    Protein,
    Region,
    Role,
    UndetectedBand,
    UndetectedReason,
    apply_change,
)
from proteia.core.plotspec import NO_VARIATION, ErrorType, ValueKind, p_text
from proteia.core.results import Level, NoticeCode, compute_results, lane_detected, lane_nets
from proteia.viz import render_svg
from test_regression_baseline import (
    BOX_SIZE,
    CONDITIONS,
    GOLDEN,
    HEIGHT,
    HIGH,
    INCLUDED,
    LANE_X,
    LOADING,
    LOW,
    REFERENCE,
    ROW_Y,
    SAMPLES,
    TARGET,
    WIDTH,
    _assert_close,
    _box,
    _compute,
)

MU = "μ"  # Greek mu; the fixture's labels use the micro sign µ (U+00B5)
# The background of a band added to the sample (img-2, ring_median_v1), whole ring.
RING = {
    "background_level": 199.88251668003335,
    "background_mode": "symmetric",
    "background_spread": 0.0,
}


def _batch(*changes: Callable[[Project], object]) -> model.Batch:
    """The sample project's batch after ``changes``, validated as one edit."""

    def change(draft: Project) -> None:
        for edit in changes:
            edit(draft)

    return apply_change(make_project(), change)[0].batch


def _loading_ids(*ids: str) -> Callable[[Project], None]:
    def edit(draft: Project) -> None:
        draft.batch.find_protein("prot-7").loading_control_ids = list(ids)

    return edit


def _lane(index: int, **fields) -> Callable[[Project], None]:
    def edit(draft: Project) -> None:
        for name, value in fields.items():
            setattr(draft.batch.lanes[index], name, value)

    return edit


def _net(band_id: str, net: float) -> Callable[[Project], None]:
    def edit(draft: Project) -> None:
        draft.batch.find_band(band_id)[1].net = net

    return edit


def _reference(condition: str | None) -> Callable[[Project], None]:
    def edit(draft: Project) -> None:
        draft.batch.reference_condition = condition

    return edit


def _codes(res: results.Results) -> list[NoticeCode]:
    return [notice.code for notice in res.notices]


def _one(res: results.Results, code: NoticeCode) -> results.Notice:
    [notice] = [n for n in res.notices if n.code is code]
    return notice


def _ratio(target: list, loading: list) -> list:
    return [
        None if t is None or lo is None else t / lo for t, lo in zip(target, loading, strict=True)
    ]


# --- the raw table ---


def test_lane_nets_join_by_stored_lane_index():
    nets = lane_nets(make_project().batch)
    assert list(nets) == ["prot-7", "prot-8", "prot-9"]  # model protein order
    # The lane-2 gap shifts nothing.
    assert nets["prot-7"] == [4279.740326695199, 4832.277498000875, None, 2485.68288495034]
    assert nets["prot-9"] == [5120.5, 4987.25, None, None]


def test_columns_carry_the_joined_nets_and_band_ids():
    batch = make_project().batch
    res = compute_results(batch)
    assert [c.protein_id for c in res.proteins] == ["prot-7", "prot-8", "prot-9"]
    assert [c.nets for c in res.proteins] == list(lane_nets(batch).values())
    beta = res.proteins[0]
    assert (beta.name, beta.role, beta.image_id) == ("β-catenin", Role.TARGET, "img-2")
    assert beta.band_ids == ["band-10", "band-11", None, "band-12"]
    assert [lane.condition for lane in res.lanes] == ["vehicle", "vehicle", "10 µM", "10 µM"]
    assert [lane.included for lane in res.lanes] == [True, True, True, False]
    # The sample project is clean but for its unchecked bands (#112): its JPEG
    # loading control, and flags its 16-bit images leave out; and that loading
    # control's strip, calibrated from its two edges (#58, D7).
    assert {notice.code for notice in res.notices} == {
        NoticeCode.CLIPPING_NOT_CHECKED,
        NoticeCode.CALIBRATION_TWO_POINTS,
    }


# --- which loading control a series uses ---


def test_series_uses_the_loading_control_the_target_names():
    batch = _batch(_loading_ids("prot-9"))  # GAPDH, the second loading control
    nets = lane_nets(batch)
    res = compute_results(batch)
    [s] = res.series
    assert (s.target_id, s.loading_id) == ("prot-7", "prot-9")
    assert (s.target, s.loading) == ("β-catenin", "GAPDH")
    assert s.normalized == _ratio(nets["prot-7"], nets["prot-9"])
    assert s.normalized != _ratio(nets["prot-7"], nets["prot-8"])
    assert s.chart is not None
    assert s.chart.title == "β-catenin fold-change vs vehicle  (/GAPDH)"
    assert {bar.label: bar.points for bar in s.chart.bars if bar.n} == s.groups
    assert all(x.loading_id != "prot-8" for x in res.series)


def test_series_naming_the_first_loading_control_uses_it_throughout():
    batch = make_project().batch  # prot-7 names prot-8
    nets = lane_nets(batch)
    [s] = compute_results(batch).series
    assert (s.loading_id, s.loading) == ("prot-8", "α-tubulin")
    assert s.normalized == _ratio(nets["prot-7"], nets["prot-8"])
    assert s.chart is not None and s.chart.title.endswith("(/α-tubulin)")
    assert {bar.label: bar.points for bar in s.chart.bars if bar.n} == s.groups
    # 10 µM's one included lane has no β-catenin box: its slot, with no value.
    assert [(bar.label, bar.n) for bar in s.chart.bars] == [("vehicle", 2), ("10 µM", 0)]


def test_target_without_a_choice_among_two_loading_controls_has_no_series():
    batch = _batch(_loading_ids())
    res = compute_results(batch)
    assert res.series == []
    notice = _one(res, NoticeCode.LOADING_CONTROL_AMBIGUOUS)
    assert notice.protein_ids == ("prot-7",)
    assert notice.level is Level.WARNING
    assert res.proteins[0].nets == lane_nets(batch)["prot-7"]  # raw nets still shown


def test_two_loading_controls_give_two_series_in_the_named_order():
    res = compute_results(_batch(_loading_ids("prot-9", "prot-8")))
    assert [(s.target_id, s.loading_id) for s in res.series] == [
        ("prot-7", "prot-9"),
        ("prot-7", "prot-8"),
    ]


# --- the reference and the baseline ---


def test_reference_with_every_lane_excluded():
    batch = _batch(_lane(0, included=False), _lane(1, included=False), _lane(3, included=True))
    res = compute_results(batch)
    assert _codes(res).count(NoticeCode.REFERENCE_ALL_EXCLUDED) == 1
    assert NoticeCode.REFERENCE_UNUSABLE not in _codes(res)  # already explained
    notice = _one(res, NoticeCode.REFERENCE_ALL_EXCLUDED)
    assert (notice.lane_indices, notice.conditions) == ((0, 1), ("vehicle",))
    [s] = res.series
    assert (s.baseline, s.fold_change, s.chart) == (None, None, None)
    assert s.value_kind is ValueKind.LOADING_NORMALIZED
    nets = lane_nets(batch)
    assert s.normalized == _ratio(nets["prot-7"], nets["prot-8"])
    assert s.groups == {"10 µM": [s.normalized[3]]}  # the included lanes only


def test_reference_without_a_value_in_one_series_only():
    # α-tubulin loses its vehicle bands: that series has no reference value.
    def drop_tubulin_vehicle_bands(draft: Project) -> None:
        tubulin = draft.batch.find_protein("prot-8")
        tubulin.bands = [b for b in tubulin.bands if b.lane_index > 1]

    batch = _batch(
        _loading_ids("prot-8", "prot-9"), _lane(3, included=True), drop_tubulin_vehicle_bands
    )
    res = compute_results(batch)
    tubulin, gapdh = res.series
    notice = _one(res, NoticeCode.REFERENCE_UNUSABLE)
    assert notice.protein_ids == ("prot-7", "prot-8")
    assert "has no value in any included lane" in notice.message
    assert (tubulin.fold_change, tubulin.chart) == (None, None)
    assert tubulin.value_kind is ValueKind.LOADING_NORMALIZED
    assert tubulin.groups == {"10 µM": [tubulin.normalized[3]]}
    assert gapdh.value_kind is ValueKind.FOLD_CHANGE
    assert gapdh.chart is not None
    assert [bar.label for bar in gapdh.chart.bars] == ["vehicle", "10 µM"]
    assert gapdh.chart.bars[0].mean == pytest.approx(1.0)
    assert gapdh.chart.bars[1].mean is None  # GAPDH has no box in the 10 µM lanes


def test_non_positive_reference_mean_is_unusable():
    res = compute_results(_batch(_net("band-10", 0.0), _net("band-11", 0.0)))
    notice = _one(res, NoticeCode.REFERENCE_UNUSABLE)
    assert "non-positive" in notice.message
    assert res.series[0].chart is None


def test_no_reference_gives_loading_normalized_values():
    batch = _batch(_reference(None))
    res = compute_results(batch)
    assert res.tier is Tier.NORMALIZED
    assert res.reference_condition is None
    [s] = res.series
    assert s.value_kind is ValueKind.LOADING_NORMALIZED
    assert (s.baseline, s.fold_change) == (None, None)
    assert s.groups == {"vehicle": s.normalized[:2]}
    assert s.chart is not None
    assert s.chart.title == "β-catenin / α-tubulin"
    assert s.chart.value_kind is ValueKind.LOADING_NORMALIZED


def test_fold_change_divides_the_reduced_values_by_the_baseline():
    [s] = compute_results(_batch(_lane(3, included=True))).series
    assert s.value_kind is ValueKind.FOLD_CHANGE
    assert s.baseline == pytest.approx((s.normalized[0] + s.normalized[1]) / 2)
    assert s.fold_change == [None if v is None else v / s.baseline for v in s.normalized]
    assert s.groups == {
        "vehicle": [s.fold_change[0], s.fold_change[1]],
        "10 µM": [s.fold_change[3]],
    }
    assert s.chart is not None
    assert [bar.lane_indices for bar in s.chart.bars] == [[0, 1], [3]]  # lane 2 has no value


# --- what compute cannot do ---


def test_no_loading_control_is_export_only():
    def drop_loading_controls(draft: Project) -> None:
        draft.batch.proteins = [p for p in draft.batch.proteins if p.role is Role.TARGET]
        draft.batch.proteins[0].loading_control_ids = []

    res = compute_results(_batch(drop_loading_controls))
    assert res.tier is Tier.EXPORT_ONLY
    assert NoticeCode.NO_LOADING_CONTROL in _codes(res)
    assert res.series == []
    assert res.proteins[0].nets == lane_nets(make_project().batch)["prot-7"]


def test_no_target_is_export_only():
    def drop_target(draft: Project) -> None:
        draft.batch.proteins = [p for p in draft.batch.proteins if p.role is not Role.TARGET]

    res = compute_results(_batch(drop_target))
    assert res.tier is Tier.EXPORT_ONLY
    assert NoticeCode.NO_TARGET in _codes(res)
    assert res.series == []


def test_no_lanes():
    def drop_lanes(draft: Project) -> None:
        draft.batch.lanes = []
        draft.batch.reference_condition = None
        for protein in draft.batch.proteins:
            protein.bands = []

    batch = _batch(drop_lanes)
    res = compute_results(batch, plot_conditions=["vehicle"])
    assert _codes(res) == [NoticeCode.NO_LANES]
    assert res.lanes == [] and res.series == []
    assert [(c.nets, c.band_ids) for c in res.proteins] == [([], [])] * 3
    assert lane_nets(batch) == {"prot-7": [], "prot-8": [], "prot-9": []}


def test_zero_loading_net_is_reported_where_the_target_has_a_value():
    # Lane 1: both have a band. Lane 2: β-catenin has none, so nothing is lost there.
    res = compute_results(_batch(_net("band-14", 0.0), _net("band-15", 0.0)))
    notice = _one(res, NoticeCode.LOADING_NOT_POSITIVE)
    assert notice.lane_indices == (1,)
    assert notice.message == (
        "'α-tubulin' has a net of 0 in lane 2: 'β-catenin' / 'α-tubulin' has no value there"
    )  # the user's lane numbers count from 1
    assert notice.protein_ids == ("prot-7", "prot-8")
    assert res.series[0].normalized[1] is None


def test_series_with_no_included_value():
    # Only lane 3 (excluded) keeps a β-catenin band.
    def keep_lane_3(draft: Project) -> None:
        beta = draft.batch.find_protein("prot-7")
        beta.bands = [b for b in beta.bands if b.lane_index == 3]

    res = compute_results(_batch(keep_lane_3, _reference(None)))
    notice = _one(res, NoticeCode.NO_VALUES)
    assert notice.protein_ids == ("prot-7", "prot-8")
    [s] = res.series
    assert (s.groups, s.chart) == ({}, None)
    assert s.normalized[3] is not None


def test_series_with_no_value_is_one_notice_even_with_a_reference():
    # The missing baseline follows from the missing values: one notice, not two.
    def keep_lane_3(draft: Project) -> None:
        beta = draft.batch.find_protein("prot-7")
        beta.bands = [b for b in beta.bands if b.lane_index == 3]

    res = compute_results(_batch(keep_lane_3))
    assert _one(res, NoticeCode.NO_VALUES).protein_ids == ("prot-7", "prot-8")
    assert NoticeCode.REFERENCE_UNUSABLE not in _codes(res)


# --- repeats, labels, the plotted subset, extra bands ---


@pytest.mark.parametrize(
    ("method", "verb"),
    [(ReduceMethod.MEAN, "averaged"), (ReduceMethod.REPRESENTATIVE, "kept one lane of")],
)
def test_technical_repeats_are_reported_once(method, verb):
    batch = _batch(_loading_ids("prot-8", "prot-9"), _lane(1, sample="v1"))
    res = compute_results(batch, method=method)
    assert len(res.series) == 2
    assert all(s.averaged == [("vehicle", "v1")] for s in res.series)
    notice = _one(res, NoticeCode.TECHNICAL_REPEATS)
    assert notice.level is Level.INFO
    assert notice.conditions == ("vehicle",)
    assert (
        notice.message == f"{verb} 1 sample(s) with technical repeats (repeats do not count as n)"
    )
    assert res.method is method


def test_similar_conditions_are_reported():
    batch = _batch(_lane(1, label="control"), _lane(0, label="Control"), _reference("Control"))
    res = compute_results(batch)
    notice = _one(res, NoticeCode.SIMILAR_CONDITIONS)
    assert notice.conditions == ("Control", "control")
    assert {"Control", "control"} <= set(res.series[0].groups)  # still two groups


def test_unknown_plot_condition_is_ignored():
    batch = _batch(_lane(3, included=True))
    res = compute_results(batch, plot_conditions=["DMSO"])
    notice = _one(res, NoticeCode.UNKNOWN_PLOT_CONDITION)
    assert notice.conditions == ("DMSO",)
    assert res.plot_conditions is None  # nothing resolved: every condition is plotted
    chart = res.series[0].chart
    assert chart is not None and [bar.label for bar in chart.bars] == ["vehicle", "10 µM"]

    res = compute_results(batch, plot_conditions=["vehicle", "DMSO", "DMSO"])
    assert _one(res, NoticeCode.UNKNOWN_PLOT_CONDITION).conditions == ("DMSO",)
    assert res.plot_conditions == ["vehicle"]


def test_plot_condition_resolves_a_look_alike_and_keeps_the_baseline():
    batch = _batch(_lane(3, included=True))
    full = compute_results(batch)
    res = compute_results(batch, plot_conditions=[f"10 {MU}M"])
    assert res.plot_conditions == ["10 µM"]
    assert NoticeCode.UNKNOWN_PLOT_CONDITION not in _codes(res)
    assert _one(res, NoticeCode.REFERENCE_NOT_PLOTTED).level is Level.INFO
    [s], [s_full] = res.series, full.series
    assert s.baseline == s_full.baseline  # the subset never moves the baseline
    assert s.groups == s_full.groups
    assert s.chart is not None and [bar.label for bar in s.chart.bars] == ["10 µM"]
    assert s.chart.bars[0].points == s_full.groups["10 µM"]


def test_plot_conditions_must_not_be_one_string():
    with pytest.raises(TypeError):
        compute_results(make_project().batch, plot_conditions="vehicle")


def test_extra_bands_are_reported_and_not_quantified():
    def add_second_band(draft: Project) -> None:
        beta = draft.batch.find_protein("prot-7")
        beta.bands.append(
            Band(
                id=draft.new_id("band"),
                lane_index=0,
                band_index=1,
                box=Box(x=18, y=80),
                net=999.0,
                source=ProposalSource.MANUAL,
                **RING,
            )
        )

    batch = _batch(add_second_band)
    res = compute_results(batch)
    notice = _one(res, NoticeCode.EXTRA_BANDS_IGNORED)
    assert (notice.protein_ids, notice.level) == (("prot-7",), Level.INFO)
    assert res.proteins[0].nets[0] == 4279.740326695199  # the first band's net, not 999
    assert res.proteins[0].band_ids[0] == "band-10"
    assert lane_nets(batch)["prot-7"][0] == 4279.740326695199


# --- the background notices (#83) ---


def _background(band_id: str, **fields) -> Callable[[Project], None]:
    def edit(draft: Project) -> None:
        band = draft.batch.find_band(band_id)[1]
        for name, value in fields.items():
            setattr(band, name, value)

    return edit


def _all(res: results.Results, code: NoticeCode) -> list[results.Notice]:
    return [notice for notice in res.notices if notice.code is code]


def test_a_ring_cut_short_is_reported_with_the_membrane_to_leave():
    # β-catenin (24x14 boxes): lanes 0 and 1, and lane 3, which is excluded;
    # α-tubulin (20x10): lane 0. A whole ring (symmetric) is never reported.
    res = compute_results(
        _batch(
            _background("band-10", background_mode="asymmetric"),
            _background("band-11", background_mode="image"),
            _background("band-12", background_mode="image"),
            _background("band-13", background_mode="asymmetric"),
        )
    )

    def notice(name: str, protein_id: str, lanes: tuple[int, ...], margins: str):
        where = "lane 1" if lanes == (0,) else "lanes 1, 2"
        return results.Notice(
            code=NoticeCode.BACKGROUND_FALLBACK,
            level=Level.WARNING,
            message=(
                f"{name!r} has too little membrane around its box in {where}: the image edge"
                " or other boxes cut it, so the background there is estimated less surely;"
                f" when cropping, leave about {margins} beside them (one box height, 0.4 box"
                " width)"
            ),
            protein_ids=(protein_id,),
            lane_indices=lanes,
        )

    assert _all(res, NoticeCode.BACKGROUND_FALLBACK) == [
        notice("β-catenin", "prot-7", (0, 1), "14 px above and below the bands and 10 px"),
        notice("α-tubulin", "prot-8", (0,), "10 px above and below the bands and 8 px"),
    ]
    # The excluded lane is reported with every lane included.
    [every] = _all(res.all_lanes, NoticeCode.BACKGROUND_FALLBACK)
    assert (every.protein_ids, every.lane_indices) == (("prot-7",), (0, 1, 3))
    assert _all(compute_results(make_project().batch), NoticeCode.BACKGROUND_FALLBACK) == []


def test_a_background_uneven_around_the_box_is_reported():
    # band-10 (β-catenin, lane 0): net 4279.74 over a 24x14 box, so its net
    # moves by 5 % at a spread of 0.05 * 4279.74 / 336 levels.
    limit = 0.05 * 4279.740326695199 / 336
    res = compute_results(
        _batch(
            _background("band-10", background_spread=limit * 1.001),
            _background("band-11", background_spread=limit * 0.999),  # net 4832: well under
            _net("band-14", 0.0),  # α-tubulin lane 1: any spread on a net of 0
            _background("band-14", background_spread=1e-9),
            _net("band-15", 0.0),  # α-tubulin lane 2: no spread, nothing to report
        )
    )
    uneven = _all(res, NoticeCode.BACKGROUND_UNEVEN)
    assert [(n.protein_ids, n.lane_indices, n.level) for n in uneven] == [
        (("prot-7",), (0,), Level.WARNING),
        (("prot-8",), (1,), Level.WARNING),
    ]
    assert uneven[0].message == (
        "'β-catenin' has an uneven background around its box in lane 1: it differs from"
        " side to side by more than 5% of the net; check the membrane there"
    )
    just_under = _batch(_background("band-10", background_spread=limit * 0.999))
    assert _all(compute_results(just_under), NoticeCode.BACKGROUND_UNEVEN) == []


def test_a_legacy_project_is_told_to_requantify():
    legacy = as_legacy(make_project())
    res = compute_results(legacy.batch)
    notice = _one(res, NoticeCode.LEGACY_BACKGROUND)
    assert notice == results.Notice(
        code=NoticeCode.LEGACY_BACKGROUND,
        level=Level.INFO,
        message=(
            "nets use the whole-image median background, as measured before the local"
            " background; requantify to measure each band's background from the membrane"
            " around its box"
        ),
        protein_ids=("prot-7", "prot-8", "prot-9"),
    )
    # Before any lane is declared too, and not once the project measures locally.
    no_lanes = legacy.batch.model_copy(update={"lanes": [], "reference_condition": None})
    assert NoticeCode.LEGACY_BACKGROUND in _codes(compute_results(no_lanes))
    assert NoticeCode.LEGACY_BACKGROUND not in _codes(compute_results(make_project().batch))
    # The same numbers either way: the notice is about the method only.
    assert lane_nets(legacy.batch) == lane_nets(make_project().batch)


def test_nets_measured_on_a_reading_no_longer_made_are_named():
    # #131: the session says which images were read differently when imported,
    # and why; the results warn once about the proteins measured on each. The
    # marker image img-3 holds no band, so it has no nets to warn about.
    why = "'blot.tif' is a CMYK file imported before Proteia converted CMYK colours"
    batch = make_project().batch
    res = compute_results(batch, outdated={"img-2": why, "img-3": "no bands"})
    assert _all(res, NoticeCode.OUTDATED_READING) == [
        results.Notice(
            code=NoticeCode.OUTDATED_READING,
            level=Level.WARNING,
            message=why,
            protein_ids=("prot-7",),
        )
    ]
    assert res.all_lanes is not None  # both sets hold it: it is kept in the first
    assert NoticeCode.OUTDATED_READING not in _codes(res.all_lanes)
    assert NoticeCode.OUTDATED_READING not in _codes(compute_results(batch))
    # Before any lane is declared too.
    no_lanes = batch.model_copy(update={"lanes": [], "reference_condition": None})
    assert NoticeCode.OUTDATED_READING in _codes(compute_results(no_lanes, outdated={"img-2": why}))


# --- purity ---


def test_each_series_is_reduced_once(monkeypatch):
    calls = []
    original = results.reduce_samples

    def counting(*args, **kwargs):
        calls.append(kwargs.get("included"))
        return original(*args, **kwargs)

    monkeypatch.setattr(results, "reduce_samples", counting)
    batch = _batch(_loading_ids("prot-8", "prot-9"))
    compute_results(batch, plot_conditions=["vehicle"])
    # Two series, one reduction each, in each of the two sets (lane 3 is excluded).
    assert len(calls) == 4
    assert calls[:2] == [[True, True, True, False]] * 2  # the lane table's
    assert calls[2:] == [[True, True, True, True]] * 2  # every lane


def test_compute_leaves_the_batch_unchanged():
    batch = _batch(_loading_ids("prot-8", "prot-9"), _lane(1, sample="v1"))
    before = batch.model_copy(deep=True)
    compute_results(batch, plot_conditions=["10 µM"], method=ReduceMethod.REPRESENTATIVE)
    compute_results(batch)
    assert batch == before


# --- pinned to the golden numbers of the regression baseline ---


def _baseline_batch(reference: str | None) -> model.Batch:
    """The regression baseline's blot as a stored batch: its lane table, both proteins
    on one image with its fixed boxes, and its dark-on-light nets as the stored nets."""
    computed = _compute()
    nets = computed["nets"]["dark_on_light"]
    background = computed["background"]
    image = ImageRef(
        id="img-2",
        file="img-2.tif",
        original_name="β-catenin α-tubulin 10 µM.tif",
        kind=ImageKind.CHEMILUMINESCENCE,
        sha256="0" * 64,  # never read: compute takes no pixels
        width=WIDTH,
        height=HEIGHT,
        polarity=Polarity.DARK_ON_LIGHT,
        background=background["image_median"]["dark_on_light"],
    )
    number = iter(range(5, 100))
    proteins = [
        Protein(
            id=protein_id,
            name=name,
            role=role,
            image_id=image.id,
            box_size=BOX_SIZE,
            bands=[
                Band(
                    id=f"band-{next(number)}",
                    lane_index=i,
                    box=_box(cx, ROW_Y[name]),
                    net=nets[name][i],
                    background_level=background["levels"]["dark_on_light"][name][i],
                    background_mode=background["modes"]["dark_on_light"][name][i],
                    background_spread=background["spreads"]["dark_on_light"][name][i],
                    source=ProposalSource.MANUAL,
                )
                for i, cx in enumerate(LANE_X)
            ],
        )
        for protein_id, name, role in (
            ("prot-3", TARGET, Role.TARGET),
            ("prot-4", LOADING, Role.LOADING_CONTROL),
        )
    ]
    lanes = [
        Lane(index=i, label=condition, sample=sample, included=included)
        for i, (condition, sample, included) in enumerate(
            zip(CONDITIONS, SAMPLES, INCLUDED, strict=True)
        )
    ]
    return model.Batch(
        lanes=lanes,
        reference_condition=reference,
        membranes=[Membrane(id="mem-1", images=[image])],
        proteins=proteins,
    )


def _chart_stats(res_sd: results.Results, res_sem: results.Results) -> dict:
    """A chart's statistics in the golden file's shape: its bars and its test,
    every comparison of it."""
    [sd], [sem] = res_sd.series, res_sem.series
    assert sd.chart is not None and sem.chart is not None
    test = sd.chart.test
    assert test is not None
    return {
        "describe": [
            {
                "label": a.label,
                "n": a.n,
                "mean": a.mean,
                "sd": a.error,
                "sem": b.error,
                "geometric_mean": a.geometric_mean,
                "geometric_sd_factor": a.geometric_sd_factor,
            }
            for a, b in zip(sd.chart.bars, sem.chart.bars, strict=True)
        ],
        "compare": {
            "test": test.id,
            "p_value": test.p_value,
            "statistic": test.statistic,
            "pairwise": [p.model_dump() for p in test.pairwise],
        },
    }


@pytest.mark.parametrize("method", list(ReduceMethod))
@pytest.mark.parametrize(
    ("reference", "kind", "golden_kind"),
    [
        (REFERENCE, ValueKind.FOLD_CHANGE, "fold_change"),
        (None, ValueKind.LOADING_NORMALIZED, "normalized"),
    ],
)
def test_compute_matches_the_regression_baseline(method, reference, kind, golden_kind):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    rel, abs_ = golden["tolerance"]["rel"], golden["tolerance"]["abs"]
    values = golden["values"]
    reduced = values["reduced"][f"{golden_kind}/{method}"]
    batch = _baseline_batch(reference)

    res = compute_results(batch, method=method)
    unchecked = (
        "was not checked for over-exposure in lanes 1, 2, 3, 4, 5, 7, 8: its image has an"
        " unknown bit depth, so saturated pixels cannot be counted; if it is over-exposed"
        " there, its net is an under-estimate"
    )
    assert res.notices == [
        # The baseline blot is a float image: no detector limit to check against (#112).
        *(
            results.Notice(
                code=NoticeCode.CLIPPING_NOT_CHECKED,
                level=Level.WARNING,
                message=f"{name!r} {unchecked}{effect}",
                protein_ids=(protein_id,),
                lane_indices=(0, 1, 2, 3, 4, 6, 7),
            )
            for name, protein_id, effect in (
                (TARGET, "prot-3", ""),
                (LOADING, "prot-4", ", which biases every value normalized to it"),
            )
        ),
        # The blot's ±2-level texture moves target lanes 1 and 7 by more than 5 %
        # of their nets from side to side (lane 6, as bad, is excluded).
        results.Notice(
            code=NoticeCode.BACKGROUND_UNEVEN,
            level=Level.WARNING,
            message=(
                f"{TARGET!r} has an uneven background around its box in lanes 1, 7: it"
                " differs from side to side by more than 5% of the net; check the membrane"
                " there"
            ),
            protein_ids=("prot-3",),
            lane_indices=(0, 6),
        ),
        results.Notice(
            code=NoticeCode.TECHNICAL_REPEATS,
            level=Level.INFO,
            message=(
                ("averaged" if method is ReduceMethod.MEAN else "kept one lane of")
                + " 1 sample(s) with technical repeats (repeats do not count as n)"
            ),
            conditions=(REFERENCE,),
        ),
    ]
    [s] = res.series
    assert s.value_kind is kind
    nets = values["nets"]["dark_on_light"]
    _assert_close([c.nets for c in res.proteins], [nets[TARGET], nets[LOADING]], rel, abs_)
    _assert_close(s.normalized, values["lanes"]["normalized"], rel, abs_, "normalized")
    if reference is not None:
        _assert_close(
            s.fold_change, values["lanes"][f"fold_change/{method}"], rel, abs_, "fold_change"
        )
    _assert_close(s.groups, reduced["groups"], rel, abs_, "groups")
    assert [list(key) for key in s.averaged] == reduced["averaged"]

    for case, plot in (("all", None), ("two", [REFERENCE, LOW])):
        charts = [
            compute_results(batch, plot_conditions=plot, error_type=error_type, method=method)
            for error_type in (ErrorType.SD, ErrorType.SEM)
        ]
        _assert_close(_chart_stats(*charts), reduced[case], rel, abs_, case)
        # The brackets: the comparisons with an adjusted p below 0.05.
        brackets = [c.model_dump() for c in charts[0].series[0].chart.comparisons]
        expected = [
            {key: pw[key] for key in ("group_a", "group_b", "p_value")}
            for pw in reduced[case]["compare"]["pairwise"]
            if pw["p_value"] < 0.05
        ]
        _assert_close(brackets, expected, rel, abs_, f"{case} brackets")


# --- review of #69: provenance, notices that must not mislead, tier ---


@pytest.mark.parametrize("method", list(ReduceMethod))
def test_chart_lanes_stay_parallel_to_points_with_technical_repeats(method):
    # Lanes 0 and 1 are one sample loaded twice: one point, whose lane is lane 0.
    res = compute_results(_batch(_lane(1, sample="v1")), method=method)
    bars = {bar.label: bar for bar in res.series[0].chart.bars}
    for bar in bars.values():
        assert len(bar.lane_indices) == len(bar.points)
    assert bars["vehicle"].lane_indices == [0]


def test_loading_net_of_zero_in_an_excluded_lane_is_not_a_warning():
    res = compute_results(_batch(_net("band-16", 0.0)))  # α-tubulin, lane 3, include=no
    assert NoticeCode.LOADING_NOT_POSITIVE not in _codes(res)
    res = compute_results(_batch(_net("band-13", 0.0)))  # α-tubulin, lane 0, included
    assert _one(res, NoticeCode.LOADING_NOT_POSITIVE).lane_indices == (0,)


def test_no_fold_change_means_no_unplotted_reference_notice_and_a_lower_tier():
    excluded = _batch(_lane(0, included=False), _lane(1, included=False))
    res = compute_results(excluded, plot_conditions=["10 µM"])
    assert NoticeCode.REFERENCE_ALL_EXCLUDED in _codes(res)
    assert NoticeCode.REFERENCE_NOT_PLOTTED not in _codes(res)
    assert res.tier is Tier.NORMALIZED  # no series can form a fold-change


def test_series_without_a_value_in_the_plotted_conditions_says_so():
    # β-catenin has no included 10 µM value (lane 2 empty, lane 3 excluded).
    res = compute_results(_batch(), plot_conditions=["10 µM"])
    [series] = res.series
    assert series.chart is None and series.groups  # values exist, just not plotted
    assert _one(res, NoticeCode.NO_PLOTTED_VALUES).protein_ids == ("prot-7", "prot-8")


# --- #71: results with and without excluded lanes ---


def test_without_excluded_lanes_there_is_one_set():
    res = compute_results(_batch(_lane(3, included=True)))
    assert res.excluded_lanes == []
    assert res.all_lanes is None


def test_excluded_lanes_come_with_an_all_lanes_set():
    res = compute_results(_batch())  # lane 3 (10 µM, sample a2) is include=no
    assert res.excluded_lanes == [3]
    everything = res.all_lanes
    assert everything is not None
    assert (everything.excluded_lanes, everything.all_lanes) == ([], None)
    [applied], [all_lanes] = res.series, everything.series
    assert "10 µM" not in applied.groups  # its only β-catenin value is in lane 3
    assert len(all_lanes.groups["10 µM"]) == 1
    assert all_lanes.chart is not None
    assert [bar.label for bar in all_lanes.chart.bars] == ["vehicle", "10 µM"]
    # Both sets use their own lanes for the fold-change baseline: here the same.
    assert applied.baseline == all_lanes.baseline
    assert everything.lanes == [row.model_copy(update={"included": True}) for row in res.lanes]


def test_excluding_the_reference_keeps_the_fold_change_in_the_all_lanes_set():
    res = compute_results(_batch(_lane(0, included=False), _lane(1, included=False)))
    assert NoticeCode.REFERENCE_ALL_EXCLUDED in _codes(res)
    assert [s.fold_change for s in res.series] == [None]
    [series] = res.all_lanes.series
    assert series.value_kind is ValueKind.FOLD_CHANGE
    assert series.chart is not None
    assert NoticeCode.REFERENCE_ALL_EXCLUDED not in _codes(res.all_lanes)


def test_too_few_samples_give_a_chart_without_statistics():
    # Lane 1 excluded, lane 3 included: vehicle and 10 µM have one sample each.
    res = compute_results(_batch(_lane(1, included=False), _lane(3, included=True)))
    [series] = res.series
    chart = series.chart
    assert chart is not None
    assert [bar.n for bar in chart.bars] == [1, 1]
    assert (chart.test_name, chart.test_p, chart.comparisons) == (None, None, [])
    # The reason no test ran names the groups, where the core's note names none.
    core = compare(series.groups, StatisticsSetting(), ratio=True, reference="vehicle")
    assert core.note == "need >=2 groups with >=2 replicates for a test"
    assert chart.test_note == "no test: 'vehicle', '10 µM' have fewer than 2 replicates"
    # The all-lanes set has vehicle n = 2, still too few groups with two samples to test.
    all_chart = res.all_lanes.series[0].chart
    assert all_chart is not None
    assert (all_chart.test_name, all_chart.test_p, all_chart.comparisons) == (None, None, [])
    assert all_chart.test_note == "no test: '10 µM' has fewer than 2 replicates"


def test_excluded_lanes_without_values_add_no_second_set():
    # A ladder or empty lane marked include=no removes no data point.
    def empty_lane_3(draft: Project) -> None:
        for protein in draft.batch.proteins:
            protein.bands = [b for b in protein.bands if b.lane_index != 3]

    res = compute_results(_batch(empty_lane_3))
    assert res.excluded_lanes == [3]
    assert res.all_lanes is None
    assert res.label is None  # one set needs no name


def test_the_label_names_only_the_excluded_lanes_that_hold_values():
    # Lane 0 is emptied (a ladder) and excluded; lane 3, excluded, holds values.
    def ladder_in_lane_0(draft: Project) -> None:
        for protein in draft.batch.proteins:
            protein.bands = [b for b in protein.bands if b.lane_index != 0]
        draft.batch.lanes[0].included = False

    res = compute_results(_batch(ladder_in_lane_0))
    assert res.excluded_lanes == [0, 3]  # both are left out of this set
    assert (res.label, res.all_lanes.label) == ("Excluding lane 4", "All lanes")


def test_notices_shared_by_both_sets_appear_once():
    res = compute_results(_batch(), plot_conditions=["vehicle", "no such condition"])
    assert NoticeCode.UNKNOWN_PLOT_CONDITION in _codes(res)
    assert NoticeCode.UNKNOWN_PLOT_CONDITION not in _codes(res.all_lanes)


def test_reference_all_excluded_says_the_set_is_the_included_lanes():
    res = compute_results(_batch(_lane(0, included=False), _lane(1, included=False)))
    notice = _one(res, NoticeCode.REFERENCE_ALL_EXCLUDED)
    assert "from the included lanes" in notice.message


def test_the_all_lanes_set_has_no_set_of_its_own():
    dump = compute_results(_batch()).model_dump()
    assert dump["all_lanes"] is not None
    # Nesting the whole result (which has its own all-lanes set) one level deeper.
    with pytest.raises(ValueError, match="no all-lanes set of its own"):
        results.Results.model_validate({**dump, "all_lanes": dump})


# --- #52: labelled sets, charts without a test, lane numbers, raw arguments ---


@pytest.mark.parametrize(
    ("excluded", "label"),
    [((3,), "Excluding lane 4"), ((3, 0), "Excluding lanes 1, 4")],  # 1-based, ascending
)
def test_the_two_sets_and_their_charts_are_labelled(excluded, label):
    batch = _batch(_loading_ids("prot-8", "prot-9"), *(_lane(i, included=False) for i in excluded))
    res = compute_results(batch)
    assert res.excluded_lanes == sorted(excluded)  # the indices stay 0-based
    assert (res.label, res.all_lanes.label) == (label, "All lanes")
    for one_set in (res, res.all_lanes):
        charts = [s.chart for s in one_set.series]
        assert len(charts) == 2 and None not in charts
        assert [chart.subtitle for chart in charts] == [one_set.label] * 2


def test_one_set_has_no_label_and_its_charts_no_subtitle():
    res = compute_results(_batch(_loading_ids("prot-8", "prot-9"), _lane(3, included=True)))
    assert (res.label, res.all_lanes) == (None, None)
    assert [s.chart.subtitle for s in res.series] == [None, None]


def _no_variation(draft: Project) -> None:
    """Every lane included and boxed, every β-catenin net 1000 and every α-tubulin
    net 2000: both conditions' fold-changes are exactly [1.0, 1.0]."""
    beta = draft.batch.find_protein("prot-7")
    beta.bands.append(
        Band(
            id=draft.new_id("band"),
            lane_index=2,
            box=Box(x=98, y=43),
            net=1000.0,
            source=ProposalSource.MANUAL,
            **RING,
        )
    )
    for band in beta.bands:
        band.net = 1000.0
    for band in draft.batch.find_protein("prot-8").bands:
        band.net = 2000.0
    draft.batch.lanes[3].included = True


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
def test_values_that_do_not_vary_give_a_chart_with_a_note_and_no_test():
    res = compute_results(_batch(_no_variation))
    [series] = res.series
    assert series.groups == {"vehicle": [1.0, 1.0], "10 µM": [1.0, 1.0]}
    core = compare(series.groups, StatisticsSetting(), ratio=True, reference="vehicle")
    assert math.isnan(core.p_value)  # what the core gives
    chart = series.chart
    assert chart is not None and [bar.n for bar in chart.bars] == [2, 2]
    assert (chart.test_name, chart.test_p, chart.comparisons) == (None, None, [])
    assert chart.test_note == NO_VARIATION == "no test: the values do not vary"
    assert_strict_json(res)


def test_a_group_of_one_gives_a_chart_with_a_note_and_no_brackets():
    # Lane 3 included: vehicle has two samples, 10 µM one, so no test can run.
    res = compute_results(_batch(_lane(3, included=True)))
    chart = res.series[0].chart
    assert chart is not None and [bar.n for bar in chart.bars] == [2, 1]
    assert (chart.test_name, chart.test_p, chart.comparisons) == (None, None, [])
    assert chart.test_note == "no test: '10 µM' has fewer than 2 replicates"


def test_an_exclusion_that_leaves_one_of_three_groups_with_one_sample_tests_the_other_two():
    # The baseline blot with lane 8 excluded too: 50 µM keeps only b1. The chart
    # shows the core's Student's t of vehicle and 10 µM, and says that 50 µM is not
    # in it.
    baseline = _baseline_batch(REFERENCE)
    lanes = [
        lane.model_copy(update={"included": False}) if lane.index == 7 else lane
        for lane in baseline.lanes
    ]
    res = compute_results(baseline.model_copy(update={"lanes": lanes}))
    assert res.label == "Excluding lanes 6, 8"
    [series] = res.series
    tested = compare(series.groups, StatisticsSetting(), ratio=True, reference=REFERENCE)
    assert (tested.test, tested.p_value < 0.05) == ("student_t", True)  # what the core gives
    chart = series.chart
    assert chart is not None
    assert [(bar.label, bar.n) for bar in chart.bars] == [(REFERENCE, 2), (LOW, 2), (HIGH, 1)]
    assert (chart.test_name, chart.test_p) == ("student_t", tested.p_value)
    assert [(c.group_a, c.group_b, c.p_value) for c in chart.comparisons] == [
        (REFERENCE, LOW, tested.p_value)  # the one bracket, between the tested conditions
    ]
    assert HIGH == "50 µM" and chart.test_note == "'50 µM' (n = 1) is not in the test"
    assert chart.subtitle == res.label
    assert_strict_json(res)
    all_chart = res.all_lanes.series[0].chart
    assert all_chart is not None and [bar.n for bar in all_chart.bars] == [2, 3, 2]
    # n of 2, 3 and 2 with the reference tested: Welch's t-tests vs the reference.
    assert (all_chart.test_name, all_chart.test_note) == ("welch_t_holm", None)


def _clipped(*band_ids: str) -> Callable[[Project], None]:
    def edit(draft: Project) -> None:
        for band_id in band_ids:
            draft.batch.find_band(band_id)[1].clipped = True

    return edit


def test_notices_count_lanes_from_one():
    res = compute_results(_batch(_clipped("band-11"), _net("band-13", 0.0), _net("band-14", 0.0)))
    clipped = _one(res, NoticeCode.CLIPPED)
    assert clipped.lane_indices == (1,)
    assert clipped.message == (
        "'β-catenin' is over-exposed in lane 2: pixels at the detector limit make its net"
        " an under-estimate; the lane stays included"
    )
    loading = _one(res, NoticeCode.LOADING_NOT_POSITIVE)
    assert loading.lane_indices == (0, 1)
    assert loading.message == (
        "'α-tubulin' has a net of 0 in lanes 1, 2: 'β-catenin' / 'α-tubulin' has no value there"
    )

    res = compute_results(_batch(_clipped("band-11", "band-10")))
    clipped = _one(res, NoticeCode.CLIPPED)
    assert clipped.lane_indices == (0, 1)
    assert "over-exposed in lanes 1, 2:" in clipped.message
    assert clipped.message.endswith("the lanes stay included")


@pytest.mark.parametrize("method", ["mean", "representative"])
@pytest.mark.parametrize("error_type", ["SD", "SEM"])
def test_raw_error_type_and_method_behave_like_their_enums(error_type, method):
    # The baseline blot has technical repeats and groups of two or more samples, so
    # every other pair of arguments gives other charts: a mix-up cannot pass.
    batch = _baseline_batch(REFERENCE)
    raw = compute_results(batch, error_type=error_type, method=method)
    enums = ErrorType(error_type), ReduceMethod(method)
    assert raw == compute_results(batch, error_type=enums[0], method=enums[1])
    assert (raw.error_type, raw.method) == enums
    for other_error in ErrorType:
        for other_method in ReduceMethod:
            if (other_error, other_method) != enums:
                other = compute_results(batch, error_type=other_error, method=other_method)
                assert other.series != raw.series


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("error_type", "sd", "'sd' is not a valid ErrorType"),
        ("method", "median", "'median' is not a valid ReduceMethod"),
    ],
)
def test_an_unknown_error_type_or_method_is_a_value_error(argument, value, message):
    with pytest.raises(ValueError, match=message):
        compute_results(make_project().batch, **{argument: value})


# --- #51: lanes where detection found no band ---


def _undetected(protein_id: str, *lanes: int, band_index: int = 0) -> Callable[[Project], None]:
    """Not-detected records for a protein in ``lanes``."""

    def edit(draft: Project) -> None:
        protein = draft.batch.find_protein(protein_id)
        for lane in lanes:
            protein.undetected.append(
                UndetectedBand(
                    lane_index=lane,
                    band_index=band_index,
                    reason=UndetectedReason.BELOW_DETECTION_LIMIT,
                    snr=1.25,
                    threshold=6.0,
                    region=Region(x0=10 * lane, y0=30, x1=10 * lane + 10, y1=60),
                    source=ProposalSource.ROW_BOX,
                )
            )

    return edit


def _no_bands(protein_id: str, *lanes: int) -> Callable[[Project], None]:
    def edit(draft: Project) -> None:
        protein = draft.batch.find_protein(protein_id)
        protein.bands = [b for b in protein.bands if b.lane_index not in lanes]

    return edit


def _expected_bands(protein_id: str, count: int) -> Callable[[Project], None]:
    def edit(draft: Project) -> None:
        draft.batch.find_protein(protein_id).expected_band_count = count

    return edit


def test_detected_tells_a_band_from_a_record_from_nothing():
    batch = _batch(_undetected("prot-7", 2), _undetected("prot-9", 3))
    detected = lane_detected(batch)
    assert detected == {
        "prot-7": [True, True, False, True],
        "prot-8": [True, True, True, True],
        "prot-9": [True, True, None, False],
    }
    res = compute_results(batch)
    assert [column.detected for column in res.proteins] == list(detected.values())
    assert lane_detected(make_project().batch)["prot-7"] == [True, True, None, True]
    # Only band index 0 counts, like the nets.
    band_one = _batch(_expected_bands("prot-9", 2), _undetected("prot-9", 2, band_index=1))
    assert lane_detected(band_one)["prot-9"] == [True, True, None, None]

    def no_lanes(draft: Project) -> None:
        draft.batch.lanes, draft.batch.reference_condition = [], None
        for protein in draft.batch.proteins:
            protein.bands = []

    assert lane_detected(_batch(no_lanes)) == {"prot-7": [], "prot-8": [], "prot-9": []}
    assert [column.detected for column in compute_results(_batch(no_lanes)).proteins] == [[]] * 3


def test_below_detection_is_a_warning_per_protein_in_the_included_lanes():
    # β-catenin: records in lane 1 (vehicle), lane 2 (10 µM) and excluded lane 3.
    # α-tubulin: a record in lane 2. GAPDH: none.
    batch = _batch(
        _no_bands("prot-7", 1, 3),
        _undetected("prot-7", 1, 2, 3),
        _no_bands("prot-8", 2),
        _undetected("prot-8", 2),
    )
    res = compute_results(batch)
    target, loading = [n for n in res.notices if n.code is NoticeCode.BELOW_DETECTION]
    assert target.level is loading.level is Level.WARNING
    assert (target.protein_ids, target.lane_indices, target.conditions) == (
        ("prot-7",),
        (1, 2),  # 0-based; lane 3 is excluded from this set
        ("vehicle", "10 µM"),
    )
    assert target.message == (
        "'β-catenin' was not detected in lanes 2, 3 (below the detection limit):"
        " those lanes have no value and are left out of the statistics"
    )
    assert (loading.protein_ids, loading.lane_indices, loading.conditions) == (
        ("prot-8",),
        (2,),
        ("10 µM",),
    )
    assert loading.message == (
        "'α-tubulin' was not detected in lane 3 (below the detection limit):"
        " targets normalized to it have no value there"
    )
    # The all-lanes set includes lane 3, so its target notice is its own.
    everything = res.all_lanes
    [own] = [n for n in everything.notices if n.code is NoticeCode.BELOW_DETECTION]
    assert (own.protein_ids, own.lane_indices) == (("prot-7",), (1, 2, 3))
    assert own.message.startswith("'β-catenin' was not detected in lanes 2, 3, 4 ")
    assert own.conditions == ("vehicle", "10 µM")


def test_one_lane_below_detection_reads_in_the_singular():
    res = compute_results(_batch(_undetected("prot-7", 2)))
    notice = _one(res, NoticeCode.BELOW_DETECTION)
    assert notice.message == (
        "'β-catenin' was not detected in lane 3 (below the detection limit):"
        " that lane has no value and is left out of the statistics"
    )
    assert notice.lane_indices == (2,)


def test_a_record_in_an_excluded_lane_gives_no_notice_in_that_set():
    batch = _batch(_no_bands("prot-7", 3), _undetected("prot-7", 3))
    res = compute_results(batch)
    assert NoticeCode.BELOW_DETECTION not in _codes(res)
    assert _one(res.all_lanes, NoticeCode.BELOW_DETECTION).lane_indices == (3,)


def test_records_carry_no_value():
    plain = compute_results(_batch())
    batch = _batch(_undetected("prot-7", 2), _undetected("prot-9", 2, 3))
    res = compute_results(batch)
    assert lane_nets(batch) == lane_nets(_batch())
    for with_records, without in ((res, plain), (res.all_lanes, plain.all_lanes)):
        assert [c.nets for c in with_records.proteins] == [c.nets for c in without.proteins]
        assert (with_records.label, with_records.excluded_lanes, with_records.tier) == (
            without.label,
            without.excluded_lanes,
            without.tier,
        )
        for a, b in zip(with_records.series, without.series, strict=True):
            assert a.model_copy(update={"undetected": {}, "chart": None}) == b.model_copy(
                update={"chart": None}
            )
            # The charts draw the same values; a record only says why a lane has none.
            assert [bar.points for bar in a.chart.bars] == [bar.points for bar in b.chart.bars]
        assert [n for n in with_records.notices if n.code is not NoticeCode.BELOW_DETECTION] == (
            without.notices
        )


def test_records_alone_never_create_the_all_lanes_set():
    # Lane 3 is excluded and holds no band once they are removed: only records.
    no_values = [_no_bands(p, 3) for p in ("prot-7", "prot-8")]
    res = compute_results(_batch(*no_values, _undetected("prot-7", 3), _undetected("prot-9", 3)))
    assert (res.all_lanes, res.label, res.excluded_lanes) == (None, None, [3])
    assert NoticeCode.BELOW_DETECTION not in _codes(res)  # lane 3 is not in this set


def test_series_lists_the_targets_not_detected_lanes():
    # β-catenin: records in lanes 2 and 3 (excluded); α-tubulin: a record in lane 1.
    batch = _batch(
        _no_bands("prot-7", 3),
        _undetected("prot-7", 2, 3),
        _no_bands("prot-8", 1),
        _undetected("prot-8", 1),
    )
    res = compute_results(batch)
    [series] = res.series
    assert series.undetected == {"10 µM": [2]}  # included lanes; loading records left out
    [every] = res.all_lanes.series
    assert every.undetected == {"10 µM": [2, 3]}
    [plain] = compute_results(_batch()).series
    assert plain.undetected == {}
    assert_strict_json(res)


def test_a_technical_repeat_with_one_lane_not_detected_keeps_its_measured_lane():
    # Lanes 0 and 1 become technical repeats of one vehicle sample; lane 1 is n.d.
    batch = _batch(_lane(1, sample="v1"), _no_bands("prot-7", 1), _undetected("prot-7", 1))
    res = compute_results(batch)
    [series] = res.series
    assert series.groups["vehicle"] == [1.0]  # lane 0 alone, as the fold-change baseline
    assert series.normalized[1] is None
    assert series.averaged == []  # one measured lane: nothing averaged
    assert series.undetected == {"vehicle": [1]}
    assert _one(res, NoticeCode.BELOW_DETECTION).lane_indices == (1,)


def _repeats_of_a1(*changes: Callable[[Project], object]) -> model.Batch:
    """The sample with lane 4 (1-based) included as a technical repeat of lane 3:
    10 µM has one sample, a1, over lanes 3 and 4."""
    return _batch(_lane(3, included=True, sample="a1"), *changes)


def test_a_repeat_measured_in_one_lane_is_a_detected_replicate_on_the_chart():
    # β-catenin not detected in lane 3 but measured in lane 4: a1 has a value.
    chart = compute_results(_repeats_of_a1(_undetected("prot-7", 2))).series[0].chart
    low, coverage = chart.bars[1], chart.coverage[1]
    assert low.label == "10 µM"
    assert (low.n, low.lane_indices, low.not_detected_lanes) == (1, [3], [])
    assert low.mean is not None  # a bar: nothing of it was below the detection limit
    assert (coverage.not_detected, coverage.left_out) == (0, "fewer_than_2")


def test_a_repeat_with_no_value_and_one_lane_not_detected_is_not_detected():
    # Neither lane of a1 has a β-catenin value; lane 3 has a not-detected record.
    batch = _repeats_of_a1(_no_bands("prot-7", 3), _undetected("prot-7", 2))
    chart = compute_results(batch).series[0].chart
    low, coverage = chart.bars[1], chart.coverage[1]
    assert (low.n, low.points, low.not_detected_lanes) == (0, [], [2])
    assert (coverage.not_detected, coverage.left_out) == (1, "not_detected")


def test_a_reference_the_target_was_not_detected_in_says_so():
    not_detected = (
        "'β-catenin' / 'α-tubulin': 'β-catenin' was not detected in the reference"
        " condition 'vehicle' (below the detection limit): no fold-change can be formed"
    )
    usual = (
        "'β-catenin' / 'α-tubulin': control condition 'vehicle' has no value in any included lane"
    )
    both = _batch(_lane(3, included=True), _no_bands("prot-7", 0, 1), _undetected("prot-7", 0, 1))
    res = compute_results(both)
    notice = _one(res, NoticeCode.REFERENCE_UNUSABLE)
    assert notice.message == not_detected
    assert (notice.protein_ids, notice.conditions) == (("prot-7", "prot-8"), ("vehicle",))
    assert res.series[0].chart is None
    # One reference lane not detected and one not measured: the usual wording.
    one = _batch(_lane(3, included=True), _no_bands("prot-7", 0, 1), _undetected("prot-7", 0))
    assert _one(compute_results(one), NoticeCode.REFERENCE_UNUSABLE).message == usual
    # Only included lanes count: excluded lane 0 keeps its band, included lane 1 is n.d.
    excluded = _batch(
        _lane(3, included=True),
        _lane(0, included=False),
        _no_bands("prot-7", 1),
        _undetected("prot-7", 1),
    )
    assert _one(compute_results(excluded), NoticeCode.REFERENCE_UNUSABLE).message == not_detected


def test_a_reference_the_loading_control_was_not_detected_in_names_it():
    # β-catenin is measured in both reference lanes; α-tubulin, its loading control,
    # is not. Lane 3 is included so the series keeps a value elsewhere.
    loading = _batch(
        _lane(3, included=True), _no_bands("prot-8", 0, 1), _undetected("prot-8", 0, 1)
    )
    notice = _one(compute_results(loading), NoticeCode.REFERENCE_UNUSABLE)
    assert notice.message == (
        "'β-catenin' / 'α-tubulin': 'α-tubulin' was not detected in the reference"
        " condition 'vehicle' (below the detection limit): no fold-change can be formed"
    )
    assert (notice.protein_ids, notice.conditions) == (("prot-7", "prot-8"), ("vehicle",))
    both = _batch(
        _lane(3, included=True),
        _no_bands("prot-7", 0, 1),
        _undetected("prot-7", 0, 1),
        _no_bands("prot-8", 0, 1),
        _undetected("prot-8", 0, 1),
    )
    assert _one(compute_results(both), NoticeCode.REFERENCE_UNUSABLE).message == (
        "'β-catenin' / 'α-tubulin': 'β-catenin' and 'α-tubulin' were not detected in the"
        " reference condition 'vehicle' (below the detection limit): no fold-change can be"
        " formed"
    )
    # Each lost one reference lane: neither is missing from the whole condition.
    mixed = _batch(
        _lane(3, included=True),
        _no_bands("prot-7", 0),
        _undetected("prot-7", 0),
        _no_bands("prot-8", 1),
        _undetected("prot-8", 1),
    )
    assert _one(compute_results(mixed), NoticeCode.REFERENCE_UNUSABLE).message == (
        "'β-catenin' / 'α-tubulin': control condition 'vehicle' has no value in any included lane"
    )


def test_extra_bands_notice_is_about_boxes_not_records():
    # Records for a second expected band: nothing is quantified from them, and no
    # box of theirs is being ignored (#58 decides what they mean).
    records = _batch(_expected_bands("prot-9", 2), _undetected("prot-9", 0, 2, band_index=1))
    res = compute_results(records)
    assert NoticeCode.EXTRA_BANDS_IGNORED not in _codes(res)
    assert NoticeCode.BELOW_DETECTION not in _codes(res)  # band index 0 only


def test_compute_takes_the_detection_state_from_lane_detected(monkeypatch):
    # One join serves the results, as it will the chart counts and the export.
    batch = _batch(_undetected("prot-7", 2), _undetected("prot-9", 3))
    seen: list[model.Batch] = []

    def spy(b: model.Batch) -> dict[str, list[bool | None]]:
        seen.append(b)
        return lane_detected(b)

    monkeypatch.setattr(results, "lane_detected", spy)
    res = compute_results(batch)
    assert len(seen) == 2  # this set and the all-lanes set (lane 3 is excluded)
    assert seen[0] is batch
    assert [column.detected for column in res.proteins] == list(lane_detected(batch).values())


# --- #52 PR 6b: the statistics setting, its notices, and replicates not detected ---


def test_the_statistics_setting_flows_into_every_chart():
    batch = _baseline_batch(REFERENCE)  # 2, 2 and 2 replicates here, 2, 3 and 2 in all lanes
    auto = compute_results(batch)
    assert auto.statistics == StatisticsSetting() == auto.all_lanes.statistics
    assert auto.series[0].chart.test.id == "dunnett"
    welch = compute_results(batch, statistics={"family": "welch", "scale": "linear"})
    assert welch.statistics == StatisticsSetting(family="welch", scale="linear")
    for one_set in (welch, welch.all_lanes):
        test = one_set.series[0].chart.test
        assert (test.id, test.scale) == ("welch_t_holm", "linear")
        assert test.chosen == {"family": "user", "comparisons": "auto", "scale": "user"}
    assert welch == compute_results(batch, statistics=welch.statistics)  # raw strings or the model


def test_an_unknown_statistics_setting_is_a_value_error():
    with pytest.raises(ValueError, match="invalid statistics setting"):
        compute_results(make_project().batch, statistics={"family": "t-test"})


def test_each_set_states_its_own_test():
    # Excluding lane 6 leaves 10 µM two replicates of three: equal n here, unequal in all lanes.
    res = compute_results(_baseline_batch(REFERENCE))
    applied, every = res.series[0].chart, res.all_lanes.series[0].chart
    assert applied.test.id == "dunnett"
    assert every.test.id == "welch_t_holm"
    for chart, name in ((applied, "Dunnett's test"), (every, "Welch's t-tests (Holm)")):
        # Each comparison's p, the legend's only numbers: there is no omnibus p.
        each = ", ".join(f"{p.group_b!r} {p_text(p.p_value)}" for p in chart.test.pairwise)
        assert [p.group_b for p in chart.test.pairwise] == [LOW, HIGH]
        assert chart.statement[1] == f"{name} on log values, each condition vs 'vehicle': {each}"


def test_a_chart_is_tested_through_chart_test(monkeypatch):
    calls = []
    real = results.chart_test

    def spy(shown, *, setting, kind, reference):
        calls.append((set(shown), setting, kind, reference))
        return real(shown, setting=setting, kind=kind, reference=reference)

    monkeypatch.setattr(results, "chart_test", spy)
    setting = StatisticsSetting(comparisons="all_pairs")
    compute_results(_baseline_batch(REFERENCE), statistics=setting)
    everything = {REFERENCE, LOW, HIGH}
    assert calls == [(everything, setting, ValueKind.FOLD_CHANGE, REFERENCE)] * 2  # both sets


def test_chart_test_is_the_core_test_of_the_value_kind():
    groups = {REFERENCE: [1.0, 1.1], LOW: [2.0, 2.2], HIGH: [0.5, 0.6]}
    ratio = results.chart_test(
        groups, setting=StatisticsSetting(), kind=ValueKind.FOLD_CHANGE, reference=REFERENCE
    )
    assert (ratio.test, ratio.plan.scale) == ("dunnett", "log")
    raw = results.chart_test(
        groups, setting=StatisticsSetting(), kind=ValueKind.RAW, reference=None
    )
    assert (raw.test, raw.plan.scale) == ("anova_tukey", "linear")


def _baseline_without(*lanes: int) -> model.Batch:
    baseline = _baseline_batch(REFERENCE)
    return baseline.model_copy(
        update={
            "lanes": [
                lane.model_copy(update={"included": False}) if lane.index in lanes else lane
                for lane in baseline.lanes
            ]
        }
    )


def test_conditions_left_out_of_a_test_are_a_warning():
    res = compute_results(_baseline_without(5, 7))  # 50 µM keeps one replicate
    notice = _one(res, NoticeCode.CONDITIONS_NOT_TESTED)
    assert notice.level is Level.WARNING
    assert notice.message == "'β-catenin' / 'α-tubulin': '50 µM' (n = 1) is not in the test"
    assert (notice.protein_ids, notice.conditions) == (("prot-3", "prot-4"), (HIGH,))
    assert NoticeCode.CONDITIONS_NOT_TESTED not in _codes(res.all_lanes)  # all tested there


def test_each_set_keeps_the_test_notices_of_its_own_charts():
    # 50 µM loses its lane-8 β-catenin box: one replicate in both sets, so both
    # charts leave it out. Shared notices are kept only in the first set, but a
    # test notice is about one set's chart: the all-lanes set keeps its own.
    batch = _baseline_batch(REFERENCE).model_copy(deep=True)
    beta = batch.find_protein("prot-3")
    beta.bands = [band for band in beta.bands if band.lane_index != 7]
    res = compute_results(batch)
    assert res.all_lanes is not None
    for one in (res, res.all_lanes):
        notice = _one(one, NoticeCode.CONDITIONS_NOT_TESTED)
        assert notice.message == "'β-catenin' / 'α-tubulin': '50 µM' (n = 1) is not in the test"
        assert one.series[0].chart.coverage[-1].left_out == "fewer_than_2"


def test_a_notice_about_one_series_names_its_target_then_its_loading_control():
    # Cyclin D1, a second target over α-tubulin, loses its lane-8 box: its chart
    # leaves 50 µM out of the test, β-catenin's does not. The page shows such a
    # notice under the series these ids name, not under every chart of α-tubulin.
    doc = _baseline_batch(REFERENCE).model_dump()
    beta = doc["proteins"][0]
    cyclin = {**beta, "id": "prot-5", "name": "Cyclin D1"}
    cyclin["bands"] = [{**band, "id": f"band-{50 + i}"} for i, band in enumerate(beta["bands"][:7])]
    res = compute_results(
        model.Batch.model_validate({**doc, "proteins": [*doc["proteins"], cyclin]})
    )
    assert [(s.target_id, s.loading_id) for s in res.series] == [
        ("prot-3", "prot-4"),
        ("prot-5", "prot-4"),
    ]
    notice = _one(res, NoticeCode.CONDITIONS_NOT_TESTED)
    assert notice.protein_ids == ("prot-5", "prot-4")
    assert results.TEST_NOTICE_CODES <= results.SERIES_NOTICE_CODES


def test_a_choice_that_cannot_apply_is_a_warning():
    res = compute_results(_baseline_batch(None), statistics={"comparisons": "vs_reference"})
    notice = _one(res, NoticeCode.TEST_NOT_APPLICABLE)
    assert notice.level is Level.WARNING
    assert notice.message == (
        "'β-catenin' / 'α-tubulin': no test: comparisons with the reference were chosen,"
        " but no reference is set"
    )
    assert res.series[0].chart.test is None


def test_a_ratio_tested_on_the_linear_scale_is_told():
    batch = _baseline_batch(REFERENCE).model_copy(deep=True)
    batch.find_band("band-8")[1].net = 0.0  # 10 µM, lane 4: a fold-change of 0
    res = compute_results(batch)
    notice = _one(res, NoticeCode.LOG_SCALE_UNAVAILABLE)
    assert notice.level is Level.INFO
    assert notice.message == (
        "'β-catenin' / 'α-tubulin' is tested on linear values:"
        " '10 µM' has a value of 0 or below, which has no log"
    )
    assert notice.conditions == (LOW,)
    assert res.series[0].chart.test.scale == "linear"
    # An explicit linear scale is the user's choice, not a fallback.
    linear = compute_results(batch, statistics={"scale": "linear"})
    assert NoticeCode.LOG_SCALE_UNAVAILABLE not in _codes(linear)


def test_a_rank_test_that_cannot_reach_significance_is_a_warning():
    res = compute_results(_baseline_batch(REFERENCE), statistics={"family": "rank"})
    notice = _one(res, NoticeCode.RANK_TEST_CANNOT_REACH_ALPHA)
    assert notice.level is Level.WARNING
    assert notice.message.startswith(
        "'β-catenin' / 'α-tubulin': Dunn's test (Holm): with these n no p can be below"
    )
    assert res.series[0].chart.test.id == "dunn_holm"


def test_a_condition_the_target_was_not_detected_in_keeps_its_slot():
    # β-catenin not detected in lane 3 (10 µM, a1); lane 4 (a2) is excluded.
    res = compute_results(_batch(_no_bands("prot-7", 2), _undetected("prot-7", 2)))
    chart = res.series[0].chart
    assert [bar.label for bar in chart.bars] == ["vehicle", "10 µM"]
    low = chart.bars[1]
    assert (low.mean, low.points, low.not_detected_lanes) == (None, [], [2])
    assert chart.coverage[1].left_out == "not_detected"
    assert chart.statement[1:] == [
        "No test: fewer than 2 conditions to test; not tested: '10 µM' (1 of 1 not detected)"
    ]
    # In all lanes a2 was measured: 10 µM is partly detected, and still draws no bar.
    every_series = res.all_lanes.series[0]
    every = every_series.chart
    assert every.bars[1].points == every_series.groups["10 µM"]
    assert (every.bars[1].mean, every.bars[1].not_detected_lanes) == (None, [2])
    assert every.coverage[1].replicates == 2
    svg = render_svg(chart).decode()
    assert 'id="nd-1-0"' in svg and 'id="slot-1"' in svg


def test_a_condition_without_any_value_keeps_its_slot_and_is_named():
    res = compute_results(_batch())  # 10 µM's only included lane has no β-catenin box
    chart = res.series[0].chart
    assert [(bar.label, bar.n) for bar in chart.bars] == [("vehicle", 2), ("10 µM", 0)]
    assert chart.coverage[1].left_out == "no_value"
    assert chart.test_note == (
        "no test: fewer than 2 conditions to test; not tested: '10 µM' (no value)"
    )


def test_a_replicate_the_loading_control_misses_has_no_value_but_counts():
    # α-tubulin not detected in lane 2 (vehicle, v2): β-catenin / α-tubulin has no
    # value there, a replicate with no value, not one of the target's not detected.
    res = compute_results(_batch(_no_bands("prot-8", 1), _undetected("prot-8", 1)))
    chart = res.series[0].chart
    assert chart.bars[0].label == "vehicle" and chart.bars[0].not_detected_lanes == []
    assert (chart.coverage[0].n, chart.coverage[0].replicates) == (1, 2)
    assert chart.coverage[0].left_out == "fewer_than_2"


def _with_a_ladder(batch: model.Batch, *, included: bool) -> model.Batch:
    """``batch`` with one more lane, ``Ladder``: no box and no record of any protein."""
    ladder = Lane(index=len(batch.lanes), label="Ladder", included=included)
    return batch.model_copy(update={"lanes": [*batch.lanes, ladder]})


@pytest.mark.parametrize("included", [False, True])
def test_a_lane_holding_nothing_for_a_series_changes_none_of_its_charts(included):
    # Excluded, the ladder is in the all-lanes set only (lane 5 is excluded and
    # holds values); included, in both. It holds nothing for β-catenin /
    # α-tubulin, so it is no replicate and no condition: no place on a chart, no
    # condition left out of a test, no notice.
    plain = compute_results(_baseline_batch(REFERENCE))
    res = compute_results(_with_a_ladder(_baseline_batch(REFERENCE), included=included))
    assert res.all_lanes is not None
    for one, without in ((res, plain), (res.all_lanes, plain.all_lanes)):
        assert [s.chart for s in one.series] == [s.chart for s in without.series]
        assert one.notices == without.notices


def test_an_empty_lane_the_user_excluded_is_not_charted_in_all_lanes():
    # Lane 0 is relabelled "Ladder", emptied and excluded; lane 3, excluded,
    # holds values, so the all-lanes set includes the ladder.
    def ladder_in_lane_0(draft: Project) -> None:
        for protein in draft.batch.proteins:
            protein.bands = [b for b in protein.bands if b.lane_index != 0]
        draft.batch.lanes[0].label = "Ladder"
        draft.batch.lanes[0].included = False

    res = compute_results(_batch(ladder_in_lane_0))
    for one in (res, res.all_lanes):
        chart = one.series[0].chart
        assert [bar.label for bar in chart.bars] == ["vehicle", "10 µM"]
        assert not any("Ladder" in line for line in chart.statement)
        assert not any("Ladder" in n.conditions or "Ladder" in n.message for n in one.notices)
    # A condition keeps its place while one of its lanes holds something for the
    # series: 10 µM's lane 2 has an α-tubulin band, and no β-catenin one.
    assert res.series[0].chart.coverage[1].left_out == "no_value"


def test_a_lane_holding_nothing_for_a_series_is_no_replicate():
    # Lane 1 (vehicle, v2) keeps only its GAPDH band: nothing of β-catenin /
    # α-tubulin, so no replicate of vehicle on that chart.
    batch = _batch(*(_no_bands(p, 1) for p in ("prot-7", "prot-8")))
    chart = compute_results(batch).series[0].chart
    assert chart.bars[0].label == "vehicle"
    assert (chart.coverage[0].n, chart.coverage[0].replicates) == (1, 1)


# --- #112: bands the over-exposure check could not run on ---

# Import warnings as an image holds them; the notice reads their codes only.
LOSSY = {"code": "lossy_format", "message": "JPEG-type compression can change pixel values."}
COLOR = {"code": "color_channels_differ", "message": "The channels were averaged."}
CMYK = {"code": "cmyk_converted", "message": "The CMYK colours were converted."}
UNKNOWN = {"code": "unknown_bit_depth", "message": "The pixel type has no fixed range."}
# The sample's bands by protein: β-catenin in lanes 0, 1 and 3 (excluded),
# α-tubulin in lanes 0-3, GAPDH in lanes 0 and 1.
BETA_BANDS = ("band-10", "band-11", "band-12")
EVERY_BAND = (*BETA_BANDS, "band-13", "band-14", "band-15", "band-16", "band-17", "band-18")
UNDER = "if it is over-exposed there, its net is an under-estimate"


def _image(image_id: str, bit_depth: int | None, *warnings: dict) -> Callable[[Project], None]:
    def edit(draft: Project) -> None:
        image = draft.batch.find_image(image_id)
        image.bit_depth = bit_depth
        image.import_warnings = [model.ImageWarning(**warning) for warning in warnings]

    return edit


def _checked(flag: bool | None, *band_ids: str) -> Callable[[Project], None]:
    def edit(draft: Project) -> None:
        for band_id in band_ids:
            draft.batch.find_band(band_id)[1].clipped = flag

    return edit


def _unchecked(res: results.Results) -> list[results.Notice]:
    return _all(res, NoticeCode.CLIPPING_NOT_CHECKED)


@pytest.mark.parametrize(
    ("bit_depth", "warnings", "because"),
    [
        (8, [LOSSY], "lossy (JPEG-type) compression"),
        (16, [COLOR], "color channels averaged into gray"),
        (None, [UNKNOWN], "an unknown bit depth"),
        (None, [], "an unknown bit depth"),  # not recorded: no limit to check against
        (8, [LOSSY, COLOR], "lossy (JPEG-type) compression and color channels averaged into gray"),
        (None, [LOSSY], "lossy (JPEG-type) compression and an unknown bit depth"),
        (
            8,
            [COLOR, LOSSY, CMYK],
            "lossy (JPEG-type) compression and CMYK converted to RGB and color channels"
            " averaged into gray",
        ),
    ],
)
def test_a_band_not_checked_for_over_exposure_is_reported_with_the_reason(
    bit_depth, warnings, because
):
    # Every band checked but β-catenin's in lanes 0 and 3 (excluded), on an
    # image the check cannot trust; where its range is known, they were
    # assessed for pixels near the limit instead (#112).
    res = compute_results(
        _batch(
            _checked(False, *EVERY_BAND),
            _checked(None, "band-10", "band-12"),
            _possibly(None if bit_depth is None else False, "band-10", "band-12"),
            _image("img-2", bit_depth, *warnings),
        )
    )
    assert _unchecked(res) == [
        results.Notice(
            code=NoticeCode.CLIPPING_NOT_CHECKED,
            level=Level.WARNING,
            message=(
                f"'β-catenin' was not checked for over-exposure in lane 1: its image has"
                f" {because}, so saturated pixels cannot be counted; {UNDER}"
            ),
            protein_ids=("prot-7",),
            lane_indices=(0,),
        )
    ]
    # The excluded lane is reported with every lane included.
    [every] = _unchecked(res.all_lanes)
    assert (every.protein_ids, every.lane_indices) == (("prot-7",), (0, 3))
    assert "in lanes 1, 4:" in every.message


def test_a_loading_control_not_checked_names_the_values_it_biases():
    res = compute_results(
        _batch(
            _checked(False, *BETA_BANDS, "band-17", "band-18"),
            _possibly(False, "band-13", "band-14", "band-15", "band-16"),
            _image("img-6", 8, LOSSY),
        )
    )
    [notice] = _unchecked(res)
    assert (notice.protein_ids, notice.lane_indices) == (("prot-8",), (0, 1, 2))
    assert notice.message == (
        "'α-tubulin' was not checked for over-exposure in lanes 1, 2, 3: its image has lossy"
        f" (JPEG-type) compression, so saturated pixels cannot be counted; {UNDER}, which"
        " biases every value normalized to it"
    )


REQUANTIFY = (
    "its boxes were measured before Proteia looked for pixels near the detector limit:"
    " requantify to look for them"
)


def test_bands_measured_before_the_possible_check_are_told_to_requantify():
    # A project saved before #112: β-catenin's bands on an 8-bit JPEG hold
    # neither flag. Its not-checked notice says how to have them assessed.
    def before(*edits: Callable[[Project], None]) -> model.Batch:
        return _batch(
            _checked(False, *EVERY_BAND),
            _checked(None, *BETA_BANDS),
            _image("img-2", 8, LOSSY),
            *edits,
        )

    [notice] = _unchecked(compute_results(before()))
    assert notice.message == (
        "'β-catenin' was not checked for over-exposure in lanes 1, 2: its image has lossy"
        f" (JPEG-type) compression, so saturated pixels cannot be counted; {UNDER};"
        f" {REQUANTIFY}"
    )
    # Not once they are assessed (flagged or not), nor on an image without a
    # range to assess against or with a limit the exact check trusts.
    for edits in (
        [_possibly(False, *BETA_BANDS)],
        [_possibly(True, "band-10"), _possibly(False, "band-11", "band-12")],
        [_image("img-2", None, LOSSY)],
        [_image("img-2", 16)],
    ):
        res = compute_results(before(*edits))
        assert not any(REQUANTIFY in n.message for n in _unchecked(res)), edits
        assert not any(REQUANTIFY in n.message for n in _unchecked(res.all_lanes)), edits
    # Only the lanes a set includes count: a band not assessed in the excluded
    # lane 3 is named in the all-lanes set alone.
    res = compute_results(before(_possibly(False, "band-10", "band-11")))
    [notice] = _unchecked(res)
    assert REQUANTIFY not in notice.message
    [every] = _unchecked(res.all_lanes)
    assert every.message.endswith(f"; {REQUANTIFY}")


def test_only_the_unchecked_bands_of_included_lanes_are_reported():
    # β-catenin: lane 0 not checked, lane 1 checked, lane 2 without a box, lane 3
    # excluded; each protein gets its own notice, in model order.
    res = compute_results(
        _batch(
            _checked(False, "band-11", "band-12", "band-15"),
            _checked(True, "band-14"),
            _image("img-2", 16, COLOR),
            _image("img-6", 8, LOSSY),
            _image("img-4", None, UNKNOWN),
        )
    )
    assert [(n.protein_ids, n.lane_indices, n.level) for n in _unchecked(res)] == [
        (("prot-7",), (0,), Level.WARNING),
        (("prot-8",), (0,), Level.WARNING),
        (("prot-9",), (0, 1), Level.WARNING),
    ]
    # A band checked and over-exposed has its own notice, not this one.
    assert _one(res, NoticeCode.CLIPPED).lane_indices == (1,)
    assert [n.lane_indices for n in _unchecked(res.all_lanes)] == [(0, 3)]  # α-tubulin's


def test_no_notice_when_every_band_is_checked():
    for flag in (False, True):
        res = compute_results(_batch(_checked(flag, *EVERY_BAND)))
        assert _unchecked(res) == [] and _unchecked(res.all_lanes) == []
    # The stored flag decides, not the image: a flag checked on a JPEG (a model
    # the operations never write) is not reported as unchecked.
    res = compute_results(_batch(_checked(False, *EVERY_BAND), _image("img-2", 8, LOSSY)))
    assert _unchecked(res) == []


def test_an_unchecked_band_without_a_reason_is_still_reported():
    # A 16-bit image without warnings has a limit the check trusts, so the
    # operations always check its bands; a flag missing anyway is reported, with
    # no reason made up.
    res = compute_results(_batch(_checked(False, *EVERY_BAND), _checked(None, "band-11")))
    assert _unchecked(res) == [
        results.Notice(
            code=NoticeCode.CLIPPING_NOT_CHECKED,
            level=Level.WARNING,
            message=f"'β-catenin' was not checked for over-exposure in lane 2; {UNDER}",
            protein_ids=("prot-7",),
            lane_indices=(1,),
        )
    ]


def test_every_warning_that_turns_the_clipping_check_off_has_a_reason():
    # results names the reason from the warning codes without importing
    # imaging, so a code that clipping_depth starts distrusting must be named
    # here too; otherwise its notice silently loses the reason.
    distrusted = {
        code
        for code, message in imaging.WARNINGS.items()
        if imaging.clipping_depth(8, [model.ImageWarning(code=code, message=message)]) is None
    }
    assert distrusted == imaging.UNTRUSTED_WARNINGS
    assert results._UNCHECKED_WARNINGS.keys() == imaging.UNTRUSTED_WARNINGS


def test_the_sample_projects_jpeg_loading_control_is_reported():
    # α-tubulin is on a JPEG with no recorded bit depth; nothing else in the
    # sample is unchecked once β-catenin's and GAPDH's bands are checked.
    res = compute_results(_batch(_checked(False, *BETA_BANDS, "band-17", "band-18")))
    # Its strip's two-point calibration (#58, D7) is the same in both sets.
    assert _codes(res) == [NoticeCode.CLIPPING_NOT_CHECKED, NoticeCode.CALIBRATION_TWO_POINTS]
    assert _codes(res.all_lanes) == [NoticeCode.CLIPPING_NOT_CHECKED]
    assert "lossy (JPEG-type) compression and an unknown bit depth" in res.notices[0].message


# --- #112: possibly over-exposed, where the exact check cannot run ---


def _possibly(flag: bool | None, *band_ids: str) -> Callable[[Project], None]:
    """The heuristic's flag on bands whose exact check did not run."""

    def edit(draft: Project) -> None:
        for band_id in band_ids:
            band = draft.batch.find_band(band_id)[1]
            band.clipped = None
            band.possibly_clipped = flag

    return edit


def _possible(res: results.Results) -> list[results.Notice]:
    return _all(res, NoticeCode.POSSIBLY_CLIPPED)


NEAR_2 = "5 or more pixels within 2 grey levels of the detector limit"


def test_a_band_possibly_over_exposed_is_reported_with_its_lanes():
    # β-catenin on an 8-bit JPEG: lane 0 flagged, lane 1 not, lane 3 (excluded)
    # flagged; every other band checked.
    res = compute_results(
        _batch(
            _checked(False, *EVERY_BAND),
            _possibly(True, "band-10", "band-12"),
            _possibly(False, "band-11"),
            _image("img-2", 8, LOSSY),
        )
    )
    assert _possible(res) == [
        results.Notice(
            code=NoticeCode.POSSIBLY_CLIPPED,
            level=Level.WARNING,
            message=(
                f"'β-catenin' is possibly over-exposed in lane 1: its box holds {NEAR_2},"
                " and its image has lossy (JPEG-type) compression, so saturation cannot be"
                f" confirmed; {UNDER}; check the imager's original capture"
            ),
            protein_ids=("prot-7",),
            lane_indices=(0,),
        )
    ]
    # The column carries the flags, lane by lane (lane 2 has no box).
    [beta] = [column for column in res.proteins if column.protein_id == "prot-7"]
    assert beta.possibly_clipped == [True, False, None, True]
    assert beta.clipped == [None, None, None, None]
    # The not-checked notice still names every unchecked lane, flagged or not.
    assert _one(res, NoticeCode.CLIPPING_NOT_CHECKED).lane_indices == (0, 1)
    # With every lane included, the excluded lane is reported too.
    [every] = _possible(res.all_lanes)
    assert (every.protein_ids, every.lane_indices) == (("prot-7",), (0, 3))
    assert every.message.startswith(
        f"'β-catenin' is possibly over-exposed in lanes 1, 4: each of those boxes holds {NEAR_2},"
    )


def test_a_possibly_over_exposed_loading_control_names_the_values_it_biases():
    res = compute_results(
        _batch(
            _checked(False, *EVERY_BAND),
            _possibly(True, "band-13", "band-14"),
            _possibly(False, "band-15", "band-16"),
            _image("img-6", 8, COLOR),
        )
    )
    [notice] = _possible(res)
    assert (notice.protein_ids, notice.lane_indices) == (("prot-8",), (0, 1))
    assert notice.message == (
        f"'α-tubulin' is possibly over-exposed in lanes 1, 2: each of those boxes holds {NEAR_2},"
        " and its image has color channels averaged into gray, so saturation cannot be"
        f" confirmed; {UNDER}, which biases every value normalized to it; check the imager's"
        " original capture"
    )
    # The target normalized to it gets no notice of its own: the chart card of
    # the series shows its loading control's (a notice about either protein).
    assert {n.protein_ids for n in _possible(res)} == {("prot-8",)}


@pytest.mark.parametrize(
    ("bit_depth", "warnings", "near", "because"),
    [
        (16, [LOSSY], "514 grey levels (2 on an 8-bit scale)", "lossy (JPEG-type) compression"),
        (8, [CMYK], "2 grey levels", "CMYK converted to RGB"),
        (
            8,
            [LOSSY, COLOR],
            "2 grey levels",
            "lossy (JPEG-type) compression and color channels averaged into gray",
        ),
    ],
)
def test_the_possible_notice_states_the_range_and_the_reason(bit_depth, warnings, near, because):
    res = compute_results(
        _batch(
            _checked(False, *EVERY_BAND),
            _possibly(True, "band-11"),
            _image("img-2", bit_depth, *warnings),
        )
    )
    [notice] = _possible(res)
    assert notice.message == (
        f"'β-catenin' is possibly over-exposed in lane 2: its box holds 5 or more pixels"
        f" within {near} of the detector limit, and its image has {because}, so saturation"
        f" cannot be confirmed; {UNDER}; check the imager's original capture"
    )


def test_a_possible_flag_without_a_reason_is_still_reported():
    # The operations assess only images the exact check distrusts; a flag on
    # another (a hand-edited file: a 16-bit TIFF) is reported with no reason
    # made up.
    res = compute_results(_batch(_checked(False, *EVERY_BAND), _possibly(True, "band-11")))
    [notice] = _possible(res)
    assert notice.message == (
        "'β-catenin' is possibly over-exposed in lane 2: its box holds 5 or more pixels"
        " within 514 grey levels (2 on an 8-bit scale) of the detector limit;"
        f" {UNDER}; check the imager's original capture"
    )


def test_no_possible_notice_without_a_flagged_band():
    for edits in (
        [_checked(False, *EVERY_BAND)],  # every band checked exactly
        [_checked(True, *EVERY_BAND)],  # over-exposed: the clipped notice says so
        [_checked(False, *EVERY_BAND), _possibly(False, *BETA_BANDS), _image("img-2", 8, LOSSY)],
        [_checked(False, *EVERY_BAND), _possibly(None, *BETA_BANDS), _image("img-2", 8, LOSSY)],
    ):
        res = compute_results(_batch(*edits))
        assert _possible(res) == [] and _possible(res.all_lanes) == []
    # The sample itself: its JPEG has no bit depth, so nothing is assessed.
    res = compute_results(_batch())
    assert all(column.possibly_clipped == [None] * 4 for column in res.proteins)
    assert _possible(res) == []
