# SPDX-License-Identifier: Apache-2.0
"""Tests for the single compute step, on the stored model only (no files, no pixels).

Most tests edit the ``conftest`` sample project through ``apply_change``, so every
variant is a valid model: β-catenin (``prot-7``) is the target, α-tubulin
(``prot-8``) and GAPDH (``prot-9``) are loading controls, lane 2 has no β-catenin
band, lane 3 is excluded and ``vehicle`` is the reference. The last tests pin the
compute step to the golden numbers of ``test_regression_baseline``.
"""

import json
from collections.abc import Callable

import pytest

from conftest import make_project
from proteia.core import model, results
from proteia.core.analyze import ReduceMethod, Tier
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
    Role,
    apply_change,
)
from proteia.core.plotspec import ErrorType, ValueKind
from proteia.core.results import Level, NoticeCode, compute_results, lane_nets
from test_regression_baseline import (
    BOX_SIZE,
    CONDITIONS,
    GOLDEN,
    HEIGHT,
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
    assert res.notices == []  # the sample project is clean


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
    assert {bar.label: bar.points for bar in s.chart.bars} == s.groups
    assert all(x.loading_id != "prot-8" for x in res.series)


def test_series_naming_the_first_loading_control_uses_it_throughout():
    batch = make_project().batch  # prot-7 names prot-8
    nets = lane_nets(batch)
    [s] = compute_results(batch).series
    assert (s.loading_id, s.loading) == ("prot-8", "α-tubulin")
    assert s.normalized == _ratio(nets["prot-7"], nets["prot-8"])
    assert s.chart is not None and s.chart.title.endswith("(/α-tubulin)")
    assert {bar.label: bar.points for bar in s.chart.bars} == s.groups


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
    assert [bar.label for bar in gapdh.chart.bars] == ["vehicle"]
    assert gapdh.chart.bars[0].mean == pytest.approx(1.0)


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
            )
        )

    batch = _batch(add_second_band)
    res = compute_results(batch)
    notice = _one(res, NoticeCode.EXTRA_BANDS_IGNORED)
    assert (notice.protein_ids, notice.level) == (("prot-7",), Level.INFO)
    assert res.proteins[0].nets[0] == 4279.740326695199  # the first band's net, not 999
    assert res.proteins[0].band_ids[0] == "band-10"
    assert lane_nets(batch)["prot-7"][0] == 4279.740326695199


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
    assert len(calls) == 2  # two series, one reduction each
    assert all(included == [True, True, True, False] for included in calls)  # the lane table's


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
    image = ImageRef(
        id="img-2",
        file="img-2.tif",
        original_name="β-catenin α-tubulin 10 µM.tif",
        kind=ImageKind.CHEMILUMINESCENCE,
        sha256="0" * 64,  # never read: compute takes no pixels
        width=WIDTH,
        height=HEIGHT,
        polarity=Polarity.DARK_ON_LIGHT,
        background=computed["background"]["dark_on_light"],
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
    """A chart's statistics in the golden file's shape (only brackets for p < 0.05)."""
    [sd], [sem] = res_sd.series, res_sem.series
    assert sd.chart is not None and sem.chart is not None
    return {
        "describe": [
            {"label": a.label, "n": a.n, "mean": a.mean, "sd": a.error, "sem": b.error}
            for a, b in zip(sd.chart.bars, sem.chart.bars, strict=True)
        ],
        "compare": {
            "test": sd.chart.test_name,
            "p_value": sd.chart.test_p,
            "pairwise": [c.model_dump() for c in sd.chart.comparisons],
        },
    }


def _golden_chart_stats(stats: dict) -> dict:
    compared = stats["compare"]
    return {
        "describe": stats["describe"],
        "compare": {
            "test": compared["test"],
            "p_value": compared["p_value"],
            "pairwise": [pw for pw in compared["pairwise"] if pw["p_value"] < 0.05],
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
    assert res.notices == [
        results.Notice(
            code=NoticeCode.TECHNICAL_REPEATS,
            level=Level.INFO,
            message=(
                ("averaged" if method is ReduceMethod.MEAN else "kept one lane of")
                + " 1 sample(s) with technical repeats (repeats do not count as n)"
            ),
            conditions=(REFERENCE,),
        )
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
        expected = _golden_chart_stats(reduced[case])
        _assert_close(_chart_stats(*charts), expected, rel, abs_, case)
