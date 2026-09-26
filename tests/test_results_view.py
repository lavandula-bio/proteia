# SPDX-License-Identifier: Apache-2.0
"""The results as the web UI receives them (#52): a translation of
:class:`~proteia.core.results.Results` into strict JSON, with nothing computed."""

from __future__ import annotations

import json
import math

import pytest

from conftest import make_project, make_project_with_undetected
from proteia.core.analyze import ReduceMethod, Tier
from proteia.core.model import Project, Role
from proteia.core.plotspec import Bar, ErrorType, PlotSpec, Significance, ValueKind
from proteia.core.results import (
    LaneRow,
    Level,
    Notice,
    NoticeCode,
    ProteinColumn,
    Results,
    SeriesResult,
    compute_results,
)
from proteia.web.results_view import results_payload

# Lane order is neither numeric ("5" before "10") nor text order ("10" before "5"),
# and a JS object would put "5" and "10" first, in numeric order.
CONDITIONS = ["vehicle", "10", "10", "5", "5"]


def _not_json(constant: str) -> float:
    raise ValueError(f"{constant} is not JSON")


def strict(payload: dict) -> dict:
    """``payload`` through strict JSON: NaN or inf anywhere raises."""
    return json.loads(json.dumps(payload, allow_nan=False), parse_constant=_not_json)


def _chart(**fields) -> PlotSpec:
    return PlotSpec(
        title="β-catenin / α-tubulin",
        value_kind=ValueKind.LOADING_NORMALIZED,
        error_type=ErrorType.SD,
        y_label="Normalized signal (target / loading)",
        bars=[
            Bar(label=label, mean=1.0, error=0.1, n=2, points=[0.9, 1.1], lane_indices=[i, i + 1])
            for label, i in (("vehicle", 0), ("10", 1), ("5", 3))
        ],
        **fields,
    )


def _series(**fields) -> SeriesResult:
    values = {
        "target_id": "prot-2",
        "target": "β-catenin",
        "loading_id": "prot-1",
        "loading": "α-tubulin",
        "normalized": [1.0, 0.5, 0.75, 2.0, 2.5],
        "value_kind": ValueKind.LOADING_NORMALIZED,
        "baseline": None,
        "fold_change": None,
        "groups": {"vehicle": [1.0], "10": [0.5, 0.75], "5": [2.0, 2.5]},
        "averaged": [("10", "s µ"), ("5", "s2")],
        "chart": _chart(),
        "undetected": {"10": [2], "5": [3, 4]},
    }
    return SeriesResult(**{**values, **fields})


def _results(**fields) -> Results:
    values = {
        "lanes": [
            LaneRow(index=i, condition=condition, sample=None, included=True)
            for i, condition in enumerate(CONDITIONS)
        ],
        "tier": Tier.NORMALIZED,
        "reference_condition": None,
        "proteins": [
            ProteinColumn(
                protein_id="prot-1",
                name="α-tubulin",
                role=Role.LOADING_CONTROL,
                image_id="img-1",
                nets=[10.0, 20.0, None, 40.0, 50.0],
                band_ids=["band-3", "band-4", None, "band-5", "band-6"],
                clipped=[False, True, None, None, False],
                detected=[True, True, False, True, True],
            )
        ],
        "series": [_series()],
        "notices": [],
        "plot_conditions": None,
        "error_type": ErrorType.SD,
        "method": ReduceMethod.MEAN,
    }
    return Results(**{**values, **fields})


def test_groups_keep_the_lane_order_of_numeric_condition_names():
    payload = strict(results_payload(_results(), open_id=1, revision=7))
    (series,) = payload["sets"][0]["series"]
    assert series["groups"] == [
        {"condition": "vehicle", "values": [1.0]},
        {"condition": "10", "values": [0.5, 0.75]},
        {"condition": "5", "values": [2.0, 2.5]},
    ]
    assert series["undetected"] == [
        {"condition": "10", "lanes": [2]},
        {"condition": "5", "lanes": [3, 4]},
    ]
    assert series["averaged"] == [
        {"condition": "10", "sample": "s µ"},
        {"condition": "5", "sample": "s2"},
    ]


def test_the_payload_is_a_translation_of_one_result_set():
    results = _results()
    payload = strict(results_payload(results, open_id=3, revision=12))
    assert payload["open_id"] == 3 and payload["revision"] == 12
    assert payload["settings"] == {"error_type": "SD", "plot_conditions": None, "method": "mean"}
    assert payload["reference_condition"] is None
    assert payload["lanes"] == [
        {"index": i, "condition": condition, "sample": None, "included": True}
        for i, condition in enumerate(CONDITIONS)
    ]
    assert payload["proteins"] == [
        {
            "protein_id": "prot-1",
            "name": "α-tubulin",
            "role": "loading control",
            "image_id": "img-1",
            "nets": [10.0, 20.0, None, 40.0, 50.0],
            "band_ids": ["band-3", "band-4", None, "band-5", "band-6"],
            "clipped": [False, True, None, None, False],
            "detected": [True, True, False, True, True],
        }
    ]
    (only,) = payload["sets"]
    assert {key: only[key] for key in ("id", "label", "excluded_lanes", "tier", "notices")} == {
        "id": "applied",
        "label": None,
        "excluded_lanes": [],
        "tier": "normalized",
        "notices": [],
    }
    (series,) = only["series"]
    assert series["chart_url"] is None
    assert {key: series[key] for key in ("target_id", "loading_id", "value_kind", "baseline")} == {
        "target_id": "prot-2",
        "loading_id": "prot-1",
        "value_kind": "loading_normalized",
        "baseline": None,
    }
    assert series["normalized"] == [1.0, 0.5, 0.75, 2.0, 2.5]
    assert series["fold_change"] is None
    assert series["chart"] == json.loads(results.series[0].chart.model_dump_json())


def test_comparisons_carry_their_stars():
    comparisons = [
        Significance(group_a="vehicle", group_b="10", p_value=0.0004),
        Significance(group_a="vehicle", group_b="5", p_value=0.004),
        Significance(group_a="10", group_b="5", p_value=0.04),
    ]
    chart = _chart(comparisons=comparisons, test_name="One-way ANOVA", test_p=0.001)
    payload = strict(
        results_payload(_results(series=[_series(chart=chart)]), open_id=1, revision=1)
    )
    (series,) = payload["sets"][0]["series"]
    assert series["chart"]["comparisons"] == [
        {"group_a": "vehicle", "group_b": "10", "p_value": 0.0004, "stars": "***"},
        {"group_a": "vehicle", "group_b": "5", "p_value": 0.004, "stars": "**"},
        {"group_a": "10", "group_b": "5", "p_value": 0.04, "stars": "*"},
    ]
    assert (series["chart"]["test_name"], series["chart"]["test_p"]) == ("One-way ANOVA", 0.001)


def test_non_finite_numbers_become_null():
    nan, inf = math.nan, math.inf
    series = _series(
        normalized=[nan, inf, -inf, 1.0, None],
        baseline=nan,
        fold_change=[inf, None, nan, 1.0, 2.0],
        groups={"vehicle": [nan], "10": [inf, 1.0]},
        chart=_chart(test_name="Welch t-test", test_p=nan),
    )
    payload = results_payload(_results(series=[series]), open_id=1, revision=1)
    (out,) = strict(payload)["sets"][0]["series"]
    assert out["normalized"] == [None, None, None, 1.0, None]
    assert out["baseline"] is None
    assert out["fold_change"] == [None, None, None, 1.0, 2.0]
    assert out["groups"] == [
        {"condition": "vehicle", "values": [None]},
        {"condition": "10", "values": [None, 1.0]},
    ]
    assert out["chart"]["test_p"] is None


def test_a_series_without_a_chart_has_neither_chart_nor_url():
    payload = results_payload(_results(series=[_series(chart=None)]), open_id=1, revision=1)
    (series,) = strict(payload)["sets"][0]["series"]
    assert (series["chart"], series["chart_url"]) == (None, None)


def test_a_chart_url_comes_from_the_chart_store_when_one_is_given():
    registered: list[PlotSpec] = []

    def charts(spec: PlotSpec) -> str:
        registered.append(spec)
        return f"/api/charts/{len(registered):032x}.svg"

    results = _results(series=[_series(), _series(target_id="prot-3", chart=None)])
    payload = strict(results_payload(results, open_id=1, revision=1, charts=charts))
    urls = [series["chart_url"] for series in payload["sets"][0]["series"]]
    assert urls == [f"/api/charts/{1:032x}.svg", None]
    assert registered == [results.series[0].chart]


def _relabelled(labels: list[str]) -> Project:
    """The sample project (the lane at index 3 excluded, with values) with these
    lane labels."""
    doc = make_project().model_dump(mode="json")
    for lane, label in zip(doc["batch"]["lanes"], labels, strict=True):
        lane["label"] = label
    return Project.model_validate(doc)


def _translated(series: SeriesResult) -> dict:
    """What the payload holds of ``series``'s groups, averaged samples and chart."""
    dumped = json.loads(series.model_dump_json())
    chart = dumped["chart"]
    if series.chart is not None:
        chart["comparisons"] = [
            {**comparison, "stars": model.stars}
            for model, comparison in zip(
                series.chart.comparisons, chart["comparisons"], strict=True
            )
        ]
    return {
        "groups": [{"condition": c, "values": dumped["groups"][c]} for c in series.groups],
        "averaged": [{"condition": c, "sample": sample} for c, sample in series.averaged],
        "chart": chart,
    }


@pytest.mark.parametrize(
    ("labels", "conditions", "tiers", "notices"),
    [
        # The excluded lane holds 10 µM's only β-catenin value.
        (
            ["vehicle", "vehicle", "10 µM", "10 µM"],
            (["vehicle"], ["vehicle", "10 µM"]),
            ("fold_change", "fold_change"),
            ([], []),
        ),
        # The excluded lane is the reference condition's only lane.
        (
            ["10 µM", "10 µM", "50 µM", "vehicle"],
            (["10 µM"], ["10 µM", "vehicle"]),
            ("normalized", "fold_change"),
            (["reference_all_excluded"], []),
        ),
    ],
    ids=["only value of a condition", "only lane of the reference"],
)
def test_excluded_lanes_with_values_give_two_labelled_sets(labels, conditions, tiers, notices):
    results = compute_results(_relabelled(labels).batch)
    assert results.all_lanes is not None
    payload = strict(results_payload(results, open_id=2, revision=5))
    applied, every = payload["sets"]
    assert (applied["id"], applied["label"], applied["excluded_lanes"]) == (
        "applied",
        "Excluding lane 4",
        [3],
    )
    assert (every["id"], every["label"], every["excluded_lanes"]) == ("all_lanes", "All lanes", [])
    # Each set is its own result set's translation, and the two differ here.
    for result_set, source in ((applied, results), (every, results.all_lanes)):
        assert result_set["tier"] == source.tier.value
        assert result_set["notices"] == [
            json.loads(notice.model_dump_json()) for notice in source.notices
        ]
        assert [s["target_id"] for s in result_set["series"]] == [
            s.target_id for s in source.series
        ]
        for series, model in zip(result_set["series"], source.series, strict=True):
            shown = {key: series[key] for key in ("groups", "averaged", "chart")}
            assert shown == _translated(model)
            assert series["chart"] is None or series["chart"]["subtitle"] == result_set["label"]
    both = (applied, every)
    assert tuple([g["condition"] for g in s["series"][0]["groups"]] for s in both) == conditions
    assert tuple(s["tier"] for s in both) == tiers
    assert tuple([n["code"] for n in s["notices"]] for s in both) == notices
    # The lanes appear once, as the lane table has them.
    assert [lane["included"] for lane in payload["lanes"]] == [True, True, True, False]
    assert payload["reference_condition"] == "vehicle"


def test_the_settings_are_those_the_results_used():
    results = compute_results(
        make_project().batch,
        plot_conditions=["10 µM", " vehicle"],  # one respelled, and not in lane order
        error_type="SEM",
        method="representative",
    )
    payload = strict(results_payload(results, open_id=1, revision=1))
    assert payload["settings"] == {
        "error_type": "SEM",
        "plot_conditions": ["vehicle", "10 µM"],  # the stored labels, in lane order
        "method": "representative",
    }


def test_a_real_computation_is_strict_json_with_detection_states():
    results = compute_results(make_project_with_undetected().batch)
    payload = results_payload(results, open_id=1, revision=1)
    assert strict(payload) == payload
    detected = {p["protein_id"]: p["detected"] for p in payload["proteins"]}
    assert detected["prot-7"] == [True, True, False, True]
    assert detected["prot-9"] == [True, True, False, False]
    notices = [n["code"] for n in payload["sets"][0]["notices"]]
    assert NoticeCode.BELOW_DETECTION.value in notices


def test_notices_keep_their_objects():
    notice = Notice(
        code=NoticeCode.CLIPPED,
        level=Level.WARNING,
        message="'α-tubulin' is over-exposed in lane 2",
        protein_ids=("prot-1",),
        lane_indices=(1,),
    )
    payload = strict(results_payload(_results(notices=[notice]), open_id=1, revision=1))
    assert payload["sets"][0]["notices"] == [
        {
            "code": "clipped",
            "level": "warning",
            "message": "'α-tubulin' is over-exposed in lane 2",
            "protein_ids": ["prot-1"],
            "lane_indices": [1],
            "conditions": [],
        }
    ]
