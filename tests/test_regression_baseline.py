# SPDX-License-Identifier: Apache-2.0
"""Regression baseline: the same boxes must give the same numbers.

Runs the chain the app uses today, from pixels to statistics, on a deterministic
synthetic blot with fixed boxes, and compares every output with the golden
values in ``data/regression_baseline.json`` (tolerance recorded in that file).
Box placement is not covered: the boxes are fixed inputs.

A deliberate change to the numbers regenerates the file with

    PROTEIA_UPDATE_BASELINE=1 uv run pytest tests/test_regression_baseline.py

and the pull request that does so says why the numbers moved.
"""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from proteia.core.analyze import (
    Batch,
    ProteinNets,
    ReduceMethod,
    Role,
    compare,
    describe,
    fold_change_lane,
    normalize_batch,
    reduce_samples,
)
from proteia.core.model import Box, BoxSize
from proteia.core.quantify import estimate_background, net_signal

GOLDEN = Path(__file__).parent / "data" / "regression_baseline.json"

# Synthetic blot: two rows (target, loading control) over eight lanes on a light
# membrane with a smooth deterministic texture. Pure arithmetic, no random
# numbers, so every platform builds the same pixels up to floating-point ulps.
HEIGHT, WIDTH = 150, 340
MEMBRANE = 200.0
LANE_X = [30 + 40 * i for i in range(8)]  # band centres
ROW_Y = {"p53": 50, "GAPDH": 105}  # row centres
DARKNESS = {
    "p53": [60.0, 66.0, 54.0, 118.0, 126.0, 40.0, 31.0, 35.0],
    "GAPDH": [100.0, 94.0, 104.0, 98.0, 103.0, 100.0, 96.0, 102.0],
}
CONDITIONS = ["ctl", "ctl", "ctl", "A", "A", "A", "B", "B"]
SAMPLES = ["c1", "c1", "c2", "a1", "a2", "a3", "b1", "b2"]  # c1 is loaded twice
INCLUDED = [True, True, True, True, True, False, True, True]  # a3 is presentation-only
REFERENCE = "ctl"
BOX_SIZE = BoxSize(width=24, height=14)


def _blot() -> np.ndarray:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(float)
    img = MEMBRANE + 2.0 * np.sin(x / 7.0) * np.cos(y / 5.0)
    for name, cy in ROW_Y.items():
        for cx, dark in zip(LANE_X, DARKNESS[name], strict=True):
            img -= dark * np.exp(-(((x - cx) / 7.0) ** 2) - ((y - cy) / 3.5) ** 2)
    return img


def _box(cx: int, cy: int) -> Box:
    return Box(x=cx - BOX_SIZE.width // 2, y=cy - BOX_SIZE.height // 2)


def _nets(image: np.ndarray, background: float, *, dark_on_light: bool) -> dict:
    return {
        name: [
            net_signal(image, _box(cx, cy), BOX_SIZE, background, dark_on_light=dark_on_light)
            for cx in LANE_X
        ]
        for name, cy in ROW_Y.items()
    }


def _stats(groups: dict[str, list[float]]) -> dict:
    result = compare(groups)
    return {
        "describe": [dataclasses.asdict(g) for g in describe(groups)],
        "compare": {
            "test": result.test,
            "p_value": result.p_value,
            "statistic": result.statistic,
            "pairwise": [dataclasses.asdict(p) for p in result.pairwise],
        },
    }


def _compute() -> dict:
    dark = _blot()
    light = 255.0 - dark  # the same blot as a light-on-dark image
    out: dict = {"background": {}, "nets": {}}
    for polarity, image, dark_on_light in (
        ("dark_on_light", dark, True),
        ("light_on_dark", light, False),
    ):
        background = estimate_background(image)
        out["background"][polarity] = background
        out["nets"][polarity] = _nets(image, background, dark_on_light=dark_on_light)

    nets = out["nets"]["dark_on_light"]
    batch = Batch(
        CONDITIONS,
        [
            ProteinNets("p53", Role.TARGET, nets["p53"]),
            ProteinNets("GAPDH", Role.LOADING_CONTROL, nets["GAPDH"]),
        ],
        control_condition=REFERENCE,
    )
    series, warnings = normalize_batch(batch)
    assert not warnings and len(series) == 1
    normalized = series[0].values
    values = {
        "normalized": normalized,
        "fold_change": fold_change_lane(
            normalized, CONDITIONS, REFERENCE, SAMPLES, included=INCLUDED
        ),
    }
    out["lanes"] = values

    out["reduced"] = {}
    for kind, lane_values in values.items():
        for method in ReduceMethod:
            reduction = reduce_samples(
                lane_values, CONDITIONS, SAMPLES, included=INCLUDED, method=method
            )
            groups = reduction.groups
            out["reduced"][f"{kind}/{method}"] = {
                "groups": groups,
                "averaged": reduction.averaged,
                "all": _stats(groups),  # 3 groups: one-way ANOVA + Tukey HSD
                "ctl_vs_A": _stats({k: groups[k] for k in ("ctl", "A")}),  # Welch's t
            }
    # Round-trip through JSON so tuples, enums and floats compare like the file.
    return json.loads(json.dumps(out))


def _assert_close(actual, expected, rel: float, abs_: float, where: str = "$") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), where
        assert actual.keys() == expected.keys(), where
        for key in expected:
            _assert_close(actual[key], expected[key], rel, abs_, f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(a, e, rel, abs_, f"{where}[{i}]")
    elif isinstance(expected, float) or (
        isinstance(expected, int) and not isinstance(expected, bool) and isinstance(actual, float)
    ):
        assert actual == pytest.approx(expected, rel=rel, abs=abs_), where
    else:
        assert actual == expected, where


def test_quantification_chain_matches_the_regression_baseline():
    actual = _compute()
    if os.environ.get("PROTEIA_UPDATE_BASELINE") == "1":
        golden = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
        golden["values"] = actual
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        pytest.skip(f"rewrote {GOLDEN.name}; commit it with the reason for the change")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    tolerance = golden["tolerance"]
    _assert_close(actual, golden["values"], tolerance["rel"], tolerance["abs"])


def test_baseline_fixture_exercises_repeats_exclusion_and_the_reference():
    # Keep the fixture meaningful: a technical repeat is collapsed, the
    # presentation-only lane is dropped, and the reference reads 1.0.
    reduced = _compute()["reduced"]["fold_change/mean"]
    assert reduced["averaged"] == [["ctl", "c1"]]
    assert [len(v) for v in reduced["groups"].values()] == [2, 2, 2]  # a3 excluded
    assert np.mean(reduced["groups"]["ctl"]) == pytest.approx(1.0)
    assert reduced["all"]["compare"]["test"] == "anova_oneway"
    assert reduced["ctl_vs_A"]["compare"]["test"] == "welch_t"
