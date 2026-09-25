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
import functools
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
# Non-ASCII names on purpose: labels travel through the analysis and the JSON file.
TARGET, LOADING = "β-catenin", "α-tubulin"
ROW_Y = {TARGET: 50, LOADING: 105}  # row centres
DARKNESS = {
    TARGET: [60.0, 66.0, 54.0, 118.0, 126.0, 40.0, 31.0, 35.0],
    LOADING: [100.0, 94.0, 104.0, 98.0, 103.0, 100.0, 96.0, 102.0],
}
REFERENCE, LOW, HIGH = "vehicle", "10 µM", "50 µM"
CONDITIONS = [REFERENCE] * 3 + [LOW] * 3 + [HIGH] * 2
SAMPLES = ["v1", "v1", "v2", "a1", "a2", "a3", "b1", "b2"]  # v1 is loaded twice
INCLUDED = [True, True, True, True, True, False, True, True]  # a3 is presentation-only
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


@functools.cache
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
            ProteinNets(TARGET, Role.TARGET, nets[TARGET]),
            ProteinNets(LOADING, Role.LOADING_CONTROL, nets[LOADING]),
        ],
        control_condition=REFERENCE,
    )
    series, warnings = normalize_batch(batch)
    assert not warnings and len(series) == 1
    normalized = series[0].values
    out["lanes"] = {"normalized": normalized}
    out["reduced"] = {}
    for method in ReduceMethod:
        # The baseline is reduced with the same method as the statistics.
        fold = fold_change_lane(
            normalized, CONDITIONS, REFERENCE, SAMPLES, included=INCLUDED, method=method
        )
        out["lanes"][f"fold_change/{method}"] = fold
        for kind, lane_values in (("normalized", normalized), ("fold_change", fold)):
            reduction = reduce_samples(
                lane_values, CONDITIONS, SAMPLES, included=INCLUDED, method=method
            )
            groups = reduction.groups
            out["reduced"][f"{kind}/{method}"] = {
                "groups": groups,
                "averaged": reduction.averaged,
                "all": _stats(groups),  # 3 groups: one-way ANOVA + Tukey HSD
                "two": _stats({k: groups[k] for k in (REFERENCE, LOW)}),  # Welch's t
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
        assert actual == pytest.approx(expected, rel=rel, abs=abs_, nan_ok=True), where
    else:
        assert actual == expected, where


def _rewrite_golden(actual: dict) -> None:
    if os.environ.get("CI"):
        pytest.fail("PROTEIA_UPDATE_BASELINE is set in CI; update the baseline locally, on purpose")
    if not GOLDEN.exists():
        pytest.fail(f"{GOLDEN} is missing; restore it from git (its tolerance is kept on update)")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    golden["values"] = actual
    # LF on every platform, like the rest of the working tree.
    text = json.dumps(golden, indent=1, ensure_ascii=False) + "\n"
    GOLDEN.write_text(text, encoding="utf-8", newline="\n")


def test_quantification_chain_matches_the_regression_baseline():
    actual = _compute()
    if os.environ.get("PROTEIA_UPDATE_BASELINE") == "1":
        _rewrite_golden(actual)
        pytest.skip(f"rewrote {GOLDEN.name}; commit it with the reason for the change")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    tolerance = golden["tolerance"]
    _assert_close(actual, golden["values"], tolerance["rel"], tolerance["abs"])


def test_baseline_fixture_exercises_repeats_exclusion_and_the_reference():
    # Keep the fixture meaningful: a technical repeat is collapsed, the
    # presentation-only lane is dropped, and the reference reads 1.0.
    for method in ReduceMethod:
        reduced = _compute()["reduced"][f"fold_change/{method}"]
        assert reduced["averaged"] == [[REFERENCE, "v1"]]
        assert [len(v) for v in reduced["groups"].values()] == [2, 2, 2]  # a3 excluded
        assert np.mean(reduced["groups"][REFERENCE]) == pytest.approx(1.0)
        assert reduced["all"]["compare"]["test"] == "anova_oneway"
        assert reduced["two"]["compare"]["test"] == "welch_t"
