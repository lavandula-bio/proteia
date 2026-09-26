# SPDX-License-Identifier: Apache-2.0
"""Tests for the plot spec builder and the matplotlib renderer."""

import json
import math

import pytest

from proteia.core import analyze
from proteia.core.analyze import compare, describe
from proteia.core.plotspec import (
    NO_P_VALUE,
    NO_VARIATION,
    NO_VARIATION_WITHIN,
    ErrorType,
    PlotSpec,
    ValueKind,
    build_plotspec,
)
from proteia.viz import render_figure, save_figure


def _not_json(constant: str) -> float:
    raise ValueError(f"{constant} is not JSON")


def _assert_strict_json(spec: PlotSpec) -> None:
    """Strict JSON with nothing lost. Pydantic writes NaN and inf as null, which
    strict JSON accepts, so only the round trip shows that none got into the spec."""
    dump = spec.model_dump_json()
    json.loads(dump, parse_constant=_not_json)
    assert PlotSpec.model_validate_json(dump) == spec


def _spec(error_type=ErrorType.SD):
    groups = {"ctl": [1.0, 1.1, 0.9], "A": [2.0, 2.1, 1.9], "B": [0.5, 0.6]}
    return build_plotspec(
        groups,
        describe(groups),
        compare(groups),
        value_kind=ValueKind.LOADING_NORMALIZED,
        error_type=error_type,
        title="test",
        lane_indices={"ctl": [0, 1, 2], "A": [3, 4, 5], "B": [6, 7]},
    )


def test_build_plotspec_carries_points_and_provenance():
    spec = _spec()
    bar = next(b for b in spec.bars if b.label == "A")
    assert bar.n == 3
    assert bar.points == [2.0, 2.1, 1.9]
    assert bar.lane_indices == [3, 4, 5]  # provenance link survives


def test_error_type_selects_sd_vs_sem():
    sd_bar = next(b for b in _spec(ErrorType.SD).bars if b.label == "ctl")
    sem_bar = next(b for b in _spec(ErrorType.SEM).bars if b.label == "ctl")
    assert sem_bar.error < sd_bar.error  # SEM = SD / sqrt(n)


def test_only_significant_comparisons_become_brackets():
    spec = _spec()
    for comp in spec.comparisons:
        assert comp.p_value < 0.05


def test_significance_stars():
    spec = _spec()
    for comp in spec.comparisons:
        assert comp.stars in {"*", "**", "***"}


def test_value_kind_sets_y_label():
    spec = _spec()
    assert "target / loading" in spec.y_label


def test_render_figure_has_one_axes():
    fig = render_figure(_spec())
    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylabel()


def test_save_figure_writes_file(tmp_path):
    out = tmp_path / "chart.png"
    save_figure(_spec(), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_first_label_moves_to_front():
    groups = {"A": [1.0, 1.1], "ctl": [1.0, 0.9], "B": [2.0, 2.1]}
    spec = build_plotspec(
        groups,
        describe(groups),
        compare(groups),
        value_kind=ValueKind.FOLD_CHANGE,
        first_label="ctl",
    )
    assert [b.label for b in spec.bars] == ["ctl", "A", "B"]  # control leftmost, rest in order


def test_a_test_that_ran_has_no_note_and_no_subtitle_by_default():
    spec = _spec()
    assert spec.test_name == "anova_oneway" and spec.test_p is not None
    assert (spec.test_note, spec.subtitle) == (None, None)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    "groups",
    [
        {"ctl": [1.0, 1.0], "A": [1.0, 1.0]},  # Welch's t
        {"ctl": [2.0, 2.0], "α": [2.0, 2.0], "β": [2.0, 2.0]},  # ANOVA + Tukey
    ],
)
def test_a_non_finite_p_means_no_test(groups):
    test = compare(groups)
    assert math.isnan(test.p_value) and test.pairwise  # what the core gives
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == NO_VARIATION
    _assert_strict_json(spec)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    ("groups", "note"),
    [
        ({"ctl": [1.0, 1.0], "A": [2.0, 2.0]}, NO_VARIATION_WITHIN),  # Welch's t: p = 0
        # ANOVA: p = 0, Tukey: ctl vs α NaN, both against β 0.
        ({"ctl": [1.0, 1.0], "α": [1.0, 1.0], "β": [3.0, 3.0]}, NO_VARIATION_WITHIN),
        # A rounded mean leaves a tiny SD, so scipy's p is finite: 1.8e-46, then 1.
        ({"ctl": [0.1] * 3, "A": [0.2] * 3}, NO_VARIATION_WITHIN),
        ({"ctl": [0.1] * 3, "A": [0.1] * 3}, NO_VARIATION),
    ],
)
def test_values_constant_within_each_condition_give_no_test(groups, note):
    test = compare(groups)
    assert math.isfinite(test.p_value)  # what the core gives: a p that means nothing
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == note
    _assert_strict_json(spec)


def test_a_non_finite_p_keeps_the_tests_own_note():
    groups = {"ctl": [1.0, 1.0], "A": [1.0, 1.0]}
    test = analyze.TestResult("welch_t", math.inf, None, note="a reason from the core")
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p, spec.test_note) == (None, None, "a reason from the core")


@pytest.mark.parametrize("p", [math.nan, math.inf])
def test_a_non_finite_p_draws_no_bracket_even_for_a_significant_pair(p):
    groups = {"ctl": [1.0, 1.2], "A": [2.0, 2.3]}  # the values vary: no reason of our own
    pairwise = [analyze.PairwiseResult("ctl", "A", 0.001)]
    test = analyze.TestResult("anova_oneway", p, p, pairwise=pairwise)
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == NO_P_VALUE


def test_a_group_of_one_gives_the_cores_reason_and_no_brackets():
    groups = {"ctl": [1.0], "10 µM": [2.0, 2.1]}  # "ctl" cannot enter a test
    test = compare(groups)
    assert test.p_value is None
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == test.note == "need >=2 groups with >=2 replicates for a test"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    ("groups", "note"),
    [
        # The core tests vehicle and 10 µM alone: Welch's t, p = 0.005, one bracket.
        (
            {"vehicle": [1.0, 1.1], "10 µM": [2.0, 2.1], "50 µM": [3.0]},
            "no test: '50 µM' has fewer than 2 replicates",
        ),
        # The two tested groups do not vary (p is NaN), but the chart's values do.
        (
            {"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [5.0]},
            "no test: 'c' has fewer than 2 replicates",
        ),
        (
            {"α": [1.0], "β": [2.0, 2.2], "γ": [3.0, 3.1], "δ": [4.0]},
            "no test: 'α', 'δ' have fewer than 2 replicates",
        ),
    ],
)
def test_a_group_of_one_beside_testable_groups_gives_no_test(groups, note):
    test = compare(groups)
    assert test.p_value is not None  # the core tests the other groups only
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert [bar.label for bar in spec.bars] == list(groups)  # every group is drawn
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == note


def test_subtitle_is_passed_through():
    groups = {"ctl": [1.0, 1.1], "A": [2.0, 2.1]}
    spec = build_plotspec(
        groups,
        describe(groups),
        compare(groups),
        value_kind=ValueKind.FOLD_CHANGE,
        subtitle="All lanes",
    )
    assert spec.subtitle == "All lanes"


def test_render_draws_the_subtitle_as_the_second_title_line():
    plain = _spec()
    without = render_figure(plain).axes[0].get_title()
    assert without == f"test\nanova_oneway: p = {plain.test_p:.3g}"  # unchanged by #52
    labelled = plain.model_copy(update={"subtitle": "Excluding lanes 3, 7"})
    title = render_figure(labelled).axes[0].get_title()
    assert title.split("\n") == ["test", "Excluding lanes 3, 7", without.split("\n")[1]]


def test_render_handles_empty_group_without_crashing():
    # A condition with no values (all-None lanes) yields a zero-point bar; the
    # renderer must not raise on it (regression for the strict-zip crash).
    groups = {"a": [1.0, 2.0], "empty": []}
    spec = build_plotspec(
        groups, describe(groups), compare(groups), value_kind=ValueKind.LOADING_NORMALIZED
    )
    fig = render_figure(spec)
    assert len(fig.axes) == 1
