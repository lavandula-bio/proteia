# SPDX-License-Identifier: Apache-2.0
"""Tests for the plot spec builder and the matplotlib renderer."""

import dataclasses
import math

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from conftest import assert_strict_json
from proteia.core import analyze
from proteia.core.analyze import StatisticsSetting, compare, describe
from proteia.core.plotspec import (
    DEFAULT_X_LABEL,
    NO_P_VALUE,
    NO_VARIATION,
    NO_VARIATION_WITHIN,
    TOO_FEW_CONDITIONS,
    Coverage,
    ErrorType,
    PlotSpec,
    ValueKind,
    build_plotspec,
    left_out_text,
    statement_lines,
)
from proteia.viz import render_figure, render_svg

AUTO = StatisticsSetting()
LINEAR = StatisticsSetting(scale="linear")


def _test(groups, *, reference=None, ratio=True, setting=AUTO, untested=()):
    """The core's test of ``groups`` less the conditions ``untested``."""
    tested = {label: values for label, values in groups.items() if label not in untested}
    return compare(tested, setting, ratio=ratio, reference=reference)


def _spec(error_type=ErrorType.SD):
    groups = {"ctl": [1.0, 1.1, 0.9], "A": [2.0, 2.1, 1.9], "B": [0.5, 0.6]}
    return build_plotspec(
        groups,
        describe(groups),
        _test(groups),
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


@pytest.mark.parametrize("error_type", ["SD", "SEM"])
def test_a_raw_error_type_behaves_like_its_enum(error_type):
    raw = _spec(error_type)
    assert raw == _spec(ErrorType(error_type))
    assert raw.error_type is ErrorType(error_type)


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


def test_first_label_moves_to_front():
    groups = {"A": [1.0, 1.1], "ctl": [1.0, 0.9], "B": [2.0, 2.1]}
    spec = build_plotspec(
        groups,
        describe(groups),
        _test(groups, reference="ctl"),
        value_kind=ValueKind.FOLD_CHANGE,
        first_label="ctl",
    )
    assert [b.label for b in spec.bars] == ["ctl", "A", "B"]  # control leftmost, rest in order


def test_a_test_that_ran_has_no_note_and_no_subtitle_by_default():
    spec = _spec()
    assert spec.test_name == "welch_anova_games_howell"  # n of 3, 3 and 2
    assert spec.test is not None and spec.test.id == spec.test_name
    assert spec.test_p == spec.test.p_value is not None
    assert (spec.test_note, spec.subtitle) == (None, None)


def test_the_titles_have_defaults_and_can_be_replaced():
    spec = _spec()
    assert (spec.title, spec.x_label) == ("test", DEFAULT_X_LABEL)
    assert spec.y_label == "Normalized signal (target / loading)"
    groups = {"ctl": [1.0, 1.1], "A": [2.0, 2.1]}
    named = build_plotspec(
        groups,
        describe(groups),
        _test(groups),
        value_kind=ValueKind.FOLD_CHANGE,
        title="p-ERK",
        x_label="",
        y_label="p-ERK / GAPDH (fold)",
    )
    assert (named.title, named.x_label, named.y_label) == ("p-ERK", "", "p-ERK / GAPDH (fold)")
    axes = render_figure(named).axes[0]
    drawn = (axes.get_title(), axes.get_xlabel(), axes.get_ylabel())
    assert drawn == ("p-ERK", "", "p-ERK / GAPDH (fold)")
    assert render_figure(spec).axes[0].get_xlabel() == DEFAULT_X_LABEL


def test_a_test_without_its_plan_is_refused():
    groups = {"ctl": [1.0, 1.2], "A": [2.0, 2.3]}
    test = analyze.TestResult("welch_t", 0.01, 5.0)
    with pytest.raises(ValueError, match="no plan"):
        build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    "groups",
    [
        {"ctl": [1.0, 1.0], "A": [1.0, 1.0]},  # Student's t
        {"ctl": [2.0, 2.0], "α": [2.0, 2.0], "β": [2.0, 2.0]},  # ANOVA + Tukey-Kramer
    ],
)
def test_a_non_finite_p_means_no_test(groups):
    test = _test(groups)
    assert math.isnan(test.p_value) and test.pairwise  # what the core gives
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p, spec.comparisons, spec.test) == (None, None, [], None)
    assert spec.test_note == NO_VARIATION
    assert_strict_json(spec)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    ("groups", "note"),
    [
        ({"ctl": [1.0, 1.0], "A": [2.0, 2.0]}, NO_VARIATION_WITHIN),  # Student's t: p = 0
        # ANOVA: p = 0, Tukey: ctl vs α NaN, both against β 0.
        ({"ctl": [1.0, 1.0], "α": [1.0, 1.0], "β": [3.0, 3.0]}, NO_VARIATION_WITHIN),
        # A rounded mean leaves a tiny SD, so scipy's p is finite: 1.8e-46, then 1.
        ({"ctl": [0.1] * 3, "A": [0.2] * 3}, NO_VARIATION_WITHIN),
        ({"ctl": [0.1] * 3, "A": [0.1] * 3}, NO_VARIATION),
    ],
)
def test_values_constant_within_each_condition_give_no_test(groups, note):
    test = _test(groups, setting=LINEAR)
    assert math.isfinite(test.p_value)  # what the core gives: a p that means nothing
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == note
    assert_strict_json(spec)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    ("groups", "note"),
    [
        # 0.1 + 0.2 is 0.30000000000000004: the t-test gives p = 1e-23, three stars.
        ({"ctl": [0.30000000000000004, 0.3], "A": [0.6, 0.6000000000000001]}, NO_VARIATION_WITHIN),
        # ANOVA p = 0, and Tukey brackets A against both others with three stars.
        (
            {
                "ctl": [0.30000000000000004, 0.3, 0.3],
                "A": [0.7, 0.7000000000000001, 0.7],
                "B": [0.3, 0.3, 0.30000000000000004],
            },
            NO_VARIATION_WITHIN,
        ),
        ({"ctl": [0.30000000000000004, 0.3], "A": [0.3, 0.3]}, NO_VARIATION),  # p = 1
        ({"ctl": [0.0, 0.0], "A": [0.0, 0.0]}, NO_VARIATION),  # all 0: no spread at all
    ],
)
@pytest.mark.parametrize("setting", [AUTO, LINEAR], ids=["log", "linear"])
def test_values_constant_up_to_rounding_give_no_test(groups, note, setting):
    test = _test(groups, setting=setting)
    assert test.test != "none"  # the core runs a test
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == note


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_values_that_vary_are_tested_at_any_magnitude(scale):
    # The rounding tolerance is relative: tiny or huge values that vary are tested.
    groups = {"ctl": [1.0 * scale, 1.1 * scale], "A": [2.0 * scale, 2.2 * scale]}
    spec = build_plotspec(
        groups, describe(groups), _test(groups, ratio=False), value_kind=ValueKind.RAW
    )
    assert (spec.test_name, spec.test_note) == ("student_t", None)
    assert spec.test_p is not None and 0 < spec.test_p < 0.05


@pytest.mark.parametrize("p", [math.nan, math.inf])
def test_a_non_finite_p_draws_no_bracket_even_for_a_significant_pair(p):
    groups = {"ctl": [1.0, 1.2], "A": [2.0, 2.3]}  # the values vary: no reason of our own
    pairwise = [analyze.PairwiseResult("ctl", "A", 0.001)]
    test = dataclasses.replace(_test(groups), p_value=p, pairwise=pairwise)
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == NO_P_VALUE


def test_a_many_to_one_test_has_no_omnibus_p_and_finite_pairwise_ones():
    groups = {"ctl": [1.0, 1.1, 0.9], "A": [2.0, 2.2, 1.9], "B": [0.5, 0.6, 0.55]}
    spec = _fold_change(groups, reference="ctl")
    assert spec.test is not None and spec.test.id == "dunnett"
    assert (spec.test.p_value, spec.test.statistic, spec.test_p) == (None, None, None)
    assert [(p.group_a, p.group_b) for p in spec.test.pairwise] == [("ctl", "A"), ("ctl", "B")]
    assert all(math.isfinite(p.p_value) for p in spec.test.pairwise)
    # Brackets pair the reference only.
    assert spec.comparisons and all(c.group_a == "ctl" for c in spec.comparisons)
    assert_strict_json(spec)


@pytest.mark.parametrize("p", [None, math.nan])
def test_a_many_to_one_test_needs_every_pairwise_p(p):
    groups = {"ctl": [1.0, 1.1, 0.9], "A": [2.0, 2.2, 1.9], "B": [0.5, 0.6, 0.55]}
    real = _test(groups, reference="ctl")
    pairwise = [real.pairwise[0], dataclasses.replace(real.pairwise[1], p_value=math.nan)]
    test = dataclasses.replace(real, pairwise=pairwise, p_value=p)
    spec = _fold_change(groups, test, reference="ctl")
    assert (spec.test, spec.comparisons, spec.test_note) == (None, [], NO_P_VALUE)


@pytest.mark.parametrize(
    ("groups", "note"),
    [
        ({"ctl": [1.0], "10 µM": [2.0, 2.1]}, "no test: 'ctl' has fewer than 2 replicates"),
        ({"ctl": [1.0], "10 µM": [2.0]}, "no test: 'ctl', '10 µM' have fewer than 2 replicates"),
    ],
)
def test_a_group_of_one_is_named_even_when_the_core_gives_its_own_note(groups, note):
    test = _test(groups)
    assert test.p_value is None and test.note  # the core's note names no group
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == note


def test_a_single_condition_says_a_test_needs_two():
    groups = {"ctl": [1.0, 1.1]}  # every group has two samples, but there is one group
    test = _test(groups)
    assert test.note == analyze.TOO_FEW_NOTE
    spec = build_plotspec(groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE)
    assert (spec.test_name, spec.test_p) == (None, None)
    assert spec.test_note == TOO_FEW_CONDITIONS


def _fold_change(groups, test=None, *, reference=None, setting=AUTO, **kwargs):
    """The chart of ``groups``, with the core's test unless ``test`` is given."""
    if test is None:
        untested = kwargs.get("not_detected_lanes") or {}
        test = _test(groups, reference=reference, setting=setting, untested=untested)
    if reference is not None:
        kwargs.setdefault("first_label", reference)
    return build_plotspec(
        groups, describe(groups), test, value_kind=ValueKind.FOLD_CHANGE, **kwargs
    )


@pytest.mark.parametrize(
    ("groups", "test", "tested", "note", "line"),
    [
        # Student's t over vehicle and 10 µM: p = 0.006, one bracket.
        (
            {"vehicle": [1.0, 1.1], "10 µM": [2.0, 2.1], "50 µM": [3.0]},
            "student_t",
            {"vehicle", "10 µM"},
            "'50 µM' (n = 1) is not in the test",
            "Not tested: '50 µM' (n = 1)",
        ),
        # ANOVA and Tukey-Kramer over three of the four groups: three brackets.
        (
            {"ctl": [1.0, 1.1, 0.9], "A": [2.0, 2.1, 2.2], "B": [3.0, 3.2, 3.1], "C": [9.0]},
            "anova_tukey",
            {"ctl", "A", "B"},
            "'C' (n = 1) is not in the test",
            "Not tested: 'C' (n = 1)",
        ),
        # Two groups left out: Student's t over β and γ, p = 0.02.
        (
            {"α": [1.0], "β": [2.0, 2.2], "γ": [3.0, 3.1], "δ": [4.0]},
            "student_t",
            {"β", "γ"},
            "'α', 'δ' (n = 1) are not in the test",
            "Not tested: 'α', 'δ' (n = 1)",
        ),
    ],
)
def test_a_group_of_one_beside_testable_groups_is_left_out_of_their_test(
    groups, test, tested, note, line
):
    core = _test(groups)
    spec = _fold_change(groups, core)
    assert [bar.label for bar in spec.bars] == list(groups)  # every group is drawn
    assert (spec.test_name, spec.test_p) == (test, core.p_value)  # the core's test of the rest
    assert spec.test_note == note
    brackets = [(c.group_a, c.group_b, c.p_value) for c in spec.comparisons]
    assert brackets == [
        (p.group_a, p.group_b, p.p_value) for p in core.pairwise if p.p_value < 0.05
    ]
    assert brackets and all({a, b} <= tested for a, b, _ in brackets)
    assert {c.label for c in spec.coverage if c.tested} == tested
    assert all(c.left_out == "fewer_than_2" for c in spec.coverage if not c.tested)
    assert spec.statement[2] == line
    assert_strict_json(spec)


def test_brackets_pair_only_the_tested_groups():
    # A test whose pairwise results name a group with n = 1 draws no bracket to it.
    groups = {"vehicle": [1.0, 1.1], "10 µM": [2.0, 2.1], "50 µM": [3.0]}
    pairwise = [
        analyze.PairwiseResult("vehicle", "10 µM", 0.01),
        analyze.PairwiseResult("vehicle", "50 µM", 0.001),
        analyze.PairwiseResult("10 µM", "50 µM", 0.001),
    ]
    spec = _fold_change(groups, dataclasses.replace(_test(groups), p_value=0.01, pairwise=pairwise))
    assert (spec.test_name, spec.test_p) == ("student_t", 0.01)
    assert [(c.group_a, c.group_b) for c in spec.comparisons] == [("vehicle", "10 µM")]
    assert spec.test_note == "'50 µM' (n = 1) is not in the test"


def test_groups_left_out_with_different_reasons_are_named_with_their_own():
    groups = {"ctl": [1.0, 1.1], "A": [2.0, 2.1], "B": [3.0], "C": []}
    spec = _fold_change(groups)
    assert spec.test_name == "student_t"
    assert spec.test_note == "'B' (n = 1), 'C' (no value) are not in the test"
    assert spec.statement[2] == "Not tested: 'B' (n = 1), 'C' (no value)"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    ("groups", "note"),
    [
        # The tested groups hold one value (p is NaN), but c's differs: "the values
        # do not vary" would be wrong.
        (
            {"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [5.0]},
            "no test: 'c' has fewer than 2 replicates,"
            " and the values of the other conditions do not vary",
        ),
        # Only the tested groups' means differ (p = 0).
        (
            {"a": [1.0, 1.0], "b": [2.0, 2.0], "c": [5.0]},
            "no test: 'c' has fewer than 2 replicates,"
            " and the values within each of the other conditions do not vary",
        ),
        # Up to rounding: the t-test gives p = 1e-23 alone.
        (
            {"a": [0.30000000000000004, 0.3], "b": [0.6, 0.6000000000000001], "c": [0.3]},
            "no test: 'c' has fewer than 2 replicates,"
            " and the values within each of the other conditions do not vary",
        ),
        (
            {"α": [1.0], "β": [2.0, 2.0], "γ": [2.0, 2.0], "δ": [4.0]},
            "no test: 'α', 'δ' have fewer than 2 replicates,"
            " and the values of the other conditions do not vary",
        ),
    ],
)
def test_a_group_of_one_beside_constant_tested_groups_gives_no_test(groups, note):
    test = _test(groups)
    assert test.test == "student_t" and test.pairwise  # the core tests the others
    spec = _fold_change(groups, test)
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == note
    assert_strict_json(spec)


@pytest.mark.parametrize("p", [None, math.nan, math.inf])
def test_a_test_of_the_others_without_a_finite_p_gives_no_test(p):
    groups = {"ctl": [1.0, 1.2], "A": [2.0, 2.3], "B": [3.0]}  # the tested values vary
    pairwise = [analyze.PairwiseResult("ctl", "A", 0.001)]
    spec = _fold_change(groups, dataclasses.replace(_test(groups), p_value=p, pairwise=pairwise))
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == (
        "no test: 'B' has fewer than 2 replicates,"
        " and the test of the other conditions gives no p-value"
    )


def test_one_condition_never_shows_a_test():
    # Whatever the result claims, a plan with one condition has no test to show.
    groups = {"ctl": [1.0, 1.1]}
    claimed = dataclasses.replace(_test(groups), test="welch_t", p_value=0.01, statistic=5.0)
    spec = _fold_change(groups, claimed)
    assert (spec.test_name, spec.test_p, spec.comparisons) == (None, None, [])
    assert spec.test_note == TOO_FEW_CONDITIONS


def test_subtitle_is_passed_through():
    groups = {"ctl": [1.0, 1.1], "A": [2.0, 2.1]}
    spec = build_plotspec(
        groups,
        describe(groups),
        _test(groups),
        value_kind=ValueKind.FOLD_CHANGE,
        subtitle="All lanes",
    )
    assert spec.subtitle == "All lanes"


# --- The statement: the chart's legend text ---

_T = {"ctl": [1.0, 1.1, 0.9], "A": [2.0, 2.2, 1.9], "B": [0.5, 0.6, 0.55]}


def _p(p):
    """A p-value as charts and the page's captions write it (plotspec.p_text)."""
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.3g}"


def test_the_statement_of_a_two_condition_test():
    groups = {"ctl": _T["ctl"], "A": _T["A"]}
    spec = _fold_change(groups, reference="ctl")
    assert spec.statement == [
        "Bars: mean ± SD; points: replicates",
        f"Student's t-test on log values: {_p(spec.test_p)}",
        "Tested: both conditions",
    ]


def test_the_statement_of_an_omnibus_test():
    spec = _fold_change(_T, error_type=ErrorType.SEM)
    assert spec.statement == [
        "Bars: mean ± SEM; points: replicates",
        f"One-way ANOVA + Tukey-Kramer on log values: ANOVA {_p(spec.test_p)}",
        "Tested: all 3 conditions",
    ]


def _vs_reference(spec):
    """``'A' p = 0.0029, 'B' p = 0.011``: each comparison's p, in the chart's order."""
    return ", ".join(f"{p.group_b!r} {_p(p.p_value)}" for p in spec.test.pairwise)


def test_the_statement_of_a_test_vs_the_reference():
    spec = _fold_change(_T, reference="ctl")
    assert [(p.group_a, p.group_b) for p in spec.test.pairwise] == [("ctl", "A"), ("ctl", "B")]
    assert spec.statement[1:] == [
        f"Dunnett's test on log values, each condition vs 'ctl': {_vs_reference(spec)}",
        "Tested: all 3 conditions",
    ]
    unequal = _fold_change({**_T, "B": [0.5, 0.6]}, reference="ctl")
    assert unequal.statement[1] == (
        f"Welch's t-tests (Holm) on log values, each condition vs 'ctl': {_vs_reference(unequal)}"
    )


def test_a_test_vs_the_reference_gives_its_p_values_when_none_is_significant():
    groups = {"ctl": [1.0, 1.4, 0.7], "A": [1.1, 0.8, 1.3], "B": [0.9, 1.2, 1.0]}
    spec = _fold_change(groups, reference="ctl")
    assert spec.test.id == "dunnett" and spec.comparisons == []  # no bracket
    ps = [p.p_value for p in spec.test.pairwise]
    assert all(p > 0.05 for p in ps)
    assert spec.statement[1] == (
        f"Dunnett's test on log values, each condition vs 'ctl': 'A' {_p(ps[0])}, 'B' {_p(ps[1])}"
    )


def test_the_statement_says_when_mann_whitney_is_the_normal_approximation():
    # 12 vs 12 with ties: too many arrangements for the exact permutation.
    groups = {
        "ctl": [1.1, 1.2, 1.1, 0.7, 1.2, 1.1, 0.9, 1.1, 1.1, 1.1, 1.0, 1.1],
        "A": [1.2, 1.3, 1.2, 1.4, 1.3, 1.2, 1.1, 1.2, 1.3, 1.2, 1.6, 1.5],
    }
    spec = _fold_change(groups, reference="ctl", setting=StatisticsSetting(family="rank"))
    assert spec.test.method == "normal approximation"
    assert spec.statement[1] == f"Mann-Whitney U test (normal approximation): {_p(spec.test_p)}"
    two = {"ctl": _T["ctl"], "A": _T["A"]}
    exact = _fold_change(two, reference="ctl", setting=StatisticsSetting(family="rank"))
    assert exact.test.method == "exact"  # the record says so; the statement adds nothing
    assert exact.statement[1] == "Mann-Whitney U test: p = 0.1"


def test_the_statement_of_a_rank_test_names_its_floor():
    groups = {"ctl": _T["ctl"], "A": _T["A"]}
    spec = _fold_change(groups, reference="ctl", setting=StatisticsSetting(family="rank"))
    assert spec.statement[1:] == [
        "Mann-Whitney U test: p = 0.1",
        "Tested: both conditions",
        "Mann-Whitney U test: with these n no p can be below 0.1,"
        " so no comparison can reach p < 0.05",
    ]
    assert spec.test.scale is None


def test_the_statement_of_a_ratio_tested_on_the_linear_scale():
    spec = _fold_change({**_T, "A": [0.0, 0.2, 0.1]}, reference="ctl")
    assert spec.statement[1:] == [
        f"Dunnett's test, each condition vs 'ctl': {_vs_reference(spec)}",
        "Tested: all 3 conditions",
        "Linear scale: 'A' has a value of 0 or below",
    ]


@pytest.mark.parametrize(
    ("setting", "reference", "line"),
    [
        (StatisticsSetting(family="none"), "ctl", "No test: statistics are turned off"),
        (
            StatisticsSetting(comparisons="vs_reference"),
            None,
            "No test: comparisons with the reference were chosen, but no reference is set",
        ),
    ],
)
def test_the_statement_says_why_there_is_no_test(setting, reference, line):
    spec = _fold_change(_T, reference=reference, setting=setting)
    assert spec.test is None and spec.comparisons == []
    assert spec.statement == ["Bars: mean ± SD; points: replicates", line]
    assert spec.test_note == line[0].lower() + line[1:]


def test_the_statement_is_the_one_statement_lines_gives():
    spec = _fold_change(_T, reference="ctl")
    assert spec.statement == statement_lines(spec)


# --- Conditions with replicates not detected (option A) and with no value ---

VEHICLE, LOW, KO = "vehicle", "10 µM", "KO"
DETECTED = {VEHICLE: [1.0, 1.1, 0.9], LOW: [2.0, 2.2, 1.9]}


def _nd_chart(ko_points, ko_lanes, undetected, *, replicates=3, **kwargs):
    """vehicle (lanes 0-2) and 10 µM (lanes 3-5) detected; KO in lanes 6-8, its
    detected values at ``ko_lanes`` and its replicates not detected at
    ``undetected``."""
    groups = {**DETECTED, KO: ko_points}
    lanes = {VEHICLE: [0, 1, 2], LOW: [3, 4, 5], KO: ko_lanes}
    counts = {VEHICLE: 3, LOW: 3, KO: replicates}
    shown = {k: v for k, v in groups.items() if v}
    return _fold_change(
        shown,
        reference=VEHICLE,
        lane_indices=lanes,
        replicates=counts,
        not_detected_lanes={KO: undetected},
        **kwargs,
    )


def test_a_condition_with_every_replicate_not_detected_keeps_its_slot():
    spec = _nd_chart([], [], [6, 7, 8])
    assert [bar.label for bar in spec.bars] == [VEHICLE, LOW, KO]  # its place on the x axis
    ko = spec.bars[2]
    assert (ko.mean, ko.error, ko.points, ko.n) == (None, None, [], 0)
    assert ko.not_detected_lanes == [6, 7, 8]
    assert spec.coverage[2] == Coverage(
        label=KO, n=0, replicates=3, not_detected=3, tested=False, left_out="not_detected"
    )
    assert spec.statement == [
        "Bars: mean ± SD; points: replicates; open circles: not detected",
        f"Student's t-test on log values: {_p(spec.test_p)}",  # k counts the tested only
        "Not tested: 'KO' (3 of 3 not detected)",
    ]
    assert spec.test_note == "'KO' (3 of 3 not detected) is not in the test"
    assert "Tested: all" not in " ".join(spec.statement)
    assert_strict_json(spec)


@pytest.mark.parametrize(
    ("points", "lanes", "undetected", "line"),
    [
        ([0.08], [6], [7, 8], "Not tested: 'KO' (2 of 3 not detected)"),
        ([0.08, 0.1], [6, 8], [7], "Not tested: 'KO' (1 of 3 not detected)"),
    ],
)
def test_a_condition_partly_detected_draws_no_bar_and_is_not_tested(
    points, lanes, undetected, line
):
    spec = _nd_chart(points, lanes, undetected)
    ko = spec.bars[2]
    assert (ko.mean, ko.error, ko.geometric_mean) == (None, None, None)
    assert ko.points == points  # its detected values are still drawn
    assert sorted(ko.lane_indices + ko.not_detected_lanes) == [6, 7, 8]  # each replicate once
    assert (spec.coverage[2].tested, spec.coverage[2].left_out) == (False, "not_detected")
    assert spec.statement[2] == line
    assert spec.test.covered == [VEHICLE, LOW]
    assert all(KO not in (c.group_a, c.group_b) for c in spec.comparisons)
    assert_strict_json(spec)


def test_constancy_is_read_from_the_tested_conditions_only():
    groups = {VEHICLE: [1.0, 1.0], LOW: [1.0, 1.0], KO: [5.0]}
    spec = _fold_change(
        groups,
        reference=VEHICLE,
        replicates={VEHICLE: 2, LOW: 2, KO: 2},
        not_detected_lanes={KO: [5]},
        lane_indices={VEHICLE: [0, 1], LOW: [2, 3], KO: [4]},
    )
    assert spec.test_note == (
        "no test: the values of the other conditions do not vary;"
        " not tested: 'KO' (1 of 2 not detected)"
    )


def test_no_test_is_left_when_only_one_condition_is_detected():
    groups = {VEHICLE: DETECTED[VEHICLE]}
    spec = _fold_change(
        groups,
        reference=VEHICLE,
        replicates={VEHICLE: 3, KO: 3},
        not_detected_lanes={KO: [3, 4, 5]},
    )
    assert spec.test is None
    assert spec.test_note == (
        "no test: fewer than 2 conditions to test; not tested: 'KO' (3 of 3 not detected)"
    )
    assert spec.statement[1] == (
        "No test: fewer than 2 conditions to test; not tested: 'KO' (3 of 3 not detected)"
    )


def test_a_condition_with_no_value_keeps_its_slot_and_is_named():
    spec = _fold_change(DETECTED, reference=VEHICLE, replicates={VEHICLE: 3, LOW: 3, KO: 2})
    assert [bar.label for bar in spec.bars] == [VEHICLE, LOW, KO]
    assert spec.coverage[2].left_out == "no_value" and spec.bars[2].not_detected_lanes == []
    assert spec.statement[0] == "Bars: mean ± SD; points: replicates"  # no open circles
    assert spec.statement[2] == "Not tested: 'KO' (no value)"


def test_the_left_out_reasons_have_an_order():
    # A replicate not detected outweighs no value, which outweighs one value.
    groups = {VEHICLE: [1.0, 1.1], LOW: [2.0, 2.1], "α": [3.0], "β": []}
    spec = _fold_change(
        {k: v for k, v in groups.items() if v},
        reference=VEHICLE,
        replicates={VEHICLE: 2, LOW: 2, "α": 2, "β": 1, KO: 1},
        not_detected_lanes={"α": [5], KO: [7]},
    )
    assert [(c.label, c.left_out) for c in spec.coverage] == [
        (VEHICLE, None),
        (LOW, None),
        ("α", "not_detected"),
        ("β", "no_value"),
        (KO, "not_detected"),
    ]
    assert left_out_text(spec.coverage) == (
        "'α' (1 of 2 not detected), 'β' (no value), 'KO' (1 of 1 not detected)"
    )


def test_bars_and_coverage_name_the_same_conditions():
    spec = _nd_chart([0.08], [6], [7, 8])
    assert [b.label for b in spec.bars] == [c.label for c in spec.coverage]
    assert [c.not_detected for c in spec.coverage] == [len(b.not_detected_lanes) for b in spec.bars]
    with pytest.raises(ValueError, match="same conditions"):
        PlotSpec.model_validate(
            {**spec.model_dump(), "coverage": spec.model_dump()["coverage"][:2]}
        )


def test_a_test_that_covers_a_condition_not_detected_is_refused():
    groups = {**DETECTED, KO: [0.08, 0.1]}
    everything = _test(groups, reference=VEHICLE)  # KO tested: the caller forgot to leave it out
    with pytest.raises(ValueError, match="covers"):
        _fold_change(
            groups,
            everything,
            reference=VEHICLE,
            replicates={VEHICLE: 3, LOW: 3, KO: 3},
            not_detected_lanes={KO: [8]},
        )


def test_names_are_kept_as_typed_in_the_statement():
    groups = {"µ": [1.0, 1.1, 0.9], "α": [2.0, 2.2, 1.9]}
    spec = _fold_change(
        groups,
        reference="µ",
        replicates={"µ": 3, "α": 3, "β": 3},
        not_detected_lanes={"β": [6, 7, 8]},
    )
    assert spec.statement[2] == "Not tested: 'β' (3 of 3 not detected)"
    assert_strict_json(spec)


def test_a_chart_without_replicates_has_coverage_from_its_bars():
    groups = {"ctl": [1.0, 1.1], "A": [2.0, 2.1], "B": [3.0]}
    spec = _fold_change(groups)  # no replicates, no not-detected lanes
    assert [(c.label, c.n, c.replicates, c.not_detected) for c in spec.coverage] == [
        ("ctl", 2, 2, 0),
        ("A", 2, 2, 0),
        ("B", 1, 1, 0),
    ]
    assert all(bar.mean is not None for bar in spec.bars)


def test_the_smallest_p_a_rank_test_could_give_is_carried():
    groups = {"ctl": _T["ctl"], "A": _T["A"]}
    spec = _fold_change(groups, reference="ctl")
    assert spec.rank_min_p == pytest.approx({"all_pairs": 0.1, "vs_reference": 0.1})
    three = _fold_change({"ctl": [1.0, 1.1], "A": [2.0, 2.1], "B": [3.0, 3.1]}, reference="ctl")
    assert set(three.rank_min_p) == {"all_pairs", "vs_reference"}
    assert all(p >= 0.05 for p in three.rank_min_p.values())  # 2 x 2 x 2 never reaches it


def test_the_rank_floors_count_the_comparisons_each_test_makes():
    # Three conditions of 3: Dunn's test makes 2 comparisons with the reference and
    # Kruskal-Wallis + Dunn 3 pairs; Holm multiplies the smallest p by that count.
    spec = _fold_change(_T, reference="ctl")
    assert spec.rank_min_p == pytest.approx(
        {
            "all_pairs": analyze.min_attainable_p([3, 3, 3], "kruskal_dunn_holm", 3),
            "vs_reference": analyze.min_attainable_p([3, 3, 3], "dunn_holm", 2),
        }
    )
    assert spec.rank_min_p["vs_reference"] < spec.rank_min_p["all_pairs"]


# --- Welch's tests weigh each condition by its own variance ---


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    ("groups", "setting", "test", "note"),
    [
        # Welch's t of ctl vs x has no standard error: scipy gives p = 0, three stars.
        (
            {"ctl": [1.0, 1.0], "x": [2.0, 2.0, 2.0], "y": [1.5, 1.9, 1.2]},
            AUTO,
            "welch_t_holm",
            "no test: the values within 'ctl', 'x' do not vary,"
            " so Welch's t-test gives no p-value between them",
        ),
        # The same up to rounding: p = 8e-32.
        (
            {
                "ctl": [1.0, 1.0000000000000002],
                "x": [2.0, 2.0000000000000004, 2.0],
                "y": [1.5, 1.9, 1.2],
            },
            AUTO,
            "welch_t_holm",
            "no test: the values within 'ctl', 'x' do not vary,"
            " so Welch's t-test gives no p-value between them",
        ),
        # Games-Howell of ctl vs x: p = 0; Welch's ANOVA weighs ctl by 1 / 0.
        (
            {
                "ctl": [1.0, 1.0000000000000002],
                "x": [2.0, 2.0000000000000004, 2.0],
                "y": [1.5, 1.9, 1.2],
            },
            StatisticsSetting(comparisons="all_pairs"),
            "welch_anova_games_howell",
            "no test: the values within 'ctl', 'x' do not vary, so Welch's ANOVA gives no p-value",
        ),
        # One condition whose values do not vary is enough for Welch's ANOVA: exactly
        # (NaN) or up to rounding (p = 0.001, from a weight of 1e32).
        (
            {"ctl": [1.0, 1.0000000000000002], "x": [2.0, 2.1, 1.9], "y": [1.5, 1.9, 1.2]},
            StatisticsSetting(comparisons="all_pairs"),
            "welch_anova_games_howell",
            "no test: the values within 'ctl' do not vary, so Welch's ANOVA gives no p-value",
        ),
        (
            {"ctl": [1.0, 1.0], "x": [2.0, 2.1, 1.9], "y": [1.5, 1.9, 1.2]},
            StatisticsSetting(comparisons="all_pairs"),
            "welch_anova_games_howell",
            "no test: the values within 'ctl' do not vary, so Welch's ANOVA gives no p-value",
        ),
        # Beside a condition left out, which the note names too.
        (
            {"ctl": [1.0, 1.0], "x": [2.0, 2.0, 2.0], "y": [1.5, 1.9, 1.2], "z": [4.0]},
            AUTO,
            "welch_t_holm",
            "no test: 'z' has fewer than 2 replicates, and the values within 'ctl', 'x'"
            " do not vary, so Welch's t-test gives no p-value between them",
        ),
    ],
)
def test_welchs_tests_of_conditions_whose_values_do_not_vary_give_no_test(
    groups, setting, test, note
):
    core = _test(groups, reference="ctl", setting=setting)
    assert core.test == test  # the core runs it
    spec = _fold_change(groups, core, reference="ctl")
    assert (spec.test, spec.test_name, spec.test_p, spec.comparisons) == (None, None, None, [])
    assert spec.test_note == note
    assert spec.statement[1:] == [note[0].upper() + note[1:]]
    assert_strict_json(spec)


def test_welchs_t_tests_with_one_condition_that_does_not_vary_still_run():
    # ctl does not vary but x and y do: each Welch t has a standard error.
    groups = {"ctl": [1.0, 1.0], "x": [2.0, 2.1, 1.9], "y": [1.5, 1.9, 1.2]}
    spec = _fold_change(groups, reference="ctl")
    assert spec.test is not None and spec.test.id == "welch_t_holm"
    assert all(math.isfinite(p.p_value) for p in spec.test.pairwise)
    # The same pair alone, as the two-condition chart tests it, runs as well.
    pair = _fold_change({"ctl": [1.0, 1.0], "x": [2.0, 2.1, 1.9]}, reference="ctl")
    assert pair.test is not None and pair.test.id == "welch_t"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
def test_a_pooled_test_of_conditions_whose_values_do_not_vary_still_runs():
    # Dunnett's test pools the variance of all three: ctl vs x has a standard error.
    groups = {"ctl": [1.0, 1.0, 1.0], "x": [2.0, 2.0, 2.0], "y": [1.5, 1.9, 1.2]}
    spec = _fold_change(groups, reference="ctl")
    assert spec.test is not None and spec.test.id == "dunnett"
    assert all(math.isfinite(p.p_value) for p in spec.test.pairwise)


# --- Drawing ---


def test_render_draws_the_subtitle_as_the_second_title_line():
    plain = _spec()
    assert render_figure(plain).axes[0].get_title() == "test"  # no statistics in the title
    labelled = plain.model_copy(update={"subtitle": "Excluding lanes 3, 7"})
    title = render_figure(labelled).axes[0].get_title()
    assert title.split("\n") == ["test", "Excluding lanes 3, 7"]


def _statement_text(fig):
    [text] = [t for t in fig.texts if t.get_gid() == "statement"]
    return text.get_text()


def _key_labels(fig):
    [legend] = fig.legends
    assert legend.get_gid() == "key"
    return [text.get_text() for text in legend.get_texts()]


def test_the_statement_is_drawn_under_the_chart_only_when_asked():
    groups = {"vehicle": [1.0, 1.1], "10 µM": [2.0, 2.1], "50 µM": [3.0]}
    spec = _fold_change(groups, title="test")
    assert not [t for t in render_figure(spec).texts if t.get_gid() == "statement"]
    fig = render_figure(spec, statement=True)
    assert _statement_text(fig).split() == " ".join(spec.statement).split()
    assert fig.get_figheight() > render_figure(spec).get_figheight()  # the plot keeps its size


@pytest.mark.parametrize("error_type", list(ErrorType))
def test_the_key_says_what_the_marks_are(error_type):
    spec = _fold_change(_T, error_type=error_type)
    assert _key_labels(render_figure(spec)) == [f"Mean ± {error_type.value}", "Replicate"]
    nd = _nd_chart([0.08], [6], [7, 8], error_type=error_type)
    assert _key_labels(render_figure(nd)) == [
        f"Mean ± {error_type.value}",
        "Replicate",
        "Not detected",
    ]


@pytest.mark.parametrize(
    ("groups", "name"),
    [
        ({"ctl": [1.0, 1.1], "A": [2.0, 2.1]}, "Student's t-test on log values"),
        ({"ctl": [1.0, 1.1], "A": [2.0, 2.1], "B": [3.0, 3.2]}, "One-way ANOVA + Tukey-Kramer"),
    ],
)
def test_render_names_the_test_as_a_reader_names_it(groups, name):
    # The page's caption names it so (the statement): the drawing says the same,
    # in a saved file.
    spec = _fold_change(groups, title="test")
    assert spec.statement[1].startswith(name)
    svg = render_svg(spec, statement=True).decode()
    assert spec.statement[1] in svg and spec.test_name not in svg


def test_a_p_value_below_0_0001_is_written_as_a_bound():
    spec = _fold_change({"ctl": [1.0, 1.01, 0.99], "A": [5.0, 5.01, 4.99]}, title="test")
    assert spec.test_p is not None and spec.test_p < 0.0001
    assert spec.statement[1].endswith(": p < 0.0001") and "e-" not in spec.statement[1]
    assert ": p < 0.0001" in _statement_text(render_figure(spec, statement=True))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    "groups",
    [
        {"ctl": [1.0, 1.1], "A": [2.0, 2.1]},  # a test ran
        {"ctl": [1.0], "A": [2.0, 2.1]},  # a group of one
        {"ctl": [1.0, 1.0], "A": [2.0, 2.0]},  # no variation within the conditions
        {"ctl": [1.0, 1.1], "A": [2.0, 2.1], "B": [3.0]},  # a test of two of three, and a note
        {"ctl": [1.0, 1.0], "A": [2.0, 2.0], "B": [3.0]},  # two of three, but no variation
    ],
)
@pytest.mark.parametrize("subtitle", [None, "All lanes"])
def test_every_chart_states_its_test_or_why_there_is_none(groups, subtitle):
    spec = _fold_change(groups, title="test", subtitle=subtitle)
    assert spec.test_name is not None or spec.test_note  # no test is never silent
    assert len(spec.statement) >= 2  # the marks, then the test or why there is none
    fig = render_figure(spec, statement=True)
    assert fig.axes[0].get_title().split("\n") == ["test", *([subtitle] if subtitle else [])]
    # A long line may be broken over lines.
    assert _statement_text(fig).split() == " ".join(spec.statement).split()


_CHART_TITLE = "p-ERK fold-change vs vehicle  (/GAPDH)"  # a fold-change title, with a double space


def _boxes_and_texts(spec):
    fig = render_figure(spec, statement=True)
    renderer = FigureCanvasAgg(fig).get_renderer()
    title = fig.axes[0].title
    [statement] = [t for t in fig.texts if t.get_gid() == "statement"]
    return (
        title.get_window_extent(renderer),
        statement.get_window_extent(renderer),
        fig.bbox,
        title.get_text(),
        statement.get_text(),
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    ("groups", "subtitle"),
    [
        ({"vehicle": [1.0, 1.0], "10 µM": [2.0, 2.0]}, None),  # no variation within
        ({"ctl": [1.0, 1.1]}, None),  # one condition
        ({"vehicle": [1.0], "10 µM": [2.0, 2.1]}, None),  # one short group, named
        ({"vehicle": [1.0], "10 µM": [2.0]}, None),  # two short groups, named
        ({"vehicle": [1.0], "10 µM": [2.0], "50 µM": [3.0]}, None),  # three
        (
            {"vehicle": [1.0, 1.1], "10 µM": [2.0, 2.1]},
            "Excluding lanes " + ", ".join(str(lane) for lane in range(1, 16)),
        ),
        # A test of two groups, and a long note naming the two it leaves out.
        (
            {
                "vehicle": [1.0, 1.1],
                "10 µM": [2.0, 2.1],
                "50 µM rapamycin + 10 nM bafilomycin A1": [3.0],
                "100 µM rapamycin + 10 nM bafilomycin A1": [4.0],
            },
            "Excluding lanes 4, 8",
        ),
    ],
)
def test_no_title_or_statement_line_runs_past_the_figure_edge(groups, subtitle):
    spec = _fold_change(groups, title=_CHART_TITLE, subtitle=subtitle)
    title_box, statement_box, figure, title, statement = _boxes_and_texts(spec)
    for box in (title_box, statement_box):  # a saved figure crops what lies outside
        assert 0 <= box.x0 and box.x1 <= figure.width
        assert 0 <= box.y0 and box.y1 <= figure.height
    wanted = " ".join([_CHART_TITLE, *([subtitle] if subtitle else [])])
    assert title.split() == wanted.split()  # every word is still drawn, in order
    assert statement.split() == " ".join(spec.statement).split()


def test_a_title_that_fits_keeps_its_lines():
    groups = {"vehicle": [1.0, 1.1], "10 µM": [2.0, 2.1]}
    spec = _fold_change(groups, title=_CHART_TITLE)
    title_box, _, figure, title, _ = _boxes_and_texts(spec)
    assert title.split("\n") == [_CHART_TITLE]
    assert 0 <= title_box.x0 and title_box.x1 <= figure.width


def test_render_handles_empty_group_without_crashing():
    # A condition with no values (all-None lanes) yields a slot with no bar; the
    # renderer must not raise on it (regression for the strict-zip crash).
    groups = {"a": [1.0, 2.0], "empty": []}
    spec = build_plotspec(
        groups, describe(groups), _test(groups), value_kind=ValueKind.LOADING_NORMALIZED
    )
    fig = render_figure(spec)
    assert len(fig.axes) == 1
