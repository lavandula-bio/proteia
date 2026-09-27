# SPDX-License-Identifier: Apache-2.0
"""The statistics setting (#52): the menu of tests, the automatic rule that picks
one from the design, and the helpers scipy does not have (Holm, Dunn).

Every registered test is checked against a direct scipy call on the same data,
so a test id always means the computation its name says."""

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from proteia.core import analyze
from proteia.core.analyze import (
    ALPHA,
    AUTO_RULE_VERSION,
    DUNNETT_SEED,
    TESTS,
    StatisticsSetting,
    compare,
    describe,
    dunn,
    holm,
    min_attainable_p,
    plan_test,
    reduce_samples,
    replicate_lanes,
    run_test,
    statistics_setting,
)

F, C, S = analyze.TestFamily, analyze.TestComparisons, analyze.TestScale

EQ2 = {"ctl": [1.0, 1.2, 0.9], "A": [2.0, 2.3, 1.8]}
UNEQ2 = {"ctl": [1.0, 1.2, 0.9], "A": [2.0, 2.3]}
EQ3 = {"ctl": [1.0, 1.2, 0.9], "A": [2.0, 2.3, 1.8], "B": [0.6, 0.5, 0.7]}
UNEQ3 = {"ctl": [1.0, 1.2, 0.9], "A": [2.0, 2.3, 1.8], "B": [0.6, 0.5]}
ZERO3 = {"ctl": [1.0, 1.2, 0.9], "A": [0.0, 0.3, 0.2], "B": [0.6, 0.5, 0.7]}
SHORT_REF = {"ctl": [1.0], "A": [2.0, 2.3, 1.8], "B": [0.6, 0.5, 0.7], "C": [1.4, 1.5, 1.3]}


def _setting(family=F.AUTO, comparisons=C.AUTO, scale=S.AUTO):
    return StatisticsSetting(family=family, comparisons=comparisons, scale=scale)


# --- The setting ---


def test_the_setting_defaults_to_automatic_on_every_field():
    assert StatisticsSetting() == _setting()
    assert statistics_setting(None) == StatisticsSetting()


def test_the_setting_takes_raw_strings():
    raw = statistics_setting({"family": "welch", "scale": "linear"})
    assert raw == _setting(F.WELCH, C.AUTO, S.LINEAR)
    assert raw.family is F.WELCH  # the enum, not the string
    assert statistics_setting(raw) is raw


@pytest.mark.parametrize(
    "raw",
    [{"family": "anova"}, {"comparisons": "each"}, {"scale": "log10"}, {"test": "welch_t"}],
)
def test_an_unknown_key_or_value_is_a_value_error(raw):
    with pytest.raises(ValueError):
        statistics_setting(raw)


def test_the_setting_is_frozen():
    with pytest.raises(ValueError):
        StatisticsSetting().family = F.WELCH  # type: ignore[misc]


def test_compare_needs_the_value_kind_and_the_reference():
    # Defaults would let a caller test linear values over all pairs while the app
    # runs Dunnett's test on log values.
    with pytest.raises(TypeError):
        compare(EQ3, StatisticsSetting())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        compare(EQ3, StatisticsSetting(), ratio=True)  # type: ignore[call-arg]


# --- The automatic rule, and every explicit choice ---

# (setting, ratio, reference, groups) -> (test, family, comparisons, scale, reference, covered)
_PLANS = [
    # The automatic rule on the documented designs.
    ((), True, "ctl", EQ2, ("student_t", "pooled", "all_pairs", "log", None, ("ctl", "A"))),
    ((), True, "ctl", UNEQ2, ("welch_t", "welch", "all_pairs", "log", None, ("ctl", "A"))),
    ((), True, "ctl", EQ3, ("dunnett", "pooled", "vs_reference", "log", "ctl", ("ctl", "A", "B"))),
    (
        (),
        True,
        "ctl",
        UNEQ3,
        ("welch_t_holm", "welch", "vs_reference", "log", "ctl", ("ctl", "A", "B")),
    ),
    ((), True, None, EQ3, ("anova_tukey", "pooled", "all_pairs", "log", None, ("ctl", "A", "B"))),
    (
        (),
        True,
        None,
        UNEQ3,
        ("welch_anova_games_howell", "welch", "all_pairs", "log", None, ("ctl", "A", "B")),
    ),
    # The reference has one replicate: it is left out, and so is "vs the reference".
    (
        (),
        True,
        "ctl",
        SHORT_REF,
        ("anova_tukey", "pooled", "all_pairs", "log", None, ("A", "B", "C")),
    ),
    # Raw values are tested as they are; a ratio with a 0 falls back to linear.
    (
        (),
        False,
        "ctl",
        EQ3,
        ("dunnett", "pooled", "vs_reference", "linear", "ctl", ("ctl", "A", "B")),
    ),
    (
        (),
        True,
        "ctl",
        ZERO3,
        ("dunnett", "pooled", "vs_reference", "linear", "ctl", ("ctl", "A", "B")),
    ),
    # Each explicit choice.
    (
        (F.WELCH,),
        True,
        "ctl",
        EQ3,
        ("welch_t_holm", "welch", "vs_reference", "log", "ctl", ("ctl", "A", "B")),
    ),
    (
        (F.POOLED,),
        True,
        None,
        UNEQ3,
        ("anova_tukey", "pooled", "all_pairs", "log", None, ("ctl", "A", "B")),
    ),
    (
        (F.RANK,),
        True,
        "ctl",
        EQ3,
        ("dunn_holm", "rank", "vs_reference", None, "ctl", ("ctl", "A", "B")),
    ),
    (
        (F.RANK, C.ALL_PAIRS),
        True,
        "ctl",
        EQ3,
        ("kruskal_dunn_holm", "rank", "all_pairs", None, None, ("ctl", "A", "B")),
    ),
    ((F.RANK,), True, "ctl", EQ2, ("mann_whitney", "rank", "all_pairs", None, None, ("ctl", "A"))),
    (
        (F.AUTO, C.ALL_PAIRS),
        True,
        "ctl",
        EQ3,
        ("anova_tukey", "pooled", "all_pairs", "log", None, ("ctl", "A", "B")),
    ),
    # Two conditions vs the reference: the two-group test, the reference first.
    (
        (F.AUTO, C.VS_REFERENCE),
        True,
        "ctl",
        {"A": EQ2["A"], "ctl": EQ2["ctl"]},
        ("student_t", "pooled", "vs_reference", "log", "ctl", ("ctl", "A")),
    ),
    (
        (F.AUTO, C.AUTO, S.LINEAR),
        True,
        "ctl",
        EQ3,
        ("dunnett", "pooled", "vs_reference", "linear", "ctl", ("ctl", "A", "B")),
    ),
    (
        (F.AUTO, C.AUTO, S.LOG),
        False,
        "ctl",
        EQ3,
        ("dunnett", "pooled", "vs_reference", "log", "ctl", ("ctl", "A", "B")),
    ),
    # A rank test ignores the scale, even a log scale a 0 would refuse.
    (
        (F.RANK, C.AUTO, S.LOG),
        True,
        "ctl",
        ZERO3,
        ("dunn_holm", "rank", "vs_reference", None, "ctl", ("ctl", "A", "B")),
    ),
]


@pytest.mark.parametrize(("choice", "ratio", "reference", "groups", "expected"), _PLANS)
def test_the_plan_resolves_every_field(choice, ratio, reference, groups, expected):
    setting = _setting(*choice)
    plan = plan_test(groups, setting, ratio=ratio, reference=reference)
    test, family, comparisons, scale, planned_reference, covered = expected
    assert plan.test == test
    assert plan.name == TESTS[test].name
    assert (plan.family, plan.comparisons, plan.scale) == (family, comparisons, scale)
    assert plan.reference == planned_reference
    assert plan.covered == covered
    assert plan.no_test is None and plan.no_test_note is None
    fields = ("family", "comparisons", "scale")
    assert dict(plan.chosen) == {
        field: "auto" if getattr(setting, field) == "auto" else "user" for field in fields
    }
    # The test the plan names runs, over the conditions it covers.
    result = run_test(groups, plan)
    assert result.test == test and result.plan == plan
    assert {name for p in result.pairwise for name in (p.group_a, p.group_b)} <= set(covered)


def test_the_automatic_rule_gives_its_reasons():
    plan = plan_test(EQ3, StatisticsSetting(), ratio=True, reference="ctl")
    assert plan.reasons == (
        "ratios: log scale",
        "equal n: pooled variance",
        "a tested reference and 3 or more conditions: each condition vs the reference",
    )
    plan = plan_test(UNEQ2, _setting(F.WELCH), ratio=False, reference=None)
    assert plan.reasons == ("raw values: linear scale", "2 conditions: all pairs")


def test_a_ratio_with_a_value_of_0_is_tested_on_the_linear_scale_and_says_so():
    plan = plan_test(ZERO3, StatisticsSetting(), ratio=True, reference="ctl")
    assert plan.scale == "linear" and plan.nonpositive == ("A",)
    assert plan.notes == ("Linear scale: 'A' has a value of 0 or below",)


def test_a_value_of_0_in_a_condition_left_out_keeps_the_log_scale():
    # B's one value is 0, but B is not tested: only the tested values decide the scale.
    groups = {"ctl": EQ2["ctl"], "A": EQ2["A"], "B": [0.0]}
    plan = plan_test(groups, StatisticsSetting(), ratio=True, reference="ctl")
    assert (plan.test, plan.scale, plan.covered) == ("student_t", "log", ("ctl", "A"))
    assert (plan.nonpositive, plan.notes) == ((), ())
    expected = stats.ttest_ind(*_logs(groups, ["ctl", "A"]), equal_var=True).pvalue
    assert run_test(groups, plan).p_value == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("choice", "reference", "groups", "note"),
    [
        (
            (F.AUTO, C.VS_REFERENCE),
            None,
            EQ3,
            "no test: comparisons with the reference were chosen, but no reference is set",
        ),
        (
            (F.AUTO, C.VS_REFERENCE),
            "ctl",
            SHORT_REF,
            "no test: comparisons with the reference were chosen,"
            " but the reference 'ctl' has fewer than 2 replicates",
        ),
        (
            (F.AUTO, C.VS_REFERENCE),
            "KO",
            EQ3,
            "no test: comparisons with the reference were chosen,"
            " but the reference 'KO' is not among the conditions tested",
        ),
        (
            (F.AUTO, C.AUTO, S.LOG),
            "ctl",
            ZERO3,
            "no test: log values were chosen, but 'A' has a value of 0 or below",
        ),
    ],
)
def test_an_explicit_choice_that_cannot_apply_gives_no_test_and_says_why(
    choice, reference, groups, note
):
    plan = plan_test(groups, _setting(*choice), ratio=True, reference=reference)
    assert (plan.test, plan.no_test, plan.no_test_note) == (None, "not_applicable", note)
    result = run_test(groups, plan)
    assert (result.test, result.p_value, result.pairwise, result.note) == ("none", None, [], note)


def test_statistics_turned_off_give_no_test():
    plan = plan_test(EQ3, _setting(F.NONE), ratio=True, reference="ctl")
    assert (plan.test, plan.no_test) == (None, "statistics_off")
    assert plan.no_test_note == "no test: statistics are turned off"
    assert dict(plan.chosen)["family"] == "user"


def test_several_batches_are_not_tested_yet():
    blocks = {c: ["batch-1"] * len(v) for c, v in EQ3.items()}
    plan = plan_test(EQ3, StatisticsSetting(), ratio=True, reference="ctl", blocks=blocks)
    assert (plan.test, plan.no_test) == (None, "several_batches")
    assert plan.no_test_note == "no test: several batches are not supported yet"
    assert plan.design == "independent"  # the only design v0.1 tests


def test_too_few_testable_conditions_give_no_test():
    plan = plan_test(
        {"ctl": [1.0], "A": [2.0, 2.1]}, StatisticsSetting(), ratio=True, reference="ctl"
    )
    assert (plan.test, plan.no_test) == (None, "too_few")
    assert plan.no_test_note == "need >=2 groups with >=2 replicates for a test"


def test_a_condition_of_one_is_left_out_and_named():
    plan = plan_test(SHORT_REF, StatisticsSetting(), ratio=True, reference="ctl")
    assert plan.covered == ("A", "B", "C")
    assert plan.excluded == (analyze.ExcludedGroup("ctl", 1),)
    assert plan.excluded[0].reason == "fewer_than_2"


# --- Each registered test is scipy's (or, for Holm and Dunn, the textbook's) ---


def _logs(groups, labels):
    return [np.log(np.asarray(groups[label], dtype=float)) for label in labels]


def _run(groups, family, comparisons=C.AUTO, *, reference="ctl"):
    return compare(groups, _setting(family, comparisons), ratio=True, reference=reference)


def test_student_t_is_scipys_pooled_t_test():
    result = _run(EQ2, F.POOLED)
    expected = stats.ttest_ind(*_logs(EQ2, ["ctl", "A"]), equal_var=True)
    assert result.test == "student_t"
    assert result.p_value == pytest.approx(expected.pvalue, rel=1e-12)
    assert result.statistic == pytest.approx(expected.statistic, rel=1e-12)
    assert [p.p_value for p in result.pairwise] == [result.p_value]


def test_welch_t_is_scipys_welch_t_test():
    result = _run(UNEQ2, F.WELCH)
    expected = stats.ttest_ind(*_logs(UNEQ2, ["ctl", "A"]), equal_var=False)
    assert result.test == "welch_t"
    assert result.p_value == pytest.approx(expected.pvalue, rel=1e-12)
    assert result.statistic == pytest.approx(expected.statistic, rel=1e-12)


@pytest.mark.parametrize(
    "groups",
    [EQ2, {"ctl": [1.0, 2.0, 2.0, 3.0], "A": [2.0, 5.0, 6.0, 7.0]}],  # the second has ties
    ids=["untied", "tied"],
)
def test_mann_whitney_is_scipys_exact_permutation_test(groups):
    result = _run(groups, F.RANK)
    ctl, a = (np.asarray(groups[k]) for k in ("ctl", "A"))
    exact = stats.mannwhitneyu(ctl, a, method=stats.PermutationMethod(n_resamples=np.inf))
    assert result.test == "mann_whitney"
    assert result.p_value == pytest.approx(exact.pvalue, rel=1e-12)
    assert result.statistic == pytest.approx(exact.statistic)
    assert result.pairwise[0].estimate is None  # a rank test estimates no ratio
    assert result.method == ("exact" if groups is EQ2 else "exact permutation")


# 12 vs 12 with ties: comb(24, 12) = 2,704,156 arrangements, which the exact
# permutation takes about 15 s to count.
TIED_12 = {
    "ctl": [1.1, 1.2, 1.1, 0.7, 1.2, 1.1, 0.9, 1.1, 1.1, 1.1, 1.0, 1.1],
    "A": [1.2, 1.3, 1.2, 1.4, 1.3, 1.2, 1.1, 1.2, 1.3, 1.2, 1.6, 1.5],
}


def test_mann_whitney_with_too_many_tied_arrangements_is_the_normal_approximation():
    assert math.comb(24, 12) > analyze.MANN_WHITNEY_PERMUTATIONS
    result = _run(TIED_12, F.RANK)
    ctl, a = (np.asarray(TIED_12[k]) for k in ("ctl", "A"))
    normal = stats.mannwhitneyu(ctl, a, method="asymptotic")
    assert (result.test, result.method) == ("mann_whitney", "normal approximation")
    assert result.p_value == pytest.approx(normal.pvalue, rel=1e-12)
    assert result.p_value != pytest.approx(stats.mannwhitneyu(ctl, a, method="exact").pvalue)


def test_only_the_mann_whitney_test_names_how_its_p_was_computed():
    for family, comparisons in ((F.POOLED, C.AUTO), (F.WELCH, C.AUTO), (F.RANK, C.ALL_PAIRS)):
        assert _run(EQ3, family, comparisons).method is None


def test_anova_tukey_is_scipys_anova_and_tukey_kramer():
    result = _run(UNEQ3, F.POOLED, C.ALL_PAIRS)
    logs = _logs(UNEQ3, ["ctl", "A", "B"])
    anova, tukey = stats.f_oneway(*logs), stats.tukey_hsd(*logs)
    assert (result.test, result.omnibus) == ("anova_tukey", "One-way ANOVA")
    assert result.p_value == pytest.approx(anova.pvalue, rel=1e-12)
    assert result.statistic == pytest.approx(anova.statistic, rel=1e-12)
    pairs = itertools.combinations(range(3), 2)
    assert [p.p_value for p in result.pairwise] == pytest.approx(
        [tukey.pvalue[i, j] for i, j in pairs], rel=1e-12
    )


def test_welch_anova_games_howell_is_scipys():
    result = _run(UNEQ3, F.WELCH, C.ALL_PAIRS)
    logs = _logs(UNEQ3, ["ctl", "A", "B"])
    anova, howell = stats.f_oneway(*logs, equal_var=False), stats.tukey_hsd(*logs, equal_var=False)
    assert (result.test, result.omnibus) == ("welch_anova_games_howell", "Welch's ANOVA")
    assert result.p_value == pytest.approx(anova.pvalue, rel=1e-12)
    pairs = itertools.combinations(range(3), 2)
    assert [p.p_value for p in result.pairwise] == pytest.approx(
        [howell.pvalue[i, j] for i, j in pairs], rel=1e-12
    )


def test_with_two_conditions_welchs_anova_and_games_howell_are_welchs_t():
    logs = _logs(UNEQ2, ["ctl", "A"])
    welch = stats.ttest_ind(*logs, equal_var=False).pvalue
    assert stats.f_oneway(*logs, equal_var=False).pvalue == pytest.approx(welch, rel=1e-12)
    assert stats.tukey_hsd(*logs, equal_var=False).pvalue[0, 1] == pytest.approx(welch, rel=1e-9)


def test_dunnett_is_scipys_seeded_dunnett():
    result = _run(EQ3, F.POOLED)
    ctl, a, b = _logs(EQ3, ["ctl", "A", "B"])
    expected = stats.dunnett(a, b, control=ctl, rng=np.random.default_rng(DUNNETT_SEED))
    assert (result.test, result.p_value, result.statistic, result.omnibus) == (
        "dunnett",
        None,
        None,
        None,
    )
    assert [(p.group_a, p.group_b) for p in result.pairwise] == [("ctl", "A"), ("ctl", "B")]
    assert [p.p_value for p in result.pairwise] == list(expected.pvalue)


def test_dunnett_gives_the_same_p_every_time():
    plan = plan_test(EQ3, _setting(F.POOLED), ratio=True, reference="ctl")
    runs = set()
    for _ in range(3):
        analyze.clear_test_cache()  # computed anew each time, not answered from the cache
        runs.add(tuple(p.p_value for p in run_test(EQ3, plan).pairwise))
    assert len(runs) == 1


def test_welch_t_holm_is_welchs_t_vs_the_reference_holm_adjusted():
    result = _run(UNEQ3, F.WELCH)
    ctl, a, b = _logs(UNEQ3, ["ctl", "A", "B"])
    raw = [stats.ttest_ind(ctl, x, equal_var=False).pvalue for x in (a, b)]
    assert result.test == "welch_t_holm" and result.p_value is None
    assert [p.p_value for p in result.pairwise] == pytest.approx(holm(raw), rel=1e-12)


def test_kruskal_dunn_holm_is_scipys_kruskal_and_dunn():
    result = _run(EQ3, F.RANK, C.ALL_PAIRS)
    samples = [np.asarray(EQ3[k]) for k in ("ctl", "A", "B")]
    kruskal = stats.kruskal(*samples)
    assert (result.test, result.omnibus) == ("kruskal_dunn_holm", "Kruskal-Wallis")
    assert result.p_value == pytest.approx(kruskal.pvalue, rel=1e-12)
    assert result.statistic == pytest.approx(kruskal.statistic, rel=1e-12)
    pairs = list(itertools.combinations(range(3), 2))
    assert [p.p_value for p in result.pairwise] == pytest.approx(dunn(samples, pairs), rel=1e-12)


def test_dunn_holm_is_dunn_vs_the_reference():
    result = _run(EQ3, F.RANK)
    samples = [np.asarray(EQ3[k]) for k in ("ctl", "A", "B")]
    assert result.test == "dunn_holm" and result.p_value is None
    expected = dunn(samples, [(0, 1), (0, 2)])
    assert [p.p_value for p in result.pairwise] == pytest.approx(expected, rel=1e-12)


def test_holm_on_a_known_vector():
    # Sorted: 0.005 x 4 = 0.02, 0.01 x 3 = 0.03, 0.03 x 2 = 0.06, 0.04 x 1 -> 0.06.
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.6]) == [1.0, 1.0]  # never above 1
    assert holm([]) == []


@pytest.mark.parametrize("p_values", [[math.nan, 0.01], [0.02, math.nan, 0.001]])
def test_holm_gives_no_p_values_when_one_is_nan(p_values):
    # A NaN sorts anywhere; unguarded, Holm would turn it into an adjusted p of 0.
    adjusted = holm(p_values)
    assert len(adjusted) == len(p_values) and all(math.isnan(p) for p in adjusted)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
def test_welch_t_holm_with_a_comparison_of_no_p_gives_no_p_values():
    # ctl and A hold one value each: their Welch t is NaN, so no comparison has a p.
    groups = {"ctl": [1.0, 1.0, 1.0], "A": [1.0, 1.0], "B": [2.0, 3.0, 4.0]}
    result = compare(groups, StatisticsSetting(), ratio=True, reference="ctl")
    assert result.test == "welch_t_holm"
    assert [(p.group_a, p.group_b) for p in result.pairwise] == [("ctl", "A"), ("ctl", "B")]
    assert all(math.isnan(p.p_value) for p in result.pairwise)


def test_dunn_on_a_hand_worked_example_with_ties():
    samples = [np.array([1.0, 2.0, 2.0]), np.array([2.0, 3.0, 4.0]), np.array([5.0, 6.0, 6.0])]
    # Ranks: 1 -> 1; the three 2s -> 3; 3 -> 5; 4 -> 6; 5 -> 7; the two 6s -> 8.5.
    # Mean ranks 7/3, 14/3 and 8. Ties: (27 - 3) + (8 - 2) = 30, so the variance
    # factor is 9 * 10 / 12 - 30 / (12 * 8) = 7.1875, times 1/3 + 1/3.
    se = math.sqrt(7.1875 * 2 / 3)
    raw = [2 * stats.norm.sf(diff / se) for diff in (7 / 3, 17 / 3, 10 / 3)]  # AB, AC, BC
    # Holm: AC x 3, BC x 2, then AB x 1 (not below the one before it).
    expected = [max(raw[0], 2 * raw[2]), 3 * raw[1], 2 * raw[2]]
    assert dunn(samples, [(0, 1), (0, 2), (1, 2)]) == pytest.approx(expected, rel=1e-12)


def test_the_smallest_attainable_p_of_the_rank_tests():
    assert min_attainable_p([3, 3], "mann_whitney", 1) == pytest.approx(0.1)
    assert min_attainable_p([4, 4], "mann_whitney", 1) == pytest.approx(2 / 70)  # 0.02857
    # Dunn: the reference and one condition at the ends, the rest tied in the middle.
    three = min_attainable_p([2, 2, 2], "kruskal_dunn_holm", 3)
    assert three == pytest.approx(3 * 2 * stats.norm.sf(4 / math.sqrt((3.5 - 18 / 60) * 1.0)))
    assert three >= ALPHA  # 2 x 2 x 2 can never reach p < 0.05
    assert min_attainable_p([3, 3, 3], "dunn_holm", 2) < ALPHA
    with pytest.raises(KeyError):
        min_attainable_p([3, 3], "student_t", 1)


def test_a_rank_test_that_cannot_reach_alpha_says_so():
    groups = {"ctl": [1.0, 1.1, 0.9], "A": [2.0, 2.1, 1.9]}
    plan = plan_test(groups, _setting(F.RANK), ratio=True, reference="ctl")
    assert plan.min_attainable_p == pytest.approx(0.1)
    assert plan.notes == (
        "Mann-Whitney U test: with these n no p can be below 0.1,"
        " so no comparison can reach p < 0.05",
    )
    run = compare(groups, _setting(F.RANK), ratio=True, reference="ctl")
    assert run.p_value == pytest.approx(0.1)  # it still runs


@pytest.mark.parametrize(
    ("comparisons", "test", "count"),
    [(C.VS_REFERENCE, "dunn_holm", 2), (C.ALL_PAIRS, "kruskal_dunn_holm", 3)],
)
def test_the_rank_floor_counts_the_comparisons_the_test_makes(comparisons, test, count):
    # Three conditions: 2 comparisons with the reference, 3 pairs. Holm multiplies
    # the smallest p by the count, so the count moves the floor.
    plan = plan_test(EQ3, _setting(F.RANK, comparisons), ratio=True, reference="ctl")
    assert plan.test == test
    assert plan.min_attainable_p == pytest.approx(min_attainable_p([3, 3, 3], test, count))
    assert plan.notes == ()  # 3 x 3 x 3 can reach p < 0.05 either way


def test_the_rank_floor_note_vs_the_reference_counts_its_comparisons():
    groups = {"ctl": [1.0, 1.1], "A": [2.0, 2.1], "B": [3.0, 3.1]}
    plan = plan_test(groups, _setting(F.RANK), ratio=True, reference="ctl")
    assert plan.test == "dunn_holm"
    assert plan.min_attainable_p == pytest.approx(min_attainable_p([2, 2, 2], "dunn_holm", 2))
    assert plan.notes == (
        "Dunn's test (Holm): with these n no p can be below 0.051,"
        " so no comparison can reach p < 0.05",
    )


# --- Scale, estimates, determinism ---


@pytest.mark.parametrize("family", [F.POOLED, F.WELCH, F.RANK], ids=lambda family: family.value)
@pytest.mark.parametrize("comparisons", [C.ALL_PAIRS, C.VS_REFERENCE], ids=["pairs", "ref"])
@pytest.mark.parametrize("groups", [EQ2, UNEQ3], ids=["k2", "k3"])
def test_a_common_factor_leaves_every_p_unchanged(family, comparisons, groups):
    setting = _setting(family, comparisons)
    base = compare(groups, setting, ratio=True, reference="ctl")
    scaled = {k: [v * 7.5 for v in vals] for k, vals in groups.items()}
    moved = compare(scaled, setting, ratio=True, reference="ctl")
    assert moved.test == base.test
    assert [p.p_value for p in moved.pairwise] == pytest.approx(
        [p.p_value for p in base.pairwise], rel=1e-9
    )
    assert (moved.p_value is None) == (base.p_value is None)
    if base.p_value is not None:
        assert moved.p_value == pytest.approx(base.p_value, rel=1e-9)


def test_the_estimate_on_log_values_is_the_ratio_of_geometric_means():
    result = compare(EQ3, StatisticsSetting(), ratio=True, reference="ctl")
    geo = {k: math.exp(np.mean(np.log(v))) for k, v in EQ3.items()}
    for pair in result.pairwise:
        assert pair.estimate == pytest.approx(geo[pair.group_b] / geo[pair.group_a], rel=1e-12)


def test_the_estimate_on_linear_values_is_the_difference_of_means():
    result = compare(EQ3, _setting(scale=S.LINEAR), ratio=True, reference="ctl")
    for pair in result.pairwise:
        expected = np.mean(EQ3[pair.group_b]) - np.mean(EQ3[pair.group_a])
        assert pair.estimate == pytest.approx(expected, rel=1e-12)


# --- The registry ---


def test_the_registry_holds_every_test_with_stable_ids():
    assert set(TESTS) == {
        "student_t",
        "welch_t",
        "mann_whitney",
        "anova_tukey",
        "welch_anova_games_howell",
        "kruskal_dunn_holm",
        "dunnett",
        "welch_t_holm",
        "dunn_holm",
    }
    for test_id, spec in TESTS.items():
        assert spec.id == test_id
        assert spec.family in (F.POOLED, F.WELCH, F.RANK)
    assert TESTS["dunnett"].adjustment == "Dunnett"
    assert TESTS["anova_tukey"].adjustment == "Tukey-Kramer"
    assert TESTS["welch_anova_games_howell"].adjustment == "Games-Howell"
    assert TESTS["dunn_holm"].adjustment == TESTS["welch_t_holm"].adjustment == "Holm"
    assert TESTS["student_t"].adjustment == "none"
    assert TESTS["kruskal_dunn_holm"].omnibus == "Kruskal-Wallis"
    assert AUTO_RULE_VERSION == 1


# --- Descriptive statistics and replicates ---


def test_describe_gives_the_geometric_mean_and_sd_factor():
    [g] = describe({"a": [1.0, 4.0]})
    assert g.geometric_mean == pytest.approx(2.0)
    assert g.geometric_sd_factor == pytest.approx(math.exp(np.std(np.log([1.0, 4.0]), ddof=1)))
    [one] = describe({"a": [3.0]})
    assert one.geometric_mean == pytest.approx(3.0) and one.geometric_sd_factor is None


@pytest.mark.parametrize("values", [[0.0, 1.0], [-1.0, 2.0], []])
def test_describe_has_no_geometric_mean_for_a_value_of_0_or_below(values):
    [g] = describe({"a": values})
    assert (g.geometric_mean, g.geometric_sd_factor) == (None, None)


def test_replicate_lanes_keys_replicates_as_the_reduction_does():
    conditions = ["a", "a", "a", "a", "b", "b"]
    samples = ["s1", "s1", None, "2", "s1", ""]
    included = [True, True, True, True, True, False]
    values = [1.0, None, 3.0, None, 5.0, 6.0]
    lanes = replicate_lanes(conditions, samples, included)
    # Every included lane, with a value or not; an unnamed lane is its own sample.
    assert lanes == {"a": [[0, 1], [2], [3]], "b": [[4]]}
    red = reduce_samples(values, conditions, samples, included=included)
    with_values = {
        c: [[i for i in rep if values[i] is not None] for rep in reps] for c, reps in lanes.items()
    }
    assert red.lanes == {c: [rep for rep in reps if rep] for c, reps in with_values.items()}
