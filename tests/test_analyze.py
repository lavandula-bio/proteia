# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the batch analysis layer, on synthetic nets (no images)."""

import math

import pytest

from proteia.core.analyze import (
    Batch,
    ProteinNets,
    ReduceMethod,
    Role,
    Tier,
    assess,
    compare,
    describe,
    fold_change_lane,
    group_by_condition,
    normalize_lane,
    reduce_samples,
)


def _batch(control_condition=None, with_loading=True, with_target=True):
    # 8 lanes: 3 conditions, replicates 3 / 3 / 2.
    conditions = ["ctl", "ctl", "ctl", "A", "A", "A", "B", "B"]
    proteins = []
    if with_target:
        proteins.append(
            ProteinNets("p53", Role.TARGET, [100, 110, 105, 200, 210, 190, 50, 60])
        )
    if with_loading:
        proteins.append(
            ProteinNets("GAPDH", Role.LOADING_CONTROL, [100, 100, 100, 100, 100, 100, 100, 100])
        )
    return Batch(conditions, proteins, control_condition=control_condition)


# --- ladder / compliance ---


def test_tier_export_only_without_loading_control():
    c = assess(_batch(with_loading=False))
    assert c.tier is Tier.EXPORT_ONLY
    assert not c.can_pool
    assert any("loading control" in w for w in c.warnings)


def test_tier_export_only_without_target():
    c = assess(_batch(with_target=False))
    assert c.tier is Tier.EXPORT_ONLY
    assert not c.can_pool


def test_tier_normalized_with_target_and_loading():
    c = assess(_batch())
    assert c.tier is Tier.NORMALIZED
    assert c.can_pool
    assert c.warnings == []


def test_tier_fold_change_with_control_condition():
    c = assess(_batch(control_condition="ctl"))
    assert c.tier is Tier.FOLD_CHANGE
    assert c.can_pool


# --- normalization ---


def test_normalize_lane_divides_by_loading():
    assert normalize_lane([100, 200, None], [100, 100, 100]) == [1.0, 2.0, None]


def test_normalize_lane_guards_against_bad_loading():
    # missing or non-positive loading -> None (cannot divide)
    assert normalize_lane([100, 100], [0, None]) == [None, None]


def test_normalize_lane_length_mismatch_raises():
    with pytest.raises(ValueError):
        normalize_lane([1, 2], [1])


# --- grouping (replicates = shared label) ---


def test_group_by_condition_pools_replicates():
    groups = group_by_condition([1, 2, 3, 4], ["a", "a", "b", "b"])
    assert groups == {"a": [1, 2], "b": [3, 4]}


def test_group_by_condition_skips_none_but_keeps_group():
    groups = group_by_condition([1, None, 3], ["a", "a", "b"])
    assert groups == {"a": [1], "b": [3]}


def test_group_preserves_left_to_right_order():
    groups = group_by_condition([1, 2, 3], ["b", "a", "b"])
    assert list(groups) == ["b", "a"]


# --- fold change ---


def test_fold_change_relative_to_control_mean():
    # control mean = 2; values become value / 2.
    fc = fold_change_lane([2, 2, 4, 8], ["c", "c", "x", "x"], "c")
    assert fc == [1.0, 1.0, 2.0, 4.0]


def test_fold_change_missing_control_raises():
    with pytest.raises(ValueError):
        fold_change_lane([1, 2], ["x", "x"], "c")


# --- sample reduction (technical vs biological replicates) ---


def test_reduce_no_samples_treats_each_lane_as_biological():
    # default: every lane is its own sample -> same as plain grouping, n = lanes
    r = reduce_samples([1, 2, 3, 4], ["a", "a", "b", "b"])
    assert r.groups == {"a": [1, 2], "b": [3, 4]}
    assert r.averaged == []


def test_reduce_averages_technical_repeats_per_sample():
    # condition a: sample s1 loaded twice (100, 120) + sample s2 once (80).
    # technical repeat s1 -> mean 110; biological n for a = 2 (s1, s2), not 3.
    r = reduce_samples(
        [100, 120, 80, 50],
        ["a", "a", "a", "b"],
        ["s1", "s1", "s2", "s3"],
    )
    assert r.groups["a"] == [110.0, 80.0]
    assert len(r.groups["a"]) == 2  # n counts samples, not lanes
    assert r.averaged == [("a", "s1")]
    assert r.warnings  # transparency about the averaging


def test_reduce_representative_keeps_first_repeat():
    r = reduce_samples(
        [100, 120, 80],
        ["a", "a", "a"],
        ["s1", "s1", "s2"],
        method=ReduceMethod.REPRESENTATIVE,
    )
    assert r.groups["a"] == [100.0, 80.0]  # first lane of s1, not the mean


def test_reduce_excludes_presentation_only_lanes():
    r = reduce_samples(
        [100, 999, 80],
        ["a", "a", "b"],
        ["s1", "s2", "s3"],
        included=[True, False, True],  # middle lane is presentation-only
    )
    assert r.groups == {"a": [100.0], "b": [80.0]}


def test_reduce_prevents_pseudoreplication_in_stats():
    # Three technical repeats of ONE sample per condition must NOT read as n=3.
    r = reduce_samples(
        [10, 11, 12, 20, 21, 22],
        ["a", "a", "a", "b", "b", "b"],
        ["a1", "a1", "a1", "b1", "b1", "b1"],
    )
    assert [len(v) for v in r.groups.values()] == [1, 1]  # n=1 each
    res = compare(r.groups)
    assert res.test == "none"  # cannot test n=1 groups -> no fake significance


# --- descriptive stats ---


def test_describe_mean_sd_sem():
    [g] = describe({"a": [2.0, 4.0, 6.0]})
    assert g.n == 3
    assert g.mean == 4.0
    assert math.isclose(g.sd, 2.0)  # sample sd of 2,4,6
    assert math.isclose(g.sem, 2.0 / math.sqrt(3))


def test_describe_singleton_has_zero_error():
    [g] = describe({"a": [5.0]})
    assert g.n == 1
    assert g.sd == 0.0 and g.sem == 0.0


# --- inferential stats ---


def test_compare_two_groups_uses_welch_t():
    res = compare({"a": [1, 2, 3], "b": [10, 11, 12]})
    assert res.test == "welch_t"
    assert res.p_value is not None and res.p_value < 0.05
    assert len(res.pairwise) == 1


def test_compare_three_groups_uses_anova_with_posthoc():
    res = compare({"a": [1, 2, 3], "b": [10, 11, 12], "c": [20, 21, 22]})
    assert res.test == "anova_oneway"
    assert res.p_value is not None and res.p_value < 0.05
    assert len(res.pairwise) == 3  # all pairs


def test_compare_too_few_replicates_returns_no_test():
    res = compare({"a": [1], "b": [2]})
    assert res.test == "none"
    assert res.p_value is None
    assert res.note


# --- end-to-end: batch -> normalized groups -> stats ---


def test_full_chain_normalize_group_compare():
    batch = _batch()
    target = batch.targets()[0].nets
    loading = batch.loading_control().nets
    norm = normalize_lane(target, loading)
    groups = group_by_condition(norm, batch.conditions)
    assert set(groups) == {"ctl", "A", "B"}
    assert [len(v) for v in groups.values()] == [3, 3, 2]
    res = compare(groups)
    assert res.test == "anova_oneway"
    assert res.p_value < 0.05  # ctl~1, A~2, B~0.55 are clearly different
