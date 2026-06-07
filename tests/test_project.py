# SPDX-License-Identifier: Apache-2.0
"""Tests for project-level lane pairing (GUI-independent)."""

import pytest

from proteia.core.analyze import reduce_samples
from proteia.core.project import (
    align_to_lanes,
    build_spine,
    join_to_spine,
    lane_nets,
    propose_positions,
    spine_axes,
    spine_from_labels,
)


def test_pairs_by_left_to_right_order():
    # Boxes given out of click order; pairing follows x position.
    boxes = [(50, 8.0), (10, 3.0), (90, 5.0)]
    labels = ["A", "B", "C"]
    assert lane_nets(boxes, labels) == [("A", 3.0), ("B", 8.0), ("C", 5.0)]


def test_extra_boxes_beyond_lanes_dropped():
    boxes = [(10, 1.0), (20, 2.0), (30, 3.0)]
    assert lane_nets(boxes, ["only"]) == [("only", 1.0)]


def test_fewer_boxes_than_lanes():
    assert lane_nets([(10, 1.0)], ["A", "B", "C"]) == [("A", 1.0)]


def test_no_lanes_returns_empty():
    assert lane_nets([(10, 1.0)], []) == []


# --- align_to_lanes: position-based grid alignment ---

# A full protein at lanes x=10,20,30,40 anchors the 4-column grid for the others.
FULL = [(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0)]


def test_align_full_protein_maps_one_to_one():
    assert align_to_lanes([FULL], 4) == [[1.0, 2.0, 3.0, 4.0]]


def test_align_missing_middle_leaves_a_gap():
    missing_mid = [(10, 5.0), (20, 6.0), (40, 8.0)]  # no box at x=30 (lane 2)
    assert align_to_lanes([FULL, missing_mid], 4) == [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, None, 8.0],
    ]


def test_align_missing_first_does_not_shift():
    missing_first = [(20, 9.0), (30, 10.0), (40, 11.0)]  # no box at x=10 (lane 0)
    assert align_to_lanes([FULL, missing_first], 4) == [
        [1.0, 2.0, 3.0, 4.0],
        [None, 9.0, 10.0, 11.0],
    ]


def test_align_single_lane_puts_all_in_column_zero():
    assert align_to_lanes([[(10, 1.0)], [(99, 2.0)]], 1) == [[1.0], [2.0]]


def test_align_no_boxes_all_none():
    assert align_to_lanes([[], []], 3) == [[None, None, None], [None, None, None]]


def test_align_requires_positive_lanes():
    with pytest.raises(ValueError, match="n_lanes"):
        align_to_lanes([FULL], 0)


# --- build_spine: declare-first lane generation ---


def test_build_spine_generates_positions_and_auto_samples():
    spine = build_spine([("ctl", 2), ("A", 3)])
    assert [(lane.index, lane.label, lane.sample) for lane in spine] == [
        (0, "ctl", "ctl1"),
        (1, "ctl", "ctl2"),
        (2, "A", "A1"),
        (3, "A", "A2"),
        (4, "A", "A3"),
    ]


def test_build_spine_samples_are_distinct_biological_by_default():
    # Default = every lane its own sample => n equals the declared count, no
    # technical-repeat collapsing until the user marks shared samples.
    spine = build_spine([("A", 3)])
    assert len({lane.sample for lane in spine}) == 3


def test_build_spine_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="distinct"):
        build_spine([("A", 1), ("A", 2)])


def test_build_spine_rejects_nonpositive_count():
    with pytest.raises(ValueError, match=">= 1"):
        build_spine([("A", 0)])


# --- spine_from_labels: per-lane list preserves order, auto-numbers samples ---


def test_spine_from_labels_keeps_order_and_numbers_per_condition():
    spine = spine_from_labels(["ctl", "A", "ctl", "A"])  # interleaved
    assert [(lane.index, lane.label, lane.sample) for lane in spine] == [
        (0, "ctl", "ctl1"),
        (1, "A", "A1"),
        (2, "ctl", "ctl2"),  # running count is per condition, not per position
        (3, "A", "A2"),
    ]


def test_spine_from_labels_empty():
    assert spine_from_labels([]) == []


# --- propose_positions: x-order demoted to an editable proposal ---


def test_propose_positions_assigns_sequential_by_x():
    assert propose_positions([(50, 8.0), (10, 3.0), (90, 5.0)]) == [
        (0, 3.0),
        (1, 8.0),
        (2, 5.0),
    ]


# --- join_to_spine: scatter by explicit identity, gaps don't shift ---


def test_join_places_by_explicit_position():
    assert join_to_spine([[(0, 1.0), (1, 2.0), (2, 3.0)]], 3) == [[1.0, 2.0, 3.0]]


def test_join_gap_is_none_and_does_not_shift_neighbours():
    # Box at slot 2 is missing; slot 3 keeps its value (no shift). This is the
    # property align_to_lanes could not guarantee through re-inference.
    assert join_to_spine([[(0, 5.0), (1, 6.0), (3, 8.0)]], 4) == [[5.0, 6.0, None, 8.0]]


def test_join_drops_out_of_range_positions():
    assert join_to_spine([[(0, 1.0), (5, 9.0)]], 2) == [[1.0, None]]


def test_join_duplicate_position_later_wins():
    assert join_to_spine([[(0, 1.0), (0, 2.0)]], 1) == [[2.0]]


def test_join_requires_positive_lanes():
    with pytest.raises(ValueError, match="n_lanes"):
        join_to_spine([[(0, 1.0)]], 0)


def test_deleting_a_box_never_scrambles_other_lanes():
    # The §9.5 regression, end to end: explicit positions mean removing the box
    # at slot 1 only empties slot 1 — every other lane is byte-for-byte identical.
    before = join_to_spine([[(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]], 4)[0]
    after = join_to_spine([[(0, 1.0), (2, 3.0), (3, 4.0)]], 4)[0]
    assert before == [1.0, 2.0, 3.0, 4.0]
    assert after == [1.0, None, 3.0, 4.0]


# --- spine_axes: unpack for the analysis layer ---


def test_spine_axes_unpacks_in_position_order():
    spine = build_spine([("ctl", 2), ("A", 1)])
    conditions, samples, included = spine_axes(spine)
    assert conditions == ["ctl", "ctl", "A"]
    assert samples == ["ctl1", "ctl2", "A1"]
    assert included == [True, True, True]


def test_spine_axes_reflects_edits():
    spine = build_spine([("A", 2)])
    spine[1].included = False  # presentation-only lane excluded by the user
    _conditions, _samples, included = spine_axes(spine)
    assert included == [True, False]


# --- end to end: spine identity feeds reduce_samples correctly ---


def test_shared_sample_collapses_technical_repeats_to_n_one():
    # Three lanes of one condition, all the same biological sample (technical
    # repeats): reduce_samples must fold them to a single value, so n == 1 and
    # pseudoreplication cannot leak through the spine.
    spine = build_spine([("A", 3)])
    for lane in spine:
        lane.sample = "A1"  # user marked all three as the same sample
    nets = join_to_spine([[(0, 2.0), (1, 4.0), (2, 6.0)]], len(spine))[0]
    conditions, samples, included = spine_axes(spine)
    reduction = reduce_samples(nets, conditions, samples, included=included)
    assert reduction.groups == {"A": [4.0]}  # mean(2,4,6), one sample


def test_excluded_lane_drops_out_of_reduction():
    spine = build_spine([("A", 3)])
    spine[2].included = False
    nets = join_to_spine([[(0, 2.0), (1, 4.0), (2, 99.0)]], len(spine))[0]
    conditions, samples, included = spine_axes(spine)
    reduction = reduce_samples(nets, conditions, samples, included=included)
    assert reduction.groups == {"A": [2.0, 4.0]}  # excluded lane absent
