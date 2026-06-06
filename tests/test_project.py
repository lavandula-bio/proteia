# SPDX-License-Identifier: Apache-2.0
"""Tests for project-level lane pairing (GUI-independent)."""

import pytest

from proteia.core.project import align_to_lanes, lane_nets


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
