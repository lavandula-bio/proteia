# SPDX-License-Identifier: Apache-2.0
"""Tests for project-level lane pairing (GUI-independent)."""

from proteia.core.project import lane_nets


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
