# SPDX-License-Identifier: Apache-2.0
"""Project-level assembly: pair per-box measurements to lanes.

A project quantifies several proteins against one shared set of lanes (the
condition / sample spine). Each protein's boxes are mapped to lanes by their
left-to-right position; this module keeps that mapping pure and GUI-independent.
"""

from __future__ import annotations

from collections.abc import Sequence


def lane_nets(
    boxes: Sequence[tuple[int, float]], lane_labels: Sequence[str]
) -> list[tuple[str, float]]:
    """Pair each box's net signal to a lane label by left-to-right position.

    ``boxes`` is a sequence of ``(x_left, net)``. Boxes are ordered by ``x_left``
    and zipped with ``lane_labels``; boxes beyond the available lanes are dropped,
    and lanes beyond the available boxes are simply absent.
    """
    ordered = sorted(boxes, key=lambda bn: bn[0])
    return [(lane_labels[i], net) for i, (_, net) in enumerate(ordered) if i < len(lane_labels)]


def align_to_lanes(
    proteins_boxes: Sequence[Sequence[tuple[int, float]]], n_lanes: int
) -> list[list[float | None]]:
    """Align every protein's boxes to a shared ``n_lanes``-column grid by position.

    Each protein is a sequence of ``(x, net)``. A single grid is built from the
    x-range across *all* proteins and split into ``n_lanes`` equal columns; every
    box is placed in its nearest column. Because placement is by position, a box
    at the second lane's x lands in column 1 even if column 0 was never filled for
    that protein, so missing-first / -middle / -last all resolve the same way.
    Columns with no box are ``None``; if two boxes of one protein map to the same
    column, the later one wins.

    Returns, per protein, a list of ``n_lanes`` cells. Requires ``n_lanes >= 1``.
    The outer lanes can only be anchored if some protein reaches them; otherwise
    the grid is a best-effort fit over the observed range.
    """
    if n_lanes < 1:
        raise ValueError("n_lanes must be >= 1")
    all_x = [x for boxes in proteins_boxes for (x, _) in boxes]
    rows: list[list[float | None]] = []
    if not all_x:
        return [[None] * n_lanes for _ in proteins_boxes]
    x_min, x_max = min(all_x), max(all_x)
    span = x_max - x_min
    for boxes in proteins_boxes:
        row: list[float | None] = [None] * n_lanes
        for x, net in boxes:
            col = 0 if (n_lanes == 1 or span == 0) else round((x - x_min) / span * (n_lanes - 1))
            row[max(0, min(n_lanes - 1, col))] = net
        rows.append(row)
    return rows
