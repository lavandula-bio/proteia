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
