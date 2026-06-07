# SPDX-License-Identifier: Apache-2.0
"""Project-level assembly: pair per-box measurements to lanes.

A project quantifies several proteins against one shared *lane spine* (the
condition / sample table). This module keeps that pairing pure and
GUI-independent.

Geometry vs. identity (locked 2026-06-07, "2b.0"). A box has *geometry* (its x
position) and an *identity* (which lane it is). Identity is the source of truth;
geometry is only one way to *propose* it. So the pipeline is three separable
parts:

* :func:`build_spine` — declare-first: from a condition structure, generate the
  N lanes (stable positions + auto sample ids). This fixes N before any box is
  drawn, so a missing box becomes an empty slot rather than a shift.
* :func:`propose_positions` — the left-to-right heuristic, demoted from
  source-of-truth to an *editable proposal* of each box's lane position.
* :func:`join_to_spine` — reads the *explicit* lane positions and scatters each
  protein's nets into the spine. A gap is a ``None`` slot that does not move its
  neighbours — this is what cures the position-inference scramble (see the legacy
  :func:`align_to_lanes` below, kept only as a no-spine fallback).

Downstream (reduce / stats) reads identity and never re-infers it. Future
auto-detect / OCR are just smarter proposers feeding the same explicit identity.
"""

from __future__ import annotations

from collections.abc import Sequence

from proteia.core.model import Lane

# A protein's net per lane, aligned to the spine; ``None`` marks a gap. Mirrors
# ``analyze.LaneNets`` without importing the stats layer (project stays upstream).
LaneNets = list[float | None]


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


def build_spine(declaration: Sequence[tuple[str, int]]) -> list[Lane]:
    """Generate the lane spine from a declared condition structure (declare-first).

    ``declaration`` is an ordered ``[(condition, count), ...]``: each entry adds
    ``count`` lanes for that condition, with positions assigned left-to-right in
    declaration order. Every lane gets a distinct auto sample id (condition +
    ordinal, e.g. ``ctl1``, ``ctl2``), i.e. *biological* replicates by default;
    technical repeats are created afterwards by giving two lanes the *same* sample
    id (then :func:`~proteia.core.analyze.reduce_samples` averages them, so they
    do not inflate n).

    Declaring N up front is the point: it fixes how many lanes exist before any
    box is drawn, so a missing box is an empty slot, not a shift. Conditions are
    contiguous here; a non-contiguous layout is expressed by editing the spine
    afterwards (its lanes are freely mutable). Condition labels must be distinct
    and every count must be >= 1.
    """
    labels = [cond for cond, _ in declaration]
    if len(set(labels)) != len(labels):
        raise ValueError("condition labels in a declaration must be distinct")
    if any(count < 1 for _, count in declaration):
        raise ValueError("every condition count must be >= 1")
    lanes: list[Lane] = []
    position = 0
    for cond, count in declaration:
        for ordinal in range(1, count + 1):
            lanes.append(Lane(index=position, label=cond, sample=f"{cond}{ordinal}"))
            position += 1
    return lanes


def spine_from_labels(labels: Sequence[str]) -> list[Lane]:
    """Build a spine from a per-lane condition list (the "assign in order" form).

    Where :func:`build_spine` takes ``(condition, count)`` and lays conditions out
    in contiguous blocks, this takes the already-expanded per-lane labels (e.g.
    ``["ctl", "A", "ctl"]``) and keeps their exact order, so non-contiguous
    layouts survive. Each lane's auto sample id is its condition plus a running
    ordinal *within that condition* (``ctl1`` ... ``ctl2`` even when interleaved),
    i.e. biological replicates by default; technical repeats are marked later by
    sharing a sample id.
    """
    counts: dict[str, int] = {}
    lanes: list[Lane] = []
    for position, label in enumerate(labels):
        counts[label] = counts.get(label, 0) + 1
        lanes.append(Lane(index=position, label=label, sample=f"{label}{counts[label]}"))
    return lanes


def propose_positions(boxes: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    """Propose a lane position for each box by left-to-right x order.

    ``boxes`` is ``(x, net)``; returns ``(position, net)`` with positions
    ``0, 1, 2, ...`` in ascending x. This is the #20 heuristic demoted to a
    *proposer*: its output is an editable suggestion of each box's identity, not
    the truth. Once a user corrects it (e.g. marks a slot empty), the explicit
    position is stored and :func:`join_to_spine` reads it directly — nothing is
    re-inferred on later refreshes, which is what the old grid inference got
    wrong when a box was missing.
    """
    ordered = sorted(boxes, key=lambda bn: bn[0])
    return [(i, net) for i, (_x, net) in enumerate(ordered)]


def join_to_spine(
    proteins_positioned: Sequence[Sequence[tuple[int, float]]], n_lanes: int
) -> list[LaneNets]:
    """Scatter each protein's nets into an ``n_lanes``-slot spine by explicit position.

    Each protein is a sequence of ``(position, net)`` where ``position`` is the
    box's stored lane identity (from :func:`propose_positions` or a user edit).
    The net is placed at exactly that slot; a slot with no box stays ``None`` and
    does **not** shift its neighbours. Positions outside ``[0, n_lanes)`` are
    dropped; if two boxes claim one slot the later one wins.

    Because placement reads identity instead of re-deriving it from geometry,
    deleting one box leaves every other lane's value exactly where it was — the
    cure for the inference scramble. Requires ``n_lanes >= 1``.
    """
    if n_lanes < 1:
        raise ValueError("n_lanes must be >= 1")
    rows: list[LaneNets] = []
    for boxes in proteins_positioned:
        row: LaneNets = [None] * n_lanes
        for position, net in boxes:
            if 0 <= position < n_lanes:
                row[position] = net
        rows.append(row)
    return rows


def spine_axes(spine: Sequence[Lane]) -> tuple[list[str], list[str | None], list[bool]]:
    """Unpack a spine into the parallel arrays the analysis layer consumes.

    Returns ``(conditions, samples, included)`` in lane-position order, ready to
    feed :class:`~proteia.core.analyze.Batch` (conditions) and
    :func:`~proteia.core.analyze.reduce_samples` (samples, included). Lanes are
    ordered by their stable ``index`` so the arrays line up with
    :func:`join_to_spine` output regardless of list order.
    """
    ordered = sorted(spine, key=lambda lane: lane.index)
    conditions = [lane.label for lane in ordered]
    samples = [lane.sample for lane in ordered]
    included = [lane.included for lane in ordered]
    return conditions, samples, included


def align_to_lanes(
    proteins_boxes: Sequence[Sequence[tuple[int, float]]], n_lanes: int
) -> list[list[float | None]]:
    """Align every protein's boxes to a shared ``n_lanes``-column grid by position.

    Legacy. Superseded by the explicit spine (:func:`build_spine` +
    :func:`join_to_spine`); kept only as the fallback when no spine exists, since
    its grid inference is what scrambles when a box is missing. Prefer the spine.

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
