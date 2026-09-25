# SPDX-License-Identifier: Apache-2.0
"""Project-level assembly: pair per-box measurements to lanes.

A project quantifies several proteins against one shared *lane spine* (the
condition / sample table). This module keeps that pairing pure and
GUI-independent.

Geometry vs. identity (locked 2026-06-07, "2b.0"). A box has *geometry* (its x
position) and an *identity* (which lane it is). Identity is the source of truth;
geometry is only one way to *propose* it. So the pipeline is three separable
parts:

* :func:`spine_from_labels` (per-lane condition list, used by the napari app) or
  :func:`build_spine` (condition counts) — declare-first: generate the N lanes
  (stable positions + auto sample ids). This fixes N before any box is drawn, so
  a missing box becomes an empty slot rather than a shift.
* :func:`propose_lane` — position demoted from source of truth to a
  *proposal* of a new box's lane, anchored on the boxes already placed.
* :func:`join_to_spine` — reads the *explicit* lane positions and scatters each
  protein's nets into the spine. A gap is a ``None`` slot that does not move its
  neighbours — this is what cures the position-inference scramble (see the legacy
  :func:`align_to_lanes` below, kept only as a no-spine fallback).

Downstream (reduce / stats) reads identity and never re-infers it. Future
auto-detect / OCR are just smarter proposers feeding the same explicit identity.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Sequence

from proteia.core.model import Lane

# A protein's net per lane, aligned to the spine; ``None`` marks a gap. Mirrors
# ``analyze.LaneNets`` without importing the stats layer (project stays upstream).
LaneNets = list[float | None]


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

    The napari app builds its spine with :func:`spine_from_labels` instead; this
    form is kept for a lane table that is declared as condition counts.
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


def propose_lane(x: float, anchors: Sequence[tuple[float, int]]) -> int | None:
    """Propose the lane of a box whose centre is at ``x``, or None when unsure.

    ``anchors`` are ``(centre x, stored lane index)`` of the boxes already placed
    on the same image, of any protein: the lanes are the same columns of the
    membrane. A proposal needs the lane pitch, so it needs boxes in at least two
    lanes whose centres rise with the lane index; with fewer the answer is None
    and the lane must be chosen. An image's margins and its first lane's offset
    are never guessed.

    Each lane's anchor is the median of its boxes' centres. Between two anchored
    lanes whose centres rise, the lane is interpolated, so uneven spacing (a
    smiling gel) is followed; elsewhere it steps from the nearest anchored lane
    by the typical pitch (the median of the rising neighbour pitches), so one box
    dragged far from its lane skews only its own neighbourhood. The result is
    rounded and may lie outside the declared lanes: the caller checks it. It is
    only a proposal: once stored, a band's lane never follows its x again
    (:func:`join_to_spine`).
    """
    by_lane: dict[int, list[float]] = {}
    for cx, lane in anchors:
        by_lane.setdefault(lane, []).append(cx)
    points = sorted((lane, statistics.median(xs)) for lane, xs in by_lane.items())
    pairs = list(itertools.pairwise(points))
    pitches = [(bx - ax) / (bl - al) for (al, ax), (bl, bx) in pairs if bx > ax]
    if not pitches:
        return None
    for (al, ax), (bl, bx) in pairs:
        if ax <= x <= bx and bx > ax:
            return math.floor(al + (x - ax) * (bl - al) / (bx - ax) + 0.5)
    lane, cx = min(points, key=lambda point: abs(point[1] - x))
    return math.floor(lane + (x - cx) / statistics.median(pitches) + 0.5)


def join_to_spine(
    proteins_positioned: Sequence[Sequence[tuple[int, float]]], n_lanes: int
) -> list[LaneNets]:
    """Scatter each protein's nets into an ``n_lanes``-slot spine by explicit position.

    Each protein is a sequence of ``(position, net)`` where ``position`` is the
    box's stored lane identity (from :func:`propose_lane` or a user edit).
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
