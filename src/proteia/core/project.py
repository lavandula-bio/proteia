# SPDX-License-Identifier: Apache-2.0
"""Project-level assembly: pair per-box measurements to lanes.

A project quantifies several proteins against one shared *lane spine* (the
condition / sample table). This module keeps that pairing pure and
GUI-independent.

Geometry vs. identity (locked 2026-06-07, "2b.0"). A box has *geometry* (its x
position) and an *identity* (which lane it is). Identity is the source of truth;
geometry is only one way to *propose* it. So the pipeline is three separable
parts:

* :func:`build_spine` (condition counts) — declare-first: generate the N lanes
  (stable positions + auto sample ids). This fixes N before any box is drawn, so
  a missing box becomes an empty slot rather than a shift.
* :func:`propose_lane` — position demoted from source of truth to a
  *proposal* of a new box's lane, anchored on the boxes already placed.
* :func:`join_to_spine` — reads the *explicit* lane positions and scatters each
  protein's nets into the spine. A gap is a ``None`` slot that does not move its
  neighbours — this is what cures the scramble that inferring every box's lane
  from its position caused when a box was missing.

Downstream (reduce / stats) reads identity and never re-infers it. Future
auto-detect / OCR are just smarter proposers feeding the same explicit identity.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Collection, Iterable, Sequence

from proteia.core.model import Batch, ImageRef, Lane

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


def propose_lane(x: float, anchors: Sequence[tuple[float, int]]) -> int | None:
    """Propose the lane of a box whose centre is at ``x``, or None when unsure.

    ``anchors`` are ``(centre x, stored lane index)`` of boxes already placed on
    the same image, of any protein: the lanes are the same columns of the
    membrane. Each lane's anchor is the median of its boxes' centres, and only
    anchors whose centres move steadily with the lane index are kept (the longest
    such run, left to right or, on a mirrored image, right to left), so a box
    dragged far from its lane is ignored. (A lane anchored by just two boxes, one
    of them dragged far, is anchored on their mean.) A proposal needs the lane
    pitch, so it needs two kept lanes; with fewer the answer is None and the lane
    must be chosen. An image's margins and its first lane's offset are never
    guessed.

    Between two neighbouring kept lanes the lane is interpolated, so uneven
    spacing (a smiling gel) is followed; elsewhere it steps from the nearest kept
    lane by the median pitch. The result is rounded and may lie outside the
    declared lanes: the caller checks it. It is only a proposal: once stored, a
    band's lane never follows its x again (:func:`join_to_spine`).
    """
    points, sign = _kept(anchors)
    if len(points) < 2:
        return None
    x *= sign
    pairs = list(itertools.pairwise(points))
    for (al, ax), (bl, bx) in pairs:
        if ax <= x <= bx:
            return math.floor(al + (x - ax) * (bl - al) / (bx - ax) + 0.5)
    lane, cx = min(points, key=lambda point: abs(point[1] - x))
    return math.floor(lane + (x - cx) / _pitch(pairs) + 0.5)


def lane_positions(
    anchors: Sequence[tuple[float, int]], lanes: Iterable[int], *, pitch: float | None = None
) -> dict[int, float]:
    """The expected centre x of each lane in ``lanes``: the inverse of
    :func:`propose_lane`, from the same kept anchors.

    Between two neighbouring kept lanes the x is interpolated; elsewhere it steps
    from the nearest kept lane by the median pitch. Empty with fewer than two kept
    lanes (no pitch). Rounding aside, ``propose_lane(lane_positions(a, [k])[k], a)
    == k``. For display, e.g. marking where a lane without a box lies.

    ``pitch`` (px per lane, positive; the kept lanes still set the direction)
    replaces the median pitch past the kept lanes: a pitch measured over more
    lanes than they span, since the step of two close lanes, repeated over many,
    drifts from where the far lanes lie.
    """
    points, sign = _kept(anchors)
    if len(points) < 2:
        return {}
    pairs = list(itertools.pairwise(points))
    if pitch is None:
        pitch = _pitch(pairs)
    positions = {}
    for lane in lanes:
        for (al, ax), (bl, bx) in pairs:
            if al <= lane <= bl:
                x = ax + (lane - al) * (bx - ax) / (bl - al)
                break
        else:
            nearest, nx = min(points, key=lambda point: abs(point[0] - lane))
            x = nx + (lane - nearest) * pitch
        positions[lane] = sign * x
    return positions


def anchoring_lanes(anchors: Sequence[tuple[float, int]], lanes: Iterable[int]) -> set[int]:
    """The kept lanes (:func:`propose_lane`) that :func:`lane_positions` reads
    the expected x of each of ``lanes`` from: a kept lane's own anchor, the two
    neighbouring kept lanes a lane lies between, or the nearest kept lane past
    them. Empty with fewer than two kept lanes, as the positions are."""
    points, _ = _kept(anchors)
    if len(points) < 2:
        return set()
    kept = [lane for lane, _ in points]
    found: set[int] = set()
    for lane in lanes:
        below = [k for k in kept if k <= lane]
        above = [k for k in kept if k >= lane]
        if below and above:
            found |= {below[-1], above[0]}
        else:
            found.add(min(kept, key=lambda k: abs(k - lane)))
    return found


def lanes_run_right_to_left(anchors: Sequence[tuple[float, int]]) -> bool | None:
    """Whether the kept anchors number the lanes right to left (a mirrored
    image), as :func:`propose_lane` and :func:`lane_positions` read them; None
    with fewer than two kept lanes, which show no direction."""
    points, sign = _kept(anchors)
    return None if len(points) < 2 else sign < 0


def lane_pitch(groups: Iterable[Sequence[tuple[float, int]]]) -> float | None:
    """The lane pitch (px per lane) that several groups of anchors show, e.g.
    each protein's boxes on an image: the median x step per lane between
    neighbouring kept anchors (:func:`propose_lane`) of each group, taken over
    every group. Each group is read on its own, so groups numbering the lanes
    opposite ways, or one a lane off, still show the spacing. None when no
    group has two kept lanes."""
    steps = []
    for anchors in groups:
        points, _ = _kept(anchors)
        steps.extend((bx - ax) / (bl - al) for (al, ax), (bl, bx) in itertools.pairwise(points))
    return statistics.median(steps) if steps else None


def _kept(anchors: Sequence[tuple[float, int]]) -> tuple[list[tuple[int, float]], int]:
    """The kept anchors as ``(lane, signed centre x)`` points, rising in both, and
    the sign (1, or -1 for lanes numbered right to left) of the signed x."""
    by_lane: dict[int, list[float]] = {}
    for cx, lane in anchors:
        by_lane.setdefault(lane, []).append(cx)
    medians = sorted((lane, statistics.median(xs)) for lane, xs in by_lane.items())
    rising = _rising(medians)
    falling = _rising([(lane, -cx) for lane, cx in medians])  # lanes numbered right to left
    if len(falling) > len(rising):
        return falling, -1
    return rising, 1


def _pitch(pairs: Sequence[tuple[tuple[int, float], tuple[int, float]]]) -> float:
    """The median x step per lane between neighbouring kept points."""
    return statistics.median((bx - ax) / (bl - al) for (al, ax), (bl, bx) in pairs)


def lane_anchors(
    batch: Batch,
    image: ImageRef,
    *,
    without: Collection[str] = (),
    only: Collection[str] | None = None,
) -> list[tuple[float, int]]:
    """``(centre x, stored lane)`` of the first bands on an image, of any protein:
    the anchors :func:`propose_lane` and :func:`lane_positions` take.

    A box touching the left or right edge is left out: it may have been shifted
    inside the image, so its centre need not be its lane's. So are the bands
    whose ids are in ``without`` (boxes a change is about to replace) and, when
    ``only`` is given, those whose ids are not in it.
    """
    return [(centre, lane) for _, centre, lane in _anchored(batch, image, without, only)]


def lane_anchor_ids(
    batch: Batch,
    image: ImageRef,
    *,
    without: Collection[str] = (),
    only: Collection[str] | None = None,
) -> list[str]:
    """The ids of the bands :func:`lane_anchors` takes with the same arguments,
    in its order."""
    return [band_id for band_id, _, _ in _anchored(batch, image, without, only)]


def _anchored(
    batch: Batch, image: ImageRef, without: Collection[str], only: Collection[str] | None
) -> list[tuple[str, float, int]]:
    """``(band id, centre x, stored lane)`` of each anchor (:func:`lane_anchors`)."""
    anchors = []
    for protein in batch.proteins:
        if protein.image_id == image.id:
            for band in protein.bands:
                x0, _, x1, _ = band.box.rect(protein.box_size)
                if (
                    band.band_index == 0
                    and x0 > 0
                    and x1 < image.width
                    and band.id not in without
                    and (only is None or band.id in only)
                ):
                    anchors.append((band.id, (x0 + x1) / 2, band.lane_index))
    return anchors


def _rising(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """The longest run of ``(lane, centre x)`` points, in lane order, whose centres
    strictly rise. Among equally long runs, the one whose lane pitches stay closest
    to the typical pitch (the median over every rising pair of points) wins, so a
    box dragged out of place is the one dropped, not a neighbour in place."""
    if not points:
        return []
    rising_pitches = [
        (xb - xa) / (lb - la) for (la, xa), (lb, xb) in itertools.combinations(points, 2) if xb > xa
    ]
    typical = statistics.median(rising_pitches) if rising_pitches else 1.0
    # best[i]: (run length, pitch misfit) of the best run ending at point i.
    best = [(1, 0.0)] * len(points)
    previous = [-1] * len(points)
    for i, (li, xi) in enumerate(points):
        for j, (lj, xj) in enumerate(points[:i]):
            if xj < xi:
                misfit = best[j][1] + abs(math.log((xi - xj) / (li - lj) / typical))
                if (best[j][0] + 1, -misfit) > (best[i][0], -best[i][1]):
                    best[i], previous[i] = (best[j][0] + 1, misfit), j
    i = min(range(len(points)), key=lambda k: (-best[k][0], best[k][1]))
    run = []
    while i != -1:
        run.append(points[i])
        i = previous[i]
    return run[::-1]


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
