# SPDX-License-Identifier: Apache-2.0
"""Geometry constraints for the ROI grid: lock every box to one shared size,
keep equal area, and forbid overlap.

These are pure functions over rectangles so the rules can be unit-tested
without a GUI. :func:`resize_all` enforces the shared size when it changes and
refuses a size that would force an overlap; :func:`normalize_corners` reads
napari shape vertices.

The placement rules of the napari app's box handlers are lifted here so the
project operations (:mod:`proteia.core.operations`) apply them too:
:func:`centered_rect` places a box of the shared size on a point, clamped into
the image; :func:`center_snap` reads an edited rect by its centre (napari's
``_center_snap``); :func:`initial_box_size` is a new protein's default size
(``_initial_size``); and :func:`grow_to_fit` fits the shared size to a
seed-grown band (``_seed_grow``), :func:`grow_to_fit_all` to several bands at
once (a detected row). Only one protein's own boxes must not overlap
(:func:`overlaps_any`); boxes of different proteins may. :func:`place_in_row`
places one row of same-size boxes left to right without overlap (row-box
detection, :mod:`proteia.core.rowdetect`).

Coordinates use the model's :data:`~proteia.core.model.Rect` convention:
``(x0, y0, x1, y1)`` in image pixels, half-open on the high edge, with the box
anchored at its top-left ``(x0, y0)`` corner.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Sequence
from typing import Literal

from proteia.core.model import BoxSize, Rect, overlaps

BoxRuleCode = Literal["overlap", "size_would_overlap"]


class BoxRuleError(ValueError):
    """A box change the no-overlap rule refuses; ``code`` says which, ``size`` is
    the shared size the change would take, and ``hits`` are the indices of the
    existing boxes the refusal is about (:func:`grow_to_fit_all`)."""

    def __init__(
        self,
        code: BoxRuleCode,
        message: str,
        *,
        size: BoxSize,
        hits: Sequence[int] = (),
    ) -> None:
        super().__init__(message)
        self.code: BoxRuleCode = code
        self.size: BoxSize = size
        self.hits: tuple[int, ...] = tuple(hits)


def normalize_corners(corners: Sequence[Sequence[float]]) -> Rect:
    """Convert napari rectangle vertices to a normalized rect.

    napari shape vertices are ``(row, col) = (y, x)`` and may be given in any
    corner order; return ``(x0, y0, x1, y1)`` with ``x0 <= x1`` and
    ``y0 <= y1``, rounded to integer pixels.
    """
    ys = [c[0] for c in corners]
    xs = [c[1] for c in corners]
    x0, x1 = round(min(xs)), round(max(xs))
    y0, y1 = round(min(ys)), round(max(ys))
    return (int(x0), int(y0), int(x1), int(y1))


def overlaps_any(rect: Rect, others: Iterable[Rect]) -> bool:
    """True if ``rect`` overlaps any of ``others``."""
    return any(overlaps(rect, other) for other in others)


def _overlaps_any(rects: Sequence[Rect], i: int) -> bool:
    """True if ``rects[i]`` overlaps any other rect."""
    return overlaps_any(rects[i], (b for j, b in enumerate(rects) if j != i))


def resize_all(
    rects: Sequence[Rect],
    size: BoxSize,
    *,
    width: int | None = None,
    height: int | None = None,
) -> list[Rect] | None:
    """Re-size every box to a new global ``size``, keeping each box *centred*.

    Enforces the "one shared size, changing it resizes all" invariant. Boxes grow
    or shrink around their own centre (so a box stays on the band it was placed
    on), not their top-left corner. ``width`` / ``height``, if given, clamp boxes
    inside the image. Returns the resized boxes, or ``None`` if the new size would
    force any overlap — in which case the caller should keep the old size.
    """
    resized = _recentred(rects, size, width, height)
    return None if _clashing(resized) else resized


def _recentred(
    rects: Sequence[Rect], size: BoxSize, width: int | None, height: int | None
) -> list[Rect]:
    """Each box re-sized to ``size`` around its centre, clamped into the image
    where ``width`` / ``height`` are given (:func:`resize_all`, unchecked)."""
    resized: list[Rect] = []
    for x0, y0, x1, y1 in rects:
        # Integer centre (not round(.../2)) so repeated resizes don't drift sideways.
        nx0 = (x0 + x1) // 2 - size.width // 2
        ny0 = (y0 + y1) // 2 - size.height // 2
        if width is not None:
            nx0 = max(0, min(nx0, width - size.width))
        if height is not None:
            ny0 = max(0, min(ny0, height - size.height))
        resized.append((nx0, ny0, nx0 + size.width, ny0 + size.height))
    return resized


def _clashing(rects: Sequence[Rect]) -> tuple[int, ...]:
    """The indices of the rects that overlap another of them."""
    return tuple(i for i in range(len(rects)) if _overlaps_any(rects, i))


def centered_rect(cx: int, cy: int, size: BoxSize, width: int, height: int) -> Rect:
    """A box of ``size`` centred on ``(cx, cy)``, shifted inside a ``width`` x
    ``height`` image where it would cross an edge."""
    x0 = max(0, min(cx - size.width // 2, width - size.width))
    y0 = max(0, min(cy - size.height // 2, height - size.height))
    return x0, y0, x0 + size.width, y0 + size.height


def center_snap(rect: Rect, size: BoxSize, width: int, height: int) -> Rect:
    """Read an edited ``rect`` by its integer centre: a box of ``size`` there,
    clamped into the image. A same-size rect inside the image maps to itself."""
    return centered_rect((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2, size, width, height)


def initial_box_size(width: int, height: int) -> BoxSize:
    """The default box size of a new protein on a ``width`` x ``height`` image:
    an eighth of the width and a twelfth of the height, at least 4 px, clamped to
    the image."""
    return BoxSize(width=min(width, max(4, width // 8)), height=min(height, max(4, height // 12)))


def place_in_row(targets: Sequence[int], width: int, lo: int, hi: int) -> list[int]:
    """Left edges for boxes of ``width`` along one row, in order, each as near its
    target left edge as the row allows.

    The result ``x0`` keeps ``x0[i + 1] >= x0[i] + width``, so the boxes never
    overlap whatever their y, and ``lo <= x0[0]``, ``x0[-1] + width <= hi``. It
    is bounded isotonic regression: the offsets ``targets[i] - i * width`` are
    pooled where they decrease (pool adjacent violators, least squares), rounded
    half up and clamped to ``[lo, hi - len(targets) * width]``, which keeps them
    non-decreasing. Integer arithmetic only: shifting the targets and the bounds
    by ``d`` shifts the result by exactly ``d``. Targets already in order and in
    bounds come back unchanged.

    Raises ValueError if the boxes cannot fit: ``len(targets) * width > hi - lo``.
    """
    top = hi - len(targets) * width
    if top < lo:
        raise ValueError(f"{len(targets)} boxes {width} px wide do not fit in [{lo}, {hi})")
    blocks: list[list[int]] = []  # pooled offsets as [sum, count]
    for i, target in enumerate(targets):
        blocks.append([operator.index(target) - i * width, 1])
        # Pool while the previous block's mean exceeds this one's.
        while len(blocks) > 1 and blocks[-2][0] * blocks[-1][1] > blocks[-1][0] * blocks[-2][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    lefts: list[int] = []
    for total, count in blocks:
        offset = min(max((2 * total + count) // (2 * count), lo), top)  # round half up, clamp
        start = len(lefts)
        lefts.extend(offset + (start + k) * width for k in range(count))
    return lefts


def grow_to_fit(
    rects: Sequence[Rect], size: BoxSize, grown: Rect, *, width: int, height: int
) -> tuple[BoxSize, list[Rect], Rect]:
    """Fit a protein's shared box size to a newly grown band and place its box:
    :func:`grow_to_fit_all` of the one band.

    ``rects`` are the protein's existing boxes, all of ``size``; ``grown`` is the
    band's fitted rect (e.g. from :func:`proteia.core.grow.grow_box`). The first
    box sets the size (at least 2 px, clamped to the image); later boxes only grow
    it, so every box of the protein keeps one area. Existing boxes are re-centred
    to the new size (:func:`resize_all`) and the new box is centred on the band.

    Returns ``(size, resized, rect)`` with ``resized`` in input order. Raises
    :class:`BoxRuleError` with ``size_would_overlap`` if the new size would make
    existing boxes overlap, or ``overlap`` if the new box overlaps one of them.
    """
    new, resized, [rect] = grow_to_fit_all(rects, size, [grown], width=width, height=height)
    return new, resized, rect


def grow_to_fit_all(
    rects: Sequence[Rect],
    size: BoxSize,
    grown: Sequence[Rect],
    *,
    width: int,
    height: int,
    need: BoxSize | None = None,
) -> tuple[BoxSize, list[Rect], list[Rect]]:
    """Fit a protein's shared box size to newly grown bands and place a box on
    each: :func:`grow_to_fit` for several bands at once (a detected row).

    ``rects`` are the protein's existing boxes, all of ``size``; ``grown`` are
    the bands' fitted rects. ``need`` is the size the bands need: by default the
    largest extent among ``grown`` in each dimension, each at least 2 px and
    clamped to the image. A caller whose bands share a size fitted elsewhere (a
    row's detector) passes it, and it counts even with no band to place. With
    no existing box, ``need`` is the size; otherwise it only grows the size, so
    every box of the protein keeps one area. Existing boxes are re-centred to
    the new size (:func:`resize_all`) and each new box is centred on its band's
    integer centre, shifted inside the image.

    Returns ``(size, resized, placed)``, both lists in input order. Raises
    :class:`BoxRuleError`, with the new ``size``: ``size_would_overlap`` if the
    new size would make existing boxes overlap (``hits``: those boxes) or the
    new boxes overlap each other (``hits`` empty), ``overlap`` if a new box
    would overlap an existing one (``hits``: the existing boxes overlapped).
    Raises ValueError with neither a grown band nor ``need``.
    """
    if need is None:
        if not grown:
            raise ValueError("fitting the box size takes a grown band or a size")
        need = BoxSize(
            width=max(min(width, max(2, x1 - x0)) for x0, _, x1, _ in grown),
            height=max(min(height, max(2, y1 - y0)) for _, y0, _, y1 in grown),
        )
    if rects:
        new = BoxSize(width=max(size.width, need.width), height=max(size.height, need.height))
    else:
        new = need
    resized = _recentred(rects, new, width, height)
    clashing = _clashing(resized)
    if clashing:
        raise BoxRuleError(
            "size_would_overlap",
            f"growing the box size to {new.width}x{new.height} would make boxes overlap",
            size=new,
            hits=clashing,
        )
    placed = [
        centered_rect((x0 + x1) // 2, (y0 + y1) // 2, new, width, height)
        for x0, y0, x1, y1 in grown
    ]
    if _clashing(placed):
        raise BoxRuleError(
            "size_would_overlap",
            f"the box size {new.width}x{new.height} would make the new boxes overlap each other",
            size=new,
        )
    hits = tuple(i for i, rect in enumerate(resized) if overlaps_any(rect, placed))
    if hits:
        which = "the new box" if len(placed) == 1 else "a new box"
        raise BoxRuleError(
            "overlap", f"{which} would overlap another box of this protein", size=new, hits=hits
        )
    return new, resized, placed
