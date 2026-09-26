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
seed-grown band (``_seed_grow``). Only one protein's own boxes must not overlap
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
    """A box change the no-overlap rule refuses; ``code`` says which."""

    def __init__(self, code: BoxRuleCode, message: str) -> None:
        super().__init__(message)
        self.code: BoxRuleCode = code


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
    for i in range(len(resized)):
        if _overlaps_any(resized, i):
            return None
    return resized


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
    """Fit a protein's shared box size to a newly grown band and place its box.

    ``rects`` are the protein's existing boxes, all of ``size``; ``grown`` is the
    band's fitted rect (e.g. from :func:`proteia.core.grow.grow_box`). The first
    box sets the size (at least 2 px, clamped to the image); later boxes only grow
    it, so every box of the protein keeps one area. Existing boxes are re-centred
    to the new size (:func:`resize_all`) and the new box is centred on the band.

    Returns ``(size, resized, rect)`` with ``resized`` in input order. Raises
    :class:`BoxRuleError` with ``size_would_overlap`` if the new size would make
    existing boxes overlap, or ``overlap`` if the new box overlaps one of them.
    """
    gx0, gy0, gx1, gy1 = grown
    gw = min(width, max(2, gx1 - gx0))
    gh = min(height, max(2, gy1 - gy0))
    if rects:
        new = BoxSize(width=max(size.width, gw), height=max(size.height, gh))
    else:
        new = BoxSize(width=gw, height=gh)
    resized = resize_all(rects, new, width=width, height=height)
    if resized is None:
        raise BoxRuleError(
            "size_would_overlap",
            f"growing the box size to {new.width}x{new.height} would make boxes overlap",
        )
    rect = centered_rect((gx0 + gx1) // 2, (gy0 + gy1) // 2, new, width, height)
    if overlaps_any(rect, resized):
        raise BoxRuleError("overlap", "the new box would overlap another box of this protein")
    return new, resized, rect
