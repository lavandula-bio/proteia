# SPDX-License-Identifier: Apache-2.0
"""Geometry constraints for the ROI grid: lock every box to one shared size,
keep equal area, and forbid overlap.

These are pure functions over rectangles so the rules can be unit-tested
without a GUI. The napari layer calls :func:`reconcile` after every user edit
and writes the corrected boxes back — the "validate-and-correct" approach
(Route A): the user edits freely, then any edit that breaks an invariant is
snapped or reverted.

Coordinates use the model's :data:`~proteia.core.model.Rect` convention:
``(x0, y0, x1, y1)`` in image pixels, half-open on the high edge, with the box
anchored at its top-left ``(x0, y0)`` corner.
"""

from __future__ import annotations

from collections.abc import Sequence

from proteia.core.model import BoxSize, Rect, overlaps


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


def snap_to_size(rect: Rect, size: BoxSize) -> Rect:
    """Force a rect to the locked size, keeping its top-left (min) corner."""
    x0, y0, _, _ = rect
    return (x0, y0, x0 + size.width, y0 + size.height)


def reconcile(prev: Sequence[Rect], new: Sequence[Rect], size: BoxSize) -> list[Rect]:
    """Return a corrected box list that respects the locked size and no-overlap.

    ``prev`` is the last valid configuration; ``new`` is the user's edited rects
    (already normalized via :func:`normalize_corners`). Every box is snapped to
    the locked ``size`` first. Then any box that *changed* relative to ``prev``
    and now overlaps another is reverted to its previous position, or dropped if
    it was newly added (no previous position to fall back to).

    Assumes a single box changes per edit, which holds for interactive
    single-mouse editing in napari. Boxes are matched to ``prev`` by index;
    napari appends newly drawn shapes, so existing boxes keep their index.
    """
    snapped: list[Rect | None] = [snap_to_size(r, size) for r in new]
    for i in range(len(snapped)):
        changed = i >= len(prev) or snapped[i] != prev[i]
        if not changed:
            continue
        if _overlaps_any(snapped, i):
            snapped[i] = prev[i] if i < len(prev) else None  # revert, or drop a new box
    return [r for r in snapped if r is not None]


def _overlaps_any(rects: Sequence[Rect | None], i: int) -> bool:
    """True if ``rects[i]`` overlaps any other (non-dropped) rect."""
    a = rects[i]
    if a is None:
        return False
    return any(b is not None and overlaps(a, b) for j, b in enumerate(rects) if j != i)


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