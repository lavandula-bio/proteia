# SPDX-License-Identifier: Apache-2.0
"""Geometry constraints for the ROI grid: lock every box to one shared size,
keep equal area, and forbid overlap.

These are pure functions over rectangles so the rules can be unit-tested
without a GUI. :func:`resize_all` enforces the shared size when it changes and
refuses a size that would force an overlap; :func:`normalize_corners` reads
napari shape vertices. (The napari app snaps a moved box and reverts an
overlapping move in its own edit handler.)

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
