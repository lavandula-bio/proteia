# SPDX-License-Identifier: Apache-2.0
"""Score proposed band boxes against reference boxes, lane by lane.

The measure of row-box detection (:mod:`proteia.core.rowdetect`): the share of
reference bands whose lane got a box overlapping it well enough. Pure apart from
reading a reference file; nothing is written and nothing is sent anywhere.

Reference file: CSV, UTF-8 with or without a byte-order mark. The first line is
the header ``lane,x,y,width,height`` (case and surrounding spaces ignored); then
one line per lane that has a band, in integers: ``x, y`` is the box's top-left
corner, ``width, height > 0``, ``x, y >= 0``, ``lane >= 0`` and each lane once.
Blank lines are skipped; a lane with no band is simply absent. A bad file raises
:class:`ReferenceFileError` naming the file and line (``name:line: ...``).

Matching is by lane only: a reference band is a hit when the box proposed for the
SAME lane covers it at IoU (intersection over union) of at least ``iou_min``,
and above 0 at any ``iou_min``: no box, or a box beside the band, is never a
hit. A box on a neighbouring lane never counts, and a box on a lane with no
reference band is a false positive. Rects are :data:`~proteia.core.model.Rect`:
``(x0, y0, x1, y1)`` in pixels, half-open on the high edge.
"""

from __future__ import annotations

import codecs
import csv
import io
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from proteia.core import rowdetect
from proteia.core.model import Rect

REFERENCE_COLUMNS: Final = ("lane", "x", "y", "width", "height")
IOU_MIN: Final = 0.5  # a hit needs at least this IoU with the same lane's box
_INT: Final = re.compile(r"[+-]?[0-9]+")


class ReferenceFileError(ValueError):
    """A reference CSV that breaks the file format; the message starts with
    ``name:line:``. (It shadows the builtin of the same name, which is about weak
    references, only where it is imported by name.)"""


@dataclass(frozen=True)
class HitRate:
    """The score of one row.

    ``hits`` of ``n_ref`` reference bands matched; ``rate`` is ``hits / n_ref``,
    NaN when there are no reference bands (nothing to find is not a perfect
    score). ``false_positive_lanes`` are the lanes, ascending, that got a box but
    have no reference band. ``iou`` maps each reference lane to the IoU of its
    lane's box, 0.0 where the lane got no box.
    """

    hits: int
    n_ref: int
    rate: float
    false_positive_lanes: tuple[int, ...]
    iou: dict[int, float]


def _area(r: Rect) -> int:
    return max(0, r[2] - r[0]) * max(0, r[3] - r[1])


def iou(a: Rect, b: Rect) -> float:
    """Intersection over union of two integer rects; 0.0 for rects that only
    touch at an edge or have no area."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0, inter_w) * max(0, inter_h)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def hit_rate(
    proposed: Mapping[int, Rect | None] | Sequence[Rect | None],
    reference: Mapping[int, Rect],
    *,
    iou_min: float = IOU_MIN,
) -> HitRate:
    """Score ``proposed`` boxes against ``reference`` boxes (see the module
    docstring). ``proposed`` maps lane to rect or None, or lists them in lane
    order (as :attr:`~proteia.core.rowdetect.RowDetection.slots`); a missing lane
    counts as no box. IoU exactly ``iou_min`` is a hit, IoU 0 never is."""
    boxes = dict(proposed) if isinstance(proposed, Mapping) else dict(enumerate(proposed))
    ious: dict[int, float] = {}
    hits = 0
    for lane, ref in reference.items():
        box = boxes.get(lane)
        value = 0.0 if box is None else iou(box, ref)
        ious[lane] = value
        if value > 0.0 and value >= iou_min:
            hits += 1
    false_positives = tuple(
        sorted(lane for lane, box in boxes.items() if box is not None and lane not in reference)
    )
    n_ref = len(reference)
    return HitRate(hits, n_ref, hits / n_ref if n_ref else math.nan, false_positives, ious)


def _read_text(path: Path) -> str:
    """The file's text: UTF-8, a leading byte-order mark dropped."""
    data = path.read_bytes().removeprefix(codecs.BOM_UTF8)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = data[: exc.start].count(b"\n") + 1
        raise ReferenceFileError(f"{path.name}:{line}: the file is not UTF-8 text") from exc


def read_reference_csv(path: Path) -> dict[int, Rect]:
    """The reference boxes of a CSV file (see the module docstring), by lane."""
    path = Path(path)
    name = path.name
    reader = csv.reader(io.StringIO(_read_text(path), newline=""))
    header = next(reader, None)
    if header is None or tuple(h.strip().lower() for h in header) != REFERENCE_COLUMNS:
        raise ReferenceFileError(f"{name}:1: the first line must be {','.join(REFERENCE_COLUMNS)}")
    boxes: dict[int, Rect] = {}
    for values in reader:
        line = reader.line_num
        cells = [c.strip() for c in values]
        if not any(cells):
            continue
        if len(cells) != len(REFERENCE_COLUMNS):
            raise ReferenceFileError(
                f"{name}:{line}: expected {len(REFERENCE_COLUMNS)} values, got {len(cells)}"
            )
        if not all(_INT.fullmatch(c) for c in cells):
            raise ReferenceFileError(f"{name}:{line}: values must be integers")
        lane, x, y, width, height = (int(c) for c in cells)
        if lane < 0 or x < 0 or y < 0:
            raise ReferenceFileError(f"{name}:{line}: lane, x and y must not be negative")
        if width <= 0 or height <= 0:
            raise ReferenceFileError(f"{name}:{line}: width and height must be positive")
        if lane in boxes:
            raise ReferenceFileError(f"{name}:{line}: lane {lane} is listed twice")
        boxes[lane] = (x, y, x + width, y + height)
    return boxes


def evaluate_row(
    gray: np.ndarray,
    row: Sequence[int],
    n_lanes: int,
    reference: Mapping[int, Rect],
    *,
    background: float,
    dark_on_light: bool,
    iou_min: float = IOU_MIN,
    size_rule: str = rowdetect.SIZE_RULE,
) -> HitRate:
    """Run :func:`~proteia.core.rowdetect.detect_row` on one row and score its
    slots against ``reference``. A refused detection (a refusing flag) proposes
    nothing, as the operation does, so every reference band of it is a miss.
    Raises :class:`~proteia.core.rowdetect.RowDetectError` as detection does."""
    found = rowdetect.detect_row(
        gray,
        row,
        n_lanes,
        background=background,
        dark_on_light=dark_on_light,
        size_rule=size_rule,
    )
    slots = (None,) * n_lanes if found.refused else found.slots
    return hit_rate(slots, reference, iou_min=iou_min)
