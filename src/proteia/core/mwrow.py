# SPDX-License-Identifier: Apache-2.0
"""Where a protein's row lies by its expected molecular weight (#58, D11, D12).

Pure: it reads the model and an image's calibration (:mod:`proteia.core.mwcal`)
and writes nothing. :func:`~proteia.core.operations.detect_mw_row` detects the
row in the slot this module predicts, and the page draws it.

The slot (D11). The rows searched for a protein expected at m kDa, with the
MW check's tolerance t: from m (1 + 2t) down to m / (1 + 2t)
(:data:`SEARCH_FACTOR` tolerances either way, within the calibrated range),
plus a margin above and below of :data:`SLOT_MARGIN_DECADES` of a decade of MW
there (half a typical band), at least :data:`SLOT_MIN_PX` px, all read at the
centre of the lanes' span. The slot follows the protein line (D1): at each
column of the span it lies as many whole pixels lower as the line at m does
there, ``round(y(m, x + 0.5) - y(m, centre))``, which the detector levels
(:func:`~proteia.core.rowdetect.detect_row_along`). The line at one MW is
straight in x, so y is read from the calibration (``curve_at``) at the span's
first and last columns and interpolated between them. With one ladder the line
is level and so is the slot. Its rows are cut to those every column's shift keeps
on the image. A line sloping more than :data:`STEEP_ROW_DEG` across the span is
noted: the boxes stay level rectangles.

The lanes' span (D12), the first of: the span the user dragged (x only); the
lanes already placed on the image, the first-band boxes of every protein less
this protein's boxes a detector placed that nobody edited (the row replaces
them), from half a lane pitch outside the first lane to half a pitch outside
the last; those of the image of its register group with the most lanes placed
(linked images match pixel for pixel); between the group's two ladders, each
taken to stand one lane pitch outside its end lane, inset by half that pitch;
else none, and the user drags across the lanes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import JsonValue

from proteia.core import mwcal
from proteia.core.model import DETECTING_SOURCES, Batch, Protein, Rect
from proteia.core.project import (
    anchoring_lanes,
    lane_anchor_ids,
    lane_anchors,
    lane_pitch,
    lane_positions,
)

# D11: the slot reaches SEARCH_FACTOR tolerances either way of the expected MW,
# plus SLOT_MARGIN_DECADES of a decade of MW above and below (half a typical
# band: 12 px at 300 px per decade), at least SLOT_MIN_PX px.
SEARCH_FACTOR: Final = 2.0
SLOT_MARGIN_DECADES: Final = 0.04
SLOT_MIN_PX: Final = 6
# A protein line sloping more than this across the span (4.8 px within a 92-px
# box) is noted: the boxes are level rectangles.
STEEP_ROW_DEG: Final = 3.0

SpanFrom = Literal["given", "anchors", "group_anchors", "ladders"]


class SlotError(ValueError):
    """A slot that cannot be placed. ``code``: ``outside_range`` (the expected
    MW has no position at the span: outside the calibrated range, or where the
    protein line between two ladders folds over), or ``out_of_image`` (no row
    of the slot stays on the image at every column of the span)."""

    def __init__(self, code: Literal["outside_range", "out_of_image"], message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LaneSpan:
    """The columns of the declared lanes, ``x0`` to ``x1 - 1``, and where they
    were read (D12). ``anchor_ids`` are the first-band boxes whose lanes gave
    it, on the image ``image_id`` (``anchors``, ``group_anchors``)."""

    x0: int
    x1: int
    source: SpanFrom
    anchor_ids: tuple[str, ...] = ()
    image_id: str | None = None


@dataclass(frozen=True)
class Slot:
    """Where a protein's row is searched (D11): the rows ``y0`` to ``y1 - 1``
    at the span's centre, ``shifts[x - x0]`` rows lower at column x."""

    mws: tuple[float, ...]  # the expected MWs, top to bottom
    x0: int
    x1: int
    centre: float  # the span's centre, x
    m_top: float  # the slot's MW ends, before the margin
    m_bot: float
    margin: float  # px above m_top's row and below m_bot's
    y0: int
    y1: int
    expected_y: tuple[float, ...]  # each expected MW's y at the centre
    shifts: tuple[int, ...]  # per column of the span
    slope_deg: float  # the protein line's slope at the top MW; positive: lower to the right

    @property
    def row(self) -> Rect:
        """The slot at the span's centre, as a row box ``(x0, y0, x1, y1)``."""
        return self.x0, self.y0, self.x1, self.y1

    @property
    def shift_ends(self) -> tuple[int, int]:
        """How many rows lower than at the centre the slot lies at the span's
        first and last columns."""
        return self.shifts[0], self.shifts[-1]

    @property
    def steep(self) -> bool:
        return abs(self.slope_deg) > STEEP_ROW_DEG

    def shift_at(self, x: float) -> int:
        """The shift of the column holding ``x``, or of the span's nearest end
        column past it."""
        return self.shifts[min(max(math.floor(x) - self.x0, 0), len(self.shifts) - 1)]


def _shifts(
    curves: tuple[mwcal.LadderCurve, mwcal.LadderCurve],
    z: float,
    x0: int,
    x1: int,
    at_centre: float,
) -> tuple[tuple[int, ...], float]:
    """Each column's shift at log10 MW ``z``, ``round(y(x + 0.5) - at_centre)``,
    and the line's slope there in degrees, from ``curves``: the calibration's
    curves at the span's first and last column (``curve_at``). The protein line
    at one MW is straight in x (D1), so y is read at those two columns and
    interpolated between them; with one ladder they are one curve, and every
    shift is 0."""
    first, last = x0 + 0.5, x1 - 0.5
    y_first, y_last = curves[0].y_at_log(z), curves[1].y_at_log(z)
    if y_first is None or y_last is None:  # never: the range does not depend on x
        raise SlotError("outside_range", f"{10.0**z:g} kDa lies outside the calibrated range")
    if x1 - x0 == 1:
        return (round(y_first - at_centre),), 0.0
    step = (y_last - y_first) / (last - first)
    shifts = tuple(round(y_first + step * (x - x0) - at_centre) for x in range(x0, x1))
    return shifts, math.degrees(math.atan2(y_last - y_first, last - first))


def _mw_at(z: float) -> float:
    """``10 ** z``; inf past the largest float, which only a ladder labelled
    near it reaches."""
    try:
        return 10.0**z
    except OverflowError:
        return math.inf


def slot(
    fitted: mwcal.Calibration,
    mws: Sequence[float],
    tolerance: float,
    x0: int,
    x1: int,
    height: int,
) -> Slot:
    """The slot of a protein expected at ``mws`` (kDa, top to bottom) with the
    MW check's ``tolerance`` (a share), across the span ``x0`` to ``x1 - 1`` of
    an image ``height`` rows high (see the module docstring). The MWs lie in
    the calibrated range (the caller checked). :class:`SlotError` where the
    line folds over within the span, or where no row stays on the image."""
    mws = tuple(float(m) for m in mws)
    centre = 0.5 * (x0 + x1)
    c0 = fitted.curve_at(centre)
    first, last = fitted.curve_at(x0 + 0.5), fitted.curve_at(x1 - 0.5)
    if c0 is None or first is None or last is None:
        # Folds occur only outside the ladders; a span with both end columns
        # clear of them is clear throughout (each breakpoint is linear in x).
        raise SlotError(
            "outside_range",
            f"the protein line folds over within x={x0}..{x1 - 1}: the two ladders disagree"
            " too strongly there to place a row by its MW",
        )
    zs = [math.log10(m) for m in mws]
    expected = [c0.y_at_log(z) for z in zs]
    if any(y is None for y in expected):
        raise SlotError("outside_range", f"{mws[0]:g} kDa lies outside the calibrated range")
    ys = tuple(y for y in expected if y is not None)
    widen = math.log10(1.0 + SEARCH_FACTOR * tolerance)
    z_top = min(zs[0] + widen, fitted.z_hi)
    z_bot = max(zs[-1] - widen, fitted.z_lo)
    y_top, y_bot = c0.y_at_log(z_top), c0.y_at_log(z_bot)
    if y_top is None or y_bot is None:  # never: both lie in the range
        raise SlotError("outside_range", f"{mws[0]:g} kDa lies outside the calibrated range")
    margin = max(float(SLOT_MIN_PX), SLOT_MARGIN_DECADES * c0.px_per_decade(ys[0]))
    shifts, slope = _shifts((first, last), zs[0], x0, x1, ys[0])
    y0 = max(math.floor(y_top - margin), -min(shifts), 0)
    y1 = min(math.ceil(y_bot + margin), height - max(shifts), height)
    if y1 <= y0:
        raise SlotError(
            "out_of_image",
            f"the rows where {mws[0]:g} kDa is searched lie off the image across x={x0}..{x1 - 1}",
        )
    return Slot(
        mws=mws,
        x0=x0,
        x1=x1,
        centre=centre,
        m_top=_mw_at(z_top),
        m_bot=_mw_at(z_bot),
        margin=margin,
        y0=y0,
        y1=y1,
        expected_y=ys,
        shifts=shifts,
        slope_deg=slope,
    )


def span_between_ladders(fitted: mwcal.Calibration, n: int, width: int) -> tuple[int, int] | None:
    """The span of ``n`` lanes between a group's two ladders, clipped to the
    image's ``width``: each ladder taken to stand one lane pitch outside its
    end lane, ``pitch = (x_right - x_left) / (n + 1)``, and the span inset by
    half of it, so each end lane's centre lies half a pitch inside. None with
    one ladder or no lanes."""
    if not fitted.two_ladders or n < 1:
        return None
    left, right = fitted.ladders
    if left.x is None or right.x is None:
        return None
    pitch = (right.x - left.x) / (n + 1)
    x0 = max(0, math.floor(left.x + pitch / 2))
    x1 = min(width, math.ceil(right.x - pitch / 2))
    return (x0, x1) if x0 < x1 else None


def span_of_anchors(
    anchors: Sequence[tuple[float, int]], n: int, width: int
) -> tuple[int, int] | None:
    """The span of ``n`` lanes that the lanes already placed show
    (``anchors``, :func:`~proteia.core.project.lane_anchors`): half their
    pitch outside the first and the last lane's expected centre
    (:func:`~proteia.core.project.lane_positions`), clipped to the image's
    ``width``. None with fewer than two kept lanes, or no lanes."""
    if n < 1:
        return None
    pitch = lane_pitch([anchors])
    positions = lane_positions(anchors, [0, n - 1], pitch=pitch)
    if pitch is None or not positions:
        return None
    x0 = max(0, math.floor(min(positions.values()) - pitch / 2))
    x1 = min(width, math.ceil(max(positions.values()) + pitch / 2))
    return (x0, x1) if x0 < x1 else None


def _kept_lanes(anchors: Sequence[tuple[float, int]]) -> int:
    """How many lanes ``anchors`` keep (:func:`~proteia.core.project.propose_lane`):
    each kept lane anchors itself, and a lane not kept its kept neighbours."""
    return len(anchoring_lanes(anchors, {lane for _, lane in anchors}))


def lane_span(batch: Batch, protein: Protein, fitted: mwcal.Calibration) -> LaneSpan | None:
    """The span of the declared lanes on the protein's image, from the lanes
    already placed there, else on the image of its register group with the
    most lanes placed (the first of them in membrane order), else between its
    two ladders; None when none shows it (see the module docstring)."""
    n = len(batch.lanes)
    if n < 1:
        return None
    image = batch.find_image(protein.image_id)
    replaced = {
        band.id
        for band in protein.bands
        if band.band_index == 0 and not band.manually_edited and band.source in DETECTING_SOURCES
    }
    anchors = lane_anchors(batch, image, without=replaced)
    span = span_of_anchors(anchors, n, image.width)
    if span is not None:
        ids = tuple(lane_anchor_ids(batch, image, without=replaced))
        return LaneSpan(*span, "anchors", ids, image.id)
    membrane = batch.membrane_of(image.id)
    group = membrane.group_of(image.id)
    best: tuple[int, str, list[tuple[float, int]]] | None = None
    for other in membrane.images:
        if other.id == image.id or other.id not in group:
            continue
        found = lane_anchors(batch, other)
        kept = _kept_lanes(found)
        if kept >= 2 and (best is None or kept > best[0]):
            best = (kept, other.id, found)
    if best is not None:
        _, other_id, found = best
        span = span_of_anchors(found, n, image.width)
        if span is not None:
            ids = tuple(lane_anchor_ids(batch, batch.find_image(other_id)))
            return LaneSpan(*span, "group_anchors", ids, other_id)
    span = span_between_ladders(fitted, n, image.width)
    return None if span is None else LaneSpan(*span, "ladders")


def placed_on_add(batch: Batch, protein: Protein) -> bool:
    """Whether adding the protein places its row by its MW at once (D12): it
    expects one band at a known MW, and its image has a curve."""
    if protein.expected_mw is None or protein.expected_band_count != 1:
        return False
    fitted = mwcal.calibration_for(batch.membrane_of(protein.image_id), protein.image_id)
    return isinstance(fitted, mwcal.Calibration)


@dataclass(frozen=True)
class Prediction:
    """Where a protein's row is predicted before it is placed: each expected
    MW's y at the span's centre (None outside the range there), and the slot,
    with the span it was read from, where they can be placed (None where the
    span is unknown or the slot cannot be placed)."""

    mws: tuple[float, ...]
    expected_y: tuple[float | None, ...]
    span: LaneSpan | None
    slot: Slot | None


def predict(batch: Batch, protein: Protein) -> Prediction | None:
    """The protein's predicted row, as :func:`~proteia.core.operations.detect_mw_row`
    would search it with no span given; None without an expected MW, for a
    protein expecting several bands, or on an image without a curve. Without
    a span, each MW's y is read midway between two ladders, or anywhere on a
    level line."""
    if not placed_on_add(batch, protein) or protein.expected_mw is None:
        return None
    image = batch.find_image(protein.image_id)
    fitted = mwcal.calibration_for(batch.membrane_of(image.id), image.id)
    if not isinstance(fitted, mwcal.Calibration):
        return None
    mws = (protein.expected_mw,)
    span = lane_span(batch, protein, fitted)
    found: Slot | None = None
    if span is not None and all(fitted.z_lo <= math.log10(m) <= fitted.z_hi for m in mws):
        try:
            found = slot(fitted, mws, protein.mw_tolerance, span.x0, span.x1, image.height)
        except SlotError:
            found = None
    if found is not None:
        return Prediction(mws, found.expected_y, span, found)
    xs = [ladder.x for ladder in fitted.ladders if ladder.x is not None]
    x: float | None = None
    if span is not None:
        x = 0.5 * (span.x0 + span.x1)
    elif fitted.two_ladders and len(xs) == 2:
        x = 0.5 * (xs[0] + xs[1])
    curve = fitted.curve_at(x)
    ys = tuple(None if curve is None else curve.y_at_log(math.log10(m)) for m in mws)
    return Prediction(mws, ys, span, None)


def _kda_at(z: float) -> str:
    """The MW ``10 ** z`` (a range end) as refusals write it: whole kDa, one
    decimal below 10 kDa, in significant digits far past any protein; ∞ past
    the largest float, which only a ladder labelled near it reaches."""
    mw = _mw_at(z)
    if not math.isfinite(mw):
        return "∞"
    if mw >= 1e6 or mw < 0.1:
        return f"{mw:.3g}"
    return str(round(mw)) if mw >= 10.0 else f"{mw:.1f}".removesuffix(".0")


def _past(shown: str, end: float, above: bool) -> bool:
    """Whether the MW written ``shown`` reads above (``above``) or below the
    range end ``end``."""
    value = float(shown)
    return value > end if above else value < end


def _end_words(z: float, mw: str, above: bool) -> str:
    """The range end ``10 ** z`` that the refused MW written ``mw`` lies past
    (above it where ``above``): as :func:`_kda_at` writes it where that shows
    ``mw`` past it, else with the fewest more significant digits that do."""
    words = _kda_at(z)
    end = _mw_at(z)
    if not math.isfinite(end) or end <= 0.0 or _past(mw, float(words), above):
        return words
    far = end >= 1e6 or end < 0.1
    first = 4 if far else max(2, math.floor(math.log10(end)) + 2)
    for digits in range(first, 18):
        words = f"{end:.{digits}g}"
        if _past(mw, float(words), above):
            return words
    return words


def outside_words(fitted: mwcal.Calibration, mw: float, names: str) -> str:
    """Why ``mw`` (kDa, outside the calibrated range of ``names``) is refused,
    naming the range heaviest first as the notices do (``40 kDa lies outside
    the calibrated range of img-1, img-3 (126–56 kDa)``), and with two ladders,
    where both reach. The end ``mw`` lies past gets the digits it takes to read
    so (``126 kDa ... (125.9–56 kDa)``), and ``mw`` its own where ``{mw:g}``
    would round it onto the range."""
    above = math.log10(mw) > fitted.z_hi
    end = _mw_at(fitted.z_hi if above else fitted.z_lo)
    shown = f"{mw:g}"
    if not _past(shown, end, above):
        shown = repr(mw).removesuffix(".0")
    hi, lo = _kda_at(fitted.z_hi), _kda_at(fitted.z_lo)
    if above:
        hi = _end_words(fitted.z_hi, shown, above=True)
    else:
        lo = _end_words(fitted.z_lo, shown, above=False)
    where = ", where both ladders reach" if fitted.two_ladders else ""
    return f"{shown} kDa lies outside the calibrated range of {names} ({hi}–{lo} kDa{where})"


def settings() -> dict[str, JsonValue]:
    """How a row is placed by its MW (#58, D11, D12), JSON-plain: what an
    export record reports under ``mw``."""
    return {
        "search_factor": SEARCH_FACTOR,
        "slot_margin_decades": SLOT_MARGIN_DECADES,
        "slot_min_px": SLOT_MIN_PX,
        "slot": (
            "from the expected MW x (1 + search_factor x tolerance) down to it / (1 +"
            " search_factor x tolerance), within the calibrated range, plus"
            " slot_margin_decades of a decade (at least slot_min_px px) above and below, at"
            " the centre of the lanes' span"
        ),
        "along_the_line": (
            "each column of the span shifted by whole pixels, round(y(mw, x + 0.5) - y(mw,"
            " centre)), so the protein line lies level for detection; boxes moved back by"
            " the shift of their centre column"
        ),
        "steep_row_deg": STEEP_ROW_DEG,
        "seed": (
            "each lane grown from its peak nearest the expected MW's row, the lane's own"
            " peaks among them, stopping at the valley to any other band; a few lanes off"
            " the row's line and the expected MW's row, with no other band on the line"
            " there and the other boxes on that row, recorded as not detected"
        ),
        "lane_span": (
            "the span dragged; else the lanes placed on the image, half a pitch past the end"
            " lanes; else those of the image of its register group with the most lanes"
            " placed; else between two ladders, each one pitch outside its end lane, inset"
            " by half a pitch"
        ),
    }
