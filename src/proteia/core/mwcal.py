# SPDX-License-Identifier: Apache-2.0
"""Molecular-weight calibration (#58): each ladder's curve, and the protein line
between two ladders.

Pure: it reads :mod:`proteia.core.model` objects and arrays it is given and
never writes them, and opens no file; only :func:`refine_point` reads pixels
(with numpy). Positions are continuous coordinates of an image's analysis array
(:class:`~proteia.core.model.CalibrationPoint`): pixel row r covers [r, r+1), so
a band peaking on row r lies at y = r + 0.5, as a box centre does.

Register groups. The images of a membrane whose pixel rows line up, a
chemiluminescence image and the marker image it is linked to, form a register
group (:meth:`~proteia.core.model.Membrane.register_groups`). Each group is
fitted from its own points (D4); a group has one ladder, or two.

One ladder. log10(MW) is linear in y between neighbouring points, and the end
segments extend a tenth of a decade past a ladder band and not at all past a
strip edge (D6). With two points this is the least-squares line. The curve does
not depend on x: the protein line is level.

Two ladders: the protein line (D1). A group may hold a second ladder, right of
the first. At each MW a band's y is then linear in x between the two ladders,
each at the median x of its points, so a tilted blot needs no rotation. Lanes
outside the ladders extrapolate the line. The range is where both ladders
reach: a MW marked on one side only is not extrapolated to the other. The two
ladders combine only when they share at least :data:`MIN_SHARED_MWS` marked
MWs; otherwise the right one is not used. The offsets between the ladders,
which give the tilt and the ladders' disagreement, are taken at the MWs both
hold: elsewhere one side's value would be an interpolation.

Quality (D2), per ladder: take one interior band away and predict it from the
bands above and below it; the largest relative MW disagreement.

Every range test compares log10(MW), never a MW computed back from it:
``10 ** log10(75)`` need not be 75, and a strip cut at 75 kDa must hold 75 kDa.

Snapping a click (:func:`refine_point`). A click on a ladder band or a strip
edge is moved to the band's peak, or the edge, nearest it within a small
window, which stops short of the points already marked on the same ladder, so
a click never snaps onto a band that is marked already. With nothing there
that stands out of the noise, the click is kept as it is.
"""

from __future__ import annotations

import bisect
import itertools
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from pydantic import JsonValue

from proteia.core.model import (
    CalibrationPoint,
    CalibrationPointSource,
    FitMethod,
    LadderSide,
    Membrane,
    Polarity,
)

EXTRAPOLATE_DECADES: Final = 0.1  # D6: past an outermost ladder band (x/÷ 1.26 in MW)
FIT_WARN: Final = 0.20  # D2: a ladder whose quality exceeds this is flagged
LADDERS_WARN: Final = 0.05  # D1: the two ladders disagree beyond this
MIN_LADDER_POINTS: Final = 2  # a side with fewer is not used
MIN_SHARED_MWS: Final = 2  # two ladders combine only when they share this many MWs
DEFAULT_FIT_METHOD: Final = FitMethod.LOG_LINEAR_PIECEWISE

# Snapping a click (refine_point): the window, px either side of the click across
# the lane and above and below it; how far towards a point already marked on the
# same ladder the window reaches, as a share of the distance to it; and how many
# noise sigmas a band's peak (or an edge's step) must reach.
SNAP_HALF_X: Final = 12
SNAP_HALF_Y: Final = 12
SNAP_MARKED_GAP: Final = 0.45
SNAP_K: Final = 4.0
# A strip edge clicked within this many px of the image's top or bottom is the
# border itself (a strip cropped at its cut): no row lies past it to step from,
# so a step found near it is noise or a band's flank, and the click is kept.
SNAP_STRIP_BORDER: Final = 1.0
# A robust sigma from the median absolute deviation of normal noise.
_MAD_SIGMA: Final = 1.4826

# Why a side of a register group is not used.
IgnoredReason = Literal["one_point", "few_shared_mws"]


def _lerp(a: float, b: float, f: float) -> float:
    # Exact at both ends: f == 0 gives a and f == 1 gives b, so a curve passes
    # through its own breakpoints to the bit.
    return (1.0 - f) * a + f * b


def _relative_error(decades: float) -> float:
    # |10**decades - 1|: the relative MW error of an error of ``decades`` in
    # log10(MW). inf where 10**decades is past the largest float, which only
    # degenerate ladders reach (points a pixel apart across a decade or more).
    try:
        return abs(10.0**decades - 1.0)
    except OverflowError:
        return math.inf


@dataclass(frozen=True)
class LadderCurve:
    """log10(MW) against y, linear between breakpoints, over the range
    ``[z_lo, z_hi]``; the end segments extend outward up to the range.

    Two breakpoints at least; ``ys`` strictly increase and ``zs`` strictly
    decrease. Outside the range, :meth:`mw_at` and :meth:`y_at` give None."""

    ys: tuple[float, ...]
    zs: tuple[float, ...]  # log10 MW
    z_lo: float
    z_hi: float

    def _segment_at_y(self, y: float) -> int:
        # The segment i with ys[i] <= y < ys[i+1]: a breakpoint takes the segment
        # below it; above the first point the first segment, at and below the
        # last point the last one.
        i = bisect.bisect_right(self.ys, y) - 1
        return min(max(i, 0), len(self.ys) - 2)

    def _segment_at_z(self, z: float) -> int:
        # The segment i with zs[i] >= z > zs[i+1], clamped to the end segments.
        i = bisect.bisect_right(self.zs, -z, key=lambda v: -v) - 1
        return min(max(i, 0), len(self.zs) - 2)

    def _z(self, y: float) -> float:
        i = self._segment_at_y(y)
        f = (y - self.ys[i]) / (self.ys[i + 1] - self.ys[i])
        return _lerp(self.zs[i], self.zs[i + 1], f)

    def _y(self, z: float) -> float:
        i = self._segment_at_z(z)
        f = (z - self.zs[i]) / (self.zs[i + 1] - self.zs[i])
        return _lerp(self.ys[i], self.ys[i + 1], f)

    def mw_at(self, y: float) -> float | None:
        """The MW at ``y``; None outside the range."""
        z = self._z(y)
        return 10.0**z if self.z_lo <= z <= self.z_hi else None

    def y_at(self, mw: float) -> float | None:
        """The y of ``mw``; None outside the range."""
        z = math.log10(mw)
        return self._y(z) if self.z_lo <= z <= self.z_hi else None

    def px_per_decade(self, y: float) -> float:
        """How many pixels one decade of MW spans at ``y``: the slope of the
        segment there (a breakpoint takes the segment below it)."""
        i = self._segment_at_y(y)
        return abs((self.ys[i + 1] - self.ys[i]) / (self.zs[i + 1] - self.zs[i]))

    @property
    def lo_mw(self) -> float:
        """The lowest MW in range, for display only (compare log10 values)."""
        return 10.0**self.z_lo

    @property
    def hi_mw(self) -> float:
        """The highest MW in range, for display only."""
        return 10.0**self.z_hi


@dataclass(frozen=True)
class Quality:
    """A ladder's quality (D2): its worst interior point left out."""

    value: float  # the largest relative MW disagreement
    mw: float  # the point it names, as labelled
    predicted_mw: float  # what its neighbours put there


@dataclass(frozen=True)
class Disagreement:
    """How far two ladders disagree once their offset is taken out: the largest
    relative MW error at a MW both hold, and that MW."""

    value: float  # inf past the largest float (degenerate ladders only)
    mw: float


@dataclass(frozen=True)
class Ladder:
    """One side of one register group, as marked."""

    side: LadderSide
    x: float | None  # the median of its points' x; None when none has one
    ys: tuple[float, ...]  # top to bottom
    mws: tuple[float, ...]
    sources: tuple[CalibrationPointSource, ...]
    curve: LadderCurve
    quality: Quality | None  # None with two points


@dataclass(frozen=True)
class Calibration:
    """A register group's calibration: one ladder, or two with the protein line
    between them. x is the lane's position across the image; with one ladder
    it is not needed (None is accepted)."""

    group: frozenset[str]
    method: FitMethod  # LOG_LINEAR_PIECEWISE
    ladders: tuple[Ladder, ...]  # 1 or 2, left first
    ignored: tuple[tuple[LadderSide, IgnoredReason], ...]  # sides not used, and why

    def curve_at(self, x: float | None) -> LadderCurve | None:
        """The curve of a lane at ``x``. One ladder: that ladder's, whatever x.
        Two: at each MW, y linear in x between the ladders; its breakpoints are
        both ladders' MWs within the range, and the range ends. None where the
        line folds over (only outside the ladders, when they disagree strongly).
        ValueError for ``x=None`` with two ladders."""
        if len(self.ladders) == 1:
            return self.ladders[0].curve
        if x is None:
            raise ValueError("a calibration from two ladders needs the lane's x")
        left, right = self.ladders
        x_left, x_right = self._ladder_xs()
        t = (x - x_left) / (x_right - x_left)
        z_lo, z_hi = self.z_lo, self.z_hi
        inner = {z for z in (*left.curve.zs, *right.curve.zs) if z_lo < z < z_hi}
        zs = (z_hi, *sorted(inner, reverse=True), z_lo)
        ys = tuple(_lerp(left.curve._y(z), right.curve._y(z), t) for z in zs)
        if any(not a < b for a, b in itertools.pairwise(ys)):
            return None
        return LadderCurve(ys, zs, z_lo, z_hi)

    def _ladder_xs(self) -> tuple[float, float]:
        # The two ladders' x. The model gives every point of a group with a right
        # ladder its x, so only a hand-built Calibration can lack one.
        left, right = self.ladders
        if left.x is None or right.x is None:
            raise ValueError("two ladders need their x")
        return left.x, right.x

    def mw_at(self, x: float | None, y: float) -> float | None:
        """The MW at (x, y); None outside the range."""
        curve = self.curve_at(x)
        return None if curve is None else curve.mw_at(y)

    def y_at(self, mw: float, x: float | None) -> float | None:
        """The y of ``mw`` in the lane at ``x``; None outside the range."""
        curve = self.curve_at(x)
        return None if curve is None else curve.y_at(mw)

    def px_per_decade(self, x: float | None, y: float) -> float | None:
        """Pixels per decade of MW at (x, y); None where the line folds over."""
        curve = self.curve_at(x)
        return None if curve is None else curve.px_per_decade(y)

    @property
    def z_lo(self) -> float:
        """The range's low end, log10 MW: the higher of the ladders' low ends."""
        return max(ladder.curve.z_lo for ladder in self.ladders)

    @property
    def z_hi(self) -> float:
        """The range's high end, log10 MW: the lower of the ladders' high ends."""
        return min(ladder.curve.z_hi for ladder in self.ladders)

    @property
    def lo_mw(self) -> float:
        """For display only (compare log10 values)."""
        return 10.0**self.z_lo

    @property
    def hi_mw(self) -> float:
        """For display only."""
        return 10.0**self.z_hi

    @property
    def two_ladders(self) -> bool:
        return len(self.ladders) == 2

    @property
    def two_point(self) -> bool:
        """Whether a ladder in use has only two points (no quality measurable)."""
        return any(len(ladder.ys) == 2 for ladder in self.ladders)

    @property
    def strip_edges_only(self) -> bool:
        """Whether every point in use is a strip edge (D7)."""
        return all(
            source is CalibrationPointSource.STRIP_EDGE
            for ladder in self.ladders
            for source in ladder.sources
        )

    @property
    def shared_mws(self) -> tuple[float, ...]:
        """The MWs both ladders hold, top to bottom; empty with one ladder."""
        if len(self.ladders) != 2:
            return ()
        left, right = self.ladders
        held = set(right.mws)
        return tuple(mw for mw in left.mws if mw in held)

    def _offsets(self) -> list[tuple[float, float]]:
        # (MW, the right ladder's y less the left one's) at each shared MW.
        left, right = self.ladders
        left_y = dict(zip(left.mws, left.ys, strict=True))
        right_y = dict(zip(right.mws, right.ys, strict=True))
        return [(mw, right_y[mw] - left_y[mw]) for mw in self.shared_mws]

    @property
    def offset_px(self) -> float | None:
        """How much lower the right ladder runs than the left one: the median
        offset at the shared MWs. None with one ladder."""
        if not self.two_ladders:
            return None
        return statistics.median(d for _, d in self._offsets())

    @property
    def tilt_deg(self) -> float | None:
        """The line's slope across the blot, in degrees; positive when the right
        side runs lower. None with one ladder."""
        offset = self.offset_px
        if offset is None:
            return None
        x_left, x_right = self._ladder_xs()
        return math.degrees(math.atan2(offset, x_right - x_left))

    @property
    def disagreement(self) -> Disagreement | None:
        """The largest relative MW error, at a shared MW, of the offset between
        the ladders there against their median offset, scaled by the pixels per
        decade midway between them. Tilt alone gives about 0; labels one band
        off do not; an error past the largest float is inf. None with one
        ladder."""
        offset = self.offset_px
        if offset is None:
            return None
        x_left, x_right = self._ladder_xs()
        middle = self.curve_at(0.5 * (x_left + x_right))
        if middle is None:  # never: between the ladders the line cannot fold
            return None
        worst: Disagreement | None = None
        for mw, d in self._offsets():
            ppd = middle.px_per_decade(middle._y(math.log10(mw)))
            value = _relative_error((d - offset) / ppd)
            if worst is None or value > worst.value:
                worst = Disagreement(value, mw)
        return worst

    @property
    def quality(self) -> Quality | None:
        """The worse of its ladders' qualities; None when neither has three points."""
        measured = [ladder.quality for ladder in self.ladders if ladder.quality is not None]
        return max(measured, key=lambda quality: quality.value, default=None)


@dataclass(frozen=True)
class NoCalibration:
    """A register group without a curve, and why."""

    group: frozenset[str]
    reason: Literal["no_points", "one_point"]


def _quality(
    ys: tuple[float, ...], zs: tuple[float, ...], mws: tuple[float, ...]
) -> Quality | None:
    if len(ys) < 3:
        return None
    worst: Quality | None = None
    for i in range(1, len(ys) - 1):
        f = (ys[i] - ys[i - 1]) / (ys[i + 1] - ys[i - 1])
        predicted = _lerp(zs[i - 1], zs[i + 1], f)
        value = _relative_error(zs[i] - predicted)
        if worst is None or value > worst.value:
            worst = Quality(value, mws[i], 10.0**predicted)
    return worst


def _ladder(side: LadderSide, points: list[CalibrationPoint]) -> Ladder:
    # The model keeps a side's points one per position and log10(MW), MWs
    # decreasing down, so the ys strictly increase and the zs strictly decrease.
    points = sorted(points, key=lambda p: p.y)
    ys = tuple(p.y for p in points)
    mws = tuple(p.mw for p in points)
    zs = tuple(math.log10(mw) for mw in mws)
    strip = CalibrationPointSource.STRIP_EDGE
    above = 0.0 if points[0].source is strip else EXTRAPOLATE_DECADES
    below = 0.0 if points[-1].source is strip else EXTRAPOLATE_DECADES
    xs = [p.x for p in points if p.x is not None]
    return Ladder(
        side=side,
        x=statistics.median(xs) if xs else None,
        ys=ys,
        mws=mws,
        sources=tuple(p.source for p in points),
        curve=LadderCurve(ys, zs, zs[-1] - below, zs[0] + above),
        quality=_quality(ys, zs, mws),
    )


def _fit(membrane: Membrane, group: frozenset[str]) -> Calibration | NoCalibration:
    points = [p for p in membrane.calibration.points if p.image_id in group]
    sides = {side: [p for p in points if p.side == side] for side in LadderSide}
    used = {side: held for side, held in sides.items() if len(held) >= MIN_LADDER_POINTS}
    ignored: list[tuple[LadderSide, IgnoredReason]] = [
        (side, "one_point") for side, held in sides.items() if 0 < len(held) < MIN_LADDER_POINTS
    ]
    if not used:
        return NoCalibration(group, "one_point" if points else "no_points")
    if len(used) == 2:
        left = {p.mw for p in used[LadderSide.LEFT]}
        if len(left & {p.mw for p in used[LadderSide.RIGHT]}) < MIN_SHARED_MWS:
            del used[LadderSide.RIGHT]
            ignored.append((LadderSide.RIGHT, "few_shared_mws"))
    return Calibration(
        group=group,
        method=DEFAULT_FIT_METHOD,
        ladders=tuple(_ladder(side, held) for side, held in used.items()),
        ignored=tuple(ignored),
    )


def calibration_for(membrane: Membrane, image_id: str) -> Calibration | NoCalibration:
    """The image's register group, fitted piecewise. A side with at least
    MIN_LADDER_POINTS points is a ladder. A side with fewer, or a right side that
    shares fewer than MIN_SHARED_MWS MWs with the left, is not used
    (``Calibration.ignored``). No ladder at all gives ``NoCalibration`` with
    ``"no_points"`` or ``"one_point"``. UnknownIdError if the image is not on the
    membrane."""
    return _fit(membrane, membrane.group_of(image_id))


def calibrations(membrane: Membrane) -> dict[frozenset[str], Calibration | NoCalibration]:
    """Every register group of the membrane, fitted, in the order of their first image."""
    return {group: _fit(membrane, group) for group in membrane.register_groups()}


def fit_quality(membrane: Membrane) -> float | None:
    """The worst ladder's quality over every register group: what
    ``MwCalibration.fit_quality`` stores. None when no ladder has three points."""
    values = [
        fitted.quality.value
        for fitted in calibrations(membrane).values()
        if isinstance(fitted, Calibration) and fitted.quality is not None
    ]
    return max(values, default=None)


def fit_method(membrane: Membrane) -> FitMethod:
    """``DEFAULT_FIT_METHOD`` when a register group has a curve, else ``LOG_LINEAR``
    (the model's default: nothing fitted)."""
    if any(isinstance(fitted, Calibration) for fitted in calibrations(membrane).values()):
        return DEFAULT_FIT_METHOD
    return FitMethod.LOG_LINEAR


def _peak_offset(below: float, at: float, above: float) -> float:
    # The vertex of the parabola through three neighbouring samples, from the
    # middle one, clipped to half a sample either way; 0 where they are flat.
    curvature = below - 2.0 * at + above
    if curvature >= 0.0:
        return 0.0
    return min(0.5, max(-0.5, 0.5 * (below - above) / curvature))


def refine_point(
    array: np.ndarray,
    x: float,
    y: float,
    *,
    source: CalibrationPointSource,
    polarity: Polarity,
    marked_ys: Sequence[float] = (),
) -> float | None:
    """Where a click at ``(x, y)`` on a ladder band or a strip edge meant, or
    None when nothing near it stands out of the noise (the click is then kept).

    The window reaches :data:`SNAP_HALF_X` px either side of the click and
    :data:`SNAP_HALF_Y` px above and below it, clipped to the image, and only
    :data:`SNAP_MARKED_GAP` of the way to the nearest point in ``marked_ys``
    (those already marked on the same ladder) on each side, so a click never
    snaps onto a marked band. Its signal is each pixel's deviation from the
    window's median: turned by the image's ``polarity`` for a band on a marker
    image (``visible_marker``), and its absolute value for a faint marker on a
    chemiluminescence image (``chemiluminescence_marker``), which may show
    either way. The profile is the signal's mean along each row, less its
    median over the window (an absolute value's mean is not 0); the robust
    sigma of its row-to-row steps is the steps' noise, and over the square
    root of two the profile's.

    A ladder band is a local maximum of the profile reaching :data:`SNAP_K`
    noise sigmas, the one nearest the click (of two as near, the higher),
    refined by a parabola through it and its neighbours (at most half a row
    either way): row r of the image lies at y = r + 0.5. A strip edge
    (``strip_edge``) is the same on the size of the profile's steps, against
    the steps' noise, the edge between rows r and r + 1 lying at y = r + 1;
    one clicked within :data:`SNAP_STRIP_BORDER` px of the image's top or
    bottom is that border (None): an edge a few rows inside it, on an image
    not cropped at the cut, snaps from a click further in."""
    height, width = array.shape[:2]
    if source is CalibrationPointSource.STRIP_EDGE and not (
        SNAP_STRIP_BORDER < y < height - SNAP_STRIP_BORDER
    ):
        return None
    top, bottom = y - SNAP_HALF_Y, y + SNAP_HALF_Y
    for marked in marked_ys:
        if marked <= y:
            top = max(top, y - SNAP_MARKED_GAP * (y - marked))
        if marked >= y:
            bottom = min(bottom, y + SNAP_MARKED_GAP * (marked - y))
    # The rows and columns whose centres lie in the window.
    r0, r1 = max(0, math.ceil(top - 0.5)), min(height - 1, math.floor(bottom - 0.5))
    c0 = max(0, math.ceil(x - SNAP_HALF_X - 0.5))
    c1 = min(width - 1, math.floor(x + SNAP_HALF_X - 0.5))
    if r1 - r0 < 2 or c1 < c0:  # a peak needs a row on each side
        return None
    window = np.asarray(array[r0 : r1 + 1, c0 : c1 + 1], dtype=float)
    deviation = window - float(np.median(window))
    if source is CalibrationPointSource.CHEMILUMINESCENCE_MARKER:
        deviation = np.abs(deviation)
    elif polarity is Polarity.DARK_ON_LIGHT:
        deviation = -deviation
    # Above the profile's own level: an absolute value's is its noise's mean.
    profile = deviation.mean(axis=1)
    profile = profile - float(np.median(profile))
    steps = np.diff(profile)
    step_noise = _MAD_SIGMA * float(np.median(np.abs(steps - np.median(steps))))
    if source is CalibrationPointSource.STRIP_EDGE:
        # Step k lies between rows k and k + 1.
        series, first, noise = np.abs(steps), r0 + 1.0, step_noise
    else:
        series, first, noise = profile, r0 + 0.5, step_noise / math.sqrt(2.0)
    found: list[float] = []
    for k in range(1, len(series) - 1):
        below, at, above = float(series[k - 1]), float(series[k]), float(series[k + 1])
        if at > below and at >= above and at > 0.0 and at >= SNAP_K * noise:
            found.append(first + k + _peak_offset(below, at, above))
    if not found:
        return None
    return min(found, key=lambda peak: (abs(peak - y), peak))


def settings() -> dict[str, JsonValue]:
    """How molecular weights are calibrated, JSON-plain: what an export record
    reports (with the ladder presets' version, which the record adds)."""
    return {
        "method": DEFAULT_FIT_METHOD.value,
        "per_ladder": "log10(MW) linear in y between neighbouring points; the end segments extend",
        "two_ladders": (
            "at each MW, y linear in x between the two ladders, each at the median x of its points"
        ),
        "extrapolate_decades": EXTRAPOLATE_DECADES,
        "past_strip_edge": 0,
        "two_ladder_range": "the intersection of the two ladders' ranges",
        "quality": (
            "per ladder, the largest relative MW disagreement of an interior point with the"
            " line between its neighbours (leave-one-out); none with two points"
        ),
        "poor_fit_warn": FIT_WARN,
        "ladders_disagree_warn": LADDERS_WARN,
        "min_ladder_points": MIN_LADDER_POINTS,
        "min_shared_mws": MIN_SHARED_MWS,
        "ladder_offsets": "at the MWs both ladders hold",
        "snap": {
            "half_x": SNAP_HALF_X,
            "half_y": SNAP_HALF_Y,
            "marked_gap": SNAP_MARKED_GAP,
            "k": SNAP_K,
            "signal": (
                "deviation from the window's median, by the image's polarity on a marker"
                " image, its absolute value for a faint marker on a chemiluminescence image;"
                " the mean along each row, less its median; noise the robust sigma of the"
                " row-to-row steps over the square root of two"
            ),
            "strip_edge_noise": "the robust sigma of the row-to-row steps",
            "strip_edge_border": SNAP_STRIP_BORDER,
            "ladder_band": (
                "the local maximum nearest the click, refined by a parabola (at most half a"
                " row), at row + 0.5"
            ),
            "strip_edge": "the same on the size of the row-to-row steps, at the edge between rows",
            "no_band": "the clicked y is kept",
        },
    }
