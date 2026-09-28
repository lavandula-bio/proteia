# SPDX-License-Identifier: Apache-2.0
"""Molecular-weight calibration (#58): each ladder's curve, and the protein line
between two ladders.

Pure: it reads :mod:`proteia.core.model` objects and arrays it is given and
never writes them, and opens no file; only :func:`refine_point`,
:func:`is_snap_position` and :func:`find_ladder` read pixels (with numpy).
Positions are continuous coordinates of an image's analysis array
(:class:`~proteia.core.model.CalibrationPoint`): pixel row r covers [r, r+1),
so a band peaking on row r lies at y = r + 0.5, as a box centre does.

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
that stands out of the noise, the click is kept as it is. The window decides
only which band a click takes: where that band lies comes from its own rows'
means across the lane, so a band is snapped to the same y wherever it was
clicked from, and a band saturated or clipped flat lies at the middle of its
flat top. So a y a snap gave can be told afterwards from one placed by hand
(:func:`is_snap_position`, at the x it was snapped at): it is where the peak
its own row climbs to lies, to the bit.

Finding a ladder (:func:`find_ladder`, D8). A click on the ladder lane finds
the bands down it and proposes which ladder MW each one is: every in-order
labelling of the peaks is scored by how far log10(MW) leaves a quadratic in y
fitted to it (real ladders are not log-linear end to end, and a curve through
every point would fit any labelling), by the ladder MWs and peaks it leaves
unmatched, by whether the strongest peaks carry the ladder's reference bands,
and, for a second ladder, by how far its height difference from the first
ladder varies. The best labelling is proposed, with the ladder MWs it left
unmatched predicted from its quadratic, and how much worse the next-best
labelling scores: a small gap means the labels may be one band off. Nothing is
stored: the user adjusts the proposal and applies it.
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
# A y a snap gave lies where the peak its own row climbs to lies, to the bit:
# the same rows' means give the same y, whatever window found the band. This
# much (px) is hundreds of times a float's rounding at 10^4 px, and far below
# how near a snap comes to a whole number: a strip edge thousands of counts
# high, which a parabola barely moves, may lie 1e-7 px from one, where a y is
# as likely typed by hand.
SNAP_POSITION_TOLERANCE: Final = 1e-9
# A robust sigma from the median absolute deviation of normal noise.
_MAD_SIGMA: Final = 1.4826

# Finding a ladder (find_ladder). The ladder lane's columns, px either side of
# the clicked x. How far, in decades of MW, a real ladder's log10(MW) leaves a
# quadratic in y (vendor ladders: 0.036-0.049), the unit of a labelling's
# residuals; a second ladder's height differences from the first are measured
# in the same unit. What each ladder MW left without a peak, each peak left
# without a MW, and each reference MW not on one of the strongest labelled
# peaks costs, in squared units. How much worse the next-best labelling must
# score for a proposal not to be doubtful. A peak must stand SNAP_K noise
# sigmas above the membrane around it: the profile's median over this share of
# the image's rows, centred on it (a band is a small part of that).
FIND_HALF_X: Final = 20
FIND_BACKGROUND: Final = 0.1
PRIOR_DECADES: Final = 0.05
MISS_COST: Final = 9.0
EXTRA_COST: Final = 9.0
REF_COST: Final = 9.0
GAP_WARN: Final = 5.0
# How many labellings the search may try: past this it stops, keeps the best
# it found and calls the proposal doubtful. A clean ladder's lane takes a few
# hundred; a lane with many noise peaks may reach it (about a second).
FIND_BUDGET: Final = 200_000

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

    def y_at_log(self, z: float) -> float | None:
        """The y of the MW whose log10 is ``z``; None outside the range. A range
        end's own y, which ``y_at(10 ** z)`` may miss by the last bit."""
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


def _signal(window: np.ndarray, source: CalibrationPointSource, polarity: Polarity) -> np.ndarray:
    """Each pixel's deviation from the window's median, turned by the image's
    polarity for a band on a marker image, and taken as its absolute value for
    a faint marker on a chemiluminescence image, which may show either way."""
    deviation = window - float(np.median(window))
    if source is CalibrationPointSource.CHEMILUMINESCENCE_MARKER:
        return np.abs(deviation)
    if polarity is Polarity.DARK_ON_LIGHT:
        return -deviation
    return deviation


def _robust_sigma(values: np.ndarray) -> float:
    # 0 for no values.
    if values.size == 0:
        return 0.0
    return _MAD_SIGMA * float(np.median(np.abs(values - np.median(values))))


def _profile(
    window: np.ndarray, source: CalibrationPointSource, polarity: Polarity
) -> tuple[np.ndarray, float]:
    """The signal down a window (:func:`_signal`), one value per row, and the
    robust sigma of its row-to-row steps: the signal's mean along each row,
    less the median over the window's rows, is the profile (an absolute
    value's mean is its noise's, not 0)."""
    profile = _signal(window, source, polarity).mean(axis=1)
    profile = profile - float(np.median(profile))
    return profile, _robust_sigma(np.diff(profile))


def _lane_noise(window: np.ndarray, signal: np.ndarray, step_noise: float) -> float:
    """The noise sigma of a ladder lane's profile: its steps' robust sigma
    ``step_noise`` over the square root of two, but at least what the pixels
    give a row that none of them is clipped in. A membrane clipped white (or a
    background clipped black), or flat within a grey level, has steps of 0 or
    nearly 0 down most of the lane however noisy its bands are, and its steps
    alone would take every ripple on a band for one. Elsewhere the steps'
    noise is the larger as a rule (it also holds what neighbouring pixels
    share), and stands.

    - The pixels' noise: the robust sigma of the differences between
      neighbouring pixels along each row of the ``signal``, each less its
      column's median (a column's own level is no noise), taken only between
      two pixels off the window's lowest and highest values (a clipped pixel
      has no noise); over the square root of two, and of the window's columns
      for a row's mean.
    - Rounding to the image's grey levels: their spacing (the smallest
      difference between two values in the window) over the square root of
      twelve times the window's columns."""
    columns = window.shape[1]
    free = (window != window.max()) & (window != window.min())
    level = signal - np.median(signal, axis=0)
    along = np.diff(level, axis=1)[free[:, 1:] & free[:, :-1]]
    pixels = _robust_sigma(along) / math.sqrt(2.0 * columns)
    values = np.unique(window)
    spacing = float(np.min(np.diff(values))) if values.size > 1 else 0.0
    return max(step_noise / math.sqrt(2.0), pixels, spacing / math.sqrt(12.0 * columns))


def _peak_offset(below: float, at: float, above: float) -> float:
    # The vertex of the parabola through three neighbouring samples, from the
    # middle one, clipped to half a sample either way; 0 where they are flat.
    curvature = below - 2.0 * at + above
    if curvature >= 0.0:
        return 0.0
    return min(0.5, max(-0.5, 0.5 * (below - above) / curvature))


def _maxima(series: Sequence[float]) -> list[tuple[int, int]]:
    """Each local maximum of ``series`` as the run of equal samples it is,
    ``(first, last)``: one sample, or a flat top of several (a band saturated
    or clipped flat), with a lower sample on each side, so none touches an
    end."""
    maxima = []
    first, count = 0, len(series)
    while first < count:
        last = first
        while last + 1 < count and series[last + 1] == series[first]:
            last += 1
        if 0 < first and last < count - 1 and series[first - 1] < series[first] > series[last + 1]:
            maxima.append((first, last))
        first = last + 1
    return maxima


def _top(series: Sequence[float], first: int, last: int, origin: float) -> float:
    """Where a local maximum ``first..last`` of ``series`` (:func:`_maxima`)
    peaks, ``origin`` being where sample 0 lies: a lone sample refined by the
    parabola through it and its neighbours (strictly the highest, so within
    half a sample of it), a flat top at its middle."""
    if first == last:
        return origin + first + _peak_offset(series[first - 1], series[first], series[first + 1])
    return origin + 0.5 * (first + last)


def _climb(series: Sequence[float], start: int) -> tuple[int, int] | None:
    """The local maximum of ``series`` (:func:`_maxima`) reached from sample
    ``start`` by stepping to a higher neighbour while there is one (the
    higher of two, the earlier of two as high); None where that ends on a
    run touching an end of the series."""
    count = len(series)
    first = last = start
    while True:
        value = series[first]
        while first > 0 and series[first - 1] == value:
            first -= 1
        while last < count - 1 and series[last + 1] == value:
            last += 1
        higher = [i for i in (first - 1, last + 1) if 0 <= i < count and series[i] > value]
        if not higher:
            return (first, last) if 0 < first and last < count - 1 else None
        first = last = max(higher, key=lambda i: (series[i], -i))


def _row_means(block: np.ndarray) -> list[float]:
    """The mean of each row of ``block`` (a lane's columns), its sum correctly
    rounded (:func:`math.fsum`): a row's mean does not depend on which other
    rows were read with it, or on the order a sum took."""
    columns = block.shape[1]
    return [math.fsum(row) / columns for row in block.tolist()]


def _shows(
    means: Sequence[float], source: CalibrationPointSource, polarity: Polarity
) -> list[list[float]]:
    """The series on which a snap finds where a band lies, from the mean of
    each row across the lane (``means``, :func:`_row_means`), one for each way
    a band may show: the means turned by the image's polarity for a band on a
    marker image, the means either way for a faint marker on a
    chemiluminescence image, the size of their row-to-row steps for a strip
    edge (step k between rows k and k + 1). None has a window's level taken
    off, so where a band peaks on it does not depend on the window a click
    gave."""
    if source is CalibrationPointSource.STRIP_EDGE:
        return [[abs(b - a) for a, b in itertools.pairwise(means)]]
    bright = list(means)
    dark = [-v for v in bright]
    if source is CalibrationPointSource.CHEMILUMINESCENCE_MARKER:
        return [bright, dark]
    return [bright] if polarity is Polarity.LIGHT_ON_DARK else [dark]


def _lane_columns(x: float, width: int) -> tuple[int, int]:
    """The first and last columns a snap at ``x`` reads: those whose centres
    lie within :data:`SNAP_HALF_X` px of it, on an image ``width`` px wide
    (the first past the last where none does)."""
    c0 = max(0, math.ceil(x - SNAP_HALF_X - 0.5))
    return c0, min(width - 1, math.floor(x + SNAP_HALF_X - 0.5))


def _origin(source: CalibrationPointSource) -> float:
    # Where sample 0 of a snap's series lies: row 0's centre, or for a strip
    # edge the step between rows 0 and 1.
    return 1.0 if source is CalibrationPointSource.STRIP_EDGE else 0.5


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
    noise sigmas (a flat top of equal rows, where a band saturates, is one),
    the one nearest the click (of two as near, the higher). Where it lies is
    found on the rows' own means across the lane (:func:`_shows`: turned by
    the polarity; for a faint marker, the way the band shows against the
    window's median), not on the profile, whose level (and, for a faint
    marker, whose absolute value) follows the window: from the band's row,
    up to the local maximum of the means there (none where that leaves the
    window; its neighbours may lie a row past it), refined by a parabola
    through it and its neighbours, or at the middle of a flat top (row r of
    the image lies at y = r + 0.5). So a band snaps to one y wherever it was
    clicked from, and :func:`is_snap_position` tells that y from one placed
    by hand. A strip edge (``strip_edge``) is the same on the size of the
    profile's steps, against the steps' noise, the edge between rows r and
    r + 1 lying at y = r + 1; one clicked within :data:`SNAP_STRIP_BORDER` px
    of the image's top or bottom is that border (None): an edge a few rows
    inside it, on an image not cropped at the cut, snaps from a click
    further in."""
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
    c0, c1 = _lane_columns(x, width)
    if r1 - r0 < 2 or c1 < c0:  # a peak needs a row on each side
        return None
    window = np.asarray(array[r0 : r1 + 1, c0 : c1 + 1], dtype=float)
    profile, step_noise = _profile(window, source, polarity)
    if source is CalibrationPointSource.STRIP_EDGE:
        # Step k lies between rows k and k + 1.
        series, noise = np.abs(np.diff(profile)), step_noise
    else:
        series, noise = profile, step_noise / math.sqrt(2.0)
    values = [float(v) for v in series]
    # The means reach a row past the window each way, where the image has
    # one: a band peaking on the window's first or last row has a neighbour.
    # Sample k of the window's series is sample k + skip of theirs.
    e0, e1 = max(0, r0 - 1), min(height - 1, r1 + 1)
    lane = np.asarray(array[e0 : e1 + 1, c0 : c1 + 1], dtype=float)
    shows = _shows(_row_means(lane), source, polarity)
    skip = r0 - e0
    found: list[float] = []
    for first, last in _maxima(values):
        if not (values[first] > 0.0 and values[first] >= SNAP_K * noise):
            continue
        means = shows[0]
        if len(shows) > 1 and np.mean(window[first : last + 1]) < np.median(window):
            means = shows[1]  # a faint marker darker than the window around it
        peak = _climb(means, skip + (first + last) // 2)
        if peak is not None and skip <= peak[0] and peak[1] < skip + len(values):
            found.append(_top(means, *peak, e0 + _origin(source)))
    if not found:
        return None
    return min(found, key=lambda peak: (abs(peak - y), peak))


def is_snap_position(
    array: np.ndarray,
    x: float,
    y: float,
    *,
    source: CalibrationPointSource,
    polarity: Polarity,
) -> bool:
    """Whether ``y`` is where :func:`refine_point` at ``x`` puts the band (or
    strip edge) it lies on: climbing from the sample ``y`` lies on (either,
    on the boundary between two) on the rows' means across the lane
    (:func:`_shows`; for a faint marker either way, for a strip edge the steps
    between rows) ends on a peak that lies at ``y``, within
    :data:`SNAP_POSITION_TOLERANCE` px. A y a snap at ``x`` gave lies there to
    the bit, wherever the snap was clicked from down the lane and whatever the
    marked points cut from its window; a y placed by hand, only where it was
    put on that peak exactly. A snap at another x reads other columns, whose
    means put the band a hair elsewhere: a y is judged at the x it was snapped
    at. The noise is not asked: whether a band stood out of it depends on the
    window a click gave, which a y alone does not tell.

    Only the rows about ``y`` are read: a peak a snap gives is at most a
    window high, and a climb that leaves them does not end at ``y``."""
    height, width = array.shape[:2]
    c0, c1 = _lane_columns(x, width)
    if c1 < c0 or not math.isfinite(y):
        return False
    origin = _origin(source)
    at = y - origin + 0.5  # sample k covers [k, k + 1) here
    r0 = max(0, math.floor(at) - 2 * SNAP_HALF_Y)
    r1 = min(height - 1, math.ceil(at) + 2 * SNAP_HALF_Y)
    if r1 - r0 < 2:
        return False
    means = _row_means(np.asarray(array[r0 : r1 + 1, c0 : c1 + 1], dtype=float))
    starts = {math.floor(at) - r0, math.ceil(at) - 1 - r0}
    for series in _shows(means, source, polarity):
        for start in starts:
            if 0 <= start < len(series):
                peak = _climb(series, start)
                if peak is not None:
                    lies = _top(series, *peak, r0 + origin)
                    if abs(lies - y) <= SNAP_POSITION_TOLERANCE:
                        return True
    return False


@dataclass(frozen=True)
class Tick:
    """One ladder MW on a proposed ruler (:class:`LadderProposal`), at a
    continuous y as a calibration point's."""

    mw: float
    y: float
    found: bool  # on a peak found down the lane; else predicted from the fitted quadratic
    # The peak's height above the membrane around it in noise sigmas; None when predicted.
    strength: float | None


@dataclass(frozen=True)
class LadderProposal:
    """What :func:`find_ladder` proposes for a ladder lane."""

    x: float  # the lane clicked
    ticks: tuple[Tick, ...]  # every ladder MW placed on the image, top to bottom
    extra: tuple[float, ...]  # the ys of the peaks no label took, top to bottom
    score: float  # the winning labelling's cost (squared prior units)
    # How much worse the next-best labelling scores: inf when there is none;
    # 0 when the search stopped at its budget (FIND_BUDGET).
    gap: float
    doubtful: bool  # gap < GAP_WARN: the labels may be one band off


def _prominences(profile: Sequence[float]) -> list[float]:
    """How far each sample stands above the profile around it: its height less
    the higher of the lowest points between it and the nearest higher sample on
    each side (or the profile's end). One pass each way, with a stack."""
    lowest: list[list[float]] = []
    for order in (range(len(profile)), range(len(profile) - 1, -1, -1)):
        side = [0.0] * len(profile)
        stack: list[tuple[float, float]] = []  # (height, the lowest point since the one before)
        for i in order:
            low = height = profile[i]
            while stack and stack[-1][0] <= height:
                low = min(low, stack.pop()[1])
            side[i] = low
            stack.append((height, low))
        lowest.append(side)
    left, right = lowest
    return [height - max(a, b) for height, a, b in zip(profile, left, right, strict=True)]


def _background(profile: np.ndarray, radius: int) -> np.ndarray:
    """The level of the membrane around each row of a profile: the median of
    the profile over the rows within ``radius`` of it (fewer at the ends),
    taken every quarter radius and interpolated between (it varies slowly,
    and a tall image's lane would otherwise take one median per row)."""
    values = np.asarray(profile, dtype=float)
    padded = np.pad(values, radius, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    rows = np.arange(0, len(values), max(1, radius // 4))
    if rows[-1] != len(values) - 1:
        rows = np.append(rows, len(values) - 1)
    return np.interp(np.arange(len(values)), rows, np.nanmedian(windows[rows], axis=1))


def _ladder_peaks(profile: np.ndarray, noise: float) -> list[tuple[float, float]]:
    """The peaks down a ladder lane's profile, top to bottom, as ``(y, strength)``:
    each local maximum that stands :data:`SNAP_K` noise sigmas above the
    membrane around it (the profile's running median over a
    :data:`FIND_BACKGROUND` share of the rows, at least :data:`SNAP_HALF_Y` each
    way, so a slow shading of the membrane is no peak) and as far above the
    profile around it (its prominence, so a ripple on a band is none), refined
    by a parabola through it and its neighbours, at y = row + 0.5 + offset, or
    at the middle of a flat top of equal rows, where a band saturates
    (:func:`_top`). Its strength is its height above the membrane (at the
    middle of a flat top) in noise sigmas. The noise is the profile's
    (:func:`_lane_noise`), and at least a billionth of the profile's largest
    value (a profile without noise has none), so a strength is finite."""
    values = [float(v) for v in profile]
    scale = max((abs(v) for v in values), default=0.0)
    noise = max(noise, 1e-9 * scale)
    if noise <= 0.0:  # a flat profile
        return []
    radius = max(SNAP_HALF_Y, round(FIND_BACKGROUND * len(values) / 2))
    level = [float(v) for v in _background(profile, radius)]
    prominence = _prominences(values)
    peaks = []
    for first, last in _maxima(values):
        height = values[first] - level[(first + last) // 2]
        if min(height, prominence[first]) >= SNAP_K * noise:
            peaks.append((_top(values, first, last, 0.5), height / noise))
    return peaks


class _Exhausted(Exception):
    """The labelling search reached :data:`FIND_BUDGET`."""


# Sums over a labelling's (u, z): n, u, u², u³, u⁴, z, zu, zu², z².
_Moments = tuple[int, float, float, float, float, float, float, float, float]
_NO_MOMENTS: Final[_Moments] = (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# Sums over a labelling's height differences d from the other ladder, weighted
# by w: w, wd, wd².
_Offsets = tuple[float, float, float]


def _with(moments: _Moments, u: float, z: float) -> _Moments:
    n, s1, s2, s3, s4, t0, t1, t2, tt = moments
    u2 = u * u
    return (
        n + 1,
        s1 + u,
        s2 + u2,
        s3 + u2 * u,
        s4 + u2 * u2,
        t0 + z,
        t1 + z * u,
        t2 + z * u2,
        tt + z * z,
    )


def _det3(m: Sequence[Sequence[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _quadratic(moments: _Moments) -> tuple[float, float, float] | None:
    """The least-squares ``z = a u² + b u + c`` of at least three points (the
    normal equations, by Cramer's rule), or the line through two; None when
    the points do not fix it."""
    n, s1, s2, s3, s4, t0, t1, t2, _ = moments
    if n == 2:
        spread = n * s2 - s1 * s1
        if spread <= 0.0:
            return None
        b = (n * t1 - s1 * t0) / spread
        return 0.0, b, (t0 - b * s1) / n
    matrix = ((s4, s3, s2), (s3, s2, s1), (s2, s1, float(n)))
    det = _det3(matrix)
    if n < 2 or det == 0.0:
        return None
    rhs = (t2, t1, t0)
    solved = []
    for column in range(3):
        replaced = [[rhs[r] if c == column else matrix[r][c] for c in range(3)] for r in range(3)]
        solved.append(_det3(replaced) / det)
    return solved[0], solved[1], solved[2]


def _residual(moments: _Moments) -> float:
    """The sum of squared residuals of log10(MW) from the labelling's
    least-squares quadratic in y; 0 for three points or fewer."""
    if moments[0] <= 3:
        return 0.0
    fitted = _quadratic(moments)
    if fitted is None:
        return 0.0
    a, b, c = fitted
    return max(0.0, moments[8] - (a * moments[7] + b * moments[6] + c * moments[5]))


def _spread(offsets: _Offsets) -> float:
    # The weighted sum of squares of the height differences about their mean.
    w, wd, wdd = offsets
    return max(0.0, wdd - wd * wd / w) if w > 0.0 else 0.0


_Pairs = tuple[tuple[int, int], ...]  # (peak index, ladder MW index), both increasing


class _Labeller:
    """The search for the best in-order labelling of a lane's peaks with the
    ladder MWs, and for the best one that labels some peak otherwise (branch and
    bound, deterministic).

    A labelling's cost, in squared units of :data:`PRIOR_DECADES`: the squared
    residuals of log10(MW) from its least-squares quadratic in y; with another
    ladder, the weighted squared spread of its height differences from that
    ladder at the same MWs, each over that ladder's pixels per decade there;
    :data:`MISS_COST` per ladder MW and :data:`EXTRA_COST` per peak it leaves
    unmatched; and :data:`REF_COST` per reference MW not carried by one of the
    strongest labelled peaks, as many as there are reference MWs. Its quadratic
    must fall over the peaks it labels. Each part only grows as a labelling
    grows (a peak labelled later can only push a reference out of the
    strongest), so what a partial labelling has cost already, with the peaks
    and MWs that can no longer be matched, bounds every labelling it leads to."""

    def __init__(
        self,
        peaks: Sequence[tuple[float, float]],
        zs: Sequence[float],
        reference: Sequence[bool],
        other: LadderCurve | None,
    ) -> None:
        self.ys = [y for y, _ in peaks]
        # y scaled to about [-1, 1], so the normal equations stay well conditioned.
        self.mid = 0.5 * (self.ys[0] + self.ys[-1])
        self.scale = max(1.0, 0.5 * (self.ys[-1] - self.ys[0]))
        self.us = [(y - self.mid) / self.scale for y in self.ys]
        self.zs = list(zs)
        self.reference = list(reference)
        self.references = sum(self.reference)
        # Each peak's place by strength, strongest first (the higher of two as strong).
        order = sorted(range(len(peaks)), key=lambda i: (-peaks[i][1], i))
        self.rank = [0] * len(peaks)
        for place, i in enumerate(order):
            self.rank[i] = place
        # Per ladder MW, with another ladder: its y there (its end segments
        # extended) and the weight 1 / (its pixels per decade there)².
        self.other: list[tuple[float, float]] | None = None
        if other is not None:
            self.other = [(at, other.px_per_decade(at) ** -2.0) for at in map(other._y, self.zs)]
        # The next pair's steps from the last one, cheapest skip first.
        k, n = len(self.ys), len(self.zs)
        self.steps = sorted(
            ((di, dj) for di in range(k) for dj in range(n)),
            key=lambda step: (EXTRA_COST * step[0] + MISS_COST * step[1], step),
        )
        self.tried = 0
        self.best = math.inf
        self.best_pairs: _Pairs | None = None
        self.differs_from: frozenset[tuple[int, int]] | None = None

    def _carried(self, pairs: _Pairs) -> int:
        # The reference MWs labelled on one of the strongest labelled peaks.
        strongest = sorted(self.rank[i] for i, _ in pairs)[: self.references]
        return sum(1 for i, j in pairs if self.reference[j] and self.rank[i] in strongest)

    def _offset(self, offsets: _Offsets, i: int, j: int) -> _Offsets:
        if self.other is None:
            return offsets
        at, weight = self.other[j]
        d = self.ys[i] - at
        return offsets[0] + weight, offsets[1] + weight * d, offsets[2] + weight * d * d

    def _falls(self, pairs: _Pairs, moments: _Moments) -> bool:
        # Whether the quadratic falls over the labelled peaks; a line through
        # two labelled peaks always does (MWs decrease down the ladder).
        if len(pairs) < 3:
            return True
        fitted = _quadratic(moments)
        if fitted is None:
            return False
        a, b, _ = fitted
        first, last = self.us[pairs[0][0]], self.us[pairs[-1][0]]
        return 2.0 * a * first + b < 0.0 and 2.0 * a * last + b < 0.0

    def _complete(self, pairs: _Pairs, moments: _Moments, offsets: _Offsets) -> float | None:
        if len(pairs) < 2 or not self._falls(pairs, moments):
            return None
        return (
            (_residual(moments) + _spread(offsets)) / PRIOR_DECADES**2
            + MISS_COST * (len(self.zs) - len(pairs))
            + EXTRA_COST * (len(self.ys) - len(pairs))
            + REF_COST * (self.references - self._carried(pairs))
        )

    def cost(self, pairs: _Pairs) -> float | None:
        """A labelling's cost; None with fewer than two pairs, or a quadratic
        that does not fall."""
        moments, offsets = _NO_MOMENTS, (0.0, 0.0, 0.0)
        for i, j in pairs:
            moments = _with(moments, self.us[i], self.zs[j])
            offsets = self._offset(offsets, i, j)
        return self._complete(pairs, moments, offsets)

    def search(self, differs_from: _Pairs | None = None, bound: float = math.inf) -> None:
        """The best labelling below ``bound`` (with ``differs_from``: the best
        one holding a pair that one lacks) into ``best`` and ``best_pairs``;
        ``best_pairs`` stays None if there is none. :class:`_Exhausted` past
        :data:`FIND_BUDGET` labellings tried, over every search."""
        self.best, self.best_pairs = bound, None
        self.differs_from = None if differs_from is None else frozenset(differs_from)
        self._grow((), _NO_MOMENTS, (0.0, 0.0, 0.0), 0, 0)

    def _grow(self, pairs: _Pairs, moments: _Moments, offsets: _Offsets, i: int, j: int) -> None:
        # ``pairs`` so far; the next pair takes a peak from i and a MW from j on.
        k, n, c = len(self.ys), len(self.zs), len(pairs)
        if self.differs_from is None or any(p not in self.differs_from for p in pairs):
            cost = self._complete(pairs, moments, offsets)
            if cost is not None and cost < self.best:
                self.best, self.best_pairs = cost, pairs
        for di, dj in self.steps:
            i2, j2 = i + di, j + dj
            # The peaks and MWs skipped for good, the cheapest first: once they
            # cost too much, so does every step after.
            skipped = EXTRA_COST * (i2 - c) + MISS_COST * (j2 - c)
            if skipped >= self.best:
                break
            if i2 >= k or j2 >= n:
                continue
            left_k, left_n = k - i2 - 1, n - j2 - 1
            floor = (
                skipped + EXTRA_COST * max(0, left_k - left_n) + MISS_COST * max(0, left_n - left_k)
            )
            if floor >= self.best:
                continue
            grown_pairs = (*pairs, (i2, j2))
            # The reference MWs missed for good: those up to j2 not carried now.
            floor += REF_COST * (sum(self.reference[: j2 + 1]) - self._carried(grown_pairs))
            if floor >= self.best:
                continue
            self.tried += 1
            if self.tried > FIND_BUDGET:
                raise _Exhausted
            grown = _with(moments, self.us[i2], self.zs[j2])
            shifted = self._offset(offsets, i2, j2)
            floor += (_residual(grown) + _spread(shifted)) / PRIOR_DECADES**2
            if floor < self.best:
                self._grow(grown_pairs, grown, shifted, i2 + 1, j2 + 1)


def _falling_root(fitted: tuple[float, float, float], z: float) -> float | None:
    """Where ``a u² + b u + c`` falls through ``z``: the root on its falling
    side, or None where it never reaches ``z`` there (past its turn)."""
    a, b, c = fitted
    discriminant = b * b - 4.0 * a * (c - z)
    if discriminant < 0.0:
        return None
    root = math.sqrt(discriminant)
    # At the root (-b - root) / 2a the slope is -root; this form of it has no
    # cancellation, and holds for a line (a = 0) too.
    if -b + root > 0.0:
        u = 2.0 * (c - z) / (-b + root)
    elif a != 0.0:
        u = (-b - root) / (2.0 * a)
    else:
        return None
    return u if 2.0 * a * u + b < 0.0 else None


def find_ladder(
    array: np.ndarray,
    x: float,
    *,
    kda: Sequence[float],
    reference_kda: Sequence[float] = (),
    source: CalibrationPointSource,
    polarity: Polarity,
    other: LadderCurve | None = None,
) -> LadderProposal | None:
    """The ladder a click at ``x`` on its lane shows, labelled with the ladder
    MWs ``kda`` (top to bottom); None when fewer than two peaks stand out
    there (the user then marks bands one by one). Pure: the same pixels and
    arguments give the same proposal.

    The profile down the lane is the mean over the columns within
    :data:`FIND_HALF_X` px of ``x`` of each pixel's deviation from the strip's
    median, turned by the image's ``polarity`` for a band on a marker image
    (``source`` ``visible_marker``), its absolute value for a faint marker on a
    chemiluminescence image (``chemiluminescence_marker``). Its peaks are the
    local maxima standing :data:`SNAP_K` noise sigmas above the membrane around
    them and above the profile around them (:func:`_ladder_peaks`), refined by
    a parabola to y = row + 0.5 + offset, a flat top of equal rows (a band
    saturated or clipped) at its middle. The noise is its steps', but at least
    what the lane's unclipped pixels and its grey levels give
    (:func:`_lane_noise`): a membrane clipped or flat has steps of 0.

    The peaks and the ladder MWs are matched in order, a MW or a peak may stay
    unmatched, and each such labelling is scored as :class:`_Labeller` says:
    the residuals of log10(MW) from a quadratic in y fitted to it, the MWs and
    peaks left unmatched, the reference MWs (``reference_kda``, the ladder's
    darker or coloured bands: colour is not used) not on its strongest peaks,
    and, with ``other`` (the first ladder's curve, for a second ladder), how
    far its height difference from that ladder varies from band to band.

    The best labelling gives a found tick for each MW it labels. Each other MW
    gets a tick predicted from its quadratic, unless the quadratic puts it off
    the image, never reaches it, or puts it out of order with the found ticks.
    ``gap`` is how much worse the best labelling that labels some peak
    otherwise scores; a proposal is doubtful below :data:`GAP_WARN`: the labels
    may be one band off. A search that reaches :data:`FIND_BUDGET` labellings
    keeps the best it found, with a gap of 0.

    ValueError for ``kda`` not positive and strictly decreasing, or a
    ``strip_edge`` source."""
    if source is CalibrationPointSource.STRIP_EDGE:
        raise ValueError("a ladder is found on a marker band source, not a strip edge")
    _ladder_zs(kda)
    height, width = array.shape[:2]
    c0 = max(0, math.ceil(x - FIND_HALF_X - 0.5))
    c1 = min(width - 1, math.floor(x + FIND_HALF_X - 0.5))
    if c1 < c0 or height < 3:
        return None
    window = np.asarray(array[:, c0 : c1 + 1], dtype=float)
    profile, step_noise = _profile(window, source, polarity)
    noise = _lane_noise(window, _signal(window, source, polarity), step_noise)
    peaks = _ladder_peaks(profile, noise)
    labelling = _label(peaks, kda, reference_kda, other)
    if labelling is None:
        return None
    labelled = {i for i, _ in labelling.pairs}
    return LadderProposal(
        x=float(x),
        ticks=_ticks(peaks, kda, labelling, height),
        extra=tuple(y for i, (y, _) in enumerate(peaks) if i not in labelled),
        score=labelling.score,
        gap=labelling.gap,
        doubtful=labelling.gap < GAP_WARN,
    )


def _ladder_zs(kda: Sequence[float]) -> list[float]:
    """log10 of a ladder's MWs; ValueError unless they are positive and
    strictly decreasing."""
    if not all(math.isfinite(m) and m > 0.0 for m in kda):
        raise ValueError("ladder MWs must be positive numbers")
    zs = [math.log10(m) for m in kda]
    if any(not above > below for above, below in itertools.pairwise(zs)):
        raise ValueError("ladder MWs must decrease strictly from top to bottom")
    return zs


@dataclass(frozen=True)
class _Labelling:
    """The best labelling of a lane's peaks (:class:`_Labeller`)."""

    pairs: _Pairs
    score: float
    # How much worse the best labelling that labels some peak otherwise
    # scores: inf when there is none, 0 when the search stopped at its budget.
    gap: float
    # Its quadratic, z = a u² + b u + c, with u = (y - mid) / scale.
    fitted: tuple[float, float, float] | None
    mid: float
    scale: float


def _label(
    peaks: Sequence[tuple[float, float]],
    kda: Sequence[float],
    reference_kda: Sequence[float],
    other: LadderCurve | None,
) -> _Labelling | None:
    """The best labelling of ``peaks`` (``(y, strength)``, top to bottom)
    with the ladder MWs ``kda``, and its gap to the next best; None with fewer
    than two peaks or MWs."""
    zs = _ladder_zs(kda)
    if len(peaks) < 2 or len(zs) < 2:
        return None
    references = {math.log10(m) for m in reference_kda if math.isfinite(m) and m > 0.0}
    labeller = _Labeller(peaks, zs, [z in references for z in zs], other)
    exhausted = False
    try:
        labeller.search()
    except _Exhausted:
        exhausted = True
    best, pairs = labeller.best, labeller.best_pairs
    if pairs is None:
        return None
    second = math.inf
    if not exhausted:
        # The same labels one band up or down bound the next best from above.
        for shift in (-1, 1):
            moved = tuple((i, j + shift) for i, j in pairs if 0 <= j + shift < len(zs))
            cost = labeller.cost(moved) if any(p not in pairs for p in moved) else None
            if cost is not None:
                second = min(second, cost)
        try:
            labeller.search(differs_from=pairs, bound=second)
            second = labeller.best
        except _Exhausted:
            exhausted = True
    moments = _NO_MOMENTS
    for i, j in pairs:
        moments = _with(moments, labeller.us[i], zs[j])
    return _Labelling(
        pairs=pairs,
        score=best,
        gap=0.0 if exhausted else second - best,
        fitted=_quadratic(moments),
        mid=labeller.mid,
        scale=labeller.scale,
    )


def _ticks(
    peaks: Sequence[tuple[float, float]],
    kda: Sequence[float],
    labelling: _Labelling,
    height: float,
) -> tuple[Tick, ...]:
    """The ruler of a labelling, top to bottom: a found tick on each labelled
    peak, and one predicted from its quadratic for each other MW, unless the
    quadratic never reaches that MW on its falling side, or puts it off the
    image (0 to ``height``) or out of order with the found ticks."""
    found = {j: i for i, j in labelling.pairs}
    ticks: list[Tick] = []
    for j, mw in enumerate(kda):
        if j in found:
            y, strength = peaks[found[j]]
            ticks.append(Tick(mw=mw, y=y, found=True, strength=strength))
            continue
        fitted = labelling.fitted
        u = None if fitted is None else _falling_root(fitted, math.log10(mw))
        if u is None:
            continue
        y = labelling.mid + u * labelling.scale
        above = [peaks[i][0] for jj, i in found.items() if jj < j]
        below = [peaks[i][0] for jj, i in found.items() if jj > j]
        if 0.0 <= y <= height and all(y > a for a in above) and all(y < b for b in below):
            ticks.append(Tick(mw=mw, y=y, found=False, strength=None))
    return tuple(ticks)


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
                "the local maximum nearest the click (a flat top of equal rows is one), placed"
                " on the rows' own means across the lane, turned by the polarity (for a faint"
                " marker, the way the band shows against the window's median), at their local"
                " maximum reached from it: refined by a parabola (at most half a row), or the"
                " middle of a flat top, at row + 0.5 (one y per band, wherever it is clicked"
                " from)"
            ),
            "strip_edge": "the same on the size of the row-to-row steps, at the edge between rows",
            "no_band": "the clicked y is kept",
        },
        "find_ladder": {
            "half_x": FIND_HALF_X,
            "k": SNAP_K,
            "background": FIND_BACKGROUND,
            "peaks": (
                "local maxima of the lane's profile (the signal as for a snap, down the whole"
                " lane) standing k noise sigmas above the membrane around them (the profile's"
                " running median over a background share of the rows, at least the snap's"
                " half_y each way) and above the profile around them (their prominence),"
                " refined by a parabola, or at the middle of a flat top of equal rows, at"
                " row + 0.5; a peak's strength is its height above the membrane in noise"
                " sigmas"
            ),
            "noise": (
                "the snap's, but at least the robust sigma of the differences between"
                " neighbouring pixels along each row (each less its column's median), taken"
                " between pixels off the window's lowest and highest values, over the square"
                " root of twice the window's columns, and the grey levels' spacing over the"
                " square root of twelve times the window's columns"
            ),
            "labelling": "peaks and ladder MWs matched in order; either may stay unmatched",
            "prior_decades": PRIOR_DECADES,
            "score": (
                "the squared residuals of log10(MW) from the labelling's least-squares"
                " quadratic in y, which must fall over its peaks; for a second ladder, plus the"
                " weighted squared spread of its height differences from the first ladder, each"
                " over the first ladder's pixels per decade; both over prior_decades squared"
            ),
            "miss_cost": MISS_COST,
            "extra_cost": EXTRA_COST,
            "reference_cost": REF_COST,
            "reference": (
                "per reference MW not carried by one of the strongest labelled peaks, as many as"
                " there are reference MWs; colour is not used"
            ),
            "predicted": (
                "each ladder MW left unmatched, from the winning quadratic on its falling side;"
                " not drawn off the image or out of order with the found ticks"
            ),
            "gap": "how much worse the best labelling that labels some peak otherwise scores",
            "gap_warn": GAP_WARN,
            "budget": FIND_BUDGET,
            "budget_reached": "the best labelling found is kept, with a gap of 0 (doubtful)",
        },
    }
