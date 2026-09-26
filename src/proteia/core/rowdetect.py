# SPDX-License-Identifier: Apache-2.0
"""Detect one band per declared lane inside a user-drawn row box.

The user drags a box over one protein's row and the lane table says how many
lanes it spans; :func:`detect_row` proposes one box per lane, all of one shared
size, or reports the lane empty. Pure and GUI-independent: it reads a 2-D
analysis array (never writes it) and the image's stored background, and returns
a :class:`RowDetection`. Committing boxes to a project is the operation's job.

Contract:

* The row box ``(x0, y0, x1, y1)`` is clipped to the image first. It must span
  every declared lane, the empty end lanes included: lanes are read from the
  bands, and a lane the box leaves out cannot be placed.
* One slot per declared lane, in lane order: a rect, or None for no band. An
  empty slot never shifts another lane's index.
* Every rect has the one shared size, no two overlap (:func:`~proteia.core.model.overlaps`),
  and all lie inside the clipped row box. Coordinates are plain ints placed with
  integer arithmetic only, so shifting the image and the row by ``(dx, dy)``
  shifts every rect by exactly ``(dx, dy)``.
* Deterministic: no randomness, and the robust fits use a fixed-stride subsample.
* Polarity is applied once. Detection measures its own local background (a
  plane over the row's band-free pixels); the stored background is only
  compared with it (``bg_offset``), never subtracted, so a stored net keeps
  ``image.background``.
* When the bands do not show which lane each is in, the result carries a
  refusing flag (:data:`REFUSING_FLAGS`) and the caller must propose nothing.

Pipeline, in crop coordinates (rects are offset back at the end):

1. Validate and clip the row box (:class:`RowDetectError`).
2. Stage 1: a plane ``a + b*x + c*y`` robustly fitted over the crop, or the
   stored background where that fits the membrane better. The detection signal
   is the 3x5 box-smoothed crop on the band side of it; its noise is the
   smaller of two robust spreads.
3. Rows of the row: a hump of the row-mean signal cut by the top or bottom box
   edge (a neighbouring row) is left out up to its valley, if an interior hump
   remains.
4. Components of the signal above ``NOISE_K`` sigma that reach ``DETECT_K``
   sigma (hysteresis), with a 30%-core at least a dust floor wide, a peak off
   the box's edge rows, and not a flat structure spanning the rows (a streak or
   stain). Their column profile is cut into pieces between peaks.
5. An ordered dynamic programme over the pieces and a pitch search assigns
   each piece to one lane, or a touching run to several, with empty lanes as
   gaps; the second-best reading measures how certain that is.
6. Per lane: growth with the click's rule (:func:`~proteia.core.grow.grow_region`
   at ``EXTENT_LEVEL`` of the lane's strongest pixel) between the lane's walls.
7. Stage 2: the plane and the noise again from the band-free pixels of the
   row, then steps 3 to 6 again.
8. Each band's lane: its separate components counted (peaks split as pieces
   are along x). Empty lanes get a reason; flags; one shared size by
   :data:`SIZE_RULE`, capped by the lane spacing and the box; bounded isotonic
   placement (:func:`~proteia.core.boxes.place_in_row`).

Flags (:attr:`RowDetection.flags`):

* ``lanes_outside_row`` (refusing): an empty end lane's expected centre lies
  outside the row box, so the box does not cover every declared lane;
* ``ambiguous_lanes`` (refusing): a different lane reading costs less than
  :data:`AMBIGUITY_MARGIN` more than the chosen one;
* ``background_mismatch``: a box's local background differs from the stored
  one by more than :data:`BG_WARN_K` pixel sigmas;
* ``size_outlier``: an extent above :data:`SIZE_GUARD` times the median was
  left out of the shared size (``"max_guarded"``);
* ``multiple_components``: a lane holds a second, separate component (see
  ``components``); its box is grown from the lane's strongest pixel, as a
  click there would be (quantifying doublets is #58's).

Every setting is a module constant, reported by :func:`settings`.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from pydantic import JsonValue
from scipy.ndimage import (
    find_objects,
    label,
    maximum_filter1d,
    maximum_position,
    median_filter,
    uniform_filter,
)
from scipy.signal import find_peaks
from scipy.special import ndtri
from skimage.morphology import reconstruction

from proteia.core.boxes import place_in_row
from proteia.core.grow import NOISE_K, REL_THRESHOLD, grow_region
from proteia.core.model import BoxSize, Rect

# --- Domain settings (maintainer decisions on #51) ---

DETECT_K: Final = 6.0  # a band's peak reaches this many smoothed-noise sigmas
EXTENT_LEVEL: Final = REL_THRESHOLD  # sized extent: this fraction of the band's own peak (click)
SIZE_RULE: Final = "max_guarded"  # the shared size: the largest extent, outliers left out
SIZE_GUARD: Final = 2.0  # an extent above this times the median does not set the size

# --- Technical settings ---

SMOOTH: Final = (3, 5)  # box smoothing of the detection signal, (rows, cols)
MIN_WIDTH_PX: Final = 7  # a band's 30%-core is at least this wide (the kernel + 2 px)...
MIN_WIDTH_PITCH: Final = 0.15  # ...and this fraction of the box pitch; also the despeckle width
DESPECKLE_MIN: Final = 3  # the despeckle median is at least this wide, px
EDGE_HUMP: Final = 0.5  # a row-mean hump is cut by the box edge if the edge is >= this of it
ROW_WALK_TOL: Final = 0.02  # the row-mean walk passes rises and dips below this of its peak
ROW_MIN_ROWS: Final = 5  # a neighbouring row is left out only of a row this high or higher...
ROW_MIN_KEEP: Final = 3  # ...and only if at least this many rows remain
FLAT_EDGE: Final = 0.8  # a row-spanning core whose end rows are >= this of its peak: a streak
VALLEY_FRAC: Final = 0.75  # cut two near peaks at a low point below this of the lower (x and y)
GAP_FRAC: Final = 0.1  # cut any two peaks at a low point below this of the lower (empty lane)
NEAR_PEAKS: Final = 1.5  # peaks closer than this many box pitches hold no band between them
ENVELOPE_HALF: Final = 0.5  # piece ends trimmed against the local max within +/- this pitch
ENVELOPE_MIN: Final = 2  # ...and at least this many px
REDUCE_MASS: Final = 0.25  # too many pieces: drop the weaker of the closest pair below this
RUN_LANE_MIN: Final = 0.6  # one lane of a touching run spans at least this many pitches
Q_WINDOW: Final = 2  # a piece covers 1 lane, or its width / pitch rounded, +/- this many
CELL_SEED: Final = 0.25  # a touching cell's seed: this far in from each side, x its width
SPACING_TOL: Final = 0.15  # relative s.d. of the lane spacing per step
GAP_SEARCH: Final = 1  # the empty lanes of a gap are tried within this many of its estimate
END_LO: Final = 0.1  # an end lane's centre sits at least this many pitches inside the box...
END_HI: Final = 1.0  # ...and at most this many
END_TOL: Final = 0.1  # softness of END_LO and END_HI, in pitches
END_SYM_TOL: Final = 0.25  # tolerated left/right margin asymmetry, in pitches
SINGLE_MAX_WIDTH: Final = 1.25  # one band wider than this many pitches is penalised
SPLIT_PENALTY: Final = 1.0  # cost of each extra lane carved out of one touching run
PITCH_PRIOR_TOL: Final = 0.25  # weak prior: the pitch is about the box width / n
PITCH_RANGE: Final = (0.55, 1.30)  # pitch search, as fractions of the box width / n
PITCH_STEPS: Final = (0.05, 0.01)  # coarse step over PITCH_RANGE, then the fine step...
PITCH_REFINE: Final = 0.04  # ...within +/- this of the best coarse fraction
AMBIGUITY_MARGIN: Final = 4.0  # a different lane reading this close in cost is ambiguous
TAIL_Q: Final = 97.5  # stage-2 noise also from this percentile of |deviation| (heavy tails)
BG_CLIP_K: Final = 3.0  # robust fits keep values within this many sigmas of the centre
BG_ITER: Final = 30  # iteration cap of the robust fits
BG_MIN_KEEP: Final = 0.10  # a plane fit keeps this fraction; stage 2 needs it of the crop
BG_MIN_PIXELS: Final = 200  # stage 2 needs at least this many band-free pixels
BG_GUARD: Final = (0.5, 0.25)  # stage-2 exclusion around a band, x its own (h, w)...
BG_GUARD_MIN: Final = 2  # ...and at least this many px
MIN_FIT: Final = 16  # fewest values a robust fit or noise estimate is made from
FIT_MAX_PIXELS: Final = 20000  # fits and noise estimates use a fixed-stride subsample
NOISE_FLOOR_FRAC: Final = 1e-3  # sigma floor: this fraction of the crop's range
EMPTY_WINDOW: Final = 0.3  # an empty lane's SNR is read within +/- this pitch of its centre
BG_WARN_K: Final = 3.0  # background_mismatch beyond this many pixel sigmas
MIN_BOX: Final = 2  # smallest box side, as boxes.grow_to_fit

# --- Vocabularies ---

SIZE_RULES: Final = ("max", "max_guarded")
REFUSING_FLAGS: Final = ("lanes_outside_row", "ambiguous_lanes")
WARNING_FLAGS: Final = ("background_mismatch", "size_outlier", "multiple_components")

RowDetectErrorCode = Literal["invalid_row", "row_outside_image", "row_too_small"]
LaneReason = Literal["band", "no_band", "artefact", "edge_signal", "unassigned"]

_TAIL_Z: Final = float(ndtri(0.5 + TAIL_Q / 200.0))  # Gaussian |z| at the TAIL_Q percentile
_P_SIGMA: Final = 68.27  # the percentile of |deviation| at one Gaussian sigma
_MAD_SIGMA: Final = 1.4826  # Gaussian sigma per median absolute deviation
_CROSS: Final = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)  # 4-connectivity, as label()


class RowDetectError(ValueError):
    """A row box :func:`detect_row` cannot use. ``code`` is stable for callers:

    * ``invalid_row``: not a 2-D array of finite values, ``n_lanes`` not a
      positive int, the row not four ints with ``x0 < x1`` and ``y0 < y1``, or
      ``background`` not a finite real number;
    * ``row_outside_image``: nothing of the row is left after clipping it to
      the image;
    * ``row_too_small``: the clipped row is narrower than :data:`MIN_BOX` px per
      lane or lower than the smoothing kernel (``SMOOTH[0]`` rows).
    """

    def __init__(self, code: RowDetectErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: RowDetectErrorCode = code


@dataclass(frozen=True)
class LaneDetection:
    """What :func:`detect_row` found in one declared lane (image coordinates).

    * ``rect``: the proposed box, of the shared size; None for an empty lane.
    * ``reason``: ``band``, or why the lane is empty: ``no_band`` (nothing
      reaches ``DETECT_K``; ``snr`` says how close it came), ``artefact`` (a
      rejected streak or stain covers it), ``edge_signal`` (only the rows left
      out as a neighbouring row reach ``DETECT_K``), ``unassigned`` (signal in
      the row's rows reaches ``DETECT_K`` but is in no assigned piece: a piece
      dropped for want of lanes, or too weak beside the bands around it).
    * ``snr``: the strongest smoothed signal in the lane (the growth seed; for an
      empty lane, within ``EMPTY_WINDOW`` pitch of its centre) over the noise.
    * ``extent``: the grown extent before sizing, None for an empty lane.
    * ``expected_x``: the lane's expected centre, also for an empty lane: between
      the present lanes by index, past them by the pitch, or an even split of
      the row when no lane holds a band.
    * ``bg_offset``: the band-side difference between the local detection
      background under the box and the stored background, in pixel sigmas;
      None for an empty lane.
    * ``components``: the separate peaks of the lane's detection signal, in its
      x-range and between its walls; a peak is separate from a higher one if
      the saddle between them is ``DETECT_K`` sigma below it and at most
      ``VALLEY_FRAC`` of it, as two pieces along x. 1 for a lone band (noise on
      its top or a dent in it never is), 2 or more for several (a doublet,
      however weak its second band; a band split by a bubble), 0 when empty.
    """

    lane: int
    rect: Rect | None
    reason: LaneReason
    snr: float
    extent: Rect | None
    expected_x: float
    bg_offset: float | None
    components: int


@dataclass(frozen=True)
class RowDetection:
    """The result of :func:`detect_row`.

    ``lanes`` holds one :class:`LaneDetection` per declared lane, indexed by
    lane. ``size`` is the shared box size, None when no lane holds a band.
    ``pitch`` is the fitted lane pitch in px, None when there was nothing to
    fit. ``noise`` is the sigma of the smoothed detection signal and
    ``pixel_noise`` the per-pixel sigma, both in pixel units. ``flags`` are the
    stable codes of the module docstring, refusing ones first. ``notes`` are
    diagnostics in plain words; they give image x and count lanes from 1, as the
    user does (index fields stay 0-based). ``cost`` is the cost of the chosen
    lane reading and ``margin`` how much more the best different reading costs
    (``inf`` when there is none); both None when there was nothing to assign.
    """

    lanes: tuple[LaneDetection, ...]
    size: BoxSize | None
    pitch: float | None
    noise: float
    pixel_noise: float
    flags: tuple[str, ...]
    notes: tuple[str, ...]
    cost: float | None
    margin: float | None

    @property
    def slots(self) -> tuple[Rect | None, ...]:
        """One rect or None per declared lane, in lane order."""
        return tuple(lane.rect for lane in self.lanes)

    @property
    def refused(self) -> bool:
        """True when a refusing flag is set: the caller proposes nothing."""
        return any(flag in REFUSING_FLAGS for flag in self.flags)


# --- Robust statistics, background and noise ---


def _center_scale(v: np.ndarray, floor: float, tail: bool = True) -> tuple[float, float]:
    """``(centre, sigma)`` of values that are mostly noise about one level, with
    structure on either side.

    The centre is the median of the values kept so far. On each side of it,
    sigma is the larger of the 68.27th percentile of the distances (one Gaussian
    sigma) and their ``TAIL_Q`` percentile over its Gaussian value (the same for
    Gaussian noise, larger for heavy tails such as JPEG blocks); the smaller side
    counts, so structure on one side (bands, bright artefacts) cannot inflate it.
    Values beyond ``BG_CLIP_K`` sigma are dropped until the kept set is stable
    (at most ``BG_ITER`` rounds). Noise-free data gives ``floor``. With
    ``tail=False`` only the 68.27th percentile is used: robust to spatial
    structure that fills the tail (a light strip), blind to heavy tails.
    """
    v = np.asarray(v, float).ravel()
    keep = np.ones(v.size, bool)
    m, s = float(np.median(v)), floor
    for _ in range(BG_ITER):
        vk = v[keep]
        m = float(np.median(vk))
        sides = []
        for d in (m - vk[vk <= m], vk[vk >= m] - m):
            c, t = np.percentile(d, [_P_SIGMA, TAIL_Q])
            sides.append(max(float(c), float(t) / _TAIL_Z) if tail else float(c))
        s = max(min(sides), floor)
        new = np.abs(v - m) <= BG_CLIP_K * s
        if new.sum() < MIN_FIT or np.array_equal(new, keep):
            break
        keep = new
    return m, s


def _pixel_noise(crop: np.ndarray) -> float:
    """Per-pixel noise sigma from the MAD of horizontal neighbour differences."""
    d = np.diff(crop, axis=1).ravel()
    if d.size < 2:
        return 0.0
    return float(_MAD_SIGMA * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0))


def _noise_floor(crop: np.ndarray) -> float:
    """The smallest sigma believed: a fraction of the range, and the quantisation
    noise of integer-valued data."""
    floor = NOISE_FLOOR_FRAC * float(np.ptp(crop))
    if np.array_equal(crop, np.round(crop)):
        floor = max(floor, 1.0 / math.sqrt(12.0))
    return max(floor, 1e-9)


def _subsample(idx: np.ndarray) -> np.ndarray:
    """At most ``FIT_MAX_PIXELS`` of ``idx``, at a fixed stride."""
    if idx.size > FIT_MAX_PIXELS:
        idx = idx[:: int(math.ceil(idx.size / FIT_MAX_PIXELS))]
    return idx


def _fit_plane(crop: np.ndarray, mask: np.ndarray, floor: float) -> np.ndarray | None:
    """Coefficients ``(a, b, c)`` of ``a + b*x + c*y`` fitted to the ``mask``
    pixels, keeping residuals within ``BG_CLIP_K`` sigmas of their centre (bands
    on one side, bright artefacts on the other). The first keep set is taken
    about the membrane level, not a mean. None if too few pixels survive."""
    w = crop.shape[1]
    idx = _subsample(np.flatnonzero(mask.ravel()))
    if idx.size < MIN_FIT:
        return None
    ys, xs = np.divmod(idx, w)
    design = np.column_stack([np.ones(idx.size), xs.astype(float), ys.astype(float)])
    vals = crop.ravel()[idx]
    m0, s0 = _center_scale(vals, floor)
    keep = np.abs(vals - m0) <= BG_CLIP_K * s0
    coef = None
    for _ in range(BG_ITER):
        if keep.sum() < max(MIN_FIT, BG_MIN_KEEP * idx.size):
            return None
        coef, *_ = np.linalg.lstsq(design[keep], vals[keep], rcond=None)
        resid = vals - design @ coef
        m, s = _center_scale(resid[keep], floor)
        new = np.abs(resid - m) <= BG_CLIP_K * s
        if np.array_equal(new, keep):
            break
        keep = new
    return coef


def _plane(coef: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    return coef[0] + coef[1] * gx + coef[2] * gy


def _membrane_spread(v: np.ndarray, surface: np.ndarray | float, sign: float) -> float:
    """Robust spread of the membrane-side values about ``surface`` (inf if too few)."""
    r = sign * (surface - v)
    mem = -r[r < 0]
    return float(_MAD_SIGMA * np.median(mem)) if mem.size >= MIN_FIT else math.inf


@dataclass(frozen=True)
class _Signal:
    plane: np.ndarray  # the detection background (crop coordinates)
    s_sm: np.ndarray  # smoothed signal, band side, >= 0
    s_ds: np.ndarray  # the same from the despeckled crop: shape decisions only
    sigma_sm: float
    sigma_px: float


def _despeckle_width(wc: int, n: int) -> int:
    """Odd width of the running median applied along x for the shape signal:
    ``MIN_WIDTH_PITCH`` of the box pitch, at least ``DESPECKLE_MIN``. It removes dust and specks
    narrower than about half of it; a band, wide along x, keeps its plateau and
    edges. Noise and thresholds never use it (a median distorts JPEG noise)."""
    k = max(DESPECKLE_MIN, int(round(MIN_WIDTH_PITCH * wc / n)))
    return k if k % 2 else k + 1


def _despeckle(crop: np.ndarray, k: int) -> np.ndarray:
    """The running median of width ``k`` along x of every row of ``crop``: the
    same values as ``median_filter(crop, size=(1, k), mode="nearest")``, row by
    row because scipy's 1-D median costs about the same at any ``k``, while the
    2-D call costs O(k) per pixel (a 2000 px row over two lanes: k = 151)."""
    out = np.empty_like(crop)
    for y in range(crop.shape[0]):
        median_filter(crop[y], size=k, mode="nearest", output=out[y])
    return out


def _signal(
    crop: np.ndarray,
    crop_ds: np.ndarray,
    plane: np.ndarray,
    sign: float,
    sigma_px: float,
    floor: float,
    free: np.ndarray | None,
) -> _Signal:
    """The detection and shape signals over ``plane`` and their noise: stage 1
    (``free`` None) or stage 2 (noise from the band-free pixels ``free``)."""
    hc, wc = crop.shape
    ky, kx = min(SMOOTH[0], hc), min(SMOOTH[1], wc)
    r = sign * (plane - uniform_filter(crop, size=(ky, kx), mode="nearest"))
    white = sigma_px / math.sqrt(ky * kx)
    sfloor = max(white, floor / math.sqrt(ky * kx))
    if free is None or free.sum() < MIN_FIT:
        # Stage 1 only has to find the clear bands to mask for stage 2: the
        # smaller of the membrane-side spread about the plane (inflated by a
        # light strip) and the central spread about the median (inflated a
        # little by many bands).
        mem = -r[r < 0]
        est = float(_MAD_SIGMA * np.median(mem)) if mem.size >= MIN_FIT else white
        sigma_sm, offset = max(est, sfloor), 0.0
        idx = _subsample(np.arange(r.size))
        off2, s2 = _center_scale(r.ravel()[idx], sfloor, tail=False)
        if s2 < sigma_sm:
            sigma_sm, offset = s2, off2
    else:
        idx = _subsample(np.flatnonzero(free.ravel()))
        offset, sigma_sm = _center_scale(r.ravel()[idx], sfloor)
        # Pixel noise as measured on the membrane (JPEG: differences are often 0).
        sigma_px = max(sigma_px, _center_scale((plane - crop).ravel()[idx], floor)[1])
    r_ds = sign * (plane - uniform_filter(crop_ds, size=(ky, kx), mode="nearest"))
    return _Signal(
        plane, np.maximum(r - offset, 0.0), np.maximum(r_ds - offset, 0.0), sigma_sm, sigma_px
    )


# --- In-row signal: rows, components, pieces ---


def _row_rows(s: np.ndarray, sigma_sm: float) -> tuple[int, int]:
    """Rows ``[lo, hi)`` that hold the row. A hump of the row-mean signal cut by
    the top or bottom box edge (the edge value is at least ``EDGE_HUMP`` of its
    peak: a neighbouring row) is left out up to its valley, if an interior hump
    remains."""
    hc = s.shape[0]
    v = s.mean(axis=1)
    if hc < ROW_MIN_ROWS or v.max() <= 0:
        return 0, hc
    tol = ROW_WALK_TOL * float(v.max())
    sig = NOISE_K * sigma_sm / math.sqrt(max(1, s.shape[1] / SMOOTH[1]))

    def walk(order: list[int]) -> tuple[bool, int]:
        k = 0
        while k + 1 < len(order) and v[order[k + 1]] >= v[order[k]] - tol:
            k += 1
        peak = v[order[k]]
        edge = peak > sig and v[order[0]] >= EDGE_HUMP * peak
        while k + 1 < len(order) and v[order[k + 1]] <= v[order[k]] + tol:
            k += 1
        return edge, order[k]

    top_edge, top_valley = walk(list(range(hc)))
    bot_edge, bot_valley = walk(list(range(hc - 1, -1, -1)))
    lo = top_valley if top_edge else 0
    hi = bot_valley + 1 if bot_edge else hc
    if hi - lo < ROW_MIN_KEEP:
        return 0, hc
    inner = v[lo:hi]
    interior = inner.max() > sig and 0 < int(np.argmax(inner)) < inner.size - 1
    return (lo, hi) if interior else (0, hc)


@dataclass(frozen=True)
class _Piece:
    """A stretch of the column profile: crop x ``[l, r)``, its peak and mass."""

    l: float  # noqa: E741
    r: float
    peak: float
    mass: float

    @property
    def w(self) -> float:
        return self.r - self.l

    @property
    def c(self) -> float:
        return 0.5 * (self.l + self.r)


Region = tuple[slice, slice]


@dataclass(frozen=True)
class _Candidates:
    kept: np.ndarray  # mask of the kept components
    rows: tuple[int, int]  # rows of the row (edge humps left out)
    rejected: list[tuple[str, Region]]  # (reason, region) of rejected structures
    pieces: list[_Piece]


def _split_run(p: np.ndarray, a: int, b: int, min_dip: float, near: float) -> list[tuple[int, int]]:
    """Split the run ``[a, b)`` of the profile into parts. Peaks are local maxima
    with prominence at least ``min_dip``. Two neighbouring peaks closer than
    ``near`` (no room for a band between them) are cut apart at the lowest point
    between them if it is below ``VALLEY_FRAC`` of the lower peak; peaks farther
    apart stay together (a weaker band between them is a shoulder, not a gap)
    unless the profile between them falls below ``GAP_FRAC`` of the lower peak
    (an empty lane between two bands whose tails touch)."""
    seg = p[a:b]
    found, _ = find_peaks(np.concatenate([[0.0], seg, [0.0]]), prominence=min_dip)
    idx = [int(i) - 1 for i in found if 0 <= i - 1 < seg.size]
    cuts = []
    for i, j in zip(idx[:-1], idx[1:], strict=True):
        v = i + int(np.argmin(seg[i : j + 1]))
        lower = min(seg[i], seg[j])
        if (j - i < near and seg[v] <= VALLEY_FRAC * lower) or seg[v] <= GAP_FRAC * lower:
            cuts.append(v)
    bounds = [0, *cuts, seg.size]
    return [(a + s0, a + s1) for s0, s1 in zip(bounds[:-1], bounds[1:], strict=True)]


def _flat(s: np.ndarray, lo: int, hi: int, c0: int, c1: int) -> bool:
    """True if the columns ``[c0, c1)`` hold no band-like hump across the rows
    ``[lo, hi)``: the mean profile stays above ``REL_THRESHOLD`` of its peak on
    every row and both end rows reach ``FLAT_EDGE`` of it (a vertical streak, a
    broad stain, or a box that cuts the bands)."""
    v = s[lo:hi, c0:c1].mean(axis=1)
    top = float(v.max())
    return top > 0 and bool(v.min() >= REL_THRESHOLD * top) and min(v[0], v[-1]) >= FLAT_EDGE * top


def _candidates(sig: _Signal, n: int) -> _Candidates:
    """The kept components and the pieces of their column profile."""
    s_all = sig.s_sm
    hc, wc = s_all.shape
    lo, hi = _row_rows(s_all, sig.sigma_sm)
    s = np.zeros_like(s_all)
    s[lo:hi] = s_all[lo:hi]
    ds = np.zeros_like(s_all)
    ds[lo:hi] = sig.s_ds[lo:hi]
    rejected: list[tuple[str, Region]] = []
    if lo > 0:
        rejected.append(("edge_signal", (slice(0, lo), slice(0, wc))))
    if hi < hc:
        rejected.append(("edge_signal", (slice(hi, hc), slice(0, wc))))
    thr = NOISE_K * sig.sigma_sm
    # Width floor: the smoothing kernel + 2 px, and a fraction of the lane
    # spacing (dust is far narrower than a lane; a narrow band is not).
    min_w = min(max(MIN_WIDTH_PX, MIN_WIDTH_PITCH * wc / n), wc)
    lab, count = label(s > thr)
    kept = np.zeros(s.shape, bool)
    if count:
        regions = find_objects(lab)
        # Only components holding a pixel at the detection level (hysteresis).
        strong = np.unique(lab[s >= DETECT_K * sig.sigma_sm])
        for k in strong[strong > 0] - 1:
            sl = regions[k]
            comp = lab[sl] == k + 1
            local = np.where(comp, s[sl], 0.0)
            ly, lx = divmod(int(np.argmax(local)), local.shape[1])
            peak = float(local[ly, lx])
            py = ly + sl[0].start
            core = grow_region(local, (lx, ly), max(REL_THRESHOLD * peak, thr))
            if core is None:
                continue
            cx0, cy0, cx1, cy1 = core
            cy0, cy1 = cy0 + sl[0].start, cy1 + sl[0].start
            if py == 0 or py == hc - 1:
                rejected.append(("edge_signal", sl))
                continue
            if cy0 <= lo and cy1 >= hi and _flat(s, lo, hi, sl[1].start + cx0, sl[1].start + cx1):
                rejected.append(("artefact", sl))
                continue
            # Width on the despeckled signal: dust and specks vanish under the
            # running median; a band keeps its plateau.
            dloc = np.where(comp, ds[sl], 0.0)
            dpk = float(dloc.max())
            if dpk < max(REL_THRESHOLD * peak, thr):
                continue  # gone under the running median: a speck
            dy, dx = divmod(int(np.argmax(dloc)), dloc.shape[1])
            dcore = grow_region(dloc, (dx, dy), max(REL_THRESHOLD * dpk, thr))
            if dcore is None or dcore[2] - dcore[0] < min_w:
                continue  # below the width floor: noise, dust
            kept[sl] |= comp
    prof = np.where(kept, ds, 0.0).max(axis=0)
    half = max(ENVELOPE_MIN, int(round(ENVELOPE_HALF * wc / n)))
    pieces: list[_Piece] = []
    padded = np.concatenate([[False], prof > 0, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for ra, rb in zip(edges[0::2], edges[1::2], strict=True):
        for a, b in _split_run(
            prof, int(ra), int(rb), DETECT_K * sig.sigma_sm, NEAR_PEAKS * wc / n
        ):
            # Trim each end to REL_THRESHOLD of the local envelope; the inside of
            # a run stays whole, so a weaker band touching a stronger one (a
            # shoulder) stays in the piece.
            seg = prof[a:b]
            env = maximum_filter1d(seg, size=2 * half + 1, mode="nearest")
            inside = np.flatnonzero(seg >= REL_THRESHOLD * env)
            if inside.size == 0:
                continue
            j0, j1 = int(inside[0]), int(inside[-1])
            if j1 + 1 - j0 < min_w:
                continue
            i = j0 + int(np.argmax(seg[j0 : j1 + 1]))
            if _flat(s, lo, hi, a + j0, a + j1 + 1):  # a stain attached to a band
                rejected.append(("artefact", (slice(lo, hi), slice(a + j0, a + j1 + 1))))
                kept[:, a + j0 : a + j1 + 1] = False
                continue
            pieces.append(_Piece(a + j0, a + j1 + 1, float(seg[i]), float(seg[j0 : j1 + 1].sum())))
    return _Candidates(kept, (lo, hi), rejected, pieces)


def _reduce(pieces: list[_Piece], n: int, x_offset: int, notes: list[str]) -> list[_Piece]:
    """More pieces than lanes: drop the weaker of the closest pair if its mass is
    below ``REDUCE_MASS`` of the other's, otherwise merge the pair."""
    pieces = list(pieces)
    while len(pieces) > n:
        gaps = [pieces[i + 1].c - pieces[i].c for i in range(len(pieces) - 1)]
        i = int(np.argmin(gaps))
        a, b = pieces[i], pieces[i + 1]
        if min(a.mass, b.mass) < REDUCE_MASS * max(a.mass, b.mass):
            drop = i if a.mass < b.mass else i + 1
            p = pieces.pop(drop)
            notes.append(f"dropped a weak piece at x={x_offset + p.l:.0f}..{x_offset + p.r:.0f}")
        else:
            notes.append(f"merged pieces at x={x_offset + a.l:.0f}..{x_offset + b.r:.0f}")
            pieces[i : i + 2] = [_Piece(a.l, b.r, max(a.peak, b.peak), a.mass + b.mass)]
    return pieces


# --- Lane assignment: an ordered dynamic programme keeping the two best costs ---


def _first(p: _Piece, q: int) -> float:
    """Centre of the first of the ``q`` lanes of piece ``p``."""
    return p.c if q == 1 else p.l + 0.5 * p.w / q


def _last(p: _Piece, q: int) -> float:
    """Centre of the last of the ``q`` lanes of piece ``p``."""
    return p.c if q == 1 else p.r - 0.5 * p.w / q


def _piece_cost(p: _Piece, q: int, pitch: float) -> float:
    if q == 1:
        over = max(0.0, p.w / pitch - SINGLE_MAX_WIDTH)
        return (over / SPACING_TOL) ** 2
    dev = (p.w / q - pitch) / pitch
    return (q - 1) * (dev / SPACING_TOL) ** 2 + SPLIT_PENALTY * (q - 1)


def _q_options(p: _Piece, pitch: float, n: int) -> list[int]:
    """How many lanes piece ``p`` may cover: 1, or within ``Q_WINDOW`` of ``w / pitch``,
    bounded by ``RUN_LANE_MIN`` pitch per lane and by ``n``."""
    qmax = max(1, min(n, int(p.w // (RUN_LANE_MIN * pitch))))
    mid = int(round(p.w / pitch))
    return sorted({1} | set(range(max(1, mid - Q_WINDOW), min(qmax, mid + Q_WINDOW) + 1)))


def _end_cost(s_l: float, s_r: float) -> float:
    """Cost of the end margins, in pitches: symmetric, and each within
    ``[END_LO, END_HI]`` (softly)."""
    c = ((s_l - s_r) / END_SYM_TOL) ** 2
    for s in (s_l, s_r):
        c += (max(0.0, END_LO - s) / END_TOL) ** 2 + (max(0.0, s - END_HI) / END_TOL) ** 2
    return c


@dataclass(frozen=True)
class _Assignment:
    cost: float
    second: float  # best cost of a different lane reading at this pitch
    pitch: float
    lanes: tuple[tuple[int, int], ...]  # per piece: (first lane, lanes covered)


def _dp(pieces: list[_Piece], n: int, box_w: float, pitch: float) -> _Assignment | None:
    """Best and second-best lane readings at one pitch. A state is (piece, its
    lane count q, the relative lane of its first lane); each keeps its two best
    costs, over numpy arrays indexed by the relative lane. The first piece's q
    is an outer loop; end margins are costed for every shift of the reading."""
    m = len(pieces)
    qs = [_q_options(p, pitch, n) for p in pieces]
    best_cost = math.inf
    best_lanes: tuple[tuple[int, int], ...] = ()
    totals: list[float] = []
    for q0 in qs[0]:
        c1 = np.full(n, math.inf)
        c1[0] = _piece_cost(pieces[0], q0, pitch)
        layers = [{q0: (c1, np.full(n, math.inf))}]
        backs: list[dict[int, tuple[np.ndarray, np.ndarray]]] = [{}]
        for j in range(1, m):
            cur: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            back: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for q in qs[j]:
                pc = _piece_cost(pieces[j], q, pitch)
                n1, n2 = np.full(n, math.inf), np.full(n, math.inf)
                bq, bg = np.full(n, -1), np.full(n, -1)
                for qp, (p1, p2) in layers[-1].items():
                    dx = _first(pieces[j], q) - _last(pieces[j - 1], qp)
                    gh = dx / pitch - 1.0
                    for g in range(
                        max(0, math.floor(gh) - GAP_SEARCH), max(0, math.ceil(gh) + GAP_SEARCH) + 1
                    ):
                        shift = qp + g
                        if shift >= n:
                            break
                        cst = ((dx - (g + 1) * pitch) / pitch) ** 2 / (
                            (g + 1) * SPACING_TOL**2
                        ) + pc
                        a1, a2 = p1[: n - shift] + cst, p2[: n - shift] + cst
                        t1, t2 = n1[shift:], n2[shift:]
                        better = a1 < t1
                        second = np.minimum(np.maximum(t1, a1), np.minimum(t2, a2))
                        bq[shift:][better] = qp
                        bg[shift:][better] = g
                        n1[shift:] = np.minimum(t1, a1)
                        n2[shift:] = second
                cur[q] = (n1, n2)
                back[q] = (bq, bg)
            layers.append(cur)
            backs.append(back)
        c_first = _first(pieces[0], q0)
        for q, (arr1, arr2) in layers[-1].items():
            c_last = _last(pieces[-1], q)
            for k in np.flatnonzero(np.isfinite(arr1)):
                span = int(k) + q
                if span > n:
                    continue
                for g0 in range(0, n - span + 1):
                    ec = _end_cost(c_first / pitch - g0, (box_w - c_last) / pitch - (n - span - g0))
                    t1, t2 = float(arr1[k]) + ec, float(arr2[k]) + ec
                    totals.extend([t1, t2])
                    if t1 < best_cost:
                        path = [(int(k), q)]
                        kk, qq = int(k), q
                        for jj in range(m - 1, 0, -1):
                            bq, bg = backs[jj][qq]
                            qp, g = int(bq[kk]), int(bg[kk])
                            kk, qq = kk - qp - g, qp
                            path.append((kk, qq))
                        path.reverse()
                        best_cost = t1
                        best_lanes = tuple((a + g0, b) for a, b in path)
    if not best_lanes:
        return None
    rest = sorted(t for t in totals if math.isfinite(t))
    return _Assignment(best_cost, rest[1] if len(rest) > 1 else math.inf, pitch, best_lanes)


def _assign(pieces: list[_Piece], n: int, box_w: float) -> tuple[_Assignment | None, float]:
    """The best reading over a coarse-to-fine pitch search, and the cost of the
    best reading with different lanes (at the same pitch or any other)."""
    p0 = box_w / n
    lo, hi = PITCH_RANGE
    coarse, fine = PITCH_STEPS
    found: list[_Assignment] = []

    def run(fracs: np.ndarray) -> None:
        for f in fracs:
            a = _dp(pieces, n, box_w, f * p0)
            if a is not None:
                prior = ((f - 1.0) / PITCH_PRIOR_TOL) ** 2
                found.append(_Assignment(a.cost + prior, a.second + prior, a.pitch, a.lanes))

    run(np.arange(lo, hi + 1e-9, coarse))
    if not found:
        return None, math.inf
    f0 = min(found, key=lambda a: a.cost).pitch / p0
    run(np.arange(max(lo, f0 - PITCH_REFINE), min(hi, f0 + PITCH_REFINE) + 1e-9, fine))
    best = min(found, key=lambda a: a.cost)
    alt = min([best.second] + [a.cost for a in found if a.lanes != best.lanes], default=math.inf)
    return best, alt


# --- Per-lane measurement ---


@dataclass
class _Lane:
    """Working state of one lane (crop coordinates), filled step by step."""

    present: bool = False
    x_range: tuple[float, float] | None = None  # assigned piece or cell
    cell: bool = False  # part of a touching run: x from the equal cell
    rect: Rect | None = None  # measured extent
    centre: float = math.nan  # expected centre: set for every lane by _fill_centres
    reason: LaneReason = "no_band"
    snr: float = 0.0
    span: tuple[int, int] | None = None  # the x-range between the walls: components counted
    components: int = 0


def _lanes_from(assign: _Assignment, pieces: list[_Piece], n: int) -> list[_Lane]:
    lanes = [_Lane() for _ in range(n)]
    for p, (k, q) in zip(pieces, assign.lanes, strict=True):
        for i in range(q):
            ln = lanes[k + i]
            ln.present = True
            ln.cell = q > 1
            ln.x_range = (p.l, p.r) if q == 1 else (p.l + i * p.w / q, p.l + (i + 1) * p.w / q)
            ln.centre = 0.5 * (ln.x_range[0] + ln.x_range[1])
    _fill_centres(lanes, assign.pitch)
    return lanes


def _fill_centres(lanes: list[_Lane], pitch: float) -> None:
    """Expected centres of the empty lanes: interpolated by index between present
    lanes, extrapolated with ``pitch`` past the ends."""
    idx = [i for i, ln in enumerate(lanes) if ln.present]
    if not idx:
        return
    cx = [lanes[i].centre for i in idx]
    for i, ln in enumerate(lanes):
        if ln.present:
            continue
        if i < idx[0]:
            ln.centre = cx[0] - (idx[0] - i) * pitch
        elif i > idx[-1]:
            ln.centre = cx[-1] + (i - idx[-1]) * pitch
        else:
            ln.centre = float(np.interp(i, idx, cx))


def _peaks(ks: np.ndarray, h: float) -> list[tuple[int, int]]:
    """``(y, x)`` of the separate peaks of ``ks`` (>= 0): the highest, if it
    reaches ``h``, and each other peak whose highest saddle to a higher peak is
    at least ``h`` below it and at most ``VALLEY_FRAC`` of it, the rule that
    separates two pieces along x (:func:`_split_run`). A peak is a 4-connected
    set of equal pixels, reported at its first pixel in raster order. Noise on
    a band's top, and a dent in it, fall far short; a second band does not.

    Candidates first, by the h-maxima transform: reconstruction by dilation of
    ``ks - h`` under ``ks`` leaves ``h`` on top of the peaks at least ``h`` above
    their saddle, and of peaks of one height joined above that. Then, from the
    highest down, a candidate is dropped if its region above the saddle level
    ``min(height - h, VALLEY_FRAC * height)`` holds a higher pixel or a peak
    already found. Only the bounding box of the nonzero pixels is searched;
    around it all is 0, which moves no saddle. ``h * 1e-9`` absorbs rounding."""
    rows, cols = np.flatnonzero(ks.any(axis=1)), np.flatnonzero(ks.any(axis=0))
    if rows.size == 0:
        return []
    y0, x0 = int(rows[0]), int(cols[0])
    box = ks[y0 : int(rows[-1]) + 1, x0 : int(cols[-1]) + 1]
    if float(box.max()) < h:
        return []
    eps = h * 1e-9
    rec = reconstruction(box - h, box, method="dilation", footprint=_CROSS)
    lab, count = label(box - rec >= h - eps)
    tops = [(int(y), int(x)) for y, x in maximum_position(box, lab, np.arange(1, count + 1))]
    tops.sort(key=lambda p: (-float(box[p]), p))
    found = tops[:1]
    for top in tops[1:]:
        height = float(box[top])
        region, _ = label(box > min(height - h, VALLEY_FRAC * height) + eps)
        joined = region == region[top]
        if float(box[joined].max()) <= height + eps and not any(joined[p] for p in found):
            found.append(top)
    return [(y + y0, x + x0) for y, x in found]


def _measure(lanes: list[_Lane], s: np.ndarray, kept: np.ndarray, sigma_sm: float) -> None:
    """Grow each present lane from its strongest kept pixel, confined between its
    walls: the midpoint to a present neighbour, the centre of an empty one, the
    box edge at the ends. A touching cell keeps its own x-range."""
    wc = s.shape[1]
    n = len(lanes)
    ks = np.where(kept, s, 0.0)
    for i, ln in enumerate(lanes):
        if not ln.present or ln.x_range is None:
            continue
        a0, b0 = ln.x_range
        if ln.cell:  # seed away from the sides of a touching cell (the central half)
            q = CELL_SEED * (b0 - a0)
            a0, b0 = a0 + q, b0 - q
        wa = max(0, int(math.floor(a0)))
        wb = min(wc, max(wa + 1, int(math.ceil(b0))))
        sub = ks[:, wa:wb]
        sy, sx = divmod(int(np.argmax(sub)), sub.shape[1])
        sx += wa
        v = float(ks[sy, sx])
        if v <= 0.0:
            ln.present = False
            continue
        walls = []
        for j in (i - 1, i + 1):
            if j < 0 or j >= n:
                walls.append(0.0 if j < 0 else float(wc))
            elif lanes[j].present:
                walls.append(0.5 * (ln.centre + lanes[j].centre))
            else:
                walls.append(lanes[j].centre)
        ga = min(max(0, int(math.floor(walls[0]))), sx)
        gb = max(min(wc, int(math.ceil(walls[1]))), sx + 1)
        window = ks[:, ga:gb]
        threshold = max(EXTENT_LEVEL * v, NOISE_K * sigma_sm)
        g = grow_region(window, (sx - ga, sy), threshold)
        if g is None:
            ln.present = False
            continue
        rect = (g[0] + ga, g[1], g[2] + ga, g[3])
        c0, c1 = ln.x_range
        if ln.cell:
            x0 = int(math.floor(c0))
            rect = (x0, rect[1], max(x0 + 1, int(math.floor(c1))), rect[3])
        ln.rect = rect
        ln.snr = v / sigma_sm
        ln.reason = "band"
        ln.span = (max(ga, int(math.floor(c0))), min(gb, int(math.ceil(c1))))


def _count_components(res: _Pass) -> None:
    """Each measured lane's components: the :func:`_peaks` of the kept signal
    at ``DETECT_K`` sigma in its span, at least the band grown. Run once, on the
    pass that gives the result."""
    tops = _peaks(np.where(res.cand.kept, res.sig.s_sm, 0.0), DETECT_K * res.sig.sigma_sm)
    for ln in res.lanes:
        if ln.present and ln.span is not None:
            lo, hi = ln.span
            ln.components = max(1, sum(1 for _, px in tops if lo <= px < hi))


# --- detect_row ---


def _is_int(v: object) -> bool:
    return isinstance(v, int | np.integer) and not isinstance(v, bool)


def _check_row(gray: np.ndarray, row: Sequence[int], n_lanes: int, background: float) -> Rect:
    """The row clipped to the image, or :class:`RowDetectError`."""
    if gray.ndim != 2:
        raise RowDetectError("invalid_row", "the analysis array must be 2-D")
    if not _is_int(n_lanes) or n_lanes < 1:
        raise RowDetectError("invalid_row", f"n_lanes must be a positive int, not {n_lanes!r}")
    if (
        isinstance(background, bool)
        or not isinstance(background, numbers.Real)
        or not math.isfinite(background)
    ):
        raise RowDetectError(
            "invalid_row", f"background must be a finite number, not {background!r}"
        )
    try:
        values = tuple(row)
    except TypeError:
        values = ()
    if len(values) != 4 or not all(_is_int(v) for v in values):
        raise RowDetectError("invalid_row", f"row must be four ints (x0, y0, x1, y1), not {row!r}")
    x0, y0, x1, y1 = (int(v) for v in values)
    if x1 <= x0 or y1 <= y0:
        raise RowDetectError("invalid_row", f"row {(x0, y0, x1, y1)} is empty or inverted")
    img_h, img_w = gray.shape
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        raise RowDetectError("row_outside_image", f"row {(x0, y0, x1, y1)} lies outside the image")
    if cx1 - cx0 < MIN_BOX * n_lanes or cy1 - cy0 < SMOOTH[0]:
        raise RowDetectError(
            "row_too_small",
            f"a {cx1 - cx0}x{cy1 - cy0} px row is too small for {int(n_lanes)} lanes",
        )
    return (cx0, cy0, cx1, cy1)


@dataclass(frozen=True)
class _Pass:
    sig: _Signal
    cand: _Candidates
    assign: _Assignment | None
    alt: float
    lanes: list[_Lane]
    notes: list[str]


def _run_pass(sig: _Signal, n: int, x_offset: int) -> _Pass:
    """Steps 3 to 6 of the pipeline on one signal."""
    wc = sig.s_sm.shape[1]
    notes: list[str] = []
    cand = _candidates(sig, n)
    pieces = _reduce(cand.pieces, n, x_offset, notes)
    assign, alt = (None, math.inf) if not pieces else _assign(pieces, n, float(wc))
    if assign is None:
        return _Pass(sig, cand, None, math.inf, [_Lane() for _ in range(n)], notes)
    lanes = _lanes_from(assign, pieces, n)
    _measure(lanes, sig.s_sm, cand.kept, sig.sigma_sm)
    return _Pass(sig, cand, assign, alt, lanes, notes)


def _stage2_free(res: _Pass, shape: tuple[int, int]) -> np.ndarray:
    """The band-free pixels of the row: its rows, minus each band's extent
    dilated by ``BG_GUARD`` of its own size (at least ``BG_GUARD_MIN`` px),
    minus the rejected regions."""
    free = np.zeros(shape, bool)
    lo, hi = res.cand.rows
    free[lo:hi] = True
    fy, fx = BG_GUARD
    for ln in res.lanes:
        if ln.present and ln.rect is not None:
            bx0, by0, bx1, by1 = ln.rect
            gy = max(BG_GUARD_MIN, int(math.ceil(fy * (by1 - by0))))
            gx = max(BG_GUARD_MIN, int(math.ceil(fx * (bx1 - bx0))))
            free[max(0, by0 - gy) : by1 + gy, max(0, bx0 - gx) : bx1 + gx] = False
    for _, region in res.cand.rejected:
        free[region] = False
    return free


def _empty_lanes(res: _Pass, n: int, wc: int, pitch: float) -> None:
    """Expected centre, window SNR and reason of every empty lane."""
    sig, cand, lanes = res.sig, res.cand, res.lanes
    if res.assign is None or not any(ln.present for ln in lanes):
        for i, ln in enumerate(lanes):
            ln.centre = (i + 0.5) * wc / n
    else:
        _fill_centres(lanes, pitch)
    lo, hi = cand.rows
    s_row = np.zeros_like(sig.s_sm)
    s_row[lo:hi] = sig.s_sm[lo:hi]
    for ln in lanes:
        if ln.present:
            continue
        a = int(max(0, math.floor(ln.centre - EMPTY_WINDOW * pitch)))
        b = int(min(wc, math.ceil(ln.centre + EMPTY_WINDOW * pitch)))
        if b <= a:
            continue
        ln.snr = float(s_row[:, a:b].max()) / sig.sigma_sm
        full = float(sig.s_sm[:, a:b].max()) / sig.sigma_sm
        if any(
            reason == "artefact" and region[1].start < b and a < region[1].stop
            for reason, region in cand.rejected
        ):
            ln.reason = "artefact"
        elif ln.snr >= DETECT_K:
            ln.reason = "unassigned"
        elif full >= DETECT_K:
            ln.reason = "edge_signal"
        else:
            ln.reason = "no_band"  # the snr tells how close it came


def _lanes_phrase(lanes: Sequence[int]) -> str:
    """``lane 3`` or ``lanes 1, 4``: lane indices counted from 1, as the user does."""
    word = "lane" if len(lanes) == 1 else "lanes"
    return f"{word} {', '.join(str(i + 1) for i in lanes)}"


def _shared_size(
    ws: Sequence[int], hs: Sequence[int], rule: str
) -> tuple[int, int, tuple[int, ...]]:
    """``(w, h, outliers)``: the shared size of the extents ``ws`` x ``hs`` by
    ``rule``, and the positions of the extents the rule left out. ``max``: the
    largest. ``max_guarded``: the largest of those within ``SIZE_GUARD`` times
    the median, per dimension (never empty: the smallest is below the median)."""
    w, h = np.asarray(ws, float), np.asarray(hs, float)
    if rule == "max":
        return int(w.max()), int(h.max()), ()
    within_w = w <= SIZE_GUARD * np.median(w)
    within_h = h <= SIZE_GUARD * np.median(h)
    outliers = tuple(int(k) for k in np.flatnonzero(~(within_w & within_h)))
    return int(w[within_w].max()), int(h[within_h].max()), outliers


def detect_row(
    gray: np.ndarray,
    row: Sequence[int],
    n_lanes: int,
    *,
    background: float,
    dark_on_light: bool = True,
    size_rule: str = SIZE_RULE,
) -> RowDetection:
    """One slot per declared lane in ``row`` ``(x0, y0, x1, y1)``: a box of one
    shared size, or None for an empty lane (see the module docstring).

    ``gray`` is the 2-D analysis array (``session.pixels``); it is read, never
    written. ``background`` is the image's stored background: it only decides
    whether a fitted plane or the stored level fits the membrane better, and
    sets ``bg_offset``. ``size_rule`` is one of :data:`SIZE_RULES`, for tests and
    evaluation; callers that commit boxes use the default :data:`SIZE_RULE`.
    Raises :class:`RowDetectError` for a row it cannot use.
    """
    if size_rule not in SIZE_RULES:
        raise ValueError(f"unknown size rule {size_rule!r}; expected one of {SIZE_RULES}")
    gray = np.asarray(gray)
    x0, y0, x1, y1 = _check_row(gray, row, n_lanes, background)
    n = int(n_lanes)
    crop = np.asarray(gray[y0:y1, x0:x1], dtype=np.float64)
    if not np.isfinite(crop).all():
        raise RowDetectError("invalid_row", "the row holds non-finite pixel values")
    hc, wc = crop.shape
    sign = 1.0 if dark_on_light else -1.0
    floor = _noise_floor(crop)
    sigma_px = max(_pixel_noise(crop), floor)

    # Stage 1: a plane over the crop, or the stored background if it fits the
    # membrane better.
    flat = np.full(crop.shape, float(background))
    plane = flat
    coef = _fit_plane(crop, np.ones(crop.shape, bool), floor)
    if coef is not None:
        plane = _plane(coef, crop.shape)
        sub = _subsample(np.arange(crop.size))
        cv = crop.ravel()[sub]
        if _membrane_spread(cv, float(background), sign) < _membrane_spread(
            cv, plane.ravel()[sub], sign
        ):
            plane = flat
    crop_ds = _despeckle(crop, _despeckle_width(wc, n))
    res = _run_pass(_signal(crop, crop_ds, plane, sign, sigma_px, floor, None), n, x0)

    # Stage 2: the plane and the noise from the band-free pixels of the row.
    free = _stage2_free(res, crop.shape)
    stage2 = bool(free.sum() >= max(BG_MIN_PIXELS, BG_MIN_KEEP * crop.size))
    if stage2:
        coef2 = _fit_plane(crop, free, floor)
        plane2 = plane if coef2 is None else _plane(coef2, crop.shape)
        res = _run_pass(_signal(crop, crop_ds, plane2, sign, sigma_px, floor, free), n, x0)
    notes = list(res.notes)
    if not stage2:
        notes.append("stage 2 skipped: too few band-free pixels")
    _count_components(res)
    sig, assign, lanes = res.sig, res.assign, res.lanes
    pitch = assign.pitch if assign is not None else wc / n
    _empty_lanes(res, n, wc, pitch)

    # Geometry the row box cannot resolve.
    flags: list[str] = []
    present = [i for i, ln in enumerate(lanes) if ln.present and ln.rect is not None]
    margin = None
    if assign is not None and present:
        if (present[0] > 0 and lanes[0].centre < 0.0) or (
            present[-1] < n - 1 and lanes[n - 1].centre > wc
        ):
            flags.append("lanes_outside_row")
        margin = float(res.alt - assign.cost)
        if margin < AMBIGUITY_MARGIN:
            flags.append("ambiguous_lanes")

    # One shared size; bounded isotonic placement inside the row.
    size = None
    out: dict[int, Rect] = {}
    outliers: tuple[int, ...] = ()
    if present:
        extents: list[Rect] = [r for i in present if (r := lanes[i].rect) is not None]
        w, h, outliers = _shared_size(
            [r[2] - r[0] for r in extents], [r[3] - r[1] for r in extents], size_rule
        )
        if len(extents) > 1:  # no wider than the closest pair of extent centres
            w = min(w, int(math.floor(min(np.diff([0.5 * (r[0] + r[2]) for r in extents])))))
        w = max(MIN_BOX, min(w, wc // len(extents)))  # len * w fits: MIN_BOX * n <= wc
        h = max(MIN_BOX, min(h, hc))
        xs = place_in_row([(r[0] + r[2]) // 2 - w // 2 for r in extents], w, 0, wc)
        for i, r, x in zip(present, extents, xs, strict=True):
            y = max(0, min((r[1] + r[3]) // 2 - h // 2, hc - h))
            out[i] = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
        size = BoxSize(width=w, height=h)

    result: list[LaneDetection] = []
    for i, ln in enumerate(lanes):
        rect = out.get(i)
        extent = bg_offset = None
        if ln.rect is not None and rect is not None:
            extent = (x0 + ln.rect[0], y0 + ln.rect[1], x0 + ln.rect[2], y0 + ln.rect[3])
            local = float(
                sig.plane[rect[1] - y0 : rect[3] - y0, rect[0] - x0 : rect[2] - x0].mean()
            )
            bg_offset = sign * (local - float(background)) / sig.sigma_px
        result.append(
            LaneDetection(
                lane=i,
                rect=rect,
                reason="band" if rect is not None else ln.reason,
                snr=float(ln.snr),
                extent=extent,
                expected_x=x0 + float(ln.centre),
                bg_offset=bg_offset,
                components=ln.components if rect is not None else 0,
            )
        )
    if any(ld.bg_offset is not None and abs(ld.bg_offset) > BG_WARN_K for ld in result):
        flags.append("background_mismatch")
    if outliers:
        flags.append("size_outlier")
        notes.append(
            f"{_lanes_phrase([present[k] for k in outliers])}: extent above "
            f"{SIZE_GUARD:g}x the median, left out of the shared size"
        )
    multiple = [ld.lane for ld in result if ld.components > 1]
    if multiple:
        flags.append("multiple_components")
        notes.append(
            f"{_lanes_phrase(multiple)}: a second separate component reaches the "
            "detection level; the box covers the one with the lane's strongest pixel"
        )
    return RowDetection(
        lanes=tuple(result),
        size=size,
        pitch=None if assign is None else float(assign.pitch),
        noise=float(sig.sigma_sm),
        pixel_noise=float(sig.sigma_px),
        flags=tuple(dict.fromkeys(flags)),
        notes=tuple(notes),
        cost=None if assign is None else float(assign.cost),
        margin=margin,
    )


def settings() -> dict[str, JsonValue]:
    """Every detection setting, JSON-plain, keyed by its constant's name in lower
    case (tuples as lists): what an action log or an export record reports."""
    return {
        "detect_k": DETECT_K,
        "noise_k": NOISE_K,
        "rel_threshold": REL_THRESHOLD,
        "extent_level": EXTENT_LEVEL,
        "size_rule": SIZE_RULE,
        "size_guard": SIZE_GUARD,
        "smooth": list(SMOOTH),
        "min_width_px": MIN_WIDTH_PX,
        "min_width_pitch": MIN_WIDTH_PITCH,
        "despeckle_min": DESPECKLE_MIN,
        "edge_hump": EDGE_HUMP,
        "row_walk_tol": ROW_WALK_TOL,
        "row_min_rows": ROW_MIN_ROWS,
        "row_min_keep": ROW_MIN_KEEP,
        "flat_edge": FLAT_EDGE,
        "valley_frac": VALLEY_FRAC,
        "gap_frac": GAP_FRAC,
        "near_peaks": NEAR_PEAKS,
        "envelope_half": ENVELOPE_HALF,
        "envelope_min": ENVELOPE_MIN,
        "reduce_mass": REDUCE_MASS,
        "run_lane_min": RUN_LANE_MIN,
        "q_window": Q_WINDOW,
        "cell_seed": CELL_SEED,
        "spacing_tol": SPACING_TOL,
        "gap_search": GAP_SEARCH,
        "end_lo": END_LO,
        "end_hi": END_HI,
        "end_tol": END_TOL,
        "end_sym_tol": END_SYM_TOL,
        "single_max_width": SINGLE_MAX_WIDTH,
        "split_penalty": SPLIT_PENALTY,
        "pitch_prior_tol": PITCH_PRIOR_TOL,
        "pitch_range": list(PITCH_RANGE),
        "pitch_steps": list(PITCH_STEPS),
        "pitch_refine": PITCH_REFINE,
        "ambiguity_margin": AMBIGUITY_MARGIN,
        "tail_q": TAIL_Q,
        "bg_clip_k": BG_CLIP_K,
        "bg_iter": BG_ITER,
        "bg_min_keep": BG_MIN_KEEP,
        "bg_min_pixels": BG_MIN_PIXELS,
        "bg_guard": list(BG_GUARD),
        "bg_guard_min": BG_GUARD_MIN,
        "min_fit": MIN_FIT,
        "fit_max_pixels": FIT_MAX_PIXELS,
        "noise_floor_frac": NOISE_FLOOR_FRAC,
        "empty_window": EMPTY_WINDOW,
        "bg_warn_k": BG_WARN_K,
        "min_box": MIN_BOX,
    }
