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
  empty slot never shifts another lane's index. Lanes are numbered from the
  box's left end, or from its right end for an image whose lanes run that way
  (``right_to_left``).
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
   stored background where the plane is off the membrane (its membrane side
   spreads over :data:`MEMBRANE_SPREAD_K` pixel sigmas: the bands pulled it,
   or a light strip) and the stored level's spreads less. The detection signal
   is the 3x5 box-smoothed crop on the band side of it; its noise is the
   smaller of two robust spreads.
3. Lines and strips across the lanes are taken out of the signal: a thin
   structure that holds its level along x over ``LINE_SPAN`` pitches beside
   the bands (a frame line), and a candidate that does so along the image's
   top or bottom edge across the whole box, cut by it (a dark strip or
   border, a vignette), up to its valley. Rows of the row: a hump of the
   row-mean signal cut by the top or bottom box edge (a neighbouring row) is
   left out up to its valley, if an interior hump remains.
4. Components of the signal above ``NOISE_K`` sigma that reach ``DETECT_K``
   sigma (hysteresis), with a 30%-core at least a dust floor wide, a peak off
   the box's edge rows, and not a flat structure spanning the rows (a streak or
   stain). Their column profile is cut into pieces between peaks; a piece that
   rises into the box's left or right edge is marked (a band the box cuts
   through there, or a dark image edge).
5. An ordered dynamic programme over the pieces and a pitch search assigns
   each piece to one lane, or a touching run to several, with empty lanes as
   gaps; the second-best reading measures how certain that is.
6. Per lane: growth with the click's rule (:func:`~proteia.core.grow.grow_region`
   at ``EXTENT_LEVEL`` of the lane's strongest pixel) between the lane's walls.
   A lane read from a marked piece is not grown: it stays empty.
7. Stage 2: the plane (or the stored background, chosen as in stage 1) and the
   noise again from the band-free pixels of the row, then steps 3 to 6 again.
8. Each band's lane: its separate components counted (peaks split as pieces
   are along x). Empty lanes get a reason; flags; one shared size by
   :data:`SIZE_RULE`, capped by the lane spacing and the box; bounded isotonic
   placement (:func:`~proteia.core.boxes.place_in_row`).
9. The row's line through the boxes' centres (:func:`_row_line`): a box off
   it is off the row.

Flags (:attr:`RowDetection.flags`):

* ``lanes_outside_row`` (refusing): an empty end lane's expected centre lies
  outside the row box, so the box does not cover every declared lane;
* ``ambiguous_lanes`` (refusing): a different lane reading costs less than
  :data:`AMBIGUITY_MARGIN` more than the chosen one, or two neighbouring
  lanes' extents are closer than the narrowest band (or out of order);
* ``off_row_line`` (refusing): a box's centre lies more than
  :data:`ROW_LINE_K` box heights above or below the row's line through the
  other boxes (see ``line_offset``), or boxes in neighbouring lanes lie more
  than :data:`ROW_SMILE` box heights apart: the row box covers more than one
  row (a neighbouring row's band is stronger in some lanes), or a lane's band
  lies above or below the others (a montage's panel, a mark beside the row);
* ``background_mismatch``: the membrane under a box differs from the stored
  background by more than :data:`BG_WARN_K` pixel sigmas;
* ``size_outlier``: an extent above :data:`SIZE_GUARD` times the median of the
  other extents was left out of the shared size (``"max_guarded"``);
* ``multiple_components``: a lane holds a second, separate component (see
  ``components``); its box is grown from the lane's strongest pixel, as a
  click there would be (quantifying doublets is #58's);
* ``cut_by_row_box``: the row box cuts through a band (see ``cut``): the
  band's extent reaches the box's top or bottom edge and that edge row, across
  the extent, still holds :data:`CUT_LEVEL` of the band's peak, so its box and
  net miss what lies beyond the edge; or, in an empty lane, the band peaks on
  that edge row itself and was left out (``edge_signal``).

Every setting is a module constant, reported by :func:`settings`.
"""

from __future__ import annotations

import itertools
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

import numpy as np
from pydantic import JsonValue
from scipy.ndimage import (
    find_objects,
    grey_opening,
    label,
    maximum,
    maximum_filter,
    maximum_filter1d,
    maximum_position,
    median_filter,
    minimum_filter1d,
    sum_labels,
    uniform_filter,
)
from scipy.signal import find_peaks
from scipy.special import ndtri
from skimage.morphology import reconstruction
from skimage.segmentation import watershed

from proteia.core.boxes import place_in_row
from proteia.core.grow import NOISE_K, REL_THRESHOLD, grow_region, mad_sigma
from proteia.core.model import BoxSize, Rect, lanes_phrase

# --- Domain settings (maintainer decisions on #51) ---

DETECT_K: Final = 6.0  # a band's peak reaches this many smoothed-noise sigmas
EXTENT_LEVEL: Final = REL_THRESHOLD  # sized extent: this fraction of the band's own peak (click)
SIZE_RULE: Final = "max_guarded"  # the shared size: the largest extent, outliers left out
SIZE_GUARD: Final = 2.0  # an extent above this times the others' median does not set the size
CUT_LEVEL: Final = EXTENT_LEVEL  # cut_by_row_box: the box's edge row holds this of a band's peak
# The row's line (#114): a box whose centre lies more than ROW_LINE_K box heights
# off it is off the row; it bends by at most ROW_SMILE box heights across the
# boxes (a smile), and is fitted from ROW_LINE_MIN boxes, as most of them lie
# within ROW_LINE_TOL box heights or ROW_LINE_TOL_PX px of it, whichever is more.
ROW_LINE_K: Final = 0.75
ROW_SMILE: Final = 2.0
ROW_LINE_MIN: Final = 4
ROW_LINE_TOL: Final = 0.25
ROW_LINE_TOL_PX: Final = 3.0

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
LINE_SPAN: Final = 2.0  # a line or strip holds its level along x over this many box pitches...
LINE_FLAT: Final = 0.8  # ...each of its pixels within this fraction of it (no dip at a lane gap)
LINE_PX: Final = 4  # a line is at most this many px high at half its height (drawn, smoothed)
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
MEMBRANE_SPREAD_K: Final = 1.5  # a spread over this x the white noise holds more than membrane
MIN_FIT: Final = 16  # fewest values a robust fit or noise estimate is made from
FIT_MAX_PIXELS: Final = 20000  # fits and noise estimates use a fixed-stride subsample
NOISE_FLOOR_FRAC: Final = 1e-3  # sigma floor: this fraction of the crop's range
EMPTY_WINDOW: Final = 0.3  # an empty lane's SNR is read within +/- this pitch of its centre
BG_WARN_K: Final = 3.0  # background_mismatch beyond this many pixel sigmas
MEMBRANE_SHIFT_K: Final = DETECT_K  # membrane_shift from which a box holds too little membrane
MIN_BOX: Final = 2  # smallest box side, as boxes.grow_to_fit

# --- Vocabularies ---

SIZE_RULES: Final = ("max", "max_guarded")
REFUSING_FLAGS: Final = ("lanes_outside_row", "ambiguous_lanes", "off_row_line")
WARNING_FLAGS: Final = (
    "background_mismatch",
    "size_outlier",
    "multiple_components",
    "cut_by_row_box",
)

RowDetectErrorCode = Literal["invalid_row", "invalid_image", "row_outside_image", "row_too_small"]
LaneReason = Literal[
    "band", "no_band", "artefact", "line", "edge_signal", "side_signal", "unassigned"
]

_TAIL_Z: Final = float(ndtri(0.5 + TAIL_Q / 200.0))  # Gaussian |z| at the TAIL_Q percentile
_P_SIGMA: Final = 68.27  # the percentile of |deviation| at one Gaussian sigma
_CROSS: Final = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)  # 4-connectivity, as label()
_COLUMN: Final = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], bool)  # runs down one column


class RowDetectError(ValueError):
    """A row box :func:`detect_row` cannot use. ``code`` is stable for callers:

    * ``invalid_row``: ``n_lanes`` not a positive int, or the row not four
      ints with ``x0 < x1`` and ``y0 < y1``;
    * ``invalid_image``: the image is at fault: the array not 2-D, the row
      holding non-finite pixel values, or ``background`` (the image's) not a
      finite real number;
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
      but dust reaches ``DETECT_K``; ``snr`` says how close it came),
      ``artefact`` (a rejected streak or stain covers it), ``line`` (a line or
      strip running across the lanes reaches ``DETECT_K`` in its slot: a frame
      line, a dark strip along the image's edge; a band may lie under it),
      ``edge_signal`` (only signal at the box's edge reaches ``DETECT_K``: in
      the rows left out as a neighbouring row, or peaking on the box's top or
      bottom row, as a band the box cuts through does), ``side_signal`` (only
      signal rising into the box's left or right edge reaches it: a band the
      box cuts through there, the lane read from it but given no box, or a dark
      image edge), ``unassigned`` (a kept candidate in the row's rows reaches
      ``DETECT_K`` but is in no assigned piece: a piece dropped for want of
      lanes, or too weak beside the bands around it).
    * ``snr``: the strongest smoothed signal in the lane (the growth seed; for an
      empty lane, within ``EMPTY_WINDOW`` pitch of its centre, leaving out the
      candidates the detector rejected, dust among them) over the noise.
    * ``extent``: the grown extent before sizing, None for an empty lane.
    * ``expected_x``: the lane's expected centre, also for an empty lane: between
      the present lanes by index, past them by the pitch, or an even split of
      the row when no lane holds a band.
    * ``bg_offset``: how far the stored background lies to the band side of the
      membrane under the box (the detection surface there, moved by the
      membrane's level in the signal as measured on the row's band-free pixels,
      also when stage 2 is skipped; the stage-1 signal's own with fewer than
      ``MIN_FIT`` of them), in pixel sigmas; None for an empty lane.
    * ``components``: the separate peaks of the lane's detection signal, in its
      x-range and between its walls; a peak is separate from a higher one if
      the saddle between them is ``DETECT_K`` sigma below it and at most
      ``VALLEY_FRAC`` of it, as two pieces along x. 1 for a lone band (noise on
      its top or a dent in it never is), 2 or more for several (a doublet,
      however weak its second band; a band split by a bubble), 0 when empty.
    * ``window``: the slot an empty lane's ``snr`` was read in: within
      ``EMPTY_WINDOW`` pitch of its expected centre along x, clipped to the row
      box, over the row's rows (a neighbouring row left out). None for a lane
      that holds a band, and for an empty lane whose slot lies outside the row
      box (not measured: its ``snr`` is 0).
    * ``cut``: the row box cuts through the lane's band (``cut_by_row_box``):
      its extent reaches the box's top or bottom edge, and that edge row, across
      the extent, holds at least ``CUT_LEVEL`` of the band's peak; or the lane
      is empty (``edge_signal``) because its band peaks on that edge row, where
      a candidate at least the dust floor wide and not flat across the rows (a
      band's hump, not a streak) reaches ``DETECT_K`` in the lane's slot.
    * ``line_offset``: how far the box's centre lies below (positive) or above
      (negative) the row's line (:func:`_row_line`), in box heights; beyond
      ``ROW_LINE_K`` either way the box is off the row (``off_row_line``).
      None for an empty lane, and for every lane of a row with fewer than
      ``ROW_LINE_MIN`` boxes (not checked) unless they lie on two rows.
    """

    lane: int
    rect: Rect | None
    reason: LaneReason
    snr: float
    extent: Rect | None
    expected_x: float
    bg_offset: float | None
    components: int
    window: Rect | None
    cut: bool
    line_offset: float | None


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
    ``membrane_shift`` says how far detection's membrane level lies inside
    the bands, by the box's own top and bottom rows: the smaller of how far
    each of those rows lies to the membrane side of that level (the detection
    surface's mean over the box, with the membrane's level in the signal) and
    of the box's deepest row (row means of the box-smoothed crop), in pixel
    sigmas from neighbour differences (which bands do not inflate). From
    ``MEMBRANE_SHIFT_K`` on, the box's edge rows are membrane around a band
    that detection took for its membrane: the box holds too little membrane to
    measure the bands against (a snug box around thick bands). The stored
    background plays no part, so a membrane darker than it is not taken for a
    band.
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
    membrane_shift: float

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
    return mad_sigma(d) / math.sqrt(2.0)


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


def _side_spread(r: np.ndarray) -> float:
    """Robust sigma of the membrane side of ``r``, a band-side signal about a
    surface: from the median distance of its negative values to the surface;
    inf if fewer than ``MIN_FIT``."""
    mem = -r[r < 0]
    return mad_sigma(mem, 0.0) if mem.size >= MIN_FIT else math.inf


def _surface(
    crop: np.ndarray,
    idx: np.ndarray,
    coef: np.ndarray,
    flat: np.ndarray,
    sign: float,
    sigma_px: float,
) -> np.ndarray:
    """The plane of ``coef``, or the stored background ``flat`` where the plane
    is off the membrane, judged on the pixels ``idx``.

    About the membrane's own level, the membrane side of the pixels spreads as
    the pixel noise ``sigma_px`` (half a Gaussian: 1.0). About a surface the
    bands pulled towards them it spreads more (1.5 at 0.7 pixel sigmas into
    them), as it does over a light strip; about one beyond the membrane, less
    (only the membrane's tail lies beyond it). So the plane stands unless its
    membrane side spreads over ``MEMBRANE_SPREAD_K`` pixel sigmas, and then
    the stored level is taken if it spreads less. The smaller spread alone
    would prefer a stored level beyond the membrane to a plane on it (a ramp's
    light end, say)."""
    plane = _plane(coef, crop.shape)
    v = crop.ravel()[idx]
    spread = _side_spread(sign * (plane.ravel()[idx] - v))
    if (
        spread > MEMBRANE_SPREAD_K * sigma_px
        and _side_spread(sign * (flat.ravel()[idx] - v)) < spread
    ):
        return flat
    return plane


@dataclass(frozen=True)
class _Signal:
    plane: np.ndarray  # the detection surface (crop coordinates)
    offset: float  # the membrane's level in the band-side signal, subtracted below
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
        # smaller of the membrane-side spread about the surface (inflated by a
        # light strip) and the central spread about the median (inflated a
        # little by many bands). With next to nothing on the membrane side, the
        # surface lies beyond the membrane (a stored level lighter than it) or
        # the box holds next to no membrane (it is filled by bands): the median
        # is the membrane's level if the values spread about it as the white
        # noise does, within MEMBRANE_SPREAD_K; otherwise it is a band's, and
        # the noise is the white noise about the surface.
        est = _side_spread(r)
        if math.isfinite(est):
            sigma_sm, tol = max(est, sfloor), 1.0
        else:
            sigma_sm, tol = sfloor, MEMBRANE_SPREAD_K
        offset = 0.0
        idx = _subsample(np.arange(r.size))
        off2, s2 = _center_scale(r.ravel()[idx], sfloor, tail=False)
        if s2 < tol * sigma_sm:
            sigma_sm, offset = s2, off2
    else:
        idx = _subsample(np.flatnonzero(free.ravel()))
        offset, sigma_sm = _center_scale(r.ravel()[idx], sfloor)
        # Pixel noise as measured on the membrane (JPEG: differences are often 0).
        sigma_px = max(sigma_px, _center_scale((plane - crop).ravel()[idx], floor)[1])
    r_ds = sign * (plane - uniform_filter(crop_ds, size=(ky, kx), mode="nearest"))
    return _Signal(
        plane,
        float(offset),
        np.maximum(r - offset, 0.0),
        np.maximum(r_ds - offset, 0.0),
        sigma_sm,
        sigma_px,
    )


# --- In-row signal: rows, components, pieces ---


def _hump(v: np.ndarray, order: Sequence[int]) -> tuple[int, int]:
    """The hump of the profile ``v`` at the first index of ``order``, walked
    in that order: up to its peak (over dips below ``ROW_WALK_TOL`` of the
    profile's maximum), then down to its valley (over rises below that).
    ``(peak, valley)``, as indices of ``v``."""
    tol = ROW_WALK_TOL * float(v.max())
    k = 0
    while k + 1 < len(order) and v[order[k + 1]] >= v[order[k]] - tol:
        k += 1
    peak = order[k]
    while k + 1 < len(order) and v[order[k + 1]] <= v[order[k]] + tol:
        k += 1
    return peak, order[k]


def _row_rows(s: np.ndarray, sigma_sm: float) -> tuple[int, int]:
    """Rows ``[lo, hi)`` that hold the row. A hump of the row-mean signal cut by
    the top or bottom box edge (the edge value is at least ``EDGE_HUMP`` of its
    peak: a neighbouring row) is left out up to its valley, if an interior hump
    remains."""
    hc = s.shape[0]
    v = s.mean(axis=1)
    if hc < ROW_MIN_ROWS or v.max() <= 0:
        return 0, hc
    sig = NOISE_K * sigma_sm / math.sqrt(max(1, s.shape[1] / SMOOTH[1]))

    def walk(order: list[int]) -> tuple[bool, int]:
        peak, valley = _hump(v, order)
        edge = v[peak] > sig and v[order[0]] >= EDGE_HUMP * v[peak]
        return edge, valley

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
    """A stretch of the column profile: crop x ``[l, r)``, its peak and mass;
    ``side``: it rises into the box's left or right edge (:func:`_rises_to_side`)."""

    l: float  # noqa: E741
    r: float
    peak: float
    mass: float
    side: bool = False

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
    dropped: np.ndarray  # mask of the candidates rejected by any rule (not kept)
    dust: np.ndarray  # mask of the candidates the dust floor rejected
    cut: np.ndarray  # mask of the band-like candidates peaking on the box's edge row
    lines: np.ndarray  # mask of the lines and strips running across the lanes (_lines)
    side: np.ndarray  # mask of the kept pieces rising into the box's left or right edge
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


def _min_width(wc: int, n: int) -> float:
    """The narrowest band, px: the smoothing kernel + 2 px, and a fraction of the
    box pitch (dust is far narrower than a lane; a narrow band is not)."""
    return min(max(MIN_WIDTH_PX, MIN_WIDTH_PITCH * wc / n), wc)


def _dust(comp: np.ndarray, ds: np.ndarray, peak: float, thr: float, min_w: float) -> bool:
    """Whether a candidate component is dust, judged on the despeckled signal
    ``ds`` of its region (``comp`` its mask there, ``peak`` its smoothed
    peak): gone under the running median, or its 30%-core there narrower than
    ``min_w``. A band keeps its plateau."""
    dloc = np.where(comp, ds, 0.0)
    dpk = float(dloc.max())
    if dpk < max(REL_THRESHOLD * peak, thr):
        return True  # gone under the running median: a speck
    dy, dx = divmod(int(np.argmax(dloc)), dloc.shape[1])
    dcore = grow_region(dloc, (dx, dy), max(REL_THRESHOLD * dpk, thr))
    return dcore is None or dcore[2] - dcore[0] < min_w  # below the width floor: noise, dust


def _lines(s: np.ndarray, sigma_sm: float, n: int, image_rows: tuple[bool, bool]) -> np.ndarray:
    """The pixels of the lines and strips that run across the lanes in the
    signal ``s``: not bands, whatever they reach.

    A pixel is flat along x when the level its row holds over ``LINE_SPAN`` box
    pitches around it (a horizontal opening of that length) is at least
    ``LINE_FLAT`` of its own: a band holds its level over one lane at most, and
    a row of bands dips between them, so only a structure that crosses the gaps
    between lanes does. So is a pixel whose level is held within
    ``LINE_PX // 2`` px up or down (the opening of the rows' running maximum
    over that height), at ``LINE_FLAT`` of its own either way: a line turned a
    little, as in a scan turned by a degree or two, drifts across the rows (a
    pixel beside a line, far lower, is not on it). A row of touching bands with
    no dip between them (and a saturated one, clipped flat) is flat too, but
    thick; and the tails of a row of thin bands are flat while their tops are
    not. So a flat pixel counts only in a run down its column that is the
    column's top there (the pixels just above and below it lower than its
    highest) and no more than ``LINE_PX`` px high at half that, a drawn line's
    height once smoothed; a thin structure is a 4-connected set of such
    pixels (pieces of them on the same rows less than a box pitch apart
    joined: a band's tail lifting a close line's edge row breaks it there over
    a band's width) over ``LINE_SPAN`` pitches along x that reaches
    ``DETECT_K``, with its fringe (up to ``LINE_PX`` px down each column from
    it while the signal falls). A row of thin touching bands is such a
    structure too, and alone in the box it is the row; a frame line runs
    beside the bands, above or below them. So a thin structure is a line
    when, in its own columns, above or below it and past the slope it falls
    down, something reaches ``DETECT_K`` over at least a band's width
    (``_min_width``) that the detector keeps (a band, or the frame's other
    line): in the row's rows (a neighbouring row's hump at the box's edge
    left out, :func:`_row_rows`), and not peaking on the box's top or bottom
    row. But a thin structure that holds its level (``LINE_FLAT`` of its
    median) a band's width past both ends of another's is that one's frame,
    no sign of it being a line: a frame's line runs on across the box past a
    row of thin bands' end shoulders, where the row's level ends, while a
    frame's two lines end alike. A band next to a line keeps its own pixels:
    a line is taken out of a component, not the component with it.

    ``image_rows`` says whether the box's top and bottom rows are the image's.
    A candidate that is flat along one of those rows over ``LINE_SPAN`` pitches
    runs along the image's edge, whatever its height, when it runs across the
    whole box (on that row, within a band's width of each of the box's sides
    it holds ``LINE_FLAT`` of its median level there) and the image cuts it
    (that row holds at least ``CUT_LEVEL`` of the hump its mean profile over
    its columns there has at the edge): a dark strip or border, or a
    vignette. It is taken out from that row to its hump's valley
    (:func:`_hump`), so bands beyond the valley keep theirs. A row of bands
    is none: it ends inside the box, which spans every lane with its margin,
    or holds far less at its sides (the tails of deep bands), and the image
    ending just past it leaves only its tails on that row.

    With ``LINE_SPAN`` pitches wider than the box (a box of fewer than two
    lanes) nothing is a line: a band may fill it."""
    hc, wc = s.shape
    span = int(math.ceil(LINE_SPAN * wc / n))
    lines = np.zeros(s.shape, bool)
    if span > wc:
        return lines
    thr = NOISE_K * sigma_sm
    strong = s >= DETECT_K * sigma_sm
    held = grey_opening(s, size=(1, span), mode="constant", cval=0.0)
    flat = (s > thr) & (held >= LINE_FLAT * s)
    # A line turned a little drifts across the rows: it holds its level
    # within half its height up or down, at about the pixel's own level.
    drift = maximum_filter1d(s, size=2 * (LINE_PX // 2) + 1, axis=0, mode="constant", cval=0.0)
    held = grey_opening(drift, size=(1, span), mode="constant", cval=0.0)
    flat |= (s > thr) & (held >= LINE_FLAT * s) & (LINE_FLAT * held <= s)
    runs, count = label(flat, structure=_COLUMN)
    if count:
        idx = np.arange(1, count + 1)
        tops = np.zeros(count + 1)
        tops[1:] = maximum(s, runs, idx)
        heights = np.zeros(count + 1)
        heights[1:] = sum_labels(s >= 0.5 * tops[runs], runs, idx)
        # The pixels just above and below each run (0 past the box): a run
        # beside a higher pixel is a flank of a taller hump, not its top.
        above = np.zeros_like(s)
        above[1:] = s[:-1]
        below = np.zeros_like(s)
        below[:-1] = s[1:]
        first = flat.copy()  # each run's top pixel
        first[1:] &= ~flat[:-1]
        last = flat.copy()  # ...and its bottom pixel
        last[:-1] &= ~flat[1:]
        beside = np.zeros(count + 1)
        beside[runs[first]] = above[first]
        beside[runs[last]] = np.maximum(beside[runs[last]], below[last])
        flat_tops = flat & (heights[runs] <= LINE_PX) & (beside[runs] < tops[runs])
        # Pieces of them on the same rows less than a pitch apart are one: a
        # band's tail lifting a line's edge row there breaks it over a band's
        # width (a closing along x, over an odd width of about a pitch, that
        # never runs a piece on past its end).
        gap = 2 * (int(math.ceil(wc / n)) // 2) + 1
        closed = maximum_filter1d(flat_tops.view(np.uint8), gap, axis=1, mode="constant")
        closed = minimum_filter1d(closed, gap, axis=1, mode="constant").view(bool)
        parts, _ = label(closed | flat_tops)
        thin = np.zeros(s.shape, bool)
        for k, (_, xs) in enumerate(find_objects(parts)):
            if xs.stop - xs.start >= span:
                part = (parts == k + 1) & flat_tops
                if strong[part].any():
                    thin |= part
        # Its slope: down each column from it while the signal falls (its
        # edges, less flat than its top), not up into a band beside it. Up to
        # LINE_PX px of it is the fringe taken out with a line, not far into a
        # band beside it; all of it is the structure's own when looking for
        # what lies beside it (the flank of a taller hump whose top it is, as
        # the image's edge leaves of bands it cuts, is no band beside it).
        slope = thin.copy()
        for step in range(hc):
            grown = np.zeros_like(slope)
            grown[:-1] |= slope[1:] & (s[:-1] <= s[1:])
            grown[1:] |= slope[:-1] & (s[1:] <= s[:-1])
            grown &= (s > thr) & ~slope
            if not grown.any():
                break
            slope |= grown
            if step < LINE_PX:
                thin |= grown
        # A line has something beside it, above or below in its own columns,
        # that the detector keeps: in the row's rows, not a candidate peaking
        # on the box's top or bottom row (a neighbouring row's edge,
        # _candidates), which a row of thin bands has beside it as a frame
        # line has the bands, or its other line.
        rest, others = label((s > thr) & ~slope)
        at_edge = np.zeros(others + 1, bool)
        if others:
            tops_at = maximum_position(s, rest, np.arange(1, others + 1))
            at_edge[1:] = [y in (0, hc - 1) for y, _ in tops_at]
        other = strong & ~slope & ~at_edge[rest]
        structures, count = label(thin)
        regions = find_objects(structures)
        at_edge = np.zeros(count + 1, bool)
        if count:
            tops_at = maximum_position(s, structures, np.arange(1, count + 1))
            at_edge[1:] = [y in (0, hc - 1) for y, _ in tops_at]
        other_thin = strong & thin & ~at_edge[structures]
        lo, hi = _row_rows(np.where(slope, 0.0, s), sigma_sm)
        for mask in (other, other_thin):
            mask[:lo] = False
            mask[hi:] = False
        # Where each holds its level along x: its first and last column at
        # LINE_FLAT of its median level (a row of bands' end shoulders left out).
        ends = np.zeros((count + 1, 2))
        for k, sl in enumerate(regions):
            top = np.where(structures[sl] == k + 1, s[sl], 0.0).max(axis=0)
            on = np.flatnonzero(top >= LINE_FLAT * float(np.median(top[top > 0])))
            ends[k + 1] = sl[1].start + on[0], sl[1].start + on[-1]
        w = _min_width(wc, n)
        down = np.arange(hc)[:, None]
        for k, sl in enumerate(regions):
            one = structures[sl] == k + 1
            cols = one.any(axis=0)
            upper = np.argmax(one, axis=0) + sl[0].start  # its first row per column
            lower = sl[0].stop - 1 - np.argmax(one[::-1], axis=0)  # ...and its last
            # A thin structure holding its level a band's width past both its
            # ends is its frame's line, beside the row: no sign of a line.
            frames = (ends[:, 0] <= ends[k + 1, 0] - w) & (ends[:, 1] >= ends[k + 1, 1] + w)
            beside = other[:, sl[1]] | (other_thin[:, sl[1]] & ~frames[structures[:, sl[1]]])
            off = beside & ((down < upper) | (down > lower)) & cols
            if np.count_nonzero(off.any(axis=0)) >= w:
                lines[:, sl[1]] |= structures[:, sl[1]] == k + 1
    if image_rows[0] or image_rows[1]:
        comps, _ = label(s > thr)
        big = set(np.unique(comps[strong]).tolist())
        w = int(math.ceil(_min_width(wc, n)))  # the box's sides: a band's width in from each
        for row, at in ((0, image_rows[0]), (hc - 1, image_rows[1])):
            if not at:
                continue
            along = np.concatenate([[False], flat[row], [False]])
            edges = np.flatnonzero(along[1:] != along[:-1])
            flat_along = {
                int(c)
                for a, b in zip(edges[0::2], edges[1::2], strict=True)
                if b - a >= span
                for c in np.unique(comps[row, a:b])
            }
            for c in sorted(flat_along & big):
                on = comps[row] == c
                # Across the whole box at its level: a row of bands ends inside
                # it, or, deep enough for its tails to reach the box's sides,
                # holds far less there.
                level = LINE_FLAT * float(np.median(s[row, on]))
                ends = [s[row, cols][on[cols]] for cols in (slice(0, w), slice(wc - w, wc))]
                if any(end.size == 0 or float(end.max()) < level for end in ends):
                    continue
                v = s[:, on].mean(axis=1)
                peak, valley = _hump(v, range(hc) if row == 0 else range(hc - 1, -1, -1))
                if v[row] < CUT_LEVEL * v[peak]:
                    continue  # the image ends past it: a row of bands' tails
                strip = slice(0, valley + 1) if row == 0 else slice(valley, hc)
                lines[strip] |= comps[strip] == c
    return lines


def _rises_to_side(p: np.ndarray, left: bool, depth: int) -> bool:
    """Whether the profile ``p`` of a piece that reaches the box's left
    (``left``) or right edge rises into it: highest on the edge column, and
    nowhere ``depth`` px or more inside it at ``VALLEY_FRAC`` of that, so no
    hump of a band lies inside the box (a band the box's side edge cuts
    through, a dark image edge). A band peaking inside, or a flat top that
    runs on inside (a saturated band), stays."""
    edge = float(p[0] if left else p[-1])
    if edge < float(p.max()):
        return False
    inner = p[depth:] if left else p[: max(0, p.size - depth)]
    return inner.size == 0 or float(inner.max()) < VALLEY_FRAC * edge


def _candidates(sig: _Signal, n: int, image_rows: tuple[bool, bool]) -> _Candidates:
    """The kept components and the pieces of their column profile.

    Lines and strips running across the lanes (:func:`_lines`) are taken out
    first. A candidate peaking on the box's top or bottom row is rejected
    (``edge_signal``) whatever it is; of those, dust counts as dust, and one
    that holds a band's hump (not flat across the rows, as a streak is) is a
    band the box cuts through (``cut``). A piece reaching the box's left or
    right edge that rises into it (:func:`_rises_to_side`) stays in the lane
    reading, as the band of the lane there the box cuts through may be, but
    is marked ``side``: its lane gets no box (``side_signal``)."""
    lines = _lines(sig.s_sm, sig.sigma_sm, n, image_rows)
    s_all = np.where(lines, 0.0, sig.s_sm)
    hc, wc = s_all.shape
    lo, hi = _row_rows(s_all, sig.sigma_sm)
    s = np.zeros_like(s_all)
    s[lo:hi] = s_all[lo:hi]
    ds = np.zeros_like(s_all)
    ds[lo:hi] = np.where(lines, 0.0, sig.s_ds)[lo:hi]
    rejected: list[tuple[str, Region]] = []
    if lo > 0:
        rejected.append(("edge_signal", (slice(0, lo), slice(0, wc))))
    if hi < hc:
        rejected.append(("edge_signal", (slice(hi, hc), slice(0, wc))))
    thr = NOISE_K * sig.sigma_sm
    min_w = _min_width(wc, n)
    depth = int(math.ceil(min_w))  # _rises_to_side: a band's width inside the edge
    lab, count = label(s > thr)
    kept = np.zeros(s.shape, bool)
    dust = np.zeros(s.shape, bool)
    cut = np.zeros(s.shape, bool)
    side = np.zeros(s.shape, bool)
    candidates = np.zeros(s.shape, bool)
    if count:
        regions = find_objects(lab)
        # Only components holding a pixel at the detection level (hysteresis).
        strong = np.unique(lab[s >= DETECT_K * sig.sigma_sm])
        strong = strong[strong > 0]
        candidates = np.isin(lab, strong)
        for k in strong - 1:
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
            c0, c1 = sl[1].start + cx0, sl[1].start + cx1
            if py == 0 or py == hc - 1:
                rejected.append(("edge_signal", sl))
                if _dust(comp, ds[sl], peak, thr, min_w):
                    dust[sl] |= comp
                elif not _flat(s, lo, hi, c0, c1):
                    cut[sl] |= comp  # a band the box's edge runs through
                continue
            if cy0 <= lo and cy1 >= hi and _flat(s, lo, hi, c0, c1):
                rejected.append(("artefact", sl))
                continue
            # Width on the despeckled signal: dust and specks vanish under the
            # running median; a band keeps its plateau.
            if _dust(comp, ds[sl], peak, thr, min_w):
                dust[sl] |= comp
                continue
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
            cols = slice(a + j0, a + j1 + 1)
            if _flat(s, lo, hi, a + j0, a + j1 + 1):  # a stain attached to a band
                rejected.append(("artefact", (slice(lo, hi), cols)))
                kept[:, cols] = False
                continue
            ends = [left for left, at in ((True, a + j0 == 0), (False, a + j1 + 1 == wc)) if at]
            at_side = any(_rises_to_side(seg[j0 : j1 + 1], left, depth) for left in ends)
            if at_side:  # its tails too: the part of the run it was cut from
                side[:, a:b] |= kept[:, a:b]
            pieces.append(
                _Piece(a + j0, a + j1 + 1, float(seg[i]), float(seg[j0 : j1 + 1].sum()), at_side)
            )
    return _Candidates(kept, candidates & ~kept, dust, cut, lines, side, (lo, hi), rejected, pieces)


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
            merged = _Piece(a.l, b.r, max(a.peak, b.peak), a.mass + b.mass, a.side and b.side)
            pieces[i : i + 2] = [merged]
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
    best reading with different lanes (at the same pitch or any other). Each
    fraction of the box pitch is tried once: the fine pass skips the coarse
    fraction it is centred on."""
    p0 = box_w / n
    lo, hi = PITCH_RANGE
    coarse, fine = PITCH_STEPS
    found: list[_Assignment] = []
    tried: set[float] = set()

    def run(fracs: np.ndarray) -> None:
        for f in fracs:
            key = round(float(f), 9)
            if key in tried:
                continue
            tried.add(key)
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
    # A different reading: the runner-up at any pitch that reads the best's lanes,
    # or the best at a pitch that reads other lanes.
    alt = min(
        [a.second for a in found if a.lanes == best.lanes]
        + [a.cost for a in found if a.lanes != best.lanes],
        default=math.inf,
    )
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
    peak: float = 0.0  # the growth seed's smoothed signal
    snr: float = 0.0
    span: tuple[int, int] | None = None  # the x-range between the walls: components counted
    components: int = 0
    window: Rect | None = None  # an empty lane's measured slot (crop coordinates)
    cut: bool = False  # an empty lane's band peaks on the box's edge row (_empty_lanes)
    side: bool = False  # read from a piece rising into the box's side: not measured


def _lanes_from(assign: _Assignment, pieces: list[_Piece], n: int) -> list[_Lane]:
    lanes = [_Lane() for _ in range(n)]
    for p, (k, q) in zip(pieces, assign.lanes, strict=True):
        for i in range(q):
            ln = lanes[k + i]
            ln.present = True
            ln.side = p.side
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
    around it all is 0, which moves no saddle. ``h * 1e-9`` absorbs rounding.

    The regions are read from the candidates' saddle graph (:func:`_saddles`)
    with one union-find, not labelled once per candidate: a region above a
    level holds a higher pixel exactly when it holds a higher candidate (the
    highest pixel of a region above a level at least ``h`` below it is an
    h-maximum), and joins two candidates exactly when the graph does."""
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
    heights = [float(box[p]) for p in tops]
    # Non-increasing, as the heights: each candidate's level joins at least
    # what the previous one's did.
    levels = [min(v - h, VALLEY_FRAC * v) + eps for v in heights]
    edges = _saddles(box, tops, levels[-1]) if len(tops) > 1 else []
    parent = list(range(len(tops)))  # a root is its set's highest candidate
    holds_found = [False] * len(tops)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    found: list[tuple[int, int]] = []
    e = 0
    for i, top in enumerate(tops):
        while e < len(edges) and edges[e][0] > levels[i]:
            a, b = sorted((root(edges[e][1]), root(edges[e][2])))
            if a != b:
                parent[b] = a
                holds_found[a] = holds_found[a] or holds_found[b]
            e += 1
        r = root(i)
        if i == 0 or (heights[r] <= heights[i] + eps and not holds_found[r]):
            found.append(top)
            holds_found[r] = True
    return [(y + y0, x + x0) for y, x in found]


def _saddles(
    box: np.ndarray, tops: list[tuple[int, int]], floor: float
) -> list[tuple[float, int, int]]:
    """The saddle graph of the pixels ``tops`` of ``box``: ``(level, i, j)``,
    highest first, such that tops ``i`` and ``j`` lie in one 4-connected region
    of ``box > t`` (for any ``t >= floor``) exactly when a path of edges above
    ``t`` joins them.

    A watershed from the tops floods the pixels above ``floor`` from the
    highest down, so each pixel's basin is a top it can reach through its
    highest path; ``reach`` is that path's lowest value (reconstruction by
    dilation from the tops). Two basins touch at the level of the highest
    ``min(reach)`` over their touching pixel pairs. Nothing at or below
    ``floor`` joins anything, so the pixels above it, in their bounding box,
    are enough."""
    above = box > floor
    rows, cols = np.flatnonzero(above.any(axis=1)), np.flatnonzero(above.any(axis=0))
    dy, dx = int(rows[0]), int(cols[0])
    sub = box[dy : int(rows[-1]) + 1, dx : int(cols[-1]) + 1]
    above = above[dy : int(rows[-1]) + 1, dx : int(cols[-1]) + 1]
    ty = np.array([y for y, _ in tops]) - dy
    tx = np.array([x for _, x in tops]) - dx
    k = len(tops)
    markers = np.zeros(sub.shape, np.int32)
    markers[ty, tx] = np.arange(1, k + 1)
    seed = np.full(sub.shape, float(sub.min()))
    seed[ty, tx] = sub[ty, tx]
    reach = reconstruction(seed, sub, method="dilation", footprint=_CROSS)
    basin = watershed(-sub, markers, connectivity=1, mask=above)
    keys, levels = [], []
    for a, b, ra, rb in (
        (basin[:, :-1], basin[:, 1:], reach[:, :-1], reach[:, 1:]),
        (basin[:-1, :], basin[1:, :], reach[:-1, :], reach[1:, :]),
    ):
        touch = (a != b) & (a > 0) & (b > 0)
        lo, hi = np.minimum(a[touch], b[touch]), np.maximum(a[touch], b[touch])
        keys.append(lo.astype(np.int64) * (k + 1) + hi)
        levels.append(np.minimum(ra[touch], rb[touch]))
    pairs, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    level = np.full(pairs.size, -np.inf)
    np.maximum.at(level, inverse, np.concatenate(levels))
    order = np.argsort(-level, kind="stable")
    return [
        (float(level[m]), int(pairs[m] // (k + 1)) - 1, int(pairs[m] % (k + 1)) - 1) for m in order
    ]


def _measure(lanes: list[_Lane], s: np.ndarray, kept: np.ndarray, sigma_sm: float) -> None:
    """Grow each present lane from its strongest kept pixel, confined between its
    walls: the midpoint to a present neighbour, the centre of an empty one, the
    box edge at the ends. A touching cell keeps its own x-range. A lane read
    from a piece rising into the box's side is not grown, and ends up empty,
    but walls its neighbour in as a present one does."""
    wc = s.shape[1]
    n = len(lanes)
    ks = np.where(kept, s, 0.0)
    for i, ln in enumerate(lanes):
        if not ln.present or ln.x_range is None or ln.side:
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
        ln.peak = v
        ln.snr = v / sigma_sm
        ln.reason = "band"
        ln.span = (max(ga, int(math.floor(c0))), min(gb, int(math.ceil(c1))))
    for ln in lanes:
        if ln.side:
            ln.present = False


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
        raise RowDetectError("invalid_image", "the analysis array must be 2-D")
    if not _is_int(n_lanes) or n_lanes < 1:
        raise RowDetectError("invalid_row", f"n_lanes must be a positive int, not {n_lanes!r}")
    if (
        isinstance(background, bool)
        or not isinstance(background, numbers.Real)
        or not math.isfinite(background)
    ):
        raise RowDetectError(
            "invalid_image", f"background must be a finite number, not {background!r}"
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


def _run_pass(sig: _Signal, n: int, x_offset: int, image_rows: tuple[bool, bool]) -> _Pass:
    """Steps 3 to 6 of the pipeline on one signal; ``image_rows``: whether the
    box's top and bottom rows are the image's (:func:`_lines`)."""
    wc = sig.s_sm.shape[1]
    notes: list[str] = []
    cand = _candidates(sig, n, image_rows)
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
    minus every kept pixel outside those (a piece dropped or left unassigned, a
    lane whose growth failed, a doublet's other band, a band's tail): each
    connected part's bounding box dilated by ``BG_GUARD_MIN`` px; minus the
    rejected regions, and the lines and strips (:func:`_lines`) dilated by
    ``BG_GUARD_MIN`` px (a frame's bounding box can be the whole box).

    An extent is a band at ``EXTENT_LEVEL`` of its peak, so its guard grows
    with it to take in the tails below that level. A kept component already
    reaches down to ``NOISE_K`` sigma, tails and all; a guard of its own size
    on top leaves too few band-free pixels for stage 2 around doublets and
    dust (on the adversarial rows of #51, stage 2 was skipped in 17 more of
    510 and 12 doublets were poorly boxed)."""
    blocked = np.zeros(shape, bool)

    def block(rect: Rect, gy: int, gx: int) -> None:
        bx0, by0, bx1, by1 = rect
        blocked[max(0, by0 - gy) : by1 + gy, max(0, bx0 - gx) : bx1 + gx] = True

    fy, fx = BG_GUARD
    for ln in res.lanes:
        if ln.rect is not None:
            w, h = ln.rect[2] - ln.rect[0], ln.rect[3] - ln.rect[1]
            gy = max(BG_GUARD_MIN, int(math.ceil(fy * h)))
            block(ln.rect, gy, max(BG_GUARD_MIN, int(math.ceil(fx * w))))
    rest, _ = label(res.cand.kept & ~blocked)
    for sy, sx in find_objects(rest):
        block((sx.start, sy.start, sx.stop, sy.stop), BG_GUARD_MIN, BG_GUARD_MIN)
    free = np.zeros(shape, bool)
    lo, hi = res.cand.rows
    free[lo:hi] = True
    free &= ~blocked
    for _, region in res.cand.rejected:
        free[region] = False
    g = 2 * BG_GUARD_MIN + 1
    free &= ~maximum_filter(res.cand.lines, size=(g, g), mode="constant", cval=False)
    return free


def _empty_lanes(res: _Pass, n: int, wc: int, pitch: float) -> None:
    """Expected centre, window SNR and reason of every empty lane.

    The SNR leaves out the candidates the detector rejected (dust, a peak on
    the box's edge rows, a streak or stain, a line, signal rising into the
    box's side), so only a kept candidate reaches ``DETECT_K`` there
    (``unassigned``); signal short of a candidate stays, so the SNR tells how
    close a faint band came. A line or strip across the lanes reaching
    ``DETECT_K`` in the slot makes a lane ``line``, and then signal rising into
    the box's left or right edge ``side_signal``: a band may lie under the
    one, and the other is a band the box cuts through or the image's edge.
    Signal at the box's top or bottom edge (in the rows left out, or a
    candidate peaking on an edge row) makes a lane ``edge_signal``; dust counts
    nowhere. An ``edge_signal`` lane is ``cut`` when a band the box's edge runs
    through (``_Candidates.cut``) reaches ``DETECT_K`` in its slot."""
    sig, cand, lanes = res.sig, res.cand, res.lanes
    if res.assign is None or not any(ln.present for ln in lanes):
        for i, ln in enumerate(lanes):
            ln.centre = (i + 0.5) * wc / n
    else:
        _fill_centres(lanes, pitch)
    lo, hi = cand.rows
    s_row = np.zeros_like(sig.s_sm)
    s_row[lo:hi] = sig.s_sm[lo:hi]
    s_row[cand.dropped | cand.lines | cand.side] = 0.0
    s_all = np.where(cand.dust | cand.lines | cand.side, 0.0, sig.s_sm)
    s_cut = np.where(cand.cut, sig.s_sm, 0.0)
    s_line = np.where(cand.lines, sig.s_sm, 0.0)
    s_side = np.where(cand.side, sig.s_sm, 0.0)
    for ln in lanes:
        if ln.present:
            continue
        a = int(max(0, math.floor(ln.centre - EMPTY_WINDOW * pitch)))
        b = int(min(wc, math.ceil(ln.centre + EMPTY_WINDOW * pitch)))
        if b <= a:
            continue  # the slot lies outside the row box: not measured
        ln.window = (a, lo, b, hi)
        ln.snr = float(s_row[:, a:b].max()) / sig.sigma_sm
        full = float(s_all[:, a:b].max()) / sig.sigma_sm
        if any(
            reason == "artefact" and region[1].start < b and a < region[1].stop
            for reason, region in cand.rejected
        ):
            ln.reason = "artefact"
        elif ln.snr >= DETECT_K:
            ln.reason = "unassigned"
        elif float(s_line[:, a:b].max()) / sig.sigma_sm >= DETECT_K:
            ln.reason = "line"
        elif float(s_side[:, a:b].max()) / sig.sigma_sm >= DETECT_K:
            ln.reason = "side_signal"
        elif full >= DETECT_K:
            ln.reason = "edge_signal"
            ln.cut = float(s_cut[:, a:b].max()) / sig.sigma_sm >= DETECT_K
        else:
            ln.reason = "no_band"  # the snr tells how close it came


def _shared_size(
    ws: Sequence[int],
    hs: Sequence[int],
    rule: str,
    *,
    cut_w: Sequence[bool] | None = None,
    cut_h: Sequence[bool] | None = None,
) -> tuple[int, int, tuple[int, ...]]:
    """``(w, h, outliers)``: the shared size of the extents ``ws`` x ``hs`` by
    ``rule``, and the positions of the extents the rule left out. ``max``: the
    largest. ``max_guarded``: per dimension, the largest of those at most
    ``SIZE_GUARD`` times the median of the OTHER extents, the upper of the two
    middle ones when the others are even in number. Of two extents that is the
    other one (twice the median of both is their sum, which neither exceeds);
    of an even count, the median of the others; of an odd count it is the whole
    row's median for every extent above it, as a guard on the median of all.
    Never empty: the smallest is at most the others' median; a lone extent is
    kept.

    ``cut_w`` / ``cut_h`` mark extents the row box cuts in that dimension: their
    true size is at least the measured one, so as another extent's reference
    they count as unbounded. A band the box cuts short then never shrinks the
    size of a complete one beside it (with two extents it would otherwise be
    the reference)."""
    w, h = np.asarray(ws, float), np.asarray(hs, float)
    if rule == "max":
        return int(w.max()), int(h.max()), ()

    def within(v: np.ndarray, cut: Sequence[bool] | None) -> np.ndarray:
        if v.size == 1:
            return np.ones(1, bool)
        ref = v if cut is None else np.where(np.asarray(cut, bool), np.inf, v)
        mid = (v.size - 1) // 2  # of the v.size - 1 others: the median, or the upper middle
        others = np.array([np.sort(np.delete(ref, k))[mid] for k in range(v.size)])
        return v <= SIZE_GUARD * others

    within_w, within_h = within(w, cut_w), within(h, cut_h)
    outliers = tuple(int(k) for k in np.flatnonzero(~(within_w & within_h)))
    return int(w[within_w].max()), int(h[within_h].max()), outliers


def _bounded_fits(u: np.ndarray, y: np.ndarray, w: np.ndarray, cmax: float) -> np.ndarray:
    """``(K, 3)`` coefficients ``(a, b, c)`` of ``a + b*u + c*u**2`` fitted by
    least squares to the boxes each row of the 0/1 weights ``w`` (``(K, n)``,
    at least three distinct ``u`` each) holds, ``c`` clipped to ``+/-cmax``
    and ``a``, ``b`` fitted again under it."""
    x = np.stack([np.ones_like(u), u, u * u], axis=1)
    gram = np.einsum("kn,ni,nj->kij", w, x, x)
    rhs = np.einsum("kn,ni,n->ki", w, x, y)
    coef = np.linalg.solve(gram, rhs[..., None])[..., 0]
    over = np.abs(coef[:, 2]) > cmax
    if over.any():
        c = np.clip(coef[over, 2], -cmax, cmax)
        rhs2 = rhs[over, :2] - c[:, None] * np.einsum("kn,ni,n->ki", w[over], x[:, :2], u * u)
        ab = np.linalg.solve(gram[over][:, :2, :2], rhs2[..., None])[..., 0]
        coef[over] = np.column_stack([ab, c])
    return coef


def _row_line(
    centres: Sequence[tuple[float, float]], height: int, lanes: Sequence[int] = ()
) -> np.ndarray | None:
    """Each box's offset from the row's line, in box heights (positive below
    it), for the box centres ``centres`` ``(x, y)``, the box height and the
    boxes' lanes ``lanes`` (lane indices in the same order, to tell
    neighbouring lanes; none given, no two are neighbours); None when the row
    is not checked.

    The row's line is ``a + b*x + c*x**2``: straight (a tilt), bent by a smile
    of at most ``ROW_SMILE`` box heights across the boxes (``c`` bounded), as
    a gel's lanes bend a row. From ``ROW_LINE_MIN`` boxes it is fitted through
    the other boxes, so a box off it cannot drag it there. First the fit most
    boxes lie near, each counting its squared distance from it up to a
    tolerance of ``ROW_LINE_TOL`` box heights or ``ROW_LINE_TOL_PX`` px,
    whichever is more, past which a box counts as off however far (M-estimator
    sample consensus), from the exact fit of every pair and triple of boxes,
    each refitted once on the boxes within the tolerance of it, and at least
    the half of the boxes (at least ``ROW_LINE_MIN``) nearest it; then the fit
    through every box within ``ROW_LINE_K`` box heights of it, again until
    those boxes stay the same. So one lane's box a box height off is fitted
    around, and four boxes are fitted all together.

    Fitted by the half of the boxes nearest it alone (least trimmed squares),
    four boxes of thin bands a pixel apart could pick a smile that fits them
    exactly and passes the others by box heights: 9% of flat rows of five
    4 px boxes with a pixel of jitter were refused, and 47 of 3000 such rows
    of 4 to 8 lanes through detection. Counting every box, up to the
    tolerance, none of those 3000 is, none of 10,000 flat rows of 4 to 8 px
    boxes with 1 to 1.5 px of jitter (3 of 2500 of 12 px boxes with 2 px, 5
    before), nor of 2000 rows of 5 and 6 px boxes bent by smiles and tilts
    (14 before). A lane 1.0 box height off is found at the end of a row of
    12 px boxes 92% of the time (77% before), of 6 px boxes 30% (55%; 1.5
    box heights off, 78% and 84%; 2 box heights off, 99%); in the middle, 98%
    or more. The tolerance: a box's centre lies on a whole pixel, and a thin
    band's centre wanders by a pixel or so.

    A box over two rows: when the boxes' heights fall in two groups more than
    a smile (``ROW_SMILE`` box heights) apart, and the fit leaves a box off,
    or a box of each group lies in neighbouring lanes (no smile or tilt steps
    a row that far from one lane to the next), the line runs through the
    larger group (on a tie, the one the fit kept more of; with nothing to
    break it, level midway, so both lie off) and the other group lies off it
    whole. So a row of two or three boxes, which no line checks (three
    centres lie on some smile), is checked for two rows alone; with four
    boxes or fewer, two rows whose boxes lie in no neighbouring lanes pass,
    as a sparse row's steep tilt does.

    A straight line alone leaves a smile's end boxes off it: through the
    other boxes, by 0.9 box heights on the bench's smile and up to 3 on the
    accuracy judge's strongest (1.8 box heights of sag), while a montage's
    lane lay 1.17 off; no threshold told them apart. Bent by a smile, no box
    of the 628 rows of the bench, the judge's rows, the recipes and the fuzz
    rows that main placed lies more than 0.44 box heights off (smiles and
    tilts 0.19; the rest a doublet's box on one of its bands), nor of 37 real
    rows placed more than 0.33; that montage's lane lies 1.17 off, a mark
    beside a row 2.4, and a row box over two rows 4.1 to 4.4."""
    n = len(centres)
    if n < 2:
        return None
    xs = np.array([c[0] for c in centres], float)
    ys = np.array([c[1] for c in centres], float)
    u = (xs - xs.mean()) / max(float(np.ptp(xs)), 1.0)  # across the boxes: -0.5..0.5
    y = ys - ys.mean()
    h = float(height)
    cmax = 4.0 * ROW_SMILE * h  # the sag of c*u**2 across -0.5..0.5 is c/4
    powers = np.stack([np.ones_like(u), u, u * u])
    offsets = None
    if n >= ROW_LINE_MIN:
        # Starts: the exact fit of every triple (a smile) and pair (a line).
        triples = np.array(list(itertools.combinations(range(n), 3)))
        w3 = np.zeros((len(triples), n))
        np.put_along_axis(w3, triples, 1.0, axis=1)
        starts = [_bounded_fits(u, y, w3, cmax)]
        pairs = np.array(list(itertools.combinations(range(n), 2)))
        b = (y[pairs[:, 1]] - y[pairs[:, 0]]) / (u[pairs[:, 1]] - u[pairs[:, 0]])
        starts.append(np.column_stack([y[pairs[:, 0]] - b * u[pairs[:, 0]], b, np.zeros(len(b))]))
        starts = np.concatenate(starts)
        # One refit of each on the boxes within the tolerance of it and the
        # half nearest it; the least sum of squares, each box's capped at the
        # tolerance, wins.
        tol = max(ROW_LINE_TOL * h, ROW_LINE_TOL_PX)
        dist = np.abs(y - starts @ powers)
        w = (dist <= tol).astype(float)
        m = max(ROW_LINE_MIN, -(-n // 2))
        np.put_along_axis(w, np.argsort(dist, axis=1, kind="stable")[:, :m], 1.0, axis=1)
        coef = _bounded_fits(u, y, w, cmax)
        cost = np.sum(np.minimum((y - coef @ powers) ** 2, tol * tol), axis=1)
        fit = coef[int(np.argmin(cost))]
        # Then through every box within ROW_LINE_K box heights of it.
        inside = np.abs(y - fit @ powers) <= ROW_LINE_K * h
        for _ in range(n):
            if inside.sum() < 3:
                break
            fit = _bounded_fits(u, y, inside[None, :].astype(float), cmax)[0]
            now = np.abs(y - fit @ powers) <= ROW_LINE_K * h
            if np.array_equal(now, inside):
                break
            inside = now
        offsets = (y - fit @ powers) / h
    # Two rows: the boxes' heights in two groups more than a smile
    # (ROW_SMILE box heights) apart, and a box off the fit, or a box of each
    # group in neighbouring lanes. The fit bent towards the other row may keep
    # some of its boxes, or all: the row's line runs through the larger group
    # (on a tie, the group the fit kept more of), and the other group lies off
    # it whole.
    order = np.argsort(y, kind="stable")
    gaps = np.diff(y[order])
    k = int(np.argmax(gaps))
    if gaps[k] <= ROW_SMILE * h:
        return offsets
    split = (order[: k + 1], order[k + 1 :])
    off = offsets is not None and bool(np.any(np.abs(offsets) > ROW_LINE_K))
    lane = list(lanes)
    neighbours = len(lane) == n and any(
        abs(lane[i] - lane[j]) == 1 for i in split[0] for j in split[1]
    )
    if not (off or neighbours):
        return offsets

    def key(group: np.ndarray) -> tuple[int, int]:
        kept = 0 if offsets is None else int(np.sum(np.abs(offsets[group]) <= ROW_LINE_K))
        return (group.size, kept)

    if key(split[0]) == key(split[1]):  # nothing tells the row's group: level midway
        fit = np.array([float(y[order[k]] + y[order[k + 1]]) / 2, 0.0, 0.0])
        return (y - fit @ powers) / h
    kept = np.zeros(n, bool)
    kept[max(split, key=key)] = True
    if kept.sum() >= 3:
        fit = _bounded_fits(u, y, kept[None, :].astype(float), cmax)[0]
    else:  # one or two boxes: level through them
        fit = np.array([float(y[kept].mean()), 0.0, 0.0])
    return (y - fit @ powers) / h


def detect_row(
    gray: np.ndarray,
    row: Sequence[int],
    n_lanes: int,
    *,
    background: float,
    dark_on_light: bool = True,
    size_rule: str = SIZE_RULE,
    right_to_left: bool = False,
) -> RowDetection:
    """One slot per declared lane in ``row`` ``(x0, y0, x1, y1)``: a box of one
    shared size, or None for an empty lane (see the module docstring).

    ``gray`` is the 2-D analysis array (``session.pixels``); it is read, never
    written. ``background`` is the image's stored background: it only stands in
    for a fitted plane that is off the membrane, and sets ``bg_offset``.
    ``size_rule`` is one of :data:`SIZE_RULES`, for tests and evaluation;
    callers that commit boxes use the default :data:`SIZE_RULE`.
    ``right_to_left`` numbers the lanes from the box's right end, for an image
    whose lanes run that way: only the numbering changes, in ``lanes`` and in
    the notes.
    Raises :class:`RowDetectError` for a row it cannot use.
    """
    if size_rule not in SIZE_RULES:
        raise ValueError(f"unknown size rule {size_rule!r}; expected one of {SIZE_RULES}")
    gray = np.asarray(gray)
    x0, y0, x1, y1 = _check_row(gray, row, n_lanes, background)
    n = int(n_lanes)
    crop = np.asarray(gray[y0:y1, x0:x1], dtype=np.float64)
    if not np.isfinite(crop).all():
        raise RowDetectError("invalid_image", "the row holds non-finite pixel values")
    hc, wc = crop.shape
    sign = 1.0 if dark_on_light else -1.0
    floor = _noise_floor(crop)
    sigma_px = max(_pixel_noise(crop), floor)

    # Stage 1: a plane over the crop, or the stored background where the plane
    # is off the membrane.
    flat = np.full(crop.shape, float(background))
    plane = flat
    coef = _fit_plane(crop, np.ones(crop.shape, bool), floor)
    if coef is not None:
        plane = _surface(crop, _subsample(np.arange(crop.size)), coef, flat, sign, sigma_px)
    crop_ds = _despeckle(crop, _despeckle_width(wc, n))
    image_rows = (y0 == 0, y1 == gray.shape[0])
    res = _run_pass(_signal(crop, crop_ds, plane, sign, sigma_px, floor, None), n, x0, image_rows)

    # Stage 2: the plane (or the stored background, the same choice) and the
    # noise from the band-free pixels of the row.
    free = _stage2_free(res, crop.shape)
    stage2 = bool(free.sum() >= max(BG_MIN_PIXELS, BG_MIN_KEEP * crop.size))
    if stage2:
        coef2 = _fit_plane(crop, free, floor)
        if coef2 is not None:
            idx = _subsample(np.flatnonzero(free.ravel()))
            plane = _surface(crop, idx, coef2, flat, sign, sigma_px)
        res = _run_pass(
            _signal(crop, crop_ds, plane, sign, sigma_px, floor, free), n, x0, image_rows
        )
    sig, assign, lanes = res.sig, res.assign, res.lanes
    # The membrane's level in the signal, for bg_offset: stage 2 measured it on
    # the band-free pixels. Too few of them to fit and detect again may still
    # be enough to read it under the stage-1 surface.
    membrane = sig.offset
    notes = list(res.notes)
    if not stage2:
        notes.append("stage 2 skipped: too few band-free pixels")
        if free.sum() >= MIN_FIT:
            membrane = _signal(crop, crop_ds, sig.plane, sign, sigma_px, floor, free).offset
    _count_components(res)
    pitch = assign.pitch if assign is not None else wc / n
    _empty_lanes(res, n, wc, pitch)

    def numbered(indices: Sequence[int]) -> list[int]:
        """Lane indices as the caller numbers them, in order."""
        return sorted(n - 1 - i if right_to_left else i for i in indices)

    # Geometry the row box cannot resolve. A lane holds a band exactly when
    # its growth gave an extent.
    flags: list[str] = []
    extents = {i: ln.rect for i, ln in enumerate(lanes) if ln.rect is not None}
    present = list(extents)
    margin = None
    if assign is not None and present:
        if (present[0] > 0 and lanes[0].centre < 0.0) or (
            present[-1] < n - 1 and lanes[n - 1].centre > wc
        ):
            flags.append("lanes_outside_row")
        margin = float(res.alt - assign.cost)
        if margin < AMBIGUITY_MARGIN:
            flags.append("ambiguous_lanes")

    # A band the row box cuts through: its extent reaches the box's top or
    # bottom edge, and that edge row still holds CUT_LEVEL of its peak there;
    # or, in an empty lane, it peaks on that edge row (_empty_lanes).
    cut = {
        i
        for i, (ex0, ey0, ex1, ey1) in extents.items()
        if any(
            float(sig.s_sm[y, ex0:ex1].mean()) >= CUT_LEVEL * lanes[i].peak
            for y, touches in ((0, ey0 <= 0), (hc - 1, ey1 >= hc))
            if touches
        )
    } | {i for i, ln in enumerate(lanes) if ln.cut and i not in extents}

    # One shared size; bounded isotonic placement inside the row.
    size = None
    out: dict[int, Rect] = {}
    outliers: tuple[int, ...] = ()
    offsets: dict[int, float] = {}  # each box's offset from the row's line (_row_line)
    if present:
        rects = list(extents.values())
        w, h, outliers = _shared_size(
            [r[2] - r[0] for r in rects],
            [r[3] - r[1] for r in rects],
            size_rule,
            cut_w=[r[0] <= 0 or r[2] >= wc for r in rects],
            cut_h=[r[1] <= 0 or r[3] >= hc for r in rects],
        )
        if len(rects) > 1:  # no wider than the closest pair of extent centres
            gaps = np.diff([0.5 * (r[0] + r[2]) for r in rects])
            min_w = _min_width(wc, n)
            crowded = np.flatnonzero(gaps < min_w)
            if crowded.size:  # one centre, crossed, or closer than a band: not two lanes
                flags.append("ambiguous_lanes")
                pairs = sorted({present[k] for k in crowded} | {present[k + 1] for k in crowded})
                notes.append(
                    f"{lanes_phrase(numbered(pairs))}: extents closer than the narrowest band "
                    f"({min_w:.0f} px) or out of order"
                )
            w = min(w, int(math.floor(max(float(gaps.min()), min_w))))
        w = max(MIN_BOX, min(w, wc // len(rects)))  # len * w fits: MIN_BOX * n <= wc
        h = max(MIN_BOX, min(h, hc))
        xs = place_in_row([(r[0] + r[2]) // 2 - w // 2 for r in rects], w, 0, wc)
        for i, r, x in zip(present, rects, xs, strict=True):
            y = max(0, min((r[1] + r[3]) // 2 - h // 2, hc - h))
            out[i] = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
        size = BoxSize(width=w, height=h)
        # The row's line through the boxes' centres (crop coordinates, so a
        # shifted row gives the same offsets): a box off it is off the row.
        line = _row_line(
            [((r[0] + r[2]) / 2 - x0, (r[1] + r[3]) / 2 - y0) for r in out.values()],
            h,
            list(out),
        )
        if line is not None:
            offsets = {i: float(v) for i, v in zip(out, line, strict=True)}
        off_line = [i for i, v in offsets.items() if abs(v) > ROW_LINE_K]
        if off_line:
            flags.append("off_row_line")
            if len(off_line) == len(out):  # two rows, neither the row's
                notes.append(
                    f"{lanes_phrase(numbered(off_line))}: box centres more than {ROW_SMILE:g}"
                    f" box heights ({ROW_SMILE * h:.0f} px) apart, on two rows"
                )
            else:
                notes.append(
                    f"{lanes_phrase(numbered(off_line))}: box centre more than {ROW_LINE_K:g}"
                    f" box height ({ROW_LINE_K * h:.0f} px) off the row's line through the"
                    " other boxes"
                )

    result: list[LaneDetection] = []
    for i, ln in enumerate(lanes):
        rect = out.get(i)
        extent = bg_offset = window = None
        if rect is None and ln.window is not None:
            wx0, wy0, wx1, wy1 = ln.window
            window = (x0 + wx0, y0 + wy0, x0 + wx1, y0 + wy1)
        if rect is not None:
            ex0, ey0, ex1, ey1 = extents[i]
            extent = (x0 + ex0, y0 + ey0, x0 + ex1, y0 + ey1)
            # The membrane level under the box: the detection surface there,
            # moved by the membrane's level in the signal (r = sign * (surface
            # - crop) - membrane is 0 on the membrane).
            under = sig.plane[rect[1] - y0 : rect[3] - y0, rect[0] - x0 : rect[2] - x0]
            level = float(under.mean()) - sign * membrane
            bg_offset = sign * (level - float(background)) / sig.sigma_px
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
                window=window,
                cut=i in cut,
                line_offset=offsets.get(i),
            )
        )
    if right_to_left:
        result = [replace(ld, lane=n - 1 - ld.lane) for ld in reversed(result)]
    if any(ld.bg_offset is not None and abs(ld.bg_offset) > BG_WARN_K for ld in result):
        flags.append("background_mismatch")
    if outliers:
        flags.append("size_outlier")
        notes.append(
            f"{lanes_phrase(numbered([present[k] for k in outliers]))}: extent above "
            f"{SIZE_GUARD:g}x the median of the other extents, left out of the shared size"
        )
    multiple = [ld.lane for ld in result if ld.components > 1]
    if multiple:
        flags.append("multiple_components")
        notes.append(
            f"{lanes_phrase(multiple)}: a second separate component reaches the "
            "detection level; the box covers the one with the lane's strongest pixel"
        )
    if cut:
        flags.append("cut_by_row_box")
        notes.append(
            f"{lanes_phrase(numbered(cut))}: the row box's top or bottom edge cuts through"
            f" the band (the box's edge row holds at least {CUT_LEVEL:.0%} of its peak)"
        )
    # How far detection's membrane lies inside the bands, by the box's top and
    # bottom rows (membrane_shift): the box-smoothed rows' means about the
    # detection surface's mean level (a surface tilted by a light margin in the
    # box would lie inside the membrane at one end), over the stage-1 pixel
    # noise (neighbour differences, which a band taken into the stage-2 noise
    # estimate does not inflate).
    ky, kx = min(SMOOTH[0], hc), min(SMOOTH[1], wc)
    smoothed = uniform_filter(crop, size=(ky, kx), mode="nearest").mean(axis=1)
    rows = sign * (float(sig.plane.mean()) - smoothed) - sig.offset
    ends = max(float(rows[0]), float(rows[-1]))
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
        membrane_shift=min(-ends, float(rows.max()) - ends) / sigma_px,
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
        "cut_level": CUT_LEVEL,
        "row_line_k": ROW_LINE_K,
        "row_smile": ROW_SMILE,
        "row_line_min": ROW_LINE_MIN,
        "row_line_tol": ROW_LINE_TOL,
        "row_line_tol_px": ROW_LINE_TOL_PX,
        "smooth": list(SMOOTH),
        "min_width_px": MIN_WIDTH_PX,
        "min_width_pitch": MIN_WIDTH_PITCH,
        "despeckle_min": DESPECKLE_MIN,
        "edge_hump": EDGE_HUMP,
        "row_walk_tol": ROW_WALK_TOL,
        "row_min_rows": ROW_MIN_ROWS,
        "row_min_keep": ROW_MIN_KEEP,
        "flat_edge": FLAT_EDGE,
        "line_span": LINE_SPAN,
        "line_flat": LINE_FLAT,
        "line_px": LINE_PX,
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
        "membrane_spread_k": MEMBRANE_SPREAD_K,
        "min_fit": MIN_FIT,
        "fit_max_pixels": FIT_MAX_PIXELS,
        "noise_floor_frac": NOISE_FLOOR_FRAC,
        "empty_window": EMPTY_WINDOW,
        "bg_warn_k": BG_WARN_K,
        "membrane_shift_k": MEMBRANE_SHIFT_K,
        "min_box": MIN_BOX,
    }
