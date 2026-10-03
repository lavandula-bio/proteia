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
   at ``EXTENT_LEVEL`` of the lane's strongest pixel, or of its peak nearest
   the row a caller expects the band on, ``prefer_y``, its seed range's own
   peaks among them, stopping at the valley to any other band above or below
   it, whatever its height: :func:`_without_other_bands`; there, where the
   saturation level is known, two clipped cores with a valley of ``DETECT_K``
   sigma between them are two peaks, :func:`_clipped_peaks`) between the
   lane's walls, joined across a burnt-out centre that splits a saturated
   band along x (:func:`_join_burnt_out`, where the saturation level is
   known). A lane read from a marked piece is not grown: it stays empty.
7. Stage 2: the plane (or the stored background, chosen as in stage 1) and the
   noise again from the band-free pixels of the row, then steps 3 to 6 again.
8. Each band's lane: its separate components counted (peaks split as pieces
   are along x; a second one only at :data:`SECOND_SHARE` of the lane's
   peak, in the band's rows and as wide as a band; a peak inside the band's
   grown extent is the band's own: :func:`_count_components`), its peaks
   listed (``peaks``: every one, wherever it lies in the lane, for a count
   of the bands in a window along it, #58), and whether the band is hollow
   (:func:`_hollow`). Empty lanes get a reason; flags; one shared size by
   :data:`SIZE_RULE`, capped by the lane spacing and the box; bounded
   isotonic placement (:func:`~proteia.core.boxes.place_in_row`).
9. The row's line through the boxes' centres (:func:`_row_line`): a box off
   it is off the row; so are lanes grown with ``prefer_y`` from bands on two
   rows (``crossed``, :func:`_measure`). With ``prefer_y``, a few lanes off
   the line whose own slot on it holds nothing are recorded as not detected
   instead (``off_expected_row``, :func:`_not_detected`), and the row is
   placed again without them.
10. The lane reading against the bands' own spacing (:func:`_doubts`): a
    reading that does not fit it is placed, its lane numbers doubtful.

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
  or, with ``prefer_y``, two neighbouring lanes grew from bands on two rows,
  each lane holding a band on both (``crossed``: the row box covers two rows
  and the expected row lies between them, nearer one in some lanes and the
  other in the rest). With ``prefer_y``, lanes off the line become
  ``off_expected_row`` instead where :func:`_not_detected` says so; else the
  row stays refused, and ``off_cause`` says why;
* ``off_expected_row``: with ``prefer_y``, fewer than half of the boxed lanes
  lay off the row's line, each also off the expected row, with no other band
  reaching ``DETECT_K`` on the line in its slot, and every other box on the
  expected row: those lanes are empty (reason ``off_expected_row``, their
  ``snr`` read on the line, their ``window`` where it was read), and the row
  is placed without them;
* ``doubtful_lanes``: the lane reading does not fit the bands' own spacing,
  so the lanes may be numbered wrong (:func:`_doubts`): the fitted pitch lies
  more than :data:`PITCH_DOUBT` of that spacing off it, the resolved bands or
  a touching run's cells span more than :data:`LANES_DOUBT` lanes off the
  lanes read, the box reaches more than :data:`END_DOUBT` spacings past an
  end lane's centre, or pieces at least :data:`APART_DOUBT` lanes apart were
  merged, or one between the others dropped, to fit the declared lanes: as a
  row box that also covers a ladder, labels or a neighbouring panel reads
  (#111). The note begins ``lane numbers doubtful:``
  (:attr:`RowDetection.doubt_note`);
* ``background_mismatch``: the membrane under a box differs from the stored
  background by more than :data:`BG_WARN_K` pixel sigmas;
* ``size_outlier``: an extent above :data:`SIZE_GUARD` times the median of the
  other extents was left out of the shared size (``"max_guarded"``);
* ``multiple_components``: a lane holds a second, separate component (see
  ``components``): a doublet's weaker band, a non-specific band, a band split
  by a bubble; its box is grown from the lane's strongest pixel, as a click
  there would be (quantifying doublets is #58's), or with ``prefer_y`` from
  its peak nearest that row, every other band left out (the note says where
  each lies from the box's centre, and whether the box holds part of it);
* ``hollow_band``: a lane's band is hollow (see ``hollow``): lighter in its
  centre than the saturated pixels on either side of it, a sign of
  over-exposure (#121); only where the caller gives the saturation level
  (``saturated_at``);
* ``cut_by_row_box``: the row box cuts through a band (see ``cut``): the
  band's extent reaches the box's top or bottom edge and that edge row, across
  the extent, still holds :data:`CUT_LEVEL` of the band's peak, so its box and
  net miss what lies beyond the edge; or, in an empty lane, the band peaks on
  that edge row itself and was left out (``edge_signal``).

Every setting is a module constant, reported by :func:`settings`.

A row along a sloping line (:func:`detect_row_along`, #58): the columns of the
row box are moved by whole pixels so the line lies level, :func:`detect_row`
runs unchanged on them, and what it found is moved back.
"""

from __future__ import annotations

import itertools
import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, NamedTuple

import numpy as np
from pydantic import JsonValue
from scipy.ndimage import (
    binary_fill_holes,
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

from proteia.core import quantify
from proteia.core.boxes import place_in_row
from proteia.core.grow import NOISE_K, REL_THRESHOLD, grow_region, mad_sigma
from proteia.core.model import BoxSize, Rect, lane_number, lanes_phrase

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
# Doubtful lane numbers (#111), at the bands' own spacing (_doubts): the fitted
# pitch off it by more than PITCH_DOUBT of it (twice SPACING_TOL); a stretch of
# the reading (its resolved bands end to end, a touching run's cells) off by
# more than LANES_DOUBT lanes (the rounding to another count); the box reaching
# more than END_DOUBT spacings past an end lane's centre (END_HI and half a
# lane: room for another lane); or pieces merged, or one between the others
# dropped, to fit the declared lanes at least APART_DOUBT lanes apart.
PITCH_DOUBT: Final = 0.3
LANES_DOUBT: Final = 0.5
END_DOUBT: Final = 1.5
APART_DOUBT: Final = 0.6
# A lane's second component (#121, R5; to be re-checked on raw scans): a peak
# outside the band's grown extent that reaches SECOND_SHARE of the lane's peak
# (in the band's rows, as wide as a band). A band is hollow when its grown
# extent holds HOLLOW_PIXELS saturated pixels, the "possibly over-exposed"
# count (#112), and HOLLOW_PIXELS of a lighter centre between them (_hollow).
# A band split along x by such a centre is joined with a piece beside it that
# shares JOIN_ROWS of the rows the two span (_join_burnt_out): its other half,
# not a line or stroke crossing its rows, nor a band above or below it.
SECOND_SHARE: Final = 0.25
HOLLOW_PIXELS: Final = quantify.POSSIBLY_CLIPPED_PIXELS
JOIN_ROWS: Final = 0.5

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
    "doubtful_lanes",
    "background_mismatch",
    "size_outlier",
    "multiple_components",
    "hollow_band",
    "cut_by_row_box",
    "off_expected_row",
)

RowDetectErrorCode = Literal["invalid_row", "invalid_image", "row_outside_image", "row_too_small"]
LaneReason = Literal[
    "band",
    "no_band",
    "artefact",
    "line",
    "edge_signal",
    "side_signal",
    "unassigned",
    "off_expected_row",
]
# Why lanes off the row's line were not recorded as not detected
# (_not_detected): half or more of the boxes off it; an off lane's box on the
# expected row; another band, or other signal, on the line in an off lane's
# slot; another box off the expected row; or a box still off the line placed
# again without them.
OffCause = Literal["half", "expected_row", "line_signal", "other_box", "again"]

_DOUBT_NOTE: Final = "lane numbers doubtful: "  # the doubtful_lanes note begins so
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


class Peak(NamedTuple):
    """One separate peak of a lane's detection signal (#58): a top of the kept
    signal at ``DETECT_K`` sigma that :func:`_count_components` finds, in image
    coordinates.

    ``y`` and ``x`` are continuous, as a box centre is (row r covers [r, r+1)):
    the top's pixel centre, ``y`` refined by a parabola through the smoothed
    signal on the rows above and below it (at most half a row either way).
    ``snr`` is its height over the noise. ``own``: it lies in the band's grown
    extent, the band's own peak (a dumbbell's or a hollow band's ends are two
    of them). ``other_band``: it reads as another band, wherever it lies along
    the lane: outside the band's grown extent, at least ``SECOND_SHARE`` of the
    lane's peak (the band's highest pixel, or a higher peak in the lane), and
    its hill as wide as a band (the candidates' width rule), as ``components``
    counts a second component; unlike there, a band apart above or below, with
    membrane between, is one. Of the tops of one such band side by side (a
    dumbbell's ends, a hollow band's), only the highest: the band's own tops
    are one band, and so are another's (:func:`_one_top_per_band`). Neither: a
    weaker peak, a band's second top, or JPEG block noise."""

    y: float
    x: float
    snr: float
    own: bool
    other_band: bool


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
      lanes, or too weak beside the bands around it), ``off_expected_row``
      (with ``prefer_y``: the band found lay off the row's line and off the
      expected row, and nothing reaches ``DETECT_K`` on the line in the
      lane's slot: not detected there, ``off_expected_row``).
    * ``snr``: the strongest smoothed signal in the lane (the growth seed; for an
      empty lane, within ``EMPTY_WINDOW`` pitch of its centre, leaving out the
      candidates the detector rejected, dust among them) over the noise. For
      an ``off_expected_row`` lane, the reading on the row's line
      (``line_snr``): the band found's own tail there less its reflection.
    * ``extent``: the grown extent before sizing (of a band split along x by
      a burnt-out centre, both halves and the centre: :func:`_join_burnt_out`),
      None for an empty lane.
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
      ``VALLEY_FRAC`` of it, as two pieces along x (with ``prefer_y`` and the
      saturation level, a clipped peak whose clipped pixels lie apart from
      the higher one's needs only the first: :func:`_clipped_peaks`). A peak
      inside the band's grown extent is the band's own: a dip between two of
      them (a dumbbell, a hollow centre) leaves one band. Any other counts
      only if it reaches ``SECOND_SHARE`` of the lane's peak, its own part of
      the signal (above its saddle to higher ground) overlaps the band's rows
      (those the band's signal above ``NOISE_K`` sigma fills in the lane: a
      doublet's band joined to it does, as does a half beside it), and it
      passes the candidates' width rule, as a band does: JPEG block noise and
      specks, and a band apart above or below with membrane between, do not.
      With ``prefer_y``, every other band above or below the one grown stays
      outside its extent, so a band joined to it counts so. 1 for a lone
      band, 2 or more for several (a doublet, a non-specific band joined to
      it, a band split by a bubble), 0 when empty.
    * ``hollow``: the lane's band is hollow (``hollow_band``, :func:`_hollow`):
      burnt out in its middle, lighter there than the pixels at or past
      ``saturated_at`` on either side of it along its rows, a sign of
      over-exposure: split along x by a lighter centre, or a ring around one.
      A centre that falls below the extent level splits the grown extent in
      two; the halves, saturated on either side of it in the same rows, are
      one band, its extent and box over both (:func:`_join_burnt_out`). Two
      bands stacked (a close doublet, one or both saturated), two side by side
      not both saturated, and a notch in a saturated band's top or bottom edge
      are not. False for an empty lane, and whenever ``saturated_at`` is None.
    * ``window``: the slot an empty lane's ``snr`` was read in: within
      ``EMPTY_WINDOW`` pitch of its expected centre along x, clipped to the row
      box, over the row's rows (a neighbouring row left out); for an
      ``off_expected_row`` lane, within ``EMPTY_WINDOW`` pitch of its band's
      box centre, over the rows its reading on the row's line read (the
      line's row, the one below it, and the smoothing's row on either side:
      four rows, clipped to the row box). None for a lane that holds a band,
      and for an empty lane whose slot lies outside the row box (not
      measured: its ``snr`` is 0).
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
      ``ROW_LINE_MIN`` boxes (not checked) unless they lie on two rows. An
      ``off_expected_row`` lane keeps its band's offset from the line the
      row was first fitted with.
    * ``peaks``: every separate peak of the lane's detection signal between its
      walls (:class:`Peak`), top to bottom, the band's own included: where a
      count of the bands in a window along the lane reads them (#58,
      :func:`bands_in`). ``()`` for an empty lane.
    * ``expected_offset``: with ``prefer_y``, how far the box's centre lies
      below (positive) or above (negative) the expected row, in box heights;
      for an ``off_expected_row`` lane, its band's box's before it was left
      out. None without ``prefer_y`` and for an empty lane.
    * ``line_snr`` and ``line_reason``: with ``prefer_y``, for a lane off the
      row's line (``off_row_line``, or ``off_expected_row``), what its slot on
      the line holds (:func:`_on_the_line`): the reading's SNR, and the reason
      it would leave an empty lane by (``no_band``: nothing reaches
      ``DETECT_K`` there; ``outside``: the line runs outside the row box
      there). None for every other lane, and when the row was not checked
      for lanes to record (a refusing flag besides the line's, or lanes on
      two rows).
    """

    lane: int
    rect: Rect | None
    reason: LaneReason
    snr: float
    extent: Rect | None
    expected_x: float
    bg_offset: float | None
    components: int
    hollow: bool
    window: Rect | None
    cut: bool
    line_offset: float | None
    peaks: tuple[Peak, ...] = ()
    expected_offset: float | None = None
    line_snr: float | None = None
    line_reason: str | None = None


def bands_in(lane: LaneDetection, top: float, bottom: float) -> int:
    """How many bands a lane holds between the rows ``top`` and ``bottom``
    (continuous, both included) by its peaks (#58): its band, and each peak
    in them that reads as another band (:attr:`Peak.other_band`). 0 for an
    empty lane."""
    if lane.rect is None:
        return 0
    return 1 + sum(1 for peak in lane.peaks if peak.other_band and top <= peak.y <= bottom)


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
    band. ``crossed`` holds the lanes, as numbered, grown with ``prefer_y``
    from bands on two rows (``off_row_line``): each next to a lane grown from
    a band on the other row, both holding a band on both rows (see
    :func:`_measure`); empty without ``prefer_y``. ``off_cause``: with
    ``prefer_y``, why lanes off the row's line were not recorded as not
    detected and the row stays refused (:data:`OffCause`,
    :func:`_not_detected`); None when they were, or when nothing was checked.
    ``again``: for ``off_cause`` ``again``, each box still off the line when
    the row was placed again without the lanes to record, as ``(lane as
    numbered, its offset from that line in box heights)``.
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
    crossed: tuple[int, ...] = ()
    off_cause: OffCause | None = None
    again: tuple[tuple[int, float], ...] = ()

    @property
    def slots(self) -> tuple[Rect | None, ...]:
        """One rect or None per declared lane, in lane order."""
        return tuple(lane.rect for lane in self.lanes)

    @property
    def refused(self) -> bool:
        """True when a refusing flag is set: the caller proposes nothing."""
        return any(flag in REFUSING_FLAGS for flag in self.flags)

    @property
    def doubt_note(self) -> str | None:
        """The note of the ``doubtful_lanes`` flag, None without it: a caller
        that checks the lanes another way drops both."""
        return next((note for note in self.notes if note.startswith(_DOUBT_NOTE)), None)


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


@dataclass(frozen=True)
class _Joined:
    """A pair of pieces :func:`_reduce` merged, or dropped the weaker of: its
    crop x ``[l, r)`` (the pair's, or the dropped piece's) and the distance
    between the pair's centres, px."""

    l: float  # noqa: E741
    r: float
    apart: float
    dropped: bool


def _reduce(
    pieces: list[_Piece], n: int, x_offset: int, notes: list[str]
) -> tuple[list[_Piece], list[_Joined]]:
    """More pieces than lanes: drop the weaker of the closest pair if its mass is
    below ``REDUCE_MASS`` of the other's, otherwise merge the pair. Returns the
    pieces left and each pair joined so (:class:`_Joined`)."""
    pieces = list(pieces)
    joined: list[_Joined] = []
    while len(pieces) > n:
        gaps = [pieces[i + 1].c - pieces[i].c for i in range(len(pieces) - 1)]
        i = int(np.argmin(gaps))
        a, b = pieces[i], pieces[i + 1]
        if min(a.mass, b.mass) < REDUCE_MASS * max(a.mass, b.mass):
            drop = i if a.mass < b.mass else i + 1
            p = pieces.pop(drop)
            notes.append(f"dropped a weak piece at x={x_offset + p.l:.0f}..{x_offset + p.r:.0f}")
            joined.append(_Joined(p.l, p.r, b.c - a.c, True))
        else:
            notes.append(f"merged pieces at x={x_offset + a.l:.0f}..{x_offset + b.r:.0f}")
            merged = _Piece(a.l, b.r, max(a.peak, b.peak), a.mass + b.mass, a.side and b.side)
            pieces[i : i + 2] = [merged]
            joined.append(_Joined(a.l, b.r, b.c - a.c, False))
    return pieces, joined


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
    grown: np.ndarray | None = None  # the grown region's mask (crop shape)
    components: int = 0
    hollow: bool = False
    # (crop y, crop x, snr, own, other_band) of each peak: _count_components
    peaks: tuple[tuple[float, float, float, bool, bool], ...] = ()
    window: Rect | None = None  # an empty lane's measured slot (crop coordinates)
    cut: bool = False  # an empty lane's band peaks on the box's edge row (_empty_lanes)
    side: bool = False  # read from a piece rising into the box's side: not measured
    crossed: bool = False  # grown from a band on another row than a neighbour's (_measure)
    # With prefer_y: the pixels of the lane's window that drain to the band it
    # grew from (its basin, _band_basins; crop shape), read on the row's line
    # (_on_the_line).
    own: np.ndarray | None = None
    # Each second component's peak (continuous crop y, x): _count_components.
    second: tuple[tuple[float, float], ...] = ()
    seed_row: int = 0  # the growth seed's row (crop)


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


def _peaks(ks: np.ndarray, h: float, frac: float = VALLEY_FRAC) -> list[tuple[int, int]]:
    """``(y, x)`` of the separate peaks of ``ks`` (>= 0): the highest, if it
    reaches ``h``, and each other peak whose highest saddle to a higher peak is
    at least ``h`` below it and at most ``frac`` of it (``VALLEY_FRAC``: the rule
    that separates two pieces along x, :func:`_split_run`). A peak is a 4-connected
    set of equal pixels, reported at its first pixel in raster order. Noise on
    a band's top, and a dent in it, fall far short; a second band does not.

    Candidates first, by the h-maxima transform: reconstruction by dilation of
    ``ks - h`` under ``ks`` leaves ``h`` on top of the peaks at least ``h`` above
    their saddle, and of peaks of one height joined above that. Then, from the
    highest down, a candidate is dropped if its region above the saddle level
    ``min(height - h, frac * height)`` holds a higher pixel or a peak
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
    levels = [min(v - h, frac * v) + eps for v in heights]
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


def _clipped_peaks(
    s: np.ndarray,
    kept: np.ndarray,
    h: float,
    saturated: np.ndarray,
    tops: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """The peaks of the kept signal ``ks`` (``s`` where ``kept``, else 0) at
    the saturation level that :func:`_peaks` at ``h`` leaves out of its
    ``tops`` for a shallow valley to a clipped band above or below them
    (#58), highest first: each peak on a pixel of ``saturated`` whose highest
    saddle to higher ground lies ``h`` below it, however near its height,
    where each top that :func:`_peaks` joins it to (the tops in its region
    above the level it reads, ``min(height - h, VALLEY_FRAC * height)``: a
    lower one lies outside it by the same rule) holds clipped pixels stacked
    above or below its own (the 4-connected pieces of ``saturated`` in
    ``ks > 0`` holding the two share too few rows to lie side by side,
    :func:`_share_rows`; one piece shares all of its own),
    and the extent a click on it would grow (``ks`` above ``EXTENT_LEVEL`` of
    it) is not hollow (:func:`_hollow`).

    Clipped, a peak's height is not its band's: the valley to a band as
    clipped above or below it is deeper than its height shows, and
    ``VALLEY_FRAC`` cannot be read from it. Two clipped bands stacked are two
    peaks where their valley falls ``h``. Clipped pixels side by side are one
    band's, as tops side by side are (:func:`_band_basins`): a clipped
    dumbbell's ends; so are the pixels of one piece, and a ring of them
    round a lighter centre, whole or, tilted, in two pieces. A top joined to
    one that is not clipped stays as :func:`_peaks` reads it."""
    ks = np.where(kept, s, 0.0)
    cores, count = label(saturated & (ks > 0.0))
    if count == 0:
        return []
    spans = [(r.start, r.stop) for r, _ in find_objects(cores)]
    found = list(tops)
    extra: list[tuple[int, int]] = []
    for t in _peaks(ks, h, frac=1.0):
        if t in found or not saturated[t]:
            continue
        v = float(ks[t])
        region, _ = label(ks > min(v - h, VALLEY_FRAC * v) + h * 1e-9)
        own = cores[t]
        joined = [cores[u] for u in found if region[u] == region[t]]  # all at least as high
        if any(c == 0 or _share_rows(spans[c - 1], spans[own - 1]) for c in joined):
            continue
        extent, _ = label(ks > EXTENT_LEVEL * v)
        if _hollow(extent == extent[t], s, saturated, h):
            continue
        found.append(t)
        extra.append(t)
    return extra


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


def _measure(
    lanes: list[_Lane],
    s: np.ndarray,
    kept: np.ndarray,
    sigma_sm: float,
    saturated: np.ndarray | None,
    tops: list[tuple[int, int]] | None = None,
    prefer: float | None = None,
) -> None:
    """Grow each present lane from its strongest kept pixel, confined between its
    walls: the midpoint to a present neighbour, the centre of an empty one, the
    box edge at the ends. Where ``saturated`` is given (the crop's pixels at the
    saturation level), a band split along x by a burnt-out centre is grown
    whole (:func:`_join_burnt_out`). A touching cell keeps its own x-range. A
    lane read from a piece rising into the box's side is not grown, and ends up
    empty, but walls its neighbour in as a present one does.

    Given a preferred row ``prefer`` (continuous, in the crop's rows) and the
    separate peaks ``tops`` of the kept signal (:func:`_peaks` at
    ``DETECT_K`` sigma, and :func:`_clipped_peaks` where the saturation level
    is known), a lane grows instead from the peak in its seed range
    nearest that row (of two as near, the higher), or from its strongest
    kept pixel when the range holds none (#58: the row an expected MW
    predicts, not a stronger band beside it). The seed range's peaks are
    those of ``tops`` in it and its own (:func:`_peaks` of the range alone),
    each of these more than a row from every one of those, and not beside
    the lane (:func:`_beside_the_lane`): a band touching a stronger
    neighbour's, its top joined to that band's, is no peak of the crop, yet
    it is the lane's own. A band grown from such a peak stops at
    the valley to every other band above or below it, whatever its height
    (:func:`_without_other_bands`): growth keeps its threshold, so the band's
    extent on its other sides is as it would be alone, and each other band
    stays outside it. The peaks in the lane's window are read as bands by
    :func:`_band_basins`; the pixels that drain to the seed's band are the
    lane's ``own``. Two neighbouring
    lanes so grown are ``crossed`` when each holds another band's top nearer
    the other's peak (moved by the step between the lanes' rows,
    :func:`_step`) than any of its own band's: they grew from bands on two
    rows, as when the expected row lies between two rows and a smile brings
    it nearer the one in some lanes and the other in the rest."""
    wc = s.shape[1]
    n = len(lanes)
    ks = np.where(kept, s, 0.0)
    # Each lane grown from a peak nearest the preferred row: that peak's row,
    # each of its seed range's peaks' rows with whether it is of its band, and
    # the seed range's highest signal in each row.
    rows: dict[int, tuple[float, list[tuple[float, bool]], np.ndarray]] = {}
    for i, ln in enumerate(lanes):
        if not ln.present or ln.x_range is None or ln.side:
            continue
        a0, b0 = ln.x_range
        if ln.cell:  # seed away from the sides of a touching cell (the central half)
            q = CELL_SEED * (b0 - a0)
            a0, b0 = a0 + q, b0 - q
        wa = max(0, int(math.floor(a0)))
        wb = min(wc, max(wa + 1, int(math.ceil(b0))))
        walls = []
        for j in (i - 1, i + 1):
            if j < 0 or j >= n:
                walls.append(0.0 if j < 0 else float(wc))
            elif lanes[j].present:
                walls.append(0.5 * (ln.centre + lanes[j].centre))
            else:
                walls.append(lanes[j].centre)
        la, lb = max(0, int(math.floor(walls[0]))), min(wc, int(math.ceil(walls[1])))
        near = [] if prefer is None or tops is None else [t for t in tops if wa <= t[1] < wb]
        own: list[tuple[int, int]] = []  # the seed range's own peaks, apart from those
        if prefer is not None and tops is not None:
            own = [
                (y, x + wa)
                for y, x in _peaks(ks[:, wa:wb], DETECT_K * sigma_sm)
                if not _beside_the_lane(ks, y, x + wa, wa, wb)
                and all(abs(y - t[0]) > 1 for t in near)
            ]
            near += own
        if near:
            sy, sx = min(near, key=lambda t: (abs(t[0] + 0.5 - prefer), t[0]))
        else:
            sub = ks[:, wa:wb]
            sy, sx = divmod(int(np.argmax(sub)), sub.shape[1])
            sx += wa
        v = float(ks[sy, sx])
        if v <= 0.0:
            ln.present = False
            continue
        ga, gb = min(la, sx), max(lb, sx + 1)
        window = ks[:, ga:gb]
        threshold = max(EXTENT_LEVEL * v, NOISE_K * sigma_sm)
        if prefer is not None:  # the band it grows from: its basin, or its part of the signal
            parts, _ = label(window > 0.0)
            mine = parts == parts[sy, sx - ga]
        if near:
            inside = [  # the seed too
                (t[0], t[1] - ga) for t in [*(tops or []), *own] if ga <= t[1] < gb
            ]
            basin, band = _band_basins(window, inside, threshold)
            seed = inside.index((sy, sx - ga))
            if basin is not None:
                mine = np.isin(basin, [k + 1 for k, b in enumerate(band) if b == band[seed]])
            window = _without_other_bands(window, basin, band, seed)
            rows[i] = (
                sy + 0.5,
                [
                    (t[0] + 0.5, band[k] == band[seed])
                    for k, t in enumerate(inside)
                    if wa <= t[1] + ga < wb
                ],
                ks[:, wa:wb].max(axis=1),
            )
        g = grow_region(window, (sx - ga, sy), threshold)
        if g is None:
            ln.present = False
            continue
        lab, _ = label(window > threshold)  # the region grow_region bounds
        grown = lab == lab[sy, sx - ga]
        rect = (g[0] + ga, g[1], g[2] + ga, g[3])
        c0, c1 = ln.x_range
        ln.span = (max(ga, int(math.floor(c0))), min(gb, int(math.ceil(c1))))
        if saturated is not None:
            span = (ln.span[0] - ga, ln.span[1] - ga)
            h = DETECT_K * sigma_sm
            joined = _join_burnt_out(lab, grown, s[:, ga:gb], saturated[:, ga:gb], h, span)
            if joined is not None:
                grown = joined
                [(ry, rx)] = find_objects(grown.astype(np.int8))
                rect = (rx.start + ga, ry.start, rx.stop + ga, ry.stop)
        ln.grown = np.zeros(s.shape, bool)
        ln.grown[:, ga:gb] = grown
        if prefer is not None:
            ln.own = np.zeros(s.shape, bool)
            ln.own[:, ga:gb] = mine
        if ln.cell:
            x0 = int(math.floor(c0))
            rect = (x0, rect[1], max(x0 + 1, int(math.floor(c1))), rect[3])
        ln.rect = rect
        ln.peak = v
        ln.seed_row = sy
        ln.snr = v / sigma_sm
        ln.reason = "band"
    for ln in lanes:
        if ln.side:
            ln.present = False
    # Two neighbouring lanes grown from bands on two rows: where the one's
    # peak lies, moved by the step between the lanes' rows (_step), the other
    # holds another band's top nearer it than any of its own band's, and the
    # other way about.
    grown = [i for i in sorted(rows) if lanes[i].present]
    for i, j in itertools.pairwise(grown):
        (yi, ti, pi), (yj, tj, pj) = rows[i], rows[j]
        step = _step(pi, pj)
        if _other_row(tj, yi + step) and _other_row(ti, yj - step):
            lanes[i].crossed = lanes[j].crossed = True


def _band_basins(
    window: np.ndarray, tops: list[tuple[int, int]], threshold: float
) -> tuple[np.ndarray | None, list[int]]:
    """The basins of the separate peaks ``tops`` ``(y, x)`` of ``window``
    (>= 0), and the band each top is in (#58). ``basin`` labels each pixel of
    ``window > 0`` with 1 + the index of the top it drains to (a watershed of
    the signal from the tops, 4-connected), 0 elsewhere; None for one top.
    ``band[k]`` is the least index of a top in top ``k``'s band. ``threshold``
    is the growth threshold (:func:`_measure`).

    Tops of one band lie side by side, along the rows; a band above or below
    another is another band. So two tops are of one band when their basins
    meet along a valley that runs down the rows at least as much as along
    them (of the pixel pairs across it, at least as many lie side by side as
    one above the other), or, with no path above 0 between them (a burnt-out
    centre below the noise splits a band along x), when the pieces above
    ``threshold`` that hold them share their rows (:func:`_share_rows`): the
    pieces :func:`_join_burnt_out` reads, by its rule (a top below
    ``threshold`` has none: no growth reaches it across the gap, and it is
    joined so to no other). A basin reaches far below the growth level, down
    a smear or tail hanging from one half, and its rows say nothing of the
    band's. A band's tops side by side (a
    dumbbell's ends, a hollow band's halves, joined or not) are one band so,
    whatever the band's slope; a band stacked above or below meets it along
    the lane's width, or shares few of its rows. Tops so in a chain are one
    band."""
    k = len(tops)
    if k == 1:
        return None, [0]
    markers = np.zeros(window.shape, np.int32)
    for i, (y, x) in enumerate(tops):
        markers[y, x] = i + 1
    mask = window > 0.0
    basin = watershed(-window, markers, connectivity=1, mask=mask)
    beside = np.zeros((k + 1, k + 1), np.int64)  # pixel pairs across two basins' valley
    stacked = np.zeros((k + 1, k + 1), np.int64)
    for a, b, pairs in (
        (basin[:, :-1], basin[:, 1:], beside),
        (basin[:-1, :], basin[1:, :], stacked),
    ):
        meet = (a != b) & (a > 0) & (b > 0)
        np.add.at(pairs, (np.minimum(a[meet], b[meet]), np.maximum(a[meet], b[meet])), 1)
    band = list(range(k))

    def root(i: int) -> int:
        while band[i] != i:
            i = band[i]
        return i

    def join(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        band[max(ra, rb)] = min(ra, rb)

    for a, b in zip(*np.nonzero((beside > 0) & (beside >= stacked)), strict=True):
        join(int(a) - 1, int(b) - 1)
    parts, _ = label(mask)  # 4-connected, as the basins
    pieces, _ = label(window > threshold)  # as grow_region and _join_burnt_out read them
    spans = find_objects(pieces)
    rows = [None if pieces[t] == 0 else spans[pieces[t] - 1][0] for t in tops]
    for a, b in itertools.combinations(range(k), 2):
        ra, rb = rows[a], rows[b]
        if (
            parts[tops[a]] != parts[tops[b]]
            and ra is not None
            and rb is not None
            and _share_rows((ra.start, ra.stop), (rb.start, rb.stop))
        ):
            join(a, b)
    return basin, [root(i) for i in range(k)]


def _beside_the_lane(ks: np.ndarray, y: int, x: int, a: int, b: int) -> bool:
    """Whether the peak ``(y, x)`` of a lane's seed range alone (its columns
    ``[a, b)``) lies beside the lane, not on it: on the range's first or last
    column (the flank of something the range's side cuts) with the range's
    middle column, in its row, under ``EXTENT_LEVEL`` of it, so that growth
    from it would not reach the lane's middle: a speck or stain beside the
    lane. A band's top cut by the range's side, the band over the lane's
    middle (a band wider than the range, or a neighbour's band over this
    lane's), is the lane's."""
    if x not in (a, b - 1):
        return False
    return float(ks[y, (a + b) // 2]) < EXTENT_LEVEL * float(ks[y, x])


def _share_rows(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether two things spanning the rows ``a`` and ``b`` (``[start, stop)``)
    lie side by side: they share ``JOIN_ROWS`` of the rows the two span, as
    the halves of a band split along x do. A band above or below another, and
    a line or stroke crossing a band's rows, do not."""
    return min(a[1], b[1]) - max(a[0], b[0]) >= JOIN_ROWS * (max(a[1], b[1]) - min(a[0], b[0]))


def _without_other_bands(
    window: np.ndarray, basin: np.ndarray | None, band: list[int], seed: int
) -> np.ndarray:
    """The lane's growth window ``window`` for a band grown from its separate
    peak ``seed`` (an index into the window's peaks, :func:`_band_basins`:
    their ``basin`` and ``band``), with every other band above or below it
    taken out (#58), whatever its height, saturated or not: the window itself
    when there is none.

    The pixels that drain to another band are set to 0: growth stops at the
    valley to it and keeps its threshold, so the band's extent on its other
    sides is as it would be alone, and the other band stays outside it, a
    band of its own. The seed's band's other tops (a dumbbell's other end, a
    hollow band's other half), however high, are its own."""
    mine = band[seed]
    others = [k + 1 for k, b in enumerate(band) if b != mine]
    if not others:
        return window
    return np.where(np.isin(basin, others), 0.0, window)


def _step(a: np.ndarray, b: np.ndarray) -> int:
    """How many rows the bands of a lane whose rows' highest signal is ``b``
    lie below those of one whose is ``a``: the shift of ``b`` that best
    matches ``a`` (the greatest sum of their products), of shifts as good the
    least. A row's smile or tilt moves every band of a lane alike, so the
    step holds for each of its rows."""
    corr = np.correlate(b, a, mode="full")
    lags = np.arange(1 - a.size, b.size)
    return int(lags[np.lexsort((np.abs(lags), -corr))[0]])


def _other_row(tops: list[tuple[float, bool]], y: float) -> bool:
    """Whether a lane's top nearest the row ``y`` is of another band than the
    one it grew from: ``tops`` are its tops' rows, each with whether it is of
    that band (which wins a tie)."""
    return not min(tops, key=lambda t: (abs(t[0] - y), not t[1]))[1]


def _join_burnt_out(
    pieces: np.ndarray,
    grown: np.ndarray,
    s: np.ndarray,
    saturated: np.ndarray,
    h: float,
    span: tuple[int, int],
) -> np.ndarray | None:
    """The band grown over ``grown`` joined across a burnt-out centre (#121), or
    None when nothing joins it. ``pieces`` labels the regions above the growth
    threshold in the lane's window, ``grown`` among them; ``s``, ``saturated``
    and ``h`` are as :func:`_hollow` takes them, over the window; ``span`` is
    the lane's x-range in it.

    A centre lighter through a saturated band's height, falling below the
    extent level, splits the band's grown extent along x: its halves grow
    apart, each saturated on its own side of the centre only, so that one of
    them would be boxed and the other counted as a second band. Another piece
    joins the band when it holds saturated pixels in the span; it shares
    ``JOIN_ROWS`` of the rows it and the grown band span, as the band's other
    half does (a line or stroke crossing the band's rows, as drawn on an
    image, and a band above or below it do not); and along the rows where
    both hold saturated pixels, the pixels between them hold ``HOLLOW_PIXELS``
    of a lighter centre by :func:`_hollow`'s rule, over the band, the piece
    and those pixels together, with ``HOLLOW_PIXELS`` saturated pixels. They
    are one hollow band, its extent over both halves and the centre. Two
    bands side by side not both saturated, and two stacked, stay apart."""
    lo, hi = span
    at_limit = np.unique(pieces[:, lo:hi][saturated[:, lo:hi]])
    own = int(pieces[grown][0])
    others = [int(k) for k in at_limit if k > 0 and k != own]
    rows = np.flatnonzero(grown.any(axis=1))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    band, joined = grown, False
    while others:
        for k in others:
            piece = pieces == k
            rows = np.flatnonzero(piece.any(axis=1))
            p0, p1 = int(rows[0]), int(rows[-1]) + 1
            if not _share_rows((y0, y1), (p0, p1)):
                continue
            between = _between(band & saturated, piece & saturated)
            if not between.any():
                continue
            both = band | piece | between
            inside = binary_fill_holes(both)
            sat = saturated & inside
            if np.count_nonzero(sat) < HOLLOW_PIXELS:
                continue
            if np.count_nonzero(_centre(inside, s, sat, h) & between) >= HOLLOW_PIXELS:
                band, joined = both, True
                others.remove(k)
                break
        else:
            break
    return band if joined else None


def _between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The pixels with a pixel of ``a`` on one side of them along their row and
    one of ``b`` on the other (those pixels included)."""

    def before(m: np.ndarray) -> np.ndarray:
        return np.logical_or.accumulate(m, axis=1)

    def after(m: np.ndarray) -> np.ndarray:
        return np.logical_or.accumulate(m[:, ::-1], axis=1)[:, ::-1]

    return (before(a) & after(b)) | (before(b) & after(a))


def _hill(ks: np.ndarray, comps: np.ndarray, top: tuple[int, int]) -> np.ndarray | None:
    """The peak ``top``'s own part of ``ks`` (>= 0): the 4-connected region above
    its saddle to higher ground that holds it (all of its component when
    nothing higher is joined to it), or None if it is no peak of its own.
    ``comps`` labels the components of ``ks > 0``. The saddle is the highest
    level at which a path joins it to a higher pixel: reconstruction by
    dilation of the higher pixels under ``ks``, read at the peak, within the
    peak's component (nothing outside it joins anything)."""
    comp = comps == comps[top]
    rows, cols = np.flatnonzero(comp.any(axis=1)), np.flatnonzero(comp.any(axis=0))
    y0, x0 = int(rows[0]), int(cols[0])
    sub = np.where(comp, ks, 0.0)[y0 : int(rows[-1]) + 1, x0 : int(cols[-1]) + 1]
    y, x = top[0] - y0, top[1] - x0
    height = float(sub[y, x])
    higher = np.where(sub > height, sub, 0.0)
    saddle = 0.0
    if higher.any():
        saddle = float(reconstruction(higher, sub, method="dilation", footprint=_CROSS)[y, x])
    if saddle >= height:
        return None
    own, _ = label(sub > saddle)
    hill = np.zeros(ks.shape, bool)
    hill[y0 : y0 + sub.shape[0], x0 : x0 + sub.shape[1]] = own == own[y, x]
    return hill


def _row_offset(column: np.ndarray, row: int) -> float:
    """How far a peak at ``row`` of ``column`` lies from that row's centre: the
    vertex of the parabola through it and its neighbours, at most half a row
    either way; 0 at the column's ends or where the three are not a peak."""
    if row <= 0 or row >= column.size - 1:
        return 0.0
    below, at, above = float(column[row - 1]), float(column[row]), float(column[row + 1])
    curvature = below - 2.0 * at + above
    if curvature >= 0.0:
        return 0.0
    return min(0.5, max(-0.5, 0.5 * (below - above) / curvature))


def _one_top_per_band(
    ks: np.ndarray, tops: list[tuple[tuple[int, int], slice]]
) -> set[tuple[int, int]]:
    """Of the tops of other bands in a lane (each with its hill's rows,
    :func:`_hill`), one per band (#58): a top lies beside a higher one, in
    the same band, when that one's row is among its hill's rows. A hill stops
    at the saddle to higher ground, so it reaches the row of a higher top
    beside it (a dumbbell's other end, or a hollow band's other half, joined
    or not) but not that of a band stacked above or below it."""
    kept: list[tuple[int, int]] = []
    for top, rows in sorted(tops, key=lambda t: (-float(ks[t[0]]), t[0])):
        if not any(rows.start <= higher[0] < rows.stop for higher in kept):
            kept.append(top)
    return set(kept)


def _count_components(res: _Pass, saturated: np.ndarray | None) -> None:
    """Each measured lane's components and peaks (:class:`Peak`), from the
    :func:`_peaks` of the kept signal at ``DETECT_K`` sigma (with a preferred
    row, the pass's ``tops``: :func:`_clipped_peaks` too), and whether its
    band is hollow (#121, :func:`_hollow`). Run once, on the pass that gives
    the result.

    A peak inside the band's grown extent is the band's own: a dip between two
    of them (a dumbbell, a hollow centre) leaves one band. Any other peak in
    the lane's span is a second component only if it reaches ``SECOND_SHARE``
    of the lane's peak (the band's highest pixel, or a higher peak in the
    span); its hill (:func:`_hill`) overlaps the band's rows, those the
    band's kept component (its signal above ``NOISE_K`` sigma) fills in the
    span, as a doublet's band joined to it does, or a band's half beside it;
    and the hill passes the candidates' width rule (:func:`_dust`).
    Every peak in the span is listed; one that passes all but the band's rows
    reads as another band (``other_band``), since a band apart above or below
    is one, the highest of its tops side by side only (#58). Each second
    component's peak is kept (``second``: its continuous row, refined as a
    :class:`Peak`'s, and column), for the note that says where it lies.
    ``saturated``: the crop's pixels at the saturation level, or None where it
    is unknown (no band is then hollow)."""
    sig = res.sig
    measured = [
        (ln, ln.span, ln.grown)
        for ln in res.lanes
        if ln.present and ln.span is not None and ln.grown is not None
    ]
    if not measured:
        return
    ks = np.where(res.cand.kept, sig.s_sm, 0.0)
    tops = res.tops if res.tops is not None else _peaks(ks, DETECT_K * sig.sigma_sm)
    comps, _ = label(ks > 0.0)
    thr = NOISE_K * sig.sigma_sm
    min_w = _min_width(ks.shape[1], len(res.lanes))
    for ln, (lo, hi), grown in measured:
        spanned = [p for p in tops if lo <= p[1] < hi]
        others = [p for p in spanned if not grown[p]]
        peak = max([float(ks[grown].max()), *(float(ks[p]) for p in others)])
        own = np.unique(comps[grown])  # a burnt-out centre it holds may lie below the noise
        band = np.isin(comps[:, lo:hi], own[own > 0]).any(axis=1)
        second: list[tuple[int, int]] = []
        passed: list[tuple[tuple[int, int], slice]] = []  # each top and its hill's rows
        for p in others:
            if ks[p] < SECOND_SHARE * peak:
                continue
            hill = _hill(ks, comps, p)
            if hill is None:
                continue  # no peak of its own
            [region] = find_objects(hill.astype(np.int8))
            if _dust(hill[region], sig.s_ds[region], float(ks[p]), thr, min_w):
                continue
            passed.append((p, region[0]))
            if (band & hill.any(axis=1)).any():  # not above or below the band
                second.append(p)
        ln.components = 1 + len(second)
        ln.second = tuple(
            (y + 0.5 + _row_offset(sig.s_sm[:, x], y), x + 0.5) for y, x in sorted(second)
        )
        other_bands = _one_top_per_band(ks, passed)
        ln.peaks = tuple(
            (
                y + 0.5 + _row_offset(sig.s_sm[:, x], y),
                x + 0.5,
                float(ks[y, x]) / sig.sigma_sm,
                bool(grown[y, x]),
                (y, x) in other_bands,
            )
            for y, x in sorted(spanned)
        )
        ln.hollow = saturated is not None and _hollow(
            grown, sig.s_sm, saturated, DETECT_K * sig.sigma_sm
        )


def _flanks(v: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """The highest value of ``v`` before each pixel along ``axis``, and the
    highest after it; -inf where there is none."""
    w = np.moveaxis(v, axis, -1)
    before = np.full(w.shape, -np.inf)
    after = np.full(w.shape, -np.inf)
    before[..., 1:] = np.maximum.accumulate(w, axis=-1)[..., :-1]
    after[..., :-1] = np.maximum.accumulate(w[..., ::-1], axis=-1)[..., ::-1][..., 1:]
    return np.moveaxis(before, -1, axis), np.moveaxis(after, -1, axis)


def _hollow(grown: np.ndarray, s: np.ndarray, saturated: np.ndarray, h: float) -> bool:
    """Whether the band grown over ``grown`` is hollow (#121): burnt out in its
    middle, lighter there than the ``saturated`` pixels on either side of it.

    Over the grown extent with its holes filled (a ring's centre may fall below
    the extent level): at least ``HOLLOW_PIXELS`` saturated pixels, and
    ``HOLLOW_PIXELS`` of a lighter centre. A pixel is one when, in the smoothed
    signal ``s``, it lies well below the saturated pixels on both sides of it
    along its row: ``h`` below the lower of the two sides' highest, and at
    most ``VALLEY_FRAC`` of it (the rule that separates two peaks,
    :func:`_peaks`); and when, along its column, it lies so far below the band
    on both sides (a ring) or on neither (a centre lighter through the band's
    height, splitting it along x), not on one side only: that is a notch in the
    band's top or bottom edge, the band's body on its other side, as where a
    band's ends curve up. Two bands stacked, a close doublet with one or both
    saturated, hold no saturated pixel on either side of the rows between
    them, whatever one grown extent holds: the note the flag gives, a centre
    lighter than the saturated pixels on either side of it, holds."""
    [region] = find_objects(grown.astype(np.int8))
    inside = binary_fill_holes(grown[region])
    sat = saturated[region] & inside
    if np.count_nonzero(sat) < HOLLOW_PIXELS:
        return False
    return int(np.count_nonzero(_centre(inside, s[region], sat, h))) >= HOLLOW_PIXELS


def _centre(inside: np.ndarray, s: np.ndarray, sat: np.ndarray, h: float) -> np.ndarray:
    """The lighter centre of the band over ``inside`` (its holes filled), whose
    saturated pixels are ``sat``: the pixels :func:`_hollow` counts."""

    def below(level: np.ndarray) -> np.ndarray:
        return s <= np.minimum(level - h, VALLEY_FRAC * level)

    left, right = _flanks(np.where(sat, s, -np.inf), axis=1)
    above, under = _flanks(np.where(inside, s, -np.inf), axis=0)
    return inside & ~sat & below(np.minimum(left, right)) & (below(above) == below(under))


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
    pieces: list[_Piece]  # the pieces the reading assigns (_reduce's), in order
    joined: list[_Joined]  # the pairs _reduce merged or dropped one of
    # The kept signal's peaks (_peaks, and _clipped_peaks where the saturation
    # level is known), found once where a preferred row seeds the lanes and
    # reused to count the components; None: not found yet.
    tops: list[tuple[int, int]] | None = None


def _run_pass(
    sig: _Signal,
    n: int,
    x_offset: int,
    image_rows: tuple[bool, bool],
    saturated: np.ndarray | None,
    prefer: float | None = None,
) -> _Pass:
    """Steps 3 to 6 of the pipeline on one signal; ``image_rows``: whether the
    box's top and bottom rows are the image's (:func:`_lines`); ``saturated``:
    the crop's pixels at the saturation level, or None (:func:`_measure`);
    ``prefer``: the row the lanes are seeded nearest, in the crop's rows, or
    None (:func:`_measure`)."""
    wc = sig.s_sm.shape[1]
    notes: list[str] = []
    cand = _candidates(sig, n, image_rows)
    pieces, joined = _reduce(cand.pieces, n, x_offset, notes)
    assign, alt = (None, math.inf) if not pieces else _assign(pieces, n, float(wc))
    if assign is None:
        return _Pass(sig, cand, None, math.inf, [_Lane() for _ in range(n)], notes, pieces, joined)
    lanes = _lanes_from(assign, pieces, n)
    if prefer is None:
        _measure(lanes, sig.s_sm, cand.kept, sig.sigma_sm, saturated)
        return _Pass(sig, cand, assign, alt, lanes, notes, pieces, joined)
    ks = np.where(cand.kept, sig.s_sm, 0.0)
    h = DETECT_K * sig.sigma_sm
    tops = _peaks(ks, h)
    if saturated is not None:
        tops += _clipped_peaks(sig.s_sm, cand.kept, h, saturated, tops)
    _measure(lanes, sig.s_sm, cand.kept, sig.sigma_sm, saturated, tops, prefer)
    return _Pass(sig, cand, assign, alt, lanes, notes, pieces, joined, tops)


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
    surfaces = _slot_surfaces(res)
    s_cut = np.where(cand.cut, sig.s_sm, 0.0)
    for ln in lanes:
        if ln.present:
            continue
        a = int(max(0, math.floor(ln.centre - EMPTY_WINDOW * pitch)))
        b = int(min(wc, math.ceil(ln.centre + EMPTY_WINDOW * pitch)))
        if b <= a:
            continue  # the slot lies outside the row box: not measured
        ln.window = (a, lo, b, hi)
        ln.snr = float(surfaces.row[:, a:b].max()) / sig.sigma_sm
        ln.reason = _slot_reason(res, surfaces, ln.snr, (slice(0, None), slice(a, b)))
        if ln.reason == "edge_signal":
            ln.cut = float(s_cut[:, a:b].max()) / sig.sigma_sm >= DETECT_K


@dataclass
class _Placed:
    """Steps 8 and 9 of :func:`detect_row` for some of its lanes (its
    ``place``): the lanes whose band the row box cuts, the shared size, the
    boxes by lane, the lanes whose extent the size left out, each box's
    offset from the row's line, and the flags and notes of these steps."""

    cut: set[int] = field(default_factory=set)
    size: BoxSize | None = None
    out: dict[int, Rect] = field(default_factory=dict)
    outliers: list[int] = field(default_factory=list)
    offsets: dict[int, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class _Surfaces(NamedTuple):
    """The signal an empty lane's slot is read on (:func:`_slot_surfaces`)."""

    row: np.ndarray  # the row's rows, the rejected candidates left out: the SNR
    all: np.ndarray  # everything but dust, lines and side signal: edge_signal
    line: np.ndarray  # lines and strips across the lanes: line
    side: np.ndarray  # signal rising into the box's sides: side_signal


def _slot_surfaces(res: _Pass) -> _Surfaces:
    """The surfaces :func:`_empty_lanes` reads an empty lane's slot on, and
    :func:`_on_the_line` the slot of a lane off the row's line: the smoothed
    signal in the row's rows (a neighbouring row left out) without the
    candidates the detector rejected (dust, a peak on the box's edge rows, a
    streak or stain, a line, signal rising into the box's side); the signal
    everywhere but dust, lines and side signal; the lines and strips; and the
    side signal."""
    sig, cand = res.sig, res.cand
    lo, hi = cand.rows
    row = np.zeros_like(sig.s_sm)
    row[lo:hi] = sig.s_sm[lo:hi]
    row[cand.dropped | cand.lines | cand.side] = 0.0
    return _Surfaces(
        row=row,
        all=np.where(cand.dust | cand.lines | cand.side, 0.0, sig.s_sm),
        line=np.where(cand.lines, sig.s_sm, 0.0),
        side=np.where(cand.side, sig.s_sm, 0.0),
    )


def _slot_reason(
    res: _Pass, surfaces: _Surfaces, snr: float, region: tuple[slice, slice]
) -> LaneReason:
    """Why a slot ``region`` (rows, columns of the crop) whose SNR is ``snr``
    leaves its lane empty, in :func:`_empty_lanes`' order: a rejected streak
    or stain over its columns (and rows), ``artefact``; a kept candidate
    reaching ``DETECT_K``, ``unassigned``; else a line or strip reaching it,
    ``line``; signal rising into the box's side, ``side_signal``; any other
    signal, ``edge_signal`` (the rows left out, a peak on an edge row); else
    ``no_band``."""
    sig = res.sig
    rows, cols = region
    top, bot, _ = rows.indices(sig.s_sm.shape[0])
    a, b, _ = cols.indices(sig.s_sm.shape[1])

    def reaches(surface: np.ndarray) -> bool:
        return float(surface[region].max()) / sig.sigma_sm >= DETECT_K

    if any(
        reason == "artefact"
        and where[1].start < b
        and a < where[1].stop
        and where[0].start < bot
        and top < where[0].stop
        for reason, where in res.cand.rejected
    ):
        return "artefact"
    if snr >= DETECT_K:
        return "unassigned"
    if reaches(surfaces.line):
        return "line"
    if reaches(surfaces.side):
        return "side_signal"
    if reaches(surfaces.all):
        return "edge_signal"
    return "no_band"  # the snr tells how close it came


def _centre_row(column: np.ndarray) -> float | None:
    """The row, in index units, where a band's column profile ``column``
    peaks: the middle of its top plateau, else its highest pixel moved to the
    vertex of the parabola through it and its neighbours (:func:`_row_offset`);
    None for a column with nothing above 0 (outside the band)."""
    top = float(column.max())
    if top <= 0.0:
        return None
    flat = np.flatnonzero(column >= top - 1e-9 * top)
    if flat.size > 1:
        return 0.5 * float(flat[0] + flat[-1])
    k = int(flat[0])
    return k + _row_offset(column, k)


def _read_at(column: np.ndarray, y: float) -> float:
    """``column`` at the continuous index ``y``, linearly interpolated; 0
    outside it."""
    if not 0.0 <= y <= column.size - 1:
        return 0.0
    return float(np.interp(y, np.arange(column.size, dtype=float), column))


def _on_the_line(
    res: _Pass,
    surfaces: _Surfaces,
    own: np.ndarray,
    seed_row: int,
    centre: tuple[float, float],
    line: float,
    pitch: float,
) -> tuple[str, float, Rect | None]:
    """What the slot of a lane off the row's line holds on the line (#58):
    ``(reason, snr, window)``. ``own`` marks the pixels that drain to
    the band the lane grew from (:class:`_Lane`), and ``seed_row`` is the
    row it grew from; ``centre`` is its box's centre (x, y) and ``line`` the
    row's line at it (continuous crop coordinates), ``pitch`` the lanes'.

    Read as :func:`_empty_lanes` reads an empty lane's slot, on its surfaces
    (:func:`_slot_surfaces`), over the columns within ``EMPTY_WINDOW`` pitch
    of the box's centre. In each column, the signal on the line, less the
    lane's own band's signal on the far side of that band's centre row as far
    from it (:func:`_centre_row`, on the seed's side of the row midway
    between it and the line: the own band's signal may hold, joined to it,
    a band on the line with no peak of its own), but never more than the own
    band's signal on the line: a band is about as deep at a distance above
    its centre as below it, so that leaves what on the line is not its tail,
    and nothing more than it holds there. The SNR is the most any column
    leaves, over the noise. The reason is the one :func:`_slot_reason` gives
    with that SNR, over the rows the reading read (the line's row and the one
    below it, each smoothed over a row either side: four rows, the window),
    with the lane's own band left out of every surface. ``outside`` with no
    window: the line runs outside the row box's rows there, and nothing on
    it was read."""
    sig = res.sig
    hc, wc = sig.s_sm.shape
    p = line - 0.5  # the line's row in index units
    if not 0.0 <= p <= hc - 1:
        return "outside", 0.0, None
    a = int(max(0, math.floor(centre[0] - EMPTY_WINDOW * pitch)))
    b = int(min(wc, math.ceil(centre[0] + EMPTY_WINDOW * pitch)))
    mine = np.where(own, surfaces.row, 0.0)
    midway = 0.5 * (seed_row + p)
    rows = np.arange(hc)
    seed_side = rows < midway if p > seed_row else rows > midway
    centred = np.where(seed_side[:, None], mine, 0.0)
    best = -math.inf
    for x in range(a, b):
        value = _read_at(surfaces.row[:, x], p)
        middle = _centre_row(centred[:, x])
        if middle is not None:  # less the own band's tail: at most its own signal there
            value -= min(_read_at(mine[:, x], 2.0 * middle - p), _read_at(mine[:, x], p))
        best = max(best, value)
    snr = best / sig.sigma_sm
    top, bot = max(0, math.floor(p) - 1), min(hc, math.floor(p) + 3)
    others = _Surfaces(*(np.where(own, 0.0, surface) for surface in surfaces))
    reason = _slot_reason(res, others, snr, (slice(top, bot), slice(a, b)))
    return reason, snr, (a, top, b, bot)


def _not_detected(
    offsets: Mapping[int, float], expected: Mapping[int, float], reasons: Mapping[int, str]
) -> tuple[OffCause | None, list[int]]:
    """Whether the lanes whose boxes lie off the row's line (#58) are
    recorded as not detected at the expected row: ``(None, those lanes)``,
    or the first condition that fails, as :data:`OffCause` names it, and the
    lanes it names. ``offsets`` holds each box's offset from the row's line
    and ``expected`` from the expected row, both in box heights, by lane;
    ``reasons`` each off lane's reading on the line (:func:`_on_the_line`).

    In order: fewer than half of the boxes lie off the line (a row whose line
    a few lanes leave, not a line between two rows; the boxes, not the
    declared lanes, place the line); each of them lies off the expected row
    too, by more than ``ROW_LINE_K`` box heights (a box on it, off the line,
    says the line runs through another row); nothing on the line in its slot
    reaches ``DETECT_K``, nor is a streak, line or other signal there
    (``no_band``: the protein is not there, rather than a band the lane's
    box missed); and every other box lies on the expected row (a record
    says the line is the expected row's)."""
    off = sorted(i for i, v in offsets.items() if abs(v) > ROW_LINE_K)
    if 2 * len(off) >= len(offsets):
        return "half", off
    named = [i for i in off if abs(expected[i]) <= ROW_LINE_K]
    if named:
        return "expected_row", named
    named = [i for i in off if reasons[i] != "no_band"]
    if named:
        return "line_signal", named
    named = sorted(i for i, v in expected.items() if i not in off and abs(v) > ROW_LINE_K)
    if named:
        return "other_box", named
    return None, off


def _phrase(values: Sequence[str]) -> str:
    """``values`` in words: "a", "a and b", "a, b and c"."""
    return values[0] if len(values) == 1 else f"{', '.join(values[:-1])} and {values[-1]}"


def shown_against(value: float, limit: float, places: int) -> str:
    """``value`` to ``places`` decimals, for words that print it with the
    ``limit`` it was compared with (#58): to more decimals where fewer would
    round it onto the limit or past it, so the number shown lies on the side
    of the limit the value does (0.7504 against 0.75: "0.7504", not "0.75";
    5.96 against 6: "5.96", not "6.0")."""
    text = f"{value:.{places}f}"
    while (float(text) > limit, float(text) < limit) != (value > limit, value < limit):
        places += 1
        text = f"{value:.{places}f}"
    return text


def _other_bands_note(
    seconds: Mapping[int, tuple[Rect, Sequence[tuple[float, float]]]], x0: int, y0: int
) -> str:
    """The ``multiple_components`` note with ``prefer_y`` (#58): where each
    lane's second components lie, from its box's centre. ``seconds`` maps a
    lane, as numbered, to its box (image coordinates) and its second
    components' peaks (continuous crop y, x: :class:`_Lane`), the crop at
    ``(x0, y0)``.

    A peak above or below the box's rows: "another band lies N px below the
    box centre; the box does not include it"; in the box: "..., inside the
    box: the box holds part of it"; in the box's rows but beside it: "another
    band lies beside the box, N px below its centre; ...". N is the peak's
    distance from the box's centre row, to the nearest px. The lanes of one
    kind and way share a clause, N then a range ("12 to 14 px"); clauses in
    that order of kinds, above before below, joined by "; "."""
    groups: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
    for lane, ((bx0, by0, bx1, by1), peaks) in seconds.items():
        for py, px in peaks:
            y, x = y0 + py, x0 + px
            offset = y - (by0 + by1) / 2
            if not by0 <= y < by1:
                kind = 0  # above or below the box
            elif bx0 <= x < bx1:
                kind = 1  # inside it
            else:
                kind = 2  # beside it
            lanes, distances = groups.setdefault((kind, int(offset > 0)), ([], []))
            lanes.append(lane)
            distances.append(math.floor(abs(offset) + 0.5))
    clauses = []
    for (kind, below), (lanes, distances) in sorted(groups.items()):
        lo, hi = min(distances), max(distances)
        px = f"{lo} px" if lo == hi else f"{lo} to {hi} px"
        way = "below" if below else "above"
        named = lanes_phrase(sorted(set(lanes)))
        if kind == 0:
            clauses.append(
                f"{named}: another band lies {px} {way} the box centre; the box does not include it"
            )
        elif kind == 1:
            clauses.append(
                f"{named}: another band lies {px} {way} the box centre, inside the box: the box"
                " holds part of it"
            )
        else:
            clauses.append(
                f"{named}: another band lies beside the box, {px} {way} its centre; the box"
                " does not include it"
            )
    return "; ".join(clauses)


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


def _doubts(res: _Pass, n: int, wc: int, x_offset: int, number: Callable[[int], int]) -> list[str]:
    """How the lane reading of ``res`` does not fit the bands' own spacing, in
    words for the ``doubtful_lanes`` note; empty when it fits. ``n`` lanes
    over a box ``wc`` px wide; ``number(k)`` is the number the caller knows
    lane ``k`` (crop order) by; x is given in the image (``x_offset``).

    The bands' own spacing is the repeated median of the slopes between the
    resolved bands, their pieces' centres over their lanes (each band's median
    slope to the others, then the median of those: a band off the others, as
    a neighbouring panel's read as the last lane, moves it little, where the
    median of every pair's slope follows it among four bands). A resolved band
    is a piece the reading gives one lane, not rising into the box's side.
    From two of them on, the reading does not fit that spacing when:

    * the fitted pitch lies more than ``PITCH_DOUBT`` of it off it;
    * the first and last resolved bands lie more than ``LANES_DOUBT`` lanes
      off the lanes read between them;
    * a touching run cut into cells spans more than ``LANES_DOUBT`` lanes more
      than its cells (its width over the spacing, as if its bands filled
      their lanes), or fewer (its width less the narrowest resolved band's
      over the spacing, plus one, as if its bands were that narrow);
    * the box reaches more than ``END_DOUBT`` spacings past an end lane's
      centre, stepped from the resolved bands: room for another lane there.

    And over that spacing, or the fitted pitch with fewer than two resolved
    bands, when ``_reduce`` merged a pair of pieces, or dropped the weaker
    between the pieces read, to fit the declared lanes, their centres
    ``APART_DOUBT`` lanes apart or more: two bands, not one band's halves or
    dust beside a band. A piece dropped past the pieces read shifts none of
    their lanes (the reading is theirs without it, which the tests above
    judge): dust in a loose box's margin, 0.7 to 0.9 lanes past the end band,
    is no doubt.

    A row box that also covers a ladder, labels or a neighbouring panel, read
    with no lanes on the image to check it, holds more objects than lanes, or
    room for more, and the pitch fitted to the box and its ends misses the
    bands' spacing (#111). Measured on the bench, the accuracy judge's rows,
    the recipes and the fuzz rows (611 rows read right) and on real drags, the
    honest rows stay within: the pitch 19% off the spacing (a loose box,
    margins of a pitch each side; 19% on a real row), the resolved bands 0.31
    lanes off (+/-35% spacing; real 0.24), runs 0.33 lanes more than their
    cells (bands twice as wide as the others; real ones none) and 0.24 fewer,
    the box 1.35 spacings past an end lane (real 1.09), and merged or dropped
    pieces 0.51 lanes apart (a band's halves split by a bubble, 0.33 to 0.40;
    dust midway between lanes, 0.45 to 0.51). The real drags read a lane or
    more off: a box over a side panel, the pitch 65% off and the box 2.1
    spacings past lane 1; over a ladder, the bands 0.9 lanes off and a run
    0.59 more; past a montage panel's frame on both sides, runs 0.61 and 0.77
    more and the box 1.9 spacings past lane 1. Rows declared with fewer lanes
    than bands merge pieces 0.64 lanes apart or more, or drop a weak one
    between the others. Not caught: a box over an arrow just past the last
    band, read as the last lane at the bands' own spacing (the reading fits
    it); a box over part of a row of touching bands, which leaves no resolved
    band to measure; and a weak band past the end one, dropped, the declared
    lanes read in order from the other end (2 of 93 fuzz rows declared with
    fewer lanes than bands: a sliver of a band the box's side cuts)."""
    assign = res.assign
    if assign is None:
        return []

    def lanes(a: int, b: int) -> str:
        lo, hi = sorted((number(a), number(b)))
        return f"lanes {lo} to {hi}"

    resolved = [
        (k, p) for p, (k, q) in zip(res.pieces, assign.lanes, strict=True) if q == 1 and not p.side
    ]
    spacing = None
    if len(resolved) >= 2:  # each band's median slope to the others, and their median
        spacing = float(
            np.median(
                [
                    np.median([(pb.c - pa.c) / (kb - ka) for kb, pb in resolved if kb != ka])
                    for ka, pa in resolved
                ]
            )
        )
    words: list[str] = []
    if spacing is not None:  # the pieces run in lane order: it is positive
        own = f"the bands' own spacing ({spacing:.1f} px)"
        at_own: list[str] = []
        off = assign.pitch / spacing - 1.0
        if abs(off) > PITCH_DOUBT:
            words.append(f"the fitted pitch ({assign.pitch:.1f} px) is {abs(off):.0%} off {own}")
        (k0, p0), (k1, p1) = resolved[0], resolved[-1]
        span = (p1.c - p0.c) / spacing
        if abs(span - (k1 - k0)) > LANES_DOUBT:
            at_own.append(f"the bands read as {lanes(k0, k1)} lie {span:.1f} lanes apart")
        narrowest = min(p.w for _, p in resolved)
        for p, (k, q) in zip(res.pieces, assign.lanes, strict=True):
            if q == 1 or p.side:
                continue
            # At most as many lanes as its cells' spacing gives, at least as
            # many as bands as narrow as the narrowest resolved one leave room for.
            most, least = p.w / spacing, (p.w - narrowest) / spacing + 1.0
            if most - q > LANES_DOUBT or q - least > LANES_DOUBT:
                cells = most if most - q > LANES_DOUBT else least
                at_own.append(
                    f"the touching bands read as {lanes(k, k + q - 1)} span {cells:.1f} lanes"
                )
        for end, room in ((0, p0.c / spacing - k0), (n - 1, (wc - p1.c) / spacing - (n - 1 - k1))):
            if room > END_DOUBT:
                at_own.append(
                    f"the row box reaches {room:.1f} lanes past lane {number(end)}'s centre"
                )
        if at_own:
            at_own[0] = ("at that spacing " if words else f"at {own}, ") + at_own[0]
            words.extend(at_own)
    step = assign.pitch if spacing is None else spacing
    first, last = res.pieces[0].l, res.pieces[-1].r  # the pieces read (an assignment has some)
    for joined in res.joined:
        if joined.dropped and (joined.r <= first or joined.l >= last):
            continue  # dropped past them: their lanes are read as without it
        apart = joined.apart / step
        if apart >= APART_DOUBT:
            done = "one dropped" if joined.dropped else "merged"
            words.append(
                f"two pieces {apart:.1f} lanes apart, {done} at"
                f" x={x_offset + joined.l:.0f}..{x_offset + joined.r:.0f} to fit the declared lanes"
            )
    return words


def detect_row(
    gray: np.ndarray,
    row: Sequence[int],
    n_lanes: int,
    *,
    background: float,
    dark_on_light: bool = True,
    size_rule: str = SIZE_RULE,
    right_to_left: bool = False,
    saturated_at: float | None = None,
    prefer_y: float | None = None,
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
    ``saturated_at`` is the pixel value from which on a pixel is saturated: at
    or below it on a dark-on-light image, at or above it on a light-on-dark
    one (the detector limit, or near it where compression or colour moved
    saturated pixels off it); None where the image has no known limit, and no
    band is then called hollow (``hollow_band``).
    ``prefer_y`` is the row, in the image's continuous rows (row r covers [r,
    r+1)), where the caller expects the band (#58: its expected MW's): each
    lane then grows from the peak in its seed range nearest it (the crop's
    peaks there and the range's own: :func:`_measure`), not from its
    strongest pixel, in both stages, so a stronger band nearby in the lane is
    not boxed, nor taken for the membrane; ``snr`` is that peak's. Its growth
    stops at the valley to any other band above or below it, stronger or
    weaker, saturated or not, which stays another band (a second component,
    ``multiple_components``, or a band apart in ``peaks``): the box is the
    band's own, never over both; with ``saturated_at``, two clipped bands
    whose clipped pixels lie apart are two peaks across a valley of
    ``DETECT_K`` sigma, however near the valley comes to their clipped height
    (:func:`_clipped_peaks`), but two whose clipped pixels touch are one.
    Lanes that grew so from bands on two rows refuse the row
    (``off_row_line``, ``crossed``). Lanes whose boxes lie off the row's line
    are recorded as not detected where :func:`_not_detected` says so
    (``off_expected_row``): each read on the line (:func:`_on_the_line`),
    the row placed again without them; a box still off the line then
    refuses it (``off_cause`` ``again``, the first placement reported).
    With None, the default, the result is as without it, bit for bit
    (``crossed`` empty, ``off_cause`` None).
    Raises :class:`RowDetectError` for a row it cannot use, and ValueError for
    an unknown ``size_rule``, or a ``saturated_at`` or ``prefer_y`` that is not
    a finite number.
    """
    if size_rule not in SIZE_RULES:
        raise ValueError(f"unknown size rule {size_rule!r}; expected one of {SIZE_RULES}")
    if saturated_at is not None and not (
        isinstance(saturated_at, numbers.Real) and math.isfinite(saturated_at)
    ):
        raise ValueError(f"saturated_at must be a finite number or None, not {saturated_at!r}")
    if prefer_y is not None and not (
        isinstance(prefer_y, numbers.Real)
        and not isinstance(prefer_y, bool)
        and math.isfinite(prefer_y)
    ):
        raise ValueError(f"prefer_y must be a finite number or None, not {prefer_y!r}")
    gray = np.asarray(gray)
    x0, y0, x1, y1 = _check_row(gray, row, n_lanes, background)
    prefer = None if prefer_y is None else float(prefer_y) - y0
    n = int(n_lanes)
    crop = np.asarray(gray[y0:y1, x0:x1], dtype=np.float64)
    if not np.isfinite(crop).all():
        raise RowDetectError("invalid_image", "the row holds non-finite pixel values")
    hc, wc = crop.shape
    sign = 1.0 if dark_on_light else -1.0
    floor = _noise_floor(crop)
    sigma_px = max(_pixel_noise(crop), floor)
    saturated = None
    if saturated_at is not None:
        saturated = crop <= saturated_at if dark_on_light else crop >= saturated_at

    # Stage 1: a plane over the crop, or the stored background where the plane
    # is off the membrane.
    flat = np.full(crop.shape, float(background))
    plane = flat
    coef = _fit_plane(crop, np.ones(crop.shape, bool), floor)
    if coef is not None:
        plane = _surface(crop, _subsample(np.arange(crop.size)), coef, flat, sign, sigma_px)
    crop_ds = _despeckle(crop, _despeckle_width(wc, n))
    image_rows = (y0 == 0, y1 == gray.shape[0])
    res = _run_pass(
        _signal(crop, crop_ds, plane, sign, sigma_px, floor, None),
        n,
        x0,
        image_rows,
        saturated,
        prefer,
    )

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
            _signal(crop, crop_ds, plane, sign, sigma_px, floor, free),
            n,
            x0,
            image_rows,
            saturated,
            prefer,
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
    _count_components(res, saturated)
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

    def place(keep: list[int]) -> _Placed:
        """Steps 8 and 9 for the lanes ``keep`` (of ``present``, in order):
        the bands the row box cuts, one shared size, the boxes placed, and the
        row's line through them, with the flags and notes of these steps."""
        placed = _Placed()
        # A band the row box cuts through: its extent reaches the box's top or
        # bottom edge, and that edge row still holds CUT_LEVEL of its peak
        # there; or, in an empty lane, it peaks on that edge row (_empty_lanes).
        placed.cut = {
            i
            for i in keep
            if any(
                float(sig.s_sm[y, extents[i][0] : extents[i][2]].mean())
                >= CUT_LEVEL * lanes[i].peak
                for y, touches in ((0, extents[i][1] <= 0), (hc - 1, extents[i][3] >= hc))
                if touches
            )
        } | {i for i, ln in enumerate(lanes) if ln.cut and i not in extents}
        if not keep:
            return placed
        # One shared size; bounded isotonic placement inside the row.
        rects = [extents[i] for i in keep]
        w, h, outliers = _shared_size(
            [r[2] - r[0] for r in rects],
            [r[3] - r[1] for r in rects],
            size_rule,
            cut_w=[r[0] <= 0 or r[2] >= wc for r in rects],
            cut_h=[r[1] <= 0 or r[3] >= hc for r in rects],
        )
        placed.outliers = [keep[k] for k in outliers]
        if len(rects) > 1:  # no wider than the closest pair of extent centres
            gaps = np.diff([0.5 * (r[0] + r[2]) for r in rects])
            min_w = _min_width(wc, n)
            crowded = np.flatnonzero(gaps < min_w)
            if crowded.size:  # one centre, crossed, or closer than a band: not two lanes
                placed.flags.append("ambiguous_lanes")
                pairs = sorted({keep[k] for k in crowded} | {keep[k + 1] for k in crowded})
                placed.notes.append(
                    f"{lanes_phrase(numbered(pairs))}: extents closer than the narrowest band "
                    f"({min_w:.0f} px) or out of order"
                )
            w = min(w, int(math.floor(max(float(gaps.min()), min_w))))
        w = max(MIN_BOX, min(w, wc // len(rects)))  # len * w fits: MIN_BOX * n <= wc
        h = max(MIN_BOX, min(h, hc))
        xs = place_in_row([(r[0] + r[2]) // 2 - w // 2 for r in rects], w, 0, wc)
        for i, r, x in zip(keep, rects, xs, strict=True):
            y = max(0, min((r[1] + r[3]) // 2 - h // 2, hc - h))
            placed.out[i] = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
        placed.size = BoxSize(width=w, height=h)
        # The row's line through the boxes' centres (crop coordinates, so a
        # shifted row gives the same offsets): a box off it is off the row.
        line = _row_line(
            [((r[0] + r[2]) / 2 - x0, (r[1] + r[3]) / 2 - y0) for r in placed.out.values()],
            h,
            list(placed.out),
        )
        if line is not None:
            placed.offsets = {i: float(v) for i, v in zip(placed.out, line, strict=True)}
        off_line = [i for i, v in placed.offsets.items() if abs(v) > ROW_LINE_K]
        if off_line:
            placed.flags.append("off_row_line")
            if len(off_line) == len(placed.out):  # two rows, neither the row's
                placed.notes.append(
                    f"{lanes_phrase(numbered(off_line))}: box centres more than {ROW_SMILE:g}"
                    f" box heights ({ROW_SMILE * h:.0f} px) apart, on two rows"
                )
            else:
                placed.notes.append(
                    f"{lanes_phrase(numbered(off_line))}: box centre more than {ROW_LINE_K:g}"
                    f" box height ({ROW_LINE_K * h:.0f} px) off the row's line through the"
                    " other boxes"
                )
        return placed

    placed = place(present)
    first = placed  # the placement the line was first fitted on
    flags += placed.flags
    notes += placed.notes
    # Lanes grown from their peaks nearest the expected row, on two rows of
    # bands: each of two neighbours holds a band on the other's row too.
    crossed = [i for i in present if lanes[i].crossed]
    if crossed:
        flags.append("off_row_line")
        notes.append(
            f"{lanes_phrase(numbered(crossed))}: grown from bands on two rows, a lane's band"
            " on one and its neighbour's on the other, each lane holding a band on both;"
            " the bands nearest the expected row do not lie on one row"
        )

    # With prefer_y, the boxes off the expected row, and the lanes off the
    # row's line read on it: a few of them, holding nothing on it, recorded as
    # not detected (_not_detected), the row placed again without them; else
    # the row stays refused, and off_cause says why.
    expected: dict[int, float] = {}
    readings: dict[int, tuple[str, float, Rect | None]] = {}
    off_cause: OffCause | None = None
    again: tuple[tuple[int, float], ...] = ()
    if prefer is not None and placed.size is not None:
        height = placed.size.height
        expected = {i: ((r[1] + r[3]) / 2 - y0 - prefer) / height for i, r in placed.out.items()}
    off = sorted(i for i, v in placed.offsets.items() if abs(v) > ROW_LINE_K)
    if (
        prefer is not None
        and off
        and not crossed
        and [flag for flag in flags if flag in REFUSING_FLAGS] == ["off_row_line"]
    ):
        surfaces = _slot_surfaces(res)
        for i in off:
            bx0, by0, bx1, by1 = placed.out[i]
            centre = ((bx0 + bx1) / 2 - x0, (by0 + by1) / 2 - y0)
            line_y = centre[1] - placed.offsets[i] * placed.size.height
            readings[i] = _on_the_line(
                res, surfaces, lanes[i].own, lanes[i].seed_row, centre, line_y, pitch
            )
        reasons = {i: reading[0] for i, reading in readings.items()}
        off_cause, _ = _not_detected(placed.offsets, expected, reasons)
        if off_cause is None:
            keep = [i for i in present if i not in off]
            again_placed = place(keep)
            if again_placed.flags:  # the line moved: a box is off it still
                off_cause = "again"
                again = tuple(
                    (numbered([i])[0], v)
                    for i, v in sorted(again_placed.offsets.items())
                    if abs(v) > ROW_LINE_K
                )
            else:
                flags = [flag for flag in flags if flag not in first.flags]
                notes = [note for note in notes if note not in first.notes]
                placed = again_placed
                flags.append("off_expected_row")
                for i in off:
                    lanes[i].reason = "off_expected_row"
                    lanes[i].snr = readings[i][1]
                    lanes[i].window = readings[i][2]
                lines = [shown_against(abs(first.offsets[i]), ROW_LINE_K, 2) for i in off]
                snrs = [shown_against(readings[i][1], DETECT_K, 1) for i in off]
                notes.append(
                    f"{lanes_phrase(numbered(off))}: the band found lies off the row's line"
                    f" through the other boxes ({_phrase(lines)} box heights, limit"
                    f" {ROW_LINE_K:g}) and off the expected MW's row, and no other band"
                    " reaches the detection limit on that line (SNR"
                    f" {_phrase(snrs)}, limit {DETECT_K:g}): recorded as not detected (n.d.)"
                    " at the expected MW"
                )
    cut, size, out, outliers = placed.cut, placed.size, placed.out, placed.outliers
    offsets = dict(placed.offsets)
    if placed is not first:  # the lanes recorded: their band's box, before it was left out
        offsets.update({i: first.offsets[i] for i in off})
        expected = {
            **{i: expected[i] for i in off},
            **{i: ((r[1] + r[3]) / 2 - y0 - prefer) / size.height for i, r in out.items()},
        }

    # The reading against the bands' own spacing: placed, its lane numbers to
    # be checked (#111).
    doubts = _doubts(res, n, wc, x0, lambda k: lane_number(numbered([k])[0]))
    if doubts:
        flags.append("doubtful_lanes")
        notes.append(_DOUBT_NOTE + "; ".join(doubts))

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
                hollow=ln.hollow and rect is not None,
                window=window,
                cut=i in cut,
                line_offset=offsets.get(i),
                peaks=()
                if rect is None
                else tuple(
                    Peak(y0 + py, x0 + px, snr, own, other) for py, px, snr, own, other in ln.peaks
                ),
                expected_offset=expected.get(i),
                line_snr=readings[i][1] if i in readings else None,
                line_reason=readings[i][0] if i in readings else None,
            )
        )
    if right_to_left:
        result = [replace(ld, lane=n - 1 - ld.lane) for ld in reversed(result)]
    if any(ld.bg_offset is not None and abs(ld.bg_offset) > BG_WARN_K for ld in result):
        flags.append("background_mismatch")
    if outliers:
        flags.append("size_outlier")
        notes.append(
            f"{lanes_phrase(numbered(outliers))}: extent above "
            f"{SIZE_GUARD:g}x the median of the other extents, left out of the shared size"
        )
    multiple = [ld.lane for ld in result if ld.components > 1]
    if multiple:
        flags.append("multiple_components")
        if prefer is None:
            notes.append(
                f"{lanes_phrase(multiple)}: a second separate component reaches"
                f" {SECOND_SHARE:.0%} of the lane's peak; the box covers the one with the"
                " lane's strongest pixel"
            )
        else:
            notes.append(
                _other_bands_note(
                    {numbered([i])[0]: (out[i], lanes[i].second) for i in out if lanes[i].second},
                    x0,
                    y0,
                )
            )
    hollow = [ld.lane for ld in result if ld.hollow]
    if hollow:
        flags.append("hollow_band")
        notes.append(
            f"{lanes_phrase(hollow)}: a hollow band, lighter in its centre than the saturated"
            " pixels on either side of it: a sign of over-exposure"
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
        crossed=tuple(numbered(crossed)),
        off_cause=off_cause,
        again=again,
    )


def detect_row_along(
    gray: np.ndarray,
    row: Sequence[int],
    n_lanes: int,
    shifts: Sequence[int],
    *,
    background: float,
    **kwargs: Any,
) -> RowDetection:
    """:func:`detect_row` along a line that slopes across the row (#58, D11: the
    protein line between two ladders).

    ``shifts`` holds one whole-pixel shift per column of ``row`` as given, x0 to
    x1 - 1: at column x the line lies ``shifts[x - x0]`` rows lower than the
    row's own rows. Each column of the row box is moved up by its shift, so
    the line lies level along the box's rows: a permutation of the image's own
    pixels, with no value new or changed (the noise, the signal-to-noise and
    the saturated pixels read as before). :func:`detect_row` then runs,
    unchanged, on the image so levelled, with the other arguments as given
    (``background``, ``dark_on_light``, ``prefer_y`` in the levelled rows, and
    so on), so its checks, the row's line among them, see the row level. What
    it found is moved back down: each rect, grown extent and empty lane's
    window by the shift of its centre column (x0 + x1) // 2, each peak by the
    shift of its own column. Boxes do not overlap along x, so they do not
    overlap once moved; each keeps the shared size.

    With every shift 0 the result is :func:`detect_row`'s, bit for bit. The
    image outside the row box is never read, as by :func:`detect_row`.

    Raises :class:`RowDetectError` as :func:`detect_row` does; ``invalid_row``
    for ``shifts`` that are not one int per column of the row; and
    ``row_outside_image`` for a row whose rows, shifted at any of its columns
    inside the image, leave the image (the caller cuts the row to the rows
    every column can be shifted in)."""
    gray = np.asarray(gray)
    cx0, cy0, cx1, cy1 = _check_row(gray, row, n_lanes, background)
    x0 = int(tuple(row)[0])
    width = int(tuple(row)[2]) - x0
    if (
        isinstance(shifts, str | bytes)
        or len(shifts) != width
        or not all(_is_int(s) for s in shifts)
    ):
        raise RowDetectError(
            "invalid_row", f"shifts must be one int per column of the row ({width}), not {shifts!r}"
        )
    s = np.asarray(shifts[cx0 - x0 : cx1 - x0], dtype=np.int64)
    if cy0 + int(s.min()) < 0 or cy1 + int(s.max()) > gray.shape[0]:
        raise RowDetectError(
            "row_outside_image",
            f"row {tuple(int(v) for v in row)} shifted along its line by"
            f" {int(s.min())} to {int(s.max())} rows leaves the image",
        )
    if not s.any():
        return detect_row(gray, row, n_lanes, background=background, **kwargs)
    levelled = gray.copy()
    rows = np.arange(cy0, cy1)[:, None] + s[None, :]
    levelled[cy0:cy1, cx0:cx1] = gray[rows, np.arange(cx0, cx1)[None, :]]
    found = detect_row(levelled, row, n_lanes, background=background, **kwargs)

    def shift(column: int) -> int:
        return int(s[column - cx0])

    def back(rect: Rect | None) -> Rect | None:
        if rect is None:
            return None
        dy = shift((rect[0] + rect[2]) // 2)
        return (rect[0], rect[1] + dy, rect[2], rect[3] + dy)

    lanes = tuple(
        replace(
            lane,
            rect=back(lane.rect),
            extent=back(lane.extent),
            window=back(lane.window),
            peaks=tuple(p._replace(y=p.y + shift(math.floor(p.x))) for p in lane.peaks),
        )
        for lane in found.lanes
    )
    return replace(found, lanes=lanes)


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
        "pitch_doubt": PITCH_DOUBT,
        "lanes_doubt": LANES_DOUBT,
        "end_doubt": END_DOUBT,
        "apart_doubt": APART_DOUBT,
        "second_share": SECOND_SHARE,
        "hollow_pixels": HOLLOW_PIXELS,
        "join_rows": JOIN_ROWS,
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
