# SPDX-License-Identifier: Apache-2.0
"""The single compute step: a stored batch in, every result the app shows out.

Pure: :func:`compute_results` takes a :class:`proteia.core.model.Batch` and reads
no file and no pixel, so live results and the export reuse it without a session.
One call gives:

* the lane rows and the raw per-lane table: each protein's stored nets joined to
  the lane table by the stored lane index, so a lane with no box is ``None`` and
  shifts nothing. A lane with a not-detected record has no value either, so the
  statistics leave it out as they do a lane with no box; its ``detected`` flag
  (:func:`lane_detected`) and a ``below_detection`` notice tell the two apart;
* one :class:`SeriesResult` per (target, resolved loading control) pairing, with
  its per-lane ratios, fold-changes, reduced groups and chart;
* typed :class:`Notice` objects, built from the model directly, never by parsing
  warning strings: the background notices, for one, read each band's stored
  background fields and its protein's box size, never pixels.

Within a result set, each series is reduced once
(:func:`~proteia.core.analyze.reduce_samples` with that set's include flags). The
fold-change baseline comes from that reduction and the chart's groups are that
reduction divided by the baseline, so the table and the chart of a series use
one loading control and one baseline, and the plotted subset of conditions,
applied afterwards, never moves the baseline. Dividing the reduced values
(rather than reducing per-lane fold values) is equal in exact arithmetic,
bit-identical for representative repeats and single lanes, and can differ in the
last ulp only for averaged technical repeats.

When excluded lanes (include=no) hold values, a second result set is computed
with every lane included (:attr:`Results.all_lanes`), so removing data points is
never hidden; lanes excluded without any value (a ladder, an empty lane) add no
second set. Each of the two sets is labelled, and every chart carries its set's
label as its subtitle, so a chart cannot be mistaken for the other set's.

Each chart is tested once, on its plotted conditions (:func:`chart_test`), by
the statistics setting (:class:`~proteia.core.analyze.StatisticsSetting`), a
compute argument like the error type: ``auto`` chooses the test from the design
(the value kind, the replicates per condition and the reference), and the chart
states the test that ran, the conditions it covers and those it leaves out
(:func:`~proteia.core.plotspec.build_plotspec`). A condition with a replicate
the target was not detected in keeps its place on the chart, draws no bar and
is not tested; so does a condition with no value while one of its lanes holds a
box or a not-detected record of the chart's target or loading control. A lane
holding none (a ladder, an empty lane) is no replicate of the chart and gives
its condition no place. The two result sets may resolve to different tests (an
exclusion can make the replicates unequal); each chart states its own. There is
no correction across charts: the charts of several targets are each their own
family of comparisons.

Molecular weights (#58). Each protein's column carries its expected MW and
tolerance, and per lane its box's apparent MW (stored, read from its image's
calibration at the box's centre) and two checks, each ``passed``, ``failed``
or ``not_run`` (None: no box). The MW check (D7) passes an apparent MW within
the tolerance of the expected one, either way; it is not run without an
expected MW, without a curve on the image (``mw_not_run`` says which), or for
a box outside the calibrated range. The band count (D10) passes a lane whose
detector found no more bands in the count window around the box than the
protein is expected to have (:func:`count_window`, ``Band.bands_found``);
missing bands are not count failures (a lane below the detection limit is
``below_detection``), and a box placed or edited by hand is not counted. The
notices about them, and about the calibration they rest on (a curve from two
points, a ladder that fits poorly, two ladders that disagree, a ladder side
not used, rows that slope against the protein line), name the images and MWs
involved (:attr:`Notice.image_ids`) and count only the lanes of their set.
Every check reads the stored model: nothing is fitted to pixels here.

Notice messages count lanes from 1, as the user does; every index field
(:attr:`Notice.lane_indices`, :attr:`Results.excluded_lanes`) stays 0-based.
MWs in them are whole kDa (one decimal below 10 kDa).
"""

from __future__ import annotations

import math
import statistics as stats
from collections.abc import Callable, Collection, Mapping, Sequence
from enum import StrEnum
from functools import partial
from typing import Final, Literal

from pydantic import BaseModel, JsonValue, model_validator

from proteia.core import analyze, ladders, model, mwcal
from proteia.core.analyze import (
    ALPHA,
    BaselineError,
    LaneNets,
    ProteinNets,
    ReduceMethod,
    StatisticsSetting,
    TestResult,
    Tier,
    assess,
    compare,
    describe,
    normalize_batch,
    rank_floor_note,
    reduce_samples,
    reference_baseline,
    repeats_message,
    replicate_lanes,
    statistics_setting,
)
from proteia.core.model import Role, lanes_phrase
from proteia.core.names import name_key, resolve_label
from proteia.core.plotspec import (
    ErrorType,
    PlotSpec,
    ValueKind,
    build_plotspec,
    not_in_the_test,
)
from proteia.core.project import spine_axes
from proteia.core.quantify import (
    BACKGROUND_UNEVEN_LIMIT,
    NEAR_LIMIT_LEVELS,
    POSSIBLY_CLIPPED_PIXELS,
    near_limit_tolerance,
)


class NoticeCode(StrEnum):
    """What a :class:`Notice` is about. The values are stable for clients."""

    NO_LANES = "no_lanes"
    NO_TARGET = "no_target"
    NO_LOADING_CONTROL = "no_loading_control"
    # The model accepts these two on purpose; compute reports them.
    LOADING_CONTROL_AMBIGUOUS = "loading_control_ambiguous"  # none chosen among 2+
    REFERENCE_ALL_EXCLUDED = "reference_all_excluded"  # every reference lane is include=no
    REFERENCE_UNUSABLE = "reference_unusable"  # per series: no included value, or mean <= 0
    LOADING_NOT_POSITIVE = "loading_not_positive"  # target has a value, loading net <= 0
    NO_VALUES = "no_values"  # a series with no included value at all
    NO_PLOTTED_VALUES = "no_plotted_values"  # a series with no value in the plotted conditions
    TECHNICAL_REPEATS = "technical_repeats"  # repeats averaged, or one repeat kept per sample
    SIMILAR_CONDITIONS = "similar_conditions"  # labels equal except for case (name_key)
    UNKNOWN_PLOT_CONDITION = "unknown_plot_condition"  # ignored: a stale selection must not break
    REFERENCE_NOT_PLOTTED = "reference_not_plotted"  # the baseline still comes from it
    EXTRA_BANDS_IGNORED = "extra_bands_ignored"  # band_index > 0 is not quantified yet
    CLIPPED = "clipped"  # bands with pixels at the detector limit: over-exposed, still included
    # Bands with no clipping flag: their image has no limit the check trusts
    # (imaging.clipping_depth). A warning, as CLIPPED is: the bias it may hide is
    # the same, and a loading control's biases every value normalized to it.
    # Says to requantify where they were measured before #112 looked near the
    # limit.
    CLIPPING_NOT_CHECKED = "clipping_not_checked"
    # Bands on such an image with several pixels near the limit (#112,
    # quantify.is_possibly_clipped): likely over-exposed, still included.
    POSSIBLY_CLIPPED = "possibly_clipped"
    BELOW_DETECTION = "below_detection"  # not-detected records in included lanes: no value
    # Bands whose ring is cut short (by the image edge or other boxes): their
    # level is a robust plane or the image median (quantify.band_backgrounds).
    BACKGROUND_FALLBACK = "background_fallback"
    # Bands whose background differs around the box by more than the QC limit.
    BACKGROUND_UNEVEN = "background_uneven"
    # Nets above the whole-image median, as quantified before #83: requantify.
    LEGACY_BACKGROUND = "legacy_background"
    # Per image: nets measured on a reading of its file this version no longer
    # makes (#131: CMYK taken for RGB, a colour space now refused): import it again.
    OUTDATED_READING = "outdated_reading"
    # Per series: a test ran, but some plotted conditions are left out of it.
    CONDITIONS_NOT_TESTED = "conditions_not_tested"
    # Per series: a choice of the statistics setting could not apply, so no test ran.
    TEST_NOT_APPLICABLE = "test_not_applicable"
    # Per series: ratios with a value of 0 or below, tested on linear values.
    LOG_SCALE_UNAVAILABLE = "log_scale_unavailable"
    # Per series: a rank test whose smallest possible p is not below alpha.
    RANK_TEST_CANNOT_REACH_ALPHA = "rank_test_cannot_reach_alpha"
    # #58. Per protein: apparent MWs beyond its tolerance of the expected MW.
    MW_DEVIATION = "mw_deviation"
    # Per protein: lanes with more bands in the count window than it is expected to have.
    BAND_COUNT = "band_count"
    # Per protein: an expected MW and boxes, but no curve on its image.
    MW_NOT_CHECKED = "mw_not_checked"
    # Per protein: boxes whose centre lies outside the calibrated range.
    MW_OUTSIDE_CALIBRATION = "mw_outside_calibration"
    # Per ladder with boxes on its images: its curve rests on two points (D7).
    CALIBRATION_TWO_POINTS = "calibration_two_points"
    # Per ladder: a point its neighbours put more than mwcal.FIT_WARN away (D2).
    CALIBRATION_POOR_FIT = "calibration_poor_fit"
    # Per image: the rows slope against the protein line, so MWs drift across it.
    ROWS_TILTED = "rows_tilted"
    # Per register group: its two ladders disagree beyond mwcal.LADDERS_WARN (D1).
    LADDERS_DISAGREE = "ladders_disagree"
    # Per register group: a ladder side marked but not used, and why.
    LADDER_SIDE_IGNORED = "ladder_side_ignored"


class Level(StrEnum):
    INFO = "info"
    WARNING = "warning"


_INFO_CODES = frozenset(
    {
        NoticeCode.TECHNICAL_REPEATS,
        NoticeCode.REFERENCE_NOT_PLOTTED,
        NoticeCode.EXTRA_BANDS_IGNORED,
        NoticeCode.LEGACY_BACKGROUND,
        NoticeCode.LOG_SCALE_UNAVAILABLE,
        NoticeCode.MW_NOT_CHECKED,
        NoticeCode.CALIBRATION_TWO_POINTS,
        NoticeCode.LADDER_SIDE_IGNORED,
    }
)
# Notices about one set's charts: the tests of the two sets are their own (an
# exclusion can leave a condition out of one and not the other), so each set
# keeps these even when the other has the same.
TEST_NOTICE_CODES = frozenset(
    {
        NoticeCode.CONDITIONS_NOT_TESTED,
        NoticeCode.TEST_NOT_APPLICABLE,
        NoticeCode.LOG_SCALE_UNAVAILABLE,
        NoticeCode.RANK_TEST_CANNOT_REACH_ALPHA,
    }
)
# Notices about one series, not about each of its proteins: their protein_ids
# are the series' target and loading control, in that order, and they concern
# its chart alone, not that of another series sharing one of the proteins (a
# second target over the same loading control). The web page keeps the same
# list, pinned by a test.
SERIES_NOTICE_CODES = frozenset(
    {
        NoticeCode.REFERENCE_UNUSABLE,
        NoticeCode.LOADING_NOT_POSITIVE,
        NoticeCode.NO_VALUES,
        NoticeCode.NO_PLOTTED_VALUES,
        *TEST_NOTICE_CODES,
    }
)
# The background modes of a ring cut short (quantify.band_backgrounds).
_FALLBACK_MODES = frozenset({"asymmetric", "image"})
# What makes imaging.clipping_depth distrust an image, besides an unknown bit
# depth: its import warnings, as the clipping_not_checked and possibly_clipped
# notices name them. The keys are imaging.UNTRUSTED_WARNINGS, kept in step by a
# test rather than an import, so that loading results does not load the image
# readers.
_UNCHECKED_WARNINGS = {
    "lossy_format": "lossy (JPEG-type) compression",
    "cmyk_converted": "CMYK converted to RGB",
    "color_channels_differ": "color channels averaged into gray",
}


class Notice(BaseModel, frozen=True):
    """Something about the results the user should know, with the objects involved."""

    code: NoticeCode
    level: Level
    message: str
    protein_ids: tuple[str, ...] = ()
    lane_indices: tuple[int, ...] = ()
    conditions: tuple[str, ...] = ()
    image_ids: tuple[str, ...] = ()  # the images a calibration or MW notice concerns (#58)


# The state of an MW or band-count check in one lane (#58); None where the lane has no box.
CheckState = Literal["passed", "failed", "not_run"]
# Why a protein's MW is checked in no lane: no expected MW, or no curve on its
# image (no calibration point in its register group, or one).
MwNotRun = Literal["no_expected_mw", "no_points", "one_point"]


class ImageCalibration(BaseModel, frozen=True):
    """The calibration of a protein's image, as the results name it (#58):
    whether it comes from two ladders, the line's slope across the blot
    (degrees, positive where the right side runs lower) and how far the two
    ladders disagree once that slope is taken out, at which MW. The last three
    are None with one ladder; an infinite disagreement (degenerate ladders
    only) is None, with ``disagreement_infinite``, since JSON has no infinity."""

    two_ladders: bool
    tilt_deg: float | None
    disagreement: float | None
    disagreement_mw: float | None
    disagreement_infinite: bool = False


class LaneRow(BaseModel, frozen=True):
    """One row of the lane table, as the results show it."""

    index: int
    condition: str
    sample: str | None
    included: bool


class ProteinColumn(BaseModel, frozen=True):
    """One protein's column of the raw per-lane table (band index 0 only)."""

    protein_id: str
    name: str
    role: Role
    image_id: str
    nets: list[float | None]  # joined on the stored lane_index; None = no box
    band_ids: list[str | None]
    clipped: list[bool | None]  # per lane: over-exposed; None = no box, or not checked
    # Per lane, where clipped could not be checked: likely over-exposed (#112);
    # None = no box, or not assessed.
    possibly_clipped: list[bool | None]
    # Per lane: True = a box; False = not detected (below the detection limit, no
    # value); None = not measured.
    detected: list[bool | None]
    # #58. The expected MWs, top band first (empty: none given), and the MW
    # check's tolerance, a share (0.1: ±10%).
    expected_mws: list[float]
    mw_tolerance: float
    # Per lane: the box's apparent MW, kDa; None = no box, or no curve at its centre.
    apparent_mw: list[float | None]
    mw_check: list[CheckState | None]  # per lane; None = no box
    # Why no lane's MW is checked; None = checked wherever a box lies in the
    # calibrated range (a lane outside it is not_run).
    mw_not_run: MwNotRun | None
    # Per lane: the bands the detector found in the count window; None = no box,
    # or not counted (placed or edited by hand, or cleared).
    bands_found: list[int | None]
    count_check: list[CheckState | None]  # per lane; None = no box; not_run = not counted
    calibration: ImageCalibration | None  # its image's; None = no curve there


class SeriesResult(BaseModel, frozen=True):
    """One target normalized against one resolved loading control."""

    target_id: str
    target: str
    loading_id: str
    loading: str
    normalized: list[float | None]  # per lane: target net / loading net
    value_kind: ValueKind  # FOLD_CHANGE with a usable baseline, else LOADING_NORMALIZED
    baseline: float | None  # mean of the reference's reduced samples
    fold_change: list[float | None] | None  # per lane: normalized / baseline
    groups: dict[str, list[float]]  # the one reduction, every condition, in value_kind units
    averaged: list[tuple[str, str]]  # (condition, sample) keys whose repeats were collapsed
    chart: PlotSpec | None  # groups restricted to the plotted conditions
    # Per condition, in lane order: the included lanes where the target was not
    # detected (the loading control's are in its below_detection notice).
    undetected: dict[str, list[int]]


class Results(BaseModel, frozen=True):
    """Everything :func:`compute_results` gives, with the arguments it used.

    ``excluded_lanes`` names the lanes this set leaves out (include=no); empty
    means every lane is in. When it is not empty, ``all_lanes`` holds the same
    results with every lane included, so what an exclusion changes is always
    visible, and each set can be shown or exported on its own.

    ``label`` names the set when there are two: ``Excluding lane 8`` or
    ``Excluding lanes 3, 7`` (1-based, ascending) for this one, ``All lanes`` for
    ``all_lanes``. It is ``None`` when there is only one set. It names only the
    excluded lanes that hold values: an excluded lane without any (a ladder)
    changes nothing, so it is in ``excluded_lanes`` but not in the label. Every
    chart of a set has the set's label as its subtitle.
    """

    lanes: list[LaneRow]
    tier: Tier
    reference_condition: str | None
    proteins: list[ProteinColumn]
    series: list[SeriesResult]  # target order, then each target's loading-control order
    notices: list[Notice]
    plot_conditions: list[str] | None  # resolved to stored labels; None = all
    error_type: ErrorType
    method: ReduceMethod
    excluded_lanes: list[int] = []
    label: str | None = None
    all_lanes: Results | None = None
    statistics: StatisticsSetting = StatisticsSetting()

    @model_validator(mode="after")
    def _one_level(self) -> Results:
        if self.all_lanes is not None and self.all_lanes.all_lanes is not None:
            raise ValueError("the all-lanes result set has no all-lanes set of its own")
        return self


def _field(bands: list[model.Band | None], attr: str) -> list:
    """One attribute of joined bands, None where a lane has no band."""
    return [None if band is None else getattr(band, attr) for band in bands]


def _join(protein: model.Protein, n: int) -> list[model.Band | None]:
    """A protein's band-index-0 band per lane, by stored lane index.

    The same join as :func:`~proteia.core.project.join_to_spine`: a lane with no box
    is ``None`` and shifts nothing. The model allows one band-index-0 band per lane.
    """
    by_lane = {band.lane_index: band for band in protein.bands if band.band_index == 0}
    return [by_lane.get(i) for i in range(n)]


def lane_detected(batch: model.Batch) -> dict[str, list[bool | None]]:
    """Each protein's detection state per lane (band index 0), keyed like
    :func:`lane_nets`: True where a box was measured, False where a detector found
    the band below its detection limit (a not-detected record: no value), None
    where the lane was not measured.

    The one join of boxes and records to the lane table, by stored lane index as
    in :func:`_join`; :func:`compute_results` takes the detection state from
    here. The model allows a lane a box or a record, never both.
    """
    n = len(batch.lanes)
    detected: dict[str, list[bool | None]] = {}
    for protein in batch.proteins:
        state = {u.lane_index: False for u in protein.undetected if u.band_index == 0}
        state.update((b.lane_index, True) for b in protein.bands if b.band_index == 0)
        detected[protein.id] = [state.get(i) for i in range(n)]
    return detected


def lane_nets(batch: model.Batch) -> dict[str, LaneNets]:
    """Each protein's stored nets per lane, keyed by protein id in model order.

    Nets are joined to the lane table by their stored lane index (band index 0
    only): a lane with no box is ``None``. With no lanes, every protein has ``[]``.
    """
    n = len(batch.lanes)
    return {protein.id: _field(_join(protein, n), "net") for protein in batch.proteins}


def lane_clipped(batch: model.Batch) -> dict[str, list[bool | None]]:
    """Each protein's clipping flags per lane, keyed like :func:`lane_nets`: None
    where there is no box, or where the band was not checked."""
    n = len(batch.lanes)
    return {protein.id: _field(_join(protein, n), "clipped") for protein in batch.proteins}


def _listed(values: Collection[object]) -> str:
    return ", ".join(repr(v) for v in values)


def _background_notices(
    protein: model.Protein,
    bands: list[model.Band | None],
    included: list[bool],
    note: Callable[..., None],
) -> None:
    """The background notices of one protein's first bands (``bands``, joined to
    the lanes) in the included lanes, from their stored fields and the box size:
    a ring cut short (``background_fallback``), and a background uneven around
    the box, its spread moving the net by more than
    :data:`~proteia.core.quantify.BACKGROUND_UNEVEN_LIMIT` of it
    (``background_uneven``, also for any spread under a net of 0)."""
    size = protein.box_size
    shown = [(i, band) for i, band in enumerate(bands) if band is not None and included[i]]
    cut = tuple(i for i, band in shown if band.background_mode in _FALLBACK_MODES)
    if cut:
        note(
            NoticeCode.BACKGROUND_FALLBACK,
            f"{protein.name!r} has too little membrane around its box in {lanes_phrase(cut)}:"
            " the image edge or other boxes cut it, so the background there is estimated"
            f" less surely; when cropping, leave about {size.height} px above and below the"
            f" bands and {round(0.4 * size.width)} px beside them (one box height, 0.4"
            " box width)",
            protein_ids=(protein.id,),
            lane_indices=cut,
        )
    uneven = tuple(
        i
        for i, band in shown
        if band.background_spread * size.area > BACKGROUND_UNEVEN_LIMIT * band.net
    )
    if uneven:
        note(
            NoticeCode.BACKGROUND_UNEVEN,
            f"{protein.name!r} has an uneven background around its box in"
            f" {lanes_phrase(uneven)}: it differs from side to side by more than"
            f" {BACKGROUND_UNEVEN_LIMIT:.0%} of the net; check the membrane there",
            protein_ids=(protein.id,),
            lane_indices=uneven,
        )


# --- Molecular weights (#58): the MW and band-count checks ---

# rows_tilted (D9 pending; D1): the MW drift across the lanes from which the
# rows' slope against the protein line warns, and the fewest boxes of one
# protein (a detector's, nobody edited, in the set's lanes) a slope is read from.
TILT_WARN: Final = 0.05
TILT_MIN_LANES: Final = 4

_Fitted = mwcal.Calibration | mwcal.NoCalibration


def count_window(fitted: _Fitted, rect: model.Rect, tolerance: float) -> tuple[float, float] | None:
    """The rows a band count reads around a box ``rect`` (#58, D10): the box's
    centre plus or minus ``log10(1 + tolerance)`` decades of MW, in the pixels
    per decade of the image's calibration at that centre (12 px on a blot of
    300 px per decade at ±10%), so a neighbouring protein further off in MW is
    not counted. None where the image has no curve there: a count then reads
    the row box's rows."""
    if not isinstance(fitted, mwcal.Calibration):
        return None
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    ppd = fitted.px_per_decade(cx, cy)
    if ppd is None or not math.isfinite(ppd):
        return None
    half = math.log10(1.0 + tolerance) * ppd
    return cy - half, cy + half


def mw_check_settings() -> dict[str, JsonValue]:
    """How the MW and band-count checks run (#58), JSON-plain: what an export
    record reports under ``mw`` (with :func:`~proteia.core.mwcal.settings`)."""
    return {
        "apparent_mw": "the image's calibration at the box's centre, padding included",
        "deviation": "apparent / expected - 1",
        "passes": "(1 - tolerance) * expected <= apparent <= (1 + tolerance) * expected",
        "default_tolerance": model.Protein.model_fields["mw_tolerance"].default,
        "count_window": (
            "the box's centre +/- log10(1 + tolerance) decades on the image's calibration"
            " there; the row box's rows without a curve"
        ),
        "count": (
            "the band, and each other peak of the lane in the window that reaches the"
            " detector's second_share of the lane's peak and is as wide as a band; tops"
            " side by side in one band count once"
        ),
        "count_passes": "at most the expected band count",
        "tilt": (
            "per protein with tilt_min_lanes detector boxes nobody edited, the least-squares"
            " slope across the lanes of each box centre's offset from the protein line at"
            " its expected MW where the calibration reaches it (else the boxes' median"
            " apparent MW); the median slope over the lanes' span, as an MW drift at the"
            " middle lane"
        ),
        "tilt_warn": TILT_WARN,
        "tilt_min_lanes": TILT_MIN_LANES,
    }


def _mw_check(
    expected: float | None, tolerance: float, fitted: _Fitted, apparent: float | None
) -> CheckState:
    if expected is None or not isinstance(fitted, mwcal.Calibration) or apparent is None:
        return "not_run"
    within = (1.0 - tolerance) * expected <= apparent <= (1.0 + tolerance) * expected
    return "passed" if within else "failed"


def _count_check(band: model.Band, expected_count: int) -> CheckState:
    if band.bands_found is None:
        return "not_run"
    return "passed" if band.bands_found <= expected_count else "failed"


def _mw_not_run(protein: model.Protein, fitted: _Fitted) -> MwNotRun | None:
    if protein.expected_mw is None:
        return "no_expected_mw"
    if isinstance(fitted, mwcal.NoCalibration):
        return fitted.reason
    return None


def _beyond_range(fitted: mwcal.Calibration, protein: model.Protein, band: model.Band) -> bool:
    """Whether a box's centre lies outside its image's calibrated range, as a
    box without an apparent MW does, but for one stored by a fit from before
    #58 (its MW stays as stored until a calibration change refits it)."""
    size = protein.box_size
    try:
        mw = fitted.mw_at(band.box.x + size.width / 2, band.box.y + size.height / 2)
    except OverflowError:  # past the largest float: only a ladder labelled near it
        return True
    return mw is None


def _image_calibration(fitted: _Fitted) -> ImageCalibration | None:
    if not isinstance(fitted, mwcal.Calibration):
        return None
    disagreement = fitted.disagreement
    infinite = disagreement is not None and not math.isfinite(disagreement.value)
    return ImageCalibration(
        two_ladders=fitted.two_ladders,
        tilt_deg=fitted.tilt_deg,
        disagreement=None if disagreement is None or infinite else disagreement.value,
        disagreement_mw=None if disagreement is None else disagreement.mw,
        disagreement_infinite=infinite,
    )


def _kda(mw: float) -> str:
    """An MW as the notices write it: whole kDa, one decimal below 10 kDa, and
    in significant digits from a million kDa up or below 0.1 kDa (a ladder
    labelled far past any protein)."""
    if not math.isfinite(mw):
        return "∞"
    if mw >= 1e6 or mw < 0.1:
        return f"{mw:.3g}"
    return str(round(mw)) if mw >= 10.0 else f"{mw:.1f}".removesuffix(".0")


def _kda_at(z: float) -> str:
    """The MW ``10 ** z`` (a range end) as the notices write it; ∞ past the
    largest float, which only a ladder labelled near it reaches."""
    try:
        return _kda(10.0**z)
    except OverflowError:
        return "∞"


def _limit(share: float) -> str:
    """A tolerance or a threshold (a share) as a percentage, as given: ``10%``,
    ``12.5%``."""
    return f"{100.0 * share:.6g}%"


def _away(value: float, decimals: int, *, up: bool) -> float:
    """``value`` rounded to ``decimals`` up or down (a hair of float error
    taken off first, so 100.1 stays 100.1)."""
    scale = 10.0**decimals
    return (math.ceil if up else math.floor)(round(value * scale, 6)) / scale


def _percent_past(share: float, limit: float, *, signed: bool = True) -> str:
    """A share past ``limit`` in size, as a percentage: whole percent, or where
    that would read as within the limit as :func:`_limit` writes it (−10% past
    ±10%), with up to three decimals, rounded away from zero, so it does not.
    ``signed``: with its sign, a true minus sign (``+8%``, ``−10.1%``)."""
    size, bound = 100.0 * abs(share), float(_limit(limit).removesuffix("%"))
    decimals, shown = 0, float(f"{size:.0f}")
    while shown <= bound and decimals < 3:
        decimals += 1
        shown = _away(size, decimals, up=True)
    text = f"{shown:.{decimals}f}%"
    if not signed:
        return text
    return ("−" if share < 0 else "+") + text


def _kda_past(mw: float, expected: float, limit: float) -> str:
    """An MW past ``limit`` of ``expected`` as :func:`_kda` writes it, or where
    that would read as within the limit (83 kDa for 82.78 against 92 at ±10%),
    with up to three more decimals, rounded away from ``expected``, so it does
    not."""
    text = _kda(mw)
    if not 0.1 <= mw < 1e6:  # written in significant digits: far past any limit
        return text
    bound = float(_limit(limit).removesuffix("%"))
    decimals = 0 if mw >= 10.0 else 1
    shown, extra = float(text), 0
    while round(100.0 * abs(shown / expected - 1.0), 9) <= bound and extra < 3:
        extra += 1
        shown = _away(mw, decimals + extra, up=mw > expected)
        text = f"{shown:.{decimals + extra}f}"
    return text


def _ladder_words(membrane: model.Membrane) -> str:
    """The membrane's ladder, for a deviation notice: a wrong buffer system reads
    true bands 10 to 19% off (D3), which no fit measure catches."""
    name = membrane.calibration.ladder
    if name is None:
        return ""
    preset = ladders.preset(name)
    if preset is None:
        return f"; its ladder is {name!r}"
    return f"; its ladder, {preset.product}, is read with the values for {preset.system}"


def _mw_protein_notices(
    batch: model.Batch,
    protein: model.Protein,
    fitted: _Fitted,
    bands: list[model.Band | None],
    included: list[bool],
    note: Callable[..., None],
) -> None:
    """The MW and band-count notices of one protein's first bands in the set's lanes."""
    shown = [(i, band) for i, band in enumerate(bands) if band is not None and included[i]]
    if not shown:
        return
    ids, image = (protein.id,), (protein.image_id,)
    expected, tolerance = protein.expected_mw, protein.mw_tolerance
    if expected is not None and isinstance(fitted, mwcal.NoCalibration):
        points = batch.membrane_of(protein.image_id).calibration.points
        sides = {point.side for point in points if point.image_id in fitted.group}
        if fitted.reason == "no_points":
            missing = "no molecular-weight calibration"
            advice = (
                "mark the ladder on its marker image, or link it to the marker image it was"
                " taken with"
            )
        elif len(sides) > 1:  # one point on each side: neither is a ladder
            missing = "one calibration point per ladder side"
            advice = f"mark at least {mwcal.MIN_LADDER_POINTS} on one side"
        else:
            missing = "one calibration point"
            advice = f"mark at least {mwcal.MIN_LADDER_POINTS}"
        note(
            NoticeCode.MW_NOT_CHECKED,
            f"{protein.name!r} has {missing} on {protein.image_id}: its bands are taken to"
            f" be at {expected:g} kDa and their MW is not checked; {advice}",
            protein_ids=ids,
            image_ids=image,
        )
    if expected is not None and isinstance(fitted, mwcal.Calibration):
        failed = [
            (i, band.apparent_mw)
            for i, band in shown
            if _mw_check(expected, tolerance, fitted, band.apparent_mw) == "failed"
        ]
        if failed:
            mws = [mw for _, mw in failed if mw is not None]
            lanes = tuple(i for i, _ in failed)
            ends = (min(mws), max(mws))
            low, high = (_kda_past(mw, expected, tolerance) for mw in ends)
            at = low if low == high else f"{low}–{high}"
            low, high = (_percent_past(mw / expected - 1.0, tolerance) for mw in ends)
            spread = low if low == high else f"{low} to {high}"
            membrane = batch.membrane_of(protein.image_id)
            note(
                NoticeCode.MW_DEVIATION,
                f"{protein.name!r} runs at {at} kDa in {lanes_phrase(lanes)} ({spread}), more"
                f" than ±{_limit(tolerance)} from its expected {expected:g} kDa"
                f"{_ladder_words(membrane)}",
                protein_ids=ids,
                lane_indices=lanes,
                image_ids=image,
            )
        outside = tuple(
            i
            for i, band in shown
            if band.apparent_mw is None and _beyond_range(fitted, protein, band)
        )
        if outside:
            where = ", where both ladders reach" if fitted.two_ladders else ""
            lies = "lies" if len(outside) == 1 else "lie"
            note(
                NoticeCode.MW_OUTSIDE_CALIBRATION,
                f"{protein.name!r} in {lanes_phrase(outside)} {lies} outside the calibrated"
                f" range of {protein.image_id} ({_kda_at(fitted.z_hi)}–{_kda_at(fitted.z_lo)} kDa"
                f"{where}): its MW is not checked there",
                protein_ids=ids,
                lane_indices=outside,
                image_ids=image,
            )
    expected_count = protein.expected_band_count
    extra = [
        (i, band.bands_found)
        for i, band in shown
        if _count_check(band, expected_count) == "failed" and band.bands_found is not None
    ]
    if extra:
        counts = " or ".join(str(n) for n in sorted({n for _, n in extra}))
        lanes = tuple(i for i, _ in extra)
        window = (
            f"within ±{_limit(tolerance)} of its box's MW"
            if isinstance(fitted, mwcal.Calibration)
            else "in its row box"
        )
        are = "is" if expected_count == 1 else "are"
        note(
            NoticeCode.BAND_COUNT,
            f"{protein.name!r}: {counts} separate bands {window} in {lanes_phrase(lanes)},"
            f" where {expected_count} {are} expected",
            protein_ids=ids,
            lane_indices=lanes,
            image_ids=image,
        )


def _ladder_images(
    membrane: model.Membrane, group: frozenset[str], side: model.LadderSide
) -> tuple[str, ...]:
    """The images of a register group holding points of one ladder side, in
    membrane order."""
    holding = {p.image_id for p in membrane.calibration.points if p.side == side}
    return tuple(image.id for image in membrane.images if image.id in group and image.id in holding)


def _two_sides(fitted: mwcal.Calibration) -> bool:
    """Whether a register group has points on both sides, used or not: its
    notices then name each ladder's side."""
    sides = {ladder.side for ladder in fitted.ladders} | {side for side, _ in fitted.ignored}
    return sides != {model.LadderSide.LEFT}


def _calibration_notices(
    membrane: model.Membrane,
    group: frozenset[str],
    fitted: mwcal.Calibration,
    proteins: tuple[str, ...],
    note: Callable[..., None],
) -> None:
    """The notices of one register group's calibration, whose images hold the
    boxes of ``proteins`` (in any lane; some of them in the set's): a ladder
    of two points, a ladder point its neighbours put elsewhere, two ladders
    that disagree, and a side not used."""
    strip = model.CalibrationPointSource.STRIP_EDGE
    two_sides = _two_sides(fitted)
    for ladder in fitted.ladders:
        images = _ladder_images(membrane, group, ladder.side)
        named = ", ".join(images)
        name = (
            f"the {ladder.side.value} ladder of {named}" if two_sides else f"the ladder of {named}"
        )
        if len(ladder.mws) == 2:
            kind = (
                "strip edges"
                if all(source is strip for source in ladder.sources)
                else "ladder bands"
                if strip not in ladder.sources
                else "a ladder band and a strip edge"
            )
            first, second = (_kda(mw) for mw in ladder.mws)
            subject = name if two_sides else f"the calibration of {named}"
            note(
                NoticeCode.CALIBRATION_TWO_POINTS,
                f"{subject} rests on two points ({kind} at {first} and {second} kDa): apparent"
                " MWs are less reliable",
                protein_ids=proteins,
                image_ids=images,
            )
        quality = ladder.quality
        if quality is not None and quality.value > mwcal.FIT_WARN:
            # The share compared: the label against where its neighbours put it.
            way = "higher" if quality.mw > quality.predicted_mw else "lower"
            off = (
                f"{_percent_past(quality.value, mwcal.FIT_WARN, signed=False)} {way}"
                if math.isfinite(quality.value)
                else f"{way} by more than any MW spans"
            )
            note(
                NoticeCode.CALIBRATION_POOR_FIT,
                f"{_kda(quality.mw)} kDa on {name} sits where the bands above and below it put"
                f" {_kda(quality.predicted_mw)} kDa, so its label is {off}: check the label",
                protein_ids=proteins,
                image_ids=images,
            )
    disagreement = fitted.disagreement
    if disagreement is not None and disagreement.value > mwcal.LADDERS_WARN:
        marked = {p.image_id for p in membrane.calibration.points}
        images = tuple(image.id for image in membrane.images if image.id in group & marked)
        by = (
            f"{disagreement.value:.0%}"
            if math.isfinite(disagreement.value)
            else "by more than any MW spans"
        )
        note(
            NoticeCode.LADDERS_DISAGREE,
            f"the left and right ladders of {', '.join(images)} disagree near"
            f" {_kda(disagreement.mw)} kDa ({by}): check both ladders' labels",
            protein_ids=proteins,
            image_ids=images,
        )
    for side, reason in fitted.ignored:
        images = _ladder_images(membrane, group, side)
        where = f"the {side.value} ladder of {', '.join(images)}"
        if reason == "one_point":
            why = f"{where} has 1 point and is not used; mark at least {mwcal.MIN_LADDER_POINTS}"
        else:
            why = (
                f"{where} shares fewer than {mwcal.MIN_SHARED_MWS} marked MWs with the left one"
                " and is not used; mark the same bands on both"
            )
        note(
            NoticeCode.LADDER_SIDE_IGNORED,
            why,
            protein_ids=proteins,
            image_ids=images,
        )


def _tilt_boxes(
    protein: model.Protein, bands: list[model.Band | None], included: list[bool]
) -> list[model.Band]:
    """The first bands of a protein the tilt is read from: a detector's, nobody
    edited, in the set's lanes."""
    return [
        band
        for i, band in enumerate(bands)
        if band is not None
        and included[i]
        and band.source in model.DETECTING_SOURCES
        and not band.manually_edited
    ]


def _tilt_reference(
    fitted: mwcal.Calibration, protein: model.Protein, boxes: list[model.Band]
) -> float | None:
    """The MW of the protein line a protein's rows are read against: its
    expected MW where the calibration reaches it, else its boxes' median
    apparent MW (with one ladder, any MW in range gives the same slope); None
    without either."""
    expected = protein.expected_mw
    if expected is not None and fitted.z_lo <= math.log10(expected) <= fitted.z_hi:
        return expected
    mws = [band.apparent_mw for band in boxes if band.apparent_mw is not None]
    return stats.median(mws) if mws else None


def _rows_tilted(
    image_id: str,
    fitted: mwcal.Calibration,
    proteins: Sequence[model.Protein],
    joined: Mapping[str, list[model.Band | None]],
    included: list[bool],
    note: Callable[..., None],
) -> None:
    """``rows_tilted`` for one image (#58): how far the rows of its proteins
    slope against the protein line (its one ladder's level line, or the line
    between its two ladders). Per protein with at least :data:`TILT_MIN_LANES`
    boxes (:func:`_tilt_boxes`), each box centre's offset from the line at
    :func:`_tilt_reference` and the least-squares slope of those offsets
    across the lanes; the median slope over the lanes' span gives the drift,
    in MW at the middle lane. The notice names every protein of ``proteins``
    (those boxed on the image, in any lane): the tilt moves all their MWs,
    and the two result sets name the same ones."""
    slopes: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    for protein in proteins:
        boxes = _tilt_boxes(protein, joined[protein.id], included)
        if len(boxes) < TILT_MIN_LANES:
            continue
        size = protein.box_size
        reference = _tilt_reference(fitted, protein, boxes)
        if reference is None:
            continue
        points: list[tuple[float, float, float]] = []
        for band in boxes:
            cx, cy = band.box.x + size.width / 2, band.box.y + size.height / 2
            line = fitted.y_at(reference, cx)
            if line is not None:
                points.append((cx, cy, cy - line))
        if len(points) < TILT_MIN_LANES or len({cx for cx, _, _ in points}) < 2:
            continue
        fit = stats.linear_regression([cx for cx, _, _ in points], [r for _, _, r in points])
        slopes.append(fit.slope)
        xs.extend(cx for cx, _, _ in points)
        ys.extend(cy for _, cy, _ in points)
    if not slopes:
        return
    slope = stats.median(slopes)
    shift = abs(slope) * (max(xs) - min(xs))
    ppd = fitted.px_per_decade(stats.median(xs), stats.median(ys))
    if ppd is None or not ppd > 0.0:
        return
    try:
        drift = 10.0 ** (shift / ppd) - 1.0
    except OverflowError:
        drift = math.inf
    if not drift >= TILT_WARN:
        return
    angle = abs(math.degrees(math.atan(slope)))
    if fitted.two_ladders:
        against = "the protein line between its ladders"
        advice = "check both ladders' marks"
    else:
        [ladder] = fitted.ladders
        against = f"the {ladder.side.value} ladder" if _two_sides(fitted) else "the ladder"
        advice = "mark the ladder on the other side of the blot too"
        for side, reason in fitted.ignored:  # the other side, marked but not used
            advice = (
                f"mark at least {mwcal.MIN_LADDER_POINTS} points on the {side.value} ladder so"
                " that it is used too"
                if reason == "one_point"
                else f"mark the same bands on both ladders so that the {side.value} one is used too"
            )
    by = f"by up to {drift:.0%}" if math.isfinite(drift) else "past any MW"
    note(
        NoticeCode.ROWS_TILTED,
        f"{image_id}'s rows slope by about {angle:.1f}° against {against}"
        f" ({shift:.0f} px across the lanes), so apparent MWs drift {by} from one side to"
        f" the other; {advice}, or widen the tolerance",
        protein_ids=tuple(protein.id for protein in proteins),
        image_ids=(image_id,),
    )


def _mw_notices(
    batch: model.Batch,
    fits: Mapping[str, _Fitted],
    joined: Mapping[str, list[model.Band | None]],
    included: list[bool],
    note: Callable[..., None],
) -> None:
    """Every MW, band-count and calibration notice of a set (#58): per protein,
    then per register group with boxes in the set's lanes, then per image.
    The notices of a group or an image name every protein boxed on its images
    in any lane, so that the two result sets' copies of one name the same
    proteins (the all-lanes view tells them apart by code and proteins)."""
    for protein in batch.proteins:
        fitted = fits[protein.image_id]
        _mw_protein_notices(batch, protein, fitted, joined[protein.id], included, note)
    boxed = [
        protein
        for protein in batch.proteins
        if any(band is not None for band in joined[protein.id])
    ]
    in_set = {
        protein.id
        for protein in boxed
        if any(band is not None and included[i] for i, band in enumerate(joined[protein.id]))
    }
    for membrane in batch.membranes:
        for group, fitted in mwcal.calibrations(membrane).items():
            on = tuple(protein.id for protein in boxed if protein.image_id in group)
            if in_set.intersection(on) and isinstance(fitted, mwcal.Calibration):
                _calibration_notices(membrane, group, fitted, on, note)
    for image in batch.iter_images():
        fitted = fits.get(image.id)
        on_image = [protein for protein in boxed if protein.image_id == image.id]
        if in_set.intersection(p.id for p in on_image) and isinstance(fitted, mwcal.Calibration):
            _rows_tilted(image.id, fitted, on_image, joined, included, note)


def chart_test(
    shown: Mapping[str, Sequence[float]],
    *,
    setting: StatisticsSetting,
    kind: ValueKind,
    reference: str | None,
) -> TestResult:
    """The test of a chart's tested conditions: the one entry for the charts and
    the regression baseline, so both run the same test on the same values.
    Normalized values and fold changes are ratios; the reference, when it is
    among ``shown``, may be what the conditions are compared with."""
    return compare(shown, setting, ratio=kind is not ValueKind.RAW, reference=reference)


def _clipping_not_checked(
    protein: model.Protein,
    image: model.ImageRef,
    bands: list[model.Band | None],
    included: list[bool],
    note: Callable[..., None],
) -> None:
    """The ``clipping_not_checked`` notice of one protein's first bands
    (``bands``, joined to the lanes) in the included lanes: those with no
    clipping flag. It names why the check could not run on the protein's image,
    as :func:`~proteia.core.imaging.clipping_depth` decides it: lossy
    compression, CMYK converted to RGB, color averaged into gray, an unknown
    bit depth. The operations check every other image, so a flag missing there
    gets no reason. Where such a band was not assessed for pixels near the limit
    either, although its image's range is known (:func:`_unassessed`: measured
    before #112), it says to requantify."""
    unchecked = tuple(
        i
        for i, band in enumerate(bands)
        if band is not None and band.clipped is None and included[i]
    )
    if not unchecked:
        return
    reasons = _unchecked_reasons(image)
    because = ""
    if reasons:
        because = f": its image has {reasons}, so saturated pixels cannot be counted"
    requantify = ""
    if any(_unassessed(image, bands[i]) for i in unchecked):
        requantify = (
            "; its boxes were measured before Proteia looked for pixels near the detector"
            " limit: requantify to look for them"
        )
    note(
        NoticeCode.CLIPPING_NOT_CHECKED,
        f"{protein.name!r} was not checked for over-exposure in {lanes_phrase(unchecked)}"
        f"{because}; {_clipping_effect(protein)}{requantify}",
        protein_ids=(protein.id,),
        lane_indices=unchecked,
    )


def _unassessed(image: model.ImageRef, band: model.Band | None) -> bool:
    """Whether a band holds neither over-exposure flag on an image whose range
    is known but whose limit the exact check distrusts: the image
    :func:`~proteia.core.imaging.possible_clipping_depth` assesses, as
    :func:`~proteia.core.operations.unassessed_images` names it (a project
    saved before #112). Its warnings are read as :data:`_UNCHECKED_WARNINGS`
    names them."""
    return (
        band is not None
        and band.clipped is None
        and band.possibly_clipped is None
        and image.bit_depth is not None
        and any(warning.code in _UNCHECKED_WARNINGS for warning in image.import_warnings)
    )


def _unchecked_reasons(image: model.ImageRef) -> str:
    """Why the clipping check cannot run on an image, as
    :func:`~proteia.core.imaging.clipping_depth` decides it: its import warnings
    and an unknown bit depth, joined by "and"; empty with none."""
    codes = {warning.code for warning in image.import_warnings}
    reasons = [text for code, text in _UNCHECKED_WARNINGS.items() if code in codes]
    if image.bit_depth is None:
        reasons.append("an unknown bit depth")
    return " and ".join(reasons)


# What a saturated band beside an over-exposed one may do to its box (#58):
# their saturated pixels touching, one box holds both, whoever placed it.
_MERGED: Final = (
    "a saturated band next to it may have merged into the same box, which then holds both"
    " bands and its net may be too high"
)


def _clipping_effect(protein: model.Protein) -> str:
    """What over-exposure would do to a protein's values."""
    effect = "if it is over-exposed there, its net is an under-estimate"
    if protein.role is Role.LOADING_CONTROL:
        effect += ", which biases every value normalized to it"
    return effect


def _near_limit(bit_depth: int | None) -> str:
    """How near the detector limit the possibly-clipped pixels lie, in the
    image's own levels (and on an 8-bit scale, for another depth)."""
    if bit_depth is None or bit_depth == 8:
        return f"{NEAR_LIMIT_LEVELS} grey levels"
    return (
        f"{near_limit_tolerance(bit_depth):g} grey levels ({NEAR_LIMIT_LEVELS} on an 8-bit scale)"
    )


def _possibly_clipped(
    protein: model.Protein,
    image: model.ImageRef,
    bands: list[model.Band | None],
    included: list[bool],
    note: Callable[..., None],
) -> None:
    """The ``possibly_clipped`` notice of one protein's first bands (``bands``,
    joined to the lanes) in the included lanes: those whose image the clipping
    check cannot trust, with :data:`~proteia.core.quantify.POSSIBLY_CLIPPED_PIXELS`
    or more pixels near the limit (#112). It names the lanes, the rule, why the
    image cannot confirm it (as ``clipping_not_checked`` does), what the bias
    reaches, that a saturated band beside it may have merged into the box,
    making the net too high (#58), and the remedy, as ``clipped`` does."""
    flagged = tuple(
        i
        for i, band in enumerate(bands)
        if band is not None and band.possibly_clipped and included[i]
    )
    if not flagged:
        return
    boxes = "its box holds" if len(flagged) == 1 else "each of those boxes holds"
    reasons = _unchecked_reasons(image)
    because = f", and its image has {reasons}, so saturation cannot be confirmed" if reasons else ""
    note(
        NoticeCode.POSSIBLY_CLIPPED,
        f"{protein.name!r} is possibly over-exposed in {lanes_phrase(flagged)}: {boxes}"
        f" {POSSIBLY_CLIPPED_PIXELS} or more pixels within {_near_limit(image.bit_depth)} of"
        f" the detector limit{because}; {_clipping_effect(protein)}, and {_MERGED}; check the"
        " imager's original capture, or shorten the exposure and image the membrane again",
        protein_ids=(protein.id,),
        lane_indices=flagged,
    )


def _chart(
    groups: dict[str, list[float]],
    point_lanes: dict[str, list[list[int]]],
    replicates: dict[str, list[list[int]]],
    values: LaneNets,
    detected: list[bool | None],
    *,
    chosen: list[str] | None,
    kind: ValueKind,
    error_type: ErrorType,
    title: str,
    reference: str | None,
    subtitle: str | None,
    setting: StatisticsSetting,
) -> tuple[PlotSpec, TestResult] | None:
    """The chart of one series (its groups restricted to the plotted
    conditions) and its test; None when no plotted condition has a value.

    ``replicates`` is every included replicate's lanes by condition
    (:func:`~proteia.core.analyze.replicate_lanes`), over the lanes that hold
    something for the series, so a plotted condition with no value keeps its
    place while a lane that holds nothing (a ladder) has none. A replicate with
    no value (``values``) whose target was not detected in one of its lanes
    (``detected``) is not detected: its condition draws no bar and is left out
    of the test."""
    plotted = [c for c in replicates if chosen is None or c in chosen]
    shown = {c: groups[c] for c in plotted if c in groups}
    if not shown:
        return None
    # Provenance parallel to the points: each sample's first lane (the lane a
    # representative reduction keeps; technical repeats share one point).
    lane_indices = {c: [lanes[0] for lanes in point_lanes[c]] for c in shown}
    undetected: dict[str, list[int]] = {}
    for condition in plotted:
        firsts = [
            lanes[0]
            for lanes in replicates[condition]
            if all(values[i] is None for i in lanes) and any(detected[i] is False for i in lanes)
        ]
        if firsts:
            undetected[condition] = firsts
    tested = {c: g for c, g in shown.items() if c not in undetected}
    # Statistics run on the plotted subset.
    test = chart_test(tested, setting=setting, kind=kind, reference=reference)
    spec = build_plotspec(
        shown,
        describe(shown),
        test,
        value_kind=kind,
        error_type=error_type,
        title=title,
        lane_indices=lane_indices,
        first_label=reference,
        subtitle=subtitle,
        replicates={c: len(replicates[c]) for c in plotted},
        not_detected_lanes=undetected,
    )
    return spec, test


def _test_notices(
    chart: PlotSpec,
    test: TestResult,
    series: str,
    protein_ids: tuple[str, ...],
    note: Callable[..., None],
) -> None:
    """The notices of a chart's test: the conditions it leaves out, ratios tested
    on linear values, a rank test that cannot reach significance, or a choice of
    the setting that could not apply (``series`` names the series)."""
    plan = test.plan
    if chart.test is None:
        if plan.no_test == "not_applicable":
            note(
                NoticeCode.TEST_NOT_APPLICABLE,
                f"{series}: {plan.no_test_note}",
                protein_ids=protein_ids,
            )
        return
    out = tuple(c.label for c in chart.coverage if c.left_out is not None)
    if out:
        note(
            NoticeCode.CONDITIONS_NOT_TESTED,
            f"{series}: {not_in_the_test(chart.coverage)}",
            protein_ids=protein_ids,
            conditions=out,
        )
    if plan.nonpositive:
        has = "has" if len(plan.nonpositive) == 1 else "have"
        note(
            NoticeCode.LOG_SCALE_UNAVAILABLE,
            f"{series} is tested on linear values: {_listed(plan.nonpositive)} {has} a value"
            " of 0 or below, which has no log",
            protein_ids=protein_ids,
            conditions=plan.nonpositive,
        )
    floor = plan.min_attainable_p
    if floor is not None and floor >= ALPHA:
        note(
            NoticeCode.RANK_TEST_CANNOT_REACH_ALPHA,
            f"{series}: {rank_floor_note(plan.name, floor)}",
            protein_ids=protein_ids,
        )


def compute_results(
    batch: model.Batch,
    *,
    plot_conditions: Collection[str] | None = None,
    error_type: ErrorType | str = ErrorType.SD,
    method: ReduceMethod | str = ReduceMethod.MEAN,
    statistics: StatisticsSetting | Mapping[str, str] | None = None,
    outdated: Mapping[str, str] | None = None,
) -> Results:
    """Everything the results view shows, from the stored batch alone.

    See :func:`_compute` for one result set. When lanes the lane table excludes
    (include=no) hold a value for any protein, the returned set applies the
    exclusions and its ``all_lanes`` is the same computation with every lane
    included: removing data points is never hidden. Excluded lanes without any
    value (a ladder, an empty lane) change nothing, so they add no second set.
    Notices the two sets share are kept only in the first, but for those about
    a set's charts (:data:`TEST_NOTICE_CODES`), which each set keeps: the two
    sets test on their own. Two sets are labelled
    (see :class:`Results`); one set has no label. A chart with a plotted group
    too small for a test (n < 2) is still drawn. When at least two other groups
    can be tested, the chart shows their test, with brackets only among them and
    a note naming the groups left out; otherwise it has no test result, no
    brackets, and a note saying why (see
    :func:`~proteia.core.plotspec.build_plotspec`).

    ``statistics`` chooses every chart's test (None: all ``auto``); both sets use
    it, and :attr:`Results.statistics` echoes it.

    ``outdated`` gives, by image id, why the image's stored nets were measured on
    a reading of its file this version no longer makes, as the session tells it
    (:meth:`~proteia.core.session.ProjectSession.outdated_reading`); the
    proteins with bands on each such image get an ``outdated_reading`` warning
    with that message. The batch alone cannot tell: its file must be read.

    ``error_type``, ``method`` and ``statistics`` may be their raw values
    (``"SEM"``, ``"mean"``, ``{"family": "welch"}``): they become their enums
    here, before anything compares them by identity. An unknown value raises
    ``ValueError``.
    """
    one_set = partial(
        _compute,
        plot_conditions=plot_conditions,
        error_type=ErrorType(error_type),
        method=ReduceMethod(method),
        statistics=statistics_setting(statistics),
        outdated=outdated or {},
    )
    per_protein = lane_nets(batch).values()
    removed = [  # the excluded lanes that hold a value: what the exclusion changes
        lane.index
        for lane in batch.lanes
        if not lane.included and any(nets[lane.index] is not None for nets in per_protein)
    ]
    if not removed:
        return one_set(batch, set_label=None)
    results = one_set(batch, set_label=f"Excluding {lanes_phrase(removed)}")
    every_lane = batch.model_copy(
        update={"lanes": [lane.model_copy(update={"included": True}) for lane in batch.lanes]}
    )
    all_lanes = one_set(every_lane, set_label="All lanes")
    own = [
        notice
        for notice in all_lanes.notices
        if notice.code in TEST_NOTICE_CODES or notice not in results.notices
    ]
    return results.model_copy(update={"all_lanes": all_lanes.model_copy(update={"notices": own})})


def _compute(
    batch: model.Batch,
    *,
    plot_conditions: Collection[str] | None,
    error_type: ErrorType,
    method: ReduceMethod,
    statistics: StatisticsSetting,
    outdated: Mapping[str, str],
    set_label: str | None,
) -> Results:
    """One result set, over the lane table's included lanes.

    ``plot_conditions`` chooses the charted conditions (None or empty: all); each
    is resolved against the lane labels (:func:`~proteia.core.names.resolve_label`),
    and one that matches no lane is reported and ignored. ``method`` reduces
    technical repeats; ``error_type`` picks the charts' error bars and
    ``statistics`` their tests. ``outdated`` names the images whose nets were
    measured on a reading no longer made (:func:`compute_results`).
    ``set_label`` names the set (:attr:`Results.label`) and is every chart's
    subtitle.

    Series come from :func:`~proteia.core.analyze.normalize_batch`. Each is reduced
    once over the lane table's included lanes. With a reference condition and a
    usable baseline, the per-lane fold-changes and the groups are divided by it;
    an unusable baseline gives loading-normalized groups and no chart for that
    series. The plotted subset is applied last and changes only the chart.
    """
    if isinstance(plot_conditions, str):
        raise TypeError("plot_conditions is a collection of condition labels, not one label")
    n = len(batch.lanes)
    ref = batch.reference_condition
    notices: list[Notice] = []

    def note(code: NoticeCode, message: str, **objects: tuple) -> None:
        level = Level.INFO if code in _INFO_CODES else Level.WARNING
        notices.append(Notice(code=code, level=level, message=message, **objects))

    # 1. The lane rows and the raw table, and what the model lacks.
    lanes = [
        LaneRow(index=lane.index, condition=lane.label, sample=lane.sample, included=lane.included)
        for lane in batch.lanes
    ]
    joined = {protein.id: _join(protein, n) for protein in batch.proteins}
    nets = {pid: _field(bands, "net") for pid, bands in joined.items()}
    detected = lane_detected(batch)
    # Each protein image's calibration (#58), from the stored points.
    fits = {
        p.image_id: mwcal.calibration_for(batch.membrane_of(p.image_id), p.image_id)
        for p in batch.proteins
    }
    columns = [
        ProteinColumn(
            protein_id=p.id,
            name=p.name,
            role=p.role,
            image_id=p.image_id,
            nets=nets[p.id],
            band_ids=_field(joined[p.id], "id"),
            clipped=_field(joined[p.id], "clipped"),
            possibly_clipped=_field(joined[p.id], "possibly_clipped"),
            detected=detected[p.id],
            expected_mws=[] if p.expected_mw is None else [p.expected_mw],
            mw_tolerance=p.mw_tolerance,
            apparent_mw=_field(joined[p.id], "apparent_mw"),
            mw_check=[
                None
                if band is None
                else _mw_check(p.expected_mw, p.mw_tolerance, fits[p.image_id], band.apparent_mw)
                for band in joined[p.id]
            ],
            mw_not_run=_mw_not_run(p, fits[p.image_id]),
            bands_found=_field(joined[p.id], "bands_found"),
            count_check=[
                None if band is None else _count_check(band, p.expected_band_count)
                for band in joined[p.id]
            ],
            calibration=_image_calibration(fits[p.image_id]),
        )
        for p in batch.proteins
    ]
    # Boxes only: a record beyond the first band has no value to ignore.
    extra = tuple(p.id for p in batch.proteins if any(b.band_index > 0 for b in p.bands))
    if extra:
        note(
            NoticeCode.EXTRA_BANDS_IGNORED,
            "extra bands (band index above 0) are not quantified yet;"
            " each lane uses its first band",
            protein_ids=extra,
        )
    # A project quantified before #83 keeps the whole-image median until it is
    # requantified; every band then has this mode (the model keeps one method).
    legacy = tuple(
        p.id for p in batch.proteins if any(b.background_mode == "global_median" for b in p.bands)
    )
    if legacy:
        note(
            NoticeCode.LEGACY_BACKGROUND,
            "nets use the whole-image median background, as measured before the local"
            " background; requantify to measure each band's background from the membrane"
            " around its box",
            protein_ids=legacy,
        )
    for image_id, why in outdated.items():  # an image with no band has no nets to warn about
        measured = tuple(p.id for p in batch.proteins if p.image_id == image_id and p.bands)
        if measured:
            note(NoticeCode.OUTDATED_READING, why, protein_ids=measured)
    targets = [p for p in batch.proteins if p.role is Role.TARGET]
    loading_controls = [p for p in batch.proteins if p.role is Role.LOADING_CONTROL]
    if not targets:
        note(NoticeCode.NO_TARGET, "no target protein: raw nets only")
    if not loading_controls:
        note(NoticeCode.NO_LOADING_CONTROL, "no loading control: cannot normalize; raw nets only")

    name_of = {p.id: p.name for p in batch.proteins}
    id_of = {p.name: p.id for p in batch.proteins}  # names are unique in the model
    protein_nets = [
        ProteinNets(
            p.name, p.role, nets[p.id], loadings=tuple(name_of[i] for i in p.loading_control_ids)
        )
        for p in batch.proteins
    ]

    # 2. No lanes: only the raw columns (all empty).
    if n == 0:
        note(NoticeCode.NO_LANES, "no lanes declared: declare the lane table")
        return Results(
            lanes=[],
            tier=assess(analyze.Batch([], protein_nets)).tier,
            reference_condition=ref,
            proteins=columns,
            series=[],
            notices=notices,
            plot_conditions=None,
            error_type=error_type,
            method=method,
            statistics=statistics,
        )

    # 3-4. The lane axes, and what the model accepts on purpose but the user should see.
    conditions, samples, included = spine_axes(batch.lanes)
    for column in columns:  # over-exposed bands in lanes this set includes
        over = tuple(i for i, flag in enumerate(column.clipped) if flag and included[i])
        if over:
            kept = "the lane stays" if len(over) == 1 else "the lanes stay"
            note(
                NoticeCode.CLIPPED,
                f"{column.name!r} is over-exposed in {lanes_phrase(over)}: pixels at the detector"
                f" limit make its net an under-estimate, and {_MERGED}; shorten the exposure"
                f" and image the membrane again; {kept} included",
                protein_ids=(column.protein_id,),
                lane_indices=over,
            )
    for protein in batch.proteins:  # bands likely over-exposed, where that cannot be checked
        image = batch.find_image(protein.image_id)
        _possibly_clipped(protein, image, joined[protein.id], included, note)
    for protein in batch.proteins:  # bands not checked for that, in lanes this set includes
        image = batch.find_image(protein.image_id)
        _clipping_not_checked(protein, image, joined[protein.id], included, note)
    for protein in batch.proteins:  # backgrounds to check, in lanes this set includes
        _background_notices(protein, joined[protein.id], included, note)
    for column in columns:  # not-detected records in lanes this set includes
        below = tuple(i for i, flag in enumerate(column.detected) if flag is False and included[i])
        if not below:
            continue
        if column.role is Role.LOADING_CONTROL:
            effect = "targets normalized to it have no value there"
        elif len(below) == 1:
            effect = "that lane has no value and is left out of the statistics"
        else:
            effect = "those lanes have no value and are left out of the statistics"
        note(
            NoticeCode.BELOW_DETECTION,
            f"{column.name!r} was not detected in {lanes_phrase(below)}"
            f" (below the detection limit): {effect}",
            protein_ids=(column.protein_id,),
            lane_indices=below,
            conditions=tuple(dict.fromkeys(conditions[i] for i in below)),
        )
    _mw_notices(batch, fits, joined, included, note)  # #58, in lanes this set includes
    labels = list(dict.fromkeys(conditions))  # distinct, in lane order
    similar: dict[str, list[str]] = {}
    for label in labels:
        similar.setdefault(name_key(label), []).append(label)
    for group in similar.values():
        if len(group) > 1:
            note(
                NoticeCode.SIMILAR_CONDITIONS,
                f"conditions {_listed(group)} differ only in case or look alike;"
                " they are separate groups",
                conditions=tuple(group),
            )
    if len(loading_controls) > 1:
        for target in targets:
            if not target.loading_control_ids:
                note(
                    NoticeCode.LOADING_CONTROL_AMBIGUOUS,
                    f"{target.name!r} has no loading control chosen among"
                    f" {len(loading_controls)}: it is not normalized",
                    protein_ids=(target.id,),
                )
    reference_lanes = tuple(i for i in range(n) if conditions[i] == ref)
    reference_all_excluded = bool(reference_lanes) and not any(included[i] for i in reference_lanes)
    if reference_all_excluded:
        note(
            NoticeCode.REFERENCE_ALL_EXCLUDED,
            f"every lane of the reference condition {ref!r} is excluded:"
            " no fold-change can be formed from the included lanes",
            lane_indices=reference_lanes,
            conditions=(ref,),
        )

    # 5. The statistics input: loading-control ids become names.
    abatch = analyze.Batch(conditions, protein_nets, control_condition=ref)
    tier = assess(abatch).tier
    if tier is Tier.FOLD_CHANGE and reference_all_excluded:
        tier = Tier.NORMALIZED  # no series can form a fold-change

    # 6. The plotted subset.
    chosen: list[str] | None = None
    if plot_conditions:
        resolved: set[str] = set()
        unknown: dict[str, None] = {}  # an ordered set
        for entry in plot_conditions:
            label = resolve_label(entry, labels)
            if label is None:
                unknown[entry] = None
            else:
                resolved.add(label)
        if unknown:
            note(
                NoticeCode.UNKNOWN_PLOT_CONDITION,
                f"plot condition(s) {_listed(unknown)} match no lane; ignored",
                conditions=tuple(unknown),
            )
        if resolved:
            chosen = [label for label in labels if label in resolved]
    if tier is Tier.FOLD_CHANGE and chosen is not None and ref not in chosen:
        note(
            NoticeCode.REFERENCE_NOT_PLOTTED,
            f"the reference condition {ref!r} is not plotted;"
            " fold-changes are still relative to it",
            conditions=(ref,),
        )

    # 7. One series per (target, resolved loading control), each reduced once.
    series: list[SeriesResult] = []
    averaged: list[tuple[str, str]] = []
    if tier is not Tier.EXPORT_ONLY:
        pairs, _ = normalize_batch(abatch)
        for s in pairs:
            target_id, loading_id = id_of[s.target], id_of[s.loading]
            pair_ids = (target_id, loading_id)
            target_nets, loading_nets = nets[target_id], nets[loading_id]
            not_positive = tuple(
                i
                for i in range(n)
                if included[i]
                and target_nets[i] is not None
                and loading_nets[i] is not None
                and loading_nets[i] <= 0
            )
            if not_positive:
                note(
                    NoticeCode.LOADING_NOT_POSITIVE,
                    f"{s.loading!r} has a net of 0 in {lanes_phrase(not_positive)}:"
                    f" {s.target!r} / {s.loading!r} has no value there",
                    protein_ids=pair_ids,
                    lane_indices=not_positive,
                )

            red = reduce_samples(s.values, conditions, samples, included=included, method=method)
            for key in red.averaged:
                if key not in averaged:
                    averaged.append(key)
            undetected: dict[str, list[int]] = {}
            for i, flag in enumerate(detected[target_id]):
                if flag is False and included[i]:
                    undetected.setdefault(conditions[i], []).append(i)
            kind = ValueKind.LOADING_NORMALIZED
            groups = red.groups
            baseline: float | None = None
            fold_change: LaneNets | None = None
            chartable = True
            if ref is not None:
                try:
                    baseline = reference_baseline(red.groups, ref)
                except BaselineError as exc:
                    chartable = False  # no baseline, so no fold change to chart
                    # Already explained when every reference lane is excluded, or when
                    # the series has no value at all (NO_VALUES below).
                    if not reference_all_excluded and red.groups:
                        reason = str(exc)
                        in_reference = [i for i in reference_lanes if included[i]]
                        # The target or its loading control (or both) was not
                        # detected in any included reference lane.
                        missing = [
                            name
                            for pid, name in ((target_id, s.target), (loading_id, s.loading))
                            if in_reference and all(detected[pid][i] is False for i in in_reference)
                        ]
                        if exc.reason == "no_value" and missing:
                            were = "was" if len(missing) == 1 else "were"
                            reason = (
                                f"{' and '.join(map(repr, missing))} {were} not detected in the"
                                f" reference condition {ref!r} (below the detection limit):"
                                " no fold-change can be formed"
                            )
                        note(
                            NoticeCode.REFERENCE_UNUSABLE,
                            f"{s.target!r} / {s.loading!r}: {reason}",
                            protein_ids=pair_ids,
                            conditions=(ref,),
                        )
                else:
                    fold_change = [None if v is None else v / baseline for v in s.values]
                    groups = {c: [v / baseline for v in vals] for c, vals in red.groups.items()}
                    kind = ValueKind.FOLD_CHANGE

            chart = None
            if not groups:
                note(
                    NoticeCode.NO_VALUES,
                    f"{s.target!r} / {s.loading!r} has no value in any included lane",
                    protein_ids=pair_ids,
                )
            elif chartable:
                if kind is ValueKind.FOLD_CHANGE:
                    title = f"{s.target} fold-change vs {ref}  (/{s.loading})"
                else:
                    title = f"{s.target} / {s.loading}"
                # A lane is a replicate's when it holds a box or a not-detected
                # record of the target or its loading control. One holding
                # neither (a ladder, an empty lane) gives no value, so it gives
                # no condition a place either, and changes nothing in either set.
                held = [
                    included[i] and any(detected[pid][i] is not None for pid in pair_ids)
                    for i in range(n)
                ]
                charted = _chart(
                    groups,
                    red.lanes,
                    replicate_lanes(conditions, samples, held),
                    s.values,
                    detected[target_id],
                    chosen=chosen,
                    kind=kind,
                    error_type=error_type,
                    title=title,
                    reference=ref,
                    subtitle=set_label,
                    setting=statistics,
                )
                if charted is None:
                    note(
                        NoticeCode.NO_PLOTTED_VALUES,
                        f"{s.target!r} / {s.loading!r} has no value in the plotted conditions",
                        protein_ids=pair_ids,
                    )
                else:
                    chart, tested = charted
                    _test_notices(chart, tested, f"{s.target!r} / {s.loading!r}", pair_ids, note)
            series.append(
                SeriesResult(
                    target_id=target_id,
                    target=s.target,
                    loading_id=loading_id,
                    loading=s.loading,
                    normalized=s.values,
                    value_kind=kind,
                    baseline=baseline,
                    fold_change=fold_change,
                    groups=groups,
                    averaged=red.averaged,
                    chart=chart,
                    undetected=undetected,
                )
            )

    # 8. Technical repeats, once for all series.
    if averaged:
        note(
            NoticeCode.TECHNICAL_REPEATS,
            repeats_message(len(averaged), method),
            conditions=tuple(dict.fromkeys(c for c, _ in averaged)),
        )

    return Results(
        lanes=lanes,
        tier=tier,
        reference_condition=ref,
        proteins=columns,
        series=series,
        notices=notices,
        plot_conditions=chosen,
        error_type=error_type,
        method=method,
        excluded_lanes=[i for i in range(n) if not included[i]],
        label=set_label,
        statistics=statistics,
    )
