# SPDX-License-Identifier: Apache-2.0
"""The project operations: one function per user action, with no GUI.

Every function takes an open :class:`~proteia.core.session.ProjectSession` (from
:func:`new_project` or :func:`open_project`) and holds its lock. A state change
runs as one :func:`~proteia.core.model.apply_change` on a copy of the project,
which re-validates the whole tree, and is committed through the session, which
then autosaves. So an edit is all or nothing:

* Checks run on the committed project first and raise :class:`OperationError`
  with a stable :class:`ErrorCode`; an unknown id raises
  :class:`~proteia.core.model.UnknownIdError`. A refusal's message numbers
  lanes from 1, as the user does (:func:`~proteia.core.model.lane_number`);
  log params and returned fields keep the stored 0-based lane index. A
  refusal changes nothing: the project (the same object), ``next_id``, the
  pixel cache and the ``images/`` listing are as they were, and the autosave
  hook does not run.
* Pixels are fetched before the change (:func:`requantify` reads them inside
  it, one image at a time), so an image-file problem changes nothing.
  New ids come only from ``new_id`` inside the change, so a refusal uses up no
  number.
* An edit that leaves the project equal to the committed one is a no-op: nothing
  is committed and the hook does not run.
* Every committed change appends one log entry (see
  :class:`~proteia.core.model.LogEntry`); a refusal or a no-op appends none, and
  so do :func:`compute_view`, :func:`compute`, :func:`export_lane_table`,
  :func:`export_bundle`, :func:`save`, :func:`propose_ladder` and
  :func:`snap_ladder`, which change no state. The params
  record the inputs as they took effect (cleaned text, the stored spelling, the
  proposed lane, the snapped rect, the size used) and the ids created or
  removed; objects are named by id, never by path or typed text.
  Clients commit a drag or a cell edit once, when it ends, not on every pointer
  move or keystroke.
* Every committed change, of any operation, can be undone: the session keeps
  each committed state and :func:`undo` and :func:`redo` restore one whole, as
  a logged change of their own, so no operation needs an inverse, and an undone
  change stays in the log.

Stored values that depend on pixels or geometry are recomputed by the operation
that invalidates them, in the same change, so the logged content hash covers
them: :func:`_quantify_image` is the one writer of a band's net, background
fields and ``clipped`` and ``possibly_clipped`` flags, :func:`_refresh_mw` the
one writer of a band's ``apparent_mw`` and a membrane's ``fit_method`` and
``fit_quality`` (see below), and :func:`_set_box` the one place a box moves. The
invariant: for every image, every band's stored net, ``background_level``,
``background_mode``, ``background_spread``, ``clipped`` and ``possibly_clipped``
equal :func:`~proteia.core.quantify.band_backgrounds`,
:func:`~proteia.core.quantify.net_signal`,
:func:`~proteia.core.quantify.is_clipped` and
:func:`~proteia.core.quantify.is_possibly_clipped` of the stored pixels, all the boxes
on that image (every protein's, each with its protein's box size), the
polarity, the bit depth and the project's background method. A band's ring
excludes every other box on its image, so any edit that adds, moves, resizes or
removes a box re-quantifies the whole image: :func:`place_box`,
:func:`move_box`, :func:`remove_box`, :func:`set_box_size`,
:func:`set_box_padding`, :func:`remove_protein`, :func:`clear_boxes` and
:func:`detect_row_boxes`; a polarity change re-quantifies its image and
:func:`requantify` every image of a legacy project, or the images
:func:`unassessed_images` names.
Removing an image removes its bands and changes no other image. Undo and redo
restore a committed state whole, which met the invariant, and recompute
nothing. A project quantified before #83 keeps the legacy method
(``global_median``) until :func:`requantify`: each band's level is its image's
median, its mode ``global_median``, its spread 0, and its net floors each
pixel at 0, so edits of a legacy project keep the legacy invariant. A project
saved before #112 holds no ``possibly_clipped`` on the bands of an image that
check assesses (a lossy, colour or CMYK image of known bit depth): the one
exception to the invariant, until an edit re-quantifies the image or
:func:`requantify` does (:func:`unassessed_images`).

Molecular weights (#58). A band's ``apparent_mw`` is its image's calibration
(:func:`~proteia.core.mwcal.calibration_for`: the image's register group,
fitted from its points) at the centre of its box, at the protein's box size,
padding included; None where the image has no curve or the centre lies outside
its range. A membrane's ``fit_method`` and ``fit_quality`` are
:func:`~proteia.core.mwcal.fit_method` and :func:`~proteia.core.mwcal.fit_quality`
of its points. Like nets, they are recomputed by the operation that
invalidates them, in the same change, and loading never recomputes them:
:func:`_refresh_mw` refits a membrane whose calibration points or marker links
changed, and rewrites a band's MW exactly when the band is new, its box moved
or was resized, or its image's curve changed; a band left where it was keeps
what it holds. The operations that place, move or resize boxes
(:func:`place_box`, :func:`move_box`, :func:`set_box_size`,
:func:`set_box_padding`, :func:`detect_row_boxes`), change a calibration or a
marker link (:func:`set_marker_image`, :func:`set_ladder`,
:func:`add_calibration_point`, :func:`edit_calibration_point`,
:func:`remove_calibration_point`, :func:`clear_calibration`,
:func:`set_ladder_points`) or remove an image call it. A calibration that
no #58 build has fitted (a project saved before #58) keeps its stored fit
and MWs until one of them refits it; that refit counts as a change to the
curve of every image of its membrane, so it rewrites every MW on the
membrane and drops the MW-guided records there. An
operation on calibration points or a marker link logs its register group's
fit after the change (:class:`CalibrationFit`), the images whose curve changed
and the records it dropped.

Band counts (#58, D10). A band a row commit places (:func:`detect_row_boxes`)
stores ``bands_found``: the bands its lane holds in the count window around its
box (:func:`~proteia.core.results.count_window`: its protein's MW tolerance
either way on its image's calibration, or the row box's rows without a
curve), read from the detector's peaks (:func:`~proteia.core.rowdetect.bands_in`).
It belongs to that detection and cannot be redone from the model, so an edit
that invalidates it clears it in the same change: the box moved or given
another lane by hand (:func:`move_box`, :func:`set_box_lane`), its image's
polarity (:func:`set_polarity`) or curve (:func:`_refresh_mw`) changed, its
protein's MW tolerance changed where the window came from it (its image has a
curve), or an MW-guided band's expected MW changed (:func:`edit_protein`).
Resizing or padding the boxes keeps it. Like nets, a cleared count is a field
of a band with an id and is not logged: a calibration operation's
``curves_changed`` names the images whose counts went.

A not-detected record (:class:`~proteia.core.model.UndetectedBand`) is a
detector's measurement that cannot be redone from the model alone, so an edit
that invalidates one drops it, in the same change, and logs it in full: a box
placed or moved into its lane replaces it (``replaced_undetected``), and a
polarity change or a lane table that cuts its lane drops it
(``dropped_undetected``). An MW-guided record searched a slot placed from the
protein's expected MW and its image's calibration, so a change to the
expected MW drops that protein's MW-guided records, and a change to the curve
of an image (its register group's calibration points, or the marker links
that make the group) drops those of every protein on it
(``dropped_undetected``). A removed protein or image takes its
proteins' records along, and the log lists them whole (``removed_undetected``),
since they have no ids; clearing a protein's boxes (:func:`clear_boxes`)
drops its records too. Removing a box never creates a record: the lane becomes
"not measured". Records are written by a row commit (:func:`detect_row_boxes`),
whose outcome in each lane replaces the record there
(``undetected_written``, ``dropped_undetected``).

Functions return ids or small frozen dataclasses, never model objects. Typed text
follows :mod:`proteia.core.names`; box placement follows :mod:`proteia.core.boxes`.

Besides the log entry that the session logs for every commit, the operations
log (Python's :mod:`logging`, at INFO: the session log of #137) the fallbacks a
committed change took, as it first took them: a band whose background was
measured another way (``asymmetric`` or ``image``: its ring cut short), a band
whose over-exposure could not be checked (its image's warnings or unknown bit
depth say why, and whether it is assessed near the limit instead), an image
imported with warnings, a row box whose detector warned or left lanes it
could not locate, a calibration point whose click found nothing to snap
to (the clicked position was kept), and a ladder ruler applied from a
proposal whose labels may be one band off (its gap is low); and the exports
written. None of it changes what they do.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import logging
import math
import shutil
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Concatenate, Final

import numpy as np
from pydantic import JsonValue, ValidationError

from proteia.core import boxes, export, ladders, mwcal, record, results, rowdetect, storage
from proteia.core.analyze import ReduceMethod, StatisticsSetting, statistics_setting
from proteia.core.export import (
    BUNDLE_RECORD_FILE,
    DEFAULT_CHART_FORMATS,
    LANE_COLUMNS,
    MIN_NAME_ROOM,
    ChartFormat,
    lane_table_bytes,
)
from proteia.core.grow import NOISE_K, REL_THRESHOLD, grow_box
from proteia.core.imaging import (
    UNTRUSTED_WARNINGS,
    assess_processed,
    clipping_depth,
    load_image,
    possible_clipping_depth,
    reads_as_palette,
)
from proteia.core.model import (
    DETECTING_SOURCES,
    IMAGE_SUFFIXES,
    LEGACY_BACKGROUND_METHOD,
    LOCAL_BACKGROUND_METHOD,
    Band,
    Batch,
    Box,
    BoxPadding,
    BoxSize,
    CalibrationPoint,
    CalibrationPointSource,
    FitMethod,
    ImageKind,
    ImageRef,
    ImageWarning,
    LadderSide,
    Lane,
    Membrane,
    Polarity,
    Project,
    ProposalSource,
    Protein,
    Rect,
    Region,
    Role,
    UndetectedBand,
    UndetectedReason,
    UnknownIdError,
    apply_change,
    format_timestamp,
    lane_number,
    lanes_phrase,
    overlaps,
)
from proteia.core.names import (
    TextError,
    clean_optional,
    clean_text,
    name_key,
    resolve_label,
    unify_spellings,
)
from proteia.core.plotspec import ErrorType
from proteia.core.project import (
    anchoring_lanes,
    lane_anchor_ids,
    lane_anchors,
    lane_pitch,
    lane_positions,
    lanes_run_right_to_left,
    propose_lane,
    spine_axes,
)
from proteia.core.quantify import (
    RING_CLAMP,
    BandBackground,
    NetClamp,
    band_backgrounds,
    estimate_background,
    is_clipped,
    is_possibly_clipped,
    net_signal,
    saturation_level,
)
from proteia.core.results import Results
from proteia.core.session import (
    ErrorCode,
    OperationError,
    ProjectSession,
    new_project,
    open_project,
)

__all__ = [
    "KEEP",
    "LANE_TABLE_FILE",
    "LANE_TABLE_RECORD_FILE",
    "CalibrationFit",
    "CalibrationUpdate",
    "Cascade",
    "ClearedBoxes",
    "ComputedView",
    "ErrorCode",
    "ExportBundle",
    "Keep",
    "LadderFit",
    "LaneInput",
    "LanesUpdate",
    "OperationError",
    "PaddingChange",
    "ProjectSession",
    "Restored",
    "RowPlacement",
    "add_calibration_point",
    "add_protein",
    "calibration_fit",
    "clear_boxes",
    "clear_calibration",
    "compute",
    "compute_view",
    "detect_row_boxes",
    "edit_calibration_point",
    "edit_protein",
    "export_bundle",
    "export_lane_table",
    "import_image",
    "move_box",
    "new_project",
    "open_project",
    "place_box",
    "proposal_json",
    "propose_ladder",
    "redo",
    "remove_box",
    "remove_calibration_point",
    "remove_image",
    "remove_protein",
    "remove_undetected",
    "requantify",
    "save",
    "set_box_lane",
    "set_box_padding",
    "set_box_size",
    "set_ladder",
    "set_ladder_points",
    "set_lanes",
    "set_marker_image",
    "set_polarity",
    "set_reference_condition",
    "snap_ladder",
    "unassessed_images",
    "undo",
]

# The lane table export and its record: fixed names chosen here, never from client text.
LANE_TABLE_FILE: Final = "lane-table.csv"
LANE_TABLE_RECORD_FILE: Final = "lane-table.record.json"
# A protein name must not read as one of the lane table's own columns.
_RESERVED_KEYS: Final = frozenset(name_key(column) for column in LANE_COLUMNS)
# How many names an export tries for its new folder: the plain one, then numbered.
_FOLDER_ATTEMPTS: Final = 1000
# A box placed, moved or resized may overlap another protein's box on its
# image by at most this share of the smaller box's area (#114): more, and the
# two would measure the same band.
COVER_SHARE: Final = 0.5
# How a band's background was measured when its ring was cut short
# (quantify.band_backgrounds), as the session log words it.
_CUT_BACKGROUNDS: Final = {
    "asymmetric": "too few paired pixels around its box: a robust plane over its ring",
    "image": "too little membrane around its box: the image's median",
}

_log = logging.getLogger(__name__)


class Keep(Enum):
    """The type of :data:`KEEP`."""

    KEEP = "keep"


# "Leave unchanged", where None is a real value (no expected MW, no reference).
KEEP: Final = Keep.KEEP


@dataclass(frozen=True)
class Cascade:
    """What a removal took with it."""

    removed: tuple[str, ...]  # every removed id: the object, then proteins, bands, membrane
    detached_targets: tuple[str, ...]  # targets that lost a loading control
    unpaired_images: tuple[str, ...]  # images whose marker_image_id was cleared
    # calibration lost points: refitted, or left without a curve
    unfitted_membranes: tuple[str, ...]


@dataclass(frozen=True)
class LaneInput:
    """One row of the lane table as the user typed it."""

    condition: str
    sample: str | None = None
    included: bool = True


@dataclass(frozen=True)
class LanesUpdate:
    """What :func:`set_lanes` did beyond storing the table."""

    respelled: tuple[int, ...]  # lanes whose condition or sample took an existing spelling
    reference_cleared: bool  # the reference no longer named a lane and was cleared
    # (protein id, lane index, band index) of each not-detected record in a dropped lane
    dropped_undetected: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class RowPlacement:
    """What :func:`detect_row_boxes` did, lane by lane (lane indices 0-based).

    ``empty`` lists every lane the detector left empty, as ``(lane, reason,
    snr, expected_x)`` (:class:`~proteia.core.rowdetect.LaneDetection`), whatever
    the lane kept; of those, ``undetected_lanes`` got a not-detected record and
    ``unmeasured_lanes`` were left with neither a box nor a record, for the user
    to place by hand (a box placed there stays through the next drag over the
    row as long as that drag finds no band there either). Of the unmeasured
    lanes, ``unlocated_lanes`` are those where no band reaches the detection
    limit but whose slot rests on one band alone (the row found a band in one
    lane only; the lanes already placed on the image check that band's lane,
    not the slots), so one band cannot show where they lie and no record is
    written there.

    ``band_ids`` names, per declared lane, the lane's band-index-0 box after
    the commit, whichever way it got there: placed new, replaced in place
    (``replaced_band_ids``) or kept (``kept_lanes``); None in a lane left
    without one.

    ``flags`` are the detector's warnings and ``notes`` its diagnostics; of
    them, ``doubtful_lanes`` and its note (the row's lane numbers to be
    checked) only while lanes on the image have not checked the lane of every
    band found (:func:`detect_row_boxes`).

    ``kept_lanes`` are the lanes whose box was kept as it was: one edited by
    hand, or one the user placed (source ``click`` or ``manual``) in a lane
    where no band was found.

    ``remeasured`` lists the bands of the other proteins on the image whose net
    the row changed (every ring on the image leaves out the row's boxes), as
    ``(band id, net before, net after)``, protein by protein and band by band
    as stored; ``largest_change`` is the band among them whose net changed by
    the largest share of its net before, ``(band id, |after - before| /
    before)``, None when none had a net before.
    """

    band_ids: tuple[str | None, ...]  # per declared lane: its first-band box after, or None
    box_size: BoxSize  # the protein's shared size after the change
    kept_lanes: tuple[int, ...]
    replaced_band_ids: tuple[str, ...]  # boxes nobody edited that took the band found, in place
    # boxes a detector placed (row_box, mw_guided) that nobody edited, where no band was found
    removed_band_ids: tuple[str, ...]
    undetected_lanes: tuple[int, ...]
    unmeasured_lanes: tuple[int, ...]
    empty: tuple[tuple[int, str, float, float], ...]
    flags: tuple[str, ...]  # the detector's warnings (rowdetect.WARNING_FLAGS)
    notes: tuple[str, ...]  # the detector's diagnostics, lanes counted from 1
    right_to_left: bool  # lanes read from the box's right end, as those on the image run
    remeasured: tuple[tuple[str, float, float], ...] = ()  # other proteins' nets it changed
    largest_change: tuple[str, float] | None = None  # (band id, share of its net before)
    unlocated_lanes: tuple[int, ...] = ()  # no_band lanes whose slot rests on one band alone


@dataclass(frozen=True)
class Restored:
    """What :func:`undo` took back or :func:`redo` made again.

    ``seq`` and ``action`` name the original change's log entry. The ids that
    went and came back are in the order :meth:`~proteia.core.model.Project.iter_ids`
    gives: by kind (membranes, images, proteins, bands), each kind as stored
    (images membrane by membrane, bands protein by protein and by lane), so not
    sorted by id. Not-detected records, which have no ids, are listed by
    (protein id, lane index, band index), as stored. An object or record in
    both states whose fields changed (a moved box, a replaced record) is in no
    list.
    """

    seq: int
    action: str
    removed: tuple[str, ...]
    restored: tuple[str, ...]
    undetected_removed: tuple[tuple[str, int, int], ...]
    undetected_restored: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class ClearedBoxes:
    """What :func:`clear_boxes` removed."""

    band_ids: tuple[str, ...]
    undetected: tuple[tuple[int, int], ...]  # (lane index, band index) of each dropped record


@dataclass(frozen=True)
class PaddingChange:
    """What :func:`set_box_padding` did.

    ``net_change`` is the smallest and the largest ``after / before - 1`` of
    the protein's nets, over its bands with a net above 0 before; None without
    one, or when nothing changed. ``remeasured`` and ``largest_change`` are the
    other proteins' nets on the image the change moved, as for
    :class:`RowPlacement`.
    """

    box_size: BoxSize  # every box of the protein after: the fitted size plus the padding
    fitted_size: BoxSize
    padding: BoxPadding
    net_change: tuple[float, float] | None
    edge_shifted: tuple[str, ...]  # its boxes the image edge kept from growing evenly
    overlapping: tuple[str, ...]  # other proteins' boxes on the image its boxes newly overlap
    remeasured: tuple[tuple[str, float, float], ...] = ()  # other proteins' nets it changed
    largest_change: tuple[str, float] | None = None  # (band id, share of its net before)


def _finite(value: float | None) -> float | None:
    """``value`` for JSON, which has no infinity: None when it is not finite."""
    return value if value is not None and math.isfinite(value) else None


def _display_mw(z: float) -> float:
    """``10 ** z``, a range end for display; inf past the largest float (only a
    ladder labelled near it reaches that)."""
    try:
        return 10.0**z
    except OverflowError:
        return math.inf


@dataclass(frozen=True)
class LadderFit:
    """One ladder of a register group as fitted (:class:`~proteia.core.mwcal.Ladder`)."""

    side: str  # "left" or "right"
    x: float | None  # the median of its points' x; None when none has one
    points: int
    # D2: the largest relative MW disagreement of an interior point with the line
    # between its neighbours; None with two points; inf past the largest float
    quality: float | None
    worst_mw: float | None  # the point it names, as labelled
    predicted_mw: float | None  # where its neighbours put it
    two_point: bool
    range: tuple[float, float]  # the lowest and highest MW in range, for display

    def as_json(self) -> dict[str, JsonValue]:
        """JSON-plain: an infinite quality is null, with ``quality_infinite``."""
        return {
            "side": self.side,
            "x": self.x,
            "points": self.points,
            "quality": _finite(self.quality),
            "quality_infinite": self.quality is not None and not math.isfinite(self.quality),
            "worst_mw": self.worst_mw,
            "predicted_mw": self.predicted_mw,
            "two_point": self.two_point,
            "range": [_finite(end) for end in self.range],
        }


@dataclass(frozen=True)
class CalibrationFit:
    """A register group's calibration as fitted
    (:class:`~proteia.core.mwcal.Calibration`): what a calibration operation
    logs and the page shows.

    ``offset_px`` is how much lower the right ladder runs than the left one
    and ``tilt_deg`` the slope that gives across the blot; ``disagreement`` is
    how far the two ladders disagree once that offset is taken out, at the MW
    ``disagreement_mw``. All three are None with one ladder. ``ignored`` lists
    the sides not used and why (``one_point``, ``few_shared_mws``)."""

    image_ids: tuple[str, ...]  # the register group, in membrane order
    method: str
    ladders: tuple[LadderFit, ...]  # 1 or 2, left first
    range: tuple[float, float]  # the MWs in range (both ladders' with two), for display
    offset_px: float | None
    tilt_deg: float | None
    disagreement: float | None  # inf past the largest float (degenerate ladders only)
    disagreement_mw: float | None
    ignored: tuple[tuple[str, str], ...]  # (side, reason)

    def as_json(self) -> dict[str, JsonValue]:
        """JSON-plain (it has no infinity): an infinite quality or disagreement
        is null, with ``quality_infinite`` or ``disagreement_infinite`` true."""
        disagreement = self.disagreement
        return {
            "image_ids": list(self.image_ids),
            "method": self.method,
            "ladders": [ladder.as_json() for ladder in self.ladders],
            "range": [_finite(end) for end in self.range],
            "offset_px": self.offset_px,
            "tilt_deg": self.tilt_deg,
            "disagreement": _finite(disagreement),
            "disagreement_infinite": disagreement is not None and not math.isfinite(disagreement),
            "disagreement_mw": self.disagreement_mw,
            "ignored": [[side, reason] for side, reason in self.ignored],
        }


@dataclass(frozen=True)
class CalibrationUpdate:
    """What a calibration operation did.

    ``point`` is the point as stored, with ``snapped`` (whether the given y
    was moved onto a band or edge), for an operation that adds or edits one;
    ``fit`` the register group's calibration after (None without a curve);
    ``curves_changed`` the images whose curve appeared, went or moved (every
    image of a membrane whose fit from before #58 was replaced), in
    membrane order; and ``dropped_undetected`` the MW-guided not-detected
    records dropped from the proteins on them, as (protein id, lane index,
    band index)."""

    point: dict[str, JsonValue] | None
    fit: CalibrationFit | None
    curves_changed: tuple[str, ...]
    dropped_undetected: tuple[tuple[str, int, int], ...]
    # set_ladder_points: the points applied, as stored, each with how it was
    # placed; and whether they went to the other side than the one given.
    points: tuple[dict[str, JsonValue], ...] = ()
    sides_swapped: bool = False


# --- Common machinery ---


def _locked[**P, R](
    fn: Callable[Concatenate[ProjectSession, P], R],
) -> Callable[Concatenate[ProjectSession, P], R]:
    """Run an operation under the session lock. A refusal leaves the pixel cache
    as it was, even if the operation read pixels before refusing."""

    @functools.wraps(fn)
    def locked(session: ProjectSession, /, *args: P.args, **kwargs: P.kwargs) -> R:
        with session.transaction():
            return fn(session, *args, **kwargs)

    return locked


def _invalid(message: str, *, ids: Sequence[str] = ()) -> OperationError:
    return OperationError(ErrorCode.INVALID_INPUT, message, ids=ids)


def _prepare[T](session: ProjectSession, change: Callable[[Project], T]) -> tuple[Project, T]:
    """Run ``change`` on a copy of the committed project and re-validate it; a
    ``ValidationError`` becomes ``INVALID_INPUT`` with its first message."""
    try:
        return apply_change(session.project, change)
    except ValidationError as exc:
        message = str(exc.errors(include_url=False)[0]["msg"]).removeprefix("Value error, ")
        raise _invalid(message) from exc


_Params = Mapping[str, JsonValue]  # a log entry's params


def _apply[T](
    session: ProjectSession,
    action: str,
    change: Callable[[Project], T],
    params: Callable[[T], _Params],
    **commit_kw,
) -> T:
    """:func:`_prepare`, then commit with the log entry ``params(result)`` unless
    the project did not change."""
    before = session.project
    new, result = _prepare(session, change)
    if new == before:  # a no-op: nothing committed, no entry, no hook
        return result
    session._commit(new, action=action, params=params(result), **commit_kw)
    _log_fallbacks(session, before)
    return result


def _unchecked_why(image: ImageRef) -> str:
    """Why over-exposure cannot be checked on ``image``, and what runs instead."""
    codes = [w.code for w in image.import_warnings if w.code in UNTRUSTED_WARNINGS]
    why = ", ".join(codes) if codes else "unknown bit depth"
    if possible_clipping_depth(image.bit_depth, image.import_warnings) is not None:
        return f"{why}; assessed near the detector limit instead"
    return f"{why}; not assessed either"


def _log_fallbacks(session: ProjectSession, before: Project) -> None:
    """Log the fallbacks of the change just committed over ``before``, each as
    it is first taken: the bands whose background is now measured another way
    than before (their ring cut short), and the bands new or newly unchecked
    for over-exposure, per image."""
    if not _log.isEnabledFor(logging.INFO):
        return
    old = {
        band.id: (band.background_mode, band.clipped)
        for protein in before.batch.proteins
        for band in protein.bands
    }
    batch, name = session.project.batch, session.folder.name
    for image in batch.iter_images():
        cut: dict[str, list[str]] = {}
        unchecked: list[str] = []
        for protein in batch.proteins:
            if protein.image_id != image.id:
                continue
            for band in protein.bands:
                was = old.get(band.id)
                mode = band.background_mode
                if mode in _CUT_BACKGROUNDS and (was is None or was[0] != mode):
                    cut.setdefault(mode, []).append(band.id)
                if band.clipped is None and (was is None or was[1] is not None):
                    unchecked.append(band.id)
        for mode, band_ids in cut.items():
            _log.info(
                "in %r: the background of %s on %s was measured another way, %s (%s)",
                name,
                ", ".join(band_ids),
                image.id,
                mode,
                _CUT_BACKGROUNDS[mode],
            )
        if unchecked:
            _log.info(
                "in %r: over-exposure could not be checked on %s on %s (%s)",
                name,
                ", ".join(unchecked),
                image.id,
                _unchecked_why(image),
            )


def _size(size: BoxSize) -> dict[str, JsonValue]:
    return {"width": size.width, "height": size.height}


def _cascade(cascade: Cascade) -> dict[str, JsonValue]:
    return {
        "removed": list(cascade.removed),
        "detached_targets": list(cascade.detached_targets),
        "unpaired_images": list(cascade.unpaired_images),
        "unfitted_membranes": list(cascade.unfitted_membranes),
    }


def _undetected_json(protein_id: str, record: UndetectedBand) -> dict[str, JsonValue]:
    """A not-detected record as a log entry names it: whole, since it has no id."""
    return {
        "protein_id": protein_id,
        "lane_index": record.lane_index,
        "band_index": record.band_index,
        "reason": record.reason.value,
        "snr": record.snr,
        "threshold": record.threshold,
        "region": list(record.region.rect()),
        "source": record.source.value,
    }


def _drop_undetected_where(
    protein: Protein, drop: Callable[[UndetectedBand], bool]
) -> list[dict[str, JsonValue]]:
    """Remove a draft protein's records for which ``drop`` is true; return their
    log forms in the stored order (plain values, as a change should return)."""
    kept: list[UndetectedBand] = []
    dropped: list[dict[str, JsonValue]] = []
    for u in protein.undetected:
        if drop(u):
            dropped.append(_undetected_json(protein.id, u))
        else:
            kept.append(u)
    if dropped:
        protein.undetected = kept
    return dropped


def _drop_mw_guided(protein: Protein) -> list[dict[str, JsonValue]]:
    """Remove a draft protein's MW-guided records: the slot each one examined was
    placed from the expected MW and its image's calibration curve (its register
    group's), so a change to either leaves the record about a slot nobody
    looked in."""
    return _drop_undetected_where(protein, lambda record: record.source is ProposalSource.MW_GUIDED)


def _drop_undetected(
    protein: Protein, lane_index: int, band_index: int
) -> dict[str, JsonValue] | None:
    """Remove a draft protein's record at (lane, band index): its log form, or None."""
    dropped = _drop_undetected_where(
        protein, lambda record: (record.lane_index, record.band_index) == (lane_index, band_index)
    )
    return dropped[0] if dropped else None


# What a band built in a change holds until _quantify_image, later in the same
# change, writes its values.
_UNQUANTIFIED: Final = {
    "net": 0.0,
    "background_level": 0.0,
    "background_mode": "image",
    "background_spread": 0.0,
}


def _quantify_image(draft: Project, image_id: str, array: np.ndarray) -> None:
    """The one writer of a band's net, background fields and clipping flags:
    quantify every band on one image, of every protein, together (protein order,
    then band order; the results do not depend on it), since each band's ring
    leaves out every box on the image. ``array`` is the image's analysis array.

    Under the project's legacy method (``global_median``) each level is the
    image's median and each net floors every pixel at 0, as before #83;
    otherwise :func:`~proteia.core.quantify.band_backgrounds` measures each
    level (falling back to the image's median) and the net floors the box total
    (:data:`~proteia.core.quantify.RING_CLAMP`). An image without a limit the
    clipping check can trust leaves its bands unchecked (``clipped`` None); if
    its range is known (a lossy, colour or CMYK-converted image), each band is
    assessed instead (``possibly_clipped``, #112), which is None everywhere else.
    """
    batch = draft.batch
    image = batch.find_image(image_id)
    placed = [
        (protein.box_size, band)
        for protein in batch.proteins
        if protein.image_id == image_id
        for band in protein.bands
    ]
    if not placed:
        return
    dark_on_light = image.polarity.dark_on_light
    clamp: NetClamp
    if draft.background_method == LEGACY_BACKGROUND_METHOD:
        legacy = BandBackground(level=image.background, mode="global_median", spread=0.0)
        found, clamp = [legacy] * len(placed), "pixel"
    else:
        found = band_backgrounds(
            array,
            [band.box.rect(size) for size, band in placed],
            [(size.width, size.height) for size, _ in placed],
            dark_on_light=dark_on_light,
            integral=image.bit_depth is not None,
            fallback=image.background,
        )
        clamp = RING_CLAMP
    depth = clipping_depth(image.bit_depth, image.import_warnings)
    near_depth = possible_clipping_depth(image.bit_depth, image.import_warnings)
    for (size, band), background in zip(placed, found, strict=True):
        band.net = net_signal(
            array, band.box, size, background.level, dark_on_light=dark_on_light, clamp=clamp
        )
        band.background_level = background.level
        band.background_mode = background.mode
        band.background_spread = background.spread
        band.clipped = is_clipped(
            array, band.box, size, bit_depth=depth, dark_on_light=dark_on_light
        )
        band.possibly_clipped = is_possibly_clipped(
            array, band.box, size, bit_depth=near_depth, dark_on_light=dark_on_light
        )


def _bands_on(batch: Batch, image_id: str) -> list[str]:
    """The ids of every band on one image, of every protein."""
    return [
        band.id
        for protein in batch.proteins
        if protein.image_id == image_id
        for band in protein.bands
    ]


def _pixels_left(
    session: ProjectSession, image_id: str, removed: Collection[str]
) -> np.ndarray | None:
    """The image's pixels, for a removal of the bands ``removed`` from it: the
    bands that stay are re-quantified, since their rings no longer leave out the
    removed boxes. None when no band stays: there is nothing to quantify."""
    gone = set(removed)
    if all(band_id in gone for band_id in _bands_on(session.project.batch, image_id)):
        return None
    return session.pixels(image_id)


def _set_box(band: Band, rect: Rect) -> None:
    # Its apparent MW follows in the same change (_refresh_mw).
    band.box = Box(x=rect[0], y=rect[1])


# --- Molecular weights: the one writer ---

_Fitted = mwcal.Calibration | mwcal.NoCalibration


def _fits(project: Project) -> dict[str, _Fitted]:
    """Each image's register group, fitted, by image id."""
    fits: dict[str, _Fitted] = {}
    for membrane in project.batch.membranes:
        for group, fitted in mwcal.calibrations(membrane).items():
            fits.update(dict.fromkeys(group, fitted))
    return fits


def _curve(fitted: _Fitted | None) -> object:
    """What an image's curve depends on, to tell whether it changed: the
    method, the ladders as marked (their x, points and sources) and the sides
    not used; None without a curve. The group itself is left out: an image
    joining another's group changes neither's curve unless its points do."""
    if isinstance(fitted, mwcal.Calibration):
        return fitted.method, fitted.ladders, fitted.ignored
    return None


def _point_key(point: CalibrationPoint) -> tuple:
    return point.side.value, point.y, point.mw, point.source.value, point.image_id, point.x


def _calibration_inputs(membrane: Membrane) -> tuple:
    """What a membrane's fit depends on: its points and its marker links."""
    return (
        sorted(_point_key(point) for point in membrane.calibration.points),
        [(image.id, image.marker_image_id) for image in membrane.images if image.marker_image_id],
    )


def _fitted_before_58(membrane: Membrane, batch: Batch) -> bool:
    """Whether a membrane holds a fit or apparent MWs no #58 build computed
    (a project saved before #58, or a hand-made file): a stored ``log_linear``
    beside a curve, a fit quality or an MW. A #58 build stores ``log_linear``
    only while no image of the membrane has a curve, and then no fit quality
    and no MW."""
    calibration = membrane.calibration
    if calibration.fit_method is not FitMethod.LOG_LINEAR:
        return False
    if (
        calibration.fit_quality is not None
        or mwcal.fit_method(membrane) is not FitMethod.LOG_LINEAR
    ):
        return True
    images = {image.id for image in membrane.images}
    return any(
        band.apparent_mw is not None
        for protein in batch.proteins
        if protein.image_id in images
        for band in protein.bands
    )


def _apparent_mw(fitted: _Fitted, box: Box, size: BoxSize) -> float | None:
    """The MW at a box's centre (at the size it is quantified at, padding
    included); None without a curve or outside its range."""
    if not isinstance(fitted, mwcal.Calibration):
        return None
    try:
        mw = fitted.mw_at(box.x + size.width / 2, box.y + size.height / 2)
    except OverflowError:  # past the largest float: only a ladder labelled near it
        return None
    return mw if mw is not None and 0.0 < mw < math.inf else None


@dataclass(frozen=True)
class _MwRefresh:
    """What :func:`_refresh_mw` did beyond the values it wrote."""

    curves_changed: tuple[str, ...]  # images whose curve changed, as its steps 2 and 3 say
    dropped: tuple[dict[str, JsonValue], ...]  # the MW-guided records dropped, whole


def _refresh_mw(committed: Project, draft: Project) -> _MwRefresh:
    """The one writer of the molecular-weight values (#58), at the end of a
    change: diff ``draft`` against ``committed`` and

    1. refit each membrane whose calibration points or marker links changed:
       its ``fit_method`` (:func:`~proteia.core.mwcal.fit_method`), then its
       ``fit_quality`` (:func:`~proteia.core.mwcal.fit_quality`);
    2. recompute the ``apparent_mw`` of every band that is new, whose box
       moved or was resized, or whose image's curve changed (appeared, went,
       or its ladders changed);
    3. drop the MW-guided not-detected records of every protein on an image
       whose curve changed, and clear the band counts (``bands_found``) of
       every band there: their slot and count window came from the old curve.

    A refit of a fit no #58 build computed (:func:`_fitted_before_58`)
    changes the curve of every image of its membrane: their stored MWs came
    from that older fit, not from the curve they are read with now.

    A calibration whose fit quality would be infinite (points labelled
    hundreds of decades apart) is refused (``INVALID_INPUT``): no file can
    store it."""
    before = {membrane.id: membrane for membrane in committed.batch.membranes}
    refitted_older: set[str] = set()  # the images of membranes whose older fit was replaced
    for membrane in draft.batch.membranes:
        stored = before.get(membrane.id)
        if stored is not None and _calibration_inputs(stored) == _calibration_inputs(membrane):
            continue
        if stored is not None and _fitted_before_58(stored, committed.batch):
            refitted_older.update(image.id for image in membrane.images)
        calibration = membrane.calibration
        calibration.fit_method = mwcal.fit_method(membrane)
        quality = mwcal.fit_quality(membrane)
        if quality is not None and not math.isfinite(quality):
            raise _invalid(
                f"membrane {membrane.id}: its calibration points' MWs are too far apart for"
                " their positions to be fitted; check their labels"
            )
        calibration.fit_quality = quality
    old, new = _fits(committed), _fits(draft)
    changed = [
        image.id
        for image in draft.batch.iter_images()
        if image.id in refitted_older or _curve(new[image.id]) != _curve(old.get(image.id))
    ]
    moved_curves = set(changed)
    boxes_before = {
        band.id: (band.box, protein.box_size)
        for protein in committed.batch.proteins
        for band in protein.bands
    }
    dropped: list[dict[str, JsonValue]] = []
    for protein in draft.batch.proteins:
        fitted, size = new[protein.image_id], protein.box_size
        for band in protein.bands:
            if protein.image_id in moved_curves or boxes_before.get(band.id) != (band.box, size):
                band.apparent_mw = _apparent_mw(fitted, band.box, size)
        if protein.image_id in moved_curves:
            dropped.extend(_drop_mw_guided(protein))
            for band in protein.bands:
                band.bands_found = None
    return _MwRefresh(curves_changed=tuple(changed), dropped=tuple(dropped))


def _refresh_box_mws(committed: Project, draft: Project) -> None:
    """:func:`_refresh_mw` after an edit of boxes, which changes no
    calibration: the MWs of the bands it placed, moved or resized."""
    if _refresh_mw(committed, draft).curves_changed:  # unreachable: no point or link changed
        raise RuntimeError("an edit of boxes changed a calibration curve")


def _remeasured(
    before: Batch, after: Batch, image_id: str, protein_id: str
) -> tuple[tuple[tuple[str, float, float], ...], tuple[str, float] | None]:
    """The other proteins' bands on an image whose net a change of one protein's
    boxes moved (their rings leave out its boxes), as ``(band id, net before,
    net after)``, protein by protein and band by band as stored; and the band
    among them whose net changed by the largest share of its net before,
    ``(band id, |after - before| / before)``, None when none had a net before.
    A no-op, ``after`` the same as ``before``, moves none."""
    nets = {
        band.id: band.net
        for other in after.proteins
        if other.image_id == image_id and other.id != protein_id
        for band in other.bands
    }
    remeasured = tuple(
        (band.id, band.net, nets[band.id])
        for other in before.proteins
        if other.image_id == image_id and other.id != protein_id
        for band in other.bands
        if nets[band.id] != band.net
    )
    shares = [(abs(new - old) / old, band_id) for band_id, old, new in remeasured if old > 0]
    largest = max(shares, key=lambda share: share[0], default=None)
    return remeasured, None if largest is None else (largest[1], largest[0])


# --- Input checks ---


def _int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(f"{what} must be an integer, not {value!r}")
    return value


def _member[E: Enum](cls: type[E], value: object, what: str) -> E:
    try:
        return cls(value)
    except ValueError as exc:
        choices = ", ".join(repr(member.value) for member in cls)
        raise _invalid(f"{what} must be one of {choices}, not {value!r}") from exc


def _statistics(value: object) -> StatisticsSetting:
    """A statistics setting from its model or raw strings; an unknown key or
    value is refused (``INVALID_INPUT``)."""
    try:
        return statistics_setting(value)  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        raise _invalid(str(exc)) from exc


def _clean(text: object, what: str) -> str:
    if not isinstance(text, str):
        raise _invalid(f"{what} must be text, not {text!r}")
    try:
        return clean_text(text)
    except TextError as exc:
        raise OperationError(ErrorCode(exc.code), f"{what} {exc}") from exc


def _clean_optional(text: object, what: str) -> str | None:
    if text is not None and not isinstance(text, str):
        raise _invalid(f"{what} must be text or None, not {text!r}")
    try:
        return clean_optional(text)
    except TextError as exc:
        raise OperationError(ErrorCode(exc.code), f"{what} {exc}") from exc


def _condition(text: object, labels: Sequence[str]) -> str:
    """The lane label a typed condition names (its stored spelling)."""
    cleaned = _clean(text, "condition")
    label = resolve_label(cleaned, labels)
    if label is None:
        raise OperationError(ErrorCode.UNKNOWN_CONDITION, f"no lane has the condition {cleaned!r}")
    return label


def _protein_name(batch: Batch, name: object, *, protein_id: str | None = None) -> str:
    """The stored form of a protein name: unique among the other proteins by
    :func:`~proteia.core.names.name_key` and not a lane-table column."""
    cleaned = _clean(name, "protein name")
    key = name_key(cleaned)
    clash = [p.id for p in batch.proteins if p.id != protein_id and name_key(p.name) == key]
    if clash:
        raise OperationError(
            ErrorCode.DUPLICATE_NAME,
            f"protein {clash[0]} already has the name {cleaned!r}"
            " (names ignore case and look-alike characters)",
            ids=clash,
        )
    if key in _RESERVED_KEYS:
        raise OperationError(
            ErrorCode.RESERVED_NAME, f"{cleaned!r} is the name of a lane-table column"
        )
    # The lane table follows each protein's column with "<name> clipped".
    for other in batch.proteins:
        if other.id != protein_id and (
            key == name_key(f"{other.name} clipped")
            or name_key(f"{cleaned} clipped") == name_key(other.name)
        ):
            raise OperationError(
                ErrorCode.RESERVED_NAME,
                f"{cleaned!r} and {other.name!r} would collide in the lane table, which"
                " names each protein's clipping column '<name> clipped'",
                ids=[other.id],
            )
    return cleaned


def _expected_mw(value: object) -> float | None:
    if value is None:
        return None
    number: float | None = None
    if not isinstance(value, bool) and isinstance(value, int | float):
        try:
            number = float(value)  # an int too large for a float raises OverflowError
        except OverflowError:
            number = None
    if number is None or not math.isfinite(number) or number <= 0:
        raise _invalid(f"expected molecular weight must be a positive number of kDa, not {value!r}")
    return number


def _mw_tolerance(value: object) -> float:
    """An MW tolerance as a share: a number above 0 and below 1 (0.1 is ±10%)."""
    number: float | None = None
    if not isinstance(value, bool) and isinstance(value, int | float):
        try:
            number = float(value)
        except OverflowError:
            number = None
    if number is None or not 0.0 < number < 1.0:  # NaN fails both
        raise _invalid(
            f"MW tolerance must be a share above 0 and below 1 (0.1 is ±10%), not {value!r}"
        )
    return number


def _count_cleared_by_tolerance(batch: Batch, protein: Protein) -> bool:
    """Whether a new MW tolerance clears the protein's band counts: their count
    window was sized from it, as it is wherever its image has a curve (a curve
    change clears them, so a count there was made with that curve); without
    one, the window is the row box's rows."""
    membrane = batch.membrane_of(protein.image_id)
    return isinstance(mwcal.calibration_for(membrane, protein.image_id), mwcal.Calibration)


def _loading_controls(batch: Batch, protein_id: str | None, role: Role, ids: object) -> list[str]:
    """A checked list of loading-control ids (its order is the series order)."""
    if isinstance(ids, str) or not isinstance(ids, Sequence):
        raise _invalid("loading control ids must be a sequence of protein ids")
    chosen = list(ids)
    if not chosen:
        return []
    if role is not Role.TARGET:
        raise _invalid("only a target has loading controls")
    for lc_id in chosen:
        if batch.find_protein(lc_id).role is not Role.LOADING_CONTROL:
            raise _invalid(f"{lc_id} is not a loading control", ids=(lc_id,))
    if len(set(chosen)) != len(chosen):
        raise _invalid("a loading control is listed twice")
    if protein_id in chosen:
        raise _invalid(f"{protein_id} cannot be its own loading control", ids=(protein_id,))
    return chosen


def _box_size(size: object) -> BoxSize:
    if not isinstance(size, BoxSize):
        raise _invalid(f"box size must be a BoxSize, not {size!r}")
    return size


def _within(size: BoxSize, image: ImageRef, pad: BoxPadding = boxes.NO_PADDING) -> BoxSize:
    """``size``, a protein's boxes under the padding ``pad``, if the image holds
    it; else ``SIZE_OUT_OF_BOUNDS``, naming the fitted size and the padding."""
    if size.width > image.width or size.height > image.height:
        raise OperationError(
            ErrorCode.SIZE_OUT_OF_BOUNDS,
            f"box size {boxes.size_words(size, pad)} exceeds the"
            f" {image.width}x{image.height} image {image.id}",
        )
    return size


def _fitting_size(size: object, image: ImageRef) -> BoxSize:
    return _within(_box_size(size), image)


# A padding's directions, as (BoxPadding field, in words, the BoxSize field it pads).
_PADDING: Final = (("across", "left and right", "width"), ("along", "above and below", "height"))
_WHERE: Final = {name: where for name, where, _ in _PADDING}


def _padded(fitted: BoxSize, pad: BoxPadding) -> BoxSize:
    """The boxes' size: ``fitted`` plus ``pad`` on each side."""
    return BoxSize(width=fitted.width + 2 * pad.across, height=fitted.height + 2 * pad.along)


def _loading_control_ids(batch: Batch) -> list[str]:
    return [p.id for p in batch.proteins if p.role is Role.LOADING_CONTROL]


def _implicit_users(batch: Batch, protein_id: str) -> list[str]:
    """Targets that normalize against ``protein_id`` without naming it: it is the
    batch's only loading control, and they chose none."""
    if _loading_control_ids(batch) != [protein_id]:
        return []
    return [p.id for p in batch.proteins if p.role is Role.TARGET and not p.loading_control_ids]


def _users(batch: Batch, protein_id: str) -> list[str]:
    """Every target normalized against ``protein_id``, named or implicit."""
    named = [p.id for p in batch.proteins if protein_id in p.loading_control_ids]
    return named + _implicit_users(batch, protein_id)


def _pin_single_loading_control(batch: Batch) -> list[str]:
    """Before a second loading control appears, write the single one into the
    targets that use it implicitly, so their normalization does not change (with
    two loading controls and none chosen, a target is not normalized at all).
    Returns the ids of the targets it pinned."""
    only = _loading_control_ids(batch)
    if len(only) != 1:
        return []
    pinned = []
    for protein in batch.proteins:
        if protein.role is Role.TARGET and not protein.loading_control_ids:
            protein.loading_control_ids = list(only)
            pinned.append(protein.id)
    return pinned


def _check_lane(
    protein: Protein, lane: int | None, n: int, taken: dict[int, str], *, proposed: bool
) -> int:
    """A lane the box may take: one of the ``n`` declared lanes, where the protein
    has no box yet. ``proposed`` words the refusal for a lane read from position."""
    if lane is None:  # position could not propose one after all
        raise OperationError(ErrorCode.LANE_REQUIRED, "choose the lane")
    number = lane_number(lane)
    where = f" (the box lies at lane {number}); choose the lane" if proposed else ""
    if not 0 <= lane < n:
        raise OperationError(
            ErrorCode.LANE_OUT_OF_RANGE, f"lane {number} is not one of the {n} lanes{where}"
        )
    if lane in taken:
        raise OperationError(
            ErrorCode.LANE_OCCUPIED,
            f"{protein.name!r} already has a box in lane {number}{where}",
            ids=[taken[lane]],
        )
    return lane


def _overlapped(rect: Rect, protein: Protein, *, skip: str | None = None) -> list[str]:
    """The protein's bands (other than ``skip``) whose box overlaps ``rect``."""
    return [
        band.id
        for band in protein.bands
        if band.id != skip and overlaps(rect, band.box.rect(protein.box_size))
    ]


def _overlap_share(rect: Rect, box: Rect) -> float:
    """How much of the smaller of two boxes' areas they share: 1.0 for a box
    wholly inside the other, whichever is larger."""
    w = min(rect[2], box[2]) - max(rect[0], box[0])
    h = min(rect[3], box[3]) - max(rect[1], box[1])
    smaller = min((r[2] - r[0]) * (r[3] - r[1]) for r in (rect, box))
    return max(0, w) * max(0, h) / smaller


def _covered(
    batch: Batch, protein: Protein, rects: Iterable[Rect]
) -> list[tuple[Protein, Band, float]]:
    """The other proteins' bands on ``protein``'s image whose box one of
    ``rects`` overlaps by more than :data:`COVER_SHARE` of the smaller box's
    area, with the largest share (:func:`_overlap_share`); protein by protein
    and band by band, as stored.

    Against the smaller box: a box wholly inside another protein's larger box
    shares all of it, and would measure its band as that box does; against
    the other box alone, it passed when under half that box's area, and the
    same pair of boxes was refused or not by which came first."""
    rects = list(rects)
    found = []
    for other in batch.proteins:
        if other.id == protein.id or other.image_id != protein.image_id:
            continue
        for band in other.bands:
            box = band.box.rect(other.box_size)
            share = max((_overlap_share(rect, box) for rect in rects), default=0.0)
            if share > COVER_SHARE:
                found.append((other, band, share))
    return found


def _cover_refusal(covered: Sequence[tuple[Protein, Band, float]], what: str) -> OperationError:
    """``OVERLAP``: ``what`` (the new box, a row's boxes, or a protein's boxes
    at a new size, with the verb: "the box would overlap") would overlap the
    ``covered`` boxes of other proteins (:func:`_covered`) by more than
    :data:`COVER_SHARE` of the smaller box's area; it names their band ids,
    with ``detail`` ``{"cause": "other_protein", "covered": [{band_id,
    protein_id, lane_index, share}]}``."""
    lanes: dict[str, tuple[Protein, list[int]]] = {}
    for other, band, _ in covered:
        lanes.setdefault(other.id, (other, []))[1].append(band.lane_index)
    parts = [
        f"{'the box' if len(indices) == 1 else 'the boxes'} of {other.name!r} in"
        f" {lanes_phrase(indices)}"
        for other, indices in lanes.values()
    ]
    return OperationError(
        ErrorCode.OVERLAP,
        f"{what} {_in_words(parts)} by more than {COVER_SHARE:.0%} of the smaller box's area:"
        " two proteins' boxes would measure the same band",
        ids=[band.id for _, band, _ in covered],
        detail={
            "cause": "other_protein",
            "covered": [
                {
                    "band_id": band.id,
                    "protein_id": other.id,
                    "lane_index": band.lane_index,
                    "share": round(share, 3),
                }
                for other, band, share in covered
            ],
        },
    )


# --- Images ---


@_locked
def import_image(
    session: ProjectSession,
    source: BinaryIO,
    original_name: str,
    *,
    kind: ImageKind,
    polarity: Polarity,
    membrane_id: str | None = None,
    max_bytes: int | None = None,
) -> str:
    """Store an image stream in the project and record it; return its id.

    The file is copied byte for byte into ``images/``, then read back: the
    recorded size, bit depth, background and warnings come from the stored copy,
    with ``looks_processed`` assessed for ``polarity``
    (:func:`~proteia.core.imaging.assess_processed`). Without ``membrane_id`` the
    image starts a new membrane. ``kind`` and ``polarity`` are required (the
    model has no silent default).
    """
    kind = _member(ImageKind, kind, "image kind")
    polarity = _member(Polarity, polarity, "polarity")
    if membrane_id is not None and all(
        m.id != membrane_id for m in session.project.batch.membranes
    ):
        raise UnknownIdError(f"unknown membrane {membrane_id!r}")
    if not isinstance(original_name, str):
        raise _invalid(f"original name must be text, not {original_name!r}")
    suffix = PurePosixPath(original_name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise OperationError(
            ErrorCode.UNSUPPORTED_IMAGE_TYPE,
            f"{original_name!r} is not a supported image type ({', '.join(IMAGE_SUFFIXES)})",
        )

    # An orphan may hold the id this import gets (an import that was never saved).
    session._remove_orphans()
    image_id = f"img-{session.project.next_id}"  # the id the change below will take
    try:
        stored = storage.store_image(
            session.folder, image_id, original_name, source, max_bytes=max_bytes
        )
    except storage.ImageTooLargeError as exc:
        raise OperationError(ErrorCode.IMAGE_TOO_LARGE, str(exc)) from exc
    except FileExistsError as exc:
        raise OperationError(
            ErrorCode.LEFTOVER_FILE,
            f"a leftover file for {image_id} in images/ could not be deleted;"
            " close the program holding it and try again",
            ids=(image_id,),
        ) from exc
    except ValueError as exc:
        raise OperationError(ErrorCode.INVALID_IMAGE, str(exc)) from exc

    path = session.folder / storage.IMAGES_DIR / stored.file
    try:
        try:
            loaded = load_image(path)
        except (ValueError, OSError) as exc:
            raise OperationError(
                ErrorCode.UNREADABLE_IMAGE, f"cannot read {original_name!r}: {exc}"
            ) from exc
        background = estimate_background(loaded.array)
        warnings = assess_processed(
            loaded.warnings,
            loaded.array,
            loaded.bit_depth,
            dark_on_light=polarity.dark_on_light,
            background=background,
            palette=loaded.palette,
        )

        def change(draft: Project) -> str:
            if draft.new_id("img") != image_id:  # the lock makes this impossible
                raise RuntimeError(f"image id {image_id} changed while the file was stored")
            if membrane_id is None:
                membrane = Membrane(id=draft.new_id("mem"))
                draft.batch.membranes.append(membrane)
            else:
                membrane = next(m for m in draft.batch.membranes if m.id == membrane_id)
            membrane.images.append(
                ImageRef(
                    id=image_id,
                    file=stored.file,
                    original_name=original_name,
                    kind=kind,
                    sha256=stored.sha256,
                    width=loaded.width,
                    height=loaded.height,
                    bit_depth=loaded.bit_depth,
                    polarity=polarity,
                    background=background,
                    import_warnings=warnings,
                )
            )
            return membrane.id

        new, membrane_used = _prepare(session, change)
        params = {
            "image_id": image_id,
            "membrane_id": membrane_used,
            "new_membrane": membrane_id is None,
            "original_name": original_name,
            "kind": kind.value,
            "polarity": polarity.value,
            "sha256": stored.sha256,  # keeps a removed image's provenance
        }
        session._commit(
            new, action="import_image", params=params, add_pixels={image_id: loaded.array}
        )
        _log_import(session, image_id)
    except BaseException:
        # _commit can raise before committing (a bad clock or params) or after
        # (a hook bug): keep the file only if the committed project uses it.
        if all(image.id != image_id for image in session.project.batch.iter_images()):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        raise
    return image_id


def _log_import(session: ProjectSession, image_id: str) -> None:
    """Log the warnings an image was imported with, and whether over-exposure
    can be checked on it."""
    image = session.project.batch.find_image(image_id)
    name = session.folder.name
    for warning in image.import_warnings:
        _log.info(
            "in %r: %s was imported with a warning, %s: %s",
            name,
            image_id,
            warning.code,
            warning.message,
        )
    if clipping_depth(image.bit_depth, image.import_warnings) is None:
        _log.info(
            "in %r: over-exposure cannot be checked on %s (%s)",
            name,
            image_id,
            _unchecked_why(image),
        )


@_locked
def remove_image(session: ProjectSession, image_id: str) -> Cascade:
    """Remove an image with the proteins, bands and not-detected records on it.

    Targets using a removed loading control, by name or as the batch's only
    one, are detached and reported, marker pairings to the image are cleared,
    and calibration points on it are dropped (``unfitted_membranes``). The
    membrane is refitted from what is left (:func:`_refresh_mw`): each image
    whose curve changed, a marker's group split or a ladder's points gone, gets
    its bands' apparent MWs recomputed and loses its proteins' MW-guided
    records. A membrane left with no image is removed. The file is deleted once
    the removal can no longer be undone, at the next save or import after that
    (see :mod:`proteia.core.session`). The log lists the removed records
    (``removed_undetected``) and the MW-guided ones dropped
    (``dropped_undetected``) in full.
    """
    committed = session.project
    committed.batch.find_image(image_id)

    def change(draft: Project) -> tuple[Cascade, list[JsonValue], list[JsonValue]]:
        batch = draft.batch
        gone = [p for p in batch.proteins if p.image_id == image_id]
        gone_ids = {p.id for p in gone}
        removed = [image_id, *(p.id for p in gone), *(b.id for p in gone for b in p.bands)]
        records: list[JsonValue] = [_undetected_json(p.id, u) for p in gone for u in p.undetected]
        # Targets using a removed loading control without naming it lose it too.
        implicit = {t for p in gone for t in _implicit_users(batch, p.id)} - gone_ids
        batch.proteins = [p for p in batch.proteins if p.id not in gone_ids]

        detached = []
        for protein in batch.proteins:
            kept = [i for i in protein.loading_control_ids if i not in gone_ids]
            if len(kept) != len(protein.loading_control_ids):
                protein.loading_control_ids = kept
                detached.append(protein.id)
            elif protein.id in implicit:
                detached.append(protein.id)

        unpaired = []
        for image in batch.iter_images():
            if image.marker_image_id == image_id:
                image.marker_image_id = None
                unpaired.append(image.id)

        membrane = batch.membrane_of(image_id)
        membrane.images = [image for image in membrane.images if image.id != image_id]
        unfitted = []
        calibration = membrane.calibration
        points = [point for point in calibration.points if point.image_id != image_id]
        if len(points) != len(calibration.points):
            calibration.points = points
            if membrane.images:
                unfitted.append(membrane.id)
        if not membrane.images:
            batch.membranes = [m for m in batch.membranes if m.id != membrane.id]
            removed.append(membrane.id)
        dropped: list[JsonValue] = list(_refresh_mw(committed, draft).dropped)
        cascade = Cascade(
            removed=tuple(removed),
            detached_targets=tuple(detached),
            unpaired_images=tuple(unpaired),
            unfitted_membranes=tuple(unfitted),
        )
        return cascade, records, dropped

    def params(result: tuple[Cascade, list[JsonValue], list[JsonValue]]) -> _Params:
        cascade, records, dropped = result
        return {
            "image_id": image_id,
            **_cascade(cascade),
            "removed_undetected": records,
            "dropped_undetected": dropped,
        }

    cascade, _, _ = _apply(session, "remove_image", change, params, evict=(image_id,))
    return cascade


@_locked
def set_polarity(session: ProjectSession, image_id: str, polarity: Polarity) -> None:
    """Set an image's polarity and re-quantify every band on it: the signal
    direction turns each ring's clip, level and haze lift around (the image's
    median does not depend on it). Its ``looks_processed`` warning, which counts
    the pixels at the background's end of the range, is assessed again
    (:func:`_processed_again`).

    The not-detected records of every protein on the image are dropped and
    logged: their SNR was measured with the other signal direction, against the
    row's fitted background, so it cannot be recomputed here. A later detection
    run writes them again. The band counts there (#58) go too, for the same
    reason.
    """
    polarity = _member(Polarity, polarity, "polarity")
    batch = session.project.batch
    image = batch.find_image(image_id)
    if image.polarity is polarity:
        return
    array = session.pixels(image_id) if _bands_on(batch, image_id) else None
    warnings = _processed_again(session, image, polarity, array)

    def change(draft: Project) -> list[dict[str, JsonValue]]:
        drafted = draft.batch.find_image(image_id)
        drafted.polarity = polarity
        drafted.import_warnings = warnings
        if array is not None:
            _quantify_image(draft, image_id, array)
        for protein in draft.batch.proteins:
            if protein.image_id == image_id:
                for band in protein.bands:
                    band.bands_found = None
        return [
            dropped
            for protein in draft.batch.proteins
            if protein.image_id == image_id
            for dropped in _drop_undetected_where(protein, lambda _: True)
        ]

    _apply(
        session,
        "set_polarity",
        change,
        lambda dropped: {
            "image_id": image_id,
            "polarity": polarity.value,
            "dropped_undetected": dropped,
        },
    )


def _processed_again(
    session: ProjectSession, image: ImageRef, polarity: Polarity, array: np.ndarray | None
) -> list[ImageWarning]:
    """The image's import warnings with ``looks_processed`` assessed for
    ``polarity`` (:func:`~proteia.core.imaging.assess_processed`), from its
    analysis array (``array``, or read now without keeping it, since an image
    with no band has no other use for it) and its file's header. An image whose
    file cannot be read loses a ``looks_processed`` assessed for the old
    polarity, which would name the wrong limit, and keeps its other warnings:
    with no band to re-quantify, its polarity changes as before #127, and every
    analysis of it is refused until the file is back."""
    try:
        if array is None:
            array = session.pixels(image.id, keep=False)
        palette = reads_as_palette(storage.image_path(session.folder, image))
    except (OperationError, ValueError, OSError):
        return [w for w in image.import_warnings if w.code != "looks_processed"]
    return assess_processed(
        image.import_warnings,
        array,
        image.bit_depth,
        dark_on_light=polarity.dark_on_light,
        background=image.background,
        palette=palette,
    )


# --- Molecular-weight calibration (#58) ---

# The image kinds each calibration point source may be marked on.
_SOURCE_KINDS: Final = {
    CalibrationPointSource.VISIBLE_MARKER: frozenset({ImageKind.VISIBLE_MARKER, ImageKind.MERGED}),
    CalibrationPointSource.CHEMILUMINESCENCE_MARKER: frozenset(
        {ImageKind.CHEMILUMINESCENCE, ImageKind.MERGED}
    ),
    CalibrationPointSource.STRIP_EDGE: frozenset(ImageKind),
}
# The image kinds a chemiluminescence image may take as its marker image (D4).
_MARKER_KINDS: Final = frozenset({ImageKind.VISIBLE_MARKER, ImageKind.MERGED})


def calibration_fit(membrane: Membrane, image_id: str) -> CalibrationFit | None:
    """The calibration of the register group holding ``image_id`` on
    ``membrane`` (:func:`~proteia.core.mwcal.calibration_for`), as the page
    shows it and the calibration operations log it; None without a curve."""
    fitted = mwcal.calibration_for(membrane, image_id)
    if not isinstance(fitted, mwcal.Calibration):
        return None
    ladder_fits = []
    for ladder in fitted.ladders:
        quality = ladder.quality
        ladder_fits.append(
            LadderFit(
                side=ladder.side.value,
                x=ladder.x,
                points=len(ladder.ys),
                quality=None if quality is None else quality.value,
                worst_mw=None if quality is None else quality.mw,
                predicted_mw=None if quality is None else quality.predicted_mw,
                two_point=len(ladder.ys) == 2,
                range=(_display_mw(ladder.curve.z_lo), _display_mw(ladder.curve.z_hi)),
            )
        )
    disagreement = fitted.disagreement
    return CalibrationFit(
        image_ids=tuple(image.id for image in membrane.images if image.id in fitted.group),
        method=fitted.method.value,
        ladders=tuple(ladder_fits),
        range=(_display_mw(fitted.z_lo), _display_mw(fitted.z_hi)),
        offset_px=fitted.offset_px,
        tilt_deg=fitted.tilt_deg,
        disagreement=None if disagreement is None else disagreement.value,
        disagreement_mw=None if disagreement is None else disagreement.mw,
        ignored=tuple((side.value, reason) for side, reason in fitted.ignored),
    )


def _point_json(point: CalibrationPoint) -> dict[str, JsonValue]:
    """A calibration point as a log entry names it: whole, since it has no id."""
    return {
        "image_id": point.image_id,
        "y": point.y,
        "mw": point.mw,
        "source": point.source.value,
        "x": point.x,
        "side": point.side.value,
    }


def _number(value: object, what: str) -> float:
    """A finite number as a float (not True or False), else ``INVALID_INPUT``."""
    number: float | None = None
    if not isinstance(value, bool) and isinstance(value, int | float):
        try:
            number = float(value) + 0.0  # an int too large for a float raises OverflowError
        except OverflowError:
            number = None
    if number is None or not math.isfinite(number):
        raise _invalid(f"{what} must be a finite number, not {value!r}")
    return number


def _kda(value: object, what: str = "molecular weight") -> float:
    """A positive, finite number of kDa, else ``INVALID_INPUT``."""
    try:
        number = _number(value, what)
    except OperationError:
        number = 0.0
    if number <= 0:
        raise _invalid(f"{what} must be a positive number of kDa, not {value!r}")
    return number


def _bool(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        raise _invalid(f"{what} must be True or False, not {value!r}")
    return value


def _within_image(image: ImageRef, *, x: float | None, y: float | None) -> None:
    """``OUT_OF_IMAGE`` for a position past the image's edges (a calibration
    point's continuous coordinates run from 0 to the width and height)."""
    for value, name, extent in ((x, "x", image.width), (y, "y", image.height)):
        if value is not None and not 0 <= value <= extent:
            raise OperationError(
                ErrorCode.OUT_OF_IMAGE,
                f"{name}={value:g} is outside the {image.width}x{image.height} image {image.id}",
                ids=(image.id,),
            )


def _group_names(membrane: Membrane, group: frozenset[str]) -> str:
    """A register group's images, in membrane order, as refusals name them."""
    return ", ".join(image.id for image in membrane.images if image.id in group)


def _ladder_refusal(
    membrane: Membrane,
    group: frozenset[str],
    points: Sequence[CalibrationPoint],
    new: CalibrationPoint | None = None,
) -> OperationError | None:
    """The refusal of the calibration points ``points`` of one register group,
    as a change would leave them, or None: the rules the model checks, in its
    order, each with its code. Per ladder side, two points at one MW
    (``DUPLICATE_MW``), two at one y or MWs out of order down the image
    (``CALIBRATION_ORDER``); then a right ladder beside a strip edge or a point
    with no x, or not right of the left one (``LADDER_SIDES``). ``new`` is the
    point added or edited, whose neighbours an order refusal names; otherwise
    it names the two points in conflict (points joined by a marker link)."""
    names = _group_names(membrane, group)
    head = f"membrane {membrane.id}:"

    def at(point: CalibrationPoint) -> str:
        return f"{point.mw:g} kDa at y={point.y:g} on {point.image_id}"

    for side in LadderSide:
        ladder = sorted((p for p in points if p.side == side), key=lambda p: (p.y, p.mw))
        where = f"the {side.value} ladder of {names}"
        # On log10(MW), as the model compares them: 100 and 100.00000000000001 are one.
        held: dict[float, CalibrationPoint] = {}
        for point in ladder:
            z = math.log10(point.mw)
            other = held.get(z)
            if other is not None:
                if new is not None and (point is new or other is new):
                    kept = other if point is new else point
                    message = f"{head} {where} already has a point at {kept.mw:g} kDa ({at(kept)})"
                else:
                    message = (
                        f"{head} {where} would hold {point.mw:g} kDa twice:"
                        f" {at(other)} and {at(point)}"
                    )
                return OperationError(ErrorCode.DUPLICATE_MW, message)
            held[z] = point
        for above, below in itertools.pairwise(ladder):
            if above.y == below.y:
                return OperationError(
                    ErrorCode.CALIBRATION_ORDER,
                    f"{head} two points at y={above.y:g} on {where}: {at(above)} and {at(below)};"
                    " each ladder band lies at its own height",
                )
        for above, below in itertools.pairwise(ladder):
            if math.log10(below.mw) < math.log10(above.mw):
                continue
            if new is not None:
                others = [p for p in ladder if p is not new]
                heavier = [p for p in others if p.mw > new.mw]
                lighter = [p for p in others if p.mw < new.mw]
                bounds = []
                if heavier:
                    bounds.append(f"below {at(min(heavier, key=lambda p: p.mw))}")
                if lighter:
                    bounds.append(f"above {at(max(lighter, key=lambda p: p.mw))}")
                message = (
                    f"{head} {new.mw:g} kDa at y={new.y:g} is out of order on {where}: it"
                    f" belongs {' and '.join(bounds)}, since lighter bands run further down"
                )
            else:
                message = (
                    f"{head} calibration points out of order on {where}: {at(below)} lies"
                    f" below {at(above)}"
                )
            return OperationError(ErrorCode.CALIBRATION_ORDER, message)
    right = [p for p in points if p.side == LadderSide.RIGHT]
    if not right:
        return None
    for point in points:
        if point.source is CalibrationPointSource.STRIP_EDGE:
            return OperationError(
                ErrorCode.LADDER_SIDES,
                f"{head} {names} would have a right ladder and a strip edge"
                f" ({at(point)}); a strip edge calibrates a group with one ladder only",
            )
        if point.x is None:
            return OperationError(
                ErrorCode.LADDER_SIDES,
                f"{head} {names} would have a right ladder, so every point there needs the x it"
                f" was marked at, and {at(point)} has none (it was saved before points recorded"
                " their x): remove it and mark it again",
            )
    left_x = [p.x for p in points if p.side == LadderSide.LEFT and p.x is not None]
    right_x = min(p.x for p in right if p.x is not None)
    if left_x and not max(left_x) < right_x:
        return OperationError(
            ErrorCode.LADDER_SIDES,
            f"{head} the right ladder of {names} (x={right_x:g}) would not lie right of its left"
            f" ladder (x={max(left_x):g}); the second ladder is the one right of the first",
        )
    return None


def _calibration_target(batch: Batch, image_id: str) -> tuple[Membrane, ImageRef, frozenset[str]]:
    """The membrane, the image and the register group a calibration request
    names by any of the group's images (``UnknownIdError`` if none)."""
    image = batch.find_image(image_id)
    membrane = batch.membrane_of(image_id)
    return membrane, image, membrane.group_of(image_id)


def _find_point(membrane: Membrane, group: frozenset[str], side: LadderSide, mw: float) -> int:
    """The index in the membrane's points of the point at ``mw`` on ``side`` of
    the group (a point's identity: its group, side and MW, compared on
    log10(MW) as the model does); ``UnknownIdError`` if there is none."""
    z = math.log10(mw)
    for index, point in enumerate(membrane.calibration.points):
        if point.image_id in group and point.side == side and math.log10(point.mw) == z:
            return index
    raise UnknownIdError(
        f"no calibration point at {mw:g} kDa on the {side.value} ladder of"
        f" {_group_names(membrane, group)}"
    )


def _snapped(
    session: ProjectSession,
    point: CalibrationPoint,
    image: ImageRef,
    marked: Sequence[CalibrationPoint],
) -> float | None:
    """Where the point's click meant (:func:`~proteia.core.mwcal.refine_point`),
    within the gaps between the points ``marked`` on its ladder; None when
    nothing stands out there."""
    if point.x is None:  # unreachable: snapping asks for the point's x first
        raise RuntimeError("a calibration point without x cannot be snapped")
    return mwcal.refine_point(
        session.pixels(image.id),
        point.x,
        point.y,
        source=point.source,
        polarity=image.polarity,
        marked_ys=[p.y for p in marked],
    )


_Calibrated = tuple[CalibrationFit | None, _MwRefresh]


def _calibration_change(
    session: ProjectSession,
    image_id: str,
    edit: Callable[[Project], None],
    params: Mapping[str, JsonValue],
) -> tuple[Callable[[Project], _Calibrated], Callable[[_Calibrated], _Params]]:
    """The change and the log params of a calibration operation, for
    :func:`_apply`: ``edit`` on the draft, then the one writer
    (:func:`_refresh_mw`). The log entry holds ``params``, then the register
    group's fit after (``fit``: :class:`CalibrationFit`, or None), the images
    whose curve changed (``curves_changed``) and the MW-guided records dropped
    from the proteins on them, in full (``dropped_undetected``). A change that
    leaves the project as it was commits nothing."""
    committed = session.project

    def change(draft: Project) -> _Calibrated:
        edit(draft)
        refresh = _refresh_mw(committed, draft)
        return calibration_fit(draft.batch.membrane_of(image_id), image_id), refresh

    def logged(result: _Calibrated) -> _Params:
        fit, refresh = result
        return {
            **params,
            "fit": None if fit is None else fit.as_json(),
            "curves_changed": list(refresh.curves_changed),
            "dropped_undetected": list(refresh.dropped),
        }

    return change, logged


def _calibration_update(
    result: _Calibrated, point: dict[str, JsonValue] | None = None
) -> CalibrationUpdate:
    """The :class:`CalibrationUpdate` of a calibration change's result."""
    fit, refresh = result
    return CalibrationUpdate(
        point=point,
        fit=fit,
        curves_changed=refresh.curves_changed,
        dropped_undetected=tuple(
            (record["protein_id"], record["lane_index"], record["band_index"])
            for record in refresh.dropped
        ),
    )


def _log_unsnapped(session: ProjectSession, before: Project, image_id: str, y: float) -> None:
    """Log, once committed, a click that found no band or edge to snap to."""
    if session.project is not before:
        _log.info(
            "in %r: no ladder band or edge stands out near y=%g on %s; the clicked position"
            " is kept",
            session.folder.name,
            y,
            image_id,
        )


@_locked
def set_marker_image(
    session: ProjectSession, image_id: str, marker_image_id: str | None
) -> CalibrationUpdate:
    """Link a chemiluminescence image to the marker image taken with it (a
    visible-light marker or merged image of the same membrane, the same size
    pixel for pixel: D4, D5), or unlink it with None. Linked images form one
    register group, calibrated from all its points; unlinking splits a group.

    Refused, changing nothing: an unknown id (``UnknownIdError``); an image
    that is not a chemiluminescence image, or a marker that is not a
    visible-light marker or merged image of its membrane (``INVALID_INPUT``);
    images of other sizes (``MARKER_SIZE_MISMATCH``); groups whose points
    conflict once joined (``DUPLICATE_MW``, ``CALIBRATION_ORDER`` or
    ``LADDER_SIDES``, naming both points). The same link is a no-op.

    The log entry holds ``membrane_id``, ``image_id``, ``marker_image_id``,
    the link it replaced (``previous``), the image's group after (``group``),
    and what :func:`_calibration_change` adds."""
    batch = session.project.batch
    image = batch.find_image(image_id)
    membrane = batch.membrane_of(image_id)
    marker = None
    if marker_image_id is not None:
        if not isinstance(marker_image_id, str):
            raise _invalid(f"marker image id must be text or None, not {marker_image_id!r}")
        marker = batch.find_image(marker_image_id)
    if image.kind is not ImageKind.CHEMILUMINESCENCE:
        raise _invalid(
            f"{image_id} is a {image.kind.value} image: only a chemiluminescence image is linked"
            " to a marker image",
            ids=(image_id,),
        )
    if marker is not None:
        if marker.kind not in _MARKER_KINDS or all(i.id != marker.id for i in membrane.images):
            raise _invalid(
                f"{marker.id} is not a visible-light marker or merged image of membrane"
                f" {membrane.id}",
                ids=(marker.id,),
            )
        if (marker.width, marker.height) != (image.width, image.height):
            raise OperationError(
                ErrorCode.MARKER_SIZE_MISMATCH,
                f"{image_id} ({image.width}x{image.height}) and {marker.id}"
                f" ({marker.width}x{marker.height}) differ in size; a marker image must match"
                " its image pixel for pixel",
                ids=(image_id, marker.id),
            )
    linked = membrane.model_copy(deep=True)
    next(i for i in linked.images if i.id == image_id).marker_image_id = marker_image_id
    for group in linked.register_groups():
        points = [p for p in linked.calibration.points if p.image_id in group]
        refusal = _ladder_refusal(linked, group, points)
        if refusal is not None:
            raise refusal
    previous = image.marker_image_id

    def edit(draft: Project) -> None:
        draft.batch.find_image(image_id).marker_image_id = marker_image_id

    group = linked.group_of(image_id)
    params = {
        "membrane_id": membrane.id,
        "image_id": image_id,
        "marker_image_id": marker_image_id,
        "previous": previous,
        "group": [i.id for i in linked.images if i.id in group],
    }
    change, logged = _calibration_change(session, image_id, edit, params)
    return _calibration_update(_apply(session, "set_marker_image", change, logged))


def _ladder_kda(kda: object) -> list[float]:
    """A ladder's MWs as given: positive numbers of kDa, strictly decreasing
    from top to bottom (on log10, as the model compares them)."""
    if isinstance(kda, str | bytes) or not isinstance(kda, Sequence):
        raise _invalid(f"ladder MWs must be a list of numbers of kDa, not {kda!r}")
    values = [_kda(value, "a ladder MW") for value in kda]
    for above, below in itertools.pairwise(values):
        if not math.log10(above) > math.log10(below):
            raise _invalid(
                f"ladder MWs must decrease strictly from top to bottom ({above:g}, {below:g} kDa)"
            )
    return values


@_locked
def set_ladder(
    session: ProjectSession,
    membrane_id: str,
    ladder: str | None,
    *,
    kda: Sequence[float] | None = None,
) -> None:
    """Choose a membrane's ladder: a preset key (:mod:`proteia.core.ladders`),
    whose MWs are copied into the calibration (``kda`` may repeat them), so the
    project keeps the values it was calibrated with, or a custom name (cleaned
    text without ``/``) with its MWs top to bottom, if known; None clears it.
    The points, the fit and every curve stay as they are.

    Refused, changing nothing: an unknown membrane (``UnknownIdError``); a
    name with ``/`` that is no preset key, MWs other than a preset's, MWs not
    positive or not strictly decreasing, or MWs without a name
    (``INVALID_INPUT``). The same ladder is a no-op. The log entry holds
    ``membrane_id``, the ladder and its MWs (``kda``) as stored."""
    batch = session.project.batch
    membrane = next((m for m in batch.membranes if m.id == membrane_id), None)
    if membrane is None:
        raise UnknownIdError(f"unknown membrane {membrane_id!r}")
    name = None if ladder is None else _clean(ladder, "ladder name")
    chosen = None
    if name is not None and "/" in name:
        chosen = ladders.preset(name)
        if chosen is None:
            raise _invalid(f"no ladder preset {name!r}; a custom ladder's name holds no '/'")
    given = None if kda is None else _ladder_kda(kda)
    if chosen is not None:
        values = list(chosen.kda)
        if given is not None and given != values:
            raise _invalid(
                f"the MWs given differ from those of the preset {name!r}"
                f" ({', '.join(f'{v:g}' for v in values)} kDa); name a custom ladder for"
                " other MWs"
            )
    elif name is None and given:
        raise _invalid("ladder MWs need a ladder name")
    else:
        values = [] if given is None else given
    committed = session.project

    def change(draft: Project) -> None:
        calibration = next(m for m in draft.batch.membranes if m.id == membrane_id).calibration
        calibration.ladder = name
        calibration.ladder_kda = list(values)
        # The one writer, as after every calibration change: no point or link
        # changed, so it refits nothing.
        _refresh_mw(committed, draft)

    params = {"membrane_id": membrane_id, "ladder": name, "kda": list(values)}
    _apply(session, "set_ladder", change, lambda _: params)


def _source(source: object, image: ImageRef) -> CalibrationPointSource:
    source = _member(CalibrationPointSource, source, "calibration point source")
    if image.kind not in _SOURCE_KINDS[source]:
        kinds = " or ".join(sorted(kind.value for kind in _SOURCE_KINDS[source]))
        raise _invalid(
            f"a {source.value} point is marked on a {kinds} image, not on {image.id}"
            f" ({image.kind.value})",
            ids=(image.id,),
        )
    return source


@_locked
def add_calibration_point(
    session: ProjectSession,
    image_id: str,
    y: float,
    mw: float,
    source: CalibrationPointSource,
    *,
    x: float | None = None,
    side: LadderSide = LadderSide.LEFT,
    snap: bool = True,
) -> CalibrationUpdate:
    """Mark a known MW at ``(x, y)`` on an image (continuous coordinates of its
    analysis array): a band of the ladder on ``side`` of its register group
    (``visible_marker`` on a marker or merged image,
    ``chemiluminescence_marker`` on a chemiluminescence or merged one), or the
    edge of a cut strip (``strip_edge``, on any image, always on the left
    ladder). A point is identified by its register group, side and MW. With
    ``snap`` the given y is moved to the band's peak, or the edge, nearest it
    (:func:`~proteia.core.mwcal.refine_point`); where nothing stands out, the
    given y is kept, and the session log says so.

    Refused, in this order, changing nothing: an unknown image
    (``UnknownIdError``); a source that is not marked on this kind of image;
    no ``x`` (every new point records where across the image it was marked,
    a strip edge's too); a strip edge on the right ladder (all
    ``INVALID_INPUT``); a position outside the image (``OUT_OF_IMAGE``); an MW
    that is not a positive number (``INVALID_INPUT``); after the snap, a
    point at that MW on that ladder already (``DUPLICATE_MW``), a point out of
    order down the ladder, or at the height of another (``CALIBRATION_ORDER``,
    naming its neighbours), or a right ladder not right of the left one, or
    beside a strip edge (``LADDER_SIDES``); an image file changed or
    unreadable, when snapping.

    The log entry holds ``membrane_id``, ``group``, ``side``, ``image_id``
    (the image of the group it is marked on), ``mw``, ``source``, ``y`` (as
    stored), ``y_given``, ``x``, ``snapped``, and what
    :func:`_calibration_change` adds."""
    batch = session.project.batch
    membrane, image, group = _calibration_target(batch, image_id)
    source = _source(source, image)
    if x is None:
        raise _invalid(
            "a calibration point needs its x: where across the image it was marked (the ladder"
            " lane's, or where the strip edge was clicked)"
        )
    side = _member(LadderSide, side, "ladder side")
    if side is LadderSide.RIGHT and source is CalibrationPointSource.STRIP_EDGE:
        raise _invalid("a strip edge belongs to the left ladder")
    given_y, x = _number(y, "y"), _number(x, "x")
    _within_image(image, x=x, y=given_y)
    mw = _kda(mw)
    snap = _bool(snap, "snap")
    points = [p for p in membrane.calibration.points if p.image_id in group]
    marked = [p for p in points if p.side == side]
    new = CalibrationPoint(image_id=image_id, y=given_y, mw=mw, source=source, x=x, side=side)
    found = _snapped(session, new, image, marked) if snap else None
    if found is not None:
        new = new.model_copy(update={"y": found})
    refusal = _ladder_refusal(membrane, group, [*points, new], new)
    if refusal is not None:
        raise refusal

    def edit(draft: Project) -> None:
        draft.batch.membrane_of(image_id).calibration.points.append(new.model_copy())

    snapped = found is not None
    params = {
        "membrane_id": membrane.id,
        "group": [i.id for i in membrane.images if i.id in group],
        "side": side.value,
        "image_id": image_id,
        "mw": mw,
        "source": source.value,
        "y": new.y,
        "y_given": given_y,
        "x": x,
        "snapped": snapped,
    }
    before = session.project
    change, logged = _calibration_change(session, image_id, edit, params)
    result = _apply(session, "add_calibration_point", change, logged)
    update = _calibration_update(result, {**_point_json(new), "snapped": snapped})
    if snap and not snapped:
        _log_unsnapped(session, before, image_id, given_y)
    return update


@_locked
def edit_calibration_point(
    session: ProjectSession,
    image_id: str,
    mw: float,
    *,
    side: LadderSide = LadderSide.LEFT,
    y: float | Keep = KEEP,
    new_mw: float | Keep = KEEP,
    snap: bool = False,
) -> CalibrationUpdate:
    """Move and/or relabel the point at ``mw`` on ``side`` of the register
    group ``image_id`` belongs to, in one change: ``y`` where it was dragged
    (exact, unless ``snap``: then moved to the band or edge nearest it, as
    :func:`add_calibration_point` does, within the gaps to the ladder's other
    points), ``new_mw`` its new label; ``KEEP`` leaves one as it is. The point
    stays on its image and its x. The same y and MW are a no-op.

    Refused, changing nothing: an unknown image, or no point at ``mw`` on
    that side of the group (``UnknownIdError``); a y outside the image
    (``OUT_OF_IMAGE``); an MW that is not a positive number, or a snap of a
    point saved without its x (``INVALID_INPUT``); after the edit, the rules
    of :func:`add_calibration_point` (``DUPLICATE_MW``,
    ``CALIBRATION_ORDER``, ``LADDER_SIDES``).

    The log entry holds ``membrane_id``, ``group``, ``side``, ``mw`` and
    ``new_mw`` (the label before and after), ``from_y``, ``y`` (as stored),
    ``y_given`` (null when kept), ``snapped``, and what :func:`_calibration_change`
    adds."""
    batch = session.project.batch
    membrane, image, group = _calibration_target(batch, image_id)
    side = _member(LadderSide, side, "ladder side")
    mw = _kda(mw)
    index = _find_point(membrane, group, side, mw)
    old = membrane.calibration.points[index]
    point_image = batch.find_image(old.image_id)
    given_y = None if y is KEEP else _number(y, "y")
    if given_y is not None:
        _within_image(point_image, x=None, y=given_y)
    label = old.mw if new_mw is KEEP else _kda(new_mw, "new molecular weight")
    snap = _bool(snap, "snap")
    if snap and old.x is None:
        raise _invalid(
            f"the point at {old.mw:g} kDa was saved without its x, so it cannot be snapped:"
            " remove it and mark it again"
        )
    moved = old.model_copy(update={"y": old.y if given_y is None else given_y, "mw": label})
    points = [p for p in membrane.calibration.points if p.image_id in group]
    found = None
    if snap:
        marked = [p for p in points if p.side == side and p is not old]
        found = _snapped(session, moved, point_image, marked)
        if found is not None:
            moved = moved.model_copy(update={"y": found})
    refusal = _ladder_refusal(membrane, group, [moved if p is old else p for p in points], moved)
    if refusal is not None:
        raise refusal

    def edit(draft: Project) -> None:
        draft.batch.membrane_of(image_id).calibration.points[index] = moved.model_copy()

    snapped = found is not None
    params = {
        "membrane_id": membrane.id,
        "group": [i.id for i in membrane.images if i.id in group],
        "side": side.value,
        "mw": old.mw,
        "new_mw": moved.mw,
        "from_y": old.y,
        "y": moved.y,
        "y_given": given_y,
        "snapped": snapped,
    }
    before = session.project
    change, logged = _calibration_change(session, image_id, edit, params)
    result = _apply(session, "edit_calibration_point", change, logged)
    update = _calibration_update(result, {**_point_json(moved), "snapped": snapped})
    if snap and not snapped:
        _log_unsnapped(session, before, old.image_id, moved.y)
    return update


@_locked
def remove_calibration_point(
    session: ProjectSession, image_id: str, mw: float, *, side: LadderSide = LadderSide.LEFT
) -> CalibrationUpdate:
    """Remove the point at ``mw`` on ``side`` of the register group
    ``image_id`` belongs to. Refused: an unknown image or no such point
    (``UnknownIdError``), an MW that is not a positive number
    (``INVALID_INPUT``). The log entry holds ``membrane_id``, ``group``,
    ``side``, the point whole (``removed``), and what :func:`_calibration_change`
    adds."""
    batch = session.project.batch
    membrane, _, group = _calibration_target(batch, image_id)
    side = _member(LadderSide, side, "ladder side")
    index = _find_point(membrane, group, side, _kda(mw))
    removed = membrane.calibration.points[index]

    def edit(draft: Project) -> None:
        del draft.batch.membrane_of(image_id).calibration.points[index]

    params = {
        "membrane_id": membrane.id,
        "group": [i.id for i in membrane.images if i.id in group],
        "side": side.value,
        "removed": _point_json(removed),
    }
    change, logged = _calibration_change(session, image_id, edit, params)
    return _calibration_update(_apply(session, "remove_calibration_point", change, logged))


@_locked
def clear_calibration(
    session: ProjectSession, image_id: str, *, side: LadderSide | None = None
) -> CalibrationUpdate:
    """Remove every point on ``side`` of the register group ``image_id``
    belongs to, or on both sides with None; the ladder chosen stays. A group
    with no such point is a no-op. The log entry holds ``membrane_id``,
    ``group``, ``side`` (null: both), the points removed, whole (``removed``),
    and what :func:`_calibration_change` adds."""
    batch = session.project.batch
    membrane, _, group = _calibration_target(batch, image_id)
    chosen = None if side is None else _member(LadderSide, side, "ladder side")

    def cleared(point: CalibrationPoint) -> bool:
        return point.image_id in group and (chosen is None or point.side == chosen)

    removed = [_point_json(p) for p in membrane.calibration.points if cleared(p)]

    def edit(draft: Project) -> None:
        calibration = draft.batch.membrane_of(image_id).calibration
        calibration.points = [p for p in calibration.points if not cleared(p)]

    params = {
        "membrane_id": membrane.id,
        "group": [i.id for i in membrane.images if i.id in group],
        "side": None if chosen is None else chosen.value,
        "removed": list(removed),
    }
    change, logged = _calibration_change(session, image_id, edit, params)
    return _calibration_update(_apply(session, "clear_calibration", change, logged))


# --- Finding a ladder and applying it (#58, D8) ---

# How close, in px, an applied point must lie to a position the server finds
# again (a found tick, a peak, a snap) to count as placed there.
PLACED_TOLERANCE: Final = 0.01


def _band_source(image: ImageRef) -> CalibrationPointSource:
    """The source of a ladder band marked on ``image``: a band of a visible-light
    marker on a marker or merged image, a faint marker on a chemiluminescence
    image."""
    if image.kind is ImageKind.CHEMILUMINESCENCE:
        return CalibrationPointSource.CHEMILUMINESCENCE_MARKER
    return CalibrationPointSource.VISIBLE_MARKER


def _lane_x(image: ImageRef, x: object, what: str = "x") -> float:
    """A ladder lane's x across ``image``: a finite number (``INVALID_INPUT``)
    on the image (``OUT_OF_IMAGE``)."""
    number = _number(x, what)
    _within_image(image, x=number, y=None)
    return number


def _proposal(
    session: ProjectSession,
    membrane: Membrane,
    image: ImageRef,
    x: float,
    side: LadderSide,
) -> mwcal.LadderProposal | None:
    """:func:`~proteia.core.mwcal.find_ladder` on the committed project: at
    ``x`` on ``image``, with the membrane's ladder MWs and its preset's
    reference bands (none for a custom ladder), and, for the right ladder, the
    register group's left ladder as the other one. ``INVALID_INPUT`` while the
    membrane has no ladder MWs (choose its ladder first)."""
    calibration = membrane.calibration
    if not calibration.ladder_kda:
        raise _invalid(
            f"membrane {membrane.id} has no ladder MWs to find: choose its ladder first",
            ids=(membrane.id,),
        )
    preset = None if calibration.ladder is None else ladders.preset(calibration.ladder)
    reference = () if preset is None else tuple(band.kda for band in preset.reference)
    other = None
    if side is LadderSide.RIGHT:
        fitted = mwcal.calibration_for(membrane, image.id)
        if isinstance(fitted, mwcal.Calibration):
            other = next(
                (ladder.curve for ladder in fitted.ladders if ladder.side is LadderSide.LEFT), None
            )
    return mwcal.find_ladder(
        session.pixels(image.id),
        x,
        kda=calibration.ladder_kda,
        reference_kda=reference,
        source=_band_source(image),
        polarity=image.polarity,
        other=other,
    )


def proposal_json(proposal: mwcal.LadderProposal) -> dict[str, JsonValue]:
    """A ladder proposal JSON-plain, as its route answers it and a log entry
    holds it: ``x``, the ``ticks`` (``{mw, y, found, strength}``, top to
    bottom), the ``extra`` peaks' ys, ``score``, ``gap`` and ``doubtful``. JSON
    has no infinity: a gap with no other labelling to measure it by is null,
    with ``gap_infinite`` true."""
    return {
        "x": proposal.x,
        "ticks": [
            {"mw": tick.mw, "y": tick.y, "found": tick.found, "strength": tick.strength}
            for tick in proposal.ticks
        ],
        "extra": list(proposal.extra),
        "score": proposal.score,
        "gap": _finite(proposal.gap),
        "gap_infinite": not math.isfinite(proposal.gap),
        "doubtful": proposal.doubtful,
    }


@_locked
def propose_ladder(
    session: ProjectSession, image_id: str, x: float, side: LadderSide = LadderSide.LEFT
) -> mwcal.LadderProposal | None:
    """Find the ladder whose lane was clicked at ``x`` on an image, and propose
    its labels (:func:`~proteia.core.mwcal.find_ladder`) with the membrane's
    ladder MWs and its preset's reference bands; for the ``right`` ladder, with
    the register group's left ladder as the other one. None where fewer than
    two bands stand out. Reads only: nothing is changed or logged.

    Refused: an unknown image (``UnknownIdError``); an unknown side, an ``x``
    that is not a finite number, or a membrane with no ladder MWs chosen
    (``INVALID_INPUT``); an ``x`` off the image (``OUT_OF_IMAGE``); an image
    file changed or unreadable."""
    batch = session.project.batch
    membrane, image, _ = _calibration_target(batch, image_id)
    side = _member(LadderSide, side, "ladder side")
    return _proposal(session, membrane, image, _lane_x(image, x), side)


@_locked
def snap_ladder(
    session: ProjectSession, image_id: str, x: float, ys: Sequence[float]
) -> tuple[tuple[float, bool], ...]:
    """Snap each of the ticks ``ys`` of a ruler drawn at ``x`` on an image to
    the ladder band nearest it (:func:`~proteia.core.mwcal.refine_point`, a band
    of the image's marker source), each within the gaps to the other ticks, so
    no tick snaps onto a neighbour's band. Gives ``(y, snapped)`` per tick, in
    the order given: the y snapped to, or the one given where nothing stands
    out. Reads only: nothing is changed or logged.

    Refused: an unknown image (``UnknownIdError``); an ``x`` or a y that is
    not a finite number, or ``ys`` not a list (``INVALID_INPUT``); a position
    off the image (``OUT_OF_IMAGE``); an image file changed or unreadable."""
    batch = session.project.batch
    _, image, _ = _calibration_target(batch, image_id)
    x = _lane_x(image, x)
    if isinstance(ys, str | bytes) or not isinstance(ys, Sequence):
        raise _invalid(f"the ticks' ys must be a list of numbers, not {ys!r}")
    given = [_number(y, "y") for y in ys]
    for y in given:
        _within_image(image, x=None, y=y)
    if not given:
        return ()
    array = session.pixels(image.id)
    source = _band_source(image)
    snapped = []
    for index, y in enumerate(given):
        others = given[:index] + given[index + 1 :]
        found = mwcal.refine_point(
            array, x, y, source=source, polarity=image.polarity, marked_ys=others
        )
        snapped.append((y, False) if found is None else (found, True))
    return tuple(snapped)


def _ladder_pairs(image: ImageRef, points: object) -> list[tuple[float, float]]:
    """The ``(y, mw)`` of each point of a ruler, checked as
    :func:`add_calibration_point` checks one: a y that is a finite number
    (``INVALID_INPUT``) on the image (``OUT_OF_IMAGE``), an MW that is a positive
    number of kDa (``INVALID_INPUT``)."""
    if isinstance(points, str | bytes) or not isinstance(points, Sequence):
        raise _invalid(f"a ladder's points must be a list of (y, MW) pairs, not {points!r}")
    pairs = []
    for point in points:
        if isinstance(point, str | bytes) or not isinstance(point, Sequence) or len(point) != 2:
            raise _invalid(f"a ladder point must be a (y, MW) pair, not {point!r}")
        y = _number(point[0], "y")
        _within_image(image, x=None, y=y)
        pairs.append((y, _kda(point[1])))
    return pairs


def _placed(
    session: ProjectSession,
    image: ImageRef,
    applied: Sequence[CalibrationPoint],
    replaced: Sequence[CalibrationPoint],
    proposal: mwcal.LadderProposal | None,
    proposed: bool,
) -> list[dict[str, JsonValue]]:
    """Each applied point as stored, with how it was placed, derived here and
    not taken from the client: ``found`` where it is a found tick of the
    proposal (the same MW, within :data:`PLACED_TOLERANCE` px); ``snapped``
    where its y is where a snap at its x puts the band its row lies on
    (:func:`~proteia.core.mwcal.is_snap_position`: a snap puts a band there
    to the bit, however it was clicked and whichever ticks cut its window in
    :func:`snap_ladder`), or where a snap put a point it ``replaced`` at the
    same y (judged at that point's x, on its image, as its source: a band
    clicked and snapped where it was clicked, then applied unchanged in a
    ruler at the lane's x, whose columns put the band a hair elsewhere), or
    where it sits on a peak the proposal found (a label moved by one band
    keeps its band's y); ``hand`` otherwise. With a proposal asked for
    (``proposed``), also ``relabelled``: whether its MW differs from that of
    the proposal's tick at its y."""
    if not applied:
        return []
    batch = session.project.batch
    array = session.pixels(image.id)
    source = _band_source(image)
    peaks = [] if proposal is None else [t.y for t in proposal.ticks if t.found]
    peaks += [] if proposal is None else list(proposal.extra)
    ticks = () if proposal is None else proposal.ticks

    def near(a: float, b: float) -> bool:
        return abs(a - b) <= PLACED_TOLERANCE

    def snapped_where(old: CalibrationPoint) -> bool:
        # Where a replaced point was snapped, on its own image: another image
        # of the group, whose stored file may no longer read. The ruler reads
        # only its own image, so it is not refused for that one: a point it
        # cannot check is not taken as snapped.
        try:
            pixels = session.pixels(old.image_id)
        except OperationError:
            return False
        return mwcal.is_snap_position(
            pixels,
            old.x,
            old.y,
            source=old.source,
            polarity=batch.find_image(old.image_id).polarity,
        )

    placed = []
    for point in applied:
        if point.x is None:  # unreachable: every applied point is marked at x
            raise RuntimeError("an applied ladder point has no x")
        on_tick = next((t for t in ticks if near(t.y, point.y)), None)
        if on_tick is not None and on_tick.found and math.log10(on_tick.mw) == math.log10(point.mw):
            how = "found"
        else:
            on_peak = any(near(peak, point.y) for peak in peaks)
            snapped = (
                on_peak
                or mwcal.is_snap_position(
                    array, point.x, point.y, source=source, polarity=image.polarity
                )
                or any(
                    snapped_where(old) for old in replaced if old.y == point.y and old.x is not None
                )
            )
            how = "snapped" if snapped else "hand"
        entry = {**_point_json(point), "placed": how}
        if proposed:
            entry["relabelled"] = on_tick is not None and math.log10(on_tick.mw) != math.log10(
                point.mw
            )
        placed.append(entry)
    return placed


@_locked
def set_ladder_points(
    session: ProjectSession,
    image_id: str,
    side: LadderSide,
    points: Sequence[tuple[float, float]],
    *,
    x: float,
    found_at: float | None = None,
) -> CalibrationUpdate:
    """Apply a ruler: replace every point of the ``side`` ladder of the register
    group ``image_id`` belongs to with ``points``, each ``(y, MW)``, all marked
    on ``image_id`` at ``x`` (the source follows the image kind: a marker band
    on a marker or merged image, a faint marker on a chemiluminescence one), in
    one change, one log entry and one undo step. An empty list clears the side.
    ``found_at`` is the x the ruler was found at (:func:`propose_ladder`), if it
    was.

    Sides follow x. A lone ladder is the left one, whichever side was given.
    A second ladder given as ``right`` but left of the only one (or as ``left``
    but right of it) makes that one the right (left) ladder and the new points
    the other, in the same change; the side given holds no ladder then when it
    has fewer than :data:`~proteia.core.mwcal.MIN_LADDER_POINTS` points, which
    the new points replace. Either way ``sides_swapped`` says the points went
    to the other side than the one given. Any other crossing is refused.

    Refused, changing nothing: an unknown image (``UnknownIdError``); an
    unknown side, an ``x`` or ``found_at`` that is not a finite number, points
    that are not ``(y, MW)`` pairs, or a found ruler on a membrane with no
    ladder MWs (``INVALID_INPUT``); then per point, as
    :func:`add_calibration_point` refuses one, a position off the image
    (``OUT_OF_IMAGE``) or an MW that is not a positive number
    (``INVALID_INPUT``); then on the group's points as they would be: two at one
    MW on a ladder (``DUPLICATE_MW``), two at one y or MWs out of order down it
    (``CALIBRATION_ORDER``), or the ladders crossing, a right ladder beside a
    strip edge or a point saved without its x (``LADDER_SIDES``); an image file
    changed or unreadable.

    The log entry holds ``membrane_id``, ``group``, ``side`` (where the points
    were stored), ``image_id``, ``x``, ``found_at``, ``sides_swapped``, the
    points of that side it replaced, whole (``removed``), the points applied as
    stored, each with ``placed`` (``found``, ``snapped`` or ``hand``, derived
    here: :func:`_placed`) and, with ``found_at``, ``relabelled``; with
    ``found_at``, the ``proposal`` found there again from the same pixels
    (:func:`proposal_json`, null where none is found); and what
    :func:`_calibration_change` adds. A proposal found there that is doubtful
    (its labels may be one band off) is also logged at INFO once committed.
    The update's ``points`` are those applied points."""
    batch = session.project.batch
    membrane, image, group = _calibration_target(batch, image_id)
    side = _member(LadderSide, side, "ladder side")
    x = _lane_x(image, x)
    found_x = None if found_at is None else _lane_x(image, found_at, "found_at")
    if found_x is not None and not membrane.calibration.ladder_kda:
        raise _invalid(
            f"membrane {membrane.id} has no ladder MWs, so no ruler was found on it: choose its"
            " ladder first",
            ids=(membrane.id,),
        )
    pairs = _ladder_pairs(image, points)
    source = _band_source(image)
    held = [p for p in membrane.calibration.points if p.image_id in group]
    replaced = [p for p in held if p.side == side]
    opposite = [p for p in held if p.side != side]
    stored, swapped = side, False
    if pairs:
        opposite_xs = [p.x for p in opposite if p.x is not None]
        if not opposite:
            # A lone ladder is the left one.
            stored, swapped = LadderSide.LEFT, side is LadderSide.RIGHT
        elif len(replaced) < mwcal.MIN_LADDER_POINTS and opposite_xs:
            # The side given holds no ladder (too few points to be used: a
            # stray point is replaced as well), so the other side's is the
            # only one.
            beyond = x < min(opposite_xs) if side is LadderSide.RIGHT else x > max(opposite_xs)
            if beyond:  # the second ladder is on the other side of the only one
                stored, swapped = _other_side(side), True
    moved = _other_side(stored) if swapped else None
    applied = [
        CalibrationPoint(image_id=image_id, y=y, mw=mw, source=source, x=x, side=stored)
        for y, mw in sorted(pairs)
    ]
    kept = [p.model_copy(update={"side": moved}) if moved is not None else p for p in opposite]
    refusal = _ladder_refusal(membrane, group, [*kept, *applied])
    if refusal is not None:
        raise refusal
    proposal = None if found_x is None else _proposal(session, membrane, image, found_x, side)
    placed = _placed(session, image, applied, replaced, proposal, found_x is not None)

    def edit(draft: Project) -> None:
        calibration = draft.batch.membrane_of(image_id).calibration
        others = [p for p in calibration.points if p.image_id not in group]
        calibration.points = [
            *others,
            *(p.model_copy() for p in kept),
            *(p.model_copy() for p in applied),
        ]

    params: dict[str, JsonValue] = {
        "membrane_id": membrane.id,
        "group": [i.id for i in membrane.images if i.id in group],
        "side": stored.value,
        "image_id": image_id,
        "x": x,
        "found_at": found_x,
        "sides_swapped": swapped,
        "removed": [_point_json(p) for p in replaced],
        "points": list(placed),
    }
    if found_x is not None:
        params["proposal"] = None if proposal is None else proposal_json(proposal)
    before = session.project
    change, logged = _calibration_change(session, image_id, edit, params)
    update = _calibration_update(_apply(session, "set_ladder_points", change, logged))
    if proposal is not None and proposal.doubtful:
        _log_doubtful(session, before, image_id, proposal)
    return replace(update, points=tuple(placed), sides_swapped=swapped)


def _log_doubtful(
    session: ProjectSession, before: Project, image_id: str, proposal: mwcal.LadderProposal
) -> None:
    """Log, once committed, a ruler applied from a proposal whose labels may be
    one band off (its gap is below :data:`~proteia.core.mwcal.GAP_WARN`)."""
    if session.project is not before:
        _log.info(
            "in %r: the ladder found at x=%g on %s may be one band off (gap %.3g to the"
            " next-best labelling, below %g); the ruler is applied as given",
            session.folder.name,
            proposal.x,
            image_id,
            proposal.gap,
            mwcal.GAP_WARN,
        )


def _other_side(side: LadderSide) -> LadderSide:
    return LadderSide.RIGHT if side is LadderSide.LEFT else LadderSide.LEFT


# --- Lanes and the reference ---


@_locked
def set_lanes(
    session: ProjectSession,
    lanes: Sequence[LaneInput],
    *,
    reference_condition: str | None | Keep = KEEP,
) -> LanesUpdate:
    """Declare or replace the whole lane table; row i is lane i.

    Text is cleaned (:func:`~proteia.core.names.clean_text`). Look-alike spellings
    (equal NFKC key, e.g. the micro sign and Greek mu) are unified: where the new
    table mixes them, the spelling already in the table wins; a look-alike typed
    for every lane of a condition renames it. Case differences are kept.
    Lanes holding a box cannot be dropped (``LANES_IN_USE``): boxes are the user's
    measurements. Not-detected records in dropped lanes are dropped instead, and
    reported: one detection run makes them again, and a lane cut from the table
    means nothing any more. Lane indices are never remapped. Each kept lane keeps
    its metadata and its records.

    ``reference_condition``: ``KEEP`` re-resolves the current reference against
    the new conditions and clears it if none matches (reported in the result); a
    string sets it in the same change (``UNKNOWN_CONDITION`` if no lane has it);
    None clears it.
    """
    if isinstance(lanes, str) or not isinstance(lanes, Sequence):
        raise _invalid("lanes must be a sequence of LaneInput")
    typed_conditions: list[str] = []
    typed_samples: list[str | None] = []
    included: list[bool] = []
    for i, lane in enumerate(lanes):
        row = f"lane {lane_number(i)}"
        if not isinstance(lane, LaneInput):
            raise _invalid(f"{row} must be a LaneInput, not {lane!r}")
        typed_conditions.append(_clean(lane.condition, f"{row} condition"))
        typed_samples.append(_clean_optional(lane.sample, f"{row} sample"))
        if not isinstance(lane.included, bool):
            raise _invalid(f"{row} included must be True or False, not {lane.included!r}")
        included.append(lane.included)

    batch = session.project.batch
    conditions = unify_spellings(typed_conditions, existing=[lane.label for lane in batch.lanes])
    samples = unify_spellings(typed_samples, existing=[lane.sample for lane in batch.lanes])
    n = len(conditions)
    respelled = tuple(
        i
        for i in range(n)
        if conditions[i] != typed_conditions[i] or samples[i] != typed_samples[i]
    )

    cut = [band for p in batch.proteins for band in p.bands if band.lane_index >= n]
    if cut:
        held = lanes_phrase({band.lane_index for band in cut})
        raise OperationError(
            ErrorCode.LANES_IN_USE,
            f"{len(cut)} box(es) are in {held}, which the new table drops; remove them first",
            ids=[band.id for band in cut],
        )

    labels = [c for c in conditions if c is not None]
    cleared = False
    if reference_condition is KEEP:
        current = batch.reference_condition
        reference = None if current is None else resolve_label(current, labels)
        cleared = current is not None and reference is None
    elif reference_condition is None:
        reference = None
    else:
        reference = _condition(reference_condition, labels)

    def change(draft: Project) -> list[dict[str, JsonValue]]:
        old = draft.batch.lanes
        draft.batch.lanes = [
            Lane(
                index=i,
                label=conditions[i],
                sample=samples[i],
                included=included[i],
                metadata=dict(old[i].metadata) if i < len(old) else {},
            )
            for i in range(n)
        ]
        draft.batch.reference_condition = reference
        return [
            dropped
            for protein in draft.batch.proteins
            for dropped in _drop_undetected_where(protein, lambda record: record.lane_index >= n)
        ]

    # Only the rows that are new or differ from the committed table, as stored.
    stored = [(lane.label, lane.sample, lane.included) for lane in batch.lanes]
    rows = list(zip(conditions, samples, included, strict=True))
    changed: list[JsonValue] = [
        {"index": i, "condition": condition, "sample": sample, "included": include}
        for i, (condition, sample, include) in enumerate(rows)
        if i >= len(stored) or rows[i] != stored[i]
    ]

    def params(dropped: list[dict[str, JsonValue]]) -> _Params:
        return {
            "lane_count": n,
            "changed": changed,
            "reference_condition": reference,
            "dropped_undetected": dropped,
        }

    dropped = _apply(session, "set_lanes", change, params)
    keys = tuple((d["protein_id"], d["lane_index"], d["band_index"]) for d in dropped)
    return LanesUpdate(respelled=respelled, reference_cleared=cleared, dropped_undetected=keys)


@_locked
def set_reference_condition(session: ProjectSession, condition: str | None) -> None:
    """Choose the fold-change reference by its condition (the stored spelling is
    the lane's), or clear it with None."""
    batch = session.project.batch
    reference = None if condition is None else _condition(condition, [x.label for x in batch.lanes])

    def change(draft: Project) -> None:
        draft.batch.reference_condition = reference

    _apply(session, "set_reference_condition", change, lambda _: {"reference_condition": reference})


# --- Proteins ---


@_locked
def add_protein(
    session: ProjectSession,
    name: str,
    role: Role,
    image_id: str,
    *,
    expected_mw: float | None = None,
    loading_control_ids: Sequence[str] = (),
    box_size: BoxSize | None = None,
    mw_tolerance: float | None = None,
) -> str:
    """Add a protein on a signal image, last in the protein order; return its id.

    The name is stored cleaned and must be unique ignoring case and look-alike
    characters. ``loading_control_ids`` (targets only) keeps its order, the series
    order. ``box_size`` defaults to :func:`~proteia.core.boxes.initial_box_size`;
    the first grown box replaces it. ``mw_tolerance`` is how far, as a share,
    an apparent MW may lie from the expected one and pass (#58; None: the
    model's default, 0.1, ±10%); it also sizes the band count's window. The
    image cannot be changed later. Adding a second loading control writes the
    first into the targets that used it without naming it, so their results
    do not change.
    """
    batch = session.project.batch
    role = _member(Role, role, "role")
    name = _protein_name(batch, name)
    image = batch.find_image(image_id)
    if image.kind is ImageKind.VISIBLE_MARKER:
        raise OperationError(
            ErrorCode.MARKER_IMAGE,
            f"{image_id} is a visible-light marker image, not a signal image",
            ids=(image_id,),
        )
    mw = _expected_mw(expected_mw)
    tolerance = None if mw_tolerance is None else _mw_tolerance(mw_tolerance)
    controls = _loading_controls(batch, None, role, loading_control_ids)
    if box_size is None:
        size = boxes.initial_box_size(image.width, image.height)
    else:
        size = _fitting_size(box_size, image)

    def change(draft: Project) -> tuple[str, list[str]]:
        pinned = _pin_single_loading_control(draft.batch) if role is Role.LOADING_CONTROL else []
        protein_id = draft.new_id("prot")
        added = Protein(
            id=protein_id,
            name=name,
            role=role,
            image_id=image_id,
            loading_control_ids=controls,
            expected_mw=mw,
            box_size=size,
        )
        if tolerance is not None:
            added.mw_tolerance = tolerance
        draft.batch.proteins.append(added)
        return protein_id, pinned

    def params(result: tuple[str, list[str]]) -> _Params:
        protein_id, pinned = result
        return {
            "protein_id": protein_id,
            "name": name,
            "role": role.value,
            "image_id": image_id,
            "expected_mw": mw,
            "mw_tolerance": Protein.model_fields["mw_tolerance"].default
            if tolerance is None
            else tolerance,
            "loading_control_ids": list(controls),
            "box_size": _size(size),
            "pinned_targets": pinned,
        }

    protein_id, _ = _apply(session, "add_protein", change, params)
    return protein_id


@_locked
def edit_protein(
    session: ProjectSession,
    protein_id: str,
    *,
    name: str | Keep = KEEP,
    role: Role | Keep = KEEP,
    expected_mw: float | None | Keep = KEEP,
    loading_control_ids: Sequence[str] | Keep = KEEP,
    mw_tolerance: float | Keep = KEEP,
) -> None:
    """Edit a protein's fields; ``KEEP`` leaves one unchanged. No net changes.

    A target turned into a loading control loses its own loading controls; if it
    becomes the second one, the first is written into the targets that used it
    without naming it. A loading control that targets use, by name or as the
    batch's only one, cannot become a target (``LOADING_CONTROL_IN_USE``, with
    those targets): choose other loading controls for them first. A changed
    expected MW drops the protein's MW-guided not-detected records, whose slot
    came from the old one (``dropped_undetected``, logged in full), and clears
    the band counts of its MW-guided bands. A changed MW tolerance (#58) is
    read by the MW check at once; it clears the band counts of the protein's
    detector bands where their window came from it (its image has a curve),
    and drops no record.
    """
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    new_name = protein.name if name is KEEP else _protein_name(batch, name, protein_id=protein_id)
    new_role = protein.role if role is KEEP else _member(Role, role, "role")
    mw = protein.expected_mw if expected_mw is KEEP else _expected_mw(expected_mw)
    tolerance = protein.mw_tolerance if mw_tolerance is KEEP else _mw_tolerance(mw_tolerance)
    recount = tolerance != protein.mw_tolerance and _count_cleared_by_tolerance(batch, protein)
    if protein.role is Role.LOADING_CONTROL and new_role is Role.TARGET:
        users = _users(batch, protein_id)
        if users:
            raise OperationError(
                ErrorCode.LOADING_CONTROL_IN_USE,
                f"{protein.name!r} is the loading control of {len(users)} target(s)",
                ids=users,
            )
    if loading_control_ids is KEEP:
        # A loading control has none; a target keeps its choice.
        controls = [] if new_role is Role.LOADING_CONTROL else list(protein.loading_control_ids)
    else:
        controls = _loading_controls(batch, protein_id, new_role, loading_control_ids)

    def change(draft: Project) -> tuple[list[str], list[JsonValue]]:
        pinned = []
        if protein.role is Role.TARGET and new_role is Role.LOADING_CONTROL:
            pinned = _pin_single_loading_control(draft.batch)
        edited = draft.batch.find_protein(protein_id)
        edited.name = new_name
        edited.role = new_role
        edited.expected_mw = mw
        edited.mw_tolerance = tolerance
        edited.loading_control_ids = controls
        dropped: list[JsonValue] = []
        if mw != protein.expected_mw:
            dropped.extend(_drop_mw_guided(edited))
        for band in edited.bands:  # counts whose window or slot the edit moved
            guided = mw != protein.expected_mw and band.source is ProposalSource.MW_GUIDED
            if recount or guided:
                band.bands_found = None
        pinned = [target for target in pinned if target != protein_id]  # its own is cleared
        return pinned, dropped

    def params(result: tuple[list[str], list[JsonValue]]) -> _Params:
        pinned, dropped = result
        # What the edit did to others, always; of the protein's own fields, only
        # those whose stored value changed, a cleared list included.
        edits: dict[str, JsonValue] = {
            "protein_id": protein_id,
            "pinned_targets": pinned,
            "dropped_undetected": dropped,
        }
        if new_name != protein.name:
            edits["name"] = new_name
        if new_role is not protein.role:
            edits["role"] = new_role.value
        if mw != protein.expected_mw:
            edits["expected_mw"] = mw
        if tolerance != protein.mw_tolerance:
            edits["mw_tolerance"] = tolerance
        if controls != protein.loading_control_ids:
            edits["loading_control_ids"] = list(controls)
        return edits

    _apply(session, "edit_protein", change, params)


@_locked
def remove_protein(session: ProjectSession, protein_id: str) -> Cascade:
    """Remove a protein with its bands and not-detected records; targets using it
    as their loading control, by name or as the batch's only one, are detached
    and reported. The log lists the records in full (``removed_undetected``).
    The other bands on its image are re-quantified: their rings no longer leave
    out its boxes."""
    protein = session.project.batch.find_protein(protein_id)
    image_id = protein.image_id
    band_ids = [band.id for band in protein.bands]
    array = _pixels_left(session, image_id, band_ids) if band_ids else None

    def change(draft: Project) -> tuple[Cascade, list[JsonValue]]:
        batch = draft.batch
        gone = batch.find_protein(protein_id)
        removed = (protein_id, *(band.id for band in gone.bands))
        records: list[JsonValue] = [_undetected_json(protein_id, u) for u in gone.undetected]
        implicit = set(_implicit_users(batch, protein_id))
        batch.proteins = [p for p in batch.proteins if p.id != protein_id]
        if array is not None:
            _quantify_image(draft, image_id, array)
        detached = []
        for protein in batch.proteins:
            if protein_id in protein.loading_control_ids:
                protein.loading_control_ids.remove(protein_id)
                detached.append(protein.id)
            elif protein.id in implicit:
                detached.append(protein.id)
        cascade = Cascade(
            removed=removed,
            detached_targets=tuple(detached),
            unpaired_images=(),
            unfitted_membranes=(),
        )
        return cascade, records

    def params(result: tuple[Cascade, list[JsonValue]]) -> _Params:
        cascade, records = result
        return {"protein_id": protein_id, **_cascade(cascade), "removed_undetected": records}

    cascade, _ = _apply(session, "remove_protein", change, params)
    return cascade


# --- Boxes ---


@_locked
def place_box(
    session: ProjectSession,
    protein_id: str,
    x: int,
    y: int,
    *,
    lane_index: int | None = None,
    grow: bool,
) -> str:
    """Place a box of a protein in a lane at the image point ``(x, y)``; return
    the band id.

    Without ``lane_index``, the lane is proposed from the position
    (:func:`~proteia.core.project.propose_lane`): the clicked x for a fixed box,
    the grown band's centre for a seed click. It needs boxes in at least two
    lanes on the image to know the lane pitch (``LANE_REQUIRED`` otherwise), and
    a proposed lane outside the table (``LANE_OUT_OF_RANGE``) or already holding
    a box of the protein (``LANE_OCCUPIED``) is refused, never moved elsewhere.
    Either way the lane is stored with the band, and moving or removing other
    boxes never changes it.

    ``grow=True`` is a seed click: the band is grown from the point and the
    protein's shared size fitted to it (the first box sets the fitted size,
    later ones only grow it, and the other boxes are re-centred); the
    protein's padding (:func:`set_box_padding`) is added to the fitted size and
    stays, even if it is then more than half of it.
    ``grow=False`` drops a box of the current size, padding included, centred
    on the point, shifted inside the image. A protein's own boxes never
    overlap (``OVERLAP``, with the box in the way), and the new box, or a box
    of the protein grown to a size the click grows, may overlap another
    protein's box on the image by at most :data:`COVER_SHARE` of the smaller
    box's area, each at its size padding included (``OVERLAP``, naming the
    boxes it would overlap, :func:`_cover_refusal`): two proteins' boxes on
    one band would measure it twice. Every band on the image, of every
    protein, is then re-quantified: the new box leaves every ring.

    A not-detected record in the lane is replaced by the box, with no
    confirmation: the log entry names it (``replaced_undetected``, else null).
    """
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    x, y = _int(x, "x"), _int(y, "y")
    lane_index = None if lane_index is None else _int(lane_index, "lane index")
    proposed = lane_index is None
    if not isinstance(grow, bool):
        raise _invalid(f"grow must be True or False, not {grow!r}")
    n = len(batch.lanes)
    if n == 0:
        raise OperationError(ErrorCode.NO_LANES, "declare the lanes before placing boxes")
    taken = {b.lane_index: b.id for b in protein.bands if b.band_index == 0}
    image = batch.find_image(protein.image_id)
    width, height = image.width, image.height
    if not (0 <= x < width and 0 <= y < height):
        raise OperationError(
            ErrorCode.OUT_OF_IMAGE, f"({x}, {y}) is outside the {width}x{height} image"
        )
    anchors: list[tuple[float, int]] = []
    if lane_index is not None:
        _check_lane(protein, lane_index, n, taken, proposed=False)
    else:
        # Before any pixel work: can a lane be proposed at all? A fixed box's lane
        # is where the user clicked; a seed click's waits for the grown band.
        anchors = lane_anchors(batch, image)
        if propose_lane(x, anchors) is None:
            raise OperationError(
                ErrorCode.LANE_REQUIRED,
                "choose the lane: position proposes one only once boxes in two lanes of"
                " this image show the lane spacing",
            )
        if not grow:
            lane_index = _check_lane(protein, propose_lane(x, anchors), n, taken, proposed=True)

    array = session.pixels(image.id)
    rects = [band.box.rect(protein.box_size) for band in protein.bands]
    if grow:
        # The settings the export record reports (record.settings), passed explicitly.
        grown = grow_box(
            array,
            (x, y),
            image.background,
            rel_threshold=REL_THRESHOLD,
            noise_k=NOISE_K,
            dark_on_light=image.polarity.dark_on_light,
        )
        if grown is None:
            raise OperationError(ErrorCode.NO_BAND_FOUND, f"no band found at ({x}, {y})")
        try:
            size, resized, rect = boxes.grow_to_fit(
                rects, protein.box_size, grown, width=width, height=height, pad=protein.box_padding
            )
        except boxes.BoxRuleError as exc:  # overlap, or size_would_overlap
            raise OperationError(ErrorCode(exc.code), str(exc)) from exc
        covered = _covered(batch, protein, [rect])
        if covered:
            raise _cover_refusal(covered, "the box would overlap")
        # The protein's other boxes, grown to the size around their centres.
        covered = _covered(
            batch, protein, [r for r, o in zip(resized, rects, strict=True) if r != o]
        )
        if covered:
            raise _cover_refusal(
                covered,
                f"at the box size {boxes.size_words(size, protein.box_padding)} this band"
                f" needs, boxes of {protein.name!r} would overlap",
            )
        source = ProposalSource.CLICK
        if lane_index is None:
            # The grown band's own centre (the box is shifted inside the image at the
            # edges); where the user clicked if the image edge cuts the band.
            gx0, _, gx1, _ = grown
            at = x if gx0 <= 0 or gx1 >= width else (gx0 + gx1) / 2
            lane_index = _check_lane(protein, propose_lane(at, anchors), n, taken, proposed=True)
    else:
        size, resized = protein.box_size, rects
        rect = boxes.centered_rect(x, y, size, width, height)
        hits = _overlapped(rect, protein)
        if hits:
            raise OperationError(
                ErrorCode.OVERLAP, "the box would overlap another box of this protein", ids=hits
            )
        covered = _covered(batch, protein, [rect])
        if covered:
            raise _cover_refusal(covered, "the box would overlap")
        source = ProposalSource.MANUAL
    if lane_index is None:  # unreachable: every branch above checks or proposes it
        raise RuntimeError("place_box left the lane unresolved")

    def change(draft: Project) -> tuple[str, dict[str, JsonValue] | None]:
        edited = draft.batch.find_protein(protein_id)
        edited.box_size = size
        for band, old, new in zip(edited.bands, rects, resized, strict=True):
            if new != old:
                _set_box(band, new)
        band = Band(
            id=draft.new_id("band"),
            lane_index=lane_index,
            band_index=0,
            box=Box(x=rect[0], y=rect[1]),
            source=source,
            **_UNQUANTIFIED,
        )
        edited.bands.append(band)
        _quantify_image(draft, edited.image_id, array)
        _refresh_box_mws(session.project, draft)
        return band.id, _drop_undetected(edited, lane_index, 0)

    def params(result: tuple[str, dict[str, JsonValue] | None]) -> _Params:
        band_id, replaced = result
        return {
            "band_id": band_id,
            "protein_id": protein_id,
            "x": x,
            "y": y,
            "grow": grow,
            "lane_index": lane_index,  # the stored lane, chosen or proposed
            "lane_proposed": proposed,
            "rect": list(rect),
            "box_size": _size(size),  # after the change: a seed click may grow it
            "replaced_undetected": replaced,
        }

    band_id, _ = _apply(session, "place_box", change, params)
    return band_id


@_locked
def move_box(session: ProjectSession, band_id: str, rect: Rect) -> None:
    """Move a box to where the user dragged or resized it.

    ``rect`` is read by its centre: the box keeps the protein's size, centred
    there and shifted inside the image. It may not overlap another box of its
    protein, nor another protein's box by more than :data:`COVER_SHARE` of the
    smaller box's area (``OVERLAP``, as :func:`place_box`). Every band on the
    image is
    re-quantified, since the box leaves one ring and may cut another; its lane
    never changes. Marks the band as manually edited, which clears its band
    count (#58): the detector counted around the box it placed.
    """
    batch = session.project.batch
    protein, band = batch.find_band(band_id)
    if isinstance(rect, str) or not isinstance(rect, Sequence) or len(rect) != 4:
        raise _invalid(f"rect must be (x0, y0, x1, y1), not {rect!r}")
    x0, y0, x1, y1 = (_int(v, "rect coordinate") for v in rect)
    rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    image = batch.find_image(protein.image_id)
    new = boxes.center_snap(rect, protein.box_size, image.width, image.height)
    if new == band.box.rect(protein.box_size):
        return
    hits = _overlapped(new, protein, skip=band_id)
    if hits:
        raise OperationError(
            ErrorCode.OVERLAP, "the box would overlap another box of this protein", ids=hits
        )
    covered = _covered(batch, protein, [new])
    if covered:
        raise _cover_refusal(covered, "the box would overlap")
    array = session.pixels(image.id)

    def change(draft: Project) -> None:
        edited, moved = draft.batch.find_band(band_id)
        _set_box(moved, new)
        moved.manually_edited = True
        moved.bands_found = None
        _quantify_image(draft, edited.image_id, array)
        _refresh_box_mws(session.project, draft)

    _apply(session, "move_box", change, lambda _: {"band_id": band_id, "rect": list(new)})


@_locked
def remove_box(session: ProjectSession, band_id: str) -> None:
    """Remove one box. The protein's size and its other boxes are unchanged; the
    bands left on the image are re-quantified, since their rings no longer leave
    out the box."""
    protein, band = session.project.batch.find_band(band_id)
    params = {"band_id": band_id, "protein_id": protein.id, "lane_index": band.lane_index}
    image_id = protein.image_id
    array = _pixels_left(session, image_id, [band_id])

    def change(draft: Project) -> None:
        protein, _ = draft.batch.find_band(band_id)
        protein.bands = [band for band in protein.bands if band.id != band_id]
        if array is not None:
            _quantify_image(draft, image_id, array)

    _apply(session, "remove_box", change, lambda _: params)


@_locked
def set_box_lane(session: ProjectSession, band_id: str, lane_index: int) -> None:
    """Store another lane as the box's lane; the box stays where it is.

    The lane must be one of the declared lanes (``LANE_OUT_OF_RANGE``) where the
    protein has no box for the same band yet (``LANE_OCCUPIED``). The box counts
    as edited by the user, so it loses its band count (#58). The same lane is a
    no-op. A not-detected record for the same band in the new lane is replaced
    (``replaced_undetected``); the lane the box leaves gets no record.
    """
    batch = session.project.batch
    protein, band = batch.find_band(band_id)
    lane = _int(lane_index, "lane index")
    if lane == band.lane_index:
        return
    taken = {
        b.lane_index: b.id
        for b in protein.bands
        if b.band_index == band.band_index and b.id != band_id
    }
    _check_lane(protein, lane, len(batch.lanes), taken, proposed=False)
    params = {
        "band_id": band_id,
        "protein_id": protein.id,
        "from_lane": band.lane_index,
        "lane_index": lane,
    }

    def change(draft: Project) -> dict[str, JsonValue] | None:
        edited_protein, edited = draft.batch.find_band(band_id)
        edited.lane_index = lane
        edited.manually_edited = True
        edited.bands_found = None
        return _drop_undetected(edited_protein, lane, edited.band_index)

    _apply(
        session,
        "set_box_lane",
        change,
        lambda replaced: {**params, "replaced_undetected": replaced},
    )


@_locked
def set_box_size(session: ProjectSession, protein_id: str, size: BoxSize) -> None:
    """Change a protein's fitted size. Its boxes become that size plus its
    padding (:func:`set_box_padding`) on each side: every box is re-sized around
    its centre (shifted inside the image), and every band on the image is
    re-quantified (the other proteins' rings leave out the resized boxes). A
    typed fitted size is kept until the next fit that needs more, the next seed
    click on the protein while it has no box, or the next row that keeps none of
    its boxes. The same fitted size is a no-op.

    Refused, changing nothing: ``size`` not a :class:`~proteia.core.model.BoxSize`
    or smaller than the fitted size in a dimension where the padding would then
    be more than half of it (``INVALID_INPUT``: type at least twice the padding,
    or lower the padding first, which only a protein with a box can); boxes
    beyond the image (``SIZE_OUT_OF_BOUNDS``) or overlapping each other
    (``SIZE_WOULD_OVERLAP``); a box at the new size, padding included,
    overlapping another protein's box on the image by more than
    :data:`COVER_SHARE` of the smaller box's area (``OVERLAP``, naming those
    boxes, :func:`_cover_refusal`). The log entry holds the size used
    (``box_size``) and the fitted size typed (``fitted_size``)."""
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    image = batch.find_image(protein.image_id)
    size = _box_size(size)
    fitted, pad = protein.fitted_size, protein.box_padding
    if size == fitted:
        return
    for typed, current, (name, where, dimension) in zip(
        (size.width, size.height), (fitted.width, fitted.height), _PADDING, strict=True
    ):
        padding = getattr(pad, name)
        if typed < current and 2 * padding > typed:
            larger = f"type a {dimension} of at least {2 * padding}"
            if protein.bands:  # without a box, set_box_padding refuses to lower it
                lower = (
                    f"lower that padding to {typed // 2} px or less"
                    if typed > 1
                    else "remove that padding"
                )
                larger = f"{lower} first, or {larger}"
            raise _invalid(
                f"a fitted {dimension} of {typed} px would leave the {padding} px of padding"
                f" {where} more than half of it; {larger}"
            )
    effective = _within(_padded(size, pad), image, pad)
    rects = [band.box.rect(protein.box_size) for band in protein.bands]
    resized = boxes.resize_all(rects, effective, width=image.width, height=image.height)
    if resized is None:
        raise OperationError(
            ErrorCode.SIZE_WOULD_OVERLAP,
            f"box size {boxes.size_words(effective, pad)} would make boxes of"
            f" {protein.name!r} overlap",
        )
    covered = _covered(batch, protein, [r for r, o in zip(resized, rects, strict=True) if r != o])
    if covered:
        raise _cover_refusal(
            covered,
            f"at the box size {boxes.size_words(effective, pad)}, boxes of {protein.name!r}"
            " would overlap",
        )
    array = session.pixels(image.id) if protein.bands else None

    def change(draft: Project) -> None:
        edited = draft.batch.find_protein(protein_id)
        edited.box_size = effective
        for band, old, new in zip(edited.bands, rects, resized, strict=True):
            if new != old:
                _set_box(band, new)
        if array is not None:
            _quantify_image(draft, edited.image_id, array)
        _refresh_box_mws(session.project, draft)

    _apply(
        session,
        "set_box_size",
        change,
        lambda _: {
            "protein_id": protein_id,
            "box_size": _size(effective),
            "fitted_size": _size(size),
        },
    )


def _padding_overlap(
    protein: Protein,
    rects: Sequence[Rect],
    old: BoxPadding,
    new: BoxPadding,
    clashing: Sequence[int],
    image: ImageRef,
) -> OperationError:
    """The refusal of a padding ``new`` that makes the protein's boxes (at
    ``rects`` now, under ``old``) overlap, the boxes ``clashing`` named. It names
    the direction raised (left and right if raising it alone makes boxes
    overlap), and the most that fits of it, the other held as asked, when any
    does: overlap only grows with a padding, so a bisection finds it."""
    fitted = protein.fitted_size

    def fits(pad: BoxPadding) -> bool:
        _, hits = boxes.resize_checked(
            rects, _padded(fitted, pad), width=image.width, height=image.height
        )
        return not hits

    raised = [name for name, _, _ in _PADDING if getattr(new, name) > getattr(old, name)]
    alone = BoxPadding(across=new.across, along=old.along)
    name = "across" if raised == ["across"] or ("across" in raised and not fits(alone)) else "along"
    other = "along" if name == "across" else "across"
    value = getattr(new, name)

    def at(v: int) -> BoxPadding:
        return BoxPadding(**{**new.model_dump(), name: v})

    most = None
    if fits(at(0)):
        most, over = 0, value  # fits(at(most)); not fits(at(over))
        while over - most > 1:
            mid = (most + over) // 2
            most, over = (mid, over) if fits(at(mid)) else (most, mid)
    held = ""
    if other in raised:
        held = f", with {getattr(new, other)} px {_WHERE[other]},"
    lanes = lanes_phrase({protein.bands[i].lane_index for i in clashing})
    message = (
        f"{value} px of padding {_WHERE[name]}{held} would make the boxes of"
        f" {protein.name!r} in {lanes} overlap"
    )
    if most is not None:
        message += f"; at most {most} px fits"
    return OperationError(
        ErrorCode.SIZE_WOULD_OVERLAP, message, ids=[protein.bands[i].id for i in clashing]
    )


@_locked
def set_box_padding(
    session: ProjectSession,
    protein_id: str,
    *,
    across: int | Keep = KEEP,
    along: int | Keep = KEEP,
) -> PaddingChange:
    """Set how far every box of a protein extends beyond its fitted size, in
    whole pixels on each side: ``across`` left and right, ``along`` above and
    below; ``KEEP`` leaves one unchanged. The fitted size is kept. Every box is
    re-sized to fitted + 2 * padding around its own centre, so each lane's box
    keeps its place in the row. A box the image edge stops is shifted inside it
    (``edge_shifted``). Every band on the image is re-quantified. A padding may
    be raised only to half the fitted size and lowered at any time, and only
    once the protein has a box. Later clicks, row boxes and MW placement fit the
    fitted size and keep the padding. The same padding is a no-op.

    Refused, in this order, changing nothing: a given value not an integer, or
    below 0 (``INVALID_INPUT``); then, unless the padding is the same (a no-op,
    even when a fit has left it above half), a protein with no box, or a value
    raised above half its fitted size (``INVALID_INPUT``); boxes beyond the
    image (``SIZE_OUT_OF_BOUNDS``); the protein's own boxes overlapping
    (``SIZE_WOULD_OVERLAP``, with those boxes, and the most of the raised
    direction that fits); a box of the protein at the new size overlapping
    another protein's box on the image by more than :data:`COVER_SHARE` of the
    smaller box's area (``OVERLAP``, naming those boxes, :func:`_cover_refusal`,
    as :func:`set_box_size`; raised or lowered: a smaller box may lie more
    inside another's); an image file changed or unreadable. Other proteins'
    boxes may overlap its boxes by less: they are reported (``overlapping``),
    not refused. No box counts as edited by the user: the padding is a
    protein-level setting.

    The log entry holds both directions as they took effect and the size used.
    """
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    given = {"across": across, "along": along}
    wanted = {
        name: _int(given[name], f"padding {where}")
        for name, where, _ in _PADDING
        if given[name] is not KEEP
    }
    for name, value in wanted.items():
        if value < 0:
            raise _invalid(f"padding {_WHERE[name]} must be 0 or more, not {value}")
    old = protein.box_padding
    new = BoxPadding(**{**old.model_dump(), **wanted})
    fitted = protein.fitted_size
    if new == old:
        return PaddingChange(
            box_size=protein.box_size.model_copy(),
            fitted_size=fitted,
            padding=old.model_copy(),
            net_change=None,
            edge_shifted=(),
            overlapping=(),
        )
    if not protein.bands:
        raise _invalid(
            f"place a box of {protein.name!r} first: its padding follows the fitted size the"
            " bands set"
        )
    for name, where, dimension in _PADDING:
        value, half = getattr(new, name), getattr(fitted, dimension) // 2
        if value > getattr(old, name) and value > half:
            raise _invalid(
                f"padding {where} can be raised to at most {half} px for {protein.name!r}"
                f" (half its fitted {dimension}, {getattr(fitted, dimension)} px), not {value}"
            )
    image = batch.find_image(protein.image_id)
    size = _within(_padded(fitted, new), image, new)
    rects = [band.box.rect(protein.box_size) for band in protein.bands]
    resized, clashing = boxes.resize_checked(rects, size, width=image.width, height=image.height)
    if clashing:
        raise _padding_overlap(protein, rects, old, new, clashing, image)
    covered = _covered(batch, protein, [r for r, o in zip(resized, rects, strict=True) if r != o])
    if covered:
        raise _cover_refusal(
            covered,
            f"at the box size {boxes.size_words(size, new)}, boxes of {protein.name!r}"
            " would overlap",
        )
    array = session.pixels(image.id)

    def change(draft: Project) -> None:
        edited = draft.batch.find_protein(protein_id)
        edited.box_padding = new.model_copy()
        edited.box_size = size
        for band, before, after in zip(edited.bands, rects, resized, strict=True):
            if after != before:
                _set_box(band, after)
        _quantify_image(draft, edited.image_id, array)
        _refresh_box_mws(session.project, draft)

    params = {
        "protein_id": protein_id,
        "box_padding": {"across": new.across, "along": new.along},
        "box_size": _size(size),
    }
    _apply(session, "set_box_padding", change, lambda _: params)

    after = session.project.batch
    nets = {band.id: band.net for band in after.find_protein(protein_id).bands}
    changes = [nets[band.id] / band.net - 1 for band in protein.bands if band.net > 0]
    da, dl = new.across - old.across, new.along - old.along
    shifted = tuple(
        band.id
        for band, (x0, y0, x1, y1), rect in zip(protein.bands, rects, resized, strict=True)
        if rect != (x0 - da, y0 - dl, x1 + da, y1 + dl)
    )
    others = [
        band.id
        for other in batch.proteins
        if other.image_id == image.id and other.id != protein_id
        for band in other.bands
        if boxes.overlaps_any(band.box.rect(other.box_size), resized)
        and not boxes.overlaps_any(band.box.rect(other.box_size), rects)
    ]
    remeasured, largest = _remeasured(batch, after, image.id, protein_id)
    return PaddingChange(
        box_size=size,
        fitted_size=fitted,
        padding=new,
        net_change=(min(changes), max(changes)) if changes else None,
        edge_shifted=shifted,
        overlapping=tuple(others),
        remeasured=remeasured,
        largest_change=largest,
    )


@_locked
def clear_boxes(session: ProjectSession, protein_id: str) -> ClearedBoxes:
    """Remove every box and every not-detected record of a protein, of every band
    index, in one change; undo is the way back.

    The protein's box size and padding are kept (the width and height fields
    show the fitted size). A seed click or a row box on the protein, which then
    has no boxes, sets the fitted size afresh from the band or bands found, and
    the padding stays even if it is then more than half of it; a fixed box uses
    the kept size. A protein with neither boxes nor records is a no-op. The
    other proteins' bands on the image are re-quantified: their rings no longer
    leave out the boxes. The log entry lists the band ids with their lanes, and
    the records in full (``dropped_undetected``).
    """
    protein = session.project.batch.find_protein(protein_id)
    removed = [band.id for band in protein.bands]
    lanes = [band.lane_index for band in protein.bands]
    image_id = protein.image_id
    array = _pixels_left(session, image_id, removed) if removed else None

    def change(draft: Project) -> list[dict[str, JsonValue]]:
        edited = draft.batch.find_protein(protein_id)
        edited.bands = []
        if array is not None:
            _quantify_image(draft, image_id, array)
        return _drop_undetected_where(edited, lambda _: True)

    dropped = _apply(
        session,
        "clear_boxes",
        change,
        lambda dropped: {
            "protein_id": protein_id,
            "removed": list(removed),
            "lane_indices": list(lanes),
            "dropped_undetected": list(dropped),
        },
    )
    return ClearedBoxes(
        band_ids=tuple(removed),
        undetected=tuple((record["lane_index"], record["band_index"]) for record in dropped),
    )


# RowDetectError codes as refusals.
_ROW_ERRORS: Final = {
    "invalid_row": ErrorCode.INVALID_INPUT,
    "invalid_image": ErrorCode.UNREADABLE_IMAGE,  # with the image's id
    "row_outside_image": ErrorCode.OUT_OF_IMAGE,
    "row_too_small": ErrorCode.ROW_TOO_SMALL,
}
# The sources of boxes the user placed: kept by a row that finds no band there.
_PLACED_BY_USER: Final = frozenset({ProposalSource.CLICK, ProposalSource.MANUAL})


def _off_lanes(centres: Mapping[int, float], expected: Mapping[int, float]) -> list[int]:
    """The lanes, in order, whose band lies nearer a neighbouring lane than its
    own.

    ``centres`` maps a lane to its band's centre x; ``expected`` maps each of
    those lanes and both its neighbours to the lane's expected x
    (:func:`~proteia.core.project.lane_positions`: rising with the lane, or
    falling on lanes numbered right to left). A band is off its lane when it
    lies more than half the local pitch from the lane's expected x, the local
    pitch being the distance to the neighbouring lane on the band's side, so
    uneven spacing is judged on each side by its own step.
    """
    off = []
    for lane, x in sorted(centres.items()):
        at, after = expected[lane], expected[lane + 1]
        neighbour = after if (after - at) * (x - at) > 0 else expected[lane - 1]
        if abs(x - at) > abs(neighbour - at) / 2:
            off.append(lane)
    return off


def _named_lanes(anchors: Sequence[tuple[float, int]], off: Iterable[int]) -> set[int]:
    """The kept lanes whose boxes decided that ``off`` lanes are off
    (:func:`_off_lanes`): those each off lane's expected x, and its neighbours'
    (the half-pitch yardstick), are read from
    (:func:`~proteia.core.project.anchoring_lanes`). A box in the wrong lane
    beside an off lane can be the cause, so it is named too."""
    return anchoring_lanes(anchors, {lane + d for lane in off for d in (-1, 0, 1)})


def _in_words(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _misnumbered_lanes(
    batch: Batch, image: ImageRef, without: Collection[str]
) -> OperationError | None:
    """The refusal of a row (:func:`detect_row_boxes`) on an image whose lanes
    already placed are numbered inconsistently, or None.

    The lanes placed are the first-band boxes on the image, of every protein
    and from any source (:func:`~proteia.core.project.lane_anchors`), less
    those whose ids are in ``without`` and the boxes grown from a click
    (source ``click``, edited by hand or not) of a protein whose fitted size
    is wider than the lane pitch (:func:`~proteia.core.project.lane_pitch` of
    each protein's): such a box is centred on what grew from the click, which
    may span several lanes' bands (touching bands), so its centre need not lie
    on its lane's column. The padding is left out: it widens every box of the
    protein around its centre, and what grew is no wider for it. A box dropped
    where the user clicked, moved there by hand or centred on a band a
    detector found shows its lane whatever its width.

    The lanes placed are numbered inconsistently when two proteins' boxes,
    each in two or more kept lanes, number the lanes opposite ways
    (:func:`~proteia.core.project.lanes_run_right_to_left` of each protein's;
    ``ids``: those proteins, the ones numbering them left to right first), or
    else when one lane's boxes have centres more than half the lane pitch
    apart, one lane number on two columns (``ids``: those lanes' boxes, in
    lane order). Boxes each within a quarter pitch of their lane's column never
    lie that far apart; two boxes dropped on opposite edges of a band wider
    than half the pitch can.
    """
    proteins = [p for p in batch.proteins if p.image_id == image.id]

    def anchors_of(skipped: Collection[str]) -> dict[str, list[tuple[float, int]]]:
        return {
            p.id: lane_anchors(batch, image, without=skipped, only={b.id for b in p.bands})
            for p in proteins
        }

    anchors = anchors_of(without)
    pitch = lane_pitch(anchors.values())
    if pitch is None:  # no protein has two kept lanes
        return None
    grown = {
        b.id
        for p in proteins
        if p.fitted_size.width > pitch
        for b in p.bands
        if b.source is ProposalSource.CLICK
    }
    if grown:
        anchors = anchors_of({*without, *grown})
    ways = {p.id: lanes_run_right_to_left(anchors[p.id]) for p in proteins}
    ltr = [p for p in proteins if ways[p.id] is False]
    rtl = [p for p in proteins if ways[p.id] is True]
    fix = "fix the lane numbers of the boxes already on this image first"
    if ltr and rtl:
        return OperationError(
            ErrorCode.ROW_LANES_UNCLEAR,
            "the lanes already placed on this image are numbered both ways: the boxes of"
            f" {_in_words([repr(p.name) for p in ltr])} left to right, those of"
            f" {_in_words([repr(p.name) for p in rtl])} right to left; {fix}",
            ids=[p.id for p in ltr + rtl],
        )
    columns: dict[int, list[float]] = {}
    for p in proteins:
        for cx, lane in anchors[p.id]:
            columns.setdefault(lane, []).append(cx)
    apart = sorted(lane for lane, xs in columns.items() if max(xs) - min(xs) > pitch / 2)
    if not apart:
        return None
    # Each protein has one first-band box per lane: those anchored in the lanes.
    held = {p.id: {lane for _, lane in anchors[p.id]} for p in proteins}
    boxes_in = [
        (lane, p, b.id)
        for lane in apart
        for p in proteins
        for b in p.bands
        if b.band_index == 0 and b.lane_index == lane and lane in held[p.id]
    ]
    named = [p for p in proteins if any(q is p for _, q, _ in boxes_in)]
    where = "lane" if len(apart) == 1 else "lanes"
    return OperationError(
        ErrorCode.ROW_LANES_UNCLEAR,
        f"the lanes already placed on this image are numbered inconsistently: in {where}"
        f" {_in_words([str(lane_number(lane)) for lane in apart])}, the boxes of"
        f" {_in_words([repr(p.name) for p in named])} lie more than {pitch / 2:.0f} px"
        f" (half the lane pitch) apart; {fix}",
        ids=[band_id for _, _, band_id in boxes_in],
    )


def _row_refusal(found: rowdetect.RowDetection, x0: int, x1: int) -> OperationError:
    """The refusal of a row whose detection :func:`detect_row_boxes` cannot
    commit, worded by what the detector saw; ``x0`` and ``x1`` are the row
    box's sides, clipped to the image.

    Refused by a refusing flag (``ROW_LANES_UNCLEAR``): a box that cuts through
    the bands says so first (cause ``cut_by_row_box``: a lane ``cut``, whether
    its band's extent reaches the box's edge or the band peaks on the edge row
    and was left out), since drawn over every lane it can still misread them,
    unless bands off the row's line are all that refuses it (then as below,
    the cut said too: the lanes were read); then a box whose left or right
    edge cuts through a band and so leaves lanes out (cause ``side_signal``:
    ``lanes_outside_row``, and a lane of signal rising into the box's side
    past the lanes holding bands, at an end where a lane's expected centre
    lies outside the box); then bands that do not lie on one row
    (``ROW_OFF_LINE``, cause ``off_row_line``: a box more than
    :data:`~proteia.core.rowdetect.ROW_LINE_K` box heights off the row's
    line, as a box over two rows, or over a lane whose band lies off the row,
    places one; every box off it, two rows and neither the row's), naming
    those lanes unless another refusing flag leaves the reading unsettled;
    otherwise the first refusing flag (signal rising into
    the box's side elsewhere, as a dark image edge leaves, does not make the
    lanes unclear). No band sized (``NO_BAND_FOUND``), by the empty lanes'
    reasons: bands kept but in no
    lane (``unassigned``), signal only at the box's top or bottom edge
    (``edge_signal``: a box through the bands, or over a neighbouring row),
    only signal rising into its left or right edge (``side_signal``: a band
    the box cuts through there, or a dark image edge), only a line or strip
    across the lanes (``line``: a frame line, a dark strip along the image's
    edge), only a streak or stain filling its height (``artefact``), a box
    with too little membrane to measure the bands against
    (``too_little_membrane``:
    :attr:`~proteia.core.rowdetect.RowDetection.membrane_shift` at least
    :data:`~proteia.core.rowdetect.MEMBRANE_SHIFT_K`: detection took a band for
    its membrane), else no band (``no_band``).

    ``detail`` holds the raw reason: the ``cause`` the words follow, the
    detector's ``flags`` and ``notes``, the reading's ``margin`` (None with no
    reading or no other one), the box's ``membrane_shift`` and each lane's
    ``lane_index``, ``reason``, ``snr``, ``cut`` and ``line_offset``. Only the
    ``off_row_line`` message names lanes: a reading the row box does not
    settle numbers them unreliably.
    """
    reasons = {lane.reason for lane in found.lanes}
    if found.refused:
        code = ErrorCode.ROW_LANES_UNCLEAR
        # Signal rising into the box's side past the bands, at an end where
        # lanes lie outside the box: the side edge cut a band and left them out.
        banded = [lane.expected_x for lane in found.lanes if lane.rect is not None]
        sides = [lane.expected_x for lane in found.lanes if lane.reason == "side_signal"]
        empty = [lane.expected_x for lane in found.lanes if lane.rect is None]
        side_cut = (
            "lanes_outside_row" in found.flags
            and bool(banded)
            and any(
                (x < min(banded) and any(e < x0 for e in empty))
                or (x > max(banded) and any(e > x1 for e in empty))
                for x in sides
            )
        )
        off = [
            lane.lane
            for lane in found.lanes
            if lane.line_offset is not None and abs(lane.line_offset) > rowdetect.ROW_LINE_K
        ]
        # The lanes, where the reading settles them.
        settled = not any(
            flag in rowdetect.REFUSING_FLAGS and flag != "off_row_line" for flag in found.flags
        )
        cut = any(lane.cut for lane in found.lanes)
        if cut and not (off and settled):
            cause = "cut_by_row_box"
            message = (
                "the row box cuts through the bands, so it does not show which lane each band"
                " is in; draw it over the whole band height and every declared lane, empty end"
                " lanes included"
            )
        elif side_cut:
            cause = "side_signal"
            message = (
                "the row box's left or right edge cuts through a band, so it does not show"
                " which lane each band is in; draw it over the whole bands of every declared"
                " lane, empty end lanes included"
            )
        elif off:
            code = ErrorCode.ROW_OFF_LINE
            cause = "off_row_line"
            if not settled:
                which, lie, those, fix = "some bands found", "lie", "some bands", ""
            elif len(off) == 1:
                which, lie, those = f"the band found in {lanes_phrase(off)}", "lies", "that band"
                fix = ", or box that lane by clicking its band"
            else:
                which, lie, those = f"the bands found in {lanes_phrase(off)}", "lie", "those bands"
                fix = ", or box those lanes by clicking their bands"
            if settled and len(off) == sum(lane.rect is not None for lane in found.lanes):
                # Every box off the line: two rows, neither the row's.
                what = (
                    f"{which} {lie} on two rows, more than {rowdetect.ROW_SMILE:g} box heights"
                    " apart: the row box covers more than one row"
                )
            else:
                what = (
                    f"{which} {lie} above or below the row's line through the other bands, by"
                    f" more than {rowdetect.ROW_LINE_K:g} of a box's height: the row box covers"
                    " more than one row, or "
                    + (f"cuts {those}" if cut else f"{those} {lie} off the row")
                )
            lead, whole = (
                ("the row box cuts through the bands, and ", ", over the whole band height")
                if cut
                else ("", "")
            )
            message = f"{lead}{what}; draw it over one row only{whole}{fix}"
        else:
            cause = next(flag for flag in found.flags if flag in rowdetect.REFUSING_FLAGS)
            message = (
                "the row box does not show which lane each band is in; draw it over every"
                " declared lane, empty end lanes included, or place the boxes by clicking"
            )
    else:
        code = ErrorCode.NO_BAND_FOUND
        if "unassigned" in reasons:
            cause = "unassigned"
            message = (
                "bands were found in the row box but do not fit the lanes; draw it over every"
                " declared lane, empty end lanes included, or place the boxes by clicking"
            )
        elif "edge_signal" in reasons:
            cause = "edge_signal"
            message = (
                "the only signal in the row box lies at its top or bottom edge: the box cuts"
                " through the bands or reaches into a neighbouring row; include the whole band"
                " height"
            )
        elif "side_signal" in reasons:
            cause = "side_signal"
            message = (
                "the only signal in the row box rises into its left or right edge: the box cuts"
                " through a band there, or the image's edge is dark; draw it over whole bands"
            )
        elif "line" in reasons:
            cause = "line"
            message = (
                "the only signal in the row box runs across the lanes as a line or strip, not as"
                " bands: a frame line, or a dark strip along the image's edge; draw the box over"
                " the bands only"
            )
        elif "artefact" in reasons:
            cause = "artefact"
            message = (
                "the only signal in the row box runs through its whole height: a streak or"
                " stain, or bands the box cuts through; include the whole band height"
            )
        elif found.membrane_shift >= rowdetect.MEMBRANE_SHIFT_K:
            cause = "too_little_membrane"
            message = (
                "the row box holds too little membrane to measure the bands against; include"
                " some membrane above and below the bands"
            )
        else:
            cause = "no_band"
            message = "no band found in the row box"
    margin = found.margin
    detail: dict[str, JsonValue] = {
        "cause": cause,
        "flags": list(found.flags),
        "notes": list(found.notes),
        "margin": margin if margin is not None and math.isfinite(margin) else None,
        "membrane_shift": found.membrane_shift,
        "lanes": [
            {
                "lane_index": lane.lane,
                "reason": lane.reason,
                "snr": lane.snr,
                "cut": lane.cut,
                "line_offset": lane.line_offset,
            }
            for lane in found.lanes
        ],
    }
    return OperationError(code, message, detail=detail)


def _bands_found(
    lane: rowdetect.LaneDetection,
    rect: Rect,
    fitted: mwcal.Calibration | mwcal.NoCalibration,
    tolerance: float,
    row: Rect,
    height: int,
) -> int:
    """A placed band's count (#58, D10): its lane's bands in the count window
    around ``rect`` (:func:`~proteia.core.results.count_window`), or in the rows
    of the row box ``row`` (clipped to the image's ``height``) where the image
    has no curve there."""
    window = results.count_window(fitted, rect, tolerance)
    top, bottom = window if window is not None else (max(0, row[1]), min(height, row[3]))
    return rowdetect.bands_in(lane, top, bottom)


def _row(row: object) -> Rect:
    """A row box as given: four ints ``(x0, y0, x1, y1)`` with ``x0 < x1`` and
    ``y0 < y1``, else ``INVALID_INPUT``. Text and bytes are sequences, but not
    of coordinates."""
    if (
        isinstance(row, str | bytes | bytearray | memoryview)
        or not isinstance(row, Sequence)
        or len(row) != 4
    ):
        raise _invalid(f"row must be (x0, y0, x1, y1), not {row!r}")
    x0, y0, x1, y1 = (_int(v, "row coordinate") for v in row)
    if x1 <= x0 or y1 <= y0:
        raise _invalid(f"row {(x0, y0, x1, y1)} is empty or inverted")
    return x0, y0, x1, y1


@_locked
def detect_row_boxes(session: ProjectSession, protein_id: str, row: Rect) -> RowPlacement:
    """Detect one band per declared lane in the row box the user dragged over a
    protein's row (:func:`~proteia.core.rowdetect.detect_row`) and commit the
    outcome for the protein's band index 0.

    ``row`` is ``(x0, y0, x1, y1)`` in image pixels, end-exclusive (a client
    normalizes drag corners with :func:`~proteia.core.boxes.normalize_corners`);
    it is clipped to the image. It must span every declared lane, include=no
    lanes and empty end lanes too: the lanes are read from the bands, and
    detection runs in every lane. In each lane:

    * a box edited by hand (moved, or given another lane) is kept as it is,
      whatever was found there, and so is a box the user placed (source
      ``click`` or ``manual``) in a lane where no band is found; both are
      reported (``kept_lanes``), and the lane gets no record;
    * otherwise this detection's outcome replaces what the lane held (a box
      nobody edited, a not-detected record, or nothing). A band found there is
      placed, with source ``row_box``; over a box nobody edited, whoever placed
      it, it is placed in place: the band keeps its id and takes the detected
      box, source and net (a box the band found leaves where it was keeps what
      its position gave it, an apparent MW), so a second drag corrects the
      first without undo, and the same drag again changes nothing. A box a
      detector placed (``row_box``, ``mw_guided``) in a lane where no band is
      found goes (``removed_band_ids``). A lane where nothing reaches the
      detection limit (``no_band``) gets a not-detected record of the slot the
      detector measured, unless that slot rests on one band alone: when the
      row finds a band in one lane only, the other lanes' slots are that
      band's position stepped by a pitch the box's width suggests, so no
      record is written there (``unlocated_lanes``); the lanes already
      placed on the image check that band's lane, not the slots. Any other
      empty lane (a stain or streak, a line or strip across the lanes, only a
      neighbouring row's signal or signal rising into the box's side, a piece
      left unassigned) is left with neither a box nor a record
      (``unmeasured_lanes``).

    So a box the user clicked into a lane the detector cannot read or finds
    nothing in stays through the next drag over its row, while one clicked
    over a band the detector finds takes that band until it is edited by hand.

    Boxes and records of another band index are left alone.

    Each band placed or replaced in place gets its band count (#58, D10,
    ``bands_found``): the band and the other bands its lane holds in the count
    window around its box (:func:`~proteia.core.results.count_window`: the
    protein's MW tolerance either way on the image's calibration at the box's
    centre, or the row box's rows without a curve there), as the detector's
    peaks show them (:func:`~proteia.core.rowdetect.bands_in`).

    The detector's size is the fitted size; the protein's padding
    (:func:`set_box_padding`) is added to it, on each side. If a box of the
    protein survives (one kept, or of another band index), the size only grows,
    as a seed click grows it, even when every band found is in a kept lane and
    nothing is placed: the survivors are re-centred and the new boxes centred on
    the detected ones. That size making survivors overlap, or new boxes overlap
    each other, is refused (``SIZE_WOULD_OVERLAP``), and so is a new box over a
    survivor (``OVERLAP``, with the survivor), or a box it places (new or in
    place), or a survivor grown to the size, that overlaps another protein's
    box on the image by more than :data:`COVER_SHARE` of the smaller box's
    area, each at its size padding included (``OVERLAP``, naming those boxes,
    :func:`_cover_refusal`: a row dragged over another protein's row, or a
    loading control's row over its target's). If none survives, this
    detection sets the fitted size afresh, and the padding stays, even if it
    is then more than half of it. A box may then extend beyond the row box,
    never beyond the image. Every band on the image, of every protein, is
    then re-quantified with the project's background method; the detector's
    own local background only finds the bands.

    The lanes already placed on the image are its first-band boxes, of any
    protein and from any source (:func:`~proteia.core.project.lane_anchors`),
    less this protein's boxes a detector placed that nobody edited, which give
    way to the row whatever it finds.

    The row reads its lanes left to right, or right to left (lane 0 at the
    box's right end, ``right_to_left``) when those lanes run that way
    (:func:`~proteia.core.project.lanes_run_right_to_left`). With fewer than
    two such lanes, the way this protein's boxes a detector placed run, as the
    last row read them (so once the row has replaced the boxes the user placed
    that turned it, the same drag again still reads the lanes their way); with
    fewer than two of those either, left to right.

    Before anything is detected, the lanes already placed, and this protein's
    boxes a detector placed when they turn the row, must be numbered
    consistently, or the row is refused (``ROW_LANES_UNCLEAR``,
    :func:`_misnumbered_lanes`): the user fixes the lane numbers of the boxes
    already on the image first, since the row cannot tell which of them are
    right. So the last row's boxes, turning the row against another protein's
    box, refuse it until they are removed or that row is undone: the row
    cannot tell whether its last reading or the other box is right. They are
    not numbered consistently when two proteins' boxes, each in two or more
    lanes, number the lanes opposite ways (``ids``: those proteins), or when
    one lane's boxes lie more than half the lane pitch apart, one lane number
    on two columns (``ids``: those boxes). A box dragged that far from its
    lane on purpose refuses the row too. A box grown from a click is left out
    of this check when the protein's boxes are wider than the lane pitch: it
    is centred on what grew, which may span several lanes' bands (touching
    bands), so its centre need not show its lane's column.

    The lanes already placed then check the reading, since a row box that
    leaves out a banded end lane can be read a lane off with no refusing flag;
    the lanes that turn the row check it, so the check reads the lanes the way
    the row was read (the last row's boxes, which turn it only when those
    lanes show no direction, check nothing: the row replaces them). This
    protein's boxes the user placed are among them, even where the row
    replaces them: they show where the user put each lane. Given two
    anchored lanes, each lane's expected x is interpolated between them and,
    past them, stepped by the row's own pitch
    (:func:`~proteia.core.project.lane_positions`: the step of two close
    lanes, repeated over many, drifts). A band found whose extent's centre
    lies more than half the local pitch from its lane's expected x
    (:func:`_off_lanes`) refuses the row (``ROW_LANES_UNCLEAR``; ``ids``: the
    boxes of the lanes those bands' expected x, and their neighbours', are
    read from (:func:`_named_lanes`), in the order of
    :func:`~proteia.core.project.lane_anchor_ids`, since one of them may be in
    the wrong lane; ``detail``: ``{"cause": "off_lanes", "off_lanes": [...]}``,
    the lanes off as read). So a row read
    a lane or more off is refused wherever its bands lie; one squeezed into
    more lanes than the box covers (its pitch too small), only where its bands
    lie between the anchored lanes, since past them the row's pitch follows
    its own reading.

    With fewer than two such lanes nothing checks the reading: a first row
    box that also covers a ladder, labels or a neighbouring panel can be read
    a lane or more off (#111). A reading that does not fit the bands' own
    spacing is placed with the warning ``doubtful_lanes`` (see
    :mod:`~proteia.core.rowdetect`): the user checks its lane numbers, and
    Undo takes the row back. Once lanes on the image have checked the lane of
    every band found, each lying between the outermost anchored lanes (or on
    one), the warning and its note are dropped: its lane numbers line up with
    theirs. A band's lane past them keeps the warning, since there the
    expected x steps by the row's own pitch, which a reading squeezed into the
    wrong lanes fits.

    Also refused, changing nothing: ``row`` not four ints, or empty or
    inverted (``INVALID_INPUT``); no lanes (``NO_LANES``); lanes on the image
    numbered inconsistently, as above; a row outside the image
    (``OUT_OF_IMAGE``) or too small for its lanes (``ROW_TOO_SMALL``);
    non-finite pixels in the row (``UNREADABLE_IMAGE``, with the image: the
    image is at fault, not the row); bands that do not show which lane each
    is in (``ROW_LANES_UNCLEAR``, with no ids: the row box alone is at fault,
    whatever boxes the image holds); bands that do not lie on one row
    (``ROW_OFF_LINE``, with no ids: a box more than
    :data:`~proteia.core.rowdetect.ROW_LINE_K` box heights off the row's line
    through the others, as when the row box covers two rows); no band in any
    lane (``NO_BAND_FOUND``: with no band located, the lane slots would rest
    only on an even split of the box, so no record is written either). Those
    three are worded by what the detector saw, with its raw reason as the
    refusal's ``detail`` (:func:`_row_refusal`). The checks run in that order,
    the lanes on the image checking the reading next, then the size
    (:func:`~proteia.core.boxes.grow_to_fit_all`), then the other proteins'
    boxes the row's would overlap (those it places, then those it keeps).

    The log entry holds the row as given; each lane's first-band box after the
    change, as ``band_ids`` names it (placed, replaced in place or kept: its
    band id and rect), with the lane's outcome (the detector's reason, ``snr``
    to 2 decimals and ``expected_x`` to 1); the kept lanes, the band ids
    replaced in place or removed, the records written and dropped (in full),
    the lanes left without one because their slot rests on one band alone, the
    size after, the fitted pitch and noise, the detector's warnings and
    notes, whether the lanes were read right to left, the saturation level
    it was given (``saturated_at``, #121), and its settings
    (:func:`~proteia.core.rowdetect.settings`: dev builds share a version
    string, so the entry names the constants that placed the boxes). The
    answer also says which other proteins' nets on the image the row changed
    (``remeasured``, ``largest_change``); the log need not, since every net is
    stored whole.
    """
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    given = _row(row)
    n = len(batch.lanes)
    if n == 0:
        raise OperationError(ErrorCode.NO_LANES, "declare the lanes before detecting a row")
    image = batch.find_image(protein.image_id)
    width, height = image.width, image.height
    # The lanes on the image, less this protein's boxes that give way to the
    # row whatever it finds, turn the row and check it. Showing no direction,
    # the row reads the lanes the way those boxes run, the last row's way:
    # then they turn it, so they count among the lanes that must be numbered
    # consistently.
    detectors = {
        b.id
        for b in protein.bands
        if b.band_index == 0 and not b.manually_edited and b.source in DETECTING_SOURCES
    }
    anchors = lane_anchors(batch, image, without=detectors)
    runs = lanes_run_right_to_left(anchors)
    last = None
    if runs is None:
        last = lanes_run_right_to_left(lane_anchors(batch, image, only=detectors))
    misnumbered = _misnumbered_lanes(batch, image, detectors if last is None else ())
    if misnumbered is not None:
        raise misnumbered
    array = session.pixels(image.id)
    right_to_left = runs is True or last is True
    # Pixels saturated as the over-exposure checks count them: a hollow band
    # is reported as over-exposed, not as two bands (#121). None for an image
    # of unknown bit depth: no band is then hollow.
    saturated_at = saturation_level(
        clipping_depth(image.bit_depth, image.import_warnings),
        possible_clipping_depth(image.bit_depth, image.import_warnings),
        dark_on_light=image.polarity.dark_on_light,
    )
    try:
        # rowdetect's settings and size rule are the defaults, which the export
        # record reports (record.settings) with how saturated_at is chosen; the
        # rest comes from the image and its lanes, and the log keeps the
        # direction and the saturation level.
        found = rowdetect.detect_row(
            array,
            given,
            n,
            background=image.background,
            dark_on_light=image.polarity.dark_on_light,
            right_to_left=right_to_left,
            saturated_at=saturated_at,
        )
    except rowdetect.RowDetectError as exc:
        ids = (image.id,) if exc.code == "invalid_image" else ()
        raise OperationError(_ROW_ERRORS[exc.code], str(exc), ids=ids) from exc
    if found.refused or found.size is None:
        raise _row_refusal(found, max(0, given[0]), min(width, given[2]))

    # Band index 0, per lane: a box edited by hand stays, and so does one the
    # user placed where no band was found; any other gives way.
    banded = {lane.lane for lane in found.lanes if lane.rect is not None}
    kept = {
        b.lane_index: b.id
        for b in protein.bands
        if b.band_index == 0
        and (b.manually_edited or (b.source in _PLACED_BY_USER and b.lane_index not in banded))
    }
    yielding = {
        b.lane_index: b.id for b in protein.bands if b.band_index == 0 and b.lane_index not in kept
    }
    # The same lanes check the reading (none when they show no direction);
    # past them, the row's own pitch.
    expected = lane_positions(anchors, range(-1, n + 1), pitch=found.pitch)
    centres = {
        lane.lane: (lane.extent[0] + lane.extent[2]) / 2
        for lane in found.lanes
        if lane.extent is not None
    }
    off = _off_lanes(centres, expected) if expected else []
    if off:
        named = _named_lanes(anchors, off)
        placed_ids = lane_anchor_ids(batch, image, without=detectors)
        raise OperationError(
            ErrorCode.ROW_LANES_UNCLEAR,
            "the bands in the row box do not line up with the lanes already placed on this"
            " image; draw the box over every declared lane, empty end lanes included",
            ids=[
                band_id
                for band_id, (_, lane) in zip(placed_ids, anchors, strict=True)
                if lane in named
            ],
            detail={"cause": "off_lanes", "off_lanes": list(off)},
        )
    # The boxes that survive the commit: those kept, and those of another band
    # index.
    keeping = set(kept.values())
    surviving = [b for b in protein.bands if b.band_index > 0 or b.id in keeping]
    survivors = [b.id for b in surviving]
    old = [b.box.rect(protein.box_size) for b in surviving]
    placed = [lane for lane in found.lanes if lane.rect is not None and lane.lane not in kept]
    pad = protein.box_padding
    try:
        # The detector's size, the fitted size, plus the protein's padding; while
        # a box survives it only grows the size, even if every band found is in a
        # kept lane and nothing is placed.
        size, resized, new_rects = boxes.grow_to_fit_all(
            old,
            protein.box_size,
            [lane.rect for lane in placed],
            need=found.size,
            width=width,
            height=height,
            pad=pad,
        )
    except boxes.BoxRuleError as exc:
        grown = boxes.size_words(exc.size, pad)
        if exc.code == "overlap":
            raise OperationError(
                ErrorCode.OVERLAP,
                "a box of the row would overlap a box the row keeps",
                ids=[survivors[i] for i in exc.hits],
            ) from exc
        if exc.hits:  # the survivors overlap each other at the grown size
            message = (
                f"box size {grown} would make the boxes of {protein.name!r}"
                " that the row keeps overlap"
            )
        else:  # the new boxes do: at the size the kept boxes need, or under the padding
            message = f"the row's boxes would overlap each other at the box size {grown}"
            if surviving:
                message += f" that the kept boxes of {protein.name!r} need"
        if pad.across:
            message += "; lower the padding left and right"
        raise OperationError(ErrorCode.SIZE_WOULD_OVERLAP, message) from exc
    rects = {lane.lane: rect for lane, rect in zip(placed, new_rects, strict=True)}
    covered = _covered(batch, protein, rects.values())
    if covered:
        raise _cover_refusal(covered, "the row's boxes would overlap")
    covered = _covered(batch, protein, [r for r, o in zip(resized, old, strict=True) if r != o])
    if covered:
        raise _cover_refusal(
            covered,
            f"at the box size {boxes.size_words(size, pad)} the row needs, the boxes of"
            f" {protein.name!r} it keeps would overlap",
        )
    # A not-detected record's slot rests on the bands found, stepped by the
    # pitch past them: one band leaves the slots to a pitch the box's width
    # suggests. The lanes on the image check the band's own lane, not them.
    measured = [
        lane
        for lane in found.lanes
        if lane.reason == "no_band" and lane.window is not None and lane.lane not in kept
    ]
    located = len(banded) > 1
    records = measured if located else []
    unlocated = () if located else tuple(lane.lane for lane in measured)
    replaced = [yielding[lane] for lane in rects if lane in yielding]
    removed = [yielding[lane] for lane in sorted(yielding) if lane not in rects]
    # Each band's count, in the window around the box it gets (#58, D10).
    fitted = mwcal.calibration_for(batch.membrane_of(image.id), image.id)
    counts = {
        lane.lane: _bands_found(lane, rects[lane.lane], fitted, protein.mw_tolerance, given, height)
        for lane in placed
    }
    warnings = [flag for flag in found.flags if flag in rowdetect.WARNING_FLAGS]
    notes = found.notes
    # The lanes on the image checked the lane numbers of the bands between
    # them; past them, the row's own pitch stepped the expected x, which a
    # squeezed reading fits.
    checked = anchoring_lanes(anchors, range(n))  # the kept lanes
    if checked and all(min(checked) <= lane <= max(checked) for lane in centres):
        warnings = [flag for flag in warnings if flag != "doubtful_lanes"]
        notes = tuple(note for note in notes if note != found.doubt_note)

    def change(
        draft: Project,
    ) -> tuple[list[str | None], list[Rect | None], list[JsonValue], list[JsonValue]]:
        edited = draft.batch.find_protein(protein_id)
        old_size = edited.box_size
        edited.box_size = size
        by_id = {band.id: band for band in edited.bands}
        for band_id, before, after in zip(survivors, old, resized, strict=True):
            if after != before:
                _set_box(by_id[band_id], after)
        edited.bands = [by_id[band_id] for band_id in survivors]
        for lane, rect in rects.items():  # in lane order: new ids too
            if lane in yielding:  # the band found takes the box nobody edited
                band = by_id[yielding[lane]]
                if rect != band.box.rect(old_size):  # else it keeps what its position gave it
                    _set_box(band, rect)
                band.source = ProposalSource.ROW_BOX
            else:
                band = Band(
                    id=draft.new_id("band"),
                    lane_index=lane,
                    band_index=0,
                    box=Box(x=rect[0], y=rect[1]),
                    source=ProposalSource.ROW_BOX,
                    **_UNQUANTIFIED,
                )
            band.bands_found = counts[lane]
            edited.bands.append(band)
        # The boxes placed, moved and removed change every ring on the image.
        _quantify_image(draft, edited.image_id, array)
        _refresh_box_mws(session.project, draft)
        # Every band-index-0 record gives way to this run's outcome in its lane (a
        # kept box's lane holds none).
        dropped: list[JsonValue] = list(
            _drop_undetected_where(edited, lambda record: record.band_index == 0)
        )
        written: list[JsonValue] = []
        for lane in records:
            x0, y0, x1, y1 = lane.window
            record = UndetectedBand(
                lane_index=lane.lane,
                band_index=0,
                reason=UndetectedReason.BELOW_DETECTION_LIMIT,
                snr=lane.snr,
                threshold=rowdetect.DETECT_K,
                region=Region(x0=x0, y0=y0, x1=x1, y1=y1),
                source=ProposalSource.ROW_BOX,
            )
            edited.undetected.append(record)
            written.append(_undetected_json(protein_id, record))
        # Each lane's first-band box after the change: placed, replaced or kept.
        after = {b.lane_index: b for b in edited.bands if b.band_index == 0}
        band_ids = [after[lane].id if lane in after else None for lane in range(n)]
        lane_rects = [after[lane].box.rect(size) if lane in after else None for lane in range(n)]
        return band_ids, lane_rects, written, dropped

    def params(
        result: tuple[list[str | None], list[Rect | None], list[JsonValue], list[JsonValue]],
    ) -> _Params:
        band_ids, lane_rects, written, dropped = result
        return {
            "protein_id": protein_id,
            "row": list(given),
            "lanes": [
                {
                    "band_id": band_ids[lane.lane],
                    "rect": None if lane_rects[lane.lane] is None else list(lane_rects[lane.lane]),
                    "reason": lane.reason,
                    "snr": round(lane.snr, 2),
                    "expected_x": round(lane.expected_x, 1),
                }
                for lane in found.lanes
            ],
            "kept_lanes": sorted(kept),
            "replaced_band_ids": list(replaced),
            "removed_band_ids": list(removed),
            "undetected_written": written,
            "dropped_undetected": dropped,
            "unlocated_lanes": list(unlocated),
            "box_size": _size(size),
            "pitch": found.pitch,
            "noise": found.noise,
            "flags": list(warnings),
            "notes": list(notes),
            "right_to_left": right_to_left,
            "saturated_at": saturated_at,
            "settings": rowdetect.settings(),
        }

    prior = session.project
    band_ids, _, _, _ = _apply(session, "detect_row_boxes", change, params)
    if session.project is not prior and (warnings or unlocated):
        _log.info(
            "in %r: the row box of %s on %s: the detector warns of %s; lanes not located"
            " (one band found): %s",
            session.folder.name,
            protein_id,
            image.id,
            ", ".join(warnings) or "nothing",
            ", ".join(str(lane_number(lane)) for lane in unlocated) or "none",
        )
    # The other proteins' nets on the image the row changed; the same drag
    # again, a no-op, changes none.
    remeasured, largest = _remeasured(batch, session.project.batch, image.id, protein_id)
    empty = tuple(
        (lane.lane, lane.reason, lane.snr, lane.expected_x)
        for lane in found.lanes
        if lane.rect is None
    )
    recorded = tuple(lane.lane for lane in records)
    return RowPlacement(
        band_ids=tuple(band_ids),
        box_size=size,
        kept_lanes=tuple(sorted(kept)),
        replaced_band_ids=tuple(replaced),
        removed_band_ids=tuple(removed),
        undetected_lanes=recorded,
        unmeasured_lanes=tuple(
            lane for lane, *_ in empty if lane not in recorded and lane not in kept
        ),
        empty=empty,
        flags=tuple(warnings),
        notes=notes,
        right_to_left=right_to_left,
        remeasured=remeasured,
        largest_change=largest,
        unlocated_lanes=unlocated,
    )


# --- Not-detected records ---


@_locked
def remove_undetected(
    session: ProjectSession, protein_id: str, lane_index: int, *, band_index: int = 0
) -> None:
    """Remove a protein's not-detected record in a lane: the lane becomes "not
    measured", as if the detector had never looked there.

    The lane must be one of the declared lanes (``LANE_OUT_OF_RANGE``). No record
    at that lane and band index is a no-op. The log entry holds the whole record.
    """
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    lane = _int(lane_index, "lane index")
    band = _int(band_index, "band index")
    n = len(batch.lanes)
    if not 0 <= lane < n:
        raise OperationError(
            ErrorCode.LANE_OUT_OF_RANGE, f"lane {lane_number(lane)} is not one of the {n} lanes"
        )
    if band < 0:
        raise _invalid(f"band index must be 0 or more, not {band}")
    if all((u.lane_index, u.band_index) != (lane, band) for u in protein.undetected):
        return

    def change(draft: Project) -> dict[str, JsonValue] | None:
        return _drop_undetected(draft.batch.find_protein(protein_id), lane, band)

    _apply(
        session,
        "remove_undetected",
        change,
        lambda removed: {
            "protein_id": protein_id,
            "lane_index": lane,
            "band_index": band,
            "removed": removed,
        },
    )


# --- The background method ---


def unassessed_images(batch: Batch) -> list[str]:
    """The ids of the images whose bands were measured before Proteia looked
    for pixels near the detector limit (#112), in membrane then image order:
    an image that check assesses
    (:func:`~proteia.core.imaging.possible_clipping_depth` known: a lossy,
    colour or CMYK-converted image of 8- or 16-bit pixels) holding a band with
    no ``possibly_clipped`` flag, as a project saved before it holds.
    :func:`requantify` assesses them."""
    return [
        image.id
        for image in batch.iter_images()
        if possible_clipping_depth(image.bit_depth, image.import_warnings) is not None
        and any(
            band.possibly_clipped is None
            for protein in batch.proteins
            if protein.image_id == image.id
            for band in protein.bands
        )
    ]


@_locked
def requantify(session: ProjectSession) -> tuple[str, ...]:
    """Re-quantify what a project saved by an earlier version left unmeasured,
    in one change: switch the project to the local background
    (``ring_median_v1``) and re-quantify every band with it, or, on the local
    background already, re-quantify the images whose bands were never assessed
    for over-exposure (:func:`unassessed_images`). Return the ids of the images
    re-quantified, in membrane then image order.

    A project quantified before #83 keeps the legacy method (``global_median``)
    until this runs, so its stored nets never change unasked; every image with
    bands is re-quantified, which assesses them too. A project saved before
    #112 keeps its bands on a lossy, colour or CMYK image unassessed until an
    edit re-quantifies their image or this runs, which leaves every other image
    as it is. With nothing to do it is a no-op. The pixels are read image by
    image inside the change, and those not cached already are not kept, so
    memory holds one image beyond the cache; a missing or changed image file
    still refuses the whole change (``IMAGE_FILE_CHANGED``), which then changes
    nothing. The log entry names the method left (``from``), the method taken
    (``to``; the same for a project on the local background already) and the
    images re-quantified (``images``).
    """
    project = session.project
    old = project.background_method
    batch = project.batch
    if old == LOCAL_BACKGROUND_METHOD:
        images = unassessed_images(batch)
        if not images:
            return ()
    else:
        images = [image.id for image in batch.iter_images() if _bands_on(batch, image.id)]

    def change(draft: Project) -> None:
        draft.background_method = LOCAL_BACKGROUND_METHOD
        for image_id in images:
            _quantify_image(draft, image_id, session.pixels(image_id, keep=False))

    _apply(
        session,
        "requantify",
        change,
        lambda _: {"from": old, "to": LOCAL_BACKGROUND_METHOD, "images": list(images)},
    )
    return tuple(images)


# --- Undo and redo ---


def _restored(params: Mapping[str, Any], verb: str) -> Restored:
    """The :class:`Restored` that an undo's (``verb`` "undone") or a redo's log
    params describe."""

    def keys(name: str) -> tuple[tuple[str, int, int], ...]:
        return tuple((protein_id, lane, band) for protein_id, lane, band in params[name])

    return Restored(
        seq=params[f"{verb}_seq"],
        action=params[f"{verb}_action"],
        removed=tuple(params["removed"]),
        restored=tuple(params["restored"]),
        undetected_removed=keys("undetected_removed"),
        undetected_restored=keys("undetected_restored"),
    )


@_locked
def undo(session: ProjectSession) -> Restored:
    """Take back the last change still in effect: restore, whole, the state
    before it, as a logged change of its own (action ``undo``).

    Repeated undo walks back one change at a time, up to
    :data:`~proteia.core.session.UNDO_LIMIT` changes or to the state the session
    began with; an undo is never itself undone (:func:`redo` reverses it). The
    restored content is exactly what was committed, stored nets, flags and
    not-detected records included, and is never recomputed from pixels; ids,
    and ``next_id``, never go back, so no id is used twice. Image files are only
    checked to exist; their bytes are checked when their pixels are next read,
    or at export. Refused, changing nothing: nothing to undo
    (``NOTHING_TO_UNDO``), or an image file the state needs is missing
    (``IMAGE_FILE_CHANGED``, with the images).

    The log entry names the change taken back (``undone_seq``,
    ``undone_action``) and the entry whose content it returns to
    (``returns_to_seq``, whose ``content_hash`` it repeats; null for content no
    entry recorded, such as a ``project.json`` edited by hand before the
    session), then lists what :class:`Restored` lists (``removed``,
    ``restored``, ``undetected_removed``, ``undetected_restored``; record keys
    as ``[protein id, lane index, band index]``).
    """
    return _restored(session._move("undo"), "undone")


@_locked
def redo(session: ProjectSession) -> Restored:
    """Make an undone change again: restore the state it left, as a logged
    change (action ``redo``), whose params mirror an undo's (``redone_seq``,
    ``redone_action``, and ``returns_to_seq``, which is ``redone_seq``). Any
    other change clears what can be redone. Refused, changing nothing: nothing
    to redo (``NOTHING_TO_REDO``), or a missing image file
    (``IMAGE_FILE_CHANGED``)."""
    return _restored(session._move("redo"), "redone")


# --- Results, export, save ---


@dataclass(frozen=True)
class ComputedView:
    """One committed project and every result computed from it: what a client
    shows together (the project's state, its table and its charts) comes from
    this one snapshot."""

    project: Project
    results: Results


def compute_view(
    session: ProjectSession,
    *,
    plot_conditions: Collection[str] | None = None,
    error_type: ErrorType | str = ErrorType.SD,
    method: ReduceMethod | str = ReduceMethod.MEAN,
    statistics: StatisticsSetting | Mapping[str, str] | None = None,
) -> ComputedView:
    """The committed project with every result of it
    (:func:`~proteia.core.results.compute_results`), which warn about nets
    measured on a reading no longer made (:func:`_outdated_readings`).

    Reads ``session.project`` once, so a change committed meanwhile is in neither
    half of the view: no lock, no pixels, no autosave. ``error_type``,
    ``method`` and ``statistics`` may be their raw values (``"SEM"``,
    ``"mean"``, ``{"family": "welch"}``); an unknown value is refused
    (``INVALID_INPUT``).
    """
    error_type = _member(ErrorType, error_type, "error type")
    method = _member(ReduceMethod, method, "method")
    setting = _statistics(statistics)
    project = session.project
    computed = results.compute_results(
        project.batch,
        plot_conditions=plot_conditions,
        error_type=error_type,
        method=method,
        statistics=setting,
        outdated=_outdated_readings(session, project),
    )
    return ComputedView(project, computed)


def _outdated_readings(session: ProjectSession, project: Project) -> dict[str, str]:
    """Why each image holding bands was measured on a reading of its stored file
    that this version no longer makes, by image id
    (:meth:`~proteia.core.session.ProjectSession.outdated_reading`, from the
    header, once per image)."""
    measured = {protein.image_id for protein in project.batch.proteins if protein.bands}
    return {
        image.id: why
        for image in project.batch.iter_images()
        if image.id in measured and (why := session.outdated_reading(image)) is not None
    }


def _refuse_outdated_nets(session: ProjectSession, project: Project) -> None:
    """Refuse an export of nets measured on a reading no longer made
    (``UNREADABLE_IMAGE``, with those images; :func:`_outdated_readings`): an
    export never carries numbers the results warn are not this reading's. The
    message is each image's why, which says what to do."""
    outdated = _outdated_readings(session, project)
    if outdated:
        raise OperationError(
            ErrorCode.UNREADABLE_IMAGE, "; and ".join(outdated.values()), ids=tuple(outdated)
        )


def compute(
    session: ProjectSession,
    *,
    plot_conditions: Collection[str] | None = None,
    error_type: ErrorType | str = ErrorType.SD,
    method: ReduceMethod | str = ReduceMethod.MEAN,
    statistics: StatisticsSetting | Mapping[str, str] | None = None,
) -> Results:
    """Every result of the committed project: :func:`compute_view`'s results."""
    return compute_view(
        session,
        plot_conditions=plot_conditions,
        error_type=error_type,
        method=method,
        statistics=statistics,
    ).results


@_locked
def export_lane_table(session: ProjectSession) -> Path:
    """Write the raw per-lane table to ``exports/lane-table.csv`` with its
    reproducibility record, ``exports/lane-table.record.json``
    (:func:`~proteia.core.record.build_record`); return the table's path.

    The same stored-index nets the results table shows, each followed by its
    clipping flags, in UTF-8 with a BOM. Each protein's columns carry its name,
    numbered as the bundle's are where a column before them has it
    (:func:`~proteia.core.export.lane_columns`; this table has no series
    columns to give way to). An image file that is missing or changed
    since import is refused (``IMAGE_FILE_CHANGED``, with those images): a record
    never vouches for pixels that are no longer on disk; so are nets measured on
    a reading this version no longer makes (``UNREADABLE_IMAGE``:
    :func:`_refuse_outdated_nets`). Both files are built
    before either is written, and each is replaced atomically, the table first;
    ``OSError`` propagates (with Excel holding the table, both old files stay; a
    first export whose record fails removes its table).
    Not a state change: no log entry, no autosave.
    """
    project = session.project  # one snapshot for both files
    batch = project.batch
    if not batch.lanes:
        raise OperationError(ErrorCode.NO_LANES, "declare the lanes before exporting them")
    bad = storage.verify_images(project, session.folder)
    if bad:
        raise OperationError(
            ErrorCode.IMAGE_FILE_CHANGED,
            f"image files changed or missing since import: {', '.join(bad)};"
            " restore them before exporting",
            ids=bad,
        )
    _refuse_outdated_nets(session, project)
    conditions, samples, included = spine_axes(batch.lanes)
    nets, clipped = results.lane_nets(batch), results.lane_clipped(batch)
    names = export.lane_columns([(p.id, p.name) for p in batch.proteins]).proteins
    table = lane_table_bytes(
        conditions,
        samples,
        included,
        [(names[p.id], nets[p.id]) for p in batch.proteins],
        clipped={names[p.id]: clipped[p.id] for p in batch.proteins},
    )
    doc = record.build_record(
        project, exported_at=session.timestamp(), files={LANE_TABLE_FILE: table}
    )
    data = record.record_bytes(doc)

    exports = session.folder / storage.EXPORTS_DIR
    exports.mkdir(parents=True, exist_ok=True)
    path, record_path = exports / LANE_TABLE_FILE, exports / LANE_TABLE_RECORD_FILE
    had_record = record_path.is_file()
    storage.write_atomic(path, table)
    try:
        storage.write_atomic(record_path, data)
    except OSError:
        # An old record names other table bytes (detectable); with none, remove
        # the table rather than leave numbers that no record describes.
        if not had_record:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        raise
    _log.info(
        "in %r: exported the lane table to %s/%s",
        session.folder.name,
        storage.EXPORTS_DIR,
        LANE_TABLE_FILE,
    )
    return path


@dataclass(frozen=True)
class ExportBundle:
    """What :func:`export_bundle` wrote."""

    folder: Path  # the new folder, exports/<name> in the project folder
    files: tuple[str, ...]  # the files in it, by name, in the order written: the record last


def _chart_formats(formats: object) -> tuple[ChartFormat, ...]:
    """The chart formats ``formats`` names, each once, in the order given."""
    if isinstance(formats, (str, bytes)) or not isinstance(formats, Iterable):
        raise _invalid(f"chart formats must be a list of formats, not {formats!r}")
    return tuple(dict.fromkeys(_member(ChartFormat, f, "a chart format") for f in formats))


def _new_folder(parent: Path, name: str) -> Path:
    """A folder made now in ``parent``: ``name``, or ``name (2)``, ``name (3)``
    and so on, the first that no file or folder has (in any case), so nothing is
    ever written into a folder that was there before."""
    for number in range(1, _FOLDER_ATTEMPTS + 1):
        folder = parent / (name if number == 1 else f"{name} ({number})")
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise FileExistsError(f"no free folder name for {name!r} in {parent}")


@_locked
def export_bundle(
    session: ProjectSession,
    *,
    formats: Iterable[ChartFormat | str] = DEFAULT_CHART_FORMATS,
    plot_conditions: Collection[str] | None = None,
    error_type: ErrorType | str = ErrorType.SD,
    method: ReduceMethod | str = ReduceMethod.MEAN,
    statistics: StatisticsSetting | Mapping[str, str] | None = None,
    statement_in_charts: bool = False,
) -> ExportBundle:
    """Write the results into a new folder of their own (#53): each result set's
    lane table and charts, a README and the reproducibility record, so files
    that belong together stay together.

    The folder is ``exports/<local date and time>`` in the project folder
    (:func:`~proteia.core.export.bundle_folder_name`), numbered ``(2)``,
    ``(3)`` and on when that name is taken: every export makes a new folder,
    and nothing that was there is overwritten. (A plain text sort puts ``(10)``
    before ``(2)``; a file manager's sort by number does not.) The file names
    are fitted to the room the folder's path leaves
    (:func:`~proteia.core.storage.name_fits`, with the longest folder name this
    export could take), so a long project path cuts the protein names and
    labels in them further. It holds, in the order written
    (:func:`~proteia.core.export.bundle_files`): each set's lane table, each
    set's charts in each of ``formats`` (drawn by the renderer the screen uses;
    a chart names its set in its subtitle and its file name, and a key under its
    axis says what its marks are; with ``statement_in_charts`` its legend is
    drawn under it too), ``README.txt`` (which gives each chart's legend as its
    caption), and ``export.record.json``
    (:func:`~proteia.core.record.build_record`), which lists every other file
    with its SHA-256, and the compute settings. When lanes the lane table
    excludes hold values, the all-lanes set is written too, in files of its own
    (#71). ``formats`` lists :class:`~proteia.core.export.ChartFormat` values
    (``"svg"``, ``"png"``, ``"pdf"``), each written once, in the order given;
    an empty list writes no chart. The default,
    :data:`~proteia.core.export.DEFAULT_CHART_FORMATS`, is provisional.
    ``plot_conditions``, ``error_type``, ``method`` and ``statistics`` are
    :func:`compute_view`'s, so the export shows what the screen does.

    Refused as :func:`export_lane_table` is, before anything is written: no
    lanes (``NO_LANES``), an image file missing or changed since import
    (``IMAGE_FILE_CHANGED``), or nets measured on a reading this version no
    longer makes (``UNREADABLE_IMAGE``); an unknown chart format, error type or method
    or statistics setting (``INVALID_INPUT``); and a project folder whose path
    leaves the file names less than :data:`~proteia.core.export.MIN_NAME_ROOM` characters
    (``PATH_TOO_LONG``: Windows takes no path over 259 characters while long
    paths are off, its default). Every file is built before the folder is made, and
    each is written atomically; a failure while writing (``OSError``, which
    propagates) removes the folder with what was written into it.
    Not a state change: no log entry, no autosave.
    """
    chosen = _chart_formats(formats)
    error_type = _member(ErrorType, error_type, "error type")
    method = _member(ReduceMethod, method, "method")
    setting = _statistics(statistics)
    project = session.project  # one snapshot for every file
    batch = project.batch
    if not batch.lanes:
        raise OperationError(ErrorCode.NO_LANES, "declare the lanes before exporting them")
    bad = storage.verify_images(project, session.folder)
    if bad:
        raise OperationError(
            ErrorCode.IMAGE_FILE_CHANGED,
            f"image files changed or missing since import: {', '.join(bad)};"
            " restore them before exporting",
            ids=bad,
        )
    _refuse_outdated_nets(session, project)
    moment = session.clock()  # one moment: the folder's name and the record's time
    exported_at = format_timestamp(moment)
    exports = session.folder / storage.EXPORTS_DIR
    folder_name = export.bundle_folder_name(moment)
    # File names are fitted to the longest name the new folder can take.
    widest = exports / f"{folder_name} ({_FOLDER_ATTEMPTS})"
    if not storage.name_fits(widest, "x" * MIN_NAME_ROOM):
        raise OperationError(
            ErrorCode.PATH_TOO_LONG,
            "the project folder's path is too long for an export: its files' paths would"
            " pass the file system's limit (259 characters on Windows); move the project"
            " to a folder with a shorter path",
        )
    computed = results.compute_results(
        batch,
        plot_conditions=plot_conditions,
        error_type=error_type,
        method=method,
        statistics=setting,
    )
    files = export.bundle_files(
        computed,
        formats=chosen,
        exported_at=exported_at,
        name_fits=functools.partial(storage.name_fits, widest),
        statement_in_charts=statement_in_charts,
    )
    doc = record.build_record(project, exported_at=exported_at, files=files, results=computed)
    files[BUNDLE_RECORD_FILE] = record.record_bytes(doc)

    exports.mkdir(parents=True, exist_ok=True)
    folder = _new_folder(exports, folder_name)
    try:
        for name, data in files.items():
            storage.write_atomic(folder / name, data)
    except BaseException:
        # Never leave numbers behind that no record describes.
        shutil.rmtree(folder, ignore_errors=True)
        raise
    _log.info(
        "in %r: exported %d files to %s/%s",
        session.folder.name,
        len(files),
        storage.EXPORTS_DIR,
        folder.name,
    )
    return ExportBundle(folder, tuple(files))


@_locked
def save(session: ProjectSession) -> Path:
    """Save now (:meth:`ProjectSession.save`); raises on failure, unlike autosave."""
    return session.save()
