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
  :func:`export_bundle` and :func:`save`, which change no state. The params
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
fields and ``clipped`` flag, and :func:`_set_box` the one place a box moves (it
clears the position-derived ``apparent_mw``). The invariant: for every image,
every band's stored net, ``background_level``, ``background_mode``,
``background_spread`` and ``clipped`` equal
:func:`~proteia.core.quantify.band_backgrounds`,
:func:`~proteia.core.quantify.net_signal` and
:func:`~proteia.core.quantify.is_clipped` of the stored pixels, all the boxes
on that image (every protein's, each with its protein's box size), the
polarity, the bit depth and the project's background method. A band's ring
excludes every other box on its image, so any edit that adds, moves, resizes or
removes a box re-quantifies the whole image: :func:`place_box`,
:func:`move_box`, :func:`remove_box`, :func:`set_box_size`,
:func:`remove_protein`, :func:`clear_boxes` and :func:`detect_row_boxes`; a
polarity change re-quantifies its image and :func:`requantify` every image.
Removing an image removes its bands and changes no other image. Undo and redo
restore a committed state whole, which met the invariant, and recompute
nothing. A project quantified before #83 keeps the legacy method
(``global_median``) until :func:`requantify`: each band's level is its image's
median, its mode ``global_median``, its spread 0, and its net floors each
pixel at 0, so edits of a legacy project keep the legacy invariant.

A not-detected record (:class:`~proteia.core.model.UndetectedBand`) is a
detector's measurement that cannot be redone from the model alone, so an edit
that invalidates one drops it, in the same change, and logs it in full: a box
placed or moved into its lane replaces it (``replaced_undetected``), and a
polarity change or a lane table that cuts its lane drops it
(``dropped_undetected``). An MW-guided record searched a slot placed from the
protein's expected MW and its membrane's calibration, so a change to the
expected MW drops that protein's MW-guided records, and a change to the
calibration (points removed with their image) drops those of every protein on
the membrane (``dropped_undetected``). A removed protein or image takes its
proteins' records along, and the log lists them whole (``removed_undetected``),
since they have no ids; clearing a protein's boxes (:func:`clear_boxes`)
drops its records too. Removing a box never creates a record: the lane becomes
"not measured". Records are written by a row commit (:func:`detect_row_boxes`),
whose outcome in each lane replaces the record there
(``undetected_written``, ``dropped_undetected``).

Functions return ids or small frozen dataclasses, never model objects. Typed text
follows :mod:`proteia.core.names`; box placement follows :mod:`proteia.core.boxes`.
"""

from __future__ import annotations

import contextlib
import functools
import math
import shutil
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Concatenate, Final

import numpy as np
from pydantic import JsonValue, ValidationError

from proteia.core import boxes, export, record, results, rowdetect, storage
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
from proteia.core.imaging import clipping_depth, load_image
from proteia.core.model import (
    DETECTING_SOURCES,
    IMAGE_SUFFIXES,
    LEGACY_BACKGROUND_METHOD,
    LOCAL_BACKGROUND_METHOD,
    Band,
    Batch,
    Box,
    BoxSize,
    ImageKind,
    ImageRef,
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
    net_signal,
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
    "Cascade",
    "ClearedBoxes",
    "ComputedView",
    "ErrorCode",
    "ExportBundle",
    "Keep",
    "LaneInput",
    "LanesUpdate",
    "OperationError",
    "ProjectSession",
    "Restored",
    "RowPlacement",
    "add_protein",
    "clear_boxes",
    "compute",
    "compute_view",
    "detect_row_boxes",
    "edit_protein",
    "export_bundle",
    "export_lane_table",
    "import_image",
    "move_box",
    "new_project",
    "open_project",
    "place_box",
    "redo",
    "remove_box",
    "remove_image",
    "remove_protein",
    "remove_undetected",
    "requantify",
    "save",
    "set_box_lane",
    "set_box_size",
    "set_lanes",
    "set_polarity",
    "set_reference_condition",
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
    unfitted_membranes: tuple[str, ...]  # lost calibration points: fit and apparent MWs cleared


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
    new, result = _prepare(session, change)
    if new == session.project:  # a no-op: nothing committed, no entry, no hook
        return result
    session._commit(new, action=action, params=params(result), **commit_kw)
    return result


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
    placed from the expected MW and the membrane's calibration, so a change to
    either leaves the record about a slot nobody looked in."""
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
    """The one writer of a band's net, background fields and clipping flag:
    quantify every band on one image, of every protein, together (protein order,
    then band order; the results do not depend on it), since each band's ring
    leaves out every box on the image. ``array`` is the image's analysis array.

    Under the project's legacy method (``global_median``) each level is the
    image's median and each net floors every pixel at 0, as before #83;
    otherwise :func:`~proteia.core.quantify.band_backgrounds` measures each
    level (falling back to the image's median) and the net floors the box total
    (:data:`~proteia.core.quantify.RING_CLAMP`). An image without a limit the
    clipping check can trust leaves its bands unchecked (None).
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
    band.box = Box(x=rect[0], y=rect[1])
    band.apparent_mw = None  # position-derived (#58); stale once the box changes


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


def _fitting_size(size: object, image: ImageRef) -> BoxSize:
    if not isinstance(size, BoxSize):
        raise _invalid(f"box size must be a BoxSize, not {size!r}")
    if size.width > image.width or size.height > image.height:
        raise OperationError(
            ErrorCode.SIZE_OUT_OF_BOUNDS,
            f"box size {size.width}x{size.height} exceeds the"
            f" {image.width}x{image.height} image {image.id}",
        )
    return size


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
    recorded size, bit depth, background and warnings come from the stored copy.
    Without ``membrane_id`` the image starts a new membrane. ``kind`` and
    ``polarity`` are required (the model has no silent default).
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
                    import_warnings=loaded.warnings,
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
    except BaseException:
        # _commit can raise before committing (a bad clock or params) or after
        # (a hook bug): keep the file only if the committed project uses it.
        if all(image.id != image_id for image in session.project.batch.iter_images()):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        raise
    return image_id


@_locked
def remove_image(session: ProjectSession, image_id: str) -> Cascade:
    """Remove an image with the proteins, bands and not-detected records on it.

    Targets using a removed loading control, by name or as the batch's only
    one, are detached and reported, marker pairings to the image are cleared, and calibration
    points on it are dropped, which clears the membrane's fit, its bands'
    apparent MWs and its proteins' MW-guided records. A membrane left with no
    image is removed. The file is deleted once the removal can no longer be
    undone, at the next save or import after that (see
    :mod:`proteia.core.session`). The log lists the removed records
    (``removed_undetected``) and the MW-guided ones dropped
    (``dropped_undetected``) in full.
    """
    session.project.batch.find_image(image_id)

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
        dropped: list[JsonValue] = []
        calibration = membrane.calibration
        points = [point for point in calibration.points if point.image_id != image_id]
        if len(points) != len(calibration.points):
            calibration.points = points
            calibration.fit_quality = None  # the fit no longer matches its points (#58 refits)
            on_membrane = {image.id for image in membrane.images}
            for protein in batch.proteins:
                if protein.image_id in on_membrane:
                    for band in protein.bands:
                        band.apparent_mw = None
                    dropped.extend(_drop_mw_guided(protein))
            if membrane.images:
                unfitted.append(membrane.id)
        if not membrane.images:
            batch.membranes = [m for m in batch.membranes if m.id != membrane.id]
            removed.append(membrane.id)
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
    median does not depend on it).

    The not-detected records of every protein on the image are dropped and
    logged: their SNR was measured with the other signal direction, against the
    row's fitted background, so it cannot be recomputed here. A later detection
    run writes them again.
    """
    polarity = _member(Polarity, polarity, "polarity")
    batch = session.project.batch
    if batch.find_image(image_id).polarity is polarity:
        return
    array = session.pixels(image_id) if _bands_on(batch, image_id) else None

    def change(draft: Project) -> list[dict[str, JsonValue]]:
        draft.batch.find_image(image_id).polarity = polarity
        if array is not None:
            _quantify_image(draft, image_id, array)
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
) -> str:
    """Add a protein on a signal image, last in the protein order; return its id.

    The name is stored cleaned and must be unique ignoring case and look-alike
    characters. ``loading_control_ids`` (targets only) keeps its order, the series
    order. ``box_size`` defaults to :func:`~proteia.core.boxes.initial_box_size`;
    the first grown box replaces it. The image cannot be changed later. Adding a
    second loading control writes the first into the targets that used it
    without naming it, so their results do not change.
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
    controls = _loading_controls(batch, None, role, loading_control_ids)
    if box_size is None:
        size = boxes.initial_box_size(image.width, image.height)
    else:
        size = _fitting_size(box_size, image)

    def change(draft: Project) -> tuple[str, list[str]]:
        pinned = _pin_single_loading_control(draft.batch) if role is Role.LOADING_CONTROL else []
        protein_id = draft.new_id("prot")
        draft.batch.proteins.append(
            Protein(
                id=protein_id,
                name=name,
                role=role,
                image_id=image_id,
                loading_control_ids=controls,
                expected_mw=mw,
                box_size=size,
            )
        )
        return protein_id, pinned

    def params(result: tuple[str, list[str]]) -> _Params:
        protein_id, pinned = result
        return {
            "protein_id": protein_id,
            "name": name,
            "role": role.value,
            "image_id": image_id,
            "expected_mw": mw,
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
) -> None:
    """Edit a protein's fields; ``KEEP`` leaves one unchanged. No net changes.

    A target turned into a loading control loses its own loading controls; if it
    becomes the second one, the first is written into the targets that used it
    without naming it. A loading control that targets use, by name or as the
    batch's only one, cannot become a target (``LOADING_CONTROL_IN_USE``, with
    those targets): choose other loading controls for them first. A changed
    expected MW drops the protein's MW-guided not-detected records, whose slot
    came from the old one (``dropped_undetected``, logged in full).
    """
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    new_name = protein.name if name is KEEP else _protein_name(batch, name, protein_id=protein_id)
    new_role = protein.role if role is KEEP else _member(Role, role, "role")
    mw = protein.expected_mw if expected_mw is KEEP else _expected_mw(expected_mw)
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
        edited.loading_control_ids = controls
        dropped: list[JsonValue] = []
        if mw != protein.expected_mw:
            dropped.extend(_drop_mw_guided(edited))
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
    protein's shared size fitted to it (the first box sets the size, later ones
    only grow it, and the other boxes are re-centred).
    ``grow=False`` drops a box of the current size centred on the point, shifted
    inside the image. A protein's own boxes never overlap (``OVERLAP``, with
    the box in the way), and the new box, or a box of the protein grown to a
    size the click grows, may overlap another protein's box on the image by
    at most :data:`COVER_SHARE` of the smaller box's area (``OVERLAP``, naming
    the boxes it would overlap, :func:`_cover_refusal`): two proteins' boxes
    on one band would measure it twice. Every band on the image, of every
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
                rects, protein.box_size, grown, width=width, height=height
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
                f"at the box size {size.width}x{size.height} this band needs, boxes of"
                f" {protein.name!r} would overlap",
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
    never changes. Marks the band as manually edited.
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
        _quantify_image(draft, edited.image_id, array)

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
    as edited by the user. The same lane is a no-op. A not-detected record for
    the same band in the new lane is replaced (``replaced_undetected``); the lane
    the box leaves gets no record.
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
        return _drop_undetected(edited_protein, lane, edited.band_index)

    _apply(
        session,
        "set_box_lane",
        change,
        lambda replaced: {**params, "replaced_undetected": replaced},
    )


@_locked
def set_box_size(session: ProjectSession, protein_id: str, size: BoxSize) -> None:
    """Change a protein's shared box size: every box is re-sized around its centre
    (shifted inside the image), and every band on the image is re-quantified
    (the other proteins' rings leave out the resized boxes). A size that would
    make boxes overlap is refused (``SIZE_WOULD_OVERLAP``), and so is one that
    would make a box overlap another protein's by more than
    :data:`COVER_SHARE` of the smaller box's area (``OVERLAP``, naming those
    boxes, :func:`_cover_refusal`)."""
    batch = session.project.batch
    protein = batch.find_protein(protein_id)
    image = batch.find_image(protein.image_id)
    size = _fitting_size(size, image)
    if size == protein.box_size:
        return
    rects = [band.box.rect(protein.box_size) for band in protein.bands]
    resized = boxes.resize_all(rects, size, width=image.width, height=image.height)
    if resized is None:
        raise OperationError(
            ErrorCode.SIZE_WOULD_OVERLAP,
            f"box size {size.width}x{size.height} would make boxes of {protein.name!r} overlap",
        )
    covered = _covered(batch, protein, [r for r, o in zip(resized, rects, strict=True) if r != o])
    if covered:
        raise _cover_refusal(
            covered,
            f"at the box size {size.width}x{size.height}, boxes of {protein.name!r} would overlap",
        )
    array = session.pixels(image.id) if protein.bands else None

    def change(draft: Project) -> None:
        edited = draft.batch.find_protein(protein_id)
        edited.box_size = size
        for band, old, new in zip(edited.bands, rects, resized, strict=True):
            if new != old:
                _set_box(band, new)
        if array is not None:
            _quantify_image(draft, edited.image_id, array)

    _apply(
        session,
        "set_box_size",
        change,
        lambda _: {"protein_id": protein_id, "box_size": _size(size)},
    )


@_locked
def clear_boxes(session: ProjectSession, protein_id: str) -> ClearedBoxes:
    """Remove every box and every not-detected record of a protein, of every band
    index, in one change; undo is the way back.

    The protein's box size is kept (the width and height fields show it). A seed
    click or a row box on the protein, which then has no boxes, sets the size
    afresh from the band or bands found; a fixed box uses the kept size. A
    protein with neither boxes nor records is a no-op. The other proteins'
    bands on the image are re-quantified: their rings no longer leave out the
    boxes. The log entry lists the band ids with their lanes, and the records
    in full (``dropped_undetected``).
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
    (source ``click``, edited by hand or not) of a protein whose boxes are
    wider than the lane pitch (:func:`~proteia.core.project.lane_pitch` of
    each protein's): such a box is centred on what grew from the click, which
    may span several lanes' bands (touching bands), so its centre need not lie
    on its lane's column. A box dropped where the user clicked, moved there by
    hand or centred on a band a detector found shows its lane whatever its
    width.

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
        if p.box_size.width > pitch
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

    The shared size is the detector's. If a box of the protein survives (one
    kept, or of another band index), the size only grows, as a seed
    click grows it, even when every band found is in a kept lane and nothing is
    placed: the survivors are re-centred and the new boxes centred on the
    detected ones. That size making survivors overlap, or new boxes overlap
    each other, is refused (``SIZE_WOULD_OVERLAP``), and so is a new box over a
    survivor (``OVERLAP``, with the survivor), or a box it places (new or in
    place), or a survivor grown to the size, that overlaps another protein's
    box on the image by more than :data:`COVER_SHARE` of the smaller box's
    area (``OVERLAP``, naming those boxes, :func:`_cover_refusal`: a row
    dragged over another protein's row, or a loading control's row over its
    target's). If none survives, this detection
    sets the size afresh. A box may then extend beyond
    the row box, never beyond the image. Every band on the image, of every
    protein, is then re-quantified with the project's background method; the
    detector's own local background only finds the bands.

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
    notes, whether the lanes were read right to left, and its settings
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
    try:
        # The settings the export record reports (record.settings): the defaults.
        found = rowdetect.detect_row(
            array,
            given,
            n,
            background=image.background,
            dark_on_light=image.polarity.dark_on_light,
            right_to_left=right_to_left,
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
    try:
        # The detector's size; while a box survives it only grows the size, even
        # if every band found is in a kept lane and nothing is placed.
        size, resized, new_rects = boxes.grow_to_fit_all(
            old,
            protein.box_size,
            [lane.rect for lane in placed],
            need=found.size,
            width=width,
            height=height,
        )
    except boxes.BoxRuleError as exc:
        grown = f"{exc.size.width}x{exc.size.height}"
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
        else:  # the new boxes do
            message = (
                f"the row's boxes would overlap each other at the box size {grown}"
                f" that the kept boxes of {protein.name!r} need"
            )
        raise OperationError(ErrorCode.SIZE_WOULD_OVERLAP, message) from exc
    rects = {lane.lane: rect for lane, rect in zip(placed, new_rects, strict=True)}
    covered = _covered(batch, protein, rects.values())
    if covered:
        raise _cover_refusal(covered, "the row's boxes would overlap")
    covered = _covered(batch, protein, [r for r, o in zip(resized, old, strict=True) if r != o])
    if covered:
        raise _cover_refusal(
            covered,
            f"at the box size {size.width}x{size.height} the row needs, the boxes of"
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
    warnings = [flag for flag in found.flags if flag in rowdetect.WARNING_FLAGS]

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
            edited.bands.append(band)
        # The boxes placed, moved and removed change every ring on the image.
        _quantify_image(draft, edited.image_id, array)
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
            "notes": list(found.notes),
            "right_to_left": right_to_left,
            "settings": rowdetect.settings(),
        }

    band_ids, _, _, _ = _apply(session, "detect_row_boxes", change, params)
    # The other proteins' nets on the image the row changed (their rings leave
    # out its boxes); the same drag again, a no-op, changes none.
    after = {
        band.id: band.net
        for other in session.project.batch.proteins
        if other.image_id == image.id and other.id != protein_id
        for band in other.bands
    }
    remeasured = tuple(
        (band.id, band.net, after[band.id])
        for other in batch.proteins
        if other.image_id == image.id and other.id != protein_id
        for band in other.bands
        if after[band.id] != band.net
    )
    shares = [(abs(new - old) / old, band_id) for band_id, old, new in remeasured if old > 0]
    largest = max(shares, key=lambda share: share[0], default=None)
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
        notes=found.notes,
        right_to_left=right_to_left,
        remeasured=remeasured,
        largest_change=None if largest is None else (largest[1], largest[0]),
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


@_locked
def requantify(session: ProjectSession) -> tuple[str, ...]:
    """Switch the project to the local background (``ring_median_v1``) and
    re-quantify every band with it, in one change; return the ids of the images
    re-quantified (those with bands), in membrane then image order.

    A project quantified before #83 keeps the legacy method (``global_median``)
    until this runs, so its stored nets never change unasked. A project already
    on ``ring_median_v1`` is a no-op. The pixels are read image by image inside
    the change, and those not cached already are not kept, so memory holds one
    image beyond the cache; a missing or changed image file still refuses the
    whole change (``IMAGE_FILE_CHANGED``), which then changes nothing. The log
    entry names the method left (``from``), the method taken (``to``) and the
    images re-quantified (``images``).
    """
    project = session.project
    old = project.background_method
    if old == LOCAL_BACKGROUND_METHOD:
        return ()
    batch = project.batch
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
    (:func:`~proteia.core.results.compute_results`).

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
    )
    return ComputedView(project, computed)


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
    never vouches for pixels that are no longer on disk. Both files are built
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
    lanes (``NO_LANES``), or an image file missing or changed since import
    (``IMAGE_FILE_CHANGED``); an unknown chart format, error type or method
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
    return ExportBundle(folder, tuple(files))


@_locked
def save(session: ProjectSession) -> Path:
    """Save now (:meth:`ProjectSession.save`); raises on failure, unlike autosave."""
    return session.save()
