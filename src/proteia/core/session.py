# SPDX-License-Identifier: Apache-2.0
"""One open project folder: the committed project, its pixels, and autosave.

GUI-independent. A :class:`ProjectSession` is what :mod:`proteia.core.operations`
works on; create one with :func:`new_project` or :func:`open_project`.

* The committed project is replaced on every change, never edited in place, so a
  caller holding :attr:`ProjectSession.project` keeps a consistent snapshot.
* Every change is committed through one point, which then runs the autosave hook
  synchronously under the session lock, so saves happen in commit order and an
  older state never overwrites a newer one. A failed save keeps the edit in
  memory (``dirty`` stays True, ``save_error`` holds the exception) and the
  previous ``project.json`` on disk (the write is atomic); the next change, or an
  explicit :meth:`ProjectSession.save`, tries again.
* A save goes only over the ``project.json`` the session last read or wrote
  (its *base*, by the SHA-256 of its bytes), or one holding the same document
  in other bytes (a BOM, CRLF, other key order: #229). One changed outside
  Proteia since (a synced or restored copy) is first kept, as it was, in a
  conflict copy beside it (:func:`~proteia.core.storage.keep_conflict_copy`);
  then the session's version is saved as usual (#101), and
  :attr:`ProjectSession.outside_change` says so for the page. A removed
  ``project.json`` is written again (and said); a removed project folder is
  never made again (:class:`~proteia.core.storage.ProjectRemovedError`).
* Every committed change appends one :class:`~proteia.core.model.LogEntry` at
  that single commit point, in the same project object, so one save writes the
  change and its entry together; refusals and no-ops append nothing. Times come
  from the session's injectable clock (UTC; the offline system clock by default).
  The one entry made elsewhere is a ``migrate`` entry, appended as a file of an
  older schema loads, once the file is kept in a backup beside it (#140:
  :func:`~proteia.core.storage.keep_backup`, which the entry names);
  :func:`open_project` then autosaves it like a change.
  Every entry is also logged (Python's :mod:`logging`, at INFO: the session log
  of #137), with its params as the entry holds them and the project named by
  its folder's name, never its path; and so are opening, reading again and
  closing a project.
* Pixels are read lazily and cached as read-only arrays, after checking the
  stored file's SHA-256 and the array's shape against the image record. An
  image's array is of the channel the committed project measures it on
  (#206): a commit that changes the channel drops the cached array, unless the
  change gives the new one.
* The session keeps its undo history: the content of each state it committed,
  up to :data:`UNDO_LIMIT` steps back, as the exact bytes the content hash
  covers (:func:`~proteia.core.storage.content_bytes`). The history moves at the
  single commit point, before the autosave hook runs. Undo and redo restore a
  state whole, never replaying a change, and commit it as a logged change of
  their own (``undo``, ``redo``), so the log only grows. The history is per
  session: :func:`open_project` starts with none, and it is lost at
  :meth:`ProjectSession.close`, a project switch, a restart, or a
  :meth:`ProjectSession.reload` of a ``project.json`` changed outside Proteia;
  the log, which keeps params and hashes but not states, cannot rebuild it.
* Files in ``images/`` that no project references (an image removed, an import
  that was never saved, a temp file left by a crash) are deleted after a
  successful save and before an import, but only if the ``project.json`` on
  disk does not reference them, no state in the undo history does, no backup
  of an older ``project.json`` names them with an id below ``next_id``
  (:func:`~proteia.core.storage.backup_references`; while one cannot be read,
  it is taken to name every such file), no other version of the project in
  its folder (:func:`~proteia.core.storage.is_version`: a conflict copy, any
  other ``.json`` file there), nor a ``project.json`` changed outside
  Proteia, names them, whatever their id (while one cannot be read, or the
  file is missing, no image file is deleted), and they carry Proteia's ``img-N``
  name: a saved project never points at a missing file, an undo never needs
  one, a restored backup finds its images even if a migration lost one (they
  hold ids below ``next_id``, which a migration keeps and the project only
  raises, so no backup keeps the file of the id an import takes next), every
  version keeps its images (an import skips the ids whose files are kept:
  :meth:`ProjectSession.image_number`), and a stray user file is never
  deleted. So a removed or undone image's file stays while the history can
  bring it back, and goes at the first save or import
  after no state references it: once the undo limit drops the last such state,
  a new change clears the redo that held it, or the session is closed or
  reloads; or else at the next session's first save. Only image files the
  session knows are deleted: those the ``project.json`` it read names, and
  those whose import the project's log records, with the bytes that import
  stored (every image a session holds is one or the other). One it never
  knew stays (an image another computer imported, synced before
  the ``project.json`` that names it; or one an earlier session stored but
  never saved, which only takes disk space), and an import skips its id.

Refusals raise :class:`OperationError`, whose :class:`ErrorCode` is stable for
clients (e.g. to map onto HTTP statuses).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import re
import threading
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Final, Literal

import numpy as np
from pydantic import JsonValue

import proteia
from proteia.core import storage
from proteia.core.imaging import (
    FileColours,
    UnsupportedColourSpaceError,
    converts_cmyk,
    file_colours,
    load_image,
    named_refusal,
    of_the_system,
    read_colours,
    refusal_cause,
)
from proteia.core.model import (
    IMAGE_SUFFIXES,
    LEGACY_BACKGROUND_METHOD,
    Channel,
    ImageRef,
    LogEntry,
    Project,
    format_timestamp,
)

_log = logging.getLogger(__name__)

# The undoable steps a session keeps. A count, not a byte budget: a state is its
# compressed content, about 24 KiB for a batch of 960 bands.
UNDO_LIMIT: Final = 100

# The causes of refusing to read a stored file whose logging a session
# remembers (#246), so that each is logged once however often the file is
# read; past this, the one met least recently is forgotten, and logged again
# if met again.
_LOGGED_CAUSES: Final = 256
# A cause of refusing to read a stored file, as a session keeps it: the
# image's id and SHA-256, the refusal's own type, its error's type and errno
# (or None), and whether that error is the system's.
type _LoggedCause = tuple[str, str, type[BaseException], type[BaseException], int | None, bool]

# Proteia's image names (img-N.<suffix>) and the temp files storage writes while
# storing or saving (.img-N.<suffix>.<random>.part/.tmp): the only orphans cleanup
# may delete. The whole name must match, so a user's img-1.tif.bak survives; case
# is ignored because Windows file names ignore it.
_SUFFIXES = "|".join(re.escape(suffix.removeprefix(".")) for suffix in IMAGE_SUFFIXES)
_ORPHAN_NAME = re.compile(
    rf"^\.?img-[1-9][0-9]{{0,8}}\.(?:{_SUFFIXES})(?:\.[a-z0-9_]+\.(?:part|tmp))?$",
    re.IGNORECASE,
)
# Of those, a stored image's file (no temp file), with its id's number: the only
# orphans a backup of an older project.json may keep.
_IMAGE_FILE = re.compile(rf"^img-([1-9][0-9]{{0,8}})\.(?:{_SUFFIXES})$", re.IGNORECASE)


class ErrorCode(StrEnum):
    """Why an operation was refused. The values are stable for clients."""

    INVALID_INPUT = "invalid_input"  # wrong type or range; also a model ValidationError
    BLANK_TEXT = "blank_text"
    CONTROL_CHARACTER = "control_character"
    DUPLICATE_NAME = "duplicate_name"
    RESERVED_NAME = "reserved_name"
    MARKER_IMAGE = "marker_image"  # a protein on a visible-light marker image
    LOADING_CONTROL_IN_USE = "loading_control_in_use"
    NO_LANES = "no_lanes"
    LANE_OUT_OF_RANGE = "lane_out_of_range"
    LANE_OCCUPIED = "lane_occupied"
    LANE_REQUIRED = "lane_required"  # position cannot propose the lane yet
    LANES_IN_USE = "lanes_in_use"
    UNKNOWN_CONDITION = "unknown_condition"
    OUT_OF_IMAGE = "out_of_image"
    NO_BAND_FOUND = "no_band_found"
    ROW_TOO_SMALL = "row_too_small"  # a row box too narrow for its lanes or too low to smooth
    # The bands in a row box, or the lanes already placed on its image, do not
    # show which lane is which.
    ROW_LANES_UNCLEAR = "row_lanes_unclear"
    # The bands in a row box do not lie on one row: it covers two rows, or a
    # lane's band lies above or below the others.
    ROW_OFF_LINE = "row_off_line"
    # A box would overlap another box of its protein, or cover most of another
    # protein's box (``detail``: the covered boxes).
    OVERLAP = "overlap"
    SIZE_WOULD_OVERLAP = "size_would_overlap"
    SIZE_OUT_OF_BOUNDS = "size_out_of_bounds"
    UNSUPPORTED_IMAGE_TYPE = "unsupported_image_type"
    IMAGE_TOO_LARGE = "image_too_large"
    INVALID_IMAGE = "invalid_image"  # an empty stream, or a name store_image refuses
    UNREADABLE_IMAGE = "unreadable_image"  # the loader cannot decode it
    # An import of a file of printing inks: CMYK, or a separated TIFF of other
    # inks (#207); one imported before is still read, converted. ``detail``:
    # inks ("cmyk" or "other"), several_images, uncompressed_copy.
    CMYK_IMAGE = "cmyk_image"
    # An import of a file whose pixels have no fixed range: floating point,
    # complex, 1-bit, or integers other than unsigned 8 or 16 bits (#224); one
    # imported before is still read, with its unknown_bit_depth warning.
    UNSUPPORTED_PIXEL_TYPE = "unsupported_pixel_type"
    IMAGE_FILE_CHANGED = "image_file_changed"  # stored file missing, other bytes, other shape
    # The project folder's path leaves an export's files too little room within
    # the file system's path limit (Windows' 259 characters, long paths off).
    PATH_TOO_LONG = "path_too_long"
    FOLDER_NOT_EMPTY = "folder_not_empty"
    # A file appeared at the image id an import took, between its check and the
    # store (an import skips the ids of files it finds: ProjectSession.image_number).
    LEFTOVER_FILE = "leftover_file"
    # The open project's folder was removed or moved outside Proteia
    # (storage.project_removed): an import stores nothing then (#101).
    PROJECT_REMOVED = "project_removed"
    NOTHING_TO_UNDO = "nothing_to_undo"
    NOTHING_TO_REDO = "nothing_to_redo"
    # Molecular-weight calibration (#58). A calibration point would be out of
    # order on its ladder (or share another's position there).
    CALIBRATION_ORDER = "calibration_order"
    DUPLICATE_MW = "duplicate_mw"  # that ladder already has a point at this MW
    # The right ladder would not lie right of the left one, or a strip edge (or
    # a point with no x) would share a register group with a right ladder.
    LADDER_SIDES = "ladder_sides"
    MARKER_SIZE_MISMATCH = "marker_size_mismatch"  # a marker and its image differ in size
    # Placing a row by its expected MW (#58, D11, D12): the image has no curve
    # (the reason in words); the protein has no expected MW (every band's, for
    # a protein expecting several); an expected MW outside the calibrated range
    # at the lanes; the lanes' span is not known yet (drag across them once).
    NO_CALIBRATION = "no_calibration"
    MW_REQUIRED = "mw_required"
    MW_OUTSIDE_CALIBRATION = "mw_outside_calibration"
    LANE_SPAN_REQUIRED = "lane_span_required"


class OperationError(ValueError):
    """An operation refused its input. Nothing changed: the project (the same
    object), ``next_id``, the undo history and the pixel cache are as they were, no
    file was added to ``images/``, and the autosave hook did not run. (An import
    removes orphan files, which no project or undo state references, before it can
    be refused.)

    ``detail``, when a refusal has one, is JSON-plain data about why, for a
    client that words or shows it its own way (a row box's: what the detector
    saw); None otherwise."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        ids: Sequence[str] = (),
        detail: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.ids: tuple[str, ...] = tuple(ids)  # the objects involved, e.g. an overlapped band
        self.detail: dict[str, JsonValue] | None = None if detail is None else dict(detail)


AutosaveHook = Callable[["ProjectSession"], None]
# Returns an aware datetime; tests inject a fake one.
Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """The default clock: the system clock, in UTC."""
    return datetime.now(UTC)


def _referenced_files(project: Project) -> frozenset[str]:
    return frozenset(image.file for image in project.batch.iter_images())


def _entry(
    *, seq: int, action: str, params: Mapping[str, JsonValue], content_hash: str, clock: Clock
) -> LogEntry:
    """The log entry of a change that leaves content hashing to ``content_hash``.
    Raises ``ValidationError`` for params that are not plain JSON, ``ValueError``
    for a naive clock."""
    return LogEntry(
        seq=seq,
        time=format_timestamp(clock()),
        action=action,
        version=proteia.__version__,  # read at call time: what is running now
        params=dict(params),
        content_hash=content_hash,
    )


def _log_entry(folder: Path, entry: LogEntry) -> None:
    """Log a committed change: its seq, action and params, and the project's
    folder by name."""
    if _log.isEnabledFor(logging.INFO):
        params = json.dumps(entry.params, ensure_ascii=False)
        _log.info("committed #%d %s in %r: %s", entry.seq, entry.action, folder.name, params)


def _log_migrated(folder: Path, entry: LogEntry) -> None:
    """Log a ``migrate`` entry: the backup it names first, then the entry."""
    params = entry.params
    _log.info(
        "in %r: kept its schema-%s %s as %r before migrating it",
        folder.name,
        params.get("from_schema"),
        storage.PROJECT_FILE,
        params.get("backup"),
    )
    _log_entry(folder, entry)


def _log_opened(folder: Path, project: Project, *, again: bool = False) -> None:
    """Log that ``project`` was read from ``folder`` (``again``: read again, as
    changed outside Proteia), and the legacy background it still uses."""
    batch = project.batch
    _log.info(
        "%s: %d images, %d proteins, %d boxes, %d log entries",
        f"read {folder.name!r} again, as its project.json was changed outside Proteia"
        if again
        else f"opened {folder.name!r}",
        sum(1 for _ in batch.iter_images()),
        len(batch.proteins),
        sum(len(batch.boxes_of(protein).bands) for protein in batch.proteins),
        len(project.log),
    )
    if project.background_method == LEGACY_BACKGROUND_METHOD and any(
        batch.boxes_of(protein).bands for protein in batch.proteins
    ):
        _log.info(
            "in %r: the nets are measured with the legacy background (%s), each image's"
            " median, until the project is requantified",
            folder.name,
            LEGACY_BACKGROUND_METHOD,
        )


@dataclass(frozen=True)
class HistoryStep:
    """A change that undo would take back or redo would make again: its log
    entry (the original change's, not an undo's or a redo's)."""

    seq: int
    action: str


@dataclass(frozen=True)
class _State:
    """One committed state in a session's undo history."""

    seq: int | None  # the entry of the change that left it; None: content no entry recorded
    action: str  # that entry's action ("" with no entry)
    hash: str  # its content hash
    content: bytes  # zlib of storage.content_bytes: its SHA-256 is ``hash``
    images: tuple[tuple[str, str], ...]  # (image id, file) of every image it references


def _state(project: Project, *, seq: int | None, action: str, data: bytes, digest: str) -> _State:
    return _State(
        seq=seq,
        action=action,
        hash=digest,
        content=zlib.compress(data, 1),  # the fastest level: 179 KiB to 24 KiB at 960 bands
        images=tuple((image.id, image.file) for image in project.batch.iter_images()),
    )


class _Move(Enum):
    """How a commit moves the undo history."""

    PUSH = "push"  # an ordinary change: clears redo
    UNDO = "undo"
    REDO = "redo"


def _record_keys(project: Project) -> list[tuple[str, int, int]]:
    """(protein id, lane index, band index) of every not-detected record."""
    batch = project.batch
    return [
        (protein.id, record.lane_index, record.band_index)
        for protein in batch.proteins
        for record in batch.boxes_of(protein).undetected
    ]


def _only_in[T](items: Sequence[T], others: Sequence[T]) -> list[T]:
    """The items of ``items`` not in ``others``, in their order."""
    exclude = set(others)
    return [item for item in items if item not in exclude]


def _outdated_reading(image: ImageRef, header: bool | str) -> str | None:
    """Why ``image``'s stored numbers come from a reading of its stored file that
    this version no longer makes, or None; ``header`` is what its file's header
    says: whether reading it converts CMYK
    (:func:`~proteia.core.imaging.converts_cmyk`), or what reading refuses in it
    (:attr:`~proteia.core.imaging.UnsupportedColourSpaceError.problem`)."""
    name = image.original_name
    if isinstance(header, str):
        return (
            f"{name!r}: {header}; it was imported before Proteia refused such files, so"
            " its stored background and nets come from a reading Proteia no longer makes:"
            " remove the image, save the file as a gray or RGB TIFF and import it again"
        )
    if header and not any(w.code == "cmyk_converted" for w in image.import_warnings):
        return (
            f"{name!r} is a CMYK file imported before Proteia converted CMYK colours: its"
            " stored background and nets were measured with its cyan, magenta and yellow"
            " read as red, green and blue; remove the image, save the file as RGB or"
            " grayscale and import it again"  # an import refuses CMYK since #207
        )
    return None


@dataclass(frozen=True)
class _Base:
    """The ``project.json`` a session last read or wrote (#101): the SHA-256 of
    its bytes, and how many entries of the committed log it holds. For the
    #229 comparison, the SHA-256 of its canonical form (:func:`storage.project_to_json`)
    once known: the bytes themselves after a write; after a read, computed
    from ``parsed``, the project those very bytes hold, when a save first
    meets other bytes; never for a file of an older schema (the save right
    after its migration goes over the same bytes)."""

    sha256: str
    log: int
    canonical: str | None = None
    parsed: Project | None = None


def _written(data: bytes, project: Project) -> _Base:
    """The base a write of ``project`` as ``data`` leaves."""
    digest = hashlib.sha256(data).hexdigest()
    return _Base(sha256=digest, log=len(project.log), canonical=digest)


def _read(data: bytes, project: Project, *, migrated: bool) -> _Base:
    """The base a read of ``data`` as ``project`` leaves (``migrated``: from an
    older schema, so ``project`` holds a ``migrate`` entry the file does not)."""
    digest = hashlib.sha256(data).hexdigest()
    if migrated:
        return _Base(sha256=digest, log=len(project.log) - 1)
    return _Base(sha256=digest, log=len(project.log), parsed=project)


class Lineage(Enum):
    """How a ``project.json`` found changed outside Proteia relates to the
    session's base (the file it last read or wrote) by their logs (#101):
    :func:`divergence`."""

    EXTENDS_BASE = "extends_base"  # its log is the base's, with entries after it
    SAME_LOG = "same_log"  # its log is the base's: changed outside the log (by hand)
    BEHIND_BASE = "behind_base"  # its log ends inside the base's: an older copy
    FORK = "fork"  # it shares the start of the base's log, then goes its own way
    UNRELATED = "unrelated"  # no first entry in common


@dataclass(frozen=True)
class Divergence:
    """Where a version changed outside Proteia and the session's version part,
    computed once by :func:`divergence` for the notice and :func:`resolve`:
    the log entries they share from the start (``common``), those only the
    other one has (``theirs``), those only the session's has (``ours``), and
    the time of the last entry only the other one has (``their_last``, in the
    log's form; None when it has none)."""

    lineage: Lineage
    common: int
    theirs: int
    ours: int
    their_last: str | None


def divergence(ours: Sequence[LogEntry], base: int, theirs: Sequence[LogEntry]) -> Divergence:
    """How the log ``theirs`` of a version changed outside Proteia relates to
    the session's log ``ours``, whose first ``base`` entries are the base's."""
    common = 0
    for mine, other in zip(ours, theirs, strict=False):
        if mine != other:
            break
        common += 1
    if common >= base:
        lineage = Lineage.EXTENDS_BASE if len(theirs) > base else Lineage.SAME_LOG
    elif common == len(theirs):
        lineage = Lineage.BEHIND_BASE
    elif common == 0:
        lineage = Lineage.UNRELATED
    else:
        lineage = Lineage.FORK
    return Divergence(
        lineage=lineage,
        common=common,
        theirs=len(theirs) - common,
        ours=len(ours) - common,
        their_last=theirs[-1].time if len(theirs) > common else None,
    )


@dataclass(frozen=True)
class Resolution:
    """What a save does with a ``project.json`` changed outside Proteia (#101):
    which version goes into the conflict copy (``copy``: the file as found,
    ``"outside"``, or the session's version, ``"session"``), and what the
    session does then (``then``: write its version over the file, ``"save"``,
    or take the file on disk as the project, ``"switch"``)."""

    copy: Literal["outside", "session"]
    then: Literal["save", "switch"]


# Keep the file as found in a conflict copy beside it, then save over it: the
# maintainer's 2026-11-01 choice, and the only resolution a save carries out.
SET_ASIDE_OUTSIDE: Final = Resolution(copy="outside", then="save")


def resolve(divergence: Divergence | None) -> Resolution:
    """What a save does when ``project.json`` changed outside Proteia: the one
    place that decides it. ``divergence`` is None for a file that does not
    read as a project. Today always :data:`SET_ASIDE_OUTSIDE`. A later rule is
    decided here: at a fork (``Lineage.FORK``), the session's version into
    the copy and a switch to the file (``Resolution("session", "switch")``,
    :meth:`ProjectSession._read_again` reads it); for a version that only
    extends the base (``Lineage.EXTENDS_BASE``), a switch and the change made
    again on it."""
    return SET_ASIDE_OUTSIDE


def _local(time: str | datetime) -> tuple[str, str]:
    """A log time (or an aware moment) as the page gives it: the local date and
    the local hour and minute, ``("2026-11-01", "15:32")``."""
    moment = datetime.fromisoformat(time) if isinstance(time, str) else time
    local = moment.astimezone()
    return local.strftime("%Y-%m-%d"), local.strftime("%H:%M")


def _changes(count: int) -> str:
    return f"{count} change" if count == 1 else f"{count} changes"


# The step that takes a version Proteia set aside back, in its notices.
_GO_ON = (
    "To go on from {which} instead: open another project, rename project.json in the"
    " project's folder (to 'project mine.json', say) and rename {copy} to project.json,"
    " then open {name!r} again."
)


def _set_aside_notice(
    name: str, copy: str, moment: datetime, version: storage.Version, split: Divergence | None
) -> str:
    """What the page says of a ``project.json`` changed outside Proteia that a
    save kept in the conflict copy ``copy`` at ``moment`` (#101)."""
    day, hour = _local(moment)
    said = (
        f"{name!r} was changed outside Proteia after Proteia last opened or saved it."
        f" Proteia kept that version as {copy!r} in the project's folder, set aside on {day}"
        f" at {hour}, and saved its own as usual."
    )
    if version.newer is not None:
        return (
            f"{said} It was saved by a newer Proteia (schema {version.newer}): to go on from"
            " it, use that version of Proteia."
        )
    if split is None:
        return f"{said} It does not read as a project, so Proteia cannot open it."
    if split.their_last is not None:
        last_day, last_hour = _local(split.their_last)
        counts = (
            f"That version has {_changes(split.theirs)} this one does not, the last on"
            f" {last_day} at {last_hour}, and this one has {split.ours} it does not."
        )
    else:
        counts = (
            "That version's log has no change this one does not, so it is older or was"
            f" edited by hand; this one has {_changes(split.ours)} it does not."
        )
    go_on = _GO_ON.format(which="that version", copy="the copy", name=name)
    return f"{said} {counts} {go_on}"


def _outside_summary(version: storage.Version, split: Divergence | None) -> str:
    """What the session log says of a version changed outside Proteia that a
    save kept in a conflict copy."""
    if version.newer is not None:
        return f"it was saved by a newer Proteia, schema {version.newer}"
    if split is None:
        return "it does not read as a project"
    return (
        f"{split.lineage.value}: that version has {_changes(split.theirs)} this one does not,"
        f" this one {split.ours}"
    )


def _rewritten_notice(name: str) -> str:
    """What the page says of a ``project.json`` removed outside Proteia that a
    save wrote again (#101)."""
    return (
        f"The project.json of {name!r} was removed outside Proteia after Proteia last opened"
        " or saved it, so Proteia wrote it again."
    )


def _found_notice(name: str, copies: Sequence[str]) -> str:
    """What the page says of the conflict copies found in the folder when the
    project was opened or read again (#101), newest first."""
    shown = [repr(copy) for copy in copies[:3]]
    if len(copies) > 3:
        listed = f"{', '.join(shown)} and {len(copies) - 3} more"
    elif len(shown) > 1:
        listed = f"{', '.join(shown[:-1])} and {shown[-1]}"
    else:
        listed = shown[0]
    held = "1 version" if len(copies) == 1 else f"{len(copies)} versions"
    go_on = _GO_ON.format(which="one", copy="that copy", name=name)
    return (
        f"The folder of {name!r} holds {held} of the project that Proteia set aside when it was"
        f" changed outside Proteia: {listed}. Proteia never deletes them. {go_on}"
    )


class ProjectSession:
    """One open project folder. Create it with :func:`new_project` or :func:`open_project`.

    ``lock`` serializes every operation (an ``RLock``, because :meth:`save`
    re-enters it). ``dirty`` is True while the in-memory project differs from
    ``project.json``; ``save_error`` is the last failed save (None after a
    successful one); ``last_action`` names the last committed operation.
    ``clock`` gives the time of each log entry and export. :attr:`undo_step` and
    :attr:`redo_step` name what undo and redo would do. :attr:`outside_change`
    says what changed in the project's files outside Proteia (#101).
    """

    def __init__(
        self,
        project: Project,
        folder: str | os.PathLike[str],
        *,
        autosave: AutosaveHook | None,
        saved_files: Iterable[str],
        base: _Base,
        known: Iterable[str] = (),
        clock: Clock = utc_now,
    ) -> None:
        self._project = project
        self._folder = Path(folder)
        # The project.json this session last read or wrote: a save goes only
        # over it, or over the same document in other bytes (#101, #229).
        self._base = base
        # The conflict copy this session last kept: its name, the SHA-256 of the
        # bytes it holds and when it was kept, so a save that meets the same
        # bytes again (one that failed after the copy) keeps no second one.
        self._copy: tuple[str, str, datetime] | None = None
        # What the page says of this opening's last change outside Proteia,
        # and the conflict copies this opening kept, newest first; replaced
        # whole, so outside_change reads them without the lock.
        self._said: str | None = None
        self._kept: tuple[str, ...] = ()
        # The conflict copies in the folder when it was opened or read again.
        self._found: tuple[str, ...] = ()
        # The image names each version file in the folder holds, by its name,
        # with the size, modification time and file id they were read at.
        self._versions: dict[str, tuple[tuple[int, int, int], frozenset[str]]] = {}
        self.autosave = autosave
        self.clock = clock
        self.lock = threading.RLock()
        self.dirty = False
        self.save_error: Exception | None = None
        self.last_action: str | None = None
        self._pixels: dict[str, np.ndarray] = {}
        # Within an operation (transaction), the mean arrays read for ladder
        # work on images measured on another channel, by image id; None
        # between operations, so none is kept (calibration_pixels).
        self._calibration_reads: dict[str, np.ndarray] | None = None
        # (image id, SHA-256) -> how its stored file holds colour (file_colours).
        self._file_colours: dict[tuple[str, str], FileColours | None] = {}
        # (image id, SHA-256) -> what its stored file's header says of reading it
        # (outdated_reading): whether it converts CMYK, or what is refused in it.
        self._headers: dict[tuple[str, str], bool | str] = {}
        # The causes of the refusals of reading a stored file this session
        # logged (_LoggedCause), met least recently first: what the library
        # said is logged once per cause (_logs_refusal).
        self._refusals_logged: OrderedDict[_LoggedCause, None] = OrderedDict()
        # Names in images/ that the project.json on disk references: never orphans.
        self._saved_files = frozenset(saved_files)
        # The image files this session knows (lowercase names): those the
        # project.json it read (at open, or again) names. With the images whose
        # import the log records (the session's own imports among them), the
        # only image files its cleanup deletes (#101): a file synced before the
        # project.json that names it is never deleted.
        self._known = {name.lower() for name in (*saved_files, *known)}
        # The undo history: the states committed in this session, oldest first,
        # from the one it began with (added at its first change), and the index of
        # the committed one; empty and -1 before the first change and after close.
        self._states: list[_State] = []
        self._cursor = -1
        # What undo and redo would do, replaced whole whenever the history moves,
        # so a reader never waits for the lock an operation holds while it runs.
        self._steps: tuple[HistoryStep | None, HistoryStep | None] = (None, None)

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def project(self) -> Project:
        """The committed project: replaced on each change, never mutated. Callers
        must not mutate it either."""
        return self._project

    @property
    def history_steps(self) -> tuple[HistoryStep | None, HistoryStep | None]:
        """:attr:`undo_step` and :attr:`redo_step` as one read, so both are of
        the same commit. Read without the lock, so a reader never waits for a
        running operation; it sees the history as the last commit left it."""
        return self._steps

    @property
    def undo_step(self) -> HistoryStep | None:
        """The change undo would take back, or None (read without the lock)."""
        return self._steps[0]

    @property
    def redo_step(self) -> HistoryStep | None:
        """The change redo would make again, or None (read without the lock)."""
        return self._steps[1]

    def _steps_at_cursor(self) -> tuple[HistoryStep | None, HistoryStep | None]:
        """What undo and redo would do at the cursor. Called with the lock held."""
        cursor = self._cursor
        undo = self._step(cursor) if cursor > 0 else None
        redo = self._step(cursor + 1) if 0 <= cursor < len(self._states) - 1 else None
        return undo, redo

    def _step(self, index: int) -> HistoryStep:
        state = self._states[index]
        if state.seq is None:  # only the state a session began with can lack an entry
            raise RuntimeError("a change in the undo history has no log entry")
        return HistoryStep(seq=state.seq, action=state.action)

    def timestamp(self) -> str:
        """Now, from the session clock, in the log's form (UTC, milliseconds, Z)."""
        return format_timestamp(self.clock())

    def pixels(self, image_id: str, *, keep: bool = True) -> np.ndarray:
        """The image's read-only analysis array, of the channel the committed
        project measures it on (:attr:`~proteia.core.model.ImageRef.measured_channel`),
        read and checked on first use.

        ``keep=False`` reads it without adding it to the cache (for a display
        preview, which should not hold every viewed image in memory); an array
        already cached is returned either way.

        Raises ``UnknownIdError``, or :class:`OperationError` with
        ``IMAGE_FILE_CHANGED`` (the file is missing, its bytes no longer match the
        recorded SHA-256, or its shape differs) or ``UNREADABLE_IMAGE`` (it
        cannot be decoded, or its stored numbers come from a reading this version
        no longer makes, which the message says: :meth:`outdated_reading`).
        """
        with self.lock:
            cached = self._pixels.get(image_id)
            if cached is not None:
                return cached
            image = self._project.batch.find_image(image_id)
            array = self._read_array(image, image.measured_channel)
            if keep:
                self._pixels[image_id] = array
            return array

    def calibration_pixels(self, image_id: str) -> np.ndarray:
        """The image's read-only analysis array on the mean of its red, green
        and blue, whatever channel it is measured on (#206), for the
        molecular-weight calibration: a ladder is found and snapped on the
        mean, as before #206, so a channel chosen for a protein never changes
        ladder work. The cached array (:meth:`pixels`) when the image is
        measured on the mean; else read and checked once per operation
        (:meth:`transaction`), however many points it judges there, and not
        kept after it; outside an operation, read each time. Raises as
        :meth:`pixels` does."""
        with self.lock:
            image = self._project.batch.find_image(image_id)
            if image.measured_channel is Channel.MEAN:
                return self.pixels(image_id)
            reads = self._calibration_reads
            if reads is not None and image_id in reads:
                return reads[image_id]
            array = self._read_array(image, Channel.MEAN)
            if reads is not None:
                reads[image_id] = array
            return array

    def channel_array(self, image_id: str, channel: Channel) -> np.ndarray:
        """The image's read-only analysis array of ``channel`` (#206), read and
        checked as :meth:`pixels` reads it, with the same errors, and never
        cached: for a change of the channel the image is measured on, whose
        commit caches it."""
        with self.lock:
            return self._read_array(self._project.batch.find_image(image_id), channel)

    def _read_array(self, image: ImageRef, channel: Channel) -> np.ndarray:
        """The image's analysis array of ``channel``, read and checked
        (:meth:`_read_checked`), read-only. Called with the lock held."""
        array = self._read_checked(image, lambda path: self._analysis_array(image, path, channel))
        array.flags.writeable = False
        return array

    def _analysis_array(self, image: ImageRef, path: Path, channel: Channel) -> np.ndarray:
        """The analysis array of the image's stored file on ``channel``
        (:func:`load_image`), refused (``UNREADABLE_IMAGE``, in the words of
        :meth:`outdated_reading`, which name the image and say what to do)
        while its stored numbers come from a reading this version no longer
        makes: boxes measured now would mix the two readings."""
        why = self.outdated_reading(image)
        if why is not None:
            raise OperationError(ErrorCode.UNREADABLE_IMAGE, why, ids=(image.id,))
        return load_image(path, channel=channel).array

    def colour_pixels(self, image_id: str) -> np.ndarray:
        """The image's stored file in its own colours, for display only (2-D gray,
        or 3-D with the channels last: :func:`~proteia.core.imaging.read_colours`).
        Read and checked as :meth:`pixels` reads the analysis array, with the
        same errors, except that a CMYK image imported before CMYK was converted
        is shown in its converted colours; never cached. Analysis never uses them."""
        with self.lock:
            image = self._project.batch.find_image(image_id)
            return self._read_checked(image, read_colours)

    def outdated_reading(self, image: ImageRef) -> str | None:
        """Why the image's stored background and nets come from a reading of its
        stored file that this version no longer makes, and what to do; None if
        they do not, or while the file cannot be read (reading it says why).

        Two readings changed with #131. A CMYK file whose record has no
        ``cmyk_converted`` warning was imported before CMYK was converted: its
        cyan, magenta and yellow were read as red, green and blue, which its
        stored background, warnings and nets (and maybe its polarity, chosen by
        how it looked) come from. A file in a colour space reading now refuses
        (:class:`~proteia.core.imaging.UnsupportedColourSpaceError`: CIELAB,
        YCbCr, ...) was read as stored. Either way the image's analysis is
        refused with this message (:meth:`pixels`), the results warn about its
        nets, and exports refuse them, until it is removed: its stored numbers
        are not recomputed.

        From the header (:func:`~proteia.core.imaging.converts_cmyk`), read once
        per image record (id and SHA-256), without the lock, as
        :meth:`file_colours` reads it."""
        key = (image.id, image.sha256)
        header = self._headers.get(key)
        if header is None:
            try:
                header = converts_cmyk(storage.image_path(self._folder, image))
            except UnsupportedColourSpaceError as exc:
                header = exc.problem
            except (ValueError, OSError):
                return None
            self._headers[key] = header
        return _outdated_reading(image, header)

    def file_colours(self, image: ImageRef) -> FileColours | None:
        """How the image's stored file holds colour
        (:func:`~proteia.core.imaging.file_colours`), read from its header once
        per image record (id and SHA-256); None while the file cannot be read,
        and the next call reads it again. Never waits for the lock, so a state
        is drawn while an operation runs: Proteia writes a stored file once, on
        import, and never changes it (a file changed outside Proteia is refused
        when its pixels are read)."""
        key = (image.id, image.sha256)
        if key in self._file_colours:
            return self._file_colours[key]
        try:
            colours = file_colours(storage.image_path(self._folder, image))
        except (ValueError, OSError):
            return None
        self._file_colours[key] = colours
        return colours

    def _logs_refusal(self, image: ImageRef, exc: ValueError | OSError) -> bool:
        """Whether this refusal of reading the image's stored file (id and
        SHA-256) is to be logged: if it has a cause to log
        (:func:`~proteia.core.imaging.refusal_cause`) that the session has
        not met for the file among the last :data:`_LOGGED_CAUSES`. A cause
        is the refusal's own type, its error's type and errno, and whether
        that error is the system's, not its words, which may vary (an offset
        or a size in them). A cause met is the most recent; past the bound,
        the least recent is forgotten. Called with the lock held."""
        cause = refusal_cause(exc)
        if cause is None:
            return False
        errno = getattr(cause, "errno", None)
        key = (image.id, image.sha256, type(exc), type(cause), errno, of_the_system(cause))
        logged = self._refusals_logged
        new = key not in logged
        if new:
            logged[key] = None
            if len(logged) > _LOGGED_CAUSES:
                logged.popitem(last=False)  # the least recent
        else:
            logged.move_to_end(key)  # met again: the most recent
        return new

    def _read_checked(self, image: ImageRef, read: Callable[[Path], np.ndarray]) -> np.ndarray:
        """``read`` of the image's stored file, once the file's bytes match the
        recorded SHA-256; its first two dimensions must be the recorded height
        and width. A colour space reading refuses is refused with what that
        means for the image (:meth:`outdated_reading`), which names it as it was
        imported. Every other refusal of reading it names it as imported too,
        never by its stored copy (:func:`~proteia.core.imaging.named_refusal`,
        #234), and says what is wrong without a remedy: the image is in the
        project already, and importing its file again would lose its boxes.
        What the library said is logged once per cause of the stored file in
        the session (:meth:`_logs_refusal`), as every view and measure reads
        it again (#246). Called with the lock held."""
        image_id = image.id
        path = storage.image_path(self._folder, image)
        if not path.is_file():
            raise OperationError(
                ErrorCode.IMAGE_FILE_CHANGED,
                f"the stored file of image {image_id} is missing",
                ids=(image_id,),
            )

        def refused(exc: ValueError | OSError, why: str | None = None) -> OperationError:
            if why is None:
                log = self._logs_refusal(image, exc)
                why = named_refusal(exc, image.original_name, path, remedy=False, log=log)
            return OperationError(ErrorCode.UNREADABLE_IMAGE, why, ids=(image_id,))

        try:
            with path.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
        except OSError as exc:
            raise refused(exc) from exc
        if digest != image.sha256:
            raise OperationError(
                ErrorCode.IMAGE_FILE_CHANGED,
                f"the stored file of image {image_id} was changed outside Proteia",
                ids=(image_id,),
            )
        try:
            pixels = read(path)
        except OperationError:  # the read's own refusal, in its words (an outdated reading)
            raise
        except (ValueError, OSError) as exc:
            colour_space = isinstance(exc, UnsupportedColourSpaceError)
            raise refused(exc, self.outdated_reading(image) if colour_space else None) from exc
        if pixels.shape[:2] != (image.height, image.width):
            raise OperationError(
                ErrorCode.IMAGE_FILE_CHANGED,
                f"image {image_id} reads as {pixels.shape[1]}x{pixels.shape[0]} pixels,"
                f" not the recorded {image.width}x{image.height}",
                ids=(image_id,),
            )
        return pixels

    @property
    def outside_change(self) -> dict[str, JsonValue] | None:
        """What the page says of changes made to the project's files outside
        Proteia (#101), or None: ``message``, the notice of this opening's
        last one (a ``project.json`` kept in a conflict copy, or one written
        again), else of the conflict copies the folder held when the project
        was opened or read again; ``copies``, the conflict copies this opening
        kept, else those found, newest first. Read without the lock, as
        ``save_error`` is."""
        said, kept, found = self._said, self._kept, self._found
        if said is None:
            if not found:
                return None
            said = _found_notice(self._folder.name, found)
        return {"message": said, "copies": list(kept or found)}

    def save(self) -> Path:
        """Write ``project.json`` now and return its path; then delete the orphans
        no saved project references. ``OSError`` and ``ProjectError`` are recorded
        in ``save_error`` and re-raised.

        Only over the session's base (the file it last read or wrote), or one
        holding the same document in other bytes (#229); first a file changed
        outside Proteia is kept in a conflict copy (#101, :meth:`_write`)."""
        with self.lock:
            project = self._project
            try:
                self._write(project)
            except (OSError, storage.ProjectError) as exc:
                self.save_error = exc
                raise
            self._saved_files = _referenced_files(project)
            self.dirty = False
            self.save_error = None
            self._remove_orphans()
            return self._folder / storage.PROJECT_FILE

    def _write(self, project: Project) -> None:
        """Write ``project`` as the folder's ``project.json``, never over a file
        changed outside Proteia without keeping it. Called with the lock held.

        The file is read first (:func:`~proteia.core.storage.read_project_file`).
        Its bytes the base's, or the same document in other bytes (a BOM, CRLF,
        other key order or spacing; or what is about to be written), it is
        written over as usual. Other bytes are kept, as they are, in a conflict
        copy beside it (:func:`~proteia.core.storage.keep_conflict_copy`; none
        again for the same bytes while the copy kept last still holds them),
        then written over, and :attr:`outside_change` says so. A missing file
        is written again, and said, while the folder holds the project's
        images; with the folder gone (or an image missing too) nothing is
        made again (:class:`~proteia.core.storage.ProjectRemovedError`).
        Missing image files refuse as before (``MissingImageError``); a file
        that cannot be read refuses with the ``OSError``, a copy that cannot be
        written with :class:`~proteia.core.storage.CopyNotKeptError`, and a file
        changed again before the replace with
        :class:`~proteia.core.storage.ChangedAgainError`: nothing is written over
        it. The base becomes the bytes written."""
        folder, name = self._folder, self._folder.name
        data = storage.project_to_json(project)
        found = storage.read_project_file(folder)
        outside: tuple[bytes, str, storage.Version] | None = None
        if found is None:
            if storage.project_removed(project, folder):
                raise storage.ProjectRemovedError(name)
            expect = None
        else:
            expect = hashlib.sha256(found).hexdigest()
            if expect != self._base.sha256:
                version = storage.read_version(found)
                same = (self._base_canonical(), hashlib.sha256(data).hexdigest())
                if version.canonical is not None and version.canonical in same:
                    _log.info(
                        "in %r: its project.json was rewritten outside Proteia with the same"
                        " content; saved over it as usual",
                        name,
                    )
                else:
                    outside = (found, expect, version)
        storage.check_images(project, folder)
        said = None
        if outside is not None:
            found_bytes, digest, version = outside
            split = None
            if version.project is not None:
                split = divergence(project.log, self._base.log, version.project.log)
            resolution = resolve(split)
            if resolution != SET_ASIDE_OUTSIDE:  # never: the only resolution built (#101)
                raise NotImplementedError(f"no save for {resolution}")
            copy, moment = self._keep_copy(found_bytes, digest)
            said = _set_aside_notice(name, copy, moment, version, split)
            _log.warning(
                "in %r: its project.json was changed outside Proteia; kept that version as %r"
                " and saved Proteia's (%s)",
                name,
                copy,
                _outside_summary(version, split),
            )
        storage.write_project_file(folder, data, expect=expect)
        self._base = _written(data, project)
        if found is None:
            said = _rewritten_notice(name)
            _log.warning(
                "in %r: its project.json was removed outside Proteia; wrote it again", name
            )
        if said is not None:
            self._said = said

    def _base_canonical(self) -> str | None:
        """The SHA-256 of the base's canonical form (:class:`_Base`), computed
        once, when first asked. Called with the lock held."""
        base = self._base
        if base.canonical is None and base.parsed is not None:
            canonical = hashlib.sha256(storage.project_to_json(base.parsed)).hexdigest()
            base = self._base = dataclasses.replace(base, canonical=canonical, parsed=None)
        return base.canonical

    def _keep_copy(self, data: bytes, digest: str) -> tuple[str, datetime]:
        """The conflict copy that keeps ``data`` (SHA-256 ``digest``), the
        ``project.json`` found changed outside Proteia, and when it was kept: the
        one this session kept last if it still holds them, else a new one (its
        moment from the session clock). Called with the lock held."""
        kept = self._copy
        if kept is not None and kept[1] == digest and storage.holds(self._folder / kept[0], data):
            return kept[0], kept[2]
        moment = self.clock()
        copy = storage.keep_conflict_copy(self._folder, data, moment)
        self._copy = (copy, digest, moment)
        self._kept = (copy, *self._kept)
        return copy, moment

    def _commit(
        self,
        project: Project,
        *,
        action: str,
        params: Mapping[str, JsonValue],
        add_pixels: Mapping[str, np.ndarray] | None = None,
        evict: Iterable[str] = (),
        move: _Move = _Move.PUSH,
        expect_hash: str | None = None,
    ) -> None:
        """Append the change's log entry to ``project``, make it the committed
        project, move the undo history, then run the autosave hook.

        ``project`` must be prepared from the committed project (it shares its
        log), else ``RuntimeError``; content that does not hash to
        ``expect_hash`` (a restored state that is not what was committed) raises
        it too. The entry is built next: params that are not plain JSON
        (``ValidationError``) or a naive clock (``ValueError``) raise before
        anything changes.

        ``move`` ``PUSH``, an ordinary change, drops the states after the
        committed one (redo is gone), appends this one and keeps the last
        :data:`UNDO_LIMIT` steps; the first in a session also records the state
        the session began with. ``UNDO`` and ``REDO`` only move to the state
        before or after, which ``project`` restores (:meth:`_move`).

        A failed save (``OSError`` or ``ProjectError``) is recorded in
        ``save_error`` and logged; the edit stays committed. Any other exception
        from the hook is a bug and propagates, after the commit.

        ``add_pixels`` caches the arrays the change read (each of its image's
        measured channel), and ``evict`` drops the cached arrays of those
        images; the array of an image whose measured channel ``project``
        changes is dropped either way, unless ``add_pixels`` gives it.
        """
        with self.lock:
            log = self._project.log
            if project.log is not log:
                raise RuntimeError("stale project: prepare the change from the committed project")
            # Serialized once: the entry's hash and the history's state share it.
            data = storage.content_bytes(project)
            digest = hashlib.sha256(data).hexdigest()
            if expect_hash is not None and digest != expect_hash:
                raise RuntimeError("the restored content does not hash to the state it restores")
            seq = log[-1].seq + 1 if log else 1
            entry = _entry(
                seq=seq, action=action, params=params, content_hash=digest, clock=self.clock
            )
            pushed: list[_State] = []
            if move is _Move.PUSH:
                if not self._states:
                    pushed.append(self._opening_state())
                pushed.append(_state(project, seq=seq, action=action, data=data, digest=digest))
            # A cached array is of the channel its image was measured on (#206):
            # one the change (or the state undo and redo restore) measures on
            # another goes, unless the change gives it below.
            measured = {
                image.id: image.measured_channel for image in self._project.batch.iter_images()
            }
            for image in project.batch.iter_images():
                if measured.get(image.id, image.measured_channel) is not image.measured_channel:
                    self._pixels.pop(image.id, None)
            # One object holds the change and its entry, so one save writes both.
            self._project = project.model_copy(update={"log": (*log, entry)})
            for image_id, array in (add_pixels or {}).items():
                array.flags.writeable = False
                self._pixels[image_id] = array
            for image_id in evict:
                self._pixels.pop(image_id, None)
            if self._calibration_reads:  # read for the project before this commit
                self._calibration_reads.clear()
            # The history moves with the commit, before the hook saves (and so
            # deletes the files no state references any more).
            if move is _Move.PUSH:
                del self._states[self._cursor + 1 :]  # redo is gone
                self._states.extend(pushed)
                del self._states[: max(0, len(self._states) - (UNDO_LIMIT + 1))]
                self._cursor = len(self._states) - 1
            elif move is _Move.UNDO:
                self._cursor -= 1
            else:
                self._cursor += 1
            self._steps = self._steps_at_cursor()
            _log_entry(self._folder, entry)
            self._changed(action)

    def _changed(self, action: str) -> None:
        """Mark the committed project unsaved after its change ``action``, then
        run the autosave hook. Called with the lock held.

        A failed save (``OSError`` or ``ProjectError``) is recorded in
        ``save_error`` and logged; the change stays in memory. Any other
        exception from the hook is a bug and propagates.
        """
        self.dirty = True
        self.last_action = action
        if self.autosave is None:
            return
        try:
            self.autosave(self)
        except (OSError, storage.ProjectError) as exc:
            self.save_error = exc
            _log.warning("autosave after %s failed; the change is kept in memory: %s", action, exc)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Run one operation under the lock. If it is refused (``OperationError``
        or ``LookupError``) before anything was committed, the pixel cache is put
        back as it was, even if the operation read pixels first. The mean
        arrays its ladder work read (:meth:`calibration_pixels`) go when it
        ends, however it ends; an operation run inside another keeps them for
        the outer one."""
        with self.lock:
            project, cached = self._project, dict(self._pixels)
            outermost = self._calibration_reads is None
            if outermost:
                self._calibration_reads = {}
            try:
                yield
            except (OperationError, LookupError):
                if self._project is project:
                    self._pixels = cached
                raise
            finally:
                if outermost:
                    self._calibration_reads = None

    def _opening_state(self) -> _State:
        """The committed project as the state the session's history begins with.
        Its entry is the log's last, if that entry's hash is the content's; else
        (``project.json`` edited by hand, or no log) no entry left it."""
        project = self._project
        data = storage.content_bytes(project)
        digest = hashlib.sha256(data).hexdigest()
        last = project.log[-1] if project.log else None
        if last is None or last.content_hash != digest:
            return _state(project, seq=None, action="", data=data, digest=digest)
        return _state(project, seq=last.seq, action=last.action, data=data, digest=digest)

    def _move(self, direction: Literal["undo", "redo"]) -> dict[str, JsonValue]:
        """Undo or redo: commit the state before or after the committed one as a
        change (action ``direction``) and return its params.

        The state is restored whole, with the current ``next_id`` and log, and
        never recomputed from pixels: the content hash of the entry equals that of
        the entry it returns to (``returns_to_seq``; None if no entry recorded
        that content). The params name the change taken back or made again
        (``undone_seq`` and ``undone_action``, or ``redone_…``) and the ids and
        not-detected record keys (protein id, lane index, band index) that go
        (``removed``, ``undetected_removed``) or come back (``restored``,
        ``undetected_restored``), as :meth:`~proteia.core.model.Project.iter_ids`
        and the stored records order them (not sorted by id); an object whose
        fields change is in no list. Pixels of images the state lacks are
        evicted, on undo and redo alike, and so are those of images it measures
        on another channel (:meth:`_commit`); those it brings back are read, and
        checked, when next used.

        Refused, changing nothing: nothing to undo or redo (``NOTHING_TO_UNDO``,
        ``NOTHING_TO_REDO``), or an image file the state needs missing from
        ``images/`` (``IMAGE_FILE_CHANGED``, with those images). A history that
        does not match the committed project, or a state that does not read back
        as what was committed, is a bug: ``RuntimeError``, before the commit.
        """
        with self.lock:
            states, cursor = self._states, self._cursor
            if direction == "undo":
                if cursor <= 0:
                    raise OperationError(ErrorCode.NOTHING_TO_UNDO, "nothing to undo")
                step, target, move, verb = states[cursor], states[cursor - 1], _Move.UNDO, "undone"
            else:
                if not 0 <= cursor < len(states) - 1:
                    raise OperationError(ErrorCode.NOTHING_TO_REDO, "nothing to redo")
                step = target = states[cursor + 1]
                move, verb = _Move.REDO, "redone"
            current = self._project
            if not current.log or current.log[-1].content_hash != states[cursor].hash:
                raise RuntimeError("the undo history does not match the committed project")
            images = self._folder / storage.IMAGES_DIR
            missing = [
                image_id for image_id, file in target.images if not (images / file).is_file()
            ]
            if missing:
                raise OperationError(
                    ErrorCode.IMAGE_FILE_CHANGED,
                    f"cannot {direction}: the stored file of {', '.join(missing)} is missing",
                    ids=missing,
                )
            try:
                restored = storage.project_from_content(
                    zlib.decompress(target.content), next_id=current.next_id, log=current.log
                )
            except (zlib.error, ValueError) as exc:  # a ValidationError is a ValueError
                raise RuntimeError("a state in the undo history cannot be read back") from exc
            old_ids, new_ids = list(current.iter_ids()), list(restored.iter_ids())
            old_keys, new_keys = _record_keys(current), _record_keys(restored)
            params: dict[str, JsonValue] = {
                f"{verb}_seq": step.seq,
                f"{verb}_action": step.action,
                "returns_to_seq": target.seq,
                "removed": _only_in(old_ids, new_ids),
                "restored": _only_in(new_ids, old_ids),
                "undetected_removed": [list(key) for key in _only_in(old_keys, new_keys)],
                "undetected_restored": [list(key) for key in _only_in(new_keys, old_keys)],
            }
            kept = {image.id for image in restored.batch.iter_images()}
            evict = [image_id for image_id in self._pixels if image_id not in kept]
            self._commit(
                restored,
                action=direction,
                params=params,
                evict=evict,
                move=move,
                expect_hash=target.hash,
            )
            return params

    def reload(self) -> bool:
        """Read ``project.json`` again if it is not what saving the committed
        project writes, as when it was replaced outside Proteia (a synced or
        restored copy); return whether it did. The session then holds what
        :func:`open_project` reads from the folder (a file of an older schema
        is migrated and saved, as there, once it is kept in a backup), with no
        cached pixels and no undo history, whose states led to a project the
        file no longer holds; the files only that history kept go at the next
        save or import.

        With unsaved changes (``dirty``) the file is not read: it is older than
        the committed project, not changed outside Proteia. ``OSError`` (a
        missing file too, or a backup of a file of an older schema that cannot
        be kept: :class:`~proteia.core.storage.BackupError`) and the
        :class:`~proteia.core.storage.ProjectError` family propagate, with the
        session as it was.

        Read again, the file is the session's base (#101): what this opening
        said of changes outside Proteia (:attr:`outside_change`) is over, and
        the conflict copies the folder holds now are said instead. Not read,
        the opening goes on as it was.
        """
        with self.lock:
            if self.dirty:
                return False
            saved = (self._folder / storage.PROJECT_FILE).read_bytes()
            if saved == storage.project_to_json(self._project):
                return False
            self._read_again(saved)
            return True

    def _read_again(self, saved: bytes) -> None:
        """Take ``saved``, the bytes of ``project.json`` now, as the project, as
        :func:`open_project` reads them (a file of an older schema is migrated
        and saved, once kept in a backup): its base (#101), with no cached
        pixels, no undo history, and nothing said yet of changes outside
        Proteia but the conflict copies the folder holds. Whatever the session
        held is dropped, saved or not; the caller decides it may be. The
        ``OSError`` and :class:`~proteia.core.storage.ProjectError` of reading
        it propagate, with the session as it was. Called with the lock held."""
        project, migrated = storage.read_project(
            self._folder, clock=self.clock, backup=True, data=saved
        )
        self._project = project
        self._base = _read(saved, project, migrated=migrated)
        self._copy, self._said, self._kept = None, None, ()
        self._saved_files = _referenced_files(project)
        self._known.update(
            name.lower() for name in (*self._saved_files, *storage.image_names(saved))
        )
        self._pixels = {}
        self._states, self._cursor = [], -1
        self._steps = (None, None)
        self.dirty, self.save_error = False, None
        self.last_action = None
        _log_opened(self._folder, project, again=True)
        self._found = _found_copies(self._folder)
        if migrated:
            _log_migrated(self._folder, project.log[-1])
            self._changed("migrate")

    def close(self, *, remove_files: bool = True) -> None:
        """Forget the undo history, then delete the files only it kept (best
        effort). The session can still be used; its history starts again.

        ``remove_files=False`` deletes nothing: for when another session has
        the folder open, whose new files this one does not know. The first save
        or import of that session deletes what only this history kept.
        """
        with self.lock:
            self._states, self._cursor = [], -1
            self._steps = (None, None)
            if remove_files:
                with contextlib.suppress(OSError):
                    self._remove_orphans()
            _log.info("closed %r", self._folder.name)

    def _retained_files(self) -> frozenset[str]:
        """The files of every image a state in the undo history references."""
        return frozenset(file for state in self._states for _, file in state.images)

    def _remove_orphans(self) -> None:
        """Delete unreferenced Proteia files in ``images/`` that neither the saved
        ``project.json``, nor a state in the undo history, nor a backup of an
        older ``project.json``, nor another version of the project references
        (best effort: a file held open on Windows stays), and, of the image
        files, only those the session knows (``_known``, or an image whose
        import the project's log records: never one synced from another
        computer before its project.json, #101). Names compare ignoring case,
        as Windows does.

        Another version (#101, :meth:`_named_by_versions`) keeps every image
        file its text names, whatever its id: a conflict copy, any other
        ``.json`` file in the folder, and ``project.json`` itself while it is
        not the session's base (changed outside Proteia and not yet kept in a
        copy). While one of them cannot be read, or ``project.json`` is
        missing, no image file is deleted (only temp files).

        A backup keeps only an image file whose id is below ``next_id``: a
        backup's images hold no other id (a migration keeps ``next_id``, and the
        project only raises it). So the file of the id an import takes next is
        never kept for a backup, even where a backup's text names it (as an
        original name) or a backup cannot be read, which keeps every image file
        below ``next_id``, as what it names is unknown."""
        referenced = self._saved_files | _referenced_files(self._project) | self._retained_files()
        keep = {name.lower() for name in referenced}
        orphans = [
            path
            for path in storage.orphan_files(self._project, self._folder)
            if path.name.lower() not in keep and _ORPHAN_NAME.match(path.name)
        ]
        # An image file only if the session knows it (a temp file always).
        strangers = [
            path
            for path in orphans
            if _IMAGE_FILE.match(path.name) and path.name.lower() not in self._known
        ]
        if strangers:  # the log is read only then
            imported = _imported(self._project)
            left = {path for path in strangers if not _of_an_import(path, imported)}
            orphans = [path for path in orphans if path not in left]
        if not orphans:  # the versions are read only then: most saves leave no orphan
            return
        versions = self._named_by_versions()
        if versions is None:  # what they name is unknown: temp files only
            orphans = [path for path in orphans if not _IMAGE_FILE.match(path.name)]
        else:
            orphans = [path for path in orphans if path.name.lower() not in versions]
        next_id = self._project.next_id
        below = [
            path
            for path in orphans
            if (match := _IMAGE_FILE.match(path.name)) and int(match[1]) < next_id
        ]
        spared: set[Path] = set()
        if below:  # the backups are read only then: most saves leave no orphan
            backed_up = storage.backup_references(self._folder)
            named = None if backed_up is None else {name.lower() for name in backed_up}
            spared = {path for path in below if named is None or path.name.lower() in named}
        for path in orphans:
            if path not in spared:
                with contextlib.suppress(OSError):
                    path.unlink()

    def _named_by_versions(self) -> frozenset[str] | None:
        """The image files (lowercase names) that the other versions of the
        project in its folder name anywhere in their text (#101): every
        version file (:func:`~proteia.core.storage.is_version`), each read once
        while its size, modification time and file id stay the same, and
        ``project.json`` itself while it is not the session's base. None when
        that is unknown: the folder cannot be listed, a file cannot be read, or
        ``project.json`` is missing. Called with the lock held."""
        names: set[str] = set()
        try:
            found = storage.read_project_file(self._folder)
            if found is None:
                return None
            if hashlib.sha256(found).hexdigest() != self._base.sha256:
                names.update(storage.image_names(found))
            for path in self._folder.iterdir():
                if storage.is_version(path.name) and path.is_file():
                    names.update(self._version_names(path))
        except OSError:
            return None
        return frozenset(name.lower() for name in names)

    def _version_names(self, path: Path) -> frozenset[str]:
        """The image names the version file ``path`` holds
        (:func:`~proteia.core.storage.named_images`), kept by its name and read
        again once its size, modification time or file id changed. An
        ``OSError`` propagates."""
        stat = path.stat()
        key = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        cached = self._versions.get(path.name)
        if cached is not None and cached[0] == key:
            return cached[1]
        names = storage.named_images(path)
        self._versions[path.name] = (key, names)
        return names

    def image_number(self) -> int:
        """The number of the id an image imported now takes: the first from
        ``next_id`` on that no file in ``images/`` holds (an image of another
        version, or one cleanup could not delete) and that no other version of
        the project names (:meth:`_named_by_versions`: an image that the
        version's computer stored, which has not arrived yet), whatever the
        suffix and case (#101). Called with the lock held."""
        held: set[int] = set()
        try:
            names = [path.name for path in (self._folder / storage.IMAGES_DIR).iterdir()]
        except OSError:
            names = []
        for name in (*names, *(self._named_by_versions() or ())):
            if match := _IMAGE_FILE.match(name):
                held.add(int(match[1]))
        number = self._project.next_id
        while number in held:
            number += 1
        return number


def _image_id(name: str) -> str:
    """The image id of a stored image file's name (``_IMAGE_FILE``):
    ``img-3.tif`` is ``img-3``, ``IMG-3.TIF`` too."""
    match = _IMAGE_FILE.match(name)
    if match is None:  # never: called only for names _IMAGE_FILE matched
        raise ValueError(f"{name!r} is no stored image file's name")
    return f"img-{int(match[1])}"


def _imported(project: Project) -> dict[str, set[str]]:
    """The images whose import ``project``'s log records, whatever happened to
    them since: each id with the SHA-256 of the file each import stored (its
    ``sha256`` param), for :func:`_of_an_import` (#101)."""
    found: dict[str, set[str]] = {}
    for entry in project.log:
        if entry.action == "import_image":
            image_id, digest = entry.params.get("image_id"), entry.params.get("sha256")
            found.setdefault(str(image_id), set()).add(str(digest))
    return found


def _of_an_import(path: Path, imported: Mapping[str, set[str]]) -> bool:
    """Whether the image file ``path`` is one the project's own history
    imported (:func:`_imported`): an import recorded its id, and its bytes are
    the ones that import stored. Not so for another computer's image of the
    same id (two versions that each took it), nor while the file cannot be
    read."""
    digests = imported.get(_image_id(path.name))
    if not digests:
        return False
    try:
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest() in digests
    except OSError:
        return False


def _found_copies(folder: Path) -> tuple[str, ...]:
    """The conflict copies in ``folder`` as a project is opened or read again,
    logged when there are any (#101)."""
    found = storage.conflict_copies(folder)
    if found:
        _log.info(
            "in %r: %d versions Proteia set aside when it was changed outside Proteia: %s",
            folder.name,
            len(found),
            ", ".join(repr(name) for name in found),
        )
    return found


def save_to_folder(session: ProjectSession) -> None:
    """The default autosave hook: save the project into its folder."""
    session.save()


def new_project(
    folder: str | os.PathLike[str],
    *,
    autosave: AutosaveHook | None = save_to_folder,
    clock: Clock = utc_now,
) -> ProjectSession:
    """Create an empty project in ``folder`` and open it.

    ``folder`` must not exist or must be an empty directory (orphan cleanup deletes
    files in ``images/``); otherwise ``FOLDER_NOT_EMPTY``, and so if another
    program writes a ``project.json`` there meanwhile, which is left as it is
    (#101). ``project.json``, ``images/`` and ``exports/`` are written at once;
    an ``OSError`` propagates. The log starts with one ``new_project`` entry.
    """
    path = Path(folder)
    not_empty = OperationError(
        ErrorCode.FOLDER_NOT_EMPTY, f"{path} exists and is not an empty folder"
    )
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise not_empty
    created = _entry(
        seq=1,
        action="new_project",
        params={},
        content_hash=storage.content_hash(Project()),
        clock=clock,
    )
    project = Project(log=(created,))
    data = storage.project_to_json(project)
    try:
        storage.write_project_file(path, data, expect=None, create=True)
    except storage.ChangedAgainError as exc:
        raise not_empty from exc
    _log_entry(path, created)
    return ProjectSession(
        project, path, autosave=autosave, saved_files=(), base=_written(data, project), clock=clock
    )


def open_project(
    folder: str | os.PathLike[str],
    *,
    autosave: AutosaveHook | None = save_to_folder,
    clock: Clock = utc_now,
) -> ProjectSession:
    """Open the project in ``folder``, with every image file required.

    ``FileNotFoundError`` and the :class:`~proteia.core.storage.ProjectError` family
    propagate. Opening deletes nothing, rewrites nothing, reads no pixels and
    logs nothing, except for a file of an older schema. That is first kept, as
    it was read, in a backup beside it (:func:`~proteia.core.storage.keep_backup`,
    which never replaces a file), then migrated with a ``migrate`` entry timed
    by ``clock`` that names the backup (:func:`~proteia.core.storage.read_project`):
    a logged change like any other, so the session starts dirty and runs the
    autosave hook at once (whose save also deletes orphans, never the file of
    an image a backup holds; a failed save is recorded in ``save_error``, as after any
    change). A backup that cannot be written refuses the open
    (:class:`~proteia.core.storage.BackupError`, an ``OSError``), and nothing is
    written. A record exported before any edit then cites the log
    ``project.json`` holds. The undo history begins from the migrated state.

    The bytes read are the session's base, which a save goes over (#101); the
    conflict copies the folder holds are said (:attr:`ProjectSession.outside_change`)
    and logged.
    """
    data = (Path(folder) / storage.PROJECT_FILE).read_bytes()
    project, migrated = storage.read_project(folder, clock=clock, backup=True, data=data)
    session = ProjectSession(
        project,
        folder,
        autosave=autosave,
        saved_files=_referenced_files(project),
        base=_read(data, project, migrated=migrated),
        known=storage.image_names(data),
        clock=clock,
    )
    _log_opened(session.folder, project)
    session._found = _found_copies(session.folder)
    if migrated:
        with session.lock:
            _log_migrated(session.folder, project.log[-1])
            session._changed("migrate")
    return session
