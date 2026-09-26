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
* Every committed change appends one :class:`~proteia.core.model.LogEntry` at
  that single commit point, in the same project object, so one save writes the
  change and its entry together; refusals and no-ops append nothing. Times come
  from the session's injectable clock (UTC; the offline system clock by default).
* Pixels are read lazily and cached as read-only arrays, after checking the
  stored file's SHA-256 and the array's shape against the image record.
* The session keeps its undo history: the content of each state it committed,
  up to :data:`UNDO_LIMIT` steps back, as the exact bytes the content hash
  covers (:func:`~proteia.core.storage.content_bytes`). The history moves at the
  single commit point, before the autosave hook runs. Undo and redo restore a
  state whole, never replaying a change, and commit it as a logged change of
  their own (``undo``, ``redo``), so the log only grows. The history is per
  session: :func:`open_project` starts with none, and it is lost at
  :meth:`ProjectSession.close`, a project switch or a restart; the log, which
  keeps params and hashes but not states, cannot rebuild it.
* Files in ``images/`` that no project references (an image removed, an import
  that was never saved, a temp file left by a crash) are deleted after a
  successful save and before an import, but only if the ``project.json`` on
  disk does not reference them, no state in the undo history does, and they
  carry Proteia's ``img-N`` name: a saved project never points at a missing
  file, an undo never needs one, and a stray user file is never deleted. So a
  removed or undone image's file stays while the history can bring it back, and
  goes at the first save or import after no state references it: once the undo
  limit drops the last such state, a new change clears the redo that held it,
  or the session is closed; or else at the next session's first save.

Refusals raise :class:`OperationError`, whose :class:`ErrorCode` is stable for
clients (e.g. to map onto HTTP statuses).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import threading
import zlib
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
from proteia.core.imaging import load_image
from proteia.core.model import IMAGE_SUFFIXES, LogEntry, Project, format_timestamp

_log = logging.getLogger(__name__)

# The undoable steps a session keeps. A count, not a byte budget: a state is its
# compressed content, about 24 KiB for a batch of 960 bands.
UNDO_LIMIT: Final = 100

# Proteia's image names (img-N.<suffix>) and the temp files storage writes while
# storing or saving (.img-N.<suffix>.<random>.part/.tmp): the only orphans cleanup
# may delete. The whole name must match, so a user's img-1.tif.bak survives; case
# is ignored because Windows file names ignore it.
_SUFFIXES = "|".join(re.escape(suffix.removeprefix(".")) for suffix in IMAGE_SUFFIXES)
_ORPHAN_NAME = re.compile(
    rf"^\.?img-[1-9][0-9]{{0,8}}\.(?:{_SUFFIXES})(?:\.[a-z0-9_]+\.(?:part|tmp))?$",
    re.IGNORECASE,
)


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
    OVERLAP = "overlap"
    SIZE_WOULD_OVERLAP = "size_would_overlap"
    SIZE_OUT_OF_BOUNDS = "size_out_of_bounds"
    UNSUPPORTED_IMAGE_TYPE = "unsupported_image_type"
    IMAGE_TOO_LARGE = "image_too_large"
    INVALID_IMAGE = "invalid_image"  # an empty stream, or a name store_image refuses
    UNREADABLE_IMAGE = "unreadable_image"  # the loader cannot decode it
    IMAGE_FILE_CHANGED = "image_file_changed"  # stored file missing, other bytes, other shape
    FOLDER_NOT_EMPTY = "folder_not_empty"
    LEFTOVER_FILE = "leftover_file"  # an orphan holding the next image id could not be deleted
    NOTHING_TO_UNDO = "nothing_to_undo"
    NOTHING_TO_REDO = "nothing_to_redo"


class OperationError(ValueError):
    """An operation refused its input. Nothing changed: the project (the same
    object), ``next_id``, the undo history and the pixel cache are as they were, no
    file was added to ``images/``, and the autosave hook did not run. (An import
    removes orphan files, which no project or undo state references, before it can
    be refused.)"""

    def __init__(self, code: ErrorCode, message: str, *, ids: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.ids: tuple[str, ...] = tuple(ids)  # the objects involved, e.g. an overlapped band


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
    return [
        (protein.id, record.lane_index, record.band_index)
        for protein in project.batch.proteins
        for record in protein.undetected
    ]


def _only_in[T](items: Sequence[T], others: Sequence[T]) -> list[T]:
    """The items of ``items`` not in ``others``, in their order."""
    exclude = set(others)
    return [item for item in items if item not in exclude]


class ProjectSession:
    """One open project folder. Create it with :func:`new_project` or :func:`open_project`.

    ``lock`` serializes every operation (an ``RLock``, because :meth:`save`
    re-enters it). ``dirty`` is True while the in-memory project differs from
    ``project.json``; ``save_error`` is the last failed save (None after a
    successful one); ``last_action`` names the last committed operation.
    ``clock`` gives the time of each log entry and export. :attr:`undo_step` and
    :attr:`redo_step` name what undo and redo would do.
    """

    def __init__(
        self,
        project: Project,
        folder: str | os.PathLike[str],
        *,
        autosave: AutosaveHook | None,
        saved_files: Iterable[str],
        clock: Clock = utc_now,
    ) -> None:
        self._project = project
        self._folder = Path(folder)
        self.autosave = autosave
        self.clock = clock
        self.lock = threading.RLock()
        self.dirty = False
        self.save_error: Exception | None = None
        self.last_action: str | None = None
        self._pixels: dict[str, np.ndarray] = {}
        # Names in images/ that the project.json on disk references: never orphans.
        self._saved_files = frozenset(saved_files)
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
        """The image's read-only analysis array, read and checked on first use.

        ``keep=False`` reads it without adding it to the cache (for a display
        preview, which should not hold every viewed image in memory); an array
        already cached is returned either way.

        Raises ``UnknownIdError``, or :class:`OperationError` with
        ``IMAGE_FILE_CHANGED`` (the file is missing, its bytes no longer match the
        recorded SHA-256, or its shape differs) or ``UNREADABLE_IMAGE``.
        """
        with self.lock:
            cached = self._pixels.get(image_id)
            if cached is not None:
                return cached
            image = self._project.batch.find_image(image_id)
            path = storage.image_path(self._folder, image)
            if not path.is_file():
                raise OperationError(
                    ErrorCode.IMAGE_FILE_CHANGED,
                    f"the stored file of image {image_id} is missing",
                    ids=(image_id,),
                )
            try:
                with path.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
            except OSError as exc:
                raise OperationError(ErrorCode.UNREADABLE_IMAGE, str(exc), ids=(image_id,)) from exc
            if digest != image.sha256:
                raise OperationError(
                    ErrorCode.IMAGE_FILE_CHANGED,
                    f"the stored file of image {image_id} was changed outside Proteia",
                    ids=(image_id,),
                )
            try:
                array = load_image(path).array
            except (ValueError, OSError) as exc:
                raise OperationError(ErrorCode.UNREADABLE_IMAGE, str(exc), ids=(image_id,)) from exc
            if array.shape != (image.height, image.width):
                raise OperationError(
                    ErrorCode.IMAGE_FILE_CHANGED,
                    f"image {image_id} reads as {array.shape[1]}x{array.shape[0]} pixels,"
                    f" not the recorded {image.width}x{image.height}",
                    ids=(image_id,),
                )
            array.flags.writeable = False
            if keep:
                self._pixels[image_id] = array
            return array

    def save(self) -> Path:
        """Write ``project.json`` now and return its path; then delete the orphans
        no saved project references. ``OSError`` and ``ProjectError`` are recorded
        in ``save_error`` and re-raised."""
        with self.lock:
            project = self._project
            try:
                path = storage.save_project(project, self._folder)
            except (OSError, storage.ProjectError) as exc:
                self.save_error = exc
                raise
            self._saved_files = _referenced_files(project)
            self.dirty = False
            self.save_error = None
            self._remove_orphans()
            return path

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
            # One object holds the change and its entry, so one save writes both.
            self._project = project.model_copy(update={"log": (*log, entry)})
            for image_id, array in (add_pixels or {}).items():
                array.flags.writeable = False
                self._pixels[image_id] = array
            for image_id in evict:
                self._pixels.pop(image_id, None)
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
            self.dirty = True
            self.last_action = action
            if self.autosave is None:
                return
            try:
                self.autosave(self)
            except (OSError, storage.ProjectError) as exc:
                self.save_error = exc
                _log.warning(
                    "autosave after %s failed; the change is kept in memory: %s", action, exc
                )

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Run one operation under the lock. If it is refused (``OperationError``
        or ``LookupError``) before anything was committed, the pixel cache is put
        back as it was, even if the operation read pixels first."""
        with self.lock:
            project, cached = self._project, dict(self._pixels)
            try:
                yield
            except (OperationError, LookupError):
                if self._project is project:
                    self._pixels = cached
                raise

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
        evicted, on undo and redo alike; those it brings back are read, and
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

    def _retained_files(self) -> frozenset[str]:
        """The files of every image a state in the undo history references."""
        return frozenset(file for state in self._states for _, file in state.images)

    def _remove_orphans(self) -> None:
        """Delete unreferenced Proteia files in ``images/`` that neither the saved
        ``project.json`` nor a state in the undo history references (best effort:
        a file held open on Windows stays). Names compare ignoring case, as
        Windows does."""
        referenced = self._saved_files | _referenced_files(self._project) | self._retained_files()
        keep = {name.lower() for name in referenced}
        for path in storage.orphan_files(self._project, self._folder):
            if path.name.lower() in keep or not _ORPHAN_NAME.match(path.name):
                continue
            with contextlib.suppress(OSError):
                path.unlink()


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
    files in ``images/``); otherwise ``FOLDER_NOT_EMPTY``. ``project.json``,
    ``images/`` and ``exports/`` are written at once; an ``OSError`` propagates.
    The log starts with one ``new_project`` entry.
    """
    path = Path(folder)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise OperationError(
            ErrorCode.FOLDER_NOT_EMPTY, f"{path} exists and is not an empty folder"
        )
    created = _entry(
        seq=1,
        action="new_project",
        params={},
        content_hash=storage.content_hash(Project()),
        clock=clock,
    )
    project = Project(log=(created,))
    storage.save_project(project, path)
    return ProjectSession(project, path, autosave=autosave, saved_files=(), clock=clock)


def open_project(
    folder: str | os.PathLike[str],
    *,
    autosave: AutosaveHook | None = save_to_folder,
    clock: Clock = utc_now,
) -> ProjectSession:
    """Open the project in ``folder``, with every image file required.

    ``FileNotFoundError`` and the :class:`~proteia.core.storage.ProjectError` family
    propagate. Opening deletes nothing, rewrites nothing, reads no pixels and
    logs nothing.
    """
    project = storage.load_project(folder)
    return ProjectSession(
        project, folder, autosave=autosave, saved_files=_referenced_files(project), clock=clock
    )
