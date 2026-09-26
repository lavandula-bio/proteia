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
* Files in ``images/`` that no project references (an image removed, an import
  that was never saved, a temp file left by a crash) are deleted after a
  successful save and before an import, but only if the ``project.json`` on disk
  does not reference them and they carry Proteia's ``img-N`` name: a saved project
  never points at a missing file, and a stray user file is never deleted.

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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import JsonValue

import proteia
from proteia.core import storage
from proteia.core.imaging import load_image
from proteia.core.model import IMAGE_SUFFIXES, LogEntry, Project, format_timestamp

_log = logging.getLogger(__name__)

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


class OperationError(ValueError):
    """An operation refused its input. Nothing changed: the project (the same
    object), ``next_id`` and the pixel cache are as they were, no file was added to
    ``images/``, and the autosave hook did not run. (An import removes orphan files,
    which no project references, before it can be refused.)"""

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
    project: Project, *, seq: int, action: str, params: Mapping[str, JsonValue], clock: Clock
) -> LogEntry:
    """The log entry of a change that leaves ``project``. Raises ``ValidationError``
    for params that are not plain JSON, ``ValueError`` for a naive clock."""
    return LogEntry(
        seq=seq,
        time=format_timestamp(clock()),
        action=action,
        version=proteia.__version__,  # read at call time: what is running now
        params=dict(params),
        content_hash=storage.content_hash(project),
    )


class ProjectSession:
    """One open project folder. Create it with :func:`new_project` or :func:`open_project`.

    ``lock`` serializes every operation (an ``RLock``, because :meth:`save`
    re-enters it). ``dirty`` is True while the in-memory project differs from
    ``project.json``; ``save_error`` is the last failed save (None after a
    successful one); ``last_action`` names the last committed operation.
    ``clock`` gives the time of each log entry and export.
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

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def project(self) -> Project:
        """The committed project: replaced on each change, never mutated. Callers
        must not mutate it either."""
        return self._project

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
    ) -> None:
        """Append the change's log entry to ``project``, make it the committed
        project, then run the autosave hook.

        ``project`` must be prepared from the committed project (it shares its
        log), else ``RuntimeError``. The entry is built first: params that are not
        plain JSON (``ValidationError``) or a naive clock (``ValueError``) raise
        before anything changes. A failed save (``OSError`` or ``ProjectError``)
        is recorded in ``save_error`` and logged; the edit stays committed. Any
        other exception from the hook is a bug and propagates, after the commit.
        """
        with self.lock:
            log = self._project.log
            if project.log is not log:
                raise RuntimeError("stale project: prepare the change from the committed project")
            seq = log[-1].seq + 1 if log else 1
            entry = _entry(project, seq=seq, action=action, params=params, clock=self.clock)
            # One object holds the change and its entry, so one save writes both.
            self._project = project.model_copy(update={"log": (*log, entry)})
            for image_id, array in (add_pixels or {}).items():
                array.flags.writeable = False
                self._pixels[image_id] = array
            for image_id in evict:
                self._pixels.pop(image_id, None)
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

    def _remove_orphans(self) -> None:
        """Delete unreferenced Proteia files in ``images/`` that the saved
        ``project.json`` does not reference either (best effort: a file held open
        on Windows stays). Names compare ignoring case, as Windows does."""
        keep = {name.lower() for name in self._saved_files | _referenced_files(self._project)}
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
    created = _entry(Project(), seq=1, action="new_project", params={}, clock=clock)
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
