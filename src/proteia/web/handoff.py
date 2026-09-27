# SPDX-License-Identifier: Apache-2.0
"""Images handed to the running app, held until its page imports them (#57, N3).

A launch given image files hands them to the running Proteia, and the page
asks how to import them (kind, membrane, and whether bands are dark or light)
before anything is imported: polarity has no default. Until then they wait
here, in the :class:`Inbox`, in memory; none of this survives the server.

Files arrive two ways, as two types, so that nothing but a copy the server made
is ever deleted:

* :class:`StagedFile`: bytes a later launch uploaded (``POST /api/incoming``),
  stored in the staging folder (``incoming/`` in the per-user state folder,
  private to the user) under a name the server made: 32 hexadecimal digits,
  never the client's name. Only a staged file can be deleted, and only in that
  folder.
* :class:`LocalFile`: a file named on the first launch's own command line,
  read where it is when imported. Its path comes from the server process
  itself, never from an HTTP client. It has no way to be deleted, and no
  cleanup takes one.

A launch then offers its files, with the arguments it refused and why
(:class:`Refusal`), as one hand-off (:meth:`Inbox.offer`). Launches that run at
once (a selection opened with Proteia in Explorer, one process per file) join
one hand-off: an offer merges into the newest one no accept has claimed if its
files began to upload within :data:`MERGE_WINDOW_S` of that hand-off's last,
and the result holds at most :data:`MAX_HANDOFF_FILES` files; measured from the
start of each upload, so a large file does not split a selection. The page
lists the hand-offs (:meth:`Inbox.listing`), then imports one (an accept claims
it, :meth:`Inbox.claim`; files offered meanwhile start another) or discards it
(:meth:`Inbox.discard`). A claimed hand-off is still listed, as claimed: no
other accept or discard takes it, but the accept may be refused and release it
as it was, so a page showing it keeps its choices until it is gone.

Limits: at most :data:`MAX_PENDING_FILES` files wait at a time, in hand-offs
or uploaded but not yet offered, and at most :data:`MAX_STAGED_BYTES` are
staged; an upload no offer takes within :data:`UPLOAD_EXPIRY_S` is deleted
(checked at each upload and offer). A refused argument never makes an offer
fail: its entry is bounded instead (:meth:`Refusal.bounded`).

Staged files are deleted once their hand-off is imported or discarded, when
their upload expires, when the server stops (:meth:`Inbox.close`), and at the
next start, which deletes whatever a crash left in the staging folder
(:meth:`Inbox.place`).
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final, Literal

from pydantic import TypeAdapter, ValidationError

from proteia.core.model import IMAGE_SUFFIXES, OriginalName, UnknownIdError
from proteia.core.session import ErrorCode, OperationError

_log = logging.getLogger(__name__)

INCOMING_DIR: Final = "incoming"  # the staging folder, in the per-user state folder
MERGE_WINDOW_S: Final = 10.0
MAX_HANDOFF_FILES: Final = 16
MAX_PENDING_FILES: Final = 32
MAX_STAGED_BYTES: Final = 2 * 1024**3
UPLOAD_EXPIRY_S: Final = 600.0
MAX_REFUSED: Final = 100  # refused entries kept per hand-off; the rest are counted
MAX_REFUSED_NAME: Final = 120  # characters of a refused entry's name kept
MAX_REFUSED_MESSAGE: Final = 200  # characters of a refused entry's message kept
# The codes a refused entry may carry: the launcher's checks of its arguments,
# and the refusals of POST /api/incoming it reports. Any other is ``other``.
REFUSAL_CODES: Final = frozenset(
    {
        "bad_argument",
        "unreadable",
        "missing",
        "folder",
        "project_file",
        "unsupported_type",
        "empty",
        "too_large",
        "bad_name",
        "unsupported_image_type",
        "invalid_image",
        "image_too_large",
        "too_many_pending",
        "stopping",
        "file_error",
        "other",
    }
)
_STAGED_NAME: Final = re.compile(r"[0-9a-f]{32}")
# Characters a refused entry never shows: control and format characters (bidi
# overrides included), unpaired surrogates, which no UTF-8 answer can carry,
# and line and paragraph separators.
_UNSHOWN: Final = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
_ORIGINAL_NAME: Final = TypeAdapter(OriginalName)

Clock = Callable[[], float]  # seconds, monotonic


class StoppingError(RuntimeError):
    """Proteia is stopping: no upload, offer or accept starts."""


class TooManyPendingError(RuntimeError):
    """Taking this upload would exceed :data:`MAX_PENDING_FILES` or
    :data:`MAX_STAGED_BYTES`."""


class FileClaimedError(RuntimeError):
    """An offer names a file another offer already handed off."""


class HandoffNotFoundError(LookupError):
    """No pending hand-off has this id: it was imported or discarded."""


class HandoffClaimedError(RuntimeError):
    """An accept is importing this hand-off."""


class HandoffChangedError(RuntimeError):
    """A request names other files than the hand-off holds (more arrived);
    ``view`` is the hand-off as it is."""

    def __init__(self, message: str, view: HandoffView) -> None:
        super().__init__(message)
        self.view = view


def check_name(name: str) -> str:
    """``name`` if it can be an uploaded image's original name: an image suffix
    (``unsupported_image_type`` otherwise, checked first), and a plain file
    name as the model stores one, with no lone surrogate (``invalid_image``
    otherwise). It is metadata only: it never becomes a path."""
    if PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES:
        raise OperationError(
            ErrorCode.UNSUPPORTED_IMAGE_TYPE,
            f"{name!r} is not a supported image type ({', '.join(IMAGE_SUFFIXES)})",
        )
    try:
        name.encode("utf-8")
        _ORIGINAL_NAME.validate_python(name, strict=True)
    except (UnicodeEncodeError, ValidationError) as exc:
        raise OperationError(ErrorCode.INVALID_IMAGE, f"{name!r} is not a plain file name") from exc
    return name


def _shown(text: str, limit: int) -> str:
    """``text`` as a refused entry shows it: each character it never shows
    (:data:`_UNSHOWN`) made U+FFFD, and cut to ``limit`` characters, the last
    of them "…", if longer."""
    text = "".join("�" if unicodedata.category(c) in _UNSHOWN else c for c in text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class Refusal:
    """An argument a launch did not hand off, and why: its base name (never a
    path), a code (:data:`REFUSAL_CODES`) and a message."""

    name: str
    code: str
    message: str

    @classmethod
    def bounded(cls, name: str, code: str, message: str) -> Refusal:
        """The entry as the page may show it, whatever the launch sent: the name
        cut to :data:`MAX_REFUSED_NAME` characters and the message to
        :data:`MAX_REFUSED_MESSAGE`, control characters and the like made U+FFFD
        (:func:`_shown`), and an unknown code ``other``."""
        return cls(
            name=_shown(name, MAX_REFUSED_NAME),
            code=code if code in REFUSAL_CODES else "other",
            message=_shown(message, MAX_REFUSED_MESSAGE),
        )


@dataclass(frozen=True)
class StagedFile:
    """A copy the server made of an uploaded file, in the staging folder
    ``folder`` under a name the server made. The only kind of file the inbox
    deletes."""

    path: Path
    folder: Path

    def open(self) -> BinaryIO:
        return self.path.open("rb")

    def delete(self) -> None:
        """Delete the copy, only if it lies directly in the staging folder under
        a name the server makes; a failure is logged, not raised."""
        if self.path.parent != self.folder or not _STAGED_NAME.fullmatch(self.path.name):
            _log.warning("not deleting %s: not a staged file", self.path)
            return
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            _log.warning("a staged file could not be deleted: %s", exc)


@dataclass(frozen=True)
class LocalFile:
    """A file named on the first launch's own command line, read where it is.
    It has no delete: nothing Proteia does removes a user's file."""

    path: Path

    def open(self) -> BinaryIO:
        return self.path.open("rb")


@dataclass
class PendingFile:
    """A file waiting to be imported. The page sees its ``file_id``, ``name``
    and ``size``, never its path. ``started`` is when its upload began (the
    request arrived, before its body was read), or when a local file was
    registered."""

    file_id: str
    name: str
    size: int
    started: float
    source: StagedFile | LocalFile
    stored: float = 0.0  # when its upload ended: expiry and settling count from then


@dataclass
class Upload:
    """An upload under way: its body goes to ``path`` in the staging folder;
    ``reserved`` is the room it holds in :data:`MAX_STAGED_BYTES`."""

    file_id: str
    name: str
    path: Path
    folder: Path
    started: float
    reserved: int

    def open(self) -> BinaryIO:
        """The file the body is written to, created now: owner-only, and never
        over an existing file; the staging folder is made first (private to
        the user) if it is not there yet."""

        def private(path: str, flags: int) -> int:
            return os.open(path, flags, 0o600)

        _private_dir(self.folder)
        return open(self.path, "xb", opener=private)  # the caller closes it


@dataclass(frozen=True)
class FileView:
    file_id: str
    name: str
    size: int


@dataclass(frozen=True)
class HandoffView:
    """A pending hand-off as the page lists it. ``more_may_arrive``: files may
    still join it (it is young, or an upload that may join it is under way or
    not yet offered). ``claimed``: an accept is importing it, so no other
    accept or discard takes it, and no file joins it; refused, that accept
    releases it as it was."""

    id: str
    kind: Literal["images", "notice"]
    files: tuple[FileView, ...]
    refused: tuple[Refusal, ...]
    more_refused: int
    more_may_arrive: bool
    claimed: bool


@dataclass
class Handoff:
    """Files offered together, and the arguments their launches refused. A
    ``notice`` holds refused entries only."""

    id: str
    kind: Literal["images", "notice"]
    created: float
    last_started: float
    files: list[PendingFile] = field(default_factory=list)
    refused: list[Refusal] = field(default_factory=list)
    more_refused: int = 0
    claimed: bool = False

    def add_refused(self, refused: Iterable[Refusal], more: int = 0) -> None:
        for entry in refused:
            if len(self.refused) < MAX_REFUSED:
                self.refused.append(entry)
            else:
                self.more_refused += 1
        self.more_refused += more


@dataclass(frozen=True)
class Offered:
    """What an offer did: the hand-off its files went to, whether that one was
    already pending (``merged``), and how many files and refused entries it
    holds now."""

    handoff_id: str
    merged: bool
    files: int
    refused: int


def _private_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(folder, 0o700)


def _new_id() -> str:
    return secrets.token_hex(8)


class Inbox:
    """The pending hand-offs, and the uploads not yet offered (see the module
    docstring). Thread-safe: one lock guards it, never held while a file is
    read, written or deleted.

    ``folder`` is the staging folder, given once (:meth:`place`); a launch
    gives it before it serves. ``clock`` measures the merge window and the
    expiry."""

    def __init__(self, folder: Path | None = None, *, clock: Clock = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._folder = folder
        self._clock = clock
        self._running: dict[str, Upload] = {}  # uploads under way, by file id
        self._staged: dict[str, PendingFile] = {}  # uploaded, not yet offered
        self._handoffs: dict[str, Handoff] = {}  # in the order they began
        self._stopping = False

    @property
    def folder(self) -> Path | None:
        return self._folder

    @property
    def stopping(self) -> bool:
        return self._stopping

    def place(self, folder: Path) -> None:
        """Use ``folder`` as the staging folder, made when the first upload
        comes (private to the user), and clear it of the staged files a crash
        left there: regular files directly in it named as the server names
        them; nothing else is followed or deleted. Called by the launch that
        holds the instance lock, before it serves."""
        leftovers = []
        if folder.is_dir():
            with os.scandir(folder) as entries:
                leftovers = [
                    Path(entry.path)
                    for entry in entries
                    if _STAGED_NAME.fullmatch(entry.name) and entry.is_file(follow_symlinks=False)
                ]
        for path in leftovers:
            StagedFile(path, folder).delete()
        with self._lock:
            self._folder = folder

    # --- Uploads ---

    def begin_upload(self, name: str, declared: int | None) -> Upload:
        """An upload of ``name`` starting now, of ``declared`` bytes if its
        request says; its room is held until it ends (:meth:`upload_stored`,
        :meth:`upload_failed`). :class:`StoppingError`, or
        :class:`TooManyPendingError` when one more file, or its bytes, would
        exceed the limits."""
        expired = self._expire()
        try:
            with self._lock:
                if self._stopping:
                    raise StoppingError("Proteia is stopping")
                folder = self._folder
                if folder is None:
                    raise RuntimeError("no staging folder: Inbox.place was not called")
                if self._pending_count() + 1 > MAX_PENDING_FILES:
                    raise TooManyPendingError(
                        f"{MAX_PENDING_FILES} images are waiting in Proteia: import or discard"
                        " them first"
                    )
                reserved = declared or 0
                if self._staged_bytes() + reserved > MAX_STAGED_BYTES:
                    raise TooManyPendingError(
                        "the images waiting in Proteia take all the room it keeps for them:"
                        " import or discard them first"
                    )
                upload = Upload(
                    file_id=_new_id(),
                    name=name,
                    path=folder / secrets.token_hex(16),
                    folder=folder,
                    started=self._clock(),
                    reserved=reserved,
                )
                self._running[upload.file_id] = upload
                return upload
        finally:
            _delete(expired)

    def make_room(self, upload: Upload, size: int) -> None:
        """Hold room for ``size`` bytes of ``upload`` (more than its request
        declared); :class:`TooManyPendingError` if the staged bytes would exceed
        :data:`MAX_STAGED_BYTES`."""
        with self._lock:
            if size <= upload.reserved:
                return
            if self._staged_bytes() - upload.reserved + size > MAX_STAGED_BYTES:
                raise TooManyPendingError(
                    "the images waiting in Proteia take all the room it keeps for them:"
                    " import or discard them first"
                )
            upload.reserved = size

    def upload_stored(self, upload: Upload, size: int) -> PendingFile:
        """``upload`` ended with ``size`` bytes stored: it waits for an offer.
        :class:`StoppingError` (and the caller deletes it, :meth:`upload_failed`)
        once Proteia is stopping."""
        with self._lock:
            if self._stopping:
                raise StoppingError("Proteia is stopping")
            self._running.pop(upload.file_id, None)
            pending = PendingFile(
                file_id=upload.file_id,
                name=upload.name,
                size=size,
                started=upload.started,
                source=StagedFile(upload.path, upload.folder),
                stored=self._clock(),
            )
            self._staged[pending.file_id] = pending
            return pending

    def upload_failed(self, upload: Upload) -> None:
        """``upload`` did not end: whatever it stored is deleted."""
        with self._lock:
            self._running.pop(upload.file_id, None)
        StagedFile(upload.path, upload.folder).delete()

    # --- Offers ---

    def offer(self, file_ids: Sequence[str], refused: Sequence[Refusal] = ()) -> Offered:
        """Hand off the uploaded files ``file_ids`` (in this order), with the
        arguments the launch refused (bounded, :meth:`Refusal.bounded`, and at
        most :data:`MAX_REFUSED` kept, the rest counted), as a hand-off, or
        into one pending (see the module docstring). ``UnknownIdError`` for a
        file id unknown or expired, :class:`FileClaimedError` for one already
        handed off, ``invalid_input`` for one given twice or an offer of
        nothing, :class:`StoppingError`; each changes nothing."""
        if not file_ids and not refused:
            raise OperationError(ErrorCode.INVALID_INPUT, "an offer needs files or refused entries")
        if len(set(file_ids)) != len(file_ids):
            raise OperationError(ErrorCode.INVALID_INPUT, "a file id is given twice")
        expired = self._expire()
        try:
            with self._lock:
                if self._stopping:
                    raise StoppingError("Proteia is stopping")
                offered = [self._handoff_id_of(file_id) for file_id in file_ids]
                taken = [file_id for file_id, where in zip(file_ids, offered, strict=True) if where]
                if taken:
                    raise FileClaimedError(f"already handed off: {', '.join(taken)}")
                unknown = [file_id for file_id in file_ids if file_id not in self._staged]
                if unknown:
                    raise UnknownIdError(
                        f"no uploaded file waits with the id {unknown[0]!r}: unknown, or expired"
                    )
                files = [self._staged.pop(file_id) for file_id in file_ids]
                return self._hand_off(files, refused)
        finally:
            _delete(expired)

    def add_local(
        self, files: Sequence[tuple[Path, str]], refused: Sequence[Refusal] = ()
    ) -> Offered | None:
        """Hand off files named on the server's own command line, as
        (path, name) pairs: each read where it is when imported
        (:class:`LocalFile`), never copied, never deleted. A file that cannot
        be read now is refused (``unreadable``). With neither files nor
        refused entries, nothing is handed off (None)."""
        pending: list[PendingFile] = []
        refused = list(refused)
        for path, name in files:
            try:
                size = os.stat(path).st_size
            except OSError as exc:
                refused.append(
                    Refusal.bounded(name, "unreadable", f"cannot be read: {exc.strerror or exc}")
                )
                continue
            started = self._clock()
            pending.append(PendingFile(_new_id(), name, size, started, LocalFile(path), started))
        if not pending and not refused:
            return None
        with self._lock:
            return self._hand_off(pending, refused)

    def _handoff_id_of(self, file_id: str) -> str | None:
        """The hand-off holding ``file_id``, if one does. Called with the lock held."""
        for handoff in self._handoffs.values():
            if any(file.file_id == file_id for file in handoff.files):
                return handoff.id
        return None

    def _hand_off(self, files: list[PendingFile], refused: Sequence[Refusal]) -> Offered:
        """Merge ``files`` and ``refused`` into a pending hand-off, or start
        one (several if the files are more than one may hold). Called with the
        lock held."""
        entries = [Refusal.bounded(r.name, r.code, r.message) for r in refused]
        kept, more = entries[:MAX_REFUSED], max(0, len(entries) - MAX_REFUSED)
        now = self._clock()
        window = MERGE_WINDOW_S
        target: Handoff | None = None
        if files:
            earliest = min(file.started for file in files)
            newest = self._newest("images")
            if (
                newest is not None
                and earliest - newest.last_started < window
                and len(newest.files) + len(files) <= MAX_HANDOFF_FILES
            ):
                target = newest
        else:
            for kind in ("images", "notice"):
                newest = self._newest(kind)
                if newest is not None and now - newest.last_started < window:
                    target = newest
                    break
        if target is not None:
            target.files.extend(files)
            if files:
                target.last_started = max(target.last_started, *(f.started for f in files))
            target.add_refused(kept, more)
            return Offered(target.id, True, len(target.files), self._refused_count(target))
        chunks = [
            files[start : start + MAX_HANDOFF_FILES]
            for start in range(0, len(files), MAX_HANDOFF_FILES)
        ] or [[]]
        first: Handoff | None = None
        for chunk in chunks:
            handoff = Handoff(
                id=_new_id(),
                kind="images" if chunk else "notice",
                created=now,
                last_started=max((f.started for f in chunk), default=now),
                files=list(chunk),
            )
            if first is None:
                handoff.add_refused(kept, more)
                first = handoff
            self._handoffs[handoff.id] = handoff
        assert first is not None
        return Offered(first.id, False, len(first.files), self._refused_count(first))

    def _newest(self, kind: str) -> Handoff | None:
        """The newest hand-off of ``kind`` no accept has claimed. Called with
        the lock held."""
        for handoff in reversed(self._handoffs.values()):
            if handoff.kind == kind and not handoff.claimed:
                return handoff
        return None

    @staticmethod
    def _refused_count(handoff: Handoff) -> int:
        return len(handoff.refused) + handoff.more_refused

    # --- The page ---

    def listing(self) -> list[HandoffView]:
        """The pending hand-offs, in the order they began: those an accept has
        claimed too, as claimed (:meth:`claim`), since it may release them."""
        with self._lock:
            return [self._view(handoff) for handoff in self._handoffs.values()]

    def _view(self, handoff: Handoff) -> HandoffView:
        """``handoff`` as the page lists it. Called with the lock held."""
        return HandoffView(
            id=handoff.id,
            kind=handoff.kind,
            files=tuple(FileView(f.file_id, f.name, f.size) for f in handoff.files),
            refused=tuple(handoff.refused),
            more_refused=handoff.more_refused,
            more_may_arrive=self._more_may_arrive(handoff),
            claimed=handoff.claimed,
        )

    def _more_may_arrive(self, handoff: Handoff) -> bool:
        """Whether files may still join ``handoff``: it is young, or an upload
        under way or not yet offered began within the merge window of its last
        (or before it); only the newest images hand-off with room takes files.
        A launch uploads its files one by one, then offers them at once, so an
        upload not yet offered counts only while an upload runs or one ended
        within the window: once none has for that long, the files left were
        left by launches that exited, and join nothing (they expire). Called
        with the lock held."""
        now, window = self._clock(), MERGE_WINDOW_S
        if handoff.kind == "notice":
            return now - handoff.last_started < window
        if handoff is not self._newest("images") or len(handoff.files) >= MAX_HANDOFF_FILES:
            return False
        if now - handoff.last_started < window:
            return True
        waiting = [u.started for u in self._running.values()]
        if self._running or any(now - f.stored < window for f in self._staged.values()):
            waiting += [f.started for f in self._staged.values()]
        return any(started - handoff.last_started < window for started in waiting)

    def claim(self, handoff_id: str, file_ids: Sequence[str]) -> Handoff:
        """Claim the hand-off ``handoff_id`` for an accept that lists its files
        as ``file_ids`` (in any order): no other accept or discard takes it
        (:class:`HandoffClaimedError`), the page lists it as claimed, and files
        offered later start another; until :meth:`release` or :meth:`finish`.
        :class:`HandoffNotFoundError`, :class:`HandoffClaimedError`,
        :class:`HandoffChangedError` if it holds other files,
        ``invalid_input`` for a notice (it has none to import),
        :class:`StoppingError`; each changes nothing."""
        with self._lock:
            if self._stopping:
                raise StoppingError("Proteia is stopping")
            handoff = self._pending(handoff_id)
            if handoff.kind == "notice":
                raise OperationError(
                    ErrorCode.INVALID_INPUT, "a notice has no images to import: discard it"
                )
            self._same_files(handoff, file_ids)
            handoff.claimed = True
            return handoff

    def release(self, handoff: Handoff) -> None:
        """An accept that claimed ``handoff`` changed nothing: it is pending
        again, and listed as it was before the claim."""
        with self._lock:
            handoff.claimed = False

    def finish(self, handoff: Handoff) -> None:
        """``handoff`` is done with (imported, or impossible to import): it is
        removed, and its staged files are deleted."""
        with self._lock:
            self._handoffs.pop(handoff.id, None)
        _delete(handoff.files)

    def discard(self, handoff_id: str, file_ids: Sequence[str], refused: int) -> None:
        """Discard the hand-off ``handoff_id`` as the page shows it: its files
        ``file_ids`` (in any order) and ``refused`` refused entries, counted as
        the page counts them (those listed and ``more_refused``). Its staged
        files are deleted; the originals are never touched.
        :class:`HandoffNotFoundError`, :class:`HandoffClaimedError`, and
        :class:`HandoffChangedError` if it holds other files or entries
        (nothing is deleted)."""
        with self._lock:
            handoff = self._pending(handoff_id)
            self._same_files(handoff, file_ids)
            if refused != self._refused_count(handoff):
                raise HandoffChangedError(
                    "more arguments were refused since this list was drawn", self._view(handoff)
                )
            del self._handoffs[handoff.id]
        _delete(handoff.files)

    def _pending(self, handoff_id: str) -> Handoff:
        """The hand-off ``handoff_id``, if no accept has claimed it. Called with
        the lock held."""
        handoff = self._handoffs.get(handoff_id)
        if handoff is None:
            raise HandoffNotFoundError(
                f"no images wait as {handoff_id!r}: they were imported or discarded"
            )
        if handoff.claimed:
            raise HandoffClaimedError("these images are being imported")
        return handoff

    def _same_files(self, handoff: Handoff, file_ids: Sequence[str]) -> None:
        """:class:`HandoffChangedError` unless ``file_ids`` are the hand-off's
        files, each once. Called with the lock held."""
        held = [file.file_id for file in handoff.files]
        if len(file_ids) != len(held) or set(file_ids) != set(held):
            raise HandoffChangedError(
                "the images waiting are not those listed: more arrived", self._view(handoff)
            )

    # --- Stopping ---

    def stop(self) -> None:
        """No upload, offer or accept starts from now on (:class:`StoppingError`),
        and an accept under way imports no further file."""
        with self._lock:
            self._stopping = True

    def close(self) -> None:
        """Delete every staged file: of uploads not offered, and of every
        hand-off. Called once no accept runs, after :meth:`stop`."""
        with self._lock:
            self._stopping = True
            files = [*self._staged.values()]
            files += [file for handoff in self._handoffs.values() for file in handoff.files]
            uploads = list(self._running.values())
            self._staged.clear()
            self._handoffs.clear()
        _delete(files)
        for upload in uploads:
            StagedFile(upload.path, upload.folder).delete()

    # --- Limits ---

    def _pending_count(self) -> int:
        """Files waiting: uploads under way, not yet offered, and in hand-offs.
        Called with the lock held."""
        in_handoffs = sum(len(handoff.files) for handoff in self._handoffs.values())
        return len(self._running) + len(self._staged) + in_handoffs

    def _staged_bytes(self) -> int:
        """Bytes staged or held for uploads under way. Called with the lock held."""
        files = [*self._staged.values()]
        files += [file for handoff in self._handoffs.values() for file in handoff.files]
        staged = sum(file.size for file in files if isinstance(file.source, StagedFile))
        return staged + sum(upload.reserved for upload in self._running.values())

    def _expire(self) -> list[PendingFile]:
        """Take the uploads no offer took within :data:`UPLOAD_EXPIRY_S` of
        their end; the caller deletes them, outside the lock."""
        with self._lock:
            now = self._clock()
            expired = [
                file for file in self._staged.values() if now - file.stored >= UPLOAD_EXPIRY_S
            ]
            for file in expired:
                del self._staged[file.file_id]
            return expired


def _delete(files: Iterable[PendingFile]) -> None:
    """Delete the staged ones of ``files``; a local file is never deleted."""
    for file in files:
        if isinstance(file.source, StagedFile):
            file.source.delete()
