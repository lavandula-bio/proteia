# SPDX-License-Identifier: Apache-2.0
"""The session log (#137): what every launch did, kept after its window closes.

The launcher calls :func:`setup` (through :func:`session`) as it starts. From
then on the app's log records go to two places:

* the console (standard error), which shows what it showed before there was a
  session log, as it showed it: uvicorn's warnings and errors as uvicorn writes
  them, after their level; any other warning or error as its message alone, as
  Python's last resort writes it; an error in a thread as Python reports it.
  What only the session log records (a refused request, the stack trace of an
  error answered 500, a line the launcher also prints) does not reach it; that
  the log file cannot be written does;
* the log file ``proteia.log`` in the ``logs`` folder of the per-user state
  folder (:func:`proteia.web.launch.state_dir`: ``%LOCALAPPDATA%\\Proteia\\logs``
  on Windows), which uninstalling keeps (ADR 0003). It is UTF-8 with LF line
  ends, one line per record: the local time with its offset from UTC, the process id, the level,
  the logger and the message (a line break in a message is written ``\\n``); an
  error's stack trace follows on lines that begin with ``  | ``.

Proteia's own records reach the file from INFO up, every other library's
(uvicorn's among them) and Python's warnings from WARNING up. Proteia logs every
committed operation with the params its log entry holds in ``project.json``
(which never hold pixel data), naming the project by its folder's name; every
refusal, with its code and message; the fallbacks it takes (a background
measured another way, a check that could not run); and every unexpected error,
with its stack trace. The log stays on the user's computer: Proteia sends it
nowhere.

The file holds no full path and no access token. As each record is written,

* the folders :func:`setup` was given, the home folder (``~``) and the temporary
  folder (``<temp>``) are replaced by their placeholders wherever a message
  spells them, with either slash or doubled backslashes (as a Python ``repr``
  or JSON writes a path), and in any case on Windows: the launcher gives the
  projects root (``<projects>``, so a project's path reads
  ``<projects>\\<its name>``) and the state folder (``<state>``);
* every token given to :func:`conceal`, and whatever is written as a token (in a
  URL fragment ``#token=``, after ``Bearer``, or as a JSON ``"token"`` field), is
  replaced by ``<hidden>``; the console gets this too;
* a message longer than :data:`MAX_MESSAGE` characters is cut there.

What anyone who can reach the loopback port can make Proteia log (a request
refused before its token is checked) cannot push the history out: such a line
gives at most :data:`MAX_LOGGED_PATH` characters of the request's path
(:func:`shorten`), and at most :data:`REFUSALS_LOGGED` of them a minute are
written one by one (:class:`Throttle`); the next one written says how many
were left out.

Rotation and retention: when the next record would take ``proteia.log`` past
:data:`MAX_BYTES` (2 MiB), it becomes ``proteia.log.1``, each older file moves
up one number, and the file past :data:`BACKUPS` (``proteia.log.9``) is deleted.
So the folder holds at most 10 files, 20 MiB, whatever the number of sessions;
files are not deleted by age. Every launch appends to ``proteia.log``: a
second launch, which opens the running one, or hands it the images named on
its command line (:mod:`proteia.web.launch`), writes its few lines there too,
so the process id tells two processes' lines apart. A
rotation moves every file to a free name and deletes the oldest only once all
have moved, so a file another program holds open (Windows can then neither
rename it nor replace it), ``proteia.log`` or a rotated one, stops the whole
rotation: whatever moved is moved back, the records go on into
``proteia.log``, and the next one tries again, with no file lost.

A log that cannot be written never stops the app. If the folder or the file
cannot be made, or a write fails later, the records go on to the console only,
and the console says so once. Nor does a temporary folder Python cannot find
(a full disk): there is then none to hide.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Final

from uvicorn.logging import DefaultFormatter

LOG_DIR: Final = "logs"  # in the per-user state folder
LOG_FILE: Final = "proteia.log"
MAX_BYTES: Final = 2 * 1024 * 1024  # a file is rotated before it passes this size
BACKUPS: Final = 9  # rotated files kept: proteia.log.1 (newest) to proteia.log.9
MAX_MESSAGE: Final = 20_000  # characters of one message written to the file
MAX_LOGGED_PATH: Final = 200  # characters of a request's path in a refusal's line
REFUSALS_LOGGED: Final = 20  # refusals of one kind written one by one a minute
HIDDEN: Final = "<hidden>"
CONTINUATION: Final = "  | "  # begins every line of a record after its first
# ``extra`` for a record the console must not show (the launcher prints its text,
# or the console never showed it): the file only.
FILE_ONLY: Final = {"proteia_file_only": True}
_LOGGED: Final = "_proteia_logged"  # set on an exception logged with its stack trace

_log = logging.getLogger(__name__)

# Anything written as a token: its value, after the prefix group.
_TOKEN_PATTERNS: Final = (
    re.compile(r"(#token=)[^\s&'\"<>]+"),
    re.compile(r"(\bBearer\s+)[^\s'\",;<>]+", re.IGNORECASE),
    re.compile(r"(\"token\"\s*:\s*\")[^\"]+"),
)
_SEPARATORS: Final = r"[\\/]+"  # one slash, either way, or doubled backslashes
# A folder's spelling ends at a separator, a quote, a space or punctuation, not
# within a longer name (C:\Users\ann is not the start of C:\Users\anna).
_FOLDER_END: Final = r"(?=[\\/'\"\s:;,)\]>|]|$)"


class Redactor:
    """What the log file and the console must not show, replaced as records are
    written: tokens (:meth:`conceal`) and, in the file, folders
    (:meth:`hide_folder`). Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._secrets: frozenset[str] = frozenset()
        # (pattern, placeholder), the longest folder first: a folder inside
        # another is replaced before it.
        self._folders: tuple[tuple[re.Pattern[str], str], ...] = ()
        self._spellings: dict[str, tuple[int, re.Pattern[str], str]] = {}

    def conceal(self, secret: str) -> None:
        """Replace ``secret`` by :data:`HIDDEN` wherever a record holds it. Text
        shorter than 16 characters is not a token and is ignored."""
        if len(secret) >= 16:
            with self._lock:
                self._secrets = self._secrets | {secret}

    def hide_folder(self, folder: Path, placeholder: str) -> None:
        """Replace ``folder`` by ``placeholder`` wherever a record written to the
        file spells it (see the module docstring). A drive or file system root
        alone is ignored: it would hide every path."""
        text = str(folder)
        parts = [part for part in re.split(_SEPARATORS, text) if part]
        if len(parts) < 2 and not (text[:1] in "\\/" and parts):
            return
        lead = _SEPARATORS if text[:1] in "\\/" else ""
        flags = re.IGNORECASE if os.name == "nt" else 0
        pattern = re.compile(
            lead + _SEPARATORS.join(re.escape(part) for part in parts) + _FOLDER_END, flags
        )
        key = pattern.pattern.casefold() if flags else pattern.pattern
        with self._lock:
            self._spellings[key] = (len(text), pattern, placeholder)
            ordered = sorted(self._spellings.values(), key=lambda item: -item[0])
            self._folders = tuple((pattern, placeholder) for _, pattern, placeholder in ordered)

    def clear(self) -> None:
        with self._lock:
            self._secrets, self._folders, self._spellings = frozenset(), (), {}

    def __call__(self, text: str, *, folders: bool = True) -> str:
        for secret in self._secrets:
            text = text.replace(secret, HIDDEN)
        for pattern in _TOKEN_PATTERNS:
            text = pattern.sub(rf"\g<1>{HIDDEN}", text)
        if folders:
            for pattern, placeholder in self._folders:
                text = pattern.sub(placeholder.replace("\\", "\\\\"), text)
        return text


_redactor = Redactor()


def conceal(secret: str) -> None:
    """Never write ``secret`` (an access token) to the log or the console
    (:meth:`Redactor.conceal`)."""
    _redactor.conceal(secret)


def hide_folder(folder: Path, placeholder: str) -> None:
    """Write ``folder`` to the log file as ``placeholder``
    (:meth:`Redactor.hide_folder`)."""
    _redactor.hide_folder(folder, placeholder)


def mark_logged(exc: BaseException) -> None:
    """Note that ``exc`` was logged with its stack trace, so a later record of it
    (uvicorn's "Exception in ASGI application") is not written to the file
    again; the console shows that one, as before."""
    with contextlib.suppress(AttributeError, TypeError):
        setattr(exc, _LOGGED, True)


def _not_logged_before(record: logging.LogRecord) -> bool:
    exc = record.exc_info[1] if record.exc_info else None
    return not getattr(exc, _LOGGED, False)


def shorten(text: str, limit: int = MAX_LOGGED_PATH) -> str:
    """``text`` (a request's path, which the client chooses) cut at ``limit``
    characters, saying how many more it had."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text) - limit} more characters)"


class Throttle:
    """At most ``limit`` records of one kind in each ``period`` (seconds, on
    ``clock``), for records anyone who can reach the port can cause. Thread-safe."""

    def __init__(
        self,
        limit: int = REFUSALS_LOGGED,
        period: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit, self.period, self.clock = limit, period, clock
        self._lock = threading.Lock()
        self._start: float | None = None
        self._taken = 0  # records in the current period, left out or not
        self._left_out = 0  # since the last one let through

    def admit(self) -> tuple[bool, int]:
        """Whether to write the next record, and how many were left out before
        it (to say so as it is written; 0 when it is not)."""
        with self._lock:
            now = self.clock()
            if self._start is None or now - self._start >= self.period:
                self._start, self._taken = now, 0
            self._taken += 1
            if self._taken > self.limit:
                self._left_out += 1
                return False, 0
            left_out, self._left_out = self._left_out, 0
            return True, left_out


class _FileFormatter(logging.Formatter):
    """One line per record, then its stack trace on continuation lines; every
    part redacted (:class:`Redactor`)."""

    def format(self, record: logging.LogRecord) -> str:
        message = _redactor(record.getMessage().strip())
        if len(message) > MAX_MESSAGE:
            cut = len(message) - MAX_MESSAGE
            message = f"{message[:MAX_MESSAGE]}... ({cut} more characters)"
        message = message.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
        when = datetime.fromtimestamp(record.created).astimezone()
        lines = [
            f"{when.isoformat(timespec='milliseconds')} [{record.process}]"
            f" {record.levelname} {record.name}: {message}"
        ]
        trace = ""
        if record.exc_info:
            trace = self.formatException(record.exc_info)
        if record.stack_info:
            trace = f"{trace}\n{self.formatStack(record.stack_info)}".strip("\n")
        lines.extend(CONTINUATION + line for line in _redactor(trace).splitlines())
        return "\n".join(lines)


class _ConsoleFormatter(logging.Formatter):
    """A record as the console showed it before the session log (see the module
    docstring): uvicorn's as uvicorn's own formatter writes them, every other one
    as its message (Python's last resort), a stack trace as Python prints one;
    tokens hidden, paths left as they are."""

    def __init__(self) -> None:
        super().__init__()  # the message alone
        # Colours only on a terminal; a build without a console has no stderr
        # (None), and uvicorn's own check would call isatty() on sys.stdout.
        colours = sys.stderr is not None and sys.stderr.isatty()
        self._uvicorn = DefaultFormatter("%(levelprefix)s %(message)s", use_colors=colours)

    def format(self, record: logging.LogRecord) -> str:
        if record.name == "uvicorn" or record.name.startswith("uvicorn."):
            text = self._uvicorn.format(record)
        else:
            text = super().format(record)
        # A Python warning's message ends its line itself (warnings.formatwarning).
        return _redactor(text.removesuffix("\n"), folders=False)


class _Console(logging.StreamHandler):
    """Standard error as it is when a record is written (the launcher or a test
    may replace it), or nothing when there is none (a build without a console)."""

    def __init__(self) -> None:
        logging.Handler.__init__(self, logging.WARNING)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    def emit(self, record: logging.LogRecord) -> None:
        if sys.stderr is None or getattr(record, "proteia_file_only", False):
            return
        super().emit(record)


class _LogFile(logging.FileHandler):
    """``proteia.log``, rotated past ``max_bytes`` with ``backups`` older files
    kept. A write that fails turns it off for the rest of the session, and the
    console says so once; a rotation another program blocks is tried again at
    the next record."""

    def __init__(self, path: Path, *, max_bytes: int, backups: int) -> None:
        super().__init__(path, mode="a", encoding="utf-8", errors="backslashreplace")
        self.max_bytes = max_bytes
        self.backups = backups
        self.failed = False

    def _open(self):
        # Lines end in LF on every system, so a file's size is what emit counts.
        return self._builtin_open(
            self.baseFilename,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            newline="\n",
        )

    def emit(self, record: logging.LogRecord) -> None:
        if self.failed:
            return
        try:
            text = self.format(record) + self.terminator
        except Exception:
            self.handleError(record)  # a bad logging call: reported as logging reports one
            return
        try:
            if self.stream is None:
                self.stream = self._open()
            size = len(text.encode("utf-8", "backslashreplace"))
            position = self.stream.tell()
            if position and position + size > self.max_bytes:
                self.rotate_files()
            self.stream.write(text)
            self.stream.flush()
        except Exception as exc:  # OSError, or ValueError on a stream closed under it
            self.failed = True
            with contextlib.suppress(Exception):
                if self.stream is not None:
                    self.stream.close()
            self.stream = None
            _log.warning(
                "Proteia cannot write its log file any more (%s); its messages now show"
                " only in this window.",
                exc,
            )

    def rotate_files(self) -> None:
        """``proteia.log`` becomes ``.1``, ``.1`` becomes ``.2`` and so on, the
        oldest past :attr:`backups` deleted; then a new ``proteia.log``.

        Every file moves to a free name: ``proteia.log`` is set aside first (if
        another program holds it open, nothing moves), the oldest is set aside
        to be deleted, then each other one moves up into the name the one above
        it left. If any move fails (another program holds that file open), the
        ones made are moved back and nothing is deleted: the records go on into
        ``proteia.log``. Only once every file has moved is the oldest deleted."""
        base = self.baseFilename
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        try:
            aside, doomed = f"{base}.rotating", f"{base}.deleting"
            moves = [(base, aside)]
            if self.backups > 0:
                oldest = f"{base}.{self.backups}"
                if os.path.exists(oldest):
                    moves.append((oldest, doomed))
                for number in range(self.backups - 1, 0, -1):
                    older = f"{base}.{number}"
                    if os.path.exists(older):
                        moves.append((older, f"{base}.{number + 1}"))
                moves.append((aside, f"{base}.1"))
            else:
                moves.append((aside, doomed))
            with contextlib.suppress(OSError):  # left by a rotation that could not delete it
                os.remove(doomed)
            if _move_all(moves):
                with contextlib.suppress(OSError):
                    os.remove(doomed)
        finally:
            self.stream = self._open()


def _move_all(moves: list[tuple[str, str]]) -> bool:
    """Rename each ``(source, target)`` in turn, each target a free name; if one
    fails, move back the ones made, and answer False."""
    made: list[tuple[str, str]] = []
    for source, target in moves:
        try:
            os.rename(source, target)
        except OSError:
            for moved, to in reversed(made):
                with contextlib.suppress(OSError):
                    os.rename(to, moved)
            return False
        made.append((source, target))
    return True


# A file a rotation set aside is left to it for this long: a rotation takes
# milliseconds, so one older than this belongs to a rotation that did not finish.
ASIDE_GRACE_S: Final = 60.0


def _prune(folder: Path, backups: int, *, now: float | None = None) -> None:
    """Delete the rotated files past ``backups`` (left by a version that kept
    more) and the files set aside by a rotation that did not finish: only once
    they are :data:`ASIDE_GRACE_S` old, since another process (the running
    instance, when a second launch sets up its log) may be rotating now."""
    now = time.time() if now is None else now
    for path in folder.glob(f"{LOG_FILE}.*"):
        suffix = path.name[len(LOG_FILE) + 1 :]
        try:
            if suffix in ("rotating", "deleting"):
                if now - path.stat().st_mtime < ASIDE_GRACE_S:
                    continue
            elif not (suffix.isdigit() and int(suffix) > backups):
                continue
            path.unlink()
        except OSError:
            continue


def _log_thread_error(args: threading.ExceptHookArgs) -> None:
    """An error no code caught in a thread: logged with its stack trace (the
    file only), then reported as before (the hook :func:`setup` replaced)."""
    if args.exc_type is not SystemExit:
        name = args.thread.name if args.thread is not None else "a thread"
        _log.error(
            "unexpected error in %s",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            extra=FILE_ONLY,
        )
    installed = _installed
    previous = installed.hook if installed is not None else threading.__excepthook__
    previous(args)  # type: ignore[operator]


class _Installed:
    """What :func:`setup` changed, for :func:`shutdown` to undo."""

    def __init__(self, handlers: list[logging.Handler], level: int, hook: object) -> None:
        self.handlers = handlers
        self.level = level
        self.hook = hook


_installed: _Installed | None = None


def setup(
    folder: Path,
    *,
    hidden: Mapping[Path, str] | None = None,
    max_bytes: int = MAX_BYTES,
    backups: int = BACKUPS,
) -> Path | None:
    """Send the app's log records to the console and to ``folder/proteia.log``
    (the folder is made, owner-only on POSIX); return the file's path, or None
    when it cannot be written (the console says so). ``hidden`` maps folders to
    the placeholders the file writes them as (:func:`hide_folder`), besides the
    home and temporary folders. Undone by :func:`shutdown`; a second call
    undoes the first."""
    global _installed
    shutdown()
    folders: dict[Path, str] = {}
    with contextlib.suppress(OSError):  # no usable temporary folder (a full disk)
        folders[Path(tempfile.gettempdir())] = "<temp>"
    folders.update(hidden or {})
    with contextlib.suppress(RuntimeError):  # no home folder can be found
        folders.setdefault(Path.home(), "~")
    for path, placeholder in folders.items():
        hide_folder(path, placeholder)
    console = _Console()
    console.setFormatter(_ConsoleFormatter())
    handlers: list[logging.Handler] = [console]
    file: _LogFile | None = None
    problem = ""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(folder, 0o700)
        _prune(folder, backups)
        file = _LogFile(folder / LOG_FILE, max_bytes=max_bytes, backups=backups)
    except OSError as exc:
        problem = str(exc)
    if file is not None:
        file.setLevel(logging.INFO)
        file.setFormatter(_FileFormatter())
        file.addFilter(_not_logged_before)
        handlers.append(file)
    root, own = logging.getLogger(), logging.getLogger("proteia")
    _installed = _Installed(handlers, own.level, threading.excepthook)
    for handler in handlers:
        root.addHandler(handler)
    own.setLevel(logging.INFO)
    logging.captureWarnings(True)
    threading.excepthook = _log_thread_error
    if file is None:
        _log.warning(
            "Proteia cannot write its log file in %s (%s); its messages show only in this window.",
            folder,
            problem,
        )
        return None
    return folder / LOG_FILE


def shutdown() -> None:
    """Undo :func:`setup`: close the file, and forget the folders and tokens."""
    global _installed
    installed, _installed = _installed, None
    if installed is None:
        return
    root = logging.getLogger()
    for handler in installed.handlers:
        root.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()
    logging.getLogger("proteia").setLevel(installed.level)
    logging.captureWarnings(False)
    threading.excepthook = installed.hook  # type: ignore[assignment]
    _redactor.clear()


@contextlib.contextmanager
def session(folder: Path, *, hidden: Mapping[Path, str] | None = None) -> Iterator[Path | None]:
    """:func:`setup` for the block, then :func:`shutdown`."""
    path = setup(folder, hidden=hidden)
    try:
        yield path
    finally:
        shutdown()
