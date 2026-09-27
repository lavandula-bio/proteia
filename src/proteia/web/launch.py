# SPDX-License-Identifier: Apache-2.0
"""Start the local web app: ``uv run proteia`` runs :func:`main` (ADR 0002).

A launch binds a listening socket to ``127.0.0.1`` on a port the operating system
assigns (port 0, read back, so no other process can take it in between; on
Windows with exclusive use of the address), and makes a new random token.

One instance runs per user. The running instance holds an exclusive operating
system lock on ``instance.lock`` in the per-user state folder (:func:`state_dir`);
the system releases it when the process ends, so a crash leaves no lock behind. A
launch that cannot take the lock waits for the running instance to answer and
opens it instead of starting another server.

The token never appears on a command line or in the console. The running instance
writes it to files only the current user can read, in the same folder:

* ``instance.json``: its port and token, for a second launch to find it;
* ``open-proteia.html``: a redirect page that sends the browser to
  ``http://127.0.0.1:<port>/#token=<token>``. The launcher opens this file, not
  the URL, and the token rides in the URL fragment, which the browser never sends
  to a server. The page keeps it for its tab and sends it in a request header.

Both files are removed as soon as the server begins to stop (Quit on the page,
Ctrl+C, or a termination signal), before it closes its listening socket, and
again, if still there, before the lock is released. So a launch meanwhile finds
no instance and waits for the lock, rather than send the token, or images, to a
port that another program may take next. A new instance removes the ones a
crash left as soon as it holds the lock, before it binds. (A launch that opens
the running instance just as it stops can write the redirect page again; its
token opens nothing, and the next launch replaces it.) On POSIX the folder is
0700 and the files 0600; on Windows the per-user local application-data folder
is readable only by its owner.

Images to import can be named on the command line: ``proteia [--] PATH...``
(:mod:`proteia.web.cli`, whose checks run in the launching process before
anything starts, relative paths against its own working folder; ``proteia
--help`` gives the exit codes). Polarity has no default, so a launch never
imports an image itself: it hands the files to the running instance
(:mod:`proteia.web.handoff`), whose page asks how to import them, into a new
project named after the first, or discards them.

* A launch that starts the instance registers the files named on its own
  command line in-process, before it opens the browser (:meth:`Instance.take`):
  the server reads them where they are when the page imports them, and never
  copies or deletes them. No path comes from an HTTP client.
* A launch that finds Proteia running first checks that its ``/api/status``
  answer says it takes files (``handoff``; an older Proteia does not: the launch
  opens it, hands nothing over, and says so), then uploads each file's bytes
  with its name, never its path (``POST /api/incoming``), once it says it has
  room for them (``GET /api/incoming/room``; a file it has none for is refused
  unsent, as is one whose upload is cut short), and offers them (``POST
  /api/handoffs``). Uploaded images are staged in ``incoming/`` in the
  state folder, under names the server makes; the instance deletes them once
  imported or discarded and when it stops, and a new one deletes what a crash
  left there once it holds the lock.
* Launches run at once (a selection opened in Explorer, one process per file)
  join one hand-off. A launch whose files joined one that another launch handed
  off a moment ago opens no browser tab: that launch opened one on it.

A path that cannot be imported (a missing file, a folder, another type) is
listed on the console, with its full path and why, and on the page, with its
name; the others proceed. Exit status: 0 opened, and every path taken; 1 not
opened (the running instance does not answer), or the files not handed over;
2 wrong arguments, and nothing done; 3 opened, but some paths not taken. A
launch that starts the instance returns when it stops.

The Windows installer's "Open with Proteia" (#54) runs this command line:
``"<install folder>\\Proteia.exe" "%1"``, where ``%1`` is the path of the file
opened, quoted (an unquoted ``%1`` can be an 8.3 short name, and a space would
split it), and no option. Explorer may run one such command per file of a
selection, which join one hand-off as above.

The connections a launch makes all go to the running instance's loopback port,
without any proxy: the check that it answers, and a second launch's hand-off
of its files.

Every launch writes its session log (:mod:`proteia.web.logs`) to the ``logs``
folder there: when it started and stopped, whether it served or opened the
running instance, and how many images it handed over (never their paths). The
token is concealed from it as soon as the launch makes or reads one, so
neither the log nor the console ever shows it, nor the redirect page's
address. The diagnostic files the page writes for bug reports, which take the
newest session log files, go in the ``diagnostics`` folder there
(:mod:`proteia.web.diagnostics`). The notices the user dismissed for good are
recorded in ``notices.json`` there (:mod:`proteia.web.cloudsync`).
"""

from __future__ import annotations

import contextlib
import html
import http.client
import json
import logging
import os
import platform
import secrets
import signal
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Final
from urllib.parse import quote

import uvicorn

import proteia
from proteia.core.storage import write_atomic
from proteia.web import cli, logs, projects
from proteia.web.api import UnsavedChangesError, Workspace
from proteia.web.cli import CommandLine, ImageFile, RefusedPath
from proteia.web.handoff import INCOMING_DIR
from proteia.web.server import APP_ID, HANDOFF, HOST, TOKEN_PATTERN, create_app

if os.name == "nt":
    import msvcrt

    def _lock(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # the first byte; raises OSError if held

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # per open file, not per process

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


LOCK_FILE: Final = "instance.lock"
INSTANCE_FILE: Final = "instance.json"
REDIRECT_FILE: Final = "open-proteia.html"
TOKEN_BYTES: Final = 32  # 43 URL-safe characters
PROBE_TIMEOUT: Final = 2.0  # seconds per check of the running instance
STARTUP_WAIT: Final = 15.0  # seconds a second launch waits for it to answer
HANDOFF_TIMEOUT: Final = 30.0  # seconds a hand-off's connection may wait on each send or read
UPLOAD_BLOCK: Final = 1024 * 1024  # an image is sent in pieces this large
_ANSWER_BYTES: Final = 64 * 1024  # of an answer to a hand-off request, read at most
_STOP_SIGNALS: Final = tuple(
    getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGBREAK") if hasattr(signal, name)
)

Opener = Callable[[str], object]  # webbrowser.open's shape: takes a URL

_log = logging.getLogger(__name__)


class NotRespondingError(RuntimeError):
    """Another instance holds the lock but did not answer within the wait."""


class HandoffError(RuntimeError):
    """The running instance answered, but a launch's files were not handed to
    it: it is an older version that takes none (it was ``opened`` in the browser
    all the same), it is stopping, or the connection failed. Files already
    uploaded expire there, unoffered."""

    def __init__(self, message: str, *, opened: bool = False) -> None:
        super().__init__(message)
        self.opened = opened


@dataclass(frozen=True)
class Opened:
    """What a launch did when Proteia was running already: it opened it in the
    browser, and handed it ``files`` of its command line, if any. ``merged``:
    they joined the images another launch handed off a moment ago, whose tab
    shows them, so no tab was opened; ``waiting``: the files in that hand-off
    now. ``refused``: the paths not taken, by the command line's checks or by
    the running instance."""

    files: tuple[ImageFile, ...] = ()
    refused: tuple[RefusedPath, ...] = ()
    merged: bool = False
    waiting: int = 0


def state_dir() -> Path:
    """The per-user folder that holds the lock, instance and redirect files."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "Proteia"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Proteia"
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base and Path(base).is_absolute() else Path.home() / ".local" / "state"
    return root / "proteia"


def _private_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(folder, 0o700)


class InstanceLock:
    """The exclusive lock on ``instance.lock`` that the running instance holds."""

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd

    @classmethod
    def acquire(cls, folder: Path) -> InstanceLock | None:
        """The lock, or None while another instance (in any process) holds it."""
        fd = os.open(folder / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _lock(fd)
        except OSError:
            os.close(fd)
            return None
        return cls(fd)

    def release(self) -> None:
        if self._fd is None:
            return
        with contextlib.suppress(OSError):
            _unlock(self._fd)
        os.close(self._fd)
        self._fd = None


@dataclass(frozen=True)
class InstanceInfo:
    """What ``instance.json`` records about the running instance."""

    pid: int
    port: int
    token: str


def read_instance(folder: Path) -> InstanceInfo | None:
    """The instance file's contents, or None when it is missing or unreadable."""
    try:
        doc = json.loads((folder / INSTANCE_FILE).read_text(encoding="utf-8"))
        pid, port, token = doc["pid"], doc["port"], doc["token"]
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if not (
        type(pid) is int
        and type(port) is int
        and 0 < port < 65536
        and isinstance(token, str)
        and TOKEN_PATTERN.fullmatch(token)
    ):
        return None
    return InstanceInfo(pid=pid, port=port, token=token)


def probe(info: InstanceInfo, *, timeout: float = PROBE_TIMEOUT) -> dict[str, Any] | None:
    """The ``/api/status`` answer of the Proteia instance on ``info.port``, if one
    answers there and admits its token (None otherwise): ``{app, version,
    handoff}``, ``handoff`` only from one that takes files from a launch.

    A direct loopback connection: ``http.client`` uses no proxy settings.
    """
    conn = http.client.HTTPConnection(HOST, info.port, timeout=timeout)
    try:
        conn.request("GET", "/api/status", headers={"Authorization": f"Bearer {info.token}"})
        response = conn.getresponse()
        body = response.read(4096)
        status = json.loads(body)
        if response.status == 200 and isinstance(status, dict) and status.get("app") == APP_ID:
            return status
        return None
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        conn.close()


def app_url(port: int, token: str) -> str:
    """The app's address with the token in the fragment."""
    return f"http://{HOST}:{port}/#token={token}"


def _redirect_page(url: str) -> bytes:
    href = html.escape(url, quote=True)
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="referrer" content="no-referrer">\n'
        f'<meta http-equiv="refresh" content="0; url={href}">\n'
        "<title>Opening Proteia</title></head>\n"
        f'<body><p>Opening Proteia. If nothing happens, <a href="{href}">open it here</a>.'
        "</p></body></html>\n"
    ).encode()


def open_in_browser(folder: Path, port: int, token: str, opener: Opener) -> Path:
    """Write the redirect page for ``port`` and ``token`` and open it; return its path."""
    path = folder / REDIRECT_FILE
    write_atomic(path, _redirect_page(app_url(port, token)), private=True)
    opener(path.as_uri())
    return path


def bind_loopback() -> socket.socket:
    """A socket listening on ``127.0.0.1`` at a port the system assigns.

    Listening at once lets the browser connect before the server loop runs: the
    connection waits in the backlog.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: no port sharing
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((HOST, 0))
        sock.listen(128)
    except BaseException:
        sock.close()
        raise
    return sock


class _Server(uvicorn.Server):
    """uvicorn's server, which calls ``on_exit`` first when a stop signal comes
    while it serves from the main thread (it then stops as its own handler
    would), as :meth:`Instance.stop` does before it stops the server."""

    def __init__(self, config: uvicorn.Config, on_exit: Callable[[], None]) -> None:
        super().__init__(config)
        self._on_exit = on_exit

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        self._on_exit()
        super().handle_exit(sig, frame)


def _server(app: object, on_exit: Callable[[], None]) -> uvicorn.Server:
    config = uvicorn.Config(
        app,
        http="h11",
        ws="none",
        lifespan="off",
        # uvicorn's records go to the handlers the session log set up (or, without
        # one, Python's last resort on standard error), not to handlers of its own.
        log_config=None,
        log_level="warning",
        access_log=False,  # the log would name every request
        server_header=False,
        timeout_graceful_shutdown=2,
    )
    return _Server(config, on_exit)


@contextlib.contextmanager
def _stop_on_signals(stop: Callable[[], None]) -> Iterator[None]:
    """While serving from the main thread, a stop signal only calls ``stop``.

    uvicorn catches SIGINT and SIGTERM (and SIGBREAK on Windows), shuts down, then
    raises the signal again with the handler it found. With the default handlers
    that raise would end the process (SIGTERM) or raise ``KeyboardInterrupt``
    (SIGINT) before the cleanup; with this one it is a no-op.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handle(signum: int, frame: object) -> None:
        stop()

    previous = {sig: signal.signal(sig, handle) for sig in _STOP_SIGNALS}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)


class Instance:
    """A started server that holds the instance lock. :meth:`serve` blocks until it
    stops; :meth:`stop` (or Quit on the page, or a stop signal when served from the
    main thread) stops it."""

    def __init__(
        self,
        folder: Path,
        sock: socket.socket,
        token: str,
        lock: InstanceLock,
        workspace: Workspace | None = None,
    ) -> None:
        self.folder = folder
        self.sock = sock
        self.token = token
        self._lock = lock
        self.port: int = sock.getsockname()[1]
        self.redirect_path = folder / REDIRECT_FILE
        app = create_app(token=token, port=self.port, on_quit=self.stop, workspace=workspace)
        self.workspace: Workspace = app.app.state.workspace
        if self.workspace.inbox.folder is None:  # held by this lock: no other instance uses it
            self.workspace.inbox.place(folder / INCOMING_DIR)
        if self.workspace.state is None:  # the session log's, and the diagnostic files'
            self.workspace.state = folder
        self.server = _server(app, on_exit=self._remove_files)
        self.taken = 0  # the files its own command line handed to the page (take)
        self.unread: tuple[RefusedPath, ...] = ()  # those it could no longer read (take)

    def take(self, command_line: CommandLine) -> None:
        """Hand the paths named on this process's own command line to its page,
        in-process, before it serves (:meth:`~proteia.web.handoff.Inbox.add_local`):
        the files, to be read where they are when the page imports them (never
        copied, never deleted), and the paths refused, for the page to list. It
        only reads each file's size, so it takes no time: launches run at the
        same moment wait for this one to answer (:data:`STARTUP_WAIT`).
        :attr:`taken` counts the files handed over: one that can no longer be
        read is refused instead, and the page says so; :attr:`unread` holds
        those, with their full paths, for the console."""
        if not command_line.paths:
            return
        inbox = self.workspace.inbox
        before = sum(len(view.files) for view in inbox.listing())
        inbox.add_local(
            [(file.location, file.name) for file in command_line.files],
            [path.refusal() for path in command_line.refused],
        )
        self.taken = sum(len(view.files) for view in inbox.listing()) - before
        if self.taken < len(command_line.files):  # it refused those it could not read
            gone = (_unreadable(file) for file in command_line.files)
            self.unread = tuple(path for path in gone if path is not None)

    def serve(self) -> None:
        """Serve until stopped, save what an autosave could not, close the open
        project (deleting the image files only its undo history kept), then
        :meth:`close`."""
        try:
            with _stop_on_signals(self.stop):
                self.server.run(sockets=[self.sock])
        finally:
            try:
                try:
                    self.workspace.flush()
                except UnsavedChangesError as exc:
                    print(f"Proteia stopped, but {exc}", file=sys.stderr)
                    _log.warning("Proteia stopped, but %s", exc, extra=logs.FILE_ONLY)
                self.workspace.close()
            finally:
                self.close()

    def stop(self) -> None:
        """Remove the files that name this instance, then stop the server: it
        closes its listening socket as it begins to stop, and a launch that
        read them after that could reach whatever takes the port next."""
        self._remove_files()
        self.server.should_exit = True

    def close(self) -> None:
        """Release the socket, remove the files that name this instance (if
        :meth:`stop` did not), then release the lock (so no new instance writes
        its files before that)."""
        self.sock.close()
        self._remove_files()
        self._lock.release()

    def _remove_files(self) -> None:
        """Remove ``instance.json`` and the redirect page if they hold this
        instance's token (a launch may have written the page again)."""
        for path in (self.folder / INSTANCE_FILE, self.redirect_path):
            with contextlib.suppress(OSError):
                if self.token in path.read_text(encoding="utf-8"):
                    path.unlink()


def _unreadable(file: ImageFile) -> RefusedPath | None:
    """``file`` refused if it can no longer be read where it is, as
    :meth:`~proteia.web.handoff.Inbox.add_local` refuses it; else None."""
    try:
        os.stat(file.location)
    except OSError as exc:
        return file.refused("unreadable", f"cannot be read: {cli.reason(exc)}")
    return None


def _remove_leftovers(folder: Path) -> None:
    """Remove the instance file and redirect page a crashed instance left:
    called once the lock is held, so no running instance has them."""
    for name in (INSTANCE_FILE, REDIRECT_FILE):
        with contextlib.suppress(OSError):
            (folder / name).unlink(missing_ok=True)


def _open_running(
    folder: Path, opener: Opener, wait: float, command_line: CommandLine
) -> InstanceLock | Opened:
    """Another process holds the lock: wait until its instance answers, then
    open it and hand it the command line's files (:func:`_hand_off`), or return
    the lock if that instance stops meanwhile (it was quitting)."""
    deadline = time.monotonic() + wait
    while True:
        info = read_instance(folder)
        if info is not None:
            logs.conceal(info.token)
            status = probe(info)
            if status is not None:
                return _hand_off(folder, info, status, opener, command_line)
        lock = InstanceLock.acquire(folder)
        if lock is not None:
            return lock
        if time.monotonic() >= deadline:
            raise NotRespondingError("Proteia is already running but does not respond")
        time.sleep(0.2)


def _hand_off(
    folder: Path,
    info: InstanceInfo,
    status: dict[str, Any],
    opener: Opener,
    command_line: CommandLine,
) -> Opened:
    """Open the running instance ``info``, which answered ``status``, in the
    browser, and hand it the command line's paths: each file's bytes, uploaded
    one by one (a file it refuses is refused, and the others go on), then one
    offer of them all, with the paths refused. No tab is opened when the offer
    joined a hand-off another launch made a moment ago (it opened one), nor when
    the hand-off fails (:class:`HandoffError`)."""
    if not command_line.paths:
        open_in_browser(folder, info.port, info.token, opener)
        return Opened()
    if status.get("handoff") != HANDOFF:
        open_in_browser(folder, info.port, info.token, opener)
        which = "an older" if "handoff" not in status else "another"
        raise HandoffError(
            f"The running Proteia is {which} version and cannot take files; quit it and try again.",
            opened=True,
        )
    sent: list[ImageFile] = []
    file_ids: list[str] = []
    refused = list(command_line.refused)
    for file in command_line.files:
        uploaded = _upload(info, file)
        if isinstance(uploaded, RefusedPath):
            refused.append(uploaded)
        else:
            sent.append(file)
            file_ids.append(uploaded)
    offer = {"files": file_ids, "refused": [asdict(path.refusal()) for path in refused]}
    code, answer = _send(
        info,
        "POST",
        "/api/handoffs",
        json.dumps(offer).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    if code != 201:
        raise HandoffError(f"Proteia did not take the files: {_message(code, answer)}")
    merged = answer.get("merged") is True
    if not merged:
        open_in_browser(folder, info.port, info.token, opener)
    waiting = answer.get("files")
    return Opened(
        files=tuple(sent),
        refused=tuple(refused),
        merged=merged,
        waiting=waiting if type(waiting) is int else len(sent),
    )


def _upload(info: InstanceInfo, file: ImageFile) -> str | RefusedPath:
    """Upload ``file``'s bytes, with its name, to the running instance, once it
    says it has room for them (:func:`_room`): the id it answers, or the file
    refused (by the instance, with its code and reason, or because it can no
    longer be read). An upload whose connection is reset or closed before its
    answer can be read (:class:`_CutShortError`) is refused too, with the
    reason the instance gives when asked again, or else as cut short; the
    instance deletes what it stored of it. :class:`HandoffError` if the
    instance is stopping, or a connection to it fails."""
    try:
        stream = file.open()
    except OSError as exc:
        return file.refused("unreadable", f"cannot be read: {cli.reason(exc)}")
    with stream:
        size = os.fstat(stream.fileno()).st_size
        refused = _room(info, file, size)
        if refused is not None:
            return refused
        headers = {"Content-Type": "application/octet-stream", "Content-Length": str(size)}
        path = f"/api/incoming?name={quote(file.name, safe='')}"
        try:
            code, answer = _send(info, "POST", path, stream, headers)
        except _CutShortError as exc:
            _log.info("an upload was cut short: %s", exc.reason, extra=logs.FILE_ONLY)
            refused = _room(info, file, size)  # why, if it says; and whether it still answers
            return refused or file.refused("other", f"the upload was cut short: {exc.reason}")
    file_id = answer.get("file_id")
    if code == 201 and isinstance(file_id, str):
        return file_id
    return _refused(file, code, answer)


def _room(info: InstanceInfo, file: ImageFile, size: int) -> RefusedPath | None:
    """Whether the running instance would take ``file``, of ``size`` bytes, now
    (``GET /api/incoming/room``, with no body): None if so, or the file
    refused as its upload would be. :class:`HandoffError` if the instance is
    stopping, or the connection fails."""
    path = f"/api/incoming/room?name={quote(file.name, safe='')}&size={size}"
    code, answer = _send(info, "GET", path, None, {})
    return None if code == 204 else _refused(file, code, answer)


def _refused(file: ImageFile, code: int, answer: dict[str, Any]) -> RefusedPath:
    """``file`` refused by the running instance's answer ``code``, ``answer``,
    with its code and reason; :class:`HandoffError` if it is stopping."""
    if code == 409 and answer.get("code") == "stopping":
        raise HandoffError("Proteia is stopping; the files were not handed to it.")
    kind = answer.get("code")
    return file.refused(kind if isinstance(kind, str) else "other", _message(code, answer))


class _CutShortError(HandoffError):
    """A request's connection was reset or closed once it had connected, and
    no answer could be read: the instance may have answered an upload it
    refused before reading its body whole, then closed the connection when
    the rest stopped arriving (on Windows the reset drops an answer not yet
    read), or it may have gone. ``reason`` is what the system said."""

    def __init__(self, reason: str) -> None:
        super().__init__(_connection_failed(reason))
        self.reason = reason


def _connection_failed(reason: str) -> str:
    return (
        "The connection to the running Proteia failed, and the files were not handed to it:"
        f" {reason}"
    )


def _send(
    info: InstanceInfo, method: str, path: str, body: Any, headers: dict[str, str]
) -> tuple[int, dict[str, Any]]:
    """One request to the running instance, with its token, on a direct loopback
    connection (no proxy): the status and the answer's JSON object (empty if it
    has none). :class:`HandoffError` if the connection fails; a
    :class:`_CutShortError` if it was reset or closed once connected and no
    answer can be read. A request refused before its body was sent whole (the
    instance answers a refused upload at once) is answered all the same, if the
    answer can still be read."""
    conn = http.client.HTTPConnection(
        HOST, info.port, timeout=HANDOFF_TIMEOUT, blocksize=UPLOAD_BLOCK
    )
    try:
        cut_short: ConnectionError | None = None
        try:
            conn.request(
                method,
                path,
                body=body,
                headers={"Authorization": f"Bearer {info.token}", **headers},
            )
        except ConnectionError as exc:
            if conn.sock is None:  # it never connected
                raise
            cut_short = exc  # it may have answered, then closed the connection
        try:
            response = conn.getresponse()
            data = response.read(_ANSWER_BYTES)
        except (OSError, http.client.HTTPException) as exc:
            failed = exc if cut_short is None else cut_short
            if isinstance(failed, ConnectionError):  # connected, then reset or closed
                raise _CutShortError(cli.reason(failed)) from failed
            raise
    except (OSError, http.client.HTTPException) as exc:
        raise HandoffError(_connection_failed(cli.reason(exc))) from exc
    finally:
        conn.close()
    try:
        answer = json.loads(data)
    except ValueError:
        answer = None
    return response.status, answer if isinstance(answer, dict) else {}


def _message(code: int, answer: dict[str, Any]) -> str:
    """What an answer that refused a request says (its ``message``, or the
    guard's ``detail``), or its status."""
    for key in ("message", "detail"):
        text = answer.get(key)
        if isinstance(text, str) and text:
            return text
    return f"answered {code}"


def start(
    *,
    folder: Path | None = None,
    opener: Opener = webbrowser.open,
    wait: float = STARTUP_WAIT,
    workspace: Workspace | None = None,
    command_line: CommandLine | None = None,
) -> Instance | Opened:
    """Open the running instance in the browser and hand it the files of
    ``command_line`` (:class:`Opened`), or start one and open it with them.

    A new instance holds the lock, has removed the files a crashed one left, is
    bound and listening, has written its files and taken the command line's
    (:meth:`Instance.take`) before the browser opens; call :meth:`Instance.serve`
    to serve it. When another instance holds the lock, this waits up to ``wait``
    seconds for it to answer, or to stop and free the lock; then
    ``NotRespondingError``. If it answers but the files cannot be handed to it,
    :class:`HandoffError`. A second server is never started.
    """
    command_line = CommandLine() if command_line is None else command_line
    folder = state_dir() if folder is None else folder
    _private_dir(folder)
    lock = InstanceLock.acquire(folder)
    if lock is None:
        found = _open_running(folder, opener, wait, command_line)
        if isinstance(found, Opened):
            return found
        lock = found
    instance: Instance | None = None
    try:
        _remove_leftovers(folder)
        sock = bind_loopback()
        try:
            token = secrets.token_urlsafe(TOKEN_BYTES)
            logs.conceal(token)
            instance = Instance(folder, sock, token, lock, workspace)
        except BaseException:
            sock.close()
            raise
        info = {"pid": os.getpid(), "port": instance.port, "token": instance.token}
        write_atomic(folder / INSTANCE_FILE, json.dumps(info).encode("utf-8"), private=True)
        instance.take(command_line)
        open_in_browser(folder, instance.port, instance.token, opener)
    except BaseException:
        if instance is None:
            lock.release()
        else:
            instance.close()
        raise
    return instance


def _tolerant_console() -> None:
    """Escape what the console encoding cannot show (a path with µ in a code-page
    console) rather than fail."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")


def main(argv: Sequence[str] | None = None) -> int:
    """The ``proteia`` console command, with the arguments ``argv``
    (``sys.argv[1:]`` when None): ``proteia [--] [PATH ...]``
    (:mod:`proteia.web.cli`). Returns the exit status: 0 opened, and every path
    taken; 1 not opened (the running instance does not answer), or the files not
    handed over; 2 wrong arguments, and nothing started (0 after ``--help`` or
    ``--version``); 3 opened, but some paths not taken, each listed on standard
    error and on the page. A launch that serves returns when the server stops.
    ``proteia --self-test`` checks the installation instead
    (:mod:`proteia.selftest`)."""
    args = sys.argv[1:] if argv is None else list(argv)
    if args[:1] == ["--self-test"]:
        from proteia import selftest

        return selftest.main(args[1:])
    _tolerant_console()
    try:
        command_line = cli.parse(args)
    except SystemExit as exc:  # a usage error, --help or --version: nothing starts
        return exc.code if isinstance(exc.code, int) else 0
    folder = state_dir()
    hidden = {folder: "<state>", projects.projects_root(): "<projects>"}
    with logs.session(folder / logs.LOG_DIR, hidden=hidden):
        _log.info(
            "session started: Proteia %s, Python %s, %s",
            proteia.__version__,
            platform.python_version(),
            platform.platform(),
        )
        try:
            return _run(command_line)
        except Exception:
            # The file only: Python prints the stack trace to the console as it exits.
            _log.exception("Proteia stopped on an unexpected error", extra=logs.FILE_ONLY)
            raise
        finally:
            _log.info("session ended")


def _run(command_line: CommandLine) -> int:
    """:func:`main` once the session log is set up. The log gets how many paths
    were given and taken, never the paths."""
    if command_line.paths:
        codes = sorted({path.code for path in command_line.refused})
        _log.info(
            "command line: %d images, %d paths not taken%s",
            len(command_line.files),
            len(command_line.refused),
            f" ({', '.join(codes)})" if codes else "",
        )
    try:
        started = start(command_line=command_line)
    except NotRespondingError:
        message = (
            "Proteia is already running but does not respond. Try again in a moment,"
            " or end the other Proteia process."
        )
        _log.error(message, extra=logs.FILE_ONLY)
        print(message)
        _print_refused(command_line.refused)
        return 1
    except HandoffError as exc:
        if exc.opened:
            print("Proteia is already running; it has been opened in your browser.")
        _log.error("%s", exc, extra=logs.FILE_ONLY)
        print(exc, file=sys.stderr)
        _print_refused(command_line.refused)
        return 1
    if isinstance(started, Opened):
        return _opened(started)
    return _serve(started, command_line)


def _opened(opened: Opened) -> int:
    """What a launch that opened the running instance says, and its exit status."""
    if opened.merged:
        _log.info(
            "Proteia is already running; %d images joined those another launch handed to it",
            len(opened.files),
        )
        print("Proteia is already running, and open in your browser.")
        if opened.files:
            names = _names([file.name for file in opened.files])
            print(f"Added {names} to the images waiting there: choose how to import them.")
    else:
        _log.info("Proteia is already running; it has been opened in the browser")
        print("Proteia is already running; it has been opened in your browser.")
        if opened.files:
            _log.info("handed %d images to it", len(opened.files))
            print(f"{_images(len(opened.files))} waiting there: choose how to import them.")
    if opened.refused:
        _log.info("%d paths not taken", len(opened.refused))
    _print_refused(opened.refused)
    return 3 if opened.refused else 0


def _serve(instance: Instance, command_line: CommandLine) -> int:
    """Serve the instance this launch started, until it stops; the exit status."""
    _log.info("serving at http://%s:%d/", HOST, instance.port)
    if command_line.paths:
        _log.info("handed %d images to the page", instance.taken)
    if instance.unread:
        _log.info("%d paths not taken (unreadable)", len(instance.unread))
    print(f"Proteia is running at http://{HOST}:{instance.port}/ and opens in your browser.")
    if instance.taken:
        print(f"{_images(instance.taken)} waiting there: choose how to import them.")
    _print_refused((*command_line.refused, *instance.unread))
    print(f"If no browser window opens, open this file in a browser: {instance.redirect_path}")
    print("To quit, use Quit on the page, or press Ctrl+C here.")
    with contextlib.suppress(KeyboardInterrupt):  # a Ctrl+C before serving begins
        instance.serve()
    not_taken = command_line.refused or instance.taken < len(command_line.files)
    return 3 if not_taken else 0


def _print_refused(refused: Sequence[RefusedPath]) -> None:
    """Each path not taken, in full, and why, on standard error."""
    for path in refused:
        print(f"Not opened: {path.shown} ({path.message})", file=sys.stderr)


def _images(count: int) -> str:
    return "1 image is" if count == 1 else f"{count} images are"


def _names(names: Sequence[str]) -> str:
    """``a.tif``, ``a.tif and b.tif``, ``a.tif, b.tif and c.tif``."""
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


if __name__ == "__main__":
    raise SystemExit(main())
