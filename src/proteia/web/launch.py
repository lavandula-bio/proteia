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

Both files are removed when the server stops (Quit on the page, Ctrl+C, or a
termination signal), before the lock is released. (A launch that opens the
running instance just as it stops can write the redirect page again; its token
opens nothing, and the next launch replaces it.) The only connection a launch
makes is the check of the running instance, to its loopback port, without any
proxy. On POSIX the folder is 0700 and the files 0600; on Windows the per-user
local application-data folder is readable only by its owner.
"""

from __future__ import annotations

import contextlib
import html
import http.client
import json
import os
import secrets
import signal
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import uvicorn

from proteia.core.storage import write_atomic
from proteia.web.api import UnsavedChangesError, Workspace
from proteia.web.server import APP_ID, HOST, TOKEN_PATTERN, create_app

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
_STOP_SIGNALS: Final = tuple(
    getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGBREAK") if hasattr(signal, name)
)

Opener = Callable[[str], object]  # webbrowser.open's shape: takes a URL


class NotRespondingError(RuntimeError):
    """Another instance holds the lock but did not answer within the wait."""


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


def probe(info: InstanceInfo, *, timeout: float = PROBE_TIMEOUT) -> bool:
    """Whether a Proteia instance answers on ``info.port`` and admits its token.

    A direct loopback connection: ``http.client`` uses no proxy settings.
    """
    conn = http.client.HTTPConnection(HOST, info.port, timeout=timeout)
    try:
        conn.request("GET", "/api/status", headers={"Authorization": f"Bearer {info.token}"})
        response = conn.getresponse()
        body = response.read(4096)
        return response.status == 200 and json.loads(body).get("app") == APP_ID
    except (OSError, ValueError, AttributeError, http.client.HTTPException):
        return False
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


def _server(app: object) -> uvicorn.Server:
    config = uvicorn.Config(
        app,
        http="h11",
        ws="none",
        lifespan="off",
        log_level="warning",
        access_log=False,  # the log would name every request
        server_header=False,
        timeout_graceful_shutdown=2,
    )
    return uvicorn.Server(config)


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
        self.server = _server(app)

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
                self.workspace.close()
            finally:
                self.close()

    def stop(self) -> None:
        self.server.should_exit = True

    def close(self) -> None:
        """Release the socket, remove the files that name this instance, then
        release the lock (so no new instance writes its files before that)."""
        self.sock.close()
        for path in (self.folder / INSTANCE_FILE, self.redirect_path):
            with contextlib.suppress(OSError):
                if self.token in path.read_text(encoding="utf-8"):
                    path.unlink()
        self._lock.release()


def _open_running(folder: Path, opener: Opener, wait: float) -> InstanceLock | None:
    """Another process holds the lock: wait until its instance answers and open it
    (None), or return the lock if that instance stops meanwhile (it was quitting)."""
    deadline = time.monotonic() + wait
    while True:
        info = read_instance(folder)
        if info is not None and probe(info):
            open_in_browser(folder, info.port, info.token, opener)
            return None
        lock = InstanceLock.acquire(folder)
        if lock is not None:
            return lock
        if time.monotonic() >= deadline:
            raise NotRespondingError("Proteia is already running but does not respond")
        time.sleep(0.2)


def start(
    *,
    folder: Path | None = None,
    opener: Opener = webbrowser.open,
    wait: float = STARTUP_WAIT,
    workspace: Workspace | None = None,
) -> Instance | None:
    """Open the running instance in the browser (None), or start one and open it.

    A new instance holds the lock, is bound and listening, and has written its
    files before the browser opens; call :meth:`Instance.serve` to serve it. When
    another instance holds the lock, this waits up to ``wait`` seconds for it to
    answer, or to stop and free the lock; then ``NotRespondingError``. A second
    server is never started.
    """
    folder = state_dir() if folder is None else folder
    _private_dir(folder)
    lock = InstanceLock.acquire(folder)
    if lock is None:
        lock = _open_running(folder, opener, wait)
        if lock is None:
            return None
    instance: Instance | None = None
    try:
        sock = bind_loopback()
        try:
            token = secrets.token_urlsafe(TOKEN_BYTES)
            instance = Instance(folder, sock, token, lock, workspace)
        except BaseException:
            sock.close()
            raise
        info = {"pid": os.getpid(), "port": instance.port, "token": instance.token}
        write_atomic(folder / INSTANCE_FILE, json.dumps(info).encode("utf-8"), private=True)
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


def main() -> int:
    """The ``proteia`` console command."""
    _tolerant_console()
    try:
        instance = start()
    except NotRespondingError:
        print(
            "Proteia is already running but does not respond. Try again in a moment,"
            " or end the other Proteia process."
        )
        return 1
    if instance is None:
        print("Proteia is already running; it has been opened in your browser.")
        return 0
    print(f"Proteia is running at http://{HOST}:{instance.port}/ and opens in your browser.")
    print(f"If no browser window opens, open this file in a browser: {instance.redirect_path}")
    print("To quit, use Quit on the page, or press Ctrl+C here.")
    with contextlib.suppress(KeyboardInterrupt):  # a Ctrl+C before serving begins
        instance.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
