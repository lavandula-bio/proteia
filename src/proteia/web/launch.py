# SPDX-License-Identifier: Apache-2.0
"""Start the local web app: ``uv run proteia`` runs :func:`main` (ADR 0002).

A launch binds a listening socket to ``127.0.0.1`` on a port the operating system
assigns (port 0, read back, so no other process can take it in between; on
Windows with exclusive use of the address), and makes a new random token.

The token never appears on a command line or in the console. The launcher writes
it to files only the current user can read, in the per-user state folder
(:func:`state_dir`):

* ``instance.json``, the lock file: the running instance's port and token, so a
  second launch opens that instance instead of starting another. A lock whose
  instance does not answer (it crashed) is replaced;
* ``open-proteia.html``, a redirect page that sends the browser to
  ``http://127.0.0.1:<port>/#token=<token>``. The launcher opens this file, not
  the URL, and the token rides in the URL fragment, which the browser never sends
  to a server. The page keeps it for its tab and sends it in a request header.

Both files are removed when the server stops. The only connection a launch makes
is the check of an existing lock, to its loopback port, without any proxy. On
POSIX the folder is 0700 and the files 0600; on Windows the per-user local
application-data folder is readable only by its owner.
"""

from __future__ import annotations

import contextlib
import html
import http.client
import json
import os
import secrets
import socket
import sys
import tempfile
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import uvicorn

from proteia.web.server import APP_ID, HOST, TOKEN_PATTERN, create_app

LOCK_FILE: Final = "instance.json"
REDIRECT_FILE: Final = "open-proteia.html"
TOKEN_BYTES: Final = 32  # 43 URL-safe characters
PROBE_TIMEOUT: Final = 2.0  # seconds

Opener = Callable[[str], object]  # webbrowser.open's shape: takes a URL


def state_dir() -> Path:
    """The per-user folder that holds the lock and redirect files."""
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


def _write_private(path: Path, data: bytes) -> None:
    """Replace ``path`` atomically with an owner-only (0600 on POSIX) file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@dataclass(frozen=True)
class InstanceInfo:
    """What the lock file records about a running instance."""

    pid: int
    port: int
    token: str


def read_lock(folder: Path) -> InstanceInfo | None:
    """The lock file's instance, or None when there is none or it is unreadable."""
    try:
        doc = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
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

    A direct loopback connection (``http.client`` uses no proxy settings). A
    crashed instance's token may reach whatever now holds the port; it no longer
    opens anything.
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
    _write_private(path, _redirect_page(app_url(port, token)))
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


class Instance:
    """A started server. :meth:`serve` blocks until it stops; :meth:`stop` (or the
    page's Quit, or Ctrl+C when served from the main thread) stops it."""

    def __init__(self, folder: Path, sock: socket.socket, token: str) -> None:
        self.folder = folder
        self.sock = sock
        self.token = token
        self.port: int = sock.getsockname()[1]
        self.redirect_path = folder / REDIRECT_FILE
        self.server = _server(create_app(token=token, port=self.port, on_quit=self.stop))

    def serve(self) -> None:
        """Serve until stopped, then :meth:`close`."""
        try:
            self.server.run(sockets=[self.sock])
        finally:
            self.close()

    def stop(self) -> None:
        self.server.should_exit = True

    def close(self) -> None:
        """Release the socket and remove the files that name this instance."""
        self.sock.close()
        running = read_lock(self.folder)
        if running is not None and running.token == self.token:
            with contextlib.suppress(OSError):
                (self.folder / LOCK_FILE).unlink()
        with contextlib.suppress(OSError):
            if self.token in self.redirect_path.read_text(encoding="utf-8"):
                self.redirect_path.unlink()


def start(*, folder: Path | None = None, opener: Opener = webbrowser.open) -> Instance | None:
    """Open the running instance in the browser (None), or start one and open it.

    The new instance is bound and listening, and its lock file written, before the
    browser opens; call :meth:`Instance.serve` to serve it.
    """
    folder = state_dir() if folder is None else folder
    _private_dir(folder)
    running = read_lock(folder)
    if running is not None and probe(running):
        open_in_browser(folder, running.port, running.token, opener)
        return None
    sock = bind_loopback()
    instance: Instance | None = None
    try:
        instance = Instance(folder, sock, secrets.token_urlsafe(TOKEN_BYTES))
        lock = {"pid": os.getpid(), "port": instance.port, "token": instance.token}
        _write_private(folder / LOCK_FILE, json.dumps(lock).encode("utf-8"))
        open_in_browser(folder, instance.port, instance.token, opener)
    except BaseException:
        if instance is None:
            sock.close()
        else:
            instance.close()
        raise
    return instance


def main() -> int:
    """The ``proteia`` console command."""
    instance = start()
    if instance is None:
        print("Proteia is already running; it has been opened in your browser.")
        return 0
    print(f"Proteia is running at http://{HOST}:{instance.port}/ and opens in your browser.")
    print(f"If no browser window opens, open this file in a browser: {instance.redirect_path}")
    print("To quit, use Quit on the page, or press Ctrl+C here.")
    instance.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
