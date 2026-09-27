# SPDX-License-Identifier: Apache-2.0
"""The local web app's HTTP layer (ADR 0002).

:func:`create_app` builds the app for one launch. It is served only on the
loopback address, and a guard in front of every route admits a request only when:

* its ``Host`` header names exactly ``127.0.0.1:<port>``, the address being
  served (this blocks DNS rebinding from web pages); and
* unless it fetches the page shell (``GET``/``HEAD`` of ``/`` or a file under
  ``/static/``, which hold no user data), it carries ``Authorization: Bearer
  <token>`` with this launch's token. The token, not the loopback binding, keeps
  out other accounts on the same computer and requests sent by other pages.

No cookie is set and no CORS header is sent, so another page can neither make the
browser send the token nor read a response. The generated API description and
documentation pages are off. Every response carries a Content-Security-Policy
that lets the page load and fetch only from this server.

The guard logs (:mod:`proteia.web.logs`) the requests it refuses, by method and
path (never a header, so never a token): anyone who can reach the port can send
them, so the path is cut short (:func:`~proteia.web.logs.shorten`) and at most
:data:`~proteia.web.logs.REFUSALS_LOGGED` a minute are logged one by one, the
rest counted (:class:`~proteia.web.logs.Throttle`). It also logs every error no
route answered, with its stack trace, once, in the log file only: the console
shows uvicorn's report of it, as before.
"""

from __future__ import annotations

import hmac
import json
import logging
import mimetypes
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import proteia
from proteia.web import api, logs, projects

HOST: Final = "127.0.0.1"
APP_ID: Final = "proteia"  # what /api/status reports, so a launcher knows it found Proteia
# What /api/status reports as ``handoff``: the version of the routes that take
# files from a launch (proteia.web.handoff), so a launcher knows, before it
# sends any, that this Proteia can take them.
HANDOFF: Final = 1
STATIC_DIR: Final = Path(__file__).parent / "static"
# secrets.token_urlsafe output: URL-safe base64 without padding.
TOKEN_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]{32,128}$")

# Windows can map .js to text/plain in the registry; with nosniff the browser
# would then refuse to run the script.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

_CSP: Final = (
    "default-src 'self'; img-src 'self' blob: data:; object-src 'none'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
_SECURITY_HEADERS: Final = (
    (b"content-security-policy", _CSP.encode("ascii")),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"cache-control", b"no-store"),
)
_REPLACED: Final = frozenset(name for name, _ in _SECURITY_HEADERS)

_log = logging.getLogger(__name__)


def _is_shell(scope: Scope) -> bool:
    """A read of the page shell, which is served without the token."""
    if scope["method"] not in ("GET", "HEAD"):
        return False
    path: str = scope["path"]
    return path == "/" or (path.startswith("/static/") and ".." not in path.split("/"))


def _with_security_headers(headers: Iterable[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    kept = [(name, value) for name, value in headers if name.lower() not in _REPLACED]
    return kept + list(_SECURITY_HEADERS)


async def _reject(send: Send, status: int, detail: str, *extra: tuple[bytes, bytes]) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        *extra,
    ]
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": _with_security_headers(headers),
        }
    )
    await send({"type": "http.response.body", "body": body})


class Guard:
    """The ASGI app that wraps the FastAPI app (``app``), outside all of its
    middleware: the Host and token checks, and the security headers on every
    response, a 500 from FastAPI's error handler included (see the module
    docstring)."""

    def __init__(self, app: ASGIApp, *, token: str, port: int) -> None:
        self.app = app
        self._host = f"{HOST}:{port}".encode("ascii")
        self._authorization = f"Bearer {token}".encode("ascii")
        self.refusals = logs.Throttle()  # the refusals logged one by one

    def _log_refusal(self, method: str, path: str, why: str) -> None:
        admitted, left_out = self.refusals.admit()
        if left_out:
            _log.info(
                "refused %d more requests on their Host header or token, not logged one by one"
                " (more than %d a minute)",
                left_out,
                self.refusals.limit,
            )
        if admitted:
            _log.info("refused %s %s: %s", method, logs.shorten(path), why)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":  # none are served
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":  # lifespan
            await self.app(scope, receive, send)
            return
        headers: list[tuple[bytes, bytes]] = scope["headers"]
        method, path = scope["method"], scope["path"]
        hosts = [value for name, value in headers if name == b"host"]
        if hosts != [self._host]:
            self._log_refusal(method, path, "400 unexpected Host header")
            await _reject(send, 400, "unexpected Host header")
            return
        if not _is_shell(scope):
            given = [value for name, value in headers if name == b"authorization"]
            if len(given) != 1 or not hmac.compare_digest(given[0], self._authorization):
                self._log_refusal(method, path, "401 missing or wrong access token")
                await _reject(
                    send, 401, "missing or wrong access token", (b"www-authenticate", b"Bearer")
                )
                return

        async def send_secured(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": _with_security_headers(message["headers"])}
            await send(message)

        try:
            await self.app(scope, receive, send_secured)
        except Exception as exc:
            # Answered 500 by FastAPI's last handler, then raised again for the
            # server, which reports it on the console: the file has it once, here,
            # with the request.
            _log.error(
                "unexpected error answering %s %s",
                method,
                logs.shorten(path),
                exc_info=exc,
                extra=logs.FILE_ONLY,
            )
            logs.mark_logged(exc)
            raise


def create_app(
    *,
    token: str,
    port: int,
    on_quit: Callable[[], None],
    workspace: api.Workspace | None = None,
) -> Guard:
    """The app for one launch served at ``127.0.0.1:port``, admitting ``token``.

    ``on_quit`` runs when the page asks the app to quit (``POST /api/quit``); it
    must return at once, and the server stops after answering. ``workspace``
    holds the projects (default: the app-managed projects root,
    :func:`~proteia.web.projects.projects_root`), served by :mod:`proteia.web.api`.
    """
    if not TOKEN_PATTERN.fullmatch(token):
        raise ValueError("the token must be 32 to 128 URL-safe characters")
    if not 0 < port < 65536:
        raise ValueError(f"port {port} is out of range")
    app = FastAPI(title="Proteia", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", include_in_schema=False)
    def page_shell() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/api/status")
    def status() -> dict[str, str | int]:
        return {"app": APP_ID, "version": proteia.__version__, "handoff": HANDOFF}

    @app.post("/api/quit", status_code=202)
    def quit_app(request: Request) -> dict[str, str]:
        # Unsaved changes (a failed autosave) are saved first; if that fails the
        # answer is 409 unsaved_changes and Proteia keeps running.
        request.app.state.workspace.flush()
        _log.info("quit from the page")
        on_quit()
        return {"status": "stopping"}

    if workspace is None:
        workspace = api.Workspace(projects.projects_root(), reveal=projects.reveal)
    api.install(app, workspace)
    return Guard(app, token=token, port=port)
