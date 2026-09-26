# SPDX-License-Identifier: Apache-2.0
"""The local web server (ADR 0002): the loopback binding, the per-launch token,
the Host check, the served page shell and the launcher's files. Requests go to a
real server over a loopback socket, sent with the standard library's http.client."""

from __future__ import annotations

import http.client
import importlib.metadata
import json
import os
import socket
import stat
import threading
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import pytest

import proteia
from proteia.web import launch, server
from proteia.web.launch import LOCK_FILE, REDIRECT_FILE

OMIT = object()  # send no Host header


class Opener:
    """Stands in for webbrowser.open: records the URLs, opens nothing."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, url: str) -> bool:
        self.urls.append(url)
        return True


@dataclass
class Running:
    instance: launch.Instance
    opener: Opener
    thread: threading.Thread

    @property
    def port(self) -> int:
        return self.instance.port

    @property
    def token(self) -> str:
        return self.instance.token


@pytest.fixture
def running(tmp_path):
    opener = Opener()
    instance = launch.start(folder=tmp_path, opener=opener)
    assert instance is not None
    thread = threading.Thread(target=instance.serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "the server did not start"
        time.sleep(0.01)
    yield Running(instance, opener, thread)
    instance.stop()
    thread.join(10)


def send(
    port: int,
    method: str,
    path: str,
    *,
    token: str | None = None,
    host: object = None,
    headers: tuple[tuple[str, str], ...] = (),
) -> tuple[int, dict[str, str], bytes]:
    """One request; ``host`` None sends the right Host header, OMIT sends none."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        if host is None:
            conn.putheader("Host", f"127.0.0.1:{port}")
        elif host is not OMIT:
            conn.putheader("Host", host)
        if token is not None:
            conn.putheader("Authorization", f"Bearer {token}")
        for name, value in headers:
            conn.putheader(name, value)
        conn.endheaders()
        response = conn.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        conn.close()


# --- The launcher ---


def test_start_binds_loopback_and_opens_the_redirect_page(tmp_path):
    opener = Opener()
    instance = launch.start(folder=tmp_path, opener=opener)
    assert instance is not None
    try:
        host, port = instance.sock.getsockname()
        assert (host, port) == ("127.0.0.1", instance.port) and port > 0
        assert server.TOKEN_PATTERN.fullmatch(instance.token) and len(instance.token) == 43
        redirect = tmp_path / REDIRECT_FILE
        assert opener.urls == [redirect.as_uri()]  # the file, not the URL with the token
        assert launch.app_url(port, instance.token) in redirect.read_text(encoding="utf-8")
        lock = json.loads((tmp_path / LOCK_FILE).read_text(encoding="utf-8"))
        assert lock == {"pid": os.getpid(), "port": port, "token": instance.token}
    finally:
        instance.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_each_launch_has_its_own_token(tmp_path):
    first = launch.start(folder=tmp_path / "a", opener=Opener())
    second = launch.start(folder=tmp_path / "b", opener=Opener())
    try:
        assert first.token != second.token
        assert first.port != second.port
    finally:
        first.close()
        second.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_launch_files_are_private(tmp_path):
    folder = tmp_path / "state"
    instance = launch.start(folder=folder, opener=Opener())
    try:
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
        for name in (LOCK_FILE, REDIRECT_FILE):
            assert stat.S_IMODE((folder / name).stat().st_mode) == 0o600
    finally:
        instance.close()


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize(
    "lock",
    [
        lambda: json.dumps({"pid": 1, "port": _closed_port(), "token": "x" * 43}),
        lambda: "not json",
        lambda: json.dumps({"pid": 1, "port": 0, "token": "x" * 43}),
        lambda: json.dumps({"pid": 1, "port": 8080, "token": "short"}),
    ],
)
def test_a_stale_or_broken_lock_is_replaced(tmp_path, lock):
    (tmp_path / LOCK_FILE).write_text(lock(), encoding="utf-8")
    instance = launch.start(folder=tmp_path, opener=Opener())
    assert instance is not None
    try:
        written = json.loads((tmp_path / LOCK_FILE).read_text(encoding="utf-8"))
        assert written["token"] == instance.token
    finally:
        instance.close()


def test_a_second_launch_opens_the_running_instance(running, tmp_path):
    lock = (tmp_path / LOCK_FILE).read_bytes()
    opener = Opener()
    assert launch.start(folder=tmp_path, opener=opener) is None
    assert opener.urls == [(tmp_path / REDIRECT_FILE).as_uri()]
    page = (tmp_path / REDIRECT_FILE).read_text(encoding="utf-8")
    assert launch.app_url(running.port, running.token) in page
    assert (tmp_path / LOCK_FILE).read_bytes() == lock


def test_the_probe_ignores_proxy_settings(running, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    info = launch.InstanceInfo(pid=os.getpid(), port=running.port, token=running.token)
    assert launch.probe(info)
    assert not launch.probe(launch.InstanceInfo(pid=1, port=running.port, token="y" * 43))


def test_quit_stops_the_server_and_removes_its_files(running, tmp_path):
    status, _, body = send(running.port, "POST", "/api/quit", token=running.token)
    assert status == 202 and json.loads(body) == {"status": "stopping"}
    running.thread.join(10)
    assert not running.thread.is_alive()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_the_console_command_starts_the_web_app():
    (entry,) = importlib.metadata.entry_points(group="console_scripts", name="proteia")
    assert entry.value == "proteia.web.launch:main"


class _FakeInstance:
    port = 1234
    redirect_path = Path("open-proteia.html")
    token = "s" * 43

    def __init__(self) -> None:
        self.served = False

    def serve(self) -> None:
        self.served = True


def test_main_serves_without_printing_the_token(monkeypatch, capsys):
    fake = _FakeInstance()
    monkeypatch.setattr(launch, "start", lambda: fake)
    assert launch.main() == 0
    assert fake.served
    out = capsys.readouterr().out
    assert "http://127.0.0.1:1234/" in out and fake.token not in out

    monkeypatch.setattr(launch, "start", lambda: None)
    assert launch.main() == 0
    assert "already running" in capsys.readouterr().out


@pytest.mark.parametrize(("token", "port"), [("short", 8080), ("x" * 43, 0), ("x" * 43, 70000)])
def test_create_app_refuses_a_bad_token_or_port(token, port):
    with pytest.raises(ValueError):
        server.create_app(token=token, port=port, on_quit=lambda: None)


# --- The guard ---


@pytest.mark.parametrize("path", ["/api/status", "/api/quit", "/api/nothing", "/docs"])
@pytest.mark.parametrize("token", [None, "z" * 43, "wrong"])
def test_a_request_without_the_right_token_is_refused(running, path, token):
    method = "POST" if path == "/api/quit" else "GET"
    status, headers, _ = send(running.port, method, path, token=token)
    assert status == 401
    assert headers["www-authenticate"] == "Bearer"
    assert running.thread.is_alive()


def test_two_authorization_headers_are_refused(running):
    auth = ("Authorization", f"Bearer {running.token}")
    status, _, _ = send(running.port, "GET", "/api/status", headers=(auth, auth))
    assert status == 401


def test_the_right_token_and_host_are_admitted(running):
    status, headers, body = send(running.port, "GET", "/api/status", token=running.token)
    assert status == 200
    assert json.loads(body) == {"app": "proteia", "version": proteia.__version__}
    assert headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "evil.example:{port}",
        "localhost:{port}",
        "127.0.0.1",
        "127.0.0.1:{other}",
        "[::1]:{port}",
        "127.0.0.1:{port}.evil.example",
        OMIT,
    ],
)
@pytest.mark.parametrize("path", ["/", "/static/app.js", "/api/status"])
def test_a_foreign_host_header_is_refused(running, host, path):
    if isinstance(host, str):
        host = host.format(port=running.port, other=running.port % 65535 + 1)
    status, _, _ = send(running.port, "GET", path, token=running.token, host=host)
    assert status == 400


def test_two_host_headers_are_refused(running):
    right = ("Host", f"127.0.0.1:{running.port}")
    status, _, _ = send(
        running.port, "GET", "/", token=running.token, host=OMIT, headers=(right, right)
    )
    assert status == 400


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/api/nothing"])
def test_no_api_description_or_unknown_route_is_served(running, path):
    status, _, _ = send(running.port, "GET", path, token=running.token)
    assert status == 404


def test_a_cross_site_request_gets_no_cors_permission(running):
    headers = (
        ("Origin", "http://evil.example"),
        ("Access-Control-Request-Method", "POST"),
        ("Access-Control-Request-Headers", "authorization"),
    )
    status, got, _ = send(running.port, "OPTIONS", "/api/quit", headers=headers)
    assert status == 401 and "access-control-allow-origin" not in got
    status, got, _ = send(running.port, "POST", "/api/quit", headers=headers[:1])
    assert status == 401 and "access-control-allow-origin" not in got
    assert running.thread.is_alive()


@pytest.mark.parametrize(
    "path", ["/static/../launch.py", "/static/%2e%2e/launch.py", "/static/..%2flaunch.py"]
)
def test_static_paths_stay_in_the_static_folder(running, path):
    status, _, body = send(running.port, "GET", path)
    assert status == 401  # not the shell: the token is required
    status, _, body = send(running.port, "GET", path, token=running.token)
    assert status == 404 and b"def start" not in body


# --- The page shell ---


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.links += [value for name, value in attrs if name in ("src", "href") and value]


def test_the_page_shell_is_served_without_the_token(running):
    status, headers, body = send(running.port, "GET", "/")
    assert status == 200
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert body == (server.STATIC_DIR / "index.html").read_bytes()
    assert "default-src 'self'" in headers["content-security-policy"]
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert headers["x-frame-options"] == "DENY"
    assert headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in headers
    for name, kind in (("app.js", "text/javascript"), ("app.css", "text/css")):
        status, headers, body = send(running.port, "GET", f"/static/{name}")
        assert status == 200 and headers["content-type"].startswith(kind)
        assert body == (server.STATIC_DIR / name).read_bytes()


def test_the_page_shell_loads_nothing_from_the_network(running):
    _, _, body = send(running.port, "GET", "/")
    parser = _Links()
    parser.feed(body.decode("utf-8"))
    assert parser.links == ["/static/app.css", "/static/app.js"]
    for path in server.STATIC_DIR.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "://" not in text and "@import" not in text and "url(" not in text, path.name
