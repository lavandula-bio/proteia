# SPDX-License-Identifier: Apache-2.0
"""The local web server (ADR 0002): the loopback binding, the per-launch token,
the Host check, the served page shell and the launcher's lock and files. Requests
go to a real server over a loopback socket, sent with the standard library's
http.client."""

from __future__ import annotations

import asyncio
import http.client
import importlib.metadata
import io
import json
import os
import re
import signal
import socket
import stat
import sys
import threading
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import get_args

import pytest

import proteia
from proteia.core import operations as ops
from proteia.core import results, rowdetect
from proteia.core.model import ImageKind, Polarity
from proteia.web import api, launch, server
from proteia.web.launch import INSTANCE_FILE, LOCK_FILE, REDIRECT_FILE

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


def wait_until_started(instance: launch.Instance, thread: threading.Thread | None = None):
    deadline = time.monotonic() + 10
    while not instance.server.started:
        assert time.monotonic() < deadline, "the server did not start"
        assert thread is None or thread.is_alive(), "the server thread ended"
        time.sleep(0.01)


@pytest.fixture
def running(tmp_path):
    opener = Opener()
    instance = launch.start(folder=tmp_path, opener=opener)
    assert instance is not None
    thread = threading.Thread(target=instance.serve, daemon=True)
    thread.start()
    wait_until_started(instance, thread)
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


def names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


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
        info = json.loads((tmp_path / INSTANCE_FILE).read_text(encoding="utf-8"))
        assert info == {"pid": os.getpid(), "port": port, "token": instance.token}
    finally:
        instance.close()
    assert names(tmp_path) == [LOCK_FILE]  # the lock file stays, released
    assert launch.InstanceLock.acquire(tmp_path) is not None


def test_each_launch_has_its_own_token(tmp_path):
    first = launch.start(folder=tmp_path / "a", opener=Opener())
    second = launch.start(folder=tmp_path / "b", opener=Opener())
    try:
        assert first.token != second.token
        assert first.port != second.port
    finally:
        first.close()
        second.close()


def test_a_state_folder_with_a_non_ascii_name_works(tmp_path):
    folder = tmp_path / "µ α β 狀態"  # µ α β and two CJK characters
    opener = Opener()
    instance = launch.start(folder=folder, opener=opener)
    try:
        (uri,) = opener.urls
        assert uri == (folder / REDIRECT_FILE).as_uri() and uri.isascii()
        assert instance.token in (folder / REDIRECT_FILE).read_text(encoding="utf-8")
    finally:
        instance.close()
    assert names(folder) == [LOCK_FILE]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_launch_files_are_private(tmp_path):
    folder = tmp_path / "state"
    instance = launch.start(folder=folder, opener=Opener())
    try:
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
        for name in (LOCK_FILE, INSTANCE_FILE, REDIRECT_FILE):
            assert stat.S_IMODE((folder / name).stat().st_mode) == 0o600
    finally:
        instance.close()


def test_the_instance_lock_is_exclusive_until_released(tmp_path):
    first = launch.InstanceLock.acquire(tmp_path)
    assert first is not None
    assert launch.InstanceLock.acquire(tmp_path) is None  # another open file, same process
    first.release()
    again = launch.InstanceLock.acquire(tmp_path)
    assert again is not None
    again.release()


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize(
    "leftover",
    [
        lambda: json.dumps({"pid": 1, "port": _closed_port(), "token": "x" * 43}),
        lambda: "not json",
        lambda: json.dumps({"pid": 1, "port": 0, "token": "x" * 43}),
        lambda: json.dumps({"pid": 1, "port": 8080, "token": "short"}),
    ],
)
def test_files_left_by_a_crashed_instance_are_replaced(tmp_path, leftover):
    (tmp_path / INSTANCE_FILE).write_text(leftover(), encoding="utf-8")
    (tmp_path / REDIRECT_FILE).write_text("old", encoding="utf-8")
    instance = launch.start(folder=tmp_path, opener=Opener())  # the lock is free
    assert instance is not None
    try:
        info = json.loads((tmp_path / INSTANCE_FILE).read_text(encoding="utf-8"))
        assert info["token"] == instance.token
        assert instance.token in (tmp_path / REDIRECT_FILE).read_text(encoding="utf-8")
    finally:
        instance.close()


def test_a_second_launch_opens_the_running_instance(running, tmp_path):
    info = (tmp_path / INSTANCE_FILE).read_bytes()
    opener = Opener()
    assert launch.start(folder=tmp_path, opener=opener) is None
    assert opener.urls == [(tmp_path / REDIRECT_FILE).as_uri()]
    page = (tmp_path / REDIRECT_FILE).read_text(encoding="utf-8")
    assert launch.app_url(running.port, running.token) in page
    assert (tmp_path / INSTANCE_FILE).read_bytes() == info


def test_a_second_launch_waits_for_a_starting_instance(running, tmp_path):
    path = tmp_path / INSTANCE_FILE
    info = path.read_bytes()
    path.unlink()  # as if the running instance had not written it yet
    timer = threading.Timer(0.5, path.write_bytes, args=(info,))
    timer.start()
    opener = Opener()
    try:
        assert launch.start(folder=tmp_path, opener=opener, wait=10) is None
    finally:
        timer.join()
    assert opener.urls == [(tmp_path / REDIRECT_FILE).as_uri()]


@pytest.mark.parametrize("with_info", [False, True])
def test_a_launch_never_starts_a_second_server(tmp_path, with_info):
    held = launch.InstanceLock.acquire(tmp_path)  # an instance that does not answer
    if with_info:
        info = {"pid": 1, "port": _closed_port(), "token": "x" * 43}
        (tmp_path / INSTANCE_FILE).write_text(json.dumps(info), encoding="utf-8")
    opener = Opener()
    try:
        with pytest.raises(launch.NotRespondingError):
            launch.start(folder=tmp_path, opener=opener, wait=0.5)
    finally:
        held.release()
    assert opener.urls == []


def test_a_launch_starts_when_the_waited_for_instance_stops(tmp_path):
    held = launch.InstanceLock.acquire(tmp_path)  # an instance that is quitting
    timer = threading.Timer(0.5, held.release)
    timer.start()
    opener = Opener()
    try:
        instance = launch.start(folder=tmp_path, opener=opener, wait=10)
    finally:
        timer.join()
    assert instance is not None
    try:
        assert opener.urls == [(tmp_path / REDIRECT_FILE).as_uri()]
        assert launch.InstanceLock.acquire(tmp_path) is None  # the new instance holds it
    finally:
        instance.close()


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
    assert names(tmp_path) == [LOCK_FILE]
    assert launch.InstanceLock.acquire(tmp_path) is not None


_SIGNALS = ["SIGINT", "SIGTERM"] + (["SIGBREAK"] if hasattr(signal, "SIGBREAK") else [])


@pytest.mark.parametrize("name", _SIGNALS)
def test_a_stop_signal_stops_the_server_and_cleans_up(tmp_path, name):
    sig = getattr(signal, name)
    before = signal.getsignal(sig)
    instance = launch.start(folder=tmp_path, opener=Opener())

    def signal_once_started() -> None:
        wait_until_started(instance)
        signal.raise_signal(sig)

    threading.Thread(target=signal_once_started, daemon=True).start()
    instance.serve()  # in the main thread, as the proteia command serves
    assert names(tmp_path) == [LOCK_FILE]
    assert signal.getsignal(sig) is before
    assert launch.InstanceLock.acquire(tmp_path) is not None


def test_the_console_command_starts_the_web_app():
    (entry,) = importlib.metadata.entry_points(group="console_scripts", name="proteia")
    assert entry.value == "proteia.web.launch:main"


@pytest.fixture
def state(tmp_path, monkeypatch):
    """The launcher's state folder in ``tmp_path``: :func:`launch.main` writes its
    session log there, not in the user's."""
    monkeypatch.setattr(launch, "state_dir", lambda: tmp_path / "state")
    return tmp_path / "state"


class _FakeInstance:
    port = 1234
    redirect_path = Path("µ α β") / "open-proteia.html"
    token = "s" * 43

    def __init__(self) -> None:
        self.served = False

    def serve(self) -> None:
        self.served = True


def test_main_serves_without_printing_the_token(state, monkeypatch, capsys):
    fake = _FakeInstance()
    monkeypatch.setattr(launch, "start", lambda: fake)
    assert launch.main() == 0
    assert fake.served
    out = capsys.readouterr().out
    assert "http://127.0.0.1:1234/" in out and fake.token not in out

    monkeypatch.setattr(launch, "start", lambda: None)
    assert launch.main() == 0
    assert "already running" in capsys.readouterr().out


def test_main_reports_an_instance_that_does_not_respond(state, monkeypatch, capsys):
    def refuse():
        raise launch.NotRespondingError("no answer")

    monkeypatch.setattr(launch, "start", refuse)
    assert launch.main() == 1
    console = capsys.readouterr()
    assert "does not respond" in console.out and not console.err  # printed once
    log = (state / "logs" / "proteia.log").read_text(encoding="utf-8")
    assert "ERROR proteia.web.launch: Proteia is already running but does not respond" in log


def test_main_prints_a_non_ascii_path_to_a_narrow_console(state, monkeypatch):
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(launch, "start", _FakeInstance)
    assert launch.main() == 0
    stdout.flush()
    assert b"\\xb5" in stdout.buffer.getvalue()  # the micro sign, escaped


@pytest.mark.parametrize(("token", "port"), [("short", 8080), ("x" * 43, 0), ("x" * 43, 70000)])
def test_create_app_refuses_a_bad_token_or_port(token, port):
    with pytest.raises(ValueError):
        server.create_app(token=token, port=port, on_quit=lambda: None)


def test_a_server_error_carries_the_security_headers():
    token, port = "t" * 43, 8123
    guard = server.create_app(token=token, port=port, on_quit=lambda: None)

    def fail() -> None:
        raise RuntimeError("a bug in a route")

    guard.app.add_api_route("/api/fail", fail)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/fail",
        "raw_path": b"/api/fail",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", f"127.0.0.1:{port}".encode()),
            (b"authorization", f"Bearer {token}".encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", port),
    }
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def record(message):
        sent.append(message)

    with pytest.raises(RuntimeError, match="a bug in a route"):
        asyncio.run(guard(scope, receive, record))
    start = sent[0]
    assert start["status"] == 500
    headers = dict(start["headers"])
    assert b"default-src 'self'" in headers[b"content-security-policy"]
    assert headers[b"cache-control"] == b"no-store"


# --- The guard ---


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/status"),
        ("POST", "/api/quit"),
        ("GET", "/api/workspace"),
        ("GET", "/api/nothing"),
        ("GET", "/docs"),
        ("POST", "/api/incoming?name=a.tif"),
        ("POST", "/api/handoffs"),
        ("POST", "/api/handoffs/0/accept"),
        ("POST", "/api/handoffs/0/discard"),
    ],
)
@pytest.mark.parametrize("token", [None, "z" * 43, "wrong"])
def test_a_request_without_the_right_token_is_refused(running, method, path, token):
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
    assert json.loads(body) == {"app": "proteia", "version": proteia.__version__, "handoff": 1}
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
    # An empty icon: the browser asks for no /favicon.ico, which would need the token.
    assert parser.links == ["data:,", "/static/app.css", "/static/app.js"]
    for path in server.STATIC_DIR.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "://" not in text and "@import" not in text and "url(" not in text, path.name


def test_every_module_the_page_imports_is_served(running):
    # The page is plain ES modules with no build step: each import must resolve.
    pending, seen = ["/static/app.js"], set()
    while pending:
        path = pending.pop()
        seen.add(path)
        status, headers, body = send(running.port, "GET", path)
        assert status == 200 and headers["content-type"].startswith("text/javascript"), path
        text = body.decode("utf-8")
        for target in re.findall(r'^import\b[^;]*?\bfrom\s+"([^"]+)";', text, re.M | re.S):
            assert target.startswith("/static/"), target
            if target not in seen:
                pending.append(target)
    assert seen == {
        "/static/app.js",
        "/static/charts.js",
        "/static/diagnostics.js",
        "/static/dock.js",
        "/static/dom.js",
        "/static/handoffs.js",
        "/static/lanes.js",
        "/static/proteins.js",
        "/static/view.js",
    }
    assert {f"/static/{p.name}" for p in server.STATIC_DIR.glob("*.js")} == seen  # none unused


def test_the_page_never_parses_text_as_markup():
    # Names, conditions and messages go into the page as text (textContent,
    # Option, append of strings), so none can become markup or script.
    for path in server.STATIC_DIR.glob("*.js"):
        text = path.read_text(encoding="utf-8")
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
            assert sink not in text, (path.name, sink)


class _Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.by_id: dict[str, dict[str, str | None]] = {}

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if fields.get("id"):
            self.by_id[fields["id"]] = fields


def test_the_import_file_input_can_take_the_keyboard_focus():
    # Its label "Import image…" is what shows. The input itself is only out of
    # sight (class file-input), never display:none (the hidden attribute) or out
    # of the tab order, so Tab reaches it and Enter or Space opens the chooser.
    parser = _Tags()
    parser.feed((server.STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    fields = parser.by_id["import-file"]
    assert (fields["type"], fields["class"]) == ("file", "file-input")
    assert "hidden" not in fields and fields.get("tabindex") != "-1"
    css = (server.STATIC_DIR / "app.css").read_text(encoding="utf-8")
    rule = css[css.index(".file-input {") :].split("}", 1)[0]
    assert "display" not in rule and "visibility" not in rule


def test_the_original_colours_switch_is_a_toggle_button_the_keyboard_reaches():
    # "Original colours" (#57): a button, so Tab reaches it and Enter or Space
    # presses it; aria-pressed tells a screen reader whether it is on, and a
    # status line says which colours the view shows once they are shown.
    parser = _Tags()
    parser.feed((server.STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    fields = parser.by_id["original-colours"]
    assert (fields["type"], fields["aria-pressed"], fields["aria-keyshortcuts"]) == (
        "button",
        "false",
        "C",
    )
    assert "hidden" in fields  # until an image whose file has colour is shown
    assert fields.get("tabindex") != "-1"
    assert "grey analysis image" in fields["title"]  # what the nets are measured on
    assert parser.by_id["view-colours-state"]["role"] == "status"


def _object_keys(script: str, name: str) -> set[str]:
    """The keys of the object literal ``const <name> = {...};`` in a script."""
    body = script[script.index(f"const {name} = {{") :].split("\n};", 1)[0]
    return set(re.findall(r"^  (\w+): ", body, re.MULTILINE))


def test_the_page_words_every_row_warning_and_empty_lane_reason():
    # Each warning the detector gives a row it placed, and each reason it
    # leaves a lane empty, has the page's own words; one without would reach
    # the user only through the log.
    script = (server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert _object_keys(script, "ROW_WARNINGS") == set(rowdetect.WARNING_FLAGS)
    assert _object_keys(script, "NOT_MEASURED") == set(get_args(rowdetect.LaneReason)) - {"band"}


def test_a_row_whose_lanes_are_doubtful_asks_to_check_them_with_an_undo():
    # #111: the doubtful_lanes warning, as signal that fits no lane does
    # (#106), makes the row's report ask to check the lane numbers, naming
    # the cause; placing the row then offers an Undo of it.
    script = _code("app.js")
    _, report = _function(script, "function rowReport(")
    assert 'answer.flags.includes("doubtful_lanes")' in report
    assert "a row box that also covers a ladder, labels or" in report
    assert "(Undo, then drag over the ${all} lanes only)" in report
    _, place = _function(script, "async function placeRow(")
    assert "check && step" in place and 'label: "Undo"' in place


def test_the_page_offers_the_boxes_in_the_way_of_a_box_or_a_row():
    # A box placed or moved, or a row, refused over boxes in its way (an
    # overlap: another box of the protein, or most of another protein's box,
    # #114) names them, protein by protein, with an Undo of the last box
    # change when it made one of them (it may be the mistake), else a Select
    # of the first of them, shown on its image to move or delete.
    script = _code("app.js")
    _, offer = _function(script, "function boxesInTheWay(")
    assert "namedBoxes(error)" in offer
    assert "lastBoxStepNamed(error)" in offer and 'label: "Undo"' in offer
    assert 'label: "Select"' in offer and "selectBox(first.band.id)" in offer
    assert "boxesInTheWay(error, again)" in _function(script, "function showBoxRefusal(")[1]
    _, place = _function(script, "async function placeBox(")
    assert 'showBoxRefusal(error, "Box not placed", "click again")' in place
    move = script[script.index("  move: (boxId, rect) =>") : script.index("  select: (boxId) =>")]
    assert 'showBoxRefusal(error, "Box not moved", "move it again")' in move
    _, row = _function(script, "function showRowRefusal(")
    assert 'showBoxRefusal(error, `Row box of ${name} not placed`, "drag again")' in row
    _, select = _function(script, "function selectBox(")
    assert "state.imageId = found.protein.image_id" in select and "state.boxId = boxId" in select
    _, named = _function(script, "function namedBoxes(")
    assert "found.protein" in named and "lanesPhrase(" in named


def _logged_actions() -> set[str]:
    """Every action the operations log, found in their source: each ``_apply``
    of an operation, each entry committed with its own ``action`` (an import,
    a new project) and undo and redo. Not ``migrate``: storage logs it when it
    upgrades a file, and it cannot be undone."""
    core = Path(ops.__file__).parent
    operations = (core / "operations.py").read_text(encoding="utf-8")
    session = (core / "session.py").read_text(encoding="utf-8")
    applied = re.findall(r'\b_apply\(\s*session,\s*"(\w+)"', operations)
    # Each call is found, however it is laid out.
    assert len(applied) == len(re.findall(r"(?<!def )\b_apply\(", operations))
    moves = re.findall(r'\._move\("(\w+)"\)', operations)
    own = re.findall(r'\baction="(\w+)"', operations + session)
    return {*applied, *moves, *own}


def test_the_page_words_every_logged_action():
    # Undo and Redo name the change they take back or make again, and so does
    # the status line after a step, in the words of the control that made it
    # (ACTION_WORDS); a change without them would be named by its log id.
    actions = _logged_actions()
    assert {"set_box_padding", "import_image", "new_project", "undo", "redo"} <= actions
    script = (server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert _object_keys(script, "ACTION_WORDS") == actions


def _set_members(script: str, name: str) -> set[str]:
    """The strings of the set literal ``const <name> = new Set([...]);`` in a script."""
    body = script[script.index(f"const {name} = new Set([") :].split("]);", 1)[0]
    return set(re.findall(r'"(\w+)"', body))


def test_the_page_shows_a_notice_about_one_series_under_that_series_only():
    # A notice about one series (its test, its values) names the series' target
    # and loading control. The page shows it, and counts it in the card header,
    # under that series' chart only, not under another target's over the same
    # loading control (#52); for that it keeps the core's list of such notices.
    script = (server.STATIC_DIR / "charts.js").read_text(encoding="utf-8")
    assert _set_members(script, "ONE_SERIES") == set(results.SERIES_NOTICE_CODES)


# --- A page that shows a project no longer open (#134) ---


def _code(name: str) -> str:
    """A page script without its comments (no string in the scripts holds ``//``)."""
    script = (server.STATIC_DIR / name).read_text(encoding="utf-8")
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", script, flags=re.DOTALL))


def _function(script: str, head: str) -> tuple[int, str]:
    """Where the top-level function that starts with ``head`` begins, and its text."""
    start = script.index(head)
    return start, script[start : script.index("\n}\n", start) + 2]


def test_the_page_words_a_box_possibly_over_exposed():
    # #112: where the exact check could not run, the box panel, the box's label
    # on the image and the lane table say what the heuristic found. The notice
    # needs none of the page's lists: it is about one protein, as the clipped
    # notice is, so the Checks list and every chart card of the protein show it.
    app, charts = _code("app.js"), _code("charts.js")
    _, text = _function(app, "function overExposureText(")
    assert "band.possibly_clipped === true" in text and '"Possibly: ' in text
    assert "band.possibly_clipped === false" in text
    _, label = _function(app, "function overExposureLabel(")
    assert "band.possibly_clipped === true" in label and '" over-exposed?"' in label
    assert "column.possibly_clipped[lane] === true" in _code("lanes.js")
    for name in ("NO_CHART", "OWN_SET", "ONE_SERIES"):
        assert results.NoticeCode.POSSIBLY_CLIPPED.value not in _set_members(charts, name)


def test_the_box_panel_does_not_reassure_where_the_heuristic_saw_nothing():
    # #112: a band the heuristic did not flag was still not checked. Its grey
    # analysis image cannot show every saturation (one colour channel saturated
    # alone moves the grey mean a third of the way), so the panel says where it
    # looked, never that few pixels came near the limit.
    _, text = _function(_code("app.js"), "function overExposureText(")
    assert '"Not checked; no sign of it in the grey analysis image"' in text
    assert "few pixels" not in text


def test_the_page_offers_to_requantify_boxes_not_assessed_for_over_exposure():
    # #112: boxes measured before Proteia looked for pixels near the detector
    # limit (the state's unassessed_images) get the header's requantify offer,
    # worded for what it does there; the legacy background keeps its own. The
    # history names the change without the background, which it may not touch.
    app = _code("app.js")
    _, reason = _function(app, "function requantifyReason(")
    assert "project.unassessed_images.length" in reason
    assert "LEGACY_BACKGROUND" in reason
    assert _object_keys(app, "REQUANTIFY_OFFERS") == {"background", "overExposure"}
    offers = app[app.index("const REQUANTIFY_OFFERS = {") :].split("\n};", 1)[0]
    assert "near the detector limit" in offers
    _, render = _function(app, "function renderRequantify(")
    assert "requantifyReason(project)" in render and "REQUANTIFY_OFFERS[" in render
    _, shown = _function(app, "function showRequantified(")
    assert "before.background_method === LEGACY_BACKGROUND" in shown
    words = app[app.index("const ACTION_WORDS = {") :].split("\n};", 1)[0]
    assert 'requantify: "requantify",' in words


def test_every_request_of_the_page_goes_through_request():
    # request() names the opening of the project the page shows in every
    # request, so the server refuses one about a project no longer open: the
    # page calls the global fetch there only, and nothing else sends requests.
    found = {}
    for path in sorted(server.STATIC_DIR.glob("*.js")):
        script = _code(path.name)
        found[path.name] = [m.start() for m in re.finditer(r"(?<![\w.$])fetch\(", script)]
        for other in ("XMLHttpRequest", "sendBeacon", "EventSource", "WebSocket"):
            assert other not in script, (path.name, other)
    assert {name: len(at) for name, at in found.items() if at} == {"app.js": 1}
    start, request = _function(_code("app.js"), "async function request(")
    assert start < found["app.js"][0] < start + len(request)


def test_the_page_names_its_opening_and_follows_another_only_when_it_did_not_open_it():
    script = _code("app.js")
    assert f'const OPENING_HEADER = "{api.OPENING_HEADER}";' in script
    assert 'const PROJECT_CHANGED = "project_changed";' in script
    _, request = _function(script, "async function request(")
    assert "headers.set(OPENING_HEADER," in request
    assert "projectChanged(" in request.split("fetch(", 1)[1]  # on a refusal
    # Not while the page opens a project itself (its reads sent before come
    # back refused), nor about an opening no newer than the one it shows.
    _, rule = _function(script, "function projectChanged(")
    follow = rule.index("followOpening(")
    assert -1 < rule.find("opening !== null") < follow
    assert -1 < rule.find("open_id") < follow
    # Both a page's own create or open and a follow show the project as new.
    _, switch = _function(script, "async function switchTo(")
    assert switch.index("setOpening(name)") < switch.index('call("POST"')
    assert "showOpened(" in switch
    assert "showOpened(" in _function(script, "function followOpening(")[1]
    _, shown = _function(script, "function showOpened(")
    assert "state.originalColours.clear()" in shown and "charts.forget()" in shown


def test_a_page_shown_again_checks_which_project_is_open():
    script = _code("app.js")
    assert re.search(r'addEventListener\("visibilitychange",[^;]*checkOpening\(', script)
    assert re.search(r'window\.addEventListener\("focus",[^;]*checkOpening\(', script)
    assert "readWorkspace()" in _function(script, "function checkOpening(")[1]
    assert '"/api/workspace"' in _function(script, "function readWorkspace(")[1]
    # The shortcuts (single keys, and Ctrl+Z) are off under any open dialog.
    assert '$("projects-dialog").open' not in script
    assert script.count('document.querySelector("dialog[open]")') == 2


def _method(script: str, head: str) -> str:
    """The text of the class method that starts with ``head`` (indented two spaces)."""
    start = script.index(f"\n  {head}") + 1
    return script[start : script.index("\n  }\n", start) + 4]


def test_an_edit_made_while_another_project_was_shown_is_never_sent():
    # An edit waits for the edits made before it: the panel's queue, and the box
    # and image edits in flight. So one made while the page still showed a
    # project no longer open (Ctrl+Z, a band clicked) can wait, behind a
    # refusal answered late, past the moment the page shows the project open
    # now. Sent then, it would name that project's opening (request() names the
    # one shown when it sends), and what was made in the project before would
    # be made in this one. It is not sent: the panel drops the edits queued
    # before another project is shown, and a box or image edit checks that the
    # opening shown is still the one shown when it was made.
    script = _code("app.js")
    assert "proteinPanel.invalidateEdits()" in _function(script, "function showOpened(")[1]
    _, ordered = _function(script, "function ordered(")
    waits = ordered.index("proteinPanel.queue.then(")
    assert -1 < ordered.find("shownOpening()") < waits  # when it is made
    checked = ordered.index("shownOpening()", waits)  # when it would be sent
    assert checked < ordered.index("sendIt()")
    # The panel runs a queued edit, and sends an add, only while no other
    # project was asked for or shown since it was made (invalidateEdits).
    panel = _code("proteins.js")
    assert "return current() ? task(current) : null;" in _method(panel, "queueEdit(")
    add = _method(panel, "async add(")
    assert -1 < add.find("opening !== this.opening") < add.index("this.handlers.send(")


def test_a_refused_undo_or_redo_is_not_called_a_change():
    # A request refused as project_changed did nothing, and the status line says
    # what, after "A is saved". A refused undo or redo lost no change: the
    # change it would have taken back or made again was made, and is saved.
    script = _code("app.js")
    body = script[script.index("const NOT_DONE = {") :].split("};", 1)[0]
    not_done = dict(re.findall(r'(\w+): "([^"]*)"', body))
    assert not_done == {
        "change": "Your last change was not made.",
        "undo": "Nothing was undone.",
        "redo": "Nothing was redone.",
        "export": "Nothing was exported.",
        "diagnostics": "No diagnostic file was written.",
    }
    body = script[script.index("const REFUSED_AS = {") :].split("};", 1)[0]
    refused_as = dict(re.findall(r'"([^"]+)": "(\w+)"', body))
    assert refused_as == {
        "/api/undo": "undo",
        "/api/redo": "redo",
        "/api/export": "export",
        "/api/diagnostics": "diagnostics",
    }
    posts = {route.path for route in api.router.routes if "POST" in route.methods}
    assert set(refused_as) <= posts
    assert "REFUSED_AS[path]" in _function(script, "function projectChanged(")[1]


def test_an_answer_about_a_newer_opening_than_its_request_named_is_never_applied():
    # The server answers each request within the opening it names (#134's
    # review), and the page does not rely on it. Applied (applyAnswer), such an
    # answer would show the project opened again as a newer revision of the
    # one shown, keeping that one's previews, typed values and queued edits,
    # with no word that it was opened again. It is taken as a project_changed
    # refusal instead: the page follows (showOpened, and its message).
    script = _code("app.js")
    start, request = _function(script, "async function request(")
    assert "const named = anyProject ? null : shownOpening();" in request
    assert "headers.set(OPENING_HEADER, String(named));" in request
    checked = request.index("if (named && now > named)")
    assert -1 < request.find("answered.project.open_id") < checked
    follows = request.index("projectChanged(error, method, path, { answered: true });", checked)
    assert follows < request.index("throw error;", follows) < request.index("return answered;")
    # Every answer is read there: call() asks for it, and so does the read of
    # GET /api/workspace (about no project), so no JSON is read elsewhere.
    assert "answer: true" in _function(script, "async function call(")[1]
    outside = [
        m.start()
        for m in re.finditer(r"\.json\(\)", script)
        if not start < m.start() < start + len(request)
    ]
    assert outside == []
    assert "answer: true" in _function(script, "function readWorkspace(")[1]
    # The requests answered about another opening on purpose name none: a
    # create or open, and the follow's read of the project open now.
    assert "anyProject: true" in _function(script, "async function switchTo(")[1]
    assert "anyProject: true" in _function(script, "function followOpening(")[1]
    # Answered, the request was not refused: nothing is said to be not done.
    assert "answered ||" in _function(script, "function projectChanged(")[1]


# --- The Import dialog: images handed to Proteia (#57) ---


def _markup(html: str, start: str, end: str) -> str:
    """The markup of the page shell from ``start`` up to ``end``."""
    at = html.index(start)
    return html[at : html.index(end, at)]


class _Controls(HTMLParser):
    """The form controls of some markup, each with whether a label holds it, and
    each select's options as (value, selected, disabled)."""

    def __init__(self) -> None:
        super().__init__()
        self.labels = 0
        self.controls: list[tuple[str, dict[str, str | None], bool]] = []
        self.options: dict[str, list[tuple[str, bool, bool]]] = {}
        self._select: str | None = None

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if tag == "label":
            self.labels += 1
        elif tag in ("input", "select", "button"):
            self.controls.append((tag, fields, self.labels > 0))
            if tag == "select":
                self._select = fields.get("id") or fields.get("class")
                self.options[self._select] = []
        elif tag == "option" and self._select is not None:
            self.options[self._select].append(
                (fields.get("value", ""), "selected" in fields, "disabled" in fields)
            )

    def handle_endtag(self, tag):
        if tag == "label":
            self.labels -= 1
        elif tag == "select":
            self._select = None


def test_the_import_dialog_is_a_labelled_modal_dialog_whose_bands_start_unchosen():
    # A real dialog element, named by its heading; each image a group (a
    # fieldset named by its file) of labelled choices: the kinds of the page's
    # own import, and whether its bands are dark or light, required and
    # unchosen (a disabled, selected placeholder), since the model has no
    # silent default. Import starts disabled, and says why.
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    parser = _Tags()
    parser.feed(html)
    dialog = parser.by_id["handoff-dialog"]
    assert parser.by_id[dialog["aria-labelledby"]] is not None
    assert "<dialog" in _markup(html, '<dialog id="handoff-dialog"', ">")
    assert '<h2 id="handoff-title">' in html
    assert dialog["aria-describedby"] in parser.by_id
    template = _markup(html, '<template id="handoff-row">', "</template>")
    assert template.index("<fieldset") < template.index("<legend>") < template.index("<select")
    controls = _Controls()
    controls.feed(_markup(html, '<dialog id="handoff-dialog"', "</dialog>") + template)
    labelled = [labelled for tag, _, labelled in controls.controls if tag != "button"]
    assert labelled and all(labelled)  # each field and choice in a label
    polarity = controls.options["handoff-polarity"]
    assert polarity[0] == ("", True, True)
    assert {value for value, _, _ in polarity[1:]} == {p.value for p in Polarity}
    assert "required" in re.search(r'<select class="handoff-polarity"[^>]*>', template).group(0)
    assert controls.options["handoff-set-all-polarity"] == polarity  # "Set all" says the same
    page_import = _Controls()
    page_import.feed(_markup(html, '<select id="import-kind">', "</select>"))
    assert controls.options["handoff-kind"] == page_import.options["import-kind"]
    assert {value for value, _, _ in page_import.options["import-kind"]} == {
        k.value for k in ImageKind
    }
    assert controls.options["handoff-membrane"] == [("new", False, False)]
    button = parser.by_id["handoff-import"]
    assert (button["type"], button["aria-describedby"]) == ("submit", "handoff-needs")
    assert "disabled" in button
    assert (
        parser.by_id["handoff-discard"]["type"] == parser.by_id["handoff-later"]["type"] == "button"
    )


def test_the_import_dialog_never_chooses_a_polarity_the_user_did_not():
    # Whether an image's bands are dark or light is set only by the user: in
    # its own row, or by "Set all", which fills the rows listed when it is
    # pressed. A row added later (more files joined the hand-off) starts
    # unchosen, and Import waits for it; a row already shown keeps its choices.
    dialog = _code("handoffs.js")
    sets = [m.start() for m in re.finditer(r"polarityOf\(row\)\.value = ", dialog)]
    set_all = _method(dialog, "setAll(")
    at = dialog.index(set_all)
    assert len(sets) == 1 and at < sets[0] < at + len(set_all)
    assert "for (const row of this.rows.values())" in set_all
    assert "polarity" not in _method(dialog, "newRow(")
    fill = _method(dialog, "fill(")
    assert "this.rows.get(file.file_id)" in fill and "this.newRow(file)" in fill
    assert "this.missing() === 0" in _method(dialog, "ready(")
    assert '$("handoff-import").disabled = !this.ready();' in _method(dialog, "renderActions(")
    assert "polarity: polarityOf(row).value," in _method(dialog, "choices(")
    # The listing refreshes the rows, never while this page's own import or
    # discard of the hand-off is awaited (its own claim lists it as claimed).
    app = _code("app.js")
    _, take = _function(app, "function takeListing(")
    assert take.index("if (shown && !importDialog.busy)") < take.index("importDialog.refresh(now)")
    assert "importDialog.hide();" in take


def test_the_page_finds_images_waiting_at_start_on_focus_and_before_quitting():
    app = _code("app.js")
    _, start = _function(app, "async function start(")
    assert start.index("readWorkspace()") < start.index("takeListing(workspace)")
    assert "takeListing(workspace)" in _function(app, "function checkOpening(")[1]
    # Quit reads them first; with images waiting it shows them and does not stop.
    quit_press = app[app.index('$("quit").addEventListener("click"') :].split("\n});\n", 1)[0]
    shown = quit_press.index('takeListing(workspace, { show: "asked" });')
    assert quit_press.index("readWorkspace()") < shown < quit_press.index("return;", shown)
    assert shown < quit_press.index('request("POST", "/api/quit")')
    # While the hand-off shown may still grow, or another tab's import holds
    # it, the listing is read every 2 s; not while this page's own import or
    # discard of it is awaited.
    assert "const SETTLE_MS = 2000;" in app
    _, settle = _function(app, "function keepSettling(")
    assert "!(shown.more_may_arrive || shown.claimed)" in settle and "importDialog.busy" in settle
    assert "SETTLE_MS" in settle and "checkOpening()" in settle


def test_an_import_of_images_waiting_is_the_pages_own_switch_naming_what_it_closes():
    # #134's rules: an import opens a project as the Projects dialog does
    # (switchTo), so the page shows it with no word of another tab; and its
    # request names the opening the page shows, which it closes, so the server
    # refuses it if another project is open now; its answer, about the
    # opening it makes, is not taken for a newer one than the request named.
    app = _code("app.js")
    _, imports = _function(app, "async function importHandoff(")
    assert "switchTo(`/api/handoffs/${handoff.id}/accept`" in imports and "exporting" in imports
    # It names the opening the dialog says the import closes, no other.
    named = imports.index("const closes = importDialog.closes;")
    assert named < imports.index("closes: closes ? closes.open_id : null,")
    _, switch = _function(app, "async function switchTo(")
    assert "{ anyProject: true, closes }" in switch
    _, request = _function(app, "async function request(")
    assert request.index("} else if (closes) {") < request.index(
        "headers.set(OPENING_HEADER, String(closes));"
    )
    route = next(r for r in api.router.routes if r.path == "/api/handoffs/{handoff_id}/accept")
    assert route.methods == {"POST"}


def test_an_import_from_a_page_showing_no_project_names_the_project_open_now():
    # A page showing no project (its Import dialog up at start, say) closes by
    # an import whatever project another tab opened meanwhile. So what the
    # dialog says an import closes, and what its request names, is the project
    # shown or, with none shown, the one open as last listed; and such a page
    # reads the listing again before it sends, and asks again should a
    # project be open that the dialog did not name.
    app = _code("app.js")
    _, closes = _function(app, "function importCloses(")
    assert closes.index("if (state.project)") < closes.index("return listedOpen;")
    assert "open_id: shownOpening()" in closes
    _, take = _function(app, "function takeListing(")
    listed = take.index("listedOpen =")
    assert "{ name: workspace.open, open_id: workspace.open_id }" in take
    assert listed < take.index("importDialog.renderCloses(importCloses());")
    assert listed < take.index("importDialog.show(next, importCloses());")
    assert "importDialog.renderCloses(importCloses());" in _function(app, "function render(")[1]
    assert app.count("renderCloses(") == app.count("renderCloses(importCloses())")
    _, imports = _function(app, "async function importHandoff(")
    reread = imports.index("await checkAgain();")
    assert imports.index("if (!state.project) {") < reread
    assert reread < imports.index("askAgain(now.name);") < imports.index("switchTo(")
    assert "this.closes = opening;" in _method(_code("handoffs.js"), "renderCloses(")


def test_the_import_dialog_answers_each_refusal_of_an_import_or_a_discard():
    # Each refusal the server gives an import or a discard has its answer in
    # the page: the dialog closes once another tab imported or discarded the
    # hand-off, waits while another tab's import holds it, shows it grown when
    # more files joined, says a name clash next to the name, and follows
    # another tab's opening before asking again.
    app = _code("app.js")
    handled = _function(app, "async function importRefused(")[1]
    handled += _function(app, "function handoffRefused(")[1]
    codes = {
        "handoff_changed",
        "handoff_claimed",
        "handoff_not_found",
        "nothing_imported",
        "unsaved_changes",
        "project_exists",
        "invalid_project_name",
    }
    source = Path(api.__file__).read_text(encoding="utf-8")
    for code in codes:
        assert f'error.code === "{code}"' in handled, code
        assert f'"{code}"' in source, code  # an error the server answers
    assert "error.code === PROJECT_CHANGED" in handled
    assert "detail.created" in handled  # the images went into a project after all
    _, discard = _function(app, "async function discardHandoff(")
    assert "json: importDialog.shownParts()," in discard
    dialog = _code("handoffs.js")
    assert "refused: refusedCount(handoff)," in _method(dialog, "shownParts(")
    assert (
        "handoff.refused.length + handoff.more_refused"
        in _function(dialog, "function refusedCount(")[1]
    )


def test_the_import_dialog_waits_while_another_tab_imports_its_images():
    # The listing keeps a hand-off another tab's import holds (claimed), since
    # that import may be refused (a name taken, say) and let it go unchanged.
    # The dialog showing it keeps its rows and choices; Import and Discard
    # wait, and a line says why; the listing is read again meanwhile (as while
    # more may arrive), and once the claim is let go they can be pressed
    # again. Only once the listing no longer has it is it gone: the dialog
    # closes and says so. A hand-off held elsewhere is never shown anew,
    # counted as waiting, or taken to hold Quit back.
    dialog = _code("handoffs.js")
    assert 'const ELSEWHERE = "Another tab is importing these images.";' in dialog
    assert "return this.handoff !== null && this.handoff.claimed === true;" in _method(
        dialog, "elsewhere("
    )
    assert "!this.elsewhere()" in _method(dialog, "ready(")
    actions = _method(dialog, "renderActions(")
    assert '$("handoff-import").disabled = !this.ready();' in actions
    assert (
        '$("handoff-discard").disabled = !this.handoff || this.busy || this.elsewhere();' in actions
    )
    needs = re.compile(r'\$\("handoff-needs"\)\.textContent = this\.elsewhere\(\)\s*\? ELSEWHERE')
    assert needs.search(actions)
    assert "elsewhere" not in _method(dialog, "fill(")  # the rows are drawn as ever
    app = _code("app.js")
    _, take = _function(app, "function takeListing(")
    images = take.index('const images = listed.filter((handoff) => handoff.kind === "images");')
    waiting = take.index("handoffs = images.filter((handoff) => !handoff.claimed);")
    assert images < waiting
    assert "if (!images.some((handoff) => handoff.id === id)) {" in take  # set aside: kept
    now = take.index("const now = images.find((handoff) => handoff.id === shown.id);")
    gone = take.index("finished.add(shown.id);", now)
    assert gone < take.index("importDialog.hide();", gone) < take.index("gone = GONE;", gone)
    assert take.count("importDialog.hide();") == 1
    # Its own import or discard refused as another tab's import holds it.
    _, refused = _function(app, "function handoffRefused(")
    claimed = refused.index('if (error.code === "handoff_claimed") {')
    branch = refused[claimed : refused.index("} else if", claimed)]
    assert "importDialog.refresh({ ...importDialog.handoff, claimed: true" in branch
    assert "checkAgain();" in branch and "hide()" not in branch
    assert refused.count("importDialog.hide();") == 1  # handoff_not_found only
    not_found = refused.index('if (error.code === "handoff_not_found") {')
    assert not_found < refused.index("importDialog.hide();") < claimed
    assert "BEING_IMPORTED" not in app
    quit_press = app[app.index('$("quit").addEventListener("click"') :].split("\n});\n", 1)[0]
    assert "!handoff.claimed" in quit_press


def test_a_refused_discard_says_what_joined_the_images_since_they_were_listed():
    # A discard names the files and the "Not opened" entries shown, so it is
    # refused (handoff_changed) once either grew. The dialog says which grew:
    # images, to choose for, or only files a launch could not open, when "More
    # images arrived" would send the user looking for rows that did not come.
    refresh = _method(_code("handoffs.js"), "refresh(")
    before = refresh.index("const before = refusedCount(this.handoff);")
    assert before < refresh.index("return { images, refused: refusedCount(handoff) - before };")
    _, refused = _function(_code("app.js"), "function handoffRefused(")
    joined = refused.index("const joined = importDialog.refresh(error.detail);")
    images = refused.index("joined.images", joined)
    assert images < refused.index('"More images arrived: check them"', images)
    only = refused.index("joined.refused", images)
    assert only < refused.index('"More files could not be opened: check the list"', only)


def test_esc_closes_the_import_dialog_and_leaves_the_images_waiting():
    # Esc (or Later) only closes it, not while its import or discard is
    # awaited; the images keep waiting, and the header's (or the Projects
    # dialog's) "N images waiting…" shows them again, with the choices made.
    dialog = _code("handoffs.js")
    constructor = _method(dialog, "constructor(")
    cancel = constructor[constructor.index('addEventListener("cancel"') :]
    assert cancel.index("if (this.busy)") < cancel.index("event.preventDefault();")
    close = constructor[constructor.index('addEventListener("close"') :]
    assert "this.handlers.closed(handoff);" in close and "reset()" not in close
    assert '$("handoff-later").addEventListener("click", () => dialog.close());' in constructor
    app = _code("app.js")
    closed = app[app.index("closed: (handoff) => {") :].split("\n  },\n", 1)[0]
    assert "setAside.add(handoff.id);" in closed and "discard" not in closed
    for button in ("handoffs-waiting", "projects-handoffs"):
        assert f'$("{button}").addEventListener("click", () => showWaiting());' in app


def test_the_import_dialog_stays_up_while_its_import_or_discard_is_awaited():
    # A second Esc with no click between gives a cancel event the page cannot
    # stop (the browser's close watcher), so the key itself is stopped while
    # the answer is awaited; and a dialog closed all the same then comes back
    # up, to say how the import or discard went (a refusal is said there).
    constructor = _method(_code("handoffs.js"), "constructor(")
    keys = constructor[constructor.index('document.addEventListener("keydown"') :]
    stopped = keys.index('this.busy && dialog.open && event.key === "Escape"')
    assert stopped < keys.index("event.preventDefault();")
    close = constructor[constructor.index('addEventListener("close"') :]
    busy = close.index("if (this.busy) {")
    assert busy < close.index("dialog.showModal();") < close.index("this.handlers.closed(handoff);")


def test_a_launch_notice_is_said_then_discarded_as_said():
    # A notice (the arguments a launch could not open, and no image) is said
    # in the status line, then discarded with the count of entries said, so
    # one that grew meanwhile is refused and said whole next time.
    _, notices = _function(_code("app.js"), "function sayNotices(")
    assert notices.index("finished.add(notice.id);") < notices.index("/discard`")
    assert "json: { files: [], refused: notice.refused.length + notice.more_refused }" in notices
    changed = notices.index('error.code === "handoff_changed"')
    assert changed < notices.index("finished.delete(notice.id);") < notices.index("checkAgain();")
    assert "return `Not opened: " in notices


def test_a_notice_is_said_with_what_the_page_just_said_never_over_it():
    # The status line holds one message. A notice a listing finds is said
    # after what the page just said there (an import's result, the Import
    # dialog closed, another tab's project followed, Quit held back), never in
    # its place, nor put in its place later: the listing gives it to say
    # (takeListing), and each caller says it with its own message. It waits
    # while this page's own import or discard is awaited, whose answer comes
    # next: the check after that answer says both.
    app = _code("app.js")
    assert "showStatus(" not in _function(app, "function sayNotices(")[1]
    _, take = _function(app, "function takeListing(")
    assert "showStatus(" not in take and "importDialog.busy ? [] :" in take
    assert 'return [gone, sayNotices(notices)].filter(Boolean).join(" ") || null;' in take
    _, check = _function(app, "function checkOpening(")
    said = check.index('const said = [note, found].filter(Boolean).join(" ") || null;')
    assert said < check.index("showStatus(said);") < check.index("followOpening({ note: said })")
    for head in ("async function importHandoff(", "async function discardHandoff("):
        _, body = _function(app, head)
        assert body.index("showStatus(said);") < body.index("checkAgain(said);"), head
    quit_press = app[app.index('$("quit").addEventListener("click"') :].split("\n});\n", 1)[0]
    assert 'showStatus([WAITING_AT_QUIT, found].filter(Boolean).join(" "));' in quit_press
    for head in ("async function start(", "async function showWaiting("):
        _, body = _function(app, head)
        assert body.index("const found = takeListing(") < body.index("showStatus(found);"), head
    # Every listing taken says what it gives.
    calls = re.findall(r"(?<!function )takeListing\(", app)
    assert len(calls) == len(re.findall(r"const found = takeListing\(", app)) == 4


# --- The box size fields (#57) ---


def test_the_box_size_fields_show_and_send_the_fitted_size():
    # PUT /api/proteins/{id}/box-size takes the fitted size, which the boxes
    # extend beyond by the protein's padding. The fields show the fitted size
    # and Apply sends what they hold: filled with the box size, each Apply
    # would add the padding again. The boxes, and the placeholders of the
    # lanes with none, are drawn at the box size.
    panel = _code("proteins.js")
    editor = _method(panel, "renderEditor(")
    for field, dimension in (("box-width", "width"), ("box-height", "height")):
        assert f'this.fill("{field}", protein.fitted_size.{dimension},' in editor
    assert "box_size" not in panel
    assert "const { width, height } = protein.box_size;" in _code("app.js")


# --- The padding fields (#57) ---


def _form(html: str, form_id: str) -> str:
    """The markup of the form ``form_id`` in the page shell."""
    start = html.index(f'<form id="{form_id}"')
    return html[start : html.index("</form>", start)]


def test_the_padding_form_offers_above_and_below_first_in_whole_pixels():
    # The size fields read as the fitted size, which the boxes extend beyond by
    # the padding. The padding is set in whole pixels per side, above and below
    # first (where padding helps fold changes), then left and right, in one
    # form with its own Apply after the size's: Tab runs W, H, Apply, above
    # and below, left and right, Apply.
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    parser = _Tags()
    parser.feed(html)
    assert "Fitted size" in _form(html, "box-size")
    form = _form(html, "box-padding")
    assert html.index('<form id="box-size"') < html.index('<form id="box-padding"')
    assert -1 < form.index('id="pad-along"') < form.index('id="pad-across"') < form.index("submit")
    for field in ("pad-along", "pad-across"):
        tags = parser.by_id[field]
        assert (tags["type"], tags["min"], tags["step"]) == ("number", "0", "1"), field
        assert "required" in tags and f"{field}-share" in tags["aria-describedby"]
        assert "tabindex" not in tags
    assert "novalidate" not in form
    # The hint says to pad a target and its loading control alike, and the
    # partner line shows the other one's padding.
    assert "loading control alike" in html
    assert "pad-partners" in parser.by_id


def test_the_padding_fields_send_only_what_changed_and_wait_for_a_box():
    panel = _code("proteins.js")
    # Only the directions whose value differs from the stored padding are sent
    # (a padding set in another tab in the other direction stays). Through the
    # panel's queue, so the request names the opening shown and is dropped once
    # another project is shown (#134). The values are read when Apply is
    # pressed, but compared once the edits before it have their answers, with
    # the protein as the last answer stored it (editChosen's `request`): a
    # second Apply pressed before the first is answered is compared with what
    # the first set, not with the padding shown when it was pressed, so it is
    # neither dropped as "the same" nor sent without the directions it changed
    # back. With none different, the body is empty: the server's no-op answer.
    apply = _method(panel, "async applyPadding(")
    queued = apply.index("this.editChosen(")
    assert "shown.box_padding" not in apply
    assert queued < apply.index("(protein) => {") < apply.index("!== protein.box_padding[key]")
    assert queued < apply.index("before = this.project;")
    assert "Object.keys(body).length" not in apply
    assert apply.index("$(field).value") < queued
    assert "/box-padding`" in apply
    # Filled from the stored padding, as the size fields are; at most half the
    # fitted size or the stored value; disabled until the protein has a box.
    render = _method(panel, "renderPadding(")
    assert "this.fill(field, protein.box_padding[key], force(field))" in render
    assert "Math.max(Math.floor(protein.fitted_size[dimension] / 2), stored)" in render
    assert "const none = !protein.bands.length;" in render
    assert "renderPadding(" in _method(panel, "renderEditor(")


def test_a_box_at_the_image_edge_shows_no_fitted_outline():
    # A padding may shift a box at the image's edge inward, off its fit's
    # centre: an outline inset from the drawn box would lie off the fit, so
    # such a box shows none.
    script = (server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    block = script[script.index("const { across, along } = protein.box_padding;") :]
    block = block[: block.index("color,")]
    assert "x0 <= 0 || y0 <= 0 || x1 >= image.width || y1 >= image.height" in block
    assert "inset && !atEdge ?" in block
