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
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import BinaryIO, get_args

import pytest

import proteia
from proteia import samples
from proteia.core import mwcal, results, rowdetect
from proteia.core import operations as ops
from proteia.core.model import ImageKind, Polarity
from proteia.web import api, cli, handoff, launch, projects, server
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
    assert isinstance(instance, launch.Instance)
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
    assert isinstance(instance, launch.Instance)
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
    assert isinstance(instance, launch.Instance)
    try:
        info = json.loads((tmp_path / INSTANCE_FILE).read_text(encoding="utf-8"))
        assert info["token"] == instance.token
        assert instance.token in (tmp_path / REDIRECT_FILE).read_text(encoding="utf-8")
    finally:
        instance.close()


def test_a_second_launch_opens_the_running_instance(running, tmp_path):
    info = (tmp_path / INSTANCE_FILE).read_bytes()
    opener = Opener()
    assert launch.start(folder=tmp_path, opener=opener) == launch.Opened()
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
        assert launch.start(folder=tmp_path, opener=opener, wait=10) == launch.Opened()
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
    assert isinstance(instance, launch.Instance)
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
    taken = 0
    unread = ()

    def __init__(self) -> None:
        self.served = False

    def serve(self) -> None:
        self.served = True


def test_main_serves_without_printing_the_token(state, monkeypatch, capsys):
    fake = _FakeInstance()
    monkeypatch.setattr(launch, "start", lambda **kwargs: fake)
    assert launch.main([]) == 0
    assert fake.served
    out = capsys.readouterr().out
    assert "http://127.0.0.1:1234/" in out and fake.token not in out

    monkeypatch.setattr(launch, "start", lambda **kwargs: launch.Opened())
    assert launch.main([]) == 0
    assert "already running" in capsys.readouterr().out


def test_main_reports_an_instance_that_does_not_respond(state, monkeypatch, capsys):
    def refuse(**kwargs):
        raise launch.NotRespondingError("no answer")

    monkeypatch.setattr(launch, "start", refuse)
    assert launch.main([]) == 1
    console = capsys.readouterr()
    assert "does not respond" in console.out and not console.err  # printed once
    log = (state / "logs" / "proteia.log").read_text(encoding="utf-8")
    assert "ERROR proteia.web.launch: Proteia is already running but does not respond" in log


def test_main_prints_a_non_ascii_path_to_a_narrow_console(state, monkeypatch):
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(launch, "start", lambda **kwargs: _FakeInstance())
    assert launch.main([]) == 0
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


# --- Images named on the command line (#57, N3) ---

START = launch.start  # the real one: tests put wrappers of it in its place


@pytest.fixture
def served(tmp_path):
    """A running instance with its state folder and projects root in ``tmp_path``."""
    workspace = api.Workspace(tmp_path / "projects", reveal=lambda folder: None)
    opener = Opener()
    instance = launch.start(folder=tmp_path / "state", opener=opener, workspace=workspace)
    thread = threading.Thread(target=instance.serve, daemon=True)
    thread.start()
    wait_until_started(instance, thread)
    yield Running(instance, opener, thread)
    instance.stop()
    thread.join(10)


def scans(folder: Path, *names: str) -> list[Path]:
    """Files that pass for images (the launcher never decodes one), each with
    bytes of its own."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / name for name in names]
    for path in paths:
        path.write_bytes(f"pixels of {path.name}".encode())
    return paths


def command(*paths: Path | str) -> cli.CommandLine:
    return cli.parse([str(path) for path in paths])


def waiting(instance: launch.Instance) -> list[list[str]]:
    """The names of the files in each pending hand-off."""
    return [[file.name for file in view.files] for view in instance.workspace.inbox.listing()]


def staged_bytes(instance: launch.Instance) -> list[bytes]:
    """What the staging folder holds, each under a name the server made."""
    folder = instance.workspace.inbox.folder
    assert folder is not None
    paths = list(folder.iterdir()) if folder.is_dir() else []
    assert all(re.fullmatch(r"[0-9a-f]{32}", path.name) for path in paths)
    return sorted(path.read_bytes() for path in paths)


def call(running: Running, method: str, path: str, body: object = None) -> tuple[int, object]:
    """One request with the token (and a JSON body, if given): the status and the
    answer's JSON, if any."""
    conn = http.client.HTTPConnection("127.0.0.1", running.port, timeout=30)
    headers = {"Authorization": f"Bearer {running.token}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    try:
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        payload = response.read()
    finally:
        conn.close()
    return response.status, json.loads(payload) if payload else None


def workspace_of(running: Running) -> dict:
    status, answer = call(running, "GET", "/api/workspace")
    assert status == 200 and isinstance(answer, dict)
    return answer


def test_a_first_launch_hands_its_own_files_to_its_page_before_the_browser_opens(tmp_path):
    blot, marker = scans(tmp_path / "scans µ", "β-actin 10 µM.tif", "marker α.tif")
    workspace = api.Workspace(tmp_path / "projects", reveal=lambda folder: None)
    seen: list[list[list[str]]] = []

    def opener(url: str) -> bool:
        seen.append([[f.name for f in view.files] for view in workspace.inbox.listing()])
        return True

    instance = launch.start(
        folder=tmp_path / "state",
        opener=opener,
        workspace=workspace,
        command_line=command(blot, marker, tmp_path / "missing.tif"),
    )
    thread = threading.Thread(target=instance.serve, daemon=True)
    thread.start()
    try:
        wait_until_started(instance, thread)
        assert seen == [[["β-actin 10 µM.tif", "marker α.tif"]]]  # before the browser opened
        assert instance.taken == 2
        (listed,) = workspace_of(Running(instance, Opener(), thread))["handoffs"]
        files = [(file["name"], file["size"]) for file in listed["files"]]
        assert files == [(path.name, path.stat().st_size) for path in (blot, marker)]
        assert listed["suggested_name"] == "β-actin 10 µM"
        assert listed["refused"] == [
            {"name": "missing.tif", "code": "missing", "message": "no such file or folder"}
        ]
        # Nothing is created before the page imports, and nothing is copied.
        assert not (tmp_path / "projects").exists()
        assert not (tmp_path / "state" / "incoming").exists()
    finally:
        instance.stop()
        thread.join(10)
    assert blot.read_bytes() == f"pixels of {blot.name}".encode()


def test_a_first_launch_whose_paths_are_all_refused_shows_a_notice(tmp_path):
    folder = tmp_path / "Blot"
    folder.mkdir()
    workspace = api.Workspace(tmp_path / "projects", reveal=lambda folder: None)
    instance = launch.start(
        folder=tmp_path / "state",
        opener=Opener(),
        workspace=workspace,
        command_line=command(folder, tmp_path / "photo.bmp"),
    )
    try:
        (view,) = workspace.inbox.listing()
        assert view.kind == "notice" and view.files == ()
        assert [(r.name, r.code) for r in view.refused] == [
            ("Blot", "folder"),
            ("photo.bmp", "missing"),
        ]
        assert instance.taken == 0
    finally:
        instance.close()


def test_a_second_launch_uploads_its_files_then_opens_a_tab(served, tmp_path, monkeypatch):
    served.instance.workspace.create("Open µ")
    before = workspace_of(served)
    paths = scans(tmp_path / "scans α β", "blot 1 µ.tif", "marker α.png")
    monkeypatch.chdir(tmp_path / "scans α β")
    command_line = cli.parse(["blot 1 µ.tif", "marker α.png"])  # resolved here, now
    monkeypatch.chdir(tmp_path)
    seen: list[list[list[str]]] = []

    def opener(url: str) -> bool:
        seen.append(waiting(served.instance))
        return True

    opened = launch.start(folder=tmp_path / "state", opener=opener, command_line=command_line)
    assert isinstance(opened, launch.Opened)
    assert [file.name for file in opened.files] == ["blot 1 µ.tif", "marker α.png"]
    assert (opened.refused, opened.merged, opened.waiting) == ((), False, 2)
    assert seen == [[["blot 1 µ.tif", "marker α.png"]]]  # the tab opened after the offer
    assert staged_bytes(served.instance) == sorted(path.read_bytes() for path in paths)
    after = workspace_of(served)
    assert (after["open"], after["open_id"]) == (before["open"], before["open_id"]) == ("Open µ", 1)
    assert served.opener.urls == [(tmp_path / "state" / REDIRECT_FILE).as_uri()]  # its own


def test_a_launch_whose_files_join_a_young_hand_off_opens_no_tab(served, tmp_path):
    a, b = scans(tmp_path, "a.tif", "b.tif")
    first, second = Opener(), Opener()
    state = tmp_path / "state"
    assert not launch.start(folder=state, opener=first, command_line=command(a)).merged
    opened = launch.start(folder=state, opener=second, command_line=command(b))
    assert (opened.merged, opened.waiting) == (True, 2)
    assert (len(first.urls), second.urls) == (1, [])
    assert waiting(served.instance) == [["a.tif", "b.tif"]]


def _at_once(count: int, run: Callable[[int], object], timeout: float = 60) -> list[object]:
    """``run(n)`` in ``count`` threads that start together: what each returned
    (or raised), by ``n``."""
    barrier = threading.Barrier(count)
    results: dict[int, object] = {}

    def one(n: int) -> None:
        barrier.wait()
        try:
            results[n] = run(n)
        except BaseException as exc:
            results[n] = exc

    threads = [threading.Thread(target=one, args=(n,), daemon=True) for n in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)
    assert len(results) == count, "a launch did not end"
    return [results[n] for n in range(count)]


def test_launches_at_once_join_one_hand_off_and_open_one_tab(served, tmp_path):
    paths = scans(tmp_path / "selection", *(f"blot {n}.tif" for n in range(5)))
    opener = Opener()
    results = _at_once(
        5,
        lambda n: launch.start(
            folder=tmp_path / "state", opener=opener, command_line=command(paths[n])
        ),
    )
    assert all(isinstance(result, launch.Opened) for result in results), results
    assert sorted(result.merged for result in results) == [False, True, True, True, True]
    assert len(opener.urls) == 1
    (names,) = waiting(served.instance)
    assert sorted(names) == [path.name for path in paths]


def test_launches_at_once_with_none_running_start_one_instance_and_one_hand_off(tmp_path):
    # Explorer may run one process per file of a selection opened with Proteia.
    paths = scans(tmp_path / "selection", *(f"blot {n}.tif" for n in range(4)))
    workspace = api.Workspace(tmp_path / "projects", reveal=lambda folder: None)
    opener = Opener()
    serving: list[tuple[launch.Instance, threading.Thread]] = []

    def run(n: int) -> object:
        result = launch.start(
            folder=tmp_path / "state",
            opener=opener,
            workspace=workspace,
            command_line=command(paths[n]),
        )
        if isinstance(result, launch.Instance):
            thread = threading.Thread(target=result.serve, daemon=True)
            serving.append((result, thread))
            thread.start()
        return result

    try:
        results = _at_once(4, run)
        listed = [[file.name for file in view.files] for view in workspace.inbox.listing()]
    finally:
        for instance, thread in serving:
            instance.stop()
            thread.join(10)
    kinds = sorted(type(result).__name__ for result in results)
    assert kinds == ["Instance", "Opened", "Opened", "Opened"], results
    assert all(result.merged for result in results if isinstance(result, launch.Opened))
    assert len(opener.urls) == 1
    (names,) = listed
    assert sorted(names) == [path.name for path in paths]


class _Read:
    """A file as a launch reads it to upload it, counting the bytes read
    (``read_bytes``). Its first read waits ``pause`` seconds (an upload that takes
    long), and each later one ``stall`` seconds (a file on a slow network
    share)."""

    def __init__(self, stream: BinaryIO, *, pause: float = 0.0, stall: float = 0.0) -> None:
        self._stream = stream
        self._pause = pause
        self._stall = stall
        self._reads = 0
        self.read_bytes = 0

    def fileno(self) -> int:
        return self._stream.fileno()

    def read(self, size: int = -1) -> bytes:
        wait = self._stall if self._reads else self._pause
        if wait:
            time.sleep(wait)
        self._reads += 1
        data = self._stream.read(size)
        self.read_bytes += len(data)
        return data

    def __enter__(self) -> _Read:
        return self

    def __exit__(self, *exc: object) -> None:
        self._stream.close()


def reading(monkeypatch, **waits: dict[str, float]) -> dict[str, _Read]:
    """Make each file a launch opens a :class:`_Read`, with the waits given by
    its name (``pause={"b.tif": 2.5}``): the files opened, by name."""
    opened: dict[str, _Read] = {}
    plain = cli.ImageFile.open

    def open_read(file: cli.ImageFile) -> _Read:
        chosen = {kind: by_name.get(file.name, 0.0) for kind, by_name in waits.items()}
        opened[file.name] = _Read(plain(file), **chosen)
        return opened[file.name]

    monkeypatch.setattr(cli.ImageFile, "open", open_read)
    return opened


def test_a_file_whose_upload_began_in_the_window_joins_however_long_it_took(
    served, tmp_path, monkeypatch
):
    monkeypatch.setattr(handoff, "MERGE_WINDOW_S", 2.0)
    a, b = scans(tmp_path, "a.tif", "b.tif")
    state = tmp_path / "state"
    assert not launch.start(folder=state, opener=Opener(), command_line=command(a)).merged
    reading(monkeypatch, pause={"b.tif": 2.5})
    began = time.monotonic()
    late = launch.start(folder=state, opener=Opener(), command_line=command(b))
    assert time.monotonic() - began > 2.0  # offered once the window had passed
    assert late.merged and waiting(served.instance) == [["a.tif", "b.tif"]]


def test_a_second_launch_hands_off_to_an_instance_still_starting(served, tmp_path):
    path = tmp_path / "state" / INSTANCE_FILE
    info = path.read_bytes()
    path.unlink()  # as if the running instance had not written it yet
    timer = threading.Timer(0.5, path.write_bytes, args=(info,))
    timer.start()
    (blot,) = scans(tmp_path, "blot.tif")
    opener = Opener()
    try:
        opened = launch.start(
            folder=tmp_path / "state", opener=opener, wait=10, command_line=command(blot)
        )
    finally:
        timer.join()
    assert [file.name for file in opened.files] == ["blot.tif"] and len(opener.urls) == 1
    assert waiting(served.instance) == [["blot.tif"]]


def test_a_launch_hands_nothing_to_an_instance_that_does_not_answer(tmp_path):
    held = launch.InstanceLock.acquire(tmp_path)
    info = {"pid": 1, "port": _closed_port(), "token": "x" * 43}
    (tmp_path / INSTANCE_FILE).write_text(json.dumps(info), encoding="utf-8")
    (blot,) = scans(tmp_path / "scans", "blot.tif")
    opener = Opener()
    try:
        with pytest.raises(launch.NotRespondingError):
            launch.start(folder=tmp_path, opener=opener, wait=0.5, command_line=command(blot))
    finally:
        held.release()
    assert opener.urls == [] and not (tmp_path / "incoming").exists()


def _older(monkeypatch, version: object = OMIT) -> None:
    """Make the running instance answer as a Proteia that takes no files (no
    ``handoff`` in its status), or takes them another way (``handoff`` is
    ``version``)."""
    probe = launch.probe

    def older(info: launch.InstanceInfo, **kwargs: float) -> dict | None:
        status = probe(info, **kwargs)
        if status is None:
            return None
        status = {k: v for k, v in status.items() if k != "handoff"}
        return status if version is OMIT else {**status, "handoff": version}

    monkeypatch.setattr(launch, "probe", older)


@pytest.mark.parametrize(
    ("version", "which"), [(OMIT, "an older"), (2, "another"), ("1", "another")]
)
def test_an_older_running_proteia_is_opened_and_handed_nothing(
    served, tmp_path, monkeypatch, version, which
):
    _older(monkeypatch, version)
    (blot,) = scans(tmp_path, "blot.tif")
    opener = Opener()
    with pytest.raises(
        launch.HandoffError, match=f"^The running Proteia is {which} version"
    ) as refused:
        launch.start(folder=tmp_path / "state", opener=opener, command_line=command(blot))
    assert refused.value.opened and len(opener.urls) == 1  # the user sees the running app
    assert waiting(served.instance) == [] and staged_bytes(served.instance) == []
    # With no paths to hand over, it is opened as ever.
    assert launch.start(folder=tmp_path / "state", opener=opener) == launch.Opened()


def test_a_refused_path_reaches_the_page_as_a_bounded_name(served, tmp_path):
    name = "x" * 100 + "\x01" + "y" * 100 + ".tif"  # a control character, 205 characters
    long = tmp_path / ("d" * 150) / name
    surrogate = tmp_path / "\udc80gone.tif"  # an undecodable name, as argv may hold one
    (blot,) = scans(tmp_path, "blot.tif")
    opened = launch.start(
        folder=tmp_path / "state", opener=Opener(), command_line=command(blot, long, surrogate)
    )
    assert [path.shown for path in opened.refused] == [
        str(long.resolve()),
        str(surrogate.resolve()),
    ]
    (listed,) = workspace_of(served)["handoffs"]
    assert [file["name"] for file in listed["files"]] == ["blot.tif"]
    bounded, replaced = (entry["name"] for entry in listed["refused"])
    assert len(bounded) == handoff.MAX_REFUSED_NAME and bounded.endswith("…")
    assert bounded.startswith("x" * 100 + "�")
    assert replaced == "�gone.tif"


def _second(monkeypatch, tmp_path: Path, opener: Opener) -> None:
    """:func:`launch.main` launches with the state folder in ``tmp_path``, where
    ``served`` runs, the projects root there too, and ``opener`` for a browser."""
    monkeypatch.setattr(launch, "state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(projects, "projects_root", lambda: tmp_path / "projects")
    monkeypatch.setattr(launch, "start", lambda **kwargs: START(opener=opener, **kwargs))


def test_a_file_the_running_proteia_refuses_is_reported_and_the_rest_handed_off(
    served, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(handoff, "MAX_PENDING_FILES", 1)
    a, b = scans(tmp_path / "scans µ", "a.tif", "b.tif")
    opener = Opener()
    _second(monkeypatch, tmp_path, opener)
    assert launch.main([str(a), str(b)]) == 3
    console = capsys.readouterr()
    assert console.out.splitlines() == [
        "Proteia is already running; it has been opened in your browser.",
        "1 image is waiting there: choose how to import them.",
    ]
    message = "1 images are waiting in Proteia: import or discard them first"
    assert console.err.splitlines() == [f"Not opened: {b.resolve()} ({message})"]
    (listed,) = workspace_of(served)["handoffs"]
    assert [file["name"] for file in listed["files"]] == ["a.tif"]
    assert [(r["name"], r["code"]) for r in listed["refused"]] == [("b.tif", "too_many_pending")]
    assert len(opener.urls) == 1 and served.token not in console.out + console.err


MIB = 1024 * 1024


def large_scans(folder: Path, *names: str, size: int) -> list[Path]:
    """Files of ``size`` bytes, each with bytes of its own: more than a
    connection's buffers hold, so an upload is sent while the server reads it."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / name for name in names]
    for path in paths:
        mark = f"pixels of {path.name} ".encode()
        path.write_bytes((mark * (size // len(mark) + 1))[:size])
    return paths


def test_files_the_running_proteia_has_no_room_for_are_refused_before_their_bytes_are_sent(
    served, tmp_path, monkeypatch, capsys
):
    # Images waiting in Proteia, and more opened than it has room for (20 and
    # 16 of at most 32, in small): the launch asks before it sends each file's
    # bytes (GET /api/incoming/room), so those it has no room for are refused
    # without being read, however large, and the others are handed off.
    monkeypatch.setattr(handoff, "MAX_PENDING_FILES", 3)
    opener = Opener()
    _second(monkeypatch, tmp_path, opener)
    earlier = scans(tmp_path / "earlier", "w1.tif", "w2.tif")
    assert launch.main([str(path) for path in earlier]) == 0
    capsys.readouterr()
    a, b, c = large_scans(tmp_path / "scans µ", "a.tif", "b.tif", "c.tif", size=8 * MIB)
    opened = reading(monkeypatch)
    assert launch.main([str(a), str(b), str(c)]) == 3
    message = "3 images are waiting in Proteia: import or discard them first"
    assert capsys.readouterr().err.splitlines() == [
        f"Not opened: {path.resolve()} ({message})" for path in (b, c)
    ]
    assert {name: file.read_bytes for name, file in opened.items()} == {
        "a.tif": 8 * MIB,
        "b.tif": 0,
        "c.tif": 0,
    }
    (listed,) = workspace_of(served)["handoffs"]
    assert [file["name"] for file in listed["files"]] == ["w1.tif", "w2.tif", "a.tif"]
    assert [(r["name"], r["code"]) for r in listed["refused"]] == [
        ("b.tif", "too_many_pending"),
        ("c.tif", "too_many_pending"),
    ]
    # Nothing is staged but the files handed off.
    assert staged_bytes(served.instance) == sorted(path.read_bytes() for path in (*earlier, a))
    assert len(opener.urls) == 1  # the first launch's: the second joined its hand-off


def test_an_upload_cut_short_refuses_that_file_and_the_others_are_handed_off(
    served, tmp_path, monkeypatch, capsys
):
    # Proteia refuses b.tif while its bytes arrive (the room ran out after the
    # launch asked), then closes the connection, as uvicorn does once no more
    # of a body it has answered arrives for its keep-alive time: 5 s, here 0.1
    # s, with b.tif on a share whose reads stall 0.5 s. The launch may find the
    # connection reset before it can read the answer. It reports b.tif not
    # taken, with the reason Proteia gives when asked again, hands off the
    # others, and leaves nothing staged but them.
    monkeypatch.setattr(served.instance.server.config, "timeout_keep_alive", 0.1)
    a, b = large_scans(tmp_path / "scans µ", "a.tif", "b.tif", size=4 * MIB)
    (c,) = scans(tmp_path / "scans µ", "c.tif")
    inbox = served.instance.workspace.inbox
    make_room = inbox.make_room
    no_room = (
        "the images waiting in Proteia take all the room it keeps for them: import or discard"
        " them first"
    )

    def room_runs_out(upload: handoff.Upload, size: int) -> None:
        if upload.name == "b.tif":  # room left for c.tif, not for b.tif
            monkeypatch.setattr(handoff, "MAX_STAGED_BYTES", 4 * MIB + 1024)
            raise handoff.TooManyPendingError(no_room)
        make_room(upload, size)

    monkeypatch.setattr(inbox, "make_room", room_runs_out)
    reading(monkeypatch, stall={"b.tif": 0.5})
    opener = Opener()
    _second(monkeypatch, tmp_path, opener)
    assert launch.main([str(a), str(b), str(c)]) == 3
    console = capsys.readouterr()
    assert "2 images are waiting there: choose how to import them." in console.out
    assert console.err.splitlines() == [f"Not opened: {b.resolve()} ({no_room})"]
    (listed,) = workspace_of(served)["handoffs"]
    assert [file["name"] for file in listed["files"]] == ["a.tif", "c.tif"]
    assert [(r["name"], r["code"]) for r in listed["refused"]] == [("b.tif", "too_many_pending")]
    assert staged_bytes(served.instance) == sorted(path.read_bytes() for path in (a, c))
    assert len(opener.urls) == 1


def test_a_launch_that_joined_a_hand_off_says_so(served, tmp_path, monkeypatch, capsys):
    a, b, c = scans(tmp_path, "a.tif", "b.tif", "c.tif")
    opener = Opener()
    _second(monkeypatch, tmp_path, opener)
    assert launch.main([str(a)]) == 0
    capsys.readouterr()
    assert launch.main([str(b), str(c)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Proteia is already running, and open in your browser.",
        "Added b.tif and c.tif to the images waiting there: choose how to import them.",
    ]
    assert len(opener.urls) == 1
    log = (tmp_path / "state" / "logs" / "proteia.log").read_text(encoding="utf-8")
    assert "2 images joined those another launch handed to it" in log
    assert "b.tif" not in log  # no path, nor name


def test_main_exits_1_when_an_older_proteia_runs(served, tmp_path, monkeypatch, capsys):
    _older(monkeypatch)
    opener = Opener()
    _second(monkeypatch, tmp_path, opener)
    missing = tmp_path / "missing.tif"
    assert launch.main([str(missing)]) == 1
    console = capsys.readouterr()
    assert console.out.splitlines() == [
        "Proteia is already running; it has been opened in your browser."
    ]
    assert console.err.splitlines() == [
        "The running Proteia is an older version and cannot take files; quit it and try again.",
        f"Not opened: {missing.resolve()} (no such file or folder)",
    ]
    assert len(opener.urls) == 1 and workspace_of(served)["handoffs"] == []


@pytest.mark.parametrize("where", ["upload", "offer"])
def test_main_exits_1_when_the_hand_off_fails(served, tmp_path, monkeypatch, capsys, where):
    def stopping(*args: object, **kwargs: object) -> None:
        raise handoff.StoppingError("Proteia is stopping")

    inbox = served.instance.workspace.inbox
    monkeypatch.setattr(inbox, "begin_upload" if where == "upload" else "offer", stopping)
    (blot,) = scans(tmp_path, "blot.tif")
    opener = Opener()
    _second(monkeypatch, tmp_path, opener)
    assert launch.main([str(blot)]) == 1
    console = capsys.readouterr()
    said = {
        "upload": "Proteia is stopping; the files were not handed to it.",
        "offer": "Proteia did not take the files: Proteia is stopping",
    }[where]
    assert (console.out, console.err.splitlines()) == ("", [said])
    assert opener.urls == [] and waiting(served.instance) == []
    log = (tmp_path / "state" / "logs" / "proteia.log").read_text(encoding="utf-8")
    assert f"ERROR proteia.web.launch: {said}" in log


def test_a_connection_that_fails_ends_the_hand_off():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def hang_up() -> None:
        conn, _ = listener.accept()
        conn.close()

    thread = threading.Thread(target=hang_up, daemon=True)
    thread.start()
    info = launch.InstanceInfo(pid=1, port=listener.getsockname()[1], token="x" * 43)
    try:
        with pytest.raises(launch.HandoffError, match="The connection to the running Proteia"):
            launch._send(info, "POST", "/api/handoffs", b"{}", {})
    finally:
        thread.join(10)
        listener.close()
    with pytest.raises(launch.HandoffError):  # nothing listens there now
        launch._send(info, "POST", "/api/handoffs", b"{}", {})


@dataclass
class _Answer:
    status: int
    body: bytes

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


class _CutShort:
    """Stands in for ``http.client.HTTPConnection``: a request that connects,
    then fails as one fails whose server answered it and closed the connection
    before its body was sent whole; that answer can still be read (``answer``),
    or cannot (None)."""

    def __init__(self, answer: _Answer | None) -> None:
        self.answer = answer
        self.sock: object = None

    def __call__(self, host: str, port: int, **kwargs: object) -> _CutShort:
        return self

    def request(self, method: str, path: str, **kwargs: object) -> None:
        self.sock = object()  # connected
        raise ConnectionResetError("the body was cut short")

    def getresponse(self) -> _Answer:
        if self.answer is None:
            raise ConnectionResetError("no answer to read")
        return self.answer

    def close(self) -> None:
        pass


def test_an_answer_sent_before_the_body_was_read_is_taken_as_the_answer(monkeypatch):
    info = launch.InstanceInfo(pid=1, port=9, token="x" * 43)
    body = b"pixels" * 1000
    refused = {"code": "too_large", "message": "the file is too large"}
    answered = _CutShort(_Answer(413, json.dumps(refused).encode()))
    monkeypatch.setattr(launch.http.client, "HTTPConnection", answered)
    assert launch._send(info, "POST", "/api/incoming?name=a.tif", body, {}) == (413, refused)
    # With no answer to read, the connection failed as the request said.
    monkeypatch.setattr(launch.http.client, "HTTPConnection", _CutShort(None))
    with pytest.raises(launch.HandoffError, match="failed.*: the body was cut short$") as failed:
        launch._send(info, "POST", "/api/incoming?name=a.tif", body, {})
    assert isinstance(failed.value.__cause__, ConnectionResetError)


class _Played:
    """Stands in for ``http.client.HTTPConnection``: each connection made plays
    the next of ``plays``: an answer (:class:`_Answer`); ``"cut"``, a request
    that connects, then is reset while its body is sent, with no answer to
    read; or ``"refused"``, one that finds nothing listening. ``requests``
    holds each request's method and path, and whether it had a body."""

    def __init__(self, *plays: _Answer | str) -> None:
        self.plays = list(plays)
        self.requests: list[tuple[str, str, bool]] = []
        self.sock: object = None
        self._play: _Answer | str = "refused"

    def __call__(self, host: str, port: int, **kwargs: object) -> _Played:
        self._play = self.plays.pop(0)
        self.sock = None
        return self

    def request(self, method: str, path: str, body: object = None, **kwargs: object) -> None:
        self.requests.append((method, path, body is not None))
        if self._play == "refused":
            raise ConnectionRefusedError("nothing listens there")
        self.sock = object()  # connected
        if self._play == "cut":
            raise ConnectionAbortedError("the connection was reset")

    def getresponse(self) -> _Answer:
        if isinstance(self._play, str):
            raise ConnectionAbortedError("no answer to read")
        return self._play

    def close(self) -> None:
        pass


def _json(status: int, **answer: str) -> _Answer:
    return _Answer(status, json.dumps(answer).encode() if answer else b"")


@pytest.mark.parametrize(
    ("again", "refused"),
    [
        (
            _json(409, code="too_many_pending", message="32 images are waiting in Proteia"),
            ("too_many_pending", "32 images are waiting in Proteia"),
        ),
        (_json(204), ("other", "the upload was cut short: the connection was reset")),
    ],
    ids=["no room now", "room"],
)
def test_an_upload_cut_short_is_refused_with_the_reason_proteia_gives_when_asked_again(
    tmp_path, monkeypatch, caplog, again, refused
):
    (blot,) = scans(tmp_path / "scans µ", "blot µ.tif")
    (file,) = command(blot).files
    played = _Played(_json(204), "cut", again)
    monkeypatch.setattr(launch.http.client, "HTTPConnection", played)
    info = launch.InstanceInfo(pid=1, port=9, token="x" * 43)
    caplog.set_level("INFO", logger=launch.__name__)
    uploaded = launch._upload(info, file)
    assert "an upload was cut short: the connection was reset" in caplog.text
    assert "blot" not in caplog.text  # no path, nor name
    assert isinstance(uploaded, cli.RefusedPath)
    assert (uploaded.shown, uploaded.name) == (str(blot.resolve()), "blot µ.tif")
    assert (uploaded.code, uploaded.message) == refused
    room = f"/api/incoming/room?name=blot%20%C2%B5.tif&size={blot.stat().st_size}"
    assert played.requests == [
        ("GET", room, False),
        ("POST", "/api/incoming?name=blot%20%C2%B5.tif", True),
        ("GET", room, False),
    ]


@pytest.mark.parametrize(
    ("again", "said"),
    [
        (_json(409, code="stopping", message="Proteia is stopping"), "^Proteia is stopping;"),
        ("refused", "^The connection to the running Proteia failed"),
    ],
    ids=["stopping", "gone"],
)
def test_an_upload_cut_short_by_a_proteia_that_stopped_ends_the_hand_off(
    tmp_path, monkeypatch, again, said
):
    (blot,) = scans(tmp_path, "blot.tif")
    (file,) = command(blot).files
    monkeypatch.setattr(launch.http.client, "HTTPConnection", _Played(_json(204), "cut", again))
    info = launch.InstanceInfo(pid=1, port=9, token="x" * 43)
    with pytest.raises(launch.HandoffError, match=said):
        launch._upload(info, file)


def test_a_file_proteia_has_no_room_for_is_refused_unread(tmp_path, monkeypatch):
    (blot,) = scans(tmp_path, "blot.tif")
    (file,) = command(blot).files
    no_room = _json(409, code="too_many_pending", message="no room")
    played = _Played(no_room)
    monkeypatch.setattr(launch.http.client, "HTTPConnection", played)
    opened = reading(monkeypatch)
    info = launch.InstanceInfo(pid=1, port=9, token="x" * 43)
    uploaded = launch._upload(info, file)
    assert isinstance(uploaded, cli.RefusedPath)
    assert (uploaded.code, uploaded.message) == ("too_many_pending", "no room")
    assert [method for method, _, _ in played.requests] == ["GET"]
    assert opened["blot.tif"].read_bytes == 0


def _first(
    monkeypatch, tmp_path: Path, argv: list[str], look: Callable[[Running], None]
) -> tuple[int, list[str]]:
    """Run :func:`launch.main` with ``argv`` as a first launch, with its state
    folder and projects root in ``tmp_path``, serving in this thread until
    ``look`` (in another) has looked and the page quits: its exit status, and
    the URLs the browser was given."""
    workspace = api.Workspace(tmp_path / "projects", reveal=lambda folder: None)
    opener = Opener()
    started: list[launch.Instance] = []
    failed: list[BaseException] = []

    def start_here(**kwargs: object) -> launch.Instance | launch.Opened:
        result = START(opener=opener, workspace=workspace, **kwargs)
        if isinstance(result, launch.Instance):
            started.append(result)
        return result

    def look_then_quit() -> None:
        deadline = time.monotonic() + 10
        while not started or not started[0].server.started:
            if time.monotonic() > deadline:
                return
            time.sleep(0.01)
        instance = started[0]
        try:
            look(Running(instance, opener, threading.main_thread()))
        except BaseException as exc:
            failed.append(exc)
        if send(instance.port, "POST", "/api/quit", token=instance.token)[0] != 202:
            instance.stop()

    monkeypatch.setattr(launch, "state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(projects, "projects_root", lambda: tmp_path / "projects")
    monkeypatch.setattr(launch, "start", start_here)
    helper = threading.Thread(target=look_then_quit, daemon=True)
    helper.start()
    code = launch.main(argv)
    helper.join(10)
    assert not failed, failed
    return code, opener.urls


def test_main_exit_codes_of_a_first_launch(tmp_path, monkeypatch, capsys):
    blot, marker = scans(tmp_path / "scans µ", "blot µ.tif", "marker α.tif")
    photo = tmp_path / "scans µ" / "photo.bmp"
    photo.write_bytes(b"BM")
    seen: list[dict] = []

    def look(running: Running) -> None:
        seen.append(workspace_of(running))

    code, urls = _first(monkeypatch, tmp_path, [str(blot), str(marker)], look)
    console = capsys.readouterr()
    assert (code, len(urls)) == (0, 1)
    assert "2 images are waiting there: choose how to import them." in console.out
    assert console.err == ""
    (listed,) = seen.pop()["handoffs"]
    assert [file["name"] for file in listed["files"]] == ["blot µ.tif", "marker α.tif"]

    code, urls = _first(monkeypatch, tmp_path, [str(photo), str(blot), str(photo.parent)], look)
    console = capsys.readouterr()
    assert (code, len(urls)) == (3, 1)
    assert "1 image is waiting there: choose how to import them." in console.out
    assert console.err.splitlines() == [
        f"Not opened: {photo.resolve()} (not an image type Proteia imports"
        " (.tif, .tiff, .png, .jpg, .jpeg))",
        f"Not opened: {photo.parent.resolve()} (a folder; open projects from Proteia's"
        " Projects dialog)",
    ]
    (listed,) = seen.pop()["handoffs"]
    assert [r["code"] for r in listed["refused"]] == ["unsupported_type", "folder"]
    # Nothing was copied, and the originals are there.
    assert blot.read_bytes() == f"pixels of {blot.name}".encode()
    assert not (tmp_path / "state" / "incoming").exists()


def _gone_after_its_checks(monkeypatch, path: Path) -> str:
    """Make ``path`` vanish once the command line has checked it, before it is
    handed over: the path the console shows for it."""
    shown = str(path.resolve())
    parse = cli.parse

    def parse_then_delete(argv: list[str]) -> cli.CommandLine:
        command_line = parse(argv)
        path.unlink()
        return command_line

    monkeypatch.setattr(cli, "parse", parse_then_delete)
    return shown


def test_a_first_launch_names_a_file_gone_before_its_page_took_it(tmp_path, monkeypatch, capsys):
    a, b = scans(tmp_path / "scans µ", "a.tif", "b α.tif")
    shown = _gone_after_its_checks(monkeypatch, b)
    seen: list[dict] = []
    code, urls = _first(
        monkeypatch, tmp_path, [str(a), str(b)], lambda running: seen.append(workspace_of(running))
    )
    console = capsys.readouterr()
    assert (code, len(urls)) == (3, 1)
    assert "1 image is waiting there: choose how to import them." in console.out
    (line,) = console.err.splitlines()  # the path in full, and why
    assert line.startswith(f"Not opened: {shown} (cannot be read: ") and line.endswith(")")
    (listed,) = seen.pop()["handoffs"]
    assert [file["name"] for file in listed["files"]] == ["a.tif"]
    assert [(r["name"], r["code"]) for r in listed["refused"]] == [("b α.tif", "unreadable")]
    log = (tmp_path / "state" / "logs" / "proteia.log").read_text(encoding="utf-8")
    assert "1 paths not taken (unreadable)" in log and "b α.tif" not in log


def test_a_second_launch_names_a_file_gone_before_its_upload(served, tmp_path, monkeypatch, capsys):
    a, c = scans(tmp_path / "scans µ", "a.tif", "c α.tif")
    shown = _gone_after_its_checks(monkeypatch, c)
    opener = Opener()
    _second(monkeypatch, tmp_path, opener)
    assert launch.main([str(a), str(c)]) == 3
    console = capsys.readouterr()
    assert "1 image is waiting there: choose how to import them." in console.out
    (line,) = console.err.splitlines()
    assert line.startswith(f"Not opened: {shown} (cannot be read: ") and line.endswith(")")
    (listed,) = workspace_of(served)["handoffs"]
    assert [file["name"] for file in listed["files"]] == ["a.tif"]
    assert [(r["name"], r["code"]) for r in listed["refused"]] == [("c α.tif", "unreadable")]
    assert len(opener.urls) == 1


def test_a_new_instance_removes_what_a_crashed_one_left_before_it_binds(tmp_path, monkeypatch):
    state = tmp_path / "state"
    incoming = state / "incoming"
    incoming.mkdir(parents=True)
    (incoming / ("0123456789abcdef" * 2)).write_bytes(b"a staged copy")
    (incoming / "notes.txt").write_bytes(b"not one")
    info = {"pid": 1, "port": _closed_port(), "token": "x" * 43}
    (state / INSTANCE_FILE).write_text(json.dumps(info), encoding="utf-8")
    (state / REDIRECT_FILE).write_text("old", encoding="utf-8")
    seen: list[list[str]] = []
    bind = launch.bind_loopback

    def bind_after_looking() -> socket.socket:
        seen.append(names(state))
        return bind()

    monkeypatch.setattr(launch, "bind_loopback", bind_after_looking)
    instance = launch.start(folder=state, opener=Opener())
    try:
        assert seen == [["incoming", LOCK_FILE]]
        assert names(incoming) == ["notes.txt"]
    finally:
        instance.close()


def test_stop_removes_the_instance_files_while_the_server_still_listens(tmp_path):
    instance = launch.start(folder=tmp_path, opener=Opener())
    try:
        instance.stop()
        assert names(tmp_path) == [LOCK_FILE] and instance.server.should_exit
        # A launch now finds no instance file, and waits for the lock.
        with socket.create_connection(("127.0.0.1", instance.port), timeout=5):
            pass
    finally:
        instance.close()


def test_a_stop_signal_removes_the_instance_files_first(tmp_path):
    instance = launch.start(folder=tmp_path, opener=Opener())
    try:
        instance.server.handle_exit(signal.SIGINT, None)  # as uvicorn's handler runs it
        assert names(tmp_path) == [LOCK_FILE] and instance.server.should_exit
    finally:
        instance.close()


def test_the_originals_are_never_deleted_or_changed(tmp_path):
    data = samples.sample_files()[samples.BLOT_FILE]
    original, other = tmp_path / "scans µ" / "β-actin blot.tif", tmp_path / "scans µ" / "b.tif"
    original.parent.mkdir()
    original.write_bytes(data)
    other.write_bytes(data)
    written = original.stat().st_mtime_ns

    def unchanged() -> None:
        assert original.read_bytes() == data and original.stat().st_mtime_ns == written

    state = tmp_path / "state"

    def serve(command_line: cli.CommandLine) -> Running:
        workspace = api.Workspace(tmp_path / "projects", reveal=lambda folder: None)
        instance = START(
            folder=state, opener=Opener(), workspace=workspace, command_line=command_line
        )
        thread = threading.Thread(target=instance.serve, daemon=True)
        thread.start()
        wait_until_started(instance, thread)
        return Running(instance, Opener(), thread)

    def choose(files: list[dict]) -> list[dict]:
        return [
            {
                "file_id": file["file_id"],
                "kind": "chemiluminescence",
                "polarity": "dark_on_light",
                "membrane": "new",
            }
            for file in files
        ]

    # (a) The first launch's own files, imported.
    running = serve(command(original, other))
    (listed,) = workspace_of(running)["handoffs"]
    body = {"name": None, "files": choose(listed["files"])}
    status, answer = call(running, "POST", f"/api/handoffs/{listed['id']}/accept", body)
    assert status == 201, answer
    assert answer["project"]["name"] == "β-actin blot" and len(answer["handoff"]["imported"]) == 2
    unchanged()
    # (b) A second launch's copy, discarded.
    START(folder=state, opener=Opener(), command_line=command(original))
    (listed,) = workspace_of(running)["handoffs"]
    body = {"files": [file["file_id"] for file in listed["files"]], "refused": 0}
    assert call(running, "POST", f"/api/handoffs/{listed['id']}/discard", body)[0] == 204
    assert staged_bytes(running.instance) == []
    unchanged()
    # (c) Stopped with a copy waiting.
    START(folder=state, opener=Opener(), command_line=command(original))
    assert staged_bytes(running.instance) == [data]
    running.instance.stop()
    running.thread.join(10)
    assert not list((state / "incoming").iterdir())
    unchanged()
    # (d) Started again with it named, and stopped before it is imported.
    running = serve(command(original))
    assert waiting(running.instance) == [["β-actin blot.tif"]]
    running.instance.stop()
    running.thread.join(10)
    unchanged()
    assert other.read_bytes() == data


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
        ("GET", "/api/incoming/room?name=a.tif&size=1"),
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
        "/static/calibration.js",
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


# --- Molecular weights: finding, adjusting and marking ladders (#58) ---


def _constants(script: str) -> dict[str, float]:
    """The numbers of a script's top-level ``const NAME = <number>;`` lines."""
    found = re.findall(r"^const (\w+) = (-?[\d.]+);", script, re.MULTILINE)
    return {name: float(value) for name, value in found}


def test_the_fit_line_warns_where_the_core_does():
    # The page words a group's fit with the core's own thresholds: a ladder
    # whose leave-one-out check (D2) is above FIT_WARN, two ladders that
    # disagree beyond LADDERS_WARN, a side with too few marks or shared MWs.
    panel = _code("calibration.js")
    constants = _constants(panel)
    names = ("FIT_WARN", "LADDERS_WARN", "MIN_LADDER_POINTS", "MIN_SHARED_MWS")
    assert {name: constants[name] for name in names} == {
        name: getattr(mwcal, name) for name in names
    }
    lines = _method(panel, "fitLines(")
    assert "take one ladder band away and predict it from the bands above and below it" in lines
    assert "left and right differ by" in lines and '"the ladders agree"' in lines
    assert "2 points · less reliable" in lines
    assert "the bands above and below it put it at" in lines and "check its label" in lines
    assert "not used;" in lines and "mark at least ${MIN_LADDER_POINTS}" in lines


def test_a_ruler_stores_its_solid_labelled_ticks_in_one_step():
    # Apply sends the ticks on a band (found, snapped to, placed by hand or
    # as stored), never one only predicted (hollow) nor a band ▲▼ left with
    # no label, at the ruler's x with the x it was found at, in one PUT: one
    # change, one undo step.
    panel = _code("calibration.js")
    assert re.search(r'function solid\(tick\) \{\s*return tick.state !== "predicted";', panel)
    assert re.search(r"function kept\(tick\) \{\s*return solid\(tick\) && labelled\(tick\);", panel)
    apply = _method(panel, "async apply(")
    assert "draft.ticks.filter(kept)" in apply
    assert "/calibration/${draft.side}/ladder`" in apply
    assert "x: draft.x," in apply and "found_at: draft.foundAt," in apply
    # A tick's snap under way lands first; a tick snapped before the ruler
    # moved sideways is snapped again at the x it is stored at, so the server
    # finds it where a snap there puts it.
    moved = apply.index("tick.snappedX !== draft.x")
    sent = apply.index('this.handlers.edit("PUT", path, body,')
    assert apply.index("await this.snapping") < moved < apply.index("this.snapAgain(draft)") < sent
    # A predicted tick moved with the whole ruler (a shift, a stretch) stays
    # predicted: only a tick dragged, nudged or snapped onto a band is stored.
    assert '(state === "predicted" ? state : "hand")' in _method(panel, "moved(")
    # Esc while it is being stored drops nothing: the answer closes it.
    escape = _method(panel, "escape(")
    assert escape.index("if (this.applying)") < escape.index("this.closeDraft();")


def test_a_proposal_is_worded_as_labels_to_check():
    # A proposal is the best labelling of the peaks down the lane clicked, and
    # a lane of samples has one too: the status line never says a ladder was
    # found, and asks to check that the lane is the ladder. A doubtful one says
    # the labels may be one band off, and how to check and move them. Where
    # no ladder stands out, the page marks by clicks instead.
    panel = _code("calibration.js")
    text = _method(panel, "proposalText(")
    assert "Check that this lane is the ladder" in text
    assert text.index("if (proposal.doubtful)") < text.index("this.doubtText()")
    doubt = _method(panel, "doubtText(")
    assert "The labels may be one band off: check the coloured reference bands" in doubt
    assert doubt.count(" move them with ▲▼.") == 1
    find = _method(panel, "async find(")
    fallback = find.index("if (!proposal)")
    assert fallback < find.index('this.tool = this.newTool({ kind: "mark", side, edges: false });')


def test_the_ruler_moves_as_its_parts_are_dragged():
    # D8: the line moves every tick up or down (or the ruler sideways), a grip
    # stretches it (every tick linear in y between the fixed end and the
    # dragged one), a tick moves alone and snaps unless Alt is held. The view
    # draws predicted ticks hollow and the peaks no label took as grey dots.
    panel = _code("calibration.js")
    moved = _method(panel, "moved(")
    assert 'part.kind === "body" && axis === "x"' in moved
    assert "(dragged - fixed) / (end - fixed)" in moved
    assert "fixed + (tick.y - fixed) * factor" in moved
    ruler = _method(panel, "ruler(")
    assert 'step.part.kind === "tick" && !step.alt' in ruler and "this.snapTick(" in ruler
    view = _code("view.js")
    drawn = _method(view, "drawRuler(")
    assert "if (tick.solid)" in drawn and "strokeRect(" in drawn and "EXTRA_PEAK_COLOR" in drawn
    # ▲▼ move every label one ladder position, and back again.
    shift = _method(panel, "shiftLabels(")
    assert "const index = tick.index + step;" in shift
    # A tick clicked offers its relabel and "Not a ladder band".
    menu = _method(panel, "relabelMenu(")
    assert 'extra: { label: "Not a ladder band", run: () => this.notABand(id) }' in menu


def test_the_ruler_keys():
    # Tab goes tick to tick (a button each, top to bottom); ↑↓ move the
    # focused one 0.5 px, Shift+↑↓ 5 px; Enter applies, on a tick or on the
    # page; Esc drops the ruler, once no popup or tool takes it, and only the
    # drag when it cancels one.
    panel = _code("calibration.js")
    constants = _constants(panel)
    assert (constants["NUDGE"], constants["NUDGE_FAR"]) == (0.5, 5.0)
    keys = _method(panel, "tickKey(")
    for key in ('"ArrowUp"', '"ArrowDown"', '"Enter"', '"Delete"'):
        assert key in keys
    assert "event.shiftKey ? NUDGE_FAR : NUDGE" in keys
    escape = _method(panel, "escape(")
    order = ["if (this.menu)", "if (gestureCancelled)", "if (this.tool)", "if (this.draft)"]
    assert [escape.index(step) for step in order] == sorted(escape.index(step) for step in order)
    app = _code("app.js")
    assert "calibration.escape({ gestureCancelled: event.defaultPrevented })" in app
    assert "calibration.enter()" in app
    # The view takes Esc when it cancels a drag, so the page's Esc drops nothing more.
    view = _code("view.js")
    assert re.search(
        r'event.key === "Escape" && this.gesture\) \{\s*event.preventDefault\(\);', view
    )
    parser = _Tags()
    parser.feed((server.STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    assert parser.by_id["cal-apply"]["aria-keyshortcuts"] == "Enter"
    assert parser.by_id["cal-drop"]["aria-keyshortcuts"] == "Escape"
    assert parser.by_id["cal-ticks"]["aria-label"] == "Ruler ticks, top to bottom"


def test_a_stored_mark_dragged_or_clicked_is_one_edit():
    # After Apply, a mark dragged up or down is moved (snapped unless Alt is
    # held) and a mark clicked relabelled or removed: each its own change and
    # undo step.
    panel = _code("calibration.js")
    point = _method(panel, "ladderPoint(")
    assert "this.movePoint(tick.point, y, !alt)" in point and "this.pointMenu(" in point
    edit = _method(panel, "async editPoint(")
    assert 'this.handlers.edit("PATCH", this.pointPath(point), body, {' in edit
    assert '"edit_calibration_point"' in edit
    remove = _method(panel, "async removePoint(")
    assert 'this.handlers.edit("DELETE", this.pointPath(point))' in remove


def test_click_marking_marks_the_ladder_the_click_is_on():
    # Marking by clicks (where no ladder stands out, or a custom ladder with no
    # MWs listed): a strip edge is the left ladder's; a band nearer the image's
    # right edge than the first ladder's lane is the second ladder's, as Find
    # second ladder would take it, never a far lane of the first.
    side = _method(_code("calibration.js"), "markSide(")
    assert 'point.source !== "strip_edge"' in side
    assert 'x > lane + (this.image.width - lane) / 2 ? "right" : "left"' in side


def test_the_ladder_section_is_labelled():
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    controls = _Controls()
    controls.feed(_markup(html, '<section id="calibration"', "</section>"))
    labelled = [labelled for tag, _, labelled in controls.controls if tag != "button"]
    assert labelled and all(labelled)
    parser = _Tags()
    parser.feed(html)
    for tool in ("cal-find", "cal-find-right", "cal-mark", "cal-mark-edges"):
        assert parser.by_id[tool]["aria-pressed"] == "false"
    marker = _Controls()
    marker.feed(_markup(html, '<label id="marker-field"', "</select>"))
    assert [(tag, labelled) for tag, _, labelled in marker.controls] == [("select", True)]


def test_a_marker_imported_beside_one_unlinked_image_offers_the_link():
    # After a marker is imported into a membrane with exactly one unlinked
    # chemiluminescence image of its size, the status line offers the link.
    app = _code("app.js")
    block = app[app.index('$("import-file").addEventListener("change"') :]
    block = block[: block.index("\n});\n")]
    assert "calibration.linkOffer(answer.project, image.id)" in block
    assert "label: `Link as the marker of ${offer.name}`" in block
    offer = _method(_code("calibration.js"), "linkOffer(")
    assert 'image.kind === "chemiluminescence"' in offer
    assert "image.marker_image_id === null" in offer
    assert "image.width === marker.width" in offer and "unlinked.length === 1" in offer


def test_an_apply_that_changes_nothing_offers_no_undo():
    # Adjust, then Apply with nothing moved, stores the marks as they are: the
    # server logs nothing, so the last undo step is an earlier change (the
    # other ladder's Apply, say), which an Undo offered here would take back.
    # The page compares the answer with the state the edit was sent against:
    # the same revision, and it says there was no change, with no Undo.
    panel = _code("calibration.js")
    apply = _method(panel, "async apply(")
    assert "sent: (project) => {" in apply
    same = apply.index("before.revision === answer.project.revision")
    block = apply.index("if (unchanged) {")
    undo = apply.index('this.undoOf(answer, "set_ladder_points"')
    assert same < block < apply.index("return;", block) < undo
    assert "No change: the ruler holds the ${which} marks as they are stored." in apply
    app = _code("app.js")
    _, edit = _function(app, "async function edit(")
    assert edit.index("sent(state.project);") < edit.index("return send(method, path, json);")


def test_a_ruler_goes_with_the_ladder_and_the_images_it_was_opened_for():
    # A ruler's labels are the membrane's ladder as it was when it opened, and
    # its Apply replaces a ladder of its register group's marks. While it is
    # open (or a ladder is being found) the ladder and the marker link are
    # disabled; changed all the same (Undo, Redo, another tab), the ruler is
    # dropped, never applied with another ladder's labels or over the marks of
    # images it did not show.
    panel = _code("calibration.js")
    opened = _method(panel, "openDraft(")
    assert "ladder: this.ladderScope()," in opened and "group: this.groupScope()," in opened
    stale = _method(panel, "staleRuler(")
    assert "draft.ladder !== this.ladderScope()" in stale
    assert "draft.group !== this.groupScope()" in stale
    assert "const stale = draft ? this.staleRuler(draft) : null;" in _method(panel, "render(")
    locks = _method(panel, "renderLocks(")
    assert "const locked = Boolean(this.draft) || this.finding;" in locks
    for control in ('$("cal-ladder")', '$("marker-image")', '$("cal-link")', 'type="submit"'):
        assert control in locks
    assert "this.renderLocks();" in _method(panel, "refresh(")
    # A proposal answered once the ladder or the link changed is not shown.
    find = _method(panel, "async find(")
    scope = find.index("this.ladderScope() !== scope[0] || this.groupScope() !== scope[1]")
    assert scope < find.index("this.openDraft(")


def test_a_tool_is_put_away_on_another_register_group():
    # Find ladder or a marking tool armed on one image acts on its register
    # group only: shown another membrane's image (or its button disabled since,
    # the ladder taken back), it is put away, never sent there.
    panel = _code("calibration.js")
    fits = _method(panel, "toolFits(")
    assert "tool.group !== this.groupScope()" in fits
    assert "this.findRefusal(tool.side) : this.markRefusal()" in fits
    assert "if (this.tool && !this.toolFits(this.tool))" in _method(panel, "render(")
    assert "this.tool = same ? null : this.newTool(tool);" in _method(panel, "arm(")
    assert "group: this.groupScope()" in _method(panel, "newTool(")


def test_a_ruler_holds_only_the_marks_its_apply_keeps():
    # Apply stores the ruler's ticks as band marks of its image at its x, in
    # place of all the marks of its side. Adjust puts only such marks on the
    # ruler (and takes its x from them): never a strip edge, nor a mark on
    # another image of the group, which Apply would turn into a band of this
    # image. Those Apply removes are named before (the ruler's panel, Apply's
    # description) and after (the status line, with Undo).
    panel = _code("calibration.js")
    adjust = _method(panel, "adjust(")
    assert "const bands = this.bandPoints(side);" in adjust
    marks = adjust.index("const marks = bands.filter((mark) => mark.image_id === imageId);")
    assert marks < adjust.index("const xs = marks.map((mark) => mark.x)")
    assert 'point.source !== "strip_edge"' in _method(panel, "bandPoints(")
    lost = _method(panel, "lostPoints(")
    assert 'point.source === "strip_edge" || point.image_id !== draft.imageId' in lost
    assert "Apply replaces this ladder's marks: it removes" in _method(panel, "renderDraft(")
    assert "which the ruler did not hold." in _method(panel, "appliedText(")
    marks_list = _method(panel, "renderMarks(")
    assert '$("cal-adjust").hidden = !this.bandPoints("left").length;' in marks_list
    parser = _Tags()
    parser.feed((server.STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    assert parser.by_id["cal-apply"]["aria-describedby"] == "cal-draft-lost"
    assert "hidden" in parser.by_id["cal-draft-lost"]


def test_the_keyboard_focus_never_stays_on_a_hidden_control():
    # Enter or Esc on a ruler tick, or Drop, hides the ruler's panel with the
    # focus still on that button: a hidden control has lost the focus too, so
    # it goes on (to Apply's Undo, or to Find ladder).
    panel = _code("calibration.js")
    lost = _method(panel, "focusLost(")
    assert "!active.getClientRects().length" in lost and "Boolean(active.disabled)" in lost
    escape = _method(panel, "escape(")
    assert escape.index("this.closeDraft();") < escape.index("this.keepFocus();")
    keep = _method(panel, "keepFocus(")
    assert '["cal-find", "cal-mark"]' in keep and "button.getClientRects().length" in keep


def test_snap_all_names_the_ticks_it_left_where_they_are():
    # A solid tick no band took stays where it is, and Apply stores it there:
    # the status line names it, as it names the hollow ones not stored.
    snap = _method(_code("calibration.js"), "async snapAll(")
    assert snap.index("if (!result.snapped) {") < snap.index("stayed.push(tick);")
    assert '"stays where it is" : "stay where they are"' in snap
    assert 'and Apply stores ${one ? "it" : "them"} there' in snap


def test_a_read_refused_as_project_changed_says_nothing_was_left_undone():
    # A ladder proposal and a snap are POSTs that change nothing: refused
    # because another project was opened, they are no change "not made".
    app = _code("app.js")
    _, changed = _function(app, "function projectChanged(")
    assert 'answered || read || method === "GET"' in changed
    _, sent = _function(app, "async function request(")
    assert "projectChanged(error, method, path, { read });" in sent
    assert "request(method, path, { json, answer: true, anyProject, read: true })" in app


def test_a_dropped_ticks_snap_lands_by_its_id():
    # The ruler may change while a dropped tick's snap is asked for (another
    # tick dragged, a nudge, ▲▼): the snap lands on that tick by its id while
    # it is still where it was dropped, and a later move of it stands.
    snap = _method(_code("calibration.js"), "snapTick(")
    assert "ruler.ticks.find((tick) => tick.id === id)" in snap
    assert "now.y === dropped.y && now.state === dropped.state" in snap
    assert "this.draft = onBand(this.draft);" in snap and "this.base = onBand(this.base);" in snap
    assert "this.draft !== draft" not in snap


def test_nothing_opens_a_ruler_or_a_popup_while_a_ladder_is_found():
    # The proposal's answer opens a ruler: until it comes, Adjust, the marking
    # tools, Remove and Clear are disabled, as Find is.
    panel = _code("calibration.js")
    busy = "const busy = Boolean(this.draft) || this.applying || this.finding;"
    assert busy in _method(panel, "renderMarks(")
    marking = _method(panel, "renderMarking(")
    assert "Boolean(this.draft) || this.applying || this.finding;" in marking
    assert "this.draft || this.applying || this.finding" in _method(panel, "adjust(")
    assert "if (this.finding || this.applying)" in _method(panel, "arm(")


def test_apply_names_what_its_ruler_was_opened_with():
    # The server refuses a ruler whose ladder or register group changed since
    # it was opened (calibration_changed, another tab's change this page does
    # not know of), but only when the page names them: a page that stopped
    # sending them would lose the check without a sign. Apply sends every
    # field the route reads, as the ruler was opened; such a refusal drops
    # the ruler, says why and reads the project again.
    panel = _code("calibration.js")
    apply = _method(panel, "async apply(")
    start = apply.index("const body = {")
    fields = re.findall(r"^\s*(\w+):", apply[start : apply.index("};", start)], re.MULTILINE)
    assert sorted(fields) == sorted(api.LadderPointsBody.model_fields)
    assert "ladder_kda: draft.ladderKda," in apply and "group: draft.groupIds," in apply
    opened = _method(panel, "openDraft(")
    assert "ladderKda: [...this.ladderKda()]," in opened
    assert "groupIds: [...this.group.image_ids]," in opened
    # A found ruler's labels are the ladder the proposal names (the server's
    # when it labelled them), opened only while it is the ladder shown.
    find = _method(panel, "async find(")
    same = find.index("if (!sameLadder(labels, this.ladderKda())) {")
    assert find.index("const labels = answer.ladder_kda;") < same < find.index("this.openDraft(")
    assert find.index("this.handlers.reread();", same) < find.index("this.openDraft(")
    assert "ladderKda: [...labels]," in find
    refused = _method(panel, "applyRefused(")
    changed = refused.index('error.code === "calibration_changed"')
    assert changed < refused.index("this.closeDraft();") < refused.index("this.handlers.reread();")
    # Its words, by the field the refusal names.
    start = panel.index("const CHANGED_WORDS = {")
    words = set(re.findall(r"^  (\w+):", panel[start : panel.index("};", start)], re.MULTILINE))
    assert words == {"ladder_kda", "group"} <= set(api.LadderPointsBody.model_fields)
    assert "reread: () => reread().catch(report)," in _code("app.js")


def test_a_ladder_refusal_is_worded_from_its_code_and_detail():
    # A refusal of the ladder's order, of an MW held twice or of the ladders'
    # sides is worded from its code and detail (the ladder side and the MWs
    # the server gives), never shown as the server's message, which names ids
    # and positions; any other keeps that message. Apply, a mark, a mark moved
    # or relabelled and a link say it so.
    panel = _code("calibration.js")
    _, words = _function(panel, "function ladderRefusalWords(")
    for code in ("calibration_order", "duplicate_mw", "ladder_sides"):
        assert f'error.code === "{code}"' in words
    for field in ("detail.upper", "detail.lower", "detail.mw", "detail.side"):
        assert field in words
    for reason in ("same_height", "strip_edge", "no_x", "not_right"):
        assert f'detail.reason === "{reason}"' in words
    assert "return null;" in words
    _, text = _function(panel, "function refusalText(")
    assert "sentence(error.message)" in text
    for method, verb in (
        ("applyRefused(", '"Not applied"'),
        ("async mark(", '"Not marked"'),
        ("async editPoint(", '"Not moved"'),
        ("async link(", '"Not linked"'),
    ):
        assert verb in _method(panel, method), method
    _, report = _function(_code("app.js"), "function report(")
    assert "showStatus(text || error.message);" in report


def test_space_presses_a_focused_control_over_the_image():
    # Space held pans the image, with the pointer over it; but Space on a
    # control with the focus is the control's (a ruler tick's button opens its
    # relabel popup), and only a drag it panned makes its release press
    # nothing.
    view = _code("view.js")
    _, pressed = _function(view, "function pressedBySpace(")
    assert "HTMLButtonElement" in pressed and "PRESSED.has(element.type)" in pressed
    keydown = view[view.index('document.addEventListener("keydown"') :]
    keydown = keydown[: keydown.index('document.addEventListener("keyup"')]
    assert "!typesSpace(target) &&\n        !pressedBySpace(target) &&" in keydown
    assert "if (this.spaceTaken || this.spacePanned) {" in view
    assert "this.untype();\n        this.spacePanned = true;" in view
