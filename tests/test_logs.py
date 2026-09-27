# SPDX-License-Identifier: Apache-2.0
"""The session log (#137): every launch writes what it did to a log file in the
per-user state folder. Operations, refusals, fallbacks and errors reach it; no
full project path, access token or pixel data does; it rotates and keeps a fixed
number of files; and a log that cannot be written never stops the app. Requests
go to a real server on a loopback socket."""

from __future__ import annotations

import asyncio
import http.client
import io
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pytest
from PIL import Image

from conftest import FakeClock, synthetic_blot, write_tiff
from proteia.core import operations as ops
from proteia.core import storage
from proteia.web import api, launch, logs, projects, server
from test_operations import FOLDER, ONE_BAND, row_session, v1_folder

ROOT = "projects"  # the projects root in tmp_path
LINE = re.compile(
    r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d \[(\d+)\] ([A-Z]+) ([\w.]+): (.*)$"
)


@dataclass
class Log:
    """The session log set up for a test: its file and the projects root it hides."""

    path: Path
    root: Path

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def records(self) -> list[str]:
        """Each record's first line (continuation lines left out)."""
        return [line for line in self.text().splitlines() if not line.startswith("  | ")]

    def messages(self, level: str | None = None) -> list[str]:
        found = []
        for line in self.records():
            match = LINE.match(line)
            assert match, line
            if level is None or match[2] == level:
                found.append(match[4])
        return found


@pytest.fixture
def log(tmp_path):
    root = tmp_path / ROOT
    path = logs.setup(tmp_path / "state" / logs.LOG_DIR, hidden={root: "<projects>"})
    assert path == tmp_path / "state" / logs.LOG_DIR / logs.LOG_FILE
    yield Log(path, root)
    logs.shutdown()


class Client:
    """JSON requests to a running instance (served by ``thread``), with its token
    unless told otherwise."""

    def __init__(self, instance: launch.Instance, thread: threading.Thread) -> None:
        self.instance, self.thread = instance, thread
        self.port, self.token = instance.port, instance.token

    def call(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        raw: bytes | None = None,
        token=None,
        host: str | None = None,
    ) -> tuple[int, Any]:
        headers = {"Authorization": f"Bearer {token or self.token}"}
        if host is not None:
            headers["Host"] = host  # instead of the one http.client sends
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request(method, path, body=data, headers=headers)
            response = conn.getresponse()
            payload = response.read()
            kind = response.getheader("content-type", "")
        finally:
            conn.close()
        return response.status, json.loads(payload) if kind.startswith(
            "application/json"
        ) else payload

    def ok(self, method: str, path: str, body: Any = None, **kw: Any) -> Any:
        status, answer = self.call(method, path, body, **kw)
        assert status in (200, 201, 202), (status, answer)
        return answer

    def upload(self, data: bytes, name: str) -> Any:
        query = f"name={quote(name, safe='')}&kind=chemiluminescence&polarity=dark_on_light"
        return self.ok("POST", f"/api/images?{query}", raw=data)


def serve(instance: launch.Instance) -> threading.Thread:
    thread = threading.Thread(target=instance.serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "the server did not start"
        time.sleep(0.01)
    return thread


@pytest.fixture
def client(tmp_path):
    workspace = api.Workspace(tmp_path / ROOT, reveal=lambda folder: None, clock=FakeClock())
    instance = launch.start(folder=tmp_path / "state", opener=lambda url: True, workspace=workspace)
    thread = serve(instance)
    yield Client(instance, thread)
    instance.stop()
    thread.join(10)


def tiff_bytes(tmp_path: Path, shape=(60, 400), xs=(50, 120, 190)) -> bytes:
    blot = synthetic_blot(shape, [(x, 30, 5.0, 3.0, 30000.0) for x in xs])
    return write_tiff(tmp_path / "source.tif", blot).read_bytes()


def jpeg_bytes(shape=(60, 60)) -> bytes:
    blot = (synthetic_blot(shape, [(30, 30, 5.0, 3.0, 30000.0)]) // 257).astype(np.uint8)
    data = io.BytesIO()
    Image.fromarray(blot).save(data, format="JPEG", quality=95)
    return data.getvalue()


def spelled(folder: Path) -> re.Pattern[str]:
    """``folder`` however a message may spell it: either slash, doubled
    backslashes, any case on Windows."""
    parts = [re.escape(part) for part in re.split(r"[\\/]+", str(folder)) if part]
    return re.compile(r"[\\/]+".join(parts), re.IGNORECASE if os.name == "nt" else 0)


# --- The file ---


def test_a_record_is_one_line_with_time_process_level_logger_and_message(log):
    own, other = logging.getLogger("proteia.test"), logging.getLogger("elsewhere")
    own.info("β-actin 10 µM\nsecond line")
    own.debug("not written: below INFO")
    other.info("not written: another library's INFO")
    other.warning("another library's warning")
    lines = log.text().splitlines()
    assert len(lines) == 2
    first = LINE.match(lines[0])
    assert first is not None
    assert (first[1], first[2], first[3]) == (str(os.getpid()), "INFO", "proteia.test")
    assert first[4] == "β-actin 10 µM\\nsecond line"  # one line, in UTF-8
    assert lines[1].endswith("WARNING elsewhere: another library's warning")


def test_python_warnings_and_thread_errors_reach_the_log_and_the_console_as_before(
    tmp_path, monkeypatch, capsys
):
    # Python's own report of an error in a thread (pytest's would warn of it).
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    path = logs.setup(tmp_path / logs.LOG_DIR)
    try:
        warnings.warn("a warning from a library", RuntimeWarning, stacklevel=1)

        def fail() -> None:
            raise ValueError("a bug in a thread")

        thread = threading.Thread(target=fail, name="worker")
        thread.start()
        thread.join()
    finally:
        logs.shutdown()
    assert threading.excepthook is threading.__excepthook__  # put back
    text = path.read_text(encoding="utf-8")
    assert "WARNING py.warnings:" in text and "RuntimeWarning: a warning from a library" in text
    assert "ERROR proteia.web.logs: unexpected error in worker" in text
    assert "  | ValueError: a bug in a thread" in text
    # The console shows both as Python shows them without a session log.
    err = capsys.readouterr().err
    warned, _, reported = err.partition("Exception in thread worker:\n")
    assert warned.startswith(f"{__file__}:")
    assert ": RuntimeWarning: a warning from a library\n  warnings.warn(" in warned
    assert warned.endswith(")\n") and "\n\n" not in warned  # no blank line after it
    assert reported.startswith("Traceback (most recent call last):\n")
    assert reported.endswith("ValueError: a bug in a thread\n")
    assert "unexpected error" not in err and "WARNING" not in err


# --- What reaches it ---


def test_operations_reach_the_log_with_their_params(log, client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot µ"})
    image_id = client.upload(tiff_bytes(tmp_path), "β-actin.tif")["image_id"]
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": c} for c in ("a", "b", "c")]})
    body = {"name": "β-actin", "role": "target", "image_id": image_id, "box_size": [12, 10]}
    protein_id = client.ok("POST", "/api/proteins", body)["protein_id"]
    for lane, x in enumerate((50, 120, 190)):
        body = {"protein_id": protein_id, "x": x, "y": 30, "lane_index": lane}
        client.ok("POST", "/api/boxes", body)
    client.ok("POST", "/api/undo")
    folder = client.ok("POST", "/api/export")["folder"]
    assert client.call("POST", "/api/project/reveal", {"folder": folder})[0] == 204
    assert client.call("POST", "/api/project/reveal")[0] == 204

    messages = log.messages("INFO")
    committed = [m for m in messages if m.startswith("committed #")]
    assert [m.split(" in ")[0] for m in committed] == [
        "committed #1 new_project",
        "committed #2 import_image",
        "committed #3 set_lanes",
        "committed #4 add_protein",
        "committed #5 place_box",
        "committed #6 place_box",
        "committed #7 place_box",
        "committed #8 undo",
    ]
    assert committed[0] == "committed #1 new_project in 'Blot µ': {}"
    # The params as project.json holds them.
    saved = json.loads((log.root / "Blot µ" / storage.PROJECT_FILE).read_text(encoding="utf-8"))
    for message, entry in zip(committed, saved["log"], strict=True):
        assert json.loads(message.split(": ", 1)[1]) == entry["params"]
    exported = rf"in 'Blot µ': exported \d+ files to {re.escape(folder)}"
    assert any(re.fullmatch(exported, m) for m in messages)
    assert messages[-2:] == [
        f"in 'Blot µ': showed {folder} in the file manager",
        "in 'Blot µ': showed its folder in the file manager",
    ]
    # No pixel data: no run of numbers longer than a box's rect.
    assert not re.search(r"(?:\d+(?:\.\d+)?[,\s]+){12}\d", log.text())


def test_refusals_reach_the_log_with_their_code_and_message(log, client):
    assert client.call("POST", "/api/boxes", {"protein_id": "prot-1", "x": 1, "y": 1})[0] == 409
    client.ok("POST", "/api/projects", {"name": "Blot"})
    body = {"name": "β-actin", "role": "target", "image_id": "img-9"}
    assert client.call("POST", "/api/proteins", body)[0] == 404
    assert client.call("PUT", "/api/lanes", {"lanes": 3})[0] == 422
    assert client.call("PUT", "/api/lanes", {"lanes": [{"condition": "  "}]})[0] == 422
    assert client.call("GET", "/api/nothing")[0] == 404
    assert client.call("DELETE", "/api/status")[0] == 405
    wrong = "w" * 43
    assert client.call("GET", "/api/project", token=wrong)[0] == 401
    assert client.call("GET", "/api/status", host=f"localhost:{client.port}")[0] == 400
    # A path as long as the server takes: its line gives its start only.
    assert client.call("GET", "/api/" + "A" * 15_000, token=wrong)[0] == 401

    refused = [m for m in log.messages("INFO") if m.startswith("refused ")]
    assert refused[0] == "refused POST /api/boxes: 409 no_project: create or open a project first"
    assert refused[1].startswith("refused POST /api/proteins: 404 unknown_id: ")
    assert refused[2].startswith("refused PUT /api/lanes: 422 invalid_input: body.lanes: ")
    assert refused[3].startswith("refused PUT /api/lanes: 422 blank_text: ")
    assert refused[4] == "refused GET /api/nothing: 404 Not Found"
    assert refused[5] == "refused DELETE /api/status: 405 Method Not Allowed"
    assert refused[6] == "refused GET /api/project: 401 missing or wrong access token"
    assert refused[7] == "refused GET /api/status: 400 unexpected Host header"
    assert refused[8] == (
        f"refused GET /api/{'A' * 195}... (14805 more characters):"
        " 401 missing or wrong access token"
    )
    assert len(refused) == 9
    assert wrong not in log.text()


def test_a_refusal_logs_its_ids(log, client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    image_id = client.upload(tiff_bytes(tmp_path), "blot.tif")["image_id"]
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": c} for c in ("a", "b", "c")]})
    body = {"name": "β-actin", "role": "target", "image_id": image_id, "box_size": [12, 10]}
    protein_id = client.ok("POST", "/api/proteins", body)["protein_id"]
    body = {"protein_id": protein_id, "x": 50, "y": 30, "lane_index": 0}
    band_id = client.ok("POST", "/api/boxes", body)["band_id"]
    body = {"protein_id": protein_id, "x": 52, "y": 30, "lane_index": 1}
    status, answer = client.call("POST", "/api/boxes", body)
    assert (status, answer["code"], answer["ids"]) == (422, "overlap", [band_id])
    (refusal,) = [m for m in log.messages() if m.startswith("refused ")]
    assert refusal.startswith("refused POST /api/boxes: 422 overlap: ")
    assert refusal.endswith(f"; ids {band_id}")


def test_fallbacks_reach_the_log(log, client):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    image_id = client.upload(jpeg_bytes(), "blot.jpg")["image_id"]
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "a"}]})
    body = {"name": "β-actin", "role": "target", "image_id": image_id, "box_size": [50, 50]}
    protein_id = client.ok("POST", "/api/proteins", body)["protein_id"]
    body = {"protein_id": protein_id, "x": 30, "y": 30, "lane_index": 0}
    answer = client.ok("POST", "/api/boxes", body)
    (band,) = answer["project"]["proteins"][0]["bands"]
    # The box leaves no membrane around it: its background is the image's median.
    assert (band["background_mode"], band["clipped"]) == ("image", None)
    band_id = band["id"]

    messages = log.messages("INFO")
    assert any(
        m.startswith(f"in 'Blot': {image_id} was imported with a warning, lossy_format: JPEG")
        for m in messages
    )
    assert (
        f"in 'Blot': over-exposure cannot be checked on {image_id}"
        " (lossy_format; assessed near the detector limit instead)"
    ) in messages
    assert (
        f"in 'Blot': the background of {band_id} on {image_id} was measured another way,"
        " image (too little membrane around its box: the image's median)"
    ) in messages
    assert (
        f"in 'Blot': over-exposure could not be checked on {band_id} on {image_id}"
        " (lossy_format; assessed near the detector limit instead)"
    ) in messages

    # Taken again, a fallback is not logged again.
    before = len(log.messages())
    client.ok("PUT", f"/api/proteins/{protein_id}/box-size", {"width": 48, "height": 50})
    later = log.messages()[before:]
    assert [m.split(" in ")[0] for m in later] == ["committed #6 set_box_size"]


def test_a_project_read_again_logs_it(log, client):
    client.ok("POST", "/api/projects", {"name": "Blot"})
    project_file = log.root / "Blot" / storage.PROJECT_FILE
    doc = json.loads(project_file.read_text(encoding="utf-8"))
    project_file.write_text(json.dumps(doc, indent=1), encoding="utf-8")  # rewritten outside
    client.ok("POST", "/api/projects/open", {"name": "Blot"})
    assert (
        "read 'Blot' again, as its project.json was changed outside Proteia: 0 images,"
        " 0 proteins, 0 boxes, 1 log entries"
    ) in log.messages("INFO")


def test_opening_migrating_reading_again_exporting_and_closing_reach_the_log(log, tmp_path):
    folder = v1_folder(tmp_path)  # a schema-1 project, its nets on the legacy background
    v1 = (folder / storage.PROJECT_FILE).read_bytes()
    s = ops.open_project(folder, clock=FakeClock())  # migrated and saved
    ops.set_reference_condition(s, None)
    (folder / storage.PROJECT_FILE).write_bytes(v1)  # an old copy restored
    assert s.reload()  # migrated again
    ops.export_lane_table(s)
    s.close()

    name = repr(FOLDER)
    legacy = (
        f"in {name}: the nets are measured with the legacy background (global_median),"
        " each image's median, until the project is requantified"
    )
    messages = log.messages("INFO")
    assert [m.split(":")[0] for m in messages] == [
        f"opened {name}",
        f"in {name}",
        f"committed #2 migrate in {name}",
        f"committed #3 set_reference_condition in {name}",
        f"read {name} again, as its project.json was changed outside Proteia",
        f"in {name}",
        f"committed #2 migrate in {name}",
        f"in {name}",
        f"closed {name}",
    ]
    assert messages[1] == messages[5] == legacy
    migrated = json.loads(messages[6].split(": ", 1)[1])
    assert migrated == s.project.log[1].params  # as project.json holds them
    assert messages[7] == f"in {name}: exported the lane table to exports/{ops.LANE_TABLE_FILE}"


def test_a_row_box_that_left_lanes_unlocated_logs_them(log, tmp_path):
    s, image, protein = row_session(tmp_path, ONE_BAND)
    placement = ops.detect_row_boxes(s, protein, ONE_BAND.row)
    assert placement.unlocated_lanes == (0, 1, 2, 4, 5)
    fallback = (
        f"in {FOLDER!r}: the row box of {protein} on {image}: the detector warns of nothing;"
        " lanes not located (one band found): 1, 2, 3, 5, 6"
    )
    assert log.messages("INFO").count(fallback) == 1
    ops.detect_row_boxes(s, protein, ONE_BAND.row)  # the same drag again: a no-op
    assert log.messages("INFO").count(fallback) == 1


def test_a_reopen_that_could_not_check_the_file_logs_it(log, monkeypatch):
    workspace = api.Workspace(log.root, reveal=lambda folder: None, clock=FakeClock())
    session = workspace.create("A µ")
    monkeypatch.setattr(api, "REOPEN_WAIT_S", 0.2)
    held, release = threading.Event(), threading.Event()

    def running() -> None:  # an operation that holds the session past the wait
        with session.lock:
            held.set()
            release.wait(10)

    operation = threading.Thread(target=running)
    operation.start()
    assert held.wait(10)
    try:
        assert workspace.open("A µ") is session
    finally:
        release.set()
        operation.join(10)
    workspace.close()
    assert (
        "'A µ' was opened again while a request still used it after 0.2 s: its project.json"
        " was not checked for changes made outside Proteia"
    ) in log.messages("INFO")


# --- Errors ---


def _failing_route(guard: server.Guard) -> None:
    def fail() -> None:
        raise RuntimeError("a bug in a route")

    guard.app.add_api_route("/api/fail", fail)


TOKEN, PORT = "t" * 43, 8123  # of an app called without a server (asgi_call)
HOST = f"127.0.0.1:{PORT}".encode()


def asgi_call(guard: server.Guard, method: str, path: str, *, host=HOST, token=TOKEN) -> int:
    """The status ``guard`` (an app for :data:`TOKEN` and :data:`PORT`) answers
    ``method path`` with, called directly."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", host), (b"authorization", f"Bearer {token}".encode())],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", PORT),
    }
    statuses: list[int] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def record(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    asyncio.run(guard(scope, receive, record))
    return statuses[0]


def test_an_unexpected_error_reaches_the_log_with_its_stack_trace(log, capsys):
    guard = server.create_app(token=TOKEN, port=PORT, on_quit=lambda: None)
    _failing_route(guard)
    with pytest.raises(RuntimeError):
        asgi_call(guard, "GET", "/api/fail")
    lines = log.text().splitlines()
    assert lines[0].endswith("ERROR proteia.web.server: unexpected error answering GET /api/fail")
    assert lines[1] == "  | Traceback (most recent call last):"
    assert lines[-1] == "  | RuntimeError: a bug in a route"
    assert all(line.startswith("  | ") for line in lines[1:])
    assert TOKEN not in log.text()
    assert capsys.readouterr().err == ""  # the server reports it there (as before)


def test_an_error_on_a_running_server_is_logged_once(log, client, capsys):
    _failing_route(client.instance.server.config.app)
    assert client.call("GET", "/api/fail")[0] == 500
    assert client.ok("GET", "/api/status")["app"] == "proteia"  # still serving
    deadline = time.monotonic() + 5
    while "RuntimeError: a bug in a route" not in log.text():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    time.sleep(0.2)  # the server's own record of it would come now
    text = log.text()
    assert text.count("Traceback (most recent call last)") == 1
    assert "unexpected error answering GET /api/fail" in text
    assert "Exception in ASGI application" not in text
    # The console shows the server's report of it, as uvicorn writes it.
    err = capsys.readouterr().err
    assert err.startswith("ERROR:    Exception in ASGI application\nTraceback (most recent")
    assert err.count("Traceback (most recent call last)") == 1
    assert err.endswith("RuntimeError: a bug in a route\n")
    assert "unexpected error answering" not in err


def test_a_file_error_is_logged_with_its_stack_trace_and_no_full_path(log, client, capsys):
    log.root.parent.mkdir(parents=True, exist_ok=True)
    log.root.write_text("a file where the projects folder should be", encoding="utf-8")
    assert client.call("POST", "/api/projects", {"name": "Blot"})[0] == 500
    (error,) = log.messages("ERROR")
    assert error.startswith("refused POST /api/projects: 500 file_error: ")
    assert "<projects>" in error
    assert "  | Traceback (most recent call last):" in log.text()
    assert not spelled(log.root).search(log.text())
    assert capsys.readouterr().err == ""  # an answered error, as before


def test_an_error_that_stops_the_app_is_logged_in_the_file_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "state_dir", lambda: tmp_path / "state")

    def fail() -> None:
        raise RuntimeError("a bug at start")

    monkeypatch.setattr(launch, "start", fail)
    with pytest.raises(RuntimeError, match="a bug at start"):
        launch.main()
    lines = (tmp_path / "state" / logs.LOG_DIR / logs.LOG_FILE).read_text("utf-8").splitlines()
    (at,) = [
        i
        for i, line in enumerate(lines)
        if line.endswith(" ERROR proteia.web.launch: Proteia stopped on an unexpected error")
    ]
    assert lines[at + 1] == "  | Traceback (most recent call last):"
    assert "  | RuntimeError: a bug at start" in lines
    assert lines[-1].endswith(" INFO proteia.web.launch: session ended")
    assert capsys.readouterr().err == ""  # Python prints it as the app exits


def test_unsaved_changes_at_the_stop_reach_the_log_and_the_console_once(
    log, client, monkeypatch, capsys
):
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    client.ok("POST", "/api/projects", {"name": "Blot"})
    project_file = log.root / "Blot" / storage.PROJECT_FILE
    project_file.unlink()
    project_file.mkdir()  # no save can replace it
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    capsys.readouterr()
    client.instance.stop()
    client.thread.join(10)
    assert not client.thread.is_alive()
    project_file.rmdir()

    (stopped,) = log.messages("WARNING")[1:]  # after the autosave's
    assert stopped.startswith("Proteia stopped, but the open project could not be saved: ")
    assert "<projects>" in stopped and not spelled(log.root).search(log.text())
    err = capsys.readouterr().err
    assert err.count("Proteia stopped, but ") == 1
    assert err.startswith("Proteia stopped, but the open project could not be saved: ")


def test_refusals_anyone_can_cause_cannot_push_the_history_out(log):
    guard = server.create_app(token=TOKEN, port=PORT, on_quit=lambda: None)
    now = [0.0]
    guard.refusals.clock = lambda: now[0]
    long = "/api/" + "A" * 15_000
    for _ in range(100):
        assert asgi_call(guard, "GET", long, token="w" * 43) == 401
    for _ in range(100):  # the page shell needs no token, a file not found neither
        assert asgi_call(guard, "GET", "/static/" + "B" * 240 + ".js", token="") == 404
    refused = log.messages("INFO")
    assert len(refused) == 2 * logs.REFUSALS_LOGGED
    assert refused[0] == (
        f"refused GET /api/{'A' * 195}... (14805 more characters):"
        " 401 missing or wrong access token"
    )
    assert refused[-1] == (
        f"refused GET /static/{'B' * 192}... (51 more characters): 404 Not Found"
    )
    assert max(len(line) for line in log.records()) < 400
    assert log.path.stat().st_size < 20_000  # of the 3 MB the paths would take

    now[0] = 60.0  # a minute on: one by one again, after the count
    assert asgi_call(guard, "GET", "/api/status", host=b"evil.example") == 400
    assert log.messages("INFO")[2 * logs.REFUSALS_LOGGED :] == [
        "refused 80 more requests on their Host header or token, not logged one by one"
        " (more than 20 a minute)",
        "refused GET /api/status: 400 unexpected Host header",
    ]


def test_a_throttle_admits_its_limit_each_period_and_counts_the_rest():
    now = [0.0]
    throttle = logs.Throttle(limit=2, period=10.0, clock=lambda: now[0])
    assert [throttle.admit() for _ in range(5)] == [(True, 0), (True, 0)] + [(False, 0)] * 3
    now[0] = 9.9
    assert throttle.admit() == (False, 0)
    now[0] = 10.0
    assert [throttle.admit() for _ in range(3)] == [(True, 4), (True, 0), (False, 0)]
    now[0] = 25.0
    assert throttle.admit() == (True, 1)
    assert logs.shorten("abc", 3) == "abc"
    assert logs.shorten("abcd", 3) == "abc... (1 more characters)"


# --- What never reaches it ---


def test_no_full_project_path_reaches_the_log(log, client, monkeypatch, capsys):
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    client.ok("POST", "/api/projects", {"name": "Blot µ"})
    folder = log.root / "Blot µ"
    project_file = folder / storage.PROJECT_FILE
    project_file.unlink()
    project_file.mkdir()  # the autosave cannot replace it
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    project_file.rmdir()
    # The console shows the warning as before: its message alone, the path whole.
    (shown,) = capsys.readouterr().err.splitlines()
    assert shown.startswith("autosave after set_lanes failed; the change is kept in memory:")
    assert spelled(project_file).search(shown)
    say = logging.getLogger("proteia.test").info
    say("as a path: %s", project_file)
    say("as Python writes it: %r", str(project_file))
    say("as JSON writes it: %s", json.dumps({"path": str(project_file)}))
    say("with forward slashes: %s", project_file.as_posix())
    say("as a URI: %s", project_file.as_uri())
    if os.name == "nt":
        say("in capitals: %s", str(project_file).upper())

    text = log.text()
    (autosave,) = log.messages("WARNING")
    assert autosave.startswith("autosave after set_lanes failed; the change is kept in memory:")
    assert "<projects>" in autosave and "Blot µ" in autosave
    assert not spelled(log.root).search(text)
    assert not spelled(Path.home()).search(text)
    assert "<projects>\\Blot µ\\project.json" in text or "<projects>/Blot µ/project.json" in text


def test_folders_are_hidden_only_where_their_name_ends(tmp_path):
    redact = logs.Redactor()
    redact.hide_folder(tmp_path / "ann", "<ann>")
    redact.hide_folder(tmp_path / "ann" / "projects", "<projects>")
    redact.hide_folder(Path(tmp_path.anchor), "<drive>")  # a root alone: ignored
    assert redact(str(tmp_path / "ann" / "x")) == str(Path("<ann>") / "x")
    assert redact(str(tmp_path / "ann" / "projects" / "B")) == str(Path("<projects>") / "B")
    assert redact(str(tmp_path / "anna")) == str(tmp_path / "anna")
    assert redact(f"'{tmp_path / 'ann'}'") == "'<ann>'"


def test_no_token_reaches_the_log_or_the_console(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(projects, "projects_root", lambda: tmp_path / ROOT)
    started: list[launch.Instance] = []
    failed: list[BaseException] = []
    start = launch.start

    def start_quietly() -> launch.Instance | None:
        instance = start(opener=lambda url: True)  # never a real browser
        started.append(instance)
        return instance

    def use_then_quit() -> None:
        deadline = time.monotonic() + 10
        while not started or not started[0].server.started:
            if time.monotonic() > deadline:
                return
            time.sleep(0.01)
        try:
            client = Client(started[0], threading.main_thread())  # main() serves it
            client.ok("GET", "/api/workspace")
            assert client.call("GET", "/api/project", token="w" * 43)[0] == 401
            logging.getLogger("proteia.test").warning(
                "a message that holds the token: %s, Bearer %s, %s",
                launch.app_url(client.port, client.token),
                client.token,
                json.dumps({"token": client.token}),
            )
            client.ok("POST", "/api/quit")
        except BaseException as exc:  # stop the server all the same: main() returns
            failed.append(exc)
            started[0].stop()

    monkeypatch.setattr(launch, "start", start_quietly)
    helper = threading.Thread(target=use_then_quit, daemon=True)
    helper.start()
    assert launch.main() == 0
    helper.join(10)
    assert not failed, failed

    (instance,) = started
    text = (tmp_path / "state" / logs.LOG_DIR / logs.LOG_FILE).read_text(encoding="utf-8")
    messages = [LINE.match(line)[4] for line in text.splitlines()]
    assert messages[0].startswith("session started: Proteia ")
    assert f"serving at http://127.0.0.1:{instance.port}/" in messages
    assert "quit from the page" in messages
    assert messages[-1] == "session ended"
    assert (
        f"a message that holds the token: http://127.0.0.1:{instance.port}/#token=<hidden>,"
        ' Bearer <hidden>, {"token": "<hidden>"}'
    ) in messages
    console = capsys.readouterr()
    for shown in (text, console.out, console.err):
        assert instance.token not in shown
    assert "#token=<hidden>" in console.err  # the console hides it too


def test_a_token_is_hidden_wherever_it_is_written():
    redact = logs.Redactor()
    token = "k" * 43
    redact.conceal(token)
    redact.conceal("short")  # not a token
    assert redact(f"x{token}y") == "x<hidden>y"
    assert redact("short") == "short"
    other = "o" * 43  # never concealed, but written as a token
    assert redact(f"/#token={other}") == "/#token=<hidden>"
    assert redact(f"bearer {other}") == "bearer <hidden>"
    assert redact(f'{{"pid": 1, "token": "{other}"}}') == '{"pid": 1, "token": "<hidden>"}'


# --- Rotation and retention ---


def test_rotation_and_retention_remove_old_files(tmp_path):
    folder = tmp_path / logs.LOG_DIR
    folder.mkdir()
    # Kept by an older version, or by a rotation that did not finish.
    an_hour_ago = time.time() - 3600
    for stale in ("proteia.log.7", "proteia.log.rotating", "proteia.log.deleting"):
        (folder / stale).write_text("old", encoding="utf-8")
        os.utime(folder / stale, (an_hour_ago, an_hour_ago))
    (folder / "notes.txt").write_text("the user's", encoding="utf-8")
    logs.setup(folder, max_bytes=1000, backups=2)
    try:
        say = logging.getLogger("proteia.test").info
        for number in range(60):
            say("record %03d %s", number, "x" * 60)
    finally:
        logs.shutdown()
    kept = sorted(path.name for path in folder.iterdir())
    assert kept == ["notes.txt", "proteia.log", "proteia.log.1", "proteia.log.2"]
    numbers = []
    for name in ("proteia.log.2", "proteia.log.1", "proteia.log"):  # oldest first
        data = (folder / name).read_bytes()
        assert len(data) <= 1000
        numbers.append([int(n) for n in re.findall(rb"record (\d{3})", data)])
    flat = [n for part in numbers for n in part]
    assert flat == list(range(flat[0], 60))  # in order, the newest kept, none twice
    assert flat[0] > 0  # the oldest records are gone


def test_a_file_another_process_is_rotating_now_is_left_to_it(tmp_path):
    # A second launch sets up its log while the running instance rotates: the
    # file it set aside a moment ago is its own, not one a rotation left.
    folder = tmp_path / logs.LOG_DIR
    folder.mkdir()
    aside = folder / "proteia.log.rotating"
    aside.write_text("the running instance's newest records", encoding="utf-8")
    logs._prune(folder, logs.BACKUPS)
    assert aside.exists()
    logs._prune(folder, logs.BACKUPS, now=time.time() + logs.ASIDE_GRACE_S + 1)
    assert not aside.exists()


def test_a_build_without_a_console_sets_up_its_log(tmp_path, monkeypatch):
    # A windowed build has no standard streams: the console formatter must not
    # ask them whether they are a terminal.
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    folder = tmp_path / logs.LOG_DIR
    assert logs.setup(folder) is not None
    try:
        logging.getLogger("proteia.test").warning("started")
    finally:
        logs.shutdown()
    assert "started" in (folder / logs.LOG_FILE).read_text(encoding="utf-8")


# The log files, oldest first, when every one is kept.
CHAIN = [f"{logs.LOG_FILE}.{number}" for number in range(logs.BACKUPS, 0, -1)] + [logs.LOG_FILE]


def hold_open(monkeypatch, name: str) -> Callable[[], None]:
    """Make the log file ``name`` act as one another program holds open on
    Windows: it can be neither renamed nor deleted, and no file can be moved
    over it. Returns what lets it go."""
    held = [name]
    rename, replace, remove, unlink = os.rename, os.replace, os.remove, os.unlink

    def check(*paths) -> None:
        if held and any(Path(path).name == held[0] for path in paths):
            raise PermissionError(13, "held open by another program", str(paths[0]))

    def moving(move):
        def moved(source, target, *args, **kw):
            check(source, target)
            return move(source, target, *args, **kw)

        return moved

    def deleting(delete):
        def deleted(path, *args, **kw):
            check(path)
            return delete(path, *args, **kw)

        return deleted

    monkeypatch.setattr(os, "rename", moving(rename))
    monkeypatch.setattr(os, "replace", moving(replace))
    monkeypatch.setattr(os, "remove", deleting(remove))
    monkeypatch.setattr(os, "unlink", deleting(unlink))
    return held.clear


def write_records(start: int, count: int) -> int:
    """Log records ``start`` to ``start + count - 1``; the number of the next."""
    for number in range(start, start + count):
        logging.getLogger("proteia.test").info("record %03d %s", number, "x" * 60)
    return start + count


def fill_every_file(folder: Path) -> int:
    """Log records until every log file of :data:`CHAIN` is there; the number
    of the next record."""
    number = 0
    while not (folder / CHAIN[0]).exists():
        number = write_records(number, 1)
    return number


def records_in(folder: Path) -> list[int]:
    """The record numbers in the log files, oldest first."""
    return [
        int(n)
        for name in CHAIN
        if (folder / name).exists()
        for n in re.findall(rb"record (\d{3})", (folder / name).read_bytes())
    ]


@pytest.mark.parametrize("held", [logs.LOG_FILE, "proteia.log.1", "proteia.log.5", "proteia.log.9"])
def test_a_rotation_another_program_blocks_loses_no_file(tmp_path, monkeypatch, held):
    folder = tmp_path / logs.LOG_DIR
    logs.setup(folder, max_bytes=500)
    try:
        number = fill_every_file(folder)
        before = {name: (folder / name).read_bytes() for name in CHAIN[:-1]}
        release = hold_open(monkeypatch, held)
        number = write_records(number, 30)  # ten rotations' worth
        # Nothing moved and no file was lost: the records went on into proteia.log.
        assert {name: (folder / name).read_bytes() for name in CHAIN[:-1]} == before
        assert sorted(path.name for path in folder.iterdir()) == sorted(CHAIN)
        assert (folder / logs.LOG_FILE).stat().st_size > 30 * 100
        release()
        number = write_records(number, 1)  # rotates now
    finally:
        logs.shutdown()
    assert sorted(path.name for path in folder.iterdir()) == sorted(CHAIN)
    records = records_in(folder)
    assert records == list(range(records[0], number))  # in order, none lost between
    for n in range(1, logs.BACKUPS):  # each moved up one, the oldest deleted
        assert (folder / f"proteia.log.{n + 1}").read_bytes() == before[f"proteia.log.{n}"]


@pytest.mark.skipif(os.name != "nt", reason="only Windows keeps an open file from moving")
def test_a_rotated_file_open_in_another_program_loses_no_file(tmp_path):
    folder = tmp_path / logs.LOG_DIR
    logs.setup(folder, max_bytes=500)
    try:
        number = fill_every_file(folder)
        before = {name: (folder / name).read_bytes() for name in CHAIN[:-1]}
        with open(folder / "proteia.log.1", "rb"):  # a viewer reading it
            number = write_records(number, 30)
            assert {name: (folder / name).read_bytes() for name in CHAIN[:-1]} == before
        number = write_records(number, 1)
    finally:
        logs.shutdown()
    assert sorted(path.name for path in folder.iterdir()) == sorted(CHAIN)
    records = records_in(folder)
    assert records == list(range(records[0], number))
    for n in range(1, logs.BACKUPS):
        assert (folder / f"proteia.log.{n + 1}").read_bytes() == before[f"proteia.log.{n}"]


# --- A log that cannot be written ---


def test_an_unwritable_log_folder_does_not_stop_the_app(tmp_path, monkeypatch, capsys):
    state = tmp_path / "state"
    state.mkdir()
    (state / logs.LOG_DIR).write_text("a file where the logs folder should be", encoding="utf-8")
    monkeypatch.setattr(launch, "state_dir", lambda: state)
    monkeypatch.setattr(launch, "start", lambda: None)  # another instance is running
    assert launch.main() == 0
    console = capsys.readouterr()
    assert "already running" in console.out
    assert console.err.count("cannot write its log file") == 1
    assert "Logging error" not in console.err

    assert logs.setup(state / logs.LOG_DIR) is None
    try:
        logging.getLogger("proteia.test").warning("still shown")
    finally:
        logs.shutdown()
    assert capsys.readouterr().err.splitlines()[-1] == "still shown"  # as before


def test_no_usable_temporary_folder_does_not_stop_the_app(tmp_path, monkeypatch, capsys):
    def none() -> str:  # as Python finds none on a full disk
        raise FileNotFoundError(2, "No usable temporary directory found")

    monkeypatch.setattr(tempfile, "gettempdir", none)
    monkeypatch.setattr(launch, "state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(launch, "start", lambda: None)  # another instance is running
    assert launch.main() == 0
    assert "already running" in capsys.readouterr().out
    text = (tmp_path / "state" / logs.LOG_DIR / logs.LOG_FILE).read_text(encoding="utf-8")
    assert "Proteia is already running; it has been opened in the browser" in text


class _Broken(io.StringIO):
    def write(self, text: str) -> int:
        raise OSError(28, "No space left on device")


def test_a_log_that_fails_while_running_says_so_once(tmp_path, capsys):
    path = logs.setup(tmp_path / logs.LOG_DIR)
    try:
        say = logging.getLogger("proteia.test")
        say.info("written")
        (file,) = [h for h in logging.getLogger().handlers if isinstance(h, logs._LogFile)]
        file.stream = _Broken()
        say.info("lost")
        say.warning("shown on the console")
        say.error("shown too")
    finally:
        logs.shutdown()
    err = capsys.readouterr().err
    assert err.count("cannot write its log file any more") == 1
    assert "Logging error" not in err
    assert err.splitlines()[1:] == ["shown on the console", "shown too"]  # as before
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [LINE.match(line)[4] for line in lines] == ["written"]
