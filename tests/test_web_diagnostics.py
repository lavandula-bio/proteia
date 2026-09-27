# SPDX-License-Identifier: Apache-2.0
"""The diagnostic file for a bug report (#138): one zip file with the newest
session log files, the versions, and the open project's project.json and export
records; its images only when asked for; a manifest of every file with its
SHA-256, and a README. Never the access token or the files that hold it.
Requests go to a real server on a loopback socket; the file manager is faked."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import platform
import re
import threading
import time
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeClock
from proteia.core import operations as ops
from proteia.core import record, storage
from proteia.web import api, diagnostics, handoff, launch, logs, server
from test_web import _code, _function, _method, _Tags
from test_web_api import LANE_X, ROW, Client, blot_bytes, upload

PROJECT = "β-actin 10 µM"  # beta-actin 10 micro-molar: a non-ASCII project name
OPENING = api.OPENING_HEADER
ANY_DIGEST = "0" * 64  # well formed as the digest of the files listed, but no listing's


@dataclass
class Served:
    """A running Proteia (:class:`Client`) and its per-user state folder."""

    client: Client
    state: Path

    @property
    def folder(self) -> Path:
        return self.state / diagnostics.DIAGNOSTICS_DIR

    def written(self) -> list[Path]:
        return sorted(self.folder.glob("*.zip")) if self.folder.is_dir() else []


@pytest.fixture
def served(tmp_path) -> Iterator[Served]:
    revealed: list[Path] = []
    root = tmp_path / "projects"
    state = tmp_path / "state µ"  # a non-ASCII state folder
    workspace = api.Workspace(root, reveal=revealed.append, clock=FakeClock())
    instance = launch.start(folder=state, opener=lambda url: True, workspace=workspace)
    assert instance is not None
    thread = threading.Thread(target=instance.serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.server.started:
        assert thread.is_alive() and time.monotonic() < deadline
        time.sleep(0.01)
    yield Served(Client(instance.port, instance.token, root, revealed, workspace), state)
    instance.stop()
    thread.join(10)


def write_logs(state: Path, sizes: dict[str, int]) -> dict[str, bytes]:
    """Log files in the state folder's logs folder, by name, each ``size`` bytes
    of numbered lines; their bytes."""
    folder = state / logs.LOG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, size in sizes.items():
        lines = b"".join(f"{name} line {i:06d}\n".encode() for i in range(size))
        data = lines[:size]
        (folder / name).write_bytes(data)
        written[name] = data
    return written


def opened(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """The files of the diagnostic file at ``path``, by name, and its manifest,
    checked: the manifest lists every other file, with its size and SHA-256."""
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        files = {info.filename: archive.read(info) for info in archive.infolist()}
    manifest = json.loads(files[diagnostics.MANIFEST_FILE].decode("utf-8"))
    listed = {entry["name"]: entry for entry in manifest["files"]}
    assert len(listed) == len(manifest["files"])  # each once
    assert set(listed) == set(files) - {diagnostics.MANIFEST_FILE}
    for name, entry in listed.items():
        assert entry["size"] == len(files[name]), name
        assert entry["sha256"] == hashlib.sha256(files[name]).hexdigest(), name
    return files, manifest


def project_with_exports(served: Served, tmp_path: Path, exports: int = 2) -> dict:
    """The project PROJECT: the blot, five lanes, a target boxed in each, and
    ``exports`` exports; its last answer."""
    client = served.client
    client.ok("POST", "/api/projects", {"name": PROJECT})
    status, answer = upload(client, blot_bytes(tmp_path))
    assert status == 201, answer
    image_id = answer["image_id"]
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": f"c{i} µM"} for i in range(5)]})
    protein = client.ok(
        "POST", "/api/proteins", {"name": "β-actin", "role": "target", "image_id": image_id}
    )["protein_id"]
    for lane, x in enumerate(LANE_X):
        body = {"protein_id": protein, "x": x, "y": ROW, "lane_index": lane}
        answer = client.ok("POST", "/api/boxes", body)
    for _ in range(exports):
        answer = client.ok("POST", "/api/export")
    return answer


def listing(served: Served, **headers: str) -> dict:
    return served.client.ok("GET", "/api/diagnostics", headers=headers)


def body(listed: dict, *, images: bool = False) -> dict:
    """The request that writes the file ``listed`` (GET /api/diagnostics's
    answer) shows: for its opening and its project files (``digest``)."""
    return {"images": images, "open_id": listed["open_id"], "digest": listed["digest"]}


def write(served: Served, listed: dict | None = None, *, images: bool = False) -> dict:
    """Write the file as ``listed`` (listed now when not given)."""
    listed = listing(served) if listed is None else listed
    return served.client.ok("POST", "/api/diagnostics", body(listed, images=images))


def project_names(names: Iterable[str]) -> set[str]:
    return {name for name in names if name.startswith("project/")}


# --- What the file holds ---


def test_with_no_project_open_it_holds_the_log_and_the_versions(served):
    logged = write_logs(served.state, {logs.LOG_FILE: 500, f"{logs.LOG_FILE}.1": 300})
    listed = listing(served)
    assert (listed["project"], listed["open_id"], listed["saved"]) == (None, None, None)
    names = [file["name"] for file in listed["files"]]
    assert names == [diagnostics.ENVIRONMENT_FILE, "logs/proteia.log", "logs/proteia.log.1"]
    assert [file["size"] for file in listed["files"][1:]] == [500, 300]
    assert (listed["images"], listed["left_out"]) == ([], [])
    assert listed["added"] == [diagnostics.README_FILE, diagnostics.MANIFEST_FILE]

    written = write(served)
    (path,) = served.written()
    assert (written["name"], written["path"]) == (path.name, str(path))
    assert written["size"] == path.stat().st_size and written["files"] == 5
    files, manifest = opened(path)
    assert sorted(files) == [
        diagnostics.README_FILE,
        diagnostics.ENVIRONMENT_FILE,
        "logs/proteia.log",
        "logs/proteia.log.1",
        diagnostics.MANIFEST_FILE,
    ]
    assert files["logs/proteia.log"] == logged["proteia.log"]
    assert files["logs/proteia.log.1"] == logged["proteia.log.1"]
    assert (manifest["project"], manifest["project_saved"]) == (None, None)
    assert (manifest["images_included"], manifest["left_out"]) == (False, [])
    readme = files[diagnostics.README_FILE].decode("utf-8")
    assert "No project was open" in readme and "sends nothing anywhere" in readme


def test_it_holds_the_open_projects_file_and_export_records_but_no_images(served, tmp_path):
    project_with_exports(served, tmp_path)
    folder = served.client.root / PROJECT
    records = sorted(
        path.relative_to(folder).as_posix() for path in folder.glob("exports/*/*.record.json")
    )
    assert len(records) == 2
    images = sorted(path.name for path in (folder / storage.IMAGES_DIR).iterdir())
    listed = listing(served)
    assert (listed["project"], listed["saved"]) == (PROJECT, True)
    names = [file["name"] for file in listed["files"]]
    assert names[:2] == [diagnostics.ENVIRONMENT_FILE, "project/project.json"]
    assert sorted(names[2:]) == [f"project/{name}" for name in records]
    assert [file["name"] for file in listed["images"]] == [f"project/images/{n}" for n in images]

    write(served, listed)
    files, manifest = opened(served.written()[0])
    assert files["project/project.json"] == (folder / storage.PROJECT_FILE).read_bytes()
    for name in records:
        assert files[f"project/{name}"] == (folder / name).read_bytes()
    assert not [name for name in files if name.startswith("project/images/")]
    # Only the records of each export: no lane table, chart or README of one.
    assert {name.rsplit("/", 1)[0] for name in files if name.startswith("project/")} == {
        "project",
        *(f"project/{name.rsplit('/', 1)[0]}" for name in records),
    }
    assert manifest["project"] == PROJECT and manifest["images_included"] is False
    assert manifest["left_out"] == [
        {
            "name": f"project/images/{name}",
            "size": (folder / storage.IMAGES_DIR / name).stat().st_size,
            "reason": "images are included only when asked for",
        }
        for name in images
    ]
    readme = files[diagnostics.README_FILE].decode("utf-8")
    assert f'"{PROJECT}"' in readme and "not its images" in readme
    assert "2 export records" in readme


def test_with_images_asked_for_the_project_folder_in_it_opens_as_the_project(served, tmp_path):
    project_with_exports(served, tmp_path, exports=1)
    folder = served.client.root / PROJECT
    write(served, images=True)
    files, manifest = opened(served.written()[0])
    (image,) = (folder / storage.IMAGES_DIR).iterdir()
    assert files[f"project/images/{image.name}"] == image.read_bytes()
    assert manifest["images_included"] is True and manifest["left_out"] == []
    # project/ is a copy of the project that Proteia opens, its images verified.
    copy = tmp_path / "unzipped"
    with zipfile.ZipFile(served.written()[0]) as archive:
        archive.extractall(copy)
    session = ops.open_project(copy / "project")
    try:
        assert storage.verify_images(session.project, session.folder) == []
        assert session.project.batch.proteins[0].name == "β-actin"
    finally:
        session.close(remove_files=False)


def test_the_environment_names_versions_but_no_computer_user_or_folder(served, tmp_path):
    write(served)
    files, _ = opened(served.written()[0])
    environment = json.loads(files[diagnostics.ENVIRONMENT_FILE].decode("utf-8"))
    assert environment["software"] == record.software_versions()  # as the record reads them
    assert environment["proteia"] == environment["software"]["proteia"]
    assert environment["python"] == platform.python_version()
    assert environment["os"] == platform.platform()
    assert environment["build"] == "source"
    text = files[diagnostics.ENVIRONMENT_FILE].decode("utf-8")
    node = platform.node()
    assert len(node) < 4 or node not in text
    for folder in (tmp_path, Path.home()):
        assert str(folder) not in text and json.dumps(str(folder))[1:-1] not in text


def test_it_never_holds_the_token_or_the_instance_files(served, tmp_path):
    # The state folder holds the running instance's files (its token in two of
    # them) and an image handed to Proteia, staged in incoming/.
    client, state = served.client, served.state
    status, staged = client.call("POST", "/api/incoming?name=handed.tif", raw=blot_bytes(tmp_path))
    assert status == 201, staged
    assert {launch.INSTANCE_FILE, launch.REDIRECT_FILE, launch.LOCK_FILE} <= {
        path.name for path in state.iterdir()
    }
    assert any((state / handoff.INCOMING_DIR).iterdir())
    project_with_exports(served, tmp_path, exports=1)
    write_logs(state, {logs.LOG_FILE: 200})
    write(served, images=True)
    files, _ = opened(served.written()[0])
    forbidden = (launch.INSTANCE_FILE, launch.REDIRECT_FILE, launch.LOCK_FILE, "incoming")
    assert not [name for name in files if any(part in name for part in forbidden)]
    token = client.token.encode()
    assert not [name for name, data in files.items() if token in data]
    # No full path either: the manifest and the README name files as the zip does.
    for name in (diagnostics.MANIFEST_FILE, diagnostics.README_FILE):
        text = files[name].decode("utf-8")
        assert str(tmp_path) not in text and json.dumps(str(tmp_path))[1:-1] not in text


def test_a_real_session_log_brings_no_token_into_it(tmp_path):
    # The launch conceals its token in the session log (#137); a request with a
    # wrong token and a message that holds the right one are logged, hidden.
    state = tmp_path / "state"
    root = tmp_path / "projects"
    logs.setup(state / logs.LOG_DIR, hidden={root: "<projects>", state: "<state>"})
    try:
        workspace = api.Workspace(root, reveal=lambda folder: None, clock=FakeClock())
        instance = launch.start(folder=state, opener=lambda url: True, workspace=workspace)
        assert instance is not None
        thread = threading.Thread(target=instance.serve, daemon=True)
        thread.start()
        while not instance.server.started:
            assert thread.is_alive()
            time.sleep(0.01)
        client = Client(instance.port, instance.token, root, [], workspace)
        try:
            assert client.call("GET", "/api/project")[0] == 409  # no project: logged
            logging.getLogger("proteia.test").info("the token is %s", instance.token)
            client.ok("POST", "/api/diagnostics", body(client.ok("GET", "/api/diagnostics")))
        finally:
            instance.stop()
            thread.join(10)
    finally:
        logs.shutdown()
    (path,) = (state / diagnostics.DIAGNOSTICS_DIR).glob("*.zip")
    files, _ = opened(path)
    log = files["logs/proteia.log"].decode("utf-8")
    assert "refused GET /api/project: 409 no_project" in log
    assert f"the token is {logs.HIDDEN}" in log
    assert not [name for name, data in files.items() if instance.token.encode() in data]


# --- Limits ---


def test_the_log_is_cut_to_its_newest_files(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "LOG_BYTES", 1000)
    state = tmp_path / "state"
    logged = write_logs(
        state,
        {
            "proteia.log": 300,
            "proteia.log.1": 400,
            "proteia.log.2": 400,  # would pass 1000: left out
            "proteia.log.3": 100,  # would fit, but older than one left out
            "proteia.log.10": 50,  # numbered as a number, not as text: the oldest
            "proteia.log.rotating": 10,  # a rotation's, under way: not a log file
            "notes.txt": 10,
        },
    )
    plan = diagnostics.plan(state, None)
    assert [item.name for item in plan.items[1:]] == ["logs/proteia.log", "logs/proteia.log.1"]
    reason = "older than the newest 1000 bytes of the log"
    assert plan.left_out == (
        diagnostics.LeftOut("logs/proteia.log.2", 400, reason),
        diagnostics.LeftOut("logs/proteia.log.3", 100, reason),
        diagnostics.LeftOut("logs/proteia.log.10", 50, reason),
    )
    written = diagnostics.write(plan, tmp_path / "out", images=False, moment=datetime.now(UTC))
    files, manifest = opened(written.path)
    assert files["logs/proteia.log.1"] == logged["proteia.log.1"]
    assert [entry["name"] for entry in manifest["left_out"]] == [
        "logs/proteia.log.2",
        "logs/proteia.log.3",
        "logs/proteia.log.10",
    ]
    assert written.left_out == 3


def test_a_current_log_longer_than_the_limit_is_cut_to_its_last_lines(tmp_path, monkeypatch):
    # A rotation another program blocks lets proteia.log grow: only its end is
    # taken, from the start of a line, and the manifest says so.
    monkeypatch.setattr(diagnostics, "LOG_BYTES", 1000)
    state = tmp_path / "state"
    logged = write_logs(state, {"proteia.log": 5000, "proteia.log.1": 10})
    plan = diagnostics.plan(state, None)
    assert [(item.name, item.size, item.tail) for item in plan.items[1:]] == [
        ("logs/proteia.log", 1000, True)
    ]
    assert [item.name for item in plan.left_out] == ["logs/proteia.log.1"]
    written = diagnostics.write(plan, tmp_path / "out", images=False, moment=datetime.now(UTC))
    files, manifest = opened(written.path)
    taken = files["logs/proteia.log"]
    assert logged["proteia.log"].endswith(taken) and 900 < len(taken) <= 1000
    assert logged["proteia.log"][-len(taken) - 1 : -len(taken)] == b"\n"  # a whole line
    (entry,) = [e for e in manifest["files"] if e["name"] == "logs/proteia.log"]
    assert entry["note"] == f"the last {len(taken)} bytes of a file of 5000"


def test_only_the_newest_export_records_are_taken(served, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "RECORDS", 2)
    project_with_exports(served, tmp_path, exports=3)
    exports = served.client.root / PROJECT / storage.EXPORTS_DIR
    folders = sorted(path for path in exports.iterdir() if path.is_dir())
    for age, folder in enumerate(reversed(folders)):  # the last one made is the newest
        stamp = time.time() - 100 * age
        os.utime(folder / "export.record.json", (stamp, stamp))
    (exports / "lane-table.record.json").write_text("{}", encoding="utf-8")  # older style
    stamp = time.time() - 1000
    os.utime(exports / "lane-table.record.json", (stamp, stamp))
    listed = listing(served)
    taken = [file["name"] for file in listed["files"] if file["name"].endswith(".record.json")]
    assert taken == [
        f"project/exports/{folder.name}/export.record.json" for folder in folders[:0:-1]
    ]
    left = [file for file in listed["left_out"] if file["name"].endswith(".record.json")]
    assert [file["name"] for file in left] == [
        f"project/exports/{folders[0].name}/export.record.json",
        "project/exports/lane-table.record.json",
    ]
    assert {file["reason"] for file in left} == {"older than the newest 2 export records"}


def test_a_file_gone_before_it_is_written_is_left_out_and_said(tmp_path):
    state = tmp_path / "state"
    write_logs(state, {"proteia.log": 100, "proteia.log.1": 100})
    plan = diagnostics.plan(state, None)
    (state / logs.LOG_DIR / "proteia.log.1").unlink()  # rotated away meanwhile, say
    written = diagnostics.write(plan, tmp_path / "out", images=False, moment=datetime.now(UTC))
    files, manifest = opened(written.path)
    assert "logs/proteia.log.1" not in files
    (left,) = manifest["left_out"]
    assert left["name"] == "logs/proteia.log.1"
    assert left["reason"].startswith("could not be read: ")


def test_files_are_named_by_their_time_numbered_and_only_the_newest_kept(tmp_path, monkeypatch):
    state = tmp_path / "state"
    out = tmp_path / "out"
    out.mkdir()
    (out / ".left.zip.part").write_bytes(b"x")  # a write that did not finish
    (out / "mine.zip").write_bytes(b"x")  # not a diagnostic file: kept
    moment = datetime(2026, 9, 26, 8, 0, tzinfo=UTC)
    stem = f"{diagnostics.FILE_STEM} {moment.astimezone().strftime('%Y-%m-%d %H%M')}"

    def written() -> Path:
        path = diagnostics.write(
            diagnostics.plan(state, None), out, images=False, moment=moment
        ).path
        assert not list(out.glob("*.part"))
        return path

    paths = []
    for age in range(4, 0, -1):  # each written after the one before, in the same minute
        path = written()
        stamp = time.time() - 100 * age
        os.utime(path, (stamp, stamp))
        paths.append(path)
    assert [path.name for path in paths] == [f"{stem}.zip"] + [
        f"{stem} ({n}).zip" for n in range(2, 5)
    ]
    # The next one keeps the KEPT newest, itself among them; nothing else goes.
    monkeypatch.setattr(diagnostics, "KEPT", 3)
    paths.append(written())
    assert paths[-1].name == f"{stem} (5).zip"
    assert sorted(path.name for path in out.iterdir()) == sorted(
        ["mine.zip", *(path.name for path in paths[2:])]
    )


def test_a_write_that_fails_leaves_no_file(tmp_path, monkeypatch):
    state = tmp_path / "state"
    write_logs(state, {"proteia.log": 100})
    out = tmp_path / "out"
    add = diagnostics._add_bytes

    def failing(archive, name, data, moment):
        if name == diagnostics.MANIFEST_FILE:
            raise OSError(28, "No space left on device")
        return add(archive, name, data, moment)

    monkeypatch.setattr(diagnostics, "_add_bytes", failing)
    with pytest.raises(OSError):
        diagnostics.write(
            diagnostics.plan(state, None), out, images=False, moment=datetime.now(UTC)
        )
    assert list(out.iterdir()) == []


# --- The routes ---


def test_a_file_is_written_only_for_the_opening_listed(served, tmp_path):
    # Listed with no project open, then a project is opened (in another tab):
    # the file listed without it is refused, and nothing is written.
    client = served.client
    listed = listing(served)
    assert listed["open_id"] is None
    opened_now = client.ok("POST", "/api/projects", {"name": PROJECT})["project"]
    status, payload = client.call("POST", "/api/diagnostics", body(listed))
    assert (status, payload["code"]) == (409, "project_changed")
    assert payload["detail"] == {"open": PROJECT, "open_id": opened_now["open_id"]}
    assert served.written() == []
    # So is one listed for an earlier opening.
    client.ok("POST", "/api/projects", {"name": "Other"})
    status, payload = client.call(
        "POST", "/api/diagnostics", {**body(listed), "open_id": opened_now["open_id"]}
    )
    assert (status, payload["code"]) == (409, "project_changed")
    assert served.written() == []
    listed = listing(served)
    assert listed["project"] == "Other"
    write(served, listed)
    assert len(served.written()) == 1


def test_project_files_made_since_the_list_are_never_written_unseen(served, tmp_path):
    # Listed, then an export made and an image imported in the same opening
    # (in another tab, say): written as listed, the file would hold files the
    # page never showed. It is refused and nothing is written, whether the
    # images are asked for or not; listed again, the list shows them.
    client = served.client
    project_with_exports(served, tmp_path, exports=1)
    listed = listing(served)
    client.ok("POST", "/api/export")
    for images in (False, True):
        status, code, _ = client.refused("POST", "/api/diagnostics", body(listed, images=images))
        assert (status, code) == (409, "files_changed")
    again = listing(served)
    assert again["open_id"] == listed["open_id"] and again["digest"] != listed["digest"]
    status, answer = upload(client, blot_bytes(tmp_path), name="second µ.tif")
    assert status == 201, answer
    for images in (False, True):
        status, code, _ = client.refused("POST", "/api/diagnostics", body(again, images=images))
        assert (status, code) == (409, "files_changed")
    assert served.written() == []
    # Listed again, it is written with the files shown; an edit since, which
    # changes project.json but none of the files, does not stop it.
    now = listing(served)
    shown = project_names(file["name"] for file in now["files"] + now["images"])
    assert shown - project_names(file["name"] for file in listed["files"] + listed["images"])
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": f"d{i} µM"} for i in range(5)]})
    write(served, now, images=True)
    files, _ = opened(served.written()[0])
    assert project_names(files) == shown
    folder = client.root / PROJECT
    assert files["project/project.json"] == (folder / storage.PROJECT_FILE).read_bytes()


def test_with_changes_not_saved_its_images_are_also_those_project_json_names(
    served, tmp_path, monkeypatch
):
    # Saving fails (a full disk, say) once an image is removed and another
    # imported: project.json, as last saved, still names the one removed, whose
    # file stays until a save. With the images asked for, the file holds the
    # image files project.json names and the one imported, so project/ opens as
    # the project last saved, as the README says.
    client = served.client
    client.ok("POST", "/api/projects", {"name": PROJECT})
    ids = []
    for name in ("first.tif", "second.tif"):
        status, answer = upload(client, blot_bytes(tmp_path), name=name)
        assert status == 201, answer
        ids.append(answer["image_id"])
    folder = client.root / PROJECT
    saved = (folder / storage.PROJECT_FILE).read_bytes()

    def full(project, folder):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(storage, "save_project", full)
    client.ok("DELETE", f"/api/images/{ids[1]}")
    status, answer = upload(client, blot_bytes(tmp_path), name="third.tif")
    assert status == 201, answer
    listed = listing(served)
    assert listed["saved"] is False
    stored = sorted(path.name for path in (folder / storage.IMAGES_DIR).iterdir())
    assert len(stored) == 3  # the removed image's file too, kept until a save
    assert sorted(file["name"] for file in listed["images"]) == [
        f"project/images/{name}" for name in stored
    ]
    write(served, listed, images=True)
    monkeypatch.undo()
    (path,) = served.written()
    files, manifest = opened(path)
    assert files["project/project.json"] == saved
    assert sorted(name for name in files if name.startswith("project/images/")) == [
        f"project/images/{name}" for name in stored
    ]
    assert (manifest["project_saved"], manifest["left_out"]) == (False, [])
    readme = " ".join(files[diagnostics.README_FILE].decode("utf-8").split())
    assert "project/ is a copy of the project that Proteia can open" in readme
    copy = tmp_path / "unzipped"
    with zipfile.ZipFile(path) as archive:
        archive.extractall(copy)
    session = ops.open_project(copy / "project")
    try:
        assert [image.id for image in session.project.batch.iter_images()] == ids
        assert storage.verify_images(session.project, session.folder) == []
    finally:
        session.close(remove_files=False)


def test_the_readme_calls_project_openable_only_with_every_image_file_it_needs(served, tmp_path):
    # An image file that cannot be read when the file is written (removed
    # meanwhile) is left out: project/ then does not open as the project, and
    # the README does not say it does.
    project_with_exports(served, tmp_path, exports=1)
    with served.client.workspace.using_any() as session:
        plan = diagnostics.plan(served.state, session)
    (image,) = plan.images
    gone = dataclasses.replace(
        plan, images=(dataclasses.replace(image, source=tmp_path / "gone.tif"),)
    )
    said = []
    for listed in (plan, gone):
        written = diagnostics.write(listed, tmp_path / "out", images=True, moment=datetime.now(UTC))
        files, manifest = opened(written.path)
        said.append(" ".join(files[diagnostics.README_FILE].decode("utf-8").split()))
    assert "project/ is a copy of the project that Proteia can open" in said[0]
    assert [entry["name"] for entry in manifest["left_out"]] == [image.name]
    assert "project/ is a copy" not in said[1]
    assert "project/ is not a copy of the project that Proteia can open" in said[1]


def test_a_page_showing_another_opening_is_refused_before_anything_is_listed(served):
    client = served.client
    shown = client.ok("POST", "/api/projects", {"name": "Blot"})["project"]["open_id"]
    now = client.ok("POST", "/api/projects", {"name": PROJECT})["project"]["open_id"]
    stale = {OPENING: str(shown)}
    refused = (409, "project_changed")
    assert client.refused("GET", "/api/diagnostics", headers=stale)[:2] == refused
    asked = {"images": False, "open_id": shown, "digest": ANY_DIGEST}
    assert client.refused("POST", "/api/diagnostics", asked, headers=stale)[:2] == refused
    assert served.written() == []
    # Naming the opening open now, it is served.
    listed = listing(served, **{OPENING: str(now)})
    assert (listed["project"], listed["open_id"]) == (PROJECT, now)
    client.ok("POST", "/api/diagnostics", body(listed), headers={OPENING: str(now)})
    assert len(served.written()) == 1


@pytest.mark.parametrize(
    "asked",
    [
        None,
        {"images": False, "digest": ANY_DIGEST},  # which opening was listed: required
        {"images": False, "open_id": None},  # which files were listed: required
        {"images": "yes", "open_id": None, "digest": ANY_DIGEST},
        {"images": False, "open_id": "1", "digest": ANY_DIGEST},
        {"images": False, "open_id": None, "digest": ANY_DIGEST[:-1]},  # not a SHA-256
        {"images": False, "open_id": None, "digest": ANY_DIGEST.replace("0", "A")},
        {"images": False, "open_id": None, "digest": ANY_DIGEST, "folder": "C:/"},
    ],
)
def test_a_request_the_route_cannot_read_writes_nothing(served, asked):
    status, code, _ = served.client.refused("POST", "/api/diagnostics", asked)
    assert (status, code) == (422, "invalid_input")
    assert served.written() == []


def test_the_routes_need_the_token(served):
    # The server refuses a request without the token before any route
    # (server.Guard), whatever its path: so each route answers 401 without it,
    # nothing written or shown, and is served with it.
    asked = [
        ("GET", "/api/diagnostics", None, 200),
        ("POST", "/api/diagnostics", body(listing(served)), 201),
        ("POST", "/api/diagnostics/reveal", None, 204),
    ]
    wrong = {"Authorization": "Bearer " + "x" * 43}
    for method, path, sent, _ in asked:
        status, payload = served.client.call(method, path, sent, headers=wrong)
        assert status == 401, (method, path, payload)
    assert served.written() == [] and served.client.revealed == []
    for method, path, sent, answered in asked:
        assert served.client.call(method, path, sent)[0] == answered, (method, path)
    assert len(served.written()) == 1 and served.client.revealed == [served.folder]


def test_the_diagnostics_folder_is_shown_in_the_file_manager(served):
    assert not served.folder.exists()
    assert served.client.call("POST", "/api/diagnostics/reveal")[0] == 204
    assert served.client.revealed == [served.folder]
    assert served.folder.is_dir()  # made, to be shown


def test_writing_the_file_is_logged(served, tmp_path, caplog):
    project_with_exports(served, tmp_path, exports=1)
    listed = listing(served)
    with caplog.at_level(logging.INFO, logger=api.__name__):
        written = write(served, listed, images=True)
        served.client.call("POST", "/api/diagnostics/reveal")
    logged = [r.getMessage() for r in caplog.records if r.name == api.__name__]
    assert (
        f"wrote the diagnostic file {written['name']!r}: {written['files']} files,"
        f" {written['size']} bytes, 0 left out; {PROJECT!r}, with its images"
    ) in logged
    assert "showed the diagnostics folder in the file manager" in logged


def test_without_a_state_folder_there_is_no_diagnostic_file(tmp_path):
    workspace = api.Workspace(tmp_path / "projects", reveal=lambda folder: None)
    with pytest.raises(api.NoStateFolderError):
        workspace.state_folder()
    with workspace.using_any() as session:
        assert session is None  # no project open, none named: nothing in use
    with pytest.raises(api.NoProjectError), workspace.using_any(1):
        pass


def test_served_without_a_state_folder_the_routes_answer_so(served):
    # Proteia served other than by a launch has no state folder: each route
    # answers no_state_folder, and nothing is written or shown.
    served.client.workspace.state = None
    asked = {"images": False, "open_id": None, "digest": ANY_DIGEST}
    for method, path, sent in (
        ("GET", "/api/diagnostics", None),
        ("POST", "/api/diagnostics", asked),
        ("POST", "/api/diagnostics/reveal", None),
    ):
        status, payload = served.client.call(method, path, sent)
        assert status == 409, (method, path, status, payload)
        assert payload["code"] == "no_state_folder", (method, path, payload)
    assert served.written() == [] and served.client.revealed == []
    assert not served.folder.exists()


# --- The page ---


def _dialog(html: str, dialog_id: str) -> str:
    start = html.index(f'<dialog id="{dialog_id}"')
    return html[start : html.index("</dialog>", start)]


def test_the_page_offers_the_file_in_the_header_and_in_the_projects_dialog():
    # With no project open the Projects dialog is modal: the header's
    # Diagnostics… cannot be reached, so the dialog has one of its own.
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    parser = _Tags()
    parser.feed(html)
    for control in ("diagnostics", "projects-diagnostics"):
        fields = parser.by_id[control]
        assert (fields["type"], fields["aria-haspopup"]) == ("button", "dialog")
        assert "Proteia sends nothing" in fields["title"]
    assert "hidden" in parser.by_id["diagnostics"]  # until the page reaches Proteia
    assert 'id="projects-diagnostics"' in _dialog(html, "projects-dialog")
    dialog = _dialog(html, "diagnostics-dialog")
    assert "Proteia sends nothing anywhere" in dialog
    images = parser.by_id["diagnostics-images"]
    assert images["type"] == "checkbox" and "checked" not in images  # images: only when ticked
    assert "disabled" in parser.by_id["diagnostics-write"]  # until the list is in
    assert parser.by_id["diagnostics-state"]["role"] == "status"
    assert parser.by_id["diagnostics-error"]["role"] == "alert"
    # Every element the dialog's script uses is in the page.
    script = _code("diagnostics.js")
    used = set(re.findall(r'\$\("([\w-]+)"\)', script))
    assert used <= set(parser.by_id), used - set(parser.by_id)
    assert all(f'id="{name}"' in dialog for name in used)


def test_the_page_lists_the_file_before_writing_it_for_the_opening_listed():
    script = _code("diagnostics.js")
    assert 'const PROJECT_CHANGED = "project_changed";' in script
    assert 'const FILES_CHANGED = "files_changed";' in script
    show = _method(script, "async show(")
    assert show.index('$("diagnostics-images").checked = false') < show.index("this.load()")
    assert show.index("showModal()") < show.index("this.load()")
    load = _method(script, "async load(")
    assert 'this.ask("GET", "/api/diagnostics")' in load
    # A page showing a project no longer open lists again once it shows the one open.
    assert load.index("error.code === PROJECT_CHANGED") < load.index("await this.settled()")
    # Written for the opening, and the project files, listed; the images only
    # when ticked. Refused for another, or for other files, it lists again,
    # saying why.
    write = _method(script, "async write(")
    assert 'this.ask("POST", "/api/diagnostics", {' in write
    assert "open_id: listing.open_id," in write
    assert "digest: listing.digest," in write
    assert '$("diagnostics-images").checked && listing.images.length > 0' in write
    assert "AGAIN.has(error.code)" in write
    assert "const AGAIN = new Set([PROJECT_CHANGED, NO_PROJECT, FILES_CHANGED]);" in script
    assert "error.code === FILES_CHANGED" in write
    assert '"/api/diagnostics/reveal"' in _method(script, "async reveal(")
    # The page wires it: its requests go as every other (request()), naming the
    # opening shown, and its follow is waited for.
    app = _code("app.js")
    wiring = app[app.index("const diagnostics = new DiagnosticsDialog({") :].split("});", 1)[0]
    assert "request(method, path, { json, answer: true })" in wiring
    assert "following.done" in wiring
    for control in ("diagnostics", "projects-diagnostics"):
        assert f'$("{control}").addEventListener("click", () => diagnostics.show());' in app
    assert '$("diagnostics").hidden = false;' in _function(app, "async function start(")[1]


def test_non_ascii_names_are_kept_in_the_file(served, tmp_path):
    # The project's name (β, µ) in the manifest and the README, and a file name
    # in it (an export folder renamed by hand) as UTF-8, flagged so, which
    # every unzip tool reads as written.
    project_with_exports(served, tmp_path, exports=1)
    exports = served.client.root / PROJECT / storage.EXPORTS_DIR
    (folder,) = [path for path in exports.iterdir() if path.is_dir()]
    folder.rename(exports / "α export µ")
    write(served)
    (path,) = served.written()
    files, manifest = opened(path)
    name = "project/exports/α export µ/export.record.json"
    assert name in files and manifest["project"] == PROJECT
    with zipfile.ZipFile(path) as archive:
        assert archive.getinfo(name).flag_bits & 0x800  # the name is UTF-8
    assert f'"{PROJECT}"' in files[diagnostics.README_FILE].decode("utf-8")
    assert served.state.name == "state µ" and path.parent == served.folder
