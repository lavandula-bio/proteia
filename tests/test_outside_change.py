# SPDX-License-Identifier: Apache-2.0
"""A ``project.json`` changed outside Proteia is never saved over without being
kept (#101), and the same document in other bytes is taken as saved (#229).

A save that finds the file changed since the session last read or wrote it
keeps it, byte for byte, in a conflict copy beside it, then saves the
session's version as usual; the page is told the copy's name and time; the
image files any other version names are never deleted, and no import takes
their ids. Folder names hold CJK and Greek letters; messages are compared word
for word; local times are built from the injected clock.
"""

from __future__ import annotations

import codecs
import dataclasses
import errno
import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from conftest import FakeClock, sample_doc_v1
from proteia.core import operations as ops
from proteia.core import storage
from proteia.core.model import ImageKind, LogEntry, Polarity, Project, Role
from proteia.core.session import (
    SET_ASIDE_OUTSIDE,
    Divergence,
    ErrorCode,
    Lineage,
    OperationError,
    ProjectSession,
    Resolution,
    divergence,
    resolve,
)
from proteia.web import api, launch, projects, server
from test_operations import FOLDER, v1_folder
from test_web_api import Client, blot_bytes, upload

CHEMI, DARK = ImageKind.CHEMILUMINESCENCE, Polarity.DARK_ON_LIGHT
START = datetime(2026, 9, 26, 8, 0, tzinfo=UTC)  # FakeClock's start: the session's clock
ELSEWHERE = START + timedelta(hours=1, minutes=30)  # the other computer's clock
MINUTE = START.astimezone().strftime("%Y-%m-%d %H%M")
COPY = f"project.conflict {MINUTE}.json"
DAY, HOUR = START.astimezone().strftime("%Y-%m-%d"), START.astimezone().strftime("%H:%M")
THEIR_DAY = ELSEWHERE.astimezone().strftime("%Y-%m-%d")


def their_hour(seconds: int) -> str:
    """The local hour and minute of the other computer's clock ``seconds`` in."""
    return (ELSEWHERE + timedelta(seconds=seconds)).astimezone().strftime("%H:%M")


def tiff(seed: int) -> bytes:
    """A small 16-bit TIFF of noise, different for each seed."""
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, rng.integers(1000, 3000, (40, 60), dtype=np.uint16))
    return buffer.getvalue()


def imported(session: ProjectSession, seed: int, name: str = "blot µ.tif") -> str:
    return ops.import_image(session, io.BytesIO(tiff(seed)), name, kind=CHEMI, polarity=DARK)


def opened(tmp_path: Path, *, images: int = 1) -> ProjectSession:
    """A new project in a non-ASCII folder, saving at each change, with ``images``."""
    session = ops.new_project(tmp_path / FOLDER, clock=FakeClock())
    for seed in range(1, images + 1):
        imported(session, seed)
    return session


def elsewhere(session: ProjectSession, start: datetime = ELSEWHERE) -> ProjectSession:
    """The project as another computer opens it from the same folder (a synced
    one): never saving, so the test puts its file in place (:func:`put`)."""
    return ops.open_project(session.folder, autosave=None, clock=FakeClock(start))


def put(folder: Path, project: Project) -> bytes:
    """Write ``project`` as ``folder``'s project.json, as a sync would; its bytes."""
    data = storage.project_to_json(project)
    (folder / storage.PROJECT_FILE).write_bytes(data)
    return data


def edit(session: ProjectSession, label: str = "vehicle") -> None:
    ops.set_lanes(session, [ops.LaneInput(label)])


def copies(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.lower().startswith("project.conflict"))


def images(folder: Path) -> list[str]:
    return sorted(p.name for p in (folder / storage.IMAGES_DIR).iterdir())


def file_bytes(session: ProjectSession) -> bytes:
    return (session.folder / storage.PROJECT_FILE).read_bytes()


def go_on(name: str, which: str = "that version", copy: str = "the copy") -> str:
    # Works with only one project, and where file name extensions are hidden.
    return (
        f"To go on from {which} instead: quit Proteia, then in the project's folder rename"
        f" project.json to project mine.json and {copy} to project.json (where file name"
        " extensions are hidden they show without .json: type project mine and project),"
        f" then start Proteia and open {name!r}."
    )


def set_aside(name: str, copy: str = COPY) -> str:
    return (
        f"{name!r} was changed outside Proteia after Proteia last opened or saved it. Proteia"
        f" kept that version as {copy!r} in the project's folder, set aside on {DAY} at {HOUR},"
        " and saves its own in its place."
    )


# --- Detection and the copy ---


def test_a_save_keeps_the_version_changed_outside_as_a_conflict_copy(tmp_path):
    s = opened(tmp_path)
    other = elsewhere(s)
    ops.set_lanes(other, [ops.LaneInput("drug")])
    theirs = put(s.folder, other.project)
    edit(s, "vehicle")
    assert (s.save_error, s.dirty) == (None, False)
    assert file_bytes(s) == storage.project_to_json(s.project)
    assert copies(s.folder) == [COPY]
    assert (s.folder / COPY).read_bytes() == theirs  # byte for byte


def test_the_copy_is_named_by_the_local_minute_and_numbered_when_taken(tmp_path):
    s = opened(tmp_path, images=0)
    folder = s.folder
    (folder / COPY).write_bytes(b"the user's own file")
    (folder / f"project.conflict {MINUTE} (2).json").mkdir()
    (folder / f"PROJECT.CONFLICT {MINUTE} (3).JSON").write_bytes(b"in other case")
    ignores_case = (folder / f"project.conflict {MINUTE} (3).json").exists()
    (folder / storage.PROJECT_FILE).write_bytes(b'{"changed": "outside"}')
    theirs = (folder / storage.PROJECT_FILE).read_bytes()
    edit(s)
    number = 4 if ignores_case else 3
    made = folder / f"project.conflict {MINUTE} ({number}).json"
    assert made.read_bytes() == theirs
    assert (folder / COPY).read_bytes() == b"the user's own file"  # nothing replaced
    assert (folder / f"project.conflict {MINUTE} (2).json").is_dir()
    assert s.outside_change["copies"] == [made.name]


def test_a_path_too_long_for_the_dated_name_gets_the_short_name(tmp_path, monkeypatch):
    s = opened(tmp_path, images=0)
    (s.folder / storage.PROJECT_FILE).write_bytes(b'{"changed": "outside"}')
    room = len(os.path.abspath(s.folder).encode("utf-16-le")) // 2 + 1
    monkeypatch.setattr(storage, "PATH_LIMIT", room + 30)  # 21 fits, 37 does not
    edit(s)
    assert copies(s.folder) == ["project.conflict.json"]
    assert s.save_error is None


def test_a_folder_too_deep_for_any_copy_refuses_the_save(tmp_path, monkeypatch):
    s = opened(tmp_path, images=0)
    project_file = s.folder / storage.PROJECT_FILE
    project_file.write_bytes(b'{"changed": "outside"}')
    room = len(os.path.abspath(s.folder).encode("utf-16-le")) // 2 + 1
    monkeypatch.setattr(storage, "PATH_LIMIT", room + 20)  # not even the short name
    edit(s)
    assert isinstance(s.save_error, storage.CopyNotKeptError) and s.dirty
    assert str(s.save_error) == (
        f"{FOLDER!r} was changed outside Proteia after Proteia last opened or saved it, and"
        " Proteia could not keep that version in a copy (the project folder's path is too long),"
        " so it did not save over it; this change is kept here and saved, with the copy, at"
        " your next change once the folder can take it"
    )
    assert project_file.read_bytes() == b'{"changed": "outside"}' and copies(s.folder) == []


def test_one_copy_per_outside_version(tmp_path, monkeypatch):
    s = opened(tmp_path)
    project_file = s.folder / storage.PROJECT_FILE
    project_file.write_bytes(b'{"changed": "outside"}')
    edit(s, "a")
    edit(s, "b")  # over the file it wrote: no second copy
    assert copies(s.folder) == [COPY]

    # A save that fails after its copy keeps no second one when it runs again.
    project_file.write_bytes(b'{"changed": "again"}')
    real = storage.os.replace

    def held(src, dst, **kwargs):
        raise PermissionError(13, "held by another process", str(dst))

    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    monkeypatch.setattr(storage.os, "replace", held)
    edit(s, "c")
    assert s.dirty and isinstance(s.save_error, PermissionError)
    second = f"project.conflict {MINUTE} (2).json"
    assert sorted(copies(s.folder)) == sorted([COPY, second])
    said = s.outside_change
    assert said["event"] == 2 and said["copies"] == [second, COPY]
    monkeypatch.setattr(storage.os, "replace", real)
    edit(s, "d")
    assert sorted(copies(s.folder)) == sorted([COPY, second]) and not s.dirty
    assert (s.folder / second).read_bytes() == b'{"changed": "again"}'
    # Not said again: a page that dismissed the notice of that copy keeps it so.
    assert s.outside_change == said
    assert f"Proteia kept that version as {second!r}" in said["message"]


def _reordered(value: Any) -> Any:
    """``value`` with every object's keys in reverse order."""
    if isinstance(value, dict):
        return {key: _reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reordered(item) for item in value]
    return value


def _rewrites() -> dict[str, Any]:
    def bom(data: bytes) -> bytes:
        return b"\xef\xbb\xbf" + data

    def crlf(data: bytes) -> bytes:
        return data.replace(b"\n", b"\r\n")

    def reordered(data: bytes) -> bytes:
        doc = json.loads(data)
        return json.dumps(_reordered(doc), ensure_ascii=False).encode("utf-8")

    def indented(data: bytes) -> bytes:
        return json.dumps(json.loads(data), indent=4, ensure_ascii=False).encode("utf-8")

    def everything(data: bytes) -> bytes:
        return bom(crlf(reordered(data)))

    def empty_undetected(data: bytes) -> bytes:
        doc = json.loads(data)
        for protein in doc["batch"]["proteins"]:
            for row in protein["rows"]:
                row["undetected"] = []  # Proteia leaves it out
        return json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8")

    return {
        "bom": bom,
        "crlf": crlf,
        "reordered": reordered,
        "indented": indented,
        "all": everything,
        "undetected": empty_undetected,
    }


@pytest.mark.parametrize("rewrite", sorted(_rewrites()))
def test_the_same_document_in_other_bytes_is_taken_as_saved(tmp_path, rewrite, caplog):
    s = opened(tmp_path)
    ops.add_protein(s, "β-actin µ", Role.TARGET, s.project.batch.images[0].id)
    project_file = s.folder / storage.PROJECT_FILE
    rewritten = _rewrites()[rewrite](project_file.read_bytes())
    assert rewritten != project_file.read_bytes()
    project_file.write_bytes(rewritten)
    caplog.set_level(logging.INFO, logger="proteia.core.session")
    edit(s)
    assert copies(s.folder) == [] and s.outside_change is None
    assert file_bytes(s) == storage.project_to_json(s.project)
    said = f"in {FOLDER!r}: its project.json was rewritten outside Proteia with the same content;"
    assert f"{said} saved over it as usual" in caplog.messages


def test_a_file_holding_what_proteia_is_about_to_write_needs_no_copy(tmp_path):
    s = opened(tmp_path)
    s.autosave = None
    edit(s)  # committed, not saved
    (s.folder / storage.PROJECT_FILE).write_bytes(
        b"\xef\xbb\xbf" + storage.project_to_json(s.project)
    )
    s.save()
    assert copies(s.folder) == [] and file_bytes(s) == storage.project_to_json(s.project)


def test_other_content_in_other_bytes_is_a_change(tmp_path):
    s = opened(tmp_path)
    edit(s, "vehicle")
    doc = json.loads(file_bytes(s))
    doc["batch"]["lane_tables"][0]["lanes"][0]["label"] = "drug"
    theirs = b"\xef\xbb\xbf" + json.dumps(doc).encode("utf-8")
    (s.folder / storage.PROJECT_FILE).write_bytes(theirs)
    edit(s, "vehicle µ")
    assert copies(s.folder) == [COPY] and (s.folder / COPY).read_bytes() == theirs


@pytest.mark.parametrize("change", ["log", "next_id"])
def test_a_log_or_next_id_difference_alone_is_a_change(tmp_path, change):
    s = opened(tmp_path)
    edit(s, "vehicle")
    if change == "log":  # changed and changed back: the same content, a longer log
        other = elsewhere(s)
        ops.set_lanes(other, [ops.LaneInput("drug")])
        ops.set_lanes(other, [ops.LaneInput("vehicle")])
        assert storage.content_hash(other.project) == storage.content_hash(s.project)
        theirs = put(s.folder, other.project)
    else:
        doc = json.loads(file_bytes(s))
        doc["next_id"] += 5
        theirs = storage.document_bytes(doc)
        (s.folder / storage.PROJECT_FILE).write_bytes(theirs)
    edit(s, "drug")
    assert copies(s.folder) == [COPY] and (s.folder / COPY).read_bytes() == theirs


def test_a_migrated_open_saves_over_the_bytes_it_read(tmp_path):
    folder = v1_folder(tmp_path)
    s = ops.open_project(folder, clock=FakeClock())  # migrated, kept in a backup, saved
    assert (s.dirty, s.save_error) == (False, None)
    assert copies(folder) == [] and (folder / "project.schema1.json").is_file()
    ops.set_reference_condition(s, None)
    assert copies(folder) == [] and s.save_error is None


def test_a_migrated_open_whose_save_failed_compares_its_bytes_only(tmp_path, monkeypatch):
    folder = v1_folder(tmp_path)

    def held(src, dst, **kwargs):
        raise PermissionError(13, "held by another process", str(dst))

    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "replace", held)
        s = ops.open_project(folder, clock=FakeClock())
    assert s.dirty  # its migration not saved: the file is the base
    assert (s._base.log, s._base.canonical, s._base.parsed) == (len(s.project.log) - 1, None, None)
    # No canonical form to compare with: the same file in other bytes is kept.
    project_file = folder / storage.PROJECT_FILE
    theirs = b"\xef\xbb\xbf" + project_file.read_bytes()
    project_file.write_bytes(theirs)
    ops.set_reference_condition(s, None)
    assert (folder / COPY).read_bytes() == theirs and not s.dirty


def test_a_missing_image_file_still_refuses_the_save(tmp_path):
    s = opened(tmp_path)
    before = file_bytes(s)
    (s.folder / storage.IMAGES_DIR / s.project.batch.images[0].file).unlink()
    edit(s)
    assert isinstance(s.save_error, storage.MissingImageError) and s.dirty
    assert file_bytes(s) == before and copies(s.folder) == []


def test_a_written_or_read_base_is_the_bytes_and_their_document(tmp_path):
    s = opened(tmp_path, images=0)
    data = file_bytes(s)
    digest = hashlib.sha256(data).hexdigest()
    assert (s._base.sha256, s._base.canonical, s._base.log) == (digest, digest, 1)
    reopened = ops.open_project(s.folder, clock=FakeClock())
    assert (reopened._base.sha256, reopened._base.canonical, reopened._base.log) == (
        digest,
        None,
        1,
    )
    assert reopened._base.parsed == reopened.project
    assert reopened._base_canonical() == digest and reopened._base.parsed is None
    with pytest.raises(dataclasses.FrozenInstanceError):  # replaced whole, never changed
        reopened._base.sha256 = "0" * 64  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kind", "words"),
    [
        ("not json", "It does not read as a project, so Proteia cannot open it."),
        ("cut short", "It does not read as a project, so Proteia cannot open it."),
        ("empty", "It does not read as a project, so Proteia cannot open it."),
        (
            "newer",
            "It was saved by a newer Proteia (schema 99): to go on from it, use that version"
            " of Proteia.",
        ),
    ],
)
def test_an_outside_version_that_is_not_a_project_is_kept_as_it_was(tmp_path, kind, words):
    s = opened(tmp_path)
    data = file_bytes(s)
    if kind == "not json":
        theirs = b"{not json at all"
    elif kind == "cut short":
        theirs = data[: len(data) // 2]
    elif kind == "empty":
        theirs = b""
    else:
        doc = json.loads(data)
        doc["schema_version"] = 99
        theirs = json.dumps(doc).encode("utf-8")
    (s.folder / storage.PROJECT_FILE).write_bytes(theirs)
    edit(s)
    assert (s.folder / COPY).read_bytes() == theirs
    assert s.outside_change == {
        "event": 1,
        "message": f"{set_aside(FOLDER)} {words}",
        "copies": [COPY],
    }


def test_an_older_schema_outside_is_kept_and_counted(tmp_path):
    s = opened(tmp_path, images=0)
    theirs = json.dumps(sample_doc_v1()).encode("utf-8")  # another project, of schema 1
    (s.folder / storage.PROJECT_FILE).write_bytes(theirs)
    edit(s)
    assert (s.folder / COPY).read_bytes() == theirs
    version = storage.read_version(theirs)
    assert version.project is not None and version.newer is None
    assert version.project.log == ()  # the sample keeps no log
    assert s.outside_change["message"] == (
        f"{set_aside(FOLDER)} That version's log has no change this one does not, so it is"
        f" older or was edited by hand; this one has 2 changes it does not. {go_on(FOLDER)}"
    )


def test_the_notice_counts_what_each_version_has(tmp_path):
    # A fork: this computer saved a change the other never saw; each then made its own.
    s = opened(tmp_path)
    other = elsewhere(s)  # opens before the change below
    edit(s, "vehicle")
    for label in ("a", "b", "c"):
        ops.set_lanes(other, [ops.LaneInput(label)])
    put(s.folder, other.project)
    edit(s, "drug")
    assert s.outside_change["message"] == (
        f"{set_aside(FOLDER)} That version has 3 changes this one does not, the last on"
        f" {THEIR_DAY} at {their_hour(2)}, and this one has 2 it does not. {go_on(FOLDER)}"
    )


def test_the_notice_says_when_the_other_version_continues_this_one(tmp_path):
    # The other computer went on from this one's last save (left open here).
    s = opened(tmp_path)
    other = elsewhere(s)
    for label in ("a", "b", "c", "d", "e"):
        ops.set_lanes(other, [ops.LaneInput(label)])
    put(s.folder, other.project)
    edit(s, "stale")
    assert s.outside_change["message"] == (
        f"{set_aside(FOLDER)} That version has 5 changes this one does not, the last on"
        f" {THEIR_DAY} at {their_hour(4)}, and this one has 1 it does not. {go_on(FOLDER)}"
    )


def test_the_notice_says_when_the_other_version_is_older(tmp_path):
    s = opened(tmp_path)
    older = file_bytes(s)  # a backup, restored below
    edit(s, "a")
    edit(s, "b")
    (s.folder / storage.PROJECT_FILE).write_bytes(older)
    edit(s, "c")
    assert s.outside_change["message"] == (
        f"{set_aside(FOLDER)} That version's log has no change this one does not, so it is"
        f" older or was edited by hand; this one has 3 changes it does not. {go_on(FOLDER)}"
    )


def _entry(seq: int) -> LogEntry:
    return LogEntry(
        seq=seq,
        time=f"2026-09-26T08:00:{seq:02d}.000Z",
        action="set_lanes",
        version="0.1",
        params={},
        content_hash="0" * 64,
    )


def _log(*seqs: int, other: int | None = None) -> list[LogEntry]:
    entries = [_entry(seq) for seq in seqs]
    if other is not None:
        entries.append(_entry(other).model_copy(update={"action": "remove_box"}))
    return entries


@pytest.mark.parametrize(
    ("ours", "base", "theirs", "expected"),
    [
        # Its log is the base's (2 entries) and goes on.
        (_log(1, 2, 3), 2, _log(1, 2, 4, other=5), (Lineage.EXTENDS_BASE, 2, 2, 1, "05")),
        # The base's log, as it is: changed by hand.
        (_log(1, 2, 3), 2, _log(1, 2), (Lineage.SAME_LOG, 2, 0, 1, None)),
        # It ends inside the base's log: an older copy.
        (_log(1, 2, 3, 4), 3, _log(1, 2), (Lineage.BEHIND_BASE, 2, 0, 2, None)),
        # The first entry in common, then its own.
        (_log(1, 2, 3), 2, _log(1, other=2), (Lineage.FORK, 1, 1, 2, "02")),
        # Nothing in common.
        (_log(1, 2), 2, _log(other=1), (Lineage.UNRELATED, 0, 1, 2, "01")),
        # No log at all, here or there.
        ([], 0, [], (Lineage.SAME_LOG, 0, 0, 0, None)),
    ],
)
def test_divergence_classifies_the_logs_once(ours, base, theirs, expected):
    lineage, common, their_count, our_count, last = expected
    found = divergence(ours, base, theirs)
    time = None if last is None else f"2026-09-26T08:00:{last}.000Z"
    assert found == Divergence(lineage, common, their_count, our_count, time)
    # One rule today, whatever the logs: the file as found into the copy, then a save.
    assert resolve(found) is SET_ASIDE_OUTSIDE is Resolution.SET_ASIDE_OUTSIDE
    assert resolve(None) is SET_ASIDE_OUTSIDE
    # Only the two that make sense can be written: the outside version into the
    # copy and a save, or the session's into the copy and a switch to the file.
    assert [r.value for r in Resolution] == ["set_aside_outside", "set_aside_session"]
    for value, field in ((found, "common"), (storage.Version(None, None), "newer")):
        with pytest.raises(dataclasses.FrozenInstanceError):  # values, never changed
            setattr(value, field, None)


def test_reading_again_takes_the_file_whatever_the_session_held(tmp_path):
    # What a switch to the file on disk would run (reload runs it today).
    s = opened(tmp_path)
    s.autosave = None
    edit(s, "unsaved")
    (s.folder / "project.conflict 2026-10-01 0800.json").write_bytes(b"{}")
    other = elsewhere(s)
    ops.set_lanes(other, [ops.LaneInput("drug")])
    theirs = put(s.folder, other.project)
    s.pixels(s.project.batch.images[0].id)
    with s.lock:
        s._read_again(theirs)
    assert s.project == other.project and (s.dirty, s.save_error, s.last_action) == (
        False,
        None,
        None,
    )
    assert (s._pixels, s.history_steps) == ({}, (None, None))
    assert s._base.sha256 == hashlib.sha256(theirs).hexdigest()
    assert s.outside_change["copies"] == ["project.conflict 2026-10-01 0800.json"]
    edit(s, "after")  # over the file it read: no copy
    assert copies(s.folder) == ["project.conflict 2026-10-01 0800.json"]


# --- A removed file, a removed folder ---


def test_a_removed_project_json_is_written_again_and_said(tmp_path, caplog):
    s = opened(tmp_path)
    (s.folder / storage.PROJECT_FILE).unlink()
    caplog.set_level(logging.INFO, logger="proteia.core.session")
    edit(s)
    assert file_bytes(s) == storage.project_to_json(s.project) and s.save_error is None
    assert s.outside_change == {
        "event": 1,
        "message": (
            f"The project.json of {FOLDER!r} was removed outside Proteia after Proteia last"
            " opened or saved it, so Proteia wrote it again."
        ),
        "copies": [],
    }
    assert f"in {FOLDER!r}: its project.json was removed outside Proteia; wrote it again" in (
        caplog.messages
    )
    assert copies(s.folder) == []


REMOVED = (
    f"{FOLDER!r} is no longer in the projects folder as it was (its folder, or its"
    " project.json and an image file, were removed or moved outside Proteia), so"
    " Proteia did not save it again; to keep this change, put the folder back as it"
    " was: opening another project or quitting drops it"
)


@pytest.mark.parametrize("count", [0, 1])
def test_a_removed_project_folder_is_not_made_again(tmp_path, count):
    s = opened(tmp_path, images=count)
    shutil.rmtree(s.folder)
    edit(s)
    assert not s.folder.exists()
    assert isinstance(s.save_error, storage.ProjectRemovedError) and s.dirty
    assert str(s.save_error) == REMOVED


def test_a_folder_replaced_by_a_file_is_a_removed_folder(tmp_path):
    s = opened(tmp_path)
    shutil.rmtree(s.folder)
    s.folder.write_bytes(b"a file where the folder was")
    edit(s)
    assert str(s.save_error) == REMOVED
    assert s.folder.read_bytes() == b"a file where the folder was"


def test_a_project_json_gone_with_an_image_is_a_removed_folder(tmp_path):
    s = opened(tmp_path)
    shutil.rmtree(s.folder)
    (s.folder / storage.EXPORTS_DIR).mkdir(parents=True)  # an export made it again
    edit(s)
    assert str(s.save_error) == REMOVED
    assert sorted(p.name for p in s.folder.iterdir()) == [storage.EXPORTS_DIR]


# Why a switch or a quit dropped a project's changes, as the log and the page say it.
GONE = "its folder, or its project.json and an image file, were removed or moved outside Proteia"


def test_switching_and_quitting_go_on_after_the_folder_was_removed(tmp_path, caplog):
    root = tmp_path / "root"
    workspace = api.Workspace(root, reveal=lambda folder: None, clock=FakeClock())
    first = workspace.create("A µ")
    imported(first, 1)
    shutil.rmtree(first.folder)
    edit(first)
    assert first.dirty
    caplog.set_level(logging.WARNING, logger="proteia.web.api")
    second = workspace.create("B β")  # not refused: A can never be saved again
    assert workspace.current() is second and not (root / "A µ").exists()
    assert caplog.messages.count(f"closed 'A µ' with changes not saved: {GONE}") == 1
    # For a page that follows: A's opening closed with its changes not saved.
    assert workspace.closed() == [{"open_id": 1, "name": "A µ", "saved": False, "why": GONE}]

    imported(second, 2)
    shutil.rmtree(second.folder)
    edit(second)
    said = f"'B β' was closed with changes not saved: {GONE}"
    assert workspace.flush() == said  # a quit goes on too, and the launcher says it
    workspace.close()
    assert caplog.messages.count(f"closed 'B β' with changes not saved: {GONE}") == 1
    assert not (root / "B β").exists()


def test_a_switch_records_whether_the_project_it_closed_was_saved(tmp_path):
    root = tmp_path / "root"
    workspace = api.Workspace(root, reveal=lambda folder: None, clock=FakeClock())
    workspace.create("A µ")
    workspace.create("B β")  # A saved
    assert workspace.closed() == [{"open_id": 1, "name": "A µ", "saved": True, "why": None}]
    for n in range(api.CLOSED_KEPT + 3):
        workspace.create(f"C {n}")
    kept = workspace.closed()
    assert len(kept) == 16 and kept[0]["open_id"] == 5 and kept[-1]["name"] == "C 17"
    assert workspace.flush() is None  # nothing unsaved: nothing dropped


def test_a_switch_from_a_project_whose_file_and_image_went_goes_on(tmp_path, caplog):
    root = tmp_path / "root"
    workspace = api.Workspace(root, reveal=lambda folder: None, clock=FakeClock())
    first = workspace.create("A µ")
    imported(first, 1)
    for path in (first.folder / storage.IMAGES_DIR).iterdir():
        path.unlink()
    (first.folder / storage.PROJECT_FILE).unlink()
    edit(first)
    assert isinstance(first.save_error, storage.ProjectRemovedError)
    caplog.set_level(logging.WARNING, logger="proteia.web.api")
    workspace.create("B β")
    assert not (first.folder / storage.PROJECT_FILE).exists()  # never written again
    said = [record.getMessage() for record in caplog.records if record.name == "proteia.web.api"]
    assert said == [f"closed 'A µ' with changes not saved: {GONE}"]


def test_the_folder_put_back_takes_the_change_at_the_next_save(tmp_path):
    s = opened(tmp_path)
    aside = tmp_path / "aside"
    shutil.copytree(s.folder, aside)
    shutil.rmtree(s.folder)
    edit(s, "a")
    assert str(s.save_error) == REMOVED
    shutil.copytree(aside, s.folder)
    edit(s, "b")
    assert (s.save_error, s.dirty, copies(s.folder)) == (None, False, [])
    assert storage.load_project(s.folder) == s.project


def test_an_import_into_a_removed_folder_stores_nothing(tmp_path):
    s = opened(tmp_path)
    shutil.rmtree(s.folder)
    with pytest.raises(OperationError) as refused:
        imported(s, 2)
    assert refused.value.code is ErrorCode.PROJECT_REMOVED
    assert str(refused.value) == (
        f"{FOLDER!r} is no longer in the projects folder as it was (its folder, or its"
        " project.json and an image file, were removed or moved outside Proteia), so"
        " nothing was imported; put the folder back as it was, or open another project"
    )
    assert not s.folder.exists()


# --- Files that cannot be read, changes during the save ---


def unread(name: str, why: str) -> str:
    return (
        f"Proteia could not read the project.json of {name!r} to check whether it was changed"
        f" outside Proteia ({why}), so it did not save over it; this change is kept here:"
        " close the program that holds the file (or put project.json back as a file), then"
        " make any change"
    )


def test_a_folder_in_place_of_project_json_fails_the_save_and_writes_nothing(tmp_path, caplog):
    s = opened(tmp_path)
    project_file = s.folder / storage.PROJECT_FILE
    kept = project_file.read_bytes()
    project_file.unlink()
    project_file.mkdir()
    caplog.set_level(logging.WARNING, logger="proteia.core.session")
    edit(s, "a")
    # Proteia's own words, no path; the read's own error, with its path, is logged.
    assert isinstance(s.save_error, storage.ProjectFileUnreadError)
    assert str(s.save_error) == unread(FOLDER, s.save_error.__cause__.strerror)
    cause = s.save_error.__cause__
    assert isinstance(cause, OSError) and cause.filename == str(project_file)
    assert f"in {FOLDER!r}: {cause}" in caplog.messages
    assert copies(s.folder) == [] and project_file.is_dir()
    project_file.rmdir()
    project_file.write_bytes(kept)
    edit(s, "b")
    assert (s.save_error, copies(s.folder)) == (None, [])


def _reads(monkeypatch, failures: int) -> list[str]:
    """Count the reads of project.json; the first ``failures`` raise a sharing
    violation, as while another program holds it."""
    real = Path.read_bytes
    seen: list[str] = []

    def read(path: Path) -> bytes:
        if path.name == storage.PROJECT_FILE:
            seen.append(path.name)
            if len(seen) <= failures:
                raise PermissionError(13, "held by another process", str(path))
        return real(path)

    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    monkeypatch.setattr(Path, "read_bytes", read)
    return seen


def test_a_held_project_json_is_read_again_while_windows_reports_a_sharing_violation(
    tmp_path, monkeypatch
):
    s = opened(tmp_path)
    seen = _reads(monkeypatch, 2)
    slept: list[float] = []
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0.05)
    monkeypatch.setattr(storage.time, "sleep", slept.append)
    edit(s)
    assert s.save_error is None
    assert len(seen) == 4  # two refused, one read, one check before the replace
    assert slept == [0.05, 0.1]  # the replace's doubling delay


def test_a_project_json_held_through_every_retry_fails_the_save(tmp_path, monkeypatch):
    s = opened(tmp_path)
    before = file_bytes(s)
    with monkeypatch.context() as patch:  # only this test's patches go after it
        seen = _reads(patch, storage.REPLACE_ATTEMPTS)
        edit(s)
    assert len(seen) == storage.REPLACE_ATTEMPTS
    assert str(s.save_error) == unread(FOLDER, "held by another process")
    assert str(tmp_path) not in str(s.save_error)
    assert file_bytes(s) == before and copies(s.folder) == []
    # The check right before the replace reads it too, with the same words.
    with monkeypatch.context() as patch:
        real = Path.read_bytes
        calls: list[int] = []

        def held_at_the_check(path: Path) -> bytes:
            if path.name == storage.PROJECT_FILE:
                calls.append(1)
                if len(calls) > 1:  # the read before writing goes through
                    raise PermissionError(13, "held by another process", str(path))
            return real(path)

        patch.setattr(storage, "REPLACE_DELAY", 0)
        patch.setattr(Path, "read_bytes", held_at_the_check)
        edit(s, "b")
    assert str(s.save_error) == unread(FOLDER, "held by another process")
    assert file_bytes(s) == before


def _changing_while_written(monkeypatch, newer: bytes) -> None:
    """Make another program write ``newer`` as project.json once, while the next
    save writes its temp file (after its check of the file, before its replace)."""
    real = storage._temp_file
    done: list[bool] = []

    def temp_file(directory: Path, name: str, suffix: str):
        made = real(directory, name, suffix)
        if name == storage.PROJECT_FILE and not done:
            done.append(True)
            (directory / storage.PROJECT_FILE).write_bytes(newer)
        return made

    monkeypatch.setattr(storage, "_temp_file", temp_file)


AGAIN = (
    f"{FOLDER!r} was changed outside Proteia again while Proteia saved it, so Proteia did"
    " not save over it; it saves this change, keeping that version too, at your next"
    " change or when you quit"
)


def test_a_change_while_the_save_writes_is_kept(tmp_path, monkeypatch):
    s = opened(tmp_path)
    project_file = s.folder / storage.PROJECT_FILE
    project_file.write_bytes(b'{"first": "outside"}')
    _changing_while_written(monkeypatch, b'{"second": "outside"}')
    edit(s, "a")
    assert isinstance(s.save_error, storage.ChangedAgainError) and str(s.save_error) == AGAIN
    assert project_file.read_bytes() == b'{"second": "outside"}'
    assert (s.folder / COPY).read_bytes() == b'{"first": "outside"}'
    edit(s, "b")
    second = f"project.conflict {MINUTE} (2).json"
    assert (s.folder / second).read_bytes() == b'{"second": "outside"}'
    assert file_bytes(s) == storage.project_to_json(s.project) and s.save_error is None


def test_a_file_removed_while_the_save_writes_is_written_at_the_next_save(tmp_path, monkeypatch):
    s = opened(tmp_path)
    project_file = s.folder / storage.PROJECT_FILE
    real = storage._temp_file
    done: list[bool] = []

    def temp_file(directory: Path, name: str, suffix: str):
        made = real(directory, name, suffix)
        if name == storage.PROJECT_FILE and not done:
            done.append(True)
            project_file.unlink()  # after the check, before the replace
        return made

    monkeypatch.setattr(storage, "_temp_file", temp_file)
    edit(s, "a")
    assert str(s.save_error) == AGAIN and not project_file.exists()
    edit(s, "b")
    assert file_bytes(s) == storage.project_to_json(s.project)
    assert s.outside_change["message"].startswith(f"The project.json of {FOLDER!r} was removed")


def test_a_version_file_is_read_once_while_it_stays_the_same(tmp_path, monkeypatch):
    s = opened(tmp_path)
    below, _ = _planted(s)
    (s.folder / "project mine.json").write_bytes(f'{{"x": "{below.name}"}}'.encode())
    read: list[str] = []
    real = storage.named_images

    def named_images(path: Path) -> frozenset[str]:
        read.append(path.name)
        return real(path)

    monkeypatch.setattr(storage, "named_images", named_images)
    edit(s, "a")
    edit(s, "b")  # the kept file is still a candidate: the versions are asked again
    assert below.is_file() and read == ["project mine.json"]


def test_a_change_while_a_clean_save_writes_is_kept(tmp_path, monkeypatch):
    s = opened(tmp_path)
    _changing_while_written(monkeypatch, b'{"while": "saving"}')
    edit(s, "a")
    assert str(s.save_error) == AGAIN and copies(s.folder) == []
    edit(s, "b")
    assert (s.folder / COPY).read_bytes() == b'{"while": "saving"}'


def test_a_change_while_a_replace_is_retried_is_kept(tmp_path, monkeypatch):
    s = opened(tmp_path)
    project_file = s.folder / storage.PROJECT_FILE
    real = storage.os.replace
    tries: list[str] = []

    def replace(src, dst, **kwargs):
        if Path(dst).name == storage.PROJECT_FILE and not tries:
            tries.append("held")
            project_file.write_bytes(b'{"while": "held"}')
            raise PermissionError(13, "held by another process", str(dst))
        return real(src, dst, **kwargs)

    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    monkeypatch.setattr(storage.os, "replace", replace)
    edit(s, "a")
    assert str(s.save_error) == AGAIN and project_file.read_bytes() == b'{"while": "held"}'


def test_a_copy_that_cannot_be_written_refuses_the_save(tmp_path, monkeypatch):
    s = opened(tmp_path)
    project_file = s.folder / storage.PROJECT_FILE
    project_file.write_bytes(b'{"changed": "outside"}')

    def full(path, data, mode=None):
        raise OSError(errno.ENOSPC, "No space left on device", str(path))

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_write_new", full)
        edit(s, "a")
    assert isinstance(s.save_error, storage.CopyNotKeptError) and s.dirty
    assert str(s.save_error) == (
        f"{FOLDER!r} was changed outside Proteia after Proteia last opened or saved it, and"
        " Proteia could not keep that version in a copy (No space left on device), so it did"
        " not save over it; this change is kept here and saved, with the copy, at your next"
        " change once the folder can take it"
    )
    assert project_file.read_bytes() == b'{"changed": "outside"}' and copies(s.folder) == []
    edit(s, "b")
    assert (s.folder / COPY).read_bytes() == b'{"changed": "outside"}' and not s.dirty


def test_saves_in_a_row_and_from_two_threads_go_over_the_last(tmp_path):
    s = opened(tmp_path)
    for label in ("a", "b", "c"):
        edit(s, label)

    def edits(prefix: str) -> None:
        for n in range(5):
            edit(s, f"{prefix}{n}")

    threads = [threading.Thread(target=edits, args=(p,)) for p in ("x", "y")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert copies(s.folder) == [] and s.save_error is None
    assert file_bytes(s) == storage.project_to_json(s.project)


def test_an_open_remembers_the_bytes_it_parsed(tmp_path, monkeypatch):
    s = opened(tmp_path)
    real = storage.read_project
    folder = s.folder

    def read_project(*args, **kwargs):
        read = real(*args, **kwargs)
        (folder / storage.PROJECT_FILE).write_bytes(b'{"after": "the read"}')
        return read

    with monkeypatch.context() as patch:
        patch.setattr(storage, "read_project", read_project)
        reopened = ops.open_project(folder, clock=FakeClock())
    edit(reopened)
    assert (folder / COPY).read_bytes() == b'{"after": "the read"}'


def test_a_reload_remembers_the_bytes_it_parsed(tmp_path, monkeypatch):
    s = opened(tmp_path)
    other = elsewhere(s)
    ops.set_lanes(other, [ops.LaneInput("drug")])
    put(s.folder, other.project)
    real = storage.read_project

    def read_project(*args, **kwargs):
        read = real(*args, **kwargs)
        (s.folder / storage.PROJECT_FILE).write_bytes(b'{"after": "the reload"}')
        return read

    with monkeypatch.context() as patch:
        patch.setattr(storage, "read_project", read_project)
        assert s.reload()
    edit(s)
    assert (s.folder / COPY).read_bytes() == b'{"after": "the reload"}'


def test_a_change_after_the_replace_keeps_its_images_from_the_cleanup(tmp_path, monkeypatch):
    s = opened(tmp_path)
    left = s.folder / storage.IMAGES_DIR / "img-50.tif"
    left.write_bytes(tiff(50))  # what the other version imported
    real = storage._fsync_dir
    done: list[bool] = []

    def fsync_dir(directory: Path) -> None:
        real(directory)
        if directory == s.folder and not done:  # right after the replace
            done.append(True)
            (s.folder / storage.PROJECT_FILE).write_bytes(b'{"file": "img-50.tif"}')

    monkeypatch.setattr(storage, "_fsync_dir", fsync_dir)
    edit(s, "a")
    assert left.is_file()  # the cleanup read the file again
    edit(s, "b")
    assert (s.folder / COPY).read_bytes() == b'{"file": "img-50.tif"}' and left.is_file()


# --- Opening again ---


def test_reopening_after_a_conflict_reads_nothing_new(tmp_path):
    root = tmp_path / "root"
    workspace = api.Workspace(root, reveal=lambda folder: None, clock=FakeClock())
    s = workspace.create("A µ")
    (s.folder / storage.PROJECT_FILE).write_bytes(b'{"changed": "outside"}')
    edit(s)
    said = s.outside_change
    _, open_id = workspace.opened()
    assert workspace.open("A µ") is s
    assert workspace.opened() == ("A µ", open_id) and s.outside_change == said


def _found(name: str, listed: str, held: str) -> str:
    if held == "1 version":
        kept = "Proteia never deletes it; once you delete it, this notice goes."
        step = go_on(name, "it", "the copy")
    else:
        kept = (
            "Proteia never deletes them; once you delete those you do not need, this notice"
            " goes with the last."
        )
        step = go_on(name, "one", "that copy")
    return (
        f"The folder of {name!r} holds {held} of the project that Proteia set aside when it was"
        f" changed outside Proteia: {listed}. {kept} {step}"
    )


def test_opening_lists_the_copies_in_the_folder(tmp_path, caplog):
    s = opened(tmp_path, images=0)
    folder = s.folder
    assert ops.open_project(folder, clock=FakeClock()).outside_change is None
    (folder / "project.conflict 2026-10-02 0915.json").write_bytes(b"{}")
    single = ops.open_project(folder, clock=FakeClock())
    assert single.outside_change == {
        "event": 0,
        "message": _found(FOLDER, "'project.conflict 2026-10-02 0915.json'", "1 version"),
        "copies": ["project.conflict 2026-10-02 0915.json"],
    }
    names = [
        "project.conflict 2026-10-01 0800.json",
        "project.conflict 2026-10-03 1200.json",
        "project.conflict 2026-10-03 1200 (2).json",
    ]
    for name in names:
        (folder / name).write_bytes(b"{}")
    (folder / "project.conflict 2026-10-04 1000.json").mkdir()  # a folder is no copy
    caplog.set_level(logging.INFO, logger="proteia.core.session")
    many = ops.open_project(folder, clock=FakeClock())
    newest = [  # by time, then by number
        "project.conflict 2026-10-03 1200 (2).json",
        "project.conflict 2026-10-03 1200.json",
        "project.conflict 2026-10-02 0915.json",
        "project.conflict 2026-10-01 0800.json",
    ]
    listed = ", ".join(repr(name) for name in newest[:3]) + " and 1 more"
    assert many.outside_change == {
        "event": 0,
        "message": _found(FOLDER, listed, "4 versions"),
        "copies": newest,
    }
    assert (
        f"in {FOLDER!r}: 4 versions Proteia set aside when it was changed outside Proteia: "
        + ", ".join(repr(name) for name in newest)
    ) in caplog.messages
    (folder / newest[3]).unlink()  # three: all named, none "more"
    three = ops.open_project(folder, clock=FakeClock())
    assert three.outside_change["message"] == _found(
        FOLDER, f"{newest[0]!r}, {newest[1]!r} and {newest[2]!r}", "3 versions"
    )
    for name in newest[2:3]:
        (folder / name).unlink()
    two = ops.open_project(folder, clock=FakeClock())
    assert two.outside_change["message"] == _found(
        FOLDER, f"{newest[0]!r} and {newest[1]!r}", "2 versions"
    )


def test_a_reload_lists_the_copies_and_ends_what_the_opening_said(tmp_path):
    s = opened(tmp_path)
    (s.folder / storage.PROJECT_FILE).write_bytes(b'{"changed": "outside"}')
    edit(s)
    assert s.outside_change["message"].startswith(set_aside(FOLDER))
    other = elsewhere(s)
    ops.set_lanes(other, [ops.LaneInput("drug")])
    put(s.folder, other.project)
    assert s.reload()
    assert s.outside_change == {
        "event": 0,
        "message": _found(FOLDER, repr(COPY), "1 version"),
        "copies": [COPY],
    }


def test_the_copy_renamed_to_project_json_opens_with_its_images(tmp_path):
    # The next step the notice gives, followed literally.
    s = opened(tmp_path)
    other = elsewhere(s)
    theirs_image = imported(other, 2, "theirs α.tif")
    ops.set_lanes(other, [ops.LaneInput("drug")])
    theirs = put(s.folder, other.project)
    ours_image = imported(s, 3, "ours β.tif")  # its save keeps the copy
    assert theirs_image != ours_image and copies(s.folder) == [COPY]
    s.close()  # "open another project"
    folder = s.folder
    (folder / storage.PROJECT_FILE).rename(folder / "project mine.json")
    (folder / COPY).rename(folder / storage.PROJECT_FILE)
    again = ops.open_project(folder, clock=FakeClock())
    assert storage.project_to_json(again.project) == theirs
    edit(again, "drug µ")
    again.close()
    files = images(folder)
    for image_id in ("img-1", theirs_image, ours_image):
        assert f"{image_id}.tif" in files


# --- Cleanup and the id an import takes ---


def test_images_only_a_conflict_copy_names_survive_every_cleanup(tmp_path):
    s = opened(tmp_path)
    other = elsewhere(s)
    theirs_image = imported(other, 2, "theirs α.tif")
    put(s.folder, other.project)
    theirs_file = s.folder / storage.IMAGES_DIR / f"{theirs_image}.tif"
    edit(s, "a")  # saved, then cleaned up
    assert theirs_file.is_file() and copies(s.folder) == [COPY]
    imported(s, 3)  # cleans up before it stores
    assert theirs_file.is_file()
    s.close()
    assert theirs_file.is_file()
    later = ops.open_project(s.folder, clock=FakeClock())
    edit(later, "b")
    later.close()
    assert theirs_file.read_bytes() == tiff(2) and copies(s.folder) == [COPY]


def test_an_image_the_outside_version_added_at_next_id_survives(tmp_path):
    s = opened(tmp_path)
    next_id = s.project.next_id
    other = elsewhere(s)
    theirs_image = imported(other, 2)
    assert theirs_image == f"img-{next_id}"
    put(s.folder, other.project)
    edit(s)
    assert (s.folder / storage.IMAGES_DIR / f"{theirs_image}.tif").read_bytes() == tiff(2)


def test_an_import_never_takes_an_id_whose_file_a_version_keeps(tmp_path):
    s = opened(tmp_path)
    next_id = s.project.next_id
    other = elsewhere(s)
    theirs_image = imported(other, 2)
    put(s.folder, other.project)
    ours = imported(s, 3)
    assert (theirs_image, ours) == (f"img-{next_id}", f"img-{next_id + 1}")
    assert s.project.next_id > next_id + 1
    assert (s.folder / storage.IMAGES_DIR / f"{theirs_image}.tif").read_bytes() == tiff(2)
    assert (s.folder / storage.IMAGES_DIR / f"{ours}.tif").read_bytes() == tiff(3)
    assert storage.load_project(s.folder) == s.project


def test_an_import_skips_ids_a_version_names_before_their_files_arrive(tmp_path):
    s = opened(tmp_path)
    next_id = s.project.next_id
    (s.folder / "project.conflict 2026-10-01 0800.json").write_bytes(
        f'{{"file": "IMG-{next_id}.TIF", "other": "img-{next_id + 1}.png"}}'.encode()
    )
    assert imported(s, 3) == f"img-{next_id + 2}"


def test_an_import_skips_ids_whose_files_are_there_in_any_case_or_suffix(tmp_path):
    s = opened(tmp_path)
    next_id = s.project.next_id
    (s.folder / "project.conflict 2026-10-01 0800.json").write_bytes(
        f'{{"a": "img-{next_id}.png", "b": "img-{next_id + 1}.tif"}}'.encode()
    )
    folder = s.folder / storage.IMAGES_DIR
    (folder / f"img-{next_id}.png").write_bytes(b"kept")
    (folder / f"IMG-{next_id + 1}.TIF").write_bytes(b"kept")
    assert imported(s, 3) == f"img-{next_id + 2}"


def test_a_held_orphan_at_next_id_no_longer_refuses_an_import(tmp_path, monkeypatch):
    s = opened(tmp_path)
    next_id = s.project.next_id
    held = s.folder / storage.IMAGES_DIR / f"img-{next_id}.tif"
    held.write_bytes(b"left by an import that was never saved")
    real = Path.unlink

    def unlink(path: Path, missing_ok: bool = False) -> None:
        if path == held:
            raise PermissionError(13, "held by another process", str(path))
        real(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert imported(s, 3) == f"img-{next_id + 1}"
    assert held.read_bytes() == b"left by an import that was never saved"


def _planted(s: ProjectSession) -> tuple[Path, Path]:
    """Two image files no project references: one below next_id, one above."""
    folder = s.folder / storage.IMAGES_DIR
    below, above = folder / "img-3.tif", folder / f"img-{s.project.next_id + 4}.tif"
    for path in (below, above):
        path.write_bytes(b"an image no project.json references")
    return below, above


def test_a_version_that_does_not_parse_keeps_the_images_its_text_names(tmp_path):
    s = opened(tmp_path)
    below, above = _planted(s)
    version = s.folder / "project.conflict 2026-10-01 0800.json"
    version.write_bytes(f'{{"cut": "{below.name}", {above.name.upper()}'.encode())
    edit(s, "a")
    assert below.is_file() and above.is_file()
    version.unlink()
    edit(s, "b")
    assert not below.exists() and not above.exists()


def test_while_a_version_cannot_be_read_no_image_file_is_deleted(tmp_path, monkeypatch):
    s = opened(tmp_path)
    below, above = _planted(s)
    temp = s.folder / storage.IMAGES_DIR / ".img-9.tif.x.part"
    temp.write_bytes(b"left by a crash")
    (s.folder / "project.conflict 2026-10-01 0800.json").write_bytes(b"{}")

    def locked(path: Path) -> frozenset[str]:
        raise PermissionError(13, "held by another process", str(path))

    with monkeypatch.context() as patch:
        patch.setattr(storage, "named_images", locked)
        edit(s, "a")
        assert below.is_file() and above.is_file() and not temp.exists()
    edit(s, "b")
    assert not below.exists() and not above.exists()


def test_a_missing_project_json_keeps_every_image_file_at_close(tmp_path):
    s = opened(tmp_path)
    below, above = _planted(s)
    (s.folder / storage.PROJECT_FILE).unlink()
    s.close()
    assert below.is_file() and above.is_file()


@pytest.mark.parametrize("doubt", ["held", "not in its encoding", "not listable"])
def test_doubt_of_project_json_or_its_folder_keeps_every_image_file(tmp_path, monkeypatch, doubt):
    s = opened(tmp_path)
    below, above = _planted(s)
    temp = s.folder / storage.IMAGES_DIR / ".img-9.tif.x.part"
    temp.write_bytes(b"left by a crash")
    if doubt == "not in its encoding":  # changed outside, not yet kept in a copy
        (s.folder / storage.PROJECT_FILE).write_bytes(b"\xff\xfe\x00\xd8")
    folder, real = s.folder, Path.iterdir

    def refused(path: Path, *args: Any) -> Any:
        raise PermissionError(13, "held by another process", str(path))

    def iterdir(path: Path) -> Any:
        return refused(path) if path == folder else real(path)

    with monkeypatch.context() as patch:
        if doubt == "held":
            patch.setattr(storage, "read_project_file", refused)
        elif doubt == "not listable":
            patch.setattr(Path, "iterdir", iterdir)
        s.close()
    assert below.is_file() and above.is_file() and not temp.exists()


def test_a_folder_named_like_a_version_is_none(tmp_path):
    s = opened(tmp_path)
    below, above = _planted(s)
    (s.folder / "old.json").mkdir()
    edit(s)  # saved, then cleaned up
    assert not below.exists() and not above.exists()


@pytest.mark.parametrize(
    ("name", "kept"),
    [
        ("project mine.json", True),
        ("project-LAB-PC.json", True),
        ("Project (Lab PC's conflicted copy 2026-11-01).JSON", True),
        ("notes.json", True),
        ("old/project.json", False),  # not directly in the folder
        ("project.json.old", False),  # no .json file
    ],
)
def test_any_json_beside_project_json_keeps_the_images_it_names(tmp_path, name, kept):
    s = opened(tmp_path)
    below, above = _planted(s)
    version = s.folder / name
    version.parent.mkdir(exist_ok=True)
    version.write_bytes(f'{{"images": ["{below.name}", "{above.name}"]}}'.encode())
    edit(s)
    assert (below.is_file(), above.is_file()) == (kept, kept)


def test_a_backup_keeps_only_the_images_below_next_id(tmp_path):
    s = opened(tmp_path)
    below, above = _planted(s)
    (s.folder / "project.schema9.json").write_bytes(f"{below.name} {above.name}".encode())
    edit(s)
    assert below.is_file() and not above.exists()


def test_a_version_file_is_read_again_once_it_changed(tmp_path):
    s = opened(tmp_path)
    below, above = _planted(s)
    version = s.folder / "project mine.json"
    first = f'{{"a": "{below.name}"}}'.encode()
    version.write_bytes(first)
    edit(s, "a")
    assert below.is_file() and not above.exists()
    stat = version.stat()
    # Replaced by another file of the same size and time (a sync keeps the remote's).
    replaced = s.folder / "replaced.tmp"
    replaced.write_bytes(f'{{"b": "{below.name.replace("3", "4")}"}}'.encode())
    assert replaced.stat().st_size == len(first)
    os.utime(replaced, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    os.replace(replaced, version)
    edit(s, "b")
    assert not below.exists()  # read again: it names img-4 now


def test_named_images_finds_a_name_across_two_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_SCAN_CHUNK", 16)
    path = tmp_path / "version.json"
    path.write_bytes(b"x" * 13 + b'"img-27.tiff" and IMG-3.PNG, ' + b"y" * 40)
    assert storage.named_images(path) == frozenset({"img-27.tiff", "IMG-3.PNG"})


def test_closing_after_a_change_outside_with_nothing_unsaved_deletes_no_image_it_names(tmp_path):
    s = opened(tmp_path)
    other = elsewhere(s)
    theirs_image = imported(other, 2)
    put(s.folder, other.project)
    s.close()
    assert (s.folder / storage.IMAGES_DIR / f"{theirs_image}.tif").read_bytes() == tiff(2)


def test_an_import_after_a_change_outside_keeps_its_images_and_sets_it_aside(tmp_path):
    s = opened(tmp_path)
    other = elsewhere(s)
    theirs_image = imported(other, 2)
    theirs = put(s.folder, other.project)
    ours = imported(s, 3)
    assert ours != theirs_image
    assert (s.folder / COPY).read_bytes() == theirs
    assert (s.folder / storage.IMAGES_DIR / f"{theirs_image}.tif").read_bytes() == tiff(2)


def test_a_project_json_made_while_a_project_is_created_is_left(tmp_path, monkeypatch):
    real = storage._temp_file

    def temp_file(directory: Path, name: str, suffix: str):
        made = real(directory, name, suffix)
        if name == storage.PROJECT_FILE:
            (directory / storage.PROJECT_FILE).write_bytes(b"another program's")
        return made

    monkeypatch.setattr(storage, "_temp_file", temp_file)
    with pytest.raises(OperationError) as refused:
        ops.new_project(tmp_path / FOLDER, clock=FakeClock())
    assert refused.value.code is ErrorCode.FOLDER_NOT_EMPTY
    assert (tmp_path / FOLDER / storage.PROJECT_FILE).read_bytes() == b"another program's"
    root = tmp_path / "root"
    with pytest.raises(projects.ProjectExistsError, match="a project named 'Blot' already exists"):
        projects.create_project(root, "Blot", clock=FakeClock())
    assert (root / "Blot" / storage.PROJECT_FILE).read_bytes() == b"another program's"


def test_a_save_never_makes_its_folder_again(tmp_path):
    gone = tmp_path / "gone µ"
    with pytest.raises(FileNotFoundError):
        storage.write_project_file(gone, b"{}", expect=None)
    assert not gone.exists()
    storage.write_project_file(gone, b"{}", expect=None, create=True)  # a new project's
    assert (gone / storage.PROJECT_FILE).read_bytes() == b"{}"


def test_the_storage_names_of_versions_and_copies():
    moment = datetime(2026, 11, 1, 7, 32, 59, tzinfo=UTC)
    local = moment.astimezone().strftime("%Y-%m-%d %H%M")
    assert storage.conflict_name(moment) == f"project.conflict {local}.json"
    assert storage.conflict_name(moment, 3) == f"project.conflict {local} (3).json"
    assert storage.conflict_name(moment, short=True) == "project.conflict.json"
    assert storage.conflict_name(moment, 2, short=True) == "project.conflict (2).json"
    for name, version, copy in [
        ("project.json", False, False),
        ("PROJECT.JSON", False, False),
        ("project.schema1.json", False, False),
        ("project.conflict 2026-11-01 1532.json", True, True),
        ("Project.Conflict.JSON", True, True),
        ("project.conflict.txt", False, False),
        ("notes.json", True, False),
        ("project.json.old", False, False),
    ]:
        assert (storage.is_version(name), storage.is_conflict_copy(name)) == (version, copy), name
    assert storage.conflict_copies(Path("no such folder µ")) == ()
    # Newest first: by the time in the name, then the number (none is the first).
    assert storage._copy_order("Project.Conflict 2026-10-03 1200.JSON") == (
        "project.conflict 2026-10-03 1200",
        1,
    )
    assert storage._copy_order("project.conflict 2026-10-03 1200 (12).json") == (
        "project.conflict 2026-10-03 1200",
        12,
    )


# --- Notices, and versions in other encodings ---


def test_a_copy_is_said_as_it_is_made_even_when_the_save_then_fails(tmp_path, monkeypatch):
    s = opened(tmp_path)
    (s.folder / storage.PROJECT_FILE).write_bytes(b'{"first": "outside"}')
    with monkeypatch.context() as patch:
        _changing_while_written(patch, b'{"second": "outside"}')
        edit(s, "a")
    assert isinstance(s.save_error, storage.ChangedAgainError)
    first = s.outside_change
    assert first["event"] == 1 and first["copies"] == [COPY]
    assert first["message"].startswith(set_aside(FOLDER))  # the page hears of it now
    edit(s, "b")
    second = f"project.conflict {MINUTE} (2).json"
    assert s.outside_change["event"] == 2 and s.outside_change["copies"] == [second, COPY]
    assert f"Proteia kept that version as {second!r}" in s.outside_change["message"]
    edit(s, "c")  # over its own file: nothing new to say
    assert s.outside_change["event"] == 2


def test_two_removals_in_one_opening_are_two_notices(tmp_path):
    s = opened(tmp_path)
    (s.folder / storage.PROJECT_FILE).unlink()
    edit(s, "a")
    first = s.outside_change
    (s.folder / storage.PROJECT_FILE).unlink()
    edit(s, "b")
    # The same words, another notice: a page that dismissed the first shows it.
    assert s.outside_change["message"] == first["message"]
    assert (first["event"], s.outside_change["event"]) == (1, 2)


def test_an_import_uses_the_names_of_the_versions_it_could_read(tmp_path, monkeypatch):
    s = opened(tmp_path)
    other = elsewhere(s)
    theirs = imported(other, 5)
    put(s.folder, other.project)
    (s.folder / storage.IMAGES_DIR / f"{theirs}.tif").unlink()  # not synced yet
    edit(s, "a")  # sets their version aside
    (s.folder / "project-LAB-PC.json").write_bytes(b"{}")  # a sync client's file, held
    real = storage.named_images

    def held(path: Path) -> frozenset[str]:
        if path.name == "project-LAB-PC.json":
            raise PermissionError(13, "held by another process", str(path))
        return real(path)

    monkeypatch.setattr(storage, "named_images", held)
    assert imported(s, 6) != theirs  # the copy it could read names that id


def _utf(data: bytes, encoding: str) -> bytes:
    return data.decode("utf-8").encode(encoding)


@pytest.mark.parametrize(
    "encoding", ["utf-16", "utf-16-le-bom", "utf-16-be-bom", "utf-32", "utf-8-sig"]
)
def test_an_outside_version_in_another_encoding_keeps_its_images(tmp_path, encoding):
    s = opened(tmp_path)
    other = elsewhere(s)
    theirs = imported(other, 5)
    data = storage.project_to_json(other.project)
    if encoding == "utf-16-le-bom":
        encoded = codecs.BOM_UTF16_LE + _utf(data, "utf-16-le")
    elif encoding == "utf-16-be-bom":
        encoded = codecs.BOM_UTF16_BE + _utf(data, "utf-16-be")
    else:
        encoded = _utf(data, encoding)
    (s.folder / storage.PROJECT_FILE).write_bytes(encoded)
    theirs_file = s.folder / storage.IMAGES_DIR / f"{theirs}.tif"
    edit(s, "a")  # cleaned up after the save, from the copy's names
    assert (s.folder / COPY).read_bytes() == encoded
    assert theirs_file.read_bytes() == tiff(5)
    imported(s, 6)  # and before an import
    s.close()
    assert theirs_file.read_bytes() == tiff(5)


def test_a_version_not_in_its_encoding_keeps_every_image_file(tmp_path):
    s = opened(tmp_path)
    other = elsewhere(s)
    theirs = imported(other, 5)  # an orphan here, which no version names below
    theirs_file = s.folder / storage.IMAGES_DIR / f"{theirs}.tif"
    version = s.folder / "project-LAB-PC.json"
    version.write_bytes(b"\xff\xfe" + b"\x00\xd8")  # UTF-16 with a lone surrogate
    edit(s, "a")
    assert theirs_file.is_file()  # what it names is unknown: nothing deleted
    version.write_bytes(b"\xc3\x28 not UTF-8")
    edit(s, "b")
    assert theirs_file.is_file()
    version.unlink()
    edit(s, "c")
    assert not theirs_file.exists()


def test_version_names_reads_the_text_its_byte_order_mark_says():
    text = '{"a": "img-3.tif", "b": "IMG-12.TIFF"}'
    found = frozenset({"img-3.tif", "IMG-12.TIFF"})
    for encoded in (
        text.encode("utf-8"),
        codecs.BOM_UTF8 + text.encode("utf-8"),
        text.encode("utf-16"),
        codecs.BOM_UTF16_BE + text.encode("utf-16-be"),
        text.encode("utf-32"),
        codecs.BOM_UTF32_BE + text.encode("utf-32-be"),
    ):
        assert storage.version_names(encoded) == found, encoded[:4]
    with pytest.raises(UnicodeDecodeError):
        storage.version_names(b"\xc3\x28")


def test_named_images_reads_a_utf16_file_across_its_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_SCAN_CHUNK", 7)  # odd: cuts code units in two
    path = tmp_path / "version.json"
    path.write_bytes(("x" * 11 + '"img-27.tiff" and IMG-3.PNG, ' + "y" * 30).encode("utf-16"))
    assert storage.named_images(path) == frozenset({"img-27.tiff", "IMG-3.PNG"})
    path.write_bytes(b"\xff\xfe\x00\xd8")
    with pytest.raises(UnicodeDecodeError):
        storage.named_images(path)


def test_a_version_that_cannot_be_looked_at_keeps_every_image_file(tmp_path, monkeypatch):
    # A version whose whole path is past Windows' limit cannot be stat'ed: doubt.
    s = opened(tmp_path)
    other = elsewhere(s)
    theirs = imported(other, 5)
    theirs_file = s.folder / storage.IMAGES_DIR / f"{theirs}.tif"
    version = s.folder / "project-LAB-PC.json"
    version.write_bytes(b"{}")
    real = Path.stat

    def stat(path: Path, *args, **kwargs):
        if path.name == version.name:
            raise FileNotFoundError(2, "The filename or extension is too long", str(path))
        return real(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", stat)
        edit(s, "a")
    assert theirs_file.is_file()
    edit(s, "b")  # looked at now: it names nothing
    assert not theirs_file.exists()


def test_an_import_whose_file_was_taken_meanwhile_names_no_id(tmp_path, monkeypatch):
    s = opened(tmp_path)

    def taken(*args, **kwargs):
        raise FileExistsError(17, "File exists")

    monkeypatch.setattr(storage, "store_image", taken)
    with pytest.raises(OperationError) as refused:
        imported(s, 2)
    assert refused.value.code is ErrorCode.LEFTOVER_FILE
    assert str(refused.value) == (
        "another program put a file in the project's images folder where Proteia was"
        " storing this image; import it again"
    )
    assert "img-" not in str(refused.value)


def test_the_steps_back_to_a_version_work_with_one_project_and_hidden_extensions(tmp_path):
    s = opened(tmp_path)
    _hand_edited(s.folder)  # a project, so the notice says how to go on from it
    edit(s)
    (s.folder / "project.conflict 2026-10-01 0800.json").write_bytes(b"{}")
    one = ops.open_project(s.folder, clock=FakeClock()).outside_change["message"]
    (s.folder / COPY).unlink()  # one copy left
    two_less = ops.open_project(s.folder, clock=FakeClock()).outside_change["message"]
    for said in (s.outside_change["message"], one, two_less):
        assert "open another project" not in said and "quit Proteia" in said
        assert "type project mine and project" in said
    assert "Proteia never deletes them; once you delete those you do not need" in one
    assert "Proteia never deletes it; once you delete it, this notice goes." in two_less
    assert "never deletes them" not in two_less


# --- The page and the web routes ---


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "projects"
    workspace = api.Workspace(root, reveal=lambda folder: None, clock=FakeClock())
    instance = launch.start(folder=tmp_path / "state", opener=lambda url: True, workspace=workspace)
    thread = threading.Thread(target=instance.serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.server.started:
        assert thread.is_alive() and time.monotonic() < deadline
        time.sleep(0.01)
    yield Client(instance.port, instance.token, root, [], workspace)
    instance.stop()
    thread.join(10)


def _hand_edited(folder: Path) -> bytes:
    """Change ``folder``'s project.json by hand, adding no log entry."""
    path = folder / storage.PROJECT_FILE
    doc = json.loads(path.read_bytes())
    doc["next_id"] += 7
    data = storage.document_bytes(doc)
    path.write_bytes(data)
    return data


def test_the_state_carries_the_outside_change_for_the_opening(client, tmp_path):
    created = client.ok("POST", "/api/projects", {"name": "Blot µ"})
    assert created["project"]["outside_change"] is None
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    folder = client.root / "Blot µ"
    _hand_edited(folder)
    answer = client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle α"}]})
    expected = {
        "event": 1,
        "message": (
            f"{set_aside('Blot µ')} That version's log has no change this one does not, so it"
            " is older or was edited by hand; this one has 1 change it does not."
            f" {go_on('Blot µ')}"
        ),
        "copies": [COPY],
    }
    assert answer["project"]["outside_change"] == expected
    later = client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle β"}]})
    assert later["project"]["outside_change"] == expected
    assert client.ok("GET", "/api/project")["project"]["outside_change"] == expected
    other = client.ok("POST", "/api/projects", {"name": "Other"})
    assert other["project"]["outside_change"] is None


def test_a_removed_folder_is_said_while_the_page_is_open_and_quit_goes_on(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "Blot µ"})
    status, _ = upload(client, blot_bytes(tmp_path))
    assert status == 201
    shutil.rmtree(client.root / "Blot µ")
    answer = client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    assert answer["project"]["save_error"] == REMOVED.replace(repr(FOLDER), repr("Blot µ"))
    status, refused = upload(client, blot_bytes(tmp_path))
    assert (status, refused["code"]) == (422, "project_removed")
    assert client.call("POST", "/api/quit")[0] == 202
    assert not (client.root / "Blot µ").exists()


def test_a_switch_that_saves_over_a_change_outside_keeps_a_copy_and_says_so_at_the_next_open(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    workspace = api.Workspace(root, reveal=lambda folder: None, clock=FakeClock())
    first = workspace.create("A µ")
    real = storage.os.replace

    def held(src, dst, **kwargs):
        raise PermissionError(13, "held by another process", str(dst))

    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "replace", held)
        edit(first)
    assert first.dirty and storage.os.replace is real
    theirs = _hand_edited(first.folder)
    workspace.create("B β")  # saves A first: over the change, keeping it
    assert (first.folder / COPY).read_bytes() == theirs
    reopened = workspace.open("A µ")
    assert reopened.outside_change == {
        "event": 0,
        "message": _found("A µ", repr(COPY), "1 version"),
        "copies": [COPY],
    }


def test_quit_keeps_a_copy_of_a_file_changed_outside(client, monkeypatch):
    client.ok("POST", "/api/projects", {"name": "Blot µ"})
    folder = client.root / "Blot µ"
    real = storage.os.replace

    def held(src, dst, **kwargs):
        raise PermissionError(13, "held by another process", str(dst))

    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    monkeypatch.setattr(storage.os, "replace", held)
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})
    monkeypatch.setattr(storage.os, "replace", real)
    theirs = _hand_edited(folder)

    def full(path, data, mode=None):
        raise OSError(errno.ENOSPC, "No space left on device", str(path))

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_write_new", full)
        status, refused = client.call("POST", "/api/quit")
    assert (status, refused["code"]) == (409, "unsaved_changes")  # no copy: no quit
    assert client.call("POST", "/api/quit")[0] == 202
    assert (folder / COPY).read_bytes() == theirs


def test_a_reopen_from_another_tab_shows_the_copies(client):
    client.ok("POST", "/api/projects", {"name": "Blot µ"})
    folder = client.root / "Blot µ"
    (folder / "project.conflict 2026-10-01 0800.json").write_bytes(b"{}")
    doc = json.loads((folder / storage.PROJECT_FILE).read_bytes())
    doc["next_id"] += 5  # changed outside, so the reopen reads it
    (folder / storage.PROJECT_FILE).write_bytes(storage.document_bytes(doc))
    answer = client.ok("POST", "/api/projects/open", {"name": "Blot µ"})
    assert answer["project"]["outside_change"] == {
        "event": 0,
        "message": _found("Blot µ", "'project.conflict 2026-10-01 0800.json'", "1 version"),
        "copies": ["project.conflict 2026-10-01 0800.json"],
    }


def _code(name: str) -> str:
    return (server.STATIC_DIR / name).read_text(encoding="utf-8")


def test_the_page_shows_the_outside_change_from_every_state_until_dismissed():
    html = _code("index.html")
    panel = html[html.index('<aside class="panel">') :]
    assert panel.index('id="outside-change"') < panel.index("<h2>Images</h2>")
    assert '<div id="outside-change" class="outside-change" hidden>' in html
    assert '<button id="outside-change-dismiss" type="button">Dismiss</button>' in html
    # The live region a screen reader hears it from: always there, never hidden,
    # outside the workspace (which a page without a project hides).
    live = re.search(r'<p id="outside-change-said"[^>]*>', html)
    assert live and 'role="status"' in live[0] and "hidden" not in live[0]
    assert html.index('id="outside-change-said"') < html.index('<main id="workspace"')
    script = _code("app.js")
    render = script[script.index("function render() {") :]
    render = render[: render.index("\n}\n")]
    assert "renderOutsideChange(project);" in render
    same = script[script.index("function sameNotice(") :]
    same = same[: same.index("\n}\n")]
    assert "seen.open_id === project.open_id && seen.event === change.event" in same
    shown = script[script.index("function renderOutsideChange(project) {") :]
    shown = shown[: shown.index("\n}\n")]
    for part in (
        "const change = project.outside_change;",
        "change !== null && !sameNotice(dismissedOutside, project, change)",
        'const text = shown ? change.message : "";',
        '$("outside-change-text").textContent = text;',
        "notice.hidden = !shown;",
        "shown && !sameNotice(saidOutside, project, change)",
    ):
        assert part in shown, part
    # A notice that appears or changes its words keeps the focused control of
    # the panel in the window (the page check measures it).
    moved = shown.index('shown && (notice.hidden || $("outside-change-text").textContent !== text)')
    kept = shown.index('focused.scrollIntoView({ block: "nearest" });')
    assert moved < shown.index("notice.hidden = !shown;") < kept
    assert "notice.parentElement.contains(focused)" in shown
    # Its words reach the live region once per notice, after the notice is shown.
    told = shown.index('$("outside-change-said").textContent = change.message;')
    assert shown.index("notice.hidden = !shown;") < told
    assert shown.index("saidOutside = { open_id: project.open_id, event: change.event };") < told
    dismiss = script[script.index('$("outside-change-dismiss").addEventListener("click"') :]
    dismiss = dismiss[: dismiss.index("\n});")]
    assert (
        "dismissedOutside = { open_id: project.open_id, event: project.outside_change.event };"
        in dismiss
    )
    assert dismiss.index('$("outside-change").hidden = true;') < dismiss.index(
        '$("switch-project").focus();'
    )
    send = script[script.index("async function send(") :]
    assert "outside-change" not in send[: send.index("\n}\n")]  # an edit's answer keeps it


def test_a_page_that_follows_a_switch_says_what_became_of_its_project():
    # #101: a switch may drop the project's changes (its folder removed): the
    # page says so, from the server's record of the opening it showed.
    script = _code("app.js")
    follow = script[script.index("function followOpening(") :]
    follow = follow[: follow.index("\n}\n")]
    assert "followedText(before, answer.project, follow.refused, answer.closed)" in follow
    text = script[script.index("function closedText(before, closed) {") :]
    text = text[: text.index("\n}\n")]
    for part in (
        "(closed || []).find((item) => item.open_id === before.open_id)",
        "return `${name} was closed.`;",
        "entry.saved ? `${name} is saved.` : `${name} was closed with changes not saved:"
        " ${entry.why}.`",
    ):
        assert part in text, part
    followed = script[script.index("function followedText(") :]
    assert "${closedText(before, closed)}" in followed[: followed.index("\n}\n")]
    assert "is saved.`;" not in followed[: followed.index("function closedText(")]


def test_the_project_answer_carries_the_openings_switches_closed(client, tmp_path):
    client.ok("POST", "/api/projects", {"name": "A µ"})
    status, _ = upload(client, blot_bytes(tmp_path))
    assert status == 201
    shutil.rmtree(client.root / "A µ")
    client.ok("PUT", "/api/lanes", {"lanes": [{"condition": "vehicle"}]})  # not saved
    client.ok("POST", "/api/projects", {"name": "B β"})  # another tab's switch drops it
    answer = client.ok("GET", "/api/project")
    assert answer["closed"] == [{"open_id": 1, "name": "A µ", "saved": False, "why": GONE}]
    client.ok("POST", "/api/projects", {"name": "C γ"})
    assert client.ok("GET", "/api/project")["closed"][-1] == {
        "open_id": 2,
        "name": "B β",
        "saved": True,
        "why": None,
    }


def test_no_path_reaches_a_notice_or_the_log(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    s = opened(tmp_path)
    (s.folder / storage.PROJECT_FILE).write_bytes(b'{"changed": "outside"}')
    edit(s, "a")
    (s.folder / storage.PROJECT_FILE).unlink()
    edit(s, "b")
    said = [s.outside_change["message"]]
    shutil.rmtree(s.folder)
    edit(s, "c")
    said.append(str(s.save_error))
    with pytest.raises(OperationError) as refused:
        imported(s, 4)
    said.append(str(refused.value))
    for text in (*said, *caplog.messages):
        assert str(tmp_path) not in text and tmp_path.as_posix() not in text, text


def test_the_session_log_says_what_was_kept(tmp_path, caplog):
    s = opened(tmp_path)
    caplog.set_level(logging.WARNING, logger="proteia.core.session")
    other = elsewhere(s)
    for label in ("a", "b"):
        ops.set_lanes(other, [ops.LaneInput(label)])
    put(s.folder, other.project)
    edit(s)
    (s.folder / storage.PROJECT_FILE).write_bytes(b"{not json")
    edit(s, "c")
    doc = json.loads(file_bytes(s))
    doc["schema_version"] = 99
    (s.folder / storage.PROJECT_FILE).write_bytes(json.dumps(doc).encode())
    edit(s, "d")
    assert caplog.messages[-3:] == [
        f"in {FOLDER!r}: its project.json was changed outside Proteia; kept that version as"
        f" {COPY!r} and saves Proteia's (extends_base: that version has 2 changes this one"
        " does not, this one 1)",
        f"in {FOLDER!r}: its project.json was changed outside Proteia; kept that version as"
        f" 'project.conflict {MINUTE} (2).json' and saves Proteia's (it does not read as a"
        " project)",
        f"in {FOLDER!r}: its project.json was changed outside Proteia; kept that version as"
        f" 'project.conflict {MINUTE} (3).json' and saves Proteia's (it was saved by a newer"
        " Proteia, schema 99)",
    ]
