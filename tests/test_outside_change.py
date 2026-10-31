# SPDX-License-Identifier: Apache-2.0
"""A ``project.json`` changed outside Proteia while its project is open (#101).

A session saves only over the ``project.json`` it last read or wrote, compared
by the SHA-256 of its bytes: a file replaced (a synced or restored copy),
edited or removed outside Proteia since is not saved over, the change stays in
memory as after any failed save, and no image file is deleted as an orphan
while the session cannot tell what the file on disk references. Opening the
project again reads the version on disk.

Both versions are kept: the refused save keeps the project as the session
holds it in a copy in the folder, ``project.unsaved-<UTC time>.json``, which
the message names, with how to open it instead; one copy per change outside
Proteia, written again only when the project changed since. The image files a
kept copy references survive every cleanup, and an import skips their ids.

The outside version is made as a sync would make it: a copy of the folder,
changed by a session of its own (importing an image only it references), is
copied back over the open folder.
"""

import codecs
import dataclasses
import errno
import hashlib
import logging
import os
import shutil
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from conftest import FakeClock
from proteia.core import operations as ops
from proteia.core import storage
from proteia.core.model import BoxSize, Role
from proteia.core.operations import ErrorCode, LaneInput, OperationError, ProjectSession
from proteia.core.session import save_to_folder
from proteia.core.storage import load_project
from proteia.web.state import project_state
from test_operations import (
    DARK,
    LIGHT,
    NARROW_X,
    ROW,
    ReplaceLock,
    band_of,
    blot,
    boxed,
    import_blot,
    listing,
    session_on,
)

# The messages, word for word: the folder by name (test_operations.FOLDER), never
# its path. These two are given where no save tried to keep a copy (an import
# before any save, or storage alone).
CHANGED = (
    "'專案 µ α β' was changed outside Proteia after Proteia last opened or saved it, so"
    " Proteia did not save over it: open the project again to read the version now on"
    " disk (the changes not saved here are then dropped)"
)
REMOVED = (
    "'專案 µ α β' was changed outside Proteia: its project.json was removed after"
    " Proteia last opened or saved it, so Proteia did not write it again; put the file"
    " back to go on saving the project"
)


def changed_kept(copy: str) -> str:
    """A refused save's message, the copy kept as ``copy``."""
    return (
        "'專案 µ α β' was changed outside Proteia after Proteia last opened or saved it, so"
        " Proteia did not save over it and kept a copy of its own version in the project's"
        f" folder as '{copy}': open the project again to read the version now on disk; to"
        " open Proteia's version instead, rename that copy to project.json while Proteia"
        " does not have the project open"
    )


def removed_kept(copy: str) -> str:
    return (
        "'專案 µ α β' was changed outside Proteia: its project.json was removed after"
        " Proteia last opened or saved it, so Proteia did not write it again and kept a"
        f" copy of its own version in the project's folder as '{copy}': put the file back"
        " to go on saving the project; to open Proteia's version instead, rename that copy"
        " to project.json while Proteia does not have the project open"
    )


def changed_not_kept(why: str) -> str:
    """A refused save's message, the copy not written for ``why``."""
    return (
        "'專案 µ α β' was changed outside Proteia after Proteia last opened or saved it, so"
        " Proteia did not save over it, and it could not keep a copy of its own version"
        f" either ({why}): the changes not saved are kept only while the project stays"
        " open; open the project again to read the version now on disk (the changes not"
        " saved here are then dropped)"
    )


def removed_not_kept(why: str) -> str:
    return (
        "'專案 µ α β' was changed outside Proteia: its project.json was removed after"
        " Proteia last opened or saved it, so Proteia did not write it again, and it could"
        f" not keep a copy of its own version either ({why}): the changes not saved are"
        " kept only while the project stays open; put the file back to go on saving the"
        " project"
    )


def reopening_kept(copy: str) -> str:
    """Opening the project again refused, its project.json removed, the copy kept."""
    return (
        "'專案 µ α β' was changed outside Proteia: its project.json was removed after"
        " Proteia last opened or saved it, so there is nothing to open again; Proteia kept"
        f" a copy of its own version in the project's folder as '{copy}': put the file"
        " back, or rename that copy to project.json, then open the project again"
    )


def reopening_not_kept(why: str) -> str:
    return (
        "'專案 µ α β' was changed outside Proteia: its project.json was removed after"
        " Proteia last opened or saved it, so there is nothing to open again, and Proteia"
        f" could not keep a copy of its own version ({why}): put the file back, then open"
        " the project again"
    )


# A clock that stays at noon, 2026-10-31 UTC: every copy is named for that second.
NOON = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
COPY = "project.unsaved-20261031T120000Z.json"
COPY_2 = "project.unsaved-20261031T120000Z (2).json"
COPY_3 = "project.unsaved-20261031T120000Z (3).json"


def at_noon() -> FakeClock:
    return FakeClock(NOON, timedelta(0))


def boxed_at_noon(tmp_path: Path) -> tuple[ProjectSession, str, str]:
    """:func:`test_operations.boxed`, autosaving, with the clock at noon."""
    s = session_on(tmp_path, save_to_folder, clock=at_noon())
    image = import_blot(s, blot())
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM"), LaneInput("50 µM")])
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    return s, image, protein


def changed_outside(s: ProjectSession, tmp_path: Path) -> tuple[ProjectSession, str]:
    """Change ``s``'s project as a synced copy would: a copy of its folder
    imports an image, and is copied back over it. Returns the copy's session
    and the image only it references (whose id ``s`` takes next, if all its
    changes were saved)."""
    other = tmp_path / "copy α"
    shutil.copytree(s.folder, other)
    remote = ops.open_project(other, clock=FakeClock())
    newer = import_blot(remote, blot(), "remote β.tif")
    shutil.copytree(other, s.folder, dirs_exist_ok=True)
    return remote, newer


def older_restored(tmp_path: Path) -> tuple[ProjectSession, str]:
    """A session at noon whose ``project.json`` is put back, outside Proteia, as
    it was before two images were imported (an older copy restored): only the
    session's version references img-4 and img-5. Returns it and its protein."""
    s, image, protein = boxed_at_noon(tmp_path)
    path = s.folder / storage.PROJECT_FILE
    older = path.read_bytes()
    membrane = s.project.batch.membrane_of(image).id
    late = [import_blot(s, blot(), name, membrane_id=membrane) for name in ("late γ.tif", "δ.tif")]
    assert (image, protein, late, s.project.next_id) == ("img-1", "prot-3", ["img-4", "img-5"], 6)
    path.write_bytes(older)
    return s, protein


def copies(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir() if path.name.startswith("project.unsaved"))


def temp_files(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir() if path.name.endswith(".tmp"))


def holds(path: Path, s: ProjectSession) -> bool:
    """Whether ``path`` holds what saving ``s`` would write."""
    return path.read_bytes() == storage.project_to_json(s.project)


# --- the refusal ---


def test_a_save_never_goes_over_a_project_json_changed_outside(tmp_path):
    s, image, protein = boxed_at_noon(tmp_path)
    _, newer = changed_outside(s, tmp_path)
    path = s.folder / storage.PROJECT_FILE
    outside = path.read_bytes()

    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)  # returns normally
    assert band_of(s, band).lane_index == 0  # the edit is in memory
    assert s.dirty and isinstance(s.save_error, storage.ChangedOutsideError)
    assert isinstance(s.save_error, storage.ProjectError)  # handled as any failed save
    assert str(s.save_error) == changed_kept(COPY) and not s.save_error.removed
    assert (s.save_error.kept, s.save_error.not_kept) == (COPY, None)
    assert str(tmp_path) not in str(s.save_error)
    assert path.read_bytes() == outside  # the outside version is kept
    # The cleanup after the save did not run: the image only it references is kept.
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])
    assert s.undo_step is not None and s.undo_step.action == "place_box"

    # The next change is refused too, and an explicit save says why.
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert s.dirty and str(s.save_error) == changed_kept(COPY)
    with pytest.raises(storage.ChangedOutsideError, match="did not save over it"):
        ops.save(s)
    assert path.read_bytes() == outside
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])


def test_the_refused_save_keeps_the_project_in_a_copy_beside_the_file(tmp_path):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)

    # Atomic: the copy is all there, its temp file gone; it reads back as the project.
    assert copies(s.folder) == [COPY]
    assert temp_files(s.folder) == []
    assert holds(s.folder / COPY, s)
    assert storage.project_from_json((s.folder / COPY).read_bytes()) == s.project


def test_a_copy_is_durable_before_it_takes_its_name(tmp_path, monkeypatch):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    events: list[tuple[str, object]] = []
    real_fsync, real_link = os.fsync, os.link

    def fsync(fd: int) -> None:  # with the bytes written to the file by then
        real_fsync(fd)
        events.append(("fsync", os.fstat(fd).st_size))

    def link(src, dst, **kwargs):
        events.append(("link", Path(dst).name))
        real_link(src, dst, **kwargs)

    def fsync_dir(directory: Path) -> None:  # only noted: on POSIX it fsyncs the folder
        events.append(("fsync_dir", Path(directory).name))

    monkeypatch.setattr(storage.os, "fsync", fsync)
    monkeypatch.setattr(storage.os, "link", link)
    monkeypatch.setattr(storage, "_fsync_dir", fsync_dir)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    size = len(storage.project_to_json(s.project))
    assert size < 8192  # less than a write buffer: unflushed bytes would not count
    assert events == [("fsync", size), ("link", COPY), ("fsync_dir", "專案 µ α β")]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_a_copy_is_as_readable_as_the_project_file(tmp_path):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    (s.folder / storage.PROJECT_FILE).chmod(0o640)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert (s.folder / COPY).stat().st_mode & 0o777 == 0o640


def test_the_page_gets_the_refusal_as_a_code_with_its_message(tmp_path, monkeypatch):
    s, _, protein = boxed_at_noon(tmp_path)
    state = project_state(s.folder.name, s, s.project, open_id=1)
    assert (state["saved"], state["save_error"], state["save_error_code"]) == (True, None, None)

    # Another failed save has a message and no code of its own, and keeps no copy.
    lock = ReplaceLock(monkeypatch)
    lock.locked = True
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    state = project_state(s.folder.name, s, s.project, open_id=1)
    assert state["saved"] is False and state["save_error"]
    assert state["save_error_code"] is None
    assert copies(s.folder) == []
    lock.locked = False

    changed_outside(s, tmp_path)
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    state = project_state(s.folder.name, s, s.project, open_id=1)
    assert (state["saved"], state["save_error"], state["save_error_code"]) == (
        False,
        changed_kept(COPY),
        "project_changed_outside",
    )


def test_a_removed_project_json_is_not_written_again(tmp_path):
    s, _, protein = boxed_at_noon(tmp_path)
    path = s.folder / storage.PROJECT_FILE
    saved = path.read_bytes()
    path.unlink()

    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert s.dirty and isinstance(s.save_error, storage.ChangedOutsideError)
    assert str(s.save_error) == removed_kept(COPY) and s.save_error.removed
    assert not path.exists()
    assert holds(s.folder / COPY, s)
    kept = (s.folder / COPY).read_bytes()

    # The file Proteia last saved, put back: the next change saves both edits.
    path.write_bytes(saved)
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert (s.dirty, s.save_error) == (False, None)
    assert load_project(s.folder) == s.project
    assert load_project(s.folder).batch.find_band(band)[1].lane_index == 0
    assert (s.folder / COPY).read_bytes() == kept  # never deleted, nor written again

    # A later change outside Proteia is another: another copy, the first as it was.
    path.write_bytes(b"{}")
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert str(s.save_error) == changed_kept(COPY_2)
    assert (s.folder / COPY).read_bytes() == kept
    assert holds(s.folder / COPY_2, s)


def test_a_removed_project_folder_is_not_made_again(tmp_path):
    s = session_on(tmp_path, save_to_folder)  # no image: nothing else refuses the save
    shutil.rmtree(s.folder)
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert str(s.save_error) == removed_not_kept("No such file or directory")
    assert (s.save_error.kept, type(s.save_error.not_kept)) == (None, FileNotFoundError)
    assert not s.folder.exists()  # neither project.json, a copy, images/ or exports/
    assert s.dirty


def test_a_file_replaced_while_the_save_writes_is_kept(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder, clock=at_noon())
    path = s.folder / storage.PROJECT_FILE
    outside = b'{"written": "outside Proteia"}\n'
    real_fsync = os.fsync
    replaced: list[int] = []

    def fsync(fd: int) -> None:  # the new bytes are in the temp file: replace the old now
        real_fsync(fd)
        if not replaced:
            replaced.append(fd)
            path.write_bytes(outside)

    monkeypatch.setattr(storage.os, "fsync", fsync)
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert replaced and str(s.save_error) == changed_kept(COPY)
    assert path.read_bytes() == outside
    assert holds(s.folder / COPY, s)
    assert temp_files(s.folder) == []


def test_a_file_replaced_while_a_replace_is_retried_is_kept(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder, clock=at_noon())
    path = s.folder / storage.PROJECT_FILE
    outside = b'{"written": "outside Proteia"}\n'
    real_replace = os.replace
    tries: list[str] = []

    def replace(src, dst, **kwargs):
        if Path(dst).name == storage.PROJECT_FILE and not tries:
            tries.append(str(src))
            path.write_bytes(outside)  # the reader holding it was a sync, writing it
            raise PermissionError(13, "held by another process", str(dst))
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    monkeypatch.setattr(storage.os, "replace", replace)
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert tries and str(s.save_error) == changed_kept(COPY)
    assert path.read_bytes() == outside
    assert temp_files(s.folder) == []


def test_a_project_json_that_cannot_be_read_fails_the_save_as_a_file_error(tmp_path):
    s, image, protein = boxed(tmp_path, save_to_folder)
    path = s.folder / storage.PROJECT_FILE
    leftover = s.folder / storage.IMAGES_DIR / f".img-9.tif.{'x' * 8}.part"
    leftover.write_bytes(b"a crash left it")
    path.unlink()
    path.mkdir()  # cannot be read as a file (Windows refuses, POSIX says it is a folder)

    # An import goes on, deletes nothing while the file cannot be compared, and
    # its save fails as a file error, kept in memory, with no copy kept.
    newer = import_blot(s, blot(), "late γ.tif")
    assert s.project.batch.find_image(newer)
    assert s.dirty and isinstance(s.save_error, OSError)
    assert not isinstance(s.save_error, storage.ChangedOutsideError)
    assert leftover.exists()
    assert listing(s) == sorted([leftover.name, f"{image}.tif", f"{newer}.tif"])
    assert copies(s.folder) == []


# --- the copy, kept again ---


def test_more_edits_after_the_refusal_update_the_one_copy(tmp_path):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    first = (s.folder / COPY).read_bytes()

    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert holds(s.folder / COPY, s) and (s.folder / COPY).read_bytes() != first
    for move in (ops.undo, ops.redo, ops.undo):  # each a change, kept so
        move(s)
        assert str(s.save_error) == changed_kept(COPY)
        assert holds(s.folder / COPY, s)
    assert copies(s.folder) == [COPY]
    assert temp_files(s.folder) == []


def test_a_refusal_with_nothing_changed_since_the_copy_writes_nothing(tmp_path, monkeypatch):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    written: list[str] = []
    real_temp = storage._temp_file

    def temp_file(directory, name, suffix):
        written.append(name)
        return real_temp(directory, name, suffix)

    monkeypatch.setattr(storage, "_temp_file", temp_file)
    for _ in range(3):  # as quitting or switching saves first, each time
        with pytest.raises(storage.ChangedOutsideError) as info:
            ops.save(s)
        assert str(info.value) == changed_kept(COPY)
        assert s.save_error is info.value
    assert written == []
    assert copies(s.folder) == [COPY] and holds(s.folder / COPY, s)


def test_a_copy_never_replaces_a_file_or_folder_of_its_name(tmp_path):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    (s.folder / COPY).write_bytes(b"mine")
    (s.folder / COPY_2).mkdir()

    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == changed_kept(COPY_3)
    assert copies(s.folder) == [COPY_2, COPY_3, COPY]
    assert (s.folder / COPY).read_bytes() == b"mine" and (s.folder / COPY_2).is_dir()
    assert holds(s.folder / COPY_3, s)


def test_a_copy_name_taken_in_another_case_is_taken_where_names_ignore_case(tmp_path):
    probe = tmp_path / "Probe"
    probe.touch()
    ignores_case = (tmp_path / "probe").exists()
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    shouting = s.folder / COPY.upper()
    shouting.write_bytes(b"mine")

    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    expected = COPY_2 if ignores_case else COPY
    assert str(s.save_error) == changed_kept(expected)
    assert shouting.read_bytes() == b"mine"
    assert holds(s.folder / expected, s)


def test_a_copy_changed_or_removed_outside_is_left_and_another_kept(tmp_path):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    (s.folder / COPY).write_bytes(b"edited by hand")

    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert str(s.save_error) == changed_kept(COPY_2)
    assert (s.folder / COPY).read_bytes() == b"edited by hand"
    assert holds(s.folder / COPY_2, s)

    (s.folder / COPY_2).unlink()  # removed: the next refusal keeps it again
    ops.set_box_size(s, protein, BoxSize(width=13, height=7))
    assert str(s.save_error) == changed_kept(COPY_2)
    assert holds(s.folder / COPY_2, s)
    assert (s.folder / COPY).read_bytes() == b"edited by hand"


def test_a_copy_edited_while_it_is_replaced_is_left_and_another_kept(tmp_path, monkeypatch):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    real_fsync = os.fsync
    edited: list[int] = []

    def fsync(fd: int) -> None:  # the new bytes are in the temp file: edit the copy now
        real_fsync(fd)
        if not edited:
            edited.append(fd)
            (s.folder / COPY).write_bytes(b"edited meanwhile")

    monkeypatch.setattr(storage.os, "fsync", fsync)
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert edited and str(s.save_error) == changed_kept(COPY_2)
    assert (s.folder / COPY).read_bytes() == b"edited meanwhile"
    assert holds(s.folder / COPY_2, s)
    assert temp_files(s.folder) == []


def test_where_the_file_system_has_no_hard_links_a_copy_is_renamed_into_place(
    tmp_path, monkeypatch
):
    def link(src, dst, **kwargs):  # as FAT and exFAT answer
        raise PermissionError(errno.EPERM, "Operation not permitted")

    real_rename = os.rename
    renamed: list[str] = []

    def rename(src, dst, **kwargs):
        renamed.append(Path(dst).name)
        real_rename(src, dst, **kwargs)

    monkeypatch.setattr(storage.os, "link", link)
    monkeypatch.setattr(storage.os, "rename", rename)
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    (s.folder / COPY).write_bytes(b"mine")  # taken: never renamed over, nor tried

    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == changed_kept(COPY_2)
    assert renamed == [COPY_2]
    assert (s.folder / COPY).read_bytes() == b"mine"
    assert holds(s.folder / COPY_2, s)
    assert temp_files(s.folder) == []


def test_a_name_taken_while_a_copy_is_renamed_into_place_is_left(tmp_path, monkeypatch):
    def link(src, dst, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "Operation not supported")

    real_rename = os.rename
    taken: list[str] = []

    def rename(src, dst, **kwargs):
        if not taken:  # another writer took the name in the instant between
            taken.append(Path(dst).name)
            Path(dst).write_bytes(b"theirs")
            raise FileExistsError(errno.EEXIST, "File exists")
        real_rename(src, dst, **kwargs)

    monkeypatch.setattr(storage.os, "link", link)
    monkeypatch.setattr(storage.os, "rename", rename)
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert taken == [COPY] and str(s.save_error) == changed_kept(COPY_2)
    assert (s.folder / COPY).read_bytes() == b"theirs"
    assert holds(s.folder / COPY_2, s)


# --- the copy, not kept ---


@pytest.mark.parametrize(
    ("error", "why"),
    [
        (OSError(errno.ENOSPC, "No space left on device"), "No space left on device"),
        (PermissionError(errno.EACCES, "Permission denied"), "Permission denied"),  # read-only
        (OSError("a reason without a system message"), "OSError"),
    ],
)
def test_a_copy_that_cannot_be_written_is_said_and_the_change_stays_in_memory(
    tmp_path, monkeypatch, error, why
):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    outside = (s.folder / storage.PROJECT_FILE).read_bytes()
    real_temp = storage._temp_file

    def temp_file(directory, name, suffix):
        if name.startswith("project.unsaved-"):
            raise error
        return real_temp(directory, name, suffix)

    monkeypatch.setattr(storage, "_temp_file", temp_file)
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == changed_not_kept(why)
    assert (s.save_error.kept, s.save_error.not_kept) == (None, error)
    assert s.dirty and band_of(s, band).lane_index == 0
    assert copies(s.folder) == [] and temp_files(s.folder) == []
    assert (s.folder / storage.PROJECT_FILE).read_bytes() == outside

    # Once it can be written, the next refusal keeps it.
    monkeypatch.undo()
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert str(s.save_error) == changed_kept(COPY)
    assert holds(s.folder / COPY, s)


def test_a_removed_file_whose_copy_cannot_be_written_says_both(tmp_path, monkeypatch):
    s, _, protein = boxed_at_noon(tmp_path)
    (s.folder / storage.PROJECT_FILE).unlink()

    def temp_file(directory, name, suffix):
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(storage, "_temp_file", temp_file)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == removed_not_kept("Permission denied")
    assert copies(s.folder) == []


def test_a_copy_whose_path_would_be_too_long_is_not_written(tmp_path, monkeypatch):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    # The temp file a copy is written through: its name is the longest.
    temp = os.path.join(os.path.abspath(s.folder), f".{COPY}.{'x' * 8}.tmp")
    length = len(temp.encode("utf-16-le")) // 2

    monkeypatch.setattr(storage, "PATH_LIMIT", length - 1)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == changed_not_kept("the project folder's path is too long for it")
    assert copies(s.folder) == [] and temp_files(s.folder) == []

    monkeypatch.setattr(storage, "PATH_LIMIT", length)  # it just fits
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert str(s.save_error) == changed_kept(COPY)


def test_a_copy_held_open_is_left_and_another_kept(tmp_path, monkeypatch):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    first = (s.folder / COPY).read_bytes()
    real_replace = os.replace

    def replace(src, dst, **kwargs):  # a viewer holds it, as on Windows
        if Path(dst).name == COPY:
            raise PermissionError(13, "held by another process", str(dst))
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    monkeypatch.setattr(storage.os, "replace", replace)
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert str(s.save_error) == changed_kept(COPY_2)
    assert (s.folder / COPY).read_bytes() == first
    assert holds(s.folder / COPY_2, s)
    assert copies(s.folder) == [COPY_2, COPY] and temp_files(s.folder) == []

    monkeypatch.undo()  # the new copy is the one brought up to date
    ops.set_box_size(s, protein, BoxSize(width=13, height=7))
    assert str(s.save_error) == changed_kept(COPY_2)
    assert holds(s.folder / COPY_2, s) and (s.folder / COPY).read_bytes() == first


# --- what goes on working ---


def test_saves_in_a_row_and_from_two_threads_each_go_over_the_last(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    errors: list[BaseException] = []

    def edit(names: list[str]) -> None:
        try:
            for name in names:
                ops.set_lanes(s, [LaneInput(name)])
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=edit, args=([f"{tab} {i}" for i in range(5)],))
        for tab in ("tab α", "tab β")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    assert (s.dirty, s.save_error) == (False, None)
    assert load_project(s.folder) == s.project
    assert len(s.project.log) == 11  # created, then ten edits, each saved
    assert copies(s.folder) == []


def test_refused_saves_and_closes_from_two_threads_keep_one_copy_and_every_image(tmp_path):
    s, image, protein = boxed_at_noon(tmp_path)
    _, newer = changed_outside(s, tmp_path)
    errors: list[BaseException] = []

    def run(step) -> None:
        try:
            for i in range(5):
                step(i)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(
            target=run,
            args=(lambda i: ops.set_box_size(s, protein, BoxSize(width=11 + 2 * i, height=7)),),
        ),
        threading.Thread(target=run, args=(lambda i: s.close(),)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    assert str(s.save_error) == changed_kept(COPY)
    assert copies(s.folder) == [COPY] and holds(s.folder / COPY, s)
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])


def test_a_save_right_after_opening_a_hand_edited_file_goes_over_it(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    ops.set_lanes(s, [LaneInput("vehicle")])
    path = s.folder / storage.PROJECT_FILE
    # A Windows editor's copy: a BOM and CRLF, which loading tolerates.
    path.write_bytes(codecs.BOM_UTF8 + path.read_bytes().replace(b"\n", b"\r\n"))
    s.close()

    reopened = ops.open_project(s.folder, autosave=save_to_folder, clock=FakeClock())
    ops.set_lanes(reopened, [LaneInput("drug")])
    assert (reopened.dirty, reopened.save_error) == (False, None)
    assert load_project(s.folder) == reopened.project


def _replaced_while_read(monkeypatch, path: Path, outside: bytes) -> None:
    """Replace ``path`` with ``outside`` once its bytes were read, before they are parsed."""
    real_read = storage.read_project

    def read_project(folder, **kwargs):
        path.write_bytes(outside)
        return real_read(folder, **kwargs)

    monkeypatch.setattr(storage, "read_project", read_project)


def test_an_open_holds_the_bytes_it_read_and_saves_over_them_only(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder)
    ops.set_lanes(s, [LaneInput("vehicle")])
    path = s.folder / storage.PROJECT_FILE
    read = path.read_bytes()
    s.close()
    ops.set_lanes(s, [LaneInput("drug")])
    outside = path.read_bytes()
    path.write_bytes(read)

    _replaced_while_read(monkeypatch, path, outside)
    opened = ops.open_project(s.folder, autosave=save_to_folder, clock=at_noon())
    monkeypatch.undo()
    assert [lane.label for lane in opened.project.batch.lanes] == ["vehicle"]
    ops.set_lanes(opened, [LaneInput("dose")])
    assert str(opened.save_error) == changed_kept(COPY) and path.read_bytes() == outside


def test_a_reload_holds_the_bytes_it_read_and_saves_over_them_only(tmp_path, monkeypatch, caplog):
    s = session_on(tmp_path, save_to_folder, clock=at_noon())
    ops.set_lanes(s, [LaneInput("vehicle")])
    path = s.folder / storage.PROJECT_FILE
    read = path.read_bytes()
    ops.set_lanes(s, [LaneInput("drug")])
    outside = path.read_bytes()
    path.write_bytes(read)  # an older copy restored

    _replaced_while_read(monkeypatch, path, outside)
    caplog.set_level(logging.WARNING, logger="proteia.core.session")
    assert s.reload()
    assert caplog.records == []  # nothing unsaved was dropped
    monkeypatch.undo()
    assert [lane.label for lane in s.project.batch.lanes] == ["vehicle"]
    ops.set_lanes(s, [LaneInput("dose")])
    assert str(s.save_error) == changed_kept(COPY) and path.read_bytes() == outside


# --- opening the project again ---


def test_opening_again_after_the_refusal_reads_the_version_on_disk(tmp_path, caplog):
    s, image, protein = boxed_at_noon(tmp_path)
    remote, newer = changed_outside(s, tmp_path)
    path = s.folder / storage.PROJECT_FILE
    # The synced copy as a Windows editor left it: other bytes, the same project.
    path.write_bytes(codecs.BOM_UTF8 + path.read_bytes().replace(b"\n", b"\r\n"))
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == changed_kept(COPY)
    kept = (s.folder / COPY).read_bytes()

    caplog.set_level(logging.INFO, logger="proteia.core.session")
    caplog.clear()
    assert s.reload()  # the unsaved change goes; the version on disk is read
    assert [(r.levelname, r.getMessage()) for r in caplog.records][0] == (
        "WARNING",
        "in '專案 µ α β': dropped the changes that could not be saved, as its project.json"
        " was changed outside Proteia; they are kept in 'project.unsaved-20261031T120000Z.json'",
    )
    assert s.project == remote.project
    assert (s.dirty, s.save_error, s.undo_step, s.last_action) == (False, None, None, None)
    ops.set_polarity(s, newer, LIGHT)  # the next change saves over the file read
    assert (s.dirty, s.save_error) == (False, None)
    assert load_project(s.folder) == s.project
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])
    assert s.project.batch.find_image(newer).polarity == LIGHT
    assert not s.reload()
    assert (s.folder / COPY).read_bytes() == kept  # left as it was

    # A change outside Proteia after the reload is another: another copy.
    path.write_bytes(b"{}")
    ops.set_polarity(s, newer, DARK)
    assert str(s.save_error) == changed_kept(COPY_2)
    assert (s.folder / COPY).read_bytes() == kept


def test_opening_again_when_no_copy_could_be_kept_says_the_changes_are_dropped(
    tmp_path, monkeypatch, caplog
):
    s, _, protein = boxed_at_noon(tmp_path)
    changed_outside(s, tmp_path)

    def temp_file(directory, name, suffix):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(storage, "_temp_file", temp_file)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == changed_not_kept("No space left on device")
    caplog.set_level(logging.WARNING, logger="proteia.core.session")
    caplog.clear()
    assert s.reload()
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        (
            "WARNING",
            "in '專案 µ α β': dropped the changes that could not be saved, as its"
            " project.json was changed outside Proteia",
        )
    ]


def test_opening_again_takes_a_file_that_holds_the_unsaved_project(tmp_path, caplog):
    s = session_on(tmp_path, save_to_folder, clock=at_noon())
    path = s.folder / storage.PROJECT_FILE
    path.write_bytes(b"{}")  # changed outside Proteia
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert str(s.save_error) == changed_kept(COPY)
    undo = s.undo_step
    # Then the kept copy is renamed to project.json: it holds the project as open.
    os.replace(s.folder / COPY, path)

    caplog.set_level(logging.INFO, logger="proteia.core.session")
    caplog.clear()
    assert not s.reload()  # nothing to read: the file is taken as saved
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        (
            "INFO",
            "in '專案 µ α β': its project.json, changed outside Proteia, holds the project as"
            " it is open, so Proteia takes it for the file it saved",
        )
    ]
    assert (s.dirty, s.save_error, s.undo_step) == (False, None, undo)
    ops.set_lanes(s, [LaneInput("drug")])
    assert (s.dirty, s.save_error) == (False, None)
    assert load_project(s.folder) == s.project
    assert copies(s.folder) == []


def test_opening_again_with_changes_unsaved_for_another_reason_reads_nothing(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder)
    lock = ReplaceLock(monkeypatch)
    lock.locked = True
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert isinstance(s.save_error, PermissionError)
    project = s.project
    (s.folder / storage.PROJECT_FILE).write_bytes(b"{}")  # changed outside meanwhile
    assert not s.reload()  # the page said nothing of a change outside: nothing is dropped
    assert s.project is project and s.dirty


def test_the_kept_copy_renamed_to_project_json_opens_as_proteia_held_it(tmp_path):
    s, protein = older_restored(tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == changed_kept(COPY)
    held = s.project

    # Opened again, the older version is read; then, Proteia no longer having
    # the project open, the copy is renamed and opened: with its images.
    assert s.reload() and [i.id for i in s.project.batch.iter_images()] == ["img-1"]
    s.close()
    os.replace(s.folder / COPY, s.folder / storage.PROJECT_FILE)
    opened = ops.open_project(s.folder, autosave=save_to_folder, clock=at_noon())
    assert opened.project == held
    ops.set_polarity(opened, "img-5", LIGHT)  # its pixels read, and it saves
    assert (opened.dirty, opened.save_error) == (False, None)
    assert listing(opened) == ["img-1.tif", "img-4.tif", "img-5.tif"]


def test_after_a_removal_the_kept_copy_renamed_to_project_json_opens(tmp_path):
    s, _, protein = boxed_at_noon(tmp_path)
    path = s.folder / storage.PROJECT_FILE
    path.unlink()
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == removed_kept(COPY)
    held = s.project

    (s.folder / COPY).rename(path)  # Proteia stopped with the change unsaved
    opened = ops.open_project(s.folder, autosave=save_to_folder, clock=at_noon())
    assert opened.project == held


def test_opening_again_a_project_whose_file_was_removed_says_what_to_do(tmp_path):
    s, _, protein = boxed_at_noon(tmp_path)
    path = s.folder / storage.PROJECT_FILE
    path.unlink()
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == removed_kept(COPY)
    held, error = s.project, s.save_error

    with pytest.raises(storage.ChangedOutsideError) as info:
        s.reload()
    assert str(info.value) == reopening_kept(COPY)
    assert (info.value.removed, info.value.kept) == (True, COPY)
    assert str(tmp_path) not in str(info.value)
    assert (s.project, s.dirty, s.save_error) == (held, True, error)  # as it was
    assert copies(s.folder) == [COPY] and holds(s.folder / COPY, s)

    (s.folder / COPY).rename(path)  # as it says: then opened again
    assert not s.reload()
    assert (s.project, s.dirty, s.save_error) == (held, False, None)


def test_opening_again_a_project_whose_file_was_removed_keeps_its_only_copy(tmp_path, monkeypatch):
    s, _, _ = boxed_at_noon(tmp_path)
    (s.folder / storage.PROJECT_FILE).unlink()  # no change since: the session holds it alone
    with pytest.raises(storage.ChangedOutsideError) as info:
        s.reload()
    assert str(info.value) == reopening_kept(COPY)
    assert holds(s.folder / COPY, s) and not s.dirty

    (s.folder / COPY).unlink()

    def temp_file(directory, name, suffix):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(storage, "_temp_file", temp_file)
    with pytest.raises(storage.ChangedOutsideError) as info:
        s.reload()
    assert str(info.value) == reopening_not_kept("No space left on device")
    assert copies(s.folder) == []


def test_the_file_put_back_as_proteia_saved_it_takes_the_changes_at_a_reopen(tmp_path):
    s, _, protein = boxed_at_noon(tmp_path)
    path = s.folder / storage.PROJECT_FILE
    saved = path.read_bytes()
    path.write_bytes(b"{}")  # changed outside Proteia
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == changed_kept(COPY)
    path.write_bytes(saved)  # then the change reverted: the file Proteia saved
    undo = s.undo_step

    assert not s.reload()  # nothing to read: the changes kept go over it now
    assert (s.dirty, s.save_error, s.undo_step) == (False, None, undo)
    assert load_project(s.folder) == s.project
    assert load_project(s.folder).batch.find_band(band)[1].lane_index == 0
    assert holds(s.folder / COPY, s)  # left as it was


def test_a_reopen_of_the_file_a_clean_session_read_writes_nothing(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    ops.set_lanes(s, [LaneInput("vehicle")])
    path = s.folder / storage.PROJECT_FILE
    # A Windows editor's copy, which the session reads: other bytes than it writes.
    edited = codecs.BOM_UTF8 + path.read_bytes().replace(b"\n", b"\r\n")
    path.write_bytes(edited)
    s.close()
    opened = ops.open_project(s.folder, autosave=save_to_folder, clock=at_noon())
    path.write_bytes(b"{}")
    with pytest.raises(storage.ChangedOutsideError):
        ops.save(opened)  # refused with nothing unsaved
    assert not opened.dirty
    path.write_bytes(edited)  # the change reverted

    assert not opened.reload()
    assert (opened.dirty, opened.save_error) == (False, None)
    assert path.read_bytes() == edited  # not written again


def test_a_reopen_leaves_another_failure_of_a_clean_session(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder)
    lock = ReplaceLock(monkeypatch)
    lock.locked = True
    with pytest.raises(PermissionError):
        ops.save(s)
    assert not s.dirty and isinstance(s.save_error, PermissionError)
    assert not s.reload()
    assert isinstance(s.save_error, PermissionError)


def test_a_reopen_takes_a_file_that_holds_the_project_in_other_bytes(tmp_path):
    s, _, protein = boxed_at_noon(tmp_path)
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    path = s.folder / storage.PROJECT_FILE
    # Rewritten outside Proteia, the project unchanged (a sync, an editor).
    path.write_bytes(codecs.BOM_UTF8 + path.read_bytes().replace(b"\n", b"\r\n"))
    undo = s.undo_step

    assert not s.reload()  # nothing new to read: history kept, no new opening
    assert (s.dirty, s.save_error, s.undo_step) == (False, None, undo)
    ops.set_box_size(s, protein, BoxSize(width=13, height=7))  # saved over it
    assert (s.dirty, s.save_error) == (False, None)
    assert path.read_bytes() == storage.project_to_json(s.project)


def test_an_import_goes_on_over_a_file_that_holds_the_project_in_other_bytes(tmp_path):
    s, image, _ = boxed_at_noon(tmp_path)
    path = s.folder / storage.PROJECT_FILE
    path.write_bytes(codecs.BOM_UTF8 + path.read_bytes().replace(b"\n", b"\r\n"))

    newer = import_blot(s, blot(), "late γ.tif")
    assert (s.dirty, s.save_error) == (False, None)
    assert load_project(s.folder) == s.project
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])


def test_a_file_changed_that_cannot_be_read_again_refuses_an_import_as_changed(
    tmp_path, monkeypatch
):
    s, image, _ = boxed_at_noon(tmp_path)
    path = s.folder / storage.PROJECT_FILE
    path.write_bytes(b"{}")  # changed outside Proteia
    real_read = Path.read_bytes
    reads: list[str] = []

    def read_bytes(read: Path) -> bytes:
        if read.name == storage.PROJECT_FILE:
            reads.append(read.name)
            if len(reads) == 2:  # held when read again, to compare what it holds
                raise PermissionError(13, "held by another process", str(read))
        return real_read(read)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(OperationError) as info:
        import_blot(s, blot(), "late γ.tif")
    assert (info.value.code, str(info.value)) == (ErrorCode.PROJECT_CHANGED_OUTSIDE, CHANGED)
    assert listing(s) == [f"{image}.tif"]


def test_a_refused_import_says_what_the_file_is_now(tmp_path):
    s, image, protein = boxed_at_noon(tmp_path)
    path = s.folder / storage.PROJECT_FILE
    path.write_bytes(b"{}")
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == changed_kept(COPY)
    path.unlink()  # then removed

    with pytest.raises(OperationError) as info:
        import_blot(s, blot(), "late γ.tif")
    assert info.value.code is ErrorCode.PROJECT_CHANGED_OUTSIDE
    assert str(info.value) == removed_kept(COPY)
    assert listing(s) == [f"{image}.tif"] and copies(s.folder) == [COPY]


def test_a_refused_import_of_a_clean_session_names_no_copy(tmp_path):
    s, _, _ = boxed_at_noon(tmp_path)
    (s.folder / storage.PROJECT_FILE).unlink()
    with pytest.raises(OperationError) as info:
        import_blot(s, blot(), "late γ.tif")
    assert str(info.value) == REMOVED and copies(s.folder) == []


def test_a_project_json_made_while_a_project_is_created_is_left(tmp_path, monkeypatch):
    folder = tmp_path / "新 µ"
    real = storage.save_project_over

    def save_over(project, path, *, expect):  # another program made it meanwhile
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / storage.PROJECT_FILE).write_bytes(b'{"theirs": true}')
        return real(project, path, expect=expect)

    monkeypatch.setattr(storage, "save_project_over", save_over)
    with pytest.raises(OperationError) as info:
        ops.new_project(folder, clock=at_noon())
    assert info.value.code is ErrorCode.FOLDER_NOT_EMPTY
    assert (folder / storage.PROJECT_FILE).read_bytes() == b'{"theirs": true}'
    assert copies(folder) == []


# --- reading project.json while another program holds it ---


class HeldProjectFile:
    """``project.json`` held by another program for its first ``held`` reads
    (``None``: always), as Windows reports it (a sharing violation); the reads
    and the waits between them are recorded."""

    def __init__(self, monkeypatch, held: int | None) -> None:
        self.reads = 0
        self.waits: list[float] = []
        real_read = Path.read_bytes

        def read_bytes(path: Path) -> bytes:
            if path.name == storage.PROJECT_FILE:
                self.reads += 1
                if held is None or self.reads <= held:
                    raise PermissionError(13, "held by another process", str(path))
            return real_read(path)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)
        monkeypatch.setattr(storage.time, "sleep", self.waits.append)


def test_a_save_reads_the_file_again_while_another_program_holds_it(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder)
    held = HeldProjectFile(monkeypatch, held=2)
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert (s.dirty, s.save_error) == (False, None)
    assert held.waits == [0.05, 0.1]  # then read; read again right before the rename
    assert held.reads == 4
    assert load_project(s.folder) == s.project


def test_a_save_fails_as_a_file_error_while_the_file_stays_held(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder)
    before = (s.folder / storage.PROJECT_FILE).read_bytes()
    held = HeldProjectFile(monkeypatch, held=None)
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert s.dirty and isinstance(s.save_error, PermissionError)
    assert held.reads == 5 and held.waits == [0.05, 0.1, 0.2, 0.4]
    monkeypatch.undo()
    assert (s.folder / storage.PROJECT_FILE).read_bytes() == before
    assert copies(s.folder) == []


def test_a_folder_in_place_of_the_file_is_not_read_again(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder)
    path = s.folder / storage.PROJECT_FILE
    path.unlink()
    path.mkdir()
    waits: list[float] = []
    monkeypatch.setattr(storage.time, "sleep", waits.append)
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert s.dirty and isinstance(s.save_error, OSError)
    assert waits == []


def test_a_save_reads_project_json_twice(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder)
    held = HeldProjectFile(monkeypatch, held=0)
    ops.set_lanes(s, [LaneInput("vehicle")])  # before writing, and right before the rename
    assert (held.reads, s.dirty) == (2, False)


def test_an_import_reads_the_kept_copies_once(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder, clock=at_noon())
    leftover = s.folder / storage.IMAGES_DIR / "img-1.tif"
    leftover.parent.mkdir(exist_ok=True)
    leftover.write_bytes(b"left by a crash")  # so the cleanup before it deletes one
    real = storage.unsaved_references
    reads: list[Path] = []

    def unsaved_references(folder):
        reads.append(Path(folder))
        return real(folder)

    monkeypatch.setattr(storage, "unsaved_references", unsaved_references)
    assert import_blot(s, blot()) == "img-1"
    assert reads[:1] == [s.folder] and len(reads) == 1


# --- no image file is deleted ---


def test_closing_after_a_change_outside_deletes_no_image_it_references(tmp_path):
    s, image, _ = boxed(tmp_path, save_to_folder)
    _, newer = changed_outside(s, tmp_path)
    assert not s.dirty
    s.close()
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])


def test_an_import_after_a_change_outside_is_refused_before_touching_images(tmp_path):
    s, image, _ = boxed(tmp_path, save_to_folder)
    _, newer = changed_outside(s, tmp_path)
    assert newer == f"img-{s.project.next_id}"  # its file holds the id the import would take
    project, before = s.project, (s.folder / storage.PROJECT_FILE).read_bytes()

    with pytest.raises(OperationError) as info:
        import_blot(s, blot(), "late γ.tif", DARK)
    assert info.value.code is ErrorCode.PROJECT_CHANGED_OUTSIDE
    assert str(info.value) == CHANGED  # no save was refused: no copy to name
    assert s.project is project and not s.dirty
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])
    assert (s.folder / storage.PROJECT_FILE).read_bytes() == before
    assert copies(s.folder) == []


def test_an_import_after_a_refused_save_names_the_copy_it_kept(tmp_path):
    s, image, protein = boxed_at_noon(tmp_path)
    _, newer = changed_outside(s, tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)

    with pytest.raises(OperationError) as info:
        import_blot(s, blot(), "late γ.tif", DARK)
    assert info.value.code is ErrorCode.PROJECT_CHANGED_OUTSIDE
    assert str(info.value) == changed_kept(COPY)
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])


def test_images_only_a_kept_copy_references_survive_every_cleanup(tmp_path):
    s, protein = older_restored(tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == changed_kept(COPY)
    every = ["img-1.tif", "img-4.tif", "img-5.tif"]

    assert s.reload()  # opened again: the older version, which has only img-1
    assert [image.id for image in s.project.batch.iter_images()] == ["img-1"]
    assert s.project.next_id == 4
    ops.set_lanes(s, [LaneInput("vehicle")])  # saved: the cleanup after a save
    assert (s.dirty, s.save_error) == (False, None)
    assert listing(s) == every
    s.close()  # the cleanup on close
    assert listing(s) == every

    # The cleanup before an import keeps them, and the import skips their ids.
    image = import_blot(s, blot(), "new ε.tif")
    assert image == "img-6" and s.project.log[-1].params["image_id"] == "img-6"
    assert s.project.batch.membrane_of(image).id == "mem-7" and s.project.next_id == 8
    assert listing(s) == [*every, "img-6.tif"]

    # So does the first save of a session that opens the folder afresh.
    s.close()
    again = ops.open_project(s.folder, autosave=save_to_folder, clock=at_noon())
    ops.set_lanes(again, [LaneInput("drug")])
    assert (again.dirty, again.save_error) == (False, None)
    assert listing(again) == [*every, "img-6.tif"]


def test_an_import_takes_next_id_when_no_kept_copy_holds_it(tmp_path):
    s = session_on(tmp_path, save_to_folder, clock=at_noon())
    other = session_on(tmp_path / "other", save_to_folder)
    import_blot(other, blot())  # a project of img-1 only, next_id 3
    data = (other.folder / storage.PROJECT_FILE).read_bytes()
    (s.folder / "project.unsaved-20261030T000000Z.json").write_bytes(data)
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert s.project.next_id == 1
    assert s.image_number() == 2  # img-1 is the copy's
    assert s.project.next_id == 1  # unchanged by asking
    image = import_blot(s, blot())
    assert (image, s.project.batch.membrane_of(image).id) == ("img-2", "mem-3")
    image = import_blot(s, blot(), "β.tif")
    assert image == "img-4"  # next_id, which no copy holds


def test_a_kept_copy_that_does_not_read_as_a_project_keeps_the_images_it_names(tmp_path, caplog):
    s = session_on(tmp_path, save_to_folder, clock=at_noon())
    image = import_blot(s, blot())  # img-1, on mem-2: next_id 3
    valid = s.folder / "project.unsaved-20261031T000000Z.json"
    valid.write_bytes((s.folder / storage.PROJECT_FILE).read_bytes())  # references img-1
    invalid = s.folder / "project.unsaved-20261030T000000Z.json"  # read first
    # Cut short (or edited by hand, or of a newer schema): its text names img-3.
    cut = b'{"batch": {"membranes": [{"images": [{"file": "IMG-3.TIF"'
    invalid.write_bytes(cut)
    caplog.set_level(logging.WARNING, logger="proteia.core.session")
    ops.set_lanes(s, [LaneInput("vehicle")])  # a save that leaves no orphan reads no copy
    assert caplog.records == []
    named = s.folder / storage.IMAGES_DIR / "img-3.tif"
    named.write_bytes(b"its image")
    leftover = s.folder / storage.IMAGES_DIR / "img-8.tif"
    leftover.write_bytes(b"left by a crash")
    ops.remove_image(s, image)
    caplog.clear()

    s.close()  # the history is gone: img-1 only the valid copy keeps, img-3 the cut one
    assert listing(s) == [f"{image}.tif", "img-3.tif"]
    assert invalid.read_bytes() == cut and valid.is_file()
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        (
            "WARNING",
            "in '專案 µ α β': 'project.unsaved-20261030T000000Z.json' does not read as a"
            " project, so the image files its text names are kept for it (the file itself"
            " is left as it is)",
        )
    ]
    # An import skips the id the file it names holds.
    assert import_blot(s, blot()) == "img-4"


def test_while_a_kept_copy_cannot_be_read_no_image_file_is_deleted(tmp_path, monkeypatch, caplog):
    s = session_on(tmp_path, save_to_folder, clock=at_noon())
    (s.folder / COPY).write_bytes(b"{}")
    leftover = s.folder / storage.IMAGES_DIR / "img-1.tif"
    leftover.parent.mkdir(exist_ok=True)
    leftover.write_bytes(b"left by a crash")
    real_read = Path.read_bytes

    def read_bytes(self):  # held by another program, as Windows refuses
        if storage.is_unsaved(self.name):
            raise PermissionError(13, "held by another process", str(self))
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    caplog.set_level(logging.WARNING, logger="proteia.core.session")
    caplog.clear()
    ops.set_lanes(s, [LaneInput("vehicle")])  # saved: the cleanup after a save
    assert (s.dirty, s.save_error) == (False, None)
    assert leftover.exists()
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        (
            "WARNING",
            "in '專案 µ α β': no image file was deleted, as the copies of the project kept in"
            " its folder could not be read",
        )
    ]
    assert s.image_number() == 1  # unknown: next_id, and the leftover refuses it
    with pytest.raises(OperationError) as info:
        import_blot(s, blot())
    assert info.value.code is ErrorCode.LEFTOVER_FILE


def test_a_temp_file_of_a_copy_is_no_copy(tmp_path, caplog):
    s = session_on(tmp_path, save_to_folder, clock=at_noon())
    temp = s.folder / f".{COPY}.abcd1234.tmp"  # left by a crash while a copy was written
    temp.write_bytes(b'{"batch": "img-1.tif"')
    leftover = s.folder / storage.IMAGES_DIR / "img-1.tif"
    leftover.parent.mkdir(exist_ok=True)
    leftover.write_bytes(b"left by a crash")
    caplog.set_level(logging.WARNING, logger="proteia.core.session")
    caplog.clear()
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert not leftover.exists() and temp.exists()
    assert caplog.records == []


# --- storage ---


def test_save_project_over_writes_only_over_the_file_it_names(tmp_path):
    folder = tmp_path / "專案 µ α β"
    session = session_on(tmp_path)  # no autosave
    project = session.project
    path = folder / storage.PROJECT_FILE
    first = path.read_bytes()
    digest = hashlib.sha256(first).hexdigest()

    # Over the file whose SHA-256 it is given: the SHA-256 of what it wrote.
    written = storage.save_project_over(project, folder, expect=digest)
    assert written == hashlib.sha256(path.read_bytes()).hexdigest() == digest

    for expect in (None, "0" * 64):  # none expected, or other bytes
        with pytest.raises(storage.ChangedOutsideError) as info:
            storage.save_project_over(project, folder, expect=expect)
        assert str(info.value) == CHANGED
        assert (info.value.kept, info.value.not_kept) == (None, None)
    path.unlink()
    with pytest.raises(storage.ChangedOutsideError) as info:
        storage.save_project_over(project, folder, expect=digest)
    assert str(info.value) == REMOVED
    assert sorted(p.name for p in folder.iterdir()) == ["exports", "images"]

    # None: where there is no file yet, as a new project's.
    assert storage.save_project_over(project, folder, expect=None) == digest
    assert path.read_bytes() == first


def test_read_project_reads_the_bytes_it_is_given(tmp_path):
    session = session_on(tmp_path, save_to_folder)
    ops.set_lanes(session, [LaneInput("vehicle")])
    data = (session.folder / storage.PROJECT_FILE).read_bytes()
    ops.set_lanes(session, [LaneInput("drug")])
    project, migrated = storage.read_project(session.folder, data=data)
    assert [lane.label for lane in project.batch.lanes] == ["vehicle"] and not migrated


def test_the_names_of_kept_copies():
    taipei = timezone(timedelta(hours=8))
    assert (
        storage.unsaved_name(datetime(2026, 10, 31, 20, 0, 5, 999000, tzinfo=taipei))
        == "project.unsaved-20261031T120005Z.json"
    )
    assert storage.unsaved_name(NOON, 2) == COPY_2
    names = [
        COPY,
        COPY_2,
        "PROJECT.UNSAVED-20261031T120000Z.JSON",
        "project.unsaved-mine.json",
        f".{COPY}.abcd1234.tmp",
        f"{COPY}.tmp",
        "my project.unsaved-20261031T120000Z.json",
        "project.unsaved.json",
        "project.json",
        "project.schema1.json",
    ]
    assert [name for name in names if storage.is_unsaved(name)] == names[:4]


def test_unsaved_references_reads_each_copy_as_a_project(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    import_blot(s, blot())
    data = (s.folder / storage.PROJECT_FILE).read_bytes()  # references img-1.tif
    folder = s.folder
    assert storage.unsaved_references(folder) == storage.KeptReferences(frozenset(), ())
    (folder / "project.unsaved-a.json").write_bytes(b"[]")  # JSON, not a project
    (folder / "project.unsaved-b.json").write_bytes(b"not JSON")
    (folder / "project.unsaved-c.json").write_bytes(data)
    (folder / "project.unsaved-d.json").mkdir()  # a folder: no copy
    (folder / "project.schema1.json").write_bytes(b"[]")  # a backup: no copy
    found = storage.unsaved_references(folder)
    assert found == storage.KeptReferences(
        frozenset({"img-1.tif"}), ("project.unsaved-a.json", "project.unsaved-b.json")
    )
    assert storage.unsaved_references(tmp_path / "gone") is None


def test_a_kept_copy_and_what_copies_reference_are_values():
    kept = storage.KeptCopy(COPY, "0" * 64)
    found = storage.KeptReferences(frozenset({"img-1.tif"}), ())
    for value, field in ((kept, "name"), (found, "files")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, None)
    assert len({kept, storage.KeptCopy(COPY, "0" * 64), found}) == 2


def test_file_sha256_is_none_for_what_is_no_readable_file(tmp_path):
    path = tmp_path / "α.bin"
    path.write_bytes(b"bytes")
    assert storage.file_sha256(path) == hashlib.sha256(b"bytes").hexdigest()
    assert storage.file_sha256(tmp_path / "gone") is None
    assert storage.file_sha256(tmp_path) is None  # a folder
