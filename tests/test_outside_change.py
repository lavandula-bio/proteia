# SPDX-License-Identifier: Apache-2.0
"""A ``project.json`` changed outside Proteia while its project is open (#101).

A session saves only over the ``project.json`` it last read or wrote, compared
by the SHA-256 of its bytes: a file replaced (a synced or restored copy),
edited or removed outside Proteia since is not saved over, the change stays in
memory as after any failed save, and no image file is deleted as an orphan
while the session cannot tell what the file on disk references. Opening the
project again reads the version on disk.

The outside version is made as a sync would make it: a copy of the folder,
changed by a session of its own (importing an image only it references), is
copied back over the open folder.
"""

import codecs
import hashlib
import logging
import os
import shutil
import threading
from pathlib import Path

import pytest

from conftest import FakeClock
from proteia.core import operations as ops
from proteia.core import storage
from proteia.core.model import BoxSize
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

# The messages, word for word: the folder by name (test_operations.FOLDER), never its path.
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


def temp_files(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir() if path.name.endswith(".tmp"))


# --- the refusal ---


def test_a_save_never_goes_over_a_project_json_changed_outside(tmp_path):
    s, image, protein = boxed(tmp_path, save_to_folder)
    _, newer = changed_outside(s, tmp_path)
    path = s.folder / storage.PROJECT_FILE
    outside = path.read_bytes()

    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)  # returns normally
    assert band_of(s, band).lane_index == 0  # the edit is in memory
    assert s.dirty and isinstance(s.save_error, storage.ChangedOutsideError)
    assert isinstance(s.save_error, storage.ProjectError)  # handled as any failed save
    assert str(s.save_error) == CHANGED and not s.save_error.removed
    assert str(tmp_path) not in str(s.save_error)
    assert path.read_bytes() == outside  # the outside version is kept
    # The cleanup after the save did not run: the image only it references is kept.
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])
    assert s.undo_step is not None and s.undo_step.action == "place_box"

    # The next change is refused too, and an explicit save says why.
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert s.dirty and str(s.save_error) == CHANGED
    with pytest.raises(storage.ChangedOutsideError, match="did not save over it"):
        ops.save(s)
    assert path.read_bytes() == outside
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])


def test_the_page_gets_the_refusal_as_a_code_with_its_message(tmp_path, monkeypatch):
    s, _, protein = boxed(tmp_path, save_to_folder)
    state = project_state(s.folder.name, s, s.project, open_id=1)
    assert (state["saved"], state["save_error"], state["save_error_code"]) == (True, None, None)

    # Another failed save has a message and no code of its own.
    lock = ReplaceLock(monkeypatch)
    lock.locked = True
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    state = project_state(s.folder.name, s, s.project, open_id=1)
    assert state["saved"] is False and state["save_error"]
    assert state["save_error_code"] is None
    lock.locked = False

    changed_outside(s, tmp_path)
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    state = project_state(s.folder.name, s, s.project, open_id=1)
    assert (state["saved"], state["save_error"], state["save_error_code"]) == (
        False,
        CHANGED,
        "project_changed_outside",
    )


def test_a_removed_project_json_is_not_written_again(tmp_path):
    s, _, protein = boxed(tmp_path, save_to_folder)
    path = s.folder / storage.PROJECT_FILE
    saved = path.read_bytes()
    path.unlink()

    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert s.dirty and isinstance(s.save_error, storage.ChangedOutsideError)
    assert str(s.save_error) == REMOVED and s.save_error.removed
    assert not path.exists()

    # The file Proteia last saved, put back: the next change saves both edits.
    path.write_bytes(saved)
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert (s.dirty, s.save_error) == (False, None)
    assert load_project(s.folder) == s.project
    assert load_project(s.folder).batch.find_band(band)[1].lane_index == 0


def test_a_removed_project_folder_is_not_made_again(tmp_path):
    s = session_on(tmp_path, save_to_folder)  # no image: nothing else refuses the save
    shutil.rmtree(s.folder)
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert str(s.save_error) == REMOVED
    assert not s.folder.exists()  # neither project.json nor images/ or exports/


def test_a_file_replaced_while_the_save_writes_is_kept(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder)
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
    assert replaced and str(s.save_error) == CHANGED
    assert path.read_bytes() == outside
    assert temp_files(s.folder) == []


def test_a_file_replaced_while_a_replace_is_retried_is_kept(tmp_path, monkeypatch):
    s = session_on(tmp_path, save_to_folder)
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
    assert tries and str(s.save_error) == CHANGED
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
    # its save fails as a file error, kept in memory.
    newer = import_blot(s, blot(), "late γ.tif")
    assert s.project.batch.find_image(newer)
    assert s.dirty and isinstance(s.save_error, OSError)
    assert not isinstance(s.save_error, storage.ChangedOutsideError)
    assert leftover.exists()
    assert listing(s) == sorted([leftover.name, f"{image}.tif", f"{newer}.tif"])


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
    opened = ops.open_project(s.folder, autosave=save_to_folder, clock=FakeClock())
    monkeypatch.undo()
    assert [lane.label for lane in opened.project.batch.lanes] == ["vehicle"]
    ops.set_lanes(opened, [LaneInput("dose")])
    assert str(opened.save_error) == CHANGED and path.read_bytes() == outside


def test_a_reload_holds_the_bytes_it_read_and_saves_over_them_only(tmp_path, monkeypatch, caplog):
    s = session_on(tmp_path, save_to_folder)
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
    assert str(s.save_error) == CHANGED and path.read_bytes() == outside


# --- opening the project again ---


def test_opening_again_after_the_refusal_reads_the_version_on_disk(tmp_path, caplog):
    s, image, protein = boxed(tmp_path, save_to_folder)
    remote, newer = changed_outside(s, tmp_path)
    path = s.folder / storage.PROJECT_FILE
    # The synced copy as a Windows editor left it: other bytes, the same project.
    path.write_bytes(codecs.BOM_UTF8 + path.read_bytes().replace(b"\n", b"\r\n"))
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert str(s.save_error) == CHANGED

    caplog.set_level(logging.INFO, logger="proteia.core.session")
    caplog.clear()
    assert s.reload()  # the unsaved change goes; the version on disk is read
    assert [(r.levelname, r.getMessage()) for r in caplog.records][0] == (
        "WARNING",
        "in '專案 µ α β': dropped the changes that could not be saved, as its project.json"
        " was changed outside Proteia",
    )
    assert s.project == remote.project
    assert (s.dirty, s.save_error, s.undo_step, s.last_action) == (False, None, None, None)
    ops.set_polarity(s, newer, LIGHT)  # the next change saves over the file read
    assert (s.dirty, s.save_error) == (False, None)
    assert load_project(s.folder) == s.project
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])
    assert s.project.batch.find_image(newer).polarity == LIGHT
    assert not s.reload()


def test_opening_again_reads_even_a_file_that_holds_the_unsaved_project(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    path = s.folder / storage.PROJECT_FILE
    path.write_bytes(b"{}")  # changed outside Proteia
    ops.set_lanes(s, [LaneInput("vehicle")])
    assert str(s.save_error) == CHANGED
    # Then the file is made to hold exactly what saving the session would write.
    path.write_bytes(storage.project_to_json(s.project))

    assert s.reload()  # not the file Proteia saved: read, so the next change saves
    assert (s.dirty, s.save_error) == (False, None)
    ops.set_lanes(s, [LaneInput("drug")])
    assert (s.dirty, s.save_error) == (False, None)
    assert load_project(s.folder) == s.project


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
    assert str(info.value) == CHANGED
    assert s.project is project and not s.dirty
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])
    assert (s.folder / storage.PROJECT_FILE).read_bytes() == before


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
