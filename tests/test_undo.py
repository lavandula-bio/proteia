# SPDX-License-Identifier: Apache-2.0
"""Tests for undo, redo and clearing a protein's boxes (#52).

The session keeps each committed state of its content; undo and redo restore
one whole and log the restore as a change of its own. These tests walk every
logged operation back and forward, check the params, the linear model, the
refusals, the ids, the image files the history keeps, the pixel cache, the
autosave and the record's check. Helpers, the box blot and
:data:`~test_operations.LOGGED_STEPS` come from ``test_operations``.
"""

import dataclasses
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

import proteia
from conftest import FakeClock, make_project
from proteia.core import operations as ops
from proteia.core import session as session_module
from proteia.core import storage
from proteia.core.grow import grow_box
from proteia.core.model import (
    Batch,
    BoxSize,
    LogEntry,
    Project,
    ProposalSource,
    Role,
    UnknownIdError,
)
from proteia.core.operations import (
    ClearedBoxes,
    ErrorCode,
    LaneInput,
    OperationError,
    ProjectSession,
    Restored,
)
from proteia.core.record import history_issues
from proteia.core.session import HistoryStep, save_to_folder
from proteia.core.storage import content_hash, document_bytes, load_project
from test_operations import (
    LIGHT,
    LOGGED_STEPS,
    NARROW_X,
    ROW,
    ROWS,
    WIDE_X,
    Recorder,
    ReplaceLock,
    _keys,
    _record,
    _record_json,
    assert_nets_current,
    blot,
    boxed,
    detected,
    import_blot,
    lane_bands,
    listing,
    open_sample,
    placed_box,
    plant,
    plant_records,
    protein_of,
    row_session,
    session_on,
)


@pytest.fixture
def replace_lock(monkeypatch) -> ReplaceLock:
    return ReplaceLock(monkeypatch)


def ids_diff(before: Project, after: Project) -> tuple[list[str], list[str]]:
    """The ids only ``before`` has, then those only ``after`` has, in the order
    :meth:`~proteia.core.model.Project.iter_ids` gives."""
    old, new = list(before.iter_ids()), list(after.iter_ids())
    return [i for i in old if i not in new], [i for i in new if i not in old]


def record_keys(project: Project) -> list[list]:
    """Every not-detected record's key, as the undo params list it."""
    return [
        [p.id, u.lane_index, u.band_index] for p in project.batch.proteins for u in p.undetected
    ]


def restore_params(verb: str, step: LogEntry, back: int | None, before: Project, after: Project):
    """The params of an undo (``verb`` "undone") or redo ("redone") of ``step``
    that takes the content from ``before`` to ``after``."""
    removed, restored = ids_diff(before, after)
    old, new = record_keys(before), record_keys(after)
    return {
        f"{verb}_seq": step.seq,
        f"{verb}_action": step.action,
        "returns_to_seq": back,
        "removed": removed,
        "restored": restored,
        "undetected_removed": [key for key in old if key not in new],
        "undetected_restored": [key for key in new if key not in old],
    }


def logged_sample(tmp_path: Path, hook=save_to_folder) -> ProjectSession:
    """The conftest sample, its log beginning with its creation, saved and opened."""
    project = make_project()
    created = LogEntry(
        seq=1,
        time="2026-09-26T07:00:00.000Z",
        action="new_project",
        version=proteia.__version__,
        content_hash=content_hash(project),
    )
    return open_sample(tmp_path, hook, project.model_copy(update={"log": (created,)}))


class Unchanged:
    """What a refused undo or redo must leave as it was: the project (the same
    object), the log, ``next_id``, the history, the pixel cache and ``images/``."""

    def __init__(self, s: ProjectSession) -> None:
        self.s = s
        self.project, self.states, self.cursor = s.project, list(s._states), s._cursor
        self.steps = s.history_steps
        self.cache, self.files = dict(s._pixels), listing(s)

    def check(self) -> None:
        s = self.s
        assert s.project is self.project
        assert s.project.next_id == self.project.next_id
        assert s._states == self.states and s._cursor == self.cursor
        assert s.history_steps == self.steps
        assert s._pixels.keys() == self.cache.keys()
        assert all(s._pixels[key] is array for key, array in self.cache.items())
        assert listing(s) == self.files


# --- 1. every operation, back and forward ---


def test_every_logged_step_is_undone_to_the_empty_batch_and_redone(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    states = [s.project]  # states[k - 1] is the content log entry k left
    for call, _, _ in LOGGED_STEPS:
        call(s)
        states.append(s.project)
    forward, final, next_id = s.project.log, s.project, s.project.next_id
    n = len(LOGGED_STEPS)
    assert len(forward) == n + 1
    assert {"clear_boxes", "remove_undetected", "plant"} <= {entry.action for entry in forward}

    def check(action: str, verb: str, step: int, back: int, before: Project) -> None:
        entry, after = s.project.log[-1], states[back - 1]
        assert entry.action == action
        assert entry.params == restore_params(verb, forward[step - 1], back, before, after)
        # The restored content hashes like the entry it returns to.
        assert entry.content_hash == s.project.log[back - 1].content_hash == content_hash(s.project)
        assert s.project.batch == after.batch
        assert history_issues(s.project) == []
        assert load_project(s.folder) == s.project  # autosaved with its entry
        assert_nets_current(s)  # the stored nets are those of the pixels on disk
        assert s.project.next_id == next_id  # only rises; nothing restores it
        assert s.history_steps == s._steps_at_cursor()  # published as the history moved

    for k in range(n + 1, 1, -1):  # undo entry k: back to the content entry k - 1 left
        before = s.project
        done = ops.undo(s)
        assert (done.seq, done.action) == (k, forward[k - 1].action)
        check("undo", "undone", k, k - 1, before)
    assert s.project.batch == Batch()  # back to the new, empty project
    assert (s.undo_step, s.redo_step) == (None, HistoryStep(seq=2, action="import_image"))
    with pytest.raises(OperationError) as info:
        ops.undo(s)
    assert info.value.code is ErrorCode.NOTHING_TO_UNDO

    for k in range(2, n + 2):  # redo entry k: its own content again
        before = s.project
        done = ops.redo(s)
        assert (done.seq, done.action) == (k, forward[k - 1].action)
        check("redo", "redone", k, k, before)
    assert s.project.batch == final.batch
    assert s.redo_step is None
    assert s.undo_step == HistoryStep(seq=n + 1, action=forward[-1].action)
    # The log is append-only: every change, undone or not, and each undo and redo.
    log = s.project.log
    assert log[: n + 1] == forward
    assert [entry.action for entry in log[n + 1 :]] == ["undo"] * n + ["redo"] * n
    assert [entry.seq for entry in log] == list(range(1, 3 * n + 2))


# --- 2. exact params ---


def test_undo_of_a_removal_lists_the_cascade_it_brings_back(tmp_path):
    s = logged_sample(tmp_path)
    before = s.project
    cascade = ops.remove_image(s, "img-6")  # α-tubulin's image, with its membrane
    removal = s.project.log[-1]
    back = ("mem-5", "img-6", "prot-8", "band-13", "band-14", "band-15", "band-16")
    assert sorted(back) == sorted(cascade.removed)

    assert ops.undo(s) == Restored(
        seq=removal.seq,
        action="remove_image",
        removed=(),
        restored=back,  # by kind: membranes, images, proteins, bands
        undetected_removed=(),
        undetected_restored=(),
    )
    assert s.project.log[-1].params == {
        "undone_seq": 2,
        "undone_action": "remove_image",
        "returns_to_seq": 1,
        "removed": [],
        "restored": list(back),
        "undetected_removed": [],
        "undetected_restored": [],
    }
    # A field the removal changed comes back too, though no list names it.
    assert s.project.batch == before.batch
    assert s.project.batch.find_protein("prot-7").loading_control_ids == ["prot-8"]

    assert ops.redo(s) == Restored(
        seq=2,
        action="remove_image",
        removed=back,
        restored=(),
        undetected_removed=(),
        undetected_restored=(),
    )
    params = s.project.log[-1].params
    assert params["redone_seq"] == params["returns_to_seq"] == 2
    assert params["redone_action"] == "remove_image"


def test_undo_of_a_placement_removes_the_box_and_a_size_change_lists_nothing(tmp_path):
    s, _, protein = boxed(tmp_path)
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    placed = s.project.log[-1].seq
    ops.set_box_size(s, protein, BoxSize(width=12, height=6))
    resized = s.project.log[-1].seq

    restored = ops.undo(s)  # a box that moved and changed its net: in no list
    assert restored == Restored(resized, "set_box_size", (), (), (), ())
    assert s.project.log[-1].params["returns_to_seq"] == placed
    assert ops.undo(s) == Restored(placed, "place_box", (band,), (), (), ())
    assert s.project.log[-1].params == {
        "undone_seq": placed,
        "undone_action": "place_box",
        "returns_to_seq": placed - 1,
        "removed": [band],
        "restored": [],
        "undetected_removed": [],
        "undetected_restored": [],
    }
    assert ops.redo(s) == Restored(placed, "place_box", (), (band,), (), ())
    assert s.project.log[-1].params["returns_to_seq"] == placed


# --- 3. the linear model ---


def test_a_new_change_clears_redo_and_an_undo_never_undoes_an_undo(tmp_path):
    s, _, protein = boxed(tmp_path)
    assert s.undo_step == HistoryStep(seq=4, action="add_protein")
    a = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)  # seq 5
    ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=False)  # B, seq 6
    assert ops.undo(s).seq == 6  # takes back B
    # The undo is logged, but the next undo takes back the change before B.
    assert s.project.log[-1].action == "undo"
    assert (s.undo_step, s.redo_step) == (
        HistoryStep(seq=5, action="place_box"),
        HistoryStep(seq=6, action="place_box"),
    )

    c = ops.place_box(s, protein, 150, ROW, lane_index=2, grow=False)  # C, seq 8
    assert s.redo_step is None  # B can no longer be redone
    assert ops.undo(s).seq == 8  # takes back C
    assert ops.undo(s).seq == 5  # then A, not the undo of B
    assert [b.id for b in protein_of(s, protein).bands] == []
    assert s.redo_step == HistoryStep(seq=5, action="place_box")

    assert ops.redo(s).restored == (a,)
    assert ops.redo(s).restored == (c,)
    with pytest.raises(OperationError) as info:
        ops.redo(s)  # B is gone from the history
    assert info.value.code is ErrorCode.NOTHING_TO_REDO
    assert [b.lane_index for b in protein_of(s, protein).bands] == [0, 2]


def test_undo_walks_back_only_to_the_sessions_opening(tmp_path):
    s, _, protein = boxed(tmp_path, save_to_folder)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
    reopened = ops.open_project(s.folder, clock=FakeClock())
    assert (reopened.undo_step, reopened.redo_step) == (None, None)  # per session
    ops.set_reference_condition(reopened, "vehicle")
    ops.undo(reopened)
    assert reopened.project.log[-1].params["returns_to_seq"] == 5  # the place_box it opened at
    assert reopened.project.batch == s.project.batch
    with pytest.raises(OperationError) as info:
        ops.undo(reopened)
    assert info.value.code is ErrorCode.NOTHING_TO_UNDO


def test_a_session_opened_after_an_undo_returns_to_that_undo(tmp_path):
    s, _, protein = boxed(tmp_path, save_to_folder)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)  # seq 5
    ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=False)  # seq 6
    ops.undo(s)  # seq 7: the session ends on an undo
    reopened = ops.open_project(s.folder, clock=FakeClock())
    ops.set_reference_condition(reopened, "vehicle")  # seq 8
    assert ops.undo(reopened).seq == 8
    entry = reopened.project.log[-1]
    assert entry.params["returns_to_seq"] == 7  # the undo entry, which left this content
    assert entry.content_hash == reopened.project.log[6].content_hash
    assert reopened.project.batch == s.project.batch
    assert history_issues(reopened.project) == []
    assert history_issues(load_project(reopened.folder)) == []
    assert ops.redo(reopened).seq == 8
    assert history_issues(reopened.project) == []


# --- 4. refusals change nothing ---


def test_nothing_to_undo_or_redo_changes_nothing(tmp_path):
    recorder = Recorder()
    fresh = session_on(tmp_path / "new", recorder)
    opened = open_sample(tmp_path / "opened", recorder)
    s, _, _ = boxed(tmp_path / "edited", recorder)  # edits, none undone: nothing to redo
    recorder.actions.clear()
    for session, calls in [
        (fresh, [(ops.undo, ErrorCode.NOTHING_TO_UNDO), (ops.redo, ErrorCode.NOTHING_TO_REDO)]),
        (opened, [(ops.undo, ErrorCode.NOTHING_TO_UNDO), (ops.redo, ErrorCode.NOTHING_TO_REDO)]),
        (s, [(ops.redo, ErrorCode.NOTHING_TO_REDO)]),
    ]:
        for call, code in calls:
            before = Unchanged(session)
            with pytest.raises(OperationError) as info:
                call(session)
            assert info.value.code is code
            assert str(info.value) == f"nothing to {code.value.removeprefix('nothing_to_')}"
            before.check()
    assert recorder.actions == []


def test_undo_to_a_state_whose_image_file_is_missing_is_refused(tmp_path):
    recorder = Recorder()
    s = session_on(tmp_path, recorder)
    image = import_blot(s, blot())
    other = import_blot(s, blot(), "GAPDH β.tif")
    file = s.folder / "images" / f"{image}.tif"
    ops.remove_image(s, image)
    data = file.read_bytes()
    file.unlink()  # deleted by hand
    recorder.actions.clear()
    before = Unchanged(s)

    with pytest.raises(OperationError) as info:
        ops.undo(s)
    assert (info.value.code, info.value.ids) == (ErrorCode.IMAGE_FILE_CHANGED, (image,))
    assert image in str(info.value)
    before.check()
    assert recorder.actions == []

    file.write_bytes(data)  # put back: undo works again
    ops.undo(s)
    assert [i.id for i in s.project.batch.iter_images()] == [image, other]
    assert recorder.actions == ["undo"]


# --- 5. ids are never reused ---


def test_ids_after_an_undo_are_new_ones(tmp_path):
    s, _, protein = boxed(tmp_path, save_to_folder)
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
    next_id = s.project.next_id
    ops.undo(s)
    assert s.project.next_id == next_id
    again = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
    assert int(again.removeprefix("band-")) > int(band.removeprefix("band-"))

    image = import_blot(s, blot(), "second α.tif")
    ops.undo(s)  # its file stays for a redo
    assert (s.folder / "images" / f"{image}.tif").exists()
    newer = import_blot(s, blot(), "third β.tif")
    assert int(newer.removeprefix("img-")) > int(image.removeprefix("img-"))
    assert listing(s) == sorted(["img-1.tif", f"{newer}.tif"])  # the redo was cleared


# --- 6. the files the history keeps ---


def test_a_removed_images_file_lives_while_the_removal_can_be_undone(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    image = import_blot(s, blot())
    file = s.folder / "images" / f"{image}.tif"
    ops.remove_image(s, image)
    assert not s.dirty and file.exists()  # survives the autosave
    assert list(load_project(s.folder).batch.iter_images()) == []  # saved without it

    ops.undo(s)
    assert load_project(s.folder) == s.project  # undo, save, load: equal
    assert_nets_current(s)
    ops.redo(s)
    assert file.exists()  # the redone removal can be undone again
    s.close()
    assert not file.exists()  # nothing can bring it back
    assert (s.undo_step, s.redo_step) == (None, None)
    assert listing(s) == []
    s.close()  # closing twice is harmless


def test_a_file_is_deleted_once_the_undo_limit_drops_every_state_with_it(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "UNDO_LIMIT", 2)
    s = session_on(tmp_path, save_to_folder)
    image = import_blot(s, blot())
    file = s.folder / "images" / f"{image}.tif"
    ops.remove_image(s, image)
    ops.set_lanes(s, [LaneInput("vehicle")])  # the import's state is still kept
    assert file.exists()
    assert len(s._states) == 3  # the limit's steps, plus the state they start from
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])  # it is dropped
    assert not file.exists()
    ops.undo(s)
    ops.undo(s)
    with pytest.raises(OperationError) as info:
        ops.undo(s)  # the removal is past the limit
    assert info.value.code is ErrorCode.NOTHING_TO_UNDO


def test_a_file_only_an_old_sessions_history_kept_goes_at_the_next_first_save(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    image = import_blot(s, blot())
    file = s.folder / "images" / f"{image}.tif"
    ops.remove_image(s, image)
    folder = s.folder
    del s  # never closed, as after a crash

    reopened = ops.open_project(folder, clock=FakeClock())
    assert file.exists()  # opening deletes nothing
    ops.set_lanes(reopened, [LaneInput("vehicle")])
    assert not file.exists()


def test_an_undone_imports_file_goes_when_a_new_edit_clears_redo(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    image = import_blot(s, blot())
    file = s.folder / "images" / f"{image}.tif"
    ops.undo(s)
    assert file.exists() and s.redo_step == HistoryStep(seq=2, action="import_image")
    ops.set_lanes(s, [LaneInput("vehicle")])  # made at the state the session began with
    assert s.redo_step is None and not file.exists()
    assert s.undo_step == HistoryStep(seq=4, action="set_lanes")
    ops.undo(s)
    assert s.undo_step is None  # the undo of the import is never undone
    with pytest.raises(OperationError) as info:
        ops.undo(s)
    assert info.value.code is ErrorCode.NOTHING_TO_UNDO


def test_an_import_keeps_the_file_of_a_removal_that_can_be_undone(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    image = import_blot(s, blot())
    file = s.folder / "images" / f"{image}.tif"
    ops.remove_image(s, image)
    other = import_blot(s, blot(), "GAPDH β.tif")  # removes orphans before it stores
    assert (image, other) == ("img-1", "img-3")
    assert file.exists()
    assert ops.undo(s).removed == ("mem-4", other)  # the import
    assert ops.undo(s).restored == ("mem-2", image)  # the removal: its file is there
    assert [i.id for i in s.project.batch.iter_images()] == [image]
    np.testing.assert_array_equal(s.pixels(image), blot().astype(np.float64))
    assert ops.redo(s).removed == ("mem-2", image)
    assert ops.redo(s).restored == ("mem-4", other)
    assert listing(s) == [f"{image}.tif", f"{other}.tif"]  # undo can still bring img-1 back


# --- 7. the pixel cache ---


def test_undo_evicts_pixels_and_a_restored_image_is_read_and_checked_again(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    pixels = blot()
    image = import_blot(s, pixels)
    assert image in s._pixels
    ops.undo(s)  # of the import
    assert image not in s._pixels
    ops.redo(s)
    assert image not in s._pixels  # read lazily, when used
    np.testing.assert_array_equal(s.pixels(image), pixels.astype(np.float64))

    ops.remove_image(s, image)
    ops.undo(s)
    s.pixels(image)  # cached again
    ops.redo(s)  # the removal again: its pixels go with it
    assert image not in s._pixels
    file = s.folder / "images" / f"{image}.tif"
    data = bytearray(file.read_bytes())
    data[-1] ^= 0xFF
    file.write_bytes(data)  # changed by hand
    ops.undo(s)  # the file exists: the undo only checks that
    assert image not in s._pixels
    with pytest.raises(OperationError) as info:
        s.pixels(image)
    assert (info.value.code, info.value.ids) == (ErrorCode.IMAGE_FILE_CHANGED, (image,))
    ops.set_lanes(s, [LaneInput("vehicle")])
    with pytest.raises(OperationError) as info:
        ops.export_lane_table(s)
    assert (info.value.code, info.value.ids) == (ErrorCode.IMAGE_FILE_CHANGED, (image,))


# --- 8. a failed autosave ---


def test_an_undo_whose_autosave_fails_stays_in_memory(tmp_path, replace_lock):
    s, _, protein = boxed(tmp_path, save_to_folder)
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    path = s.folder / storage.PROJECT_FILE
    before = path.read_bytes()

    replace_lock.locked = True
    ops.undo(s)  # returns normally
    assert s.dirty and isinstance(s.save_error, PermissionError)
    assert path.read_bytes() == before
    assert protein_of(s, protein).bands == []
    assert s.redo_step is not None  # the history moved with the commit

    replace_lock.locked = False
    ops.save(s)
    assert load_project(s.folder) == s.project
    assert s.project.log[-1].params["removed"] == [band]


# --- 9. content integrity ---


def test_a_corrupt_state_raises_before_anything_is_committed(tmp_path):
    recorder = Recorder()
    s, _, protein = boxed(tmp_path, recorder)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
    ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=False)
    target = s._states[s._cursor - 1]
    recorder.actions.clear()
    # Valid content, but not the state's own: its hash gives it away.
    other = s._states[s._cursor - 2].content
    for content in (other, zlib.compress(b"not a project"), b"not zlib"):
        s._states[s._cursor - 1] = dataclasses.replace(target, content=content)
        before = Unchanged(s)
        with pytest.raises(RuntimeError):
            ops.undo(s)
        before.check()
    assert recorder.actions == []
    s._states[s._cursor - 1] = target
    ops.undo(s)
    assert recorder.actions == ["undo"]


def test_a_history_out_of_step_with_the_project_raises(tmp_path):
    s, _, protein = boxed(tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
    s._states[s._cursor] = dataclasses.replace(s._states[s._cursor], hash="0" * 64)
    before = Unchanged(s)
    with pytest.raises(RuntimeError, match="does not match"):
        ops.undo(s)
    before.check()


# --- 10. clearing a protein's boxes ---


def test_clear_boxes_removes_every_box_and_record_in_one_change(tmp_path):
    s, image, protein = boxed(tmp_path, save_to_folder)
    wide = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
    narrow = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    lane_2 = _record(2)
    plant_records(s, protein, lane_2)
    before, length = s.project, len(s.project.log)
    kept = protein_of(s, protein).box_size

    cleared = ops.clear_boxes(s, protein)
    assert cleared == ClearedBoxes(band_ids=(narrow, wide), undetected=((2, 0),))  # lane order
    assert len(s.project.log) == length + 1
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "clear_boxes",
        {
            "protein_id": protein,
            "removed": [narrow, wide],
            "lane_indices": [0, 1],
            "dropped_undetected": [_record_json(protein, lane_2)],
        },
    )
    assert entry.content_hash == content_hash(s.project)
    cleared_protein = protein_of(s, protein)
    assert (cleared_protein.bands, cleared_protein.undetected) == ([], [])
    assert cleared_protein.box_size == kept  # the width and height fields show it
    assert load_project(s.folder) == s.project

    # Undo restores the boxes, their nets and flags, and the record exactly.
    assert ops.undo(s) == Restored(
        seq=entry.seq,
        action="clear_boxes",
        removed=(),
        restored=(narrow, wide),  # as stored, by lane: not in id order
        undetected_removed=(),
        undetected_restored=((protein, 2, 0),),
    )
    assert s.project.batch == before.batch
    assert_nets_current(s)

    # Cleared again: a seed click sets the size from its band alone.
    ops.redo(s)
    grown = grow_box(s.pixels(image), (NARROW_X, ROW), s.project.batch.find_image(image).background)
    assert grown is not None
    alone = BoxSize(width=grown[2] - grown[0], height=grown[3] - grown[1])
    assert alone.width < kept.width  # the kept size was grown by the wide band
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert protein_of(s, protein).box_size == alone
    # A fixed box instead uses the kept size.
    ops.undo(s)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
    assert protein_of(s, protein).box_size == kept
    assert_nets_current(s)


def test_a_row_after_clear_boxes_sets_the_size_from_its_bands_alone(tmp_path):
    case = ROWS["all_present"]
    s, _, protein = row_session(tmp_path, case)
    found = detected(s, protein, case.row)
    placed_box(s, protein, case, 0, ProposalSource.MANUAL)
    kept = BoxSize(width=found.size.width + 7, height=found.size.height + 5)
    ops.set_box_size(s, protein, kept)
    ops.clear_boxes(s, protein)
    assert protein_of(s, protein).box_size == kept
    ops.detect_row_boxes(s, protein, case.row)
    assert protein_of(s, protein).box_size == found.size  # not grown to the kept size
    assert_nets_current(s)


def test_clear_boxes_of_records_only_of_nothing_or_of_an_unknown_protein(tmp_path):
    recorder = Recorder()
    s, image, protein = boxed(tmp_path, recorder)
    plant_records(s, protein, _record(1, snr=-0.25), _record(1, band_index=1), bands=2)
    cleared = ClearedBoxes(band_ids=(), undetected=((1, 0), (1, 1)))  # every band index
    assert ops.clear_boxes(s, protein) == cleared
    assert _keys(s, protein) == []
    assert s.project.log[-1].params["removed"] == []
    restored = ops.undo(s)
    assert restored.undetected_restored == ((protein, 1, 0), (protein, 1, 1))
    assert _keys(s, protein) == [(1, 0), (1, 1)]
    ops.redo(s)
    assert s.project.log[-1].params["undetected_removed"] == [[protein, 1, 0], [protein, 1, 1]]

    recorder.actions.clear()
    committed = s.project
    assert ops.clear_boxes(s, protein) == ClearedBoxes(band_ids=(), undetected=())  # a no-op
    with pytest.raises(UnknownIdError):
        ops.clear_boxes(s, "prot-999")
    assert s.project is committed and recorder.actions == []

    other = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    kept = ops.place_box(s, other, WIDE_X, ROW, lane_index=0, grow=False)
    plant_records(s, other, _record(2))
    first = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
    second = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=False)
    plant(s, lambda draft: setattr(draft.batch.find_band(second)[1], "band_index", 1))
    assert ops.clear_boxes(s, protein).band_ids == (first, second)  # every band index
    assert protein_of(s, protein).bands == []
    # Another protein's boxes and records stay.
    assert [b.id for b in protein_of(s, other).bands] == [kept]
    assert _keys(s, other) == [(2, 0)]


# --- 11. not-detected records under undo ---


def test_undo_brings_back_the_records_a_change_dropped(tmp_path):
    s, image, protein = boxed(tmp_path, save_to_folder)
    plant_records(s, protein, _record(1), _record(2, snr=-0.5))
    records = protein_of(s, protein).undetected

    band = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)  # replaces lane 1's
    placed = s.project.log[-1].seq
    assert ops.undo(s) == Restored(placed, "place_box", (band,), (), (), ((protein, 1, 0),))
    assert protein_of(s, protein).undetected == records

    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])  # cuts lane 2
    assert _keys(s, protein) == [(1, 0)]
    restored = ops.undo(s)
    assert (restored.action, restored.undetected_restored) == ("set_lanes", ((protein, 2, 0),))
    assert protein_of(s, protein).undetected == records
    assert len(s.project.batch.lanes) == 3

    ops.set_polarity(s, image, LIGHT)  # drops every record on the image
    assert _keys(s, protein) == []
    restored = ops.undo(s)
    assert restored.undetected_restored == ((protein, 1, 0), (protein, 2, 0))
    assert protein_of(s, protein).undetected == records
    ops.redo(s)
    assert s.project.log[-1].params["undetected_removed"] == [[protein, 1, 0], [protein, 2, 0]]
    assert load_project(s.folder) == s.project


def test_undo_of_a_row_restores_the_boxes_and_records_before_it(tmp_path):
    case = ROWS["missing_two"]  # lanes 0 and 3 are empty
    s, _, protein = row_session(tmp_path, case, hook=save_to_folder)
    manual = placed_box(s, protein, case, 1, ProposalSource.MANUAL)
    guided = _record(5, source=ProposalSource.MW_GUIDED, region=(0, 0, 20, 20))
    plant_records(s, protein, guided)
    before = protein_of(s, protein)

    placement = ops.detect_row_boxes(s, protein, case.row)
    row_seq = s.project.log[-1].seq
    after = protein_of(s, protein)
    assert manual in placement.replaced_band_ids  # replaced in place: the same id
    assert _keys(s, protein) == [(0, 0), (3, 0)]

    new = tuple(lane_bands(s, protein)[lane].id for lane in (2, 4, 5))
    assert ops.undo(s) == Restored(
        seq=row_seq,
        action="detect_row_boxes",
        removed=new,
        restored=(),  # the manual box came back under its own id: a changed field
        undetected_removed=((protein, 0, 0), (protein, 3, 0)),
        undetected_restored=((protein, 5, 0),),
    )
    assert protein_of(s, protein) == before
    assert_nets_current(s)
    ops.redo(s)
    assert protein_of(s, protein) == after
    assert_nets_current(s)


# --- 12. the record ---


def test_undo_to_content_the_log_never_recorded_is_flagged(tmp_path):
    s, _, _ = boxed(tmp_path, save_to_folder)
    path = s.folder / storage.PROJECT_FILE
    doc = json.loads(path.read_bytes())
    doc["batch"]["lanes"][0]["label"] = "DMSO"  # edited by hand, outside the log
    path.write_bytes(document_bytes(doc))

    opened = ops.open_project(s.folder, clock=FakeClock())
    assert history_issues(opened.project) == ["content_changed_outside_log"]
    ops.set_reference_condition(opened, "DMSO")
    assert history_issues(opened.project) == []  # the last entry vouches for the content
    ops.undo(opened)
    assert opened.project.log[-1].params["returns_to_seq"] is None
    assert history_issues(opened.project) == ["undo_mismatch"]
    assert load_project(s.folder).batch.lanes[0].label == "DMSO"


def test_a_clean_history_has_no_undo_mismatch(tmp_path):
    s, _, protein = boxed(tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
    ops.undo(s)
    ops.redo(s)
    ops.undo(s)
    assert history_issues(s.project) == []


def test_a_project_saved_before_not_detected_records_seeds_with_its_last_entry(tmp_path):
    # The content hash a build without not-detected records stored for the
    # sample project (test_storage and test_record pin the same value).
    entry = LogEntry(
        seq=1,
        time="2026-09-20T08:00:00.000Z",
        action="new_project",
        version="0.1.0.dev0",
        content_hash="28dfdbcbca235bb7359164954bf76a6d743a2c4448e51049172f42cc00b3df6a",
    )
    s = open_sample(tmp_path, project=make_project().model_copy(update={"log": (entry,)}))
    ops.set_reference_condition(s, None)
    ops.undo(s)
    assert s.project.log[-1].params["returns_to_seq"] == 1
    assert s.project.log[-1].content_hash == entry.content_hash
    assert history_issues(s.project) == []
