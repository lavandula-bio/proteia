# SPDX-License-Identifier: Apache-2.0
"""Tests for the project operations and the session they work on.

No napari and no Qt: every test drives :mod:`proteia.core.operations` directly,
in project folders with non-ASCII names, on real TIFF files written by
``conftest.write_tiff``. The box tests use a 16-bit ``synthetic_blot`` with a
narrow and a wide band in one row; the parity tests at the end pin the whole
path, from an imported file to the chart, to the golden numbers of
``test_regression_baseline``. The row-box tests at the end commit rows of
:mod:`rowcases`.
"""

import codecs
import csv
import dataclasses
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import proteia
from conftest import (
    FakeClock,
    make_project,
    make_project_with_undetected,
    synthetic_blot,
    write_image_files,
    write_tiff,
)
from proteia.core import boxes, record, results, rowdetect, storage
from proteia.core import operations as ops
from proteia.core.analyze import ReduceMethod
from proteia.core.grow import grow_box
from proteia.core.imaging import clipping_depth
from proteia.core.model import (
    Band,
    Box,
    BoxSize,
    ImageKind,
    Polarity,
    Project,
    ProposalSource,
    Region,
    Role,
    UndetectedBand,
    UndetectedReason,
    UnknownIdError,
    apply_change,
    revalidate,
)
from proteia.core.operations import (
    Cascade,
    ErrorCode,
    LaneInput,
    LanesUpdate,
    OperationError,
    ProjectSession,
)
from proteia.core.plotspec import ErrorType
from proteia.core.project import lane_anchors, lane_positions
from proteia.core.quantify import estimate_background, is_clipped, net_signal
from proteia.core.record import history_issues
from proteia.core.results import Level, NoticeCode
from proteia.core.rowdetect import DETECT_K, RowDetection
from proteia.core.session import save_to_folder
from proteia.core.storage import canonical_json, content_hash, load_project
from rowcases import RowCase, adversarial, adversarial_row, bench_cases, synthetic_row
from test_regression_baseline import (
    BOX_SIZE,
    CONDITIONS,
    GOLDEN,
    INCLUDED,
    LANE_X,
    LOADING,
    LOW,
    REFERENCE,
    ROW_Y,
    SAMPLES,
    TARGET,
    _assert_close,
    _blot,
)
from test_results import _chart_stats, _golden_chart_stats

FOLDER = "專案 µ α β"
MICRO, MU = "\N{MICRO SIGN}", "\N{GREEK SMALL LETTER MU}"  # look-alikes
ZWSP = "\N{ZERO WIDTH SPACE}"  # a format (Cf) character
WIDE_SPACE = "\N{IDEOGRAPHIC SPACE}"
CHEMI = ImageKind.CHEMILUMINESCENCE
DARK, LIGHT = Polarity.DARK_ON_LIGHT, Polarity.LIGHT_ON_DARK

# The box tests' blot: one row with a narrow and a wide dark band.
H, W = 60, 160
ROW, NARROW_X, WIDE_X = 30, 40, 100
BACKGROUND_POINT = (5, 5)


def blot() -> np.ndarray:
    return synthetic_blot(
        (H, W), [(NARROW_X, ROW, 4.0, 2.5, 30000.0), (WIDE_X, ROW, 9.0, 2.5, 30000.0)]
    )


# --- helpers ---


class Recorder:
    """An autosave hook that records ``last_action`` instead of saving."""

    def __init__(self) -> None:
        self.actions: list[str | None] = []

    def __call__(self, session: ProjectSession) -> None:
        self.actions.append(session.last_action)


def session_on(tmp_path: Path, hook=None, clock=None) -> ProjectSession:
    """A new project in a non-ASCII folder; ``hook`` None means no autosave.
    The clock is a fresh :class:`FakeClock` unless one is given."""
    return ops.new_project(tmp_path / FOLDER, autosave=hook, clock=clock or FakeClock())


def import_blot(
    session: ProjectSession,
    pixels: np.ndarray,
    name: str = "β-actin 10 µM.tif",
    polarity: Polarity = DARK,
    *,
    kind: ImageKind = CHEMI,
    membrane_id: str | None = None,
) -> str:
    """Write ``pixels`` as a TIFF next to the project folder, then import it."""
    source = write_tiff(session.folder.parent / "sources" / name, pixels)
    with source.open("rb") as f:
        return ops.import_image(
            session, f, name, kind=kind, polarity=polarity, membrane_id=membrane_id
        )


def assert_nets_current(session: ProjectSession) -> None:
    """Every stored net and clipping flag equals its value recomputed from the
    session's pixels."""
    batch = session.project.batch
    for protein in batch.proteins:
        image = batch.find_image(protein.image_id)
        pixels = session.pixels(image.id)
        dark = image.polarity.dark_on_light
        for band in protein.bands:
            expected = net_signal(
                pixels, band.box, protein.box_size, image.background, dark_on_light=dark
            )
            assert band.net == expected, band.id
            depth = clipping_depth(image.bit_depth, image.import_warnings)
            clipped = is_clipped(
                pixels, band.box, protein.box_size, bit_depth=depth, dark_on_light=dark
            )
            assert band.clipped is clipped, band.id


def open_sample(
    tmp_path: Path, hook=save_to_folder, project: Project | None = None
) -> ProjectSession:
    """The conftest sample project (or ``project``) saved with stand-in image
    files, then opened."""
    folder = tmp_path / FOLDER
    project = make_project() if project is None else project
    write_image_files(folder, project)
    storage.save_project(project, folder)
    return ops.open_project(folder, autosave=hook, clock=FakeClock())


def boxed(tmp_path: Path, hook=None) -> tuple[ProjectSession, str, str]:
    """A session with the blot imported, three lanes and one target without boxes."""
    session = session_on(tmp_path, hook)
    image = import_blot(session, blot())
    ops.set_lanes(session, [LaneInput("vehicle"), LaneInput("10 µM"), LaneInput("50 µM")])
    protein = ops.add_protein(session, "β-catenin", Role.TARGET, image)
    return session, image, protein


def listing(session: ProjectSession) -> list[str]:
    return sorted(p.name for p in (session.folder / storage.IMAGES_DIR).iterdir())


def band_of(session: ProjectSession, band_id: str):
    return session.project.batch.find_band(band_id)[1]


def protein_of(session: ProjectSession, protein_id: str):
    return session.project.batch.find_protein(protein_id)


def _listed(cascade: Cascade) -> dict[str, list[str]]:
    """A cascade as its log entry records it: every field a list of ids."""
    return {name: list(ids) for name, ids in dataclasses.asdict(cascade).items()}


def plant(session: ProjectSession, change) -> None:
    """Commit a change no operation makes yet (e.g. an apparent MW from #58)."""
    project, _ = apply_change(session.project, change)
    session._commit(project, action="plant", params={})


class ReplaceLock:
    """While ``locked``, replacing project.json fails like a Windows sharing
    violation; storing images still works."""

    def __init__(self, monkeypatch) -> None:
        self.locked = False
        real_replace = os.replace

        def replace(src, dst, **kwargs):
            if self.locked and Path(dst).name == storage.PROJECT_FILE:
                raise PermissionError(13, "held by another process", str(dst))
            real_replace(src, dst, **kwargs)

        monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
        monkeypatch.setattr(storage.os, "replace", replace)


@pytest.fixture
def replace_lock(monkeypatch) -> ReplaceLock:
    return ReplaceLock(monkeypatch)


# --- lifecycle ---


def test_new_project_needs_an_empty_folder(tmp_path):
    folder = tmp_path / FOLDER
    session = ops.new_project(folder, clock=FakeClock())
    assert sorted(p.name for p in folder.iterdir()) == ["exports", "images", "project.json"]
    # Empty content, and a log that starts with the creation.
    [created] = session.project.log
    assert (created.seq, created.action, created.params) == (1, "new_project", {})
    assert created.time == "2026-09-26T08:00:00.000Z"
    assert created.version == proteia.__version__
    assert created.content_hash == content_hash(Project()) == content_hash(session.project)
    assert session.project.model_copy(update={"log": ()}) == Project()
    assert load_project(folder) == session.project
    assert (session.dirty, session.save_error, session.last_action) == (False, None, None)
    assert session.folder == folder

    with pytest.raises(OperationError) as info:
        ops.new_project(folder)  # holds project.json
    assert info.value.code is ErrorCode.FOLDER_NOT_EMPTY
    notes = tmp_path / "筆記 α"
    notes.mkdir()
    (notes / "readme β.txt").write_text("keep", encoding="utf-8")
    for occupied in (notes, notes / "readme β.txt"):
        with pytest.raises(OperationError) as info:
            ops.new_project(occupied)
        assert info.value.code is ErrorCode.FOLDER_NOT_EMPTY
    empty = tmp_path / "空的 µ"
    empty.mkdir()
    assert content_hash(ops.new_project(empty).project) == content_hash(Project())


def test_open_project_reads_no_pixels_and_changes_nothing(tmp_path):
    folder = tmp_path / FOLDER
    project = make_project()
    write_image_files(folder, project)
    storage.save_project(project, folder)
    orphan = folder / "images" / "img-99.tif"
    orphan.write_bytes(b"an import that was never saved")
    before = (folder / "project.json").read_bytes()

    session = ops.open_project(folder)
    assert session.project == project
    assert not session.dirty
    assert ops.compute(session).series  # results need no pixels
    assert session._pixels == {}
    assert orphan.exists()
    assert (folder / "project.json").read_bytes() == before


def test_autosave_hook_runs_once_after_each_change(tmp_path):
    recorder = Recorder()
    s = session_on(tmp_path, recorder)
    image = import_blot(s, blot())
    table = [LaneInput("vehicle"), LaneInput("10 µM"), LaneInput("10 µM")]
    ops.set_lanes(s, table)
    ops.set_lanes(s, table)  # identical: a no-op
    ops.set_reference_condition(s, "vehicle")
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    ops.edit_protein(s, protein, expected_mw=92)
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    with pytest.raises(OperationError):
        ops.place_box(s, protein, W, 0, lane_index=1, grow=False)  # refused: outside
    ops.move_box(s, band, (50, 20, 70, 40))
    ops.move_box(s, band, band_of(s, band).box.rect(protein_of(s, protein).box_size))  # no-op
    ops.set_box_size(s, protein, BoxSize(width=12, height=6))
    ops.set_box_size(s, protein, BoxSize(width=12, height=6))  # no-op
    ops.set_polarity(s, image, LIGHT)
    ops.set_polarity(s, image, LIGHT)  # no-op
    ops.compute(s)
    ops.export_lane_table(s)
    ops.remove_box(s, band)
    ops.remove_protein(s, protein)
    ops.remove_image(s, image)
    assert recorder.actions == [
        "import_image",
        "set_lanes",
        "set_reference_condition",
        "add_protein",
        "edit_protein",
        "place_box",
        "move_box",
        "set_box_size",
        "set_polarity",
        "remove_box",
        "remove_protein",
        "remove_image",
    ]
    # One entry per commit: the no-ops, the refusal, compute and export add none.
    log = s.project.log
    assert [entry.action for entry in log] == ["new_project", *recorder.actions]
    assert [entry.seq for entry in log] == list(range(1, len(log) + 1))
    seconds = [*range(10), 11, 12, 13]  # second 10 is the export's timestamp
    assert [entry.time for entry in log] == [f"2026-09-26T08:00:{i:02d}.000Z" for i in seconds]


def test_default_autosave_writes_every_change(tmp_path):
    s, image, target = boxed(tmp_path, save_to_folder)
    assert load_project(s.folder) == s.project and not s.dirty
    loading = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    for protein in (target, loading):
        ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
        ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
    ops.set_reference_condition(s, "vehicle")
    assert load_project(s.folder) == s.project
    assert not s.dirty and s.save_error is None

    reopened = ops.open_project(s.folder)
    assert ops.compute(reopened) == ops.compute(s)
    assert ops.compute(s).series[0].chart is not None


def test_failed_autosave_keeps_the_edit_in_memory(tmp_path, replace_lock):
    s, _, protein = boxed(tmp_path, save_to_folder)
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    path = s.folder / storage.PROJECT_FILE
    before = path.read_bytes()

    replace_lock.locked = True
    ops.move_box(s, band, (50, 20, 70, 40))  # returns normally
    size = protein_of(s, protein).box_size
    assert band_of(s, band).box.rect(size) == boxes.centered_rect(60, 30, size, W, H)
    assert s.dirty
    assert isinstance(s.save_error, PermissionError)
    assert path.read_bytes() == before

    replace_lock.locked = False
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))  # the next change retries
    assert not s.dirty and s.save_error is None
    saved = load_project(s.folder)
    assert saved == s.project
    assert saved.batch.find_band(band)[1].manually_edited  # the earlier edit is saved too


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to replace an open file")
def test_autosave_while_a_reader_holds_project_json(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    s, _, protein = boxed(tmp_path, save_to_folder)
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    path = s.folder / storage.PROJECT_FILE
    before = path.read_bytes()
    with open(path, "rb"):
        ops.move_box(s, band, (50, 20, 70, 40))
    assert band_of(s, band).manually_edited
    assert s.dirty and isinstance(s.save_error, PermissionError)
    assert path.read_bytes() == before

    ops.set_box_size(s, protein, BoxSize(width=11, height=7))
    assert not s.dirty and s.save_error is None
    assert load_project(s.folder) == s.project


def test_explicit_save_raises_and_records_the_error(tmp_path, replace_lock):
    s, _, _ = boxed(tmp_path)  # no autosave: the edits are unsaved
    assert s.dirty
    replace_lock.locked = True
    with pytest.raises(PermissionError):
        ops.save(s)
    assert isinstance(s.save_error, PermissionError) and s.dirty

    replace_lock.locked = False
    assert ops.save(s) == s.folder / storage.PROJECT_FILE
    assert not s.dirty and s.save_error is None
    assert load_project(s.folder) == s.project


# --- refused operations change nothing ---


def _refusal_scene(tmp_path: Path) -> tuple[ProjectSession, Recorder, dict[str, str]]:
    """A saved project reopened with an empty pixel cache and a recording hook:
    GAPDH is β-catenin's loading control; β-catenin has boxes in lanes 0 and 1."""
    s = session_on(tmp_path, save_to_folder)
    image = import_blot(s, blot())
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("vehicle"), LaneInput("10 µM")])
    loading = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    target = ops.add_protein(s, "β-catenin", Role.TARGET, image, loading_control_ids=[loading])
    a = ops.place_box(s, target, NARROW_X, ROW, lane_index=0, grow=True)
    b = ops.place_box(s, target, WIDE_X, ROW, lane_index=1, grow=True)
    recorder = Recorder()
    reopened = ops.open_project(s.folder, autosave=recorder)
    ids = {"image": image, "loading": loading, "target": target, "a": a, "b": b}
    return reopened, recorder, ids


def _drop_lanes(s: ProjectSession, ids: dict[str, str]) -> None:
    ops.remove_box(s, ids["a"])
    ops.remove_box(s, ids["b"])
    ops.set_lanes(s, [])


REFUSALS = [
    pytest.param(
        None,
        lambda s, ids: ops.add_protein(s, "gapdh", Role.LOADING_CONTROL, ids["image"]),
        ErrorCode.DUPLICATE_NAME,
        id="duplicate-name",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.add_protein(s, "Condition", Role.TARGET, ids["image"]),
        ErrorCode.RESERVED_NAME,
        id="reserved-name",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.place_box(s, ids["target"], 70, ROW, lane_index=0, grow=False),
        ErrorCode.LANE_OCCUPIED,
        id="occupied-lane",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.place_box(s, ids["target"], 70, ROW, lane_index=3, grow=False),
        ErrorCode.LANE_OUT_OF_RANGE,
        id="lane-out-of-range",
    ),
    pytest.param(
        _drop_lanes,
        lambda s, ids: ops.place_box(s, ids["target"], 70, ROW, lane_index=0, grow=False),
        ErrorCode.NO_LANES,
        id="no-lanes",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.place_box(s, ids["target"], W, ROW, lane_index=2, grow=False),
        ErrorCode.OUT_OF_IMAGE,
        id="outside-the-image",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.move_box(s, ids["b"], (NARROW_X - 5, ROW - 5, NARROW_X + 5, ROW + 5)),
        ErrorCode.OVERLAP,
        id="move-into-overlap",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.set_box_size(s, ids["target"], BoxSize(width=80, height=5)),
        ErrorCode.SIZE_WOULD_OVERLAP,
        id="size-forces-overlap",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.place_box(s, ids["target"], *BACKGROUND_POINT, lane_index=2, grow=True),
        ErrorCode.NO_BAND_FOUND,
        id="click-on-background",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.remove_box(s, "band-999"),
        None,
        id="unknown-id",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.edit_protein(s, ids["loading"], role=Role.TARGET),
        ErrorCode.LOADING_CONTROL_IN_USE,
        id="loading-control-in-use",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.remove_undetected(s, ids["target"], 3),
        ErrorCode.LANE_OUT_OF_RANGE,
        id="remove-undetected-lane-out-of-range",
    ),
    pytest.param(
        None,
        lambda s, ids: ops.remove_undetected(s, "prot-999", 0),
        None,
        id="remove-undetected-unknown-protein",
    ),
]


@pytest.mark.parametrize(("setup", "call", "code"), REFUSALS)
def test_refused_operation_changes_nothing(tmp_path, setup, call, code):
    s, recorder, ids = _refusal_scene(tmp_path)
    if setup is not None:
        setup(s, ids)
    recorder.actions.clear()
    before = s.project
    next_id, log_length = before.next_id, len(before.log)
    files = listing(s)
    cache = dict(s._pixels)

    with pytest.raises(UnknownIdError if code is None else OperationError) as info:
        call(s, ids)
    if code is not None:
        assert info.value.code is code
    assert s.project is before
    assert s.project.next_id == next_id
    assert len(s.project.log) == log_length  # a failed attempt is not history
    assert recorder.actions == []
    assert listing(s) == files
    assert s._pixels.keys() == cache.keys()
    assert all(s._pixels[key] is array for key, array in cache.items())


def test_no_op_logs_nothing(tmp_path):
    s, recorder, ids = _refusal_scene(tmp_path)
    before = s.project
    lanes = before.batch.lanes
    target = before.batch.find_protein(ids["target"])
    ops.set_lanes(s, [LaneInput(lane.label, lane.sample, lane.included) for lane in lanes])
    ops.move_box(s, ids["a"], before.batch.find_band(ids["a"])[1].box.rect(target.box_size))
    ops.set_box_size(s, ids["target"], target.box_size)
    ops.set_polarity(s, ids["image"], DARK)
    ops.edit_protein(s, ids["target"], name=" β-catenin ")  # the stored name, once cleaned
    ops.set_reference_condition(s, None)  # no reference is set
    ops.remove_undetected(s, ids["target"], 2)  # no record there
    assert s.project is before
    assert len(s.project.log) == len(before.log)
    assert recorder.actions == []


def test_refusals_name_the_objects_involved(tmp_path):
    s, _, ids = _refusal_scene(tmp_path)
    with pytest.raises(OperationError) as info:
        ops.move_box(s, ids["b"], (NARROW_X - 5, ROW - 5, NARROW_X + 5, ROW + 5))
    assert info.value.ids == (ids["a"],)
    with pytest.raises(OperationError) as info:
        ops.edit_protein(s, ids["loading"], role=Role.TARGET)
    assert info.value.ids == (ids["target"],)
    with pytest.raises(OperationError) as info:
        ops.place_box(s, ids["target"], 70, ROW, lane_index=0, grow=False)
    assert info.value.ids == (ids["a"],)


@pytest.mark.parametrize(
    ("call", "code", "message"),
    [
        pytest.param(
            lambda s, ids: ops.place_box(s, ids["target"], 70, ROW, lane_index=0, grow=False),
            ErrorCode.LANE_OCCUPIED,
            "'β-catenin' already has a box in lane 1",
            id="place-occupied",
        ),
        pytest.param(
            lambda s, ids: ops.place_box(s, ids["target"], 70, ROW, lane_index=3, grow=False),
            ErrorCode.LANE_OUT_OF_RANGE,
            "lane 4 is not one of the 3 lanes",
            id="place-past-the-last-lane",
        ),
        pytest.param(
            lambda s, ids: ops.place_box(s, ids["target"], 70, ROW, lane_index=-1, grow=False),
            ErrorCode.LANE_OUT_OF_RANGE,
            "lane 0 is not one of the 3 lanes",
            id="place-before-the-first-lane",
        ),
        pytest.param(
            lambda s, ids: ops.set_box_lane(s, ids["a"], 1),
            ErrorCode.LANE_OCCUPIED,
            "'β-catenin' already has a box in lane 2",
            id="set-box-lane-occupied",
        ),
        pytest.param(
            lambda s, ids: ops.set_box_lane(s, ids["a"], 3),
            ErrorCode.LANE_OUT_OF_RANGE,
            "lane 4 is not one of the 3 lanes",
            id="set-box-lane-out-of-range",
        ),
        pytest.param(
            lambda s, ids: ops.remove_undetected(s, ids["target"], 3),
            ErrorCode.LANE_OUT_OF_RANGE,
            "lane 4 is not one of the 3 lanes",
            id="remove-undetected-out-of-range",
        ),
        pytest.param(
            lambda s, ids: ops.set_lanes(s, [LaneInput("vehicle")]),
            ErrorCode.LANES_IN_USE,
            "1 box(es) are in lane 2, which the new table drops; remove them first",
            id="lanes-in-use",
        ),
        pytest.param(
            lambda s, ids: ops.set_lanes(s, []),
            ErrorCode.LANES_IN_USE,
            "2 box(es) are in lanes 1, 2, which the new table drops; remove them first",
            id="lanes-in-use-all",
        ),
        pytest.param(
            lambda s, ids: ops.set_lanes(s, [LaneInput("vehicle"), LaneInput(f" {ZWSP} ")]),
            ErrorCode.BLANK_TEXT,
            "lane 2 condition must not be blank",
            id="set-lanes-blank-condition",
        ),
        pytest.param(
            lambda s, ids: ops.set_lanes(s, [LaneInput("vehicle", "a\x07b")]),
            ErrorCode.CONTROL_CHARACTER,
            "lane 1 sample must not contain the control character '\\x07'",
            id="set-lanes-control-character",
        ),
        pytest.param(
            lambda s, ids: ops.set_lanes(s, [LaneInput("a"), LaneInput("b", included=1)]),
            ErrorCode.INVALID_INPUT,
            "lane 2 included must be True or False, not 1",
            id="set-lanes-included",
        ),
        pytest.param(
            lambda s, ids: ops.set_lanes(s, [LaneInput("a"), LaneInput("b"), "c"]),
            ErrorCode.INVALID_INPUT,
            "lane 3 must be a LaneInput, not 'c'",
            id="set-lanes-not-a-lane-input",
        ),
    ],
)
def test_refusals_number_lanes_from_1(tmp_path, call, code, message):
    # The UI numbers lanes from 1 and shows a refusal's message as it is; the
    # ids keep the stored lanes' objects. β-catenin's boxes are in the first
    # two of three lanes (stored lanes 0 and 1).
    s, _, ids = _refusal_scene(tmp_path)
    with pytest.raises(OperationError) as info:
        call(s, ids)
    assert (info.value.code, str(info.value)) == (code, message)


def test_a_change_that_invalidates_the_project_uses_up_no_id(tmp_path):
    s = session_on(tmp_path)
    before = s.project

    def change(draft: Project) -> None:
        draft.new_id("band")
        draft.batch.reference_condition = "nowhere"  # not a lane condition

    with pytest.raises(OperationError) as info:
        ops._prepare(s, change)
    assert info.value.code is ErrorCode.INVALID_INPUT
    assert str(info.value) == "reference condition 'nowhere' is not a lane condition"
    assert s.project is before and s.project.next_id == 1


def test_parallel_operations_get_distinct_ids(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    image = import_blot(s, blot())
    start = s.project.next_id
    barrier = threading.Barrier(8)
    ids: list[str | None] = [None] * 8

    def add(i: int) -> None:
        barrier.wait()
        ids[i] = ops.add_protein(s, f"protein {i} α", Role.LOADING_CONTROL, image)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(ids)) == 8 and None not in ids
    assert {p.id for p in s.project.batch.proteins} == set(ids)
    assert s.project.next_id == start + 8
    assert load_project(s.folder) == s.project
    log = s.project.log
    assert [entry.seq for entry in log] == list(range(1, len(log) + 1))
    added = [entry.params["protein_id"] for entry in log if entry.action == "add_protein"]
    assert added == [p.id for p in s.project.batch.proteins]  # in commit order


def test_core_imports_no_gui_toolkit():
    code = (
        "import sys\n"
        "import proteia.core.operations, proteia.core.results\n"
        "gui = ('napari', 'qtpy', 'magicgui', 'PySide6')\n"
        "print([name for name in gui if name in sys.modules])\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert done.stdout.strip() == "[]"


# --- images ---


def test_import_records_the_stored_file(tmp_path):
    s = session_on(tmp_path)
    pixels = blot()
    name = "β-actin 10 µM.tif"
    image_id = import_blot(s, pixels, name, LIGHT)
    assert image_id == "img-1"
    [membrane] = s.project.batch.membranes
    assert membrane.id == "mem-2"
    [image] = membrane.images
    source = (tmp_path / "sources" / name).read_bytes()
    stored = s.folder / "images" / "img-1.tif"
    assert stored.read_bytes() == source
    assert image.file == "img-1.tif"
    assert image.sha256 == hashlib.sha256(source).hexdigest()
    assert (image.width, image.height, image.bit_depth) == (W, H, 16)
    assert image.original_name == name
    assert image.kind is CHEMI and image.polarity is LIGHT
    assert image.import_warnings == []
    array = s.pixels(image_id)
    np.testing.assert_array_equal(array, pixels.astype(np.float64))
    assert image.background == estimate_background(array)
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array[0, 0] = 0.0

    float_id = import_blot(s, pixels.astype(np.float64), "float α.tif")
    float_image = s.project.batch.find_image(float_id)
    assert float_image.bit_depth is None
    assert [w.code for w in float_image.import_warnings] == ["unknown_bit_depth"]

    ops.remove_image(s, image_id)
    assert image_id not in s._pixels  # evicted with the image
    assert float_id in s._pixels


def test_import_into_an_existing_membrane(tmp_path):
    s = session_on(tmp_path)
    first = import_blot(s, blot())
    membrane = s.project.batch.membrane_of(first).id
    second = import_blot(s, blot(), "reprobe β.tif", membrane_id=membrane)
    [only] = s.project.batch.membranes
    assert [image.id for image in only.images] == [first, second]

    files, next_id = listing(s), s.project.next_id
    with pytest.raises(UnknownIdError):
        import_blot(s, blot(), "other α.tif", membrane_id="mem-99")
    assert listing(s) == files and s.project.next_id == next_id


@pytest.mark.parametrize(
    ("name", "data", "max_bytes", "code"),
    [
        pytest.param(
            "broken α.tif", b"not a TIFF at all", None, "unreadable_image", id="undecodable"
        ),
        pytest.param("blot β.bmp", b"BM" + bytes(64), None, "unsupported_image_type", id="bmp"),
        pytest.param("empty µ.tif", b"", None, "invalid_image", id="empty"),
        pytest.param("big α.tif", bytes(5000), 1000, "image_too_large", id="too-large"),
    ],
)
def test_refused_import_leaves_no_file(tmp_path, name, data, max_bytes, code):
    recorder = Recorder()
    s = session_on(tmp_path, recorder)
    with pytest.raises(OperationError) as info:
        ops.import_image(s, io.BytesIO(data), name, kind=CHEMI, polarity=DARK, max_bytes=max_bytes)
    assert info.value.code == code
    assert listing(s) == []
    assert s.project.next_id == 1
    assert recorder.actions == []


def test_import_removes_orphans_that_hold_its_id(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    images = s.folder / "images"
    (images / "img-1.tif").write_bytes(b"left by an import that was never saved")
    (images / ".img-1.tif.x.part").write_bytes(b"left by a crash")
    (images / "notes.txt").write_text("the user's own file", encoding="utf-8")
    image_id = import_blot(s, blot(), "β.tif")
    assert image_id == "img-1"
    assert listing(s) == ["img-1.tif", "notes.txt"]
    assert (images / "img-1.tif").read_bytes() == (tmp_path / "sources" / "β.tif").read_bytes()


def test_saved_references_survive_a_failed_autosave(tmp_path, replace_lock):
    s = session_on(tmp_path, save_to_folder)
    a = import_blot(s, blot(), "A α.tif")
    a_file = s.folder / "images" / "img-1.tif"
    replace_lock.locked = True
    ops.remove_image(s, a)
    b = import_blot(s, blot(), "B β.tif")  # cleans orphans first
    assert s.save_error is not None and s.dirty
    assert a_file.exists()  # the project.json on disk still references it
    crashed = load_project(s.folder)  # as if the app died now
    assert [image.id for image in crashed.batch.iter_images()] == [a]

    replace_lock.locked = False
    ops.save(s)
    assert not a_file.exists()
    assert listing(s) == [f"{b}.tif"]


def test_remove_image_cascade(tmp_path):
    s = open_sample(tmp_path / "tubulin")
    cascade = ops.remove_image(s, "img-6")
    assert cascade == Cascade(
        removed=("img-6", "prot-8", "band-13", "band-14", "band-15", "band-16", "mem-5"),
        detached_targets=("prot-7",),
        unpaired_images=(),
        unfitted_membranes=(),
    )
    assert s.project.log[-1].params == {
        "image_id": "img-6",
        **_listed(cascade),
        "removed_undetected": [],
        "dropped_undetected": [],
    }
    batch = s.project.batch
    assert [p.id for p in batch.proteins] == ["prot-7", "prot-9"]
    assert batch.find_protein("prot-7").loading_control_ids == []
    assert [m.id for m in batch.membranes] == ["mem-1"]
    assert not (s.folder / "images" / "img-6.jpg").exists()  # gone after the autosave
    assert load_project(s.folder) == s.project
    # prot-7 falls back to GAPDH, now the batch's single loading control.
    assert [(x.target_id, x.loading_id) for x in ops.compute(s).series] == [("prot-7", "prot-9")]

    s = open_sample(tmp_path / "marker")
    cascade = ops.remove_image(s, "img-3")
    assert cascade == Cascade(
        removed=("img-3",),
        detached_targets=(),
        unpaired_images=("img-2",),
        unfitted_membranes=("mem-1",),
    )
    assert s.project.log[-1].params == {
        "image_id": "img-3",
        **_listed(cascade),
        "removed_undetected": [],
        "dropped_undetected": [],
    }
    batch = s.project.batch
    assert batch.find_image("img-2").marker_image_id is None
    calibration = batch.membranes[0].calibration
    assert [point.image_id for point in calibration.points] == ["img-2"]
    assert calibration.fit_quality is None
    assert batch.find_band("band-12")[1].apparent_mw is None
    assert not (s.folder / "images" / "img-3.png").exists()


def test_set_polarity_recomputes_the_nets_on_that_image(tmp_path):
    s = session_on(tmp_path)
    bright = import_blot(s, (65535 - blot()).astype(np.uint16), "fluorescence α.tif")
    dark = import_blot(s, blot(), "chemi β.tif")
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    on_bright = ops.add_protein(s, "β-catenin", Role.TARGET, bright)
    on_dark = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, dark)
    for protein in (on_bright, on_dark):
        ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
        ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=False)

    def mark_clipped(draft: Project) -> None:  # a stale flag the polarity change must redo
        for protein in draft.batch.proteins:
            for band in protein.bands:
                band.clipped = True

    plant(s, mark_clipped)
    background = s.project.batch.find_image(bright).background
    dark_bands = protein_of(s, on_dark).bands

    ops.set_polarity(s, bright, LIGHT)
    image = s.project.batch.find_image(bright)
    assert image.polarity is LIGHT and image.background == background
    protein = protein_of(s, on_bright)
    pixels = s.pixels(bright)
    for band in protein.bands:
        expected = net_signal(pixels, band.box, protein.box_size, background, dark_on_light=False)
        assert band.net == expected and band.net > 0
        assert band.clipped is False  # recomputed for the new polarity (16-bit, below 65535)
    assert protein_of(s, on_dark).bands == dark_bands  # the other image is untouched


def test_changed_image_file_is_refused(tmp_path):
    s, _, protein = boxed(tmp_path, save_to_folder)
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    reopened = ops.open_project(s.folder)
    path = reopened.folder / "images" / "img-1.tif"
    write_tiff(path, blot() // 2)  # other bytes, same shape
    before = reopened.project
    for problem in ("other bytes", "missing"):
        if problem == "missing":
            path.unlink()
        with pytest.raises(OperationError) as info:
            ops.move_box(reopened, band, (50, 20, 70, 40))
        assert info.value.code is ErrorCode.IMAGE_FILE_CHANGED, problem
        assert info.value.ids == ("img-1",)
        assert reopened.project is before
        assert reopened._pixels == {}
        assert ops.compute(reopened).proteins[0].band_ids[0] == band  # results still work


def test_pixels_of_another_shape_are_refused(tmp_path):
    s = session_on(tmp_path)
    image_id = import_blot(s, blot())

    def swap_sides(draft: Project) -> None:
        image = draft.batch.find_image(image_id)
        image.width, image.height = image.height, image.width

    project, _ = apply_change(s.project, swap_sides)
    s._commit(project, action="plant", params={}, evict=[image_id])
    with pytest.raises(OperationError) as info:
        s.pixels(image_id)
    assert info.value.code is ErrorCode.IMAGE_FILE_CHANGED


# --- lanes and the reference ---


def _four(first: str, second: str, *, last: str = "10 µM") -> list[LaneInput]:
    """The sample project's lane table with the two vehicle lanes relabelled."""
    return [
        LaneInput(first, "v1"),
        LaneInput(second, "v2"),
        LaneInput("10 µM", "a1"),
        LaneInput(last, "a2", included=False),
    ]


def test_set_lanes(tmp_path):
    s = open_sample(tmp_path, hook=None)
    update = ops.set_lanes(
        s,
        [
            LaneInput(" vehicle ", " v1\t"),
            LaneInput("vehicle", "v2"),
            LaneInput(f"10{WIDE_SPACE}µM", "   "),
            LaneInput("10 µM", "a2", included=False),
        ],
    )
    assert update == LanesUpdate(respelled=(), reference_cleared=False)
    lanes = s.project.batch.lanes
    assert [lane.label for lane in lanes] == ["vehicle", "vehicle", "10 µM", "10 µM"]
    assert [lane.sample for lane in lanes] == ["v1", "v2", None, "a2"]
    assert lanes[3].metadata == {"dose": "10 µM", "day": "1"}  # kept

    # A look-alike of a stored condition takes the stored spelling.
    update = ops.set_lanes(s, _four("vehicle", "vehicle", last=f"10 {MU}M"))
    assert update.respelled == (3,)
    assert s.project.batch.lanes[3].label == f"10 {MICRO}M"
    ops.set_lanes(s, [*_four("vehicle", "vehicle")[:3], LaneInput(f"10 {MU}M", "a2")])
    [series] = ops.compute(s).series
    assert list(series.groups) == ["vehicle", f"10 {MICRO}M"]  # one group, not two

    for lanes_in, code in [
        ([LaneInput(f" {ZWSP} ")], ErrorCode.BLANK_TEXT),
        ([LaneInput("a\x07b")], ErrorCode.CONTROL_CHARACTER),
        ([LaneInput("vehicle", included=1)], ErrorCode.INVALID_INPUT),
        (["vehicle"], ErrorCode.INVALID_INPUT),
    ]:
        with pytest.raises(OperationError) as info:
            ops.set_lanes(s, lanes_in)
        assert info.value.code is code

    # Lanes that hold a box cannot be dropped; the table can grow.
    with pytest.raises(OperationError) as info:
        ops.set_lanes(s, _four("vehicle", "vehicle")[:3])
    assert info.value.code is ErrorCode.LANES_IN_USE
    assert info.value.ids == ("band-12", "band-16")
    ops.set_lanes(s, [*_four("vehicle", "vehicle"), LaneInput("50 µM", "b1")])
    assert len(s.project.batch.lanes) == 5
    ops.remove_box(s, "band-12")
    ops.remove_box(s, "band-16")
    ops.set_lanes(s, _four("vehicle", "vehicle")[:3])
    assert len(s.project.batch.lanes) == 3


def test_reference_follows_the_lane_table(tmp_path):
    s = open_sample(tmp_path, hook=None)
    ops.set_lanes(s, _four("DMSO", "vehicle"))  # one vehicle lane is left
    assert s.project.batch.reference_condition == "vehicle"

    update = ops.set_lanes(s, _four("DMSO", "DMSO"))  # the last vehicle lane renamed
    assert update.reference_cleared
    assert s.project.batch.reference_condition is None

    ops.set_lanes(s, _four("vehicle", "vehicle"), reference_condition="vehicle")
    update = ops.set_lanes(s, _four("DMSO", "DMSO"), reference_condition="DMSO")  # atomic
    assert not update.reference_cleared
    assert s.project.batch.reference_condition == "DMSO"

    ops.set_reference_condition(s, f"10 {MU}M")
    assert s.project.batch.reference_condition == f"10 {MICRO}M"  # the lane's spelling
    for call in (
        lambda: ops.set_reference_condition(s, "vehicle"),
        lambda: ops.set_lanes(s, _four("DMSO", "DMSO"), reference_condition="vehicle"),
    ):
        with pytest.raises(OperationError) as info:
            call()
        assert info.value.code is ErrorCode.UNKNOWN_CONDITION
    ops.set_reference_condition(s, None)
    assert s.project.batch.reference_condition is None


# --- proteins ---


def test_protein_names(tmp_path):
    s = session_on(tmp_path)
    image = import_blot(s, blot())
    gapdh = ops.add_protein(s, "  GAPDH  ", Role.LOADING_CONTROL, image)
    assert protein_of(s, gapdh).name == "GAPDH"
    calpain = ops.add_protein(s, f"{MICRO}-calpain", Role.TARGET, image)

    for name, owner in [
        ("gapdh", gapdh),
        ("ＧＡＰＤＨ", gapdh),
        (f"GAPDH{ZWSP}", gapdh),
        (f"{MU}-calpain", calpain),
    ]:
        with pytest.raises(OperationError) as info:
            ops.add_protein(s, name, Role.TARGET, image)
        assert info.value.code is ErrorCode.DUPLICATE_NAME
        assert info.value.ids == (owner,)
    with pytest.raises(OperationError) as info:
        ops.edit_protein(s, calpain, name="Gapdh")
    assert info.value.code is ErrorCode.DUPLICATE_NAME

    ops.edit_protein(s, gapdh, name="Gapdh")  # a case-only rename of itself
    assert protein_of(s, gapdh).name == "Gapdh"

    for name, code in [
        ("Condition", ErrorCode.RESERVED_NAME),
        (" LANE ", ErrorCode.RESERVED_NAME),
        ("include", ErrorCode.RESERVED_NAME),
        (WIDE_SPACE, ErrorCode.BLANK_TEXT),
        ("a\x07b", ErrorCode.CONTROL_CHARACTER),
    ]:
        with pytest.raises(OperationError) as info:
            ops.add_protein(s, name, Role.TARGET, image)
        assert info.value.code is code

    atpase = ops.add_protein(s, "Na⁺/K⁺-ATPase", Role.TARGET, image)
    assert protein_of(s, atpase).name == "Na⁺/K⁺-ATPase"  # visible text is never rewritten


def test_add_protein(tmp_path):
    s = session_on(tmp_path)
    image = import_blot(s, blot())
    marker = import_blot(s, blot(), "marker α.tif", kind=ImageKind.VISIBLE_MARKER)
    loading = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    target = ops.add_protein(
        s, "β-catenin", Role.TARGET, image, expected_mw=92, loading_control_ids=[loading]
    )
    assert protein_of(s, loading).box_size == boxes.initial_box_size(W, H)
    added = protein_of(s, target)
    assert (added.expected_mw, added.loading_control_ids) == (92.0, [loading])
    assert [p.id for p in s.project.batch.proteins] == [loading, target]  # appended last

    refusals = [
        ({"image_id": marker}, ErrorCode.MARKER_IMAGE),
        ({"role": Role.LOADING_CONTROL, "loading_control_ids": [loading]}, ErrorCode.INVALID_INPUT),
        ({"loading_control_ids": ["prot-99"]}, UnknownIdError),
        ({"loading_control_ids": [target]}, ErrorCode.INVALID_INPUT),
        ({"loading_control_ids": [loading, loading]}, ErrorCode.INVALID_INPUT),
        ({"box_size": BoxSize(width=W + 1, height=4)}, ErrorCode.SIZE_OUT_OF_BOUNDS),
        ({"expected_mw": 0}, ErrorCode.INVALID_INPUT),
        ({"expected_mw": float("nan")}, ErrorCode.INVALID_INPUT),
        ({"expected_mw": -1}, ErrorCode.INVALID_INPUT),
        ({"expected_mw": True}, ErrorCode.INVALID_INPUT),
        ({"image_id": "img-99"}, UnknownIdError),
    ]
    for overrides, expected in refusals:
        kwargs = {"role": Role.TARGET, "image_id": image, **overrides}
        error = UnknownIdError if expected is UnknownIdError else OperationError
        with pytest.raises(error) as info:
            ops.add_protein(s, "α-tubulin", **kwargs)
        if error is OperationError:
            assert info.value.code is expected, overrides

    removed = ops.remove_protein(s, target)
    assert removed.removed == (target,)
    again = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    number = int(again.rsplit("-", 1)[1])
    assert number > int(target.rsplit("-", 1)[1])  # ids increase and never return
    assert again != target


def test_edit_and_remove_protein(tmp_path):
    s, image, beta = boxed(tmp_path)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    assert protein_of(s, beta).loading_control_ids == []  # GAPDH is used implicitly
    tubulin = ops.add_protein(s, "α-tubulin", Role.LOADING_CONTROL, image)
    assert protein_of(s, beta).loading_control_ids == [gapdh]  # made explicit, not ambiguous
    target = ops.add_protein(
        s, "β-actin", Role.TARGET, image, expected_mw=42, loading_control_ids=[gapdh]
    )

    ops.edit_protein(s, target, name="β-actin (AC-15)")
    assert protein_of(s, target).expected_mw == 42.0  # KEEP leaves it
    ops.edit_protein(s, target, expected_mw=None)
    assert protein_of(s, target).expected_mw is None

    with pytest.raises(OperationError) as info:
        ops.edit_protein(s, gapdh, role=Role.TARGET)
    assert info.value.code is ErrorCode.LOADING_CONTROL_IN_USE
    assert info.value.ids == (beta, target)
    with pytest.raises(OperationError) as info:
        ops.edit_protein(s, target, role=Role.LOADING_CONTROL, loading_control_ids=[tubulin])
    assert info.value.code is ErrorCode.INVALID_INPUT

    ops.edit_protein(s, target, role=Role.LOADING_CONTROL)  # its own list is cleared
    edited = protein_of(s, target)
    assert (edited.role, edited.loading_control_ids) == (Role.LOADING_CONTROL, [])
    ops.edit_protein(s, beta, loading_control_ids=[tubulin])
    ops.edit_protein(s, gapdh, role=Role.TARGET, loading_control_ids=[tubulin, target])
    assert protein_of(s, gapdh).loading_control_ids == [tubulin, target]

    bands = [
        ops.place_box(s, tubulin, NARROW_X, ROW, lane_index=0, grow=True),
        ops.place_box(s, tubulin, WIDE_X, ROW, lane_index=1, grow=True),
    ]
    cascade = ops.remove_protein(s, tubulin)
    assert cascade == Cascade(
        removed=(tubulin, *bands),
        detached_targets=(beta, gapdh),
        unpaired_images=(),
        unfitted_membranes=(),
    )
    assert protein_of(s, gapdh).loading_control_ids == [target]
    with pytest.raises(UnknownIdError):
        s.project.batch.find_band(bands[0])


# --- boxes ---


def test_first_click_sets_the_box_size(tmp_path):
    s, image, protein = boxed(tmp_path)
    pixels = s.pixels(image)
    grown = grow_box(pixels, (NARROW_X, ROW), s.project.batch.find_image(image).background)
    assert grown is not None
    band_id = ops.place_box(s, protein, NARROW_X, ROW, lane_index=1, grow=True)
    placed = protein_of(s, protein)
    size = BoxSize(width=grown[2] - grown[0], height=grown[3] - grown[1])
    assert placed.box_size == size  # replaces the add_protein default
    assert size != boxes.initial_box_size(W, H)
    [band] = placed.bands
    assert band.id == band_id
    centre = ((grown[0] + grown[2]) // 2, (grown[1] + grown[3]) // 2)
    assert band.box.rect(size) == boxes.centered_rect(*centre, size, W, H)
    assert (band.lane_index, band.band_index) == (1, 0)
    assert band.source is ProposalSource.CLICK
    assert not band.manually_edited
    assert_nets_current(s)


def test_a_wider_band_grows_every_box(tmp_path):
    s, _, protein = boxed(tmp_path)
    first = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    old_size = protein_of(s, protein).box_size
    old = band_of(s, first)
    x0, y0, x1, y1 = old.box.rect(old_size)

    second = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
    size = protein_of(s, protein).box_size
    assert size.width > old_size.width and size.height == old_size.height
    moved = band_of(s, first)
    nx0, ny0, nx1, ny1 = moved.box.rect(size)
    assert ((nx0 + nx1) // 2, (ny0 + ny1) // 2) == ((x0 + x1) // 2, (y0 + y1) // 2)
    assert moved.net > old.net  # recomputed with the wider box
    assert not moved.manually_edited  # a size change is a protein-level parameter
    assert band_of(s, second).lane_index == 1
    assert_nets_current(s)


def test_fixed_box_is_clamped_into_the_image(tmp_path):
    s, _, protein = boxed(tmp_path)
    size = protein_of(s, protein).box_size
    band_id = ops.place_box(s, protein, W - 2, ROW, lane_index=2, grow=False)
    band = band_of(s, band_id)
    assert band.box == Box(x=W - size.width, y=ROW - size.height // 2)
    assert band.source is ProposalSource.MANUAL
    assert protein_of(s, protein).box_size == size
    assert_nets_current(s)


def test_move_box_reads_the_rect_by_its_centre(tmp_path):
    s, _, protein = boxed(tmp_path)
    a = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    b = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)

    def calibrate(draft: Project) -> None:  # #58 will compute apparent MWs
        for protein in draft.batch.proteins:
            for band in protein.bands:
                band.apparent_mw = 55.0

    plant(s, calibrate)
    untouched = band_of(s, b)
    size = protein_of(s, protein).box_size

    ops.move_box(s, a, (30, 44, 10, 20))  # corners in any order; a 20x24 rect centred (20, 32)
    moved = band_of(s, a)
    assert moved.box.rect(size) == boxes.centered_rect(20, 32, size, W, H)
    assert moved.manually_edited
    assert moved.lane_index == 0  # never derived from x
    assert moved.apparent_mw is None  # position-derived: stale
    assert band_of(s, b) == untouched  # bit-identical, apparent MW kept
    assert protein_of(s, protein).box_size == size
    assert_nets_current(s)


def test_set_box_size_recentres_and_requantifies(tmp_path):
    s, _, protein = boxed(tmp_path)
    for x, lane in ((NARROW_X, 0), (WIDE_X, 1)):
        ops.place_box(s, protein, x, ROW, lane_index=lane, grow=True)
    old = protein_of(s, protein)
    size = BoxSize(width=24, height=9)
    ops.set_box_size(s, protein, size)
    resized = protein_of(s, protein)
    assert resized.box_size == size
    expected = boxes.resize_all(
        [band.box.rect(old.box_size) for band in old.bands], size, width=W, height=H
    )
    assert [band.box.rect(size) for band in resized.bands] == expected
    assert [band.net for band in resized.bands] != [band.net for band in old.bands]
    assert_nets_current(s)

    before = s.project
    for refused, code in [
        (BoxSize(width=80, height=5), ErrorCode.SIZE_WOULD_OVERLAP),
        (BoxSize(width=W + 1, height=5), ErrorCode.SIZE_OUT_OF_BOUNDS),
    ]:
        with pytest.raises(OperationError) as info:
            ops.set_box_size(s, protein, refused)
        assert info.value.code is code
        assert s.project is before


def test_remove_box_keeps_the_size_and_the_other_boxes(tmp_path):
    s, _, protein = boxed(tmp_path)
    a = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    b = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
    before = protein_of(s, protein)
    ops.remove_box(s, a)
    after = protein_of(s, protein)
    assert after.box_size == before.box_size
    assert after.bands == [band for band in before.bands if band.id == b]


def test_stored_nets_stay_current_through_a_session_and_a_reopen(tmp_path):
    s, image, protein = boxed(tmp_path, save_to_folder)
    loading = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    clicked = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    fixed = ops.place_box(s, protein, 70, ROW, lane_index=2, grow=False)
    ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)  # grows, re-centres
    ops.place_box(s, loading, WIDE_X, ROW, lane_index=0, grow=False)
    ops.move_box(s, fixed, (62, 22, 70, 36))
    ops.set_box_size(s, protein, BoxSize(width=21, height=7))
    ops.set_polarity(s, image, LIGHT)
    ops.set_polarity(s, image, DARK)
    ops.remove_box(s, clicked)
    assert_nets_current(s)

    reopened = ops.open_project(s.folder)
    assert reopened.project == s.project
    assert_nets_current(reopened)


# --- compute and export through the operations ---


def _parity_session(tmp_path: Path, pixels: np.ndarray, polarity: Polarity) -> ProjectSession:
    """The regression baseline's blot imported as a float64 TIFF, with its lane
    table and its fixed boxes placed through the operations."""
    s = session_on(tmp_path)
    image = import_blot(s, pixels, "β-catenin α-tubulin 10 µM.tif", polarity)
    lanes = [
        LaneInput(condition, sample, included)
        for condition, sample, included in zip(CONDITIONS, SAMPLES, INCLUDED, strict=True)
    ]
    ops.set_lanes(s, lanes, reference_condition=REFERENCE)
    target = ops.add_protein(s, TARGET, Role.TARGET, image, box_size=BOX_SIZE)
    loading = ops.add_protein(s, LOADING, Role.LOADING_CONTROL, image, box_size=BOX_SIZE)
    for protein, name in ((target, TARGET), (loading, LOADING)):
        for i, x in enumerate(LANE_X):
            ops.place_box(s, protein, x, ROW_Y[name], lane_index=i, grow=False)
    return s


def _golden() -> tuple[dict, float, float]:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return golden["values"], golden["tolerance"]["rel"], golden["tolerance"]["abs"]


@pytest.mark.parametrize("method", list(ReduceMethod))
def test_operations_reproduce_the_regression_baseline(tmp_path, method):
    values, rel, abs_ = _golden()
    s = _parity_session(tmp_path, _blot(), DARK)
    [image] = s.project.batch.iter_images()
    assert image.bit_depth is None  # a float64 TIFF
    _assert_close(image.background, values["background"]["dark_on_light"], rel, abs_)

    res = ops.compute(s, method=method)
    nets = values["nets"]["dark_on_light"]
    _assert_close([c.nets for c in res.proteins], [nets[TARGET], nets[LOADING]], rel, abs_)
    [series] = res.series
    _assert_close(series.normalized, values["lanes"]["normalized"], rel, abs_, "normalized")
    _assert_close(
        series.fold_change, values["lanes"][f"fold_change/{method}"], rel, abs_, "fold_change"
    )
    reduced = values["reduced"][f"fold_change/{method}"]
    _assert_close(series.groups, reduced["groups"], rel, abs_, "groups")
    for case, plot in (("all", None), ("two", [REFERENCE, LOW])):
        charts = [
            ops.compute(s, plot_conditions=plot, error_type=error_type, method=method)
            for error_type in (ErrorType.SD, ErrorType.SEM)
        ]
        _assert_close(_chart_stats(*charts), _golden_chart_stats(reduced[case]), rel, abs_, case)


def test_set_polarity_reproduces_the_light_on_dark_baseline(tmp_path):
    values, rel, abs_ = _golden()
    s = _parity_session(tmp_path, 255.0 - _blot(), DARK)  # boxed while still dark-on-light
    [image] = s.project.batch.iter_images()
    ops.set_polarity(s, image.id, LIGHT)
    [image] = s.project.batch.iter_images()
    _assert_close(image.background, values["background"]["light_on_dark"], rel, abs_)
    nets = values["nets"]["light_on_dark"]
    res = ops.compute(s)
    _assert_close([c.nets for c in res.proteins], [nets[TARGET], nets[LOADING]], rel, abs_)
    assert_nets_current(s)


def test_a_plotted_subset_without_the_reference_keeps_the_baseline(tmp_path):
    s = _parity_session(tmp_path, _blot(), DARK)
    full, part = ops.compute(s), ops.compute(s, plot_conditions=[LOW])
    [whole], [subset] = full.series, part.series
    assert subset.baseline == whole.baseline
    assert subset.groups == whole.groups
    assert subset.chart is not None and whole.chart is not None
    assert [bar.label for bar in subset.chart.bars] == [LOW]
    bars = {bar.label: bar for bar in whole.chart.bars}
    assert subset.chart.bars[0].points == bars[LOW].points
    assert bars[REFERENCE].mean == pytest.approx(1.0)
    assert NoticeCode.REFERENCE_NOT_PLOTTED in [notice.code for notice in part.notices]


def test_table_and_chart_use_the_same_loading_control(tmp_path):
    s = open_sample(tmp_path)
    [series] = ops.compute(s).series
    assert (series.target_id, series.loading_id) == ("prot-7", "prot-8")

    ops.edit_protein(s, "prot-7", loading_control_ids=["prot-9"])  # GAPDH, the second one
    res = ops.compute(s)
    [series] = res.series
    assert (series.target_id, series.loading_id) == ("prot-7", "prot-9")
    columns = {column.protein_id: column.nets for column in res.proteins}
    assert series.normalized == [
        None if t is None or g is None else t / g
        for t, g in zip(columns["prot-7"], columns["prot-9"], strict=True)
    ]
    assert series.chart is not None
    assert series.chart.title == "β-catenin fold-change vs vehicle  (/GAPDH)"
    assert {bar.label: bar.points for bar in series.chart.bars} == series.groups
    assert series.groups == {"vehicle": [v / series.baseline for v in series.normalized[:2]]}
    assert all(x.loading_id != "prot-8" for x in res.series)
    assert load_project(s.folder) == s.project  # saved as well


def test_export_lane_table(tmp_path, monkeypatch):
    recorder = Recorder()
    s = open_sample(tmp_path, recorder)
    path = ops.export_lane_table(s)
    assert path == s.folder / "exports" / "lane-table.csv"
    data = path.read_bytes()
    assert data.startswith(codecs.BOM_UTF8)
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    assert rows[0] == [
        "lane",
        "condition",
        "sample",
        "include",
        "β-catenin",
        "β-catenin clipped",
        "α-tubulin",
        "α-tubulin clipped",
        "GAPDH",
        "GAPDH clipped",
    ]
    assert rows[1] == [
        "0",
        "vehicle",
        "v1",
        "yes",
        str(round(4279.740326695199, 3)),
        "",  # not checked in the sample project
        str(round(7389.877572928557, 3)),
        "",
        "5120.5",
        "",
    ]
    assert rows[3][:6] == ["2", "10 µM", "a1", "yes", "", ""]  # no β-catenin box in lane 2
    assert rows[4][3] == "no"
    assert rows[4][5] == "no"  # band-12 was checked and is not clipped
    assert len(rows) == 5
    assert recorder.actions == []  # not a state change

    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    record_path = s.folder / "exports" / ops.LANE_TABLE_RECORD_FILE
    earlier = record_path.read_bytes()
    path.unlink()
    path.mkdir()  # something in the way
    before = s.project
    with pytest.raises(OSError):
        ops.export_lane_table(s)
    assert s.project is before
    assert record_path.read_bytes() == earlier  # the table goes first: the record is kept
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "lane-table.csv",
        "lane-table.record.json",
    ]  # no temp file left

    empty = session_on(tmp_path / "empty")
    with pytest.raises(OperationError) as info:
        ops.export_lane_table(empty)
    assert info.value.code is ErrorCode.NO_LANES


def test_a_first_export_whose_record_fails_leaves_no_table(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)
    s = open_sample(tmp_path, Recorder())
    exports = s.folder / "exports"
    (exports / ops.LANE_TABLE_RECORD_FILE).mkdir(parents=True)  # the record cannot be written
    with pytest.raises(OSError):
        ops.export_lane_table(s)
    assert [p.name for p in exports.iterdir()] == [ops.LANE_TABLE_RECORD_FILE]


# --- review of #70 ---


def test_removing_the_only_loading_control_reports_its_implicit_users(tmp_path):
    s, image, beta = boxed(tmp_path)  # beta chose no loading control
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    cascade = ops.remove_protein(s, gapdh)
    assert cascade.detached_targets == (beta,)

    other = import_blot(s, blot(), "GAPDH blot.tif")
    ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, other)
    cascade = ops.remove_image(s, other)
    assert cascade.detached_targets == (beta,)


def test_orphan_cleanup_keeps_files_that_only_start_like_proteia_names(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    import_blot(s, blot(), "β.tif")  # img-1
    images = s.folder / "images"
    for name in ("img-1.tif.bak", "img-2.orig.png", "img-01.tif", "notes.txt"):
        (images / name).write_bytes(b"the user's own file")
    (images / "IMG-9.TIF").write_bytes(b"a Proteia file in other case, no longer referenced")
    ops.save(s)
    assert listing(s) == ["img-01.tif", "img-1.tif", "img-1.tif.bak", "img-2.orig.png", "notes.txt"]


def test_an_expected_mw_too_large_for_a_float_is_refused(tmp_path):
    s, _, beta = boxed(tmp_path)
    with pytest.raises(OperationError) as info:
        ops.edit_protein(s, beta, expected_mw=10**400)
    assert info.value.code is ErrorCode.INVALID_INPUT


def test_a_hook_error_after_the_commit_keeps_the_committed_cache(tmp_path):
    def broken_hook(session: ProjectSession) -> None:
        raise KeyError("a bug in a custom autosave hook")

    s, image, _ = boxed(tmp_path)
    s.autosave = broken_hook
    s.pixels(image)
    with pytest.raises(KeyError):
        ops.remove_image(s, image)
    assert image not in {i.id for i in s.project.batch.iter_images()}  # committed
    assert image not in s._pixels  # the removed image's pixels do not come back


# --- #43: lane identity when boxes are placed without a lane ---

LANES = 6
LANE_W, LANE_H = 360, 60  # six lanes centred at 30, 90, ..., 330
LANE_ROW = 30


def lane_x(lane: int) -> int:
    return 30 + 60 * lane


def lanes_session(tmp_path: Path) -> tuple[ProjectSession, str]:
    """Six declared lanes, one blot with a band in every lane, one target."""
    s = session_on(tmp_path)
    bands = [(lane_x(i), LANE_ROW, 6.0, 3.0, 30000.0) for i in range(LANES)]
    image = import_blot(s, synthetic_blot((LANE_H, LANE_W), bands))
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(LANES)])
    return s, ops.add_protein(s, "β-catenin", Role.TARGET, image)


@pytest.mark.parametrize(
    "lanes",
    [[1, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 1, 2, 3, 4], [1, 2, 3, 4], [4, 1, 3, 2, 5]],
    ids=["missing-first", "missing-middle", "missing-last", "missing-first-and-last", "any-order"],
)
@pytest.mark.parametrize("grow", [True, False], ids=["seed-click", "fixed-box"])
def test_boxes_placed_without_a_lane_land_in_their_lanes(tmp_path, lanes, grow):
    # No other protein anchors the grid: the first two lanes are chosen, the rest proposed.
    s, protein = lanes_session(tmp_path)
    first, second, *rest = lanes
    ops.place_box(s, protein, lane_x(first), LANE_ROW, lane_index=first, grow=grow)
    ops.place_box(s, protein, lane_x(second), LANE_ROW, lane_index=second, grow=grow)
    for lane in rest:
        band = ops.place_box(s, protein, lane_x(lane), LANE_ROW, grow=grow)
        assert band_of(s, band).lane_index == lane
    nets = results.lane_nets(s.project.batch)[protein]
    assert [i for i, net in enumerate(nets) if net is not None] == sorted(lanes)


def test_a_lane_is_required_until_two_lanes_show_the_spacing(tmp_path):
    s, protein = lanes_session(tmp_path)
    with pytest.raises(OperationError) as info:
        ops.place_box(s, protein, lane_x(2), LANE_ROW, grow=False)
    assert info.value.code is ErrorCode.LANE_REQUIRED  # no box on the image yet
    ops.place_box(s, protein, lane_x(2), LANE_ROW, lane_index=2, grow=False)
    with pytest.raises(OperationError) as info:
        ops.place_box(s, protein, lane_x(3), LANE_ROW, grow=False)
    assert info.value.code is ErrorCode.LANE_REQUIRED  # one lane: no spacing yet
    ops.place_box(s, protein, lane_x(3), LANE_ROW, lane_index=3, grow=False)
    band = ops.place_box(s, protein, lane_x(5), LANE_ROW, grow=False)
    assert band_of(s, band).lane_index == 5


def test_lanes_are_proposed_on_an_image_with_margins(tmp_path):
    # Six lanes from x=200 with a pitch of 70 on a 900-pixel-wide image.
    s = session_on(tmp_path)
    xs = [200 + 70 * i for i in range(LANES)]
    blot = synthetic_blot((LANE_H, 900), [(x, LANE_ROW, 6.0, 3.0, 30000.0) for x in xs])
    image = import_blot(s, blot)
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(LANES)])
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    ops.place_box(s, protein, xs[0], LANE_ROW, lane_index=0, grow=True)
    ops.place_box(s, protein, xs[1], LANE_ROW, lane_index=1, grow=True)
    placed = [ops.place_box(s, protein, x, LANE_ROW, grow=True) for x in xs[2:]]
    assert [band_of(s, band).lane_index for band in placed] == [2, 3, 4, 5]


def test_a_proposal_never_moves_a_box_to_another_lane(tmp_path):
    s, protein = lanes_session(tmp_path)
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(4)])  # the blot shows six
    for lane in (0, 1, 2):
        ops.place_box(s, protein, lane_x(lane), LANE_ROW, lane_index=lane, grow=False)
    # Another box over lane 1 (a second band, or a misclick) is refused, not moved.
    with pytest.raises(OperationError) as info:
        ops.place_box(s, protein, lane_x(1), 8, grow=False)
    assert info.value.code is ErrorCode.LANE_OCCUPIED
    # The message numbers lanes from 1, as the UI does, the proposed one too.
    assert str(info.value) == (
        "'β-catenin' already has a box in lane 2 (the box lies at lane 2); choose the lane"
    )
    # A box beyond the declared lanes is refused too, though lane 3 is free.
    with pytest.raises(OperationError) as info:
        ops.place_box(s, protein, lane_x(5), LANE_ROW, grow=False)
    assert info.value.code is ErrorCode.LANE_OUT_OF_RANGE
    assert str(info.value) == (
        "lane 6 is not one of the 4 lanes (the box lies at lane 6); choose the lane"
    )


def test_a_fixed_box_at_the_edge_is_proposed_from_the_click(tmp_path):
    # A box wider than two lanes is shifted inside the image at the right edge;
    # its lane still comes from where the user clicked.
    s, protein = lanes_session(tmp_path)
    ops.set_box_size(s, protein, BoxSize(width=150, height=10))
    ops.place_box(s, protein, lane_x(1), LANE_ROW, lane_index=1, grow=False)
    ops.place_box(s, protein, lane_x(3), 8, lane_index=3, grow=False)  # another row
    band = ops.place_box(s, protein, lane_x(5), 50, grow=False)
    assert band_of(s, band).box.x == LANE_W - 150  # shifted inside the image
    assert band_of(s, band).lane_index == 5


def test_moving_or_removing_a_box_never_changes_another_lane(tmp_path):
    s, protein = lanes_session(tmp_path)
    bands = {
        lane: ops.place_box(s, protein, lane_x(lane), LANE_ROW, lane_index=lane, grow=False)
        for lane in (0, 2, 4, 5)
    }
    ops.remove_box(s, bands[2])
    ops.move_box(s, bands[4], (lane_x(4) - 9, 20, lane_x(4) + 1, 40))  # nudged left
    kept = (bands[0], bands[4], bands[5])
    assert [band_of(s, band_id).lane_index for band_id in kept] == [0, 4, 5]
    # The next box without a lane is proposed among the free lanes, anchored on the rest.
    new = ops.place_box(s, protein, lane_x(2), LANE_ROW, grow=False)
    assert band_of(s, new).lane_index == 2
    # Dragging a box over another lane's position keeps its lane: identity, not x.
    ops.move_box(s, bands[5], (lane_x(3) - 5, 20, lane_x(3) + 5, 40))
    assert band_of(s, bands[5]).lane_index == 5
    assert [band_of(s, band_id).lane_index for band_id in (bands[0], bands[4], new)] == [0, 4, 2]


def test_a_protein_with_a_box_in_every_lane_refuses_one_more(tmp_path):
    s, protein = lanes_session(tmp_path)
    bands = [
        ops.place_box(s, protein, lane_x(lane), LANE_ROW, lane_index=lane, grow=False)
        for lane in range(LANES)
    ]
    with pytest.raises(OperationError) as info:
        ops.place_box(s, protein, lane_x(2), 8, grow=False)  # another row, over lane 2
    assert info.value.code is ErrorCode.LANE_OCCUPIED
    assert info.value.ids == (bands[2],)


def test_the_lane_is_required_before_any_pixel_work(tmp_path, monkeypatch):
    # A seed click on background with no lanes to anchor on asks for the lane,
    # not "no band found": choosing the lane is what the user must do.
    s, protein = lanes_session(tmp_path)
    monkeypatch.setattr(ops, "grow_box", lambda *a, **k: pytest.fail("grew before asking"))
    with pytest.raises(OperationError) as info:
        ops.place_box(s, protein, 5, 5, grow=True)
    assert info.value.code is ErrorCode.LANE_REQUIRED


def test_a_seed_click_at_the_edge_is_proposed_from_the_band(tmp_path):
    # A wide shared box is shifted inside the image at the right edge; the lane
    # comes from the grown band's own centre, and the shifted box anchors nothing.
    s, protein = lanes_session(tmp_path)
    ops.set_box_size(s, protein, BoxSize(width=150, height=10))
    ops.place_box(s, protein, lane_x(1), LANE_ROW, lane_index=1, grow=False)
    ops.place_box(s, protein, lane_x(3), 8, lane_index=3, grow=False)  # another row
    band = ops.place_box(s, protein, lane_x(5), LANE_ROW, grow=True)
    assert band_of(s, band).box.x + protein_of(s, protein).box_size.width == LANE_W
    assert band_of(s, band).lane_index == 5
    other = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, protein_of(s, protein).image_id)
    ops.place_box(s, other, lane_x(0), LANE_ROW, lane_index=0, grow=False)
    ops.place_box(s, other, lane_x(2), LANE_ROW, lane_index=2, grow=False)
    fourth = ops.place_box(s, other, lane_x(4), LANE_ROW, grow=False)
    assert band_of(s, fourth).lane_index == 4  # not skewed by the shifted lane-5 box


def test_a_box_dragged_out_of_order_does_not_capture_other_lanes(tmp_path):
    s, protein = lanes_session(tmp_path)
    bands = {
        lane: ops.place_box(s, protein, lane_x(lane), LANE_ROW, lane_index=lane, grow=False)
        for lane in (0, 1, 2, 3)
    }
    ops.move_box(s, bands[2], (lane_x(5) - 5, 20, lane_x(5) + 5, 40))  # lane 2 dragged right
    other = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, protein_of(s, protein).image_id)
    band = ops.place_box(s, other, lane_x(4), LANE_ROW, grow=False)
    assert band_of(s, band).lane_index == 4


def _grows_to(monkeypatch, rect) -> None:
    """Make every seed click grow to ``rect``, to pin where the band lies."""
    monkeypatch.setattr(ops, "grow_box", lambda *args, **kwargs: rect)


def test_a_seed_click_takes_the_lane_of_the_band_not_of_the_click(tmp_path, monkeypatch):
    # The click on the band's flank lies over occupied lane 3; the band is in lane 4.
    s, protein = lanes_session(tmp_path)
    for lane in (1, 3):
        ops.place_box(s, protein, lane_x(lane), LANE_ROW, lane_index=lane, grow=False)
    _grows_to(monkeypatch, (lane_x(4) - 10, 25, lane_x(4) + 11, 36))
    band = ops.place_box(s, protein, lane_x(3) + 5, 12, grow=True)
    assert band_of(s, band).lane_index == 4


def test_a_band_cut_by_the_image_edge_is_proposed_from_the_click(tmp_path, monkeypatch):
    # The band runs off the left edge, so its visible centre sits a lane too far left.
    s, protein = lanes_session(tmp_path)
    for lane in (2, 4):
        ops.place_box(s, protein, lane_x(lane), LANE_ROW, lane_index=lane, grow=False)
    _grows_to(monkeypatch, (0, 25, 100, 36))  # centre 50: lane 0 by the band, lane 1 by the click
    band = ops.place_box(s, protein, 85, LANE_ROW, grow=True)
    assert band_of(s, band).lane_index == 1


def test_a_click_outside_the_image_is_out_of_image_with_or_without_a_lane(tmp_path):
    s, protein = lanes_session(tmp_path)
    for lane_index in (None, 0):
        with pytest.raises(OperationError) as info:
            ops.place_box(s, protein, -1, LANE_ROW, lane_index=lane_index, grow=False)
        assert info.value.code is ErrorCode.OUT_OF_IMAGE


# --- #44: over-exposed bands ---


def test_placed_boxes_carry_the_clipping_flag(tmp_path):
    # Dark on light, 16-bit: the lane-1 band bottoms out at 0 (saturated), lane 0 does not.
    s = session_on(tmp_path)
    bands = [(lane_x(0), LANE_ROW, 6.0, 3.0, 30000.0), (lane_x(1), LANE_ROW, 6.0, 3.0, 60000.0)]
    image = import_blot(s, synthetic_blot((LANE_H, LANE_W), bands))
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(2)])
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    fine = ops.place_box(s, protein, lane_x(0), LANE_ROW, lane_index=0, grow=True)
    over = ops.place_box(s, protein, lane_x(1), LANE_ROW, lane_index=1, grow=True)
    assert (band_of(s, fine).clipped, band_of(s, over).clipped) == (False, True)
    assert_nets_current(s)

    res = ops.compute(s)
    [column] = res.proteins
    assert column.clipped == [False, True]
    notice = next(n for n in res.notices if n.code is NoticeCode.CLIPPED)
    assert (notice.protein_ids, notice.lane_indices) == ((protein,), (1,))
    assert "over-exposed in lane 2:" in notice.message  # the user counts lanes from 1

    ops.set_polarity(s, image, LIGHT)  # 65535 is never reached: nothing is clipped
    assert (band_of(s, fine).clipped, band_of(s, over).clipped) == (False, False)


def test_an_image_without_a_detector_limit_is_not_checked(tmp_path):
    s = session_on(tmp_path)
    pixels = synthetic_blot((LANE_H, LANE_W), [(lane_x(0), LANE_ROW, 6.0, 3.0, 60000.0)])
    image = import_blot(s, pixels.astype(np.float64))  # a float TIFF: bit depth unknown
    ops.set_lanes(s, [LaneInput("c0")])
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    band = ops.place_box(s, protein, lane_x(0), LANE_ROW, lane_index=0, grow=False)
    assert band_of(s, band).clipped is None


@pytest.mark.parametrize(
    ("name", "pixels"),
    [
        ("color blot.tif", "rgb"),  # color averaged into gray
        ("lossy blot.jpg", "jpeg"),  # compression moves saturated pixels off the limit
    ],
)
def test_images_whose_limit_cannot_be_trusted_are_not_checked(tmp_path, name, pixels):
    s = session_on(tmp_path)
    gray = synthetic_blot((LANE_H, LANE_W), [(lane_x(0), LANE_ROW, 6.0, 3.0, 60000.0)])
    gray8 = (gray // 257).astype(np.uint8)
    source = tmp_path / "sources" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    if pixels == "rgb":
        write_tiff(source, np.stack([gray8, gray8 // 2, gray8 // 3], axis=-1))
    else:
        from skimage import io as skio

        skio.imsave(source, gray8, check_contrast=False)
    with source.open("rb") as f:
        image = ops.import_image(s, f, name, kind=CHEMI, polarity=DARK)
    ops.set_lanes(s, [LaneInput("c0")])
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    band = ops.place_box(s, protein, lane_x(0), LANE_ROW, lane_index=0, grow=False)
    assert band_of(s, band).clipped is None


def test_the_clipped_notice_lists_only_included_lanes(tmp_path):
    s = session_on(tmp_path)
    bands = [(lane_x(i), LANE_ROW, 6.0, 3.0, 60000.0) for i in range(2)]
    image = import_blot(s, synthetic_blot((LANE_H, LANE_W), bands))
    ops.set_lanes(s, [LaneInput("c0"), LaneInput("c1", included=False)])
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    for lane in range(2):
        ops.place_box(s, protein, lane_x(lane), LANE_ROW, lane_index=lane, grow=False)
    res = ops.compute(s)
    [notice] = [n for n in res.notices if n.code is NoticeCode.CLIPPED]
    assert notice.lane_indices == (0,)  # lane 1 is excluded: the user already acted
    assert "over-exposed in lane 1:" in notice.message  # lane numbers count from 1
    [all_lanes] = [n for n in res.all_lanes.notices if n.code is NoticeCode.CLIPPED]
    assert all_lanes.lane_indices == (0, 1)  # every lane is included in the all-lanes set
    assert "over-exposed in lanes 1, 2:" in all_lanes.message


def test_a_protein_name_that_would_clash_with_a_clipped_column_is_refused(tmp_path):
    s, image, _ = boxed(tmp_path)  # β-catenin
    for name in ("β-catenin clipped", "β-CATENIN  clipped"):
        with pytest.raises(OperationError) as info:
            ops.add_protein(s, name, Role.TARGET, image)
        assert info.value.code is ErrorCode.RESERVED_NAME
    ops.add_protein(s, "GAPDH clipped", Role.LOADING_CONTROL, image)
    with pytest.raises(OperationError) as info:
        ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    assert info.value.code is ErrorCode.RESERVED_NAME


# --- #46: the action log and the export record ---


def _file_sha256(s: ProjectSession, file: str) -> str:
    return hashlib.sha256((s.folder / storage.IMAGES_DIR / file).read_bytes()).hexdigest()


def _rect_of(s: ProjectSession, band_id: str) -> list[int]:
    protein, band = s.project.batch.find_band(band_id)
    return list(band.box.rect(protein.box_size))


def _size_of(s: ProjectSession, protein_id: str) -> dict[str, int]:
    size = s.project.batch.find_protein(protein_id).box_size
    return {"width": size.width, "height": size.height}


def _protein(protein_id: str, name: str, role: str, image_id: str, **fields) -> dict:
    """add_protein's params with its defaults (no expected MW, no loading
    controls, the initial box size of the blot, nothing pinned)."""
    return {
        "protein_id": protein_id,
        "name": name,
        "role": role,
        "image_id": image_id,
        "expected_mw": None,
        "loading_control_ids": [],
        "box_size": {"width": 20, "height": 5},  # boxes.initial_box_size(W, H)
        "pinned_targets": [],
        **fields,
    }


# Every logged operation, as (call, action, the params it must log). ``expected``
# reads the session after the call, for values the pixels decide (grown rects).
# Ids follow the one counter: img-1, mem-2, img-3, then proteins and bands.
LOGGED_STEPS = [
    (
        lambda s: import_blot(s, blot(), "β-actin 10 µM.tif"),
        "import_image",
        lambda s: {
            "image_id": "img-1",
            "membrane_id": "mem-2",
            "new_membrane": True,
            "original_name": "β-actin 10 µM.tif",
            "kind": "chemiluminescence",
            "polarity": "dark_on_light",
            "sha256": _file_sha256(s, "img-1.tif"),
        },
    ),
    (
        lambda s: import_blot(s, blot(), "reprobe β.tif", LIGHT, membrane_id="mem-2"),
        "import_image",
        lambda s: {
            "image_id": "img-3",
            "membrane_id": "mem-2",
            "new_membrane": False,
            "original_name": "reprobe β.tif",
            "kind": "chemiluminescence",
            "polarity": "light_on_dark",
            "sha256": _file_sha256(s, "img-3.tif"),
        },
    ),
    (
        lambda s: ops.set_lanes(
            s,
            [
                LaneInput(" vehicle ", " v1\t"),
                LaneInput("10 µM", "a1"),
                LaneInput("10 µM", "a2", included=False),
            ],
        ),
        "set_lanes",
        lambda s: {
            "lane_count": 3,
            "changed": [  # every row is new; the text as stored
                {"index": 0, "condition": "vehicle", "sample": "v1", "included": True},
                {"index": 1, "condition": "10 µM", "sample": "a1", "included": True},
                {"index": 2, "condition": "10 µM", "sample": "a2", "included": False},
            ],
            "reference_condition": None,
            "dropped_undetected": [],
        },
    ),
    (
        lambda s: ops.set_lanes(
            s,
            [LaneInput("vehicle", "v1"), LaneInput("10 µM", "a1"), LaneInput(f"10 {MU}M", "a2")],
            reference_condition="vehicle",
        ),
        "set_lanes",
        lambda s: {
            "lane_count": 3,
            # Only the changed row, in the stored spelling (micro sign, not mu).
            "changed": [
                {"index": 2, "condition": f"10 {MICRO}M", "sample": "a2", "included": True}
            ],
            "reference_condition": "vehicle",
            "dropped_undetected": [],
        },
    ),
    (
        lambda s: ops.set_reference_condition(s, f"10 {MU}M"),
        "set_reference_condition",
        lambda s: {"reference_condition": f"10 {MICRO}M"},
    ),
    (
        lambda s: ops.add_protein(s, " β-catenin ", Role.TARGET, "img-1", expected_mw=92),
        "add_protein",
        lambda s: _protein("prot-4", "β-catenin", "target", "img-1", expected_mw=92.0),
    ),
    (
        lambda s: ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, "img-3"),
        "add_protein",
        lambda s: _protein("prot-5", "GAPDH", "loading control", "img-3"),
    ),
    (
        lambda s: ops.add_protein(s, "p53", Role.TARGET, "img-1"),
        "add_protein",
        lambda s: _protein("prot-6", "p53", "target", "img-1"),
    ),
    (
        # A target becomes the second loading control: β-catenin, which used GAPDH
        # implicitly, is pinned to it. p53 itself is not listed, and its name,
        # typed the same once cleaned, is not a change.
        lambda s: ops.edit_protein(s, "prot-6", name=" p53 ", role=Role.LOADING_CONTROL),
        "edit_protein",
        lambda s: {
            "protein_id": "prot-6",
            "pinned_targets": ["prot-4"],
            "role": "loading control",
            "dropped_undetected": [],
        },
    ),
    (
        lambda s: ops.add_protein(
            s, "β-actin", Role.TARGET, "img-1", loading_control_ids=["prot-5"]
        ),
        "add_protein",
        lambda s: _protein("prot-7", "β-actin", "target", "img-1", loading_control_ids=["prot-5"]),
    ),
    (
        lambda s: ops.remove_protein(s, "prot-5"),
        "remove_protein",
        lambda s: {
            "protein_id": "prot-5",
            "removed": ["prot-5"],
            "detached_targets": ["prot-4", "prot-7"],
            "unpaired_images": [],
            "unfitted_membranes": [],
            "removed_undetected": [],
        },
    ),
    (
        # A second loading control is added: the targets using p53 implicitly are pinned.
        lambda s: ops.add_protein(
            s, "α-tubulin", Role.LOADING_CONTROL, "img-3", box_size=BoxSize(width=10, height=6)
        ),
        "add_protein",
        lambda s: _protein(
            "prot-8",
            "α-tubulin",
            "loading control",
            "img-3",
            box_size={"width": 10, "height": 6},
            pinned_targets=["prot-4", "prot-7"],
        ),
    ),
    (
        # A target turned into a loading control loses its list: an implicit change.
        lambda s: ops.edit_protein(s, "prot-7", role=Role.LOADING_CONTROL, expected_mw=42),
        "edit_protein",
        lambda s: {
            "protein_id": "prot-7",
            "pinned_targets": [],
            "role": "loading control",
            "expected_mw": 42.0,
            "loading_control_ids": [],
            "dropped_undetected": [],
        },
    ),
    (
        lambda s: ops.place_box(s, "prot-4", NARROW_X, ROW, lane_index=0, grow=True),
        "place_box",
        lambda s: {
            "band_id": "band-9",
            "protein_id": "prot-4",
            "x": NARROW_X,
            "y": ROW,
            "grow": True,
            "lane_index": 0,
            "lane_proposed": False,
            "rect": _rect_of(s, "band-9"),
            "box_size": _size_of(s, "prot-4"),  # the first seed click sets it
            "replaced_undetected": None,
        },
    ),
    (
        lambda s: ops.place_box(s, "prot-4", WIDE_X, ROW, lane_index=1, grow=True),
        "place_box",
        lambda s: {
            "band_id": "band-10",
            "protein_id": "prot-4",
            "x": WIDE_X,
            "y": ROW,
            "grow": True,
            "lane_index": 1,
            "lane_proposed": False,
            "rect": _rect_of(s, "band-10"),
            "box_size": _size_of(s, "prot-4"),  # grown by the wide band
            "replaced_undetected": None,
        },
    ),
    (
        lambda s: ops.place_box(s, "prot-4", 150, ROW, grow=False),
        "place_box",
        lambda s: {
            "band_id": "band-11",
            "protein_id": "prot-4",
            "x": 150,
            "y": ROW,
            "grow": False,
            "lane_index": 2,  # proposed from the two boxes before
            "lane_proposed": True,
            "rect": list(boxes.centered_rect(150, ROW, protein_of(s, "prot-4").box_size, W, H)),
            "box_size": _size_of(s, "prot-4"),
            "replaced_undetected": None,
        },
    ),
    (
        lambda s: ops.move_box(s, "band-9", (30, 44, 10, 20)),
        "move_box",
        lambda s: {  # after the centre snap, not as dragged
            "band_id": "band-9",
            "rect": list(boxes.centered_rect(20, 32, protein_of(s, "prot-4").box_size, W, H)),
        },
    ),
    (
        lambda s: ops.set_box_size(s, "prot-4", BoxSize(width=12, height=6)),
        "set_box_size",
        lambda s: {"protein_id": "prot-4", "box_size": {"width": 12, "height": 6}},
    ),
    (
        lambda s: ops.set_polarity(s, "img-1", LIGHT),
        "set_polarity",
        lambda s: {"image_id": "img-1", "polarity": "light_on_dark", "dropped_undetected": []},
    ),
    (
        lambda s: ops.remove_box(s, "band-10"),
        "remove_box",
        lambda s: {"band_id": "band-10", "protein_id": "prot-4", "lane_index": 1},
    ),
    (
        lambda s: ops.set_box_lane(s, "band-11", 1),  # into the lane band-10 left
        "set_box_lane",
        lambda s: {
            "band_id": "band-11",
            "protein_id": "prot-4",
            "from_lane": 2,
            "lane_index": 1,
            "replaced_undetected": None,
        },
    ),
    (
        lambda s: ops.remove_image(s, "img-1"),
        "remove_image",
        lambda s: {
            "image_id": "img-1",
            "removed": ["img-1", "prot-4", "prot-6", "prot-7", "band-9", "band-11"],
            "detached_targets": [],
            "unpaired_images": [],
            "unfitted_membranes": [],
            "removed_undetected": [],
            "dropped_undetected": [],
        },
    ),
]


def _run_logged_steps(tmp_path: Path, check) -> ProjectSession:
    """Run :data:`LOGGED_STEPS` with autosave; ``check(s, action, expected)`` after each."""
    s = session_on(tmp_path, save_to_folder)
    for call, action, expected in LOGGED_STEPS:
        length = len(s.project.log)
        call(s)
        assert len(s.project.log) == length + 1, action
        check(s, action, expected)
    return s


def test_set_box_lane_changes_the_stored_lane_not_the_box(tmp_path):
    s, _, protein = boxed(tmp_path)
    a = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    b = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=False)
    before = band_of(s, a)
    ops.set_box_lane(s, a, 2)
    after = band_of(s, a)
    assert (after.lane_index, after.manually_edited) == (2, True)
    assert (after.box, after.net, after.clipped) == (before.box, before.net, before.clipped)
    entry = s.project.log[-1]
    assert (entry.action, entry.params["from_lane"], entry.params["lane_index"]) == (
        "set_box_lane",
        0,
        2,
    )
    committed = s.project
    ops.set_box_lane(s, a, 2)  # the same lane: a no-op
    assert s.project is committed

    with pytest.raises(OperationError) as info:
        ops.set_box_lane(s, a, 1)
    assert (info.value.code, info.value.ids) == (ErrorCode.LANE_OCCUPIED, (b,))
    for lane in (3, -1):
        with pytest.raises(OperationError) as info:
            ops.set_box_lane(s, a, lane)
        assert info.value.code is ErrorCode.LANE_OUT_OF_RANGE
    with pytest.raises(OperationError) as info:
        ops.set_box_lane(s, a, "1")
    assert info.value.code is ErrorCode.INVALID_INPUT
    with pytest.raises(UnknownIdError):
        ops.set_box_lane(s, "band-999", 0)
    assert s.project is committed


def test_each_operation_logs_its_params(tmp_path):
    def check(s: ProjectSession, action: str, expected) -> None:
        entry = s.project.log[-1]
        assert (entry.action, entry.params) == (action, expected(s))

    s = _run_logged_steps(tmp_path, check)
    first = s.project.log[0]
    assert (first.action, first.params) == ("new_project", {})
    assert {entry.action for entry in s.project.log} == {
        "new_project",
        "import_image",
        "remove_image",
        "set_polarity",
        "set_lanes",
        "set_reference_condition",
        "add_protein",
        "edit_protein",
        "remove_protein",
        "place_box",
        "move_box",
        "remove_box",
        "set_box_lane",
        "set_box_size",
    }
    text = (s.folder / storage.PROJECT_FILE).read_text(encoding="utf-8")
    for leak in (str(tmp_path), json.dumps(str(tmp_path))[1:-1], FOLDER):
        assert leak not in text  # no path is ever recorded


def test_each_entry_hashes_the_content_it_left(tmp_path):
    def check(s: ProjectSession, action: str, expected) -> None:
        assert s.project.log[-1].content_hash == content_hash(s.project), action
        assert history_issues(s.project) == [], action

    s = _run_logged_steps(tmp_path, check)
    assert load_project(s.folder).log == s.project.log


def test_the_entry_is_saved_with_its_change(tmp_path, replace_lock):
    s, image, protein = boxed(tmp_path, save_to_folder)
    assert load_project(s.folder).log == s.project.log
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert load_project(s.folder).log == s.project.log
    ops.set_polarity(s, image, LIGHT)
    assert load_project(s.folder).log == s.project.log

    path = s.folder / storage.PROJECT_FILE
    before = path.read_bytes()
    replace_lock.locked = True
    ops.move_box(s, band, (50, 20, 70, 40))
    assert s.project.log[-1].action == "move_box" and s.dirty  # both in memory
    assert path.read_bytes() == before
    on_disk = load_project(s.folder)  # the older pair, still consistent
    assert on_disk.log == s.project.log[:-1]
    assert on_disk.log[-1].content_hash == content_hash(on_disk)

    replace_lock.locked = False
    ops.set_box_size(s, protein, BoxSize(width=11, height=7))  # saves both changes
    saved = load_project(s.folder)
    assert saved == s.project
    assert [entry.action for entry in saved.log[-2:]] == ["move_box", "set_box_size"]

    data = path.read_bytes()
    ops.save(ops.open_project(s.folder))
    assert path.read_bytes() == data  # reopened and saved: byte-identical


def test_equal_content_hashes_equal_whatever_the_log(tmp_path):
    source = write_tiff(tmp_path / "sources" / "β-actin 10 µM.tif", blot())
    taipei = timezone(timedelta(hours=8))

    def build(name: str, clock: FakeClock, *, detour: bool) -> ProjectSession:
        s = ops.new_project(tmp_path / name, autosave=save_to_folder, clock=clock)
        with source.open("rb") as f:
            image = ops.import_image(s, f, source.name, kind=CHEMI, polarity=DARK)
        ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
        if detour:  # there and back: the log grows, the content does not change
            ops.set_reference_condition(s, "10 µM")
            ops.set_reference_condition(s, None)
        protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
        ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
        return s

    a = build("a α", FakeClock(), detour=False)
    b = build("b β", FakeClock(start=datetime(2030, 1, 1, 9, 0, tzinfo=taipei)), detour=True)
    assert content_hash(a.project) == content_hash(b.project)
    assert a.project.log != b.project.log
    assert len(b.project.log) == len(a.project.log) + 2
    assert b.project.log[0].time == "2030-01-01T01:00:00.000Z"  # stored in UTC
    project_json = storage.PROJECT_FILE
    assert (a.folder / project_json).read_bytes() != (b.folder / project_json).read_bytes()

    hashed, length = content_hash(a.project), len(a.project.log)
    ops.set_lanes(a, [LaneInput("vehicle", included=False), LaneInput("10 µM")])
    assert content_hash(a.project) != hashed
    ops.set_lanes(a, [LaneInput("vehicle"), LaneInput("10 µM")])
    assert content_hash(a.project) == hashed
    assert len(a.project.log) == length + 2


def test_content_hash_changes_when_a_box_moves_or_a_lane_is_excluded(tmp_path):
    s, _, protein = boxed(tmp_path)
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    hashes = [content_hash(s.project)]
    ops.move_box(s, band, (50, 20, 70, 40))
    hashes.append(content_hash(s.project))
    lanes = [LaneInput("vehicle", included=False), LaneInput("10 µM"), LaneInput("50 µM")]
    ops.set_lanes(s, lanes)
    hashes.append(content_hash(s.project))
    assert len(set(hashes)) == 3


def test_a_naive_clock_commits_nothing(tmp_path):
    recorder = Recorder()
    s = session_on(tmp_path, recorder)
    before = s.project
    s.clock = lambda: datetime(2026, 9, 26)  # no time zone
    with pytest.raises(ValueError, match="aware") as info:
        ops.set_lanes(s, [LaneInput("vehicle")])
    assert not isinstance(info.value, OperationError)  # a bug, not a refusal
    assert s.project is before
    assert recorder.actions == []


def test_an_import_that_fails_to_commit_leaves_no_file(tmp_path):
    recorder = Recorder()
    s = session_on(tmp_path, recorder)
    before = s.project
    s.clock = lambda: datetime(2026, 9, 26)  # no time zone: _commit raises
    with pytest.raises(ValueError, match="aware"):
        import_blot(s, blot())
    assert listing(s) == []
    assert s.project is before
    assert recorder.actions == []


def test_a_stale_draft_is_refused(tmp_path):
    recorder = Recorder()
    s = session_on(tmp_path, recorder)
    stale = s.project
    ops.set_lanes(s, [LaneInput("vehicle")])
    before = s.project
    old_draft, _ = apply_change(stale, lambda draft: draft.new_id("band"))
    # Prepared from an older project, and a copy whose log is not the committed one.
    for draft in (old_draft, revalidate(before)):
        with pytest.raises(RuntimeError, match="stale project"):
            s._commit(draft, action="plant", params={})
        assert s.project is before
    assert recorder.actions == ["set_lanes"]


def test_export_writes_the_record(tmp_path):
    recorder = Recorder()
    s = open_sample(tmp_path, recorder)
    project_json = (s.folder / storage.PROJECT_FILE).read_bytes()
    log = s.project.log
    table = ops.export_lane_table(s).read_bytes()
    data = (s.folder / "exports" / "lane-table.record.json").read_bytes()

    assert not data.startswith(codecs.BOM_UTF8)
    assert b"\r" not in data and data.endswith(b"}\n")
    for text in ("µ", "α", "β"):
        assert text.encode() in data  # raw UTF-8
    assert b"\\u" not in data
    doc = json.loads(data.decode("utf-8"))
    assert record.record_bytes(doc) == data  # the canonical file form

    assert doc["files"] == {
        "lane-table.csv": {"sha256": hashlib.sha256(table).hexdigest(), "bytes": len(table)}
    }
    assert doc["content_hash"] == content_hash(s.project)
    assert hashlib.sha256(canonical_json(doc["content"])).hexdigest() == doc["content_hash"]
    assert doc["log"] == [entry.model_dump(mode="json") for entry in s.project.log]
    assert doc["exported_at"] == "2026-09-26T08:00:00.000Z"
    assert doc["history_issues"] == ["no_history"]  # the sample was saved without a log
    assert doc["results"] is None  # the lane table uses no compute settings
    assert doc["settings"] == record.settings()
    text = data.decode("utf-8")
    for leak in (str(tmp_path), json.dumps(str(tmp_path))[1:-1], FOLDER):
        assert leak not in text

    assert recorder.actions == [] and s.project.log is log  # not a state change
    assert (s.folder / storage.PROJECT_FILE).read_bytes() == project_json


@pytest.mark.parametrize("problem", ["changed", "missing"])
def test_export_refuses_changed_or_missing_images(tmp_path, problem):
    s = open_sample(tmp_path)
    ops.export_lane_table(s)
    exports = s.folder / "exports"
    earlier = {path.name: path.read_bytes() for path in exports.iterdir()}
    assert set(earlier) == {"lane-table.csv", "lane-table.record.json"}
    image = s.folder / "images" / "img-2.tif"
    if problem == "changed":
        image.write_bytes(b"other pixels")
    else:
        image.unlink()
    with pytest.raises(OperationError) as info:
        ops.export_lane_table(s)
    assert info.value.code is ErrorCode.IMAGE_FILE_CHANGED
    assert info.value.ids == ("img-2",)
    assert {path.name: path.read_bytes() for path in exports.iterdir()} == earlier


def test_record_grow_settings_are_what_place_box_uses(tmp_path, monkeypatch):
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return grow_box(*args, **kwargs)

    monkeypatch.setattr(ops, "grow_box", spy)
    s, _, protein = boxed(tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    [(args, kwargs)] = calls
    bound = inspect.signature(grow_box).bind(*args, **kwargs)
    bound.apply_defaults()
    recorded = record.settings()["grow_box"]
    assert {name: bound.arguments[name] for name in recorded} == recorded


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("error_type", "sd", "error type must be one of 'SD', 'SEM', not 'sd'"),
        ("method", "median", "method must be one of 'mean', 'representative', not 'median'"),
    ],
)
def test_compute_refuses_an_unknown_error_type_or_method(tmp_path, argument, value, message):
    s = session_on(tmp_path)
    with pytest.raises(OperationError) as info:
        ops.compute(s, **{argument: value})
    assert info.value.code is ErrorCode.INVALID_INPUT
    assert str(info.value) == message


def test_compute_takes_the_raw_error_type_and_method(tmp_path):
    s = _parity_session(tmp_path, _blot(), DARK)
    raw = ops.compute(s, error_type="SEM", method="representative")
    assert raw == ops.compute(s, error_type=ErrorType.SEM, method=ReduceMethod.REPRESENTATIVE)


def test_compute_view_pairs_the_committed_project_with_its_results(tmp_path):
    s = _parity_session(tmp_path, _blot(), DARK)
    s._pixels.clear()
    log = s.project.log
    settings = {"plot_conditions": ["vehicle"], "error_type": "SEM", "method": "representative"}
    view = ops.compute_view(s, **settings)
    assert view.project is s.project
    assert view.results == results.compute_results(s.project.batch, **settings)
    assert view.results == ops.compute(s, **settings)  # compute is the view's results
    assert s.project.log is log and s._pixels == {}  # no log entry, no pixels read
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.project = None  # type: ignore[misc]
    with pytest.raises(OperationError) as info:
        ops.compute_view(s, error_type="sd")
    assert info.value.code is ErrorCode.INVALID_INPUT


def test_compute_view_reads_the_committed_project_once(tmp_path, monkeypatch):
    # A commit that lands while the results are computed (another request, since
    # compute_view takes no lock) is in neither half of the view.
    s = _parity_session(tmp_path, _blot(), DARK)
    before = s.project
    compute_results = results.compute_results

    def commit_meanwhile(batch, **settings):
        ops.set_reference_condition(s, None)
        return compute_results(batch, **settings)

    monkeypatch.setattr(results, "compute_results", commit_meanwhile)
    view = ops.compute_view(s)
    assert view.project is before and s.project is not before
    assert view.project.batch.reference_condition == view.results.reference_condition
    assert view.results.reference_condition is not None


# --- #51: not-detected records ---


def _record(
    lane: int,
    *,
    band_index: int = 0,
    snr: float = 1.5,
    region: tuple[int, int, int, int] = (60, 20, 80, 40),
    source: ProposalSource = ProposalSource.ROW_BOX,
) -> UndetectedBand:
    """A not-detected record, from row-box detection unless ``source`` says
    otherwise (the blot is 160x60)."""
    x0, y0, x1, y1 = region
    return UndetectedBand(
        lane_index=lane,
        band_index=band_index,
        reason=UndetectedReason.BELOW_DETECTION_LIMIT,
        snr=snr,
        threshold=6.0,
        region=Region(x0=x0, y0=y0, x1=x1, y1=y1),
        source=source,
    )


def plant_records(
    session: ProjectSession, protein_id: str, *records: UndetectedBand, bands: int = 1
) -> None:
    """Store records no operation writes yet (the row commit is #51's next step);
    ``bands`` is the protein's expected band count."""

    def change(draft: Project) -> None:
        protein = draft.batch.find_protein(protein_id)
        protein.expected_band_count = bands
        protein.undetected.extend(records)

    plant(session, change)


def _record_json(protein_id: str, u: UndetectedBand) -> dict:
    """A record as a log entry names it."""
    return {
        "protein_id": protein_id,
        "lane_index": u.lane_index,
        "band_index": u.band_index,
        "reason": "below_detection_limit",
        "snr": u.snr,
        "threshold": u.threshold,
        "region": list(u.region.rect()),
        "source": u.source.value,
    }


def _keys(session: ProjectSession, protein_id: str) -> list[tuple[int, int]]:
    return [(u.lane_index, u.band_index) for u in protein_of(session, protein_id).undetected]


def test_place_box_replaces_the_record_in_its_lane(tmp_path):
    s, _, protein = boxed(tmp_path)
    lane_1, lane_1_band_1, lane_2 = _record(1), _record(1, band_index=1), _record(2, snr=-0.5)
    plant_records(s, protein, lane_1, lane_1_band_1, lane_2, bands=2)
    length = len(s.project.log)

    band = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
    assert band_of(s, band).lane_index == 1
    assert _keys(s, protein) == [(1, 1), (2, 0)]  # only band index 0 in lane 1
    assert len(s.project.log) == length + 1  # one entry: the box and the dropped record
    entry = s.project.log[-1]
    assert entry.action == "place_box"
    assert entry.params["replaced_undetected"] == _record_json(protein, lane_1)
    assert entry.content_hash == content_hash(s.project)

    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert s.project.log[-1].params["replaced_undetected"] is None  # no record there
    assert _keys(s, protein) == [(1, 1), (2, 0)]
    [column] = ops.compute(s).proteins
    assert column.detected == [True, True, False]


def test_set_box_lane_replaces_the_record_of_its_new_lane(tmp_path):
    s, _, protein = boxed(tmp_path)
    a = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    lane_2 = _record(2)
    plant_records(s, protein, lane_2)

    ops.set_box_lane(s, a, 2)
    assert _keys(s, protein) == []  # and lane 0, which the box left, gets no record
    assert s.project.log[-1].params == {
        "band_id": a,
        "protein_id": protein,
        "from_lane": 0,
        "lane_index": 2,
        "replaced_undetected": _record_json(protein, lane_2),
    }
    ops.set_box_lane(s, a, 1)
    assert s.project.log[-1].params["replaced_undetected"] is None
    [column] = ops.compute(s).proteins
    assert column.detected == [None, True, None]


def test_set_box_lane_replaces_only_the_record_of_the_moved_band(tmp_path):
    s, _, protein = boxed(tmp_path)
    a = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    plant_records(s, protein, _record(2), bands=2)
    # The box becomes the second band's (#58 stores them; a loaded file may hold one).
    plant(s, lambda draft: setattr(draft.batch.find_band(a)[1], "band_index", 1))

    ops.set_box_lane(s, a, 2)
    assert _keys(s, protein) == [(2, 0)]  # the first band's record stays
    assert s.project.log[-1].params["replaced_undetected"] is None

    lane_1_band_1 = _record(1, band_index=1)
    plant_records(s, protein, _record(1), lane_1_band_1, bands=2)
    ops.set_box_lane(s, a, 1)
    assert _keys(s, protein) == [(1, 0), (2, 0)]
    assert s.project.log[-1].params["replaced_undetected"] == _record_json(protein, lane_1_band_1)


def test_set_polarity_drops_the_records_on_that_image(tmp_path):
    s = session_on(tmp_path)
    first = import_blot(s, blot(), "chemi α.tif")
    second = import_blot(s, blot(), "chemi β.tif")
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM"), LaneInput("50 µM")])
    beta = ops.add_protein(s, "β-catenin", Role.TARGET, first)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, first)  # records only, no box
    tubulin = ops.add_protein(s, "α-tubulin", Role.LOADING_CONTROL, second)
    ops.place_box(s, beta, NARROW_X, ROW, lane_index=0, grow=True)
    beta_records = [_record(2, snr=-0.5), _record(1)]
    gapdh_record = _record(0, snr=3.0)
    plant_records(s, beta, *beta_records)
    plant_records(s, gapdh, gapdh_record)
    plant_records(s, tubulin, _record(2))
    kept = protein_of(s, tubulin).undetected

    ops.set_polarity(s, first, LIGHT)
    assert (_keys(s, beta), _keys(s, gapdh)) == ([], [])
    assert protein_of(s, tubulin).undetected == kept  # another image's records stay
    assert s.project.log[-1].params == {
        "image_id": first,
        "polarity": "light_on_dark",
        # Protein order, then lane order.
        "dropped_undetected": [
            _record_json(beta, beta_records[1]),
            _record_json(beta, beta_records[0]),
            _record_json(gapdh, gapdh_record),
        ],
    }
    ops.set_polarity(s, first, DARK)
    assert s.project.log[-1].params["dropped_undetected"] == []
    assert_nets_current(s)


def test_set_polarity_drops_the_records_on_an_image_without_boxes(tmp_path):
    # No protein holds a box on the image, so no net is recomputed; the records
    # still go, since their SNR was measured with the other signal direction.
    s, image, beta = boxed(tmp_path)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    beta_record, gapdh_record = _record(1), _record(2, snr=-0.5)
    plant_records(s, beta, beta_record)
    plant_records(s, gapdh, gapdh_record)
    length = len(s.project.log)

    ops.set_polarity(s, image, LIGHT)
    assert (_keys(s, beta), _keys(s, gapdh)) == ([], [])
    assert len(s.project.log) == length + 1
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "set_polarity",
        {
            "image_id": image,
            "polarity": "light_on_dark",
            "dropped_undetected": [
                _record_json(beta, beta_record),
                _record_json(gapdh, gapdh_record),
            ],
        },
    )
    assert entry.content_hash == content_hash(s.project)


def test_set_lanes_drops_the_records_of_the_lanes_it_cuts(tmp_path):
    s, _, protein = boxed(tmp_path)
    four = [LaneInput("vehicle"), LaneInput("10 µM"), LaneInput("50 µM"), LaneInput("100 µM")]
    ops.set_lanes(s, four)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
    last = ops.place_box(s, protein, 140, ROW, lane_index=3, grow=False)
    lane_1, lane_2 = _record(1), _record(2, snr=-1.0)
    plant_records(s, protein, lane_2, lane_1)

    # Boxes still block, and a refusal leaves the records as they were.
    before = s.project
    with pytest.raises(OperationError) as info:
        ops.set_lanes(s, four[:2])
    assert (info.value.code, info.value.ids) == (ErrorCode.LANES_IN_USE, (last,))
    assert s.project is before

    ops.remove_box(s, last)
    update = ops.set_lanes(s, four[:2])
    assert update == LanesUpdate(
        respelled=(), reference_cleared=False, dropped_undetected=((protein, 2, 0),)
    )
    assert _keys(s, protein) == [(1, 0)]  # a kept lane keeps its record
    params = s.project.log[-1].params
    assert (params["lane_count"], params["dropped_undetected"]) == (
        2,
        [_record_json(protein, lane_2)],
    )

    update = ops.set_lanes(s, four[:3])  # the table grows: nothing to drop
    assert update.dropped_undetected == ()
    assert s.project.log[-1].params["dropped_undetected"] == []
    assert _keys(s, protein) == [(1, 0)]


def test_remove_undetected(tmp_path):
    s, _, protein = boxed(tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    lane_1, lane_0_band_1 = _record(1), _record(0, band_index=1)
    plant_records(s, protein, lane_1, lane_0_band_1, bands=2)

    ops.remove_undetected(s, protein, 1)
    assert _keys(s, protein) == [(0, 1)]
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "remove_undetected",
        {
            "protein_id": protein,
            "lane_index": 1,
            "band_index": 0,
            "removed": _record_json(protein, lane_1),
        },
    )
    [column] = ops.compute(s).proteins
    assert column.detected == [True, None, None]  # the lane is now "not measured"

    committed = s.project
    ops.remove_undetected(s, protein, 1)  # nothing there any more: a no-op
    ops.remove_undetected(s, protein, 0)  # band index 0 of lane 0 is a band, not a record
    assert s.project is committed

    ops.remove_undetected(s, protein, 0, band_index=1)
    assert _keys(s, protein) == []
    assert s.project.log[-1].params["removed"] == _record_json(protein, lane_0_band_1)

    committed = s.project
    for lane, band_index, code in [
        (3, 0, ErrorCode.LANE_OUT_OF_RANGE),
        (-1, 0, ErrorCode.LANE_OUT_OF_RANGE),
        ("1", 0, ErrorCode.INVALID_INPUT),
        (1, -1, ErrorCode.INVALID_INPUT),
        (1, True, ErrorCode.INVALID_INPUT),
    ]:
        with pytest.raises(OperationError) as info:
            ops.remove_undetected(s, protein, lane, band_index=band_index)
        assert info.value.code is code, (lane, band_index)
    with pytest.raises(UnknownIdError):
        ops.remove_undetected(s, "prot-999", 0)
    assert s.project is committed


def test_remove_box_leaves_the_lane_not_measured(tmp_path):
    s, _, protein = boxed(tmp_path)
    plant_records(s, protein, _record(1))
    band = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)  # replaces it
    ops.remove_box(s, band)
    assert _keys(s, protein) == []  # no record is created, and the old one does not return
    assert s.project.log[-1].params == {"band_id": band, "protein_id": protein, "lane_index": 1}
    [column] = ops.compute(s).proteins
    assert column.detected == [None, None, None]


def test_move_and_resize_keep_the_records(tmp_path):
    s, _, protein = boxed(tmp_path)
    band = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    plant_records(s, protein, _record(2))
    kept = protein_of(s, protein).undetected
    ops.move_box(s, band, (30, 22, 50, 38))
    ops.set_box_size(s, protein, BoxSize(width=12, height=6))
    ops.edit_protein(s, protein, name="β-catenin (E-5)", expected_mw=92)
    assert protein_of(s, protein).undetected == kept


def _records_json(session: ProjectSession, protein_id: str) -> list[dict]:
    return [_record_json(protein_id, u) for u in protein_of(session, protein_id).undetected]


def test_removals_take_the_records_along(tmp_path):
    s = open_sample(tmp_path, project=make_project_with_undetected())
    assert _keys(s, "prot-9") == [(2, 0), (3, 0)]
    gapdh = _records_json(s, "prot-9")
    cascade = ops.remove_protein(s, "prot-9")  # GAPDH: records only in lanes 2 and 3
    assert cascade.removed == ("prot-9", "band-17", "band-18")  # records have no ids
    # The log keeps them whole, since they have no ids to name them by.
    assert s.project.log[-1].params == {
        "protein_id": "prot-9",
        **_listed(cascade),
        "removed_undetected": gapdh,
    }
    beta = _records_json(s, "prot-7")
    cascade = ops.remove_image(s, "img-2")  # β-catenin, with its lane-2 record
    assert cascade.removed == ("img-2", "prot-7", "band-10", "band-11", "band-12")
    assert s.project.log[-1].params == {
        "image_id": "img-2",
        **_listed(cascade),
        "removed_undetected": beta,
        "dropped_undetected": [],
    }
    assert all(not p.undetected for p in s.project.batch.proteins)
    saved = (s.folder / storage.PROJECT_FILE).read_bytes()
    assert b'"undetected"' not in saved  # an empty list is never written
    assert load_project(s.folder) == s.project


# --- review of PR #86 ---


def test_edit_protein_drops_mw_guided_records_when_the_expected_mw_changes(tmp_path):
    s, image, beta = boxed(tmp_path)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    guided, row = _record(1, source=ProposalSource.MW_GUIDED), _record(2, snr=-0.5)
    plant_records(s, beta, guided, row)
    plant_records(s, gapdh, _record(0, source=ProposalSource.MW_GUIDED))

    ops.edit_protein(s, beta, name="β-catenin (E-5)")  # the MW is kept, and so are the records
    assert _keys(s, beta) == [(1, 0), (2, 0)]
    assert s.project.log[-1].params == {
        "protein_id": beta,
        "pinned_targets": [],
        "name": "β-catenin (E-5)",
        "dropped_undetected": [],
    }

    length = len(s.project.log)
    ops.edit_protein(s, beta, expected_mw=92)
    assert _keys(s, beta) == [(2, 0)]  # a row-box record does not depend on the MW
    assert _keys(s, gapdh) == [(0, 0)]  # nor does another protein's
    assert len(s.project.log) == length + 1  # one entry: the edit and the dropped record
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "edit_protein",
        {
            "protein_id": beta,
            "pinned_targets": [],
            "expected_mw": 92.0,
            "dropped_undetected": [_record_json(beta, guided)],
        },
    )
    assert entry.content_hash == content_hash(s.project)

    plant_records(s, beta, guided)
    committed = s.project
    ops.edit_protein(s, beta, expected_mw=92)  # the same MW: a no-op, the record stays
    assert s.project is committed
    ops.edit_protein(s, beta, expected_mw=None)  # cleared: the searched slot means nothing
    assert _keys(s, beta) == [(2, 0)]
    assert s.project.log[-1].params["dropped_undetected"] == [_record_json(beta, guided)]


def test_a_calibration_change_drops_the_membranes_mw_guided_records(tmp_path):
    s = open_sample(tmp_path / "marker", project=make_project_with_undetected())
    # GAPDH (img-4, mem-1): a row-box record in lane 2, an MW-guided one in lane 3.
    [_, gapdh_guided] = _records_json(s, "prot-9")
    # α-tubulin is on mem-5, whose calibration does not change.
    tubulin = _record(0, band_index=1, region=(0, 10, 20, 30), source=ProposalSource.MW_GUIDED)
    plant_records(s, "prot-8", tubulin, bands=2)

    cascade = ops.remove_image(s, "img-3")  # the marker, with two calibration points
    assert cascade.unfitted_membranes == ("mem-1",)
    assert _keys(s, "prot-9") == [(2, 0)]
    assert _keys(s, "prot-7") == [(2, 0)]  # a row-box record stays
    assert _keys(s, "prot-8") == [(0, 1)]  # another membrane's stays
    entry = s.project.log[-1]
    assert entry.params == {
        "image_id": "img-3",
        **_listed(cascade),
        "removed_undetected": [],
        "dropped_undetected": [gapdh_guided],
    }
    assert entry.content_hash == content_hash(s.project)

    # An image without calibration points leaves the calibration, and the MW-guided
    # records of the membrane's other proteins, as they were.
    s = open_sample(tmp_path / "reprobe", project=make_project_with_undetected())

    def guided(draft: Project) -> None:
        draft.batch.find_protein("prot-7").undetected[0].source = ProposalSource.MW_GUIDED

    plant(s, guided)
    gapdh = _records_json(s, "prot-9")
    cascade = ops.remove_image(s, "img-4")  # GAPDH's image
    assert cascade.unfitted_membranes == ()
    assert _keys(s, "prot-7") == [(2, 0)]
    assert s.project.log[-1].params == {
        "image_id": "img-4",
        **_listed(cascade),
        "removed_undetected": gapdh,
        "dropped_undetected": [],
    }


def test_dropping_records_asks_the_predicate_once_per_record(tmp_path):
    s, _, protein = boxed(tmp_path)
    plant_records(s, protein, _record(0), _record(1), _record(2))
    draft = s.project.model_copy(deep=True)
    asked: list[int] = []

    def drop(u: UndetectedBand) -> bool:
        asked.append(u.lane_index)
        return u.lane_index == 1

    dropped = ops._drop_undetected_where(draft.batch.find_protein(protein), drop)
    assert asked == [0, 1, 2]
    assert dropped == [_record_json(protein, _record(1))]
    assert [u.lane_index for u in draft.batch.find_protein(protein).undetected] == [0, 2]


# --- #51: committing a dragged row ---

ROWS = {case.name: case for case in bench_cases()}
# Lane 0 is empty. The same image under a box that stops 40 px short of it
# leaves that lane's expected centre outside the box; 70 px short (the
# adversarial case box_omits_empty_first), the bands fit the lanes two ways.
SCENE = adversarial_row("scene", 1000, missing=[0])
UNCLEAR_ROW = (SCENE.row[0] + 40, *SCENE.row[1:])
AMBIGUOUS_ROW = (SCENE.row[0] + 70, *SCENE.row[1:])
SCENE_W = SCENE.image.shape[1]


def _lane_inputs(n: int, excluded: tuple[int, ...] = ()) -> list[LaneInput]:
    """Two conditions in alternate lanes, one sample per pair of lanes."""
    return [
        LaneInput(
            "vehicle" if i % 2 == 0 else "10 µM", f"rep {i // 2 + 1}", included=i not in excluded
        )
        for i in range(n)
    ]


def row_session(
    tmp_path: Path, case: RowCase, *, excluded: tuple[int, ...] = (), hook=None
) -> tuple[ProjectSession, str, str]:
    """The case's image imported as a 16-bit TIFF, its lanes declared, and one
    target without boxes."""
    s = session_on(tmp_path, hook)
    polarity = DARK if case.dark_on_light else LIGHT
    image = import_blot(s, case.image.astype(np.uint16), "row α.tif", polarity)
    ops.set_lanes(s, _lane_inputs(case.n_lanes, excluded))
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    return s, image, protein


def detected(s: ProjectSession, protein_id: str, row) -> RowDetection:
    """What detection finds in ``row`` on the stored pixels, called as the
    operation calls it."""
    batch = s.project.batch
    image = batch.find_image(batch.find_protein(protein_id).image_id)
    return rowdetect.detect_row(
        s.pixels(image.id),
        row,
        len(batch.lanes),
        background=image.background,
        dark_on_light=image.polarity.dark_on_light,
    )


def lane_bands(s: ProjectSession, protein_id: str) -> dict[int, Band]:
    """The protein's band-index-0 boxes by lane."""
    return {b.lane_index: b for b in protein_of(s, protein_id).bands if b.band_index == 0}


def rects_by_lane(s: ProjectSession, protein_id: str) -> dict[int, tuple[int, int, int, int]]:
    size = protein_of(s, protein_id).box_size
    return {lane: band.box.rect(size) for lane, band in lane_bands(s, protein_id).items()}


def at_lane(case: RowCase, lane: int) -> tuple[int, int]:
    """A point on the lane's band (its true centre)."""
    return round(case.lane_cx[lane]), round(case.lane_cy[lane])


def edit_by_hand(s: ProjectSession, band_id: str, *, to: tuple[int, int] | None = None) -> None:
    """Move a box 1 px to the right, or centre it on ``to``: the user edited it."""
    protein, band = s.project.batch.find_band(band_id)
    x0, y0, x1, y1 = band.box.rect(protein.box_size)
    rect = (x0 + 1, y0, x1 + 1, y1) if to is None else (*to, *to)
    ops.move_box(s, band_id, rect)
    assert band_of(s, band_id).manually_edited


def centred(s: ProjectSession, image_id: str, rect, size: BoxSize) -> tuple[int, int, int, int]:
    """A box of ``size`` on ``rect``'s integer centre, shifted inside the image."""
    image = s.project.batch.find_image(image_id)
    return boxes.center_snap(rect, size, image.width, image.height)


def row_case(key: str) -> RowCase:
    """A bench row by name, or an adversarial recipe at seed 1000."""
    return ROWS[key] if key in ROWS else adversarial(key, 1000)


def placed_box(
    s: ProjectSession, protein_id: str, case: RowCase, lane: int, source: ProposalSource
) -> str:
    """A box of the protein's size on the lane's centre that nobody edited,
    placed as ``source`` says: by the user (``manual``, or a grown ``click``) or
    by a detector (``row_box``, ``mw_guided``)."""
    band_id = ops.place_box(s, protein_id, *at_lane(case, lane), lane_index=lane, grow=False)
    if source is not ProposalSource.MANUAL:
        plant(s, lambda draft: setattr(draft.batch.find_band(band_id)[1], "source", source))
    assert not band_of(s, band_id).manually_edited
    return band_id


def other_protein_in_lanes(
    s: ProjectSession,
    image_id: str,
    case: RowCase,
    lanes: Sequence[int] | None = None,
    *,
    dx: float = 0.0,
    mirrored: bool = False,
    name: str = "GAPDH",
    offset: int = 0,
) -> str:
    """A second protein on the image with a small box in each of ``lanes``
    (every lane by default), at the lane's true centre moved ``dx`` px, in a row
    of its own above the case's: lanes already placed on the image. Mirrored,
    the lanes are numbered right to left; ``offset`` is added to each lane
    number (a protein numbered a lane off)."""
    other = ops.add_protein(
        s, name, Role.LOADING_CONTROL, image_id, box_size=BoxSize(width=20, height=8)
    )
    n = case.n_lanes
    for lane in range(n) if lanes is None else lanes:
        index = (n - 1 - lane if mirrored else lane) + offset
        ops.place_box(s, other, round(case.lane_cx[lane] + dx), 20, lane_index=index, grow=False)
    return other


def shifted(case: RowCase, dx: int) -> tuple[int, int, int, int]:
    """The case's row box moved ``dx`` px along x, clipped to the image."""
    x0, y0, x1, y1 = case.row
    return max(0, x0 + dx), y0, min(case.image.shape[1], x1 + dx), y1


@pytest.mark.parametrize("name", ["all_present", "light_on_dark"])
def test_a_row_places_one_box_per_lane(tmp_path, name):
    case = ROWS[name]
    s, _, protein = row_session(tmp_path, case, hook=save_to_folder)
    found = detected(s, protein, case.row)
    length = len(s.project.log)

    placement = ops.detect_row_boxes(s, protein, case.row)
    bands = lane_bands(s, protein)
    assert rects_by_lane(s, protein) == dict(enumerate(found.slots))
    assert protein_of(s, protein).box_size == found.size
    for band in bands.values():
        assert (band.source, band.manually_edited, band.apparent_mw) == (
            ProposalSource.ROW_BOX,
            False,
            None,
        )
    assert placement == ops.RowPlacement(
        band_ids=tuple(bands[lane].id for lane in range(6)),
        box_size=found.size,
        kept_lanes=(),
        replaced_band_ids=(),
        removed_band_ids=(),
        undetected_lanes=(),
        unmeasured_lanes=(),
        empty=(),
        flags=(),
        notes=(),
        right_to_left=False,
    )
    numbers = [int(band_id.removeprefix("band-")) for band_id in placement.band_ids]
    assert numbers == list(range(numbers[0], numbers[0] + 6))  # new ids in lane order
    assert_nets_current(s)
    assert len(s.project.log) == length + 1  # one change, one entry
    entry = s.project.log[-1]
    assert entry.action == "detect_row_boxes"
    assert entry.content_hash == content_hash(s.project)
    assert load_project(s.folder) == s.project  # autosaved with its entry
    [column] = ops.compute(s).proteins
    assert column.detected == [True] * 6


def test_lanes_without_a_band_get_a_not_detected_record(tmp_path):
    case = ROWS["missing_two"]  # lanes 0 and 3 are empty
    s, _, protein = row_session(tmp_path, case)
    found = detected(s, protein, case.row)

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert sorted(lane_bands(s, protein)) == [1, 2, 4, 5]
    records = protein_of(s, protein).undetected
    assert [(u.lane_index, u.band_index) for u in records] == [(0, 0), (3, 0)]
    for u in records:
        lane = found.lanes[u.lane_index]
        assert lane.reason == "no_band"
        assert u.reason is UndetectedReason.BELOW_DETECTION_LIMIT
        assert (u.snr, u.threshold) == (lane.snr, DETECT_K)
        assert u.snr < u.threshold
        assert u.region.rect() == lane.window  # the slot the detector measured
        assert u.source is ProposalSource.ROW_BOX
    assert (placement.band_ids[0], placement.band_ids[3]) == (None, None)
    assert (placement.undetected_lanes, placement.unmeasured_lanes) == ((0, 3), ())
    assert placement.empty == tuple(
        (i, "no_band", found.lanes[i].snr, found.lanes[i].expected_x) for i in (0, 3)
    )
    params = s.project.log[-1].params
    assert params["undetected_written"] == [_record_json(protein, u) for u in records]
    assert params["dropped_undetected"] == []

    res = ops.compute(s)
    [column] = res.proteins
    assert column.detected == [False, True, True, False, True, True]
    assert (column.nets[0], column.nets[3]) == (None, None)
    [notice] = [n for n in res.notices if n.code is NoticeCode.BELOW_DETECTION]
    assert notice.level is Level.WARNING
    assert (notice.protein_ids, notice.lane_indices, notice.conditions) == (
        (protein,),
        (0, 3),
        ("vehicle", "10 µM"),
    )
    assert notice.message == (
        "'β-catenin' was not detected in lanes 1, 4 (below the detection limit):"
        " those lanes have no value and are left out of the statistics"
    )


# Adversarial rows whose one empty lane the detector cannot read.
UNREADABLE_LANES = [
    ("blotch_empty", 4, "artefact"),  # a stain over the empty lane
    ("vstreak_empty", 4, "unassigned"),  # a streak kept, but in no piece
    ("nbr_above_miss", 2, "edge_signal"),  # only the neighbouring row reaches the limit
]


@pytest.mark.parametrize(("key", "lane", "reason"), UNREADABLE_LANES)
def test_other_empty_lanes_are_left_not_measured(tmp_path, key, lane, reason):
    # Only "nothing reached the detection limit" is a not-detected claim. Any
    # other empty lane gets no record, and an older record there goes: this
    # run's outcome replaces it.
    case = adversarial(key, 1000)
    s, _, protein = row_session(tmp_path, case)
    old = _record(lane)
    plant_records(s, protein, old)

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert protein_of(s, protein).undetected == []
    assert sorted(lane_bands(s, protein)) == [i for i in range(6) if i != lane]
    assert placement.band_ids[lane] is None
    assert (placement.undetected_lanes, placement.unmeasured_lanes) == ((), (lane,))
    [(empty_lane, empty_reason, _, _)] = placement.empty
    assert (empty_lane, empty_reason) == (lane, reason)
    params = s.project.log[-1].params
    assert params["lanes"][lane]["reason"] == reason
    assert (params["undetected_written"], params["dropped_undetected"]) == (
        [],
        [_record_json(protein, old)],
    )
    assert ops.compute(s).proteins[0].detected[lane] is None


# Rows whose one empty lane holds nothing that reaches the detection limit, or
# something the detector cannot read.
EMPTY_LANES = [
    pytest.param("missing_middle", 2, "no_band", id="no_band"),
    *(pytest.param(*lane, id=lane[2]) for lane in UNREADABLE_LANES),
]


@pytest.mark.parametrize("source", [ProposalSource.CLICK, ProposalSource.MANUAL])
@pytest.mark.parametrize(("key", "lane", "reason"), EMPTY_LANES)
def test_a_box_the_user_placed_where_no_band_is_found_is_kept(tmp_path, key, lane, reason, source):
    # The user placed the box where the detector finds nothing (a band too
    # faint for it, a lane it cannot read): it is kept as a box edited by hand
    # is, and its lane gets no record.
    case = row_case(key)
    s, image, protein = row_session(tmp_path, case)
    ops.set_box_size(s, protein, BoxSize(width=20, height=30))
    box = placed_box(s, protein, case, lane, source)
    before = tuple(_rect_of(s, box))
    found = detected(s, protein, case.row)
    assert found.lanes[lane].reason == reason
    assert found.size.width > 20 and found.size.height < 30

    placement = ops.detect_row_boxes(s, protein, case.row)
    kept = band_of(s, box)
    # The lane's box after the commit is the kept one.
    assert lane_bands(s, protein)[lane] is kept and placement.band_ids[lane] == box
    assert (kept.source, kept.manually_edited) == (source, False)
    assert placement.kept_lanes == (lane,)
    assert (placement.replaced_band_ids, placement.removed_band_ids) == ((), ())
    assert (placement.undetected_lanes, placement.unmeasured_lanes) == ((), ())
    assert protein_of(s, protein).undetected == []
    [(empty_lane, empty_reason, _, _)] = placement.empty
    assert (empty_lane, empty_reason) == (lane, reason)
    # The box survives, so the size only grows; it stays on its own centre.
    size = BoxSize(width=found.size.width, height=30)
    assert placement.box_size == protein_of(s, protein).box_size == size
    assert tuple(_rect_of(s, box)) == centred(s, image, before, size)
    rects = rects_by_lane(s, protein)
    for other, rect in enumerate(found.slots):
        if rect is not None:
            assert rects[other] == centred(s, image, rect, size)
    params = s.project.log[-1].params
    assert (params["kept_lanes"], params["removed_band_ids"]) == ([lane], [])
    assert params["undetected_written"] == []
    assert (params["lanes"][lane]["band_id"], params["lanes"][lane]["rect"]) == (
        box,
        _rect_of(s, box),
    )
    assert ops.compute(s).proteins[0].detected[lane] is True
    assert_nets_current(s)

    committed = s.project
    again = ops.detect_row_boxes(s, protein, case.row)
    assert (again.kept_lanes, again.band_ids) == ((lane,), placement.band_ids)
    assert s.project is committed  # the same drag again: a no-op


@pytest.mark.parametrize("source", [ProposalSource.ROW_BOX, ProposalSource.MW_GUIDED])
@pytest.mark.parametrize(("key", "lane", "reason"), EMPTY_LANES)
def test_a_detectors_box_where_no_band_is_found_goes(tmp_path, key, lane, reason, source):
    # A box an earlier detection placed gives way to this one's outcome: where
    # it finds no band, the box goes, and the lane gets a record or is left
    # not measured.
    case = row_case(key)
    s, _, protein = row_session(tmp_path, case)
    box = placed_box(s, protein, case, lane, source)
    found = detected(s, protein, case.row)
    assert found.lanes[lane].reason == reason

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert lane not in lane_bands(s, protein) and placement.band_ids[lane] is None
    assert (placement.kept_lanes, placement.removed_band_ids) == ((), (box,))
    recorded = (lane,) if reason == "no_band" else ()
    assert placement.undetected_lanes == recorded
    assert placement.unmeasured_lanes == (() if recorded else (lane,))
    assert _keys(s, protein) == [(i, 0) for i in recorded]
    # Nothing survives, so the size is this detection's.
    assert placement.box_size == protein_of(s, protein).box_size == found.size
    assert s.project.log[-1].params["removed_band_ids"] == [box]
    assert ops.compute(s).proteins[0].detected[lane] is (False if recorded else None)


def test_a_box_the_user_placed_in_a_lane_left_unmeasured_stays_on_the_next_drag(tmp_path):
    # The first drag leaves the stained lane not measured; the user places a
    # box there by hand; dragging over the row again keeps it and replaces the
    # rest in place.
    case = adversarial("blotch_empty", 1000)  # lane 4: a stain over the empty lane
    s, _, protein = row_session(tmp_path, case)
    first = ops.detect_row_boxes(s, protein, case.row)
    assert first.unmeasured_lanes == (4,)
    box = placed_box(s, protein, case, 4, ProposalSource.MANUAL)

    second = ops.detect_row_boxes(s, protein, case.row)
    assert lane_bands(s, protein)[4].id == box
    assert second.kept_lanes == (4,) and second.unmeasured_lanes == ()
    ids = tuple(band_id for band_id in first.band_ids if band_id is not None)
    assert second.replaced_band_ids == ids
    assert second.band_ids == (*first.band_ids[:4], box, *first.band_ids[5:])


def test_a_record_from_any_detector_gives_way(tmp_path):
    # An MW-guided record (#58) gives way to this run's outcome as a row-box
    # record does, even in a lane this run could not measure.
    case = adversarial("blotch_empty", 1000)  # lane 4: a stain over the empty lane
    s, _, protein = row_session(tmp_path, case)
    old = _record(4, region=(350, 68, 393, 93), source=ProposalSource.MW_GUIDED)
    plant_records(s, protein, old)

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.unmeasured_lanes == (4,)
    assert protein_of(s, protein).undetected == []
    assert s.project.log[-1].params["dropped_undetected"] == [_record_json(protein, old)]


def test_include_no_lanes_are_detected_too(tmp_path):
    case = ROWS["all_present"]
    s, _, protein = row_session(tmp_path, case, excluded=(1, 4))
    placement = ops.detect_row_boxes(s, protein, case.row)
    assert None not in placement.band_ids
    assert sorted(lane_bands(s, protein)) == list(range(6))
    res = ops.compute(s)
    # The excluded lanes hold values, so the all-lanes set shows what excluding
    # them changes.
    assert (res.label, res.excluded_lanes) == ("Excluding lanes 2, 5", [1, 4])
    assert res.all_lanes is not None and res.all_lanes.label == "All lanes"
    assert None not in res.proteins[0].nets


def test_an_empty_include_no_lane_is_recorded_too(tmp_path):
    case = ROWS["missing_two"]  # lanes 0 and 3 are empty
    s, _, protein = row_session(tmp_path, case, excluded=(0,))
    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.undetected_lanes == (0, 3)
    assert _keys(s, protein) == [(0, 0), (3, 0)]
    res = ops.compute(s)
    assert res.proteins[0].detected[0] is False
    # The notice names only the lanes this set includes.
    [notice] = [n for n in res.notices if n.code is NoticeCode.BELOW_DETECTION]
    assert notice.lane_indices == (3,)


def test_a_row_leaves_the_other_proteins_on_its_image_alone(tmp_path):
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    other = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    for lane in (1, 2):  # over the target's bands: boxes of two proteins may overlap
        ops.place_box(s, other, *at_lane(case, lane), lane_index=lane, grow=True)
    plant_records(s, other, _record(0))
    before = protein_of(s, other)

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert None not in placement.band_ids
    assert protein_of(s, other) == before
    assert_nets_current(s)


def test_a_row_replaces_boxes_nobody_edited(tmp_path):
    case = ROWS["missing_middle"]  # lane 2 is empty
    s, _, protein = row_session(tmp_path, case)
    clicked = ops.place_box(s, protein, *at_lane(case, 0), lane_index=0, grow=True)
    plant(s, lambda draft: setattr(draft.batch.find_band(clicked)[1], "apparent_mw", 92.0))
    fixed = placed_box(s, protein, case, 1, ProposalSource.MANUAL)
    guided = placed_box(s, protein, case, 4, ProposalSource.MW_GUIDED)
    dropped = placed_box(s, protein, case, 2, ProposalSource.ROW_BOX)
    old = _record(3)
    plant_records(s, protein, old)
    found = detected(s, protein, case.row)

    placement = ops.detect_row_boxes(s, protein, case.row)
    bands = lane_bands(s, protein)
    # Lanes 0, 1 and 4: the band found takes the box nobody edited in place,
    # whoever placed it; its id stays, and the MW read from the clicked box's
    # old position (#58) goes with it.
    for lane, band_id in ((0, clicked), (1, fixed), (4, guided)):
        assert bands[lane].id == band_id == placement.band_ids[lane]
        assert (bands[lane].source, bands[lane].manually_edited, bands[lane].apparent_mw) == (
            ProposalSource.ROW_BOX,
            False,
            None,
        )
    # Lane 2: nothing reaches the detection limit; the box an earlier detection
    # placed goes, a record comes.
    assert 2 not in bands and placement.band_ids[2] is None
    assert [u.lane_index for u in protein_of(s, protein).undetected] == [2]
    # Lane 3: the record gives way to the band found there.
    assert bands[3].id == placement.band_ids[3] not in (clicked, fixed, guided, dropped)
    # Nothing survives, so the size and the boxes are this detection's.
    assert protein_of(s, protein).box_size == found.size
    assert rects_by_lane(s, protein) == {i: r for i, r in enumerate(found.slots) if r}
    assert (placement.replaced_band_ids, placement.removed_band_ids) == (
        (clicked, fixed, guided),
        (dropped,),
    )
    assert placement.kept_lanes == ()
    params = s.project.log[-1].params
    assert (params["replaced_band_ids"], params["removed_band_ids"]) == (
        [clicked, fixed, guided],
        [dropped],
    )
    assert params["dropped_undetected"] == [_record_json(protein, old)]
    assert_nets_current(s)


def test_a_hand_edited_box_is_kept_and_reported(tmp_path):
    case = ROWS["missing_middle"]  # lane 2 is empty
    s, image, protein = row_session(tmp_path, case)
    on_band = ops.place_box(s, protein, *at_lane(case, 1), lane_index=1, grow=True)
    on_empty = ops.place_box(s, protein, *at_lane(case, 2), lane_index=2, grow=False)
    edit_by_hand(s, on_band)
    edit_by_hand(s, on_empty)
    before = {band_id: tuple(_rect_of(s, band_id)) for band_id in (on_band, on_empty)}
    old_size = protein_of(s, protein).box_size
    found = detected(s, protein, case.row)

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.kept_lanes == (1, 2)
    assert (placement.band_ids[1], placement.band_ids[2]) == (on_band, on_empty)
    bands = lane_bands(s, protein)
    assert (bands[1].id, bands[2].id) == (on_band, on_empty)
    assert (bands[1].source, bands[2].source) == (ProposalSource.CLICK, ProposalSource.MANUAL)
    assert bands[1].manually_edited and bands[2].manually_edited
    # A kept box's lane gets no record, even where nothing was detected.
    assert protein_of(s, protein).undetected == []
    assert (placement.undetected_lanes, placement.unmeasured_lanes) == ((), ())
    assert placement.empty[0][:2] == (2, "no_band")
    # Boxes survive, so the size only grows; they stay on their own centres.
    size = BoxSize(
        width=max(old_size.width, found.size.width),
        height=max(old_size.height, found.size.height),
    )
    assert protein_of(s, protein).box_size == placement.box_size == size
    for band_id, rect in before.items():
        assert tuple(_rect_of(s, band_id)) == centred(s, image, rect, size)
    rects = rects_by_lane(s, protein)
    for lane in (0, 3, 4, 5):
        assert rects[lane] == centred(s, image, found.slots[lane], size)
    assert_nets_current(s)


def test_the_size_only_grows_while_a_box_survives(tmp_path):
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    ops.set_box_size(s, protein, BoxSize(width=30, height=20))
    kept = ops.place_box(s, protein, *at_lane(case, 0), lane_index=0, grow=False)
    edit_by_hand(s, kept)
    found = detected(s, protein, case.row)
    assert found.size.width > 30 and found.size.height < 20

    placement = ops.detect_row_boxes(s, protein, case.row)
    size = BoxSize(width=found.size.width, height=20)  # the larger of each
    assert placement.box_size == protein_of(s, protein).box_size == size
    rects = rects_by_lane(s, protein)
    logged = s.project.log[-1].params["lanes"]
    for lane in range(1, 6):
        assert rects[lane] == centred(s, image, found.slots[lane], size)
        # The log names the committed box, not the detector's lower one.
        assert logged[lane]["rect"] == list(rects[lane]) != list(found.slots[lane])
    assert placement.kept_lanes == (0,)
    assert_nets_current(s)


def test_a_row_whose_bands_all_have_edited_boxes_grows_them_and_records(tmp_path):
    # Every band found is in a lane whose box was edited by hand: nothing is
    # placed, but those boxes survive, so the size only grows, as with any
    # survivor, and the empty lane is still recorded.
    case = ROWS["missing_middle"]  # lane 2 is empty
    s, image, protein = row_session(tmp_path, case)
    small = BoxSize(width=40, height=10)
    ops.set_box_size(s, protein, small)
    for lane in (0, 1, 3, 4, 5):
        edit_by_hand(
            s, ops.place_box(s, protein, *at_lane(case, lane), lane_index=lane, grow=False)
        )
    before = rects_by_lane(s, protein)
    ids = {lane: band.id for lane, band in lane_bands(s, protein).items()}
    found = detected(s, protein, case.row)
    size = BoxSize(width=max(40, found.size.width), height=max(10, found.size.height))
    assert size != small  # the detector's size is larger

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.band_ids == tuple(ids.get(lane) for lane in range(6))  # the kept boxes
    assert placement.kept_lanes == (0, 1, 3, 4, 5)
    assert placement.box_size == protein_of(s, protein).box_size == size
    assert rects_by_lane(s, protein) == {
        lane: centred(s, image, rect, size) for lane, rect in before.items()
    }
    assert (placement.undetected_lanes, _keys(s, protein)) == ((2,), [(2, 0)])
    assert_nets_current(s)

    committed = s.project
    ops.detect_row_boxes(s, protein, case.row)
    assert s.project is committed  # the same record again: a no-op


def test_another_bands_box_and_record_are_left_alone(tmp_path):
    # A second expected band (#58) is not what the row looked for: its box
    # survives, so the size only grows, and its record stays. Edited by hand,
    # it keeps nothing but itself: the lane still takes the band found.
    case = ROWS["missing_middle"]  # lane 2 is empty
    s, image, protein = row_session(tmp_path, case)
    ops.set_box_size(s, protein, BoxSize(width=40, height=16))
    other = ops.place_box(s, protein, round(case.lane_cx[0]), 20, lane_index=0, grow=False)
    second = _record(2, band_index=1, region=(209, 10, 253, 30))
    plant_records(s, protein, second, bands=2)
    plant(s, lambda draft: setattr(draft.batch.find_band(other)[1], "band_index", 1))
    edit_by_hand(s, other)
    found = detected(s, protein, case.row)

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert band_of(s, other).band_index == 1 and other not in placement.band_ids
    assert placement.band_ids[0] is not None
    assert _keys(s, protein) == [(2, 0), (2, 1)]
    assert protein_of(s, protein).undetected[1] == second
    size = BoxSize(width=max(40, found.size.width), height=16)
    assert placement.box_size == size
    assert placement.kept_lanes == ()  # only band-index-0 boxes are kept or replaced
    assert_nets_current(s)


def test_a_second_drag_corrects_the_first(tmp_path):
    # The first drag went over the wrong row (taller bands above the right
    # one). Dragging the right row replaces every box; with nothing edited by
    # hand the size is set afresh, not grown.
    tall = synthetic_row("tall", "", 51, h=24.0)
    short = synthetic_row("short", "", 51)  # the same lanes, bands half as high
    offset = tall.image.shape[0]
    case = dataclasses.replace(short, image=np.vstack([tall.image, short.image]))
    s, _, protein = row_session(tmp_path, case)
    x0, y0, x1, y1 = short.row
    right = (x0, y0 + offset, x1, y1 + offset)
    first = ops.detect_row_boxes(s, protein, tall.row)
    found = detected(s, protein, right)
    assert found.size.height < first.box_size.height

    second = ops.detect_row_boxes(s, protein, right)
    assert second.band_ids == second.replaced_band_ids == first.band_ids  # replaced in place
    assert protein_of(s, protein).box_size == second.box_size == found.size
    assert rects_by_lane(s, protein) == dict(enumerate(found.slots))
    assert_nets_current(s)

    committed = s.project
    assert ops.detect_row_boxes(s, protein, right) == second
    assert s.project is committed  # the same drag again: a no-op, no entry


def test_a_box_the_same_drag_replaces_where_it_is_keeps_its_mw(tmp_path):
    # Replaced in place by a box of the same rect, a band keeps what its
    # position gave it (an apparent MW, #58): the same drag again is a no-op.
    case = ROWS["all_present"]
    s, _, protein = row_session(tmp_path, case)
    first = ops.detect_row_boxes(s, protein, case.row)
    band_id = first.band_ids[2]
    plant(s, lambda draft: setattr(draft.batch.find_band(band_id)[1], "apparent_mw", 92.0))
    committed, length = s.project, len(s.project.log)

    again = ops.detect_row_boxes(s, protein, case.row)
    assert again.replaced_band_ids == first.band_ids
    assert s.project is committed and len(s.project.log) == length
    assert band_of(s, band_id).apparent_mw == 92.0


def test_warnings_are_reported_and_logged(tmp_path):
    case = adversarial("tall_band", 1000)  # lane 3 far taller than the others
    s, _, protein = row_session(tmp_path, case)
    found = detected(s, protein, case.row)
    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.flags == found.flags == ("size_outlier",)
    assert placement.notes == found.notes != ()
    params = s.project.log[-1].params
    assert (params["flags"], params["notes"]) == (["size_outlier"], list(found.notes))


def test_the_row_commit_logs_every_lane(tmp_path):
    case = ROWS["missing_middle"]  # lane 2 is empty
    s, _, protein = row_session(tmp_path, case)
    ops.set_box_size(s, protein, BoxSize(width=40, height=10))
    replaced = ops.place_box(s, protein, *at_lane(case, 0), lane_index=0, grow=False)
    removed = placed_box(s, protein, case, 2, ProposalSource.ROW_BOX)
    kept = ops.place_box(s, protein, *at_lane(case, 4), lane_index=4, grow=False)
    edit_by_hand(s, kept)
    old = _record(3, snr=2.5)
    plant_records(s, protein, old)
    found = detected(s, protein, case.row)
    next_id = s.project.next_id

    placement = ops.detect_row_boxes(s, protein, case.row)
    # Each lane's box after the commit: replaced in place, new, or kept (lane 4).
    ids = [replaced, f"band-{next_id}", None, f"band-{next_id + 1}", kept, f"band-{next_id + 2}"]
    assert placement.band_ids == tuple(ids)
    [written] = protein_of(s, protein).undetected
    lanes = [
        {
            "band_id": band_id,
            "rect": None if band_id is None else _rect_of(s, band_id),
            "reason": lane.reason,
            "snr": round(lane.snr, 2),
            "expected_x": round(lane.expected_x, 1),
        }
        for band_id, lane in zip(ids, found.lanes, strict=True)
    ]
    assert lanes[4]["reason"] == "band"  # found, but the edited box was kept
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "detect_row_boxes",
        {
            "protein_id": protein,
            "row": list(case.row),
            "lanes": lanes,
            "kept_lanes": [4],
            "replaced_band_ids": [replaced],
            "removed_band_ids": [removed],
            "undetected_written": [_record_json(protein, written)],
            "dropped_undetected": [_record_json(protein, old)],
            "box_size": _size_of(s, protein),
            "pitch": found.pitch,
            "noise": found.noise,
            "flags": [],
            "notes": [],
            "right_to_left": False,
            # Dev builds share a version string: the constants that placed the
            # boxes go with each commit.
            "settings": rowdetect.settings(),
        },
    )
    assert entry.content_hash == content_hash(s.project)


def test_record_settings_are_what_the_row_commit_uses(tmp_path, monkeypatch):
    calls = []
    real = rowdetect.detect_row

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(rowdetect, "detect_row", spy)
    case = ROWS["all_present"]
    s, image_id, protein = row_session(tmp_path, case)
    ops.detect_row_boxes(s, protein, case.row)
    [(args, kwargs)] = calls
    bound = inspect.signature(real).bind(*args, **kwargs)
    bound.apply_defaults()
    image = s.project.batch.find_image(image_id)
    assert bound.arguments["size_rule"] == record.settings()["detect_row"]["size_rule"]
    assert (bound.arguments["background"], bound.arguments["dark_on_light"]) == (
        image.background,
        True,
    )
    assert record.settings()["detect_row"] == rowdetect.settings()


def _row_scene(tmp_path: Path, setup) -> tuple[ProjectSession, Recorder, dict[str, str]]:
    """:data:`SCENE` with a target and a clicked box in lane 1, then ``setup``
    (autosaved), reopened with an empty pixel cache and a recording hook."""
    s, image, protein = row_session(tmp_path, SCENE, hook=save_to_folder)
    band = ops.place_box(s, protein, *at_lane(SCENE, 1), lane_index=1, grow=True)
    ids = {"image": image, "protein": protein, "band": band}
    if setup is not None:
        setup(s, ids)
    recorder = Recorder()
    return ops.open_project(s.folder, autosave=recorder, clock=FakeClock()), recorder, ids


def _no_lanes(s: ProjectSession, ids: dict[str, str]) -> None:
    ops.remove_box(s, ids["band"])
    ops.set_lanes(s, [])


def _edited_pair(s: ProjectSession, ids: dict[str, str]) -> None:
    """Two surviving boxes 14 px apart, lane 1's edited by hand and one of a
    second band (#58): the detected width cannot fit them. (Two lanes' boxes
    that close would not line up with the row's bands.)"""
    ops.set_box_size(s, ids["protein"], BoxSize(width=10, height=8))
    x, y = at_lane(SCENE, 1)
    near = ops.place_box(s, ids["protein"], x + 14, y, lane_index=2, grow=False)

    def second_band(draft: Project) -> None:
        protein, band = draft.batch.find_band(near)
        protein.expected_band_count = 2
        band.band_index = 1

    plant(s, second_band)
    edit_by_hand(s, ids["band"])


def _wide_survivor(s: ProjectSession, ids: dict[str, str]) -> None:
    """A hand-edited box wider than the lane pitch, above the row: the new
    boxes, grown to its width, would overlap each other."""
    ops.set_box_size(s, ids["protein"], BoxSize(width=80, height=8))
    edit_by_hand(s, ids["band"], to=(round(SCENE.lane_cx[1]), 20))


def _moved_onto_lane_2(s: ProjectSession, ids: dict[str, str]) -> None:
    """Lane 1's box, edited by hand, sits on lane 2's band."""
    ops.set_box_size(s, ids["protein"], BoxSize(width=40, height=10))
    edit_by_hand(s, ids["band"], to=at_lane(SCENE, 2))


def _touching_lane_2(s: ProjectSession, ids: dict[str, str]) -> None:
    """Lane 1's box, edited by hand, ends where lane 2's detected box begins:
    clear of it at its own size, over it once the size grows to the detector's."""
    found = detected(s, ids["protein"], SCENE.row)
    size = BoxSize(width=found.size.width - 4, height=found.size.height - 2)
    ops.set_box_size(s, ids["protein"], size)
    x0, y0, _, y1 = found.slots[2]
    top = (y0 + y1) // 2 - size.height // 2
    ops.move_box(s, ids["band"], (x0 - size.width, top, x0, top + size.height))
    rect = _rect_of(s, ids["band"])
    assert band_of(s, ids["band"]).manually_edited
    assert rect[2] == x0 and not boxes.overlaps_any(tuple(rect), [found.slots[2]])


def _placed_lanes(s: ProjectSession, ids: dict[str, str]) -> None:
    """Another protein's boxes in the true lanes of :data:`SCENE`."""
    other_protein_in_lanes(s, ids["image"], SCENE)


def _numbered_both_ways(s: ProjectSession, ids: dict[str, str]) -> None:
    """Another protein's boxes number the lanes right to left, the target's
    boxes in lanes 1 and 2 left to right."""
    other_protein_in_lanes(s, ids["image"], SCENE, mirrored=True)
    ops.place_box(s, ids["protein"], *at_lane(SCENE, 2), lane_index=2, grow=True)


def _lane_in_two_columns(s: ProjectSession, ids: dict[str, str]) -> None:
    """Another protein numbered a lane off: its box in lane 1 lies over true
    lane 0, the target's over true lane 1."""
    other_protein_in_lanes(s, ids["image"], SCENE, range(5), offset=1)


# The setups whose rows are refused for the lanes on the image: the bands
# found would not line up with those lanes either, so the code alone does not
# show which check refused them.
NUMBERING_SETUPS = (_numbered_both_ways, _lane_in_two_columns)
NUMBERED = "the lanes already placed on this image are numbered"

ROW_REFUSALS = [
    pytest.param(None, "prot-999", SCENE.row, None, id="unknown-protein"),
    pytest.param(None, None, (1, 2, 3), ErrorCode.INVALID_INPUT, id="three-values"),
    pytest.param(None, None, "0 0 9 9", ErrorCode.INVALID_INPUT, id="text"),
    pytest.param(
        None, None, (*SCENE.row[:3], SCENE.row[3] + 0.5), ErrorCode.INVALID_INPUT, id="float"
    ),
    pytest.param(None, None, (*SCENE.row[:3], True), ErrorCode.INVALID_INPUT, id="bool"),
    # Bytes are a sequence of ints: (62, 68, 255, 93) would be a row.
    pytest.param(None, None, bytes((62, 68, 255, 93)), ErrorCode.INVALID_INPUT, id="bytes"),
    pytest.param(None, None, bytearray((62, 68, 255, 93)), ErrorCode.INVALID_INPUT, id="bytearray"),
    pytest.param(
        None, None, memoryview(bytes((62, 68, 255, 93))), ErrorCode.INVALID_INPUT, id="memoryview"
    ),
    pytest.param(
        None,
        None,
        (SCENE.row[2], SCENE.row[1], SCENE.row[0], SCENE.row[3]),
        ErrorCode.INVALID_INPUT,
        id="inverted",
    ),
    # The row is checked before the lanes.
    pytest.param(
        _no_lanes,
        None,
        (SCENE.row[0], SCENE.row[3], SCENE.row[2], SCENE.row[1]),
        ErrorCode.INVALID_INPUT,
        id="inverted-without-lanes",
    ),
    pytest.param(
        _no_lanes,
        None,
        (SCENE.row[0], SCENE.row[1], SCENE.row[0], SCENE.row[3]),
        ErrorCode.INVALID_INPUT,
        id="empty-without-lanes",
    ),
    pytest.param(_no_lanes, None, SCENE.row, ErrorCode.NO_LANES, id="no-lanes"),
    # The lanes on the image are checked before anything is detected: even a
    # row outside the image is refused for them.
    pytest.param(
        _numbered_both_ways,
        None,
        SCENE.row,
        ErrorCode.ROW_LANES_UNCLEAR,
        id="numbered-both-ways",
    ),
    pytest.param(
        _lane_in_two_columns,
        None,
        SCENE.row,
        ErrorCode.ROW_LANES_UNCLEAR,
        id="lane-in-two-columns",
    ),
    pytest.param(
        _lane_in_two_columns,
        None,
        (SCENE_W + 10, 0, SCENE_W + 50, 20),
        ErrorCode.ROW_LANES_UNCLEAR,
        id="lane-in-two-columns-outside",
    ),
    pytest.param(
        None, None, (SCENE_W + 10, 0, SCENE_W + 50, 20), ErrorCode.OUT_OF_IMAGE, id="outside"
    ),
    pytest.param(
        None,
        None,
        (SCENE.row[0], SCENE.row[1], SCENE.row[0] + 11, SCENE.row[3]),
        ErrorCode.ROW_TOO_SMALL,
        id="too-narrow",
    ),
    pytest.param(
        None,
        None,
        (SCENE.row[0], SCENE.row[1], SCENE.row[2], SCENE.row[1] + 2),
        ErrorCode.ROW_TOO_SMALL,
        id="too-low",
    ),
    pytest.param(None, None, UNCLEAR_ROW, ErrorCode.ROW_LANES_UNCLEAR, id="lanes-unclear"),
    pytest.param(None, None, AMBIGUOUS_ROW, ErrorCode.ROW_LANES_UNCLEAR, id="lanes-ambiguous"),
    pytest.param(
        _placed_lanes,
        None,
        shifted(SCENE, 70),
        ErrorCode.ROW_LANES_UNCLEAR,
        id="off-the-placed-lanes",
    ),
    pytest.param(
        None, None, (SCENE.row[0], 5, SCENE.row[2], 40), ErrorCode.NO_BAND_FOUND, id="no-band"
    ),
    pytest.param(
        _edited_pair, None, SCENE.row, ErrorCode.SIZE_WOULD_OVERLAP, id="kept-boxes-overlap"
    ),
    pytest.param(
        _wide_survivor, None, SCENE.row, ErrorCode.SIZE_WOULD_OVERLAP, id="new-boxes-overlap"
    ),
    pytest.param(_moved_onto_lane_2, None, SCENE.row, ErrorCode.OVERLAP, id="onto-a-kept-box"),
    pytest.param(_touching_lane_2, None, SCENE.row, ErrorCode.OVERLAP, id="onto-a-kept-box-grown"),
]


@pytest.mark.parametrize(("setup", "protein_id", "row", "code"), ROW_REFUSALS)
def test_a_refused_row_changes_nothing(tmp_path, setup, protein_id, row, code):
    s, recorder, ids = _row_scene(tmp_path, setup)
    before = s.project
    next_id, log_length = before.next_id, len(before.log)
    files = listing(s)

    with pytest.raises(UnknownIdError if code is None else OperationError) as info:
        ops.detect_row_boxes(s, protein_id or ids["protein"], row)
    if code is not None:
        assert info.value.code is code
    assert str(info.value).startswith(NUMBERED) is (setup in NUMBERING_SETUPS)
    assert s.project is before
    assert s.project.next_id == next_id
    assert len(s.project.log) == log_length
    assert recorder.actions == []
    assert listing(s) == files
    assert s._pixels == {}  # pixels read for the detection are not kept
    if code is ErrorCode.OVERLAP:
        assert info.value.ids == (ids["band"],)


def _not_2d(pixels: np.ndarray) -> np.ndarray:
    return np.stack([pixels] * 3, axis=-1)


def _nan_in_row(pixels: np.ndarray) -> np.ndarray:
    pixels = pixels.astype(np.float64)
    pixels[SCENE.row[1] + 5, SCENE.row[0] + 5] = np.nan
    return pixels


@pytest.mark.parametrize("damage", [_nan_in_row, _not_2d], ids=["nan-in-row", "not-2d"])
def test_pixels_the_detector_cannot_read_are_an_unreadable_image(tmp_path, damage):
    # The image is at fault, not the row: the refusal names the image.
    s, image, protein = row_session(tmp_path, SCENE)
    pixels = damage(np.array(s.pixels(image)))
    pixels.flags.writeable = False
    s._pixels[image] = pixels
    before = s.project
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, protein, SCENE.row)
    assert (info.value.code, info.value.ids) == (ErrorCode.UNREADABLE_IMAGE, (image,))
    assert s.project is before


@pytest.mark.parametrize(
    ("setup", "words"),
    [(_edited_pair, "that the row keeps overlap"), (_wide_survivor, "overlap each other")],
)
def test_a_size_grown_for_kept_boxes_is_refused_either_way(tmp_path, setup, words):
    # The kept boxes cannot take the grown size, or the row's boxes cannot.
    s, _, ids = _row_scene(tmp_path, setup)
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, ids["protein"], SCENE.row)
    assert info.value.code is ErrorCode.SIZE_WOULD_OVERLAP
    assert words in str(info.value)


@pytest.mark.parametrize(
    ("row", "flag"),
    [
        pytest.param(UNCLEAR_ROW, "lanes_outside_row", id="40px-short"),
        pytest.param(AMBIGUOUS_ROW, "ambiguous_lanes", id="70px-short"),
    ],
)
def test_a_row_that_leaves_out_an_empty_end_lane_is_refused(tmp_path, row, flag):
    # Either refusing flag refuses the commit.
    s, _, protein = row_session(tmp_path, SCENE)
    assert not detected(s, protein, SCENE.row).refused  # the whole row is clear
    assert detected(s, protein, row).flags == (flag,)
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, protein, row)
    assert info.value.code is ErrorCode.ROW_LANES_UNCLEAR
    assert str(info.value) == (
        "the row box does not show which lane each band is in; draw it over every"
        " declared lane, empty end lanes included, or place the boxes by clicking"
    )


# --- #51: a row checked against the lanes already placed on its image ---

OFF_LANES = (
    "the bands in the row box do not line up with the lanes already placed on this"
    " image; draw the box over every declared lane, empty end lanes included"
)


def reading(case: RowCase, found: RowDetection) -> dict[int, int]:
    """Each band found, as its lane read -> the lane whose true centre is
    nearest its extent's."""
    return {
        lane.lane: min(
            range(case.n_lanes),
            key=lambda true: abs(case.lane_cx[true] - (lane.extent[0] + lane.extent[2]) / 2),
        )
        for lane in found.lanes
        if lane.extent is not None
    }


def refused_as(s: ProjectSession, protein_id: str, row, message: str) -> None:
    """The row is refused as not lining up with the image's lanes, changing nothing."""
    before = s.project
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, protein_id, row)
    assert info.value.code is ErrorCode.ROW_LANES_UNCLEAR
    assert str(info.value) == message
    assert s.project is before


@pytest.mark.parametrize(
    "name", ["all_present", "missing_first", "smile", "uneven_spacing", "touching", "twelve_lanes"]
)
def test_a_row_that_lines_up_with_the_lanes_on_its_image_is_committed(tmp_path, name):
    case = ROWS[name]
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case)
    found = detected(s, protein, case.row)
    assert all(read == true for read, true in reading(case, found).items())

    ops.detect_row_boxes(s, protein, case.row)
    assert rects_by_lane(s, protein) == {i: r for i, r in enumerate(found.slots) if r}


@pytest.mark.parametrize(
    ("dx", "committed"),
    [(-28, True), (28, True), (-42, False), (42, False)],
    ids=["0.4-pitch-left", "0.4-pitch-right", "0.6-pitch-left", "0.6-pitch-right"],
)
def test_a_band_lines_up_within_half_a_pitch_of_its_lane(tmp_path, dx, committed):
    # Lanes bend between rows (a smile, a fan): another protein's boxes
    # 0.4 pitch off this row's bands still show the same lanes; 0.6 pitch off,
    # every band lies nearer a neighbouring lane.
    case = ROWS["all_present"]  # pitch 70
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case, dx=dx)
    if committed:
        placement = ops.detect_row_boxes(s, protein, case.row)
        assert None not in placement.band_ids
    else:
        refused_as(s, protein, case.row, OFF_LANES)


@pytest.mark.parametrize(
    "lanes",
    [None, (3, 4), (0, 1), (4, 5)],
    ids=["every-lane", "lanes-3-4", "first-two", "last-two"],
)
@pytest.mark.parametrize("dx", [70, -70], ids=["right", "left"])
@pytest.mark.parametrize("name", ["all_present", "missing_middle"])
def test_a_row_read_a_lane_off_the_lanes_on_its_image_is_refused(tmp_path, name, dx, lanes):
    # Moved a pitch along the row, the box leaves out a banded end lane and
    # the detector reads every band one lane off without a refusing flag. Two
    # lanes another protein placed on the image show it.
    case = ROWS[name]
    row = shifted(case, dx)
    s, image, protein = row_session(tmp_path, case)
    found = detected(s, protein, row)
    assert not found.refused
    assert all(true == read + (1 if dx > 0 else -1) for read, true in reading(case, found).items())
    other_protein_in_lanes(s, image, case, lanes)
    refused_as(s, protein, row, OFF_LANES)


def test_one_lane_on_the_image_does_not_show_the_lanes(tmp_path):
    # Lane positions need the pitch, so two placed lanes: with one, the row is
    # not checked (and this one is committed a lane off).
    case = ROWS["all_present"]
    row = shifted(case, 70)
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case, [3])
    placement = ops.detect_row_boxes(s, protein, row)
    assert placement.band_ids[5] is None and placement.undetected_lanes == (5,)


def test_the_boxes_a_detector_placed_are_not_lanes_a_row_is_checked_against(tmp_path):
    # The first drag, a lane off, is committed: nothing is placed to check it
    # against. The right drag replaces its boxes in place; they do not refuse it.
    # (Boxes the user placed that a row replaces do check it: see
    # test_a_row_read_a_lane_off_the_lanes_that_turn_it_is_refused.)
    case = ROWS["all_present"]
    s, _, protein = row_session(tmp_path, case)
    first = ops.detect_row_boxes(s, protein, shifted(case, 70))
    assert first.undetected_lanes == (5,)
    found = detected(s, protein, case.row)

    second = ops.detect_row_boxes(s, protein, case.row)
    assert second.replaced_band_ids == first.band_ids[:5]
    assert rects_by_lane(s, protein) == dict(enumerate(found.slots))


def test_the_boxes_a_row_keeps_are_lanes_it_is_checked_against(tmp_path):
    # This protein's own kept boxes show the lanes too: one edited by hand, and
    # one placed by the user where the row, read a lane off, finds no band.
    # (Committed, lane 3's box would also overlap the kept one of lane 4.)
    case = ROWS["all_present"]
    row = shifted(case, 70)
    s, _, protein = row_session(tmp_path, case)
    ops.set_box_size(s, protein, BoxSize(width=40, height=12))
    edit_by_hand(s, placed_box(s, protein, case, 4, ProposalSource.MANUAL))
    placed_box(s, protein, case, 5, ProposalSource.MANUAL)
    assert detected(s, protein, row).lanes[5].reason == "no_band"
    refused_as(s, protein, row, OFF_LANES)


def at_true_lanes(s: ProjectSession, protein_id: str, case: RowCase) -> dict[int, int]:
    """Each band-index-0 box of the protein, as its stored lane -> the lane
    whose true centre is nearest the box's."""
    return {
        lane: min(range(case.n_lanes), key=lambda true: abs(case.lane_cx[true] - (x0 + x1) / 2))
        for lane, (x0, _, x1, _) in rects_by_lane(s, protein_id).items()
    }


# Rows whose other protein's boxes in two neighbouring lanes, extended by
# their own pitch, miss the far lanes by over half a pitch: uneven spacing
# (56, 84, 63, 80.5, 59.5 px) and twenty lanes whose first two lie 34 px apart
# against a mean pitch of about 36.
SPARSE_ANCHORS = [
    pytest.param("uneven_spacing", (0, 1), id="uneven-first-two"),
    pytest.param("uneven_spacing", (1, 2), id="uneven-lanes-1-2"),
    pytest.param("uneven_spacing", (4, 5), id="uneven-last-two"),
    pytest.param("twenty_lanes/1001", (0, 1), id="twenty-1001-first-two"),
    pytest.param("twenty_lanes/1001", (18, 19), id="twenty-1001-last-two"),
    pytest.param("twenty_lanes/1002", (0, 1), id="twenty-1002-first-two"),
    pytest.param("twenty_lanes/1002", (1, 2), id="twenty-1002-lanes-1-2"),
]


def named_case(name: str) -> RowCase:
    """A bench row by name, or an adversarial one as ``recipe/seed``."""
    if name in ROWS:
        return ROWS[name]
    recipe, seed = name.rsplit("/", 1)
    return adversarial(recipe, int(seed))


@pytest.mark.parametrize(("name", "lanes"), SPARSE_ANCHORS)
def test_a_row_is_committed_beside_two_lanes_placed_on_its_image(tmp_path, name, lanes):
    # Past the lanes placed, a lane's expected x steps by the row's own pitch,
    # not by the spacing of the two placed lanes.
    case = named_case(name)
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case, lanes)
    found = detected(s, protein, case.row)
    assert all(read == true for read, true in reading(case, found).items())

    ops.detect_row_boxes(s, protein, case.row)
    assert rects_by_lane(s, protein) == {i: r for i, r in enumerate(found.slots) if r}


@pytest.mark.parametrize(
    ("name", "lanes"),
    [
        ("uneven_spacing", (0, 1)),
        ("uneven_spacing", (4, 5)),
        ("twenty_lanes/1001", (17, 18)),
        ("bubble_band/1000", (0, 1)),
        ("bubble_band/1001", (0, 1)),
    ],
)
def test_the_same_drag_after_two_boxes_are_edited_by_hand_is_committed(tmp_path, name, lanes):
    # The two edited boxes are the only lanes placed besides the boxes the
    # second drag replaces. (A bubble splits lane 1's band, so its box lies
    # about 15 px off the lane: the two edited boxes are 85 px apart against a
    # pitch of 70.)
    case = named_case(name)
    s, _, protein = row_session(tmp_path, case)
    first = ops.detect_row_boxes(s, protein, case.row)
    for lane in lanes:
        edit_by_hand(s, first.band_ids[lane])

    second = ops.detect_row_boxes(s, protein, case.row)
    assert second.kept_lanes == lanes
    assert second.replaced_band_ids == tuple(
        band_id for lane, band_id in enumerate(first.band_ids) if band_id and lane not in lanes
    )
    assert all(read == true for read, true in at_true_lanes(s, protein, case).items())


@pytest.mark.parametrize("name", ["all_present", "missing_first"])
def test_a_row_on_lanes_numbered_right_to_left_reads_them_right_to_left(tmp_path, name):
    # The other protein's boxes number the lanes right to left, so the row
    # does too: lane 0 at the box's right end, each band in the lane of the
    # other protein's box above it, and the record of an empty lane in its own
    # lane.
    case = ROWS[name]
    n = case.n_lanes
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case, mirrored=True)
    found = detected(s, protein, case.row)

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert rects_by_lane(s, protein) == {n - 1 - i: r for i, r in enumerate(found.slots) if r}
    assert at_true_lanes(s, protein, case) == {
        n - 1 - true: true for true in range(n) if true in case.reference
    }
    empty = [n - 1 - lane.lane for lane in found.lanes if lane.reason == "no_band"]
    assert list(placement.undetected_lanes) == sorted(empty)
    assert [u.lane_index for u in protein_of(s, protein).undetected] == sorted(empty)
    assert placement.right_to_left
    entry = s.project.log[-1]
    assert entry.params["right_to_left"] is True
    assert [lane["band_id"] for lane in entry.params["lanes"]] == list(placement.band_ids)


def test_a_row_read_a_lane_off_lanes_numbered_right_to_left_is_refused(tmp_path):
    case = ROWS["all_present"]
    row = shifted(case, 70)
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case, (3, 4), mirrored=True)
    refused_as(s, protein, row, OFF_LANES)


def test_boxes_the_user_placed_right_to_left_turn_the_row_around(tmp_path):
    # The protein's own boxes, placed by the user in lanes numbered right to
    # left, show the direction; the row finds bands under them and replaces
    # them in place.
    case = ROWS["all_present"]
    n = case.n_lanes
    s, _, protein = row_session(tmp_path, case)
    ops.set_box_size(s, protein, BoxSize(width=40, height=12))
    clicked = [
        ops.place_box(s, protein, *at_lane(case, true), lane_index=n - 1 - true, grow=False)
        for true in (1, 2)
    ]

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.right_to_left
    assert placement.replaced_band_ids == (clicked[1], clicked[0])  # lanes 3, 4
    assert placement.band_ids[n - 2] == clicked[0] and placement.band_ids[n - 3] == clicked[1]
    assert at_true_lanes(s, protein, case) == {n - 1 - true: true for true in range(n)}

    # The boxes that turned the row are the row's now, and nothing else is
    # placed: the same drag again reads the lanes the way the row's boxes run,
    # so it changes nothing.
    committed, length = s.project, len(s.project.log)
    again = ops.detect_row_boxes(s, protein, case.row)
    assert again.right_to_left and again.band_ids == placement.band_ids
    assert s.project is committed and len(s.project.log) == length


def test_lanes_placed_after_a_row_turn_the_next_drag(tmp_path):
    # The row's own boxes give way to the lanes placed on the image: read left
    # to right with nothing placed, the same drag is read right to left once
    # another protein's boxes number the lanes that way.
    case = ROWS["all_present"]
    n = case.n_lanes
    s, image, protein = row_session(tmp_path, case)
    first = ops.detect_row_boxes(s, protein, case.row)
    assert not first.right_to_left
    other_protein_in_lanes(s, image, case, mirrored=True)

    second = ops.detect_row_boxes(s, protein, case.row)
    assert second.right_to_left
    assert set(second.replaced_band_ids) == set(first.band_ids)
    assert at_true_lanes(s, protein, case) == {n - 1 - true: true for true in range(n)}


def test_lane_anchors_of_some_bands_only(tmp_path):
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case, (2, 3))
    own = {band_id for band_id in ops.detect_row_boxes(s, protein, case.row).band_ids if band_id}
    batch = s.project.batch
    ref = batch.find_image(image)

    mine, others = lane_anchors(batch, ref, only=own), lane_anchors(batch, ref, without=own)
    assert sorted(lane for _, lane in mine) == list(range(6))
    assert sorted(lane for _, lane in others) == [2, 3]
    assert sorted(mine + others) == sorted(lane_anchors(batch, ref))
    assert lane_anchors(batch, ref, without=own, only=own) == []


def clicked_lanes(tmp_path: Path, mirrored: bool) -> tuple[ProjectSession, str, str, list[str]]:
    """The protein's boxes clicked in true lanes 0, 1 and 4, numbered right to
    left when ``mirrored``."""
    case = ROWS["all_present"]
    n = case.n_lanes
    s, image, protein = row_session(tmp_path, case)
    lanes = {true: n - 1 - true if mirrored else true for true in (0, 1, 4)}
    clicked = [
        ops.place_box(s, protein, *at_lane(case, true), lane_index=lane, grow=True)
        for true, lane in lanes.items()
    ]
    return s, image, protein, clicked


@pytest.mark.parametrize("mirrored", [False, True], ids=["clicked-ltr", "clicked-rtl"])
def test_the_lanes_that_turn_a_row_check_it(tmp_path, mirrored):
    # The clicked boxes turn the row and check it, and it replaces them.
    case = ROWS["all_present"]
    n = case.n_lanes
    s, _, protein, clicked = clicked_lanes(tmp_path, mirrored)

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.right_to_left is mirrored
    assert sorted(placement.replaced_band_ids) == sorted(clicked)
    assert at_true_lanes(s, protein, case) == {
        n - 1 - true if mirrored else true: true for true in range(n)
    }


@pytest.mark.parametrize("dx", [70, -70], ids=["right", "left"])
@pytest.mark.parametrize("mirrored", [False, True], ids=["clicked-ltr", "clicked-rtl"])
def test_a_row_read_a_lane_off_the_lanes_that_turn_it_is_refused(tmp_path, mirrored, dx):
    # The clicked boxes the row would replace are among the lanes that check
    # it: they show where the user put each lane.
    case = ROWS["all_present"]
    s, _, protein, _ = clicked_lanes(tmp_path, mirrored)
    refused_as(s, protein, shifted(case, dx), OFF_LANES)


# --- #51: a row on an image whose lanes are numbered inconsistently ---


def both_ways(ltr: str, rtl: str) -> str:
    """The refusal of a row on an image whose lanes are numbered both ways."""
    return (
        f"the lanes already placed on this image are numbered both ways: the boxes of {ltr}"
        f" left to right, those of {rtl} right to left; fix the lane numbers of the boxes"
        " already on this image first"
    )


def two_columns(lanes: str, names: str, half: int) -> str:
    """The refusal of a row on an image where one lane number labels two
    columns."""
    return (
        f"the lanes already placed on this image are numbered inconsistently: in {lanes},"
        f" the boxes of {names} lie more than {half} px (half the lane pitch) apart; fix"
        " the lane numbers of the boxes already on this image first"
    )


def numbering_refused(
    s: ProjectSession, protein_id: str, row, message: str, monkeypatch
) -> tuple[str, ...]:
    """The row is refused for the lanes on its image before anything is
    detected, changing nothing; the refusal's ids."""

    def no_detection(*args, **kwargs):
        raise AssertionError("the row was detected")

    monkeypatch.setattr(rowdetect, "detect_row", no_detection)
    before = s.project
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, protein_id, row)
    assert info.value.code is ErrorCode.ROW_LANES_UNCLEAR
    assert str(info.value) == message
    assert s.project is before
    return info.value.ids


@pytest.mark.parametrize("mirrored", [False, True], ids=["gapdh-ltr", "gapdh-rtl"])
def test_two_proteins_numbering_the_lanes_opposite_ways_refuse_a_row(
    tmp_path, monkeypatch, mirrored
):
    # Whatever the row box, and before it is read: the user fixes the lane
    # numbers first. The refusal names the proteins, left to right first.
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    gapdh = other_protein_in_lanes(s, image, case, (1, 2, 3), mirrored=mirrored)
    actin = other_protein_in_lanes(s, image, case, (3, 4), mirrored=not mirrored, name="actin")
    ltr, rtl = ("'actin'", "'GAPDH'") if mirrored else ("'GAPDH'", "'actin'")
    for row in (case.row, shifted(case, 70), UNCLEAR_ROW):
        ids = numbering_refused(s, protein, row, both_ways(ltr, rtl), monkeypatch)
        assert ids == ((actin, gapdh) if mirrored else (gapdh, actin))


@pytest.mark.parametrize("mirrored", [False, True], ids=["clicked-ltr", "clicked-rtl"])
def test_the_proteins_own_boxes_count_among_the_lanes_numbered(tmp_path, monkeypatch, mirrored):
    # The protein's boxes clicked in four lanes number them one way, another
    # protein's boxes in two lanes the other. (Read the clicked way and
    # committed, the clicked boxes would be the row's, and the same drag again
    # would read the lanes the other protein's way.)
    case = ROWS["all_present"]
    n = case.n_lanes
    s, image, protein = row_session(tmp_path, case)
    other = other_protein_in_lanes(s, image, case, (2, 3), mirrored=not mirrored)
    for true in (0, 1, 4, 5):
        lane = n - 1 - true if mirrored else true
        ops.place_box(s, protein, *at_lane(case, true), lane_index=lane, grow=True)
    names = ("'GAPDH'", "'β-catenin'") if mirrored else ("'β-catenin'", "'GAPDH'")
    ids = numbering_refused(s, protein, case.row, both_ways(*names), monkeypatch)
    assert ids == ((other, protein) if mirrored else (protein, other))


@pytest.mark.parametrize("mirrored", [False, True], ids=["clicked-ltr", "clicked-rtl"])
def test_a_box_dragged_far_enough_to_turn_its_proteins_lanes_refuses_a_row(
    tmp_path, monkeypatch, mirrored
):
    # Another protein's boxes in true lanes 2 and 3, numbered the clicked way,
    # its box in lane 3 then moved by hand to between lanes 0 and 1 as
    # numbered: on their own, its two boxes run the other way.
    case = ROWS["all_present"]
    n = case.n_lanes

    def lane(true: int) -> int:
        return n - 1 - true if mirrored else true

    s, image, protein, _ = clicked_lanes(tmp_path, mirrored)
    other = other_protein_in_lanes(s, image, case, (2, 3), mirrored=mirrored)
    between = round((case.lane_cx[lane(0)] + case.lane_cx[lane(1)]) / 2)
    edit_by_hand(s, lane_bands(s, other)[3].id, to=(between, 20))
    names = ("'GAPDH'", "'β-catenin'") if mirrored else ("'β-catenin'", "'GAPDH'")
    numbering_refused(s, protein, case.row, both_ways(*names), monkeypatch)


def test_one_lane_number_on_two_columns_refuses_a_row(tmp_path, monkeypatch):
    # The protein's box clicked over true lane 2 but numbered 3: lane 3 lies
    # in two columns a pitch (70 px) apart. The refusal names its boxes.
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    other = other_protein_in_lanes(s, image, case)
    clicked = ops.place_box(s, protein, *at_lane(case, 2), lane_index=3, grow=False)
    # Messages number lanes from 1: lane index 3 prints as lane 4.
    message = two_columns("lane 4", "'β-catenin' and 'GAPDH'", 35)
    ids = numbering_refused(s, protein, case.row, message, monkeypatch)
    assert ids == (clicked, lane_bands(s, other)[3].id)


@pytest.mark.parametrize(
    ("dx", "refused"),
    [(-28, False), (28, False), (-42, True), (42, True)],
    ids=["0.4-pitch-left", "0.4-pitch-right", "0.6-pitch-left", "0.6-pitch-right"],
)
def test_a_lanes_boxes_within_half_a_pitch_of_each_other_are_one_column(
    tmp_path, monkeypatch, dx, refused
):
    # A third protein's boxes in lanes 2 and 3, dx px from the other
    # protein's: 0.4 pitch apart they still show one column per lane.
    case = ROWS["all_present"]  # pitch 70
    s, image, protein = row_session(tmp_path, case)
    gapdh = other_protein_in_lanes(s, image, case)
    actin = other_protein_in_lanes(s, image, case, (2, 3), dx=dx, name="actin")
    if refused:
        # Messages number lanes from 1: lane indices 2 and 3 print as lanes 3 and 4.
        message = two_columns("lanes 3 and 4", "'GAPDH' and 'actin'", 35)
        ids = numbering_refused(s, protein, case.row, message, monkeypatch)
        assert ids == (
            lane_bands(s, gapdh)[2].id,
            lane_bands(s, actin)[2].id,
            lane_bands(s, gapdh)[3].id,
            lane_bands(s, actin)[3].id,
        )
    else:
        placement = ops.detect_row_boxes(s, protein, case.row)
        assert None not in placement.band_ids


@pytest.mark.parametrize("true", [0, 5], ids=["lane-0", "lane-5"])
def test_clicks_grown_wider_than_the_lane_pitch_are_not_among_the_lanes_numbered(tmp_path, true):
    # A click on a band of a row whose bands touch grows over the whole row:
    # the box, wider than the lane pitch (48 px), is centred on the row, not on
    # its lane. It does not show its lane's column; the row replaces it.
    case = ROWS["touching"]
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case)
    clicked = ops.place_box(s, protein, *at_lane(case, true), lane_index=true, grow=True)
    x0, _, x1, _ = rects_by_lane(s, protein)[true]
    assert x1 - x0 > 5 * 48 and abs((x0 + x1) / 2 - case.lane_cx[true]) > 48

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.replaced_band_ids == (clicked,)
    assert at_true_lanes(s, protein, case) == {lane: lane for lane in range(case.n_lanes)}


def test_a_mirrored_image_numbered_one_way_is_read_right_to_left(tmp_path):
    # Two proteins' boxes and the protein's own clicked ones all number the
    # lanes right to left.
    case = ROWS["all_present"]
    n = case.n_lanes
    s, image, protein, clicked = clicked_lanes(tmp_path, mirrored=True)
    other_protein_in_lanes(s, image, case, mirrored=True)
    other_protein_in_lanes(s, image, case, (0, 2, 5), mirrored=True, name="actin")

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.right_to_left
    assert sorted(placement.replaced_band_ids) == sorted(clicked)
    assert at_true_lanes(s, protein, case) == {n - 1 - true: true for true in range(n)}


def test_a_second_drag_corrects_a_row_the_lanes_placed_since_disagree_with(tmp_path):
    # The first drag, a lane off with nothing placed to check it, is
    # committed; another protein's boxes then placed in the true lanes put each
    # lane in two columns with the row's. The row's boxes give way to the next
    # drag whatever it finds, so they are not among the lanes numbered, and the
    # right drag replaces them.
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    first = ops.detect_row_boxes(s, protein, shifted(case, 70))
    other_protein_in_lanes(s, image, case)

    second = ops.detect_row_boxes(s, protein, case.row)
    assert set(second.replaced_band_ids) == {b for b in first.band_ids if b}
    assert all(read == true for read, true in at_true_lanes(s, protein, case).items())


def test_the_row_boxes_that_turn_the_next_drag_count_among_the_lanes_numbered(
    tmp_path, monkeypatch
):
    # Another protein's one box numbers the lanes right to left: lane 5 over
    # true lane 0. One box shows no direction, so the first drag reads the
    # lanes left to right and puts lane 5 over true lane 5. Its boxes then turn
    # the next drag, so they count among the lanes numbered: lane 5 lies in two
    # columns, and the same drag again is refused, as a third protein's is.
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    gapdh = other_protein_in_lanes(s, image, case, (0,), mirrored=True)
    first = ops.detect_row_boxes(s, protein, case.row)
    assert not first.right_to_left
    actin = ops.add_protein(s, "actin", Role.TARGET, image)
    # Messages number lanes from 1: lane index 5 prints as lane 6.
    message = two_columns("lane 6", "'β-catenin' and 'GAPDH'", 35)
    for row_of in (protein, actin):
        ids = numbering_refused(s, row_of, case.row, message, monkeypatch)
        assert ids == (first.band_ids[5], lane_bands(s, gapdh)[5].id)


def test_the_row_boxes_that_turn_the_next_drag_agreeing_with_the_lanes_placed(tmp_path):
    # Another protein's one box in lane 2 lies on the first drag's lane 2: the
    # row's boxes that turn the next drag agree with it, and the same drag
    # again changes nothing.
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case, (2,))
    first = ops.detect_row_boxes(s, protein, case.row)
    committed, length = s.project, len(s.project.log)

    again = ops.detect_row_boxes(s, protein, case.row)
    assert not again.right_to_left and again.band_ids == first.band_ids
    assert s.project is committed and len(s.project.log) == length


def test_fixed_boxes_wider_than_the_lane_pitch_count_among_the_lanes_numbered(
    tmp_path, monkeypatch
):
    # On touching bands (pitch 48) a new protein's box, an eighth of the image
    # width (55 px), is already wider than the pitch, but a box dropped where
    # the user clicked still sits on its lane. The protein's two fixed boxes
    # number the lanes right to left, another protein's boxes left to right.
    case = ROWS["touching"]
    s, image, protein = row_session(tmp_path, case)
    gapdh = other_protein_in_lanes(s, image, case, (2, 3, 5))
    for true, lane in ((5, 0), (1, 4)):
        ops.place_box(s, protein, *at_lane(case, true), lane_index=lane, grow=False)
    assert protein_of(s, protein).box_size.width > 48
    message = both_ways("'GAPDH'", "'β-catenin'")
    assert numbering_refused(s, protein, case.row, message, monkeypatch) == (gapdh, protein)


# Expected lane centres: pitch 70 left of lane 0, then 60, 80, 70, 70.
EXPECTED = {-1: -70.0, 0: 0.0, 1: 60.0, 2: 140.0, 3: 210.0, 4: 280.0}


@pytest.mark.parametrize(
    ("centres", "off"),
    [
        ({0: 0.0, 1: 60.0, 2: 140.0, 3: 210.0}, []),
        ({0: 29.0}, []),  # towards lane 1: within half of 60
        ({0: 31.0}, [0]),
        ({0: -34.0}, []),  # towards lane -1: within half of 70
        ({0: -36.0}, [0]),
        ({1: 99.0}, []),  # towards lane 2: within half of 80
        ({1: 101.0}, [1]),
        ({1: 31.0}, []),  # towards lane 0: within half of 60
        ({1: 29.0}, [1]),
        ({3: 210.0, 0: 31.0, 2: 99.0}, [0, 2]),
    ],
)
def test_a_band_is_off_its_lane_past_half_the_local_pitch(centres, off):
    assert ops._off_lanes(centres, EXPECTED) == off


def test_a_band_is_off_its_lane_on_lanes_numbered_right_to_left():
    anchors = [(330.0, 0), (270.0, 1), (190.0, 2)]
    expected = lane_positions(anchors, range(-1, 5))
    assert expected == {-1: 400.0, 0: 330.0, 1: 270.0, 2: 190.0, 3: 120.0, 4: 50.0}
    assert ops._off_lanes({0: 301.0, 1: 231.0, 2: 225.0, 3: 154.0}, expected) == []
    assert ops._off_lanes({0: 299.0, 1: 301.0, 2: 231.0, 3: 156.0}, expected) == [0, 1, 2, 3]
