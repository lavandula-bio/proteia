# SPDX-License-Identifier: Apache-2.0
"""Tests for the project operations and the session they work on.

No napari and no Qt: every test drives :mod:`proteia.core.operations` directly,
in project folders with non-ASCII names, on real TIFF files written by
``conftest.write_tiff``. The box tests use a 16-bit ``synthetic_blot`` with a
narrow and a wide band in one row; the parity tests at the end pin the whole
path, from an imported file to the chart, to the golden numbers of
``test_regression_baseline``.
"""

import codecs
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from conftest import make_project, synthetic_blot, write_image_files, write_tiff
from proteia.core import boxes, results, storage
from proteia.core import operations as ops
from proteia.core.analyze import ReduceMethod
from proteia.core.grow import grow_box
from proteia.core.model import (
    Box,
    BoxSize,
    ImageKind,
    Polarity,
    Project,
    ProposalSource,
    Role,
    UnknownIdError,
    apply_change,
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
from proteia.core.quantify import estimate_background, net_signal
from proteia.core.results import NoticeCode
from proteia.core.session import save_to_folder
from proteia.core.storage import load_project
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


def session_on(tmp_path: Path, hook=None) -> ProjectSession:
    """A new project in a non-ASCII folder; ``hook`` None means no autosave."""
    return ops.new_project(tmp_path / FOLDER, autosave=hook)


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
    """Every stored net equals net_signal recomputed from the session's pixels."""
    batch = session.project.batch
    for protein in batch.proteins:
        image = batch.find_image(protein.image_id)
        pixels = session.pixels(image.id)
        for band in protein.bands:
            expected = net_signal(
                pixels,
                band.box,
                protein.box_size,
                image.background,
                dark_on_light=image.polarity.dark_on_light,
            )
            assert band.net == expected, band.id


def open_sample(tmp_path: Path, hook=save_to_folder) -> ProjectSession:
    """The conftest sample project saved with stand-in image files, then opened."""
    folder = tmp_path / FOLDER
    project = make_project()
    write_image_files(folder, project)
    storage.save_project(project, folder)
    return ops.open_project(folder, autosave=hook)


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


def plant(session: ProjectSession, change) -> None:
    """Commit a change no operation makes yet (e.g. an apparent MW from #58)."""
    project, _ = apply_change(session.project, change)
    session._commit(project, action="plant")


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
    session = ops.new_project(folder)
    assert sorted(p.name for p in folder.iterdir()) == ["exports", "images", "project.json"]
    assert session.project == Project() == load_project(folder)
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
    assert ops.new_project(empty).project == Project()


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
]


@pytest.mark.parametrize(("setup", "call", "code"), REFUSALS)
def test_refused_operation_changes_nothing(tmp_path, setup, call, code):
    s, recorder, ids = _refusal_scene(tmp_path)
    if setup is not None:
        setup(s, ids)
    recorder.actions.clear()
    before = s.project
    next_id = before.next_id
    files = listing(s)
    cache = dict(s._pixels)

    with pytest.raises(UnknownIdError if code is None else OperationError) as info:
        call(s, ids)
    if code is not None:
        assert info.value.code is code
    assert s.project is before
    assert s.project.next_id == next_id
    assert recorder.actions == []
    assert listing(s) == files
    assert s._pixels.keys() == cache.keys()
    assert all(s._pixels[key] is array for key, array in cache.items())


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

    def mark_checked(draft: Project) -> None:  # #44 will set the flag
        for protein in draft.batch.proteins:
            for band in protein.bands:
                band.clipped = False

    plant(s, mark_checked)
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
        assert band.clipped is None
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
    s._commit(project, action="plant", evict=[image_id])
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


def test_export_lane_table(tmp_path):
    recorder = Recorder()
    s = open_sample(tmp_path, recorder)
    path = ops.export_lane_table(s)
    assert path == s.folder / "exports" / "lane-table.csv"
    data = path.read_bytes()
    assert data.startswith(codecs.BOM_UTF8)
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    assert rows[0] == ["lane", "condition", "sample", "include", "β-catenin", "α-tubulin", "GAPDH"]
    assert rows[1] == [
        "0",
        "vehicle",
        "v1",
        "yes",
        str(round(4279.740326695199, 3)),
        str(round(7389.877572928557, 3)),
        "5120.5",
    ]
    assert rows[3][:5] == ["2", "10 µM", "a1", "yes", ""]  # no β-catenin box in lane 2
    assert rows[4][3] == "no"
    assert len(rows) == 5
    assert recorder.actions == []  # not a state change

    path.unlink()
    path.mkdir()  # something in the way
    before = s.project
    with pytest.raises(OSError):
        ops.export_lane_table(s)
    assert s.project is before

    empty = session_on(tmp_path / "empty")
    with pytest.raises(OperationError) as info:
        ops.export_lane_table(empty)
    assert info.value.code is ErrorCode.NO_LANES


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
    # A box beyond the declared lanes is refused too, though lane 3 is free.
    with pytest.raises(OperationError) as info:
        ops.place_box(s, protein, lane_x(5), LANE_ROW, grow=False)
    assert info.value.code is ErrorCode.LANE_OUT_OF_RANGE


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
    for lane in range(LANES):
        ops.place_box(s, protein, lane_x(lane), LANE_ROW, lane_index=lane, grow=False)
    with pytest.raises(OperationError) as info:
        ops.place_box(s, protein, lane_x(0), LANE_ROW, grow=False)
    assert info.value.code is ErrorCode.LANE_OCCUPIED
    assert len(info.value.ids) == LANES
