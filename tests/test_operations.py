# SPDX-License-Identifier: Apache-2.0
"""Tests for the project operations and the session they work on.

With no front end, every test drives :mod:`proteia.core.operations` directly,
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
import errno
import functools
import hashlib
import inspect
import io
import json
import logging
import math
import os
import shutil
import struct
import sys
import threading
import weakref
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple
from unittest import mock

import numpy as np
import PIL.Image
import pytest
import tifffile

import proteia
from conftest import (
    V1_CONTENT_HASH,
    FakeClock,
    make_project,
    make_project_with_clashing_names,
    make_project_with_undetected,
    sample_doc_v1,
    stored_copy_or_path_in,
    synthetic_blot,
    write_image_files,
    write_lzw_tiff,
    write_tiff,
)
from proteia.core import boxes, imaging, mwcal, quantify, record, results, rowdetect, storage
from proteia.core import operations as ops
from proteia.core import session as session_module
from proteia.core.analyze import ReduceMethod
from proteia.core.grow import grow_box
from proteia.core.imaging import clipping_depth, possible_clipping_depth
from proteia.core.model import (
    Band,
    Box,
    BoxPadding,
    BoxSize,
    CalibrationPointSource,
    FitMethod,
    ImageKind,
    LadderSide,
    Polarity,
    Project,
    ProposalSource,
    Region,
    Role,
    UndetectedBand,
    UndetectedReason,
    UnknownIdError,
    apply_change,
    lanes_phrase,
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
from proteia.core.project import lane_anchor_ids, lane_anchors, lane_positions
from proteia.core.quantify import (
    RING_CLAMP,
    BandBackground,
    band_backgrounds,
    estimate_background,
    is_clipped,
    is_possibly_clipped,
    net_signal,
    quantify_nets,
    saturation_level,
)
from proteia.core.record import history_issues
from proteia.core.results import Level, NoticeCode
from proteia.core.rowdetect import DETECT_K, RowDetection
from proteia.core.session import save_to_folder
from proteia.core.storage import canonical_json, content_hash, load_project
from proteia.web.state import project_state
from rowcases import (
    MEMBRANE,
    STRESS,
    RowCase,
    adversarial,
    adversarial_row,
    band_between,
    bench_cases,
    beside,
    bottom_strip,
    hstripe,
    stress,
    synthetic_row,
    vstreak,
)
from test_background_truth import LANES as TRUTH_LANES
from test_background_truth import _blot as truth_blot
from test_background_truth import _fold_changes as truth_fold_changes
from test_background_truth import _worst as truth_worst
from test_imaging import _MPO_SIZES, _cmyk_profile, _declared_as, _mpo, _packed_tiff
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
from test_results import _chart_stats

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


def import_before_224(
    session: ProjectSession,
    pixels: np.ndarray,
    name: str = "β-actin 10 µM.tif",
    polarity: Polarity = DARK,
    **options,
) -> str:
    """:func:`import_blot` as Proteia imported before #224, which took a file
    whose pixels have no fixed range (floating point, or integers other than
    unsigned 8 or 16 bits) with an unknown_bit_depth warning: the same import
    without its refusal. The image's id."""
    with mock.patch.object(ops, "_refuse_pixel_type", lambda name, pixels: None):
        return import_blot(session, pixels, name, polarity, **options)


def assert_nets_current(session: ProjectSession, *, only: Sequence[str] | None = None) -> None:
    """The operations' invariant: every stored net, background (level, mode,
    spread) and clipping flag (exact or possible) equals, exactly, its value recomputed from the
    session's pixels, image by image, with every box on the image (every
    protein's, each with its size) measured together under the project's
    background method. ``only`` limits the check to those images.
    """
    project = session.project
    batch = project.batch
    legacy = project.background_method == "global_median"
    for image in batch.iter_images():
        if only is not None and image.id not in only:
            continue
        placed = [  # each band with its protein's boxes (their size)
            (batch.boxes_of(p), b)
            for p in batch.proteins
            if batch.image_id_of(p) == image.id
            for b in batch.boxes_of(p).bands
        ]
        if not placed:
            continue
        pixels = session.pixels(image.id)
        dark = image.polarity.dark_on_light
        rects = [band.box.rect(protein.box_size) for protein, band in placed]
        sizes = [(protein.box_size.width, protein.box_size.height) for protein, _ in placed]
        integral = image.bit_depth is not None
        if legacy:
            found = [BandBackground(image.background, "global_median", 0.0)] * len(placed)
            clamp = "pixel"
        else:
            found = band_backgrounds(
                pixels,
                rects,
                sizes,
                dark_on_light=dark,
                integral=integral,
                fallback=image.background,
            )
            clamp = RING_CLAMP
        # On the channel the session's pixels are of (#206).
        depth, near_depth = image.exact_check_depth, image.near_check_depth
        for (protein, band), background in zip(placed, found, strict=True):
            size = protein.box_size
            net = net_signal(
                pixels, band.box, size, background.level, dark_on_light=dark, clamp=clamp
            )
            stored = (band.net, band.background_level, band.background_mode, band.background_spread)
            assert stored == (net, background.level, background.mode, background.spread), band.id
            clipped = is_clipped(pixels, band.box, size, bit_depth=depth, dark_on_light=dark)
            assert band.clipped is clipped, band.id
            possibly = is_possibly_clipped(
                pixels, band.box, size, bit_depth=near_depth, dark_on_light=dark
            )
            assert band.possibly_clipped is possibly, band.id
        # The product's one entry point gives the same nets.
        method = "global_median" if legacy else "ring_median"
        nets = quantify_nets(
            pixels, rects, sizes, method=method, dark_on_light=dark, integral=integral
        )
        assert [band.net for _, band in placed] == nets, image.id


def expected_mw(session: ProjectSession, band_id: str) -> float | None:
    """A band's apparent MW recomputed (#58): its image's calibration at its
    box's centre, at its protein's box size; None without a curve or outside
    its range."""
    batch = session.project.batch
    protein, band = batch.find_band(band_id)
    image_id = batch.image_id_of(protein)
    fitted = mwcal.calibration_for(batch, batch.membrane_of(image_id), image_id)
    if not isinstance(fitted, mwcal.Calibration):
        return None
    size = batch.boxes_of(protein).box_size
    return fitted.mw_at(band.box.x + size.width / 2, band.box.y + size.height / 2)


def assert_mw_current(session: ProjectSession) -> None:
    """The operations' molecular-weight invariant (#58): every membrane's
    ``fit_method`` and ``fit_quality`` are those of its points
    (:func:`~proteia.core.mwcal.fit_method`, :func:`~proteia.core.mwcal.fit_quality`),
    and every band's apparent MW is, exactly, its image's calibration at the
    centre of its box (:func:`expected_mw`)."""
    batch = session.project.batch
    for membrane in batch.membranes:
        calibration = membrane.calibration
        assert calibration.fit_method is mwcal.fit_method(batch, membrane), membrane.id
        assert calibration.fit_quality == mwcal.fit_quality(batch, membrane), membrane.id
    for protein in batch.proteins:
        for band in batch.boxes_of(protein).bands:
            assert band.apparent_mw == expected_mw(session, band.id), band.id


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


def image_of(session: ProjectSession, protein_id: str) -> str:
    """The id of the image a protein is quantified on."""
    batch = session.project.batch
    return batch.image_id_of(batch.find_protein(protein_id))


def boxes_of(session: ProjectSession, protein_id: str):
    """A protein's boxes in the committed project (its row)."""
    batch = session.project.batch
    return batch.boxes_of(batch.find_protein(protein_id))


def _listed(cascade: Cascade) -> dict[str, list[str]]:
    """A cascade as its log entry records it: every field a list of ids."""
    return {name: list(ids) for name, ids in dataclasses.asdict(cascade).items()}


def plant(session: ProjectSession, change) -> None:
    """Commit a change no operation makes yet (e.g. an apparent MW from #58)."""
    project, _ = apply_change(session.project, change)
    session._commit(project, action="plant", params={})


def plant_58_fit(session: ProjectSession, membrane_id: str) -> None:
    """Store a membrane's fit and its bands' apparent MWs as a #58 build
    computes them: the sample's mem-1 holds a fit from before #58, whose refit
    counts as a change to the curve of every image of it."""

    def change(draft: Project) -> None:
        batch = draft.batch
        membrane = next(m for m in batch.membranes if m.id == membrane_id)
        membrane.calibration.fit_method = mwcal.fit_method(batch, membrane)
        membrane.calibration.fit_quality = mwcal.fit_quality(batch, membrane)
        images = {image.id for image in batch.images_of(membrane)}
        for protein in batch.proteins:
            image_id = batch.image_id_of(protein)
            if image_id in images:
                fitted = mwcal.calibration_for(batch, membrane, image_id)
                boxes = batch.boxes_of(protein)
                for band in boxes.bands:
                    band.apparent_mw = ops._apparent_mw(fitted, band.box, boxes.box_size)

    plant(session, change)


def plant_legacy(session: ProjectSession) -> None:
    """Make the project one quantified before #83 (a migrated schema-1 project):
    the legacy method, every band on every image quantified under it."""
    batch = session.project.batch
    arrays = {
        image.id: session.pixels(image.id)
        for image in batch.iter_images()
        if any(batch.boxes_of(p).bands for p in batch.proteins if batch.image_id_of(p) == image.id)
    }

    def change(draft: Project) -> None:
        draft.background_method = "global_median"
        for image_id, array in arrays.items():
            ops._quantify_image(draft, image_id, array)

    plant(session, change)


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


def v1_folder(tmp_path: Path) -> Path:
    """The conftest sample as a schema-1 build (before #83) saved it, with its
    stand-in image files; its log begins with its creation, whose hash is the
    one that build computed."""
    folder = tmp_path / FOLDER
    write_image_files(folder, make_project())
    created = {
        "seq": 1,
        "time": "2026-09-20T08:00:00.000Z",
        "action": "new_project",
        "version": "0.1.0.dev0",
        "params": {},
        "content_hash": V1_CONTENT_HASH,
    }
    doc = {**sample_doc_v1(), "log": [created]}
    (folder / storage.PROJECT_FILE).write_bytes(storage.document_bytes(doc))
    return folder


def test_opening_a_schema_1_project_saves_its_migration_at_once(tmp_path):
    folder = v1_folder(tmp_path)
    recorder = Recorder()

    def hook(session: ProjectSession) -> None:
        recorder(session)
        save_to_folder(session)

    s = ops.open_project(folder, autosave=hook, clock=FakeClock())
    migrated = s.project.log[-1]
    assert (migrated.seq, migrated.action) == (2, "migrate")
    assert migrated.time == "2026-09-26T08:00:00.000Z"  # from the session's clock
    # A logged change like any other: autosaved at once, entry and all.
    assert recorder.actions == ["migrate"]
    assert (s.dirty, s.save_error, s.last_action) == (False, None, "migrate")
    assert load_project(folder) == s.project
    assert project_state(folder.name, s, s.project, open_id=1)["saved"] is True

    # A record exported before any edit cites the log the file holds, and the
    # next session neither migrates again nor logs another entry.
    ops.export_lane_table(s)
    exported = json.loads((folder / storage.EXPORTS_DIR / ops.LANE_TABLE_RECORD_FILE).read_bytes())
    s.close()
    reopened = ops.open_project(folder, clock=FakeClock(datetime(2026, 11, 5, tzinfo=UTC)))
    assert reopened.project == s.project and not reopened.dirty
    ops.set_reference_condition(reopened, None)
    saved = [entry.model_dump(mode="json") for entry in load_project(folder).log]
    assert saved[: len(exported["log"])] == exported["log"]
    assert exported["history_issues"] == [] == history_issues(load_project(folder))
    # Its edits come after the migration in time too.
    assert reopened.project.log[-1].time > migrated.time


def test_a_migration_is_unsaved_until_a_save_without_autosave(tmp_path):
    folder = v1_folder(tmp_path)
    before = (folder / storage.PROJECT_FILE).read_bytes()
    s = ops.open_project(folder, autosave=None, clock=FakeClock())
    assert s.project.log[-1].action == "migrate"
    # project.json does not hold the migrated project yet.
    assert (s.dirty, s.save_error) == (True, None)
    assert (folder / storage.PROJECT_FILE).read_bytes() == before
    ops.save(s)
    assert not s.dirty and load_project(folder) == s.project


def test_a_failed_save_of_a_migration_keeps_it_in_memory(tmp_path, replace_lock):
    folder = v1_folder(tmp_path)
    before = (folder / storage.PROJECT_FILE).read_bytes()
    replace_lock.locked = True
    s = ops.open_project(folder, clock=FakeClock())  # opens all the same
    assert s.dirty and isinstance(s.save_error, PermissionError)
    assert (folder / storage.PROJECT_FILE).read_bytes() == before

    replace_lock.locked = False
    ops.set_reference_condition(s, None)  # the next change saves both
    assert (s.dirty, s.save_error) == (False, None)
    assert [entry.action for entry in load_project(folder).log] == [
        "new_project",
        "migrate",
        "set_reference_condition",
    ]


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
    ops.move_box(s, band, band_of(s, band).box.rect(boxes_of(s, protein).box_size))  # no-op
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
    ops.place_box(s, target, NARROW_X, ROW, lane_index=0, grow=True)
    ops.place_box(s, target, WIDE_X, ROW, lane_index=1, grow=True)
    # The loading control's boxes lower down: two proteins' boxes may not
    # measure one band (#114).
    for lane, x in enumerate((NARROW_X, WIDE_X)):
        ops.place_box(s, loading, x, ROW + 6, lane_index=lane, grow=False)
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
    size = boxes_of(s, protein).box_size
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


def test_reload_reads_project_json_again_only_once_changed_outside_proteia(tmp_path):
    s, image, protein = boxed(tmp_path, save_to_folder)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    s.pixels(image)
    project = s.project
    assert not s.reload()  # the file holds what the session saved
    assert s.project is project and s.undo_step is not None and image in s._pixels

    # A copy of the folder (another machine, a restore) changes the project.
    other = tmp_path / "copy α"
    shutil.copytree(s.folder, other)
    remote = ops.open_project(other, clock=FakeClock())
    ops.remove_protein(remote, protein)
    newer = import_blot(remote, blot(), "remote β.tif")
    shutil.copytree(other, s.folder, dirs_exist_ok=True)

    assert s.reload()
    assert s.project == remote.project
    assert (s.undo_step, s.redo_step, s._pixels, s.dirty, s.last_action) == (
        None,
        None,
        {},
        False,
        None,
    )
    assert not s.reload()
    ops.set_polarity(s, newer, LIGHT)  # the next change saves the file's project
    assert load_project(s.folder) == s.project
    assert listing(s) == sorted([f"{image}.tif", f"{newer}.tif"])
    ops.undo(s)  # the history begins at the project read again
    assert s.project.batch.find_image(newer).polarity == DARK
    with pytest.raises(OperationError) as info:
        ops.undo(s)
    assert info.value.code is ErrorCode.NOTHING_TO_UNDO


def test_reload_keeps_unsaved_changes_and_a_session_it_cannot_read(tmp_path, replace_lock):
    s, _, protein = boxed(tmp_path, save_to_folder)
    replace_lock.locked = True
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    project, steps = s.project, s.history_steps
    assert s.dirty
    assert not s.reload()  # project.json is older than the session: not read
    assert (s.project, s.history_steps, s.dirty) == (project, steps, True)

    replace_lock.locked = False
    ops.save(s)
    for broken in (b"{not json", b"{}"):
        (s.folder / storage.PROJECT_FILE).write_bytes(broken)
        with pytest.raises(storage.ProjectError):
            s.reload()
        assert (s.project, s.history_steps, s.dirty) == (project, steps, False)


def test_reload_migrates_and_saves_a_file_of_an_older_schema(tmp_path):
    folder = v1_folder(tmp_path)
    v1 = (folder / storage.PROJECT_FILE).read_bytes()
    s = ops.open_project(folder, clock=FakeClock())  # migrated and saved
    ops.set_reference_condition(s, None)
    (folder / storage.PROJECT_FILE).write_bytes(v1)  # an old copy restored

    assert s.reload()
    assert [entry.action for entry in s.project.log] == ["new_project", "migrate"]
    assert (s.dirty, s.last_action, s.undo_step) == (False, "migrate", None)
    assert load_project(folder) == s.project


# --- #140: an older project.json is kept before its migration ---

BACKUP = "project.schema1.json"


def backups(folder: Path) -> list[str]:
    """The backups of an older project.json in ``folder``, by name."""
    return sorted(path.name for path in folder.iterdir() if path.name.startswith("project.schema"))


def files_in(folder: Path) -> dict[str, bytes]:
    """Every file under ``folder``, by its path relative to it, with its bytes."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


def test_opening_a_schema_1_project_keeps_its_project_json_first(tmp_path):
    folder = v1_folder(tmp_path)
    v1 = (folder / storage.PROJECT_FILE).read_bytes()
    s = ops.open_project(folder, clock=FakeClock())  # migrated and saved
    assert backups(folder) == [BACKUP]
    assert (folder / BACKUP).read_bytes() == v1  # byte for byte
    migrated = s.project.log[-1]
    assert migrated.params == {
        "from_schema": 1,
        "to_schema": 3,
        "from_content_hash": V1_CONTENT_HASH,
        "backup": BACKUP,
    }
    assert load_project(folder) == s.project  # project.json names it too
    assert history_issues(s.project) == []

    # The migrated project opens as it was saved, and keeps no other backup.
    s.close()
    reopened = ops.open_project(folder, clock=FakeClock())
    assert reopened.project == s.project and not reopened.dirty
    ops.set_reference_condition(reopened, None)
    reopened.close()
    assert backups(folder) == [BACKUP]
    assert (folder / BACKUP).read_bytes() == v1

    # Renamed back, the backup is the project as it was: migrated again, to
    # the same content, and kept again.
    (folder / BACKUP).replace(folder / storage.PROJECT_FILE)
    restored = ops.open_project(folder, clock=FakeClock())
    assert restored.project.log[-1].content_hash == migrated.content_hash
    assert (folder / BACKUP).read_bytes() == v1


def test_a_project_of_the_current_schema_gets_no_backup(tmp_path):
    folder = tmp_path / FOLDER
    project = make_project()
    write_image_files(folder, project)
    storage.save_project(project, folder)
    s = ops.open_project(folder)
    ops.set_reference_condition(s, None)
    assert backups(folder) == []
    assert "backup" not in json.dumps([entry.params for entry in s.project.log])


def test_a_second_migrating_open_never_replaces_a_backup(tmp_path):
    folder = v1_folder(tmp_path)
    path = folder / storage.PROJECT_FILE
    v1 = path.read_bytes()
    ops.open_project(folder, clock=FakeClock()).close()

    # The same file restored: the backup holding it already is the one named.
    path.write_bytes(v1)
    again = ops.open_project(folder, clock=FakeClock())
    assert again.project.log[-1].params["backup"] == BACKUP
    assert backups(folder) == [BACKUP]
    again.close()

    # Other bytes of schema 1 (a BOM and CRLF, as a Windows editor saves them):
    # a backup of their own, numbered as export folders are; the first stays.
    edited = codecs.BOM_UTF8 + v1.replace(b"\n", b"\r\n")
    path.write_bytes(edited)
    s = ops.open_project(folder, clock=FakeClock())
    second = "project.schema1 (2).json"
    assert s.project.log[-1].params["backup"] == second
    assert backups(folder) == [second, BACKUP]
    assert (folder / BACKUP).read_bytes() == v1
    assert (folder / second).read_bytes() == edited

    # Read again while open, the first file names the first backup.
    ops.set_reference_condition(s, None)
    path.write_bytes(v1)
    assert s.reload()
    assert s.project.log[-1].params["backup"] == BACKUP
    assert backups(folder) == [second, BACKUP]
    assert (folder / BACKUP).read_bytes() == v1


def _refuse_backups(monkeypatch, failure: str) -> None:
    """Make writing a backup fail: its file cannot be created (``create``, as in
    a folder that cannot be written), or fails once created (``write``, as on a
    full disk)."""
    real_open = os.open

    def refuse(path, flags, *args, **kwargs):
        if Path(path).name.startswith("project.schema"):
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    def fail(fd):
        raise OSError(28, "No space left on device")

    if failure == "create":
        monkeypatch.setattr(storage.os, "open", refuse)
    else:
        monkeypatch.setattr(storage.os, "fsync", fail)


@pytest.mark.parametrize(
    ("failure", "reason"), [("create", "Permission denied"), ("write", "No space left on device")]
)
def test_an_open_that_cannot_keep_the_backup_changes_nothing(
    tmp_path, monkeypatch, failure, reason
):
    folder = v1_folder(tmp_path)
    before = files_in(folder)
    recorder = Recorder()
    with monkeypatch.context() as patch:
        _refuse_backups(patch, failure)
        with pytest.raises(storage.BackupError) as info:
            ops.open_project(folder, autosave=recorder, clock=FakeClock())
    assert isinstance(info.value, OSError)  # answered as a file error
    assert str(info.value) == (
        f"{FOLDER!r} was saved by an older Proteia (schema 1), and a copy of its"
        f" project.json could not be kept before updating it ({reason}); nothing was changed"
    )
    assert str(tmp_path) not in str(info.value)  # the folder by name, never its path
    assert recorder.actions == []
    assert files_in(folder) == before  # no backup, not even a partial one

    # Once it can be written, the project opens.
    s = ops.open_project(folder, clock=FakeClock())
    assert s.project.log[-1].params["backup"] == BACKUP


@pytest.mark.parametrize("failure", ["create", "write"])
def test_a_reload_that_cannot_keep_the_backup_keeps_the_session(tmp_path, monkeypatch, failure):
    folder = v1_folder(tmp_path)
    v1 = (folder / storage.PROJECT_FILE).read_bytes()
    s = ops.open_project(folder, clock=FakeClock())
    ops.set_reference_condition(s, None)
    (folder / BACKUP).unlink()  # deleted by hand
    (folder / storage.PROJECT_FILE).write_bytes(v1)  # an old copy restored
    before, project, steps = files_in(folder), s.project, s.history_steps
    with monkeypatch.context() as patch:
        _refuse_backups(patch, failure)
        with pytest.raises(storage.BackupError):
            s.reload()
    assert (s.project, s.history_steps, s.dirty, s.save_error) == (project, steps, False, None)
    assert files_in(folder) == before
    assert s.reload() and (folder / BACKUP).read_bytes() == v1


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="POSIX permission bits, which root ignores"
)
def test_a_folder_that_cannot_be_written_refuses_to_open_an_older_project(tmp_path):
    folder = v1_folder(tmp_path)
    before = files_in(folder)
    folder.chmod(0o555)
    try:
        with pytest.raises(storage.BackupError, match="could not be kept"):
            ops.open_project(folder, clock=FakeClock())
    finally:
        folder.chmod(0o755)
    assert files_in(folder) == before


def _dropping_img_4(doc: dict) -> dict:
    """A faulty schema 1 to 2 step: it loses the reprobe (img-4) and GAPDH on it."""
    doc = storage._v1_to_v2(doc)
    batch = doc["batch"]
    for membrane in batch["membranes"]:
        membrane["images"] = [image for image in membrane["images"] if image["id"] != "img-4"]
    batch["proteins"] = [protein for protein in batch["proteins"] if protein["image_id"] != "img-4"]
    return doc


def test_the_cleanup_keeps_every_image_file_a_backup_names(tmp_path, monkeypatch):
    # A migration that dropped an image reference would leave its file to the
    # cleanup of the save that follows: the backup could not be restored.
    monkeypatch.setitem(storage.MIGRATIONS, 1, _dropping_img_4)
    folder = v1_folder(tmp_path)
    reprobe = folder / "images" / "img-4.tif"
    s = ops.open_project(folder, clock=FakeClock())  # migrated, saved and cleaned up
    assert "img-4" not in {image.id for image in s.project.batch.iter_images()}
    assert reprobe.exists()
    # Nor at the saves and imports that follow, in this session or the next.
    ops.set_reference_condition(s, None)
    import_blot(s, blot(), "β.tif")
    s.close()
    assert reprobe.exists()
    reopened = ops.open_project(folder, clock=FakeClock())
    ops.set_reference_condition(reopened, "vehicle")
    assert reprobe.exists()

    # Once the backup is deleted, the file is an orphan like any other.
    (folder / BACKUP).unlink()
    ops.set_reference_condition(reopened, None)
    assert not reprobe.exists()


def test_the_cleanup_keeps_an_image_file_named_in_any_backup(tmp_path):
    folder = v1_folder(tmp_path)
    s = ops.open_project(folder, clock=FakeClock())  # migrated and saved: next_id 27
    images = folder / "images"
    for name in ("img-7.tif", "IMG-8.TIFF", "img-9.png", "img-27.tif", "img-33.tif"):
        (images / name).write_bytes(b"an image no project.json references")
    # A backup is read as bytes, whatever its schema, even one that no longer
    # loads; its name is compared ignoring case, as Windows does. Only an id
    # below next_id is kept: a backup's images hold no other (a migration only
    # raises next_id, and so does the project), and the next import takes 27.
    (folder / "Project.Schema3 (2).json").write_bytes(
        b'{"broken": "img-7.tif", img-8.tiff, img-27.tif, img-33.tif'
    )
    ops.set_reference_condition(s, None)  # saved, then cleaned up
    assert listing(s) == [
        "IMG-8.TIFF",
        "img-2.tif",
        "img-3.png",
        "img-4.tif",
        "img-6.jpg",
        "img-7.tif",
    ]


def _crash_after_an_import(folder: Path) -> None:
    """Import into the project in ``folder`` and never save, as a crash leaves
    it: the image's file stays in images/, at the id the next import takes."""
    crashed = ops.open_project(folder, autosave=None, clock=FakeClock())
    import_blot(crashed, blot(), "lost.tif")


def test_a_stored_name_in_a_backup_s_text_never_blocks_the_next_import(tmp_path):
    # An original name kept in a backup reads as a stored name too, here the
    # name of the id the next import takes: the leftover file of that id goes
    # all the same, as no backup's image can hold that id.
    folder = v1_folder(tmp_path)
    path = folder / storage.PROJECT_FILE
    doc = json.loads(path.read_bytes())
    doc["batch"]["membranes"][0]["images"][0]["original_name"] = "Blot IMG-27.TIF"
    path.write_bytes(storage.document_bytes(doc))
    ops.open_project(folder, clock=FakeClock()).close()  # migrated and saved: next_id 27
    assert b"IMG-27.TIF" in (folder / BACKUP).read_bytes()
    _crash_after_an_import(folder)
    leftover = folder / "images" / "img-27.tif"
    assert leftover.exists()

    s = ops.open_project(folder, clock=FakeClock())
    assert import_blot(s, blot(), "β.tif") == "img-27"
    assert load_project(folder) == s.project
    assert leftover.read_bytes() == (folder.parent / "sources" / "β.tif").read_bytes()


def test_a_backup_that_cannot_be_read_never_blocks_the_next_import(tmp_path, monkeypatch):
    # What such a backup names is unknown, so the cleanup keeps every file it
    # could name: none at or above next_id, which no backup's image holds.
    folder = v1_folder(tmp_path)
    ops.open_project(folder, clock=FakeClock()).close()  # migrated and saved: next_id 27
    _crash_after_an_import(folder)  # img-27.tif left behind
    below = folder / "images" / "img-5.tif"
    below.write_bytes(b"an image no project.json references")
    read_bytes = Path.read_bytes

    def locked(path: Path) -> bytes:
        if storage.is_backup(path.name):  # as while another program holds it
            raise PermissionError(13, "Permission denied", str(path))
        return read_bytes(path)

    s = ops.open_project(folder, clock=FakeClock())
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", locked)
        assert import_blot(s, blot(), "β.tif") == "img-27"  # saved, then cleaned up
        assert below.exists()
    # Once the backup can be read, it does not name img-5: the file goes.
    ops.set_reference_condition(s, None)
    assert not below.exists()


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
        lambda s, ids: ops.set_box_padding(s, ids["target"], along=3),  # the fitted height is 5
        ErrorCode.INVALID_INPUT,
        id="padding-above-half",
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
    lanes = before.batch.only_lanes()
    target = before.batch.boxes_of(before.batch.find_protein(ids["target"]))
    ops.set_lanes(s, [LaneInput(lane.label, lane.sample, lane.included) for lane in lanes])
    ops.move_box(s, ids["a"], before.batch.find_band(ids["a"])[1].box.rect(target.box_size))
    ops.set_box_size(s, ids["target"], target.box_size)
    ops.set_box_padding(s, ids["target"], across=0, along=0)
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
    rows = {s.project.batch.only_row(p).id for p in s.project.batch.proteins}
    assert len(rows) == 8 and not rows & set(ids)
    assert s.project.next_id == start + 16  # each protein and its row
    assert load_project(s.folder) == s.project
    log = s.project.log
    assert [entry.seq for entry in log] == list(range(1, len(log) + 1))
    added = [entry.params["protein_id"] for entry in log if entry.action == "add_protein"]
    assert added == [p.id for p in s.project.batch.proteins]  # in commit order


# --- images ---


def test_import_records_the_stored_file(tmp_path):
    s = session_on(tmp_path)
    pixels = blot()
    name = "β-actin 10 µM.tif"
    image_id = import_blot(s, pixels, name, LIGHT)
    assert image_id == "img-1"
    [membrane] = s.project.batch.membranes
    assert membrane.id == "mem-2"
    [image] = s.project.batch.images_of(membrane)
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

    float_id = import_before_224(s, pixels.astype(np.float64), "float α.tif")  # as before #224
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
    assert [image.id for image in s.project.batch.images_of(only)] == [first, second]

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
    assert a_file.exists()
    # Closed while saving still fails, as when the server stops after a failed
    # flush: no undo state keeps the file now, but the project.json on disk does.
    s.close()
    assert a_file.exists()
    crashed = load_project(s.folder)  # as if the app died now
    assert [image.id for image in crashed.batch.iter_images()] == [a]

    replace_lock.locked = False
    ops.save(s)
    assert not a_file.exists()  # saved without it, and no undo state keeps it
    assert listing(s) == [f"{b}.tif"]


def test_remove_image_cascade(tmp_path):
    s = open_sample(tmp_path / "tubulin")
    cascade = ops.remove_image(s, "img-6")
    assert cascade == Cascade(
        removed=(
            "img-6",
            "arena-23",
            "prot-8",
            "row-25",
            "band-13",
            "band-14",
            "band-15",
            "band-16",
            "mem-5",
        ),
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
    removed = s.folder / "images" / "img-6.jpg"
    assert removed.exists()  # autosaved without it, but the removal can still be undone
    assert load_project(s.folder) == s.project
    # prot-7 falls back to GAPDH, now the batch's single loading control.
    assert [(x.target_id, x.loading_id) for x in ops.compute(s).series] == [("prot-7", "prot-9")]
    s.close()
    assert not removed.exists()  # deleted once the removal can no longer be undone

    s = open_sample(tmp_path / "marker")
    cascade = ops.remove_image(s, "img-3")
    assert cascade == Cascade(
        removed=("img-3", "arena-21"),
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
    assert (s.folder / "images" / "img-3.png").exists()
    s.close()
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
            for band in draft.batch.boxes_of(protein).bands:
                band.clipped = True

    plant(s, mark_clipped)
    background = s.project.batch.find_image(bright).background
    dark_bands = boxes_of(s, on_dark).bands

    ops.set_polarity(s, bright, LIGHT)
    image = s.project.batch.find_image(bright)
    assert image.polarity is LIGHT and image.background == background
    for band in boxes_of(s, on_bright).bands:
        assert band.net > 0  # bright bands now read as signal
        assert band.clipped is False  # recomputed for the new polarity (16-bit, below 65535)
    assert_nets_current(s, only=[bright])  # every band on it, the other signal direction
    assert boxes_of(s, on_dark).bands == dark_bands  # the other image is untouched


# --- The looks_processed warning at import and on a polarity change (#127) ---


def processed(s: ProjectSession, image_id: str) -> str | None:
    """The image's looks_processed warning message; None without one."""
    image = s.project.batch.find_image(image_id)
    found = [w.message for w in image.import_warnings if w.code == "looks_processed"]
    return found[0] if found else None


def rows_at(background: int, dtype, share: float, level: int) -> np.ndarray:
    """A 40 x 200 image at ``background`` whose top ``share`` of rows is at
    ``level``, as saturated band cores or a figure's panel cover it."""
    pixels = np.full((40, 200), background, dtype=dtype)
    pixels[: round(share * 40)] = level
    return pixels


FIGURE_PREFIX = "This looks like a processed figure rather than a raw scan: "


@pytest.mark.parametrize(
    ("case", "pixels", "polarity"),
    [
        # A 16-bit membrane below mid-range, with band cores cut at 0.
        ("membrane below mid-range", rows_at(20000, np.uint16, 0.2, 0), DARK),
        # A crop the over-exposed band fills: its median is the band's 0.
        ("crop of a saturated band", rows_at(215, np.uint8, 0.65, 0), DARK),
        # A long exposure: light bands at 65535 on a background raised past mid-range.
        ("raised background", rows_at(34000, np.uint16, 0.2, 65535), LIGHT),
        ("light crop of a saturated band", rows_at(9000, np.uint16, 0.65, 65535), LIGHT),
    ],
)
def test_saturated_bands_never_look_processed_whichever_half_the_median_is_in(
    tmp_path, case, pixels, polarity
):
    # The background's end of the range comes from the polarity, not from the
    # half of the range the median falls in: over-exposed band cores, at the
    # other end, never count as a background levelled to the limit.
    s = session_on(tmp_path)
    image = import_blot(s, pixels, f"{case} µ.tif", polarity)
    assert processed(s, image) is None
    assert s.project.batch.find_image(image).import_warnings == []


def test_a_levelled_background_looks_processed_whichever_half_the_median_is_in(tmp_path):
    # Dark bands filling most of a crop whose background was levelled to white:
    # the median is the bands', and the background still counts.
    s = session_on(tmp_path)
    white = import_blot(s, rows_at(255, np.uint8, 0.7, 60), "white α.tif", DARK)
    assert processed(s, white) == (
        f"{FIGURE_PREFIX}30% of its pixels are pure white. Its background may then not be"
        " the membrane, and over-exposure may be hidden; quantify the original scan if you"
        " have it."
    )
    black = import_blot(s, rows_at(0, np.uint16, 0.7, 50000), "black β.tif", LIGHT)
    assert processed(s, black).startswith(f"{FIGURE_PREFIX}30% of its pixels are pure black.")


def test_a_polarity_change_reassesses_the_looks_processed_warning(tmp_path):
    from PIL import Image

    s = session_on(tmp_path)
    figure = rows_at(255, np.uint8, 0.2, 90)  # a background levelled to white
    image = import_blot(s, figure, "figure α.tif", LIGHT)  # the wrong polarity chosen
    assert processed(s, image) is None  # nothing at the limit it names: black
    ops.set_polarity(s, image, DARK)
    assert processed(s, image).startswith(
        f"{FIGURE_PREFIX}its median level, the image-wide background, is pure white; 80% of"
        " its pixels are pure white."
    )
    ops.undo(s)
    assert processed(s, image) is None
    ops.redo(s)
    assert processed(s, image) is not None
    # With bands on the image, whose re-quantification reads its pixels too.
    ops.set_lanes(s, [LaneInput("vehicle")])
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    ops.place_box(s, protein, NARROW_X, 20, lane_index=0, grow=False)
    ops.set_polarity(s, image, LIGHT)
    assert processed(s, image) is None

    # The other warnings stay, in their order; the new one comes last.
    jpeg = import_encoded(s, "figure β.jpg", Image.fromarray(figure), LIGHT, quality=95)
    assert [w.code for w in s.project.batch.find_image(jpeg).import_warnings] == ["lossy_format"]
    ops.set_polarity(s, jpeg, DARK)
    codes = [w.code for w in s.project.batch.find_image(jpeg).import_warnings]
    assert codes == ["lossy_format", "looks_processed"]
    assert "80% of its pixels are at or near pure white" in processed(s, jpeg)

    # A palette's levels do not depend on the polarity: that signal stays.
    even = np.repeat(np.arange(0, 255, 2, dtype=np.uint8), 20)[None, :]
    palette = import_encoded(s, "palette γ.png", Image.fromarray(even).convert("P"), LIGHT)
    only_palette = (
        f"{FIGURE_PREFIX}it is a palette image using only 128 of the 255 grey levels in its range."
    )
    assert processed(s, palette).startswith(only_palette)
    ops.set_polarity(s, palette, DARK)
    assert processed(s, palette).startswith(only_palette)


def test_a_polarity_change_drops_the_looks_processed_warning_of_an_image_it_cannot_read(
    tmp_path,
):
    # No band needs its pixels, so the polarity still changes, as before #127;
    # every analysis of the image is refused until its file is back. The
    # looks_processed warning, assessed for the old polarity, would name the
    # wrong limit: it goes; the other warnings stay.
    from PIL import Image

    s = session_on(tmp_path, save_to_folder)
    figure = rows_at(255, np.uint8, 0.2, 90)
    image = import_encoded(s, "figure α.jpg", Image.fromarray(figure), DARK, quality=95)
    warnings = s.project.batch.find_image(image).import_warnings
    assert [w.code for w in warnings] == ["lossy_format", "looks_processed"]
    reopened = ops.open_project(s.folder)
    (reopened.folder / "images" / f"{image}.jpg").unlink()
    ops.set_polarity(reopened, image, LIGHT)
    stored = reopened.project.batch.find_image(image)
    assert stored.polarity is LIGHT
    assert stored.import_warnings == [warnings[0]]


def test_a_polarity_change_on_an_image_with_no_band_keeps_no_pixels(tmp_path):
    # The warning is assessed from the pixels, which nothing else on an image
    # with no band needs: they are read, not kept in the session's cache.
    s = session_on(tmp_path, save_to_folder)
    image = import_blot(s, rows_at(255, np.uint8, 0.2, 90), "figure α.tif", DARK)
    reopened = ops.open_project(s.folder)
    ops.set_polarity(reopened, image, LIGHT)
    assert image not in reopened._pixels


def test_an_import_takes_the_image_median_once(tmp_path, monkeypatch):
    # The looks_processed warning reuses the background the import stores, the
    # image's median, which on a large image costs a full copy of it.
    s = session_on(tmp_path)
    calls = []
    median = np.median

    def counted(*args, **kwargs):
        calls.append(args[0].shape)
        return median(*args, **kwargs)

    monkeypatch.setattr(np, "median", counted)
    image = import_blot(s, rows_at(255, np.uint8, 0.2, 90), "figure α.tif", DARK)
    assert calls == [(40, 200)]
    assert processed(s, image) is not None


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
    with pytest.raises(OperationError) as info:
        s.colour_pixels(image_id)
    assert (info.value.code, info.value.ids) == (ErrorCode.IMAGE_FILE_CHANGED, (image_id,))


def test_colour_pixels_are_the_files_own_read_with_the_same_checks(tmp_path):
    # For showing a colour file in its own colours (#57): the pixels as stored,
    # channels last, never cached; refused as the analysis array would be.
    s = session_on(tmp_path)
    gray = blot()
    rgb = np.dstack([gray, gray // 2, gray // 3])
    image_id = import_blot(s, rgb, "stain.tif")
    s._pixels.clear()
    stored = s.colour_pixels(image_id)
    assert stored.shape == (H, W, 3) and np.array_equal(stored, rgb)
    assert s._pixels == {}  # display only: nothing kept
    path = s.folder / storage.IMAGES_DIR / f"{image_id}.tif"
    write_tiff(path, rgb[:, ::-1])  # other bytes, same shape
    for problem in ("other bytes", "missing"):
        if problem == "missing":
            path.unlink()
        with pytest.raises(OperationError) as info:
            s.colour_pixels(image_id)
        assert (info.value.code, info.value.ids) == (ErrorCode.IMAGE_FILE_CHANGED, (image_id,))
    with pytest.raises(UnknownIdError):
        s.colour_pixels("img-99")


def test_a_palette_tiff_is_analysed_as_its_indices_and_shown_through_its_map(tmp_path):
    # An 8-bit TIFF with a colour lookup table, as ImageJ saves one: the nets
    # are measured on the indices, its own colours come from the map (#57).
    s = session_on(tmp_path)
    indices = (blot() // 257).astype(np.uint8)
    level = np.arange(256, dtype=np.uint16)
    colormap = np.stack([level * 257, level * 128, (255 - level) * 257])  # 16-bit, as TIFF keeps it
    data = io.BytesIO()
    tifffile.imwrite(data, indices, photometric="palette", colormap=colormap)
    data.seek(0)
    image_id = ops.import_image(s, data, "fire µ.tif", kind=CHEMI, polarity=DARK)
    np.testing.assert_array_equal(s.pixels(image_id), indices)
    shown = s.colour_pixels(image_id)
    assert (shown.shape, shown.dtype) == ((H, W, 3), np.uint8)
    np.testing.assert_array_equal(shown, (colormap.T >> 8).astype(np.uint8)[indices])
    assert s.file_colours(s.project.batch.find_image(image_id)) == "palette"


def _cmyk_tiff(gray: np.ndarray) -> io.BytesIO:
    """A 16-bit gray blot as a CMYK TIFF, in black ink alone."""
    data = io.BytesIO()
    inks = np.zeros((*gray.shape, 4), dtype=np.uint16)
    inks[..., 3] = 65535 - gray
    tifffile.imwrite(data, inks, photometric="separated")
    data.seek(0)
    return data


def _before_207(s: ProjectSession, source, name: str, *, kind: ImageKind = CHEMI, **options):
    """Import ``source`` as Proteia did from #131 to #207, which converted a CMYK
    file instead of refusing it: the same import without the refusal, from the
    header or as read. The image's id."""
    with mock.patch.object(ops, "_refuse_inks", lambda name, inks: None):
        return ops.import_image(s, source, name, kind=kind, polarity=DARK, **options)


def _tinted_rgb() -> np.ndarray:
    """:func:`blot` in 8 bits as RGB, with a red mark away from the bands, so
    that its channels differ."""
    gray = (blot() >> 8).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[:4, :4] = (200, 30, 30)
    return rgb


def _encoded(image, fmt: str, **options) -> io.BytesIO:
    data = io.BytesIO()
    image.save(data, format=fmt, **options)
    data.seek(0)
    return data


def _separated(pixels: np.ndarray, **options) -> io.BytesIO:
    data = io.BytesIO()
    tifffile.imwrite(data, pixels, photometric="separated", **options)
    data.seek(0)
    return data


def _cmyk_files() -> dict[str, io.BytesIO]:
    """Every kind of single-image CMYK file the import refuses (#207) whose
    8-bit RGB or grayscale copy imports however it is saved, by the name it is
    imported under."""
    from PIL import Image

    cmyk = Image.fromarray(_tinted_rgb()).convert("CMYK")
    inks = np.asarray(cmyk)
    return {
        "cmyk µ.jpg": _encoded(cmyk, "JPEG", quality=95),
        "profile α.jpg": _encoded(cmyk, "JPEG", quality=95, icc_profile=_cmyk_profile()),
        "cmyk β.tif": _separated(inks),
        "profile.tif": _separated(inks, iccprofile=_cmyk_profile()),
        "lzw.TIF": _encoded(cmyk, "TIFF", compression="tiff_lzw"),
        # A reduced-resolution thumbnail page is no second image (reading takes it).
        "thumbnail.tif": _tiff_pages(
            (inks, {"photometric": "separated"}),
            (inks[::4, ::4], {"photometric": "separated", "subfiletype": 1}),
        ),
        # What the file holds, whatever its suffix (#207 review).
        "jpeg named β.png": _encoded(cmyk, "JPEG", quality=95),
        "jpeg named γ.tif": _encoded(cmyk, "JPEG", quality=95),
        "tiff named δ.png": _separated(inks),
        "tiff named.jpg": _separated(inks),
    }


def _alpha(inks: np.ndarray) -> np.ndarray:
    """``inks`` with an opaque alpha channel."""
    opaque = np.full(inks.shape[:2], np.iinfo(inks.dtype).max, dtype=inks.dtype)
    return np.dstack([inks, opaque])


def _wide_cmyk_files() -> dict[str, io.BytesIO]:
    """Single-image CMYK files whose RGB or grayscale copy is more than 8-bit
    or keeps an extra channel, which reading decodes for certain only
    uncompressed (#207 review, #232), by the name they are imported under."""
    from PIL import Image

    inks = np.asarray(Image.fromarray(_tinted_rgb()).convert("CMYK"))
    return {
        "black ink 16-bit.tif": _cmyk_tiff(blot()),
        "alpha µ.tif": _separated(_alpha(inks), extrasamples=[2]),
        "five inks.tiff": _separated(
            np.dstack([inks, inks[..., :1]]), extratags=[(334, 3, 1, 5, True)]
        ),
        "float inks.tif": _separated(inks.astype(np.float32) / 255),
    }


def _tiff_pages(*pages: tuple[np.ndarray, dict]) -> io.BytesIO:
    """A TIFF of these pages, each written on its own with its options: by
    tifffile, an image series each, unless it is a thumbnail."""
    data = io.BytesIO()
    with tifffile.TiffWriter(data) as tif:
        for pixels, options in pages:
            tif.write(pixels, **options)
    data.seek(0)
    return data


def _cmyk_stacks() -> dict[str, tuple[str, bool, io.BytesIO]]:
    """Files of inks and of more than one image, which reading refuses too, by
    the name they are imported under: their ink set (``cmyk`` or ``other``),
    whether the copy to import must be an uncompressed TIFF, and the file."""
    from PIL import Image

    cmyk = Image.fromarray(_tinted_rgb()).convert("CMYK")
    inks = np.asarray(cmyk)
    inks16 = inks.astype(np.uint16) * 257
    gray = (blot() >> 8).astype(np.uint8)
    separated = {"photometric": "separated"}
    spots = separated | {"extratags": [(332, 3, 1, 2, True)]}
    return {
        "stack µ.tif": ("cmyk", False, _separated(np.stack([inks, inks]))),
        "pillow stack α.tif": (
            "cmyk",
            False,
            _encoded(cmyk, "TIFF", save_all=True, append_images=[cmyk.copy()]),
        ),
        "lzw stack.tif": (
            "cmyk",
            False,
            _encoded(
                cmyk, "TIFF", save_all=True, append_images=[cmyk.copy()], compression="tiff_lzw"
            ),
        ),
        "planar stack.tif": (
            "cmyk",
            False,
            _separated(np.moveaxis(np.stack([inks, inks]), -1, 1), planarconfig="separate"),
        ),
        "two sizes.tif": (
            "cmyk",
            False,
            _tiff_pages((inks, separated), (inks[:20, :30], separated)),
        ),
        "spot stack β.tif": (
            "other",
            False,
            _separated(np.stack([inks, inks]), extratags=[(332, 3, 1, 2, True)]),
        ),
        # #207 review: inks after a gray or RGB first image (which reading
        # finds), a stack whatever its suffix, and copies of more than 8 bits.
        "gray then cmyk.tif": ("cmyk", False, _tiff_pages((gray, {}), (inks, separated))),
        "rgb then cmyk γ.tif": (
            "cmyk",
            False,
            _tiff_pages((_tinted_rgb(), {"photometric": "rgb"}), (inks, separated)),
        ),
        "gray then spots.tif": ("other", False, _tiff_pages((gray, {}), (inks, spots))),
        "stack named δ.png": ("cmyk", False, _separated(np.stack([inks, inks]))),
        "16-bit stack.tif": ("cmyk", True, _separated(np.stack([inks16, inks16]))),
        "gray then 16-bit cmyk.tif": (
            "cmyk",
            True,
            _tiff_pages((gray, {}), (inks16, separated)),
        ),
    }


def _refusal(name: str, inks: str, *, several: bool, uncompressed: bool) -> str:
    """#207's refusal of the file ``name`` of ``inks``, written out: why, that
    it holds several images if it does, and what to save."""
    reason = {
        "cmyk": (
            f"{name!r} is a CMYK file, which Proteia does not import: its gray levels could"
            " only be estimated from its inks"
        ),
        "other": (
            f"{name!r} holds printing inks other than CMYK (a separated TIFF), which"
            " Proteia does not import"
        ),
    }[inks]
    if several:
        reason += (
            "; it also holds more than one image (such as a stack, the channels of a"
            " composite, or an image with a mask or a thumbnail), and only a single image can"
            " be quantified"
        )
    what = "one of its images" if several else "it"
    copy = "an uncompressed RGB or grayscale TIFF" if uncompressed else "RGB or grayscale"
    return f"{reason}; save {what} as {copy} and import it again"


def _still_guarded() -> bool:
    """Whether the guards of conftest's autouse fixtures are in place, as a
    test that takes back patches of its own checks it took back only those
    (``monkeypatch.undo()`` takes these back too): the import's refusal
    check, the browser that fails the test, the test's own state folder."""
    import webbrowser

    state = [os.environ.get(name, "") for name in ("LOCALAPPDATA", "XDG_STATE_HOME")]
    return (
        hasattr(ops.import_image, "__wrapped__")
        and getattr(webbrowser, "_tryorder", None) == ["proteia-test-guard"]
        and all(Path(folder).name.startswith("user-state") for folder in state)
    )


def test_a_test_tells_the_guards_are_in_place(monkeypatch):
    # #233 (#248 review): monkeypatch.undo() takes back the guards of the
    # autouse fixtures too, so a test that took back its own patches that
    # way ran the rest unguarded; the check tells them taken back.
    assert _still_guarded()
    monkeypatch.undo()  # nothing runs after this but the check
    assert not _still_guarded()


def _imports_nothing(s: ProjectSession, refused) -> OperationError:
    """``refused`` raises an OperationError and changes nothing (the project,
    its log and next id, the stored files, the undo history): the error."""
    before, files, steps = s.project, _stored_files(s), s.history_steps
    with pytest.raises(OperationError) as info:
        refused()
    assert s.project is before
    assert _stored_files(s) == files  # the stored copy deleted
    assert s.history_steps == steps
    return info.value


def _stored_files(s: ProjectSession) -> list[str]:
    return sorted(path.name for path in (s.folder / storage.IMAGES_DIR).iterdir())


@pytest.mark.parametrize("name", list(_cmyk_files()))
def test_a_cmyk_file_is_refused_on_import_naming_it_and_what_to_do(tmp_path, name):
    # #207: a CMYK file is no longer converted on import; the user saves it as
    # RGB or grayscale and imports that.
    # #207 review: what the file holds decides, not its suffix, and the
    # answer's detail says which inks, for a client that words it its own way.
    s = session_on(tmp_path)
    source = _cmyk_files()[name]
    error = _imports_nothing(
        s, lambda: ops.import_image(s, source, name, kind=CHEMI, polarity=DARK)
    )
    assert (error.code, error.ids, error.detail) == (
        ErrorCode.CMYK_IMAGE,
        (),
        {"inks": "cmyk", "several_images": False, "uncompressed_copy": False},
    )
    assert str(error) == (
        f"{name!r} is a CMYK file, which Proteia does not import: its gray levels could"
        " only be estimated from its inks; save it as RGB or grayscale and import it again"
    )


@pytest.mark.parametrize("name", list(_wide_cmyk_files()))
def test_a_cmyk_file_of_more_than_8_bits_or_with_alpha_asks_for_an_uncompressed_tiff(
    tmp_path, name
):
    # #207 review: its RGB copy is 16-bit (or float) or keeps an extra channel,
    # which reading decodes for certain only uncompressed (a 16-bit RGB TIFF
    # saved with LZW is refused, and so is an RGBA one unless its alpha is
    # unassociated, #232), so the next step says so.
    s = session_on(tmp_path)
    source = _wide_cmyk_files()[name]
    error = _imports_nothing(
        s, lambda: ops.import_image(s, source, name, kind=CHEMI, polarity=DARK)
    )
    assert (error.code, error.ids, error.detail) == (
        ErrorCode.CMYK_IMAGE,
        (),
        {"inks": "cmyk", "several_images": False, "uncompressed_copy": True},
    )
    assert str(error) == (
        f"{name!r} is a CMYK file, which Proteia does not import: its gray levels could"
        " only be estimated from its inks; save it as an uncompressed RGB or grayscale TIFF"
        " and import it again"
    )


@pytest.mark.parametrize("name", list(_cmyk_stacks()))
def test_a_cmyk_file_of_several_images_is_refused_naming_both_problems(tmp_path, name):
    # #207 review: a stack (or images of several sizes) is refused by reading
    # too; a refusal naming only the inks would send a user who saves the
    # stack as RGB back for a second refusal, so it names both, also when the
    # inks are not in its first image (which reading finds).
    s = session_on(tmp_path)
    inks, uncompressed, source = _cmyk_stacks()[name]
    error = _imports_nothing(
        s, lambda: ops.import_image(s, source, name, kind=CHEMI, polarity=DARK)
    )
    assert (error.code, error.ids, error.detail) == (
        ErrorCode.CMYK_IMAGE,
        (),
        {"inks": inks, "several_images": True, "uncompressed_copy": uncompressed},
    )
    assert str(error) == _refusal(name, inks, several=True, uncompressed=uncompressed)


def test_the_refusals_of_files_of_several_images_in_full(tmp_path):
    # _refusal's words, checked once in full.
    s = session_on(tmp_path)
    answers = {}
    for name in ("gray then spots.tif", "16-bit stack.tif"):
        _, _, source = _cmyk_stacks()[name]
        answers[name] = str(
            _imports_nothing(
                s,
                functools.partial(ops.import_image, s, source, name, kind=CHEMI, polarity=DARK),
            )
        )
    assert answers == {
        "gray then spots.tif": (
            "'gray then spots.tif' holds printing inks other than CMYK (a separated TIFF),"
            " which Proteia does not import; it also holds more than one image (such as a"
            " stack, the channels of a composite, or an image with a mask or a thumbnail), and"
            " only a single image can be quantified; save one of its images as RGB or"
            " grayscale and import it again"
        ),
        "16-bit stack.tif": (
            "'16-bit stack.tif' is a CMYK file, which Proteia does not import: its gray levels"
            " could only be estimated from its inks; it also holds more than one image (such"
            " as a stack, the channels of a composite, or an image with a mask or a"
            " thumbnail), and only a single image can be quantified; save one of its images as"
            " an uncompressed RGB or grayscale TIFF and import it again"
        ),
    }


@pytest.mark.parametrize(
    "name",
    [
        "cmyk µ.jpg",
        "cmyk β.tif",
        "tiff named δ.png",
        "black ink 16-bit.tif",
        "stack µ.tif",
        "gray then cmyk.tif",
        "gray then 16-bit cmyk.tif",
        "spot stack β.tif",
        pytest.param(
            "alpha µ.tif",
            marks=pytest.mark.xfail(
                strict=True,
                reason="reading drops the alpha channel, so the check after it cannot"
                " ask for an uncompressed copy (a known limit)",
            ),
        ),
    ],
)
def test_inks_the_header_check_misses_are_refused_as_read_with_the_same_answer(tmp_path, name):
    # #207 review (fail closed): should the header check ever miss a file of
    # inks, the import refuses what reading converts or finds in it, with the
    # same answer, rather than import its colours converted.
    def files() -> dict[str, io.BytesIO]:
        stacks = {name: source for name, (_, _, source) in _cmyk_stacks().items()}
        return _cmyk_files() | _wide_cmyk_files() | stacks

    s = session_on(tmp_path)
    source = files()[name]
    expected = _imports_nothing(
        s, lambda: ops.import_image(s, source, name, kind=CHEMI, polarity=DARK)
    )
    source = files()[name]
    with mock.patch.object(ops, "ink_set", return_value=None):
        missed = _imports_nothing(
            s, lambda: ops.import_image(s, source, name, kind=CHEMI, polarity=DARK)
        )
    assert missed.code is ErrorCode.CMYK_IMAGE
    assert (str(missed), missed.detail) == (str(expected), expected.detail)


@pytest.mark.parametrize(
    ("name", "why"),
    [
        # Reading picks its reader by the suffix, as for the images imported
        # before: a JPEG named .tif is no TIFF to it.
        ("jpeg named γ.tif", "it is damaged or not an image Proteia can read (TiffFileError)"),
        # Inks reading does not convert, as before #207 (#234: named as chosen).
        ("five inks.tiff", "this separated TIFF has 5 inks"),
        ("float inks.tif", "this separated TIFF has float32 inks"),
    ],
)
def test_inks_the_header_check_misses_that_reading_refuses_are_not_imported(tmp_path, name, why):
    # #207 review (fail closed): refused, though not as inks. The header check
    # of the pixel type (#224) would refuse float inks first: it misses too.
    s = session_on(tmp_path)
    source = (_cmyk_files() | _wide_cmyk_files())[name]
    with (
        mock.patch.object(ops, "ink_set", return_value=None),
        mock.patch.object(ops, "declared_pixel_type", return_value=None),
    ):
        error = _imports_nothing(
            s, lambda: ops.import_image(s, source, name, kind=CHEMI, polarity=DARK)
        )
    assert error.code is ErrorCode.UNREADABLE_IMAGE
    assert str(error).startswith(f"cannot read {name!r}: {why}")


def _tiff_bytes(pixels: np.ndarray, **options) -> bytes:
    data = io.BytesIO()
    tifffile.imwrite(data, pixels, **options)
    return data.getvalue()


def _several_images() -> dict[str, bytes]:
    """TIFFs of more than one image and no inks, which reading refuses when
    named .tif (#215), by layout: a stack as tifffile, Pillow and ImageJ write
    one, pages written apart, images of two sizes, a multi-channel composite,
    a thumbnail written before its image (Pillow takes the thumbnail), 16-bit
    RGB (Pillow narrows it), and both byte orders of a classic TIFF and of a
    BigTIFF (Pillow reads no big-endian BigTIFF)."""
    from PIL import Image

    gray = (blot() >> 8).astype(np.uint8)
    other = np.ascontiguousarray(gray[::-1])
    stack = np.stack([gray, other])
    rgb16 = np.dstack([blot(), blot() // 2, blot() // 3])
    pages = [Image.fromarray(gray), Image.fromarray(other)]
    return {
        "stack": _tiff_bytes(stack),
        "pillow stack": _encoded(pages[0], "TIFF", save_all=True, append_images=pages[1:]).read(),
        "lzw stack": _encoded(
            pages[0], "TIFF", save_all=True, append_images=pages[1:], compression="tiff_lzw"
        ).read(),
        "imagej stack": _tiff_bytes(stack, imagej=True),
        "pages": _tiff_pages((gray, {}), (other, {})).read(),
        "two sizes": _tiff_pages((gray, {}), (gray[:20, :30], {})).read(),
        "composite": _tiff_bytes(
            np.stack([blot(), blot() // 2, blot() // 3]),
            photometric="minisblack",
            metadata={"axes": "CYX"},
        ),
        "thumbnail first": _tiff_pages((gray[::4, ::4], {"subfiletype": 1}), (gray, {})).read(),
        "16-bit rgb stack": _tiff_bytes(np.stack([rgb16, rgb16[::-1]]), photometric="rgb"),
        "big-endian stack": _tiff_bytes(stack, byteorder=">"),
        "bigtiff stack": _tiff_bytes(stack, bigtiff=True),
        "big-endian bigtiff stack": _tiff_bytes(stack, bigtiff=True, byteorder=">"),
    }


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".PNG"])
@pytest.mark.parametrize("layout", list(_several_images()))
def test_a_tiff_of_several_images_is_refused_whatever_its_suffix(tmp_path, layout, suffix):
    # #215: named otherwise, it was read by Pillow, which took its first image
    # (or its thumbnail) and dropped the rest without a notice. It is told by
    # its first bytes and refused as it is when named .tif.
    s = session_on(tmp_path)
    data = _several_images()[layout]
    as_tif = _imports_nothing(
        s,
        lambda: ops.import_image(s, io.BytesIO(data), f"{layout} µ.tif", kind=CHEMI, polarity=DARK),
    )
    name = f"{layout} µ{suffix}"
    error = _imports_nothing(
        s, lambda: ops.import_image(s, io.BytesIO(data), name, kind=CHEMI, polarity=DARK)
    )
    assert (as_tif.code, error.code) == (ErrorCode.UNREADABLE_IMAGE, ErrorCode.UNREADABLE_IMAGE)
    assert (error.ids, error.detail) == ((), as_tif.detail)
    # #220: the same words, naming the file as chosen, never the stored copy.
    assert str(error) == str(as_tif).replace(f"{layout} µ.tif", name)
    assert "img-1" not in str(error)


# A TIFF that is not a single image (#220), by the name it is offered under:
# a stack of two, a multi-channel composite, images of two sizes, a
# thumbnail before the image, and a file of no image, named .tif and
# otherwise (#215).
_NOT_A_SINGLE_IMAGE = {
    "stack of two µ.tif": lambda: _several_images()["stack"],
    "stack of two α.png": lambda: _several_images()["stack"],
    "composite β.tif": lambda: _several_images()["composite"],
    "composite µ.jpg": lambda: _several_images()["composite"],
    "two sizes α.tif": lambda: _several_images()["two sizes"],
    "thumbnail first β.png": lambda: _several_images()["thumbnail first"],
    "no image α.tif": lambda: b"II*\x00" + bytes(4),  # the first page's offset is 0: no page
    "no image β.PNG": lambda: b"II*\x00" + bytes(4),
}


def test_the_refusals_of_a_tiff_that_is_not_a_single_image_in_full(tmp_path):
    # #220: the refusal named the stored copy ("img-1.png"), said "holds 1
    # image series with axes 'QYX'" of a stack, and gave no next step. It
    # names the file as chosen, says in plain words that it holds more than
    # one image, or none, and how to import one: whatever the suffix.
    # #220 review: words true of every layout refused so, and a next step for
    # a file of no image that can be done (there is no image in it to save).
    s = session_on(tmp_path)
    answers = {}
    for name, data in _NOT_A_SINGLE_IMAGE.items():
        source = io.BytesIO(data())
        error = _imports_nothing(
            s,
            functools.partial(ops.import_image, s, source, name, kind=CHEMI, polarity=DARK),
        )
        assert (error.code, error.ids, error.detail) == (ErrorCode.UNREADABLE_IMAGE, (), None)
        answers[name] = str(error)
    several = (
        "{!r} holds more than one image (such as a stack, the channels of a composite, or an"
        " image with a mask or a thumbnail), which Proteia does not import: only a single"
        " image can be quantified; save one of its images as its own file and import it again"
    )
    no_image = (
        "{!r} holds no image, so there is nothing to quantify: export the image again from"
        " the program that made it, as a TIFF or PNG file, and import that file"
    )
    assert answers == {
        name: (no_image if name.startswith("no image") else several).format(name)
        for name in _NOT_A_SINGLE_IMAGE
    }
    assert answers["two sizes α.tif"] == (
        "'two sizes α.tif' holds more than one image (such as a stack, the channels of a"
        " composite, or an image with a mask or a thumbnail), which Proteia does not import:"
        " only a single image can be quantified; save one of its images as its own file and"
        " import it again"
    )
    assert answers["no image α.tif"] == (
        "'no image α.tif' holds no image, so there is nothing to quantify: export the image"
        " again from the program that made it, as a TIFF or PNG file, and import that file"
    )


@pytest.mark.parametrize("fmt", ["TIFF", "PNG"])
def test_the_image_of_a_tiff_of_no_image_exported_again_is_imported(tmp_path, fmt):
    # #220 review: the refusal's next step, done literally: the image the
    # file should have held, exported again as a TIFF or PNG file, imports
    # and measures as that image.
    from PIL import Image

    s = session_on(tmp_path)
    error = _imports_nothing(
        s,
        lambda: ops.import_image(
            s, io.BytesIO(b"II*\x00" + bytes(4)), "no image µ.tif", kind=CHEMI, polarity=DARK
        ),
    )
    assert str(error).endswith(
        ": export the image again from the program that made it, as a TIFF or PNG file, and"
        " import that file"
    )
    gray = (blot() >> 8).astype(np.uint8)
    exported = _encoded(Image.fromarray(gray), fmt)
    image_id = ops.import_image(
        s, exported, f"exported again α.{fmt.lower()}", kind=CHEMI, polarity=DARK
    )
    np.testing.assert_array_equal(s.pixels(image_id), gray.astype(np.float64))
    image = s.project.batch.find_image(image_id)
    assert (image.bit_depth, image.import_warnings) == (8, [])


@pytest.mark.parametrize("suffix", [".tif", ".png"])
@pytest.mark.parametrize(("layout", "index"), [("stack", 1), ("composite", 2)])
def test_one_image_of_a_tiff_of_several_saved_as_its_own_file_is_imported(
    tmp_path, layout, index, suffix
):
    # #220: the refusal's next step, done literally: one of its images (the
    # second page of the stack, the third channel of the composite) saved as
    # its own file imports, and measures as that image.
    s = session_on(tmp_path)
    data = _several_images()[layout]
    error = _imports_nothing(
        s,
        lambda: ops.import_image(
            s, io.BytesIO(data), f"{layout} µ{suffix}", kind=CHEMI, polarity=DARK
        ),
    )
    assert str(error).endswith("; save one of its images as its own file and import it again")
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        one = tif.series[0].asarray()[index]
    image_id = ops.import_image(
        s, io.BytesIO(_tiff_bytes(one)), f"{layout} image α.tif", kind=CHEMI, polarity=DARK
    )
    np.testing.assert_array_equal(s.pixels(image_id), one.astype(np.float64))
    assert s.project.batch.find_image(image_id).bit_depth == one.dtype.itemsize * 8


# --- A reading refusal names the file as chosen (#234) ---


def _gray8() -> np.ndarray:
    return (blot() >> 8).astype(np.uint8)


def _cielab(gray: np.ndarray) -> bytes:
    """A CIELAB TIFF of 8-bit ``gray`` levels as its lightness, with no colour."""
    lab = np.zeros((*gray.shape, 3), dtype=np.uint8)
    lab[..., 0] = gray
    return _tiff_bytes(lab, photometric="cielab")


def _animated(fmt: str, *frames: np.ndarray) -> bytes:
    """An animated PNG or GIF (``fmt``) of ``frames``."""
    from PIL import Image

    images = [Image.fromarray(frame) for frame in frames]
    return _encoded(images[0], fmt, save_all=True, append_images=images[1:]).read()


_SEVERAL_FRAMES = (
    "it holds more than one frame (such as an animated image), and only a single image can be"
    " quantified; save one frame as its own file and import it again"
)
_UNIDENTIFIED = (
    "it is damaged or not an image Proteia can read (UnidentifiedImageError); save the image"
    " again as a TIFF, PNG or JPEG file and import that"
)
_GRAY_OR_RGB = "which Proteia does not convert to gray; save the image as a gray or RGB TIFF"
# The refusals of #234 by the name the file is offered under: the file, and
# what the refusal says after "cannot read '<name>': ".
_REFUSALS_234 = {
    "cielab µ.tif": (
        lambda: _cielab(_gray8()),
        f"its pixels are stored in the CIELAB colour space, {_GRAY_OR_RGB}",
    ),
    "ycbcr α.tif": (
        lambda: _tiff_bytes(_tinted_rgb(), photometric="ycbcr"),
        f"its pixels are stored in the YCbCr colour space, {_GRAY_OR_RGB}",
    ),
    "camera raw β.tif": (
        lambda: _tiff_bytes(blot(), photometric=tifffile.PHOTOMETRIC.CFA),
        f"its pixels are stored with photometric interpretation CFA, {_GRAY_OR_RGB}",
    ),
    "two frames µ.png": (lambda: _animated("PNG", _gray8(), _gray8()[::-1]), _SEVERAL_FRAMES),
    "two rgb frames α.png": (
        lambda: _animated("PNG", _tinted_rgb(), _tinted_rgb()[::-1]),
        _SEVERAL_FRAMES,
    ),
    "gif named β.png": (lambda: _animated("GIF", _gray8(), _gray8()[::-1]), _SEVERAL_FRAMES),
    "not an image µ.png": (lambda: b"not an image at all", _UNIDENTIFIED),
    "not an image α.jpg": (lambda: b"not an image at all", _UNIDENTIFIED),
    "truncated header β.png": (lambda: _encoded_png()[:30], _UNIDENTIFIED),
    "damaged tiff named µ.png": (lambda: b"II*\x00\x08\x00\x00\x00" + b"\xff" * 40, _UNIDENTIFIED),
}


def _encoded_png() -> bytes:
    from PIL import Image

    return _encoded(Image.fromarray(_gray8()), "PNG").read()


@pytest.mark.filterwarnings("ignore:Corrupt EXIF data")  # Pillow, on the damaged TIFF
def test_the_refusals_of_234_name_only_the_chosen_file_in_full(tmp_path):
    # #234: they named the stored copy ("img-1.tif: its pixels are stored
    # ..."), or showed its full local path ("cannot identify image file
    # 'C:\\...\\images\\img-1.png'"), and two gave no next step ("img-1.png:
    # unsupported image layout (2, 40, 60)"). Each names the file as chosen,
    # says what is wrong and what to do; which files are refused is as before.
    s = session_on(tmp_path)
    answers = {}
    for name, (data, _) in _REFUSALS_234.items():
        source = io.BytesIO(data())
        error = _imports_nothing(
            s, functools.partial(ops.import_image, s, source, name, kind=CHEMI, polarity=DARK)
        )
        assert (error.code, error.ids, error.detail) == (ErrorCode.UNREADABLE_IMAGE, (), None)
        answers[name] = str(error)
    assert answers == {
        name: f"cannot read {name!r}: {words}" for name, (_, words) in _REFUSALS_234.items()
    }
    assert answers["two frames µ.png"] == (
        "cannot read 'two frames µ.png': it holds more than one frame (such as an animated"
        " image), and only a single image can be quantified; save one frame as its own file and"
        " import it again"
    )
    assert answers["not an image µ.png"] == (
        "cannot read 'not an image µ.png': it is damaged or not an image Proteia can read"
        " (UnidentifiedImageError); save the image again as a TIFF, PNG or JPEG file and import"
        " that"
    )
    assert answers["cielab µ.tif"] == (
        "cannot read 'cielab µ.tif': its pixels are stored in the CIELAB colour space, which"
        " Proteia does not convert to gray; save the image as a gray or RGB TIFF"
    )


@pytest.mark.parametrize("copy", ["gray", "RGB"])
def test_the_image_of_a_cielab_tiff_saved_as_a_gray_or_rgb_tiff_is_imported(tmp_path, copy):
    # #234: the refusal's next step, done: the image saved as a gray or an RGB
    # TIFF imports, with its lightness as the gray levels.
    from skimage.color import lab2rgb

    s = session_on(tmp_path)
    gray = _gray8()
    error = _imports_nothing(
        s,
        lambda: ops.import_image(
            s, io.BytesIO(_cielab(gray)), "cielab µ.tif", kind=CHEMI, polarity=DARK
        ),
    )
    assert str(error).endswith("; save the image as a gray or RGB TIFF")
    lab = np.zeros((*gray.shape, 3))
    lab[..., 0] = gray / 255 * 100  # CIELAB TIFF lightness: 0 to 255 for 0 to 100
    rgb = np.round(lab2rgb(lab) * 255).astype(np.uint8)
    pixels = rgb if copy == "RGB" else rgb[..., 0]
    saved = _tiff_bytes(pixels, photometric="rgb" if copy == "RGB" else "minisblack")
    image_id = ops.import_image(
        s, io.BytesIO(saved), f"saved {copy} α.tif", kind=CHEMI, polarity=DARK
    )
    image = s.project.batch.find_image(image_id)
    assert image.bit_depth == 8
    expected = rgb.astype(np.float64).mean(axis=-1) if copy == "RGB" else rgb[..., 0]
    np.testing.assert_allclose(s.pixels(image_id), expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("fmt", ["PNG", "GIF"])
def test_one_frame_of_an_animated_file_saved_as_its_own_file_is_imported(tmp_path, fmt):
    # #234: the refusal's next step, done literally: the second frame saved
    # as its own file imports, and measures as that frame.
    from PIL import Image

    s = session_on(tmp_path)
    gray = _gray8()
    data = _animated(fmt, gray, gray[::-1])
    error = _imports_nothing(
        s,
        lambda: ops.import_image(s, io.BytesIO(data), "frames µ.png", kind=CHEMI, polarity=DARK),
    )
    assert str(error).endswith("; save one frame as its own file and import it again")
    with Image.open(io.BytesIO(data)) as animated:
        animated.seek(1)
        frame = _encoded(animated.convert("L"), "PNG")
    image_id = ops.import_image(s, frame, "frame 2 α.png", kind=CHEMI, polarity=DARK)
    np.testing.assert_array_equal(s.pixels(image_id), gray[::-1].astype(np.float64))


@pytest.mark.parametrize("case", list(_MPO_SIZES))
def test_a_jpeg_of_several_pictures_is_imported_as_its_first_as_before(tmp_path, case):
    # #234 review: a phone's photo of a blot, which keeps a preview or a depth
    # map (MPO, as Pillow opens it), is imported as its first picture, as
    # before #234: never refused as frames of an animation.
    from PIL import Image

    s = session_on(tmp_path)
    source = tmp_path / "phone photo µ.jpg"
    _mpo(_MPO_SIZES[case])(source)
    with Image.open(source) as image:
        first = np.asarray(image.convert("RGB")).astype(np.float64)
    with source.open("rb") as f:
        image_id = ops.import_image(s, f, "phone photo µ.jpg", kind=CHEMI, polarity=DARK)
    image = s.project.batch.find_image(image_id)
    assert (image.width, image.height, image.bit_depth) == (200, 120, 8)
    assert [w.code for w in image.import_warnings] == ["lossy_format", "color_channels_differ"]
    np.testing.assert_allclose(s.pixels(image_id), first.mean(axis=-1), rtol=0, atol=1e-9)


@pytest.mark.parametrize("fmt", ["TIFF", "PNG", "JPEG"])
@pytest.mark.parametrize(
    ("name", "data", "step"),
    [
        (
            "unknown µ.png",
            b"not an image at all",
            "; save the image again as a TIFF, PNG or JPEG file and import that",
        ),
        (
            "damaged α.tif",
            b"II*\x00",
            "; save the image again as a TIFF, PNG or JPEG file and import that",
        ),
    ],
    ids=["unidentified", "damaged"],
)
def test_the_image_of_a_file_proteia_cannot_tell_saved_as_tiff_png_or_jpeg_is_imported(
    tmp_path, fmt, name, data, step
):
    # #234: the refusal's next step, done: the image saved (again) as a TIFF,
    # PNG or JPEG file imports.
    from PIL import Image

    s = session_on(tmp_path)
    error = _imports_nothing(
        s, lambda: ops.import_image(s, io.BytesIO(data), name, kind=CHEMI, polarity=DARK)
    )
    assert str(error).endswith(step)
    suffix = {"TIFF": "tif", "PNG": "png", "JPEG": "jpg"}[fmt]
    saved = _encoded(Image.fromarray(_gray8()), fmt)
    image_id = ops.import_image(s, saved, f"saved α.{suffix}", kind=CHEMI, polarity=DARK)
    assert s.project.batch.find_image(image_id).bit_depth == 8


def _refused_on_reading() -> dict[str, tuple[bytes, ErrorCode]]:
    """Files an import refuses on reading them, by the name they are offered
    under, with the code of the refusal (#234: as before): damaged, of a
    layout or colour space reading refuses, compressed as this installation
    cannot decode, or not a single image."""
    from PIL import Image

    gray = _gray8()
    palette = Image.fromarray(gray).convert("P")
    lzw = _encoded(palette, "TIFF", compression="tiff_lzw").read()  # Pillow reads indices
    unreadable = {
        "signature µ.tif": b"II*\x00",
        "signature α.png": b"II*\x00",
        "damaged page β.tif": b"II*\x00\x08\x00\x00\x00" + b"\xff" * 40,
        "not a tiff µ.tif": _encoded_png(),
        "truncated α.tif": _tiff_bytes(blot())[:3000],
        "untyped β.tif": _declared_as(128, 4),
        "five samples µ.tif": _tiff_bytes(
            np.dstack([gray] * 5), photometric="minisblack", planarconfig="contig"
        ),
        "lzw palette α.tif": lzw,
        "truncated idat β.png": _encoded_png()[:-200],
        "stack µ.tif": _several_images()["stack"],
        "stack α.png": _several_images()["stack"],
        "no image β.tif": b"II*\x00" + bytes(4),
        **{name: data() for name, (data, _) in _REFUSALS_234.items()},
    }
    return {
        **{name: (data, ErrorCode.UNREADABLE_IMAGE) for name, data in unreadable.items()},
        "float µ.tif": (_tiff_bytes(blot().astype(np.float32)), ErrorCode.UNSUPPORTED_PIXEL_TYPE),
        "cmyk α.jpg": (_cmyk_files()["cmyk µ.jpg"].read(), ErrorCode.CMYK_IMAGE),
    }


@pytest.mark.filterwarnings("ignore:Corrupt EXIF data")  # Pillow, on the damaged TIFF
@pytest.mark.parametrize("name", list(_refused_on_reading()))
def test_no_reading_refusal_names_the_stored_copy_or_a_local_path(tmp_path, name):
    # #234: whatever reading refuses, the refusal names the file as chosen;
    # the stored copy (img-1.tif) and the project's folder are Proteia's own.
    data, code = _refused_on_reading()[name]
    s = session_on(tmp_path)
    error = _imports_nothing(
        s, lambda: ops.import_image(s, io.BytesIO(data), name, kind=CHEMI, polarity=DARK)
    )
    assert error.code is code
    message = str(error)
    assert repr(name) in message
    assert not stored_copy_or_path_in(message, name, s.folder), message


@pytest.mark.parametrize(
    ("error", "reason", "words"),
    [
        (FileNotFoundError, "No such file or directory", "No such file or directory"),
        (PermissionError, "Permission denied", "Permission denied"),
        (OSError, None, "OSError"),  # no reason given: its kind
    ],
)
def test_a_stored_copy_the_system_cannot_read_is_refused_without_its_path(
    tmp_path, monkeypatch, error, reason, words
):
    # #234: the system's error names the stored copy's full path, which is
    # Proteia's own: the refusal gives its reason alone.
    def unreadable(path):
        raise error(5, reason, str(path))  # errno 5 (EIO): no subclass of its own

    monkeypatch.setattr(ops, "read_file", unreadable)
    s = session_on(tmp_path)
    refused = _imports_nothing(
        s,
        lambda: ops.import_image(
            s, io.BytesIO(_tiff_bytes(blot())), "gray µ.tif", kind=CHEMI, polarity=DARK
        ),
    )
    assert (refused.code, str(refused)) == (
        ErrorCode.UNREADABLE_IMAGE,
        f"cannot read 'gray µ.tif': {words}",
    )


def test_a_librarys_words_never_reach_a_refusal(tmp_path, caplog):
    # #234 review: a damaged file the library refuses in its own words gives
    # Proteia's instead, with the error's kind; the library's are logged.
    s = session_on(tmp_path)
    answers = {}
    for name, data in (
        ("png named µ.tif", _encoded_png()),
        ("truncated α.png", _encoded_png()[:-200]),
    ):
        source = io.BytesIO(data)
        with caplog.at_level(logging.INFO, logger="proteia"):
            error = _imports_nothing(
                s,
                functools.partial(ops.import_image, s, source, name, kind=CHEMI, polarity=DARK),
            )
        answers[name] = str(error)
    words = (
        "it is damaged or not an image Proteia can read ({}); save the image again as a TIFF,"
        " PNG or JPEG file and import that"
    )
    assert answers == {
        "png named µ.tif": "cannot read 'png named µ.tif': " + words.format("TiffFileError"),
        "truncated α.png": "cannot read 'truncated α.png': " + words.format("OSError"),
    }
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "not a TIFF file: header=b'\\x89PNG'" in logged
    assert "image file is truncated" in logged


# Errors of reading that name the stored copy in their words, as a library
# may (imageio: "`<path>` can not be opened"), by case: what is raised for the
# stored copy at a path, and what the refusal says after "cannot read
# '<name>': ".
_NAMING_THE_COPY = {
    "path in a ValueError": (lambda path: ValueError(f"`{path}` can not be opened"), "ValueError"),
    "posix path in an OSError": (
        lambda path: OSError(f"cannot open {path.as_posix()}"),
        "OSError",
    ),
    "escaped path": (lambda path: ValueError(f"cannot open {str(path)!r}"), "ValueError"),
    "folder": (lambda path: OSError(f"no room in {path.parent}"), "OSError"),
    "file name": (lambda path: ValueError(f"{path.name} is cut short"), "ValueError"),
    "damaged": (
        lambda path: imaging._unreadable(path.name, OSError(f"at {path}")),
        "OSError",
    ),
}


@pytest.mark.parametrize("case", list(_NAMING_THE_COPY))
def test_an_error_naming_the_stored_copy_names_the_chosen_file_on_import(
    tmp_path, monkeypatch, case
):
    # #234 review: only an OSError's filename was left out; any other words
    # that named the stored copy or its folder showed them. A refusal holds
    # Proteia's words only, with the error's kind.
    raised, kind = _NAMING_THE_COPY[case]

    def unreadable(path):
        raise raised(path)

    monkeypatch.setattr(ops, "read_file", unreadable)
    s = session_on(tmp_path)
    refused = _imports_nothing(
        s,
        lambda: ops.import_image(
            s, io.BytesIO(_tiff_bytes(blot())), "gel µ.tif", kind=CHEMI, polarity=DARK
        ),
    )
    assert (refused.code, str(refused)) == (
        ErrorCode.UNREADABLE_IMAGE,
        f"cannot read 'gel µ.tif': it is damaged or not an image Proteia can read ({kind});"
        " save the image again as a TIFF, PNG or JPEG file and import that",
    )


def _imported_then_reopened(tmp_path: Path) -> tuple[ProjectSession, str]:
    """A project with the blot imported as 'β-actin µ.tif', saved and opened
    again, so that nothing of its file is read yet: the session, the id."""
    s = session_on(tmp_path, save_to_folder)
    image = import_blot(s, blot(), "β-actin µ.tif")
    return ops.open_project(s.folder, autosave=save_to_folder, clock=FakeClock()), image


def _unidentified(path: Path) -> imaging.ReadingRefusal:
    """Reading's refusal of the file at ``path`` that Pillow cannot tell the
    format of, with Pillow's error, which names it by its full path."""
    from PIL import UnidentifiedImageError

    error = UnidentifiedImageError(f"cannot identify image file {str(path)!r}")
    return imaging._unreadable(path.name, error)


# How reading an image already in the project fails, by case: where (the
# session's read of its pixels, of its colours, or of its bytes for their
# SHA-256), what is raised for the stored copy at a path, and what the
# refusal says: never the stored copy, and never to import the file again,
# which would lose the image's boxes.
_READING_AGAIN = {
    "unidentified": (
        "load_image",
        _unidentified,
        "cannot read 'β-actin µ.tif': it is damaged or not an image Proteia can read"
        " (UnidentifiedImageError)",
    ),
    "several frames": (
        "load_image",
        lambda path: imaging.ReadingRefusal(
            path.name, "it holds more than one frame", "save one frame and import it again"
        ),
        "cannot read 'β-actin µ.tif': it holds more than one frame",
    ),
    "several images": (
        "load_image",
        lambda path: imaging.NotSingleImageError(path.name, None, no_image=False),
        "'β-actin µ.tif' holds more than one image (such as a stack, the channels of a"
        " composite, or an image with a mask or a thumbnail), which Proteia does not import:"
        " only a single image can be quantified",
    ),
    "library naming the copy": (
        "load_image",
        lambda path: ValueError(f"`{path}` can not be opened"),
        "cannot read 'β-actin µ.tif': it is damaged or not an image Proteia can read (ValueError)",
    ),
    "colours": (
        "read_colours",
        lambda path: imaging._unreadable(path.name, struct.error("bad bytes")),
        "cannot read 'β-actin µ.tif': it is damaged or not an image Proteia can read"
        " (struct.error)",
    ),
    "bytes": (
        "file_digest",
        lambda path: PermissionError(13, "Permission denied", str(path)),
        "cannot read 'β-actin µ.tif': Permission denied",
    ),
}


@pytest.mark.parametrize("case", list(_READING_AGAIN))
def test_reading_an_image_already_imported_names_it_as_imported(tmp_path, monkeypatch, case):
    # #234 review: the session gave the reading's own words, which name the
    # stored copy ("img-1.tif: ...") and may say to import the file again.
    where, raised, message = _READING_AGAIN[case]
    reopened, image = _imported_then_reopened(tmp_path)
    path = storage.image_path(reopened.folder, reopened.project.batch.find_image(image))

    def unreadable(*args, **kwargs):
        raise raised(path)

    owner = hashlib if where == "file_digest" else session_module
    monkeypatch.setattr(owner, where, unreadable)
    read = reopened.colour_pixels if where == "read_colours" else reopened.pixels
    before = reopened.project
    with pytest.raises(OperationError) as info:
        read(image)
    assert (info.value.code, info.value.ids, str(info.value)) == (
        ErrorCode.UNREADABLE_IMAGE,
        (image,),
        message,
    )
    assert "again" not in str(info.value)
    assert (reopened.project is before, reopened._pixels) == (True, {})


def _nan_blot() -> np.ndarray:
    """:func:`blot` as floating point, with one NaN pixel."""
    pixels = (blot() / 65535).astype(np.float32)
    pixels[3, 4] = np.nan
    return pixels


# What reading a TIFF meets that is no damage of the file, by case: what
# its reader (imaging._read_tiff) raises, or gives as its pixels, and what
# the refusal says is wrong, after "cannot read '<name>': ".
_NO_DAMAGE = {
    "memory": (
        lambda: np._core._exceptions._ArrayMemoryError((2**40,), np.dtype(np.uint8)),
        "Proteia ran out of memory reading it; close other programs and try again",
    ),
    "stack": (
        lambda: RecursionError("maximum recursion depth exceeded"),
        "Proteia ran out of stack reading it",
    ),
    "input/output": (lambda: OSError(errno.EIO, "Input/output error"), "Input/output error"),
    "access": (lambda: PermissionError(errno.EACCES, "Permission denied"), "Permission denied"),
    "nan": (_nan_blot, "it has NaN or infinite pixel values"),
}


def _reading_tiffs_as(made) -> object:
    """A TIFF reader (imaging._read_tiff) that raises what ``made`` makes, or
    gives it as the file's pixels."""

    def read_tiff(path):
        outcome = made()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, False, None

    return read_tiff


@pytest.mark.parametrize("case", list(_NO_DAMAGE))
def test_reading_an_image_already_imported_says_what_is_no_damage(tmp_path, monkeypatch, case):
    # #234 review 3: running out of memory or of stack, an error of the
    # system with no path (a network drive that drops) and Proteia's own
    # refusal of NaN pixels were each said to be damage ("damaged or not an
    # image Proteia can read (numpy._core._exceptions._ArrayMemoryError)").
    # Each says what happened, with no advice to import the file.
    made, words = _NO_DAMAGE[case]
    reopened, image = _imported_then_reopened(tmp_path)
    monkeypatch.setattr(imaging, "_read_tiff", _reading_tiffs_as(made))
    # NaN pixels are refused in the analysis array; colours show them.
    reads = [reopened.pixels] if case == "nan" else [reopened.pixels, reopened.colour_pixels]
    for read in reads:
        with pytest.raises(OperationError) as info:
            read(image)
        assert (info.value.code, info.value.ids, str(info.value)) == (
            ErrorCode.UNREADABLE_IMAGE,
            (image,),
            f"cannot read 'β-actin µ.tif': {words}",
        )
        assert "import" not in str(info.value)


@pytest.mark.parametrize("case", ["memory", "stack", "input/output", "access"])
def test_an_import_says_what_is_no_damage_and_what_to_do(tmp_path, monkeypatch, case):
    # #234 review 3: as reading an image already in the project, with what
    # to do for the file: of memory, a smaller copy of a very large file (a
    # copy cropped to the blot, done here, imports); of stack, the image
    # saved again; of the system, nothing to do to the file.
    made, words = _NO_DAMAGE[case]
    steps = {
        "memory": ", or, if the file is very large, save a smaller copy (such as one cropped to"
        " the blot) and import that",
        "stack": "; save the image again as a TIFF, PNG or JPEG file and import that",
    }
    real = imaging._read_tiff
    pixels = blot()
    cropped = pixels[:, :120]

    def read_tiff(path):
        read = real(path)
        if read[0].size > cropped.size:  # runs out on the whole blot only
            raise made()
        return read

    monkeypatch.setattr(imaging, "_read_tiff", read_tiff)
    s = session_on(tmp_path)
    refused = _imports_nothing(
        s,
        lambda: ops.import_image(
            s, io.BytesIO(_tiff_bytes(pixels)), "gel µ.tif", kind=CHEMI, polarity=DARK
        ),
    )
    assert (refused.code, refused.ids, str(refused)) == (
        ErrorCode.UNREADABLE_IMAGE,
        (),
        f"cannot read 'gel µ.tif': {words}{steps.get(case, '')}",
    )
    image_id = ops.import_image(
        s, io.BytesIO(_tiff_bytes(cropped)), "gel cropped µ.tif", kind=CHEMI, polarity=DARK
    )
    np.testing.assert_array_equal(s.pixels(image_id), cropped.astype(np.float64))


def test_a_refusal_of_reading_an_image_already_imported_is_logged_once(
    tmp_path, monkeypatch, caplog
):
    # #234 review 3: the library's words were logged on every read of a
    # stored file reading refuses (each view, each measure): they are logged
    # once per stored file (image id and SHA-256) in a session, each image's
    # own, and again in the next session.
    s = session_on(tmp_path, save_to_folder)
    images = [import_blot(s, blot(), name) for name in ("β-actin µ.tif", "GAPDH α.tif")]
    cause = struct.error("unpack requires a buffer of 4 bytes")

    def damaged(path):
        raise cause

    monkeypatch.setattr(imaging, "_read_tiff", damaged)

    def logged_by_a_session() -> list[str]:
        reopened = ops.open_project(s.folder, autosave=save_to_folder, clock=FakeClock())
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="proteia"):
            for _ in range(3):
                for image in images:
                    for read in (reopened.pixels, reopened.colour_pixels):
                        with pytest.raises(OperationError):
                            read(image)
        return [record.getMessage() for record in caplog.records]

    def failed(image: str, name: str) -> str:
        path = storage.image_path(s.folder, s.project.batch.find_image(image))
        return f"reading {path} (offered as {name!r}) failed: struct.error: {cause}"

    expected = [failed(images[0], "β-actin µ.tif"), failed(images[1], "GAPDH α.tif")]
    assert logged_by_a_session() == expected
    assert logged_by_a_session() == expected


def test_a_refusal_of_reading_another_file_under_an_id_is_logged_too(tmp_path, monkeypatch, caplog):
    # A project folder replaced outside Proteia (a synced or restored copy)
    # may hold another file under an image's id, which the session reads
    # once it reloads the project: its refusal is logged too, as it is
    # another stored file (its SHA-256).
    s = session_on(tmp_path / "a", save_to_folder)
    cause = struct.error("unpack requires a buffer of 4 bytes")

    def damaged(path):
        raise cause

    def refused_and_logged(image: str) -> list[str]:
        caplog.clear()
        with monkeypatch.context() as patch, caplog.at_level(logging.INFO, logger="proteia"):
            patch.setattr(imaging, "_read_tiff", damaged)
            for _ in range(2):
                with pytest.raises(OperationError):
                    s.colour_pixels(image)  # never cached: read each time
        return [
            record.getMessage() for record in caplog.records if "failed:" in record.getMessage()
        ]

    image = import_blot(s, blot(), "β-actin µ.tif")
    other = session_on(tmp_path / "b", save_to_folder)
    assert import_blot(other, blot()[::-1], "GAPDH α.tif") == image
    stored, copy = s.project.batch.find_image(image), other.project.batch.find_image(image)
    assert (stored.file, stored.sha256 != copy.sha256) == (copy.file, True)
    logged = refused_and_logged(image)
    assert len(logged) == 1 and "(offered as 'β-actin µ.tif')" in logged[0]
    for name in (storage.PROJECT_FILE, f"{storage.IMAGES_DIR}/{copy.file}"):
        shutil.copyfile(other.folder / name, s.folder / name)  # restored from the copy
    assert s.reload()
    logged = refused_and_logged(image)
    assert len(logged) == 1 and "(offered as 'GAPDH α.tif')" in logged[0]


def _animations() -> dict[str, bytes]:
    """Animations main imported as one image, by the name they are offered
    under: 3 frames of the blot's size, which scikit-image moves to the
    channels; 2 frames 3 pixels wide, which pass for channels; a WebP, read
    as its first frame."""
    from PIL import Image

    gray = _gray8()
    blots = [Image.fromarray(gray), Image.fromarray(gray[::-1]), Image.fromarray(255 - gray)]
    narrow = [Image.fromarray(np.full((40, 3), level, dtype=np.uint8)) for level in (60, 120)]
    rgb = [image.convert("RGB") for image in blots[:2]]

    def animated(images, fmt, **options):
        return _encoded(images[0], fmt, save_all=True, append_images=images[1:], **options).read()

    return {
        "three frames µ.png": animated(blots, "PNG"),
        "narrow frames α.png": animated(narrow, "PNG"),
        "animated webp β.png": animated(rgb, "WEBP", lossless=True),
    }


@pytest.mark.parametrize("name", list(_animations()))
def test_an_animation_imported_before_reads_and_measures_as_before(tmp_path, name):
    # #234 review: the frame check is the import's, not reading's: a project
    # holding an animation that main imported (it never looked for frames)
    # opens, reads and measures it as before, in what scikit-image gives.
    from skimage import io as skimage_io

    s = session_on(tmp_path, save_to_folder)
    data = _animations()[name]
    refused = _imports_nothing(
        s, lambda: ops.import_image(s, io.BytesIO(data), name, kind=CHEMI, polarity=DARK)
    )
    assert "it holds more than one frame" in str(refused)
    with mock.patch.object(ops, "check_frames", lambda path: None):  # as main imported it
        image_id = ops.import_image(s, io.BytesIO(data), name, kind=CHEMI, polarity=DARK)
    reopened = ops.open_project(s.folder, autosave=save_to_folder, clock=FakeClock())
    image = reopened.project.batch.find_image(image_id)
    decoded = skimage_io.imread(storage.image_path(reopened.folder, image))
    expected = quantify.to_grayscale(decoded).astype(np.float64)
    np.testing.assert_array_equal(reopened.pixels(image_id), expected)
    np.testing.assert_array_equal(reopened.colour_pixels(image_id), decoded)
    if image.width >= 160:  # room for the blot's boxes
        ops.set_lanes(reopened, [LaneInput("vehicle"), LaneInput("10 µM")])
        protein = ops.add_protein(reopened, "β-actin", Role.TARGET, image_id)
        ops.place_box(reopened, protein, NARROW_X, ROW, lane_index=0, grow=True)
        assert_nets_current(reopened)


def test_a_colour_space_refused_where_the_header_cannot_tell_names_the_image_as_imported(
    tmp_path, monkeypatch
):
    # The header (converts_cmyk) cannot be read, so the refusal of what
    # reading says (a colour space) cannot say what the image's numbers are:
    # it names the image as imported, with what is wrong only.
    reopened, image = _imported_then_reopened(tmp_path)

    def no_header(path):
        raise ValueError("cut short")

    def lab(path, *, channel=None):  # as the session reads it: on the measured channel
        raise imaging.UnsupportedColourSpaceError(
            Path(path).name,
            "its pixels are stored in the CIELAB colour space, which Proteia does not convert"
            " to gray",
        )

    monkeypatch.setattr(session_module, "converts_cmyk", no_header)
    monkeypatch.setattr(session_module, "load_image", lab)
    with pytest.raises(OperationError) as info:
        reopened.pixels(image)
    assert str(info.value) == (
        "cannot read 'β-actin µ.tif': its pixels are stored in the CIELAB colour space, which"
        " Proteia does not convert to gray"
    )


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
@pytest.mark.parametrize(
    "name",
    [
        "gray then cmyk.tif",
        "rgb then cmyk γ.tif",
        "gray then spots.tif",
        "gray then 16-bit cmyk.tif",
    ],
)
def test_inks_after_a_first_image_of_none_are_refused_whatever_the_suffix(tmp_path, name, suffix):
    # #215: the header check looks at the first page only (#207 review); the
    # later inks of a TIFF named otherwise are found when its images are
    # counted, as reading finds those of one named .tif, with the same answer.
    s = session_on(tmp_path)
    inks, uncompressed, source = _cmyk_stacks()[name]
    named = name.replace(".tif", suffix)
    error = _imports_nothing(
        s, lambda: ops.import_image(s, source, named, kind=CHEMI, polarity=DARK)
    )
    assert (error.code, error.ids, error.detail) == (
        ErrorCode.CMYK_IMAGE,
        (),
        {"inks": inks, "several_images": True, "uncompressed_copy": uncompressed},
    )
    assert str(error) == _refusal(named, inks, several=True, uncompressed=uncompressed)


def _single_tiffs() -> dict[str, bytes]:
    """Single-image TIFFs, by layout: those that tifffile, which reads a file
    named .tif, reads otherwise than Pillow (#214 review: white-is-zero gray,
    a palette, 16-bit RGB, associated alpha) or refuses (CIELAB); plain gray
    in either byte order and as a BigTIFF; and an image followed by its
    reduced-resolution thumbnail, which is no second image."""
    gray = (blot() >> 8).astype(np.uint8)
    rgb = _tinted_rgb()
    alpha = np.full(gray.shape, 200, dtype=np.uint8)
    level = np.arange(256, dtype=np.uint16)
    colormap = np.stack([level * 257, (255 - level) * 257, level * 128])
    return {
        "white is zero": _tiff_bytes(gray, photometric="miniswhite"),
        "palette": _tiff_bytes(gray, photometric="palette", colormap=colormap),
        "16-bit rgb": _tiff_bytes(np.dstack([blot(), blot() // 2, blot() // 3]), photometric="rgb"),
        "associated alpha": _tiff_bytes(
            np.dstack([rgb, alpha]), photometric="rgb", extrasamples=[1]
        ),
        "cielab": _tiff_bytes(rgb, photometric="cielab"),
        "gray": _tiff_bytes(gray),
        "16-bit big-endian": _tiff_bytes(blot(), byteorder=">"),
        "bigtiff": _tiff_bytes(gray, bigtiff=True),
        "thumbnail after": _tiff_pages((gray, {}), (gray[::4, ::4], {"subfiletype": 1})).read(),
    }


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
@pytest.mark.parametrize("layout", list(_single_tiffs()))
def test_a_single_image_tiff_named_otherwise_keeps_the_reader_its_suffix_picks(
    tmp_path, layout, suffix
):
    # #215 counts its images only: read by its content, as a .tif is, five of
    # these layouts would change their values or be refused, for files that
    # imported before and still read so in the projects that hold them.
    from skimage import io as skimage_io

    s = session_on(tmp_path)
    data = _single_tiffs()[layout]
    image_id = ops.import_image(
        s, io.BytesIO(data), f"{layout} µ{suffix}", kind=CHEMI, polarity=DARK
    )
    copy = tmp_path / f"copy{suffix}"
    copy.write_bytes(data)
    by_suffix = quantify.to_grayscale(skimage_io.imread(copy)).astype(np.float64)
    np.testing.assert_array_equal(s.pixels(image_id), by_suffix)
    as_tif = tmp_path / "copy.tif"
    as_tif.write_bytes(data)
    try:
        by_content = imaging.load_image(as_tif).array
    except ValueError:
        by_content = None  # CIELAB: refused by name
    differ = {"white is zero", "palette", "16-bit rgb", "associated alpha", "cielab"}
    assert (by_content is None or not np.array_equal(by_content, by_suffix)) == (layout in differ)


def test_a_tiff_of_several_images_imported_before_215_reads_as_before(tmp_path):
    # #215 changes the import only: a project that holds such a file, imported
    # as its first image, opens and reads it so.
    s = session_on(tmp_path, save_to_folder)
    data = _several_images()["pages"]
    with mock.patch.object(ops, "check_misnamed_tiff"):  # the import before #215
        image_id = ops.import_image(s, io.BytesIO(data), "pages µ.png", kind=CHEMI, polarity=DARK)
    s.close()
    reopened = ops.open_project(tmp_path / FOLDER, clock=FakeClock())
    np.testing.assert_array_equal(reopened.pixels(image_id), (blot() >> 8).astype(np.uint8))
    assert reopened.project.batch.find_image(image_id).import_warnings == []


@pytest.mark.parametrize(
    "data",
    [b"II*\x00", b"II*\x00\x08\x00\x00\x00" + b"\xff" * 40],
    ids=["signature-only", "damaged-page"],
)
@pytest.mark.filterwarnings("ignore:Corrupt EXIF data")  # Pillow, on the damaged page
def test_a_tiff_named_otherwise_whose_images_cannot_be_counted_is_left_to_reading(tmp_path, data):
    # #215: refused by the reader its suffix picks, as before, with its words;
    # #234: which name the file as chosen, not the stored copy's full path.
    s = session_on(tmp_path)
    error = _imports_nothing(
        s, lambda: ops.import_image(s, io.BytesIO(data), "damaged µ.png", kind=CHEMI, polarity=DARK)
    )
    assert error.code is ErrorCode.UNREADABLE_IMAGE
    assert str(error) == f"cannot read 'damaged µ.png': {_UNIDENTIFIED}"


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_one_image_of_a_cmyk_stack_saved_as_rgb_or_grayscale_is_imported(tmp_path, mode):
    # The stack refusal's next step, done: one of its images saved as RGB or
    # grayscale imports and measures; the whole stack saved so would not (it
    # is refused as a stack, which is why the refusal says one image).
    from PIL import Image

    s = session_on(tmp_path)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    _, _, stack = _cmyk_stacks()["pillow stack α.tif"]
    with Image.open(stack) as cmyk:
        pages = []
        for index in range(2):
            cmyk.seek(index)
            pages.append(cmyk.convert(mode))
    whole = _encoded(pages[0], "TIFF", save_all=True, append_images=pages[1:])
    error = _imports_nothing(
        s, lambda: ops.import_image(s, whole, "stack saved.tif", kind=CHEMI, polarity=DARK)
    )
    assert error.code is ErrorCode.UNREADABLE_IMAGE
    assert str(error) == (
        "'stack saved.tif' holds more than one image (such as a stack, the channels of a"
        " composite, or an image with a mask or a thumbnail), which Proteia does not import:"
        " only a single image can be quantified; save one of its images as its own file and"
        " import it again"
    )
    image_id = ops.import_image(
        s, _encoded(pages[1], "TIFF"), "stack page 2 µ.tif", kind=CHEMI, polarity=DARK
    )
    image = s.project.batch.find_image(image_id)
    assert image.bit_depth == 8
    assert [w.code for w in image.import_warnings] == (
        ["color_channels_differ"] if mode == "RGB" else []
    )
    protein = ops.add_protein(s, "β-actin", Role.TARGET, image_id)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert_nets_current(s)


def test_a_separated_tiff_of_other_inks_is_refused_on_import(tmp_path):
    from PIL import Image

    s = session_on(tmp_path)
    inks = np.asarray(Image.fromarray(_tinted_rgb()).convert("CMYK"))
    source = _separated(inks, extratags=[(332, 3, 1, 2, True)])  # InkSet: not CMYK
    error = _imports_nothing(
        s, lambda: ops.import_image(s, source, "spot inks µ.tif", kind=CHEMI, polarity=DARK)
    )
    assert (error.code, error.detail) == (
        ErrorCode.CMYK_IMAGE,
        {"inks": "other", "several_images": False, "uncompressed_copy": False},
    )
    assert str(error) == (
        "'spot inks µ.tif' holds printing inks other than CMYK (a separated TIFF), which"
        " Proteia does not import; save it as RGB or grayscale and import it again"
    )


def test_a_cmyk_marker_image_is_refused_too(tmp_path):
    s = session_on(tmp_path)
    blot_id = import_blot(s, blot())
    membrane = s.project.batch.membrane_of(blot_id).id
    source = _cmyk_files()["cmyk µ.jpg"]
    error = _imports_nothing(
        s,
        lambda: ops.import_image(
            s,
            source,
            "marker.jpg",
            kind=ImageKind.VISIBLE_MARKER,
            polarity=DARK,
            membrane_id=membrane,
        ),
    )
    assert error.code is ErrorCode.CMYK_IMAGE
    assert str(error).startswith("'marker.jpg' is a CMYK file, which Proteia does not import:")


@pytest.mark.parametrize(
    ("mode", "fmt", "options", "warnings"),
    [
        ("RGB", "PNG", {}, ["color_channels_differ"]),
        ("RGB", "TIFF", {}, ["color_channels_differ"]),
        ("RGB", "TIFF", {"compression": "tiff_lzw"}, ["color_channels_differ"]),
        ("L", "PNG", {}, []),
        ("L", "TIFF", {}, []),
        ("L", "TIFF", {"compression": "tiff_lzw"}, []),
    ],
)
def test_a_cmyk_file_saved_as_rgb_or_grayscale_is_imported(tmp_path, mode, fmt, options, warnings):
    # The refusal's next step, done: the CMYK JPEG and the separated TIFF, saved
    # as RGB or grayscale by an image editor (Pillow here), import and measure;
    # in 8 bits, also compressed.
    from PIL import Image

    s = session_on(tmp_path)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    suffix = ".png" if fmt == "PNG" else ".tif"
    for name in ("cmyk µ.jpg", "cmyk β.tif"):
        with Image.open(_cmyk_files()[name]) as cmyk:
            saved = _encoded(cmyk.convert(mode), fmt, **options)
        image_id = ops.import_image(s, saved, f"{name} saved{suffix}", kind=CHEMI, polarity=DARK)
        image = s.project.batch.find_image(image_id)
        assert ([w.code for w in image.import_warnings], image.bit_depth) == (warnings, 8)
        protein = ops.add_protein(s, f"{name} β", Role.TARGET, image_id)
        ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert_nets_current(s)


def _rgb16_lzw(rgb: np.ndarray) -> io.BytesIO:
    """``rgb`` (16-bit) as an RGB TIFF compressed with LZW, as an image editor
    may save it. Built here, as neither library writes one without
    imagecodecs: Pillow's LZW of the same bytes as a 16-bit gray image three
    times as wide, in a TIFF whose tags declare 16-bit RGB."""
    from PIL import Image

    height, width = rgb.shape[:2]
    rows = np.ascontiguousarray(rgb.astype("<u2").reshape(height, width * 3))
    gray = _encoded(Image.fromarray(rows), "TIFF", compression="tiff_lzw", strip_size=2**30)
    with tifffile.TiffFile(gray) as tif:
        (offset,), (count,) = tif.pages[0].dataoffsets, tif.pages[0].databytecounts
    strip = gray.getvalue()[offset : offset + count]
    tags = [  # tag, type (3 short, 4 long), count, value; a short's value fills 4 bytes
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 3, 134),  # BitsPerSample: 16, 16, 16 after the 126-byte IFD
        (259, 3, 1, 5),  # LZW
        (262, 3, 1, 2),  # RGB
        (273, 4, 1, 140),  # the one strip, after them
        (277, 3, 1, 3),
        (278, 4, 1, height),
        (279, 4, 1, len(strip)),
        (284, 3, 1, 1),
    ]
    ifd = struct.pack("<H", len(tags)) + b"".join(struct.pack("<HHII", *t) for t in tags)
    data = io.BytesIO(
        b"II*\x00" + struct.pack("<I", 8) + ifd + bytes(4) + struct.pack("<3H", 16, 16, 16) + strip
    )
    with tifffile.TiffFile(data) as tif:  # what it declares
        page = tif.pages[0]
        assert (page.shape, page.dtype, page.compression) == (
            (height, width, 3),
            np.uint16,
            tifffile.COMPRESSION.LZW,
        )
    with Image.open(data) as image:  # and LZW that decodes, to 8 bits in Pillow
        np.testing.assert_array_equal(np.asarray(image), (rgb >> 8).astype(np.uint8))
    data.seek(0)
    return data


@pytest.mark.parametrize("copy", ["RGB", "gray"])
def test_a_16_bit_cmyk_file_saved_as_an_uncompressed_tiff_is_imported(tmp_path, copy):
    # #207 review: the next step for 16-bit inks, done. The 16-bit RGB or
    # grayscale copy, saved as an uncompressed TIFF, imports and measures at 16
    # bits; the RGB copy saved with LZW is refused, which is why it says so.
    s = session_on(tmp_path)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    name = "black ink 16-bit.tif"
    error = _imports_nothing(
        s, lambda: ops.import_image(s, _cmyk_tiff(blot()), name, kind=CHEMI, polarity=DARK)
    )
    assert str(error).endswith(
        "; save it as an uncompressed RGB or grayscale TIFF and import it again"
    )
    cmyk = tmp_path / "cmyk.tif"
    cmyk.write_bytes(_cmyk_tiff(blot()).getvalue())
    rgb = imaging.read_pixels(cmyk)  # its colours, as an image editor converts them
    assert (rgb.shape, rgb.dtype) == ((H, W, 3), np.uint16)
    if copy == "RGB":
        lzw = _rgb16_lzw(rgb)
        refused = _imports_nothing(
            s, lambda: ops.import_image(s, lzw, "copy lzw.tif", kind=CHEMI, polarity=DARK)
        )
        assert refused.code is ErrorCode.UNREADABLE_IMAGE
        assert str(refused).endswith(
            "cannot be decoded by this installation; save the image as an uncompressed TIFF"
        )
        saved = io.BytesIO()
        tifffile.imwrite(saved, rgb, photometric="rgb")
    else:  # black ink alone: its channels are the same gray
        saved = io.BytesIO()
        tifffile.imwrite(saved, rgb[..., 0])
    saved.seek(0)
    image_id = ops.import_image(s, saved, f"copy {copy} µ.tif", kind=CHEMI, polarity=DARK)
    image = s.project.batch.find_image(image_id)
    assert (image.bit_depth, image.import_warnings) == (16, [])
    protein = ops.add_protein(s, "β-actin", Role.TARGET, image_id)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert_nets_current(s)


@pytest.mark.parametrize(("mode", "warnings"), [("RGBA", ["color_channels_differ"]), ("LA", [])])
def test_a_cmyk_file_with_alpha_saved_as_an_uncompressed_tiff_is_imported(tmp_path, mode, warnings):
    # #207 review: the next step for inks with an alpha channel, done. The
    # copy keeps the alpha channel: uncompressed, it imports and measures.
    # Saved with LZW, it imports too if its alpha is unassociated, as Pillow
    # saves it (#232), with the same nets; with associated alpha, which an
    # image editor may write instead, it is refused, which is why the step
    # still says uncompressed.
    from PIL import Image

    s = session_on(tmp_path)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    name = "alpha µ.tif"
    error = _imports_nothing(
        s,
        lambda: ops.import_image(s, _wide_cmyk_files()[name], name, kind=CHEMI, polarity=DARK),
    )
    assert str(error).endswith(
        "; save it as an uncompressed RGB or grayscale TIFF and import it again"
    )
    opaque = np.full((H, W), 255, dtype=np.uint8)
    copy = Image.fromarray(np.dstack([_tinted_rgb(), opaque])).convert(mode)
    associated = write_lzw_tiff(
        tmp_path / "associated.tif",
        np.asarray(copy),
        photometric="rgb" if mode == "RGBA" else "minisblack",
        extrasamples=[1],
    )
    with associated.open("rb") as source:
        refused = _imports_nothing(
            s,
            lambda: ops.import_image(s, source, "copy associated.tif", kind=CHEMI, polarity=DARK),
        )
    assert refused.code is ErrorCode.UNREADABLE_IMAGE
    assert str(refused).endswith("save the image as an uncompressed TIFF")
    nets = []
    for named, options in (
        (f"copy {mode} µ.tif", {}),
        (f"copy {mode} lzw α.tif", {"compression": "tiff_lzw"}),
    ):
        image_id = ops.import_image(
            s, _encoded(copy, "TIFF", **options), named, kind=CHEMI, polarity=DARK
        )
        image = s.project.batch.find_image(image_id)
        assert (image.bit_depth, [w.code for w in image.import_warnings]) == (8, warnings)
        protein = ops.add_protein(s, f"β-actin {named}", Role.TARGET, image_id)
        ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
        nets.append([band.net for band in boxes_of(s, protein).bands])
    assert nets[0] == nets[1]
    assert_nets_current(s)


def _over_exposed_8_bit() -> np.ndarray:
    """:func:`blot` in 8 bits, its narrow band's core at 0 (over-exposed)."""
    gray = (blot() >> 8).astype(np.uint8)
    gray[ROW - 1 : ROW + 2, NARROW_X - 1 : NARROW_X + 2] = 0
    return gray


def _translucent() -> np.ndarray:
    """An alpha of every level from 1 to 254, in some order."""
    return (np.arange(H * W).reshape(H, W) * 37 % 254 + 1).astype(np.uint8)


@pytest.mark.parametrize(
    ("layout", "warnings", "check"),
    [
        ("rgba", [], "exact"),
        ("rgba translucent", [], "exact"),
        ("rgba planar", [], "exact"),
        ("rgba tinted", ["color_channels_differ"], "possible"),
        ("gray alpha", [], "exact"),
    ],
)
def test_a_compressed_tiff_with_unassociated_alpha_imports_as_its_uncompressed_copy(
    tmp_path, layout, warnings, check
):
    # #232: the LZW file and its uncompressed copy import with the same
    # pixels, size, bit depth, background and notes, and measure the same
    # nets, backgrounds and over-exposure, whatever its alpha; its colours,
    # shown, are the same too. The alpha makes no colour note: only the
    # tinted copy's channels differ.
    s = session_on(tmp_path)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    gray = _over_exposed_8_bit()
    alpha = _translucent() if "translucent" in layout else np.full((H, W), 255, np.uint8)
    if layout.startswith("gray"):
        pixels, photometric = np.dstack([gray, alpha]), "minisblack"
    else:
        rgb = np.dstack([gray, gray, gray])
        if "tinted" in layout:
            rgb[:4, :4] = (200, 30, 30)  # a red mark away from the bands
        pixels, photometric = np.dstack([rgb, alpha]), "rgb"
    planar = "planar" in layout
    sources = tmp_path / "sources"
    sources.mkdir()
    plain = sources / f"{layout} µ.tif"
    tifffile.imwrite(
        plain,
        np.moveaxis(pixels, -1, 0) if planar else pixels,
        photometric=photometric,
        extrasamples=[2],
        planarconfig="separate" if planar else None,
    )
    lzw = write_lzw_tiff(
        sources / f"{layout} lzw α.tif",
        pixels,
        photometric=photometric,
        extrasamples=[2],
        planar=planar,
    )
    measured = []
    for path in (plain, lzw):
        with path.open("rb") as source:
            image_id = ops.import_image(s, source, path.name, kind=CHEMI, polarity=DARK)
        image = s.project.batch.find_image(image_id)
        protein = ops.add_protein(s, f"β-actin {path.stem}", Role.TARGET, image_id)
        ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
        ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
        bands = boxes_of(s, protein).bands
        measured.append(
            (
                (image.width, image.height, image.bit_depth, image.background),
                image.import_warnings,
                [
                    (b.box, b.net, b.background_level, b.background_mode, b.background_spread)
                    for b in bands
                ],
                [(b.clipped, b.possibly_clipped) for b in bands],
                s.pixels(image_id),
                s.colour_pixels(image_id),
            )
        )
    (size, notes, nets, flags, array, colours), lzw_measured = measured
    assert lzw_measured[:4] == (size, notes, nets, flags)
    np.testing.assert_array_equal(lzw_measured[4], array)
    np.testing.assert_array_equal(lzw_measured[5], colours)
    np.testing.assert_array_equal(colours, pixels)
    assert [w.code for w in notes] == warnings
    assert size[2] == 8
    # The narrow band holds the pixels at the limit: by the exact check where
    # the channels are the same gray, else by the possible one.
    assert flags == (
        [(True, None), (False, None)] if check == "exact" else [(None, True), (None, False)]
    )
    assert_nets_current(s)


def test_each_image_of_a_gray_then_cmyk_tiff_saved_alone_is_imported(tmp_path):
    # #207 review: the next step of a file whose inks follow a gray image,
    # done: either image, saved alone as RGB or grayscale, imports.
    from PIL import Image

    s = session_on(tmp_path)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    _, _, source = _cmyk_stacks()["gray then cmyk.tif"]
    error = _imports_nothing(
        s, lambda: ops.import_image(s, source, "two.tif", kind=CHEMI, polarity=DARK)
    )
    assert error.code is ErrorCode.CMYK_IMAGE
    source.seek(0)
    with tifffile.TiffFile(source) as tif:
        gray, inks = (page.asarray() for page in tif.pages)
    cmyk = Image.frombytes("CMYK", (W, H), inks.tobytes())
    for name, image in (("gray µ.tif", Image.fromarray(gray)), ("inks α.png", cmyk.convert("RGB"))):
        image_id = ops.import_image(
            s,
            _encoded(image, "PNG" if name.endswith(".png") else "TIFF"),
            name,
            kind=CHEMI,
            polarity=DARK,
        )
        protein = ops.add_protein(s, f"{name} β", Role.TARGET, image_id)
        ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert_nets_current(s)


def test_a_cmyk_image_imported_before_207_is_measured_on_its_converted_gray(tmp_path):
    # #131: the gray of a CMYK file is that of its inks converted to RGB, so
    # its nets are those of the same blot stored as gray. Imported before #207,
    # it keeps being read so.
    s = session_on(tmp_path)
    gray = blot()
    cmyk = _before_207(s, _cmyk_tiff(gray), "cmyk µ.tif")
    plain = import_blot(s, gray)
    image = s.project.batch.find_image(cmyk)
    assert [w.code for w in image.import_warnings] == ["cmyk_converted"]
    assert image.background == s.project.batch.find_image(plain).background
    np.testing.assert_array_equal(s.pixels(cmyk), s.pixels(plain))
    assert s.file_colours(image) == "rgb"  # converted: its channels are red, green and blue


# The nets origin/main (327f878) measured on the images of
# _project_before_207, by protein: lanes 1 and 2.
NETS_BEFORE_207 = {
    "cmyk 8 µ": [3135.8953488372103, 6089.729591836733],
    "cmyk 16 α": [798893.2142857142, 1555767.3913043481],
}
CONVERTED_BY_FORMULA = (
    "The file's colours are CMYK: they were converted to red, green and blue with the"
    " standard formula, as it embeds no ICC colour profile, then to gray. Values from a"
    " converted file are approximate, so over-exposure cannot be checked."
)


def _project_before_207(s: ProjectSession, imports) -> None:
    """The project of a CMYK JPEG and two separated TIFFs (8 and 16 bits)
    imported with ``imports`` (a function as :func:`_before_207`), a protein on
    each with a box in lanes 1 and 2; saved."""
    from PIL import Image

    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    cmyk = Image.fromarray(_tinted_rgb()).convert("CMYK")
    sources = {
        "cmyk 8 µ": ("cmyk 8 µ.tif", _separated(np.asarray(cmyk))),
        "cmyk 16 α": ("cmyk 16 α.tif", _cmyk_tiff(blot())),
        "cmyk jpeg β": ("cmyk β.jpg", _encoded(cmyk, "JPEG", quality=95)),
    }
    for protein_name, (name, source) in sources.items():
        image_id = imports(s, source, name)
        protein = ops.add_protein(s, protein_name, Role.TARGET, image_id)
        ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
        ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
    s.save()


def _nets(s: ProjectSession) -> dict[str, list[float | None]]:
    batch = s.project.batch
    return {p.name: [b.net for b in batch.boxes_of(p).bands] for p in batch.proteins}


def test_a_project_with_cmyk_images_imported_before_207_opens_with_its_nets(tmp_path):
    # #207 refuses CMYK on import only: images imported converted before keep
    # opening, reading and measuring as they did, with their warnings, and
    # opening and measuring rewrites nothing.
    s = session_on(tmp_path)
    _project_before_207(s, _before_207)
    folder = s.folder
    saved = (folder / "project.json").read_bytes()
    measured = _nets(s)
    reopened = ops.open_project(folder, autosave=save_to_folder, clock=FakeClock())
    assert not reopened.dirty
    nets = _nets(reopened)
    assert nets == measured
    for protein, expected in NETS_BEFORE_207.items():
        assert nets[protein] == pytest.approx(expected, rel=1e-12)
    images = {i.original_name: i for i in reopened.project.batch.iter_images()}
    assert {name: [w.code for w in i.import_warnings] for name, i in images.items()} == {
        "cmyk 8 µ.tif": ["cmyk_converted", "color_channels_differ"],
        "cmyk 16 α.tif": ["cmyk_converted"],
        "cmyk β.jpg": ["lossy_format", "cmyk_converted", "color_channels_differ"],
    }
    assert images["cmyk 16 α.tif"].import_warnings[0].message == CONVERTED_BY_FORMULA
    assert all(reopened.outdated_reading(i) is None for i in images.values())
    assert_nets_current(reopened)  # measured again from the stored files: the same nets
    notices = [
        n for n in ops.compute(reopened).notices if n.code is NoticeCode.CLIPPING_NOT_CHECKED
    ]
    assert len(notices) == 3
    assert all("CMYK converted to RGB" in n.message for n in notices)
    assert reopened.colour_pixels(images["cmyk 8 µ.tif"].id).shape == (H, W, 3)
    assert (folder / "project.json").read_bytes() == saved
    # A box placed now measures on the same reading.
    protein = next(p.id for p in reopened.project.batch.proteins if p.name == "cmyk 16 α")
    ops.remove_box(
        reopened,
        reopened.project.batch.boxes_of(reopened.project.batch.find_protein(protein)).bands[1].id,
    )
    ops.place_box(reopened, protein, WIDE_X, ROW, lane_index=1, grow=True)
    assert _nets(reopened)["cmyk 16 α"] == pytest.approx(NETS_BEFORE_207["cmyk 16 α"], rel=1e-12)


def test_a_cmyk_image_imported_before_conversion_is_refused_until_imported_again(tmp_path):
    # An image imported before #131 read its CMYK file as C, M and Y taken for
    # R, G and B: its record has no cmyk_converted warning, and its background
    # and nets came from that reading. Measured now on the converted gray, its
    # boxes would mix the two readings, so its pixels are refused.
    s = session_on(tmp_path)
    image_id = _before_207(s, _cmyk_tiff(blot()), "old cmyk.tif")
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image_id)

    def imported_before(draft: Project) -> None:
        draft.batch.find_image(image_id).import_warnings = []

    project, _ = apply_change(s.project, imported_before)
    s._commit(project, action="plant", params={}, evict=[image_id])
    before = s.project
    place = functools.partial(ops.place_box, s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    for read in (lambda: s.pixels(image_id), place):
        with pytest.raises(OperationError) as info:
            read()
        assert (info.value.code, info.value.ids) == (ErrorCode.UNREADABLE_IMAGE, (image_id,))
        assert str(info.value) == (
            "'old cmyk.tif' is a CMYK file imported before Proteia converted CMYK colours:"
            " its stored background and nets were measured with its cyan, magenta and yellow"
            " read as red, green and blue; remove the image, save the file as RGB or"
            " grayscale and import it again"
        )
    assert s.project is before
    assert s._pixels == {}
    assert s.colour_pixels(image_id).shape == (H, W, 3)  # display only: its converted colours
    # What it says to do, done: the file itself is refused since #207, but saved
    # as grayscale it is imported and measured.
    ops.remove_image(s, image_id)
    error = _imports_nothing(
        s,
        lambda: ops.import_image(s, _cmyk_tiff(blot()), "old cmyk.tif", kind=CHEMI, polarity=DARK),
    )
    assert error.code is ErrorCode.CMYK_IMAGE
    again = import_blot(s, blot(), "old cmyk.tif")
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, again)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    assert s.outdated_reading(s.project.batch.find_image(again)) is None
    assert_nets_current(s)


OLD_CMYK = (
    "'old cmyk.tif' is a CMYK file imported before Proteia converted CMYK colours:"
    " its stored background and nets were measured with its cyan, magenta and yellow"
    " read as red, green and blue; remove the image, save the file as RGB or grayscale"
    " and import it again"
)
OLD_LAB = (
    "'lab blot µ.tif': its pixels are stored in the CIELAB colour space, which Proteia"
    " does not convert to gray; it was imported before Proteia refused such files, so"
    " its stored background and nets come from a reading Proteia no longer makes:"
    " remove the image, save the file as a gray or RGB TIFF and import it again"
)


def _imported_before_cmyk(s: ProjectSession) -> tuple[str, str]:
    """An image imported before #131 converted CMYK, with a box in lane 1: its
    record has no cmyk_converted warning. Its image and protein ids."""
    image_id = _before_207(s, _cmyk_tiff(blot()), "old cmyk.tif")
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image_id)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)

    def imported_before(draft: Project) -> None:
        draft.batch.find_image(image_id).import_warnings = []

    project, _ = apply_change(s.project, imported_before)
    s._commit(project, action="plant", params={}, evict=[image_id])
    return image_id, protein


def _imported_before_lab(s: ProjectSession) -> tuple[str, str]:
    """An image imported before #131 refused CIELAB, with a box in lane 1: its
    stored file is a CIELAB TIFF, whose L*, a* and b* were read as red, green and
    blue. Its image and protein ids."""
    lab = np.dstack([(blot() >> 8).astype(np.uint8)] * 3)
    image_id = import_blot(s, lab, "lab blot µ.tif")
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image_id)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    path = storage.image_path(s.folder, s.project.batch.find_image(image_id))
    tifffile.imwrite(path, lab, photometric="cielab")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def imported_before(draft: Project) -> None:
        draft.batch.find_image(image_id).sha256 = digest

    project, _ = apply_change(s.project, imported_before)
    s._commit(project, action="plant", params={}, evict=[image_id])
    return image_id, protein


def test_an_image_in_a_colour_space_now_refused_names_its_file_and_what_to_do(tmp_path):
    # Refused on import since #131, a CIELAB file imported before is refused
    # when read: by the name it was imported under, not its stored file's, and
    # with what its stored numbers are and how to measure it again.
    s = session_on(tmp_path)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    image_id, protein = _imported_before_lab(s)
    before = s.project
    place = functools.partial(ops.place_box, s, protein, WIDE_X, ROW, lane_index=1, grow=True)
    for read in (lambda: s.pixels(image_id), lambda: s.colour_pixels(image_id), place):
        with pytest.raises(OperationError) as info:
            read()
        assert (info.value.code, info.value.ids) == (ErrorCode.UNREADABLE_IMAGE, (image_id,))
        assert str(info.value) == OLD_LAB
    assert s.project is before


@pytest.mark.parametrize(
    ("imported_before", "why"),
    [(_imported_before_cmyk, OLD_CMYK), (_imported_before_lab, OLD_LAB)],
    ids=["cmyk", "cielab"],
)
def test_nets_measured_on_a_reading_no_longer_made_are_named_and_not_exported(
    tmp_path, imported_before, why
):
    # An image read differently since its import keeps its stored nets until
    # it is removed: the results warn about them, and no export carries them.
    s = session_on(tmp_path)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    image_id, protein = imported_before(s)
    plain = import_blot(s, blot())  # an image read as it was imported: no notice
    control = ops.add_protein(s, "α-tubulin", Role.LOADING_CONTROL, plain)
    ops.place_box(s, control, NARROW_X, ROW, lane_index=0, grow=True)
    outdated = [n for n in ops.compute(s).notices if n.code is NoticeCode.OUTDATED_READING]
    assert outdated == [
        results.Notice(
            code=NoticeCode.OUTDATED_READING,
            level=Level.WARNING,
            message=why,
            protein_ids=(protein,),
        )
    ]
    exports = (
        functools.partial(ops.export_bundle, s, formats=["svg"]),
        functools.partial(ops.export_lane_table, s),
    )
    for export in exports:
        with pytest.raises(OperationError) as info:
            export()
        assert (info.value.code, info.value.ids) == (ErrorCode.UNREADABLE_IMAGE, (image_id,))
        assert str(info.value) == why  # the page says it was not exported
    assert list((s.folder / storage.EXPORTS_DIR).iterdir()) == []


def test_file_colours_read_the_header_once_per_image_without_the_lock(tmp_path, monkeypatch):
    s = session_on(tmp_path)
    gray = blot()
    image_id = import_blot(s, np.dstack([gray, gray // 2, gray // 3]), "stain.tif")
    image = s.project.batch.find_image(image_id)
    reads: list[Path] = []

    def counting(path: Path) -> str | None:
        reads.append(path)
        return imaging.file_colours(path)

    monkeypatch.setattr(session_module, "file_colours", counting)
    path = storage.image_path(s.folder, image)
    moved = path.with_name("away.tif")
    path.rename(moved)
    assert s.file_colours(image) is None  # unreadable: not kept, read again next time
    moved.rename(path)
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with s.lock:  # an operation running
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(10)
        assert s.file_colours(image) == "rgb"  # not waiting for the operation
        assert s.file_colours(image) == "rgb"
    finally:
        release.set()
        holder.join()
    assert reads == [path, path]  # kept once read


# --- pixel types with no fixed range (#224) ---


def _unknown_depth_sources() -> dict[str, tuple[str, np.ndarray]]:
    """Single-image TIFFs of :func:`blot` whose pixels have no fixed range, as
    an import took them before #224 (with an unknown_bit_depth warning), by
    the name of the protein measured on each: floating point (ImageJ's 32-bit
    type), 32-bit integers and signed 16-bit integers."""
    pixels = blot()
    return {
        "float µ": ("32-bit µ.tif", (pixels / 65535).astype(np.float32)),
        "uint32 α": ("uint32 α.tif", pixels.astype(np.uint32) * 3),
        "int16 β": ("int16 β.tif", (pixels // 2).astype(np.int16)),
    }


def _bilevel_png() -> io.BytesIO:
    from PIL import Image

    return _encoded(Image.fromarray(blot() < 40000), "PNG")


def _lzw_float() -> io.BytesIO:
    from PIL import Image

    return _encoded(
        Image.fromarray((blot() / 65535).astype(np.float32)), "TIFF", compression="tiff_lzw"
    )


def _float_with(value: float) -> bytes:
    """A float TIFF of :func:`blot` with one pixel set to ``value``."""
    pixels = (blot() / 65535).astype(np.float32)
    pixels[3, 4] = value
    return _tiff_bytes(pixels)


def _float_stack() -> bytes:
    """A TIFF of three float images (a stack), as ImageJ's 32-bit type."""
    data = io.BytesIO()
    with tifffile.TiffWriter(data) as tif:
        for _ in range(3):
            tif.write((blot() / 65535).astype(np.float32))
    return data.getvalue()


def _five_samples() -> bytes:
    """A float TIFF of one image with five samples a pixel, a layout reading
    refuses."""
    pixels = np.dstack([blot() / 65535] * 5).astype(np.float32)
    return _tiff_bytes(pixels, photometric="minisblack", planarconfig="contig")


# Files whose pixels have no known bit depth, which an import refuses (#224),
# by the name they are offered under: what the refusal says they hold (a
# TIFF's integers by the width its header says, which a reader may widen),
# and the copy it asks for.
_SIXTEEN, _UNSIGNED = "a 16-bit TIFF", "an unsigned 16-bit TIFF"
_UNKNOWN_DEPTH_FILES = {
    "32-bit µ.tif": (
        lambda: _tiff_bytes((blot() / 65535).astype(np.float32)),
        "floating-point",
        _SIXTEEN,
    ),
    "double α.tif": (lambda: _tiff_bytes(blot().astype(np.float64)), "floating-point", _SIXTEEN),
    "half β.tif": (lambda: _tiff_bytes(blot().astype(np.float16)), "floating-point", _SIXTEEN),
    "big-endian.TIF": (
        lambda: _tiff_bytes(blot().astype(">f4"), byteorder=">"),
        "floating-point",
        _SIXTEEN,
    ),
    "rgb float.tif": (
        lambda: _tiff_bytes(np.dstack([blot() / 65535] * 3).astype(np.float32), photometric="rgb"),
        "floating-point",
        _SIXTEEN,
    ),
    "lzw float.tiff": (lambda: _lzw_float().getvalue(), "floating-point", _SIXTEEN),
    # What the file holds decides, not its suffix: a TIFF named .png.
    "float named.png": (
        lambda: _tiff_bytes((blot() / 65535).astype(np.float32)),
        "floating-point",
        _SIXTEEN,
    ),
    # #224 review: refused for its type, not as a file reading refuses: one
    # NaN or infinite pixel (common in ratio images), a layout of five
    # samples, a type Pillow (which reads a TIFF named .png) cannot read.
    # Each is refused from its header, before it is decoded. (A stack is
    # refused as several images: _SEVERAL_IMAGES_FIRST.)
    "nan µ.tif": (lambda: _float_with(np.nan), "floating-point", _SIXTEEN),
    "infinite α.tif": (lambda: _float_with(np.inf), "floating-point", _SIXTEEN),
    "minus infinite β.tif": (lambda: _float_with(-np.inf), "floating-point", _SIXTEEN),
    "five samples µ.tif": (
        _five_samples,
        "floating-point",
        _SIXTEEN,
    ),
    "half named β.png": (
        lambda: _tiff_bytes(blot().astype(np.float16)),
        "floating-point",
        _SIXTEEN,
    ),
    # Complex pixels, refused without the warning casting them gives.
    "complex.tif": (lambda: _tiff_bytes(blot().astype(np.complex64)), "complex-valued", _SIXTEEN),
    "uint32.tif": (lambda: _tiff_bytes(blot().astype(np.uint32)), "32-bit integer", _SIXTEEN),
    # #224 review: the width the file stores, not the reader's: tifffile
    # widens 24 bits to 32 (and reads them only with imagecodecs), Pillow a
    # TIFF named .png of 16-bit integers to 32-bit ones, signed.
    "24-bit µ.tif": (
        lambda: _packed_tiff(blot().astype(np.uint32) * 200, 24),
        "24-bit integer",
        _SIXTEEN,
    ),
    "uint32 named α.png": (
        lambda: _tiff_bytes(blot().astype(np.uint32) * 3),
        "32-bit integer",
        _SIXTEEN,
    ),
    "int16 named β.png": (
        lambda: _tiff_bytes((blot() // 2).astype(np.int16)),
        "signed 16-bit integer",
        _UNSIGNED,
    ),
    # Pillow reads it as unsigned 8-bit (128 for -128): refused by its header.
    "int8 named µ.png": (
        lambda: _tiff_bytes((blot() >> 9).astype(np.int8) - 64),
        "signed 8-bit integer",
        _UNSIGNED,
    ),
    "uint64.tif": (lambda: _tiff_bytes(blot().astype(np.uint64)), "64-bit integer", _SIXTEEN),
    # Widths tifffile has no type for, which reading cannot decode: named by
    # their header's sample format.
    "signed 12-bit µ.tif": (lambda: _declared_as(12, 2), "signed 12-bit integer", _UNSIGNED),
    "128-bit α.tif": (lambda: _declared_as(128, 1), "128-bit integer", _SIXTEEN),
    "2-bit float β.tif": (lambda: _declared_as(2, 3), "floating-point", _SIXTEEN),
    "int32.tif": (
        lambda: _tiff_bytes(blot().astype(np.int32)),
        "signed 32-bit integer",
        _UNSIGNED,
    ),
    "int16.tif": (
        lambda: _tiff_bytes((blot() // 2).astype(np.int16)),
        "signed 16-bit integer",
        _UNSIGNED,
    ),
    "int8.tif": (
        lambda: _tiff_bytes((blot() >> 9).astype(np.int8)),
        "signed 8-bit integer",
        _UNSIGNED,
    ),
}


def _refused_for_type(tmp_path: Path, name: str, data: bytes) -> OperationError:
    """Import ``data`` offered as ``name`` into a new project: the refusal,
    which changed nothing (no stored file, no id taken, no autosave)."""
    recorder = Recorder()
    s = session_on(tmp_path, recorder)
    source = io.BytesIO(data)
    error = _imports_nothing(
        s, lambda: ops.import_image(s, source, name, kind=CHEMI, polarity=DARK)
    )
    assert (error.code, error.ids, error.detail) == (ErrorCode.UNSUPPORTED_PIXEL_TYPE, (), None)
    assert _stored_files(s) == []
    assert s.project.next_id == 1
    assert recorder.actions == []
    return error


# A refused file warns about nothing: casting complex pixels to gray did.
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("name", list(_UNKNOWN_DEPTH_FILES))
def test_a_file_of_unknown_bit_depth_is_refused_on_import_naming_it_and_what_to_do(tmp_path, name):
    # #224: its pixels have no fixed range, so over-exposure could not be
    # checked on it; until #224 an import took it with an unknown_bit_depth
    # warning. The refusal names the file as offered and its pixel type, and
    # changes nothing (no stored file, no autosave).
    data, what, copy = _UNKNOWN_DEPTH_FILES[name]
    error = _refused_for_type(tmp_path, name, data())
    assert str(error) == (
        f"{name!r} holds {what} pixels, which Proteia does not import: save it as {copy}"
        " and import it again"
    )


# 1-bit files: each pixel black or white (a mask, a thresholded image).
_BILEVEL_FILES = {
    "bilevel µ.tif": lambda: _tiff_bytes(blot() < 40000, bitspersample=1),
    "bilevel α.png": lambda: _bilevel_png().getvalue(),  # no TIFF: refused as read
    "bilevel named β.png": lambda: _tiff_bytes(blot() < 40000, bitspersample=1),
    "bilevel named µ.jpg": lambda: _tiff_bytes(blot() < 40000, bitspersample=1),
}


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("name", list(_BILEVEL_FILES))
def test_a_1_bit_file_is_refused_as_holding_no_gray_levels(tmp_path, name):
    # #224 review: a 16-bit copy of it would hold the same two levels, so
    # the refusal asks for the image it was made from instead.
    error = _refused_for_type(tmp_path, name, _BILEVEL_FILES[name]())
    assert str(error) == (
        f"{name!r} holds 1-bit pixels, which Proteia does not import: each pixel is black or"
        " white, so the file holds no gray levels to measure; import the original grayscale"
        " image instead"
    )


def test_the_original_of_a_1_bit_file_imports(tmp_path):
    # The 1-bit refusal's next step done: the grayscale image it was
    # thresholded from imports, with its bit depth.
    s = session_on(tmp_path)
    error = _imports_nothing(s, lambda: import_blot(s, blot() < 40000, "mask µ.tif"))
    assert str(error).startswith("'mask µ.tif' holds 1-bit pixels")
    image_id = import_blot(s, blot(), "mask µ.tif")
    image = s.project.batch.find_image(image_id)
    assert (image_id, image.bit_depth, image.import_warnings) == ("img-1", 16, [])


def _not_called(what: str):
    def fail(*args, **kwargs):
        raise AssertionError(f"{what} was called")

    return fail


_REFUSED_TIFFS = {
    **{name: data for name, (data, _, _) in _UNKNOWN_DEPTH_FILES.items()},
    **{name: data for name, data in _BILEVEL_FILES.items() if name != "bilevel α.png"},
}


@pytest.mark.parametrize("name", list(_REFUSED_TIFFS))
def test_a_tiff_refused_for_its_type_is_not_decoded(tmp_path, name, monkeypatch):
    # #224 review: a TIFF refused for its pixel type is refused from its
    # header: its pixels are not read, converted or scanned (memory, time).
    monkeypatch.setattr(imaging, "_read_tiff", _not_called("the TIFF reader"))
    monkeypatch.setattr(imaging, "_read_other", _not_called("the other reader"))
    _refused_for_type(tmp_path, name, _REFUSED_TIFFS[name]())


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    ("name", "data", "what", "copy"),
    [
        ("bilevel α.png", lambda: _bilevel_png().getvalue(), None, None),
        # A TIFF the header check let through, as a reader may give a type its
        # header does not say: refused as read, before a NaN pixel stops the
        # conversion, and named without a width.
        ("nan µ.tif", lambda: _float_with(np.nan), "floating-point pixels", _SIXTEEN),
        (
            "complex β.tif",
            lambda: _tiff_bytes(blot().astype(np.complex64)),
            "complex-valued pixels",
            _SIXTEEN,
        ),
        (
            "int16 µ.tif",
            lambda: _tiff_bytes((blot() // 2).astype(np.int16)),
            "integer pixels that are not unsigned 8- or 16-bit",
            _UNSIGNED,
        ),
        (
            "uint32 α.tif",
            lambda: _tiff_bytes(blot().astype(np.uint32)),
            "integer pixels that are not unsigned 8- or 16-bit",
            _UNSIGNED,
        ),
        ("bilevel β.tif", lambda: _tiff_bytes(blot() < 40000, bitspersample=1), None, None),
    ],
)
def test_pixels_of_a_type_the_header_did_not_say_are_refused_as_read_before_conversion(
    tmp_path, monkeypatch, name, data, what, copy
):
    # Failing closed (#224): a file whose header declares no such type (a PNG
    # is no TIFF) is refused for the pixels its reader gives, before they are
    # converted to gray or scanned.
    monkeypatch.setattr(ops, "_header_pixel_type", lambda path: None)
    monkeypatch.setattr(imaging, "to_analysis_array", _not_called("the conversion"))
    error = _refused_for_type(tmp_path, name, data())
    if what is None:
        assert str(error).startswith(f"{name!r} holds 1-bit pixels, which Proteia does not import")
    else:
        assert str(error) == (
            f"{name!r} holds {what}, which Proteia does not import: save it as {copy} and"
            " import it again"
        )


def test_a_header_that_cannot_be_read_leaves_the_reading_to_say_why(tmp_path):
    # The header check only refuses: a header it cannot read (damaged, or
    # untyped samples of a width tifffile has no type for) is refused by
    # reading it, as before #224.
    path = tmp_path / "damaged µ.tif"
    path.write_bytes(b"II*\x00")
    assert ops._header_pixel_type(path) is None
    assert ops._header_pixel_type(tmp_path / "missing.tif") is None
    s = session_on(tmp_path)
    for name, data in (
        ("damaged µ.tif", b"II*\x00"),
        ("untyped 128-bit α.tif", _declared_as(128, 4)),
        ("undefined format β.tif", _declared_as(16, 7)),
    ):
        source = io.BytesIO(data)
        error = _imports_nothing(
            s,
            lambda source=source, name=name: ops.import_image(
                s, source, name, kind=CHEMI, polarity=DARK
            ),
        )
        assert error.code is ErrorCode.UNREADABLE_IMAGE
        assert str(error).startswith(f"cannot read {name!r}: ")


def _gray_png() -> bytes:
    from PIL import Image

    return _encoded(Image.fromarray((blot() >> 8).astype(np.uint8)), "PNG").read()


def _run_out_in(caller: str | None, error: BaseException):
    """A replacement for a library call that raises ``error`` when called
    from the function named ``caller`` (from any, if None), and else calls
    the real one (given)."""

    def replacement(real):
        def call(*args, **kwargs):
            if caller is None or sys._getframe(1).f_code.co_name == caller:
                raise error
            return real(*args, **kwargs)

        return call

    return replacement


def _animation_runs_out(error: BaseException):
    """What Pillow says of whether a GIF is animated, which seeks its second
    frame, running out instead."""
    from PIL import GifImagePlugin

    def is_animated(self):
        raise error

    return GifImagePlugin.GifImageFile, "is_animated", property(is_animated)


# The reading steps of an import before its pixels are decoded, by case
# (#233): the file, by the name it is offered under; what is made to run
# out, as (owner, name, caller) of a library call or a Pillow image's
# property; and what the import says of the file once it can read it (None:
# it imports). Each check stopped the import had it taken the file for a
# damaged one: #230's ink check (of a TIFF by tifffile, of a PNG by Pillow),
# #224's pixel type (it would read a signed TIFF named .png as unsigned),
# #215's count of a TIFF named .png (Pillow takes its first image) and
# #234's frame check (scikit-image stacks the frames).
_HEADER_CHECKS = {
    "ink check of a tiff": (
        "gray µ.tif",
        lambda: _tiff_bytes(blot()),
        (imaging.tifffile, "TiffFile", "ink_set"),
        None,
    ),
    "ink check of a png": ("gray α.png", _gray_png, (imaging, "_pillow_mode", "ink_set"), None),
    "pixel type": (
        "int8 µ.png",
        lambda: _tiff_bytes((blot() >> 9).astype(np.int8) - 64),
        (imaging.tifffile, "TiffFile", "declared_pixel_type"),
        ErrorCode.UNSUPPORTED_PIXEL_TYPE,
    ),
    "count of a misnamed tiff": (
        "stack β.png",
        lambda: _tiff_bytes(np.stack([blot(), blot()[::-1]])),
        (imaging.tifffile, "TiffFile", "check_misnamed_tiff"),
        ErrorCode.UNREADABLE_IMAGE,
    ),
    "frame check": (
        "two frames µ.png",
        lambda: _animated("PNG", _gray8(), _gray8()[::-1]),
        (PIL.Image, "open", "check_frames"),
        ErrorCode.UNREADABLE_IMAGE,
    ),
    "frame check's seek": (
        "gif frames α.png",
        lambda: _animated("GIF", _gray8(), _gray8()[::-1]),
        None,  # Pillow's is_animated of a GIF
        ErrorCode.UNREADABLE_IMAGE,
    ),
}
# What an import that runs out of memory or of stack reading a file says
# after "cannot read '<name>': ", by the error (#242's words, the words
# reading gives).
_RAN_OUT = {
    MemoryError: (
        "Proteia ran out of memory reading it; close other programs and try again, or, if the"
        " file is very large, save a smaller copy (such as one cropped to the blot) and import"
        " that"
    ),
    RecursionError: (
        "Proteia ran out of stack reading it; save the image again as a TIFF, PNG or JPEG file"
        " and import that"
    ),
}
# What an import that runs out of memory measuring the pixels of a file it
# has read says after "cannot read '<name>': " (#233): the same next steps.
_RAN_OUT_MEASURING = (
    "Proteia ran out of memory measuring it; close other programs and try again, or, if the"
    " file is very large, save a smaller copy (such as one cropped to the blot) and import that"
)


def _running_out(monkeypatch, case: str, error: BaseException) -> None:
    """Make the header check of ``case`` (:data:`_HEADER_CHECKS`) run out with ``error``."""
    where = _HEADER_CHECKS[case][2]
    if where is None:
        monkeypatch.setattr(*_animation_runs_out(error))
        return
    owner, name, caller = where
    monkeypatch.setattr(owner, name, _run_out_in(caller, error)(getattr(owner, name)))


@pytest.mark.parametrize("kind", [MemoryError, RecursionError])
@pytest.mark.parametrize("case", list(_HEADER_CHECKS))
def test_running_out_on_a_header_check_is_refused_as_reading_and_imports_nothing(
    tmp_path, monkeypatch, caplog, case, kind
):
    # #224 review 2, #230: a header check that took running out for a
    # damaged header let the import carry on without it, so these errors were
    # raised as they are, and the import answered 500. #233: they are refused
    # as reading refuses them (UNREADABLE_IMAGE), in its words: the file by
    # the name it was offered under, what happened and what to do. Nothing is
    # stored (no image, no stored copy, project.json as it was); the error is
    # the refusal's cause, and is logged with the stored copy. Then, with
    # memory back, the check runs: the file imports, or is refused for what
    # it holds.
    name, data, _, then = _HEADER_CHECKS[case]
    error = kind("no room")
    s = session_on(tmp_path, save_to_folder)
    saved = (s.folder / storage.PROJECT_FILE).read_bytes()
    before = s.dirty
    with monkeypatch.context() as failing:  # only this patch is taken back below
        _running_out(failing, case, error)
        with caplog.at_level(logging.INFO, logger="proteia"):
            refused = _imports_nothing(
                s,
                lambda: ops.import_image(s, io.BytesIO(data()), name, kind=CHEMI, polarity=DARK),
            )
    assert (refused.code, refused.ids, refused.detail, str(refused)) == (
        ErrorCode.UNREADABLE_IMAGE,
        (),
        None,
        f"cannot read {name!r}: {_RAN_OUT[kind]}",
    )
    assert refused.__cause__ is error
    assert (s.dirty, s.project.next_id, (s.folder / storage.PROJECT_FILE).read_bytes()) == (
        before,
        1,
        saved,
    )
    stored = s.folder / storage.IMAGES_DIR / f"img-1{Path(name).suffix}"
    assert [record.getMessage() for record in caplog.records] == [
        f"reading {stored} (offered as {name!r}) failed: {kind.__name__}: no room"
    ]
    # Close other programs and try again, with the guards in place.
    source = io.BytesIO(data())
    if then is None:
        assert ops.import_image(s, source, name, kind=CHEMI, polarity=DARK) == "img-1"
    else:
        again = _imports_nothing(
            s, lambda: ops.import_image(s, source, name, kind=CHEMI, polarity=DARK)
        )
        assert again.code is then
        assert "ran out" not in str(again)
    assert _still_guarded()


def test_running_out_of_memory_on_a_header_check_of_a_large_file_imports_a_smaller_copy(
    tmp_path, monkeypatch
):
    # #233: the next step of the refusal, done as it says: the ink check runs
    # out on the whole blot only (a file too large for the memory left); a
    # copy cropped to the blot imports, with its pixels.
    pixels = blot()
    cropped = np.ascontiguousarray(pixels[:, :120])
    real = imaging.tifffile.TiffFile

    def run_out_on_the_whole_blot(path, *args, **kwargs):
        if sys._getframe(1).f_code.co_name == "ink_set" and Path(path).stat().st_size > (
            cropped.nbytes + 4096
        ):
            raise MemoryError("no room")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(imaging.tifffile, "TiffFile", run_out_on_the_whole_blot)
    s = session_on(tmp_path)
    refused = _imports_nothing(
        s,
        lambda: ops.import_image(
            s, io.BytesIO(_tiff_bytes(pixels)), "gel µ.tif", kind=CHEMI, polarity=DARK
        ),
    )
    assert str(refused) == f"cannot read 'gel µ.tif': {_RAN_OUT[MemoryError]}"
    image_id = ops.import_image(
        s, io.BytesIO(_tiff_bytes(cropped)), "gel cropped µ.tif", kind=CHEMI, polarity=DARK
    )
    np.testing.assert_array_equal(s.pixels(image_id), cropped.astype(np.float64))


def test_running_out_of_stack_on_a_header_check_imports_the_image_saved_again(
    tmp_path, monkeypatch
):
    # #233: the next step of the refusal, done as it says: the ink check of a
    # TIFF runs out of stack (tifffile's parse of its header); the image
    # saved again as a PNG file imports, with its pixels.
    from PIL import Image

    gray = _gray8()
    real = imaging.tifffile.TiffFile
    monkeypatch.setattr(
        imaging.tifffile, "TiffFile", _run_out_in("ink_set", RecursionError("no room"))(real)
    )
    s = session_on(tmp_path)
    refused = _imports_nothing(
        s,
        lambda: ops.import_image(
            s, io.BytesIO(_tiff_bytes(gray)), "gel µ.tif", kind=CHEMI, polarity=DARK
        ),
    )
    assert str(refused) == f"cannot read 'gel µ.tif': {_RAN_OUT[RecursionError]}"
    saved_again = _encoded(Image.fromarray(gray), "PNG")
    image_id = ops.import_image(s, saved_again, "gel α.png", kind=CHEMI, polarity=DARK)
    np.testing.assert_array_equal(s.pixels(image_id), gray.astype(np.float64))


def _raising(error: BaseException):
    def raising(*args, **kwargs):
        raise error

    return raising


@pytest.mark.parametrize(
    ("where", "kind"),
    [
        *[
            (where, RecursionError)
            for where in (
                "_refuse_inks",
                "_refuse_pixel_type",
                "load_pixels",
                "estimate_background",
                "assess_processed",
                "_prepare",
            )
        ],
        *[(where, MemoryError) for where in ("_refuse_pixel_type", "_prepare")],
    ],
)
def test_running_out_in_the_imports_own_code_still_surfaces(tmp_path, monkeypatch, where, kind):
    # #233 review of #241: only the reading of the file is refused when it
    # runs out; running out of stack in Proteia's own code around it (a
    # check's refusal, the conversion and measuring of the pixels, the
    # change) is a bug, and surfaces as it does: raised as it is, and
    # nothing is stored. So does running out of memory anywhere but in
    # reading the file and measuring its pixels (#233): a check's refusal,
    # the change.
    monkeypatch.setattr(ops, where, _raising(kind("no room")))
    s = session_on(tmp_path, save_to_folder)
    saved = (s.folder / storage.PROJECT_FILE).read_bytes()
    with pytest.raises(kind, match="^no room$"):
        ops.import_image(s, io.BytesIO(_tiff_bytes(blot())), "gel µ.tif", kind=CHEMI, polarity=DARK)
    assert (_stored_files(s), s.project.next_id) == ([], 1)
    assert (s.folder / storage.PROJECT_FILE).read_bytes() == saved


def _numpy_memory_error() -> MemoryError:
    """numpy's MemoryError of an array it cannot allocate, of a private class."""
    return np._core._exceptions._ArrayMemoryError((2**40,), np.dtype(np.float64))


@pytest.mark.parametrize(
    ("made", "kind"),
    [
        (lambda: MemoryError("no room"), "MemoryError"),
        (_numpy_memory_error, "numpy._core._exceptions._ArrayMemoryError"),
    ],
    ids=["memory", "numpy's memory"],
)
@pytest.mark.parametrize("where", ["load_pixels", "estimate_background", "assess_processed"])
def test_running_out_of_memory_measuring_an_imports_pixels_is_refused(
    tmp_path, monkeypatch, caplog, where, made, kind
):
    # #233: the steps a large blot most likely runs out of memory in, once
    # the file is read, answered 500: the conversion of the pixels to the
    # analysis array (8 bytes a pixel), their background (a median), the
    # processed check (a mask of the image, a copy of a palette's). Running
    # out of memory there is refused, nothing stored, saying it ran out
    # measuring the file, which was read; the next step works as it says: a
    # copy cropped to the blot imports.
    error = made()
    pixels = blot()
    cropped = np.ascontiguousarray(pixels[:, :120])
    real = getattr(ops, where)

    def run_out_on_the_whole_blot(*args, **kwargs):
        value = args[1] if where == "assess_processed" else args[0]  # (warnings, array, ...)
        array = value.pixels if where == "load_pixels" else value
        if array.shape[1] > cropped.shape[1]:
            raise error
        return real(*args, **kwargs)

    monkeypatch.setattr(ops, where, run_out_on_the_whole_blot)
    s = session_on(tmp_path, save_to_folder)
    saved = (s.folder / storage.PROJECT_FILE).read_bytes()
    with caplog.at_level(logging.INFO, logger="proteia"):
        refused = _imports_nothing(
            s,
            lambda: ops.import_image(
                s, io.BytesIO(_tiff_bytes(pixels)), "gel µ.tif", kind=CHEMI, polarity=DARK
            ),
        )
    assert (refused.code, refused.ids, refused.detail, str(refused)) == (
        ErrorCode.UNREADABLE_IMAGE,
        (),
        None,
        f"cannot read 'gel µ.tif': {_RAN_OUT_MEASURING}",
    )
    assert refused.__cause__ is error
    assert (s.dirty, s.project.next_id, (s.folder / storage.PROJECT_FILE).read_bytes()) == (
        False,
        1,
        saved,
    )
    stored = s.folder / storage.IMAGES_DIR / "img-1.tif"
    assert [record.getMessage() for record in caplog.records] == [
        f"reading {stored} (offered as 'gel µ.tif') failed: {kind}: {error}"
    ]
    image_id = ops.import_image(
        s, io.BytesIO(_tiff_bytes(cropped)), "gel cropped µ.tif", kind=CHEMI, polarity=DARK
    )
    np.testing.assert_array_equal(s.pixels(image_id), cropped.astype(np.float64))


@pytest.mark.parametrize("where", ["estimate_background", "_header_inks"])
def test_running_out_of_memory_lets_the_arrays_of_the_step_go_at_once(tmp_path, monkeypatch, where):
    # #233: the refusal keeps the error as its cause, whose traceback holds
    # the frames of the step that ran out, and so the large arrays they had
    # made, as long as the refusal is kept: closing other programs and
    # trying again would not free them. Those frames are cleared first, of
    # measuring the pixels as of reading the file; the cause stays.
    made = []

    def run_out(*args, **kwargs):
        large = np.ones(1_000_000)  # what the step had made when it ran out
        made.append(weakref.ref(large))
        raise MemoryError("no room")

    monkeypatch.setattr(ops, where, run_out)
    s = session_on(tmp_path)
    refused = _imports_nothing(
        s,
        lambda: ops.import_image(
            s, io.BytesIO(_tiff_bytes(blot())), "gel µ.tif", kind=CHEMI, polarity=DARK
        ),
    )
    assert (type(refused.__cause__), str(refused.__cause__)) == (MemoryError, "no room")
    assert [ref() for ref in made] == [None]


def test_running_out_in_the_autosave_hook_of_an_import_still_surfaces(tmp_path):
    # #233: as before, the hook's own error is raised after the commit, and
    # the image it imported stays, with its stored copy.
    def hook(session: ProjectSession) -> None:
        raise RecursionError("no room")

    s = session_on(tmp_path, hook)
    with pytest.raises(RecursionError, match="^no room$"):
        ops.import_image(s, io.BytesIO(_tiff_bytes(blot())), "gel µ.tif", kind=CHEMI, polarity=DARK)
    assert ([image.id for image in s.project.batch.iter_images()], _stored_files(s)) == (
        ["img-1"],
        ["img-1.tif"],
    )


@pytest.mark.parametrize("kind", [MemoryError, RecursionError])
def test_a_polarity_change_whose_palette_check_runs_out_still_drops_looks_processed(
    tmp_path, monkeypatch, kind
):
    # #233 (#235 review): the palette check of a polarity change, Pillow's
    # header read, takes running out for a file it cannot read, as before:
    # the polarity changes, and the looks_processed warning goes (no other
    # read of the image changes with #233). Read, a palette's levels would
    # keep it under either polarity.
    from PIL import Image

    s = session_on(tmp_path)
    even = np.repeat(np.arange(0, 255, 2, dtype=np.uint8), 20)[None, :]
    image = import_encoded(s, "palette γ.png", Image.fromarray(even).convert("P"), LIGHT)
    assert [w.code for w in s.project.batch.find_image(image).import_warnings] == [
        "looks_processed"
    ]
    real = imaging._pillow_mode
    monkeypatch.setattr(
        imaging, "_pillow_mode", _run_out_in("reads_as_palette", kind("no room"))(real)
    )
    ops.set_polarity(s, image, DARK)
    changed = s.project.batch.find_image(image)
    assert (changed.polarity, changed.import_warnings) == (DARK, [])


def _image_after(first: np.ndarray, **options) -> bytes:
    """A TIFF of :func:`blot` as a 16-bit image after a page ``first`` (a
    mask or a thumbnail, by ``options``): two image series."""
    data = io.BytesIO()
    with tifffile.TiffWriter(data) as tif:
        tif.write(first, **options)
        tif.write(blot())
    return data.getvalue()


# TIFFs of several images whose first is of a type with no known bit depth,
# by the name they are offered under (#215).
_SEVERAL_IMAGES_FIRST = {
    "mask first µ.tif": lambda: _image_after(blot() < 40000, subfiletype=4, bitspersample=1),
    "mask first α.png": lambda: _image_after(blot() < 40000, subfiletype=4, bitspersample=1),
    "float thumbnail first β.tif": lambda: _image_after(
        (blot()[::4, ::4] / 65535).astype(np.float32), subfiletype=1
    ),
    "float stack µ.tif": _float_stack,
    "float stack in one series α.tif": lambda: _tiff_bytes(
        np.stack([blot() / 65535] * 2).astype(np.float32)
    ),
}


@pytest.mark.parametrize("name", list(_SEVERAL_IMAGES_FIRST))
def test_a_tiff_of_several_images_is_refused_as_such_whatever_the_type_of_the_first(tmp_path, name):
    # #224 review 2: the header check judges the image reading takes, and
    # only in a file of one: a 1-bit mask or a float thumbnail before a
    # 16-bit image says nothing of the image. The file is refused as several
    # images (#215), whose next step (save one image) leads on.
    s = session_on(tmp_path)
    source = io.BytesIO(_SEVERAL_IMAGES_FIRST[name]())
    error = _imports_nothing(
        s, lambda: ops.import_image(s, source, name, kind=CHEMI, polarity=DARK)
    )
    assert error.code is ErrorCode.UNREADABLE_IMAGE
    assert str(error) == (
        f"{name!r} holds more than one image (such as a stack, the channels of a composite,"
        " or an image with a mask or a thumbnail), which Proteia does not import: only a"
        " single image can be quantified; save one of its images as its own file and import"
        " it again"
    )


# Files of a type with no known bit depth in a colour space reading refuses
# too, by name: what the refusal says they hold, the copy it asks for, and
# that copy's pixels.
_GRAY_OR_RGB = "a 16-bit grayscale or RGB TIFF"
_IN_A_COLOUR_SPACE = {
    "float CIELAB µ.tif": (
        lambda: _tiff_bytes(
            np.dstack([blot() / 65535] * 3).astype(np.float32), photometric="cielab"
        ),
        "floating-point pixels in the CIELAB colour space",
        _GRAY_OR_RGB,
    ),
    "int16 YCbCr α.tif": (
        lambda: _tiff_bytes(np.dstack([blot() // 2] * 3).astype(np.int16), photometric="ycbcr"),
        "signed 16-bit integer pixels in the YCbCr colour space",
        "an unsigned 16-bit grayscale or RGB TIFF",
    ),
    "float CFA named β.png": (
        lambda: _tiff_bytes((blot() / 65535).astype(np.float32), photometric=32803),
        "floating-point pixels with photometric interpretation CFA",
        _GRAY_OR_RGB,
    ),
}


@pytest.mark.parametrize("name", list(_IN_A_COLOUR_SPACE))
def test_a_file_also_in_a_colour_space_reading_refuses_is_refused_for_both(tmp_path, name):
    # #224 review 2: "save it as a 16-bit TIFF" alone would leave the colour
    # space, which reading then refuses; the refusal names both, and its
    # next step done (a 16-bit grayscale copy) imports.
    data, what, copy = _IN_A_COLOUR_SPACE[name]
    s = session_on(tmp_path)
    source = io.BytesIO(data())
    error = _imports_nothing(
        s, lambda: ops.import_image(s, source, name, kind=CHEMI, polarity=DARK)
    )
    assert (error.code, str(error)) == (
        ErrorCode.UNSUPPORTED_PIXEL_TYPE,
        f"{name!r} holds {what}, which Proteia does not import: save it as {copy} and import"
        " it again",
    )
    image_id = import_blot(s, blot(), Path(name).stem + ".tif")  # the gray copy
    image = s.project.batch.find_image(image_id)
    assert (image_id, image.bit_depth, image.import_warnings) == ("img-1", 16, [])


@pytest.mark.parametrize(("suffix", "warnings"), [(".png", []), (".jpg", ["lossy_format"])])
def test_a_big_endian_16_bit_tiff_named_otherwise_is_imported_with_its_bit_depth(
    tmp_path, suffix, warnings
):
    # Not refused for its pixel type (#224): Pillow, which reads it by its
    # suffix, keeps the file's byte order, and its pixels are still 16 bits
    # unsigned, whose over-exposure is checked. Its depth was unknown before,
    # so its bands' backgrounds were not measured as those of the same file
    # named .tif; now they are, and so are its nets.
    s = session_on(tmp_path)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    nets, images = [], []
    for name in (f"big-endian µ{suffix}", "big-endian µ.tif"):
        data = io.BytesIO(_tiff_bytes(blot(), byteorder=">"))
        image_id = ops.import_image(s, data, name, kind=CHEMI, polarity=DARK)
        np.testing.assert_array_equal(s.pixels(image_id), blot().astype(np.float64))
        protein = ops.add_protein(s, f"{name} β", Role.TARGET, image_id)
        ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
        ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
        nets.append(
            [
                band.net
                for band in s.project.batch.boxes_of(s.project.batch.find_protein(protein)).bands
            ]
        )
        images.append(s.project.batch.find_image(image_id))
    assert (images[0].bit_depth, [w.code for w in images[0].import_warnings]) == (16, warnings)
    assert nets[0] == nets[1]
    # origin/main (1bf6d4a) measured 798900.0 in lane 1 on it named .png.
    assert nets[0] == pytest.approx([798893.2142857142, 1555767.3913043481], rel=1e-12)


def test_a_pixel_type_no_reader_gives_is_named_as_numpy_names_it():
    # Readers give unsigned, signed, floating-point, complex or 1-bit pixels;
    # a refusal of any other kind names it rather than failing on it, and
    # nothing is refused for a known bit depth.
    with pytest.raises(OperationError) as info:
        ops._refuse_pixel_type("odd µ.tif", imaging.pixel_type(np.zeros(3, dtype="m8[s]")))
    assert (info.value.code, str(info.value)) == (
        ErrorCode.UNSUPPORTED_PIXEL_TYPE,
        "'odd µ.tif' holds timedelta64[s] pixels, which Proteia does not import: save it as"
        " a 16-bit TIFF and import it again",
    )
    assert ops._refuse_pixel_type("blot α.tif", None) is None


def test_a_floating_point_tiff_is_refused_in_the_words_of_224(tmp_path):
    s = session_on(tmp_path)
    error = _imports_nothing(
        s, lambda: import_blot(s, (blot() / 65535).astype(np.float32), "32-bit µ.tif")
    )
    assert str(error) == (
        "'32-bit µ.tif' holds floating-point pixels, which Proteia does not import: save it"
        " as a 16-bit TIFF and import it again"
    )
    assert error.code == "unsupported_pixel_type"  # the value clients see


@pytest.mark.parametrize(
    ("name", "pixels", "copy"),
    [
        # ImageJ's 16-bit conversion of its 32-bit type scales to the range.
        ("32-bit µ.tif", (blot() / 65535).astype(np.float32), lambda p: np.rint(p * 65535)),
        ("uint32 α.tif", blot().astype(np.uint32) * 3, lambda p: p // 3),
        # A signed image saved unsigned is moved up by 32768.
        ("int16 β.tif", (blot() // 2 - 16384).astype(np.int16), lambda p: p.astype(int) + 32768),
    ],
)
def test_a_file_refused_for_its_pixel_type_imports_once_saved_as_16_bit(
    tmp_path, name, pixels, copy
):
    # The refusal's next step done literally: the same pixels saved as a
    # 16-bit TIFF (unsigned) import, with a known bit depth and no warning.
    s = session_on(tmp_path)
    _imports_nothing(s, lambda: import_blot(s, pixels, name))
    sixteen = copy(pixels).astype(np.uint16)
    image_id = import_blot(s, sixteen, name)
    image = s.project.batch.find_image(image_id)
    assert (image.bit_depth, image.import_warnings) == (16, [])
    np.testing.assert_array_equal(s.pixels(image_id), sixteen.astype(np.float64))
    assert image_id == "img-1"  # the refusal took no id


# The nets origin/main (1bf6d4a) measured on the images of
# _unknown_depth_sources, imported as an import took them before #224, by
# protein: lanes 1 and 2; and their backgrounds.
NETS_BEFORE_224 = {
    "float µ": [12.190430879592896, 23.73929762840271],
    "uint32 α": [2396700.0, 4667277.391304346],
    "int16 β": [399466.0, 777855.4000000001],
}
BACKGROUNDS_BEFORE_224 = {
    "32-bit µ.tif": 0.7629510760307312,
    "uint32 α.tif": 150000.0,
    "int16 β.tif": 25000.0,
}
UNKNOWN_DEPTH = "The pixel type has no fixed detector range, so over-exposure cannot be checked."


def test_a_project_with_images_of_unknown_depth_imported_before_224_opens_with_its_nets(
    tmp_path,
):
    # #224 refuses such files on import only, not in the reader that opens
    # stored images: images imported before keep opening, showing and
    # measuring as they did, with their warning, and opening and measuring
    # rewrites nothing.
    s = session_on(tmp_path)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    for protein_name, (name, pixels) in _unknown_depth_sources().items():
        image_id = import_before_224(s, pixels, name)
        protein = ops.add_protein(s, protein_name, Role.TARGET, image_id)
        ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
        ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
    s.save()
    folder = s.folder
    saved = (folder / "project.json").read_bytes()
    measured = _nets(s)
    reopened = ops.open_project(folder, autosave=save_to_folder, clock=FakeClock())
    assert not reopened.dirty
    nets = _nets(reopened)
    assert nets == measured
    for protein, expected in NETS_BEFORE_224.items():
        assert nets[protein] == pytest.approx(expected, rel=1e-12)
    images = {i.original_name: i for i in reopened.project.batch.iter_images()}
    assert {
        name: (i.bit_depth, [(w.code, w.message) for w in i.import_warnings])
        for name, i in images.items()
    } == {name: (None, [("unknown_bit_depth", UNKNOWN_DEPTH)]) for name in BACKGROUNDS_BEFORE_224}
    for name, background in BACKGROUNDS_BEFORE_224.items():
        assert images[name].background == pytest.approx(background, rel=1e-12)
    assert all(reopened.outdated_reading(i) is None for i in images.values())
    assert_nets_current(reopened)  # measured again from the stored files: the same nets
    for image in images.values():  # shown as before: gray, the stored size
        assert reopened.colour_pixels(image.id).shape == (H, W)
    notices = [
        n for n in ops.compute(reopened).notices if n.code is NoticeCode.CLIPPING_NOT_CHECKED
    ]
    assert len(notices) == 3
    assert all("its image has an unknown bit depth" in n.message for n in notices)
    assert (folder / "project.json").read_bytes() == saved
    # A box placed now measures on the same reading.
    protein = next(p.id for p in reopened.project.batch.proteins if p.name == "float µ")
    ops.remove_box(
        reopened,
        reopened.project.batch.boxes_of(reopened.project.batch.find_protein(protein)).bands[1].id,
    )
    ops.place_box(reopened, protein, WIDE_X, ROW, lane_index=1, grow=True)
    assert _nets(reopened)["float µ"] == pytest.approx(NETS_BEFORE_224["float µ"], rel=1e-12)


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
    lanes = s.project.batch.only_lanes()
    assert [lane.label for lane in lanes] == ["vehicle", "vehicle", "10 µM", "10 µM"]
    assert [lane.sample for lane in lanes] == ["v1", "v2", None, "a2"]
    assert lanes[3].metadata == {"dose": "10 µM", "day": "1"}  # kept

    # A look-alike of a stored condition takes the stored spelling.
    update = ops.set_lanes(s, _four("vehicle", "vehicle", last=f"10 {MU}M"))
    assert update.respelled == (3,)
    assert s.project.batch.only_lanes()[3].label == f"10 {MICRO}M"
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
    assert len(s.project.batch.only_lanes()) == 5
    # Without the boxes in lane 3 (removed with their proteins' others: removing
    # one box re-quantifies the rest, and the sample's images are no pixels).
    ops.clear_boxes(s, "prot-7")
    ops.clear_boxes(s, "prot-8")
    ops.set_lanes(s, _four("vehicle", "vehicle")[:3])
    assert len(s.project.batch.only_lanes()) == 3


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
    assert boxes_of(s, loading).box_size == boxes.initial_box_size(W, H)
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

    row = s.project.batch.only_row(protein_of(s, target)).id
    removed = ops.remove_protein(s, target)
    assert removed.removed == (target, row)
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
    row = s.project.batch.only_row(protein_of(s, tubulin)).id
    cascade = ops.remove_protein(s, tubulin)
    assert cascade == Cascade(
        removed=(tubulin, row, *bands),
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
    placed = boxes_of(s, protein)
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
    old_size = boxes_of(s, protein).box_size
    old = band_of(s, first)
    x0, y0, x1, y1 = old.box.rect(old_size)

    second = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
    size = boxes_of(s, protein).box_size
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
    size = boxes_of(s, protein).box_size
    band_id = ops.place_box(s, protein, W - 2, ROW, lane_index=2, grow=False)
    band = band_of(s, band_id)
    assert band.box == Box(x=W - size.width, y=ROW - size.height // 2)
    assert band.source is ProposalSource.MANUAL
    assert boxes_of(s, protein).box_size == size
    assert_nets_current(s)


def test_move_box_reads_the_rect_by_its_centre(tmp_path):
    s, _, protein = boxed(tmp_path)
    a = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    b = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)

    def calibrate(draft: Project) -> None:  # MWs where the image has no curve
        for protein in draft.batch.proteins:
            for band in draft.batch.boxes_of(protein).bands:
                band.apparent_mw = 55.0

    plant(s, calibrate)
    untouched = band_of(s, b)
    size = boxes_of(s, protein).box_size

    ops.move_box(s, a, (30, 44, 10, 20))  # corners in any order; a 20x24 rect centred (20, 32)
    moved = band_of(s, a)
    assert moved.box.rect(size) == boxes.centered_rect(20, 32, size, W, H)
    assert moved.manually_edited
    assert moved.lane_index == 0  # never derived from x
    assert moved.apparent_mw is None  # recomputed at its new centre: no curve there
    assert band_of(s, b) == untouched  # bit-identical, apparent MW kept
    assert boxes_of(s, protein).box_size == size
    assert_nets_current(s)


def test_set_box_size_recentres_and_requantifies(tmp_path):
    s, _, protein = boxed(tmp_path)
    for x, lane in ((NARROW_X, 0), (WIDE_X, 1)):
        ops.place_box(s, protein, x, ROW, lane_index=lane, grow=True)
    old = boxes_of(s, protein)
    size = BoxSize(width=24, height=9)
    ops.set_box_size(s, protein, size)
    resized = boxes_of(s, protein)
    assert resized.box_size == size
    expected = boxes.resize_all(
        [band.box.rect(old.box_size) for band in old.bands], size, width=W, height=H
    )
    assert [band.box.rect(size) for band in resized.bands] == expected
    assert [band.net for band in resized.bands] != [band.net for band in old.bands]
    assert not any(band.manually_edited for band in resized.bands)  # a protein-level setting
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
    before = boxes_of(s, protein)
    ops.remove_box(s, a)
    after = boxes_of(s, protein)
    assert after.box_size == before.box_size
    assert after.bands == [band for band in before.bands if band.id == b]


def test_stored_nets_stay_current_through_a_session_and_a_reopen(tmp_path):
    s, image, protein = boxed(tmp_path, save_to_folder)
    loading = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    clicked = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    fixed = ops.place_box(s, protein, 70, ROW, lane_index=2, grow=False)
    ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)  # grows, re-centres
    ops.place_box(s, loading, WIDE_X, ROW + 6, lane_index=0, grow=False)  # below the band
    ops.move_box(s, fixed, (62, 22, 70, 36))
    ops.set_box_size(s, protein, BoxSize(width=21, height=7))
    ops.set_polarity(s, image, LIGHT)
    ops.set_polarity(s, image, DARK)
    ops.remove_box(s, clicked)
    assert_nets_current(s)

    reopened = ops.open_project(s.folder)
    assert reopened.project == s.project
    assert_nets_current(reopened)


# --- the local background (#83): every box on an image, quantified together ---


def textured_blot(seed: int = 83) -> np.ndarray:
    """The box tests' blot on a membrane with a gradient across it and noise, in
    16-bit counts: every ring pixel counts, so a box placed or removed beside a
    band moves its level."""
    _, x = np.mgrid[0:H, 0:W].astype(float)
    noise = np.random.default_rng(seed).normal(0.0, 300.0, (H, W))
    image = blot().astype(float) + 20.0 * (x - W / 2) + noise
    return np.clip(np.round(image), 0, 65535).astype(np.uint16)


def two_proteins(tmp_path: Path, hook=None) -> tuple[ProjectSession, str, str, str]:
    """A session with the textured blot, three lanes, and a target and a loading
    control on it (16x8 and 10x8 boxes), neither with a box yet."""
    s = session_on(tmp_path, hook)
    image = import_blot(s, textured_blot())
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM"), LaneInput("50 µM")])
    a = ops.add_protein(s, "β-catenin", Role.TARGET, image, box_size=BoxSize(width=16, height=8))
    b = ops.add_protein(
        s, "GAPDH", Role.LOADING_CONTROL, image, box_size=BoxSize(width=10, height=8)
    )
    return s, image, a, b


MEASURED = (
    "net",
    "background_level",
    "background_mode",
    "background_spread",
    "clipped",
    "possibly_clipped",
)


def measured(band: Band) -> list:
    """What quantifying the band's image gave it."""
    return [getattr(band, name) for name in MEASURED]


def _bands_on_image(s: ProjectSession, image_id: str) -> list[str]:
    batch = s.project.batch
    return [
        b.id
        for p in batch.proteins
        if batch.image_id_of(p) == image_id
        for b in batch.boxes_of(p).bands
    ]


def test_a_box_of_another_protein_moves_a_neighbours_net_in_the_same_change(tmp_path):
    s, _, a, b = two_proteins(tmp_path)
    band_a = ops.place_box(s, a, NARROW_X, ROW, lane_index=0, grow=False)
    alone = band_of(s, band_a)
    band_b = ops.place_box(s, b, NARROW_X + 16, ROW, lane_index=1, grow=False)  # in A's ring
    beside = band_of(s, band_a)
    assert (beside.net, beside.background_level) != (alone.net, alone.background_level)
    assert s.project.log[-1].content_hash == content_hash(s.project)
    assert_nets_current(s)

    length = len(s.project.log)
    ops.remove_box(s, band_b)
    after = band_of(s, band_a)
    # One change, B's removal, whose logged hash covers A's new net: A's ring
    # is whole again.
    assert len(s.project.log) == length + 1
    assert s.project.log[-1].action == "remove_box"
    assert s.project.log[-1].content_hash == content_hash(s.project)
    assert measured(after) == measured(alone) != measured(beside)
    assert_nets_current(s)


def test_a_removal_reads_the_pixels_only_when_bands_stay_on_the_image(tmp_path):
    s, image, a, b = two_proteins(tmp_path, save_to_folder)
    band_a = ops.place_box(s, a, NARROW_X, ROW, lane_index=0, grow=False)
    band_b = ops.place_box(s, b, WIDE_X, ROW, lane_index=1, grow=False)
    reopened = ops.open_project(s.folder)
    source = reopened.folder / "images" / f"{image}.tif"
    kept = source.read_bytes()
    source.write_bytes(b"not the imported file")
    before = reopened.project
    for remove in (
        lambda: ops.remove_box(reopened, band_a),
        lambda: ops.clear_boxes(reopened, a),
        lambda: ops.remove_protein(reopened, a),
    ):  # GAPDH's box stays, and its ring would change
        with pytest.raises(OperationError) as info:
            remove()
        assert (info.value.code, info.value.ids) == (ErrorCode.IMAGE_FILE_CHANGED, (image,))
        assert reopened.project is before
        assert reopened._pixels == {}

    source.write_bytes(kept)
    ops.remove_box(s, band_b)  # saved; now only β-catenin's box is on the image
    reopened = ops.open_project(s.folder)
    source.write_bytes(b"not the imported file")
    ops.remove_box(reopened, band_a)  # the last box: nothing is left to quantify
    assert _bands_on_image(reopened, image) == []
    assert reopened._pixels == {}


def test_a_ring_cut_short_stores_its_fallback_and_is_reported(tmp_path):
    """Both fallbacks through the operations: a box in the image's corner, whose
    ring has too few pixels paired across it (``asymmetric``, a robust plane),
    and a box that leaves too little ring (``image``, the image's median)."""
    s, image, a, _ = two_proteins(tmp_path)
    corner = ops.place_box(s, a, 8, 4, lane_index=0, grow=False)
    whole = ops.place_box(s, a, WIDE_X, ROW, lane_index=1, grow=False)
    other = import_blot(s, textured_blot(seed=5), "reprobe β.tif")
    c = ops.add_protein(
        s, "α-tubulin", Role.LOADING_CONTROL, other, box_size=BoxSize(width=150, height=56)
    )
    filling = ops.place_box(s, c, W // 2, H // 2, lane_index=0, grow=False)
    assert_nets_current(s)

    median = {i.id: i.background for i in s.project.batch.iter_images()}
    cut, kept, fallback = band_of(s, corner), band_of(s, whole), band_of(s, filling)
    assert (cut.background_mode, kept.background_mode) == ("asymmetric", "symmetric")
    assert cut.background_level != median[image]  # a plane through its ring, not the median
    assert (fallback.background_mode, fallback.background_level) == ("image", median[other])
    assert fallback.net > 0.0
    notices = ops.compute(s).notices
    assert [
        (n.protein_ids, n.lane_indices, n.level)
        for n in notices
        if n.code is NoticeCode.BACKGROUND_FALLBACK
    ] == [((a,), (0,), Level.WARNING), ((c,), (0,), Level.WARNING)]


def test_every_box_writer_requantifies_the_whole_image(tmp_path):
    s, image, a, b = two_proteins(tmp_path)
    for lane, x in ((0, NARROW_X), (1, WIDE_X)):
        ops.place_box(s, a, x, ROW, lane_index=lane, grow=False)
    guard = ops.place_box(s, b, 62, ROW, lane_index=2, grow=False)  # in both of A's rings
    steps = [
        ("place_box", lambda: ops.place_box(s, b, 80, ROW, lane_index=0, grow=False)),
        ("move_box", lambda: ops.move_box(s, guard, (54, 26, 64, 34))),
        ("set_box_size", lambda: ops.set_box_size(s, b, BoxSize(width=12, height=10))),
        ("set_box_padding", lambda: ops.set_box_padding(s, b, across=1, along=1)),
        ("remove_box", lambda: ops.remove_box(s, guard)),
        ("clear_boxes", lambda: ops.clear_boxes(s, b)),
    ]
    for action, step in steps:
        before = [measured(band) for band in boxes_of(s, a).bands]
        step()  # an edit of GAPDH's boxes only
        assert s.project.log[-1].action == action
        after = [measured(band) for band in boxes_of(s, a).bands]
        assert after != before, action  # β-catenin's bands are measured again
        assert_nets_current(s)
    ops.place_box(s, b, 62, ROW, lane_index=2, grow=False)
    before = [measured(band) for band in boxes_of(s, a).bands]
    ops.remove_protein(s, b)
    assert [measured(band) for band in boxes_of(s, a).bands] != before
    assert_nets_current(s)
    ops.set_polarity(s, image, LIGHT)
    assert_nets_current(s)


# The steps of the property test: each takes the session, a random generator
# and the row case, and either commits a change or is refused.
def _pick(rng: np.random.Generator, items: Sequence):
    if not items:
        raise UnknownIdError("nothing to pick")  # a refusal like any other
    return items[int(rng.integers(len(items)))]


def _band_ids(s: ProjectSession) -> list[str]:
    return [
        band.id
        for protein in s.project.batch.proteins
        for band in s.project.batch.boxes_of(protein).bands
    ]


def _protein_ids(s: ProjectSession) -> list[str]:
    return [protein.id for protein in s.project.batch.proteins]


def _step_place(s: ProjectSession, rng: np.random.Generator, case: RowCase) -> None:
    protein, lane = _pick(rng, _protein_ids(s)), int(rng.integers(case.n_lanes))
    x = round(case.lane_cx[lane]) + int(rng.integers(-15, 16))
    y = round(case.lane_cy[lane]) + int(rng.integers(-25, 26))
    ops.place_box(s, protein, x, y, lane_index=lane, grow=bool(rng.random() < 0.4))


def _step_move(s: ProjectSession, rng: np.random.Generator, case: RowCase) -> None:
    protein, band = s.project.batch.find_band(_pick(rng, _band_ids(s)))
    x0, y0, x1, y1 = band.box.rect(s.project.batch.boxes_of(protein).box_size)
    dx, dy = int(rng.integers(-12, 13)), int(rng.integers(-10, 11))
    ops.move_box(s, band.id, (x0 + dx, y0 + dy, x1 + dx, y1 + dy))


def _step_resize(s: ProjectSession, rng: np.random.Generator, case: RowCase) -> None:
    size = BoxSize(width=int(rng.integers(8, 60)), height=int(rng.integers(4, 24)))
    ops.set_box_size(s, _pick(rng, _protein_ids(s)), size)


def _step_add_protein(s: ProjectSession, rng: np.random.Generator, case: RowCase) -> None:
    [image] = s.project.batch.iter_images()
    size = BoxSize(width=int(rng.integers(10, 40)), height=int(rng.integers(6, 16)))
    name = f"protein {len(s.project.log)}"
    ops.add_protein(s, name, Role.LOADING_CONTROL, image.id, box_size=size)


def _step_polarity(s: ProjectSession, rng: np.random.Generator, case: RowCase) -> None:
    [image] = s.project.batch.iter_images()
    ops.set_polarity(s, image.id, LIGHT if image.polarity is DARK else DARK)


def _step_row(s: ProjectSession, rng: np.random.Generator, case: RowCase) -> None:
    ops.detect_row_boxes(s, _pick(rng, _protein_ids(s)), case.row)


def _step_lane(s: ProjectSession, rng: np.random.Generator, case: RowCase) -> None:
    ops.set_box_lane(s, _pick(rng, _band_ids(s)), int(rng.integers(case.n_lanes)))


PROPERTY_STEPS = {
    "place_box": _step_place,
    "move_box": _step_move,
    "remove_box": lambda s, rng, case: ops.remove_box(s, _pick(rng, _band_ids(s))),
    "set_box_size": _step_resize,
    "remove_protein": lambda s, rng, case: ops.remove_protein(s, _pick(rng, _protein_ids(s))),
    "add_protein": _step_add_protein,
    "set_polarity": _step_polarity,
    "detect_row_boxes": _step_row,
    "clear_boxes": lambda s, rng, case: ops.clear_boxes(s, _pick(rng, _protein_ids(s))),
    "set_box_lane": _step_lane,
    "undo": lambda s, rng, case: ops.undo(s),
    "redo": lambda s, rng, case: ops.redo(s),
}
# Every operation that adds, moves, resizes or removes a box, or turns the signal
# around: each re-quantifies the whole image (remove_image takes the image along).
BOX_WRITERS = frozenset(
    {
        "place_box",
        "move_box",
        "remove_box",
        "set_box_size",
        "remove_protein",
        "set_polarity",
        "detect_row_boxes",
        "clear_boxes",
    }
)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_random_edits_keep_every_stored_value_what_the_pixels_give(tmp_path, seed):
    """Property: after any sequence of edits, refused or not, undone or redone,
    every band on the image holds exactly what quantifying the image from
    scratch gives (with two proteins' boxes on it, one in the image's corner,
    whose ring falls back to a plane)."""
    case = ROWS["all_present"]
    s, image, target = row_session(tmp_path, case)
    other = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    ops.detect_row_boxes(s, target, case.row)
    for lane in (1, 3):  # below the target's bands: two proteins' boxes on one band are refused
        x, y = at_lane(case, lane)
        ops.place_box(s, other, x, y + 12, lane_index=lane, grow=False)
    corner = ops.place_box(s, other, 0, 0, lane_index=5, grow=False)
    assert band_of(s, corner).background_mode == "asymmetric"
    assert_nets_current(s)
    rng = np.random.default_rng(seed)
    names = sorted(PROPERTY_STEPS)
    # Every step three times, then random ones, in a random order.
    schedule = [*names * 3, *(names[int(i)] for i in rng.integers(len(names), size=24))]
    rng.shuffle(schedule)
    for name in schedule:
        before = s.project
        try:
            PROPERTY_STEPS[name](s, rng, case)
        except (OperationError, UnknownIdError):
            assert s.project is before, name  # a refusal changes nothing
        assert s.project.log[-1].content_hash == content_hash(s.project), name
        assert_nets_current(s)
    done = {entry.action for entry in s.project.log}
    assert BOX_WRITERS <= done, BOX_WRITERS - done


# --- requantify: from a project quantified before #83 to the local background ---


def legacy_scene(tmp_path: Path) -> tuple[ProjectSession, list[str], list[str]]:
    """Two images with boxes of three proteins, and a third image without any,
    quantified as before #83 (the legacy method, planted); autosaved. Returns the
    session, the images with boxes and the proteins."""
    s, image, a, b = two_proteins(tmp_path, save_to_folder)
    other = import_blot(s, textured_blot(seed=5), "reprobe β.tif")
    import_blot(s, textured_blot(seed=6), "empty α.tif")
    c = ops.add_protein(s, "α-tubulin", Role.LOADING_CONTROL, other)
    for protein, x, lane in ((a, NARROW_X, 0), (a, WIDE_X, 1), (b, 70, 2), (c, NARROW_X, 0)):
        ops.place_box(s, protein, x, ROW, lane_index=lane, grow=False)
    plant_legacy(s)
    return s, [image, other], [a, b, c]


def test_a_legacy_project_keeps_the_image_median_through_its_edits(tmp_path):
    s, _, proteins = legacy_scene(tmp_path)
    assert s.project.background_method == "global_median"
    for protein in s.project.batch.proteins:
        median = s.project.batch.image_of(protein).background
        for band in s.project.batch.boxes_of(protein).bands:
            assert (band.background_level, band.background_mode) == (median, "global_median")
    assert_nets_current(s)  # the legacy invariant: the median, each pixel floored
    [first, *_] = boxes_of(s, proteins[0]).bands
    ops.move_box(s, first.id, (60, 20, 76, 28))
    ops.set_box_size(s, proteins[0], BoxSize(width=18, height=8))
    assert s.project.background_method == "global_median"  # edits never switch it
    assert_nets_current(s)
    notices = [n for n in ops.compute(s).notices if n.code is NoticeCode.LEGACY_BACKGROUND]
    assert [n.protein_ids for n in notices] == [tuple(proteins)]


def test_requantify_switches_a_legacy_project_to_the_local_background(tmp_path):
    s, images, _ = legacy_scene(tmp_path)
    legacy = {
        band.id: band.net
        for p in s.project.batch.proteins
        for band in s.project.batch.boxes_of(p).bands
    }

    assert ops.requantify(s) == tuple(images)  # the images with bands, in order
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "requantify",
        {"from": "global_median", "to": "ring_median_v1", "images": images},
    )
    assert entry.content_hash == content_hash(s.project)
    assert s.project.background_method == "ring_median_v1"
    bands = [band for p in s.project.batch.proteins for band in s.project.batch.boxes_of(p).bands]
    assert all(band.net != legacy[band.id] for band in bands)
    assert {band.background_mode for band in bands} <= {"symmetric", "asymmetric", "image"}
    assert_nets_current(s)
    assert NoticeCode.LEGACY_BACKGROUND not in [n.code for n in ops.compute(s).notices]
    assert load_project(s.folder) == s.project  # saved with its entry
    assert history_issues(s.project) == []

    committed = s.project
    assert ops.requantify(s) == ()  # already local: a no-op, with no entry
    assert s.project is committed

    ops.undo(s)  # the switch taken back whole: the legacy nets, not recomputed
    assert s.project.background_method == "global_median"
    assert {
        b.id: b.net for p in s.project.batch.proteins for b in s.project.batch.boxes_of(p).bands
    } == legacy
    assert_nets_current(s)


def test_requantify_keeps_no_pixels_it_had_not_cached(tmp_path):
    s, images, _ = legacy_scene(tmp_path)
    reopened = ops.open_project(s.folder)
    assert ops.requantify(reopened) == tuple(images)
    assert reopened._pixels == {}  # read image by image and let go
    ops.requantify(s)
    assert reopened.project.batch == s.project.batch


def test_requantify_refuses_a_changed_image_and_changes_nothing(tmp_path):
    s, images, _ = legacy_scene(tmp_path)
    reopened = ops.open_project(s.folder)
    (reopened.folder / "images" / f"{images[1]}.tif").write_bytes(b"other bytes")
    before, length = reopened.project, len(reopened.project.log)
    with pytest.raises(OperationError) as info:
        ops.requantify(reopened)
    assert (info.value.code, info.value.ids) == (ErrorCode.IMAGE_FILE_CHANGED, (images[1],))
    assert reopened.project is before and len(reopened.project.log) == length
    assert reopened._pixels == {}


def test_requantify_switches_a_legacy_project_without_boxes(tmp_path):
    s = session_on(tmp_path)
    plant(s, lambda draft: setattr(draft, "background_method", "global_median"))
    assert ops.requantify(s) == ()
    assert s.project.background_method == "ring_median_v1"
    params = s.project.log[-1].params
    assert params == {"from": "global_median", "to": "ring_median_v1", "images": []}


def test_a_schema_1_project_opens_with_its_nets_until_requantified(tmp_path):
    s, _, proteins = legacy_scene(tmp_path)
    nets = {
        band.id: band.net
        for p in s.project.batch.proteins
        for band in s.project.batch.boxes_of(p).bands
    }
    # Its project.json as a schema-1 build wrote it (without a log, here): its
    # content as schema 1 held it.
    path = s.folder / storage.PROJECT_FILE
    doc = storage._EARLIER_CONTENT[1](storage.content_document(s.project))
    assert doc["schema_version"] == 1 and "background_method" not in doc
    doc.update(next_id=s.project.next_id, log=[])
    path.write_bytes(storage.document_bytes(doc))

    reopened = ops.open_project(s.folder, clock=FakeClock())
    project = reopened.project
    assert [entry.action for entry in project.log] == ["migrate"]
    assert project.background_method == "global_median"
    assert {
        b.id: b.net for p in project.batch.proteins for b in project.batch.boxes_of(p).bands
    } == nets
    assert_nets_current(reopened)  # the stored nets are what the pixels give
    [first, *_] = boxes_of(reopened, proteins[0]).bands
    ops.move_box(reopened, first.id, (60, 20, 76, 28))  # still the median's
    assert_nets_current(reopened)
    ops.requantify(reopened)
    assert reopened.project.background_method == "ring_median_v1"
    assert_nets_current(reopened)


# --- the local background against a known truth, through the operations ---


@pytest.mark.parametrize(
    ("membrane", "limit"), [("gradient", 0.03), ("hump", 0.03), ("haze", 0.10)]
)
def test_operations_measure_fold_changes_near_the_truth(tmp_path, membrane, limit):
    """The #83 truth blots (test_background_truth) through the whole chain: the
    image imported, the boxes placed, the fold-changes computed. The target's
    fold-change of lane i is (t_i / l_i) / (t_0 / l_0), lane 0 the reference;
    with seed 83 it is off the truth by 1.1 % (gradient), 1.2 % (hump) and
    7.9 % (haze) at most, the whole-image median's by 42 %, 71 % and 18 %."""
    pixels, rects, sizes, true_total, _ = truth_blot(membrane)
    s = session_on(tmp_path)
    image = import_blot(s, pixels.astype(np.uint8), f"{membrane} µ.tif")
    labels = [f"dose {lane}" for lane in range(TRUTH_LANES)]
    ops.set_lanes(s, [LaneInput(label) for label in labels], reference_condition=labels[0])
    proteins = [
        ops.add_protein(s, name, role, image, box_size=BoxSize(width=w, height=h))
        for name, role, (w, h) in (
            ("β-catenin", Role.TARGET, sizes[0]),
            ("α-tubulin", Role.LOADING_CONTROL, sizes[TRUTH_LANES]),
        )
    ]
    for k, (x0, y0, x1, y1) in enumerate(rects):
        x, y = x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2  # centered_rect gives x0, y0 back
        lane = k % TRUTH_LANES
        ops.place_box(s, proteins[k // TRUTH_LANES], x, y, lane_index=lane, grow=False)
    assert_nets_current(s)

    res = ops.compute(s)
    [series] = res.series
    assert truth_worst(series.fold_change[1:], truth_fold_changes(true_total)) <= limit
    # The nets of the one entry point, on the file's pixels and these boxes.
    nets = [net for column in res.proteins for net in column.nets]
    assert nets == quantify_nets(
        pixels, rects, sizes, method="ring_median", dark_on_light=True, integral=True
    )


# --- compute and export through the operations ---


def _parity_session(tmp_path: Path, pixels: np.ndarray, polarity: Polarity) -> ProjectSession:
    """The regression baseline's blot imported as a float64 TIFF, as an import
    took one before #224 refused it, with its lane table and its fixed boxes
    placed through the operations."""
    s = session_on(tmp_path)
    image = import_before_224(s, pixels, "β-catenin α-tubulin 10 µM.tif", polarity)
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


def _stored_backgrounds(s: ProjectSession) -> list[list]:
    """Each protein's stored band backgrounds, by lane: (level, mode, spread)."""
    return [
        [
            [b.background_level, b.background_mode, b.background_spread]
            for b in s.project.batch.boxes_of(protein).bands
        ]
        for protein in s.project.batch.proteins
    ]


def _golden_backgrounds(background: dict, polarity: str) -> list[list]:
    """The golden file's band backgrounds in :func:`_stored_backgrounds`' shape."""
    fields = [background[key][polarity] for key in ("levels", "modes", "spreads")]
    return [
        [list(band) for band in zip(*(field[name] for field in fields), strict=True)]
        for name in (TARGET, LOADING)
    ]


def _golden() -> tuple[dict, float, float]:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return golden["values"], golden["tolerance"]["rel"], golden["tolerance"]["abs"]


@pytest.mark.parametrize("method", list(ReduceMethod))
def test_operations_reproduce_the_regression_baseline(tmp_path, method):
    values, rel, abs_ = _golden()
    s = _parity_session(tmp_path, _blot(), DARK)
    [image] = s.project.batch.iter_images()
    assert image.bit_depth is None  # a float64 TIFF
    background = values["background"]
    assert (s.project.background_method, background["method"]) == ("ring_median_v1",) * 2
    _assert_close(image.background, background["image_median"]["dark_on_light"], rel, abs_)
    _assert_close(
        _stored_backgrounds(s), _golden_backgrounds(background, "dark_on_light"), rel, abs_
    )

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
        _assert_close(_chart_stats(*charts), reduced[case], rel, abs_, case)


def test_set_polarity_reproduces_the_light_on_dark_baseline(tmp_path):
    values, rel, abs_ = _golden()
    s = _parity_session(tmp_path, 255.0 - _blot(), DARK)  # boxed while still dark-on-light
    [image] = s.project.batch.iter_images()
    ops.set_polarity(s, image.id, LIGHT)
    [image] = s.project.batch.iter_images()
    background = values["background"]
    _assert_close(image.background, background["image_median"]["light_on_dark"], rel, abs_)
    _assert_close(
        _stored_backgrounds(s), _golden_backgrounds(background, "light_on_dark"), rel, abs_
    )
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
    assert {bar.label: bar.points for bar in series.chart.bars if bar.n} == series.groups
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
        "1",  # lanes numbered from 1, as the app does
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
    assert rows[3][:6] == ["3", "10 µM", "a1", "yes", "", ""]  # no β-catenin box in lane 3
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


def test_export_lane_table_numbers_columns_that_would_share_a_name(tmp_path):
    # "GAPDH clipped" names the clipping column of GAPDH (a project.json may hold
    # both names; the operations refuse the second): the bundle's rule applies.
    s = open_sample(tmp_path, None, make_project_with_clashing_names())
    path = ops.export_lane_table(s)
    data = path.read_bytes()
    table = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    assert table[0] == [
        "lane",
        "condition",
        "sample",
        "include",
        "GAPDH",
        "GAPDH clipped",
        "α-tubulin",
        "α-tubulin clipped",
        "GAPDH clipped (2)",
        "GAPDH clipped (2) clipped",
    ]
    rows = [dict(zip(table[0], row, strict=True)) for row in table[1:]]
    nets = results.lane_nets(s.project.batch)
    for column, protein_id in (("GAPDH", "prot-7"), ("GAPDH clipped (2)", "prot-9")):
        assert [row[column] for row in rows] == [
            "" if net is None else str(round(net, 3)) for net in nets[protein_id]
        ]
    assert [row["GAPDH clipped"] for row in rows] == ["yes", "no", "", "no"]
    assert [row["GAPDH clipped (2) clipped"] for row in rows] == ["no", "yes", "", ""]
    doc = json.loads((path.parent / ops.LANE_TABLE_RECORD_FILE).read_bytes())
    assert doc["files"] == {
        ops.LANE_TABLE_FILE: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    }
    # Both exports agree: the bundle's table of the lane table's include flags
    # holds these columns first, under the same names.
    bundle = ops.export_bundle(s, formats=[])
    applied = (bundle.folder / bundle.files[0]).read_bytes().decode("utf-8-sig")
    assert [row[:10] for row in csv.reader(io.StringIO(applied, newline=""))] == table


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
    assert band_of(s, band).box.x + boxes_of(s, protein).box_size.width == LANE_W
    assert band_of(s, band).lane_index == 5
    other = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image_of(s, protein))
    # In a row of its own below: inside the target's wide boxes, two proteins'
    # boxes would measure one band (#114).
    ops.place_box(s, other, lane_x(0), LANE_ROW + 20, lane_index=0, grow=False)
    ops.place_box(s, other, lane_x(2), LANE_ROW + 20, lane_index=2, grow=False)
    fourth = ops.place_box(s, other, lane_x(4), LANE_ROW + 20, grow=False)
    assert band_of(s, fourth).lane_index == 4  # not skewed by the shifted lane-5 box


def test_a_box_dragged_out_of_order_does_not_capture_other_lanes(tmp_path):
    s, protein = lanes_session(tmp_path)
    bands = {
        lane: ops.place_box(s, protein, lane_x(lane), LANE_ROW, lane_index=lane, grow=False)
        for lane in (0, 1, 2, 3)
    }
    ops.move_box(s, bands[2], (lane_x(5) - 5, 20, lane_x(5) + 5, 40))  # lane 2 dragged right
    other = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image_of(s, protein))
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
    # A float TIFF, bit depth unknown, as imported before #224 refused it.
    image = import_before_224(s, pixels.astype(np.float64))
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
    # #112: the band is saturated, so it is flagged as possibly over-exposed.
    assert band_of(s, band).possibly_clipped is True
    assert_nets_current(s)


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


# --- #112: possibly over-exposed bands, where the exact check cannot run ---

# Six dark bands, one per lane, on a 16-bit truth: two well clear of the limit,
# two clipped at 0 over 17 and 47 pixels of their 20x10 boxes, one that touches
# it with a single pixel (a negligible bias: the rule's accepted miss) and a
# faint one. Saved as 8-bit, lossy or in colour, the heuristic flags lanes 2, 3.
POSSIBLE_DEPTHS = (30000.0, 45000.0, 70000.0, 120000.0, 51000.0, 10000.0)
POSSIBLE_FLAGS = [False, False, True, True, False, False]
POSSIBLE_BOX = BoxSize(width=20, height=10)


def possible_truth() -> np.ndarray:
    """The six bands, 16-bit, dark on light, clipped at 0 as a detector would."""
    bands = [(lane_x(i), LANE_ROW, 6.0, 3.0, depth) for i, depth in enumerate(POSSIBLE_DEPTHS)]
    return synthetic_blot((LANE_H, LANE_W), bands)


def possible_gray8() -> np.ndarray:
    return np.round(possible_truth() / 257.0).astype(np.uint8)


def import_encoded(
    s: ProjectSession, name: str, image, polarity: Polarity = DARK, **options: int
) -> str:
    """Save a Pillow image next to the project (the format from the suffix, with
    ``options``), then import it."""
    source = s.folder.parent / "sources" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    image.save(source, **options)
    with source.open("rb") as f:
        return ops.import_image(s, f, name, kind=CHEMI, polarity=polarity)


def six_boxes(s: ProjectSession, image: str, role: Role = Role.TARGET) -> tuple[str, list[str]]:
    """A protein on ``image`` with a fixed-size box on each of the six bands."""
    if not s.project.batch.only_lanes():
        ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(6)])
    name = "β-catenin" if role is Role.TARGET else "GAPDH"
    protein = ops.add_protein(s, name, role, image, box_size=POSSIBLE_BOX)
    placed = [
        ops.place_box(s, protein, lane_x(i), LANE_ROW, lane_index=i, grow=False) for i in range(6)
    ]
    return protein, placed


def flags(s: ProjectSession, band_ids: Sequence[str]) -> tuple[list, list]:
    """The bands' exact and possible over-exposure flags."""
    bands = [band_of(s, band_id) for band_id in band_ids]
    return [b.clipped for b in bands], [b.possibly_clipped for b in bands]


def import_jpeg(
    s: ProjectSession, pixels: np.ndarray, quality: int, polarity: Polarity = DARK
) -> str:
    """Import 8-bit gray ``pixels`` saved as a JPEG of that quality."""
    from PIL import Image

    return import_encoded(s, "blot β.jpg", Image.fromarray(pixels), polarity, quality=quality)


@pytest.mark.parametrize("quality", [95, 85, 75])
@pytest.mark.parametrize("polarity", [DARK, LIGHT])
def test_bands_on_a_jpeg_are_flagged_when_possibly_over_exposed(tmp_path, quality, polarity):
    # A light-on-dark blot is the same bands inverted: its limit is 255.
    gray = possible_gray8() if polarity is DARK else 255 - possible_gray8()
    s = session_on(tmp_path)
    image = import_jpeg(s, gray, quality, polarity)
    protein, band_ids = six_boxes(s, image)
    assert flags(s, band_ids) == ([None] * 6, POSSIBLE_FLAGS)
    assert_nets_current(s)

    res = ops.compute(s)
    [column] = res.proteins
    assert (column.clipped, column.possibly_clipped) == ([None] * 6, POSSIBLE_FLAGS)
    [possibly] = [n for n in res.notices if n.code is NoticeCode.POSSIBLY_CLIPPED]
    assert (possibly.protein_ids, possibly.lane_indices, possibly.level) == (
        (protein,),
        (2, 3),
        Level.WARNING,
    )
    assert possibly.message.startswith(
        "'β-catenin' is possibly over-exposed in lanes 3, 4: each of those boxes holds 5 or"
        " more pixels within 2 grey levels of the detector limit, and its image has lossy"
        " (JPEG-type) compression, so saturation cannot be confirmed;"
    )
    # Every band is still reported as not checked, flagged or not.
    [unchecked] = [n for n in res.notices if n.code is NoticeCode.CLIPPING_NOT_CHECKED]
    assert unchecked.lane_indices == tuple(range(6))

    # The other polarity's limit is the far end of the range: nothing is near it.
    ops.set_polarity(s, image, LIGHT if polarity is DARK else DARK)
    assert flags(s, band_ids) == ([None] * 6, [False] * 6)
    assert_nets_current(s)
    assert not [n for n in ops.compute(s).notices if n.code is NoticeCode.POSSIBLY_CLIPPED]


@pytest.mark.parametrize("pixels", ["tinted rgb", "rgb16", "cmyk"])
def test_colour_and_converted_images_are_flagged_on_their_gray_mean(tmp_path, pixels):
    from PIL import Image

    gray = possible_gray8()
    s = session_on(tmp_path)
    if pixels == "tinted rgb":
        # Blue 40 levels below gray, as a tinted export: in lane 1, 11 pixels of
        # blue come within 2 levels of 0 while the gray mean stays above 12, so
        # the band is not flagged (a single channel near the limit is not enough);
        # the saturated cores are 0 in all three channels.
        blue = np.clip(gray.astype(int) - 40, 0, 255).astype(np.uint8)
        assert np.count_nonzero(blue[LANE_ROW - 5 : LANE_ROW + 5, 80:100] <= 2) == 11
        image = import_blot(s, np.stack([gray, gray, blue], axis=-1), "tinted blot.tif")
        expected = (8, ["color_channels_differ"])
    elif pixels == "rgb16":
        # 16-bit colour: 5 pixels within 514 counts of 0 (2 levels of 255).
        truth = possible_truth()
        rgb = np.stack([truth, truth, truth], axis=-1)
        rgb[:4, :4] = (65535, 0, 0)  # a red marking, away from the bands
        image = import_blot(s, rgb, "blot 16-bit colour.tif")
        expected = (16, ["color_channels_differ"])
    else:  # imported before #207 refused CMYK
        cmyk = io.BytesIO()
        Image.fromarray(gray).convert("CMYK").save(cmyk, format="TIFF")
        cmyk.seek(0)
        image = _before_207(s, cmyk, "blot CMYK.tif")
        expected = (8, ["cmyk_converted"])
    stored = s.project.batch.find_image(image)
    assert (stored.bit_depth, [w.code for w in stored.import_warnings]) == expected
    _, band_ids = six_boxes(s, image)
    assert flags(s, band_ids) == ([None] * 6, POSSIBLE_FLAGS)
    assert_nets_current(s)
    [possibly] = [n for n in ops.compute(s).notices if n.code is NoticeCode.POSSIBLY_CLIPPED]
    assert possibly.lane_indices == (2, 3)
    if pixels == "rgb16":
        assert "within 514 grey levels (2 on an 8-bit scale) of the detector limit" in (
            possibly.message
        )


@pytest.mark.parametrize("pixels", ["tiff16", "png8"])
def test_no_possible_flag_where_the_exact_check_runs(tmp_path, pixels):
    # A lossless gray image of known depth is checked exactly (#44): the single
    # pixel at 0 in lane 4 counts there, and nothing is assessed as possible.
    from PIL import Image

    s = session_on(tmp_path)
    if pixels == "tiff16":
        image = import_blot(s, possible_truth())
    else:
        image = import_encoded(s, "blot.png", Image.fromarray(possible_gray8()))
    assert s.project.batch.find_image(image).import_warnings == []
    _, band_ids = six_boxes(s, image)
    assert flags(s, band_ids) == ([False, False, True, True, True, False], [None] * 6)
    assert_nets_current(s)
    res = ops.compute(s)
    codes = {n.code for n in res.notices}
    assert NoticeCode.POSSIBLY_CLIPPED not in codes
    assert NoticeCode.CLIPPING_NOT_CHECKED not in codes
    assert res.proteins[0].possibly_clipped == [None] * 6
    # Nothing unset is written: the saved project holds no possibly_clipped key.
    assert b"possibly_clipped" not in storage.project_to_json(s.project)


def test_an_image_of_unknown_depth_is_not_assessed(tmp_path):
    # A float image has no range limit: neither check runs, and only the
    # clipping_not_checked notice speaks for its bands. An import refuses one
    # since #224; one imported before keeps being measured so.
    s = session_on(tmp_path)
    image = import_before_224(s, possible_truth().astype(np.float64))
    _, band_ids = six_boxes(s, image)
    assert flags(s, band_ids) == ([None] * 6, [None] * 6)
    codes = [n.code for n in ops.compute(s).notices]
    assert NoticeCode.POSSIBLY_CLIPPED not in codes
    assert NoticeCode.CLIPPING_NOT_CHECKED in codes
    assert s.project.batch.find_image(image).bit_depth is None


def test_a_possibly_over_exposed_loading_control_names_the_values_it_biases(tmp_path):
    s = session_on(tmp_path)
    image = import_jpeg(s, possible_gray8(), 85)
    loading, _ = six_boxes(s, image, Role.LOADING_CONTROL)
    ops.set_lanes(
        s,
        [LaneInput(f"c{i}", included=i != 3) for i in range(6)],
    )
    res = ops.compute(s)
    [possibly] = [n for n in res.notices if n.code is NoticeCode.POSSIBLY_CLIPPED]
    assert (possibly.protein_ids, possibly.lane_indices) == ((loading,), (2,))
    assert possibly.message == (
        "'GAPDH' is possibly over-exposed in lane 3: its box holds 5 or more pixels within"
        " 2 grey levels of the detector limit, and its image has lossy (JPEG-type)"
        " compression, so saturation cannot be confirmed; if it is over-exposed there, its"
        " net is an under-estimate, which biases every value normalized to it, and a"
        " saturated band next to it may have merged into the same box, which then holds both"
        " bands and its net may be too high; check the imager's original capture, or shorten"
        " the exposure and image the membrane again"
    )
    # The excluded lane 4 holds a value: the all-lanes set has its own notice.
    [every] = [n for n in res.all_lanes.notices if n.code is NoticeCode.POSSIBLY_CLIPPED]
    assert every.lane_indices == (2, 3)
    assert every.message.startswith("'GAPDH' is possibly over-exposed in lanes 3, 4:")


def test_a_band_edit_fills_the_flag_of_a_project_saved_before_it(tmp_path):
    # A project quantified before #112 holds no possibly_clipped on its JPEG
    # bands (None: not assessed); the next re-quantification of the image
    # assesses every band on it.
    s = session_on(tmp_path)
    image = import_jpeg(s, possible_gray8(), 95)
    protein, band_ids = six_boxes(s, image)

    def forget(draft: Project) -> None:
        for band in draft.batch.boxes_of(draft.batch.find_protein(protein)).bands:
            band.possibly_clipped = None

    plant(s, forget)
    assert flags(s, band_ids)[1] == [None] * 6
    assert not [n for n in ops.compute(s).notices if n.code is NoticeCode.POSSIBLY_CLIPPED]
    ops.move_box(s, band_ids[5], [lane_x(5) - 10, LANE_ROW - 4, lane_x(5) + 10, LANE_ROW + 6])
    assert flags(s, band_ids)[1] == POSSIBLE_FLAGS
    assert_nets_current(s, only=[image])


def test_a_single_colour_export_hides_its_saturation_from_the_grey_mean(tmp_path):
    # A known limit: a light-on-dark export in one colour (a green lookup
    # table) saturates its green channel alone, so the grey mean it is assessed
    # on stays at a third of the range. Its bands are assessed with no sign of
    # it (False), and every one is still reported as not checked.
    s = session_on(tmp_path)
    green = 255 - possible_gray8()
    zeros = np.zeros_like(green)
    image = import_blot(s, np.stack([zeros, green, zeros], axis=-1), "green blot.tif", LIGHT)
    _, band_ids = six_boxes(s, image)
    assert np.count_nonzero(green == 255) > 60
    assert flags(s, band_ids) == ([None] * 6, [False] * 6)
    res = ops.compute(s)
    assert NoticeCode.POSSIBLY_CLIPPED not in {n.code for n in res.notices}
    [unchecked] = [n for n in res.notices if n.code is NoticeCode.CLIPPING_NOT_CHECKED]
    assert unchecked.lane_indices == tuple(range(6))


REQUANTIFY_HINT = (
    "; its boxes were measured before Proteia looked for pixels near the detector limit:"
    " requantify to look for them"
)


def test_requantify_assesses_the_bands_of_a_project_saved_before_it(tmp_path):
    # A project quantified before #112, on the local background already: its
    # JPEG bands hold no possibly_clipped. The not-checked notice says to
    # requantify, and requantify assesses the bands of that image alone, in one
    # change, leaving an image already assessed as it was.
    from PIL import Image

    s = session_on(tmp_path, save_to_folder)
    old = import_jpeg(s, possible_gray8(), 95)
    target, target_ids = six_boxes(s, old)
    assessed = import_encoded(s, "blot α.jpg", Image.fromarray(possible_gray8()), quality=95)
    loading, loading_ids = six_boxes(s, assessed, Role.LOADING_CONTROL)

    def forget(draft: Project) -> None:
        for band in draft.batch.boxes_of(draft.batch.find_protein(target)).bands:
            band.possibly_clipped = None

    plant(s, forget)
    assert ops.unassessed_images(s.project.batch) == [old]
    res = ops.compute(s)
    hinted = {
        n.protein_ids: n.message.endswith(REQUANTIFY_HINT)
        for n in res.notices
        if n.code is NoticeCode.CLIPPING_NOT_CHECKED
    }
    assert hinted == {(target,): True, (loading,): False}
    possibly = [n.protein_ids for n in res.notices if n.code is NoticeCode.POSSIBLY_CLIPPED]
    assert possibly == [(loading,)]
    kept = [band_of(s, band_id) for band_id in loading_ids]

    assert ops.requantify(s) == (old,)
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "requantify",
        {"from": "ring_median_v1", "to": "ring_median_v1", "images": [old]},
    )
    assert entry.content_hash == content_hash(s.project)
    assert flags(s, target_ids) == ([None] * 6, POSSIBLE_FLAGS)
    assert [band_of(s, band_id) for band_id in loading_ids] == kept
    assert_nets_current(s)
    assert ops.unassessed_images(s.project.batch) == []
    res = ops.compute(s)
    assert not [n for n in res.notices if n.message.endswith(REQUANTIFY_HINT)]
    possibly = [n for n in res.notices if n.code is NoticeCode.POSSIBLY_CLIPPED]
    assert [(n.protein_ids, n.lane_indices) for n in possibly] == [
        ((target,), (2, 3)),
        ((loading,), (2, 3)),
    ]
    assert load_project(s.folder) == s.project
    assert history_issues(s.project) == []

    committed = s.project
    assert ops.requantify(s) == ()  # nothing left to assess: a no-op, with no entry
    assert s.project is committed
    ops.undo(s)  # taken back whole: not assessed again
    assert flags(s, target_ids)[1] == [None] * 6
    assert ops.unassessed_images(s.project.batch) == [old]


@pytest.mark.parametrize("pixels", ["png8", "float"])
def test_images_the_possible_check_does_not_run_on_are_never_unassessed(tmp_path, pixels):
    # The exact check runs on a lossless gray image; an unknown depth has no
    # limit: neither is waiting for requantify, and its notice has no hint.
    from PIL import Image

    s = session_on(tmp_path)
    if pixels == "png8":
        image = import_encoded(s, "blot.png", Image.fromarray(possible_gray8()))
    else:  # as imported before #224 refused it
        image = import_before_224(s, possible_truth().astype(np.float64))
    six_boxes(s, image)
    assert ops.unassessed_images(s.project.batch) == []
    assert not [n for n in ops.compute(s).notices if n.message.endswith(REQUANTIFY_HINT)]
    assert ops.requantify(s) == ()


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
    return list(band.box.rect(s.project.batch.boxes_of(protein).box_size))


def _size_of(s: ProjectSession, protein_id: str) -> dict[str, int]:
    size = s.project.batch.boxes_of(s.project.batch.find_protein(protein_id)).box_size
    return {"width": size.width, "height": size.height}


def _protein(protein_id: str, row_id: str, name: str, role: str, image_id: str, **fields) -> dict:
    """add_protein's params with its defaults (no expected MW, the default MW
    tolerance, no loading controls, the initial box size of the blot, nothing
    pinned), and its row's id."""
    return {
        "protein_id": protein_id,
        "row_id": row_id,
        "name": name,
        "role": role,
        "image_id": image_id,
        "expected_mw": None,
        "mw_tolerance": 0.1,
        "loading_control_ids": [],
        "box_size": {"width": 20, "height": 5},  # boxes.initial_box_size(W, H)
        "pinned_targets": [],
        **fields,
    }


def _plant_step_records(s: ProjectSession) -> None:
    """:data:`LOGGED_STEPS`' not-detected records, in one planted change."""

    def change(draft: Project) -> None:
        draft.batch.boxes_of(draft.batch.find_protein("prot-7")).undetected.append(_record(2))
        draft.batch.boxes_of(draft.batch.find_protein("prot-15")).undetected.append(_record(1))

    plant(s, change)


# Every logged operation, as (call, action, the params it must log). ``expected``
# reads the session after the call, for values the pixels decide (grown rects).
# Ids follow the one counter: img-1, mem-2, the lane table lt-3 and the area
# arena-4; img-5 and its area arena-6; then each protein and its row, and bands.
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
            "arena_id": "arena-4",
            "lane_table_id": "lt-3",
        },
    ),
    (
        lambda s: import_blot(s, blot(), "reprobe β.tif", LIGHT, membrane_id="mem-2"),
        "import_image",
        lambda s: {
            "image_id": "img-5",
            "membrane_id": "mem-2",
            "new_membrane": False,
            "original_name": "reprobe β.tif",
            "kind": "chemiluminescence",
            "polarity": "light_on_dark",
            "sha256": _file_sha256(s, "img-5.tif"),
            "arena_id": "arena-6",
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
        lambda s: _protein("prot-7", "row-8", "β-catenin", "target", "img-1", expected_mw=92.0),
    ),
    (
        lambda s: ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, "img-5"),
        "add_protein",
        lambda s: _protein("prot-9", "row-10", "GAPDH", "loading control", "img-5"),
    ),
    (
        lambda s: ops.add_protein(s, "p53", Role.TARGET, "img-1"),
        "add_protein",
        lambda s: _protein("prot-11", "row-12", "p53", "target", "img-1"),
    ),
    (
        # A target becomes the second loading control: β-catenin, which used GAPDH
        # implicitly, is pinned to it. p53 itself is not listed, and its name,
        # typed the same once cleaned, is not a change.
        lambda s: ops.edit_protein(s, "prot-11", name=" p53 ", role=Role.LOADING_CONTROL),
        "edit_protein",
        lambda s: {
            "protein_id": "prot-11",
            "pinned_targets": ["prot-7"],
            "role": "loading control",
            "dropped_undetected": [],
        },
    ),
    (
        lambda s: ops.add_protein(
            s, "β-actin", Role.TARGET, "img-1", loading_control_ids=["prot-9"]
        ),
        "add_protein",
        lambda s: _protein(
            "prot-13", "row-14", "β-actin", "target", "img-1", loading_control_ids=["prot-9"]
        ),
    ),
    (
        lambda s: ops.remove_protein(s, "prot-9"),
        "remove_protein",
        lambda s: {
            "protein_id": "prot-9",
            "removed": ["prot-9", "row-10"],
            "detached_targets": ["prot-7", "prot-13"],
            "unpaired_images": [],
            "unfitted_membranes": [],
            "removed_undetected": [],
        },
    ),
    (
        # A second loading control is added: the targets using p53 implicitly are pinned.
        lambda s: ops.add_protein(
            s, "α-tubulin", Role.LOADING_CONTROL, "img-5", box_size=BoxSize(width=10, height=6)
        ),
        "add_protein",
        lambda s: _protein(
            "prot-15",
            "row-16",
            "α-tubulin",
            "loading control",
            "img-5",
            box_size={"width": 10, "height": 6},
            pinned_targets=["prot-7", "prot-13"],
        ),
    ),
    (
        # A target turned into a loading control loses its list: an implicit change.
        lambda s: ops.edit_protein(s, "prot-13", role=Role.LOADING_CONTROL, expected_mw=42),
        "edit_protein",
        lambda s: {
            "protein_id": "prot-13",
            "pinned_targets": [],
            "role": "loading control",
            "expected_mw": 42.0,
            "loading_control_ids": [],
            "dropped_undetected": [],
        },
    ),
    (
        lambda s: ops.place_box(s, "prot-7", NARROW_X, ROW, lane_index=0, grow=True),
        "place_box",
        lambda s: {
            "band_id": "band-17",
            "protein_id": "prot-7",
            "x": NARROW_X,
            "y": ROW,
            "grow": True,
            "lane_index": 0,
            "lane_proposed": False,
            "rect": _rect_of(s, "band-17"),
            "box_size": _size_of(s, "prot-7"),  # the first seed click sets it
            "replaced_undetected": None,
        },
    ),
    (
        lambda s: ops.place_box(s, "prot-7", WIDE_X, ROW, lane_index=1, grow=True),
        "place_box",
        lambda s: {
            "band_id": "band-18",
            "protein_id": "prot-7",
            "x": WIDE_X,
            "y": ROW,
            "grow": True,
            "lane_index": 1,
            "lane_proposed": False,
            "rect": _rect_of(s, "band-18"),
            "box_size": _size_of(s, "prot-7"),  # grown by the wide band
            "replaced_undetected": None,
        },
    ),
    (
        lambda s: ops.place_box(s, "prot-7", 150, ROW, grow=False),
        "place_box",
        lambda s: {
            "band_id": "band-19",
            "protein_id": "prot-7",
            "x": 150,
            "y": ROW,
            "grow": False,
            "lane_index": 2,  # proposed from the two boxes before
            "lane_proposed": True,
            "rect": list(boxes.centered_rect(150, ROW, boxes_of(s, "prot-7").box_size, W, H)),
            "box_size": _size_of(s, "prot-7"),
            "replaced_undetected": None,
        },
    ),
    (
        lambda s: ops.move_box(s, "band-17", (30, 44, 10, 20)),
        "move_box",
        lambda s: {  # after the centre snap, not as dragged
            "band_id": "band-17",
            "rect": list(boxes.centered_rect(20, 32, boxes_of(s, "prot-7").box_size, W, H)),
        },
    ),
    (
        lambda s: ops.set_box_size(s, "prot-7", BoxSize(width=12, height=6)),
        "set_box_size",
        lambda s: {
            "protein_id": "prot-7",
            "box_size": {"width": 12, "height": 6},
            "fitted_size": {"width": 12, "height": 6},  # no padding yet: the same size
        },
    ),
    (
        lambda s: ops.set_box_padding(s, "prot-7", across=2, along=3),
        "set_box_padding",
        lambda s: {
            "protein_id": "prot-7",
            "box_padding": {"across": 2, "along": 3},
            "box_size": {"width": 16, "height": 12},  # the fitted 12x6 plus the padding
        },
    ),
    (
        lambda s: ops.set_polarity(s, "img-1", LIGHT),
        "set_polarity",
        lambda s: {"image_id": "img-1", "polarity": "light_on_dark", "dropped_undetected": []},
    ),
    (
        # As a project quantified before #83 (a migrated one), planted.
        lambda s: plant_legacy(s),
        "plant",
        lambda s: {},
    ),
    (
        lambda s: ops.requantify(s),
        "requantify",
        lambda s: {"from": "global_median", "to": "ring_median_v1", "images": ["img-1"]},
    ),
    (
        lambda s: ops.remove_box(s, "band-18"),
        "remove_box",
        lambda s: {"band_id": "band-18", "protein_id": "prot-7", "lane_index": 1},
    ),
    (
        lambda s: ops.set_box_lane(s, "band-19", 1),  # into the lane band-18 left
        "set_box_lane",
        lambda s: {
            "band_id": "band-19",
            "protein_id": "prot-7",
            "from_lane": 2,
            "lane_index": 1,
            "replaced_undetected": None,
        },
    ),
    (
        # Records no operation writes on this blot (a row commit would), planted
        # through apply_change: in β-catenin's empty lane 2 and α-tubulin's lane 1.
        _plant_step_records,
        "plant",
        lambda s: {},
    ),
    (
        lambda s: ops.remove_undetected(s, "prot-7", 2),
        "remove_undetected",
        lambda s: {
            "protein_id": "prot-7",
            "lane_index": 2,
            "band_index": 0,
            "removed": _record_json("prot-7", _record(2)),
        },
    ),
    (
        lambda s: ops.place_box(s, "prot-15", NARROW_X, ROW, lane_index=0, grow=False),
        "place_box",
        lambda s: {
            "band_id": "band-20",
            "protein_id": "prot-15",
            "x": NARROW_X,
            "y": ROW,
            "grow": False,
            "lane_index": 0,
            "lane_proposed": False,
            "rect": list(boxes.centered_rect(NARROW_X, ROW, BoxSize(width=10, height=6), W, H)),
            "box_size": {"width": 10, "height": 6},
            "replaced_undetected": None,
        },
    ),
    (
        lambda s: ops.clear_boxes(s, "prot-15"),
        "clear_boxes",
        lambda s: {
            "protein_id": "prot-15",
            "removed": ["band-20"],
            "lane_indices": [0],
            "dropped_undetected": [_record_json("prot-15", _record(1))],
        },
    ),
    (
        lambda s: ops.remove_image(s, "img-1"),
        "remove_image",
        lambda s: {
            "image_id": "img-1",
            "removed": [
                "img-1",
                "arena-4",
                "prot-7",
                "prot-11",
                "prot-13",
                "row-8",
                "row-12",
                "row-14",
                "band-17",
                "band-19",
            ],
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
        "set_box_padding",
        "remove_undetected",
        "clear_boxes",
        "requantify",
        "plant",  # the test's own records and legacy method, as no operation writes them
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
    assert doc["settings"] == {**record.settings(), "background_method": "ring_median_v1"}
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
        draft.batch.boxes_of(protein).undetected.extend(records)

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
    return [(u.lane_index, u.band_index) for u in boxes_of(session, protein_id).undetected]


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
    kept = boxes_of(s, tubulin).undetected

    ops.set_polarity(s, first, LIGHT)
    assert (_keys(s, beta), _keys(s, gapdh)) == ([], [])
    assert boxes_of(s, tubulin).undetected == kept  # another image's records stay
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
    kept = boxes_of(s, protein).undetected
    ops.move_box(s, band, (30, 22, 50, 38))
    ops.set_box_size(s, protein, BoxSize(width=12, height=6))
    ops.edit_protein(s, protein, name="β-catenin (E-5)", expected_mw=92)
    assert boxes_of(s, protein).undetected == kept


def _records_json(session: ProjectSession, protein_id: str) -> list[dict]:
    return [_record_json(protein_id, u) for u in boxes_of(session, protein_id).undetected]


def test_removals_take_the_records_along(tmp_path):
    s = open_sample(tmp_path, project=make_project_with_undetected())
    assert _keys(s, "prot-9") == [(2, 0), (3, 0)]
    gapdh = _records_json(s, "prot-9")
    cascade = ops.remove_protein(s, "prot-9")  # GAPDH: records only in lanes 2 and 3
    assert cascade.removed == ("prot-9", "row-26", "band-17", "band-18")  # records: no ids
    # The log keeps them whole, since they have no ids to name them by.
    assert s.project.log[-1].params == {
        "protein_id": "prot-9",
        **_listed(cascade),
        "removed_undetected": gapdh,
    }
    beta = _records_json(s, "prot-7")
    cascade = ops.remove_image(s, "img-2")  # β-catenin, with its lane-2 record
    assert cascade.removed == (
        "img-2",
        "arena-20",
        "prot-7",
        "row-24",
        "band-10",
        "band-11",
        "band-12",
    )
    assert s.project.log[-1].params == {
        "image_id": "img-2",
        **_listed(cascade),
        "removed_undetected": beta,
        "dropped_undetected": [],
    }
    assert all(not s.project.batch.boxes_of(p).undetected for p in s.project.batch.proteins)
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


def test_edit_protein_drops_mw_guided_records_when_the_tolerance_changes(tmp_path):
    # The rows a row placed by its expected MW searched span the MW tolerance
    # (twice it, either way): a new tolerance leaves its records about rows
    # nobody looked in, and they go, logged in full. A row box's record does
    # not depend on it, nor does another protein's; the same tolerance again
    # changes nothing.
    s, image, beta = boxed(tmp_path)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    guided, row = _record(1, source=ProposalSource.MW_GUIDED), _record(2, snr=-0.5)
    plant_records(s, beta, guided, row)
    plant_records(s, gapdh, _record(0, source=ProposalSource.MW_GUIDED))
    committed = s.project
    ops.edit_protein(s, beta, mw_tolerance=protein_of(s, beta).mw_tolerance)
    assert s.project is committed and _keys(s, beta) == [(1, 0), (2, 0)]
    ops.edit_protein(s, beta, mw_tolerance=0.2)
    assert _keys(s, beta) == [(2, 0)]
    assert _keys(s, gapdh) == [(0, 0)]
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "edit_protein",
        {
            "protein_id": beta,
            "pinned_targets": [],
            "mw_tolerance": 0.2,
            "dropped_undetected": [_record_json(beta, guided)],
        },
    )
    assert entry.content_hash == content_hash(s.project)
    ops.undo(s)
    assert _keys(s, beta) == [(1, 0), (2, 0)]


def test_a_calibration_change_drops_the_membranes_mw_guided_records(tmp_path):
    # Only on the images whose curve changed (#58): the reprobe img-4, a register
    # group of its own with no points, keeps its curve (none) and GAPDH its records.
    s = open_sample(tmp_path / "marker", project=make_project_with_undetected())
    plant_58_fit(s, "mem-1")  # as a #58 build leaves it
    # GAPDH (img-4, mem-1): a row-box record in lane 2, an MW-guided one in lane 3.
    gapdh = _records_json(s, "prot-9")
    # β-catenin (img-2, linked to the marker img-3): its lane-2 record, MW-guided.

    def guided(draft: Project) -> None:
        draft.batch.boxes_of(draft.batch.find_protein("prot-7")).undetected[
            0
        ].source = ProposalSource.MW_GUIDED

    plant(s, guided)
    [beta_guided] = _records_json(s, "prot-7")
    # α-tubulin is on mem-5, whose calibration does not change.
    tubulin = _record(0, band_index=1, region=(0, 10, 20, 30), source=ProposalSource.MW_GUIDED)
    plant_records(s, "prot-8", tubulin, bands=2)

    cascade = ops.remove_image(s, "img-3")  # the marker, with two calibration points
    assert cascade.unfitted_membranes == ("mem-1",)
    assert _keys(s, "prot-7") == []  # img-2 is left with one point: its curve went
    assert _records_json(s, "prot-9") == gapdh  # img-4's curve did not change
    assert _keys(s, "prot-8") == [(0, 1)]  # another membrane's stays
    entry = s.project.log[-1]
    assert entry.params == {
        "image_id": "img-3",
        **_listed(cascade),
        "removed_undetected": [],
        "dropped_undetected": [beta_guided],
    }
    assert entry.content_hash == content_hash(s.project)

    # Stored as before #58 (fitted over the whole membrane, img-4 included), the
    # refit changes the curve of every image of it: GAPDH's MW-guided record goes too.
    s = open_sample(tmp_path / "legacy", project=make_project_with_undetected())
    [_, gapdh_guided] = _records_json(s, "prot-9")
    plant(s, guided)
    ops.remove_image(s, "img-3")
    assert (_keys(s, "prot-7"), _keys(s, "prot-9")) == ([], [(2, 0)])  # the row-box record stays
    assert s.project.log[-1].params["dropped_undetected"] == [beta_guided, gapdh_guided]

    # An image without calibration points leaves the calibration, and the MW-guided
    # records of the membrane's other proteins, as they were.
    s = open_sample(tmp_path / "reprobe", project=make_project_with_undetected())

    def guided(draft: Project) -> None:
        draft.batch.boxes_of(draft.batch.find_protein("prot-7")).undetected[
            0
        ].source = ProposalSource.MW_GUIDED

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

    dropped = ops._drop_undetected_where(draft.batch, draft.batch.find_protein(protein), drop)
    assert asked == [0, 1, 2]
    assert dropped == [_record_json(protein, _record(1))]
    assert [
        u.lane_index for u in draft.batch.boxes_of(draft.batch.find_protein(protein)).undetected
    ] == [0, 2]


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
    image = batch.image_of(batch.find_protein(protein_id))
    return rowdetect.detect_row(
        s.pixels(image.id),
        row,
        len(batch.only_lanes()),
        background=image.background,
        dark_on_light=image.polarity.dark_on_light,
        saturated_at=saturation_level(
            clipping_depth(image.bit_depth, image.import_warnings),
            possible_clipping_depth(image.bit_depth, image.import_warnings),
            dark_on_light=image.polarity.dark_on_light,
        ),
    )


def lane_bands(s: ProjectSession, protein_id: str) -> dict[int, Band]:
    """The protein's band-index-0 boxes by lane."""
    return {b.lane_index: b for b in boxes_of(s, protein_id).bands if b.band_index == 0}


def rects_by_lane(s: ProjectSession, protein_id: str) -> dict[int, tuple[int, int, int, int]]:
    size = boxes_of(s, protein_id).box_size
    return {lane: band.box.rect(size) for lane, band in lane_bands(s, protein_id).items()}


def at_lane(case: RowCase, lane: int) -> tuple[int, int]:
    """A point on the lane's band (its true centre)."""
    return round(case.lane_cx[lane]), round(case.lane_cy[lane])


def edit_by_hand(s: ProjectSession, band_id: str, *, to: tuple[int, int] | None = None) -> None:
    """Move a box 1 px to the right, or centre it on ``to``: the user edited it."""
    protein, band = s.project.batch.find_band(band_id)
    x0, y0, x1, y1 = band.box.rect(s.project.batch.boxes_of(protein).box_size)
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
    y: int = 20,
) -> str:
    """A second protein on the image with a small box in each of ``lanes``
    (every lane by default), at the lane's true centre moved ``dx`` px, in a row
    of its own above the case's, centred on image row ``y``: lanes already
    placed on the image. Mirrored, the lanes are numbered right to left;
    ``offset`` is added to each lane number (a protein numbered a lane off).
    Another such protein in the same lanes takes another ``y``: two proteins'
    boxes on one band are refused (#114)."""
    other = ops.add_protein(
        s, name, Role.LOADING_CONTROL, image_id, box_size=BoxSize(width=20, height=8)
    )
    n = case.n_lanes
    for lane in range(n) if lanes is None else lanes:
        index = (n - 1 - lane if mirrored else lane) + offset
        ops.place_box(s, other, round(case.lane_cx[lane] + dx), y, lane_index=index, grow=False)
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
    assert boxes_of(s, protein).box_size == found.size
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
    records = boxes_of(s, protein).undetected
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
    ("wide_next_empty", 3, "unassigned"),  # lane 2's band spills over it, in no piece
    ("nbr_above_miss", 2, "edge_signal"),  # only the neighbouring row reaches the limit
    ("vstreak_empty", 4, "edge_signal"),  # a streak peaking on an edge row of the box
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
    assert boxes_of(s, protein).undetected == []
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


# #116: a band in lane 3 alone. The other lanes' slots are that band's centre
# stepped by a pitch the box's width suggests, whatever lanes the image holds:
# those check the band's own lane, not the slots.
ONE_BAND = adversarial_row("one band", 1000, missing=[0, 1, 2, 4, 5])
OTHER_LANES = (0, 1, 2, 4, 5)
# The box dragged 60 px loose on the left, under a pitch: the slots the box's
# width suggests lie up to 44 px off the lanes, lane 0's beside its band.
LOOSE_ONE_BAND_ROW = (ONE_BAND.row[0] - 60, *ONE_BAND.row[1:])


@pytest.mark.parametrize(
    ("placed", "row"),
    [
        ((), ONE_BAND.row),
        ((2,), ONE_BAND.row),
        ((0, 5), ONE_BAND.row),
        ((0, 5), LOOSE_ONE_BAND_ROW),
        (tuple(range(6)), LOOSE_ONE_BAND_ROW),
    ],
    ids=[
        "no-lanes-placed",
        "one-lane-placed",
        "two-lanes-placed",
        "two-lanes-placed-loose-box",
        "every-lane-placed-loose-box",
    ],
)
def test_one_band_writes_no_record_at_slots_that_rest_on_it(tmp_path, placed, row):
    # The slots rest on the one band: the box is placed, the other lanes get
    # no record (an older one goes, as this run's outcome replaces it), and
    # the answer says which.
    s, image, protein = row_session(tmp_path, ONE_BAND)
    if placed:
        other_protein_in_lanes(s, image, ONE_BAND, placed)
    old = _record(0)
    plant_records(s, protein, old)
    found = detected(s, protein, row)
    assert [lane.reason for lane in found.lanes] == ["no_band"] * 3 + ["band"] + ["no_band"] * 2

    placement = ops.detect_row_boxes(s, protein, row)
    assert sorted(lane_bands(s, protein)) == [3]
    assert boxes_of(s, protein).undetected == []
    assert (placement.undetected_lanes, placement.unmeasured_lanes) == ((), OTHER_LANES)
    assert placement.unlocated_lanes == OTHER_LANES
    assert [lane for lane, *_ in placement.empty] == list(OTHER_LANES)
    params = s.project.log[-1].params
    assert (params["undetected_written"], params["unlocated_lanes"]) == ([], list(OTHER_LANES))
    assert params["dropped_undetected"] == [_record_json(protein, old)]
    assert ops.compute(s).proteins[0].detected == [None, None, None, True, None, None]


# Rows whose one empty lane holds nothing that reaches the detection limit, or
# something the detector cannot read.
EMPTY_LANES = [
    pytest.param("missing_middle", 2, "no_band", id="no_band"),
    *(pytest.param(*lane, id=f"{lane[2]}-{lane[0]}") for lane in UNREADABLE_LANES),
]
# Lane 2's box in wide_next_empty would overlap a box kept in its empty lane 3.
KEPT_EMPTY_LANES = [lane for lane in EMPTY_LANES if lane.values[0] != "wide_next_empty"]


@pytest.mark.parametrize("source", [ProposalSource.CLICK, ProposalSource.MANUAL])
@pytest.mark.parametrize(("key", "lane", "reason"), KEPT_EMPTY_LANES)
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
    assert boxes_of(s, protein).undetected == []
    [(empty_lane, empty_reason, _, _)] = placement.empty
    assert (empty_lane, empty_reason) == (lane, reason)
    # The box survives, so the size only grows; it stays on its own centre.
    size = BoxSize(width=found.size.width, height=30)
    assert placement.box_size == boxes_of(s, protein).box_size == size
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
    assert placement.box_size == boxes_of(s, protein).box_size == found.size
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
    assert boxes_of(s, protein).undetected == []
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
    for lane in (1, 2):  # over the lower part of the target's bands
        x, y = at_lane(case, lane)
        ops.place_box(s, other, x, y + 8, lane_index=lane, grow=False)
    plant_records(s, other, _record(0))
    before, before_boxes = protein_of(s, other), boxes_of(s, other)

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert None not in placement.band_ids
    # Its boxes, size and records stay; only what its pixels give is measured
    # again, since every ring on the image now leaves out the row's boxes.
    measured = {
        "net",
        "background_level",
        "background_mode",
        "background_spread",
        "clipped",
        "possibly_clipped",
    }
    after, after_boxes = protein_of(s, other), boxes_of(s, other)
    assert after.model_dump(exclude={"rows"}) == before.model_dump(exclude={"rows"})
    kept = after_boxes.model_dump(exclude={"bands"})
    assert kept == before_boxes.model_dump(exclude={"bands"})
    assert [b.model_dump(exclude=measured) for b in after_boxes.bands] == [
        b.model_dump(exclude=measured) for b in before_boxes.bands
    ]
    assert [b.net for b in after_boxes.bands] != [b.net for b in before_boxes.bands]
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
    assert [u.lane_index for u in boxes_of(s, protein).undetected] == [2]
    # Lane 3: the record gives way to the band found there.
    assert bands[3].id == placement.band_ids[3] not in (clicked, fixed, guided, dropped)
    # Nothing survives, so the size and the boxes are this detection's.
    assert boxes_of(s, protein).box_size == found.size
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
    old_size = boxes_of(s, protein).box_size
    found = detected(s, protein, case.row)

    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.kept_lanes == (1, 2)
    assert (placement.band_ids[1], placement.band_ids[2]) == (on_band, on_empty)
    bands = lane_bands(s, protein)
    assert (bands[1].id, bands[2].id) == (on_band, on_empty)
    assert (bands[1].source, bands[2].source) == (ProposalSource.CLICK, ProposalSource.MANUAL)
    assert bands[1].manually_edited and bands[2].manually_edited
    # A kept box's lane gets no record, even where nothing was detected.
    assert boxes_of(s, protein).undetected == []
    assert (placement.undetected_lanes, placement.unmeasured_lanes) == ((), ())
    assert placement.empty[0][:2] == (2, "no_band")
    # Boxes survive, so the size only grows; they stay on their own centres.
    size = BoxSize(
        width=max(old_size.width, found.size.width),
        height=max(old_size.height, found.size.height),
    )
    assert boxes_of(s, protein).box_size == placement.box_size == size
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
    assert placement.box_size == boxes_of(s, protein).box_size == size
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
    assert placement.box_size == boxes_of(s, protein).box_size == size
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
    assert boxes_of(s, protein).undetected[1] == second
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
    assert boxes_of(s, protein).box_size == second.box_size == found.size
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


@pytest.mark.parametrize(("stored", "hollow"), [("uint16", True), ("jpeg", True), ("float", False)])
def test_a_hollow_band_is_over_exposed_where_the_image_s_limit_is_known(tmp_path, stored, hollow):
    # #121: lane 2's band clipped flat at 0 around a lighter centre. Its
    # saturated pixels are counted as the over-exposure checks count them: at
    # the limit of a 16-bit TIFF, within 2 levels of it on an 8-bit JPEG
    # export. A float image has no known limit: one band, nothing more.
    case = adversarial("hollow_band", 1000)
    s = session_on(tmp_path)
    if stored == "jpeg":
        gray8 = np.clip(np.round(case.image * 200.0 / MEMBRANE), 0.0, 255.0).astype(np.uint8)
        image = import_jpeg(s, gray8, 75)
    elif stored == "uint16":
        image = import_blot(s, case.image.astype(np.uint16))
    else:  # as imported before #224 refused it
        image = import_before_224(s, case.image.astype(np.float64))
    ops.set_lanes(s, _lane_inputs(case.n_lanes, ()))
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.flags == (("hollow_band",) if hollow else ())
    assert any(note.startswith("lane 2: a hollow band") for note in placement.notes) is hollow
    assert s.project.log[-1].params["flags"] == list(placement.flags)
    assert len([b for b in boxes_of(s, protein).bands if b.lane_index == 1]) == 1


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
    [written] = boxes_of(s, protein).undetected
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
            "unlocated_lanes": [],
            "box_size": _size_of(s, protein),
            "pitch": found.pitch,
            "noise": found.noise,
            "flags": [],
            "notes": [],
            "right_to_left": False,
            # A 16-bit TIFF: saturated at the exact limit (#121).
            "saturated_at": 0.0,
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
    # A 16-bit TIFF: saturated at the exact limit, 0 on a dark-on-light image (#121).
    assert bound.arguments["saturated_at"] == 0.0
    # The record gives detect_row's settings, and how the commit chose the
    # saturation level for each image; the log gives the level it used.
    reported = record.settings()["detect_row"]
    assert {k: v for k, v in reported.items() if k != "saturated_at"} == rowdetect.settings()
    assert "saturation_level" in reported["saturated_at"]
    assert s.project.log[-1].params["saturated_at"] == 0.0


# --- #111: a first row read with no lanes on the image to check it ---

# Two lane steps of bare membrane past the last of six bands: the reading is
# right, but the box has room for more lanes than it reads.
WIDE = adversarial_row("wide", 1000, margin_right=200, box_adjust=(0, 0, 150, 0))


def test_a_first_row_over_a_neighbouring_panel_is_placed_with_its_lanes_doubtful(tmp_path):
    # The row box also covers an arrow and a neighbouring panel's band beside
    # the row: read with no lanes on the image, the bands are numbered two
    # lanes early. The row is placed with the warning and its note, logged,
    # and one undo takes it back.
    case = adversarial("panel_beside", 1000)
    s, _, protein = row_session(tmp_path, case)
    found = detected(s, protein, case.row)
    placement = ops.detect_row_boxes(s, protein, case.row)
    assert placement.flags == ("doubtful_lanes",)
    assert placement.notes == found.notes and found.doubt_note in placement.notes
    assert sum(band is not None for band in placement.band_ids) == 5
    params = s.project.log[-1].params
    assert (params["flags"], params["notes"]) == (["doubtful_lanes"], list(found.notes))
    ops.undo(s)
    assert boxes_of(s, protein).bands == []


def test_lanes_on_the_image_that_check_the_reading_drop_the_doubt(tmp_path):
    s, image, protein = row_session(tmp_path, WIDE)
    assert detected(s, protein, WIDE.row).doubt_note is not None
    other_protein_in_lanes(s, image, WIDE, [0, 5])  # they check the reading: it lines up
    placement = ops.detect_row_boxes(s, protein, WIDE.row)
    assert (placement.flags, placement.notes) == ((), ())
    params = s.project.log[-1].params
    assert (params["flags"], params["notes"]) == ([], [])


def test_one_lane_on_the_image_checks_nothing_and_the_doubt_stays(tmp_path):
    s, image, protein = row_session(tmp_path, WIDE)
    other_protein_in_lanes(s, image, WIDE, [0])
    placement = ops.detect_row_boxes(s, protein, WIDE.row)
    assert placement.flags == ("doubtful_lanes",)
    assert detected(s, protein, WIDE.row).doubt_note in placement.notes


# Eight lanes and a ladder's band 1.8 lane steps past the last: the bands of
# lanes 3 and 4 are merged into lane 3, and the ladder's band read as lane 7.
LADDER = adversarial_row(
    "ladder8",
    1001,
    n=8,
    artefacts=[beside(1.8, 26, 12, 14000)],
    margin_right=200,
    box_adjust=(0, 0, 150, 0),
)


@pytest.mark.parametrize("dx", [0.0, 4.0, -4.0])
@pytest.mark.parametrize("lanes", [[0, 1], [0, 1, 2]])
def test_lanes_on_the_image_at_one_end_leave_the_doubt_past_them(tmp_path, lanes, dx):
    # They check the reading between them only: past them the expected x
    # steps by the row's own pitch, so the lanes squeezed a lane early there
    # are not refused. The row is placed, its lane numbers still doubtful;
    # so too with the other protein's boxes dragged 4 px either way (#180).
    s, image, protein = row_session(tmp_path, LADDER)
    found = detected(s, protein, LADDER.row)
    other_protein_in_lanes(s, image, LADDER, lanes, dx=dx)
    placement = ops.detect_row_boxes(s, protein, LADDER.row)
    step = float(np.median(np.diff(LADDER.lane_cx)))
    off = [
        lane
        for lane, rect in rects_by_lane(s, protein).items()
        if abs((rect[0] + rect[2]) / 2 - LADDER.lane_cx[lane]) > step / 2
    ]
    assert off == [4, 5, 6, 7]  # what the doubt is there for
    assert "doubtful_lanes" in placement.flags, (
        f"lanes 5 to 8 boxed a lane off, placed with flags {placement.flags}"
    )
    assert found.doubt_note is not None and found.doubt_note in placement.notes
    params = s.project.log[-1].params
    assert (params["flags"], params["notes"]) == (list(placement.flags), list(found.notes))


def test_lanes_on_the_image_refuse_a_row_read_off_whatever_its_doubt(tmp_path):
    # The row over the neighbouring panel again, now with another protein's
    # boxes in the lanes that hold bands: it is refused, not placed doubtful.
    case = adversarial("panel_beside", 1000)
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case, [2, 3, 4])
    with pytest.raises(OperationError) as refused:
        ops.detect_row_boxes(s, protein, case.row)
    assert refused.value.code is ErrorCode.ROW_LANES_UNCLEAR
    assert refused.value.detail["cause"] == "off_lanes"


# --- #180: lanes on the image at one end of the row ---


def flipped(case: RowCase) -> RowCase:
    """``case`` flipped left to right: the image, the row box and the
    references mirrored, the lanes renumbered so lane 0 is again the
    leftmost (the mirror of the original last lane). Numbering them right to
    left, as a mirrored blot is read, is :func:`anchored_row`'s
    ``mirrored=True``."""
    width = case.image.shape[1]
    x0, y0, x1, y1 = case.row
    n = case.n_lanes
    return dataclasses.replace(
        case,
        image=case.image[:, ::-1].copy(),
        row=(width - x1, y0, width - x0, y1),
        reference={
            n - 1 - k: (width - r[2], r[1], width - r[0], r[3]) for k, r in case.reference.items()
        },
        lane_cx=tuple(width - 1 - x for x in reversed(case.lane_cx)),
        lane_cy=tuple(reversed(case.lane_cy)),
    )


class Placed(NamedTuple):
    """What the operation did with a row (:func:`anchored_row`): ``kind``
    ``"refused"`` (``flags`` the error code and cause) or ``"placed"``
    (``flags`` the placement's); ``offsets`` maps each stored lane to its box
    centre's x minus the lane's true centre (px); ``half_step`` is half the
    median step between the lanes' centres; ``empty`` the stored lanes
    without a band."""

    kind: str
    flags: tuple[str, ...]
    offsets: dict[int, float]
    half_step: float
    empty: frozenset[int]

    @property
    def off(self) -> list[int]:
        """The stored lanes whose box lies more than half a step off."""
        return [lane for lane, dx in sorted(self.offsets.items()) if abs(dx) > self.half_step]

    def off_in_words(self) -> str:
        """The lanes boxed more than half a step off, each with its offset."""
        lanes = "; ".join(
            f"lane {lane + 1}: no band, boxed {self.offsets[lane]:+.1f} px from the lane's centre"
            if lane in self.empty
            else f"lane {lane + 1}: box centre {self.offsets[lane]:+.1f} px from the band's"
            for lane in self.off
        )
        return (
            f"placed with flags {self.flags}: {lanes} (half the lane step {self.half_step:.1f} px)"
        )


def anchored_row(
    tmp_path: Path,
    case: RowCase,
    lanes: Sequence[int] | None,
    *,
    dx: float = 0.0,
    mirrored: bool = False,
) -> Placed:
    """The row placed by the operation with another protein's boxes in
    ``lanes`` (none if None; numbered right to left when ``mirrored``)."""
    s, image, protein = row_session(tmp_path, case)
    if lanes is not None:
        other_protein_in_lanes(s, image, case, lanes, dx=dx, mirrored=mirrored)
    n = case.n_lanes
    half_step = float(np.median(np.abs(np.diff(case.lane_cx)))) / 2

    def drawn(lane: int) -> int:
        return n - 1 - lane if mirrored else lane

    empty = frozenset(lane for lane in range(n) if drawn(lane) not in case.reference)
    try:
        placement = ops.detect_row_boxes(s, protein, case.row)
    except OperationError as exc:
        cause = (exc.code.name, exc.detail.get("cause", ""))
        return Placed("refused", cause, {}, half_step, empty)
    offsets = {
        lane: (rect[0] + rect[2]) / 2 - case.lane_cx[drawn(lane)]
        for lane, rect in rects_by_lane(s, protein).items()
    }
    return Placed("placed", placement.flags, offsets, half_step, empty)


def read_with_care(outcome: Placed) -> bool:
    """Refused, placed with its lane numbers doubtful, or placed right: never
    placed a lane off without a flag."""
    return outcome.kind == "refused" or "doubtful_lanes" in outcome.flags or not outcome.off


@pytest.mark.parametrize("dx", [0.0, 4.0, -4.0])
@pytest.mark.parametrize("lanes", [[6, 7], [5, 6, 7], [0, 7]])
def test_lanes_on_the_image_at_the_far_end_or_both_ends_leave_no_lane_off_silently(
    tmp_path, lanes, dx
):
    # The ladder's band past lane 7 squeezes the bands a lane early, which
    # the lanes at the far end, or at both ends, show: the row is refused or
    # placed with its lane numbers doubtful, never a lane off unflagged.
    outcome = anchored_row(tmp_path, LADDER, lanes, dx=dx)
    assert read_with_care(outcome), outcome.off_in_words()


@pytest.mark.parametrize(("lanes", "kind"), [([0, 1], "refused"), ([6, 7], "placed")])
def test_lanes_on_a_flipped_image_are_read_from_its_own_end(tmp_path, lanes, kind):
    # The same row flipped, its lanes numbered right to left: lanes 0 and 1
    # now lie at the ladder's end and refuse the row; lanes 6 and 7, at the
    # other end, leave it placed with its numbers doubtful.
    outcome = anchored_row(tmp_path, flipped(LADDER), lanes, mirrored=True)
    assert read_with_care(outcome), outcome
    assert outcome[0] == kind


@pytest.mark.parametrize("end", ["first", "last"])
@pytest.mark.parametrize("seed", [1000, 1001, 1002])
@pytest.mark.parametrize("key", ["ladder_beside", "label_beside", "panel_beside"])
def test_lanes_on_the_image_at_either_end_refuse_a_row_over_what_lies_beside(
    tmp_path, key, seed, end
):
    case = adversarial(key, seed)
    n = case.n_lanes
    outcome = anchored_row(tmp_path, case, [0, 1] if end == "first" else [n - 2, n - 1])
    assert outcome[:2] == ("refused", ("ROW_LANES_UNCLEAR", "off_lanes")), outcome


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#111 an arrow past the last lane, the first lane empty, shifts the lane numbers"
    " without a flag",
)
@pytest.mark.parametrize("seed", [1000, 1001])
@pytest.mark.parametrize("key", ["arrow_0.9", "arrow_1.3"])
def test_a_row_box_over_an_arrow_past_the_last_lane_is_not_read_off_silently(tmp_path, key, seed):
    outcome = anchored_row(tmp_path, stress(key, seed), None)
    assert read_with_care(outcome), outcome.off_in_words()


@pytest.mark.parametrize("end", ["first", "last"])
@pytest.mark.parametrize("seed", [1000, 1001, 1002])
@pytest.mark.parametrize("key", ["arrow_0.9", "arrow_1", "arrow_1.1", "arrow_1.3"])
def test_lanes_on_the_image_refuse_a_row_box_over_an_arrow(tmp_path, key, seed, end):
    case = stress(key, seed)
    lanes = [1, 2] if end == "first" else [case.n_lanes - 2, case.n_lanes - 1]
    outcome = anchored_row(tmp_path, case, lanes)
    assert outcome[:2] == ("refused", ("ROW_LANES_UNCLEAR", "off_lanes")), outcome


# --- #180: a stain on one side of a band's background ring ---

STAIN_LANE = 2  # the lane the ring stains lie beside


_RING_NETS: dict[tuple, tuple[float, tuple[str, ...], dict]] = {}


def cropped(case: RowCase, top: int) -> RowCase:
    """``case`` with its image's first ``top`` rows cut off, the row box's top
    edge at the image's top at most."""
    x0, y0, x1, y1 = case.row
    return dataclasses.replace(
        case,
        image=case.image[top:].copy(),
        row=(x0, max(0, y0 - top), x1, y1 - top),
        reference={k: (r[0], r[1] - top, r[2], r[3] - top) for k, r in case.reference.items()},
        lane_cy=tuple(y - top for y in case.lane_cy),
    )


def _lane_net(
    tmp_path_factory, key: str | None, seed: int, depth: tuple | None, top: int = 0
) -> tuple[float, tuple[str, ...], dict]:
    """Lane 2's net once the row is placed by the operation and its results
    computed, the background notices on lane 2, and the row's boxes: ``key``
    a :data:`rowcases.STRESS` ring-stain recipe, None for the same row (same
    draws) without the stain; ``depth`` the bands' depth range (the
    recipe's); ``top`` image rows cut off above the row. Kept for the tests
    that share a row (numbers only)."""
    if (key, seed, depth, top) not in _RING_NETS:
        recipe = {} if key is None else dict(STRESS[key])
        if depth is not None:
            recipe["depth_range"] = depth
        case = cropped(adversarial_row("ring", seed, **recipe), top)
        s, _, protein = row_session(tmp_path_factory.mktemp("ring"), case)
        ops.detect_row_boxes(s, protein, case.row)
        notices = sorted(
            notice.code.name
            for notice in results.compute_results(s.project.batch).notices
            if notice.code in (NoticeCode.BACKGROUND_UNEVEN, NoticeCode.BACKGROUND_FALLBACK)
            and STAIN_LANE in notice.lane_indices
        )
        net = lane_bands(s, protein)[STAIN_LANE].net
        _RING_NETS[key, seed, depth, top] = (net, tuple(notices), rects_by_lane(s, protein))
    return _RING_NETS[key, seed, depth, top]


def stained_net(
    tmp_path_factory, key: str, seed: int, top: int = 0
) -> tuple[float, tuple[str, ...], bool]:
    """Lane 2's net with the stain over its net without it, minus 1; the
    notices on lane 2 with the stain; whether every box stayed where it was
    (both images cut ``top`` rows short)."""
    depth = STRESS[key].get("depth_range")
    net, notices, rects = _lane_net(tmp_path_factory, key, seed, depth, top)
    clean, _, clean_rects = _lane_net(tmp_path_factory, None, seed, depth, top)
    return net / clean - 1.0, notices, rects == clean_rects


@pytest.mark.parametrize("seed", range(1000, 1005))
def test_a_faint_stain_on_a_weak_bands_ring_moves_no_box(tmp_path_factory, seed):
    assert stained_net(tmp_path_factory, "faint_ring_stain", seed)[2]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#177 a faint stain on one side of a weak band's ring lowers its net without a notice",
)
@pytest.mark.parametrize("seed", range(1000, 1005))
def test_a_faint_stain_on_a_weak_bands_ring_moves_its_net_little_or_is_noticed(
    tmp_path_factory, seed
):
    # Bands 3000 deep, a stain one noise sigma deep over the ring's upper
    # side: the net must stay within 5% of the unstained row's, or the lane
    # must show a background notice.
    bias, notices, _ = stained_net(tmp_path_factory, "faint_ring_stain", seed)
    assert abs(bias) <= 0.05 or notices, f"lane 3: net {bias:+.1%} with the stain, no notice"


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
@pytest.mark.parametrize("key", ["ring_stain", "ring_stain_x4"])
def test_a_faint_stain_on_an_ordinary_bands_ring_moves_its_net_little(tmp_path_factory, key, seed):
    bias, notices, same = stained_net(tmp_path_factory, key, seed)
    assert same
    assert abs(bias) <= 0.05 or notices, f"lane 3: net {bias:+.1%} with the stain, no notice"


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_a_deep_stain_on_a_ring_is_clipped_out_and_noticed(tmp_path_factory, seed):
    # Ten noise sigmas deep: the ring's clip drops it (the net within 1%),
    # and the spread it leaves raises the uneven-background notice.
    bias, notices, same = stained_net(tmp_path_factory, "deep_ring_stain", seed)
    assert same
    assert abs(bias) <= 0.01, f"lane 3: net {bias:+.2%} with the stain"
    assert "BACKGROUND_UNEVEN" in notices, f"lane 3: no uneven-background notice ({notices})"


@pytest.mark.parametrize("seed", range(1000, 1005))
def test_a_stain_on_the_only_side_left_of_a_ring_is_noticed(tmp_path_factory, seed):
    # The image cropped 2 px above the boxes the operation places on the
    # unstained row: the ring keeps its lower side only, where a stain ten
    # noise sigmas deep lies. The net may move a lot (here -28% to -43%); the
    # lane must then show a background notice.
    boxes_uncropped = _lane_net(tmp_path_factory, None, seed, None)[2]
    top = min(rect[1] for rect in boxes_uncropped.values()) - 2
    boxes = _lane_net(tmp_path_factory, None, seed, None, top)[2]
    assert min(rect[1] for rect in boxes.values()) == 2
    bias, notices, same = stained_net(tmp_path_factory, "stain_below", seed, top)
    assert same
    assert abs(bias) <= 0.05 or notices, f"lane 3: net {bias:+.1%} with the stain, no notice"


# --- #181: the operations degraded on purpose fail the recipes ---
# Each runs one recipe test above under a degraded operation and expects its
# named assertion to fail: the recipes would catch the regression.


def test_a_ring_never_clipped_fails_the_deep_stain_recipe(tmp_path_factory, monkeypatch):
    # The stain ten noise sigmas deep enters the ring's median: the net moves
    # by about 2%. The nets are taken under the degraded ring, never kept.
    monkeypatch.setattr(sys.modules[__name__], "_RING_NETS", {})
    monkeypatch.setattr(quantify, "RING_CLIP_K", math.inf)
    with pytest.raises(AssertionError, match=r"lane 3: net [-+]\d+\.\d\d% with the stain"):
        test_a_deep_stain_on_a_ring_is_clipped_out_and_noticed(tmp_path_factory, 1000)


def test_no_uneven_background_notice_fails_the_deep_stain_recipe(tmp_path_factory, monkeypatch):
    # The stain is clipped out (the net within 1%) but nothing says the ring
    # was uneven.
    monkeypatch.setattr(sys.modules[__name__], "_RING_NETS", {})
    monkeypatch.setattr(results, "BACKGROUND_UNEVEN_LIMIT", math.inf)
    with pytest.raises(AssertionError, match=r"lane 3: no uneven-background notice \(\(\)\)"):
        test_a_deep_stain_on_a_ring_is_clipped_out_and_noticed(tmp_path_factory, 1000)


def test_lanes_on_the_image_taken_to_check_every_lane_fail_the_one_end_recipe(
    tmp_path, monkeypatch
):
    # Lanes 1 and 2 on the image taken to check the lanes past them too (the
    # gap #155's review closed): the doubt is dropped, lanes 5 to 8 a lane off.
    def every_lane(anchors, lanes):
        return set(lanes) if anchors else set()

    monkeypatch.setattr(ops, "anchoring_lanes", every_lane)
    with pytest.raises(AssertionError, match="lanes 5 to 8 boxed a lane off, placed with flags"):
        test_lanes_on_the_image_at_one_end_leave_the_doubt_past_them(tmp_path, [0, 1], 0.0)


def test_lanes_on_the_image_that_never_refuse_fail_the_far_end_recipe(tmp_path, monkeypatch):
    # No band ever found off its lane: lanes 1 and 8 on the image place the
    # squeezed row with no flag on its lane numbers.
    monkeypatch.setattr(ops, "_off_lanes", lambda centres, expected: [])
    with pytest.raises(AssertionError, match=r"placed with flags \(.*\): lane 5: box centre"):
        test_lanes_on_the_image_at_the_far_end_or_both_ends_leave_no_lane_off_silently(
            tmp_path, [0, 7], 0.0
        )


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
    assert info.value.ids == ()  # the row box alone is at fault


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


def test_a_row_off_the_lanes_on_its_image_names_the_boxes_that_placed_them(tmp_path):
    # One of them may be in the wrong lane (a click given the wrong lane), so a
    # client can offer to take back the change that placed it. This protein's
    # boxes a detector placed give way to the row and are not named; its box
    # clicked in lane 4, which shows where the user put that lane, is.
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    ops.detect_row_boxes(s, protein, case.row)
    ops.remove_box(s, lane_bands(s, protein)[4].id)
    clicked = ops.place_box(s, protein, *at_lane(case, 4), lane_index=4, grow=False)
    other = other_protein_in_lanes(s, image, case, (3, 4))
    before = s.project
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, protein, shifted(case, 70))
    assert (info.value.code, str(info.value)) == (ErrorCode.ROW_LANES_UNCLEAR, OFF_LANES)
    others = tuple(b.id for b in boxes_of(s, other).bands)
    assert len(others) == 2
    assert info.value.ids == (clicked, *others)  # proteins in order, then their boxes
    assert s.project is before


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
    assert [u.lane_index for u in boxes_of(s, protein).undetected] == sorted(empty)
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
    # The ids of the same anchors, in the same order.
    lanes = {b.id: b.lane_index for p in batch.proteins for b in batch.boxes_of(p).bands}
    for kwargs in ({}, {"only": own}, {"without": own}):
        ids = lane_anchor_ids(batch, ref, **kwargs)
        assert [lanes[i] for i in ids] == [lane for _, lane in lane_anchors(batch, ref, **kwargs)]
    assert set(lane_anchor_ids(batch, ref, only=own)) == own


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
    actin = other_protein_in_lanes(
        s, image, case, (3, 4), mirrored=not mirrored, name="actin", y=36
    )
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


@pytest.mark.parametrize("mirrored", [False, True], ids=["clicked-ltr", "clicked-rtl"])
def test_clicks_padded_wider_than_the_lane_pitch_count_among_the_lanes_numbered(
    tmp_path, monkeypatch, mirrored
):
    # Another protein's boxes clicked in true lanes 0, 2 and 4, each grown over
    # its own band (41 px, under the pitch of 70), then padded 16 px left and
    # right (#57): 73 px wide, but centred on their lanes as before. What grew
    # from the click is the fitted size, so they number their lanes, the other
    # way from a third protein's boxes in the other lanes.
    case = ROWS["all_present"]
    n = case.n_lanes
    s, image, protein = row_session(tmp_path, case)
    actin = ops.add_protein(s, "actin", Role.TARGET, image)
    for true in (0, 2, 4):
        lane = n - 1 - true if mirrored else true
        ops.place_box(s, actin, *at_lane(case, true), lane_index=lane, grow=True)
    ops.set_box_padding(s, actin, across=16)
    grown = boxes_of(s, actin)
    assert grown.fitted_size.width < 69 and grown.box_size.width > 71
    gapdh = other_protein_in_lanes(s, image, case, (1, 3, 5), mirrored=not mirrored)
    names = ("'GAPDH'", "'actin'") if mirrored else ("'actin'", "'GAPDH'")
    ids = numbering_refused(s, protein, case.row, both_ways(*names), monkeypatch)
    assert ids == ((gapdh, actin) if mirrored else (actin, gapdh))


def test_a_mirrored_image_numbered_one_way_is_read_right_to_left(tmp_path):
    # Two proteins' boxes and the protein's own clicked ones all number the
    # lanes right to left.
    case = ROWS["all_present"]
    n = case.n_lanes
    s, image, protein, clicked = clicked_lanes(tmp_path, mirrored=True)
    other_protein_in_lanes(s, image, case, mirrored=True)
    other_protein_in_lanes(s, image, case, (0, 2, 5), mirrored=True, name="actin", y=36)

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
    assert boxes_of(s, protein).box_size.width > 48
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


# Kept lanes 0, 2, 4 and 7, at 60 px per lane.
NAMING = [(30.0, 0), (150.0, 2), (270.0, 4), (450.0, 7)]


@pytest.mark.parametrize(
    ("off", "named"),
    [
        ([4], {2, 4, 7}),  # its own box, and those of lanes 3 and 5 beside it
        ([2], {0, 2, 4}),
        ([3], {2, 4}),  # between 2 and 4, as are its neighbours
        ([8], {7}),  # past the last kept lane
        ([1, 5], {0, 2, 4, 7}),
        ([], set()),
    ],
)
def test_an_off_lane_names_the_boxes_its_yardstick_comes_from(off, named):
    # _off_lanes measures a lane against its own expected x and a
    # neighbour's: a wrong box behind either can make it off.
    assert ops._named_lanes(NAMING, off) == named


# --- #117: a refused row says why, by what the detector saw ---

CENTRE_ROW = round(float(np.mean(SCENE.lane_cy)))  # the row through SCENE's band centres
CUT_UNCLEAR_ROW = (AMBIGUOUS_ROW[0], CENTRE_ROW - 2, *AMBIGUOUS_ROW[2:])
EDGE_ROW = (SCENE.row[0], CENTRE_ROW, *SCENE.row[2:])  # top edge through the band centres
BLANK_ROW = (SCENE.row[0], 5, SCENE.row[2], 40)  # membrane above the bands

UNCLEAR = (
    "the row box does not show which lane each band is in; draw it over every"
    " declared lane, empty end lanes included, or place the boxes by clicking"
)
CUT_UNCLEAR = (
    "the row box cuts through the bands, so it does not show which lane each band is in;"
    " draw it over the whole band height and every declared lane, empty end lanes included"
)
UNFIT = (
    "bands were found in the row box but do not fit the lanes; draw it over every"
    " declared lane, empty end lanes included, or place the boxes by clicking"
)
AT_THE_EDGE = (
    "the only signal in the row box lies at its top or bottom edge: the box cuts through"
    " the bands or reaches into a neighbouring row; include the whole band height"
)
FILLS_THE_BOX = (
    "the only signal in the row box runs through its whole height: a streak or stain,"
    " or bands the box cuts through; include the whole band height"
)
TOO_LITTLE_MEMBRANE = (
    "the row box holds too little membrane to measure the bands against; include some"
    " membrane above and below the bands"
)
NO_BAND = "no band found in the row box"
AT_THE_SIDE = (
    "the only signal in the row box rises into its left or right edge: the box cuts through a"
    " band there, or the image's edge is dark; draw it over whole bands"
)
SIDE_UNCLEAR = (
    "the row box's left or right edge cuts through a band, so it does not show which lane each"
    " band is in; draw it over the whole bands of every declared lane, empty end lanes included"
)
ACROSS = (
    "the only signal in the row box runs across the lanes as a line or strip, not as bands: a"
    " frame line, or a dark strip along the image's edge; draw the box over the bands only"
)

# Six touching, saturated bands filling a snug box: the detector takes their
# level for the membrane and finds nothing.
THICK = adversarial_row(
    "thick", 1000, h=30.0, pitch=48.0, w=60.0, depth_range=(75000.0, 80000.0), my=0
)
# A box just inside the 20% extents of six saturated bands: no membrane above
# or below them, and the detector's noise estimate takes them in.
SNUG = adversarial_row("snug", 1000, w=56.0, h=12.0, depth_range=(75000.0, 80000.0), my=-1)
STREAK = adversarial_row(
    "streak", 1000, missing=range(6), artefacts=[vstreak(2, 14, 3000)]
)  # nothing but a streak down lane 2
# No band, on a darker stretch of membrane: 3000 (7.5 noise sigmas) to the band
# side of the stored background, the image's median.
DARK_STRIPE = adversarial_row(
    "dark stripe", 1000, missing=range(6), artefacts=[hstripe(30.0, 3000.0)]
)
# A stray band under the middle of the row, and a box whose top edge runs
# through the bands' centres: the stray band is the only one kept, midway
# between two lanes (an ambiguous reading), and every lane's band peaks on
# the box's top row, so none of them reaches the edge as a kept extent.
STRAY = adversarial_row("stray", 1001, artefacts=[band_between(2, 10.0, 6.0, 20000.0, dy=12.0)])
STRAY_ROW = (STRAY.row[0], round(float(np.mean(STRAY.lane_cy))), STRAY.row[2], STRAY.row[3] + 4)
# #116: a box over a screenshot's toolbar along the image's bottom, and boxes
# whose right edge runs 8 px past the last lane's band centre, over that one
# band alone and over the whole row.
STRIP = adversarial_row(
    "strip",
    1000,
    missing=range(6),
    artefacts=[bottom_strip(14, 20000.0)],
    box_adjust=(0, 52, 0, 80),
)
SIDE_ONLY = adversarial_row("side only", 1000, missing=range(5))
SIDE_ROW = (*SIDE_ONLY.row[:2], math.floor(SIDE_ONLY.lane_cx[-1]) - 8, SIDE_ONLY.row[3])


def row_refused(s: ProjectSession, protein_id: str, row) -> OperationError:
    """The row's refusal, which changes nothing and carries the detector's raw
    reason as JSON-plain detail."""
    before = s.project
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, protein_id, row)
    assert s.project is before
    detail = info.value.detail
    assert detail is not None
    assert json.loads(json.dumps(detail, allow_nan=False)) == detail
    return info.value


def assert_raw_reason(detail: dict, found: RowDetection) -> None:
    """The refusal's detail holds what the detector saw, lane by lane."""
    assert detail["flags"] == list(found.flags)
    assert detail["notes"] == list(found.notes)
    assert detail["margin"] == (
        None if found.margin is None or not math.isfinite(found.margin) else found.margin
    )
    assert detail["membrane_shift"] == found.membrane_shift
    assert detail["lanes"] == [
        {
            "lane_index": lane.lane,
            "reason": lane.reason,
            "snr": lane.snr,
            "cut": lane.cut,
            "line_offset": lane.line_offset,
        }
        for lane in found.lanes
    ]


@pytest.mark.parametrize(
    ("case", "row", "code", "message", "cause"),
    [
        pytest.param(
            SCENE,
            UNCLEAR_ROW,
            ErrorCode.ROW_LANES_UNCLEAR,
            UNCLEAR,
            "lanes_outside_row",
            id="lanes-outside-row",
        ),
        pytest.param(
            SCENE,
            AMBIGUOUS_ROW,
            ErrorCode.ROW_LANES_UNCLEAR,
            UNCLEAR,
            "ambiguous_lanes",
            id="ambiguous",
        ),
        # A box a lane short whose top edge also runs through the bands: the
        # cut comes first, whatever else is unclear.
        pytest.param(
            SCENE,
            CUT_UNCLEAR_ROW,
            ErrorCode.ROW_LANES_UNCLEAR,
            CUT_UNCLEAR,
            "cut_by_row_box",
            id="cut-and-unclear",
        ),
        # The bands the box cuts through peak on its edge row and are not kept;
        # a lone stray band leaves the reading unclear. Still the cut first.
        pytest.param(
            STRAY,
            STRAY_ROW,
            ErrorCode.ROW_LANES_UNCLEAR,
            CUT_UNCLEAR,
            "cut_by_row_box",
            id="cut-bands-not-kept",
        ),
        # Every band peaks on the box's top row: none is kept, and not "no band".
        pytest.param(
            SCENE, EDGE_ROW, ErrorCode.NO_BAND_FOUND, AT_THE_EDGE, "edge_signal", id="at-the-edge"
        ),
        pytest.param(
            STREAK, None, ErrorCode.NO_BAND_FOUND, FILLS_THE_BOX, "artefact", id="streak-only"
        ),
        pytest.param(
            THICK,
            None,
            ErrorCode.NO_BAND_FOUND,
            TOO_LITTLE_MEMBRANE,
            "too_little_membrane",
            id="filled-by-bands",
        ),
        pytest.param(
            SNUG,
            None,
            ErrorCode.NO_BAND_FOUND,
            TOO_LITTLE_MEMBRANE,
            "too_little_membrane",
            id="snug",
        ),
        # Membrane darker than the stored background is still membrane.
        pytest.param(
            DARK_STRIPE, None, ErrorCode.NO_BAND_FOUND, NO_BAND, "no_band", id="darker-membrane"
        ),
        # The last lane's band rises into the box's right edge; the other
        # lanes are empty, or hold bands that leave that lane outside the box.
        pytest.param(
            SIDE_ONLY,
            SIDE_ROW,
            ErrorCode.NO_BAND_FOUND,
            AT_THE_SIDE,
            "side_signal",
            id="at-the-side",
        ),
        pytest.param(
            SCENE,
            (*SCENE.row[:2], math.floor(SCENE.lane_cx[-1]) - 8, SCENE.row[3]),
            ErrorCode.ROW_LANES_UNCLEAR,
            SIDE_UNCLEAR,
            "side_signal",
            id="side-and-unclear",
        ),
        pytest.param(STRIP, None, ErrorCode.NO_BAND_FOUND, ACROSS, "line", id="strip-only"),
        pytest.param(SCENE, BLANK_ROW, ErrorCode.NO_BAND_FOUND, NO_BAND, "no_band", id="blank"),
    ],
)
def test_a_refused_row_is_worded_by_its_cause(tmp_path, case, row, code, message, cause):
    row = case.row if row is None else row
    s, _, protein = row_session(tmp_path, case)
    found = detected(s, protein, row)
    error = row_refused(s, protein, row)
    assert (error.code, str(error), error.ids) == (code, message, ())
    assert error.detail["cause"] == cause
    assert_raw_reason(error.detail, found)


@pytest.mark.parametrize(("case", "row"), [(SCENE, CUT_UNCLEAR_ROW), (STRAY, STRAY_ROW)])
def test_a_cut_row_that_is_refused_names_the_cut_lanes_in_its_detail(tmp_path, case, row):
    s, _, protein = row_session(tmp_path, case)
    found = detected(s, protein, row)
    assert found.refused and any(lane.cut for lane in found.lanes)
    error = row_refused(s, protein, row)
    assert "cut_by_row_box" in error.detail["flags"]
    assert [lane["cut"] for lane in error.detail["lanes"]] == [lane.cut for lane in found.lanes]


def test_bands_that_fit_no_lane_are_not_no_band(tmp_path, monkeypatch):
    # Kept bands the lane reading could not place (every lane "unassigned",
    # nothing sized): the words say they were found.
    s, _, protein = row_session(tmp_path, SCENE)
    blank = detected(s, protein, BLANK_ROW)
    unfit = dataclasses.replace(
        blank,
        lanes=tuple(
            dataclasses.replace(lane, reason="unassigned", snr=40.0 + lane.lane)
            for lane in blank.lanes
        ),
    )
    monkeypatch.setattr(rowdetect, "detect_row", lambda *args, **kwargs: unfit)
    error = row_refused(s, protein, BLANK_ROW)
    assert (error.code, str(error)) == (ErrorCode.NO_BAND_FOUND, UNFIT)
    assert error.detail["cause"] == "unassigned"
    assert_raw_reason(error.detail, unfit)


def test_crowded_lanes_beside_signal_at_the_side_are_worded_as_crowded(tmp_path, monkeypatch):
    # A row over every lane refused because two lanes' extents crowd each
    # other, whose empty first lane holds only signal rising into the box's
    # left edge (a dark image edge): that edge leaves no lane out, so the
    # words are the crowded lanes', not a side edge cutting a band.
    s, _, protein = row_session(tmp_path, SCENE)
    measure = rowdetect._measure

    def crowd(lanes, *args):
        measure(lanes, *args)
        a, b = lanes[2].rect, lanes[3].rect
        x = (a[0] + a[2]) // 2 - (b[2] - b[0]) // 2
        lanes[3].rect = (x, b[1], x + b[2] - b[0], b[3])

    monkeypatch.setattr(rowdetect, "_measure", crowd)
    found = detected(s, protein, SCENE.row)
    assert found.flags == ("ambiguous_lanes",) and found.lanes[0].reason == "no_band"
    assert any("extents closer than the narrowest band" in note for note in found.notes)
    first = dataclasses.replace(found.lanes[0], reason="side_signal")
    edged = dataclasses.replace(found, lanes=(first, *found.lanes[1:]))
    monkeypatch.setattr(rowdetect, "detect_row", lambda *args, **kwargs: edged)
    error = row_refused(s, protein, SCENE.row)
    assert (error.code, str(error)) == (ErrorCode.ROW_LANES_UNCLEAR, UNCLEAR)
    assert error.detail["cause"] == "ambiguous_lanes"
    assert_raw_reason(error.detail, edged)


def test_a_refusal_before_detection_carries_no_detail(tmp_path):
    s, _, protein = row_session(tmp_path, SCENE)
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, protein, (SCENE_W + 10, 0, SCENE_W + 50, 20))
    assert (info.value.code, info.value.detail) == (ErrorCode.OUT_OF_IMAGE, None)


def test_a_row_off_the_lanes_names_only_the_boxes_next_to_the_off_bands(tmp_path):
    # Another protein's boxes in lanes 0 and 1, and the target's box put on
    # lane 4's band but given lane 5: lanes 2 to 4, interpolated between the
    # boxes of lanes 1 and 5, are squeezed, so the bands of lanes 3, 4 and 5
    # lie nearer a neighbouring lane than their own. The refusal names the
    # boxes those lanes are read from (lanes 1 and 5), not lane 0's.
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    ops.set_box_size(s, protein, BoxSize(width=40, height=12))
    clicked = ops.place_box(s, protein, *at_lane(case, 4), lane_index=5, grow=False)
    other = other_protein_in_lanes(s, image, case, (0, 1))
    error = row_refused(s, protein, case.row)
    assert (error.code, str(error)) == (ErrorCode.ROW_LANES_UNCLEAR, OFF_LANES)
    assert error.ids == (clicked, lane_bands(s, other)[1].id)
    assert error.detail == {"cause": "off_lanes", "off_lanes": [3, 4, 5]}


# --- #124: a row says which other proteins' nets it changed ---


def test_a_row_reports_the_nets_it_changed_of_other_proteins_on_its_image(tmp_path):
    # GAPDH's boxes over the lower part of two of the target's bands: every
    # ring on the image leaves out the row's boxes, so GAPDH's nets change with
    # the row.
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    other = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    for lane in (1, 2):
        x, y = at_lane(case, lane)
        ops.place_box(s, other, x, y + 8, lane_index=lane, grow=False)
    before = {b.id: b.net for b in boxes_of(s, other).bands}

    placement = ops.detect_row_boxes(s, protein, case.row)
    after = {b.id: b.net for b in boxes_of(s, other).bands}
    changed = tuple((i, before[i], after[i]) for i in before if after[i] != before[i])
    assert changed and placement.remeasured == changed
    band_id, change = placement.largest_change
    assert change == max(abs(a - b) / b for _, b, a in changed)
    assert change == abs(after[band_id] - before[band_id]) / before[band_id]

    # The same drag again changes nothing, and says so.
    again = ops.detect_row_boxes(s, protein, case.row)
    assert (again.remeasured, again.largest_change) == ((), None)


def test_a_row_alone_on_its_image_remeasures_no_other_protein(tmp_path):
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    other_protein_in_lanes(s, image, case, (0, 1))  # in a row of its own, far above
    placement = ops.detect_row_boxes(s, protein, case.row)
    assert (placement.remeasured, placement.largest_change) == ((), None)


def test_a_net_that_was_zero_is_remeasured_but_sets_no_largest_change(tmp_path):
    # A net measured from 0 changes by no share of it: listed, but no largest
    # change. (GAPDH's net planted at 0; the row measures it again.)
    case = ROWS["all_present"]
    s, image, protein = row_session(tmp_path, case)
    other = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    x, y = at_lane(case, 1)
    band_id = ops.place_box(s, other, x, y + 8, lane_index=1, grow=False)
    plant(s, lambda draft: setattr(draft.batch.find_band(band_id)[1], "net", 0.0))

    placement = ops.detect_row_boxes(s, protein, case.row)
    net = band_of(s, band_id).net
    assert net > 0
    assert (placement.remeasured, placement.largest_change) == (((band_id, 0.0, net),), None)


# --- #114: a row off its line, and a box over another protein's ---

# Two rows 40 px apart and the row box over both: the row above (a loading
# control's) is twice as deep as the row's own (a target's) in lanes 0-2 and
# half as deep in lanes 3-5, so each lane's strongest band lies in the row
# above there and in the row's own here.
TWO_ROWS = adversarial_row(
    "two rows",
    1000,
    neighbour_dy=-40.0,
    neighbour_rel=0.5,
    neighbour_rels={0: 2.0, 1: 2.0, 2: 2.0},
    box_adjust=(0, -40, 0, 0),
    img_h=200,
)
ABOVE_ROW = (*TWO_ROWS.row[:3], TWO_ROWS.row[3] - 40)  # over the row above alone
OWN_ROW = (TWO_ROWS.row[0], TWO_ROWS.row[1] + 40, *TWO_ROWS.row[2:])  # over the row's own alone
# Lanes 4 and 5 alone hold their strongest band in the row above.
TWO_LANES_ABOVE = adversarial_row(
    "two rows",
    1000,
    neighbour_dy=-40.0,
    neighbour_rel=0.5,
    neighbour_rels={4: 2.0, 5: 2.0},
    box_adjust=(0, -40, 0, 0),
    img_h=200,
)
# A montage: lane 2's band 14 px (1.2 box heights) above the others'.
MONTAGE = adversarial_row("montage", 1000, shifts={2: -14.0})
OFF_LINE = (
    "{which} above or below the row's line through the other bands, by more than 0.75 of a"
    " box's height: the row box covers more than one row, or {those} off the row; draw it over"
    " one row only{fix}"
)
OFF_LINE_TWO = OFF_LINE.format(
    which="the bands found in lanes 5, 6 lie",
    those="those bands lie",
    fix=", or box those lanes by clicking their bands",
)
OFF_LINE_ONE = OFF_LINE.format(
    which="the band found in lane 3 lies",
    those="that band lies",
    fix=", or box that lane by clicking its band",
)


@pytest.mark.parametrize(
    ("case", "message", "off"),
    [
        pytest.param(TWO_LANES_ABOVE, OFF_LINE_TWO, [4, 5], id="over-two-rows"),
        pytest.param(MONTAGE, OFF_LINE_ONE, [2], id="a-lane-off-the-row"),
    ],
)
def test_a_row_whose_boxes_are_off_its_line_is_refused_naming_the_lanes(
    tmp_path, case, message, off
):
    s, _, protein = row_session(tmp_path, case)
    found = detected(s, protein, case.row)
    assert found.flags[0] == "off_row_line"
    error = row_refused(s, protein, case.row)
    assert (error.code, str(error), error.ids) == (ErrorCode.ROW_OFF_LINE, message, ())
    assert error.detail["cause"] == "off_row_line"
    assert_raw_reason(error.detail, found)
    offsets = [lane["line_offset"] for lane in error.detail["lanes"]]
    assert [i for i, v in enumerate(offsets) if abs(v) > rowdetect.ROW_LINE_K] == off


def test_a_row_off_its_line_whose_lanes_are_unclear_names_none(tmp_path, monkeypatch):
    # With the reading unsettled too, the lanes are not named.
    s, _, protein = row_session(tmp_path, MONTAGE)
    found = detected(s, protein, MONTAGE.row)
    unclear = dataclasses.replace(found, flags=("ambiguous_lanes", *found.flags))
    monkeypatch.setattr(rowdetect, "detect_row", lambda *args, **kwargs: unclear)
    error = row_refused(s, protein, MONTAGE.row)
    assert error.code is ErrorCode.ROW_OFF_LINE
    assert str(error) == OFF_LINE.format(
        which="some bands found lie", those="some bands lie", fix=""
    )


def two_rows_session(tmp_path: Path, case: RowCase = TWO_ROWS) -> tuple[ProjectSession, str, str]:
    """``case`` (TWO_ROWS, or one of its geometry) with a loading control over
    the row above and its target over the row's own, each placed by a row box
    over its own row alone; (the session, the loading control, the target)."""
    s, image, target = row_session(tmp_path, case)
    loading = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    ops.edit_protein(s, target, loading_control_ids=[loading])
    ops.set_reference_condition(s, "vehicle")
    ops.detect_row_boxes(s, loading, (*case.row[:3], case.row[3] - 40))
    ops.detect_row_boxes(s, target, (case.row[0], case.row[1] + 40, *case.row[2:]))
    return s, loading, target


def test_a_loading_control_dragged_again_over_both_rows_is_refused(tmp_path):
    # As on the smoke test's blot: the loading control's row box dragged again
    # over its row and its target's, whose boxes are placed. Committed, its
    # boxes in lanes 3-5 would lie on the target's bands, and the target's
    # values there would be normalized by its own bands. Refused, and nothing
    # changes: the fold changes stay.
    s, loading, target = two_rows_session(tmp_path)
    before, results = s.project, ops.compute(s)
    assert None not in results.series[0].fold_change
    error = row_refused(s, loading, TWO_ROWS.row)
    assert error.code is ErrorCode.ROW_OFF_LINE and error.ids == ()
    assert s.project is before and ops.compute(s) == results


@pytest.mark.parametrize(
    ("dragged", "row", "other", "name"),
    [
        pytest.param("loading", OWN_ROW, "target", "β-catenin", id="onto-its-target"),
        pytest.param("target", ABOVE_ROW, "loading", "GAPDH", id="onto-its-loading-control"),
    ],
)
def test_a_row_dragged_over_another_proteins_row_is_refused_naming_its_boxes(
    tmp_path, dragged, row, other, name
):
    s, loading, target = two_rows_session(tmp_path)
    ids = {"loading": loading, "target": target}
    covered = [band.id for band in boxes_of(s, ids[other]).bands]
    error = covered_refusal(s, lambda: ops.detect_row_boxes(s, ids[dragged], row))
    assert error.ids == tuple(covered)
    assert str(error) == (
        f"the row's boxes would overlap the boxes of {name!r} in lanes 1, 2, 3, 4, 5, 6 by"
        " more than 50% of the smaller box's area: two proteins' boxes would measure the same"
        " band"
    )
    assert [c["band_id"] for c in error.detail["covered"]] == covered
    assert {c["protein_id"] for c in error.detail["covered"]} == {ids[other]}
    assert [c["lane_index"] for c in error.detail["covered"]] == list(range(6))
    assert all(0.5 < c["share"] <= 1.0 for c in error.detail["covered"])


def covered_refusal(s: ProjectSession, call) -> OperationError:
    """The refusal of a box over another protein's, which changes nothing."""
    before = s.project
    with pytest.raises(OperationError) as info:
        call()
    assert s.project is before
    assert info.value.code is ErrorCode.OVERLAP
    assert info.value.detail["cause"] == "other_protein"
    return info.value


def test_a_box_clicked_or_moved_over_another_proteins_box_is_refused(tmp_path):
    s, loading, target = two_rows_session(tmp_path)
    x, y = at_lane(TWO_ROWS, 1)
    lc_box = lane_bands(s, loading)[1].id
    t_boxes = lane_bands(s, target)
    ops.remove_box(s, t_boxes[1].id)
    # Clicked (grown from the band) on the loading control's band in lane 1.
    error = covered_refusal(s, lambda: ops.place_box(s, target, x, y - 40, lane_index=1, grow=True))
    assert error.ids == (lc_box,)
    assert str(error) == (
        "the box would overlap the box of 'GAPDH' in lane 2 by more than 50% of the smaller"
        " box's area: two proteins' boxes would measure the same band"
    )
    [covered] = error.detail["covered"]
    assert (covered["band_id"], covered["protein_id"], covered["lane_index"]) == (
        lc_box,
        loading,
        1,
    )
    # Shift+clicked (a box of the protein's size) there.
    error = covered_refusal(
        s, lambda: ops.place_box(s, target, x, y - 40, lane_index=1, grow=False)
    )
    assert error.ids == (lc_box,)
    # Its lane-0 box moved there.
    moved = t_boxes[0]
    error = covered_refusal(s, lambda: ops.move_box(s, moved.id, rects_by_lane(s, loading)[1]))
    assert error.ids == (lc_box,)
    assert band_of(s, moved.id) == moved  # where it was


def test_a_box_over_half_of_another_proteins_box_or_less_is_placed(tmp_path):
    # A box of the loading control's size over its box in lane 1, lower by 5
    # or 6 of its 12 px height: over 7/12 of it (refused), or half of it.
    s, loading, target = two_rows_session(tmp_path)
    size = BoxSize(width=44, height=12)
    ops.set_box_size(s, loading, size)
    for band in list(boxes_of(s, target).bands):
        ops.remove_box(s, band.id)
    ops.set_box_size(s, target, size)
    x0, y0, x1, y1 = rects_by_lane(s, loading)[1]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    error = covered_refusal(
        s, lambda: ops.place_box(s, target, cx, cy + 5, lane_index=1, grow=False)
    )
    assert error.detail["covered"][0]["share"] == round(7 / 12, 3)
    band_id = ops.place_box(s, target, cx, cy + 6, lane_index=1, grow=False)
    rect = band_of(s, band_id).box.rect(size)
    assert (rect[0] - x0, rect[1] - y0) == (0, 6)  # over half of it: placed
    assert overlap_messages(s) == both_overlap("in lane 2 by 50%")  # and noted (#208)
    ops.move_box(s, band_id, (rect[0], rect[1] + 1, rect[2], rect[3] + 1))
    assert overlap_messages(s) == both_overlap("in lane 2 by 42%")  # 5 of 12 rows
    assert_nets_current(s)


def overlap_messages(s: ProjectSession) -> list[str]:
    """The boxes_overlap notices of the session's results (#208)."""
    return [n.message for n in ops.compute(s).notices if n.code is NoticeCode.BOXES_OVERLAP]


def both_overlap(where: str, *, lanes: int = 1) -> list[str]:
    """The overlap notices of β-catenin and GAPDH, each over the other ``where``."""
    its = "its net there also counts" if lanes == 1 else "its nets there also count"
    end = (
        f" of the smaller box's area: {its} the pixels the boxes share; move a box so they do"
        " not overlap"
    )
    return [
        f"'β-catenin' overlaps 'GAPDH' {where}{end}",
        f"'GAPDH' overlaps 'β-catenin' {where}{end}",
    ]


def test_a_box_wholly_inside_a_larger_box_of_another_protein_is_refused(tmp_path):
    # The target's boxes enlarged to 53x22 (to hold whole bands): the loading
    # control's row box over the target's row alone puts each of its boxes
    # wholly inside one of them, over less than half of it. The two would
    # still measure one band: the overlap counts against the smaller box, all
    # of it. Refused, the fold changes as they were; and so is a box of the
    # loading control's size dropped on one of them.
    s, loading, target = two_rows_session(tmp_path)
    ops.set_box_size(s, target, BoxSize(width=53, height=22))
    targets = [band.id for band in boxes_of(s, target).bands]
    results = ops.compute(s)
    error = covered_refusal(s, lambda: ops.detect_row_boxes(s, loading, OWN_ROW))
    assert error.ids == tuple(targets)
    assert [c["share"] for c in error.detail["covered"]] == [1.0] * 6
    assert ops.compute(s) == results
    ops.remove_box(s, lane_bands(s, loading)[1].id)
    x, y = at_lane(TWO_ROWS, 1)
    error = covered_refusal(s, lambda: ops.place_box(s, loading, x, y, lane_index=1, grow=False))
    assert error.ids == (lane_bands(s, target)[1].id,)


def test_a_box_size_over_another_proteins_boxes_is_refused(tmp_path):
    # The loading control's boxes made 100 px high reach down over the
    # target's, 40 px below: refused, the target's fold changes as they were.
    # At 80 px high they cover half of each: placed.
    s, loading, target = two_rows_session(tmp_path)
    targets = tuple(band.id for band in boxes_of(s, target).bands)
    results = ops.compute(s)
    error = covered_refusal(s, lambda: ops.set_box_size(s, loading, BoxSize(width=44, height=100)))
    assert error.ids == targets
    assert str(error) == (
        "at the box size 44x100, boxes of 'GAPDH' would overlap the boxes of 'β-catenin' in"
        " lanes 1, 2, 3, 4, 5, 6 by more than 50% of the smaller box's area: two proteins'"
        " boxes would measure the same band"
    )
    assert ops.compute(s) == results
    ops.set_box_size(s, loading, BoxSize(width=44, height=80))
    assert boxes_of(s, loading).box_size == BoxSize(width=44, height=80)
    # 6 of the target's 12 rows in each lane: half, noted (#208).
    assert overlap_messages(s) == both_overlap("in lanes 1, 2, 3, 4, 5, 6 by 50%", lanes=6)


def _smear(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
    """A dark smear 36 px wide in lane 0, from 130 px above the row's band down
    into the band of the row 40 px above it."""
    inside = (np.abs(X - lcx[0]) < 18) & (Y > lcy[0] - 130) & (Y < lcy[0] - 36)
    return np.where(inside, 20000.0, 0.0)


# TWO_ROWS on a taller image, with a smear above the loading control's band
# in lane 0.
SMEARED = adversarial_row(
    "two rows",
    1000,
    neighbour_dy=-40.0,
    neighbour_rel=0.5,
    neighbour_rels={0: 2.0, 1: 2.0, 2: 2.0},
    box_adjust=(0, -40, 0, 0),
    img_h=400,
    artefacts=[_smear],
)


def test_a_click_that_grows_boxes_over_another_proteins_is_refused(tmp_path):
    # The loading control's lane-1 box removed, then the smear above its band
    # clicked: the band grown from the click needs boxes 96 px high. The new
    # box, over the smear, lies clear of the target's box in lane 1, but the
    # loading control's other boxes, grown to that around their centres,
    # would reach down over the target's. Refused.
    s, loading, target = two_rows_session(tmp_path, SMEARED)
    assert boxes_of(s, loading).box_size == BoxSize(width=44, height=17)
    ops.remove_box(s, lane_bands(s, loading)[0].id)
    x, y = round(SMEARED.lane_cx[0]), round(SMEARED.lane_cy[0] - 80)
    error = covered_refusal(s, lambda: ops.place_box(s, loading, x, y, lane_index=0, grow=True))
    assert error.ids == tuple(band.id for band in boxes_of(s, target).bands)[1:]
    assert str(error) == (
        "at the box size 44x96 this band needs, boxes of 'GAPDH' would overlap the boxes of"
        " 'β-catenin' in lanes 2, 3, 4, 5, 6 by more than 50% of the smaller box's area: two"
        " proteins' boxes would measure the same band"
    )


def test_a_row_that_grows_its_kept_boxes_over_another_proteins_is_refused(tmp_path, monkeypatch):
    # The loading control's boxes all moved by hand, which a row keeps, and a
    # row box over its row whose bands need boxes 100 px high: the kept
    # boxes, grown to that, would reach down over the target's.
    s, loading, target = two_rows_session(tmp_path)
    for band in list(boxes_of(s, loading).bands):
        edit_by_hand(s, band.id)
    found = detected(s, loading, ABOVE_ROW)
    tall = dataclasses.replace(found, size=BoxSize(width=44, height=100))
    monkeypatch.setattr(rowdetect, "detect_row", lambda *args, **kwargs: tall)
    error = covered_refusal(s, lambda: ops.detect_row_boxes(s, loading, ABOVE_ROW))
    assert error.ids == tuple(band.id for band in boxes_of(s, target).bands)
    assert str(error) == (
        "at the box size 44x100 the row needs, the boxes of 'GAPDH' it keeps would overlap"
        " the boxes of 'β-catenin' in lanes 1, 2, 3, 4, 5, 6 by more than 50% of the smaller"
        " box's area: two proteins' boxes would measure the same band"
    )


def test_a_row_that_grows_its_kept_boxes_over_half_of_another_proteins_or_less_is_noted(
    tmp_path, monkeypatch
):
    # As above, the row's bands needing boxes 80 px high: the kept boxes,
    # moved 1 px right and grown to that, cover 43 columns by 6 rows of the
    # target's 44x12 in each lane (258 of 528 px, 49%). Placed, and noted
    # (#208).
    s, loading, target = two_rows_session(tmp_path)
    for band in list(boxes_of(s, loading).bands):
        edit_by_hand(s, band.id)
    found = detected(s, loading, ABOVE_ROW)
    tall = dataclasses.replace(found, size=BoxSize(width=44, height=80))
    monkeypatch.setattr(rowdetect, "detect_row", lambda *args, **kwargs: tall)
    assert overlap_messages(s) == []
    ops.detect_row_boxes(s, loading, ABOVE_ROW)
    assert boxes_of(s, loading).box_size.height == 80
    assert overlap_messages(s) == both_overlap("in lanes 1, 2, 3, 4, 5, 6 by 49%", lanes=6)


def test_a_row_box_over_two_rows_that_cuts_bands_names_the_lanes_off_its_line(tmp_path):
    # The row box over both rows, its top edge through the row above's bands
    # and its bottom edge through the row's own: the lanes are read, off the
    # row's line is the only refusing flag. Said so, naming those lanes, with
    # the cut: not that the box does not show which lane each band is in.
    case = adversarial_row(
        "two rows",
        1000,
        neighbour_dy=-40.0,
        neighbour_rel=0.5,
        neighbour_rels={0: 2.0, 1: 2.0, 2: 2.0},
        box_adjust=(0, -32, 0, -8),
        img_h=200,
    )
    s, _, protein = row_session(tmp_path, case)
    found = detected(s, protein, case.row)
    assert [flag for flag in found.flags if flag in rowdetect.REFUSING_FLAGS] == ["off_row_line"]
    assert any(lane.cut for lane in found.lanes)
    off = [lane.lane for lane in found.lanes if abs(lane.line_offset) > rowdetect.ROW_LINE_K]
    assert off in ([0, 1, 2], [3, 4, 5])
    error = row_refused(s, protein, case.row)
    assert (error.code, error.detail["cause"]) == (ErrorCode.ROW_OFF_LINE, "off_row_line")
    assert str(error) == (
        f"the row box cuts through the bands, and the bands found in {lanes_phrase(off)} lie"
        " above or below the row's line through the other bands, by more than 0.75 of a"
        " box's height: the row box covers more than one row, or cuts those bands; draw it"
        " over one row only, over the whole band height, or box those lanes by clicking"
        " their bands"
    )
    assert_raw_reason(error.detail, found)


def _crossed(dy: float, rel: float, **kwargs) -> tuple[RowCase, RowDetection]:
    """A row smiling 8 px with another row ``dy`` px below it (above if
    negative), ``rel`` times as deep, the row box over both, detected with the
    expected row 3.5 px off the row's bands towards the other row where they
    lie nearest it: its middle lanes (with the other row above) or its end
    lanes (below) grow from the other row's bands, the rest from the row's
    own (#58)."""
    adjust = (0, -30, 0, 0) if dy < 0 else (0, 0, 0, 30)
    case = adversarial_row(
        "smiling",
        1000,
        smile=8.0,
        neighbour_dy=dy,
        neighbour_rel=rel,
        box_adjust=kwargs.pop("box_adjust", adjust),
        **kwargs,
    )
    ys = [cy + 0.5 for cy in case.lane_cy]
    found = rowdetect.detect_row(
        case.image,
        case.row,
        case.n_lanes,
        background=estimate_background(case.image),
        prefer_y=min(ys) - 3.5 if dy < 0 else max(ys) + 3.5,
    )
    return case, found


def _off_line(found: RowDetection) -> list[int]:
    return [
        lane.lane
        for lane in found.lanes
        if lane.line_offset is not None and abs(lane.line_offset) > rowdetect.ROW_LINE_K
    ]


def test_lanes_grown_from_bands_on_two_rows_are_named_before_a_lane_off_the_line():
    # Lanes 1, 2, 5, 6 grew from bands on two rows, and lane 3's bands lie
    # 9 px below the others' (a montage's panel), off the row's line. The
    # lanes grown from two rows say why the row box covers more than one row:
    # they are named, not lane 3.
    case, found = _crossed(-16.0, 2.0, shifts={2: 9.0})
    assert found.crossed == (0, 1, 4, 5)
    assert _off_line(found) == [2]
    error = ops._row_refusal(found, case.row[0], case.row[2])
    assert (error.code, error.detail["cause"]) == (ErrorCode.ROW_OFF_LINE, "off_row_line")
    assert str(error) == (
        "the bands found in lanes 1, 2, 5, 6 lie on two rows, a lane's on one and its"
        " neighbour's on the other, each lane holding a band on both: the row box covers more"
        " than one row, and the bands nearest the expected row do not lie on one; draw it over"
        " one row only, or box those lanes by clicking their bands"
    )


def test_lanes_grown_from_bands_on_two_rows_refuse_the_row_as_off_its_line_when_it_cuts():
    # The row box's bottom edge runs 2 px above the other row's bands'
    # centres in lanes 3 and 4, cutting them. The lanes are read, and lanes
    # 1, 2, 5, 6 grew from bands on two rows: said so, with the cut, not that
    # the box does not show which lane each band is in.
    case, found = _crossed(16.0, 3.0, box_adjust=(0, 0, 0, 2))
    assert found.crossed == (0, 1, 4, 5)
    assert [lane.lane for lane in found.lanes if lane.cut] == [2, 3]
    assert _off_line(found) == []
    error = ops._row_refusal(found, case.row[0], case.row[2])
    assert (error.code, error.detail["cause"]) == (ErrorCode.ROW_OFF_LINE, "off_row_line")
    assert str(error).startswith(
        "the row box cuts through the bands, and the bands found in lanes 1, 2, 5, 6 lie on"
        " two rows,"
    )
    assert str(error).endswith(
        "draw it over one row only, over the whole band height, or box those lanes by"
        " clicking their bands"
    )


def test_two_boxes_on_two_rows_are_both_named(tmp_path):
    # Two lanes, the first one's strongest band in the row above: neither box
    # is the row's.
    case = adversarial_row(
        "two rows",
        1000,
        n=2,
        neighbour_dy=-40.0,
        neighbour_rel=0.5,
        neighbour_rels={0: 2.0},
        box_adjust=(0, -40, 0, 0),
        img_h=200,
    )
    s, _, protein = row_session(tmp_path, case)
    error = row_refused(s, protein, case.row)
    assert (error.code, error.detail["cause"]) == (ErrorCode.ROW_OFF_LINE, "off_row_line")
    assert str(error) == (
        "the bands found in lanes 1, 2 lie on two rows, more than 2 box heights apart: the"
        " row box covers more than one row; draw it over one row only, or box those lanes by"
        " clicking their bands"
    )


def test_boxes_on_another_image_are_not_in_the_way(tmp_path):
    # The same pixels in a second image: its protein's boxes lie elsewhere.
    s, loading, _ = two_rows_session(tmp_path)
    other_image = import_blot(s, TWO_ROWS.image.astype(np.uint16), "reprobe β.tif")
    other = ops.add_protein(s, "α-tubulin", Role.LOADING_CONTROL, other_image)
    placement = ops.detect_row_boxes(s, other, ABOVE_ROW)
    assert None not in placement.band_ids


# --- #57: a protein's box padding ---

PAD_CASE = ROWS["all_present"]  # six lanes 67 to 72 px apart on a 533x120 image


def padded_row(tmp_path: Path, hook=None) -> tuple[ProjectSession, str, str]:
    """:func:`row_session` with β-catenin's row detected: six boxes of the fitted
    size 44x12, no padding yet."""
    s, image, protein = row_session(tmp_path, PAD_CASE, hook=hook)
    ops.detect_row_boxes(s, protein, PAD_CASE.row)
    assert boxes_of(s, protein).box_size == BoxSize(width=44, height=12)
    return s, image, protein


def rects_of(s: ProjectSession, protein_id: str) -> list[tuple[int, int, int, int]]:
    """The protein's boxes, as stored (by lane, then band index)."""
    protein = boxes_of(s, protein_id)
    return [band.box.rect(protein.box_size) for band in protein.bands]


def grown_by(rect: Sequence[int], across: int, along: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    return (x0 - across, y0 - along, x1 + across, y1 + along)


def test_set_box_padding_grows_each_box_by_the_padding_around_its_centre(tmp_path):
    s, _, protein = padded_row(tmp_path)
    before = boxes_of(s, protein)
    old = rects_of(s, protein)

    change = ops.set_box_padding(s, protein, across=3, along=5)
    after = boxes_of(s, protein)
    assert after.box_padding == BoxPadding(across=3, along=5)
    assert after.fitted_size == before.box_size == BoxSize(width=44, height=12)  # kept
    assert after.box_size == BoxSize(width=50, height=22)
    # Every box grew by the padding on each side: its centre, and so each
    # lane's place in the row, is where it was.
    assert rects_of(s, protein) == [grown_by(rect, 3, 5) for rect in old]
    for was, now in zip(before.bands, after.bands, strict=True):
        assert (now.id, now.lane_index, now.source) == (was.id, was.lane_index, was.source)
        assert not now.manually_edited  # a protein-level setting, as a size change is
        assert now.net > was.net  # the band's tails come in
    changes = [now.net / was.net - 1 for was, now in zip(before.bands, after.bands, strict=True)]
    assert change == ops.PaddingChange(
        box_size=BoxSize(width=50, height=22),
        fitted_size=BoxSize(width=44, height=12),
        padding=BoxPadding(across=3, along=5),
        net_change=(min(changes), max(changes)),
        edge_shifted=(),
        overlapping=(),
    )
    assert_nets_current(s)
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "set_box_padding",
        {
            "protein_id": protein,
            "box_padding": {"across": 3, "along": 5},
            "box_size": {"width": 50, "height": 22},
        },
    )
    assert entry.content_hash == content_hash(s.project)

    # Lowered to nothing, every box and every net is what it was.
    ops.set_box_padding(s, protein, across=0, along=0)
    assert boxes_of(s, protein) == before


def test_set_box_padding_same_is_a_noop(tmp_path):
    recorder = Recorder()
    s, _, protein = padded_row(tmp_path, recorder)
    ops.set_box_padding(s, protein, along=2)
    committed, length = s.project, len(s.project.log)
    recorder.actions.clear()
    for kwargs in ({}, {"along": 2}, {"across": 0}, {"across": 0, "along": 2}):
        assert ops.set_box_padding(s, protein, **kwargs) == ops.PaddingChange(
            box_size=BoxSize(width=44, height=16),
            fitted_size=BoxSize(width=44, height=12),
            padding=BoxPadding(across=0, along=2),
            net_change=None,
            edge_shifted=(),
            overlapping=(),
        )
        assert s.project is committed, kwargs
    assert len(s.project.log) == length
    assert recorder.actions == []
    # Without a box, the padding kept by a clear is sent again: a no-op, not a refusal.
    ops.clear_boxes(s, protein)
    committed = s.project
    assert ops.set_box_padding(s, protein, along=2).padding == BoxPadding(along=2)
    assert s.project is committed
    with pytest.raises(OperationError, match="place a box of 'β-catenin' first"):
        ops.set_box_padding(s, protein, along=1)  # lowering needs a box too


def test_set_box_padding_keeps_a_field_left_out(tmp_path):
    s, _, protein = padded_row(tmp_path)
    ops.set_box_padding(s, protein, across=2, along=3)
    ops.set_box_padding(s, protein, along=1)  # as a form that sends one direction
    assert boxes_of(s, protein).box_padding == BoxPadding(across=2, along=1)
    # The log names both directions as they took effect, the kept one too.
    assert s.project.log[-1].params["box_padding"] == {"across": 2, "along": 1}
    ops.set_box_padding(s, protein, across=0)
    assert boxes_of(s, protein).box_padding == BoxPadding(across=0, along=1)
    assert boxes_of(s, protein).box_size == BoxSize(width=44, height=14)


def _padding_scene(tmp_path: Path) -> tuple[ProjectSession, Recorder, dict[str, str]]:
    """:func:`padded_row`, saved and reopened with an empty pixel cache and a
    recording hook, with GAPDH's one fixed 40x80 box in lane 1, touching the
    left side of β-catenin's there (over it, two proteins' boxes would measure
    one band, #114), and p53, which has no box."""
    s, image, beta = padded_row(tmp_path, save_to_folder)
    gapdh = ops.add_protein(
        s, "GAPDH", Role.LOADING_CONTROL, image, box_size=BoxSize(width=40, height=80)
    )
    x, y = at_lane(PAD_CASE, 0)
    ops.place_box(s, gapdh, x - 42, y, lane_index=0, grow=False)
    p53 = ops.add_protein(s, "p53", Role.TARGET, image)
    recorder = Recorder()
    reopened = ops.open_project(s.folder, autosave=recorder, clock=FakeClock())
    return reopened, recorder, {"image": image, "beta": beta, "gapdh": gapdh, "p53": p53}


def _spoil_image(s: ProjectSession, ids: dict[str, str]) -> None:
    (s.folder / storage.IMAGES_DIR / f"{ids['image']}.tif").write_bytes(b"not the image")


def _stack_a_second_band(s: ProjectSession, ids: dict[str, str]) -> None:
    """A box of β-catenin's second expected band 1 px below its lane-1 box."""

    def change(draft: Project) -> None:
        beta = draft.batch.find_protein(ids["beta"])
        beta.expected_band_count = 2
        [first] = [band for band in draft.batch.boxes_of(beta).bands if band.lane_index == 0]
        draft.batch.boxes_of(beta).bands.append(
            Band(
                id=draft.new_id("band"),
                lane_index=0,
                band_index=1,
                box=Box(x=first.box.x, y=first.box.y + 13),
                source=ProposalSource.MANUAL,
                **ops._UNQUANTIFIED,
            )
        )

    plant(s, change)


def _beta_boxes(*keys: tuple[int, int]):
    """The ids of β-catenin's boxes at these (lane, band index), as a refusal names them."""

    def named(s: ProjectSession, ids: dict[str, str]) -> tuple[str, ...]:
        found = {(b.lane_index, b.band_index): b.id for b in boxes_of(s, ids["beta"]).bands}
        return tuple(found[key] for key in keys)

    return named


PADDING_REFUSALS = [
    pytest.param(
        None,
        "beta",
        {"across": True},
        ErrorCode.INVALID_INPUT,
        "padding left and right must be an integer, not True",
        None,
        id="a-bool",
    ),
    pytest.param(
        None,
        "beta",
        {"along": 2.0},
        ErrorCode.INVALID_INPUT,
        "padding above and below must be an integer, not 2.0",
        None,
        id="a-float",
    ),
    pytest.param(
        None,
        "beta",
        {"along": "2"},
        ErrorCode.INVALID_INPUT,
        "padding above and below must be an integer, not '2'",
        None,
        id="text",
    ),
    pytest.param(
        None,
        "beta",
        {"along": -1},
        ErrorCode.INVALID_INPUT,
        "padding above and below must be 0 or more, not -1",
        None,
        id="negative",
    ),
    pytest.param(
        None,
        "beta",
        {"across": -1, "along": True},  # every type first
        ErrorCode.INVALID_INPUT,
        "padding above and below must be an integer, not True",
        None,
        id="types-before-signs",
    ),
    pytest.param(
        None,
        "p53",
        {"along": 1},
        ErrorCode.INVALID_INPUT,
        "place a box of 'p53' first: its padding follows the fitted size the bands set",
        None,
        id="no-box-yet",
    ),
    pytest.param(
        None,
        "beta",
        {"along": 7},
        ErrorCode.INVALID_INPUT,
        "padding above and below can be raised to at most 6 px for 'β-catenin' (half its"
        " fitted height, 12 px), not 7",
        None,
        id="above-half",
    ),
    pytest.param(
        None,
        "beta",
        {"across": 23},  # the half comes before the overlap
        ErrorCode.INVALID_INPUT,
        "padding left and right can be raised to at most 22 px for 'β-catenin' (half its"
        " fitted width, 44 px), not 23",
        None,
        id="above-half-across",
    ),
    pytest.param(
        None,
        "gapdh",
        {"along": 21},
        ErrorCode.SIZE_OUT_OF_BOUNDS,
        "box size 40x122 (fitted 40x80 plus 21 px above and below) exceeds the 533x120 image img-1",
        None,
        id="beyond-the-image",
    ),
    pytest.param(
        None,
        "beta",
        {"across": 12},
        ErrorCode.SIZE_WOULD_OVERLAP,
        "12 px of padding left and right would make the boxes of 'β-catenin' in lanes 1, 2"
        " overlap; at most 11 px fits",
        _beta_boxes((0, 0), (1, 0)),
        id="overlap",
    ),
    pytest.param(
        None,
        "beta",
        {"across": 12, "along": 2},
        ErrorCode.SIZE_WOULD_OVERLAP,
        "12 px of padding left and right, with 2 px above and below, would make the boxes of"
        " 'β-catenin' in lanes 1, 2 overlap; at most 11 px fits",
        _beta_boxes((0, 0), (1, 0)),
        id="overlap-with-both",
    ),
    pytest.param(
        _stack_a_second_band,
        "beta",
        {"along": 1},
        ErrorCode.SIZE_WOULD_OVERLAP,
        "1 px of padding above and below would make the boxes of 'β-catenin' in lane 1"
        " overlap; at most 0 px fits",
        _beta_boxes((0, 0), (0, 1)),
        id="stacked-boxes",
    ),
    pytest.param(
        _stack_a_second_band,
        "beta",
        {"across": 2, "along": 1},  # 2 px left and right alone fits: above and below is named
        ErrorCode.SIZE_WOULD_OVERLAP,
        "1 px of padding above and below, with 2 px left and right, would make the boxes of"
        " 'β-catenin' in lane 1 overlap; at most 0 px fits",
        _beta_boxes((0, 0), (0, 1)),
        id="overlap-named-by-the-direction-that-makes-it",
    ),
    pytest.param(
        _stack_a_second_band,
        "beta",
        {"across": 12, "along": 1},  # none of left and right fits under 1 px above and below
        ErrorCode.SIZE_WOULD_OVERLAP,
        "12 px of padding left and right, with 1 px above and below, would make the boxes of"
        " 'β-catenin' in lanes 1, 2 overlap",
        _beta_boxes((0, 0), (0, 1), (1, 0)),
        id="overlap-where-none-fits",
    ),
    pytest.param(
        _spoil_image,
        "beta",
        {"along": 1},
        ErrorCode.IMAGE_FILE_CHANGED,
        None,
        lambda s, ids: (ids["image"],),
        id="image-file-changed",
    ),
    pytest.param(None, "prot-99", {"along": 1}, None, None, None, id="unknown-id"),
]


@pytest.mark.parametrize(
    ("setup", "protein", "kwargs", "code", "message", "named"), PADDING_REFUSALS
)
def test_set_box_padding_refusals(tmp_path, setup, protein, kwargs, code, message, named):
    s, recorder, ids = _padding_scene(tmp_path)
    if setup is not None:
        setup(s, ids)
    recorder.actions.clear()
    before = s.project
    cache = dict(s._pixels)
    with pytest.raises(UnknownIdError if code is None else OperationError) as info:
        ops.set_box_padding(s, ids.get(protein, protein), **kwargs)
    if code is not None:
        assert info.value.code is code
        if message is not None:
            assert str(info.value) == message
        assert info.value.ids == (() if named is None else named(s, ids))
    assert s.project is before  # nothing changed, and no log entry
    assert recorder.actions == []
    assert s._pixels.keys() == cache.keys()


def test_a_row_keeps_a_padding_above_half(tmp_path):
    s, _, protein = padded_row(tmp_path)
    # A padding set on a taller fitted size (20 px) ...
    ops.set_box_size(s, protein, BoxSize(width=44, height=20))
    ops.set_box_padding(s, protein, along=10)
    # ... is kept by the next row, which fits 12 px afresh: 10 px is now more
    # than half of it, and nothing is refused or rewritten.
    ops.clear_boxes(s, protein)
    ops.detect_row_boxes(s, protein, PAD_CASE.row)
    beta = boxes_of(s, protein)
    assert (beta.fitted_size, beta.box_padding) == (
        BoxSize(width=44, height=12),
        BoxPadding(along=10),
    )
    assert beta.box_size == BoxSize(width=44, height=32)
    assert_nets_current(s)
    committed = s.project
    ops.set_box_padding(s, protein, along=10)  # sent again: a no-op, not a refusal
    assert s.project is committed
    with pytest.raises(OperationError) as info:
        ops.set_box_padding(s, protein, along=11)  # raised: refused
    assert "at most 6 px" in str(info.value)
    assert s.project is committed
    # A larger fitted size, though still under twice the padding, is not refused.
    ops.set_box_size(s, protein, BoxSize(width=44, height=14))
    ops.set_box_padding(s, protein, along=8)  # lowered, though still above half: accepted
    beta = boxes_of(s, protein)
    assert (beta.fitted_size, beta.box_padding) == (
        BoxSize(width=44, height=14),
        BoxPadding(along=8),
    )
    assert_nets_current(s)


def test_set_box_size_refuses_a_shrink_under_twice_the_padding(tmp_path):
    s, _, protein = padded_row(tmp_path)
    ops.set_box_size(s, protein, BoxSize(width=44, height=14))
    ops.set_box_padding(s, protein, along=7)
    committed = s.project
    with pytest.raises(OperationError) as info:
        ops.set_box_size(s, protein, BoxSize(width=44, height=6))
    assert (info.value.code, str(info.value)) == (
        ErrorCode.INVALID_INPUT,
        "a fitted height of 6 px would leave the 7 px of padding above and below more than"
        " half of it; lower that padding to 3 px or less first, or type a height of at least"
        " 14",
    )
    with pytest.raises(OperationError) as info:
        ops.set_box_size(s, protein, BoxSize(width=44, height=1))
    assert "remove that padding first, or type a height of at least 14" in str(info.value)
    for refused in [(44, 14), None]:  # not a BoxSize: refused, not an AttributeError
        with pytest.raises(OperationError) as info:
            ops.set_box_size(s, protein, refused)
        assert info.value.code is ErrorCode.INVALID_INPUT
    assert s.project is committed
    # Twice the padding, or larger than the fitted size, is accepted.
    ops.set_box_size(s, protein, BoxSize(width=40, height=14))  # the same height
    ops.set_box_size(s, protein, BoxSize(width=40, height=20))
    ops.set_box_padding(s, protein, across=5)
    with pytest.raises(OperationError) as info:
        ops.set_box_size(s, protein, BoxSize(width=9, height=20))
    assert str(info.value) == (
        "a fitted width of 9 px would leave the 5 px of padding left and right more than half"
        " of it; lower that padding to 4 px or less first, or type a width of at least 10"
    )
    ops.set_box_size(s, protein, BoxSize(width=10, height=14))
    assert boxes_of(s, protein).box_size == BoxSize(width=20, height=28)


def test_set_box_size_without_a_box_asks_only_for_a_larger_size(tmp_path):
    # With no box, lowering the padding is refused (a box comes first), so the
    # refusal of a shrink under twice the padding does not offer it.
    s, _, protein = padded_row(tmp_path)
    ops.set_box_padding(s, protein, along=6)
    ops.clear_boxes(s, protein)
    committed = s.project
    for height, message in [
        (
            8,
            "a fitted height of 8 px would leave the 6 px of padding above and below more than"
            " half of it; type a height of at least 12",
        ),
        (
            1,
            "a fitted height of 1 px would leave the 6 px of padding above and below more than"
            " half of it; type a height of at least 12",
        ),
    ]:
        with pytest.raises(OperationError) as info:
            ops.set_box_size(s, protein, BoxSize(width=44, height=height))
        assert (info.value.code, str(info.value)) == (ErrorCode.INVALID_INPUT, message)
        assert s.project is committed
    with pytest.raises(OperationError, match="place a box of 'β-catenin' first"):
        ops.set_box_padding(s, protein, along=4)
    # What the refusal asks for is accepted.
    ops.set_box_size(s, protein, BoxSize(width=44, height=14))
    beta = boxes_of(s, protein)
    assert (beta.fitted_size, beta.box_padding) == (
        BoxSize(width=44, height=14),
        BoxPadding(along=6),
    )


def test_a_typed_fitted_size_lasts_until_a_seed_click_on_no_box(tmp_path):
    # A fixed click keeps a typed fitted size, and so does a seed click whose
    # band needs less while a box survives; a seed click on the protein with
    # no box sets it afresh, even smaller than the size typed. The padding stays.
    s, _, protein = boxed(tmp_path)
    typed = BoxSize(width=30, height=20)
    ops.set_box_size(s, protein, typed)  # before any box
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=False)
    assert boxes_of(s, protein).fitted_size == typed
    ops.set_box_padding(s, protein, along=2)
    ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)  # needs 19x5
    assert boxes_of(s, protein).fitted_size == typed
    ops.clear_boxes(s, protein)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)  # needs 9x5
    beta = boxes_of(s, protein)
    assert (beta.fitted_size, beta.box_padding) == (BoxSize(width=9, height=5), BoxPadding(along=2))
    assert_nets_current(s)


def test_set_box_size_sets_the_fitted_size_under_padding(tmp_path):
    s, _, protein = padded_row(tmp_path)
    ops.set_box_padding(s, protein, across=2, along=5)
    old = rects_of(s, protein)
    ops.set_box_size(s, protein, BoxSize(width=40, height=14))
    beta = boxes_of(s, protein)
    assert (beta.fitted_size, beta.box_padding, beta.box_size) == (
        BoxSize(width=40, height=14),
        BoxPadding(across=2, along=5),
        BoxSize(width=44, height=24),  # the typed size plus the padding
    )
    assert rects_of(s, protein) == [grown_by(rect, -2, 1) for rect in old]
    assert_nets_current(s)
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "set_box_size",
        {
            "protein_id": protein,
            "box_size": {"width": 44, "height": 24},  # the size used
            "fitted_size": {"width": 40, "height": 14},  # the size typed
        },
    )
    committed = s.project
    ops.set_box_size(s, protein, BoxSize(width=40, height=14))  # the fitted size: a no-op
    assert s.project is committed
    # The refusals name the fitted size and the padding.
    for refused, code, message in [
        (
            BoxSize(width=40, height=115),
            ErrorCode.SIZE_OUT_OF_BOUNDS,
            "box size 44x125 (fitted 40x115 plus 2 px left and right and 5 px above and below)"
            " exceeds the 533x120 image img-1",
        ),
        (
            BoxSize(width=66, height=14),
            ErrorCode.SIZE_WOULD_OVERLAP,
            "box size 70x24 (fitted 66x14 plus 2 px left and right and 5 px above and below)"
            " would make boxes of 'β-catenin' overlap",
        ),
    ]:
        with pytest.raises(OperationError) as info:
            ops.set_box_size(s, protein, refused)
        assert (info.value.code, str(info.value), info.value.ids) == (code, message, ())
        assert s.project is committed


def test_set_box_padding_reports_net_change_edge_and_overlap(tmp_path):
    # β-catenin's grown box in lane 1 and a fixed one the image's right edge
    # stops; GAPDH's box touching the first one's right side.
    s, image, beta = boxed(tmp_path)
    ops.place_box(s, beta, NARROW_X, ROW, lane_index=0, grow=True)
    edge = ops.place_box(s, beta, W - 2, ROW, lane_index=2, grow=False)
    size = boxes_of(s, beta).box_size
    assert size == BoxSize(width=9, height=5)
    assert band_of(s, edge).box.rect(size) == (W - 9, 28, W, 33)
    small = BoxSize(width=6, height=5)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image, box_size=small)
    beside = ops.place_box(s, gapdh, 48, ROW, lane_index=0, grow=False)
    ops.place_box(s, gapdh, WIDE_X, ROW, lane_index=1, grow=False)  # far from β-catenin's
    before = s.project.batch
    old = rects_of(s, beta)
    assert not boxes.overlaps_any(band_of(s, beside).box.rect(small), old)

    change = ops.set_box_padding(s, beta, across=2, along=1)
    grown, stopped = rects_of(s, beta)
    assert grown == grown_by(old[0], 2, 1)
    # The edge box could not grow evenly: shifted inside the image, it still
    # holds its fitted region, and extends 4 px further on its left.
    assert stopped == (W - 13, 27, W, 34) != grown_by(old[1], 2, 1)
    assert change.edge_shifted == (edge,)
    assert change.overlapping == (beside,)  # other proteins' boxes may overlap: reported
    # Its nets: the edge box's, on the background, is 0 before and left out.
    [band] = [b for b in before.boxes_of(before.find_protein(beta)).bands if b.net > 0]
    ratio = band_of(s, band.id).net / band.net - 1
    assert change.net_change == (ratio, ratio)
    # GAPDH's ring leaves out the grown box: its net moved, reported as a row reports it.
    assert [band_id for band_id, _, _ in change.remeasured] == [beside]
    [(_, net_before, net_after)] = change.remeasured
    assert net_after == band_of(s, beside).net != net_before
    assert change.largest_change == (beside, abs(net_after - net_before) / net_before)
    assert_nets_current(s)


def test_set_box_padding_reports_only_the_boxes_it_newly_overlaps(tmp_path):
    # GAPDH's box in lane 1 overlaps β-catenin's before the padding; its box in
    # lane 2 touches β-catenin's right side. The padding makes only the second
    # overlap, so only that box is reported. (The first overlaps its bottom
    # row alone, and two rows once padded: at most half of GAPDH's box, which
    # more would refuse, #114.)
    s, image, beta = boxed(tmp_path)
    ops.place_box(s, beta, NARROW_X, ROW, lane_index=0, grow=True)
    small = BoxSize(width=6, height=5)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image, box_size=small)
    over = ops.place_box(s, gapdh, NARROW_X - 2, ROW + 4, lane_index=0, grow=False)
    beside = ops.place_box(s, gapdh, 48, ROW, lane_index=1, grow=False)
    old = rects_of(s, beta)
    assert boxes.overlaps_any(band_of(s, over).box.rect(small), old)
    assert not boxes.overlaps_any(band_of(s, beside).box.rect(small), old)

    change = ops.set_box_padding(s, beta, across=2, along=1)
    grown = rects_of(s, beta)
    for band_id in (over, beside):
        assert boxes.overlaps_any(band_of(s, band_id).box.rect(small), grown)
    assert change.overlapping == (beside,)
    assert_nets_current(s)


def test_set_box_padding_over_half_of_another_proteins_box_is_refused(tmp_path):
    # GAPDH's 6x5 box touches the right side of β-catenin's 9x5 box. Padded 3
    # px left and right, β-catenin's box overlaps half of it: accepted and
    # reported. Padded 4 px, two thirds of it: refused, as a box size is (#114),
    # naming GAPDH's box, and nothing changes.
    s, image, beta = boxed(tmp_path)
    ops.place_box(s, beta, NARROW_X, ROW, lane_index=0, grow=True)
    small = BoxSize(width=6, height=5)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image, box_size=small)
    beside = ops.place_box(s, gapdh, 48, ROW, lane_index=1, grow=False)
    assert band_of(s, beside).box.rect(small) == (45, 28, 51, 33)
    assert rects_of(s, beta) == [(36, 28, 45, 33)]

    committed = s.project
    with pytest.raises(OperationError) as info:
        ops.set_box_padding(s, beta, across=4)
    assert (info.value.code, str(info.value), info.value.ids) == (
        ErrorCode.OVERLAP,
        "at the box size 17x5 (fitted 9x5 plus 4 px left and right), boxes of 'β-catenin'"
        " would overlap the box of 'GAPDH' in lane 2 by more than 50% of the smaller box's"
        " area: two proteins' boxes would measure the same band",
        (beside,),
    )
    assert info.value.detail == {
        "cause": "other_protein",
        "covered": [
            {"band_id": beside, "protein_id": gapdh, "lane_index": 1, "share": round(2 / 3, 3)}
        ],
    }
    assert s.project is committed

    change = ops.set_box_padding(s, beta, across=3)  # half of GAPDH's box: accepted
    assert rects_of(s, beta) == [(33, 28, 48, 33)]
    assert change.overlapping == (beside,)
    assert_nets_current(s)

    # Lowered too: a smaller box may lie more inside another. GAPDH's 31x31 box
    # over the lower right of β-catenin's box padded 4 px left and right and 2
    # above and below holds under half of it, and over half of it unpadded.
    ops.remove_box(s, beside)
    ops.set_box_padding(s, beta, across=4, along=2)
    assert rects_of(s, beta) == [(32, 26, 49, 35)]
    ops.set_box_size(s, gapdh, BoxSize(width=31, height=31))
    corner = ops.place_box(s, gapdh, 54, 44, lane_index=1, grow=False)
    assert band_of(s, corner).box.rect(BoxSize(width=31, height=31)) == (39, 29, 70, 60)
    committed = s.project
    with pytest.raises(OperationError) as info:
        ops.set_box_padding(s, beta, across=0, along=0)
    assert (info.value.code, info.value.ids) == (ErrorCode.OVERLAP, (corner,))
    assert info.value.detail["covered"][0]["share"] == round(24 / 45, 3)
    assert s.project is committed


def test_padding_survives_the_same_row_again(tmp_path):
    s, _, protein = padded_row(tmp_path)
    ops.set_box_padding(s, protein, across=3, along=4)
    committed = s.project
    again = ops.detect_row_boxes(s, protein, PAD_CASE.row)
    assert s.project is committed  # a no-op: the padding counts in the size it needs
    assert again.box_size == BoxSize(width=50, height=20)


def test_padding_survives_clear_then_row(tmp_path):
    s, _, protein = padded_row(tmp_path)
    ops.set_box_padding(s, protein, across=3, along=4)
    padded = rects_of(s, protein)
    ops.clear_boxes(s, protein)
    assert boxes_of(s, protein).box_padding == BoxPadding(across=3, along=4)  # kept
    placement = ops.detect_row_boxes(s, protein, PAD_CASE.row)
    assert placement.box_size == BoxSize(width=50, height=20)
    assert rects_of(s, protein) == padded  # the same boxes, under new ids
    assert s.project.log[-1].params["box_size"] == {"width": 50, "height": 20}
    assert_nets_current(s)


def test_row_with_padding_places_detector_boxes_grown_by_it(tmp_path):
    s, _, protein = padded_row(tmp_path)
    found = detected(s, protein, PAD_CASE.row)
    ops.set_box_padding(s, protein, across=2, along=5)
    ops.clear_boxes(s, protein)
    ops.detect_row_boxes(s, protein, PAD_CASE.row)
    assert boxes_of(s, protein).fitted_size == found.size  # the detector's size
    assert rects_by_lane(s, protein) == {
        lane.lane: grown_by(lane.rect, 2, 5) for lane in found.lanes
    }


def test_click_grow_keeps_the_padding(tmp_path):
    s, image, protein = boxed(tmp_path)
    background = s.project.batch.find_image(image).background
    narrow = grow_box(s.pixels(image), (NARROW_X, ROW), background)
    wide = grow_box(s.pixels(image), (WIDE_X, ROW), background)
    first = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    ops.set_box_padding(s, protein, across=1, along=2)
    assert band_of(s, first).box.rect(boxes_of(s, protein).box_size) == grown_by(narrow, 1, 2)

    # A wider band grows the fitted size to its own, and the padding stays.
    second = ops.place_box(s, protein, WIDE_X, ROW, lane_index=1, grow=True)
    beta = boxes_of(s, protein)
    fitted = BoxSize(width=wide[2] - wide[0], height=wide[3] - wide[1])
    assert fitted.width > narrow[2] - narrow[0]
    assert (beta.fitted_size, beta.box_padding) == (fitted, BoxPadding(across=1, along=2))
    assert band_of(s, second).box.rect(beta.box_size) == grown_by(wide, 1, 2)
    assert not band_of(s, first).manually_edited
    assert_nets_current(s)

    # The first box after a clear sets the fitted size afresh, padded.
    ops.clear_boxes(s, protein)
    third = ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    beta = boxes_of(s, protein)
    assert beta.fitted_size == BoxSize(width=narrow[2] - narrow[0], height=narrow[3] - narrow[1])
    assert band_of(s, third).box.rect(beta.box_size) == grown_by(narrow, 1, 2)


def test_a_fixed_box_takes_the_padded_size(tmp_path):
    s, _, protein = boxed(tmp_path)
    ops.place_box(s, protein, NARROW_X, ROW, lane_index=0, grow=True)
    ops.set_box_padding(s, protein, along=2)
    size = boxes_of(s, protein).box_size
    band_id = ops.place_box(s, protein, 70, ROW, lane_index=1, grow=False)
    assert band_of(s, band_id).box.rect(size) == boxes.centered_rect(70, ROW, size, W, H)
    assert boxes_of(s, protein).box_size == size == BoxSize(width=9, height=9)
    assert_nets_current(s)


def test_row_refusal_names_the_padding_not_kept_boxes(tmp_path):
    # One box, grown where the row finds a band (so it gives way to the row:
    # nothing survives), padded 12 px left and right: the row's boxes, 44 + 24
    # px wide, would overlap each other in lanes 67 px apart.
    s, _, protein = row_session(tmp_path, PAD_CASE)
    ops.place_box(s, protein, *at_lane(PAD_CASE, 2), lane_index=2, grow=True)
    ops.set_box_size(s, protein, BoxSize(width=44, height=12))
    ops.set_box_padding(s, protein, across=12)
    committed = s.project
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, protein, PAD_CASE.row)
    assert (info.value.code, str(info.value)) == (
        ErrorCode.SIZE_WOULD_OVERLAP,
        "the row's boxes would overlap each other at the box size 68x12 (fitted 44x12 plus"
        " 12 px left and right); lower the padding left and right",
    )
    assert s.project is committed
    # Lowered to what fits, the row places its boxes, padded.
    ops.set_box_padding(s, protein, across=11)
    ops.detect_row_boxes(s, protein, PAD_CASE.row)
    assert boxes_of(s, protein).box_size == BoxSize(width=66, height=12)
    assert len(boxes_of(s, protein).bands) == PAD_CASE.n_lanes


def _padded_kept_pair(s: ProjectSession, ids: dict[str, str]) -> None:
    """The box :func:`_row_scene` clicked, edited by hand, and a box of a second
    band (#58) about 49 px to its left, in the lane before: a row keeps both.
    Padded 4 px left and right: the detected width fits between their centres,
    and fits with 2 px of padding, not with 4."""
    ops.set_box_size(s, ids["protein"], BoxSize(width=10, height=8))
    x, y = at_lane(SCENE, 1)
    near = ops.place_box(s, ids["protein"], x - 48, y, lane_index=0, grow=False)

    def second_band(draft: Project) -> None:
        protein, band = draft.batch.find_band(near)
        protein.expected_band_count = 2
        band.band_index = 1

    plant(s, second_band)
    edit_by_hand(s, ids["band"])
    ops.set_box_padding(s, ids["protein"], across=4)


def test_a_row_refused_for_its_kept_boxes_names_the_padding(tmp_path):
    s, recorder, ids = _row_scene(tmp_path, _padded_kept_pair)
    committed = s.project
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, ids["protein"], SCENE.row)
    assert (info.value.code, str(info.value)) == (
        ErrorCode.SIZE_WOULD_OVERLAP,
        "box size 52x12 (fitted 44x12 plus 4 px left and right) would make the boxes of"
        " 'β-catenin' that the row keeps overlap; lower the padding left and right",
    )
    assert s.project is committed
    assert recorder.actions == []
    # Lowered as it asks, the row keeps both boxes and places the others, padded.
    kept = [band.id for band in boxes_of(s, ids["protein"]).bands]
    ops.set_box_padding(s, ids["protein"], across=2)
    ops.detect_row_boxes(s, ids["protein"], SCENE.row)
    beta = boxes_of(s, ids["protein"])
    assert beta.box_size == BoxSize(width=48, height=12)
    assert set(kept) < {band.id for band in beta.bands}
    assert_nets_current(s)


def _step_pad(s: ProjectSession, rng: np.random.Generator, case: RowCase) -> None:
    """A padding of 0 to 3 px each way on a protein with boxes."""
    batch = s.project.batch
    protein = _pick(rng, [p.id for p in batch.proteins if batch.boxes_of(p).bands])
    across, along = (int(v) for v in rng.integers(0, 4, size=2))
    ops.set_box_padding(s, protein, across=across, along=along)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_edits_keep_the_padding_and_the_nets(tmp_path, seed):
    """Property: with padding changes among the edits of
    :func:`test_random_edits_keep_every_stored_value_what_the_pixels_give`,
    refused or not, undone or redone, only a padding change or a restore
    changes a protein's padding (fits, resizes and clears keep it), and every
    stored value stays what the pixels give."""
    s, image, target = row_session(tmp_path, PAD_CASE)
    other = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    ops.detect_row_boxes(s, target, PAD_CASE.row)
    for lane in (1, 3):  # below the target's bands: two proteins' boxes on one band are refused
        x, y = at_lane(PAD_CASE, lane)
        ops.place_box(s, other, x, y + 12, lane_index=lane, grow=False)
    steps = {**PROPERTY_STEPS, "set_box_padding": _step_pad}
    rng = np.random.default_rng(seed)
    names = sorted(steps)
    schedule = [*names * 3, *(names[int(i)] for i in rng.integers(len(names), size=24))]
    rng.shuffle(schedule)
    for name in schedule:
        before = s.project
        paddings = {p.id: before.batch.boxes_of(p).box_padding for p in before.batch.proteins}
        try:
            steps[name](s, rng, PAD_CASE)
        except (OperationError, UnknownIdError):
            assert s.project is before, name  # a refusal changes nothing
        if name not in {"set_box_padding", "undo", "redo"}:
            for protein in s.project.batch.proteins:
                padding = s.project.batch.boxes_of(protein).box_padding
                assert padding == paddings.get(protein.id, BoxPadding()), name
        assert s.project.log[-1].content_hash == content_hash(s.project), name
        assert_nets_current(s)
    padded = [e for e in s.project.log if e.action == "set_box_padding"]
    assert any(e.params["box_padding"] != {"across": 0, "along": 0} for e in padded)


# --- Molecular weights (#58): the one writer ---

# A two-ladder blot: a visible-light marker with a ladder by each side, and a
# chemiluminescence image of the same membrane with one row of four lanes, both
# 16-bit, CAL_H x CAL_W, with seeded noise. A band of m kDa lies at cal_y(m, x):
# log-linear in the MW, and CAL_TILT px lower at the right ladder than at the
# left one (a tilted blot), so the protein line slopes across the lanes.
CAL_H, CAL_W = 200, 480
CAL_LEFT_X, CAL_RIGHT_X = 30.0, 450.0
CAL_KDA = (250, 130, 100, 70, 55, 35, 25, 15)
CAL_TILT = 6.0
CAL_LANES = (120, 200, 280, 360)
CAL_ROW = 100  # the row's bands peak on this row: at y = 100.5
CAL_ROW_BOX = (80, 85, 400, 116)
MARKER_BAND = CalibrationPointSource.VISIBLE_MARKER
LEFT, RIGHT = LadderSide.LEFT, LadderSide.RIGHT


def cal_x(side: LadderSide) -> float:
    return CAL_LEFT_X if side is LEFT else CAL_RIGHT_X


def cal_y(kda: float, x: float) -> float:
    """Where a band of ``kda`` kDa lies at column ``x``, in continuous y."""
    t = (x - CAL_LEFT_X) / (CAL_RIGHT_X - CAL_LEFT_X)
    return 20.0 + 120.0 * math.log10(250.0 / kda) + CAL_TILT * t


def cal_mw(x: float, y: float) -> float:
    """The MW the blot's law puts at ``(x, y)``: what its two ladders read there."""
    t = (x - CAL_LEFT_X) / (CAL_RIGHT_X - CAL_LEFT_X)
    return 250.0 / 10 ** ((y - 20.0 - CAL_TILT * t) / 120.0)


def noisy16(image: np.ndarray, seed: int) -> np.ndarray:
    """``image`` with seeded Gaussian noise of 60 counts, as 16-bit pixels."""
    noisy = image + np.random.default_rng(seed).normal(0.0, 60.0, image.shape)
    return np.clip(np.round(noisy), 0, 65535).astype(np.uint16)


def cal_marker(kda: Sequence[float] = CAL_KDA) -> np.ndarray:
    """The marker: each ladder's bands at ``cal_y`` (a band on row r's centre
    lies at r + 0.5, so its index centre is half a row higher)."""
    bands = [
        (x, cal_y(m, x) - 0.5, 10.0, 2.0, 20000.0) for x in (CAL_LEFT_X, CAL_RIGHT_X) for m in kda
    ]
    return noisy16(synthetic_blot((CAL_H, CAL_W), bands, dtype=np.float64), 58)


def cal_blot() -> np.ndarray:
    """The chemiluminescence image: one row of bands, one in each lane."""
    bands = [(x, CAL_ROW, 8.0, 3.0, 30000.0) for x in CAL_LANES]
    return noisy16(synthetic_blot((CAL_H, CAL_W), bands, dtype=np.float64), 59)


@dataclasses.dataclass(frozen=True)
class Calibrated:
    session: ProjectSession
    membrane: str
    blot: str
    marker: str
    protein: str


def calibrated(tmp_path: Path, hook=None, *, sides=(LEFT, RIGHT)) -> Calibrated:
    """The two-ladder blot imported (the blot, then its marker on the same
    membrane) and linked, the PageRuler Plus preset chosen, every CAL_KDA band
    of the ladders ``sides`` marked where it lies (not snapped), four lanes
    declared and one target without boxes."""
    s = session_on(tmp_path, hook)
    blot_id = import_blot(s, cal_blot(), "blot β.tif")
    membrane = s.project.batch.membrane_of(blot_id).id
    marker = import_blot(
        s, cal_marker(), "marker α.tif", kind=ImageKind.VISIBLE_MARKER, membrane_id=membrane
    )
    ops.set_marker_image(s, blot_id, marker)
    ops.set_ladder(s, membrane, "pageruler_plus/tris_glycine")
    for side in sides:
        x = cal_x(side)
        for kda in CAL_KDA:
            ops.add_calibration_point(
                s, marker, cal_y(kda, x), kda, MARKER_BAND, x=x, side=side, snap=False
            )
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(len(CAL_LANES))])
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, blot_id, expected_mw=50)
    return Calibrated(s, membrane, blot_id, marker, protein)


# The operations that can change an apparent MW or a membrane's fit, each of
# which calls the one writer (_refresh_mw). Every other function of
# ops.__all__ reads, or changes nothing an MW or a fit depends on: a box removed
# takes its MW along, and a lane, a name, a polarity, a channel or a record is
# not read.
MW_WRITERS = frozenset(
    {
        "place_box",
        "move_box",
        "set_box_size",
        "set_box_padding",
        "detect_row_boxes",
        "detect_mw_row",
        "remove_image",
        "set_marker_image",
        "set_ladder",
        "add_calibration_point",
        "edit_calibration_point",
        "remove_calibration_point",
        "clear_calibration",
        "set_ladder_points",
    }
)
MW_NEUTRAL = frozenset(
    {
        "add_protein",
        "calibration_fit",
        "clear_boxes",
        "compute",
        "compute_view",
        "edit_protein",
        "export_bundle",
        "export_lane_table",
        "import_image",
        "new_project",
        "open_project",
        "proposal_json",
        "propose_ladder",
        "redo",
        "remove_box",
        "remove_protein",
        "remove_undetected",
        "requantify",
        "save",
        "set_box_lane",
        "set_channel",
        "set_lanes",
        "set_polarity",
        "set_reference_condition",
        "snap_ladder",
        "unassessed_images",
        "undo",
    }
)


def test_every_operation_is_classified_for_the_mw_writer():
    # A new operation must be classified: if it can move a box or change a
    # calibration, it calls the writer and joins test_mw_current_after_each_operation.
    functions = {name for name in ops.__all__ if inspect.isfunction(getattr(ops, name))}
    assert MW_WRITERS | MW_NEUTRAL == functions
    assert not MW_WRITERS & MW_NEUTRAL
    writers = ("_refresh_mw(", "_refresh_box_mws(", "_calibration_change(", "_place_row(")
    for name in MW_WRITERS:
        source = inspect.getsource(getattr(ops, name))
        assert any(call in source for call in writers), name
    # The commit every row shares (detect_row_boxes, detect_mw_row) calls it.
    assert "_refresh_box_mws(" in inspect.getsource(ops._place_row)


def test_mw_current_after_each_operation(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    assert_mw_current(s)
    ran: set[str] = set()

    def step(name: str, call) -> None:
        before = s.project
        call()
        assert s.project is not before, name  # each step commits a change
        ran.add(name)
        assert_mw_current(s)
        assert_nets_current(s)

    lanes = CAL_LANES
    step(
        "place_box", lambda: ops.place_box(s, c.protein, lanes[0], CAL_ROW, lane_index=0, grow=True)
    )
    step(
        "place_box",
        lambda: ops.place_box(s, c.protein, lanes[1], CAL_ROW, lane_index=1, grow=False),
    )
    step("detect_row_boxes", lambda: ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX))
    assert all(band.apparent_mw is not None for band in boxes_of(s, c.protein).bands)
    # The same row placed by its expected MW, between the ladders: in place, MW-guided.
    step("detect_mw_row", lambda: ops.detect_mw_row(s, c.protein))
    assert {band.source for band in boxes_of(s, c.protein).bands} == {ProposalSource.MW_GUIDED}
    moved = lane_bands(s, c.protein)[2]
    x0, y0, x1, y1 = moved.box.rect(boxes_of(s, c.protein).box_size)
    step("move_box", lambda: ops.move_box(s, moved.id, (x0 + 2, y0 + 3, x1 + 2, y1 + 3)))
    fitted = boxes_of(s, c.protein).fitted_size
    size = BoxSize(width=fitted.width + 2, height=fitted.height + 3)
    step("set_box_size", lambda: ops.set_box_size(s, c.protein, size))
    step("set_box_padding", lambda: ops.set_box_padding(s, c.protein, across=1, along=2))
    ten = cal_y(10, CAL_LEFT_X)
    step(
        "add_calibration_point",
        lambda: ops.add_calibration_point(
            s, c.marker, ten, 10, MARKER_BAND, x=CAL_LEFT_X, snap=False
        ),
    )
    # Named by the other image of its register group.
    step(
        "edit_calibration_point",
        lambda: ops.edit_calibration_point(s, c.blot, 100, y=cal_y(100, CAL_LEFT_X) + 1.5),
    )
    step("remove_calibration_point", lambda: ops.remove_calibration_point(s, c.marker, 10))
    step("set_ladder", lambda: ops.set_ladder(s, c.membrane, "our ladder", kda=[250, 100, 15]))
    step("set_marker_image", lambda: ops.set_marker_image(s, c.blot, None))
    assert all(band.apparent_mw is None for band in boxes_of(s, c.protein).bands)
    step("set_marker_image", lambda: ops.set_marker_image(s, c.blot, c.marker))
    step("clear_calibration", lambda: ops.clear_calibration(s, c.marker, side=RIGHT))
    right = [(cal_y(kda, CAL_RIGHT_X), kda) for kda in CAL_KDA[:5]]
    step(
        "set_ladder_points",
        lambda: ops.set_ladder_points(s, c.marker, RIGHT, right, x=CAL_RIGHT_X),
    )
    step("remove_image", lambda: ops.remove_image(s, c.marker))
    assert ran == MW_WRITERS
    calibration = s.project.batch.membranes[0].calibration
    assert (calibration.points, calibration.fit_method, calibration.fit_quality) == (
        [],
        FitMethod.LOG_LINEAR,
        None,
    )


def test_two_ladder_apparent_mw_uses_box_x(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX)
    size = boxes_of(s, c.protein).box_size
    mws = []
    for _, band in sorted(lane_bands(s, c.protein).items()):
        cx, cy = band.box.x + size.width / 2, band.box.y + size.height / 2
        # Two ladders read the blot's law exactly: log-linear, linear in x.
        assert math.isclose(band.apparent_mw, cal_mw(cx, cy), rel_tol=1e-9), band.id
        mws.append(band.apparent_mw)
    # The row lies level while the protein line slopes down to the right, so at
    # one height a lane further right reads heavier.
    assert mws == sorted(mws) and len(set(mws)) == len(mws)
    fit = ops.calibration_fit(s.project.batch, s.project.batch.membranes[0], c.blot)
    assert fit is not None and fit.image_ids == (c.blot, c.marker)
    assert [ladder.side for ladder in fit.ladders] == ["left", "right"]
    assert math.isclose(fit.offset_px, CAL_TILT, rel_tol=1e-9)
    assert math.isclose(fit.tilt_deg, math.degrees(math.atan2(CAL_TILT, 420.0)), rel_tol=1e-9)
    # One ladder: the line is level, and every lane reads one MW at one height.
    ops.clear_calibration(s, c.blot, side=RIGHT)
    assert len({band.apparent_mw for band in boxes_of(s, c.protein).bands}) == 1
    assert_mw_current(s)


def test_set_box_padding_refreshes_mw(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX)
    band = lane_bands(s, c.protein)[1].id
    plant(s, lambda draft: setattr(draft.batch.find_band(band)[1], "apparent_mw", 1.0))
    ops.set_box_padding(s, c.protein, across=2, along=3)
    assert band_of(s, band).apparent_mw == expected_mw(s, band) != 1.0
    assert_mw_current(s)
    # A box the image's edge keeps from growing evenly reads the MW at its
    # centre, which moved (edge_shifted): at the top of the image, y 0 to 20,
    # inside the range (which ends a tenth of a decade above 250 kDa, at y 8).
    top = ops.add_protein(
        s, "GAPDH", Role.LOADING_CONTROL, c.blot, box_size=BoxSize(width=10, height=20)
    )
    edge = ops.place_box(s, top, CAL_LANES[0], 10, lane_index=0, grow=False)
    before = band_of(s, edge).apparent_mw
    assert before is not None
    change = ops.set_box_padding(s, top, along=2)
    assert change.edge_shifted == (edge,)
    assert band_of(s, edge).box.y == 0  # 24 px high from the top: its centre 2 px lower
    assert band_of(s, edge).apparent_mw == expected_mw(s, edge) < before
    assert_mw_current(s)


def test_set_box_size_keeps_apparent_mw(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX)
    before = {b.id: b.apparent_mw for b in boxes_of(s, c.protein).bands}
    fitted = boxes_of(s, c.protein).fitted_size
    # Each box grows by 2 px each way around its centre: the same centre, the same MW.
    ops.set_box_size(s, c.protein, BoxSize(width=fitted.width + 4, height=fitted.height + 4))
    assert {b.id: b.apparent_mw for b in boxes_of(s, c.protein).bands} == before
    # One more row of height moves each centre by half a pixel: recomputed.
    ops.set_box_size(s, c.protein, BoxSize(width=fitted.width + 4, height=fitted.height + 5))
    after = {b.id: b.apparent_mw for b in boxes_of(s, c.protein).bands}
    assert all(after[band_id] != mw for band_id, mw in before.items())
    assert_mw_current(s)


def reprobe_on(c: Calibrated, name: str = "reprobe α.tif") -> str:
    """Another chemiluminescence image of the blot's membrane, unlinked: a
    register group of its own."""
    return import_blot(c.session, cal_blot(), name, membrane_id=c.membrane)


def test_curve_change_drops_that_images_mw_guided_records_only(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    reprobe = reprobe_on(c)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, reprobe)
    guided = _record(1, region=(180, 90, 220, 110), source=ProposalSource.MW_GUIDED)
    boxed_row = _record(2, region=(260, 90, 300, 110))
    plant_records(s, c.protein, guided, boxed_row)
    plant_records(s, gapdh, guided)

    update = ops.edit_calibration_point(s, c.marker, 55, y=cal_y(55, CAL_LEFT_X) + 1.0)
    assert update.curves_changed == (c.blot, c.marker)  # not the reprobe: its own group
    assert update.dropped_undetected == ((c.protein, 1, 0),)
    assert _keys(s, c.protein) == [(2, 0)]  # a row-box record stays
    assert _keys(s, gapdh) == [(1, 0)]  # the reprobe's curve (none) did not change
    entry = s.project.log[-1]
    assert entry.params["curves_changed"] == [c.blot, c.marker]
    assert entry.params["dropped_undetected"] == [_record_json(c.protein, guided)]
    assert entry.content_hash == content_hash(s.project)

    # A change that leaves the curve as it was, a ladder chosen, drops nothing.
    plant_records(s, c.protein, guided)
    ops.set_ladder(s, c.membrane, None)
    assert _keys(s, c.protein) == [(1, 0), (2, 0)]


def test_remove_image_refits_remaining_groups(tmp_path):
    # Removing an image with points refits what is left (#58); before, the
    # fit was cleared and every MW on the membrane dropped.
    c = calibrated(tmp_path, sides=(LEFT,))
    s = c.session
    reprobe = reprobe_on(c)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, reprobe)
    ops.place_box(s, gapdh, CAL_LANES[0], CAL_ROW, lane_index=0, grow=True)
    # The reprobe's own marks, log-linear: a quarter of the MW every 70 px.
    for kda, y in ((160, 30.0), (40, 100.0), (10, 170.0)):
        ops.add_calibration_point(
            s, reprobe, y, kda, CalibrationPointSource.STRIP_EDGE, x=5.0, snap=False
        )
    ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX)
    # A faint marker band on the blot, 3 px off the law: the worst point.
    y10 = cal_y(10, CAL_LEFT_X) + 3.0
    ops.add_calibration_point(
        s,
        c.blot,
        y10,
        10,
        CalibrationPointSource.CHEMILUMINESCENCE_MARKER,
        x=CAL_LEFT_X,
        snap=False,
    )
    reprobe_mw = band_of(s, boxes_of(s, gapdh).bands[0].id).apparent_mw
    before = s.project.batch.membranes[0].calibration.fit_quality

    cascade = ops.remove_image(s, c.blot)
    assert cascade.unfitted_membranes == (c.membrane,)
    calibration = s.project.batch.membranes[0].calibration
    assert calibration.fit_method is FitMethod.LOG_LINEAR_PIECEWISE
    assert (
        calibration.fit_quality
        == mwcal.fit_quality(s.project.batch, s.project.batch.membranes[0])
        != before
    )
    assert len(calibration.points) == len(CAL_KDA) + 3  # the marker's and the reprobe's
    # The reprobe's curve did not change: its MW is the one it had.
    assert band_of(s, boxes_of(s, gapdh).bands[0].id).apparent_mw == reprobe_mw
    assert s.project.log[-1].params["dropped_undetected"] == []
    assert_mw_current(s)


def test_refitting_a_pre_58_calibration_rewrites_every_mw_on_its_membrane(tmp_path):
    # mem-1 of the sample is stored as no #58 build fits (log_linear, a fit
    # quality and band-12's MW from an older fit). A point on the reprobe, a
    # register group of its own, refits the membrane; the curve of img-2 (read
    # piecewise all along) does not change, but its stored MW was not computed
    # from it, so every image of the membrane counts as changed.
    s = open_sample(tmp_path, hook=None)
    guided = _record(2, source=ProposalSource.MW_GUIDED)
    plant_records(s, "prot-7", guided)
    legacy = s.project
    stored = band_of(s, "band-12").apparent_mw
    assert stored == 90.5 != expected_mw(s, "band-12")

    first = ops.add_calibration_point(
        s, "img-4", 5.0, 200, CalibrationPointSource.STRIP_EDGE, x=10.0, snap=False
    )
    assert first.curves_changed == ("img-2", "img-3", "img-4")  # not img-6: mem-5
    assert first.dropped_undetected == (("prot-7", 2, 0),)
    mem_1, mem_5 = s.project.batch.membranes
    assert mem_1.calibration.fit_method is FitMethod.LOG_LINEAR_PIECEWISE
    assert mem_1.calibration.fit_quality == mwcal.fit_quality(s.project.batch, mem_1)
    assert band_of(s, "band-12").apparent_mw == expected_mw(s, "band-12") != stored
    assert _keys(s, "prot-7") == []
    # mem-5, untouched, keeps what it holds.
    assert mem_5 == legacy.batch.membranes[1]

    # Refitted by #58 now: the next point changes only the curve it makes.
    second = ops.add_calibration_point(
        s, "img-4", 140.0, 20, CalibrationPointSource.STRIP_EDGE, x=10.0, snap=False
    )
    assert second.curves_changed == ("img-4",)
    assert band_of(s, "band-12").apparent_mw == expected_mw(s, "band-12")
    ops.undo(s)
    ops.undo(s)
    assert s.project.batch == legacy.batch


def test_a_first_curve_changes_only_its_group(tmp_path):
    # A membrane a #58 build fitted holds log_linear only while no image of it
    # has a curve (one point, here): its first curve is no refit of an older fit.
    c = calibrated(tmp_path, sides=())
    s = c.session
    reprobe_on(c)
    ops.add_calibration_point(
        s, c.marker, cal_y(100, CAL_LEFT_X), 100, MARKER_BAND, x=CAL_LEFT_X, snap=False
    )
    update = ops.add_calibration_point(
        s, c.marker, cal_y(35, CAL_LEFT_X), 35, MARKER_BAND, x=CAL_LEFT_X, snap=False
    )
    assert update.curves_changed == (c.blot, c.marker)
    assert_mw_current(s)


def test_undo_restores_calibration_and_mws(tmp_path):
    c = calibrated(tmp_path, save_to_folder)
    s = c.session
    ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX)
    before = s.project
    ops.edit_calibration_point(s, c.marker, 70, side=RIGHT, y=cal_y(70, CAL_RIGHT_X) + 2.0)
    after = s.project
    assert after.batch.membranes != before.batch.membranes
    assert [b.apparent_mw for b in boxes_of(s, c.protein).bands] != [
        b.apparent_mw for b in before.batch.boxes_of(before.batch.find_protein(c.protein)).bands
    ]
    done = ops.undo(s)
    assert (done.action, s.project.batch) == ("edit_calibration_point", before.batch)
    assert_mw_current(s)
    ops.redo(s)
    assert s.project.batch == after.batch
    assert load_project(s.folder) == s.project
    assert_mw_current(s)
