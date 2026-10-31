# SPDX-License-Identifier: Apache-2.0
"""Golden schema-2 projects (#221): built through the operations at 05b4af5,
with what they store and what the build derives from them.

Each case folder (``f1``, ``f2``, ``f3``) holds a project as 05b4af5 wrote it
(``project.json``, ``images/``), ``stored.json`` and ``derived.json``.

* ``f1``: two membranes. Membrane A holds two 16-bit exposures (E1, E2, dark
  on light) and an 8-bit PNG visible marker (M) linked to E1, so it has two
  register groups ({E1, M} and {E2}); M carries a preset ladder marked three
  times on each side, the right ladder 3 px lower (a tilt), and E2 two faint
  chemiluminescence-marker points (a two-point curve). Membrane B holds one
  16-bit exposure (E3, light on dark, another size) with two strip-edge
  points. Eight lanes: ``vehicle`` three times with a technical repeat,
  ``10 µM`` three times with one lane excluded, ``50 µM`` twice; the
  reference is ``vehicle``. The target ``β-catenin`` on E1 is placed by a
  dragged row (detector boxes, bands found, one lane not detected) and has an
  expected MW; the loading control ``α-tubulin`` on E3 (the other membrane)
  by seed clicks, with a padding of 2 px across and 1 px along; the target
  ``p53 µ`` on E2 by its expected MW (one lane not detected); the loading
  control ``GAPDH`` on E1 by fixed boxes in chosen lanes, one of them
  over-exposed. Adding GAPDH names α-tubulin in β-catenin; ``p53 µ`` names
  GAPDH.
* ``f2``: a JPEG exposure (a band possibly over-exposed) and a colour PNG
  whose channels differ (its bands not checked for clipping), on two
  membranes; the target ``β-catenin`` on the JPEG, the loading control
  ``α-tubulin`` and the target ``p53 µ`` on the PNG, where one box of each
  shares pixels with the other (under half of a box).
* ``f3``: a history with a removed image, kept small: a 16-bit exposure (S)
  linked to an 8-bit PNG marker (N) with a ladder of three points, and a
  16-bit reprobe (R) on a membrane of its own; four lanes; the target
  ``β-catenin`` on S by a dragged row, with an expected MW, and the loading
  control ``α-tubulin`` on R by fixed boxes. Then N is removed, which unlinks
  S and takes the ladder and with it β-catenin's apparent MWs; the removal is
  undone, redone and undone again, so the undo and redo entries check out.

``stored.json`` holds every value the project stores but its schema version
(:func:`stored_values`), under the names of the schema-2 file: the background
method and the next id; each lane (index, label, sample, inclusion,
metadata) and the reference; each membrane's images (file, names, kind,
polarity, marker link, hash, recorded size, depth, background, and each
import warning's code and message) and calibration (ladder, points, fit
method and quality); each protein's settings and boxes, with every band's
net, background level, mode and spread, clipping flags, apparent MW and band
count, and its not-detected records with their reasons; and every log entry
whole. It also records what it was written from (:data:`FROZEN_ONLY`): its
case, the commit and version that built it, the SHA-256 of the frozen
``project.json`` and the content hash at schema 2. It was written once, at
05b4af5 (``built_at``), and is never written again: ``--write-frozen``
refuses at any other commit or with any other build or installed
dependencies, and a change that would need another ``stored.json`` changes
what old projects store. The stored values are compared by value
(:func:`stored_differences`), never through a hash of the content in the
build's own shape, so a project held in another shape with every value
intact still matches.

``derived.json`` holds what the build under test derives from the frozen
project (:func:`derived_output`): the results with their notices, and with
``10 µM`` plotted alone what that changes (the notices and each chart); each
result set's lane table, as an export writes it; each register group's
calibration fit, as the page shows it; the history check; and how the
values a fresh build of the same project from the same image files stores
differ from the frozen ones (:func:`build`, :func:`fresh_changes`: not at
all, while the operations keep their numbers). It describes current
behaviour. A pull request that means to change it runs, at its own commit,

    uv run python tests/data/schema2/make_fixtures.py --refresh-derived

which rewrites only the ``derived.json`` files, commits the diff and says why
in its description. ``derive.py`` prints a fixture's derived output as
canonical JSON, to compare two builds exactly on one machine.

The images are drawn here from fixed numbers with additions, multiplications
and divisions only, and integer noise from fixed seeds, so their pixels are
the same on every platform. The fresh build imports the frozen image files
(an import copies them byte for byte), so no encoder takes part in it.

Frozen at 05b4af5 by copying this file and ``derive.py`` into a checkout of
that commit, synced with ``uv sync --locked``, and running ``--write-frozen``
there; ``--out DIR`` writes elsewhere, to rebuild them for a comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import tifffile
from PIL import Image

import proteia
from proteia.core import export, record, results, storage
from proteia.core import operations as ops
from proteia.core.model import (
    Band,
    BoxSize,
    CalibrationPointSource,
    ImageKind,
    ImageRef,
    LadderSide,
    Lane,
    LogEntry,
    Membrane,
    Polarity,
    Project,
    Protein,
    Role,
    UndetectedBand,
)
from proteia.core.operations import LaneInput

HERE = Path(__file__).resolve().parent
FROZEN_COMMIT = "05b4af5b08e51b450436e9446651f5650bff48e4"
CASES = ("f1", "f2", "f3")
REFRESH_COMMAND = "uv run python tests/data/schema2/make_fixtures.py --refresh-derived"
# What stored.json records about the fixture rather than a value its project
# stores: the case, the commit and version that built it, the SHA-256 of its
# project.json, and its content hash at schema 2 (a hash of the content in the
# schema-2 shape, which a change of shape changes whole).
FROZEN_ONLY = ("case", "built_at", "version", "project_sha256", "content_hash")
# What a fresh build's log entries leave out: the content hash, which a
# last-bit difference of any float changes whole, and the version, which a
# release changes with no change of behaviour.
NOT_IN_FRESH_LOG = ("content_hash", "version")
REL_TOL = 1e-12  # derived floats, relative
PLOTTED = "10 µM"

CHEMI = ImageKind.CHEMILUMINESCENCE
MARKER = ImageKind.VISIBLE_MARKER
DARK = Polarity.DARK_ON_LIGHT
LIGHT = Polarity.LIGHT_ON_DARK
MARKER_BAND = CalibrationPointSource.VISIBLE_MARKER
FAINT_MARKER = CalibrationPointSource.CHEMILUMINESCENCE_MARKER
STRIP_EDGE = CalibrationPointSource.STRIP_EDGE


class Clock:
    """Aware UTC time from 2026-10-31T09:00:00Z, one second later on each call."""

    def __init__(self) -> None:
        self.now = datetime(2026, 10, 31, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        now, self.now = self.now, self.now + timedelta(seconds=1)
        return now


# --- The images ---


def _noise(shape: tuple[int, int], seed: int, amplitude: int) -> np.ndarray:
    """Integer noise: the sum of three uniform draws in [-amplitude, amplitude]."""
    rng = np.random.default_rng(seed)
    total = np.zeros(shape, dtype=np.int64)
    for _ in range(3):
        total += rng.integers(-amplitude, amplitude + 1, size=shape)
    return total


def _bands(shape: tuple[int, int], bands: Sequence[tuple[float, float, float, float, float]]):
    """The summed signal of ``bands``, each ``(cx, cy, half width, half height,
    depth)``: flat-topped across, softer down, with no function but + - * /."""
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width].astype(float)
    total = np.zeros(shape)
    for cx, cy, wx, hy, depth in bands:
        u = (xs + 0.5 - cx) / wx
        v = (ys + 0.5 - cy) / hy
        u2, v2 = u * u, v * v
        total += depth / ((1.0 + u2 * u2 * u2) * (1.0 + v2 * v2))
    return total


def _as(values: np.ndarray, dtype: type) -> np.ndarray:
    info = np.iinfo(dtype)
    return np.clip(np.round(values), info.min, info.max).astype(dtype)


# f1, membrane A (E1, E2, M: 160 x 80 px). Lanes 16 px apart; the ladders at
# x = 8 (left) and 152 (right), the right one 3 px lower; a band of m kDa lies
# at y = 10 + 91.247 log10(130 / m) on the left ladder.
A_SHAPE = (80, 160)
A_LANES = tuple(24.0 + 16.0 * i for i in range(8))
A_TILT = 3.0 / 144.0  # px down per px right
LEFT_X, RIGHT_X = 8.0, 152.0
LADDER = ((130.0, 10.0), (70.0, 34.53), (35.0, 62.0))  # (kDa, y on the left ladder)
BETA_Y, GAPDH_Y = 23.7, 60.88  # 92 and 36 kDa on the left ladder
E2_POINTS = ((100.0, 15.0), (25.0, 65.0))  # (kDa, y)
P53_Y = 37.9  # 53 kDa on E2's two-point curve
# f1, membrane B (E3: 150 x 72 px, light on dark).
B_SHAPE = (72, 150)
B_LANES = tuple(20.0 + 16.0 * i for i in range(8))
ALPHA_Y = 36.0
STRIP_EDGES = ((70.0, 8.0), (35.0, 64.0))  # (kDa, y)
# f2 (120 x 80 px), six lanes.
C_SHAPE = (80, 120)
C_LANES = tuple(15.0 + 18.0 * i for i in range(6))
JPEG_Y, ALPHA2_Y, P53_2_Y = 30.0, 25.0, 50.0
# In this lane p53 µ runs higher, and its box, placed above its band, shares a
# few rows of pixels with α-tubulin's.
OVERLAP_LANE, OVERLAP_BAND_Y, OVERLAP_Y = 3, 38.0, 32
# f3 (S, N, R: 96 x 48 px), four lanes; the ladder at x = 12 on N, a band of
# m kDa at y = 8 + 70.19 log10(130 / m).
D_SHAPE = (48, 96)
D_LANES = tuple(32.0 + 16.0 * i for i in range(4))
D_LADDER = ((130.0, 8.0), (70.0, 26.87), (35.0, 47.0))  # (kDa, y)
D_BETA_Y, D_ALPHA_Y = 18.54, 28.0  # 92 kDa on the ladder; α-tubulin on R


def _tilted(x: float, y: float) -> float:
    return y + (x - LEFT_X) * A_TILT


def _e1() -> np.ndarray:
    beta = (14000, 16000, 12000, 18000, 20000, 15000, 0, 17000)  # lane 7: no band
    gapdh = (22000, 21000, 23000, 80000, 22000, 20000, 21500, 22500)  # lane 4: over-exposed
    bands = [(x, _tilted(x, BETA_Y), 5.5, 4.0, d) for x, d in zip(A_LANES, beta, strict=True) if d]
    bands += [(x, _tilted(x, GAPDH_Y), 5.5, 4.0, d) for x, d in zip(A_LANES, gapdh, strict=True)]
    return _as(50000.0 - _bands(A_SHAPE, bands) + _noise(A_SHAPE, 11, 60), np.uint16)


def _e2() -> np.ndarray:
    p53 = (9000, 11000, 10000, 12000, 13000, 0, 9500, 10500)  # lane 6: no band
    bands = [(x, P53_Y, 5.5, 4.0, d) for x, d in zip(A_LANES, p53, strict=True) if d]
    return _as(48000.0 - _bands(A_SHAPE, bands) + _noise(A_SHAPE, 12, 60), np.uint16)


def _m() -> np.ndarray:
    bands = [(LEFT_X, y, 4.0, 2.5, 120.0) for _, y in LADDER]
    bands += [(RIGHT_X, y + 3.0, 4.0, 2.5, 120.0) for _, y in LADDER]
    return _as(205.0 - _bands(A_SHAPE, bands) + _noise(A_SHAPE, 13, 2), np.uint8)


def _e3() -> np.ndarray:
    alpha = (15000, 16000, 14500, 15500, 16500, 14000, 15800, 15200)
    bands = [(x, ALPHA_Y, 4.5, 4.0, d) for x, d in zip(B_LANES, alpha, strict=True)]
    return _as(3000.0 + _bands(B_SHAPE, bands) + _noise(B_SHAPE, 14, 60), np.uint16)


def _s() -> np.ndarray:
    beta = (15000, 13000, 19000, 17000)
    bands = [(x, D_BETA_Y, 5.5, 4.0, d) for x, d in zip(D_LANES, beta, strict=True)]
    return _as(50000.0 - _bands(D_SHAPE, bands) + _noise(D_SHAPE, 19, 60), np.uint16)


def _n() -> np.ndarray:
    bands = [(12.0, y, 4.0, 2.5, 120.0) for _, y in D_LADDER]
    return _as(205.0 - _bands(D_SHAPE, bands) + _noise(D_SHAPE, 20, 2), np.uint8)


def _r() -> np.ndarray:
    alpha = (21000, 20000, 22000, 20500)
    bands = [(x, D_ALPHA_Y, 5.5, 4.0, d) for x, d in zip(D_LANES, alpha, strict=True)]
    return _as(50000.0 - _bands(D_SHAPE, bands) + _noise(D_SHAPE, 21, 60), np.uint16)


def _jpeg_pixels() -> np.ndarray:
    beta = (110.0, 125.0, 260.0, 100.0, 120.0, 115.0)  # lane 3 runs into 0
    bands = [(x, JPEG_Y, 5.5, 4.0, d) for x, d in zip(C_LANES, beta, strict=True)]
    return _as(220.0 - _bands(C_SHAPE, bands) + _noise(C_SHAPE, 15, 2), np.uint8)


def _colour_pixels() -> np.ndarray:
    alpha = (90.0, 95.0, 85.0, 100.0, 92.0, 88.0)
    p53 = (60.0, 70.0, 65.0, 80.0, 75.0, 72.0)
    bands = [(x, ALPHA2_Y, 5.0, 3.5, d) for x, d in zip(C_LANES, alpha, strict=True)]
    bands += [
        (x, OVERLAP_BAND_Y if i == OVERLAP_LANE else P53_2_Y, 5.0, 3.5, d)
        for i, (x, d) in enumerate(zip(C_LANES, p53, strict=True))
    ]
    gray = 210.0 - _bands(C_SHAPE, bands)
    noise = [_noise(C_SHAPE, seed, 2) for seed in (16, 17, 18)]
    tint = (1.0, 0.96, 0.82)  # blue lowest: the channels differ
    channels = [_as(gray * t + n, np.uint8) for t, n in zip(tint, noise, strict=True)]
    return np.stack(channels, axis=-1)


def _tiff(pixels: np.ndarray) -> bytes:
    out = io.BytesIO()
    tifffile.imwrite(out, pixels, photometric="minisblack", metadata=None)
    return out.getvalue()


def _encoded(pixels: np.ndarray, image_format: str, **options: Any) -> bytes:
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format=image_format, **options)
    return out.getvalue()


# The original names of each case's images, as imported.
E1, E2, M, E3 = "E1 β-catenin 10 s.tif", "E2 p53 µ 60 s.tif", "M marker.png", "E3 α-tubulin.tif"
JPEG, COLOUR = "J lossy β.jpg", "C colour α.png"
S, N, R = "S signal β.tif", "N marker µ.png", "R reprobe α.tif"


def drawn(case: str) -> dict[str, np.ndarray]:
    """The pixels of the images ``case`` imports, by original name, in the
    order it imports them."""
    if case == "f1":
        return {E1: _e1(), E2: _e2(), M: _m(), E3: _e3()}
    if case == "f2":
        return {JPEG: _jpeg_pixels(), COLOUR: _colour_pixels()}
    if case == "f3":
        return {S: _s(), N: _n(), R: _r()}
    raise ValueError(f"unknown case {case!r}")


def _file(name: str, pixels: np.ndarray) -> bytes:
    if name.endswith(".tif"):
        return _tiff(pixels)
    if name.endswith(".jpg"):
        return _encoded(pixels, "JPEG", quality=90)
    return _encoded(pixels, "PNG")


def sources(case: str) -> dict[str, bytes]:
    """The image files ``case`` imports, by original name: :func:`drawn`
    written as uncompressed TIFF, PNG or JPEG (quality 90), as named."""
    return {name: _file(name, pixels) for name, pixels in drawn(case).items()}


def decoded(data: bytes) -> np.ndarray:
    """An image file's pixels as stored (no conversion)."""
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image)


# --- The projects ---


def _import(s, files, name, kind, polarity, membrane_id=None) -> str:
    return ops.import_image(
        s, io.BytesIO(files[name]), name, kind=kind, polarity=polarity, membrane_id=membrane_id
    )


def _f1(s: ops.ProjectSession, files: Mapping[str, bytes]) -> None:
    e1 = _import(s, files, E1, CHEMI, DARK)
    membrane_a = s.project.batch.membrane_of(e1).id
    e2 = _import(s, files, E2, CHEMI, DARK, membrane_a)
    m = _import(s, files, M, MARKER, DARK, membrane_a)
    e3 = _import(s, files, E3, CHEMI, LIGHT)
    ops.set_marker_image(s, e1, m)
    ops.set_ladder(s, membrane_a, "pageruler_plus/tris_glycine")
    for kda, y in LADDER:
        ops.add_calibration_point(s, m, y, kda, MARKER_BAND, x=LEFT_X, snap=False)
        ops.add_calibration_point(
            s, m, y + 3.0, kda, MARKER_BAND, x=RIGHT_X, side=LadderSide.RIGHT, snap=False
        )
    for kda, y in E2_POINTS:
        ops.add_calibration_point(s, e2, y, kda, FAINT_MARKER, x=LEFT_X, snap=False)
    for kda, y in STRIP_EDGES:
        ops.add_calibration_point(s, e3, y, kda, STRIP_EDGE, x=70.0)
    ops.set_lanes(
        s,
        [
            LaneInput("vehicle", "V1"),
            LaneInput("vehicle", "V1"),
            LaneInput("vehicle", "V2"),
            LaneInput("10 µM", "A1"),
            LaneInput("10 µM", "A2"),
            LaneInput("10 µM", "A3", included=False),
            LaneInput("50 µM", "B1"),
            LaneInput("50 µM", "B2"),
        ],
        reference_condition="vehicle",
    )
    beta = ops.add_protein(s, "β-catenin", Role.TARGET, e1, expected_mw=92.0)
    ops.detect_row_boxes(s, beta, (14, 14, 146, 36))
    alpha = ops.add_protein(s, "α-tubulin", Role.LOADING_CONTROL, e3)
    for lane, x in enumerate(B_LANES):
        ops.place_box(s, alpha, int(x), int(ALPHA_Y), lane_index=lane, grow=True)
    ops.set_box_padding(s, alpha, across=2, along=1)
    gapdh = ops.add_protein(
        s, "GAPDH", Role.LOADING_CONTROL, e1, box_size=BoxSize(width=12, height=10)
    )
    for lane, x in enumerate(A_LANES):
        if lane != 6:
            y = round(_tilted(x, GAPDH_Y))
            ops.place_box(s, gapdh, int(x), y, lane_index=lane, grow=False)
    p53 = ops.add_protein(
        s, "p53 µ", Role.TARGET, e2, expected_mw=53.0, loading_control_ids=[gapdh]
    )
    ops.detect_mw_row(s, p53, span=(14, 146))


def _f2(s: ops.ProjectSession, files: Mapping[str, bytes]) -> None:
    jpeg = _import(s, files, JPEG, CHEMI, DARK)
    colour = _import(s, files, COLOUR, CHEMI, DARK)
    ops.set_lanes(
        s,
        [LaneInput("vehicle", f"V{i}") for i in (1, 2, 3)]
        + [LaneInput("10 µM", f"A{i}") for i in (1, 2, 3)],
        reference_condition="vehicle",
    )
    size = BoxSize(width=12, height=10)
    beta = ops.add_protein(s, "β-catenin", Role.TARGET, jpeg, box_size=size)
    for lane, x in enumerate(C_LANES):
        ops.place_box(s, beta, int(x), int(JPEG_Y), lane_index=lane, grow=False)
    alpha = ops.add_protein(s, "α-tubulin", Role.LOADING_CONTROL, colour)
    for lane, x in enumerate(C_LANES):
        ops.place_box(s, alpha, int(x), int(ALPHA2_Y), lane_index=lane, grow=True)
    p53 = ops.add_protein(s, "p53 µ", Role.TARGET, colour, box_size=size)
    for lane, x in enumerate(C_LANES):
        y = OVERLAP_Y if lane == OVERLAP_LANE else int(P53_2_Y)
        ops.place_box(s, p53, int(x), y, lane_index=lane, grow=False)


def _f3(s: ops.ProjectSession, files: Mapping[str, bytes]) -> None:
    signal = _import(s, files, S, CHEMI, DARK)
    membrane = s.project.batch.membrane_of(signal).id
    marker = _import(s, files, N, MARKER, DARK, membrane)
    reprobe = _import(s, files, R, CHEMI, DARK)
    ops.set_marker_image(s, signal, marker)
    ops.set_ladder(s, membrane, "pageruler_plus/tris_glycine")
    for kda, y in D_LADDER:
        ops.add_calibration_point(s, marker, y, kda, MARKER_BAND, x=12.0, snap=False)
    ops.set_lanes(
        s,
        [LaneInput("vehicle", "V1"), LaneInput("vehicle", "V2")]
        + [LaneInput("10 µM", "A1"), LaneInput("10 µM", "A2")],
        reference_condition="vehicle",
    )
    beta = ops.add_protein(s, "β-catenin", Role.TARGET, signal, expected_mw=92.0)
    ops.detect_row_boxes(s, beta, (22, 9, 90, 29))
    alpha = ops.add_protein(
        s, "α-tubulin", Role.LOADING_CONTROL, reprobe, box_size=BoxSize(width=12, height=10)
    )
    for lane, x in enumerate(D_LANES):
        ops.place_box(s, alpha, int(x), int(D_ALPHA_Y), lane_index=lane, grow=False)
    # Removing the marker unlinks the signal image and takes the ladder's
    # points, and with them β-catenin's apparent MWs; undo brings them back.
    ops.remove_image(s, marker)
    ops.undo(s)
    ops.redo(s)
    ops.undo(s)


_STEPS: Mapping[str, Callable[[ops.ProjectSession, Mapping[str, bytes]], None]] = {
    "f1": _f1,
    "f2": _f2,
    "f3": _f3,
}


def build(folder: Path, case: str, *, files: Mapping[str, bytes] | None = None) -> Project:
    """Build ``case`` in ``folder`` (new or empty) through the operations and
    return the project as saved. ``files`` are the image files to import, by
    original name (:func:`frozen_sources`); by default those drawn here
    (:func:`sources`)."""
    steps = _STEPS[case]
    session = ops.new_project(folder, clock=Clock())  # saved after every step
    try:
        steps(session, sources(case) if files is None else files)
        return session.project
    finally:
        session.close()  # deletes the files only the undo history kept


def frozen_sources(folder: Path) -> dict[str, bytes]:
    """The image files of the project in ``folder``, by original name."""
    project = storage.load_project(folder)
    return {
        image.original_name: storage.image_path(folder, image).read_bytes()
        for image in project.batch.iter_images()
    }


def copy_project(source: Path, target: Path) -> None:
    """Copy a project's ``project.json`` and ``images/`` from ``source`` to
    ``target`` (created)."""
    (target / storage.IMAGES_DIR).mkdir(parents=True)
    shutil.copyfile(source / storage.PROJECT_FILE, target / storage.PROJECT_FILE)
    for path in sorted((source / storage.IMAGES_DIR).iterdir()):
        shutil.copyfile(path, target / storage.IMAGES_DIR / path.name)


# --- What a project stores ---


def stored_values(project: Project) -> dict[str, Any]:
    """Every value ``project`` stores but its schema version, in stored
    order, under the names of the schema-2 file: the background method, the
    next id, the batch (its lanes and reference, each membrane with its images
    and calibration, each protein with its boxes and not-detected records)
    and the log. A default the file leaves out (no padding, no record, no
    band count, a left-hand point) is written out here."""
    batch = project.batch
    return {
        "background_method": project.background_method,
        "next_id": project.next_id,
        "batch": {
            "lanes": [_lane(lane) for lane in batch.lanes],
            "reference_condition": batch.reference_condition,
            "membranes": [_membrane(membrane) for membrane in batch.membranes],
            "proteins": [_protein(protein) for protein in batch.proteins],
        },
        "log": [_entry(entry) for entry in project.log],
    }


def _lane(lane: Lane) -> dict[str, Any]:
    return {
        "index": lane.index,
        "label": lane.label,
        "sample": lane.sample,
        "included": lane.included,
        "metadata": dict(lane.metadata),
    }


def _image(image: ImageRef) -> dict[str, Any]:
    return {
        "id": image.id,
        "file": image.file,
        "original_name": image.original_name,
        "kind": image.kind.value,
        "marker_image_id": image.marker_image_id,
        "sha256": image.sha256,
        "width": image.width,
        "height": image.height,
        "bit_depth": image.bit_depth,
        "polarity": image.polarity.value,
        "background": image.background,
        "import_warnings": [
            {"code": warning.code, "message": warning.message} for warning in image.import_warnings
        ],
    }


def _membrane(membrane: Membrane) -> dict[str, Any]:
    calibration = membrane.calibration
    return {
        "id": membrane.id,
        "images": [_image(image) for image in membrane.images],
        "calibration": {
            "ladder": calibration.ladder,
            "ladder_kda": list(calibration.ladder_kda),
            "points": [
                {
                    "image_id": point.image_id,
                    "y": point.y,
                    "mw": point.mw,
                    "source": point.source.value,
                    "x": point.x,
                    "side": point.side.value,
                }
                for point in calibration.points
            ],
            "fit_method": calibration.fit_method.value,
            "fit_quality": calibration.fit_quality,
        },
    }


def _protein(protein: Protein) -> dict[str, Any]:
    return {
        "id": protein.id,
        "name": protein.name,
        "role": protein.role.value,
        "image_id": protein.image_id,
        "loading_control_ids": list(protein.loading_control_ids),
        "expected_mw": protein.expected_mw,
        "expected_band_count": protein.expected_band_count,
        "mw_tolerance": protein.mw_tolerance,
        "box_size": {"width": protein.box_size.width, "height": protein.box_size.height},
        "box_padding": {
            "across": protein.box_padding.across,
            "along": protein.box_padding.along,
        },
        "bands": [_band(band) for band in protein.bands],
        "undetected": [_undetected(record) for record in protein.undetected],
    }


def _band(band: Band) -> dict[str, Any]:
    return {
        "id": band.id,
        "lane_index": band.lane_index,
        "band_index": band.band_index,
        "box": {"x": band.box.x, "y": band.box.y},
        "net": band.net,
        "background_level": band.background_level,
        "background_mode": band.background_mode,
        "background_spread": band.background_spread,
        "apparent_mw": band.apparent_mw,
        "bands_found": band.bands_found,
        "clipped": band.clipped,
        "possibly_clipped": band.possibly_clipped,
        "source": band.source.value,
        "manually_edited": band.manually_edited,
    }


def _undetected(record: UndetectedBand) -> dict[str, Any]:
    region = record.region
    return {
        "lane_index": record.lane_index,
        "band_index": record.band_index,
        "reason": record.reason.value,
        "snr": record.snr,
        "threshold": record.threshold,
        "region": {"x0": region.x0, "y0": region.y0, "x1": region.x1, "y1": region.y1},
        "source": record.source.value,
    }


def _entry(entry: LogEntry) -> dict[str, Any]:
    # The time comes from the build's clock and the version is the frozen
    # build's: both are stored values like any other.
    return {
        "seq": entry.seq,
        "time": entry.time,
        "action": entry.action,
        "version": entry.version,
        "params": dict(entry.params),
        "content_hash": entry.content_hash,
    }


def stored_differences(golden: Mapping[str, Any], project: Project) -> list[str]:
    """Where the values ``project`` stores differ from those of a
    ``stored.json`` (``golden``), exactly (:func:`differences`); what
    ``golden`` records about the fixture itself (:data:`FROZEN_ONLY`) is left
    out. No hash of the content in the build's own shape takes part, so a
    project held in another shape with every value intact matches."""
    expected = {key: value for key, value in golden.items() if key not in FROZEN_ONLY}
    return differences(expected, json.loads(canonical_json(stored_values(project))))


def changes(old: Any, new: Any, path: str = "$") -> list[list[Any]]:
    """Where the JSON value ``new`` differs from ``old``, one ``[path, old
    part, new part]`` each: a value that differs (a float, beyond
    :data:`REL_TOL` relative), or whole, an object whose keys or a list whose
    length differ."""
    if isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
        return [found for key in old for found in changes(old[key], new[key], f"{path}.{key}")]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        pairs = enumerate(zip(old, new, strict=True))
        return [found for i, (a, b) in pairs for found in changes(a, b, f"{path}[{i}]")]
    return [[path, old, new]] if differences(old, new, rel_tol=REL_TOL) else []


def fresh_changes(frozen: Mapping[str, Any], project: Project) -> list[list[Any]]:
    """How the values a fresh build (``project``) stores differ from the
    frozen ones (``frozen``, a ``stored.json``), by :func:`changes`; what
    ``frozen`` records about the fixture (:data:`FROZEN_ONLY`) and, in each
    log entry, :data:`NOT_IN_FRESH_LOG` are left out. Empty while the
    operations store the numbers they stored at 05b4af5."""

    def compared(values: Mapping[str, Any]) -> dict[str, Any]:
        kept = {key: value for key, value in values.items() if key not in FROZEN_ONLY}
        kept["log"] = [
            {key: value for key, value in entry.items() if key not in NOT_IN_FRESH_LOG}
            for entry in kept["log"]
        ]
        return kept

    fresh = json.loads(canonical_json(stored_values(project)))
    return changes(compared(frozen), compared(fresh))


# --- What the build derives from it ---


def table_rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text, newline="")))


def table_text(rows: Sequence[Sequence[str]]) -> str:
    out = io.StringIO(newline="")
    csv.writer(out).writerows(rows)
    return out.getvalue()


def near_midpoint(value: float, decimals: int, rel: float = REL_TOL) -> bool:
    """Whether ``value`` lies within ``rel`` (relative) of a midpoint between
    two numbers of ``decimals`` decimals: where rounding may go either way."""
    scaled = abs(value) * 10**decimals
    return abs(scaled - math.floor(scaled) - 0.5) <= rel * scaled


def _table(results_set: results.Results, columns: export.LaneColumns) -> str:
    return export.result_set_table(results_set, columns).decode(export.CSV_ENCODING)


def midpoint_cells(text: str, unrounded: Mapping[int, str]) -> list[list[int]]:
    """The cells of the lane table ``text`` whose value lies within
    :data:`REL_TOL` of a rounding midpoint, as ``[row, column, decimals]``.
    ``unrounded`` maps a number of decimals to the same table with the
    columns rounded to it written unrounded."""
    rows = table_rows(text)
    near = []
    for decimals, twin_text in unrounded.items():
        for r, (row, twin) in enumerate(zip(rows, table_rows(twin_text), strict=True)):
            for c, (cell, value) in enumerate(zip(row, twin, strict=True)):
                if cell != value and near_midpoint(float(value), decimals):
                    near.append([r, c, decimals])
    return sorted(near)


def _result_sets(computed: results.Results) -> list[results.Results]:
    """The result sets an export writes: the included lanes, then all lanes
    when excluded lanes hold values."""
    return [computed] if computed.all_lanes is None else [computed, computed.all_lanes]


def _lane_tables(computed: results.Results) -> list[dict[str, Any]]:
    """Each result set's lane table as an export writes it, with the cells
    whose value lies within :data:`REL_TOL` of a rounding midpoint."""
    columns = export.result_columns(computed)
    tables = []
    for one in _result_sets(computed):
        unrounded = {}
        for name in ("LANE_TABLE_DECIMALS", "LANE_TABLE_RATIO_DECIMALS"):
            decimals = getattr(export, name)
            with mock.patch.object(export, name, 17):  # unrounded, in those columns only
                unrounded[decimals] = _table(one, columns)
        text = _table(one, columns)
        near = midpoint_cells(text, unrounded)
        tables.append({"label": one.label, "csv": text, "near_midpoint": near})
    return tables


def _plotted(computed: results.Results) -> list[dict[str, Any]]:
    """What plotting some conditions only changes, in each result set: the
    conditions plotted, the notices and each series' chart."""
    shown = []
    for one in _result_sets(computed):
        dump = one.model_dump(mode="json")
        shown.append(
            {
                "label": dump["label"],
                "plot_conditions": dump["plot_conditions"],
                "notices": dump["notices"],
                "charts": [series["chart"] for series in dump["series"]],
            }
        )
    return shown


def _calibration(batch) -> list[dict[str, Any]]:
    """Each membrane's register groups with their fit, as the page shows them."""
    shown = []
    for membrane in batch.membranes:
        groups = []
        for group in membrane.register_groups():
            image_ids = [image.id for image in membrane.images if image.id in group]
            fit = ops.calibration_fit(membrane, image_ids[0])
            groups.append({"image_ids": image_ids, "fit": None if fit is None else fit.as_json()})
        shown.append({"membrane_id": membrane.id, "groups": groups})
    return shown


def case_of(folder: Path) -> str:
    """The case of the fixture in ``folder``, as its ``stored.json`` names it
    (whatever the folder's name); ``ValueError`` if it names none."""
    path = Path(folder) / "stored.json"
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the case of {folder}: {exc}") from exc
    case = frozen.get("case") if isinstance(frozen, dict) else None
    if not isinstance(case, str) or case not in _STEPS:
        raise ValueError(f"{path} names no case of {', '.join(CASES)}: {case!r}")
    return case


def derived_output(folder: Path, work: Path, case: str | None = None) -> dict[str, Any]:
    """What the running build derives from the frozen project in ``folder``,
    as plain JSON values; ``fresh_changes`` are those of a fresh build of it
    from its image files (:func:`fresh_changes`). ``case`` says how to build
    it; by default it is the one the fixture names (:func:`case_of`).
    ``work`` (new or empty) takes a copy of the project, opened there (and
    migrated in memory, if the build reads a newer schema), and the fresh
    build."""
    folder, work = Path(folder), Path(work)
    case = case_of(folder) if case is None else case
    frozen = json.loads((folder / "stored.json").read_text(encoding="utf-8"))
    copied = work / "copy"
    copy_project(folder, copied)
    session = ops.open_project(copied, autosave=None)
    try:
        project = session.project
        batch = project.batch
        computed = results.compute_results(batch)
        plotted = results.compute_results(batch, plot_conditions=[PLOTTED])
        output = {
            "results": computed.model_dump(mode="json"),
            "plotted": _plotted(plotted),
            "lane_tables": _lane_tables(computed),
            "calibration": _calibration(batch),
            "history_issues": record.history_issues(project),
        }
    finally:
        session.close()
    fresh = build(work / "fresh", case, files=frozen_sources(folder))
    output["fresh_changes"] = fresh_changes(frozen, fresh)
    return json.loads(canonical_json(output))


def _scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _lines(value: Any, depth: int) -> str:
    inner, outer = " " * (depth + 1), " " * depth
    if isinstance(value, dict) and not all(isinstance(key, str) for key in value):
        raise TypeError(f"JSON keys are text: {list(value)!r}")
    if isinstance(value, dict) and value:
        items = [f"{inner}{_scalar(key)}: {_lines(value[key], depth + 1)}" for key in sorted(value)]
        return "{\n" + ",\n".join(items) + "\n" + outer + "}"
    if isinstance(value, (list, tuple)) and any(isinstance(v, (dict, list, tuple)) for v in value):
        items = [inner + _lines(v, depth + 1) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + outer + "]"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{}"
    return _scalar(value)  # TypeError for anything JSON has no form for


def canonical_json(value: Any) -> bytes:
    """``value`` as JSON written one way only: keys sorted, one key per line
    indented by one space a level, a list of plain values on one line, any
    other list one item a line; floats as ``repr`` writes them, text as UTF-8;
    LF and a final newline. NaN and infinity are refused."""
    return (_lines(value, 0) + "\n").encode("utf-8")


# --- The comparison ---


def differences(
    golden: Any, actual: Any, *, rel_tol: float | None = None, path: str = "$"
) -> list[str]:
    """Where ``actual`` differs from ``golden`` (JSON values), one line each:
    types, text, integers, booleans, None, dict keys and list order exactly;
    floats exactly too, or within ``rel_tol`` relative (no absolute
    tolerance) when it is given."""
    if type(golden) is not type(actual):
        return [f"{path}: {golden!r} != {actual!r}"]
    if isinstance(golden, dict):
        found = []
        for key, value in golden.items():
            if key in actual:
                found += differences(value, actual[key], rel_tol=rel_tol, path=f"{path}.{key}")
            else:
                found.append(f"{path}.{key}: missing")
        found += [f"{path}.{key}: not in the golden" for key in actual if key not in golden]
        return found
    if isinstance(golden, list):
        if len(golden) != len(actual):
            return [f"{path}: {len(golden)} items != {len(actual)}"]
        found = []
        for i, (g, a) in enumerate(zip(golden, actual, strict=True)):
            found += differences(g, a, rel_tol=rel_tol, path=f"{path}[{i}]")
        return found
    if isinstance(golden, float):
        if rel_tol is None:
            same = repr(golden) == repr(actual)
        else:
            same = math.isclose(golden, actual, rel_tol=rel_tol, abs_tol=0.0)
        return [] if same else [f"{path}: {golden!r} != {actual!r}"]
    return [] if golden == actual else [f"{path}: {golden!r} != {actual!r}"]


def _one_unit_apart(a: str, b: str, decimals: int) -> bool:
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return abs(round(x * 10**decimals) - round(y * 10**decimals)) == 1


def table_differences(golden: dict[str, Any], actual: dict[str, Any], path: str) -> list[str]:
    """Where a lane table differs from its golden: the label and every cell
    exactly, but a number may be one unit off in its column's last decimal
    where the golden marks it as near a rounding midpoint."""
    found = differences(golden["label"], actual["label"], path=f"{path}.label")
    rows, other = table_rows(golden["csv"]), table_rows(actual["csv"])
    if len(rows) != len(other):
        return [*found, f"{path}.csv: {len(rows)} rows != {len(other)}"]
    allowed = {(r, c): d for r, c, d in golden["near_midpoint"]}
    for r, (row, other_row) in enumerate(zip(rows, other, strict=True)):
        if len(row) != len(other_row):
            found.append(f"{path}.csv[{r}]: {len(row)} cells != {len(other_row)}")
            continue
        for c, (cell, other_cell) in enumerate(zip(row, other_row, strict=True)):
            decimals = allowed.get((r, c))
            if cell == other_cell or (
                decimals is not None and _one_unit_apart(cell, other_cell, decimals)
            ):
                continue
            found.append(f"{path}.csv[{r}][{c}]: {cell!r} != {other_cell!r}")
    return found


def derived_differences(golden: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Where a derived output differs from its golden: lane tables by
    :func:`table_differences`, everything else by :func:`differences` with
    floats within :data:`REL_TOL`."""
    golden, actual = dict(golden), dict(actual)
    tables, other_tables = golden.pop("lane_tables", None), actual.pop("lane_tables", None)
    found = differences(golden, actual, rel_tol=REL_TOL)
    if not isinstance(tables, list) or not isinstance(other_tables, list):
        found += differences(tables, other_tables, path="$.lane_tables")
    elif len(tables) != len(other_tables):
        found.append(f"$.lane_tables: {len(tables)} items != {len(other_tables)}")
    else:
        for i, (table, other) in enumerate(zip(tables, other_tables, strict=True)):
            found += table_differences(table, other, f"$.lane_tables[{i}]")
    return found


def check_derived(golden: dict[str, Any], actual: dict[str, Any], case: str) -> None:
    """Raise ``AssertionError``, naming each difference and the refresh
    command, unless ``actual`` matches the golden derived output of ``case``."""
    found = derived_differences(golden, actual)
    if found:
        shown = found[:40] + ([f"... and {len(found) - 40} more"] if len(found) > 40 else [])
        raise AssertionError(
            f"the derived output of {case} differs from {case}/derived.json:\n  "
            + "\n  ".join(shown)
            + "\nIf the change is intended, refresh the derived goldens with\n    "
            + REFRESH_COMMAND
            + "\nand say why in the pull request description (stored.json never changes)."
        )


# --- The commands ---


def _normalized(name: str) -> str:
    """A distribution's name as a lock file writes it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_versions() -> dict[str, str]:
    """The version of each distribution the running interpreter finds, by
    normalized name (the first one found on ``sys.path``, as imports do)."""
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        versions.setdefault(_normalized(distribution.metadata["Name"]), distribution.version)
    return versions


def lock_mismatches(
    lock_text: str, installed: Mapping[str, str], project: str = "proteia"
) -> list[str]:
    """The runtime dependencies of ``project`` locked in ``lock_text`` (a
    ``uv.lock``) that ``installed`` (name -> version) does not hold at a
    locked version, one line each. They are the project's own dependencies
    (not its dependency groups) and theirs, with the extras they ask for. One
    that only a marker asks for (say, on one platform) may be missing, but
    when installed it must be at its locked version."""
    have = {_normalized(name): version for name, version in installed.items()}
    locked: dict[str, list[dict[str, Any]]] = {}
    for package in tomllib.loads(lock_text).get("package", []):
        locked.setdefault(_normalized(package["name"]), []).append(package)
    root = _normalized(project)
    if root not in locked:
        return [f"{project}: not in uv.lock"]
    required: dict[str, bool] = {}  # name -> must be installed
    seen: set[tuple[str, bool, tuple[str, ...]]] = set()
    todo: list[tuple[str, bool, tuple[str, ...]]] = [(root, True, ())]
    while todo:
        name, must, extras = todo.pop()
        if (name, must, extras) in seen:
            continue
        seen.add((name, must, extras))
        required[name] = required.get(name, False) or must
        for package in locked.get(name, []):
            asked = list(package.get("dependencies", []))
            for extra in extras:
                asked += package.get("optional-dependencies", {}).get(extra, [])
            for dependency in asked:
                todo.append(
                    (
                        _normalized(dependency["name"]),
                        must and "marker" not in dependency,
                        tuple(dependency.get("extra", ())),
                    )
                )
    found = []
    for name in sorted(required.keys() - {root}):
        versions = [package["version"] for package in locked.get(name, [])]
        version = have.get(name)
        if version in versions or (version is None and not required[name]):
            continue
        found.append(
            f"{name}: locked {' or '.join(versions) or 'nowhere'}, installed {version or 'nowhere'}"
        )
    return found


def frozen_refusal(
    *,
    head: str,
    changes: str,
    module: Path,
    toplevel: Path,
    mismatched: Sequence[str] = (),
) -> str | None:
    """Why the frozen files may not be written here, or None: only in a
    checkout at :data:`FROZEN_COMMIT` whose build has no local change, is
    the one imported, and runs on the dependencies its ``uv.lock`` locks
    (``mismatched``: :func:`lock_mismatches`)."""
    if head != FROZEN_COMMIT:
        return (
            f"this checkout is at {head}, not at {FROZEN_COMMIT}: the frozen files are written"
            f" there, once (to refresh derived.json, run {REFRESH_COMMAND})"
        )
    if changes.strip():
        return f"the build has local changes, so it is not {FROZEN_COMMIT}:\n{changes}"
    if not Path(module).resolve().is_relative_to((Path(toplevel) / "src").resolve()):
        return f"proteia is imported from {module}, not from {toplevel / 'src'}"
    if mismatched:
        return (
            "the installed dependencies are not those uv.lock locks (run uv sync --locked):\n  "
            + "\n  ".join(mismatched)
        )
    return None


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=HERE, capture_output=True, encoding="utf-8", check=True
    ).stdout


def _refusal_here() -> str | None:
    try:
        toplevel = Path(_git("rev-parse", "--show-toplevel").strip())
        head = _git("rev-parse", "HEAD").strip()
        changes = _git(
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "pyproject.toml",
            "uv.lock",
        )
        lock_text = (toplevel / "uv.lock").read_text(encoding="utf-8")
        mismatched = lock_mismatches(lock_text, installed_versions())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        return (
            f"cannot tell whether this checkout is at {FROZEN_COMMIT}: {type(exc).__name__}: {exc}"
        )
    return frozen_refusal(
        head=head,
        changes=changes,
        module=Path(proteia.__file__),
        toplevel=toplevel,
        mismatched=mismatched,
    )


class Refused(Exception):
    """The frozen files may not be written here."""


def _write(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value))


def write_frozen(out: Path) -> None:
    """Build every case in ``out`` (whose case folders must not exist yet)
    with its frozen files and a first ``derived.json``; only in a checkout at
    :data:`FROZEN_COMMIT` (:func:`frozen_refusal`), else :class:`Refused`."""
    refusal = _refusal_here()
    if refusal is not None:
        raise Refused(refusal)
    taken = [case for case in CASES if (out / case).exists()]
    if taken:
        raise Refused(f"{', '.join(taken)} exist in {out}: the frozen files are written once")
    head = _git("rev-parse", "HEAD").strip()
    with tempfile.TemporaryDirectory(prefix="schema2-", ignore_cleanup_errors=True) as tmp:
        for case in CASES:
            folder = out / case
            built = build(folder, case)
            frozen = {
                "case": case,
                "built_at": head,
                "version": proteia.__version__,
                "project_sha256": hashlib.sha256(
                    (folder / storage.PROJECT_FILE).read_bytes()
                ).hexdigest(),
                "content_hash": storage.content_hash(built),
                **stored_values(built),
            }
            _write(folder / "stored.json", frozen)
            derived = derived_output(folder, Path(tmp) / case, case)
            if derived["fresh_changes"]:  # never: one build, one machine
                raise Refused(f"{case}: a second build of it stores other values")
            _write(folder / "derived.json", derived)
            print(f"wrote {case}")


def refresh_derived(root: Path, work: Path) -> None:
    """Rewrite each case's ``derived.json`` under ``root`` from its frozen
    project, with the running build (``work``, new or empty, takes the
    copies); nothing else is written under ``root``."""
    for case in CASES:
        derived = derived_output(Path(root) / case, Path(work) / case, case)
        _write(Path(root) / case / "derived.json", derived)
        print(f"refreshed {case}/derived.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--write-frozen", action="store_true", help="only at 05b4af5, once")
    command.add_argument("--refresh-derived", action="store_true", help="rewrite derived.json")
    parser.add_argument("--out", type=Path, default=HERE, help="the case folders' folder")
    args = parser.parse_args(argv)
    if args.refresh_derived:
        with tempfile.TemporaryDirectory(prefix="schema2-", ignore_cleanup_errors=True) as tmp:
            refresh_derived(args.out, Path(tmp))
        return 0
    try:
        write_frozen(args.out)
    except Refused as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")  # a console of any code page
    sys.exit(main())
