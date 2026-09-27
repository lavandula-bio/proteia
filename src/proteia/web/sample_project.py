# SPDX-License-Identifier: Apache-2.0
"""The sample project (#55): a new project on the synthetic sample blot
(:mod:`proteia.samples`), set up up to the row boxes, which the user drags.

The sample images are not in the repository (the maintainer's choice, C1): they
are made on demand. :func:`create_sample_project` creates the project
:data:`SAMPLE_NAME` in the projects root, or ``Sample blot (2)``, ``(3)`` and so
on when that name is taken (:func:`~proteia.web.projects.free_name`), and sets
it up through the ordinary operations, so each step is logged and undo takes it
back:

1. ``sample-blot.tif`` imported as a chemiluminescence image, dark on light;
2. ``sample-marker.tif`` imported onto the same membrane as a visible-light
   marker image: the prestained ladder of that membrane, for the
   molecular-weight calibration (#58); nothing reads it yet, and no protein
   can be placed on it;
3. the eight lanes declared, vehicle V1-V4 and treatment T1-T4, with vehicle
   as the reference;
4. α-tubulin added as the loading control and β-catenin as the target using
   it, both on the blot and with no boxes.

The images are the bytes :func:`proteia.samples.generate` writes
(:func:`~proteia.samples.sample_files`), imported from memory, so no temporary
folder is written or left behind. The truth table is written next to
``project.json`` as :data:`~proteia.samples.TRUTH_FILE`: the true signal of
every band, to check the lane tables an export writes against. It lies outside
``exports/``, under its own name, so it is never taken for an export, and it is
not part of the project: no operation logs it and undo leaves it. If any step
fails, the new project's folder is removed with everything written into it,
and the error propagates. If some of it cannot be removed (a file in it held
open by another program, which Windows refuses to delete), the folder stays,
without a ``project.json``: the Projects dialog does not list it, but its name
is taken. :class:`SampleFolderLeftError` then says so, naming the folder, in
place of the error.

A hook for the first-use tour to come: ``POST /api/projects/sample`` answers
:func:`sample_payload` as ``sample``: each protein's row, top to bottom, as a
drag over it that boxes its eight bands.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from proteia import samples
from proteia.core import operations as ops
from proteia.core.model import ImageKind, Polarity, Project, Rect, Role
from proteia.core.session import Clock, ProjectSession, utc_now
from proteia.core.storage import write_atomic
from proteia.web import projects

SAMPLE_NAME: Final = "Sample blot"
# How far a row's drag reaches above the row in the end lanes, and below it in
# the middle lanes (which ran SMILE_PX further), px.
ROW_MARGIN_PX: Final = 25


class SampleFolderLeftError(OSError):
    """The sample project's setup failed, and its folder could not all be
    removed after: it is left in the projects root, unfinished."""


@dataclass(frozen=True)
class SampleRow:
    """A protein's row on the sample blot."""

    protein: str  # the protein's name, as the sample project adds it
    rect: Rect  # a drag over the row that boxes its bands: x0, y0, x1, y1, end-exclusive


def sample_rows() -> tuple[SampleRow, ...]:
    """Each protein's row, top to bottom, as a drag over it: from half a lane
    pitch left of lane 1's centre to half a pitch right of lane 8's (the ladder,
    on the marker image only, lies further left), and from
    :data:`ROW_MARGIN_PX` above the end lanes' bands to as far below the middle
    lanes' bands."""
    half = samples.LANE_PITCH / 2
    x0, x1 = round(samples.LANE_X[0] - half), round(samples.LANE_X[-1] + half)
    rows = []
    for row in sorted(samples.ROWS, key=lambda r: -r.kda):  # heavier runs higher
        y = samples.band_y(row.kda, samples.LANE_X[0])
        rect = (x0, round(y - ROW_MARGIN_PX), x1, round(y + samples.SMILE_PX + ROW_MARGIN_PX))
        rows.append(SampleRow(row.protein, rect))
    return tuple(rows)


def sample_payload(project: Project) -> dict[str, JsonValue]:
    """What the answer of ``POST /api/projects/sample`` carries as ``sample``:
    the truth table's name in the project folder, and each protein's row, top
    to bottom, as ``{protein_id, rect}`` (:func:`sample_rows`), for the sample
    proteins ``project`` holds."""
    ids = {protein.name: protein.id for protein in project.batch.proteins}
    return {
        "truth_file": samples.TRUTH_FILE,
        "rows": [
            {"protein_id": ids[row.protein], "rect": list(row.rect)}
            for row in sample_rows()
            if row.protein in ids
        ],
    }


def create_sample_project(root: Path, *, clock: Clock = utc_now) -> ProjectSession:
    """Create the sample project in ``root`` and open it (see the module
    docstring); saved before it is answered, so an ``OSError`` or
    :class:`~proteia.core.storage.ProjectError` while saving propagates too,
    after its folder is removed; or, if the folder cannot all be removed,
    :class:`SampleFolderLeftError` in place of any error."""
    files = samples.sample_files()
    session = projects.create_project(root, projects.free_name(root, SAMPLE_NAME), clock=clock)
    try:
        _set_up(session, files)
        if session.dirty:  # an autosave failed: fail now, not when it is next edited
            ops.save(session)
    except BaseException as exc:
        shutil.rmtree(session.folder, ignore_errors=True)
        if isinstance(exc, Exception) and session.folder.exists():
            raise SampleFolderLeftError(
                f"the sample project could not be set up ({exc}), and its unfinished folder"
                f" {session.folder.name!r} could not be removed: delete it from the projects"
                " folder"
            ) from exc
        raise
    return session


def _set_up(session: ProjectSession, files: dict[str, bytes]) -> None:
    dark = Polarity.DARK_ON_LIGHT
    blot = ops.import_image(
        session,
        io.BytesIO(files[samples.BLOT_FILE]),
        samples.BLOT_FILE,
        kind=ImageKind.CHEMILUMINESCENCE,
        polarity=dark,
    )
    ops.import_image(
        session,
        io.BytesIO(files[samples.MARKER_FILE]),
        samples.MARKER_FILE,
        kind=ImageKind.VISIBLE_MARKER,
        polarity=dark,
        membrane_id=session.project.batch.membrane_of(blot).id,
    )
    ops.set_lanes(
        session,
        [
            ops.LaneInput(condition, sample)
            for condition, sample in zip(samples.CONDITIONS, samples.SAMPLES, strict=True)
        ],
        reference_condition=samples.REFERENCE,
    )
    loading = ops.add_protein(session, samples.LOADING_CONTROL, Role.LOADING_CONTROL, blot)
    ops.add_protein(session, samples.TARGET, Role.TARGET, blot, loading_control_ids=[loading])
    write_atomic(session.folder / samples.TRUTH_FILE, files[samples.TRUTH_FILE])
