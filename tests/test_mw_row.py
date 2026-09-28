# SPDX-License-Identifier: Apache-2.0
"""Rows placed by their expected molecular weight, along the protein line (#58,
D11, D12).

The blots are drawn here, on the sample blot's migration law (log-linear, 300 px
per decade of MW, the middle lanes 6 px lower than the end ones: a smile), 1330
px wide, so that a second ladder fits one lane pitch right of lane 8 as the
first stands one pitch left of lane 1 (x = 77 and 1229; the lanes at x = 205 +
128 i). A 16-bit chemiluminescence blot and its visible-light marker are drawn
together and turned about (653, 250) where a test tilts them; the ladders are
marked where their bands lie. Positions are continuous (pixel row r covers
[r, r+1)). The public sample stays one-ladder and 1200 px wide. Names use µ, α
and β.
"""

import dataclasses
import functools
import io
import itertools
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_blot
from proteia import samples
from proteia.core import mwcal, mwrow, rowdetect
from proteia.core import operations as ops
from proteia.core.model import (
    CalibrationPoint,
    CalibrationPointSource,
    ImageKind,
    LadderSide,
    Polarity,
    ProposalSource,
    Role,
    UnknownIdError,
)
from proteia.core.operations import ErrorCode, LaneInput, OperationError, ProjectSession
from proteia.core.session import save_to_folder
from proteia.core.storage import load_project
from proteia.web.state import project_state
from test_checks import VENDOR_YS
from test_checks import blot as model_blot
from test_checks import ladder as model_ladder
from test_checks import law as model_law
from test_operations import (
    assert_mw_current,
    assert_nets_current,
    import_blot,
    lane_bands,
    noisy16,
    open_sample,
    plant,
    protein_of,
    session_on,
)

WIDTH, HEIGHT = 1330, 500
LANES = tuple(205.0 + 128.0 * i for i in range(8))
X_LEFT, X_RIGHT = 77.0, 1229.0
TURN_CENTRE = (653.0, 250.0)
SMILE = 6.0
KDA = samples.LADDER_KDA  # PageRuler Plus, Tris-glycine: 250 ... 10 kDa
PRESET = "pageruler_plus/tris_glycine"
MARKER_BAND = CalibrationPointSource.VISIBLE_MARKER
LEFT, RIGHT = LadderSide.LEFT, LadderSide.RIGHT
MEMBRANE = 52000.0
NOISE = 300.0
BAND_WIDTH = 92.0  # across a lane, at 20% of the peak
# (kDa, depth, height at 20% of the peak) of each row of bands, one per lane;
# after them a row may give its bands' dip and asym (_darken).
LOADING = (50.0, 14000.0, 18.0)  # α-tubulin
TARGET = (92.0, 9000.0, 16.0)  # β-catenin
ROWS = (LOADING, TARGET)
_UX4 = (2.0 * math.log(5.0)) ** 0.25  # a super-Gaussian's half-extent at 20%
_UX2 = math.sqrt(2.0 * math.log(5.0))  # a Gaussian's


def smile(x: float) -> float:
    return SMILE * (1.0 - ((x - 653.0) / 448.0) ** 2)


def law(kda: float, x: float) -> float:
    """Where a band of ``kda`` kDa lies at column ``x`` on the level blot."""
    return 45.5 + 300.0 * math.log10(250.0 / kda) + smile(x)


def turn(x, y, degrees: float):
    """(x, y) turned by ``degrees`` about the blot's centre; positive turns the
    right side down. Arrays too."""
    theta = math.radians(degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    dx, dy = x - TURN_CENTRE[0], y - TURN_CENTRE[1]
    return TURN_CENTRE[0] + dx * cos - dy * sin, TURN_CENTRE[1] + dx * sin + dy * cos


def truth(kda: float, x: float, degrees: float = 0.0) -> tuple[float, float]:
    """Where the band of ``kda`` kDa of the lane (or ladder) at ``x`` lies on
    the blot turned by ``degrees``."""
    return turn(x, law(kda, x), degrees)


def _darken(
    darkening: np.ndarray,
    cx: float,
    kda: float,
    depth: float,
    height: float,
    degrees,
    dip: float = 0.0,
    asym: float = 0.0,
):
    """Add one band: flat-topped across the lane, Gaussian down it, following
    the smile, drawn turned; ``dip`` of it lighter in its middle (a dumbbell,
    its ends darker), and ``asym`` darker at its right end, lighter at its
    left."""
    cy = law(kda, cx)
    tx, ty = turn(cx, cy, degrees)
    x0, x1 = max(0, int(tx - BAND_WIDTH)), min(WIDTH, int(tx + BAND_WIDTH) + 1)
    y0, y1 = max(0, int(ty - 4 * height)), min(HEIGHT, int(ty + 4 * height) + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    u, v = turn(xs + 0.5, ys + 0.5, -degrees)
    across = np.exp(-0.5 * np.abs((u - cx) / (BAND_WIDTH / (2.0 * _UX4))) ** 4)
    if dip or asym:
        across *= 1.0 - dip * np.exp(-0.5 * ((u - cx) / (BAND_WIDTH / 6.0)) ** 2)
        across *= 1.0 + asym * np.clip((u - cx) / (BAND_WIDTH / 2.0), -1.0, 1.0)
    line = cy + SMILE * (1.0 - ((u - 653.0) / 448.0) ** 2) - smile(cx)
    down = np.exp(-0.5 * ((v - line) / (height / (2.0 * _UX2))) ** 2)
    darkening[y0:y1, x0:x1] += depth * across * down


def _image(darkening: np.ndarray, seed: int) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(0.0, NOISE, darkening.shape)
    pixels = np.clip(np.round(MEMBRANE - darkening + noise), 0, 65535).astype(np.uint16)
    pixels.flags.writeable = False  # cached: shared by the tests
    return pixels


@functools.cache
def blot_pixels(degrees: float = 0.0, rows: tuple = ROWS) -> np.ndarray:
    """The chemiluminescence blot: each row's band in every lane."""
    darkening = np.zeros((HEIGHT, WIDTH))
    for kda, depth, height, *shape in rows:
        for x in LANES:
            _darken(darkening, x, kda, depth, height, degrees, *shape)
    return _image(darkening, 58)


@functools.cache
def marker_pixels(degrees: float = 0.0) -> np.ndarray:
    """The marker: a ladder at each side, 250 ... 10 kDa."""
    darkening = np.zeros((HEIGHT, WIDTH))
    for x in (X_LEFT, X_RIGHT):
        for kda in KDA:
            _darken(darkening, x, kda, 12000.0, 9.0, degrees)
    return _image(darkening, 59)


@dataclasses.dataclass(frozen=True)
class Blot:
    session: ProjectSession
    membrane: str
    blot: str
    marker: str


def calibrated(
    tmp_path: Path,
    *,
    degrees: float = 0.0,
    rows: tuple = ROWS,
    sides: Sequence[LadderSide] = (LEFT, RIGHT),
    hook=None,
    pixels: np.ndarray | None = None,
) -> Blot:
    """The blot and its marker imported onto one membrane and linked, the
    PageRuler Plus preset chosen, every ladder band of ``sides`` marked where
    it lies (not snapped), and the sample's eight lanes declared. ``pixels``
    replaces the blot drawn from ``rows``."""
    s = session_on(tmp_path, hook)
    blot = blot_pixels(degrees, rows) if pixels is None else pixels
    blot_id = import_blot(s, np.array(blot), "blot β.tif")
    membrane = s.project.batch.membrane_of(blot_id).id
    marker_id = import_blot(
        s,
        np.array(marker_pixels(degrees)),
        "marker α.tif",
        kind=ImageKind.VISIBLE_MARKER,
        membrane_id=membrane,
    )
    ops.set_marker_image(s, blot_id, marker_id)
    ops.set_ladder(s, membrane, PRESET)
    for side in sides:
        at = X_LEFT if side is LEFT else X_RIGHT
        for kda in KDA:
            x, y = truth(kda, at, degrees)
            ops.add_calibration_point(s, marker_id, y, kda, MARKER_BAND, x=x, side=side, snap=False)
    ops.set_lanes(
        s,
        [
            LaneInput(c, sample)
            for c, sample in zip(samples.CONDITIONS, samples.SAMPLES, strict=True)
        ],
    )
    return Blot(s, membrane, blot_id, marker_id)


def add(b: Blot, row: tuple = TARGET, name: str = "β-catenin", **kwargs) -> str:
    role = Role.TARGET if row is not LOADING else Role.LOADING_CONTROL
    return ops.add_protein(b.session, name, role, b.blot, expected_mw=row[0], **kwargs)


def on_truth(s: ProjectSession, protein_id: str, kda: float, degrees: float = 0.0) -> list[int]:
    """The lanes whose box holds the true centre of the protein's band there."""
    size = protein_of(s, protein_id).box_size
    held = []
    for lane, band in sorted(lane_bands(s, protein_id).items()):
        x0, y0, x1, y1 = band.box.rect(size)
        x, y = truth(kda, LANES[lane], degrees)
        if x0 <= x <= x1 and y0 <= y <= y1:
            held.append(lane)
    return held


def mws(s: ProjectSession, protein_id: str) -> list[float]:
    return [band.apparent_mw for _, band in sorted(lane_bands(s, protein_id).items())]


def unchanged(s: ProjectSession, call) -> OperationError:
    """A refusal that changed nothing: the same project object, no entry."""
    before, entries = s.project, len(s.project.log)
    with pytest.raises(OperationError) as refused:
        call()
    assert s.project is before and len(s.project.log) == entries
    return refused.value


def strict_json(value: object) -> None:
    json.dumps(value, allow_nan=False)


# --- The slot (D11), from the calibration alone ---


def fitted_from(points: Sequence[CalibrationPoint]) -> mwcal.Calibration:
    """The calibration of ``img-2`` in a model-only project whose marker
    ``img-1`` holds ``points`` (:func:`test_checks.blot`, 1330 x 500)."""
    project = model_blot(points)
    fitted = mwcal.calibration_for(project.batch.membranes[0], "img-2")
    assert isinstance(fitted, mwcal.Calibration)
    return fitted


def marked(*points: tuple[float, float], x: float = X_LEFT) -> list[CalibrationPoint]:
    """Ladder bands ``(y, kDa)`` on ``img-1`` at ``x``."""
    return [
        CalibrationPoint(image_id="img-1", y=y, mw=mw, source=MARKER_BAND, x=x) for y, mw in points
    ]


def test_slot_margin_from_curve_scale():
    # 0.04 decade of MW above and below: 12 px on the sample's 300 px per
    # decade, and at least 6 px.
    fitted = fitted_from(model_ladder(X_LEFT))  # the law, one ladder
    slot = mwrow.slot(fitted, [92.0], 0.1, 141, 1165, HEIGHT)
    assert math.isclose(slot.margin, 12.0, rel_tol=1e-9)
    assert math.isclose(slot.expected_y[0], model_law(92.0), rel_tol=1e-12)
    top, bottom = model_law(92.0 * 1.2) - 12.0, model_law(92.0 / 1.2) + 12.0
    assert slot.y0 <= top + 1e-9 and top < slot.y0 + 1  # floor
    assert slot.y1 >= bottom - 1e-9 and bottom > slot.y1 - 1  # ceil
    assert (slot.x0, slot.x1, slot.row) == (141, 1165, (141, slot.y0, 1165, slot.y1))
    assert math.isclose(slot.m_top, 92.0 * 1.2) and math.isclose(slot.m_bot, 92.0 / 1.2)
    # One ladder: the protein line is level, and so is the slot.
    assert slot.shifts == (0,) * 1024 and slot.shift_ends == (0, 0)
    assert slot.slope_deg == 0.0 and not slot.steep
    # 100 px per decade: 4 px, less than the least margin.
    tight = fitted_from(marked((100.0, 100.0), (200.0, 10.0)))
    assert mwrow.slot(tight, [50.0], 0.1, 141, 1165, HEIGHT).margin == 6.0


def test_slot_clipped_to_range():
    # Points at 100 and 70 kDa reach 55.6 to 125.9 kDa (a tenth of a decade
    # past each): a 60-kDa target's slot stops at 55.6 kDa, not at 50.
    fitted = fitted_from(marked((100.0, 100.0), (140.0, 70.0)))
    slot = mwrow.slot(fitted, [60.0], 0.1, 141, 1165, HEIGHT)
    assert math.isclose(slot.m_bot, 10.0**fitted.z_lo) and math.isclose(slot.m_top, 72.0)
    assert math.isclose(slot.expected_y[0], 157.29, abs_tol=0.005)
    end = fitted.curve_at(None).y_at_log(fitted.z_lo)
    assert slot.y1 == math.ceil(end + slot.margin)
    # And to the image: the rows every column can be shifted in.
    assert mwrow.slot(fitted, [60.0], 0.1, 141, 1165, 170).y1 == 170


def test_the_slot_follows_the_protein_line():
    # Two ladders turned by 2 degrees: at 92 kDa the line drops 2 degrees to
    # the right, and each column's shift is its whole rows below the centre.
    fitted = fitted_from([*model_ladder(X_LEFT, 2.0), *model_ladder(X_RIGHT, 2.0, side=RIGHT)])
    slot = mwrow.slot(fitted, [92.0], 0.1, 141, 1165, HEIGHT)
    assert math.isclose(slot.slope_deg, 2.0, abs_tol=1e-6) and not slot.steep
    centre = fitted.y_at(92.0, slot.centre)
    for x in (141, 400, 653, 1000, 1164):
        assert slot.shifts[x - 141] == round(fitted.y_at(92.0, x + 0.5) - centre)
    assert slot.shift_ends == (slot.shifts[0], slot.shifts[-1]) == (-18, 18)
    assert slot.shift_at(0.0) == -18 and slot.shift_at(5000.0) == 18
    # Past 3 degrees, noted: the boxes stay level rectangles.
    steep = fitted_from([*model_ladder(X_LEFT, 3.5), *model_ladder(X_RIGHT, 3.5, side=RIGHT)])
    assert mwrow.slot(steep, [92.0], 0.1, 141, 1165, HEIGHT).steep
    # A slot no row of which stays on the image at every column.
    with pytest.raises(mwrow.SlotError) as refused:
        mwrow.slot(fitted, [92.0], 0.1, 141, 1165, 30)
    assert refused.value.code == "out_of_image"


def test_span_between_ladders_rule():
    # Each ladder a lane pitch outside its end lane: 128 px on the sample's
    # geometry, and the span half a pitch inside each ladder.
    two = fitted_from([*model_ladder(X_LEFT), *model_ladder(X_RIGHT, side=RIGHT)])
    assert mwrow.span_between_ladders(two, 8, WIDTH) == (141, 1165)
    assert mwrow.span_between_ladders(two, 8, 1100) == (141, 1100)  # clipped to the image
    assert mwrow.span_between_ladders(two, 0, WIDTH) is None
    assert mwrow.span_between_ladders(fitted_from(model_ladder(X_LEFT)), 8, WIDTH) is None
    # From the lanes placed: half their pitch past the end lanes' expected centres.
    anchors = [(333.0, 1), (461.0, 2), (845.0, 5)]
    assert mwrow.span_of_anchors(anchors, 8, WIDTH) == (141, 1165)
    assert mwrow.span_of_anchors(anchors[:1], 8, WIDTH) is None
    # Numbered right to left (a mirrored image): the same columns.
    mirrored = [(1330.0 - x, lane) for x, lane in anchors]
    assert mwrow.span_of_anchors(mirrored, 8, WIDTH) == (165, 1189)


# --- Placing a row by its MW ---


def placed_well(s: ProjectSession, protein_id: str, row: tuple, degrees: float = 0.0) -> None:
    """Every lane boxed on its band, MW-guided and counted, its apparent MW
    within ±10% of the expected one (the smile reads the middle lanes up to 8%
    light); the stored values current."""
    kda = row[0]
    bands = lane_bands(s, protein_id)
    assert sorted(bands) == list(range(8))
    assert on_truth(s, protein_id, kda, degrees) == list(range(8))
    assert all(band.source is ProposalSource.MW_GUIDED for band in bands.values())
    assert all(band.bands_found == 1 for band in bands.values())
    assert all(0.9 * kda <= mw <= 1.0 * kda for mw in mws(s, protein_id)), mws(s, protein_id)
    assert_mw_current(s)
    assert_nets_current(s)


def test_two_ladder_row_at_minus_2_degrees(tmp_path):
    # The blot and its ladders turned by -2 degrees (the right side up): two
    # ladders give the protein line, and the slot follows it, 18 px higher at
    # the right end than in the middle. Both rows, 16 of 16 lanes, each box on
    # its band and within ±10% of its MW.
    b = calibrated(tmp_path, degrees=-2.0)
    s = b.session
    loading, target = add(b, LOADING, "α-tubulin"), add(b, TARGET)
    first = ops.detect_mw_row(s, loading)
    assert (first.span_from, first.two_ladders) == ("ladders", True)
    assert math.isclose(first.tilt_deg, -2.0, abs_tol=0.01)
    assert first.shift_ends == (18, -18) and first.flags == () and first.notes == ()
    second = ops.detect_mw_row(s, target)
    assert second.span_from == "anchors"  # the lanes α-tubulin's boxes placed
    assert second.shift_ends == (18, -18)
    for protein, row in ((loading, LOADING), (target, TARGET)):
        placed_well(s, protein, row, -2.0)
    # One ladder, the left, cannot see the tilt: its level slot reads the
    # right half of the blot too heavy (8 of 16 outside ±10%).
    c = calibrated(tmp_path / "one ladder", degrees=-2.0, sides=(LEFT,))
    within = 0
    for row, name in ((LOADING, "α-tubulin"), (TARGET, "β-catenin")):
        protein = add(c, row, name)
        ops.detect_mw_row(c.session, protein, span=first.span)
        within += sum(abs(mw / row[0] - 1.0) <= 0.1 for mw in mws(c.session, protein))
    assert within < 16


def test_a_steep_row_is_noted(tmp_path, monkeypatch):
    # A protein line sloping past STEEP_ROW_DEG (3 degrees) across the span is
    # noted: the boxes stay level rectangles. At 2 degrees against a limit of
    # 1.5 here.
    monkeypatch.setattr(mwrow, "STEEP_ROW_DEG", 1.5)
    b = calibrated(tmp_path, degrees=-2.0)
    placed = ops.detect_mw_row(b.session, add(b, LOADING, "α-tubulin"))
    note = "the row slopes steeply along the protein line (about 2.0°); boxes are level rectangles"
    assert placed.notes == (note,) and placed.flags == ()
    assert b.session.project.log[-1].params["notes"] == [note]
    assert math.isclose(placed.slope_deg, -2.0, abs_tol=0.01)


@pytest.mark.parametrize("degrees", [0.0, 2.0, -2.0])
def test_span_between_ladders(tmp_path, degrees):
    # No lanes placed yet: the span lies between the two ladders, each taken a
    # lane pitch outside its end lane (128 px here), inset by half a pitch:
    # 141 to 1165 on the level blot. Every lane placed, none doubtful.
    b = calibrated(tmp_path, degrees=degrees)
    target = add(b, TARGET)
    placed = ops.detect_mw_row(b.session, target)
    assert placed.span_from == "ladders" and placed.span_hint is None
    x0, x1 = placed.span
    assert abs(x0 - 141) <= 1 and abs(x1 - 1165) <= 1
    assert "doubtful_lanes" not in placed.flags
    placed_well(b.session, target, TARGET, degrees)


def test_span_from_anchors_after_first_row(tmp_path):
    # One ladder: the first row is dragged (only its x counts), and the next
    # protein's row by MW takes its span from the lanes that row placed, half
    # a pitch past the end lanes.
    b = calibrated(tmp_path, sides=(LEFT,))
    s = b.session
    loading, target = add(b, LOADING, "α-tubulin"), add(b, TARGET)
    dragged = ops.detect_mw_row(s, loading, span=(141, 1165))
    assert (dragged.span, dragged.span_from) == ((141, 1165), "given")
    placed = ops.detect_mw_row(s, target)
    assert placed.span_from == "anchors"
    boxes = lane_bands(s, loading)
    size = protein_of(s, loading).box_size
    centres = [boxes[lane].box.x + size.width / 2 for lane in range(8)]
    pitch = (centres[-1] - centres[0]) / 7
    assert abs(placed.span[0] - (centres[0] - pitch / 2)) <= 3
    assert abs(placed.span[1] - (centres[-1] + pitch / 2)) <= 3
    entry = s.project.log[-1].params
    assert entry["span_from"] == "anchors" and entry["anchor_image_id"] == b.blot
    assert entry["anchor_ids"] == [boxes[lane].id for lane in range(8)]
    placed_well(s, target, TARGET)
    # Its own MW-guided boxes give way to its next placement: the loading
    # control's lanes set the span again.
    before = s.project
    again = ops.detect_mw_row(s, target)
    assert again.span == placed.span and s.project is before


def test_span_from_group_anchors_on_reprobe(tmp_path):
    # A reprobe imaged in the same position, linked to the same marker: one
    # register group. Its first row by MW takes the span of the blot's lanes.
    b = calibrated(tmp_path, sides=(LEFT,))
    s = b.session
    reprobe = import_blot(s, np.array(blot_pixels()), "reprobe γ.tif", membrane_id=b.membrane)
    ops.set_marker_image(s, reprobe, b.marker)
    loading = add(b, LOADING, "α-tubulin")
    ops.detect_mw_row(s, loading, span=(141, 1165))
    target = ops.add_protein(s, "β-catenin", Role.TARGET, reprobe, expected_mw=TARGET[0])
    placed = ops.detect_mw_row(s, target)
    assert placed.span_from == "group_anchors"
    assert s.project.log[-1].params["anchor_image_id"] == b.blot
    assert on_truth(s, target, TARGET[0]) == list(range(8))
    assert_mw_current(s)


def test_lane_span_required_without_anchors(tmp_path):
    b = calibrated(tmp_path, sides=(LEFT,))
    s = b.session
    target = add(b, TARGET)
    error = unchanged(s, lambda: ops.detect_mw_row(s, target))
    assert (error.code, error.ids) == (ErrorCode.LANE_SPAN_REQUIRED, (b.blot,))
    assert str(error) == (
        f"no lanes are placed on {b.blot}, {b.marker} and no two ladders show where they"
        " lie: drag across all 8 lanes (only the left and right ends are used)"
    )


def test_span_given(tmp_path):
    # The columns dragged across, clipped to the image; the rows are the slot's.
    b = calibrated(tmp_path)
    target = add(b, TARGET)
    placed = ops.detect_mw_row(b.session, target, span=(141, 5000))
    assert (placed.span, placed.span_from) == ((141, WIDTH), "given")
    assert (placed.row[0], placed.row[2]) == (141, WIDTH)
    assert b.session.project.log[-1].params["anchor_ids"] == []
    assert on_truth(b.session, target, TARGET[0]) == list(range(8))


def test_target_outside_range_refused(tmp_path):
    # Points at 100 and 70 kDa reach 55.6 to 126 kDa (named in whole kDa): 40 kDa
    # is refused, not guessed.
    b = calibrated(tmp_path, sides=())
    s = b.session
    for kda in (100, 70):
        x, y = truth(kda, X_LEFT)
        ops.add_calibration_point(s, b.marker, y, kda, MARKER_BAND, x=x, snap=False)
    target = ops.add_protein(s, "LC3-II", Role.TARGET, b.blot, expected_mw=40)
    error = unchanged(s, lambda: ops.detect_mw_row(s, target, span=(141, 1165)))
    assert (error.code, error.ids) == (ErrorCode.MW_OUTSIDE_CALIBRATION, (b.blot,))
    assert str(error) == (
        f"40 kDa lies outside the calibrated range of {b.blot}, {b.marker} (126–56 kDa)"
    )
    # Just past the top (125.89 kDa), the top is named to the decimal that shows it.
    heavier = ops.add_protein(s, "HSP90", Role.TARGET, b.blot, expected_mw=126)
    error = unchanged(s, lambda: ops.detect_mw_row(s, heavier, span=(141, 1165)))
    assert error.code is ErrorCode.MW_OUTSIDE_CALIBRATION
    assert str(error) == (
        f"126 kDa lies outside the calibrated range of {b.blot}, {b.marker} (125.9–56 kDa)"
    )
    # With two ladders, the range is where both reach.
    c = calibrated(tmp_path / "two", sides=())
    for side, at, kdas in ((LEFT, X_LEFT, (250, 100, 55, 35)), (RIGHT, X_RIGHT, (100, 55, 35))):
        for kda in kdas:
            x, y = truth(kda, at)
            ops.add_calibration_point(
                c.session, c.marker, y, kda, MARKER_BAND, x=x, side=side, snap=False
            )
    heavy = ops.add_protein(c.session, "HSP110", Role.TARGET, c.blot, expected_mw=150)
    error = unchanged(c.session, lambda: ops.detect_mw_row(c.session, heavy))
    assert error.code is ErrorCode.MW_OUTSIDE_CALIBRATION
    assert str(error).endswith("(126–28 kDa, where both ladders reach)")


def test_a_refused_mw_reads_past_the_range_named():
    # The range is named in whole kDa, as the notices name it, but never so
    # that the refused MW seems to lie inside it: the end it lies past gets the
    # digits it takes, and the MW its own where {mw:g} would round it onto the
    # range.
    fitted = fitted_from(marked((100.0, 100.0), (140.0, 70.0)))  # 125.893 to 55.603 kDa
    words = functools.partial(mwrow.outside_words, fitted, names="img-1, img-2")
    assert words(40.0) == "40 kDa lies outside the calibrated range of img-1, img-2 (126–56 kDa)"
    for mw, named in (
        (300.0, "126–56"),
        (126.0, "125.9–56"),
        (125.95, "125.9–56"),
        (125.9, "125.89–56"),
        (125.8926, "125.89–56"),
        (55.59, "126–56"),
        (55.6029, "126–56"),
    ):
        assert words(mw) == (
            f"{mw:g} kDa lies outside the calibrated range of img-1, img-2 ({named} kDa)"
        )
    # A bottom end that whole kDa round down: 55.21 kDa.
    lower = fitted_from(marked((100.0, 100.0), (140.0, 69.5)))
    low = functools.partial(mwrow.outside_words, lower, names="img-1, img-2")
    assert low(50.0).endswith("(126–55 kDa)")
    assert low(55.0).endswith("(126–55.2 kDa)")
    assert low(55.2).endswith("(126–55.21 kDa)")
    # 55.20005005 kDa, just under 55.2000501, would be written 55.2001: inside it.
    edge = fitted_from(marked((100.0, 100.0), (140.0, 55.2000501 * 10**0.1)))
    assert f"{55.20005005:g}" == "55.2001" and 55.20005005 < 10.0**edge.z_lo
    assert mwrow.outside_words(edge, 55.20005005, "img-1") == (
        "55.20005005 kDa lies outside the calibrated range of img-1 (126–55.2001 kDa)"
    )
    # With two ladders, where both reach.
    two = fitted_from([*model_ladder(X_LEFT), *model_ladder(X_RIGHT, side=RIGHT)])
    assert two.two_ladders
    assert mwrow.outside_words(two, 1000.0, "img-1").endswith(", where both ladders reach)")


def test_refusals_change_nothing(tmp_path):
    b = calibrated(tmp_path)
    s = b.session
    target = add(b, TARGET)
    for span in ((1165, 141), (141,), "141", (141.0, 1165), (True, 1165)):
        error = unchanged(s, lambda span=span: ops.detect_mw_row(s, target, span=span))
        assert error.code is ErrorCode.INVALID_INPUT
    error = unchanged(s, lambda: ops.detect_mw_row(s, target, span=(-500, -10)))
    assert error.code is ErrorCode.OUT_OF_IMAGE
    with pytest.raises(UnknownIdError):
        ops.detect_mw_row(s, "prot-99")
    # No expected MW.
    plain = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, b.blot)
    error = unchanged(s, lambda: ops.detect_mw_row(s, plain))
    assert (error.code, str(error), error.ids) == (
        ErrorCode.MW_REQUIRED,
        "'GAPDH' has no expected MW: enter it to place its row by MW",
        (plain,),
    )
    # Several bands expected: their MWs come with #58's multi-band rows.
    plant(s, lambda draft: setattr(draft.batch.find_protein(target), "expected_band_count", 2))
    error = unchanged(s, lambda: ops.detect_mw_row(s, target))
    assert error.code is ErrorCode.MW_REQUIRED and "each of the 2" in str(error)
    # No lanes.
    c = calibrated(tmp_path / "no lanes")
    ops.set_lanes(c.session, [])
    lanes_later = add(c, TARGET)
    error = unchanged(c.session, lambda: ops.detect_mw_row(c.session, lanes_later))
    assert error.code is ErrorCode.NO_LANES


def test_no_calibration_is_refused_with_the_reason(tmp_path):
    b = calibrated(tmp_path, sides=())
    s = b.session
    target = add(b, TARGET)
    error = unchanged(s, lambda: ops.detect_mw_row(s, target, span=(141, 1165)))
    assert (error.code, str(error), error.ids) == (
        ErrorCode.NO_CALIBRATION,
        f"{b.blot}, {b.marker} have no calibration points: mark the ladder on the marker"
        f" image {b.marker}",
        (b.blot,),
    )
    x, y = truth(100, X_LEFT)
    ops.add_calibration_point(s, b.marker, y, 100, MARKER_BAND, x=x, snap=False)
    error = unchanged(s, lambda: ops.detect_mw_row(s, target, span=(141, 1165)))
    assert str(error) == (
        f"{b.blot}, {b.marker}: each ladder there has one calibration point; mark at least two"
    )
    ops.set_marker_image(s, b.blot, None)
    error = unchanged(s, lambda: ops.detect_mw_row(s, target, span=(141, 1165)))
    assert str(error) == (
        f"{b.blot} has no calibration points: mark the ladder on its marker image and link"
        f" {b.blot} to it, or mark the ladder on {b.blot} itself"
    )


STRONGER = (TARGET[0] / 1.15, 2 * TARGET[1], 12.0)  # 15% lighter, twice as deep
THIN_TARGET = (TARGET[0], TARGET[1], 12.0)


def test_prefer_y_picks_predicted_band(tmp_path):
    # A band twice as deep 15% lighter (18 px lower) in every lane, in the slot
    # too (±20%): each lane grows from the peak nearest the expected MW's row,
    # so the protein's own band is boxed. The same rows dragged as a row box
    # take the deeper band. (Nearest the row: in the middle lanes the smile
    # puts each band 10 px below the ladders' line, so a band that far above
    # it would be nearer.)
    b = calibrated(tmp_path, rows=(STRONGER, THIN_TARGET))
    s = b.session
    target = add(b, THIN_TARGET)
    placed = ops.detect_mw_row(s, target)
    assert on_truth(s, target, TARGET[0]) == list(range(8))
    assert all(0.9 * TARGET[0] <= mw <= TARGET[0] for mw in mws(s, target))
    assert all(band.bands_found == 1 for band in lane_bands(s, target).values())
    dragged = ops.add_protein(s, "β-catenin dragged", Role.TARGET, b.blot, expected_mw=92)
    ops.detect_row_boxes(s, dragged, placed.row)
    assert on_truth(s, dragged, STRONGER[0]) == list(range(8))


FAINT_TARGET = (TARGET[0], 3000.0, 16.0)
DEEPER_BELOW = (80.0, 9000.0, 16.0)  # three times as deep, 18 px below the target
DEEPER_ABOVE = (106.0, 9000.0, 16.0)  # ... 18 px above it


def test_a_deeper_band_below_is_not_boxed_with_the_target(tmp_path):
    # Each lane grows from the target's own peak, at 30% of it: growth used to
    # climb the dip into a band three times as deep 18 px below, one box over
    # both (100 x 36) in every lane. It stops at the valley: each box on the
    # target alone, of a lone target's size, its apparent MW the target's, and
    # the deeper band reported as a second component.
    lone = calibrated(tmp_path / "lone", rows=(FAINT_TARGET,))
    alone = add(lone, FAINT_TARGET)
    ops.detect_mw_row(lone.session, alone)
    b = calibrated(tmp_path, rows=(FAINT_TARGET, DEEPER_BELOW))
    s = b.session
    target = add(b, FAINT_TARGET)
    placed = ops.detect_mw_row(s, target)
    assert on_truth(s, target, TARGET[0]) == list(range(8))
    assert on_truth(s, target, DEEPER_BELOW[0]) == []
    size, lone_size = protein_of(s, target).box_size, protein_of(lone.session, alone).box_size
    assert size.height <= lone_size.height + 1 and size.width <= lone_size.width + 2
    assert "multiple_components" in placed.flags
    assert all(0.9 * TARGET[0] <= mw <= TARGET[0] for mw in mws(s, target))
    assert all(band.bands_found == 1 for band in lane_bands(s, target).values())  # 18 px off
    assert_mw_current(s)


@pytest.mark.parametrize(
    ("kda", "rel", "crossed"),
    [(104.0, 2.0, True), (104.0, 3.0, True), (106.0, 3.0, False)],
)
def test_a_deeper_band_above_is_never_boxed_with_the_target(tmp_path, kda, rel, crossed):
    # A band rel times as deep 16 px (104 kDa) or 18 px (106 kDa) above the
    # target. In the middle lanes the smile puts the expected row 10 px above
    # the target, nearer the deeper band's peak: those lanes grow from it, the
    # others from the target's. At 104 kDa twice as deep, the growth from the
    # deeper band's peak ran over the target: boxes 86 x 29 (a lone target's
    # 86 x 15), the target its own in those lanes, apparent MWs 87-92 kDa.
    # Three times as deep, lanes 3-6 boxed the deeper band alone, their
    # apparent MWs 96-98 kDa, within 10%, as a smile. Neighbouring lanes grown
    # from bands on two rows, each holding both, now refuse the row, and
    # nothing is placed. (At 106 kDa the boxes lie too far apart for a smile.)
    b = calibrated(tmp_path, rows=(FAINT_TARGET, (kda, rel * FAINT_TARGET[1], 16.0)))
    s = b.session
    target = add(b, FAINT_TARGET)
    error = unchanged(s, lambda: ops.detect_mw_row(s, target))
    assert error.code is ErrorCode.ROW_OFF_LINE
    assert error.detail["cause"] == "off_row_line"
    if crossed:  # lanes 2 and 3, 6 and 7 grew from bands on two rows
        assert str(error).startswith("the bands found in lanes 2, 3, 6, 7 lie on two rows,")
        assert "expected row" in str(error)


@pytest.mark.parametrize(
    ("depth", "other", "deeper"),
    [
        (70000.0, 80.0, 150000.0),
        (90000.0, 80.0, 150000.0),
        (70000.0, 80.0, 110000.0),
        (150000.0, 80.0, 150000.0),
        (70000.0, 104.0, 150000.0),
        (90000.0, 104.0, 150000.0),
    ],
)
def test_a_band_as_saturated_as_the_target_is_not_boxed_with_it(tmp_path, depth, other, deeper):
    # The target clipped at 0, and a band 18 px below it (80 kDa) or 16 px
    # above it (104 kDa) as deep or deeper, clipped too: their tops differ by
    # the background alone, and the growth from the target's peak climbed
    # into the other band, every box 96 x 34 to 100 x 38 over both, at most
    # the row box's edge flagged. Neither is clearly lower: each box on the
    # target alone, the other band a second component. Where the valley
    # between the two clipped cores stays above VALLEY_FRAC of their clipped
    # height (both 150000 deep at 80 kDa, 104 kDa), each lane held one peak
    # over both; clipped, the valley is deeper than it shows, and the cores
    # are two peaks.
    row = (TARGET[0], depth, 16.0)
    b = calibrated(tmp_path, rows=(row, (other, deeper, 16.0)))
    s = b.session
    target = add(b, row)
    placed = ops.detect_mw_row(s, target)
    assert on_truth(s, target, TARGET[0]) == list(range(8))
    assert on_truth(s, target, other) == []
    assert "multiple_components" in placed.flags
    assert all(0.9 * TARGET[0] <= mw <= 1.1 * TARGET[0] for mw in mws(s, target))


BURNT_TARGET = (TARGET[0], 120000.0, 10.0)  # clipped at 0, over twice the membrane


def burnt_out_pixels(light: float, smear: float) -> np.ndarray:
    """The blot of BURNT_TARGET, each band lightened by ``light`` in its
    centre through its whole height and more (burnt out), and a smear
    ``smear`` deep under its left half (from 40 to 16 px left of its centre)
    from 8 px below its centre down, fading over 20 px."""
    kda, depth, height = BURNT_TARGET
    darkening = np.zeros((HEIGHT, WIDTH))
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    for x in LANES:
        _darken(darkening, x, kda, depth, height, 0.0)
        cx, cy = truth(kda, x)
        darkening -= light * np.exp(-0.5 * (((xs - cx) / 12.0) ** 2 + ((ys - cy) / 18.0) ** 2))
        across = (
            1.0 / (1.0 + np.exp(-(xs - cx + 40.0) / 2.0)) / (1.0 + np.exp((xs - cx + 16.0) / 2.0))
        )
        d = ys - cy
        down = np.where(d < 0.0, 0.0, np.where(d < 8.0, 1.0, np.exp(-(d - 8.0) / 20.0)))
        darkening += smear * across * down
    return _image(darkening, 58)


@pytest.mark.parametrize("smear", [0.0, 2500.0, 4000.0])
def test_a_burnt_out_band_with_a_smear_under_one_half_is_boxed_whole(tmp_path, smear):
    # #121: each target band clipped, its centre burnt out below the noise,
    # splitting it along x; a faint smear under its left half. The smear
    # drained into that half's basin, which then spanned far more rows than
    # the other half's: the halves were read as two bands, the other half
    # left out of the growth, every box 42 x 10 on one half (none on the
    # band's centre), a second component flagged, and the over-exposure
    # warning lost. The halves span the same rows at the growth level: one
    # hollow band, boxed whole, as without the smear.
    b = calibrated(tmp_path, rows=(BURNT_TARGET,), pixels=burnt_out_pixels(120000.0, smear))
    s = b.session
    target = add(b, BURNT_TARGET)
    placed = ops.detect_mw_row(s, target)
    assert "hollow_band" in placed.flags
    assert "multiple_components" not in placed.flags
    assert on_truth(s, target, TARGET[0]) == list(range(8))
    assert protein_of(s, target).box_size.width >= 90


DUMBBELL_TARGET = (TARGET[0], TARGET[1], 8.0, 0.5, 0.05)  # half as deep in its middle


@pytest.mark.parametrize("degrees", [-1.0, -2.0])
def test_a_sloping_dumbbell_band_is_boxed_whole(tmp_path, degrees):
    # Thin bands half as deep in their middle as at their ends, the right end
    # a little darker, on a blot turned 1 or 2 degrees with one ladder: each
    # band's ends lie a pixel or two apart in rows. Grown from its weaker end,
    # a band's other end was read as a band above or below it and cut off at
    # the middle: the box 23 px off the band, over membrane, in lane 8 (at 2
    # degrees lanes 6-8), two bands counted there. Each band is boxed whole.
    b = calibrated(tmp_path, degrees=degrees, rows=(DUMBBELL_TARGET,), sides=(LEFT,))
    s = b.session
    target = add(b, DUMBBELL_TARGET)
    placed = ops.detect_mw_row(s, target, span=(140, 1170))
    assert on_truth(s, target, TARGET[0], degrees) == list(range(8))
    size = protein_of(s, target).box_size
    for lane, band in lane_bands(s, target).items():
        x0, _, x1, _ = band.box.rect(size)
        assert abs((x0 + x1) / 2 - truth(TARGET[0], LANES[lane], degrees)[0]) <= 3.0, lane
        assert band.bands_found == 1, lane
    assert "multiple_components" not in placed.flags


def test_log_params(tmp_path):
    b = calibrated(tmp_path, degrees=2.0, hook=save_to_folder)
    s = b.session
    target = add(b, TARGET)
    placed = ops.detect_mw_row(s, target)
    entry = s.project.log[-1]
    assert entry.action == "detect_mw_row"
    params = entry.params
    fit = ops.calibration_fit(s.project.batch.membranes[0], b.blot)
    assert list(params)[:18] == [
        "protein_id",
        "expected_mws",
        "mw_tolerance",
        "search_factor",
        "fit",
        "two_ladders",
        "tilt_deg",
        "expected_y",
        "row",
        "shift_ends",
        "m_top",
        "m_bot",
        "margin",
        "slope_deg",
        "span",
        "span_from",
        "anchor_ids",
        "anchor_image_id",
    ]
    assert {
        key: params[key]
        for key in ("protein_id", "expected_mws", "mw_tolerance", "search_factor", "fit")
    } == {
        "protein_id": target,
        "expected_mws": [92.0],
        "mw_tolerance": 0.1,
        "search_factor": 2.0,
        "fit": fit.as_json(),
    }
    assert params["two_ladders"] is True and math.isclose(params["tilt_deg"], 2.0, abs_tol=0.01)
    assert params["expected_y"] == [round(placed.expected_y[0], 2)]
    assert (params["row"], params["shift_ends"]) == (list(placed.row), [-18, 18])
    assert math.isclose(params["m_top"], 92 * 1.2) and math.isclose(params["m_bot"], 92 / 1.2)
    assert math.isclose(params["margin"], 12.0, abs_tol=0.05)  # 300 px per decade, turned
    assert (params["span"], params["span_from"]) == (list(placed.span), "ladders")
    assert (params["anchor_ids"], params["anchor_image_id"]) == ([], None)
    assert params["settings"] == rowdetect.settings()
    strict_json(params)
    assert load_project(s.folder) == s.project
    # Then what a row box logs, from lanes to settings (the same rows, dragged).
    ops.detect_row_boxes(s, target, placed.row)
    assert list(params)[18:] == list(s.project.log[-1].params)[2:]


@pytest.mark.parametrize("padding", [None, (2, 3)])
def test_same_placement_again_is_noop(tmp_path, padding):
    # The same placement again finds the same bands where its boxes are: no
    # change, no entry. With padding (N1 §8): the fitted size plus the padding.
    b = calibrated(tmp_path)
    s = b.session
    target = add(b, TARGET)
    ops.detect_mw_row(s, target)
    if padding is not None:
        ops.set_box_padding(s, target, across=padding[0], along=padding[1])
        fitted = protein_of(s, target).fitted_size
        size = protein_of(s, target).box_size
        assert (size.width, size.height) == (fitted.width + 4, fitted.height + 6)
    before = s.project
    again = ops.detect_mw_row(s, target)
    assert s.project is before
    assert again.band_ids == tuple(band.id for _, band in sorted(lane_bands(s, target).items()))
    assert again.remeasured == ()


def test_undo_and_redo_a_placement(tmp_path):
    b = calibrated(tmp_path, hook=save_to_folder)
    s = b.session
    target = add(b, TARGET)
    before = s.project
    placed = ops.detect_mw_row(s, target)
    after = s.project
    done = ops.undo(s)
    assert (done.action, s.project.batch) == ("detect_mw_row", before.batch)
    assert done.removed == tuple(band_id for band_id in placed.band_ids if band_id is not None)
    redone = ops.redo(s)
    assert (redone.action, s.project.batch) == ("detect_mw_row", after.batch)
    assert_mw_current(s)
    assert load_project(s.folder) == s.project


def test_a_drawn_row_and_an_mw_row_replace_each_other(tmp_path):
    # Both are detector boxes nobody edited: each placement replaces the
    # other's in place, and the source follows the last one.
    b = calibrated(tmp_path)
    s = b.session
    target = add(b, TARGET)
    placed = ops.detect_mw_row(s, target)
    ids = placed.band_ids
    dragged = ops.detect_row_boxes(s, target, placed.row)
    assert dragged.band_ids == ids and dragged.replaced_band_ids == ids
    assert {band.source for band in lane_bands(s, target).values()} == {ProposalSource.ROW_BOX}
    again = ops.detect_mw_row(s, target)
    assert again.band_ids == ids
    assert {band.source for band in lane_bands(s, target).values()} == {ProposalSource.MW_GUIDED}
    # A box edited by hand stays.
    band = lane_bands(s, target)[3]
    x0, y0, x1, y1 = band.box.rect(protein_of(s, target).box_size)
    ops.move_box(s, band.id, (x0 + 1, y0, x1 + 1, y1))
    kept = ops.detect_mw_row(s, target)
    assert kept.kept_lanes == (3,)


def test_the_state_predicts_the_row(tmp_path):
    # Before a protein has boxes, the page draws its predicted slot along the
    # protein line and its missing lanes on it; after, the prediction stays.
    b = calibrated(tmp_path, degrees=-2.0)
    s = b.session
    target = add(b, TARGET)
    plain = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, b.blot)

    def state_of(protein_id: str) -> dict:
        project = project_state("blot", s, s.project, open_id=1)
        strict_json(project)
        return next(p for p in project["proteins"] if p["id"] == protein_id)

    predicted = state_of(target)["predicted_row"]
    fitted = mwcal.calibration_for(s.project.batch.membranes[0], b.blot)
    slot = mwrow.slot(fitted, [92.0], 0.1, *mwrow.span_between_ladders(fitted, 8, WIDTH), HEIGHT)
    assert predicted == {
        "row": list(slot.row),
        "shift_ends": [18, -18],
        "span_from": "ladders",
        "bands": [{"mw": 92.0, "y": slot.expected_y[0]}],
    }
    # The missing lanes lie along it: 18 px higher at the right end.
    missing = state_of(target)["missing_lanes"]
    assert [lane["y"] - slot.expected_y[0] for lane in missing] == [0.0] * 8  # no lanes placed
    assert state_of(plain)["predicted_row"] is None  # no expected MW
    placed = ops.detect_mw_row(s, target)
    assert state_of(target)["predicted_row"]["row"] == list(placed.row)
    loading = add(b, LOADING, "α-tubulin")
    predicted = state_of(loading)["predicted_row"]
    assert predicted["span_from"] == "anchors"  # the lanes β-catenin placed
    ys = {lane["lane_index"]: lane["y"] for lane in state_of(loading)["missing_lanes"]}
    assert ys[0] - ys[7] >= 30  # higher to the right, along the line
    # Outside the range: the bands' y is unknown, and no slot is drawn.
    ops.edit_protein(s, loading, expected_mw=5.0)
    assert state_of(loading)["predicted_row"] == {
        "row": None,
        "shift_ends": None,
        "span_from": "anchors",
        "bands": [{"mw": 5.0, "y": None}],
    }


def test_an_old_project_loads_and_predicts_unchanged(tmp_path):
    # The conftest sample (saved before #58): its β-catenin on img-2 reads a
    # predicted row from img-2's older calibration (three points, no x), and
    # nothing about the project changes.
    s = open_sample(tmp_path, hook=None)
    stored = (s.folder / "project.json").read_bytes()
    before = s.project
    state = project_state(s.folder.name, s, s.project, open_id=1)
    strict_json(state)
    [target] = [p for p in state["proteins"] if p["id"] == "prot-7"]
    predicted = target["predicted_row"]
    assert predicted["span_from"] == "anchors" and predicted["shift_ends"] == [0, 0]
    assert predicted["bands"][0]["mw"] == 92.0
    assert s.project is before and (s.folder / "project.json").read_bytes() == stored
    assert [p["predicted_row"] for p in state["proteins"] if p["id"] != "prot-7"] == [None, None]


# --- The public sample (one ladder, 1200 px) ---


@functools.cache
def _sample_bytes() -> dict[str, bytes]:
    return samples.sample_files()


def sample_calibrated(tmp_path: Path) -> Blot:
    """The public sample's blot and marker on one membrane, linked, PageRuler
    Plus chosen, its nine ladder bands marked where they lie, its eight lanes."""
    s = session_on(tmp_path)
    files = _sample_bytes()
    dark = Polarity.DARK_ON_LIGHT
    blot = ops.import_image(
        s,
        io.BytesIO(files[samples.BLOT_FILE]),
        samples.BLOT_FILE,
        kind=ImageKind.CHEMILUMINESCENCE,
        polarity=dark,
    )
    membrane = s.project.batch.membrane_of(blot).id
    marker = ops.import_image(
        s,
        io.BytesIO(files[samples.MARKER_FILE]),
        samples.MARKER_FILE,
        kind=ImageKind.VISIBLE_MARKER,
        polarity=dark,
        membrane_id=membrane,
    )
    ops.set_marker_image(s, blot, marker)
    ops.set_ladder(s, membrane, PRESET)
    for kda in samples.LADDER_KDA:
        y = samples.band_y(kda, samples.LADDER_X) + 0.5
        ops.add_calibration_point(s, marker, y, kda, MARKER_BAND, x=samples.LADDER_X, snap=False)
    ops.set_lanes(
        s,
        [
            LaneInput(c, sample)
            for c, sample in zip(samples.CONDITIONS, samples.SAMPLES, strict=True)
        ],
    )
    return Blot(s, membrane, blot, marker)


def sample_on_truth(s: ProjectSession, protein_id: str, row: samples.Row) -> list[int]:
    """The lanes whose box holds the sample band's nominal centre (its lane's
    nominal x and the law's y: within 2 px and 1 px of the drawn one)."""
    size = protein_of(s, protein_id).box_size
    held = []
    for lane, band in sorted(lane_bands(s, protein_id).items()):
        x0, y0, x1, y1 = band.box.rect(size)
        x = samples.LANE_X[lane]
        y = samples.band_y(row.kda, x) + 0.5
        if x0 + 2 <= x <= x1 - 2 and y0 + 1 <= y <= y1 - 1:
            held.append(lane)
    return held


def test_sample_loading_row(tmp_path):
    # α-tubulin at 50 kDa, the lanes dragged across once: rows 215 to 288,
    # every lane on its band, within ±10% (the smile reads the middle lanes low).
    b = sample_calibrated(tmp_path)
    s = b.session
    loading = ops.add_protein(
        s, samples.LOADING_CONTROL, Role.LOADING_CONTROL, b.blot, expected_mw=50
    )
    placed = ops.detect_mw_row(s, loading, span=(141, 1165))
    # 50 kDa x 1.2 at y = 227.2, 50 / 1.2 at 275.04, and 12 px more each way.
    assert placed.row == (141, 215, 1165, 288) and placed.shift_ends == (0, 0)
    assert (placed.flags, placed.two_ladders, placed.tilt_deg) == ((), False, None)
    assert sample_on_truth(s, loading, samples.LOADING_ROW) == list(range(8))
    assert all(0.9 * 50 <= mw <= 50 for mw in mws(s, loading))
    assert_mw_current(s)


def test_sample_target_row_lands_on_truth(tmp_path):
    # β-catenin at 92 kDa once α-tubulin is placed: no drag, rows 136 to 208,
    # every box on its band, MW-guided, within ±10% of 92 kDa.
    b = sample_calibrated(tmp_path)
    s = b.session
    loading = ops.add_protein(
        s, samples.LOADING_CONTROL, Role.LOADING_CONTROL, b.blot, expected_mw=50
    )
    ops.detect_mw_row(s, loading, span=(141, 1165))
    target = ops.add_protein(s, samples.TARGET, Role.TARGET, b.blot, expected_mw=92)
    placed = ops.detect_mw_row(s, target)
    assert placed.span_from == "anchors" and placed.row[1::2] == (136, 208)
    assert sample_on_truth(s, target, samples.TARGET_ROW) == list(range(8))
    bands = lane_bands(s, target).values()
    assert all(band.source is ProposalSource.MW_GUIDED for band in bands)
    assert all(0.9 * 92 <= mw <= 92 for mw in mws(s, target))
    assert_mw_current(s)
    assert_nets_current(s)


def test_curved_ladder_mw_within_tolerance(tmp_path):
    # A ladder at the vendor chart's Tris-glycine 4-20% positions, which no
    # single line fits (21-48% off at the ends); the blot's bands lie where
    # that curve puts them. Point to point, a 40-kDa target (between 55 and
    # 35 kDa) and a 180-kDa one (between 250 and 130) are placed on them.
    ys = [20.0 + 0.9 * (y - 336.5) for y in VENDOR_YS]  # into 400 rows
    points = [(y, kda) for y, kda in zip(ys, KDA, strict=True)]

    def curve_y(kda: float) -> float:
        z = math.log10(kda)
        for (y0, m0), (y1, m1) in itertools.pairwise(points):
            z0, z1 = math.log10(m0), math.log10(m1)
            if z1 <= z <= z0:
                return y0 + (z0 - z) / (z0 - z1) * (y1 - y0)
        raise AssertionError(kda)

    lanes = (120.0, 200.0, 280.0, 360.0)
    rows = (40.0, 180.0)
    blot = synthetic_blot(
        (420, 480),
        [(x, curve_y(kda) - 0.5, 14.0, 3.0, 20000.0) for x in lanes for kda in rows],
        dtype=np.float64,
    )
    s = session_on(tmp_path)
    blot_id = import_blot(s, noisy16(blot, 60), "vendor α.tif")
    for y, kda in points:
        ops.add_calibration_point(
            s, blot_id, y, kda, CalibrationPointSource.CHEMILUMINESCENCE_MARKER, x=30.0, snap=False
        )
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(4)])
    for kda in rows:
        protein = ops.add_protein(s, f"µ-{kda:g}", Role.TARGET, blot_id, expected_mw=kda)
        ops.detect_mw_row(s, protein, span=(80, 400))
        found = mws(s, protein)
        assert len(found) == 4 and all(abs(mw / kda - 1.0) <= 0.03 for mw in found), found
    assert_mw_current(s)


@pytest.mark.parametrize("lanes", [10, 7])
def test_lanes_read_between_the_ladders_ask_for_a_drag(tmp_path, lanes):
    # The span between the ladders assumes the ladders stand a lane pitch
    # outside the end lanes. Declared lanes that include the ladder lanes (10),
    # or too few (7), are read wrong: the reading is refused, or placed with
    # its lane numbers doubtful, and either way the answer asks for the drag.
    b = calibrated(tmp_path)
    s = b.session
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(lanes)])
    target = add(b, TARGET)
    drag = f"drag across all {lanes} lanes (only the left and right ends are used)"
    if lanes == 10:
        error = unchanged(s, lambda: ops.detect_mw_row(s, target))
        assert error.code is ErrorCode.ROW_LANES_UNCLEAR
        assert str(error).endswith(f"; the lanes were taken to lie between the two ladders: {drag}")
        assert (error.detail["span_from"], error.detail["hint"]) == (
            "ladders",
            "lane_span_required",
        )
        return
    placed = ops.detect_mw_row(s, target)
    assert "doubtful_lanes" in placed.flags
    assert placed.span_hint is not None and placed.span_hint.endswith(drag)
    # Dragged across the lanes, no hint.
    assert ops.detect_mw_row(s, target, span=placed.span).span_hint is None
