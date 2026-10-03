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
from proteia.core.grow import NOISE_K, REL_THRESHOLD, grow_box
from proteia.core.model import (
    BoxSize,
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
from proteia.core.quantify import estimate_background
from proteia.core.session import save_to_folder
from proteia.core.storage import load_project
from proteia.web.state import project_state
from rowcases import adversarial_row, blob, mw_slot_row
from test_checks import VENDOR_YS
from test_checks import Row as ModelRow
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


def _reprobes(b: Blot, lanes: Sequence[Sequence[int]]) -> list[tuple[str, set[str]]]:
    """A reprobe in the blot's register group for each of ``lanes``, in
    membrane order, a protein's box clicked on the target's band in each of
    its lanes; (the image, its boxes) each."""
    s = b.session
    out = []
    for k, boxed in enumerate(lanes):
        image = import_blot(
            s, np.array(blot_pixels()), f"reprobe {k} γ.tif", membrane_id=b.membrane
        )
        ops.set_marker_image(s, image, b.marker)
        protein = ops.add_protein(s, f"probe {k}", Role.TARGET, image)
        for lane in boxed:
            x, y = truth(TARGET[0], LANES[lane])
            ops.place_box(s, protein, round(x), round(y), lane_index=lane, grow=True)
        out.append((image, {band.id for band in protein_of(s, protein).bands}))
    return out


def _span_of(b: Blot, protein_id: str) -> mwrow.LaneSpan | None:
    batch = b.session.project.batch
    fitted = mwcal.calibration_for(batch.membrane_of(b.blot), b.blot)
    return mwrow.lane_span(batch, batch.find_protein(protein_id), fitted)


def test_span_from_the_group_image_with_the_most_lanes_placed(tmp_path):
    # One ladder, no lanes placed on the blot: three reprobes in its register
    # group with 3, 5 and 5 lanes placed. The span is read from the image
    # with the most, the first of them in membrane order (the second), from
    # its boxes; half a pitch (64 px) past lanes 1 and 8 at x 205 and 1101.
    b = calibrated(tmp_path, sides=(LEFT,))
    images = _reprobes(b, [(0, 1, 2), (1, 2, 3, 4, 5), (2, 3, 4, 5, 6)])
    span = _span_of(b, add(b))
    assert (span.source, span.image_id) == ("group_anchors", images[1][0])
    assert set(span.anchor_ids) == images[1][1] and len(span.anchor_ids) == 5
    assert abs(span.x0 - 141) <= 2 and abs(span.x1 - 1165) <= 2, span
    # Only two lanes placed on the one other image that has any: enough.
    c = calibrated(tmp_path / "two lanes", sides=(LEFT,))
    [(image, boxes)] = _reprobes(c, [(3, 4)])
    span = _span_of(c, add(c))
    assert (span.source, span.image_id, set(span.anchor_ids)) == ("group_anchors", image, boxes)


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


def _ladders(left_x: float, right_x: float, left, right) -> list[CalibrationPoint]:
    """Ladder bands ``(y, kDa)`` on ``img-1``, a left ladder at ``left_x`` and
    a right one at ``right_x``."""
    return [
        CalibrationPoint(image_id="img-1", y=y, mw=mw, source=MARKER_BAND, x=x, side=side)
        for x, side, bands in ((left_x, LEFT, left), (right_x, RIGHT, right))
        for y, mw in bands
    ]


# Two ladders that disagree strongly: 100 to 25 kDa over 20 px on the left
# (x 400), over 380 px on the right (x 900). Past the left one, the protein
# line folds over (at x 141.5 already).
FOLD_LEFT = ((100.0, 100.0), (110.0, 50.0), (120.0, 25.0))
FOLD_RIGHT = ((100.0, 100.0), (290.0, 50.0), (480.0, 25.0))


def test_the_slot_s_refusals_and_its_one_column_span():
    fitted = fitted_from(_ladders(400.0, 900.0, FOLD_LEFT, FOLD_RIGHT))
    assert fitted.curve_at(141.5) is None and fitted.curve_at(399.5) is not None
    with pytest.raises(mwrow.SlotError) as refused:
        mwrow.slot(fitted, [60.0], 0.1, 100, 800, 500)
    assert (refused.value.code, str(refused.value)) == (
        "outside_range",
        "the protein line folds over within x=100..799: the two ladders disagree too strongly"
        " there to place a row by its MW",
    )
    # Just past the range's top (126 kDa), though the rows searched reach
    # into it: refused all the same (the caller checks the range first).
    assert 10.0**fitted.z_hi < 130.0 < 10.0 ** (fitted.z_hi + math.log10(1.2))
    with pytest.raises(mwrow.SlotError) as refused:
        mwrow.slot(fitted, [130.0], 0.1, 400, 900, 500)
    assert (refused.value.code, str(refused.value)) == (
        "outside_range",
        "130 kDa lies outside the calibrated range",
    )
    # A span one column wide: that column's shift, 0, and no slope.
    one = mwrow.slot(fitted, [60.0], 0.1, 600, 601, 500)
    assert (one.shifts, one.slope_deg) == ((0,), 0.0)
    # Slots and spans are values.
    with pytest.raises(dataclasses.FrozenInstanceError):
        one.x0 = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        mwrow.LaneSpan(0, 10, "given").x0 = 1  # type: ignore[misc]


def test_spans_reach_the_image_s_first_column():
    # Ladders at x 0 and 10 around 8 lanes: the pitch 10 / 9, half of it
    # left of the first lane: column 0. Anchors 20 px apart at x 10 and 30:
    # half a pitch left of the first, column 0 too; no lanes, no span.
    fitted = fitted_from(
        _ladders(0.0, 10.0, ((100.0, 100.0), (140.0, 50.0)), ((100.0, 100.0), (140.0, 50.0)))
    )
    assert mwrow.span_between_ladders(fitted, 8, 1330) == (0, 10)
    assert mwrow.span_of_anchors([(10.0, 0), (30.0, 1)], 2, 1000) == (0, 40)
    assert mwrow.span_of_anchors([(10.0, 0), (30.0, 1)], 0, 1000) is None


def test_range_ends_in_words_far_from_any_protein():
    # Past the largest float, infinity; a million kDa or more, or under 0.1,
    # in 3 significant digits; an end that 3 digits round onto the refused
    # MW, in as many more as show the MW past it (4 for one far from any
    # protein); an end equal to the MW written, as many digits as can be.
    assert mwrow._mw_at(400.0) == math.inf
    assert mwrow._kda_at(400.0) == "∞"
    assert mwrow._kda_at(math.log10(2e6)) == "2e+06"
    assert mwrow._kda_at(math.log10(0.05)) == "0.05"
    assert mwrow._end_words(math.log10(1.2351e6), "1.2352e+06", above=True) == "1.235e+06"
    assert mwrow._end_words(2.0, "100", above=True) == "100"


def test_range_ends_in_words_at_each_format_s_limit():
    # A million kDa exactly: in significant digits; 12.3 kDa: whole kDa;
    # 0.12 kDa: one decimal.
    assert mwrow._kda_at(6.0) == "1e+06"
    assert mwrow._kda_at(math.log10(12.3)) == "12"
    assert mwrow._kda_at(math.log10(0.12)) == "0.1"
    # The fewest digits that show the MW past the end, from as many as the
    # end's whole kDa take plus one (4 for 124.6 and 125.94: never fewer, as
    # 1.2e+02 would show 124.8 above 124.6 by rounding the end down), or 2
    # under 1 kDa (0.55 for 0.6, which reads 0.56 inside), from 2 at 0.1 to
    # 0.15 kDa too (0.123: 0.12 reads 0.12 on it).
    assert mwrow._end_words(math.log10(124.6), "124.8", above=True) == "124.6"
    assert mwrow._end_words(math.log10(125.94), "126", above=True) == "125.9"
    assert mwrow._end_words(math.log10(0.55), "0.56", above=True) == "0.55"
    assert mwrow._end_words(math.log10(0.1234), "0.12", above=False) == "0.123"


def test_a_slot_s_end_columns_and_slope_read_its_own():
    # A slot 4 columns wide (x 10 to 13) whose shift differs at every column:
    # its ends are the first and the last column's, and a column past either
    # end reads that end's. A slope of exactly 3 degrees is not steep; past
    # it, either way, it is.
    found = mwrow.Slot(
        mws=(92.0,),
        x0=10,
        x1=14,
        centre=12.0,
        m_top=110.4,
        m_bot=76.7,
        margin=12.0,
        y0=100,
        y1=140,
        expected_y=(120.0,),
        shifts=(-3, -1, 2, 5),
        slope_deg=3.0,
    )
    assert found.shift_ends == (-3, 5)
    columns = (-50.0, 9.9, 10.0, 10.99, 11.0, 12.5, 13.0, 13.99, 14.0, 99.0)
    assert [found.shift_at(x) for x in columns] == [-3, -3, -3, -3, -1, 2, 5, 5, 5, 5]
    assert not found.steep and not dataclasses.replace(found, slope_deg=-3.0).steep
    assert dataclasses.replace(found, slope_deg=3.0001).steep
    assert dataclasses.replace(found, slope_deg=-3.0001).steep


def test_a_slot_is_cut_to_the_image_at_its_top_and_bottom():
    # One ladder at rows 10 and 110 for 100 and 10 kDa (100 px a decade, the
    # range reaching 125.9 kDa at row 0): a 100-kDa target's rows searched,
    # 6 px past 120 kDa at row 2.1, would start above the image: at row 0.
    near_top = fitted_from(marked((10.0, 100.0), (110.0, 10.0)))
    assert mwrow.slot(near_top, [100.0], 0.1, 141, 1165, HEIGHT).y0 == 0
    # Two ladders turned 2 degrees: the slot lies 18 rows lower at the
    # span's right end than at its centre, so on an image 10 rows taller than
    # the rows searched (to row 212) it stops 18 rows short of the bottom.
    tilted = fitted_from([*model_ladder(X_LEFT, 2.0), *model_ladder(X_RIGHT, 2.0, side=RIGHT)])
    full = mwrow.slot(tilted, [92.0], 0.1, 141, 1165, HEIGHT)
    assert (full.y1, max(full.shifts)) == (212, 18)
    assert mwrow.slot(tilted, [92.0], 0.1, 141, 1165, 222).y1 == 204
    # Level, an image ending at the slot's first row leaves no row; one row
    # more leaves one.
    level = fitted_from(model_ladder(X_LEFT))
    y0 = mwrow.slot(level, [92.0], 0.1, 141, 1165, HEIGHT).y0
    with pytest.raises(mwrow.SlotError) as refused:
        mwrow.slot(level, [92.0], 0.1, 141, 1165, y0)
    assert (refused.value.code, str(refused.value)) == (
        "out_of_image",
        "the rows where 92 kDa is searched lie off the image across x=141..1164",
    )
    one = mwrow.slot(level, [92.0], 0.1, 141, 1165, y0 + 1)
    assert (one.y0, one.y1) == (y0, y0 + 1)


def test_spans_of_one_lane_or_of_no_column():
    # One lane between two ladders 1152 px apart: each a lane pitch (576 px)
    # outside it, the span the middle half, x 365 to 941.
    two = fitted_from([*model_ladder(X_LEFT), *model_ladder(X_RIGHT, side=RIGHT)])
    assert mwrow.span_between_ladders(two, 1, WIDTH) == (365, 941)
    # Cut by the image's width to no column (x 141 to 141): no span; to one
    # column, that column.
    assert mwrow.span_between_ladders(two, 8, 141) is None
    assert mwrow.span_between_ladders(two, 8, 142) == (141, 142)
    anchors = [(333.0, 1), (461.0, 2), (845.0, 5)]
    assert mwrow.span_of_anchors(anchors, 8, 141) is None
    assert mwrow.span_of_anchors(anchors, 8, 142) == (141, 142)


def test_a_one_lane_batch_reads_its_lane_between_the_ladders(tmp_path):
    b = calibrated(tmp_path)
    ops.set_lanes(b.session, [LaneInput(samples.CONDITIONS[0], samples.SAMPLES[0])])
    target = add(b)
    batch = b.session.project.batch
    fitted = mwcal.calibration_for(batch.membrane_of(b.blot), b.blot)
    span = mwrow.lane_span(batch, batch.find_protein(target), fitted)
    assert span == mwrow.LaneSpan(365, 941, "ladders")


@pytest.mark.parametrize("end", ["top", "bottom"])
def test_an_expected_mw_on_a_range_end_is_searched(tmp_path, end):
    # The calibrated range's ends, 314.7 and 7.94 kDa (a tenth of a decade
    # past the ladder's 250 and 10): an MW whose log10 is exactly an end lies
    # in the range. Its slot is predicted, and placing it searches the rows
    # there (no band in them) rather than refusing the MW.
    b = calibrated(tmp_path)
    s = b.session
    fitted = mwcal.calibration_for(s.project.batch.membrane_of(b.blot), b.blot)
    z = fitted.z_hi if end == "top" else fitted.z_lo
    mw = 10.0**z
    assert math.log10(mw) == z  # not vacuous: exactly on the end
    target = ops.add_protein(s, "end", Role.TARGET, b.blot, expected_mw=mw)
    predicted = mwrow.predict(s.project.batch, s.project.batch.find_protein(target))
    assert predicted.slot is not None and predicted.slot.expected_y[0] is not None
    searched = "from 315 to 262 kDa" if end == "top" else "from 9.53 to 7.94 kDa"
    assert predicted.slot.y0 == (0 if end == "top" else 455)
    error = unchanged(s, lambda: ops.detect_mw_row(s, target))
    assert (error.code, str(error)) == (
        ErrorCode.NO_BAND_FOUND,
        f"no band reaches the detection limit around the expected MW ({searched}) in any lane."
        " Check the expected MW and the ladder marks, or drag a row box over the protein's band",
    )


def test_a_span_of_no_column_is_refused(tmp_path):
    # A span given as no column, or cut by the image to none: refused,
    # nothing changed.
    b = calibrated(tmp_path)
    s = b.session
    target = add(b)
    error = unchanged(s, lambda: ops.detect_mw_row(s, target, span=(400, 400)))
    assert (error.code, str(error)) == (
        ErrorCode.INVALID_INPUT,
        "span (400, 400) is empty or inverted",
    )
    error = unchanged(s, lambda: ops.detect_mw_row(s, target, span=(WIDTH, WIDTH + 40)))
    assert (error.code, str(error)) == (
        ErrorCode.OUT_OF_IMAGE,
        f"span ({WIDTH}, {WIDTH + 40}) lies outside the {WIDTH}x{HEIGHT} image {b.blot}",
    )


def test_a_prediction_off_its_slot_reads_the_span_s_centre():
    # A target expected at 60 kDa between the folding ladders, another
    # protein's boxes in the 8 lanes (x 205 to 1101): the span they give
    # reaches past the left ladder, where the line folds, so no slot is
    # placed; the row is predicted at the span's centre all the same.
    project = model_blot(
        _ladders(400.0, 900.0, FOLD_LEFT, FOLD_RIGHT),
        ModelRow("anchors", 60.0, source="click"),
        ModelRow("target", 60.0, expected=60.0, lanes=()),
    )
    target = next(p for p in project.batch.proteins if p.name == "target")
    predicted = mwrow.predict(project.batch, target)
    assert predicted.slot is None and predicted.span.source == "anchors"
    fitted = mwcal.calibration_for(project.batch.membranes[0], "img-2")
    centre = 0.5 * (predicted.span.x0 + predicted.span.x1)
    assert predicted.expected_y == (fitted.y_at(60.0, centre),)
    assert predicted.expected_y[0] is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        predicted.slot = None  # type: ignore[misc]


def test_a_prediction_with_no_lanes_reads_between_the_ladders():
    # No lanes declared: no span. With two ladders the row is predicted
    # midway between them (x 650: 60 kDa at y 213.7 here), not at either.
    project = model_blot(
        _ladders(400.0, 900.0, FOLD_LEFT, FOLD_RIGHT),
        ModelRow("target", 60.0, expected=60.0, lanes=()),
    )
    batch = project.batch.model_copy(update={"lanes": []})
    predicted = mwrow.predict(batch, batch.proteins[0])
    fitted = mwcal.calibration_for(batch.membranes[0], "img-2")
    assert (predicted.span, predicted.slot) == (None, None)
    assert predicted.expected_y == (fitted.y_at(60.0, 650.0),)
    assert predicted.expected_y[0] != fitted.y_at(60.0, 400.0)


def test_a_row_by_mw_where_the_line_folds_is_refused(tmp_path):
    # The folding ladders marked on the marker, the span dragged from x 0:
    # the protein line folds over within it, and the row is refused for
    # the calibration, changing nothing.
    b = calibrated(tmp_path, sides=())
    s = b.session
    for point in _ladders(400.0, 900.0, FOLD_LEFT, FOLD_RIGHT):
        ops.add_calibration_point(
            s, b.marker, point.y, point.mw, MARKER_BAND, x=point.x, side=point.side, snap=False
        )
    target = ops.add_protein(s, "PSD-95", Role.TARGET, b.blot, expected_mw=60)
    error = unchanged(s, lambda: ops.detect_mw_row(s, target, span=(0, 1330)))
    assert (error.code, error.ids) == (ErrorCode.MW_OUTSIDE_CALIBRATION, (b.blot,))
    assert str(error).startswith("the protein line folds over within x=0..1329")


def test_a_span_dragged_past_the_image_starts_at_its_first_column(tmp_path):
    b = calibrated(tmp_path)
    target = add(b, TARGET)
    placed = ops.detect_mw_row(b.session, target, span=(-5, 1165))
    assert placed.span == (0, 1165) and placed.row[0] == 0


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
        assert str(error).startswith(
            "no row placed: the bands nearest the expected MW in lanes 2, 3, 6, 7 lie on two"
            " rows, a lane's on one and its neighbour's on the other: the expected MW's row"
            " lies between two bands."
        )
    assert "draw it" not in str(error) and "the row box" not in str(error)


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
    # the row box's edge flagged. Growth stops at the valley to any other
    # band: each box on the target alone, the other band a second component.
    # Where the valley
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
    # change, no entry. With padding: the fitted size plus the padding.
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


def _lane(
    i: int,
    *,
    reason: str = "band",
    line_offset: float = 0.0,
    expected_offset: float = 0.1,
    expected_x: float | None = None,
    cut: bool = False,
    line_reason: str | None = None,
    line_snr: float | None = None,
) -> rowdetect.LaneDetection:
    """A lane as the detector reports it: a band boxed at lane i's place, its
    box's offsets from the row's line and the expected row; or empty."""
    band = reason == "band"
    x = 100.0 + 70.0 * i if expected_x is None else expected_x
    rect = (int(x) - 20, 80, int(x) + 20, 92) if band else None
    return rowdetect.LaneDetection(
        lane=i,
        rect=rect,
        reason=reason,
        snr=60.0 if band else 1.5,
        extent=rect,
        expected_x=x,
        bg_offset=0.0 if band else None,
        components=1 if band else 0,
        hollow=False,
        window=None,
        cut=cut,
        line_offset=line_offset if band else None,
        expected_offset=expected_offset if band else None,
        line_reason=line_reason,
        line_snr=line_snr,
    )


def _found(lanes, flags=(), **kwargs) -> rowdetect.RowDetection:
    """A detection of the lanes ``lanes`` with ``flags``."""
    boxed = any(lane.rect is not None for lane in lanes)
    return rowdetect.RowDetection(
        lanes=tuple(lanes),
        size=BoxSize(width=40, height=12) if boxed else None,
        pitch=70.0,
        noise=100.0,
        pixel_noise=400.0,
        flags=tuple(flags),
        notes=(),
        cost=1.0,
        margin=10.0,
        membrane_shift=kwargs.pop("membrane_shift", 0.0),
        **kwargs,
    )


def _off(offsets: dict[int, tuple[float, float]], n: int = 6, extra: dict | None = None) -> list:
    """n boxed lanes, those of ``offsets`` off the row's line and the expected
    row by ``(line, expected)`` box heights, the others on both; ``extra``
    sets more of a lane's fields."""
    extra = extra or {}
    return [
        _lane(i, line_offset=offsets[i][0], expected_offset=offsets[i][1], **extra.get(i, {}))
        if i in offsets
        else _lane(i, **extra.get(i, {}))
        for i in range(n)
    ]


LADDERS = ops._MwSearch(6, 110.4, 92.0 / 1.2, "ladders", "between the two ladders")
DRAGGED = ops._MwSearch(6, 110.4, 92.0 / 1.2, "given", None)
OFF = (
    "the band found lies off the row's line through the other lanes' bands ({}"
    " box heights, limit 0.75): {} another band, or the other edge of one band (a band"
    " with a light line along its middle reads as two)"
)
CLICK = (
    ". In {} the band found lies off the expected MW's row: box the protein's bands by"
    " clicking them in the lanes that hold it; a row box may box that other band there"
)
ROW_BOX = (
    ". Drag a row box over the protein's band, its whole height, across all 6 lanes; if a"
    " lane holds no protein, box the protein's bands by clicking them instead, since a row"
    " box may box another band in that lane"
)
SPAN = (
    "; the lanes were taken to lie between the two ladders: drag across all 6 lanes (only"
    " the left and right ends are used)"
)
AGAIN = (
    ". Drag across all 6 lanes again, from the outer edge of the first lane's band to the"
    " outer edge of the last's"
)
SLOT_KDA = "(from 110 to 76.7 kDa)"
_WORDS = [
    (  # half or more of the boxes off the line
        "half",
        _found(
            _off({0: (1.1, 1.1), 1: (-1.2, -1.2), 2: (1.3, 1.3)}),
            ["off_row_line"],
            off_cause="half",
        ),
        LADDERS,
        "no row placed: in lanes 1, 2, 3 "
        + OFF.format("1.10, 1.20 and 1.30", "those lanes picked")
        + "; 3 of 6 boxes are off the line, half or more, so none is recorded as not detected"
        + CLICK.format("lanes 1, 2, 3"),
    ),
    (  # an off lane's box on the expected row; boxes elsewhere off it
        "expected_row",
        _found(
            _off({1: (0.1, 1.0), 2: (0.1, -1.0), 5: (1.1, 0.2)}),
            ["off_row_line"],
            off_cause="expected_row",
        ),
        LADDERS,
        "no row placed: in lane 6 "
        + OFF.format("1.10", "that lane picked")
        + "; lane 6's box lies on the expected MW's row (0.20 box heights from it, limit"
        " 0.75), so the line may run through another row" + CLICK.format("lanes 2, 3"),
    ),
    (  # two of them; a lane cut by the rows' edge
        "expected_row_cut",
        _found(
            _off({1: (0.9, 0.5), 4: (-1.0, -0.75)}, n=7, extra={0: {"cut": True}}),
            ["off_row_line", "cut_by_row_box"],
            off_cause="expected_row",
        ),
        ops._MwSearch(7, 110.4, 92.0 / 1.2, "ladders", "between the two ladders"),
        "no row placed: bands cross the top or bottom edge of the rows searched around the"
        " expected MW, and in lanes 2, 5 "
        + OFF.format("0.90 and 1.00", "those lanes picked")
        + "; the boxes of lanes 2, 5 lie on the expected MW's row (0.50 and 0.75 box heights"
        " from it, limit 0.75), so the line may run through another row"
        + ROW_BOX.replace("6 lanes", "7 lanes"),
    ),
    (  # something on the line in the off lanes' slots
        "line_signal",
        _found(
            _off(
                {1: (1.0, 1.0), 3: (1.1, 1.2), 4: (0.8, 0.9)},
                n=8,
                extra={
                    1: {"line_reason": "unassigned", "line_snr": 61.3},
                    3: {"line_reason": "line", "line_snr": 2.0},
                    4: {"line_reason": "outside", "line_snr": 0.0},
                },
            ),
            ["off_row_line"],
            off_cause="line_signal",
        ),
        ops._MwSearch(8, 110.4, 92.0 / 1.2, "given", None),
        "no row placed: in lanes 2, 4, 5 "
        + OFF.format("1.00, 1.10 and 0.80", "those lanes picked")
        + "; another band reaches the detection limit on the row's line in lane 2 (SNR 61.3,"
        " limit 6), a line or strip across the lanes reaches the detection limit on the"
        " row's line in lane 4 and the row's line runs outside the rows searched in lane 5,"
        " so they are not recorded as not detected" + CLICK.format("lanes 2, 4, 5"),
    ),
    (  # one lane, the other readings
        "line_signal_one",
        _found(
            _off(
                {0: (1.0, 1.0), 2: (1.0, 1.0), 3: (1.0, 1.0)},
                n=7,
                extra={
                    0: {"line_reason": "side_signal", "line_snr": 1.0},
                    2: {"line_reason": "no_band", "line_snr": 1.0},
                    3: {"line_reason": "edge_signal", "line_snr": 1.0},
                },
            ),
            ["off_row_line"],
            off_cause="line_signal",
        ),
        LADDERS,
        "no row placed: in lanes 1, 3, 4 "
        + OFF.format("1.00, 1.00 and 1.00", "those lanes picked")
        + "; signal rising into the left or right end of the lanes' span reaches the"
        " detection limit on the row's line in lane 1 and signal at the top or bottom edge of"
        " the rows searched reaches the detection limit on the row's line in lane 4, so they"
        " are not recorded as not detected" + CLICK.format("lanes 1, 3, 4"),
    ),
    (
        "line_signal_artefact",
        _found(
            _off({2: (-1.4, -1.4)}, extra={2: {"line_reason": "artefact", "line_snr": 3.0}}),
            ["off_row_line"],
            off_cause="line_signal",
        ),
        LADDERS,
        "no row placed: in lane 3 "
        + OFF.format("1.40", "that lane picked")
        + "; a streak or stain lies on the row's line in lane 3, so it is not recorded as not"
        " detected" + CLICK.format("lane 3"),
    ),
    (  # another box off the expected row
        "other_box",
        _found(
            _off({3: (1.0, 1.0), 5: (0.2, 1.5)}),
            ["off_row_line"],
            off_cause="other_box",
        ),
        LADDERS,
        "no row placed: in lane 4 "
        + OFF.format("1.00", "that lane picked")
        + "; lane 6's box lies off the expected MW's row (1.50 box heights, limit 0.75), so the"
        " line may run through another row" + CLICK.format("lanes 4, 6"),
    ),
    (  # boxes exactly on the expected row (0, and the limit) are not named off it
        "other_box_limits",
        _found(
            _off({0: (0.0, 0.0), 1: (0.0, 0.75), 3: (1.0, 1.0), 5: (0.2, 1.5)}),
            ["off_row_line"],
            off_cause="other_box",
        ),
        LADDERS,
        "no row placed: in lane 4 "
        + OFF.format("1.00", "that lane picked")
        + "; lane 6's box lies off the expected MW's row (1.50 box heights, limit 0.75), so the"
        " line may run through another row" + CLICK.format("lanes 4, 6"),
    ),
    (  # two off the line: one on the expected row (0 from it), one off it
        "expected_row_mixed",
        _found(
            _off({1: (1.2, 1.3), 4: (1.0, 0.0)}),
            ["off_row_line"],
            off_cause="expected_row",
        ),
        LADDERS,
        "no row placed: in lanes 2, 5 "
        + OFF.format("1.20 and 1.00", "those lanes picked")
        + "; lane 5's box lies on the expected MW's row (0.00 box heights from it, limit"
        " 0.75), so the line may run through another row" + CLICK.format("lane 2"),
    ),
    (  # numbers a hair past or short of the limit, shown on their side of it
        "near_limits",
        _found(
            _off({2: (0.7504, 1.0), 4: (1.3, 0.7499)}),
            ["off_row_line"],
            off_cause="expected_row",
        ),
        LADDERS,
        "no row placed: in lanes 3, 5 "
        + OFF.format("0.7504 and 1.30", "those lanes picked")
        + "; lane 5's box lies on the expected MW's row (0.7499 box heights from it, limit"
        " 0.75), so the line may run through another row" + CLICK.format("lane 3"),
    ),
    (  # an SNR a hair past DETECT_K, on its side of it
        "line_signal_near_limit",
        _found(
            _off({1: (1.0, 1.0)}, extra={1: {"line_reason": "unassigned", "line_snr": 6.04}}),
            ["off_row_line"],
            off_cause="line_signal",
        ),
        LADDERS,
        "no row placed: in lane 2 "
        + OFF.format("1.00", "that lane picked")
        + "; another band reaches the detection limit on the row's line in lane 2 (SNR 6.04,"
        " limit 6), so it is not recorded as not detected" + CLICK.format("lane 2"),
    ),
    (  # placed again, still off the line
        "again",
        _found(
            _off({2: (1.2, 1.2), 5: (-1.1, -1.1)}),
            ["off_row_line"],
            off_cause="again",
            again=((4, 1.1), (0, -0.9)),
        ),
        LADDERS,
        "no row placed: in lanes 3, 6 "
        + OFF.format("1.20 and 1.10", "those lanes picked")
        + "; placed again without lanes 3, 6, the boxes of lanes 1, 5 lie off the line (0.90"
        " and 1.10 box heights, limit 0.75), so none is recorded as not detected"
        + CLICK.format("lanes 3, 6"),
    ),
    (  # neighbours grown from bands on two rows
        "crossed",
        _found(_off({}), ["off_row_line"], crossed=(0, 1, 4, 5)),
        LADDERS,
        "no row placed: the bands nearest the expected MW in lanes 1, 2, 5, 6 lie on two"
        " rows, a lane's on one and its neighbour's on the other: the expected MW's row lies"
        " between two bands" + ROW_BOX,
    ),
]


@pytest.mark.parametrize(
    ("found", "mw", "message"), [w[1:] for w in _WORDS], ids=[w[0] for w in _WORDS]
)
def test_a_row_off_its_line_placed_by_mw_says_why_and_what_to_do(found, mw, message):
    # A row placed by its expected MW (#58) and refused off its line says
    # which lanes lie off it and the numbers they were refused by, why none
    # was recorded as not detected (the first condition that failed), and
    # the next step: where boxes lie off the expected MW's row, click the
    # protein's bands (a row box would box that other band); else a row box
    # over the whole band, clicks where a lane holds no protein. Never a row
    # box the user did not draw.
    error = ops._row_refusal(found, 60, 600, mw)
    assert (error.code, error.detail["cause"]) == (ErrorCode.ROW_OFF_LINE, "off_row_line")
    assert str(error) == message
    assert error.detail["off_cause"] == found.off_cause
    assert [entry["expected_offset"] for entry in error.detail["lanes"]] == [
        lane.expected_offset for lane in found.lanes
    ]
    assert [(e["line_reason"], e["line_snr"]) for e in error.detail["lanes"]] == [
        (lane.line_reason, lane.line_snr) for lane in found.lanes
    ]
    assert "hint" not in error.detail and "span_from" not in error.detail
    assert "draw it" not in message and "the row box" not in message


def _empty(reason: str, n: int = 6) -> list:
    return [_lane(i, reason=reason) for i in range(n)]


_NO_ROW = [
    # (cause, found, mw, code, message, asks for the span)
    (
        "cut_by_row_box",
        _found(_off({}, extra={0: {"cut": True}}), ["ambiguous_lanes", "cut_by_row_box"]),
        LADDERS,
        ErrorCode.ROW_LANES_UNCLEAR,
        f"no row placed: bands cross the top or bottom edge of the rows searched around the"
        f" expected MW {SLOT_KDA}, so they do not show which lane each band is in. Drag a row box"
        " over the protein's band, its whole height, across all 6 lanes, or check the expected"
        " MW and its tolerance",
        False,
    ),
    (  # cut, and off the line with the lanes unsettled: the cut first
        "cut_by_row_box",
        _found(
            _off({2: (1.2, 1.2)}, extra={0: {"cut": True}}),
            ["ambiguous_lanes", "off_row_line", "cut_by_row_box"],
        ),
        LADDERS,
        ErrorCode.ROW_LANES_UNCLEAR,
        f"no row placed: bands cross the top or bottom edge of the rows searched around the"
        f" expected MW {SLOT_KDA}, so they do not show which lane each band is in. Drag a row box"
        " over the protein's band, its whole height, across all 6 lanes, or check the expected"
        " MW and its tolerance",
        False,
    ),
    (
        "side_signal",
        _found(
            [_lane(0, reason="side_signal", expected_x=50.0), *_off({}, n=6)[1:]],
            ["lanes_outside_row"],
        ),
        LADDERS,
        ErrorCode.ROW_LANES_UNCLEAR,
        "no row placed: the left or right end of the lanes' span cuts through a band, so lanes"
        " lie outside it" + SPAN,
        True,
    ),
    (
        "side_signal",
        _found(
            [_lane(0, reason="side_signal", expected_x=50.0), *_off({}, n=6)[1:]],
            ["lanes_outside_row"],
        ),
        DRAGGED,
        ErrorCode.ROW_LANES_UNCLEAR,
        "no row placed: the left or right end of the lanes' span cuts through a band, so lanes"
        " lie outside it" + AGAIN,
        False,
    ),
    (
        "ambiguous_lanes",
        _found(_off({}), ["ambiguous_lanes"]),
        LADDERS,
        ErrorCode.ROW_LANES_UNCLEAR,
        "no row placed: the bands found around the expected MW do not show which lane each"
        " band is in. Drag a row box over the protein's band across all 6 lanes, or box the"
        " bands by clicking them" + SPAN,
        True,
    ),
    (
        "ambiguous_lanes",
        _found(_off({}), ["ambiguous_lanes"]),
        DRAGGED,
        ErrorCode.ROW_LANES_UNCLEAR,
        "no row placed: the bands found around the expected MW do not show which lane each"
        " band is in" + AGAIN + ", or drag a row box over the protein's band across all 6"
        " lanes, or box the bands by clicking them",
        False,
    ),
    (  # off the line, the lanes unsettled: as unclear
        "off_row_line",
        _found(_off({2: (1.2, 1.2)}), ["ambiguous_lanes", "off_row_line"]),
        DRAGGED,
        ErrorCode.ROW_OFF_LINE,
        "no row placed: the bands found around the expected MW do not show which lane each"
        " band is in" + AGAIN + ", or drag a row box over the protein's band across all 6"
        " lanes, or box the bands by clicking them",
        False,
    ),
    (
        "unassigned",
        _found(_empty("unassigned")),
        LADDERS,
        ErrorCode.NO_BAND_FOUND,
        f"bands were found around the expected MW {SLOT_KDA} but do not fit the lanes. Drag across"
        " all 6 lanes, or box the bands by clicking them",
        False,
    ),
    (
        "edge_signal",
        _found(_empty("edge_signal")),
        LADDERS,
        ErrorCode.NO_BAND_FOUND,
        f"the only signal around the expected MW {SLOT_KDA} lies at the top or bottom edge of the"
        " rows searched: the protein's band may lie outside them. Check the expected MW, or"
        " drag a row box over the protein's band",
        False,
    ),
    (
        "side_signal",
        _found(_empty("side_signal")),
        LADDERS,
        ErrorCode.NO_BAND_FOUND,
        "the only signal around the expected MW rises into the left or right end of the lanes'"
        " span. Drag across all 6 lanes",
        False,
    ),
    (
        "line",
        _found(_empty("line")),
        LADDERS,
        ErrorCode.NO_BAND_FOUND,
        "the only signal around the expected MW runs across the lanes as a line or strip, not"
        " as bands. Drag a row box over the bands only, or box them by clicking",
        False,
    ),
    (
        "artefact",
        _found(_empty("artefact")),
        LADDERS,
        ErrorCode.NO_BAND_FOUND,
        "the only signal around the expected MW runs through the whole height searched: a"
        " streak or stain. Drag a row box over the protein's band, or box it by clicking",
        False,
    ),
    (
        "too_little_membrane",
        _found(_empty("no_band"), membrane_shift=rowdetect.MEMBRANE_SHIFT_K),
        LADDERS,
        ErrorCode.NO_BAND_FOUND,
        f"the rows searched around the expected MW {SLOT_KDA} hold too little membrane to measure"
        " the bands against. Drag a row box over the protein's band, with some membrane above"
        " and below it",
        False,
    ),
    (
        "no_band",
        _found(_empty("no_band")),
        DRAGGED,
        ErrorCode.NO_BAND_FOUND,
        f"no band reaches the detection limit around the expected MW {SLOT_KDA} in any lane. Check"
        " the expected MW and the ladder marks, or drag a row box over the protein's band",
        False,
    ),
]


@pytest.mark.parametrize(
    ("cause", "found", "mw", "code", "message", "asks"),
    _NO_ROW,
    ids=[f"{w[0]}-{w[2].span_from}" for w in _NO_ROW],
)
def test_every_other_mw_refusal_is_worded_by_what_it_searched(
    cause, found, mw, code, message, asks
):
    # Every refusal of a row placed by its expected MW, by the same causes
    # and codes as a row box's, worded by what it searched: the rows around
    # the expected MW (the slot's MWs, 3 significant digits), the lanes'
    # span; a span not dragged says where the lanes were taken to lie and
    # asks for the drag (detail: span_from, hint), a dragged one asks for it
    # again, band to band. No "draw it", no row box the user did not draw.
    error = ops._row_refusal(found, 60, 600, mw)
    assert (error.code, error.detail["cause"], str(error)) == (code, cause, message)
    assert ("hint" in error.detail, error.detail.get("span_from")) == (
        asks,
        mw.span_from if asks else None,
    )
    if asks:
        assert error.detail["hint"] == "lane_span_required"
    assert "draw it" not in message and "the row box" not in message
    # The same detection as a dragged row box: words of the row box, no MW.
    drawn = ops._row_refusal(found, 60, 600)
    assert (drawn.code, drawn.detail["cause"]) == (code, cause)
    assert "expected MW" not in str(drawn) and "off_cause" not in drawn.detail


@pytest.mark.parametrize(
    ("row", "side_x", "cause"),
    [
        # Lane 8's expected centre (x 350) lies past the box's right edge (x
        # 300), outside it: the edge cut a band and left the lane out.
        ((40, 20, 300, 480), 350.0, "side_signal"),
        # Lane 1's (x 0.5) lies inside the box from x 0: no lane left out.
        ((0, 300, 400, 480), 0.5, "lanes_outside_row"),
    ],
)
def test_a_drawn_row_s_side_edge_is_read_at_its_columns(tmp_path, monkeypatch, row, side_x, cause):
    # A row box refused with lanes outside it, signal rising into its side at
    # an end lane past the bands: whether that side edge cut a band and left
    # the lane out is read against the box's left and right columns, not its
    # rows.
    b = calibrated(tmp_path)
    s = b.session
    target = add(b)
    end = 7 if side_x > 100.0 else 0
    lanes = [
        _lane(i, reason="side_signal", expected_x=side_x)
        if i == end
        else _lane(i, expected_x=60.0 + 30.0 * i)
        for i in range(8)
    ]
    outside = _found(lanes, ["lanes_outside_row"])
    monkeypatch.setattr(rowdetect, "detect_row", lambda *args, **kwargs: outside)
    error = unchanged(s, lambda: ops.detect_row_boxes(s, target, row))
    assert (error.code, error.detail["cause"]) == (ErrorCode.ROW_LANES_UNCLEAR, cause)


def test_mw_refusal_words_name_lanes_off_the_line_past_its_limit_only():
    # A box exactly 0.75 box heights off the row's line is on it: neither
    # named off it nor counted among the boxes off it.
    found = _found(
        _off({0: (1.1, 1.1), 1: (-1.2, -1.2), 2: (1.3, 1.3), 3: (0.75, 0.2)}),
        ["off_row_line"],
        off_cause="half",
    )
    message = str(ops._row_refusal(found, 60, 600, LADDERS))
    assert message.startswith("no row placed: in lanes 1, 2, 3 the band found")
    assert "; 3 of 6 boxes are off the line, half or more," in message
    # Nothing checked (a detection the detector does not give with an
    # expected row): no reason given.
    unchecked = _found(_off({2: (1.2, 1.2)}), ["off_row_line"])
    assert str(ops._row_refusal(unchecked, 60, 600, LADDERS)) == (
        "no row placed: in lane 3 "
        + OFF.format("1.20", "that lane picked")
        + CLICK.format("lane 3")
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        LADDERS.lanes = 7  # type: ignore[misc]


@pytest.mark.parametrize("span", [None, (141, 1165)])
def test_an_mw_row_off_the_lanes_placed_asks_for_the_span(tmp_path, monkeypatch, span):
    # Bands that do not line up with the lanes already placed: worded by
    # the bands found around the expected MW, and the span's step (where
    # the lanes were taken to lie, with the hint, or the drag again).
    b = calibrated(tmp_path)
    s = b.session
    target = add(b, TARGET)
    monkeypatch.setattr(ops, "_off_lanes", lambda centres, expected: [2])
    monkeypatch.setattr(ops, "lane_positions", lambda *args, **kwargs: {0: 1.0})
    error = unchanged(s, lambda: ops.detect_mw_row(s, target, span=span))
    assert (error.code, error.detail["cause"]) == (ErrorCode.ROW_LANES_UNCLEAR, "off_lanes")
    lead = (
        "no row placed: the bands found around the expected MW do not line up with the lanes"
        " already placed on this image"
    )
    if span is None:
        assert str(error) == (
            f"{lead}; the lanes were taken to lie between the two ladders: drag across all 8"
            " lanes (only the left and right ends are used)"
        )
        assert (error.detail["span_from"], error.detail["hint"]) == (
            "ladders",
            "lane_span_required",
        )
    else:
        assert str(error) == (
            f"{lead}. Drag across all 8 lanes again, from the outer edge of the first lane's band"
            " to the outer edge of the last's"
        )
        assert "hint" not in error.detail


def _refused_by_mw(case, expected):
    """The detection of a row placed by its expected MW on ``case`` (its row
    box the rows searched) and the refusal it words."""
    found = rowdetect.detect_row(
        case.image,
        case.row,
        case.n_lanes,
        background=estimate_background(case.image),
        saturated_at=0.0,
        prefer_y=expected,
    )
    mw = ops._MwSearch(case.n_lanes, 110.4, 92.0 / 1.2, "ladders", "between the two ladders")
    return found, ops._row_refusal(found, case.row[0], case.row[2], mw)


def _follow(case, message: str, holding: Sequence[int]) -> dict[int, tuple[int, ...]]:
    """The boxes a user places following a refusal's next step, knowing the
    lanes ``holding`` the protein: clicks on their bands where the step says
    to click, or where a lane holds no protein; else a row box drawn over the
    protein's whole band across the lanes (its bands' 20% extents, 6 px more
    above and below), lane by lane."""
    background = estimate_background(case.image)
    click = "by clicking them in the lanes that hold it" in message
    if click or len(holding) < case.n_lanes:
        return {
            i: grow_box(
                case.image,
                (round(case.lane_cx[i]), round(case.lane_cy[i])),
                background,
                rel_threshold=REL_THRESHOLD,
                noise_k=NOISE_K,
            )
            for i in holding
        }
    top = min(r[1] for r in case.reference.values()) - 6
    bottom = max(r[3] for r in case.reference.values()) + 6
    drawn = rowdetect.detect_row(
        case.image,
        (case.row[0], top, case.row[2], bottom),
        case.n_lanes,
        background=background,
        saturated_at=0.0,
    )
    assert not drawn.refused, drawn.flags
    return dict(enumerate(drawn.slots))


def _crossed_row():
    """A row smiling 8 px, another twice as deep 16 px above it, the
    expected row level 3.5 px above the end lanes' bands."""
    case = adversarial_row(
        "crossed", 1000, smile=8.0, neighbour_dy=-16.0, neighbour_rel=2.0, box_adjust=(0, -30, 0, 0)
    )
    return case, min(case.lane_cy) + 0.5 - 3.5, -16.0


def _speck_row():
    """Lane 3 knocked out, another row twice as deep 16 px below, and a speck
    in lane 3's margin at the expected row."""
    case = adversarial_row(
        "speck",
        1000,
        neighbour_dy=16.0,
        neighbour_rel=2.0,
        missing=(2,),
        artefacts=[blob(2, 4.0, 15000.0, dx=24.0)],
        box_adjust=(0, 0, 0, 30),
    )
    return case, float(np.mean(case.lane_cy)) + 0.5, 16.0


def _mw_row(dy: float, **kwargs):
    case, expected = mw_slot_row("refused", kwargs.pop("seed", 1000), neighbour_dy=dy, **kwargs)
    return case, expected, dy


def _expected_off(built, by: float):
    """A row built for a refusal, its expected row ``by`` px lower (the
    protein running that far off its expected MW)."""
    case, expected, dy = built
    return case, expected + by, dy


_REFUSED = {
    # every lane's band clipped and lighter along its middle: its two edges
    "slit": (
        lambda: (*mw_slot_row("slit", 1000, depths=[75000.0] * 6, slit=(0.5, 1.2)), None),
        None,
    ),
    "half": (
        lambda: _mw_row(26.0, neighbour_depths=[48000.0] * 6, missing=(1, 2, 3)),
        "half",
    ),
    "expected_row_18px": (
        lambda: _mw_row(18.0, seed=1001, neighbour_depths=[24000.0] * 6, missing=(1, 2)),
        "expected_row",
    ),
    "line_signal": (_speck_row, "line_signal"),
    "other_box": (
        lambda: _mw_row(
            -20.0, n=5, missing=(3, 4), neighbour_depths=[0.0, 0.0, 0.0, 24000.0, 24000.0]
        ),
        "other_box",
    ),
    "crossed": (_crossed_row, None),
}
# The same refusals with another band 12 px from the protein's, half as deep
# to twice as deep: a click on the protein's band grows over it.
_MERGED = {
    "expected_row_2x_below": (
        lambda: _mw_row(12.0, seed=1001, neighbour_depths=[48000.0] * 6, missing=(1, 2)),
        "expected_row",
    ),
    "expected_row_half_below": (
        lambda: _mw_row(12.0, seed=1001, neighbour_depths=[12000.0] * 6, missing=(1, 2)),
        "expected_row",
    ),
    "expected_row_1x_above": (
        lambda: _mw_row(-12.0, neighbour_depths=[24000.0] * 6, missing=(1, 2)),
        "expected_row",
    ),
    "line_signal_2x_above": (
        lambda: _mw_row(-12.0, seed=1001, neighbour_depths=[48000.0] * 6, missing=(0,)),
        "line_signal",
    ),
    "line_signal_shoulder": (
        lambda: _mw_row(
            12.0,
            depths=[8000.0, 12000.0, 18000.0, 26000.0, 36000.0, 48000.0],
            neighbour_depths=[24000.0] * 6,
        ),
        "line_signal",
    ),
    "crossed_knocked_out": (
        lambda: _expected_off(_mw_row(12.0, neighbour_depths=[24000.0] * 6, missing=(1,)), 6.0),
        None,
    ),
}


@pytest.mark.parametrize(
    "name",
    [
        *_REFUSED,
        *(
            pytest.param(
                name,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="known limit: a click on the protein's band grows over another"
                    " band 12 px away, half as deep to twice as deep (no valley under 30% of"
                    " the clicked pixel), so the step to click the protein's bands boxes"
                    " both, as a click does on a shoulder; from 18 px on it boxes the"
                    " protein's alone",
                ),
            )
            for name in _MERGED
        ),
    ],
)
def test_following_each_mw_refusal_leaves_no_silent_wrong_box(name):
    # Each refusal of a row placed by its expected MW names a next step; a
    # user who follows it, knowing which lanes hold the protein, boxes the
    # protein's band in each of them and no other band: clicking the bands
    # where boxes lay off the expected MW's row (a row box would take that
    # other band) or where a lane holds no protein, else a row box over the
    # protein's whole band. (A click is a seed click, grow_box; a row box is
    # detect_row on the box drawn.)
    build, off_cause = {**_REFUSED, **_MERGED}[name]
    built = build()
    case, expected = built[0], built[1]
    dy = built[2] if len(built) > 2 else None
    found, error = _refused_by_mw(case, expected)
    assert found.refused and found.off_cause == off_cause, (found.flags, found.off_cause)
    assert error.code is ErrorCode.ROW_OFF_LINE
    holding = sorted(case.reference)
    boxes = _follow(case, str(error), holding)
    assert sorted(i for i, rect in boxes.items() if rect is not None) == holding
    for i in holding:
        x0, y0, x1, y1 = boxes[i]
        cx, cy = case.lane_cx[i] + 0.5, case.lane_cy[i] + 0.5
        assert x0 <= cx <= x1 and y0 <= cy <= y1, (i, boxes[i])
        if dy is not None:  # not the other band
            assert not y0 <= cy + dy <= y1, (i, boxes[i])


def test_a_slit_band_row_is_refused_and_the_drawn_row_boxes_it():
    # Bands lighter along their middle (clipped, the slit half their
    # depth) read as two edges, picked in different lanes: the row by its
    # expected MW stays refused, its lanes on two rows. Nothing off the
    # expected MW's row: the message asks for a row box over the whole band,
    # which boxes every lane's band whole (the box over both edges).
    case, expected = mw_slot_row("slit", 1000, depths=[75000.0] * 6, slit=(0.5, 1.2))
    found, error = _refused_by_mw(case, expected)
    assert found.refused and found.crossed and found.off_cause is None
    assert str(error).endswith(
        ". Drag a row box over the protein's band, its whole height, across all 6 lanes; if a"
        " lane holds no protein, box the protein's bands by clicking them instead, since a row"
        " box may box another band in that lane"
    )
    boxes = _follow(case, str(error), range(6))
    for i, (x0, y0, x1, y1) in boxes.items():
        top, bottom = case.lane_cy[i] - 3.0, case.lane_cy[i] + 4.0  # both edges
        assert y0 <= top and bottom <= y1 and x0 <= case.lane_cx[i] <= x1, (i, boxes[i])


@pytest.mark.parametrize(
    ("rows", "cause", "words"),
    [
        (
            (LOADING,),
            "no_band",
            "no band reaches the detection limit around the expected MW (from 110 to 76.7 kDa)"
            " in any lane. Check the expected MW and the ladder marks, or drag a row box over"
            " the protein's band",
        ),
        (
            ((130.0, 30000.0, 16.0), (65.0, 30000.0, 16.0)),
            "edge_signal",
            "the only signal around the expected MW (from 110 to 76.7 kDa) lies at the top or"
            " bottom edge of the rows searched: the protein's band may lie outside them. Check"
            " the expected MW, or drag a row box over the protein's band",
        ),
    ],
)
def test_a_row_by_mw_with_no_band_names_the_mws_it_searched(tmp_path, rows, cause, words):
    # The rows searched around 92 kDa (92 x 1.2 = 110.4 down to 92 / 1.2 =
    # 76.7 kDa, before the margin) hold no band (only α-tubulin's, at 50
    # kDa), or only bands across their top and bottom edges (130 and 65
    # kDa): refused, changing nothing, worded by the MWs searched.
    b = calibrated(tmp_path, rows=rows)
    target = add(b, TARGET)
    error = unchanged(b.session, lambda: ops.detect_mw_row(b.session, target))
    assert (error.code, error.detail["cause"], str(error)) == (
        ErrorCode.NO_BAND_FOUND,
        cause,
        words,
    )


OTHER_BELOW = (80.0, 18000.0, 16.0)  # twice as deep, 18 px below the target


def knocked_out_pixels(lane: int) -> np.ndarray:
    """The blot of TARGET in every lane but ``lane`` (a knockout), and
    OTHER_BELOW, a non-specific band, in every lane."""
    darkening = np.zeros((HEIGHT, WIDTH))
    for i, x in enumerate(LANES):
        if i != lane:
            _darken(darkening, x, *TARGET, 0.0)
        _darken(darkening, x, *OTHER_BELOW, 0.0)
    return _image(darkening, 58)


def test_a_knocked_out_lane_is_recorded_not_detected_by_mw(tmp_path):
    # Lane 4 holds no β-catenin; every lane holds a band twice as deep 18 px
    # below it. Lane 4 grows from that band, off the row's line and the
    # expected MW's row, nothing on the line there: placed, lane 4 recorded
    # as not detected at the expected MW (an MW-guided record of the rows
    # read on the line), every other lane boxed on its band. A new MW
    # tolerance drops the record, as the rows it searched moved.
    b = calibrated(tmp_path, rows=(TARGET, OTHER_BELOW), pixels=knocked_out_pixels(3))
    s = b.session
    target = add(b, TARGET)
    placed = ops.detect_mw_row(s, target)
    assert (placed.undetected_lanes, placed.unmeasured_lanes) == ((3,), ())
    assert "off_expected_row" in placed.flags and placed.span_hint is None
    [note] = [note for note in placed.notes if "recorded as not detected" in note]
    assert note.startswith("lane 4: the band found lies off the row's line through the other")
    assert on_truth(s, target, TARGET[0]) == [0, 1, 2, 4, 5, 6, 7]
    assert sorted(lane_bands(s, target)) == [0, 1, 2, 4, 5, 6, 7]
    [record] = protein_of(s, target).undetected
    assert (record.lane_index, record.source) == (3, ProposalSource.MW_GUIDED)
    assert record.reason.value == "below_detection_limit" and record.snr < record.threshold
    region = record.region
    assert region.y1 - region.y0 == 4
    x, y = truth(TARGET[0], LANES[3])
    assert region.x0 < x < region.x1 and region.y0 <= y <= region.y1
    entry = s.project.log[-1]
    assert entry.params["lanes"][3]["reason"] == "off_expected_row"
    assert [w["lane_index"] for w in entry.params["undetected_written"]] == [3]
    assert_mw_current(s)
    ops.edit_protein(s, target, mw_tolerance=0.2)
    assert protein_of(s, target).undetected == []
    assert s.project.log[-1].params["dropped_undetected"] == entry.params["undetected_written"]
