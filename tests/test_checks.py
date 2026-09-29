# SPDX-License-Identifier: Apache-2.0
"""The molecular-weight and band-count checks and their notices (#58, D7, D10).

The results tests build their projects from the model, with no pixels: a
visible-light marker ``img-1`` carrying the ladders and a chemiluminescence
image ``img-2`` linked to it, on the sample blot's migration law (the lanes at
x = 205 + 128 i, a ladder lane one pitch outside each end lane), turned about
the blot's centre where a test tilts it. Each box's apparent MW is its image's
calibration at its centre, as the operations store it. The operations tests draw
synthetic 16-bit blots and run the row detector (the helpers of
:mod:`test_operations`). Names use µ, α and β.
"""

import dataclasses
import json
import math
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from conftest import make_project, make_project_with_undetected, synthetic_blot
from proteia.core import mwcal, results
from proteia.core import operations as ops
from proteia.core.model import (
    Band,
    Batch,
    Box,
    BoxSize,
    CalibrationPoint,
    CalibrationPointSource,
    ImageKind,
    ImageRef,
    LadderSide,
    Lane,
    Membrane,
    MwCalibration,
    Polarity,
    Project,
    ProposalSource,
    Protein,
    Role,
    apply_change,
    revalidate,
)
from proteia.core.operations import ErrorCode, LaneInput, OperationError, ProjectSession
from proteia.core.results import Level, NoticeCode, compute_results
from proteia.web.results_view import results_payload
from test_operations import (
    CAL_LANES,
    CAL_ROW_BOX,
    CAL_W,
    MARKER_BAND,
    RIGHT,
    band_of,
    calibrated,
    import_blot,
    lane_bands,
    noisy16,
    plant,
    protein_of,
    row_session,
    session_on,
)
from test_rowdetect import two_topped_partner

LEFT = LadderSide.LEFT
STRIP = CalibrationPointSource.STRIP_EDGE
KDA = (250, 130, 100, 70, 55, 35, 25, 15, 10)  # PageRuler Plus, Tris-glycine
VENDOR_YS = (336.5, 373.5, 405.5, 443.0, 485.5, 550.0, 603.5, 672.5, 739.5)
LANES = tuple(205.0 + 128.0 * i for i in range(8))
X_LEFT, X_RIGHT = 77.0, 1229.0
SIZE = BoxSize(width=40, height=16)
MEASURED = {
    "net": 1000.0,
    "background_level": 200.0,
    "background_mode": "symmetric",
    "background_spread": 0.0,
}
PRESET = "pageruler_plus/tris_glycine"
EVERY_LANE = ", ".join(str(i) for i in range(1, 9))


def law(mw: float) -> float:
    """The sample blot's migration law without its smile: y of a band of ``mw`` kDa."""
    return 45.0 + 300.0 * math.log10(250.0 / mw)


def rotate(x: float, y: float, degrees: float) -> tuple[float, float]:
    """(x, y) turned by ``degrees`` about the blot's centre (653, 250);
    positive turns the right side down."""
    theta = math.radians(degrees)
    dx, dy = x - 653.0, y - 250.0
    return (
        653.0 + dx * math.cos(theta) - dy * math.sin(theta),
        250.0 + dx * math.sin(theta) + dy * math.cos(theta),
    )


def ladder(
    x: float,
    degrees: float = 0.0,
    *,
    side: LadderSide = LEFT,
    at: Sequence[float] = KDA,
    labels: Sequence[float] | None = None,
) -> list[CalibrationPoint]:
    """A ladder lane at ``x`` on ``img-1``: its bands at the MWs ``at`` on the
    law, turned by ``degrees``, labelled ``labels`` (``at`` unless given)."""
    points = []
    for mw, label in zip(at, labels or at, strict=True):
        rx, ry = rotate(x, law(mw), degrees)
        points.append(
            CalibrationPoint(image_id="img-1", y=ry, mw=label, source=MARKER_BAND, x=rx, side=side)
        )
    return points


@dataclasses.dataclass(frozen=True)
class Row:
    """A protein on ``img-2`` boxed along the law at ``mw`` kDa, turned by
    ``degrees``, in ``lanes``. Its detector boxes count ``counts`` bands per lane
    (1 where not given); boxes placed another way count none."""

    name: str
    mw: float
    expected: float | None = None
    tolerance: float | None = None
    degrees: float = 0.0
    lanes: Sequence[int] = tuple(range(8))
    source: str = "row_box"
    counts: Mapping[int, int] = dataclasses.field(default_factory=dict)
    expected_count: int = 1


def _image(image_id: str, kind: ImageKind, width: int, height: int, **fields) -> ImageRef:
    return ImageRef(
        id=image_id,
        file=f"{image_id}.tif",
        original_name=f"blot α {image_id}.tif",
        kind=kind,
        sha256="0" * 64,
        width=width,
        height=height,
        bit_depth=16,
        polarity=Polarity.DARK_ON_LIGHT,
        background=200.0,
        **fields,
    )


def blot(
    points: Sequence[CalibrationPoint],
    *rows: Row,
    ladder_key: str | None = None,
    width: int = 1330,
    height: int = 500,
    excluded: Sequence[int] = (),
) -> Project:
    """A project of one membrane, mem-9: ``img-1`` holding ``points``, ``img-2`` linked
    to it holding the proteins ``rows``, eight lanes (``excluded`` ones
    include=no)."""
    membrane = Membrane(
        id="mem-9",
        images=[
            _image("img-1", ImageKind.VISIBLE_MARKER, width, height),
            _image("img-2", ImageKind.CHEMILUMINESCENCE, width, height, marker_image_id="img-1"),
        ],
        calibration=MwCalibration(
            ladder=ladder_key,
            ladder_kda=[] if ladder_key is None else list(KDA),
            points=list(points),
        ),
    )
    fitted = mwcal.calibration_for(membrane, "img-2")
    proteins = []
    number = 10
    for k, row in enumerate(rows):
        bands = []
        for lane in row.lanes:
            cx, cy = rotate(LANES[lane], law(row.mw), row.degrees)
            box = Box(x=round(cx - SIZE.width / 2), y=round(cy - SIZE.height / 2))
            counted = row.source in ("row_box", "mw_guided")
            bands.append(
                Band(
                    id=f"band-{number}",
                    lane_index=lane,
                    box=box,
                    source=row.source,
                    apparent_mw=ops._apparent_mw(fitted, box, SIZE),
                    bands_found=row.counts.get(lane, 1) if counted else None,
                    **MEASURED,
                )
            )
            number += 1
        extra = {} if row.tolerance is None else {"mw_tolerance": row.tolerance}
        proteins.append(
            Protein(
                id=f"prot-{3 + k}",
                name=row.name,
                role=Role.TARGET,
                image_id="img-2",
                expected_mw=row.expected,
                expected_band_count=row.expected_count,
                box_size=SIZE,
                bands=bands,
                **extra,
            )
        )
    lanes = [Lane(index=i, label=f"c{i}", included=i not in excluded) for i in range(8)]
    project = Project(
        next_id=number,
        batch=Batch(lanes=lanes, membranes=[membrane], proteins=proteins),
    )
    return revalidate(project)


def of(res: results.Results, code: NoticeCode) -> list[results.Notice]:
    return [notice for notice in res.notices if notice.code is code]


def codes(res: results.Results) -> set[NoticeCode]:
    return {notice.code for notice in res.notices}


MW_CODES = {
    NoticeCode.MW_DEVIATION,
    NoticeCode.BAND_COUNT,
    NoticeCode.MW_NOT_CHECKED,
    NoticeCode.MW_OUTSIDE_CALIBRATION,
    NoticeCode.CALIBRATION_TWO_POINTS,
    NoticeCode.CALIBRATION_POOR_FIT,
    NoticeCode.ROWS_TILTED,
    NoticeCode.LADDERS_DISAGREE,
    NoticeCode.LADDER_SIDE_IGNORED,
}


def no_json_constants(value: object) -> None:
    """``value`` holds no NaN or infinity: strict JSON takes it as it is."""
    json.dumps(value, allow_nan=False, default=str)


# --- Without a curve: the MW is not checked, never passed ---


def test_no_curve_check_not_run_not_passed():
    res = compute_results(blot([], Row("β-catenin", 92, expected=92)).batch)
    [column] = res.proteins
    assert column.expected_mws == [92.0] and column.mw_tolerance == 0.1
    assert column.apparent_mw == [None] * 8
    assert column.mw_check == ["not_run"] * 8
    assert column.mw_not_run == "no_points"
    assert column.calibration is None
    # The band count still runs: the detector counted in the row box.
    assert column.bands_found == [1] * 8 and column.count_check == ["passed"] * 8
    [notice] = of(res, NoticeCode.MW_NOT_CHECKED)
    assert notice == results.Notice(
        code=NoticeCode.MW_NOT_CHECKED,
        level=Level.INFO,
        message=(
            "'β-catenin' has no molecular-weight calibration on img-2: its bands are taken to"
            " be at 92 kDa and their MW is not checked; mark the ladder on its marker image,"
            " or link it to the marker image it was taken with"
        ),
        protein_ids=("prot-3",),
        image_ids=("img-2",),
    )
    assert codes(res) & MW_CODES == {NoticeCode.MW_NOT_CHECKED}


def test_other_group_has_points_reason():
    # mem-1's points are on img-2 and img-3; the reprobe img-4 is a register
    # group of its own (D4), with none.
    def expect_36(project: Project) -> None:
        project.batch.find_protein("prot-9").expected_mw = 36.0

    project, _ = apply_change(make_project(), expect_36)
    res = compute_results(project.batch)
    gapdh = next(column for column in res.proteins if column.protein_id == "prot-9")
    assert (gapdh.mw_not_run, gapdh.mw_check) == ("no_points", ["not_run", "not_run", None, None])
    [notice] = of(res, NoticeCode.MW_NOT_CHECKED)
    assert notice.image_ids == ("img-4",) and "on img-4:" in notice.message
    # One point on the group: no curve either.
    one = compute_results(blot(ladder(X_LEFT, at=(100,)), Row("β-catenin", 92, 92)).batch)
    assert one.proteins[0].mw_not_run == "one_point"
    [notice] = of(one, NoticeCode.MW_NOT_CHECKED)
    assert notice.message == (
        "'β-catenin' has one calibration point on img-2: its bands are taken to be at 92 kDa"
        " and their MW is not checked; mark at least 2"
    )


def test_one_point_on_each_side_is_named_so():
    # Two points, one per side: no side has the two a ladder needs.
    split = [*ladder(X_LEFT, at=(100,)), *ladder(X_RIGHT, side=RIGHT, at=(70,))]
    res = compute_results(blot(split, Row("β-catenin", 92, 92)).batch)
    assert res.proteins[0].mw_not_run == "one_point"
    [notice] = of(res, NoticeCode.MW_NOT_CHECKED)
    assert notice.message == (
        "'β-catenin' has one calibration point per ladder side on img-2: its bands are taken"
        " to be at 92 kDa and their MW is not checked; mark at least 2 on one side"
    )
    assert codes(res) & MW_CODES == {NoticeCode.MW_NOT_CHECKED}


def test_mw_not_checked_notice():
    # No expected MW, or no box in the set's lanes: nothing to say.
    res = compute_results(blot([], Row("β-catenin", 92)).batch)
    assert of(res, NoticeCode.MW_NOT_CHECKED) == []
    assert res.proteins[0].mw_not_run == "no_expected_mw"
    res = compute_results(blot([], Row("β-catenin", 92, 92, lanes=(7,)), excluded=(7,)).batch)
    assert of(res, NoticeCode.MW_NOT_CHECKED) == []  # its one box is excluded
    assert [n.code for n in of(res.all_lanes, NoticeCode.MW_NOT_CHECKED)] == [
        NoticeCode.MW_NOT_CHECKED
    ]


# --- The MW check and the band count, with a curve ---


def test_mw_deviation_fires():
    project = blot(
        ladder(X_LEFT), Row("β-catenin", 80, expected=92), ladder_key=PRESET, excluded=(7,)
    )
    res = compute_results(project.batch)
    [column] = res.proteins
    assert all(math.isclose(mw, 80.0, rel_tol=0.005) for mw in column.apparent_mw)
    assert column.mw_check == ["failed"] * 8  # the column covers every lane
    assert column.mw_not_run is None
    [notice] = of(res, NoticeCode.MW_DEVIATION)
    assert notice == results.Notice(
        code=NoticeCode.MW_DEVIATION,
        level=Level.WARNING,
        message=(
            "'β-catenin' runs at 80 kDa in lanes 1, 2, 3, 4, 5, 6, 7 (−13%), more than ±10%"
            " from its expected 92 kDa; its ladder, PageRuler Plus Prestained Protein Ladder,"
            " 10 to 250 kDa, is read with the values for Tris-glycine (Laemmli, incl. TGX)"
        ),
        protein_ids=("prot-3",),
        lane_indices=(0, 1, 2, 3, 4, 5, 6),  # lane 8 is excluded
        image_ids=("img-2",),
    )
    assert of(res.all_lanes, NoticeCode.MW_DEVIATION)[0].lane_indices == tuple(range(8))
    # A custom ladder is named as typed; no ladder, not at all.
    custom = blot(ladder(X_LEFT), Row("β-catenin", 80, expected=92), ladder_key="our µ ladder")
    [notice] = of(compute_results(custom.batch), NoticeCode.MW_DEVIATION)
    assert notice.message.endswith("from its expected 92 kDa; its ladder is 'our µ ladder'")
    bare = blot(ladder(X_LEFT), Row("β-catenin", 80, expected=92))
    [notice] = of(compute_results(bare.batch), NoticeCode.MW_DEVIATION)
    assert notice.message.endswith("from its expected 92 kDa")


def test_mw_deviation_never_reads_as_within_the_tolerance():
    # The check compares exact values. Boxes at 82.78 kDa lie 10.02% below 92
    # kDa: whole numbers (83 kDa, −10%) would read as within ±10%, so the MW
    # and the share get a decimal, rounded away from the expected MW.
    res = compute_results(blot(ladder(X_LEFT), Row("β-catenin", 82.5, expected=92)).batch)
    [column] = res.proteins
    assert all(math.isclose(mw, 82.7828, abs_tol=1e-4) for mw in column.apparent_mw)
    assert column.mw_check == ["failed"] * 8
    [notice] = of(res, NoticeCode.MW_DEVIATION)
    assert notice.message == (
        f"'β-catenin' runs at 82.7 kDa in lanes {EVERY_LANE} (−10.1%), more than ±10% from"
        " its expected 92 kDa"
    )
    # The same above the expected MW: 100.29 kDa is 9.01% past it.
    above = Row("β-catenin", 100.5, expected=92, tolerance=0.09)
    [notice] = of(compute_results(blot(ladder(X_LEFT), above).batch), NoticeCode.MW_DEVIATION)
    assert notice.message == (
        f"'β-catenin' runs at 100.3 kDa in lanes {EVERY_LANE} (+9.1%), more than ±9% from"
        " its expected 92 kDa"
    )
    # Away from the boundary, whole numbers; a tolerance is written as given.
    off = Row("β-catenin", 80, expected=92, tolerance=0.125)
    [notice] = of(compute_results(blot(ladder(X_LEFT), off).batch), NoticeCode.MW_DEVIATION)
    assert notice.message == (
        f"'β-catenin' runs at 80 kDa in lanes {EVERY_LANE} (−13%), more than ±12.5% from its"
        " expected 92 kDa"
    )


def test_tolerance_per_protein():
    rows = (
        Row("β-catenin", 80, expected=92, tolerance=0.15),  # −13%: within ±15%
        Row("α-catenin", 80, expected=92, lanes=(0, 1)),  # the default ±10%
    )
    res = compute_results(blot(ladder(X_LEFT), *rows).batch)
    beta, alpha = res.proteins
    assert (beta.mw_tolerance, beta.mw_check) == (0.15, ["passed"] * 8)
    assert alpha.mw_check == ["failed", "failed", None, None, None, None, None, None]
    assert [n.protein_ids for n in of(res, NoticeCode.MW_DEVIATION)] == [("prot-4",)]


def test_band_count_fires_on_extra_band():
    row = Row("β-catenin", 92, expected=92, counts={4: 2, 6: 3})
    res = compute_results(blot(ladder(X_LEFT), row).batch)
    [column] = res.proteins
    assert column.bands_found == [1, 1, 1, 1, 2, 1, 3, 1]
    assert column.count_check == ["passed"] * 4 + ["failed", "passed", "failed", "passed"]
    [notice] = of(res, NoticeCode.BAND_COUNT)
    assert notice == results.Notice(
        code=NoticeCode.BAND_COUNT,
        level=Level.WARNING,
        message=(
            "'β-catenin': 2 or 3 separate bands within ±10% of its box's MW in lanes 5, 7,"
            " where 1 is expected"
        ),
        protein_ids=("prot-3",),
        lane_indices=(4, 6),
        image_ids=("img-2",),
    )
    # Without a curve the count read the row box.
    [notice] = of(compute_results(blot([], row).batch), NoticeCode.BAND_COUNT)
    assert "2 or 3 separate bands in its row box in lanes 5, 7" in notice.message
    # A protein expected to show two bands fails only past two.
    pair = compute_results(blot(ladder(X_LEFT), dataclasses.replace(row, expected_count=2)).batch)
    assert pair.proteins[0].count_check[4:7] == ["passed", "passed", "failed"]
    assert "3 separate bands" in of(pair, NoticeCode.BAND_COUNT)[0].message
    assert "where 2 are expected" in of(pair, NoticeCode.BAND_COUNT)[0].message


def test_count_not_run_for_click_and_moved_box():
    rows = (Row("β-catenin", 92, expected=92, source="click"),)
    res = compute_results(blot(ladder(X_LEFT), *rows).batch)
    [column] = res.proteins
    assert column.bands_found == [None] * 8
    assert column.count_check == ["not_run"] * 8
    assert of(res, NoticeCode.BAND_COUNT) == []
    # The model keeps a count to a detector's box nobody edited.
    [band] = blot(ladder(X_LEFT), Row("β-catenin", 92, lanes=(0,))).batch.proteins[0].bands
    doc = band.model_dump()
    for edit in ({"manually_edited": True}, {"source": "click"}, {"source": "manual"}):
        with pytest.raises(ValueError, match="comes from a detector's box that nobody edited"):
            Band.model_validate({**doc, **edit})


def test_below_detection_is_not_count_failure():
    res = compute_results(make_project_with_undetected().batch)
    beta = res.proteins[0]
    assert beta.detected[2] is False  # a not-detected record in lane 3
    assert (beta.bands_found[2], beta.count_check[2], beta.mw_check[2]) == (None, None, None)
    assert NoticeCode.BELOW_DETECTION in codes(res)
    assert NoticeCode.BAND_COUNT not in codes(res)


def test_outside_calibration_notice():
    # One ladder marked from 100 to 55 kDa reaches a tenth of a decade past them.
    one = blot(ladder(X_LEFT, at=(100, 70, 55)), Row("β-catenin", 250, 250, lanes=(0, 1)))
    res = compute_results(one.batch)
    assert res.proteins[0].mw_check[:2] == ["not_run", "not_run"]
    assert res.proteins[0].mw_not_run is None
    [notice] = of(res, NoticeCode.MW_OUTSIDE_CALIBRATION)
    assert notice == results.Notice(
        code=NoticeCode.MW_OUTSIDE_CALIBRATION,
        level=Level.WARNING,
        message=(
            "'β-catenin' in lanes 1, 2 lie outside the calibrated range of img-2 (126–44 kDa):"
            " its MW is not checked there"
        ),
        protein_ids=("prot-3",),
        lane_indices=(0, 1),
        image_ids=("img-2",),
    )
    # Two ladders: where both reach, the right one marked from 130 to 15 kDa.
    points = [*ladder(X_LEFT), *ladder(X_RIGHT, side=RIGHT, at=KDA[1:-1])]
    two = blot(points, Row("β-catenin", 10, 10, lanes=(3,)))
    [notice] = of(compute_results(two.batch), NoticeCode.MW_OUTSIDE_CALIBRATION)
    assert notice.message == (
        "'β-catenin' in lane 4 lies outside the calibrated range of img-2 (164–12 kDa, where"
        " both ladders reach): its MW is not checked there"
    )


def test_poor_fit_notice():
    # The 55-kDa band was not marked, so every label below it moved up one band.
    ys = [y for y, m in zip(VENDOR_YS, KDA, strict=True) if m != 55]
    points = [
        CalibrationPoint(image_id="img-1", y=y, mw=m, source=MARKER_BAND, x=X_LEFT)
        for y, m in zip(ys, KDA[:-1], strict=True)
    ]
    res = compute_results(blot(points, Row("β-catenin", 92), height=800).batch)
    [notice] = of(res, NoticeCode.CALIBRATION_POOR_FIT)
    assert notice == results.Notice(
        code=NoticeCode.CALIBRATION_POOR_FIT,
        level=Level.WARNING,
        message=(
            "55 kDa on the ladder of img-1 sits where the bands above and below it put"
            " 44 kDa, so its label is 25% higher: check the label"
        ),
        protein_ids=("prot-3",),
        image_ids=("img-1",),
    )
    # Per ladder: the same left ladder beside a right one that fits.
    right = [
        CalibrationPoint(image_id="img-1", y=y, mw=m, source=MARKER_BAND, x=X_RIGHT, side=RIGHT)
        for y, m in zip(VENDOR_YS, KDA, strict=True)
    ]
    res = compute_results(blot([*points, *right], Row("β-catenin", 92), height=800).batch)
    [notice] = of(res, NoticeCode.CALIBRATION_POOR_FIT)
    assert notice.message.startswith("55 kDa on the left ladder of img-1 sits where")
    # A ladder that fits says nothing; so does one with no box on its images.
    fits = compute_results(blot(right, Row("β-catenin", 92), height=800).batch)
    assert of(fits, NoticeCode.CALIBRATION_POOR_FIT) == []
    alone = compute_results(blot(points, height=800).batch)
    assert codes(alone) & MW_CODES == set()


@pytest.mark.parametrize(
    ("at", "label", "share"),
    [
        (58.0, 70.0, "21% higher"),  # 70 / 58: +20.7%, past FIT_WARN
        (63.0, 50.0, "21% lower"),  # 50 / 63: −20.6%
        (70.0 / 1.2004, 70.0, "20.1% higher"),  # +20.04%: whole percent would read 20%
    ],
)
def test_poor_fit_states_the_share_it_compares(at, label, share):
    # The notice fires on |label / its neighbours' MW − 1| > FIT_WARN, and
    # states that share: never a figure within the threshold.
    points = ladder(X_LEFT, at=(100, at, 30), labels=(100, label, 30))
    [ladder_fit] = mwcal.calibration_for(blot(points).batch.membranes[0], "img-2").ladders
    assert ladder_fit.quality.value > mwcal.FIT_WARN
    [notice] = of(
        compute_results(blot(points, Row("β-catenin", 92)).batch), NoticeCode.CALIBRATION_POOR_FIT
    )
    assert notice.message == (
        f"{label:g} kDa on the ladder of img-1 sits where the bands above and below it put"
        f" {round(at)} kDa, so its label is {share}: check the label"
    )
    # A share within it says nothing.
    near = ladder(X_LEFT, at=(100, 60, 30), labels=(100, 50, 30))  # 50 / 60: −16.7%
    fits = compute_results(blot(near, Row("β-catenin", 92)).batch)
    assert of(fits, NoticeCode.CALIBRATION_POOR_FIT) == []


@pytest.mark.parametrize("degrees", [-1.0, 1.0])
def test_rows_tilted_against_protein_line(degrees):
    row = Row("β-catenin", 55, expected=55, tolerance=0.5, degrees=degrees)
    two = [*ladder(X_LEFT, degrees), *ladder(X_RIGHT, degrees, side=RIGHT)]
    res = compute_results(blot(two, row).batch)
    # Two ladders follow the turned blot: the rows lie along the protein line.
    assert of(res, NoticeCode.ROWS_TILTED) == []
    calibration = res.proteins[0].calibration
    assert calibration is not None and calibration.two_ladders
    assert math.isclose(calibration.tilt_deg, degrees, abs_tol=1e-9)
    # One ladder reads the rows against its level line: they slope across it.
    res = compute_results(blot(ladder(X_LEFT, degrees), row).batch)
    [notice] = of(res, NoticeCode.ROWS_TILTED)
    assert re.fullmatch(
        r"img-2's rows slope by about 1\.0° against the ladder"
        r" \(1[56] px across the lanes\), so apparent MWs drift by up to 1[23]% from one side"
        r" to the other; mark the ladder on the other side of the blot too, or widen the"
        r" tolerance",
        notice.message,
    ), notice.message
    assert (notice.level, notice.protein_ids, notice.image_ids) == (
        Level.WARNING,
        ("prot-3",),
        ("img-2",),
    )
    assert res.proteins[0].calibration == results.ImageCalibration(
        two_ladders=False, tilt_deg=None, disagreement=None, disagreement_mw=None
    )


def test_rows_tilted_reads_only_detector_boxes_nobody_edited():
    level = compute_results(blot(ladder(X_LEFT), Row("β-catenin", 55, expected=55)).batch)
    assert of(level, NoticeCode.ROWS_TILTED) == []  # a level row on a level ladder
    tilted = Row("β-catenin", 55, expected=55, tolerance=0.5, degrees=1.0)
    for row in (
        dataclasses.replace(tilted, source="click"),  # placed by hand
        dataclasses.replace(tilted, lanes=(0, 1, 2)),  # fewer than TILT_MIN_LANES
    ):
        res = compute_results(blot(ladder(X_LEFT, 1.0), row).batch)
        assert of(res, NoticeCode.ROWS_TILTED) == [], row
    # Without an expected MW, the line at the boxes' median apparent MW.
    unexpected = dataclasses.replace(tilted, expected=None)
    res = compute_results(blot(ladder(X_LEFT, 1.0), unexpected).batch)
    assert [n.code for n in of(res, NoticeCode.ROWS_TILTED)] == [NoticeCode.ROWS_TILTED]
    # Excluded lanes leave the set too few boxes.
    few = blot(ladder(X_LEFT, 1.0), tilted, excluded=(0, 1, 2, 3, 4))
    res = compute_results(few.batch)
    assert of(res, NoticeCode.ROWS_TILTED) == []
    assert len(of(res.all_lanes, NoticeCode.ROWS_TILTED)) == 1


def test_rows_tilted_with_an_expected_mw_past_the_calibrated_range():
    # One ladder marked from 100 to 55 kDa reaches 126 kDa; rows at 112 kDa,
    # turned 1°. An expected MW past the range has no line: the tilt is read
    # at the boxes' median apparent MW, as without one (with one ladder, any
    # MW in range gives the same slope).
    one = ladder(X_LEFT, 1.0, at=(100, 70, 55))
    row = Row("β-catenin", 112, tolerance=0.2, degrees=1.0)

    def tilted(points: list[CalibrationPoint], expected: float | None) -> list[str]:
        res = compute_results(blot(points, dataclasses.replace(row, expected=expected)).batch)
        return [notice.message for notice in of(res, NoticeCode.ROWS_TILTED)]

    [message] = tilted(one, None)
    assert tilted(one, 112) == tilted(one, 130) == tilted(one, 40) == [message]
    # At 130 kDa the MW check fails the lanes the slope carries farthest off.
    res = compute_results(blot(one, dataclasses.replace(row, expected=130)).batch)
    assert of(res, NoticeCode.MW_DEVIATION)[0].lane_indices == (4, 5, 6, 7)
    # Two level ladders marked the same, the rows turned across them.
    level = [*ladder(X_LEFT, at=(100, 70, 55)), *ladder(X_RIGHT, side=RIGHT, at=(100, 70, 55))]
    [message] = tilted(level, None)
    assert tilted(level, 130) == [message]


@pytest.mark.parametrize(
    ("right", "advice"),
    [
        (
            ladder(X_RIGHT, 1.0, side=RIGHT, at=(100,)),
            "mark at least 2 points on the right ladder so that it is used too",
        ),
        (
            ladder(X_RIGHT, 1.0, side=RIGHT, at=(70, 55, 35, 25, 15)),
            "mark the same bands on both ladders so that the right one is used too",
        ),
    ],
    ids=["one_point", "few_shared_mws"],
)
def test_rows_tilted_advice_names_a_side_marked_but_not_used(right, advice):
    # The right side is marked but not used: the advice is to make it count,
    # not to mark it.
    left = ladder(X_LEFT, 1.0, at=(250, 130, 100, 70))
    row = Row("β-catenin", 100, expected=100, tolerance=0.5, degrees=1.0)
    res = compute_results(blot([*left, *right], row).batch)
    assert len(of(res, NoticeCode.LADDER_SIDE_IGNORED)) == 1
    [notice] = of(res, NoticeCode.ROWS_TILTED)
    assert re.fullmatch(
        r"img-2's rows slope by about 1\.0° against the left ladder \(1[56] px across the"
        rf" lanes\), so apparent MWs drift by up to \d+% from one side to the other; {advice},"
        r" or widen the tolerance",
        notice.message,
    ), notice.message


def test_ladders_disagree_fires_one_band_off():
    left = ladder(X_LEFT, 0.5)
    row = Row("β-catenin", 55, expected=55, tolerance=0.5)
    right = ladder(X_RIGHT, 0.5, side=RIGHT)
    assert of(compute_results(blot([*left, *right], row).batch), NoticeCode.LADDERS_DISAGREE) == []
    # The right ladder's labels one band up: its 130-kDa band called 250, and so on.
    off = ladder(X_RIGHT, 0.5, side=RIGHT, at=KDA[1:], labels=KDA[:-1])
    res = compute_results(blot([*left, *off], row).batch)
    [notice] = of(res, NoticeCode.LADDERS_DISAGREE)
    assert notice == results.Notice(
        code=NoticeCode.LADDERS_DISAGREE,
        level=Level.WARNING,
        message=(
            "the left and right ladders of img-1 disagree near 250 kDa (48%): check both"
            " ladders' labels"
        ),
        protein_ids=("prot-3",),
        image_ids=("img-1",),
    )
    calibration = res.proteins[0].calibration
    assert calibration.two_ladders and calibration.disagreement_mw == 250
    assert calibration.disagreement > mwcal.LADDERS_WARN and not calibration.disagreement_infinite


@pytest.mark.parametrize(
    ("right", "message"),
    [
        (
            ladder(X_RIGHT, side=RIGHT, at=(100,)),
            "the right ladder of img-1 has 1 point and is not used; mark at least 2",
        ),
        (
            ladder(X_RIGHT, side=RIGHT, at=(55, 35, 25, 15, 10)),
            "the right ladder of img-1 shares fewer than 2 marked MWs with the left one and"
            " is not used; mark the same bands on both",
        ),
    ],
    ids=["one_point", "few_shared_mws"],
)
def test_ladder_side_ignored_names_the_reason(right, message):
    left = ladder(X_LEFT, at=(250, 130, 100))
    res = compute_results(blot([*left, *right], Row("β-catenin", 130, expected=130)).batch)
    [notice] = of(res, NoticeCode.LADDER_SIDE_IGNORED)
    assert notice == results.Notice(
        code=NoticeCode.LADDER_SIDE_IGNORED,
        level=Level.INFO,
        message=message,
        protein_ids=("prot-3",),
        image_ids=("img-1",),
    )
    assert not res.proteins[0].calibration.two_ladders
    assert res.proteins[0].mw_check == ["passed"] * 8  # the left ladder alone reads them


def test_strip_edge_only_curve_checks():
    # D7: a strip cut at 100 and 75 kDa checks like any curve, with a notice.
    edges = [
        CalibrationPoint(image_id="img-1", y=150.0, mw=100, source=STRIP, x=10.0),
        CalibrationPoint(image_id="img-1", y=250.0, mw=75, source=STRIP, x=900.0),
    ]
    res = compute_results(blot(edges, Row("β-catenin", 90, expected=92)).batch)
    [column] = res.proteins
    assert column.mw_check == ["passed"] * 8 and column.mw_not_run is None
    [notice] = of(res, NoticeCode.CALIBRATION_TWO_POINTS)
    assert notice == results.Notice(
        code=NoticeCode.CALIBRATION_TWO_POINTS,
        level=Level.INFO,
        message=(
            "the calibration of img-1 rests on two points (strip edges at 100 and 75 kDa):"
            " apparent MWs are less reliable"
        ),
        protein_ids=("prot-3",),
        image_ids=("img-1",),
    )
    # Two ladder bands, beside a right ladder of many.
    two = ladder(X_LEFT, at=(100, 55))
    right = ladder(X_RIGHT, side=RIGHT)
    [notice] = of(
        compute_results(blot([*two, *right], Row("β-catenin", 70)).batch),
        NoticeCode.CALIBRATION_TWO_POINTS,
    )
    assert notice.message == (
        "the left ladder of img-1 rests on two points (ladder bands at 100 and 55 kDa):"
        " apparent MWs are less reliable"
    )


def test_mw_results_are_strict_json_with_degenerate_ladders():
    # Ladders labelled hundreds of decades apart: their disagreement is past the
    # largest float. Nothing in the results is NaN or infinite.
    def point(y: float, mw: float, x: float, side: LadderSide) -> CalibrationPoint:
        return CalibrationPoint(image_id="img-1", y=y, mw=mw, source=MARKER_BAND, x=x, side=side)

    points = [
        point(10.0, 1e300, X_LEFT, LEFT),
        point(11.0, 1e-300, X_LEFT, LEFT),
        point(10.0, 1e300, X_RIGHT, RIGHT),
        point(20.0, 1e-300, X_RIGHT, RIGHT),
    ]
    res = compute_results(blot(points, Row("β-catenin", 55, expected=55)).batch)
    calibration = res.proteins[0].calibration
    assert calibration == results.ImageCalibration(
        two_ladders=True,
        tilt_deg=calibration.tilt_deg,
        disagreement=None,
        disagreement_mw=1e-300,  # 4.5 px past the median offset: 491 decades
        disagreement_infinite=True,
    )
    [notice] = of(res, NoticeCode.LADDERS_DISAGREE)
    assert "disagree near 1e-300 kDa (by more than any MW spans)" in notice.message
    no_json_constants(res.model_dump())
    payload = results_payload(res, open_id=1, revision=1)
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_mw_columns_on_the_sample_project():
    # mem-1's fit is from before #58: its stored MWs stand until a change refits
    # it. band-12 (lane 4, excluded) reads 90.5 kDa against β-catenin's 92.
    res = compute_results(make_project().batch)
    beta = res.proteins[0]
    assert beta.apparent_mw == [None, None, None, 90.5]
    assert beta.mw_check == ["not_run", "not_run", None, "passed"]
    assert beta.bands_found == [None] * 4  # counted by no detector yet
    assert beta.count_check == ["not_run", "not_run", None, "not_run"]
    # Its boxes lie inside the range: a missing MW there is no reason to warn.
    assert NoticeCode.MW_OUTSIDE_CALIBRATION not in codes(res)
    no_json_constants(res.model_dump())


def all_lanes_view(res: results.Results) -> list[results.Notice]:
    """The all-lanes set's notices as the page lists them (charts.js
    ``setNotices``): its own, then those of the applied set whose code and
    proteins none of its own has, but for the applied set's own kinds."""
    own_set = results.TEST_NOTICE_CODES | {NoticeCode.REFERENCE_ALL_EXCLUDED}
    assert res.all_lanes is not None
    keys = {(n.code, n.protein_ids) for n in res.all_lanes.notices}
    shared = [
        n for n in res.notices if n.code not in own_set and (n.code, n.protein_ids) not in keys
    ]
    return [*res.all_lanes.notices, *shared]


def test_calibration_notices_are_listed_once_in_the_all_lanes_view():
    # A ladder with its 55-kDa band unmarked, and three proteins on img-2,
    # turned 1°, lane 8 excluded: prot-3 boxed in every lane, prot-4 in lanes
    # 5 to 8 (the tilt reads it in the all-lanes set only), prot-5 in lane 8
    # only. The notices about the calibration, and the tilt, name every
    # protein boxed on their images, in any lane: the same in both sets, so
    # the all-lanes view shows each once.
    ys = [y for y, m in zip(VENDOR_YS, KDA, strict=True) if m != 55]
    points = [
        CalibrationPoint(image_id="img-1", y=y, mw=m, source=MARKER_BAND, x=X_LEFT)
        for y, m in zip(ys, KDA[:-1], strict=True)
    ]
    rows = (
        Row("β-catenin", 92, expected=92, tolerance=0.5, degrees=1.0),
        Row("α-tubulin", 50, expected=50, tolerance=0.5, degrees=1.0, lanes=(4, 5, 6, 7)),
        Row("µ-calpain", 35, expected=35, tolerance=0.5, degrees=1.0, lanes=(7,)),
    )
    res = compute_results(blot(points, *rows, height=800, excluded=(7,)).batch)
    shown = all_lanes_view(res)
    for code in (NoticeCode.CALIBRATION_POOR_FIT, NoticeCode.ROWS_TILTED):
        [notice] = of(res, code)
        assert notice.protein_ids == ("prot-3", "prot-4", "prot-5"), code
        assert [n for n in shown if n.code is code] == (of(res.all_lanes, code) or [notice])


# --- Operations: the count window, and counts that follow their boxes ---

# A blot calibrated at 400 px per decade (25 kDa at y = 20): LC3-I at 16 kDa and
# LC3-II at 14 kDa lie 23 px apart, farther than the ±10% count window (17 px).
LC3_I, LC3_II = 16.0, 14.0
LC3_H = 240


def lc3_y(kda: float) -> float:
    return 20.0 + 400.0 * math.log10(25.0 / kda)


def lc3(tmp_path) -> tuple[ProjectSession, str, str, str]:
    """A two-row blot of LC3-I over LC3-II, one band of each per lane, linked
    to a marker marked at 25, 15 and 10 kDa (not snapped), four lanes, and a
    protein for each row; (session, marker id, LC3-I's id, LC3-II's id)."""
    s = session_on(tmp_path)
    bands = [
        (x, lc3_y(kda) - 0.5, 8.0, 3.0, depth)
        for x in CAL_LANES
        for kda, depth in ((LC3_I, 30000.0), (LC3_II, 20000.0))
    ]
    pixels = noisy16(synthetic_blot((LC3_H, CAL_W), bands, dtype=np.float64), 61)
    image = import_blot(s, pixels, "LC3 β.tif")
    membrane = s.project.batch.membrane_of(image).id
    flat = noisy16(np.full((LC3_H, CAL_W), 50000.0), 62)
    kind = ImageKind.VISIBLE_MARKER
    marker = import_blot(s, flat, "marker µ.tif", kind=kind, membrane_id=membrane)
    ops.set_marker_image(s, image, marker)
    for kda in (25, 15, 10):
        ops.add_calibration_point(s, marker, lc3_y(kda), kda, MARKER_BAND, x=20.0, snap=False)
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(len(CAL_LANES))])
    first = ops.add_protein(s, "LC3-I", Role.TARGET, image, expected_mw=LC3_I)
    second = ops.add_protein(s, "LC3-II", Role.TARGET, image, expected_mw=LC3_II)
    return s, marker, first, second


LC3_ROW = (80, 82, 400, 135)  # LC3-I's row box, LC3-II's band inside it
LC3_II_ROW = (80, 110, 400, 135)


def test_count_window_ignores_neighbour_protein(tmp_path):
    s, marker, first, second = lc3(tmp_path)
    ops.detect_row_boxes(s, first, LC3_ROW)
    ops.detect_row_boxes(s, second, LC3_II_ROW)
    for protein, kda in ((first, LC3_I), (second, LC3_II)):
        bands = list(lane_bands(s, protein).values())
        assert len(bands) == len(CAL_LANES)
        assert all(abs(band.apparent_mw / kda - 1) < 0.02 for band in bands), protein
        assert [band.bands_found for band in bands] == [1] * len(CAL_LANES), protein
    res = ops.compute(s)
    first_column, second_column = res.proteins
    assert first_column.count_check == second_column.count_check == ["passed"] * 4
    assert first_column.mw_check == second_column.mw_check == ["passed"] * 4
    assert NoticeCode.BAND_COUNT not in codes(res)
    # Without a curve, LC3-I's count reads its row box, which holds LC3-II.
    ops.clear_calibration(s, marker)
    assert all(b.bands_found is None for b in protein_of(s, first).bands)  # cleared
    ops.detect_row_boxes(s, first, LC3_ROW)
    assert [b.bands_found for b in lane_bands(s, first).values()] == [2] * 4
    [notice] = of(ops.compute(s), NoticeCode.BAND_COUNT)
    assert notice.message == (
        "'LC3-I': 2 separate bands in its row box in lanes 1, 2, 3, 4, where 1 is expected"
    )


def test_a_band_with_two_tops_counts_once(tmp_path):
    # Lane 3's band has another below it, lighter across its middle than at
    # its ends: two tops, one band.
    case = two_topped_partner(1000)
    s, _, protein = row_session(tmp_path, case)
    ops.detect_row_boxes(s, protein, case.row)
    assert _counts(s, protein) == {0: 1, 1: 1, 2: 2, 3: 1, 4: 1, 5: 1}
    [notice] = of(ops.compute(s), NoticeCode.BAND_COUNT)
    assert notice.message == (
        "'β-catenin': 2 separate bands in its row box in lane 3, where 1 is expected"
    )


def _counts(s, protein_id: str) -> dict[int, int | None]:
    return {lane: band.bands_found for lane, band in lane_bands(s, protein_id).items()}


def test_band_counts_follow_their_boxes(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    placement = ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX)
    counted = {lane: 1 for lane in range(len(CAL_LANES))}
    assert _counts(s, c.protein) == counted
    entry = s.project.log[-1]
    assert "bands_found" not in json.dumps(entry.params)  # stored, not logged
    # The same drag again changes nothing.
    before = s.project
    assert ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX).band_ids == placement.band_ids
    assert s.project is before
    # Resizing and padding keep the counts: the boxes' centres stay.
    fitted = protein_of(s, c.protein).fitted_size
    ops.set_box_size(s, c.protein, BoxSize(width=fitted.width + 2, height=fitted.height + 2))
    ops.set_box_padding(s, c.protein, across=1, along=1)
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(len(CAL_LANES) + 1)])
    assert _counts(s, c.protein) == counted
    # A box moved or given another lane by hand loses its count.
    bands = lane_bands(s, c.protein)
    size = protein_of(s, c.protein).box_size
    x0, y0, x1, y1 = bands[1].box.rect(size)
    ops.move_box(s, bands[1].id, (x0 + 2, y0 + 1, x1 + 2, y1 + 1))
    ops.set_box_lane(s, bands[3].id, 4)
    assert _counts(s, c.protein) == {0: 1, 1: None, 2: 1, 4: None}
    # Undo brings a count back with its box; redo takes it again.
    ops.undo(s)
    assert _counts(s, c.protein) == {0: 1, 1: None, 2: 1, 3: 1}
    ops.redo(s)
    assert _counts(s, c.protein) == {0: 1, 1: None, 2: 1, 4: None}
    # A polarity change, then a curve change, clear every count on the image.
    ops.undo(s)
    ops.undo(s)
    ops.set_polarity(s, c.blot, Polarity.LIGHT_ON_DARK)
    assert set(_counts(s, c.protein).values()) == {None}
    ops.undo(s)
    assert _counts(s, c.protein) == counted
    update = ops.edit_calibration_point(s, c.marker, 55, y=point_y(s, 55) + 1.0)
    assert c.blot in update.curves_changed
    assert set(_counts(s, c.protein).values()) == {None}
    assert "bands_found" not in json.dumps(s.project.log[-1].params)
    ops.undo(s)
    assert _counts(s, c.protein) == counted
    ops.redo(s)
    assert set(_counts(s, c.protein).values()) == {None}


def point_y(s: ProjectSession, mw: float) -> float:
    """Where the left ladder's point at ``mw`` kDa lies now."""
    [point] = [
        p
        for p in s.project.batch.membranes[0].calibration.points
        if p.mw == mw and p.side is LadderSide.LEFT
    ]
    return point.y


def test_edit_protein_takes_a_tolerance_and_clears_counts_it_sized(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX)
    counted = _counts(s, c.protein)
    assert set(counted.values()) == {1}
    ops.edit_protein(s, c.protein, mw_tolerance=0.2)
    assert protein_of(s, c.protein).mw_tolerance == 0.2
    assert set(_counts(s, c.protein).values()) == {None}  # the window was ±10% of the curve
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "edit_protein",
        {
            "protein_id": c.protein,
            "pinned_targets": [],
            "mw_tolerance": 0.2,
            "dropped_undetected": [],
        },
    )
    assert ops.compute(s).proteins[0].mw_tolerance == 0.2
    ops.undo(s)
    assert protein_of(s, c.protein).mw_tolerance == 0.1 and _counts(s, c.protein) == counted
    ops.redo(s)
    assert protein_of(s, c.protein).mw_tolerance == 0.2
    before = s.project
    ops.edit_protein(s, c.protein, mw_tolerance=0.2)  # the same: a no-op
    assert s.project is before
    for bad in (0, 1, 1.5, -0.1, math.nan, math.inf, True, "0.1", None):
        with pytest.raises(OperationError) as info:
            ops.edit_protein(s, c.protein, mw_tolerance=bad)
        assert info.value.code is ErrorCode.INVALID_INPUT, bad
        assert s.project is before

    # Without a curve the window is the row box's: a new tolerance keeps the counts.
    bare = calibrated(tmp_path / "bare", sides=())
    ops.detect_row_boxes(bare.session, bare.protein, CAL_ROW_BOX)
    kept = _counts(bare.session, bare.protein)
    assert set(kept.values()) == {1}
    ops.edit_protein(bare.session, bare.protein, mw_tolerance=0.3)
    assert _counts(bare.session, bare.protein) == kept


def test_a_new_expected_mw_clears_the_counts_of_mw_guided_boxes(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX)
    guided = lane_bands(s, c.protein)[0].id
    mw_guided = ProposalSource.MW_GUIDED
    plant(s, lambda draft: setattr(draft.batch.find_band(guided)[1], "source", mw_guided))
    ops.edit_protein(s, c.protein, name="β-catenin (E-5)")  # the MW is kept: so are the counts
    assert set(_counts(s, c.protein).values()) == {1}
    ops.edit_protein(s, c.protein, expected_mw=92)
    assert _counts(s, c.protein) == {0: None, 1: 1, 2: 1, 3: 1}  # a row box's does not depend on it


def test_add_protein_takes_a_tolerance(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    protein = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, c.blot, mw_tolerance=0.25)
    assert protein_of(s, protein).mw_tolerance == 0.25
    assert s.project.log[-1].params["mw_tolerance"] == 0.25
    # Undo takes the protein back with its tolerance; redo brings both.
    ops.undo(s)
    assert all(p.id != protein for p in s.project.batch.proteins)
    ops.redo(s)
    assert protein_of(s, protein).mw_tolerance == 0.25
    before = s.project
    with pytest.raises(OperationError) as info:
        ops.add_protein(s, "α-tubulin", Role.LOADING_CONTROL, c.blot, mw_tolerance=1.0)
    assert info.value.code is ErrorCode.INVALID_INPUT
    assert "MW tolerance must be a share above 0 and below 1" in str(info.value)
    assert s.project is before


def test_a_count_on_an_edited_box_cannot_be_stored(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    ops.detect_row_boxes(s, c.protein, CAL_ROW_BOX)
    band = lane_bands(s, c.protein)[0].id

    def edit(draft: Project) -> None:
        draft.batch.find_band(band)[1].manually_edited = True

    with pytest.raises(ValueError, match="comes from a detector's box that nobody edited"):
        plant(s, edit)
    assert band_of(s, band).bands_found == 1
