# SPDX-License-Identifier: Apache-2.0
"""The molecular-weight calibration operations (#58): marker links, the
ladder, and calibration points marked, snapped, moved, relabelled and removed
on one ladder or two, with their refusals, logs and undo.

Every test works on synthetic 16-bit images written as TIFF files: the
two-ladder blot of ``test_operations`` (:func:`~test_operations.calibrated`),
strips and faint markers drawn here. The molecular-weight invariant
(:func:`~test_operations.assert_mw_current`) is checked after the changes.
"""

import json
import logging
import math

import numpy as np
import pytest

from conftest import MEMBRANE_LEVEL, synthetic_blot
from proteia.core import ladders, mwcal
from proteia.core import operations as ops
from proteia.core.model import (
    CalibrationPoint,
    CalibrationPointSource,
    ImageKind,
    Polarity,
    Project,
    Role,
    UnknownIdError,
)
from proteia.core.operations import CalibrationFit, ErrorCode, LadderFit, OperationError
from proteia.core.session import save_to_folder
from test_operations import (
    CAL_H,
    CAL_KDA,
    CAL_LANES,
    CAL_LEFT_X,
    CAL_RIGHT_X,
    CAL_ROW,
    CAL_W,
    LEFT,
    MARKER_BAND,
    RIGHT,
    Calibrated,
    Recorder,
    assert_mw_current,
    assert_nets_current,
    band_of,
    cal_blot,
    cal_marker,
    cal_y,
    calibrated,
    import_blot,
    noisy16,
    plant,
    protein_of,
    reprobe_on,
)

CHEMI_MARKER = CalibrationPointSource.CHEMILUMINESCENCE_MARKER
STRIP = CalibrationPointSource.STRIP_EDGE


def refused(code: ErrorCode, call, *args, **kwargs) -> OperationError:
    """Run a call that must be refused with ``code``; the project must be the
    same object after it (a refusal changes nothing)."""
    session = args[0]
    before = session.project
    with pytest.raises(OperationError) as info:
        call(*args, **kwargs)
    assert info.value.code is code, info.value
    assert session.project is before
    return info.value


def points(c: Calibrated, side=None) -> list[tuple[str, float, float, str, float | None, str]]:
    """The membrane's points as (image, y, MW, source, x, side), as stored."""
    return [
        (p.image_id, p.y, p.mw, p.source.value, p.x, p.side.value)
        for p in c.session.project.batch.membranes[0].calibration.points
        if side is None or p.side == side
    ]


def fitted(c: Calibrated, image_id: str):
    batch = c.session.project.batch
    return mwcal.calibration_for(batch.membrane_of(image_id), image_id)


# --- Points from each source ---


def test_point_from_marker_image(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    y70 = cal_y(70, CAL_LEFT_X)
    update = ops.add_calibration_point(s, c.marker, y70 + 3.0, 70, MARKER_BAND, x=CAL_LEFT_X)
    snapped_y = update.point["y"]
    assert abs(snapped_y - y70) < 0.2  # clicked 3 px low, snapped onto the band
    assert update.point == {
        "image_id": c.marker,
        "y": snapped_y,
        "mw": 70.0,
        "source": "visible_marker",
        "x": CAL_LEFT_X,
        "side": "left",
        "snapped": True,
    }
    # One point: no curve yet, so none changed.
    assert (update.fit, update.curves_changed, update.dropped_undetected) == (None, (), ())
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "add_calibration_point",
        {
            "membrane_id": c.membrane,
            "group": [c.blot, c.marker],
            "side": "left",
            "image_id": c.marker,  # which image of the group it was marked on
            "mw": 70.0,
            "source": "visible_marker",
            "y": snapped_y,
            "y_given": y70 + 3.0,
            "x": CAL_LEFT_X,
            "snapped": True,
            "fit": None,
            "curves_changed": [],
            "dropped_undetected": [],
        },
    )
    # A second point makes a curve for both images of the group, from two
    # points: no quality to measure.
    update = ops.add_calibration_point(
        s, c.marker, cal_y(35, CAL_LEFT_X) - 4.0, 35, "visible_marker", x=30
    )
    assert update.curves_changed == (c.blot, c.marker)
    fit = update.fit
    assert fit is not None and (fit.image_ids, fit.method) == (
        (c.blot, c.marker),
        "log_linear_piecewise",
    )
    [ladder] = fit.ladders
    assert (ladder.side, ladder.x, ladder.points, ladder.quality, ladder.two_point) == (
        "left",
        CAL_LEFT_X,
        2,
        None,
        True,
    )
    assert (fit.offset_px, fit.tilt_deg, fit.disagreement, fit.ignored) == (None, None, None, ())
    assert s.project.log[-1].params["fit"] == fit.as_json()
    calibration = s.project.batch.membranes[0].calibration
    assert (calibration.fit_method.value, calibration.fit_quality) == ("log_linear_piecewise", None)
    assert_mw_current(s)


@pytest.mark.parametrize("brighter", [False, True], ids=["dark", "bright"])
def test_point_from_faint_chemi_marker(tmp_path, brighter):
    # A faint prestained marker on the chemiluminescence image itself, which
    # may show darker or brighter than the membrane.
    depth = -2500.0 if brighter else 2500.0
    bands = [(CAL_LEFT_X, cal_y(kda, CAL_LEFT_X) - 0.5, 10.0, 2.0, depth) for kda in (100, 55)]
    image = synthetic_blot((CAL_H, CAL_W), bands, dtype=np.float64)
    c = calibrated(tmp_path, sides=())
    s = c.session
    chemi = import_blot(s, noisy16(image, 7), "faint µ.tif", membrane_id=c.membrane)
    for kda, off in ((100, 3.0), (55, -3.0)):
        update = ops.add_calibration_point(
            s, chemi, cal_y(kda, CAL_LEFT_X) + off, kda, CHEMI_MARKER, x=CAL_LEFT_X
        )
        assert update.point["snapped"], kda
        assert abs(update.point["y"] - cal_y(kda, CAL_LEFT_X)) < 0.3, kda
    assert isinstance(fitted(c, chemi), mwcal.Calibration)
    assert isinstance(fitted(c, c.blot), mwcal.NoCalibration)  # another register group


def strip_image() -> np.ndarray:
    """A strip cut between 100 and 75 kDa on film: the membrane from row 30 to
    row 149, the edges at y = 30 and y = 150."""
    image = np.full((CAL_H, CAL_W), 5000.0)
    image[30:150] = MEMBRANE_LEVEL
    return noisy16(image, 11)


def test_strip_edges_100_75(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    strip = import_blot(s, strip_image(), "strip α.tif")  # a membrane of its own
    top = ops.add_calibration_point(s, strip, 32.0, 100, STRIP, x=200.0)
    bottom = ops.add_calibration_point(s, strip, 148.0, 75, STRIP, x=260.0)
    assert (top.point["snapped"], bottom.point["snapped"]) == (True, True)
    assert abs(top.point["y"] - 30.0) < 0.5 and abs(bottom.point["y"] - 150.0) < 0.5
    curve = fitted(c, strip)
    assert curve.strip_edges_only
    # Nothing is extrapolated past a strip edge; its own MWs are inside.
    assert (curve.z_lo, curve.z_hi) == (math.log10(75), 2.0)
    assert curve.y_at(75, None) == bottom.point["y"] and curve.y_at(100, None) == top.point["y"]
    assert curve.mw_at(None, top.point["y"] - 0.1) is None
    assert bottom.fit.ladders[0].x == 230.0  # the median of the clicks' x


def test_strip_edges_on_the_image_border_keep_their_clicks(tmp_path):
    # A strip cropped at its cut, as mem-5 of the sample: its edges are the
    # image's top and bottom, with half a marker band the top cut runs through
    # and a band ten rows above the bottom, whose flanks a snap would take.
    bands = [(x, y, 10.0, 2.5, 12000.0) for x in (100, 300) for y in (0.0, CAL_H - 11.0)]
    image = noisy16(synthetic_blot((CAL_H, CAL_W), bands, dtype=np.float64), 13)
    c = calibrated(tmp_path, sides=())
    s = c.session
    strip = import_blot(s, image, "strip β.tif")
    top = ops.add_calibration_point(s, strip, 0.0, 100, STRIP, x=100.0)
    bottom = ops.add_calibration_point(s, strip, float(CAL_H), 75, STRIP, x=300.0)
    assert [(u.point["y"], u.point["snapped"]) for u in (top, bottom)] == [
        (0.0, False),
        (float(CAL_H), False),
    ]
    assert [(e.params["y"], e.params["snapped"]) for e in s.project.log[-2:]] == [
        (0.0, False),
        (float(CAL_H), False),
    ]
    assert (fitted(c, strip).z_lo, fitted(c, strip).z_hi) == (math.log10(75), 2.0)
    assert_mw_current(s)


def test_source_must_fit_image_kind(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    merged = import_blot(
        s, cal_marker(), "merged β.tif", kind=ImageKind.MERGED, membrane_id=c.membrane
    )
    y = cal_y(100, CAL_LEFT_X)
    error = refused(
        ErrorCode.INVALID_INPUT,
        ops.add_calibration_point,
        s,
        c.blot,
        y,
        100,
        MARKER_BAND,
        x=CAL_LEFT_X,
    )
    assert str(error) == (
        f"a visible_marker point is marked on a merged or visible_marker image, not on {c.blot}"
        " (chemiluminescence)"
    )
    refused(
        ErrorCode.INVALID_INPUT,
        ops.add_calibration_point,
        s,
        c.marker,
        y,
        100,
        CHEMI_MARKER,
        x=CAL_LEFT_X,
    )
    refused(ErrorCode.INVALID_INPUT, ops.add_calibration_point, s, c.marker, y, 100, "ladder", x=1)
    # A merged image takes both kinds of marker band; a strip edge goes anywhere.
    ops.add_calibration_point(s, merged, y, 100, MARKER_BAND, x=CAL_LEFT_X, snap=False)
    ops.add_calibration_point(
        s, merged, cal_y(55, CAL_LEFT_X), 55, CHEMI_MARKER, x=CAL_LEFT_X, snap=False
    )
    ops.add_calibration_point(s, c.marker, 1.0, 400, STRIP, x=5.0, snap=False)
    with pytest.raises(UnknownIdError):
        ops.add_calibration_point(s, "img-99", y, 100, MARKER_BAND, x=CAL_LEFT_X)


def test_every_new_point_records_its_x(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    for source in CalibrationPointSource:
        image = c.blot if source is CHEMI_MARKER else c.marker
        error = refused(
            ErrorCode.INVALID_INPUT, ops.add_calibration_point, s, image, 50.0, 90, source
        )
        assert "needs its x" in str(error)
    # The position is checked against the image, in continuous coordinates.
    for x, y in ((CAL_W + 0.5, 50.0), (-1.0, 50.0), (10.0, CAL_H + 0.1), (10.0, -0.5)):
        refused(ErrorCode.OUT_OF_IMAGE, ops.add_calibration_point, s, c.marker, y, 90, STRIP, x=x)
    ops.add_calibration_point(s, c.marker, float(CAL_H), 5, STRIP, x=float(CAL_W), snap=False)
    for bad in ("50", True, math.nan, math.inf):
        refused(
            ErrorCode.INVALID_INPUT, ops.add_calibration_point, s, c.marker, bad, 90, STRIP, x=1
        )
    for bad in (0, -5, math.inf, "90", True):
        refused(
            ErrorCode.INVALID_INPUT, ops.add_calibration_point, s, c.marker, 50.0, bad, STRIP, x=1
        )
    refused(
        ErrorCode.INVALID_INPUT,
        ops.add_calibration_point,
        s,
        c.marker,
        50.0,
        90,
        STRIP,
        x=1,
        snap=1,
    )
    refused(
        ErrorCode.INVALID_INPUT,
        ops.add_calibration_point,
        s,
        c.marker,
        50.0,
        90,
        STRIP,
        x=1,
        side="middle",
    )
    c = calibrated(tmp_path / "both")
    assert all(x is not None for *_, x, _ in points(c))


def test_right_ladder_point_needs_x(tmp_path):
    c = calibrated(tmp_path, sides=(LEFT,))
    s = c.session
    y = cal_y(70, CAL_RIGHT_X)
    error = refused(
        ErrorCode.INVALID_INPUT,
        ops.add_calibration_point,
        s,
        c.marker,
        y,
        70,
        STRIP,
        x=400,
        side=RIGHT,
    )
    assert str(error) == "a strip edge belongs to the left ladder"

    # A point saved before points recorded their x (a file from before #58).
    def legacy(draft: Project) -> None:
        points = draft.batch.membranes[0].calibration.points
        points.append(
            CalibrationPoint(image_id=c.marker, y=cal_y(12, CAL_LEFT_X), mw=12, source=MARKER_BAND)
        )

    plant(s, legacy)
    error = refused(
        ErrorCode.LADDER_SIDES,
        ops.add_calibration_point,
        s,
        c.marker,
        y,
        70,
        MARKER_BAND,
        x=CAL_RIGHT_X,
        side=RIGHT,
    )
    assert "12 kDa" in str(error) and "remove it and mark it again" in str(error)
    ops.remove_calibration_point(s, c.marker, 12)
    # A strip edge in the group: a right ladder cannot join it.
    ops.add_calibration_point(s, c.marker, 1.0, 400, STRIP, x=5.0, snap=False)
    error = refused(
        ErrorCode.LADDER_SIDES,
        ops.add_calibration_point,
        s,
        c.marker,
        y,
        70,
        MARKER_BAND,
        x=CAL_RIGHT_X,
        side=RIGHT,
    )
    assert "strip edge" in str(error)
    ops.remove_calibration_point(s, c.marker, 400)
    ops.add_calibration_point(
        s, c.marker, y, 70, MARKER_BAND, x=CAL_RIGHT_X, side=RIGHT, snap=False
    )
    # And now no strip edge can join the group.
    refused(
        ErrorCode.LADDER_SIDES,
        ops.add_calibration_point,
        s,
        c.marker,
        1.0,
        400,
        STRIP,
        x=5.0,
        snap=False,
    )


def test_sides_crossing_refused(tmp_path):
    c = calibrated(tmp_path, sides=(LEFT,))
    s = c.session
    error = refused(
        ErrorCode.LADDER_SIDES,
        ops.add_calibration_point,
        s,
        c.marker,
        cal_y(70, 10.0),
        70,
        MARKER_BAND,
        x=10.0,
        side=RIGHT,
        snap=False,
    )
    assert str(error) == (
        f"membrane {c.membrane}: the right ladder of {c.blot}, {c.marker} (x=10) would not lie"
        " right of its left ladder (x=30); the second ladder is the one right of the first"
    )
    c = calibrated(tmp_path / "two")
    refused(
        ErrorCode.LADDER_SIDES,
        ops.add_calibration_point,
        c.session,
        c.marker,
        cal_y(10, 460.0),
        10,
        MARKER_BAND,
        x=460.0,
        snap=False,
    )


def test_point_identity_is_group_side_mw(tmp_path):
    c = calibrated(tmp_path)  # every MW on both ladders
    s = c.session
    reprobe = reprobe_on(c)
    assert len(points(c, LEFT)) == len(points(c, RIGHT)) == len(CAL_KDA)
    with pytest.raises(UnknownIdError):  # the reprobe is a group of its own
        ops.remove_calibration_point(s, reprobe, 70)
    # Named by either image of the group; the side tells the two 70-kDa points apart.
    update = ops.remove_calibration_point(s, c.blot, 70, side=RIGHT)
    assert 70.0 not in [mw for _, _, mw, *_ in points(c, RIGHT)]
    assert 70.0 in [mw for _, _, mw, *_ in points(c, LEFT)]
    assert s.project.log[-1].params["removed"] == {
        "image_id": c.marker,
        "y": cal_y(70, CAL_RIGHT_X),
        "mw": 70.0,
        "source": "visible_marker",
        "x": CAL_RIGHT_X,
        "side": "right",
    }
    assert update.curves_changed == (c.blot, c.marker)
    with pytest.raises(UnknownIdError):
        ops.remove_calibration_point(s, c.marker, 70, side=RIGHT)
    with pytest.raises(UnknownIdError):
        ops.edit_calibration_point(s, c.marker, 71, y=10.0)
    refused(ErrorCode.INVALID_INPUT, ops.remove_calibration_point, s, c.marker, -70)
    assert_mw_current(s)


def test_reprobe_group_gets_its_own_curve(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    reprobe = reprobe_on(c)
    blot_curve = fitted(c, c.blot)
    update = ops.add_calibration_point(s, reprobe, 10.0, 150, STRIP, x=100.0, snap=False)
    assert update.curves_changed == ()  # one point: no curve anywhere changed
    update = ops.add_calibration_point(s, reprobe, 190.0, 20, STRIP, x=100.0, snap=False)
    assert update.curves_changed == (reprobe,)
    assert update.fit.image_ids == (reprobe,)
    assert fitted(c, c.blot) == blot_curve
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, reprobe)
    ops.set_lanes(s, [ops.LaneInput(f"c{i}") for i in range(4)])
    band = ops.place_box(s, gapdh, CAL_LANES[0], CAL_ROW, lane_index=0, grow=True)
    # Read from the reprobe's strip: log-linear from 150 kDa at y 10 to 20 at y 190.
    size = protein_of(s, gapdh).box_size
    cy = band_of(s, band).box.y + size.height / 2
    expected = 150.0 * (20.0 / 150.0) ** ((cy - 10.0) / 180.0)
    assert math.isclose(band_of(s, band).apparent_mw, expected, rel_tol=1e-12)
    assert_mw_current(s)


def test_merged_image_as_marker(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    merged = import_blot(
        s, cal_marker(), "merged β.tif", kind=ImageKind.MERGED, membrane_id=c.membrane
    )
    ops.set_marker_image(s, c.blot, merged)
    for kda in (100, 55, 35):
        ops.add_calibration_point(
            s, merged, cal_y(kda, CAL_LEFT_X) + 2.0, kda, MARKER_BAND, x=CAL_LEFT_X
        )
    curve = fitted(c, c.blot)
    assert isinstance(curve, mwcal.Calibration) and curve.group == {c.blot, merged}
    assert isinstance(fitted(c, c.marker), mwcal.NoCalibration)  # no longer linked
    assert s.project.batch.find_image(c.blot).marker_image_id == merged


def test_link_refused_for_other_size(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    small = import_blot(
        s,
        cal_marker()[:190],
        "small α.tif",
        kind=ImageKind.VISIBLE_MARKER,
        membrane_id=c.membrane,
    )
    error = refused(ErrorCode.MARKER_SIZE_MISMATCH, ops.set_marker_image, s, c.blot, small)
    assert error.ids == (c.blot, small)
    assert str(error) == (
        f"{c.blot} ({CAL_W}x{CAL_H}) and {small} ({CAL_W}x190) differ in size; a marker image"
        " must match its image pixel for pixel"
    )
    other = import_blot(s, cal_marker(), "other membrane.tif", kind=ImageKind.VISIBLE_MARKER)
    refused(ErrorCode.INVALID_INPUT, ops.set_marker_image, s, c.blot, other)  # another membrane
    refused(ErrorCode.INVALID_INPUT, ops.set_marker_image, s, c.marker, c.marker)  # not chemi
    second = import_blot(s, cal_blot(), "second chemi.tif", membrane_id=c.membrane)
    refused(ErrorCode.INVALID_INPUT, ops.set_marker_image, s, c.blot, second)  # not a marker
    with pytest.raises(UnknownIdError):
        ops.set_marker_image(s, c.blot, "img-99")
    with pytest.raises(UnknownIdError):
        ops.set_marker_image(s, "img-99", c.marker)
    committed = s.project
    ops.set_marker_image(s, c.blot, c.marker)  # the same link: a no-op
    assert s.project is committed


def test_link_merging_conflicting_groups_refused(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    ops.set_marker_image(s, c.blot, None)
    y70 = cal_y(70, CAL_LEFT_X)
    ops.add_calibration_point(s, c.blot, y70 + 5.0, 70, CHEMI_MARKER, x=CAL_LEFT_X, snap=False)
    error = refused(ErrorCode.DUPLICATE_MW, ops.set_marker_image, s, c.blot, c.marker)
    assert str(error) == (
        f"membrane {c.membrane}: the left ladder of {c.blot}, {c.marker} would hold 70 kDa twice:"
        f" 70 kDa at y={y70:g} on {c.marker} and 70 kDa at y={y70 + 5.0:g} on {c.blot}"
    )
    ops.edit_calibration_point(s, c.blot, 70, new_mw=60, y=cal_y(100, CAL_LEFT_X) - 1.0)
    error = refused(ErrorCode.CALIBRATION_ORDER, ops.set_marker_image, s, c.blot, c.marker)
    assert "out of order on the left ladder" in str(error) and f"on {c.blot}" in str(error)
    ops.clear_calibration(s, c.blot)
    ops.add_calibration_point(s, c.blot, 1.0, 400, STRIP, x=5.0, snap=False)
    error = refused(ErrorCode.LADDER_SIDES, ops.set_marker_image, s, c.blot, c.marker)
    assert "strip edge" in str(error)
    # Unlinking never conflicts; linking two groups that agree does.
    ops.clear_calibration(s, c.blot)
    ops.add_calibration_point(
        s, c.blot, cal_y(10, CAL_LEFT_X), 10, CHEMI_MARKER, x=CAL_LEFT_X, snap=False
    )
    update = ops.set_marker_image(s, c.blot, c.marker)
    assert update.curves_changed == (c.blot, c.marker)
    assert len(points(c, LEFT)) == len(CAL_KDA) + 1
    assert s.project.log[-1].params == {
        "membrane_id": c.membrane,
        "image_id": c.blot,
        "marker_image_id": c.marker,
        "previous": None,
        "group": [c.blot, c.marker],
        "fit": update.fit.as_json(),
        "curves_changed": [c.blot, c.marker],
        "dropped_undetected": [],
    }
    assert_mw_current(s)


def test_remove_shared_marker_splits_groups(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    reprobe = reprobe_on(c)
    ops.set_marker_image(s, reprobe, c.marker)
    marks = ((c.marker, (250, 130, 100)), (c.blot, (55, 35)), (reprobe, (25, 15, 10)))
    for image, kdas in marks:
        source = MARKER_BAND if image == c.marker else CHEMI_MARKER
        for kda in kdas:
            # 35 kDa marked 3 px low: the one point off the blot's law.
            y = cal_y(kda, CAL_LEFT_X) + (3.0 if kda == 35 else 0.0)
            ops.add_calibration_point(s, image, y, kda, source, x=CAL_LEFT_X, snap=False)
    # One group of three images: each entry names the image its point is on.
    added = [entry.params for entry in s.project.log if entry.action == "add_calibration_point"]
    assert {len(params["group"]) for params in added} == {3}
    assert [params["image_id"] for params in added] == [i for i, kdas in marks for _ in kdas]
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, reprobe)
    for protein in (c.protein, gapdh):
        ops.place_box(s, protein, CAL_LANES[0], CAL_ROW, lane_index=0, grow=True)
    one = fitted(c, c.blot)
    assert one.group == {c.blot, c.marker, reprobe} and len(one.ladders[0].ys) == 8
    quality = s.project.batch.membranes[0].calibration.fit_quality
    assert quality > 0.01  # 35 kDa, off the line between 55 and 25

    cascade = ops.remove_image(s, c.marker)
    assert cascade.unpaired_images == (c.blot, reprobe)
    assert cascade.unfitted_membranes == (c.membrane,)
    # Each image is its own group now, fitted from its own points: the blot's
    # two (no quality), the reprobe's three (on the law: about 0).
    blot_curve, reprobe_curve = fitted(c, c.blot), fitted(c, reprobe)
    assert blot_curve.ladders[0].mws == (55, 35) and reprobe_curve.ladders[0].mws == (25, 15, 10)
    calibration = s.project.batch.membranes[0].calibration
    assert calibration.fit_quality == reprobe_curve.quality.value < 1e-12
    assert_mw_current(s)
    assert_nets_current(s)


# --- Refusals by position and label ---


def test_duplicate_mw_refused(tmp_path):
    c = calibrated(tmp_path, sides=(LEFT,))
    s = c.session
    y100 = cal_y(100, CAL_LEFT_X)
    error = refused(
        ErrorCode.DUPLICATE_MW,
        ops.add_calibration_point,
        s,
        c.blot,
        190.0,
        100,
        CHEMI_MARKER,
        x=30,
        snap=False,
    )
    assert str(error) == (
        f"membrane {c.membrane}: the left ladder of {c.blot}, {c.marker} already has a point at"
        f" 100 kDa (100 kDa at y={y100:g} on {c.marker})"
    )
    # The same MW on the other ladder is another point.
    ops.add_calibration_point(
        s,
        c.marker,
        cal_y(100, CAL_RIGHT_X),
        100,
        MARKER_BAND,
        x=CAL_RIGHT_X,
        side=RIGHT,
        snap=False,
    )
    refused(ErrorCode.DUPLICATE_MW, ops.edit_calibration_point, s, c.marker, 70, new_mw=100)


def test_out_of_order_refused(tmp_path):
    c = calibrated(tmp_path, sides=(LEFT,))
    s = c.session
    y70, y55 = cal_y(70, CAL_LEFT_X), cal_y(55, CAL_LEFT_X)
    error = refused(
        ErrorCode.CALIBRATION_ORDER,
        ops.add_calibration_point,
        s,
        c.marker,
        y70 - 1.0,
        60,
        MARKER_BAND,
        x=CAL_LEFT_X,
        snap=False,
    )
    assert str(error) == (
        f"membrane {c.membrane}: 60 kDa at y={y70 - 1.0:g} is out of order on the left ladder of"
        f" {c.blot}, {c.marker}: it belongs below 70 kDa at y={y70:g} on {c.marker} and above"
        f" 55 kDa at y={y55:g} on {c.marker}, since lighter bands run further down"
    )
    error = refused(
        ErrorCode.CALIBRATION_ORDER,
        ops.add_calibration_point,
        s,
        c.marker,
        5.0,
        10,
        MARKER_BAND,
        x=30,
        snap=False,
    )
    assert "it belongs below 15 kDa" in str(error) and "above" not in str(error).split("belongs")[1]
    error = refused(
        ErrorCode.CALIBRATION_ORDER,
        ops.add_calibration_point,
        s,
        c.marker,
        y70,
        60,
        MARKER_BAND,
        x=30,
        snap=False,
    )
    assert f"two points at y={y70:g}" in str(error)
    refused(ErrorCode.CALIBRATION_ORDER, ops.edit_calibration_point, s, c.marker, 55, y=y70 - 2.0)


def test_faint_click_keeps_given_y(tmp_path, caplog):
    c = calibrated(tmp_path, sides=())
    s = c.session
    with caplog.at_level(logging.INFO, logger="proteia.core.operations"):
        update = ops.add_calibration_point(s, c.marker, 190.0, 10, MARKER_BAND, x=CAL_LEFT_X)
    assert (update.point["y"], update.point["snapped"]) == (190.0, False)
    assert s.project.log[-1].params["snapped"] is False
    assert [r.getMessage() for r in caplog.records] == [
        f"in {s.folder.name!r}: no ladder band or edge stands out near y=190 on {c.marker}; the"
        " clicked position is kept"
    ]


def test_snap_never_reaches_a_marked_band(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    y70, y55 = cal_y(70, CAL_LEFT_X), cal_y(55, CAL_LEFT_X)
    ops.add_calibration_point(s, c.marker, y70, 70, MARKER_BAND, x=CAL_LEFT_X)
    # 5 px below the marked 70-kDa band, 7.6 px above the 55-kDa one: the
    # window stops short of the marked band, so the click takes the other.
    update = ops.add_calibration_point(s, c.marker, y70 + 5.0, 55, MARKER_BAND, x=CAL_LEFT_X)
    assert abs(update.point["y"] - y55) < 0.2
    # The right ladder's points do not shorten the left one's window.
    snapped = mwcal.refine_point(
        s.pixels(c.marker),
        CAL_LEFT_X,
        y70 + 5.0,
        source=MARKER_BAND,
        polarity=Polarity.DARK_ON_LIGHT,
    )
    assert abs(snapped - y70) < 0.2


def test_edit_point_relabel_is_one_step(tmp_path):
    c = calibrated(tmp_path, sides=(LEFT,))
    s = c.session
    before, length = s.project, len(s.project.log)
    y55 = cal_y(55, CAL_LEFT_X)
    update = ops.edit_calibration_point(s, c.blot, 55, y=y55 + 1.0, new_mw=50)
    assert len(s.project.log) == length + 1
    assert update.point == {
        "image_id": c.marker,
        "y": y55 + 1.0,
        "mw": 50.0,
        "source": "visible_marker",
        "x": CAL_LEFT_X,
        "side": "left",
        "snapped": False,
    }
    assert s.project.log[-1].params == {
        "membrane_id": c.membrane,
        "group": [c.blot, c.marker],
        "side": "left",
        "mw": 55.0,
        "new_mw": 50.0,
        "from_y": y55,
        "y": y55 + 1.0,
        "y_given": y55 + 1.0,
        "snapped": False,
        "fit": update.fit.as_json(),
        "curves_changed": [c.blot, c.marker],
        "dropped_undetected": [],
    }
    # A drag snapped back onto the band; then the same position again: a no-op.
    update = ops.edit_calibration_point(s, c.marker, 50, y=y55 + 2.5, snap=True)
    assert update.point["snapped"] and abs(update.point["y"] - y55) < 0.2
    committed = s.project
    ops.edit_calibration_point(s, c.marker, 50, y=update.point["y"])
    assert s.project is committed
    refused(ErrorCode.OUT_OF_IMAGE, ops.edit_calibration_point, s, c.marker, 50, y=CAL_H + 1.0)
    refused(ErrorCode.INVALID_INPUT, ops.edit_calibration_point, s, c.marker, 50, new_mw=0)
    ops.undo(s)
    ops.undo(s)
    assert s.project.batch == before.batch


# --- The ladder ---


def test_set_ladder_copies_preset_kda(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    preset = ladders.preset("pageruler/bis_tris_mops")
    ops.set_ladder(s, c.membrane, preset.key)
    calibration = s.project.batch.membranes[0].calibration
    assert (calibration.ladder, calibration.ladder_kda) == (preset.key, list(preset.kda))
    assert s.project.log[-1].params == {
        "membrane_id": c.membrane,
        "ladder": preset.key,
        "kda": list(preset.kda),
    }
    committed = s.project
    ops.set_ladder(s, c.membrane, preset.key, kda=list(preset.kda))  # the same: a no-op
    assert s.project is committed
    error = refused(
        ErrorCode.INVALID_INPUT, ops.set_ladder, s, c.membrane, preset.key, kda=[140, 115]
    )
    assert "differ from those of the preset" in str(error)


def test_custom_ladder_name_and_kda(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    curve = fitted(c, c.blot)
    ops.set_ladder(s, c.membrane, "  In-house  µ ladder ", kda=[180, 100, 40])
    calibration = s.project.batch.membranes[0].calibration
    assert (calibration.ladder, calibration.ladder_kda) == (
        "In-house µ ladder",
        [180.0, 100.0, 40.0],
    )
    assert fitted(c, c.blot) == curve  # the ladder chosen changes no curve
    assert s.project.log[-1].params == {
        "membrane_id": c.membrane,
        "ladder": "In-house µ ladder",
        "kda": [180.0, 100.0, 40.0],
    }
    ops.set_ladder(s, c.membrane, "no list")
    assert s.project.batch.membranes[0].calibration.ladder_kda == []
    for bad in ([40, 100], [100, 100], [100, -5], "180, 100", [100, True]):
        refused(ErrorCode.INVALID_INPUT, ops.set_ladder, s, c.membrane, "ours", kda=bad)
    error = refused(ErrorCode.INVALID_INPUT, ops.set_ladder, s, c.membrane, None, kda=[100])
    assert str(error) == "ladder MWs need a ladder name"
    refused(ErrorCode.BLANK_TEXT, ops.set_ladder, s, c.membrane, "   ")
    ops.set_ladder(s, c.membrane, None)
    calibration = s.project.batch.membranes[0].calibration
    assert (calibration.ladder, calibration.ladder_kda) == (None, [])
    assert len(calibration.points) == 2 * len(CAL_KDA)  # the points stay


def test_unknown_preset_refused(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    error = refused(ErrorCode.INVALID_INPUT, ops.set_ladder, s, c.membrane, "pageruler/nope")
    assert str(error) == "no ladder preset 'pageruler/nope'; a custom ladder's name holds no '/'"
    with pytest.raises(UnknownIdError):
        ops.set_ladder(s, "mem-99", "ours")


# --- Clearing, undo, and the fit's JSON ---


def test_clear_calibration_by_side(tmp_path):
    c = calibrated(tmp_path)
    s = c.session
    update = ops.clear_calibration(s, c.blot, side=RIGHT)
    assert points(c, RIGHT) == [] and len(points(c, LEFT)) == len(CAL_KDA)
    assert [ladder.side for ladder in update.fit.ladders] == ["left"]
    params = s.project.log[-1].params
    assert (params["side"], len(params["removed"])) == ("right", len(CAL_KDA))
    committed = s.project
    ops.clear_calibration(s, c.blot, side=RIGHT)  # nothing left there: a no-op
    assert s.project is committed
    update = ops.clear_calibration(s, c.marker)
    assert (update.fit, points(c)) == (None, [])
    assert s.project.log[-1].params["side"] is None
    calibration = s.project.batch.membranes[0].calibration
    assert calibration.ladder == "pageruler_plus/tris_glycine"  # the ladder stays
    assert_mw_current(s)


def test_every_calibration_operation_is_undone_and_redone(tmp_path):
    c = calibrated(tmp_path, save_to_folder, sides=(LEFT,))
    s = c.session
    ops.detect_row_boxes(s, c.protein, (80, 85, 400, 116))
    y = cal_y(70, CAL_RIGHT_X)
    steps = [
        lambda: ops.set_ladder(s, c.membrane, "ours", kda=[250, 70]),
        lambda: ops.add_calibration_point(
            s, c.marker, y, 70, MARKER_BAND, x=CAL_RIGHT_X, side=RIGHT
        ),
        lambda: ops.add_calibration_point(
            s,
            c.marker,
            cal_y(35, CAL_RIGHT_X),
            35,
            MARKER_BAND,
            x=CAL_RIGHT_X,
            side=RIGHT,
            snap=False,
        ),
        lambda: ops.edit_calibration_point(
            s, c.marker, 35, side=RIGHT, y=cal_y(35, CAL_RIGHT_X) + 2
        ),
        lambda: ops.remove_calibration_point(s, c.blot, 250),
        lambda: ops.set_marker_image(s, c.blot, None),
        lambda: ops.set_marker_image(s, c.blot, c.marker),
        lambda: ops.clear_calibration(s, c.blot, side=LEFT),
        lambda: ops.set_ladder_points(
            s, c.marker, LEFT, [(cal_y(m, CAL_LEFT_X), m) for m in (250, 70)], x=CAL_LEFT_X
        ),
    ]
    actions = []
    for call in steps:
        before = s.project
        call()
        after = s.project
        assert after is not before
        actions.append(after.log[-1].action)
        assert ops.undo(s).action == actions[-1]
        assert s.project.batch == before.batch
        assert_mw_current(s)
        ops.redo(s)
        assert s.project.batch == after.batch
        assert_mw_current(s)
    assert set(actions) == {
        "set_ladder",
        "add_calibration_point",
        "edit_calibration_point",
        "remove_calibration_point",
        "set_marker_image",
        "clear_calibration",
        "set_ladder_points",
    }


def test_fit_json_has_no_infinity():
    ladder = LadderFit("left", 30.0, 3, math.inf, 70.0, 55.0, False, (10.0, math.inf))
    fit = CalibrationFit(
        image_ids=("img-1",),
        method="log_linear_piecewise",
        ladders=(ladder,),
        range=(10.0, math.inf),
        offset_px=2.0,
        tilt_deg=0.5,
        disagreement=math.inf,
        disagreement_mw=250.0,
        ignored=(("right", "one_point"),),
    )
    doc = fit.as_json()
    json.dumps(doc, allow_nan=False)  # strict JSON
    assert (doc["disagreement"], doc["disagreement_infinite"], doc["range"]) == (
        None,
        True,
        [10.0, None],
    )
    [entry] = doc["ladders"]
    assert (entry["quality"], entry["quality_infinite"], entry["worst_mw"]) == (None, True, 70.0)
    assert doc["ignored"] == [["right", "one_point"]]
    finite = CalibrationFit(
        ("img-1",), "log_linear_piecewise", (), (1.0, 2.0), None, None, None, None, ()
    )
    assert (finite.as_json()["disagreement"], finite.as_json()["disagreement_infinite"]) == (
        None,
        False,
    )


def test_a_fit_past_the_largest_float_is_refused(tmp_path):
    # Labels hundreds of decades apart: the leave-one-out quality overflows a
    # float, which no file can store.
    c = calibrated(tmp_path, sides=())
    s = c.session
    ops.add_calibration_point(s, c.marker, 10.0, 1e300, MARKER_BAND, x=CAL_LEFT_X, snap=False)
    ops.add_calibration_point(s, c.marker, 99.0, 1e299, MARKER_BAND, x=CAL_LEFT_X, snap=False)
    error = refused(
        ErrorCode.INVALID_INPUT,
        ops.add_calibration_point,
        s,
        c.marker,
        100.0,
        1e-300,
        MARKER_BAND,
        x=CAL_LEFT_X,
        snap=False,
    )
    assert "too far apart" in str(error)
    assert json.dumps(
        ops.calibration_fit(s.project.batch.membranes[0], c.blot).as_json(), allow_nan=False
    )


# --- Finding a ladder and applying it (D8) ---


def ruler_ready(tmp_path, *, sides=(), hook=None) -> Calibrated:
    """The two-ladder blot with the ladders ``sides`` marked, and its own MWs
    chosen as a custom ladder (its bands are all as dark: no reference band)."""
    c = calibrated(tmp_path, hook, sides=sides)
    ops.set_ladder(c.session, c.membrane, "ladder µ", kda=list(CAL_KDA))
    return c


def ruler(proposal: mwcal.LadderProposal) -> list[tuple[float, float]]:
    """A proposal applied as it is: each tick's (y, MW)."""
    return [(tick.y, tick.mw) for tick in proposal.ticks]


def placed(update: ops.CalibrationUpdate) -> list[tuple[float, str, bool | None]]:
    return [(p["mw"], p["placed"], p.get("relabelled")) for p in update.points]


def test_propose_and_snap_change_nothing(tmp_path):
    hook = Recorder()
    c = ruler_ready(tmp_path, hook=hook)
    s = c.session
    before, entries, calls = s.project, len(s.project.log), len(hook.actions)
    for side, x in ((LEFT, CAL_LEFT_X), (RIGHT, CAL_RIGHT_X)):
        proposal = ops.propose_ladder(s, c.marker, x, side)
        assert [(t.mw, t.found) for t in proposal.ticks] == [(mw, True) for mw in CAL_KDA]
        assert all(abs(t.y - cal_y(t.mw, x)) < 0.05 for t in proposal.ticks)
        assert (proposal.extra, proposal.doubtful, proposal.x) == ((), False, x)
    # Named by the other image of the group, the blot: a chemiluminescence
    # image, read as a faint marker; no ladder there.
    assert ops.propose_ladder(s, c.blot, CAL_LEFT_X) is None
    ys = [cal_y(mw, CAL_LEFT_X) + 3.0 for mw in (250, 70, 15)] + [190.0]
    snapped = ops.snap_ladder(s, c.marker, CAL_LEFT_X, ys)
    assert [moved for _, moved in snapped] == [True, True, True, False]
    for (y, _), mw in zip(snapped[:3], (250, 70, 15), strict=True):
        assert abs(y - cal_y(mw, CAL_LEFT_X)) < 0.2
    assert snapped[3] == (190.0, False)  # nothing stands out there: kept
    assert ops.snap_ladder(s, c.marker, CAL_LEFT_X, []) == ()
    # Nothing changed, logged or saved.
    assert s.project is before and len(s.project.log) == entries
    assert len(hook.actions) == calls


def test_snap_keeps_each_tick_off_its_neighbours_bands(tmp_path):
    c = ruler_ready(tmp_path)
    s = c.session
    y70, y55 = cal_y(70, CAL_LEFT_X), cal_y(55, CAL_LEFT_X)
    # A tick 5 px below 70 kDa with a neighbour on the 70-kDa band snaps to 55.
    [(high, _), (low, moved)] = ops.snap_ladder(s, c.marker, CAL_LEFT_X, [y70, y70 + 5.0])
    assert abs(high - y70) < 0.2 and moved and abs(low - y55) < 0.2


def test_proposal_takes_the_membranes_ladder_and_the_left_ladder(tmp_path, monkeypatch):
    c = calibrated(tmp_path, sides=(LEFT,))
    s = c.session
    calls = []
    find = mwcal.find_ladder

    def recording(array, x, **kw):
        calls.append(kw)
        return find(array, x, **kw)

    monkeypatch.setattr(mwcal, "find_ladder", recording)
    ops.propose_ladder(s, c.marker, CAL_RIGHT_X, RIGHT)
    ops.propose_ladder(s, c.marker, CAL_LEFT_X)
    ops.set_ladder(s, c.membrane, "ladder µ", kda=list(CAL_KDA))
    ops.propose_ladder(s, c.blot, CAL_LEFT_X, "left")
    preset = ladders.preset("pageruler_plus/tris_glycine")
    left = fitted(c, c.marker).ladders[0].curve
    assert calls == [
        {
            "kda": list(preset.kda),
            "reference_kda": (70.0, 25.0, 10.0),
            "source": MARKER_BAND,
            "polarity": Polarity.DARK_ON_LIGHT,
            "other": left,  # the right ladder is found against the left one
        },
        {
            "kda": list(preset.kda),
            "reference_kda": (70.0, 25.0, 10.0),
            "source": MARKER_BAND,
            "polarity": Polarity.DARK_ON_LIGHT,
            "other": None,
        },
        {
            "kda": [float(mw) for mw in CAL_KDA],
            "reference_kda": (),  # a custom ladder has no reference bands
            "source": CHEMI_MARKER,
            "polarity": Polarity.DARK_ON_LIGHT,
            "other": None,
        },
    ]


def test_proposal_refusals(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    for x in (math.nan, math.inf, "30", True):
        refused(ErrorCode.INVALID_INPUT, ops.propose_ladder, s, c.marker, x)
        refused(ErrorCode.INVALID_INPUT, ops.snap_ladder, s, c.marker, x, [50.0])
    for x in (-1.0, CAL_W + 0.5):
        refused(ErrorCode.OUT_OF_IMAGE, ops.propose_ladder, s, c.marker, x)
    refused(ErrorCode.INVALID_INPUT, ops.propose_ladder, s, c.marker, 30.0, "middle")
    refused(ErrorCode.OUT_OF_IMAGE, ops.snap_ladder, s, c.marker, 30.0, [50.0, CAL_H + 1.0])
    refused(ErrorCode.INVALID_INPUT, ops.snap_ladder, s, c.marker, 30.0, [50.0, math.nan])
    refused(ErrorCode.INVALID_INPUT, ops.snap_ladder, s, c.marker, 30.0, "50")
    with pytest.raises(UnknownIdError):
        ops.propose_ladder(s, "img-99", 30.0)
    with pytest.raises(UnknownIdError):
        ops.snap_ladder(s, "img-99", 30.0, [50.0])
    ops.set_ladder(s, c.membrane, None)
    error = refused(ErrorCode.INVALID_INPUT, ops.propose_ladder, s, c.marker, 30.0)
    assert str(error) == f"membrane {c.membrane} has no ladder MWs to find: choose its ladder first"


def test_set_ladder_points_is_one_step(tmp_path):
    c = ruler_ready(tmp_path, hook=save_to_folder)
    s = c.session
    ops.detect_row_boxes(s, c.protein, (80, 85, 400, 116))
    before, entries = s.project, len(s.project.log)
    proposal = ops.propose_ladder(s, c.marker, CAL_LEFT_X)
    update = ops.set_ladder_points(
        s, c.marker, LEFT, ruler(proposal), x=CAL_LEFT_X, found_at=CAL_LEFT_X
    )
    after = s.project
    # One change and one log entry for the whole ruler.
    assert len(after.log) == entries + 1
    entry = after.log[-1]
    assert entry.action == "set_ladder_points"
    stored = [(t.y, t.mw) for t in proposal.ticks]
    assert points(c) == [
        (c.marker, y, float(mw), "visible_marker", CAL_LEFT_X, "left") for y, mw in stored
    ]
    assert placed(update) == [(float(mw), "found", False) for mw in CAL_KDA]
    assert update.points == tuple(entry.params["points"])
    assert update.sides_swapped is False and update.point is None
    assert update.curves_changed == (c.blot, c.marker)
    assert entry.params == {
        "membrane_id": c.membrane,
        "group": [c.blot, c.marker],
        "side": "left",
        "image_id": c.marker,
        "x": CAL_LEFT_X,
        "found_at": CAL_LEFT_X,
        "sides_swapped": False,
        "removed": [],
        "points": list(update.points),
        "proposal": ops.proposal_json(proposal),
        "fit": update.fit.as_json(),
        "curves_changed": [c.blot, c.marker],
        "dropped_undetected": [],
    }
    json.dumps(entry.params, allow_nan=False)
    # The apparent MWs follow the new curve.
    assert all(band.apparent_mw is not None for band in protein_of(s, c.protein).bands)
    assert_mw_current(s)
    assert_nets_current(s)
    # One undo takes the whole ruler back; redo applies it again.
    assert ops.undo(s).action == "set_ladder_points"
    assert s.project.batch == before.batch
    ops.redo(s)
    assert s.project.batch == after.batch
    # Applied again with another ruler, it replaces the side's points whole.
    update = ops.set_ladder_points(s, c.marker, LEFT, ruler(proposal)[:3], x=CAL_LEFT_X)
    assert len(points(c, LEFT)) == 3
    assert s.project.log[-1].params["removed"] == [
        {"image_id": c.marker, "y": y, "mw": float(mw), "source": "visible_marker"}
        | {"x": CAL_LEFT_X, "side": "left"}
        for y, mw in stored
    ]
    assert_mw_current(s)


def test_placed_labels_found_snapped_hand(tmp_path):
    c = ruler_ready(tmp_path)
    s = c.session
    proposal = ops.propose_ladder(s, c.marker, CAL_LEFT_X)
    ticks = ruler(proposal)
    # 130 kDa dragged 3 px and snapped back: onto its found band again.
    [(snapped_y, moved)] = ops.snap_ladder(s, c.marker, CAL_LEFT_X, [ticks[1][0] + 3.0])
    assert moved and abs(snapped_y - ticks[1][0]) <= ops.PLACED_TOLERANCE
    # 100 kDa dragged 2 px without a snap; 70 kDa relabelled 60 on its band.
    hand_y = ticks[2][0] + 2.0
    edited = [ticks[0], (snapped_y, 130), (hand_y, 100), (ticks[3][0], 60), *ticks[4:]]
    update = ops.set_ladder_points(s, c.marker, LEFT, edited, x=CAL_LEFT_X, found_at=CAL_LEFT_X)
    assert placed(update)[:5] == [
        (250.0, "found", False),
        (130.0, "found", False),
        (100.0, "hand", False),
        (60.0, "snapped", True),
        (55.0, "found", False),
    ]
    # The client says nothing of how a point was placed. Without found_at no
    # ruler was found: a point that a snap gives back is snapped, the rest are
    # placed by hand.
    update = ops.set_ladder_points(s, c.marker, LEFT, [(snapped_y, 130), (hand_y, 100)], x=30.0)
    assert placed(update) == [(130.0, "snapped", None), (100.0, "hand", None)]
    assert s.project.log[-1].params["found_at"] is None
    assert "proposal" not in s.project.log[-1].params


def test_every_snapped_tick_is_logged_snapped(tmp_path):
    # Each tick of a ruler dragged 1.5 to 3 px off its band and snapped back
    # is logged as snapped, wherever the snap was clicked from: on the
    # marker, and on a faint marker of a chemiluminescence image (bands 5 to
    # 10 noise sigmas deep), whose absolute value moved a snap's parabola with
    # the window it read. A tick placed by hand half a pixel or a few px off
    # its band is logged as placed by hand.
    c = ruler_ready(tmp_path)
    s = c.session
    images = [c.marker]
    for depth in (300.0, 450.0, 600.0):
        bands = [(CAL_LEFT_X, cal_y(mw, CAL_LEFT_X) - 0.5, 10.0, 2.0, depth) for mw in CAL_KDA]
        pixels = noisy16(synthetic_blot((CAL_H, CAL_W), bands, dtype=np.float64), 7)
        images.append(import_blot(s, pixels, f"faint {depth:g}.tif", membrane_id=c.membrane))
    drags = (2.0, -2.5, 3.0, -1.5, 2.5, -3.0, 1.5, -2.0)
    for image in images:
        given = [cal_y(mw, CAL_LEFT_X) + d for mw, d in zip(CAL_KDA, drags, strict=True)]
        snapped = ops.snap_ladder(s, image, CAL_LEFT_X, given)
        assert all(moved for _, moved in snapped), image
        ys = [y for y, _ in snapped]
        update = ops.set_ladder_points(
            s, image, LEFT, list(zip(ys, CAL_KDA, strict=True)), x=CAL_LEFT_X
        )
        assert placed(update) == [(float(mw), "snapped", None) for mw in CAL_KDA], image
        # 100 kDa half a pixel below where it snapped; 35 kDa 4 px below its
        # band, on the membrane between bands.
        by_hand = {100: ys[2] + 0.5, 35: ys[5] + 4.0}
        ruler = [(by_hand.get(mw, y), mw) for y, mw in zip(ys, CAL_KDA, strict=True)]
        update = ops.set_ladder_points(s, image, LEFT, ruler, x=CAL_LEFT_X)
        assert placed(update) == [
            (float(mw), "hand" if mw in by_hand else "snapped", None) for mw in CAL_KDA
        ], image


def test_a_point_snapped_where_it_was_clicked_is_logged_snapped_in_a_ruler(tmp_path):
    # Ladder bands marked by single clicks 2 to 3 px either side of the
    # lane's x, each snapped where it was clicked, then applied unchanged as
    # one ruler at the lane's x (adjusted, then applied): each is logged as
    # snapped. A snap reads the columns about its click; those about the
    # ruler's x put its band a hair elsewhere, so a point keeping the y of one
    # it replaces is judged where that one was snapped, on its image: also in
    # a ruler applied on the other image of the register group, and with its
    # label changed. A point clicked without a snap is placed by hand.
    c = ruler_ready(tmp_path)
    s = c.session
    dxs = (2.0, -2.5, 3.0, -2.0, 2.5, -3.0, 2.0, -2.5)
    drags = (1.5, -2.0, 2.0, -1.5, 1.0, -2.0, 1.5, -1.0)
    kept = 55  # clicked without a snap
    for mw, dx, dy in zip(CAL_KDA, dxs, drags, strict=True):
        x = CAL_LEFT_X + dx
        y = cal_y(mw, x) + dy
        update = ops.add_calibration_point(s, c.marker, y, mw, MARKER_BAND, x=x, snap=mw != kept)
        assert update.point["snapped"] is (mw != kept)
    clicked = [(y, mw) for _, y, mw, _, _, _ in points(c, LEFT)]
    marker = s.pixels(c.marker)
    at_ruler = [
        mwcal.is_snap_position(
            marker, CAL_LEFT_X, y, source=MARKER_BAND, polarity=Polarity.DARK_ON_LIGHT
        )
        for y, _ in clicked
    ]
    assert not any(at_ruler)  # a snap at the ruler's x puts no band where these lie
    expected = [(float(mw), "hand" if mw == kept else "snapped", None) for mw in CAL_KDA]
    for image in (c.marker, c.blot):
        update = ops.set_ladder_points(s, image, LEFT, clicked, x=CAL_LEFT_X)
        assert placed(update) == expected, image
        ops.undo(s)
    relabelled = [(y, 140.0 if mw == 130 else mw) for y, mw in clicked]
    update = ops.set_ladder_points(s, c.marker, LEFT, relabelled, x=CAL_LEFT_X)
    assert placed(update) == [(140.0 if mw == 130 else mw, how, None) for mw, how, _ in expected]


DENSE_KDA = (300, 250, 200, 150, 120, 100, 85, 70, 60, 50, 40, 30, 25, 20, 15, 10)
DENSE_YS = tuple(25.3 + 10.0 * i for i in range(len(DENSE_KDA)))


def dense_ladder(depth: float, rows: float, seed: int) -> np.ndarray:
    """A ladder band at each of :data:`DENSE_YS` (10 px apart) down the left
    lane, ``depth`` counts deep, 20 px wide and ``rows`` px (1/e) high, under
    60 counts of seeded noise, as 16-bit pixels."""
    bands = [(CAL_LEFT_X, y - 0.5, 10.0, rows, depth) for y in DENSE_YS]
    return noisy16(synthetic_blot((CAL_H, CAL_W), bands, dtype=np.float64), seed)


def test_every_tick_snap_all_moves_on_a_dense_ladder_is_logged_snapped(tmp_path):
    # Bands 10 px apart, on a marker and on a faint marker of a
    # chemiluminescence image; a ruler's ticks dragged 1.2 to 2.4 px off
    # them. Snap all moves most ticks onto their bands, each within the gaps
    # to the other ticks, and each tick it moved is logged as snapped: a snap
    # from its y within the gaps to the ticks as applied, or with nothing
    # marked, need not find its band again. The ticks it kept are placed by
    # hand. Then ticks moved by hand from where they snapped, half a pixel, a
    # fiftieth of one, onto their row's centre, 3 px off their bands or
    # halfway to the next band, are logged as placed by hand.
    c = ruler_ready(tmp_path)
    s = c.session
    marker = import_blot(
        s,
        dense_ladder(960.0, 1.5, 1),
        "dense marker µ.tif",
        kind=ImageKind.VISIBLE_MARKER,
        membrane_id=c.membrane,
    )
    faint = import_blot(s, dense_ladder(4000.0, 2.0, 0), "dense α.tif", membrane_id=c.membrane)
    drags = (1.6, -2.0, 2.4, -1.2, 2.0, -2.4, 1.2, -1.6)
    for image, moves in ((marker, 12), (faint, 10)):
        given = [y + drags[k % len(drags)] for k, y in enumerate(DENSE_YS)]
        snapped = ops.snap_ladder(s, image, CAL_LEFT_X, given)
        moved = [k for k, (_, m) in enumerate(snapped) if m]
        assert len(moved) >= moves, (image, moved)
        assert all(abs(snapped[k][0] - DENSE_YS[k]) < 1.0 for k in moved), image
        ys = [y for y, _ in snapped]
        update = ops.set_ladder_points(
            s, image, LEFT, list(zip(ys, DENSE_KDA, strict=True)), x=CAL_LEFT_X
        )
        assert placed(update) == [
            (float(mw), "snapped" if k in moved else "hand", None) for k, mw in enumerate(DENSE_KDA)
        ], image
        half, fiftieth, centre, off, between = [k for k in moved if k < len(DENSE_YS) - 1][:5]
        by_hand = {
            half: ys[half] + 0.5,
            fiftieth: ys[fiftieth] - 0.02,
            centre: math.floor(ys[centre]) + 0.5,
            off: DENSE_YS[off] + 3.0,
            between: 0.5 * (DENSE_YS[between] + DENSE_YS[between + 1]),
        }
        assert all(y != ys[k] for k, y in by_hand.items())
        ruler = [
            (by_hand.get(k, y), mw) for k, (y, mw) in enumerate(zip(ys, DENSE_KDA, strict=True))
        ]
        update = ops.set_ladder_points(s, image, LEFT, ruler, x=CAL_LEFT_X)
        assert placed(update) == [
            (float(mw), "snapped" if k in moved and k not in by_hand else "hand", None)
            for k, mw in enumerate(DENSE_KDA)
        ], image


def test_relabelled_points_are_logged(tmp_path):
    c = ruler_ready(tmp_path)
    s = c.session
    proposal = ops.propose_ladder(s, c.marker, CAL_LEFT_X)
    ticks = ruler(proposal)
    # Every label moved one band down (the ▲▼ of the page): each point keeps
    # its band's y, so it sits on a peak found (snapped), under another label.
    shifted = [(y, mw) for (y, _), (_, mw) in zip(ticks[:-1], ticks[1:], strict=True)]
    update = ops.set_ladder_points(s, c.marker, LEFT, shifted, x=CAL_LEFT_X, found_at=CAL_LEFT_X)
    assert placed(update) == [(float(mw), "snapped", True) for mw in CAL_KDA[1:]]
    assert [p["relabelled"] for p in s.project.log[-1].params["points"]] == [True] * 7
    # One tick relabelled by hand, the rest as found.
    relabelled = [ticks[0], (ticks[1][0], 120.0), *ticks[2:]]
    update = ops.set_ladder_points(s, c.marker, LEFT, relabelled, x=CAL_LEFT_X, found_at=CAL_LEFT_X)
    assert placed(update)[:3] == [
        (250.0, "found", False),
        (120.0, "snapped", True),
        (100.0, "found", False),
    ]


def test_applying_a_doubtful_proposal_is_logged(tmp_path, caplog):
    c = ruler_ready(tmp_path)
    s = c.session
    # The membrane's ladder lacks its top MW: the marker's top band is left
    # over, and the labels may be one band off.
    ops.set_ladder(s, c.membrane, "no 250", kda=list(CAL_KDA[1:]))
    proposal = ops.propose_ladder(s, c.marker, CAL_LEFT_X)
    assert proposal.doubtful and proposal.extra
    with caplog.at_level(logging.INFO, logger="proteia.core.operations"):
        ops.set_ladder_points(s, c.marker, LEFT, ruler(proposal), x=CAL_LEFT_X, found_at=CAL_LEFT_X)
    assert [r.getMessage() for r in caplog.records] == [
        f"in {s.folder.name!r}: the ladder found at x={CAL_LEFT_X:g} on {c.marker} may be one"
        f" band off (gap {proposal.gap:.3g} to the next-best labelling, below"
        f" {mwcal.GAP_WARN:g}); the ruler is applied as given"
    ]
    assert s.project.log[-1].params["proposal"]["doubtful"] is True
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="proteia.core.operations"):
        # The same ruler again changes nothing: nothing committed, nothing logged.
        committed = s.project
        ops.set_ladder_points(s, c.marker, LEFT, ruler(proposal), x=CAL_LEFT_X, found_at=CAL_LEFT_X)
        assert s.project is committed
        # Not found (no found_at), or found with no doubt: no line.
        ops.set_ladder_points(s, c.marker, LEFT, ruler(proposal)[1:], x=CAL_LEFT_X)
        ops.set_ladder(s, c.membrane, "all of it", kda=list(CAL_KDA))
        sure = ops.propose_ladder(s, c.marker, CAL_LEFT_X)
        assert not sure.doubtful
        ops.set_ladder_points(s, c.marker, LEFT, ruler(sure), x=CAL_LEFT_X, found_at=CAL_LEFT_X)
    assert caplog.records == []


def test_second_ladder_left_of_the_first_swaps_sides(tmp_path):
    c = ruler_ready(tmp_path)
    s = c.session
    # The first ladder found is the right-hand one: a lone ladder is the left.
    first = ops.propose_ladder(s, c.marker, CAL_RIGHT_X)
    update = ops.set_ladder_points(s, c.marker, LEFT, ruler(first), x=CAL_RIGHT_X)
    assert [p[4:] for p in points(c)] == [(CAL_RIGHT_X, "left")] * len(CAL_KDA)
    assert update.sides_swapped is False
    before = s.project
    # The second, found as the right one, lies left of it: the sides swap.
    second = ops.propose_ladder(s, c.marker, CAL_LEFT_X, RIGHT)
    assert all(t.found for t in second.ticks) and not second.doubtful
    update = ops.set_ladder_points(
        s, c.marker, RIGHT, ruler(second), x=CAL_LEFT_X, found_at=CAL_LEFT_X
    )
    assert update.sides_swapped is True
    assert {p[4:] for p in points(c, LEFT)} == {(CAL_LEFT_X, "left")}
    assert {p[4:] for p in points(c, RIGHT)} == {(CAL_RIGHT_X, "right")}
    params = s.project.log[-1].params
    assert (params["side"], params["sides_swapped"], params["removed"]) == ("left", True, [])
    assert [p["side"] for p in params["points"]] == ["left"] * len(CAL_KDA)
    # Two ladders: the protein line tilts as the blot does.
    assert [ladder.side for ladder in update.fit.ladders] == ["left", "right"]
    assert math.isclose(update.fit.tilt_deg, math.degrees(math.atan2(6.0, 420.0)), abs_tol=0.01)
    assert_mw_current(s)
    assert ops.undo(s).action == "set_ladder_points"
    assert s.project.batch == before.batch
    ops.redo(s)
    # Found again as the right one, on its own side now: replaced in place.
    update = ops.set_ladder_points(s, c.marker, RIGHT, ruler(first)[1:], x=CAL_RIGHT_X)
    assert update.sides_swapped is False and len(points(c, RIGHT)) == len(CAL_KDA) - 1


def test_a_single_point_on_the_side_given_does_not_stop_the_swap(tmp_path):
    c = ruler_ready(tmp_path)
    s = c.session
    first = ops.propose_ladder(s, c.marker, CAL_RIGHT_X)
    ops.set_ladder_points(s, c.marker, LEFT, ruler(first), x=CAL_RIGHT_X)
    # One point marked as the right ladder's, right of the lone ladder: one
    # point is no ladder, so the lone one is still the only ladder.
    stray_x = CAL_RIGHT_X + 20.0
    stray_y = cal_y(70, stray_x)
    ops.add_calibration_point(
        s, c.marker, stray_y, 70, MARKER_BAND, x=stray_x, side=RIGHT, snap=False
    )
    assert fitted(c, c.marker).ignored == ((RIGHT, "one_point"),)
    before = s.project
    # A second ladder found as the right one, left of the only ladder: it
    # replaces the right side's point, and the sides swap.
    second = ops.propose_ladder(s, c.marker, CAL_LEFT_X, RIGHT)
    update = ops.set_ladder_points(
        s, c.marker, RIGHT, ruler(second), x=CAL_LEFT_X, found_at=CAL_LEFT_X
    )
    assert update.sides_swapped is True
    assert {p[4:] for p in points(c, LEFT)} == {(CAL_LEFT_X, "left")}
    assert {p[4:] for p in points(c, RIGHT)} == {(CAL_RIGHT_X, "right")}
    assert len(points(c)) == 2 * len(CAL_KDA)
    params = s.project.log[-1].params
    assert (params["side"], params["sides_swapped"]) == ("left", True)
    assert params["removed"] == [
        {"image_id": c.marker, "y": stray_y, "mw": 70.0, "source": "visible_marker"}
        | {"x": stray_x, "side": "right"}
    ]
    assert [ladder.side for ladder in update.fit.ladders] == ["left", "right"]
    assert_mw_current(s)
    assert ops.undo(s).action == "set_ladder_points"
    assert s.project.batch == before.batch
    # Two points on it are a ladder: replacing it left of the other ladder is
    # a crossing, refused.
    ops.add_calibration_point(
        s, c.marker, cal_y(35, stray_x), 35, MARKER_BAND, x=stray_x, side=RIGHT, snap=False
    )
    error = refused(
        ErrorCode.LADDER_SIDES,
        ops.set_ladder_points,
        s,
        c.marker,
        RIGHT,
        ruler(second),
        x=CAL_LEFT_X,
    )
    assert "would not lie right of its left ladder" in str(error)


def test_a_lone_ladder_is_the_left_one_whichever_side_is_given(tmp_path):
    c = ruler_ready(tmp_path)
    s = c.session
    left = ruler(ops.propose_ladder(s, c.marker, CAL_LEFT_X))
    right = ruler(ops.propose_ladder(s, c.marker, CAL_RIGHT_X))
    update = ops.set_ladder_points(s, c.marker, RIGHT, left, x=CAL_LEFT_X)
    assert update.sides_swapped is True
    assert {p[4:] for p in points(c)} == {(CAL_LEFT_X, "left")}
    assert s.project.log[-1].params["side"] == "left"
    # A second ladder right of it, given as the right one: as given.
    update = ops.set_ladder_points(s, c.marker, RIGHT, right, x=CAL_RIGHT_X)
    assert update.sides_swapped is False
    assert {p[4:] for p in points(c, RIGHT)} == {(CAL_RIGHT_X, "right")}
    # A lone right ladder (marked point by point as the right one), then a
    # ladder given as the left one but right of it: the sides swap back.
    ops.clear_calibration(s, c.marker)
    for y, mw in left:
        ops.add_calibration_point(
            s, c.marker, y, mw, MARKER_BAND, x=CAL_LEFT_X, side=RIGHT, snap=False
        )
    update = ops.set_ladder_points(s, c.marker, LEFT, right, x=CAL_RIGHT_X)
    assert update.sides_swapped is True
    assert {p[4:] for p in points(c, LEFT)} == {(CAL_LEFT_X, "left")}
    assert {p[4:] for p in points(c, RIGHT)} == {(CAL_RIGHT_X, "right")}
    assert s.project.log[-1].params["side"] == "right"
    assert_mw_current(s)


def test_crossing_ladders_refused(tmp_path):
    c = ruler_ready(tmp_path, sides=(LEFT, RIGHT))
    s = c.session
    right = [(cal_y(mw, CAL_RIGHT_X), mw) for mw in CAL_KDA]
    # Replacing the right ladder with one left of the left ladder, or on it.
    for x in (10.0, CAL_LEFT_X):
        error = refused(
            ErrorCode.LADDER_SIDES, ops.set_ladder_points, s, c.marker, RIGHT, right, x=x
        )
        assert "would not lie right of its left ladder" in str(error)
    # Replacing the left one right of the right ladder.
    refused(ErrorCode.LADDER_SIDES, ops.set_ladder_points, s, c.marker, LEFT, right, x=460.0)
    # A strip edge in the group: a second ladder on either side of it.
    ops.clear_calibration(s, c.marker)
    ops.add_calibration_point(s, c.marker, 5.0, 300, STRIP, x=100.0, snap=False)
    for x in (CAL_RIGHT_X, CAL_LEFT_X):
        error = refused(
            ErrorCode.LADDER_SIDES, ops.set_ladder_points, s, c.marker, RIGHT, right, x=x
        )
        assert "strip edge" in str(error)


def test_set_ladder_points_refusals(tmp_path):
    c = ruler_ready(tmp_path, sides=(LEFT,))
    s = c.session
    good = [(cal_y(mw, CAL_RIGHT_X), mw) for mw in CAL_KDA]

    def apply(session, points=good, *, x=CAL_RIGHT_X, side=RIGHT, **kw):
        return ops.set_ladder_points(session, c.marker, side, points, x=x, **kw)

    with pytest.raises(UnknownIdError):
        ops.set_ladder_points(s, "img-99", RIGHT, good, x=CAL_RIGHT_X)
    refused(ErrorCode.INVALID_INPUT, apply, s, side="middle")
    for x in (math.nan, "450", None, True):
        refused(ErrorCode.INVALID_INPUT, apply, s, x=x)
    refused(ErrorCode.OUT_OF_IMAGE, apply, s, x=CAL_W + 1.0)
    refused(ErrorCode.OUT_OF_IMAGE, apply, s, found_at=-2.0)
    refused(ErrorCode.INVALID_INPUT, apply, s, found_at=math.inf)
    for bad in ("50, 100", [(50.0,)], [50.0], [(50.0, 100, 1)], [("50", 100)]):
        refused(ErrorCode.INVALID_INPUT, apply, s, bad)
    refused(ErrorCode.OUT_OF_IMAGE, apply, s, [*good[:3], (CAL_H + 0.5, 15)])
    for mw in (0, -5, math.inf, "15", None):
        refused(ErrorCode.INVALID_INPUT, apply, s, [*good[:3], (180.0, mw)])
    refused(ErrorCode.DUPLICATE_MW, apply, s, [*good[:3], (180.0, 100)])
    refused(ErrorCode.CALIBRATION_ORDER, apply, s, [*good[:3], (10.0, 15)])
    refused(ErrorCode.CALIBRATION_ORDER, apply, s, [*good[:3], (good[2][0], 15)])
    ops.set_ladder(s, c.membrane, None)
    error = refused(ErrorCode.INVALID_INPUT, apply, s, found_at=CAL_RIGHT_X)
    assert "no ladder MWs" in str(error)
    apply(s)  # as given, it applies
    assert len(points(c, RIGHT)) == len(CAL_KDA)


def test_set_ladder_points_empty_clears_the_side(tmp_path):
    c = ruler_ready(tmp_path, sides=(LEFT, RIGHT))
    s = c.session
    update = ops.set_ladder_points(s, c.blot, RIGHT, [], x=CAL_RIGHT_X)
    assert points(c, RIGHT) == [] and len(points(c, LEFT)) == len(CAL_KDA)
    params = s.project.log[-1].params
    assert (params["points"], len(params["removed"])) == ([], len(CAL_KDA))
    assert [ladder.side for ladder in update.fit.ladders] == ["left"]
    committed = s.project
    ops.set_ladder_points(s, c.blot, RIGHT, [], x=CAL_RIGHT_X)  # nothing there: a no-op
    assert s.project is committed
    assert_mw_current(s)


def test_set_ladder_points_on_a_faint_marker_of_a_chemiluminescence_image(tmp_path):
    bands = [(CAL_LEFT_X, cal_y(kda, CAL_LEFT_X) - 0.5, 10.0, 2.0, 2500.0) for kda in CAL_KDA]
    image = synthetic_blot((CAL_H, CAL_W), bands, dtype=np.float64)
    c = ruler_ready(tmp_path)
    s = c.session
    chemi = import_blot(s, noisy16(image, 7), "faint µ.tif", membrane_id=c.membrane)
    proposal = ops.propose_ladder(s, chemi, CAL_LEFT_X)
    assert [t.mw for t in proposal.ticks if t.found] == list(CAL_KDA)
    update = ops.set_ladder_points(
        s, chemi, LEFT, ruler(proposal), x=CAL_LEFT_X, found_at=CAL_LEFT_X
    )
    assert {p["source"] for p in update.points} == {"chemiluminescence_marker"}
    assert {p["placed"] for p in update.points} == {"found"}
    assert update.curves_changed == (chemi,)  # its own register group


def test_a_proposal_without_another_labelling_logs_no_infinity(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    ops.set_ladder(s, c.membrane, "two bands", kda=[250, 15])
    image = synthetic_blot(
        (CAL_H, CAL_W),
        [(CAL_LEFT_X, cal_y(kda, CAL_LEFT_X) - 0.5, 10.0, 2.0, 20000.0) for kda in (250, 15)],
        dtype=np.float64,
    )
    marker = import_blot(
        s, noisy16(image, 3), "two α.tif", kind=ImageKind.VISIBLE_MARKER, membrane_id=c.membrane
    )
    proposal = ops.propose_ladder(s, marker, CAL_LEFT_X)
    assert proposal.gap == math.inf
    doc = ops.proposal_json(proposal)
    assert (doc["gap"], doc["gap_infinite"], doc["doubtful"]) == (None, True, False)
    ops.set_ladder_points(s, marker, LEFT, ruler(proposal), x=CAL_LEFT_X, found_at=CAL_LEFT_X)
    logged = s.project.log[-1].params["proposal"]
    assert logged == doc
    json.dumps(s.project.log[-1].params, allow_nan=False)
    # Where nothing stands out at found_at, the log says so: no proposal.
    ops.set_ladder_points(
        s, marker, LEFT, ruler(proposal)[:1] + [(150.0, 15)], x=CAL_LEFT_X, found_at=CAL_RIGHT_X
    )
    assert s.project.log[-1].params["proposal"] is None
