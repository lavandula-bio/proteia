# SPDX-License-Identifier: Apache-2.0
"""Tests for the molecular-weight calibration maths (#58): the per-ladder curve,
its range and quality, and the protein line between two ladders.

The literals come from exact synthetic geometry: the sample blot's migration
law without its smile, rotated about the blot's centre, and a vendor-shaped
ladder (PageRuler Plus on a Tris-glycine 4-20 % gel, the vendor's band
positions). Only the snapping tests (:func:`~proteia.core.mwcal.refine_point`)
read pixels: synthetic ladder bands and strips, with seeded noise.
"""

import ast
import math
import statistics
from pathlib import Path

import numpy as np
import pytest

import proteia.core.mwcal as mwcal_module
from conftest import MEMBRANE_LEVEL, make_project, synthetic_blot
from proteia.core import ladders, mwcal
from proteia.core.model import (
    CalibrationPoint,
    CalibrationPointSource,
    FitMethod,
    ImageKind,
    ImageRef,
    LadderSide,
    Membrane,
    MwCalibration,
    Polarity,
    UnknownIdError,
)
from proteia.core.mwcal import (
    DEFAULT_FIT_METHOD,
    EXTRAPOLATE_DECADES,
    FIT_WARN,
    LADDERS_WARN,
    MIN_LADDER_POINTS,
    MIN_SHARED_MWS,
    SNAP_HALF_X,
    SNAP_HALF_Y,
    SNAP_K,
    SNAP_MARKED_GAP,
    SNAP_STRIP_BORDER,
    Calibration,
    NoCalibration,
)

LEFT, RIGHT = LadderSide.LEFT, LadderSide.RIGHT
MARKER = CalibrationPointSource.VISIBLE_MARKER
KDA = (250, 130, 100, 70, 55, 35, 25, 15, 10)  # PageRuler Plus, Tris-glycine
# The vendor's band positions for KDA on a Tris-glycine 4-20 % gel, in px.
VENDOR_YS = (336.5, 373.5, 405.5, 443.0, 485.5, 550.0, 603.5, 672.5, 739.5)
LANES = tuple(205.0 + 128.0 * i for i in range(8))  # the sample blot's lane centres
X_LEFT, X_RIGHT = 77.0, 1229.0  # a ladder lane one pitch outside each end lane
ANGLES = (0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0)


def _close(a: float | None, b: float) -> bool:
    """Equal up to rounding (a y of 0 has no relative tolerance)."""
    return a is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def _image(image_id: str, kind=ImageKind.VISIBLE_MARKER, **fields) -> ImageRef:
    return ImageRef(
        id=image_id,
        file=f"{image_id}.png",
        original_name=f"marker α {image_id}.png",
        kind=kind,
        sha256="0" * 64,
        width=fields.pop("width", 1330),
        height=fields.pop("height", 800),
        polarity=Polarity.DARK_ON_LIGHT,
        background=200.0,
        **fields,
    )


def _point(y: float, mw: float, *, x=None, side=LEFT, source=MARKER, image_id="img-1"):
    return CalibrationPoint(image_id=image_id, y=y, mw=mw, source=source, x=x, side=side)


def _ladder(pairs, *, x=None, side=LEFT, image_id="img-1") -> list[CalibrationPoint]:
    """A ladder of (y, kDa) pairs, every point at ``x``."""
    return [_point(y, mw, x=x, side=side, image_id=image_id) for y, mw in pairs]


def _membrane(*points: CalibrationPoint, width: int = 1330, height: int = 800) -> Membrane:
    """mem-1 with one visible-light marker image, img-1, holding ``points``."""
    return Membrane(
        id="mem-1",
        images=[_image("img-1", width=width, height=height)],
        calibration=MwCalibration(points=list(points)),
    )


def _calibration(*points: CalibrationPoint, **image) -> Calibration:
    fitted = mwcal.calibration_for(_membrane(*points, **image), "img-1")
    assert isinstance(fitted, Calibration)
    return fitted


def _law(mw: float) -> float:
    """The sample blot's migration law without its smile: y of a band of ``mw`` kDa."""
    return 45.0 + 300.0 * math.log10(250.0 / mw)


def _rotate(x: float, y: float, degrees: float) -> tuple[float, float]:
    """(x, y) turned by ``degrees`` about the sample blot's centre (653, 250);
    positive turns the right side down."""
    theta = math.radians(degrees)
    dx, dy = x - 653.0, y - 250.0
    return (
        653.0 + dx * math.cos(theta) - dy * math.sin(theta),
        250.0 + dx * math.sin(theta) + dy * math.cos(theta),
    )


def _rotated(x: float, degrees: float, *, side=LEFT, at=KDA, labels=KDA):
    """A ladder lane at ``x`` of a blot turned by ``degrees``: the bands of ``at``,
    labelled ``labels``, each point at its own turned x and y."""
    points = []
    for mw, label in zip(at, labels, strict=True):
        rx, ry = _rotate(x, _law(mw), degrees)
        points.append(_point(ry, label, x=rx, side=side))
    return points


def _pair(*, scale: float = 1.0, shift: float = 5.0) -> Calibration:
    """Left: every KDA band on the law, at X_LEFT. Right: 130 to 15 kDa at
    X_RIGHT, at ``scale`` times the law's y plus ``shift``."""
    left = _ladder(((_law(m), m) for m in KDA), x=X_LEFT)
    right = _ladder(((scale * _law(m) + shift, m) for m in KDA[1:-1]), x=X_RIGHT, side=RIGHT)
    return _calibration(*left, *right, height=500)


# --- One ladder ---


def test_curve_passes_through_its_points():
    fitted = _calibration(*_ladder(zip(VENDOR_YS, KDA, strict=True)))
    [ladder] = fitted.ladders
    assert (ladder.side, ladder.x, ladder.ys, ladder.mws) == (LEFT, None, VENDOR_YS, KDA)
    assert fitted.method is FitMethod.LOG_LINEAR_PIECEWISE
    for y, m in zip(VENDOR_YS, KDA, strict=True):
        assert _close(fitted.mw_at(None, y), m)
        assert _close(fitted.y_at(m, None), y)
        assert _close(ladder.curve.mw_at(y), m)
        assert _close(ladder.curve.y_at(m), y)


def test_piecewise_between_neighbours():
    fitted = _calibration(*_ladder(((10, 200), (50, 100), (90, 60), (130, 30))))
    curve = fitted.ladders[0].curve
    # Halfway between 200 and 100 kDa in y is halfway in log10(MW).
    assert math.isclose(fitted.mw_at(None, 30.0), math.sqrt(200 * 100), rel_tol=1e-12)
    assert math.isclose(fitted.y_at(math.sqrt(100 * 60), None), 70.0, rel_tol=1e-12)
    top = 40 / math.log10(200 / 100)
    middle = 40 / math.log10(100 / 60)
    bottom = 40 / math.log10(60 / 30)
    for y, slope in [
        (30.0, top),
        (0.0, top),  # above the first point: the first segment
        (10.0, top),
        (50.0, middle),  # a breakpoint takes the segment below it
        (89.9, middle),
        (90.0, bottom),
        (130.0, bottom),  # at and below the last point: the last segment
        (140.0, bottom),
    ]:
        assert math.isclose(curve.px_per_decade(y), slope, rel_tol=1e-12), y
        assert math.isclose(fitted.px_per_decade(None, y), slope, rel_tol=1e-12), y


def test_two_points_equal_the_least_squares_line():
    pairs = ((120.0, 90.0), (180.5, 40.0))
    fitted = _calibration(*_ladder(pairs))
    slope, intercept = statistics.linear_regression(
        [y for y, _ in pairs], [math.log10(m) for _, m in pairs]
    )
    for y in (105.0, 120.0, 133.3, 150.0, 180.5, 195.0):
        assert math.isclose(fitted.mw_at(None, y), 10 ** (slope * y + intercept), rel_tol=1e-12)


def test_a_stored_log_linear_calibration_is_read_piecewise():
    project = make_project()
    mem_1 = project.batch.membranes[0]
    assert mem_1.calibration.fit_method is FitMethod.LOG_LINEAR  # as the fixture stores it
    fitted = mwcal.calibration_for(mem_1, "img-2")
    assert isinstance(fitted, Calibration)
    assert fitted.method is FitMethod.LOG_LINEAR_PIECEWISE
    ys, mws = (20.0, 61.5, 95.25), (250, 100, 55)
    for y, m in zip(ys, mws, strict=True):
        assert _close(fitted.mw_at(None, y), m)  # through all three points
    # One least-squares line would miss the middle point.
    slope, intercept = statistics.linear_regression(ys, [math.log10(m) for m in mws])
    assert abs(10 ** (slope * 61.5 + intercept) / 100 - 1) > 0.03
    # Reading fits nothing into the project: the stored values stay as stored.
    assert mem_1.calibration.fit_quality == 0.998
    assert project.batch.find_band("band-12")[1].apparent_mw == 90.5
    assert project == make_project()


def test_range_extrapolates_a_tenth_of_a_decade_past_ladder_points():
    fitted = _calibration(_point(100, 100), _point(140, 70))
    assert EXTRAPOLATE_DECADES == 0.1
    assert math.isclose(fitted.lo_mw, 55.602976430699705, rel_tol=1e-12)
    assert math.isclose(fitted.hi_mw, 125.89254117941675, rel_tol=1e-12)
    assert math.isclose(fitted.lo_mw, 70 / 10**0.1, rel_tol=1e-12)
    assert math.isclose(fitted.y_at(60, None), 157.2875255127243, rel_tol=1e-12)
    assert fitted.y_at(40, None) is None
    assert fitted.y_at(130, None) is None
    assert fitted.mw_at(None, 0.0) is None
    assert fitted.mw_at(None, 157.0) is not None


def test_no_extrapolation_past_strip_edges():
    mem_5 = make_project().batch.membranes[1]  # strip edges: 100 kDa at y=0, 75 at y=100
    fitted = mwcal.calibration_for(mem_5, "img-6")
    assert isinstance(fitted, Calibration)
    assert (fitted.z_lo, fitted.z_hi) == (math.log10(75), 2.0)
    # The strip's own MWs are inside, although 10**log10(75) may print 75.00000000000001.
    assert fitted.y_at(75, None) == 100
    assert fitted.y_at(100, None) == 0
    assert math.isclose(fitted.lo_mw, 75, rel_tol=1e-12)
    assert fitted.hi_mw == 100
    for y in (-0.1, 100.1):
        assert fitted.mw_at(None, y) is None
    for mw in (74.99, 100.01):
        assert fitted.y_at(mw, None) is None
    assert fitted.strip_edges_only and fitted.two_point
    assert fitted.quality is None


def test_quality_is_the_largest_leave_one_out_disagreement():
    fitted = _calibration(*_ladder(((10, 200), (50, 100), (90, 60), (130, 30))))
    quality = fitted.ladders[0].quality
    assert quality == fitted.quality
    assert math.isclose(quality.value, 0.09545, rel_tol=1e-4)
    assert quality.mw == 60
    assert math.isclose(quality.predicted_mw, 54.77, rel_tol=1e-4)


def test_vendor_shaped_ladder_quality_below_warn():
    quality = _calibration(*_ladder(zip(VENDOR_YS, KDA, strict=True))).quality
    assert math.isclose(quality.value, 0.1501, rel_tol=1e-3)
    assert quality.mw == 130
    assert math.isclose(quality.predicted_mw, 152.95, rel_tol=1e-4)
    assert quality.value < FIT_WARN == 0.2


def test_skipped_band_quality_above_warn():
    # The 55-kDa band was not marked, so every label below it moved up one band.
    ys = [y for y, m in zip(VENDOR_YS, KDA, strict=True) if m != 55]
    quality = _calibration(*_ladder(zip(ys, KDA[:-1], strict=True))).quality
    assert math.isclose(quality.value, 0.2472, rel_tol=1e-3)
    assert quality.mw == 55
    assert math.isclose(quality.predicted_mw, 44.10, rel_tol=1e-3)
    assert quality.value > FIT_WARN


def test_two_points_have_no_quality():
    fitted = _calibration(_point(100, 100), _point(140, 70))
    assert fitted.ladders[0].quality is None
    assert fitted.quality is None
    assert fitted.two_point
    assert not _calibration(*_ladder(zip(VENDOR_YS, KDA, strict=True))).two_point


def test_no_calibration_reasons():
    assert mwcal.calibration_for(_membrane(), "img-1") == NoCalibration(
        frozenset({"img-1"}), "no_points"
    )
    assert mwcal.calibration_for(_membrane(_point(100, 100)), "img-1") == NoCalibration(
        frozenset({"img-1"}), "one_point"
    )
    one_each = _membrane(_point(100, 100, x=10.0), _point(110, 100, x=900.0, side=RIGHT))
    assert mwcal.calibration_for(one_each, "img-1") == NoCalibration(
        frozenset({"img-1"}), "one_point"
    )
    # Other groups' points are not used: img-4 of mem-1 is not linked to the marker.
    mem_1 = make_project().batch.membranes[0]
    assert mwcal.calibration_for(mem_1, "img-4") == NoCalibration(frozenset({"img-4"}), "no_points")
    with pytest.raises(UnknownIdError):
        mwcal.calibration_for(mem_1, "img-6")


def test_register_group_decides_the_points():
    mem_1 = make_project().batch.membranes[0]
    fitted = mwcal.calibration_for(mem_1, "img-2")
    assert isinstance(fitted, Calibration)
    assert fitted == mwcal.calibration_for(mem_1, "img-3")
    assert fitted.group == frozenset({"img-2", "img-3"})
    [ladder] = fitted.ladders
    assert (ladder.ys, ladder.mws) == ((20.0, 61.5, 95.25), (250, 100, 55))
    assert ladder.sources == (MARKER, MARKER, CalibrationPointSource.CHEMILUMINESCENCE_MARKER)
    assert mwcal.calibrations(mem_1) == {
        frozenset({"img-2", "img-3"}): fitted,
        frozenset({"img-4"}): NoCalibration(frozenset({"img-4"}), "no_points"),
    }


def test_one_ladder_is_independent_of_x():
    fitted = _calibration(*_ladder(zip(VENDOR_YS, KDA, strict=True)))
    assert fitted.curve_at(None) is fitted.curve_at(0.0) is fitted.ladders[0].curve
    for y in (340.0, 500.0, 700.0):
        found = {fitted.mw_at(x, y) for x in (None, 0.0, 1000.0)}
        assert len(found) == 1 and None not in found
    for side_x in (None, 0.0, 1000.0):
        assert fitted.tilt_deg is None and fitted.offset_px is None
        assert fitted.disagreement is None and fitted.shared_mws == ()
        assert fitted.y_at(100, side_x) == fitted.y_at(100, None)


# --- Which ladders are used ---


def test_a_side_with_one_point_is_ignored():
    ladder = [(_law(m), m) for m in KDA]
    fitted = _calibration(
        *_ladder(ladder, x=X_LEFT), _point(100, 130, x=X_RIGHT, side=RIGHT), height=500
    )
    assert [one.side for one in fitted.ladders] == [LEFT]
    assert fitted.ignored == ((RIGHT, "one_point"),)
    assert not fitted.two_ladders and fitted.tilt_deg is None
    # The same with the sides swapped: the right ladder alone is the calibration.
    swapped = _calibration(
        _point(100, 130, x=X_LEFT), *_ladder(ladder, x=X_RIGHT, side=RIGHT), height=500
    )
    assert [one.side for one in swapped.ladders] == [RIGHT]
    assert swapped.ignored == ((LEFT, "one_point"),)
    assert swapped.mw_at(0.0, 45.0) == swapped.mw_at(None, 45.0)
    assert _close(swapped.mw_at(None, 45.0), 250)


def test_ladders_sharing_fewer_than_two_mws_are_not_combined():
    left = [_point(y, m, x=X_LEFT) for y, m in zip(VENDOR_YS, KDA, strict=True) if m >= 100]
    alone = _calibration(*left)
    assert MIN_SHARED_MWS == 2
    # Disjoint (250-100 and 55-10 kDa: the ranges do not meet), a thin overlap
    # holding no ladder MW (70-55 kDa), and one MW in common.
    for kept in ((55, 35, 25, 15, 10), (70, 55), (100, 70, 55)):
        right = [
            _point(y, m, x=X_RIGHT, side=RIGHT)
            for y, m in zip(VENDOR_YS, KDA, strict=True)
            if m in kept
        ]
        fitted = _calibration(*left, *right)
        assert fitted.ladders == alone.ladders, kept  # the left ladder alone
        assert fitted.ignored == ((RIGHT, "few_shared_mws"),), kept
        assert (fitted.z_lo, fitted.z_hi) == (alone.z_lo, alone.z_hi)
        assert fitted.shared_mws == () and fitted.tilt_deg is None
        assert fitted.disagreement is None and fitted.offset_px is None
        assert fitted.mw_at(None, 380.0) == alone.mw_at(None, 380.0)


# --- Two ladders: the protein line ---


def test_protein_line_passes_through_both_ladders():
    fitted = _pair(scale=1.1, shift=-20.0)
    left, right = fitted.ladders
    assert (left.side, left.x, right.side, right.x) == (LEFT, X_LEFT, RIGHT, X_RIGHT)
    for m in (163.0, 130, 92, 50, 25, 15, 12):
        assert _close(fitted.y_at(m, X_LEFT), left.curve.y_at(m))
        assert _close(fitted.y_at(m, X_RIGHT), right.curve.y_at(m))
    for y, m in zip(right.ys, right.mws, strict=True):
        assert _close(fitted.y_at(m, X_RIGHT), y)
        assert _close(fitted.mw_at(X_RIGHT, y), m)


def test_protein_line_is_linear_in_x():
    fitted = _pair(scale=1.1, shift=-20.0)
    left, right = fitted.ladders
    x = X_LEFT + 0.25 * (X_RIGHT - X_LEFT)
    for m in (163.0, 130, 92, 50, 25, 15, 12):
        expected = 0.75 * left.curve.y_at(m) + 0.25 * right.curve.y_at(m)
        assert _close(fitted.y_at(m, x), expected), m


def test_curve_at_x_breaks_at_the_union_within_the_range():
    fitted = _pair()  # the right ladder 5 px lower, holding 130 to 15 kDa
    assert fitted.two_ladders
    lo, hi = 15 / 10**0.1, 130 * 10**0.1  # the right ladder's range, inside the left's
    assert math.isclose(fitted.lo_mw, lo, rel_tol=1e-12) and round(fitted.lo_mw, 3) == 11.915
    assert math.isclose(fitted.hi_mw, hi, rel_tol=1e-12) and round(fitted.hi_mw, 2) == 163.66
    assert (fitted.z_lo, fitted.z_hi) == (
        fitted.ladders[1].curve.z_lo,
        fitted.ladders[1].curve.z_hi,
    )
    curve = fitted.curve_at(653.0)
    expected = (hi, 130, 100, 70, 55, 35, 25, 15, lo)
    assert len(curve.zs) == len(curve.ys) == len(expected)
    for z, mw in zip(curve.zs, expected, strict=True):
        assert math.isclose(10**z, mw, rel_tol=1e-12)
    assert (curve.z_lo, curve.z_hi) == (fitted.z_lo, fitted.z_hi)
    # A MW marked on the left only is not extrapolated to the right.
    assert fitted.y_at(250, 653.0) is None and fitted.y_at(10, X_LEFT) is None
    assert fitted.ladders[0].curve.y_at(250) is not None


def test_curve_at_x_inverts_exactly():
    fitted = _pair(scale=1.1, shift=-20.0)
    mws = sorted({kda for preset in ladders.PRESETS for kda in preset.kda})
    inside = [m for m in mws if fitted.z_lo <= math.log10(m) <= fitted.z_hi]
    assert len(inside) >= 15
    for x in (0.0, X_LEFT, 653.0, X_RIGHT, 1330.0):
        for m in mws:
            y = fitted.y_at(m, x)
            if m not in inside:
                assert y is None, (x, m)
                continue
            assert math.isclose(fitted.mw_at(x, y), m, rel_tol=1e-12), (x, m)


def test_ladder_x_is_the_median_of_its_points():
    fitted = _calibration(
        _point(100, 100, x=70.0), _point(140, 70, x=84.0), _point(170, 50, x=77.0)
    )
    assert fitted.ladders[0].x == 77.0
    # Files written before #58 hold no x.
    assert _calibration(_point(100, 100), _point(140, 70)).ladders[0].x is None


@pytest.mark.parametrize("degrees", ANGLES)
def test_two_ladders_follow_a_rotated_blot(degrees):
    # The right ladder reaches x = 1236 at 2 degrees: wider than the 1200-px sample.
    left, right = _rotated(X_LEFT, degrees), _rotated(X_RIGHT, degrees, side=RIGHT)
    fitted = _calibration(*left, *right, width=1330, height=500)
    assert fitted.two_ladders and fitted.ignored == ()
    assert fitted.shared_mws == KDA
    assert math.isclose(fitted.tilt_deg, degrees, abs_tol=1e-9)
    offset = (X_RIGHT - X_LEFT) * math.sin(math.radians(degrees))
    assert math.isclose(fitted.offset_px, offset, rel_tol=1e-9, abs_tol=1e-9)
    assert fitted.disagreement.value < 1e-9
    for m in KDA:
        for x in LANES:
            bx, by = _rotate(x, _law(m), degrees)
            y = fitted.y_at(m, bx)
            assert y is not None and abs(y - by) < 0.3, (m, x)  # at most 0.271 px
            found = fitted.mw_at(bx, by)
            assert found is not None and abs(found / m - 1) < 0.0025, (m, x)  # 0.208 %


def test_one_ladder_misses_far_lanes_on_a_rotated_blot():
    fitted = _calibration(*_rotated(X_LEFT, 1.0), width=1330, height=500)
    assert not fitted.two_ladders

    def worst(x: float) -> float:
        errors = []
        for m in KDA:
            bx, by = _rotate(x, _law(m), 1.0)
            found = fitted.mw_at(bx, by)
            assert found is not None
            errors.append(abs(found / m - 1))
        return max(errors)

    assert worst(LANES[-1]) > 0.10  # 12.8 %
    assert worst(LANES[0]) < worst(LANES[-1])


def test_disagreement_catches_a_right_ladder_one_band_off():
    left = _rotated(X_LEFT, 0.5)
    right = _rotated(X_RIGHT, 0.5, side=RIGHT)
    assert _calibration(*left, *right, height=500).disagreement.value < 1e-6
    # The right ladder's labels one band up: its 130-kDa band called 250, and so on.
    off = _rotated(X_RIGHT, 0.5, side=RIGHT, at=KDA[1:], labels=KDA[:-1])
    disagreement = _calibration(*left, *off, height=500).disagreement
    assert disagreement.value > 0.4 > LADDERS_WARN
    assert disagreement.mw == 250


def test_disagreement_ignores_a_band_marked_on_one_side():
    # Tilt only: the right ladder 10 px lower, without its 130-kDa band. At the
    # union of both ladders' MWs the missing band would read 16 % off.
    left = _ladder(zip(VENDOR_YS, KDA, strict=True), x=X_LEFT)
    right = [
        _point(y + 10.0, m, x=X_RIGHT, side=RIGHT)
        for y, m in zip(VENDOR_YS, KDA, strict=True)
        if m != 130
    ]
    fitted = _calibration(*left, *right)
    assert fitted.shared_mws == tuple(m for m in KDA if m != 130)
    assert math.isclose(fitted.offset_px, 10.0, rel_tol=1e-12)
    assert fitted.disagreement.value < 1e-9


def test_a_quality_past_what_a_float_holds_is_infinite():
    # MWs the model takes (finite, above 0) a pixel apart across hundreds of
    # decades: the relative error at the middle point is past the largest float.
    ys = (0.0, 999.9, 1000.0)
    mws = (1e308, 1e-10, 1e-320)
    worst = mwcal_module._quality(ys, tuple(math.log10(m) for m in mws), mws)
    assert worst is not None and worst.value == math.inf and worst.mw == 1e-10


def test_disagreement_past_what_a_float_holds_is_infinite():
    # 25 kDa one pixel under 250 on both sides, and the right ladder's top 400 px
    # low: midway, one decade spans a pixel there, so the error is 10**400 - 1,
    # past the largest float. The ladders are flagged, not an OverflowError.
    left = _ladder(((100.0, 250), (101.0, 25), (700.0, 15), (800.0, 10), (900.0, 5)), x=10.0)
    right = _ladder(
        ((500.0, 250), (501.0, 25), (700.0, 15), (800.0, 10), (900.0, 5)), x=300.0, side=RIGHT
    )
    fitted = _calibration(*left, *right, height=1000)
    assert fitted.two_ladders and fitted.offset_px == 0.0
    disagreement = fitted.disagreement
    assert disagreement.value == math.inf > LADDERS_WARN
    assert disagreement.mw == 250


def test_line_that_folds_outside_the_ladders_is_out_of_range():
    # The right ladder is ten times as compressed as the left: extrapolated to
    # t = 3, the line would put 50 kDa above 100 kDa.
    fitted = _calibration(
        _point(100, 100, x=0.0),
        _point(200, 50, x=0.0),
        _point(100, 100, x=100.0, side=RIGHT),
        _point(110, 50, x=100.0, side=RIGHT),
    )
    assert fitted.curve_at(50.0) is not None
    assert fitted.curve_at(300.0) is None
    assert fitted.mw_at(300.0, 100.0) is None
    assert fitted.y_at(75, 300.0) is None
    assert fitted.px_per_decade(300.0, 100.0) is None


def test_two_ladders_need_x():
    fitted = _pair()
    for call in (
        lambda: fitted.curve_at(None),
        lambda: fitted.mw_at(None, 200.0),
        lambda: fitted.y_at(100, None),
        lambda: fitted.px_per_decade(None, 200.0),
    ):
        with pytest.raises(ValueError, match="x"):
            call()


# --- The membrane ---


def test_fit_quality_is_the_worst_ladder():
    marker, signal, other = (
        _image("img-1"),
        _image("img-2", ImageKind.CHEMILUMINESCENCE, marker_image_id="img-1"),
        _image("img-3"),
    )
    points = [
        # img-1 and img-2 (one group): two ladders; the right one is the worse.
        *_ladder(((10, 200), (50, 100), (90, 60), (130, 30)), x=50.0),
        *_ladder(((10, 200), (50, 100), (100, 60), (130, 30)), x=300.0, side=RIGHT),
        # img-3 (its own group): the vendor-shaped ladder.
        *_ladder(zip(VENDOR_YS, KDA, strict=True), image_id="img-3"),
    ]
    membrane = Membrane(
        id="mem-1", images=[marker, signal, other], calibration=MwCalibration(points=points)
    )
    linked = mwcal.calibration_for(membrane, "img-2")
    left, right = linked.ladders
    assert right.quality.value > left.quality.value
    assert linked.quality == right.quality
    alone = mwcal.calibration_for(membrane, "img-3")
    assert alone.group == frozenset({"img-3"})
    assert right.quality.value > alone.quality.value > left.quality.value
    assert mwcal.fit_quality(membrane) == right.quality.value
    # No ladder with three points: nothing to measure.
    assert mwcal.fit_quality(_membrane(_point(100, 100), _point(140, 70))) is None
    assert mwcal.fit_quality(_membrane()) is None


def test_fit_method_default_with_a_curve():
    assert DEFAULT_FIT_METHOD is FitMethod.LOG_LINEAR_PIECEWISE
    assert mwcal.fit_method(_membrane()) is FitMethod.LOG_LINEAR
    assert mwcal.fit_method(_membrane(_point(100, 100))) is FitMethod.LOG_LINEAR
    assert mwcal.fit_method(_membrane(_point(100, 100), _point(140, 70))) is DEFAULT_FIT_METHOD
    project = make_project()
    assert [mwcal.fit_method(m) for m in project.batch.membranes] == [DEFAULT_FIT_METHOD] * 2


def test_settings_are_the_constants():
    assert (MIN_LADDER_POINTS, MIN_SHARED_MWS, LADDERS_WARN) == (2, 2, 0.05)
    assert mwcal.settings() == {
        "method": "log_linear_piecewise",
        "per_ladder": "log10(MW) linear in y between neighbouring points; the end segments extend",
        "two_ladders": (
            "at each MW, y linear in x between the two ladders, each at the median x of its points"
        ),
        "extrapolate_decades": EXTRAPOLATE_DECADES,
        "past_strip_edge": 0,
        "two_ladder_range": "the intersection of the two ladders' ranges",
        "quality": (
            "per ladder, the largest relative MW disagreement of an interior point with the"
            " line between its neighbours (leave-one-out); none with two points"
        ),
        "poor_fit_warn": FIT_WARN,
        "ladders_disagree_warn": LADDERS_WARN,
        "min_ladder_points": MIN_LADDER_POINTS,
        "min_shared_mws": MIN_SHARED_MWS,
        "ladder_offsets": "at the MWs both ladders hold",
        "snap": {
            "half_x": SNAP_HALF_X,
            "half_y": SNAP_HALF_Y,
            "marked_gap": SNAP_MARKED_GAP,
            "k": SNAP_K,
            "signal": (
                "deviation from the window's median, by the image's polarity on a marker"
                " image, its absolute value for a faint marker on a chemiluminescence image;"
                " the mean along each row, less its median; noise the robust sigma of the"
                " row-to-row steps over the square root of two"
            ),
            "ladder_band": (
                "the local maximum nearest the click, refined by a parabola (at most half a"
                " row), at row + 0.5"
            ),
            "strip_edge": "the same on the size of the row-to-row steps, at the edge between rows",
            "strip_edge_noise": "the robust sigma of the row-to-row steps",
            "strip_edge_border": SNAP_STRIP_BORDER,
            "no_band": "the clicked y is kept",
        },
    }
    assert (SNAP_HALF_X, SNAP_HALF_Y, SNAP_MARKED_GAP, SNAP_K) == (12, 12, 0.45, 4.0)
    assert SNAP_STRIP_BORDER == 1.0


def test_mwcal_imports_only_the_model():
    tree = ast.parse(Path(mwcal_module.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module)
    assert {m for m in modules if m.split(".")[0] == "proteia"} == {"proteia.core.model"}
    # numpy for the pixels a click is snapped on (refine_point), nothing more.
    assert not {m for m in modules if m.split(".")[0] in {"scipy", "skimage"}}


# --- Snapping a click (refine_point) ---

SNAP_W, SNAP_H = 60, 200
SNAP_X = 30.0
DARK, LIGHT = Polarity.DARK_ON_LIGHT, Polarity.LIGHT_ON_DARK
CHEMI_MARKER = CalibrationPointSource.CHEMILUMINESCENCE_MARKER
STRIP = CalibrationPointSource.STRIP_EDGE


def _bands(*bands: tuple[float, float]) -> np.ndarray:
    """Dark bands ``(y, depth)`` wide across the window, centred at the
    continuous ``y`` (row r covers [r, r+1), so a band on row r's centre lies
    at r + 0.5), on a flat membrane; float, without noise."""
    spots = [(SNAP_X, y - 0.5, 40.0, 2.5, depth) for y, depth in bands]
    return synthetic_blot((SNAP_H, SNAP_W), spots, dtype=np.float64)


def _noisy(array: np.ndarray, seed: int = 58) -> np.ndarray:
    return array + np.random.default_rng(seed).normal(0.0, 60.0, array.shape)


def _ladder_pixels(ys, *, depth: float = 12000.0) -> np.ndarray:
    return _noisy(_bands(*((y, depth) for y in ys)))


def _snap(array, y, *, source=MARKER, polarity=DARK, marked=()):
    return mwcal.refine_point(array, SNAP_X, y, source=source, polarity=polarity, marked_ys=marked)


def test_snap_lands_on_the_band_centre_in_continuous_coordinates():
    truth = [40.5, 71.3, 103.8, 150.0]
    array = _ladder_pixels(truth)
    for y in truth:
        for off in (-8.0, -3.0, 0.0, 3.0, 8.0):
            got = _snap(array, y + off)
            assert got is not None and abs(got - y) < 0.2, (y, off, got)
    # Without noise, a band on row 70's centre lies at 70.5, not at the row index.
    assert _close(_snap(_bands((70.5, 12000.0)), 66.0), 70.5)


def test_snap_prefers_the_nearest_band_not_the_strongest():
    faint, strong = 60.5, 76.5
    array = _noisy(_bands((faint, 6000.0), (strong, 30000.0)), seed=3)
    assert abs(_snap(array, 65.0) - faint) < 0.2  # 4.5 px from it, 11.5 from the strong one
    assert abs(_snap(array, 72.0) - strong) < 0.2
    # Two bands as near: the higher one.
    assert _close(_snap(_bands((60.5, 12000.0), (76.5, 12000.0)), 68.5), 60.5)


def test_snap_never_reaches_a_marked_band():
    array = _ladder_pixels([50.5, 76.5])
    assert abs(_snap(array, 70.0) - 76.5) < 0.2  # the nearer band, unmarked
    # Marked already, the band at 76.5 is out of reach: the window stops 0.45 of
    # the way to it, and the band at 50.5 lies past its other end.
    assert _snap(array, 70.0, marked=[76.5]) is None
    assert abs(_snap(array, 60.0, marked=[76.5]) - 50.5) < 0.2
    assert _snap(array, 76.5, marked=[76.5]) is None  # on a marked point: no window at all


def test_snap_keeps_the_click_where_nothing_stands_out():
    flat = _noisy(np.full((SNAP_H, SNAP_W), MEMBRANE_LEVEL), seed=1)
    for source in CalibrationPointSource:  # an absolute value's mean is no band either
        assert _snap(flat, 100.0, source=source) is None, source
    # A band too faint for SNAP_K noise sigmas, and one past the window.
    assert _snap(_ladder_pixels([100.5], depth=40.0), 100.0) is None
    assert _snap(_ladder_pixels([100.5]), 100.0 + SNAP_HALF_Y + 3.0) is None
    # A window cut to under three rows by the image's edge.
    assert _snap(_ladder_pixels([1.0]), 0.2) is None


def test_snap_follows_the_polarity_and_takes_a_faint_marker_either_way():
    dark = _ladder_pixels([80.5])
    light = 2 * MEMBRANE_LEVEL - dark  # the same band, bright on a dark membrane
    assert abs(_snap(dark, 84.0) - 80.5) < 0.2
    assert abs(_snap(light, 84.0, polarity=LIGHT) - 80.5) < 0.2
    # Read the wrong way round, a band is a trough: no peak reaches the noise.
    assert _snap(light, 84.0, polarity=DARK) is None
    # A faint marker on a chemiluminescence image may show either way.
    for array in (dark, light):
        for polarity in (DARK, LIGHT):
            got = _snap(array, 84.0, source=CHEMI_MARKER, polarity=polarity)
            assert got is not None and abs(got - 80.5) < 0.2


def test_snap_finds_a_strip_edge_between_rows():
    array = np.full((SNAP_H, SNAP_W), 2000.0)
    array[:120] = MEMBRANE_LEVEL  # the strip ends between rows 119 and 120: y = 120
    array = _noisy(array, seed=5)
    for click in (118.0, 122.0, 110.0):
        got = _snap(array, click, source=STRIP)
        assert got is not None and abs(got - 120.0) < 0.5, (click, got)
    assert _snap(array, 60.0, source=STRIP) is None  # no edge within reach


def test_snap_keeps_a_strip_edge_clicked_on_the_image_border():
    # A strip cropped at its cut: the edge is the image's border, past which no
    # row lies to step from, so a step found from a click there is noise or a
    # band's flank. Here half a marker band the cut runs through at the top,
    # and a band ten rows above the bottom.
    array = _noisy(_bands((0.5, 12000.0), (SNAP_H - 10.5, 12000.0)), seed=7)
    for click in (0.0, 0.5, 1.0, SNAP_H - 1.0, SNAP_H - 0.5, float(SNAP_H)):
        assert _snap(array, click, source=STRIP) is None, click
    # Only a strip edge: a ladder band by the border snaps from a click on it.
    near = _ladder_pixels([2.5, SNAP_H - 2.5])
    assert abs(_snap(near, 0.0) - 2.5) < 0.2 and abs(_snap(near, SNAP_H) - (SNAP_H - 2.5)) < 0.2
    # An image not cropped at the cut: the edge three rows inside snaps from a
    # click more than a pixel from the border; clicked on the border, it is kept.
    inner = np.full((SNAP_H, SNAP_W), 2000.0)
    inner[3:] = MEMBRANE_LEVEL  # the strip begins between rows 2 and 3: y = 3
    inner = _noisy(inner, seed=5)
    assert abs(_snap(inner, 2.0, source=STRIP) - 3.0) < 0.5
    assert _snap(inner, 1.0, source=STRIP) is None
