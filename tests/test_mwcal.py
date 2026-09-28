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
from proteia import samples
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
    EXTRA_COST,
    EXTRAPOLATE_DECADES,
    FIND_BACKGROUND,
    FIND_BUDGET,
    FIND_HALF_X,
    FIT_WARN,
    GAP_WARN,
    LADDERS_WARN,
    MIN_LADDER_POINTS,
    MIN_SHARED_MWS,
    MISS_COST,
    PRIOR_DECADES,
    REF_COST,
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
                "the local maximum nearest the click (a flat top of equal rows is one), placed"
                " on the rows' own means across the lane, turned by the polarity (for a faint"
                " marker, the way the band shows against the window's median), at their local"
                " maximum reached from it: refined by a parabola (at most half a row), or the"
                " middle of a flat top, at row + 0.5 (one y per band, wherever it is clicked"
                " from)"
            ),
            "strip_edge": "the same on the size of the row-to-row steps, at the edge between rows",
            "strip_edge_noise": "the robust sigma of the row-to-row steps",
            "strip_edge_border": SNAP_STRIP_BORDER,
            "no_band": "the clicked y is kept",
        },
        "find_ladder": {
            "half_x": FIND_HALF_X,
            "k": SNAP_K,
            "background": FIND_BACKGROUND,
            "peaks": (
                "local maxima of the lane's profile (the signal as for a snap, down the whole"
                " lane) standing k noise sigmas above the membrane around them (the profile's"
                " running median over a background share of the rows, at least the snap's"
                " half_y each way) and above the profile around them (their prominence),"
                " refined by a parabola, or at the middle of a flat top of equal rows, at"
                " row + 0.5; a peak's strength is its height above the membrane in noise"
                " sigmas"
            ),
            "noise": (
                "the snap's, but at least the robust sigma of the differences between"
                " neighbouring pixels along each row (each less its column's median), taken"
                " between pixels off the window's lowest and highest values, over the square"
                " root of twice the window's columns, and the grey levels' spacing over the"
                " square root of twelve times the window's columns"
            ),
            "labelling": "peaks and ladder MWs matched in order; either may stay unmatched",
            "prior_decades": PRIOR_DECADES,
            "score": (
                "the squared residuals of log10(MW) from the labelling's least-squares"
                " quadratic in y, which must fall over its peaks; for a second ladder, plus the"
                " weighted squared spread of its height differences from the first ladder, each"
                " over the first ladder's pixels per decade; both over prior_decades squared"
            ),
            "miss_cost": MISS_COST,
            "extra_cost": EXTRA_COST,
            "reference_cost": REF_COST,
            "reference": (
                "per reference MW not carried by one of the strongest labelled peaks, as many as"
                " there are reference MWs; colour is not used"
            ),
            "predicted": (
                "each ladder MW left unmatched, from the winning quadratic on its falling side;"
                " not drawn off the image or out of order with the found ticks"
            ),
            "gap": "how much worse the best labelling that labels some peak otherwise scores",
            "gap_warn": GAP_WARN,
            "budget": FIND_BUDGET,
            "budget_reached": "the best labelling found is kept, with a gap of 0 (doubtful)",
        },
    }
    assert (SNAP_HALF_X, SNAP_HALF_Y, SNAP_MARKED_GAP, SNAP_K) == (12, 12, 0.45, 4.0)
    assert SNAP_STRIP_BORDER == 1.0
    assert mwcal.SNAP_POSITION_TOLERANCE == 1e-9


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


def _faint_lane(sign: float, seed: int) -> tuple[np.ndarray, list[float]]:
    """Eight bands 250 counts deep (dark; bright where ``sign`` is negative),
    20 px wide and a few rows high, under 60 counts of seeded noise, as 16-bit
    pixels: a faint marker, whose bands reach the lane's edge columns under
    the noise. With the bands' ys."""
    ys = [30.5 + 20.3 * i for i in range(8)]
    spots = [(SNAP_X, y - 0.5, 10.0, 2.0, sign * 250.0) for y in ys]
    noisy = _noisy(synthetic_blot((SNAP_H, SNAP_W), spots, dtype=np.float64), seed)
    return np.clip(np.round(noisy), 0, 65535).astype(np.uint16), ys


FAINT_LANES = {
    "marker": (MARKER, DARK, 1.0),
    "chemiluminescence, dark": (CHEMI_MARKER, DARK, 1.0),
    "chemiluminescence, bright": (CHEMI_MARKER, LIGHT, -1.0),
}


@pytest.mark.parametrize("case", FAINT_LANES)
def test_snap_lands_on_one_y_per_band_wherever_it_is_clicked_from(case):
    # Where a snap lands on a band depends on the band's rows alone, not on
    # the window the click and the points marked cut: every click that takes
    # a band lands on the same y, and a snap from there with nothing marked
    # lands there again. (For a faint marker, an absolute value moved the
    # parabola with the window's median.) From half a pixel or 4 px off the
    # band, a snap lands elsewhere.
    source, polarity, sign = FAINT_LANES[case]
    for seed in range(3):
        array, ys = _faint_lane(sign, seed)
        for i, y in enumerate(ys):
            got = set()
            for click in (y - 3.0, y - 1.5, y + 0.25, y + 1.5, y + 3.0):
                for marked in ((), ys[:i] + ys[i + 1 :], (y - 12.0, y + 9.0), (y + 7.0,)):
                    snapped = _snap(array, click, source=source, polarity=polarity, marked=marked)
                    if snapped is not None and abs(snapped - y) < 1.0:
                        got.add(snapped)
            assert len(got) == 1, (seed, y, got)
            [band] = got
            assert _close(_snap(array, band, source=source, polarity=polarity), band)
            for off in (0.5, 4.0):
                again = _snap(array, band + off, source=source, polarity=polarity)
                assert again is None or abs(again - band - off) > 0.01, (seed, y, off)


def _is_snap(array, y, *, source=MARKER, polarity=DARK):
    return mwcal.is_snap_position(array, SNAP_X, y, source=source, polarity=polarity)


def test_every_y_a_snap_lands_on_is_a_snap_position():
    # From clicks all down the image, with points marked around them or not,
    # each ladder band, faint marker and strip edge a snap lands on lies where
    # the climb from its own row, on the rows' means across the lane, ends
    # (is_snap_position), whatever window the click and the points marked
    # gave: so a y a snap gave can be told from one placed by hand. Half a
    # pixel, or a fiftieth of one, off it is no such y.
    cases = [
        (_ladder_pixels([40.5, 71.3, 103.8, 150.0]), MARKER, DARK),
        (_noisy(_bands((60.5, 6000.0), (76.5, 30000.0)), seed=3), MARKER, DARK),
        (2 * MEMBRANE_LEVEL - _ladder_pixels([80.5]), CHEMI_MARKER, DARK),
        (_ladder_pixels([80.5]), CHEMI_MARKER, LIGHT),
        (_noisy(np.full((SNAP_H, SNAP_W), MEMBRANE_LEVEL), seed=1), MARKER, DARK),
        (_faint_lane(1.0, 5)[0], CHEMI_MARKER, DARK),
        (_faint_lane(-1.0, 6)[0], MARKER, LIGHT),
    ]
    strip = np.full((SNAP_H, SNAP_W), 2000.0)
    strip[:120] = MEMBRANE_LEVEL
    cases.append((_noisy(strip, seed=5), STRIP, DARK))
    for array, source, polarity in cases:
        for click in np.arange(2.0, SNAP_H - 2.0, 0.7):
            for marked in ((), (click - 6.0, click + 4.0), (click + 3.0,)):
                got = _snap(array, click, source=source, polarity=polarity, marked=marked)
                if got is None:
                    continue
                assert _is_snap(array, got, source=source, polarity=polarity), (source, click, got)
                for off in (-0.5, -0.02, 0.02, 0.5):
                    assert not _is_snap(array, got + off, source=source, polarity=polarity), (
                        source,
                        click,
                        got,
                        off,
                    )


def test_a_snap_position_is_where_the_climb_from_its_row_ends():
    # A band on row 80's centre, without noise, peaks at 80.5 exactly: a snap
    # lands there, and only there is a snap position; not on the row's edges,
    # the next row's centre, the band's flank or the membrane below it.
    clean = _bands((80.5, 12000.0))
    assert _snap(clean, 84.0) == 80.5 and _is_snap(clean, 80.5)
    for y in (80.0, 81.0, 81.5, 80.5 + 1e-8, 84.0, 120.5, -3.0, SNAP_H + 3.0):
        assert not _is_snap(clean, y), y
    # Read 12 px to the side, the band peaks at the same y: it is as high
    # either side of row 80 in every column. Past the image's side, no column
    # is read.
    assert mwcal.is_snap_position(clean, SNAP_X + 12.0, 80.5, source=MARKER, polarity=DARK)
    assert not mwcal.is_snap_position(clean, -20.0, 80.5, source=MARKER, polarity=DARK)
    # A faint marker may show either way; read as a marker band of the wrong
    # polarity, a band is a trough, and its peak no snap position.
    dark = _ladder_pixels([80.5])
    light = 2 * MEMBRANE_LEVEL - dark
    for array in (dark, light):
        got = _snap(array, 84.0, source=CHEMI_MARKER)
        assert _is_snap(array, got, source=CHEMI_MARKER)
        assert _is_snap(array, got, source=CHEMI_MARKER, polarity=LIGHT)
    got = _snap(light, 84.0, polarity=LIGHT)
    assert _is_snap(light, got, polarity=LIGHT) and not _is_snap(light, got, polarity=DARK)
    # A strip edge: on the steps between rows.
    strip = np.full((SNAP_H, SNAP_W), 2000.0)
    strip[:120] = MEMBRANE_LEVEL
    strip = _noisy(strip, seed=5)
    got = _snap(strip, 118.0, source=STRIP)
    assert _is_snap(strip, got, source=STRIP) and not _is_snap(strip, got, source=MARKER)
    assert not _is_snap(strip, 120.0, source=STRIP) and got != 120.0


def test_a_strong_broad_faint_marker_snaps_from_most_clicks_near_it():
    # One faint marker band, 30000 counts (500 noise sigmas) deep, 12 px
    # (1/e) across, with a sigma of 4.5 rows down, dark or bright on the
    # membrane: a click within 5 px of it snaps onto it, but for a few whose
    # windows the band's own flanks fill with steps (the noise is the steps')
    # as large as the band stands out. A band need not stand out of a window
    # around itself as well: from there, where its flanks fill the window, it
    # often does not.
    snapped, clicks = 0, 0
    for seed in range(6):
        y = 100.5 + 0.2 * seed
        spot = [(SNAP_X, y - 0.5, 12.0, 4.5 * math.sqrt(2.0), 30000.0)]
        dark = _noisy(synthetic_blot((SNAP_H, SNAP_W), spot, dtype=np.float64), seed)
        for array in (dark, 2 * MEMBRANE_LEVEL - dark):
            for click in y + np.arange(-5.0, 5.01, 0.5):
                clicks += 1
                got = _snap(array, click, source=CHEMI_MARKER)
                if got is not None and abs(got - y) < 1.0:
                    snapped += 1
                    assert _is_snap(array, got, source=CHEMI_MARKER)
    assert clicks == 252 and snapped >= 240


def _dense_ladder(depth: float, rows: float, seed: int, *, bright: bool = False):
    """Sixteen bands about 10 px apart (each up to half a pixel off its
    place), ``depth`` counts deep, 20 px wide and ``rows`` px (1/e) high,
    under 60 counts of seeded noise, as 16-bit pixels (bright on the membrane
    where ``bright``). With the bands' ys and a ruler's drags off them, 0.3
    to 3 px either way, all seeded."""
    rng = np.random.default_rng(seed)
    ys = [25.5 + 10.0 * i + float(rng.uniform(-0.5, 0.5)) for i in range(16)]
    spots = [(SNAP_X, y - 0.5, 10.0, rows, depth) for y in ys]
    array = synthetic_blot((SNAP_H, SNAP_W), spots, dtype=np.float64)
    array = array + rng.normal(0.0, 60.0, array.shape)
    if bright:
        array = 2 * MEMBRANE_LEVEL - array
    drags = [float(rng.uniform(0.3, 3.0) * rng.choice([-1, 1])) for _ in ys]
    return np.clip(np.round(array), 0, 65535).astype(np.uint16), ys, drags


DENSE_LADDERS = {
    # source, polarity, depth, rows, bright: of 112 interior bands, how many
    # Snap all and single clicks snap onto.
    "marker": (MARKER, DARK, 960.0, 1.5, False, 60, 78),
    "chemiluminescence, strong": (CHEMI_MARKER, LIGHT, 4000.0, 2.0, True, 38, 60),
}


@pytest.mark.parametrize("case", DENSE_LADDERS)
def test_a_dense_ladder_snaps_as_far_as_the_marked_gaps_allow(case):
    # Bands 10 px apart, on eight seeded images. Snap all snaps each tick of a
    # dragged ruler within the gaps to the other ticks; a click marks the
    # bands top to bottom, each within the gaps to those marked before it.
    # How many bands they take is set by the windows the neighbours cut (a
    # band needs a lower row on each side in it) and what stands out of them;
    # a band need not stand out of a window around itself as well, which its
    # neighbours' flanks fill. Every y they land on is a snap position.
    source, polarity, depth, rows, bright, snap_all, single = DENSE_LADDERS[case]
    took = {"snap_all": 0, "single": 0}
    for seed in range(8):
        array, ys, drags = _dense_ladder(depth, rows, seed, bright=bright)
        given = [y + d for y, d in zip(ys, drags, strict=True)]
        marked: list[float] = []
        for k, y in enumerate(ys):
            others = given[:k] + given[k + 1 :]
            all_ = _snap(array, given[k], source=source, polarity=polarity, marked=others)
            one = _snap(array, given[k], source=source, polarity=polarity, marked=marked)
            marked.append(given[k] if one is None else one)
            for how, got in (("snap_all", all_), ("single", one)):
                assert got is None or _is_snap(array, got, source=source, polarity=polarity)
                took[how] += 0 < k < len(ys) - 1 and got is not None and abs(got - y) < 1.0
    assert took["snap_all"] >= snap_all and took["single"] >= single, took


SATURATED_KDA = (220, 120, 100, 80, 60, 50, 40, 30, 20)
SATURATED_YS = (79.5, 145.5, 175.5, 236.5, 289.5, 395.5, 468.5, 596.5, 727.5)


def _saturated_lane(kind: str) -> tuple[np.ndarray, CalibrationPointSource, Polarity]:
    """A ladder at x = 60 whose bands saturate over their middle nine rows,
    80 px wide (wider than the columns a snap or a found ladder reads): a
    chemiluminescence marker exposed long enough to reach 65535 on a 16-bit
    chemiluminescence image, or a dark visible marker clipped at 0 on an 8-bit
    marker image. With its source and polarity."""
    rows, columns = np.mgrid[0:860, 0:120]
    shape = sum(np.exp(-((rows - (y - 0.5)) ** 2) / 18.0) for y in SATURATED_YS)
    shape = shape * (np.abs(columns - 60) < 40)
    noise = np.random.default_rng(0).normal(0.0, 1.0, shape.shape)
    if kind == "chemiluminescence":
        image = np.clip(np.round(800.0 + 200000.0 * shape + 30.0 * noise), 0, 65535)
        return image.astype(np.uint16), CHEMI_MARKER, LIGHT
    image = np.clip(np.round(200.0 - 700.0 * shape + 4.0 * noise), 0, 255)
    return image.astype(np.uint8), MARKER, DARK


@pytest.mark.parametrize("kind", ("chemiluminescence", "visible"))
def test_a_saturated_band_lies_at_the_middle_of_its_flat_top(kind):
    # Each band's middle nine rows are clipped to one value: its flat top is
    # found, and snapped to, at its middle (not 3.5 px above it, at the lower
    # edge of its first row).
    array, source, polarity = _saturated_lane(kind)
    assert all(
        np.all(array[round(y - 0.5) + d, 40:81] == array[round(y - 0.5), 60])
        for y in SATURATED_YS
        for d in range(-4, 5)
    )
    proposal = mwcal.find_ladder(array, 60.0, kda=SATURATED_KDA, source=source, polarity=polarity)
    assert [(t.mw, t.y, t.found) for t in proposal.ticks] == [
        (mw, y, True) for mw, y in zip(SATURATED_KDA, SATURATED_YS, strict=True)
    ]
    for y in SATURATED_YS:
        for click in (y - 3.0, y, y + 3.0):
            assert mwcal.refine_point(array, 60.0, click, source=source, polarity=polarity) == y


# --- Finding a ladder (find_ladder) ---

FIND_X = 40.0
# The vendor's band positions for PageRuler Plus on four gels (px, top to
# bottom; the 10 kDa band ran off the 10 % gel), with the ladder's MWs and its
# two orange reference bands.
VENDOR_LADDERS = {
    "TG 4-20%": (VENDOR_YS, KDA, (70, 25)),
    "TG 10%": ((277.5, 317.5, 352.5, 392.5, 437.5, 528.5, 587.5, 720.5), KDA[:-1], (70, 25)),
    "BT 4-12% MOPS": (
        (320.5, 363.0, 405.5, 454.0, 512.5, 614.0, 662.0, 736.5, 773.5),
        (185, 115, 80, 65, 50, 30, 25, 15, 10),
        (65, 25),
    ),
    "BT 4-12% MES": (
        (309.5, 352.5, 384.5, 405.5, 469.5, 560.5, 598.0, 688.5, 768.5),
        (190, 115, 80, 70, 50, 30, 25, 15, 10),
        (70, 25),
    ),
}
CHANGES = ("none", "top", "bottom", "extra")
# The gap to the next-best labelling of each gel and change, with the
# reference bands as anchors: every labelling right, and the false band of
# "extra" leaves every gel doubtful. (Anchors taken as a hard rule would put
# the 10 % gel without its bottom band at 15.1; a reference MW off the
# strongest peaks costs REF_COST here.)
ANCHORED_GAPS = {
    "TG 4-20%": (12.7, 10.4, 6.8, 1.4),
    "TG 10%": (13.2, 10.6, 9.9, 1.6),
    "BT 4-12% MOPS": (11.1, 6.7, 8.8, 1.7),
    "BT 4-12% MES": (13.3, 10.4, 6.1, 2.8),
}
# From the band positions alone: 10 of the 16 right, and the gap of the
# labelling proposed, right or wrong.
POSITION_GAPS = {
    "TG 4-20%": ((12.7, True), (3.2, True), (2.5, False), (2.6, False)),
    "TG 10%": ((13.2, True), (2.5, True), (0.3, False), (1.7, False)),
    "BT 4-12% MOPS": ((11.1, True), (1.1, True), (0.4, True), (1.4, False)),
    "BT 4-12% MES": ((13.3, True), (5.6, True), (1.6, True), (0.5, False)),
}


def _marker_lane(bands, *, height: int = 860, width: int = 80, seed: int = 7) -> np.ndarray:
    """A visible-light marker's ladder lane at FIND_X: dark bands ``(y, depth)``
    centred at the continuous ``y``, 60 px wide and a few rows high, with
    seeded noise of 60 counts on a flat membrane."""
    spots = [(FIND_X, y - 0.5, 30.0, 3.0, depth) for y, depth in bands]
    return _noisy(synthetic_blot((height, width), spots, dtype=np.float64), seed)


def _vendor_case(name: str, change: str):
    """A gel's band positions changed as ``change`` says, as the lane's pixels,
    its MWs, its reference MWs and the true y of each MW: the reference bands
    twice as dark; ``top`` and ``bottom`` leave that band out, ``extra`` adds a
    fainter false band halfway between the 4th and the 5th."""
    ys, kda, reference = VENDOR_LADDERS[name]
    bands = [(y, 12000.0 if mw in reference else 6000.0) for y, mw in zip(ys, kda, strict=True)]
    if change == "top":
        bands = bands[1:]
    elif change == "bottom":
        bands = bands[:-1]
    elif change == "extra":
        bands.append((0.5 * (ys[3] + ys[4]), 4800.0))
    return _marker_lane(bands), kda, reference, dict(zip(kda, ys, strict=True))


def _find(array, kda, reference=(), *, x=FIND_X, polarity=DARK, source=MARKER, other=None):
    return mwcal.find_ladder(
        array, x, kda=kda, reference_kda=reference, source=source, polarity=polarity, other=other
    )


def _found(proposal: mwcal.LadderProposal) -> dict[float, float]:
    return {tick.mw: tick.y for tick in proposal.ticks if tick.found}


def _right(proposal: mwcal.LadderProposal | None, truth: dict[float, float]) -> bool:
    """Every found tick within 3 px of its MW's band."""
    return proposal is not None and all(
        abs(truth[mw] - y) <= 3.0 for mw, y in _found(proposal).items()
    )


@pytest.mark.parametrize("name", VENDOR_LADDERS)
def test_find_ladder_on_vendor_band_positions(name):
    for change, gap in zip(CHANGES, ANCHORED_GAPS[name], strict=True):
        array, kda, reference, truth = _vendor_case(name, change)
        proposal = _find(array, kda, reference)
        assert proposal is not None
        # Every band drawn is found where it lies, and labelled right; the
        # false band is left over.
        found = _found(proposal)
        left_out = {"top": kda[0], "bottom": kda[-1]}.get(change)
        assert set(found) == {mw for mw in kda if mw != left_out}, change
        assert all(abs(found[mw] - truth[mw]) < 0.05 for mw in found), change
        assert len(proposal.extra) == (change == "extra")
        assert math.isclose(proposal.gap, gap, abs_tol=0.05), (change, proposal.gap)
        assert proposal.doubtful is (change == "extra") is (proposal.gap < GAP_WARN)


@pytest.mark.parametrize("name", VENDOR_LADDERS)
def test_find_ladder_on_vendor_band_positions_alone_is_often_one_band_off(name):
    # Without the reference bands, the positions alone mislabel 6 of the 16.
    for change, (gap, right) in zip(CHANGES, POSITION_GAPS[name], strict=True):
        array, kda, _, truth = _vendor_case(name, change)
        proposal = _find(array, kda)
        assert _right(proposal, truth) is right, change
        assert math.isclose(proposal.gap, gap, abs_tol=0.05), (change, proposal.gap)


def _sample_marker_cases():
    """The sample marker's ladder lane: level, turned by 2 degrees
    either way (bilinear, as a rotated photo is), with the 250 kDa band cut off,
    and with a speck of dust on the 35 kDa band; each as the image, the lane's
    x and the true y of each band."""
    from scipy import ndimage

    marker = samples.render_marker().astype(float)
    height, width = marker.shape
    truth = {mw: samples.band_y(mw, samples.LADDER_X) + 0.5 for mw in samples.LADDER_KDA}
    cases = {"level": (marker, samples.LADDER_X, truth)}
    cx, cy = (width - 1) / 2, (height - 1) / 2
    for degrees in (2.0, -2.0):
        turned = ndimage.rotate(marker, degrees, reshape=False, order=1, mode="nearest")
        cos, sin = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        x = cx + (samples.LADDER_X - cx) * cos + (250 - cy) * sin
        moved = {
            mw: cy - (samples.LADDER_X - cx) * sin + (y - 0.5 - cy) * cos + 0.5
            for mw, y in truth.items()
        }
        cases[f"turned {degrees:+.0f}"] = (turned, x, moved)
    cases["250 cut off"] = (marker[70:], samples.LADDER_X, {m: y - 70 for m, y in truth.items()})
    rows, columns = np.mgrid[0:height, 0:width]
    dust = np.exp(-((rows - 300) ** 2 + (columns - samples.LADDER_X) ** 2) / (2 * 3.0**2))
    cases["dust"] = (marker - 60 * dust, samples.LADDER_X, truth)
    return cases


# The gap on each of the sample marker's cases, with its darker bands or with
# the preset's reference bands (the sample does not darken the green 10 kDa
# band).
SAMPLE_GAPS = {
    "level": 19.2,
    "turned +2": 19.3,
    "turned -2": 19.3,
    "250 cut off": 5.3,
    "dust": 19.2,
}


@pytest.mark.parametrize("case", SAMPLE_GAPS)
def test_find_ladder_on_the_sample_marker(case):
    array, x, truth = _sample_marker_cases()[case]
    preset = ladders.preset("pageruler_plus/tris_glycine")
    for reference in (samples.LADDER_REFERENCE_KDA, tuple(b.kda for b in preset.reference)):
        proposal = _find(array, samples.LADDER_KDA, reference, x=x)
        found = _found(proposal)
        # Every band on the image found and labelled right: none left over,
        # none predicted (the band cut off would lie above the image).
        expected = [mw for mw in samples.LADDER_KDA if 0 < truth[mw] < array.shape[0]]
        assert list(found) == expected and len(proposal.ticks) == len(expected)
        tolerance = 0.3 if case in ("level", "250 cut off", "dust") else 1.0
        assert all(abs(found[mw] - truth[mw]) < tolerance for mw in found), found
        assert proposal.extra == ()
        assert math.isclose(proposal.gap, SAMPLE_GAPS[case], abs_tol=0.1), proposal.gap
        assert not proposal.doubtful
        assert all(tick.strength > 100 for tick in proposal.ticks)
        assert proposal.x == x


def test_find_ladder_positions_are_continuous_and_follow_the_click():
    # Without noise, bands centred on rows 99 and 199 lie at y = 99.5 and 199.5.
    array = synthetic_blot(
        (300, 80), [(FIND_X, y, 30.0, 3.0, 6000.0) for y in (99.0, 199.0)], dtype=np.float64
    )
    proposal = _find(array, (100, 50))
    assert [(t.mw, t.y, t.found) for t in proposal.ticks] == [(100, 99.5, True), (50, 199.5, True)]
    # A lane 90 px to the right, on bare membrane: nothing to find there.
    wide = np.pad(array, ((0, 0), (0, 80)), constant_values=MEMBRANE_LEVEL)
    assert _find(wide, (100, 50), x=FIND_X + 90) is None


def test_find_ladder_predicts_the_bands_it_did_not_find():
    ys, kda, reference = VENDOR_LADDERS["TG 4-20%"]
    # The 250 and 55 kDa bands left out.
    bands = [(y, 12000.0 if mw in reference else 6000.0) for y, mw in zip(ys, kda, strict=True)]
    proposal = _find(_marker_lane(bands[1:4] + bands[5:]), kda, reference)
    ticks = {tick.mw: tick for tick in proposal.ticks}
    assert [tick.mw for tick in proposal.ticks] == list(kda)  # top to bottom
    assert [mw for mw, tick in ticks.items() if not tick.found] == [250, 55]
    assert ticks[250].strength is None and ticks[55].strength is None
    # Each where the quadratic through the found bands puts it: 55 kDa between
    # its neighbours, near where the vendor draws it; 250 kDa above 130.
    assert ticks[70].y < ticks[55].y < ticks[35].y
    assert abs(ticks[55].y - ys[4]) < 10.0
    assert 0.0 < ticks[250].y < ticks[130].y


def test_find_ladder_draws_no_tick_the_quadratic_never_reaches():
    # Five bands on a ladder whose log10(MW) turns over just below them: its
    # quadratic never reaches 35 kDa, so the lighter MWs get no tick.
    turn, a = 600.0, (math.log10(250) - math.log10(45)) / 500.0**2
    ys = [turn - math.sqrt((math.log10(mw) - math.log10(45)) / a) for mw in KDA[:5]]
    proposal = _find(_marker_lane([(y, 6000.0) for y in ys], height=700), KDA)
    assert [(tick.mw, tick.found) for tick in proposal.ticks] == [(mw, True) for mw in KDA[:5]]
    assert all(abs(tick.y - y) < 0.05 for tick, y in zip(proposal.ticks, ys, strict=True))


def test_find_ladder_draws_no_predicted_tick_out_of_order_or_off_the_image():
    # A quadratic that misses its found bands can put a predicted tick above a
    # heavier found band: it is not drawn. Here z = -u, u = (y - 250) / 150.
    peaks = [(100.0, 5.0), (110.0, 5.0), (400.0, 5.0)]
    labelling = mwcal._Labelling(
        pairs=((0, 0), (1, 2), (2, 4)),
        score=0.0,
        gap=math.inf,
        fitted=(0.0, -1.0, 0.0),
        mid=250.0,
        scale=150.0,
    )
    kda = tuple(10.0**z for z in (1.2, 1.1, 0.9, 0.5, -0.8))
    ticks = mwcal._ticks(peaks, kda, labelling, 500.0)
    # 1.1 would lie at y = 85, above the found 1.2 at 100; 0.5 at y = 175.
    assert [(t.mw, t.y, t.found) for t in ticks] == [
        (kda[0], 100.0, True),
        (kda[2], 110.0, True),
        (kda[3], pytest.approx(175.0), False),
        (kda[4], 400.0, True),
    ]
    # On an image 150 px high, 175 lies off it.
    assert [t.mw for t in mwcal._ticks(peaks, kda, labelling, 150.0)] == [kda[0], kda[2], kda[4]]


def test_find_ladder_follows_the_polarity_and_takes_a_faint_marker_either_way():
    array, kda, reference, truth = _vendor_case("TG 4-20%", "none")
    light = 2 * MEMBRANE_LEVEL - array  # the same bands, bright on a dark membrane
    assert _right(_find(light, kda, reference, polarity=LIGHT), truth)
    # Read the wrong way round, the bands are troughs: no band is found.
    wrong = _find(light, kda, reference, polarity=DARK)
    assert wrong is None or not any(
        abs(y - band) < 5.0 for y in _found(wrong).values() for band in truth.values()
    )
    for image in (array, light):
        for polarity in (DARK, LIGHT):
            proposal = _find(image, kda, reference, polarity=polarity, source=CHEMI_MARKER)
            assert _right(proposal, truth) and len(_found(proposal)) == len(kda)


def test_find_ladder_takes_no_shading_for_a_band():
    # A membrane shaded from top to bottom by 14 noise sigmas of its profile:
    # no band.
    shaded = np.full((860, 80), MEMBRANE_LEVEL) + np.linspace(-400.0, 400.0, 860)[:, None]
    for seed in range(5):
        assert _find(_noisy(shaded, seed), KDA) is None, seed
    # The vendor's bands on it are found as on a flat membrane.
    array, kda, reference, truth = _vendor_case("TG 4-20%", "none")
    proposal = _find(array + np.linspace(-400.0, 400.0, 860)[:, None], kda, reference)
    assert _right(proposal, truth) and len(_found(proposal)) == len(kda) and not proposal.extra


def _written_lane(membrane: float, noise: float, *, depth: float = 110.0, dtype=np.uint8):
    """The vendor's TG 4-20 % bands on a membrane at ``membrane`` counts with
    seeded noise of ``noise``, rounded and clipped to ``dtype`` as an imager
    writes them: a membrane past its range is clipped flat, one with little
    noise is flat within a grey level. The bands are ``depth`` counts darker
    (brighter where negative), the reference bands half as much again."""
    ys, kda, reference = VENDOR_LADDERS["TG 4-20%"]
    spots = [
        (FIND_X, y - 0.5, 30.0, 3.0, depth * (1.5 if mw in reference else 1.0))
        for y, mw in zip(ys, kda, strict=True)
    ]
    image = synthetic_blot((860, 80), spots, dtype=np.float64) - MEMBRANE_LEVEL + membrane
    image += np.random.default_rng(3).normal(0.0, noise, image.shape)
    info = np.iinfo(dtype)
    return np.clip(np.round(image), info.min, info.max).astype(dtype)


# An 8-bit marker lane whose membrane is clipped white: wholly (its steps are
# 0, however noisy the bands are) or all but a few pixels (its steps are far
# smaller than the bands' noise); one on a membrane flat within a grey level;
# a faint marker on a 16-bit chemiluminescence image whose background the
# imager clipped at 0, wholly or in part.
WRITTEN_LANES = {
    "clipped white": (lambda: _written_lane(265.0, 4.0), MARKER, DARK),
    "clipped white but a few pixels": (lambda: _written_lane(264.0, 4.0), MARKER, DARK),
    "flat within a grey level": (lambda: _written_lane(200.0, 0.15), MARKER, DARK),
    "background clipped at 0": (
        lambda: _written_lane(-100.0, 30.0, depth=-600.0, dtype=np.uint16),
        CHEMI_MARKER,
        LIGHT,
    ),
    "background clipped at 0 in part": (
        lambda: _written_lane(-40.0, 30.0, depth=-600.0, dtype=np.uint16),
        CHEMI_MARKER,
        LIGHT,
    ),
}


@pytest.mark.parametrize("case", WRITTEN_LANES)
def test_find_ladder_on_a_membrane_clipped_or_flat_within_a_grey_level(case):
    # The noise is the bands' own, not the flat membrane's: found as on a
    # membrane with noise, with no ripple on a band taken for one.
    make, source, polarity = WRITTEN_LANES[case]
    ys, kda, reference = VENDOR_LADDERS["TG 4-20%"]
    reference = reference if source is MARKER else ()
    proposal = _find(make(), kda, reference, source=source, polarity=polarity)
    truth = dict(zip(kda, ys, strict=True))
    assert _right(proposal, truth) and len(_found(proposal)) == len(kda)
    assert proposal.extra == () and not proposal.doubtful
    assert math.isclose(proposal.gap, ANCHORED_GAPS["TG 4-20%"][0], abs_tol=0.1), proposal.gap
    assert all(tick.strength < 1e4 for tick in proposal.ticks)


def test_find_ladder_needs_two_bands():
    flat = _noisy(np.full((400, 80), MEMBRANE_LEVEL), seed=1)
    assert _find(flat, KDA) is None
    assert _find(_marker_lane([(200.5, 6000.0)], height=400), KDA) is None
    lane = _marker_lane([(100.5, 6000.0), (200.5, 6000.0)], height=400)
    assert _find(lane, KDA[:1]) is None  # a ladder of one MW
    assert _find(lane, KDA, x=-40.0) is None  # no column of the lane on the image
    assert _find(lane[:2], KDA) is None  # two rows: no peak can stand out


def test_find_ladder_gap_without_another_labelling_is_infinite():
    # Two bands and two MWs: one labelling only, none to be one band off.
    proposal = _find(_marker_lane([(100.5, 6000.0), (200.5, 6000.0)], height=400), (100, 50))
    assert [(t.mw, t.found) for t in proposal.ticks] == [(100, True), (50, True)]
    assert (proposal.gap, proposal.doubtful, proposal.score) == (math.inf, False, 0.0)


def test_find_ladder_second_ladder_takes_the_first_as_other():
    # The gels whose bottom band is missing mislabel from positions alone. A
    # second ladder 10 px lower than a first one marked on every band keeps its
    # height difference from that one steady only when labelled right.
    for name in ("TG 4-20%", "TG 10%"):
        ys, kda, reference = VENDOR_LADDERS[name]
        zs = tuple(math.log10(mw) for mw in kda)
        first = mwcal.LadderCurve(ys, zs, zs[-1] - 0.1, zs[0] + 0.1)
        bands = [
            (y + 10.0, 12000.0 if mw in reference else 6000.0)
            for y, mw in zip(ys, kda, strict=True)
        ]
        lane = _marker_lane(bands[:-1])
        truth = {mw: y + 10.0 for mw, y in zip(kda, ys, strict=True)}
        assert not _right(_find(lane, kda), truth), name
        proposal = _find(lane, kda, other=first)
        assert _right(proposal, truth) and set(_found(proposal)) == set(kda[:-1]), name
        # With the reference bands too, the next labelling lies much further.
        alone, both = _find(lane, kda, reference), _find(lane, kda, reference, other=first)
        assert _right(both, truth) and not both.doubtful
        assert both.gap > alone.gap + 5.0, (alone.gap, both.gap)


def test_find_ladder_is_deterministic():
    array, kda, reference, _ = _vendor_case("BT 4-12% MES", "extra")
    assert _find(array, kda, reference) == _find(array.copy(), kda, reference)


def test_find_ladder_stops_at_its_budget(monkeypatch):
    array, kda, reference, _ = _vendor_case("TG 4-20%", "none")
    monkeypatch.setattr(mwcal, "FIND_BUDGET", 5)
    proposal = _find(array, kda, reference)
    # The best labelling it found, doubtful: nothing shows it is the best.
    assert (proposal.gap, proposal.doubtful) == (0.0, True)
    assert proposal.ticks


def test_find_ladder_refuses_a_strip_edge_or_mws_out_of_order():
    lane = _marker_lane([(100.5, 6000.0), (200.5, 6000.0)], height=400)
    with pytest.raises(ValueError, match="strip edge"):
        _find(lane, KDA, source=STRIP)
    for kda in ((100, 100), (50, 100), (100, 0), (100, math.nan)):
        with pytest.raises(ValueError, match="ladder MWs"):
            _find(lane, kda)


def test_find_ladder_constants():
    assert (FIND_HALF_X, FIND_BACKGROUND, PRIOR_DECADES, GAP_WARN) == (20, 0.1, 0.05, 5.0)
    assert (MISS_COST, EXTRA_COST, REF_COST, FIND_BUDGET) == (9.0, 9.0, 9.0, 200_000)
