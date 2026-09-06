# SPDX-License-Identifier: Apache-2.0
"""Tests for the single-image quantification data model and its invariants."""

import pytest
from pydantic import ValidationError

from proteia.core.model import Analysis, Band, Box, BoxSize, ImageRef, Lane

IMAGE = ImageRef(path="blot.tif", sha256="abc123", width=100, height=100)
SIZE = BoxSize(width=10, height=4)


def _band(lane: int, x: int, y: int, raw=None, bg=None) -> Band:
    """A band whose measurement box is at (x, y) and background box 10px below."""
    return Band(
        lane_index=lane,
        box=Box(x=x, y=y),
        background=Box(x=x, y=y + 10),
        raw=raw,
        background_signal=bg,
    )


def _analysis(bands: list[Band]) -> Analysis:
    return Analysis(
        image=IMAGE,
        protein="BDNF",
        expected_mw=27.0,
        box_size=SIZE,
        lanes=[Lane(index=0, label="control"), Lane(index=1, label="treated")],
        bands=bands,
    )


def _valid_bands() -> list[Band]:
    # Lanes spaced on x (0, 30); background box 10px below each measurement box.
    return [_band(0, 0, 0, raw=120, bg=20), _band(1, 30, 0, raw=80, bg=30)]


def test_valid_analysis_ok():
    analysis = _analysis(_valid_bands())
    assert analysis.protein == "BDNF"
    assert len(analysis.bands) == 2


def test_equal_area_for_all_boxes():
    analysis = _analysis(_valid_bands())
    areas = {(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in analysis.all_rects()}
    assert areas == {SIZE.area}


def test_net_is_raw_minus_background():
    assert _band(0, 0, 0, raw=120, bg=20).net == 100
    assert _band(0, 0, 0).net is None  # missing measurements


def test_overlapping_boxes_rejected():
    bands = _valid_bands()
    bands[1] = _band(1, 0, 0)  # same position as band 0 -> overlap
    with pytest.raises(ValidationError, match="overlap"):
        _analysis(bands)


def test_box_out_of_bounds_rejected():
    bands = _valid_bands()
    bands[0] = _band(0, 95, 0)  # x1 = 105 > image width 100
    with pytest.raises(ValidationError, match="bounds"):
        _analysis(bands)


def test_band_referencing_unknown_lane_rejected():
    bands = _valid_bands()
    bands.append(_band(9, 60, 0))  # lane 9 does not exist
    with pytest.raises(ValidationError, match="unknown lane"):
        _analysis(bands)


def test_serialization_roundtrip():
    original = _analysis(_valid_bands())
    restored = Analysis.model_validate_json(original.model_dump_json())
    assert restored == original
