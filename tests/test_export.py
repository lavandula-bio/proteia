# SPDX-License-Identifier: Apache-2.0
"""Tests for GUI-independent file exports."""

import csv

import pytest

from proteia.core.export import (
    LANE_TABLE_DECIMALS,
    LANE_TABLE_RATIO_DECIMALS,
    lane_table_bytes,
    write_lane_table,
)

BOM = b"\xef\xbb\xbf"


def _read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def test_lane_table_round_trips_non_ascii_names(tmp_path):
    # µ is not in cp950 (Traditional Chinese Windows), so a locale-encoded write
    # raised UnicodeEncodeError; α and β were written in the locale code page.
    path = tmp_path / "β-actin 10 µM.csv"
    write_lane_table(
        path,
        conditions=["vehicle", "10 µM", "10 µM"],
        samples=["α1", "β2", None],
        included=[True, True, False],
        proteins=[("β-actin", [1.0, 2.5, None]), ("α-tubulin", [0.5, None, 3.25])],
    )
    assert path.read_bytes().startswith(BOM)  # Excel detects UTF-8 from the BOM
    assert _read_rows(path) == [
        ["lane", "condition", "sample", "include", "β-actin", "α-tubulin"],
        ["1", "vehicle", "α1", "yes", "1.0", "0.5"],  # lanes numbered from 1, as the app does
        ["2", "10 µM", "β2", "yes", "2.5", ""],
        ["3", "10 µM", "", "no", "", "3.25"],
    ]


def test_lane_table_rounds_nets_to_three_decimals(tmp_path):
    path = tmp_path / "table.csv"
    write_lane_table(path, ["a"], ["s1"], [True], [("p", [1.23456])])
    assert _read_rows(path)[1][-1] == "1.235"


def test_lane_table_rejects_misaligned_columns(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        write_lane_table(tmp_path / "t.csv", ["a", "b"], ["s1"], [True, True], [])
    with pytest.raises(ValueError, match="same length"):
        write_lane_table(tmp_path / "t.csv", ["a", "b"], ["s1", "s2"], [True], [])
    assert not (tmp_path / "t.csv").exists()  # checked before the file is opened
    with pytest.raises(ValueError, match="2 lanes"):
        write_lane_table(tmp_path / "t.csv", ["a", "b"], ["s1", "s2"], [True, True], [("p", [1])])


def test_lane_table_marks_clipped_bands(tmp_path):
    path = tmp_path / "clipped.csv"
    write_lane_table(
        path,
        ["a", "a", "b"],
        ["s1", "s2", "s3"],
        [True, True, True],
        [("β-actin", [1.0, 2.0, None]), ("α-tubulin", [3.0, 4.0, 5.0])],
        clipped={"β-actin": [True, False, None]},
    )
    rows = _read_rows(path)
    assert rows[0] == [
        "lane",
        "condition",
        "sample",
        "include",
        "β-actin",
        "β-actin clipped",
        "α-tubulin",
    ]
    assert [row[5] for row in rows[1:]] == ["yes", "no", ""]
    with pytest.raises(ValueError, match="2 clipping flags but there are 1 lanes"):
        write_lane_table(path, ["a"], ["s1"], [True], [("p", [1.0])], clipped={"p": [True, False]})
    with pytest.raises(ValueError, match="not in the table"):
        write_lane_table(path, ["a"], ["s1"], [True], [("p", [1.0])], clipped={"q": [True]})
    with pytest.raises(ValueError, match="share a name"):
        write_lane_table(
            path,
            ["a"],
            ["s1"],
            [True],
            [("p", [1.0]), ("p clipped", [2.0])],
            clipped={"p": [True]},
        )


def test_lane_table_bytes_match_the_written_file(tmp_path):
    table = (
        ["vehicle", "10 µM"],
        ["α1", None],
        [True, False],
        [("β-actin", [1.23456, None])],
    )
    data = lane_table_bytes(*table, clipped={"β-actin": [False, None]})
    path = tmp_path / "table β.csv"
    write_lane_table(path, *table, clipped={"β-actin": [False, None]})
    assert path.read_bytes() == data
    assert LANE_TABLE_DECIMALS == 3
    rows = (
        "lane,condition,sample,include,β-actin,β-actin clipped\r\n"
        "1,vehicle,α1,yes,1.235,no\r\n"  # rounded to LANE_TABLE_DECIMALS
        "2,10 µM,,no,,\r\n"
    )
    assert data == BOM + rows.encode()

    collision = tmp_path / "collision.csv"
    with pytest.raises(ValueError, match="share a name"):
        write_lane_table(
            collision,
            ["a"],
            ["s1"],
            [True],
            [("p", [1.0]), ("p clipped", [2.0])],
            clipped={"p": [True]},
        )
    assert not collision.exists()  # checked before the file is opened


def test_lane_table_series_columns_follow_the_proteins():
    data = lane_table_bytes(
        ["vehicle", "10 µM", "10 µM"],
        ["α1", "β2", None],
        [True, True, False],
        [("β-actin", [1.0, 2.0, None])],
        clipped={"β-actin": [False, None, None]},
        series=[
            ("β-actin ÷ GAPDH normalized", [0.123456789, None, 2.0]),
            ("β-actin ÷ GAPDH fold change vs vehicle", [1.0, float("nan"), float("inf")]),
        ],
    )
    assert LANE_TABLE_RATIO_DECIMALS == 6
    rows = (
        "lane,condition,sample,include,β-actin,β-actin clipped,"
        "β-actin ÷ GAPDH normalized,β-actin ÷ GAPDH fold change vs vehicle\r\n"
        "1,vehicle,α1,yes,1.0,no,0.123457,1.0\r\n"  # rounded to LANE_TABLE_RATIO_DECIMALS
        "2,10 µM,β2,yes,2.0,,,\r\n"  # no value, or none that is finite: an empty cell
        "3,10 µM,,no,,,2.0,\r\n"
    )
    assert data == BOM + rows.encode()
    with pytest.raises(ValueError, match="2 values but there are 3 lanes"):
        lane_table_bytes(["a"] * 3, [None] * 3, [True] * 3, [], series=[("r", [1.0, 2.0])])
    with pytest.raises(ValueError, match="share a name"):
        lane_table_bytes(["a"], [None], [True], [("r", [1.0])], series=[("r", [1.0])])


def test_lane_table_cells_are_written_as_typed():
    # Provisional (#53): a cell that a spreadsheet reads as a formula is not
    # changed; whether to guard such cells is the maintainer's decision.
    data = lane_table_bytes(["-DOX", "+LPS", "=1+1", "@x"], [None] * 4, [True] * 4, [])
    rows = data.decode("utf-8-sig").splitlines()[1:]
    assert [row.split(",")[1] for row in rows] == ["-DOX", "+LPS", "=1+1", "@x"]
