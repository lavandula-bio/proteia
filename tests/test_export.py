# SPDX-License-Identifier: Apache-2.0
"""Tests for GUI-independent file exports."""

import csv

import pytest

from proteia.core.export import write_lane_table

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
        ["0", "vehicle", "α1", "yes", "1.0", "0.5"],
        ["1", "10 µM", "β2", "yes", "2.5", ""],
        ["2", "10 µM", "", "no", "", "3.25"],
    ]


def test_lane_table_rounds_nets_to_three_decimals(tmp_path):
    path = tmp_path / "table.csv"
    write_lane_table(path, ["a"], ["s1"], [True], [("p", [1.23456])])
    assert _read_rows(path)[1][-1] == "1.235"


def test_lane_table_rejects_misaligned_columns(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        write_lane_table(tmp_path / "t.csv", ["a", "b"], ["s1"], [True, True], [])
    with pytest.raises(ValueError, match="2 lanes"):
        write_lane_table(tmp_path / "t.csv", ["a", "b"], ["s1", "s2"], [True, True], [("p", [1])])
