# SPDX-License-Identifier: Apache-2.0
"""Tests for the evaluation helper: reference CSV files, IoU, the same-lane hit
rate, and the hit rate of row-box detection on the synthetic rows (#51)."""

import math
from pathlib import Path

import pytest

from proteia.core.evaluate import (
    REFERENCE_COLUMNS,
    HitRate,
    ReferenceFileError,
    evaluate_row,
    hit_rate,
    iou,
    read_reference_csv,
)
from proteia.core.quantify import estimate_background
from rowcases import adversarial, bench_cases

HEADER = ",".join(REFERENCE_COLUMNS)


def write(folder: Path, name: str, text: str, *, bom: bool = False) -> Path:
    path = folder / name
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return path


# --- Reading reference files ---


@pytest.mark.parametrize("bom", [True, False])
def test_reads_utf8_with_or_without_bom(tmp_path, bom):
    folder = tmp_path / "參考 µ α β"
    folder.mkdir()
    path = write(folder, "β-actin µ.csv", f"{HEADER}\n0,10,20,40,12\n2,100,21,38,11\n", bom=bom)
    assert read_reference_csv(path) == {0: (10, 20, 50, 32), 2: (100, 21, 138, 32)}


def test_header_case_spaces_crlf_and_blank_lines(tmp_path):
    text = "Lane, X ,y,Width,HEIGHT\r\n\r\n1, 5, 6, 7, 8\r\n , , , , \r\n3,0,0,1,1\r\n\r\n"
    assert read_reference_csv(write(tmp_path, "crlf.csv", text)) == {
        1: (5, 6, 12, 14),
        3: (0, 0, 1, 1),
    }


def test_header_only_means_no_band(tmp_path):
    assert read_reference_csv(write(tmp_path, "none.csv", HEADER + "\n")) == {}


def test_accepts_a_str_path(tmp_path):
    path = write(tmp_path, "s.csv", f"{HEADER}\n0,1,2,3,4\n")
    assert read_reference_csv(str(path)) == {0: (1, 2, 4, 6)}


@pytest.mark.parametrize(
    ("text", "line", "words"),
    [
        ("", 1, "first line"),
        ("lane,x0,y0,x1,y1\n0,1,2,3,4\n", 1, "first line"),
        ("lane,x,y,width\n0,1,2,3\n", 1, "first line"),
        ("lane,x,y,width,height,note\n", 1, "first line"),
        ("\nlane,x,y,width,height\n", 1, "first line"),  # the header comes first
        (f"{HEADER}\n0,1,2,3\n", 2, "expected 5 values"),
        (f"{HEADER}\n0,1,2,3,4,5\n", 2, "expected 5 values"),
        (f"{HEADER}\n0,1.5,2,3,4\n", 2, "integers"),
        (f"{HEADER}\n0,1_0,2,3,4\n", 2, "integers"),
        (f"{HEADER}\n0,x,2,3,4\n", 2, "integers"),
        (f"{HEADER}\n0,,2,3,4\n", 2, "integers"),
        (f"{HEADER}\n\n\n0,-1,2,3,4\n", 4, "negative"),
        (f"{HEADER}\n0,1,-2,3,4\n", 2, "negative"),
        (f"{HEADER}\n-1,1,2,3,4\n", 2, "negative"),
        (f"{HEADER}\n0,1,2,0,4\n", 2, "positive"),
        (f"{HEADER}\n0,1,2,3,-4\n", 2, "positive"),
        (f"{HEADER}\n0,1,2,3,4\n1,1,2,3,4\n0,5,6,7,8\n", 4, "listed twice"),
    ],
)
def test_bad_files_name_the_file_and_line(tmp_path, text, line, words):
    path = write(tmp_path, "α ref.csv", text)
    with pytest.raises(ReferenceFileError, match=words) as err:
        read_reference_csv(path)
    assert str(err.value).startswith(f"α ref.csv:{line}: ")
    assert isinstance(err.value, ValueError)


def test_non_utf8_file_names_its_line(tmp_path):
    path = tmp_path / "cp950.csv"
    path.write_bytes(f"{HEADER}\n0,1,2,3,4\n".encode() + "1,參,2,3,4\n".encode("cp950"))
    with pytest.raises(ReferenceFileError, match="not UTF-8") as err:
        read_reference_csv(path)
    assert str(err.value).startswith("cp950.csv:3: ")


# --- IoU ---


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
        ((0, 0, 10, 10), (5, 0, 15, 10), 50 / 150),
        ((0, 0, 10, 10), (0, 0, 10, 20), 0.5),
        ((0, 0, 10, 10), (2, 2, 4, 4), 4 / 100),
        ((0, 0, 10, 10), (10, 0, 20, 10), 0.0),  # edge contact
        ((0, 0, 10, 10), (0, 10, 10, 20), 0.0),
        ((0, 0, 10, 10), (30, 30, 40, 40), 0.0),
        ((0, 0, 0, 10), (0, 0, 0, 10), 0.0),  # no area
    ],
)
def test_iou(a, b, expected):
    assert iou(a, b) == pytest.approx(expected)
    assert iou(b, a) == pytest.approx(expected)


# --- Hit rate ---

REFERENCE = {0: (0, 0, 10, 10), 1: (20, 0, 30, 10), 3: (60, 0, 70, 10)}


def test_hit_rate_matches_by_lane_and_counts_false_positives():
    proposed = {0: (0, 0, 10, 10), 1: (25, 0, 35, 10), 2: (40, 0, 50, 10), 3: None, 7: (1, 1, 2, 2)}
    found = hit_rate(proposed, REFERENCE)
    assert (found.hits, found.n_ref) == (1, 3)
    assert found.rate == pytest.approx(1 / 3)
    assert found.false_positive_lanes == (2, 7)
    assert found.iou == {0: 1.0, 1: pytest.approx(1 / 3), 3: 0.0}


def test_false_positive_lanes_ascend_whatever_the_order_given():
    found = hit_rate({7: (0, 0, 1, 1), 2: (0, 0, 1, 1), 0: (0, 0, 10, 10)}, {0: (0, 0, 10, 10)})
    assert found.false_positive_lanes == (2, 7)


@pytest.mark.parametrize("iou_min", [0.0, -1.0])
def test_a_hit_needs_an_overlapping_box_at_any_minimum(iou_min):
    # With no minimum left, a box must still overlap its band: no box, a box
    # beside the band and an empty proposal are misses.
    reference = {0: (0, 0, 10, 10), 3: (60, 0, 70, 10)}
    assert hit_rate([None], reference, iou_min=iou_min).hits == 0
    assert hit_rate({0: (50, 50, 60, 60)}, reference, iou_min=iou_min).hits == 0
    assert hit_rate({0: (10, 0, 20, 10)}, reference, iou_min=iou_min).hits == 0  # edge contact
    assert hit_rate([], reference, iou_min=iou_min).rate == 0.0
    assert hit_rate({0: (9, 9, 20, 20)}, reference, iou_min=iou_min).hits == 1  # 1 px overlap


def test_a_box_on_the_neighbouring_lane_is_not_a_hit():
    # Lane 1's box sits exactly on lane 0's band: a miss for 0, a false positive.
    found = hit_rate({1: (0, 0, 10, 10)}, {0: (0, 0, 10, 10)})
    assert (found.hits, found.false_positive_lanes) == (0, (1,))


def test_hit_rate_takes_slots_in_lane_order():
    found = hit_rate([(0, 0, 10, 10), None, None, (60, 0, 70, 10)], REFERENCE)
    assert (found.hits, found.false_positive_lanes) == (2, ())
    assert found.iou[1] == 0.0  # lane 1 got no box


def test_iou_exactly_at_the_minimum_is_a_hit():
    assert hit_rate({0: (0, 0, 10, 10)}, {0: (0, 0, 10, 20)}).hits == 1
    assert hit_rate({0: (0, 0, 10, 10)}, {0: (0, 0, 10, 20)}, iou_min=0.51).hits == 0


def test_rate_is_nan_without_reference_bands():
    found = hit_rate({0: (0, 0, 1, 1)}, {})
    assert math.isnan(found.rate)
    assert (found.hits, found.n_ref, found.false_positive_lanes) == (0, 0, (0,))
    assert math.isnan(hit_rate([None, None], {}).rate)


# --- Row-box detection on the synthetic rows ---


def test_hit_rate_of_detection_on_the_synthetic_rows():
    # The fifteen bench51 rows: 91 bands, all found in their own lane.
    hits = n_ref = 0
    for case in bench_cases():
        found = evaluate_row(
            case.image,
            case.row,
            case.n_lanes,
            case.reference,
            background=estimate_background(case.image),
            dark_on_light=case.dark_on_light,
        )
        assert isinstance(found, HitRate)
        assert found.false_positive_lanes == (), case.name
        assert found.hits == found.n_ref == len(case.reference), case.name
        assert min(found.iou.values()) >= 0.6, case.name
        hits, n_ref = hits + found.hits, n_ref + found.n_ref
    assert (hits, n_ref) == (91, 91)


def test_size_rule_passes_through():
    case = bench_cases()[0]
    kwargs = {"background": estimate_background(case.image), "dark_on_light": True}
    by_max = evaluate_row(case.image, case.row, 6, case.reference, size_rule="max", **kwargs)
    assert by_max.hits == 6
    with pytest.raises(ValueError, match="size rule"):
        evaluate_row(case.image, case.row, 6, case.reference, size_rule="mean", **kwargs)


def test_iou_min_passes_through():
    case = bench_cases()[0]  # IoU 0.86 to 0.96 with the references
    kwargs = {"background": estimate_background(case.image), "dark_on_light": True}
    strict = evaluate_row(case.image, case.row, 6, case.reference, iou_min=0.99, **kwargs)
    assert (strict.hits, strict.n_ref) == (0, 6)
    assert evaluate_row(case.image, case.row, 6, case.reference, iou_min=0.8, **kwargs).hits == 6


def test_a_refused_detection_proposes_nothing():
    case = adversarial("box_omits_empty_first", 1000)  # flagged ambiguous_lanes
    found = evaluate_row(
        case.image,
        case.row,
        case.n_lanes,
        case.reference,
        background=estimate_background(case.image),
        dark_on_light=True,
    )
    assert (found.hits, found.n_ref, found.false_positive_lanes) == (0, 5, ())
    assert set(found.iou.values()) == {0.0}
