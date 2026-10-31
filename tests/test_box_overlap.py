# SPDX-License-Identifier: Apache-2.0
"""Boxes of two proteins that overlap (#208).

Boxes of different proteins on one image may overlap by at most half the
smaller box's area; more is refused (#114). Below that, each net counts the
pixels the two boxes share, and the results say so: one ``boxes_overlap``
warning per protein whose counted box (its first band, in a lane the set
includes) overlaps another protein's box, naming the other proteins, the lanes
and the share of the smaller box's area. It is computed from the stored boxes,
never stored.

The results tests build their projects from the model (no pixels): the
conftest sample's images, with proteins whose boxes the tests place. The
operations tests draw a synthetic blot 200 px wide and 80 px high, with two
bands 22 px apart on one row: β-catenin's 20x10 box is placed on the first,
at (50, 25, 70, 35), and GAPDH's boxes are placed, moved or resized against
it. Names use µ, α and β.
"""

import hashlib
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from conftest import (
    FakeClock,
    converted,
    make_project,
    sample_doc,
    sample_doc_v1,
    synthetic_blot,
    write_image_files,
    write_tiff,
)
from proteia.core import operations as ops
from proteia.core import results, storage
from proteia.core.model import BoxSize, ImageKind, Polarity, Project, Role
from proteia.core.operations import ErrorCode, LaneInput, OperationError, ProjectSession
from proteia.core.results import Level, compute_results
from test_checks import all_lanes_view
from test_mw_row import LOADING, calibrated
from test_mw_row import add as add_by_mw
from test_operations import boxes_of, lane_bands

OVERLAP = "boxes_overlap"
# The end of every overlap notice: one lane of the protein, or several.
ONE = (
    " of the smaller box's area: its net there also counts the pixels the boxes share;"
    " move a box so they do not overlap"
)
MANY = (
    " of the smaller box's area: its nets there also count the pixels the boxes share;"
    " move a box so they do not overlap"
)
# The same, of a notice naming two or more other proteins: their shares are each
# of the smaller box's area.
EACH_ONE = (
    ", each of the smaller box's area: its net there also counts the pixels the boxes share;"
    " move a box so they do not overlap"
)
EACH_MANY = (
    ", each of the smaller box's area: its nets there also count the pixels the boxes share;"
    " move a box so they do not overlap"
)


# --- Projects built from the model ---


@dataclass(frozen=True)
class Boxed:
    """A protein of a model-built project: its name, its boxes as (lane, x, y)
    or (lane, x, y, band index), its box size, its image and its role, and its
    not-detected records as (lane, (x0, y0, x1, y1))."""

    name: str
    boxes: Sequence[tuple[int, ...]]
    size: tuple[int, int] = (20, 10)
    image: str = "img-2"
    role: str = "target"
    undetected: Sequence[tuple[int, tuple[int, int, int, int]]] = ()


def project_with(*proteins: Boxed, lanes: int = 4, excluded: Collection[int] = ()) -> Project:
    """The conftest sample's images (img-2 and img-4 on one membrane, 340x150;
    img-6 on another), ``lanes`` lanes (vehicle and 10 µM in turn, the
    reference vehicle; ``excluded`` lanes include=no) and ``proteins``, ids in
    order: prot-100, prot-101, ...; band-200, band-201, ... Every net is 1000."""
    doc = sample_doc()  # as a schema-2 file holds it: converted below
    doc["next_id"] = 1000
    batch = doc["batch"]
    batch["lanes"] = [
        {
            "index": i,
            "label": "vehicle" if i % 2 == 0 else "10 µM",
            "sample": f"s{i}",
            "included": i not in excluded,
        }
        for i in range(lanes)
    ]
    batch["reference_condition"] = "vehicle"
    band_ids = iter(range(200, 1000))
    batch["proteins"] = [
        {
            "id": f"prot-{100 + k}",
            "name": protein.name,
            "role": protein.role,
            "image_id": protein.image,
            "box_size": {"width": protein.size[0], "height": protein.size[1]},
            "bands": [
                {
                    "id": f"band-{next(band_ids)}",
                    "lane_index": box[0],
                    "band_index": box[3] if len(box) > 3 else 0,
                    "box": {"x": box[1], "y": box[2]},
                    "net": 1000.0,
                    "background_level": 200.0,
                    "background_mode": "symmetric",
                    "background_spread": 0.0,
                    "source": "click",
                }
                for box in protein.boxes
            ],
            "undetected": [
                {
                    "lane_index": lane,
                    "reason": "below_detection_limit",
                    "snr": 2.5,
                    "threshold": 6.0,
                    "region": dict(zip(("x0", "y0", "x1", "y1"), region, strict=True)),
                    "source": "row_box",
                }
                for lane, region in protein.undetected
            ],
        }
        for k, protein in enumerate(proteins)
    ]
    return converted(doc)


def overlaps_of(notices: Sequence[results.Notice]) -> list[tuple[tuple[str, ...], tuple, str]]:
    """The overlap notices, in order: (protein ids, lane indices, message)."""
    return [(n.protein_ids, n.lane_indices, n.message) for n in notices if n.code == OVERLAP]


def overlap_notices(project: Project) -> list[tuple[tuple[str, ...], tuple, str]]:
    return overlaps_of(compute_results(project.batch).notices)


A, B, C = "prot-100", "prot-101", "prot-102"
# β-catenin's box in lane 1 of every results test, and GAPDH's around it.
AT = (0, 50, 25)


def beta(*boxes: tuple[int, ...], **kw) -> Boxed:
    return Boxed("β-catenin", boxes or (AT,), **kw)


def gapdh(*boxes: tuple[int, ...], **kw) -> Boxed:
    return Boxed("GAPDH", boxes, role="loading control", **kw)


# --- What counts as an overlap ---


@pytest.mark.parametrize(
    "at",
    [
        pytest.param((0, 80, 25), id="apart"),
        pytest.param((0, 70, 25), id="touching-on-the-right"),
        pytest.param((0, 30, 25), id="touching-on-the-left"),
        pytest.param((0, 50, 35), id="touching-below"),
        pytest.param((0, 50, 15), id="touching-above"),
        pytest.param((0, 70, 35), id="touching-at-a-corner"),
    ],
)
def test_boxes_apart_or_only_touching_give_no_notice(at):
    # Boxes are half-open: (50, 25, 70, 35) and (70, 25, 90, 35) share an edge
    # and no pixel.
    project = project_with(beta(), gapdh(at))
    res = compute_results(project.batch)
    assert overlaps_of(res.notices) == []
    assert res.all_lanes is None


@pytest.mark.parametrize(
    ("at", "share"),
    [
        pytest.param((0, 69, 34), "under 1%", id="one-pixel"),  # 1 of 200 px
        pytest.param((0, 62, 25), "40%", id="40%"),  # 8 columns of 20
        pytest.param((0, 60, 25), "50%", id="50%"),  # 10 columns of 20: half, accepted
        pytest.param((0, 31, 25), "5%", id="one-column-on-the-left"),  # 1 x 10 px
        pytest.param((0, 31, 16), "under 1%", id="one-pixel-top-left"),
        pytest.param((0, 54, 25), "80%", id="80%-in-an-old-file"),  # refused since #114
    ],
)
def test_two_proteins_boxes_that_share_pixels_give_each_a_warning(at, share):
    project = project_with(beta(), gapdh(at))
    res = compute_results(project.batch)
    assert overlaps_of(res.notices) == [
        ((A,), (0,), f"'β-catenin' overlaps 'GAPDH' in lane 1 by {share}{ONE}"),
        ((B,), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by {share}{ONE}"),
    ]
    assert {n.level for n in res.notices if n.code == OVERLAP} == {Level.WARNING}
    assert all((n.conditions, n.image_ids) == ((), ()) for n in res.notices if n.code == OVERLAP)


def test_a_one_pixel_corner_share_is_under_1_percent_of_the_smaller_box():
    # (69, 34, 89, 44) shares its top-left pixel with (50, 25, 70, 35).
    project = project_with(beta(), gapdh((0, 69, 34)))
    [(_, _, message), _] = overlap_notices(project)
    assert message == f"'β-catenin' overlaps 'GAPDH' in lane 1 by under 1%{ONE}"


@pytest.mark.parametrize(
    ("small", "large", "at", "share"),
    [
        # 37x27 boxes sharing 25x20 px: 500 of 999, just past half: never "50%".
        pytest.param((37, 27), (37, 27), (0, 62, 32), "50.1%", id="just-past-half"),
        # A 10x5 box wholly inside a 20x10 one: all of the smaller box.
        pytest.param((20, 10), (10, 5), (0, 55, 27), "100%", id="wholly-inside"),
    ],
)
def test_an_old_files_overlap_past_half_reads_past_half(small, large, at, share):
    project = project_with(beta(size=small), gapdh(at, size=large))
    assert overlap_notices(project) == [
        ((A,), (0,), f"'β-catenin' overlaps 'GAPDH' in lane 1 by {share}{ONE}"),
        ((B,), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by {share}{ONE}"),
    ]


@pytest.mark.parametrize(
    ("shared", "smaller", "words"),
    [
        (1, 200, "under 1%"),
        (199, 20000, "under 1%"),
        (1, 101, "under 1%"),
        (1, 100, "1%"),  # exactly 1%
        (2, 200, "1%"),
        (3, 200, "2%"),  # 1.5%: half up
        (5, 200, "3%"),  # 2.5%: half up, not to the even 2%
        (29, 1000, "3%"),  # 2.9%
        (145, 1000, "15%"),  # 14.5%
        (80, 200, "40%"),
        (99, 200, "50%"),  # 49.5%: at most half, so it reads 50%
        (100, 200, "50%"),
        (101, 200, "50.5%"),  # past half
        (1001, 2000, "50.1%"),
        (500, 999, "50.1%"),
        (102, 200, "51%"),
        (160, 200, "80%"),
        (200, 200, "100%"),
    ],
)
def test_the_share_is_written_as_a_whole_percent_never_reading_past_or_within_half_wrongly(
    shared, smaller, words
):
    assert results.share_words(shared, smaller) == words


def test_boxes_on_different_images_share_no_pixels():
    # The same coordinates on another image of the membrane (a reprobe), and
    # on an image of another membrane.
    for image in ("img-4", "img-6"):
        project = project_with(beta(), gapdh((0, 50, 25), image=image))
        assert overlap_notices(project) == [], image


def test_a_not_detected_slot_over_another_proteins_box_is_no_overlap():
    # β-catenin was not detected in lane 2: its slot there covers GAPDH's
    # lane-2 box, but it has no box and no value. Its box in lane 1 overlaps
    # GAPDH's lane-1 box: only that lane is named.
    project = project_with(
        beta(AT, undetected=[(1, (90, 20, 130, 40))]),
        gapdh((0, 62, 25), (1, 100, 25)),
    )
    assert overlap_notices(project) == [
        ((A,), (0,), f"'β-catenin' overlaps 'GAPDH' in lane 1 by 40%{ONE}"),
        ((B,), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by 40%{ONE}"),
    ]
    alone = project_with(
        beta((0, 10, 25), undetected=[(1, (90, 20, 130, 40))]), gapdh((1, 100, 25))
    )
    assert overlap_notices(alone) == []


# --- Lanes, proteins and shares named ---


def test_boxes_of_two_lanes_that_overlap_name_both_lanes():
    # β-catenin's lane-1 box overlaps GAPDH's box of lane 2: each protein's
    # notice is on its own lane, and names the other's.
    project = project_with(beta(), gapdh((1, 62, 25)))
    assert overlap_notices(project) == [
        (
            (A,),
            (0,),
            f"'β-catenin' overlaps 'GAPDH' in lane 1 (the box of 'GAPDH' in lane 2) by 40%{ONE}",
        ),
        (
            (B,),
            (1,),
            "'GAPDH' overlaps 'β-catenin' in lane 2 (the box of 'β-catenin' in lane 1) by"
            f" 40%{ONE}",
        ),
    ]


def test_one_box_over_two_boxes_of_another_protein_names_both():
    # GAPDH's lane-1 box covers 6 of β-catenin's 20 columns (30%); its lane-2
    # box, left of it, 14 columns of β-catenin's bottom row (7%).
    project = project_with(beta(), gapdh((0, 64, 25), (1, 44, 34)))
    assert overlap_notices(project) == [
        (
            (A,),
            (0,),
            "'β-catenin' overlaps 'GAPDH' in lane 1 by 30% and in lane 1 (the box of 'GAPDH'"
            f" in lane 2) by 7%{ONE}",
        ),
        (
            (B,),
            (0, 1),
            "'GAPDH' overlaps 'β-catenin' in lane 1 by 30% and in lane 2 (the box of"
            f" 'β-catenin' in lane 1) by 7%{MANY}",
        ),
    ]


@pytest.mark.parametrize(
    "xs",
    [
        pytest.param((10, 110, 210), id="left-to-right"),
        # Lanes stored right to left: the boxes are met lane 3 first, and the
        # lanes of one share still take the place of their lowest lane.
        pytest.param((210, 110, 10), id="right-to-left"),
    ],
)
@pytest.mark.parametrize(
    ("offsets", "words"),
    [
        # (dx, dy) of GAPDH's box from β-catenin's in lanes 1, 2, 3: 12 columns
        # by 2 rows (12%), 8 columns (40%).
        pytest.param(
            [(8, 8), (12, 0), (8, 8)], "in lanes 1, 3 by 12% and in lane 2 by 40%", id="12-40-12"
        ),
        pytest.param(
            [(12, 0), (8, 8), (8, 8)], "in lane 1 by 40% and in lanes 2, 3 by 12%", id="40-12-12"
        ),
        pytest.param([(12, 0)] * 3, "in lanes 1, 2, 3 by 40%", id="40-40-40"),
    ],
)
def test_the_lanes_of_one_share_are_named_together(offsets, words, xs):
    project = project_with(
        beta(*((lane, x, 25) for lane, x in enumerate(xs))),
        gapdh(
            *(
                (lane, x + dx, 25 + dy)
                for lane, (x, (dx, dy)) in enumerate(zip(xs, offsets, strict=True))
            )
        ),
    )
    assert overlap_notices(project) == [
        ((A,), (0, 1, 2), f"'β-catenin' overlaps 'GAPDH' {words}{MANY}"),
        ((B,), (0, 1, 2), f"'GAPDH' overlaps 'β-catenin' {words}{MANY}"),
    ]


def test_a_box_over_two_proteins_names_both_in_the_proteins_order():
    # µ-calpain's box covers β-catenin's lower left (10 x 5 px, 25%), GAPDH's
    # its right (40%); GAPDH and µ-calpain do not touch.
    calpain = Boxed("µ-calpain", [(0, 40, 30)])
    project = project_with(beta(), gapdh((0, 62, 25)), calpain)
    assert overlap_notices(project) == [
        (
            (A,),
            (0,),
            "'β-catenin' overlaps 'GAPDH' in lane 1 by 40%; and 'µ-calpain' in lane 1 by"
            f" 25%{EACH_ONE}",
        ),
        ((B,), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by 40%{ONE}"),
        ((C,), (0,), f"'µ-calpain' overlaps 'β-catenin' in lane 1 by 25%{ONE}"),
    ]
    # µ-calpain listed before GAPDH: so are its notice and its name.
    swapped = project_with(beta(), calpain, gapdh((0, 62, 25)))
    assert overlap_notices(swapped) == [
        (
            (A,),
            (0,),
            "'β-catenin' overlaps 'µ-calpain' in lane 1 by 25%; and 'GAPDH' in lane 1 by"
            f" 40%{EACH_ONE}",
        ),
        ((B,), (0,), f"'µ-calpain' overlaps 'β-catenin' in lane 1 by 25%{ONE}"),
        ((C,), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by 40%{ONE}"),
    ]


def test_a_protein_over_three_proteins_names_them_apart_in_the_proteins_order():
    # β-catenin's boxes of lanes 1 to 4: GAPDH's over those of lanes 1 to 3
    # (12%, 40%, 12%, as above), and in lane 4 α-tubulin's at its lower left
    # (10 x 5 px, 25%) and µ-calpain's at its right (40%), apart from each
    # other. The proteins' order is neither their names' nor their shares'.
    project = project_with(
        beta((0, 10, 25), (1, 110, 25), (2, 210, 25), (3, 300, 25)),
        gapdh((0, 18, 33), (1, 122, 25), (2, 218, 33)),
        Boxed("α-tubulin", [(3, 290, 30)]),
        Boxed("µ-calpain", [(3, 312, 25)]),
    )
    assert overlap_notices(project) == [
        (
            (A,),
            (0, 1, 2, 3),
            "'β-catenin' overlaps 'GAPDH' in lanes 1, 3 by 12% and in lane 2 by 40%;"
            f" 'α-tubulin' in lane 4 by 25%; and 'µ-calpain' in lane 4 by 40%{EACH_MANY}",
        ),
        (
            (B,),
            (0, 1, 2),
            f"'GAPDH' overlaps 'β-catenin' in lanes 1, 3 by 12% and in lane 2 by 40%{MANY}",
        ),
        ((C,), (3,), f"'α-tubulin' overlaps 'β-catenin' in lane 4 by 25%{ONE}"),
        (("prot-103",), (3,), f"'µ-calpain' overlaps 'β-catenin' in lane 4 by 40%{ONE}"),
    ]


def test_proteins_on_other_images_are_left_out_of_the_search():
    # A third protein's box at the same place on another image changes nothing.
    project = project_with(beta(), gapdh((0, 62, 25)), Boxed("µ-calpain", [AT], image="img-4"))
    assert [ids for ids, _, _ in overlap_notices(project)] == [(A,), (B,)]


def test_an_overlap_is_found_whatever_order_the_boxes_are_stored_in():
    # GAPDH's box lies far right of β-catenin's, µ-calpain's (stored last)
    # left of it, over its lower left: found all the same.
    calpain = Boxed("µ-calpain", [(0, 40, 30)])
    project = project_with(beta(), gapdh((2, 200, 25)), calpain)
    assert overlap_notices(project) == [
        ((A,), (0,), f"'β-catenin' overlaps 'µ-calpain' in lane 1 by 25%{ONE}"),
        ((C,), (0,), f"'µ-calpain' overlaps 'β-catenin' in lane 1 by 25%{ONE}"),
    ]


def test_the_boxes_over_one_box_are_named_by_lane_then_by_the_other_proteins_lane():
    # β-catenin's box of lane 2 lies over GAPDH's of lane 2 (30%) and, at its
    # lower left, GAPDH's of lane 1 (7%): lane 1's comes first.
    project = project_with(beta((1, 50, 25)), gapdh((1, 64, 25), (0, 44, 34)))
    assert overlap_notices(project) == [
        (
            (A,),
            (1,),
            "'β-catenin' overlaps 'GAPDH' in lane 2 (the box of 'GAPDH' in lane 1) by 7% and in"
            f" lane 2 by 30%{ONE}",
        ),
        (
            (B,),
            (0, 1),
            "'GAPDH' overlaps 'β-catenin' in lane 1 (the box of 'β-catenin' in lane 2) by 7% and"
            f" in lane 2 by 30%{MANY}",
        ),
    ]


def test_the_lanes_of_a_notice_are_in_order_past_lane_8():
    # Lanes 2 and 9 (indices 1 and 8) of ten.
    project = project_with(
        beta((1, 10, 25), (8, 210, 25)), gapdh((1, 22, 25), (8, 222, 25)), lanes=10
    )
    assert overlap_notices(project) == [
        ((A,), (1, 8), f"'β-catenin' overlaps 'GAPDH' in lanes 2, 9 by 40%{MANY}"),
        ((B,), (1, 8), f"'GAPDH' overlaps 'β-catenin' in lanes 2, 9 by 40%{MANY}"),
    ]


@pytest.mark.parametrize(
    ("first", "second", "its_share"),
    [
        # 40% (8 columns) at the right, 5% (10 x 1 px) at the upper left.
        pytest.param((0, 62, 25, 0), (0, 40, 16, 1), "40%", id="right-first-band-the-larger"),
        pytest.param((0, 40, 16, 0), (0, 62, 25, 1), "5%", id="right-second-band-the-larger"),
        # 40% at the left, 5% (10 x 1 px) at the lower right.
        pytest.param((0, 38, 25, 0), (0, 60, 34, 1), "40%", id="left-first-band-the-larger"),
        pytest.param((0, 60, 34, 0), (0, 38, 25, 1), "5%", id="left-second-band-the-larger"),
    ],
)
def test_two_boxes_of_one_lane_over_a_box_name_the_lane_once_with_the_larger_share(
    first, second, its_share
):
    # An old file: GAPDH's first and second bands of lane 1 both over
    # β-catenin's box, by 40% and 5%, the larger left or right of the other.
    # Lane 1 is named once, by 40%. Of GAPDH's, only its first band's box is
    # counted.
    project = project_with(beta(), gapdh(first, second))
    assert overlap_notices(project) == [
        ((A,), (0,), f"'β-catenin' overlaps 'GAPDH' in lane 1 by 40%{ONE}"),
        ((B,), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by {its_share}{ONE}"),
    ]


def test_a_box_of_a_second_band_gets_no_notice_of_its_own():
    # GAPDH has only a second band in lane 1, over β-catenin's box: β-catenin's
    # net counts its pixels; GAPDH's value there is not this box's.
    project = project_with(beta(), gapdh((0, 62, 25, 1)))
    assert overlap_notices(project) == [
        ((A,), (0,), f"'β-catenin' overlaps 'GAPDH' in lane 1 by 40%{ONE}"),
    ]


def test_names_are_written_as_typed():
    project = project_with(
        Boxed("α-actinin µ2", [AT]), Boxed("β-catenin (p-S33)", [(0, 62, 25)], role="target")
    )
    assert overlap_notices(project) == [
        ((A,), (0,), f"'α-actinin µ2' overlaps 'β-catenin (p-S33)' in lane 1 by 40%{ONE}"),
        ((B,), (0,), f"'β-catenin (p-S33)' overlaps 'α-actinin µ2' in lane 1 by 40%{ONE}"),
    ]


def test_lane_numbers_follow_the_stored_lane_not_the_position():
    # β-catenin's box of lane 3 lies left of GAPDH's box of lane 1: lanes are
    # stored indices, whichever way they run across the image.
    project = project_with(beta((2, 50, 25)), gapdh((0, 62, 25)))
    assert overlap_notices(project) == [
        (
            (A,),
            (2,),
            f"'β-catenin' overlaps 'GAPDH' in lane 3 (the box of 'GAPDH' in lane 1) by 40%{ONE}",
        ),
        (
            (B,),
            (0,),
            "'GAPDH' overlaps 'β-catenin' in lane 1 (the box of 'β-catenin' in lane 3) by"
            f" 40%{ONE}",
        ),
    ]


# --- Their place among the other notices ---


def test_the_overlap_notices_come_right_after_the_background_notices():
    # The order of the Checks list, the bundle README's Checks and the record's
    # notices: β-catenin's ring in lane 2 is cut short (a background notice),
    # β-catenin and GAPDH overlap in lane 1, GAPDH was not detected in lane 3,
    # and µ-calpain, alone on img-6, brings that image's two-point calibration.
    doc = project_with(
        beta(AT, (1, 150, 25)),
        gapdh((0, 62, 25), undetected=[(2, (250, 25, 270, 35))]),
        Boxed("µ-calpain", [AT], image="img-6"),
    ).model_dump(mode="json")
    doc["batch"]["proteins"][0]["rows"][0]["bands"][1]["background_mode"] = "asymmetric"
    res = compute_results(Project.model_validate(doc).batch)
    assert [(n.code.value, n.protein_ids) for n in res.notices] == [
        ("clipping_not_checked", (A,)),
        ("clipping_not_checked", (B,)),
        ("clipping_not_checked", (C,)),
        ("background_fallback", (A,)),
        ("boxes_overlap", (A,)),
        ("boxes_overlap", (B,)),
        ("below_detection", (B,)),
        ("calibration_two_points", (C,)),
    ]


# --- Excluded lanes: the two result sets ---


def test_an_overlap_in_an_excluded_lane_is_in_the_all_lanes_set_only():
    project = project_with(beta(AT, (1, 150, 25)), gapdh((0, 62, 25), (1, 160, 25)), excluded=[0])
    res = compute_results(project.batch)
    assert res.all_lanes is not None
    assert overlaps_of(res.notices) == [
        ((A,), (1,), f"'β-catenin' overlaps 'GAPDH' in lane 2 by 50%{ONE}"),
        ((B,), (1,), f"'GAPDH' overlaps 'β-catenin' in lane 2 by 50%{ONE}"),
    ]
    assert overlaps_of(res.all_lanes.notices) == [
        ((A,), (0, 1), f"'β-catenin' overlaps 'GAPDH' in lane 1 by 40% and in lane 2 by 50%{MANY}"),
        ((B,), (0, 1), f"'GAPDH' overlaps 'β-catenin' in lane 1 by 40% and in lane 2 by 50%{MANY}"),
    ]
    alone = project_with(beta(AT), gapdh((0, 62, 25)), excluded=[0])
    res = compute_results(alone.batch)
    assert overlaps_of(res.notices) == []
    assert overlaps_of(res.all_lanes.notices) == [
        ((A,), (0,), f"'β-catenin' overlaps 'GAPDH' in lane 1 by 40%{ONE}"),
        ((B,), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by 40%{ONE}"),
    ]


def test_a_box_in_an_excluded_lane_still_overlaps_a_counted_box():
    # GAPDH's box of excluded lane 2 overlaps β-catenin's of lane 1: β-catenin's
    # net counts the shared pixels in both sets; GAPDH's lane-2 value is counted
    # in the all-lanes set only, so its notice is there alone.
    project = project_with(beta(), gapdh((1, 62, 25)), excluded=[1])
    res = compute_results(project.batch)
    mine = (
        (A,),
        (0,),
        f"'β-catenin' overlaps 'GAPDH' in lane 1 (the box of 'GAPDH' in lane 2) by 40%{ONE}",
    )
    theirs = (
        (B,),
        (1,),
        f"'GAPDH' overlaps 'β-catenin' in lane 2 (the box of 'β-catenin' in lane 1) by 40%{ONE}",
    )
    assert overlaps_of(res.notices) == [mine]
    assert overlaps_of(res.all_lanes.notices) == [theirs]  # the same as the applied set's: once
    assert overlaps_of(all_lanes_view(res)) == [theirs, mine]


def test_the_all_lanes_view_keeps_an_overlap_its_own_notice_of_the_protein_also_names():
    # β-catenin's boxes of lanes 1 and 4 (excluded) overlap GAPDH's (lane 1)
    # and µ-calpain's (lane 4). The all-lanes set's notice of β-catenin names
    # both; the page's all-lanes view lists, per code and protein, the set's
    # own notice in place of the applied set's, so that notice must still
    # name GAPDH.
    calpain = Boxed("µ-calpain", [(3, 312, 25)])
    project = project_with(
        beta(AT, (3, 300, 25)), gapdh((0, 62, 25)), calpain, lanes=4, excluded=[3]
    )
    res = compute_results(project.batch)
    assert overlaps_of(res.notices) == [
        ((A,), (0,), f"'β-catenin' overlaps 'GAPDH' in lane 1 by 40%{ONE}"),
        ((B,), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by 40%{ONE}"),
    ]
    view = overlaps_of(all_lanes_view(res))
    assert view == [
        (
            (A,),
            (0, 3),
            "'β-catenin' overlaps 'GAPDH' in lane 1 by 40%; and 'µ-calpain' in lane 4 by"
            f" 40%{EACH_MANY}",
        ),
        ((C,), (3,), f"'µ-calpain' overlaps 'β-catenin' in lane 4 by 40%{ONE}"),
        ((B,), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by 40%{ONE}"),
    ]


# --- Nothing is stored ---


@pytest.mark.parametrize(
    ("at", "noted", "content_hash", "file_sha256"),
    [
        # The content hash and the SHA-256 of project.json, as Proteia wrote
        # them before #208 (at schema 2: the converted project maps back to them).
        pytest.param(
            (0, 80, 25),
            False,
            "77b423f25cb253b96bbf5c2d8b14509b4ebe3707051a478c4277beaef323b7eb",
            "920557743bdf6eccf1d1b6a1fbbeed58c88da8d1490e1d8a86faac7bf523bb55",
            id="apart",
        ),
        pytest.param(
            (0, 62, 25),
            True,
            "2c31c01cf5823038bb0835ed11262e76e6af9858f1fdf507ffae20e50d1b352f",
            "bbbe9c7577ca76600b5b6f96c5ba32d10e99ae8abeb4c9bec639084278230fb9",
            id="overlapping",
        ),
    ],
)
def test_an_overlap_is_computed_not_stored(at, noted, content_hash, file_sha256):
    project = project_with(beta(), gapdh(at))
    assert bool(overlap_notices(project)) is noted
    earlier = storage._EARLIER_CONTENT[2](storage.content_document(project))
    assert storage.document_hash(earlier) == content_hash
    written = storage.document_bytes({**earlier, "next_id": 1000, "log": []})
    assert hashlib.sha256(written).hexdigest() == file_sha256
    data = storage.project_to_json(project)
    assert storage.project_from_json(data) == project
    assert b"overlap" not in data


# --- The operations: a blot with two bands on one row ---


def blot_session(tmp_path: Path) -> tuple[ProjectSession, str, str, str]:
    """The blot, three lanes, β-catenin's box placed at (50, 25, 70, 35) in
    lane 1 and GAPDH with none; (the session, β-catenin, GAPDH, β-catenin's box)."""
    s = ops.new_project(tmp_path / "blot µ", autosave=None, clock=FakeClock())
    pixels = synthetic_blot((80, 200), [(60, 30, 6.0, 2.5, 30000.0), (82, 30, 6.0, 2.5, 30000.0)])
    source = write_tiff(tmp_path / "sources" / "two bands α.tif", pixels)
    with source.open("rb") as f:
        image = ops.import_image(
            s,
            f,
            "two bands α.tif",
            kind=ImageKind.CHEMILUMINESCENCE,
            polarity=Polarity.DARK_ON_LIGHT,
        )
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM"), LaneInput("50 µM")])
    size = BoxSize(width=20, height=10)
    a = ops.add_protein(s, "β-catenin", Role.TARGET, image, box_size=size)
    b = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image, box_size=size)
    box = ops.place_box(s, a, 60, 30, lane_index=0, grow=False)
    return s, a, b, box


def rect_of(s: ProjectSession, band_id: str) -> tuple[int, int, int, int]:
    protein, band = s.project.batch.find_band(band_id)
    return band.box.rect(s.project.batch.boxes_of(protein).box_size)


def notices_now(s: ProjectSession) -> list[tuple[tuple[str, ...], tuple, str]]:
    return overlaps_of(ops.compute(s).notices)


def same_lane(a: str, b: str, share: str) -> list[tuple[tuple[str, ...], tuple, str]]:
    """The two notices of an overlap in lane 1, β-catenin's and GAPDH's."""
    return [
        ((a,), (0,), f"'β-catenin' overlaps 'GAPDH' in lane 1 by {share}{ONE}"),
        ((b,), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by {share}{ONE}"),
    ]


def across_lanes(a: str, b: str, share: str) -> list[tuple[tuple[str, ...], tuple, str]]:
    """The two notices of β-catenin's lane-1 box over GAPDH's lane-2 box."""
    return [
        (
            (a,),
            (0,),
            f"'β-catenin' overlaps 'GAPDH' in lane 1 (the box of 'GAPDH' in lane 2) by"
            f" {share}{ONE}",
        ),
        (
            (b,),
            (1,),
            f"'GAPDH' overlaps 'β-catenin' in lane 2 (the box of 'β-catenin' in lane 1) by"
            f" {share}{ONE}",
        ),
    ]


REFUSED = (
    "the box would overlap the box of 'β-catenin' in lane 1 by more than 50% of the smaller"
    " box's area: two proteins' boxes would measure the same band"
)


def refused_unchanged(s: ProjectSession, call) -> OperationError:
    """A refusal over β-catenin's box: nothing changes, no overlap is noted."""
    before = s.project
    with pytest.raises(OperationError) as info:
        call()
    assert s.project is before
    assert (info.value.code, info.value.detail["cause"]) == (ErrorCode.OVERLAP, "other_protein")
    assert notices_now(s) == []
    return info.value


@pytest.mark.parametrize(
    ("centre", "rect", "share"),
    [
        pytest.param((80, 30), (70, 25, 90, 35), None, id="touching"),
        pytest.param((79, 39), (69, 34, 89, 44), "under 1%", id="one-pixel"),
        pytest.param((72, 30), (62, 25, 82, 35), "40%", id="40%"),
        pytest.param((70, 30), (60, 25, 80, 35), "50%", id="50%"),
    ],
)
def test_a_box_placed_over_part_of_another_proteins_box_is_noted(tmp_path, centre, rect, share):
    s, a, b, _ = blot_session(tmp_path)
    placed = ops.place_box(s, b, *centre, lane_index=0, grow=False)
    assert rect_of(s, placed) == rect
    assert notices_now(s) == ([] if share is None else same_lane(a, b, share))


def test_a_box_placed_over_more_than_half_of_another_proteins_box_is_refused(tmp_path):
    # 17 columns by 6 rows of 200 px: 51%.
    s, _, b, _ = blot_session(tmp_path)
    error = refused_unchanged(s, lambda: ops.place_box(s, b, 63, 34, lane_index=0, grow=False))
    assert str(error) == REFUSED
    assert error.detail["covered"][0]["share"] == 0.51


def test_a_box_moved_over_another_proteins_box_is_noted_and_moved_off_it_is_not(tmp_path):
    # GAPDH's lane-1 box placed at the right of the blot, then moved over
    # β-catenin's, past half of it (refused) and off it again: the notices
    # follow every move, and moving the box to touch it only clears them.
    s, a, b, _ = blot_session(tmp_path)
    moved = ops.place_box(s, b, 160, 30, lane_index=0, grow=False)
    assert notices_now(s) == []
    for rect, share in (
        ((69, 34, 89, 44), "under 1%"),
        ((62, 25, 82, 35), "40%"),
        ((60, 25, 80, 35), "50%"),
        ((70, 25, 90, 35), None),  # touching: the next step the notice gives
        ((62, 25, 82, 35), "40%"),
    ):
        ops.move_box(s, moved, rect)
        assert rect_of(s, moved) == rect
        assert notices_now(s) == ([] if share is None else same_lane(a, b, share)), rect
    error = refused_unchanged_after(s, lambda: ops.move_box(s, moved, (53, 29, 73, 39)))
    assert str(error) == REFUSED
    assert notices_now(s) == same_lane(a, b, "40%")
    ops.move_box(s, moved, (140, 25, 160, 35))
    assert notices_now(s) == []


def refused_unchanged_after(s: ProjectSession, call) -> OperationError:
    """A refusal that changes nothing, from a state that may hold an overlap."""
    before, noted = s.project, notices_now(s)
    with pytest.raises(OperationError) as info:
        call()
    assert s.project is before and notices_now(s) == noted
    assert info.value.code is ErrorCode.OVERLAP
    return info.value


@pytest.mark.parametrize(
    ("centre_y", "width", "share"),
    [
        pytest.param(30, 40, None, id="touching"),  # (70, 25, 110, 35)
        pytest.param(39, 42, "under 1%", id="one-pixel"),  # (69, 34, 111, 44)
        pytest.param(30, 56, "40%", id="40%"),  # (62, 25, 118, 35)
        pytest.param(30, 60, "50%", id="50%"),  # (60, 25, 120, 35)
    ],
)
def test_a_box_size_that_reaches_over_another_proteins_box_is_noted(
    tmp_path, centre_y, width, share
):
    # GAPDH's box of lane 2 centred 30 px right of β-catenin's, widened.
    s, a, b, _ = blot_session(tmp_path)
    ops.place_box(s, b, 90, centre_y, lane_index=1, grow=False)
    assert notices_now(s) == []
    ops.set_box_size(s, b, BoxSize(width=width, height=10))
    assert notices_now(s) == ([] if share is None else across_lanes(a, b, share))


def test_a_box_size_over_more_than_half_of_another_proteins_box_is_refused(tmp_path):
    s, _, b, _ = blot_session(tmp_path)
    ops.place_box(s, b, 90, 30, lane_index=1, grow=False)
    error = refused_unchanged(s, lambda: ops.set_box_size(s, b, BoxSize(width=62, height=10)))
    assert str(error) == (
        "at the box size 62x10, boxes of 'GAPDH' would overlap the box of 'β-catenin' in"
        " lane 1 by more than 50% of the smaller box's area: two proteins' boxes would"
        " measure the same band"
    )


@pytest.mark.parametrize(
    ("centre_y", "across", "share"),
    [
        pytest.param(39, 1, "under 1%", id="one-pixel"),  # (69, 34, 91, 44)
        pytest.param(30, 8, "40%", id="40%"),  # (62, 25, 98, 35)
        pytest.param(30, 10, "50%", id="50%"),  # (60, 25, 100, 35)
    ],
)
def test_a_box_padding_that_reaches_over_another_proteins_box_is_noted(
    tmp_path, centre_y, across, share
):
    # GAPDH's box of lane 2 touching β-catenin's, padded left and right: the
    # padding's answer names the box it now overlaps, as before, and so do
    # the results.
    s, a, b, box = blot_session(tmp_path)
    ops.place_box(s, b, 80, centre_y, lane_index=1, grow=False)
    change = ops.set_box_padding(s, b, across=across)
    assert change.overlapping == (box,)
    assert notices_now(s) == across_lanes(a, b, share)


def test_a_box_padding_over_more_than_half_of_another_proteins_box_is_refused(tmp_path):
    s, _, b, _ = blot_session(tmp_path)
    ops.place_box(s, b, 79, 30, lane_index=1, grow=False)  # (69, 25, 89, 35): 1 column
    refused_unchanged_after(s, lambda: ops.set_box_padding(s, b, across=10))


def test_a_box_grown_from_a_click_over_part_of_another_proteins_box_is_noted(tmp_path):
    # β-catenin's box moved 8 px right, to (58, 25, 78, 35); GAPDH's band
    # clicked: its grown box, (76, 28, 89, 33), shares 2 columns by 5 rows of
    # its 65 px (15%).
    s, a, b, box = blot_session(tmp_path)
    ops.move_box(s, box, (58, 25, 78, 35))
    grown = ops.place_box(s, b, 82, 30, lane_index=0, grow=True)
    assert rect_of(s, grown) == (76, 28, 89, 33)
    assert notices_now(s) == same_lane(a, b, "15%")


def test_a_row_placed_by_its_mw_over_part_of_another_proteins_box_is_noted(tmp_path):
    # α-tubulin's row placed by its MW, cleared, and placed again over a box
    # of µ-calpain's put across the bottom 6 of its 16 rows in lane 1 (86 x 6
    # of 1376 px: 37.5%, written 38%).
    blot = calibrated(tmp_path)
    s = blot.session
    loading = add_by_mw(blot, LOADING, "α-tubulin")
    ops.detect_mw_row(s, loading)
    size = boxes_of(s, loading).box_size
    first = {lane: band.box.rect(size) for lane, band in lane_bands(s, loading).items()}
    assert first[0] == (162, 247, 248, 263)
    ops.clear_boxes(s, loading)
    calpain = ops.add_protein(s, "µ-calpain", Role.TARGET, blot.blot, box_size=size)
    ops.place_box(s, calpain, 205, 265, lane_index=0, grow=False)  # (162, 257, 248, 273)
    ops.detect_mw_row(s, loading)
    again = {lane: band.box.rect(size) for lane, band in lane_bands(s, loading).items()}
    assert again == first
    assert notices_now(s) == [
        ((loading,), (0,), f"'α-tubulin' overlaps 'µ-calpain' in lane 1 by 38%{ONE}"),
        ((calpain,), (0,), f"'µ-calpain' overlaps 'α-tubulin' in lane 1 by 38%{ONE}"),
    ]


# --- The notices follow every change ---


def test_the_notices_follow_undo_redo_and_every_edit_of_the_boxes(tmp_path):
    s, a, b, box = blot_session(tmp_path)
    placed = ops.place_box(s, b, 72, 30, lane_index=0, grow=False)
    assert notices_now(s) == same_lane(a, b, "40%")
    ops.undo(s)
    assert notices_now(s) == []
    ops.redo(s)
    assert notices_now(s) == same_lane(a, b, "40%")
    ops.requantify(s)
    assert notices_now(s) == same_lane(a, b, "40%")
    # The lane GAPDH's box is in, and β-catenin's name.
    ops.set_box_lane(s, placed, 2)
    ops.edit_protein(s, a, name="β-catenin µ")
    assert notices_now(s) == [
        (
            (a,),
            (0,),
            f"'β-catenin µ' overlaps 'GAPDH' in lane 1 (the box of 'GAPDH' in lane 3) by 40%{ONE}",
        ),
        (
            (b,),
            (2,),
            f"'GAPDH' overlaps 'β-catenin µ' in lane 3 (the box of 'β-catenin µ' in lane 1) by"
            f" 40%{ONE}",
        ),
    ]
    ops.remove_box(s, placed)
    assert notices_now(s) == []
    ops.undo(s)
    ops.clear_boxes(s, b)
    assert notices_now(s) == []
    ops.undo(s)
    ops.remove_protein(s, a)
    assert notices_now(s) == []
    ops.undo(s)
    ops.remove_box(s, box)
    assert notices_now(s) == []


def test_the_notices_follow_the_include_flags(tmp_path):
    s, a, b, _ = blot_session(tmp_path)
    ops.place_box(s, b, 72, 30, lane_index=0, grow=False)
    ops.set_lanes(s, [LaneInput("vehicle", included=False), LaneInput("10 µM"), LaneInput("50 µM")])
    res = ops.compute(s)
    assert overlaps_of(res.notices) == []
    assert overlaps_of(res.all_lanes.notices) == same_lane(a, b, "40%")


def test_an_old_project_with_overlapping_boxes_opens_with_the_notices(tmp_path):
    # A file written before #114 (or by hand), its boxes overlapping by 80%:
    # it opens as it is, unchanged and not migrated, with the notices; moving
    # the box to under half of it is accepted.
    s, a, b, _ = blot_session(tmp_path)
    placed = ops.place_box(s, b, 160, 30, lane_index=0, grow=False)
    ops.save(s)
    path = s.folder / storage.PROJECT_FILE
    doc = json.loads(path.read_text(encoding="utf-8"))
    for protein in doc["batch"]["proteins"]:
        if protein["id"] == b:
            protein["rows"][0]["bands"][0]["box"]["x"] = 54  # (54, 25, 74, 35): 16 of 20
    path.write_text(json.dumps(doc), encoding="utf-8")
    data = path.read_bytes()
    s.close()
    s = ops.open_project(s.folder, autosave=None, clock=FakeClock())
    assert notices_now(s) == same_lane(a, b, "80%")
    assert [entry.action for entry in s.project.log].count("migrate") == 0
    assert path.read_bytes() == data
    ops.move_box(s, placed, (62, 25, 82, 35))
    assert notices_now(s) == same_lane(a, b, "40%")


def test_a_schema_1_project_with_overlapping_boxes_opens_with_the_notices(tmp_path):
    # The sample before #83 with GAPDH moved onto β-catenin's image, its lane-1
    # box over the right 8 of the 24 columns of β-catenin's: one third.
    doc = sample_doc_v1()
    beta_protein, _, gapdh_protein = doc["batch"]["proteins"]
    gapdh_protein["image_id"] = beta_protein["image_id"]
    gapdh_protein["bands"] = [b for b in gapdh_protein["bands"] if b["lane_index"] == 0]
    gapdh_protein["bands"][0]["box"] = {"x": 34, "y": 43}  # β-catenin's: (18, 43, 42, 57)
    folder = tmp_path / "old µ"
    write_image_files(folder, make_project())  # the same images
    (folder / storage.PROJECT_FILE).write_text(json.dumps(doc), encoding="utf-8")
    s = ops.open_project(folder, autosave=None, clock=FakeClock())
    assert notices_now(s) == [
        (("prot-7",), (0,), f"'β-catenin' overlaps 'GAPDH' in lane 1 by 33%{ONE}"),
        (("prot-9",), (0,), f"'GAPDH' overlaps 'β-catenin' in lane 1 by 33%{ONE}"),
    ]
    assert [entry.action for entry in s.project.log][-1] == "migrate"
