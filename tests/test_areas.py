# SPDX-License-Identifier: Apache-2.0
"""Membrane areas drawn on an image (#243): several membranes on one image.

Each membrane records where it lies on each of its images: the whole image,
or an outline (a rectangle, turned by an angle about its centre). A box, a
not-detected record and a calibration point belong to the one area that holds
its centre; finding bands along a row reads only the row's area; lane
proposals and row checks read only the boxes of the same area; and each
membrane keeps its own calibration on an image it shares. An image linked to
its marker holds the same areas, so an area operation on one acts on both.
The angle is recorded only: no net and no apparent MW changes with it.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import make_project, synthetic_blot
from proteia.core import operations as ops
from proteia.core.model import (
    Arena,
    CalibrationPointSource,
    ImageKind,
    Outline,
    Project,
    Role,
    UnknownIdError,
    apply_change,
)
from proteia.core.operations import ErrorCode, LaneInput, OperationError, RingContext
from proteia.core.project import lane_anchors
from proteia.core.results import compute_results
from proteia.core.storage import content_hash
from test_operations import assert_nets_current, import_blot, session_on

STRIPS = "two strips µ.tif"
CHEMI_MARKER = CalibrationPointSource.CHEMILUMINESCENCE_MARKER
STRIP_EDGE = CalibrationPointSource.STRIP_EDGE
LANE_X = (40, 100, 160)


def two_strips() -> np.ndarray:
    """200x100: two strips of one row each, at y 24 and 76, in three lanes."""
    bands = [(x, y, 6.0, 2.5, 30000.0) for y in (24, 76) for x in LANE_X]
    return synthetic_blot((100, 200), bands)


def refused(call, code: ErrorCode) -> OperationError:
    with pytest.raises(OperationError) as info:
        call()
    assert info.value.code is code, info.value
    return info.value


def area_on(s, image_id: str, number: int = 1) -> str:
    """The id of the ``number``-th membrane's area on an image (membrane order)."""
    return s.project.batch.arenas_on(image_id)[number - 1][1].id


def one_image(tmp_path: Path):
    """The two strips imported (img-1 on mem-2, its whole-image area arena-4),
    three lanes, and a target on it."""
    s = session_on(tmp_path)
    image = import_blot(s, two_strips(), STRIPS)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM"), LaneInput("50 µM")])
    return s, image


def split(tmp_path: Path):
    """:func:`one_image`, split: membrane 1 (mem-2) on the top strip
    (arena-4, y 0-50), membrane 2 (mem-5) on the bottom one (arena-6, y
    50-100), a target on each (β-catenin on the top, p53 µ on the bottom)."""
    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    ops.add_arena(s, image, (0, 50, 200, 100))
    top = ops.add_protein(s, "β-catenin", Role.TARGET, arena_id="arena-4")
    bottom = ops.add_protein(s, "p53 µ", Role.TARGET, arena_id="arena-6")
    return s, image, top, bottom


def nets(s) -> dict[str, float]:
    batch = s.project.batch
    return {b.id: b.net for p in batch.proteins for b in batch.boxes_of(p).bands}


def box_all(s, protein: str, y: int) -> list[str]:
    return [
        ops.place_box(s, protein, x, y, lane_index=lane, grow=False)
        for lane, x in enumerate(LANE_X)
    ]


# --- Drawing the first areas ---


def test_second_area_needs_first_outlined(tmp_path):
    s, image = one_image(tmp_path)
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    ops.place_box(s, protein, 40, 24, lane_index=0, grow=False)
    error = refused(
        lambda: ops.add_arena(s, image, (0, 50, 200, 100)), ErrorCode.ARENA_COVERS_IMAGE
    )
    assert str(error) == (
        f"membrane 1's area covers all of {STRIPS!r}; draw membrane 1's outline first, then"
        " add the next membrane's area"
    )
    # The step taken, the same area is drawn.
    first = ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    assert first.arena_ids == ("arena-4",)
    second = ops.add_arena(s, image, (0, 50, 200, 100))
    assert (second.arena_ids, second.membrane_id, second.new_membrane) == (
        ("arena-9",),
        "mem-8",
        True,
    )
    batch = s.project.batch
    [(_, edited)] = [(m, a) for m, a in batch.arenas_on(image) if a.id == "arena-4"]
    assert edited.source.value == "manual"
    assert batch.lane_table_of(batch.membranes[1]).id == "lt-3"  # the first area's membrane's
    # Membrane 1's boxes stay off the bottom strip.
    error = refused(
        lambda: ops.place_box(s, protein, 100, 76, lane_index=1, grow=False),
        ErrorCode.OUT_OF_ARENA,
    )
    assert str(error) == (
        f"(100, 76) lies outside membrane 1's area on {STRIPS!r}; click inside it, or redraw"
        " the area"
    )
    ops.place_box(s, protein, 100, 24, lane_index=1, grow=False)  # inside: placed
    [entry] = [e for e in s.project.log if e.action == "add_arena"]
    assert entry.params == {
        "image_id": image,
        "outline": [0, 50, 200, 100],
        "angle_deg": 0.0,
        "membrane_id": "mem-8",
        "new_membrane": True,
        "arena_ids": ["arena-9"],
        "lane_table_id": "lt-3",
    }


def test_a_flat_outline_is_refused_then_a_drag_draws_it(tmp_path):
    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    for flat in ((10, 60, 10, 90), (10, 60, 90, 60)):
        error = refused(lambda flat=flat: ops.add_arena(s, image, flat), ErrorCode.INVALID_INPUT)
        assert str(error) == (
            "a membrane area needs a width and a height of at least 1 pixel; drag across the"
            " membrane"
        )
    # A drag from its bottom right corner gives both.
    made = ops.add_arena(s, image, (90, 99, 10, 60))
    [(_, arena)] = [(m, a) for m, a in s.project.batch.arenas_on(image) if a.id in made.arena_ids]
    assert arena.outline == Outline(x0=10, y0=60, x1=90, y1=99)


def test_area_past_edge(tmp_path):
    s, image = one_image(tmp_path)
    # 30 px past the right edge, its centre (190, 25) inside: stored as drawn.
    ops.edit_arena(s, "arena-4", outline=(150, 0, 230, 50))
    [(_, arena)] = s.project.batch.arenas_on(image)
    assert arena.outline == Outline(x0=150, y0=0, x1=230, y1=50)
    assert arena.bounds(s.project.batch.find_image(image)) == (150, 0, 230, 50)
    error = refused(lambda: ops.add_arena(s, image, (180, 50, 270, 100)), ErrorCode.OUT_OF_IMAGE)
    assert str(error) == (
        f"the area's centre (225, 75) lies outside the 200x100 image {STRIPS!r}; draw it over"
        " the membrane"
    )
    ops.add_arena(s, image, (0, 50, 140, 100))


def test_an_area_turns_at_most_45_degrees(tmp_path):
    s, image = one_image(tmp_path)
    for angle in (45.0, -45.0, 45, -0.0):
        ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50), angle_deg=angle)
        [(_, arena)] = s.project.batch.arenas_on(image)
        assert math.copysign(1.0, arena.angle_deg) == 1.0 or arena.angle_deg < 0
        assert arena.angle_deg == float(angle)
    for angle, words in ((45.1, "45.1"), (-45.1, "-45.1"), (90, "90")):
        error = refused(
            lambda angle=angle: ops.edit_arena(s, "arena-4", angle_deg=angle),
            ErrorCode.INVALID_INPUT,
        )
        assert str(error) == (
            f"turn a membrane area by at most 45° either way ({words}° given); for a membrane"
            " lying on its side, save the image turned a quarter turn in another program and"
            " import that file"
        )
    for bad in (math.nan, "3", True):
        refused(
            lambda bad=bad: ops.edit_arena(s, "arena-4", angle_deg=bad), ErrorCode.INVALID_INPUT
        )


def test_a_second_area_of_a_membrane_on_an_image_is_refused(tmp_path):
    s, image, _, _ = split(tmp_path)
    error = refused(
        lambda: ops.add_arena(s, image, (0, 0, 100, 50), membrane_id="mem-2"),
        ErrorCode.ARENA_EXISTS,
    )
    assert str(error) == (
        f"membrane 1 already has an area on {STRIPS!r}; edit that area, or choose another membrane"
    )
    assert error.ids == ("arena-4",)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 100, 50))


def test_an_area_for_an_existing_membrane_on_another_image(tmp_path):
    s, image = one_image(tmp_path)
    other = import_blot(s, two_strips(), "second.tif")  # mem-6, its area arena-8
    ops.edit_arena(s, area_on(s, other), outline=(0, 50, 200, 100))
    made = ops.add_arena(s, other, (0, 0, 200, 50), membrane_id="mem-2")
    assert (made.membrane_id, made.new_membrane) == ("mem-2", False)
    batch = s.project.batch
    assert [a.image_id for a in batch.membranes[0].arenas] == [image, other]
    assert [i.id for i in batch.iter_images()] == [image, other]
    with pytest.raises(UnknownIdError):
        ops.add_arena(s, other, (0, 0, 200, 50), membrane_id="mem-99")


def test_an_outline_that_strands_boxes_names_them(tmp_path):
    s, image = one_image(tmp_path)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    bands = box_all(s, gapdh, 24)
    ops.add_calibration_point(s, image, 30.0, 100.0, STRIP_EDGE, x=180.0, snap=False)
    error = refused(
        lambda: ops.edit_arena(s, "arena-4", outline=(0, 0, 70, 100)), ErrorCode.ARENA_IN_USE
    )
    assert str(error) == (
        "the new outline would leave the boxes of 'GAPDH' in lanes 2, 3 and the 100 kDa ladder"
        f" mark on {STRIPS!r} outside membrane 1's area; draw the outline around them, or move"
        " or remove them first"
    )
    assert error.ids == ("arena-4", bands[1], bands[2])
    assert error.detail["band_ids"] == bands[1:]
    # The next step taken: the things listed go, then the same outline is drawn.
    ops.remove_box(s, bands[1])
    ops.remove_box(s, bands[2])
    ops.remove_calibration_point(s, image, 100.0)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 70, 100))


def test_an_outline_that_strands_a_record_names_it(tmp_path):
    s, image = one_image(tmp_path)
    p53 = ops.add_protein(s, "p53", Role.TARGET, image)
    placed = ops.detect_row_boxes(s, p53, (10, 64, 190, 90))  # the bottom strip's row
    assert placed.band_ids
    from proteia.core.model import Region, UndetectedBand
    from test_operations import plant

    def record(draft: Project) -> None:
        own = draft.batch.boxes_of(draft.batch.find_protein(p53))
        own.bands = [b for b in own.bands if b.lane_index != 2]
        own.undetected = [
            UndetectedBand(
                lane_index=2,
                reason="below_detection_limit",
                snr=1.0,
                threshold=6.0,
                region=Region(x0=150, y0=66, x1=170, y1=86),
                source="row_box",
            )
        ]

    plant(s, record)
    error = refused(
        lambda: ops.edit_arena(s, "arena-4", outline=(0, 50, 140, 100)), ErrorCode.ARENA_IN_USE
    )
    assert "a not-detected mark of 'p53' in lane 3" in str(error)
    assert error.detail["undetected"] == [[p53, 2, 0]]


def test_the_only_area_of_an_image_stays(tmp_path):
    s, image = one_image(tmp_path)
    error = refused(lambda: ops.remove_arena(s, "arena-4"), ErrorCode.ARENA_REQUIRED)
    assert str(error) == (
        f"{STRIPS!r} needs at least one membrane area; remove the image instead, or give its"
        " area to another membrane"
    )
    moved = ops.assign_arena(s, "arena-4", None)
    assert (moved.membrane_id, moved.removed_membrane, moved.renumbered) == ("mem-5", "mem-2", None)
    assert [m.id for m in s.project.batch.membranes] == ["mem-5"]
    ops.remove_image(s, image)
    assert s.project.batch.membranes == []


def test_a_protein_on_an_image_of_several_membranes_names_its_area(tmp_path):
    s, image, _, _ = split(tmp_path)
    error = refused(
        lambda: ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image), ErrorCode.ARENA_REQUIRED
    )
    assert str(error) == f"{STRIPS!r} holds 2 membranes; choose the membrane the protein is on"
    assert error.ids == ("arena-4", "arena-6")
    protein = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, arena_id="arena-6")
    batch = s.project.batch
    assert batch.arena_of(batch.find_protein(protein)).id == "arena-6"
    assert s.project.log[-1].params["arena_id"] == "arena-6"
    for both in ({"arena_id": "arena-6"}, {}):
        refused(
            lambda both=both: ops.add_protein(s, "x", Role.TARGET, image if both else None, **both),
            ErrorCode.INVALID_INPUT,
        )


def test_marks_on_an_image_of_several_membranes_name_their_area(tmp_path):
    s, image, _, _ = split(tmp_path)
    error = refused(
        lambda: ops.add_calibration_point(s, image, 70.0, 50.0, STRIP_EDGE, x=5.0, snap=False),
        ErrorCode.ARENA_REQUIRED,
    )
    assert str(error) == f"{STRIPS!r} holds 2 membranes; choose the membrane to calibrate"
    update = ops.add_calibration_point(s, "arena-6", 70.0, 50.0, STRIP_EDGE, x=5.0, snap=False)
    assert update.point["image_id"] == image
    batch = s.project.batch
    assert [len(m.calibration.points) for m in batch.membranes] == [0, 1]
    assert s.project.log[-1].params["arena_id"] == "arena-6"
    # A mark outside the area named, or inside the other, is refused.
    error = refused(
        lambda: ops.add_calibration_point(s, "arena-6", 20.0, 40.0, STRIP_EDGE, x=5.0, snap=False),
        ErrorCode.OUT_OF_ARENA,
    )
    assert str(error) == (
        f"(5, 20) lies outside membrane 2's area on {STRIPS!r}; click inside it, or redraw the area"
    )
    for call in (
        lambda: ops.edit_calibration_point(s, image, 50.0, y=80.0),
        lambda: ops.remove_calibration_point(s, image, 50.0),
        lambda: ops.clear_calibration(s, image),
        lambda: ops.propose_ladder(s, image, 5.0),
        lambda: ops.snap_ladder(s, image, 5.0, [70.0]),
        lambda: ops.set_ladder_points(s, image, "left", [(70.0, 50.0)], x=5.0),
    ):
        refused(call, ErrorCode.ARENA_REQUIRED)
    ops.edit_calibration_point(s, "arena-6", 50.0, y=80.0)
    refused(lambda: ops.edit_calibration_point(s, "arena-6", 50.0, y=30.0), ErrorCode.OUT_OF_ARENA)
    ops.remove_calibration_point(s, "arena-6", 50.0)
    assert [len(m.calibration.points) for m in s.project.batch.membranes] == [0, 0]


def test_a_click_outside_the_area_is_refused(tmp_path):
    s, image, top, bottom = split(tmp_path)
    error = refused(
        lambda: ops.place_box(s, bottom, 40, 24, lane_index=0, grow=False),
        ErrorCode.OUT_OF_ARENA,
    )
    assert str(error) == (
        f"(40, 24) lies outside membrane 2's area on {STRIPS!r}; click inside it, or redraw"
        " the area"
    )
    # The outline's centre, (100, 75), is inside: placed, as a seed click.
    ops.place_box(s, bottom, 100, 75, lane_index=1, grow=True)


def test_a_box_moved_out_of_its_area_is_refused(tmp_path):
    s, image, top, _ = split(tmp_path)
    band = ops.place_box(s, top, 40, 24, lane_index=0, grow=False)
    rect = s.project.batch.find_band(band)[1].box.rect(
        s.project.batch.boxes_of(s.project.batch.find_protein(top)).box_size
    )
    down = (rect[0], rect[1] + 40, rect[2], rect[3] + 40)
    error = refused(lambda: ops.move_box(s, band, down), ErrorCode.OUT_OF_ARENA)
    assert str(error) == (
        "the box's centre would lie outside membrane 1's area; move it inside, or redraw the area"
    )
    ops.move_box(s, band, (rect[0], rect[1] + 10, rect[2], rect[3] + 10))


def test_boxes_resized_out_of_their_area_are_refused(tmp_path):
    s, image, top, _ = split(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 30))
    band = ops.place_box(s, top, 40, 24, lane_index=0, grow=False)  # 3 x 3, centred at y 24.5
    size = s.project.batch.boxes_of(s.project.batch.find_protein(top)).box_size
    del band
    # A size that pushes the box down from the image's top keeps its centre;
    # one that the image's bottom edge cannot hold is checked as moved.
    error = refused(
        lambda: ops.set_box_size(s, top, ops.BoxSize(width=size.width, height=99)),
        ErrorCode.OUT_OF_ARENA,
    )
    assert "the box's centre would lie outside membrane 1's area" in str(error)


def test_a_row_box_off_its_area_is_refused(tmp_path):
    s, image, top, bottom = split(tmp_path)
    error = refused(lambda: ops.detect_row_boxes(s, top, (10, 60, 190, 90)), ErrorCode.OUT_OF_ARENA)
    assert str(error) == (
        "the row box lies outside membrane 1's area; drag across that membrane's lanes"
    )
    placed = ops.detect_row_boxes(s, top, (10, 12, 190, 38))
    assert all(placed.band_ids)
    placed = ops.detect_row_boxes(s, bottom, (10, 64, 190, 90))
    assert all(placed.band_ids)
    assert_nets_current(s)


def side_by_side(tmp_path: Path, right: tuple[int, int, int, int] = (130, 0, 200, 50)):
    """The top strip in two areas side by side: membrane 1's over lanes 1
    and 2 (x 0-130), membrane 2's over lane 3 (``right``), and the bottom
    strip a third's; a target on membrane 1's."""
    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 130, 50))
    ops.add_arena(s, image, right)
    ops.add_arena(s, image, (0, 52, 200, 100))
    return s, image, ops.add_protein(s, "β-catenin", Role.TARGET, arena_id="arena-4")


def test_lane_slots_of_a_row_outside_its_area_are_refused(tmp_path):
    # Lane 3 lies in membrane 2's area: the row reads none of it, and its
    # slot there is refused, as the row's lanes run past the area.
    s, image, protein = side_by_side(tmp_path)
    error = refused(
        lambda: ops.detect_row_boxes(s, protein, (10, 12, 190, 38)), ErrorCode.OUT_OF_ARENA
    )
    assert str(error) == (
        "the row's lane slots in lane 3 lie outside membrane 1's area; drag the row inside the"
        " area, or redraw the area"
    )
    assert error.detail["lanes"] == [2]
    # Redrawn over every lane, the same row is read.
    ops.remove_arena(s, "arena-6")
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    placed = ops.detect_row_boxes(s, protein, (10, 12, 190, 38))
    assert all(placed.band_ids)


def test_bands_a_row_finds_outside_its_area_are_refused(tmp_path, monkeypatch):
    # Were the row to read the whole image, lane 3's band, in membrane 2's
    # area, would be found: refused, naming its lane.
    s, image, protein = side_by_side(tmp_path)
    monkeypatch.setattr(ops, "_masked", lambda array, image, area, window: array)
    error = refused(
        lambda: ops.detect_row_boxes(s, protein, (10, 12, 190, 38)), ErrorCode.OUT_OF_ARENA
    )
    assert str(error) == (
        "the row found bands outside membrane 1's area in lane 3; drag the row inside the area,"
        " or redraw the area"
    )
    assert error.detail["lanes"] == [2]
    assert len(error.detail["rects"]) == 1


def test_a_row_box_in_two_areas(tmp_path):
    # Membrane 1's area over every lane, membrane 2's over lane 2 (x 90-130):
    # the row's box there lies in both.
    s, image, protein = side_by_side(tmp_path, right=(90, 0, 130, 50))
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    error = refused(
        lambda: ops.detect_row_boxes(s, protein, (10, 12, 190, 38)), ErrorCode.IN_TWO_ARENAS
    )
    assert "lies inside both membrane 1's and membrane 2's areas" in str(error)


def test_an_area_given_to_a_membrane_already_there_is_refused(tmp_path):
    s, image, top, _ = split(tmp_path)
    error = refused(lambda: ops.assign_arena(s, "arena-4", "mem-5"), ErrorCode.ARENA_EXISTS)
    assert str(error) == (
        f"membrane 2 already has an area on {STRIPS!r}; choose another membrane, or a new one"
    )
    refused(lambda: ops.assign_arena(s, "arena-4", "mem-2"), ErrorCode.INVALID_INPUT)
    moved = ops.assign_arena(s, "arena-4", None)
    assert moved.membrane_id == "mem-11"
    assert moved.moved_proteins == (top,)
    assert moved.removed_membrane == "mem-2"
    assert moved.renumbered == "membrane 2 is now membrane 1"
    assert [m.id for m in s.project.batch.membranes] == ["mem-5", "mem-11"]
    assert s.project.batch.membrane_of_protein(s.project.batch.find_protein(top)).id == "mem-11"


def test_an_unknown_area_is_worded_for_another_window(tmp_path):
    s, _ = one_image(tmp_path)
    for call in (
        lambda: ops.edit_arena(s, "arena-99", angle_deg=1.0),
        lambda: ops.remove_arena(s, "arena-99"),
        lambda: ops.assign_arena(s, "arena-99", None),
        lambda: ops.add_protein(s, "x", Role.TARGET, arena_id="arena-99"),
        lambda: ops.add_calibration_point(s, "arena-99", 1.0, 50.0, STRIP_EDGE, x=1.0),
    ):
        with pytest.raises(UnknownIdError) as info:
            call()
        assert str(info.value) == (
            "that membrane area no longer exists; it may have been removed in another window"
        )
        assert info.value.detail == {"arena_id": "arena-99"}


# --- Overlapping areas ---


def test_overlapping_areas_redraw_then_place(tmp_path):
    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 60))
    ops.add_arena(s, image, (0, 40, 200, 100))  # 20 px of overlap, y 40-60
    bottom = ops.add_protein(s, "p53", Role.TARGET, arena_id="arena-6")
    error = refused(
        lambda: ops.place_box(s, bottom, 100, 50, lane_index=1, grow=False),
        ErrorCode.IN_TWO_ARENAS,
    )
    assert str(error) == (
        f"(100, 50) lies inside both membrane 1's and membrane 2's areas on {STRIPS!r}; redraw"
        " one of them so they do not overlap there"
    )
    # Outside the overlap, as before; then a box in it once it is gone.
    ops.place_box(s, bottom, 100, 76, lane_index=1, grow=False)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 45))
    ops.place_box(s, bottom, 40, 50, lane_index=0, grow=False)
    # An outline that would take that box into a second area is refused.
    error = refused(
        lambda: ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 55)), ErrorCode.IN_TWO_ARENAS
    )
    assert "lies inside both membrane 1's and membrane 2's areas" in str(error)


def test_box_on_area_edge(tmp_path):
    s, image, top, bottom = split(tmp_path)
    # A 3x3 box centred at y 49.5 (clicked at y 49) belongs to the top strip (y 0-50).
    ops.place_box(s, top, 40, 49, lane_index=0, grow=False)
    refused(
        lambda: ops.place_box(s, bottom, 100, 49, lane_index=1, grow=False), ErrorCode.OUT_OF_ARENA
    )
    # A click on the shared edge lies in both areas (each holds its edge).
    refused(
        lambda: ops.place_box(s, bottom, 100, 50, lane_index=1, grow=False),
        ErrorCode.IN_TWO_ARENAS,
    )
    ops.place_box(s, bottom, 100, 51, lane_index=1, grow=False)  # centred at 51.5


# --- The angle (recorded only) ---


def corner_project(tmp_path: Path):
    """1000x600, one membrane over the whole image, boxes centred near
    each corner and a ladder mark at (3, 595)."""
    s = session_on(tmp_path)
    bands = [(x, y, 3.0, 2.0, 20000.0) for x, y in ((12, 10), (988, 10), (12, 590), (988, 590))]
    image = import_blot(s, synthetic_blot((600, 1000), bands), "corners.tif")
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(4)])
    protein = ops.add_protein(
        s, "corners", Role.TARGET, image, box_size=ops.BoxSize(width=8, height=6)
    )
    for lane, (x, y) in enumerate(((12, 10), (988, 10), (12, 590), (988, 590))):
        ops.place_box(s, protein, x, y, lane_index=lane, grow=False)
    ops.add_calibration_point(s, image, 595.0, 50.0, STRIP_EDGE, x=3.0, snap=False)
    ops.add_calibration_point(s, image, 5.0, 100.0, STRIP_EDGE, x=3.0, snap=False)
    return s, image, protein


def test_angle_on_whole_image_area_keeps_corners(tmp_path):
    s, image, protein = corner_project(tmp_path)
    before = s.project.batch.model_copy(deep=True)
    results = compute_results(s.project.batch).model_dump()
    for angle in (3.0, -45.0, 45.0):
        ops.edit_arena(s, "arena-4", angle_deg=angle)
        batch = s.project.batch
        assert batch.arenas_on(image)[0][1].angle_deg == angle
        assert [p.rows for p in batch.proteins] == [p.rows for p in before.proteins]
        assert compute_results(batch).model_dump() == results
    # A whole-image outline turned 3° would leave the corner box out.
    doc = s.project.model_dump(mode="json")
    doc["batch"]["membranes"][0]["arenas"][0]["outline"] = {
        "x0": 0,
        "y0": 0,
        "x1": 1000,
        "y1": 600,
    }
    doc["batch"]["membranes"][0]["arenas"][0]["angle_deg"] = 3.0
    with pytest.raises(ValueError, match="lies outside its membrane area arena-4"):
        Project.model_validate(doc)


def test_angle_changes_no_number(tmp_path):
    s, image, top, bottom = split(tmp_path)
    box_all(s, top, 24)
    box_all(s, bottom, 76)
    ops.add_calibration_point(s, "arena-6", 52.0, 100.0, STRIP_EDGE, x=100.0, snap=False)
    ops.add_calibration_point(s, "arena-6", 98.0, 30.0, STRIP_EDGE, x=100.0, snap=False)
    before = s.project
    results = compute_results(before.batch).model_dump()
    reads = []
    real = s.pixels

    def counted(image_id, **kwargs):
        reads.append(image_id)
        return real(image_id, **kwargs)

    s.pixels = counted
    ops.edit_arena(s, "arena-6", angle_deg=3.7)
    after = s.project.batch
    assert reads == []  # no ring reads anything new
    assert after.find_arena("arena-6")[1].angle_deg == 3.7
    for protein in before.batch.proteins:
        assert after.find_protein(protein.id).rows[0].bands == protein.rows[0].bands
    assert after.membranes[1].calibration == before.batch.membranes[1].calibration
    assert compute_results(after).model_dump() == results
    # The content changed in the area alone.
    edited = after.model_copy(deep=True)
    edited.find_arena("arena-6")[1].angle_deg = 0.0
    edited.find_arena("arena-6")[1].source = before.batch.find_arena("arena-6")[1].source
    assert edited == before.batch


def test_angle_is_stored(tmp_path):
    s, image, _, _ = split(tmp_path)
    ops.edit_arena(s, "arena-6", angle_deg=-2.5)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 49), angle_deg=1.25)
    ops.save(s)
    reopened = ops.open_project(s.folder)
    angles = [a.angle_deg for _, a in reopened.project.batch.arenas_on(image)]
    assert angles == [1.25, -2.5]
    assert s.project.log[-1].params == {
        "arena_id": "arena-4",
        "outline": [0, 0, 200, 49],
        "angle_deg": 1.25,
        "arena_ids": ["arena-4"],
    }
    # The same outline and angle again change nothing.
    count = len(s.project.log)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 49), angle_deg=1.25)
    assert len(s.project.log) == count
    reopened.close()


def test_whole_image_only_while_alone(tmp_path):
    s, image, _, _ = split(tmp_path)
    error = refused(lambda: ops.edit_arena(s, "arena-4", outline=None), ErrorCode.INVALID_INPUT)
    assert str(error) == (
        f"{STRIPS!r} holds 2 membranes' areas, so none covers the whole image; draw this one's"
        " outline instead"
    )
    ops.remove_arena(s, "arena-6")
    ops.edit_arena(s, "arena-4", outline=None)
    assert s.project.batch.find_arena("arena-4")[1].outline is None


# --- Removing an area, and the membranes' numbers ---


def test_remove_area_cascade(tmp_path):
    s, image, top, bottom = split(tmp_path)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, arena_id="arena-6")
    ops.edit_protein(s, bottom, loading_control_ids=[gapdh])
    ops.edit_protein(s, top, loading_control_ids=[gapdh])
    top_bands = box_all(s, top, 24)
    bottom_bands = box_all(s, bottom, 76)
    ops.add_calibration_point(s, "arena-6", 52.0, 100.0, STRIP_EDGE, x=5.0, snap=False)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 48))
    before = nets(s)
    removal = ops.remove_arena(s, "arena-6")
    cascade = removal.cascade
    assert cascade.removed == (
        "arena-6",
        bottom,
        gapdh,
        s.project.log[-1].params["removed"][3],
        s.project.log[-1].params["removed"][4],
        *bottom_bands,
        "mem-5",
    )
    assert cascade.detached_targets == (top,)
    assert cascade.unfitted_membranes == ()
    assert removal.renumbered is None  # membrane 2 was the last
    batch = s.project.batch
    assert [m.id for m in batch.membranes] == ["mem-2"]
    # The rings of the bands that stay no longer leave out the boxes gone.
    moved = {band_id for band_id, _, _ in removal.remeasured}
    assert moved == {b for b in top_bands if nets(s)[b] != before[b]}
    assert_nets_current(s)
    restored = ops.undo(s)
    assert "arena-6" in restored.restored
    assert nets(s) == before


def test_labels_after_removal_are_told(tmp_path):
    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 50, 100))
    for x0 in (50, 100, 150):
        ops.add_arena(s, image, (x0, 0, x0 + 50, 100))
    ids = [a.id for _, a in s.project.batch.arenas_on(image)]
    removal = ops.remove_arena(s, ids[1])
    assert removal.renumbered == "membranes 3 and 4 are now membranes 2 and 3"
    assert [
        f"Membrane {s.project.batch.membrane_number(m)}" for m in s.project.batch.membranes
    ] == [
        "Membrane 1",
        "Membrane 2",
        "Membrane 3",
    ]


# --- Giving an area to another membrane ---


def test_assign_area_moves_rows_and_points(tmp_path):
    s, image = one_image(tmp_path)
    other = import_blot(s, two_strips(), "by mistake.tif", membrane_id="mem-2")
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, other)
    box_all(s, protein, 24)
    ops.add_calibration_point(s, other, 10.0, 100.0, STRIP_EDGE, x=5.0, snap=False)
    ops.add_calibration_point(s, other, 90.0, 20.0, STRIP_EDGE, x=5.0, snap=False)
    mws = {
        b.id: b.apparent_mw
        for b in s.project.batch.boxes_of(s.project.batch.find_protein(protein)).bands
    }
    assert all(mw is not None for mw in mws.values())
    third = import_blot(s, two_strips(), "B.tif")  # mem-N
    target = s.project.batch.membranes_on(third)[0].id
    moved = ops.assign_arena(s, area_on(s, other), target)
    assert moved.moved_points == 2
    assert moved.moved_proteins == (protein,)
    assert moved.removed_membrane is None
    batch = s.project.batch
    membrane = batch.membrane_of_protein(batch.find_protein(protein))
    assert membrane.id == target
    assert [p.image_id for p in membrane.calibration.points] == [other, other]
    assert batch.membranes[0].calibration.points == []
    # The same points on the same image: the same curve, the same MWs.
    assert {b.id: b.apparent_mw for b in batch.boxes_of(batch.find_protein(protein)).bands} == mws
    assert moved.curves_changed == ()
    assert [i.id for i in batch.iter_images()] == [image, third, other]
    entry = s.project.log[-1]
    assert entry.action == "assign_arena"
    assert entry.params["from_membrane_id"] == "mem-2"
    assert len(entry.params["moved_points"]) == 2


# --- Linked images hold the same areas ---


def marker_project(tmp_path: Path):
    """An exposure E with three membranes' areas (thirds of it), and a visible
    marker M imported onto membrane 1 (its whole-image area, nothing on it)."""
    s = session_on(tmp_path)
    exposure = import_blot(s, two_strips(), "E.tif")
    ops.set_lanes(s, [LaneInput("a"), LaneInput("b"), LaneInput("c")])
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 33))
    ops.add_arena(s, exposure, (0, 33, 200, 66))
    ops.add_arena(s, exposure, (0, 66, 200, 100))
    marker = import_blot(
        s, two_strips(), "M marker.tif", kind=ImageKind.VISIBLE_MARKER, membrane_id="mem-2"
    )
    return s, exposure, marker


def test_marker_file_three_membranes(tmp_path):
    s, exposure, marker = marker_project(tmp_path)
    update = ops.set_marker_image(s, exposure, marker)
    batch = s.project.batch
    on_marker = batch.arenas_on(marker)
    assert [m.id for m, _ in on_marker] == ["mem-2", "mem-5", "mem-7"]
    assert update.adopted_arena_ids == tuple(a.id for _, a in on_marker)
    for (m, a), (_, e) in zip(on_marker, batch.arenas_on(exposure), strict=True):
        assert (a.outline, a.angle_deg) == (e.outline, e.angle_deg)
        assert batch.group_of(m, exposure) == frozenset({exposure, marker})
    assert s.project.log[-1].params["adopted_arena_ids"] == list(update.adopted_arena_ids)
    # Then an edit of membrane 2's outline on E redraws it on M too, in one change.
    count = len(s.project.log)
    change = ops.edit_arena(s, "arena-6", outline=(0, 30, 200, 66))
    assert len(change.arena_ids) == 2 and len(s.project.log) == count + 1
    batch = s.project.batch
    assert batch.arenas_on(marker)[1][1].outline == Outline(x0=0, y0=30, x1=200, y1=66)
    # Each membrane its own curve: marks on membrane 3's area of M.
    ops.add_calibration_point(
        s, batch.arenas_on(marker)[2][1].id, 70.0, 100.0, STRIP_EDGE, x=5.0, snap=False
    )
    ops.add_calibration_point(
        s, batch.arenas_on(marker)[2][1].id, 95.0, 30.0, STRIP_EDGE, x=5.0, snap=False
    )
    fits = [ops.calibration_fit(s.project.batch, m, exposure) for m in s.project.batch.membranes]
    assert [fit is None for fit in fits] == [True, True, False]
    ops.undo(s)
    ops.undo(s)
    ops.undo(s)
    ops.undo(s)
    assert s.project.batch.arenas_on(marker)[0][1].outline is None  # back to before the link


def test_marker_adoption_refusals(tmp_path):
    s, exposure, marker = marker_project(tmp_path)
    # (i) a ladder mark on M outside membrane 1's area as E holds it.
    ops.add_calibration_point(s, marker, 80.0, 100.0, STRIP_EDGE, x=5.0, snap=False)
    error = refused(lambda: ops.set_marker_image(s, exposure, marker), ErrorCode.ARENA_IN_USE)
    assert str(error) == (
        "the new outline would leave the 100 kDa ladder mark on 'M marker.tif' outside membrane"
        " 1's area; draw the outline around them, or move or remove them first"
    )
    ops.remove_calibration_point(s, marker, 100.0)
    # (ii) M holding an area of a membrane E does not show.
    fourth = import_blot(s, two_strips(), "D.tif")
    ops.edit_arena(s, area_on(s, fourth), outline=(0, 0, 200, 50))
    ops.edit_arena(s, area_on(s, marker), outline=(0, 0, 200, 50))
    ops.add_arena(s, marker, (0, 50, 200, 100), membrane_id=s.project.batch.membranes[-1].id)
    error = refused(lambda: ops.set_marker_image(s, exposure, marker), ErrorCode.INVALID_INPUT)
    assert str(error) == (
        "'M marker.tif' holds an area of membrane 4, which 'E.tif' does not show; linked images"
        " show the same membranes, so give that area to one of 'E.tif''s membranes or remove it"
        " first"
    )
    ops.remove_arena(s, area_on(s, marker, 2))
    ops.set_marker_image(s, exposure, marker)
    # (iii) M linked from E already: a second exposure takes the group's areas.
    second = import_blot(s, two_strips(), "exposure 2.tif", membrane_id="mem-2")
    update = ops.set_marker_image(s, second, marker)
    batch = s.project.batch
    assert [m.id for m, _ in batch.arenas_on(second)] == ["mem-2", "mem-5", "mem-7"]
    assert len(update.adopted_arena_ids) == 3
    # A marker of another membrane altogether, as before.
    lone = import_blot(s, two_strips(), "lone.tif", kind=ImageKind.VISIBLE_MARKER)
    error = refused(lambda: ops.set_marker_image(s, exposure, lone), ErrorCode.INVALID_INPUT)
    assert str(error) == f"{lone} is not a visible-light marker or merged image of membrane mem-2"


def test_area_ops_follow_register_group(tmp_path):
    s, exposure, marker = marker_project(tmp_path)
    ops.set_marker_image(s, exposure, marker)
    batch = s.project.batch
    e2, m2 = batch.arenas_on(exposure)[1][1].id, batch.arenas_on(marker)[1][1].id
    # Removing membrane 2's area on M removes it on E too.
    removal = ops.remove_arena(s, m2)
    assert removal.cascade.removed[:2] == (e2, m2)
    batch = s.project.batch
    assert len(batch.arenas_on(exposure)) == len(batch.arenas_on(marker)) == 2
    # Assigning membrane 3's area on E to a new membrane moves it on M too.
    third = batch.arenas_on(exposure)[1][1].id
    moved = ops.assign_arena(s, third, None)
    assert len(moved.arena_ids) == 2
    assert moved.renumbered is None
    batch = s.project.batch
    assert [m.id for m, _ in batch.arenas_on(marker)] == [
        m.id for m, _ in batch.arenas_on(exposure)
    ]
    # A degraded edit of one image alone is refused by the model.

    def one_only(draft: Project) -> None:
        draft.batch.arenas_on(marker)[0][1].angle_deg = 2.0

    with pytest.raises(ValueError, match="hold different membrane areas"):
        apply_change(s.project, one_only)


# --- Reading one area: the row mask, anchors, calibrations ---

LEVEL = 50000.0


def tilted(shape, strips, lanes=10, pitch=100.0):
    """Strips of ``lanes`` bands, each ``(cx, cy, angle, depth)``: the bands
    on the strip's turned midline, ``pitch`` px apart along it."""
    height, width = shape
    y, x = np.mgrid[0:height, 0:width].astype(float)
    image = np.full(shape, LEVEL)
    for cx, cy, angle, depth in strips:
        t = math.radians(angle)
        for i in range(lanes):
            u = (i - (lanes - 1) / 2) * pitch
            bx, by = cx + u * math.cos(t), cy + u * math.sin(t)
            image -= depth * np.exp(-(((x - bx) / 18.0) ** 2) - ((y - by) / 3.0) ** 2)
    return np.clip(np.round(image), 0, 65535).astype(np.uint16)


def rotated_strips(tmp_path: Path):
    """Three strips 1000 px wide and 40 high on a 1100x320 image, ten
    lanes each: strip 1 at +4° centred (550, 70); strip 2 parallel to it, 6 px
    below it across its width, with stronger bands; strip 3 at -3° centred
    (550, 250). A 4° strip rises 69.9 px across its width, so strip 1's bounds
    reach over strip 2's bands at its left end."""
    t = math.radians(4.0)
    c2 = (550.0 - 46 * math.sin(t), 70.0 + 46 * math.cos(t))
    strips = [(550.0, 70.0, 4.0, 20000.0), (*c2, 4.0, 30000.0), (550.0, 250.0, -3.0, 20000.0)]
    s = session_on(tmp_path)
    image = import_blot(s, tilted((320, 1100), strips), "strips α.tif")
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(10)])
    ops.edit_arena(s, "arena-4", outline=(50, 50, 1050, 90), angle_deg=4.0)
    ops.add_arena(s, image, (47, 96, 1047, 136), angle_deg=4.0)
    ops.add_arena(s, image, (50, 230, 1050, 270), angle_deg=-3.0)
    proteins = [
        ops.add_protein(s, name, Role.TARGET, arena_id=arena)
        for name, arena in (("A", "arena-4"), ("B", "arena-6"), ("C", "arena-8"))
    ]
    return s, image, proteins


def centres_inside(s, protein: str) -> bool:
    batch = s.project.batch
    found = batch.find_protein(protein)
    arena, image = batch.arena_of(found), batch.image_of(found)
    own = batch.boxes_of(found)
    return all(
        arena.contains(image, b.box.x + own.box_size.width / 2, b.box.y + own.box_size.height / 2)
        for b in own.bands
    )


def test_rotated_strips_row_drag_refuses_neighbour_bands(tmp_path, monkeypatch):
    s, image, (a, _, _) = rotated_strips(tmp_path)
    batch = s.project.batch
    bounds = batch.find_arena("arena-4")[1].bounds(batch.find_image(image))
    assert bounds == (49, 15, 1051, 125)
    placed = ops.detect_row_boxes(s, a, bounds)
    assert all(placed.band_ids)
    assert centres_inside(s, a)
    ys = [
        b.box.y
        for b in sorted(
            s.project.batch.boxes_of(s.project.batch.find_protein(a)).bands,
            key=lambda b: b.lane_index,
        )
    ]
    assert ys == sorted(ys) and ys[-1] - ys[0] > 55  # strip 1's own rising row
    assert_nets_current(s)
    ops.undo(s)
    # Degraded: clipped to the bounds only, the row takes strip 2's bands.
    monkeypatch.setattr(ops, "_masked", lambda array, image, area, window: array)
    with pytest.raises(OperationError) as info:
        ops.detect_row_boxes(s, a, bounds)
    assert info.value.code in (ErrorCode.ROW_OFF_LINE, ErrorCode.OUT_OF_ARENA)


def test_a_row_over_a_turned_strip_reads_it_alone(tmp_path):
    s, image, (_, _, c) = rotated_strips(tmp_path)
    # Strip 3 at -3°: a drag over its bounds, then one inside it, both read it alone.
    batch = s.project.batch
    bounds = batch.find_arena("arena-8")[1].bounds(batch.find_image(image))
    placed = ops.detect_row_boxes(s, c, bounds)
    assert all(placed.band_ids)
    assert centres_inside(s, c)


def test_mask_never_reaches_quantify(tmp_path, monkeypatch):
    s, image, (a, _, _) = rotated_strips(tmp_path)
    batch = s.project.batch
    bounds = batch.find_arena("arena-4")[1].bounds(batch.find_image(image))
    ops.detect_row_boxes(s, a, bounds)
    assert_nets_current(s)  # the nets read the stored pixels, not the masked copy


def test_mask_matches_contains():
    ref = make_project().batch.find_image("img-2").model_copy(update={"width": 300, "height": 200})
    rng = np.random.default_rng(7)
    for angle in (0.0, 4.0, -3.0, 45.0, -45.0, 17.3):
        area = Arena(
            id="arena-1",
            image_id="img-2",
            outline=Outline(x0=40, y0=30, x1=260, y1=90),
            angle_deg=angle,
            source="manual",
        )
        mask = ops._inside_mask(area, ref, (0, 0, 300, 200))
        cols, rows = rng.integers(0, 300, 400), rng.integers(0, 200, 400)
        for c, r in zip(cols, rows, strict=True):
            assert bool(mask[r, c]) == area.contains(ref, c + 0.5, r + 0.5), (angle, c, r)
    whole = Arena(id="arena-1", image_id="img-2", source="whole_image")
    array = np.arange(60, dtype=np.uint16).reshape(6, 10)
    assert ops._masked(array, ref, whole, (0, 0, 10, 6)) is array
    drawn = Arena(
        id="arena-1",
        image_id="img-2",
        outline=Outline(x0=2, y0=1, x1=6, y1=4),
        source="manual",
    )
    masked = ops._masked(array, ref, drawn, (0, 0, 10, 6))
    assert masked is not array and np.array_equal(array, np.arange(60).reshape(6, 10))
    inside = ops._inside_mask(drawn, ref, (0, 0, 10, 6))
    assert np.array_equal(masked[inside], array[inside])
    assert (masked[~inside] == round(ref.background)).all() or np.allclose(
        masked[~inside], ref.background, atol=1
    )
    # Only the window is masked.
    part = ops._masked(array, ref, drawn, (0, 0, 3, 6))
    assert np.array_equal(part[:, 3:], array[:, 3:])


def mirrored(tmp_path: Path):
    """Two strips on one image, each with a protein boxed in lanes 1 and
    2: strip 1 numbered left to right (lane 1 at x 40), strip 2 right to left
    (lane 1 at x 160)."""
    s, image, top, bottom = split(tmp_path)
    ops.place_box(s, top, 40, 24, lane_index=0, grow=False)
    ops.place_box(s, top, 100, 24, lane_index=1, grow=False)
    ops.place_box(s, bottom, 160, 76, lane_index=0, grow=False)
    ops.place_box(s, bottom, 100, 76, lane_index=1, grow=False)
    return s, image, top, bottom


def test_mirrored_strip(tmp_path):
    s, image, top, bottom = mirrored(tmp_path)
    batch = s.project.batch
    assert lane_anchors(batch, "arena-4") == [(40.5, 0), (100.5, 1)]
    assert lane_anchors(batch, "arena-6") == [(160.5, 0), (100.5, 1)]
    # Each strip's third lane, proposed from its own boxes.
    band = ops.place_box(s, top, 160, 24, grow=False)
    assert s.project.batch.find_band(band)[1].lane_index == 2
    band = ops.place_box(s, bottom, 40, 76, grow=False)
    assert s.project.batch.find_band(band)[1].lane_index == 2


def test_anchors_per_area(tmp_path, monkeypatch):
    s, image, top, bottom = mirrored(tmp_path)
    # Degraded: every box on the image anchors every area, as on an image of
    # one membrane; the bottom strip's third lane is then proposed wrong.
    real = ops.lane_anchors

    def image_wide(batch, arena_id, **kwargs):
        return [
            anchor
            for _, other in batch.arenas_on(batch.find_arena(arena_id)[1].image_id)
            for anchor in real(batch, other.id, **kwargs)
        ]

    monkeypatch.setattr(ops, "lane_anchors", image_wide)
    with pytest.raises(OperationError) as info:
        band = ops.place_box(s, bottom, 40, 76, grow=False)
        assert s.project.batch.find_band(band)[1].lane_index != 2
        raise OperationError(ErrorCode.LANE_REQUIRED, "wrong lane")
    assert info.value.code in (ErrorCode.LANE_REQUIRED, ErrorCode.LANE_OCCUPIED)


def test_row_checks_read_their_own_area(tmp_path):
    s, image, top, bottom = mirrored(tmp_path)
    # Rows on each strip, numbered as their own boxes run.
    up = ops.detect_row_boxes(s, top, (10, 12, 190, 38))
    down = ops.detect_row_boxes(s, bottom, (10, 64, 190, 90))
    assert (up.right_to_left, down.right_to_left) == (False, True)
    assert_nets_current(s)


def strip_y(k: int) -> int:
    """The centre row of strip ``k`` (from 0) of :func:`nine_strips`."""
    return 12 + 18 * k


def nine_strips(tmp_path: Path, lanes: int = 6):
    """Nine strips on one image, 12 rows each with 6 px gaps (strip k's
    area y_k - 6 to y_k + 6, y_k = 12 + 18 k), ``lanes`` lanes 30 px apart,
    each strip a membrane with one protein boxed (14x8) on its band in every
    lane. The rings of one strip's bands reach the next strip's boxes."""
    pitch = 30
    width = lanes * pitch + 20
    bands = [
        (25 + lane * pitch, strip_y(k), 5.0, 3.0, 20000.0 + 1000 * k)
        for k in range(9)
        for lane in range(lanes)
    ]
    s = session_on(tmp_path)
    image = import_blot(s, synthetic_blot((170, width), bands), "nine.tif")
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(lanes)])
    ops.edit_arena(s, "arena-4", outline=(0, strip_y(0) - 6, width, strip_y(0) + 6))
    areas = ["arena-4"]
    for k in range(1, 9):
        outline = (0, strip_y(k) - 6, width, strip_y(k) + 6)
        areas.extend(ops.add_arena(s, image, outline).arena_ids)
    proteins = []
    for k, arena in enumerate(areas):
        protein = ops.add_protein(
            s, f"P{k + 1}", Role.TARGET, arena_id=arena, box_size=ops.BoxSize(width=14, height=8)
        )
        for lane in range(lanes):
            ops.place_box(s, protein, 25 + lane * pitch, strip_y(k), lane_index=lane, grow=False)
        proteins.append(protein)
    return s, image, areas, proteins


def test_nine_strips_calibrations_are_independent(tmp_path):
    s, image, areas, proteins = nine_strips(tmp_path)
    assert len(s.project.batch.membranes) == 9
    ops.add_calibration_point(s, areas[2], 43.0, 100.0, STRIP_EDGE, x=3.0, snap=False)
    ops.add_calibration_point(s, areas[2], 53.0, 30.0, STRIP_EDGE, x=3.0, snap=False)
    batch = s.project.batch
    fits = [ops.calibration_fit(batch, m, image) for m in batch.membranes]
    assert [fit is not None for fit in fits] == [k == 2 for k in range(9)]
    mws = {
        protein: [b.apparent_mw for b in batch.boxes_of(batch.find_protein(protein)).bands]
        for protein in proteins
    }
    assert all(mw is not None for mw in mws[proteins[2]])
    assert all(mw is None for p in proteins if p != proteins[2] for mw in mws[p])
    # Marks on strip 4 do not shape strip 3's curve.
    ops.add_calibration_point(s, areas[3], 61.0, 100.0, STRIP_EDGE, x=3.0, snap=False)
    ops.add_calibration_point(s, areas[3], 71.0, 10.0, STRIP_EDGE, x=3.0, snap=False)
    after = s.project.batch
    assert ops.calibration_fit(after, after.membranes[2], image) == fits[2]
    assert [b.apparent_mw for b in after.boxes_of(after.find_protein(proteins[2])).bands] == mws[
        proteins[2]
    ]
    results = compute_results(after)
    assert not [n for n in results.notices if n.code == "rows_tilted"]


def test_nine_strips_nets_equal_the_unsplit_image(tmp_path):
    s, image, areas, proteins = nine_strips(tmp_path)
    split_nets = nets(s)
    # The same boxes on the image of one membrane: the same nets (the rings
    # read across the outlines and leave out every box on the image).
    whole = s.project.model_copy(deep=True)

    def one_membrane(draft: Project) -> None:
        batch = draft.batch
        keep = batch.membranes[0]
        keep.arenas[0].outline = None
        for protein in batch.proteins:
            protein.rows[0].arena_id = keep.arenas[0].id
        batch.membranes = [keep]
        ops._quantify_image(draft, image, s.pixels(image))

    joined, _ = apply_change(whole, one_membrane)
    assert {b.id: b.net for p in joined.batch.proteins for b in p.rows[0].bands} == split_nets


def rings_leave_out_own_area_only(batch, image_id, row_id):
    """A ring rule the maintainer may choose instead (#243): a row's rings
    leave out the boxes of the rows on its own area only, and read across
    the outlines."""
    arena_id = next(row.arena_id for _, row in batch.rows_on(image_id) if row.id == row_id)
    return RingContext(left_out=frozenset(row.id for _, row in batch.rows_on_arena(arena_id)))


def rings_read_own_area_only(batch, image_id, row_id):
    """Another: as :func:`rings_leave_out_own_area_only`, and its rings read
    no pixel outside its own area."""
    arena_id = next(row.arena_id for _, row in batch.rows_on(image_id) if row.id == row_id)
    _, arena = batch.find_arena(arena_id)
    return RingContext(
        left_out=frozenset(row.id for _, row in batch.rows_on_arena(arena_id)),
        read_within=ops.area_key(arena),
    )


@pytest.mark.parametrize("rule", [rings_leave_out_own_area_only, rings_read_own_area_only])
def test_ring_rule_is_one_function(tmp_path, monkeypatch, rule):
    s, image, areas, proteins = nine_strips(tmp_path)
    before = nets(s)
    monkeypatch.setattr(ops, "_ring_context", rule)
    # A box of strip 1 moved: under another rule the nets of strip 2 are its own.
    project, _ = apply_change(s.project, lambda d: ops._quantify_image(d, image, s.pixels(image)))
    swapped = {b.id: b.net for p in project.batch.proteins for b in p.rows[0].bands}
    assert swapped != before


def edit_strip_1(s, areas, proteins, what):
    if what == "boxes":
        band = s.project.batch.boxes_of(s.project.batch.find_protein(proteins[0])).bands[2]
        ops.move_box(s, band.id, band.box.rect(ops.BoxSize(width=14, height=8)))
        x0, y0, x1, y1 = band.box.rect(ops.BoxSize(width=14, height=8))
        ops.move_box(s, band.id, (x0 + 3, y0 + 4, x1 + 3, y1 + 4))
    elif what == "lanes":
        lanes = s.project.batch.only_lanes()
        ops.set_lanes(
            s,
            [LaneInput(lane.label, f"s{lane.index}", included=lane.index != 1) for lane in lanes],
        )
    else:
        ops.add_calibration_point(s, areas[0], 7.0, 100.0, STRIP_EDGE, x=3.0, snap=False)
        ops.add_calibration_point(s, areas[0], 17.0, 30.0, STRIP_EDGE, x=3.0, snap=False)


def others_unchanged(s, before, proteins):
    after = s.project.batch
    for protein in proteins[1:]:
        old = [(b.net, b.apparent_mw) for b in before.boxes_of(before.find_protein(protein)).bands]
        new = [(b.net, b.apparent_mw) for b in after.boxes_of(after.find_protein(protein)).bands]
        assert new == old, protein
    assert [m.calibration for m in after.membranes[1:]] == [
        m.calibration for m in before.membranes[1:]
    ]


@pytest.mark.xfail(
    strict=True,
    reason="the background ring rule across membranes on one image is still the current one:"
    " a ring leaves out every box on the image, so moving one membrane's box moves the nets"
    " of another membrane's bands whose rings reach it, until that rule is decided (#243)",
)
def test_a_box_edit_of_one_membrane_leaves_the_others_nets(tmp_path):
    s, image, areas, proteins = nine_strips(tmp_path)
    before = s.project.batch.model_copy(deep=True)
    edit_strip_1(s, areas, proteins, "boxes")
    others_unchanged(s, before, proteins)


@pytest.mark.parametrize("what", ["lanes", "calibration"])
def test_lanes_and_marks_of_one_membrane_leave_the_others_alone(tmp_path, what):
    # Under the current rule too: the nets do not read the lane table or the
    # marks, and each membrane's curve is its own.
    s, image, areas, proteins = nine_strips(tmp_path)
    before = s.project.batch.model_copy(deep=True)
    edit_strip_1(s, areas, proteins, what)
    others_unchanged(s, before, proteins)


def test_split_strip_without_ladder_is_told():
    from conftest import make_project

    before = make_project().batch
    after = before.model_copy(deep=True)
    mem_1 = after.membranes[0]
    mem_1.calibration.points = mem_1.calibration.points[:1]
    lost = []
    for protein in after.proteins:
        for band in after.boxes_of(protein).bands:
            if band.apparent_mw is not None and after.membrane_of_protein(protein) is mem_1:
                band.apparent_mw = None
                lost.append(band.id)
    assert lost
    [told] = ops._mw_lost(before, after)
    assert told == {
        "membrane_id": "mem-1",
        "membrane": 1,
        "bands": len(lost),
        "message": f"Membrane 1 has fewer than 2 ladder marks: {len(lost)} bands lost their"
        " apparent MW; mark its ladder to read them again",
    }
    assert ops._mw_lost(before, before) == ()


def test_row_on_marker_area(tmp_path):
    s, exposure, marker = marker_project(tmp_path)
    ops.set_marker_image(s, exposure, marker)
    area = area_on(s, marker, 2)
    error = refused(
        lambda: ops.add_protein(s, "GAPDH", Role.TARGET, arena_id=area), ErrorCode.MARKER_IMAGE
    )
    assert str(error) == "'M marker.tif' is a visible-light marker image, not a signal image"


def test_cover_counts_all_rows(tmp_path):
    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 52))
    ops.add_arena(s, image, (0, 52, 200, 100))
    top = ops.add_protein(
        s, "A", Role.TARGET, arena_id="arena-4", box_size=ops.BoxSize(width=10, height=10)
    )
    bottom = ops.add_protein(
        s, "B", Role.TARGET, arena_id="arena-6", box_size=ops.BoxSize(width=10, height=10)
    )
    ops.place_box(s, top, 40, 49, lane_index=0, grow=False)  # y 44-54, centre 49.5: top
    error = refused(
        lambda: ops.place_box(s, bottom, 40, 53, lane_index=0, grow=False), ErrorCode.OVERLAP
    )
    assert "the box would overlap the box of 'A' in lane 1" in str(error)


def test_undo_redo_each_area_op(tmp_path):
    s, image, top, bottom = split(tmp_path)
    box_all(s, top, 24)
    steps = [
        lambda: ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 45), angle_deg=1.0),
        lambda: ops.add_arena(s, image, (0, 46, 100, 50)),
        lambda: ops.assign_arena(s, "arena-6", None),
        lambda: ops.remove_arena(s, s.project.batch.arenas_on(image)[1][1].id),
    ]
    for step in steps:
        before = s.project
        step()
        after = s.project
        ops.undo(s)
        assert content_hash(s.project) == content_hash(before)
        ops.redo(s)
        assert content_hash(s.project) == content_hash(after)


def test_remove_image_takes_every_membrane_on_it(tmp_path):
    s, image, top, bottom = split(tmp_path)
    box_all(s, top, 24)
    cascade = ops.remove_image(s, image)
    assert cascade.removed[:3] == (image, "arena-4", "arena-6")
    assert cascade.removed[-2:] == ("mem-2", "mem-5")
    assert s.project.batch.membranes == []


def test_every_export_maps_the_membrane_labels(tmp_path):
    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 50, 100))
    for x0 in (50, 100, 150):
        ops.add_arena(s, image, (x0, 0, x0 + 50, 100))
    ops.edit_arena(s, area_on(s, image, 3), angle_deg=-2.5)
    ops.remove_arena(s, area_on(s, image, 2))
    bundle = ops.export_bundle(s, formats=[])
    readme = (bundle.folder / "README.txt").read_text(encoding="utf-8").splitlines()
    start = readme.index("Membranes")
    assert readme[start + 1 : start + 3] == ["---------", ""]
    mapped = readme[readme.index("Membrane 1", start) :]
    assert mapped[:6] == [
        "Membrane 1",
        f"    {STRIPS}: the rectangle (0, 0)-(50, 100)",
        "Membrane 2",
        f"    {STRIPS}: the rectangle (100, 0)-(150, 100), turned -2.5°",
        "Membrane 3",
        f"    {STRIPS}: the rectangle (150, 0)-(200, 100)",
    ]
    # An image of one membrane: the whole image.
    lone, _ = one_image(tmp_path / "lone")
    lines = (ops.export_bundle(lone, formats=[]).folder / "README.txt").read_text("utf-8")
    assert lines.splitlines()[-2:] == ["Membrane 1", f"    {STRIPS}: the whole image"]
