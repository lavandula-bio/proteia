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
from proteia.core import ladders, storage
from proteia.core import operations as ops
from proteia.core import session as session_module
from proteia.core.model import (
    Arena,
    ArenaSource,
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
from test_operations import (
    assert_mw_current,
    assert_nets_current,
    import_blot,
    plant,
    session_on,
)

STRIPS = "two strips µ.tif"
CHEMI_MARKER = CalibrationPointSource.CHEMILUMINESCENCE_MARKER
STRIP_EDGE = CalibrationPointSource.STRIP_EDGE
LANE_X = (40, 100, 160)
# The next steps a refusal of an outline that would leave boxes outside
# their area gives, for an area not turned and an outline not turned.
CUT_STEP = (
    "; draw the outline around them; or cut the area in two: draw the outline to three of"
    " its edges, its fourth edge across the area with each protein's boxes all on one side of"
    " it, and the far side becomes a new membrane's area with what lies there; or move or"
    " remove them first"
)
# The next step of the same refusal for an outline that cuts its area: what
# lies past the cut with the rest of its row inside stays.
CUT_DRAWN = (
    ", and a cut takes each row to one side whole; draw its edge across the area past"
    " them, or move or remove them first"
)


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
        f" mark on {STRIPS!r} outside membrane 1's area" + CUT_DRAWN
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


# --- An area holding two membranes' rows, cut in two ---


def two_rows(tmp_path: Path):
    """The two strips as a project from before membrane areas holds them: one
    membrane, its area the whole image (arena-4), p53 µ boxed on the top strip
    and GAPDH, its loading control, on the bottom one."""
    s, image = one_image(tmp_path)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, image)
    p53 = ops.add_protein(s, "p53 µ", Role.TARGET, image, loading_control_ids=[gapdh])
    box_all(s, p53, 24)
    box_all(s, gapdh, 76)
    return s, image, p53, gapdh


WHOLE = None
CUTS = [
    # (the area's outline, its angle, the outline drawn, its angle, the far side)
    (WHOLE, 0.0, (0, 0, 200, 50), 0.0, (0, 50, 200, 100)),  # the bottom edge across
    (WHOLE, 0.0, (0, 50, 200, 100), 0.0, (0, 0, 200, 50)),  # the top edge across
    (WHOLE, 0.0, (0, 0, 70, 100), 0.0, (70, 0, 200, 100)),  # the right edge across
    (WHOLE, 0.0, (130, 0, 200, 100), 0.0, (0, 0, 130, 100)),  # the left edge across
    (WHOLE, 0.0, (-5, -5, 205, 50), 0.0, (0, 50, 200, 100)),  # past the image's edges
    (WHOLE, 3.0, (0, 0, 200, 50), 0.0, (0, 50, 200, 100)),  # a whole image's angle reads nothing
    (WHOLE, 0.0, (10, 5, 190, 45), 0.0, None),  # in from every side
    (WHOLE, 0.0, (10, 0, 200, 50), 0.0, None),  # in from two sides
    (WHOLE, 0.0, (0, 0, 200, 100), 0.0, None),  # over all of it
    (WHOLE, 0.0, (0, 0, 200, 50), 1.0, None),  # turned
    ((0, 0, 100, 100), 0.0, (0, 0, 100, 40), 0.0, (0, 40, 100, 100)),  # a drawn area
    ((0, 0, 100, 100), 0.0, (150, 0, 200, 50), 0.0, None),  # beside it, not across it
    ((-20, 0, 220, 50), 0.0, (0, 0, 120, 50), 0.0, (120, 0, 200, 50)),  # as it lies in the image
    ((0, 0, 200, 50), 2.0, (0, 0, 200, 30), 0.0, None),  # a turned area
    # Beside the area, touching one of its sides from outside: no cut.
    ((0, 0, 100, 100), 0.0, (100, 0, 200, 100), 0.0, None),
    ((0, 0, 100, 100), 0.0, (-100, 0, 0, 100), 0.0, None),
    ((0, 0, 100, 100), 0.0, (0, 100, 100, 200), 0.0, None),
    ((0, 0, 100, 100), 0.0, (0, -100, 100, 0), 0.0, None),
]


@pytest.mark.parametrize(("old", "turn", "drawn", "angle", "far"), CUTS)
def test_which_redraws_cut_an_area_in_two(tmp_path, old, turn, drawn, angle, far):
    s, image = one_image(tmp_path)
    found = s.project.batch.find_image(image)
    area = Arena(
        id="arena-4",
        image_id=image,
        outline=None if old is None else Outline(x0=old[0], y0=old[1], x1=old[2], y1=old[3]),
        angle_deg=turn,
        source=ArenaSource.MANUAL,
    )
    drawn = Outline(x0=drawn[0], y0=drawn[1], x1=drawn[2], y1=drawn[3])
    rest = ops._cut_rest(area, found, drawn, angle)
    assert rest == (None if far is None else Outline(x0=far[0], y0=far[1], x1=far[2], y1=far[3]))


def test_an_area_cut_in_two_gives_the_far_side_a_new_membrane(tmp_path):
    s, image, p53, gapdh = two_rows(tmp_path)
    measured = nets(s)
    # Drawn around the top strip alone: the bottom strip's boxes would be left out.
    error = refused(
        lambda: ops.edit_arena(s, "arena-4", outline=(10, 5, 190, 45)), ErrorCode.ARENA_IN_USE
    )
    assert str(error) == (
        "the new outline would leave the boxes of 'GAPDH' in lanes 1, 2, 3 outside membrane 1's"
        " area" + CUT_STEP
    )
    # That step, taken as it reads: the outline to the image's top and sides,
    # its bottom edge across the image between the strips.
    before, count = s.project, len(s.project.log)
    change = ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    cut = s.project
    batch = cut.batch
    assert len(cut.log) == count + 1
    [(first, kept), (second, rest)] = batch.arenas_on(image)
    assert (kept.id, kept.outline) == ("arena-4", Outline(x0=0, y0=0, x1=200, y1=50))
    assert (rest.outline, rest.angle_deg, rest.source) == (
        Outline(x0=0, y0=50, x1=200, y1=100),
        0.0,
        ArenaSource.MANUAL,
    )
    assert first.lane_table_id == second.lane_table_id
    assert (change.arena_ids, change.membrane_id) == (("arena-4",), first.id)
    assert (change.rest_membrane_id, change.rest_arena_ids) == (second.id, (rest.id,))
    assert (change.moved_proteins, change.moved_points) == ((gapdh,), 0)
    assert (change.curves_changed, change.dropped_undetected, change.mw_lost) == ((), (), ())
    assert change.handed_off == "the rest of membrane 1's area is now membrane 2's, with 'GAPDH'"
    assert batch.arena_id_of(batch.find_protein(p53)) == "arena-4"
    assert batch.arena_id_of(batch.find_protein(gapdh)) == rest.id
    assert batch.find_protein(p53).loading_control_ids == [gapdh]  # still its loading control
    # Each side's rings now read its own side alone: the nets they move are answered.
    moved = {band for band, net in nets(s).items() if net != measured[band]}
    assert {band for band, _, _ in change.remeasured} == moved
    assert_nets_current(s)
    params = cut.log[-1].params
    assert (params["rest_membrane_id"], params["rest_arena_ids"]) == (second.id, [rest.id])
    assert (params["moved_proteins"], params["moved_points"]) == ([gapdh], [])
    # The bottom strip's boxes are placed in its own area now: one taken out
    # and put back, its lane proposed from the strip's own boxes.
    third = next(b.id for b in batch.boxes_of(batch.find_protein(gapdh)).bands if b.lane_index == 2)
    ops.remove_box(s, third)
    band = ops.place_box(s, gapdh, 160, 76, grow=False)
    assert s.project.batch.find_band(band)[1].lane_index == 2
    assert s.project.batch.arena_id_of(s.project.batch.find_protein(gapdh)) == rest.id
    # The cut is one change: one undo puts the image back as it was.
    ops.undo(s)
    ops.undo(s)
    assert content_hash(s.project) == content_hash(cut)
    ops.undo(s)
    assert content_hash(s.project) == content_hash(before)
    ops.redo(s)
    assert content_hash(s.project) == content_hash(cut)


@pytest.mark.parametrize(
    ("outline", "turn", "offered"),
    [
        (None, 3.0, True),  # a whole image's recorded angle turns nothing
        ((0, 0, 200, 100), 0.0, True),  # drawn, not turned
        ((0, 0, 200, 100), 2.0, False),  # drawn and turned: no line cuts it
    ],
)
def test_the_cut_is_offered_where_one_can_be_drawn(tmp_path, outline, turn, offered):
    s, image, p53, gapdh = two_rows(tmp_path)
    ops.edit_arena(s, "arena-4", outline=outline, angle_deg=turn)
    error = refused(
        lambda: ops.edit_arena(s, "arena-4", outline=(10, 5, 190, 45), angle_deg=0.0),
        ErrorCode.ARENA_IN_USE,
    )
    assert str(error).endswith(CUT_STEP) is offered
    if offered:  # and taken
        change = ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50), angle_deg=0.0)
        assert change.moved_proteins == (gapdh,)


def test_a_box_just_past_the_cut_goes_with_the_far_side(tmp_path):
    # A box centred half a pixel below the cut, at the far end of it, where
    # an outline turned by even 1° would still hold it, goes with GAPDH.
    s, image, p53, gapdh = two_rows(tmp_path)
    actin = ops.add_protein(
        s, "β-actin", Role.TARGET, image, box_size=ops.BoxSize(width=10, height=5)
    )
    ops.place_box(s, actin, 194, 50, lane_index=2, grow=False)  # centre (194, 50.5)
    change = ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    assert change.moved_proteins == (gapdh, actin)


def test_a_cut_needs_each_row_on_one_side(tmp_path):
    s, image, p53, gapdh = two_rows(tmp_path)
    actin = ops.add_protein(s, "β-actin", Role.TARGET, image)
    ops.place_box(s, actin, 40, 40, lane_index=0, grow=False)
    lower = ops.place_box(s, actin, 100, 60, lane_index=1, grow=False)
    before = s.project
    # GAPDH would go with the rest, but one box of β-actin lies past the cut.
    error = refused(
        lambda: ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50)), ErrorCode.ARENA_IN_USE
    )
    assert str(error) == (
        "the new outline would leave the box of 'β-actin' in lane 2 outside membrane 1's area"
        + CUT_DRAWN
    )
    assert error.detail["band_ids"] == [lower]
    # Across the lanes, every row lies on both sides: nothing to hand over.
    error = refused(
        lambda: ops.edit_arena(s, "arena-4", outline=(0, 0, 130, 100)), ErrorCode.ARENA_IN_USE
    )
    assert str(error) == (
        "the new outline would leave the box of 'GAPDH' in lane 3 and the box of 'p53 µ' in lane 3"
        " outside membrane 1's area" + CUT_DRAWN
    )
    # A turned outline cuts nothing, and the step is not offered.
    error = refused(
        lambda: ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50), angle_deg=1.0),
        ErrorCode.ARENA_IN_USE,
    )
    assert str(error).endswith("; draw the outline around them, or move or remove them first")
    assert s.project is before
    # β-actin's box moved above the cut: the same cut hands GAPDH over.
    size = s.project.batch.boxes_of(s.project.batch.find_protein(actin)).box_size
    x0, y0, x1, y1 = s.project.batch.find_band(lower)[1].box.rect(size)
    ops.move_box(s, lower, (x0, y0 - 20, x1, y1 - 20))
    change = ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    assert change.moved_proteins == (gapdh,)


def test_a_cut_reaches_the_linked_marker_and_its_ladder(tmp_path):
    s = session_on(tmp_path)
    exposure = import_blot(s, two_strips(), "E.tif")
    ops.set_lanes(s, [LaneInput("a"), LaneInput("b"), LaneInput("c")])
    membrane = s.project.batch.membrane_of(exposure).id
    marker = import_blot(
        s, two_strips(), "M marker.tif", kind=ImageKind.VISIBLE_MARKER, membrane_id=membrane
    )
    ops.set_marker_image(s, exposure, marker)
    ops.set_ladder(s, membrane, PRESET)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, exposure)
    p53 = ops.add_protein(s, "p53 µ", Role.TARGET, exposure, loading_control_ids=[gapdh])
    box_all(s, p53, 24)
    box_all(s, gapdh, 76)
    # One ladder marked down the marker, as before areas: two marks above the
    # cut to come.
    for y, kda in ((10.0, 250.0), (40.0, 100.0)):
        ops.add_calibration_point(s, marker, y, kda, STRIP_EDGE, x=5.0, snap=False)

    def mws(protein: str) -> list:
        batch = s.project.batch
        return [band.apparent_mw for band in batch.boxes_of(batch.find_protein(protein)).bands]

    # GAPDH, below the marks' range, has no MW; cut off, it is read with no
    # curve at all, and its image is told as one whose curve changed.
    assert mws(gapdh) == [None, None, None] and None not in mws(p53)
    change = ops.edit_arena(s, area_on(s, exposure), outline=(0, 0, 200, 50))
    assert (change.moved_points, change.curves_changed, change.mw_lost) == (0, (exposure,), ())
    assert_mw_current(s)
    ops.undo(s)
    # And one mark below the cut.
    ops.add_calibration_point(s, marker, 90.0, 25.0, STRIP_EDGE, x=5.0, snap=False)
    assert None not in mws(p53) + mws(gapdh)
    before = s.project
    change = ops.edit_arena(s, area_on(s, exposure), outline=(0, 0, 200, 50))
    batch = s.project.batch
    first, second = batch.membranes
    for image_id in (exposure, marker):
        assert [(m.id, a.outline) for m, a in batch.arenas_on(image_id)] == [
            (first.id, Outline(x0=0, y0=0, x1=200, y1=50)),
            (second.id, Outline(x0=0, y0=50, x1=200, y1=100)),
        ]
    assert change.rest_arena_ids == (area_on(s, exposure, 2), area_on(s, marker, 2))
    assert [(p.y, p.mw) for p in first.calibration.points] == [(10.0, 250.0), (40.0, 100.0)]
    assert [(p.y, p.mw) for p in second.calibration.points] == [(90.0, 25.0)]
    assert ladder_of(s, second.id) == ladder_of(s, first.id) != (None, [])
    assert (change.moved_proteins, change.moved_points) == ((gapdh,), 1)
    assert change.curves_changed == (exposure, marker)  # membrane 1's curve, one mark less
    assert change.handed_off == (
        "the rest of membrane 1's area is now membrane 2's, with 'GAPDH' and 1 ladder mark"
    )
    # Membrane 2 is left one mark: GAPDH's bands lose their apparent MW, and are told.
    assert [(lost["membrane"], lost["bands"]) for lost in change.mw_lost] == [(2, 3)]
    assert change.mw_lost[0]["image_ids"] == [exposure, marker]
    assert change.mw_lost[0]["message"] == (  # one register group: no image named
        "Membrane 2 has fewer than 2 ladder marks: 3 bands lost their apparent MW; mark its"
        " ladder to read them again"
    )
    assert mws(gapdh) == [None, None, None] and None not in mws(p53)
    assert_mw_current(s)
    assert_nets_current(s)
    ops.undo(s)
    assert content_hash(s.project) == content_hash(before)
    # A second mark below the cut: the bottom strip keeps a curve of its own.
    ops.add_calibration_point(s, marker, 60.0, 55.0, STRIP_EDGE, x=5.0, snap=False)
    change = ops.edit_arena(s, area_on(s, marker), outline=(0, 0, 200, 50))
    assert (change.moved_points, change.mw_lost) == (2, ())
    assert None not in mws(p53) + mws(gapdh)
    assert_mw_current(s)


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
    monkeypatch.setattr(ops, "_shown", lambda array, image, area, window, background: array)
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
    # A click on the edge the two areas share belongs to the earlier
    # membrane's area, the top strip's: the bottom strip's protein is refused.
    error = refused(
        lambda: ops.place_box(s, bottom, 100, 50, lane_index=1, grow=False),
        ErrorCode.IN_TWO_ARENAS,
    )
    assert str(error) == (
        f"(100, 50) lies on the edge membrane 1's and membrane 2's areas share on {STRIPS!r},"
        " which belongs to membrane 1's area; place it inside membrane 2's area, past that"
        " edge, or redraw the areas"
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


def test_an_angle_edit_crops_the_rings_that_reach_the_outline(tmp_path):
    # Turning strip 3's area moves its outline, so the rings of its bands,
    # which reach the outline, read other pixels: the edit measures the image
    # again and answers whose nets changed, and by how much; the other strips
    # keep theirs.
    s, image, areas, proteins = nine_strips(tmp_path)
    before = nets(s)
    reads = []
    real = s.pixels

    def counted(image_id, **kwargs):
        reads.append(image_id)
        return real(image_id, **kwargs)

    s.pixels = counted
    measured: list[int] = []
    real_backgrounds = ops.band_backgrounds

    def counted_backgrounds(array, rects, *args, **kwargs):
        measured.append(len(rects))
        return real_backgrounds(array, rects, *args, **kwargs)

    ops.band_backgrounds = counted_backgrounds
    try:
        change = ops.edit_arena(s, areas[2], angle_deg=1.0)
    finally:
        ops.band_backgrounds = real_backgrounds
        s.pixels = real
    assert reads == [image]
    assert measured == [6]  # strip 3's boxes alone: the other strips' rings read as before
    after = nets(s)
    moved = {band for band in after if after[band] != before[band]}
    batch = s.project.batch
    strip_3 = {b.id for b in batch.boxes_of(batch.find_protein(proteins[2])).bands}
    assert moved and moved <= strip_3
    assert {band for band, _, _ in change.remeasured} == moved
    assert change.largest_change is not None and change.largest_change[0] in moved
    assert_nets_current(s)
    # Turned back, the same nets as before.
    ops.edit_arena(s, areas[2], angle_deg=0.0)
    assert nets(s) == before


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


def test_outline_and_area_inputs_are_checked(tmp_path):
    s, image = one_image(tmp_path)
    for bad in ((0, 0, 10), "0,0,10,10", (0, 0, 10, 10.5), (0, 0, True, 10)):
        refused(lambda bad=bad: ops.edit_arena(s, "arena-4", outline=bad), ErrorCode.INVALID_INPUT)
    refused(lambda: ops.add_arena(s, image, None), ErrorCode.INVALID_INPUT)
    for far in ((-1_000_001, 0, 10, 10), (0, 0, 10, 1_000_001)):
        error = refused(
            lambda far=far: ops.edit_arena(s, "arena-4", outline=far), ErrorCode.INVALID_INPUT
        )
        assert str(error) == "an outline reaches at most 1000000 px from the image's corner"
    ops.edit_arena(s, "arena-4", outline=(-999_800, 0, 1_000_000, 100))  # centre (100, 50)
    error = refused(lambda: ops.edit_arena(s, 4, angle_deg=1.0), ErrorCode.INVALID_INPUT)
    assert str(error) == "a membrane area id must be text, not 4"
    with pytest.raises(UnknownIdError):
        ops.add_arena(s, "img-99", (0, 0, 10, 10))


def test_removing_one_area_refits_the_membrane(tmp_path):
    # Membrane 1 on two images: its marks on the image whose area goes go
    # too, and it is refitted from what is left.
    s, image = one_image(tmp_path)
    other = import_blot(s, two_strips(), "second.tif", membrane_id="mem-2")
    ops.edit_arena(s, area_on(s, other), outline=(0, 0, 200, 50))
    ops.add_arena(s, other, (0, 50, 200, 100))
    ops.add_calibration_point(s, area_on(s, other), 10.0, 100.0, STRIP_EDGE, x=5.0, snap=False)
    removal = ops.remove_arena(s, area_on(s, other))
    assert removal.cascade.unfitted_membranes == ("mem-2",)
    assert removal.renumbered is None
    batch = s.project.batch
    assert [a.image_id for a in batch.membranes[0].arenas] == [image]
    assert batch.membranes[0].calibration.points == []
    assert [i.id for i in batch.iter_images()] == [image, other]


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
    # The rings of the bands that stay never left out the boxes gone, nor
    # read their area: none is measured again.
    assert removal.remeasured == ()
    assert all(nets(s)[b] == before[b] for b in top_bands)
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


def test_a_link_that_strands_boxes_offers_no_cut(tmp_path):
    # A second exposure taking the group's areas through a marker link leaves
    # its boxes outside membrane 1's: nothing was drawn, so no cut is offered.
    s, exposure, marker = marker_project(tmp_path)
    ops.set_marker_image(s, exposure, marker)
    second = import_blot(s, two_strips(), "exposure 2.tif", membrane_id="mem-2")
    p53 = ops.add_protein(s, "p53", Role.TARGET, arena_id=area_on(s, second))
    box_all(s, p53, 76)
    error = refused(lambda: ops.set_marker_image(s, second, marker), ErrorCode.ARENA_IN_USE)
    assert str(error) == (
        "the new outline would leave the boxes of 'p53' in lanes 1, 2, 3 outside membrane 1's"
        " area; draw the outline around them, or move or remove them first"
    )


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
    monkeypatch.setattr(ops, "_shown", lambda array, image, area, window, background: array)
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
    level = lambda: ref.background  # noqa: E731
    assert ops._shown(array, ref, whole, (0, 0, 10, 6), level) is array
    drawn = Arena(
        id="arena-1",
        image_id="img-2",
        outline=Outline(x0=2, y0=1, x1=6, y1=4),
        source="manual",
    )
    masked = ops._shown(array, ref, drawn, (0, 0, 10, 6), level)
    assert masked is not array and np.array_equal(array, np.arange(60).reshape(6, 10))
    inside = ops._inside_mask(drawn, ref, (0, 0, 10, 6))
    assert np.array_equal(masked[inside], array[inside])
    assert (masked[~inside] == round(ref.background)).all() or np.allclose(
        masked[~inside], ref.background, atol=1
    )
    # Only the window is read: the rest is never filled.
    part = ops._shown(array, ref, drawn, (0, 0, 3, 6), level)
    assert np.array_equal(part[:, :3], masked[:, :3])
    assert not part[:, 3:].any()


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


def cut_out_nets(s, arena_id: str) -> dict[str, float]:
    """The nets of the bands on an unturned drawn area as the image cut along
    its outline gives them: its pixels cropped there, its boxes shifted with
    them, the image's median the level a ring with too little membrane falls
    back to."""
    from proteia.core.model import Box
    from proteia.core.quantify import RING_CLAMP, band_backgrounds, net_signal

    batch = s.project.batch
    _, arena = batch.find_arena(arena_id)
    image = batch.find_image(arena.image_id)
    o = arena.outline
    x0, y0 = max(0, o.x0), max(0, o.y0)
    x1, y1 = min(image.width, o.x1), min(image.height, o.y1)
    pixels = s.pixels(image.id)[y0:y1, x0:x1]
    placed = [
        (batch.boxes_of(protein).box_size, band)
        for protein, _ in batch.rows_on_arena(arena_id)
        for band in batch.boxes_of(protein).bands
    ]
    shifted = [Box(x=band.box.x - x0, y=band.box.y - y0) for _, band in placed]
    dark = image.polarity.dark_on_light
    found = band_backgrounds(
        pixels,
        [box.rect(size) for box, (size, _) in zip(shifted, placed, strict=True)],
        [(size.width, size.height) for size, _ in placed],
        dark_on_light=dark,
        integral=image.bit_depth is not None,
        fallback=image.background,
    )
    return {
        band.id: net_signal(pixels, box, size, level.level, dark_on_light=dark, clamp=RING_CLAMP)
        for box, (size, band), level in zip(shifted, placed, found, strict=True)
    }


def test_each_strip_measures_as_if_cut_out_of_the_image(tmp_path):
    # Drawing an outline is cutting the image there: each strip's nets are
    # those of the image cut along its outline, though the rings of its
    # bands reach the next strip's boxes, as the same boxes on an image of
    # one membrane show.
    s, image, areas, proteins = nine_strips(tmp_path)
    stored = nets(s)
    for arena_id in areas:
        cut = cut_out_nets(s, arena_id)
        assert cut == {band: stored[band] for band in cut}, arena_id
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
    unsplit = {b.id: b.net for p in joined.batch.proteins for b in p.rows[0].bands}
    assert all(unsplit[band] != stored[band] for band in stored)


def test_the_ring_rule_is_the_area_s(tmp_path):
    # The one function that decides: a row's rings leave out the boxes of the
    # rows on its own area that have any, and read inside that area alone.
    s, image, top, bottom = split(tmp_path)
    box_all(s, top, 24)
    box_all(s, bottom, 76)
    ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, arena_id="arena-6")  # no box
    batch = s.project.batch
    for arena_id, protein in (("arena-4", top), ("arena-6", bottom)):
        arena = batch.find_arena(arena_id)[1]
        assert ops._ring_context(batch, arena) == RingContext(
            area=ops.area_key(arena),
            left_out=frozenset({batch.boxes_of(batch.find_protein(protein)).id}),
        )


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


def test_a_box_edit_of_one_membrane_leaves_the_others_nets(tmp_path):
    s, image, areas, proteins = nine_strips(tmp_path)
    before = s.project.batch.model_copy(deep=True)
    edit_strip_1(s, areas, proteins, "boxes")
    others_unchanged(s, before, proteins)


@pytest.mark.parametrize("what", ["lanes", "calibration"])
def test_lanes_and_marks_of_one_membrane_leave_the_others_alone(tmp_path, what):
    # The nets do not read the lane table or the marks, and each membrane's
    # curve is its own.
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
    # mem-1 has two register groups (the pair, a reprobe): the pair is named.
    assert told == {
        "membrane_id": "mem-1",
        "membrane": 1,
        "image_ids": ["img-2", "img-3"],
        "bands": len(lost),
        "message": "Membrane 1 has fewer than 2 ladder marks on 'β-actin 10 µM.tif' and"
        f" 'marker α.png': {len(lost)} bands lost their apparent MW; mark its ladder there to"
        " read them again",
    }
    assert ops._mw_lost(before, before) == ()
    # With two marks left, a band's MW lost (outside the range) is no split's.
    kept = before.model_copy(deep=True)
    for protein in kept.proteins:
        for band in kept.boxes_of(protein).bands:
            band.apparent_mw = None
    assert len(kept.membranes[0].calibration.points) >= 2
    assert all(told["membrane_id"] != "mem-1" for told in ops._mw_lost(before, kept))


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
    assert readme[start + 1 : start + 5] == [
        "---------",
        "",
        "Each membrane's label, its place in the project when this was exported, and",
        "where it lies on each of its images, in image pixels from the top left.",
    ]
    mapped = readme[readme.index("Membrane 1", start) :]
    assert mapped[:6] == [
        "Membrane 1",
        f"    {STRIPS}: the rectangle (0, 0)-(50, 100)",
        "Membrane 2",
        f"    {STRIPS}: the rectangle (100, 0)-(150, 100), turned -2.5°",
        "Membrane 3",
        f"    {STRIPS}: the rectangle (150, 0)-(200, 100)",
    ]
    # A long file name is kept whole on its line, hyphens and all.
    long_name = "-".join(["a-long-file-name"] * 6) + ".tif"
    lone = session_on(tmp_path / "long")
    import_blot(lone, two_strips(), long_name)
    ops.set_lanes(lone, [LaneInput("vehicle")])
    lines = (ops.export_bundle(lone, formats=[]).folder / "README.txt").read_text("utf-8")
    assert f"    {long_name}:" in lines.splitlines()
    # An image of one membrane: the whole image.
    lone, _ = one_image(tmp_path / "lone")
    lines = (ops.export_bundle(lone, formats=[]).folder / "README.txt").read_text("utf-8")
    assert lines.splitlines()[-2:] == ["Membrane 1", f"    {STRIPS}: the whole image"]


def test_area_lookups_refuse_what_is_not_there(tmp_path):
    s, image, _, _ = split(tmp_path)
    batch = s.project.batch
    with pytest.raises(ValueError, match="holds 2 membranes' areas"):
        batch.membrane_of(image)
    with pytest.raises(UnknownIdError):
        batch.membrane_of("img-99")
    stranger = batch.membranes[0].model_copy(update={"id": "mem-99"})
    with pytest.raises(UnknownIdError):
        batch.membrane_number(stranger)
    with pytest.raises(UnknownIdError, match="membrane mem-2 has no area on 'img-99'"):
        batch.area_on(batch.membranes[0], "img-99")
    assert [m.id for m in batch.membranes_on(image)] == ["mem-2", "mem-5"]
    assert [row.arena_id for _, row in batch.rows_on(image)] == ["arena-4", "arena-6"]


def test_a_lane_is_proposed_from_the_boxes_of_the_area(tmp_path):
    s, image, top, bottom = split(tmp_path)
    ops.place_box(s, top, 40, 24, lane_index=0, grow=False)
    ops.place_box(s, top, 100, 24, lane_index=1, grow=False)
    ops.place_box(s, bottom, 40, 76, lane_index=0, grow=False)
    # The bottom strip has a box of its own, so it reads its own lanes: one so far.
    error = refused(lambda: ops.place_box(s, bottom, 160, 76, grow=False), ErrorCode.LANE_REQUIRED)
    assert str(error) == (
        "choose the lane: position proposes one only once boxes in two lanes of membrane 2's"
        " area show the lane spacing"
    )


def test_an_area_without_boxes_reads_the_lanes_of_its_image(tmp_path):
    s, image, top, bottom = split(tmp_path)
    ops.place_box(s, top, 40, 24, lane_index=0, grow=False)
    ops.place_box(s, top, 100, 24, lane_index=1, grow=False)
    batch = s.project.batch
    # No box on the bottom strip yet: every box on the image anchors it, as
    # on an image of one membrane, so a click there is given a lane.
    assert lane_anchors(batch, "arena-6") == lane_anchors(batch, "arena-4")
    assert lane_anchors(batch, "arena-6") == [(40.5, 0), (100.5, 1)]
    band = ops.place_box(s, bottom, 160, 76, grow=False)
    assert s.project.batch.find_band(band)[1].lane_index == 2
    # From its first box on, its own boxes alone.
    assert lane_anchors(s.project.batch, "arena-6") == [(160.5, 2)]


def test_a_box_moved_into_two_areas_is_refused(tmp_path):
    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 60))
    ops.add_arena(s, image, (0, 40, 200, 100))
    protein = ops.add_protein(s, "p53", Role.TARGET, arena_id="arena-6")
    band = ops.place_box(s, protein, 100, 76, lane_index=1, grow=False)
    rect = s.project.batch.find_band(band)[1].box.rect(
        s.project.batch.boxes_of(s.project.batch.find_protein(protein)).box_size
    )
    up = (rect[0], rect[1] - 26, rect[2], rect[3] - 26)
    error = refused(lambda: ops.move_box(s, band, up), ErrorCode.IN_TWO_ARENAS)
    assert "lies inside both membrane 1's and membrane 2's areas" in str(error)


def test_a_kept_box_the_row_resizes_out_of_its_area_is_refused(tmp_path, monkeypatch):
    # The row keeps a box placed by hand and resizes it; were that box to
    # come out of the area (as the image's edge may shift it), the row is
    # refused as a move there would be.
    s, image, top, _ = split(tmp_path)
    band = ops.place_box(s, top, 40, 24, lane_index=0, grow=False)
    rect = s.project.batch.find_band(band)[1].box.rect(
        s.project.batch.boxes_of(s.project.batch.find_protein(top)).box_size
    )
    ops.move_box(s, band, (rect[0] + 1, rect[1], rect[2] + 1, rect[3]))  # edited: kept
    grow = ops.boxes.grow_to_fit_all

    def shifted(old, *args, **kwargs):
        size, resized, rects = grow(old, *args, **kwargs)
        return size, [(x0, y0 + 40, x1, y1 + 40) for x0, y0, x1, y1 in resized], rects

    monkeypatch.setattr(ops.boxes, "grow_to_fit_all", shifted)
    error = refused(lambda: ops.detect_row_boxes(s, top, (10, 12, 190, 38)), ErrorCode.OUT_OF_ARENA)
    assert "the box's centre would lie outside membrane 1's area" in str(error)


def test_an_area_operation_reaches_every_linked_image(tmp_path):
    # E and E2 linked to one marker M: an edit on E2 redraws the area on M
    # and, through M, on E.
    s, exposure, marker = marker_project(tmp_path)
    ops.set_marker_image(s, exposure, marker)
    second = import_blot(s, two_strips(), "exposure 2.tif", membrane_id="mem-2")
    ops.set_marker_image(s, second, marker)
    change = ops.edit_arena(s, area_on(s, second), outline=(0, 0, 200, 30))
    assert len(change.arena_ids) == 3
    batch = s.project.batch
    for image_id in (exposure, marker, second):
        assert batch.area_on(batch.membranes[0], image_id).outline == Outline(
            x0=0, y0=0, x1=200, y1=30
        )


def test_a_mark_across_the_width_is_read_on_the_outline_middle(tmp_path):
    # A mark saved without its x (before marks kept it), on an area over the
    # image's right part: read at the outline's middle (x 175), inside.
    from proteia.core.model import CalibrationPoint

    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(150, 0, 200, 100))
    point = CalibrationPoint(image_id=image, y=20.0, mw=100.0, source="strip_edge")

    def keep(draft: Project) -> None:
        draft.batch.membranes[0].calibration.points.append(point)

    from test_operations import plant

    plant(s, keep)
    ops.edit_arena(s, "arena-4", outline=(150, 0, 200, 99))
    assert s.project.batch.find_arena("arena-4")[1].outline.y1 == 99


def test_a_whole_image_area_masks_nothing():
    whole = Arena(id="arena-1", image_id="img-2", source="whole_image")
    ref = make_project().batch.find_image("img-2")
    assert ops._inside_mask(whole, ref, (0, 0, 7, 3)).tolist() == [[True] * 7] * 3


def test_removing_an_area_leaves_the_others_nets(tmp_path):
    # The rings of the other strips never left out strip 2's boxes nor read
    # its pixels: nothing is measured again, and no pixel is read.
    s, image, areas, proteins = nine_strips(tmp_path)
    before = nets(s)
    reads = []
    real = s.pixels

    def counted(image_id, **kwargs):
        reads.append(image_id)
        return real(image_id, **kwargs)

    s.pixels = counted
    removal = ops.remove_arena(s, areas[1])
    assert (removal.remeasured, removal.largest_change, reads) == ((), None, [])
    after = nets(s)
    assert after == {band: net for band, net in before.items() if band in after}
    s.pixels = real
    assert_nets_current(s)


def test_removing_an_area_without_boxes_reads_no_pixels(tmp_path):
    # The rows without boxes change no ring: removing their area reads no
    # pixel, so a stored file changed since is no reason to refuse it.
    s, image, top, bottom = split(tmp_path)
    box_all(s, top, 24)
    ops.save(s)
    reopened = ops.open_project(s.folder)
    stored = reopened.folder / storage.IMAGES_DIR / reopened.project.batch.find_image(image).file
    stored.write_bytes(b"not the image any more")
    removal = ops.remove_arena(reopened, "arena-6")
    assert removal.cascade.removed[:2] == ("arena-6", bottom)
    assert removal.remeasured == ()
    reopened.close()


def linked_pair_project(tmp_path: Path):
    """:func:`marker_project` linked, then a second exposure E2 linked to
    the same marker (it takes the group's three areas); a protein on
    membrane 2's area on each exposure, boxed on E."""
    s, exposure, marker = marker_project(tmp_path)
    ops.set_marker_image(s, exposure, marker)
    second = import_blot(s, two_strips(), "exposure 2.tif", membrane_id="mem-2")
    ops.set_marker_image(s, second, marker)
    on_e = ops.add_protein(s, "p53", Role.TARGET, arena_id=area_on(s, exposure, 2))
    ops.place_box(s, on_e, 100, 50, lane_index=1, grow=False)
    on_e2 = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, arena_id=area_on(s, second, 2))
    return s, (exposure, marker, second), (on_e, on_e2)


def test_removal_scope_follows_every_link(tmp_path):
    s, images, proteins = linked_pair_project(tmp_path)
    exposure, marker, second = images
    batch = s.project.batch
    scope = ops.area_removal(batch, area_on(s, second, 2))
    assert scope.image_ids == images  # in the batch's order
    assert scope.arena_ids == tuple(area_on(s, i, 2) for i in images)
    assert scope.protein_ids == proteins
    assert (scope.boxes, scope.records, scope.marks, scope.membrane_goes) == (1, 0, 0, True)
    # The removal takes exactly that.
    removal = ops.remove_arena(s, area_on(s, second, 2))
    assert removal.cascade.removed[:5] == (*scope.arena_ids, *proteins)


def test_marker_link_direction(tmp_path):
    s, exposure, marker = marker_project(tmp_path)
    batch = s.project.batch
    assert ops.marker_adoption(batch, exposure, marker) == "marker"  # M takes E's areas
    ops.set_marker_image(s, exposure, marker)
    batch = s.project.batch
    assert ops.marker_adoption(batch, exposure, marker) is None  # the same areas
    second = import_blot(s, two_strips(), "exposure 2.tif", membrane_id="mem-2")
    batch = s.project.batch
    assert ops.marker_adoption(batch, second, marker) == "image"  # E2 takes the group's
    update = ops.set_marker_image(s, second, marker)
    assert set(update.adopted_arena_ids) == {a.id for _, a in s.project.batch.arenas_on(second)}


def touching_strips(tmp_path: Path):
    """Two strips at 4°, one just below the other (0.07 px apart along
    their normal), each with a band near their shared edge, 10 px apart:
    the bands' signal joins across the edge."""
    t = math.radians(4.0)
    c1, c2 = (250.0, 60.0), (248.0, 90.0)
    b1 = (c1[0] - 10 * math.sin(t), c1[1] + 10 * math.cos(t))
    b2 = (c2[0] + 10 * math.sin(t), c2[1] - 10 * math.cos(t))
    y, x = np.mgrid[0:130, 0:500].astype(float)
    image = np.full((130, 500), LEVEL)
    for bx, by in (b1, b2):
        image -= 20000.0 * np.exp(-(((x - bx) / 15.0) ** 2) - ((y - by) / 4.0) ** 2)
    s = session_on(tmp_path)
    blot_id = import_blot(s, np.round(image).astype(np.uint16), "touching.tif")
    ops.set_lanes(s, [LaneInput("a"), LaneInput("b")])
    ops.edit_arena(s, "arena-4", outline=(50, 45, 450, 75), angle_deg=4.0)
    ops.add_arena(s, blot_id, (48, 75, 448, 105), angle_deg=4.0)
    protein = ops.add_protein(s, "A", Role.TARGET, arena_id="arena-4")
    return s, blot_id, protein


def test_a_seed_click_grows_within_its_area(tmp_path, monkeypatch):
    s, image, protein = touching_strips(tmp_path)
    band = ops.place_box(s, protein, 250, 70, lane_index=0, grow=True)
    batch = s.project.batch
    size = batch.boxes_of(batch.find_protein(protein)).box_size
    box = batch.find_band(band)[1].box
    assert box.y + size.height <= 77  # stopped at strip 1's edge (y 75 there)
    ops.undo(s)
    # Degraded: grown over the stored pixels, the box takes strip 2's band too.
    monkeypatch.setattr(
        ops, "_detector_pixels", lambda s_, image, area, window, **_: s_.pixels(image.id)
    )
    try:
        band = ops.place_box(s, protein, 250, 70, lane_index=0, grow=True)
    except OperationError:
        return  # refused: its centre left strip 1's area
    batch = s.project.batch
    size = batch.boxes_of(batch.find_protein(protein)).box_size
    assert batch.find_band(band)[1].box.y + size.height > 80


DETECTORS = (
    ("grow_box", None),
    ("detect_row", "rowdetect"),
    ("detect_row_along", "rowdetect"),
    ("find_ladder", "mwcal"),
    ("refine_point", "mwcal"),
    ("is_snap_position", "mwcal"),
)


def test_every_detector_reads_through_the_area_accessor(tmp_path, monkeypatch):
    # Every detector the operations call is given pixels from the one
    # accessor that shows a membrane area alone, on turned strips where every
    # window crosses an outline: no call site reads the stored pixels.
    import sys

    s, image, (a, _, c) = rotated_strips(tmp_path)
    produced: list[np.ndarray] = []
    real = ops._detector_pixels

    def recording(*args, **kwargs):
        produced.append(real(*args, **kwargs))
        return produced[-1]

    monkeypatch.setattr(ops, "_detector_pixels", recording)
    seen: list[tuple[str, np.ndarray]] = []
    for name, module in DETECTORS:
        owner = ops if module is None else getattr(ops, module)
        detector = getattr(owner, name)

        def spy(array, *args, _real=detector, _name=name, **kwargs):
            if sys._getframe(1).f_globals["__name__"] == ops.__name__:
                seen.append((_name, array))
            return _real(array, *args, **kwargs)

        monkeypatch.setattr(owner, name, spy)
    batch = s.project.batch
    bounds = batch.find_arena("arena-4")[1].bounds(batch.find_image(image))
    ops.detect_row_boxes(s, a, bounds)
    ops.place_box(s, c, 500, 253, lane_index=4, grow=True)  # strip 3's band in lane 5
    ops.set_ladder(s, "mem-2", "pageruler_plus/tris_glycine")
    ops.propose_ladder(s, "arena-4", 100.0)
    ops.snap_ladder(s, "arena-4", 100.0, [40.0, 60.0])
    ops.add_calibration_point(s, "arena-4", 40.0, 100.0, CHEMI_MARKER, x=100.0, snap=True)
    ops.set_ladder_points(
        s, "arena-4", "left", [(30.0, 100.0), (50.0, 70.0)], x=100.0, found_at=100.0
    )
    names = {name for name, _ in seen}
    assert names >= {"grow_box", "detect_row", "find_ladder", "refine_point", "is_snap_position"}
    for name, array in seen:
        assert any(np.shares_memory(array, shown) for shown in produced), name


def full_copy(session, image, area, window, *, calibration=False):
    """What the area accessor gave before it copied only the window: the
    whole image copied, the window masked."""
    array = session.calibration_pixels(image.id) if calibration else session.pixels(image.id)
    if area.outline is None:
        return array
    x0, y0 = max(0, window[0]), max(0, window[1])
    x1, y1 = min(image.width, window[2]), min(image.height, window[3])
    masked = array.copy()
    if x1 > x0 and y1 > y0:
        masked[y0:y1, x0:x1][~ops._inside_mask(area, image, (x0, y0, x1, y1))] = image.background
    return masked


def detections(folder: Path) -> list:
    """On :func:`rotated_strips` built in ``folder``: a row dragged over the
    bounds of strips 1 and 3 and a ruler snapped on each, what each gave."""
    s, image, (a, _, c) = rotated_strips(folder)
    out = []
    for protein, arena in ((a, "arena-4"), (c, "arena-8")):
        batch = s.project.batch
        bounds = batch.find_arena(arena)[1].bounds(batch.find_image(image))
        out.append(ops.detect_row_boxes(s, protein, bounds))
        out.append(s.project.batch.model_dump())
        out.append(ops.snap_ladder(s, arena, 300.0, [bounds[1] + 10.0, bounds[3] - 10.0]))
        ops.undo(s)
    return out


def test_detector_pixels_equal_a_full_copy(tmp_path, monkeypatch):
    windowed = detections(tmp_path / "windowed")
    monkeypatch.setattr(ops, "_detector_pixels", full_copy)
    assert detections(tmp_path / "copied") == windowed
    # A band its strip's outline cuts, grown from a click (the full copy
    # still in force), then by the accessor that copies only the window.
    t, _, protein = touching_strips(tmp_path / "touching")
    band = ops.place_box(t, protein, 250, 70, lane_index=0, grow=True)
    grown = t.project.batch.find_band(band)[1].box
    ops.undo(t)
    monkeypatch.undo()
    band = ops.place_box(t, protein, 250, 70, lane_index=0, grow=True)
    assert t.project.batch.find_band(band)[1].box == grown


def test_a_window_inside_its_area_is_read_as_stored(tmp_path):
    s, image, _ = rotated_strips(tmp_path)
    batch = s.project.batch
    arena = batch.find_arena("arena-4")[1]
    stored = s.pixels(image)
    shown = ops._detector_pixels(s, batch.find_image(image), arena, (500, 65, 600, 75))
    assert shown is stored  # every pixel of the window lies inside: nothing to mask
    shown = ops._detector_pixels(s, batch.find_image(image), arena, (40, 15, 1060, 125))
    assert shown is not stored
    assert not shown[200:, :].any()  # past the window: never read, never filled


def test_a_ladder_on_a_chosen_channel_reads_the_mean_off_its_area(tmp_path):
    # A colour image measured on green (#206), split in two: a row reads
    # green, a ladder the mean of the colours, each with that array's own
    # median off the area. Red's background is raised, so the two differ.
    from proteia.core.model import Channel
    from proteia.core.quantify import estimate_background
    from test_channels import H, W, colour_blot, import_pixels

    s = session_on(tmp_path)
    rgb = colour_blot(16)
    rgb[..., 0] += 20000
    image = import_pixels(s, rgb, "GFP µ.tif")
    ops.set_channel(s, image, Channel.GREEN)
    area = area_on(s, image)
    ops.edit_arena(s, area, outline=(0, 0, W, H // 2))
    ops.add_arena(s, image, (0, H // 2, W, H))
    ref = s.project.batch.find_image(image)
    arena = s.project.batch.find_arena(area)[1]
    window = (0, 0, W, H)
    off = ~ops._inside_mask(arena, ref, window)
    measured = ops._detector_pixels(s, ref, arena, window)
    ladder = ops._detector_pixels(s, ref, arena, window, calibration=True)
    assert np.array_equal(measured[~off], s.pixels(image)[~off])
    assert np.array_equal(ladder[~off], s.calibration_pixels(image)[~off])
    mean_level = estimate_background(s.calibration_pixels(image))
    assert abs(mean_level - ref.background) > 5000
    assert np.allclose(measured[off], ref.background, atol=1)
    assert np.allclose(ladder[off], mean_level, atol=1)


def test_a_window_on_its_area_s_edges_is_read_as_stored(tmp_path, monkeypatch):
    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    ops.add_arena(s, image, (0, 50, 200, 100))
    batch = s.project.batch
    ref, arena = batch.find_image(image), batch.find_arena("arena-4")[1]
    stored = s.pixels(image)
    # The window the outline is: every pixel's centre lies inside, the edge
    # pixels' too, so nothing is masked; one row more, and the window is.
    assert ops._detector_pixels(s, ref, arena, (0, 0, 200, 50)) is stored
    assert ops._detector_pixels(s, ref, arena, (0, 0, 200, 51)) is not stored
    # A ladder read of an image measured on the mean takes the stored
    # median off the area: no median is worked out again.

    def no_median(array):
        raise AssertionError("a median worked out again")

    monkeypatch.setattr(ops, "estimate_background", no_median)
    monkeypatch.setattr(session_module, "estimate_background", no_median)
    shown = ops._detector_pixels(s, ref, arena, (0, 0, 200, 60), calibration=True)
    assert np.allclose(shown[50:60, :], ref.background, atol=1)


# --- Review round 3 ---

PRESET = "pageruler_plus/tris_glycine"


def ladder_of(s, membrane_id: str) -> tuple:
    [calibration] = [m.calibration for m in s.project.batch.membranes if m.id == membrane_id]
    return calibration.ladder, list(calibration.ladder_kda)


def marked_middle(tmp_path: Path):
    """:func:`marker_project` linked, PageRuler Plus chosen for membrane 2
    (mem-5, the middle third), two of its marks on the marker."""
    s, exposure, marker = marker_project(tmp_path)
    ops.set_marker_image(s, exposure, marker)
    ops.set_ladder(s, "mem-5", PRESET)
    middle = area_on(s, marker, 2)
    for y, kda in ((40.0, 100.0), (60.0, 70.0)):
        ops.add_calibration_point(s, middle, y, kda, STRIP_EDGE, x=5.0, snap=False)
    return s, exposure, marker


def test_a_new_membrane_given_an_area_takes_its_ladder(tmp_path):
    s, exposure, marker = marked_middle(tmp_path)
    moved = ops.assign_arena(s, area_on(s, exposure, 2), None)
    assert moved.moved_points == 2
    assert ladder_of(s, moved.membrane_id) == (PRESET, list(ladders.preset(PRESET).kda))


def test_an_area_with_marks_joins_a_membrane_of_its_ladder_only(tmp_path):
    s, exposure, marker = marked_middle(tmp_path)
    other = import_blot(s, two_strips(), "D.tif")
    target = s.project.batch.membranes_on(other)[0].id
    preset = ladders.preset(PRESET)
    error = refused(
        lambda: ops.assign_arena(s, area_on(s, exposure, 2), target), ErrorCode.INVALID_INPUT
    )
    assert str(error) == (
        "the 2 ladder marks on membrane 2's area would join membrane 4, whose ladder is not"
        f" chosen, while membrane 2's is {preset.product} ({preset.system}); choose the same"
        " ladder for both membranes first, or remove the marks from this area"
    )
    # An area without marks joins it whatever its ladder; with the same
    # ladder, one with marks too.
    ops.set_ladder(s, "mem-7", PRESET)
    ops.assign_arena(s, area_on(s, exposure, 3), target)
    ops.undo(s)
    ops.set_ladder(s, target, PRESET)
    moved = ops.assign_arena(s, area_on(s, exposure, 2), target)
    assert (moved.membrane_id, moved.moved_points) == (target, 2)


def test_what_lies_on_the_cut_stays_with_the_outline_drawn(tmp_path):
    # A box, a not-detected record and a mark across the whole width, each
    # centred on the cut line: the edge two areas share belongs to the
    # earlier membrane's area, as the cut reads them, so they stay.
    from proteia.core.model import CalibrationPoint, Region, UndetectedBand

    s, image, p53, gapdh = two_rows(tmp_path)
    actin = ops.add_protein(
        s, "β-actin", Role.TARGET, image, box_size=ops.BoxSize(width=10, height=6)
    )
    ops.place_box(s, actin, 194, 50, lane_index=2, grow=False)  # centred at (194, 50)

    def on_the_line(draft: Project) -> None:
        own = draft.batch.boxes_of(draft.batch.find_protein(actin))
        own.undetected = [
            UndetectedBand(
                lane_index=1,
                reason="below_detection_limit",
                snr=1.0,
                threshold=6.0,
                region=Region(x0=90, y0=47, x1=110, y1=53),
                source="row_box",
            )
        ]
        draft.batch.membranes[0].calibration.points.append(
            CalibrationPoint(image_id=image, y=50.0, mw=100.0, source=STRIP_EDGE)
        )

    plant(s, on_the_line)
    change = ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    assert (change.moved_proteins, change.moved_points) == ((gapdh,), 0)
    batch = s.project.batch
    assert batch.arena_id_of(batch.find_protein(actin)) == "arena-4"
    assert [p.y for p in batch.membranes[0].calibration.points] == [50.0]


def test_a_membrane_s_group_left_without_a_ladder_is_told(tmp_path):
    # Membrane 1 on a marker-linked pair (marks on the marker) and on an
    # exposure of its own (two strip-edge marks there): the cut leaves the
    # pair one mark, so p53's bands there lose their MW, though the membrane
    # keeps three marks in all.
    s = session_on(tmp_path)
    exposure = import_blot(s, two_strips(), "E.tif")
    ops.set_lanes(s, [LaneInput("a"), LaneInput("b"), LaneInput("c")])
    membrane = s.project.batch.membrane_of(exposure).id
    marker = import_blot(
        s, two_strips(), "M marker.tif", kind=ImageKind.VISIBLE_MARKER, membrane_id=membrane
    )
    ops.set_marker_image(s, exposure, marker)
    alone = import_blot(s, two_strips(), "F.tif", membrane_id=membrane)
    for y, kda in ((10.0, 250.0), (90.0, 25.0)):
        ops.add_calibration_point(s, alone, y, kda, STRIP_EDGE, x=5.0, snap=False)
    for y, kda in ((10.0, 250.0), (60.0, 70.0), (90.0, 25.0)):
        ops.add_calibration_point(s, marker, y, kda, STRIP_EDGE, x=5.0, snap=False)
    gapdh = ops.add_protein(s, "GAPDH", Role.LOADING_CONTROL, exposure)
    p53 = ops.add_protein(s, "p53 µ", Role.TARGET, exposure, loading_control_ids=[gapdh])
    box_all(s, p53, 24)
    box_all(s, gapdh, 76)
    change = ops.edit_arena(s, area_on(s, exposure), outline=(0, 0, 200, 50))
    assert change.moved_points == 2
    assert [lost["membrane"] for lost in change.mw_lost] == [1]
    [lost] = change.mw_lost
    assert (lost["bands"], lost["image_ids"]) == (3, [exposure, marker])
    assert lost["message"] == (
        "Membrane 1 has fewer than 2 ladder marks on 'E.tif' and 'M marker.tif': 3 bands lost"
        " their apparent MW; mark its ladder there to read them again"
    )
    assert_mw_current(s)


def test_removal_scopes_and_link_directions_are_built_once_per_state(tmp_path):
    # Built for every area and every image at once, they are what each one
    # alone gives, on images linked through one marker.
    s, images, proteins = linked_pair_project(tmp_path)
    batch = s.project.batch
    scopes = ops.area_removals(batch)
    assert scopes == {
        arena.id: ops.area_removal(batch, arena.id)
        for membrane in batch.membranes
        for arena in batch.arenas_of(membrane)
    }
    third = import_blot(s, two_strips(), "exposure 3.tif", membrane_id="mem-2")
    batch = s.project.batch
    directions = ops.marker_adoptions(batch)
    assert directions[third] == {images[1]: "image"}
    for image_id, by_marker in directions.items():
        for marker_id, direction in by_marker.items():
            assert direction == ops.marker_adoption(batch, image_id, marker_id)


def test_a_ladder_read_works_its_median_out_once_per_operation(tmp_path, monkeypatch):
    from proteia.core.model import Channel
    from proteia.core.quantify import estimate_background
    from test_channels import H, W, colour_blot, import_pixels

    s = session_on(tmp_path)
    rgb = colour_blot(16)
    rgb[..., 0] += 20000
    image = import_pixels(s, rgb, "GFP µ.tif")
    ops.set_channel(s, image, Channel.GREEN)
    area = area_on(s, image)
    ops.edit_arena(s, area, outline=(0, 0, W, H // 2))
    ops.add_arena(s, image, (0, H // 2, W, H))
    ref = s.project.batch.find_image(image)
    arena = s.project.batch.find_arena(area)[1]
    worked: list[float] = []

    def counted(array):
        worked.append(estimate_background(array))
        return worked[-1]

    monkeypatch.setattr(session_module, "estimate_background", counted)
    monkeypatch.setattr(ops, "estimate_background", counted)
    # A window inside the area masks nothing: no median.
    ops._detector_pixels(s, ref, arena, (0, 0, W, H // 4), calibration=True)
    assert worked == []
    # Within one operation, once; the next operation works it out again.
    with s.transaction():
        ops._detector_pixels(s, ref, arena, (0, 0, W, H), calibration=True)
        again = ops._detector_pixels(s, ref, arena, (0, 10, W, H), calibration=True)
    assert len(worked) == 1
    assert np.allclose(again[H // 2 :, :], worked[0], atol=1)  # the median kept, not lost
    with s.transaction():
        ops._detector_pixels(s, ref, arena, (0, 0, W, H), calibration=True)
    assert len(worked) == 2
    # A commit within an operation drops it with the arrays read before it.
    with s.transaction():
        ops._detector_pixels(s, ref, arena, (0, 0, W, H), calibration=True)
        ops.set_channel(s, image, Channel.RED)  # which works out red's median
        ref, done = s.project.batch.find_image(image), len(worked)
        ops._detector_pixels(s, ref, arena, (0, 0, W, H), calibration=True)
        assert len(worked) == done + 1 and worked[-1] == worked[0]


def test_a_point_inside_one_area_on_another_s_edge_is_in_two(tmp_path):
    # Membrane 2's area reaching ten rows into membrane 1's: a click on
    # membrane 1's lower edge lies inside membrane 2's past its edge, so in both.
    s, image = one_image(tmp_path)
    ops.edit_arena(s, "arena-4", outline=(0, 0, 200, 50))
    ops.add_arena(s, image, (0, 40, 200, 100))
    bottom = ops.add_protein(s, "p53 µ", Role.TARGET, arena_id="arena-6")
    error = refused(
        lambda: ops.place_box(s, bottom, 100, 50, lane_index=1, grow=False),
        ErrorCode.IN_TWO_ARENAS,
    )
    assert str(error) == (
        f"(100, 50) lies inside both membrane 1's and membrane 2's areas on {STRIPS!r}; redraw"
        " one of them so they do not overlap there"
    )


def test_a_register_group_with_two_marks_keeps_its_ladder():
    # mem-1's pair (img-2, img-3) left two marks: a band's MW lost there
    # (outside the range) is not a group left without a ladder.
    before = make_project().batch
    after = before.model_copy(deep=True)
    mem_1 = after.membranes[0]
    mem_1.calibration.points = [p for p in mem_1.calibration.points if p.image_id == "img-3"]
    after.find_band("band-12")[1].apparent_mw = None
    assert before.find_band("band-12")[1].apparent_mw is not None
    assert ops._mw_lost(before, after) == ()


def test_drawing_an_area_refits_a_calibration_from_before_58(tmp_path):
    # The sample's mem-1 holds a fit no #58 build made. Its area drawn on a
    # linked pair changes its marker links, so it is refitted in the same
    # change, and band-12's MW, from the older fit, is read again.
    from test_operations import band_of, expected_mw, open_sample

    s = open_sample(tmp_path, hook=None)
    stored = band_of(s, "band-12").apparent_mw
    assert stored == 90.5 != expected_mw(s, "band-12")
    exposure = import_blot(s, two_strips(), "E.tif")
    membrane = s.project.batch.membranes_on(exposure)[0].id
    marker = import_blot(
        s, two_strips(), "M marker.tif", kind=ImageKind.VISIBLE_MARKER, membrane_id=membrane
    )
    ops.set_marker_image(s, exposure, marker)
    ops.edit_arena(s, area_on(s, exposure), outline=(0, 0, 200, 50))
    before = s.project
    ops.add_arena(s, exposure, (0, 50, 200, 100), membrane_id="mem-1")
    assert band_of(s, "band-12").apparent_mw == expected_mw(s, "band-12") != stored
    assert before.batch.find_band("band-12")[1].apparent_mw == stored  # left as it was
    calibration = s.project.batch.membranes[0].calibration  # mem-1's
    assert calibration.fit_method.value == "log_linear_piecewise"
    ops.undo(s)  # the project before it, as it was
    assert band_of(s, "band-12").apparent_mw == stored
