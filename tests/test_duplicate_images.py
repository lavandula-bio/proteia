# SPDX-License-Identifier: Apache-2.0
"""Images of the same file (#216).

A file may be imported more than once, under any name and into any membrane;
the project then holds images with the same SHA-256. The results say so, for
each group of such images that can hold bands (every kind but the visible-light
marker, on which no protein may be): a ``duplicate_images_measured`` warning
when two or more of them are measured (hold a box or a not-detected record of
any protein), since a band measured on two of them is counted twice; otherwise
a ``duplicate_images`` note, so the user learns of it before boxing. Each
names the images by file name, with their place in the project's image list
where two images of the project share a name, and counts their membranes. It
is worked out from the images and proteins the project holds now, never
stored, once per computation (not per result set): removing a copy or its
boxes, undo and redo update it, and a project saved before shows it when opened.

The results tests build their projects from the model (no pixels): the
conftest sample's images, in image-list order 'β-actin 10 µM.tif' (img-2,
β-catenin's), 'marker α.png' (img-3, img-2's visible-light marker) and
'reprobe β.tif' (img-4, GAPDH's) on one membrane and 'α-tubulin µ.jpg'
(img-6, α-tubulin's) on another, given each other's hashes. The operations
tests import a synthetic 16-bit TIFF; test_web_api has the routes'.
"""

import hashlib
import io
import json
from collections.abc import Callable, Collection, Mapping
from pathlib import Path

import numpy as np
import pytest
import tifffile
from pydantic import ValidationError

from conftest import (
    FakeClock,
    image_bytes,
    make_project,
    make_project_with_undetected,
    sample_doc_v1,
    write_tiff,
)
from proteia.core import operations as ops
from proteia.core import storage
from proteia.core.export import BUNDLE_README_FILE, BUNDLE_RECORD_FILE
from proteia.core.model import ImageKind, Polarity, Project, apply_change
from proteia.core.operations import LaneInput, ProjectSession
from proteia.core.results import Level, Notice, NoticeCode, compute_results
from proteia.core.session import save_to_folder
from test_operations import FOLDER, NARROW_X, ROW, WIDE_X, blot, session_on

NOTE, WARNING = "duplicate_images", "duplicate_images_measured"
# The ends of the note and of the warning, of two images or more.
NOTE_TWO = (
    ": a band boxed on both would be counted twice. If one should be a different image,"
    " import that image's file; nothing needs to change for a copy imported on purpose"
)
NOTE_MORE = (
    ": a band boxed on more than one of them would be counted more than once. If one should"
    " be a different image, import that image's file; nothing needs to change for a copy"
    " imported on purpose"
)
WARN_TWO = (
    ": a band measured on both is counted twice. Check that no band is measured on both;"
    " if one should be a different image, import that image's file"
)
WARN_MORE = (
    ": a band measured on more than one of them is counted more than once. Check that no"
    " band is measured on more than one of them; if one should be a different image,"
    " import that image's file"
)
SAMPLE = ("img-2", "img-3", "img-4", "img-6")
# The bytes of each sample image's stand-in file, by its SHA-256.
STAND_INS = {hashlib.sha256(image_bytes(i)).hexdigest(): image_bytes(i) for i in SAMPLE}
CHEMI, MARKER, MERGED = ImageKind.CHEMILUMINESCENCE, ImageKind.VISIBLE_MARKER, ImageKind.MERGED
DARK = Polarity.DARK_ON_LIGHT


def changed(
    project: Project,
    *,
    copies: Mapping[str, str] | None = None,
    names: Mapping[str, str] | None = None,
    kinds: Mapping[str, ImageKind] | None = None,
    unmeasured: Collection[str] = (),
    edit: Callable[[Project], None] | None = None,
) -> Project:
    """``project`` with each image of ``copies`` holding the bytes of the image
    it maps to (its SHA-256), each image of ``names`` renamed, each image of
    ``kinds`` of that kind, the proteins on the ``unmeasured`` images without
    boxes or not-detected records, and ``edit`` made last."""

    def change(draft: Project) -> None:
        hashes = {image.id: image.sha256 for image in draft.batch.iter_images()}
        for image_id, source in (copies or {}).items():
            draft.batch.find_image(image_id).sha256 = hashes[source]
        for image_id, name in (names or {}).items():
            draft.batch.find_image(image_id).original_name = name
        for image_id, kind in (kinds or {}).items():
            draft.batch.find_image(image_id).kind = kind
        for protein in draft.batch.proteins:
            if protein.image_id in unmeasured:
                protein.bands, protein.undetected = [], []
        if edit is not None:
            edit(draft)

    return apply_change(project, change)[0]


def duplicates(notices: list[Notice]) -> list[tuple[str, Level, str, tuple[str, ...]]]:
    """The duplicate notices: (code, level, message, image ids)."""
    return [(n.code, n.level, n.message, n.image_ids) for n in notices if n.code in (NOTE, WARNING)]


def others(notices: list[Notice]) -> list[Notice]:
    return [n for n in notices if n.code not in (NOTE, WARNING)]


def warned(message: str, *ids: str) -> tuple[str, Level, str, tuple[str, ...]]:
    return (WARNING, Level.WARNING, message, ids)


def noted(message: str, *ids: str) -> tuple[str, Level, str, tuple[str, ...]]:
    return (NOTE, Level.INFO, message, ids)


# --- From the stored images and proteins ---


def test_two_measured_copies_are_warned_of_once_before_every_other_check():
    res = compute_results(changed(make_project(), copies={"img-4": "img-2"}).batch)
    message = (
        "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes),"
        " on one membrane, and both are measured" + WARN_TWO
    )
    # The first of the checks, about the two images and nothing else.
    assert res.notices[0] == Notice(
        code=NoticeCode(WARNING),
        level=Level.WARNING,
        message=message,
        image_ids=("img-2", "img-4"),
    )
    assert duplicates(res.notices) == [warned(message, "img-2", "img-4")]
    # It is about the images, not a set's lanes: the set the include flags
    # apply to has it, and the all-lanes set does not.
    assert res.all_lanes is not None and duplicates(res.all_lanes.notices) == []
    # It adds a check and changes no other.
    plain = compute_results(make_project().batch)
    assert others(res.notices) == plain.notices
    assert res.all_lanes.notices == plain.all_lanes.notices


@pytest.mark.parametrize(
    "unmeasured",
    [
        pytest.param({"img-2", "img-4"}, id="neither-measured"),
        pytest.param({"img-4"}, id="the-copy-not-measured"),
        pytest.param({"img-2"}, id="the-first-not-measured"),
    ],
)
def test_copies_not_both_measured_are_a_note(unmeasured):
    project = changed(make_project(), copies={"img-4": "img-2"}, unmeasured=unmeasured)
    res = compute_results(project.batch)
    message = (
        "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes),"
        " on one membrane" + NOTE_TWO
    )
    assert duplicates(res.notices) == [noted(message, "img-2", "img-4")]
    assert res.notices[0].code == NOTE


def test_a_not_detected_record_measures_an_image_as_a_box_does():
    # GAPDH (img-4) keeps its not-detected records and loses its boxes.
    def boxless(draft: Project) -> None:
        draft.batch.find_protein("prot-9").bands = []

    project = changed(make_project_with_undetected(), copies={"img-4": "img-2"}, edit=boxless)
    assert project.batch.find_protein("prot-9").undetected
    assert [code for code, *_ in duplicates(compute_results(project.batch).notices)] == [WARNING]
    project = changed(project, unmeasured={"img-4"})
    assert [code for code, *_ in duplicates(compute_results(project.batch).notices)] == [NOTE]


def test_a_marker_copy_is_never_counted():
    # One file imported as the signal image and as its visible-light marker,
    # linked: no band can be on the marker, so nothing is said.
    project = changed(make_project(), copies={"img-3": "img-2"})
    assert project.batch.find_image("img-2").marker_image_id == "img-3"
    assert duplicates(compute_results(project.batch).notices) == []
    # With a third copy that can hold bands, the two that can are named.
    project = changed(make_project(), copies={"img-3": "img-2", "img-4": "img-2"})
    assert duplicates(compute_results(project.batch).notices) == [
        warned(
            "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes),"
            " on one membrane, and both are measured" + WARN_TWO,
            "img-2",
            "img-4",
        )
    ]


@pytest.mark.parametrize(
    ("unmeasured", "expected"),
    [
        # A merged copy used as a marker only: a note.
        pytest.param(
            {"img-4"},
            noted(
                "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes),"
                " on one membrane" + NOTE_TWO,
                "img-2",
                "img-4",
            ),
            id="the-merged-copy-not-measured",
        ),
        # Bands measured on the merged copy too: a warning.
        pytest.param(
            set(),
            warned(
                "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes),"
                " on one membrane, and both are measured" + WARN_TWO,
                "img-2",
                "img-4",
            ),
            id="both-measured",
        ),
    ],
)
def test_a_merged_copy_can_hold_bands(unmeasured, expected):
    project = changed(
        make_project(), copies={"img-4": "img-2"}, kinds={"img-4": MERGED}, unmeasured=unmeasured
    )
    assert duplicates(compute_results(project.batch).notices) == [expected]


@pytest.mark.parametrize("kind", list(ImageKind))
def test_the_kinds_that_can_hold_bands_are_those_the_model_takes_proteins_on(kind):
    # The notice reads the model's rule without importing it: kept in step here.
    try:
        changed(make_project(), kinds={"img-4": kind})  # GAPDH's image
        takes_proteins = True
    except ValidationError:
        takes_proteins = False
    assert takes_proteins is (kind is not MARKER)

    def unlinked(draft: Project) -> None:
        draft.batch.find_image("img-2").marker_image_id = None

    project = changed(
        make_project(), copies={"img-3": "img-2"}, kinds={"img-3": kind}, edit=unlinked
    )
    found = [ids for *_, ids in duplicates(compute_results(project.batch).notices)]
    assert found == ([("img-2", "img-3")] if takes_proteins else [])


@pytest.mark.parametrize(
    "copies",
    [
        pytest.param({"img-6": "img-2"}, id="the-copy-on-the-later-membrane"),
        pytest.param({"img-2": "img-6"}, id="the-copy-on-the-earlier-membrane"),
    ],
)
def test_copies_on_two_membranes_are_named_in_the_image_lists_order(copies):
    res = compute_results(changed(make_project(), copies=copies).batch)
    assert duplicates(res.notices) == [
        warned(
            "'β-actin 10 µM.tif' and 'α-tubulin µ.jpg' are the same file (identical bytes),"
            " on 2 membranes, and both are measured" + WARN_TWO,
            "img-2",
            "img-6",
        )
    ]


@pytest.mark.parametrize(
    ("names", "named"),
    [
        # Both copies under one name: each by its place in the image list.
        pytest.param(
            {"img-4": "β-actin 10 µM.tif"},
            "'β-actin 10 µM.tif' (image 1) and 'β-actin 10 µM.tif' (image 3)",
            id="both-copies",
        ),
        # A name the copy shares with an image of other bytes.
        pytest.param(
            {"img-6": "reprobe β.tif"},
            "'β-actin 10 µM.tif' and 'reprobe β.tif' (image 3)",
            id="a-copy-and-another-image",
        ),
        # Or with a marker image, which the image list shows too.
        pytest.param(
            {"img-3": "reprobe β.tif"},
            "'β-actin 10 µM.tif' and 'reprobe β.tif' (image 3)",
            id="a-copy-and-a-marker",
        ),
        # Names that differ in case are other names: no place.
        pytest.param(
            {"img-6": "Reprobe β.tif"},
            "'β-actin 10 µM.tif' and 'reprobe β.tif'",
            id="another-case",
        ),
    ],
)
def test_a_name_two_images_of_the_project_share_comes_with_its_place(names, named):
    project = changed(make_project(), copies={"img-4": "img-2"}, names=names)
    message = (
        f"{named} are the same file (identical bytes), on one membrane, and both are measured"
        + WARN_TWO
    )
    assert duplicates(compute_results(project.batch).notices) == [warned(message, "img-2", "img-4")]


def test_the_name_of_an_image_with_other_bytes_is_no_duplicate():
    project = changed(make_project(), names={"img-4": "β-actin 10 µM.tif"})
    # Renaming changes no check: the copies are found by their stored hashes.
    assert compute_results(project.batch).notices == compute_results(make_project().batch).notices


def test_each_group_is_listed_by_the_place_of_its_first_image():
    # With img-3 a merged image (it may still be img-2's marker), 'marker α.png'
    # and 'reprobe β.tif' (images 2, 3) are one file, and 'β-actin 10 µM.tif'
    # and 'α-tubulin µ.jpg' (images 1, 4) another: the group whose first image
    # comes first is listed first, though its last comes last.
    project = changed(
        make_project(),
        copies={"img-6": "img-2", "img-4": "img-3"},
        kinds={"img-3": MERGED},
        unmeasured={"img-2", "img-4", "img-6"},
    )
    res = compute_results(project.batch)
    assert duplicates(res.notices) == [
        noted(
            "'β-actin 10 µM.tif' and 'α-tubulin µ.jpg' are the same file (identical bytes),"
            " on 2 membranes" + NOTE_TWO,
            "img-2",
            "img-6",
        ),
        noted(
            "'marker α.png' and 'reprobe β.tif' are the same file (identical bytes),"
            " on one membrane" + NOTE_TWO,
            "img-3",
            "img-4",
        ),
    ]
    assert [n.code for n in res.notices[:2]] == [NOTE, NOTE]


@pytest.mark.parametrize(
    ("unmeasured", "expected"),
    [
        pytest.param(
            {"img-4", "img-6"},
            noted(
                "'β-actin 10 µM.tif', 'reprobe β.tif' and 'α-tubulin µ.jpg' are the same file"
                " (identical bytes), on 2 membranes" + NOTE_MORE,
                "img-2",
                "img-4",
                "img-6",
            ),
            id="one-measured",
        ),
        pytest.param(
            {"img-6"},
            warned(
                "'β-actin 10 µM.tif', 'reprobe β.tif' and 'α-tubulin µ.jpg' are the same file"
                " (identical bytes), on 2 membranes, and 2 of them are measured" + WARN_MORE,
                "img-2",
                "img-4",
                "img-6",
            ),
            id="two-measured",
        ),
        pytest.param(
            set(),
            warned(
                "'β-actin 10 µM.tif', 'reprobe β.tif' and 'α-tubulin µ.jpg' are the same file"
                " (identical bytes), on 2 membranes, and all 3 are measured" + WARN_MORE,
                "img-2",
                "img-4",
                "img-6",
            ),
            id="all-measured",
        ),
    ],
)
def test_three_copies_are_each_named(unmeasured, expected):
    project = changed(
        make_project(), copies={"img-4": "img-2", "img-6": "img-2"}, unmeasured=unmeasured
    )
    assert duplicates(compute_results(project.batch).notices) == [expected]


@pytest.mark.parametrize(
    ("names", "first"),
    [
        pytest.param({}, "'β-actin 10 µM.tif'", id="its-own-name"),
        pytest.param({"img-6": "β-actin 10 µM.tif"}, "'β-actin 10 µM.tif' (image 1)", id="shared"),
    ],
)
def test_more_than_three_copies_name_the_first_and_count_the_others(names, first):
    # img-3, a merged image, holds no band: three of the four are measured.
    project = changed(
        make_project(),
        copies={"img-3": "img-2", "img-4": "img-2", "img-6": "img-2"},
        kinds={"img-3": MERGED},
        names=names,
    )
    assert duplicates(compute_results(project.batch).notices) == [
        warned(
            f"{first} and 3 more copies are the same file (identical bytes), on 2 membranes,"
            " and 3 of them are measured" + WARN_MORE,
            "img-2",
            "img-3",
            "img-4",
            "img-6",
        )
    ]


def test_a_project_without_lanes_or_proteins_is_noted_too():
    def bare(draft: Project) -> None:
        draft.batch.proteins, draft.batch.lanes = [], []
        draft.batch.reference_condition = None

    project = changed(make_project(), copies={"img-4": "img-2"}, edit=bare)
    res = compute_results(project.batch)
    assert [n.code for n in res.notices] == [NOTE, "no_target", "no_loading_control", "no_lanes"]
    assert res.notices[0].level is Level.INFO


# --- Imports, boxes, removals, undo and redo ---


def import_file(
    s: ProjectSession,
    data: bytes,
    name: str,
    *,
    membrane_id: str | None = None,
    kind: ImageKind = CHEMI,
) -> str:
    """Import ``data`` as the file ``name``: the same bytes under any name."""
    return ops.import_image(
        s, io.BytesIO(data), name, kind=kind, polarity=DARK, membrane_id=membrane_id
    )


def said(s: ProjectSession) -> list[tuple[str, Level, str, tuple[str, ...]]]:
    """The duplicate notices of the session's results, as they are now."""
    return duplicates(ops.compute(s).notices)


def nudged(pixels: np.ndarray) -> np.ndarray:
    """``pixels`` with one pixel a level darker: other bytes, almost the same image."""
    other = pixels.copy()
    other[0, 0] -= 1
    return other


def test_an_import_of_the_same_bytes_is_noted_and_nothing_is_stored(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    first = import_file(s, data, "β-actin 10 µM.tif")
    assert said(s) == []
    saved_alone = (s.folder / storage.PROJECT_FILE).read_bytes()
    membrane = s.project.batch.membrane_of(first).id
    copy = import_file(s, data, "copy α.tif", membrane_id=membrane)
    message = (
        "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes),"
        " on one membrane" + NOTE_TWO
    )
    assert said(s) == [noted(message, first, copy)]
    # Nothing is stored with either image, so the project file holds no word of it.
    assert [i.import_warnings for i in s.project.batch.iter_images()] == [[], []]
    saved = (s.folder / storage.PROJECT_FILE).read_bytes()
    assert saved == storage.project_to_json(s.project)
    assert b"same file" not in saved and NOTE.encode() not in saved
    # The first image's record is as it was before the copy came.
    alone = json.loads(saved_alone)["batch"]["membranes"][0]["images"]
    assert json.loads(saved)["batch"]["membranes"][0]["images"][:1] == alone


def test_the_same_bytes_in_another_membrane_are_named_in_the_image_lists_order(tmp_path):
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    first = import_file(s, data, "β-actin 10 µM.tif")
    copy = import_file(s, data, "copy α.tif")  # a new membrane
    assert said(s) == [
        noted(
            "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes),"
            " on 2 membranes" + NOTE_TWO,
            first,
            copy,
        )
    ]
    # A copy imported later into the first membrane comes before 'copy α.tif'
    # in the image list, and so in the notice.
    again = import_file(s, data, "again β.tif", membrane_id=s.project.batch.membrane_of(first).id)
    assert said(s) == [
        noted(
            "'β-actin 10 µM.tif', 'again β.tif' and 'copy α.tif' are the same file"
            " (identical bytes), on 2 membranes" + NOTE_MORE,
            first,
            again,
            copy,
        )
    ]


def test_a_marker_imported_from_the_signal_images_file_is_not_noted(tmp_path):
    # One file imported as the chemiluminescence image and as its marker, then linked.
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "merged.tif", blot()).read_bytes()
    signal = import_file(s, data, "merged α.tif")
    membrane = s.project.batch.membrane_of(signal).id
    marker = import_file(s, data, "merged α.tif", membrane_id=membrane, kind=MARKER)
    ops.set_marker_image(s, signal, marker)
    assert s.project.batch.find_image(signal).marker_image_id == marker
    assert said(s) == []


def test_other_bytes_under_the_same_name_are_not_noted(tmp_path):
    s = session_on(tmp_path)
    pixels = blot()
    name = "blot µ.tif"
    first = import_file(s, write_tiff(tmp_path / "a" / name, pixels).read_bytes(), name)
    membrane = s.project.batch.membrane_of(first).id
    one_pixel = write_tiff(tmp_path / "b" / name, nudged(pixels)).read_bytes()
    import_file(s, one_pixel, name, membrane_id=membrane)
    assert said(s) == []
    # The check compares bytes, not pixels: the same pixels saved another way
    # (compressed) are not found.
    resaved = tmp_path / "c" / name
    resaved.parent.mkdir()
    tifffile.imwrite(resaved, pixels, compression="zlib")
    copy = import_file(s, resaved.read_bytes(), name)
    np.testing.assert_array_equal(s.pixels(copy), s.pixels(first))
    assert said(s) == []


def test_boxes_on_a_second_copy_turn_the_note_into_a_warning(tmp_path):
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    first = import_file(s, data, "β-actin 10 µM.tif")
    copy = import_file(s, data, "copy α.tif", membrane_id=s.project.batch.membrane_of(first).id)
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM")])
    note = noted(
        "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes),"
        " on one membrane" + NOTE_TWO,
        first,
        copy,
    )
    warning = warned(
        "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes),"
        " on one membrane, and both are measured" + WARN_TWO,
        first,
        copy,
    )
    # A protein on each copy, boxed on one only: still a note.
    target = ops.add_protein(s, "β-actin", "target", first)
    loading = ops.add_protein(s, "GAPDH α", "loading control", copy)
    ops.place_box(s, target, NARROW_X, ROW, lane_index=0, grow=False)
    assert said(s) == [note]
    # A box on the second copy too: a warning.
    ops.place_box(s, loading, WIDE_X, ROW, lane_index=1, grow=False)
    assert said(s) == [warning]
    # Clearing its boxes makes it a note again; undo and redo follow.
    ops.clear_boxes(s, loading)
    assert said(s) == [note]
    ops.undo(s)
    assert said(s) == [warning]
    ops.redo(s)
    assert said(s) == [note]
    ops.undo(s)
    # Removing the protein, or the copy, takes it away.
    ops.remove_protein(s, loading)
    assert said(s) == [note]
    ops.undo(s)
    ops.remove_image(s, copy)
    assert said(s) == []
    ops.undo(s)
    assert said(s) == [warning]


def test_removing_a_copy_undo_and_redo_update_the_note(tmp_path):
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    first = import_file(s, data, "β-actin 10 µM.tif")
    membrane = s.project.batch.membrane_of(first).id
    second = import_file(s, data, "copy α.tif", membrane_id=membrane)
    third = import_file(s, data, "copy β.tif")  # alone on a second membrane
    three = noted(
        "'β-actin 10 µM.tif', 'copy α.tif' and 'copy β.tif' are the same file"
        " (identical bytes), on 2 membranes" + NOTE_MORE,
        first,
        second,
        third,
    )
    assert said(s) == [three]
    # Removing the copy alone on its membrane leaves two, on one membrane.
    ops.remove_image(s, third)
    pair = noted(
        "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes),"
        " on one membrane" + NOTE_TWO,
        first,
        second,
    )
    assert said(s) == [pair]
    ops.undo(s)
    assert said(s) == [three]
    ops.redo(s)
    assert said(s) == [pair]
    # Removing either of the last two leaves one image of those bytes: nothing.
    ops.remove_image(s, first)
    assert said(s) == []
    assert [image.id for image in s.project.batch.iter_images()] == [second]
    ops.undo(s)
    assert said(s) == [pair]
    # Undoing the import of a copy takes the note with it; redoing it brings it back.
    ops.undo(s)  # the removal of the third image
    ops.undo(s)  # its import
    assert [image.id for image in s.project.batch.iter_images()] == [first, second]
    ops.undo(s)  # the import of the second
    assert said(s) == []
    ops.redo(s)
    assert said(s) == [pair]


def test_many_copies_give_one_short_note(tmp_path):
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    ids = [import_file(s, data, f"copy {n} µ.tif") for n in range(1, 13)]
    assert said(s) == [
        noted(
            "'copy 1 µ.tif' and 11 more copies are the same file (identical bytes),"
            " on 12 membranes" + NOTE_MORE,
            *ids,
        )
    ]


# --- A project saved before ---


def saved_project(folder: Path, doc: dict) -> None:
    """Write ``doc`` as the project file of ``folder``, with each image's stand-in
    file holding the bytes its SHA-256 names."""
    images = folder / storage.IMAGES_DIR
    images.mkdir(parents=True)
    for membrane in doc["batch"]["membranes"]:
        for image in membrane["images"]:
            (images / image["file"]).write_bytes(STAND_INS[image["sha256"]])
    (folder / storage.PROJECT_FILE).write_bytes(storage.document_bytes(doc))


@pytest.mark.parametrize("schema", [1, 2])
def test_a_project_saved_before_is_warned_of_when_opened_and_kept_as_it_was(tmp_path, schema):
    # A schema-1 project (saved before #83) and one of the current schema, as
    # main saved them: img-4's file holds img-2's bytes.
    if schema == 1:
        doc = sample_doc_v1()
    else:
        doc = json.loads(storage.project_to_json(make_project()))
    mem_1 = doc["batch"]["membranes"][0]["images"]
    assert [image["id"] for image in mem_1] == ["img-2", "img-3", "img-4"]
    mem_1[2]["sha256"] = mem_1[0]["sha256"]
    folder = tmp_path / FOLDER
    saved_project(folder, doc)
    before = (folder / storage.PROJECT_FILE).read_bytes()
    s = ops.open_project(folder, autosave=None, clock=FakeClock())
    message = (
        "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes),"
        " on one membrane, and both are measured" + WARN_TWO
    )
    assert said(s) == [warned(message, "img-2", "img-4")]
    # Opening wrote nothing; saving writes the project's own content, with no
    # field for the warning: a current project saves to the bytes it was read
    # from, its content hash as it was.
    assert (folder / storage.PROJECT_FILE).read_bytes() == before
    ops.save(s)
    saved = (folder / storage.PROJECT_FILE).read_bytes()
    assert saved == storage.project_to_json(s.project)
    assert b"same file" not in saved and NOTE.encode() not in saved
    if schema == 2:
        assert saved == before
        assert storage.content_hash(s.project) == storage.content_hash(Project.model_validate(doc))


# --- Exports ---


def test_an_export_lists_the_warning_once_under_checks_and_in_its_record(tmp_path):
    folder = tmp_path / FOLDER
    project = changed(make_project(), copies={"img-4": "img-2"})
    saved_project(folder, json.loads(storage.project_to_json(project)))
    s = ops.open_project(folder, autosave=None, clock=FakeClock())
    message = (
        "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes),"
        " on one membrane, and both are measured" + WARN_TWO
    )
    bundle = ops.export_bundle(s, formats=["svg"])
    lines = (bundle.folder / BUNDLE_README_FILE).read_text(encoding="utf-8").splitlines()
    section = "\n".join(lines[lines.index("Checks") + 2 : lines.index("Files")])
    _, *checks = [" ".join(p.split()) for p in section.split("\n\n") if p.strip()]
    # As the page's Checks list says it: a sentence, the first warning, once
    # (the set the include flags apply to has it; the all-lanes set does not).
    assert checks[0] == f"{message}."
    assert [line for line in checks if "same file" in line] == [f"{message}."]
    record = json.loads((bundle.folder / BUNDLE_RECORD_FILE).read_bytes())
    assert [n for n in record["results"]["notices"] if n["code"] in (NOTE, WARNING)] == [
        {
            "result_set": "Excluding lane 4",
            "code": WARNING,
            "level": "warning",
            "message": message,
            "protein_ids": [],
            "lane_indices": [],
            "conditions": [],
            "image_ids": ["img-2", "img-4"],
        }
    ]
    # The lane tables carry no check, as for every other one.
    tables = [name for name in bundle.files if name.endswith(".csv")]
    assert len(tables) == 2
    for name in tables:
        assert "same file" not in (bundle.folder / name).read_text(encoding="utf-8-sig")
