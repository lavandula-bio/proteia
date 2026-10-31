# SPDX-License-Identifier: Apache-2.0
"""Images of the same file (#216).

A file may be imported more than once, under any name and into any membrane;
the project then holds images with the same SHA-256. The results say so in one
``duplicate_images`` note per group of such images, stating the fact and a
next step that removes nothing. Left out of the groups: a visible-light
marker image (no protein can be on it) and an image that is another image's
marker and holds no protein (a merged image used as a marker). The note names
the images in the order of the project's image list, each by its file name
quoted as every notice quotes names, with its place in the list where two
images of the project share a name; a group of more than three names its first
image and counts the others. It is worked out from the images the project
holds now, once per computation, never stored: removing a copy, undo and redo
update it, and a project saved before shows it when opened.

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

from conftest import FakeClock, image_bytes, make_project, sample_doc, sample_doc_v1, write_tiff
from proteia.core import operations as ops
from proteia.core import storage
from proteia.core.export import BUNDLE_README_FILE, BUNDLE_RECORD_FILE
from proteia.core.model import ImageKind, Polarity, Project, apply_change
from proteia.core.operations import LaneInput, ProjectSession
from proteia.core.results import Level, Notice, NoticeCode, compute_results
from proteia.core.session import save_to_folder
from test_operations import FOLDER, NARROW_X, ROW, WIDE_X, blot, session_on

CODE = "duplicate_images"
# The end of every note: what to do.
TAIL = (
    ". If one should be a different image, import that image's file; a copy imported"
    " on purpose needs no change"
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
            if draft.batch.image_id_of(protein) in unmeasured:
                boxes = draft.batch.boxes_of(protein)
                boxes.bands, boxes.undetected = [], []
        if edit is not None:
            edit(draft)

    return apply_change(project, change)[0]


def unlink(draft: Project) -> None:
    """img-2 without its marker image."""
    draft.batch.find_image("img-2").marker_image_id = None


def onto_img_3(draft: Project) -> None:
    """GAPDH moved from img-4 onto img-3 (the same size): its row onto img-3's area."""
    [(_, arena)] = draft.batch.arenas_on("img-3")
    draft.batch.only_row(draft.batch.find_protein("prot-9")).arena_id = arena.id


def notes(notices: list[Notice]) -> list[tuple[str, tuple[str, ...]]]:
    """The duplicate_images notes: (message, image ids)."""
    return [(n.message, n.image_ids) for n in notices if n.code == CODE]


def others(notices: list[Notice]) -> list[Notice]:
    return [n for n in notices if n.code != CODE]


def found(project: Project) -> list[tuple[str, ...]]:
    """The image ids of each note of ``project``'s results."""
    return [ids for _, ids in notes(compute_results(project.batch).notices)]


# --- From the stored images ---


def test_two_copies_get_one_note_first_in_the_set_the_include_flags_apply_to():
    res = compute_results(changed(make_project(), copies={"img-4": "img-2"}).batch)
    message = "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes)" + TAIL
    assert res.notices[0] == Notice(
        code=NoticeCode(CODE), level=Level.INFO, message=message, image_ids=("img-2", "img-4")
    )
    assert notes(res.notices) == [(message, ("img-2", "img-4"))]
    # It is about the images, not a set's lanes: the all-lanes set has none.
    assert res.all_lanes is not None and notes(res.all_lanes.notices) == []
    # It adds a note and changes no other check.
    plain = compute_results(make_project().batch)
    assert others(res.notices) == plain.notices
    assert res.all_lanes.notices == plain.all_lanes.notices


@pytest.mark.parametrize(
    "unmeasured",
    [
        pytest.param(set(), id="both-boxed"),
        pytest.param({"img-4"}, id="one-boxed"),
        pytest.param({"img-2", "img-4"}, id="neither-boxed"),
    ],
)
def test_the_note_is_the_same_whatever_is_boxed_on_the_copies(unmeasured):
    project = changed(make_project(), copies={"img-4": "img-2"}, unmeasured=unmeasured)
    assert notes(compute_results(project.batch).notices) == [
        (
            "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes)" + TAIL,
            ("img-2", "img-4"),
        )
    ]


@pytest.mark.parametrize(
    ("kind", "edit", "ids"),
    [
        # One file as the signal image and as its visible-light marker, linked.
        pytest.param(MARKER, None, [], id="visible-marker"),
        # As a merged image used as the signal image's marker, holding no protein.
        pytest.param(MERGED, None, [], id="merged-marker"),
        # A merged marker a protein is on holds bands like any signal image.
        pytest.param(MERGED, onto_img_3, [("img-2", "img-3")], id="merged-marker-with-a-protein"),
        # A merged image no image takes as its marker is a signal image.
        pytest.param(MERGED, unlink, [("img-2", "img-3")], id="merged-signal-image"),
        pytest.param(CHEMI, unlink, [("img-2", "img-3")], id="two-chemiluminescence"),
    ],
)
def test_a_marker_copy_holding_no_protein_is_left_out(kind, edit, ids):
    project = changed(make_project(), copies={"img-3": "img-2"}, kinds={"img-3": kind}, edit=edit)
    assert (project.batch.find_image("img-2").marker_image_id == "img-3") is (edit is not unlink)
    assert found(project) == ids


def test_the_marker_of_an_image_of_other_bytes_is_left_out_too():
    # img-3, a merged image, is img-2's marker; img-4 holds img-3's bytes.
    project = changed(make_project(), copies={"img-4": "img-3"}, kinds={"img-3": MERGED})
    assert found(project) == []
    # A third copy of those bytes makes a pair of the other two.
    project = changed(project, copies={"img-6": "img-3"})
    assert found(project) == [("img-4", "img-6")]


@pytest.mark.parametrize("kind", list(ImageKind))
def test_the_kinds_left_out_are_those_the_model_takes_no_protein_on(kind):
    # The note reads the model's rule without importing it: kept in step here.
    try:
        changed(make_project(), kinds={"img-4": kind})  # GAPDH's image
        takes_proteins = True
    except ValidationError:
        takes_proteins = False
    assert takes_proteins is (kind is not MARKER)
    project = changed(make_project(), copies={"img-3": "img-2"}, kinds={"img-3": kind}, edit=unlink)
    assert found(project) == ([("img-2", "img-3")] if takes_proteins else [])


@pytest.mark.parametrize(
    "copies",
    [
        pytest.param({"img-6": "img-2"}, id="the-copy-on-the-later-membrane"),
        pytest.param({"img-2": "img-6"}, id="the-copy-on-the-earlier-membrane"),
    ],
)
def test_copies_are_named_in_the_image_lists_order(copies):
    assert notes(compute_results(changed(make_project(), copies=copies).batch).notices) == [
        (
            "'β-actin 10 µM.tif' and 'α-tubulin µ.jpg' are the same file (identical bytes)" + TAIL,
            ("img-2", "img-6"),
        )
    ]


@pytest.mark.parametrize(
    ("names", "named"),
    [
        # Both copies under one name: each with its place in the image list.
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
    assert notes(compute_results(project.batch).notices) == [
        (f"{named} are the same file (identical bytes)" + TAIL, ("img-2", "img-4"))
    ]


def test_a_name_with_an_apostrophe_is_quoted_as_every_notice_quotes_names():
    # The notices quote a name as Python writes a string: in single quotes,
    # or in double quotes when it holds one; a protein's name the same way.
    project = changed(make_project(), copies={"img-4": "img-2"}, names={"img-4": "5'-UTR β.tif"})

    def rename(draft: Project) -> None:
        draft.batch.find_protein("prot-9").name = "5'-GAPDH"

    project = changed(project, edit=rename)
    res = compute_results(project.batch)
    assert notes(res.notices) == [
        (
            "'β-actin 10 µM.tif' and \"5'-UTR β.tif\" are the same file (identical bytes)" + TAIL,
            ("img-2", "img-4"),
        )
    ]
    gapdh = [n.message for n in res.notices if n.protein_ids == ("prot-9",)]
    assert gapdh[0].startswith('"5\'-GAPDH" was not checked for over-exposure')


def test_the_name_of_an_image_with_other_bytes_is_no_duplicate():
    project = changed(make_project(), names={"img-4": "β-actin 10 µM.tif"})
    # Renaming changes no check: the copies are found by their stored hashes.
    assert compute_results(project.batch).notices == compute_results(make_project().batch).notices


def test_each_group_is_listed_by_the_place_of_its_first_image():
    # With img-3 a merged image no image takes as its marker, 'marker α.png'
    # and 'reprobe β.tif' (images 2, 3) are one file, and 'β-actin 10 µM.tif'
    # and 'α-tubulin µ.jpg' (images 1, 4) another: the group whose first image
    # comes first is listed first, though its last comes last.
    project = changed(
        make_project(),
        copies={"img-6": "img-2", "img-4": "img-3"},
        kinds={"img-3": MERGED},
        edit=unlink,
    )
    res = compute_results(project.batch)
    assert notes(res.notices) == [
        (
            "'β-actin 10 µM.tif' and 'α-tubulin µ.jpg' are the same file (identical bytes)" + TAIL,
            ("img-2", "img-6"),
        ),
        (
            "'marker α.png' and 'reprobe β.tif' are the same file (identical bytes)" + TAIL,
            ("img-3", "img-4"),
        ),
    ]
    assert [n.code for n in res.notices[:2]] == [CODE, CODE]


@pytest.mark.parametrize(
    ("copies", "names", "message", "ids"),
    [
        # Three images: each named.
        pytest.param(
            {"img-4": "img-2", "img-6": "img-2"},
            {},
            "'β-actin 10 µM.tif', 'reprobe β.tif' and 'α-tubulin µ.jpg' are the same file"
            " (identical bytes)" + TAIL,
            ("img-2", "img-4", "img-6"),
            id="three",
        ),
        # Four: the first named, the others counted.
        pytest.param(
            {"img-3": "img-2", "img-4": "img-2", "img-6": "img-2"},
            {},
            "'β-actin 10 µM.tif' and 3 more copies are the same file (identical bytes)" + TAIL,
            ("img-2", "img-3", "img-4", "img-6"),
            id="four",
        ),
        pytest.param(
            {"img-3": "img-2", "img-4": "img-2", "img-6": "img-2"},
            {"img-6": "β-actin 10 µM.tif"},
            "'β-actin 10 µM.tif' (image 1) and 3 more copies are the same file"
            " (identical bytes)" + TAIL,
            ("img-2", "img-3", "img-4", "img-6"),
            id="four-the-first-name-shared",
        ),
    ],
)
def test_three_copies_are_named_and_more_are_counted(copies, names, message, ids):
    # img-3 a chemiluminescence image no image takes as its marker.
    project = changed(
        make_project(), copies=copies, names=names, kinds={"img-3": CHEMI}, edit=unlink
    )
    assert notes(compute_results(project.batch).notices) == [(message, ids)]


def test_a_project_without_lanes_or_proteins_is_noted_too():
    def bare(draft: Project) -> None:
        draft.batch.proteins, draft.batch.lane_tables[0].lanes = [], []
        draft.batch.reference_condition = None

    project = changed(make_project(), copies={"img-4": "img-2"}, edit=bare)
    res = compute_results(project.batch)
    assert [n.code for n in res.notices] == [CODE, "no_target", "no_loading_control", "no_lanes"]
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


def said(s: ProjectSession) -> list[tuple[str, tuple[str, ...]]]:
    """The duplicate_images notes of the session's results, as they are now."""
    return notes(ops.compute(s).notices)


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
    message = "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes)" + TAIL
    assert said(s) == [(message, (first, copy))]
    # Nothing is stored with either image, so the project file holds no word of it.
    assert [i.import_warnings for i in s.project.batch.iter_images()] == [[], []]
    saved = (s.folder / storage.PROJECT_FILE).read_bytes()
    assert saved == storage.project_to_json(s.project)
    assert b"same file" not in saved and CODE.encode() not in saved
    # The first image's record is as it was before the copy came.
    alone = json.loads(saved_alone)["batch"]["images"]
    assert json.loads(saved)["batch"]["images"][:1] == alone


def test_the_same_bytes_in_another_membrane_are_named_in_the_image_lists_order(tmp_path):
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    first = import_file(s, data, "β-actin 10 µM.tif")
    copy = import_file(s, data, "copy α.tif")  # a new membrane
    assert said(s) == [
        (
            "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes)" + TAIL,
            (first, copy),
        )
    ]
    # A copy imported later into the first membrane comes before 'copy α.tif'
    # in the image list, and so in the note.
    again = import_file(s, data, "again β.tif", membrane_id=s.project.batch.membrane_of(first).id)
    assert said(s) == [
        (
            "'β-actin 10 µM.tif', 'again β.tif' and 'copy α.tif' are the same file"
            " (identical bytes)" + TAIL,
            (first, again, copy),
        )
    ]


@pytest.mark.parametrize("kind", [MARKER, MERGED])
def test_a_marker_imported_from_the_signal_images_file_is_left_out(tmp_path, kind):
    # One file imported as the chemiluminescence image and as its marker, linked.
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "merged.tif", blot()).read_bytes()
    signal = import_file(s, data, "merged α.tif")
    membrane = s.project.batch.membrane_of(signal).id
    marker = import_file(s, data, "merged α.tif", membrane_id=membrane, kind=kind)
    pair = [
        (
            "'merged α.tif' (image 1) and 'merged α.tif' (image 2) are the same file"
            " (identical bytes)" + TAIL,
            (signal, marker),
        )
    ]
    # Not linked yet, a merged image is a signal image like the other.
    assert said(s) == ([] if kind is MARKER else pair)
    ops.set_marker_image(s, signal, marker)
    assert s.project.batch.find_image(signal).marker_image_id == marker
    assert said(s) == []
    if kind is MERGED:
        # A protein on the merged marker: it may hold bands, so it is a copy again.
        ops.add_protein(s, "β-actin", "target", marker)
        assert said(s) == pair
        ops.undo(s)
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


def test_boxes_on_the_copies_change_nothing(tmp_path):
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    first = import_file(s, data, "β-actin 10 µM.tif")
    copy = import_file(s, data, "copy α.tif", membrane_id=s.project.batch.membrane_of(first).id)
    note = [
        (
            "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes)" + TAIL,
            (first, copy),
        )
    ]
    ops.set_lanes(s, [LaneInput("vehicle"), LaneInput("10 µM", included=False)])
    target = ops.add_protein(s, "β-actin", "target", first)
    loading = ops.add_protein(s, "GAPDH α", "loading control", copy)
    ops.place_box(s, target, NARROW_X, ROW, lane_index=0, grow=False)
    ops.place_box(s, loading, WIDE_X, ROW, lane_index=1, grow=False)
    assert said(s) == note
    ops.clear_boxes(s, loading)
    assert said(s) == note


def test_removing_a_copy_undo_and_redo_update_the_note(tmp_path):
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    first = import_file(s, data, "β-actin 10 µM.tif")
    membrane = s.project.batch.membrane_of(first).id
    second = import_file(s, data, "copy α.tif", membrane_id=membrane)
    third = import_file(s, data, "copy β.tif")  # alone on a second membrane
    three = (
        "'β-actin 10 µM.tif', 'copy α.tif' and 'copy β.tif' are the same file"
        " (identical bytes)" + TAIL,
        (first, second, third),
    )
    assert said(s) == [three]
    ops.remove_image(s, third)
    pair = (
        "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes)" + TAIL,
        (first, second),
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
        (
            "'copy 1 µ.tif' and 11 more copies are the same file (identical bytes)" + TAIL,
            tuple(ids),
        )
    ]


# --- A project saved before ---


def saved_project(folder: Path, doc: dict) -> None:
    """Write ``doc`` as the project file of ``folder``, with each image's stand-in
    file holding the bytes its SHA-256 names."""
    images = folder / storage.IMAGES_DIR
    images.mkdir(parents=True)
    held = doc["batch"].get("images")  # the batch's (schema 3), else each membrane's
    if held is None:
        held = [image for membrane in doc["batch"]["membranes"] for image in membrane["images"]]
    for image in held:
        (images / image["file"]).write_bytes(STAND_INS[image["sha256"]])
    (folder / storage.PROJECT_FILE).write_bytes(storage.document_bytes(doc))


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_a_project_saved_before_is_noted_when_opened_and_kept_as_it_was(tmp_path, schema):
    # A schema-1 project (saved before #83), a schema-2 one (before #223) and
    # one of the current schema, as main saved them: img-4's file holds img-2's
    # bytes.
    if schema == 1:
        doc = sample_doc_v1()
    elif schema == 2:
        doc = sample_doc()
    else:
        doc = json.loads(storage.project_to_json(make_project()))
    if schema == 3:
        mem_1 = doc["batch"]["images"][:3]
    else:
        mem_1 = doc["batch"]["membranes"][0]["images"]
    assert [image["id"] for image in mem_1] == ["img-2", "img-3", "img-4"]
    mem_1[2]["sha256"] = mem_1[0]["sha256"]
    folder = tmp_path / FOLDER
    saved_project(folder, doc)
    before = (folder / storage.PROJECT_FILE).read_bytes()
    s = ops.open_project(folder, autosave=None, clock=FakeClock())
    message = "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes)" + TAIL
    assert said(s) == [(message, ("img-2", "img-4"))]
    # Opening wrote nothing; saving writes the project's own content, with no
    # field for the note: a current project saves to the bytes it was read
    # from, its content hash as it was.
    assert (folder / storage.PROJECT_FILE).read_bytes() == before
    ops.save(s)
    saved = (folder / storage.PROJECT_FILE).read_bytes()
    assert saved == storage.project_to_json(s.project)
    assert b"same file" not in saved and CODE.encode() not in saved
    if schema == 3:
        assert saved == before
        assert storage.content_hash(s.project) == storage.content_hash(Project.model_validate(doc))


# --- Exports ---


def test_an_export_lists_the_note_once_under_checks_and_in_its_record(tmp_path):
    folder = tmp_path / FOLDER
    project = changed(make_project(), copies={"img-4": "img-2"})
    saved_project(folder, json.loads(storage.project_to_json(project)))
    s = ops.open_project(folder, autosave=None, clock=FakeClock())
    message = "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes)" + TAIL
    bundle = ops.export_bundle(s, formats=["svg"])
    lines = (bundle.folder / BUNDLE_README_FILE).read_text(encoding="utf-8").splitlines()
    section = "\n".join(lines[lines.index("Checks") + 2 : lines.index("Files")])
    _, *checks = [" ".join(p.split()) for p in section.split("\n\n") if p.strip()]
    # As the page's Checks list says it: a sentence, once; the all-lanes set
    # does not repeat it.
    assert [line for line in checks if "same file" in line] == [f"{message}."]
    record = json.loads((bundle.folder / BUNDLE_RECORD_FILE).read_bytes())
    # The first of the notes of the set the include flags apply to.
    infos = [n["code"] for n in record["results"]["notices"] if n["level"] == "info"]
    assert infos[0] == CODE
    assert [n for n in record["results"]["notices"] if n["code"] == CODE] == [
        {
            "result_set": "Excluding lane 4",
            "code": CODE,
            "level": "info",
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
