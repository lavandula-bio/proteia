# SPDX-License-Identifier: Apache-2.0
"""Images of the same file (#216).

A file may be imported more than once, under any name and into any membrane;
the project then holds images with the same SHA-256. The results say so: one
``duplicate_images`` warning per group of such images, naming them as the
page names images (by file name, with their place in the image list where two
share a name) and counting the membranes they are on. It is worked out from
the images the project holds now, never stored, so removing a copy, undo and
redo update it, and a project saved before it shows it when opened.

The results tests build their projects from the model (no pixels): the
conftest sample's images, in image-list order 'β-actin 10 µM.tif' (img-2),
'marker α.png' (img-3) and 'reprobe β.tif' (img-4) on one membrane and
'α-tubulin µ.jpg' (img-6) on another, given each other's hashes. The
operations tests import a synthetic 16-bit TIFF; test_web_api has the routes'.
"""

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import tifffile

from conftest import FakeClock, image_bytes, make_project, sample_doc_v1, write_tiff
from proteia.core import operations as ops
from proteia.core import storage
from proteia.core.export import BUNDLE_README_FILE, BUNDLE_RECORD_FILE
from proteia.core.model import ImageKind, Polarity, Project, apply_change
from proteia.core.operations import ProjectSession
from proteia.core.results import Level, Notice, NoticeCode, compute_results
from proteia.core.session import save_to_folder
from test_operations import FOLDER, blot, session_on

CODE = "duplicate_images"
# The end of every notice: its next step.
NEXT = (
    " If one should be a different image, import that image's file; nothing needs to"
    " change for a copy imported on purpose"
)
TWICE = ": a band boxed on both is counted twice."
AGAIN = ": a band boxed on more than one of them is counted more than once."
SAMPLE = ("img-2", "img-3", "img-4", "img-6")
# The bytes of each sample image's stand-in file, by its SHA-256.
STAND_INS = {hashlib.sha256(image_bytes(i)).hexdigest(): image_bytes(i) for i in SAMPLE}
CHEMI, DARK = ImageKind.CHEMILUMINESCENCE, Polarity.DARK_ON_LIGHT


def changed(
    project: Project,
    *,
    copies: Mapping[str, str] | None = None,
    names: Mapping[str, str] | None = None,
) -> Project:
    """``project`` with each image of ``copies`` holding the bytes of the image
    it maps to (its SHA-256) and each image of ``names`` renamed."""

    def change(draft: Project) -> None:
        hashes = {image.id: image.sha256 for image in draft.batch.iter_images()}
        for image_id, source in (copies or {}).items():
            draft.batch.find_image(image_id).sha256 = hashes[source]
        for image_id, name in (names or {}).items():
            draft.batch.find_image(image_id).original_name = name

    return apply_change(project, change)[0]


def duplicates(notices: list[Notice]) -> list[tuple[str, tuple[str, ...]]]:
    """The duplicate_images notices: (message, image ids)."""
    return [(n.message, n.image_ids) for n in notices if n.code == CODE]


def others(notices: list[Notice]) -> list[Notice]:
    return [n for n in notices if n.code != CODE]


# --- From the stored hashes ---


def test_two_images_of_one_file_on_one_membrane_get_one_warning_naming_both():
    res = compute_results(changed(make_project(), copies={"img-4": "img-2"}).batch)
    message = (
        "'β-actin 10 µM.tif' and 'reprobe β.tif' are the same file (identical bytes),"
        " on one membrane" + TWICE + NEXT
    )
    # The first of the checks, a warning about the two images and nothing else.
    assert res.notices[0] == Notice(
        code=NoticeCode(CODE), level=Level.WARNING, message=message, image_ids=("img-2", "img-4")
    )
    assert duplicates(res.notices) == [(message, ("img-2", "img-4"))]
    # It is about the project, not a set's lanes: the all-lanes set leaves it
    # to the set listed first, as it does every check the two share.
    assert duplicates(res.all_lanes.notices) == []
    # It adds a check and changes no other.
    plain = compute_results(make_project().batch)
    assert others(res.notices) == plain.notices
    assert res.all_lanes.notices == plain.all_lanes.notices


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
        (
            "'β-actin 10 µM.tif' and 'α-tubulin µ.jpg' are the same file (identical bytes),"
            " on 2 membranes" + TWICE + NEXT,
            ("img-2", "img-6"),
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
        # Names the page tells apart as they are (another case) need no place.
        pytest.param(
            {"img-6": "Reprobe β.tif"},
            "'β-actin 10 µM.tif' and 'reprobe β.tif'",
            id="another-case",
        ),
    ],
)
def test_a_name_two_images_share_is_told_apart_by_its_place_in_the_image_list(names, named):
    project = changed(make_project(), copies={"img-4": "img-2"}, names=names)
    res = compute_results(project.batch)
    message = f"{named} are the same file (identical bytes), on one membrane" + TWICE + NEXT
    assert duplicates(res.notices) == [(message, ("img-2", "img-4"))]


def test_the_name_of_an_image_with_other_bytes_is_no_duplicate():
    project = changed(make_project(), names={"img-4": "β-actin 10 µM.tif"})
    assert duplicates(compute_results(project.batch).notices) == []
    # Nor is the same pixels saved another way: the check reads the stored
    # hashes of the files, and the batch holds nothing else (see below for files).
    assert compute_results(project.batch).notices == compute_results(make_project().batch).notices


def test_each_group_gets_one_warning_in_the_order_of_its_first_image():
    # 'marker α.png' and 'reprobe β.tif' (images 2, 3) are one file, and
    # 'β-actin 10 µM.tif' and 'α-tubulin µ.jpg' (images 1, 4) another: the
    # group whose first image comes first is listed first.
    project = changed(make_project(), copies={"img-6": "img-2", "img-4": "img-3"})
    res = compute_results(project.batch)
    assert duplicates(res.notices) == [
        (
            "'β-actin 10 µM.tif' and 'α-tubulin µ.jpg' are the same file (identical bytes),"
            " on 2 membranes" + TWICE + NEXT,
            ("img-2", "img-6"),
        ),
        (
            "'marker α.png' and 'reprobe β.tif' are the same file (identical bytes),"
            " on one membrane" + TWICE + NEXT,
            ("img-3", "img-4"),
        ),
    ]
    assert [n.code for n in res.notices[:2]] == [CODE, CODE]


@pytest.mark.parametrize(
    ("copies", "names", "message", "ids"),
    [
        # Three images: each named.
        pytest.param(
            {"img-3": "img-2", "img-6": "img-2"},
            {},
            "'β-actin 10 µM.tif', 'marker α.png' and 'α-tubulin µ.jpg' are the same file"
            " (identical bytes), on 2 membranes" + AGAIN + NEXT,
            ("img-2", "img-3", "img-6"),
            id="three",
        ),
        # Four: the first named, the others by their places in the image list.
        pytest.param(
            {"img-3": "img-2", "img-4": "img-2", "img-6": "img-2"},
            {},
            "'β-actin 10 µM.tif' and 3 more (images 2, 3, 4) are the same file"
            " (identical bytes), on 2 membranes" + AGAIN + NEXT,
            ("img-2", "img-3", "img-4", "img-6"),
            id="four",
        ),
        pytest.param(
            {"img-3": "img-2", "img-4": "img-2", "img-6": "img-2"},
            {"img-6": "β-actin 10 µM.tif"},
            "'β-actin 10 µM.tif' (image 1) and 3 more (images 2, 3, 4) are the same file"
            " (identical bytes), on 2 membranes" + AGAIN + NEXT,
            ("img-2", "img-3", "img-4", "img-6"),
            id="four-the-first-name-shared",
        ),
    ],
)
def test_three_copies_are_named_and_more_are_counted(copies, names, message, ids):
    res = compute_results(changed(make_project(), copies=copies, names=names).batch)
    assert duplicates(res.notices) == [(message, ids)]


def test_a_project_without_lanes_or_proteins_is_warned_too():
    def change(draft: Project) -> None:
        draft.batch.proteins, draft.batch.lanes = [], []
        draft.batch.reference_condition = None

    project = apply_change(changed(make_project(), copies={"img-4": "img-2"}), change)[0]
    res = compute_results(project.batch)
    assert [n.code for n in res.notices] == [CODE, "no_target", "no_loading_control", "no_lanes"]


# --- Imports, removals, undo and redo ---


def import_file(
    s: ProjectSession, data: bytes, name: str, *, membrane_id: str | None = None
) -> str:
    """Import ``data`` as the file ``name``: the same bytes under any name."""
    return ops.import_image(
        s, io.BytesIO(data), name, kind=CHEMI, polarity=DARK, membrane_id=membrane_id
    )


def noted(s: ProjectSession) -> list[tuple[str, tuple[str, ...]]]:
    """The duplicate_images notices of the session's results, as they are now."""
    return duplicates(ops.compute(s).notices)


def nudged(pixels: np.ndarray) -> np.ndarray:
    """``pixels`` with one pixel a level darker: other bytes, almost the same image."""
    other = pixels.copy()
    other[0, 0] -= 1
    return other


def test_an_import_of_the_same_bytes_is_warned_of_and_nothing_is_stored(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    first = import_file(s, data, "β-actin 10 µM.tif")
    assert noted(s) == []
    saved_alone = (s.folder / storage.PROJECT_FILE).read_bytes()
    membrane = s.project.batch.membrane_of(first).id
    copy = import_file(s, data, "copy α.tif", membrane_id=membrane)
    message = (
        "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes),"
        " on one membrane" + TWICE + NEXT
    )
    assert noted(s) == [(message, (first, copy))]
    # Nothing is stored with either image, so the project file holds no word of it.
    assert [i.import_warnings for i in s.project.batch.iter_images()] == [[], []]
    saved = (s.folder / storage.PROJECT_FILE).read_bytes()
    assert saved == storage.project_to_json(s.project)
    assert b"same file" not in saved and CODE.encode() not in saved
    # The first image's record is as it was before the copy came.
    alone = json.loads(saved_alone)["batch"]["membranes"][0]["images"]
    assert json.loads(saved)["batch"]["membranes"][0]["images"][:1] == alone


def test_the_same_bytes_in_another_membrane_are_named_in_the_image_lists_order(tmp_path):
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    first = import_file(s, data, "β-actin 10 µM.tif")
    copy = import_file(s, data, "copy α.tif")  # a new membrane
    assert noted(s) == [
        (
            "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes),"
            " on 2 membranes" + TWICE + NEXT,
            (first, copy),
        )
    ]
    # A copy imported later into the first membrane comes before 'copy α.tif'
    # in the image list, and so in the notice.
    again = import_file(s, data, "again β.tif", membrane_id=s.project.batch.membrane_of(first).id)
    assert noted(s) == [
        (
            "'β-actin 10 µM.tif', 'again β.tif' and 'copy α.tif' are the same file"
            " (identical bytes), on 2 membranes" + AGAIN + NEXT,
            (first, again, copy),
        )
    ]


def test_other_bytes_under_the_same_name_are_not_warned_of(tmp_path):
    s = session_on(tmp_path)
    pixels = blot()
    name = "blot µ.tif"
    first = import_file(s, write_tiff(tmp_path / "a" / name, pixels).read_bytes(), name)
    membrane = s.project.batch.membrane_of(first).id
    one_pixel = write_tiff(tmp_path / "b" / name, nudged(pixels)).read_bytes()
    import_file(s, one_pixel, name, membrane_id=membrane)
    assert noted(s) == []
    # The check compares bytes, not pixels: the same pixels saved another way
    # (compressed) are not found.
    resaved = tmp_path / "c" / name
    resaved.parent.mkdir()
    tifffile.imwrite(resaved, pixels, compression="zlib")
    copy = import_file(s, resaved.read_bytes(), name)
    np.testing.assert_array_equal(s.pixels(copy), s.pixels(first))
    assert noted(s) == []


def test_removing_a_copy_undo_and_redo_update_the_warning(tmp_path):
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    first = import_file(s, data, "β-actin 10 µM.tif")
    membrane = s.project.batch.membrane_of(first).id
    second = import_file(s, data, "copy α.tif", membrane_id=membrane)
    third = import_file(s, data, "copy β.tif")  # alone on a second membrane
    three = (
        "'β-actin 10 µM.tif', 'copy α.tif' and 'copy β.tif' are the same file"
        " (identical bytes), on 2 membranes" + AGAIN + NEXT
    )
    assert noted(s) == [(three, (first, second, third))]
    # Removing the copy alone on its membrane leaves two, on one membrane.
    ops.remove_image(s, third)
    pair = (
        "'β-actin 10 µM.tif' and 'copy α.tif' are the same file (identical bytes),"
        " on one membrane" + TWICE + NEXT
    )
    assert noted(s) == [(pair, (first, second))]
    ops.undo(s)
    assert noted(s) == [(three, (first, second, third))]
    ops.redo(s)
    assert noted(s) == [(pair, (first, second))]
    # Removing either of the last two leaves one image of those bytes: no warning.
    ops.remove_image(s, first)
    assert noted(s) == []
    assert [image.id for image in s.project.batch.iter_images()] == [second]
    ops.undo(s)
    assert noted(s) == [(pair, (first, second))]
    # Undoing the import of a copy takes the warning with it; redoing it brings it back.
    ops.undo(s)  # the removal of the third image
    ops.undo(s)  # its import
    assert [image.id for image in s.project.batch.iter_images()] == [first, second]
    ops.undo(s)  # the import of the second
    assert noted(s) == []
    ops.redo(s)
    assert noted(s) == [(pair, (first, second))]


def test_many_copies_give_one_short_warning(tmp_path):
    s = session_on(tmp_path)
    data = write_tiff(tmp_path / "sources" / "blot.tif", blot()).read_bytes()
    ids = [import_file(s, data, f"copy {n} µ.tif") for n in range(1, 13)]
    assert noted(s) == [
        (
            "'copy 1 µ.tif' and 11 more (images 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) are the"
            " same file (identical bytes), on 12 membranes" + AGAIN + NEXT,
            tuple(ids),
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
        " on one membrane" + TWICE + NEXT
    )
    assert noted(s) == [(message, ("img-2", "img-4"))]
    # Opening wrote nothing; saving writes the project's own content, with no
    # field for the warning: a current project saves to the bytes it was read
    # from, its content hash as it was.
    assert (folder / storage.PROJECT_FILE).read_bytes() == before
    ops.save(s)
    saved = (folder / storage.PROJECT_FILE).read_bytes()
    assert saved == storage.project_to_json(s.project)
    assert b"same file" not in saved and CODE.encode() not in saved
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
        " on one membrane" + TWICE + NEXT
    )
    bundle = ops.export_bundle(s, formats=["svg"])
    lines = (bundle.folder / BUNDLE_README_FILE).read_text(encoding="utf-8").splitlines()
    section = "\n".join(lines[lines.index("Checks") + 2 : lines.index("Files")])
    _, *checks = [" ".join(p.split()) for p in section.split("\n\n") if p.strip()]
    # As the page's Checks list says it: a sentence, the first warning, once
    # (the set the include flags apply to has it; the all-lanes set shares it).
    assert checks[0] == f"{message}."
    assert [line for line in checks if "same file" in line] == [f"{message}."]
    record = json.loads((bundle.folder / BUNDLE_RECORD_FILE).read_bytes())
    assert [n for n in record["results"]["notices"] if n["code"] == CODE] == [
        {
            "result_set": "Excluding lane 4",
            "code": CODE,
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
