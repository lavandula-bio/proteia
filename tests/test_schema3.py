# SPDX-License-Identifier: Apache-2.0
"""Schema 3 (#223): where each membrane lies on an image, lane tables with ids,
and a protein's boxes on one membrane area as its row; and the conversion of
every older project, which changes no stored value.

A membrane holds one area per image it is on (an outline, or none for the
whole image, and an angle that is recorded only); the images themselves are
the batch's, kept once. Every membrane names its lane table, and every protein
holds its boxes in one row on one area. Until the work that draws areas, adds
lane tables and puts a protein on several membranes, a project holds one area
per image, one lane table and one row per protein; a file that holds more is
refused with a message that names it.
"""

import copy
import json
import math
import re

import pytest

from conftest import (
    V1_CONTENT_HASH,
    FakeClock,
    clashing_names_doc,
    make_project,
    sample_doc,
    sample_doc_v1,
    undetected_doc,
    write_image_files,
)
from proteia.core import operations as ops
from proteia.core import record, storage
from proteia.core.model import (
    MAX_ARENA_ANGLE,
    SCHEMA_VERSION,
    Arena,
    ArenaSource,
    ImageKind,
    Outline,
    Project,
    Role,
)
from proteia.core.results import compute_results
from proteia.core.storage import (
    MIGRATIONS,
    ProjectFormatError,
    SchemaVersionError,
    content_document,
    document_hash,
    project_from_json,
    project_to_json,
)
from test_operations import blot, import_blot, session_on

# The content hash each project had at schema 2 (built and hashed by the
# schema-2 build): what mapping its converted content back must give exactly.
V2_CONTENT_HASHES = {
    "sample": "d667c01c0856fa2c7f8b046729995b0c50cf9f11c2481b11f817599eb206b6d6",
    "undetected": "dcef6f3deed6968ef3251c79f6ad1350a7771b25ac8d633dbab822fd37824529",
    "clashing": "8abb21f7fb04638b5fe204e75d4e40dea94a9c8795370c3c3885186f83f134a5",
    "legacy": "0dce6684d7272024bd15f761df64baa56909d967669bf545874088a36cd8a9b2",
    "padded": "67d1870d1a5d6d3a35cf4d44425cfa8b1df9d939414c56a5b884529de795636b",
    "no_lanes": "c22ace588bbe4b78f5fd40e11a22e10ea71c898c206602b0008e4c2b8f35042c",
    "membrane_without_images": "4301dbaeab90c3f10e2bc035c4ee4fe54454885f5a9e5a13cc6c57e1b4d61429",
    # The same content: a schema-2 membrane's images default to none.
    "no_images_key": "4301dbaeab90c3f10e2bc035c4ee4fe54454885f5a9e5a13cc6c57e1b4d61429",
    "empty": "f6ab14dda115562f0f767894caa17ffd8ab5558fa2ba16be7d9650b4727fdb36",
}
NEWER = "which this Proteia build cannot read; open the project with a newer build"
ID = re.compile(r"\b(?:mem|img|prot|band|arena|row|lt)-[0-9]+\b")
WHOLE = {"angle_deg": 0.0, "source": "whole_image"}


def _encode(doc: dict) -> bytes:
    return json.dumps(doc, ensure_ascii=False).encode("utf-8")


def _v3(doc: dict | None = None) -> dict:
    """A schema-3 document: ``doc`` (the sample by default) converted."""
    return storage.migrate(sample_doc() if doc is None else doc)


def _refusal(doc: dict) -> str:
    """Why ``doc`` (a schema-3 document) does not open."""
    with pytest.raises(ProjectFormatError) as info:
        project_from_json(_encode(doc))
    return str(info.value)


def _arena(doc: dict, arena_id: str) -> dict:
    for membrane in doc["batch"]["membranes"]:
        for arena in membrane["arenas"]:
            if arena["id"] == arena_id:
                return arena
    raise KeyError(arena_id)


def _protein(doc: dict, protein_id: str) -> dict:
    return next(p for p in doc["batch"]["proteins"] if p["id"] == protein_id)


# --- The older schemas, as their files hold them ---


def _padded_doc() -> dict:
    doc = sample_doc()
    doc["batch"]["proteins"][1]["box_padding"] = {"across": 2, "along": 1}
    return doc


def _legacy_doc() -> dict:
    """The sample quantified before #83: each band at its image's median."""
    doc = sample_doc()
    doc["background_method"] = "global_median"
    medians = {i["id"]: i["background"] for m in doc["batch"]["membranes"] for i in m["images"]}
    for protein in doc["batch"]["proteins"]:
        for band in protein["bands"]:
            band.update(
                background_level=medians[protein["image_id"]],
                background_mode="global_median",
                background_spread=0.0,
            )
    return doc


def _no_lanes_doc() -> dict:
    doc = sample_doc()
    doc["batch"].update(lanes=[], reference_condition=None)
    for protein in doc["batch"]["proteins"]:
        protein["bands"] = []
    return doc


def _membrane_without_images_doc(key: bool = True) -> dict:
    doc = sample_doc()
    doc["batch"]["membranes"].append({"id": "mem-19", "images": []} if key else {"id": "mem-19"})
    doc["next_id"] = 20
    return doc


SCHEMA_2_DOCS = {
    "sample": sample_doc,
    "undetected": undetected_doc,
    "clashing": clashing_names_doc,
    "legacy": _legacy_doc,
    "padded": _padded_doc,
    "no_lanes": _no_lanes_doc,
    "membrane_without_images": _membrane_without_images_doc,
    "no_images_key": lambda: _membrane_without_images_doc(key=False),
    "empty": lambda: {"schema_version": 2},
}


# --- The schema ---


def test_the_schema_is_3_with_a_step_from_each_older_one():
    assert SCHEMA_VERSION == 3
    assert set(MIGRATIONS) == {1, 2}
    assert Project().schema_version == 3
    assert set(storage._EARLIER_CONTENT) == {1, 2}


def test_an_empty_project_holds_no_lane_table_and_no_image():
    assert project_to_json(Project()) == (
        b'{\n "background_method": "ring_median_v1",\n "batch": {\n  "images": [],\n'
        b'  "lane_tables": [],\n  "membranes": [],\n  "proteins": [],\n'
        b'  "reference_condition": null\n },\n "log": [],\n "next_id": 1,\n'
        b' "schema_version": 3\n}\n'
    )


# --- The conversion ---


def test_migration_ids_follow_next_id():
    doc = sample_doc()
    assert doc["next_id"] == 19
    out = storage.migrate(doc)
    assert (out["schema_version"], out["next_id"]) == (3, 27)
    batch = out["batch"]
    assert "lanes" not in batch
    assert batch["lane_tables"] == [{"id": "lt-19", "lanes": doc["batch"]["lanes"]}]
    one, five = doc["batch"]["membranes"]
    assert batch["images"] == [*one["images"], *five["images"]]  # membrane, then image order
    held = [(m["id"], m["lane_table_id"], m["arenas"], sorted(m)) for m in batch["membranes"]]
    assert held == [
        (
            "mem-1",
            "lt-19",
            [
                {"id": "arena-20", "image_id": "img-2", **WHOLE},
                {"id": "arena-21", "image_id": "img-3", **WHOLE},
                {"id": "arena-22", "image_id": "img-4", **WHOLE},
            ],
            ["arenas", "calibration", "id", "lane_table_id"],
        ),
        (
            "mem-5",
            "lt-19",
            [{"id": "arena-23", "image_id": "img-6", **WHOLE}],
            ["arenas", "calibration", "id", "lane_table_id"],
        ),
    ]
    old = {p["id"]: p for p in doc["batch"]["proteins"]}
    for protein, arena in zip(batch["proteins"], ("arena-20", "arena-23", "arena-22"), strict=True):
        v2 = old[protein["id"]]
        [row] = protein["rows"]
        moved = {key: v2[key] for key in ("box_size", "bands")}
        assert row == {"id": row["id"], "arena_id": arena, **moved}
        kept = {k: v for k, v in v2.items() if k not in ("image_id", "box_size", "bands")}
        assert {k: v for k, v in protein.items() if k != "rows"} == kept
    assert [p["rows"][0]["id"] for p in batch["proteins"]] == ["row-24", "row-25", "row-26"]
    # The input is left as it was.
    assert doc == sample_doc()


def test_the_converted_sample_opens_with_its_ids():
    project = make_project()
    assert [i for i in project.iter_ids() if not i.startswith("band-")] == [
        "mem-1",
        "mem-5",
        "img-2",
        "img-3",
        "img-4",
        "img-6",
        "lt-19",
        "arena-20",
        "arena-21",
        "arena-22",
        "arena-23",
        "prot-7",
        "prot-8",
        "prot-9",
        "row-24",
        "row-25",
        "row-26",
    ]
    assert project.next_id == 27


def test_migration_writes_no_outline():
    doc = _v3()
    arenas = [a for m in doc["batch"]["membranes"] for a in m["arenas"]]
    assert len(arenas) == 4 and all("outline" not in arena for arena in arenas)
    content = content_document(make_project())
    assert all("outline" not in a for m in content["batch"]["membranes"] for a in m["arenas"])


def test_migration_keeps_left_out_keys():
    rows = [p["rows"][0] for p in _v3()["batch"]["proteins"]]
    assert all("box_padding" not in row and "undetected" not in row for row in rows)
    content = content_document(make_project())
    assert all(
        "box_padding" not in row and "undetected" not in row
        for p in content["batch"]["proteins"]
        for row in p["rows"]
    )
    # Present where they were: moved as they are.
    alpha = _v3(_padded_doc())["batch"]["proteins"][1]["rows"][0]
    assert alpha["box_padding"] == {"across": 2, "along": 1}
    gapdh = _v3(undetected_doc())["batch"]["proteins"][2]["rows"][0]
    assert gapdh["undetected"] == undetected_doc()["batch"]["proteins"][2]["undetected"]


def test_reference_stays_on_batch():
    batch = _v3()["batch"]
    assert batch["reference_condition"] == "vehicle"
    assert sorted(batch["lane_tables"][0]) == ["id", "lanes"]
    assert make_project().batch.reference_condition == "vehicle"


@pytest.mark.parametrize("name", SCHEMA_2_DOCS)
def test_the_conversion_maps_back_to_the_schema_2_content(name):
    doc = SCHEMA_2_DOCS[name]()
    project = project_from_json(_encode(doc), clock=FakeClock())
    assert project.schema_version == 3
    earlier = storage._EARLIER_CONTENT[2](content_document(project))
    assert earlier["schema_version"] == 2
    assert document_hash(earlier) == V2_CONTENT_HASHES[name]
    [entry] = project.log
    assert entry.params == {
        "from_schema": 2,
        "to_schema": 3,
        "from_content_hash": V2_CONTENT_HASHES[name],
    }
    assert entry.content_hash == storage.content_hash(project)
    assert record.history_issues(project) == ["history_starts_late"]


def test_v1_chain_reaches_v1_content_hash():
    doc = {
        **sample_doc_v1(),
        "log": [
            {
                "seq": 1,
                "time": "2026-09-20T08:00:00.000Z",
                "action": "new_project",
                "version": "0.1.0.dev0",
                "params": {},
                "content_hash": V1_CONTENT_HASH,
            }
        ],
    }
    project = project_from_json(_encode(doc), clock=FakeClock())
    assert (project.schema_version, project.background_method) == (3, "global_median")
    content = content_document(project)
    assert document_hash(storage._EARLIER_CONTENT[1](content)) == V1_CONTENT_HASH
    created, migrated = project.log
    assert migrated.params == {
        "from_schema": 1,
        "to_schema": 3,
        "from_content_hash": V1_CONTENT_HASH,
    }
    assert record.history_issues(project) == []


def test_a_file_edited_before_its_conversion_is_reported():
    doc = sample_doc()
    doc["log"] = [
        {
            "seq": 1,
            "time": "2026-10-01T08:00:00.000Z",
            "action": "new_project",
            "version": "0.1.0.dev0",
            "params": {},
            "content_hash": V2_CONTENT_HASHES["sample"],
        }
    ]
    assert record.history_issues(project_from_json(_encode(doc))) == []
    doc["batch"]["proteins"][0]["bands"][0]["net"] += 1.0  # by hand, outside the log
    project = project_from_json(_encode(doc))
    assert project.log[-1].params["from_content_hash"] != V2_CONTENT_HASHES["sample"]
    assert record.history_issues(project) == ["content_changed_outside_log"]


def test_load_save_is_a_fixed_point_at_schema_3(tmp_path):
    for doc in (sample_doc(), undetected_doc(), _padded_doc()):
        data = project_to_json(project_from_json(_encode(doc), clock=FakeClock()))
        assert b'"schema_version": 3' in data
        assert project_to_json(project_from_json(data)) == data
        assert project_from_json(data).log == project_from_json(data).log


def test_newer_schema_refused():
    with pytest.raises(SchemaVersionError, match=r"this version reads schema <= 3") as info:
        project_from_json(_encode({**_v3(), "schema_version": 4}))
    assert (info.value.found, info.value.supported) == (4, 3)


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda d: d["batch"].update(membranes="x"), id="membranes-not-a-list"),
        pytest.param(
            lambda d: d["batch"]["membranes"][0].update(images="x"), id="images-not-a-list"
        ),
        pytest.param(lambda d: d["batch"]["membranes"][0]["images"][0].pop("id"), id="image-no-id"),
        pytest.param(
            lambda d: d["batch"]["membranes"][0]["images"][0].update(id=["img-2"]),
            id="image-id-list",
        ),
        pytest.param(lambda d: d["batch"]["membranes"][0]["images"].append(1), id="image-number"),
        pytest.param(lambda d: d["batch"]["membranes"].append(1), id="membrane-number"),
        pytest.param(lambda d: d["batch"]["proteins"][0].update(image_id="img-99"), id="unknown"),
        pytest.param(lambda d: d["batch"]["proteins"][0].update(image_id=["img-2"]), id="list-id"),
        pytest.param(lambda d: d["batch"]["proteins"][0].update(bands="x"), id="bands-not-a-list"),
        pytest.param(lambda d: d["batch"].update(proteins=[1]), id="protein-not-an-object"),
        pytest.param(lambda d: d.update(batch=[]), id="batch-not-an-object"),
    ],
)
def test_a_malformed_schema_2_file_is_a_format_error(edit):
    doc = sample_doc()
    edit(doc)
    with pytest.raises(ProjectFormatError):
        project_from_json(_encode(doc))


# --- Degenerate projects ---


def test_degenerate_no_lanes():
    project = project_from_json(_encode(_no_lanes_doc()))
    assert project.batch.only_lanes() == []
    assert [n.code for n in compute_results(project.batch).notices][-1:] == ["no_lanes"]


def test_degenerate_membrane_without_area():
    project = project_from_json(_encode(_membrane_without_images_doc()))
    membrane = project.batch.membranes[-1]
    assert (membrane.id, membrane.arenas, project.batch.images_of(membrane)) == ("mem-19", [], [])
    assert project.batch.register_groups(membrane) == []


def test_degenerate_empty_project():
    project = project_from_json(_encode({"schema_version": 2}))
    assert [table.id for table in project.batch.lane_tables] == ["lt-1"]  # always made
    assert project.next_id == 2
    assert compute_results(project.batch).notices[-1].code == "no_lanes"


# --- What a schema-3 file may hold (#223: one area, one table, one row) ---


@pytest.mark.parametrize("case", ["two_tables", "two_areas_on_image", "two_rows"])
def test_pr2_restrictions(case):
    doc = _v3()
    doc["next_id"] = 28
    batch = doc["batch"]
    if case == "two_tables":
        batch["lane_tables"].append({"id": "lt-27", "lanes": []})
        expected = f"the project holds 2 lane tables, {NEWER}"
    elif case == "two_areas_on_image":
        batch["membranes"][0]["arenas"].append(
            {"id": "arena-27", "image_id": "img-6", "source": "manual"}
        )
        expected = f"'α-tubulin µ.jpg' holds 2 membrane areas, {NEWER}"
    else:
        row = copy.deepcopy(batch["proteins"][0]["rows"][0])
        row.update(id="row-27", arena_id="arena-22", bands=[])
        batch["proteins"][0]["rows"].append(row)
        expected = f"'β-catenin' has rows on 2 membrane areas, {NEWER}"
    assert expected in _refusal(doc)
    assert not ID.search(expected)


def test_ids_stay_unique_across_the_new_kinds():
    # Two images of one id (each with its area) hold two areas on "one" image;
    # an area or a row whose number another object holds is refused too.
    doc = _v3()
    image = copy.deepcopy(doc["batch"]["images"][2])  # img-4, as a second img-2
    image.update(id="img-2", file="img-2.tif")
    doc["batch"]["images"][2] = image
    _arena(doc, "arena-22")["image_id"] = "img-2"
    assert "membrane areas" in _refusal(doc)
    for kind, edit in (
        (
            "area",  # img-4's area, and GAPDH's row on it, renumbered onto prot-7's number
            lambda d: (
                _arena(d, "arena-22").update(id="arena-7"),
                _protein(d, "prot-9")["rows"][0].update(arena_id="arena-7"),
            ),
        ),
        ("row", lambda d: _protein(d, "prot-9")["rows"][0].update(id="row-24")),
        ("table", lambda d: d["batch"]["lane_tables"][0].update(id="lt-12")),
    ):
        doc = _v3()
        edit(doc)
        if kind == "table":
            for membrane in doc["batch"]["membranes"]:
                membrane["lane_table_id"] = "lt-12"
        assert "duplicate id number" in _refusal(doc), kind


def test_a_protein_needs_a_row():
    doc = _v3()
    doc["batch"]["proteins"][0]["rows"] = []
    assert "rows" in _refusal(doc)


def test_image_without_area_is_invalid():
    doc = _v3()
    doc["batch"]["membranes"][0]["arenas"].pop(2)  # img-4's
    doc["batch"]["proteins"].pop(2)  # and GAPDH on it
    assert "image img-4 is on no membrane: no membrane area is on it" in _refusal(doc)


def test_unknown_references_are_invalid():
    doc = _v3()
    doc["batch"]["membranes"][1]["lane_table_id"] = "lt-99"
    assert "membrane mem-5: lane table 'lt-99' is not one of the project's" in _refusal(doc)
    doc = _v3()
    _arena(doc, "arena-23")["image_id"] = "img-99"
    assert (
        "membrane mem-5: area arena-23 is on 'img-99', which is not an image of the project"
        in _refusal(doc)
    )
    doc = _v3()
    _protein(doc, "prot-9")["rows"][0]["arena_id"] = "arena-99"
    assert "protein prot-9: unknown membrane area 'arena-99'" in _refusal(doc)


def test_a_row_on_a_visible_marker_is_invalid():
    doc = _v3()
    _protein(doc, "prot-9")["rows"][0]["arena_id"] = "arena-21"  # img-3, the marker
    assert "protein prot-9: img-3 is a visible-light marker image, not a signal image" in (
        _refusal(doc)
    )


def test_box_centre_outside_area_is_invalid():
    # GAPDH's boxes on img-4 (340x150) are 24x14 at x = 18 and 58: centres 30 and 70.
    doc = _v3()
    _arena(doc, "arena-22").update(outline={"x0": 0, "y0": 0, "x1": 70, "y1": 150})
    project_from_json(_encode(doc))  # the centre on the edge belongs (closed)
    _arena(doc, "arena-22")["outline"]["x1"] = 69
    assert (
        "protein prot-9: the centre of band-18's box lies outside its membrane area arena-22"
        in (_refusal(doc))
    )


def test_a_not_detected_region_outside_its_area_is_invalid():
    # GAPDH's records on img-4: lane 3 at x 98-128 (centre 113), lane 4 at 138-168 (153).
    doc = _v3(undetected_doc())
    _arena(doc, "arena-22").update(outline={"x0": 0, "y0": 0, "x1": 153, "y1": 150})
    project_from_json(_encode(doc))
    _arena(doc, "arena-22")["outline"]["x1"] = 152
    assert (
        "protein prot-9: the not-detected region in lane 4 lies outside its membrane area"
        " arena-22" in _refusal(doc)
    )


def test_box_and_record_centres_are_read_down_the_image_too():
    # GAPDH's boxes (y 100-114) and records (y 93-121) on img-4 are centred at
    # y 107; an area from y 100 holds them, but not a centre read higher up.
    doc = _v3(undetected_doc())
    _arena(doc, "arena-22").update(outline={"x0": 0, "y0": 100, "x1": 153, "y1": 150})
    project_from_json(_encode(doc))
    _arena(doc, "arena-22")["outline"]["y0"] = 108
    assert "protein prot-9: the centre of band-17's box lies outside" in _refusal(doc)


def test_a_calibration_point_outside_its_area_is_invalid():
    # mem-5's strip edges on img-6 (200x100) at y 0 and 100, with no x: read at
    # the outline's centre x. α-tubulin's boxes there are centred at y 45.
    doc = _v3()
    _arena(doc, "arena-23").update(outline={"x0": 0, "y0": 0, "x1": 200, "y1": 100})
    project_from_json(_encode(doc))
    _arena(doc, "arena-23")["outline"]["y0"] = 1
    assert (
        "membrane mem-5: calibration point at y=0.0 on img-6 lies outside its membrane area"
        " arena-23" in _refusal(doc)
    )


def test_a_mark_across_the_width_is_read_on_the_outline_middle():
    # mem-5's strip edges (y 0 and 100, no x) on img-6 under an outline over its
    # right half, x 100-200, on its top and bottom edges: read at x 150, inside;
    # read at the outline's centre y (50), or the image's middle (100), they
    # would lie left of it or on its edge. (α-tubulin, boxed on the left, goes.)
    doc = _v3()
    doc["batch"]["proteins"].pop(1)
    for protein in doc["batch"]["proteins"]:
        protein["loading_control_ids"] = []
    _arena(doc, "arena-23").update(outline={"x0": 101, "y0": 0, "x1": 199, "y1": 100})
    project = project_from_json(_encode(doc))
    [_, arena] = project.batch.find_arena("arena-23")
    assert arena.outline.centre == (150.0, 50.0)
    assert not arena.contains(project.batch.find_image("img-6"), 50.0, 0.0)


def test_a_calibration_point_with_x_is_read_where_it_was_marked():
    # A point at x 5 inside an outline whose centre x is far from it.
    doc = _v3()
    point = doc["batch"]["membranes"][1]["calibration"]["points"][0]
    point["x"] = 5.0
    _arena(doc, "arena-23").update(outline={"x0": 0, "y0": 0, "x1": 200, "y1": 100})
    project_from_json(_encode(doc))
    _arena(doc, "arena-23")["outline"]["x0"] = 6
    assert "membrane mem-5: calibration point at y=" in _refusal(doc)


def test_marker_areas_must_be_equal():
    # img-2 is linked to its marker img-3: their areas (arena-20, arena-21).
    message = (
        "image img-2 and its marker image img-3 hold different membrane areas; linked images"
        " hold the same areas"
    )
    doc = _v3()
    _arena(doc, "arena-20")["angle_deg"] = 1.0
    assert message in _refusal(doc)
    _arena(doc, "arena-21")["angle_deg"] = 1.0
    project_from_json(_encode(doc))
    outline = {"x0": 0, "y0": 0, "x1": 340, "y1": 150}
    _arena(doc, "arena-21")["outline"] = outline
    assert message in _refusal(doc)
    _arena(doc, "arena-20")["outline"] = dict(outline)
    project_from_json(_encode(doc))
    _arena(doc, "arena-20")["outline"]["y1"] = 149
    assert message in _refusal(doc)


def test_a_marker_of_another_membrane_is_invalid():
    doc = _v3()
    _arena(doc, "arena-21")["image_id"] = "img-6"  # mem-1's area moves to img-6...
    _arena(doc, "arena-23")["image_id"] = "img-3"  # ...and mem-5's to the marker img-3
    assert (
        "image img-2: marker image 'img-3' is not a visible-light marker or merged image of"
        " membrane mem-1" in _refusal(doc)
    )


def test_outline_bounds():
    Outline(x0=-(10**6), y0=-(10**6), x1=10**6, y1=10**6)
    for bad in ({"x1": 10**6 + 1}, {"x0": -(10**6) - 1}):
        with pytest.raises(ValueError):
            Outline(**{"x0": 0, "y0": 0, "x1": 5, "y1": 5, **bad})
    for x1, y1 in ((0, 5), (5, 0), (-1, 5)):
        with pytest.raises(ValueError, match="needs a width and a height of at least 1 pixel"):
            Outline(x0=0, y0=0, x1=x1, y1=y1)


def test_an_outline_centre_outside_the_image_is_invalid():
    doc = _v3()
    _arena(doc, "arena-23").update(outline={"x0": 150, "y0": 0, "x1": 251, "y1": 100})
    assert (
        "membrane mem-5: the centre (200.5, 50) of area arena-23 lies outside the 200x100 image"
        " img-6" in _refusal(doc)
    )


@pytest.mark.parametrize(
    ("outline", "centre"),
    [
        pytest.param((-60, 0, 59, 100), "(-0.5, 50)", id="left"),
        pytest.param((0, -60, 200, 59), "(100, -0.5)", id="top"),
        pytest.param((0, 41, 200, 160), "(100, 100.5)", id="bottom"),
        pytest.param((150, 0, 251, 100), "(200.5, 50)", id="right"),
    ],
)
def test_an_outline_centre_on_each_side_of_the_image(outline, centre):
    # img-6 is 200x100; the centre may lie on the edge (closed), not past it.
    corners = dict(zip(("x0", "y0", "x1", "y1"), outline, strict=True))
    doc = _v3()
    _arena(doc, "arena-23")["outline"] = corners
    assert f"the centre {centre} of area arena-23 lies outside the 200x100 image" in (_refusal(doc))


def test_an_outline_centre_on_the_edge_belongs():
    # Centres (0, 50), (200, 50), (100, 0) and (100, 100), reaching past the image.
    for outline in ((-100, 0, 100, 100), (100, 0, 300, 100), (0, -50, 200, 50), (0, 50, 200, 150)):
        doc = _v3()
        # Over a part of img-6 the boxes and the strip edges (x 100) may not reach:
        # no box, record or point there, so only the centre is checked.
        doc["batch"]["proteins"].pop(1)  # α-tubulin
        for protein in doc["batch"]["proteins"]:
            protein["loading_control_ids"] = []
        doc["batch"]["membranes"][1]["calibration"]["points"] = []
        _arena(doc, "arena-23")["outline"] = dict(
            zip(("x0", "y0", "x1", "y1"), outline, strict=True)
        )
        project_from_json(_encode(doc))


def test_angle_bounds_and_zero():
    assert MAX_ARENA_ANGLE == 45.0
    for angle in (-45.0, 45.0, 3.7):
        assert Arena(
            id="arena-1", image_id="img-2", angle_deg=angle, source="manual"
        ).angle_deg == (angle)
    for angle in (-45.1, 45.1):
        with pytest.raises(ValueError):
            Arena(id="arena-1", image_id="img-2", angle_deg=angle, source="manual")
    arena = Arena(id="arena-1", image_id="img-2", angle_deg=-0.0, source=ArenaSource.WHOLE_IMAGE)
    assert math.copysign(1.0, arena.angle_deg) == 1.0
    assert arena.model_dump(mode="json") == {
        "id": "arena-1",
        "image_id": "img-2",
        "angle_deg": 0.0,
        "source": "whole_image",
    }


# --- The geometry of an area (Arena.contains) ---


def _image(width: int = 1000, height: int = 600):
    return (
        make_project()
        .batch.find_image("img-2")
        .model_copy(update={"width": width, "height": height})
    )


def _area(outline: tuple[int, int, int, int] | None, angle: float = 0.0) -> Arena:
    corners = ("x0", "y0", "x1", "y1")
    drawn = None if outline is None else Outline(**dict(zip(corners, outline, strict=True)))
    return Arena(id="arena-1", image_id="img-2", outline=drawn, angle_deg=angle, source="manual")


def test_a_whole_image_area_holds_the_image_whatever_its_angle():
    image = _image()
    for angle in (0.0, 3.0, -45.0, 45.0):
        area = _area(None, angle)
        for x, y in ((0, 0), (1000, 600), (12, 10), (988, 590), (500, 0)):
            assert area.contains(image, x, y), (angle, x, y)
        for x, y in ((-0.5, 0), (0, -0.5), (1000.5, 600), (500, 600.5)):
            assert not area.contains(image, x, y), (angle, x, y)


def test_a_drawn_area_is_closed_and_turned_about_its_centre():
    image = _image()
    area = _area((10, 20, 30, 40))
    for x, y in ((10, 20), (30, 40), (20, 30), (10, 40)):
        assert area.contains(image, x, y)
    for x, y in ((9.99, 30), (30.01, 30), (20, 19.99), (20, 40.01)):
        assert not area.contains(image, x, y)
    # The whole image drawn as an outline and turned 3°: its corner point
    # (12, 10) falls outside (|u| = 502.51 > 500), its centre stays inside.
    turned = _area((0, 0, 1000, 600), 3.0)
    assert not turned.contains(image, 12, 10)
    assert turned.contains(image, 500, 300)
    assert _area((0, 0, 1000, 600)).contains(image, 12, 10)


def test_a_positive_angle_runs_the_right_side_lower():
    # A strip 100 px wide and 10 high: turned 4°, its right end lies about
    # 3.5 px lower; turned -4°, as much higher.
    image = _image()
    lower, higher = (99.0, 8.4), (99.0, 1.6)
    down, up = _area((0, 0, 100, 10), 4.0), _area((0, 0, 100, 10), -4.0)
    assert down.contains(image, *lower) and not down.contains(image, *higher)
    assert up.contains(image, *higher) and not up.contains(image, *lower)
    # The left end mirrors it.
    assert down.contains(image, 1.0, 1.6) and not down.contains(image, 1.0, 8.4)
    assert up.contains(image, 1.0, 8.4) and not up.contains(image, 1.0, 1.6)


def test_a_turned_square_holds_its_turned_corner():
    image = _image()
    square = _area((0, 0, 10, 10), 45.0)
    reach = 50**0.5  # the half diagonal
    assert square.contains(image, 5, 5 + reach - 1e-6)
    assert not square.contains(image, 5, 5 + reach + 1e-6)
    assert not square.contains(image, 0.5, 0.5)  # an unturned corner falls outside
    # 6 px right of or below the centre: 4.24 px along each turned side, inside
    # (a turn that divided by the cosine would put them 8.49 px out).
    for x, y in ((11, 5), (5, 11), (-1, 5), (5, -1)):
        assert square.contains(image, x, y), (x, y)


def test_the_edge_is_kept_up_to_a_billionth_of_a_pixel():
    # Centred at 0, a point 1e-9 px past the edge lies exactly on the bound
    # (half the side plus 1e-9): still in; 2e-9 px past, out.
    image = _image()
    square = _area((-10, -10, 10, 10))
    for x, y in ((10 + 1e-9, 0), (0, 10 + 1e-9), (-10 - 1e-9, 0), (0, -10 - 1e-9)):
        assert square.contains(image, x, y), (x, y)
    for x, y in ((10 + 1.2e-9, 0), (0, 10 + 1.2e-9)):
        assert not square.contains(image, x, y), (x, y)


# --- The operations write the new ids ---


def test_import_and_add_protein_make_and_log_their_ids(tmp_path):
    s = session_on(tmp_path)
    first = import_blot(s, blot(), "A1 µ.tif")
    entry = s.project.log[-1]
    assert first == "img-1"
    assert (entry.params["arena_id"], entry.params["lane_table_id"]) == ("arena-4", "lt-3")
    assert entry.params["membrane_id"] == "mem-2"
    batch = s.project.batch
    [membrane] = batch.membranes
    [arena] = membrane.arenas
    assert (membrane.lane_table_id, arena.id, arena.image_id) == ("lt-3", "arena-4", "img-1")
    assert (arena.outline, arena.angle_deg, arena.source) == (None, 0.0, ArenaSource.WHOLE_IMAGE)
    second = import_blot(s, blot(), "B1.tif")  # a membrane of its own, on the same table
    entry = s.project.log[-1]
    assert (second, entry.params["membrane_id"], entry.params["arena_id"]) == (
        "img-5",
        "mem-6",
        "arena-7",
    )
    assert "lane_table_id" not in entry.params
    third = import_blot(s, blot(), "A2.tif", membrane_id="mem-2")
    assert (third, s.project.log[-1].params["arena_id"]) == ("img-8", "arena-9")
    assert "lane_table_id" not in s.project.log[-1].params
    # Onto mem-2: right after its last image, so images stay in membrane order.
    batch = s.project.batch
    assert [image.id for image in batch.iter_images()] == ["img-1", "img-8", "img-5"]
    assert [[a.image_id for a in m.arenas] for m in batch.membranes] == [
        ["img-1", "img-8"],
        ["img-5"],
    ]
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, third)
    entry = s.project.log[-1]
    assert (protein, entry.params["row_id"]) == ("prot-10", "row-11")
    [row] = s.project.batch.find_protein(protein).rows
    assert (row.id, row.arena_id) == ("row-11", "arena-9")
    assert s.project.batch.image_id_of(s.project.batch.find_protein(protein)) == "img-8"
    assert {t.id for t in s.project.batch.lane_tables} == {"lt-3"}
    # The export record's log carries them.
    exported = record.build_record(s.project, exported_at=s.timestamp(), files={})
    logged = [e["params"].get(k) for e in exported["log"] for k in ("arena_id", "row_id")]
    assert [v for v in logged if v] == ["arena-4", "arena-7", "arena-9", "row-11"]


def test_set_lanes_makes_the_first_lane_table_and_logs_it(tmp_path):
    s = session_on(tmp_path)
    ops.set_lanes(s, [ops.LaneInput("vehicle"), ops.LaneInput("10 µM")])
    entry = s.project.log[-1]
    assert entry.params["lane_table_id"] == "lt-1"
    assert [t.id for t in s.project.batch.lane_tables] == ["lt-1"]
    ops.set_lanes(s, [ops.LaneInput("vehicle")])
    assert "lane_table_id" not in s.project.log[-1].params
    image = import_blot(s, blot())
    assert (image, s.project.log[-1].params["arena_id"]) == ("img-2", "arena-4")
    assert "lane_table_id" not in s.project.log[-1].params
    assert s.project.batch.membranes[0].lane_table_id == "lt-1"
    assert [lane.label for lane in s.project.batch.only_lanes()] == ["vehicle"]


def test_undo_and_redo_list_the_new_ids_in_order(tmp_path):
    s = session_on(tmp_path)
    image = import_blot(s, blot())  # img-1, mem-2, lt-3, arena-4
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)  # prot-5, row-6
    ops.remove_image(s, image)
    restored = ops.undo(s)
    assert restored.restored == ("mem-2", "img-1", "arena-4", "prot-5", "row-6")
    assert restored.removed == ()
    assert s.project.log[-1].params["restored"] == list(restored.restored)
    again = ops.redo(s)
    assert again.removed == ("mem-2", "img-1", "arena-4", "prot-5", "row-6")
    assert protein == "prot-5"
    assert [t.id for t in s.project.batch.lane_tables] == ["lt-3"]  # the table stays


def test_undo_after_migration_restores_migrated_state(tmp_path):
    folder = tmp_path / "v2 µ"
    project = make_project()
    write_image_files(folder, project)
    doc = sample_doc()
    doc["log"] = [
        {
            "seq": 1,
            "time": "2026-10-01T08:00:00.000Z",
            "action": "new_project",
            "version": "0.1.0.dev0",
            "params": {},
            "content_hash": V2_CONTENT_HASHES["sample"],
        }
    ]
    (folder / storage.PROJECT_FILE).write_bytes(storage.document_bytes(doc))
    s = ops.open_project(folder, autosave=None, clock=FakeClock())
    try:
        assert (folder / "project.schema2.json").read_bytes() == storage.document_bytes(doc)
        migrated = s.project
        entry = migrated.log[-1]
        assert (entry.action, entry.params["backup"]) == ("migrate", "project.schema2.json")
        ops.set_reference_condition(s, None)
        ops.undo(s)
        assert storage.content_bytes(s.project) == storage.content_bytes(migrated)
        assert s.project.next_id == migrated.next_id == 27
        assert record.history_issues(s.project) == []
    finally:
        s.close()


def test_the_page_counts_the_same_after_an_undo(tmp_path):
    # The new kinds' ids in an undo's lists name nothing the page counts
    # (images, proteins, boxes): it reports what it reported before.
    s = session_on(tmp_path)
    image = import_blot(s, blot())
    ops.add_protein(s, "β-catenin", Role.TARGET, image)
    ops.remove_image(s, image)
    restored = ops.undo(s)
    batch = s.project.batch
    counted = {i.id for i in batch.iter_images()} | {p.id for p in batch.proteins}
    assert [i for i in restored.restored if i in counted] == ["img-1", "prot-5"]


def test_a_marker_kind_is_refused_a_row(tmp_path):
    s = session_on(tmp_path)
    marker = import_blot(s, blot(), "M.tif", kind=ImageKind.VISIBLE_MARKER)
    with pytest.raises(ops.OperationError) as info:
        ops.add_protein(s, "β-catenin", Role.TARGET, marker)
    assert info.value.code is ops.ErrorCode.MARKER_IMAGE
