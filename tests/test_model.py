# SPDX-License-Identifier: Apache-2.0
"""Tests for the project data model: invariants, ids, canonical order, apply_change.

Most tests edit the sample project from ``conftest`` in place, the way a buggy
caller could, and check that re-validation accepts or rejects the result.
"""

import hashlib
import math
from typing import get_args

import pytest
from pydantic import ValidationError

from conftest import make_project
from proteia.core import analyze, model
from proteia.core.model import (
    SCHEMA_VERSION,
    Band,
    Batch,
    Box,
    BoxSize,
    CalibrationPoint,
    FitMethod,
    ImageKind,
    ImageRef,
    Lane,
    Membrane,
    Polarity,
    Project,
    Protein,
    Role,
    UnknownIdError,
    apply_change,
    revalidate,
)


def _get(project: Project, obj_id: str):
    """The membrane, image, protein or band with this id."""
    batch = project.batch
    kind = obj_id.split("-")[0]
    if kind == "mem":
        return next(m for m in batch.membranes if m.id == obj_id)
    if kind == "img":
        return batch.find_image(obj_id)
    if kind == "prot":
        return batch.find_protein(obj_id)
    return batch.find_band(obj_id)[1]


def _set(obj, **fields) -> None:
    for name, value in fields.items():
        setattr(obj, name, value)


def _edit(project: Project, obj_id: str, **fields) -> Project:
    """Set fields of one object in place (no validation) and return the project."""
    _set(_get(project, obj_id), **fields)
    return project


def _point(project: Project, membrane_id: str = "mem-1") -> CalibrationPoint:
    # mem-1's first point (by y) is on img-3, which is 150 px high.
    return _get(project, membrane_id).calibration.points[0]


def _rejected(project: Project, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        revalidate(project)


def _check(project: Project, match: str | None) -> None:
    """Re-validate: it must fail with ``match``, or pass when ``match`` is None."""
    if match is None:
        revalidate(project)
    else:
        _rejected(project, match)


def test_sample_project_is_valid(project):
    assert revalidate(project) == project
    batch = project.batch
    assert [image.id for image in batch.iter_images()] == ["img-2", "img-3", "img-4", "img-6"]
    assert batch.membrane_of("img-6").id == "mem-5"
    protein, band = batch.find_band("band-12")
    assert (protein.id, band.lane_index) == ("prot-7", 3)
    # The target and its loading control sit on different membranes.
    target = batch.find_protein("prot-7")
    control = batch.find_protein(target.loading_control_ids[0])
    assert batch.membrane_of(target.image_id) is not batch.membrane_of(control.image_id)


def test_empty_project_defaults():
    empty = Project()
    assert (empty.schema_version, empty.next_id) == (SCHEMA_VERSION, 1)
    assert empty.batch == Batch()
    assert list(empty.iter_ids()) == []
    # The Literal must follow SCHEMA_VERSION.
    assert get_args(Project.model_fields["schema_version"].annotation) == (SCHEMA_VERSION,)


def test_field_defaults():
    protein = Protein(
        id="prot-1",
        name="p53",
        role=Role.TARGET,
        image_id="img-2",
        box_size=BoxSize(width=4, height=4),
    )
    assert protein.expected_band_count == 1
    assert protein.mw_tolerance == 0.10
    assert protein.loading_control_ids == []
    assert protein.expected_mw is None

    band = Band(id="band-1", lane_index=0, box=Box(x=0, y=0), net=0.0, source="click")
    assert (band.band_index, band.clipped, band.apparent_mw) == (0, None, None)
    assert band.manually_edited is False

    image = ImageRef(
        id="img-1",
        file="img-1.tif",
        original_name="blot.tif",
        kind=ImageKind.CHEMILUMINESCENCE,
        sha256="0" * 64,
        width=10,
        height=10,
        polarity=Polarity.DARK_ON_LIGHT,
        background=200.0,
    )
    assert (image.bit_depth, image.marker_image_id, image.import_warnings) == (None, None, [])

    calibration = Membrane(id="mem-1").calibration
    assert (calibration.ladder, calibration.points, calibration.fit_quality) == (None, [], None)
    assert calibration.fit_method is FitMethod.LOG_LINEAR


def test_polarity_gives_the_dark_on_light_flag():
    assert Polarity.DARK_ON_LIGHT.dark_on_light is True
    assert Polarity.LIGHT_ON_DARK.dark_on_light is False


def test_reprobe_is_another_image_of_the_same_membrane(project):
    batch = project.batch
    assert batch.membrane_of(batch.find_protein("prot-9").image_id).id == "mem-1"
    revalidate(_edit(project, "prot-9", image_id="img-2"))


def test_band_in_undeclared_lane_rejected(project):
    _rejected(_edit(project, "band-10", lane_index=4), "unknown lane")


@pytest.mark.parametrize("indices", [[0, 2], [1, 0], [0, 0]])
def test_lane_indices_follow_list_order(indices):
    with pytest.raises(ValidationError, match="lane indices"):
        Batch(lanes=[Lane(index=i, label="vehicle") for i in indices])


def test_box_beyond_its_own_image_rejected():
    # α-tubulin's boxes are 20x10 on img-6 (200x100); img-2 is 340x150.
    beyond = Box(x=185, y=40)
    _rejected(_edit(make_project(), "band-13", box=beyond), "bounds")
    moved = _edit(make_project(), "prot-8", image_id="img-2")
    revalidate(_edit(moved, "band-13", box=beyond))
    _rejected(_edit(make_project(), "band-13", box=Box(x=10, y=91)), "bounds")
    revalidate(_edit(make_project(), "band-13", box=Box(x=180, y=90)))  # flush with both edges


def test_box_size_larger_than_image_rejected():
    # Checked for the box size itself, so also for a protein with no bands yet.
    for size in (BoxSize(width=201, height=10), BoxSize(width=20, height=101)):
        _rejected(_edit(make_project(), "prot-8", bands=[], box_size=size), "bounds")
    revalidate(_edit(make_project(), "prot-8", bands=[], box_size=BoxSize(width=200, height=100)))


def test_overlapping_boxes_of_one_protein_rejected():
    # band-10 spans x 18..42 (half-open); band-11 sits at x 58 on the same row.
    _rejected(_edit(make_project(), "band-11", box=Box(x=41, y=43)), "overlap")
    revalidate(_edit(make_project(), "band-11", box=Box(x=42, y=43)))  # edges touch only


def test_boxes_of_different_proteins_may_overlap(project):
    # GAPDH moved onto β-catenin's image, with a box exactly on band-10's.
    _edit(project, "prot-9", image_id="img-2")
    _edit(project, "band-17", box=Box(x=18, y=43))
    revalidate(project)


def test_lane_and_band_index_pair_unique():
    _rejected(_edit(make_project(), "band-11", lane_index=0), "two bands")
    two_in_lane_0 = revalidate(_edit(make_project(), "band-11", lane_index=0, band_index=1))
    bands = two_in_lane_0.batch.find_protein("prot-7").bands
    assert [(b.id, b.lane_index, b.band_index) for b in bands[:2]] == [
        ("band-10", 0, 0),
        ("band-11", 0, 1),
    ]


def test_bands_and_points_are_canonically_ordered(project):
    batch = project.batch
    for protein in batch.proteins:
        keys = [(b.lane_index, b.band_index) for b in protein.bands]
        assert keys == sorted(keys)
    for membrane in batch.membranes:
        keys = [(p.y, p.mw, p.source.value, p.image_id) for p in membrane.calibration.points]
        assert keys == sorted(keys)
    assert [b.id for b in batch.find_protein("prot-8").bands] == [
        f"band-{n}" for n in range(13, 17)
    ]

    # Another input order gives an equal model.
    doc = project.model_dump()
    for protein in doc["batch"]["proteins"]:
        protein["bands"].reverse()
    for membrane in doc["batch"]["membranes"]:
        membrane["calibration"]["points"].reverse()
    assert Project.model_validate(doc) == project


@pytest.mark.parametrize(
    ("reference", "match"),
    [(None, None), ("10 µM", None), ("DMSO", "reference condition")],
)
def test_reference_condition_must_be_a_lane_condition(project, reference, match):
    project.batch.reference_condition = reference
    _check(project, match)


def test_protein_image_must_resolve(project):
    _rejected(_edit(project, "prot-7", image_id="img-99"), "unknown image")


def test_protein_on_visible_marker_image_rejected(project):
    _rejected(_edit(project, "prot-9", image_id="img-3"), "marker image")


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        pytest.param(
            lambda p: _edit(p, "prot-7", loading_control_ids=["prot-99"]),
            "loading control",
            id="unknown-id",
        ),
        pytest.param(
            lambda p: _edit(
                _edit(p, "prot-9", role=Role.TARGET), "prot-7", loading_control_ids=["prot-9"]
            ),
            "loading control",
            id="a-target",
        ),
        pytest.param(
            lambda p: _edit(p, "prot-7", loading_control_ids=["prot-7"]),
            "loading control",
            id="itself",
        ),
        pytest.param(
            lambda p: _edit(p, "prot-7", loading_control_ids=["prot-8", "prot-8"]),
            "loading control",
            id="duplicate",
        ),
        pytest.param(
            lambda p: _edit(p, "prot-9", loading_control_ids=["prot-8"]),
            "loading control",
            id="loading-control-lists-one",
        ),
        pytest.param(
            lambda p: _edit(p, "prot-7", loading_control_ids=[]),
            None,
            id="target-with-empty-list",
        ),
    ],
)
def test_loading_control_references(project, edit, match):
    edit(project)
    _check(project, match)


def test_protein_names_unique_and_non_blank():
    _rejected(_edit(make_project(), "prot-8", name="GAPDH"), "duplicate protein name")
    for blank in ("", "  "):
        _rejected(_edit(make_project(), "prot-8", name=blank), "blank")
    revalidate(_edit(make_project(), "prot-8", name="gapdh"))  # the case policy is #42's


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        pytest.param(lambda p: p, None, id="fixture-pairing"),
        pytest.param(
            lambda p: _edit(p, "img-2", marker_image_id="img-4"),
            "marker image",
            id="to-a-chemiluminescence-image",
        ),
        pytest.param(
            lambda p: _edit(p, "img-2", marker_image_id="img-6"),
            "marker image",
            id="to-another-membrane",
        ),
        pytest.param(
            lambda p: _edit(p, "img-2", marker_image_id="img-2"),
            "marker image",
            id="to-itself",
        ),
        pytest.param(
            lambda p: _edit(p, "img-3", marker_image_id="img-2"),
            "marker image",
            id="marker-image-with-a-marker",
        ),
        pytest.param(
            lambda p: _edit(p, "img-4", kind=ImageKind.MERGED, marker_image_id="img-3"),
            "marker image",
            id="merged-image-with-a-marker",
        ),
    ],
)
def test_marker_pairing(project, edit, match):
    edit(project)
    _check(project, match)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        pytest.param(
            lambda p: _set(_point(p, "mem-5"), image_id="img-2"),
            "calibration point",
            id="on-another-membrane",
        ),
        pytest.param(lambda p: _set(_point(p), y=150.5), "calibration point", id="below-bottom"),
        pytest.param(lambda p: _set(_point(p), mw=0.0), "greater than 0", id="zero-mw"),
        pytest.param(lambda p: _set(_point(p), y=150.0), None, id="at-the-bottom"),
    ],
)
def test_calibration_point_rules(project, edit, match):
    edit(project)
    _check(project, match)


def test_fit_quality_needs_two_points(project):
    calibration = _get(project, "mem-5").calibration
    calibration.fit_quality = 0.99
    revalidate(project)  # two points
    calibration.points.pop()
    _rejected(project, "at least two points")
    calibration.points.clear()
    calibration.fit_quality = None
    revalidate(project)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        *(
            pytest.param(lambda p, bad=bad: _edit(p, "prot-7", image_id=bad), "pattern", id=bad)
            for bad in ["prot-7", "img-01", "img-0", "IMG-1", "img-1\n", "../img-1", "img-1.tif"]
        ),
        pytest.param(
            lambda p: _edit(
                _edit(p, "img-4", id="img-2", file="img-2.tif"), "prot-9", image_id="img-2"
            ),
            "duplicate id",
            id="two-images-with-one-id",
        ),
        pytest.param(
            lambda p: _edit(
                _edit(p, "img-4", id="img-7", file="img-7.tif"), "prot-9", image_id="img-7"
            ),
            "duplicate id",
            id="img-7-and-prot-7",
        ),
        pytest.param(lambda p: _set(p, next_id=18), "next_id", id="number-not-below-next_id"),
    ],
)
def test_id_rules(project, edit, match):
    edit(project)
    _rejected(project, match)


def test_new_id_is_sequential_and_never_reused(project):
    assert project.new_id("prot") == "prot-19"
    assert project.new_id("band") == "band-20"

    def remove_gapdh(draft: Project) -> None:
        draft.batch.proteins = [p for p in draft.batch.proteins if p.id != "prot-9"]

    project, _ = apply_change(project, remove_gapdh)
    assert project.new_id("prot") == "prot-21"  # 9 is not reused


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        *(
            pytest.param({"file": bad}, "file name", id=f"file={bad}")
            for bad in [
                "img-2.TIF",
                "img-3.tif",
                "images/img-2.tif",
                "img-2.bmp",
                "img-2",
                "img-23.tif",
            ]
        ),
        *(
            pytest.param({"original_name": bad}, "plain file name", id=f"original_name={bad!r}")
            for bad in ["dir/blot.tif", "dir\\blot.tif", ""]
        ),
        pytest.param({"original_name": "β-actin 10 µM.tif"}, None, id="non-ascii-name"),
    ],
)
def test_image_file_name_rules(project, fields, match):
    _check(_edit(project, "img-2", **fields), match)
    if match is None:  # kept verbatim: no normalization, no path handling
        assert revalidate(project).batch.find_image("img-2").original_name == "β-actin 10 µM.tif"


@pytest.mark.parametrize("digest", ["abc123", hashlib.sha256(b"x").hexdigest().upper()])
def test_sha256_is_lowercase_hex(project, digest):
    _rejected(_edit(project, "img-2", sha256=digest), "pattern")


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        pytest.param(
            lambda p: _edit(p, "img-2", background=math.nan), "finite number", id="nan-background"
        ),
        pytest.param(lambda p: _edit(p, "band-10", net=math.inf), "finite number", id="inf-net"),
        pytest.param(lambda p: _set(_point(p), y=math.nan), "finite number", id="nan-y"),
        pytest.param(lambda p: _edit(p, "band-10", net=-1.0), "greater than", id="negative-net"),
        pytest.param(lambda p: _edit(p, "prot-7", expected_mw=0.0), "greater than", id="mw-0"),
        pytest.param(
            lambda p: _edit(p, "prot-7", mw_tolerance=0.0), "greater than", id="tolerance-0"
        ),
        pytest.param(lambda p: _edit(p, "prot-7", mw_tolerance=1.0), "less than", id="tolerance-1"),
        pytest.param(
            lambda p: _edit(p, "prot-7", expected_band_count=0),
            "greater than",
            id="band-count-0",
        ),
        pytest.param(lambda p: _edit(p, "img-2", bit_depth=7), "greater than", id="bit-depth-7"),
        pytest.param(lambda p: _edit(p, "img-2", bit_depth=17), "less than", id="bit-depth-17"),
    ],
)
def test_non_finite_and_out_of_range_numbers_rejected(project, edit, match):
    edit(project)
    _rejected(project, match)


def test_negative_zero_stored_as_positive_zero(project):
    stored = revalidate(_edit(project, "img-2", background=-0.0))
    assert math.copysign(1, stored.batch.find_image("img-2").background) == 1


@pytest.mark.parametrize(
    ("locate", "key", "value"),
    [
        pytest.param(
            lambda d: d["batch"]["proteins"][0]["bands"][0],
            "background",  # the old Band's background box
            {"x": 0, "y": 0},
            id="band",
        ),
        pytest.param(lambda d: d, "content_hash", "0" * 64, id="project"),
        pytest.param(lambda d: d["batch"]["lanes"][0], "condition", "vehicle", id="lane"),
    ],
)
def test_unknown_fields_rejected(project, locate, key, value):
    doc = project.model_dump(mode="json")
    locate(doc)[key] = value
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Project.model_validate(doc)


def test_apply_change_returns_validated_copy(project):
    snapshot = project.model_copy(deep=True)

    def add_p53(draft: Project) -> str:
        protein_id = draft.new_id("prot")
        draft.batch.proteins.append(
            Protein(
                id=protein_id,
                name="p53",
                role=Role.TARGET,
                image_id="img-4",
                box_size=BoxSize(width=24, height=14),
            )
        )
        return protein_id

    changed, protein_id = apply_change(project, add_p53)
    assert protein_id == "prot-19"
    assert changed.batch.find_protein(protein_id).name == "p53"
    assert changed.next_id == 20
    assert project == snapshot
    assert project.next_id == 19


def test_apply_change_failure_leaves_original(project):
    snapshot = project.model_copy(deep=True)

    def move_onto_band_10(draft: Project) -> None:
        draft.new_id("band")  # consumed in the draft only
        _edit(draft, "band-11", box=Box(x=18, y=43))

    with pytest.raises(ValidationError, match="overlap"):
        apply_change(project, move_onto_band_10)
    assert project == snapshot
    assert project.next_id == 19


def test_lookup_of_unknown_id_raises(project):
    batch = project.batch
    lookups = [
        (batch.find_image, "img-99"),
        (batch.membrane_of, "img-99"),
        (batch.find_protein, "prot-99"),
        (batch.find_band, "band-99"),
        (batch.find_image, "prot-7"),  # an id of another kind
    ]
    for lookup, obj_id in lookups:
        with pytest.raises(UnknownIdError, match=obj_id):
            lookup(obj_id)
    assert issubclass(UnknownIdError, LookupError)


def test_role_is_shared_with_analyze():
    assert analyze.Role is model.Role
    assert [role.value for role in model.Role] == ["target", "loading control"]
