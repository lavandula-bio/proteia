# SPDX-License-Identifier: Apache-2.0
"""The batch's access points (#222, #223): one per concept, for what the stored
shape holds, and no reader outside the model that goes around them.

Each concept is read through one accessor of :class:`~proteia.core.model.Batch`,
so that a change of the stored shape changes the accessor and not its readers:
the images in order and by id, a membrane's images and register groups, the
membrane of an image and the membrane areas on it, the project's lanes and a
membrane's lane table, the lanes of a protein, a protein's row, image (and its
id), membrane and boxes, the proteins on an image, and the area and band
lookups. The stored shape (schema 3): the batch holds the images, a membrane
holds its area on each of its images and names its lane table, and a protein
holds its boxes in its one row, on one area.

The second half reads the source of the package (the model aside, which holds
the stored shape and the accessors) and lists every access that goes around
them, whatever the names of the variables: any read, write or deletion of a
field of the stored shape (:data:`SHAPE_FIELDS`); those fields and the box
fields given to the constructor of a stored record or to a ``model_copy``
update; a membrane's own ``register_groups()`` or ``group_of(image_id)``,
which the batch serves (taking the membrane first); and a protein's box
fields read or written on anything but a ``boxes_of(...)`` call or a name that
every binding in its function sets to one, on which nothing else is read. An
access of a field of the same name on something else, such as a calibration
point's ``image_id``, a row detection's ``lanes`` or a request body's
``box_size``, and the writers of the stored shape are allowed by name in
:data:`ALLOWED`, with the reason.
"""

import ast
import inspect
import typing
from dataclasses import dataclass
from pathlib import Path

import pytest

from conftest import converted, make_project, sample_doc
from proteia.core import operations as ops
from proteia.core import storage
from proteia.core.model import (
    Batch,
    Box,
    Boxes,
    CalibrationPointSource,
    ImageKind,
    Project,
    UnknownIdError,
)
from proteia.core.operations import ErrorCode, OperationError
from proteia.core.project import lane_anchors
from proteia.web import state
from test_operations import (
    CAL_LEFT_X,
    NARROW_X,
    ROW,
    band_of,
    boxed,
    cal_blot,
    cal_marker,
    cal_y,
    calibrated,
    import_blot,
    plant,
)

CHEMI_MARKER = CalibrationPointSource.CHEMILUMINESCENCE_MARKER

SRC = Path(__file__).resolve().parents[1] / "src" / "proteia"


def _shuffled() -> Project:
    """The sample project with its membranes, images and proteins out of id
    order: membranes mem-5 (img-6) and mem-1, whose images are img-4, img-2
    (linked to its marker img-3) and img-3; proteins GAPDH (img-4), p53 (img-2,
    no band, padded, one not-detected record), α-tubulin (img-6) and β-catenin
    (img-2). Converted from schema 2 (next_id 20): the lane table lt-20, the
    areas arena-21 (img-6), arena-22 (img-4), arena-23 (img-2) and arena-24
    (img-3), and the rows row-25 to row-28, in protein order."""
    doc = sample_doc()
    batch = doc["batch"]
    mem_1, mem_5 = batch["membranes"]
    img_2, img_3, img_4 = mem_1["images"]
    mem_1["images"] = [img_4, img_2, img_3]
    batch["membranes"] = [mem_5, mem_1]
    beta, alpha, gapdh = batch["proteins"]
    p53 = {
        "id": "prot-19",
        "name": "p53",
        "role": "target",
        "image_id": "img-2",
        "loading_control_ids": ["prot-9"],
        "box_size": {"width": 14, "height": 10},
        "box_padding": {"across": 2, "along": 1},
        "undetected": [
            {
                "lane_index": 1,
                "reason": "below_detection_limit",
                "snr": 2.5,
                "threshold": 6.0,
                "region": {"x0": 40, "y0": 30, "x1": 54, "y1": 40},
                "source": "row_box",
            }
        ],
    }
    batch["proteins"] = [gapdh, p53, alpha, beta]
    doc["next_id"] = 20
    return converted(doc)


# --- The accessors ---


def test_images_in_order_and_by_id():
    batch = _shuffled().batch
    mem_5, mem_1 = batch.membranes
    assert [image.id for image in batch.iter_images()] == ["img-6", "img-4", "img-2", "img-3"]
    # A membrane's images in its own order, not by id.
    assert [image.id for image in batch.images_of(mem_1)] == ["img-4", "img-2", "img-3"]
    assert [image.id for image in batch.images_of(mem_5)] == ["img-6"]
    assert batch.find_image("img-3").original_name == "marker α.png"
    with pytest.raises(UnknownIdError):
        batch.find_image("img-99")


def test_the_membrane_of_an_image():
    batch = _shuffled().batch
    mem_5, mem_1 = batch.membranes
    assert batch.membrane_of("img-6") is mem_5
    for image_id in ("img-4", "img-2", "img-3"):
        assert batch.membrane_of(image_id) is mem_1
    for unknown in ("img-99", "mem-1", "prot-9"):
        with pytest.raises(UnknownIdError):
            batch.membrane_of(unknown)


def test_register_groups_and_group_of():
    batch = _shuffled().batch
    mem_5, mem_1 = batch.membranes
    # In the order of each group's first image on the membrane.
    assert batch.register_groups(mem_1) == [frozenset({"img-4"}), frozenset({"img-2", "img-3"})]
    assert batch.register_groups(mem_5) == [frozenset({"img-6"})]
    assert batch.group_of(mem_1, "img-3") == frozenset({"img-2", "img-3"})
    assert batch.group_of(mem_1, "img-4") == frozenset({"img-4"})
    assert batch.group_of(mem_5, "img-6") == frozenset({"img-6"})
    with pytest.raises(UnknownIdError):
        batch.group_of(mem_1, "img-6")  # an image of the other membrane


def test_lanes_of_a_protein():
    batch = _shuffled().batch
    for protein in batch.proteins:
        assert [lane.sample for lane in batch.lanes_of_protein(protein)] == ["v1", "v2", "a1", "a2"]


def test_a_proteins_image_membrane_and_boxes():
    batch = _shuffled().batch
    mem_5, mem_1 = batch.membranes
    gapdh, p53, alpha, beta = batch.proteins
    assert batch.image_of(gapdh) is batch.find_image("img-4")
    assert batch.image_of(alpha) is batch.find_image("img-6")
    assert batch.image_of(beta).id == batch.image_of(p53).id == "img-2"
    ids = [batch.image_id_of(protein) for protein in batch.proteins]
    assert ids == ["img-4", "img-2", "img-6", "img-2"]
    assert batch.membrane_of_protein(gapdh) is mem_1
    assert batch.membrane_of_protein(alpha) is mem_5
    assert batch.membrane_of_protein(p53) is mem_1

    boxes = batch.boxes_of(alpha)
    assert [band.id for band in boxes.bands] == ["band-13", "band-14", "band-15", "band-16"]
    assert (boxes.box_size.width, boxes.box_size.height) == (20, 10)
    assert boxes.undetected == []
    boxes = batch.boxes_of(p53)
    assert boxes.bands == []
    assert (boxes.box_padding.across, boxes.box_padding.along) == (2, 1)
    assert (boxes.fitted_size.width, boxes.fitted_size.height) == (10, 8)
    assert [(record.lane_index, record.snr) for record in boxes.undetected] == [(1, 2.5)]


def test_an_image_id_is_read_without_a_lookup(monkeypatch):
    # The hot paths that only key by a protein's image ask for its id: no scan
    # of the images, so no find_image call.
    batch = _shuffled().batch

    def no_lookup(self, image_id):
        raise AssertionError(f"looked up {image_id}")

    monkeypatch.setattr(Batch, "find_image", no_lookup)
    assert [batch.image_id_of(p) for p in batch.proteins] == ["img-4", "img-2", "img-6", "img-2"]


def test_boxes_are_asked_of_the_batch_and_hold_the_box_fields_only():
    # An instance method, so the batch can answer from wherever the boxes are
    # stored; its answer is typed as the box fields alone.
    assert not isinstance(inspect.getattr_static(Batch, "boxes_of"), staticmethod)
    assert typing.get_type_hints(Batch.boxes_of)["return"] is Boxes
    declared = {name for name in vars(Boxes) if not name.startswith("_")}
    assert declared | set(Boxes.__annotations__) == {
        "bands",
        "box_size",
        "box_padding",
        "undetected",
        "fitted_size",
    }
    batch = _shuffled().batch
    with pytest.raises(TypeError):
        Batch.boxes_of(batch.proteins[0])  # no batch: no answer


def test_proteins_on_an_image():
    batch = _shuffled().batch
    # In the batch's order, not by id; each image's own proteins only.
    assert [protein.id for protein in batch.proteins_on("img-2")] == ["prot-19", "prot-7"]
    assert [protein.id for protein in batch.proteins_on("img-4")] == ["prot-9"]
    assert [protein.id for protein in batch.proteins_on("img-6")] == ["prot-8"]
    assert batch.proteins_on("img-3") == []  # the marker holds none


def test_find_band():
    batch = _shuffled().batch
    protein, band = batch.find_band("band-18")
    assert (protein.id, band.lane_index, band.box.y) == ("prot-9", 1, 100)
    protein, band = batch.find_band("band-12")
    assert (protein.id, band.lane_index, band.box.x) == ("prot-7", 3, 138)
    for unknown in ("band-99", "prot-7"):
        with pytest.raises(UnknownIdError):
            batch.find_band(unknown)


def test_the_areas_on_an_image_and_the_area_lookup():
    batch = _shuffled().batch
    mem_5, mem_1 = batch.membranes
    held = [[(arena.id, arena.image_id) for arena in m.arenas] for m in batch.membranes]
    assert held == [
        [("arena-21", "img-6")],
        [("arena-22", "img-4"), ("arena-23", "img-2"), ("arena-24", "img-3")],
    ]
    [(membrane, arena)] = batch.arenas_on("img-2")
    assert (membrane is mem_1, arena.id) == (True, "arena-23")
    [(membrane, arena)] = batch.arenas_on("img-6")
    assert (membrane is mem_5, arena.id) == (True, "arena-21")
    assert batch.arenas_on("img-99") == []
    membrane, arena = batch.find_arena("arena-24")
    assert (membrane is mem_1, arena.image_id) == (True, "img-3")
    for unknown in ("arena-99", "img-2"):
        with pytest.raises(UnknownIdError):
            batch.find_arena(unknown)


def test_the_lanes_of_the_project_and_of_a_membrane():
    batch = _shuffled().batch
    [table] = batch.lane_tables
    assert table.id == "lt-20"
    assert batch.only_lanes() is table.lanes
    for membrane in batch.membranes:
        assert batch.lane_table_of(membrane) is table
    with pytest.raises(UnknownIdError):
        batch.lane_table_of(batch.membranes[0].model_copy(update={"lane_table_id": "lt-99"}))
    assert Batch().only_lanes() == []


def test_a_proteins_row():
    batch = _shuffled().batch
    rows = [batch.only_row(protein) for protein in batch.proteins]
    assert [(row.id, row.arena_id) for row in rows] == [
        ("row-25", "arena-22"),
        ("row-26", "arena-23"),
        ("row-27", "arena-21"),
        ("row-28", "arena-23"),
    ]
    assert all(batch.boxes_of(p) is batch.only_row(p) for p in batch.proteins)
    beta = batch.proteins[3]
    for held in ([], [rows[3], rows[3]]):
        with pytest.raises(ValueError):
            batch.only_row(beta.model_copy(update={"rows": held}))


# --- Readers moved onto the accessors, pinned where no test pinned them ---


def test_lane_anchors_leave_out_boxes_touching_an_edge():
    # β-catenin's lane-4 box (24 px wide) on the 340 px img-2, at each edge and
    # one pixel inside it.
    found = {}
    for x in (0, 1, 315, 316):
        batch = make_project().batch
        batch.find_band("band-12")[1].box = Box(x=x, y=43)
        found[x] = lane_anchors(batch, batch.arenas_on("img-2")[0][1].id)
    kept = [(30.0, 0), (70.0, 1)]
    assert found == {0: kept, 1: [*kept, (13.0, 3)], 315: [*kept, (327.0, 3)], 316: kept}


def test_missing_lanes_sit_on_the_row_of_the_first_bands():
    batch = make_project().batch
    beta = batch.find_protein("prot-7")  # boxes 14 px high at y=43 in lanes 1, 2 and 4
    anchors = lane_anchors(batch, batch.arenas_on("img-2")[0][1].id)
    assert state._missing_lanes(batch, beta, anchors, None) == [
        {"lane_index": 2, "x": 110.0, "y": 50.0}
    ]


def test_an_older_fit_is_read_from_the_bands_of_its_own_membrane():
    # mem-5 has no point, so no curve: a #58 build leaves log_linear there. Only
    # an MW on its own image's bands tells of a fit from before #58, not
    # β-catenin's MW on mem-1.
    doc = make_project().model_dump(mode="json")
    doc["batch"]["membranes"][1]["calibration"] = {}
    batch = Project.model_validate(doc).batch
    mem_5 = batch.membranes[1]
    assert ops._fitted_before_58(mem_5, batch) is False
    batch.find_band("band-13")[1].apparent_mw = 50.0  # α-tubulin, on img-6
    assert ops._fitted_before_58(mem_5, batch) is True


@pytest.mark.parametrize(("width", "refused"), [(20, True), (21, False)])
def test_a_click_box_as_wide_as_the_lane_pitch_shows_its_lane(width, refused):
    # β-catenin's row boxes on img-2 at centres 30, 50 and 70 (a pitch of 20 px);
    # GAPDH's box grown from a click in lane 2, centred 12 px off that lane's
    # column. Only a box wider than the pitch is left out of the check.
    doc = sample_doc()
    beta, _, gapdh = doc["batch"]["proteins"]
    beta["box_size"] = {"width": 10, "height": 14}
    in_lane_order = sorted(beta["bands"], key=lambda band: band["lane_index"])  # as stored
    for band, x in zip(in_lane_order, (25, 45, 65), strict=True):
        band.update(box={"x": x, "y": 43}, source="row_box", manually_edited=False)
        band["lane_index"] = {"band-10": 0, "band-11": 1, "band-12": 2}[band["id"]]
    gapdh["image_id"] = "img-2"
    gapdh["box_size"] = {"width": width, "height": 14}
    gapdh["bands"] = [
        {**gapdh["bands"][0], "lane_index": 1, "box": {"x": 62 - width // 2, "y": 80}}
    ]
    batch = converted(doc).batch
    error = ops._misnumbered_lanes(batch, batch.arenas_on("img-2")[0][1], ())
    if not refused:
        assert error is None
        return
    assert error is not None and error.code is ErrorCode.ROW_LANES_UNCLEAR
    assert error.ids == ("band-11", "band-17")


def test_a_marker_of_another_membrane_or_kind_is_refused_in_its_own_words(tmp_path):
    c = calibrated(tmp_path, sides=())
    s = c.session
    other = import_blot(s, cal_marker(), "other membrane.tif", kind=ImageKind.VISIBLE_MARKER)
    second = import_blot(s, cal_blot(), "second chemi.tif", membrane_id=c.membrane)
    for marker in (other, second):
        before = s.project
        with pytest.raises(OperationError) as info:
            ops.set_marker_image(s, c.blot, marker)
        assert info.value.code is ErrorCode.INVALID_INPUT
        assert str(info.value) == (
            f"{marker} is not a visible-light marker or merged image of membrane {c.membrane}"
        )
        assert info.value.ids == (marker,)
        assert s.project is before


def test_a_refused_link_leaves_the_images_as_they_were(tmp_path):
    # The link is tried on a copy before the change: a refusal must leave the
    # committed image unlinked, not only the session on the same project.
    c = calibrated(tmp_path)
    s = c.session
    ops.set_marker_image(s, c.blot, None)
    y70 = cal_y(70, CAL_LEFT_X)
    ops.add_calibration_point(s, c.blot, y70 + 5.0, 70, CHEMI_MARKER, x=CAL_LEFT_X, snap=False)
    with pytest.raises(OperationError) as info:
        ops.set_marker_image(s, c.blot, c.marker)
    assert info.value.code is ErrorCode.DUPLICATE_MW
    assert s.project.batch.find_image(c.blot).marker_image_id is None
    assert storage.content_hash(s.project) == s.project.log[-1].content_hash


def test_a_link_is_checked_on_a_copy_and_leaves_the_committed_project_alone(tmp_path):
    # The check copies the image it links, not the batch's images it shares:
    # the project committed before the link keeps its image unlinked, and
    # the link is a change (logged), not a no-op.
    c = calibrated(tmp_path)
    s = c.session
    ops.set_marker_image(s, c.blot, None)
    before = s.project
    entries = len(before.log)
    ops.set_marker_image(s, c.blot, c.marker)
    assert before.batch.find_image(c.blot).marker_image_id is None
    assert s.project.batch.find_image(c.blot).marker_image_id == c.marker
    assert len(s.project.log) == entries + 1
    assert s.project.log[-1].action == "set_marker_image"


def test_a_padding_reports_the_change_of_a_net_under_one(tmp_path):
    s, _, beta = boxed(tmp_path)
    band_id = ops.place_box(s, beta, NARROW_X, ROW, lane_index=0, grow=True)

    def faint(draft: Project) -> None:
        draft.batch.find_band(band_id)[1].net = 0.5

    plant(s, faint)
    change = ops.set_box_padding(s, beta, across=1)
    ratio = band_of(s, band_id).net / 0.5 - 1
    assert change.net_change == (ratio, ratio)


# --- No reader around them ---

# A protein's box fields: read and written only on what Batch.boxes_of
# returns, on which nothing else is read.
BOX_FIELDS = frozenset({"bands", "box_size", "box_padding", "undetected", "fitted_size"})
# The stored shape's other fields the accessors serve: any read, write or
# deletion of one outside the model, whatever holds it, is listed.
SHAPE_FIELDS = frozenset(
    {"lanes", "images", "image_id", "arenas", "rows", "lane_tables", "lane_table_id", "arena_id"}
)
# The register-group methods the batch serves, by the number of arguments a
# membrane's own would take (the batch's take the membrane first).
GROUP_METHODS = {"register_groups": 0, "group_of": 1}
# The stored records whose constructors take those fields.
RECORDS = frozenset({"Batch", "LaneTable", "Membrane", "Arena", "Protein", "Row"})


@dataclass(frozen=True, order=True)
class Reader:
    """An access around the accessors: the file (under ``src/proteia``), the
    function (dotted, nested functions included), the receiver as written
    (``Protein(...)`` for a constructor, ``x.model_copy(update=...)`` for an
    update), and the field."""

    path: str
    function: str
    receiver: str
    attr: str

    def __str__(self) -> str:
        return f"{self.path} {self.function}: {self.receiver}.{self.attr}"


def _is_boxes_call(node: ast.expr | None) -> bool:
    """Whether ``node`` is a call of ``<anything>.boxes_of(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "boxes_of"
    )


def _bindings(function: ast.AST) -> dict[str, bool]:
    """The names a function binds in its own body (not in nested functions),
    each True only if every binding of it is a ``boxes_of`` call."""
    bound: dict[str, bool] = {}

    def bind(name: str, holds_boxes: bool) -> None:
        bound[name] = bound.get(name, True) and holds_boxes

    def targets(node: ast.AST, value: ast.expr | None) -> None:
        if isinstance(node, ast.Name):
            bind(node.id, _is_boxes_call(value))
        elif isinstance(node, (ast.Tuple, ast.List)):
            # a, b = x, y binds a to x and b to y; any other unpacking, to no call.
            pairs = (
                isinstance(value, (ast.Tuple, ast.List))
                and len(value.elts) == len(node.elts)
                and not any(isinstance(e, ast.Starred) for e in [*node.elts, *value.elts])
            )
            for i, element in enumerate(node.elts):
                targets(element, value.elts[i] if pairs else None)
        elif isinstance(node, ast.Starred):
            targets(node.value, None)

    if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        arguments = function.args
        for arg in [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *filter(None, (arguments.vararg, arguments.kwarg)),
        ]:
            bind(arg.arg, False)
    body = function.body if isinstance(function.body, list) else [function.body]
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(node.name, False)
            continue  # a nested scope binds its own names
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                targets(target, node.value)
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            targets(node.target, node.value)
        elif isinstance(node, (ast.AugAssign, ast.For, ast.AsyncFor, ast.comprehension)):
            targets(node.target, None)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            targets(node.optional_vars, None)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bind(node.name, False)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bind((alias.asname or alias.name).split(".")[0], False)
        stack.extend(ast.iter_child_nodes(node))
    return bound


class _Checker(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.names: list[str] = []  # the enclosing functions and classes
        self.scopes: list[dict[str, bool]] = []  # their bindings, innermost last
        self.found: list[Reader] = []
        self.called: set[int] = set()  # the group methods judged by their call

    def _note(self, receiver: str, field: str) -> None:
        function = ".".join(self.names) or "<module>"
        self.found.append(Reader(self.path, function, receiver, field))

    def _enter(self, node: ast.AST, name: str) -> None:
        self.names.append(name)
        self.scopes.append(_bindings(node))
        self.generic_visit(node)
        self.scopes.pop()
        self.names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter(node, node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._enter(node, "<lambda>")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.append(node.name)
        self.generic_visit(node)
        self.names.pop()

    def _holds_boxes(self, receiver: ast.expr) -> bool:
        if _is_boxes_call(receiver):
            return True
        if isinstance(receiver, ast.Name):
            for scope in reversed(self.scopes):
                if receiver.id in scope:
                    return scope[receiver.id]
        return False

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if isinstance(func, ast.Attribute) and func.attr in GROUP_METHODS:
            self.called.add(id(func))
            if len(node.args) + len(node.keywords) == GROUP_METHODS[func.attr]:
                self._note(ast.unparse(func.value), func.attr)  # the membrane's own
        if name in RECORDS:
            for keyword in node.keywords:
                if keyword.arg in SHAPE_FIELDS | BOX_FIELDS:
                    self._note(f"{name}(...)", keyword.arg)
        if name == "model_copy" and isinstance(func, ast.Attribute):
            for keyword in node.keywords:
                if keyword.arg == "update" and isinstance(keyword.value, ast.Dict):
                    for key in keyword.value.keys:
                        if isinstance(key, ast.Constant) and key.value in SHAPE_FIELDS | BOX_FIELDS:
                            self._note(
                                f"{ast.unparse(func.value)}.model_copy(update=...)", key.value
                            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        holds_boxes = self._holds_boxes(node.value)
        if node.attr in BOX_FIELDS:
            around = not holds_boxes  # read, write or deletion
        elif holds_boxes:
            around = True  # an answer of boxes_of holds the box fields only
        elif node.attr in SHAPE_FIELDS:
            around = True  # read, write or deletion, whatever holds it
        elif node.attr in GROUP_METHODS:
            around = id(node) not in self.called  # a call is judged by its arguments
        else:
            around = False
        if around:
            self._note(ast.unparse(node.value), node.attr)
        self.generic_visit(node)


def direct_readers(source: str, path: str) -> list[Reader]:
    """The reads in ``source`` (the file ``path``) that go around the accessors."""
    checker = _Checker(path)
    checker.visit(ast.parse(source))
    return checker.found


def _package_readers() -> set[Reader]:
    found: set[Reader] = set()
    for file in sorted(SRC.rglob("*.py")):
        path = file.relative_to(SRC).as_posix()
        if path == "core/model.py":  # the stored shape and the accessors themselves
            continue
        found.update(direct_readers(file.read_text(encoding="utf-8"), path))
    return found


def _allow(why: str, *reads: str) -> dict[tuple[str, str, str, str], str]:
    """Each read, written as the check prints it (``file function:
    receiver.attr``), allowed for ``why``."""
    allowed = {}
    for read in reads:
        path, rest = read.split(" ", 1)
        function, expression = rest.split(": ")
        receiver, attr = expression.rsplit(".", 1)
        allowed[path, function, receiver, attr] = why
    return allowed


# The reads the check finds that do not go around an accessor, or that no
# accessor of today's shape serves: (file, function, receiver, attribute) -> why.
ALLOWED: dict[tuple[str, str, str, str], str] = {
    **_allow(
        "a calibration point's image, not a protein's",
        "core/mwcal.py _fit: p.image_id",
        "core/operations.py _find_point: point.image_id",
        "core/operations.py _ladder_refusal.at: point.image_id",
        "core/operations.py _placed.snapped_where: old.image_id",
        "core/operations.py _point_json: point.image_id",
        "core/operations.py _point_key: point.image_id",
        "core/operations.py add_calibration_point: p.image_id",
        "core/operations.py clear_calibration.cleared: point.image_id",
        "core/operations.py edit_calibration_point: old.image_id",
        "core/operations.py edit_calibration_point: p.image_id",
        "core/operations.py remove_image.change: point.image_id",
        "core/operations.py set_ladder_points: p.image_id",
        "core/operations.py set_ladder_points.edit: p.image_id",
        "core/results.py _calibration_notices: p.image_id",
        "core/results.py _ladder_images: p.image_id",
        "core/results.py _mw_protein_notices: point.image_id",
        "web/state.py _membrane: point.image_id",
    ),
    **_allow(
        "the image a lane span was read from (mwrow.LaneSpan), not a protein's",
        "core/operations.py detect_mw_row: lanes_span.image_id",
    ),
    **_allow(
        "a row detection's lanes (rowdetect.RowDetection and its parts) or the lane count an"
        " MW search names (_MwSearch), not the lane table",
        "core/operations.py _mw_refusal_words: found.lanes",
        "core/operations.py _mw_refusal_words: mw.lanes",
        "core/operations.py _not_recorded: found.lanes",
        "core/operations.py _off_row_step: found.lanes",
        "core/operations.py _place_row: found.lanes",
        "core/operations.py _place_row.params: found.lanes",
        "core/operations.py _row_refusal: found.lanes",
        "core/operations.py _span_step: mw.lanes",
        "core/rowdetect.py RowDetection.slots: self.lanes",
        "core/rowdetect.py _assign: a.lanes",
        "core/rowdetect.py _assign: best.lanes",
        "core/rowdetect.py _assign.run: a.lanes",
        "core/rowdetect.py _count_components: res.lanes",
        "core/rowdetect.py _doubts: assign.lanes",
        "core/rowdetect.py _empty_lanes: res.lanes",
        "core/rowdetect.py _lanes_from: assign.lanes",
        "core/rowdetect.py _stage2_free: res.lanes",
        "core/rowdetect.py detect_row: res.lanes",
        "core/rowdetect.py detect_row_along: found.lanes",
    ),
    **_allow(
        "a row detector's candidate rows of pixels (rowdetect), not a protein's rows",
        "core/rowdetect.py _empty_lanes: cand.rows",
        "core/rowdetect.py _slot_surfaces: cand.rows",
        "core/rowdetect.py _stage2_free: res.cand.rows",
    ),
    **_allow(
        "an answer, a request body, a result set, a series, reduced samples or the undo"
        " history: not the stored batch",
        "core/export.py result_set_table: results.lanes",
        "core/results.py _compute: red.lanes",
        "core/session.py ProjectSession._move: target.images",
        "core/session.py ProjectSession._retained_files: state.images",
        "web/api.py _padding_change: change.box_size",
        "web/api.py _padding_change: change.fitted_size",
        "web/api.py _row_placement: placement.box_size",
        "web/api.py add_protein: body.box_size",
        "web/api.py add_arena: body.image_id",
        "web/api.py add_protein: body.arena_id",
        "web/api.py add_protein: body.image_id",
        "web/api.py clear_boxes: cleared.undetected",
        "web/api.py list_diagnostics: plan.images",
        "web/api.py set_lanes: body.lanes",
        "web/api.py write_diagnostics: body.images",
        "web/diagnostics.py Plan.digest: self.images",
        "web/diagnostics.py Plan.files: self.images",
        "web/diagnostics.py Plan.omitted: self.images",
        "web/results_view.py _series: series.undetected",
    ),
    **_allow(
        "a writer of the stored shape, which changes with it: the import adds an image to the"
        " batch, after its membrane's images, and its membrane's area on it, and makes the"
        " project's lane table while it has none; the removal takes the image and that area"
        " out; adding a protein builds it with its row on its image's area; the lane table's"
        " declaration writes the project's one table; and the results build a copy of the"
        " batch with every lane of its tables included",
        "core/operations.py _image_slot: arena.image_id",
        "core/operations.py _image_slot: batch.images",
        "core/operations.py _image_slot: m.arenas",
        "core/operations.py _lane_table: draft.batch.lane_tables",
        "core/operations.py add_protein.change: Protein(...).rows",
        "core/operations.py add_protein.change: Row(...).arena_id",
        "core/operations.py add_protein.change: Row(...).box_size",
        "core/operations.py import_image.change: Arena(...).image_id",
        "core/operations.py import_image.change: Membrane(...).lane_table_id",
        "core/operations.py import_image.change: batch.images",
        "core/operations.py import_image.change: membrane.arenas",
        "core/operations.py remove_image.change: arena.image_id",
        "core/operations.py remove_image.change: batch.images",
        "core/operations.py remove_image.change: membrane.arenas",
        "core/operations.py set_lanes.change: draft.batch.lane_tables",
        "core/operations.py set_lanes.change: table.lanes",
        "core/results.py compute_results: batch.lane_tables",
        "core/results.py compute_results: batch.model_copy(update=...).lane_tables",
        "core/results.py compute_results: table.lanes",
        "core/results.py compute_results: table.model_copy(update=...).lanes",
    ),
    **_allow(
        "a calibration point's image, not a protein's (#243: an area operation moves or"
        " keeps the points on its images, and an outline holds or leaves them)",
        "core/operations.py _held: point.image_id",
        "core/operations.py assign_arena: p.image_id",
        "core/operations.py assign_arena.change: p.image_id",
        "core/operations.py remove_arena.change: p.image_id",
        "core/operations.py set_marker_image.edit: p.image_id",
    ),
    **_allow(
        "a membrane area's own image (Arena.image_id), not a protein's: where an area lies,"
        " which the readers of areas take from the area the accessors give them",
        "core/mwrow.py lane_span: other.image_id",
        "core/operations.py _area_target: arena.image_id",
        "core/operations.py _fits: arena.image_id",
        "core/operations.py _here: arena.image_id",
        "core/operations.py _refresh_mw: arena.image_id",
        "core/operations.py assign_arena: held.image_id",
        "core/export.py membrane_lines: arena.image_id",
        "core/project.py _anchored: arena.image_id",
        "core/results.py _compute: arena.image_id",
        "core/results.py _mw_notices: arena.image_id",
        "web/state.py _arena: arena.image_id",
    ),
    **_allow(
        "a writer of the stored shape, which changes with it (#243): drawing, redrawing, giving"
        " and removing a membrane area, a marker link that makes the linked images' areas"
        " equal, and the order of the batch's images, membrane by membrane, after them",
        "core/operations.py _adopt: Arena(...).image_id",
        "core/operations.py _adopt: a.image_id",
        "core/operations.py _adopt: membrane.arenas",
        "core/operations.py _order_images: arena.image_id",
        "core/operations.py _order_images: batch.images",
        "core/operations.py _order_images: m.arenas",
        "core/operations.py add_arena.change: Arena(...).image_id",
        "core/operations.py add_arena.change: Membrane(...).lane_table_id",
        "core/operations.py add_arena.change: target.arenas",
        "core/operations.py assign_arena.change: Membrane(...).lane_table_id",
        "core/operations.py assign_arena.change: a.image_id",
        "core/operations.py assign_arena.change: new.arenas",
        "core/operations.py assign_arena.change: old.arenas",
        "core/operations.py remove_arena.change: edited.arenas",
    ),
    **_allow(
        "an answer (AreaChange), not the stored batch",
        "core/operations.py add_arena.params: result.lane_table_id",
    ),
}


def test_no_direct_reader_left():
    left = sorted(
        reader
        for reader in _package_readers()
        if (reader.path, reader.function, reader.receiver, reader.attr) not in ALLOWED
    )
    assert not left, f"{len(left)} reads go around the batch's accessors:\n" + "\n".join(
        map(str, left)
    )


def test_allowed_readers_are_still_there():
    found = {(r.path, r.function, r.receiver, r.attr) for r in _package_readers()}
    stale = sorted(set(ALLOWED) - found)
    assert not stale, "allowed reads no longer in the source:\n" + "\n".join(map(str, stale))


@pytest.mark.parametrize(
    ("source", "attr"),
    [
        ("def f(batch, protein):\n    return protein.bands\n", "bands"),
        ("def f(protein, size):\n    protein.box_size = size\n", "box_size"),
        ("def f(protein):\n    return protein.box_padding\n", "box_padding"),
        ("def f(protein):\n    return [u for u in protein.undetected]\n", "undetected"),
        ("def f(protein):\n    return protein.fitted_size\n", "fitted_size"),
        ("def f(batch):\n    return len(batch.lanes)\n", "lanes"),
        ("def f(session):\n    return session.project.batch.lanes\n", "lanes"),
        # Whatever the receiver is called: a batch is not always named "batch".
        ("def f(committed):\n    return committed.lanes\n", "lanes"),
        ("def f(before):\n    return [i.id for i in before.membranes[0].images]\n", "images"),
        ("def f(membrane):\n    return membrane.images\n", "images"),
        ("def f(protein):\n    return protein.image_id\n", "image_id"),
        # The schema-3 shape (#223): areas, rows and lane tables.
        ("def f(membrane):\n    return [a.id for a in membrane.arenas]\n", "arenas"),
        ("def f(protein):\n    return protein.rows[0]\n", "rows"),
        ("def f(batch):\n    return batch.lane_tables[0].id\n", "lane_tables"),
        ("def f(membrane):\n    return membrane.lane_table_id\n", "lane_table_id"),
        ("def f(row):\n    return row.arena_id\n", "arena_id"),
        ("def f(table):\n    return len(table.lanes)\n", "lanes"),
        ("def f(i):\n    return Arena(id='arena-1', image_id=i)\n", "image_id"),
        ("def f(a, s):\n    return Row(id='row-1', arena_id=a)\n", "arena_id"),
        ("def f(s):\n    return model.Row(id='row-1', box_size=s)\n", "box_size"),
        ("def f(t):\n    return Membrane(id='mem-1', lane_table_id=t)\n", "lane_table_id"),
        ("def f(r):\n    return Protein(id='prot-1', rows=r)\n", "rows"),
        ("def f(n):\n    return LaneTable(id='lt-1', lanes=n)\n", "lanes"),
        ("def f(b):\n    return b.model_copy(update={'arenas': []})\n", "arenas"),
        # Writes and deletions, constructors and updates of the stored shape.
        ("def f(membrane, kept):\n    membrane.images = kept\n", "images"),
        ("def f(after, lanes):\n    after.lanes = lanes\n", "lanes"),
        ("def f(protein):\n    del protein.image_id\n", "image_id"),
        ("def f(image):\n    return Protein(id='prot-1', image_id=image)\n", "image_id"),
        ("def f(size):\n    return model.Protein(id='prot-1', box_size=size)\n", "box_size"),
        ("def f(images):\n    return Membrane(id='mem-1', images=images)\n", "images"),
        ("def f(b):\n    return b.model_copy(update={'lanes': []})\n", "lanes"),
        # A membrane's own group methods, by their arguments; or not called.
        ("def f(membrane, image_id):\n    return membrane.group_of(image_id)\n", "group_of"),
        ("def f(membrane):\n    return membrane.register_groups()\n", "register_groups"),
        ("def f(m):\n    return m.register_groups\n", "register_groups"),
        # The answer of boxes_of holds the box fields only.
        ("def f(b, p):\n    return b.boxes_of(p).name\n", "name"),
        ("def f(b, p):\n    own = b.boxes_of(p)\n    return own.role\n", "role"),
        # Named like the accessor's answer, but not bound to it here.
        ("def f(protein):\n    boxes = protein\n    return boxes.bands\n", "bands"),
        (
            "def f(b, a, c):\n    boxes = b.boxes_of(a)\n    boxes = c\n    return boxes.bands\n",
            "bands",
        ),
        (
            "def f(b, a):\n    boxes = b.boxes_of(a)\n    return lambda boxes: boxes.bands\n",
            "bands",
        ),
        ("def f(b, ps):\n    return [boxes.bands for boxes in ps]\n", "bands"),
        ("def f(b, a):\n    other, boxes = b.boxes_of(a), a\n    return boxes.bands\n", "bands"),
        (
            "def f(b, a):\n    boxes, other = (b.boxes_of(a), a)[::-1]\n    return boxes.bands\n",
            "bands",
        ),
    ],
)
def test_the_check_finds_each_kind_of_read(source, attr):
    [reader] = direct_readers(source, "x.py")
    assert (reader.path, reader.function.split(".")[0], reader.attr) == ("x.py", "f", attr)


def test_the_check_passes_reads_through_the_accessors():
    source = (
        "def f(batch, committed, protein, membrane, image_id, size):\n"
        "    boxes = batch.boxes_of(protein)\n"
        "    boxes.box_size = size\n"
        "    own, n = committed.boxes_of(protein), 1\n"
        "    def g():\n"
        "        return boxes.bands\n"
        "    return (\n"
        "        boxes.box_padding,\n"
        "        own.bands,\n"
        "        batch.boxes_of(protein).undetected,\n"
        "        committed.boxes_of(protein).fitted_size,\n"
        "        batch.lanes_of_protein(protein),\n"
        "        batch.images_of(membrane),\n"
        "        batch.image_of(protein).id,\n"
        "        committed.image_id_of(protein),\n"
        "        committed.group_of(membrane, image_id),\n"
        "        batch.register_groups(membrane),\n"
        "        batch.only_lanes(),\n"
        "        batch.lane_table_of(membrane).id,\n"
        "        batch.arenas_on(image_id),\n"
        "        batch.find_arena(image_id)[1].id,\n"
        "        batch.only_row(protein).id,\n"
        "        Arena(id='arena-1', source='manual'),\n"
        "        Protein(id='prot-1', name='p53'),\n"
        "        protein.model_copy(update={'name': 'p53'}),\n"
        "        g,\n"
        "    )\n"
    )
    assert direct_readers(source, "x.py") == []
