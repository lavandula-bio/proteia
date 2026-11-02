# SPDX-License-Identifier: Apache-2.0
"""Golden schema-2 projects (#221): what projects built at 05b4af5 store, and
what the current build derives from them.

``tests/data/schema2/<case>/`` holds three small projects built through the
operations on synthetic images (``make_fixtures.py`` says what each holds):
``project.json`` and ``images/`` as 05b4af5 wrote them; ``stored.json``, every
value they store (and the hashes of the files it was written from), written
once at that commit and never again; and ``derived.json``, what the current
build derives from them (the results with their notices, the exported lane
tables, the calibration fits as the page shows them, the history check, and
how the values a fresh build of the same project from the same image files
stores differ from the frozen ones: not at all, so far).

Stored values are copied, never recomputed, so they match exactly on every
platform; they are compared value by value, never through a hash of the
content in the build's own shape, so a project held in another shape with every
value intact matches. Derived floats pass through logarithms and statistics
functions that may differ in the last bits between platforms: they match
within a relative 1e-12, and a lane-table cell may be one unit off in its last
decimal only where the golden marks its value as within 1e-12 of a rounding
midpoint. Text, integers, booleans, None, keys and list order match exactly.

A pull request that means to change the derived output refreshes it with
``make_fixtures.REFRESH_COMMAND`` and says why in its description; the frozen
files never change.
"""

import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_doc
from proteia.core import operations as ops
from proteia.core import record, storage

DATA = Path(__file__).parent / "data" / "schema2"
FROZEN = "05b4af5b08e51b450436e9446651f5650bff48e4"
# How far the frozen JPEG's pixels may lie from those drawn: its compression
# (measured: at most 8 levels, 1.64 on average), not another drawing (another
# noise seed: 12 levels, 2.46-2.51 on average).
JPEG_MAX_ERROR, JPEG_MEAN_ERROR = 10, 1.9
# The defaults the schema-2 file leaves out and stored.json writes out, by
# path without list positions.
WRITTEN_OUT = {
    "$.batch.membranes[].calibration.ladder_kda": [],
    "$.batch.membranes[].calibration.points[].side": "left",
    "$.batch.proteins[].box_padding.across": 0,
    "$.batch.proteins[].box_padding.along": 0,
    "$.batch.proteins[].bands[].bands_found": None,
    "$.batch.proteins[].bands[].possibly_clipped": None,
    "$.batch.proteins[].undetected": [],
}


def _load(name: str):
    """A script of ``tests/data/schema2`` as a module of its own name."""
    spec = importlib.util.spec_from_file_location(f"schema2_{name}", DATA / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fixtures = _load("make_fixtures")
CASES = ("f1", "f2", "f3")


def _golden(case: str, name: str) -> dict:
    return json.loads((DATA / case / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def derived(tmp_path_factory) -> dict[str, dict]:
    """Each case's derived output from the current build, computed once."""
    work = tmp_path_factory.mktemp("schema2-derived")
    return {case: fixtures.derived_output(DATA / case, work / case) for case in CASES}


@pytest.fixture(scope="module")
def projects(tmp_path_factory) -> dict:
    """Each frozen project as the current build opens it, read once."""
    work = tmp_path_factory.mktemp("schema2-opened")
    opened = {}
    for case in CASES:
        fixtures.copy_project(DATA / case, work / case)
        session = ops.open_project(work / case, autosave=None)
        try:
            opened[case] = session.project
        finally:
            session.close()
    return opened


def _stored_differences(case: str, folder: Path) -> list[str]:
    """Where the values the project in ``folder`` stores, opened as Proteia
    opens a project, differ from ``case``'s ``stored.json``."""
    session = ops.open_project(folder, autosave=None)
    try:
        return fixtures.stored_differences(_golden(case, "stored.json"), session.project)
    finally:
        session.close()


def _paths(found: list[str]) -> list[str]:
    """Where each difference is, without what it is."""
    return [line.split(": ", 1)[0] for line in found]


def _leaves(value, path: str = "$"):
    """``(path, value)`` for each value in a JSON value that holds no other
    (an empty object or list is one), in order."""
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}")
    elif isinstance(value, list) and value:
        for i, item in enumerate(value):
            yield from _leaves(item, f"{path}[{i}]")
    else:
        yield path, value


def _intact_issues(folder: Path, stored: dict) -> list[str]:
    """What in the fixture in ``folder`` is not as its ``stored.json``
    (``stored``) records it was written: the bytes of ``project.json``, those of
    each image file, and the content hash its log ends with."""
    issues = []
    data = (folder / "project.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != stored["project_sha256"]:
        issues.append("project.json")
    images = {i["file"]: i["sha256"] for m in stored["batch"]["membranes"] for i in m["images"]}
    found = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (folder / "images").iterdir()
    }
    if found != images:
        issues.append("images")
    if json.loads(data)["log"][-1]["content_hash"] != stored["content_hash"]:
        issues.append("content hash")
    return issues


# --- The frozen files ---


def test_the_cases_are_what_the_generator_builds():
    assert fixtures.CASES == CASES
    assert fixtures.FROZEN_COMMIT == FROZEN


@pytest.mark.parametrize("case", CASES)
def test_frozen_fixtures_are_built_at_05b4af5(case):
    stored = _golden(case, "stored.json")
    assert stored["built_at"] == FROZEN
    assert stored["version"] == "0.1.0.dev0"
    assert stored["case"] == case
    project = json.loads((DATA / case / "project.json").read_text(encoding="utf-8"))
    assert project["schema_version"] == 2
    assert {entry["version"] for entry in project["log"]} == {"0.1.0.dev0"}


@pytest.mark.parametrize("case", CASES)
def test_frozen_files_are_intact(case):
    assert _intact_issues(DATA / case, _golden(case, "stored.json")) == []


def test_intact_check_reads_every_frozen_file(tmp_path):
    stored = _golden("f3", "stored.json")
    image = tmp_path / "image"
    fixtures.copy_project(DATA / "f3", image)
    last = sorted((image / "images").iterdir())[-1]
    data = last.read_bytes()
    last.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    assert _intact_issues(image, stored) == ["images"]
    extra = tmp_path / "extra"
    fixtures.copy_project(DATA / "f3", extra)
    (extra / "images" / "img-9.png").write_bytes(b"")
    assert _intact_issues(extra, stored) == ["images"]
    hashed = tmp_path / "hashed"
    fixtures.copy_project(DATA / "f3", hashed)
    path = hashed / "project.json"
    text = path.read_text(encoding="utf-8")
    # The state before the removal, and each undo back to it; the last one changes.
    assert text.count(stored["content_hash"]) == 3
    changed = text[::-1].replace(stored["content_hash"][::-1], "0" * 64, 1)[::-1]
    path.write_text(changed, encoding="utf-8", newline="\n")
    assert _intact_issues(hashed, stored) == ["project.json", "content hash"]


@pytest.mark.parametrize("case", CASES)
def test_stored_golden_matches(case, tmp_path):
    fixtures.copy_project(DATA / case, tmp_path / case)
    assert _stored_differences(case, tmp_path / case) == []


@pytest.mark.parametrize("case", CASES)
def test_stored_json_holds_every_value_of_the_frozen_file(case):
    # Every value of project.json but its schema version is in stored.json at
    # the same place; stored.json holds nothing else but the defaults the file
    # leaves out and what it records about the fixture itself.
    project = json.loads((DATA / case / "project.json").read_text(encoding="utf-8"))
    assert project.pop("schema_version") == 2
    held = dict(_leaves(_golden(case, "stored.json")))
    in_file = dict(_leaves(project))
    assert len(in_file) > 600
    assert [
        p for p, v in in_file.items() if p not in held or fixtures.differences(v, held[p])
    ] == []
    extra = {path: value for path, value in held.items() if path not in in_file}
    about_fixture = {path for path in extra if path[2:] in fixtures.FROZEN_ONLY}
    assert about_fixture == {f"$.{key}" for key in fixtures.FROZEN_ONLY}
    defaults = {re.sub(r"\[\d+\]", "[]", p): v for p, v in extra.items() if p not in about_fixture}
    assert set(defaults) <= set(WRITTEN_OUT)
    assert [p for p, v in defaults.items() if fixtures.differences(WRITTEN_OUT[p], v)] == []


def test_stored_values_cover_every_written_out_default():
    # Each default stored.json writes out is left out of some frozen file.
    seen = set()
    for case in CASES:
        project = json.loads((DATA / case / "project.json").read_text(encoding="utf-8"))
        in_file = dict(_leaves(project))
        for path, _ in _leaves(_golden(case, "stored.json")):
            if path not in in_file:
                seen.add(re.sub(r"\[\d+\]", "[]", path))
    assert set(WRITTEN_OUT) <= seen


class _OtherReason(StrEnum):
    """A not-detected reason the schema-2 build does not know."""

    OFF_ROW = "off_expected_row"


def _set(*steps):
    """A change of a project: set each ``(path, value)``, a path of attribute
    names and list positions from the project."""

    def change(project):
        for path, value in steps:
            target = project
            for step in path[:-1]:
                target = target[step] if isinstance(step, int) else getattr(target, step)
            setattr(target, path[-1], value)

    return change


def _log_entry(i: int, **update):
    """A change of a project: its log entry ``i`` with ``update``."""

    def change(project):
        log = list(project.log)
        log[i] = log[i].model_copy(update=update)
        project.log = tuple(log)

    return change


def _drop_param(i: int, name: str):
    def change(project):
        params = {k: v for k, v in project.log[i].params.items() if k != name}
        _log_entry(i, params=params)(project)

    return change


def _drop_last_entry(project):
    # The last entry before the one that converted the project on opening.
    project.log = (*project.log[:-2], project.log[-1])


# Values besides the bands and calibrations, each changed in an opened project
# as a migration could leave it: (case, change, where it is caught).
DEGRADED = {
    "undetected_reason": (
        "f1",
        _set(
            (("batch", "proteins", 3, "rows", 0, "undetected", 0, "reason"), _OtherReason.OFF_ROW)
        ),
        "$.batch.proteins[3].undetected[0].reason",
    ),
    "warning_code": (
        "f2",
        _set((("batch", "images", 1, "import_warnings", 0, "code"), "lossy")),
        "$.batch.membranes[1].images[0].import_warnings[0].code",
    ),
    "warning_message": (
        "f2",
        _set((("batch", "images", 0, "import_warnings", 0, "message"), "JPEG.")),
        "$.batch.membranes[0].images[0].import_warnings[0].message",
    ),
    "lane_index": (
        "f1",
        _set((("batch", "lane_tables", 0, "lanes", 7, "index"), 8)),
        "$.batch.lanes[7].index",
    ),
    "lane_label": (
        "f1",
        _set((("batch", "lane_tables", 0, "lanes", 0, "label"), "Vehicle")),
        "$.batch.lanes[0].label",
    ),
    "lane_sample": (
        "f1",
        _set((("batch", "lane_tables", 0, "lanes", 1, "sample"), None)),
        "$.batch.lanes[1].sample",
    ),
    "lane_included": (
        "f1",
        _set((("batch", "lane_tables", 0, "lanes", 5, "included"), True)),
        "$.batch.lanes[5].included",
    ),
    "lane_metadata": (
        "f1",
        _set((("batch", "lane_tables", 0, "lanes", 2, "metadata"), {"dose": "0"})),
        "$.batch.lanes[2].metadata.dose",
    ),
    "next_id": ("f3", _set((("next_id",), 17)), "$.next_id"),
    "log_seq": ("f3", _log_entry(19, seq=21), "$.log[19].seq"),
    "log_time": ("f3", _log_entry(19, time="2026-10-31T09:00:19.001Z"), "$.log[19].time"),
    "log_action": ("f3", _log_entry(19, action="undo"), "$.log[19].action"),
    "log_version": ("f3", _log_entry(19, version="0.1.0"), "$.log[19].version"),
    "log_params": ("f3", _drop_param(19, "returns_to_seq"), "$.log[19].params.returns_to_seq"),
    "log_content_hash": ("f3", _log_entry(19, content_hash="0" * 64), "$.log[19].content_hash"),
    "log_entry_dropped": ("f3", _drop_last_entry, "$.log"),
}


@pytest.mark.parametrize("name", DEGRADED)
def test_stored_check_catches_a_degraded_value(name, projects):
    case, change, where = DEGRADED[name]
    golden = _golden(case, "stored.json")
    project = projects[case].model_copy(deep=True)
    assert fixtures.stored_differences(golden, project) == []
    change(project)
    assert _paths(fixtures.stored_differences(golden, project)) == [where]


def _write_reshaped(folder: Path, change=None) -> None:
    """Rewrite the project in ``folder`` in another shape, with every value
    intact: each object's keys in reverse order, and each default the file
    leaves out written out; then ``change`` the JSON, if given."""

    def reversed_keys(value):
        if isinstance(value, dict):
            return {key: reversed_keys(value[key]) for key in reversed(value)}
        if isinstance(value, list):
            return [reversed_keys(item) for item in value]
        return value

    path = folder / "project.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    for membrane in doc["batch"]["membranes"]:
        membrane["calibration"].setdefault("ladder_kda", [])
        for point in membrane["calibration"]["points"]:
            point.setdefault("side", "left")
    for protein in doc["batch"]["proteins"]:
        protein.setdefault("box_padding", {"across": 0, "along": 0})
        protein.setdefault("undetected", [])
        for band in protein["bands"]:
            band.setdefault("bands_found", None)
            band.setdefault("possibly_clipped", None)
    if change is not None:
        change(doc)
    text = json.dumps(reversed_keys(doc), ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def test_stored_check_passes_a_reshaped_project_with_every_value(tmp_path, monkeypatch):
    folder = tmp_path / "f1"
    fixtures.copy_project(DATA / "f1", folder)
    _write_reshaped(folder)
    stored = _golden("f1", "stored.json")
    assert _intact_issues(folder, stored) == ["project.json"]  # not the frozen bytes
    # Held in another shape, the content hashes otherwise, as after a migration.
    monkeypatch.setattr(storage, "content_hash", lambda project: "0" * 64)
    assert _stored_differences("f1", folder) == []


def _drop(*path):
    def change(doc):
        target = doc
        for step in path[:-1]:
            target = target[step]
        del target[path[-1]]

    return change


def _none(*path):
    def change(doc):
        target = doc
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = None

    return change


# A value a migration could drop, in a reshaped project: (case, change, where
# it is caught).
DROPPED = {
    "reference_condition": (
        "f1",
        _none("batch", "reference_condition"),
        ["$.batch.reference_condition"],
    ),
    "box_padding": (
        "f1",
        _drop("batch", "proteins", 1, "box_padding"),
        ["$.batch.proteins[1].box_padding.across", "$.batch.proteins[1].box_padding.along"],
    ),
    "undetected": (
        "f1",
        _drop("batch", "proteins", 0, "undetected"),
        ["$.batch.proteins[0].undetected"],
    ),
    "marker_link": (
        "f1",
        _none("batch", "membranes", 0, "images", 0, "marker_image_id"),
        ["$.batch.membranes[0].images[0].marker_image_id"],
    ),
    "import_warnings": (
        "f2",
        _drop("batch", "membranes", 1, "images", 0, "import_warnings"),
        ["$.batch.membranes[1].images[0].import_warnings"],
    ),
    "log_param": (
        "f2",
        _drop("log", 1, "params", "original_name"),
        ["$.log[1].params.original_name"],
    ),
}


@pytest.mark.parametrize("name", DROPPED)
def test_stored_check_catches_a_dropped_value(name, tmp_path):
    case, change, where = DROPPED[name]
    folder = tmp_path / case
    fixtures.copy_project(DATA / case, folder)
    _write_reshaped(folder, change)
    assert _paths(_stored_differences(case, folder)) == where


@pytest.mark.parametrize("case", CASES)
def test_derived_golden_matches(case, derived):
    fixtures.check_derived(_golden(case, "derived.json"), derived[case], case)


def test_changes():
    changes = fixtures.changes
    assert changes({"a": [1.0, "x"]}, {"a": [1.0 + 1e-13, "x"]}) == []
    off = 1.0 + 1e-9
    assert changes({"a": [1.0, "x"]}, {"a": [off, "x"]}) == [["$.a[0]", 1.0, off]]
    assert changes({"a": [1.0, "x"]}, {"a": [1.0, "y"]}) == [["$.a[1]", "x", "y"]]
    assert changes({"a": {"b": 1}}, {"a": {"b": 1.0}}) == [["$.a.b", 1, 1.0]]
    assert changes([{"a": None}], [{"a": 0}]) == [["$[0].a", None, 0]]
    # Whole, where the shape differs.
    assert changes({"a": [1, 2]}, {"a": [1, 2, 3]}) == [["$.a", [1, 2], [1, 2, 3]]]
    assert changes({"a": 1, "b": 2}, {"a": 1, "c": 2}) == [
        ["$", {"a": 1, "b": 2}, {"a": 1, "c": 2}]
    ]
    assert changes({"a": [1]}, {"a": {"0": 1}}) == [["$.a", [1], {"0": 1}]]
    # Each change, in order.
    assert changes([1, 2, 3], [0, 2, 4]) == [["$[0]", 1, 0], ["$[2]", 3, 4]]


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> dict:
    """f1 and f3 built afresh by the current build from their frozen image
    files, built once."""
    work = tmp_path_factory.mktemp("schema2-fresh")
    return {
        case: fixtures.build(work / case, case, files=fixtures.frozen_sources(DATA / case))
        for case in ("f1", "f3")
    }


def test_fresh_changes(fresh):
    golden = _golden("f3", "stored.json")
    project = fresh["f3"].model_copy(deep=True)
    assert fixtures.fresh_changes(golden, project) == []
    # A fresh build's log entries are compared without their hashes and versions.
    _log_entry(19, content_hash="0" * 64, version="9")(project)
    assert fixtures.fresh_changes(golden, project) == []
    _log_entry(19, time="2026-10-31T09:00:19.001Z")(project)
    assert fixtures.fresh_changes(golden, project) == [
        ["$.log[19].time", "2026-10-31T09:00:19.000Z", "2026-10-31T09:00:19.001Z"]
    ]
    project = fresh["f3"].model_copy(deep=True)
    band = project.batch.proteins[1].rows[0].bands[2]
    net = band.net
    band.net = net * (1 + 1e-13)  # within the derived tolerance
    assert fixtures.fresh_changes(golden, project) == []
    band.net = net * (1 + 1e-9)
    assert fixtures.fresh_changes(golden, project) == [
        ["$.batch.proteins[1].bands[2].net", net, band.net]
    ]
    # Each is part of the derived output, so the derived check catches it.
    derived = _golden("f3", "derived.json")
    actual = {**derived, "fresh_changes": fixtures.fresh_changes(golden, project)}
    assert fixtures.derived_differences(derived, actual) == ["$.fresh_changes: 0 items != 1"]
    # What stored.json records about the fixture itself is not compared.
    assert fixtures.fresh_changes({**golden, "case": "f9", "content_hash": "0"}, project) == [
        ["$.batch.proteins[1].bands[2].net", net, band.net]
    ]


def test_derived_failure_prints_the_refresh_command(derived):
    golden = _golden("f1", "derived.json")
    degraded = copy.deepcopy(derived["f1"])
    degraded["history_issues"].append("undo_mismatch")
    with pytest.raises(AssertionError) as failed:
        fixtures.check_derived(golden, degraded, "f1")
    message = str(failed.value)
    assert fixtures.REFRESH_COMMAND in message
    assert "\n  $.history_issues: 0 items != 1\n" in message  # the golden's, then the build's
    assert fixtures.REFRESH_COMMAND == (
        "uv run python tests/data/schema2/make_fixtures.py --refresh-derived"
    )


# --- The comparison catches each degraded case (#221) ---


def _first_float(values: list) -> int:
    return next(i for i, v in enumerate(values) if isinstance(v, float))


def test_comparator_catches_stored_ulp():
    golden = _golden("f1", "stored.json")
    actual = copy.deepcopy(golden)
    band = actual["batch"]["proteins"][0]["bands"][0]
    old = golden["batch"]["proteins"][0]["bands"][0]["net"]
    band["net"] = math.nextafter(band["net"], math.inf)
    assert band["net"] != old
    found = fixtures.differences(golden, actual)
    assert found == [f"$.batch.proteins[0].bands[0].net: {old!r} != {band['net']!r}"]
    # Within the derived tolerance it would pass: exactness is what catches it.
    assert fixtures.differences(golden, actual, rel_tol=1e-12) == []


def test_stored_check_catches_a_net_one_ulp_off_in_the_file(tmp_path):
    folder = tmp_path / "f1"
    fixtures.copy_project(DATA / "f1", folder)
    path = folder / "project.json"
    text = path.read_text(encoding="utf-8")
    old = json.loads(text)["batch"]["proteins"][0]["bands"][0]["net"]
    new = math.nextafter(old, math.inf)
    assert text.count(f'"net": {old!r}') == 1
    changed = text.replace(f'"net": {old!r}', f'"net": {new!r}')
    path.write_text(changed, encoding="utf-8", newline="\n")
    assert _stored_differences("f1", folder) == [
        f"$.batch.proteins[0].bands[0].net: {old!r} != {new!r}"
    ]


def test_comparator_catches_derived_1e9(derived):
    golden = _golden("f1", "derived.json")
    actual = copy.deepcopy(derived["f1"])
    values = actual["results"]["series"][0]["normalized"]
    i = _first_float(values)
    values[i] *= 1.0 + 1e-9
    found = fixtures.derived_differences(golden, actual)
    assert [f.split(":")[0] for f in found] == [f"$.results.series[0].normalized[{i}]"]
    # A change of 1e-13 is within the tolerance.
    actual = copy.deepcopy(derived["f1"])
    actual["results"]["series"][0]["normalized"][i] *= 1.0 + 1e-13
    assert fixtures.derived_differences(golden, actual) == []
    assert fixtures.differences(golden["results"], actual["results"]) != []


def test_comparator_catches_notice_removed(derived):
    golden = _golden("f1", "derived.json")
    actual = copy.deepcopy(derived["f1"])
    removed = actual["results"]["notices"].pop(0)
    assert removed["code"]
    found = fixtures.derived_differences(golden, actual)
    assert found and all(f.startswith("$.results.notices") for f in found)


def test_comparator_catches_notice_text(derived):
    golden = _golden("f1", "derived.json")
    actual = copy.deepcopy(derived["f1"])
    notice = actual["results"]["notices"][-1]
    text = notice["message"]
    notice["message"] = text[:-1] + ("," if text[-1] != "," else ";")
    found = fixtures.derived_differences(golden, actual)
    last = len(actual["results"]["notices"]) - 1
    assert [f.split(":")[0] for f in found] == [f"$.results.notices[{last}].message"]


def test_comparator_catches_series_swap(derived):
    golden = _golden("f1", "derived.json")
    actual = copy.deepcopy(derived["f1"])
    series = actual["results"]["series"]
    assert len(series) == 2 and series[0]["target"] != series[1]["target"]
    series[0], series[1] = series[1], series[0]
    found = fixtures.derived_differences(golden, actual)
    assert any(f.startswith("$.results.series[0].target:") for f in found)
    assert any(f.startswith("$.results.series[1].target:") for f in found)


def test_comparator_catches_table_cell(derived):
    golden = _golden("f1", "derived.json")
    for change in ("number", "text"):
        actual = copy.deepcopy(derived["f1"])
        table = actual["lane_tables"][0]
        rows = fixtures.table_rows(table["csv"])
        if change == "number":
            # The first net of the first protein, one unit up in its last decimal.
            column = 4
            rows[1][column] = repr(round(float(rows[1][column]) + 0.001, 3))
        else:
            column = 2
            rows[1][column] = rows[1][column] + "x"
        table["csv"] = fixtures.table_text(rows)
        found = fixtures.derived_differences(golden, actual)
        assert [f.split(":")[0] for f in found] == [f"$.lane_tables[0].csv[1][{column}]"], change
    # Golden first, then the build's, in every line.
    old = fixtures.table_rows(golden["lane_tables"][0]["csv"])[1][2]
    assert found == [f"$.lane_tables[0].csv[1][2]: {old!r} != {old + 'x'!r}"]


def test_comparator_catches_lane_tables_missing_or_dropped(derived):
    golden = _golden("f1", "derived.json")
    actual = copy.deepcopy(derived["f1"])
    actual["lane_tables"].pop()
    assert fixtures.derived_differences(golden, actual) == ["$.lane_tables: 2 items != 1"]
    del actual["lane_tables"]
    found = fixtures.derived_differences(golden, actual)
    assert (
        len(found) == 1
        and found[0].startswith("$.lane_tables: [{")
        and found[0].endswith(" != None")
    )


def _table(csv: str, near: list | None = None, label: str | None = None) -> dict:
    return {"label": label, "csv": csv, "near_midpoint": near or []}


def test_table_cell_one_unit_off_only_at_a_midpoint():
    golden = _table("lane,a\r\n1,2.345\r\n")
    one_up = _table("lane,a\r\n1,2.346\r\n")
    two_up = _table("lane,a\r\n1,2.347\r\n")
    assert fixtures.table_differences(golden, one_up, "$") == ["$.csv[1][1]: '2.345' != '2.346'"]
    golden["near_midpoint"] = [[1, 1, 3]]
    assert fixtures.table_differences(golden, one_up, "$") == []
    assert fixtures.table_differences(one_up, _table("lane,a\r\n1,2.345\r\n", [[1, 1, 3]]), "$")
    assert (
        fixtures.table_differences(_table("lane,a\r\n1,2.346\r\n", [[1, 1, 3]]), golden, "$") == []
    )
    assert fixtures.table_differences(golden, two_up, "$") == ["$.csv[1][1]: '2.345' != '2.347'"]
    assert fixtures.table_differences(golden, _table("lane,a\r\n1,x\r\n"), "$") != []
    # One unit of a 6-decimal column is not one unit of a 3-decimal one.
    golden["near_midpoint"] = [[1, 1, 6]]
    assert fixtures.table_differences(golden, one_up, "$") != []
    # Only the cell marked.
    golden["near_midpoint"] = [[1, 0, 3]]
    assert fixtures.table_differences(golden, one_up, "$") != []
    # Rows and columns must match in number, and the label exactly.
    longer = _table("lane,a\r\n1,2.345\r\n2,1.0\r\n")
    assert fixtures.table_differences(golden, longer, "$") == ["$.csv: 2 rows != 3"]
    wider = _table("lane,a,b\r\n1,2.345,\r\n")
    assert fixtures.table_differences(golden, wider, "$") == [
        "$.csv[0]: 2 cells != 3",
        "$.csv[1]: 2 cells != 3",
    ]
    labelled = _table("lane,a\r\n1,2.345\r\n", label="All lanes")
    assert fixtures.table_differences(golden, labelled, "$") == ["$.label: None != 'All lanes'"]


def test_near_midpoint():
    assert fixtures.near_midpoint(1.2345, 3)  # 1234.5 on a 3-decimal grid
    assert not fixtures.near_midpoint(1.2346, 3)
    assert not fixtures.near_midpoint(1.2345, 6)
    assert fixtures.near_midpoint(0.000015, 5)
    assert not fixtures.near_midpoint(0.0000125, 5)
    assert fixtures.near_midpoint(-2.5, 0)
    assert not fixtures.near_midpoint(2.5 + 1e-9, 0)
    assert fixtures.near_midpoint(2.5, 0, rel=0.0)  # on the midpoint itself
    assert fixtures.near_midpoint(2.5 * (1 + 0.9e-12), 0)
    assert not fixtures.near_midpoint(2.5 * (1 + 1.1e-12), 0)


def test_midpoint_cells():
    text = "lane,net,ratio\r\n1,2.345,0.25\r\n2,2.346,0.25\r\n"
    unrounded = {
        3: "lane,net,ratio\r\n1,2.3445,0.25\r\n2,2.3461,0.25\r\n",
        6: "lane,net,ratio\r\n1,2.345,0.2500004999999\r\n2,2.346,0.25000049\r\n",
    }
    assert fixtures.midpoint_cells(text, unrounded) == [[1, 1, 3], [1, 2, 6]]
    assert fixtures.midpoint_cells(text, {3: text}) == []


def test_comparator_is_strict_about_types_keys_and_order():
    d = fixtures.differences
    assert fixtures.REL_TOL == 1e-12
    assert d({"a": 1}, {"a": 1}) == []
    assert d({"a": 1}, {"a": 1.0}) == ["$.a: 1 != 1.0"]
    assert d({"a": 1.0}, {"a": 1}) != []
    assert d({"a": True}, {"a": 1}) != []
    assert d({"a": None}, {"a": 0}) != []
    assert d({"a": "β"}, {"a": "β "}) != []
    assert d({"a": 1}, {"a": 1, "b": 2}) == ["$.b: not in the golden"]
    assert d({"a": 1, "b": 2}, {"a": 1}) == ["$.b: missing"]
    assert d([{"a": 1}], [{"a": 1, "b": 2}]) == ["$[0].b: not in the golden"]
    assert d([1, 2], [2, 1]) == ["$[0]: 1 != 2", "$[1]: 2 != 1"]
    assert d([1, 2], [1, 2, 3]) == ["$: 2 items != 3"]
    assert d(0.0, -0.0) == ["$: 0.0 != -0.0"]
    # Floats: exact unless a tolerance is given, and then relative only.
    assert d(1.0, 1.0 + 1e-15) != []
    assert d(1.0, 1.0 + 1e-15, rel_tol=1e-12) == []
    assert d(0.0, 1e-300, rel_tol=1e-12) != []
    assert d(1e6, 1e6 * (1 + 2e-12), rel_tol=1e-12) != []
    # The tolerance reaches floats nested anywhere.
    assert d({"x": [[1.0]]}, {"x": [[1.0 + 1e-14]]}, rel_tol=1e-12) == []
    assert d({"x": [[1.0]]}, {"x": [[1.0 + 1e-14]]}) == ["$.x[0][0]: 1.0 != 1.00000000000001"]
    # The derived rule: 1.2e-12 relative is too far.
    assert fixtures.derived_differences({"x": 1.0}, {"x": 1.0 + 1.2e-12}) != []
    assert fixtures.derived_differences({"x": 1.0}, {"x": 1.0 + 0.8e-12}) == []


def test_derived_failure_lists_at_most_40_differences():
    with pytest.raises(AssertionError) as failed:
        fixtures.check_derived({"x": [0] * 45}, {"x": [1] * 45}, "f9")
    lines = str(failed.value).splitlines()
    assert lines[0] == "the derived output of f9 differs from f9/derived.json:"
    assert lines[1] == "  $.x[0]: 0 != 1"
    assert lines[40] == "  $.x[39]: 0 != 1"
    assert lines[41] == "  ... and 5 more"
    with pytest.raises(AssertionError) as failed:
        fixtures.check_derived({"x": [0] * 41}, {"x": [1] * 41}, "f9")
    assert str(failed.value).splitlines()[40:42] == ["  $.x[39]: 0 != 1", "  ... and 1 more"]
    with pytest.raises(AssertionError) as failed:
        fixtures.check_derived({"x": [0] * 40}, {"x": [1] * 40}, "f9")
    assert "more" not in str(failed.value)


def test_canonical_json():
    value = {"b": [1, 2.5, None, "µ", True], "a": {"c": [{"d": 0.1}], "e": {}}, "f": []}
    assert fixtures.canonical_json(value).decode("utf-8") == (
        "{\n"
        ' "a": {\n'
        '  "c": [\n'
        "   {\n"
        '    "d": 0.1\n'
        "   }\n"
        "  ],\n"
        '  "e": {}\n'
        " },\n"
        ' "b": [1, 2.5, null, "µ", true],\n'
        ' "f": []\n'
        "}\n"
    )
    with pytest.raises(TypeError, match="JSON keys are text"):
        fixtures.canonical_json({"a": {1: 2}})
    for bad in ({"a": {1, 2}}, [b"x"]):
        with pytest.raises(TypeError):
            fixtures.canonical_json(bad)
    for bad in ({"a": float("nan")}, [float("inf")]):
        with pytest.raises(ValueError):
            fixtures.canonical_json(bad)


@pytest.mark.parametrize("case", CASES)
def test_goldens_are_canonical(case):
    for name in ("stored.json", "derived.json"):
        data = (DATA / case / name).read_bytes()
        assert data == fixtures.canonical_json(json.loads(data)), name


# --- The commands ---


def test_refresh_derived_leaves_frozen_files(tmp_path, monkeypatch, capsys, derived):
    root = tmp_path / "schema2"
    for case in CASES:
        fixtures.copy_project(DATA / case, root / case)
        for name in ("stored.json", "derived.json"):
            (root / case / name).write_bytes((DATA / case / name).read_bytes())
    (root / "f1" / "derived.json").write_text("{}\n", encoding="utf-8")  # stale: rewritten

    def listing() -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "folder"
            )
            for path in sorted(root.rglob("*"))
        }

    before = listing()
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))  # the copies it makes go here
    assert fixtures.main(["--refresh-derived", "--out", str(root)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"refreshed {case}/derived.json" for case in CASES
    ]
    assert list(temp.iterdir()) == []  # and are gone
    after = listing()
    assert set(after) == set(before)
    changed = {name for name in after if after[name] != before[name]}
    assert changed <= {f"{case}/derived.json" for case in CASES}
    assert "f1/derived.json" in changed
    for case in CASES:
        # What the module's derived output gives: the same build, one machine.
        written = (root / case / "derived.json").read_bytes()
        assert written == fixtures.canonical_json(derived[case])


def test_derived_output_closes_the_project_it_opens(tmp_path, monkeypatch):
    opened, closed = [], []
    real_open, real_close = ops.open_project, ops.ProjectSession.close

    def open_project(*args, **kwargs):
        opened.append(real_open(*args, **kwargs))
        return opened[-1]

    def close(self, *args, **kwargs):
        closed.append(self)
        return real_close(self, *args, **kwargs)

    def fail(*args, **kwargs):
        raise RuntimeError("stopped")

    monkeypatch.setattr(ops, "open_project", open_project)
    monkeypatch.setattr(ops.ProjectSession, "close", close)
    # Stopped while the copy is open, and after it is read (at the fresh build).
    for module, name in ((fixtures.results, "compute_results"), (fixtures, "build")):
        with monkeypatch.context() as patched:
            patched.setattr(module, name, fail)
            with pytest.raises(RuntimeError, match="stopped"):
                fixtures.derived_output(DATA / "f3", tmp_path / name)
    assert len(opened) == 2 and closed == opened


def test_frozen_refusal():
    toplevel = Path("checkout")
    module = toplevel / "src" / "proteia" / "__init__.py"
    ok = {"head": FROZEN, "changes": "", "module": module, "toplevel": toplevel}
    assert fixtures.frozen_refusal(**ok) is None
    assert fixtures.frozen_refusal(**ok, mismatched=[]) is None
    other = fixtures.frozen_refusal(**{**ok, "head": "1" * 40})
    assert other is not None and FROZEN in other and "1" * 40 in other
    changed = fixtures.frozen_refusal(**{**ok, "changes": " M src/proteia/core/model.py\n"})
    assert changed is not None and "src/proteia/core/model.py" in changed
    elsewhere = Path("elsewhere") / "proteia" / "__init__.py"
    imported = fixtures.frozen_refusal(**{**ok, "module": elsewhere})
    assert imported is not None and "elsewhere" in imported
    lines = ["numpy: locked 2.2.6, installed 2.3.4", "scipy: locked 1.16.0, installed nowhere"]
    mismatched = fixtures.frozen_refusal(**ok, mismatched=lines)
    assert mismatched == (
        "the installed dependencies are not those uv.lock locks (run uv sync --locked):\n"
        "  numpy: locked 2.2.6, installed 2.3.4\n"
        "  scipy: locked 1.16.0, installed nowhere"
    )
    # The commit is named first.
    assert fixtures.frozen_refusal(**{**ok, "head": "1" * 40}, mismatched=lines) == other


LOCK = """\
version = 1

[[package]]
name = "app"
version = "1.0"
source = { editable = "." }
dependencies = [
    { name = "click" },
    { name = "numpy" },
    { name = "uvicorn", extra = ["standard"] },
]

[package.dev-dependencies]
dev = [
    { name = "pytest" },
]

[[package]]
name = "click"
version = "8.2.1"
dependencies = [
    { name = "colorama", marker = "sys_platform == 'win32'" },
]

[[package]]
name = "colorama"
version = "0.4.6"

[[package]]
name = "numpy"
version = "2.2.6"
resolution-markers = ["python_full_version < '3.14'"]

[[package]]
name = "numpy"
version = "2.3.4"
resolution-markers = ["python_full_version >= '3.14'"]

[[package]]
name = "pytest"
version = "8.4.0"

[[package]]
name = "typing-extensions"
version = "4.15.0"

[[package]]
name = "uvicorn"
version = "0.54.0"
dependencies = [
    { name = "click" },
]

[package.optional-dependencies]
standard = [
    { name = "typing-extensions" },
]
"""
# Installed as the lock asks, colorama (a marker's) left out, names as
# distributions spell them.
INSTALLED = {
    "Click": "8.2.1",
    "NumPy": "2.2.6",
    "uvicorn": "0.54.0",
    "typing_extensions": "4.15.0",
    "pytest": "8.4.0",
    "other": "1",
}


def test_lock_mismatches():
    def found(lock=LOCK, **versions):
        installed = {k: v for k, v in {**INSTALLED, **versions}.items() if v is not None}
        return fixtures.lock_mismatches(lock, installed, project="app")

    assert found() == []
    assert found(NumPy="2.3.4") == []  # either version the lock holds
    assert found(NumPy="2.4.0") == ["numpy: locked 2.2.6 or 2.3.4, installed 2.4.0"]
    assert found(Click=None) == ["click: locked 8.2.1, installed nowhere"]
    # Through a dependency, and through an extra it asks for.
    assert found(typing_extensions="4.14.0") == [
        "typing-extensions: locked 4.15.0, installed 4.14.0"
    ]
    # Asked for by a marker only: may be missing, but not at another version.
    assert found(colorama="0.4.5") == ["colorama: locked 0.4.6, installed 0.4.5"]
    assert found(colorama="0.4.6") == []
    unmarked = LOCK.replace(
        '    { name = "typing-extensions" },\n]',
        '    { name = "typing-extensions" },\n    { name = "colorama" },\n]',
    )
    assert unmarked != LOCK
    assert found(unmarked) == ["colorama: locked 0.4.6, installed nowhere"]
    # A cycle (click and uvicorn asking for each other) is followed once.
    cyclic = LOCK.replace(
        'dependencies = [\n    { name = "colorama", marker',
        'dependencies = [\n    { name = "uvicorn" },\n    { name = "colorama", marker',
    )
    assert cyclic != LOCK
    assert found(cyclic) == []
    assert found(cyclic, uvicorn="0.53.0") == ["uvicorn: locked 0.54.0, installed 0.53.0"]
    # A dependency group is not the runtime's.
    assert found(pytest="9.0.0") == []
    assert found(Click="8.2.0", NumPy="1.0") == [
        "click: locked 8.2.1, installed 8.2.0",
        "numpy: locked 2.2.6 or 2.3.4, installed 1.0",
    ]
    assert fixtures.lock_mismatches(LOCK, INSTALLED, project="other") == ["other: not in uv.lock"]
    # A dependency the lock does not hold.
    broken = LOCK.replace('{ name = "click" },\n    { name = "numpy" }', '{ name = "gone" }')
    assert broken != LOCK
    assert found(broken) == ["gone: locked nowhere, installed nowhere"]


def test_this_checkout_runs_on_its_locked_dependencies():
    # The real lock, with its markers, read without a false refusal.
    checkout = Path(fixtures.proteia.__file__).resolve().parents[2]
    lock = (checkout / "uv.lock").read_text(encoding="utf-8")
    assert fixtures.lock_mismatches(lock, fixtures.installed_versions()) == []
    assert {"numpy", "scipy", "pillow", "tifffile", "pydantic"} <= set(
        fixtures.installed_versions()
    )


def test_refusal_here_checks_the_installed_dependencies(tmp_path, monkeypatch):
    checkout = Path(fixtures.proteia.__file__).resolve().parents[2]
    answers = {
        ("rev-parse", "--show-toplevel"): f"{checkout}\n",
        ("rev-parse", "HEAD"): f"{FROZEN}\n",
        ("status", "--porcelain"): "",
    }
    monkeypatch.setattr(fixtures, "_git", lambda *args: answers[args[:2]])
    assert fixtures._refusal_here() is None  # as at 05b4af5, on the locked dependencies
    installed = {**fixtures.installed_versions(), "numpy": "0.0.1"}
    monkeypatch.setattr(fixtures, "installed_versions", lambda: installed)
    refusal = fixtures._refusal_here()
    assert refusal is not None
    assert refusal.startswith("the installed dependencies are not those uv.lock locks")
    assert refusal.endswith(", installed 0.0.1") and "\n  numpy: locked " in refusal
    answers[("rev-parse", "--show-toplevel")] = f"{tmp_path}\n"  # no uv.lock there
    refusal = fixtures._refusal_here()
    assert refusal is not None
    assert refusal.startswith(f"cannot tell whether this checkout is at {FROZEN}: ")
    assert "uv.lock" in refusal


def _head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=DATA,
            capture_output=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def test_write_frozen_refuses_at_this_commit(tmp_path, capsys):
    # This file exists only after 05b4af5, so this checkout is never at it;
    # where git cannot name the commit, it cannot tell, and refuses too.
    head = _head()
    out = tmp_path / "out"
    assert fixtures.main(["--write-frozen", "--out", str(out)]) == 2
    assert not out.exists()
    err = capsys.readouterr().err
    if head is None:
        assert err.startswith(f"refused: cannot tell whether this checkout is at {FROZEN}: ")
    else:
        assert err.startswith(f"refused: this checkout is at {head}, not at {FROZEN}: ")
    # And as a script.
    script = DATA / "make_fixtures.py"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(
        [sys.executable, str(script), "--write-frozen", "--out", str(out)],
        capture_output=True,
        encoding="utf-8",
        env=env,
    )
    assert done.returncode == 2 and done.stderr.startswith("refused: ")
    assert not out.exists()


def test_write_frozen_writes_a_case_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "_refusal_here", lambda: None)  # as if at 05b4af5
    (tmp_path / "f2").mkdir()
    with pytest.raises(fixtures.Refused, match="^f2 exist in "):
        fixtures.write_frozen(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f2"]


def test_main_leaves_the_console_alone(tmp_path, monkeypatch):
    # Only the scripts, run as such, let the console print any text; main()
    # called in a process of its own (this one) leaves its streams alone.
    class Console(io.StringIO):
        def reconfigure(self, **kwargs):
            raise AssertionError(f"reconfigured with {kwargs}")

    monkeypatch.setattr(sys, "stdout", Console())
    monkeypatch.setattr(sys, "stderr", Console())
    monkeypatch.setattr(fixtures, "_refusal_here", lambda: "not here, µ")
    assert fixtures.main(["--write-frozen", "--out", str(tmp_path / "out")]) == 2
    assert sys.stderr.getvalue() == "refused: not here, µ\n"
    monkeypatch.syspath_prepend(str(DATA))  # as when it runs as a script
    assert _load("derive").main([str(tmp_path)]) == 2
    assert "derive.py: cannot read the case of " in sys.stderr.getvalue()


def test_write_frozen_writes_every_case(tmp_path, monkeypatch, capsys, derived):
    # As at 05b4af5 (a stand-in commit), from the frozen image files: what it
    # writes holds the frozen values, as far as this platform's floats allow.
    monkeypatch.setattr(fixtures, "_refusal_here", lambda: None)
    real_git = fixtures._git
    monkeypatch.setattr(
        fixtures,
        "_git",
        lambda *args: "f" * 40 + "\n" if args == ("rev-parse", "HEAD") else real_git(*args),
    )
    monkeypatch.setattr(fixtures, "sources", lambda case: fixtures.frozen_sources(DATA / case))
    # The derived output of the same build of the same image files, computed
    # by the module already (one build less per case); the build written is
    # compared with another of its own, which stores what it stores.
    asked = []

    def derived_output(folder, work, case=None):
        asked.append((Path(folder), case, (Path(folder) / "stored.json").is_file()))
        return {**copy.deepcopy(derived[case]), "fresh_changes": []}

    monkeypatch.setattr(fixtures, "derived_output", derived_output)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    assert fixtures.main(["--write-frozen", "--out", str(out)]) == 0
    assert capsys.readouterr().out.splitlines() == [f"wrote {case}" for case in CASES]
    assert asked == [(out / case, case, True) for case in CASES]
    for case in CASES:
        stored = json.loads((out / case / "stored.json").read_text(encoding="utf-8"))
        project = (out / case / "project.json").read_bytes()
        assert stored["case"] == case
        assert stored["project_sha256"] == hashlib.sha256(project).hexdigest()
        assert stored["content_hash"] == json.loads(project)["log"][-1]["content_hash"]
        frozen = _golden(case, "stored.json")
        assert stored.pop("built_at") == "f" * 40 and frozen.pop("built_at") == FROZEN
        # Each hash changes whole with the last bit of any float.
        for values in (stored, frozen):
            del values["project_sha256"], values["content_hash"]
            for entry in values["log"]:
                del entry["content_hash"]
        # Every value but those a fresh build of this one stores otherwise
        # (fresh_changes: since #206, F2's colour image's pick).
        fresh = [path for path, _, _ in derived[case]["fresh_changes"]]
        assert [path for path, _, _ in fixtures.changes(frozen, stored)] == fresh
        first = {**derived[case], "fresh_changes": []}
        assert (out / case / "derived.json").read_bytes() == fixtures.canonical_json(first)
        # Named by the ids this build drew, read as the schema-2 build drew them.
        new = fixtures.new_numbers(storage.load_project(out / case))
        written = [p.name for p in (out / case / "images").iterdir()]
        assert sorted(fixtures.renumbered(written, new)) == sorted(
            p.name for p in (DATA / case / "images").iterdir()
        )
    assert list(tmp_path.iterdir()) == [out]  # its temporary folder is gone


def test_fixtures_stay_under_400_kb():
    # Every file of the three cases: the projects, their images and goldens.
    files = [p for case in CASES for p in (DATA / case).rglob("*") if p.is_file()]
    assert {p.name for p in files} >= {"project.json", "stored.json", "derived.json"}
    assert {p.suffix for p in files} == {".json", ".tif", ".png", ".jpg"}
    assert sum(p.stat().st_size for p in files) < 400_000


@pytest.mark.parametrize("case", CASES)
def test_generator_draws_the_frozen_images(case):
    # The lossless files hold the pixels drawn; the JPEG holds them up to its
    # compression.
    drawn = fixtures.drawn(case)
    frozen = fixtures.frozen_sources(DATA / case)
    assert set(drawn) == set(frozen)
    for name, pixels in drawn.items():
        stored = fixtures.decoded(frozen[name])
        assert (stored.shape, stored.dtype) == (pixels.shape, pixels.dtype), name
        if name.endswith(".jpg"):
            error = np.abs(stored.astype(int) - pixels.astype(int))
            assert error.max() <= JPEG_MAX_ERROR and error.mean() < JPEG_MEAN_ERROR, name
        else:
            assert np.array_equal(stored, pixels), name
    with pytest.raises(ValueError, match="unknown case 'f9'"):
        fixtures.drawn("f9")


def test_generator_writes_the_formats_named():
    magic = {".tif": "49492a00", ".png": "89504e470d0a1a0a", ".jpg": "ffd8ff"}
    for case in CASES:
        drawn = fixtures.drawn(case)
        for name, data in fixtures.sources(case).items():
            suffix = name[-4:]
            assert data.startswith(bytes.fromhex(magic[suffix])), name
            if suffix != ".jpg":
                assert np.array_equal(fixtures.decoded(data), drawn[name]), name


def test_build_imports_the_files_given(tmp_path, monkeypatch):
    files = fixtures.frozen_sources(DATA / "f3")
    marker = fixtures.decoded(files[fixtures.N])
    files[fixtures.N] = fixtures._encoded(marker, "PNG", compress_level=1)  # same pixels
    assert files[fixtures.N] != fixtures.frozen_sources(DATA / "f3")[fixtures.N]
    built, closed = [], []
    real_new, real_close = ops.new_project, ops.ProjectSession.close

    def new_project(*args, **kwargs):
        built.append(real_new(*args, **kwargs))
        return built[-1]

    def close(self, *args, **kwargs):
        closed.append(self)
        return real_close(self, *args, **kwargs)

    monkeypatch.setattr(ops, "new_project", new_project)
    monkeypatch.setattr(ops.ProjectSession, "close", close)
    project = fixtures.build(tmp_path / "f3", "f3", files=files)
    assert len(built) == 1 and closed == built  # and it closes the project it builds
    stored = {image.original_name: image.sha256 for image in project.batch.iter_images()}
    assert stored == {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def test_main_needs_a_command(capsys):
    with pytest.raises(SystemExit) as stopped:
        fixtures.main([])
    assert stopped.value.code == 2
    assert "one of the arguments --write-frozen --refresh-derived is required" in (
        capsys.readouterr().err
    )


def _copy_fixture(case: str, folder: Path) -> Path:
    """A copy of ``case``'s fixture (its project and ``stored.json``) in
    ``folder``, named anything."""
    fixtures.copy_project(DATA / case, folder)
    (folder / "stored.json").write_bytes((DATA / case / "stored.json").read_bytes())
    return folder


def test_derive_writes_a_case(tmp_path, monkeypatch, capsysbinary):
    monkeypatch.syspath_prepend(str(DATA))  # as when it runs as a script
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    derive = _load("derive")
    asked = []

    def derived_output(folder, work, case=None):
        asked.append((Path(folder), case, Path(work).parent == tmp_path))
        return {"case": case, "µ": [0.1]}

    monkeypatch.setattr(derive.make_fixtures, "derived_output", derived_output)
    out = tmp_path / "f3.json"
    assert derive.main(["f3", "--out", str(out)]) == 0  # a case, by its name
    assert out.read_bytes() == '{\n "case": "f3",\n "µ": [0.1]\n}\n'.encode()
    assert capsysbinary.readouterr().out == b""
    # A fixture folder named otherwise: the case its stored.json names, to stdout.
    folder = _copy_fixture("f2", tmp_path / "copy of f3")
    assert derive.main([str(folder)]) == 0
    assert capsysbinary.readouterr().out == '{\n "case": "f2",\n "µ": [0.1]\n}\n'.encode()
    assert asked == [(DATA / "f3", "f3", True), (folder.resolve(), "f2", True)]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["copy of f3", "f3.json"]


def test_derive_refuses_a_folder_that_names_no_case(tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(DATA))
    derive = _load("derive")
    monkeypatch.setattr(derive.make_fixtures, "derived_output", None)  # never reached
    fixtures.copy_project(DATA / "f3", tmp_path / "f3")  # a project, no stored.json
    folder = (tmp_path / "f3").resolve()
    assert derive.main([str(folder)]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"derive.py: cannot read the case of {folder}: ")
    for text, named in (("[]", None), ('{"case": "f9"}', "'f9'"), ('{"case": ["f1"]}', "['f1']")):
        (folder / "stored.json").write_text(text, encoding="utf-8")
        assert derive.main([str(folder)]) == 2
        assert capsys.readouterr().err == (
            f"derive.py: {folder / 'stored.json'} names no case of f1, f2, f3: {named}\n"
        )


def test_derive_script_prints_canonical_json(tmp_path, derived):
    # As a script, on a fixture folder named otherwise.
    temp = tmp_path / "temp"
    temp.mkdir()
    env = {**os.environ, "TMP": str(temp), "TEMP": str(temp), "TMPDIR": str(temp)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    folder = _copy_fixture("f2", tmp_path / "fixture")
    script = DATA / "derive.py"
    printed = subprocess.run(
        [sys.executable, str(script), str(folder)], check=True, env=env, capture_output=True
    ).stdout
    assert printed == fixtures.canonical_json(derived["f2"])
    assert list(temp.iterdir()) == []  # its copies are gone


# --- What each project holds (the issue's list) ---


def _project(case: str) -> dict:
    return json.loads((DATA / case / "project.json").read_text(encoding="utf-8"))


def _notice_codes(case: str) -> set[str]:
    results = _golden(case, "derived.json")["results"]
    return {n["code"] for n in results["notices"]}


def test_f1_holds_two_membranes_a_marker_two_groups_two_ladders_and_strip_edges():
    batch = _project("f1")["batch"]
    a, b = batch["membranes"]
    kinds = [
        (i["original_name"][:2], i["kind"], i["bit_depth"], i["polarity"]) for i in a["images"]
    ]
    assert kinds == [
        ("E1", "chemiluminescence", 16, "dark_on_light"),
        ("E2", "chemiluminescence", 16, "dark_on_light"),
        ("M ", "visible_marker", 8, "dark_on_light"),
    ]
    e1, e2, m = a["images"]
    assert e1["marker_image_id"] == m["id"] and e2["marker_image_id"] is None
    assert [i["file"][-4:] for i in a["images"]] == [".tif", ".tif", ".png"]
    points = a["calibration"]["points"]
    by = {(p["image_id"], p["source"], p.get("side", "left")) for p in points}
    assert by == {
        (m["id"], "visible_marker", "left"),
        (m["id"], "visible_marker", "right"),
        (e2["id"], "chemiluminescence_marker", "left"),
    }
    assert sum(p["image_id"] == m["id"] for p in points) == 6
    assert a["calibration"]["ladder"] == "pageruler_plus/tris_glycine"
    (e3,) = b["images"]
    assert e3["polarity"] == "light_on_dark" and e3["bit_depth"] == 16
    assert (e3["width"], e3["height"]) != (e1["width"], e1["height"])
    assert [p["source"] for p in b["calibration"]["points"]] == ["strip_edge", "strip_edge"]
    calibration = _golden("f1", "derived.json")["calibration"]
    assert [len(m["groups"]) for m in calibration] == [2, 1]
    assert calibration[0]["groups"][0]["image_ids"] == [e1["id"], m["id"]]
    two_ladders = calibration[0]["groups"][0]["fit"]
    assert [ladder["side"] for ladder in two_ladders["ladders"]] == ["left", "right"]
    assert two_ladders["tilt_deg"] > 0
    assert calibration[0]["groups"][1]["fit"]["ladders"][0]["two_point"] is True


def test_f1_holds_its_lanes_and_four_ways_of_placing_boxes():
    batch = _project("f1")["batch"]
    lanes = [(x["label"], x["sample"], x["included"]) for x in batch["lanes"]]
    assert lanes == [
        ("vehicle", "V1", True),
        ("vehicle", "V1", True),
        ("vehicle", "V2", True),
        ("10 µM", "A1", True),
        ("10 µM", "A2", True),
        ("10 µM", "A3", False),
        ("50 µM", "B1", True),
        ("50 µM", "B2", True),
    ]
    assert batch["reference_condition"] == "vehicle"
    proteins = {p["name"]: p for p in batch["proteins"]}
    assert list(proteins) == ["β-catenin", "α-tubulin", "GAPDH", "p53 µ"]
    images = {i["original_name"][:2]: i["id"] for m in batch["membranes"] for i in m["images"]}
    on = {name: p["image_id"] for name, p in proteins.items()}
    assert on == {
        "β-catenin": images["E1"],
        "α-tubulin": images["E3"],
        "GAPDH": images["E1"],
        "p53 µ": images["E2"],
    }
    sources = {name: {b["source"] for b in p["bands"]} for name, p in proteins.items()}
    assert sources == {
        "β-catenin": {"row_box"},
        "α-tubulin": {"click"},
        "GAPDH": {"manual"},
        "p53 µ": {"mw_guided"},
    }
    assert all("bands_found" in b for b in proteins["β-catenin"]["bands"])
    assert [u["source"] for u in proteins["β-catenin"]["undetected"]] == ["row_box"]
    assert [u["source"] for u in proteins["p53 µ"]["undetected"]] == ["mw_guided"]
    assert proteins["α-tubulin"]["box_padding"] == {"across": 2, "along": 1}
    assert [b["clipped"] for b in proteins["GAPDH"]["bands"]].count(True) == 1
    ids = {name: p["id"] for name, p in proteins.items()}
    assert proteins["β-catenin"]["loading_control_ids"] == [ids["α-tubulin"]]
    assert proteins["p53 µ"]["loading_control_ids"] == [ids["GAPDH"]]
    assert proteins["β-catenin"]["expected_mw"] == 92.0
    assert proteins["p53 µ"]["expected_mw"] == 53.0
    results = _golden("f1", "derived.json")["results"]
    assert results["all_lanes"] is not None  # the excluded lane holds values
    assert len(_golden("f1", "derived.json")["lane_tables"]) == 2
    assert {"technical_repeats", "clipped", "below_detection", "calibration_two_points"} <= (
        _notice_codes("f1")
    )


def test_f2_holds_a_lossy_and_a_colour_image_with_overlapping_boxes():
    batch = _project("f2")["batch"]
    images = [i for m in batch["membranes"] for i in m["images"]]
    assert [i["file"][-4:] for i in images] == [".jpg", ".png"]
    warnings = [[w["code"] for w in i["import_warnings"]] for i in images]
    assert "lossy_format" in warnings[0] and "color_channels_differ" in warnings[1]
    bands = {p["name"]: p["bands"] for p in batch["proteins"]}
    assert any(b.get("possibly_clipped") for b in bands["β-catenin"])
    assert {"boxes_overlap", "possibly_clipped", "clipping_not_checked"} <= _notice_codes("f2")


def test_f3_holds_an_undo_and_redo_history_with_a_removed_image():
    project = _project("f3")
    log = project["log"]
    assert [entry["action"] for entry in log[-4:]] == ["remove_image", "undo", "redo", "undo"]
    removal, undo, redo, last = log[-4:]
    before = log[-5]
    assert undo["params"]["returns_to_seq"] == before["seq"]
    assert redo["params"]["returns_to_seq"] == removal["seq"]
    assert last["params"]["returns_to_seq"] == before["seq"]
    assert last["content_hash"] == before["content_hash"] != removal["content_hash"]
    assert _golden("f3", "stored.json")["content_hash"] == before["content_hash"]
    assert _golden("f3", "derived.json")["history_issues"] == []
    # The removal took the marker link, the ladder and the apparent MWs along;
    # the last undo brought them back.
    batch = project["batch"]
    signal, marker = batch["membranes"][0]["images"]
    assert signal["marker_image_id"] == marker["id"]
    assert removal["params"]["image_id"] == marker["id"]
    assert len(batch["membranes"][0]["calibration"]["points"]) == 3
    beta = batch["proteins"][0]
    assert beta["name"] == "β-catenin" and all(b["apparent_mw"] for b in beta["bands"])


# --- The conversion to schema 3 (#223) ---


@pytest.mark.parametrize("case", CASES)
def test_the_conversion_maps_back_to_the_frozen_content(case, projects):
    # Every stored value and its order survived: the converted content, read
    # back as schema 2 held it, hashes as the frozen file's content did.
    content = storage.content_document(projects[case])
    earlier = storage._EARLIER_CONTENT[2](content)
    assert storage.document_hash(earlier) == _golden(case, "stored.json")["content_hash"]


@pytest.mark.parametrize("case", CASES)
def test_the_conversion_is_logged_with_its_backup(case, tmp_path):
    folder = tmp_path / case
    fixtures.copy_project(DATA / case, folder)
    frozen = _golden(case, "stored.json")
    session = ops.open_project(folder, autosave=None)
    try:
        project = session.project
        assert fixtures.migration_issues(frozen, project) == []
        entry = project.log[-1]
        assert entry.params == {
            "from_schema": 2,
            "to_schema": 3,
            "from_content_hash": frozen["log"][-1]["content_hash"],
            "backup": "project.schema2.json",
        }
        assert record.history_issues(project) == []
    finally:
        session.close()
    assert (folder / "project.schema2.json").read_bytes() == (
        DATA / case / "project.json"
    ).read_bytes()


def test_the_conversion_writes_no_outline_and_its_ids_follow_next_id(projects):
    project = projects["f1"]
    frozen = _golden("f1", "stored.json")
    batch = project.batch
    assert all(a.outline is None for m in batch.membranes for a in m.arenas)
    content = storage.content_document(project)
    assert all("outline" not in a for m in content["batch"]["membranes"] for a in m["arenas"])
    # After every id the frozen file held: the table, the areas, then the rows.
    first = frozen["next_id"]
    drawn = [i for i in project.iter_ids() if i.split("-")[0] in fixtures.NEW_KINDS]
    assert drawn == [
        f"lt-{first}",
        *(f"arena-{first + 1 + i}" for i in range(4)),
        *(f"row-{first + 5 + i}" for i in range(4)),
    ]
    assert project.next_id == first + 9


def _migrate_entry(project, **params):
    """``project`` with its last log entry's params updated."""
    log = list(project.log)
    log[-1] = log[-1].model_copy(update={"params": {**log[-1].params, **params}})
    return project.model_copy(update={"log": tuple(log)})


def test_migration_issues_catch_a_wrong_conversion_entry(projects):
    frozen = _golden("f3", "stored.json")
    project = projects["f3"]
    assert fixtures.migration_issues(frozen, project) == []
    wrong = _migrate_entry(project, from_content_hash="0" * 64)
    assert fixtures.migration_issues(frozen, wrong) == [
        f"from_content_hash: {'0' * 64!r} != {frozen['content_hash']!r}"
    ]
    assert fixtures.migration_issues(frozen, _migrate_entry(project, from_schema=1)) == [
        "from_schema: 1 != 2"
    ]
    log = list(project.log)
    log[-1] = log[-1].model_copy(update={"content_hash": "0" * 64})
    stale = project.model_copy(update={"log": tuple(log)})
    assert fixtures.migration_issues(frozen, stale) == ["content_hash: not the converted project's"]
    dropped = project.model_copy(update={"log": project.log[:-1]})
    assert fixtures.migration_issues(frozen, dropped) == [
        "the log does not end with the conversion to schema 3"
    ]


# Each rule that reads a schema-3 project back as schema 2, degraded: (rule,
# the degraded function, the cases a stored or fresh comparison then fails on).
def _off_by_one_renumbering(value, new):
    return _real_renumbered(value, new[1:])  # one new number too few: ids one too high


def _over_dropped_ids(entries):
    # Drops a band id from every list too (an undo that brought one back).
    dropped = _real_without_new_ids(entries)
    return [
        {
            **entry,
            "params": {
                key: [v for v in value if not str(v).startswith("band-")]
                if isinstance(value, list)
                else value
                for key, value in entry["params"].items()
            },
        }
        for entry in dropped
    ]


_real_renumbered = fixtures.renumbered
_real_without_new_ids = fixtures.without_new_ids
DEGRADED_RULES = {
    "renumbered_off_by_one": ("renumbered", _off_by_one_renumbering),
    "renumbered_not_at_all": ("renumbered", lambda value, new: value),
    "next_id_off_by_one": ("schema2_next_id", lambda next_id, new: next_id - len(new) + 1),
    "next_id_kept": ("schema2_next_id", lambda next_id, new: next_id),
    "new_params_kept": ("without_new_params", lambda entries: entries),
    "new_ids_kept": ("without_new_ids", lambda entries: entries),
    "a_band_id_dropped_from_an_undo_list": ("without_new_ids", _over_dropped_ids),
    "migration_not_split": ("split_migration", lambda log: (list(log), None)),
    "last_entry_always_split": ("split_migration", lambda log: (list(log[:-1]), log[-1])),
}


@pytest.mark.parametrize("name", DEGRADED_RULES)
def test_each_reading_rule_is_needed(name, projects, fresh, monkeypatch):
    # Undegraded, the opened f3, its fresh build and the literal values all
    # read back as schema 2 held them.
    golden = _golden("f3", "stored.json")
    assert fixtures.stored_differences(golden, projects["f3"]) == []
    assert fixtures.fresh_changes(golden, fresh["f3"]) == []
    assert _literal_rule_failures() == []
    rule, degraded = DEGRADED_RULES[name]
    monkeypatch.setattr(fixtures, rule, degraded)
    found = [
        *fixtures.stored_differences(golden, projects["f3"]),
        *(str(change) for change in fixtures.fresh_changes(golden, fresh["f3"])),
        *_literal_rule_failures(),
    ]
    assert found, name


# Ids a build drew: lt-3, arena-4 (with img-1 and mem-2), arena-6 (with
# img-5) and row-8 (with prot-7); a schema-2 build drew img-1, mem-2, img-3,
# prot-4 and band-5 for the same.
NEW = [3, 4, 6, 8]


def _literal_rule_failures() -> list[str]:
    """Which reading rule gives another value than schema 2 held, on literal
    values: its name, once each."""
    failures = []
    if fixtures.schema2_next_id(12, NEW) != 8:
        failures.append("schema2_next_id")
    given = ["img-5", "img-5.tif", "prot-7", "mem-2", "band-9", "arena-6", "x img-5", "lt-3"]
    held = ["img-3", "img-3.tif", "prot-4", "mem-2", "band-5", "arena-6", "x img-5", "lt-3"]
    if fixtures.renumbered({"ids": given}, NEW) != {"ids": held}:
        failures.append("renumbered")
    made = {"image_id": "img-1", "arena_id": "arena-4", "lane_table_id": "lt-3", "row_id": "row-8"}
    if fixtures.without_new_params([{"seq": 2, "params": made}]) != [
        {"seq": 2, "params": {"image_id": "img-1"}}
    ]:
        failures.append("without_new_params")
    undo = {
        "returns_to_seq": 1,
        "restored": ["mem-2", "img-1", "arena-4", "prot-7", "row-8", "band-9"],
        "undetected_restored": [["prot-7", 0, 0]],
    }
    if fixtures.without_new_ids([{"seq": 9, "params": undo}]) != [
        {"seq": 9, "params": {**undo, "restored": ["mem-2", "img-1", "prot-7", "band-9"]}}
    ]:
        failures.append("without_new_ids")
    return failures


def _relabelled(project, renames: dict[str, str], next_id: int | None = None):
    """``project`` with these ids renamed everywhere (its log too), and each
    renamed image's file with it, validated."""
    renames = dict(renames)
    for image in project.batch.iter_images():
        if image.id in renames:
            renames[image.file] = renames[image.id] + image.file[len(image.id) :]

    def renamed(value):
        if isinstance(value, dict):
            return {key: renamed(item) for key, item in value.items()}
        if isinstance(value, list):
            return [renamed(item) for item in value]
        return renames.get(value, value) if isinstance(value, str) else value

    doc = renamed(project.model_dump(mode="json"))
    if next_id is not None:
        doc["next_id"] = next_id
    return type(project).model_validate(doc)


def test_allocation_issues_catch_new_ids_drawn_elsewhere(fresh):
    project = fresh["f3"]
    assert fixtures.allocation_issues(project) == []
    batch = project.batch
    beta = batch.proteins[0]
    row = batch.only_row(beta).id
    last = project.next_id
    # The row drawn after the protein's bands (last), not right after it.
    late_row = _relabelled(project, {row: f"row-{last}"}, next_id=last + 1)
    [issue] = fixtures.allocation_issues(late_row)
    assert issue[:2] == [f"$.allocation.row-{last}", beta.id] and issue[2] != beta.id
    # An image's area drawn before the image.
    arena = batch.membranes[0].arenas[0]
    early = _relabelled(project, {arena.id: "arena-98", arena.image_id: "img-99"}, next_id=100)
    assert [issue[0] for issue in fixtures.allocation_issues(early)] == ["$.allocation.arena-98"]
    # Something else drawn between an image and its area.
    other = batch.membranes[1].arenas[0]
    between = _relabelled(project, {other.id: f"arena-{last}"}, next_id=last + 1)
    assert [issue[0] for issue in fixtures.allocation_issues(between)] == [
        f"$.allocation.arena-{last}"
    ]


def test_fresh_changes_catch_images_drawn_out_of_order(fresh):
    # The frozen kinds' ids drawn in another order: two images swapped.
    golden = _golden("f3", "stored.json")
    project = fresh["f3"]
    first, second = (image.id for image in list(project.batch.iter_images())[:2])
    swapped = _relabelled(project, {first: second, second: first})
    assert fixtures.fresh_changes(golden, swapped) != []


def _degraded_conversion(kind: str):
    """The conversion to schema 3 with one thing done wrong."""
    real = storage._v2_to_v3

    def step(doc):
        given = doc.get("next_id", 1)
        out = real(doc)
        batch = out["batch"]
        images = {image["id"]: image for image in batch["images"]}
        if kind == "areas_reversed":
            for membrane in batch["membranes"]:
                membrane["arenas"].reverse()
        elif kind == "row_on_the_first_area":
            first = {
                arena["id"]: membrane["arenas"][0]["id"]
                for membrane in batch["membranes"]
                for arena in membrane["arenas"]
            }
            for protein in batch["proteins"]:
                protein["rows"][0]["arena_id"] = first[protein["rows"][0]["arena_id"]]
        elif kind == "reference_dropped":
            batch["reference_condition"] = None
        elif kind in ("padding_dropped", "undetected_dropped"):
            key = kind.removesuffix("_dropped").replace("padding", "box_padding")
            for protein in batch["proteins"]:
                protein["rows"][0].pop(key, None)
        elif kind == "lane_metadata_dropped":
            for lane in batch["lane_tables"][0]["lanes"]:
                lane["metadata"] = {}
        elif kind == "next_id_kept":
            out["next_id"] = given
        elif kind == "outline_written":
            for membrane in batch["membranes"]:
                for arena in membrane["arenas"]:
                    image = images[arena["image_id"]]
                    arena["outline"] = {
                        "x0": 0,
                        "y0": 0,
                        "x1": image["width"],
                        "y1": image["height"],
                    }
        elif kind == "marker_link_dropped":
            for image in batch["images"]:
                image["marker_image_id"] = None
        return out

    return step


CONVERSION_DEFECTS = (
    "areas_reversed",
    "row_on_the_first_area",
    "reference_dropped",
    "padding_dropped",
    "undetected_dropped",
    "lane_metadata_dropped",
    "next_id_kept",
    "outline_written",
    "marker_link_dropped",
)


def _conversion_findings(case: str, folder: Path) -> list[str]:
    """Which of the conversion's checks fail on ``case`` (a frozen fixture,
    or ``sample``, the conftest sample as a schema-2 file) opened from
    ``folder``: opening, the content mapped back, the stored values, no
    outline, and the conversion's entry."""
    if case == "sample":
        data = storage.document_bytes(sample_doc())
        try:
            project = storage.project_from_json(data)
        except storage.ProjectFormatError:
            return ["opens"]
        frozen = {
            "content_hash": "d667c01c0856fa2c7f8b046729995b0c50cf9f11c2481b11f817599eb206b6d6"
        }
        found = []
    else:
        fixtures.copy_project(DATA / case, folder)
        frozen = _golden(case, "stored.json")
        try:
            session = ops.open_project(folder, autosave=None)
        except storage.ProjectFormatError:
            return ["opens"]
        project = session.project
        session.close()
        found = ["values"] if fixtures.stored_differences(frozen, project) else []
    content = storage.content_document(project)
    if storage.document_hash(storage._EARLIER_CONTENT[2](content)) != frozen["content_hash"]:
        found.append("inverse")
    if any("outline" in a for m in content["batch"]["membranes"] for a in m["arenas"]):
        found.append("no_outline")
    if case != "sample" and fixtures.migration_issues(frozen, project):
        found.append("history")
    return found


def test_the_conversion_checks_pass_the_conversion(tmp_path):
    for case in ("f1", "sample"):
        assert _conversion_findings(case, tmp_path / case) == [], case


@pytest.mark.parametrize("kind", CONVERSION_DEFECTS)
def test_each_defect_of_the_conversion_fails_a_check(kind, tmp_path, monkeypatch):
    monkeypatch.setitem(storage.MIGRATIONS, 2, _degraded_conversion(kind))
    found = {case: _conversion_findings(case, tmp_path / case) for case in ("f1", "sample")}
    assert any(found.values()), (kind, found)
