# SPDX-License-Identifier: Apache-2.0
"""Tests for the project folder: bytes, content hash, save, load and the image store."""

import hashlib
import io
import json
import math
import os
import random
import struct
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import image_bytes, make_project, make_project_with_undetected, write_image_files
from proteia.core import storage
from proteia.core.model import (
    SCHEMA_VERSION,
    Band,
    Box,
    ImageKind,
    ImageRef,
    LogEntry,
    Polarity,
    Project,
    ProposalSource,
    apply_change,
    revalidate,
)
from proteia.core.project import join_to_spine
from proteia.core.storage import (
    HASH_EXCLUDE,
    MIGRATIONS,
    ImageTooLargeError,
    MissingImageError,
    ProjectFormatError,
    SchemaVersionError,
    canonical_json,
    content_document,
    content_hash,
    document_bytes,
    document_hash,
    image_path,
    load_project,
    migrate,
    orphan_files,
    project_from_json,
    project_to_json,
    save_project,
    store_image,
    verify_images,
)

GOLDEN = Path(__file__).parent / "data" / "regression_baseline.json"


@pytest.fixture
def no_delay(monkeypatch):
    monkeypatch.setattr(storage, "REPLACE_DELAY", 0)


def _saved(folder: Path, project: Project) -> Path:
    """Save ``project`` with its stand-in image files into ``folder``."""
    write_image_files(folder, project)
    return save_project(project, folder)


def _doc(project: Project) -> dict:
    """The saved form of ``project`` as a dict, for editing a file by hand."""
    return json.loads(project_to_json(project))


def _encode(doc: dict) -> bytes:
    return json.dumps(doc, ensure_ascii=False).encode()


def _logged(project: Project, *entries: dict) -> Project:
    """``project`` with a log of ``entries`` (seq 1, 2, ... in order)."""
    log = []
    for seq, fields in enumerate(entries, start=1):
        entry = {
            "time": "2026-09-26T08:00:00.000Z",
            "action": "set_lanes",
            "version": "0.1.0",
            "content_hash": content_hash(project),
            **fields,
        }
        log.append(LogEntry(seq=seq, **entry))
    return Project.model_validate({**project.model_dump(), "log": log})


def test_round_trip_through_project_folder(tmp_path):
    folder = tmp_path / "專案 β-blot 10 µM"
    project = make_project()
    stored = {}
    for image in project.batch.iter_images():
        pixels = random.Random(image.id).randbytes(1_500_000)  # more than one copy chunk
        stored[image.id] = store_image(folder, image.id, image.original_name, io.BytesIO(pixels))

    def record(draft: Project) -> None:
        for image in draft.batch.iter_images():
            image.file = stored[image.id].file
            image.sha256 = stored[image.id].sha256

    project, _ = apply_change(project, record)
    path = save_project(project, folder)
    assert path == folder / "project.json"
    assert load_project(folder) == project
    assert (folder / "exports").is_dir()
    for image in project.batch.iter_images():
        assert image_path(folder, image).is_file()
    assert verify_images(project, folder) == []


def test_serialization_is_deterministic():
    project = make_project()
    assert project_to_json(project) == project_to_json(project)
    other = make_project()  # built independently
    assert project_to_json(other) == project_to_json(project)
    assert content_hash(other) == content_hash(project)


def test_resave_after_load_is_byte_identical(tmp_path):
    first = _saved(tmp_path / "a", make_project())
    loaded = load_project(tmp_path / "a")
    second = _saved(tmp_path / "b", loaded)
    assert second.read_bytes() == first.read_bytes()


def test_project_json_encoding(tmp_path):
    data = _saved(tmp_path, make_project()).read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert data.endswith(b"}\n")
    assert "β-catenin".encode() in data
    assert b"\xc2\xb5" in data  # µ as raw UTF-8
    assert b"\\u" not in data
    assert b'"schema_version": 1' in data


def test_empty_project_bytes_exact():
    assert project_to_json(Project()) == (
        b'{\n "batch": {\n  "lanes": [],\n  "membranes": [],\n  "proteins": [],\n'
        b'  "reference_condition": null\n },\n "log": [],\n "next_id": 1,\n'
        b' "schema_version": 1\n}\n'
    )


def test_content_hash_is_pinned():
    """The content hash of the conftest sample project, pinned across platforms.

    Regenerate the constant only for a deliberate change to the schema or the
    canonical form, and say why in the pull request. After the v0.1 tag such a
    change also bumps SCHEMA_VERSION and registers a migration. The constant lives
    in code, not in a committed file, because the Windows CI checkout converts text
    files to CRLF. It did not move when the action log was added: the log is not
    content, and the hash leaves it out.
    """
    assert content_hash(make_project()) == (
        "28dfdbcbca235bb7359164954bf76a6d743a2c4448e51049172f42cc00b3df6a"
    )


def test_content_hash_is_sha256_of_canonical_json():
    project = make_project()
    doc = _doc(project)
    for key in HASH_EXCLUDE:
        del doc[key]
    compact = canonical_json(doc)
    assert content_hash(project) == hashlib.sha256(compact).hexdigest()
    assert b": " not in compact
    assert b"\n" not in compact


def test_content_bytes_are_what_the_hash_covers():
    for project in (make_project(), make_project_with_undetected(), Project()):
        data = storage.content_bytes(project)
        assert data == canonical_json(content_document(project))
        assert hashlib.sha256(data).hexdigest() == content_hash(project)


def test_project_from_content_restores_the_content_with_the_given_next_id_and_log():
    project = _logged(make_project_with_undetected(), {"action": "new_project"})
    data = storage.content_bytes(project)
    restored = storage.project_from_content(data, next_id=25, log=project.log)
    assert restored.log is project.log  # shared, not copied or validated again
    assert restored.next_id == 25
    assert restored.model_copy(update={"next_id": project.next_id}) == project
    assert storage.content_bytes(restored) == data
    assert project_to_json(storage.project_from_content(data, next_id=19, log=project.log)) == (
        project_to_json(project)
    )

    # Validated strictly, as project_from_json validates a file.
    for edit in (
        lambda d: d["batch"]["lanes"][0].update(included=0),
        lambda d: d["batch"].update(notes="x"),
    ):
        doc = json.loads(data)
        edit(doc)
        with pytest.raises(ValidationError):
            storage.project_from_content(canonical_json(doc), next_id=19, log=())
    with pytest.raises(ValidationError):  # an id at or above next_id
        storage.project_from_content(data, next_id=18, log=())
    # A repeated key is refused as such, not left to model validation (the last
    # value would otherwise win, and this one is valid).
    repeated = data.replace(b"{", b'{"schema_version":1,', 1)
    assert repeated.count(b'"schema_version"') == 2
    with pytest.raises(ValueError, match="duplicate key 'schema_version'") as info:
        storage.project_from_content(repeated, next_id=19, log=())
    assert not isinstance(info.value, ValidationError)


def test_canonical_json_rules():
    assert canonical_json({"b": "β", "a": 1e-07}) == '{"a":1e-07,"b":"β"}'.encode()
    with pytest.raises(ValueError):
        canonical_json({"a": math.nan})


def test_content_hash_ignores_representation(tmp_path):
    project = make_project()
    expected = content_hash(project)

    doc = project.model_dump()
    for protein in doc["batch"]["proteins"]:
        protein["bands"].reverse()
    lane = doc["batch"]["lanes"][3]
    lane["metadata"] = dict(reversed(lane["metadata"].items()))
    assert content_hash(Project.model_validate(doc)) == expected

    bumped, _ = apply_change(project, lambda p: p.new_id("img"))
    assert bumped.next_id != project.next_id
    assert content_hash(bumped) == expected

    _saved(tmp_path, project)
    assert content_hash(load_project(tmp_path)) == expected


def _band(project: Project, band_id: str) -> Band:
    return project.batch.find_band(band_id)[1]


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda p: setattr(_band(p, "band-10"), "box", Box(x=19, y=43)), id="box"),
        pytest.param(lambda p: setattr(p.batch.lanes[1], "included", False), id="included"),
        pytest.param(
            lambda p: setattr(p.batch.find_image("img-2"), "polarity", Polarity.LIGHT_ON_DARK),
            id="polarity",
        ),
        pytest.param(
            lambda p: setattr(p.batch.find_image("img-2"), "sha256", "0" * 64), id="pixels"
        ),
        pytest.param(lambda p: setattr(p.batch.find_protein("prot-9"), "name", "Gapdh"), id="name"),
        pytest.param(
            lambda p: setattr(
                _band(p, "band-10"), "net", math.nextafter(_band(p, "band-10").net, 0)
            ),
            id="net-one-ulp",
        ),
        pytest.param(lambda p: setattr(_band(p, "band-10"), "clipped", False), id="clipped"),
    ],
)
def test_content_hash_changes_with_content(change):
    project = make_project()
    changed, _ = apply_change(project, change)
    assert content_hash(changed) != content_hash(project)


# --- Not-detected records in the saved form ---


def test_content_hash_with_records_is_pinned():
    """The content hash of the sample project with not-detected records: the records
    are content. Regenerate it on the same terms as the pinned hash above."""
    assert content_hash(make_project_with_undetected()) == (
        "805e1f12d680a4466887a9f53e52e6405dc960ebde5334ea94b0bda7719389b4"
    )


def test_records_round_trip_and_resave_byte_identically(tmp_path):
    project = make_project_with_undetected()
    first = _saved(tmp_path / "a", project)
    data = first.read_bytes()
    assert data.count(b'"undetected"') == 2  # β-catenin and GAPDH; α-tubulin has none
    assert b'"below_detection_limit"' in data
    loaded = load_project(tmp_path / "a")
    assert loaded == project
    assert content_hash(loaded) == content_hash(project)
    second = _saved(tmp_path / "b", loaded)
    assert second.read_bytes() == data


def test_an_empty_record_list_is_never_written():
    project = make_project()
    doc = _doc(project)
    assert all("undetected" not in protein for protein in doc["batch"]["proteins"])
    for protein in doc["batch"]["proteins"]:
        protein["undetected"] = []  # a hand-edited file: loads, and is saved without it
    loaded = project_from_json(_encode(doc))
    assert loaded == project
    assert content_hash(loaded) == content_hash(project)
    assert project_to_json(loaded) == project_to_json(project)
    # Writing the empty list would have moved the hash of every existing project.
    written = {key: value for key, value in doc.items() if key not in HASH_EXCLUDE}
    assert document_hash(written) != content_hash(project)
    assert content_document(loaded) == content_document(project)


def _gapdh_records(p: Project) -> list:
    return p.batch.find_protein("prot-9").undetected


def _swap_record_lanes(p: Project) -> None:
    lane_2, lane_3 = _gapdh_records(p)
    lane_2.lane_index, lane_3.lane_index = 3, 2


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_swap_record_lanes, id="lane_index"),
        pytest.param(
            lambda p: setattr(p.batch.find_protein("prot-7").undetected[0], "band_index", 1),
            id="band_index",
        ),
        pytest.param(
            lambda p: setattr(
                _gapdh_records(p)[0], "snr", math.nextafter(_gapdh_records(p)[0].snr, 0)
            ),
            id="snr-one-ulp",
        ),
        pytest.param(lambda p: setattr(_gapdh_records(p)[0], "threshold", 7.0), id="threshold"),
        pytest.param(lambda p: setattr(_gapdh_records(p)[0].region, "x0", 99), id="region"),
        pytest.param(
            lambda p: setattr(_gapdh_records(p)[0], "source", ProposalSource.MW_GUIDED),
            id="source",
        ),
        pytest.param(lambda p: _gapdh_records(p).pop(), id="removed"),
    ],
)
def test_content_hash_changes_with_any_record_field(change):
    # Two bands expected for β-catenin, so its record may move to band index 1.
    project, _ = apply_change(
        make_project_with_undetected(),
        lambda p: setattr(p.batch.find_protein("prot-7"), "expected_band_count", 2),
    )
    changed, _ = apply_change(project, change)
    assert content_hash(changed) != content_hash(project)


def test_record_order_does_not_change_the_hash():
    doc = make_project_with_undetected().model_dump()
    # A second band's record in β-catenin's lane 2: order within a lane counts too.
    beta = doc["batch"]["proteins"][0]
    beta["expected_band_count"] = 2
    beta["undetected"].append({**beta["undetected"][0], "band_index": 1})
    project = Project.model_validate(doc)
    for protein in doc["batch"]["proteins"]:
        protein.get("undetected", []).reverse()
    assert content_hash(Project.model_validate(doc)) == content_hash(project)


# --- The action log in project.json ---


def test_content_hash_ignores_the_log():
    plain = make_project()
    logged = _logged(
        plain,
        {"action": "new_project", "content_hash": content_hash(Project())},
        {"time": "2031-01-01T00:00:00.999Z", "version": "0.2.0.dev1", "params": {"n": 4}},
    )
    assert content_hash(logged) == content_hash(plain)
    assert project_to_json(logged) != project_to_json(plain)
    assert HASH_EXCLUDE == {"next_id", "log"}


def test_log_round_trips_byte_identically(tmp_path):
    params = {
        "rel_threshold": 0.3,
        "noise_k": 3.0,
        "tiny": 1e-07,
        "lane_index": 2,
        "sample": None,
        "rect": [18, 43, 42, 57],
        "changed": [{"condition": "10 µM", "sample": "α1", "included": False}],
        "name": "β-catenin",
    }
    project = _logged(make_project(), {"action": "new_project"}, {"params": params})
    first = _saved(tmp_path / "a", project)
    loaded = load_project(tmp_path / "a")
    assert loaded == project
    stored = loaded.log[1].params
    assert stored == params
    assert [type(stored[key]) for key in ("noise_k", "lane_index")] == [float, int]  # 3.0 stays
    second = _saved(tmp_path / "b", loaded)
    assert second.read_bytes() == first.read_bytes()


def test_project_json_without_a_log_loads():
    project = _logged(make_project(), {"action": "new_project"})
    doc = _doc(project)
    del doc["log"]  # a file saved before the log existed
    loaded = project_from_json(_encode(doc))
    assert loaded.log == ()
    assert content_hash(loaded) == content_hash(project)


def test_load_rejects_a_log_out_of_sequence():
    doc = _doc(_logged(make_project(), {"action": "new_project"}, {}, {}))
    del doc["log"][1]  # an entry deleted by hand
    with pytest.raises(ProjectFormatError, match="log entry 3 is out of sequence: expected 2"):
        project_from_json(_encode(doc))


def test_document_bytes_rules():
    data = document_bytes({"b": "µ α β", "a": [1, 0.3, None]})
    assert data == '{\n "a": [\n  1,\n  0.3,\n  null\n ],\n "b": "µ α β"\n}\n'.encode()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert b"\\u" not in data
    with pytest.raises(ValueError):
        document_bytes({"a": math.nan})
    project = _logged(make_project(), {"action": "new_project"})
    assert project_to_json(project) == document_bytes(revalidate(project).model_dump(mode="json"))


def _spine(project: Project) -> list:
    """Every protein's nets scattered into the lane spine by stored lane index."""
    positioned = [[(b.lane_index, b.net) for b in p.bands] for p in project.batch.proteins]
    return join_to_spine(positioned, len(project.batch.lanes))


def test_numbers_survive_round_trip_exactly(tmp_path):
    nets_by_polarity = json.loads(GOLDEN.read_text(encoding="utf-8"))["values"]["nets"]
    values = [net for nets in nets_by_polarity.values() for row in nets.values() for net in row]
    values += [0.1, 1e-7, 1e16, 5e-324, sys.float_info.max]
    project = make_project()
    band_ids = [band.id for protein in project.batch.proteins for band in protein.bands]

    # Nine bands per round: set their nets, save, load, compare bit for bit.
    for start in range(0, len(values), len(band_ids)):
        chunk = values[start : start + len(band_ids)]

        def set_nets(draft: Project, chunk=chunk) -> None:
            for band_id, net in zip(band_ids, chunk, strict=False):
                draft.batch.find_band(band_id)[1].net = net

        project, _ = apply_change(project, set_nets)
        folder = tmp_path / f"round-{start}"
        _saved(folder, project)
        loaded = load_project(folder)
        for band_id, net in zip(band_ids, chunk, strict=False):
            got = loaded.batch.find_band(band_id)[1].net
            assert got == net
            assert struct.pack("<d", got) == struct.pack("<d", net)
        assert _spine(loaded) == _spine(project)


def test_load_rejects_newer_schema_version():
    doc = _doc(make_project())
    doc["schema_version"] = 2
    with pytest.raises(SchemaVersionError, match="newer Proteia") as info:
        project_from_json(_encode(doc))
    assert (info.value.found, info.value.supported) == (2, SCHEMA_VERSION)


_MISSING = object()


@pytest.mark.parametrize(
    "version", [_MISSING, "1", True, 0, -1, 1.0], ids=["missing", "str", "true", "0", "-1", "1.0"]
)
def test_load_rejects_bad_schema_version(version):
    doc = _doc(make_project())
    if version is _MISSING:
        del doc["schema_version"]
    else:
        doc["schema_version"] = version
    with pytest.raises(SchemaVersionError, match="schema_version"):
        project_from_json(_encode(doc))


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"\xff", id="not-utf-8"),
        pytest.param(b"{", id="truncated"),
        pytest.param(b"[]", id="not-an-object"),
        pytest.param(
            b'{"schema_version": 1, "batch": {"lanes": [], "lanes": []}}', id="duplicate-key"
        ),
        pytest.param(b'{"schema_version": 1, "next_id": NaN}', id="nan"),
        pytest.param(b'{"schema_version": 1, "next_id": Infinity}', id="infinity"),
    ],
)
def test_load_rejects_malformed_file(data):
    with pytest.raises(ProjectFormatError):
        project_from_json(data)


def test_load_rejects_absurdly_deep_nesting():
    # A corrupt file must be a format error, not a RecursionError.
    with pytest.raises(ProjectFormatError):
        project_from_json(b"[" * 100_000 + b"]" * 100_000)


def test_load_accepts_bom_and_crlf():
    project = make_project()
    data = b"\xef\xbb\xbf" + project_to_json(project).replace(b"\n", b"\r\n")
    assert project_from_json(data) == project


def test_load_wraps_invalid_content():
    doc = _doc(make_project())
    bands = doc["batch"]["proteins"][0]["bands"]
    bands[1]["box"] = dict(bands[0]["box"])
    with pytest.raises(ProjectFormatError, match="overlap") as info:
        project_from_json(_encode(doc))
    assert isinstance(info.value.__cause__, ValidationError)


def test_load_errors_number_lanes_from_1():
    # Opening shows the message to the user, who counts lanes from 1: the
    # file's lane_index 0 is lane 1.
    doc = _doc(make_project())
    bands = doc["batch"]["proteins"][0]["bands"]
    bands[1]["lane_index"] = bands[0]["lane_index"] = 0
    with pytest.raises(ProjectFormatError) as info:
        project_from_json(_encode(doc))
    assert "protein prot-7: two bands in lane 1 with band index 0" in str(info.value)


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(
            lambda d: d["batch"]["membranes"][0]["images"][0].update(width="340"), id="str-int"
        ),
        pytest.param(lambda d: d["batch"]["lanes"][0].update(included=0), id="int-bool"),
        pytest.param(lambda d: d["batch"]["lanes"][1].update(index=1.0), id="float-int"),
        pytest.param(lambda d: d["batch"].update(notes="x"), id="unknown-key"),
    ],
)
def test_load_is_strict_about_types(edit):
    doc = _doc(make_project())
    edit(doc)
    with pytest.raises(ProjectFormatError):
        project_from_json(_encode(doc))


def test_migrate_chain():
    def one_to_two(doc):
        doc["schema_version"] = 2
        doc["batch"]["lanes"].append("added by 1->2")  # in place: must not reach the input
        return doc

    def two_to_three(doc):
        return {**doc, "schema_version": 3, "seen": list(doc["batch"]["lanes"])}

    doc = {"schema_version": 1, "batch": {"lanes": []}}
    steps = {1: one_to_two, 2: two_to_three}
    assert migrate(doc, target=3, migrations=steps) == {
        "schema_version": 3,
        "batch": {"lanes": ["added by 1->2"]},
        "seen": ["added by 1->2"],
    }
    assert doc == {"schema_version": 1, "batch": {"lanes": []}}

    with pytest.raises(SchemaVersionError, match="no migration from schema 2"):
        migrate(doc, target=3, migrations={1: one_to_two})
    with pytest.raises(RuntimeError, match="did not produce 2"):
        migrate(doc, target=2, migrations={1: lambda d: d})
    same = migrate(doc, target=1, migrations={})
    assert same == doc and same is not doc


def test_migrations_cover_every_older_version():
    assert set(MIGRATIONS) == set(range(1, SCHEMA_VERSION))


def test_load_reports_missing_image(tmp_path):
    project = make_project()
    _saved(tmp_path, project)
    (tmp_path / "images" / "img-6.jpg").unlink()
    with pytest.raises(MissingImageError, match="img-6") as info:
        load_project(tmp_path)
    assert info.value.image_ids == ["img-6"]
    assert load_project(tmp_path, require_images=False) == project


def test_save_refuses_missing_image_files(tmp_path):
    folder = tmp_path / "new"
    with pytest.raises(MissingImageError) as info:
        save_project(make_project(), folder)
    assert info.value.image_ids == ["img-2", "img-3", "img-4", "img-6"]
    assert not folder.exists()


def test_save_revalidates_in_place_edits(tmp_path):
    project = make_project()
    path = _saved(tmp_path, project)
    before = path.read_bytes()
    # Onto band-10's box, bypassing apply_change.
    project.batch.find_protein("prot-7").bands.append(
        Band(id="band-19", lane_index=2, box=Box(x=20, y=45), net=1.0, source="manual")
    )
    project.next_id = 20
    with pytest.raises(ValidationError, match="overlap"):
        save_project(project, tmp_path)
    assert path.read_bytes() == before


def _renamed(project: Project) -> Project:
    changed, _ = apply_change(project, lambda p: setattr(p.batch.proteins[2], "name", "Gapdh"))
    return changed


def test_failed_replace_keeps_previous_file(tmp_path, monkeypatch, no_delay):
    project = make_project()
    path = _saved(tmp_path, project)
    before = path.read_bytes()
    calls = []

    def locked(src, dst):
        calls.append(dst)
        raise PermissionError(13, "held by another process", str(dst))

    monkeypatch.setattr(storage.os, "replace", locked)
    with pytest.raises(PermissionError):
        save_project(_renamed(project), tmp_path)
    assert len(calls) == storage.REPLACE_ATTEMPTS
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["exports", "images", "project.json"]


def test_transient_permission_error_is_retried(tmp_path, monkeypatch, no_delay):
    project = make_project()
    path = _saved(tmp_path, project)
    real_replace = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) <= 2:
            raise PermissionError(13, "held by another process", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(storage.os, "replace", flaky)
    renamed = _renamed(project)
    save_project(renamed, tmp_path)
    assert len(calls) == 3
    assert path.read_bytes() == project_to_json(renamed)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["exports", "images", "project.json"]


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to replace an open file")
def test_save_while_reader_holds_file(tmp_path, no_delay):
    project = make_project()
    path = _saved(tmp_path, project)
    before = path.read_bytes()
    with open(path, "rb"), pytest.raises(PermissionError):
        save_project(_renamed(project), tmp_path)
    assert path.read_bytes() == before


def test_store_image_copies_under_generated_name(tmp_path):
    pixels = random.Random(2).randbytes(3_000_000)
    stored = store_image(tmp_path, "img-2", "β-actin 10 µM.TIF", io.BytesIO(pixels))
    assert stored.file == "img-2.tif"
    assert (tmp_path / "images" / "img-2.tif").read_bytes() == pixels
    assert stored.sha256 == hashlib.sha256(pixels).hexdigest()
    assert stored.size == len(pixels)
    assert [p.name for p in (tmp_path / "images").iterdir()] == ["img-2.tif"]  # no .part left
    ImageRef(
        id="img-2",
        file=stored.file,
        original_name="β-actin 10 µM.TIF",
        kind=ImageKind.CHEMILUMINESCENCE,
        sha256=stored.sha256,
        width=340,
        height=150,
        polarity=Polarity.DARK_ON_LIGHT,
        background=200.0,
    )


@pytest.mark.parametrize(
    ("image_id", "original_name", "data", "max_bytes", "error"),
    [
        pytest.param("img-1", "blot.bmp", b"x", None, ValueError, id="bmp"),
        pytest.param("img-1", "blot", b"x", None, ValueError, id="no-suffix"),
        pytest.param("../img-1", "blot.tif", b"x", None, ValueError, id="path-in-id"),
        pytest.param("prot-1", "blot.tif", b"x", None, ValueError, id="not-an-image-id"),
        pytest.param("img-9", "blot.tif", b"x", None, FileExistsError, id="existing-target"),
        pytest.param("img-1", "blot.tif", b"", None, ValueError, id="empty"),
        pytest.param(
            "img-1", "blot.tif", b"x" * 3_000_000, 2_000_000, ImageTooLargeError, id="too-large"
        ),
    ],
)
def test_store_image_refusals(tmp_path, image_id, original_name, data, max_bytes, error):
    images = tmp_path / "images"
    images.mkdir()
    (images / "img-9.tif").write_bytes(image_bytes("img-9"))
    with pytest.raises(error):
        store_image(tmp_path, image_id, original_name, io.BytesIO(data), max_bytes=max_bytes)
    assert [p.name for p in images.iterdir()] == ["img-9.tif"]
    assert (images / "img-9.tif").read_bytes() == image_bytes("img-9")


def test_image_path_stays_inside_images_folder(tmp_path):
    for image in make_project().batch.iter_images():
        assert image_path(tmp_path, image).parent == tmp_path / "images"


def test_verify_images_reports_changed_and_missing(tmp_path):
    project = make_project()
    write_image_files(tmp_path, project)
    assert verify_images(project, tmp_path) == []
    changed = tmp_path / "images" / "img-2.tif"
    data = bytearray(changed.read_bytes())
    data[0] ^= 0xFF
    changed.write_bytes(bytes(data))
    (tmp_path / "images" / "img-6.jpg").unlink()
    assert verify_images(project, tmp_path) == ["img-2", "img-6"]


def test_orphan_files_lists_unreferenced_files(tmp_path):
    project = make_project()
    assert orphan_files(project, tmp_path) == []  # no images/ folder yet
    write_image_files(tmp_path, project)
    assert orphan_files(project, tmp_path) == []
    # An import that was never saved, and a temp file left by a crash.
    store_image(tmp_path, "img-19", "unsaved α.tif", io.BytesIO(b"pixels"))
    stale = tmp_path / "images" / ".img-20.tif.abc.part"
    stale.write_bytes(b"partial")
    assert orphan_files(project, tmp_path) == [stale, tmp_path / "images" / "img-19.tif"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_saved_files_get_default_permissions(tmp_path):
    umask = os.umask(0)
    os.umask(umask)
    expected = 0o666 & ~umask
    project = make_project()
    path = _saved(tmp_path, project)
    assert path.stat().st_mode & 0o777 == expected  # not mkstemp's owner-only 0600
    path.chmod(0o640)
    save_project(project, tmp_path)
    assert path.stat().st_mode & 0o777 == 0o640  # a replaced file keeps its mode
    stored = store_image(tmp_path, "img-19", "new β.tif", io.BytesIO(b"pixels"))
    assert (tmp_path / "images" / stored.file).stat().st_mode & 0o777 == expected
