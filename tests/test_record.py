# SPDX-License-Identifier: Apache-2.0
"""Tests for the reproducibility record: its shape, its self-check, the history
issues, the software versions and the code-level settings it reports."""

import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import pytest
from pydantic import ValidationError

import proteia
from conftest import (
    FakeClock,
    make_project,
    make_project_with_undetected,
    synthetic_blot,
    write_tiff,
)
from proteia.core import operations as ops
from proteia.core import rowdetect
from proteia.core.analyze import ReduceMethod
from proteia.core.export import LANE_TABLE_DECIMALS
from proteia.core.grow import NOISE_K, REL_THRESHOLD, grow_box
from proteia.core.model import ImageKind, LogEntry, Polarity, Project, Role
from proteia.core.plotspec import ErrorType
from proteia.core.quantify import CLIPPED_PIXELS_THRESHOLD
from proteia.core.record import (
    RECORD_FORMAT,
    build_record,
    history_issues,
    record_bytes,
    results_settings,
    settings,
    software_versions,
)
from proteia.core.results import compute_results
from proteia.core.storage import (
    canonical_json,
    content_document,
    content_hash,
    document_bytes,
    load_project,
)

EXPORTED_AT = "2026-09-26T09:30:00.250Z"


def _with_log(project: Project, *actions: str) -> Project:
    """``project`` with one entry per action, each naming its current content hash."""
    digest = content_hash(project)
    log = tuple(
        LogEntry(
            seq=seq,
            time=f"2026-09-26T08:00:0{seq}.000Z",
            action=action,
            version=proteia.__version__,
            content_hash=digest,
        )
        for seq, action in enumerate(actions, start=1)
    )
    return project.model_copy(update={"log": log})


def _built_through_operations(folder: Path) -> ops.ProjectSession:
    """A saved project with one image, two lanes and one box, made by the operations."""
    s = ops.new_project(folder / "專案 µ α β", clock=FakeClock())
    pixels = synthetic_blot((60, 160), [(40, 30, 4.0, 2.5, 30000.0)])
    source = write_tiff(folder / "sources" / "β-actin 10 µM.tif", pixels)
    with source.open("rb") as f:
        image = ops.import_image(
            s, f, source.name, kind=ImageKind.CHEMILUMINESCENCE, polarity=Polarity.DARK_ON_LIGHT
        )
    ops.set_lanes(s, [ops.LaneInput("vehicle"), ops.LaneInput("10 µM")])
    protein = ops.add_protein(s, "β-catenin", Role.TARGET, image)
    ops.place_box(s, protein, 40, 30, lane_index=0, grow=False)
    return s


def test_history_issues(tmp_path):
    assert history_issues(make_project()) == ["no_history"]
    assert history_issues(_with_log(make_project(), "set_lanes")) == ["history_starts_late"]

    s = _built_through_operations(tmp_path)
    assert history_issues(s.project) == []

    path = s.folder / "project.json"
    doc = json.loads(path.read_bytes())
    doc["batch"]["proteins"][0]["bands"][0]["net"] += 1.0  # a net edited by hand
    path.write_bytes(document_bytes(doc))
    assert history_issues(load_project(s.folder)) == ["content_changed_outside_log"]


def _restore_entry(seq: int, action: str, returns_to: object, digest: str) -> LogEntry:
    verb = "undone" if action == "undo" else "redone"
    return LogEntry(
        seq=seq,
        time=f"2026-09-26T08:00:0{seq}.000Z",
        action=action,
        version=proteia.__version__,
        params={f"{verb}_seq": 2, f"{verb}_action": "set_lanes", "returns_to_seq": returns_to},
        content_hash=digest,
    )


def test_an_undo_or_redo_must_return_to_content_an_earlier_entry_left():
    project = make_project()
    digest = content_hash(project)
    other = "0" * 64
    base = _with_log(project, "new_project").log  # seq 1 left this content
    second = base[0].model_copy(update={"seq": 2, "action": "set_lanes", "content_hash": other})

    def issues(*entries: LogEntry) -> list[str]:
        return history_issues(project.model_copy(update={"log": (*base, second, *entries)}))

    assert issues(_restore_entry(3, "undo", 1, digest)) == []
    assert issues(_restore_entry(3, "undo", 1, digest), _restore_entry(4, "redo", 2, other)) == [
        "content_changed_outside_log"  # the redo is consistent; the content is not its own
    ]
    # A later session opened at the undo (seq 3), made a change (4) and undid it:
    # it returns to the undo, whose entry left content too.
    later = (
        _restore_entry(3, "undo", 1, digest),
        second.model_copy(update={"seq": 4}),
        _restore_entry(5, "undo", 3, digest),
    )
    assert issues(*later) == []
    for returns_to in (None, 99, 3, 2, True, "1"):  # unknown, itself, another hash, not a seq
        assert issues(_restore_entry(3, "undo", returns_to, digest)) == ["undo_mismatch"], (
            returns_to
        )
    missing = base[0].model_copy(update={"seq": 3, "action": "redo", "params": {}})
    assert issues(missing) == ["undo_mismatch"]


def test_build_record_shape():
    project = _with_log(make_project(), "new_project", "set_lanes")
    data = "lane,condition\r\n0,10 µM\r\n".encode("utf-8-sig")
    files = {"lane-table.csv": data}
    record = build_record(project, exported_at=EXPORTED_AT, files=files)
    assert set(record) == {
        "record_format",
        "exported_at",
        "software",
        "settings",
        "content",
        "content_hash",
        "log",
        "history_issues",
        "files",
        "results",
    }
    assert record["record_format"] == RECORD_FORMAT == 1
    assert record["exported_at"] == EXPORTED_AT
    # The record verifies itself: no Proteia, no knowledge of the excluded keys.
    digest = hashlib.sha256(canonical_json(record["content"])).hexdigest()
    assert digest == record["content_hash"] == content_hash(project)
    assert record["content"] == content_document(project)
    assert not {"next_id", "log"} & set(record["content"])
    assert record["log"] == [entry.model_dump(mode="json") for entry in project.log]
    assert record["history_issues"] == []
    assert record["files"] == {
        "lane-table.csv": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    }
    software = record["software"]
    assert set(software) == {
        "proteia",
        "python",
        "numpy",
        "scipy",
        "scikit-image",
        "pillow",
        "tifffile",
    }
    assert software["proteia"] == proteia.__version__
    assert software["python"] == platform.python_version()
    assert record["settings"] == settings()
    assert record["results"] is None

    canonical_json(record)  # sorted keys, no NaN: ready to be signed as it is
    again = build_record(project, exported_at=EXPORTED_AT, files=dict(files))
    assert record_bytes(again) == record_bytes(record)
    assert json.loads(record_bytes(record)) == record
    with pytest.raises(ValidationError):
        build_record(project, exported_at="2026-09-26 09:30:00", files={})


def test_record_names_the_compute_settings():
    project = make_project()
    res = compute_results(
        project.batch,
        method=ReduceMethod.REPRESENTATIVE,
        error_type=ErrorType.SEM,
        plot_conditions=["10 µM", "vehicle"],
    )
    expected = {
        "method": "representative",
        "error_type": "SEM",
        "plot_conditions": ["vehicle", "10 µM"],  # resolved, in lane order
        "excluded_lanes": [3],
    }
    assert results_settings(res) == expected
    assert build_record(project, exported_at=EXPORTED_AT, files={}, results=res)["results"] == (
        expected
    )
    assert results_settings(compute_results(project.batch))["plot_conditions"] is None


def test_settings_are_the_code_constants():
    assert settings() == {
        "grow_box": {
            "rel_threshold": REL_THRESHOLD,
            "noise_k": NOISE_K,
            "max_width": None,
            "max_height": None,
        },
        "clipped_pixels_threshold": CLIPPED_PIXELS_THRESHOLD,
        "lane_table_decimals": LANE_TABLE_DECIMALS,
        "detect_row": rowdetect.settings(),  # every row-box detection constant
    }
    parameters = inspect.signature(grow_box).parameters
    assert parameters["rel_threshold"].default == REL_THRESHOLD == 0.3
    assert parameters["noise_k"].default == NOISE_K == 3.0


def test_a_missing_distribution_is_null(monkeypatch):
    real = importlib.metadata.version

    def version(name: str) -> str:
        if name == "tifffile":
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    software = software_versions()
    assert software["tifffile"] is None
    assert software["numpy"] == real("numpy")


def test_the_content_includes_not_detected_records():
    project = _with_log(make_project_with_undetected(), "new_project")
    record = build_record(project, exported_at=EXPORTED_AT, files={})
    proteins = record["content"]["batch"]["proteins"]
    assert ["undetected" in protein for protein in proteins] == [True, False, True]
    assert proteins[2]["undetected"] == [
        u.model_dump(mode="json") for u in project.batch.find_protein("prot-9").undetected
    ]
    # The record verifies itself, and its hash covers the records.
    digest = hashlib.sha256(canonical_json(record["content"])).hexdigest()
    assert digest == record["content_hash"] == content_hash(project)
    assert digest != content_hash(make_project())
    assert record["history_issues"] == []


def test_a_log_written_before_records_existed_has_no_history_issues():
    # The content hash a build without not-detected records stored for the
    # sample project (test_storage pins the same value).
    before = "28dfdbcbca235bb7359164954bf76a6d743a2c4448e51049172f42cc00b3df6a"
    entry = LogEntry(
        seq=1,
        time="2026-09-20T08:00:00.000Z",
        action="new_project",
        version="0.1.0.dev0",
        content_hash=before,
    )
    project = make_project().model_copy(update={"log": (entry,)})
    assert history_issues(project) == []
    record = build_record(project, exported_at=EXPORTED_AT, files={})
    assert record["history_issues"] == []
    assert all("undetected" not in p for p in record["content"]["batch"]["proteins"])
