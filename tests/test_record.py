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
    V1_CONTENT_HASH,
    FakeClock,
    make_project,
    make_project_with_undetected,
    sample_doc_v1,
    synthetic_blot,
    write_tiff,
)
from proteia.core import analyze, quantify, rowdetect
from proteia.core import operations as ops
from proteia.core.analyze import ReduceMethod, StatisticsSetting
from proteia.core.export import CHART_PNG_DPI, LANE_TABLE_DECIMALS, LANE_TABLE_RATIO_DECIMALS
from proteia.core.grow import NOISE_K, REL_THRESHOLD, grow_box
from proteia.core.model import ImageKind, LogEntry, Polarity, Project, Role, apply_change
from proteia.core.plotspec import ErrorType
from proteia.core.quantify import (
    CLIPPED_PIXELS_THRESHOLD,
    NEAR_LIMIT_LEVELS,
    POSSIBLY_CLIPPED_PIXELS,
)
from proteia.core.record import (
    RECORD_FORMAT,
    build_record,
    history_issues,
    record_bytes,
    results_settings,
    results_statistics,
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
    project_from_json,
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
    # The code's settings, and the method the stored nets used.
    assert record["settings"] == {**settings(), "background_method": "ring_median_v1"}
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
        "statistics": results_statistics(res),
    }
    assert results_settings(res) == expected
    assert build_record(project, exported_at=EXPORTED_AT, files={}, results=res)["results"] == (
        expected
    )
    assert results_settings(compute_results(project.batch))["plot_conditions"] is None


def test_settings_are_the_code_constants():
    assert settings() == {
        "background": quantify.background_settings(),  # ring_median v1 (#83)
        "grow_box": {
            "rel_threshold": REL_THRESHOLD,
            "noise_k": NOISE_K,
            "max_width": None,
            "max_height": None,
        },
        "clipped_pixels_threshold": CLIPPED_PIXELS_THRESHOLD,
        "possibly_clipped": {  # #112: where the exact check cannot run
            "min_pixels": POSSIBLY_CLIPPED_PIXELS,
            "near_limit_levels": NEAR_LIMIT_LEVELS,
            "levels": "on an 8-bit scale, scaled to the image's range",
            "values": "gray: the mean of red, green and blue for colour",
        },
        "lane_table_decimals": LANE_TABLE_DECIMALS,
        "lane_table_ratio_decimals": LANE_TABLE_RATIO_DECIMALS,
        "lane_table_first_lane": 1,  # the lane column numbers lanes as the app does
        "chart_png_dpi": CHART_PNG_DPI,
        "detect_row": {
            **rowdetect.settings(),  # every row-box detection constant
            # How the row commit chooses the saturation level for an image (#121).
            "saturated_at": (
                "quantify.saturation_level: the detector limit where the exact over-exposure"
                " check runs (clipped_pixels_threshold), else that limit moved in by"
                " possibly_clipped's near_limit_levels (a lossy, colour or CMYK-converted"
                " image); none for an image of unknown bit depth (float), and then no band"
                " is called hollow (hollow_band)"
            ),
        },
        "statistics": {
            "alpha": 0.05,
            "dunnett_rng_seed": 0,
            "mann_whitney": (
                "exact; with tied values, the exact permutation distribution up to"
                " mann_whitney_tied_permutations arrangements, else the normal"
                " approximation with the tie correction"
            ),
            "mann_whitney_tied_permutations": analyze.MANN_WHITNEY_PERMUTATIONS,
            "kruskal_wallis_p": "chi-square approximation",
            "dunn_adjustment": "holm",
            "log_base": "e",
            "auto_rule": analyze.AUTO_RULE_VERSION,
            "tests": sorted(analyze.TESTS),
        },
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


def test_a_log_written_before_the_band_backgrounds_has_no_history_issues():
    # A schema-1 file (before #83, and before not-detected records), its log
    # holding the hash that build computed: the migration's entry carries it on.
    doc = sample_doc_v1()
    doc["log"] = [
        {
            "seq": 1,
            "time": "2026-09-20T08:00:00.000Z",
            "action": "new_project",
            "version": "0.1.0.dev0",
            "params": {},
            "content_hash": V1_CONTENT_HASH,
        }
    ]
    project = project_from_json(document_bytes(doc))
    assert [entry.action for entry in project.log] == ["new_project", "migrate"]
    assert history_issues(project) == []
    record = build_record(project, exported_at=EXPORTED_AT, files={})
    assert record["history_issues"] == []
    assert all("undetected" not in p for p in record["content"]["batch"]["proteins"])
    # The record names the legacy method its nets still use.
    assert record["settings"]["background_method"] == "global_median"
    assert record["content"]["background_method"] == "global_median"


# --- #52: the statistics each exported chart used ---


def test_the_record_pins_what_each_charts_test_resolved_to():
    project = make_project()
    res = compute_results(project.batch, statistics={"family": "welch"})
    pinned = results_statistics(res)
    assert pinned["setting"] == {"family": "welch", "comparisons": "auto", "scale": "auto"}
    charts = pinned["charts"]
    # One chart per series of each set: this set, then the all-lanes set.
    assert [(c["result_set"], c["target_id"], c["loading_id"]) for c in charts] == [
        ("Excluding lane 4", "prot-7", "prot-8"),
        ("All lanes", "prot-7", "prot-8"),
    ]
    applied, every = charts
    # Excluding lane 4, 10 µM has no β-catenin value: no test, and why.
    assert applied["test"] is None
    assert applied["note"] == (
        "no test: fewer than 2 conditions to test; not tested: '10 µM' (no value)"
    )
    assert applied["not_tested"] == [
        {"condition": "10 µM", "n": 0, "replicates": 1, "not_detected": 0, "reason": "no_value"}
    ]
    assert applied["chosen"] == {"family": "user", "comparisons": "auto", "scale": "auto"}
    assert every["test"] is None and every["not_tested"][0]["reason"] == "fewer_than_2"
    json.dumps(pinned, allow_nan=False)


def test_the_record_pins_a_test_that_ran():
    res = compute_results(_baseline_like_batch())
    [chart] = results_statistics(res)["charts"]
    assert {key: chart[key] for key in ("result_set", "test", "family", "comparisons")} == {
        "result_set": None,
        "test": "student_t",
        "family": "pooled",
        "comparisons": "all_pairs",
    }
    assert (chart["scale"], chart["reference"], chart["covered"]) == (
        "log",
        None,
        ["vehicle", "10 µM"],
    )
    assert chart["chosen"] == {"family": "auto", "comparisons": "auto", "scale": "auto"}
    assert chart["reasons"] == [
        "ratios: log scale",
        "equal n: pooled variance",
        "2 conditions: all pairs",
    ]
    assert (chart["not_tested"], chart["note"]) == ([], None)
    # The p-values, unrounded: the legend gives them rounded.
    test = res.series[0].chart.test
    assert chart["p_value"] == test.p_value and chart["statistic"] == test.statistic
    assert chart["pairwise"] == [
        {"group_a": "vehicle", "group_b": "10 µM", "p_value": test.p_value, "estimate": est}
        for est in [test.pairwise[0].estimate]
    ]
    assert chart["method"] is None  # only the Mann-Whitney U test has several


def test_the_record_gives_each_p_value_of_a_test_vs_the_reference():
    from test_results import _baseline_batch  # the regression baseline's blot

    res = compute_results(_baseline_batch("vehicle"))
    applied, every = results_statistics(res)["charts"]
    for pinned, one in ((applied, res), (every, res.all_lanes)):
        test = one.series[0].chart.test
        assert pinned["test"] == test.id
        assert (pinned["p_value"], pinned["statistic"]) == (None, None)  # no omnibus test
        assert pinned["pairwise"] == [
            {
                "group_a": p.group_a,
                "group_b": p.group_b,
                "p_value": p.p_value,
                "estimate": p.estimate,
            }
            for p in test.pairwise
        ]
        assert [p["group_a"] for p in pinned["pairwise"]] == ["vehicle", "vehicle"]
    assert (applied["test"], every["test"]) == ("dunnett", "welch_t_holm")
    json.dumps(results_statistics(res), allow_nan=False)


def test_the_record_says_how_a_mann_whitney_p_was_computed():
    res = compute_results(_baseline_like_batch(), statistics={"family": "rank"})
    [chart] = results_statistics(res)["charts"]
    assert (chart["test"], chart["method"]) == ("mann_whitney", "exact")


def _baseline_like_batch():
    """The sample project with a β-catenin box in lane 3 and lane 4 included:
    vehicle and 10 µM, two replicates each."""

    def change(draft: Project) -> None:
        draft.batch.lanes[3].included = True
        beta = draft.batch.find_protein("prot-7")
        extra = beta.bands[0].model_copy(update={"id": draft.new_id("band"), "lane_index": 2})
        extra.box = extra.box.model_copy(update={"x": 98})
        beta.bands.append(extra)

    return apply_change(make_project(), change)[0].batch


def test_the_record_statistics_have_the_setting_even_without_charts():
    res = compute_results(make_project().batch, statistics=StatisticsSetting(family="none"))
    pinned = results_statistics(res)
    assert pinned["setting"]["family"] == "none"
    assert all(chart["test"] is None for chart in pinned["charts"])
    assert pinned["charts"][0]["note"] == "no test: statistics are turned off"
