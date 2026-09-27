# SPDX-License-Identifier: Apache-2.0
"""The export bundle (#53): one action writes the lane tables, the charts and
the reproducibility record of a project into a new folder of its own.

The project is the conftest sample, saved with stand-in image files in a folder
with a non-ASCII name: β-catenin (target) normalized to α-tubulin, GAPDH a
second loading control, lanes vehicle, vehicle, 10 µM and 10 µM with vehicle
the reference, and lane 4 excluded while β-catenin has a value there, so the
results have two sets (#71).
"""

import codecs
import csv
import hashlib
import io
import json
import os
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from conftest import FakeClock, make_project, make_project_with_clashing_names, write_image_files
from proteia import viz
from proteia.core import operations as ops
from proteia.core import record, storage
from proteia.core.analyze import compare, describe
from proteia.core.export import (
    BUNDLE_README_FILE,
    BUNDLE_RECORD_FILE,
    CHART_PNG_DPI,
    DEFAULT_CHART_FORMATS,
    LANE_TABLE_DECIMALS,
    LANE_TABLE_RATIO_DECIMALS,
    MIN_NAME_ROOM,
    ChartFormat,
    bundle_folder_name,
    file_stem,
    unique_stems,
)
from proteia.core.model import Project, Role, apply_change
from proteia.core.operations import ErrorCode, LaneInput, OperationError, ProjectSession
from proteia.core.plotspec import ErrorType, ValueKind, build_plotspec
from proteia.core.session import save_to_folder
from proteia.viz import render_figure, render_pdf, render_png, render_svg

FOLDER = "專案 µ α β"
START = datetime(2026, 9, 26, 8, 0, tzinfo=UTC)  # FakeClock's first time
APPLIED, ALL = "Excluding lane 4", "All lanes"
SERIES = "β-catenin ÷ α-tubulin"
FILES = [
    f"lane-table ({APPLIED}).csv",
    f"lane-table ({ALL}).csv",
    f"chart {SERIES} ({APPLIED}).svg",
    f"chart {SERIES} ({APPLIED}).png",
    f"chart {SERIES} ({ALL}).svg",
    f"chart {SERIES} ({ALL}).png",
    BUNDLE_README_FILE,
    BUNDLE_RECORD_FILE,
]


class Recorder:
    """An autosave hook that records ``last_action`` and then saves."""

    def __init__(self) -> None:
        self.actions: list[str | None] = []

    def __call__(self, session: ProjectSession) -> None:
        self.actions.append(session.last_action)
        save_to_folder(session)


def open_sample(
    tmp_path, hook=None, project: Project | None = None, *, folder: Path | None = None, **clock
) -> ProjectSession:
    folder = tmp_path / FOLDER if folder is None else folder
    project = make_project() if project is None else project
    write_image_files(folder, project)
    storage.save_project(project, folder)
    return ops.open_project(folder, autosave=hook, clock=FakeClock(**clock))


def local_name(moment: datetime) -> str:
    """The folder name of an export at ``moment``: the local date and time."""
    return moment.astimezone().strftime("%Y-%m-%d %H%M")


def exports(session: ProjectSession) -> list[str]:
    folder = session.folder / storage.EXPORTS_DIR
    return sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []


def read_rows(data: bytes) -> list[dict[str, str]]:
    assert data.startswith(codecs.BOM_UTF8)
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def files_of(bundle: ops.ExportBundle) -> dict[str, bytes]:
    return {name: (bundle.folder / name).read_bytes() for name in bundle.files}


def readme_text(bundle: ops.ExportBundle) -> str:
    """The bundle's README with its lines joined: each run of whitespace one space."""
    return " ".join((bundle.folder / BUNDLE_README_FILE).read_text(encoding="utf-8").split())


def ratios(values) -> list[str]:
    """Normalized values or fold changes as a lane table's cells hold them."""
    return ["" if v is None else str(round(v, LANE_TABLE_RATIO_DECIMALS)) for v in values]


# --- The folder and its files ---


def test_an_export_writes_every_file_of_both_sets_into_a_new_folder(tmp_path):
    recorder = Recorder()
    s = open_sample(tmp_path, recorder)
    project_json = (s.folder / storage.PROJECT_FILE).read_bytes()
    before = s.project

    bundle = ops.export_bundle(s)

    assert bundle.folder == s.folder / storage.EXPORTS_DIR / local_name(START)
    assert list(bundle.files) == FILES
    assert sorted(path.name for path in bundle.folder.iterdir()) == sorted(FILES)  # no temp file
    assert all((bundle.folder / name).stat().st_size > 0 for name in FILES)
    # Not a state change: no log entry, no autosave.
    assert recorder.actions == [] and s.project is before
    assert (s.folder / storage.PROJECT_FILE).read_bytes() == project_json


def test_the_default_formats_are_svg_and_png_at_300_dpi():
    # Provisional until the maintainer chooses (#53).
    assert DEFAULT_CHART_FORMATS == (ChartFormat.SVG, ChartFormat.PNG)
    assert CHART_PNG_DPI == 300
    assert [f.value for f in ChartFormat] == ["svg", "png", "pdf"]


def test_one_result_set_gives_one_table_and_unlabelled_charts(tmp_path):
    project = make_project()

    def include_every_lane(draft: Project) -> None:
        for lane in draft.batch.lanes:
            lane.included = True

    s = open_sample(tmp_path, project=apply_change(project, include_every_lane)[0])
    bundle = ops.export_bundle(s, formats=["pdf"])
    assert list(bundle.files) == [
        "lane-table.csv",
        f"chart {SERIES}.pdf",
        BUNDLE_README_FILE,
        BUNDLE_RECORD_FILE,
    ]


def test_the_lane_tables_number_lanes_from_1_and_hold_each_sets_values(tmp_path):
    s = open_sample(tmp_path)
    bundle = ops.export_bundle(s)
    res = ops.compute(s)
    for data, results in (
        ((bundle.folder / FILES[0]).read_bytes(), res),
        ((bundle.folder / FILES[1]).read_bytes(), res.all_lanes),
    ):
        rows = read_rows(data)
        assert list(rows[0]) == [
            "lane",
            "condition",
            "sample",
            "include",
            "β-catenin",
            "β-catenin clipped",
            "α-tubulin",
            "α-tubulin clipped",
            "GAPDH",
            "GAPDH clipped",
            f"{SERIES} normalized",
            f"{SERIES} fold change vs vehicle",
        ]
        assert [row["lane"] for row in rows] == ["1", "2", "3", "4"]
        assert [row["condition"] for row in rows] == ["vehicle", "vehicle", "10 µM", "10 µM"]
        assert [row["sample"] for row in rows] == ["v1", "v2", "a1", "a2"]
        # Each set's own include flags: the all-lanes set includes lane 4.
        assert [row["include"] == "yes" for row in rows] == [
            lane.included for lane in results.lanes
        ]
        for column in results.proteins:
            assert [row[column.name] for row in rows] == [
                "" if net is None else str(round(net, LANE_TABLE_DECIMALS)) for net in column.nets
            ]
        [series] = results.series
        assert [row[f"{SERIES} normalized"] for row in rows] == [
            "" if v is None else str(round(v, LANE_TABLE_RATIO_DECIMALS)) for v in series.normalized
        ]
        assert [row[f"{SERIES} fold change vs vehicle"] for row in rows] == [
            "" if v is None else str(round(v, LANE_TABLE_RATIO_DECIMALS))
            for v in series.fold_change
        ]
    assert [row["include"] for row in read_rows((bundle.folder / FILES[0]).read_bytes())] == [
        "yes",
        "yes",
        "yes",
        "no",
    ]


def test_each_sets_files_hold_that_sets_own_values(tmp_path):
    # Lane 2, a reference lane, excluded too: the set that applies the
    # exclusions forms its fold changes against lane 1 alone, the all-lanes set
    # against lanes 1 and 2, so each file has numbers of its own to share (#71).
    def exclude_a_reference_lane(draft: Project) -> None:
        draft.batch.lanes[1].included = False

    s = open_sample(tmp_path, project=apply_change(make_project(), exclude_a_reference_lane)[0])
    bundle = ops.export_bundle(s, formats=["svg"])
    res = ops.compute(s)
    assert (res.label, res.all_lanes.label) == ("Excluding lanes 2, 4", ALL)
    column = f"{SERIES} fold change vs vehicle"
    written = []
    for results in (res, res.all_lanes):
        rows = read_rows((bundle.folder / f"lane-table ({results.label}).csv").read_bytes())
        [series] = results.series
        assert [row["include"] == "yes" for row in rows] == [
            lane.included for lane in results.lanes
        ]
        assert [row[f"{SERIES} normalized"] for row in rows] == ratios(series.normalized)
        assert [row[column] for row in rows] == ratios(series.fold_change)
        written.append([row[column] for row in rows])
        chart = (bundle.folder / f"chart {SERIES} ({results.label}).svg").read_bytes()
        assert chart == render_svg(series.chart)
    applied, every = written
    assert applied[0] == "1.0" != every[0]  # lane 1 is the applied set's whole reference
    assert applied != every


def test_a_project_with_no_reference_exports_normalized_values(tmp_path):
    # Until the user picks a reference condition, each series is normalized to
    # its loading control with no fold change: the tables have no fold-change
    # column, and the charts show the normalized values.
    def no_reference(draft: Project) -> None:
        draft.batch.reference_condition = None

    s = open_sample(tmp_path, project=apply_change(make_project(), no_reference)[0])
    bundle = ops.export_bundle(s)
    assert list(bundle.files) == FILES
    res = ops.compute(s)
    for label, results in ((APPLIED, res), (ALL, res.all_lanes)):
        [series] = results.series
        assert series.value_kind is ValueKind.LOADING_NORMALIZED and series.fold_change is None
        rows = read_rows((bundle.folder / f"lane-table ({label}).csv").read_bytes())
        assert list(rows[0])[-2:] == ["GAPDH clipped", f"{SERIES} normalized"]
        assert [row[f"{SERIES} normalized"] for row in rows] == ratios(series.normalized)
        chart = (bundle.folder / f"chart {SERIES} ({label}).svg").read_bytes()
        assert chart == render_svg(series.chart)
    text = readme_text(bundle)
    for label in (APPLIED, ALL):
        assert f'Chart of β-catenin / α-tubulin (normalized), set "{label}"; SVG' in text
    assert "fold change vs" not in text


def test_the_charts_are_the_screens_drawings_of_each_set(tmp_path):
    s = open_sample(tmp_path)
    bundle = ops.export_bundle(s, formats=["svg", "png", "pdf"])
    res = ops.compute(s)
    for label, results in ((APPLIED, res), (ALL, res.all_lanes)):
        [series] = results.series
        spec = series.chart
        assert spec is not None and spec.subtitle == label  # the set's label in the image
        stem = f"chart {SERIES} ({label})"

        svg = (bundle.folder / f"{stem}.svg").read_bytes()
        assert svg == render_svg(spec)  # the renderer the screen uses
        assert ET.fromstring(svg).tag == "{http://www.w3.org/2000/svg}svg"

        png = (bundle.folder / f"{stem}.png").read_bytes()
        assert png == render_png(spec, dpi=CHART_PNG_DPI)
        with Image.open(io.BytesIO(png)) as image:
            image.verify()
            assert image.format == "PNG"
            assert round(image.info["dpi"][0]) == CHART_PNG_DPI

        pdf = (bundle.folder / f"{stem}.pdf").read_bytes()
        assert pdf == render_pdf(spec)
        assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
        assert b"CreationDate" not in pdf  # the same bytes for the same chart


def test_a_chart_states_its_error_bars():
    groups = {"vehicle": [1.0, 1.1, 0.9], "10 µM": [2.0, 2.2, 1.9]}
    for error_type in ErrorType:
        spec = build_plotspec(
            groups,
            describe(groups),
            compare(groups),
            value_kind=ValueKind.FOLD_CHANGE,
            error_type=error_type,
            title="β-catenin / α-tubulin",
        )
        assert render_figure(spec).axes[0].get_xlabel() == f"Bars: mean ± {error_type.value}"


def test_a_png_and_a_pdf_are_the_same_bytes_every_time():
    groups = {"vehicle": [1.0, 1.1, 0.9], "10 µM": [2.0, 2.2, 1.9]}
    spec = build_plotspec(
        groups, describe(groups), compare(groups), value_kind=ValueKind.FOLD_CHANGE, title="β"
    )
    assert render_png(spec, dpi=72) == render_png(spec, dpi=72) != render_png(spec, dpi=150)
    assert render_pdf(spec) == render_pdf(spec)
    assert b"Matplotlib" not in render_png(spec, dpi=72)  # no software tag
    for key in (b"/Creator", b"/Producer", b"/CreationDate"):
        assert key not in render_pdf(spec)


def test_a_pdf_embeds_truetype_fonts():
    # Type 42 (TrueType, embedded as /FontFile2), not Type 3, which many
    # journals and vector editors refuse.
    groups = {"vehicle": [1.0, 1.1, 0.9], "10 µM": [2.0, 2.2, 1.9]}
    spec = build_plotspec(
        groups, describe(groups), compare(groups), value_kind=ValueKind.FOLD_CHANGE, title="β"
    )
    pdf = render_pdf(spec)
    assert b"/FontFile2" in pdf
    assert b"/Type3" not in pdf


# --- The record ---


def test_the_record_is_the_projects_current_record_and_lists_every_file(tmp_path):
    s = open_sample(tmp_path)
    bundle = ops.export_bundle(s, formats=["svg", "png", "pdf"])
    written = files_of(bundle)
    data = written.pop(BUNDLE_RECORD_FILE)
    doc = json.loads(data.decode("utf-8"))
    assert record.record_bytes(doc) == data  # the canonical file form

    assert doc["exported_at"] == "2026-09-26T08:00:00.000Z"
    # Every other file, with its hash: the record cannot list itself.
    assert doc["files"] == {
        name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        for name, content in written.items()
    }
    expected = record.build_record(
        s.project, exported_at=doc["exported_at"], files=written, results=ops.compute(s)
    )
    assert doc == json.loads(record.record_bytes(expected))
    assert doc["content_hash"] == storage.content_hash(s.project)
    assert doc["results"] == {
        "method": "mean",
        "error_type": "SD",
        "plot_conditions": None,
        "excluded_lanes": [3],
    }
    assert doc["settings"]["lane_table_ratio_decimals"] == LANE_TABLE_RATIO_DECIMALS
    assert doc["settings"]["lane_table_first_lane"] == 1
    assert doc["settings"]["chart_png_dpi"] == CHART_PNG_DPI
    text = data.decode("utf-8")
    for leak in (str(tmp_path), json.dumps(str(tmp_path))[1:-1], FOLDER):
        assert leak not in text


def test_the_export_uses_the_compute_settings_given(tmp_path):
    s = open_sample(tmp_path)
    bundle = ops.export_bundle(
        s, formats=["svg"], error_type="SEM", method="representative", plot_conditions=["vehicle"]
    )
    doc = json.loads((bundle.folder / BUNDLE_RECORD_FILE).read_bytes())
    assert doc["results"] == {
        "method": "representative",
        "error_type": "SEM",
        "plot_conditions": ["vehicle"],
        "excluded_lanes": [3],
    }
    res = ops.compute(
        s, error_type=ErrorType.SEM, method="representative", plot_conditions=["vehicle"]
    )
    svg = (bundle.folder / f"chart {SERIES} ({APPLIED}).svg").read_bytes()
    assert svg == render_svg(res.series[0].chart)
    # The README states the settings the charts were drawn with.
    text = readme_text(bundle)
    for words in (
        "each bar is the mean ± SEM of a condition's samples",
        "are represented by one lane each, the first.",
        "Plotted conditions: 'vehicle'.",
    ):
        assert words in text, words
    for words in ("mean ± SD", "averaged", "Plotted conditions: all"):
        assert words not in text, words


def test_the_readme_names_every_file(tmp_path):
    s = open_sample(tmp_path)
    bundle = ops.export_bundle(s)
    data = (bundle.folder / BUNDLE_README_FILE).read_bytes()
    assert not data.startswith(codecs.BOM_UTF8)
    text = data.decode("utf-8")
    lines = text.splitlines()
    for name in FILES:
        assert name in lines, name  # each file on a line of its own
    for words in (
        "2026-09-26T08:00:00.000Z",
        "numbered from 1",
        "mean ± SD",
        f'"{APPLIED}"',
        f'"{ALL}"',
        "SHA-256",
    ):
        assert words in text, words
    joined = readme_text(bundle)
    for words in (
        "each bar is the mean ± SD of a condition's samples",
        "are averaged.",
        "Plotted conditions: all.",
        f'Chart of β-catenin / α-tubulin (fold change vs vehicle), set "{APPLIED}"; SVG',
        f'Chart of β-catenin / α-tubulin (fold change vs vehicle), set "{ALL}"; PNG, 300 dpi.',
    ):
        assert words in joined, words
    assert "SEM" not in joined and "represented by one lane" not in joined


# --- Folders ---


def test_every_export_makes_a_new_folder_named_to_sort_in_time(tmp_path):
    s = open_sample(tmp_path, step=timedelta(seconds=25))
    made = [ops.export_bundle(s, formats=["svg"]).folder.name for _ in range(4)]
    base = local_name(START)
    assert made[:3] == [base, f"{base} (2)", f"{base} (3)"]  # 08:00:00, :25 and :50
    assert made[3] == local_name(START + timedelta(seconds=75))  # the next minute
    assert exports(s) == sorted(made) == made
    for name in made:
        assert sorted(p.name for p in (s.folder / "exports" / name).iterdir()) == sorted(
            [f"lane-table ({APPLIED}).csv", f"lane-table ({ALL}).csv", BUNDLE_README_FILE]
            + [BUNDLE_RECORD_FILE, f"chart {SERIES} ({APPLIED}).svg", f"chart {SERIES} ({ALL}).svg"]
        )


def test_an_existing_folder_is_never_overwritten(tmp_path):
    s = open_sample(tmp_path)
    taken = s.folder / "exports" / local_name(START)
    taken.mkdir(parents=True)
    (taken / "notes.txt").write_text("mine", encoding="utf-8")
    bundle = ops.export_bundle(s)
    assert bundle.folder.name == f"{local_name(START)} (2)"
    assert [p.name for p in taken.iterdir()] == ["notes.txt"]


def test_a_folder_name_is_the_local_minute():
    moment = datetime(2026, 9, 27, 7, 32, 59, 999000, tzinfo=UTC)
    assert bundle_folder_name(moment) == moment.astimezone().strftime("%Y-%m-%d %H%M")
    assert not set(bundle_folder_name(moment)) & set('<>:"/\\|?*')


# --- Refusals and failures ---


def test_an_export_without_lanes_is_refused_and_writes_nothing(tmp_path):
    s = ops.new_project(tmp_path / FOLDER, autosave=None, clock=FakeClock())
    with pytest.raises(OperationError) as info:
        ops.export_bundle(s)
    assert info.value.code is ErrorCode.NO_LANES
    assert exports(s) == []


@pytest.mark.parametrize("problem", ["changed", "missing"])
def test_an_export_with_changed_or_missing_images_is_refused(tmp_path, problem):
    s = open_sample(tmp_path)
    image = s.folder / "images" / "img-2.tif"
    if problem == "changed":
        image.write_bytes(b"other pixels")
    else:
        image.unlink()
    with pytest.raises(OperationError) as info:
        ops.export_bundle(s)
    assert (info.value.code, info.value.ids) == (ErrorCode.IMAGE_FILE_CHANGED, ("img-2",))
    assert exports(s) == []


@pytest.mark.parametrize(
    "formats",
    [["svg", "gif"], ["SVG"], "svg", [None], [ChartFormat.PNG, 3]],
    ids=["unknown", "upper-case", "one-string", "none", "number"],
)
def test_an_unknown_chart_format_is_refused(tmp_path, formats):
    s = open_sample(tmp_path)
    with pytest.raises(OperationError) as info:
        ops.export_bundle(s, formats=formats)
    assert info.value.code is ErrorCode.INVALID_INPUT
    assert exports(s) == []


def test_formats_are_written_once_each_in_the_order_given(tmp_path):
    s = open_sample(tmp_path)
    bundle = ops.export_bundle(s, formats=["pdf", ChartFormat.SVG, "pdf"])
    charts = [name for name in bundle.files if name.startswith("chart ")]
    assert charts == [
        f"chart {SERIES} ({APPLIED}).pdf",
        f"chart {SERIES} ({APPLIED}).svg",
        f"chart {SERIES} ({ALL}).pdf",
        f"chart {SERIES} ({ALL}).svg",
    ]
    tables_only = ops.export_bundle(s, formats=[])
    assert list(tables_only.files) == FILES[:2] + FILES[-2:]
    readme = (tables_only.folder / BUNDLE_README_FILE).read_text(encoding="utf-8")
    assert "No chart" in readme


def test_a_failed_write_removes_the_folder(tmp_path, monkeypatch):
    s = open_sample(tmp_path)
    write = storage.write_atomic
    calls = []

    def failing(path, data, **kw):
        calls.append(path.name)
        if len(calls) == 3:
            raise OSError("disk full")
        write(path, data, **kw)

    monkeypatch.setattr(storage, "write_atomic", failing)
    before = s.project
    with pytest.raises(OSError, match="disk full"):
        ops.export_bundle(s)
    assert len(calls) == 3
    assert exports(s) == []  # the folder and the two files in it are gone
    assert s.project is before


def test_a_failed_drawing_writes_nothing(tmp_path, monkeypatch):
    s = open_sample(tmp_path)

    def broken(spec, **kw):
        raise RuntimeError("cannot draw")

    monkeypatch.setattr(viz, "render_png", broken)
    with pytest.raises(RuntimeError, match="cannot draw"):
        ops.export_bundle(s)
    assert exports(s) == []


# --- Names ---


def test_names_keep_greek_and_micro_and_replace_what_windows_refuses():
    assert file_stem("β-catenin 10 µM") == "β-catenin 10 µM"
    assert file_stem('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"
    assert file_stem("tab\there\nnew") == "tab here new"  # whitespace collapsed
    assert file_stem("bell\x07 rlo‮gpj.exe zwsp​") == "bell_ rlo_gpj.exe zwsp_"
    assert file_stem("no￾ \ud800") == "no_ _"  # a noncharacter, an unpaired surrogate
    long = "α" * 100
    assert file_stem(long, limit=40) == "α" * 39 + "…"
    assert len(file_stem("x " * 60, limit=40)) <= 40


def test_colliding_names_get_a_number():
    stems = ["chart p_q ÷ α", "chart P_Q ÷ α", "chart p_q ÷ α", "chart other"]
    assert unique_stems(stems) == [
        "chart p_q ÷ α",
        "chart P_Q ÷ α (2)",  # names that differ only in case are one file on Windows
        "chart p_q ÷ α (3)",
        "chart other",
    ]
    assert unique_stems(["a", "a (2)", "a"]) == ["a", "a (2)", "a (3)"]


def test_protein_names_that_make_one_file_name_are_told_apart(tmp_path):
    def rename(draft: Project) -> None:
        beta, _, gapdh = draft.batch.proteins
        beta.name = "p/q"
        gapdh.name = "p:q"
        gapdh.role = Role.TARGET
        gapdh.loading_control_ids = ["prot-8"]

    s = open_sample(tmp_path, project=apply_change(make_project(), rename)[0])
    bundle = ops.export_bundle(s, formats=["svg"])
    charts = [name for name in bundle.files if name.startswith("chart ")]
    assert charts == [
        f"chart p_q ÷ α-tubulin ({APPLIED}).svg",
        f"chart p_q ÷ α-tubulin ({APPLIED}) (2).svg",
        f"chart p_q ÷ α-tubulin ({ALL}).svg",
        f"chart p_q ÷ α-tubulin ({ALL}) (2).svg",
    ]
    res = ops.compute(s)
    assert [series.target for series in res.series] == ["p/q", "p:q"]
    for name, series in zip(charts[:2], res.series, strict=True):
        assert (bundle.folder / name).read_bytes() == render_svg(series.chart)


def test_an_export_only_project_writes_its_nets_and_no_chart(tmp_path):
    s = ops.new_project(tmp_path / FOLDER, autosave=None, clock=FakeClock())
    ops.set_lanes(s, [LaneInput("-DOX"), LaneInput("+DOX", "=1+1", included=False)])
    bundle = ops.export_bundle(s)
    assert list(bundle.files) == ["lane-table.csv", BUNDLE_README_FILE, BUNDLE_RECORD_FILE]
    rows = read_rows((bundle.folder / "lane-table.csv").read_bytes())
    # Provisional (#53): text is written as typed, even where a spreadsheet
    # would read a formula.
    assert [(row["lane"], row["condition"], row["sample"], row["include"]) for row in rows] == [
        ("1", "-DOX", "", "yes"),
        ("2", "+DOX", "=1+1", "no"),
    ]
    readme = (bundle.folder / BUNDLE_README_FILE).read_text(encoding="utf-8")
    assert "No chart" in readme


# --- Column names ---


def table_rows(bundle: ops.ExportBundle, name: str) -> tuple[list[str], list[dict[str, str]]]:
    """A lane table's header and its rows by header, which must be unique."""
    data = (bundle.folder / name).read_bytes()
    header = next(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    assert len(set(header)) == len(header), header
    return header, read_rows(data)


def nets(values) -> list[str]:
    """Nets as a lane table's cells hold them."""
    return ["" if v is None else str(round(v, LANE_TABLE_DECIMALS)) for v in values]


def test_proteins_whose_columns_would_share_a_name_take_a_number(tmp_path):
    # "GAPDH clipped" names the clipping column of GAPDH: a project.json may
    # hold both names, though the operations refuse the second.
    s = open_sample(tmp_path, project=make_project_with_clashing_names())
    bundle = ops.export_bundle(s, formats=["svg"])
    renamed = "GAPDH clipped (2)"
    assert list(bundle.files) == [
        f"lane-table ({APPLIED}).csv",
        f"lane-table ({ALL}).csv",
        f"chart GAPDH ÷ α-tubulin ({APPLIED}).svg",
        f"chart GAPDH clipped ÷ α-tubulin ({APPLIED}).svg",  # file names are not renamed
        f"chart GAPDH ÷ α-tubulin ({ALL}).svg",
        f"chart GAPDH clipped ÷ α-tubulin ({ALL}).svg",
        BUNDLE_README_FILE,
        BUNDLE_RECORD_FILE,
    ]
    res = ops.compute(s)
    for name, results in ((bundle.files[0], res), (bundle.files[1], res.all_lanes)):
        header, rows = table_rows(bundle, name)
        assert header[4:] == [
            "GAPDH",
            "GAPDH clipped",
            "α-tubulin",
            "α-tubulin clipped",
            renamed,
            f"{renamed} clipped",
            "GAPDH ÷ α-tubulin normalized",
            "GAPDH ÷ α-tubulin fold change vs vehicle",
            f"{renamed} ÷ α-tubulin normalized",  # its series carry its column name
            f"{renamed} ÷ α-tubulin fold change vs vehicle",
        ]
        by_id = {column.protein_id: column for column in results.proteins}
        assert [by_id["prot-7"].name, by_id["prot-9"].name] == ["GAPDH", "GAPDH clipped"]
        assert [row["GAPDH"] for row in rows] == nets(by_id["prot-7"].nets)
        assert [row[renamed] for row in rows] == nets(by_id["prot-9"].nets)
        assert [row["GAPDH clipped"] for row in rows] == ["yes", "no", "", "no"]
        assert [row[f"{renamed} clipped"] for row in rows] == ["no", "yes", "", ""]
        first, second = results.series
        assert (first.target_id, second.target_id) == ("prot-7", "prot-9")
        assert [row["GAPDH ÷ α-tubulin normalized"] for row in rows] == ratios(first.normalized)
        assert [row[f"{renamed} ÷ α-tubulin normalized"] for row in rows] == ratios(
            second.normalized
        )
        assert [row[f"{renamed} ÷ α-tubulin fold change vs vehicle"] for row in rows] == ratios(
            second.fold_change
        )

    assert (
        "Renamed columns: no two columns of a lane table share a name. Where two would,"
        " one takes a number."
        ' The protein "GAPDH clipped" is named "GAPDH clipped (2)" in its columns,'
        ' "GAPDH clipped (2)" and "GAPDH clipped (2) clipped", and in those of its series:'
        ' the column "GAPDH clipped" holds the clipping flags of the protein "GAPDH".'
    ) in readme_text(bundle)
    # The record lists every file as written, the README among them.
    written = files_of(bundle)
    doc = json.loads(written.pop(BUNDLE_RECORD_FILE))
    assert doc["files"] == {
        name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        for name, content in written.items()
    }
    assert doc["content_hash"] == storage.content_hash(s.project)


def test_a_protein_named_like_a_series_column_gives_way_to_it(tmp_path):
    s = open_sample(tmp_path)
    ops.edit_protein(s, "prot-9", name=f"{SERIES} normalized")  # a name the operations take
    bundle = ops.export_bundle(s, formats=[])
    res = ops.compute(s)
    for name, results in ((bundle.files[0], res), (bundle.files[1], res.all_lanes)):
        header, rows = table_rows(bundle, name)
        assert header[4:] == [
            "β-catenin",
            "β-catenin clipped",
            "α-tubulin",
            "α-tubulin clipped",
            f"{SERIES} normalized (2)",
            f"{SERIES} normalized (2) clipped",
            f"{SERIES} normalized",
            f"{SERIES} fold change vs vehicle",
        ]
        [series] = results.series
        assert [row[f"{SERIES} normalized"] for row in rows] == ratios(series.normalized)
        [gapdh] = [column for column in results.proteins if column.protein_id == "prot-9"]
        assert [row[f"{SERIES} normalized (2)"] for row in rows] == nets(gapdh.nets)
    assert (
        f'The protein "{SERIES} normalized" is named "{SERIES} normalized (2)" in its'
        f' columns, "{SERIES} normalized (2)" and "{SERIES} normalized (2) clipped": the'
        f' column "{SERIES} normalized" holds the normalized values of the target'
        ' "β-catenin" over the loading control "α-tubulin".'
    ) in readme_text(bundle)


def test_two_series_that_one_name_would_name_are_told_apart(tmp_path):
    s = open_sample(tmp_path)  # names the operations take, holding " ÷ "
    ops.edit_protein(s, "prot-7", name="X ÷ Y")  # the target of prot-8
    ops.edit_protein(s, "prot-8", name="Z")
    ops.edit_protein(s, "prot-9", name="Y ÷ Z")
    ops.add_protein(s, "X", Role.TARGET, "img-4", loading_control_ids=["prot-9"])
    bundle = ops.export_bundle(s, formats=[])
    header, rows = table_rows(bundle, bundle.files[0])
    assert header[12:] == [
        "X ÷ Y ÷ Z normalized",
        "X ÷ Y ÷ Z fold change vs vehicle",
        "X ÷ Y ÷ Z normalized (2)",  # X over Y ÷ Z: no value, no fold change
    ]
    first, _ = ops.compute(s).series
    assert [row["X ÷ Y ÷ Z normalized"] for row in rows] == ratios(first.normalized)
    assert [row["X ÷ Y ÷ Z normalized (2)"] for row in rows] == [""] * 4
    assert (
        'The normalized values of the target "X" over the loading control "Y ÷ Z" are in'
        ' the column "X ÷ Y ÷ Z normalized (2)"'
    ) in readme_text(bundle)


# --- Long paths ---

# Antibody names as users type them, longer than a file name shows whole.
LONG_TARGET = "phospho-p44/42 MAPK (Erk1/2) (Thr202/Tyr204)"
LONG_LOADING = "GAPDH (D16H11) XP Rabbit mAb"
# Windows' longest path while long paths are off, its default: MAX_PATH less its NUL.
WINDOWS_PATH_LIMIT = 259
# The export folder's longest name (the minute, then " (1000)") between the
# project folder and a file.
WIDEST_FOLDER = f"{os.sep}exports{os.sep}{local_name(START)} (1000){os.sep}"
# The longest project path an export takes under that limit: the file names
# need room for MIN_NAME_ROOM characters and write_atomic's temp file adds
# ".<name>.<8 random characters>.tmp" around each.
LONGEST_PROJECT_PATH = (
    WINDOWS_PATH_LIMIT - len(WIDEST_FOLDER) - MIN_NAME_ROOM - len("..12345678.tmp")
)


def long_named(draft: Project) -> None:
    beta, alpha, _ = draft.batch.proteins
    beta.name, alpha.name = LONG_TARGET, LONG_LOADING


def folder_of_length(tmp_path, length: int) -> Path:
    """A folder in ``tmp_path`` whose absolute path has ``length`` characters."""
    base = os.path.abspath(tmp_path)
    return Path(base) / ("p" * (length - len(base) - 1))


def temp_path_length(folder: Path, name: str) -> int:
    """The length, as Windows counts it (UTF-16 code units), of the path of the
    temp file write_atomic writes ``name`` through."""
    temp = os.path.join(os.path.abspath(folder), f".{name}.12345678.tmp")
    return len(temp.encode("utf-16-le")) // 2


@pytest.mark.parametrize(
    "length",
    # A 100-character project name (the most allowed) in a 39-character
    # projects root (D:\CloudSync\OneDrive\Documents\Proteia); the longest.
    [140, LONGEST_PROJECT_PATH],
)
def test_long_names_under_a_long_project_path_are_cut_to_fit(tmp_path, monkeypatch, length):
    # The limit is Windows' own (long paths off), applied here on every system.
    monkeypatch.setattr(storage, "PATH_LIMIT", WINDOWS_PATH_LIMIT)
    project = apply_change(make_project(), long_named)[0]
    s = open_sample(tmp_path, project=project, folder=folder_of_length(tmp_path, length))
    assert len(os.path.abspath(s.folder)) == length

    bundle = ops.export_bundle(s, formats=["svg", "png", "pdf"])

    assert sorted(p.name for p in bundle.folder.iterdir()) == sorted(bundle.files)
    for name in bundle.files:
        assert temp_path_length(bundle.folder, name) <= WINDOWS_PATH_LIMIT, name
    assert bundle.files[:2] == tuple(FILES[:2]) and bundle.files[-2:] == tuple(FILES[-2:])
    charts = bundle.files[2:-2]
    assert [name.rsplit(".", 1)[1] for name in charts] == ["svg", "png", "pdf"] * 2
    assert all(name.startswith("chart phospho-") and "…" in name for name in charts)
    assert charts[0].rsplit(".", 1)[0] != charts[3].rsplit(".", 1)[0]  # the two sets
    if length == 140:  # room enough for each set's label whole
        assert f"({APPLIED})." in charts[0] and f"({ALL})." in charts[3]
    res = ops.compute(s)
    assert (bundle.folder / charts[0]).read_bytes() == render_svg(res.series[0].chart)
    assert (bundle.folder / charts[3]).read_bytes() == render_svg(res.all_lanes.series[0].chart)


def test_a_project_path_too_long_for_an_export_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "PATH_LIMIT", WINDOWS_PATH_LIMIT)
    s = open_sample(tmp_path, folder=folder_of_length(tmp_path, LONGEST_PROJECT_PATH + 1))
    with pytest.raises(OperationError) as info:
        ops.export_bundle(s, formats=[])
    assert info.value.code is ErrorCode.PATH_TOO_LONG
    assert exports(s) == []


@pytest.mark.filterwarnings("ignore:Glyph .* missing from font")  # the chart's font has no CJK
def test_names_are_cut_to_the_name_limit_in_bytes(tmp_path, monkeypatch):
    # Most file systems count a name's UTF-8 bytes (255 at most), so a name in
    # a script of three bytes a character is cut sooner; a lower limit here.
    monkeypatch.setattr(storage, "PATH_LIMIT", None)
    monkeypatch.setattr(storage, "NAME_LIMIT_BYTES", 120)

    def cjk_named(draft: Project) -> None:
        beta, alpha, _ = draft.batch.proteins
        beta.name, alpha.name = "磷酸化蛋白" * 6, "微管蛋白" * 6

    s = open_sample(tmp_path, project=apply_change(make_project(), cjk_named)[0])
    bundle = ops.export_bundle(s, formats=["svg"])
    assert sorted(p.name for p in bundle.folder.iterdir()) == sorted(bundle.files)
    for name in bundle.files:
        assert len(f".{name}.12345678.tmp".encode()) <= 120, name
    charts = [name for name in bundle.files if name.startswith("chart ")]
    assert len(charts) == 2 and all("…" in name for name in charts)
