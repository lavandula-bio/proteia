# SPDX-License-Identifier: Apache-2.0
"""A check of an installation: ``proteia --self-test`` (#54, ADR 0003).

It takes the analysis through every path the app offers, in a new temporary
folder, and runs the local web app on a loopback port, then exits 0 when every
step gave what it should, or 1, naming each step that failed. The installer
build runs it inside the frozen bundle, where a module or a package's metadata
that PyInstaller left out fails a step instead of going unnoticed, and compares
its ``numbers`` with an unfrozen run (``packaging/windows/build.py``).

It writes only in the temporary folder (and where the libraries keep their
caches, such as matplotlib's font list): no project in ``Documents/Proteia``, no
file in the per-user state folder, no browser (the launcher is given an opener
that only records the address), and no connection but to its own loopback
port. The names it gives folders, projects and files include µ, α and β, as
users' names may.

The steps, each run even when an earlier one failed:

* ``imports``: the analysis stack, the chart backends (Agg, SVG, PDF), the web
  stack and every Proteia module the app uses; in a frozen bundle, that napari,
  Qt and tkinter are not there (ADR 0003);
* ``versions``: every library version the reproducibility record reads
  (:func:`~proteia.core.record.software_versions`), none missing;
* ``images``: every way an image is read (:func:`~proteia.core.imaging.load_image`):
  16- and 8-bit TIFF, deflate and LZW TIFF, PNG, JPEG, and CMYK as a separated
  TIFF and as a JPEG with an embedded profile;
* ``sample analysis``: the synthetic sample (:mod:`proteia.samples`) imported
  from files, lanes, a loading control and a target, a row box over each row,
  the results, whose fold changes must come within
  :data:`FOLD_CHANGE_TOLERANCE` of the truth; the chart as SVG, PNG and PDF, the
  lane table, and an export folder whose record names every library version
  and every file it holds, with its SHA-256;
* ``statistics``: every registered test (:data:`~proteia.core.analyze.TESTS`)
  on fixed values;
* ``web app``: the launcher's server with a new token; ``/api/status`` with the
  token (200) and without it (401), the page shell and every static file it
  loads, the sample project through the API, a row box over each row, the
  charts, the images' previews, an export, and Quit, after which the server stops
  and the launcher's files are gone.

``--json PATH`` writes the report there too (as UTF-8); ``--keep`` leaves the
temporary folder. The report's ``numbers`` are what the analysis computed, the
same in every installation of one build on one machine; ``info`` holds the
rest (sizes, times, paths).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import http.client
import importlib
import importlib.util
import json
import platform
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import proteia

FOLDER: Final = "樣本 µ α β"  # the samples' folder
PROJECT: Final = "µ α β"  # short: an export must fit in 259 characters
# The per-lane fold changes come within 1.5 % of the truth (tests/test_samples.py
# checks them with this margin).
FOLD_CHANGE_TOLERANCE: Final = 0.025
SERVER_WAIT: Final = 30.0  # seconds for the server to start, and to stop after Quit

# Imported by the ``imports`` step: what the analysis and the web app load,
# some of it only when first used (a chart format, a file type).
MODULES: Final = (
    "numpy",
    "scipy.ndimage",
    "scipy.signal",
    "scipy.special",
    "scipy.stats",
    "skimage.io",
    "skimage.morphology",
    "skimage.segmentation",
    "PIL.Image",
    "PIL.ImageCms",
    "tifffile",
    "matplotlib",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_svg",
    "fastapi",
    "starlette",
    "uvicorn",
    "pydantic",
    "proteia.core.analyze",
    "proteia.core.boxes",
    "proteia.core.export",
    "proteia.core.grow",
    "proteia.core.imaging",
    "proteia.core.model",
    "proteia.core.names",
    "proteia.core.operations",
    "proteia.core.plotspec",
    "proteia.core.project",
    "proteia.core.quantify",
    "proteia.core.record",
    "proteia.core.results",
    "proteia.core.rowdetect",
    "proteia.core.session",
    "proteia.core.storage",
    "proteia.samples",
    "proteia.viz",
    "proteia.web.api",
    "proteia.web.charts",
    "proteia.web.launch",
    "proteia.web.projects",
    "proteia.web.results_view",
    "proteia.web.sample_project",
    "proteia.web.server",
    "proteia.web.state",
)
# Not in a frozen bundle (ADR 0003): the napari GUI and what only it needs.
LEFT_OUT: Final = ("proteia.gui", "napari", "PySide6", "shiboken6", "qtpy", "tkinter")

# Fixed values for the statistics step: two or three conditions of 4 values, and
# two with tied values (the Mann-Whitney U test's permutation path).
_GROUPS: Final = {
    "A": [1.00, 1.20, 0.90, 1.10],
    "B": [2.10, 1.80, 2.40, 1.90],
    "C": [1.50, 1.40, 1.70, 1.65],
}
_TIED: Final = {"A": [1.0, 1.0, 2.0, 3.0], "B": [2.0, 4.0, 4.0, 5.0]}

Report = dict[str, Any]


class CheckError(AssertionError):
    """A step gave something other than what it should."""


def check(condition: object, message: str) -> None:
    if not condition:
        raise CheckError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- The steps ---


def _imports(folder: Path, report: Report) -> None:
    for name in MODULES:
        importlib.import_module(name)
    if getattr(sys, "frozen", False):
        present = [name for name in LEFT_OUT if _importable(name)]
        check(not present, f"the bundle holds modules it should leave out: {present}")


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _versions(folder: Path, report: Report) -> None:
    from proteia.core import record

    versions = record.software_versions()
    report["versions"] = versions
    missing = sorted(name for name, version in versions.items() if version is None)
    check(not missing, f"no version found for {missing} (package metadata left out?)")


def _image_files(folder: Path) -> dict[str, tuple[str, Callable[[Path], None]]]:
    """Each image case: its file name and how to write it."""
    import numpy as np
    import tifffile
    from PIL import Image, ImageCms

    ramp16 = (np.arange(64 * 48, dtype=np.uint32).reshape(48, 64) * 21).astype(np.uint16)
    ramp8 = (np.arange(64 * 48).reshape(48, 64) % 251).astype(np.uint8)
    rgb = np.stack([ramp8, ramp8[::-1], np.flipud(ramp8)], axis=-1)
    inks = np.stack([ramp8, ramp8 // 2, ramp8 // 3, ramp8 // 4], axis=-1)
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    def pillow(image: Image.Image, **options: Any) -> Callable[[Path], None]:
        return lambda path: image.save(path, **options)

    return {
        "tiff 16-bit": ("gray 16 µ.tif", lambda path: tifffile.imwrite(path, ramp16)),
        "tiff deflate": (
            "gray 8 deflate α.tif",
            lambda path: tifffile.imwrite(path, ramp8, compression="zlib"),
        ),
        "tiff lzw": ("gray 8 lzw β.tif", pillow(Image.fromarray(ramp8), compression="tiff_lzw")),
        "png 16-bit": ("gray 16 µ.png", pillow(Image.fromarray(ramp16))),
        "png rgb": ("rgb α.png", pillow(Image.fromarray(rgb))),
        "jpeg": ("gray β.jpg", pillow(Image.fromarray(ramp8), quality=95)),
        "tiff cmyk": (
            "cmyk µ.tif",
            lambda path: tifffile.imwrite(path, inks, photometric="separated"),
        ),
        # An RGB profile in a CMYK file: read, found unusable, and the formula used.
        "jpeg cmyk": (
            "cmyk α.jpg",
            pillow(Image.fromarray(rgb).convert("CMYK"), quality=95, icc_profile=srgb),
        ),
    }


def _images(folder: Path, report: Report) -> None:
    from proteia.core import imaging

    target = folder / "images µ"
    target.mkdir()
    for case, (name, write) in _image_files(folder).items():
        path = target / name
        write(path)
        loaded = imaging.load_image(path)
        check(loaded.array.ndim == 2 and loaded.array.size > 0, f"{case}: no analysis array")
        report["numbers"][f"image {case}"] = {
            "pixels": [str(loaded.pixels.dtype), list(loaded.pixels.shape)],
            "pixels_sha256": _sha256(loaded.pixels.tobytes()),
            "array_sha256": _sha256(loaded.array.tobytes()),
            "bit_depth": loaded.bit_depth,
            "warnings": sorted(warning.code for warning in loaded.warnings),
        }
    cmyk = report["numbers"]["image jpeg cmyk"]["warnings"]
    check("cmyk_converted" in cmyk, f"jpeg cmyk: not recorded as converted ({cmyk})")


def _sample_analysis(folder: Path, report: Report) -> None:
    from proteia import samples
    from proteia.core import operations as ops
    from proteia.core.analyze import Tier
    from proteia.core.export import BUNDLE_RECORD_FILE, CHART_PNG_DPI
    from proteia.core.model import ImageKind, Polarity, Role
    from proteia.core.operations import LaneInput
    from proteia.viz import render_pdf, render_png, render_svg
    from proteia.web.sample_project import sample_rows

    blot_path, marker_path, truth_path = samples.generate(folder / FOLDER)
    session = ops.new_project(folder / "projects" / PROJECT, autosave=None)
    dark = Polarity.DARK_ON_LIGHT
    with blot_path.open("rb") as f:
        blot = ops.import_image(
            session, f, blot_path.name, kind=ImageKind.CHEMILUMINESCENCE, polarity=dark
        )
    [membrane] = session.project.batch.membranes
    with marker_path.open("rb") as f:
        ops.import_image(
            session,
            f,
            marker_path.name,
            kind=ImageKind.VISIBLE_MARKER,
            polarity=dark,
            membrane_id=membrane.id,
        )
    lanes = [LaneInput(c, s) for c, s in zip(samples.CONDITIONS, samples.SAMPLES, strict=True)]
    ops.set_lanes(session, lanes, reference_condition=samples.REFERENCE)
    loading = ops.add_protein(session, samples.LOADING_CONTROL, Role.LOADING_CONTROL, blot)
    target = ops.add_protein(
        session, samples.TARGET, Role.TARGET, blot, loading_control_ids=[loading]
    )
    ids = {samples.LOADING_CONTROL: loading, samples.TARGET: target}
    for row in sample_rows():
        placement = ops.detect_row_boxes(session, ids[row.protein], row.rect)
        check(None not in placement.band_ids, f"{row.protein}: a lane has no band")

    results = ops.compute(session)
    check(results.tier is Tier.FOLD_CHANGE, f"results tier {results.tier}, not fold change")
    [series] = results.series
    with truth_path.open(encoding="utf-8-sig", newline="") as f:
        truth = [float(row["fold change vs vehicle"]) for row in csv.DictReader(f)]
    check(series.fold_change is not None, "no fold changes")
    errors = [abs(fc / t - 1) for fc, t in zip(series.fold_change or [], truth, strict=True)]
    worst = max(errors)
    check(worst <= FOLD_CHANGE_TOLERANCE, f"a fold change is {worst:.1%} off the truth")
    chart = series.chart
    check(chart is not None and chart.test_p is not None, "the chart has no test")
    report["numbers"]["sample"] = {
        "normalized": series.normalized,
        "fold_change": series.fold_change,
        "test": chart.test_name,
        "test_p": chart.test_p,
    }

    svg, png, pdf = render_svg(chart), render_png(chart, dpi=CHART_PNG_DPI), render_pdf(chart)
    check(svg.startswith(b"<?xml") and b"<svg" in svg, "the SVG chart is not SVG")
    check(png.startswith(b"\x89PNG\r\n\x1a\n"), "the PNG chart is not PNG")
    check(pdf.startswith(b"%PDF-"), "the PDF chart is not PDF")
    table = ops.export_lane_table(session)
    rows = table.read_text(encoding="utf-8-sig").splitlines()
    check(len(rows) == 1 + samples.LANES, f"the lane table has {len(rows)} lines")

    bundle = ops.export_bundle(session, formats=("svg", "png", "pdf"))
    record = json.loads((bundle.folder / BUNDLE_RECORD_FILE).read_bytes())
    missing = sorted(name for name, version in record["software"].items() if version is None)
    check(not missing, f"the export record has no version for {missing}")
    _check_record_files(bundle.folder, record)
    # The renderer writes the same bytes for the same chart (no date or version).
    report["numbers"]["sample"]["charts_sha256"] = [_sha256(data) for data in (svg, png, pdf)]
    report["info"]["sample"] = {
        "chart_bytes": {"svg": len(svg), "png": len(png), "pdf": len(pdf)},
        "export_files": list(bundle.files),
    }


def _check_record_files(folder: Path, record: dict[str, Any]) -> None:
    """Every file the record lists is in ``folder``, with the SHA-256 it gives."""
    listed: dict[str, dict[str, Any]] = record["files"]
    check(listed, "the export record lists no file")
    for name, entry in listed.items():
        path = folder / name
        check(path.is_file(), f"the export has no {name!r}")
        check(_sha256(path.read_bytes()) == entry["sha256"], f"{name!r} is not what was recorded")


def _statistics(folder: Path, report: Report) -> None:
    from proteia.core import analyze
    from proteia.core.analyze import StatisticsSetting, TestComparisons, TestScale

    two = {name: _GROUPS[name] for name in ("A", "B")}
    cases = [
        (spec, _GROUPS if spec.conditions == "three_or_more" else two)
        for spec in analyze.TESTS.values()
    ]
    cases.append((analyze.TESTS["mann_whitney"], _TIED))
    for spec, groups in cases:
        setting = StatisticsSetting(
            family=spec.family,
            comparisons=spec.comparisons or TestComparisons.ALL_PAIRS,
            scale=TestScale.LINEAR,
        )
        result = analyze.compare(groups, setting, ratio=False, reference="A")
        label = f"{spec.id} ({result.method})" if result.method else spec.id
        check(result.test == spec.id, f"{spec.id}: ran {result.test} ({result.note})")
        p_values = [pair.p_value for pair in result.pairwise]
        if result.p_value is not None:
            p_values.append(result.p_value)
        check(p_values, f"{label}: no p-value")
        check(all(0.0 <= p <= 1.0 for p in p_values), f"{label}: p-values {p_values}")
        report["numbers"][f"test {label}"] = {
            "statistic": result.statistic,
            "p": result.p_value,
            "pairwise": [[pair.group_a, pair.group_b, pair.p_value] for pair in result.pairwise],
        }


class _Client:
    """Requests to the server at ``port``, with ``token`` unless told otherwise."""

    def __init__(self, port: int, token: str) -> None:
        self.port = port
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        body: object = None,
        *,
        token: bool = True,
        expect: int = 200,
    ) -> tuple[bytes, str]:
        """The answer's body and content type; ``CheckError`` for another status."""
        headers = {"Authorization": f"Bearer {self.token}"} if token else {}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        try:
            conn.request(method, path, body=data, headers=headers)
            response = conn.getresponse()
            payload = response.read()
        finally:
            conn.close()
        check(
            response.status == expect,
            f"{method} {path}: {response.status}, not {expect}: {payload[:300]!r}",
        )
        return payload, response.getheader("content-type", "")

    def json(self, method: str, path: str, body: object = None, *, expect: int = 200) -> Any:
        payload, kind = self.request(method, path, body, expect=expect)
        check(kind.startswith("application/json"), f"{method} {path}: {kind}, not JSON")
        return json.loads(payload)


_STATIC_REFERENCE: Final = re.compile(r"""["'](/static/[A-Za-z0-9_./-]+)["']""")


def _web_app(folder: Path, report: Report) -> None:
    from proteia.core.export import BUNDLE_RECORD_FILE
    from proteia.web import api, launch

    opened: list[str] = []
    revealed: list[Path] = []
    root = folder / "web µ"
    state = folder / "state α"
    workspace = api.Workspace(root, reveal=revealed.append)
    instance = launch.start(folder=state, opener=opened.append, workspace=workspace)
    check(instance is not None, "another instance holds the lock in a new folder")
    assert instance is not None
    thread = threading.Thread(target=instance.serve, name="proteia self-test", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + SERVER_WAIT
        while not instance.server.started:
            check(thread.is_alive() and time.monotonic() < deadline, "the server did not start")
            time.sleep(0.01)
        client = _Client(instance.port, instance.token)
        _web_checks(client, report)
        _web_flow(client, root, report, BUNDLE_RECORD_FILE)
        check(opened == [instance.redirect_path.as_uri()], f"the browser was given {opened}")
        check(not revealed, "a folder was shown in the file manager")
        client.json("POST", "/api/quit", expect=202)
        thread.join(SERVER_WAIT)
        check(not thread.is_alive(), "the server did not stop after Quit")
        left = sorted(path.name for path in state.iterdir())
        check(left == [launch.LOCK_FILE], f"the launcher left {left} behind")
    finally:
        if thread.is_alive():
            instance.stop()
            thread.join(SERVER_WAIT)


def _web_checks(client: _Client, report: Report) -> None:
    """The token, the status, and every file the page loads."""
    from proteia.web.launch import InstanceInfo, probe

    status = client.json("GET", "/api/status")
    # The fields that name the app; the status may carry others (``handoff``).
    named = {key: status.get(key) for key in ("app", "version")}
    check(named == {"app": "proteia", "version": proteia.__version__}, f"status {status}")
    client.request("GET", "/api/status", token=False, expect=401)
    check(probe(InstanceInfo(0, client.port, client.token)), "the launcher's probe failed")
    shell, kind = client.request("GET", "/", token=False)
    check(kind.startswith("text/html"), f"the page shell is {kind}")
    seen: set[str] = set()
    todo = _STATIC_REFERENCE.findall(shell.decode("utf-8"))
    check(todo, "the page shell loads no static file")
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        body, _ = client.request("GET", path, token=False)
        if path.endswith((".js", ".css", ".html")):
            todo += _STATIC_REFERENCE.findall(body.decode("utf-8"))
    report["info"]["static_files"] = sorted(seen)


def _web_flow(client: _Client, root: Path, report: Report, record_file: str) -> None:
    """The sample project through the API, up to an export."""
    answer = client.json("POST", "/api/projects/sample", expect=201)
    rows = answer["sample"]["rows"]
    check(len(rows) == 2, f"the sample project has {len(rows)} rows")
    for row in rows:
        answer = client.json("POST", "/api/boxes/row", row, expect=201)
    charts = sorted(set(re.findall(r"/api/charts/[A-Za-z0-9_-]+\.svg", json.dumps(answer))))
    check(charts, "no chart in the answer")
    for url in charts:
        svg, _ = client.request("GET", url)
        check(b"<svg" in svg, f"{url} is not SVG")
    images = [image["id"] for image in answer["project"]["images"]]
    check(len(images) == 2, f"the sample project has {len(images)} images, not the blot and marker")
    for image_id in images:
        png, _ = client.request("GET", f"/api/images/{image_id}/preview")
        check(png.startswith(b"\x89PNG\r\n\x1a\n"), f"the preview of {image_id} is not PNG")
    exported = client.json("POST", "/api/export", {"formats": ["svg", "png", "pdf"]}, expect=201)
    folder = root / answer["project"]["name"] / exported["folder"]
    record = json.loads((folder / record_file).read_bytes())
    missing = sorted(name for name, version in record["software"].items() if version is None)
    check(not missing, f"the export record has no version for {missing}")
    _check_record_files(folder, record)
    report["info"]["web"] = {"charts": len(charts), "previews": len(images)}


STEPS: Final[tuple[tuple[str, Callable[[Path, Report], None]], ...]] = (
    ("imports", _imports),
    ("versions", _versions),
    ("images", _images),
    ("sample analysis", _sample_analysis),
    ("statistics", _statistics),
    ("web app", _web_app),
)


def run(folder: Path) -> Report:
    """Run every step in ``folder`` (a new, empty folder of its own); the report."""
    report: Report = {
        "ok": False,
        "proteia": proteia.__version__,
        "python": platform.python_version(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "steps": [],
        "numbers": {},
        "versions": {},
        "info": {},
    }
    for number, (name, step) in enumerate(STEPS, 1):
        start = time.perf_counter()
        outcome: dict[str, Any] = {"name": name, "ok": True}
        try:
            (folder / str(number)).mkdir()  # short names: a deep TEMP leaves little room
            step(folder / str(number), report)
        except Exception as exc:  # every failure is reported, and the other steps run
            outcome.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            outcome["traceback"] = traceback.format_exc()
        outcome["seconds"] = round(time.perf_counter() - start, 3)
        report["steps"].append(outcome)
    report["ok"] = all(outcome["ok"] for outcome in report["steps"])
    return report


def _summary(report: Report) -> str:
    lines = [
        f"Proteia {report['proteia']} self-test, Python {report['python']}"
        f"{' (frozen)' if report['frozen'] else ''}:"
    ]
    for outcome in report["steps"]:
        mark = "ok  " if outcome["ok"] else "FAIL"
        lines.append(f"  {mark} {outcome['name']} ({outcome['seconds']:.2f} s)")
        if not outcome["ok"]:
            lines.append(f"       {outcome['error']}")
    lines.append("passed" if report["ok"] else "FAILED")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``proteia --self-test [--json PATH] [--keep]``: 0 when every step passed."""
    parser = argparse.ArgumentParser(prog="proteia --self-test", description=__doc__.split("\n")[0])
    parser.add_argument("--json", type=Path, help="also write the report to this file")
    parser.add_argument("--keep", action="store_true", help="keep the temporary folder")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # a code-page console cannot show µ
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")
    folder = Path(tempfile.mkdtemp(prefix="proteia-"))
    try:
        report = run(folder)
        report["info"]["folder"] = str(folder)
    finally:
        if not args.keep:
            shutil.rmtree(folder, ignore_errors=True)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(_summary(report))
    for outcome in report["steps"]:
        if not outcome["ok"]:
            print(outcome["traceback"], file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
