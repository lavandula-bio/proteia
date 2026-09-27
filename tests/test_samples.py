# SPDX-License-Identifier: Apache-2.0
"""Tests for the synthetic sample images (#55).

The samples are written into folders with non-ASCII names, imported through the
operations and taken through the whole flow: lanes of two conditions, a loading
control and a target, a row box over each row, the results. The tolerances come
from the measured values (in the comments) with a margin.
"""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from conftest import FakeClock
from proteia import samples
from proteia.core import operations as ops
from proteia.core.analyze import Tier
from proteia.core.imaging import read_pixels
from proteia.core.model import ImageKind, Polarity, Role
from proteia.core.operations import LaneInput, ProjectSession
from proteia.core.plotspec import ValueKind

FOLDER = "樣本 µ α β"
DARK = Polarity.DARK_ON_LIGHT


@pytest.fixture(scope="module")
def sample_folder(tmp_path_factory) -> Path:
    """The samples, written once for the tests that only read them."""
    return samples.generate(tmp_path_factory.mktemp("samples") / FOLDER)[0].parent


def truth(folder: Path) -> list[dict[str, str]]:
    with (folder / samples.TRUTH_FILE).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def import_samples(session: ProjectSession, folder: Path) -> tuple[str, str]:
    """Import the blot, then its marker image onto the same membrane."""
    with (folder / samples.BLOT_FILE).open("rb") as f:
        blot = ops.import_image(
            session, f, samples.BLOT_FILE, kind=ImageKind.CHEMILUMINESCENCE, polarity=DARK
        )
    [membrane] = session.project.batch.membranes
    with (folder / samples.MARKER_FILE).open("rb") as f:
        marker = ops.import_image(
            session,
            f,
            samples.MARKER_FILE,
            kind=ImageKind.VISIBLE_MARKER,
            polarity=DARK,
            membrane_id=membrane.id,
        )
    return blot, marker


def row_box(row: samples.Row) -> tuple[int, int, int, int]:
    """The box a user drags over ``row``: half a lane pitch past the end lanes'
    centres, 25 px above the end lanes' bands and 25 px below the middle lanes'
    (which ran SMILE_PX further)."""
    half = samples.LANE_PITCH / 2
    y = samples.band_y(row.kda, samples.LANE_X[0])
    return (
        round(samples.LANE_X[0] - half),
        round(y - 25),
        round(samples.LANE_X[-1] + half),
        round(y + samples.SMILE_PX + 25),
    )


def band_centre(pixels: np.ndarray, y: float, x0: int, x1: int) -> float:
    """Where the band near row ``y`` lies in columns ``x0`` to ``x1 - 1``: the
    centre of its darkening within 12 px of ``y``, below the membrane measured in
    the 12 px beyond that on either side."""
    profile = pixels[:, x0:x1].astype(float).mean(axis=1)
    top, bottom = round(y) - 12, round(y) + 12
    membrane = np.concatenate([profile[top - 12 : top], profile[bottom + 1 : bottom + 13]]).mean()
    darkening = np.clip(membrane - profile[top : bottom + 1], 0.0, None)
    return float((np.arange(top, bottom + 1) * darkening).sum() / darkening.sum())


# --- The files ---


def test_generating_twice_gives_identical_files(tmp_path):
    first = samples.generate(tmp_path / "first" / FOLDER)
    second = samples.generate(tmp_path / "second" / FOLDER)
    assert [path.name for path in first] == list(samples.FILES)
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name
    blot, marker, _ = first
    np.testing.assert_array_equal(read_pixels(blot), read_pixels(second[0]))
    np.testing.assert_array_equal(read_pixels(marker), read_pixels(second[1]))
    # The files hold what the renderers give, and the renderers repeat themselves.
    np.testing.assert_array_equal(read_pixels(blot), samples.render_blot().pixels)
    np.testing.assert_array_equal(read_pixels(marker), samples.render_marker())


def test_generating_again_leaves_the_same_files_alone(tmp_path):
    paths = samples.generate(tmp_path / FOLDER)
    before = [path.stat().st_mtime_ns for path in paths]
    assert samples.generate(tmp_path / FOLDER) == paths
    assert [path.stat().st_mtime_ns for path in paths] == before  # not rewritten


def test_other_contents_are_overwritten_only_with_force(tmp_path):
    folder = tmp_path / FOLDER
    blot, marker, table = samples.generate(folder)
    expected = {path.name: path.read_bytes() for path in (blot, marker, table)}
    table.write_bytes(b"edited")
    blot.write_bytes(b"not a blot")
    marker.unlink()
    notes = folder / "notes.txt"
    notes.write_text("mine", encoding="utf-8")

    with pytest.raises(samples.SampleConflictError) as info:
        samples.generate(folder)
    assert info.value.names == (samples.BLOT_FILE, samples.TRUTH_FILE)
    assert isinstance(info.value, FileExistsError)
    # Nothing was written, not even the missing file.
    assert (table.read_bytes(), blot.read_bytes()) == (b"edited", b"not a blot")
    assert not marker.exists()

    assert samples.generate(folder, force=True) == (blot, marker, table)
    assert {path.name: path.read_bytes() for path in (blot, marker, table)} == expected
    assert notes.read_text(encoding="utf-8") == "mine"  # never touched
    assert sorted(p.name for p in folder.iterdir()) == sorted([*samples.FILES, "notes.txt"])


def test_the_blot_is_a_16_bit_scan_without_clipping(sample_folder):
    with tifffile.TiffFile(sample_folder / samples.BLOT_FILE) as tif:
        [page] = tif.pages
        assert page.compression == tifffile.COMPRESSION.NONE
        assert "synthetic" in page.description and "not a real blot" in page.description
    pixels = read_pixels(sample_folder / samples.BLOT_FILE)
    assert pixels.dtype == np.uint16
    assert pixels.shape == (samples.HEIGHT, samples.WIDTH) == (500, 1200)
    # measured: 26811 to 54227
    assert 20000 < pixels.min() and pixels.max() < 60000  # far from 0 and 65535
    # Dark bands on a light membrane: the membrane is the bulk of the image.
    assert abs(float(np.median(pixels)) - samples.MEMBRANE) < 200  # measured 68 below


def test_the_blot_rows_follow_the_migration_law_and_smile(sample_folder):
    # The marker calibrates molecular weight on this membrane (#58), so each row
    # must run where band_y puts its protein; and row detection should meet
    # curved rows, as on a real gel.
    pixels = read_pixels(sample_folder / samples.BLOT_FILE)
    for row in samples.ROWS:
        centres, ends = [], []
        for x in samples.LANE_X:
            y, c = samples.band_y(row.kda, x), round(x)
            centre = band_centre(pixels, y, c - 10, c + 11)
            # Y_JITTER moves a band up to 1 px; measured: 0.97 px at most
            assert abs(centre - y) <= 1.3, (row.protein, x)
            centres.append(centre)
            left, right = (
                band_centre(pixels, y, c - 40, c - 29),
                band_centre(pixels, y, c + 30, c + 41),
            )
            ends.append((left + right) / 2 - centre)
        # The middle lanes ran further down than the end lanes.
        # measured: β-catenin 6.00 px, α-tubulin 6.16 px
        smile = (centres[3] + centres[4] - centres[0] - centres[7]) / 2
        assert 5.0 <= smile <= 7.0, row.protein
        # Each band's own ends curve up. measured: 0.70 to 0.94 px
        assert all(-1.2 <= end <= -0.5 for end in ends), (row.protein, ends)


def test_the_marker_shows_the_ladder_where_the_migration_law_puts_it(sample_folder):
    pixels = read_pixels(sample_folder / samples.MARKER_FILE)
    assert (pixels.dtype, pixels.shape) == (np.uint8, (samples.HEIGHT, samples.WIDTH))
    assert 0 < pixels.min() and pixels.max() < 255
    x = round(samples.LADDER_X)
    profile = pixels[:, x - 10 : x + 11].astype(float).mean(axis=1)
    for kda in samples.LADDER_KDA:
        y = samples.band_y(kda, samples.LADDER_X)
        lo = round(y) - 6
        darkest = lo + int(np.argmin(profile[lo : lo + 13]))
        assert abs(darkest - y) <= 1.0, kda
        assert profile[darkest] < samples.MARKER_MEMBRANE - 50
    # Nothing but membrane where the sample lanes are.
    lanes = pixels[:, round(samples.LANE_X[0]) - 60 :].astype(float)
    assert lanes.min() > samples.MARKER_MEMBRANE - 20


def test_the_truth_table(sample_folder):
    data = (sample_folder / samples.TRUTH_FILE).read_bytes()
    assert data.startswith(b"\xef\xbb\xbf") and data.endswith(b"\r\n")
    rows = truth(sample_folder)
    assert list(rows[0]) == [
        "lane",
        "condition",
        "sample",
        "β-catenin true signal",
        "α-tubulin true signal",
        "β-catenin / α-tubulin",
        "fold change vs vehicle",
    ]
    # The design, written out here rather than read back from the module.
    assert [(r["lane"], r["condition"], r["sample"]) for r in rows] == [
        ("1", "vehicle", "V1"),
        ("2", "vehicle", "V2"),
        ("3", "vehicle", "V3"),
        ("4", "vehicle", "V4"),
        ("5", "treatment", "T1"),
        ("6", "treatment", "T2"),
        ("7", "treatment", "T3"),
        ("8", "treatment", "T4"),
    ]
    signals = samples.render_blot().signals
    for row in samples.ROWS:
        name = row.protein
        assert [float(r[f"{name} true signal"]) for r in rows] == [round(s) for s in signals[name]]
        # Each band's signal is its designed amount: loading, times expression for the target.
        amounts = [
            samples.LOADED[i] * (samples.EXPRESSION[i] if row is samples.TARGET_ROW else 1.0)
            for i in range(8)
        ]
        np.testing.assert_allclose(signals[name], [a * row.signal for a in amounts], rtol=1e-12)
    # The loading cancels. β-catenin / α-tubulin is the expression times 8e6 / 2e7
    # (each row's signal per unit amount), and the fold change is the expression.
    assert [r["β-catenin / α-tubulin"] for r in rows] == [
        "0.4480",
        "0.3440",
        "0.4600",
        "0.3480",
        "0.9240",
        "0.6880",
        "0.8760",
        "0.7120",
    ]
    assert [r["fold change vs vehicle"] for r in rows] == [
        "1.1200",
        "0.8600",
        "1.1500",
        "0.8700",
        "2.3100",
        "1.7200",
        "2.1900",
        "1.7800",
    ]
    fold = [float(r["fold change vs vehicle"]) for r in rows]
    assert np.mean(fold[:4]) == pytest.approx(1.0) and np.mean(fold[4:]) == pytest.approx(2.0)


# --- Through the product ---


@pytest.mark.filterwarnings("error")
def test_the_samples_import_without_warnings(tmp_path, sample_folder):
    s = ops.new_project(tmp_path / FOLDER, autosave=None, clock=FakeClock())
    blot, marker = import_samples(s, sample_folder)
    [membrane] = s.project.batch.membranes
    assert [image.id for image in membrane.images] == [blot, marker]
    found = {
        image.id: (image.kind, image.bit_depth, image.width, image.height, image.import_warnings)
        for image in membrane.images
    }
    assert found == {
        blot: (ImageKind.CHEMILUMINESCENCE, 16, 1200, 500, []),
        marker: (ImageKind.VISIBLE_MARKER, 8, 1200, 500, []),
    }


def test_the_full_flow_gives_the_true_fold_changes(tmp_path, sample_folder):
    s = ops.new_project(tmp_path / FOLDER, autosave=None, clock=FakeClock())
    blot, _ = import_samples(s, sample_folder)
    ops.set_lanes(
        s,
        [
            LaneInput(c, sample)
            for c, sample in zip(samples.CONDITIONS, samples.SAMPLES, strict=True)
        ],
        reference_condition=samples.REFERENCE,
    )
    loading = ops.add_protein(s, samples.LOADING_CONTROL, Role.LOADING_CONTROL, blot)
    target = ops.add_protein(s, samples.TARGET, Role.TARGET, blot, loading_control_ids=[loading])
    for protein, row in ((target, samples.TARGET_ROW), (loading, samples.LOADING_ROW)):
        placement = ops.detect_row_boxes(s, protein, row_box(row))
        assert None not in placement.band_ids, row.protein
        assert (placement.flags, placement.empty, placement.notes) == ((), (), ()), row.protein

    res = ops.compute(s)
    assert res.tier is Tier.FOLD_CHANGE
    assert res.notices == []
    rows = truth(sample_folder)
    for column in res.proteins:
        assert column.clipped == [False] * 8
        true = [float(r[f"{column.name} true signal"]) for r in rows]
        share = [net / t for net, t in zip(column.nets, true, strict=True)]
        # The net misses the band tails outside the box.
        # measured: β-catenin 0.855-0.886, α-tubulin 0.863-0.904
        assert 0.83 <= min(share) and max(share) <= 0.93, column.name

    # The lane-table export numbers lanes from 1, as the truth table and the app
    # do, so the two files' rows match on the lane as well as on the sample.
    # Matched so, every net meets its own truth.
    with ops.export_lane_table(s).open(encoding="utf-8-sig", newline="") as f:
        exported = {lane["sample"]: lane for lane in csv.DictReader(f)}
    assert sorted(exported) == sorted(r["sample"] for r in rows)
    for r in rows:
        lane = exported[r["sample"]]
        assert lane["lane"] == r["lane"], r["sample"]
        for name in (samples.TARGET, samples.LOADING_CONTROL):
            assert 0.83 <= float(lane[name]) / float(r[f"{name} true signal"]) <= 0.93

    [series] = res.series
    assert (series.target, series.loading) == (samples.TARGET, samples.LOADING_CONTROL)
    assert series.value_kind is ValueKind.FOLD_CHANGE
    true_fold = [float(r["fold change vs vehicle"]) for r in rows]
    errors = [f / t - 1 for f, t in zip(series.fold_change, true_fold, strict=True)]
    # measured: -1.23 % (lane 4) to +1.48 % (lane 7)
    assert max(map(abs, errors)) <= 0.025

    chart = series.chart
    vehicle, treatment = chart.bars
    assert (vehicle.label, vehicle.n, treatment.label, treatment.n) == (
        "vehicle",
        4,
        "treatment",
        4,
    )
    assert vehicle.mean == pytest.approx(1.0)  # the baseline is the vehicle mean
    assert treatment.mean == pytest.approx(2.0, rel=0.015)  # measured 2.0088
    assert chart.test_name == "welch_t"
    assert chart.test_p < 0.01  # measured 0.0026
    [comparison] = chart.comparisons
    assert (comparison.group_a, comparison.group_b, comparison.stars) == (
        "vehicle",
        "treatment",
        "**",
    )


# --- The command ---


def test_the_command_writes_the_default_folder_and_refuses_to_overwrite(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    assert samples.main([]) == 0
    folder = Path(samples.DEFAULT_FOLDER)
    assert capsys.readouterr().out.splitlines() == [str(folder / n) for n in samples.FILES]
    blot = folder / samples.BLOT_FILE
    expected = blot.read_bytes()

    blot.write_bytes(b"edited")
    assert samples.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert samples.BLOT_FILE in captured.err and "--force" in captured.err
    assert blot.read_bytes() == b"edited"

    assert samples.main(["--force"]) == 0
    assert blot.read_bytes() == expected


def run_module(*args: str, hash_seed: str = "0") -> subprocess.CompletedProcess:
    """``python -m proteia.samples`` in a new process with a fixed hash seed, an
    80-column console (argparse wraps the help to it) and a strict ASCII
    console. Like a code-page console (cp950) meeting µ, that console cannot
    show the paths: the command must escape them rather than fail after writing
    the files."""
    env = {
        **os.environ,
        "PYTHONIOENCODING": "ascii:strict",
        "PYTHONHASHSEED": hash_seed,
        "COLUMNS": "80",
    }
    return subprocess.run(
        [sys.executable, "-m", "proteia.samples", *args],
        capture_output=True,
        encoding="ascii",
        env=env,
        timeout=120,
        check=False,
    )


def escaped(text: str) -> str:
    """``text`` as the ASCII console shows it."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


# Fixed seeds, so every run checks the same thing. They order the hashes of the
# two protein names both ways (0 and 1 one way, 2 the other, on CPython 3.13),
# so anything drawn in hash order differs from the test process in at least one.
@pytest.mark.parametrize("hash_seed", ["0", "1", "2"])
def test_python_m_writes_the_same_bytes_whatever_the_hash_seed(tmp_path, sample_folder, hash_seed):
    folder = tmp_path / FOLDER
    done = run_module(str(folder), hash_seed=hash_seed)
    assert (done.returncode, done.stderr) == (0, "")
    assert done.stdout.splitlines() == [escaped(str(folder / name)) for name in samples.FILES]
    for name in samples.FILES:  # another process writes the same bytes
        assert (folder / name).read_bytes() == (sample_folder / name).read_bytes(), name


def test_python_m_shows_the_help():
    shown = run_module("--help")
    assert (shown.returncode, shown.stderr) == (0, "")
    assert samples.DEFAULT_FOLDER in shown.stdout and "--force" in shown.stdout
    text = " ".join(shown.stdout.split())
    # How to match the results with the truth table (see the full-flow test).
    assert "match rows on the lane or the sample" in text and "number lanes from 1" in text


def test_git_ignores_the_default_folder():
    # Run in the repository root, the command leaves nothing for `git add -A`
    # to pick up: sample images go into the repository only on purpose.
    root = Path(__file__).resolve().parents[1]
    git = shutil.which("git")
    if git is None or not (root / ".git").exists():
        pytest.skip("needs git and a git checkout")
    paths = [f"{samples.DEFAULT_FOLDER}/{name}" for name in samples.FILES]
    done = subprocess.run(
        [git, "check-ignore", "--no-index", "--", *paths],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert (done.returncode, done.stdout.splitlines()) == (0, paths)


def test_a_sample_held_by_another_program_stops_every_write(tmp_path, monkeypatch):
    folder = tmp_path / "samples µ"
    samples.generate(folder)
    before = {name: (folder / name).read_bytes() for name in samples.FILES}
    for name in samples.FILES:
        (folder / name).write_bytes(b"an older version")
    older = {name: (folder / name).read_bytes() for name in samples.FILES}
    monkeypatch.setattr(samples, "_held", lambda path: path.name == samples.TRUTH_FILE)

    with pytest.raises(samples.SampleWriteError) as caught:
        samples.generate(folder, force=True)
    assert (caught.value.replaced, caught.value.failed) == ((), (samples.TRUTH_FILE,))
    assert "nothing was written" in str(caught.value)
    assert {name: (folder / name).read_bytes() for name in samples.FILES} == older
    assert older != before


def test_a_write_failing_after_others_names_what_was_replaced(tmp_path, monkeypatch):
    folder = tmp_path / "samples"
    samples.generate(folder)
    for name in samples.FILES:
        (folder / name).write_bytes(b"an older version")
    real = samples.write_atomic

    def write(path, data):
        if path.name == samples.TRUTH_FILE:
            raise PermissionError(13, "held", str(path))
        real(path, data)

    monkeypatch.setattr(samples, "write_atomic", write)
    with pytest.raises(samples.SampleWriteError) as caught:
        samples.generate(folder, force=True)
    assert caught.value.replaced == (samples.BLOT_FILE, samples.MARKER_FILE)
    assert caught.value.failed == (samples.TRUTH_FILE,)
    assert "two versions" in str(caught.value)
    assert (folder / samples.TRUTH_FILE).read_bytes() == b"an older version"


def test_held_tells_a_file_another_program_holds(tmp_path, monkeypatch):
    path = tmp_path / "sample-truth.csv"
    assert not samples._held(path)  # missing
    path.write_bytes(b"x")
    assert not samples._held(path)

    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "held", str(self))

    monkeypatch.setattr(Path, "open", refuse)
    assert samples._held(path)
