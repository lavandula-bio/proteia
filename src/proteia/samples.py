# SPDX-License-Identifier: Apache-2.0
"""Synthetic sample images to try the whole flow on (#55).

``uv run python -m proteia.samples [FOLDER] [--force]`` writes them into
``FOLDER``, by default :data:`DEFAULT_FOLDER` (``proteia-samples``) in the
current directory; :func:`generate` does the same from code. Every pixel is
computed here from fixed numbers and fixed seeds: no image derived from a real
blot is involved, so the samples carry no third-party image rights.

Writing is safe to repeat. The folder is created if missing, and a sample file
already there with the same bytes is left as it is. One with other bytes (an
edited copy, or the samples of another version) is replaced only with
``--force`` (``force=True``); without it, nothing at all is written and the
files are named (:class:`SampleConflictError`). Other files in the folder are
never touched.

The files, named to sort in the order they are used:

* ``sample-blot.tif``: a 16-bit chemiluminescence scan, dark bands on a light
  membrane, 1200 x 500 px, uncompressed. Import it as a chemiluminescence
  image, dark on light.
* ``sample-marker.tif``: the same membrane under visible light, 8-bit: only a
  prestained ladder shows, in a lane left of lane 1. It is for the
  molecular-weight calibration (#58); the flow up to the fold-change chart does
  not need it.
* ``sample-truth.csv``: the true signal of every band, one row per lane, UTF-8
  with a byte-order mark like Proteia's own CSV files.

The experiment. Eight lanes, each its own biological replicate: ``vehicle`` in
lanes 1-4 (samples V1-V4) and ``treatment`` in lanes 5-8 (T1-T4). Two rows: the
target β-catenin (92 kDa) and, lower, the loading control α-tubulin (50 kDa).
Declare the lanes with these conditions, make ``vehicle`` the reference, add
α-tubulin as the loading control and β-catenin as the target, and drag a row box
over each row.

The truth. A band's true signal is the sum over the image of its noise-free
darkening below the membrane, in 16-bit counts x pixels. α-tubulin follows the
protein loaded in each lane (:data:`LOADED`, within 8 % of each other);
β-catenin is its expression in the sample (:data:`EXPRESSION`) times the same
loading. The loading therefore cancels in β-catenin / α-tubulin, and a lane's
fold change (its ratio over the mean ratio of the vehicle lanes, as Proteia
computes it) is its sample's expression: vehicle 1.12, 0.86, 1.15, 0.87 (mean
1.00), treatment 2.31, 1.72, 2.19, 1.78 (mean 2.00). The truth table numbers
lanes from 1, as the app shows them, but Proteia's lane-table export
(``exports/lane-table.csv``) numbers them from 0: match the two files' rows on
``sample`` (V1-V4, T1-T4), not on ``lane``.

What Proteia measures on them. A net integrates inside the box, above the ring
background, so it reads 85-90 % of the true signal: the band's tails lie outside
the box, a share that varies a little with each band's width and height against
its row's one box size. The share mostly cancels in the ratios: the per-lane
fold changes come within 1.5 % of the truth and the treatment mean within 0.5 %
(Welch's t-test p = 0.003). ``tests/test_samples.py`` checks these with a margin.

The image. The membrane is 52000 counts with a mild linear gradient (+/-800
across, +/-300 down: within what row detection takes without a
``background_mismatch`` flag), Gaussian read noise of 300 counts, and a
shot-noise term that grows with a band's darkening; no pixel reaches 0 or 65535,
so nothing is clipped. A band is flat-topped across its lane with soft edges (a
super-Gaussian) and Gaussian down it. The gel smiles: the middle lanes ran
:data:`SMILE_PX` px further than the end lanes, each band's centre line follows
that curve, and its own ends curve up a little more. Lanes sit up to 2 px off
their nominal place, and bands vary a few percent in width and height.

The marker. The membrane is 205 of 255 with a faint gradient and noise. The
ladder's nine bands (:data:`LADDER_KDA`; the 70 and 25 kDa ones darker, as
prestained reference bands often are) lie where the gel's migration law puts
them: a band of ``m`` kDa is centred at :func:`band_y` ``(m, x)``, log-linear in
the molecular weight, plus the smile.

Determinism: one installation writes the same bytes every time. numpy's
vectorized ``exp`` may differ in the last bit between machines, which can move a
rare pixel by one count, far below the noise.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import tifffile

from proteia.core.export import CSV_ENCODING
from proteia.core.storage import write_atomic

DEFAULT_FOLDER: Final = "proteia-samples"
BLOT_FILE: Final = "sample-blot.tif"
MARKER_FILE: Final = "sample-marker.tif"
TRUTH_FILE: Final = "sample-truth.csv"
FILES: Final = (BLOT_FILE, MARKER_FILE, TRUTH_FILE)

# --- The experiment ---

LANES: Final = 8
CONDITIONS: Final = ("vehicle",) * 4 + ("treatment",) * 4
SAMPLES: Final = ("V1", "V2", "V3", "V4", "T1", "T2", "T3", "T4")
REFERENCE: Final = "vehicle"
TARGET: Final = "β-catenin"
LOADING_CONTROL: Final = "α-tubulin"
# β-catenin in each sample, relative to the vehicle mean: vehicle mean 1, treatment mean 2.
EXPRESSION: Final = (1.12, 0.86, 1.15, 0.87, 2.31, 1.72, 2.19, 1.78)
# Protein loaded in each lane, relative: the loading control follows it.
LOADED: Final = (1.00, 0.94, 1.05, 0.97, 1.03, 0.92, 1.07, 0.99)


@dataclass(frozen=True)
class Row:
    """One protein's row of bands."""

    protein: str
    kda: float  # molecular weight: where the row runs (band_y)
    height: float  # band height down the lane at 20 % of its peak, px
    signal: float  # true signal of a band of relative amount 1, counts x px


TARGET_ROW: Final = Row(TARGET, 92.0, 16.0, 8.0e6)
LOADING_ROW: Final = Row(LOADING_CONTROL, 50.0, 18.0, 2.0e7)
ROWS: Final = (TARGET_ROW, LOADING_ROW)

# --- The gel and the membrane ---

WIDTH: Final = 1200
HEIGHT: Final = 500
LANE_PITCH: Final = 128.0
LANE_X: Final = tuple(205.0 + i * LANE_PITCH for i in range(LANES))  # nominal lane centres
LADDER_X: Final = LANE_X[0] - LANE_PITCH  # the ladder lane, left of lane 1
BAND_WIDTH: Final = 92.0  # across the lane at 20 % of the peak, px
# Migration: a band of MW_TOP kDa at Y_TOP in the end lanes, PX_PER_DECADE px
# further down per tenfold lighter.
MW_TOP: Final = 250.0
Y_TOP: Final = 45.0
PX_PER_DECADE: Final = 300.0
SMILE_PX: Final = 6.0  # how much further the middle lanes ran than the end lanes
BAND_CURVE_PX: Final = 1.5  # a band's own ends curve up this much more
X_JITTER: Final = 2.0  # px off the nominal lane centre, at most
Y_JITTER: Final = 1.0  # px off the row's line, at most
WIDTH_JITTER: Final = 0.04  # relative, per lane
HEIGHT_JITTER: Final = 0.08  # relative, per band

MEMBRANE: Final = 52000.0
GRADIENT: Final = (800.0, 300.0)  # membrane change from the centre to the right, and down
READ_NOISE: Final = 300.0  # counts, 1 sigma
SHOT_NOISE: Final = 4.0  # noise variance added per count of darkening
BLOT_SEED: Final = 55

# --- The visible-light marker image (8-bit) ---

LADDER_KDA: Final = (250, 130, 100, 70, 55, 35, 25, 15, 10)
LADDER_REFERENCE_KDA: Final = (70, 25)  # the darker reference bands
LADDER_DEPTH: Final = 70.0  # peak darkening of a ladder band
LADDER_REFERENCE_DEPTH: Final = 110.0  # ... and of a reference band
LADDER_HEIGHT: Final = 9.0  # at 20 % of the peak, px
MARKER_MEMBRANE: Final = 205.0
MARKER_GRADIENT: Final = (3.0, 2.0)
MARKER_NOISE: Final = 1.5
MARKER_SEED: Final = 56

# Half-extents at 20 % of the peak, in units of the profile's scale: a
# super-Gaussian exp(-0.5 |t|**4) across the lane and a Gaussian down it.
_UX4 = (2.0 * math.log(5.0)) ** 0.25
_UX2 = math.sqrt(2.0 * math.log(5.0))
_BLOT_NOTE = (
    "Proteia synthetic sample blot, not a real blot: 16-bit chemiluminescence, dark"
    " bands on a light membrane. True band signals: sample-truth.csv."
)
_MARKER_NOTE = (
    "Proteia synthetic sample, not a real blot: the membrane of sample-blot.tif under"
    " visible light, showing a prestained ladder left of lane 1."
)


def _smile(x: np.ndarray | float) -> np.ndarray | float:
    """How much further down the gel ran at ``x`` than in the end lanes, px."""
    centre = 0.5 * (LANE_X[0] + LANE_X[-1])
    half = 0.5 * (LANE_X[-1] - LANE_X[0])
    return SMILE_PX * (1.0 - ((x - centre) / half) ** 2)


def band_y(kda: float, x: float) -> float:
    """Where the centre of a band of ``kda`` kDa lies at column ``x``: the
    migration law (log-linear in the molecular weight) plus the smile."""
    return Y_TOP + PX_PER_DECADE * math.log10(MW_TOP / kda) + float(_smile(x))


def _band(cx: float, cy: float, width: float, height: float) -> np.ndarray:
    """A band of unit peak over the whole image, centred at ``(cx, cy)``:
    flat-topped across the lane with soft edges, Gaussian down it, its centre
    line following the smile, its own ends curving up."""
    xs = np.arange(WIDTH, dtype=float)
    ys = np.arange(HEIGHT, dtype=float)
    across = np.exp(-0.5 * np.abs((xs - cx) / (width / (2.0 * _UX4))) ** 4)
    line = cy + (_smile(xs) - _smile(cx)) - BAND_CURVE_PX * ((xs - cx) / (0.5 * width)) ** 2
    down = np.exp(-0.5 * ((ys[:, None] - line[None, :]) / (height / (2.0 * _UX2))) ** 2)
    return down * across[None, :]


def _membrane(level: float, gradient: tuple[float, float]) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, WIDTH)
    ys = np.linspace(-1.0, 1.0, HEIGHT)
    return level + gradient[0] * xs[None, :] + gradient[1] * ys[:, None]


@dataclass(frozen=True)
class SampleBlot:
    """The chemiluminescence sample and the truth it was made from."""

    pixels: np.ndarray  # uint16, HEIGHT x WIDTH
    signals: dict[str, tuple[float, ...]]  # protein -> true signal per lane


def render_blot() -> SampleBlot:
    """The chemiluminescence sample: the same pixels on every call."""
    rng = np.random.default_rng(BLOT_SEED)
    lane_x = np.asarray(LANE_X) + rng.uniform(-X_JITTER, X_JITTER, LANES)
    widths = BAND_WIDTH * rng.uniform(1.0 - WIDTH_JITTER, 1.0 + WIDTH_JITTER, LANES)
    darkening = np.zeros((HEIGHT, WIDTH))
    signals: dict[str, tuple[float, ...]] = {}
    for row in ROWS:
        dy = rng.uniform(-Y_JITTER, Y_JITTER, LANES)
        heights = row.height * rng.uniform(1.0 - HEIGHT_JITTER, 1.0 + HEIGHT_JITTER, LANES)
        found = []
        for i in range(LANES):
            amount = LOADED[i] * (EXPRESSION[i] if row is TARGET_ROW else 1.0)
            unit = _band(lane_x[i], band_y(row.kda, lane_x[i]) + dy[i], widths[i], heights[i])
            band = unit * (amount * row.signal / unit.sum())
            darkening += band
            found.append(float(band.sum()))
        signals[row.protein] = tuple(found)
    noise = rng.standard_normal((HEIGHT, WIDTH)) * np.sqrt(READ_NOISE**2 + SHOT_NOISE * darkening)
    image = np.round(_membrane(MEMBRANE, GRADIENT) - darkening + noise)
    pixels = np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return SampleBlot(pixels, signals)


def render_marker() -> np.ndarray:
    """The visible-light marker sample (uint8, HEIGHT x WIDTH): the same pixels
    on every call."""
    rng = np.random.default_rng(MARKER_SEED)
    darkening = np.zeros((HEIGHT, WIDTH))
    for kda in LADDER_KDA:
        depth = LADDER_REFERENCE_DEPTH if kda in LADDER_REFERENCE_KDA else LADDER_DEPTH
        darkening += depth * _band(LADDER_X, band_y(kda, LADDER_X), BAND_WIDTH, LADDER_HEIGHT)
    noise = MARKER_NOISE * rng.standard_normal((HEIGHT, WIDTH))
    image = np.round(_membrane(MARKER_MEMBRANE, MARKER_GRADIENT) - darkening + noise)
    return np.clip(image, 0, np.iinfo(np.uint8).max).astype(np.uint8)


def truth_table(signals: dict[str, tuple[float, ...]]) -> bytes:
    """The truth as CSV bytes (:data:`~proteia.core.export.CSV_ENCODING`, CRLF
    rows). Per lane, numbered from 1: its condition and sample, each protein's
    true signal, β-catenin / α-tubulin, and the fold change: that ratio over the
    mean ratio of the vehicle lanes."""
    target, loading = signals[TARGET], signals[LOADING_CONTROL]
    ratios = [t / lc for t, lc in zip(target, loading, strict=True)]
    reference = [r for r, c in zip(ratios, CONDITIONS, strict=True) if c == REFERENCE]
    baseline = sum(reference) / len(reference)
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(
        [
            "lane",
            "condition",
            "sample",
            f"{TARGET} true signal",
            f"{LOADING_CONTROL} true signal",
            f"{TARGET} / {LOADING_CONTROL}",
            f"fold change vs {REFERENCE}",
        ]
    )
    for i in range(LANES):
        writer.writerow(
            [
                i + 1,
                CONDITIONS[i],
                SAMPLES[i],
                round(target[i]),
                round(loading[i]),
                f"{ratios[i]:.4f}",
                f"{ratios[i] / baseline:.4f}",
            ]
        )
    return text.getvalue().encode(CSV_ENCODING)


def _tiff(pixels: np.ndarray, description: str) -> bytes:
    """Uncompressed single-image TIFF bytes, with no date or version in them."""
    buffer = io.BytesIO()
    tifffile.imwrite(
        buffer,
        pixels,
        photometric="minisblack",
        description=description,
        software="proteia.samples",
        metadata=None,
    )
    return buffer.getvalue()


def sample_files() -> dict[str, bytes]:
    """Every sample file's name and bytes, in :data:`FILES` order."""
    blot = render_blot()
    return {
        BLOT_FILE: _tiff(blot.pixels, _BLOT_NOTE),
        MARKER_FILE: _tiff(render_marker(), _MARKER_NOTE),
        TRUTH_FILE: truth_table(blot.signals),
    }


class SampleConflictError(FileExistsError):
    """Sample files in the folder hold other bytes than these; nothing was written."""

    def __init__(self, folder: Path, names: Sequence[str]) -> None:
        self.folder = folder
        self.names = tuple(names)
        super().__init__(
            f"{folder} already holds {', '.join(self.names)} with other contents;"
            " nothing was written"
        )


class SampleWriteError(OSError):
    """The sample files could not all be written. ``replaced`` names those
    already replaced, so the folder may hold files of two versions; ``failed``
    names the rest."""

    def __init__(self, folder: Path, replaced: Sequence[str], failed: Sequence[str]) -> None:
        self.folder = folder
        self.replaced = tuple(replaced)
        self.failed = tuple(failed)
        if self.replaced:
            state = (
                f"replaced {', '.join(self.replaced)} but not {', '.join(self.failed)}:"
                " the folder now holds files of two versions"
            )
        else:
            state = f"{', '.join(self.failed)} is open in another program; nothing was written"
        super().__init__(
            f"in {folder}, {state}. Close the files in other programs and run again with --force."
        )


def generate(folder: str | os.PathLike[str], *, force: bool = False) -> tuple[Path, ...]:
    """Write the sample files into ``folder`` and return their paths, in
    :data:`FILES` order.

    ``folder`` is created with its parents if missing. A sample file already
    there with the same bytes is left as it is. One with other bytes is
    replaced only with ``force``; without it, :class:`SampleConflictError` names
    every such file and nothing is written. Other files in the folder are never
    touched. Each file is replaced atomically
    (:func:`~proteia.core.storage.write_atomic`), the truth table last. Before
    any is replaced, those another program holds (a table open in a spreadsheet
    on Windows) are named in :class:`SampleWriteError` and nothing is written; a
    write failing after others were replaced raises it naming both, since the
    images and the truth table must be of one version.
    """
    folder = Path(folder)
    files = sample_files()
    paths = {name: folder / name for name in files}
    same = {name for name, data in files.items() if _holds(paths[name], data)}
    differ = [name for name in files if name not in same and paths[name].exists()]
    if differ and not force:
        raise SampleConflictError(folder, differ)
    folder.mkdir(parents=True, exist_ok=True)
    pending = [name for name in files if name not in same]
    held = [name for name in pending if _held(paths[name])]
    if held:
        raise SampleWriteError(folder, (), held)
    for index, name in enumerate(pending):
        try:
            write_atomic(paths[name], files[name])
        except OSError as exc:
            raise SampleWriteError(folder, pending[:index], pending[index:]) from exc
    return tuple(paths.values())


def _holds(path: Path, data: bytes) -> bool:
    return path.is_file() and path.read_bytes() == data


def _held(path: Path) -> bool:
    """Whether another program holds the existing file so it cannot be
    replaced (Windows locks a file a spreadsheet has open)."""
    try:
        with path.open("r+b"):
            return False
    except FileNotFoundError:
        return False
    except PermissionError:
        return True


def _tolerant_console() -> None:
    """Escape what the console encoding cannot show (a path with µ in a
    code-page console) rather than fail."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m proteia.samples``: write the samples and print their paths;
    0 on success, 1 when nothing could be written."""
    _tolerant_console()
    parser = argparse.ArgumentParser(
        prog="python -m proteia.samples",
        description=(
            "Write Proteia's synthetic sample images (a 16-bit chemiluminescence blot"
            " and a visible-light marker image of the same membrane) and the table of"
            " their true band signals into FOLDER."
        ),
        epilog=(
            "Sample files already in FOLDER with the same contents are left as they"
            " are. If any differ, nothing is written unless --force is given. Other"
            " files in FOLDER are never touched. To check Proteia's results against"
            f" {TRUTH_FILE}, match rows on the sample (V1-V4, T1-T4), not the lane:"
            " the truth table numbers lanes from 1, as the app shows them, and the"
            " lane-table export numbers them from 0."
        ),
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=DEFAULT_FOLDER,
        metavar="FOLDER",
        help=f"where to write them (default: {DEFAULT_FOLDER} in the current"
        " directory); created if missing",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite sample files in FOLDER that differ"
    )
    args = parser.parse_args(argv)
    try:
        paths = generate(args.folder, force=args.force)
    except SampleConflictError as exc:
        print(
            f"{exc}. Give another folder, or run again with --force to overwrite.", file=sys.stderr
        )
        return 1
    except OSError as exc:
        print(f"Cannot write the samples: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
