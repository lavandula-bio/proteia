# SPDX-License-Identifier: Apache-2.0
"""Synthetic rows for the row-box detection tests (#51).

A band is a horizontal smear, ``depth * fx(x) * fy(y)``: along x a flat-topped
super-Gaussian ``exp(-0.5 |(x - cx) / sx|**4)`` (or a Gaussian, whose tails are
longer), along y a Gaussian. The membrane is 50000 in 16-bit units, darkened by
the bands, with Gaussian noise, clipped to [0, 65535] (an over-exposed dark band
is flat at 0) and rounded; a light-on-dark row is the inverse ``65535 - image``.

A band's reference rect is its own noise-free, unclipped extent at 20% of its
peak depth, rounded inward to whole pixels (half-open), so neighbouring
references may overlap. The row box spans every declared lane's nominal band,
the empty ones included, plus a margin.

* :func:`synthetic_row` and :func:`bench_cases`: the fifteen rows of the #51
  benchmark (bench51), rebuilt bit for bit: same parameters, same random draws;
  :data:`BENCH_RECIPES` and :func:`bench_row` give each by name, at any seed
  and membrane noise.
* :func:`adversarial_row` and :data:`ADVERSARIAL`: the accuracy judge's
  generator and the recipes the tests use (neighbouring rows, doublets,
  streaks, stains, dust between lanes, bubbles, a panel's frame), plus rows of
  the tests' own (a row large enough for the fits' subsample, guards that
  bind).
* :data:`STRESS` (#180): rows that make each known kind of silent error show
  (a stain on one side of a band's background ring, row boxes cut through
  bands, dust, a box over two rows, a mark past the last lane), kept apart
  from :data:`ADVERSARIAL`; :func:`neighbour_grid_row`: a row with a
  neighbouring band at a set distance and strength in every lane.
* :class:`Recipe` and :func:`score_row` (#180): a row's lanes scored against
  the recipe's own truth: wrong lanes, and wrong lanes the page shows no flag
  for.
* :func:`jpeg`: a row as an 8-bit JPEG export, read back (block artefacts).
* :func:`fuzz_row`: seeded random geometry for the invariant checks.
"""

import dataclasses
import io
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from proteia.core import evaluate, rowdetect
from proteia.core.model import Rect

MEMBRANE = 50000.0
FULL_SCALE = 65535.0
NOISE_SIGMA = 400.0
REF_FRACTION = 0.2
_UX4 = (2.0 * math.log(1.0 / REF_FRACTION)) ** 0.25  # super-Gaussian half-extent at 20%
_UX2 = math.sqrt(2.0 * math.log(1.0 / REF_FRACTION))  # Gaussian half-extent at 20%

Artefact = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RowCase:
    """One synthetic row: ``image`` (float64, integer-valued), the ``row`` box,
    ``n_lanes`` and the ``reference`` rect of each lane with a band.
    ``lane_cx`` and ``lane_cy`` are every lane's true centre, the empty ones
    included. ``neighbour_cy`` is the centre of each lane's neighbouring band
    above or below, where :func:`neighbour_grid_row` draws one; None
    otherwise."""

    name: str
    description: str
    image: np.ndarray
    dark_on_light: bool
    row: Rect
    n_lanes: int
    reference: dict[int, Rect]
    lane_cx: tuple[float, ...]
    lane_cy: tuple[float, ...]
    neighbour_cy: tuple[float, ...] | None = None


def _ref_rect(cx: float, cy: float, w: float, h: float) -> Rect:
    return (
        math.ceil(cx - w / 2),
        math.ceil(cy - h / 2),
        math.floor(cx + w / 2) + 1,
        math.floor(cy + h / 2) + 1,
    )


# --- The bench51 rows ---

_BENCH_MARGIN_X = 70  # px of membrane left and right of the row
_BENCH_HEIGHT = 120
_BENCH_ROW_CY = 60.0


def synthetic_row(
    name: str,
    description: str,
    seed: int,
    *,
    n: int = 6,
    pitch: float = 70.0,
    w: float = 44.0,
    h: float = 12.0,
    pitches: Sequence[float] | None = None,
    missing: Sequence[int] = (),
    smile: float = 0.0,
    depth_range: tuple[float, float] = (18000.0, 30000.0),
    depth_override: Mapping[int, float | str] | None = None,
    gradient: tuple[float, float] = (0.0, 0.0),
    light_on_dark: bool = False,
    mx: int = 10,
    my: int = 6,
    x_jitter: float = 2.0,
    y_jitter: float = 1.0,
    w_var: float = 0.08,
    h_var: float = 0.10,
    noise: float = NOISE_SIGMA,
) -> RowCase:
    """One bench51 row: ``n`` lanes at ``pitch`` (or the steps ``pitches``), bands
    about ``w`` x ``h`` px (the 20% extents), ``missing`` lanes empty. ``smile``
    lifts the outer bands; ``gradient`` tilts the membrane across and down the
    row; ``depth_override`` sets a lane's depth, ``"faint10"`` a tenth of the
    others' mean; ``mx``/``my`` are the row box margins (negative cuts bands).
    ``noise`` is the membrane noise's sigma: any value scales the same draws,
    so a noisier row keeps its bands, box and noise pattern."""
    rng = np.random.default_rng(seed)
    steps = list(pitches) if pitches is not None else [pitch] * (n - 1)
    cx0 = _BENCH_MARGIN_X + w / 2
    lane_cx = cx0 + np.concatenate([[0.0], np.cumsum(steps)])
    lane_cx = lane_cx + rng.uniform(-x_jitter, x_jitter, n)
    xc = 0.5 * (lane_cx[0] + lane_cx[-1])
    half = max(0.5 * (lane_cx[-1] - lane_cx[0]), 1.0)
    u = (lane_cx - xc) / half
    lane_cy = _BENCH_ROW_CY + smile * (0.5 - u**2) + rng.uniform(-y_jitter, y_jitter, n)
    widths = w * rng.uniform(1 - w_var, 1 + w_var, n)
    heights = h * rng.uniform(1 - h_var, 1 + h_var, n)
    depths = rng.uniform(*depth_range, n)
    for lane, value in (depth_override or {}).items():
        if value == "faint10":
            depths[lane] = float(np.mean([depths[i] for i in range(n) if i != lane])) / 10.0
        else:
            depths[lane] = float(value)

    width_img = int(math.ceil(lane_cx[-1] + widths[-1] / 2 + _BENCH_MARGIN_X))
    xs = np.arange(width_img, dtype=float)
    ys = np.arange(_BENCH_HEIGHT, dtype=float)
    gx, gy = gradient
    bg = MEMBRANE + np.zeros((_BENCH_HEIGHT, width_img))
    if gx or gy:
        bg = (
            bg + gx * ((xs - xc) / half)[None, :] + gy * ((ys - _BENCH_ROW_CY) / (2.0 * h))[:, None]
        )
    depth_map = np.zeros_like(bg)
    all_refs: dict[int, Rect] = {}
    reference: dict[int, Rect] = {}
    for i in range(n):
        all_refs[i] = _ref_rect(lane_cx[i], lane_cy[i], widths[i], heights[i])
        if i in set(missing):
            continue
        sx = widths[i] / (2.0 * _UX4)
        sy = heights[i] / (2.0 * _UX2)
        fx = np.exp(-0.5 * np.abs((xs - lane_cx[i]) / sx) ** 4)
        fy = np.exp(-0.5 * ((ys - lane_cy[i]) / sy) ** 2)
        depth_map += depths[i] * np.outer(fy, fx)
        reference[i] = all_refs[i]
    grain = rng.normal(0.0, noise, bg.shape)
    image = np.round(np.clip(bg - depth_map + grain, 0.0, FULL_SCALE))
    if light_on_dark:
        image = FULL_SCALE - image
    row = (
        max(0, min(r[0] for r in all_refs.values()) - mx),
        max(0, min(r[1] for r in all_refs.values()) - my),
        min(width_img, max(r[2] for r in all_refs.values()) + mx),
        min(_BENCH_HEIGHT, max(r[3] for r in all_refs.values()) + my),
    )
    return RowCase(
        name,
        description,
        image.astype(np.float64),
        not light_on_dark,
        row,
        n,
        reference,
        tuple(float(c) for c in lane_cx),
        tuple(float(c) for c in lane_cy),
    )


_BASE = 51  # all_present and the missing-lane rows share geometry and noise

# The fifteen bench51 rows by name: (description, seed, keyword arguments of
# synthetic_row).
BENCH_RECIPES: dict[str, tuple[str, int, dict]] = {
    "all_present": ("6 lanes, pitch 70, W~44 H~12, depths 18k-30k", _BASE, {}),
    "missing_first": ("lane 0 empty (the box covers it)", _BASE, {"missing": [0]}),
    "missing_middle": ("lane 2 empty", _BASE, {"missing": [2]}),
    "missing_last": ("lane 5 empty (the box covers it)", _BASE, {"missing": [5]}),
    "missing_two": ("lanes 0 and 3 empty", _BASE, {"missing": [0, 3]}),
    "touching": (
        "pitch 48, W 60: neighbours overlap at half height, no valley between them",
        52,
        {
            "pitch": 48.0,
            "w": 60.0,
            "w_var": 0.0,
            "x_jitter": 0.0,
            "depth_range": (22000.0, 26000.0),
        },
    ),
    "smile": ("outer bands 8 px higher than the middle", 53, {"smile": 8.0}),
    "uneven_spacing": (
        "pitches 56/84/63/80/60 (70 +/-20%), W~40",
        54,
        {"w": 40.0, "pitches": [56.0, 84.0, 63.0, 80.5, 59.5], "x_jitter": 0.0},
    ),
    "faint_band": ("lane 3 ten times fainter", 55, {"depth_override": {3: "faint10"}}),
    "overexposed": ("lane 2 clipped flat at 0", 56, {"depth_override": {2: 1.5 * MEMBRANE}}),
    "uneven_background": (
        "membrane -6000..+6000 across the row, +/-1000 down it",
        57,
        {"gradient": (6000.0, 1000.0)},
    ),
    "light_on_dark": ("inverted: membrane 15535, bright bands", 58, {"light_on_dark": True}),
    "loose_box": ("margins half a pitch across, 15 px down", 59, {"mx": 35, "my": 15}),
    "tight_box": ("cuts 4 px off both outer bands, 1 px down", 60, {"mx": -4, "my": 1}),
    "twelve_lanes": (
        "12 lanes, pitch 50, W~34 H~10",
        61,
        {"n": 12, "pitch": 50.0, "w": 34.0, "h": 10.0},
    ),
}


def bench_row(name: str, seed: int | None = None, *, noise: float = NOISE_SIGMA) -> RowCase:
    """The bench51 row ``name`` at its own seed (or ``seed``), with membrane
    noise ``noise``."""
    description, own_seed, recipe = BENCH_RECIPES[name]
    return synthetic_row(
        name, description, own_seed if seed is None else seed, noise=noise, **recipe
    )


def bench_cases() -> list[RowCase]:
    """The fifteen bench51 rows (91 reference bands), rebuilt on every call."""
    return [bench_row(name) for name in BENCH_RECIPES]


# --- The accuracy judge's generator ---


def _band(
    xs: np.ndarray,
    ys: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    depth: float,
    shape: str,
) -> np.ndarray:
    if shape == "super":
        fx = np.exp(-0.5 * np.abs((xs - cx) / (w / (2 * _UX4))) ** 4)
    else:
        fx = np.exp(-0.5 * ((xs - cx) / (w / (2 * _UX2))) ** 2)
    fy = np.exp(-0.5 * ((ys - cy) / (h / (2 * _UX2))) ** 2)
    return depth * np.outer(fy, fx)


def adversarial_row(
    name: str,
    seed: int,
    *,
    n: int = 6,
    pitch: float = 70.0,
    pitches: Sequence[float] | None = None,
    w: float = 44.0,
    h: float = 12.0,
    missing: Sequence[int] = (),
    depth_range: tuple[float, float] = (18000.0, 30000.0),
    depths: Mapping[int, float] | None = None,
    widths: Mapping[int, float] | None = None,
    heights: Mapping[int, float] | None = None,
    shape: str = "super",
    tilt: float = 0.0,
    smile: float = 0.0,
    doublet: Mapping[int, tuple[float, float]] | None = None,
    holes: Sequence[tuple[int, float, float, float]] = (),
    neighbour_dy: float | None = None,
    neighbour_rel: float = 1.0,
    neighbour_rels: Mapping[int, float] | None = None,
    shifts: Mapping[int, float] | None = None,
    artefacts: Sequence[Artefact] = (),
    noise: float = NOISE_SIGMA,
    light_on_dark: bool = False,
    mx: int = 10,
    my: int = 6,
    box_adjust: tuple[int, int, int, int] = (0, 0, 0, 0),
    img_h: int = 160,
    x_jitter: float = 2.0,
    y_jitter: float = 1.0,
    margin_left: int = 70,
    margin_right: int = 70,
) -> RowCase:
    """A row from the accuracy judge's generator (acc_adv), the same draws.

    Beyond :func:`synthetic_row`: ``doublet`` maps a lane to ``(dy, frac)``, two
    components ``dy`` apart, the second ``frac`` as deep (reference: their union);
    ``holes`` are ``(lane, dx, dy, r)``: a round hole of radius ``r`` punched into
    the bands (multiplicative, a transfer bubble) that far from the lane's centre;
    ``neighbour_dy`` adds a neighbouring row that far below (above if
    negative), ``neighbour_rel`` as deep (``neighbour_rels`` sets a lane's
    own); ``shifts`` moves single lanes' bands down (up if negative) by that
    many px, as a montage's panel or a mark beside the row lies, the row box
    spanning them;
    ``artefacts`` add darkening maps ``f(X, Y, lane_cx, lane_cy)``; ``widths``,
    ``heights`` and ``depths`` override single bands; ``box_adjust`` moves the
    row box's edges; ``margin_left`` and ``margin_right`` are the membrane
    left of the first band and right of the last (room for what lies beside
    the row, :func:`beside`)."""
    rng = np.random.default_rng(seed)
    steps = list(pitches) if pitches is not None else [pitch] * (n - 1)
    lane_cx = (margin_left + w / 2 + np.concatenate([[0.0], np.cumsum(steps)])) + rng.uniform(
        -x_jitter, x_jitter, n
    )
    row_cy = img_h / 2
    xc = 0.5 * (lane_cx[0] + lane_cx[-1])
    half = max(0.5 * (lane_cx[-1] - lane_cx[0]), 1.0)
    u = (lane_cx - xc) / half
    lane_cy = row_cy + smile * (0.5 - u**2) + tilt * u / 2 + rng.uniform(-y_jitter, y_jitter, n)
    for k, v in (shifts or {}).items():
        lane_cy[k] += v
    ws = w * rng.uniform(0.92, 1.08, n)
    hs = h * rng.uniform(0.9, 1.1, n)
    for k, v in (widths or {}).items():
        ws[k] = v
    for k, v in (heights or {}).items():
        hs[k] = v
    dps = rng.uniform(*depth_range, n)
    for k, v in (depths or {}).items():
        dps[k] = v
    width_img = int(math.ceil(lane_cx[-1] + ws[-1] / 2 + margin_right))
    xs = np.arange(width_img, dtype=float)
    ys = np.arange(img_h, dtype=float)
    base = np.full((img_h, width_img), MEMBRANE)
    dmap = np.zeros_like(base)
    refs_all: dict[int, Rect] = {}
    reference: dict[int, Rect] = {}
    for i in range(n):
        refs_all[i] = _ref_rect(lane_cx[i], lane_cy[i], ws[i], hs[i])
        if i in set(missing):
            continue
        if doublet and i in doublet:
            dy, frac = doublet[i]
            hh = hs[i] * 0.7
            c1, c2 = lane_cy[i] - dy / 2, lane_cy[i] + dy / 2
            dmap += _band(xs, ys, lane_cx[i], c1, ws[i], hh, dps[i], shape)
            dmap += _band(xs, ys, lane_cx[i], c2, ws[i], hh, dps[i] * frac, shape)
            r1, r2 = _ref_rect(lane_cx[i], c1, ws[i], hh), _ref_rect(lane_cx[i], c2, ws[i], hh)
            reference[i] = (
                min(r1[0], r2[0]),
                min(r1[1], r2[1]),
                max(r1[2], r2[2]),
                max(r1[3], r2[3]),
            )
        else:
            dmap += _band(xs, ys, lane_cx[i], lane_cy[i], ws[i], hs[i], dps[i], shape)
            reference[i] = refs_all[i]
    for lane, hdx, hdy, hr in holes:
        hx, hy = lane_cx[lane] + hdx, lane_cy[lane] + hdy
        dmap *= 1.0 - np.exp(
            -0.5 * (((xs[None, :] - hx) / hr) ** 2 + ((ys[:, None] - hy) / hr) ** 2)
        )
    if neighbour_dy is not None:
        for i in range(n):
            cy = lane_cy[i] + neighbour_dy
            rel = (neighbour_rels or {}).get(i, neighbour_rel)
            dmap += _band(xs, ys, lane_cx[i], cy, ws[i], h, dps[i] * rel, shape)
    for f in artefacts:
        dmap += f(xs[None, :], ys[:, None], lane_cx, lane_cy)
    nz = rng.normal(0.0, 1.0, base.shape)
    image = np.clip(base - dmap + noise * nz, 0.0, FULL_SCALE)
    if light_on_dark:
        image = FULL_SCALE - image
    image = np.round(image)  # after the inversion, as the judge's generator
    row = (
        max(0, min(r[0] for r in refs_all.values()) - mx + box_adjust[0]),
        max(0, min(r[1] for r in refs_all.values()) - my + box_adjust[1]),
        min(width_img, max(r[2] for r in refs_all.values()) + mx + box_adjust[2]),
        min(img_h, max(r[3] for r in refs_all.values()) + my + box_adjust[3]),
    )
    return RowCase(
        name,
        "",
        image.astype(np.float64),
        not light_on_dark,
        row,
        n,
        reference,
        tuple(float(c) for c in lane_cx),
        tuple(float(c) for c in lane_cy),
    )


def blob(
    lane: int,
    r: float,
    depth: float,
    *,
    ry: float | None = None,
    dy: float = 0.0,
    dx: float = 0.0,
) -> Artefact:
    """A round dark blob (dust, a stain) of radius ``r`` on ``lane``'s band centre
    (``dy`` px below it, ``dx`` px right of it); a negative depth is a light
    spot. ``ry`` makes it an ellipse, ``r`` px across and ``ry`` px high."""
    ry = r if ry is None else ry

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        cx, cy = lcx[lane] + dx, lcy[lane] + dy
        return depth * np.exp(-0.5 * (((X - cx) / r) ** 2 + ((Y - cy) / ry) ** 2))

    return f


def blob_between(left: int, r: float, depth: float) -> Artefact:
    """A round blob of radius ``r`` midway between lanes ``left`` and ``left + 1``,
    at the band height of ``left``; a negative depth is a light spot."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        cx = 0.5 * (lcx[left] + lcx[left + 1])
        return depth * np.exp(-0.5 * (((X - cx) / r) ** 2 + ((Y - lcy[left]) / r) ** 2))

    return f


def band_between(left: int, w: float, h: float, depth: float, dy: float = 0.0) -> Artefact:
    """An extra flat-topped band, ``w`` x ``h`` px at 20%, midway between lanes
    ``left`` and ``left + 1``, at the band height of ``left`` (``dy`` px below
    it)."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        cx = 0.5 * (lcx[left] + lcx[left + 1])
        return _band(X[0], Y[:, 0], cx, lcy[left] + dy, w, h, depth, "super")

    return f


def beside(
    lanes: float,
    w: float,
    h: float,
    depth: float,
    *,
    marks: int = 1,
    gap: float = 0.0,
    dy: float = 0.0,
) -> Artefact:
    """What lies beside the row and is no lane of it: ``marks`` flat-topped
    marks ``w`` x ``h`` px at 20%, ``gap`` px apart, the first centred
    ``lanes`` lane steps past the last lane's centre (before the first lane's
    if negative, the marks then running left), at the row's mean band height
    (``dy`` px below it). A ladder's band or a tick mark is one narrow mark, a
    label's text a few thin strokes, a neighbouring panel's lane a band."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        if lanes >= 0:
            x0, step = lcx[-1] + lanes * (lcx[-1] - lcx[-2]), w + gap
        else:
            x0, step = lcx[0] + lanes * (lcx[1] - lcx[0]), -(w + gap)
        cy = float(np.mean(lcy)) + dy
        out = np.zeros((Y.shape[0], X.shape[1]))
        for k in range(marks):
            out += _band(X[0], Y[:, 0], x0 + k * step, cy, w, h, depth, "super")
        return out

    return f


def vstreak(lane: int, half_w: float, depth: float) -> Artefact:
    """A dark vertical streak down ``lane`` over the whole image height."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        return depth * np.exp(-0.5 * np.abs((X - lcx[lane]) / half_w) ** 4) + 0 * Y

    return f


def hstripe(half_h: float, depth: float) -> Artefact:
    """A darker stretch of membrane: a flat stripe across the whole image width,
    ``2 * half_h`` px high about the row's mean band height."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        return depth * (np.abs(Y - float(np.mean(lcy))) < half_h) + 0 * X

    return f


def frame(dy: float, px: int, depth: float, *, dx: float = 50.0, slope: float = 0.0) -> Artefact:
    """A panel's drawn frame: sharp lines ``px`` px thick, ``dy`` px above and
    below the row's mean band height, joined by sides ``dx`` px outside the
    end lanes' centres. ``slope`` tilts the lines about the row's centre (a
    scan turned a little); the sides run between them."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        cy = float(np.mean(lcy))
        shift = slope * (X - 0.5 * (lcx[0] + lcx[-1]))
        top, bottom = np.floor(cy - dy + shift), np.floor(cy + dy + shift)
        left, right = math.floor(lcx[0] - dx), math.floor(lcx[-1] + dx)
        across = (X >= left) & (X < right + px)
        down = (Y >= top) & (Y < bottom + px)
        lines = ((Y >= top) & (Y < top + px)) | ((Y >= bottom) & (Y < bottom + px))
        sides = ((X >= left) & (X < left + px)) | ((X >= right) & (X < right + px))
        return depth * ((across & lines) | (down & sides))

    return f


def bottom_strip(rows: int, depth: float, fade: float = 0.25) -> Artefact:
    """A dark strip over the image's bottom ``rows`` rows, across its whole
    width (a screenshot's toolbar): ``depth`` on its top row, fading by
    ``fade`` of that to the image's last row."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        top = Y.max() + 1 - rows
        return depth * (Y >= top) * (1.0 - fade * (Y - top) / max(1, rows - 1)) + 0 * X

    return f


def dark_edge(cols: int, depth: float) -> Artefact:
    """The image's right edge darkening over its last ``cols`` columns, to
    ``depth`` on its last one (a vignetted scan)."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        ramp = np.clip((X - (X.max() - cols)) / cols, 0.0, 1.0)
        return depth * ramp**2 + 0 * Y

    return f


def shade_above(end: int, rows: int, depth: float) -> Artefact:
    """The membrane darkening over the ``rows`` rows above image row ``end``,
    to ``depth`` on the last of them, across the whole width: a vignetted
    scan whose image ends at ``end`` (cut it there with :func:`image_cut`)."""

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        ramp = np.clip((Y - (end - 1 - rows)) / rows, 0.0, 1.0)
        return depth * ramp**2 * (Y < end) + 0 * X

    return f


def ring_stain(
    lane: int,
    side: str,
    depth: float,
    *,
    near: float = 13.0,
    far: float = 40.0,
    half: float = 40.0,
) -> Artefact:
    """A flat stain beside ``lane``'s band, ``depth`` deep, on its ``side``
    (``"top"`` or ``"bottom"``): from ``near`` to ``far`` px above or below the
    band's centre, ``half`` px either side of it along the row. Over a band's
    background ring, it covers that side of the ring and none of the band."""
    if side not in ("top", "bottom"):
        raise ValueError(f"side must be 'top' or 'bottom', not {side!r}")
    sign = -1.0 if side == "top" else 1.0

    def f(X: np.ndarray, Y: np.ndarray, lcx: np.ndarray, lcy: np.ndarray) -> np.ndarray:
        away = sign * (Y - lcy[lane])  # px from the band's centre, on the stain's side
        return depth * ((away >= near) & (away <= far) & (np.abs(X - lcx[lane]) <= half))

    return f


def image_cut(case: RowCase, *, top: int = 0, bottom: int | None = None) -> RowCase:
    """``case`` with its image cut to the rows ``[top, bottom)`` and its row
    box dragged to each cut (to the image's new top or bottom row), as over a
    tightly cropped image; the reference and the lanes' heights move with the
    image."""
    x0, y0, x1, y1 = case.row
    end = case.image.shape[0] if bottom is None else bottom
    return dataclasses.replace(
        case,
        image=case.image[top:end],
        row=(x0, 0 if top else y0, x1, (end if bottom is not None else y1) - top),
        reference={
            lane: (r[0], r[1] - top, r[2], r[3] - top) for lane, r in case.reference.items()
        },
        lane_cy=tuple(cy - top for cy in case.lane_cy),
    )


def frame_recipe(dy: int, px: int) -> dict:
    """The recipe of a row with lane 2 empty inside a panel's drawn frame (#116),
    lines ``px`` px thick ``dy`` px above and below the bands' centres, and a
    box drawn over the whole frame, its sides included."""
    return {
        "missing": [2],
        "artefacts": [frame(dy, px, 20000.0)],
        "box_adjust": (-30, -(dy - 4), 30, dy - 4),
    }


def framed(dy: int, px: int) -> RowCase:
    """The row of :func:`frame_recipe` at seed 1000."""
    return adversarial_row("framed", 1000, **frame_recipe(dy, px))


# The judge's recipes the tests use (seeds 1000 and up, as the judge ran them),
# then rows of the tests' own.
ADVERSARIAL: dict[str, dict] = {
    "nbr_above_miss": {
        "missing": [2],
        "neighbour_dy": -22.0,
        "box_adjust": (0, -10, 0, 0),
    },
    "blotch_empty": {"missing": [4], "artefacts": [blob(4, 40, 6000)]},
    "vstreak_empty": {"missing": [4], "artefacts": [vstreak(4, 14, 3000)]},
    "box_omits_empty_first": {"missing": [0], "box_adjust": (70, 0, 0, 0)},
    "doublet_two": {"doublet": {1: (9, 0.6), 4: (9, 0.6)}},
    "gauss_tails_miss": {"shape": "gauss", "pitch": 56.0, "missing": [2]},
    "twenty_lanes": {
        "n": 20,
        "pitch": 36.0,
        "w": 24.0,
        "h": 10.0,
        "missing": [0, 7, 8, 19],
        "mx": 6,
    },
    # Dust between lanes 2 and 3: one piece more than lanes, merged or dropped.
    "blob_gap": {"artefacts": [blob_between(2, 5, 15000)]},
    # A bubble splits lane 1's band in two; a light spot between lanes 3 and 4.
    "bubble_band": {"holes": [(1, 0.0, 0.0, 7.0)], "artefacts": [blob_between(3, 7, -1500)]},
    # The tests' own rows. Lane 3 above twice the median height: size_outlier.
    "tall_band": {"heights": {3: 30.0}},
    # Lane 2 holds two components 14 px apart, the lower one 0.7x as deep.
    "doublet_deep": {"doublet": {2: (14, 0.7)}, "my": 8},
    # A smile, a tall band and a tight box: the vertical placement clamp binds.
    "smile_tall_tight": {"smile": 10.0, "heights": {4: 23.7}, "box_adjust": (0, 3, 0, -3)},
    # Three times the bench scale: every fit and noise estimate subsamples.
    "wide_row": {
        "pitch": 210.0,
        "w": 132.0,
        "h": 36.0,
        "img_h": 240,
        "mx": 30,
        "my": 18,
        "missing": [2],
    },
    # Touching bands, the first a quarter as deep as the rest: its end of the
    # run is trimmed against its neighbour's peak once the envelope reaches it.
    "touching_weak_end": {"pitch": 48.0, "w": 60.0, "depths": {0: 6000.0}},
    # Lane 2 spread to 1.4 pitches over empty lane 3 (the judge's
    # wide_band_next_empty): the piece is lane 2's, and the kept signal it
    # spills into lane 3 fits no lane.
    "wide_next_empty": {"widths": {2: 100.0}, "missing": [3]},
    # A tilted row (lane 1 highest) whose box's top edge runs 3 px above the
    # highest band's centre: it cuts the bands of lanes 1 and 2 only.
    "tilt_cut": {"tilt": 10.0, "box_adjust": (0, 9, 0, 0)},
    # #111: a first row box that also covers what lies beside the row, read a
    # lane or more off. A ladder's band 1.8 lane steps past the last lane,
    # lane 0 empty: read as the last lane, each band a lane early.
    "ladder_beside": {
        "missing": [0],
        "artefacts": [beside(1.8, 26, 12, 14000)],
        "margin_right": 200,
        "box_adjust": (0, 0, 150, 0),
    },
    # The same before the first lane, the last lane empty.
    "ladder_before": {
        "missing": [5],
        "artefacts": [beside(-1.8, 26, 12, 14000)],
        "margin_left": 200,
        "box_adjust": (-150, 0, 0, 0),
    },
    # A label's text, four strokes 1.3 lane steps past the last lane, lane 0 empty.
    "label_beside": {
        "missing": [0],
        "artefacts": [beside(1.3, 10, 14, 20000, marks=4, gap=4)],
        "margin_right": 200,
        "box_adjust": (0, 0, 150, 0),
    },
    # Five lanes, the first two empty; an arrow 0.8 lane steps past the last
    # and a neighbouring panel's band 2.2 past it.
    "panel_beside": {
        "n": 5,
        "missing": [0, 1],
        "artefacts": [beside(0.8, 20, 6, 15000), beside(2.2, 44, 12, 25000)],
        "margin_right": 240,
        "box_adjust": (0, 0, 190, 0),
    },
    # #121: lane 1's band pale across its middle, its two ends about 1.7x as
    # dark (a dumbbell); the same band over-exposed, clipped flat at 0 around
    # its lighter centre (a hollow band).
    "dumbbell_band": {"depths": {1: 24000.0}, "artefacts": [blob(1, 6.0, -11000.0, ry=12.0)]},
    "hollow_band": {
        "depths": {1: 1.5 * MEMBRANE},
        "artefacts": [blob(1, 6.0, -45000.0, ry=12.0)],
    },
    # #121: lane 1's band 2.5x as deep as the membrane, clipped at 0 all around
    # a lighter centre, back to 40000 there (a ring: burnt out in its middle);
    # the band of hollow_band with dark spots at its two ends, 6 px above its
    # middle (a notch in its top edge, as a band whose ends curve up shows).
    "hollow_ring": {
        "depths": {1: 2.5 * MEMBRANE},
        "artefacts": [blob(1, 7.0, -2.3 * MEMBRANE, ry=2.0)],
        "my": 10,
    },
    "notched_band": {
        "depths": {1: 1.5 * MEMBRANE},
        "artefacts": [blob(1, 4.0, 60000.0, dx=x, dy=-6.0) for x in (-13.0, 13.0)],
        "my": 10,
    },
    # #116: lane 2 empty inside a panel's drawn frame, 2 px lines 16 px above
    # and below the bands, the box over the whole frame: the lines are no bands.
    "frame": frame_recipe(16, 2),
}


def adversarial(key: str, seed: int) -> RowCase:
    """The recipe ``key`` of :data:`ADVERSARIAL` at ``seed``."""
    return adversarial_row(key, seed, **ADVERSARIAL[key])


# --- #180: rows that make a silent error show ---


def two_rows_recipe(above: Mapping[int, float], rel: float = 1.0) -> dict:
    """The recipe of a row with another row 40 px above it, the row box dragged
    over both: the row above ``above[lane]`` (else ``rel``) times as deep as
    the row's own band, so a lane where it is deeper holds its strongest band
    there."""
    return {
        "neighbour_dy": -40.0,
        "neighbour_rel": rel,
        "neighbour_rels": dict(above),
        "box_adjust": (0, -40, 0, 0),
        "img_h": 200,
    }


def two_rows(seed: int, above: Mapping[int, float], rel: float = 1.0, **kwargs) -> RowCase:
    """The row of :func:`two_rows_recipe` at ``seed``; ``kwargs`` go to
    :func:`adversarial_row`."""
    return adversarial_row("two rows", seed, **two_rows_recipe(above, rel), **kwargs)


def mark_past_end(lanes: float) -> dict:
    """The recipe of a row whose first lane is empty and whose box also covers
    an arrow (a short mark, 20 x 6 px) ``lanes`` lane steps past the last
    lane's centre, the box's right edge 30 px past the arrow's centre: the
    bands fit the lanes shifted by one as well as the true ones (#111)."""
    return {
        "missing": [0],
        "artefacts": [beside(lanes, 20, 6, 15000)],
        "margin_right": 200,
        "box_adjust": (0, 0, round(70 * lanes + 30), 0),
    }


# Rows the adversarial tests (#180) score beside ADVERSARIAL, each the kind
# of row a known error shows on, at seeds 1000 and up. They are kept out of
# ADVERSARIAL, whose every recipe the golden file pins and the honest-row
# checks read: some of these are refused by design, and some are read wrong
# today (their tests are expected failures, each citing its issue).
STRESS: dict[str, dict] = {
    # Weak bands (3000 deep) and a faint flat stain, one noise sigma deep, over
    # the side of lane 2's background ring above its band (#177).
    "faint_ring_stain": {
        "depth_range": (3000.0, 3000.0),
        "artefacts": [ring_stain(2, "top", 400.0)],
    },
    # The same stain over ordinary bands, then four and ten times as deep.
    "ring_stain": {"artefacts": [ring_stain(2, "top", 400.0)]},
    "ring_stain_x4": {"artefacts": [ring_stain(2, "top", 1600.0)]},
    "deep_ring_stain": {"artefacts": [ring_stain(2, "top", 4000.0)]},
    # A stain ten sigma deep below lane 2's band (the tests crop the image
    # 2 px above the boxes: the ring keeps only the stained side).
    "stain_below": {"artefacts": [ring_stain(2, "bottom", 4000.0)]},
    # The row box's left and right edges 12 px inside both end bands'
    # extents (#178); only its right edge, 12 px inside the last band's.
    "side_cut": {"mx": -12},
    "right_side_cut": {"box_adjust": (0, 0, -22, 0)},
    # Both side edges 4 px inside the end bands, as the bench's tight_box:
    # a little signal cut off, honestly boxed.
    "tight_sides": {"mx": -4},
    # The box's bottom edge 10 px up: it cuts every band.
    "bottom_cut": {"box_adjust": (0, 0, 0, -10)},
    # Lanes 1, 3 and 4 empty (the tests raise the noise).
    "empty_three": {"missing": [1, 3, 4]},
    # Dust on empty lane 3: a round speck of sigma 1.5 or 2 px.
    "dust": {"missing": [3], "artefacts": [blob(3, 1.5, 20000.0)]},
    "dust_2px": {"missing": [3], "artefacts": [blob(3, 2.0, 20000.0)]},
    # The box dragged over its row and the row 40 px above, whose bands are
    # twice as deep in lanes 0-2 and half as deep elsewhere; the same with
    # lanes 0-2 empty in the row and the row above only there.
    "two_rows": two_rows_recipe({0: 2.0, 1: 2.0, 2: 2.0}, rel=0.5),
    "two_rows_near_empty": {
        **two_rows_recipe({0: 2.0, 1: 2.0, 2: 2.0}, rel=0.0),
        "missing": [0, 1, 2],
    },
    # An arrow past the last lane, 0.9 to 1.3 lane steps away (#111).
    **{f"arrow_{lanes:g}": mark_past_end(lanes) for lanes in (0.9, 1.0, 1.1, 1.3)},
}


def stress(key: str, seed: int, *, noise: float = NOISE_SIGMA) -> RowCase:
    """The recipe ``key`` of :data:`STRESS` at ``seed``."""
    return adversarial_row(key, seed, noise=noise, **STRESS[key])


def neighbour_grid_row(
    name: str,
    seed: int,
    *,
    n: int = 6,
    pitch: float = 70.0,
    w: float = 44.0,
    h: float = 12.0,
    depth_range: tuple[float, float] = (18000.0, 30000.0),
    depths: Sequence[float] | None = None,
    faint: int | None = None,
    missing: Iterable[int] = (),
    neighbour_dy: float | None = None,
    neighbour_rel: float = 1.0,
    neighbour_depths: Sequence[float] | None = None,
    smile: float = 0.0,
    tilt: float = 0.0,
    doublet: tuple[float, float] | None = None,
    slit: tuple[float, float] | None = None,
    smears: Sequence[tuple[int, float, float]] = (),
    artefacts: Sequence[Artefact] = (),
    noise: float = NOISE_SIGMA,
    img_h: int = 200,
    margin: int = 70,
) -> RowCase:
    """A row of the neighbouring-band grid (#180): a band in each lane and,
    with ``neighbour_dy``, another band that far below it (above if negative)
    in every lane, empty or not.

    ``n`` lanes ``pitch`` apart, the first ``margin`` px plus half a band in,
    each drawn 2 px either way along the row and 1 px down it; bands about
    ``w`` x ``h`` px (20% extents, drawn 8% and 10% either way), ``depths``
    deep or drawn from ``depth_range``; ``faint`` makes one lane a tenth as
    deep as the others' mean. The membrane is a pitch wider on the right than
    the bands need. ``smile`` and ``tilt`` are measured from the lanes'
    nominal centres. A neighbouring band has its lane's drawn width and height
    and is ``neighbour_rel`` times as deep as the lane's band (or
    ``neighbour_depths[lane]``). ``doublet`` = ``(dy, frac)``: every band two
    components ``dy`` apart, 0.7 times as high, the lower ``frac`` as deep;
    ``slit`` = ``(frac, sigma)``: a pale line across each band's middle,
    ``frac`` of its depth, ``sigma`` px high; ``smears`` = ``(lane, depth,
    length)``: a streak 4 px wide down from the left half of the lane's band,
    fading over ``length`` px; ``artefacts`` as :func:`adversarial_row`'s (a
    burnt-out band's light centre, specks).

    The row box is the one a user drags snugly over the row: from half a
    pitch before the first lane's nominal centre to half a pitch past the
    last's, 6 px above and below the bands' 20% extents. ``neighbour_cy``
    holds the neighbouring bands' centres."""
    rng = np.random.default_rng(seed)
    centre = margin + w / 2 + pitch * (n - 1) / 2
    half = max(pitch * (n - 1) / 2, 1.0)
    lane_cx = margin + w / 2 + pitch * np.arange(n) + rng.uniform(-2.0, 2.0, n)
    u = (lane_cx - centre) / half
    lane_cy = img_h / 2 + smile * (0.5 - u**2) + tilt * u / 2 + rng.uniform(-1.0, 1.0, n)
    ws = w * rng.uniform(0.92, 1.08, n)
    hs = h * rng.uniform(0.9, 1.1, n)
    dps = rng.uniform(*depth_range, n)
    if depths is not None:
        dps = np.asarray(depths, dtype=float)
    if faint is not None:
        dps[faint] = float(np.mean([dps[i] for i in range(n) if i != faint])) / 10.0
    width_img = int(math.ceil(lane_cx[-1] + ws[-1] / 2 + margin + pitch))
    xs = np.arange(width_img, dtype=float)
    ys = np.arange(img_h, dtype=float)
    X, Y = xs[None, :], ys[:, None]
    # Summed as artefacts and smears, then the lanes' bands, then the
    # neighbouring ones.
    dmap = np.zeros((img_h, width_img))
    for f in artefacts:
        dmap += f(X, Y, lane_cx, lane_cy)
    for lane, depth, length in smears:
        x0 = lane_cx[lane] - ws[lane] / 4
        below = np.where(Y > lane_cy[lane], np.exp(-(Y - lane_cy[lane]) / length), 0.0)
        dmap += depth * (np.exp(-0.5 * ((X - x0) / 4.0) ** 2) * below)
    refs_all: dict[int, Rect] = {}
    reference: dict[int, Rect] = {}
    gone = set(missing)
    for i in range(n):
        refs_all[i] = _ref_rect(lane_cx[i], lane_cy[i], ws[i], hs[i])
        if i in gone:
            continue
        if doublet is not None:
            dy, frac = doublet
            hh = hs[i] * 0.7
            c1, c2 = lane_cy[i] - dy / 2, lane_cy[i] + dy / 2
            band = _band(xs, ys, lane_cx[i], c1, ws[i], hh, dps[i], "super")
            band += _band(xs, ys, lane_cx[i], c2, ws[i], hh, dps[i] * frac, "super")
            r1, r2 = _ref_rect(lane_cx[i], c1, ws[i], hh), _ref_rect(lane_cx[i], c2, ws[i], hh)
            reference[i] = (
                min(r1[0], r2[0]),
                min(r1[1], r2[1]),
                max(r1[2], r2[2]),
                max(r1[3], r2[3]),
            )
        else:
            band = _band(xs, ys, lane_cx[i], lane_cy[i], ws[i], hs[i], dps[i], "super")
            if slit is not None:
                frac, sigma = slit
                band = band * (1.0 - frac * np.exp(-0.5 * ((Y - lane_cy[i]) / sigma) ** 2))
            reference[i] = refs_all[i]
        dmap += band
    neighbour_cy = None
    if neighbour_dy is not None:
        neighbour_cy = lane_cy + neighbour_dy
        for i in range(n):
            depth = dps[i] * neighbour_rel if neighbour_depths is None else neighbour_depths[i]
            dmap += _band(xs, ys, lane_cx[i], neighbour_cy[i], ws[i], hs[i], depth, "super")
    grain = rng.normal(0.0, 1.0, dmap.shape) * noise
    image = np.round(np.clip(MEMBRANE - dmap + grain, 0.0, FULL_SCALE))
    top = min(lane_cy[i] - hs[i] / 2 for i in range(n)) + 0.5
    bottom = max(lane_cy[i] + hs[i] / 2 for i in range(n)) + 0.5
    row = (
        max(0, math.floor(margin + w / 2 - pitch / 2)),
        max(0, math.floor(top) - 6),
        min(width_img, math.ceil(margin + w / 2 + pitch * (n - 1) + pitch / 2)),
        min(img_h, math.ceil(bottom) + 6),
    )
    return RowCase(
        name,
        "",
        image,
        True,
        row,
        n,
        reference,
        tuple(float(c) for c in lane_cx),
        tuple(float(c) for c in lane_cy),
        None if neighbour_cy is None else tuple(float(c) for c in neighbour_cy),
    )


def jpeg(case: RowCase, quality: int = 75, membrane: float = 200.0) -> RowCase:
    """A dark-on-light ``case`` as an 8-bit JPEG export of that quality, read
    back: the membrane at ``membrane`` grey levels and the bands scaled with it
    (about 70 to 120 levels deep), rounded, then compressed. On a smooth
    membrane the compression leaves 8x8 block artefacts a few levels deep, as
    a JPEG blot shows them (#121)."""
    scaled = np.clip(np.round(case.image * membrane / MEMBRANE), 0.0, 255.0).astype(np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(scaled).save(encoded, format="JPEG", quality=quality)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        image = np.asarray(decoded, dtype=np.float64)
    return dataclasses.replace(case, image=image)


def fuzz_row(seed: int) -> RowCase:
    """A row of random geometry for the invariant checks: 2 to 8 lanes, some
    empty, a smile, one band up to 2.5x the usual height, the box edges moved
    (the vertical ones tight) and the declared lane count off by one either way.
    What detection proposes here is not checked; the invariants of a result are.
    The reference keeps the generated lanes."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 9))
    tall = int(rng.integers(n))
    missing = [i for i in range(n) if rng.random() < 0.2]
    case = adversarial_row(
        f"fuzz/{seed}",
        seed,
        n=n,
        missing=missing,
        smile=float(rng.uniform(0.0, 12.0)),
        heights={tall: float(rng.uniform(12.0, 30.0))},
        box_adjust=(
            int(rng.integers(-40, 41)),
            int(rng.integers(-4, 7)),
            int(rng.integers(-40, 41)),
            int(rng.integers(-6, 5)),
        ),
    )
    return dataclasses.replace(case, n_lanes=max(1, n + int(rng.integers(-1, 2))))


# --- #180: a row's lanes scored against its recipe ---

# The score's thresholds, written as literals: never the detector's settings,
# which a degraded detector would move along with what it finds.
CUT_SHARE = 0.15  # a boxed band with this share of its signal outside the row box is cut
CAPTURE_MIN = 0.85  # grid rows: a box holding less of its band than the ideal box is partial
LIMIT_SNR = 6.0  # a miss below this expected SNR is an honest n.d.
SNR_SMOOTH = (3, 5)  # (rows, columns) the expected SNR is smoothed over
_OUTLIER_NOTE = "left out of the shared size"  # the size_outlier note ends so
_LANES_NOTE = re.compile(r"^lanes? (\d+(?:, \d+)*):")  # a note's lanes, counted from 1

RecipeKind = Literal["bench", "adversarial", "grid"]
LaneStatus = Literal["flagged", "row_warned", "silent", "limit", "refused"]

# What each generator leaves out to draw a row's own bands alone.
_OWN_ONLY: dict[str, dict] = {
    "bench": {},
    "adversarial": {"artefacts": (), "neighbour_dy": None},
    "grid": {"artefacts": (), "smears": (), "neighbour_dy": None},
}


@dataclass(frozen=True, eq=False)
class Recipe:
    """A row to score (#180): drawn by ``kind``'s generator (``"bench"``:
    :func:`synthetic_row`, ``"adversarial"``: :func:`adversarial_row`,
    ``"grid"``: :func:`neighbour_grid_row`) from ``params`` at ``seed``, with
    membrane noise ``noise``. ``detect`` holds what detection is called with
    besides the image's own (``saturated_at``, as the operation passes it).
    A grid row is scored by where each box lies against the lane's band and
    its neighbouring band, any other by the box's overlap with the band's
    reference rect."""

    label: str
    kind: RecipeKind
    seed: int
    params: Mapping[str, Any] = field(default_factory=dict)
    noise: float = NOISE_SIGMA
    detect: Mapping[str, Any] = field(default_factory=dict)

    def build(self, **changes: Any) -> RowCase:
        """The row, ``changes`` overriding its generator's arguments."""
        kwargs = {"noise": self.noise, **self.params, **changes}
        if self.kind == "bench":
            return synthetic_row(self.label, "", self.seed, **kwargs)
        if self.kind == "adversarial":
            return adversarial_row(self.label, self.seed, **kwargs)
        if self.kind == "grid":
            return neighbour_grid_row(self.label, self.seed, **kwargs)
        raise ValueError(f"unknown recipe kind {self.kind!r}")


def own_bands(recipe: Recipe) -> np.ndarray:
    """The row's own bands, noise-free, in image units with the bands positive:
    the recipe's row drawn from the same draws with no noise, no artefact and
    no neighbouring row (a hole or a doublet is part of its band), taken from
    the same row with every lane empty. Clipped and rounded as the image is:
    an over-exposed band is as deep as the membrane."""
    only = {**_OWN_ONLY[recipe.kind], "noise": 0.0}
    banded = recipe.build(**only)
    bare = recipe.build(missing=range(banded.n_lanes), **only)
    sign = 1.0 if banded.dark_on_light else -1.0
    return sign * (bare.image - banded.image)


@dataclass(frozen=True)
class LaneTruth:
    """What the recipe says of one reference band. ``cut_share``: the share of
    its noise-free signal in the lane's column (one pitch wide about its true
    centre) that lies outside the row box. ``snr``: its expected detection SNR,
    the peak of that signal smoothed over :data:`SNR_SMOOTH` px within its
    reference rect, over the smoothed noise (``noise / sqrt(15)``); inf on a
    noise-free row."""

    cut_share: float
    snr: float


def _column(case: RowCase, lane: int, width: int) -> tuple[int, int]:
    """The image columns ``[c0, c1)`` of a lane: one pitch (the median step
    between the true centres, whichever way the lanes run) about its true
    centre; every column for a single lane."""
    if len(case.lane_cx) < 2:
        return 0, width
    pitch = float(np.median(np.abs(np.diff(case.lane_cx))))
    cx = case.lane_cx[lane]
    return max(0, math.floor(cx - pitch / 2)), min(width, math.ceil(cx + pitch / 2))


def lane_truth(
    recipe: Recipe, case: RowCase, own: np.ndarray | None = None
) -> dict[int, LaneTruth]:
    """The :class:`LaneTruth` of each reference band of ``case``, the recipe's
    row (its row box may be another): ``own`` is :func:`own_bands`, drawn when
    not given. Only these numbers are kept, never the bands' image."""
    own = own_bands(recipe) if own is None else own
    height, width = own.shape
    content = np.clip(own, 0.0, None)
    smooth = uniform_filter(own, size=SNR_SMOOTH, mode="nearest")
    sigma = recipe.noise / math.sqrt(SNR_SMOOTH[0] * SNR_SMOOTH[1])
    x0, y0, x1, y1 = case.row
    truth = {}
    for lane, (rx0, ry0, rx1, ry1) in sorted(case.reference.items()):
        c0, c1 = _column(case, lane, width)
        total = float(content[:, c0:c1].sum())
        a, b = max(c0, x0), min(c1, x1)
        inside = float(content[max(0, y0) : min(height, y1), a:b].sum()) if b > a else 0.0
        peak = float(smooth[max(0, ry0) : min(height, ry1), max(0, rx0) : min(width, rx1)].max())
        truth[lane] = LaneTruth(
            cut_share=1.0 - inside / total if total > 0 else 0.0,
            snr=peak / sigma if sigma > 0 else math.inf,
        )
    return truth


@dataclass(frozen=True)
class LaneScore:
    """A lane the score does not count right; ``lane`` is its index (0-based,
    shown counted from 1).

    ``kind``: ``miss`` (a band, no box), ``wrong_box`` (the box's IoU with the
    band's reference rect under :data:`~proteia.core.evaluate.IOU_MIN`, or 0),
    ``cut`` (the box is on the band but :data:`CUT_SHARE` or more of the band
    lies outside the row box), ``fp`` (a box, no band); on a grid row
    ``missed``, ``neighbour`` (the box holds the neighbouring band's centre
    and not the band's), ``both`` (both centres), ``partial`` (it holds the
    band's centre but less than :data:`CAPTURE_MIN` of what the box of its
    size centred on the band holds, or neither centre), ``false_box`` (a box
    in an empty lane). ``value`` is the number its kind was decided by (the
    IoU, the cut share, the capture, a miss's expected SNR), None where there
    is none.

    ``status``: ``flagged`` (the page shows the lane in doubt: ``shown_by``
    says how), ``row_warned`` (only a warning about other lanes or the row),
    ``silent``, ``limit`` (a silent miss of expected SNR under
    :data:`LIMIT_SNR`: an honest n.d., not wrong), ``refused`` (the row was
    refused: nothing placed, the user told)."""

    lane: int
    kind: str
    status: LaneStatus
    value: float | None = None
    shown_by: str | None = None

    def __str__(self) -> str:
        value = "" if self.value is None else f" {self.value:.3f}"
        shown = f" ({self.shown_by})" if self.shown_by else ""
        return f"lane {self.lane + 1} {self.kind}{value} {self.status}{shown}"


@dataclass(frozen=True)
class RowScore:
    """One row's score: ``refused`` names the refusing flag (or
    ``no_band_found``), None for a placed row; ``hits`` of the ``n_ref``
    reference bands right; ``lanes`` the lanes not right, in lane order."""

    label: str
    refused: str | None
    n_ref: int
    hits: int
    lanes: tuple[LaneScore, ...]

    @property
    def wrong(self) -> int:
        """Wrong lanes: all of :attr:`lanes` but the honest n.d.s."""
        return sum(1 for lane in self.lanes if lane.status != "limit")

    @property
    def silent(self) -> int:
        """Wrong lanes the page shows nothing about."""
        return sum(1 for lane in self.lanes if lane.status == "silent")

    def __str__(self) -> str:
        refused = f"refused ({self.refused}), " if self.refused else ""
        lanes = "".join(f"; {lane}" for lane in self.lanes)
        return f"{self.label}: {refused}{self.hits}/{self.n_ref} right{lanes}"


def _holds(rect: Rect, x: float, y: float) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _capture(own: np.ndarray, case: RowCase, lane: int, rect: Rect) -> float:
    """The band's signal in ``rect`` over the signal in a box of its size
    centred on the band's true centre, both within the lane's column."""
    height, width = own.shape
    c0, c1 = _column(case, lane, width)

    def mass(x0: int, y0: int, x1: int, y1: int) -> float:
        a, b, top, bottom = max(x0, c0), min(x1, c1), max(0, y0), min(height, y1)
        return (
            float(np.clip(own[top:bottom, a:b], 0.0, None).sum()) if b > a and bottom > top else 0.0
        )

    w, h = rect[2] - rect[0], rect[3] - rect[1]
    ix = math.floor(case.lane_cx[lane] + 0.5 - w / 2)
    iy = math.floor(case.lane_cy[lane] + 0.5 - h / 2)
    ideal = mass(ix, iy, ix + w, iy + h)
    return mass(*rect) / ideal if ideal > 0 else 1.0


def _overlap_kinds(
    case: RowCase, found: rowdetect.RowDetection, truth: Mapping[int, LaneTruth]
) -> dict[int, tuple[str, float | None]]:
    """The kind of each lane not right, by its box's overlap with the band."""
    hits = evaluate.hit_rate(found.slots, case.reference)
    kinds: dict[int, tuple[str, float | None]] = {}
    for lane, rect in enumerate(found.slots):
        if lane not in case.reference:
            if rect is not None:
                kinds[lane] = ("fp", None)
        elif rect is None:
            kinds[lane] = ("miss", truth[lane].snr)
        elif not (hits.iou[lane] > 0.0 and hits.iou[lane] >= evaluate.IOU_MIN):
            kinds[lane] = ("wrong_box", hits.iou[lane])
        elif truth[lane].cut_share >= CUT_SHARE:
            kinds[lane] = ("cut", truth[lane].cut_share)
    return kinds


def _grid_kinds(
    case: RowCase,
    found: rowdetect.RowDetection,
    truth: Mapping[int, LaneTruth],
    own: np.ndarray,
) -> dict[int, tuple[str, float | None]]:
    """The kind of each lane not right, by which centres its box holds (each
    as a pixel's centre, x + 0.5) and how much of the band."""
    kinds: dict[int, tuple[str, float | None]] = {}
    for lane, rect in enumerate(found.slots):
        if lane not in case.reference:
            if rect is not None:
                kinds[lane] = ("false_box", None)
            continue
        if rect is None:
            kinds[lane] = ("missed", truth[lane].snr)
            continue
        x = case.lane_cx[lane] + 0.5
        on_band = _holds(rect, x, case.lane_cy[lane] + 0.5)
        on_neighbour = case.neighbour_cy is not None and _holds(
            rect, x, case.neighbour_cy[lane] + 0.5
        )
        if on_neighbour:
            kinds[lane] = ("both" if on_band else "neighbour", None)
            continue
        capture = _capture(own, case, lane, rect)
        if not on_band or capture < CAPTURE_MIN:
            kinds[lane] = ("partial", capture)
    return kinds


def _named_lanes(notes: Iterable[str], marker: str) -> set[int]:
    """The lane indices the notes holding ``marker`` name (notes count lanes
    from 1)."""
    named: set[int] = set()
    for note in notes:
        match = _LANES_NOTE.match(note)
        if marker in note and match:
            named.update(int(number) - 1 for number in match.group(1).split(", "))
    return named


def _shown_by(
    lane: rowdetect.LaneDetection,
    index: int,
    row_wide: str | None,
    boxed: int,
    outliers: set[int],
    rows: tuple[int, int],
) -> str | None:
    """What the page shows of a lane in doubt, None for nothing: the whole
    row's doubt; an empty lane not measured or not recorded; a box flagged,
    named by the size note, or holding more bands in the row box's rows than
    one (the results' band count)."""
    if row_wide is not None:
        return row_wide
    if lane.rect is None:
        if lane.reason != "no_band":
            return f"not measured: {lane.reason}"
        if lane.window is None:
            return "not measured: outside the row box"
        if boxed < 2:
            return "not recorded: fewer than two bands"
        return None
    if lane.cut:
        return "cut_by_row_box"
    if lane.components > 1:
        return "multiple_components"
    if lane.hollow:
        return "hollow_band"
    if index in outliers:
        return "size_outlier"
    if rowdetect.bands_in(lane, *rows) > 1:
        return "band count"
    return None


def score_row(
    recipe: Recipe,
    case: RowCase,
    found: rowdetect.RowDetection,
    truth: Mapping[int, LaneTruth] | None = None,
) -> RowScore:
    """Score ``found``, detection on ``case`` (the recipe's row), lane by lane
    against the recipe's truth (:class:`LaneScore` says how), never against
    what the detector reports of itself.

    A refused row (a refusing flag, or no band at all: the operation refuses
    both) counts every reference band wrong and none silent; its lanes are not
    read. ``truth`` (:func:`lane_truth`) is drawn when not given, and a grid
    row draws the row's own bands either way. A row without reference bands
    counts only its boxes. Raises ValueError when ``case`` declares another
    number of lanes than it draws, or ``found`` reads another: nothing to
    score."""
    n = case.n_lanes
    if len(case.lane_cx) != n or len(found.lanes) != n:
        raise ValueError(
            f"{recipe.label}: {n} lanes declared, {len(case.lane_cx)} drawn,"
            f" {len(found.lanes)} read"
        )
    if found.refused or found.size is None:
        cause = found.flags[0] if found.refused else "no_band_found"
        refused = tuple(
            LaneScore(lane, "miss", "refused", shown_by=cause) for lane in sorted(case.reference)
        )
        return RowScore(recipe.label, cause, len(case.reference), 0, refused)
    grid = recipe.kind == "grid"
    own = own_bands(recipe) if grid or truth is None else None
    truth = lane_truth(recipe, case, own) if truth is None else truth
    kinds = _grid_kinds(case, found, truth, own) if grid else _overlap_kinds(case, found, truth)
    if "doubtful_lanes" in found.flags:
        row_wide: str | None = "doubtful_lanes"
    elif any(lane.reason == "unassigned" for lane in found.lanes):
        row_wide = "unassigned piece"
    else:
        row_wide = None
    boxed = sum(rect is not None for rect in found.slots)
    outliers = _named_lanes(found.notes, _OUTLIER_NOTE)
    rows = (max(0, case.row[1]), min(case.image.shape[0], case.row[3]))
    lanes = []
    for index, (kind, value) in sorted(kinds.items()):
        shown = _shown_by(found.lanes[index], index, row_wide, boxed, outliers, rows)
        if shown is not None:
            status: LaneStatus = "flagged"
        elif found.flags:
            status = "row_warned"
        elif kind in ("miss", "missed") and value is not None and value < LIMIT_SNR:
            status = "limit"
        else:
            status = "silent"
        lanes.append(LaneScore(index, kind, status, value, shown))
    hits = sum(1 for lane in case.reference if lane not in kinds)
    return RowScore(recipe.label, None, len(case.reference), hits, tuple(lanes))
