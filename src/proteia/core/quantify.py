# SPDX-License-Identifier: Apache-2.0
"""Pixel-level quantification: integrate intensity within a box above a background.

``net_signal`` is the densitometry quantity: integrated intensity *above a
background level*, with the signal direction handled. On a dark-band-on-light
image (chemiluminescence/colorimetric) a darker band must read as *more* signal,
so the contribution of each pixel is its darkness relative to the background
level. (A raw pixel sum is not comparable on its own: it conflates signal, box
area, and how much background the box happens to include.)

Two background methods give the level ``net_signal`` subtracts;
:func:`quantify_nets` runs either over the boxes of one image.

* ``global_median`` (:func:`estimate_background`): one level per image, the
  median of the whole analysis array, with each pixel's contribution floored at
  0 (``clamp="pixel"``). Import stores it as ``ImageRef.background``; box growing
  seeds from it, and the ring method falls back to it.
* ``ring_median`` version 1 (:func:`band_backgrounds`, #83): one level per band,
  from the membrane around its own box, with the box total floored once at 0
  (:data:`RING_CLAMP`), so noise around the level cancels instead of adding up
  as signal. A gradient or a hump across the membrane then no longer biases a
  band by where it sits.

The ring_median level of a box ``w`` wide and ``h`` tall (area ``S``), measured
on the signal-up values ``u`` (``-p`` for a dark-on-light image, ``+p``
otherwise):

1. Region sampled. The box's *zone* is the box widened by a gap of
   ``ceil(0.2 w)`` px left and right (across the lane) and ``ceil(0.5 h)`` px
   above and below (along the lane): :data:`RING_GAP_ACROSS`,
   :data:`RING_GAP_ALONG`. Ring pixels lie at Chebyshev distance 1 to
   ``max(w, h)`` outside the zone, inside the image, and outside the zone of
   every box on the image (each with its own gap, whatever its protein). A
   pixel is used only when its point reflection through the box centre is too,
   so a linear gradient cancels whatever the image edge or the other boxes cut
   away. The ring widens from :data:`RING_MIN_WIDTH` px until it holds
   :data:`RING_PIXELS` box areas.
2. Ring level. An iterated two-sided clip at :data:`RING_CLIP_K` robust sigmas
   (1.4826 median absolute deviations, floored at half a count for detector
   data) around the median drops bubbles, specks and band tails; the level is
   the grouped median of the rest (for whole-number values the median
   interpolated within its unit class, otherwise the plain median).
3. In-lane haze. Above and below the box, the in-lane level is the median of
   the means of the ring rows complete over the box's own columns (a side needs
   :data:`RING_HAZE_ROWS` such rows and ``max(10, S/8)`` pixels). When both
   sides read more signal than the ring level, the level is raised to
   ``min(top, bottom)`` less :data:`RING_HAZE_Z` standard errors: haze running
   down the lane is background, as under a lane-profile baseline, while a
   feature on one side only (a bubble, an unboxed band, a neighbour's tail)
   cannot raise it, nor can a linear gradient (one side always reads less).
4. Fallbacks, reported as the band's mode. Fewer than :data:`RING_SYMMETRIC_MIN`
   box areas of paired pixels (``asymmetric``): the ring drops the pairing and
   the level is a robust plane at the box centre, with a slope only along an
   axis holding :data:`RING_PLANE_SIDE` of the ring on each side of the centre,
   clamped to the range of the kept pixels (never an extrapolation). Fewer than
   ``max(10, S/4)`` ring pixels (``image``, :data:`RING_IMAGE_MIN`): the image
   median.
5. QC. A symmetric band's ``spread`` is how far apart its candidate levels lie
   beyond three standard errors. The candidates are the ring level, the lifted
   level and the midpoints of opposite sides: the mean of the left and right
   levels (each the clipped grouped median of the ring on that side of the box)
   and the mean of the top and bottom in-lane levels. A difference between
   opposite sides that is symmetric about the box, such as any linear
   gradient, leaves their midpoint at the ring level and adds nothing. A
   spread that moves the net by more than :data:`BACKGROUND_UNEVEN_LIMIT`
   marks the background as uneven.

The ``RING_*`` constants and the statistics constants beside them are versioned
together: changing any of them makes a new method version, and stored nets must
then be requantified. :func:`background_settings` reports them for the
reproducibility record. The method is pure and deterministic (medians,
partitions, elementwise arithmetic, and a least-squares fit in the asymmetric
fallback only), and a band's result depends on the other boxes only through the
zones they exclude, not on their order.

Multi-channel images are reduced to a single grayscale channel by
:func:`to_grayscale`, the one place that rule lives.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from pydantic import JsonValue

from proteia.core.grow import mad_sigma
from proteia.core.model import BackgroundMode, Box, BoxSize, Rect  # the mode: a band field

BackgroundMethod = Literal["global_median", "ring_median"]
# "pixel": each pixel's contribution floored at 0; "total": the box total floored at 0.
NetClamp = Literal["pixel", "total"]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Reduce an image to a 2D grayscale array.

    Color (3 or 4 channels last) becomes the unweighted mean of red, green and
    blue, which matches ImageJ's default conversion; alpha is ignored. Gray plus
    alpha (2 channels) keeps the gray channel.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] == 2:
        return image[..., 0]
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(axis=-1)
    raise ValueError(f"unsupported image layout {image.shape}")


def _box_pixels(image: np.ndarray, box: Box, size: BoxSize) -> np.ndarray:
    """The grayscale pixels inside the box.

    Raises ValueError if the box extends beyond the image bounds.
    """
    gray = to_grayscale(image)
    height, width = gray.shape
    x0, y0, x1, y1 = box.rect(size)
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError("box extends beyond the image bounds")
    return gray[y0:y1, x0:x1]


def estimate_background(image: np.ndarray) -> float:
    """A robust background (membrane) level: the median of the grayscale image.

    On a typical blot the membrane dominates the frame, so its median is a good
    zeroth-order baseline to subtract: the ``global_median`` method, and the
    level :func:`band_backgrounds` falls back to where a box has too little
    membrane around it.
    """
    return float(np.median(to_grayscale(image)))


def net_signal(
    image: np.ndarray,
    box: Box,
    size: BoxSize,
    background: float,
    *,
    dark_on_light: bool = True,
    clamp: NetClamp = "pixel",
) -> float:
    """Integrated signal inside the box, *above* the background level.

    For ``dark_on_light`` (the default — dark bands on a light membrane), each
    pixel contributes how much *darker* than ``background`` it is; for a
    light-on-dark image (e.g. fluorescence) the direction is flipped. With
    ``clamp="pixel"`` (the global-median rule) a pixel on the far side of the
    background contributes nothing; with ``clamp="total"`` (:data:`RING_CLAMP`)
    every pixel contributes its signed difference and only the box total is
    floored at 0, so membrane noise inside the box cancels. That sum is taken in
    double precision whatever the image's type: its terms cancel, which would
    magnify a level rounded to the type. The result is non-negative and grows
    with band strength. Equal-area boxes keep it comparable.

    Raises ValueError if the box extends beyond the image bounds, or for an
    unknown ``clamp``.
    """
    pixels = _box_pixels(image, box, size)
    if clamp == "total":
        pixels = pixels.astype(np.float64, copy=False)
    elif clamp != "pixel":
        raise ValueError(f"unknown clamp {clamp!r}")
    difference = background - pixels if dark_on_light else pixels - background
    if clamp == "pixel":
        return float(np.maximum(difference, 0.0).sum())
    return max(float(difference.sum()), 0.0)


# --- Local background: ring_median, version 1 (#83) ---
# Changing any setting below except the QC limit makes a new method version.

RING_METHOD: Final = "ring_median"
RING_VERSION: Final = 1
RING_GAP_ACROSS: Final = 0.2  # gap left and right of the box, fraction of its width
RING_GAP_ALONG: Final = 0.5  # gap above and below the box, fraction of its height
RING_MIN_WIDTH: Final = 3  # narrowest ring (px)
RING_PIXELS: Final = 2.0  # the ring widens until it holds this many box areas
RING_SYMMETRIC_MIN: Final = 1.0  # paired ring used when it holds this many box areas
RING_IMAGE_MIN: Final = 0.25  # below max(10, this many box areas) ring pixels: image median
RING_CLIP_K: Final = 3.0  # two-sided clip at this many robust sigmas
RING_MAX_ITER: Final = 20  # clip and plane-fit iterations at most
RING_SIDE_MIN: Final = 0.125  # a side of the ring needs max(10, this many box areas) pixels
RING_HAZE_ROWS: Final = 2  # complete in-lane rows a side needs for the haze lift
RING_HAZE_Z: Final = 2.0  # the haze lift stops this many standard errors short
RING_PLANE_SIDE: Final = 0.1  # a plane slope needs this share of the ring each side of centre
RING_CLAMP: Final[NetClamp] = "total"  # decision 3 on #83: the box total floored at 0
# QC: a background spread that moves the net by more than this share is uneven.
BACKGROUND_UNEVEN_LIMIT: Final = 0.05

_MIN_SAMPLE: Final = 10  # fewest pixels any ring statistic reads
_SPREAD_Z: Final = 3.0  # the QC spread counts beyond this many standard errors
_MEDIAN_SE: Final = 1.2533  # standard error of a median: 1.2533 sigma / sqrt(n), normal data
_SIGMA_FLOOR_INTEGRAL: Final = 0.5  # half a count: whole-number data cannot resolve less
_SIGMA_FLOOR_FLOAT: Final = 1e-12


@dataclass(frozen=True)
class BandBackground:
    """One band's background, as :func:`band_backgrounds` measures it.

    ``level`` is in pixel units: what :func:`net_signal` subtracts. ``mode`` is
    how it was measured (:data:`BackgroundMode`): ``symmetric`` (the paired
    ring), ``asymmetric`` (too few pairs: a robust plane over the unpaired ring),
    ``image`` (too little membrane: the image median) or ``global_median``.
    ``spread`` is QC in level units: how far apart the ring level, the lifted
    level and the midpoints of opposite sides of the box (left and right, top
    and bottom) lie beyond noise; 0 unless ``mode`` is ``symmetric``.
    """

    level: float
    mode: BackgroundMode
    spread: float


def band_backgrounds(
    array: np.ndarray,
    rects: Sequence[Rect],
    sizes: Sequence[tuple[int, int]],
    *,
    dark_on_light: bool,
    integral: bool,
    fallback: float,
) -> list[BandBackground]:
    """The ring_median background of every box on one image, in input order.

    ``rects`` holds every box on the image, of every protein, as ``(x0, y0, x1,
    y1)`` half-open on the high edge; ``sizes`` holds each one's ``(width,
    height)``, its protein's box size. Each ring excludes the zones of all of
    them, so they are measured together; a result does not depend on the order
    of the other boxes. ``integral`` says the array holds whole detector counts
    (the image has a bit depth), which floors the robust sigma at half a count.
    ``fallback`` is the image-mode level: the image median
    (:func:`estimate_background`, stored as ``ImageRef.background``).

    Raises ValueError if the lists differ in length, a rect is not its size, or
    a box extends beyond the image bounds.
    """
    gray = to_grayscale(array)
    boxes = _checked_rects(gray.shape, rects, sizes)
    excluded = _exclusion_mask(gray.shape, boxes)
    sign = -1.0 if dark_on_light else 1.0
    floor = _SIGMA_FLOOR_INTEGRAL if integral else _SIGMA_FLOOR_FLOAT
    return [_band_background(gray, rect, excluded, sign, floor, float(fallback)) for rect in boxes]


def quantify_nets(
    array: np.ndarray,
    rects: Sequence[Rect],
    sizes: Sequence[tuple[int, int]],
    *,
    method: BackgroundMethod,
    dark_on_light: bool,
    integral: bool,
) -> list[float]:
    """The net of every box on one image under either background method, in input
    order: the one entry point that runs a method end to end, for comparisons
    with reference measurements and for tests.

    ``global_median`` is :func:`estimate_background` and :func:`net_signal` with
    ``clamp="pixel"``; ``ring_median`` is :func:`band_backgrounds` (falling back to
    the image median) and :func:`net_signal` with :data:`RING_CLAMP`. Arguments are
    as for :func:`band_backgrounds`; ``integral`` only matters to ``ring_median``.
    Raises ValueError as it does, or for an unknown ``method``.
    """
    gray = to_grayscale(array)
    boxes = _checked_rects(gray.shape, rects, sizes)
    image_median = estimate_background(gray)
    placed = [(Box(x=x0, y=y0), BoxSize(width=x1 - x0, height=y1 - y0)) for x0, y0, x1, y1 in boxes]
    if method == "global_median":
        return [
            net_signal(gray, box, size, image_median, dark_on_light=dark_on_light)
            for box, size in placed
        ]
    if method == "ring_median":
        found = band_backgrounds(
            gray,
            boxes,
            sizes,
            dark_on_light=dark_on_light,
            integral=integral,
            fallback=image_median,
        )
        return [
            net_signal(gray, box, size, bg.level, dark_on_light=dark_on_light, clamp=RING_CLAMP)
            for (box, size), bg in zip(placed, found, strict=True)
        ]
    raise ValueError(f"unknown background method {method!r}")


def background_settings() -> dict[str, JsonValue]:
    """The ring_median method and its settings, JSON-plain: what the
    reproducibility record reports as its background."""
    clamp = "box total floored at 0" if RING_CLAMP == "total" else "per pixel at 0"
    return {
        "method": RING_METHOD,
        "version": RING_VERSION,
        "gap": {"across_lane": RING_GAP_ACROSS, "along_lane": RING_GAP_ALONG, "unit": "box size"},
        "ring": {
            "initial_width_px": RING_MIN_WIDTH,
            "min_pixels_per_box_area": RING_PIXELS,
            "max_width": "max(box width, box height)",
            "exclude": "every box on the image, dilated by its gap",
            "pairing": "point-symmetric about the box centre",
        },
        "statistic": {
            "clip": f"two-sided, {RING_CLIP_K} x 1.4826 MAD, iterated",
            "level": "grouped median",
        },
        "lane_haze": {
            "in_lane_level": "median of complete row means over the box columns, per side",
            "rule": "raised to min(T, B) - z se when both exceed the ring level by more",
            "z": RING_HAZE_Z,
        },
        "fallbacks": {
            "asymmetric": f"symmetric ring < {RING_SYMMETRIC_MIN} box area: robust plane at "
            "box centre over the full ring (slopes only where pixels lie on both sides; "
            "clamped to kept range)",
            "image": f"< max({_MIN_SAMPLE}, {RING_IMAGE_MIN} box area) ring pixels: image median",
        },
        "clamp": clamp,
        "qc": {"uneven_limit": BACKGROUND_UNEVEN_LIMIT},
    }


def _checked_rects(
    shape: tuple[int, ...], rects: Sequence[Rect], sizes: Sequence[tuple[int, int]]
) -> list[Rect]:
    """The rects as plain int tuples, each checked against its size and the image."""
    if len(rects) != len(sizes):
        raise ValueError(f"{len(rects)} boxes but {len(sizes)} sizes")
    height, width = shape
    boxes = []
    for rect, (w, h) in zip(rects, sizes, strict=True):
        x0, y0, x1, y1 = (int(v) for v in rect)
        if w <= 0 or h <= 0 or (x1 - x0, y1 - y0) != (w, h):
            raise ValueError(f"box {rect} is not {w} x {h}")
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise ValueError("box extends beyond the image bounds")
        boxes.append((x0, y0, x1, y1))
    return boxes


def _zone(rect: Rect) -> Rect:
    """The box widened by its gap: no ring may use these pixels."""
    x0, y0, x1, y1 = rect
    gap_x = math.ceil(RING_GAP_ACROSS * (x1 - x0))
    gap_y = math.ceil(RING_GAP_ALONG * (y1 - y0))
    return x0 - gap_x, y0 - gap_y, x1 + gap_x, y1 + gap_y


def _exclusion_mask(shape: tuple[int, ...], rects: Sequence[Rect]) -> np.ndarray:
    """Pixels no ring may use: the zone of every box, clipped to the image."""
    height, width = shape
    excluded = np.zeros(shape, dtype=bool)
    for rect in rects:
        zx0, zy0, zx1, zy1 = _zone(rect)
        excluded[max(0, zy0) : min(height, zy1), max(0, zx0) : min(width, zx1)] = True
    return excluded


def _ring_candidates(
    excluded: np.ndarray, rect: Rect, reach: int, *, paired: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pixels within ``reach`` of the zone of ``rect`` that may join its ring:
    their ``(y, x)`` coordinates in row-major order and their Chebyshev distance
    ``d >= 1`` outside the zone. They lie in the image and outside ``excluded``,
    which holds the zone of ``rect`` (:func:`_exclusion_mask`); with ``paired``,
    so do their point reflections through the box centre.

    The window searched is clipped to the image, and with ``paired`` to its
    reflection too, so a box as large as the image costs no more memory than the
    image. It always holds the box, and it is centred on the box, so flipping it
    on both axes reflects a pixel.
    """
    height, width = excluded.shape
    x0, y0, x1, y1 = rect
    zx0, zy0, zx1, zy1 = _zone(rect)
    wx0, wy0 = max(zx0 - reach, 0), max(zy0 - reach, 0)
    wx1, wy1 = min(zx1 + reach, width), min(zy1 + reach, height)
    if paired:  # x reflects to x0 + x1 - 1 - x, which must lie in the image too
        wx0, wy0 = max(wx0, x0 + x1 - width), max(wy0, y0 + y1 - height)
        wx1, wy1 = min(wx1, x0 + x1), min(wy1, y0 + y1)
    usable = ~excluded[wy0:wy1, wx0:wx1]
    if paired:
        usable = usable & usable[::-1, ::-1]
    iy, ix = np.nonzero(usable)
    del usable
    xs, ys = np.arange(wx0, wx1), np.arange(wy0, wy1)
    dx = np.maximum(np.maximum(zx0 - xs, xs - (zx1 - 1)), 0)
    dy = np.maximum(np.maximum(zy0 - ys, ys - (zy1 - 1)), 0)
    distance = dx[ix]
    np.maximum(distance, dy[iy], out=distance)
    iy += wy0
    ix += wx0
    return iy, ix, distance


def _ring_width(distance: np.ndarray, need: int) -> int | None:
    """The narrowest ring width, at least :data:`RING_MIN_WIDTH`, at which the
    pixels at ``distance`` number ``need`` or more; None when all of them fall
    short."""
    counts = np.cumsum(np.bincount(distance))
    if counts.size == 0 or counts[-1] < need:
        return None
    return max(RING_MIN_WIDTH, int(np.searchsorted(counts, need)))


def _initial_reach(zone: Rect, need: int) -> int:
    """A first window reach: twice the width at which a full ring around ``zone``
    would hold ``need`` pixels. Only speed depends on it: a window that falls
    short is widened to the full reach."""
    zx0, zy0, zx1, zy1 = zone
    half_perimeter = (zx1 - zx0) + (zy1 - zy0)
    # a ring t wide around the zone holds 2 t (zone width + height) + 4 t^2 pixels
    t = (math.sqrt(half_perimeter * half_perimeter + 4 * need) - half_perimeter) / 4
    return 2 * math.ceil(t)


def _ring(
    excluded: np.ndarray, rect: Rect
) -> tuple[Literal["symmetric", "asymmetric"], np.ndarray, np.ndarray] | None:
    """The ring of one box: its mode and its pixels' ``(y, x)`` coordinates in
    row-major order, or None when it holds too few pixels (image mode)."""
    x0, y0, x1, y1 = rect
    area = (x1 - x0) * (y1 - y0)
    full_reach = max(x1 - x0, y1 - y0)
    need = math.ceil(RING_PIXELS * area)
    reach = min(full_reach, max(RING_MIN_WIDTH, _initial_reach(_zone(rect), need)))
    mode: Literal["symmetric", "asymmetric"] = "symmetric"
    while True:
        ys, xs, distance = _ring_candidates(excluded, rect, reach, paired=True)
        width = _ring_width(distance, need)
        if width is not None:  # width <= reach, or the window is already full
            break
        if reach == full_reach:  # the whole ring falls short of need
            if distance.size < RING_SYMMETRIC_MIN * area:
                mode = "asymmetric"
                ys, xs, distance = _ring_candidates(excluded, rect, full_reach, paired=False)
                width = _ring_width(distance, need)
            if width is None:
                width = full_reach
            break
        reach = full_reach
    ring = distance <= width
    if np.count_nonzero(ring) < max(_MIN_SAMPLE, RING_IMAGE_MIN * area):
        return None
    return mode, ys[ring], xs[ring]


def _band_background(
    gray: np.ndarray, rect: Rect, excluded: np.ndarray, sign: float, floor: float, fallback: float
) -> BandBackground:
    """One box's :class:`BandBackground`; ``sign`` turns pixels into signal-up
    values (``-1`` for a dark-on-light image) and back."""
    ring = _ring(excluded, rect)
    if ring is None:
        return BandBackground(level=fallback, mode="image", spread=0.0)
    mode, ys, xs = ring
    x0, y0, x1, y1 = rect
    w = x1 - x0
    area = w * (y1 - y0)
    u = sign * gray[ys, xs].astype(np.float64)
    if mode == "asymmetric":
        ring_level, sigma = _plane_level(u, xs, ys, rect, floor)
    else:
        keep, sigma = _clip(u, floor)
        ring_level = _grouped_median(u[keep])

    side_min = max(_MIN_SAMPLE, RING_SIDE_MIN * area)
    in_lane = (xs >= x0) & (xs < x1)
    above, below = in_lane & (ys < y0), in_lane & (ys >= y1)
    top = _lane_side(u[above], ys[above], w, sigma, side_min)
    bottom = _lane_side(u[below], ys[below], w, sigma, side_min)
    lift = 0.0
    if top is not None and bottom is not None:
        se = max(top[2], bottom[2])
        lift = max(0.0, min(top[0], bottom[0]) - ring_level - RING_HAZE_Z * se)
    level = ring_level + lift

    spread = 0.0
    if mode == "symmetric":
        levels, counts = [ring_level, level], []
        left, right = u[xs < x0], u[xs >= x1]
        if left.size >= side_min and right.size >= side_min:
            sides = [_grouped_median(side[_clip(side, floor)[0]]) for side in (left, right)]
            levels.append((sides[0] + sides[1]) / 2)
            counts += [left.size, right.size]
        if top is not None and bottom is not None:
            levels.append((top[0] + bottom[0]) / 2)
            counts += [top[1], bottom[1]]
        if counts:
            se = _MEDIAN_SE * sigma * math.sqrt(2.0 / min(counts))
            spread = max(0.0, max(levels) - min(levels) - _SPREAD_Z * se)
    return BandBackground(level=sign * level, mode=mode, spread=spread)


def _clip(values: np.ndarray, floor: float) -> tuple[np.ndarray, float]:
    """Iterated two-sided clip at :data:`RING_CLIP_K` robust sigmas around the
    median of the kept values (sigma = 1.4826 MAD, at least ``floor``): the kept
    mask and the last sigma."""
    keep = np.ones(values.size, dtype=bool)
    sigma = floor
    for _ in range(RING_MAX_ITER):
        kept = values[keep]
        center = float(np.median(kept))
        sigma = max(mad_sigma(kept, center), floor)
        new = np.abs(values - center) <= RING_CLIP_K * sigma
        if np.array_equal(new, keep) or not new.any():
            break
        keep = new
    return keep, sigma


def _grouped_median(values: np.ndarray) -> float:
    """The median of whole-number data interpolated within its unit class
    ``(c - 0.5, c + 0.5]``, which resolves a level finer than one count; the plain
    median of any other data. ``values`` must not be empty."""
    if not np.all(values == np.round(values)):
        return float(np.median(values))
    n = values.size
    k = math.ceil(n / 2) - 1
    c = float(np.partition(values, k)[k])
    below = int(np.count_nonzero(values < c))
    ties = int(np.count_nonzero(values == c))
    return c - 0.5 + (n / 2 - below) / ties


def _plane_level(
    u: np.ndarray, xs: np.ndarray, ys: np.ndarray, rect: Rect, floor: float
) -> tuple[float, float]:
    """The asymmetric-ring level: a robust plane evaluated at the box centre, and
    the robust sigma of its residuals.

    A slope is fitted only along an axis with :data:`RING_PLANE_SIDE` of the
    pixels (at least 10) on each side of the centre, so the plane interpolates;
    with neither, this is the clipped grouped median. The fit starts from the
    constant clip, so a minority of band pixels (an unboxed band on one side) is
    gone before any slope is fitted, and the level is clamped to the range of the
    kept pixels.
    """
    x0, y0, x1, y1 = rect
    dx = xs.astype(np.float64) - (x0 + x1 - 1) / 2.0
    dy = ys.astype(np.float64) - (y0 + y1 - 1) / 2.0
    side = max(_MIN_SAMPLE, int(RING_PLANE_SIDE * u.size))
    columns = [np.ones(u.size)]
    for offset in (dx, dy):
        if np.count_nonzero(offset < 0) >= side and np.count_nonzero(offset > 0) >= side:
            columns.append(offset)
    keep, sigma = _clip(u, floor)
    if len(columns) == 1:
        return _grouped_median(u[keep]), sigma
    design = np.column_stack(columns)
    params = design.shape[1]
    coef = np.zeros(params)
    for _ in range(RING_MAX_ITER):
        if np.count_nonzero(keep) <= params:
            break
        coef = np.linalg.lstsq(design[keep], u[keep], rcond=None)[0]
        residual = u - design @ coef
        kept = residual[keep]
        center = float(np.median(kept))
        sigma = max(mad_sigma(kept, center), floor)
        new = np.abs(residual - center) <= RING_CLIP_K * sigma
        if np.array_equal(new, keep) or np.count_nonzero(new) <= params:
            break
        keep = new
    level = float(coef[0] + np.median(u[keep] - design[keep] @ coef))
    kept = u[keep]
    return min(max(level, float(kept.min())), float(kept.max())), sigma


def _lane_side(
    u: np.ndarray, ys: np.ndarray, width: int, sigma: float, side_min: float
) -> tuple[float, int, float] | None:
    """The in-lane level of the ring on one side of the box: the median of the
    means of its rows complete over the box's ``width`` columns (a lane-profile
    sample per row), the pixels in those rows and its standard error; None with
    fewer than :data:`RING_HAZE_ROWS` such rows or ``side_min`` pixels."""
    if u.size == 0:
        return None
    rows = ys - ys.min()
    complete = np.bincount(rows) == width
    n_rows = int(np.count_nonzero(complete))
    if n_rows < RING_HAZE_ROWS or n_rows * width < side_min:
        return None
    means = np.bincount(rows, weights=u)[complete] / width
    se = max(
        _MEDIAN_SE * sigma / math.sqrt(width * n_rows),
        _MEDIAN_SE * mad_sigma(means) / math.sqrt(n_rows),
    )
    return float(np.median(means)), n_rows * width, se


# A band is flagged as clipped when at least this many pixels inside its box sit
# at the detector limit (maintainer decision on #44: any pixel). Flagged bands stay
# in the statistics; the user decides whether to exclude the lane.
CLIPPED_PIXELS_THRESHOLD = 1


def detector_limit(bit_depth: int, *, dark_on_light: bool) -> int:
    """The pixel value a saturated detector leaves in a ``bit_depth`` image.

    Signal makes a light-on-dark image brighter, so saturation pins it at the
    maximum, ``2**bit_depth - 1``; a dark-on-light image shows signal as darkness,
    so saturation pins it at the minimum, 0.
    """
    return 0 if dark_on_light else 2**bit_depth - 1


def clipped_pixels(
    image: np.ndarray, box: Box, size: BoxSize, *, bit_depth: int, dark_on_light: bool
) -> int:
    """How many pixels inside the box sit at the detector limit.

    Densitometry holds only while the detector responds linearly: a band with
    pixels at the limit is flat-topped, and its net under-estimates it. Raises
    ValueError if the box extends beyond the image bounds.
    """
    limit = detector_limit(bit_depth, dark_on_light=dark_on_light)
    return int(np.count_nonzero(_box_pixels(image, box, size) == limit))


def is_clipped(
    image: np.ndarray, box: Box, size: BoxSize, *, bit_depth: int | None, dark_on_light: bool
) -> bool | None:
    """Whether a band is over-exposed: :data:`CLIPPED_PIXELS_THRESHOLD` or more
    pixels of its box at the detector limit. None (not checked, never "passed")
    when ``bit_depth`` is None: the image has no limit the check can trust
    (:func:`proteia.core.imaging.clipping_depth`)."""
    if bit_depth is None:
        return None
    count = clipped_pixels(image, box, size, bit_depth=bit_depth, dark_on_light=dark_on_light)
    return count >= CLIPPED_PIXELS_THRESHOLD


# On an image whose limit the exact check cannot trust (lossy compression,
# colour averaged into gray, CMYK converted), a band is possibly over-exposed
# when at least POSSIBLY_CLIPPED_PIXELS pixels of its box lie within
# NEAR_LIMIT_LEVELS grey levels of the detector limit (maintainer decision on
# #112): compression moves saturated pixels a level or two off the limit, so the
# exact count misses them. In the measurements behind it, the rule raised no
# false positive on 480 unclipped synthetic bands and missed no band that lost 1%
# or more of its signal. A count, not a share of the box: saturated pixels sit in
# the band's core, whatever the box's size. Changing either makes stored flags
# stale.
POSSIBLY_CLIPPED_PIXELS: Final = 5
NEAR_LIMIT_LEVELS: Final = 2  # on an 8-bit scale; scaled to the image's range


def near_limit_tolerance(bit_depth: int) -> float:
    """How far from the detector limit a pixel counts as near it, in the image's
    own values: :data:`NEAR_LIMIT_LEVELS` levels of an 8-bit scale, the same
    share of any range (2 at 8 bits, 514 at 16)."""
    return NEAR_LIMIT_LEVELS * (2**bit_depth - 1) / 255


def near_limit_pixels(
    image: np.ndarray, box: Box, size: BoxSize, *, bit_depth: int, dark_on_light: bool
) -> int:
    """How many pixels inside the box lie within :func:`near_limit_tolerance` of
    the detector limit (:func:`detector_limit`), on the gray analysis values: a
    colour image's mean of red, green and blue, as it is quantified, never a
    single channel (a tinted export takes one channel to the limit early).
    Raises ValueError if the box extends beyond the image bounds."""
    pixels = _box_pixels(image, box, size)
    tolerance = near_limit_tolerance(bit_depth)
    if dark_on_light:
        near = pixels <= detector_limit(bit_depth, dark_on_light=True) + tolerance
    else:
        near = pixels >= detector_limit(bit_depth, dark_on_light=False) - tolerance
    return int(np.count_nonzero(near))


def is_possibly_clipped(
    image: np.ndarray, box: Box, size: BoxSize, *, bit_depth: int | None, dark_on_light: bool
) -> bool | None:
    """Whether a band looks over-exposed on an image the exact check
    (:func:`is_clipped`) cannot trust: :data:`POSSIBLY_CLIPPED_PIXELS` or more
    pixels of its box near the detector limit (:func:`near_limit_pixels`). None
    (not assessed) when ``bit_depth`` is None: the exact check runs, or the
    image has no known limit
    (:func:`proteia.core.imaging.possible_clipping_depth`).

    False says only that the gray values show no sign of it, never that the
    band is not over-exposed: a colour channel saturated alone moves the gray
    mean a third of the way (a known limit). A single-colour export, such as a
    fluorescence image with a green lookup table, never brings its gray mean
    near the limit, so its bands are False whatever their exposure."""
    if bit_depth is None:
        return None
    count = near_limit_pixels(image, box, size, bit_depth=bit_depth, dark_on_light=dark_on_light)
    return count >= POSSIBLY_CLIPPED_PIXELS


def saturation_level(
    exact_depth: int | None, near_depth: int | None, *, dark_on_light: bool
) -> float | None:
    """The pixel value from which on a pixel counts as saturated, as the
    over-exposure checks count them: at or below it on a dark-on-light image,
    at or above it on a light-on-dark one. The detector limit of
    ``exact_depth`` (:func:`is_clipped`), else the limit of ``near_depth``
    moved in by :func:`near_limit_tolerance` (:func:`is_possibly_clipped`);
    None when neither is given (:func:`proteia.core.imaging.clipping_depth`
    and :func:`~proteia.core.imaging.possible_clipping_depth` give them).
    Row detection calls a band lighter in its centre than such pixels on
    either side of it hollow (#121)."""
    if exact_depth is not None:
        return float(detector_limit(exact_depth, dark_on_light=dark_on_light))
    if near_depth is None:
        return None
    limit = detector_limit(near_depth, dark_on_light=dark_on_light)
    tolerance = near_limit_tolerance(near_depth)
    return limit + tolerance if dark_on_light else limit - tolerance
