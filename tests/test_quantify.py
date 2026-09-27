# SPDX-License-Identifier: Apache-2.0
"""Tests for pixel-level box integration and the local band background."""

import json
import math
import tracemalloc

import numpy as np
import pytest

from proteia.core import quantify
from proteia.core.grow import mad_sigma
from proteia.core.model import Box, BoxSize
from proteia.core.quantify import (
    RING_CLAMP,
    BandBackground,
    _grouped_median,
    background_settings,
    band_backgrounds,
    clipped_pixels,
    detector_limit,
    estimate_background,
    is_clipped,
    net_signal,
    quantify_nets,
    to_grayscale,
)


def test_to_grayscale_reduces_rgb():
    rgb = np.ones((4, 4, 3))
    gray = to_grayscale(rgb)
    assert gray.shape == (4, 4)
    assert gray[0, 0] == 1.0


def test_to_grayscale_ignores_alpha_and_keeps_gray_of_gray_alpha():
    rgba = np.zeros((2, 2, 4))
    rgba[..., :3] = [30.0, 60.0, 90.0]
    rgba[..., 3] = 255.0
    assert to_grayscale(rgba)[0, 0] == 60.0  # (30 + 60 + 90) / 3; alpha ignored
    la = np.zeros((2, 2, 2))
    la[..., 0], la[..., 1] = 42.0, 255.0
    assert to_grayscale(la)[0, 0] == 42.0


def test_to_grayscale_refuses_other_layouts():
    with pytest.raises(ValueError, match="unsupported image layout"):
        to_grayscale(np.zeros((2, 2, 5)))


def test_estimate_background_is_membrane_median():
    img = np.full((10, 10), 200.0)  # uniform "membrane"
    img[4:6, 4:6] = 50.0  # a small dark band
    assert estimate_background(img) == 200.0  # band is a minority -> median = membrane


def test_net_signal_dark_band_above_background():
    img = np.full((10, 10), 200.0)
    img[2:6, 1:4] = 150.0  # 4x3 darker band, 50 below background
    net = net_signal(img, Box(x=1, y=2), BoxSize(width=3, height=4), background=200.0)
    assert net == 600.0  # 12 pixels * (200 - 150)


def test_net_signal_blank_box_is_zero():
    img = np.full((10, 10), 200.0)
    net = net_signal(img, Box(x=0, y=0), BoxSize(width=5, height=5), background=200.0)
    assert net == 0.0  # nothing darker than background


def test_net_signal_darker_band_reads_larger():
    img = np.full((12, 12), 220.0)
    img[2:5, 1:4] = 180.0  # faint band
    img[2:5, 6:9] = 100.0  # strong band
    size = BoxSize(width=3, height=3)
    bg = 220.0
    faint = net_signal(img, Box(x=1, y=2), size, bg)
    strong = net_signal(img, Box(x=6, y=2), size, bg)
    assert strong > faint > 0


def test_net_signal_light_on_dark_flips_direction():
    img = np.full((10, 10), 30.0)  # dark membrane
    img[2:5, 1:4] = 90.0  # bright band, 60 above background
    net = net_signal(
        img, Box(x=1, y=2), BoxSize(width=3, height=3), background=30.0, dark_on_light=False
    )
    assert net == 540.0  # 9 pixels * (90 - 30)


def test_net_signal_out_of_bounds_raises():
    img = np.ones((10, 10))
    with pytest.raises(ValueError, match="bounds"):
        net_signal(img, Box(x=8, y=0), BoxSize(width=5, height=2), background=0.5)


def test_net_signal_total_clamp_lets_membrane_noise_cancel():
    img = np.full((10, 10), 200.0)
    img[2:4, 2:5] = [[198.0, 202.0, 199.0], [201.0, 200.0, 203.0]]  # noise, no band
    box, size = Box(x=2, y=2), BoxSize(width=3, height=2)
    assert net_signal(img, box, size, 200.0) == 3.0  # 2 + 1: the darker pixels only
    assert net_signal(img, box, size, 200.0, clamp="total") == 0.0  # -3, floored once
    img[2:4, 2:5] -= 50.0  # a band on the same noise
    assert net_signal(img, box, size, 200.0, clamp="total") == 297.0  # 6 * 50 - 3
    light = net_signal(255.0 - img, box, size, 55.0, dark_on_light=False, clamp="total")
    assert light == 297.0


def test_a_total_clamp_sums_in_double_precision():
    # Whole 16-bit counts as float32: the signed differences cancel in the total,
    # so a level rounded to single precision (spacing 0.002 at 30000) would move
    # the net; float32 holds these counts exactly, so the net must not change.
    image = np.round(30000.0 + 40.0 * np.random.default_rng(11).standard_normal((60, 120)))
    image[10:50, 10:110] -= 5.0  # a faint band
    box, size = Box(x=10, y=10), BoxSize(width=100, height=40)
    exact = net_signal(image, box, size, 30001.83, clamp="total")
    assert net_signal(image.astype(np.float32), box, size, 30001.83, clamp="total") == exact
    rects, sizes = [(10, 10, 110, 50)], [(100, 40)]
    for dark_on_light in (True, False):
        kwargs = {"method": "ring_median", "dark_on_light": dark_on_light, "integral": True}
        nets = quantify_nets(image, rects, sizes, **kwargs)
        assert quantify_nets(image.astype(np.float32), rects, sizes, **kwargs) == nets


def test_net_signal_refuses_an_unknown_clamp():
    with pytest.raises(ValueError, match="clamp"):
        net_signal(np.ones((4, 4)), Box(x=0, y=0), BoxSize(width=2, height=2), 1.0, clamp="lane")


# --- over-exposure (clipping) ---


def test_detector_limit_follows_bit_depth_and_polarity():
    assert detector_limit(16, dark_on_light=False) == 65535
    assert detector_limit(8, dark_on_light=False) == 255
    assert detector_limit(16, dark_on_light=True) == 0


def test_a_band_at_the_detector_limit_is_clipped():
    # Light on dark, 16-bit: two pixels of the band hit 65535.
    img = np.full((10, 10), 1000.0)
    img[4:6, 4:7] = 40000.0
    img[5, 5] = img[4, 5] = 65535.0
    box, size = Box(x=3, y=3), BoxSize(width=5, height=4)
    assert clipped_pixels(img, box, size, bit_depth=16, dark_on_light=False) == 2
    assert is_clipped(img, box, size, bit_depth=16, dark_on_light=False)
    img[5, 5] = img[4, 5] = 65534.0  # one level below the limit: not clipped
    assert not is_clipped(img, box, size, bit_depth=16, dark_on_light=False)


def test_a_dark_band_pinned_at_zero_is_clipped():
    # Dark on light, 8-bit: saturation shows as black.
    img = np.full((10, 10), 200.0)
    img[4:6, 4:7] = 0.0
    box, size = Box(x=3, y=3), BoxSize(width=5, height=4)
    assert is_clipped(img, box, size, bit_depth=8, dark_on_light=True)
    assert not is_clipped(img, box, size, bit_depth=8, dark_on_light=False)  # 255 never reached
    assert not is_clipped(
        img, Box(x=0, y=0), BoxSize(width=3, height=3), bit_depth=8, dark_on_light=True
    )


def test_is_clipped_without_a_trusted_limit_is_not_checked():
    img = np.zeros((4, 4))
    assert (
        is_clipped(
            img, Box(x=0, y=0), BoxSize(width=2, height=2), bit_depth=None, dark_on_light=True
        )
        is None
    )


# --- possibly over-exposed, where the exact check cannot run (#112) ---

# A 6 x 4 box at (3, 3) on a 12 x 10 membrane.
NEAR_BOX, NEAR_SIZE = Box(x=3, y=3), BoxSize(width=6, height=4)


def _near(pixels: int, value: float, *, background: float, dtype=np.float64) -> np.ndarray:
    """A membrane at ``background`` with ``pixels`` pixels of the box at ``value``."""
    img = np.full((10, 12), background, dtype=dtype)
    img[3:7, 3:9][_first(pixels)] = value
    return img


def _first(pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """The first ``pixels`` pixels of the box, row by row, as indices into it."""
    return np.unravel_index(np.arange(pixels), (NEAR_SIZE.height, NEAR_SIZE.width))


def _possibly(img: np.ndarray, bit_depth: int | None, *, dark_on_light: bool) -> bool | None:
    return quantify.is_possibly_clipped(
        img, NEAR_BOX, NEAR_SIZE, bit_depth=bit_depth, dark_on_light=dark_on_light
    )


def test_the_possibly_over_exposed_rule_is_the_decided_one():
    # The maintainer's decision on #112: at least 5 pixels within 2 grey levels
    # (8-bit) of the limit.
    assert quantify.POSSIBLY_CLIPPED_PIXELS == 5
    assert quantify.NEAR_LIMIT_LEVELS == 2
    # The same share of any range: 2 levels of 255, so 514 counts of 65535.
    assert quantify.near_limit_tolerance(8) == 2.0
    assert quantify.near_limit_tolerance(16) == 514.0


@pytest.mark.parametrize(
    ("bit_depth", "dark_on_light", "near", "beyond", "background"),
    [
        (8, True, 2.0, 3.0, 200.0),  # dark on light: the limit is 0
        (8, False, 253.0, 252.0, 20.0),  # light on dark: the limit is 255
        (16, True, 514.0, 515.0, 50000.0),
        (16, False, 65021.0, 65020.0, 1000.0),  # 65535 - 514
    ],
)
def test_five_pixels_within_two_levels_of_the_limit_are_possibly_over_exposed(
    bit_depth, dark_on_light, near, beyond, background
):
    # The two edges of the rule: 4 or 5 pixels, 2 or 3 levels (8-bit scale).
    def possibly(pixels: int, value: float) -> bool | None:
        img = _near(pixels, value, background=background)
        return _possibly(img, bit_depth, dark_on_light=dark_on_light)

    assert possibly(5, near) is True
    assert possibly(4, near) is False
    assert possibly(5, beyond) is False
    assert possibly(24, beyond) is False  # the whole box one level too far
    img = _near(5, near, background=background)
    count = quantify.near_limit_pixels(
        img, NEAR_BOX, NEAR_SIZE, bit_depth=bit_depth, dark_on_light=dark_on_light
    )
    assert count == 5
    # The other polarity's limit is the far end of the range: nothing is near it.
    assert _possibly(img, bit_depth, dark_on_light=not dark_on_light) is False


def test_pixels_near_the_limit_outside_the_box_do_not_count():
    img = _near(4, 0.0, background=200.0)
    img[0:2, 0:3] = 0.0  # six pixels at the limit beside the box
    assert _possibly(img, 8, dark_on_light=True) is False
    img[3, 8] = 1.0  # the box's last column: now five
    assert _possibly(img, 8, dark_on_light=True) is True


def test_integer_pixels_are_measured_without_wrapping():
    # The analysis array is float, but raw 8- and 16-bit pixels must give the
    # same answer: nothing below 0 wraps round to the top of the range.
    for dtype, bit_depth, background in ((np.uint8, 8, 200), (np.uint16, 16, 50000)):
        img = _near(5, 0, background=background, dtype=dtype)
        assert _possibly(img, bit_depth, dark_on_light=True) is True
        assert _possibly(img, bit_depth, dark_on_light=False) is False


def test_a_colour_box_is_measured_on_its_gray_mean():
    # The mean of red, green and blue, as the image is quantified: (0, 0, 6)
    # averages to 2, near the limit; (0, 0, 7) to 2.33, not. A tinted export
    # whose blue channel alone reaches 0 is not flagged (#112's measurements).
    rgb = np.full((10, 12, 3), 200.0)
    box = rgb[3:7, 3:9]  # a view into rgb
    box[_first(5)] = (0.0, 0.0, 6.0)
    assert _possibly(rgb, 8, dark_on_light=True) is True
    box[_first(5)] = (0.0, 0.0, 7.0)
    assert _possibly(rgb, 8, dark_on_light=True) is False
    box[:] = (40.0, 12.0, 0.0)  # a tint: the gray mean is 17.3
    assert _possibly(rgb, 8, dark_on_light=True) is False


def test_possibly_over_exposed_is_not_assessed_without_a_depth():
    # None: the exact check runs on the image, or it has no known range.
    img = _near(24, 0.0, background=200.0)
    assert _possibly(img, None, dark_on_light=True) is None


@pytest.mark.parametrize(
    ("exact", "near", "dark_on_light", "level"),
    [
        (16, None, True, 0.0),  # the exact check's limit
        (16, None, False, 65535.0),
        (8, 8, True, 0.0),  # the exact depth wins
        (None, 8, True, 2.0),  # the near-limit check's: 2 levels in from it
        (None, 8, False, 253.0),
        (None, 16, True, 514.0),
        (None, 16, False, 65021.0),
        (None, None, True, None),  # no known limit
    ],
)
def test_the_saturation_level_is_where_the_over_exposure_checks_count_pixels(
    exact, near, dark_on_light, level
):
    # Row detection calls a band hollow from the pixels these checks count
    # (#121): at or below the level on a dark-on-light image, at or above it
    # on a light-on-dark one. The near levels are the last values the
    # possibly-over-exposed test above counts.
    assert quantify.saturation_level(exact, near, dark_on_light=dark_on_light) == level


# --- local background: ring_median (#83) ---

# A 24 x 10 box whose centre is (71.5, 54.5), on a 160 x 120 image.
H, W = 120, 160
RECT, SIZE = (60, 50, 84, 60), (24, 10)


def _grid() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:H, 0:W].astype(float)
    return y, x


def _membrane(level: float = 200.0, noise: float = 2.0, seed: int = 1) -> np.ndarray:
    """A flat membrane with white noise (not yet rounded)."""
    return level + noise * np.random.default_rng(seed).standard_normal((H, W))


def _with_band(image: np.ndarray, rect, depth: float = 60.0) -> np.ndarray:
    """A dark band centred in ``rect``: flat-topped across the lane, Gaussian down it."""
    y, x = _grid()
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2
    return image - depth * np.exp(-(((x - cx) / 9.0) ** 4) - ((y - cy) / 3.0) ** 2)


def _with_disk(image: np.ndarray, cx: float, cy: float, radius: float, delta: float):
    y, x = _grid()
    return image + delta * ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2)


def _level(image: np.ndarray, *, integral: bool = True) -> float:
    """The level of the one box at RECT, dark on light."""
    (found,) = band_backgrounds(
        image,
        [RECT],
        [SIZE],
        dark_on_light=True,
        integral=integral,
        fallback=estimate_background(image),
    )
    assert found.mode == "symmetric"
    return found.level


def _blot() -> tuple[np.ndarray, list, list]:
    """Two proteins of different, odd box sizes (27 x 11 and 22 x 13: every gap
    is rounded up) over four lanes on a gradient membrane, with haze down the
    third lane (lifted) and a smear left of the first lane (uneven), whole
    counts, plus a box in the image corner (an asymmetric ring)."""
    y, x = _grid()
    image = _membrane() + 0.1 * x - 0.05 * y
    image[:, 86:115] -= 6.0  # haze down the third lane
    image[:, :15] -= 20.0  # a smear left of the first lane
    rects, sizes = [], []
    for row_y, (w, h) in ((40, (27, 11)), (80, (22, 13))):
        for lane_x in (30, 65, 100, 135):
            rect = (lane_x - w // 2, row_y - h // 2, lane_x - w // 2 + w, row_y - h // 2 + h)
            image = _with_band(image, rect)
            rects.append(rect)
            sizes.append((w, h))
    corner = (0, 0, 20, 8)
    image = _with_band(image, corner, depth=40.0)
    return np.round(image), [*rects, corner], [*sizes, (20, 8)]


@pytest.mark.parametrize("dark_on_light", [True, False])
def test_a_noise_free_plane_reads_the_plane_at_the_box_centre(dark_on_light):
    y, x = _grid()
    plane = 150.3 + 0.37 * x - 0.21 * y  # not whole numbers: the plain median
    kwargs = {"dark_on_light": dark_on_light, "integral": False, "fallback": 0.0}
    alone = band_backgrounds(plane, [RECT], [SIZE], **kwargs)
    # Another protein's box, of another size: its zone (rows 30-45, columns
    # 39-80) removes the ring's rows 40-44 above the box, and their pairs below.
    # A feature inside that zone (3 levels lighter) is then never read.
    other, other_size = (45, 34, 75, 42), (30, 8)
    featured = plane.copy()
    featured[30:46, 39:81] += 3.0
    cut = band_backgrounds(featured, [RECT, other], [SIZE, other_size], **kwargs)
    for found in (alone[0], cut[0]):
        assert found.mode == "symmetric"
        assert found.level == pytest.approx(150.3 + 0.37 * 71.5 - 0.21 * 54.5, abs=1e-9)
    assert cut[1].mode == "symmetric"
    assert cut[1].level == pytest.approx(150.3 + 0.37 * 59.5 - 0.21 * 37.5, abs=1e-9)
    # without the other box the ring reads the feature: 166.81 instead of 165.31
    (uncut,) = band_backgrounds(featured, [RECT], [SIZE], **kwargs)
    assert uncut.level > alone[0].level + 1.0


def test_the_gap_is_rounded_up_and_the_ring_measured_from_the_zone():
    # gaps: ceil(0.2 w) across the lane, ceil(0.5 h) along it
    assert quantify._zone((60, 50, 87, 61)) == (54, 44, 93, 67)  # 27 x 11: 6 and 6
    assert quantify._zone((60, 50, 82, 63)) == (55, 43, 87, 70)  # 22 x 13: 5 and 7
    # The ring of the 22 x 13 box (need 2 x 286 = 572 px): 4 px around the zone
    # hold 536, so it is 5 px wide, 42 x 37 less the 32 x 27 zone.
    rect = (60, 50, 82, 63)
    mode, ys, xs = quantify._ring(quantify._exclusion_mask((H, W), [rect]), rect)
    assert mode == "symmetric"
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (50, 91, 38, 74)
    assert xs.size == 42 * 37 - 32 * 27


def test_the_ring_reaches_as_far_as_the_longer_box_side():
    # A 20-row strip: the zone of a 24 x 10 box spans its height, so the ring lies
    # left and right of it. Two columns fit left of the gap, too few pairs: the
    # ring is asymmetric and grows until it holds 2 box areas (480 px) within
    # max(24, 10) px of the zone: 40 px left of the box centre, 440 right.
    rect = (7, 5, 31, 15)
    mode, ys, xs = quantify._ring(quantify._exclusion_mask((20, 200), [rect]), rect)
    assert mode == "asymmetric"
    assert (np.count_nonzero(xs < 18.5), np.count_nonzero(xs > 18.5)) == (40, 440)
    assert xs.max() == 57  # 22 px past the zone, which ends at column 35


@pytest.mark.parametrize(("columns", "mode"), [(5, "asymmetric"), (6, "symmetric")])
def test_the_ring_stays_paired_while_its_pairs_hold_one_box_area(columns, mode):
    # A 20-row strip: only the `columns` left of the zone of a 24 x 10 box (and
    # as many right of it) pair, 40 px per column: 200 px or 240 px, one box area.
    x0 = columns + 5
    image = np.random.default_rng(0).standard_normal((20, x0 + 229))
    (found,) = band_backgrounds(
        image, [(x0, 5, x0 + 24, 15)], [(24, 10)], dark_on_light=True, integral=False, fallback=9.0
    )
    assert found.mode == mode


@pytest.mark.parametrize(("width", "mode"), [(36, "image"), (37, "asymmetric")])
def test_a_ring_under_a_quarter_box_area_falls_back_to_the_image_level(width, mode):
    # A 24 x 10 box in a 20-row strip: its ring is column 0 and the columns from
    # 35, 40 px or 60 px, against max(10, 240 / 4) = 60.
    image = np.random.default_rng(0).standard_normal((20, width))
    (found,) = band_backgrounds(
        image, [(6, 5, 30, 15)], [(24, 10)], dark_on_light=True, integral=False, fallback=9.0
    )
    assert found.mode == mode
    assert (found.level == 9.0) == (mode == "image")


# _blot() under ring_median v1 as the #83 reference prototype (m_final) measures
# it, dark on light: (level, mode, spread). The third lane's boxes are lifted;
# the first lane's (beside the smear) and the third lane's spread.
V1_BLOT = [
    (201.4277108433735, "symmetric", 9.13753247649587),
    (203.95505617977528, "symmetric", 0.0),
    (203.37593681248632, "symmetric", 3.520134522952485),
    (211.41666666666666, "symmetric", 0.0),
    (199.41071428571428, "symmetric", 8.642378881005017),
    (202.43684210526317, "symmetric", 0.0),
    (201.65391075221766, "symmetric", 3.263972344567808),
    (209.32584269662922, "symmetric", 0.0),
    (199.94042289175445, "asymmetric", 0.0),
]


def test_ring_median_v1_matches_the_reference_prototype():
    # A change here is a new method version: stored nets must be requantified.
    image, rects, sizes = _blot()
    for dark_on_light, pixels in ((True, image), (False, 255.0 - image)):
        found = band_backgrounds(
            pixels,
            rects,
            sizes,
            dark_on_light=dark_on_light,
            integral=True,
            fallback=estimate_background(pixels),
        )
        for bg, (level, mode, spread) in zip(found, V1_BLOT, strict=True):
            assert bg.mode == mode
            assert bg.level == pytest.approx(level if dark_on_light else 255.0 - level, rel=1e-9)
            assert bg.spread == pytest.approx(spread, rel=1e-9, abs=1e-12)


def test_the_v1_settings_outside_the_record_are_pinned():
    # The record states the method's main parameters; these finer ones are covered
    # by its version number, so changing one makes a new version too.
    pinned = {
        "RING_VERSION": 1,
        "RING_MAX_ITER": 20,
        "RING_SIDE_MIN": 0.125,
        "RING_HAZE_ROWS": 2,
        "RING_PLANE_SIDE": 0.1,
        "_MIN_SAMPLE": 10,
        "_SPREAD_Z": 3.0,
        "_MEDIAN_SE": 1.2533,
        "_SIGMA_FLOOR_INTEGRAL": 0.5,
    }
    assert {name: getattr(quantify, name) for name in pinned} == pinned


def test_the_polarity_mirror_gives_equal_nets():
    image, rects, sizes = _blot()
    kwargs = {"method": "ring_median", "integral": True}
    nets = quantify_nets(image, rects, sizes, dark_on_light=True, **kwargs)
    mirrored = quantify_nets(255.0 - image, rects, sizes, dark_on_light=False, **kwargs)
    assert mirrored == pytest.approx(nets, rel=1e-12)
    modes = [
        found.mode
        for found in band_backgrounds(
            image, rects, sizes, dark_on_light=True, integral=True, fallback=0.0
        )
    ]
    assert set(modes) == {"symmetric", "asymmetric"}  # both paths mirrored


def test_results_do_not_depend_on_the_order_of_boxes_or_proteins():
    image, rects, sizes = _blot()
    kwargs = {"dark_on_light": True, "integral": True, "fallback": estimate_background(image)}
    base = band_backgrounds(image, rects, sizes, **kwargs)
    nets = quantify_nets(
        image, rects, sizes, method="ring_median", dark_on_light=True, integral=True
    )
    proteins_swapped = [4, 5, 6, 7, 0, 1, 2, 3, 8]
    reversed_boxes = list(range(len(rects)))[::-1]
    shuffled = [int(i) for i in np.random.default_rng(7).permutation(len(rects))]
    for order in (proteins_swapped, reversed_boxes, shuffled):
        got = band_backgrounds(
            image, [rects[i] for i in order], [sizes[i] for i in order], **kwargs
        )
        assert [got[order.index(k)] for k in range(len(rects))] == base
        got_nets = quantify_nets(
            image,
            [rects[i] for i in order],
            [sizes[i] for i in order],
            method="ring_median",
            dark_on_light=True,
            integral=True,
        )
        assert [got_nets[order.index(k)] for k in range(len(rects))] == nets


def test_the_first_ring_window_changes_speed_only(monkeypatch):
    # Crowded: six lanes 2 px apart in three rows, the second only 16 px below the
    # first, so most rings outgrow the window sized for an uncut ring.
    image, rects, sizes = _membrane(), [], []
    for row_y in (40, 56, 90):
        for lane in range(6):
            rect = (6 + 26 * lane, row_y - 5, 30 + 26 * lane, row_y + 5)
            image = _with_band(image, rect)
            rects.append(rect)
            sizes.append((24, 10))
    image = np.round(image)
    kwargs = {"dark_on_light": True, "integral": True, "fallback": 200.0}
    reaches = []
    window = quantify._ring_candidates

    def counted(excluded, rect, reach, *, paired):
        reaches.append(reach)
        return window(excluded, rect, reach, paired=paired)

    monkeypatch.setattr(quantify, "_ring_candidates", counted)
    found = band_backgrounds(image, rects, sizes, **kwargs)
    # measured 31 paired windows for 18 rings, and 6 unpaired for the asymmetric
    assert len(reaches) > len(rects)
    assert {bg.mode for bg in found} == {"symmetric", "asymmetric"}
    monkeypatch.setattr(quantify, "_initial_reach", lambda zone, need: 10**9)  # full reach
    assert band_backgrounds(image, rects, sizes, **kwargs) == found


@pytest.mark.parametrize(
    ("rects", "sizes"),
    [
        ([(0, 0, W, H)], [(W, H)]),  # a box filling the image: image mode
        ([(30, 20, 130, 100)], [(100, 80)]),  # its zone spans the image height
        # six whole-lane boxes whose rings need the full reach
        ([(6 + 24 * i, 10, 18 + 24 * i, 110) for i in range(6)], [(12, 100)] * 6),
    ],
    ids=["filling", "large", "lanes"],
)
def test_a_large_box_costs_memory_in_proportion_to_the_image(rects, sizes):
    image = np.round(_membrane())
    kwargs = {"dark_on_light": True, "integral": True, "fallback": 200.0}
    band_backgrounds(image, rects, sizes, **kwargs)  # one-time costs (the first lstsq)
    tracemalloc.start()
    try:
        band_backgrounds(image, rects, sizes, **kwargs)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # measured 0.4, 2.2 and 2.2 times the image, mostly the plane fit over the
    # ring; a window reaching max(w, h) past the zone everywhere took 29, 12 and 7
    assert peak < 4 * image.nbytes


def test_the_two_sided_clip_ignores_a_lighter_bubble_beside_the_box(monkeypatch):
    clean = _with_band(_membrane(), RECT)
    # 30 levels lighter, left of the box: 16 % of the ring's 496 pixels
    bubbled = _with_disk(clean, 48, 55, 11, 30.0)
    shift = _level(np.round(bubbled)) - _level(np.round(clean))
    assert abs(shift) < 0.11  # the ring's noise se, 1.2533 * 2 / sqrt(496); measured 0.023
    # without the clip the bubble pulls the level towards the light side
    monkeypatch.setattr(quantify, "RING_CLIP_K", math.inf)
    assert _level(np.round(bubbled)) - _level(np.round(clean)) > 0.3  # measured 0.47


def test_haze_down_the_lane_lifts_the_level_to_the_lane(monkeypatch):
    hazy = _membrane()
    hazy[:, 58:86] -= 8.0  # the lane, 2 px wider than the box each side, full height
    image = np.round(_with_band(hazy, RECT))
    # the lift moves the level to the in-lane level, 192.0, less two standard
    # errors of 0.76 (of the median of 96 px on the ring's sigma)
    assert _level(image) == pytest.approx(193.517, abs=0.01)
    # the ring alone reads mostly membrane
    monkeypatch.setattr(quantify, "RING_HAZE_ROWS", 10**6)
    assert _level(image) > 197.0  # measured 197.4


def test_the_lift_uses_the_median_row_mean_and_the_larger_standard_error():
    hazy = _membrane()
    hazy[:, 58:86] -= 8.0
    # The ring's four in-lane rows above the box (41-44) alternate 2 levels
    # lighter and darker: their means scatter, so the standard error of that
    # side is 1.2533 x 1.4826 x 2 / sqrt(4) = 1.86, not 0.76. The lift stops the
    # larger of the two sides' errors short of the in-lane level, 192.0.
    hazy[41:45, 58:86] += 2.0 * np.array([1.0, -1.0, 1.0, -1.0])[:, None]
    level = _level(np.round(_with_band(hazy, RECT)))
    assert level == pytest.approx(192.0 + 2 * 1.86, abs=0.01)  # measured 195.716
    # a lighter row below (a scratch across the lane) moves the mean of that
    # side's row means, not their median
    hazy[66, 58:86] += 15.0
    assert _level(np.round(_with_band(hazy, RECT))) == level


def test_two_complete_rows_a_side_are_enough_to_lift():
    hazy = _membrane()
    hazy[:, 58:86] -= 8.0
    image = np.round(_with_band(hazy, RECT))
    # Another protein's box just above: its zone ends at row 42, leaving the ring
    # two complete in-lane rows above the box (43-44) and, paired, two below
    # (65-66): 48 px a side, over the S / 8 = 30 a side needs.
    above, above_size = (60, 31, 84, 39), (24, 8)
    (found, _) = band_backgrounds(
        image,
        [RECT, above],
        [SIZE, above_size],
        dark_on_light=True,
        integral=True,
        fallback=200.0,
    )
    assert found.level == pytest.approx(193.07, abs=0.01)  # 197.4 without the lift


def test_the_spread_reports_a_background_that_differs_around_the_box():
    kwargs = {"dark_on_light": True, "integral": True, "fallback": 200.0}
    flat = np.round(_with_band(_membrane(), RECT))
    assert band_backgrounds(flat, [RECT], [SIZE], **kwargs)[0].spread == 0.0
    hazy = _membrane()
    hazy[:, 58:86] -= 8.0
    (found,) = band_backgrounds(np.round(_with_band(hazy, RECT)), [RECT], [SIZE], **kwargs)
    # the ring, the lifted level and the side pairs disagree by levels, not noise
    assert found.spread > 3.0  # measured 5.0


def test_the_spread_compares_the_midpoints_of_opposite_sides():
    y, x = _grid()
    noise = 2.0 * np.random.default_rng(1).standard_normal((H, W))
    kwargs = {"dark_on_light": True, "integral": True, "fallback": 200.0}
    # Steep gradients: the left and right sides differ by 73 levels, the top and
    # bottom by 48, but each pair's midpoint is the ring level: no spread.
    for membrane in (200.0 + 2.0 * (x - 71.5), 200.0 + 2.0 * (y - 54.5)):
        (found,) = band_backgrounds(np.round(membrane + noise), [RECT], [SIZE], **kwargs)
        assert found.spread == 0.0
    # 15 levels darker left of the box only: the left-right midpoint (192.52)
    # lies 7.20 from the top-bottom one (199.72), less 3 standard errors of a
    # median of the smaller sides' 96 px: 7.20 - 3 x 1.2533 x 1.48 x sqrt(2 / 96)
    step = np.where(x < 60, 185.0, 200.0)
    (found,) = band_backgrounds(np.round(step + noise), [RECT], [SIZE], **kwargs)
    assert found.spread == pytest.approx(6.3975, abs=1e-4)  # measured 6.397484
    # The ring level is a candidate too. A smear left of the lane, 25 levels
    # darker, is clipped out of the ring but read by the left side, and haze in
    # the lane lifts the level: the ring level (197.35) reads less signal than
    # the lifted level (196.76) and both midpoints (187.45 and 195.72).
    smear = np.where(x < 59, 175.0, 200.0) - 4.0 * ((x >= 59) & (x < 85))
    (found,) = band_backgrounds(np.round(smear + noise), [RECT], [SIZE], **kwargs)
    assert found.spread == pytest.approx(197.353 - 187.446 - 3 * 0.5364, abs=1e-3)


@pytest.mark.parametrize(
    "feature",
    ["unboxed band above", "unboxed band below", "bubble above", "dark spot above"],
)
def test_a_feature_on_one_side_of_the_lane_does_not_lift(feature):
    image = _with_band(_membrane(), RECT)
    plain = _level(np.round(image))  # 199.81
    if feature == "unboxed band above":
        image = _with_band(image, (60, 34, 84, 44), depth=40.0)
    elif feature == "unboxed band below":
        image = _with_band(image, (60, 66, 84, 76), depth=40.0)
    elif feature == "bubble above":
        image = _with_disk(image, 72, 40, 6, 30.0)
    else:
        image = _with_disk(image, 72, 40, 6, -15.0)
    assert abs(_level(np.round(image)) - plain) < 0.3  # measured at most 0.12


def test_a_vertical_gradient_does_not_lift():
    y, _ = _grid()
    image = _with_band(_membrane() + 0.8 * (y - 54.5), RECT)  # 200 at the box centre
    # one in-lane side always reads less signal than the ring: no lift
    assert _level(np.round(image)) == pytest.approx(200.0, abs=0.3)  # measured 200.0


def test_a_darker_step_down_the_lane_reads_the_box_mean_background():
    membrane = np.full((H, W), 200.25)
    membrane[:, 76:84] -= 10.0  # under the box's right third, the whole lane long
    image = membrane.copy()
    image[50:60, 60:84] -= 60.0 * np.exp(-(((np.arange(50, 60) - 54.5) / 3.0) ** 2))[:, None]
    # The ring reads 200.25 (the darker pixels are a minority and clipped). The
    # rows above and below over the box's columns read its mean background; the
    # median of their pixels would read the membrane, 200.25.
    assert _level(image, integral=False) == pytest.approx(200.25 - 10.0 * 8 / 24, abs=1e-9)
    assert estimate_background(image) == 200.25


def test_the_asymmetric_plane_interpolates_and_never_extrapolates():
    y, x = np.mgrid[0:60, 0:90].astype(float)
    noise = 0.5 * np.random.default_rng(3).standard_normal((60, 90))
    corner, size = (0, 0, 20, 10), (20, 10)  # centre (9.5, 4.5): no pixel pairs

    def level(image):
        (found,) = band_backgrounds(
            image, [corner], [size], dark_on_light=True, integral=False, fallback=0.0
        )
        assert found.mode == "asymmetric"
        return found.level

    # a gentle gradient: the plane at the box centre, 106.1
    gentle = 100.0 + 0.5 * x + 0.3 * y + noise
    assert level(gentle) == pytest.approx(106.1, abs=0.3)
    # An unboxed band below and right of the box covers 45 % of its ring: the
    # constant clip drops it before any slope is fitted, so the plane holds (a
    # least-squares start over all the pixels would tilt to 117.0).
    lower = 60.0 * np.exp(-(((x - 30.0) / 8.0) ** 4) - ((y - 18.0) / 6.0) ** 2)
    assert level(gentle - lower) == pytest.approx(106.1, abs=0.3)  # measured 106.25
    # A steep one: the plane at the centre (118.5) lies below every ring pixel,
    # whose clean values span 124 (at x = 24, y = 0) to 211. An unboxed band next
    # to the box is clipped before any slope is fitted.
    steep = 100.0 + 1.0 * x + 2.0 * y + noise
    band = 40.0 * np.exp(-(((x - 33.0) / 5.0) ** 4) - ((y - 4.5) / 3.0) ** 2)
    alone, beside_band = level(steep), level(steep - band)
    assert 124.0 - 4 * 0.5 <= alone <= 211.0 + 4 * 0.5  # measured 123.91: clamped
    assert beside_band == pytest.approx(alone, abs=0.05)


def test_the_asymmetric_plane_fits_no_slope_along_an_axis_it_does_not_straddle():
    # The strip ring above: 40 px left of the box centre, fewer than the 10 % (48)
    # a slope across needs, so none is fitted there. Its level is then the clipped
    # median of the right side, columns 36-57 (the far-left columns are clipped);
    # a slope across would extrapolate to the plane at the centre, 116.65.
    x = np.mgrid[0:20, 0:200][1].astype(float)
    (found,) = band_backgrounds(
        100.0 + 0.9 * x,
        [(7, 5, 31, 15)],
        [(24, 10)],
        dark_on_light=True,
        integral=False,
        fallback=0.0,
    )
    assert found.mode == "asymmetric"
    assert found.level == pytest.approx(100.0 + 0.9 * 46.5, abs=1e-9)
    # the QC spread is measured on paired rings only
    assert found.spread == 0.0


def test_a_box_filling_the_image_falls_back_to_the_image_level():
    image = np.round(_with_band(_membrane(), RECT))[50:60, 60:84]
    found = band_backgrounds(
        image, [(0, 0, 24, 10)], [SIZE], dark_on_light=True, integral=True, fallback=123.25
    )
    assert found == [BandBackground(level=123.25, mode="image", spread=0.0)]
    nets = quantify_nets(
        image, [(0, 0, 24, 10)], [SIZE], method="ring_median", dark_on_light=True, integral=True
    )
    box, size = Box(x=0, y=0), BoxSize(width=24, height=10)
    assert nets == [net_signal(image, box, size, estimate_background(image), clamp="total")]


def test_the_grouped_median_interpolates_whole_counts():
    assert _grouped_median(np.array([1.0, 2.0, 2.0, 2.0, 3.0])) == 2.0
    assert _grouped_median(np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])) == 2.5
    assert _grouped_median(np.array([5.0, 5.0, 5.0, 6.0])) == pytest.approx(4.5 + 2 / 3)
    # an even count interpolates in the lower middle value's class: the top of 1's
    assert _grouped_median(np.array([3.0, 1.0, 3.0, 1.0])) == 1.5
    assert _grouped_median(np.array([1.5, 2.5, 7.0])) == 2.5  # not whole: the plain median


def test_the_clip_iterates_until_the_kept_values_stop_changing(monkeypatch):
    # membrane noise with a long one-sided tail (a band's flank): each pass
    # narrows sigma and drops more of the tail
    rng = np.random.default_rng(4)
    tail = 200.0 + rng.exponential(15.0, 200)
    values = np.round(np.concatenate([200.0 + 2.0 * rng.standard_normal(400), tail]))

    def settled(keep, sigma):
        center = float(np.median(values[keep]))
        return sigma == max(mad_sigma(values[keep], center), 0.5) and np.array_equal(
            np.abs(values - center) <= 3.0 * sigma, keep
        )

    assert settled(*quantify._clip(values, 0.5))
    monkeypatch.setattr(quantify, "RING_MAX_ITER", 1)
    assert not settled(*quantify._clip(values, 0.5))  # one pass is not enough here


def test_a_ring_of_whole_counts_resolves_a_level_between_counts():
    # 30 % of the membrane reads one count lighter: the background is about 200.2
    image = 200.0 + (np.random.default_rng(5).random((H, W)) < 0.3)
    assert estimate_background(image) == 200.0
    assert 200.1 < _level(image) < 200.4  # measured 200.24; 199.5 + 0.5 / 0.7 = 200.21


def test_quantify_nets_runs_either_method_through_the_one_path():
    image, rects, sizes = _blot()
    median = estimate_background(image)
    boxes = [
        (Box(x=r[0], y=r[1]), BoxSize(width=s[0], height=s[1]))
        for r, s in zip(rects, sizes, strict=True)
    ]
    for dark_on_light in (True, False):
        nets = quantify_nets(
            image,
            rects,
            sizes,
            method="global_median",
            dark_on_light=dark_on_light,
            integral=True,
        )
        assert nets == [
            net_signal(image, box, size, median, dark_on_light=dark_on_light) for box, size in boxes
        ]
        found = band_backgrounds(
            image, rects, sizes, dark_on_light=dark_on_light, integral=True, fallback=median
        )
        nets = quantify_nets(
            image, rects, sizes, method="ring_median", dark_on_light=dark_on_light, integral=True
        )
        assert nets == [
            net_signal(image, box, size, bg.level, dark_on_light=dark_on_light, clamp=RING_CLAMP)
            for (box, size), bg in zip(boxes, found, strict=True)
        ]
    assert RING_CLAMP == "total"


def test_band_backgrounds_refuses_boxes_that_do_not_fit():
    image = np.zeros((20, 20))
    kwargs = {"dark_on_light": True, "integral": True, "fallback": 0.0}
    assert band_backgrounds(image, [], [], **kwargs) == []
    with pytest.raises(ValueError, match="bounds"):
        band_backgrounds(image, [(15, 0, 25, 5)], [(10, 5)], **kwargs)
    with pytest.raises(ValueError, match="is not 10 x 6"):
        band_backgrounds(image, [(0, 0, 10, 5)], [(10, 6)], **kwargs)
    with pytest.raises(ValueError, match="sizes"):
        band_backgrounds(image, [(0, 0, 10, 5)], [], **kwargs)
    with pytest.raises(ValueError, match="method"):
        quantify_nets(
            image, [(0, 0, 10, 5)], [(10, 5)], method="lane", dark_on_light=True, integral=True
        )


def test_background_settings_state_the_method_and_its_parameters():
    settings = background_settings()
    assert settings == {
        "method": "ring_median",
        "version": 1,
        "gap": {"across_lane": 0.2, "along_lane": 0.5, "unit": "box size"},
        "ring": {
            "initial_width_px": 3,
            "min_pixels_per_box_area": 2.0,
            "max_width": "max(box width, box height)",
            "exclude": "every box on the image, dilated by its gap",
            "pairing": "point-symmetric about the box centre",
        },
        "statistic": {"clip": "two-sided, 3.0 x 1.4826 MAD, iterated", "level": "grouped median"},
        "lane_haze": {
            "in_lane_level": "median of complete row means over the box columns, per side",
            "rule": "raised to min(T, B) - z se when both exceed the ring level by more",
            "z": 2.0,
        },
        "fallbacks": {
            "asymmetric": "symmetric ring < 1.0 box area: robust plane at box centre over the "
            "full ring (slopes only where pixels lie on both sides; clamped to kept range)",
            "image": "< max(10, 0.25 box area) ring pixels: image median",
        },
        "clamp": "box total floored at 0",
        "qc": {"uneven_limit": 0.05},
    }
    assert json.loads(json.dumps(settings)) == settings
