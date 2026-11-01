# SPDX-License-Identifier: Apache-2.0
"""The channel rule (#206): the channel Proteia picks for a colour image when
it imports it or its polarity changes, the evidence it stores, and the notes
that say what each image is measured on.

The images are synthetic, 200 x 120 pixels, with six lanes of bands drawn
with additions, multiplications and divisions only and integer noise from
fixed seeds (five per image), so their pixels are the same everywhere: bands
in one colour, light on black as a fluorescence scan or dark on white;
saturated in one lane; two targets in two colours at several strengths; a
gray blot saved in colour, tinted, or merged with a coloured ladder; palettes;
dust, ramps, noise and an edge strip; a gray image saved as RGB and with alpha;
a prestained ladder photographed in visible light; a gray blot exported with
its saturated pixels painted in a colour (a known limit: measured on the mean,
its saturation goes unflagged, which an expected failure pins). Each must
pick the channel written beside it, and each simpler rule one might use
instead (the widest range, the most mass without the region filter, the best
signal to noise, the fewest saturated pixels, the mean alone, the largest
spread, weighted luminance) picks another on at least one of them.

The tests in the groups marked so pin a choice still pending: a colour image
whose three channels all hold its bands (every channel's mass at least half
the largest, painted saturation included) is measured on their mean, and a
second colour of a two-colour file is measured on a copy of the image imported
again.
"""

import io
import json
import time
import tracemalloc
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import tifffile
from PIL import Image

from conftest import FakeClock, as_legacy
from proteia.core import imaging, record, results, storage
from proteia.core import operations as ops
from proteia.core.model import (
    BoxSize,
    Channel,
    ChannelChoice,
    ChannelSource,
    ImageKind,
    Polarity,
    Role,
    apply_change,
)
from proteia.core.operations import ErrorCode, LaneInput, OperationError
from proteia.core.quantify import quantify_nets
from proteia.core.session import save_to_folder
from proteia.web import api
from proteia.web.state import project_state
from test_operations import session_on
from test_schema2_golden import DATA

CHEMI, MARKER, MERGED = ImageKind.CHEMILUMINESCENCE, ImageKind.VISIBLE_MARKER, ImageKind.MERGED
DARK, LIGHT = Polarity.DARK_ON_LIGHT, Polarity.LIGHT_ON_DARK
RED, GREEN, BLUE, MEAN = Channel.RED, Channel.GREEN, Channel.BLUE, Channel.MEAN
PICKED, CHOSEN = ChannelSource.PICKED, ChannelSource.CHOSEN
SEEDS = (1, 2, 3, 4, 5)

# --- The synthetic images ---

H, W = 120, 200
LANE_X = tuple(25.0 + 30.0 * i for i in range(6))
ROW_A, ROW_B = 40.0, 84.0
AMPS = (60.0, 120.0, 90.0, 150.0, 30.0, 100.0)
SATURATING = (60.0, 120.0, 90.0, 400.0, 30.0, 100.0)  # lane 4 runs past 255
WEAK = (25.0, 50.0, 37.0, 62.0, 12.0, 42.0)
FAINT = (10.0,) * 6
BLACK, WHITE = 12.0, 235.0
BOX = BoxSize(width=20, height=12)
SATURATED_LANE = 3  # lane 4
LANES = tuple(range(6))
LADDER_Y = (16.0, 34.0, 52.0, 70.0, 88.0, 106.0)
# A prestained ladder's bands as each absorbs red, green and blue (dark on
# light): blue, blue, orange, blue, green, blue.
LADDER_INKS = (
    (1.0, 0.9, 0.1),
    (1.0, 0.9, 0.1),
    (0.1, 0.7, 1.0),
    (1.0, 0.9, 0.1),
    (0.9, 0.1, 0.9),
    (1.0, 0.9, 0.1),
)


def noise(shape, seed: int, amplitude: int = 2) -> np.ndarray:
    """The sum of three uniform integer draws in [-amplitude, amplitude]."""
    rng = np.random.default_rng(seed)
    total = np.zeros(shape, dtype=np.int64)
    for _ in range(3):
        total += rng.integers(-amplitude, amplitude + 1, size=shape)
    return total


def bands(shape, placed) -> np.ndarray:
    """Flat-topped bands, each (x, y, half width, half height, depth)."""
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    total = np.zeros(shape)
    for cx, cy, wx, hy, depth in placed:
        u, v = (xs + 0.5 - cx) / wx, (ys + 0.5 - cy) / hy
        u2, v2 = u * u, v * v
        total += depth / ((1.0 + u2 * u2 * u2) * (1.0 + v2 * v2))
    return total


def row(y: float, amps, shape=(H, W), half_height: float = 4.0) -> np.ndarray:
    return bands(shape, [(x, y, 8.0, half_height, a) for x, a in zip(LANE_X, amps, strict=True)])


def as_type(values: np.ndarray, dtype=np.uint8) -> np.ndarray:
    info = np.iinfo(dtype)
    return np.clip(np.round(values), info.min, info.max).astype(dtype)


def stack(channels, seed: int, amplitude=(2, 2, 2), dtype=np.uint8) -> np.ndarray:
    return np.stack(
        [
            as_type(values + noise(values.shape, seed * 10 + c, amplitude[c]), dtype)
            for c, values in enumerate(channels)
        ],
        axis=-1,
    )


def flat(level: float = BLACK, shape=(H, W)) -> np.ndarray:
    return np.full(shape, level)


class Scene:
    """A synthetic image as a file holds it, and how it is imported."""

    def __init__(self, pixels, *, fmt="png", polarity=LIGHT, kind=CHEMI, palette=None):
        self.pixels, self.fmt, self.polarity, self.kind = pixels, fmt, polarity, kind
        self.palette = palette

    def file(self) -> tuple[str, bytes]:
        """The suffix and bytes of the file."""
        out = io.BytesIO()
        if self.fmt == "palette":
            image = Image.frombytes("P", (W, H), self.pixels.tobytes())
            image.putpalette(self.palette.reshape(-1).tolist())
            image.save(out, format="PNG")
            return ".png", out.getvalue()
        if self.fmt == "tif":
            tifffile.imwrite(out, self.pixels, photometric="rgb", metadata=None)
            return ".tif", out.getvalue()
        if self.fmt == "jpg":
            Image.fromarray(self.pixels).save(out, format="JPEG", quality=90)
            return ".jpg", out.getvalue()
        Image.fromarray(self.pixels).save(out, format="PNG")
        return ".png", out.getvalue()


def one_colour(colour: int, seed: int, *, polarity=LIGHT, amps=AMPS, crosstalk=0.0) -> Scene:
    field = row(ROW_A, amps)
    share = [1.0 if c == colour else crosstalk for c in range(3)]
    if polarity is LIGHT:
        return Scene(stack([BLACK + s * field for s in share], seed))
    return Scene(stack([WHITE - s * field for s in share], seed), polarity=DARK)


def two_targets(seed: int, red_amps, green_amps) -> Scene:
    a, b = row(ROW_A, red_amps), row(ROW_B, green_amps)
    return Scene(stack([BLACK + a, BLACK + b, flat()], seed))


def gray_in_colour(seed: int, tint, chroma: int) -> np.ndarray:
    gray = 210.0 - row(ROW_A, AMPS)
    shared = noise((H, W), seed * 10 + 7, 2)
    return np.stack(
        [
            as_type(gray * t + shared + noise((H, W), seed * 10 + c, chroma))
            for c, t in enumerate(tint)
        ],
        axis=-1,
    )


def palette(seed: int, kind: str) -> Scene:
    index = as_type(BLACK + row(ROW_A, AMPS) + noise((H, W), seed * 10, 2))
    i = np.arange(256, dtype=np.int64)
    if kind == "green":
        table = np.stack([0 * i, i, 0 * i], axis=-1)
    else:  # "hot": red rises first, then green, then blue
        table = np.stack(
            [np.minimum(255, 3 * i), np.clip(3 * i - 255, 0, 255), np.clip(3 * i - 510, 0, 255)],
            axis=-1,
        )
    return Scene(index, fmt="palette", palette=table.astype(np.uint8))


def ladder(x: float, depth: float) -> list[np.ndarray]:
    out = [np.zeros((H, W)) for _ in range(3)]
    for y, inks in zip(LADDER_Y, LADDER_INKS, strict=True):
        field = bands((H, W), [(x, y, 6.0, 2.5, depth)])
        for c in range(3):
            out[c] += inks[c] * field
    return out


def dust(seed: int, count: int, size: int, channels) -> Scene:
    rng = np.random.default_rng(2000 + seed)
    base = [flat(), BLACK + row(ROW_A, AMPS), flat()]
    for _ in range(count):
        y, x = int(rng.integers(0, H - size)), int(rng.integers(0, W - size))
        for c in channels:
            base[c][y : y + size, x : x + size] = 255.0
    return Scene(stack(base, seed))


def red_specks(seed: int) -> Scene:
    rng = np.random.default_rng(1000 + seed)
    red = flat()
    for _ in range(30):
        y, x = int(rng.integers(0, H - 3)), int(rng.integers(0, W - 3))
        red[y : y + 3, x : x + 3] = 255.0
    return Scene(stack([red, BLACK + row(ROW_A, AMPS), flat()], seed))


def saturated_16_bit(seed: int) -> Scene:
    field = row(ROW_A, SATURATING) * 257.0
    floor = BLACK * 257
    channels = [floor + 0.1 * field, floor + field, floor + 0.1 * field]
    return Scene(stack(channels, seed, amplitude=(500, 500, 500), dtype=np.uint16), fmt="tif")


def saturated_jpeg(seed: int) -> Scene:
    scene = one_colour(1, seed, amps=SATURATING, crosstalk=0.1)
    scene.fmt = "jpg"
    return scene


def thin(seed: int) -> Scene:
    field = row(ROW_A, AMPS, half_height=1.0)
    return Scene(stack([flat(), BLACK + field, flat()], seed))


def one_band(seed: int) -> Scene:
    shape = (400, 600)
    field = bands(shape, [(300.0, 200.0, 8.0, 4.0, 100.0)])
    return Scene(stack([flat(shape=shape), BLACK + field, flat(shape=shape)], seed))


def blue_ramp(seed: int) -> Scene:
    ramp = np.broadcast_to(np.linspace(0.0, 120.0, W), (H, W))
    return Scene(stack([flat(), BLACK + row(ROW_A, AMPS), BLACK + ramp], seed))


def noisy_blue(seed: int) -> Scene:
    level = flat(BLACK + 20.0)
    return Scene(stack([level - 20.0, BLACK + row(ROW_A, WEAK), level], seed, amplitude=(2, 2, 8)))


def yellow(seed: int) -> Scene:
    field = row(ROW_A, AMPS)
    return Scene(stack([BLACK + 0.8 * field, BLACK + field, flat()], seed))


def merged(seed: int) -> Scene:
    chemi = row(ROW_A, AMPS)
    marks = ladder(8.0, 150.0)
    channels = [230.0 - chemi - marks[c] for c in range(3)]
    return Scene(stack(channels, seed), fmt="jpg", polarity=DARK, kind=MERGED)


def edge_strip(seed: int) -> Scene:
    blue = flat()
    blue[:, :10] += 50.0
    return Scene(stack([flat(), BLACK + row(ROW_A, AMPS), blue], seed))


def rotated(order) -> object:
    def make(seed: int) -> Scene:
        scene = one_colour(1, seed)
        scene.pixels = scene.pixels[..., list(order)]
        return scene

    return make


def with_alpha(seed: int) -> Scene:
    rgb = one_colour(1, seed).pixels
    return Scene(np.concatenate([rgb, np.full((H, W, 1), 255, np.uint8)], axis=-1))


def gray_as_rgb(seed: int) -> Scene:
    gray = as_type(210.0 - row(ROW_A, AMPS) + noise((H, W), seed * 10, 2))
    return Scene(np.stack([gray, gray, gray], axis=-1), polarity=DARK)


def gray_alpha(seed: int) -> Scene:
    gray = as_type(210.0 - row(ROW_A, AMPS) + noise((H, W), seed * 10, 2))
    return Scene(np.stack([gray, np.full_like(gray, 255)], axis=-1), polarity=DARK)


def prestained(seed: int) -> Scene:
    marks = ladder(100.0, 150.0)
    channels = [235.0 - marks[c] for c in range(3)]
    return Scene(stack(channels, seed), fmt="jpg", polarity=DARK, kind=MARKER)


# What each image must pick, seed by seed (1 to 5): R, G, B or M (the mean).
SCENES = {
    "green on black": (lambda seed: one_colour(1, seed), "GGGGG"),
    "red on black": (lambda seed: one_colour(0, seed), "RRRRR"),
    "blue on black": (lambda seed: one_colour(2, seed), "BBBBB"),
    "dark green on white": (lambda seed: one_colour(1, seed, polarity=DARK), "GGGGG"),
    "dark blue on white": (lambda seed: one_colour(2, seed, polarity=DARK), "BBBBB"),
    "green saturated in lane 4": (
        lambda seed: one_colour(1, seed, amps=SATURATING, crosstalk=0.1),
        "GGGGG",
    ),
    "16-bit green saturated in lane 4": (saturated_16_bit, "GGGGG"),
    "JPEG green saturated in lane 4": (saturated_jpeg, "GGGGG"),
    # Two targets of one strength: whichever holds more, seed by seed.
    "red and green targets alike": (lambda seed: two_targets(seed, AMPS, AMPS), "RRGGR"),
    "green target, red at a half": (
        lambda seed: two_targets(seed, AMPS, tuple(2 * a for a in AMPS)),
        "GGGGG",
    ),
    "green target, red at a quarter": (
        lambda seed: two_targets(seed, tuple(0.5 * a for a in AMPS), tuple(2 * a for a in AMPS)),
        "GGGGG",
    ),
    "green target, red at a tenth": (
        lambda seed: two_targets(seed, tuple(0.2 * a for a in AMPS), tuple(2 * a for a in AMPS)),
        "GGGGG",
    ),
    "green palette": (lambda seed: palette(seed, "green"), "GGGGG"),
    "hot palette": (lambda seed: palette(seed, "hot"), "RRRRR"),  # a known limit: it saturates
    "green with alpha": (with_alpha, "GGGGG"),
    "blue ramp beside green bands": (blue_ramp, "GGGGG"),
    "red specks of 3 px": (red_specks, "GGGGG"),
    "red specks of 2 px": (lambda seed: dust(seed, 8, 2, (0,)), "GGGGG"),
    "white specks": (lambda seed: dust(seed, 30, 3, (0, 1, 2)), "GGGGG"),
    "weak green beside noisy blue": (noisy_blue, "GGGGG"),
    "yellow bands": (yellow, "GGGGG"),
    "noise only": (lambda seed: Scene(stack([flat(10.0), flat(14.0), flat(8.0)], seed)), "MMMMM"),
    "green on black, read dark on light": (
        lambda seed: Scene(one_colour(1, seed).pixels, polarity=DARK),
        "MMMMM",
    ),
    "faint green": (lambda seed: one_colour(1, seed, amps=FAINT), "GGGGG"),
    "one small band": (one_band, "GGGGG"),
    "green moved to red": (rotated((1, 2, 0)), "RRRRR"),
    "green moved to blue": (rotated((2, 0, 1)), "BBBBB"),
    "bright blue strip at an edge": (edge_strip, "GGGGG"),
    "thin green bands": (thin, "GGGGG"),
}
# Images whose three channels all hold their bands: a gray blot saved as a
# colour JPEG, tinted, and merged with a coloured ladder. These pin a choice
# still pending (the mean once every channel holds half the largest mass).
ALL_CARRY_SCENES = {
    "gray saved as a colour JPEG": (
        lambda seed: Scene(gray_in_colour(seed, (1.0, 1.0, 1.0), 1), fmt="jpg", polarity=DARK),
        "MMMMM",
    ),
    "tinted gray": (
        lambda seed: Scene(gray_in_colour(seed, (1.0, 0.96, 0.82), 2), polarity=DARK),
        "MMMMM",
    ),
    "merged with a coloured ladder": (merged, "MMMMM"),
}
LETTERS = {RED: "R", GREEN: "G", BLUE: "B", MEAN: "M"}


def painted(seed: int, polarity: Polarity, share: float, paint: int, fmt: str) -> Scene:
    """A gray blot exported in colour (three equal channels) whose lane 4 runs
    past the detector limit until ``share`` of its box's pixels sit at the
    limit, each of them then painted pure red, green or blue (``paint``: 0, 1
    or 2), as some imagers mark over-exposure."""
    light = polarity is LIGHT
    limit = 255 if light else 0
    shared = noise((H, W), seed * 10 + 7, 2)
    rest = row(ROW_A, [0.0 if lane == SATURATED_LANE else a for lane, a in enumerate(AMPS)])
    unit = row(ROW_A, [1.0 if lane == SATURATED_LANE else 0.0 for lane in LANES])
    x, y = int(LANE_X[SATURATED_LANE]), int(ROW_A)
    window = (slice(y - BOX.height // 2, y + BOX.height // 2), slice(x - 10, x + 10))
    wanted = max(1, round(share * BOX.width * BOX.height))

    def gray(depth: float) -> np.ndarray:
        field = rest + depth * unit
        return as_type((BLACK + field if light else WHITE - field) + shared)

    low, high = 100.0, 5000.0
    for _ in range(60):  # the band's depth that puts the wanted pixels at the limit
        middle = (low + high) / 2
        if np.count_nonzero(gray(middle)[window] == limit) >= wanted:
            high = middle
        else:
            low = middle
    plain = gray(high)
    rgb = np.stack([plain, plain, plain], axis=-1)
    rgb[plain == limit] = [255 if c == paint else 0 for c in range(3)]
    return Scene(rgb, fmt=fmt, polarity=polarity)


# Saturation painted in a colour: the three channels still hold the bands
# alike (a known limit: measured on the mean, its saturation goes unflagged).
PAINTED = {
    "PNG light on dark, a tenth painted red": lambda seed: painted(seed, LIGHT, 0.10, 0, "png"),
    "PNG dark on light, 1% painted blue": lambda seed: painted(seed, DARK, 0.01, 2, "png"),
    "JPEG light on dark, a tenth painted green": lambda seed: painted(seed, LIGHT, 0.1, 1, "jpg"),
    "JPEG dark on light, a tenth painted green": lambda seed: painted(seed, DARK, 0.1, 1, "jpg"),
}


def read_back(scene: Scene, folder: Path) -> np.ndarray:
    """The scene's pixels as an import reads its file."""
    suffix, data = scene.file()
    path = folder / f"scene{suffix}"
    path.write_bytes(data)
    return imaging.read_file(path).pixels


def evidence(scene: Scene, folder: Path):
    pixels = read_back(scene, folder)
    return imaging.channel_masses(
        pixels, imaging.bit_depth_of(pixels), dark_on_light=scene.polarity.dark_on_light
    )


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> dict:
    """Each scene's pixels as read and evidence, seed by seed, measured once."""
    folder = tmp_path_factory.mktemp("scenes")
    found = {}
    for name, (make, _) in {**SCENES, **ALL_CARRY_SCENES}.items():
        for seed in SEEDS:
            scene = make(seed)
            pixels = read_back(scene, folder)
            found[name, seed] = (
                scene,
                pixels,
                imaging.channel_masses(
                    pixels, imaging.bit_depth_of(pixels), dark_on_light=scene.polarity.dark_on_light
                ),
            )
    return found


# --- The rule on the images ---


def _picks(measured, scenes) -> dict[str, str]:
    from proteia.core import channelrule

    return {
        name: "".join(LETTERS[channelrule.pick(measured[name, seed][2][0])] for seed in SEEDS)
        for name in scenes
    }


def test_each_image_picks_the_channel_that_holds_its_bands(measured):
    assert _picks(measured, SCENES) == {name: must for name, (_, must) in SCENES.items()}


class TestWhenEveryChannelCarries:
    # These pin a choice still pending: the mean once every channel holds at
    # least half the largest band mass.

    def test_an_image_whose_three_channels_hold_its_bands_is_measured_on_their_mean(self, measured):
        assert _picks(measured, ALL_CARRY_SCENES) == {
            name: must for name, (_, must) in ALL_CARRY_SCENES.items()
        }

    @pytest.mark.parametrize("name", list(PAINTED))
    def test_an_export_with_its_saturation_painted_in_a_colour_is_measured_on_the_mean(
        self, tmp_path, name
    ):
        from proteia.core import channelrule

        picks = [channelrule.pick(evidence(PAINTED[name](seed), tmp_path)[0]) for seed in SEEDS]
        assert picks == [MEAN] * 5

    def test_the_weakest_channel_holds_half_the_largest_mass_or_more(self, measured):
        for name in ALL_CARRY_SCENES:
            for seed in SEEDS:
                masses = measured[name, seed][2][0]
                assert min(masses) >= 0.5 * max(masses) > 0

    def test_the_mean_at_half_and_one_channel_just_below_it(self):
        from proteia.core import channelrule

        assert channelrule.pick((1.0, 0.5, 0.5)) is MEAN
        assert channelrule.pick((0.5, 0.5, 1.0)) is MEAN
        assert channelrule.pick((1.0, 0.5, 0.4999)) is RED
        assert channelrule.pick((0.4999, 1.0, 0.9)) is GREEN

    def test_import_measures_a_tinted_gray_on_the_mean_and_says_so(self, tmp_path):
        s = session_on(tmp_path)
        image = import_scene(s, ALL_CARRY_SCENES["tinted gray"][0](1), "tinted µ.png")
        choice = s.project.batch.find_image(image).channel
        assert (choice.channel, choice.source) == (MEAN, PICKED)
        assert np.array_equal(s.pixels(image), read_pixels(s, image)[..., :3].mean(axis=-1))
        protein = boxes(s, image, "α-tubulin", ROW_A)
        texts = messages(s, "colour_channel")
        assert texts == [
            "'tinted µ.png' is measured on the mean of its red, green and blue, as ImageJ"
            " measures colour by default: all three channels hold its bands"
        ]
        assert state_channel(s, image)["words"] == (
            "Picked by Proteia: all three channels hold its bands."
        )
        # Lossless, so the exact check would run on one channel: said.
        [unchecked] = [n for n in notices(s) if n.code == "clipping_not_checked"]
        assert unchecked.protein_ids == (protein,)
        assert unchecked.message.endswith(
            "; choose the channel that holds its bands for its image to check it there"
        )


def test_the_masses_of_one_colour_lie_in_that_channel(measured):
    _, _, (masses, apart) = measured["green on black", 1]
    assert masses[0] == 0.0 and masses[2] == 0.0 and masses[1] > 0
    assert apart == (0.0, 0.0, 0.0)
    _, _, (masses, apart) = measured["noise only", 1]
    assert masses == (0.0, 0.0, 0.0) and apart == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("name", "others"),
    [
        ("red and green targets alike", True),
        ("green target, red at a half", True),
        ("green target, red at a quarter", True),
        ("bright blue strip at an edge", True),  # a known limit: a visible false alarm
        ("green target, red at a tenth", False),  # a known limit: too faint to be told
        ("green saturated in lane 4", False),  # crosstalk lies on the bands
        ("yellow bands", False),  # one target in two channels
        ("red specks of 3 px", False),
    ],
)
def test_signal_apart_from_the_pick_is_found_where_another_target_lies(measured, name, others):
    from proteia.core import channelrule

    for seed in SEEDS:
        masses, apart = measured[name, seed][2]
        choice = ChannelChoice(
            channel=channelrule.pick(masses), source=PICKED, rule=1, masses=masses, apart=apart
        )
        assert (len(channelrule.signal_channels(choice)) > 1) is others


def _range(u):
    return float(np.percentile(u, 99.9, method="higher") - np.median(u))


def _sigma(u):
    median = np.median(u)
    return max(1.4826 * float(np.median(np.abs(u - median))), 0.5)


def _signal_up(scene, pixels):
    top = 2 ** imaging.bit_depth_of(pixels) - 1
    rgb = pixels[..., :3].astype(np.int64)
    return [top - rgb[..., c] if scene.polarity is DARK else rgb[..., c] for c in range(3)], top


def _argmax(values):
    return "RGB"[int(np.argmax(values))]


def _widest(scene, pixels):
    return _argmax([_range(u) for u in _signal_up(scene, pixels)[0]])


def _clearest(scene, pixels):
    return _argmax([_range(u) / _sigma(u) for u in _signal_up(scene, pixels)[0]])


def _least_saturated(scene, pixels):
    us, top = _signal_up(scene, pixels)
    counts = [int(np.count_nonzero(u == top)) for u in us]
    tied = [c for c in range(3) if counts[c] == min(counts)]
    return "RGB"[max(tied, key=lambda c: (_range(us[c]), -c))]


def _spread(scene, pixels):
    return _argmax([float(u.std()) for u in _signal_up(scene, pixels)[0]])


@pytest.mark.parametrize("rule", [_widest, _clearest, _least_saturated, _spread])
def test_each_other_rule_weighed_picks_another_channel_somewhere(measured, rule):
    musts = {**SCENES, **ALL_CARRY_SCENES}
    wrong = [
        (name, seed)
        for name, (_, must) in musts.items()
        for seed, letter in zip(SEEDS, must, strict=True)
        if rule(*measured[name, seed][:2]) != letter
    ]
    assert wrong


@pytest.mark.parametrize(
    ("patch", "row"),
    [
        ({"MIN_REGION": 1}, "red specks of 3 px"),  # the region filter takes dust out
        ({"ALL_CARRY": 2.0}, "gray saved as a colour JPEG"),  # the mean where all carry
    ],
)
def test_the_rule_without_one_of_its_steps_picks_wrong(tmp_path, patch, row):
    from proteia.core import channelrule

    make, must = {**SCENES, **ALL_CARRY_SCENES}[row]
    with mock.patch.multiple(channelrule, **patch):
        got = "".join(
            LETTERS[channelrule.pick(evidence(make(seed), tmp_path)[0])] for seed in SEEDS
        )
    assert got != must


def test_the_mean_alone_loses_the_range_of_one_colour_and_its_saturation(measured):
    # Measuring every colour image on the mean (as before) keeps a third of a
    # single colour's signal, and a band saturated in green never brings the
    # mean to the limit; weighted luminance keeps a ninth of blue's.
    scene, pixels, _ = measured["green saturated in lane 4", 1]
    rgb = pixels[..., :3].astype(np.float64)
    x0, x1 = int(LANE_X[SATURATED_LANE]) - 10, int(LANE_X[SATURATED_LANE]) + 10
    lane = (slice(int(ROW_A) - 6, int(ROW_A) + 6), slice(x0, x1))
    assert rgb[..., 1][lane].max() == 255
    assert rgb.mean(axis=-1)[lane].max() < 255
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    assert luminance[lane].max() < 255
    _, blue, _ = measured["blue on black", 1]
    weighted = 0.299 * blue[..., 0] + 0.587 * blue[..., 1] + 0.114 * blue[..., 2]
    assert np.ptp(weighted) < np.ptp(blue[..., 2].astype(np.float64)) / 4


# --- The masses, worked out by hand ---


def _square(top_left: tuple[int, int], side: int, value: int, shape=(20, 20)) -> np.ndarray:
    """RGB of zeros with a ``side`` square of ``value`` in green."""
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    y, x = top_left
    rgb[y : y + side, x : x + side, 1] = value
    return rgb


def test_a_region_of_sixteen_pixels_counts_and_one_of_fifteen_does_not():
    # A 4 x 4 square of 100 in the first rows of a 20 x 20 image of zeros: the
    # median and its deviation are 0, so sigma is 0.5 and the cut 1.5; its mass
    # is 16 pixels 98.5 above the cut, over the image's 400 pixels.
    masses, apart = imaging.channel_masses(_square((0, 0), 4, 100), 8, dark_on_light=False)
    assert masses == (0.0, 16 * 98.5 / 400, 0.0) and apart == (0.0, 0.0, 0.0)
    smaller = _square((0, 0), 4, 100)
    smaller[3, 3, 1] = 0  # 15 pixels: dust, not a band
    assert imaging.channel_masses(smaller, 8, dark_on_light=False)[0] == (0.0, 0.0, 0.0)
    # Dark bands on a light background: the same image upside down in value.
    mirrored = imaging.channel_masses(255 - _square((0, 0), 4, 100), 8, dark_on_light=True)
    assert mirrored == (masses, apart)


@pytest.mark.parametrize("width", [33, 34])  # an odd and an even pixel count
def test_the_cut_is_the_median_plus_three_robust_sigmas_exactly(width):
    # One row: the background values 0, 1, ..., width - 17 and a line of 16
    # pixels of 100 (8-connected). The median is the middle value (odd count)
    # or the mean of the two middle values (even); so is the median deviation:
    # 33 pixels, median 16, deviations 0 to 16 then 84s: 16; 34 pixels, median
    # 16.5, deviations 0.5, 0.5, 1.5, ..., 16.5 then 83.5s: (15.5 + 16.5) / 2.
    values = list(range(width - 16)) + [100] * 16
    rgb = np.zeros((1, width, 3), dtype=np.uint8)
    rgb[0, :, 0] = values
    median, deviation = (16.0, 16.0) if width == 33 else (16.5, 16.0)
    cut = median + 3.0 * (1.4826 * deviation)
    masses, _ = imaging.channel_masses(rgb, 8, dark_on_light=False)
    assert masses == ((16 * 100 - 16 * cut) / width, 0.0, 0.0)


def test_the_masses_are_counted_alike_in_bands_of_rows(monkeypatch):
    # The counts run a band of rows at a time, for memory: any band size,
    # down to one row, gives the same masses and shares, on a wide image and
    # on a tall one (the same turned a quarter).
    wide = two_targets(2, AMPS, tuple(2 * a for a in AMPS)).pixels
    tall = np.ascontiguousarray(np.rot90(wide))
    whole = [imaging.channel_masses(p, 8, dark_on_light=False) for p in (wide, tall)]
    monkeypatch.setattr(imaging, "_PASS_PIXELS", 7)
    assert [imaging.channel_masses(p, 8, dark_on_light=False) for p in (wide, tall)] == whole
    assert whole[0] == whole[1]  # turning the image changes no region
    assert whole[0][1][0] > 0.4  # red's target lies apart from green's


# --- The rule's decision and words ---


def test_the_rule_picks_the_largest_mass_red_first_on_a_tie():
    from proteia.core import channelrule

    assert channelrule.pick((0.0, 0.0, 0.0)) is MEAN
    assert channelrule.pick((0.0, 1e-9, 0.0)) is GREEN
    assert channelrule.pick((2.0, 2.0, 0.0)) is RED
    assert channelrule.pick((0.0, 2.0, 2.0)) is GREEN
    assert channelrule.pick((1.0, 0.0, 3.0)) is BLUE
    assert channelrule.pick([0.2, 0.0, 3.0]) is BLUE  # any sequence of three


def _choice(channel, source=PICKED, masses=(0.0, 2.0, 0.0), apart=(0.0, 0.0, 0.0)):
    return ChannelChoice(channel=channel, source=source, rule=1, masses=masses, apart=apart)


def test_signal_channels_are_the_pick_and_those_apart_from_it_from_a_tenth():
    from proteia.core import channelrule

    assert channelrule.signal_channels(None) == ()
    assert channelrule.signal_channels(ChannelChoice(channel=GREEN, source=CHOSEN)) == ()
    assert channelrule.signal_channels(_choice(GREEN)) == (GREEN,)
    assert channelrule.signal_channels(_choice(GREEN, apart=(0.1, 0.0, 0.0999))) == (RED, GREEN)
    assert channelrule.signal_channels(_choice(GREEN, apart=(0.0999, 0.0, 0.5))) == (GREEN, BLUE)
    # The rule's pick, whatever was chosen since; none for a mean pick.
    chosen = _choice(RED, CHOSEN, masses=(1.0, 3.0, 0.0), apart=(0.4, 0.0, 0.0))
    assert channelrule.signal_channels(chosen) == (RED, GREEN)
    assert channelrule.signal_channels(_choice(MEAN, masses=(0.0, 0.0, 0.0))) == ()


def test_the_page_line_says_who_chose_the_channel_and_why():
    from proteia.core import channelrule

    picked = "Picked by Proteia: it holds the most signal above its background"
    assert channelrule.line(_choice(GREEN, masses=(0.06, 2.0, 0.0201))) == (
        f"{picked} (red 3%, blue 1% of it)."
    )
    # Whole percentages, half up (12.5% is 13%), never 100% short of all.
    assert channelrule.line(_choice(GREEN, masses=(0.25, 2.0, 0.0))) == (
        f"{picked} (red 13%, blue 0% of it)."
    )
    assert channelrule.line(_choice(BLUE, masses=(0.996, 0.0049, 1.0))) == (
        f"{picked} (red 99%, green 0% of it)."
    )
    assert channelrule.line(_choice(RED, masses=(2.0, 2.0, 0.0))) == (
        f"{picked} (green 100%, blue 0% of it)."
    )
    assert channelrule.line(_choice(MEAN, masses=(1.0, 0.9, 0.8))) == (
        "Picked by Proteia: all three channels hold its bands."
    )
    assert channelrule.line(_choice(MEAN, masses=(0.0, 0.0, 0.0))) == (
        "Picked by Proteia: no channel held bands above its background. Choose the channel that"
        " holds its bands to measure it there (its boxes are measured again)."
    )
    assert channelrule.line(_choice(RED, CHOSEN)) == "Chosen by you."
    assert channelrule.line(ChannelChoice(channel=MEAN, source=CHOSEN)) == "Chosen by you."
    assert channelrule.line(None) == (
        "No channel chosen: measured on the mean of its red, green and blue. Choose the channel"
        " that holds its bands to measure it there (its boxes are measured again)."
    )


IMAGEJ = " To compare with ImageJ, measure the same channel (Image > Color > Split Channels)"
CHOOSE = "; choose the channel that holds its bands for its image to check it there"


def test_the_note_says_what_an_image_is_measured_on():
    from proteia.core import channelrule

    name = "'GFP α.png'"
    assert channelrule.note(_choice(GREEN), name) == (
        "'GFP α.png' is measured on its green channel, which Proteia picked: it holds the most"
        " signal above its background of the three." + IMAGEJ
    )
    assert channelrule.note(_choice(RED, CHOSEN), name) == (
        "'GFP α.png' is measured on its red channel, as chosen." + IMAGEJ
    )
    assert channelrule.note(ChannelChoice(channel=BLUE, source=CHOSEN), name) == (
        "'GFP α.png' is measured on its blue channel, as chosen." + IMAGEJ
    )
    assert channelrule.note(_choice(MEAN, CHOSEN), name) == (
        "'GFP α.png' is measured on the mean of its red, green and blue, as chosen (ImageJ"
        " measures colour on the mean by default)"
    )
    assert channelrule.note(_choice(MEAN, masses=(1.0, 0.9, 0.8)), name) == (
        "'GFP α.png' is measured on the mean of its red, green and blue, as ImageJ measures"
        " colour by default: all three channels hold its bands"
    )
    assert channelrule.note(_choice(MEAN, masses=(0.0, 0.0, 0.0)), name) == (
        "'GFP α.png' is measured on the mean of its red, green and blue: no channel held bands"
        " above its background when Proteia picked its channel; if its bands are of one colour,"
        " choose that channel for it under Images"
    )
    assert channelrule.note(None, name) == (
        "'GFP α.png' is measured on the mean of its red, green and blue, as Proteia measured"
        " colour images before it chose a channel; choose a channel for it under Images to"
        " measure it there (its boxes are measured again)"
    )


class TestASecondColourOnACopy:
    # These pin a choice still pending: a second colour of a two-colour file
    # is measured on a copy of the image, imported again into the project.

    def test_the_warning_names_the_channel_and_the_way_to_measure_it(self):
        from proteia.core import channelrule

        assert channelrule.elsewhere("'two µ.png'", [RED]) == (
            "'two µ.png' holds signal in its red channel apart from the bands it is measured on:"
            " if that is another protein, measure it on its own: import the same file again and"
            " choose red for the new image"
        )
        assert channelrule.elsewhere("'two µ.png'", [RED, BLUE]) == (
            "'two µ.png' holds signal in its red and blue channels apart from the bands it is"
            " measured on: if those are other proteins, measure each on its own: import the same"
            " file again for each and choose its channel for the new image"
        )

    def test_a_copy_measured_on_the_other_colour_measures_each_target_on_its_own(self, tmp_path):
        s = session_on(tmp_path)
        scene = two_targets(1, AMPS, tuple(2 * a for a in AMPS))  # green twice red
        first = import_scene(s, scene, "two µ.png")
        a = boxes(s, first, "β-catenin", ROW_B)  # the green target
        warned = [n for n in notices(s) if n.code == "channel_signal_elsewhere"]
        assert [(n.message, n.image_ids, n.protein_ids, n.level) for n in warned] == [
            (
                "'two µ.png' holds signal in its red channel apart from the bands it is measured"
                " on: if that is another protein, measure it on its own: import the same file"
                " again and choose red for the new image",
                (first,),
                (a,),
                results.Level.WARNING,
            )
        ]
        # As the warning says: the same file again, onto the same membrane,
        # measured on red, and the second protein on it.
        membrane = s.project.batch.membranes_on(first)[0].id
        copy = import_scene(s, scene, "two µ.png", membrane_id=membrane)
        assert stored(s, copy).measured_channel is GREEN  # picked as the first was
        ops.set_channel(s, copy, RED)
        b = boxes(s, copy, "p53 µ", ROW_A)
        assert [n.code for n in notices(s) if n.code == "channel_signal_elsewhere"] == []
        # Each target measured on its own colour, lane by lane in one table.
        pixels = read_pixels(s, first)
        assert nets(s, a) == expected(s, first, pixels[..., 1], a)
        assert nets(s, b) == expected(s, copy, pixels[..., 0], b)
        assert all(net > 1000 for net in nets(s, a) + nets(s, b))
        # The note that they are the same file stays, first; then each image's.
        computed = results.compute_results(s.project.batch)
        codes = [n.code for n in computed.notices][:3]
        assert codes == ["duplicate_images", "colour_channel", "colour_channel"]
        # Choosing green on the copy again leaves red unmeasured: said again.
        ops.set_channel(s, copy, GREEN)
        assert len([n for n in notices(s) if n.code == "channel_signal_elsewhere"]) == 2


# --- Import ---


def import_scene(
    s: ops.ProjectSession, scene: Scene, name: str, *, membrane_id: str | None = None
) -> str:
    suffix, data = scene.file()
    assert name.endswith(suffix)
    return ops.import_image(
        s,
        io.BytesIO(data),
        name,
        kind=scene.kind,
        polarity=scene.polarity,
        membrane_id=membrane_id,
    )


def stored(s: ops.ProjectSession, image_id: str):
    return s.project.batch.find_image(image_id)


def read_pixels(s: ops.ProjectSession, image_id: str) -> np.ndarray:
    return imaging.read_file(storage.image_path(s.folder, stored(s, image_id))).pixels


def boxes(s, image_id: str, name: str, y: float, lanes=LANES) -> str:
    """A protein on the image with a fixed box on each lane's band at ``y``."""
    if not s.project.batch.only_lanes():
        ops.set_lanes(s, [LaneInput(f"c{i} µ", f"s{i}") for i in range(6)])
    protein = ops.add_protein(s, name, Role.TARGET, image_id, box_size=BOX)
    for lane in lanes:
        ops.place_box(s, protein, int(LANE_X[lane]), int(y), lane_index=lane, grow=False)
    return protein


def nets(s, protein_id: str) -> list[float]:
    return [
        band.net
        for band in s.project.batch.boxes_of(s.project.batch.find_protein(protein_id)).bands
    ]


def expected(s, image_id: str, plane: np.ndarray, protein_id: str) -> list[float]:
    """The nets of ``protein_id`` measured on ``plane`` with every box of the
    image (quantify_nets, the product's one entry point)."""
    batch = s.project.batch
    placed = [
        (p.id, batch.boxes_of(p).box_size, band)
        for p in batch.proteins_on(image_id)
        for band in batch.boxes_of(p).bands
    ]
    found = quantify_nets(
        plane.astype(np.float64),
        [band.box.rect(size) for _, size, band in placed],
        [(size.width, size.height) for _, size, _ in placed],
        method="ring_median",
        dark_on_light=stored(s, image_id).polarity.dark_on_light,
        integral=True,
    )
    return [net for (owner, _, _), net in zip(placed, found, strict=True) if owner == protein_id]


def notices(s) -> list:
    return list(results.compute_results(s.project.batch).notices)


def messages(s, code: str) -> list[str]:
    return [n.message for n in notices(s) if n.code == code]


def state_channel(s, image_id: str):
    state = project_state("p", s, s.project, open_id=1)
    return next(i for i in state["images"] if i["id"] == image_id)["channel"]


def test_import_picks_the_channel_that_holds_the_bands(tmp_path):
    s = session_on(tmp_path)
    scene = one_colour(1, 2)
    image = import_scene(s, scene, "GFP α.png")
    found = stored(s, image)
    choice = found.channel
    assert (choice.channel, choice.source, choice.rule) == (GREEN, PICKED, 1)
    pixels = read_pixels(s, image)
    masses, apart = imaging.channel_masses(pixels, 8, dark_on_light=False)
    assert (choice.masses, choice.apart) == (masses, apart)
    assert masses[0] == masses[2] == 0.0 < masses[1]
    # Measured on green: the analysis array, its median and its trust.
    green = pixels[..., 1].astype(np.float64)
    assert np.array_equal(s.pixels(image), green)
    assert found.background == float(np.median(green))
    assert found.over_exposure_check == "exact"
    # The import logs the choice as stored; the warnings as the page shows them.
    params = s.project.log[-1].params
    assert params["channel"] == {
        "channel": "green",
        "source": "picked",
        "rule": 1,
        "masses": list(masses),
        "apart": [0.0, 0.0, 0.0],
    }
    assert [w.message for w in found.shown_warnings] == [
        "The red, green and blue channels differ; the image is measured on its green channel."
    ]


def test_the_import_log_says_the_channel_measured(tmp_path, caplog):
    s = session_on(tmp_path)
    with caplog.at_level("INFO", logger="proteia.core.operations"):
        image = import_scene(s, one_colour(1, 1), "GFP α.png")
    said = [r.getMessage() for r in caplog.records if image in r.getMessage()]
    assert said == [
        f"in {s.folder.name!r}: {image} was imported with a warning, color_channels_differ: The"
        " red, green and blue channels differ; the image is measured on its green channel."
    ]


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (gray_as_rgb, "gray as RGB µ.png"),  # its channels are equal
        (gray_alpha, "gray α.png"),  # gray and alpha
        (prestained, "marker β.jpg"),  # a visible-light marker
    ],
)
def test_import_picks_nothing_without_a_channel_to_pick(tmp_path, make, name):
    s = session_on(tmp_path)
    image = import_scene(s, make(1), name)
    assert stored(s, image).channel is None
    assert "channel" not in s.project.log[-1].params


def test_a_compressed_rgba_tiff_is_picked_on_its_colours_alone(tmp_path):
    # An LZW RGB TIFF with an alpha sample, as an image editor saves one: read
    # (#232), and the rule reads red, green and blue, never the alpha.
    s = session_on(tmp_path)
    rgba = with_alpha(1).pixels
    path = tmp_path / "GFP α.tif"
    Image.fromarray(rgba, "RGBA").save(path, compression="tiff_lzw")
    with path.open("rb") as f:
        image = ops.import_image(s, f, "GFP α.tif", kind=CHEMI, polarity=LIGHT)
    choice = stored(s, image).channel
    assert (choice.channel, choice.source) == (GREEN, PICKED)
    assert choice.masses == imaging.channel_masses(rgba[..., :3], 8, dark_on_light=False)[0]


def test_import_measures_the_mean_where_no_channel_holds_bands(tmp_path):
    s = session_on(tmp_path)
    scene = SCENES["green on black, read dark on light"][0](1)
    image = import_scene(s, scene, "GFP α.png")
    choice = stored(s, image).channel
    assert (choice.channel, choice.source) == (MEAN, PICKED)
    assert choice.masses == (0.0, 0.0, 0.0) and choice.apart == (0.0, 0.0, 0.0)
    pixels = read_pixels(s, image)
    assert np.array_equal(s.pixels(image), pixels.mean(axis=-1))


def test_a_colour_jpeg_is_assessed_near_the_limit_of_the_channel_picked(tmp_path):
    s = session_on(tmp_path)
    image = import_scene(s, saturated_jpeg(1), "GFP α.jpg")
    assert stored(s, image).measured_channel is GREEN
    assert stored(s, image).over_exposure_check == "near_limit"
    protein = boxes(s, image, "GFP α", ROW_A)
    flagged = [band.possibly_clipped for band in protein_bands(s, protein)]
    assert flagged == [False, False, False, True, False, False]


def protein_bands(s, protein_id: str) -> list:
    batch = s.project.batch
    return list(batch.boxes_of(batch.find_protein(protein_id)).bands)


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (lambda seed: one_colour(1, seed, amps=SATURATING, crosstalk=0.1), "GFP α.png"),
        (saturated_16_bit, "GFP α.tif"),
    ],
)
def test_a_band_saturated_in_the_channel_picked_is_flagged_and_named(tmp_path, make, name):
    s = session_on(tmp_path)
    image = import_scene(s, make(1), name)
    assert stored(s, image).measured_channel is GREEN
    protein = boxes(s, image, "GFP α", ROW_A)
    assert [band.clipped for band in protein_bands(s, protein)] == [
        False, False, False, True, False, False
    ]  # fmt: skip
    [clipped] = [n for n in notices(s) if n.code == "clipped"]
    assert clipped.message == (
        "'GFP α' is over-exposed in lane 4: pixels at the detector limit of its image's green"
        " channel make its net an under-estimate, and a saturated band next to it may have"
        " merged into the same box, which then holds both bands and its net may be too high;"
        " shorten the exposure and image the membrane again; the lane stays included"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a known limit: an export that paints its saturated pixels in a colour is measured on"
        " the mean, where a painted pixel is a third of the limit, so the lane is not flagged;"
        " measured on a channel that keeps the saturated pixels at the limit, it is"
    ),
)
@pytest.mark.parametrize(
    "name", ["JPEG light on dark, a tenth painted green", "PNG dark on light, 1% painted blue"]
)
def test_a_lane_whose_saturated_pixels_the_imager_painted_in_a_colour_is_flagged(tmp_path, name):
    flagged = []
    for seed in SEEDS:
        s = session_on(tmp_path / str(seed))
        scene = PAINTED[name](seed)
        image = import_scene(s, scene, f"painted µ.{scene.fmt}")
        protein = boxes(s, image, "β-actin", ROW_A)
        band = protein_bands(s, protein)[SATURATED_LANE]
        flagged.append(bool(band.clipped or band.possibly_clipped))
    assert flagged == [True] * 5


@pytest.mark.parametrize(
    ("name", "kept"),
    [
        ("JPEG light on dark, a tenth painted green", GREEN),  # painted at 255 in green
        ("PNG dark on light, 1% painted blue", RED),  # red stays at 0 under blue paint
    ],
)
def test_measured_on_a_channel_that_keeps_painted_saturation_the_lane_is_flagged(
    tmp_path, name, kept
):
    # What the clipping_not_checked notice of such an image says to do (on a
    # lossless file) or the user may: choose the channel that keeps them.
    for seed in SEEDS:
        s = session_on(tmp_path / str(seed))
        scene = PAINTED[name](seed)
        image = import_scene(s, scene, f"painted µ.{scene.fmt}")
        protein = boxes(s, image, "β-actin", ROW_A)
        ops.set_channel(s, image, kept)
        bands = protein_bands(s, protein)
        assert [bool(b.clipped or b.possibly_clipped) for b in bands] == [
            False, False, False, True, False, False
        ]  # fmt: skip


def test_the_clipped_notice_of_a_gray_image_is_as_before():
    derived = json.loads((DATA / "f1" / "derived.json").read_text(encoding="utf-8"))
    [clipped] = [n for n in derived["results"]["notices"] if n["code"] == "clipped"]
    assert clipped["message"].startswith(
        "'GAPDH' is over-exposed in lane 4: pixels at the detector limit make its net an"
        " under-estimate"
    )


def test_import_runs_the_rule_before_the_analysis_array(tmp_path, monkeypatch):
    # For memory: the rule's passes never coincide with the analysis array.
    s = session_on(tmp_path)
    calls = []
    real_masses, real_load = ops.channel_masses, ops.load_pixels
    monkeypatch.setattr(
        ops, "channel_masses", lambda *a, **k: calls.append("rule") or real_masses(*a, **k)
    )
    monkeypatch.setattr(
        ops, "load_pixels", lambda *a, **k: calls.append("array") or real_load(*a, **k)
    )
    import_scene(s, one_colour(1, 1), "GFP α.png")
    assert calls == ["rule", "array"]


def test_import_of_a_colour_image_runs_out_of_memory_in_the_rule_as_in_any_measuring(
    tmp_path, monkeypatch
):
    s = session_on(tmp_path)

    def exhausted(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(ops, "channel_masses", exhausted)
    with pytest.raises(OperationError) as refused:
        import_scene(s, one_colour(1, 1), "GFP α.png")
    assert refused.value.code is ErrorCode.UNREADABLE_IMAGE
    assert "ran out of memory" in str(refused.value)
    assert list(s.project.batch.iter_images()) == []


# --- A polarity change ---


def test_a_polarity_change_picks_the_channel_again(tmp_path):
    s = session_on(tmp_path)
    scene = SCENES["green on black, read dark on light"][0](3)  # the wrong polarity
    image = import_scene(s, scene, "GFP α.png")
    assert stored(s, image).measured_channel is MEAN
    protein = boxes(s, image, "GFP α", ROW_A)
    ops.set_polarity(s, image, LIGHT)
    found = stored(s, image)
    pixels = read_pixels(s, image)
    masses, apart = imaging.channel_masses(pixels, 8, dark_on_light=False)
    assert found.channel == ChannelChoice(
        channel=GREEN, source=PICKED, rule=1, masses=masses, apart=apart
    )
    green = pixels[..., 1].astype(np.float64)
    assert found.background == float(np.median(green))
    assert np.array_equal(s.pixels(image), green)  # the cache follows the commit
    assert nets(s, protein) == expected(s, image, green, protein)
    assert s.project.log[-1].params["channel"] == found.channel.model_dump(mode="json")
    # Undone: the mean again, and a box moved then is measured on it.
    ops.undo(s)
    assert stored(s, image).measured_channel is MEAN
    band = protein_bands(s, protein)[0]
    x0, y0, x1, y1 = band.box.rect(BOX)
    ops.move_box(s, band.id, (x0 + 1, y0, x1 + 1, y1))
    mean = pixels.mean(axis=-1)
    assert nets(s, protein) == expected(s, image, mean, protein)


def _counting_reads(monkeypatch) -> list:
    """Every read of a stored file's pixels from now on, by any reader."""
    import proteia.core.session as session_module

    reads = []
    real = imaging.read_file

    def counted(path):
        reads.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(imaging, "read_file", counted)
    monkeypatch.setattr(session_module, "read_file", counted)
    return reads


def test_a_polarity_change_reads_the_file_once_and_keeps_the_array_it_made(tmp_path, monkeypatch):
    s = session_on(tmp_path)
    scene = SCENES["green on black, read dark on light"][0](1)
    image = import_scene(s, scene, "GFP α.png")
    other = import_scene(s, one_colour(1, 2), "GFP β.png")  # no band: nothing to keep
    ops.set_channel(s, other, RED)  # chosen: it stays
    protein = boxes(s, image, "GFP α", ROW_A)
    reads = _counting_reads(monkeypatch)
    ops.set_polarity(s, image, LIGHT)  # picks green again: measured on it, cached
    ops.set_polarity(s, other, DARK)  # the same channel: read once, nothing cached
    assert stored(s, image).measured_channel is GREEN
    assert stored(s, other).measured_channel is RED
    green = s.pixels(image)
    assert len(reads) == 2  # each change read its file once; the cache was the change's
    assert not green.flags.writeable
    assert not s.file_pixels(image).flags.writeable
    assert nets(s, protein) == expected(s, image, np.asarray(green), protein)


def test_a_float_colour_image_imported_before_224_is_chosen_without_evidence(tmp_path):
    # Pixels of no known bit depth have no range to measure the masses in: the
    # import stored no choice then, and a choice now holds no evidence.
    s = session_on(tmp_path)
    rgb = one_colour(1, 1).pixels.astype(np.float32)
    path = tmp_path / "float α.tif"
    tifffile.imwrite(path, rgb, photometric="rgb", metadata=None)
    with (
        mock.patch.object(ops, "_refuse_pixel_type", lambda name, pixels: None),
        path.open("rb") as f,
    ):
        image = ops.import_image(s, f, "float α.tif", kind=CHEMI, polarity=LIGHT)
    assert stored(s, image).channel is None and stored(s, image).bit_depth is None
    protein = boxes(s, image, "GFP α", ROW_A)
    ops.set_channel(s, image, GREEN)
    assert stored(s, image).channel == ChannelChoice(channel=GREEN, source=CHOSEN)
    assert nets(s, protein) == expected(s, image, rgb[..., 1], protein)
    ops.set_polarity(s, image, DARK)  # nothing to measure again
    assert stored(s, image).channel == ChannelChoice(channel=GREEN, source=CHOSEN)


def test_a_polarity_change_keeps_a_chosen_channel_and_measures_its_evidence_again(tmp_path):
    s = session_on(tmp_path)
    image = import_scene(s, one_colour(1, 1), "GFP α.png")
    ops.set_channel(s, image, RED)
    before = stored(s, image).channel
    ops.set_polarity(s, image, DARK)
    after = stored(s, image).channel
    assert (after.channel, after.source) == (RED, CHOSEN)
    masses, apart = imaging.channel_masses(read_pixels(s, image), 8, dark_on_light=True)
    assert (after.masses, after.apart) == (masses, apart) != (before.masses, before.apart)


def test_a_polarity_change_of_a_picked_image_with_its_file_gone_is_refused(tmp_path):
    s = session_on(tmp_path)
    image = import_scene(s, one_colour(1, 1), "GFP α.png")
    path = storage.image_path(s.folder, stored(s, image))
    data = path.read_bytes()
    path.unlink()
    before = s.project
    with pytest.raises(OperationError) as refused:
        ops.set_polarity(s, image, DARK)
    assert refused.value.code is ErrorCode.IMAGE_FILE_CHANGED
    assert s.project is before
    path.write_bytes(data)  # as the refusal's next step: the file back
    ops.set_polarity(s, image, DARK)
    assert stored(s, image).measured_channel is MEAN  # nothing above the background dark on light


def test_an_older_colour_image_changes_polarity_with_its_file_gone(tmp_path):
    # No choice to measure again (imported before Proteia picked a channel):
    # as before, the polarity changes with no band on it.
    s = session_on(tmp_path)
    with mock.patch.object(ops, "_picked", lambda read, kind, polarity: None):
        image = import_scene(s, one_colour(1, 1), "GFP α.png")
    storage.image_path(s.folder, stored(s, image)).unlink()
    ops.set_polarity(s, image, DARK)
    assert stored(s, image).polarity is DARK and stored(s, image).channel is None


def test_a_polarity_re_pick_measures_a_drawn_area_on_the_new_channel(tmp_path):
    s = session_on(tmp_path)
    scene = SCENES["green on black, read dark on light"][0](2)
    image = import_scene(s, scene, "GFP α.png")
    [(_, whole)] = s.project.batch.arenas_on(image)
    [arena] = ops.edit_arena(s, whole.id, outline=(5, 20, 195, 70)).arena_ids
    protein = boxes(s, image, "GFP α", ROW_A)
    ops.set_polarity(s, image, LIGHT)
    assert stored(s, image).measured_channel is GREEN
    green = read_pixels(s, image)[..., 1].astype(np.float64)
    # As the same boxes measure on the green channel saved gray, area and all.
    gray = session_on(tmp_path / "gray")
    source = tmp_path / "green.tif"
    tifffile.imwrite(source, green.astype(np.uint8), photometric="minisblack", metadata=None)
    with source.open("rb") as f:
        twin = ops.import_image(gray, f, "green µ.tif", kind=CHEMI, polarity=LIGHT)
    [(_, twin_whole)] = gray.project.batch.arenas_on(twin)
    ops.edit_arena(gray, twin_whole.id, outline=(5, 20, 195, 70))
    same = boxes(gray, twin, "GFP α", ROW_A)
    assert nets(s, protein) == nets(gray, same)
    assert s.project.batch.find_arena(arena)[1].outline is not None


def test_a_polarity_re_pick_under_the_legacy_method_takes_the_new_channel_s_median(tmp_path):
    s = session_on(tmp_path)
    scene = SCENES["green on black, read dark on light"][0](4)
    image = import_scene(s, scene, "GFP α.png")
    protein = boxes(s, image, "GFP α", ROW_A)
    s._commit(as_legacy(s.project), action="plant", params={})
    ops.set_polarity(s, image, LIGHT)
    green = read_pixels(s, image)[..., 1].astype(np.float64)
    levels = {band.background_level for band in protein_bands(s, protein)}
    assert levels == {float(np.median(green))}


def test_requantify_never_picks_again(tmp_path):
    s = session_on(tmp_path)
    image = import_scene(s, one_colour(1, 1), "GFP α.png")
    boxes(s, image, "GFP α", ROW_A)
    ops.set_channel(s, image, BLUE)
    s._commit(as_legacy(s.project), action="plant", params={})
    ops.requantify(s)
    assert (stored(s, image).channel.channel, stored(s, image).channel.source) == (BLUE, CHOSEN)


# --- Choosing a channel ---


def test_choosing_the_channel_picked_changes_nothing(tmp_path):
    s = session_on(tmp_path)
    image = import_scene(s, one_colour(1, 1), "GFP α.png")
    before = s.project
    ops.set_channel(s, image, GREEN)
    assert s.project is before


def test_choosing_a_channel_keeps_the_evidence_of_the_pick(tmp_path):
    s = session_on(tmp_path)
    image = import_scene(s, two_targets(2, AMPS, tuple(2 * a for a in AMPS)), "two µ.png")
    picked = stored(s, image).channel
    reads = []
    real = imaging.read_file
    with mock.patch.object(imaging, "read_file", lambda path: reads.append(path) or real(path)):
        ops.set_channel(s, image, RED)
    chosen = stored(s, image).channel
    assert chosen == ChannelChoice(
        channel=RED, source=CHOSEN, rule=1, masses=picked.masses, apart=picked.apart
    )
    assert len(reads) == 1


def test_choosing_a_channel_for_an_older_image_measures_its_evidence_on_one_read(
    tmp_path, monkeypatch
):
    s = session_on(tmp_path)
    with mock.patch.object(ops, "_picked", lambda read, kind, polarity: None):
        image = import_scene(s, two_targets(2, AMPS, tuple(2 * a for a in AMPS)), "two µ.png")
    protein = boxes(s, image, "β-catenin", ROW_B)
    reads = []
    real = imaging.read_file
    monkeypatch.setattr(imaging, "read_file", lambda path: reads.append(path) or real(path))
    import proteia.core.session as session_module

    monkeypatch.setattr(session_module, "read_file", imaging.read_file)
    ops.set_channel(s, image, GREEN)
    assert len(reads) == 1
    monkeypatch.undo()
    pixels = read_pixels(s, image)
    masses, apart = imaging.channel_masses(pixels, 8, dark_on_light=False)
    assert stored(s, image).channel == ChannelChoice(
        channel=GREEN, source=CHOSEN, rule=1, masses=masses, apart=apart
    )
    assert nets(s, protein) == expected(s, image, pixels[..., 1], protein)
    # So the results can tell that red holds another target.
    assert [n.code for n in notices(s)][:2] == ["colour_channel", "channel_signal_elsewhere"]


def test_a_choice_saved_before_the_rule_measured_any_opens_as_it_was(tmp_path):
    # A project saved by a build that stored the user's channel alone: it opens
    # and saves byte for byte; its note says it was chosen, and nothing can
    # tell where else signal lies; choosing again measures the evidence.
    s = session_on(tmp_path, save_to_folder)
    with mock.patch.object(ops, "_picked", lambda read, kind, polarity: None):
        image = import_scene(s, two_targets(2, AMPS, tuple(2 * a for a in AMPS)), "two µ.png")
    boxes(s, image, "β-catenin", ROW_B)

    def plain_choice(draft):
        draft.batch.find_image(image).channel = ChannelChoice(channel=GREEN, source=CHOSEN)

    project, _ = apply_change(s.project, plain_choice)
    s._commit(project, action="plant", params={})
    s.save()
    data = (s.folder / storage.PROJECT_FILE).read_bytes()
    saved = json.loads(data)["batch"]["images"][0]["channel"]
    assert saved == {"channel": "green", "source": "chosen"}
    reopened = ops.open_project(s.folder, autosave=save_to_folder)
    assert storage.project_to_json(reopened.project) == data
    assert reopened.project.log[-1].content_hash == storage.content_hash(reopened.project)
    assert [n.code for n in notices(reopened)][:1] == ["colour_channel"]
    assert "channel_signal_elsewhere" not in [n.code for n in notices(reopened)]
    assert messages(reopened, "colour_channel") == [
        "'two µ.png' is measured on its green channel, as chosen." + IMAGEJ
    ]
    ops.set_channel(reopened, image, RED)
    assert stored(reopened, image).channel.rule == 1


# --- The notes ---


def test_each_colour_image_with_a_band_gets_one_note_after_the_copies_note(tmp_path):
    s = session_on(tmp_path)
    green = import_scene(s, one_colour(1, 1), "GFP α.png")
    gray = import_scene(s, gray_as_rgb(1), "gray µ.png")
    empty = import_scene(s, one_colour(0, 2), "RFP β.png")  # no protein on it
    a = boxes(s, green, "GFP α", ROW_A)
    boxes(s, gray, "β-actin", ROW_A)
    ops.add_protein(s, "no band", Role.TARGET, empty)
    computed = results.compute_results(s.project.batch)
    colour = [n for n in computed.notices if n.code == "colour_channel"]
    assert [(n.message, n.image_ids, n.protein_ids, n.level) for n in colour] == [
        (
            "'GFP α.png' is measured on its green channel, which Proteia picked: it holds the"
            " most signal above its background of the three." + IMAGEJ,
            (green,),
            (a,),
            results.Level.INFO,
        )
    ]
    assert computed.notices[0].code == "colour_channel"  # before the set's own


def test_the_notes_name_an_image_by_its_place_where_names_repeat(tmp_path):
    s = session_on(tmp_path)
    first = import_scene(s, one_colour(1, 1), "GFP α.png")
    second = import_scene(s, one_colour(2, 2), "GFP α.png")
    boxes(s, first, "a", ROW_A)
    boxes(s, second, "b", ROW_A)
    assert [m.split(" is measured")[0] for m in messages(s, "colour_channel")] == [
        "'GFP α.png' (image 1)",
        "'GFP α.png' (image 2)",
    ]


@pytest.mark.parametrize(
    ("name", "warned"),
    [
        ("green target, red at a half", True),
        ("green target, red at a quarter", True),
        ("green target, red at a tenth", False),
        ("green saturated in lane 4", False),
        ("yellow bands", False),
    ],
)
def test_signal_in_another_channel_is_warned_about_where_another_target_lies(
    tmp_path, name, warned
):
    s = session_on(tmp_path)
    image = import_scene(s, SCENES[name][0](1), "two µ.png")
    boxes(s, image, "β-catenin", ROW_A if "lane 4" in name or "yellow" in name else ROW_B)
    assert bool(messages(s, "channel_signal_elsewhere")) is warned


def test_two_targets_alike_warn_about_the_other_whichever_is_picked(tmp_path):
    for seed in (1, 3):  # red picked, then green
        s = session_on(tmp_path / str(seed))
        image = import_scene(s, two_targets(seed, AMPS, AMPS), "two µ.png")
        picked = stored(s, image).measured_channel
        boxes(s, image, "β-catenin", ROW_A if picked is RED else ROW_B)
        other = "green" if picked is RED else "red"
        [message] = messages(s, "channel_signal_elsewhere")
        assert f"holds signal in its {other} channel apart" in message


def test_no_note_for_an_image_measured_on_the_mean_about_other_channels(tmp_path):
    s = session_on(tmp_path)
    image = import_scene(s, two_targets(1, AMPS, tuple(2 * a for a in AMPS)), "two µ.png")
    boxes(s, image, "β-catenin", ROW_B)
    ops.set_channel(s, image, MEAN)
    assert messages(s, "channel_signal_elsewhere") == []
    assert messages(s, "colour_channel") == [
        "'two µ.png' is measured on the mean of its red, green and blue, as chosen (ImageJ"
        " measures colour on the mean by default)"
    ]


def test_an_older_colour_image_s_note_says_it_is_measured_as_before(tmp_path):
    s = session_on(tmp_path)
    with mock.patch.object(ops, "_picked", lambda read, kind, polarity: None):
        image = import_scene(s, one_colour(1, 1), "GFP α.png")
    protein = boxes(s, image, "GFP α", ROW_A)
    assert messages(s, "colour_channel") == [
        "'GFP α.png' is measured on the mean of its red, green and blue, as Proteia measured"
        " colour images before it chose a channel; choose a channel for it under Images to"
        " measure it there (its boxes are measured again)"
    ]
    # Lossless: the exact check would run on a channel, and the notice says so.
    [unchecked] = [n for n in notices(s) if n.code == "clipping_not_checked"]
    assert unchecked.protein_ids == (protein,)
    assert unchecked.message == (
        "'GFP α' was not checked for over-exposure in lanes 1, 2, 3, 4, 5, 6: its image has"
        " color channels averaged into gray, so saturated pixels cannot be counted; if it is"
        " over-exposed there, its net is an under-estimate; choose the channel that holds its"
        " bands for its image to check it there"
    )


def test_a_lossy_or_gray_image_is_not_told_to_choose_a_channel(tmp_path):
    s = session_on(tmp_path)
    with mock.patch.object(ops, "_picked", lambda read, kind, polarity: None):
        jpeg = import_scene(s, saturated_jpeg(1), "GFP α.jpg")
    gray = import_scene(s, gray_as_rgb(1), "gray µ.png")
    boxes(s, jpeg, "GFP α", ROW_A)
    boxes(s, gray, "β-actin", ROW_A)
    unchecked = [n for n in notices(s) if n.code == "clipping_not_checked"]
    assert len(unchecked) == 1  # the JPEG's; the gray image is checked
    assert "choose the channel" not in unchecked[0].message
    assert "lossy (JPEG-type) compression and color channels averaged" in unchecked[0].message


def test_the_notes_follow_undo_and_redo(tmp_path):
    s = session_on(tmp_path)
    image = import_scene(s, one_colour(1, 1), "GFP α.png")
    boxes(s, image, "GFP α", ROW_A)
    ops.set_channel(s, image, RED)
    assert messages(s, "colour_channel") == [
        "'GFP α.png' is measured on its red channel, as chosen." + IMAGEJ
    ]
    ops.undo(s)
    assert "which Proteia picked" in messages(s, "colour_channel")[0]


# --- The page, the record and the golden fixtures ---


def test_the_state_says_proteia_picked_the_channel(tmp_path):
    s = session_on(tmp_path)
    image = import_scene(s, one_colour(1, 1), "GFP α.png")
    assert state_channel(s, image) == {
        "measured": "green",
        "source": "picked",
        "offered": True,
        "words": (
            "Picked by Proteia: it holds the most signal above its background (red 0%, blue 0%"
            " of it)."
        ),
    }


def test_a_preview_follows_a_polarity_re_pick(tmp_path):
    workspace = api.Workspace(tmp_path / "root", reveal=lambda folder: None, clock=FakeClock())
    s = workspace.create("A µ")
    scene = SCENES["green on black, read dark on light"][0](1)
    image = import_scene(s, scene, "GFP α.png")
    assert workspace.preview(s, image)[1] is MEAN
    ops.set_polarity(s, image, LIGHT)
    assert workspace.preview(s, image)[1] is GREEN
    assert state_channel(s, image)["measured"] == "green"


def test_an_export_record_holds_the_choice_and_the_rule(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    image = import_scene(s, one_colour(1, 1), "GFP α.png")
    boxes(s, image, "GFP α", ROW_A)
    built = record.build_record(s.project, exported_at="2026-11-01T12:00:00.000Z", files={})
    assert built["history_issues"] == []
    assert built["content_hash"] == s.project.log[-1].content_hash
    [held] = built["content"]["batch"]["images"]
    assert held["channel"] == stored(s, image).channel.model_dump(mode="json")
    assert built["settings"]["channel"]["rule"] == 1
    assert built["settings"]["measured_on"] == {image: {"channel": "green", "check": "exact"}}
    bundle = ops.export_bundle(s, formats=["svg"])
    readme = (bundle.folder / "README.txt").read_text(encoding="utf-8")
    assert "Split Channels" in " ".join(readme.split())


def test_the_golden_colour_project_opens_as_before_and_says_so():
    derived = json.loads((DATA / "f2" / "derived.json").read_text(encoding="utf-8"))
    notes = [n for n in derived["results"]["notices"] if n["code"] == "colour_channel"]
    assert [(n["message"], n["image_ids"], n["level"]) for n in notes] == [
        (
            "'C colour α.png' is measured on the mean of its red, green and blue, as Proteia"
            " measured colour images before it chose a channel; choose a channel for it under"
            " Images to measure it there (its boxes are measured again)",
            ["img-3"],
            "info",
        )
    ]
    # Its lossless colour image's proteins are told to choose a channel; the
    # JPEG's protein is not (no channel would check it exactly).
    unchecked = [n for n in derived["results"]["notices"] if n["code"] == "clipping_not_checked"]
    told = {n["message"].split("'")[1]: n["message"].endswith(CHOOSE) for n in unchecked}
    assert told == {"β-catenin": False, "α-tubulin": True, "p53 µ": True}
    # A fresh build of it picks the colour image's channel: its record and the
    # import's log entry, nothing else.
    assert [path for path, _, _ in derived["fresh_changes"]] == [
        "$.batch.membranes[1].images[0]",
        "$.log[2].params",
    ]
    _, old, new = derived["fresh_changes"][0]
    assert {key: value for key, value in new.items() if key != "channel"} == old
    assert (new["channel"]["channel"], new["channel"]["source"]) == ("mean", "picked")
    for case in ("f1", "f3"):
        other = json.loads((DATA / case / "derived.json").read_text(encoding="utf-8"))
        assert other["fresh_changes"] == []
        assert "colour_channel" not in [n["code"] for n in other["results"]["notices"]]


# --- Worst scale ---


def test_the_rule_on_a_large_image_stays_within_its_time_and_memory(tmp_path):
    # 2000 x 3000 16-bit RGB: measured at about 0.23 s and 30.2 MB traced
    # beyond the pixels (most of it one channel's region labels); bounded here
    # with margin for slower machines.
    rng = np.random.default_rng(7)
    h, w = 2000, 3000
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.zeros((h, w))
    for x in range(150, w, 300):
        for y in range(200, h, 400):
            field += 39000.0 * np.exp(-(((xx - x) / 40.0) ** 2 + ((yy - y) / 12.0) ** 2))
    del yy, xx
    channels = [3000.0 + 0.1 * field, 3000.0 + field, 3000.0 + 0 * field]
    pixels = np.dstack(
        [np.clip(np.rint(c + rng.normal(0, 260.0, (h, w))), 0, 65535) for c in channels]
    ).astype(np.uint16)
    del field, channels
    for dark in (False, True):
        image = 65535 - pixels if dark else pixels
        start = time.perf_counter()
        masses, _ = imaging.channel_masses(image, 16, dark_on_light=dark)
        took = time.perf_counter() - start
        tracemalloc.start()
        before, _ = tracemalloc.get_traced_memory()
        imaging.channel_masses(image, 16, dark_on_light=dark)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert masses[1] > masses[0] > 0 and masses[2] == 0.0
        assert took < 2.0
        assert peak - before < 40 * 2**20
