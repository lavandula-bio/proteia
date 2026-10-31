# SPDX-License-Identifier: Apache-2.0
"""Measuring a colour image on one of its channels (#206): the stored choice
and the images it may be made for, reading an image on a channel, the
over-exposure checks' trust in one channel, and the operation that sets it
(and what undo, a polarity change, row detection and a seed click do with it).

The images are synthetic: bands in the green channel alone, light on dark as a
fluorescence scan, or dark on a light membrane, with or without one lane
saturated in green and a share of green's signal in red. Every import here
stores no choice, as before #206: the image is measured on the mean of its
red, green and blue until a channel is chosen for it.
"""

import contextlib
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image
from pydantic import ValidationError

from conftest import FakeClock, write_tiff
from proteia.core import imaging, results, storage
from proteia.core import operations as ops
from proteia.core.grow import NOISE_K, REL_THRESHOLD, grow_box
from proteia.core.model import (
    UNTRUSTED_WARNINGS,
    BoxSize,
    Channel,
    ChannelChoice,
    ChannelSource,
    ImageKind,
    ImageRef,
    ImageWarning,
    Polarity,
    Project,
    ProposalSource,
    Role,
    UnknownIdError,
    apply_change,
    limit_distrusted_by,
)
from proteia.core.operations import ErrorCode, LaneInput, OperationError
from proteia.core.quantify import band_backgrounds, quantify_nets, to_grayscale
from proteia.core.session import save_to_folder
from test_operations import (
    Recorder,
    _before_207,
    _record,
    _record_json,
    assert_nets_current,
    boxes_of,
    plant,
    plant_records,
    session_on,
)
from test_schema2_golden import DATA, fixtures

CHEMI, MARKER, MERGED = ImageKind.CHEMILUMINESCENCE, ImageKind.VISIBLE_MARKER, ImageKind.MERGED
DARK, LIGHT = Polarity.DARK_ON_LIGHT, Polarity.LIGHT_ON_DARK
RED, GREEN, BLUE, MEAN = Channel.RED, Channel.GREEN, Channel.BLUE, Channel.MEAN

H, W = 60, 230
ROW = 30
LANES = (25, 60, 95, 130, 165, 200)
SATURATED = 3  # the lane saturated in green: lane 4
BOX = BoxSize(width=20, height=12)
NEUTRAL = (
    "The red, green and blue channels differ; the image is measured on one of them"
    " or on their mean, as its channel says."
)


def _bands(amplitudes) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.zeros((H, W))
    for x, amplitude in zip(LANES, amplitudes, strict=True):
        out += amplitude * np.exp(-(((xx - x) / 5.0) ** 2 + ((yy - ROW) / 3.0) ** 2))
    return out


def colour_blot(
    depth: int = 8,
    *,
    saturate: bool = False,
    crosstalk: float = 0.0,
    dark: bool = False,
    floor: float = 0.04,
    noise: float = 0.004,
    seed: int = 0,
) -> np.ndarray:
    """Six bands in the green channel alone, of rising strength, light on a
    dark background (or, with ``dark``, dark on a light membrane): red holds
    ``crosstalk`` of green's signal, blue none; with ``saturate``, lane 4 is
    saturated in green (red is not, at 10% crosstalk). Every channel has the
    background ``floor`` and normal noise of SD ``noise`` (shares of the
    range)."""
    top = 2**depth - 1
    rng = np.random.default_rng(seed)
    amplitudes = [(0.25 + 0.07 * lane) * top for lane in range(6)]
    if saturate:
        amplitudes[SATURATED] = 3.0 * top
    signal = _bands(amplitudes)
    channels = [
        floor * top + share * signal + rng.normal(0, noise * top, (H, W))
        for share in (crosstalk, 1.0, 0.0)
    ]
    rgb = np.clip(np.rint(np.dstack(channels)), 0, top)
    if dark:
        rgb = top - rgb
    return rgb.astype(np.uint8 if depth == 8 else np.uint16)


def write(path: Path, pixels: np.ndarray, **options) -> Path:
    """``pixels`` as a TIFF (uncompressed), or by Pillow in the suffix's format."""
    if path.suffix == ".tif":
        return write_tiff(path, pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, **options)
    return path


def import_pixels(
    s: ops.ProjectSession,
    pixels: np.ndarray,
    name: str,
    polarity: Polarity = LIGHT,
    *,
    kind: ImageKind = CHEMI,
    **options,
) -> str:
    source = write(s.folder.parent / "sources" / name, pixels, **options)
    with source.open("rb") as f:
        return ops.import_image(s, f, name, kind=kind, polarity=polarity)


def six_boxes(s: ops.ProjectSession, image_id: str, name: str = "GFP α") -> tuple[str, list[str]]:
    """A protein on the image with a fixed box on each of the six bands."""
    if not s.project.batch.only_lanes():
        ops.set_lanes(s, [LaneInput(f"c{i} µ") for i in range(6)])
    protein = ops.add_protein(s, name, Role.TARGET, image_id, box_size=BOX)
    bands = [
        ops.place_box(s, protein, x, ROW, lane_index=lane, grow=False)
        for lane, x in enumerate(LANES)
    ]
    return protein, bands


def stored(s: ops.ProjectSession, image_id: str) -> ImageRef:
    return s.project.batch.find_image(image_id)


def bands_on(s: ops.ProjectSession, image_id: str) -> list:
    """Every band on the image, as _quantify_image orders them: protein order,
    then band order; with each protein's box size."""
    batch = s.project.batch
    return [
        (batch.boxes_of(p).box_size, band)
        for p in batch.proteins_on(image_id)
        for band in batch.boxes_of(p).bands
    ]


def expected_nets(s: ops.ProjectSession, image_id: str, plane: np.ndarray) -> list[float]:
    """The nets of every band on the image measured on ``plane`` (2-D), by the
    product's one entry point (quantify_nets), every box together."""
    placed = bands_on(s, image_id)
    return quantify_nets(
        plane.astype(np.float64),
        [band.box.rect(size) for size, band in placed],
        [(size.width, size.height) for size, _ in placed],
        method="ring_median",
        dark_on_light=stored(s, image_id).polarity.dark_on_light,
        integral=True,
    )


def flags(s: ops.ProjectSession, protein_id: str) -> list[tuple[bool | None, bool | None]]:
    return [(b.clipped, b.possibly_clipped) for b in boxes_of(s, protein_id).bands]


def warning(code: str) -> ImageWarning:
    return ImageWarning(code=code, message=f"{code} µ")


def image_doc(**fields) -> dict:
    """An image record as a file holds it: a colour chemiluminescence TIFF
    whose channels differ."""
    return {
        "id": "img-3",
        "file": "img-3.tif",
        "original_name": "GFP α.tif",
        "kind": "chemiluminescence",
        "sha256": "a" * 64,
        "width": W,
        "height": H,
        "bit_depth": 8,
        "polarity": "light_on_dark",
        "background": 10.0,
        "import_warnings": [{"code": "color_channels_differ", "message": "differ µ"}],
        **fields,
    }


# --- The model: the stored choice ---


def test_the_channels_and_sources_are_the_file_format():
    assert [c.value for c in Channel] == ["red", "green", "blue", "mean"]
    assert [s.value for s in ChannelSource] == ["chosen"]


def test_a_choice_is_left_out_of_the_file_while_there_is_none():
    image = ImageRef.model_validate(image_doc())
    assert image.channel is None and image.measured_channel is MEAN
    assert "channel" not in image.model_dump(mode="json")
    chosen = ImageRef.model_validate(image_doc(channel={"channel": "green", "source": "chosen"}))
    assert chosen.channel == ChannelChoice(channel=GREEN, source=ChannelSource.CHOSEN)
    assert chosen.measured_channel is GREEN
    dumped = chosen.model_dump(mode="json")
    assert dumped["channel"] == {"channel": "green", "source": "chosen"}
    assert list(dumped)[-1] == "channel"  # after every field an older file holds
    assert ImageRef.model_validate(dumped) == chosen
    mean = ImageRef.model_validate(image_doc(channel={"channel": "mean", "source": "chosen"}))
    assert mean.measured_channel is MEAN and mean.channel is not None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"import_warnings": []},
            "image img-3: a channel is chosen only for an image whose red, green and blue differ",
        ),
        (
            {"import_warnings": [{"code": "lossy_format", "message": "lossy"}]},
            "image img-3: a channel is chosen only for an image whose red, green and blue differ",
        ),
        (
            {"kind": "visible_marker"},
            "image img-3: a visible-light marker image is measured on the mean of its"
            " colours, never on one channel",
        ),
        (
            {
                "import_warnings": [
                    {"code": "cmyk_converted", "message": "cmyk"},
                    {"code": "color_channels_differ", "message": "differ"},
                ]
            },
            "image img-3: an image converted from CMYK is measured on the mean of its"
            " colours, never on one channel",
        ),
    ],
    ids=["gray", "lossy gray", "marker", "cmyk"],
)
def test_a_choice_is_refused_where_no_channel_can_be_chosen(fields, message):
    doc = image_doc(**fields, channel={"channel": "green", "source": "chosen"})
    with pytest.raises(ValidationError) as caught:
        ImageRef.model_validate(doc)
    assert caught.value.errors()[0]["msg"] == f"Value error, {message}"
    # Without a choice, each is an image as before.
    del doc["channel"]
    assert ImageRef.model_validate(doc).channel is None


@pytest.mark.parametrize("kind", ["chemiluminescence", "merged"])
def test_a_choice_is_taken_for_a_colour_image_of_a_signal_kind(kind):
    image = ImageRef.model_validate(
        image_doc(kind=kind, channel={"channel": "blue", "source": "chosen"})
    )
    assert image.measured_channel is BLUE
    assert image.channel_blocker is None


def test_an_unknown_channel_or_source_in_a_file_is_refused():
    for choice in (
        {"channel": "purple", "source": "chosen"},
        {"channel": "green", "source": "picked"},
        {"channel": "green"},
        {"channel": "green", "source": "chosen", "masses": [1, 0, 0]},
    ):
        with pytest.raises(ValidationError):
            ImageRef.model_validate(image_doc(channel=choice))


def test_why_no_channel_can_be_chosen():
    def blocker(**fields):
        return ImageRef.model_validate(image_doc(**fields)).channel_blocker

    differ = {"code": "color_channels_differ", "message": "d"}
    cmyk = {"code": "cmyk_converted", "message": "c"}
    assert blocker() is None
    assert blocker(import_warnings=[]) == "gray"
    assert blocker(import_warnings=[], kind="visible_marker") == "gray"  # gray first
    assert blocker(kind="visible_marker") == "marker"
    assert blocker(import_warnings=[cmyk, differ]) == "cmyk"
    assert blocker(import_warnings=[differ, cmyk], kind="visible_marker") == "marker"


# --- The trust predicate: one source for imaging and results ---


def test_one_colour_channel_is_not_distrusted_for_the_channels_differing():
    both = [warning("lossy_format"), warning("color_channels_differ")]
    mirrored = [warning("color_channels_differ"), warning("lossy_format")]
    processed = [warning("looks_processed"), warning("color_channels_differ")]
    cmyk = [warning("cmyk_converted"), warning("color_channels_differ")]
    for channel in (None, MEAN):
        assert limit_distrusted_by(both, channel) == ["lossy_format", "color_channels_differ"]
        assert limit_distrusted_by(mirrored, channel) == ["color_channels_differ", "lossy_format"]
        assert limit_distrusted_by(processed, channel) == ["color_channels_differ"]
    for channel in (RED, GREEN, BLUE):
        assert limit_distrusted_by(both, channel) == ["lossy_format"]
        assert limit_distrusted_by(mirrored, channel) == ["lossy_format"]
        assert limit_distrusted_by(processed, channel) == []
        assert limit_distrusted_by(cmyk, channel) == ["cmyk_converted"]
    assert limit_distrusted_by([]) == []
    assert limit_distrusted_by(both) == ["lossy_format", "color_channels_differ"]
    assert UNTRUSTED_WARNINGS == {"lossy_format", "color_channels_differ", "cmyk_converted"}
    assert imaging.UNTRUSTED_WARNINGS is UNTRUSTED_WARNINGS


def test_the_clipping_depths_on_a_channel():
    differ = [warning("color_channels_differ")]
    lossy = [warning("lossy_format"), *differ]
    cmyk = [warning("cmyk_converted"), *differ]
    for depth in (8, 16):
        # The two-argument call is the mean's, as before #206.
        assert imaging.clipping_depth(depth, differ) is None
        assert imaging.possible_clipping_depth(depth, differ) == depth
        for channel in (None, MEAN):
            assert imaging.clipping_depth(depth, differ, channel) is None
            assert imaging.possible_clipping_depth(depth, differ, channel) == depth
        for channel in (RED, GREEN, BLUE):
            assert imaging.clipping_depth(depth, differ, channel) == depth
            assert imaging.possible_clipping_depth(depth, differ, channel) is None
            assert imaging.clipping_depth(depth, lossy, channel) is None
            assert imaging.possible_clipping_depth(depth, lossy, channel) == depth
            assert imaging.clipping_depth(depth, cmyk, channel) is None
            assert imaging.possible_clipping_depth(depth, cmyk, channel) == depth
        assert imaging.clipping_depth(depth, [], GREEN) == depth
    for channel in (None, GREEN):
        assert imaging.clipping_depth(None, differ, channel) is None
        assert imaging.possible_clipping_depth(None, differ, channel) is None


def test_the_results_name_what_distrusts_the_channel_measured():
    # A lossy colour image measured on one channel: only its compression keeps
    # the exact check off; measured on the mean, its colours averaged too.
    lossy_colour = [
        {"code": "lossy_format", "message": "l"},
        {"code": "color_channels_differ", "message": "d"},
    ]
    mean = ImageRef.model_validate(image_doc(import_warnings=lossy_colour))
    green = ImageRef.model_validate(
        image_doc(import_warnings=lossy_colour, channel={"channel": "green", "source": "chosen"})
    )
    assert results._unchecked_reasons(mean) == (
        "lossy (JPEG-type) compression and color channels averaged into gray"
    )
    assert results._unchecked_reasons(green) == "lossy (JPEG-type) compression"
    lossless_green = ImageRef.model_validate(
        image_doc(channel={"channel": "green", "source": "chosen"})
    )
    assert results._unchecked_reasons(lossless_green) == ""
    band = boxes_band(clipped=None, possibly_clipped=None)
    assert results._unassessed(mean, band)
    assert results._unassessed(green, band)  # its compression still distrusts it
    assert not results._unassessed(lossless_green, band)


def boxes_band(**fields):
    from proteia.core.model import Band, Box

    return Band(
        id="band-1",
        lane_index=0,
        band_index=0,
        box=Box(x=0, y=0),
        net=0.0,
        background_level=0.0,
        background_mode="image",
        background_spread=0.0,
        source="manual",
        **fields,
    )


def test_the_log_says_why_a_channel_is_not_checked_exactly():
    lossy_colour = [
        {"code": "lossy_format", "message": "l"},
        {"code": "color_channels_differ", "message": "d"},
    ]
    mean = ImageRef.model_validate(image_doc(import_warnings=lossy_colour))
    green = ImageRef.model_validate(
        image_doc(import_warnings=lossy_colour, channel={"channel": "green", "source": "chosen"})
    )
    near = "assessed near the detector limit instead"
    assert ops._unchecked_why(mean) == f"lossy_format, color_channels_differ; {near}"
    assert ops._unchecked_why(green) == f"lossy_format; {near}"
    unknown = ImageRef.model_validate(
        image_doc(bit_depth=None, channel={"channel": "red", "source": "chosen"})
    )
    assert ops._unchecked_why(unknown) == "unknown bit depth; not assessed either"


# --- Reading an image on a channel ---


def test_to_grayscale_takes_the_channel_asked_for():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
    rgba = np.dstack([rgb, rng.integers(0, 256, (5, 7), dtype=np.uint8)])
    for pixels in (rgb, rgba):
        assert np.array_equal(to_grayscale(pixels, RED), rgb[..., 0])
        assert np.array_equal(to_grayscale(pixels, GREEN), rgb[..., 1])
        assert np.array_equal(to_grayscale(pixels, BLUE), rgb[..., 2])
        assert to_grayscale(pixels, GREEN).dtype == np.uint8  # as read
        mean = rgb.mean(axis=-1)
        assert np.array_equal(to_grayscale(pixels), mean)
        assert np.array_equal(to_grayscale(pixels, MEAN), mean)
    gray = rgb[..., 0]
    gray_alpha = np.dstack([gray, rgb[..., 1]])
    for channel in (None, MEAN):
        assert to_grayscale(gray, channel) is gray
        assert np.array_equal(to_grayscale(gray_alpha, channel), gray)
    for pixels, layout in ((gray, "gray"), (gray_alpha, "gray")):
        for channel in (RED, GREEN, BLUE):
            with pytest.raises(ValueError, match=f"^a {layout} image has no {channel} channel$"):
                to_grayscale(pixels, channel)
    with pytest.raises(ValueError, match="unsupported image layout"):
        to_grayscale(np.zeros((2, 2, 5)), GREEN)


def test_channels_differ():
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, (4, 6), dtype=np.uint8)
    equal = np.dstack([gray, gray, gray])
    assert not imaging.channels_differ(gray)
    assert not imaging.channels_differ(np.dstack([gray, gray]))  # gray and alpha
    assert not imaging.channels_differ(equal)
    assert not imaging.channels_differ(np.dstack([equal, 255 - gray]))  # alpha apart
    for channel in range(3):
        one_off = equal.copy()
        one_off[2, 3, channel] ^= 1
        assert imaging.channels_differ(one_off), channel
        assert imaging.channels_differ(np.dstack([one_off, gray])), channel


def test_an_image_is_read_on_its_channel_with_the_warnings_of_its_file(tmp_path):
    rgb = colour_blot(16, crosstalk=0.1)
    path = write_tiff(tmp_path / "GFP µ.tif", rgb)
    mean = imaging.load_image(path)
    assert np.array_equal(mean.array, rgb.mean(axis=-1))
    assert [w.code for w in mean.warnings] == ["color_channels_differ"]
    for channel, index in ((RED, 0), (GREEN, 1), (BLUE, 2)):
        for loaded in (
            imaging.load_image(path, channel=channel),
            imaging.load_pixels(imaging.read_file(path), channel=channel),
            imaging.from_pixels(rgb, channel=channel),
        ):
            assert loaded.array.dtype == np.float64
            assert np.array_equal(loaded.array, rgb[..., index])
            assert loaded.warnings == mean.warnings  # the file's, whatever the channel
            assert loaded.pixels is not None and loaded.bit_depth == 16
        array, warnings = imaging.to_analysis_array(rgb, channel=channel)
        assert np.array_equal(array, rgb[..., index]) and warnings == mean.warnings
    assert np.array_equal(imaging.load_image(path, channel=MEAN).array, mean.array)
    with pytest.raises(ValueError, match="gray image has no green channel"):
        imaging.from_pixels(rgb[..., 1], channel=GREEN)


def test_looks_processed_is_assessed_on_the_channel_and_its_check():
    # Light on dark, every pixel at 1: within the near-limit tolerance of black
    # where the exact check is off (the mean of colours that differ), but not
    # at the limit where it runs (one channel of a lossless file).
    differ = [warning("color_channels_differ")]
    array = np.ones((10, 10))
    kwargs = {"dark_on_light": False, "background": 1.0}
    for channel in (None, MEAN):
        codes = [
            w.code for w in imaging.assess_processed(differ, array, 8, **kwargs, channel=channel)
        ]
        assert codes == ["color_channels_differ", "looks_processed"]
    for channel in (RED, GREEN, BLUE):
        codes = [
            w.code for w in imaging.assess_processed(differ, array, 8, **kwargs, channel=channel)
        ]
        assert codes == ["color_channels_differ"]
    assert [w.code for w in imaging.assess_processed(differ, array, 8, **kwargs)] == [
        "color_channels_differ",
        "looks_processed",
    ]


# --- set_channel ---


def test_set_channel_measures_every_band_on_the_image_on_that_channel(tmp_path):
    s = session_on(tmp_path)
    rgb = colour_blot(crosstalk=0.1)
    image = import_pixels(s, rgb, "GFP µ.tif")
    other = import_pixels(s, colour_blot(seed=1), "RFP β.tif")
    gfp, gfp_bands = six_boxes(s, image)
    # A second protein on the image, its boxes below the bands, and records
    # and band counts the change must drop.
    tubulin = ops.add_protein(s, "α-tubulin", Role.LOADING_CONTROL, image, box_size=BOX)
    for lane in (0, 2):
        ops.place_box(s, tubulin, LANES[lane], ROW + 20, lane_index=lane, grow=False)
    on_other, _ = six_boxes(s, other, "RFP β")
    actin = ops.add_protein(s, "β-actin", Role.LOADING_CONTROL, other)  # records only
    records = [_record(4, region=(150, 20, 180, 40)), _record(1, region=(40, 20, 80, 40))]
    plant_records(s, tubulin, *records)
    plant_records(s, actin, _record(5))

    def counted(draft: Project) -> None:
        for band in draft.batch.boxes_of(draft.batch.find_protein(gfp)).bands:
            band.source, band.bands_found = ProposalSource.ROW_BOX, 1

    plant(s, counted)
    other_before = (boxes_of(s, on_other), boxes_of(s, actin), stored(s, other))
    other_pixels = s.pixels(other)

    # Before: the mean, as before #206.
    mean = rgb.mean(axis=-1)
    before = stored(s, image)
    assert before.channel is None and before.background == float(np.median(mean))
    assert [band.net for _, band in bands_on(s, image)] == expected_nets(s, image, mean)
    assert flags(s, gfp) == [(None, False)] * 6
    old_message = next(w for w in before.import_warnings if w.code == "color_channels_differ")
    assert "averaged into one gray channel" in old_message.message

    ops.set_channel(s, image, "green")
    # The change caches the array it read; another image's stays cached.
    green = rgb[..., 1].astype(np.float64)
    assert np.array_equal(s._pixels[image], green)
    assert s._pixels[other] is other_pixels
    after = stored(s, image)
    assert after.channel == ChannelChoice(channel=GREEN, source=ChannelSource.CHOSEN)
    assert after.background == float(np.median(green))
    nets = [band.net for _, band in bands_on(s, image)]
    assert nets == expected_nets(s, image, green)
    assert nets != expected_nets(s, image, mean)
    placed = bands_on(s, image)
    found = band_backgrounds(
        green,
        [band.box.rect(size) for size, band in placed],
        [(size.width, size.height) for size, _ in placed],
        dark_on_light=False,
        integral=True,
        fallback=after.background,
    )
    assert [(b.background_level, b.background_mode, b.background_spread) for _, b in placed] == [
        (f.level, f.mode, f.spread) for f in found
    ]
    assert flags(s, gfp) == [(False, None)] * 6  # the exact check runs on one channel
    assert [b.bands_found for b in boxes_of(s, gfp).bands] == [None] * 6
    assert boxes_of(s, tubulin).undetected == []
    assert [w.code for w in after.import_warnings] == ["color_channels_differ"]
    assert after.import_warnings[0].message == NEUTRAL
    entry = s.project.log[-1]
    assert (entry.action, entry.params) == (
        "set_channel",
        {
            "image_id": image,
            "channel": "green",
            "from": "mean",
            "dropped_undetected": [_record_json(tubulin, r) for r in records[::-1]],  # by lane
        },
    )
    assert entry.content_hash == storage.content_hash(s.project)
    assert (boxes_of(s, on_other), boxes_of(s, actin), stored(s, other)) == other_before
    assert_nets_current(s)

    # Another channel, then the mean again: chosen too.
    ops.set_channel(s, image, Channel.RED)
    red = rgb[..., 0].astype(np.float64)
    assert stored(s, image).background == float(np.median(red))
    assert [band.net for _, band in bands_on(s, image)] == expected_nets(s, image, red)
    assert s.project.log[-1].params["from"] == "green"
    ops.set_channel(s, image, MEAN)
    back = stored(s, image)
    assert back.channel == ChannelChoice(channel=MEAN, source=ChannelSource.CHOSEN)
    assert back.background == before.background
    assert [band.net for _, band in bands_on(s, image)] == expected_nets(s, image, mean)
    assert flags(s, gfp) == [(None, False)] * 6  # the mean's near-limit check again
    assert back.import_warnings[0].message == NEUTRAL
    assert_nets_current(s)


def test_a_channel_measures_as_the_same_channel_saved_gray(tmp_path):
    # The oracle: the green channel saved as a gray TIFF and measured as such
    # gives every stored value of the colour image measured on green.
    for dark, polarity in ((False, LIGHT), (True, DARK)):
        rgb = colour_blot(16, dark=dark, crosstalk=0.2, saturate=True)
        colour = session_on(tmp_path / f"colour {dark}")
        image = import_pixels(colour, rgb, "GFP µ.tif", polarity)
        protein, _ = six_boxes(colour, image)
        ops.set_channel(colour, image, GREEN)
        gray = session_on(tmp_path / f"gray {dark}")
        plane = import_pixels(gray, np.ascontiguousarray(rgb[..., 1]), "green α.tif", polarity)
        on_plane, _ = six_boxes(gray, plane)
        assert stored(colour, image).background == stored(gray, plane).background
        assert boxes_of(colour, protein).bands == boxes_of(gray, on_plane).bands
        assert flags(colour, protein)[SATURATED] == (True, None)


@pytest.mark.parametrize(("depth", "suffix"), [(8, "png"), (8, "tif"), (16, "tif")])
def test_a_band_saturated_in_the_chosen_channel_is_flagged(tmp_path, depth, suffix):
    s = session_on(tmp_path)
    rgb = colour_blot(depth, saturate=True, crosstalk=0.1)
    image = import_pixels(s, rgb, f"GFP µ.{suffix}")
    protein, _ = six_boxes(s, image)
    assert flags(s, protein) == [(None, False)] * 6  # the mean hides it, as before #206
    ops.set_channel(s, image, GREEN)
    expected = [(lane == SATURATED, None) for lane in range(6)]
    assert flags(s, protein) == expected
    ops.set_channel(s, image, RED)  # red holds a tenth of green's signal: below its limit
    assert flags(s, protein) == [(False, None)] * 6
    ops.set_channel(s, image, MEAN)
    assert flags(s, protein) == [(None, False)] * 6
    assert_nets_current(s)


def test_a_dark_band_saturated_in_the_chosen_channel_is_flagged(tmp_path):
    s = session_on(tmp_path)
    rgb = colour_blot(16, saturate=True, dark=True)
    image = import_pixels(s, rgb, "dark GFP µ.tif", DARK)
    protein, _ = six_boxes(s, image)
    ops.set_channel(s, image, GREEN)
    assert flags(s, protein) == [(lane == SATURATED, None) for lane in range(6)]


def test_on_a_jpeg_the_chosen_channel_is_assessed_near_its_limit(tmp_path):
    s = session_on(tmp_path)
    rgb = colour_blot(saturate=True, crosstalk=0.1)
    image = import_pixels(s, rgb, "GFP µ.jpg", quality=95)
    protein, _ = six_boxes(s, image)
    assert flags(s, protein) == [(None, False)] * 6
    ops.set_channel(s, image, GREEN)
    assert flags(s, protein) == [(None, lane == SATURATED) for lane in range(6)]
    # Measured on the decoded file's green, its stored file's pixels.
    decoded = np.asarray(Image.open(storage.image_path(s.folder, stored(s, image))))
    assert [b.net for b in boxes_of(s, protein).bands] == expected_nets(s, image, decoded[..., 1])
    assert_nets_current(s)


def test_an_image_with_no_band_takes_the_channel_with_its_median_and_warnings(tmp_path):
    s = session_on(tmp_path, Recorder())
    rgb = colour_blot(crosstalk=0.3)
    image = import_pixels(s, rgb, "GFP µ.png")
    ops.set_channel(s, image, BLUE)
    after = stored(s, image)
    assert after.measured_channel is BLUE
    assert after.background == float(np.median(rgb[..., 2]))
    assert after.import_warnings == [ImageWarning(code="color_channels_differ", message=NEUTRAL)]
    assert s.project.log[-1].params == {
        "image_id": image,
        "channel": "blue",
        "from": "mean",
        "dropped_undetected": [],
    }
    assert s.autosave.actions == ["import_image", "set_channel"]


def test_the_looks_processed_warning_is_assessed_on_the_chosen_channel(tmp_path):
    # Light on dark: red and blue at 1 everywhere, the bands in green on a
    # floor of 1. The mean's median, 1, is near black where the near-limit
    # check counts (two levels); green's is not at black, where the exact
    # check counts. Red, all at 0, is pure black.
    s = session_on(tmp_path)
    rgb = colour_blot(floor=1 / 255, noise=0)
    rgb[..., 0] = 0
    rgb[..., 2] = 1
    image = import_pixels(s, rgb, "GFP µ.tif")
    codes = [w.code for w in stored(s, image).import_warnings]
    assert codes == ["color_channels_differ", "looks_processed"]
    ops.set_channel(s, image, GREEN)
    assert [w.code for w in stored(s, image).import_warnings] == ["color_channels_differ"]
    ops.set_channel(s, image, RED)
    warnings = stored(s, image).import_warnings
    assert [w.code for w in warnings] == ["color_channels_differ", "looks_processed"]
    assert warnings[1].message.startswith(
        "This looks like a processed figure rather than a raw scan: its median level, the"
        " image-wide background, is pure black; 100% of its pixels are pure black."
    )


def test_a_legacy_project_measures_on_the_chosen_channel_with_its_median(tmp_path):
    # A project quantified before #83 (global_median): every band's level is its
    # image's median, so the channel's median.
    from conftest import as_legacy

    s = session_on(tmp_path)
    rgb = colour_blot(crosstalk=0.1)
    image = import_pixels(s, rgb, "GFP µ.tif")
    protein, _ = six_boxes(s, image)
    s._commit(as_legacy(s.project), action="plant", params={})
    ops.set_channel(s, image, GREEN)
    green = rgb[..., 1]
    level = float(np.median(green))
    assert {b.background_level for b in boxes_of(s, protein).bands} == {level}
    placed = bands_on(s, image)
    assert [b.net for _, b in placed] == quantify_nets(
        green.astype(np.float64),
        [b.box.rect(size) for size, b in placed],
        [(size.width, size.height) for size, _ in placed],
        method="global_median",
        dark_on_light=False,
        integral=True,
    )
    assert_nets_current(s)


# --- Refusals ---

GRAY = (
    "{name!r} has one gray channel (its red, green and blue are equal, or it is gray):"
    " there is no channel to choose, and it is measured as it is"
)


def _refused(s: ops.ProjectSession, image_id: str, channel) -> OperationError:
    project, cached = s.project, dict(s._pixels)
    with pytest.raises(OperationError) as caught:
        ops.set_channel(s, image_id, channel)
    assert s.project is project  # nothing committed
    assert s._pixels.keys() == cached.keys()
    return caught.value


def test_set_channel_refuses_an_image_with_one_gray_channel(tmp_path):
    s = session_on(tmp_path)
    plane = colour_blot()[..., 1]
    for name, pixels, options in (
        ("gray µ.tif", plane, {}),
        ("gray as RGB α.tif", np.dstack([plane] * 3), {}),
        ("gray and alpha β.png", np.dstack([plane, np.full_like(plane, 255)]), {}),
        ("gray γ.jpg", plane, {"quality": 95}),
    ):
        image = import_pixels(s, pixels, name, **options)
        for channel in ("green", "mean"):
            refused = _refused(s, image, channel)
            assert (refused.code, str(refused), refused.ids) == (
                ErrorCode.INVALID_INPUT,
                GRAY.format(name=name),
                (image,),
            )


def test_set_channel_refuses_a_marker_image(tmp_path):
    s = session_on(tmp_path)
    marker = import_pixels(s, colour_blot(dark=True), "marker µ.png", DARK, kind=MARKER)
    refused = _refused(s, marker, "red")
    assert (refused.code, str(refused), refused.ids) == (
        ErrorCode.INVALID_INPUT,
        "'marker µ.png' is a marker image: it is read as the mean of its colours so that"
        " every ladder band shows; choose a channel on the images that hold proteins",
        (marker,),
    )
    merged = import_pixels(s, colour_blot(dark=True), "merged β.png", DARK, kind=MERGED)
    ops.set_channel(s, merged, "red")  # an overlay is not a marker
    assert stored(s, merged).measured_channel is RED


def test_set_channel_refuses_an_image_converted_from_cmyk(tmp_path):
    # Imported between #131 and #207: its colours converted from cyan ink.
    s = session_on(tmp_path)
    gray = (colour_blot(16, dark=True)[..., 1]).astype(np.uint16)
    inks = np.zeros((*gray.shape, 4), dtype=np.uint16)
    inks[..., 0] = 65535 - gray
    path = tmp_path / "cyan µ.tif"
    tifffile.imwrite(path, inks, photometric="separated")
    with path.open("rb") as f:
        image = _before_207(s, f, "cyan µ.tif")
    codes = [w.code for w in stored(s, image).import_warnings]
    assert codes == ["cmyk_converted", "color_channels_differ"]
    refused = _refused(s, image, "green")
    assert (refused.code, str(refused)) == (
        ErrorCode.INVALID_INPUT,
        "'cyan µ.tif' was converted from CMYK when it was imported, so its red, green and"
        " blue are estimates; save the original as an uncompressed RGB TIFF and import it"
        " again to measure one channel",
    )


@pytest.mark.parametrize("given", ["purple", "Green", "", None, 1, ["green"]])
def test_set_channel_refuses_what_is_not_a_channel(tmp_path, given):
    s = session_on(tmp_path)
    image = import_pixels(s, colour_blot(), "GFP µ.tif")
    refused = _refused(s, image, given)
    assert (refused.code, str(refused)) == (
        ErrorCode.INVALID_INPUT,
        f"channel must be one of 'red', 'green', 'blue', 'mean', not {given!r}",
    )


def test_set_channel_refuses_an_unknown_image(tmp_path):
    s = session_on(tmp_path)
    import_pixels(s, colour_blot(), "GFP µ.tif")
    project = s.project
    with pytest.raises(UnknownIdError):
        ops.set_channel(s, "img-99", "green")
    assert s.project is project


def test_set_channel_refuses_a_file_changed_or_missing(tmp_path):
    s = session_on(tmp_path)
    image = import_pixels(s, colour_blot(), "GFP µ.tif")
    six_boxes(s, image)
    s.pixels(image)  # the mean's array cached
    path = storage.image_path(s.folder, stored(s, image))
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    refused = _refused(s, image, "green")
    assert (refused.code, str(refused), refused.ids) == (
        ErrorCode.IMAGE_FILE_CHANGED,
        f"the stored file of image {image} was changed outside Proteia",
        (image,),
    )
    path.unlink()
    refused = _refused(s, image, "green")
    assert (refused.code, str(refused)) == (
        ErrorCode.IMAGE_FILE_CHANGED,
        f"the stored file of image {image} is missing",
    )
    path.write_bytes(original)
    ops.set_channel(s, image, "green")  # the file back: it is taken
    assert stored(s, image).measured_channel is GREEN


def test_set_channel_refuses_an_image_read_in_a_way_proteia_no_longer_reads(tmp_path):
    # A CMYK file imported before #131 converted CMYK: its inks were read as
    # red, green and blue, which differed. Its stored numbers come from that
    # reading, so it is measured on no channel of this one.
    s = session_on(tmp_path)
    gray = (colour_blot(16, dark=True)[..., 1]).astype(np.uint16)
    inks = np.zeros((*gray.shape, 4), dtype=np.uint16)
    inks[..., 0] = 65535 - gray
    path = tmp_path / "old cmyk µ.tif"
    tifffile.imwrite(path, inks, photometric="separated")
    with path.open("rb") as f:
        image = _before_207(s, f, "old cmyk µ.tif")

    def before_131(draft: Project) -> None:
        draft.batch.find_image(image).import_warnings = [
            ImageWarning(code="color_channels_differ", message="the channels differ")
        ]

    project, _ = apply_change(s.project, before_131)
    s._commit(project, action="plant", params={}, evict=[image])
    refused = _refused(s, image, "green")
    assert refused.code is ErrorCode.UNREADABLE_IMAGE
    assert str(refused).startswith(
        "'old cmyk µ.tif' is a CMYK file imported before Proteia converted CMYK colours"
    )


def test_the_channel_measured_already_is_no_change(tmp_path):
    s = session_on(tmp_path, Recorder())
    image = import_pixels(s, colour_blot(), "GFP µ.tif")
    six_boxes(s, image)
    project, actions = s.project, list(s.autosave.actions)
    for given in ("mean", MEAN):  # no choice: the mean
        ops.set_channel(s, image, given)
        assert s.project is project and s.autosave.actions == actions
    ops.set_channel(s, image, "green")
    project, actions = s.project, list(s.autosave.actions)
    ops.set_channel(s, image, "green")
    assert s.project is project and s.autosave.actions == actions


def test_set_channel_is_an_operation():
    assert "set_channel" in ops.__all__


# --- Undo and redo, a polarity change, requantify ---


def test_undo_of_a_channel_change_measures_new_boxes_on_the_restored_channel(tmp_path):
    s = session_on(tmp_path)
    rgb = colour_blot(crosstalk=0.1)
    image = import_pixels(s, rgb, "GFP µ.tif")
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(6)])
    protein = ops.add_protein(s, "GFP α", Role.TARGET, image, box_size=BOX)
    for lane in range(5):
        ops.place_box(s, protein, LANES[lane], ROW, lane_index=lane, grow=False)
    ops.set_channel(s, image, GREEN)
    ops.undo(s)
    assert stored(s, image).channel is None
    mean = rgb.mean(axis=-1)
    ops.place_box(s, protein, LANES[5], ROW, lane_index=5, grow=False)
    assert [b.net for b in boxes_of(s, protein).bands] == expected_nets(s, image, mean)
    assert_nets_current(s)
    # And redo: the green channel again, for what is placed after it.
    ops.undo(s)
    ops.redo(s)
    ops.set_channel(s, image, GREEN)
    ops.undo(s)
    ops.redo(s)
    assert stored(s, image).measured_channel is GREEN
    band = boxes_of(s, protein).bands[-1]
    ops.remove_box(s, band.id)
    ops.place_box(s, protein, LANES[5], ROW, lane_index=5, grow=False)
    assert [b.net for b in boxes_of(s, protein).bands] == expected_nets(s, image, rgb[..., 1])
    assert_nets_current(s)


def test_a_polarity_change_keeps_the_chosen_channel(tmp_path):
    s = session_on(tmp_path)
    rgb = colour_blot(crosstalk=0.1)
    image = import_pixels(s, rgb, "GFP µ.tif", DARK)  # the wrong polarity chosen
    protein, _ = six_boxes(s, image)
    ops.set_channel(s, image, GREEN)
    ops.set_polarity(s, image, LIGHT)
    after = stored(s, image)
    assert after.channel == ChannelChoice(channel=GREEN, source=ChannelSource.CHOSEN)
    assert after.background == float(np.median(rgb[..., 1]))
    assert [b.net for b in boxes_of(s, protein).bands] == expected_nets(s, image, rgb[..., 1])
    assert flags(s, protein) == [(False, None)] * 6
    assert s.project.log[-1].params == {
        "image_id": image,
        "polarity": "light_on_dark",
        "dropped_undetected": [],
    }
    assert_nets_current(s)


def test_the_polarity_change_assesses_looks_processed_on_the_chosen_channel(tmp_path):
    # The array the polarity change assesses is the channel's, with its trust.
    s = session_on(tmp_path)
    rgb = colour_blot(floor=1 / 255, noise=0)
    rgb[..., 0] = 0
    rgb[..., 2] = 1
    image = import_pixels(s, rgb, "GFP µ.tif", DARK)
    ops.set_channel(s, image, GREEN)
    assert [w.code for w in stored(s, image).import_warnings] == ["color_channels_differ"]
    ops.set_polarity(s, image, LIGHT)
    assert [w.code for w in stored(s, image).import_warnings] == ["color_channels_differ"]
    ops.set_channel(s, image, MEAN)
    codes = [w.code for w in stored(s, image).import_warnings]
    assert codes == ["color_channels_differ", "looks_processed"]


def test_requantify_measures_on_the_chosen_channel(tmp_path):
    from conftest import as_legacy

    s = session_on(tmp_path)
    rgb = colour_blot(crosstalk=0.1)
    image = import_pixels(s, rgb, "GFP µ.tif")
    protein, _ = six_boxes(s, image)
    ops.set_channel(s, image, GREEN)
    s._commit(as_legacy(s.project), action="plant", params={})
    assert ops.requantify(s) == (image,)
    assert [b.net for b in boxes_of(s, protein).bands] == expected_nets(s, image, rgb[..., 1])
    assert flags(s, protein)[SATURATED] == (False, None)
    assert_nets_current(s)


def test_unassessed_images_read_the_channel_measured(tmp_path):
    # A band with neither flag on a lossless colour image: assessed near the
    # limit only while it is measured on the mean.
    s = session_on(tmp_path)
    image = import_pixels(s, colour_blot(), "GFP µ.tif")
    protein, _ = six_boxes(s, image)

    def unassessed(draft: Project) -> None:
        for band in draft.batch.boxes_of(draft.batch.find_protein(protein)).bands:
            band.clipped = band.possibly_clipped = None

    plant(s, unassessed)
    assert ops.unassessed_images(s.project.batch) == [image]
    ops.set_channel(s, image, GREEN)
    plant(s, unassessed)
    assert ops.unassessed_images(s.project.batch) == []


def test_row_detection_counts_saturation_on_the_chosen_channel(tmp_path, monkeypatch):
    s = session_on(tmp_path)
    image = import_pixels(s, colour_blot(saturate=True, crosstalk=0.1), "GFP µ.tif")
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(6)])
    protein = ops.add_protein(s, "GFP α", Role.TARGET, image)
    given: list = []
    detect = ops.rowdetect.detect_row

    def recording(*args, **kwargs):
        given.append(kwargs["saturated_at"])
        return detect(*args, **kwargs)

    monkeypatch.setattr(ops.rowdetect, "detect_row", recording)
    row = (5, ROW - 12, W - 5, ROW + 12)
    with contextlib.suppress(OperationError):
        ops.detect_row_boxes(s, protein, row)
    ops.set_channel(s, image, GREEN)
    with contextlib.suppress(OperationError):
        ops.detect_row_boxes(s, protein, row)
    assert given == [253.0, 255.0]  # near the mean's limit; at green's


def test_a_seed_click_grows_on_the_chosen_channel(tmp_path):
    # Red holds a wide haze around lane 3's band: grown on the mean, its box
    # would be wider than green's band.
    rgb = colour_blot()
    yy, xx = np.mgrid[0:H, 0:W]
    haze = 120 * np.exp(-(((xx - LANES[2]) / 14.0) ** 2 + ((yy - ROW) / 8.0) ** 2))
    rgb[..., 0] = np.clip(rgb[..., 0] + np.rint(haze), 0, 255).astype(np.uint8)
    s = session_on(tmp_path)
    image = import_pixels(s, rgb, "GFP µ.tif")
    ops.set_channel(s, image, GREEN)
    ops.set_lanes(s, [LaneInput(f"c{i}") for i in range(6)])
    protein = ops.add_protein(s, "GFP α", Role.TARGET, image)
    band_id = ops.place_box(s, protein, LANES[2], ROW, lane_index=2, grow=True)
    green = rgb[..., 1].astype(np.float64)
    grown = grow_box(
        green,
        (LANES[2], ROW),
        float(np.median(green)),
        rel_threshold=REL_THRESHOLD,
        noise_k=NOISE_K,
        dark_on_light=False,
    )
    own = boxes_of(s, protein)
    band = next(b for b in own.bands if b.id == band_id)
    assert band.box.rect(own.box_size) == grown
    mean_grown = grow_box(
        rgb.mean(axis=-1),
        (LANES[2], ROW),
        float(np.median(rgb.mean(axis=-1))),
        rel_threshold=REL_THRESHOLD,
        noise_k=NOISE_K,
        dark_on_light=False,
    )
    assert grown != mean_grown  # the two channels give different boxes here


# --- An older project ---


def test_a_project_saved_before_206_opens_and_is_measured_as_before(tmp_path):
    # The frozen F2 holds a colour PNG (tinted gray): no choice, measured on
    # the mean with the near-limit check, whatever is re-measured.
    folder = tmp_path / "f2 µ"
    fixtures.copy_project(DATA / "f2", folder)
    s = ops.open_project(folder, autosave=None, clock=FakeClock())
    batch = s.project.batch
    colour = [i for i in batch.iter_images() if not i.channel_blocker]
    assert [i.channel for i in colour] == [None] and colour[0].measured_channel is MEAN
    project = s.project
    assert ops.requantify(s) == ()
    assert s.project is project
    on = [p for p in batch.proteins_on(colour[0].id) if batch.boxes_of(p).bands]
    band = batch.boxes_of(on[0]).bands[0]
    x0, y0, x1, y1 = band.box.rect(batch.boxes_of(on[0]).box_size)
    ops.move_box(s, band.id, (x0 + 1, y0, x1 + 1, y1))
    assert stored(s, colour[0].id).channel is None
    assert all(
        b.clipped is None and b.possibly_clipped is not None
        for p in s.project.batch.proteins_on(colour[0].id)
        for b in s.project.batch.boxes_of(p).bands
    )
    assert_nets_current(s)  # on the mean, with the mean's checks
    pixels = imaging.read_file(storage.image_path(folder, colour[0])).pixels
    assert np.array_equal(s.pixels(colour[0].id), to_grayscale(pixels).astype(np.float64))


@pytest.mark.parametrize("case", ["f1", "f2", "f3"])
def test_a_project_without_a_choice_saves_as_before(case, tmp_path):
    folder = tmp_path / case
    fixtures.copy_project(DATA / case, folder)
    s = ops.open_project(folder, autosave=None, clock=FakeClock())
    data = storage.project_to_json(s.project)
    assert all("channel" not in image for image in _images(data))
    assert s.project.log[-1].content_hash == storage.content_hash(s.project)
    s.save()
    reopened = ops.open_project(folder, autosave=None, clock=FakeClock())
    assert storage.project_to_json(reopened.project) == data


def _images(data: bytes) -> list[dict]:
    import json

    return json.loads(data)["batch"]["images"]


def test_a_chosen_channel_is_saved_and_read_back(tmp_path):
    s = session_on(tmp_path, save_to_folder)
    rgb = colour_blot(crosstalk=0.1)
    image = import_pixels(s, rgb, "GFP µ.tif")
    protein, _ = six_boxes(s, image)
    ops.set_channel(s, image, BLUE)
    images = _images((s.folder / storage.PROJECT_FILE).read_bytes())
    assert images[0]["channel"] == {"channel": "blue", "source": "chosen"}
    reopened = ops.open_project(s.folder, autosave=None)
    assert stored(reopened, image).measured_channel is BLUE
    assert np.array_equal(reopened.pixels(image), rgb[..., 2])
    assert_nets_current(reopened)
