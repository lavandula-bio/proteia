# SPDX-License-Identifier: Apache-2.0
"""Tests for reading image files and rendering previews; files are generated here."""

import dataclasses
import functools
import hashlib
import itertools
import os
import re
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms, ImageDraw
from skimage import io

from conftest import synthetic_blot
from proteia import samples
from proteia.core import imaging
from proteia.core.imaging import (
    UNTRUSTED_WARNINGS,
    WARNINGS,
    LoadedImage,
    UnsupportedColourSpaceError,
    _Declared,
    _read_tiff_with_pillow,
    assess_processed,
    clipping_depth,
    converts_cmyk,
    display_rgb,
    file_colours,
    from_pixels,
    load_image,
    possible_clipping_depth,
    preview,
    read_colours,
    read_pixels,
    reads_as_palette,
    to_analysis_array,
)
from proteia.core.model import ImageKind, ImageRef, ImageWarning, Polarity, Project
from proteia.core.quantify import estimate_background, near_limit_tolerance
from proteia.core.storage import load_project, save_project, store_image


def _gray(dtype, top: int) -> np.ndarray:
    ramp = np.linspace(0, top, 6 * 8).reshape(6, 8)
    return np.round(ramp).astype(dtype)


def _codes(loaded) -> list[str]:
    return [w.code for w in loaded.warnings]


@pytest.mark.parametrize(("dtype", "top", "depth"), [(np.uint8, 255, 8), (np.uint16, 65535, 16)])
def test_grayscale_tiff_keeps_values_and_records_bit_depth(tmp_path, dtype, top, depth):
    pixels = _gray(dtype, top)
    path = tmp_path / f"β-actin {depth}-bit µ.tif"
    tifffile.imwrite(path, pixels)
    loaded = load_image(path)
    assert loaded.bit_depth == depth
    assert loaded.array.dtype == np.float64
    np.testing.assert_array_equal(loaded.array, pixels.astype(np.float64))
    assert loaded.array.max() == top  # the original scale, not rescaled to 0-255
    assert (loaded.height, loaded.width) == (6, 8)
    assert loaded.warnings == []


@pytest.mark.parametrize(("dtype", "top", "depth"), [(np.uint8, 255, 8), (np.uint16, 65535, 16)])
def test_rgb_tiff_with_equal_channels_is_its_gray_channel(tmp_path, dtype, top, depth):
    gray = _gray(dtype, top)
    path = tmp_path / "rgb.tif"
    tifffile.imwrite(path, np.stack([gray, gray, gray], axis=-1), photometric="rgb")
    loaded = load_image(path)
    assert loaded.bit_depth == depth
    np.testing.assert_array_equal(loaded.array, gray.astype(np.float64))
    assert loaded.warnings == []


def test_rgb_with_different_channels_is_averaged_with_a_warning(tmp_path):
    # 16-bit values stay in their own scale: (60000 + 30000 + 0) / 3 = 30000.
    rgb = np.zeros((4, 5, 3), dtype=np.uint16)
    rgb[..., 0], rgb[..., 1] = 60000, 30000
    path = tmp_path / "color α.tif"
    tifffile.imwrite(path, rgb, photometric="rgb")
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.array, np.full((4, 5), 30000.0))
    assert loaded.bit_depth == 16
    assert _codes(loaded) == ["color_channels_differ"]


def test_alpha_channel_is_ignored():
    rgba = np.zeros((3, 3, 4), dtype=np.uint8)
    rgba[..., :3] = 90
    rgba[..., 3] = 255
    array, warnings = to_analysis_array(rgba)
    np.testing.assert_array_equal(array, np.full((3, 3), 90.0))
    assert warnings == []


@pytest.mark.parametrize("suffix", [".jpg", ".JPEG", ".jpe"])
def test_jpeg_import_records_a_lossy_format_warning(tmp_path, suffix):
    path = tmp_path / f"blot β{suffix}"
    io.imsave(path, _gray(np.uint8, 255), check_contrast=False)
    loaded = load_image(path)
    assert loaded.bit_depth == 8
    assert _codes(loaded) == ["lossy_format"]


def test_png_loads_without_warnings(tmp_path):
    path = tmp_path / "blot.png"
    pixels = _gray(np.uint16, 65535)
    io.imsave(path, pixels, check_contrast=False)
    loaded = load_image(path)
    assert loaded.bit_depth == 16
    np.testing.assert_array_equal(loaded.array, pixels.astype(np.float64))
    assert loaded.warnings == []


def test_float_tiff_has_no_bit_depth_and_a_warning(tmp_path):
    path = tmp_path / "float.tif"
    tifffile.imwrite(path, np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4))
    loaded = load_image(path)
    assert loaded.bit_depth is None
    assert _codes(loaded) == ["unknown_bit_depth"]


def test_multi_page_tiff_is_refused(tmp_path):
    path = tmp_path / "stack.tif"
    stack = np.zeros((3, 6, 8), dtype=np.uint16)
    with tifffile.TiffWriter(path) as tif:
        for page in stack:
            tif.write(page)
    with pytest.raises(ValueError, match="stacks"):
        read_pixels(path)


@pytest.mark.parametrize("shape", [(4, 5, 5), (4, 5, 6), (2, 4, 5, 3)])
def test_unsupported_layout_is_refused(shape):
    with pytest.raises(ValueError, match="unsupported image layout"):
        to_analysis_array(np.zeros(shape, dtype=np.uint8))


def test_preview_keeps_16_bit_levels_above_255_apart():
    levels = np.array([[1000, 2000, 3000, 60000]], dtype=np.uint16)
    view = preview(levels)
    assert view.dtype == np.uint8
    assert len(set(view.ravel().tolist())) == 4  # distinct levels stay distinct
    assert view.min() == 0 and view.max() == 255


def test_preview_shows_8_bit_data_as_stored_and_keeps_layout():
    rgb = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    view = preview(rgb)
    np.testing.assert_array_equal(view, rgb)
    assert view is not rgb  # a copy, never the stored pixels
    assert preview(np.zeros((2, 2), dtype=np.uint16)).tolist() == [[0, 0], [0, 0]]


def test_import_warnings_survive_save_and_load(tmp_path):
    source = tmp_path / "source" / "blot µ.jpg"
    source.parent.mkdir()
    io.imsave(source, _gray(np.uint8, 255), check_contrast=False)
    loaded = load_image(source)
    folder = tmp_path / "project"
    with source.open("rb") as f:
        stored = store_image(folder, "img-1", source.name, f)
    assert stored.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    image = ImageRef(
        id="img-1",
        file=stored.file,
        original_name=source.name,
        kind=ImageKind.CHEMILUMINESCENCE,
        sha256=stored.sha256,
        width=loaded.width,
        height=loaded.height,
        bit_depth=loaded.bit_depth,
        polarity=Polarity.DARK_ON_LIGHT,
        background=float(np.median(loaded.array)),
        import_warnings=loaded.warnings,
    )
    project = Project.model_validate(
        {"next_id": 3, "batch": {"membranes": [{"id": "mem-2", "images": [image]}]}}
    )
    save_project(project, folder)
    reloaded = load_project(folder)
    assert reloaded == project
    assert [w.code for w in reloaded.batch.find_image("img-1").import_warnings] == ["lossy_format"]


def test_tiff_with_a_thumbnail_page_is_one_image(tmp_path):
    path = tmp_path / "scan with thumbnail.tif"
    main = _gray(np.uint16, 65535)
    with tifffile.TiffWriter(path) as tif:
        tif.write(main)
        tif.write(main[::2, ::2], subfiletype=1)  # reduced-resolution thumbnail
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.array, main.astype(np.float64))
    assert loaded.bit_depth == 16


def test_planar_rgb_tiff_is_read_channels_last(tmp_path):
    gray = _gray(np.uint8, 255)
    rgb = np.stack([gray, gray // 2, gray // 4], axis=-1)
    path = tmp_path / "planar.tif"
    # A planar file stores the channels first; tifffile writes (S, Y, X) input as is.
    tifffile.imwrite(path, np.moveaxis(rgb, -1, 0), photometric="rgb", planarconfig="separate")
    pixels = read_pixels(path)
    assert pixels.shape == (6, 8, 3)
    np.testing.assert_array_equal(pixels, rgb)


def test_gray_plus_alpha_keeps_the_gray_channel():
    la = np.zeros((3, 3, 2), dtype=np.uint8)
    la[..., 0], la[..., 1] = 70, 255
    array, warnings = to_analysis_array(la)
    np.testing.assert_array_equal(array, np.full((3, 3), 70.0))
    assert warnings == []


def test_non_finite_pixels_are_refused():
    with pytest.raises(ValueError, match="NaN or infinite"):
        to_analysis_array(np.array([[0.0, 1.0, np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or infinite"):
        to_analysis_array(np.array([[0.0, np.inf]]))


def test_preview_ignores_nan_pixels():
    view = preview(np.array([[0.0, 1.0, np.nan]], dtype=np.float32))
    assert view.tolist() == [[0, 255, 0]]


def test_pixels_in_memory_follow_the_same_rules():
    loaded = from_pixels(np.full((4, 6), 20.0))  # float, like a generated demo image
    assert loaded.bit_depth is None
    assert _codes(loaded) == ["unknown_bit_depth"]
    assert _codes(from_pixels(_gray(np.uint8, 255), lossy=True)) == ["lossy_format"]


def test_every_warning_that_turns_the_clipping_check_off_says_so(tmp_path):
    # #112: the over-exposure check does not run on these images
    # (clipping_depth), and each warning tells the user so.
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    cmyk = tmp_path / "gray in CMYK.tif"
    Image.fromarray(_gray(np.uint8, 255)).convert("CMYK").save(cmyk)
    loaded = [
        from_pixels(_gray(np.uint8, 255), lossy=True),
        from_pixels(rgb),
        from_pixels(np.full((4, 6), 20.0)),
        load_image(cmyk),
    ]
    warnings = [warning for image in loaded for warning in image.warnings]
    assert [w.code for w in warnings] == [
        "lossy_format",
        "color_channels_differ",
        "unknown_bit_depth",
        "cmyk_converted",
    ]
    for image in loaded:
        assert clipping_depth(image.bit_depth, image.warnings) is None
    for warning in warnings:
        assert "over-exposure cannot be checked" in warning.message, warning.code
    path = tmp_path / "blot β.jpg"
    io.imsave(path, _gray(np.uint8, 255), check_contrast=False)
    (warning,) = load_image(path).warnings
    assert warning.message == (
        "JPEG-type compression can change pixel values, so over-exposure cannot be"
        " checked; quantify an uncompressed or losslessly compressed original if you have it."
    )
    assert warnings[1].message == (
        "The red, green and blue channels differ; they were averaged into one gray"
        " channel, so over-exposure cannot be checked."
    )


def test_possible_over_exposure_is_assessed_only_where_the_exact_check_cannot_run():
    # #112: on an image of known bit depth that clipping_depth distrusts, the
    # heuristic measures against the limit of that depth; nowhere else.
    def warning(code: str) -> ImageWarning:
        return ImageWarning(code=code, message=WARNINGS[code])

    for depth in (8, 16):
        assert possible_clipping_depth(depth, []) is None  # the exact check runs
        assert clipping_depth(depth, []) == depth
        for code in UNTRUSTED_WARNINGS:
            assert possible_clipping_depth(depth, [warning(code)]) == depth, code
            assert clipping_depth(depth, [warning(code)]) is None, code
        both = [warning("lossy_format"), warning("color_channels_differ")]
        assert possible_clipping_depth(depth, both) == depth
    # An unknown bit depth has no limit: neither check runs.
    for warnings in ([], [warning("unknown_bit_depth")], [warning("lossy_format")]):
        assert possible_clipping_depth(None, warnings) is None
        assert clipping_depth(None, warnings) is None


def test_a_converted_cmyk_file_is_assessed_against_its_rgb_limit(tmp_path):
    # Gray saved as CMYK converts back exactly by the formula: black stays 0,
    # so the heuristic's limit is the converted colours' own 0.
    path = tmp_path / "gray in CMYK.tif"
    Image.fromarray(_gray(np.uint8, 255)).convert("CMYK").save(path)
    loaded = load_image(path)
    assert (loaded.bit_depth, _codes(loaded)) == (8, ["cmyk_converted"])
    assert possible_clipping_depth(loaded.bit_depth, loaded.warnings) == 8
    assert loaded.array.min() == 0.0 and loaded.array.max() == 255.0


def test_jpeg_compressed_tiff_records_a_lossy_format_warning(tmp_path):
    path = tmp_path / "jpeg inside.tif"
    Image.fromarray(_gray(np.uint8, 255)).save(path, compression="jpeg")
    loaded = load_image(path)
    assert loaded.bit_depth == 8
    assert _codes(loaded) == ["lossy_format"]


def test_multi_channel_composite_is_refused(tmp_path):
    path = tmp_path / "composite.tif"
    data = np.zeros((3, 6, 8), dtype=np.uint16)
    tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "CYX"})
    with pytest.raises(ValueError, match="composites"):
        read_pixels(path)


@pytest.mark.parametrize(
    ("name", "data"),
    [("truncated.tif", b"II*\x00"), ("truncated.jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")],
)
def test_damaged_file_is_a_value_error(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises((ValueError, OSError)):
        load_image(path)


def _rgb8() -> np.ndarray:
    gray = _gray(np.uint8, 255)
    return np.stack([gray, gray // 2, gray // 3], axis=-1)


@pytest.mark.parametrize(
    ("pixels", "depth", "compression"),
    [
        (_gray(np.uint16, 65535), 16, "tiff_lzw"),
        (_gray(np.uint8, 255), 8, "tiff_lzw"),
        (_gray(np.uint16, 65535).astype(">u2"), 16, "tiff_lzw"),
        (_rgb8(), 8, "tiff_lzw"),
    ],
    ids=["lzw-16-bit", "lzw-8-bit", "lzw-16-bit-big-endian", "lzw-rgb"],
)
def test_compressed_tiff_is_read_exactly(tmp_path, pixels, depth, compression):
    # tifffile needs imagecodecs for LZW; Pillow decodes it with the same values.
    path = tmp_path / f"{compression} µ.tif"
    Image.fromarray(pixels).save(path, compression=compression)
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.pixels, pixels)  # values, whatever the byte order
    assert loaded.pixels.dtype.isnative
    assert loaded.bit_depth == depth


def _declared(photometric, samples, shape=(6, 8), dtype=np.uint8) -> _Declared:
    return _Declared("LZW", shape, np.dtype(dtype), photometric, samples)


@pytest.mark.parametrize(
    "declared",
    [
        # Pillow would change these values: it inverts white-is-zero gray and
        # un-premultiplies associated alpha.
        _declared(tifffile.PHOTOMETRIC.MINISWHITE, 1),
        _declared(tifffile.PHOTOMETRIC.RGB, 4, (6, 8, 4)),
        _declared(tifffile.PHOTOMETRIC.PALETTE, 1),
        # A result that differs from the declaration: never read with another scale.
        _declared(tifffile.PHOTOMETRIC.MINISBLACK, 1, dtype=np.uint16),
        _declared(tifffile.PHOTOMETRIC.RGB, 3, (6, 8, 3)),
    ],
    ids=["min-is-white", "rgba", "palette", "other-bit-depth", "other-shape"],
)
def test_pillow_is_used_only_where_it_reads_the_same_values(tmp_path, declared):
    path = tmp_path / "lzw.tif"
    Image.fromarray(_gray(np.uint8, 255)).save(path, compression="tiff_lzw")
    with pytest.raises(ValueError, match="uncompressed TIFF"):
        _read_tiff_with_pillow(path, declared)


def test_a_compressed_tiff_over_pillows_size_limit_says_so(tmp_path, monkeypatch):
    path = tmp_path / "large lzw.tif"
    Image.fromarray(_gray(np.uint16, 65535)).save(path, compression="tiff_lzw")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)  # 48 pixels is over twice the limit
    with pytest.raises(ValueError, match="more pixels than the compressed-TIFF reader"):
        load_image(path)


def test_display_rgb_views():
    gray16 = np.array([[0, 65535]], dtype=np.uint16)
    assert display_rgb(gray16).tolist() == [[[0, 0, 0], [255, 255, 255]]]
    la = np.zeros((1, 2, 2), dtype=np.uint8)
    la[..., 0], la[..., 1] = [10, 20], 255
    assert display_rgb(la).tolist() == [[[10, 10, 10], [20, 20, 20]]]
    rgba = np.zeros((1, 1, 4), dtype=np.uint8)
    rgba[..., :3], rgba[..., 3] = [1, 2, 3], 255
    assert display_rgb(rgba).tolist() == [[[1, 2, 3]]]


# --- A file's own colours, for display only (#57) ---


def _palette_map(gray: bool = False) -> np.ndarray:
    """A 16-bit TIFF colour map (3 x 256): a gray ramp, or a red-to-yellow one."""
    level = np.arange(256, dtype=np.uint16)
    if gray:
        return np.stack([level * 257] * 3)
    return np.stack([level * 257, level * 128, (255 - level) * 64])


def _write(path, pixels, **options):
    if path.suffix == ".tif":
        tifffile.imwrite(path, pixels, **options)
    else:
        (pixels if isinstance(pixels, Image.Image) else Image.fromarray(pixels)).save(
            path, **options
        )
    return path


def _palette_png() -> Image.Image:
    image = Image.fromarray(_gray(np.uint8, 2), mode="L").convert("P")
    image.putpalette([200, 30, 40, 250, 220, 225, 120, 10, 20] + [0] * (253 * 3))
    return image


@pytest.mark.parametrize(
    ("name", "pixels", "options", "expected"),
    [
        ("rgb.tif", _rgb8(), {}, "rgb"),
        ("rgb lzw.tif", Image.fromarray(_rgb8()), {"compression": "tiff_lzw"}, "rgb"),
        ("rgba.tif", np.dstack([_rgb8(), _gray(np.uint8, 255)]), {}, "rgb"),
        (
            "palette.tif",
            _gray(np.uint8, 255),
            {"photometric": "palette", "colormap": _palette_map()},
            "palette",
        ),
        (
            "gray map.tif",
            _gray(np.uint8, 255),
            {"photometric": "palette", "colormap": _palette_map(gray=True)},
            None,
        ),
        ("gray.tif", _gray(np.uint16, 65535), {}, None),
        # Three gray planes are not red, green and blue.
        ("planes.tif", _rgb8(), {"photometric": "minisblack", "planarconfig": "contig"}, None),
        ("rgb.png", _rgb8(), {}, "rgb"),
        ("palette.png", _palette_png(), {}, "rgb"),
        ("gray.png", _gray(np.uint8, 255), {}, None),
        ("rgb.jpg", _rgb8(), {"quality": 95}, "rgb"),
        # CMYK is read converted to red, green and blue (#131).
        ("cmyk.jpg", Image.fromarray(_rgb8()).convert("CMYK"), {"quality": 95}, "rgb"),
        ("cmyk.tif", Image.fromarray(_rgb8()).convert("CMYK"), {}, "rgb"),
        # CIELAB is refused on import: it has no colours to show.
        ("lab.tif", _rgb8(), {"photometric": "cielab"}, None),
    ],
)
def test_file_colours_say_from_the_header_how_a_file_holds_colour(
    tmp_path, name, pixels, options, expected
):
    path = tmp_path / f"µ {name}"
    if isinstance(pixels, Image.Image) and path.suffix == ".tif":
        pixels.save(path, **options)  # LZW, or CMYK as a separated TIFF
    else:
        _write(path, pixels, **options)
    assert file_colours(path) == expected


def test_file_colours_of_a_damaged_file_is_a_value_error(tmp_path):
    for name in ("damaged.tif", "damaged.png"):
        path = tmp_path / name
        path.write_bytes(b"not an image")
        with pytest.raises((ValueError, OSError)):
            file_colours(path)


def test_read_colours_looks_a_palette_tiff_up_in_its_colour_map(tmp_path):
    indices = _gray(np.uint8, 255)
    colormap = _palette_map()
    path = _write(tmp_path / "fire.tif", indices, photometric="palette", colormap=colormap)
    np.testing.assert_array_equal(read_pixels(path), indices)  # what analysis reads
    shown = read_colours(path)
    assert (shown.shape, shown.dtype) == ((6, 8, 3), np.uint8)
    np.testing.assert_array_equal(shown[..., 0], indices)  # the high byte of level * 257
    np.testing.assert_array_equal(shown[..., 1], indices // 2)
    np.testing.assert_array_equal(shown[..., 2], (255 - indices) // 4)
    # Anything else reads as read_pixels reads it: CMYK converted.
    cmyk = _write(tmp_path / "cmyk.jpg", Image.fromarray(_rgb8()).convert("CMYK"), quality=95)
    np.testing.assert_array_equal(read_colours(cmyk), read_pixels(cmyk))
    assert read_colours(cmyk).shape == (6, 8, 3)
    for name, pixels in (("rgb.tif", _rgb8()), ("gray.png", _gray(np.uint8, 255))):
        other = _write(tmp_path / name, pixels)
        np.testing.assert_array_equal(read_colours(other), pixels)


# --- CMYK and colour spaces other than gray and RGB (#131) ---

# A CMYK press profile Windows ships; other systems may have none.
RSWOP = Path(os.environ.get("SystemRoot", "C:/Windows"), "System32/spool/drivers/color/RSWOP.icm")
SRGB = ImageCms.createProfile("sRGB")


def _blot_rgb() -> np.ndarray:
    """The blot of #131: a light membrane (RGB 230) with a dark band (RGB 40),
    and a first row of colours, so that the channels differ."""
    rgb = np.full((6, 8, 3), 230, dtype=np.uint8)
    rgb[2:5, 2:6] = 40
    rgb[0] = _rgb8()[3]
    return rgb


def _inks(rgb: np.ndarray) -> np.ndarray:
    """CMYK inks that the standard formula reads back as ``rgb`` exactly (8- or
    16-bit): no black ink, and C = top - R and so on, as Pillow converts RGB."""
    top = np.iinfo(rgb.dtype).max
    return np.dstack([top - rgb, np.zeros(rgb.shape[:2], dtype=rgb.dtype)])


def _s15(value: float) -> bytes:
    return struct.pack(">i", round(value * 65536))


def _cmyk_profile() -> bytes:
    """A minimal ICC v2 profile for CMYK input, built here since Pillow builds
    none: CMYK to CIELAB through a 2-point table, gray only, L* falling with the
    inks (black the most). No library or file ships one on every system."""
    table = []
    for c, m, y, k in itertools.product((0, 1), repeat=4):  # the first ink varies slowest
        lightness = (1 - k) * (1 - 0.25 * (c + m + y))
        table += [round(lightness * 0xFF00), 0x8000, 0x8000]  # v2 16-bit L*, a* = b* = 0
    identity = b"".join(_s15(v) for v in (1, 0, 0, 0, 1, 0, 0, 0, 1))
    a2b0 = b"".join(
        [
            b"mft2",
            bytes(4),
            bytes([4, 3, 2, 0]),  # 4 inks in, 3 out, 2 grid points
            identity,
            struct.pack(">HH", 2, 2),
            struct.pack(">8H", *[0, 0xFFFF] * 4),
            struct.pack(f">{len(table)}H", *table),
            struct.pack(">6H", *[0, 0xFFFF] * 3),
        ]
    )
    name = b"CMYK test\x00"
    desc = b"desc" + bytes(4) + struct.pack(">I", len(name)) + name + bytes(78)
    d50 = _s15(0.9642) + _s15(1.0) + _s15(0.8249)
    tags = [(b"desc", desc), (b"wtpt", b"XYZ " + bytes(4) + d50), (b"A2B0", a2b0)]
    start = 128 + 4 + 12 * len(tags)
    directory, data = struct.pack(">I", len(tags)), b""
    for signature, body in tags:
        body += bytes(-len(body) % 4)
        directory += signature + struct.pack(">II", start + len(data), len(body))
        data += body
    header = b"".join(
        [
            struct.pack(">I", start + len(data)),
            bytes(4),
            struct.pack(">I", 0x02100000),
            b"scnrCMYKLab ",
            bytes(12),
            b"acsp",
            bytes(28),
            d50,
            bytes(48),
        ]
    )
    return header + directory + data


def _through(image: Image.Image, profile: bytes) -> np.ndarray:
    """``image``'s CMYK in sRGB through ``profile``, as a colour-managed reader converts it."""
    return np.asarray(
        ImageCms.profileToProfile(
            image,
            ImageCms.ImageCmsProfile(BytesIO(profile)),
            SRGB,
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
    )


def _write_cmyk(path: Path, rgb: np.ndarray, writer: str, **options) -> Path:
    if writer == "pillow":
        Image.fromarray(rgb).convert("CMYK").save(path, **options)
    elif writer == "tifffile":
        tifffile.imwrite(path, _inks(rgb), photometric="separated", **options)
    else:  # planar: the inks stored one plane after another
        planes = np.moveaxis(_inks(rgb), -1, 0)
        tifffile.imwrite(path, planes, photometric="separated", planarconfig="separate")
    return path


@pytest.mark.parametrize(
    ("writer", "options"),
    [
        ("pillow", {}),
        ("pillow", {"compression": "tiff_lzw"}),  # decoded by Pillow, not tifffile
        ("tifffile", {}),
        ("planar", {}),
    ],
    ids=["pillow", "pillow-lzw", "tifffile", "planar"],
)
def test_a_cmyk_tiff_is_converted_to_red_green_and_blue_then_gray(tmp_path, writer, options):
    # Read as if C, M and Y were R, G and B, the membrane was 25 and the band 215.
    rgb = _blot_rgb()
    path = _write_cmyk(tmp_path / "cmyk µ.tif", rgb, writer, **options)
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.pixels, rgb)  # exact: no black ink, no profile
    np.testing.assert_array_equal(loaded.array, rgb.mean(axis=-1))
    assert (loaded.array[5, 0], loaded.array[3, 3]) == (230, 40)
    assert loaded.bit_depth == 8
    assert _codes(loaded) == ["cmyk_converted", "color_channels_differ"]
    converted, differ = loaded.warnings
    assert converted.message == (
        "The file's colours are CMYK: they were converted to red, green and blue with"
        " the standard formula, as it embeds no ICC colour profile, then to gray. Values"
        " from a converted file are approximate, so over-exposure cannot be checked."
    )
    assert differ.message == (
        "The red, green and blue channels converted from the file's CMYK differ; they"
        " were averaged into one gray channel, so over-exposure cannot be checked."
    )
    assert clipping_depth(loaded.bit_depth, loaded.warnings) is None
    np.testing.assert_array_equal(read_pixels(path), rgb)
    np.testing.assert_array_equal(read_colours(path), rgb)  # its colours, converted


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_the_black_ink_of_a_cmyk_tiff_is_not_dropped(tmp_path, dtype):
    # Gray printed with black ink alone: C, M and Y are 0 everywhere, so the
    # mean of the first three channels read the whole image as 0.
    top = np.iinfo(dtype).max
    gray = _gray(dtype, top)
    inks = np.zeros((*gray.shape, 4), dtype=dtype)
    inks[..., 3] = top - gray
    path = tmp_path / "black ink.tif"
    tifffile.imwrite(path, inks, photometric="separated")
    loaded = load_image(path)
    assert loaded.pixels.dtype == dtype  # 16-bit inks keep their scale
    assert loaded.bit_depth == (16 if dtype is np.uint16 else 8)
    np.testing.assert_array_equal(loaded.array, gray.astype(np.float64))
    assert _codes(loaded) == ["cmyk_converted"]  # equal channels: gray


def _formula(inks: np.ndarray) -> np.ndarray:
    """R = (1 - C)(1 - K) and so on, on inks scaled to 1, rounded to the nearest
    level (a product over an odd top level never falls on a half)."""
    top = np.iinfo(inks.dtype).max
    white = top - inks.astype(np.float64)
    return np.floor(white[..., :3] * white[..., 3:] / top + 0.5).astype(inks.dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_the_standard_formula_takes_every_ink_into_account(tmp_path, dtype):
    # Every cyan and black ink level on 8 bits (magenta and yellow vary too),
    # and the same inks on 16 bits; at 8 bits, what Pillow's CMYK to RGB gives.
    level = np.arange(256, dtype=np.uint8)
    cyan, black = np.meshgrid(level, level)
    inks = np.dstack([cyan, black[::-1], cyan + black, black]).astype(np.uint8)
    if dtype is np.uint16:
        inks = inks.astype(np.uint16) * 257
    path = tmp_path / "every level.tif"
    tifffile.imwrite(path, inks, photometric="separated")
    np.testing.assert_array_equal(read_pixels(path), _formula(inks))
    if dtype is np.uint8:
        pillow = Image.frombytes("CMYK", (256, 256), inks.tobytes()).convert("RGB")
        np.testing.assert_array_equal(read_pixels(path), np.asarray(pillow))


def test_a_cmyk_jpeg_is_converted_after_its_adobe_inversion(tmp_path):
    # Pillow writes a CMYK JPEG as Adobe software does, with the inks inverted
    # and an Adobe marker saying so, and reads the inks back the right way up.
    rgb = _blot_rgb()
    path = tmp_path / "cmyk β.jpg"
    Image.fromarray(rgb).convert("CMYK").save(path, quality=95)
    with Image.open(path) as stored:
        assert (stored.mode, "adobe" in stored.info) == ("CMYK", True)
    loaded = load_image(path)
    assert _codes(loaded) == ["lossy_format", "cmyk_converted", "color_channels_differ"]
    assert np.abs(loaded.array - rgb.mean(axis=-1)).max() <= 3  # JPEG's own error at quality 95
    assert loaded.array[5, 0] > 225 and loaded.array[3, 3] < 45
    np.testing.assert_array_equal(read_colours(path), loaded.pixels)
    assert "standard formula" in loaded.warnings[1].message


@pytest.mark.parametrize(
    ("name", "save"),
    [
        ("pillow.tif", lambda path, inks, profile: inks.save(path, icc_profile=profile)),
        (
            "tifffile.tif",
            lambda path, inks, profile: tifffile.imwrite(
                path, np.asarray(inks), photometric="separated", iccprofile=profile
            ),
        ),
        ("β.jpg", lambda path, inks, profile: inks.save(path, quality=95, icc_profile=profile)),
    ],
    ids=["tiff-pillow", "tiff-tifffile", "jpeg"],
)
def test_a_cmyk_file_is_converted_through_the_profile_it_embeds(tmp_path, name, save):
    profile = _cmyk_profile()
    path = tmp_path / name
    save(path, Image.fromarray(_blot_rgb()).convert("CMYK"), profile)
    with Image.open(path) as stored:  # the inks as decoded, JPEG's error included
        expected, formula = _through(stored, profile), np.asarray(stored.convert("RGB"))
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.pixels, expected)
    assert np.abs(expected.astype(int) - formula).max() > 20  # the profile, not the formula
    (converted,) = [w for w in loaded.warnings if w.code == "cmyk_converted"]
    assert "converted to red, green and blue through the ICC colour profile it embeds" in (
        converted.message
    )
    assert clipping_depth(loaded.bit_depth, loaded.warnings) is None


@pytest.mark.parametrize(
    "profile",
    [ImageCms.ImageCmsProfile(SRGB).tobytes(), b"not a profile"],
    ids=["rgb-profile", "damaged-profile"],
)
def test_a_profile_that_cannot_convert_cmyk_leaves_the_standard_formula(tmp_path, profile):
    rgb = _blot_rgb()
    path = _write_cmyk(tmp_path / "profile.tif", rgb, "tifffile", iccprofile=profile)
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.pixels, rgb)
    assert (
        "with the standard formula, as the ICC colour profile it embeds could not be used,"
        in loaded.warnings[0].message
    )


def test_a_16_bit_cmyk_tiff_is_not_converted_through_its_profile(tmp_path):
    # Pillow's colour management takes 8-bit CMYK only: 16-bit inks keep their
    # scale through the standard formula instead.
    rgb = _blot_rgb().astype(np.uint16) * 257
    path = tmp_path / "16-bit.tif"
    tifffile.imwrite(path, _inks(rgb), photometric="separated", iccprofile=_cmyk_profile())
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.pixels, rgb)
    assert loaded.bit_depth == 16
    assert (
        "with the standard formula, as its ICC colour profile is applied to 8-bit CMYK only,"
        in loaded.warnings[0].message
    )


def test_an_rgb_file_with_a_profile_is_read_as_stored(tmp_path):
    # Only CMYK is converted: RGB keeps its values, whatever profile it embeds.
    rgb = _rgb8()
    path = tmp_path / "rgb with profile.tif"
    srgb = ImageCms.ImageCmsProfile(SRGB).tobytes()
    tifffile.imwrite(path, rgb, photometric="rgb", iccprofile=srgb)
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.pixels, rgb)
    assert _codes(loaded) == ["color_channels_differ"]
    assert loaded.warnings[0].message.startswith("The red, green and blue channels differ;")


@pytest.mark.skipif(not RSWOP.is_file(), reason="no CMYK press profile on this system")
@pytest.mark.parametrize(
    ("intent", "flags", "within"),
    [
        (ImageCms.Intent.RELATIVE_COLORIMETRIC, ImageCms.Flags.BLACKPOINTCOMPENSATION, 15),
        (ImageCms.Intent.RELATIVE_COLORIMETRIC, ImageCms.Flags.NONE, 12),
        (ImageCms.Intent.PERCEPTUAL, ImageCms.Flags.NONE, 27),
    ],
    ids=["relative-bpc", "relative", "perceptual"],
)
def test_cmyk_made_through_a_press_profile_reads_close_to_its_rgb_original(
    tmp_path, intent, flags, within
):
    # A gray ramp converted to SWOP press CMYK and back: the press's paper white
    # and ink black bound what the file can hold, so the darkest grays come back
    # lighter. How close the rest comes depends on the intent the file was made
    # with, which it does not say. Made as it is read back (relative colorimetric
    # with black point compensation), from 40 to 230 the gray levels stay within
    # 15 of the original, on a line of slope 1; made with the perceptual intent,
    # Pillow's default, only within 27. The standard formula strays further.
    ramp = np.arange(256, dtype=np.uint8)
    rgb = np.stack([np.tile(ramp, (4, 1))] * 3, axis=-1)
    press = ImageCms.getOpenProfile(str(RSWOP))
    inks = ImageCms.profileToProfile(
        Image.fromarray(rgb),
        SRGB,
        press,
        renderingIntent=intent,
        outputMode="CMYK",
        flags=flags,
    )
    path = tmp_path / "swop.tif"
    inks.save(path, icc_profile=press.tobytes())
    gray = load_image(path).array[0]
    formula = np.asarray(inks.convert("RGB")).mean(axis=-1)[0]
    mid = slice(40, 231)
    assert np.abs(gray[mid] - ramp[mid]).max() <= within
    assert np.abs(gray[mid] - ramp[mid]).max() < np.abs(formula[mid] - ramp[mid]).max()
    if flags == ImageCms.Flags.BLACKPOINTCOMPENSATION:
        assert abs(np.polyfit(ramp[mid], gray[mid], 1)[0] - 1) <= 0.02


def test_cmyk_with_an_alpha_channel_drops_the_alpha(tmp_path):
    rgb = _blot_rgb()
    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    path = tmp_path / "cmyk alpha.tif"
    inks = np.dstack([_inks(rgb), alpha])
    tifffile.imwrite(path, inks, photometric="separated", extrasamples=[2])
    np.testing.assert_array_equal(read_pixels(path), rgb)


@pytest.mark.parametrize(
    ("pixels", "options", "words"),
    [
        (_inks(_rgb8()), {"extratags": [(332, 3, 1, 2, True)]}, "inks that are not CMYK"),
        (
            np.dstack([_inks(_rgb8()), _gray(np.uint8, 255)]),
            {"extratags": [(334, 3, 1, 5, True)]},
            "5 inks",
        ),
        (_inks(_rgb8()).astype(np.float32) / 255, {}, "float32 inks"),
    ],
    ids=["not-cmyk-inks", "five-inks", "float"],
)
def test_a_separated_tiff_that_is_not_8_or_16_bit_cmyk_is_refused(tmp_path, pixels, options, words):
    path = tmp_path / "separated.tif"
    tifffile.imwrite(path, pixels, photometric="separated", **options)
    for read in (load_image, read_pixels, read_colours):
        with pytest.raises(ValueError, match=re.escape(words)) as info:
            read(path)
        assert str(info.value).startswith("separated.tif:")


@pytest.mark.parametrize(
    ("photometric", "pixels", "words"),
    [
        ("cielab", _rgb8(), "in the CIELAB colour space"),
        ("icclab", _rgb8(), "in the ICC L*a*b* colour space"),
        ("itulab", _rgb8(), "in the ITU L*a*b* colour space"),
        ("ycbcr", _rgb8(), "in the YCbCr colour space"),
        # A camera's raw mosaic is no gray image either.
        ("cfa", _gray(np.uint16, 65535), "with photometric interpretation CFA"),
    ],
)
def test_a_tiff_in_another_colour_space_is_refused_by_name(tmp_path, photometric, pixels, words):
    # Read as they are stored, their channels would be taken for red, green and blue.
    path = tmp_path / f"{photometric} α.tif"
    options = {"subsampling": (1, 1)} if photometric == "ycbcr" else {}
    tifffile.imwrite(path, pixels, photometric=photometric, **options)
    for read in (load_image, read_pixels, read_colours):
        with pytest.raises(ValueError, match=re.escape(words)) as info:
            read(path)
        assert str(info.value).startswith(f"{photometric} α.tif:")


def test_a_cielab_tiff_saved_by_pillow_is_refused(tmp_path):
    path = tmp_path / "lab.tif"
    Image.fromarray(_rgb8()).convert("LAB").save(path)
    with pytest.raises(ValueError, match="CIELAB"):
        load_image(path)


@pytest.mark.parametrize(
    ("name", "write", "converts"),
    [
        ("gray.tif", lambda path: tifffile.imwrite(path, _gray(np.uint16, 65535)), False),
        ("rgb.tif", lambda path: tifffile.imwrite(path, _rgb8(), photometric="rgb"), False),
        ("rgb.jpg", lambda path: Image.fromarray(_rgb8()).save(path, quality=95), False),
        ("rgb.png", lambda path: Image.fromarray(_rgb8()).save(path), False),
        ("cmyk µ.tif", lambda path: _write_cmyk(path, _blot_rgb(), "pillow"), True),
        (
            "lzw.tif",
            lambda path: _write_cmyk(path, _blot_rgb(), "pillow", compression="tiff_lzw"),
            True,
        ),
        ("planar.tif", lambda path: _write_cmyk(path, _blot_rgb(), "planar"), True),
        ("cmyk.jpg", lambda path: _write_cmyk(path, _blot_rgb(), "pillow", quality=95), True),
    ],
)
def test_the_header_tells_whether_reading_converts_cmyk(tmp_path, name, write, converts):
    # For an image imported before CMYK was converted (#131): whether reading its
    # file now converts CMYK, from the header alone, as reading it does.
    path = tmp_path / name
    write(path)
    assert converts_cmyk(path) is converts
    assert ("cmyk_converted" in _codes(load_image(path))) is converts


@pytest.mark.parametrize(
    ("photometric", "pixels", "problem"),
    [
        (
            "cielab",
            _rgb8(),
            "its pixels are stored in the CIELAB colour space, which Proteia does not"
            " convert to gray",
        ),
        (
            "separated",
            _inks(_rgb8()).astype(np.float32) / 255,
            "this separated TIFF has float32 inks, and only CMYK with 8- or 16-bit inks"
            " can be converted to gray",
        ),
    ],
    ids=["cielab", "float-inks"],
)
def test_the_header_refuses_a_colour_space_as_reading_does(tmp_path, photometric, pixels, problem):
    # The refusal says what the file holds on its own (problem), for a message
    # that names the file otherwise.
    path = tmp_path / "refused α.tif"
    tifffile.imwrite(path, pixels, photometric=photometric)
    refusals = []
    for read in (converts_cmyk, load_image, read_colours):
        with pytest.raises(UnsupportedColourSpaceError) as info:
            read(path)
        refusals.append((info.value.problem, str(info.value)))
    assert (
        refusals
        == [(problem, f"refused α.tif: {problem}; save the image as a gray or RGB TIFF")] * 3
    )


@pytest.mark.parametrize("name", ["damaged.tif", "damaged.png"])
def test_the_header_of_a_damaged_file_is_no_colour_space_refusal(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"not an image")
    with pytest.raises((ValueError, OSError)) as info:
        converts_cmyk(path)
    assert not isinstance(info.value, UnsupportedColourSpaceError)


def test_a_jpeg_compressed_cmyk_tiff_is_lossy_and_converted(tmp_path):
    rgb = _blot_rgb()
    path = _write_cmyk(tmp_path / "jpeg cmyk.tif", rgb, "pillow", compression="jpeg", quality=95)
    loaded = load_image(path)
    assert _codes(loaded) == ["lossy_format", "cmyk_converted", "color_channels_differ"]
    assert np.abs(loaded.array - rgb.mean(axis=-1)).max() <= 3


# --- Processed figures (#127) ---


@functools.cache
def _sample_blot() -> np.ndarray:
    """The 16-bit sample blot of proteia.samples: a raw-like scan."""
    return samples.render_blot().pixels


def _blot8() -> np.ndarray:
    return np.round(_sample_blot() / 257).astype(np.uint8)


def _over_exposed() -> np.ndarray:
    """The sample blot exposed three times as long: its band cores cut at 0."""
    over = 3.0 * _sample_blot() - 2.0 * samples.MEMBRANE
    return np.round(np.clip(over, 0, 65535)).astype(np.uint16)


def _white_clipped() -> np.ndarray:
    """The 8-bit sample blot with its levels set for a figure: the white point
    below most of the membrane, so its background is pure white."""
    blot = _blot8().astype(float)
    white = np.percentile(blot, 30)
    return np.clip(np.round((blot - 20) / (white - 20) * 255), 0, 255).astype(np.uint8)


def _figure(canvas: int = 255) -> Image.Image:
    """An annotated figure (8-bit gray): a blot panel and a strip below it on a
    canvas, with black frames, a rule and labels."""
    blot = Image.fromarray(_blot8())
    image = Image.new("L", (700, 720), canvas)
    image.paste(blot.crop((130, 40, 1150, 360)).resize((520, 380)), (130, 170))
    image.paste(blot.crop((130, 300, 1150, 380)).resize((400, 60)), (130, 600))
    draw = ImageDraw.Draw(image)
    draw.rectangle((129, 169, 650, 550), outline=0, width=2)
    draw.rectangle((129, 599, 530, 660), outline=0, width=3)
    draw.line((140, 55, 640, 55), fill=0, width=3)
    for i, label in enumerate(("(A)", "control", "a", "b", "kDa", "250", "(B)", "WB")):
        draw.text((10 + 80 * i, 20 + 60 * (i % 2)), label, fill=0)
    return image


def _screenshot() -> np.ndarray:
    """A screenshot of a viewer showing the blot (RGB): a title bar, a toolbar, a
    side panel listing files, a status bar, and white around the blot."""
    image = Image.new("RGB", (1280, 800), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 32), fill=(32, 96, 200))
    draw.rectangle((0, 32, 1280, 72), fill=(240, 240, 240))
    draw.rectangle((0, 72, 220, 800), fill=(248, 248, 248))
    for i in range(20):
        draw.text((12, 90 + 30 * i), f"blot {i}.tif  1200 x 500", fill=(0, 0, 0))
    image.paste(Image.fromarray(_blot8()).resize((960, 400)).convert("RGB"), (240, 90))
    draw.rectangle((0, 776, 1280, 800), fill=(230, 230, 230))
    return np.asarray(image)


def _signals(loaded) -> list[str]:
    """The signals a looks_processed warning names; empty without one."""
    found = [w.message for w in loaded.warnings if w.code == "looks_processed"]
    if not found:
        return []
    (message,) = found
    prefix = "This looks like a processed figure rather than a raw scan: "
    assert message.startswith(prefix)
    return message[len(prefix) : message.index(". Its background")].split("; ")


def _levels(counts: dict[int, int], dtype=np.uint8) -> np.ndarray:
    """A one-row image holding each level as many times as its count says."""
    return np.repeat(np.array(list(counts), dtype=dtype), list(counts.values()))[None, :]


_SIGNAL_WORDS = {
    "median": "its median level, the image-wide background, is",
    "share": "of its pixels are",
    "palette": "it is a palette image",
}


def _kinds(signals: list[str]) -> set[str]:
    return {kind for kind, words in _SIGNAL_WORDS.items() if any(words in s for s in signals)}


_TEST_BANDS = [(x, 30, 5.0, 3.0, 30000.0) for x in (50, 120, 190, 260, 330)]


DARK, LIGHT = Polarity.DARK_ON_LIGHT, Polarity.LIGHT_ON_DARK


def _imported(loaded: LoadedImage, polarity: Polarity = DARK) -> LoadedImage:
    """``loaded`` with the warnings an import with ``polarity`` records: its own,
    and looks_processed assessed against the median the import stores."""
    warnings = assess_processed(
        loaded.warnings,
        loaded.array,
        loaded.bit_depth,
        dark_on_light=polarity.dark_on_light,
        background=estimate_background(loaded.array),
        palette=loaded.palette,
    )
    return dataclasses.replace(loaded, warnings=warnings)


@pytest.mark.parametrize(
    ("name", "make", "options", "polarity"),
    [
        ("sample blot 16-bit.tif", _sample_blot, {}, DARK),
        ("sample marker 8-bit.tif", samples.render_marker, {}, DARK),
        ("sample blot 16-bit.png", _sample_blot, {}, DARK),
        ("sample blot 8-bit.png", _blot8, {}, DARK),
        ("sample blot 8-bit.jpg", _blot8, {"quality": 75}, DARK),
        ("sample blot rgb.jpg", lambda: np.dstack([_blot8()] * 3), {"quality": 90}, DARK),
        ("light on dark 16-bit.tif", lambda: 65535 - _sample_blot(), {}, LIGHT),
        ("light on dark 8-bit.jpg", lambda: 255 - _blot8(), {"quality": 90}, LIGHT),
        # Saturation lies at the other end of the range from the background.
        ("over-exposed 16-bit.tif", _over_exposed, {}, DARK),
        ("over-exposed light on dark.tif", lambda: 65535 - _over_exposed(), {}, LIGHT),
        (
            "over-exposed 8-bit.jpg",
            lambda: np.round(_over_exposed() / 257).astype(np.uint8),
            {},
            DARK,
        ),
        ("test blot 16-bit.tif", lambda: synthetic_blot((60, 400), _TEST_BANDS), {}, DARK),
        (
            "test blot 8-bit.tif",
            lambda: (synthetic_blot((60, 400), _TEST_BANDS) // 257).astype(np.uint8),
            {},
            DARK,
        ),
        # Raw 8-bit data as a palette: a palette PNG, and an ImageJ lookup table.
        ("sample blot palette.png", lambda: Image.fromarray(_blot8()).convert("P"), {}, DARK),
        (
            "sample blot lut.tif",
            _blot8,
            {"photometric": "palette", "colormap": _palette_map()},
            DARK,
        ),
        (
            "sample marker lut.tif",
            samples.render_marker,
            {"photometric": "palette", "colormap": _palette_map()},
            DARK,
        ),
        # Noise-free, its bands' flanks leave levels empty, as a reduced palette does.
        (
            "test blot lut.tif",
            lambda: (synthetic_blot((60, 400), _TEST_BANDS) // 257).astype(np.uint8),
            {"photometric": "palette", "colormap": _palette_map()},
            DARK,
        ),
    ],
)
def test_a_raw_like_scan_does_not_look_processed(tmp_path, name, make, options, polarity):
    path = _write(tmp_path / f"µ {name}", make(), **options)
    assert "looks_processed" not in _codes(_imported(load_image(path), polarity))


@pytest.mark.parametrize(
    ("name", "make", "options", "polarity", "kinds"),
    [
        ("white clipped.png", _white_clipped, {}, DARK, {"median", "share"}),
        (
            "white clipped 16-bit.tif",
            lambda: _white_clipped().astype(np.uint16) * 257,
            {},
            DARK,
            {"median", "share"},
        ),
        ("white clipped.jpg", _white_clipped, {"quality": 90}, DARK, {"median", "share"}),
        (
            "white clipped rgb.jpg",
            lambda: np.dstack([_white_clipped()] * 3),
            {"quality": 75},
            DARK,
            {"median", "share"},
        ),
        # Levelled the other way round: a light-on-dark figure's black background.
        ("black clipped.png", lambda: 255 - _white_clipped(), {}, LIGHT, {"median", "share"}),
        ("figure.png", lambda: _figure().quantize(128), {}, DARK, {"median", "share", "palette"}),
        ("figure on grey.png", lambda: _figure(230).quantize(128), {}, DARK, {"palette"}),
        (
            "figure.jpg",
            lambda: _figure().convert("RGB"),
            {"quality": 85},
            DARK,
            {"median", "share"},
        ),
        ("screenshot.png", _screenshot, {}, DARK, {"share"}),
        ("screenshot.jpg", _screenshot, {"quality": 90}, DARK, {"share"}),
    ],
)
def test_a_processed_figure_looks_processed(tmp_path, name, make, options, polarity, kinds):
    path = _write(tmp_path / f"µ {name}", make(), **options)
    loaded = _imported(load_image(path), polarity)
    assert "looks_processed" in _codes(loaded)
    assert _kinds(_signals(loaded)) == kinds


def test_the_warning_names_the_signals_that_fired():
    loaded = _imported(from_pixels(_levels({200: 400, 255: 600}), palette=True))
    assert _codes(loaded) == ["looks_processed"]
    assert loaded.warnings[0].message == (
        "This looks like a processed figure rather than a raw scan: its median level, the"
        " image-wide background, is pure white; 60% of its pixels are pure white; it is a"
        " palette image using only 2 of the 56 grey levels in its range. Its background may"
        " then not be the membrane, and over-exposure may be hidden; quantify the original"
        " scan if you have it."
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_a_tenth_of_the_pixels_at_the_limit_looks_processed(dtype):
    top = np.iinfo(dtype).max
    membrane = top * 3 // 4
    assert _signals(_imported(from_pixels(_levels({membrane: 900, top: 100}, dtype)))) == [
        "10% of its pixels are pure white"
    ]
    assert _codes(_imported(from_pixels(_levels({membrane: 901, top: 99}, dtype)))) == []
    # A light-on-dark image's background is at the other end.
    dark = _levels({top // 4: 900, 0: 100}, dtype)
    assert _signals(_imported(from_pixels(dark), LIGHT)) == ["10% of its pixels are pure black"]
    assert _codes(_imported(from_pixels(_levels({top // 4: 901, 0: 99}, dtype)), LIGHT)) == []


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_saturated_bands_at_the_other_end_never_count_whichever_half_the_median_is_in(dtype):
    # The limit is the background's end of the range, from the polarity: an
    # over-exposed band sits at the other end, and its pixels never count, even
    # where the median falls in the bands' half of the range (a 16-bit membrane
    # below mid-range, a crop the bands fill).
    top = np.iinfo(dtype).max
    for membrane, share in ((top * 3 // 4, 400), (top // 4, 400), (top * 3 // 4, 650)):
        dark_on_light = _levels({membrane: 1000 - share, 0: share}, dtype)
        assert _codes(_imported(from_pixels(dark_on_light), DARK)) == []
        light_on_dark = _levels({top - membrane: 1000 - share, top: share}, dtype)
        assert _codes(_imported(from_pixels(light_on_dark), LIGHT)) == []
    # Under the wrong polarity they do: a polarity change assesses it again.
    wrong = _imported(from_pixels(_levels({top // 4: 600, 0: 400}, dtype)), LIGHT)
    assert _signals(wrong) == ["40% of its pixels are pure black"]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_a_median_at_the_limit_is_named(dtype):
    # It never decides alone: half the pixels at the limit, which the share counts.
    top = np.iinfo(dtype).max
    at = _imported(from_pixels(_levels({top - 1: 499, top: 501}, dtype)))
    assert _signals(at) == [
        "its median level, the image-wide background, is pure white",
        "50% of its pixels are pure white",
    ]
    below = from_pixels(_levels({top - 1: 501, top: 499}, dtype))  # its median is top - 1
    assert _kinds(_signals(_imported(below))) == {"share"}
    dark = _imported(from_pixels(_levels({1: 499, 0: 501}, dtype)), LIGHT)
    assert _signals(dark)[0] == "its median level, the image-wide background, is pure black"


@pytest.mark.parametrize(("dtype", "depth"), [(np.uint8, 8), (np.uint16, 16)])
def test_where_the_exact_check_is_off_near_the_limit_counts(dtype, depth):
    # Compression, colour averaged into gray and CMYK conversion move values off
    # the limit: there it takes the possible over-exposure check's tolerance.
    top = np.iinfo(dtype).max
    near = round(near_limit_tolerance(depth))  # 2 at 8 bits, 514 at 16
    membrane = top * 3 // 4
    close = _levels({membrane: 900, top - near: 100}, dtype)
    assert _codes(_imported(from_pixels(close))) == []  # the exact check runs: not at the limit
    lossy = _imported(from_pixels(close, lossy=True))
    assert _codes(lossy) == ["lossy_format", "looks_processed"]
    assert _signals(lossy) == ["10% of its pixels are at or near pure white"]
    farther = _levels({membrane: 900, top - near - 1: 100}, dtype)
    assert _codes(_imported(from_pixels(farther, lossy=True))) == ["lossy_format"]
    # Colour: white with less blue averages to near white.
    rgb = np.repeat(close[..., None], 3, axis=-1)
    rgb[0, -100:] = (top, top, top - 3 * near)
    assert _codes(_imported(from_pixels(rgb))) == ["color_channels_differ", "looks_processed"]


def test_an_unknown_bit_depth_has_no_limit_to_be_at():
    assert _codes(_imported(from_pixels(np.full((10, 10), 1.0)))) == ["unknown_bit_depth"]


def test_assessing_again_replaces_the_warning_and_keeps_the_others_first():
    pixels = _levels({200: 400, 255: 600})
    loaded = _imported(from_pixels(pixels, lossy=True))
    assert _codes(loaded) == ["lossy_format", "looks_processed"]
    white = loaded.warnings[1]
    unmoved = dataclasses.replace(loaded, warnings=[white, loaded.warnings[0]])
    assert _imported(unmoved).warnings == [loaded.warnings[0], white]  # one, and last
    assert _codes(_imported(loaded, LIGHT)) == ["lossy_format"]  # nothing at 0: dropped


def test_a_palette_using_fewer_than_three_in_four_levels_of_its_range_looks_processed():
    def image(used: int) -> np.ndarray:  # levels 0 to 99, each used one 40 times
        levels = np.round(np.linspace(0, 99, used)).astype(int)
        assert len(set(levels)) == used
        return _levels(dict.fromkeys(levels.tolist(), 40))

    assert _codes(_imported(from_pixels(image(75), palette=True))) == []  # 75 of 100: 3 in 4
    assert _signals(_imported(from_pixels(image(74), palette=True))) == [
        "it is a palette image using only 74 of the 100 grey levels in its range"
    ]
    assert _codes(_imported(from_pixels(image(74)))) == []  # few levels alone: not a palette


def test_a_palettes_range_leaves_out_its_darkest_and_lightest_specks(tmp_path):
    # A raw 8-bit scan saved as a palette with a black and a white speck: its
    # range is the membrane's and the bands', where it uses every level.
    blot = _blot8().copy()
    blot[0, :2] = (0, 255)
    path = _write(tmp_path / "specks.png", Image.fromarray(blot).convert("P"))
    assert _codes(_imported(load_image(path))) == []


def test_only_a_palette_read_as_its_colours_is_assessed_as_one(tmp_path):
    # Every other gray level: a palette PNG reads as those levels, its colours.
    even = _levels(dict.fromkeys(range(0, 255, 2), 20))
    png = _write(tmp_path / "even levels.png", Image.fromarray(even).convert("P"))
    assert load_image(png).palette and reads_as_palette(png)
    assert _signals(_imported(load_image(png))) == [
        "it is a palette image using only 128 of the 255 grey levels in its range"
    ]
    gray = _write(tmp_path / "even levels gray.png", even)
    assert not reads_as_palette(gray)
    assert _codes(_imported(load_image(gray))) == []
    # A palette TIFF reads as its indices, a scan's own values when ImageJ
    # saves one with a lookup table: not assessed.
    options = {"photometric": "palette", "colormap": _palette_map(gray=True)}
    tiff = _write(tmp_path / "even levels.tif", even, **options)
    assert not load_image(tiff).palette and not reads_as_palette(tiff)
    assert _codes(_imported(load_image(tiff))) == []
    damaged = tmp_path / "damaged µ.png"
    damaged.write_bytes(png.read_bytes()[:40])
    with pytest.raises((ValueError, OSError)):
        reads_as_palette(damaged)


def test_looking_processed_changes_no_analysis():
    # A warning only: the over-exposure check stays exact on the pixels the
    # file holds, and the analysis array is the file's.
    pixels = _white_clipped()
    loaded = _imported(from_pixels(pixels))
    assert _codes(loaded) == ["looks_processed"]
    assert "looks_processed" not in UNTRUSTED_WARNINGS
    assert clipping_depth(loaded.bit_depth, loaded.warnings) == 8
    assert possible_clipping_depth(loaded.bit_depth, loaded.warnings) is None
    np.testing.assert_array_equal(loaded.array, pixels.astype(np.float64))


def test_reading_an_image_does_not_assess_whether_it_looks_processed(tmp_path, monkeypatch):
    # Only an import and a polarity change assess it, with the image's polarity
    # and the median the import stores; every other read (a reopened project's
    # pixels, a re-quantification, a preview) keeps only the array, so it takes
    # no median and counts nothing.
    def refused(*args, **kwargs):
        raise AssertionError("assessed on a read")

    monkeypatch.setattr(imaging, "processed_signals", refused)
    monkeypatch.setattr(np, "median", refused)
    figure = Image.fromarray(_white_clipped()).convert("P")
    loaded = load_image(_write(tmp_path / "figure µ.png", figure))
    assert _codes(loaded) == []
    assert loaded.palette  # for the import to assess
    assert not load_image(_write(tmp_path / "figure µ.tif", _white_clipped())).palette
