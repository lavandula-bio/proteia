# SPDX-License-Identifier: Apache-2.0
"""Tests for reading image files and rendering previews; files are generated here."""

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
from PIL import Image, ImageCms
from skimage import io

from proteia.core.imaging import (
    UNTRUSTED_WARNINGS,
    WARNINGS,
    UnsupportedColourSpaceError,
    _Declared,
    _read_tiff_with_pillow,
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
    to_analysis_array,
)
from proteia.core.model import ImageKind, ImageRef, ImageWarning, Polarity, Project
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
