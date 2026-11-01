# SPDX-License-Identifier: Apache-2.0
"""Tests for reading image files and rendering previews; files are generated here."""

import copy
import dataclasses
import errno
import functools
import hashlib
import itertools
import logging
import os
import pickle
import re
import struct
from io import BytesIO
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms, ImageDraw, UnidentifiedImageError
from skimage import io

from conftest import lzw_encode, synthetic_blot, write_lzw_tiff
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
    with pytest.raises(ValueError) as info:
        read_pixels(path)
    assert str(info.value) == (
        "'stack.tif' holds more than one image (such as a stack, the channels of a composite,"
        " or an image with a mask or a thumbnail), which Proteia does not import: only a"
        " single image can be quantified; save one of its images as its own file and import"
        " it again"
    )


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
    area = {"id": "arena-4", "image_id": "img-1", "source": "whole_image"}
    membrane = {"id": "mem-2", "lane_table_id": "lt-3", "arenas": [area]}
    batch = {"images": [image], "lane_tables": [{"id": "lt-3"}], "membranes": [membrane]}
    project = Project.model_validate({"next_id": 5, "batch": batch})
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
    # #234 review 3: in Proteia's own words (ReadingRefusal), which a refusal
    # of reading an image already in the project gives as they are, not as
    # a damaged file's.
    for pixels in (np.array([[0.0, 1.0, np.nan]], dtype=np.float32), np.array([[0.0, np.inf]])):
        with pytest.raises(imaging.ReadingRefusal) as info:
            to_analysis_array(pixels)
        assert str(info.value) == "image: it has NaN or infinite pixel values"
        assert info.value.refusal("ratio µ.tif", remedy=False) == (
            "cannot read 'ratio µ.tif': it has NaN or infinite pixel values"
        )


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
    with pytest.raises(ValueError) as info:
        read_pixels(path)
    assert str(info.value) == (
        "'composite.tif' holds more than one image (such as a stack, the channels of a"
        " composite, or an image with a mask or a thumbnail), which Proteia does not import:"
        " only a single image can be quantified; save one of its images as its own file and"
        " import it again"
    )


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


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_a_big_endian_16_bit_tiff_named_otherwise_has_its_bit_depth(tmp_path, suffix):
    # Pillow, which reads a TIFF named .png or .jpg by its suffix, keeps a
    # big-endian file's byte order, and reading gives its pixels so, as it did
    # before #224 (the self-test records their bytes). The bit depth is the
    # same in either byte order: its depth was unknown before, which an import
    # now refuses (#224).
    pixels = _gray(np.uint16, 65535)
    path = tmp_path / f"big-endian µ{suffix}"
    tifffile.imwrite(path, pixels.astype(">u2"), byteorder=">")
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.pixels, pixels)
    assert loaded.pixels.dtype == np.dtype(">u2")  # as read, not converted
    assert loaded.bit_depth == 16
    assert _codes(loaded) == (["lossy_format"] if suffix == ".jpg" else [])


@pytest.mark.parametrize(
    ("dtype", "depth"),
    [
        (np.uint8, 8),
        (np.uint16, 16),
        (np.int8, None),
        (np.int16, None),
        (np.uint32, None),
        (np.float16, None),
        (np.float32, None),
        (np.complex64, None),
        (np.bool_, None),
    ],
)
@pytest.mark.filterwarnings("ignore:Casting complex values to real")  # from_pixels of complex
def test_the_bit_depth_is_the_same_in_either_byte_order(dtype, depth):
    # #224 review: an unsigned 16-bit image in the other byte order (Pillow
    # keeps a big-endian TIFF's) has its 16 bits, not an unknown depth.
    native = np.dtype(dtype)
    for order in (native, native.newbyteorder("S")):
        assert imaging.bit_depth_of(np.zeros((2, 3), dtype=order)) == depth
        assert from_pixels(np.ones((2, 3), dtype=order)).bit_depth == depth


def _packed_tiff(values: np.ndarray, bits: int, sampleformat: int = 1) -> bytes:
    """A little-endian, uncompressed, single-strip gray TIFF of ``values``
    stored ``bits`` wide (a whole number of bytes, each in the file's byte
    order), with the SampleFormat tag ``sampleformat`` (1 unsigned, 2 signed
    integer): a layout tifffile writes only with imagecodecs, such as 24-bit
    unsigned integers."""
    height, width = values.shape
    size = bits // 8
    signed = sampleformat == 2
    data = b"".join(int(v).to_bytes(size, "little", signed=signed) for v in values.ravel())
    tags = [
        (256, 3, width),  # ImageWidth
        (257, 3, height),  # ImageLength
        (258, 3, bits),  # BitsPerSample
        (259, 3, 1),  # Compression: none
        (262, 3, 1),  # PhotometricInterpretation: black is zero
        (273, 4, 0),  # StripOffsets: set below
        (277, 3, 1),  # SamplesPerPixel
        (278, 3, height),  # RowsPerStrip
        (279, 4, len(data)),  # StripByteCounts
        (339, 3, sampleformat),  # SampleFormat
    ]
    offset = 8 + 2 + 12 * len(tags) + 4
    ifd = struct.pack("<H", len(tags))
    for tag, kind, value in tags:
        value = offset if tag == 273 else value
        packed = struct.pack("<HH", value, 0) if kind == 3 else struct.pack("<I", value)
        ifd += struct.pack("<HHI", tag, kind, 1) + packed
    return b"II*\x00" + struct.pack("<I", 8) + ifd + struct.pack("<I", 0) + data


def test_a_hand_made_24_bit_tiff_is_what_its_header_says():
    # The helper's file is a TIFF of 24-bit unsigned integers, which tifffile
    # widens to 32 bits but reads only with imagecodecs.
    values = np.array([[0, 1, 2**23], [2**24 - 1, 70000, 5]], dtype=np.uint32)
    with tifffile.TiffFile(BytesIO(_packed_tiff(values, 24))) as tif:
        page = tif.pages.first
        assert (page.shape, page.bitspersample, page.dtype) == ((2, 3), 24, np.uint32)
        assert page.tags["StripByteCounts"].value == (18,)
    signed = np.array([[-3, 0], [32767, -32768]])
    with tifffile.TiffFile(BytesIO(_packed_tiff(signed, 16, 2))) as tif:
        page = tif.pages.first
        assert (page.bitspersample, page.dtype) == (16, np.int16)
        np.testing.assert_array_equal(page.asarray(), signed)


def _three_pages(pixels: np.ndarray):
    def write(path):
        with tifffile.TiffWriter(path) as tif:
            for _ in range(3):
                tif.write(pixels)

    return write


def _image_after(first: np.ndarray, **options):
    """A writer of a TIFF of a 16-bit image after a page ``first`` (a mask or
    a thumbnail, by ``options``): two image series, which reading refuses."""
    return _pages((first, options), (_gray(np.uint16, 65535), {}))


def _patched_header(data: bytes, bits: int, sampleformat: int) -> bytes:
    """A :func:`_packed_tiff` file whose header says its samples are ``bits``
    wide, of ``sampleformat`` (1 unsigned, 2 signed integer, 3 floating
    point, 4 untyped), whatever its data."""
    patched = bytearray(data)
    for tag, value in ((258, bits), (339, sampleformat)):
        at = patched.index(struct.pack("<HHI", tag, 3, 1)) + 8
        patched[at : at + 2] = struct.pack("<H", value)
    return bytes(patched)


def _declared_as(bits: int, sampleformat: int) -> bytes:
    """A TIFF whose header declares samples ``bits`` wide, of ``sampleformat``
    (:func:`_patched_header`); its data is not read."""
    return _patched_header(_packed_tiff(np.zeros((6, 8), dtype=np.int64), 16), bits, sampleformat)


# A TIFF's header and the pixel type it declares when its bit depth is
# unknown (#224), by file: numpy's kind of it, and the bits per sample the
# file stores; None when the depth is known or the file no TIFF.
_DECLARED_TYPES = {
    "float µ.tif": (
        lambda path: tifffile.imwrite(path, _gray(np.uint16, 65535).astype(np.float32)),
        ("f", 32),
    ),
    "half α.tif": (
        lambda path: tifffile.imwrite(path, _gray(np.uint8, 255).astype(np.float16)),
        ("f", 16),
    ),
    # Whatever its suffix: Pillow, which reads it by its suffix, cannot.
    "double named β.png": (
        lambda path: tifffile.imwrite(path, _gray(np.uint16, 65535).astype(np.float64)),
        ("f", 64),
    ),
    "big-endian float.TIF": (
        lambda path: tifffile.imwrite(path, _gray(np.uint8, 255).astype(">f4"), byteorder=">"),
        ("f", 32),
    ),
    "24-bit µ.tif": (
        lambda path: path.write_bytes(_packed_tiff(_gray(np.uint32, 2**24 - 1), 24)),
        ("u", 24),
    ),
    # Pillow reads it into 32-bit integers: the file's are 16.
    "int16 named α.png": (lambda path: tifffile.imwrite(path, _gray(np.int16, 32767)), ("i", 16)),
    "uint32 β.tif": (lambda path: tifffile.imwrite(path, _gray(np.uint32, 2**32 - 1)), ("u", 32)),
    "bilevel µ.tif": (
        lambda path: tifffile.imwrite(path, _gray(np.uint8, 255) > 99, bitspersample=1),
        ("b", 1),
    ),
    "complex α.tif": (
        lambda path: tifffile.imwrite(path, _gray(np.uint8, 255).astype(np.complex64)),
        ("c", 64),
    ),
    # The first image's: a stack of them is refused as one when read.
    # #224 review 2: a TIFF of several images is judged by none of them: its
    # first page may be a mask or a thumbnail of another type than the image,
    # and reading refuses the file as several images.
    "float stack β.tif": (_three_pages(_gray(np.uint8, 255).astype(np.float32)), None),
    "float stack in one series µ.tif": (
        lambda path: tifffile.imwrite(
            path, np.stack([_gray(np.uint8, 255)] * 2).astype(np.float32)
        ),
        None,
    ),
    "mask first α.tif": (
        lambda path: _image_after(_gray(np.uint8, 255) > 99, subfiletype=4, bitspersample=1)(path),
        None,
    ),
    "float thumbnail first β.tif": (
        lambda path: _image_after(_gray(np.uint8, 255).astype(np.float32), subfiletype=1)(path),
        None,
    ),
    # #224 review 2: reading refuses the colour space too, which the
    # refusal names with the type.
    "float CIELAB µ.tif": (
        lambda path: tifffile.imwrite(
            path, np.dstack([_gray(np.uint8, 255)] * 3).astype(np.float32), photometric="cielab"
        ),
        ("f", 32, "in the CIELAB colour space"),
    ),
    "int16 YCbCr α.tif": (
        lambda path: tifffile.imwrite(
            path, np.dstack([_gray(np.int16, 99)] * 3), photometric="ycbcr"
        ),
        ("i", 16, "in the YCbCr colour space"),
    ),
    "float CFA β.tif": (
        lambda path: tifffile.imwrite(
            path, _gray(np.uint8, 255).astype(np.float32), photometric=32803
        ),
        ("f", 32, "with photometric interpretation CFA"),
    ),
    # Gray, RGB and palette are read as stored, white-is-zero too, and CMYK
    # converted (whose inks an import refuses first, #207).
    "float CMYK β.tif": (
        lambda path: tifffile.imwrite(
            path, np.dstack([_gray(np.uint8, 255)] * 4).astype(np.float32), photometric="separated"
        ),
        ("f", 32),
    ),
    "float white-is-zero µ.tif": (
        lambda path: tifffile.imwrite(
            path, _gray(np.uint8, 255).astype(np.float32), photometric="miniswhite"
        ),
        ("f", 32),
    ),
    "float RGB α.tif": (
        lambda path: tifffile.imwrite(
            path, np.dstack([_gray(np.uint8, 255)] * 3).astype(np.float32), photometric="rgb"
        ),
        ("f", 32),
    ),
    # Widths tifffile has no type for, named by their sample format.
    "signed 12-bit µ.tif": (lambda path: path.write_bytes(_declared_as(12, 2)), ("i", 12)),
    "128-bit α.tif": (lambda path: path.write_bytes(_declared_as(128, 1)), ("u", 128)),
    "2-bit float β.tif": (lambda path: path.write_bytes(_declared_as(2, 3)), ("f", 2)),
    # A known bit depth, in either byte order.
    "uint16 µ.tif": (lambda path: tifffile.imwrite(path, _gray(np.uint16, 65535)), None),
    "uint8 α.tif": (lambda path: tifffile.imwrite(path, _gray(np.uint8, 255)), None),
    "big-endian 16 named β.png": (
        lambda path: tifffile.imwrite(path, _gray(np.uint16, 65535).astype(">u2"), byteorder=">"),
        None,
    ),
    # Untyped samples, which tifffile reads as unsigned integers.
    "untyped 16-bit µ.tif": (lambda path: path.write_bytes(_declared_as(16, 4)), None),
    # Not a TIFF: its pixel type is told as read.
    "bilevel α.png": (lambda path: Image.fromarray(_gray(np.uint8, 255) > 99).save(path), None),
}


@pytest.mark.parametrize("name", list(_DECLARED_TYPES))
def test_a_tiff_header_declares_the_type_of_pixels_with_no_known_bit_depth(tmp_path, name):
    # #224 review: an import refuses such a file from its header, before
    # decoding it, and names the width the file stores, which a reader may
    # widen (tifffile 24 bits into 32, Pillow 16-bit integers into 32).
    write, expected = _DECLARED_TYPES[name]
    path = tmp_path / name
    write(path)
    declared = imaging.declared_pixel_type(path)
    if expected is None:
        assert declared is None
    else:
        kind, bits, *held = expected
        assert declared == imaging.PixelType(kind, bits, "", *held or [""])
        assert type(declared.bits) is int


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        (b"II*\x00", ValueError, None),  # truncated
        (b"MM\x00*\x00\x00\x00\x08\x00\x01\x01\x00\x00\x09", ValueError, None),  # damaged
        (None, OSError, None),  # no such file
        # Untyped samples of a width tifffile has no type for, complex ones,
        # and sample formats TIFF does not define.
        (_declared_as(128, 4), ValueError, "(128-bit samples of format 4)"),
        (_declared_as(16, 6), ValueError, "(16-bit samples of format 6)"),
        (_declared_as(16, 7), ValueError, "(16-bit samples of format 7)"),
        (_declared_as(16, 0), ValueError, "(16-bit samples of format 0)"),
    ],
)
def test_a_header_that_cannot_be_read_declares_no_type(tmp_path, data, error, message):
    path = tmp_path / "damaged µ.tif"
    if data is not None:
        path.write_bytes(data)
    with pytest.raises(error) as info:
        imaging.declared_pixel_type(path)
    if message is not None:
        assert str(info.value) == f"damaged µ.tif: not a readable image {message}"


@pytest.mark.parametrize("error", [MemoryError, RecursionError])
def test_running_out_of_memory_while_reading_a_header_is_not_a_damaged_file(
    tmp_path, monkeypatch, error
):
    # #224 review 2: a header that cannot be read is left to reading (a
    # ValueError the import ignores); a MemoryError must not be, or a signed
    # 8-bit TIFF named .png would be read by Pillow as unsigned. As #217's
    # check_misnamed_tiff.
    path = tmp_path / "int8 µ.png"
    tifffile.imwrite(path, _gray(np.int8, 127))

    def run_out(*args, **kwargs):
        raise error("no room")

    monkeypatch.setattr(imaging.tifffile, "TiffFile", run_out)
    with pytest.raises(error, match="^no room$"):
        imaging.declared_pixel_type(path)


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


def _declared(
    photometric, samples, shape=(6, 8), dtype=np.uint8, extrasamples=(), planar=False
) -> _Declared:
    return _Declared("LZW", shape, np.dtype(dtype), photometric, samples, extrasamples, planar)


@pytest.mark.parametrize(
    "declared",
    [
        # Pillow would change these values: it inverts white-is-zero gray and
        # un-premultiplies associated alpha.
        _declared(tifffile.PHOTOMETRIC.MINISWHITE, 1),
        _declared(tifffile.PHOTOMETRIC.RGB, 4, (6, 8, 4), extrasamples=(1,)),
        _declared(tifffile.PHOTOMETRIC.PALETTE, 1),
        # A result that differs from the declaration: never read with another scale.
        _declared(tifffile.PHOTOMETRIC.MINISBLACK, 1, dtype=np.uint16),
        _declared(tifffile.PHOTOMETRIC.RGB, 3, (6, 8, 3)),
        _declared(tifffile.PHOTOMETRIC.RGB, 4, (6, 8, 4), extrasamples=(2,)),
    ],
    ids=[
        "min-is-white",
        "associated-alpha",
        "palette",
        "other-bit-depth",
        "other-shape",
        "rgba-other-shape",
    ],
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


# --- Compressed TIFFs with an extra sample (#232) ---

_OPAQUE = np.full((6, 8), 255, dtype=np.uint8)
_CLEAR = np.zeros((6, 8), dtype=np.uint8)
# Every level from 1 to 254 in some order: neither opaque nor transparent.
_TRANSLUCENT = (np.arange(48).reshape(6, 8) * 37 % 254 + 1).astype(np.uint8)


def _tiles_40() -> np.ndarray:
    """40 x 40 RGB plus alpha, in 3 x 3 tiles of 16 x 16 (the last ones padded)."""
    level = np.arange(40 * 40).reshape(40, 40)
    return np.dstack([level % 256, level * 3 % 256, level * 7 % 256, level * 11 % 255 + 1]).astype(
        np.uint8
    )


def _gray_alpha(alpha: np.ndarray) -> np.ndarray:
    return np.dstack([_gray(np.uint8, 255), alpha])


def _gray_rgba(alpha: np.ndarray) -> np.ndarray:
    gray = _gray(np.uint8, 255)
    return np.dstack([gray, gray, gray, alpha])


_RGB, _GRAY = {"photometric": "rgb"}, {"photometric": "minisblack"}
_UNASSOCIATED, _ASSOCIATED, _UNSPECIFIED = (
    {"extrasamples": [2]},
    {"extrasamples": [1]},
    {"extrasamples": [0]},
)


def _plain_and_lzw(folder: Path, pixels: np.ndarray, **layout) -> tuple[Path, Path]:
    """The same pixels written uncompressed (read by tifffile) and with LZW
    (decoded by Pillow, as tifffile cannot here), in the same layout."""
    plain, lzw = folder / "copy β.tif", folder / "lzw α µ.tif"
    tifffile.imwrite(
        plain,
        np.moveaxis(pixels, -1, 0) if layout.get("planar") else pixels,
        byteorder=layout.get("byteorder", "<"),
        planarconfig="separate" if layout.get("planar") else None,
        rowsperstrip=layout.get("rowsperstrip"),
        tile=layout.get("tile"),
        **{key: value for key, value in layout.items() if key in ("photometric", "extrasamples")},
    )
    write_lzw_tiff(lzw, pixels, **layout)
    with tifffile.TiffFile(lzw) as tif:  # the premise: Pillow decodes it, not tifffile
        page = tif.pages.first
        assert page.compression == tifffile.COMPRESSION.LZW
        assert page.compression not in tifffile.TIFF.DECOMPRESSORS
    return plain, lzw


@pytest.mark.parametrize(
    ("pixels", "layout", "warnings"),
    [
        (_gray_rgba(_OPAQUE), _RGB | _UNASSOCIATED, []),
        (_gray_rgba(_TRANSLUCENT), _RGB | _UNASSOCIATED, []),
        (_gray_rgba(_CLEAR), _RGB | _UNASSOCIATED, []),
        (np.dstack([_rgb8(), _TRANSLUCENT]), _RGB | _UNASSOCIATED, ["color_channels_differ"]),
        (
            np.dstack([_rgb8(), _TRANSLUCENT]),
            _RGB | _UNASSOCIATED | {"planar": True},
            ["color_channels_differ"],
        ),
        (
            np.dstack([_rgb8(), _TRANSLUCENT]),
            _RGB | _UNASSOCIATED | {"byteorder": ">"},
            ["color_channels_differ"],
        ),
        (
            np.dstack([_rgb8(), _TRANSLUCENT]),
            _RGB | _UNASSOCIATED | {"rowsperstrip": 2},
            ["color_channels_differ"],
        ),
        (
            np.dstack([_rgb8(), _TRANSLUCENT]),
            _RGB | _UNASSOCIATED | {"predictor": True},
            ["color_channels_differ"],
        ),
        (_tiles_40(), _RGB | _UNASSOCIATED | {"tile": (16, 16)}, ["color_channels_differ"]),
        (
            _tiles_40(),
            _RGB | _UNASSOCIATED | {"tile": (16, 16), "planar": True},
            ["color_channels_differ"],
        ),
        (
            _gray_rgba(_TRANSLUCENT),
            _RGB | _UNASSOCIATED | {"planar": True, "rowsperstrip": 2, "byteorder": ">"},
            [],
        ),
        (_gray_alpha(_OPAQUE), _GRAY | _UNASSOCIATED, []),
        (_gray_alpha(_TRANSLUCENT), _GRAY | _UNASSOCIATED, []),
        (_gray_alpha(_CLEAR), _GRAY | _UNASSOCIATED, []),
        (
            _gray_alpha(_TRANSLUCENT),
            _GRAY | _UNASSOCIATED | {"byteorder": ">", "rowsperstrip": 2},
            [],
        ),
        (_gray_alpha(_TRANSLUCENT), _GRAY | _UNASSOCIATED | {"predictor": True}, []),
        (_tiles_40()[..., 2:], _GRAY | _UNASSOCIATED | {"tile": (16, 16)}, []),
    ],
    ids=[
        "rgba-gray-opaque",
        "rgba-gray-translucent",
        "rgba-gray-transparent",
        "rgba-colour",
        "rgba-planar",
        "rgba-big-endian",
        "rgba-strips",
        "rgba-predictor",
        "rgba-tiles",
        "rgba-planar-tiles",
        "rgba-planar-strips-big-endian",
        "gray-alpha-opaque",
        "gray-alpha-translucent",
        "gray-alpha-transparent",
        "gray-alpha-big-endian-strips",
        "gray-alpha-predictor",
        "gray-alpha-tiles",
    ],
)
def test_a_compressed_tiff_with_unassociated_alpha_reads_as_its_uncompressed_copy(
    tmp_path, pixels, layout, warnings
):
    # #232: Pillow reads an unassociated alpha sample as stored, so the LZW
    # file reads as tifffile reads its uncompressed copy: its colours and its
    # alpha, channels last, which the analysis array then ignores.
    plain, lzw = _plain_and_lzw(tmp_path, pixels, **layout)
    copy, read = imaging.read_file(plain), imaging.read_file(lzw)
    np.testing.assert_array_equal(read.pixels, pixels)
    assert (read.pixels.shape, read.pixels.dtype) == (pixels.shape, np.uint8)
    assert (read.lossy, read.cmyk, read.palette) == (copy.lossy, copy.cmyk, copy.palette)
    np.testing.assert_array_equal(read.pixels, copy.pixels)
    loaded, expected = imaging.load_pixels(read), imaging.load_pixels(copy)
    np.testing.assert_array_equal(loaded.array, expected.array)
    np.testing.assert_array_equal(loaded.array, pixels[..., :-1].mean(axis=-1))
    assert (loaded.bit_depth, loaded.warnings) == (8, expected.warnings)
    assert _codes(loaded) == warnings


def test_a_compressed_rgba_tiff_saved_by_an_image_editor_is_read(tmp_path):
    # #232: a gray blot saved as RGBA with LZW (here by Pillow, which writes an
    # unassociated alpha) reads as its uncompressed copy, with no colour note:
    # the alpha does not make the channels differ.
    for mode, pixels in (("RGBA", _gray_rgba(_OPAQUE)), ("LA", _gray_alpha(_TRANSLUCENT))):
        lzw = tmp_path / f"saved {mode} µ.tif"
        Image.fromarray(pixels, mode).save(lzw, compression="tiff_lzw")
        with tifffile.TiffFile(lzw) as tif:
            page = tif.pages.first
            assert (page.compression, page.extrasamples) == (tifffile.COMPRESSION.LZW, (2,))
        loaded = load_image(lzw)
        np.testing.assert_array_equal(loaded.pixels, pixels)
        np.testing.assert_array_equal(loaded.array, _gray(np.uint8, 255).astype(np.float64))
        assert (loaded.bit_depth, loaded.warnings) == (8, [])


@pytest.mark.parametrize("alpha", ["opaque", "translucent"])
def test_a_jpeg_compressed_rgba_tiff_is_read_as_lossy(tmp_path, alpha):
    # Any compression Pillow decodes for tifffile takes the same rule (#232):
    # unassociated alpha after the colours is kept as stored; JPEG is lossy,
    # so the colours come back near those of the uncompressed copy, and the
    # import says so. A translucent alpha (20 to 235 down the image) would
    # move them far off if they were premultiplied or un-premultiplied.
    y, x = np.mgrid[0:32, 0:48]
    level = (40 + 160 * x / 47).astype(np.uint8)
    opacity = (
        np.full_like(level, 255) if alpha == "opaque" else (20 + 215 * y / 31).astype(np.uint8)
    )
    pixels = np.dstack([level, 255 - level, level // 2, opacity])
    path = tmp_path / f"jpeg rgba {alpha} α.tif"
    Image.fromarray(pixels, "RGBA").save(path, compression="tiff_jpeg")
    plain = tmp_path / f"copy {alpha} β.tif"
    tifffile.imwrite(plain, pixels, photometric="rgb", extrasamples=[2])
    read, copy = imaging.read_file(path), imaging.read_file(plain)
    assert (read.pixels.shape, read.pixels.dtype, read.lossy) == ((32, 48, 4), np.uint8, True)
    assert np.abs(read.pixels.astype(int) - copy.pixels).max() <= 8
    assert "lossy_format" in _codes(imaging.load_pixels(read))


def _refusal(name: str, shape: str, dtype: str, photometric: str) -> str:
    return (
        f"{name}: LZW compression of this {shape} {dtype} {photometric} image cannot be"
        " decoded by this installation; save the image as an uncompressed TIFF"
    )


@pytest.mark.parametrize(
    ("pixels", "layout", "declared"),
    [
        # Pillow un-premultiplies associated alpha: these colours would come
        # back unchanged only because the alpha is opaque.
        (_gray_rgba(_OPAQUE), _RGB | _ASSOCIATED, "6x8x4 uint8 RGB"),
        (_gray_rgba(_TRANSLUCENT), _RGB | _ASSOCIATED, "6x8x4 uint8 RGB"),
        (_gray_rgba(_TRANSLUCENT), _RGB | _ASSOCIATED | {"planar": True}, "6x8x4 uint8 RGB"),
        # Pillow drops an unspecified extra sample (tifffile keeps it).
        (_gray_rgba(_TRANSLUCENT), _RGB | _UNSPECIFIED, "6x8x4 uint8 RGB"),
        (_gray_rgba(_TRANSLUCENT), _RGB | _UNSPECIFIED | {"planar": True}, "6x8x4 uint8 RGB"),
        (
            np.dstack([_gray_rgba(_TRANSLUCENT), _OPAQUE]),
            _RGB | {"extrasamples": [2, 0]},
            "6x8x5 uint8 RGB",
        ),
        (
            np.dstack([_gray_rgba(_TRANSLUCENT), _OPAQUE]),
            _RGB | {"extrasamples": [2, 2]},
            "6x8x5 uint8 RGB",
        ),
        (_gray_alpha(_TRANSLUCENT), _GRAY | _ASSOCIATED, "6x8x2 uint8 MINISBLACK"),
        (_gray_alpha(_TRANSLUCENT), _GRAY | _UNSPECIFIED, "6x8x2 uint8 MINISBLACK"),
        # Pillow loses the alpha of planar gray (its gray is right).
        (
            _gray_alpha(_TRANSLUCENT),
            _GRAY | _UNASSOCIATED | {"planar": True},
            "6x8x2 uint8 MINISBLACK",
        ),
        (
            _tiles_40()[..., 2:],
            _GRAY | _UNASSOCIATED | {"planar": True, "tile": (16, 16)},
            "40x40x2 uint8 MINISBLACK",
        ),
        # Pillow inverts white-is-zero gray.
        (
            _gray_alpha(_TRANSLUCENT),
            {"photometric": "miniswhite"} | _UNASSOCIATED,
            "6x8x2 uint8 MINISWHITE",
        ),
    ],
    ids=[
        "rgb-associated-opaque",
        "rgb-associated",
        "rgb-associated-planar",
        "rgb-unspecified",
        "rgb-unspecified-planar",
        "rgb-alpha-and-unspecified",
        "rgb-two-alphas",
        "gray-associated",
        "gray-unspecified",
        "gray-alpha-planar",
        "gray-alpha-planar-tiles",
        "min-is-white-alpha",
    ],
)
def test_a_compressed_tiff_whose_extra_sample_pillow_changes_is_refused_unread(
    tmp_path, monkeypatch, pixels, layout, declared
):
    # #232: refused from its header, with the message it had before, and
    # never decoded: Pillow would give other values, or another shape.
    _, lzw = _plain_and_lzw(tmp_path, pixels, **layout)
    opened = []
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: opened.append(args))
    with pytest.raises(ValueError) as info:
        imaging.read_file(lzw)
    shape, dtype, photometric = declared.split()
    assert str(info.value) == _refusal("lzw α µ.tif", shape, dtype, photometric)
    assert opened == []


@pytest.mark.parametrize(
    ("pixels", "layout", "declared"),
    [
        # Pillow narrows 16-bit colour to 8 bits, alpha and all.
        (
            np.dstack([_rgb8(), _TRANSLUCENT]).astype(np.uint16) * 257,
            _RGB | _UNASSOCIATED,
            "6x8x4 uint16 RGB",
        ),
        (
            np.dstack([_rgb8(), _TRANSLUCENT]).astype(np.uint16) * 257,
            _RGB | _UNASSOCIATED | {"planar": True, "byteorder": ">"},
            "6x8x4 uint16 RGB",
        ),
        # And it cannot read 16-bit gray with alpha.
        (
            _gray_alpha(_TRANSLUCENT).astype(np.uint16) * 257,
            _GRAY | _UNASSOCIATED,
            "6x8x2 uint16 MINISBLACK",
        ),
    ],
    ids=["rgb16-alpha", "rgb16-alpha-planar-big-endian", "gray16-alpha"],
)
def test_a_compressed_16_bit_tiff_with_alpha_stays_refused_unread(
    tmp_path, monkeypatch, pixels, layout, declared
):
    # Refused from its header, with the message it had before, and never
    # decoded: a large one must not reach Pillow's size limit first.
    _, lzw = _plain_and_lzw(tmp_path, pixels, **layout)
    opened = []
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: opened.append(args))
    with pytest.raises(ValueError) as info:
        imaging.read_file(lzw)
    shape, dtype, photometric = declared.split()
    assert str(info.value) == _refusal("lzw α µ.tif", shape, dtype, photometric)
    assert opened == []


def test_a_large_compressed_16_bit_rgba_tiff_is_refused_for_its_layout(tmp_path, monkeypatch):
    # Refused for what it holds, not for Pillow's size limit, which it never reaches.
    pixels = np.dstack([_rgb8(), _TRANSLUCENT]).astype(np.uint16) * 257
    _, lzw = _plain_and_lzw(tmp_path, pixels, **(_RGB | _UNASSOCIATED))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)  # 48 pixels is over twice the limit
    with pytest.raises(ValueError) as info:
        imaging.read_file(lzw)
    assert str(info.value) == _refusal("lzw α µ.tif", "6x8x4", "uint16", "RGB")


def test_planar_rgba_is_read_compressed_only_with_lzw(tmp_path, monkeypatch):
    # Planar RGB with alpha is read compressed only as measured: with LZW,
    # whose result equals the uncompressed copy (test above). The same strips
    # labelled with another compression Pillow decodes (JPEG here) are refused
    # from the header, unread.
    pixels = np.dstack([_rgb8(), _TRANSLUCENT])
    path = write_lzw_tiff(
        tmp_path / "planar jpeg µ.tif",
        pixels,
        planar=True,
        compression=tifffile.COMPRESSION.JPEG,
        **(_RGB | _UNASSOCIATED),
    )
    with tifffile.TiffFile(path) as tif:
        page = tif.pages.first
        assert (page.compression, page.planarconfig) == (
            tifffile.COMPRESSION.JPEG,
            tifffile.PLANARCONFIG.SEPARATE,
        )
    opened = []
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: opened.append(args))
    with pytest.raises(ValueError) as info:
        imaging.read_file(path)
    assert str(info.value) == (
        "planar jpeg µ.tif: JPEG compression of this 6x8x4 uint8 RGB image cannot be"
        " decoded by this installation; save the image as an uncompressed TIFF"
    )
    assert opened == []


def test_the_lzw_writer_of_the_tests_is_decoded_as_written(tmp_path):
    # write_lzw_tiff starts its table again before codes need 10 bits: data
    # long enough to fill the table many times decodes as written.
    pixels = np.random.default_rng(232).integers(0, 256, size=(64, 96, 4), dtype=np.uint8)
    path = write_lzw_tiff(
        tmp_path / "long µ.tif", pixels, photometric="rgb", extrasamples=[2], rowsperstrip=40
    )
    with Image.open(path) as image:
        np.testing.assert_array_equal(np.asarray(image), pixels)
    codes = len(lzw_encode(pixels[:40].tobytes())) * 8 // 9  # in its first strip alone
    assert codes > 10 * 511


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


def _separated(pixels: np.ndarray, **options):
    return lambda path: tifffile.imwrite(path, pixels, photometric="separated", **options)


def _cmyk_jpeg(path: Path) -> None:
    """A CMYK JPEG, whatever the suffix of ``path``."""
    Image.fromarray(_blot_rgb()).convert("CMYK").save(path, format="JPEG", quality=95)


_INKS = {"photometric": "separated"}


def _pages(*pages: tuple[np.ndarray, dict]):
    """A writer of a TIFF of these pages, each written on its own with its
    options: by tifffile, an image series each, unless it is a thumbnail."""

    def write(path: Path) -> None:
        with tifffile.TiffWriter(path) as tif:
            for pixels, options in pages:
                tif.write(pixels, **options)

    return write


def _pillow_cmyk_stack(path: Path) -> None:
    """Two CMYK pages, as Pillow saves a multi-page TIFF."""
    page = Image.fromarray(_blot_rgb()).convert("CMYK")
    page.save(path, save_all=True, append_images=[page.copy()])


_SPOT = {"photometric": "separated", "extratags": [(332, 3, 1, 2, True)]}  # InkSet: not CMYK
_ALPHA = np.full((6, 8), 255, dtype=np.uint8)


def _five_inks_untagged(path: Path) -> None:
    """A separated TIFF of five 8-bit inks with no ExtraSamples tag (tifffile
    marks a fifth sample an extra one, so it is built here): uncompressed,
    little-endian, NumberOfInks 5."""
    inks = np.dstack([_inks(_rgb8()), _gray(np.uint8, 255)])
    height, width, samples = inks.shape
    data = inks.tobytes()
    tags = [  # tag, type (3 short, 4 long), count, value; a short's value fills 4 bytes
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, samples, 146),  # BitsPerSample: 8 each, after the 138-byte IFD
        (259, 3, 1, 1),  # uncompressed
        (262, 3, 1, 5),  # separated
        (273, 4, 1, 156),  # the one strip, after them
        (277, 3, 1, samples),
        (278, 4, 1, height),
        (279, 4, 1, len(data)),
        (284, 3, 1, 1),
        (334, 3, 1, samples),  # NumberOfInks
    ]
    ifd = struct.pack("<H", len(tags)) + b"".join(struct.pack("<HHII", *t) for t in tags)
    path.write_bytes(
        b"II*\x00" + struct.pack("<I", 8) + ifd + bytes(4) + struct.pack("<5H", *[8] * 5) + data
    )
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        assert (page.samplesperpixel, page.extrasamples) == (5, ())
        np.testing.assert_array_equal(page.asarray(), inks)


@pytest.mark.parametrize(
    ("name", "write", "inks"),
    [
        ("gray.tif", lambda path: tifffile.imwrite(path, _gray(np.uint16, 65535)), None),
        ("rgb.tif", lambda path: tifffile.imwrite(path, _rgb8(), photometric="rgb"), None),
        (
            "palette.tif",
            lambda path: tifffile.imwrite(
                path, _gray(np.uint8, 255), photometric="palette", colormap=_palette_map()
            ),
            None,
        ),
        ("lab α.tif", lambda path: tifffile.imwrite(path, _rgb8(), photometric="cielab"), None),
        ("rgb.jpg", lambda path: Image.fromarray(_rgb8()).save(path, quality=95), None),
        ("gray.jpg", lambda path: Image.fromarray(_gray(np.uint8, 255)).save(path), None),
        ("rgb.png", lambda path: Image.fromarray(_rgb8()).save(path), None),
        ("palette.png", lambda path: _palette_png().save(path), None),
        ("gray stack.tif", _pages((_gray(np.uint8, 255), {}), (_gray(np.uint8, 255), {})), None),
        (
            "cmyk µ.tif",
            lambda path: _write_cmyk(path, _blot_rgb(), "pillow"),
            ("cmyk", True, False),
        ),
        (
            "lzw.tif",
            lambda path: _write_cmyk(path, _blot_rgb(), "pillow", compression="tiff_lzw"),
            ("cmyk", True, False),
        ),
        (
            "planar.tif",
            lambda path: _write_cmyk(path, _blot_rgb(), "planar"),
            ("cmyk", True, False),
        ),
        (
            "tagged cmyk.tif",
            _separated(_inks(_rgb8()), extratags=[(332, 3, 1, 1, True)]),
            ("cmyk", True, False),
        ),
        (
            "not cmyk.tif",
            _separated(_inks(_rgb8()), extratags=[(332, 3, 1, 2, True)]),
            ("other", True, False),
        ),
        (
            "ink set 3.tif",
            _separated(_inks(_rgb8()), extratags=[(332, 3, 1, 3, True)]),
            ("other", True, False),
        ),
        ("cmyk β.jpg", _cmyk_jpeg, ("cmyk", True, False)),
        (
            "jpeg cmyk.tif",
            lambda path: _write_cmyk(path, _blot_rgb(), "pillow", compression="jpeg"),
            ("cmyk", True, False),
        ),
        # A copy saved as RGB or grayscale that is more than 8-bit or keeps an
        # extra channel (alpha, a fifth ink) is read for certain only
        # uncompressed (#207 review; compressed, only unassociated alpha is
        # read, #232): the refusal says so.
        (
            "16-bit.tif",
            _separated(_inks(_blot_rgb().astype(np.uint16) * 257)),
            ("cmyk", True, True),
        ),
        (
            "alpha.tif",
            _separated(np.dstack([_inks(_rgb8()), _ALPHA]), extrasamples=[2]),
            ("cmyk", True, True),
        ),
        (
            "five inks.tif",
            _separated(
                np.dstack([_inks(_rgb8()), _gray(np.uint8, 255)]),
                extratags=[(334, 3, 1, 5, True)],
            ),
            ("cmyk", True, True),
        ),
        ("five inks, no extra samples.tif", _five_inks_untagged, ("cmyk", True, True)),
        (
            "float inks.tif",
            _separated(_inks(_rgb8()).astype(np.float32) / 255),
            ("cmyk", True, True),
        ),
        # What the file holds, not its suffix (#207 review): a CMYK JPEG named
        # .png or .tif, and separated TIFFs named .png or .jpg.
        ("cmyk jpeg.png", _cmyk_jpeg, ("cmyk", True, False)),
        ("cmyk jpeg µ.tif", _cmyk_jpeg, ("cmyk", True, False)),
        ("cmyk tiff.png", _separated(_inks(_rgb8())), ("cmyk", True, False)),
        (
            "spot inks.jpg",
            _separated(_inks(_rgb8()), extratags=[(332, 3, 1, 2, True)]),
            ("other", True, False),
        ),
        # A TIFF of either byte order, classic or BigTIFF; of other inks or
        # several images, which Pillow would take for one image of CMYK.
        (
            "big-endian spots.png",
            _separated(_inks(_rgb8()), byteorder=">", extratags=[(332, 3, 1, 2, True)]),
            ("other", True, False),
        ),
        (
            "bigtiff stack.png",
            _separated(np.stack([_inks(_rgb8())] * 2), bigtiff=True),
            ("cmyk", False, False),
        ),
        (
            "big-endian bigtiff stack.jpg",
            _separated(np.stack([_inks(_rgb8())] * 2), bigtiff=True, byteorder=">"),
            ("cmyk", False, False),
        ),
        ("cmyk stack.png", _separated(np.stack([_inks(_rgb8())] * 2)), ("cmyk", False, False)),
        ("rgb jpeg.tif", lambda path: Image.fromarray(_rgb8()).save(path, format="JPEG"), None),
        ("png.tif", lambda path: Image.fromarray(_rgb8()).save(path, format="PNG"), None),
        ("gray tiff.png", lambda path: tifffile.imwrite(path, _gray(np.uint8, 255)), None),
        # Several images, which reading refuses too (#207 review): a stack as
        # tifffile and Pillow write one, and two images of different sizes.
        (
            "stack µ.tif",
            _separated(np.stack([_inks(_rgb8())] * 2)),
            ("cmyk", False, False),
        ),
        ("pillow stack.tif", _pillow_cmyk_stack, ("cmyk", False, False)),
        (
            "two sizes.tif",
            _pages((_inks(_rgb8()), _INKS), (_inks(_rgb8())[:3, :4], _INKS)),
            ("cmyk", False, False),
        ),
        (
            "spot stack.tif",
            _separated(np.stack([_inks(_rgb8())] * 2), extratags=[(332, 3, 1, 2, True)]),
            ("other", False, False),
        ),
        (
            "16-bit stack.tif",
            _separated(np.stack([_inks(_rgb8().astype(np.uint16) * 257)] * 2)),
            ("cmyk", False, True),
        ),
        # The ink set of the first separated image names the inks.
        (
            "cmyk then spot.tif",
            _pages((_inks(_rgb8()), _INKS), (_inks(_rgb8()), _SPOT)),
            ("cmyk", False, False),
        ),
        (
            "spot then cmyk.tif",
            _pages((_inks(_rgb8()), _SPOT), (_inks(_rgb8()), _INKS)),
            ("other", False, False),
        ),
        # Of several images, any one may be the one saved: one of more than 8
        # bits or with alpha asks for an uncompressed copy, whatever its inks.
        (
            "cmyk then 16-bit rgb.tif",
            _pages(
                (_inks(_rgb8()), _INKS), (_rgb8().astype(np.uint16) * 257, {"photometric": "rgb"})
            ),
            ("cmyk", False, True),
        ),
        (
            "cmyk then rgba.tif",
            _pages(
                (_inks(_rgb8()), _INKS),
                (np.dstack([_rgb8(), _ALPHA]), {"photometric": "rgb", "extrasamples": [2]}),
            ),
            ("cmyk", False, True),
        ),
        (
            "cmyk then gray.tif",
            _pages((_inks(_rgb8()), _INKS), (_gray(np.uint8, 255), {})),
            ("cmyk", False, False),
        ),
        # A reduced-resolution thumbnail page is no second image: reading takes it.
        (
            "thumbnail.tif",
            _pages((_inks(_rgb8()), _INKS), (_inks(_rgb8())[::2, ::2], _INKS | {"subfiletype": 1})),
            ("cmyk", True, False),
        ),
        # Printed first, a thumbnail is an image series of its own: two images.
        (
            "separated thumbnail then rgb.tif",
            _pages(
                (_inks(_rgb8())[::2, ::2], _INKS | {"subfiletype": 1}),
                (_rgb8(), {"photometric": "rgb"}),
            ),
            ("cmyk", False, False),
        ),
        # A first page of no inks is all the header looks at (#207 review: an
        # ordinary TIFF is not parsed twice); reading the file finds the inks
        # of a later image (test_reading_a_tiff_of_several_images_says_its_inks).
        ("gray then cmyk.tif", _pages((_gray(np.uint8, 255), {}), (_inks(_rgb8()), _INKS)), None),
        (
            "rgb thumbnail then cmyk.tif",
            _pages(
                (_rgb8()[::2, ::2], {"photometric": "rgb", "subfiletype": 1}),
                (_inks(_rgb8()), _INKS),
            ),
            None,
        ),
    ],
)
def test_the_header_tells_whether_a_file_holds_printing_inks(tmp_path, name, write, inks):
    # #207: an import refuses a file of printing inks; whether one is, from its
    # header alone: CMYK whatever the number or type of its inks, else the ink
    # set its separated TIFF names; whether it holds the single image reading
    # takes, so that the refusal can name both problems; and whether the copy
    # to import must be an uncompressed TIFF, so that following it works.
    path = tmp_path / name
    write(path)
    assert imaging.ink_set(path) == inks


def test_reading_takes_a_file_of_inks_as_one_image_as_its_ink_set_says(tmp_path):
    # The ink set's single_image is reading's own rule (one source): a file it
    # says holds several images is refused as a stack, with the same ink set,
    # and the thumbnail file it says is one image is read.
    stack = tmp_path / "stack.tif"
    _separated(np.stack([_inks(_rgb8())] * 2))(stack)
    two_sizes = tmp_path / "two sizes.tif"
    _pages((_inks(_rgb8()), _INKS), (_inks(_rgb8())[:3, :4], _INKS))(two_sizes)
    spots = tmp_path / "spot then cmyk 16.tif"
    _pages((_inks(_rgb8()), _SPOT), (_inks(_rgb8().astype(np.uint16) * 257), _INKS))(spots)
    for path in (stack, two_sizes, spots):
        with pytest.raises(
            ValueError, match=f"^{re.escape(repr(path.name))} holds more than one image "
        ) as info:
            load_image(path)
        assert info.value.inks == imaging.ink_set(path)
    assert info.value.inks == ("other", False, True)
    thumbnail = tmp_path / "thumbnail.tif"
    _pages((_inks(_rgb8()), _INKS), (_inks(_rgb8())[::2, ::2], _INKS | {"subfiletype": 1}))(
        thumbnail
    )
    assert load_image(thumbnail).pixels.tolist() == _rgb8().tolist()


@pytest.mark.parametrize(
    ("write", "inks"),
    [
        (_pages((_gray(np.uint8, 255), {}), (_inks(_rgb8()), _INKS)), ("cmyk", False, False)),
        (
            _pages((_rgb8(), {"photometric": "rgb"}), (_inks(_rgb8()), _INKS)),
            ("cmyk", False, False),
        ),
        (_pages((_gray(np.uint8, 255), {}), (_inks(_rgb8()), _SPOT)), ("other", False, False)),
        (
            _pages((_gray(np.uint8, 255), {}), (_inks(_rgb8()), _SPOT), (_inks(_rgb8()), _INKS)),
            ("other", False, False),
        ),
        (
            _pages((_gray(np.uint8, 255), {}), (_inks(_rgb8().astype(np.uint16) * 257), _INKS)),
            ("cmyk", False, True),
        ),
        (
            _pages(
                (_rgb8()[::2, ::2], {"photometric": "rgb", "subfiletype": 1}),
                (_inks(_rgb8()), _INKS),
            ),
            ("cmyk", False, False),
        ),
        (_pages((_gray(np.uint8, 255), {}), (_gray(np.uint8, 255), {})), None),
        (
            _pages((_rgb8(), {"photometric": "rgb"}), (_rgb8()[:3, :4], {"photometric": "rgb"})),
            None,
        ),
    ],
    ids=[
        "gray-then-cmyk",
        "rgb-then-cmyk",
        "gray-then-spot",
        "gray-spot-cmyk",
        "gray-then-16-bit-cmyk",
        "rgb-thumbnail-then-cmyk",
        "gray-stack",
        "rgb-two-sizes",
    ],
)
def test_reading_a_tiff_of_several_images_says_its_inks(tmp_path, write, inks):
    # #207 review: a TIFF whose first image is gray or RGB and a later one
    # inks is refused as several images, as before, and the refusal says the
    # inks, which the header (looking at the first page only) does not see:
    # so that the import can name both problems.
    path = tmp_path / "several µ.tif"
    write(path)
    with pytest.raises(imaging.NotSingleImageError) as info:
        load_image(path)
    assert str(info.value) == (
        "'several µ.tif' holds more than one image (such as a stack, the channels of a"
        " composite, or an image with a mask or a thumbnail), which Proteia does not import:"
        " only a single image can be quantified; save one of its images as its own file and"
        " import it again"
    )
    assert info.value.inks == inks


def _counting_series(monkeypatch) -> list[object]:
    """Count each time a TIFF's image series are asked for; the list of them."""
    built = []
    series = tifffile.TiffFile.series
    monkeypatch.setattr(
        tifffile.TiffFile, "series", property(lambda tif: built.append(tif) or series.fget(tif))
    )
    return built


@pytest.mark.parametrize(
    ("write", "builds"),
    [
        (lambda path: tifffile.imwrite(path, _gray(np.uint8, 255)), 0),
        (_pages(*[(_gray(np.uint8, 255), {})] * 3), 0),
        (_pages((_rgb8(), {"photometric": "rgb"}), (_inks(_rgb8()), _INKS)), 0),
        (lambda path: tifffile.imwrite(path, _rgb8(), photometric="cielab"), 0),
        (_separated(_inks(_rgb8())), 1),
        (_separated(np.stack([_inks(_rgb8())] * 2)), 1),
    ],
    ids=["gray", "gray-stack", "rgb-then-cmyk", "cielab", "cmyk", "cmyk-stack"],
)
def test_the_ink_set_reads_image_series_only_of_a_separated_first_page(
    tmp_path, monkeypatch, write, builds
):
    # #207 review: the header check costs an ordinary TIFF one page, so that a
    # multi-page one is not parsed into image series twice (here and when read).
    path = tmp_path / "pages.tif"
    write(path)
    built = _counting_series(monkeypatch)
    imaging.ink_set(path)
    assert len(built) == builds


def test_the_ink_set_of_a_file_over_pillows_size_limit_is_a_value_error(tmp_path, monkeypatch):
    # Pillow refuses it with an error of its own, which is neither; #234: in
    # Proteia's words, which say it has too many pixels, not that it is damaged.
    path = tmp_path / "large µ.jpg"
    _cmyk_jpeg(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)  # 48 pixels is over twice the limit
    with pytest.raises(imaging.ReadingRefusal) as info:
        imaging.ink_set(path)
    assert str(info.value) == (
        "large µ.jpg: it has more pixels than Proteia reads from a file of this kind; save the"
        " image as an uncompressed TIFF and import that"
    )


@pytest.mark.parametrize("name", ["damaged.tif", "damaged.png", "damaged.jpg"])
def test_the_ink_set_of_a_file_it_cannot_read_is_a_value_or_os_error(tmp_path, name):
    path = tmp_path / name
    with pytest.raises(FileNotFoundError):  # as the system says it, not rewrapped
        imaging.ink_set(path)
    path.write_bytes(b"not an image" * 10)
    with pytest.raises((ValueError, OSError)):
        imaging.ink_set(path)


@pytest.mark.parametrize(
    ("data", "words"),
    [
        (
            b"II*\x00",
            r"^damaged µ\.tif: it is damaged or not an image Proteia can read \(struct\.error\)",
        ),
        (b"II*\x00\x08\x00\x00\x00" + b"\xff" * 40, r"^suspicious number of tags"),  # tifffile's
    ],
    ids=["signature-only", "damaged-page"],
)
def test_the_ink_set_of_a_damaged_tiff_header_is_a_value_error(tmp_path, data, words):
    # #230: still so, as the import leaves a damaged header to reading.
    path = tmp_path / "damaged µ.tif"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=words):
        imaging.ink_set(path)


@pytest.mark.parametrize("error", [MemoryError, RecursionError])
@pytest.mark.parametrize("name", ["gray µ.tif", "gray α.png"])
def test_running_out_of_memory_while_reading_the_ink_set_is_not_a_damaged_header(
    tmp_path, monkeypatch, name, error
):
    # #230: a damaged header is a ValueError, which the import takes as a
    # header it cannot read and carries on without the ink check; running out
    # of memory or stack must not be, as in #217's and #224's header checks.
    # A TIFF's header is read by tifffile, any other file's by Pillow.
    path = tmp_path / name
    if path.suffix == ".tif":
        tifffile.imwrite(path, _gray(np.uint16, 65535))
    else:
        Image.fromarray(_gray(np.uint8, 255)).save(path)

    def run_out(*args, **kwargs):
        raise error("no room")

    monkeypatch.setattr(imaging.tifffile, "TiffFile", run_out)
    monkeypatch.setattr(imaging, "_pillow_mode", run_out)
    with pytest.raises(error, match="^no room$"):
        imaging.ink_set(path)
    monkeypatch.undo()
    assert imaging.ink_set(path) is None  # a gray file: the failure alone raised


def test_the_header_of_a_tiff_with_no_image_says_so(tmp_path, monkeypatch):
    # #207 review: the ink set says it as the other header checks do; #234
    # review 3: in the words of reading's refusal of a file of no image,
    # with its next step (it said "holds no image" alone).
    def refusal(name: str) -> str:
        return (
            f"{name!r} holds no image, so there is nothing to quantify: export the image again"
            " from the program that made it, as a TIFF or PNG file, and import that file"
        )

    empty = tmp_path / "empty.tif"
    empty.write_bytes(b"II*\x00" + bytes(4))  # the first page's offset is 0: no page
    for check in (imaging.ink_set, converts_cmyk, file_colours, load_image):
        with pytest.raises(imaging.NotSingleImageError) as info:
            check(empty)
        assert (str(info.value), info.value.inks, info.value.no_image) == (
            refusal("empty.tif"),
            None,
            True,
        )
        assert info.value.refusal("scan µ.tif", remedy=False) == (
            "'scan µ.tif' holds no image, so there is nothing to quantify"
        )
    separated = tmp_path / "no series.tif"
    _separated(_inks(_rgb8()))(separated)
    monkeypatch.setattr(tifffile.TiffFile, "series", property(lambda self: []))
    for check in (imaging.ink_set, converts_cmyk, file_colours):
        with pytest.raises(imaging.NotSingleImageError) as info:
            check(separated)
        assert str(info.value) == refusal("no series.tif")


# --- A TIFF named with another suffix (#215) ---


def _stack_of_two(**options):
    """A writer of a TIFF stack of two unlike 8-bit gray images."""
    gray = _gray(np.uint8, 255)
    return lambda path: tifffile.imwrite(path, np.stack([gray, gray[::-1]]), **options)


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".PNG"])
@pytest.mark.parametrize(
    ("options", "signature"),
    [
        ({}, b"II*\x00"),
        ({"byteorder": ">"}, b"MM\x00*"),
        ({"bigtiff": True}, b"II+\x00"),
        ({"bigtiff": True, "byteorder": ">"}, b"MM\x00+"),
    ],
    ids=["classic", "big-endian", "bigtiff", "big-endian-bigtiff"],
)
def test_a_tiff_of_several_images_named_otherwise_is_refused_as_reading_refuses_a_tif(
    tmp_path, options, signature, suffix
):
    # Told by its first bytes, whatever its suffix: Pillow, which reads a file
    # by its suffix, would take its first image and drop the other.
    path = tmp_path / f"stack µ{suffix}"
    _stack_of_two(**options)(path)
    assert path.read_bytes()[:4] == signature
    named_tif = tmp_path / "stack µ.tif"
    named_tif.write_bytes(path.read_bytes())
    with pytest.raises(imaging.NotSingleImageError) as read:
        load_image(named_tif)
    with pytest.raises(imaging.NotSingleImageError) as counted:
        imaging.check_misnamed_tiff(path)
    assert str(counted.value) == str(read.value).replace("stack µ.tif", path.name)
    assert str(counted.value) == (
        f"'stack µ{suffix}' holds more than one image (such as a stack, the channels of a"
        " composite, or an image with a mask or a thumbnail), which Proteia does not import:"
        " only a single image can be quantified; save one of its images as its own file and"
        " import it again"
    )
    assert counted.value.inks is read.value.inks is None


@pytest.mark.parametrize(
    ("write", "inks"),
    [
        (_pages((_gray(np.uint8, 255), {}), (_inks(_rgb8()), _INKS)), ("cmyk", False, False)),
        (_pages((_gray(np.uint8, 255), {}), (_inks(_rgb8()), _SPOT)), ("other", False, False)),
        (
            _pages((_gray(np.uint8, 255), {}), (_inks(_rgb8().astype(np.uint16) * 257), _INKS)),
            ("cmyk", False, True),
        ),
        (
            _pages(
                (_rgb8()[::2, ::2], {"photometric": "rgb", "subfiletype": 1}),
                (_inks(_rgb8()), _INKS),
            ),
            ("cmyk", False, False),
        ),
    ],
    ids=["gray-then-cmyk", "gray-then-spot", "gray-then-16-bit-cmyk", "rgb-thumbnail-then-cmyk"],
)
def test_counting_the_images_of_a_tiff_named_otherwise_says_its_inks_as_reading_does(
    tmp_path, write, inks
):
    # The header check looks at the first page only; the inks of a later
    # image are found when the images are counted, so that the import can
    # name both problems, as for a .tif.
    path = tmp_path / "several µ.jpg"
    write(path)
    with pytest.raises(
        imaging.NotSingleImageError, match=r"^'several µ\.jpg' holds more than one image "
    ) as info:
        imaging.check_misnamed_tiff(path)
    assert info.value.inks == inks


def _apng(path: Path) -> None:
    frames = [Image.fromarray(_gray(np.uint8, 255)), Image.fromarray(_gray(np.uint8, 255)[::-1])]
    frames[0].save(path, format="PNG", save_all=True, append_images=frames[1:])


@pytest.mark.parametrize(
    ("name", "write", "builds"),
    [
        # Named .tif or .tiff, reading counts its images: not parsed twice.
        ("stack.tif", _stack_of_two(), 0),
        ("stack.TIFF", _stack_of_two(), 0),
        # Not a TIFF: only its first bytes are read.
        ("gray.png", lambda path: Image.fromarray(_gray(np.uint8, 255)).save(path), 0),
        ("gray.jpg", lambda path: Image.fromarray(_gray(np.uint8, 255)).save(path), 0),
        ("frames.png", _apng, 0),
        ("rgb.tif.png", lambda path: Image.fromarray(_rgb8()).save(path, format="PNG"), 0),
        # A single image, which keeps the reader its suffix picks.
        ("gray tiff.png", lambda path: tifffile.imwrite(path, _gray(np.uint16, 65535)), 1),
        (
            "white is zero.jpg",
            lambda path: tifffile.imwrite(path, _gray(np.uint8, 255), photometric="miniswhite"),
            1,
        ),
        (
            "palette.jpeg",
            lambda path: tifffile.imwrite(
                path, _gray(np.uint8, 255), photometric="palette", colormap=_palette_map()
            ),
            1,
        ),
        ("lab.png", lambda path: tifffile.imwrite(path, _rgb8(), photometric="cielab"), 1),
        ("cmyk.png", _separated(_inks(_rgb8())), 1),
        (
            "thumbnail.png",
            _pages(
                (_gray(np.uint8, 255), {}), (_gray(np.uint8, 255)[::2, ::2], {"subfiletype": 1})
            ),
            1,
        ),
    ],
)
def test_only_a_tiff_named_otherwise_has_its_images_counted(
    tmp_path, monkeypatch, name, write, builds
):
    path = tmp_path / name
    write(path)
    built = _counting_series(monkeypatch)
    assert imaging.check_misnamed_tiff(path) is None
    assert len(built) == builds


@pytest.mark.parametrize(
    ("data", "words"),
    [
        (b"II*\x00", r"^damaged µ\.png: it is damaged or not an image Proteia can read \("),
        (b"II*\x00\x08\x00\x00\x00" + b"\xff" * 40, "suspicious number of tags"),
    ],
    ids=["signature-only", "damaged-page"],
)
def test_a_tiff_named_otherwise_whose_images_cannot_be_counted_is_a_value_error(
    tmp_path, data, words
):
    # So that the import leaves it to reading, which says why it cannot read it.
    path = tmp_path / "damaged µ.png"
    with pytest.raises(FileNotFoundError):  # as the system says it, not rewrapped
        imaging.check_misnamed_tiff(path)
    path.write_bytes(data)
    with pytest.raises(ValueError, match=words) as info:
        imaging.check_misnamed_tiff(path)
    assert not isinstance(info.value, imaging.NotSingleImageError)


@pytest.mark.parametrize("error", [MemoryError, RecursionError])
def test_running_out_of_memory_while_counting_is_not_a_damaged_file(tmp_path, monkeypatch, error):
    # A damaged file is left to reading (a ValueError the import ignores); a
    # MemoryError must not be, or a stack would be read as its first image.
    path = tmp_path / "stack µ.png"
    _stack_of_two()(path)

    def run_out(*args, **kwargs):
        raise error("no room")

    monkeypatch.setattr(imaging.tifffile, "TiffFile", run_out)
    with pytest.raises(error, match="^no room$"):
        imaging.check_misnamed_tiff(path)


def test_a_tiff_of_no_image_named_otherwise_is_refused_as_reading_refuses_a_tif(tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b"II*\x00" + bytes(4))  # the first page's offset is 0: no page
    with pytest.raises(imaging.NotSingleImageError) as info:
        imaging.check_misnamed_tiff(path)
    assert str(info.value) == (
        "'empty.png' holds no image, so there is nothing to quantify: export the image again"
        " from the program that made it, as a TIFF or PNG file, and import that file"
    )


@pytest.mark.parametrize(
    ("write", "no_image"),
    [
        (_stack_of_two(), False),
        (_pages((_gray(np.uint8, 255), {}), (_gray(np.uint8, 255)[:3, :4], {})), False),
        (lambda path: path.write_bytes(b"II*\x00" + bytes(4)), True),
    ],
    ids=["stack", "two-sizes", "no-image"],
)
def test_the_refusal_of_a_tiff_that_is_not_a_single_image_can_name_the_file_otherwise(
    tmp_path, write, no_image
):
    # #220: reading names the file it reads (the stored copy, on import); the
    # import names it as the user chose it, in the same words.
    path = tmp_path / "img-1.tif"
    write(path)
    with pytest.raises(imaging.NotSingleImageError) as info:
        load_image(path)
    assert info.value.no_image is no_image
    chosen = "β-actin 10 µM α.tif"
    assert info.value.refusal(chosen) == str(info.value).replace("'img-1.tif'", repr(chosen))
    assert info.value.refusal(chosen) == (
        "'β-actin 10 µM α.tif' holds no image, so there is nothing to quantify: export the"
        " image again from the program that made it, as a TIFF or PNG file, and import that"
        " file"
        if no_image
        else "'β-actin 10 µM α.tif' holds more than one image (such as a stack, the channels"
        " of a composite, or an image with a mask or a thumbnail), which Proteia does not"
        " import: only a single image can be quantified; save one of its images as its own"
        " file and import it again"
    )


@pytest.mark.parametrize(
    ("write", "inks", "no_image"),
    [
        (_stack_of_two(), None, False),
        (
            _pages((_gray(np.uint8, 255), {}), (_inks(_rgb8()), _INKS)),
            ("cmyk", False, False),
            False,
        ),
        (lambda path: path.write_bytes(b"II*\x00" + bytes(4)), None, True),
    ],
    ids=["stack", "gray-then-cmyk", "no-image"],
)
def test_the_refusal_of_a_tiff_that_is_not_a_single_image_survives_pickle_and_copy(
    tmp_path, write, inks, no_image
):
    # #220 review: built from the file's name, not from its message, it must
    # still be rebuilt by pickle (another process) and copy, with its inks
    # and whether it holds no image.
    path = tmp_path / "several µ.tif"
    write(path)
    with pytest.raises(imaging.NotSingleImageError) as info:
        load_image(path)
    error = info.value
    assert (error.inks, error.no_image) == (inks, no_image)
    protocols = range(pickle.HIGHEST_PROTOCOL + 1)
    rebuilt = [pickle.loads(pickle.dumps(error, protocol)) for protocol in protocols]
    for again in [*rebuilt, copy.copy(error), copy.deepcopy(error)]:
        assert type(again) is imaging.NotSingleImageError
        assert (str(again), again.args, again.inks, again.no_image) == (
            str(error),
            error.args,
            inks,
            no_image,
        )
        assert again.refusal("β α.png") == error.refusal("β α.png")


def test_reading_a_tiff_of_several_images_named_otherwise_takes_its_first_as_before(tmp_path):
    # #215 refuses it on import only: reading, for the images imported before,
    # still picks the reader by the suffix.
    path = tmp_path / "stack µ.png"
    _stack_of_two()(path)
    np.testing.assert_array_equal(load_image(path).array, _gray(np.uint8, 255))


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


# --- A reading refusal apart from the file's name (#234) ---


def _rebuilt(error: Exception) -> list[Exception]:
    """``error`` rebuilt by pickle, at every protocol, and by copy."""
    protocols = range(pickle.HIGHEST_PROTOCOL + 1)
    return [
        *(pickle.loads(pickle.dumps(error, protocol)) for protocol in protocols),
        copy.copy(error),
        copy.deepcopy(error),
    ]


@pytest.mark.parametrize(
    ("remedy", "as_read", "as_chosen"),
    [
        (
            "save it again",
            "img-1.tif: holds no frames; save it again",
            "cannot read 'β α.png': holds no frames; save it again",
        ),
        ("", "img-1.tif: holds no frames", "cannot read 'β α.png': holds no frames"),
    ],
    ids=["remedy", "no-remedy"],
)
def test_a_reading_refusal_names_the_file_read_and_gives_its_refusal_for_another(
    remedy, as_read, as_chosen
):
    # #234: the message names the file read (an image's stored copy); the
    # refusal names it as the user chose it, with the same words.
    error = imaging.ReadingRefusal("img-1.tif", "holds no frames", remedy)
    assert isinstance(error, ValueError)
    assert (str(error), error.refusal("β α.png")) == (as_read, as_chosen)
    assert (error.name, error.problem, error.remedy) == ("img-1.tif", "holds no frames", remedy)
    for again in _rebuilt(error):
        assert type(again) is imaging.ReadingRefusal
        assert (str(again), again.args, again.refusal("β α.png")) == (
            as_read,
            error.args,
            as_chosen,
        )


def test_a_colour_space_refusal_is_a_reading_refusal_that_survives_pickle_and_copy(tmp_path):
    path = tmp_path / "cielab µ.tif"
    tifffile.imwrite(path, np.zeros((6, 8, 3), dtype=np.uint8), photometric="cielab")
    with pytest.raises(UnsupportedColourSpaceError) as info:
        load_image(path)
    error = info.value
    problem = (
        "its pixels are stored in the CIELAB colour space, which Proteia does not convert to gray"
    )
    assert isinstance(error, imaging.ReadingRefusal)
    assert (error.problem, error.remedy) == (problem, "save the image as a gray or RGB TIFF")
    assert error.refusal("lab α.tif") == (
        f"cannot read 'lab α.tif': {problem}; save the image as a gray or RGB TIFF"
    )
    for again in _rebuilt(error):
        assert type(again) is UnsupportedColourSpaceError
        assert (str(again), again.problem, again.refusal("β.tif")) == (
            str(error),
            problem,
            error.refusal("β.tif"),
        )


def _frames_of(fmt: str, frames: list[np.ndarray]):
    def write(path):
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(path, format=fmt, save_all=True, append_images=images[1:])

    return write


_RGB_FRAMES = [_gray(np.uint8, 255)[..., None].repeat(3, axis=-1)] * 2


@pytest.mark.parametrize(
    ("name", "write", "words"),
    [
        (
            "two frames µ.png",
            _frames_of("PNG", [_gray(np.uint8, 255), _gray(np.uint8, 255)[::-1]]),
            "unsupported image layout (2, 6, 8)",
        ),
        (
            "rgb frames α.png",
            _frames_of("PNG", [_RGB_FRAMES[0], _RGB_FRAMES[1][::-1]]),
            "unsupported image layout (2, 6, 8, 3)",
        ),
        (
            "gif β.png",
            _frames_of("GIF", [_gray(np.uint8, 255), _gray(np.uint8, 255)[::-1]]),
            "unsupported image layout (2, 6, 8, 3)",
        ),
        (
            "five samples µ.tif",
            lambda path: tifffile.imwrite(
                path,
                np.zeros((6, 8, 5), dtype=np.uint8),
                photometric="minisblack",
                planarconfig="contig",
            ),
            "unsupported image layout (6, 8, 5)",
        ),
        (
            "not an image α.png",
            lambda path: path.write_bytes(b"not an image"),
            "it is damaged or not an image Proteia can read (UnidentifiedImageError); save the"
            " image again as a TIFF, PNG or JPEG file and import that",
        ),
        (
            "not an image β.jpg",
            lambda path: path.write_bytes(b"not an image"),
            "it is damaged or not an image Proteia can read (UnidentifiedImageError); save the"
            " image again as a TIFF, PNG or JPEG file and import that",
        ),
        (
            "lzw palette µ.tif",
            lambda path: (
                Image.fromarray(_gray(np.uint8, 255))
                .convert("P")
                .save(path, compression="tiff_lzw")
            ),
            "LZW compression of this 6x8 uint8 PALETTE image cannot be decoded by this"
            " installation; save the image as an uncompressed TIFF",
        ),
        (
            "truncated α.tif",
            lambda path: path.write_bytes(b"II*\x00"),
            "it is damaged or not an image Proteia can read (struct.error); save the image"
            " again as a TIFF, PNG or JPEG file and import that",
        ),
    ],
    ids=["animated", "animated-rgb", "gif", "five-samples", "png", "jpeg", "lzw", "damaged"],
)
def test_reading_refuses_a_file_apart_from_its_name(tmp_path, name, write, words):
    # #234: every refusal of reading itself names the file read, and gives
    # its refusal for another name: one Pillow cannot tell, and a damaged
    # one, in plain words with a next step; the others as before. A file
    # of several frames reads as before: the import refuses it.
    path = tmp_path / name
    write(path)
    with pytest.raises(imaging.ReadingRefusal) as info:
        imaging.read_file(path)
    assert str(info.value) == f"{name}: {words}"
    assert info.value.refusal("chosen µ.tif") == f"cannot read 'chosen µ.tif': {words}"
    assert str(path.parent) not in str(info.value)


def _mpo(sizes: list[tuple[int, int]]):
    """A writer of a JPEG of several pictures (MPO, as Pillow names it) of
    unlike RGB pixels, of ``sizes`` (height, width): a phone's photo with a
    smaller preview or depth map, or a stereo pair of the same size."""

    def write(path):
        rng = np.random.default_rng(len(sizes))
        pictures = [
            Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
            for height, width in sizes
        ]
        pictures[0].save(path, format="MPO", save_all=True, append_images=pictures[1:])

    return write


_MPO_SIZES = {
    "phone preview": [(120, 200), (60, 100)],
    "same size": [(120, 200), (120, 200)],
    "three pictures": [(120, 200), (60, 100), (30, 50)],
}


def _narrow_frames(count: int, width: int) -> list[np.ndarray]:
    """``count`` unlike 8-bit gray frames, 6 pixels high and ``width`` wide."""
    return [np.full((6, width), 40 * (index + 1), dtype=np.uint8) for index in range(count)]


_SEVERAL_FRAMES_WORDS = (
    "it holds more than one frame (such as an animated image), and only a single image can be"
    " quantified; save one frame as its own file and import it again"
)


@pytest.mark.parametrize(
    ("count", "width"), [(2, 2), (2, 3), (3, 4), (5, 3), (3, 2), (3, 8), (4, 8), (4, 60)]
)
def test_an_import_refuses_an_animated_png_whatever_its_frames(tmp_path, count, width):
    # #234 review: scikit-image reads its frames as (frames, height, width),
    # which frames 2 to 4 pixels wide pass for one image's channels, and 3
    # or 4 frames of any width it moves to the channels: it was imported as
    # one wrong image. The import refuses it (check_frames).
    path = tmp_path / "narrow frames µ.png"
    _frames_of("PNG", _narrow_frames(count, width))(path)
    with Image.open(path) as image:
        assert (image.n_frames, image.is_animated) == (count, True)
    if count in (3, 4) and width not in (3, 4):
        assert io.imread(path).shape == (6, width, count)  # why the layout cannot tell
    with pytest.raises(imaging.ReadingRefusal) as info:
        imaging.check_frames(path)
    assert str(info.value) == f"narrow frames µ.png: {_SEVERAL_FRAMES_WORDS}"


@pytest.mark.parametrize(("count", "width"), [(2, 3), (3, 60), (2, 60)])
def test_reading_an_animated_png_reads_it_as_before(tmp_path, count, width):
    # #234 review: reading itself is unchanged, so that an image a project
    # imported before (as main imported it) reads and measures as it did:
    # what scikit-image gives, or its layout refused as before.
    path = tmp_path / "frames µ.png"
    _frames_of("PNG", _narrow_frames(count, width))(path)
    decoded = io.imread(path)
    if decoded.ndim == 2 or (decoded.ndim == 3 and decoded.shape[-1] in (2, 3, 4)):
        np.testing.assert_array_equal(imaging.read_file(path).pixels, decoded)
    else:
        with pytest.raises(imaging.ReadingRefusal) as info:
            imaging.read_file(path)
        assert str(info.value) == f"frames µ.png: unsupported image layout {decoded.shape}"


def test_an_import_refuses_an_animated_gif_and_webp_and_takes_still_ones(tmp_path):
    # #234 review: Pillow reads a GIF or WebP named .png by its content; an
    # animated WebP was imported as its first frame, the others dropped.
    frames = [Image.fromarray(frame).convert("RGB") for frame in _narrow_frames(2, 8)]
    files = {}
    for fmt, options in (("GIF", {}), ("WEBP", {"lossless": True})):
        animated, still = tmp_path / f"animated {fmt} µ.png", tmp_path / f"still {fmt} α.png"
        frames[0].save(animated, format=fmt, save_all=True, append_images=frames[1:], **options)
        frames[0].save(still, format=fmt, **options)
        files[fmt] = animated, still
    for fmt, (animated, still) in files.items():
        with Image.open(animated) as image:
            assert (image.format, image.is_animated) == (fmt, True)
        with pytest.raises(imaging.ReadingRefusal) as info:
            imaging.check_frames(animated)
        assert str(info.value) == f"{animated.name}: {_SEVERAL_FRAMES_WORDS}"
        assert imaging.check_frames(still) is None


def test_frames_are_looked_for_only_in_formats_that_hold_them(tmp_path, monkeypatch):
    # Pillow reads a damaged TIFF named .jpg as its first image where it
    # cannot count its pages (a random sweep found one), and a JPEG of
    # several pictures (MPO) holds no frames: neither, nor a JPEG or a TIFF,
    # is looked at; a PNG is; a TIFF named .tif is not even opened.
    looked = []
    real = imaging._animated

    def looking(image):
        looked.append(image.format)
        return real(image)

    monkeypatch.setattr(imaging, "_animated", looking)
    files = {
        "gray tiff named µ.png": lambda path: tifffile.imwrite(path, _gray(np.uint8, 255)),
        "gray α.jpg": lambda path: Image.fromarray(_gray(np.uint8, 255)).save(path),
        "phone photo µ.jpg": _mpo([(6, 8), (3, 4)]),
        "stack β.tif": _stack_of_two(),
        "gray β.png": lambda path: Image.fromarray(_gray(np.uint8, 255)).save(path),
    }
    for name, write in files.items():
        path = tmp_path / name
        write(path)
        assert imaging.check_frames(path) is None
    assert looked == ["PNG"]


def test_frames_are_looked_for_without_opening_a_tiff_or_warning_of_size(
    tmp_path, monkeypatch, recwarn
):
    # A TIFF named .tif is not opened by Pillow at all (reading counts its
    # images); and a local file the user chose is opened without Pillow's
    # size warning, which is for untrusted input. 48 pixels a frame is over
    # the limit of 40, and under twice it, where Pillow refuses instead.
    opened = []
    real = Image.open

    def opening(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", opening)
    tif = tmp_path / "gray µ.tif"
    tifffile.imwrite(tif, _gray(np.uint8, 255))
    assert imaging.check_frames(tif) is None
    assert opened == []
    animated = tmp_path / "two frames α.png"
    _frames_of("PNG", _narrow_frames(2, 8))(animated)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 40)
    recwarn.clear()
    with pytest.raises(imaging.ReadingRefusal):
        imaging.check_frames(animated)
    assert (opened, [w.category for w in recwarn]) == (["two frames α.png"], [])


def test_a_file_pillow_cannot_open_is_left_to_reading(tmp_path):
    path = tmp_path / "not an image µ.png"
    path.write_bytes(b"not an image")
    assert imaging.check_frames(path) is None


def test_animation_is_told_by_pillow_and_a_failing_check_is_none():
    # is_animated stops at a second frame, where n_frames reads them all; a
    # check that fails says nothing of the pixels, which are read as before;
    # running out of memory or stack is no answer.
    class Animated:
        is_animated = True

    class Still:
        is_animated = False

    class NoCheck:
        pass

    def broken(error):
        class Broken:
            @property
            def is_animated(self):
                raise error("broken frame")

        return Broken()

    answers = [imaging._animated(image) for image in (Animated(), Still(), NoCheck())]
    answers += [imaging._animated(broken(error)) for error in (SyntaxError, OSError, EOFError)]
    assert answers == [True, False, False, False, False, False]
    for error in (MemoryError, RecursionError):
        with pytest.raises(error):
            imaging._animated(broken(error))


@pytest.mark.parametrize(
    "made",
    [
        lambda: OSError(errno.EIO, "Input/output error"),
        lambda: PermissionError(errno.EACCES, "Permission denied"),
        lambda: OSError(None, "The drive was disconnected", "img-1.png"),
    ],
    ids=["input/output", "access", "naming the file"],
)
def test_an_error_of_the_systems_in_the_frame_check_is_no_answer(tmp_path, monkeypatch, made):
    # #248 review: the frame check took any error for a still image, so a
    # drive's that a read trying again may not meet skipped the refusal of
    # an animation (scikit-image reads frames 3 pixels wide as channels).
    # An error of the system's, opening the file or telling its frames, is
    # raised as it is; a library's of a damaged file is still no answer.
    path = tmp_path / "two frames µ.png"
    narrow = [Image.fromarray(np.full((40, 3), level, dtype=np.uint8)) for level in (60, 120)]
    narrow[0].save(path, save_all=True, append_images=narrow[1:])

    class Broken:
        format = "PNG"

        def __init__(self, error: BaseException) -> None:
            self.error = error

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        @property
        def is_animated(self):
            raise self.error

    def raising(error: BaseException):
        def open_(*args, **kwargs):
            raise error

        return open_

    error = made()
    assert imaging._animated(Broken(OSError("broken frame"))) is False
    with pytest.raises(OSError) as raised:
        imaging._animated(Broken(error))
    assert raised.value is error
    for opened in (raising(error), lambda *args, **kwargs: Broken(error)):
        with monkeypatch.context() as patch:
            patch.setattr(Image, "open", opened)
            with pytest.raises(OSError) as raised:
                imaging.check_frames(path)
        assert raised.value is error
    with monkeypatch.context() as patch:  # a damaged file's: left to reading
        patch.setattr(Image, "open", raising(OSError("cannot identify image file")))
        assert imaging.check_frames(path) is None
    with pytest.raises(imaging.ReadingRefusal):  # with the drive back: refused for its frames
        imaging.check_frames(path)


@pytest.mark.parametrize("suffix", [".jpg", ".jpeg"])
@pytest.mark.parametrize("case", list(_MPO_SIZES))
def test_a_jpeg_of_several_pictures_reads_as_its_first_as_before(tmp_path, case, suffix):
    # #234 review: Pillow opens a JPEG of several pictures as MPO and counts
    # them as frames, but they are no frames of the picture (a phone's photo
    # of a blot keeps a preview or a depth map): the file reads as its first
    # picture, as before #234, and an import does not refuse it.
    path = tmp_path / f"phone photo µ{suffix}"
    _mpo(_MPO_SIZES[case])(path)
    with Image.open(path) as image:
        assert (image.format, image.n_frames) == ("MPO", len(_MPO_SIZES[case]))
        first = np.asarray(image.convert("RGB"))
    assert imaging.check_frames(path) is None
    read = imaging.read_file(path)
    assert first.shape == (120, 200, 3)
    np.testing.assert_array_equal(read.pixels, first)
    assert (read.lossy, read.cmyk, read.palette) == (True, None, False)


def test_a_layout_refused_in_a_file_of_one_frame_is_named_as_before(tmp_path, monkeypatch):
    # scikit-image giving a layout reading refuses is refused as before.
    for name in ("one frame µ.png", "one frame α.jpg"):
        path = tmp_path / name
        Image.fromarray(_gray(np.uint8, 255)).save(path)
        monkeypatch.setattr(io, "imread", lambda path: np.zeros((2, 6, 8), dtype=np.uint8))
        with pytest.raises(imaging.ReadingRefusal) as info:
            imaging.read_file(path)
        assert str(info.value) == f"{name}: unsupported image layout (2, 6, 8)"
        monkeypatch.undo()


_DAMAGED_WORDS = (
    "it is damaged or not an image Proteia can read ({}); save the image again as a TIFF, PNG"
    " or JPEG file and import that"
)
_STORED = PureWindowsPath(r"D:\lab µ\project α\images\img-1.png")


def _library_errors() -> dict[str, BaseException]:
    """Errors of the libraries that read a file, in their own words, by case:
    words that name the stored copy, its folder or nothing of it."""
    path = str(_STORED)
    return {
        "imageio's": ValueError(f"`{path}` can not be opened"),
        "tifffile's": tifffile.TiffFileError("not a TIFF file: header=b'\\x89PNG'"),
        "pillow's": OSError("image file is truncated"),
        "pillow's unidentified": UnidentifiedImageError(f"cannot identify image file {path!r}"),
        "a folder": OSError(f"no room in {_STORED.parent}"),
    }


@pytest.mark.parametrize("case", list(_library_errors()))
def test_a_refusal_holds_proteias_words_and_logs_the_librarys(caplog, case):
    # #234 review: a library's words may name the stored copy, in forms no
    # pattern can foresee; the refusal never holds them. It says the file is
    # damaged or not an image Proteia can read, and the error's kind; the
    # library's words go to the log, with the stored copy's path.
    error = _library_errors()[case]
    kind = {"tifffile's": "TiffFileError", "pillow's unidentified": "UnidentifiedImageError"}
    with caplog.at_level(logging.INFO, logger="proteia"):
        answer = imaging.named_refusal(error, "β gel µ.png", _STORED)
    assert answer == "cannot read 'β gel µ.png': " + _DAMAGED_WORDS.format(
        kind.get(case, type(error).__name__)
    )
    assert str(error) not in answer
    logged = [record.getMessage() for record in caplog.records]
    assert logged == [
        f"reading {_STORED} (offered as 'β gel µ.png') failed:"
        f" {kind.get(case, type(error).__name__)}: {error}"
    ]


def test_a_refusal_caused_by_a_refusal_of_proteias_has_no_cause_to_log(caplog):
    # #248: what is logged is an error of a library's or of the system's; a
    # refusal of reading whose cause is Proteia's own refusal (no words of a
    # library) logs none, and has no cause to log once per stored file.
    try:
        try:
            raise imaging.NotSingleImageError("img-1.png", None, no_image=False)
        except imaging.ReadingRefusal as inner:
            raise imaging.ReadingRefusal("img-1.png", "it holds frames") from inner
    except imaging.ReadingRefusal as outer:
        refusal = outer
    with caplog.at_level(logging.INFO, logger="proteia"):
        answer = imaging.named_refusal(refusal, "gel µ.png", _STORED)
    assert (answer, caplog.records) == ("cannot read 'gel µ.png': it holds frames", [])
    assert imaging.refusal_cause(refusal) is None


def test_a_refusal_of_reading_logs_its_cause_and_names_no_one_else(caplog):
    # Reading's own refusals are Proteia's words: those that a library's
    # error caused log that error; the others log nothing.
    cause = struct.error("unpack requires a buffer of 4 bytes")
    try:
        raise imaging._unreadable("img-1.png", cause) from cause
    except imaging.ReadingRefusal as refusal:
        damaged = refusal
    with caplog.at_level(logging.INFO, logger="proteia"):
        answers = [
            imaging.named_refusal(damaged, "gel µ.png", _STORED),
            imaging.named_refusal(
                imaging.UnsupportedColourSpaceError("img-1.png", "its pixels are in CIELAB"),
                "gel µ.png",
                _STORED,
            ),
        ]
    assert answers == [
        "cannot read 'gel µ.png': " + _DAMAGED_WORDS.format("struct.error"),
        "cannot read 'gel µ.png': its pixels are in CIELAB; save the image as a gray or RGB TIFF",
    ]
    assert [record.getMessage() for record in caplog.records] == [
        f"reading {_STORED} (offered as 'gel µ.png') failed: struct.error: {cause}"
    ]


# What a refusal of an error of the system's says to do (#246): the file
# read is the stored copy in the project's folder, whose drive or the file
# itself may be at fault (EINVAL is both a decoder's seek where a damaged
# file points and a drive that drops, on Windows), so the step holds for
# both: if it still fails, an import says to save the image again, which an
# image already in the project could not take without losing its boxes; of
# that, the refusal says the file may be damaged.
_CHECK_THE_DRIVE = "check that the project's folder and its drive can be read, then try again"
_SAVE_IT_AGAIN = (
    "; if it still fails, save the image again as a TIFF, PNG or JPEG file and import that"
)
_MAY_BE_DAMAGED = "; if it still fails, the file may be damaged"


def test_a_refusal_gives_the_systems_reason_without_its_path_and_what_to_do():
    # #234 review 3: an error of the system's (it has an errno, with a path
    # or without: a read that fails on a network drive) is given by its
    # reason, on an import as on an image already in the project. #246: with
    # a next step true whether the drive or the file is at fault. An error
    # without an errno is a library's (damage).
    stored = _STORED
    errors = (
        FileNotFoundError(2, "No such file or directory", str(stored)),
        PermissionError(13, "Permission denied", str(stored)),
        OSError(5, None, str(stored)),
        OSError(None, "The drive was disconnected", str(stored)),
        OSError(errno.EIO, "Input/output error"),
        PermissionError(errno.EACCES, "Permission denied"),
        OSError(errno.EINVAL, "Invalid argument"),
        OSError(errno.EIO, None),
    )
    reasons = [
        "No such file or directory",
        "Permission denied",
        "OSError",
        "The drive was disconnected",
        "Input/output error",
        "Permission denied",
        "Invalid argument",
        "OSError",
    ]
    for remedy, steps in (
        (True, f"{_CHECK_THE_DRIVE}{_SAVE_IT_AGAIN}"),
        (False, f"{_CHECK_THE_DRIVE}{_MAY_BE_DAMAGED}"),
    ):
        answers = [
            imaging.named_refusal(error, "gel µ.png", stored, remedy=remedy) for error in errors
        ]
        assert answers == [f"cannot read 'gel µ.png': {reason}; {steps}" for reason in reasons]
    assert imaging.named_refusal(OSError("image file is truncated"), "gel µ.png", stored) == (
        f"cannot read 'gel µ.png': {_DAMAGED_WORDS.format('OSError')}"
    )


# Of an error of running out of memory or of stack while reading, by case:
# the error, what the refusal says is wrong, and what an import adds.
_RUNNING_OUT = {
    "memory": (
        lambda: MemoryError(),
        "Proteia ran out of memory reading it; close other programs and try again",
        ", or, if the file is very large, save a smaller copy (such as one cropped to the blot)"
        " and import that",
    ),
    # numpy's, of a private class: "numpy._core._exceptions._ArrayMemoryError".
    "numpy's memory": (
        lambda: np._core._exceptions._ArrayMemoryError((2**40,), np.dtype(np.uint8)),
        "Proteia ran out of memory reading it; close other programs and try again",
        ", or, if the file is very large, save a smaller copy (such as one cropped to the blot)"
        " and import that",
    ),
    "stack": (
        lambda: RecursionError("maximum recursion depth exceeded"),
        "Proteia ran out of stack reading it",
        "; save the image again as a TIFF, PNG or JPEG file and import that",
    ),
}


def _running_out_reads(tmp_path: Path) -> dict[str, tuple[object, Path, object, str]]:
    """Each reader's read of a file whose reading runs out of memory or of
    stack, by reader: the reader, the file, and the owner and name of what
    is made to run out (Pillow's open of a compressed TIFF, which tifffile
    cannot decode without imagecodecs)."""
    gray_tif = tmp_path / "gray µ.tif"
    _write(gray_tif, _gray(np.uint8, 255))
    gray_png = tmp_path / "gray β.png"
    Image.fromarray(_gray(np.uint8, 255)).save(gray_png)
    palette_tif = tmp_path / "palette µ.tif"
    _write(palette_tif, _gray(np.uint8, 255), photometric="palette", colormap=_palette_map())
    lzw_tif = tmp_path / "lzw α.tif"
    write_lzw_tiff(lzw_tif, _gray(np.uint16, 65535))
    return {
        "read_file": (imaging.read_file, gray_tif, imaging, "_read_tiff"),
        "read_file of a png": (imaging.read_file, gray_png, imaging, "_read_other"),
        "converts_cmyk": (converts_cmyk, gray_tif, imaging.tifffile, "TiffFile"),
        "reads_as_palette": (reads_as_palette, gray_png, imaging, "_pillow_mode"),
        "file_colours": (file_colours, gray_tif, imaging, "_tiff_colours"),
        "file_colours of a png": (file_colours, gray_png, imaging, "_pillow_mode"),
        "read_colours": (read_colours, palette_tif, imaging, "_tiff_colours"),
        "compressed tiff": (imaging.read_file, lzw_tif, Image, "open"),
    }


@pytest.mark.parametrize("case", list(_RUNNING_OUT))
@pytest.mark.parametrize(
    "reader",
    [
        "read_file",
        "read_file of a png",
        "converts_cmyk",
        "reads_as_palette",
        "file_colours",
        "file_colours of a png",
        "read_colours",
        "compressed tiff",
    ],
)
def test_running_out_of_memory_or_stack_while_reading_is_no_damage(
    tmp_path, monkeypatch, reader, case
):
    # #234 review 3: they were refused as damaged, with the error's kind
    # (numpy's private class path); a compressed TIFF's as a compression this
    # installation cannot decode. Where they are caught is as before (#233);
    # the refusal says what happened and what to do: close other programs
    # and try again; on an import, for a very large file, a smaller copy.
    read, path, owner, name = _running_out_reads(tmp_path)[reader]
    made, problem, step = _RUNNING_OUT[case]
    error = made()

    def run_out(*args, **kwargs):
        raise error

    monkeypatch.setattr(owner, name, run_out)
    with pytest.raises(imaging.ReadingRefusal) as info:
        read(path)
    assert info.value.__cause__ is error
    assert str(info.value) == f"{path.name}: {problem}{step}"
    assert info.value.refusal("gel µ.tif") == f"cannot read 'gel µ.tif': {problem}{step}"
    assert imaging.named_refusal(info.value, "gel µ.tif", _STORED, remedy=False) == (
        f"cannot read 'gel µ.tif': {problem}"
    )
    assert pickle.loads(pickle.dumps(info.value)).refusal("gel µ.tif") == (
        info.value.refusal("gel µ.tif")
    )


# An error of the system's that a library raises as it decodes a file, by
# case (#246): an errno, or a file name (an open's). Reading raises it as it
# is, for its refusal to give its reason and what to do (named_refusal):
# from the error alone, the file's drive or the file may be at fault.
_SYSTEM_ERRORS = {
    "invalid argument": lambda: OSError(errno.EINVAL, "Invalid argument"),
    "input/output": lambda: OSError(errno.EIO, "Input/output error"),
    "access": lambda: PermissionError(errno.EACCES, "Permission denied"),
    "naming the file": lambda: OSError(None, "The drive was disconnected", "gel µ.tif"),
}


def _decoding_reads(tmp_path: Path) -> dict[str, tuple[object, Path, object, str]]:
    """Each reader's decoding of a file, by case: the reader, the file, and
    the owner and name of what decodes it: tifffile's pixels of a TIFF,
    Pillow's of a PNG through scikit-image, tifffile's colour map of a
    palette TIFF read in its colours, and Pillow's of an LZW TIFF, which
    tifffile cannot decode without imagecodecs."""
    from PIL import ImageFile, TiffImagePlugin

    gray_tif = tmp_path / "gray µ.tif"
    _write(gray_tif, _gray(np.uint16, 65535))
    gray_png = tmp_path / "gray β.png"
    Image.fromarray(_gray(np.uint8, 255)).save(gray_png)
    palette_tif = tmp_path / "palette α.tif"
    _write(palette_tif, _gray(np.uint8, 255), photometric="palette", colormap=_palette_map())
    lzw_tif = tmp_path / "lzw α.tif"
    write_lzw_tiff(lzw_tif, _gray(np.uint16, 65535))
    return {
        "tiff pixels": (imaging.read_file, gray_tif, tifffile.TiffPageSeries, "asarray"),
        "png pixels": (imaging.read_file, gray_png, ImageFile.ImageFile, "load"),
        "tiff colours": (read_colours, palette_tif, tifffile.TiffPage, "colormap"),
        "lzw tiff pixels": (imaging.read_file, lzw_tif, TiffImagePlugin.TiffImageFile, "load"),
    }


def _decoder_raising(monkeypatch, owner: object, name: str, error: BaseException) -> None:
    """Make what decodes a file (``owner.name``, a method or a property) raise ``error``."""

    def decode(*args, **kwargs):
        raise error

    monkeypatch.setattr(owner, name, property(decode) if name == "colormap" else decode)


@pytest.mark.parametrize("case", list(_SYSTEM_ERRORS))
@pytest.mark.parametrize("reader", ["tiff pixels", "png pixels", "tiff colours", "lzw tiff pixels"])
def test_an_error_of_the_systems_while_a_library_decodes_is_raised_as_it_is(
    tmp_path, monkeypatch, reader, case
):
    # #246: whether a decoder's seek where a damaged file points or the
    # drive failed, the error cannot tell: it is raised as it is, for the
    # refusal to give its reason and a next step true of both. #248 review:
    # the decoding of a compressed TIFF took it for its compression, which
    # this installation cannot decode ("save the image as an uncompressed
    # TIFF"), whatever it was.
    read, path, owner, name = _decoding_reads(tmp_path)[reader]
    error = _SYSTEM_ERRORS[case]()
    _decoder_raising(monkeypatch, owner, name, error)
    with pytest.raises(OSError) as raised:
        read(path)
    assert raised.value is error


def test_a_compressed_tiff_whose_codec_fails_is_still_refused_for_its_compression(
    tmp_path, monkeypatch
):
    # #248 review: only an error of the system's passes; a codec's failure,
    # an error of no errno, is still refused as the compression this
    # installation cannot decode, with the error as its cause.
    from PIL import TiffImagePlugin

    read, path, owner, name = _decoding_reads(tmp_path)["lzw tiff pixels"]
    height, width = _gray(np.uint16, 65535).shape
    error = OSError("decoder error -2")
    _decoder_raising(monkeypatch, TiffImagePlugin.TiffImageFile, "load", error)
    with pytest.raises(imaging.ReadingRefusal) as refused:
        read(path)
    assert str(refused.value) == (
        f"{path.name}: LZW compression of this {height}x{width} uint16 MINISBLACK image cannot be"
        " decoded by this installation; save the image as an uncompressed TIFF"
    )
    assert refused.value.__cause__ is error


def test_a_refusal_for_an_image_already_imported_gives_no_remedy():
    # #234 review: an image already in the project is read again; a remedy
    # that says to import it again would lose its boxes. Its refusal says
    # what is wrong only.
    refusals = [
        imaging.ReadingRefusal("img-1.png", "it holds frames", "save one and import it again"),
        imaging.UnsupportedColourSpaceError("img-1.png", "its pixels are stored in CIELAB"),
        imaging.NotSingleImageError("img-1.png", None, no_image=False),
        imaging.NotSingleImageError("img-1.png", None, no_image=True),
        ValueError("image file is truncated"),
    ]
    answers = [imaging.named_refusal(e, "gel µ.png", _STORED, remedy=False) for e in refusals]
    assert answers == [
        "cannot read 'gel µ.png': it holds frames",
        "cannot read 'gel µ.png': its pixels are stored in CIELAB",
        "'gel µ.png' holds more than one image (such as a stack, the channels of a composite,"
        " or an image with a mask or a thumbnail), which Proteia does not import: only a single"
        " image can be quantified",
        "'gel µ.png' holds no image, so there is nothing to quantify",
        "cannot read 'gel µ.png': it is damaged or not an image Proteia can read (ValueError)",
    ]
    assert [imaging.named_refusal(e, "gel µ.png", _STORED) for e in refusals[:4]] == [
        "cannot read 'gel µ.png': it holds frames; save one and import it again",
        "cannot read 'gel µ.png': its pixels are stored in CIELAB; save the image as a gray or"
        " RGB TIFF",
        "'gel µ.png' holds more than one image (such as a stack, the channels of a composite,"
        " or an image with a mask or a thumbnail), which Proteia does not import: only a single"
        " image can be quantified; save one of its images as its own file and import it again",
        "'gel µ.png' holds no image, so there is nothing to quantify: export the image again"
        " from the program that made it, as a TIFF or PNG file, and import that file",
    ]


def test_a_tiff_that_is_not_a_single_image_is_a_reading_refusal():
    # #234 review: one kind of error for every refusal of reading, so that an
    # import names the file as chosen in one place; its words are #220's.
    error = imaging.NotSingleImageError("stack µ.tif", None, no_image=False)
    assert isinstance(error, imaging.ReadingRefusal)
    assert str(error) == error.refusal("stack µ.tif")
    assert error.args == (str(error),)


def _damaged_reads(tmp_path: Path) -> dict[str, tuple[object, Path]]:
    """Each reader's read of a damaged file, by reader: the reader and the
    file. A TIFF of a signature only is cut off in its header; Pillow's open,
    and a TIFF's colour map, are made to fail with an error of their own."""
    truncated = tmp_path / "truncated µ.tif"
    truncated.write_bytes(b"II*\x00")
    named_png = tmp_path / "truncated α.png"
    named_png.write_bytes(b"II*\x00")
    gray_png = tmp_path / "gray β.png"
    Image.fromarray(_gray(np.uint8, 255)).save(gray_png)
    palette_tif = tmp_path / "palette µ.tif"
    _write(palette_tif, _gray(np.uint8, 255), photometric="palette", colormap=_palette_map())
    return {
        "read_file": (imaging.read_file, truncated),
        "converts_cmyk": (converts_cmyk, truncated),
        "check_misnamed_tiff": (imaging.check_misnamed_tiff, named_png),
        "ink_set": (imaging.ink_set, truncated),
        "reads_as_palette": (reads_as_palette, gray_png),
        "declared_pixel_type": (imaging.declared_pixel_type, truncated),
        "file_colours": (file_colours, truncated),
        "file_colours of a png": (file_colours, gray_png),
        "read_colours": (read_colours, palette_tif),
    }


@pytest.mark.parametrize(
    "reader",
    [
        "read_file",
        "converts_cmyk",
        "check_misnamed_tiff",
        "ink_set",
        "reads_as_palette",
        "declared_pixel_type",
        "file_colours",
        "file_colours of a png",
        "read_colours",
    ],
)
def test_every_reader_refuses_a_damaged_file_in_proteias_words(tmp_path, monkeypatch, reader):
    # #234 review: they gave the library's words ("not a readable image
    # (unpack requires a buffer of 4 bytes)"); a refusal holds Proteia's
    # only, with the error's kind, and the error as its cause.
    read, path = _damaged_reads(tmp_path)[reader]

    def broken(*args):
        raise struct.error("bad bytes")

    if path.name == "gray β.png":
        monkeypatch.setattr(imaging, "_pillow_mode", broken)
    if read is read_colours:
        monkeypatch.setattr(imaging, "_tiff_colours", broken)
    with pytest.raises(imaging.ReadingRefusal) as info:
        read(path)
    assert type(info.value) is imaging.ReadingRefusal
    assert str(info.value) == f"{path.name}: {_DAMAGED_WORDS.format('struct.error')}"
    assert type(info.value.__cause__) is struct.error


def test_a_file_over_pillows_size_limit_has_too_many_pixels_not_damage(tmp_path, monkeypatch):
    # #234 review: a scan of more pixels than Pillow reads (twice its limit,
    # about 179 million) is no damaged file: the refusal says so, and that
    # an uncompressed TIFF of it is read (tifffile has no such limit). #234
    # review 3: it names no format, as the file may be of another (a GIF
    # named .png, read by Pillow too).
    png, jpeg = tmp_path / "large µ.png", tmp_path / "large α.jpg"
    gif = tmp_path / "large gif β.png"
    Image.fromarray(_gray(np.uint8, 255)).save(png)
    Image.fromarray(_gray(np.uint8, 255)).save(jpeg)
    Image.fromarray(_gray(np.uint8, 255)).save(gif, format="GIF")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)  # 48 pixels is over twice the limit
    for path in (png, jpeg, gif):
        with pytest.raises(imaging.ReadingRefusal) as info:
            imaging.read_file(path)
        assert str(info.value) == (
            f"{path.name}: it has more pixels than Proteia reads from a file of this kind; save"
            " the image as an uncompressed TIFF and import that"
        )
        assert type(info.value.__cause__) is Image.DecompressionBombError
    copy = tmp_path / "large β.tif"
    tifffile.imwrite(copy, _gray(np.uint8, 255))
    np.testing.assert_array_equal(imaging.read_file(copy).pixels, _gray(np.uint8, 255))


def test_a_header_of_untyped_samples_or_of_no_image_is_a_reading_refusal(tmp_path):
    untyped = tmp_path / "untyped µ.tif"
    untyped.write_bytes(_declared_as(128, 4))
    empty = tmp_path / "no image α.tif"
    empty.write_bytes(b"II*\x00" + bytes(4))  # the first page's offset is 0: no page
    no_image = (
        "'no image α.tif' holds no image, so there is nothing to quantify: export the image"
        " again from the program that made it, as a TIFF or PNG file, and import that file"
    )
    cases = [
        (
            imaging.declared_pixel_type,
            untyped,
            imaging.ReadingRefusal,
            "untyped µ.tif: not a readable image (128-bit samples of format 4)",
        ),
        (converts_cmyk, empty, imaging.NotSingleImageError, no_image),
        (file_colours, empty, imaging.NotSingleImageError, no_image),
    ]
    for read, path, kind, words in cases:
        with pytest.raises(imaging.ReadingRefusal) as info:
            read(path)
        assert (type(info.value), str(info.value)) == (kind, words)


def test_a_file_pillow_cannot_tell_is_refused_with_its_error_as_cause(tmp_path):
    path = tmp_path / "unknown µ.png"
    path.write_bytes(b"not an image")
    with pytest.raises(imaging.ReadingRefusal) as info:
        imaging.read_file(path)
    assert type(info.value.__cause__) is UnidentifiedImageError
    # #234 review 3: in the words of a damaged file, with the error's kind
    # (they were two wordings of one thing).
    assert str(info.value) == f"{path.name}: {_DAMAGED_WORDS.format('UnidentifiedImageError')}"
    with pytest.raises(UnidentifiedImageError):  # its header, as read before
        imaging.reads_as_palette(path)


def test_a_compressed_tiff_over_pillows_size_limit_is_a_reading_refusal(tmp_path, monkeypatch):
    path = tmp_path / "large lzw µ.tif"
    Image.fromarray(_gray(np.uint16, 65535)).save(path, compression="tiff_lzw")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)  # 48 pixels is over twice the limit
    with pytest.raises(imaging.ReadingRefusal) as info:
        load_image(path)
    assert info.value.refusal("big α.tif") == (
        "cannot read 'big α.tif': this compressed TIFF has more pixels than the compressed-TIFF"
        " reader accepts; save the image as an uncompressed TIFF"
    )
    assert str(info.value).startswith("large lzw µ.tif: this compressed TIFF has more pixels")


@pytest.mark.parametrize(
    "declared",
    [
        _declared(tifffile.PHOTOMETRIC.MINISWHITE, 1),
        _declared(tifffile.PHOTOMETRIC.MINISBLACK, 1, dtype=np.uint16),
    ],
    ids=["layout-pillow-changes", "result-differs"],
)
def test_a_tiff_pillow_would_read_otherwise_is_a_reading_refusal(tmp_path, declared):
    path = tmp_path / "lzw µ.tif"
    Image.fromarray(_gray(np.uint8, 255)).save(path, compression="tiff_lzw")
    with pytest.raises(imaging.ReadingRefusal) as info:
        _read_tiff_with_pillow(path, declared)
    words = (
        f"LZW compression of this 6x8 {declared.dtype} {declared.photometric.name} image cannot"
        " be decoded by this installation; save the image as an uncompressed TIFF"
    )
    assert (str(info.value), info.value.refusal("x.tif")) == (
        f"lzw µ.tif: {words}",
        f"cannot read 'x.tif': {words}",
    )


def test_a_decoder_error_of_a_compressed_tiff_is_a_reading_refusal(tmp_path, monkeypatch):
    path = tmp_path / "lzw µ.tif"
    Image.fromarray(_gray(np.uint8, 255)).save(path, compression="tiff_lzw")
    real = Image.open

    def broken(*args, **kwargs):
        image = real(*args, **kwargs)
        image.close()
        raise SyntaxError("broken")

    monkeypatch.setattr(Image, "open", broken)
    declared = _declared(tifffile.PHOTOMETRIC.MINISBLACK, 1)
    with pytest.raises(imaging.ReadingRefusal) as info:
        _read_tiff_with_pillow(path, declared)
    assert info.value.refusal("x.tif").startswith("cannot read 'x.tif': LZW compression of this")
    assert type(info.value.__cause__) is SyntaxError
