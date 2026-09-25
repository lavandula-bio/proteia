# SPDX-License-Identifier: Apache-2.0
"""Tests for reading image files and rendering previews; files are generated here."""

import hashlib

import numpy as np
import pytest
import tifffile
from skimage import io

from proteia.core.imaging import load_image, preview, read_pixels, to_analysis_array
from proteia.core.model import ImageKind, ImageRef, Polarity, Project
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


def test_jpeg_import_records_a_lossy_format_warning(tmp_path):
    path = tmp_path / "blot β.jpg"
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
    with pytest.raises(ValueError, match="multi-page"):
        read_pixels(path)


@pytest.mark.parametrize("shape", [(4, 5, 2), (4, 5, 6), (2, 4, 5, 3)])
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
