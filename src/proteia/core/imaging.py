# SPDX-License-Identifier: Apache-2.0
"""Reading image files into analysis arrays, and 8-bit display previews.

GUI-independent. The analysis array is always 2-D float64 in the file's own
value scale: a 16-bit scan keeps values up to 65535, so a saturated 16-bit pixel
stays distinguishable from a mid-range one. RGB input is reduced to gray by the
unweighted channel mean, (R+G+B)/3, which matches ImageJ's default conversion;
an alpha channel is ignored.

``bit_depth`` is the container depth of unsigned integer data (8 or 16), which
sets the detector limit for the over-exposure check. Other pixel types (float,
32-bit) have no fixed limit, so their depth is ``None`` and the import records a
warning. Problems found on import become :class:`~proteia.core.model.ImageWarning`
records, which the project keeps with the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from proteia.core.model import ImageWarning

TIFF_SUFFIXES = (".tif", ".tiff")
LOSSY_SUFFIXES = (".jpg", ".jpeg")
_BIT_DEPTHS = {np.dtype(np.uint8): 8, np.dtype(np.uint16): 16}

# Warning codes and messages recorded on import.
WARNINGS = {
    "lossy_format": (
        "JPEG compression changes pixel values; quantify the original TIFF if you have it."
    ),
    "color_channels_differ": (
        "The red, green and blue channels differ; they were averaged into one gray channel."
    ),
    "unknown_bit_depth": (
        "The pixel type has no fixed detector range, so over-exposure cannot be checked."
    ),
}


def _warning(code: str) -> ImageWarning:
    return ImageWarning(code=code, message=WARNINGS[code])


@dataclass(frozen=True)
class LoadedImage:
    """An image file read for analysis."""

    array: np.ndarray  # 2-D float64 analysis array, original value scale
    pixels: np.ndarray  # the pixels as stored (2-D, or 3-D with 3/4 channels), for display
    bit_depth: int | None  # container depth of unsigned integer data; None otherwise
    warnings: list[ImageWarning] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])


def read_pixels(path: str | os.PathLike[str]) -> np.ndarray:
    """Read the stored pixels: 2-D grayscale, or 3-D with 3 or 4 channels last.

    TIFF files are read with tifffile, and a multi-page stack is refused rather
    than mistaken for a color image. Other formats go through scikit-image.
    """
    path = Path(path)
    if path.suffix.lower() in TIFF_SUFFIXES:
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) > 1:
                raise ValueError(f"{path.name}: multi-page TIFF stacks are not supported")
            pixels = tif.asarray()
    else:
        from skimage import io

        pixels = io.imread(path)
    _check_layout(pixels, path.name)
    return pixels


def _check_layout(pixels: np.ndarray, name: str) -> None:
    if pixels.ndim == 2 or (pixels.ndim == 3 and pixels.shape[-1] in (3, 4)):
        return
    raise ValueError(f"{name}: unsupported image layout {pixels.shape}")


def to_analysis_array(pixels: np.ndarray) -> tuple[np.ndarray, list[ImageWarning]]:
    """The 2-D float64 analysis array, and warnings about the conversion."""
    _check_layout(pixels, "image")
    if pixels.ndim == 2:
        return pixels.astype(np.float64), []
    rgb = pixels[..., :3]
    warnings: list[ImageWarning] = []
    if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
        warnings.append(_warning("color_channels_differ"))
    return rgb.astype(np.float64).mean(axis=-1), warnings


def bit_depth_of(pixels: np.ndarray) -> int | None:
    """8 or 16 for unsigned integer pixels; ``None`` for any other pixel type."""
    return _BIT_DEPTHS.get(pixels.dtype)


def load_image(path: str | os.PathLike[str]) -> LoadedImage:
    """Read an image file into an analysis array, with its bit depth and warnings."""
    path = Path(path)
    pixels = read_pixels(path)
    array, warnings = to_analysis_array(pixels)
    depth = bit_depth_of(pixels)
    if depth is None:
        warnings.append(_warning("unknown_bit_depth"))
    if path.suffix.lower() in LOSSY_SUFFIXES:
        warnings.insert(0, _warning("lossy_format"))
    return LoadedImage(array=array, pixels=pixels, bit_depth=depth, warnings=warnings)


def preview(pixels: np.ndarray) -> np.ndarray:
    """An 8-bit display image with the same layout (2-D, or channels last).

    8-bit data is shown as stored. Anything else is stretched linearly from its
    minimum to its maximum, so 16-bit levels above 255 stay distinguishable; a
    constant image becomes black. For display only: analysis never uses it.
    """
    if pixels.dtype == np.uint8:
        return pixels.copy()
    values = pixels.astype(np.float64)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(pixels.shape, dtype=np.uint8)
    return np.round((values - lo) / (hi - lo) * 255.0).astype(np.uint8)
