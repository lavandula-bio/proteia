# SPDX-License-Identifier: Apache-2.0
"""Reading image files into analysis arrays, and 8-bit display previews.

GUI-independent. The analysis array is always 2-D float64 in the file's own
value scale: a 16-bit scan keeps values up to 65535, so a saturated 16-bit pixel
stays distinguishable from a mid-range one. Color is reduced to gray by
:func:`proteia.core.quantify.to_grayscale`: the unweighted mean of the red, green
and blue channels, (R+G+B)/3, which matches ImageJ's default conversion; an alpha
channel is ignored, and gray plus alpha keeps the gray channel.

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
from proteia.core.quantify import to_grayscale

TIFF_SUFFIXES = (".tif", ".tiff")
LOSSY_SUFFIXES = (".jpg", ".jpeg", ".jpe", ".jfif")
# TIFF compression schemes that can change pixel values: old- and new-style JPEG,
# lossy JPEG (DNG), JPEG 2000, JPEG XR, WebP and JPEG XL.
_LOSSY_TIFF_COMPRESSION = frozenset({6, 7, 34892, 33003, 33005, 34712, 22610, 50001, 34927, 50002})
# Axes of a single 2-D image as tifffile reports them: gray, or samples last/first.
_SINGLE_IMAGE_AXES = ("YX", "YXS", "SYX")
_BIT_DEPTHS = {np.dtype(np.uint8): 8, np.dtype(np.uint16): 16}

# Warning codes and messages recorded on import.
WARNINGS = {
    "lossy_format": (
        "JPEG-type compression can change pixel values; quantify an uncompressed or"
        " losslessly compressed original if you have it."
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
    """An image read for analysis."""

    array: np.ndarray  # 2-D float64 analysis array, original value scale
    pixels: np.ndarray  # the pixels as stored (2-D, or 3-D with channels last), for display
    bit_depth: int | None  # container depth of unsigned integer data; None otherwise
    warnings: list[ImageWarning] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])


def _read_tiff(path: Path) -> tuple[np.ndarray, bool]:
    """The pixels of a single-image TIFF (channels last), and whether it is lossy.

    Extra pages that are only a reduced-resolution thumbnail of the image are
    fine; a stack or a multi-channel composite is refused rather than mistaken
    for color.
    """
    with tifffile.TiffFile(path) as tif:
        series = tif.series
        axes = series[0].axes if series else ""
        if len(series) != 1 or axes not in _SINGLE_IMAGE_AXES:
            raise ValueError(
                f"{path.name}: holds {len(series)} image series with axes {axes!r};"
                " only a single 2-D gray or color image can be quantified"
                " (stacks and multi-channel composites are not supported)"
            )
        keyframe = series[0].keyframe
        lossy = int(keyframe.compression) in _LOSSY_TIFF_COMPRESSION
        try:
            pixels = series[0].asarray()
        except ValueError as exc:
            if "imagecodecs" not in str(exc):
                raise
            raise ValueError(
                f"{path.name}: {keyframe.compression.name} compression cannot be decoded"
                " by this installation; save the image as an uncompressed TIFF"
            ) from exc
    # Planar color stores the channels first.
    if axes == "SYX" and pixels.ndim == 3 and pixels.shape[0] in (2, 3, 4):
        pixels = np.moveaxis(pixels, 0, -1)
    return pixels, lossy


def read_pixels(path: str | os.PathLike[str]) -> np.ndarray:
    """Read the stored pixels: 2-D gray, or 3-D with 2, 3 or 4 channels last.

    TIFF files are read with tifffile; other formats go through scikit-image.
    """
    return _read(Path(path))[0]


def _read(path: Path) -> tuple[np.ndarray, bool]:
    """Read a file; every failure to read it becomes a ``ValueError`` or ``OSError``."""
    try:
        if path.suffix.lower() in TIFF_SUFFIXES:
            pixels, lossy = _read_tiff(path)
        else:
            from skimage import io

            pixels, lossy = io.imread(path), path.suffix.lower() in LOSSY_SUFFIXES
    except (ValueError, OSError):
        raise
    except Exception as exc:  # a damaged file: struct.error, SyntaxError from Pillow, ...
        raise ValueError(f"{path.name}: not a readable image ({exc})") from exc
    _check_layout(pixels, path.name)
    return pixels, lossy


def _check_layout(pixels: np.ndarray, name: str) -> None:
    if pixels.ndim == 2 or (pixels.ndim == 3 and pixels.shape[-1] in (2, 3, 4)):
        return
    raise ValueError(f"{name}: unsupported image layout {pixels.shape}")


def to_analysis_array(pixels: np.ndarray) -> tuple[np.ndarray, list[ImageWarning]]:
    """The 2-D float64 analysis array, and warnings about the conversion.

    Refuses NaN or infinite pixels: they would make the background and every
    net signal meaningless.
    """
    _check_layout(pixels, "image")
    warnings: list[ImageWarning] = []
    if pixels.ndim == 3 and pixels.shape[-1] >= 3:
        rgb = pixels[..., :3]
        if not (
            np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])
        ):
            warnings.append(_warning("color_channels_differ"))
    array = to_grayscale(pixels).astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError("image has NaN or infinite pixel values")
    return array, warnings


def bit_depth_of(pixels: np.ndarray) -> int | None:
    """8 or 16 for unsigned integer pixels; ``None`` for any other pixel type."""
    return _BIT_DEPTHS.get(pixels.dtype)


def from_pixels(pixels: np.ndarray, *, lossy: bool = False) -> LoadedImage:
    """A :class:`LoadedImage` from pixels already in memory (e.g. generated)."""
    array, warnings = to_analysis_array(pixels)
    depth = bit_depth_of(pixels)
    if depth is None:
        warnings.append(_warning("unknown_bit_depth"))
    if lossy:
        warnings.insert(0, _warning("lossy_format"))
    return LoadedImage(array=array, pixels=pixels, bit_depth=depth, warnings=warnings)


def load_image(path: str | os.PathLike[str]) -> LoadedImage:
    """Read an image file into an analysis array, with its bit depth and warnings.

    ``lossy_format`` is recorded for JPEG files and for TIFF files that use JPEG
    compression.
    """
    pixels, lossy = _read(Path(path))
    return from_pixels(pixels, lossy=lossy)


def preview(pixels: np.ndarray) -> np.ndarray:
    """An 8-bit display image with the same layout (2-D, or channels last).

    8-bit data is shown as stored. Anything else is stretched linearly from its
    minimum to its maximum, so 16-bit levels above 255 stay distinguishable; a
    constant image becomes black, and NaN pixels black. For display only:
    analysis never uses it.
    """
    if pixels.dtype == np.uint8:
        return pixels.copy()
    if np.issubdtype(pixels.dtype, np.integer):  # every value is finite
        lo, hi = float(pixels.min()), float(pixels.max())
        finite = None
    else:
        finite = np.isfinite(pixels)
        valid = pixels[finite]
        if valid.size == 0:
            return np.zeros(pixels.shape, dtype=np.uint8)
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return np.zeros(pixels.shape, dtype=np.uint8)
    # float32 halves the working memory of a large 16-bit scan.
    scaled = (pixels.astype(np.float32) - np.float32(lo)) * np.float32(255.0 / (hi - lo))
    if finite is not None:
        scaled = np.where(finite, scaled, np.float32(0.0))
    return np.round(np.clip(scaled, 0.0, 255.0)).astype(np.uint8)


def display_rgb(pixels: np.ndarray) -> np.ndarray:
    """A ``uint8`` RGB view for display: color keeps its channels (alpha dropped);
    gray, with or without alpha, is repeated in three channels."""
    _check_layout(pixels, "image")
    if pixels.ndim == 3 and pixels.shape[-1] >= 3:
        return preview(pixels[..., :3])
    view = preview(to_grayscale(pixels))
    return np.stack([view, view, view], axis=-1)
