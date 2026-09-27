# SPDX-License-Identifier: Apache-2.0
"""Reading image files into analysis arrays, and 8-bit display previews.

GUI-independent. The analysis array is always 2-D float64 in the file's own
value scale: a 16-bit scan keeps values up to 65535, so a saturated 16-bit pixel
stays distinguishable from a mid-range one. Color is reduced to gray by
:func:`proteia.core.quantify.to_grayscale`: the unweighted mean of the red, green
and blue channels, (R+G+B)/3, which matches ImageJ's default conversion; an alpha
channel is ignored, and gray plus alpha keeps the gray channel.

A CMYK file (a CMYK JPEG, or a separated TIFF with cyan, magenta, yellow and
black inks) is converted to red, green and blue as it is read, then to gray like
any colour file (#131): through the ICC colour profile it embeds, to sRGB, or
else with the standard formula, and its ``cmyk_converted`` warning says which.
Every reader here gives the converted colours, so nothing downstream sees inks.
A TIFF in any other colour space that is not gray, RGB or a palette (CIELAB,
YCbCr, ...) is refused with the colour space's name rather than read wrong.

``bit_depth`` is the container depth of unsigned integer data (8 or 16), which
sets the detector limit for the over-exposure check. Other pixel types (float,
32-bit) have no fixed limit, so their depth is ``None`` and the import records a
warning. Problems found on import become :class:`~proteia.core.model.ImageWarning`
records, which the project keeps with the image.

For display only, :func:`read_colours` reads a file in its own colours, and
:func:`file_colours` says from the header whether it has colour to show.
"""

from __future__ import annotations

import io
import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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

# TIFF colour spaces read as stored: gray (either way round), red, green and blue,
# and a palette's indices. CMYK (SEPARATED) is converted; any other is refused.
_TIFF_AS_STORED = frozenset(
    {
        tifffile.PHOTOMETRIC.MINISWHITE,
        tifffile.PHOTOMETRIC.MINISBLACK,
        tifffile.PHOTOMETRIC.RGB,
        tifffile.PHOTOMETRIC.PALETTE,
    }
)
# The names a refusal gives the other TIFF colour spaces; the rest (a camera's
# raw mosaic, a mask) are named by their photometric interpretation.
_COLOUR_SPACES = {
    tifffile.PHOTOMETRIC.YCBCR: "YCbCr",
    tifffile.PHOTOMETRIC.CIELAB: "CIELAB",
    tifffile.PHOTOMETRIC.ICCLAB: "ICC L*a*b*",
    tifffile.PHOTOMETRIC.ITULAB: "ITU L*a*b*",
    tifffile.PHOTOMETRIC.LOGL: "LogL",
    tifffile.PHOTOMETRIC.LOGLUV: "LogLuv",
}
# The TIFF tags that say which inks a separated image holds (defaults: CMYK, 4).
_INK_SET, _NUMBER_OF_INKS, _INK_SET_CMYK = 332, 334, 1

# Warning codes and messages recorded on import. Each turns the over-exposure
# check off (clipping_depth), and its message says so. The cmyk_converted
# message names how the colours were converted: one of CMYK_CONVERSIONS.
WARNINGS = {
    "lossy_format": (
        "JPEG-type compression can change pixel values, so over-exposure cannot be"
        " checked; quantify an uncompressed or losslessly compressed original if you have it."
    ),
    "color_channels_differ": (
        "The red, green and blue channels differ; they were averaged into one gray"
        " channel, so over-exposure cannot be checked."
    ),
    "unknown_bit_depth": (
        "The pixel type has no fixed detector range, so over-exposure cannot be checked."
    ),
    "cmyk_converted": (
        "The file's colours are CMYK: they were converted to red, green and blue {how},"
        " then to gray. Values from a converted file are approximate, so over-exposure"
        " cannot be checked."
    ),
}
# How a CMYK file's colours were converted, as its cmyk_converted warning says.
CMYK_CONVERSIONS = {
    "profile": "through the ICC colour profile it embeds",
    "formula": "with the standard formula, as it embeds no ICC colour profile",
    "formula_16_bit": (
        "with the standard formula, as its ICC colour profile is applied to 8-bit CMYK only"
    ),
    "formula_unusable_profile": (
        "with the standard formula, as the ICC colour profile it embeds could not be used"
    ),
}
# The color_channels_differ message of a CMYK file: its channels are converted.
_CMYK_CHANNELS_DIFFER = (
    "The red, green and blue channels converted from the file's CMYK differ; they were"
    " averaged into one gray channel, so over-exposure cannot be checked."
)
# The warnings that make clipping_depth distrust a known bit depth. results
# names each in its clipping_not_checked notice (_UNCHECKED_WARNINGS, kept in
# step by a test), so a code added here needs a reason there.
UNTRUSTED_WARNINGS = frozenset({"lossy_format", "color_channels_differ", "cmyk_converted"})


def _warning(code: str, **fields: str) -> ImageWarning:
    return ImageWarning(code=code, message=WARNINGS[code].format(**fields))


@dataclass(frozen=True)
class LoadedImage:
    """An image read for analysis."""

    array: np.ndarray  # 2-D float64 analysis array, original value scale
    # The pixels as read (2-D, or 3-D with channels last; CMYK converted), for display.
    pixels: np.ndarray
    bit_depth: int | None  # container depth of unsigned integer data; None otherwise
    warnings: list[ImageWarning] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])


def _read_tiff(path: Path) -> tuple[np.ndarray, bool, str | None]:
    """The pixels of a single-image TIFF (channels last), whether it is lossy, and
    how its CMYK was converted (a :data:`CMYK_CONVERSIONS` key; None if it is
    not CMYK).

    Extra pages that are only a reduced-resolution thumbnail of the image are
    fine; a stack or a multi-channel composite is refused rather than mistaken
    for color, and so is a colour space other than gray, RGB, a palette or CMYK.
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
        cmyk = _holds_cmyk(keyframe, path.name)
        profile = keyframe.iccprofile if cmyk else None
        lossy = int(keyframe.compression) in _LOSSY_TIFF_COMPRESSION
        if keyframe.compression not in tifffile.TIFF.DECOMPRESSORS:
            # No decoder here without imagecodecs (LZW, JPEG): Pillow decodes it.
            # Pillow returns channels last; its result must match the declaration.
            shape = tuple(series[0].shape)
            expected = (shape[1], shape[2], shape[0]) if axes == "SYX" else shape
            declared = _Declared(
                keyframe.compression.name,
                expected,
                keyframe.dtype,
                keyframe.photometric,
                keyframe.samplesperpixel,
            )
            pixels = None
        else:
            pixels = series[0].asarray()
    if pixels is None:
        pixels = _read_tiff_with_pillow(path, declared)
    elif axes == "SYX" and pixels.ndim == 3:  # planar color stores the channels first
        pixels = np.moveaxis(pixels, 0, -1)
    if not cmyk:
        return pixels, lossy, None
    rgb, how = _cmyk_to_rgb(pixels[..., :4], profile)  # extra samples (alpha) dropped
    return rgb, lossy, how


class UnsupportedColourSpaceError(ValueError):
    """A file whose pixels are in a colour space Proteia does not read: a TIFF
    that is not gray, RGB, a palette or CMYK (CIELAB, YCbCr, ...), or a
    separated TIFF whose inks are not 8- or 16-bit CMYK. ``problem`` says what
    the file holds without naming it, for a message that names it otherwise."""

    def __init__(self, name: str, problem: str) -> None:
        super().__init__(f"{name}: {problem}; save the image as a gray or RGB TIFF")
        self.problem = problem


def _holds_cmyk(page: tifffile.TiffPage, name: str) -> bool:
    """Whether a TIFF's image holds CMYK inks, which reading converts; a colour
    space it does not read is refused (:class:`UnsupportedColourSpaceError`)."""
    if page.photometric == tifffile.PHOTOMETRIC.SEPARATED:
        _check_inks(page, name)
        return True
    if page.photometric not in _TIFF_AS_STORED:
        raise _colour_space_refusal(name, page.photometric)
    return False


def _colour_space_refusal(
    name: str, photometric: tifffile.PHOTOMETRIC | int
) -> UnsupportedColourSpaceError:
    space = _COLOUR_SPACES.get(photometric)
    if space is None:
        held = f"with photometric interpretation {getattr(photometric, 'name', photometric)}"
    else:
        held = f"in the {space} colour space"
    return UnsupportedColourSpaceError(
        name, f"its pixels are stored {held}, which Proteia does not convert to gray"
    )


def _check_inks(page: tifffile.TiffPage, name: str) -> None:
    """Refuse a separated TIFF whose inks are not 8- or 16-bit CMYK: the InkSet
    tag says CMYK (its default), and there are four inks besides extra samples
    such as alpha, as NumberOfInks says if present."""
    inks = page.samplesperpixel - len(page.extrasamples)
    declared = page.tags.valueof(_NUMBER_OF_INKS, inks)
    if page.tags.valueof(_INK_SET, _INK_SET_CMYK) != _INK_SET_CMYK:
        problem = "inks that are not CMYK (its InkSet tag says so)"
    elif inks != 4 or declared != 4:
        count = declared if declared != 4 else inks
        problem = f"{count} ink{'' if count == 1 else 's'}"
    elif page.dtype not in _BIT_DEPTHS:
        problem = f"{page.dtype} inks"
    else:
        return
    raise UnsupportedColourSpaceError(
        name,
        f"this separated TIFF has {problem}, and only CMYK with 8- or 16-bit inks"
        " can be converted to gray",
    )


def _cmyk_to_rgb(inks: np.ndarray, profile: bytes | None) -> tuple[np.ndarray, str]:
    """Red, green and blue from 8- or 16-bit CMYK inks (4 channels last), in the
    inks' own scale, and how (a :data:`CMYK_CONVERSIONS` key).

    8-bit inks with an embedded ICC profile go through it to sRGB (Pillow's
    ImageCms, which is LittleCMS), relative colorimetric with black point
    compensation, as Photoshop converts by default: of the usual intents, it
    brings a gray ramp converted to press CMYK the same way back closest to a
    line of slope 1 (the press's own black and white bound it at the ends).
    How close the colours come back depends on the intent the file was made
    with, which it does not say: through SWOP press CMYK, levels 40 to 230 of a
    ramp made with this intent come back within 14, made with the relative
    intent alone within 12, but made with the perceptual intent (Pillow's
    default) only within 27, where the formula strays 29.
    Otherwise, or when the profile cannot convert CMYK, the standard formula on
    inks scaled to 1: R = (1 - C)(1 - K), G = (1 - M)(1 - K), B = (1 - Y)(1 - K),
    rounded half up, as Pillow converts CMYK to RGB.
    """
    from PIL import ImageCms

    if profile and inks.dtype == np.uint8:
        try:
            return _through_profile(inks, profile), "profile"
        except (OSError, ValueError, ImageCms.PyCMSError):  # damaged, or not for CMYK
            how = "formula_unusable_profile"
    else:
        how = "formula_16_bit" if profile else "formula"
    top = np.iinfo(inks.dtype).max
    # (top - C)(top - K) + top // 2 fits in 32 bits at 16 bits: 65535**2 + 32767 < 2**32.
    rgb = top - inks[..., :3].astype(np.uint32)
    rgb *= top - inks[..., 3:].astype(np.uint32)
    rgb += top // 2
    rgb //= top
    return rgb.astype(inks.dtype), how


def _through_profile(inks: np.ndarray, profile: bytes) -> np.ndarray:
    """8-bit CMYK inks through the ICC ``profile`` to 8-bit sRGB."""
    from PIL import Image, ImageCms

    height, width = inks.shape[:2]
    image = Image.frombytes("CMYK", (width, height), np.ascontiguousarray(inks).tobytes())
    converted = ImageCms.profileToProfile(
        image,
        ImageCms.ImageCmsProfile(io.BytesIO(profile)),
        ImageCms.createProfile("sRGB"),
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        outputMode="RGB",
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    return np.asarray(converted)


@dataclass(frozen=True)
class _Declared:
    """What a TIFF's first image declares about its pixels."""

    codec: str
    shape: tuple[int, ...]  # channels last
    dtype: np.dtype
    photometric: tifffile.PHOTOMETRIC
    samples: int


def _read_tiff_with_pillow(path: Path, declared: _Declared) -> np.ndarray:
    """Decode a TIFF whose compression tifffile cannot decode on its own.

    Only the layouts Pillow reads with the same values as tifffile are accepted:
    8- or 16-bit black-is-zero gray, 8-bit RGB without alpha, and 8-bit CMYK
    without extra samples (its inks, converted afterwards as tifffile's are).
    Pillow inverts white-is-zero gray and un-premultiplies associated alpha, and
    it narrows 16-bit color, so those, and any result whose shape or pixel type
    differs from the declaration, are refused rather than read with changed values.
    """
    from PIL import Image

    refusal = (
        f"{path.name}: {declared.codec} compression of this"
        f" {'x'.join(map(str, declared.shape))} {declared.dtype}"
        f" {declared.photometric.name} image cannot be decoded by this installation;"
        " save the image as an uncompressed TIFF"
    )
    gray = declared.photometric is tifffile.PHOTOMETRIC.MINISBLACK and declared.samples == 1
    color = declared.photometric is tifffile.PHOTOMETRIC.RGB and declared.samples == 3
    cmyk = declared.photometric is tifffile.PHOTOMETRIC.SEPARATED and declared.samples == 4
    if not (gray or color or cmyk):
        raise ValueError(refusal)
    try:
        with warnings.catch_warnings():
            # Local files the user chose: Pillow's size guard is for untrusted input.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                pixels = np.asarray(image)
    except Image.DecompressionBombError as exc:
        raise ValueError(
            f"{path.name}: this compressed TIFF has more pixels than the compressed-TIFF"
            " reader accepts; save the image as an uncompressed TIFF"
        ) from exc
    except Exception as exc:  # Pillow's decoder errors are not all OSError
        raise ValueError(refusal) from exc
    if not pixels.dtype.isnative:  # e.g. a big-endian ("MM") 16-bit file
        pixels = pixels.astype(pixels.dtype.newbyteorder("="))
    if pixels.shape != declared.shape or pixels.dtype != declared.dtype.newbyteorder("="):
        raise ValueError(refusal)
    return pixels


def read_pixels(path: str | os.PathLike[str]) -> np.ndarray:
    """Read the pixels: 2-D gray, or 3-D with 2, 3 or 4 channels last, as stored
    except that CMYK is converted to red, green and blue (:func:`load_image`
    records how).

    TIFF files are read with tifffile (with Pillow for compressions tifffile cannot
    decode on its own); a CMYK JPEG with Pillow; other formats go through
    scikit-image.
    """
    return _read(Path(path))[0]


def _read(path: Path) -> tuple[np.ndarray, bool, str | None]:
    """Read a file: its pixels, whether it is lossy, and how its CMYK was converted
    (None if it is not CMYK); every failure to read it becomes a ``ValueError`` or
    ``OSError``."""
    try:
        if path.suffix.lower() in TIFF_SUFFIXES:
            pixels, lossy, cmyk = _read_tiff(path)
        else:
            pixels, cmyk = _read_other(path)
            lossy = path.suffix.lower() in LOSSY_SUFFIXES
    except (ValueError, OSError):
        raise
    except Exception as exc:  # a damaged file: struct.error, SyntaxError from Pillow, ...
        raise ValueError(f"{path.name}: not a readable image ({exc})") from exc
    _check_layout(pixels, path.name)
    return pixels, lossy, cmyk


def _read_other(path: Path) -> tuple[np.ndarray, str | None]:
    """The pixels of a file that is not a TIFF, and how its CMYK was converted.

    A CMYK JPEG is decoded by Pillow, which reads an Adobe JPEG's inverted inks
    the right way up, and converted with the ICC profile it embeds, if any;
    anything else goes through scikit-image.
    """
    from PIL import Image

    with warnings.catch_warnings():
        # Local files the user chose: Pillow's size guard is for untrusted input.
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if image.mode == "CMYK":
                inks, profile = np.asarray(image), image.info.get("icc_profile")
            else:
                inks = profile = None
    if inks is None:
        from skimage import io

        return io.imread(path), None
    return _cmyk_to_rgb(inks, profile)


def converts_cmyk(path: str | os.PathLike[str]) -> bool:
    """Whether reading the file converts CMYK (:func:`load_image` then records
    ``cmyk_converted``), from its header alone: a separated TIFF, or a JPEG in
    CMYK. A TIFF in a colour space reading refuses raises
    :class:`UnsupportedColourSpaceError`, as reading it does; a file it cannot
    read raises another ``ValueError``, or ``OSError``.
    """
    path = Path(path)
    try:
        if path.suffix.lower() in TIFF_SUFFIXES:
            with tifffile.TiffFile(path) as tif:
                if not tif.series:
                    raise ValueError(f"{path.name}: holds no image")
                return _holds_cmyk(tif.series[0].keyframe, path.name)
        return _pillow_mode(path) == "CMYK"
    except (ValueError, OSError):
        raise
    except Exception as exc:  # a damaged file: struct.error, Pillow's size limit, ...
        raise ValueError(f"{path.name}: not a readable image ({exc})") from exc


def _pillow_mode(path: Path) -> str:
    """The mode Pillow opens a file in, from its header."""
    from PIL import Image

    with warnings.catch_warnings():
        # A local file already imported: Pillow's size guard is for untrusted input.
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            return image.mode


def _check_layout(pixels: np.ndarray, name: str) -> None:
    if pixels.ndim == 2 or (pixels.ndim == 3 and pixels.shape[-1] in (2, 3, 4)):
        return
    raise ValueError(f"{name}: unsupported image layout {pixels.shape}")


def to_analysis_array(
    pixels: np.ndarray, *, cmyk: bool = False
) -> tuple[np.ndarray, list[ImageWarning]]:
    """The 2-D float64 analysis array, and warnings about the conversion; with
    ``cmyk``, the pixels are a CMYK file's converted colours, and a
    ``color_channels_differ`` warning says so.

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
            warnings.append(
                ImageWarning(code="color_channels_differ", message=_CMYK_CHANNELS_DIFFER)
                if cmyk
                else _warning("color_channels_differ")
            )
    array = to_grayscale(pixels).astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError("image has NaN or infinite pixel values")
    return array, warnings


def bit_depth_of(pixels: np.ndarray) -> int | None:
    """8 or 16 for unsigned integer pixels; ``None`` for any other pixel type."""
    return _BIT_DEPTHS.get(pixels.dtype)


def from_pixels(pixels: np.ndarray, *, lossy: bool = False, cmyk: str | None = None) -> LoadedImage:
    """A :class:`LoadedImage` from pixels already in memory (e.g. generated);
    ``cmyk`` says how a CMYK file's colours were converted into them (a
    :data:`CMYK_CONVERSIONS` key), for pixels read from one."""
    array, warnings = to_analysis_array(pixels, cmyk=cmyk is not None)
    depth = bit_depth_of(pixels)
    if depth is None:
        warnings.append(_warning("unknown_bit_depth"))
    if cmyk is not None:
        warnings.insert(0, _warning("cmyk_converted", how=CMYK_CONVERSIONS[cmyk]))
    if lossy:
        warnings.insert(0, _warning("lossy_format"))
    return LoadedImage(array=array, pixels=pixels, bit_depth=depth, warnings=warnings)


def clipping_depth(bit_depth: int | None, warnings: Iterable[ImageWarning]) -> int | None:
    """The bit depth whose detector limit the over-exposure check may trust, or None.

    None when the depth is unknown (float data), when lossy compression moved
    saturated pixels off the limit (``lossy_format``), when color was averaged
    into gray (``color_channels_differ``): a channel saturated alone never brings
    the mean to the limit, or when the colours were converted from CMYK
    (``cmyk_converted``): its values are inks turned into approximate colours,
    not detector counts, and a profile may keep black off 0. A 12- or 14-bit
    camera writing a 16-bit file is checked against the container limit, so its
    saturation goes unseen (a known limit).
    """
    if bit_depth is None or any(w.code in UNTRUSTED_WARNINGS for w in warnings):
        return None
    return bit_depth


def load_image(path: str | os.PathLike[str]) -> LoadedImage:
    """Read an image file into an analysis array, with its bit depth and warnings.

    ``lossy_format`` is recorded for JPEG files and for TIFF files that use JPEG
    compression; ``cmyk_converted`` for a CMYK file, naming how its colours were
    converted to red, green and blue.
    """
    pixels, lossy, cmyk = _read(Path(path))
    return from_pixels(pixels, lossy=lossy, cmyk=cmyk)


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


# PNG and JPEG modes whose pixels read as red, green and blue (then alpha): a
# palette is read as its colours, and CMYK converted.
_RGB_MODES = frozenset({"RGB", "RGBA", "P", "PA", "CMYK"})

FileColours = Literal["rgb", "palette"]


def _tiff_colours(path: Path) -> tuple[int, np.ndarray | None]:
    """The photometric interpretation of a TIFF's image and its colour map (3 x N,
    16-bit), from the header."""
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path.name}: holds no image")
        page = tif.series[0].keyframe
        photometric, colormap = int(page.photometric), page.colormap
    if colormap is not None and (colormap.ndim != 2 or colormap.shape[0] != 3):
        colormap = None  # not red, green and blue levels
    return photometric, colormap


def _is_gray_map(colormap: np.ndarray) -> bool:
    return bool(
        np.array_equal(colormap[0], colormap[1]) and np.array_equal(colormap[0], colormap[2])
    )


def file_colours(path: str | os.PathLike[str]) -> FileColours | None:
    """How a file holds colour, read from its header alone.

    ``"rgb"``: the channels :func:`read_pixels` gives, when there are three or
    four, are red, green and blue (then alpha); a palette PNG reads as its
    colours, and CMYK as its colours converted to red, green and blue (a
    separated TIFF whose inks are not CMYK is refused when read).
    ``"palette"``: a TIFF whose pixels read as indices into a colour map that
    is not gray, as an 8-bit ImageJ image with a colour lookup table is saved.
    None otherwise: a gray file (a gray colour map too), or a TIFF in another
    colour space (CIELAB, YCbCr), which reading refuses. For display only.
    Raises ``ValueError`` or ``OSError`` for a file it cannot read.
    """
    path = Path(path)
    try:
        if path.suffix.lower() in TIFF_SUFFIXES:
            photometric, colormap = _tiff_colours(path)
            if photometric in (tifffile.PHOTOMETRIC.RGB, tifffile.PHOTOMETRIC.SEPARATED):
                return "rgb"
            if (
                photometric == tifffile.PHOTOMETRIC.PALETTE
                and colormap is not None
                and not _is_gray_map(colormap)
            ):
                return "palette"
            return None
        mode = _pillow_mode(path)
    except (ValueError, OSError):
        raise
    except Exception as exc:  # a damaged file: struct.error, Pillow's size limit, ...
        raise ValueError(f"{path.name}: not a readable image ({exc})") from exc
    return "rgb" if mode in _RGB_MODES else None


def read_colours(path: str | os.PathLike[str]) -> np.ndarray:
    """A file's pixels in its own colours, for display only: as :func:`read_pixels`
    gives them (CMYK converted to red, green and blue), except that a palette
    TIFF's indices are looked up in its colour map (8-bit, 3 channels last), as
    a viewer shows them. Analysis never uses it.
    """
    path = Path(path)
    pixels = read_pixels(path)
    if pixels.ndim != 2 or path.suffix.lower() not in TIFF_SUFFIXES:
        return pixels
    try:
        photometric, colormap = _tiff_colours(path)
    except (ValueError, OSError):
        raise
    except Exception as exc:
        raise ValueError(f"{path.name}: not a readable image ({exc})") from exc
    if photometric != tifffile.PHOTOMETRIC.PALETTE or colormap is None:
        return pixels
    # A TIFF colour map is 16-bit: its high byte is the 8-bit level.
    table = (np.asarray(colormap, dtype=np.uint16).T >> 8).astype(np.uint8)
    return table[np.minimum(pixels, len(table) - 1)]
