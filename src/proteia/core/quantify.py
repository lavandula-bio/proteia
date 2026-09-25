# SPDX-License-Identifier: Apache-2.0
"""Pixel-level quantification: integrate intensity within a box.

``net_signal`` is the densitometry quantity: integrated intensity *above a
background level*, with the signal direction handled. On a dark-band-on-light
image (chemiluminescence/colorimetric) a darker band must read as *more* signal,
so the contribution of each pixel is its darkness relative to the background
level. ``estimate_background`` gives a robust membrane baseline. (A raw pixel sum
is not comparable on its own: it conflates signal, box area, and how much
background the box happens to include.)

Multi-channel images are reduced to a single grayscale channel by
:func:`to_grayscale`, the one place that rule lives.
"""

from __future__ import annotations

import numpy as np

from proteia.core.model import Box, BoxSize


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Reduce an image to a 2D grayscale array.

    Color (3 or 4 channels last) becomes the unweighted mean of red, green and
    blue, which matches ImageJ's default conversion; alpha is ignored. Gray plus
    alpha (2 channels) keeps the gray channel.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] == 2:
        return image[..., 0]
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image[..., :3].mean(axis=-1)
    raise ValueError(f"unsupported image layout {image.shape}")


def _box_pixels(image: np.ndarray, box: Box, size: BoxSize) -> np.ndarray:
    """The grayscale pixels inside the box.

    Raises ValueError if the box extends beyond the image bounds.
    """
    gray = to_grayscale(image)
    height, width = gray.shape
    x0, y0, x1, y1 = box.rect(size)
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError("box extends beyond the image bounds")
    return gray[y0:y1, x0:x1]


def estimate_background(image: np.ndarray) -> float:
    """A robust background (membrane) level: the median of the grayscale image.

    On a typical blot the membrane dominates the frame, so its median is a good
    zeroth-order baseline to subtract. Local / lane-profile backgrounds are a
    later refinement.
    """
    return float(np.median(to_grayscale(image)))


def net_signal(
    image: np.ndarray,
    box: Box,
    size: BoxSize,
    background: float,
    *,
    dark_on_light: bool = True,
) -> float:
    """Integrated signal inside the box, *above* the background level.

    For ``dark_on_light`` (the default — dark bands on a light membrane), each
    pixel contributes how much *darker* than ``background`` it is; pixels at or
    above the background contribute nothing. For a light-on-dark image (e.g.
    fluorescence) the direction is flipped. The result is non-negative and grows
    with band strength. Equal-area boxes keep it comparable.

    Raises ValueError if the box extends beyond the image bounds.
    """
    pixels = _box_pixels(image, box, size)
    if dark_on_light:
        contribution = np.maximum(background - pixels, 0.0)
    else:
        contribution = np.maximum(pixels - background, 0.0)
    return float(contribution.sum())
