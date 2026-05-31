# SPDX-License-Identifier: Apache-2.0
"""Pixel-level quantification: integrate intensity within a box.

Walking-skeleton scope: sum the raw pixel intensities inside a box. Multi-channel
images are reduced to a single grayscale channel by averaging. Signal-direction
handling (inverting dark-on-light chemiluminescence, fluorescence channels) is a
later quantification-correctness concern and is deliberately not done here.
"""

from __future__ import annotations

import numpy as np

from proteia.core.model import Box, BoxSize


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Reduce an image to a 2D grayscale array (average across channels if RGB)."""
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image.mean(axis=-1)
    raise ValueError(f"unsupported image with {image.ndim} dimensions")


def integrate_box(image: np.ndarray, box: Box, size: BoxSize) -> float:
    """Sum of raw grayscale pixel intensities inside the box.

    Raises ValueError if the box extends beyond the image bounds.
    """
    gray = to_grayscale(image)
    height, width = gray.shape
    x0, y0, x1, y1 = box.rect(size)
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError("box extends beyond the image bounds")
    return float(gray[y0:y1, x0:x1].sum())
