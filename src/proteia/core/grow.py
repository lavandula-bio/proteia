# SPDX-License-Identifier: Apache-2.0
"""Grow an ROI box from a seed point by region growing on the signal.

The user points at a band (cheap); the box is grown to fit it (the part the
system is good at). Growth keeps the connected pixels whose signal — darkness
relative to the membrane background, for a dark-on-light image — is above a
fraction of the seed's signal, and stops where the signal decays back to
background. So a band surrounded by background grows to its own extent and no
further, and two bands separated by background grow independently without
colliding. An optional width/height cap is a safety valve against leaking into a
neighbouring lane on poor blots where bands touch.

Pure and GUI-independent so the growth rule can be unit-tested.
"""

from __future__ import annotations

import numpy as np

from proteia.core.model import Rect


def _signal(gray: np.ndarray, background: float, dark_on_light: bool) -> np.ndarray:
    """Per-pixel signal above background, with the direction handled."""
    if dark_on_light:
        return np.maximum(background - gray, 0.0)
    return np.maximum(gray - background, 0.0)


def _membrane_noise(gray: np.ndarray) -> float:
    """Robust estimate of membrane noise (1.4826 * MAD), in pixel units."""
    return float(1.4826 * np.median(np.abs(gray - np.median(gray))))


def grow_box(
    gray: np.ndarray,
    seed: tuple[int, int],
    background: float,
    *,
    rel_threshold: float = 0.3,
    noise_k: float = 3.0,
    max_width: int | None = None,
    max_height: int | None = None,
    dark_on_light: bool = True,
) -> Rect | None:
    """Grow a fitted box around ``seed`` (an ``(x, y)`` pixel) and return its
    rect ``(x0, y0, x1, y1)``, half-open on the high edge.

    The grow threshold is ``max(rel_threshold * seed_signal, noise_k * noise)``,
    where ``noise`` is the membrane noise level. The absolute noise floor is what
    makes growth stop at background: a fraction of the seed alone is too low for a
    faint band and would flood the whole membrane. Returns ``None`` if the seed
    sits at or below that threshold (i.e. on background / indistinguishable from
    noise). ``max_width`` / ``max_height``, if given, cap the box around the seed
    as a safety valve against leaking into a neighbour.
    """
    from scipy.ndimage import label

    sx, sy = seed
    s = _signal(gray.astype(float), background, dark_on_light)
    threshold = max(s[sy, sx] * rel_threshold, noise_k * _membrane_noise(gray))
    if s[sy, sx] <= threshold:
        return None

    mask = s > threshold
    labels, _ = label(mask)
    ys, xs = np.where(labels == labels[sy, sx])
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    h_img, w_img = gray.shape
    if max_width is not None and (x1 - x0) > max_width:
        x0 = max(0, min(sx - max_width // 2, w_img - max_width))
        x1 = x0 + max_width
    if max_height is not None and (y1 - y0) > max_height:
        y0 = max(0, min(sy - max_height // 2, h_img - max_height))
        y1 = y0 + max_height
    return (x0, y0, x1, y1)
