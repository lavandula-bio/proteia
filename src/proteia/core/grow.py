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

from typing import Final

import numpy as np

from proteia.core.model import Rect

# The growth settings place_box uses, reported in every export record.
REL_THRESHOLD: Final = 0.3  # grow while the signal is above this fraction of the seed's
NOISE_K: Final = 3.0  # ...and above this many times the membrane noise

_MAD_SIGMA: Final = 1.4826  # Gaussian sigma per median absolute deviation


def _signal(gray: np.ndarray, background: float, dark_on_light: bool) -> np.ndarray:
    """Per-pixel signal above background, with the direction handled."""
    if dark_on_light:
        return np.maximum(background - gray, 0.0)
    return np.maximum(gray - background, 0.0)


def mad_sigma(values: np.ndarray, center: float | None = None) -> float:
    """Robust Gaussian sigma of ``values``: 1.4826 times their median absolute
    deviation from ``center`` (their median by default), in their units. The
    membrane noise of :func:`grow_box`; :mod:`proteia.core.rowdetect` measures
    its pixel noise and one-sided spreads with it too."""
    c = np.median(values) if center is None else center
    return float(_MAD_SIGMA * np.median(np.abs(values - c)))


def grow_region(signal: np.ndarray, seed: tuple[int, int], threshold: float) -> Rect | None:
    """The growth rule itself: the bounding rect ``(x0, y0, x1, y1)``, half-open
    on the high edge, of the 4-connected region of ``signal > threshold`` that
    holds ``seed`` (an ``(x, y)`` pixel inside ``signal``); None if the seed's own
    signal is not above ``threshold``.

    :func:`grow_box` measures its signal and threshold and calls this; a caller
    that measures its own (:mod:`proteia.core.rowdetect`, with a local
    background and noise) passes them here and so grows a band exactly as a
    click does.
    """
    from scipy.ndimage import label

    sx, sy = seed
    if not signal[sy, sx] > threshold:
        return None
    labels, _ = label(signal > threshold)
    ys, xs = np.where(labels == labels[sy, sx])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def grow_box(
    gray: np.ndarray,
    seed: tuple[int, int],
    background: float,
    *,
    rel_threshold: float = REL_THRESHOLD,
    noise_k: float = NOISE_K,
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
    sx, sy = seed
    s = _signal(gray.astype(float), background, dark_on_light)
    threshold = max(s[sy, sx] * rel_threshold, noise_k * mad_sigma(gray))
    grown = grow_region(s, seed, threshold)
    if grown is None:
        return None
    x0, y0, x1, y1 = grown

    h_img, w_img = gray.shape
    if max_width is not None and (x1 - x0) > max_width:
        x0 = max(0, min(sx - max_width // 2, w_img - max_width))
        x1 = x0 + max_width
    if max_height is not None and (y1 - y0) > max_height:
        y0 = max(0, min(sy - max_height // 2, h_img - max_height))
        y1 = y0 + max_height
    return (x0, y0, x1, y1)
