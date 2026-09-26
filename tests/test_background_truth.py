# SPDX-License-Identifier: Apache-2.0
"""The local band background against a known truth (#83).

Synthetic blots built like the #83 benchmark: two proteins (target, loading
control) over eight lanes on an 8-bit dark-on-light membrane, bands of known
signal with grow-sized boxes centred on them, white noise of 2 levels, one fixed
seed. Four membranes: flat, a horizontal gradient of 25 levels, a broad hump 45
levels deep, and haze down each lane up to 18 levels. The fold-change of lane i
is ``(target_i / loading_i) / (target_0 / loading_0)``, its truth taken from the
bands' noise-free signal.

The tolerances come from the measured values (in the comments; seed 83) with a
margin. Where the whole-image median fails, the ring median removes the bias
(the gradient and the hump) or halves it (lane haze, which varies down each
lane); on a flat membrane its nets stay within 1.5 % of the whole-image median's.
"""

import math

import numpy as np
import pytest

from proteia.core.quantify import quantify_nets

LANES, PITCH, MARGIN, HEIGHT, MEMBRANE, NOISE = 8, 48, 40, 250, 215.0, 2.0
JITTER_X = (0, 1, -1, 2, 0, -2, 1, 0)  # lane-to-lane placement irregularity (px)
JITTER_Y = (0, -1, 1, 0, 1, -1, 0, 1)
HAZE_REL = (0.50, 1.00, 0.30, 0.80, 0.60, 0.20, 0.90, 0.40)
BAND_WX = 13.0  # across the lane; exp(-|dx / wx|^4), flat-topped
# protein: (row centre y, peak, wy down the lane, per-lane strength)
ROWS = {
    "target": (80, 140.0, 5.0, (0.45, 0.50, 0.40, 0.95, 1.00, 0.30, 0.20, 0.28)),
    "loading": (170, 120.0, 6.0, (0.85, 0.80, 0.88, 0.83, 0.86, 0.84, 0.79, 0.87)),
}


def _box_size(wy: float) -> tuple[int, int]:
    """The band's 30 %-of-peak extent: what growing a box from a click gives."""
    w = 2 * BAND_WX * math.log(1 / 0.3) ** 0.25
    h = 2 * wy * math.sqrt(math.log(1 / 0.3))
    return round(w), round(h)


def _blot(membrane_kind: str):
    """The image, every box with its size (target lanes, then loading lanes), and
    each band's true signal over the whole image and inside its box."""
    sizes = {name: _box_size(wy) for name, (_, _, wy, _) in ROWS.items()}
    widest = max(w for w, _ in sizes.values())
    width = 2 * MARGIN + (LANES - 1) * PITCH + widest
    y, x = np.mgrid[0:HEIGHT, 0:width].astype(float)
    lane_x = [MARGIN + widest // 2 + i * PITCH + JITTER_X[i] for i in range(LANES)]
    signal = np.zeros((HEIGHT, width))
    rects, box_sizes, true_total, true_box = [], [], [], []
    for name, (row_y, peak, wy, strength) in ROWS.items():
        w, h = sizes[name]
        for i in range(LANES):
            u = (i - (LANES - 1) / 2) / ((LANES - 1) / 2)
            x0 = lane_x[i] - w // 2
            y0 = row_y - h // 2 + round(2.0 * u * u) + JITTER_Y[i]  # a slight smile
            cx, cy = x0 + (w - 1) / 2, y0 + (h - 1) / 2
            band = peak * strength[i] * np.exp(-(np.abs((x - cx) / BAND_WX) ** 4))
            band *= np.exp(-(((y - cy) / wy) ** 2))
            signal += band
            rects.append((x0, y0, x0 + w, y0 + h))
            box_sizes.append((w, h))
            true_total.append(float(band.sum()))
            true_box.append(float(band[y0 : y0 + h, x0 : x0 + w].sum()))
    membrane = np.full((HEIGHT, width), MEMBRANE)
    if membrane_kind == "gradient":
        membrane += 12.5 * (x - (width - 1) / 2) / ((width - 1) / 2)
    elif membrane_kind == "hump":
        s = 0.25 * width
        spot = (x - 0.3 * (width - 1)) ** 2 + (y - 0.45 * (HEIGHT - 1)) ** 2
        membrane -= 45.0 * np.exp(-spot / (2 * s * s))
    elif membrane_kind == "haze":
        ends = 1 / (1 + np.exp(-(y - 30) / 4)) / (1 + np.exp((y - (HEIGHT - 30)) / 4))
        for i in range(LANES):
            down_lane = 0.6 + 0.4 * np.cos(2 * np.pi * y / (0.8 * HEIGHT) + i)
            across = np.exp(-(np.abs((x - lane_x[i]) / 14.0) ** 4))
            membrane -= 18.0 * HAZE_REL[i] * across * down_lane * ends
    noise = NOISE * np.random.default_rng(83).standard_normal((HEIGHT, width))
    image = np.clip(np.round(membrane - signal + noise), 0, 255)
    return image, rects, box_sizes, true_total, true_box


def _fold_changes(nets: list[float]) -> list[float]:
    target, loading = nets[:LANES], nets[LANES:]
    return [(target[i] / loading[i]) / (target[0] / loading[0]) for i in range(1, LANES)]


def _worst(measured: list[float], truth: list[float]) -> float:
    return max(abs(m / t - 1) for m, t in zip(measured, truth, strict=True))


def _measure(membrane_kind: str) -> dict[str, tuple[float, float, list[float]]]:
    """Per method: the worst fold-change error, the worst net error against the
    band's signal inside its box, and the nets."""
    image, rects, sizes, true_total, true_box = _blot(membrane_kind)
    out = {}
    for method in ("global_median", "ring_median"):
        nets = quantify_nets(image, rects, sizes, method=method, dark_on_light=True, integral=True)
        out[method] = (
            _worst(_fold_changes(nets), _fold_changes(true_total)),
            _worst(nets, true_box),
            nets,
        )
    return out


# membrane: (ring worst FC error, global-median worst FC error at least,
#            ring worst net error, global-median worst net error at least)
REMOVED = {
    # measured: ring 1.10 %, global 41.8 %; nets ring 0.65 %, global 46.0 %
    "gradient": (0.03, 0.30, 0.02, 0.30),
    # measured: ring 1.20 %, global 71.3 %; nets ring 1.95 %, global 75.9 %
    "hump": (0.03, 0.50, 0.03, 0.50),
    # measured: ring 7.88 %, global 17.6 %; nets ring 11.9 %, global 23.5 %
    "haze": (0.10, 0.15, 0.15, 0.20),
}


@pytest.mark.parametrize("membrane_kind", list(REMOVED))
def test_the_ring_removes_the_bias_the_image_median_leaves(membrane_kind):
    ring_fc, global_fc, ring_net, global_net = REMOVED[membrane_kind]
    found = _measure(membrane_kind)
    assert found["ring_median"][0] <= ring_fc
    assert found["global_median"][0] >= global_fc
    assert found["ring_median"][1] <= ring_net
    assert found["global_median"][1] >= global_net


def test_a_flat_membrane_keeps_the_image_median_results():
    found = _measure("flat")
    ring, image = found["ring_median"], found["global_median"]
    # measured: FC error ring 0.85 %, global 1.09 %; nets 0.52 % and 0.60 % off the truth
    assert ring[0] <= 0.02 and image[0] <= 0.02
    assert ring[1] <= 0.015
    # measured: nets 0.54 % and fold-changes 0.53 % apart
    assert _worst(ring[2], image[2]) <= 0.015
    assert _worst(_fold_changes(ring[2]), _fold_changes(image[2])) <= 0.015
