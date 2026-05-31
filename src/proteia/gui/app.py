# SPDX-License-Identifier: Apache-2.0
"""Minimal napari GUI: load an image, draw a box, integrate its intensity.

Walking skeleton — proves the napari <-> core loop end to end. Run:

    uv run python -m proteia.gui.app [IMAGE_PATH]

If no path is given, a synthetic image is shown so the loop can be tried without
a file. napari/Qt and magicgui imports are local so importing this module stays
headless-safe (importing it must not require a display).
"""

from __future__ import annotations

import numpy as np

from proteia.core.model import Box, BoxSize
from proteia.core.quantify import integrate_box


def _synthetic_image() -> np.ndarray:
    """A small grayscale image with a few bright 'bands' to draw boxes over."""
    img = np.full((120, 200), 20.0)
    for cx in (40, 90, 140):
        img[50:70, cx : cx + 30] += 200.0
    return img


def _load_image(path: str | None) -> np.ndarray:
    if path is None:
        return _synthetic_image()
    from skimage import io

    return io.imread(path)


def launch(image_path: str | None = None) -> None:
    """Open napari with the image and a ROI layer; the box raw intensity updates live."""
    import napari
    from magicgui.widgets import Label
    from napari.utils.notifications import show_info

    image = _load_image(image_path)
    viewer = napari.Viewer()
    image_layer = viewer.add_image(image, name="blot")
    # Start with one default rectangle so the loop works with a single click;
    # the user can move, resize, or draw more.
    ih, iw = image.shape[0], image.shape[1]
    dy0, dy1 = int(ih * 0.40), int(ih * 0.55)
    dx0, dx1 = int(iw * 0.30), int(iw * 0.45)
    default_box = np.array([[dy0, dx0], [dy0, dx1], [dy1, dx1], [dy1, dx0]])
    shapes = viewer.add_shapes(
        [default_box],
        shape_type="rectangle",
        name="ROI",
        edge_color="red",
        face_color="transparent",
    )
    viewer.layers.selection = {shapes}

    result = Label(value="raw intensity: (draw or move a box)")
    viewer.window.add_dock_widget(result, area="right", name="Quantify")

    def update(*_) -> None:
        # Quantify the most recently drawn/edited box. Runs automatically whenever
        # the ROI layer changes, so no button click is needed.
        if len(shapes.data) == 0:
            result.value = "raw intensity: no box"
            return
        # napari shape vertices are (row, col) = (y, x); use the last shape.
        corners = np.asarray(shapes.data[-1])
        ys, xs = corners[:, 0], corners[:, 1]
        x0, y0 = int(round(xs.min())), int(round(ys.min()))
        w, h = int(round(xs.max() - xs.min())), int(round(ys.max() - ys.min()))
        if w <= 0 or h <= 0:
            result.value = "raw intensity: box has zero area"
            return
        try:
            raw = integrate_box(image_layer.data, Box(x=x0, y=y0), BoxSize(width=w, height=h))
        except Exception as exc:  # noqa: BLE001  (surface any failure to the user)
            result.value = f"raw intensity: cannot quantify ({exc})"
            return
        text = f"raw = {raw:.1f}  (box {w}x{h} at x={x0}, y={y0})"
        result.value = text
        show_info(text)  # also logged to the console

    shapes.events.data.connect(update)
    update()  # initial value for the default box
    show_info("Draw or move a box on the image; the value updates automatically.")
    napari.run()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Proteia walking skeleton")
    parser.add_argument("image", nargs="?", default=None, help="image file (optional)")
    args = parser.parse_args()
    launch(args.image)


if __name__ == "__main__":
    main()
