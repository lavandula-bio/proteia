# SPDX-License-Identifier: Apache-2.0
"""napari GUI for placing equal-area ROI boxes and reading their intensity.

Every box shares one global, locked size (equal area) and no two boxes may
overlap. The user edits boxes freely; after each edit the constraints are
re-applied ("validate-and-correct", Route A — see :mod:`proteia.core.boxes`):

* a resized box snaps back to the locked size,
* a box moved onto another reverts to its last valid position,
* a newly drawn box that would overlap is dropped.

A width/height control changes the one shared size and resizes every box at
once; the change is rejected if it would force an overlap. Run:

    uv run python -m proteia.gui.app [IMAGE_PATH]

If no path is given, a synthetic image is shown so the loop can be tried
without a file. napari/Qt and magicgui imports are local so importing this
module stays headless-safe (importing it must not require a display).
"""

from __future__ import annotations

import numpy as np

from proteia.core.boxes import normalize_corners, reconcile, resize_all
from proteia.core.model import Box, BoxSize
from proteia.core.quantify import estimate_background, integrate_box, net_signal

# (x0, y0, x1, y1), half-open on the high edge — matches proteia.core.model.Rect.
Rect = tuple[int, int, int, int]


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


def _rect_to_corners(rect: Rect) -> np.ndarray:
    """A rect as napari rectangle vertices, in (row, col) = (y, x) order."""
    x0, y0, x1, y1 = rect
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])


def _initial_size(image: np.ndarray) -> BoxSize:
    """A starting locked box size proportional to the image."""
    ih, iw = int(image.shape[0]), int(image.shape[1])
    return BoxSize(width=max(4, iw // 8), height=max(4, ih // 12))


def launch(image_path: str | None = None) -> None:
    """Open napari with the image and an ROI layer governed by the box rules."""
    import napari
    from magicgui.widgets import CheckBox, Container, Label, SpinBox
    from napari.utils.notifications import show_info

    image = _load_image(image_path)
    ih, iw = int(image.shape[0]), int(image.shape[1])
    size = _initial_size(image)
    background = estimate_background(image)  # membrane baseline to subtract

    viewer = napari.Viewer()
    image_layer = viewer.add_image(image, name="blot")

    # One default box so the loop works immediately; the user can move it,
    # resize it (it snaps back), or draw more.
    x0, y0 = int(iw * 0.30), int(ih * 0.40)
    default = (x0, y0, x0 + size.width, y0 + size.height)
    shapes = viewer.add_shapes(
        [_rect_to_corners(default)],
        shape_type="rectangle",
        name="ROI",
        edge_color="red",
        face_color="transparent",
    )
    viewer.layers.selection = {shapes}

    # Source of truth: the last configuration that satisfied every invariant.
    state: dict = {"valid": [default], "size": size, "syncing": False}

    readout = Label(value="")
    width_in = SpinBox(value=size.width, min=2, max=iw, label="box width")
    height_in = SpinBox(value=size.height, min=2, max=ih, label="box height")
    dark_on_light = CheckBox(value=True, label="dark band on light")
    panel = Container(widgets=[width_in, height_in, dark_on_light, readout], labels=True)
    viewer.window.add_dock_widget(panel, area="right", name="Quantify")

    def _write_boxes(rects: list[Rect]) -> None:
        """Push the authoritative boxes back to the layer (suppressing events)."""
        state["syncing"] = True
        try:
            shapes.data = []
            if rects:
                shapes.add([_rect_to_corners(r) for r in rects], shape_type="rectangle")
        finally:
            state["syncing"] = False

    def _refresh_readout() -> None:
        sz = state["size"]
        invert = bool(dark_on_light.value)
        direction = "dark-on-light" if invert else "light-on-dark"
        lines = [
            f"box size: {sz.width} x {sz.height} (locked, equal area)",
            f"background level: {background:.0f}  ({direction})",
        ]
        if not state["valid"]:
            lines.append("no boxes — draw one on the image")
        for i, (bx0, by0, _, _) in enumerate(state["valid"]):
            try:
                img = image_layer.data
                net = net_signal(
                    img, Box(x=bx0, y=by0), sz, background, dark_on_light=invert
                )
                raw = integrate_box(img, Box(x=bx0, y=by0), sz)
                lines.append(f"  box {i}: net = {net:.0f}  (raw {raw:.0f})  at x={bx0}, y={by0}")
            except Exception as exc:  # noqa: BLE001  (surface any failure to the user)
                lines.append(f"  box {i}: cannot quantify ({exc})")
        readout.value = "\n".join(lines)

    def on_edit(*_) -> None:
        """Re-apply the box rules after any user edit, then refresh values."""
        if state["syncing"]:
            return
        new = [normalize_corners(np.asarray(c)) for c in shapes.data]
        corrected = reconcile(state["valid"], new, state["size"])
        if corrected != new:
            _write_boxes(corrected)  # an edit was rejected/snapped
        state["valid"] = corrected
        _refresh_readout()

    def on_resize(*_) -> None:
        """Change the one shared size; resize every box, or reject on overlap."""
        if state["syncing"]:
            return
        new_size = BoxSize(width=int(width_in.value), height=int(height_in.value))
        resized = resize_all(state["valid"], new_size)
        if resized is None:
            show_info("Size rejected: it would make boxes overlap.")
            state["syncing"] = True  # revert the spinboxes without re-triggering
            try:
                width_in.value = state["size"].width
                height_in.value = state["size"].height
            finally:
                state["syncing"] = False
            return
        state["size"] = new_size
        state["valid"] = resized
        _write_boxes(resized)
        _refresh_readout()

    shapes.events.data.connect(on_edit)
    width_in.changed.connect(on_resize)
    height_in.changed.connect(on_resize)
    dark_on_light.changed.connect(lambda *_: _refresh_readout())

    _refresh_readout()  # initial value for the default box
    show_info("Boxes share one locked size and cannot overlap; move or draw more.")
    napari.run()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Proteia ROI quantification")
    parser.add_argument("image", nargs="?", default=None, help="image file (optional)")
    args = parser.parse_args()
    launch(args.image)


if __name__ == "__main__":
    main()