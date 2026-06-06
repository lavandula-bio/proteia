# SPDX-License-Identifier: Apache-2.0
"""napari GUI: a project quantifies several proteins against shared lanes.

Quantify each protein as a repeatable step: Ctrl+click its bands, name it, pick
its role (target / loading control), and "Add protein" to capture its net per
lane (by left-to-right position) and clear the canvas for the next. Conditions
(one sample label per lane) are filled once at the end — after measuring you know
how many lanes there are — and label the positions in the results. Role is just a
label here; normalization consumes it later.

Per-protein placement (seed-grow): Ctrl+click a band and a box is grown to fit it.
The shared,
locked box size is the largest extent across every band clicked so far (so the
order of clicking does not matter and the biggest band is still fully captured);
every box uses that one size, keeping the measurement equal-area. Clicking the
same molecular weight across lanes yields one comparable value per condition.

A padding control adds an equal-area buffer (a percentage) around the auto-fitted
size without disturbing the detected base size. Boxes cannot overlap. Remove
boxes with a right-click (any box), Ctrl+Z (the last one), or the Clear all
button. Existing boxes can be dragged; an edit that breaks the size or overlap is
corrected. The readout is ordered left-to-right by position, not click order.

    uv run python -m proteia.gui.app [IMAGE_PATH]

If no path is given, a synthetic image is shown. napari/Qt and magicgui imports
are local so importing this module stays headless-safe.
"""

from __future__ import annotations

import numpy as np

from proteia.core.boxes import normalize_corners, reconcile, resize_all
from proteia.core.grow import grow_box
from proteia.core.model import Box, BoxSize, overlaps
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
    """A starting base box size proportional to the image (until the first grow)."""
    ih, iw = int(image.shape[0]), int(image.shape[1])
    return BoxSize(width=max(4, iw // 8), height=max(4, ih // 12))


def _padded(base: BoxSize, px_w: int, px_h: int) -> BoxSize:
    """The base size buffered by ``px_w`` / ``px_h`` pixels *per side*.

    Per side, so the change is always symmetric (both edges move together). The
    50%-200% bounds are enforced by the input widgets, whose step scales with the
    base so a tweak is fine on a small band and coarse on a large one.
    """
    return BoxSize(width=max(2, base.width + 2 * px_w), height=max(2, base.height + 2 * px_h))


def _pad_limits(base_dim: int) -> tuple[int, int, int]:
    """(min, max, step) for a per-side padding input given a base dimension.

    min/max keep the box within [50%, 200%] of base; step scales with the base
    (~5% per side) so large bands are not adjusted one pixel at a time.
    """
    return -round(base_dim * 0.25), round(base_dim * 0.5), max(1, round(base_dim * 0.05))


def launch(image_path: str | None = None) -> None:
    """Open napari with seed-grow ROI placement and net-signal readout."""
    import napari
    from magicgui.widgets import (
        CheckBox,
        ComboBox,
        Container,
        Label,
        LineEdit,
        PushButton,
        SpinBox,
    )
    from napari.utils.notifications import show_info

    image = _load_image(image_path)
    ih, iw = int(image.shape[0]), int(image.shape[1])
    background = estimate_background(image)  # membrane baseline to subtract

    viewer = napari.Viewer()
    image_layer = viewer.add_image(image, name="blot")
    shapes = viewer.add_shapes(
        name="ROI", edge_color="red", face_color="transparent", ndim=2
    )
    shapes.mode = "select"  # boxes can be moved, not free-drawn; grow adds them

    # Source of truth. ``base`` is the detected size (max over clicked bands);
    # ``size`` is what is drawn = base + padding. ``valid`` keeps click order.
    base0 = _initial_size(image)
    state: dict = {
        "valid": [], "base": base0, "size": base0, "pad_w": 0, "pad_h": 0,
        "lanes": [], "proteins": [], "syncing": False,
    }

    readout = Label(value="")
    proteins_lbl = Label(value="")
    # Project spine: one condition/sample label per lane, left-to-right.
    conditions_in = LineEdit(value="", label="conditions")
    width_in = SpinBox(value=base0.width, min=2, max=iw, label="box W")
    height_in = SpinBox(value=base0.height, min=2, max=ih, label="box H")
    # Width/height buffered independently, in pixels per side (symmetric). Bounds
    # and step are set from the base by _configure_padding_inputs (step scales with
    # size; the inputs pin at +-50%/+200% so there is no endless pressing).
    padding_w_in = SpinBox(value=0, min=-1000, max=1000, step=1, label="pad W")
    padding_h_in = SpinBox(value=0, min=-1000, max=1000, step=1, label="pad H")
    dark_on_light = CheckBox(value=True, label="dark on light")
    protein_in = LineEdit(value="", label="protein")
    mw_in = LineEdit(value="", label="MW (kDa)")
    role_in = ComboBox(choices=["target", "loading control"], value="target", label="role")
    add_btn = PushButton(text="Add protein")
    remove_protein_btn = PushButton(text="Remove last protein")
    clear_btn = PushButton(text="Clear all (boxes)")
    panel = Container(
        widgets=[
            conditions_in, width_in, height_in, padding_w_in, padding_h_in, dark_on_light,
            protein_in, mw_in, role_in, add_btn, remove_protein_btn, clear_btn,
            readout, proteins_lbl,
        ],
        labels=True,
    )
    viewer.window.add_dock_widget(panel, area="right", name="Quantify")

    def _write_boxes(rects: list[Rect]) -> None:
        """Push the authoritative boxes back to the layer (suppressing events)."""
        state["syncing"] = True
        try:
            shapes.selected_data = set()  # drop stale highlight before replacing data
            shapes.data = []
            if rects:
                shapes.add([_rect_to_corners(r) for r in rects], shape_type="rectangle")
            shapes.selected_data = set()
            shapes.mode = "select"
        finally:
            state["syncing"] = False

    def _set_size_inputs(base: BoxSize) -> None:
        state["syncing"] = True
        try:
            width_in.value, height_in.value = base.width, base.height
        finally:
            state["syncing"] = False

    def _configure_padding_inputs(base: BoxSize) -> None:
        """Set each padding input's bounds/step from the base, clamping its value.

        Bounds pin the box to [50%, 200%] of base (no endless pressing); the step
        scales with the base so adjustment stays usable on a 1000px band.
        """
        state["syncing"] = True
        try:
            for spin, dim in ((padding_w_in, base.width), (padding_h_in, base.height)):
                lo, hi, step = _pad_limits(dim)
                spin.min, spin.max, spin.step = lo, hi, step
                spin.value = min(max(int(spin.value), lo), hi)
        finally:
            state["syncing"] = False

    def _refresh_readout() -> None:
        sz = state["size"]
        invert = bool(dark_on_light.value)
        direction = "dark-on-light" if invert else "light-on-dark"
        b = state["base"]
        lines = [
            f"box size: {sz.width} x {sz.height} (base {b.width}x{b.height} + "
            f"{int(padding_w_in.value)}/{int(padding_h_in.value)} px/side)",
            f"background: {background:.0f}  ({direction})",
            "Ctrl+click a band to add a box; right-click to remove; Ctrl+Z to undo.",
        ]
        if not state["valid"]:
            lines.append("no boxes yet")
        # Order left-to-right by position, not click order.
        for i, (bx0, by0, _, _) in enumerate(sorted(state["valid"], key=lambda r: (r[0], r[1]))):
            try:
                img = image_layer.data
                net = net_signal(img, Box(x=bx0, y=by0), sz, background, dark_on_light=invert)
                raw = integrate_box(img, Box(x=bx0, y=by0), sz)
                lines.append(f"  {i}: net = {net:.0f}  (raw {raw:.0f})  at x={bx0}, y={by0}")
            except Exception as exc:  # noqa: BLE001  (surface any failure to the user)
                lines.append(f"  {i}: cannot quantify ({exc})")
        readout.value = "\n".join(lines)

    def _apply_size() -> bool:
        """Resize all boxes to base+padding; return False (and warn) on overlap."""
        eff = _padded(state["base"], padding_w_in.value, padding_h_in.value)
        resized = resize_all(state["valid"], eff, width=iw, height=ih)
        if resized is None:
            show_info("Size rejected: it would make boxes overlap.")
            return False
        state["size"] = eff
        state["valid"] = resized
        _write_boxes(resized)
        _refresh_readout()
        return True

    def _click_data_xy(event) -> tuple[int, int] | None:
        """Click position in integer image (x, y), or None if outside the image."""
        row, col = image_layer.world_to_data(event.position)[:2]
        x, y = int(round(col)), int(round(row))
        return (x, y) if (0 <= x < iw and 0 <= y < ih) else None

    def _seed_grow(x: int, y: int) -> None:
        invert = bool(dark_on_light.value)
        box = grow_box(image_layer.data, (x, y), background, dark_on_light=invert)
        if box is None:
            show_info("No band detected at that point.")
            return
        gx0, gy0, gx1, gy1 = box
        gw, gh = max(2, gx1 - gx0), max(2, gy1 - gy0)
        # The shared base is the largest band seen, so click order does not matter.
        if not state["valid"]:
            new_base = BoxSize(width=gw, height=gh)
        else:
            cur = state["base"]
            new_base = BoxSize(width=max(cur.width, gw), height=max(cur.height, gh))
        eff = _padded(new_base, padding_w_in.value, padding_h_in.value)
        resized = resize_all(state["valid"], eff, width=iw, height=ih)  # existing at new size
        if resized is None:
            show_info("Cannot grow the shared box size here without overlap; skipped.")
            return
        w, h = eff.width, eff.height
        cx, cy = (gx0 + gx1) // 2, (gy0 + gy1) // 2
        nx0 = min(max(0, cx - w // 2), iw - w)
        ny0 = min(max(0, cy - h // 2), ih - h)
        rect = (nx0, ny0, nx0 + w, ny0 + h)
        if any(overlaps(rect, r) for r in resized):
            show_info("Box would overlap an existing one; skipped.")
            return
        state["base"] = new_base
        state["size"] = eff
        state["valid"] = resized + [rect]
        _set_size_inputs(new_base)
        _configure_padding_inputs(new_base)  # base grew -> widen padding bounds/step
        _write_boxes(state["valid"])
        _refresh_readout()

    def _remove_at(x: int, y: int) -> None:
        for i, (rx0, ry0, rx1, ry1) in enumerate(state["valid"]):
            if rx0 <= x < rx1 and ry0 <= y < ry1:
                del state["valid"][i]
                _write_boxes(state["valid"])
                _refresh_readout()
                return

    def on_mouse(_viewer, event) -> None:
        """Ctrl+left-click grows a box; right-click removes the box under it."""
        ctrl = "Control" in event.modifiers
        if event.button == 1 and ctrl:
            xy = _click_data_xy(event)
            if xy:
                _seed_grow(*xy)
        elif event.button == 2:
            xy = _click_data_xy(event)
            if xy:
                _remove_at(*xy)

    def on_edit(*_) -> None:
        """Re-apply the box rules after a manual drag, then refresh values."""
        if state["syncing"]:
            return
        new = [normalize_corners(np.asarray(c)) for c in shapes.data]
        corrected = reconcile(state["valid"], new, state["size"])
        if corrected != new:
            _write_boxes(corrected)
        state["valid"] = corrected
        _refresh_readout()

    def on_size_change(*_) -> None:
        """Manual base width/height override."""
        if state["syncing"]:
            return
        prev = state["base"]
        state["base"] = BoxSize(width=int(width_in.value), height=int(height_in.value))
        _configure_padding_inputs(state["base"])  # rescale padding bounds/step to new base
        if not _apply_size():
            state["base"] = prev
            _set_size_inputs(prev)
            _configure_padding_inputs(prev)

    def on_padding_change(*_) -> None:
        if state["syncing"]:
            return
        if _apply_size():
            state["pad_w"] = int(padding_w_in.value)
            state["pad_h"] = int(padding_h_in.value)
        else:
            # Overlap: keep the last working padding rather than resetting.
            state["syncing"] = True
            try:
                padding_w_in.value = state["pad_w"]
                padding_h_in.value = state["pad_h"]
            finally:
                state["syncing"] = False

    def _clear(*_) -> None:
        state["valid"] = []
        _write_boxes([])
        _refresh_readout()

    def _undo(_viewer=None) -> None:
        if state["valid"]:
            state["valid"].pop()
            _write_boxes(state["valid"])
            _refresh_readout()

    def _refresh_proteins() -> None:
        # Compact, one short line per protein (a "folded" view); the full per-lane
        # values are shown while placing (readout) and in the table increment.
        lanes = state["lanes"]
        n = max((len(p["boxes"]) for p in state["proteins"]), default=0)
        hint = "  (one condition per lane)" if n and len(lanes) != n else ""
        lines = [f"lanes measured: {n}  |  conditions: {len(lanes)}{hint}"]
        if not state["proteins"]:
            lines.append("proteins: (none yet)")
        else:
            lines.append("proteins:")
            for p in state["proteins"]:
                mw = f" mw={p['mw']}" if p.get("mw") else ""
                lines.append(f"  {p['name']} [{p['role']}]{mw} - {len(p['boxes'])} lanes")
        proteins_lbl.value = "\n".join(lines)

    def on_conditions_change(*_) -> None:
        state["lanes"] = [s.strip() for s in conditions_in.value.split(",") if s.strip()]
        _refresh_proteins()

    def _add_protein(*_) -> None:
        if not state["valid"]:
            show_info("No boxes to capture.")
            return
        name = protein_in.value.strip() or f"protein {len(state['proteins']) + 1}"
        sz, invert, img = state["size"], bool(dark_on_light.value), image_layer.data
        # Store nets ordered left-to-right; conditions label these positions later.
        boxes = sorted(
            (
                (bx0, net_signal(img, Box(x=bx0, y=by0), sz, background, dark_on_light=invert))
                for (bx0, by0, _, _) in state["valid"]
            ),
            key=lambda bn: bn[0],
        )
        state["proteins"].append(
            {"name": name, "role": role_in.value, "mw": mw_in.value.strip(), "boxes": boxes}
        )
        state["valid"] = []  # clear the canvas for the next protein
        _write_boxes([])
        _refresh_readout()
        _refresh_proteins()
        show_info(f"Captured {name} ({role_in.value}, {len(boxes)} lanes); canvas cleared.")

    def _remove_last_protein(*_) -> None:
        if state["proteins"]:
            removed = state["proteins"].pop()
            _refresh_proteins()
            show_info(f"Removed protein {removed['name']}.")
        else:
            show_info("No captured proteins to remove.")

    viewer.mouse_drag_callbacks.append(on_mouse)
    shapes.events.data.connect(on_edit)
    width_in.changed.connect(on_size_change)
    height_in.changed.connect(on_size_change)
    padding_w_in.changed.connect(on_padding_change)
    padding_h_in.changed.connect(on_padding_change)
    dark_on_light.changed.connect(lambda *_: _refresh_readout())
    clear_btn.changed.connect(_clear)
    conditions_in.changed.connect(on_conditions_change)
    add_btn.changed.connect(_add_protein)
    remove_protein_btn.changed.connect(_remove_last_protein)
    viewer.bind_key("Control-Z", _undo, overwrite=True)

    _configure_padding_inputs(base0)  # set padding bounds/step from the initial base
    _refresh_readout()
    _refresh_proteins()
    show_info("Ctrl+click bands, name + Add protein (repeat); fill conditions at the end.")
    napari.run()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Proteia ROI quantification")
    parser.add_argument("image", nargs="?", default=None, help="image file (optional)")
    args = parser.parse_args()
    launch(args.image)


if __name__ == "__main__":
    main()
