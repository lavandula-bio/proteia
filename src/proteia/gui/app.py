# SPDX-License-Identifier: Apache-2.0
"""napari GUI: a project quantifies several proteins against shared lanes.

Encapsulated, repeatable per-protein flow:
- "New protein" begins an editing session (a fresh protein, given a colour).
- While editing, set name / MW / role and Ctrl+click bands (drawn in the
  protein's colour); right-click removes a box; "Confirm protein" ends the
  session.
- An "editing" selector re-enters an existing protein to add/remove its boxes.
- Outside an editing session (IDLE), Ctrl+click does nothing.

Captured proteins stay on the image in their colours; selecting a box and moving
it re-quantifies it live. Conditions label the result-table columns. Colour is
protein identity only.

    uv run python -m proteia.gui.app [IMAGE_PATH]

napari/Qt and magicgui imports are local so importing this module stays
headless-safe.
"""

from __future__ import annotations

import numpy as np

from proteia.core.analyze import (
    Batch,
    ProteinNets,
    Role,
    assess,
    compare,
    describe,
    fold_change_lane,
    normalize_lane,
    reduce_samples,
)
from proteia.core.boxes import normalize_corners, resize_all
from proteia.core.grow import grow_box
from proteia.core.model import Box, BoxSize, overlaps
from proteia.core.plotspec import ErrorType, ValueKind, build_plotspec
from proteia.core.project import (
    align_to_lanes,
    spine_axes,
    spine_from_labels,
)
from proteia.core.quantify import estimate_background, net_signal

Rect = tuple[int, int, int, int]
PALETTE = ["#ff4d4d", "#4dd2ff", "#ffe14d", "#7cfc00", "#ff66ff", "#ffa64d", "#66b3ff", "#b39ddb"]
NONE_CHOICE = "(idle — New protein to start)"


def _synthetic_image() -> np.ndarray:
    img = np.full((120, 200), 20.0)
    for cx in (40, 90, 140):
        img[50:70, cx : cx + 30] += 200.0
    return img


def _load_image(path: str | None) -> np.ndarray:
    """Read raw pixels (2D grayscale or 3D RGB/RGBA) from disk."""
    if path is None:
        return _synthetic_image()
    from skimage import io

    return io.imread(path)


def _to_gray(raw: np.ndarray) -> np.ndarray:
    """The analysis array: 2D grayscale intensity (densitometry works on this, and
    a consistent ndim lets the viewer swap images without a napari dims crash)."""
    if raw.ndim == 3:
        from skimage.color import rgb2gray

        return rgb2gray(raw[..., :3]) * 255.0  # rgb2gray is [0,1]; keep 8-bit-like range
    return raw


def _to_rgb(raw: np.ndarray) -> np.ndarray:
    """The display array: a uint8 RGB view of the original (colour survives for
    fluorescence). Grayscale sources are stacked to 3 channels."""
    rgb = raw[..., :3] if raw.ndim == 3 else np.stack([raw, raw, raw], axis=-1)
    if rgb.dtype != np.uint8:
        r = rgb.astype(float)
        lo, hi = float(r.min()), float(r.max())
        rgb = ((r - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else rgb.astype(np.uint8)
    return rgb


def _rect_to_corners(rect: Rect) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])


def _initial_size(image: np.ndarray) -> BoxSize:
    ih, iw = int(image.shape[0]), int(image.shape[1])
    return BoxSize(width=max(4, iw // 8), height=max(4, ih // 12))


def _make_image(raw: np.ndarray, name: str, path: str | None) -> dict:
    """One member of a batch's image set: a grayscale ``array`` (analysis) plus an
    RGB ``original`` (display), its background and dimensions. Net signal is always
    computed against the protein's own ``array``, so a target and a loading control
    on different membranes still join correctly by lane; ``original`` only feeds the
    optional 'show original' display toggle."""
    array = _to_gray(raw)
    return {
        "name": name,
        "path": path,
        "array": array,
        "original": _to_rgb(raw),
        "background": estimate_background(array),
        "iw": int(array.shape[1]),
        "ih": int(array.shape[0]),
        "dark": True,  # dark-on-light polarity, per image (chemi vs fluorescence differ)
    }


def _padded(base: BoxSize, px_w: int, px_h: int) -> BoxSize:
    return BoxSize(width=max(2, base.width + 2 * px_w), height=max(2, base.height + 2 * px_h))


def _pad_limits(base_dim: int) -> tuple[int, int, int]:
    return -round(base_dim * 0.25), round(base_dim * 0.5), max(1, round(base_dim * 0.05))


def _center_snap(rect: Rect, size: BoxSize, iw: int, ih: int) -> Rect:
    x0, y0, x1, y1 = rect
    nx0 = max(0, min((x0 + x1) // 2 - size.width // 2, iw - size.width))
    ny0 = max(0, min((y0 + y1) // 2 - size.height // 2, ih - size.height))
    return (nx0, ny0, nx0 + size.width, ny0 + size.height)


def _protein_size(p: dict) -> BoxSize:
    return _padded(p["base"], p["pad_w"], p["pad_h"])


def launch(image_path: str | None = None) -> None:
    import napari
    from magicgui.widgets import (
        CheckBox,
        ComboBox,
        Container,
        Label,
        LineEdit,
        PushButton,
        SpinBox,
        Table,
    )
    from napari.utils.notifications import show_info
    from qtpy.QtCore import QTimer

    viewer = napari.Viewer()
    # Two image layers for the same membrane: a colour "raw" view (display only) and
    # the grayscale "blot" used for analysis. The 'show original' toggle flips which
    # is visible; the ROI layer sits on top of both, so boxes stay put either way.
    raw_layer = viewer.add_image(
        np.zeros((1, 1, 3), dtype=np.uint8), name="raw", rgb=True, visible=False
    )
    image_layer = viewer.add_image(np.zeros((1, 1), dtype=float), name="blot")
    shapes = viewer.add_shapes(name="ROI", face_color="transparent", ndim=2)
    shapes.mode = "select"

    # A batch holds an image set; each protein binds to one image (by index), so a
    # target on image A and a loading control on image B join later by lane.
    # The app starts with NO image (like a document app); the user imports one or
    # more, so active is None until then. placed: every box {rect, pid}. proteins:
    # list by id, each carries "image" (its image index). editing: pid or None.
    state: dict = {
        "placed": [], "proteins": [], "lanes": [], "spine": [], "editing": None,
        "syncing": False, "plot_docks": [], "last_specs": [],
        "images": [], "active": None,
    }

    # --- widgets ---
    readout = Label(value="")
    status_lbl = Label(value="")
    results_table = Table(value={"data": [], "columns": [], "index": []})
    open_image_btn = PushButton(text="Import image…")
    image_combo = ComboBox(choices=["(no image)"], value="(no image)", label="image")
    remove_image_btn = PushButton(text="Remove this image")
    conditions_in = LineEdit(value="", label="conditions")
    # Data card: one row per lane. condition seeds from `conditions`; sample +
    # include are editable. Lanes sharing a (condition, sample) are technical
    # repeats (averaged, not counted as n); include=no drops a lane from stats.
    card_table = Table(
        value={"data": [], "columns": ["condition", "sample", "include"], "index": []}
    )
    edit_combo = ComboBox(choices=[NONE_CHOICE], value=NONE_CHOICE, label="editing")
    protein_in = LineEdit(value="", label="protein")
    mw_in = LineEdit(value="", label="MW (kDa)")
    role_in = ComboBox(choices=["target", "loading control"], value="target", label="role")
    width_in = SpinBox(value=64, min=2, max=10000, label="box W")  # max retuned per image
    height_in = SpinBox(value=20, min=2, max=10000, label="box H")
    padding_w_in = SpinBox(value=0, min=-1000, max=1000, step=1, label="pad W")
    padding_h_in = SpinBox(value=0, min=-1000, max=1000, step=1, label="pad H")
    dark_on_light = CheckBox(value=True, label="dark on light")
    manual_box = CheckBox(value=False, label="manual box (no auto-grow)")
    show_original = CheckBox(value=False, label="show original (raw)")
    new_btn = PushButton(text="New protein")
    confirm_btn = PushButton(text="Confirm protein")
    remove_protein_btn = PushButton(text="Remove last protein")
    clear_btn = PushButton(text="Clear this protein's boxes")
    # The reference condition is the fold-change denominator (=1); by role, not
    # name — it need not be called "control". It sits leftmost on the chart.
    control_in = LineEdit(value="", label="reference")
    error_in = ComboBox(
        choices=[e.value for e in ErrorType], value=ErrorType.SD.value, label="error bar"
    )
    plot_btn = PushButton(text="Plot")
    save_chart_btn = PushButton(text="Save chart…")
    export_csv_btn = PushButton(text="Export table (CSV)…")
    panel = Container(
        widgets=[
            open_image_btn, image_combo, remove_image_btn,
            conditions_in, card_table, edit_combo, protein_in, mw_in, role_in,
            width_in, height_in, padding_w_in, padding_h_in,
            dark_on_light, manual_box, show_original,
            new_btn, confirm_btn, remove_protein_btn, clear_btn,
            control_in, error_in, plot_btn, save_chart_btn, export_csv_btn,
            readout, status_lbl, results_table,
        ],
        labels=True,
    )
    viewer.window.add_dock_widget(panel, area="right", name="Quantify")

    # --- helpers ---
    def _ep() -> dict | None:
        return None if state["editing"] is None else state["proteins"][state["editing"]]

    def _pid_rects(pid: int) -> list[Rect]:
        return [pl["rect"] for pl in state["placed"] if pl["pid"] == pid]

    def _active() -> dict | None:
        i = state["active"]
        return state["images"][i] if i is not None else None

    def _img_of(pid: int) -> dict:
        return state["images"][state["proteins"][pid]["image"]]

    def _adims() -> tuple[int, int]:
        a = _active()
        return (a["iw"], a["ih"]) if a else (0, 0)

    def _active_placed() -> list[dict]:
        """Boxes whose protein lives on the currently displayed image. Only these
        are drawn / editable — boxes on other images are in different pixel space."""
        act = state["active"]
        return [pl for pl in state["placed"] if state["proteins"][pl["pid"]]["image"] == act]

    def _redraw() -> None:
        state["syncing"] = True
        try:
            shapes.mode = "pan_zoom"  # drop selection/highlight before rebuilding
            shapes.selected_data = set()
            shapes.data = []
            aps = _active_placed()
            if aps:
                shapes.add(
                    [_rect_to_corners(pl["rect"]) for pl in aps],
                    shape_type="rectangle",
                )
                shapes.edge_color = [PALETTE[pl["pid"] % len(PALETTE)] for pl in aps]
                shapes.face_color = "transparent"
            shapes.selected_data = set()
            shapes.mode = "select"
        finally:
            state["syncing"] = False

    def _sync_image_combo() -> None:
        state["syncing"] = True
        try:
            if state["images"]:
                choices = [f"{i}: {img['name']}" for i, img in enumerate(state["images"])]
                image_combo.choices = choices
                image_combo.value = choices[state["active"]]
            else:
                image_combo.choices = ["(no image)"]
                image_combo.value = "(no image)"
        finally:
            state["syncing"] = False

    def _set_active_image(idx: int | None) -> None:
        if not state["images"] or idx is None:
            state["active"] = None
            state["syncing"] = True
            try:
                image_layer.data = np.zeros((1, 1), dtype=float)  # blank: no image
                raw_layer.data = np.zeros((1, 1, 3), dtype=np.uint8)
            finally:
                state["syncing"] = False
            _sync_image_combo()
            _redraw()
            return
        a = state["images"][idx]
        state["syncing"] = True
        try:
            image_layer.data = a["array"]
            raw_layer.data = a["original"]
            # Swapping .data keeps the previous contrast limits (the blank 1x1
            # placeholder's near-[0,0]), which renders a real image almost all
            # white; recompute them for the new image.
            try:
                image_layer.reset_contrast_limits()
            except (AttributeError, ValueError, RuntimeError):
                pass
            width_in.max, height_in.max = a["iw"], a["ih"]
            dark_on_light.value = a["dark"]  # polarity is per image
        finally:
            state["syncing"] = False
        state["active"] = idx  # commit only after the layer swapped without error
        viewer.reset_view()  # refit: images may differ in size
        _sync_image_combo()
        _redraw()

    def _protein_boxes(pid: int) -> list[tuple[int, float]]:
        size = _protein_size(state["proteins"][pid])
        img_d = _img_of(pid)
        invert = img_d["dark"]  # polarity of this protein's own image
        boxes = []
        for rect in _pid_rects(pid):
            x0, y0 = rect[0], rect[1]
            net = net_signal(
                img_d["array"], Box(x=x0, y=y0), size, img_d["background"], dark_on_light=invert
            )
            boxes.append((x0, net))
        return sorted(boxes, key=lambda b: b[0])

    def _aligned_to_spine(n: int) -> list[list[float | None]]:
        """Align every protein's boxes to the n-lane spine, **per image**.

        Box x-position is only comparable within one image (each membrane has its
        own coordinate frame), so proteins are grouped by their image and gridded
        among themselves — never pooled across images. A protein on a reprobed
        membrane (same lanes) therefore lines up with its target; a protein on an
        image covering different lanes still needs explicit lane numbers (future).
        """
        proteins = state["proteins"]
        rows: list[list[float | None]] = [[None] * n for _ in proteins]
        by_image: dict[int, list[int]] = {}
        for pid in range(len(proteins)):
            by_image.setdefault(proteins[pid]["image"], []).append(pid)
        for pids in by_image.values():
            aligned = align_to_lanes([_protein_boxes(pid) for pid in pids], n)
            for k, pid in enumerate(pids):
                rows[pid] = aligned[k]
        return rows

    def _load_controls(p: dict) -> None:
        state["syncing"] = True
        try:
            protein_in.value, mw_in.value, role_in.value = p["name"], p["mw"], p["role"]
            width_in.value, height_in.value = p["base"].width, p["base"].height
            for spin, dim, val in (
                (padding_w_in, p["base"].width, p["pad_w"]),
                (padding_h_in, p["base"].height, p["pad_h"]),
            ):
                lo, hi, step = _pad_limits(dim)
                spin.min, spin.max, spin.step = lo, hi, step
                spin.value = min(max(int(val), lo), hi)
        finally:
            state["syncing"] = False

    def _sync_combo() -> None:
        state["syncing"] = True
        try:
            choices = [NONE_CHOICE] + [
                f"{i}: {p['name'] or '(unnamed)'}" for i, p in enumerate(state["proteins"])
            ]
            edit_combo.choices = choices
            ed = state["editing"]
            edit_combo.value = NONE_CHOICE if ed is None else choices[ed + 1]
        finally:
            state["syncing"] = False

    def _refresh_readout() -> None:
        if not state["images"]:
            readout.value = "No image. Press Import image… to start."
            return
        p = _ep()
        img_d = _img_of(state["editing"]) if p is not None else _active()
        invert = img_d["dark"]
        direction = "dark-on-light" if invert else "light-on-dark"
        if p is None:
            a = _active()
            readout.value = (
                f"image: {a['name']}   background: {a['background']:.0f}  ({direction})\n"
                "IDLE — press New protein (or pick one in 'editing') to place boxes."
            )
            return
        sz = _protein_size(p)
        lines = [
            f"editing: {p['name'] or '(unnamed)'} [{p['role']}]  colour {p['color']}",
            f"image: {img_d['name']}",
            f"box: {sz.width} x {sz.height} (base {p['base'].width}x{p['base'].height} + "
            f"{p['pad_w']}/{p['pad_h']} px/side)",
            f"background: {img_d['background']:.0f}  ({direction})",
            "Ctrl+click bands; right-click a box to remove; Confirm when done.",
        ]
        rects = sorted(_pid_rects(state["editing"]), key=lambda r: r[0])
        for i, (bx0, by0, _, _) in enumerate(rects):
            net = net_signal(
                img_d["array"], Box(x=bx0, y=by0), sz, img_d["background"], dark_on_light=invert
            )
            lines.append(f"  {i}: net = {net:.0f}  at x={bx0}")
        readout.value = "\n".join(lines)

    def _rebuild_spine() -> None:
        """Regenerate the lane spine from the declared conditions, then redraw the
        data card. Resets per-lane sample/include edits — conditions is the bulk
        declaration; fine-tune samples/include in the card afterwards.
        """
        state["spine"] = spine_from_labels(state["lanes"])
        _refresh_card()

    def _refresh_card() -> None:
        spine = state["spine"]
        state["syncing"] = True
        try:
            data = [[lane.label, lane.sample or "", "yes" if lane.included else "no"]
                    for lane in spine]
            index = [str(lane.index) for lane in spine]
            card_table.value = {
                "data": data, "columns": ["condition", "sample", "include"], "index": index
            }
        finally:
            state["syncing"] = False

    def _raw_label(p: dict) -> str:
        mw = f" mw={p['mw']}" if p.get("mw") else ""
        mark = "" if p["confirmed"] else " *"
        return f"{p['name'] or '(unnamed)'} [{p['role']}]{mw}{mark}"

    def _refresh_table() -> None:
        proteins, spine = state["proteins"], state["spine"]
        per = [_protein_boxes(pid) for pid in range(len(proteins))]
        max_boxes = max((len(b) for b in per), default=0)
        n = len(spine)
        a = _active()
        hint = "  (one condition per lane)" if max_boxes and n != max_boxes else ""
        where = (
            f"image: {a['name']} ({state['active'] + 1}/{len(state['images'])})  |  "
            if a else "no image  |  "
        )
        status_lbl.value = f"{where}lanes: {max_boxes}  |  conditions: {n}{hint}"
        if not spine:
            columns = [f"#{j + 1}" for j in range(max_boxes)]
            data = [[round(b[j][1]) if j < len(b) else "" for j in range(max_boxes)] for b in per]
            results_table.value = {
                "data": data, "columns": columns, "index": [_raw_label(p) for p in proteins]
            }
            return
        # Master table: raw detection rows (net per lane), then a ratio row per
        # non-loading protein (target / loading control) — the value charts plot.
        conditions, _samples, _included = spine_axes(spine)
        columns = list(conditions)
        aligned = _aligned_to_spine(n)  # per-image; no cross-image pooling
        data = [[round(v) if v is not None else "" for v in row] for row in aligned]
        index = [_raw_label(p) for p in proteins]
        lc = next((i for i, p in enumerate(proteins) if p["role"] == "loading control"), None)
        if lc is not None:
            lo = aligned[lc]
            lo_name = proteins[lc]["name"] or "loading"
            for i, p in enumerate(proteins):
                if i == lc:
                    continue
                ratio = [
                    t / lval if (t is not None and lval not in (None, 0)) else None
                    for t, lval in zip(aligned[i], lo, strict=True)
                ]
                data.append([round(v, 3) if v is not None else "" for v in ratio])
                index.append(f"{p['name'] or '(unnamed)'} / {lo_name} (ratio)")
        results_table.value = {"data": data, "columns": columns, "index": index}

    def _refresh_all() -> None:
        _redraw()
        _refresh_readout()
        _refresh_table()

    # --- editing session ---
    def _set_editing(pid: int | None) -> None:
        state["editing"] = pid
        if pid is not None:
            p = state["proteins"][pid]
            if state["active"] != p["image"]:
                _set_active_image(p["image"])  # show the image this protein lives on
            _load_controls(p)
        _sync_combo()
        _refresh_readout()

    def on_open_image(*_) -> None:
        from qtpy.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            panel.native, "Open image", "", "Images (*.png *.tif *.tiff *.jpg *.jpeg)"
        )
        if not path:
            return
        name = path.split("/")[-1]
        state["images"].append(_make_image(_load_image(path), name, path))  # normalized to 2D gray
        _set_active_image(len(state["images"]) - 1)
        _refresh_all()
        show_info(f"Opened {name}. New proteins you add now bind to this image.")

    def on_image_combo_change(*_) -> None:
        if state["syncing"] or not state["images"]:
            return
        idx = int(image_combo.value.split(":", 1)[0])
        # Switching away from the image the edited protein lives on stops editing,
        # so clicks don't land on a protein bound to a different image.
        if state["editing"] is not None and state["proteins"][state["editing"]]["image"] != idx:
            state["editing"] = None
        _set_active_image(idx)
        _sync_combo()
        _refresh_readout()

    def on_remove_image(*_) -> None:
        if state["active"] is None:
            show_info("No image to remove.")
            return
        idx = state["active"]
        name = state["images"][idx]["name"]
        removed = [i for i, p in enumerate(state["proteins"]) if p["image"] == idx]
        # Reindex surviving proteins (image refs shift; pids renumber) and drop the
        # removed proteins' boxes.
        kept: list[dict] = []
        remap: dict[int, int] = {}
        for i, p in enumerate(state["proteins"]):
            if p["image"] == idx:
                continue
            if p["image"] > idx:
                p["image"] -= 1
            remap[i] = len(kept)
            kept.append(p)
        state["placed"] = [
            {**pl, "pid": remap[pl["pid"]]}
            for pl in state["placed"]
            if pl["pid"] not in removed
        ]
        state["proteins"] = kept
        del state["images"][idx]
        state["editing"] = None
        _set_active_image(min(idx, len(state["images"]) - 1))
        _sync_combo()
        _refresh_all()
        extra = f" and {len(removed)} protein(s) on it" if removed else ""
        show_info(f"Removed image {name}{extra}.")

    def on_dark_change(*_) -> None:
        if state["syncing"] or _active() is None:
            return
        _active()["dark"] = bool(dark_on_light.value)  # polarity is per image
        _refresh_all()

    def on_show_original(*_) -> None:
        # Display-only: flip which membrane layer is visible. ROIs and all analysis
        # stay on the grayscale array regardless.
        show = bool(show_original.value)
        raw_layer.visible = show
        image_layer.visible = not show

    def _new_protein(*_) -> None:
        if state["active"] is None:
            show_info("Import an image first (Import image…).")
            return
        pid = len(state["proteins"])
        state["proteins"].append({
            "name": "", "role": role_in.value, "mw": "", "color": PALETTE[pid % len(PALETTE)],
            "base": _initial_size(_active()["array"]), "pad_w": 0, "pad_h": 0,
            "confirmed": False, "image": state["active"],  # bind to the displayed image
        })
        state["syncing"] = True
        try:
            protein_in.value, mw_in.value = "", ""
        finally:
            state["syncing"] = False
        _set_editing(pid)
        _refresh_table()
        show_info("New protein: set name / MW / role, Ctrl+click bands, then Confirm.")

    def _confirm_protein(*_) -> None:
        p = _ep()
        if p is None:
            show_info("Not editing a protein.")
            return
        if not _pid_rects(state["editing"]):
            show_info("No boxes for this protein yet.")
            return
        p["confirmed"] = True
        _set_editing(None)
        _refresh_table()
        show_info(f"Confirmed {p['name'] or '(unnamed)'}.")

    # --- interactions ---
    def _click_xy(event) -> tuple[int, int] | None:
        row, col = image_layer.world_to_data(event.position)[:2]
        x, y = int(round(col)), int(round(row))
        iw, ih = _adims()
        return (x, y) if (0 <= x < iw and 0 <= y < ih) else None

    def _seed_grow(x: int, y: int) -> None:
        if state["editing"] is None:
            show_info("Press New protein (or pick one in 'editing') before placing boxes.")
            return
        pid = state["editing"]
        p = state["proteins"][pid]
        a = _active()
        iw, ih = a["iw"], a["ih"]
        rects = _pid_rects(pid)
        if manual_box.value:
            # Messy / streaky blots where region grow runs into neighbours: drop a
            # fixed box (current box W/H) centred on the click, no auto-grow.
            eff = _protein_size(p)
            w, h = eff.width, eff.height
            nx0 = min(max(0, x - w // 2), iw - w)
            ny0 = min(max(0, y - h // 2), ih - h)
            rect = (nx0, ny0, nx0 + w, ny0 + h)
            if any(overlaps(rect, r) for r in rects):
                show_info("Box would overlap one of this protein's boxes; skipped.")
                return
            others = [pl for pl in state["placed"] if pl["pid"] != pid]
            state["placed"] = others + [{"rect": r, "pid": pid} for r in (*rects, rect)]
            _load_controls(p)
            _refresh_all()
            return
        invert = a["dark"]  # polarity of the displayed image
        box = grow_box(a["array"], (x, y), a["background"], dark_on_light=invert)
        if box is None:
            show_info("No band detected at that point.")
            return
        gx0, gy0, gx1, gy1 = box
        gw, gh = max(2, gx1 - gx0), max(2, gy1 - gy0)
        if not rects:
            p["base"] = BoxSize(width=gw, height=gh)
        else:
            p["base"] = BoxSize(width=max(p["base"].width, gw), height=max(p["base"].height, gh))
        eff = _protein_size(p)
        resized = resize_all(rects, eff, width=iw, height=ih)
        if resized is None:
            show_info("Cannot grow this protein's box size here without overlap; skipped.")
            return
        w, h = eff.width, eff.height
        cx, cy = (gx0 + gx1) // 2, (gy0 + gy1) // 2
        nx0 = min(max(0, cx - w // 2), iw - w)
        ny0 = min(max(0, cy - h // 2), ih - h)
        rect = (nx0, ny0, nx0 + w, ny0 + h)
        if any(overlaps(rect, r) for r in resized):
            show_info("Box would overlap one of this protein's boxes; skipped.")
            return
        others = [pl for pl in state["placed"] if pl["pid"] != pid]
        kept = [{"rect": r, "pid": pid} for r in (*resized, rect)]
        state["placed"] = others + kept
        _load_controls(p)
        _refresh_all()

    def _remove_at(x: int, y: int) -> None:
        # Only act while editing a protein, and only on that protein's boxes
        # (IDLE right-click does nothing; you cannot delete another protein's box).
        if state["editing"] is None:
            return
        pid = state["editing"]
        for i, pl in enumerate(state["placed"]):
            if pl["pid"] != pid:
                continue
            rx0, ry0, rx1, ry1 = pl["rect"]
            if rx0 <= x < rx1 and ry0 <= y < ry1:
                del state["placed"][i]
                _refresh_all()
                return

    def on_mouse(_viewer, event) -> None:
        add = event.button == 1 and "Control" in event.modifiers
        remove = event.button == 2
        if not (add or remove):
            return
        if state["editing"] is None:
            QTimer.singleShot(
                0, lambda: show_info("Press New protein (or pick one in 'editing') first.")
            )
            return
        xy = _click_xy(event)
        if not xy:
            return
        # Defer the mutation so we don't rebuild the layer *during* napari's own
        # handling of this click (which leaves a stale selection index).
        QTimer.singleShot(0, lambda: (_seed_grow(*xy) if add else _remove_at(*xy)))

    def on_edit(*_) -> None:
        if state["syncing"]:
            return
        # The shapes layer only carries the active image's boxes; map edits back to
        # those entries by their index in state["placed"].
        gidx = [i for i, pl in enumerate(state["placed"])
                if state["proteins"][pl["pid"]]["image"] == state["active"]]
        new = [normalize_corners(np.asarray(c)) for c in shapes.data]
        if len(new) != len(gidx):
            _redraw()
            return
        iw, ih = _adims()
        cand = {
            gi: _center_snap(new[k], _protein_size(state["proteins"][state["placed"][gi]["pid"]]),
                             iw, ih)
            for k, gi in enumerate(gidx)
        }
        by_pid: dict[int, list[int]] = {}
        for gi in gidx:
            by_pid.setdefault(state["placed"][gi]["pid"], []).append(gi)
        for idxs in by_pid.values():
            rects = [cand[gi] for gi in idxs]
            pairs = ((a, b) for a in range(len(rects)) for b in range(a + 1, len(rects)))
            if any(overlaps(rects[a], rects[b]) for a, b in pairs):
                for gi in idxs:
                    cand[gi] = state["placed"][gi]["rect"]
        for gi in gidx:
            state["placed"][gi]["rect"] = cand[gi]
        _refresh_all()

    def _remove_last_protein(*_) -> None:
        if not state["proteins"]:
            show_info("No proteins.")
            return
        pid = len(state["proteins"]) - 1
        state["placed"] = [pl for pl in state["placed"] if pl["pid"] != pid]
        removed = state["proteins"].pop()
        if state["editing"] == pid:
            state["editing"] = None
        _set_editing(state["editing"])
        _refresh_all()
        show_info(f"Removed protein {removed['name'] or '(unnamed)'}.")

    def _clear_boxes(*_) -> None:
        if state["editing"] is None:
            show_info("Not editing a protein.")
            return
        pid = state["editing"]
        state["placed"] = [pl for pl in state["placed"] if pl["pid"] != pid]
        _refresh_all()

    def _undo(_viewer=None) -> None:
        if state["editing"] is None:
            return
        pid = state["editing"]
        idxs = [i for i, pl in enumerate(state["placed"]) if pl["pid"] == pid]
        if idxs:
            del state["placed"][idxs[-1]]
            _refresh_all()

    def on_size_change(*_) -> None:
        if state["syncing"] or state["editing"] is None:
            return
        pid = state["editing"]
        p = state["proteins"][pid]
        prev = (p["base"], p["pad_w"], p["pad_h"])
        p["base"] = BoxSize(width=int(width_in.value), height=int(height_in.value))
        p["pad_w"], p["pad_h"] = int(padding_w_in.value), int(padding_h_in.value)
        iw, ih = _adims()
        resized = resize_all(_pid_rects(pid), _protein_size(p), width=iw, height=ih)
        if resized is None:
            show_info("Size rejected: it would make this protein's boxes overlap.")
            p["base"], p["pad_w"], p["pad_h"] = prev
            _load_controls(p)
            return
        others = [pl for pl in state["placed"] if pl["pid"] != pid]
        state["placed"] = others + [{"rect": r, "pid": pid} for r in resized]
        _load_controls(p)
        _refresh_all()

    def on_meta_change(*_) -> None:
        if state["syncing"] or state["editing"] is None:
            return
        p = state["proteins"][state["editing"]]
        p["name"], p["mw"], p["role"] = protein_in.value.strip(), mw_in.value.strip(), role_in.value
        _sync_combo()
        _refresh_table()

    def on_combo_change(*_) -> None:
        if state["syncing"]:
            return
        val = edit_combo.value
        _set_editing(None if val == NONE_CHOICE else int(val.split(":", 1)[0]))

    def on_conditions_change(*_) -> None:
        state["lanes"] = [s.strip() for s in conditions_in.value.split(",") if s.strip()]
        _rebuild_spine()
        _refresh_table()

    def _sync_spine_from_card() -> None:
        """Pull the data card's current cells into the lane spine.

        condition / sample are free text; include is yes/no (anything not
        affirmative reads as excluded). Marking two lanes the same (condition,
        sample) makes them technical repeats; set include=no to drop a lane. Read
        on demand (also at plot time) so edits land even if the table widget does
        not emit a change event per cell.
        """
        spine = state["spine"]
        rows = (card_table.value or {}).get("data", [])
        if len(rows) != len(spine):
            return  # stale / mid-rebuild; ignore
        for lane, row in zip(spine, rows, strict=True):
            label, sample, include = (str(c).strip() for c in row)
            lane.label = label
            lane.sample = sample or None
            lane.included = include.lower() in ("yes", "y", "true", "1")

    def on_card_change(*_) -> None:
        if state["syncing"]:
            return
        _sync_spine_from_card()
        _refresh_table()

    def _show_figures(named_specs: list[tuple[str, object]]) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from proteia.viz import render_figure

        # Drop previous charts. A user may have closed one already, so guard each.
        for dock in state["plot_docks"]:
            try:
                viewer.window.remove_dock_widget(dock)
            except (RuntimeError, KeyError, ValueError, AttributeError):
                pass
        state["plot_docks"] = []
        for i, (name, spec) in enumerate(named_specs):
            canvas = FigureCanvasQTAgg(render_figure(spec))
            dock = viewer.window.add_dock_widget(canvas, area="bottom", name=f"Chart: {name}")
            # Float each, cascaded, so multiple targets are visibly separate windows.
            try:
                dock.setFloating(True)
                dock.setGeometry(80 + i * 48, 80 + i * 48, 720, 560)
            except (AttributeError, RuntimeError):
                pass
            state["plot_docks"].append(dock)

    def on_plot(*_) -> None:
        proteins, spine = state["proteins"], state["spine"]
        if not spine:
            show_info("Set conditions first (comma-separated; repeats = replicates).")
            return
        _sync_spine_from_card()  # honour the latest card edits even without a change event
        n = len(spine)
        conditions, samples, included = spine_axes(spine)
        per = [_protein_boxes(pid) for pid in range(len(proteins))]
        # Drawing more bands than declared conditions is ambiguous (which band is
        # the orphan?), so we refuse rather than silently collide two into one slot.
        overfilled = [
            f"{p['name'] or '(unnamed)'} ({len(b)} boxes)"
            for p, b in zip(proteins, per, strict=True)
            if len(b) > n
        ]
        if overfilled:
            show_info(
                f"More boxes than the {n} declared conditions: {', '.join(overfilled)}. "
                "Add the missing condition(s) so every band has a lane."
            )
            return
        # A band's lane is proposed from its x position, anchored across *all*
        # proteins: a fully-drawn loading control fixes the grid so a missing
        # target band lands in its real empty slot. Per-box explicit lane numbers
        # (next UI step) will remove the guess entirely.
        aligned = _aligned_to_spine(n)  # per-image; no cross-image pooling
        # Fewer boxes than lanes is fine (a gap) but warn — anchoring can still
        # misplace if the protein anchoring the grid is itself short.
        mismatched = [
            f"{p['name'] or '(unnamed)'} ({len(b)})"
            for p, b in zip(proteins, per, strict=True)
            if b and len(b) != n
        ]
        pnets = [
            ProteinNets(
                p["name"] or "(unnamed)",
                Role.LOADING_CONTROL if p["role"] == "loading control" else Role.TARGET,
                nets,
            )
            for p, nets in zip(proteins, aligned, strict=True)
        ]
        control = control_in.value.strip() or None
        try:
            batch = Batch(conditions, pnets, control_condition=control)
        except ValueError as exc:
            show_info(f"Cannot plot: {exc}")
            return
        comp = assess(batch)
        if not comp.can_pool:
            show_info("Export only — " + "; ".join(comp.warnings))
            return

        loading = batch.loading_control()
        # Provenance: condition -> its included source lanes (for later cropping/audit).
        provenance: dict[str, list[int]] = {}
        for i, cond in enumerate(conditions):
            if included[i]:
                provenance.setdefault(cond, []).append(i)
        error_type = ErrorType(error_in.value)

        # One figure per target, all normalized to the batch's single loading control.
        named_specs: list[tuple[str, object]] = []
        notes = list(comp.warnings)
        for target in batch.targets():
            norm = normalize_lane(target.nets, loading.nets)
            if control:
                try:
                    values = fold_change_lane(norm, conditions, control)
                except ValueError as exc:
                    notes.append(f"{target.name}: {exc}")
                    continue
                kind = ValueKind.FOLD_CHANGE
                title = f"{target.name} fold-change vs {control}"
            else:
                values, kind = norm, ValueKind.LOADING_NORMALIZED
                title = f"{target.name} / {loading.name}"
            # Sample-aware: lanes sharing (condition, sample) are averaged before
            # stats (technical repeats don't inflate n); include=no lanes dropped.
            reduction = reduce_samples(values, conditions, samples, included=included)
            groups = reduction.groups
            # Reference condition sits leftmost (it is the fold-change denominator).
            spec = build_plotspec(
                groups, describe(groups), compare(groups),
                value_kind=kind, error_type=error_type, title=title,
                lane_indices=provenance, first_label=control,
            )
            named_specs.append((target.name, spec))
            for w in reduction.warnings:
                if w not in notes:
                    notes.append(w)
        if not named_specs:
            show_info("Nothing to plot: " + ("; ".join(notes) or "no usable target"))
            return
        state["last_specs"] = named_specs  # remember for "Save chart"
        _show_figures(named_specs)
        if mismatched:
            notes.append(
                f"box count != conditions ({n}) for {', '.join(mismatched)}; "
                "lane numbers proposed by position and may be misaligned"
            )
        if notes:
            show_info("; ".join(notes))

    def _save_path(caption: str, file_filter: str) -> str | None:
        from qtpy.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(panel.native, caption, "", file_filter)
        return path or None

    def on_save_chart(*_) -> None:
        specs = state["last_specs"]
        if not specs:
            show_info("Plot a chart first, then save it.")
            return
        from proteia.viz import save_figure

        if len(specs) == 1:
            path = _save_path("Save chart", "Image (*.png *.pdf *.svg)")
            if not path:
                return
            save_figure(specs[0][1], path)
            show_info(f"Saved chart to {path}")
            return
        # Several targets -> save each to its own file in a chosen folder.
        from qtpy.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(panel.native, "Save charts to folder")
        if not folder:
            return
        for name, spec in specs:
            safe = name.replace("/", "_").strip() or "chart"
            save_figure(spec, f"{folder}/{safe}.png")
        show_info(f"Saved {len(specs)} charts to {folder}")

    def on_export_csv(*_) -> None:
        """Export the raw per-lane table: lane identity + each protein's net.

        This is the first ("raw") staged output — one row per lane, columns for
        condition / sample / include and every protein's net signal aligned to the
        spine. Empty cells are gaps (no box for that protein on that lane).
        """
        proteins, spine = state["proteins"], state["spine"]
        if not spine:
            show_info("Set conditions first.")
            return
        _sync_spine_from_card()
        n = len(spine)
        per = [_protein_boxes(pid) for pid in range(len(proteins))]
        if any(len(b) > n for b in per):
            show_info("More boxes than conditions; fix the data card before exporting.")
            return
        aligned = _aligned_to_spine(n)  # per-image; no cross-image pooling
        conditions, samples, included = spine_axes(spine)
        path = _save_path("Export table", "CSV (*.csv)")
        if not path:
            return
        import csv

        names = [p["name"] or f"protein{i}" for i, p in enumerate(proteins)]
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["lane", "condition", "sample", "include", *names])
            for i in range(n):
                row = [i, conditions[i], samples[i] or "", "yes" if included[i] else "no"]
                row += ["" if aligned[pid][i] is None else round(aligned[pid][i], 3)
                        for pid in range(len(proteins))]
                writer.writerow(row)
        show_info(f"Exported {n} lanes x {len(proteins)} proteins to {path}")

    shapes.events.data.connect(on_edit)
    viewer.mouse_drag_callbacks.append(on_mouse)
    width_in.changed.connect(on_size_change)
    height_in.changed.connect(on_size_change)
    padding_w_in.changed.connect(on_size_change)
    padding_h_in.changed.connect(on_size_change)
    protein_in.changed.connect(on_meta_change)
    mw_in.changed.connect(on_meta_change)
    role_in.changed.connect(on_meta_change)
    dark_on_light.changed.connect(on_dark_change)
    show_original.changed.connect(on_show_original)
    open_image_btn.changed.connect(on_open_image)
    image_combo.changed.connect(on_image_combo_change)
    remove_image_btn.changed.connect(on_remove_image)
    conditions_in.changed.connect(on_conditions_change)
    card_table.changed.connect(on_card_change)
    edit_combo.changed.connect(on_combo_change)
    new_btn.changed.connect(_new_protein)
    confirm_btn.changed.connect(_confirm_protein)
    remove_protein_btn.changed.connect(_remove_last_protein)
    clear_btn.changed.connect(_clear_boxes)
    plot_btn.changed.connect(on_plot)
    save_chart_btn.changed.connect(on_save_chart)
    export_csv_btn.changed.connect(on_export_csv)
    viewer.bind_key("Control-Z", _undo, overwrite=True)

    # A path on the command line is imported as the first image; otherwise the app
    # opens empty and the user imports.
    if image_path:
        state["images"].append(
            _make_image(_load_image(image_path), image_path.split("/")[-1], image_path)
        )
        _set_active_image(0)
    _refresh_all()
    show_info(
        "Import an image to start (Import image…)."
        if not state["images"]
        else "Press New protein to start; set name/MW/role, Ctrl+click bands, then Confirm."
    )
    napari.run()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Proteia ROI quantification")
    parser.add_argument("image", nargs="?", default=None, help="image file (optional)")
    args = parser.parse_args()
    launch(args.image)


if __name__ == "__main__":
    main()
