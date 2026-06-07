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
    join_to_spine,
    propose_positions,
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
    if path is None:
        return _synthetic_image()
    from skimage import io

    return io.imread(path)


def _rect_to_corners(rect: Rect) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])


def _initial_size(image: np.ndarray) -> BoxSize:
    ih, iw = int(image.shape[0]), int(image.shape[1])
    return BoxSize(width=max(4, iw // 8), height=max(4, ih // 12))


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

    image = _load_image(image_path)
    ih, iw = int(image.shape[0]), int(image.shape[1])
    background = estimate_background(image)

    viewer = napari.Viewer()
    image_layer = viewer.add_image(image, name="blot")
    shapes = viewer.add_shapes(name="ROI", face_color="transparent", ndim=2)
    shapes.mode = "select"

    # placed: every box, tagged with its protein id. proteins: list by id, each
    # {name, role, mw, color, base, pad_w, pad_h, confirmed}. editing: pid or None.
    state: dict = {
        "placed": [], "proteins": [], "lanes": [], "spine": [], "editing": None,
        "syncing": False, "plot_dock": None,
    }

    # --- widgets ---
    readout = Label(value="")
    status_lbl = Label(value="")
    results_table = Table(value={"data": [], "columns": [], "index": []})
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
    width_in = SpinBox(value=64, min=2, max=iw, label="box W")
    height_in = SpinBox(value=20, min=2, max=ih, label="box H")
    padding_w_in = SpinBox(value=0, min=-1000, max=1000, step=1, label="pad W")
    padding_h_in = SpinBox(value=0, min=-1000, max=1000, step=1, label="pad H")
    dark_on_light = CheckBox(value=True, label="dark on light")
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
    panel = Container(
        widgets=[
            conditions_in, card_table, edit_combo, protein_in, mw_in, role_in,
            width_in, height_in, padding_w_in, padding_h_in, dark_on_light,
            new_btn, confirm_btn, remove_protein_btn, clear_btn,
            control_in, error_in, plot_btn,
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

    def _redraw() -> None:
        state["syncing"] = True
        try:
            shapes.mode = "pan_zoom"  # drop selection/highlight before rebuilding
            shapes.selected_data = set()
            shapes.data = []
            if state["placed"]:
                shapes.add(
                    [_rect_to_corners(pl["rect"]) for pl in state["placed"]],
                    shape_type="rectangle",
                )
                shapes.edge_color = [PALETTE[pl["pid"] % len(PALETTE)] for pl in state["placed"]]
                shapes.face_color = "transparent"
            shapes.selected_data = set()
            shapes.mode = "select"
        finally:
            state["syncing"] = False

    def _protein_boxes(pid: int) -> list[tuple[int, float]]:
        invert = bool(dark_on_light.value)
        size = _protein_size(state["proteins"][pid])
        img = image_layer.data
        boxes = []
        for rect in _pid_rects(pid):
            x0, y0 = rect[0], rect[1]
            net = net_signal(img, Box(x=x0, y=y0), size, background, dark_on_light=invert)
            boxes.append((x0, net))
        return sorted(boxes, key=lambda b: b[0])

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
        p = _ep()
        invert = bool(dark_on_light.value)
        direction = "dark-on-light" if invert else "light-on-dark"
        if p is None:
            readout.value = (
                f"background: {background:.0f}  ({direction})\n"
                "IDLE — press New protein (or pick one in 'editing') to place boxes."
            )
            return
        sz = _protein_size(p)
        lines = [
            f"editing: {p['name'] or '(unnamed)'} [{p['role']}]  colour {p['color']}",
            f"box: {sz.width} x {sz.height} (base {p['base'].width}x{p['base'].height} + "
            f"{p['pad_w']}/{p['pad_h']} px/side)",
            f"background: {background:.0f}  ({direction})",
            "Ctrl+click bands; right-click a box to remove; Confirm when done.",
        ]
        rects = sorted(_pid_rects(state["editing"]), key=lambda r: r[0])
        for i, (bx0, by0, _, _) in enumerate(rects):
            net = net_signal(
                image_layer.data, Box(x=bx0, y=by0), sz, background, dark_on_light=invert
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

    def _refresh_table() -> None:
        proteins, spine = state["proteins"], state["spine"]
        per = [propose_positions(_protein_boxes(pid)) for pid in range(len(proteins))]
        max_boxes = max((len(b) for b in per), default=0)
        n = len(spine)
        hint = "  (one condition per lane)" if max_boxes and n != max_boxes else ""
        status_lbl.value = f"lanes: {max_boxes}  |  conditions: {n}{hint}"
        if spine:
            conditions, _samples, _included = spine_axes(spine)
            columns = list(conditions)
            aligned = join_to_spine(per, n)
            data = [[round(v) if v is not None else "" for v in row] for row in aligned]
        else:
            columns = [f"#{j + 1}" for j in range(max_boxes)]
            data = [[round(b[j][1]) if j < len(b) else "" for j in range(max_boxes)] for b in per]
        index = []
        for p in proteins:
            mw = f" mw={p['mw']}" if p.get("mw") else ""
            mark = "" if p["confirmed"] else " *"
            index.append(f"{p['name'] or '(unnamed)'} [{p['role']}]{mw}{mark}")
        results_table.value = {"data": data, "columns": columns, "index": index}

    def _refresh_all() -> None:
        _redraw()
        _refresh_readout()
        _refresh_table()

    # --- editing session ---
    def _set_editing(pid: int | None) -> None:
        state["editing"] = pid
        if pid is not None:
            _load_controls(state["proteins"][pid])
        _sync_combo()
        _refresh_readout()

    def _new_protein(*_) -> None:
        pid = len(state["proteins"])
        state["proteins"].append({
            "name": "", "role": role_in.value, "mw": "", "color": PALETTE[pid % len(PALETTE)],
            "base": _initial_size(image), "pad_w": 0, "pad_h": 0, "confirmed": False,
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
        return (x, y) if (0 <= x < iw and 0 <= y < ih) else None

    def _seed_grow(x: int, y: int) -> None:
        if state["editing"] is None:
            show_info("Press New protein (or pick one in 'editing') before placing boxes.")
            return
        pid = state["editing"]
        p = state["proteins"][pid]
        invert = bool(dark_on_light.value)
        box = grow_box(image_layer.data, (x, y), background, dark_on_light=invert)
        if box is None:
            show_info("No band detected at that point.")
            return
        gx0, gy0, gx1, gy1 = box
        gw, gh = max(2, gx1 - gx0), max(2, gy1 - gy0)
        rects = _pid_rects(pid)
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
        new = [normalize_corners(np.asarray(c)) for c in shapes.data]
        if len(new) != len(state["placed"]):
            _redraw()
            return
        cand = [
            _center_snap(new[i], _protein_size(state["proteins"][pl["pid"]]), iw, ih)
            for i, pl in enumerate(state["placed"])
        ]
        by_pid: dict[int, list[int]] = {}
        for i, pl in enumerate(state["placed"]):
            by_pid.setdefault(pl["pid"], []).append(i)
        for idxs in by_pid.values():
            rects = [cand[i] for i in idxs]
            pairs = ((a, b) for a in range(len(rects)) for b in range(a + 1, len(rects)))
            if any(overlaps(rects[a], rects[b]) for a, b in pairs):
                for i in idxs:
                    cand[i] = state["placed"][i]["rect"]
        for i, pl in enumerate(state["placed"]):
            pl["rect"] = cand[i]
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

    def _show_figure(spec) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from proteia.viz import render_figure

        if state["plot_dock"] is not None:
            # The user may have closed the previous chart; its Qt object is then
            # already gone, so removing it raises. Guard and reset either way.
            try:
                viewer.window.remove_dock_widget(state["plot_dock"])
            except (RuntimeError, KeyError, ValueError, AttributeError):
                pass
            state["plot_dock"] = None
        canvas = FigureCanvasQTAgg(render_figure(spec))
        dock = viewer.window.add_dock_widget(canvas, area="bottom", name="Chart")
        # Float it so the chart is a movable, resizable window instead of being
        # trapped in the cramped bottom dock.
        try:
            dock.setFloating(True)
            dock.resize(720, 560)
        except (AttributeError, RuntimeError):
            pass
        state["plot_dock"] = dock

    def on_plot(*_) -> None:
        proteins, spine = state["proteins"], state["spine"]
        if not spine:
            show_info("Set conditions first (comma-separated; repeats = replicates).")
            return
        _sync_spine_from_card()  # honour the latest card edits even without a change event
        n = len(spine)
        conditions, samples, included = spine_axes(spine)
        # Each protein's nets joined to the spine by its boxes' proposed positions
        # (sorted left-to-right). A missing box leaves an empty slot, not a shift.
        per = [propose_positions(_protein_boxes(pid)) for pid in range(len(proteins))]
        aligned = join_to_spine(per, n)
        # Per-box lane numbers are still proposed from x order, so a missing box can
        # still mis-propose. Warn on count mismatch until boxes carry explicit,
        # user-editable lane numbers (2b per-box numbering).
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

        targets = batch.targets()
        if len(targets) > 1:
            ignored = ", ".join(t.name for t in targets[1:])
            show_info(f"Multiple targets; plotting {targets[0].name}, ignoring: {ignored}")
        target, loading = targets[0], batch.loading_control()
        norm = normalize_lane(target.nets, loading.nets)
        # Provenance: condition -> its included source lanes (for later cropping/audit).
        provenance: dict[str, list[int]] = {}
        for i, cond in enumerate(conditions):
            if included[i]:
                provenance.setdefault(cond, []).append(i)
        error_type = ErrorType(error_in.value)

        if control:
            try:
                values = fold_change_lane(norm, conditions, control)
            except ValueError as exc:
                show_info(f"Cannot plot fold-change: {exc}")
                return
            kind = ValueKind.FOLD_CHANGE
            title = f"{target.name} fold-change vs {control}"
        else:
            values, kind = norm, ValueKind.LOADING_NORMALIZED
            title = f"{target.name} / {loading.name}"
        # Sample-aware: lanes sharing (condition, sample) are averaged before stats
        # (technical repeats don't inflate n); include=no lanes are dropped.
        reduction = reduce_samples(values, conditions, samples, included=included)
        groups = reduction.groups
        # Reference condition sits leftmost (it is the fold-change denominator).
        spec = build_plotspec(
            groups, describe(groups), compare(groups),
            value_kind=kind, error_type=error_type, title=title,
            lane_indices=provenance, first_label=control,
        )
        _show_figure(spec)
        notes = comp.warnings + reduction.warnings
        if mismatched:
            notes.append(
                f"box count != conditions ({n}) for {', '.join(mismatched)}; "
                "lane numbers proposed by position and may be misaligned"
            )
        if notes:
            show_info("; ".join(notes))

    shapes.events.data.connect(on_edit)
    viewer.mouse_drag_callbacks.append(on_mouse)
    width_in.changed.connect(on_size_change)
    height_in.changed.connect(on_size_change)
    padding_w_in.changed.connect(on_size_change)
    padding_h_in.changed.connect(on_size_change)
    protein_in.changed.connect(on_meta_change)
    mw_in.changed.connect(on_meta_change)
    role_in.changed.connect(on_meta_change)
    dark_on_light.changed.connect(lambda *_: _refresh_all())
    conditions_in.changed.connect(on_conditions_change)
    card_table.changed.connect(on_card_change)
    edit_combo.changed.connect(on_combo_change)
    new_btn.changed.connect(_new_protein)
    confirm_btn.changed.connect(_confirm_protein)
    remove_protein_btn.changed.connect(_remove_last_protein)
    clear_btn.changed.connect(_clear_boxes)
    plot_btn.changed.connect(on_plot)
    viewer.bind_key("Control-Z", _undo, overwrite=True)

    _refresh_all()
    show_info("Press New protein to start; set name/MW/role, Ctrl+click bands, then Confirm.")
    napari.run()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Proteia ROI quantification")
    parser.add_argument("image", nargs="?", default=None, help="image file (optional)")
    args = parser.parse_args()
    launch(args.image)


if __name__ == "__main__":
    main()
