// SPDX-License-Identifier: Apache-2.0
// The image view: a server-rendered preview on a canvas, with pan, zoom and box
// overlays. Everything it reports is in image pixels (x right, y down, a box's
// end exclusive), so a box drawn at any zoom lands on the same pixels the server
// quantifies. It edits nothing itself: it asks the app through its handlers.
import { counted } from "/static/dom.js";

const MIN_SCALE_FACTOR = 0.5; // of the fitted scale
const MAX_SCALE = 40; // screen pixels per image pixel
const CLICK_SLOP = 4; // screen pixels a click may wander before it is a drag
const MIDDLE_BUTTON = 1;

// A ladder ruler and the ladder marks, in screen pixels (#58): a tick reaches
// this far either side of its lane's x; a press this near a tick (or its
// label) takes it; the grips that stretch the ruler sit this far past its end
// ticks, and take a press within their radius; a press this near the ruler's
// line between its ends takes the ruler.
const TICK_HALF = 12;
const TICK_HIT = 6;
const GRIP_GAP = 16;
const GRIP_RADIUS = 7;
const BODY_HIT = 7;
const LABEL_FONT = "12px system-ui, sans-serif";

export const CLIPPED_COLOR = "#e0187a";
export const MISSING_COLOR = "#8a8a8a";
const EXTRA_PEAK_COLOR = "#bdbdbd";

// Box geometry in image pixels.
function inside(rect, x, y) {
  return x >= rect[0] && x < rect[2] && y >= rect[1] && y < rect[3];
}

// A straight line on the canvas, in the stroke style set.
function stroke(ctx, x0, y0, x1, y1) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

// Input types where Space is not typed.
const PRESSED = new Set([
  "button",
  "checkbox",
  "color",
  "file",
  "image",
  "radio",
  "reset",
  "submit",
]);

// Whether Space types or chooses in `element`: a text field, a select, editable text.
function typesSpace(element) {
  return (
    (element instanceof HTMLInputElement && !PRESSED.has(element.type)) ||
    element instanceof HTMLTextAreaElement ||
    element instanceof HTMLSelectElement ||
    (element instanceof HTMLElement && element.isContentEditable)
  );
}

// Whether `element` is a text field with a caret (whose typing can be taken back).
function hasCaret(element) {
  return (
    (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) &&
    element.selectionStart !== null
  );
}

// The row box dragged between two image points, as the server takes it: the
// corners rounded to whole pixels, x0 < x1 and y0 < y1 (end exclusive).
function rowRect(a, b) {
  const x0 = Math.round(Math.min(a.x, b.x));
  const y0 = Math.round(Math.min(a.y, b.y));
  const x1 = Math.max(x0 + 1, Math.round(Math.max(a.x, b.x)));
  const y1 = Math.max(y0 + 1, Math.round(Math.max(a.y, b.y)));
  return [x0, y0, x1, y1];
}

export class ImageView {
  // handlers: place(x, y, {grow, clientX, clientY, proteinId, laneIndex}),
  // row(rect, proteinId) (a Promise, settled once the row has its answer),
  // move(boxId, rect), select(boxId or null). A click on the membrane grows a
  // box from the band under it and Shift+click drops a box of the protein's
  // size; inside a lane's placeholder or n.d. mark, the click names its protein
  // and lane (otherwise both are null). A drag from the membrane draws a row
  // box while the app offers one (setRowTool), and pans otherwise; a drag with
  // Space held (whatever has the focus) or with the middle button always pans,
  // and so does a second finger on a touch screen. Esc cancels a drag, and so
  // does another image being shown.
  //
  // Molecular weights (#58) add: pick(x, y, {clientX, clientY}), a click on
  // the image while a pick tool is set (setPickTool), in continuous image
  // coordinates; ruler({phase, part, dx, dy, axis, alt, clientX, clientY}),
  // a press on the ruler (setRuler) taken as a drag ("move" as it goes, then
  // "drop", or "cancel") or a click ("click"), `part` being the ruler's line
  // ({kind: "body"}), a grip ({kind: "grip", end: "top" or "bottom"}) or a
  // tick ({kind: "tick", index, id}: its place in the ruler's ticks, and the
  // id the ruler gave it), `dx` and `dy` how far it went in image
  // pixels and `axis` ("x" or "y") the way a drag of the line goes; and
  // ladderPoint({phase, tick, y, alt, clientX, clientY}), a ladder mark
  // (setLadderTicks) dragged up or down ("drop", with its new y) or clicked
  // ("click"). While a pick tool or a ruler is set, a click places no box and
  // a drag moves no box and draws no row.
  constructor(canvas, handlers) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.handlers = handlers;
    this.bitmap = null;
    this.width = 0;
    this.height = 0;
    this.scale = 1;
    this.offsetX = 0; // the image point at the canvas's top-left corner
    this.offsetY = 0;
    this.boxes = []; // {id, rect, color, label, clipped, fitted (a rect inside, or null)}
    this.ghosts = []; // {rect, color, label, proteinId, laneIndex}: lanes without a box
    this.marks = []; // {rect, color, label, proteinId, laneIndex}: not-detected records
    this.selectedId = null;
    this.gesture = null;
    this.rowTool = null; // {proteinId, color, lanes}: what a drag on the membrane boxes, or null
    this.sentRow = null; // {rect, color}: the row box sent, shown until its answer
    // #58: the ladder marks of the shown image's register group ({x (null: the
    // left edge), y, label, color, labelsLeft}); the ruler being adjusted
    // ({x, labelsLeft, ticks: [{id, y, label, color, reference, solid,
    // focused}], extra: [y]}) or null; and the pick tool (a click reports
    // where it was) or null.
    this.ladderTicks = [];
    this.ruler = null;
    this.pickTool = null;
    this.rulerLabels = []; // per ruler tick, its label's [x0, x1] on the canvas, as last drawn
    this.ladderSpans = new Map(); // ladder mark -> its label's [x0, x1], as last drawn
    this.spaceHeld = false; // Space is down: a drag pans
    this.spaceTaken = false; // ...and its default (a scroll, a button press) was prevented
    this.spaceTyped = null; // {field, value, start, end}: the text field it types into, before
    this.bindEvents();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  // --- State from the app ---

  setImage(bitmap, width, height) {
    const changed = this.bitmap !== bitmap;
    this.bitmap = bitmap;
    this.width = width;
    this.height = height;
    if (changed) {
      // A drag, and a row box sent, belong to the image they were made on
      // (another project's image may share its ids).
      this.cancelGesture();
      this.sentRow = null;
      this.fit();
    }
    this.draw();
  }

  // The image shown drawn from another bitmap of it (its original colours, or
  // its grey analysis image again): the zoom, the pan and a drag under way stay,
  // and so do the boxes, since both cover the same image pixels.
  setBitmap(bitmap) {
    if (bitmap !== this.bitmap) {
      this.bitmap = bitmap;
      this.draw();
    }
  }

  setOverlay(boxes, ghosts, selectedId, marks = []) {
    this.boxes = boxes;
    this.ghosts = ghosts;
    this.marks = marks;
    this.selectedId = selectedId;
    this.draw();
  }

  // A drag from the membrane boxes a row of this protein ({proteinId, color,
  // lanes}), or pans (null). A drag under way keeps what it started with.
  setRowTool(tool) {
    this.rowTool = tool;
  }

  // The ladder marks drawn on the image (#58), each a tick at its x labelled
  // with its MW; a mark can be dragged up or down, or clicked, while no ruler
  // or pick tool is set. A drag of one under way keeps the mark it started on.
  setLadderTicks(ticks) {
    this.ladderTicks = ticks;
    this.ladderSpans = new Map();
    this.draw();
  }

  // The ruler being adjusted, or null. A drag of it under way goes on (its
  // handler answers each step with the ruler as it now is).
  setRuler(ruler) {
    this.ruler = ruler;
    this.draw();
  }

  // While set, a click on the image reports where it was (handlers.pick): a
  // ladder lane to find, a band to mark. Drags pan.
  setPickTool(tool) {
    this.pickTool = tool;
    this.showCursor();
  }

  // Where an image point lies on the screen (client coordinates): for a
  // popup put beside a tick.
  clientOf(x, y) {
    const bounds = this.canvas.getBoundingClientRect();
    const [sx, sy] = this.toScreen(x, y);
    return { clientX: bounds.left + sx, clientY: bounds.top + sy };
  }

  // --- View transform ---

  fittedScale() {
    const { clientWidth: w, clientHeight: h } = this.canvas;
    if (!this.width || !this.height || !w || !h) {
      return 1;
    }
    return Math.min(w / this.width, h / this.height) * 0.95;
  }

  fit() {
    this.scale = this.fittedScale();
    const { clientWidth: w, clientHeight: h } = this.canvas;
    this.offsetX = this.width / 2 - w / 2 / this.scale;
    this.offsetY = this.height / 2 - h / 2 / this.scale;
    this.draw();
  }

  zoomAt(factor, screenX, screenY) {
    const fitted = this.fittedScale();
    const scale = Math.min(MAX_SCALE, Math.max(fitted * MIN_SCALE_FACTOR, this.scale * factor));
    const imageX = this.offsetX + screenX / this.scale;
    const imageY = this.offsetY + screenY / this.scale;
    this.scale = scale;
    this.offsetX = imageX - screenX / scale; // the point under the cursor stays put
    this.offsetY = imageY - screenY / scale;
    this.draw();
  }

  zoomCentre(factor) {
    this.zoomAt(factor, this.canvas.clientWidth / 2, this.canvas.clientHeight / 2);
  }

  toImage(event) {
    const bounds = this.canvas.getBoundingClientRect();
    return {
      x: this.offsetX + (event.clientX - bounds.left) / this.scale,
      y: this.offsetY + (event.clientY - bounds.top) / this.scale,
    };
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    const { clientWidth: w, clientHeight: h } = this.canvas;
    const hadSize = this.canvas.width > 0 && this.canvas.height > 0;
    this.canvas.width = Math.round(w * ratio);
    this.canvas.height = Math.round(h * ratio);
    if (!hadSize) {
      this.fit();
    }
    this.draw();
  }

  // --- Drawing ---

  draw() {
    const ctx = this.context;
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.bitmap) {
      return;
    }
    const s = this.scale * ratio;
    ctx.setTransform(s, 0, 0, s, -this.offsetX * s, -this.offsetY * s);
    ctx.imageSmoothingEnabled = this.scale < 2; // show pixels when zoomed in
    // Over the image's own pixels, whatever the bitmap's size, so boxes stay on them.
    ctx.drawImage(this.bitmap, 0, 0, this.width, this.height);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); // overlays in screen pixels
    for (const ghost of this.ghosts) {
      this.drawRect(ghost.rect, ghost.color, { dashed: true, label: ghost.label });
    }
    for (const mark of this.marks) {
      this.drawRect(mark.rect, mark.color, { dotted: true, label: mark.label });
    }
    const moving = this.gesture && this.gesture.kind === "move" ? this.gesture : null;
    for (const box of this.boxes) {
      const moved = moving && moving.boxId === box.id;
      const shift = (rect) =>
        moved
          ? [rect[0] + moving.dx, rect[1] + moving.dy, rect[2] + moving.dx, rect[3] + moving.dy]
          : rect;
      const color = box.clipped ? CLIPPED_COLOR : box.color;
      this.drawRect(shift(box.rect), color, {
        selected: box.id === this.selectedId,
        label: box.label,
        clipped: box.clipped,
      });
      if (box.fitted) {
        this.drawRect(shift(box.fitted), color, { faint: true }); // its fitted size, inside
      }
    }
    const g = this.gesture;
    for (const tick of this.ladderTicks) {
      const dragged = g && g.kind === "point" && g.tick === tick;
      this.drawLadderTick(tick, dragged ? tick.y + g.dy : tick.y, dragged);
    }
    if (this.ruler) {
      this.drawRuler(this.ruler);
    }
    if (g && g.kind === "row") {
      const label = `Row box → ${counted(g.row.lanes, "lane", "lanes")}`;
      this.drawRect(rowRect(g.start, g.end), g.row.color, { dashed: true, label });
    } else if (this.sentRow) {
      this.drawRect(this.sentRow.rect, this.sentRow.color, { dashed: true });
    }
  }

  toScreen(x, y) {
    return [(x - this.offsetX) * this.scale, (y - this.offsetY) * this.scale];
  }

  drawRect(
    rect,
    color,
    {
      dashed = false,
      dotted = false,
      selected = false,
      label = "",
      clipped = false,
      faint = false,
    },
  ) {
    const ctx = this.context;
    const [x0, y0] = this.toScreen(rect[0], rect[1]);
    const [x1, y1] = this.toScreen(rect[2], rect[3]);
    ctx.save();
    if (selected) {
      ctx.lineWidth = 5;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }
    ctx.lineWidth = selected ? 3 : faint ? 1 : 2;
    ctx.strokeStyle = color;
    if (faint) {
      // A padded box's fitted size: a thin line, at low alpha. Not dashed (a
      // lane's placeholder, a row box) nor dotted (an n.d. mark).
      ctx.globalAlpha = 0.55;
    }
    ctx.setLineDash(dashed ? [5, 4] : dotted ? [2, 3] : []);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    if (dotted) {
      // A not-detected mark: a light wash sets it apart from a lane's placeholder.
      ctx.globalAlpha = 0.15;
      ctx.fillStyle = color;
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.globalAlpha = 1;
    }
    if (clipped) {
      // A filled corner: over-exposure reads without relying on colour alone.
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x1, y0);
      ctx.lineTo(x1 - 10, y0);
      ctx.lineTo(x1, y0 + 10);
      ctx.closePath();
      ctx.fill();
    }
    if (label) {
      ctx.font = "12px system-ui, sans-serif";
      ctx.textBaseline = "bottom";
      const width = ctx.measureText(label).width + 6;
      ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
      ctx.fillRect(x0, y0 - 16, width, 15);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x0 + 3, y0 - 2);
    }
    ctx.restore();
  }

  // A tick's label in a dark box, beside `edge` (screen x) at height `sy`: on
  // its left when `left`, else on its right; a reference band's (`color`) has
  // a strip of that colour on the side towards the tick; a predicted tick's
  // (`faint`) is grey. Gives the label's [x0, x1] on the canvas.
  drawTickLabel(text, edge, sy, { left, color = null, bold = false, faint = false }) {
    const ctx = this.context;
    ctx.font = bold ? `600 ${LABEL_FONT}` : LABEL_FONT;
    const strip = color ? 4 : 0;
    const width = ctx.measureText(text).width + 8 + strip;
    const x0 = left ? edge - 3 - width : edge + 3;
    ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
    ctx.fillRect(x0, sy - 8, width, 16);
    if (color) {
      ctx.fillStyle = color;
      ctx.fillRect(left ? x0 + width - strip : x0, sy - 8, strip, 16);
    }
    ctx.fillStyle = faint ? "#bdbdbd" : "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x0 + 4 + (left ? 0 : strip), sy + 0.5);
    return [x0, x0 + width];
  }

  // A ladder mark (#58): a tick across its lane at its x (the image's left
  // edge when it has none), labelled on the side away from the lanes; a
  // dragged one thicker. Gives its label's [x0, x1] on the canvas.
  drawLadderTick(tick, y, dragged) {
    const ctx = this.context;
    const [sx, sy] = this.toScreen(tick.x === null ? 0 : tick.x, y);
    ctx.save();
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
    ctx.lineWidth = dragged ? 7 : 5;
    stroke(ctx, sx - TICK_HALF, sy, sx + TICK_HALF, sy);
    ctx.strokeStyle = tick.color;
    ctx.lineWidth = dragged ? 4 : 2;
    stroke(ctx, sx - TICK_HALF, sy, sx + TICK_HALF, sy);
    const edge = tick.labelsLeft ? sx - TICK_HALF : sx + TICK_HALF;
    const span = this.drawTickLabel(tick.label, edge, sy, { left: tick.labelsLeft });
    ctx.restore();
    this.ladderSpans.set(tick, span);
  }

  // The ruler being adjusted (#58): a line between its end ticks, a grip past
  // each end that stretches it, a tick per ladder MW (solid on a band found,
  // snapped to or placed, hollow where only predicted; a reference band's in
  // its colour; the one with the keyboard focus haloed), each labelled, and a
  // grey dot at each peak no label took.
  drawRuler(ruler) {
    const ctx = this.context;
    this.rulerLabels = [];
    if (!ruler.ticks.length) {
      return;
    }
    const [sx] = this.toScreen(ruler.x, 0);
    const ys = ruler.ticks.map((tick) => this.toScreen(0, tick.y)[1]);
    const top = Math.min(...ys);
    const bottom = Math.max(...ys);
    ctx.save();
    ctx.lineCap = "round";
    // Over a dark edge, so it shows on a light and a dark membrane alike.
    ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
    ctx.lineWidth = 5;
    stroke(ctx, sx, top, sx, bottom);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    stroke(ctx, sx, top, sx, bottom);
    for (const [cy, sign] of [
      [top - GRIP_GAP, -1],
      [bottom + GRIP_GAP, 1],
    ]) {
      ctx.beginPath();
      ctx.arc(sx, cy, GRIP_RADIUS, 0, 2 * Math.PI);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
      ctx.stroke();
      // An arrow along the ruler: the grip pulls its end up or down.
      ctx.beginPath();
      ctx.moveTo(sx - 3, cy - 2 * sign);
      ctx.lineTo(sx, cy + 2 * sign);
      ctx.lineTo(sx + 3, cy - 2 * sign);
      ctx.strokeStyle = "#333333";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    for (const y of ruler.extra) {
      const [, py] = this.toScreen(0, y);
      ctx.beginPath();
      ctx.arc(sx, py, 4, 0, 2 * Math.PI);
      ctx.fillStyle = EXTRA_PEAK_COLOR;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
      ctx.stroke();
    }
    ruler.ticks.forEach((tick, index) => {
      const sy = ys[index];
      if (tick.focused) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 10;
        stroke(ctx, sx - TICK_HALF - 2, sy, sx + TICK_HALF + 2, sy);
      }
      if (tick.solid) {
        ctx.strokeStyle = "rgba(0, 0, 0, 0.85)";
        ctx.lineWidth = 6;
        stroke(ctx, sx - TICK_HALF, sy, sx + TICK_HALF, sy);
        ctx.strokeStyle = tick.color;
        ctx.lineWidth = 3;
        stroke(ctx, sx - TICK_HALF, sy, sx + TICK_HALF, sy);
      } else {
        // Hollow: an outline only, where the ladder's shape puts the band.
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = "rgba(0, 0, 0, 0.85)";
        ctx.strokeRect(sx - TICK_HALF, sy - 2.5, 2 * TICK_HALF, 5);
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = tick.color;
        ctx.strokeRect(sx - TICK_HALF, sy - 2.5, 2 * TICK_HALF, 5);
      }
      const edge = ruler.labelsLeft ? sx - TICK_HALF - 2 : sx + TICK_HALF + 2;
      this.rulerLabels[index] = this.drawTickLabel(tick.label, edge, sy, {
        left: ruler.labelsLeft,
        color: tick.reference ? tick.color : null,
        bold: tick.focused,
        faint: !tick.solid,
      });
    });
    ctx.restore();
  }

  // --- Pointer input ---

  boxAt(point) {
    for (let i = this.boxes.length - 1; i >= 0; i -= 1) {
      if (inside(this.boxes[i].rect, point.x, point.y)) {
        return this.boxes[i];
      }
    }
    return null;
  }

  // Where a pointer event lies on the canvas, in screen pixels.
  toCanvas(event) {
    const bounds = this.canvas.getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  }

  // The part of the ruler under a canvas point (`at`, screen pixels): a grip,
  // the tick nearest it (or whose label it is on), or the line; null for none.
  rulerAt(at) {
    const ruler = this.ruler;
    if (!ruler || !ruler.ticks.length) {
      return null;
    }
    const [sx] = this.toScreen(ruler.x, 0);
    const ys = ruler.ticks.map((tick) => this.toScreen(0, tick.y)[1]);
    const top = Math.min(...ys);
    const bottom = Math.max(...ys);
    if (Math.hypot(at.x - sx, at.y - (top - GRIP_GAP)) <= GRIP_RADIUS + 2) {
      return { kind: "grip", end: "top" };
    }
    if (Math.hypot(at.x - sx, at.y - (bottom + GRIP_GAP)) <= GRIP_RADIUS + 2) {
      return { kind: "grip", end: "bottom" };
    }
    let best = null;
    ys.forEach((sy, index) => {
      const [l0, l1] = this.rulerLabels[index] || [sx, sx];
      const near = Math.abs(at.y - sy);
      const across = at.x >= Math.min(sx - TICK_HALF, l0) && at.x <= Math.max(sx + TICK_HALF, l1);
      if (near <= TICK_HIT && across && (best === null || near < best.near)) {
        best = { index, near };
      }
    });
    if (best !== null) {
      return { kind: "tick", index: best.index, id: ruler.ticks[best.index].id };
    }
    if (Math.abs(at.x - sx) <= BODY_HIT && at.y >= top - TICK_HIT && at.y <= bottom + TICK_HIT) {
      return { kind: "body" };
    }
    return null;
  }

  // The ladder mark under a canvas point (its tick or its label), or null.
  ladderTickAt(at) {
    let best = null;
    for (const tick of this.ladderTicks) {
      const [sx, sy] = this.toScreen(tick.x === null ? 0 : tick.x, tick.y);
      const [l0, l1] = this.ladderSpans.get(tick) || [sx, sx];
      const near = Math.abs(at.y - sy);
      const across = at.x >= Math.min(sx - TICK_HALF, l0) && at.x <= Math.max(sx + TICK_HALF, l1);
      if (near <= TICK_HIT && across && (best === null || near < best.near)) {
        best = { tick, near };
      }
    }
    return best && best.tick;
  }

  // What a press at this canvas point would take, for the cursor: part of the
  // ruler, a ladder mark, or nothing ("").
  overAt(at) {
    const part = this.rulerAt(at);
    if (part) {
      return part.kind;
    }
    return !this.ruler && !this.pickTool && this.ladderTickAt(at) ? "mark" : "";
  }

  // The protein and lane under a point: a lane's placeholder, or an n.d. mark
  // of a first band; null elsewhere.
  laneAt(point) {
    const shapes = [...this.ghosts, ...this.marks];
    for (let i = shapes.length - 1; i >= 0; i -= 1) {
      const shape = shapes[i];
      if (shape.laneIndex !== null && inside(shape.rect, point.x, point.y)) {
        return { proteinId: shape.proteinId, laneIndex: shape.laneIndex };
      }
    }
    return null;
  }

  bindEvents() {
    const canvas = this.canvas;
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const bounds = canvas.getBoundingClientRect();
      this.zoomAt(
        Math.exp(-event.deltaY * 0.0015),
        event.clientX - bounds.left,
        event.clientY - bounds.top,
      );
    }, { passive: false });

    // The middle button pans here: no auto-scroll.
    canvas.addEventListener("mousedown", (event) => {
      if (event.button === MIDDLE_BUTTON) {
        event.preventDefault();
      }
    });

    canvas.addEventListener("pointerdown", (event) => {
      const g = this.gesture;
      if (g && event.pointerId !== g.pointerId) {
        this.secondPointer(g);
        return;
      }
      const middle = event.button === MIDDLE_BUTTON;
      if ((event.button !== 0 && !middle) || !this.bitmap) {
        return;
      }
      canvas.setPointerCapture(event.pointerId);
      const point = this.toImage(event);
      const pan = middle || this.spaceHeld;
      if (pan && !middle) {
        this.untype();
      }
      // A ladder being marked or adjusted (#58) takes the press before a box.
      const at = this.toCanvas(event);
      const calibrating = this.ruler !== null || this.pickTool !== null;
      const part = pan ? null : this.rulerAt(at);
      const tick = pan || calibrating ? null : this.ladderTickAt(at);
      const box = pan || calibrating || tick ? null : this.boxAt(point);
      const onImage = point.x >= 0 && point.y >= 0 && point.x < this.width && point.y < this.height;
      this.gesture = {
        kind: pan ? "pan" : "press",
        pointerId: event.pointerId,
        boxId: box ? box.id : null,
        part, // the part of the ruler pressed, or null
        tick, // the ladder mark pressed, or null
        axis: null, // the way a drag of the ruler's line goes, once it is a drag
        calibrating, // a click here is a pick, or does nothing beside the ruler
        // What a drag from here boxes: a row, from the membrane only.
        row: !pan && !box && !tick && !calibrating && onImage ? this.rowTool : null,
        startX: event.clientX,
        startY: event.clientY,
        lastX: event.clientX, // where the pointer is now
        lastY: event.clientY,
        start: point,
        end: point,
        offsetX: this.offsetX,
        offsetY: this.offsetY,
        dx: 0,
        dy: 0,
      };
      if (box && box.id !== this.selectedId) {
        this.handlers.select(box.id);
      }
      this.showCursor();
    });

    canvas.addEventListener("pointermove", (event) => {
      const g = this.gesture;
      if (!g) {
        // What a press here would take, for the cursor (app.css).
        canvas.dataset.over = this.spaceHeld ? "" : this.overAt(this.toCanvas(event));
        return;
      }
      if (event.pointerId !== g.pointerId) {
        return; // another finger: the gesture follows the first one
      }
      g.lastX = event.clientX;
      g.lastY = event.clientY;
      const sx = event.clientX - g.startX;
      const sy = event.clientY - g.startY;
      if (g.kind === "press" && Math.hypot(sx, sy) > CLICK_SLOP) {
        g.kind = g.part ? "ruler" : g.tick ? "point" : g.boxId ? "move" : g.row ? "row" : "pan";
        // The ruler's line goes the way the drag first went: up and down, or sideways.
        g.axis = Math.abs(sx) > Math.abs(sy) ? "x" : "y";
        this.showCursor();
      }
      if (g.kind === "pan") {
        this.offsetX = g.offsetX - sx / this.scale;
        this.offsetY = g.offsetY - sy / this.scale;
        this.draw();
      } else if (g.kind === "move") {
        g.dx = Math.round(sx / this.scale);
        g.dy = Math.round(sy / this.scale);
        this.draw();
      } else if (g.kind === "row") {
        g.end = this.toImage(event);
        this.draw();
      } else if (g.kind === "ruler") {
        g.dx = sx / this.scale;
        g.dy = sy / this.scale;
        this.handlers.ruler(this.rulerStep(g, "move", event));
      } else if (g.kind === "point") {
        g.dy = sy / this.scale;
        this.draw();
      }
    });

    const finish = (event, cancelled) => {
      const g = this.gesture;
      if (g && event.pointerId !== g.pointerId) {
        return; // a second finger lifted: the first one still drags
      }
      this.gesture = null;
      this.showCursor();
      if (!g || cancelled) {
        if (g) {
          this.abandon(g);
        }
        this.draw();
        return;
      }
      const clicked = { clientX: event.clientX, clientY: event.clientY };
      if (g.kind === "ruler") {
        this.handlers.ruler(this.rulerStep(g, "drop", event));
      } else if (g.kind === "point") {
        const y = g.tick.y + g.dy;
        const { tick } = g;
        this.handlers.ladderPoint({ phase: "drop", tick, y, alt: event.altKey, ...clicked });
      } else if (g.kind === "press" && g.part) {
        if (g.part.kind === "tick") {
          this.handlers.ruler(this.rulerStep(g, "click", event));
        }
      } else if (g.kind === "press" && g.tick) {
        const { tick } = g;
        const step = { phase: "click", tick, y: tick.y, alt: event.altKey };
        this.handlers.ladderPoint({ ...step, ...clicked });
      } else if (g.kind === "press" && g.calibrating) {
        // A pick tool takes the click where it was, on the image; beside the
        // ruler a click does nothing.
        const { x, y } = g.start;
        if (this.pickTool && x >= 0 && y >= 0 && x <= this.width && y <= this.height) {
          this.handlers.pick(x, y, clicked);
        }
      } else if (g.kind === "move") {
        const box = this.boxes.find((b) => b.id === g.boxId);
        if (box && (g.dx || g.dy)) {
          const r = box.rect;
          this.handlers.move(box.id, [r[0] + g.dx, r[1] + g.dy, r[2] + g.dx, r[3] + g.dy]);
        }
      } else if (g.kind === "row") {
        this.sendRow(rowRect(g.start, g.end), g.row);
      } else if (g.kind === "press" && !g.boxId) {
        const x = Math.floor(g.start.x);
        const y = Math.floor(g.start.y);
        if (this.selectedId !== null) {
          this.handlers.select(null); // a click away from a selected box only deselects
        } else if (x >= 0 && y >= 0 && x < this.width && y < this.height) {
          const lane = this.laneAt(g.start);
          this.handlers.place(x, y, {
            grow: !event.shiftKey,
            clientX: event.clientX,
            clientY: event.clientY,
            proteinId: lane ? lane.proteinId : null,
            laneIndex: lane ? lane.laneIndex : null,
          });
        } else {
          this.handlers.select(null);
        }
      }
      this.draw();
    };
    canvas.addEventListener("pointerup", (event) => finish(event, false));
    canvas.addEventListener("pointercancel", (event) => finish(event, true));
    // The capture lost otherwise (not by a pointerup or cancel): the drag ends,
    // so no press is left waiting for a release that never comes.
    canvas.addEventListener("lostpointercapture", (event) => {
      if (this.gesture && this.gesture.pointerId === event.pointerId) {
        this.cancelGesture();
      }
    });

    // Space held makes a drag pan, whatever has the focus. With the pointer
    // over the image it is also taken (it neither scrolls nor presses a focused
    // button, which it would on its release after the drag), but never from a
    // field it types into, a select or a dialog: typed into a text field, the
    // press on the image takes the spaces back (untype).
    const isSpace = (event) => event.code === "Space" || event.key === " ";
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && this.gesture) {
        // Taken: the app's Esc then leaves the ruler being dragged as it was
        // before the drag, and drops nothing more (#58).
        event.preventDefault();
        this.cancelGesture();
        return;
      }
      if (!isSpace(event) || event.ctrlKey || event.metaKey || event.altKey || event.isComposing) {
        return;
      }
      const target = event.target;
      const overImage = !document.querySelector("dialog[open]") && canvas.matches(":hover");
      if (target instanceof HTMLSelectElement) {
        // Space opens a select's list, and the page may never see its release:
        // over the image it pans instead (the list stays shut); elsewhere it
        // opens the list and is not held.
        if (overImage) {
          event.preventDefault();
          this.spaceTaken = true;
          this.holdSpace(true);
        }
        return;
      }
      if (!this.spaceHeld && hasCaret(target)) {
        const { value, selectionStart: start, selectionEnd: end } = target;
        this.spaceTyped = { field: target, value, start, end };
      }
      this.holdSpace(true);
      if (
        !typesSpace(target) &&
        !document.querySelector("dialog[open]") &&
        canvas.matches(":hover")
      ) {
        event.preventDefault();
        this.spaceTaken = true;
      }
    });
    document.addEventListener("keyup", (event) => {
      if (isSpace(event)) {
        if (this.spaceTaken) {
          event.preventDefault();
        }
        this.holdSpace(false);
      }
    });
    window.addEventListener("blur", () => this.holdSpace(false));
  }

  holdSpace(held) {
    if (!held) {
      this.spaceTaken = false;
      this.spaceTyped = null;
    }
    if (this.spaceHeld !== held) {
      this.spaceHeld = held;
      this.showCursor();
    }
  }

  // The press that makes Space+drag a pan leaves the text field Space was
  // typed into: the spaces it typed there go, and the field is as it was.
  untype() {
    const typed = this.spaceTyped;
    this.spaceTyped = null;
    if (!typed || document.activeElement !== typed.field) {
      return;
    }
    const { field, value, start, end } = typed;
    const head = value.slice(0, start);
    const tail = value.slice(end);
    const now = field.value;
    const spaces = now.slice(head.length, now.length - tail.length);
    if (
      now.length > head.length + tail.length &&
      now.startsWith(head) &&
      now.endsWith(tail) &&
      /^ +$/.test(spaces)
    ) {
      field.value = value;
      field.setSelectionRange(start, end);
    }
  }

  // A second finger while one drags: from here the drag pans with the first
  // finger (a pinch); it moves no box and draws no row.
  secondPointer(g) {
    if (g.kind !== "pan") {
      this.abandon(g);
      Object.assign(g, {
        kind: "pan",
        boxId: null,
        part: null,
        tick: null,
        row: null,
        startX: g.lastX,
        startY: g.lastY,
        offsetX: this.offsetX,
        offsetY: this.offsetY,
        dx: 0,
        dy: 0,
      });
      this.showCursor();
      this.draw();
    }
  }

  // An open hand while a drag would pan (Space held), a closed one while it
  // does; while a pick tool is set, the pointer says a click picks (#58).
  showCursor() {
    const g = this.gesture;
    this.canvas.classList.toggle("grabbing", Boolean(g && g.kind === "pan"));
    this.canvas.classList.toggle("grab", this.spaceHeld && !g);
    this.canvas.classList.toggle("picking", this.pickTool !== null && !this.spaceHeld);
    if (this.spaceHeld || (g && g.kind === "pan")) {
      this.canvas.dataset.over = "";
    }
  }

  // What a drag or click of the ruler reports (handlers.ruler), in image pixels.
  rulerStep(g, phase, event) {
    return {
      phase,
      part: g.part,
      dx: g.dx,
      dy: g.dy,
      axis: g.axis,
      alt: event.altKey,
      clientX: event.clientX,
      clientY: event.clientY,
    };
  }

  // A drag given up (Esc, another finger, another image): the ruler goes back
  // to how it was before it; a ladder mark or a box being dragged just stays.
  abandon(g) {
    if (g.kind === "ruler") {
      this.handlers.ruler({ phase: "cancel", part: g.part, dx: 0, dy: 0, axis: g.axis });
    }
  }

  // The drag under way, if any, does nothing more (a box being moved goes back).
  cancelGesture() {
    const g = this.gesture;
    if (!g) {
      return;
    }
    this.gesture = null;
    if (this.canvas.hasPointerCapture(g.pointerId)) {
      this.canvas.releasePointerCapture(g.pointerId);
    }
    this.abandon(g);
    this.showCursor();
    this.draw();
  }

  // Send a row box; it stays drawn, without its label, until it has its answer.
  sendRow(rect, tool) {
    const sent = { rect, color: tool.color };
    this.sentRow = sent;
    const done = () => {
      if (this.sentRow === sent) {
        this.sentRow = null;
        this.draw();
      }
    };
    Promise.resolve(this.handlers.row(rect, tool.proteinId)).then(done, done);
  }
}
