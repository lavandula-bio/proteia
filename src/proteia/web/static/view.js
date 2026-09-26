// SPDX-License-Identifier: Apache-2.0
// The image view: a server-rendered preview on a canvas, with pan, zoom and box
// overlays. Everything it reports is in image pixels (x right, y down, a box's
// end exclusive), so a box drawn at any zoom lands on the same pixels the server
// quantifies. It edits nothing itself: it asks the app through its handlers.
"use strict";

const MIN_SCALE_FACTOR = 0.5; // of the fitted scale
const MAX_SCALE = 40; // screen pixels per image pixel
const CLICK_SLOP = 4; // screen pixels a click may wander before it is a drag

export const CLIPPED_COLOR = "#e0187a";
export const MISSING_COLOR = "#8a8a8a";

// Box geometry in image pixels.
function inside(rect, x, y) {
  return x >= rect[0] && x < rect[2] && y >= rect[1] && y < rect[3];
}

export class ImageView {
  // handlers: place(x, y, {grow, clientX, clientY}), move(boxId, rect),
  // select(boxId or null).
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
    this.boxes = []; // {id, rect, color, label, clipped}
    this.ghosts = []; // {rect, color, label}: declared lanes without a box
    this.selectedId = null;
    this.gesture = null;
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
      this.fit();
    }
    this.draw();
  }

  setOverlay(boxes, ghosts, selectedId) {
    this.boxes = boxes;
    this.ghosts = ghosts;
    this.selectedId = selectedId;
    this.draw();
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
    ctx.drawImage(this.bitmap, 0, 0);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); // overlays in screen pixels
    for (const ghost of this.ghosts) {
      this.drawRect(ghost.rect, ghost.color, { dashed: true, label: ghost.label });
    }
    const moving = this.gesture && this.gesture.kind === "move" ? this.gesture : null;
    for (const box of this.boxes) {
      let rect = box.rect;
      if (moving && moving.boxId === box.id) {
        rect = [rect[0] + moving.dx, rect[1] + moving.dy, rect[2] + moving.dx, rect[3] + moving.dy];
      }
      this.drawRect(rect, box.clipped ? CLIPPED_COLOR : box.color, {
        selected: box.id === this.selectedId,
        label: box.label,
        clipped: box.clipped,
      });
    }
  }

  toScreen(x, y) {
    return [(x - this.offsetX) * this.scale, (y - this.offsetY) * this.scale];
  }

  drawRect(rect, color, { dashed = false, selected = false, label = "", clipped = false }) {
    const ctx = this.context;
    const [x0, y0] = this.toScreen(rect[0], rect[1]);
    const [x1, y1] = this.toScreen(rect[2], rect[3]);
    ctx.save();
    if (selected) {
      ctx.lineWidth = 5;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }
    ctx.lineWidth = selected ? 3 : 2;
    ctx.strokeStyle = color;
    ctx.setLineDash(dashed ? [5, 4] : []);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
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

  // --- Pointer input ---

  boxAt(point) {
    for (let i = this.boxes.length - 1; i >= 0; i -= 1) {
      if (inside(this.boxes[i].rect, point.x, point.y)) {
        return this.boxes[i];
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

    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !this.bitmap) {
        return;
      }
      canvas.setPointerCapture(event.pointerId);
      const point = this.toImage(event);
      const box = this.boxAt(point);
      this.gesture = {
        kind: "press",
        boxId: box ? box.id : null,
        startX: event.clientX,
        startY: event.clientY,
        start: point,
        offsetX: this.offsetX,
        offsetY: this.offsetY,
        dx: 0,
        dy: 0,
      };
      if (box && box.id !== this.selectedId) {
        this.handlers.select(box.id);
      }
    });

    canvas.addEventListener("pointermove", (event) => {
      const g = this.gesture;
      if (!g) {
        return;
      }
      const sx = event.clientX - g.startX;
      const sy = event.clientY - g.startY;
      if (g.kind === "press" && Math.hypot(sx, sy) > CLICK_SLOP) {
        g.kind = g.boxId ? "move" : "pan";
      }
      if (g.kind === "pan") {
        this.offsetX = g.offsetX - sx / this.scale;
        this.offsetY = g.offsetY - sy / this.scale;
        this.draw();
      } else if (g.kind === "move") {
        g.dx = Math.round(sx / this.scale);
        g.dy = Math.round(sy / this.scale);
        this.draw();
      }
    });

    const finish = (event, cancelled) => {
      const g = this.gesture;
      this.gesture = null;
      if (!g || cancelled) {
        this.draw();
        return;
      }
      if (g.kind === "move") {
        const box = this.boxes.find((b) => b.id === g.boxId);
        if (box && (g.dx || g.dy)) {
          const r = box.rect;
          this.handlers.move(box.id, [r[0] + g.dx, r[1] + g.dy, r[2] + g.dx, r[3] + g.dy]);
        }
      } else if (g.kind === "press" && !g.boxId) {
        const x = Math.floor(g.start.x);
        const y = Math.floor(g.start.y);
        if (x >= 0 && y >= 0 && x < this.width && y < this.height) {
          this.handlers.place(x, y, {
            grow: event.shiftKey,
            clientX: event.clientX,
            clientY: event.clientY,
          });
        } else {
          this.handlers.select(null);
        }
      }
      this.draw();
    };
    canvas.addEventListener("pointerup", (event) => finish(event, false));
    canvas.addEventListener("pointercancel", (event) => finish(event, true));
  }
}
