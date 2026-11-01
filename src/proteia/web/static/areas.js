// SPDX-License-Identifier: Apache-2.0
// "Membranes on this image" (#243): where each membrane lies on the image
// shown. One item per membrane area (its membrane, Edit outline, Remove), and
// "Add membrane area". An image of one membrane has one area over the whole
// image, drawn as a thin frame; "Edit outline" on it starts a split (an
// outline cut across it gives the far side, with the rows there, to a new
// membrane, as the server does it). Drawing and editing an outline happen in
// an editor: a drag on the image draws the
// rectangle, a drag inside it moves it and the handle above it turns it; the
// fields type its corners and angle; arrow keys move it (Shift: 10 px) and
// [ and ] turn it (Shift: 1°). Enter or Done sends it, Esc cancels; leaving
// the image, the area or the project discards it, and so does Undo. The page
// draws what the server gives: an area's corners and bounds come from the
// state, never worked out here; only an outline being drawn, not sent yet, is
// drawn from its own corners and angle.
import { $, counted, inWords, isolate, sentence } from "/static/dom.js";

export const MAX_ANGLE = 45; // degrees either way, as the server takes
export const DISCARDED = "The unfinished membrane area was discarded.";
export const REMOVED_ELSEWHERE = "That membrane area was removed in another window.";

// A membrane's colour on the image, by its place in the project.
const AREA_COLORS = [
  "#1f77b4",
  "#e6550d",
  "#2ca02c",
  "#9467bd",
  "#d62728",
  "#8c564b",
  "#e377c2",
  "#17becf",
  "#bcbd22",
];

export function areaColor(project, membraneId) {
  const index = project.membranes.findIndex((membrane) => membrane.id === membraneId);
  return AREA_COLORS[Math.max(0, index) % AREA_COLORS.length];
}

// The membrane areas on an image, in membrane order.
export function areasOn(project, imageId) {
  return project.arenas.filter((area) => area.image_id === imageId);
}

function labelOf(project, membraneId) {
  const membrane = project.membranes.find((each) => each.id === membraneId);
  return membrane ? membrane.label : "Membrane";
}

// "membrane 2", as the server's messages name it.
function lower(label) {
  return label.charAt(0).toLowerCase() + label.slice(1);
}

// What removing an area takes, as the server's removal scope gives it (the
// membrane's area on every linked image, the proteins on them and their
// boxes), in the confirmation's words.
function removalWords(project, area) {
  const removal = area.removal;
  const images = removal.image_ids
    .map((id) => project.images.find((each) => each.id === id))
    .filter(Boolean)
    .map((each) => isolate(each.original_name));
  const names = removal.protein_ids
    .map((id) => project.proteins.find((each) => each.id === id))
    .filter(Boolean)
    .map((each) => `'${isolate(each.name)}'`);
  let what = " Undo brings it back.";
  if (names.length) {
    const who = names.length === 1 ? `The protein ${names[0]}` : `The proteins ${inWords(names)}`;
    const theirs = names.length === 1 ? "its" : "their";
    const boxes = removal.boxes ? ` and ${theirs} ${counted(removal.boxes, "box", "boxes")}` : "";
    const verb = names.length === 1 && !removal.boxes ? "goes" : "go";
    what = ` ${who}${boxes} ${verb} too; Undo brings them back.`;
  }
  return { images: inWords(images), what };
}

const FIELDS = ["x0", "y0", "x1", "y1"];

export class AreaPanel {
  // handlers: edit(method, path, json, {refused}) (the page's edit: in order
  // with the other edits, its answer shown through the stale-answer guard;
  // gives the answer, or null for one about another opening), refusal(error,
  // text) (says `text` unless the page says the refusal elsewhere), status(text),
  // reread() (a Promise), choose(imageId, arenaId) (remember the area chosen
  // on that image and show it if the image is shown), settled() (a Promise
  // that settles once every edit sent before has its answer: an area change
  // waits for them, so a box placed on an area is there before the area changes).
  constructor(view, handlers) {
    this.view = view;
    this.handlers = handlers;
    this.project = null;
    this.image = null;
    this.arena = null;
    // The outline being drawn or edited, not sent yet: {openId, imageId,
    // arenaId (null for a new area), outline [x0, y0, x1, y1] before its turn,
    // angle}; or null.
    this.draft = null;
    this.busy = false; // the draft is sent: Done and the fields wait for its answer
    this.removing = new Set(); // areas whose removal is sent
    this.assigning = new Set(); // areas whose membrane change is sent
    this.rereading = null; // the project read again after a refusal, or null
    // An outline discarded because another project is shown: said with the
    // words of that switch (takeDiscarded), which replace the status line.
    this.leftProject = false;
    this.bind();
  }

  bind() {
    $("area-add").addEventListener("click", () => this.open(null));
    $("area-done").addEventListener("click", () => this.commit());
    $("area-cancel").addEventListener("click", () => this.cancel(true));
    for (const name of [...FIELDS, "angle"]) {
      $(`area-${name}`).addEventListener("input", () => this.typed());
    }
    $("area-editor").addEventListener("keydown", (event) => this.key(event));
  }

  drafting() {
    return this.draft !== null;
  }

  // What another project shown says of an outline it discarded, once ("").
  takeDiscarded() {
    const said = this.leftProject ? DISCARDED : "";
    this.leftProject = false;
    return said;
  }

  // --- Drawing the section ---

  render(project, image, arena) {
    this.project = project;
    this.image = image;
    this.arena = arena;
    const draft = this.draft;
    if (draft && !this.busy) {
      const gone =
        draft.openId !== project.open_id ||
        !image ||
        draft.imageId !== image.id ||
        (draft.arenaId !== null &&
          (!arena || arena.id !== draft.arenaId || !project.arenas.some((a) => a.id === draft.arenaId)));
      if (gone) {
        this.draft = null;
        this.leftProject = draft.openId !== project.open_id;
        this.handlers.status(DISCARDED);
      }
    }
    const section = $("areas");
    section.hidden = !image;
    if (!image) {
      this.view.setAreas([], null);
      this.view.setAreaTool(null);
      return;
    }
    const areas = areasOn(project, image.id);
    this.renderList(areas);
    const covering = areas.find((area) => area.outline === null);
    const add = $("area-add");
    add.disabled = Boolean(covering) || this.draft !== null || this.busy;
    const hint = $("area-hint");
    hint.hidden = !covering;
    if (covering) {
      const label = labelOf(project, covering.membrane_id);
      hint.textContent = `Draw ${lower(label)}'s outline first (Edit outline), then add the next membrane's area.`;
    }
    this.renderEditor();
    this.drawOnView(areas);
  }

  renderList(areas) {
    const project = this.project;
    const list = $("area-list");
    const active = document.activeElement;
    const key = active && list.contains(active) ? active.dataset.key : undefined;
    list.replaceChildren();
    for (const area of areas) {
      const label = labelOf(project, area.membrane_id);
      const chosen = this.arena && this.arena.id === area.id;
      const item = document.createElement("li");
      item.className = chosen ? "area chosen" : "area";
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.backgroundColor = areaColor(project, area.membrane_id);
      const pick = document.createElement("button");
      pick.type = "button";
      pick.className = "area-pick";
      pick.dataset.key = `pick:${area.id}`;
      pick.setAttribute("aria-pressed", String(chosen));
      pick.append(swatch, label);
      pick.title = this.whereText(area);
      pick.addEventListener("click", () => this.handlers.choose(area.image_id, area.id));
      const select = document.createElement("select");
      select.dataset.key = `membrane:${area.id}`;
      select.setAttribute("aria-label", `Membrane of ${label}'s area`);
      for (const membrane of project.membranes) {
        select.append(new Option(membrane.label, membrane.id));
      }
      select.append(new Option("New membrane", ""));
      select.value = area.membrane_id;
      select.disabled = this.assigning.has(area.id) || this.busy;
      select.addEventListener("change", () => this.assign(area, select.value));
      const edit = document.createElement("button");
      edit.type = "button";
      edit.dataset.key = `edit:${area.id}`;
      edit.textContent = "Edit outline";
      edit.setAttribute("aria-label", `Edit the outline of ${label}'s area`);
      edit.disabled = this.draft !== null || this.busy;
      edit.addEventListener("click", () => this.open(area));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.dataset.key = `remove:${area.id}`;
      remove.textContent = "Remove";
      remove.setAttribute("aria-label", `Remove ${label}'s area`);
      const alone = areas.length < 2;
      remove.disabled = alone || this.removing.has(area.id) || this.busy;
      remove.title = alone
        ? "An image keeps at least one membrane area: remove the image instead, or give its area to another membrane"
        : "";
      remove.addEventListener("click", () => this.remove(area));
      const controls = document.createElement("span");
      controls.className = "area-controls";
      controls.append(select, edit, remove);
      item.append(pick, controls);
      list.append(item);
    }
    if (key !== undefined) {
      const same = [...list.querySelectorAll("[data-key]")].find((e) => e.dataset.key === key);
      if (same && !same.disabled) {
        same.focus();
      }
    }
  }

  // Where an area lies, for its item's tooltip.
  whereText(area) {
    const outline = area.outline;
    const where = outline
      ? `the rectangle (${outline[0]}, ${outline[1]})–(${outline[2]}, ${outline[3]})`
      : "the whole image";
    return area.angle_deg ? `${where}, turned ${area.angle_deg}°` : where;
  }

  renderEditor() {
    const draft = this.draft;
    const editor = $("area-editor");
    editor.hidden = !draft;
    if (!draft) {
      return;
    }
    const label = draft.arenaId
      ? labelOf(this.project, (this.project.arenas.find((a) => a.id === draft.arenaId) || {}).membrane_id)
      : "A new membrane";
    $("area-editor-title").textContent = draft.arenaId
      ? `Outline of ${lower(label)}'s area`
      : "The new membrane's area";
    const values = [...draft.outline, draft.angle];
    [...FIELDS, "angle"].forEach((name, index) => {
      const field = $(`area-${name}`);
      field.disabled = this.busy;
      if (document.activeElement !== field || field.dataset.sync === "1") {
        field.value = String(values[index]);
        field.dataset.sync = "";
      }
    });
    $("area-done").disabled = this.busy;
    $("area-cancel").disabled = this.busy;
  }

  drawOnView(areas) {
    const project = this.project;
    const several = areas.length > 1;
    this.view.setAreas(
      areas.map((area) => ({
        id: area.id,
        corners: area.corners,
        color: areaColor(project, area.membrane_id),
        whole: area.outline === null,
        label: several ? labelOf(project, area.membrane_id) : "",
        hidden: this.draft !== null && this.draft.arenaId === area.id,
      })),
      several && this.arena ? this.arena.id : null,
    );
    const draft = this.draft;
    this.view.setAreaTool(
      draft
        ? {
            outline: draft.outline,
            angle: draft.angle,
            color: draft.arenaId
              ? areaColor(project, (project.arenas.find((a) => a.id === draft.arenaId) || {}).membrane_id)
              : "#ffd54a",
          }
        : null,
    );
  }

  // --- The editor ---

  // Open the editor on `area` (Edit outline), or on a new area (null).
  open(area) {
    const image = this.image;
    if (!image || this.busy) {
      return;
    }
    let outline;
    if (area) {
      outline = area.outline ? [...area.outline] : [0, 0, image.width, image.height];
    } else {
      outline = [0, Math.ceil(image.height / 2), image.width, image.height];
    }
    this.draft = {
      openId: this.project.open_id,
      imageId: image.id,
      arenaId: area ? area.id : null,
      outline,
      angle: area ? area.angle_deg : 0,
    };
    if (area) {
      this.handlers.choose(image.id, area.id);
    }
    this.render(this.project, this.image, this.arena);
    $("area-keys").focus();
  }

  // Esc or Cancel: nothing is sent, the outline goes back to the stored one.
  cancel(refocus) {
    if (!this.draft || this.busy) {
      return false;
    }
    const arenaId = this.draft.arenaId;
    this.draft = null;
    this.render(this.project, this.image, this.arena);
    if (refocus) {
      this.focusItem(arenaId);
    }
    return true;
  }

  // Undo or another change asked for while an outline is unfinished: it is
  // dropped first, unsent.
  discard() {
    if (!this.draft || this.busy) {
      return;
    }
    this.draft = null;
    if (this.project) {
      this.render(this.project, this.image, this.arena);
    }
  }

  // Esc on the page while an outline is unfinished.
  escape() {
    return this.cancel(true);
  }

  // The fields as typed: the draft follows each that reads as a number.
  typed() {
    const draft = this.draft;
    if (!draft || this.busy) {
      return;
    }
    const corners = FIELDS.map((name) => Number.parseInt($(`area-${name}`).value, 10));
    if (corners.every(Number.isFinite)) {
      draft.outline = corners;
    }
    const angle = Number.parseFloat($("area-angle").value);
    if (Number.isFinite(angle)) {
      draft.angle = angle;
    }
    this.drawOnView(areasOn(this.project, this.image.id));
  }

  // A drag on the image while the editor is open (the view's): the outline
  // drawn, moved or turned.
  fromView(outline, angle) {
    const draft = this.draft;
    if (!draft || this.busy) {
      return;
    }
    draft.outline = outline;
    draft.angle = angle;
    for (const name of [...FIELDS, "angle"]) {
      $(`area-${name}`).dataset.sync = "1";
    }
    this.render(this.project, this.image, this.arena);
  }

  key(event) {
    const draft = this.draft;
    if (!draft || this.busy) {
      return;
    }
    const shift = event.shiftKey;
    if (event.key === "Enter" && !(event.target instanceof HTMLButtonElement)) {
      event.preventDefault();
      this.commit();
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      this.cancel(true);
      return;
    }
    // [ and ] by the key's place too: with Shift held they read { and }.
    const turn =
      event.key === "]" || event.code === "BracketRight"
        ? 1
        : event.key === "[" || event.code === "BracketLeft"
          ? -1
          : 0;
    if (turn) {
      event.preventDefault();
      const step = turn * (shift ? 1 : 0.1);
      const turned = Math.round((draft.angle + step) * 10) / 10;
      this.fromView(draft.outline, Math.max(-MAX_ANGLE, Math.min(MAX_ANGLE, turned)));
      return;
    }
    const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (moves[event.key] && event.target === $("area-keys")) {
      event.preventDefault();
      const [dx, dy] = moves[event.key].map((d) => d * (shift ? 10 : 1));
      const [x0, y0, x1, y1] = draft.outline;
      this.fromView([x0 + dx, y0 + dy, x1 + dx, y1 + dy], draft.angle);
    }
  }

  // Done or Enter: the outline sent, once (Done and the fields wait for its
  // answer). Its answer selects the area; a refusal puts the stored outline
  // back and says why.
  async commit() {
    const draft = this.draft;
    if (!draft || this.busy) {
      return;
    }
    const [a, b, c, d] = draft.outline;
    const outline = [Math.min(a, c), Math.min(b, d), Math.max(a, c), Math.max(b, d)];
    this.busy = true;
    this.render(this.project, this.image, this.arena);
    const body = draft.arenaId
      ? { outline, angle_deg: draft.angle }
      : { image_id: draft.imageId, outline, angle_deg: draft.angle, membrane_id: null };
    const path = draft.arenaId ? `/api/arenas/${draft.arenaId}` : "/api/arenas";
    let answer = null;
    try {
      await this.handlers.settled();
      answer = await this.handlers.edit(draft.arenaId ? "PUT" : "POST", path, body, {
        refused: (error) => this.refused(error, draft.arenaId ? "Outline not changed" : "Membrane area not added"),
      });
    } catch {
      answer = null;
    } finally {
      this.busy = false;
      if (this.draft === draft) {
        this.draft = null;
      }
    }
    // The keyboard goes back to the section only while it still shows the
    // image the outline was drawn on, in the project it was drawn in.
    const here = () =>
      this.project && this.project.open_id === draft.openId && this.image &&
      this.image.id === draft.imageId;
    if (!answer) {
      if (this.project) {
        this.render(this.project, this.image, this.arena);
      }
      await this.reread();
      if (here()) {
        this.focusItem(draft.arenaId);
      }
      return;
    }
    const made = answer.project.arenas.find(
      (area) => answer.arena_ids.includes(area.id) && area.image_id === draft.imageId,
    );
    const arenaId = made ? made.id : draft.arenaId;
    this.handlers.choose(draft.imageId, arenaId);
    // A cut says what went to the new membrane, in the server's words.
    const words = [
      answer.handed_off ? sentence(answer.handed_off) : "",
      ...answer.mw_lost.map((lost) => lost.message),
    ];
    this.handlers.status(words.filter(Boolean).join(" "));
    if (here()) {
      this.focusItem(arenaId);
    }
  }

  // A refusal of an area change: the server's reason; one about an area
  // another window removed reads the project again and says so.
  refused(error, head) {
    if (error.status === 404 && error.code === "unknown_id" && error.detail && error.detail.arena_id) {
      this.rereading = Promise.resolve(this.handlers.reread()).finally(() => {
        this.handlers.status(REMOVED_ELSEWHERE);
      });
      this.handlers.status(REMOVED_ELSEWHERE);
      return;
    }
    this.handlers.refusal(error, `${head}: ${sentence(error.message)}`);
  }

  // Once the project read again after a refusal is shown: what the focus goes to.
  async reread() {
    const reading = this.rereading;
    this.rereading = null;
    if (reading) {
      await reading.catch(() => {});
    }
  }

  // The keyboard to an area's item (its name), else to Add membrane area.
  focusItem(arenaId) {
    const target =
      (arenaId && document.querySelector(`#area-list [data-key="pick:${CSS.escape(arenaId)}"]`)) ||
      $("area-add");
    if (target && !target.disabled && target.getClientRects().length) {
      target.focus();
    } else {
      const first = document.querySelector("#area-list .area-pick");
      if (first) {
        first.focus();
      }
    }
  }

  // --- Membrane, removal ---

  async assign(area, membraneId) {
    if (this.assigning.has(area.id)) {
      return;
    }
    this.assigning.add(area.id);
    this.render(this.project, this.image, this.arena);
    try {
      await this.handlers.settled();
      const answer = await this.handlers.edit(
        "PUT",
        `/api/arenas/${area.id}/membrane`,
        { membrane_id: membraneId || null },
        { refused: (error) => this.refused(error, "Membrane not changed") },
      );
      if (answer) {
        const words = [
          answer.renumbered ? `${sentence(answer.renumbered)}` : "",
          ...answer.mw_lost.map((lost) => lost.message),
        ];
        this.handlers.status(words.filter(Boolean).join(" "));
      }
    } catch {
      // said by refused
    } finally {
      this.assigning.delete(area.id);
      if (this.project) {
        this.render(this.project, this.image, this.arena);
      }
    }
    const select = document.querySelector(
      `#area-list [data-key="membrane:${CSS.escape(area.id)}"]`,
    );
    if (select && (document.activeElement === document.body || !document.activeElement)) {
      select.focus();
    }
  }

  // Remove: confirmed first, naming what goes with it as the server's removal
  // scope says (every linked image, whose area goes too, and the proteins on
  // them); the confirm is about the area it named.
  async remove(area) {
    const project = this.project;
    const image = project.images.find((each) => each.id === area.image_id);
    if (!image || this.removing.has(area.id)) {
      return;
    }
    const label = labelOf(project, area.membrane_id);
    const { images, what } = removalWords(project, area);
    if (!window.confirm(`Remove ${lower(label)}'s area on ${images}?${what}`)) {
      return;
    }
    const areas = areasOn(project, image.id);
    const after = areas[areas.findIndex((each) => each.id === area.id) + 1] || null;
    this.removing.add(area.id);
    this.render(this.project, this.image, this.arena);
    let answer = null;
    try {
      await this.handlers.settled();
      answer = await this.handlers.edit("DELETE", `/api/arenas/${area.id}`, undefined, {
        refused: (error) => this.refused(error, "Membrane area not removed"),
      });
    } catch {
      answer = null;
    } finally {
      this.removing.delete(area.id);
      if (this.project) {
        this.render(this.project, this.image, this.arena);
      }
    }
    if (answer) {
      const boxes = answer.removed.filter((id) => id.startsWith("band-")).length;
      const words = [
        boxes ? `Removed ${lower(label)}'s area and ${counted(boxes, "box", "boxes")} on it.` : `Removed ${lower(label)}'s area.`,
        answer.renumbered ? sentence(answer.renumbered) : "",
      ];
      this.handlers.status(words.filter(Boolean).join(" "));
    }
    await this.reread();
    const next = after && document.querySelector(`#area-list [data-key="pick:${CSS.escape(after.id)}"]`);
    this.focusItem(next ? after.id : null);
  }
}
