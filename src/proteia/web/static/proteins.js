// SPDX-License-Identifier: Apache-2.0
// The "Proteins on this image" panel: the list with each protein's colour and
// role, the form that adds a protein, and the editor of the chosen one (name,
// role, loading controls, fitted size and padding, not-detected marks, clearing
// its boxes, removal). It stores nothing itself: each change goes to the server
// through the app, and the panel is drawn again from the state the server
// answers.
import {
  $,
  counted,
  focusLost,
  inWords,
  isolate,
  lanesPhrase,
  rebuild,
  span,
  swatch,
} from "/static/dom.js";

const PROTEIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#bcbd22"];
const LOADING_CONTROL = "loading control";

export function colorOf(project, proteinId) {
  const index = project.proteins.findIndex((p) => p.id === proteinId);
  return PROTEIN_COLORS[index % PROTEIN_COLORS.length];
}

function loadingControls(project) {
  return project.proteins.filter((p) => p.role === LOADING_CONTROL);
}

// The loading controls a target is normalized to, as the server computes it:
// those it names, in series order, or else the batch's only one.
function usedLoadingControls(project, protein) {
  if (protein.role === LOADING_CONTROL) {
    return [];
  }
  if (protein.loading_control_ids.length) {
    return protein.loading_control_ids
      .map((id) => project.proteins.find((p) => p.id === id))
      .filter(Boolean);
  }
  const all = loadingControls(project);
  return all.length === 1 ? all : [];
}

function names(proteins) {
  return proteins.map((p) => p.name).join(", ");
}

// A protein's padding directions as the editor shows them, above and below
// first: its key in box_padding, its field, and the fitted size's dimension it
// pads.
const PADDING = [
  { key: "along", field: "pad-along", dimension: "height", words: "above and below" },
  { key: "across", field: "pad-across", dimension: "width", words: "left and right" },
];

// "5 px above and below and 2 px left and right"; "" for no padding.
function paddingWords(padding) {
  const sides = PADDING.filter(({ key }) => padding[key]);
  return inWords(sides.map(({ key, words }) => `${padding[key]} px ${words}`));
}

// The size of a protein's boxes, the one quantified and drawn: its fitted size
// plus its padding on each side.
function paddedSize({ fitted_size: fitted, box_padding: padding }) {
  return {
    width: fitted.width + 2 * padding.across,
    height: fitted.height + 2 * padding.along,
  };
}

// How a protein's nets moved, from the smallest and largest share of change:
// "rose 8.6–11.5%", "fell 1.2%", "changed −1.2 to +0.4%".
function netChangeWords([low, high]) {
  const percent = (value) => Math.abs(value * 100).toFixed(1);
  const range = (a, b) =>
    percent(a) === percent(b) ? `${percent(a)}%` : `${percent(a)}–${percent(b)}%`;
  if (percent(low) === "0.0" && percent(high) === "0.0") {
    return "did not change";
  }
  if (low >= 0) {
    return `rose ${range(low, high)}`;
  }
  if (high <= 0) {
    return `fell ${range(high, low)}`;
  }
  const signed = (value) => `${value < 0 ? "−" : "+"}${percent(value)}`;
  return `changed ${signed(low)} to ${signed(high)}%`;
}

export class ProteinPanel {
  // handlers: send(method, path, json) gives a Promise of the server's answer
  // once the app has applied it (null if the answer is about a project opened
  // before the one shown), and rejects with the server's refusal (code,
  // message, ids); choose(proteinId) makes a protein the click target;
  // status(text, action) shows a line to the user, with an optional action
  // button ({label, name, seq, run}) after it, and gives that button (or
  // null); undo(seq) takes back the change logged as `seq` if it is still the
  // last; pending() gives a Promise that settles once the edits the app made
  // outside the queue have their answers; laneName(index) names a lane;
  // remeasured(answer) says which other proteins' nets an edit re-measured, or
  // gives null.
  constructor(handlers) {
    this.handlers = handlers;
    this.project = null;
    this.results = null; // the results of that same project state
    this.image = null;
    this.protein = null;
    this.filled = new Map(); // input id -> the value the panel last put in it
    this.editorFor = null; // the protein the editor was last drawn for
    // Edits, undo and redo run one at a time, in order; the app's box and image
    // edits and the adds wait for it as it stands when they are made.
    this.queue = Promise.resolve();
    this.adding = Promise.resolve(); // the adds on their way (they are not queued)
    this.opening = 0; // counts invalidateEdits(): a queued edit never reaches another project
    this.clearing = new Set(); // the proteins whose Clear boxes has no answer yet
    this.bind();
  }

  // Resolves once every edit made in the panel, and every undo and redo (they
  // run in its queue), has its answer, and what the answer shows (the editor, a
  // refusal, the add form closing, a status line) is shown. The app waits for it
  // before asking to open another project, so those edits end in the project
  // they were made in.
  settled() {
    return Promise.all([this.queue, this.adding]);
  }

  // No edit made before this, nor the answer to one, is applied from now on:
  // called just before another project is asked for, since its ids repeat those
  // of the one open (prot-3 in each), and again once it is shown, since an edit
  // made in the one before could still be waiting then (behind a refusal
  // answered late). After settled() nothing is left to drop; this keeps it so.
  invalidateEdits() {
    this.opening += 1;
  }

  // Forget what was typed in the editor and the add form: another project is
  // shown now. Not before: a failed open leaves the open project's panel as it was.
  forgetTyped() {
    this.filled.clear();
    this.editorFor = null;
    this.closeAdd(false);
  }

  render(project, image, proteinId, results = null) {
    this.project = project;
    this.results = results;
    this.image = image;
    const proteins = image ? project.proteins.filter((p) => p.image_id === image.id) : [];
    this.protein = proteins.find((p) => p.id === proteinId) || null;
    rebuild($("proteins"), () => this.buildList(proteins));
    this.renderAdd();
    this.renderEditor();
  }

  bind() {
    $("add-protein-open").addEventListener("click", () => this.openAdd());
    $("add-protein-cancel").addEventListener("click", () => this.closeAdd(true));
    $("add-protein").addEventListener("submit", (event) => {
      event.preventDefault();
      this.adding = Promise.all([this.adding, this.add()]).catch(() => null);
    });
    $("add-protein").addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        this.closeAdd(true);
      }
    });
    $("protein-name").addEventListener("change", () => this.rename());
    $("protein-role").addEventListener("change", (event) => {
      const role = event.target.value;
      this.editChosen("PATCH", (protein) => [`/api/proteins/${protein.id}`, { role }], {
        near: $("protein-role").closest("label"),
      });
    });
    $("box-size").addEventListener("submit", (event) => {
      event.preventDefault();
      const size = { width: Number($("box-width").value), height: Number($("box-height").value) };
      this.editChosen("PUT", (protein) => [`/api/proteins/${protein.id}/box-size`, size], {
        fields: ["box-width", "box-height"],
        near: $("box-size"),
      });
    });
    $("box-padding").addEventListener("submit", (event) => {
      event.preventDefault();
      this.applyPadding();
    });
    $("box-padding").addEventListener("input", () => this.renderPaddingNotes());
    for (const { field } of PADDING) {
      $(field).title = $(field).labels[0].title; // the tooltip on the field too
    }
    $("box-padding").addEventListener("keydown", (event) => {
      const direction = PADDING.find(({ field }) => field === event.target.id);
      if (event.key === "Escape" && direction && this.protein) {
        // The stored padding back: always within the field's range.
        event.preventDefault();
        this.fill(direction.field, this.protein.box_padding[direction.key], true);
        this.renderPaddingNotes();
      }
    });
    $("clear-boxes").addEventListener("click", () => this.clear());
    $("remove-protein").addEventListener("click", () => this.remove());
  }

  // --- The list ---

  roleText(protein) {
    if (protein.role === LOADING_CONTROL) {
      return "loading control";
    }
    const used = usedLoadingControls(this.project, protein);
    if (used.length) {
      return `÷ ${names(used)}`;
    }
    return loadingControls(this.project).length ? "target, no loading control chosen" : "target";
  }

  buildList(proteins) {
    const list = $("proteins");
    for (const protein of proteins) {
      const chosen = protein === this.protein;
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.key = protein.id;
      button.className = chosen ? "chosen" : "";
      button.setAttribute("aria-pressed", String(chosen));
      const text = document.createElement("span");
      text.className = "choice-text";
      text.append(span("choice-name", protein.name), span("role", this.roleText(protein)));
      button.append(swatch(colorOf(this.project, protein.id)), text);
      button.addEventListener("click", () => this.handlers.choose(protein.id));
      const item = document.createElement("li");
      item.append(button);
      list.append(item);
    }
    if (!proteins.length) {
      const item = document.createElement("li");
      item.className = "hint";
      item.textContent = this.image ? "No protein on this image yet." : "Import an image first.";
      list.append(item);
    }
  }

  // --- Adding a protein ---

  renderAdd() {
    const marker = this.image !== null && this.image.kind === "visible_marker";
    const open = $("add-protein-open");
    open.disabled = this.image === null || marker;
    const hint = $("add-protein-hint");
    hint.hidden = !marker;
    hint.textContent = marker ? "This is a marker image: proteins are measured on a signal image." : "";
    if (open.disabled && !$("add-protein").hidden) {
      this.closeAdd(false);
    }
  }

  openAdd() {
    $("add-protein").hidden = false;
    $("add-protein-open").hidden = true;
    $("add-protein-open").setAttribute("aria-expanded", "true");
    $("add-protein-name").focus();
  }

  closeAdd(refocus) {
    const form = $("add-protein");
    form.reset();
    form.hidden = true;
    $("add-protein-error").textContent = "";
    const open = $("add-protein-open");
    open.hidden = false;
    open.setAttribute("aria-expanded", "false");
    if (refocus && !open.disabled) {
      open.focus();
    }
  }

  async add() {
    if (!this.image) {
      return;
    }
    const opening = this.opening;
    const body = {
      name: $("add-protein-name").value,
      role: $("add-protein-role").value,
      image_id: this.image.id,
    };
    // After the edits, undos and redos asked for before it, so none takes it back.
    const before = this.queue;
    try {
      await before;
      if (opening !== this.opening) {
        return;
      }
      const answer = await this.handlers.send("POST", "/api/proteins", body);
      if (!answer || opening !== this.opening) {
        return; // another project is shown now: its ids are not this answer's
      }
      this.closeAdd(true);
      this.handlers.choose(answer.protein_id); // the new protein takes the next click
    } catch (error) {
      if (opening === this.opening) {
        const line = $("add-protein-error");
        line.textContent = this.explain(error, null);
        line.scrollIntoView({ block: "nearest" });
      }
    }
  }

  // --- The editor ---

  // Put the server's value in an input, unless the user has typed into it since
  // the panel last filled it; `force` puts it there anyway.
  fill(id, value, force) {
    const input = $(id);
    const text = String(value);
    if (force || !this.filled.has(id) || input.value === this.filled.get(id)) {
      input.value = text;
    }
    this.filled.set(id, text);
  }

  // `forced`: the ids of the inputs an edit just sent, which show the server's
  // value again whatever was typed; the others keep what the user is typing.
  renderEditor(forced = []) {
    const protein = this.protein;
    $("protein-editor").hidden = !protein;
    if (!protein) {
      this.editorFor = null;
      return;
    }
    const fresh = this.editorFor !== protein.id;
    this.editorFor = protein.id;
    if (fresh) {
      this.showError("");
    }
    const force = (id) => fresh || forced.includes(id);
    this.fill("protein-name", protein.name, force("protein-name"));
    $("protein-role").value = protein.role;
    this.renderLoadings(protein);
    const image = this.project.images.find((i) => i.id === protein.image_id);
    $("box-width").max = String(image.width);
    $("box-height").max = String(image.height);
    // The fitted size, which Apply sends: the boxes extend beyond it by the
    // protein's padding, which the server adds (the box size would get it twice).
    this.fill("box-width", protein.fitted_size.width, force("box-width"));
    this.fill("box-height", protein.fitted_size.height, force("box-height"));
    this.renderPadding(protein, force);
    this.renderUndetected(protein);
    this.renderClear();
  }

  // The padding form: whole pixels per side, each at most half the fitted size
  // (so a box at most doubles), or the stored value when a fit has left it
  // above that since: the stored value is always valid, and can be lowered.
  // Disabled until the protein has a box: before that, its fitted size is a
  // placeholder no band set (after Clear boxes, the kept padding shows).
  renderPadding(protein, force) {
    const none = !protein.bands.length;
    for (const { key, field, dimension } of PADDING) {
      const stored = protein.box_padding[key];
      $(field).max = String(Math.max(Math.floor(protein.fitted_size[dimension] / 2), stored));
      $(field).disabled = none;
      this.fill(field, protein.box_padding[key], force(field));
    }
    $("pad-apply").disabled = none;
    $("pad-none").hidden = !none;
    this.renderPaddingNotes();
    const { width, height } = paddedSize(protein);
    $("box-boxes").textContent = `Boxes ${width} × ${height} px`;
    this.renderPartners(protein);
  }

  // Beside each padding field: its share of the fitted size as typed, and a
  // note when the stored padding is more than half of it (a fit has set a
  // smaller fitted size since) or the value typed is more than the field takes.
  renderPaddingNotes() {
    const protein = this.protein;
    if (!protein) {
      return;
    }
    for (const { key, field, dimension } of PADDING) {
      const typed = $(field).value === "" ? NaN : Number($(field).value);
      const fitted = protein.fitted_size[dimension];
      const half = Math.floor(fitted / 2);
      const stored = protein.box_padding[key];
      const share = $(`${field}-share`);
      const shown = Number.isInteger(typed) && typed >= 0;
      share.textContent = shown ? `${Math.round((100 * typed) / fitted)}%` : "";
      share.title = shown ? `of the fitted ${dimension}, ${fitted} px` : "";
      let text = "";
      if (stored > half) {
        text =
          `More than half the fitted ${dimension}: lower it to ${half} px or less to keep` +
          " boxes within twice the fitted size.";
      } else if (typed > half) {
        text = `At most ${half} px: half the fitted ${dimension}, ${fitted} px.`;
      }
      const note = $(`${field}-note`);
      note.textContent = text;
      note.hidden = !text;
    }
  }

  // The chosen protein's partners as the applied results pair them (a target's
  // loading controls, a loading control's targets), each with its padding:
  // fold changes lose least when both are padded alike.
  renderPartners(protein) {
    const sets = this.results ? this.results.sets : [];
    const ids = [];
    for (const series of sets.length ? sets[0].series : []) {
      let other = null;
      if (series.target_id === protein.id) {
        other = series.loading_id;
      } else if (series.loading_id === protein.id) {
        other = series.target_id;
      }
      if (other !== null && !ids.includes(other)) {
        ids.push(other);
      }
    }
    const partners = ids
      .map((id) => this.project.proteins.find((p) => p.id === id))
      .filter(Boolean);
    const line = $("pad-partners");
    line.hidden = !partners.length;
    if (!partners.length) {
      line.textContent = "";
      return;
    }
    const plural = partners.length === 1 ? "" : "s";
    const kind = protein.role === LOADING_CONTROL ? "target" : "loading control";
    const each = partners.map((p) => `${p.name}: ${paddingWords(p.box_padding) || "not padded"}`);
    line.textContent = `Its ${kind}${plural} ${each.join("; ")}.`;
  }

  // Clear boxes: disabled when the chosen protein has nothing to clear, and
  // while its clear has no answer (a second press would clear nothing).
  renderClear() {
    const protein = this.protein;
    $("clear-boxes").disabled =
      !protein ||
      this.clearing.has(protein.id) ||
      (!protein.bands.length && !protein.undetected.length);
  }

  // Show a refusal right after the part of the editor it is about (`near`, by
  // default its buttons), in view, so it is read next to the control it put
  // back; "" hides it.
  showError(text, near = null) {
    const line = $("protein-error");
    near = near || $("protein-actions");
    if (line.previousElementSibling !== near) {
      near.after(line);
    }
    line.textContent = text;
    line.hidden = !text;
    if (text) {
      line.scrollIntoView({ block: "nearest" });
    }
  }

  membraneOf(protein) {
    const image = this.project.images.find((i) => i.id === protein.image_id);
    return image ? image.membrane_id : null;
  }

  renderLoadings(protein) {
    const fieldset = $("loading-controls");
    fieldset.hidden = protein.role === LOADING_CONTROL;
    if (fieldset.hidden) {
      $("loading-choices").replaceChildren();
      return;
    }
    const all = loadingControls(this.project);
    // A choice appears once there is something to choose: two loading controls.
    rebuild($("loading-choices"), () => this.buildLoadings(protein, all.length > 1 ? all : []));
    const hint = $("loading-hint");
    const used = usedLoadingControls(this.project, protein);
    hint.classList.toggle("warning", all.length > 1 && !used.length);
    if (!all.length) {
      hint.textContent =
        "No loading control yet: add one (role: loading control) to normalize this target.";
    } else if (!used.length) {
      hint.textContent = "Choose a loading control; no chart until you do.";
    } else if (!protein.loading_control_ids.length) {
      hint.textContent = `Uses ${names(used)} (the only loading control).`;
    } else {
      hint.textContent = `Uses ${names(used)}.`;
    }
  }

  buildLoadings(protein, controls) {
    // Those on the target's membrane first: they went through the same transfer.
    const membrane = this.membraneOf(protein);
    const same = controls.filter((p) => this.membraneOf(p) === membrane);
    const other = controls.filter((p) => this.membraneOf(p) !== membrane);
    for (const control of [...same, ...other]) {
      const box = document.createElement("input");
      box.type = "checkbox";
      box.dataset.key = control.id;
      box.checked = protein.loading_control_ids.includes(control.id);
      box.addEventListener("change", () => this.tick(protein.id, control.id, box.checked));
      const line = document.createElement("span");
      line.append(control.name);
      if (same.includes(control)) {
        line.append(" ", span("tag", "same membrane"));
      }
      const text = document.createElement("span");
      text.className = "choice-text";
      text.append(line);
      const image = this.project.images.find((i) => i.id === control.image_id);
      if (image) {
        text.append(span("role", `on ${isolate(image.original_name)}`));
      }
      const label = document.createElement("label");
      label.className = "check";
      label.append(box, swatch(colorOf(this.project, control.id)), text);
      $("loading-choices").append(label);
    }
  }

  renderUndetected(protein) {
    $("undetected").hidden = !protein.undetected.length;
    const list = $("undetected-list");
    rebuild(list, () => {
      for (const record of protein.undetected) {
        const lane = this.handlers.laneName(record.lane_index);
        const band = record.band_index ? `, band ${record.band_index + 1}` : "";
        const measured = `SNR ${record.snr.toFixed(1)}, limit ${record.threshold.toFixed(1)}`;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.dataset.key = `${protein.id}:${record.lane_index}:${record.band_index}`;
        remove.textContent = "Remove";
        remove.setAttribute("aria-label", `Remove the n.d. mark in lane ${record.lane_index + 1}`);
        remove.addEventListener("click", () => this.removeUndetected(protein.id, record));
        const item = document.createElement("li");
        item.append(span("", `${lane}${band}: ${measured}`), remove);
        list.append(item);
      }
    });
  }

  // Where the keyboard goes after a removal: the n.d. mark's Remove that took
  // the removed one's place (`position`, if given), else the chosen protein in
  // the list, else "+ Add protein".
  refocus(position = null) {
    const removes =
      position === null || $("undetected").hidden
        ? []
        : [...$("undetected-list").querySelectorAll("button")];
    const target =
      removes[Math.min(position, removes.length - 1)] ||
      $("proteins").querySelector("button.chosen") ||
      $("add-protein-open");
    if (!target.disabled && !target.hidden) {
      target.focus();
    }
  }

  // --- Edits ---

  // Run `task(current)` once the edits queued before it have their answers and
  // `after` (a Promise of edits sent before it outside the queue) has settled,
  // so edits reach the server in the order they were made. It is not run once
  // another project is asked for or shown (invalidateEdits), and `current()`
  // turns false then: nothing about its answer is shown. Each `task` sends its
  // request before it awaits anything, so the request names the opening shown
  // when the edit was made (the app's request()). The app's undo and redo run
  // here too, so they take back what was sent before them, and an open waits
  // for them. Gives the task's result, or null.
  queueEdit(task, { after = null } = {}) {
    const opening = this.opening;
    const current = () => opening === this.opening;
    const run = this.queue.then(async () => {
      await after;
      return current() ? task(current) : null;
    });
    this.queue = run.catch(() => null);
    return run;
  }

  // Run an edit of a protein (the chosen one by default) after the edits before
  // it (queueEdit): `request` gives the path and body from the protein as the
  // last answer stored it. `fields` are the inputs the edit sends, which then
  // show the server's values; a refusal is shown `near` the part of the editor
  // it is about. Gives the answer, or null.
  editChosen(
    method,
    request,
    { proteinId = this.protein && this.protein.id, fields = [], near = null, after = null } = {},
  ) {
    return this.queueEdit(
      async (current) => {
        const protein = this.project.proteins.find((p) => p.id === proteinId);
        if (!protein) {
          return null;
        }
        const [path, body] = request(protein);
        try {
          const answer = await this.handlers.send(method, path, body);
          if (!answer || !current()) {
            return null; // another project is shown now
          }
          this.showError("");
          this.renderEditor(fields);
          return answer;
        } catch (error) {
          if (!current()) {
            return null;
          }
          this.renderEditor(fields);
          const message = this.explain(error, protein);
          if (this.protein && this.protein.id === protein.id) {
            this.showError(message, near);
          } else {
            this.handlers.status(message); // the editor shows another protein by now
          }
          return null;
        }
      },
      { after },
    );
  }

  // A refusal in the user's terms: the proteins it names by their names.
  explain(error, protein) {
    const named = (ids) =>
      ids.map((id) => this.project.proteins.find((p) => p.id === id)).filter(Boolean);
    if (error.code === "loading_control_in_use") {
      const targets = named(error.ids);
      const them = targets.length === 1 ? "it" : "them";
      return (
        `${protein.name} is the loading control of ${names(targets)}:` +
        ` choose another loading control for ${them} first.`
      );
    }
    if (error.code === "duplicate_name") {
      const [other] = named(error.ids);
      if (other) {
        return (
          `There is already a protein named ${other.name}` +
          " (names ignore case and look-alike characters): choose another name."
        );
      }
    }
    return error.message;
  }

  // Apply the padding typed. The values are read now; once the edits before it
  // have their answers, only the directions whose value differs from the
  // padding the last answer stored are sent (editChosen), so a direction
  // another tab changed since stays as it set it, and an Apply pressed before
  // the one before it was answered is compared with what that one set. With
  // none, the body is empty: the server answers it as a no-op ("No change"),
  // and the fields show the stored padding ("05" reads 5 again). Said in the
  // status line once done.
  async applyPadding() {
    const shown = this.protein;
    if (!shown) {
      return;
    }
    const typed = PADDING.map(({ key, field }) => [key, Number($(field).value)]);
    let before = null;
    const answer = await this.editChosen(
      "PUT",
      (protein) => {
        before = this.project;
        const body = {};
        for (const [key, value] of typed) {
          if (value !== protein.box_padding[key]) {
            body[key] = value;
          }
        }
        return [`/api/proteins/${protein.id}/box-padding`, body];
      },
      { proteinId: shown.id, fields: PADDING.map(({ field }) => field), near: $("box-padding") },
    );
    if (answer) {
      this.handlers.status(this.paddingText(answer, shown.id, before));
    }
  }

  // What a padding change did, from its answer: the boxes' size, how the
  // protein's nets moved, the other proteins it re-measured, its boxes the
  // image edge kept from growing evenly, and the boxes they now overlap.
  // `before`: the state shown when it was sent.
  paddingText(answer, proteinId, before) {
    const project = answer.project;
    const protein = project.proteins.find((p) => p.id === proteinId);
    const name = protein ? protein.name : proteinId;
    const { width, height } = paddedSize(answer);
    const fitted = answer.fitted_size;
    const padding = paddingWords(answer.box_padding);
    const boxes = `boxes ${width} × ${height} px`;
    const size = padding
      ? `${boxes} (fitted ${fitted.width} × ${fitted.height}, plus ${padding})`
      : `${boxes}, not padded`;
    const unchanged =
      before !== null &&
      before.open_id === project.open_id &&
      before.revision === project.revision;
    const parts = [unchanged ? `No change: ${name} has ${size}` : `${name}: ${size}`];
    if (answer.net_change) {
      parts.push(`its nets ${netChangeWords(answer.net_change)}`);
    }
    const remeasured = this.handlers.remeasured(answer);
    if (remeasured) {
      parts.push(remeasured);
    }
    const bands = new Map(
      project.proteins.flatMap((p) => p.bands.map((band) => [band.id, { protein: p, band }])),
    );
    const lanes = (ids) =>
      ids
        .map((id) => bands.get(id))
        .filter(Boolean)
        .map(({ band }) => band.lane_index)
        .sort((a, b) => a - b);
    if (answer.edge_shifted.length) {
      const those = answer.edge_shifted.length === 1 ? "that box extends" : "those boxes extend";
      parts.push(
        `the image edge stops the padding of ${lanesPhrase(lanes(answer.edge_shifted))}:` +
          ` ${those} further on the other side`,
      );
    }
    const overlapped = new Map(); // another protein -> its band ids the boxes now overlap
    for (const id of answer.overlapping) {
      const found = bands.get(id);
      if (found) {
        overlapped.set(found.protein, [...(overlapped.get(found.protein) || []), id]);
      }
    }
    for (const [other, ids] of overlapped) {
      parts.push(
        `the boxes now overlap ${other.name}'s in ${lanesPhrase(lanes(ids))}:` +
          " each counts part of the other's band",
      );
    }
    return parts.join(" · ");
  }

  rename() {
    const name = $("protein-name").value;
    if (this.protein && name !== this.protein.name) {
      this.editChosen("PATCH", (protein) => [`/api/proteins/${protein.id}`, { name }], {
        fields: ["protein-name"],
        near: $("protein-name").closest("label"),
      });
    }
  }

  // Tick or untick a loading control; the others keep their order (the series order).
  tick(proteinId, controlId, on) {
    this.editChosen(
      "PATCH",
      (protein) => {
        const kept = protein.loading_control_ids.filter((id) => id !== controlId);
        const ids = on ? [...kept, controlId] : kept;
        return [`/api/proteins/${protein.id}`, { loading_control_ids: ids }];
      },
      { proteinId, near: $("loading-controls") },
    );
  }

  async removeUndetected(proteinId, record) {
    const query = `band_index=${record.band_index}`;
    const path = `/api/proteins/${proteinId}/undetected/${record.lane_index}?${query}`;
    const protein = this.project.proteins.find((p) => p.id === proteinId);
    const position = protein
      ? protein.undetected.findIndex(
          (r) => r.lane_index === record.lane_index && r.band_index === record.band_index,
        )
      : 0;
    const answer = await this.editChosen("DELETE", () => [path, undefined], {
      proteinId,
      near: $("undetected"),
    });
    if (answer) {
      if (focusLost()) {
        this.refocus(Math.max(0, position));
      }
      this.handlers.status(
        `Removed the n.d. mark in lane ${record.lane_index + 1}: that lane is not measured now.`,
      );
    }
  }

  // Remove every box and n.d. mark of the chosen protein, without asking: the
  // status line says what went, with an Undo that brings it back. It waits for
  // the boxes the app is placing or moving, so it clears those too. Pressed
  // twice (a double click), it clears once: the button is disabled at once,
  // and the status line keeps its place, so the second press lands on it.
  async clear() {
    const protein = this.protein;
    if (!protein || this.clearing.has(protein.id)) {
      return;
    }
    this.clearing.add(protein.id);
    this.renderClear();
    const path = `/api/proteins/${protein.id}/boxes`;
    let answer = null;
    try {
      answer = await this.editChosen("DELETE", () => [path, undefined], {
        proteinId: protein.id,
        after: this.handlers.pending(),
      });
    } finally {
      this.clearing.delete(protein.id);
      this.renderClear();
    }
    if (!answer) {
      if (focusLost() && !$("clear-boxes").disabled) {
        $("clear-boxes").focus(); // refused: the keyboard stays on it
      }
      return;
    }
    const parts = [];
    if (answer.removed.length) {
      parts.push(counted(answer.removed.length, "box", "boxes"));
    }
    if (answer.dropped_undetected.length) {
      parts.push(
        counted(answer.dropped_undetected.length, "not-detected mark", "not-detected marks"),
      );
    }
    if (!parts.length) {
      this.handlers.status(`${protein.name} has no boxes or n.d. marks to clear.`);
      return;
    }
    // This Undo takes back the clear only: it goes once the history moves on.
    const step = answer.project.history.undo;
    const undo =
      step && step.action === "clear_boxes"
        ? {
            label: "Undo",
            name: `Undo clearing the boxes of ${protein.name}`,
            seq: step.seq,
            run: () => this.handlers.undo(step.seq),
          }
        : null;
    const button = this.handlers.status(`Cleared ${inWords(parts)}`, undo);
    // Clear boxes is disabled now: the keyboard goes on to that Undo.
    if (button && (focusLost() || document.activeElement === $("clear-boxes"))) {
      button.focus();
    }
  }

  async remove() {
    const protein = this.protein;
    if (!protein) {
      return;
    }
    const parts = [];
    if (protein.bands.length) {
      parts.push(counted(protein.bands.length, "box", "boxes"));
    }
    if (protein.undetected.length) {
      parts.push(counted(protein.undetected.length, "n.d. mark", "n.d. marks"));
    }
    const what = parts.length ? ` and its ${parts.join(" and ")}` : "";
    if (!window.confirm(`Remove ${protein.name}${what}?`)) {
      return;
    }
    const path = `/api/proteins/${protein.id}`;
    const answer = await this.editChosen("DELETE", () => [path, undefined], {
      proteinId: protein.id,
    });
    if (!answer) {
      return;
    }
    // Not on the next protein's "Remove protein": its entry in the list.
    if (focusLost() || document.activeElement === $("remove-protein")) {
      this.refocus();
    }
    // What became of the targets that used it, from the answer's own state.
    const notes = answer.detached_targets
      .map((id) => answer.project.proteins.find((p) => p.id === id))
      .filter(Boolean)
      .map((target) => {
        const used = usedLoadingControls(answer.project, target);
        return used.length
          ? `${target.name} now uses ${names(used)}`
          : `${target.name} is not normalized now`;
      });
    this.handlers.status(`Removed ${protein.name}.${notes.length ? ` ${notes.join("; ")}.` : ""}`);
  }
}
