// SPDX-License-Identifier: Apache-2.0
// The "Proteins on this image" panel: the list with each protein's colour and
// role, the form that adds a protein, and the editor of the chosen one (name,
// role, loading controls, box size, not-detected marks, removal). It stores
// nothing itself: each change goes to the server through the app, and the panel
// is drawn again from the state the server answers.
import { $, focusLost, isolate, rebuild, span, swatch } from "/static/dom.js";

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

function counted(count, one, many) {
  return `${count} ${count === 1 ? one : many}`;
}

export class ProteinPanel {
  // handlers: send(method, path, json) gives a Promise of the server's answer
  // once the app has applied it (null if the answer is about a project opened
  // before the one shown), and rejects with the server's refusal (code,
  // message, ids); choose(proteinId) makes a protein the click target;
  // status(text) shows a line to the user; laneName(index) names a lane.
  constructor(handlers) {
    this.handlers = handlers;
    this.project = null;
    this.image = null;
    this.protein = null;
    this.filled = new Map(); // input id -> the value the panel last put in it
    this.editorFor = null; // the protein the editor was last drawn for
    this.queue = Promise.resolve(); // edits run one at a time, in order
    this.adding = Promise.resolve(); // the adds on their way (they are not queued)
    this.opening = 0; // counts invalidateEdits(): a queued edit never reaches another project
    this.bind();
  }

  // Resolves once every edit made in the panel has its answer, and what the
  // answer shows (the editor, a refusal, the add form closing) is shown. The app
  // waits for it before asking to open another project, so those edits end in
  // the project they were made in.
  settled() {
    return Promise.all([this.queue, this.adding]);
  }

  // No edit made before this, nor the answer to one, is applied from now on:
  // called just before another project is asked for, since its ids repeat those
  // of the one open (prot-3 in each). After settled() nothing is left to drop;
  // this keeps it so.
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

  render(project, image, proteinId) {
    this.project = project;
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
    try {
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
    this.fill("box-width", protein.box_size.width, force("box-width"));
    this.fill("box-height", protein.box_size.height, force("box-height"));
    this.renderUndetected(protein);
  }

  // Show a refusal right after the part of the editor it is about (`near`), in
  // view, so it is read next to the control it put back; "" hides it.
  showError(text, near = null) {
    const line = $("protein-error");
    if (near && line.previousElementSibling !== near) {
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

  // Run an edit of a protein (the chosen one by default) after the edits before
  // it: `request` gives the path and body from the protein as the last answer
  // stored it. `fields` are the inputs the edit sends, which then show the
  // server's values; a refusal is shown `near` the part of the editor it is
  // about. Gives the answer, or null.
  editChosen(
    method,
    request,
    { proteinId = this.protein && this.protein.id, fields = [], near = null } = {},
  ) {
    const opening = this.opening;
    const run = this.queue.then(async () => {
      const protein = this.project.proteins.find((p) => p.id === proteinId);
      if (!protein || opening !== this.opening) {
        return null;
      }
      const [path, body] = request(protein);
      try {
        const answer = await this.handlers.send(method, path, body);
        if (!answer || opening !== this.opening) {
          return null; // another project is shown now
        }
        this.showError("");
        this.renderEditor(fields);
        return answer;
      } catch (error) {
        if (opening !== this.opening) {
          return null;
        }
        this.renderEditor(fields);
        const message = this.explain(error, protein);
        if (this.protein && this.protein.id === protein.id) {
          this.showError(message, near || $("remove-protein"));
        } else {
          this.handlers.status(message); // the editor shows another protein by now
        }
        return null;
      }
    });
    this.queue = run.catch(() => null);
    return run;
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
