// SPDX-License-Identifier: Apache-2.0
// The Import dialog (#57): images a launch handed to Proteia (`proteia a.tif
// b.tif`, or "Open with Proteia"), which wait on the server until the page
// imports them into a new project or discards them. Each image gets a row: how
// it is imported (the page import's kinds), its membrane (a new one, or the
// one an earlier image of the list goes on), and whether its bands are dark or
// light. That last starts unchosen, as in the page import: the model has no
// silent default, so Import waits until every row has it. "Set all" fills the
// rows listed when it is pressed, never one that arrives later; a row keeps
// what was chosen in it while the list grows. The page (app.js) finds the
// hand-offs, sends the import or discard, and says what came of it.
import { $, counted, isolate } from "/static/dom.js";

const SIZES = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });

// A file size as a file manager shows one: KB below a megabyte, else MB.
export function sizeText(bytes) {
  return bytes < 1024 * 1024
    ? `${SIZES.format(Math.max(1, Math.round(bytes / 1024)))} KB`
    : `${SIZES.format(bytes / (1024 * 1024))} MB`;
}

// Files or arguments not taken, and why, in words: "a.bmp (not an image type
// Proteia imports); b.tif (no such file or folder); and 3 more".
export function refusedText(entries, more = 0) {
  const parts = entries.map((entry) => `${isolate(entry.name)} (${entry.message})`);
  if (more) {
    parts.push(`and ${more} more`);
  }
  return parts.join("; ");
}

// The select of a row that says whether its image's bands are dark or light.
function polarityOf(row) {
  return row.querySelector(".handoff-polarity");
}

// How many "Not opened" entries `handoff` has, those listed and those counted.
function refusedCount(handoff) {
  return handoff.refused.length + handoff.more_refused;
}

export class ImportDialog {
  // `handlers`: accept() and discard(), pressed; closed(handoff), the dialog
  // closed with Esc or Later (the images keep waiting on the server).
  constructor(handlers) {
    this.handlers = handlers;
    this.handoff = null; // the hand-off shown, as last listed, or null
    // The hand-off the rows are of: the one shown, or the last one closed with
    // Esc or Later, whose choices are kept should it be shown again.
    this.kept = null;
    this.rows = new Map(); // file id -> its row (a list item)
    this.nameEdited = false; // the name typed: sent as typed, else the server names it
    this.busy = false; // an import or discard of it is awaiting its answer
    this.blocked = null; // why Import cannot be pressed now (an export runs), or null
    // What an import closes, as the dialog says (renderCloses): {name, open_id}
    // of the project open, or null for none.
    this.closes = null;
    const dialog = $("handoff-dialog");
    $("handoff-form").addEventListener("submit", (event) => {
      event.preventDefault();
      if (this.ready()) {
        this.handlers.accept();
      }
    });
    $("handoff-discard").addEventListener("click", () => this.handlers.discard());
    $("handoff-later").addEventListener("click", () => dialog.close());
    $("handoff-name").addEventListener("input", () => {
      this.nameEdited = true;
      this.sayName("");
    });
    $("handoff-files").addEventListener("change", () => this.renderActions());
    $("handoff-set-all-polarity").addEventListener("change", () => {
      $("handoff-set-all-apply").disabled = !$("handoff-set-all-polarity").value;
    });
    $("handoff-set-all-apply").addEventListener("click", () => this.setAll());
    // Esc closes it, the images left waiting; not while its import or
    // discard is awaited, which says how it went here. The key itself is
    // stopped then: the browser lets a page stop one cancel only (a second Esc
    // with no click between closes the dialog however its cancel is answered).
    document.addEventListener("keydown", (event) => {
      if (this.busy && dialog.open && event.key === "Escape") {
        event.preventDefault();
      }
    });
    dialog.addEventListener("cancel", (event) => {
      if (this.busy) {
        event.preventDefault();
      }
    });
    // Closed by Esc or Later: the page sets it aside, and its rows are kept.
    // Closed by hide(), it shows none by now; opened again meanwhile (the
    // close event comes later), it was not closed. Closed all the same while
    // its import or discard is awaited, it comes back up to say how it went.
    dialog.addEventListener("close", () => {
      const handoff = this.handoff;
      if (!handoff || dialog.open) {
        return;
      }
      if (this.busy) {
        dialog.showModal();
        return;
      }
      this.handoff = null;
      this.handlers.closed(handoff);
    });
  }

  // Show `handoff` ({id, files, suggested_name, refused, more_refused,
  // more_may_arrive}, as listed); `closes`: what an import closes
  // (renderCloses). The hand-off last closed with Esc or Later comes back with
  // what was chosen in it; another starts afresh.
  show(handoff, closes) {
    if (this.kept !== handoff.id) {
      this.reset();
    }
    for (const id of ["handoff-news", "handoff-error"]) {
      $(id).textContent = "";
    }
    this.kept = handoff.id;
    this.arrived(this.fill(handoff));
    this.renderCloses(closes);
    const dialog = $("handoff-dialog");
    if (!dialog.open) {
      dialog.showModal();
    }
  }

  // The hand-off shown as listed now (more files, or more arguments not
  // opened, may have joined it): each row keeps what was chosen in it; a new
  // row starts with its bands unchosen, so Import waits for it again, and the
  // dialog says so. Gives what joined: {images, refused}, how many rows and
  // "Not opened" entries are new.
  refresh(handoff) {
    const before = refusedCount(this.handoff);
    const images = this.fill(handoff);
    this.arrived(images);
    return { images, refused: refusedCount(handoff) - before };
  }

  // Say that `added` rows joined those shown (none: nothing to say).
  arrived(added) {
    if (!added) {
      return;
    }
    $("handoff-set-all-polarity").value = ""; // it set the rows listed then, not these
    $("handoff-set-all-apply").disabled = true;
    const them = added === 1 ? "its" : "their";
    $("handoff-news").textContent =
      `${counted(added, "more image", "more images")} arrived:` +
      ` choose whether ${them} bands are dark or light.`;
  }

  // Close it, and forget the hand-off shown (imported, discarded, or gone);
  // the page says why.
  hide() {
    this.handoff = null;
    this.reset();
    const dialog = $("handoff-dialog");
    if (dialog.open) {
      dialog.close();
    }
  }

  reset() {
    this.kept = null;
    this.rows.clear();
    $("handoff-files").replaceChildren();
    this.nameEdited = false;
    $("handoff-name").value = "";
    $("handoff-set-all-polarity").value = "";
    $("handoff-set-all-apply").disabled = true;
    for (const id of ["handoff-news", "handoff-error"]) {
      $(id).textContent = "";
    }
    this.sayName("");
    this.setBusy(null);
  }

  // Draw `handoff`'s rows, reusing those of files already shown, in its
  // order; gives how many rows are new to a dialog that showed rows before.
  fill(handoff) {
    const shown = this.rows.size;
    this.handoff = handoff;
    const list = $("handoff-files");
    let added = 0;
    handoff.files.forEach((file, index) => {
      let row = this.rows.get(file.file_id);
      if (!row) {
        row = this.newRow(file);
        this.rows.set(file.file_id, row);
        added += 1;
      }
      if (list.children[index] !== row) {
        list.insertBefore(row, list.children[index] || null);
      }
    });
    const listed = new Set(handoff.files.map((file) => file.file_id));
    for (const [id, row] of [...this.rows]) {
      if (!listed.has(id)) {
        row.remove();
        this.rows.delete(id);
      }
    }
    this.renderMembranes();
    const count = handoff.files.length;
    const images = counted(count, "image", "images");
    $("handoff-title").textContent = `Import ${images} into a new project`;
    $("handoff-set-all").hidden = count < 2;
    if (!this.nameEdited) {
      $("handoff-name").value = handoff.suggested_name || "";
    }
    $("handoff-arriving").hidden = !handoff.more_may_arrive;
    this.renderRefused(handoff);
    this.renderActions();
    return shown ? added : 0;
  }

  // A row for `file`: its name and size, and its three choices, the bands' unchosen.
  newRow(file) {
    const row = $("handoff-row").content.firstElementChild.cloneNode(true);
    row.dataset.fileId = file.file_id;
    row.querySelector(".handoff-file-name").textContent = file.name;
    row.querySelector(".handoff-file-size").textContent = sizeText(file.size);
    return row;
  }

  // Each row's Membrane: a new one, or the one an earlier image goes on
  // ("Same as" it; a name listed twice is told apart by its place). A choice
  // stays while that image is listed before the row.
  renderMembranes() {
    const files = this.handoff.files;
    files.forEach((file, index) => {
      const select = this.rows.get(file.file_id).querySelector(".handoff-membrane");
      const kept = select.value;
      const options = [new Option("New membrane", "new")];
      for (const earlier of files.slice(0, index)) {
        const twice = files.filter((other) => other.name === earlier.name).length > 1;
        const place = twice ? ` (image ${files.indexOf(earlier) + 1})` : "";
        options.push(new Option(`Same as ${isolate(earlier.name)}${place}`, earlier.file_id));
      }
      select.replaceChildren(...options);
      select.value = options.some((option) => option.value === kept) ? kept : "new";
    });
  }

  renderRefused(handoff) {
    const list = $("handoff-refused-list");
    list.replaceChildren();
    for (const entry of handoff.refused) {
      const item = document.createElement("li");
      item.className = "warning";
      item.textContent = `${isolate(entry.name)}: ${entry.message}`;
      list.append(item);
    }
    if (handoff.more_refused) {
      const item = document.createElement("li");
      item.className = "warning";
      item.textContent = `and ${handoff.more_refused} more`;
      list.append(item);
    }
    $("handoff-refused").hidden = !list.children.length;
  }

  // What an import does to the project open (`opening`: its {name, open_id},
  // or null for none), which the import's request names (app.js), so the
  // server refuses it once another project is open.
  renderCloses(opening) {
    this.closes = opening;
    const line = $("handoff-closes");
    line.textContent = opening
      ? `Importing saves and closes ${isolate(opening.name)}, and clears its undo history.`
      : "";
    line.hidden = !opening;
  }

  // "Set all": the rows listed now take the polarity chosen there.
  setAll() {
    const value = $("handoff-set-all-polarity").value;
    if (!value || this.busy) {
      return;
    }
    for (const row of this.rows.values()) {
      polarityOf(row).value = value;
    }
    const words = $("handoff-set-all-polarity").selectedOptions[0].textContent;
    const images = counted(this.rows.size, "image", "images");
    $("handoff-news").textContent = `Bands in ${images}: ${words}.`;
    this.renderActions();
  }

  // How many rows have no polarity chosen yet.
  missing() {
    return [...this.rows.values()].filter((row) => !polarityOf(row).value).length;
  }

  ready() {
    return this.handoff !== null && !this.busy && this.blocked === null && this.missing() === 0;
  }

  // Import waits for every row's polarity, and for `blocked` to clear (why it
  // cannot be pressed now: an export being written, or null).
  block(blocked) {
    this.blocked = blocked;
    this.renderActions();
  }

  renderActions() {
    const missing = this.missing();
    $("handoff-import").disabled = !this.ready();
    $("handoff-discard").disabled = !this.handoff || this.busy;
    $("handoff-later").disabled = this.busy;
    $("handoff-needs").textContent =
      this.blocked ||
      (missing
        ? `Choose whether the bands are dark or light in ${counted(missing, "image", "images")}.`
        : "");
  }

  // The import or discard awaiting its answer (`text` says which), or none
  // (null): the choices and buttons wait meanwhile.
  setBusy(text) {
    this.busy = text !== null;
    $("handoff-fields").disabled = this.busy;
    $("handoff-dialog").setAttribute("aria-busy", String(this.busy));
    const line = $("handoff-state");
    line.textContent = text || "";
    line.hidden = !text;
    if (text) {
      $("handoff-error").textContent = "";
    }
    this.renderActions();
  }

  // The dialog's own line for what went wrong ("" empties it).
  say(text) {
    $("handoff-error").textContent = text;
  }

  // What is wrong with the name typed, next to it ("" for nothing).
  sayName(text) {
    const line = $("handoff-name-error");
    line.textContent = text;
    line.hidden = !text;
    $("handoff-name").setAttribute("aria-invalid", String(Boolean(text)));
  }

  // What the import sends: the name typed, or null (the server names the
  // project after the first image, numbered if taken); and each image, in the
  // hand-off's order, with its kind, polarity and membrane ("new", or the
  // place of the earlier image whose membrane it joins).
  choices() {
    const ids = this.handoff.files.map((file) => file.file_id);
    const files = ids.map((id) => {
      const row = this.rows.get(id);
      const membrane = row.querySelector(".handoff-membrane").value;
      return {
        file_id: id,
        kind: row.querySelector(".handoff-kind").value,
        polarity: polarityOf(row).value,
        membrane: membrane === "new" ? "new" : ids.indexOf(membrane),
      };
    });
    const typed = $("handoff-name").value;
    return { name: this.nameEdited && typed.trim() ? typed : null, files };
  }

  // What a discard sends: the files and the count of refused entries shown,
  // so one that grew since is refused, not dropped unseen.
  shownParts() {
    const handoff = this.handoff;
    return {
      files: handoff.files.map((file) => file.file_id),
      refused: refusedCount(handoff),
    };
  }

  // Where the keyboard goes back to once an answer is in: the first image
  // whose bands are unchosen, else Import, else the name.
  focusBack() {
    const unchosen = [...this.rows.values()].find((row) => !polarityOf(row).value);
    const target = unchosen
      ? polarityOf(unchosen)
      : !$("handoff-import").disabled
        ? $("handoff-import")
        : $("handoff-name");
    target.focus();
  }
}
