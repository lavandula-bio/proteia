// SPDX-License-Identifier: Apache-2.0
// The diagnostic file for a bug report (#138): Diagnostics… writes one zip
// file the user attaches to a report themselves; Proteia sends nothing. Before
// it is written the dialog lists what it will hold, as the server lists it
// (GET /api/diagnostics): the newest session log files, the versions, and the
// open project's project.json and export records, which name its proteins,
// conditions and image files; the project's images only when ticked. Writing
// it (POST /api/diagnostics) names the opening whose files were listed, and the
// digest of those files, so neither a project opened since nor a file made
// since in the same project (in another tab) is ever written unseen: the
// server refuses, and the dialog lists what the file would hold now. Once
// written, it says where, and Show in folder opens that folder in the file
// manager.
import { $, counted, isolate, sentence } from "/static/dom.js";

const PROJECT_CHANGED = "project_changed";
const NO_PROJECT = "no_project";
const FILES_CHANGED = "files_changed";
// The refusals of a write listed otherwise than now: listed again.
const AGAIN = new Set([PROJECT_CHANGED, NO_PROJECT, FILES_CHANGED]);

// A size as the file manager shows it: bytes, KB, MB or GB (of 1024).
export function sizeText(bytes) {
  const units = ["KB", "MB", "GB"];
  if (bytes < 1024) {
    return counted(bytes, "byte", "bytes");
  }
  let value = bytes;
  let unit = "";
  for (unit of units) {
    value /= 1024;
    if (value < 1024) {
      break;
    }
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${unit}`;
}

function total(files) {
  return files.reduce((sum, file) => sum + file.size, 0);
}

function listItem(text) {
  const item = document.createElement("li");
  item.textContent = text;
  return item;
}

export class DiagnosticsDialog {
  // `ask(method, path, json)`: the JSON answer to a request sent as the page
  // sends every one (with the token, and naming the opening it shows);
  // rejects with the server's refusal. `settled()`: settles once the page
  // shows the project open now after a refusal named another (its follow).
  constructor({ ask, settled }) {
    this.ask = ask;
    this.settled = settled;
    this.dialog = $("diagnostics-dialog");
    this.listing = null; // the server's last list: what a file written now holds
    this.writing = false;
    $("diagnostics-images").addEventListener("change", () => this.renderList());
    $("diagnostics-write").addEventListener("click", () => this.write());
    $("diagnostics-reveal").addEventListener("click", () => this.reveal());
    $("diagnostics-close").addEventListener("click", () => {
      if (!this.writing) {
        this.dialog.close();
      }
    });
    // Escape closes it, but not while the file is being written.
    this.dialog.addEventListener("cancel", (event) => {
      if (this.writing) {
        event.preventDefault();
      }
    });
  }

  // Open the dialog with a fresh list; images unticked.
  async show() {
    this.listing = null;
    $("diagnostics-images").checked = false;
    $("diagnostics-content").hidden = false;
    $("diagnostics-result").hidden = true;
    $("diagnostics-write").hidden = false;
    $("diagnostics-close").textContent = "Cancel";
    $("diagnostics-error").textContent = "";
    this.renderList();
    if (!this.dialog.open) {
      this.dialog.showModal();
    }
    await this.load();
  }

  // Ask what a file written now would hold, and list it; `note` says why it
  // is listed again. A refusal as project_changed (the page shows a project no
  // longer open) is followed by the page; then it asks again, once. A list
  // that fails drops the one before: nothing is shown, and Write file stays
  // disabled until a list comes in, so no file is asked for with an opening and
  // digest the dialog no longer shows.
  async load(note = "", again = true) {
    const line = $("diagnostics-state");
    line.className = "hint";
    line.textContent = "Listing what the file will hold…";
    let listing;
    try {
      listing = await this.ask("GET", "/api/diagnostics");
    } catch (error) {
      if (again && error.code === PROJECT_CHANGED) {
        await this.settled();
        return this.load(note, false);
      }
      line.textContent = "";
      $("diagnostics-error").textContent = `Nothing can be listed: ${sentence(error.message)}`;
      this.listing = null;
      $("diagnostics-images").checked = false; // ticked for a list no longer shown
      this.renderList();
      return;
    }
    if (this.listing && this.listing.open_id !== listing.open_id) {
      $("diagnostics-images").checked = false; // ticked for another project
    }
    this.listing = listing;
    line.textContent = note;
    line.className = note ? "note warning" : "hint"; // a refused write stands out
    this.renderList();
  }

  // What the file will hold, as listed, with or without the images as ticked:
  // a summary, then every file with its size, the total, and what is left out.
  renderList() {
    const listing = this.listing;
    const box = $("diagnostics-images");
    const write = $("diagnostics-write");
    write.disabled = this.writing || !listing;
    const summary = $("diagnostics-summary");
    const rows = $("diagnostics-files");
    summary.replaceChildren();
    rows.replaceChildren();
    $("diagnostics-total").textContent = "";
    $("diagnostics-left-out").textContent = "";
    if (!listing) {
      box.disabled = true;
      $("diagnostics-images-label").textContent = "Include the project's images";
      return;
    }
    const images = box.checked && listing.images.length > 0;
    const files = [...listing.files, ...(images ? listing.images : [])];
    const logs = listing.files.filter((file) => file.name.startsWith("logs/"));
    const records = listing.files.filter((file) => file.name.endsWith(".record.json"));
    summary.append(
      listItem(
        logs.length
          ? `Proteia's session log, its newest part (${counted(logs.length, "file", "files")},` +
              ` ${sizeText(total(logs))}): what Proteia did, refused and failed at, naming` +
              " the projects, proteins, conditions and image files it worked on. Access tokens" +
              " and folder paths are hidden in it."
          : "No session log was found.",
      ),
      listItem("The versions of Proteia, Python, the operating system and the image libraries."),
    );
    if (listing.project === null) {
      summary.append(listItem("No project is open, so no project files."));
    } else {
      const parts = listing.files.some((file) => file.name === "project/project.json")
        ? ["project.json"]
        : [];
      if (records.length) {
        parts.push(counted(records.length, "export record", "export records"));
      }
      const unsaved = listing.saved ? "" : " (its latest changes could not be saved)";
      const taken = parts.length ? parts.join(" and ") : "none of its files could be found";
      summary.append(
        listItem(
          `The open project ${isolate(listing.project)}: ${taken}${unsaved}.` +
            " They hold the names of the project, its proteins, conditions, samples and" +
            " image files: check that you can share them.",
        ),
      );
    }
    box.disabled = this.writing || listing.images.length === 0;
    $("diagnostics-images-label").textContent =
      listing.project === null
        ? "Include the project's images (no project is open)"
        : listing.images.length === 0
          ? "Include the project's images (it has none)"
          : `Include the project's images (${counted(listing.images.length, "file", "files")},` +
            ` ${sizeText(total(listing.images))})`;
    for (const file of files) {
      const row = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = file.name;
      const size = document.createElement("td");
      size.className = "size";
      size.textContent = sizeText(file.size);
      row.append(name, size);
      rows.append(row);
    }
    $("diagnostics-total").textContent =
      `${counted(files.length + listing.added.length, "file", "files")},` +
      ` ${sizeText(total(files))} before compression, with ${listing.added.join(" and ")},` +
      " which lists each file with its size and SHA-256.";
    const left = [...listing.left_out, ...(images ? [] : listing.images)];
    if (left.length) {
      const reasons = new Map();
      for (const file of listing.left_out) {
        reasons.set(file.reason, (reasons.get(file.reason) || 0) + 1);
      }
      const said = [...reasons].map(
        ([reason, count]) => `${counted(count, "file", "files")}: ${reason}`,
      );
      if (!images && listing.images.length) {
        said.push(`${counted(listing.images.length, "image", "images")}: not ticked`);
      }
      $("diagnostics-left-out").textContent = `Left out: ${said.join("; ")}.`;
    }
  }

  // Write the file as listed; a project opened since, or a project file made
  // since, makes the server refuse, and the dialog lists what a file would hold
  // now, to be written again.
  async write() {
    if (this.writing || !this.listing) {
      return;
    }
    this.writing = true;
    const listing = this.listing;
    const images = $("diagnostics-images").checked && listing.images.length > 0;
    $("diagnostics-error").textContent = "";
    $("diagnostics-state").textContent = "Writing the file…";
    $("diagnostics-close").disabled = true;
    this.renderList();
    try {
      const written = await this.ask("POST", "/api/diagnostics", {
        images,
        open_id: listing.open_id,
        digest: listing.digest,
      });
      this.showWritten(written);
    } catch (error) {
      $("diagnostics-state").textContent = "";
      if (AGAIN.has(error.code)) {
        this.writing = false;
        await this.settled();
        const why =
          error.code === FILES_CHANGED
            ? "the project's files changed meanwhile (an export made, or an image imported or" +
              " removed)"
            : "another project was opened meanwhile";
        await this.load(
          `Nothing was written: ${why}. The list shows what the file will hold now.`,
        );
      } else {
        $("diagnostics-error").textContent = `Not written: ${sentence(error.message)}`;
      }
    } finally {
      this.writing = false;
      $("diagnostics-close").disabled = false;
      if (!$("diagnostics-content").hidden) {
        this.renderList();
      }
    }
  }

  // Where the file was written, with Show in folder; the keyboard goes to it.
  showWritten(written) {
    $("diagnostics-content").hidden = true;
    $("diagnostics-write").hidden = true;
    $("diagnostics-result").hidden = false;
    $("diagnostics-close").textContent = "Close";
    const line = $("diagnostics-written");
    line.textContent =
      `Written: ${isolate(written.name)} (${sizeText(written.size)},` +
      ` ${counted(written.files, "file", "files")}), in Proteia's diagnostics folder.` +
      ` Attach it to your bug report. Proteia keeps the ${this.listing.kept} newest there.`;
    line.title = written.path; // the whole path, which the dialog does not show
    $("diagnostics-reveal").focus();
  }

  // Open the diagnostics folder in the system file manager.
  async reveal() {
    $("diagnostics-error").textContent = "";
    try {
      await this.ask("POST", "/api/diagnostics/reveal");
    } catch (error) {
      $("diagnostics-error").textContent = sentence(error.message);
    }
  }
}
