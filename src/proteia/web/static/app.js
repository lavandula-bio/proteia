// SPDX-License-Identifier: Apache-2.0
// The page: keeps this launch's access token, talks to the local server, and
// shows the open project's images, proteins, boxes and checks. Every edit goes
// to the server, which answers with the stored project and its results; the
// page only draws what it is given.
import { $, isolate, rebuild, span } from "/static/dom.js";
import { colorOf, ProteinPanel } from "/static/proteins.js";
import { ImageView, MISSING_COLOR } from "/static/view.js";

const TOKEN_KEY = "proteia-token";
const NEEDS_LAUNCH =
  "This page needs the link Proteia opens when it starts. Start Proteia again to open it.";

// The launcher puts the token in the URL fragment, which the browser never sends
// to a server. Keep it for this tab only and remove it from the address bar.
function takeToken() {
  const match = /^#token=([A-Za-z0-9_-]{32,128})$/.exec(window.location.hash);
  if (match) {
    sessionStorage.setItem(TOKEN_KEY, match[1]);
    history.replaceState(null, "", window.location.pathname + window.location.search);
  }
  return sessionStorage.getItem(TOKEN_KEY);
}

const token = takeToken();

// --- Talking to the server ---

class ApiError extends Error {
  constructor(status, body) {
    super(body && body.message ? body.message : `the server answered ${status}`);
    this.status = status;
    this.code = body && body.code;
    this.ids = (body && body.ids) || [];
  }
}

// Every request names the token in a header; nothing relies on cookies.
async function request(method, path, { json, body, contentType } = {}) {
  const headers = new Headers({ Authorization: `Bearer ${token}` });
  if (json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(json);
  } else if (contentType) {
    headers.set("Content-Type", contentType);
  }
  const response = await fetch(path, { method, headers, body, cache: "no-store" });
  if (response.status === 401) {
    showStatus(NEEDS_LAUNCH);
    throw new ApiError(401, null);
  }
  if (!response.ok) {
    let detail = null;
    try {
      detail = await response.json();
    } catch (error) {
      // not JSON: keep the status
    }
    throw new ApiError(response.status, detail);
  }
  return response;
}

async function call(method, path, json) {
  const response = await request(method, path, { json });
  return response.status === 204 ? null : response.json();
}

// --- Page state ---

const state = {
  project: null, // the last project state the server sent
  results: null, // the results computed from that same state
  answered: null, // {open_id, revision} of that state
  imageId: null,
  proteinId: null, // the protein a click on the image places a box of
  boxId: null,
  bitmaps: new Map(), // image id -> Promise of its preview's ImageBitmap
  shownImageId: null, // the image the view shows (null while a preview loads)
};

function forgetBitmap(imageId) {
  const pending = state.bitmaps.get(imageId);
  state.bitmaps.delete(imageId);
  if (pending) {
    pending.then((bitmap) => bitmap.close()).catch(() => {});
  }
}

function showStatus(text) {
  const line = $("status");
  line.textContent = text;
  line.hidden = !text;
}

function report(error) {
  if (error instanceof ApiError && error.status === 401) {
    return;
  }
  showStatus(error.message);
}

// Answers can arrive out of order: one is shown only if it is not older than
// the one shown, by (open_id, revision). The open id counts the server's
// creates and opens, so an answer about an earlier opening is older whatever
// its revision, and the answer of the latest open is newer than all before it.
function isCurrent(project) {
  const shown = state.answered;
  return (
    shown === null ||
    project.open_id > shown.open_id ||
    (project.open_id === shown.open_id && project.revision >= shown.revision)
  );
}

// Whether an answer is about the project opening the page shows (at any revision).
function sameOpening(answer) {
  return state.answered !== null && answer.project.open_id === state.answered.open_id;
}

// Show an answer's state and results; `choose` sets page state along with it
// (such as the image or protein it made). False if the answer was older. The
// choice still applies to an older answer about the opening shown: what it
// made is in the newer state too, or select() drops it.
function applyAnswer(answer, { choose = {} } = {}) {
  const current = isCurrent(answer.project);
  if (current) {
    state.answered = { open_id: answer.project.open_id, revision: answer.project.revision };
    state.project = answer.project;
    state.results = answer.results;
  } else if (!sameOpening(answer) || !Object.keys(choose).length) {
    return false;
  }
  Object.assign(state, choose);
  select();
  render();
  return current;
}

// The open id of the project shown. What an edit does after its answer (a
// status line, a lane question) is dropped once another project is shown.
function shownOpening() {
  return state.answered && state.answered.open_id;
}

// Send an edit; its answer replaces the project state. Gives the answer, or
// null if it is about a project opened before the one shown now.
async function send(method, path, json) {
  const answer = await call(method, path, json);
  applyAnswer(answer);
  if (!sameOpening(answer)) {
    return null;
  }
  showStatus("");
  return answer;
}

// Send an edit and show a refusal in the status line (unless another project
// is shown by then).
async function edit(method, path, json) {
  const opened = shownOpening();
  try {
    return await send(method, path, json);
  } catch (error) {
    if (opened === shownOpening()) {
      report(error);
    }
    throw error;
  }
}

const view = new ImageView($("view"), {
  place: (x, y, options) =>
    placeBox(x, y, options, options.laneIndex, options.proteinId || state.proteinId),
  move: (boxId, rect) => edit("PUT", `/api/boxes/${boxId}`, { rect }).catch(() => {}),
  select: (boxId) => {
    state.boxId = boxId;
    render();
  },
});

const proteinPanel = new ProteinPanel({
  send,
  choose: (proteinId) => {
    state.proteinId = proteinId;
    select();
    render();
  },
  status: showStatus,
  laneName: (index) => laneName(state.project, index),
});

// --- Projects ---

async function showProjects() {
  const dialog = $("projects-dialog");
  $("projects-error").textContent = "";
  const listing = await call("GET", "/api/projects");
  $("projects-root").textContent = `Projects are saved in ${listing.root}`;
  const list = $("project-list");
  list.replaceChildren();
  for (const entry of listing.projects) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.name;
    button.addEventListener("click", () => openProject("/api/projects/open", entry.name));
    const item = document.createElement("li");
    item.append(button);
    list.append(item);
  }
  $("projects-close").hidden = !state.project;
  if (!dialog.open) {
    dialog.showModal();
  }
}

// One create or open at a time: while it runs the dialog stays open and takes
// no other choice, so no two opens race and no edit is made from the page
// until the server's newly open project is shown.
let opening = null; // the name being opened, or null

function setOpening(name) {
  opening = name;
  const dialog = $("projects-dialog");
  dialog.setAttribute("aria-busy", String(name !== null));
  const line = $("projects-state");
  line.textContent = name === null ? "" : `Opening ${name}…`;
  line.hidden = name === null;
}

async function openProject(path, name) {
  if (opening !== null) {
    return;
  }
  setOpening(name);
  // Before asking: no queued panel edit may reach the project opened next.
  proteinPanel.reset();
  $("lane-picker").hidden = true;
  $("projects-error").textContent = "";
  try {
    const answer = await call("POST", path, { name });
    if (!isCurrent(answer.project)) {
      $("projects-error").textContent = "Another project was opened meanwhile.";
      return;
    }
    // Image ids repeat across projects (img-1 in each): drop everything shown.
    view.setImage(null, 0, 0);
    view.setOverlay([], [], null);
    state.shownImageId = null;
    for (const id of [...state.bitmaps.keys()]) {
      forgetBitmap(id);
    }
    applyAnswer(answer, { choose: { imageId: null, proteinId: null, boxId: null } });
    $("projects-dialog").close();
    showStatus("");
  } catch (error) {
    $("projects-error").textContent = error.message;
  } finally {
    setOpening(null);
  }
}

$("new-project").addEventListener("submit", (event) => {
  event.preventDefault();
  openProject("/api/projects", $("new-project-name").value);
});
$("projects-close").addEventListener("click", () => {
  if (opening === null) {
    $("projects-dialog").close();
  }
});
$("projects-dialog").addEventListener("cancel", (event) => {
  if (!state.project || opening !== null) {
    event.preventDefault(); // a project must be open to work, and the open must finish
  }
});
$("switch-project").addEventListener("click", () => showProjects().catch(report));
$("reveal").addEventListener("click", () => call("POST", "/api/project/reveal").catch(report));

// --- Applying the server's state ---

// Keep the chosen image, protein and box while they exist; otherwise the first.
function select() {
  const project = state.project;
  const images = project.images;
  if (!images.some((image) => image.id === state.imageId)) {
    state.imageId = images.length ? images[0].id : null;
  }
  const proteins = project.proteins.filter((p) => p.image_id === state.imageId);
  if (!proteins.some((p) => p.id === state.proteinId)) {
    state.proteinId = proteins.length ? proteins[0].id : null;
  }
  const boxes = project.proteins.flatMap((p) => p.bands);
  if (!boxes.some((band) => band.id === state.boxId)) {
    state.boxId = null;
  }
  for (const id of [...state.bitmaps.keys()]) {
    if (!images.some((image) => image.id === id)) {
      forgetBitmap(id);
    }
  }
}

function laneName(project, index) {
  const lane = project.lanes[index];
  const label = lane ? ` (${lane.condition}${lane.sample ? `, ${lane.sample}` : ""})` : "";
  return `Lane ${index + 1}${label}`;
}

function render() {
  $("lane-picker").hidden = true; // its question was about the state before
  const project = state.project;
  $("workspace").hidden = !project;
  $("switch-project").hidden = false;
  $("reveal").hidden = !project;
  if (!project) {
    return;
  }
  $("project-name").textContent = project.name;
  $("save-state").textContent = project.save_error
    ? `Not saved: ${project.save_error}`
    : project.saved
      ? "Saved"
      : "Saving…";
  const image = project.images.find((i) => i.id === state.imageId) || null;
  renderImages(project, image);
  proteinPanel.render(project, image, state.proteinId);
  renderBox(project);
  renderNotices(project);
  renderHint(project, image);
  renderView(project);
}

function renderImages(project, image) {
  rebuild($("images"), () => {
    for (const each of project.images) {
      const chosen = each === image;
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.key = each.id;
      button.className = chosen ? "chosen" : "";
      button.setAttribute("aria-pressed", String(chosen));
      button.textContent = each.original_name;
      button.addEventListener("click", () => {
        state.imageId = each.id;
        state.boxId = null;
        select();
        render();
      });
      const item = document.createElement("li");
      item.append(button);
      $("images").append(item);
    }
  });
  $("image-controls").hidden = !image;
  const warnings = $("image-warnings");
  warnings.replaceChildren();
  if (image) {
    $("polarity").value = image.polarity;
    for (const warning of image.warnings) {
      const item = document.createElement("li");
      item.className = "warning";
      item.textContent = warning.message;
      warnings.append(item);
    }
  }
  renderMembranes(project);
}

let membranesOpening = null; // the open id the Membrane choice was made in

// The import's membrane: a new one, or one an earlier image is on. A choice
// made in another project never carries over (membrane ids repeat across
// projects, mem-2 in each).
function renderMembranes(project) {
  const select = $("import-membrane");
  const kept = membranesOpening === project.open_id ? select.value : "";
  membranesOpening = project.open_id;
  const membranes = new Map(); // membrane id -> its images
  for (const image of project.images) {
    if (!membranes.has(image.membrane_id)) {
      membranes.set(image.membrane_id, []);
    }
    membranes.get(image.membrane_id).push(image);
  }
  const firstNames = [...membranes.values()].map((images) => images[0].original_name);
  const options = [new Option("New membrane", "")];
  for (const [id, images] of membranes) {
    const first = images[0];
    const notes = [];
    if (firstNames.filter((name) => name === first.original_name).length > 1) {
      // Two membranes named by the same file name: which image, by its place in the list.
      notes.push(`image ${project.images.indexOf(first) + 1}`);
    }
    if (images.length > 1) {
      notes.push(`and ${images.length - 1} more`);
    }
    const more = notes.length ? ` (${notes.join(", ")})` : "";
    options.push(new Option(`Same as ${isolate(first.original_name)}${more}`, id));
  }
  select.replaceChildren(...options);
  select.value = membranes.has(kept) ? kept : "";
}

function findBox(project, boxId) {
  for (const protein of project.proteins) {
    const band = protein.bands.find((b) => b.id === boxId);
    if (band) {
      return { protein, band };
    }
  }
  return null;
}

const PLACED_BY = {
  click: "Click, grown from the band",
  manual: "Shift+click, fixed size",
  row_box: "Row box detection",
  mw_guided: "Molecular-weight guide",
};
const WHOLE = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });
const SMALL = new Intl.NumberFormat("en", { maximumSignificantDigits: 4 });

// The box's net from the results: the lane whose band is this box.
function netText(protein, band) {
  const column = state.results.proteins.find((c) => c.protein_id === protein.id);
  const lane = column ? column.band_ids.indexOf(band.id) : -1;
  if (lane < 0) {
    return band.band_index > 0 ? "Not quantified (an extra band)" : "—";
  }
  const net = column.nets[lane];
  if (net === null) {
    return "—";
  }
  return Math.abs(net) >= 100 ? WHOLE.format(net) : SMALL.format(net);
}

function renderBox(project) {
  const found = state.boxId ? findBox(project, state.boxId) : null;
  $("box-panel").hidden = !found;
  if (!found) {
    return;
  }
  const { protein, band } = found;
  $("box-summary").textContent = `${protein.name}, ${laneName(project, band.lane_index)}`;
  $("box-clipped").textContent =
    band.clipped === true
      ? "Yes: pixels at the detector limit, so the net is an under-estimate"
      : band.clipped === null
        ? "Not checked"
        : "No";
  $("box-net").textContent = netText(protein, band);
  const edited = band.manually_edited ? "; moved or re-laned by hand since" : "";
  $("box-source").textContent = `${PLACED_BY[band.source] || band.source}${edited}`;
  const select = $("box-lane");
  select.replaceChildren();
  for (const lane of project.lanes) {
    select.append(new Option(laneName(project, lane.index), String(lane.index)));
  }
  select.value = String(band.lane_index);
}

// A core notice as a sentence: capitalized, with a full stop.
function sentence(text) {
  const capital = text.charAt(0).toUpperCase() + text.slice(1);
  return /[.!?]$/.test(capital) ? capital : `${capital}.`;
}

function renderNotices(project) {
  const warnings = [];
  const infos = [];
  for (const set of state.results.sets) {
    const prefix = set.id === "all_lanes" ? "All lanes: " : "";
    for (const notice of set.notices) {
      const line = [`${prefix}${sentence(notice.message)}`, notice.level];
      (notice.level === "warning" ? warnings : infos).push(line);
    }
  }
  const missing = [];
  for (const protein of project.proteins.filter((p) => p.image_id === state.imageId)) {
    if (protein.missing_lanes.length && protein.bands.length) {
      const lanes = protein.missing_lanes.map((m) => m.lane_index + 1);
      const which = `lane${lanes.length > 1 ? "s" : ""} ${lanes.join(", ")}`;
      missing.push([`${protein.name}: no box in ${which}.`, "missing"]);
    }
  }
  const list = $("notices");
  list.replaceChildren();
  const lines = [...warnings, ...missing, ...infos];
  if (!lines.length) {
    lines.push(["No problems found.", "ok"]);
  }
  for (const [text, kind] of lines) {
    const item = document.createElement("li");
    item.className = kind;
    item.textContent = text;
    list.append(item);
  }
}

function renderHint(project, image) {
  const hint = $("view-hint");
  hint.hidden = !image;
  if (!image) {
    return;
  }
  const protein = project.proteins.find((p) => p.id === state.proteinId);
  if (image.kind === "visible_marker") {
    hint.textContent = "A marker image: boxes are placed on signal images.";
  } else if (!protein) {
    hint.textContent = "Add a protein to place boxes on this image.";
  } else {
    hint.replaceChildren(
      "Boxes go to ",
      span("hint-target", protein.name),
      " · Click a band: one box · Shift+click: fixed box · Drag: pan · Wheel: zoom",
    );
  }
}

// One fetch per preview, shared by every render that waits for it.
function bitmapOf(imageId) {
  if (!state.bitmaps.has(imageId)) {
    const pending = request("GET", `/api/images/${imageId}/preview`)
      .then((response) => response.blob())
      .then((blob) => createImageBitmap(blob));
    pending.catch(() => {
      if (state.bitmaps.get(imageId) === pending) {
        state.bitmaps.delete(imageId); // the next render tries again
      }
    });
    state.bitmaps.set(imageId, pending);
  }
  return state.bitmaps.get(imageId);
}

let renderGeneration = 0;

function renderView(project) {
  const generation = ++renderGeneration;
  const image = project.images.find((i) => i.id === state.imageId);
  if (!image || state.shownImageId !== image.id) {
    // Never show one image while edits target another: blank until it loads.
    state.shownImageId = null;
    view.setImage(null, 0, 0);
    view.setOverlay([], [], null);
  }
  if (!image) {
    return;
  }
  const boxes = [];
  const ghosts = [];
  const marks = [];
  for (const protein of project.proteins.filter((p) => p.image_id === image.id)) {
    const color = colorOf(project, protein.id);
    for (const band of protein.bands) {
      boxes.push({
        id: band.id,
        rect: band.rect,
        color,
        clipped: band.clipped === true,
        label: `${band.lane_index + 1}${band.clipped === true ? " over-exposed" : ""}`,
      });
    }
    const { width, height } = protein.box_size;
    for (const missing of protein.missing_lanes) {
      if (missing.x === null || missing.y === null) {
        continue;
      }
      const x0 = Math.round(missing.x - width / 2);
      const y0 = Math.round(missing.y - height / 2);
      ghosts.push({
        rect: [x0, y0, x0 + width, y0 + height],
        color: MISSING_COLOR,
        label: `${missing.lane_index + 1}: no box`,
        proteinId: protein.id,
        laneIndex: missing.lane_index,
      });
    }
    for (const record of protein.undetected) {
      const band = record.band_index ? ` band ${record.band_index + 1}` : "";
      marks.push({
        rect: record.region,
        color,
        label: `${record.lane_index + 1}${band}: n.d.`,
        proteinId: protein.id,
        // A click there places a first-band box in that lane.
        laneIndex: record.band_index === 0 ? record.lane_index : null,
      });
    }
  }
  bitmapOf(image.id)
    .then((bitmap) => {
      if (generation !== renderGeneration) {
        return; // a later render owns the view
      }
      view.setImage(bitmap, image.width, image.height);
      state.shownImageId = image.id;
      view.setOverlay(boxes, ghosts, state.boxId, marks);
    })
    .catch(report);
}

// --- Edits ---

// When the lane cannot be proposed from the position, the user chooses it.
const LANE_QUESTIONS = {
  lane_required: "Proteia works out lanes from boxes in two lanes of this image.",
  lane_occupied: "The lane at this position already has a box of this protein.",
  lane_out_of_range: "This position lies outside the declared lanes.",
};

// A box of `proteinId` at the image point: grown from the band there, or of the
// protein's size (`options.grow`); in `laneIndex`, or in the lane the server
// proposes from the position.
async function placeBox(x, y, options, laneIndex, proteinId) {
  if (!proteinId) {
    const image = state.project.images.find((i) => i.id === state.imageId);
    showStatus(
      image && image.kind === "visible_marker"
        ? "This is a marker image: boxes are placed on signal images."
        : "Add a protein to this image first (Proteins on this image).",
    );
    return;
  }
  const { grow, clientX, clientY } = options;
  const body = { protein_id: proteinId, x, y, lane_index: laneIndex, grow };
  const opened = shownOpening();
  try {
    const answer = await call("POST", "/api/boxes", body);
    applyAnswer(answer, { choose: { proteinId } }); // it stays the click target
    if (sameOpening(answer)) {
      showStatus(""); // not selected, so the next click places the next box
    }
  } catch (error) {
    if (opened !== shownOpening()) {
      return; // about the project shown before: its lanes are not the ones shown now
    }
    const why = LANE_QUESTIONS[error.code];
    if (why && laneIndex === null) {
      const retry = (lane) => placeBox(x, y, options, lane, proteinId);
      askLane(why, proteinId, retry, clientX, clientY);
    } else if (error.code === "no_band_found") {
      showStatus(
        "No band found where you clicked. Click on a band, or Shift+click to place a box" +
          " of the protein's box size.",
      );
    } else {
      report(error);
    }
  }
}

function askLane(message, proteinId, onChoose, clientX, clientY) {
  const picker = $("lane-picker");
  const buttons = $("lane-picker-buttons");
  buttons.replaceChildren();
  const project = state.project;
  const protein = project.proteins.find((p) => p.id === proteinId) || { bands: [] };
  // Taken as the server counts it: the lanes holding the protein's first band.
  const taken = new Set(protein.bands.filter((b) => b.band_index === 0).map((b) => b.lane_index));
  for (const lane of project.lanes) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = String(lane.index + 1);
    button.title = laneName(project, lane.index);
    button.disabled = taken.has(lane.index);
    button.addEventListener("click", () => {
      picker.hidden = true;
      onChoose(lane.index);
    });
    buttons.append(button);
  }
  picker.querySelector("p").textContent = `${message} Which lane is this box in?`;
  picker.hidden = false;
  // At the click, kept inside the stage so every button can be reached.
  const stage = picker.parentElement.getBoundingClientRect();
  const left = Math.min(clientX - stage.left, stage.width - picker.offsetWidth - 8);
  const top = Math.min(clientY - stage.top, stage.height - picker.offsetHeight - 8);
  picker.style.left = `${Math.max(8, left)}px`;
  picker.style.top = `${Math.max(8, top)}px`;
}
$("lane-picker-cancel").addEventListener("click", () => {
  $("lane-picker").hidden = true;
});

async function deleteSelected() {
  if (state.boxId) {
    await edit("DELETE", `/api/boxes/${state.boxId}`).catch(() => {});
  }
}

$("delete-box").addEventListener("click", deleteSelected);
$("box-lane").addEventListener("change", (event) => {
  const lane = Number(event.target.value);
  edit("PUT", `/api/boxes/${state.boxId}/lane`, { lane_index: lane }).catch(() => render());
});
$("polarity").addEventListener("change", (event) => {
  edit("PUT", `/api/images/${state.imageId}/polarity`, { polarity: event.target.value }).catch(
    () => render(),
  );
});
$("remove-image").addEventListener("click", () => {
  const image = state.project.images.find((i) => i.id === state.imageId);
  if (image && window.confirm(`Remove ${isolate(image.original_name)} and its boxes?`)) {
    edit("DELETE", `/api/images/${image.id}`).catch(() => {});
  }
});

$("import-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) {
    return;
  }
  const query = new URLSearchParams({
    name: file.name,
    kind: $("import-kind").value,
    polarity: $("import-polarity").value, // chosen before the file input is enabled
  });
  const membrane = $("import-membrane").value;
  if (membrane) {
    query.set("membrane_id", membrane);
  }
  const opened = shownOpening();
  showStatus(`Importing ${isolate(file.name)}…`);
  try {
    const response = await request("POST", `/api/images?${query}`, {
      body: file,
      contentType: "application/octet-stream",
    });
    const answer = await response.json();
    applyAnswer(answer, { choose: { imageId: answer.image_id, boxId: null } });
    const image = answer.project.images.find((i) => i.id === answer.image_id);
    const name = isolate(image ? image.original_name : file.name);
    if (!sameOpening(answer)) {
      // The import went into the project open when it started, not the one shown.
      showStatus(`${name} was imported into ${isolate(answer.project.name)}, which is not open now.`);
      return;
    }
    // Each import starts a new membrane unless the user chooses otherwise for it.
    $("import-membrane").value = "";
    // What the import found about the file, until it is dismissed by the next action.
    const found = image ? image.warnings.map((warning) => warning.message) : [];
    showStatus(found.length ? `Imported ${name}. ${found.join(" ")}` : "");
  } catch (error) {
    if (opened === shownOpening()) {
      report(error);
    }
  }
});

// The model has no silent default for polarity: it is chosen before importing.
$("import-polarity").addEventListener("change", (event) => {
  $("import-file").disabled = !event.target.value;
  $("import-button").classList.toggle("disabled", !event.target.value);
});

$("zoom-in").addEventListener("click", () => view.zoomCentre(1.25));
$("zoom-out").addEventListener("click", () => view.zoomCentre(0.8));
$("zoom-fit").addEventListener("click", () => view.fit());

document.addEventListener("keydown", (event) => {
  const target = event.target;
  if (
    event.ctrlKey ||
    event.metaKey ||
    event.altKey || // browser shortcuts stay the browser's
    $("projects-dialog").open ||
    target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLTextAreaElement ||
    (target instanceof Element && target.closest("dialog"))
  ) {
    return;
  }
  if (event.key === "Delete" || event.key === "Backspace") {
    event.preventDefault();
    deleteSelected();
  } else if (event.key === "Escape") {
    state.boxId = null;
    $("lane-picker").hidden = true;
    render();
  } else if (event.key === "f" || event.key === "F") {
    view.fit();
  } else if (event.key === "+" || event.key === "=") {
    view.zoomCentre(1.25);
  } else if (event.key === "-") {
    view.zoomCentre(0.8);
  }
});

// --- Quit ---

$("quit").addEventListener("click", async () => {
  $("quit").disabled = true;
  try {
    await request("POST", "/api/quit");
    showStatus("Proteia has stopped. You can close this tab.");
    $("workspace").hidden = true;
    $("quit").hidden = true;
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      $("quit").hidden = true; // this tab cannot reach the running Proteia
    } else {
      showStatus(error.message || "Proteia did not stop. Try Quit again.");
      $("quit").disabled = false;
    }
  }
});

// --- Start ---

async function start() {
  if (!token) {
    showStatus(NEEDS_LAUNCH);
    return;
  }
  try {
    const listing = await call("GET", "/api/projects");
    $("quit").hidden = false;
    showStatus("");
    if (listing.open) {
      applyAnswer(await call("GET", "/api/project"));
    } else {
      await showProjects();
    }
  } catch (error) {
    if (!(error instanceof ApiError && error.status === 401)) {
      showStatus("Proteia is not responding. Start it again to reopen this page.");
    }
  }
}

start();
