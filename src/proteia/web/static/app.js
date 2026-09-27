// SPDX-License-Identifier: Apache-2.0
// The page: keeps this launch's access token, talks to the local server, and
// shows the open project's images, proteins, boxes and checks. Every edit, and
// every undo and redo, goes to the server, which answers with the stored
// project and its results; the page only draws what it is given.
import { ChartCards } from "/static/charts.js";
import { Dock } from "/static/dock.js";
import {
  $,
  counted,
  focusLost,
  inWords,
  isolate,
  netText,
  rebuild,
  sentence,
  span,
} from "/static/dom.js";
import { LaneTable } from "/static/lanes.js";
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

// The results dock: its "Updating…" counts every answer awaited (call()).
const dock = new Dock();

// --- Talking to the server ---

class ApiError extends Error {
  constructor(status, body) {
    super(body && body.message ? body.message : `the server answered ${status}`);
    this.status = status;
    this.code = body && body.code;
    this.ids = (body && body.ids) || [];
  }
}

// Every request names the token in a header; nothing relies on cookies. An
// aborted `signal` cancels it; `priority` orders it among those waiting for a
// connection (the browser's fetch priority).
async function request(method, path, { json, body, contentType, signal, priority } = {}) {
  const headers = new Headers({ Authorization: `Bearer ${token}` });
  if (json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(json);
  } else if (contentType) {
    headers.set("Content-Type", contentType);
  }
  const response = await fetch(path, {
    method,
    headers,
    body,
    cache: "no-store",
    signal,
    priority,
  });
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
  return dock.track(
    request(method, path, { json }).then((response) =>
      response.status === 204 ? null : response.json(),
    ),
  );
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

// A press on `button` leaves the keyboard focus where it is. Undo and Redo name
// the change they take back or make again: were the focus to leave the protein
// name field, what is typed there would be committed first (its change event)
// and the step would act on that rename instead. The typing stays, uncommitted.
function keepsFocus(button) {
  button.addEventListener("mousedown", (event) => event.preventDefault());
}

let statusUndo = null; // the change the status line's Undo takes back: {seq}, or null

// Show `text` in the status line ("" empties it), with an optional action after
// it: {label, name (its accessible name), seq, run}. The action takes back the
// change logged as `seq`, and goes once the history has moved past it. Gives
// the action's button, or null.
function showStatus(text, action = null) {
  const line = $("status");
  line.textContent = text;
  statusUndo = null;
  let button = null;
  if (text && action) {
    button = document.createElement("button");
    button.type = "button";
    button.textContent = action.label;
    button.setAttribute("aria-label", action.name);
    keepsFocus(button);
    // Taken once: a second press before the answer (a double click) would find
    // the change no longer the last, and say so over what the first one did.
    button.addEventListener("click", () => {
      action.run();
      button.disabled = true;
    });
    const part = document.createElement("span");
    part.className = "status-action";
    part.append(" — ", button);
    line.append(part);
    statusUndo = { seq: action.seq };
  }
  placeStatus();
  return button;
}

let statusFloor = 0; // px: the tallest status line shown since the window was resized

// While a project is shown the status line keeps its place under the header,
// empty or not: a message coming or going (after every undo, and the next edit)
// never moves the panel or the image under the pointer, so a second click, or
// a click on a band, lands where it was aimed. It is two lines tall (app.css),
// or as tall as the tallest message yet at this window size, so a shorter one
// after it moves nothing back. Otherwise it shows only a message.
function placeStatus() {
  const line = $("status");
  line.hidden = !line.textContent && $("workspace").hidden;
  if ($("workspace").hidden) {
    statusFloor = 0;
    line.style.minHeight = "";
  } else if (line.offsetHeight > statusFloor) {
    statusFloor = line.offsetHeight;
    line.style.minHeight = `${statusFloor}px`;
  }
}

// Lines wrap anew at another width: the status line is as tall as its message.
window.addEventListener("resize", () => {
  statusFloor = 0;
  $("status").style.minHeight = "";
  placeStatus();
});

// Drop the status line's Undo once undo would take back another change.
function renderStatusUndo(history) {
  if (statusUndo && !(history.undo && history.undo.seq === statusUndo.seq)) {
    statusUndo = null;
    const part = $("status").querySelector(".status-action");
    if (part) {
      part.remove();
    }
  }
}

function report(error) {
  if (error instanceof ApiError && error.status === 401) {
    return;
  }
  showStatus(error.message);
}

// Answers can arrive out of order: one is shown only if it is not older than
// the one shown, by (open_id, revision). The open id counts the server's
// openings (creates, opens of another project, and rereads of the open one's
// project.json after it changed outside Proteia), so an answer about an
// earlier opening is older whatever its revision, and the answer of the latest
// open is newer than all before it. Opening the project already open otherwise
// answers its own open id, at its latest revision: openProject asks only once
// every edit has its answer, so that answer is not older than the one shown,
// and is shown.
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

// The edits made outside the panel's queue (boxes, images) that have no answer
// yet, counted from the moment they are made. An undo waits for them, so it
// takes back the last change made, not the one before it; so does "Clear
// boxes", so it clears a box being placed; and so does opening another
// project, so they end in the project they were made in.
const inFlight = new Set();

// Make an edit outside the panel's queue: `sendIt()` sends it once the queue as
// it stands has run, so an undo, redo or clear asked for before it reaches the
// server first and never takes back or clears this edit (a band clicked while
// undos queue up). Gives its answer.
function ordered(sendIt) {
  const promise = proteinPanel.queue.then(sendIt);
  inFlight.add(promise);
  const done = () => inFlight.delete(promise);
  promise.then(done, done);
  return promise;
}

// Settles once every edit in flight now has its answer (or refusal).
function pending() {
  return Promise.allSettled([...inFlight]);
}

// Send an edit and show a refusal in the status line (unless another project
// is shown by then).
async function edit(method, path, json) {
  const opened = shownOpening();
  try {
    return await ordered(() => send(method, path, json));
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
  row: (rect, proteinId) => placeRow(rect, proteinId),
  move: (boxId, rect) =>
    edit("PUT", `/api/boxes/${boxId}`, { rect })
      .then((answer) => noteBoxStep(answer, "move_box", [boxId]))
      .catch(() => {}),
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
  undo: (seq) => takeStep("undo", { seq }),
  pending,
  laneName: (index) => laneName(state.project, index),
});

// Its edits run in the panel's queue: in order with the protein edits, the
// undos and redos, and before the box edits made after them.
const laneTable = new LaneTable({
  queueEdit: (task, options) => proteinPanel.queueEdit(task, options),
  pending,
  send,
  status: showStatus,
});

// Each chart's drawing is fetched with the token ("Updating…" counts it until
// it arrives or no card awaits it), at a low priority: an edit waiting for a
// connection goes before the drawings waiting with it. One the server no
// longer keeps is asked for again after the project is read again; that
// answer is shown only if it is about the opening shown, and not while
// another project is being opened (openProject shows that one).
const charts = new ChartCards({
  fetch: (path, signal) =>
    dock.track(
      request("GET", path, { signal, priority: "low" }).then((response) => response.blob()),
      { chart: true },
    ),
  reread: async () => {
    const answer = await call("GET", "/api/project");
    if (opening === null && sameOpening(answer)) {
      applyAnswer(answer);
    }
  },
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
  $("projects-error").textContent = "";
  try {
    // Every edit made before, in the panel's queue (its edits, the undos,
    // redos and clears) or outside it (boxes and images, which wait for that
    // queue before they are sent), ends in the project it was made in, its
    // refusal shown (the dialog is modal and the shortcuts are off while it is
    // open: none is made meanwhile). Then, before asking, none may reach the
    // project opened next: an undo sent after the open would take back a
    // change of that project.
    await Promise.all([proteinPanel.settled(), pending()]);
    proteinPanel.invalidateEdits();
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
    proteinPanel.forgetTyped();
    laneTable.forgetTyped();
    charts.forget(); // its object URLs revoked: chart URLs repeat across projects too
    $("lane-picker").hidden = true; // its retry places a box in the project it asked about
    applyAnswer(answer, { choose: { imageId: null, proteinId: null, boxId: null } });
    lastBoxStep = null; // log numbers repeat across projects
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
  $("undo").hidden = !project;
  $("redo").hidden = !project;
  placeStatus();
  if (!project) {
    return;
  }
  $("project-name").textContent = project.name;
  $("project-name").title = project.name; // whole, when the header cuts it short
  renderHistory(project.history);
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
  laneTable.render(project, state.results);
  charts.render(state.results);
  dock.render(state.results);
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

// The box's net from the results: the lane whose band is this box.
function boxNetText(protein, band) {
  const column = state.results.proteins.find((c) => c.protein_id === protein.id);
  const lane = column ? column.band_ids.indexOf(band.id) : -1;
  if (lane < 0) {
    return band.band_index > 0 ? "Not quantified (an extra band)" : "—";
  }
  const net = column.nets[lane];
  return net === null ? "—" : netText(net);
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
  $("box-net").textContent = boxNetText(protein, band);
  const edited = band.manually_edited ? "; moved or re-laned by hand since" : "";
  $("box-source").textContent = `${PLACED_BY[band.source] || band.source}${edited}`;
  const select = $("box-lane");
  select.replaceChildren();
  for (const lane of project.lanes) {
    select.append(new Option(laneName(project, lane.index), String(lane.index)));
  }
  select.value = String(band.lane_index);
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

// What a drag on the membrane boxes: a row of the chosen protein, once it is on
// this signal image and the lanes are declared ({proteinId, color, lanes});
// otherwise null, and the drag pans.
function rowTool(project, image) {
  const protein = image
    ? project.proteins.find((p) => p.id === state.proteinId && p.image_id === image.id)
    : null;
  if (!protein || image.kind === "visible_marker" || !project.lanes.length) {
    return null;
  }
  const color = colorOf(project, protein.id);
  return { proteinId: protein.id, color, lanes: project.lanes.length };
}

function renderHint(project, image) {
  const hint = $("view-hint");
  hint.hidden = !image;
  if (!image) {
    return;
  }
  const protein = project.proteins.find((p) => p.id === state.proteinId);
  const lanes = project.lanes.length;
  if (image.kind === "visible_marker") {
    hint.textContent = "A marker image: boxes are placed on signal images.";
  } else if (!protein) {
    hint.textContent = "Add a protein to place boxes on this image.";
  } else if (!lanes) {
    hint.replaceChildren(
      "Boxes go to ",
      span("hint-target", protein.name),
      " · Click a band: one box · Shift+click: fixed box · Drag: pan · Wheel: zoom" +
        " · Declare the lanes to box a whole row at once",
    );
  } else {
    // The row box is the analysis region: every declared lane, and no ladder.
    const over = lanes === 1 ? "over the lane" : `over all ${lanes} lanes`;
    hint.replaceChildren(
      "Boxes go to ",
      span("hint-target", protein.name),
      ` · Drag across a row (${over}, not a ladder): one box per lane · Click a band: one box` +
        " · Shift+click: fixed box · Space+drag: pan · Wheel: zoom",
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
  view.setRowTool(rowTool(project, image));
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
    const answer = await ordered(() => call("POST", "/api/boxes", body));
    applyAnswer(answer, { choose: { proteinId } }); // it stays the click target
    if (sameOpening(answer)) {
      noteBoxStep(answer, "place_box", [answer.band_id]);
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

// --- A row of boxes from a row box ---

// Stored lane indices in words, numbered from 1: "lane 8", "lanes 4 and 8".
function lanesPhrase(indices) {
  const numbers = indices.map((index) => String(index + 1));
  return `${numbers.length === 1 ? "lane" : "lanes"} ${inWords(numbers)}`;
}

// Why a row left a lane with neither a box nor an n.d. mark, by the
// detector's reason for the empty lane (LaneReason in core/rowdetect.py). A
// lane where no band reaches the detection limit (no_band) gets an n.d. mark,
// unless its place lies outside the row box.
const NOT_MEASURED = {
  artefact: "a stain or streak",
  edge_signal: "only a neighbouring row's signal",
  unassigned: "signal that fits no lane",
  no_band: "outside the row box",
};

// The detector's warnings about a row it placed (WARNING_FLAGS in
// core/rowdetect.py), in words. `note`: words of the detector's note on it,
// which names its lanes first ("lane 5: …", "lanes 3, 7: …").
const ROW_WARNINGS = {
  background_mismatch: {
    note: null,
    words: () => "uneven background under some boxes: check their nets",
  },
  size_outlier: {
    note: "left out of the shared size",
    words: (lanes) =>
      `a band much larger than the others${lanes ? ` (${lanes})` : ""} did not set the box size`,
  },
  multiple_components: {
    note: "second separate component",
    words: (lanes) => `two bands in ${lanes || "a lane"}: the box covers the stronger one`,
  },
};

function warningText(flag, notes) {
  const warning = ROW_WARNINGS[flag];
  if (!warning) {
    return null; // a warning this page does not know: the log keeps it
  }
  const note = warning.note && notes.find((text) => text.includes(warning.note));
  const match = note ? /^(lanes? [\d, ]+):/.exec(note) : null;
  return warning.words(match ? match[1] : null);
}

// What a row did, lane by lane, from its answer (lanes numbered from 1): the
// boxes placed, the lanes with no band (n.d.), those kept as they were, those
// not measured and why, the boxes an earlier row placed that went, and the
// detector's warnings. `before`: the state shown before, where those boxes are.
// Gives {text, check, unchanged}: `unchanged` when the row changed nothing (the
// same drag again), `check` when it changed the project and left signal that
// fits no lane, the mark of a row box over part of the row (its bands then
// read as several lanes each): the text asks to check the lane numbers.
function rowReport(answer, name, before) {
  const empty = new Map(answer.empty.map((lane) => [lane.lane_index, lane]));
  const kept = new Set(answer.kept_lanes);
  const placed = answer.band_ids.filter(
    (id, lane) => id !== null && !empty.has(lane) && !kept.has(lane),
  ).length;
  const replaced = answer.replaced_band_ids.length;
  let head = `Placed ${counted(placed, "box", "boxes")} of ${name}`;
  const unchanged = Boolean(
    before &&
      before.open_id === answer.project.open_id &&
      before.revision === answer.project.revision,
  );
  if (unchanged) {
    head = `No change: the row box finds the boxes of ${name} where they are`; // nothing logged
  } else if (replaced) {
    head += ` (${replaced === placed ? "all" : replaced} in place of the earlier ones)`;
  }
  const parts = [head];
  if (answer.undetected_lanes.length) {
    parts.push(`no band in ${lanesPhrase(answer.undetected_lanes)} (n.d.)`);
  }
  const bands = new Map(answer.project.proteins.flatMap((p) => p.bands).map((b) => [b.id, b]));
  const keptLanes = answer.kept_lanes.map((lane) => [lane, bands.get(answer.band_ids[lane])]);
  const edited = keptLanes.filter(([, band]) => band && band.manually_edited).map(([l]) => l);
  const yours = keptLanes.filter(([, band]) => !(band && band.manually_edited)).map(([l]) => l);
  if (edited.length) {
    parts.push(`kept ${lanesPhrase(edited)} (edited)`);
  }
  if (yours.length) {
    const boxes = yours.length === 1 ? "box" : "boxes";
    parts.push(`kept your ${boxes} in ${lanesPhrase(yours)} (no band found)`);
  }
  const unmeasured = new Map(); // reason -> lanes
  for (const lane of answer.unmeasured_lanes) {
    const reason = empty.has(lane) ? empty.get(lane).reason : "";
    unmeasured.set(reason, [...(unmeasured.get(reason) || []), lane]);
  }
  for (const [reason, lanes] of unmeasured) {
    parts.push(`${lanesPhrase(lanes)} not measured: ${NOT_MEASURED[reason] || reason}`);
  }
  if (answer.removed_band_ids.length) {
    const was = answer.removed_band_ids.map((id) => (before ? findBox(before, id) : null));
    parts.push(
      was.every(Boolean)
        ? `removed the earlier ${was.length === 1 ? "box" : "boxes"} in ${lanesPhrase(
            was.map((found) => found.band.lane_index).sort((a, b) => a - b),
          )}`
        : `removed ${counted(was.length, "earlier box", "earlier boxes")}`,
    );
  }
  for (const flag of answer.flags) {
    const text = warningText(flag, answer.notes);
    if (text) {
      parts.push(text);
    }
  }
  if (answer.right_to_left) {
    parts.push("lanes read right to left, as the boxes on this image run");
  }
  const check = !unchanged && unmeasured.has("unassigned");
  if (check) {
    const all = answer.band_ids.length;
    parts.push(
      "check the boxes' lane numbers: a row box over part of the row misreads the lanes" +
        ` (Undo, then drag across all ${all})`,
    );
  } else if (unmeasured.size) {
    parts.push("click a dashed placeholder to box a lane by hand");
  }
  return { text: parts.join(" · "), check, unchanged };
}

// What to do about a refused row, by the refusal's code.
const ROW_HINTS = {
  row_too_small: "Drag across the whole row, over every lane.",
  no_band_found: "Drag over a row of bands, or click a band to box one lane.",
  out_of_image: "Drag over the image.",
  no_lanes: "Declare the lanes in the lane table first.",
  size_would_overlap: "Move apart or delete the boxes it keeps (edited by hand), then drag again.",
};

// The last box change made from this page, as logged: {seq, ids}, the boxes it
// placed, moved or gave a lane (a row: the boxes it placed or replaced); or
// null. A row refused over the boxes already on the image names them (or
// their proteins): if it names one of these, the last change may have put it
// in the wrong lane.
let lastBoxStep = null;

// Note the box change `answer` logged as `action` (another change logged
// meanwhile leaves the last one unknown).
function noteBoxStep(answer, action, ids) {
  const step = answer && sameOpening(answer) ? answer.project.history.undo : null;
  lastBoxStep = step && step.action === action ? { seq: step.seq, ids } : null;
}

// Whether the change undo would take back is the last box change, and made
// one of the boxes (or boxes of the proteins) the refusal `error` names.
function lastBoxStepNamed(error) {
  const step = state.project.history.undo;
  if (!lastBoxStep || !step || step.seq !== lastBoxStep.seq) {
    return false;
  }
  const named = new Set(error.ids);
  return lastBoxStep.ids.some((id) => {
    const found = findBox(state.project, id);
    return named.has(id) || (found !== null && named.has(found.protein.id));
  });
}

// A refused row in the status line: the server's reason with the protein
// named, what to do, and, when the boxes already on the image did not let the
// row read its lanes just after a box was placed or changed, an Undo of that
// change.
function showRowRefusal(error, name) {
  if (error instanceof ApiError && error.status === 401) {
    return;
  }
  const sentences = [`Row box of ${name} not placed: ${sentence(error.message)}`];
  let action = null;
  if (error.code === "overlap") {
    const lanes = error.ids.map((id) => findBox(state.project, id)).filter(Boolean);
    const which = lanes.length
      ? `the box in ${lanesPhrase(lanes.map((found) => found.band.lane_index))}`
      : "that box";
    sentences.push(`Move or delete ${which}, then drag again.`);
  } else if (error.code === "row_lanes_unclear") {
    // Named: the boxes on the image the row disagrees with, or their proteins
    // (none when the row box alone does not show the lanes).
    const step = state.project.history.undo;
    if (lastBoxStepNamed(error)) {
      const words = actionWords(step.action);
      sentences.push(
        `If the last change (${words}) put a box in the wrong lane, Undo takes it back.`,
      );
      action = {
        label: "Undo",
        name: `Undo ${words}`,
        seq: step.seq,
        run: () => takeStep("undo", { seq: step.seq }),
      };
    }
  } else if (ROW_HINTS[error.code]) {
    sentences.push(ROW_HINTS[error.code]);
  }
  showStatus(sentences.join(" "), action);
}

// Box the first band of `proteinId` in every declared lane from the row box
// dragged over its row (image pixels, end exclusive), then say what the row
// did. Sent in order with the other edits (ordered), so an undo or an open
// asked for after it waits for it; its answer goes through the stale-answer
// guard (applyAnswer). A refusal changes nothing.
async function placeRow(rect, proteinId) {
  const protein = state.project.proteins.find((p) => p.id === proteinId);
  if (!protein) {
    return;
  }
  const opened = shownOpening();
  let before = null;
  try {
    const answer = await ordered(() => {
      before = state.project;
      return call("POST", "/api/boxes/row", { protein_id: proteinId, rect });
    });
    applyAnswer(answer, { choose: { proteinId } }); // it stays the drag's protein
    if (sameOpening(answer)) {
      const name = (answer.project.proteins.find((p) => p.id === proteinId) || protein).name;
      const { text, check, unchanged } = rowReport(answer, name, before);
      if (!unchanged) {
        // Its boxes: those it placed or replaced, not those it kept as they were.
        const kept = new Set(answer.kept_lanes);
        const made = answer.band_ids.filter((id, lane) => id !== null && !kept.has(lane));
        noteBoxStep(answer, "detect_row_boxes", made);
      }
      // Asked to check the lane numbers: an Undo of this row at hand. It goes
      // once the history moves on.
      const step = answer.project.history.undo;
      const undo =
        check && step && step.action === "detect_row_boxes"
          ? {
              label: "Undo",
              name: `Undo the row box of ${name}`,
              seq: step.seq,
              run: () => takeStep("undo", { seq: step.seq }),
            }
          : null;
      showStatus(text, undo);
    }
  } catch (error) {
    if (opened === shownOpening()) {
      showRowRefusal(error, protein.name);
    }
  }
}

async function deleteSelected() {
  if (state.boxId) {
    await edit("DELETE", `/api/boxes/${state.boxId}`).catch(() => {});
  }
}

$("delete-box").addEventListener("click", deleteSelected);
$("box-lane").addEventListener("change", (event) => {
  const lane = Number(event.target.value);
  const boxId = state.boxId;
  edit("PUT", `/api/boxes/${boxId}/lane`, { lane_index: lane })
    .then((answer) => noteBoxStep(answer, "set_box_lane", [boxId]))
    .catch(() => render());
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
    const answer = await ordered(() =>
      dock.track(
        request("POST", `/api/images?${query}`, {
          body: file,
          contentType: "application/octet-stream",
        }).then((response) => response.json()),
      ),
    );
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

// --- Undo and redo ---

// Each change the server logs, in the words of the control that makes it.
const ACTION_WORDS = {
  new_project: "create project",
  import_image: "import image",
  remove_image: "remove image",
  set_polarity: "change band polarity",
  set_lanes: "set lanes",
  set_reference_condition: "set reference condition",
  add_protein: "add protein",
  edit_protein: "edit protein",
  remove_protein: "remove protein",
  place_box: "place box",
  move_box: "move box",
  remove_box: "delete box",
  set_box_lane: "change box lane",
  set_box_size: "change box size",
  clear_boxes: "clear boxes",
  detect_row_boxes: "detect row boxes",
  remove_undetected: "remove n.d. mark",
  undo: "undo",
  redo: "redo",
};

// A logged change in words; one this page does not know, by its id.
function actionWords(action) {
  return Object.hasOwn(ACTION_WORDS, action) ? ACTION_WORDS[action] : action;
}

const STEPS = {
  undo: { verb: "Undo", done: "Undid", keys: "Ctrl+Z" },
  redo: { verb: "Redo", done: "Redid", keys: "Ctrl+Shift+Z or Ctrl+Y" },
};

// The Undo and Redo buttons, each named after the change it would take back or
// make again, and disabled when there is none. The tooltip holds the whole
// label, which a narrow window cuts short.
function renderHistory(history) {
  for (const [direction, { verb, keys }] of Object.entries(STEPS)) {
    const step = history[direction];
    const button = $(direction);
    button.disabled = !step;
    button.textContent = step ? `${verb}: ${actionWords(step.action)}` : verb;
    button.title = step ? `${button.textContent} (${keys})` : `Nothing to ${direction}`;
  }
  renderStatusUndo(history);
}

// What `ids` and not-detected `records` name in `project`, counted: images,
// proteins and boxes (a membrane goes with its images), and n.d. marks.
function counts(project, ids, records) {
  const named = new Set(ids);
  const count = (items) => items.filter((item) => named.has(item.id)).length;
  const parts = [
    [count(project.images), "image", "images"],
    [count(project.proteins), "protein", "proteins"],
    [count(project.proteins.flatMap((p) => p.bands)), "box", "boxes"],
    [records.length, "not-detected mark", "not-detected marks"],
  ];
  return inWords(parts.filter(([n]) => n > 0).map(([n, one, many]) => counted(n, one, many)));
}

// What an undo or redo did: the change, then what went (found in the state the
// page showed `before`) and what came back.
function stepText(direction, answer, before) {
  const sentences = [`${STEPS[direction].done}: ${actionWords(answer.action)}.`];
  const went = counts(before, answer.removed, answer.undetected_removed);
  const back = counts(answer.project, answer.restored, answer.undetected_restored);
  if (went) {
    sentences.push(`Removed ${went}.`);
  }
  if (back) {
    sentences.push(`Brought back ${back}.`);
  }
  return sentences.join(" ");
}

function reportStep(direction, error) {
  if (error.code === `nothing_to_${direction}`) {
    showStatus(`Nothing to ${direction}.`);
  } else if (error.code === "image_file_changed") {
    // The image is often not in the project shown (undoing a removal), so its
    // name may be unknown here; and the file may be missing or changed.
    const names = error.ids
      .map((id) => state.project.images.find((image) => image.id === id))
      .filter(Boolean)
      .map((image) => isolate(image.original_name));
    const which = names.length ? `the file of ${inWords(names)} is` : "an image file it needs is";
    showStatus(
      `Cannot ${direction}: ${which} missing from the project's images folder` +
        " or was changed outside Proteia.",
    );
  } else {
    report(error);
  }
}

// Undo or redo once every edit made before it has its answer (the panel's
// queue and adds, the box and image edits in flight), so it takes back or
// makes again the change the history names by then. `seq`: only if that change
// is still the one logged as `seq` (the status line's Undo of a clear). It runs
// in the panel's queue and its answer goes through send(), so it never reaches
// or shows a project opened after it was asked for (see openProject); the
// edits made after it wait for it (ordered, and the queue).
function takeStep(direction, { seq = null } = {}) {
  if (!state.project || $("workspace").hidden) {
    return;
  }
  const had = document.activeElement;
  const offered = state.project.history[direction]; // what the page showed when asked
  const after = Promise.allSettled([pending(), proteinPanel.adding]);
  proteinPanel.queueEdit(
    async (current) => {
      const step = state.project.history[direction];
      if (seq !== null && !(step && step.seq === seq)) {
        showStatus(
          "Not undone: other changes were made since. Undo at the top takes back the last one.",
        );
        keepFocus(had, direction, true);
        return null;
      }
      if (!step) {
        // Gone meanwhile: taken by the steps asked for before it, or, for a
        // redo, dropped by an edit made before it (an edit ends what redo can
        // make again). Said, unless the page offered none when it was asked.
        if (offered) {
          showStatus(`Nothing to ${direction}.`);
        }
        return null;
      }
      const before = state.project;
      const opened = shownOpening();
      try {
        const answer = await send("POST", `/api/${direction}`);
        if (!answer || !current()) {
          return null;
        }
        showStatus(stepText(direction, answer, before));
        keepFocus(had, direction, seq !== null);
        return answer;
      } catch (error) {
        if (current() && opened === shownOpening()) {
          reportStep(direction, error);
          keepFocus(had, direction, seq !== null);
        }
        return null;
      }
    },
    { after },
  );
}

// The control a step was taken from (`had`, focused then) may be gone (the
// status line's Undo) or disabled (the last Undo): if the focus was still on
// it, the keyboard goes on to the next useful one.
function keepFocus(had, direction, fromStatus) {
  const gone = had && (!had.isConnected || had.disabled);
  if (!gone || !(focusLost() || document.activeElement === had)) {
    return;
  }
  const other = direction === "undo" ? "redo" : "undo";
  const targets = [...(fromStatus ? [$("clear-boxes")] : []), $(direction), $(other)];
  const target = targets.find((t) => !t.disabled && t.getClientRects().length);
  if (target) {
    target.focus();
  }
}

for (const direction of Object.keys(STEPS)) {
  keepsFocus($(direction));
  $(direction).addEventListener("click", () => takeStep(direction));
}

// Inputs where Ctrl+Z is not the browser's text undo.
const NOT_TEXT = new Set([
  "button",
  "checkbox",
  "color",
  "file",
  "image",
  "radio",
  "range",
  "reset",
  "submit",
]);

function editsText(element) {
  return (
    (element instanceof HTMLInputElement && !NOT_TEXT.has(element.type)) ||
    element instanceof HTMLTextAreaElement ||
    (element instanceof HTMLElement && element.isContentEditable)
  );
}

// Ctrl+Z undoes; Ctrl+Shift+Z and Ctrl+Y redo (Cmd on macOS). Not while the
// focus is where the browser undoes typing (a text field, editable text), nor
// while a dialog is open. A select has no text undo: right after choosing in
// one (polarity, a box's lane) is when Ctrl+Z is wanted.
document.addEventListener("keydown", (event) => {
  if (!(event.ctrlKey || event.metaKey) || event.altKey || event.isComposing) {
    return;
  }
  // The letter typed; by the key's place on a layout without Latin letters.
  const letter = /^[a-z]$/i.test(event.key)
    ? event.key.toLowerCase()
    : { KeyZ: "z", KeyY: "y" }[event.code];
  let direction = null;
  if (letter === "z") {
    direction = event.shiftKey ? "redo" : "undo";
  } else if (letter === "y" && !event.shiftKey) {
    direction = "redo";
  }
  if (
    !direction ||
    editsText(event.target) ||
    document.querySelector("dialog[open]") ||
    $("workspace").hidden
  ) {
    return;
  }
  event.preventDefault();
  takeStep(direction);
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
    $("undo").hidden = true;
    $("redo").hidden = true;
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
