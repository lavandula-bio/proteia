// SPDX-License-Identifier: Apache-2.0
// The page: keeps this launch's access token, talks to the local server, and
// shows the open project's images and boxes. Every edit goes to the server,
// which answers with the stored project; the page only draws what it is given.
import { ImageView, MISSING_COLOR } from "/static/view.js";

const TOKEN_KEY = "proteia-token";
const PROTEIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#bcbd22"];
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
const $ = (id) => document.getElementById(id);

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
  imageId: null,
  proteinId: null,
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

// Run an edit; its answer replaces the project state.
async function edit(method, path, json) {
  try {
    const answer = await call(method, path, json);
    showStatus("");
    applyProject(answer.project);
    return answer;
  } catch (error) {
    report(error);
    throw error;
  }
}

const view = new ImageView($("view"), {
  place: (x, y, options) => placeBox(x, y, options),
  move: (boxId, rect) => edit("PUT", `/api/boxes/${boxId}`, { rect }).catch(() => {}),
  select: (boxId) => {
    state.boxId = boxId;
    render();
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

async function openProject(path, name) {
  try {
    const answer = await call("POST", path, { name });
    // Image ids repeat across projects (img-1 in each): drop everything shown.
    view.setImage(null, 0, 0);
    view.setOverlay([], [], null);
    state.shownImageId = null;
    for (const id of [...state.bitmaps.keys()]) {
      forgetBitmap(id);
    }
    state.imageId = null;
    state.proteinId = null;
    state.boxId = null;
    applyProject(answer.project);
    $("projects-dialog").close();
    showStatus("");
  } catch (error) {
    $("projects-error").textContent = error.message;
  }
}

$("new-project").addEventListener("submit", (event) => {
  event.preventDefault();
  openProject("/api/projects", $("new-project-name").value);
});
$("projects-close").addEventListener("click", () => $("projects-dialog").close());
$("projects-dialog").addEventListener("cancel", (event) => {
  if (!state.project) {
    event.preventDefault(); // a project must be open to work
  }
});
$("switch-project").addEventListener("click", () => showProjects().catch(report));
$("reveal").addEventListener("click", () => call("POST", "/api/project/reveal").catch(report));

// --- Applying the server's state ---

function applyProject(project) {
  state.project = project;
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
  render();
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
  renderImages(project);
  renderProteins(project);
  renderBox(project);
  renderNotices(project);
  renderView(project);
}

function choice(list, label, selected, onChoose, swatch) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = selected ? "chosen" : "";
  button.setAttribute("aria-pressed", String(selected));
  if (swatch) {
    const mark = document.createElement("span");
    mark.className = "swatch";
    mark.style.backgroundColor = swatch;
    button.append(mark);
  }
  button.append(label);
  button.addEventListener("click", onChoose);
  const item = document.createElement("li");
  item.append(button);
  list.append(item);
}

function renderImages(project) {
  const list = $("images");
  list.replaceChildren();
  for (const image of project.images) {
    choice(list, image.original_name, image.id === state.imageId, () => {
      state.imageId = image.id;
      state.boxId = null;
      applyProject(state.project);
    });
  }
  const image = project.images.find((i) => i.id === state.imageId);
  $("image-controls").hidden = !image;
  if (image) {
    $("polarity").value = image.polarity;
  }
}

function colorOf(project, proteinId) {
  const index = project.proteins.findIndex((p) => p.id === proteinId);
  return PROTEIN_COLORS[index % PROTEIN_COLORS.length];
}

function renderProteins(project) {
  const list = $("proteins");
  list.replaceChildren();
  const proteins = project.proteins.filter((p) => p.image_id === state.imageId);
  for (const protein of proteins) {
    const role = protein.role === "loading control" ? " (loading control)" : "";
    choice(
      list,
      `${protein.name}${role}`,
      protein.id === state.proteinId,
      () => {
        state.proteinId = protein.id;
        render();
      },
      colorOf(project, protein.id),
    );
  }
  if (!proteins.length) {
    const item = document.createElement("li");
    item.className = "hint";
    item.textContent = "No protein on this image yet.";
    list.append(item);
  }
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

function renderBox(project) {
  const found = state.boxId ? findBox(project, state.boxId) : null;
  $("box-panel").hidden = !found;
  if (!found) {
    return;
  }
  const { protein, band } = found;
  const clipped =
    band.clipped === true ? " Over-exposed." : band.clipped === null ? " Not checked." : "";
  $("box-summary").textContent = `${protein.name}, ${laneName(project, band.lane_index)}.${clipped}`;
  const select = $("box-lane");
  select.replaceChildren();
  for (const lane of project.lanes) {
    const option = document.createElement("option");
    option.value = String(lane.index);
    option.textContent = laneName(project, lane.index);
    select.append(option);
  }
  select.value = String(band.lane_index);
}

function renderNotices(project) {
  const list = $("notices");
  list.replaceChildren();
  const add = (text, kind) => {
    const item = document.createElement("li");
    item.className = kind;
    item.textContent = text;
    list.append(item);
  };
  if (!project.lanes.length) {
    add("No lanes are declared yet.", "warning");
  }
  for (const protein of project.proteins.filter((p) => p.image_id === state.imageId)) {
    const clipped = protein.bands.filter((b) => b.clipped === true);
    if (clipped.length) {
      const lanes = clipped.map((b) => b.lane_index + 1).join(", ");
      add(`${protein.name}: over-exposed in lane ${lanes}.`, "clipped");
    }
    if (protein.missing_lanes.length && protein.bands.length) {
      const lanes = protein.missing_lanes.map((m) => m.lane_index + 1).join(", ");
      add(`${protein.name}: no box in lane ${lanes}.`, "missing");
    }
  }
  if (!list.children.length) {
    add("No problems found.", "ok");
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
      view.setOverlay(boxes, ghosts, state.boxId);
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

async function placeBox(x, y, options, laneIndex = null, proteinId = state.proteinId) {
  if (!proteinId) {
    showStatus("Choose a protein on this image first.");
    return;
  }
  const { grow, clientX, clientY } = options;
  const body = { protein_id: proteinId, x, y, lane_index: laneIndex, grow };
  try {
    const answer = await call("POST", "/api/boxes", body);
    showStatus(""); // not selected, so the next click places the next box
    applyProject(answer.project);
  } catch (error) {
    const why = LANE_QUESTIONS[error.code];
    if (why && laneIndex === null) {
      const retry = (lane) => placeBox(x, y, options, lane, proteinId);
      askLane(why, proteinId, retry, clientX, clientY);
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
  if (image && window.confirm(`Remove ${image.original_name} and its boxes?`)) {
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
  showStatus(`Importing ${file.name}…`);
  try {
    const response = await request("POST", `/api/images?${query}`, {
      body: file,
      contentType: "application/octet-stream",
    });
    const answer = await response.json();
    state.imageId = answer.image_id;
    showStatus("");
    applyProject(answer.project);
  } catch (error) {
    report(error);
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
      applyProject((await call("GET", "/api/project")).project);
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
