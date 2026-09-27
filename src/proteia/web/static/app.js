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
  lanesPhrase,
  netText,
  rebuild,
  sentence,
  span,
} from "/static/dom.js";
import { ImportDialog, refusedText } from "/static/handoffs.js";
import { LaneTable } from "/static/lanes.js";
import { colorOf, ProteinPanel } from "/static/proteins.js";
import { ImageView, MISSING_COLOR } from "/static/view.js";

const TOKEN_KEY = "proteia-token";
const NEEDS_LAUNCH =
  "This page needs the link Proteia opens when it starts. Start Proteia again to open it.";
// Every request names the opening of the project the page shows (its open id)
// in this header. The server refuses one about another opening with this code,
// before it does anything: another tab opened a project since, or Proteia read
// the open one's project.json again. So no page edits or reads a project it
// does not show (projectChanged).
const OPENING_HEADER = "Proteia-Opening";
const PROJECT_CHANGED = "project_changed";

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
    this.detail = (body && body.detail) || null; // why, when the refusal says (a row box's)
  }
}

// An edit never sent: it was made while the page showed a project it no longer
// shows (ordered). The page says why it shows another one (followOpening).
class NotSent extends Error {
  constructor() {
    super("Not sent: another project is shown now.");
  }
}

// Every request names the token in a header; nothing relies on cookies. It
// names the opening of the project shown too (OPENING_HEADER), as it is when
// the request is sent, unless `anyProject`: a read of the project open now,
// whichever it is, or a create or open, whose answer is of the opening it
// makes. An edit that waits for others is sent only while the page shows the
// project it was made in (ordered, and the panel's queue once showOpened drops
// its edits), so that is the opening shown when it was made. With `answer`, it
// gives the JSON answer (null for none) in place of the response; but not an
// answer about an opening newer than the one the request named: that is taken
// as a project_changed refusal (projectChanged, which shows the project open
// now), never shown as a newer state of the project shown, as applyAnswer
// would show it (keeping what showOpened drops: previews, typed values, queued
// edits). The server answers each request within the opening it names; the
// page does not rely on it. `closes`: with `anyProject`, the opening a request
// that opens another project closes (an import of images handed to Proteia),
// named all the same, so the server refuses it if another project is open now;
// its answer is of the opening it makes. An aborted `signal` cancels it;
// `priority` orders it among those waiting for a connection (the browser's
// fetch priority).
async function request(
  method,
  path,
  {
    json,
    body,
    contentType,
    signal,
    priority,
    anyProject = false,
    closes = null,
    answer = false,
  } = {},
) {
  const headers = new Headers({ Authorization: `Bearer ${token}` });
  const named = anyProject ? null : shownOpening();
  if (named) {
    headers.set(OPENING_HEADER, String(named));
  } else if (closes) {
    headers.set(OPENING_HEADER, String(closes));
  }
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
    const error = new ApiError(response.status, detail);
    if (error.code === PROJECT_CHANGED) {
      projectChanged(error, method, path);
    }
    throw error;
  }
  if (!answer) {
    return response;
  }
  const answered = response.status === 204 ? null : await response.json();
  const now = answered && answered.project ? answered.project.open_id : null;
  if (named && now > named) {
    const error = new ApiError(response.status, {
      code: PROJECT_CHANGED,
      message: `The answer is about opening ${now}, not ${named}, which this page named.`,
      detail: { open: answered.project.name, open_id: now },
    });
    projectChanged(error, method, path, { answered: true });
    throw error;
  }
  return answered;
}

// Gives the answer (request()'s `answer`); `options`: request()'s, such as
// `anyProject`.
async function call(method, path, json, options = {}) {
  return dock.track(request(method, path, { ...options, json, answer: true }));
}

// --- Page state ---

const state = {
  project: null, // the last project state the server sent
  results: null, // the results computed from that same state
  answered: null, // {open_id, revision} of that state
  imageId: null,
  proteinId: null, // the protein a click on the image places a box of
  boxId: null,
  // image id -> {grey, original}: Promises of its previews' ImageBitmaps, each
  // fetched when first shown
  bitmaps: new Map(),
  shownImageId: null, // the image the view shows (null while a preview loads)
  // The images shown in their original colours ("Original colours" on): kept
  // for this page only, never in the project.
  originalColours: new Set(),
};

function forgetBitmap(imageId) {
  const previews = state.bitmaps.get(imageId);
  state.bitmaps.delete(imageId);
  for (const pending of Object.values(previews || {})) {
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
// it: {label, name (its accessible name), seq, run}. With `seq`, the action
// takes back the change logged as `seq`, and goes once the history has moved
// past it. Without, it does what changes nothing (Show folder after an export)
// and goes with the message. Gives the action's button, or null.
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
    const undoes = action.seq !== undefined;
    // An Undo is taken once: a second press before the answer (a double click)
    // would find the change no longer the last, and say so over what the first
    // one did. Another action may be taken again once its answer is in (`run`
    // gives a Promise that settles then): one press, one request.
    button.addEventListener("click", () => {
      const running = action.run();
      button.disabled = true;
      if (!undoes) {
        const again = () => {
          button.disabled = false;
          if (button.isConnected && focusLost()) {
            button.focus(); // the keyboard stays on it
          }
        };
        Promise.resolve(running).then(again, again);
      }
    });
    const part = document.createElement("span");
    part.className = "status-action";
    part.append(" — ", button);
    line.append(part);
    statusUndo = undoes ? { seq: action.seq } : null;
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

// A refusal the page says nothing about where it was asked for: 401 (the page
// says it needs the launch link) and project_changed (the page shows the
// project open now and says why, or it was about a project this page has
// since switched from itself); and an edit not sent for that same reason.
function saidElsewhere(error) {
  return (
    error instanceof NotSent ||
    (error instanceof ApiError && (error.status === 401 || error.code === PROJECT_CHANGED))
  );
}

function report(error) {
  if (saidElsewhere(error)) {
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
// undos queue up). Not once the page shows another project than when it was
// made (one opened in another tab, shown while the queue ran): sent then, it
// would name that project's opening and be made there. Gives its answer;
// rejects with NotSent if not sent.
function ordered(sendIt) {
  const made = shownOpening();
  const promise = proteinPanel.queue.then(() => {
    if (shownOpening() !== made) {
      throw new NotSent();
    }
    return sendIt();
  });
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
// is shown by then): the server's reason, or as `refused(error)` shows it.
async function edit(method, path, json, { refused = report } = {}) {
  const opened = shownOpening();
  try {
    return await ordered(() => send(method, path, json));
  } catch (error) {
    if (opened === shownOpening()) {
      refused(error);
    }
    throw error;
  }
}

const view = new ImageView($("view"), {
  place: (x, y, options) =>
    placeBox(x, y, options, options.laneIndex, options.proteinId || state.proteinId),
  row: (rect, proteinId) => placeRow(rect, proteinId),
  move: (boxId, rect) =>
    edit(
      "PUT",
      `/api/boxes/${boxId}`,
      { rect },
      { refused: (error) => showBoxRefusal(error, "Box not moved", "move it again") },
    )
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
  remeasured: (answer) => remeasuredText(answer),
});

// Its edits run in the panel's queue: in order with the protein edits, the
// undos and redos, and before the box edits made after them.
const laneTable = new LaneTable({
  queueEdit: (task, options) => proteinPanel.queueEdit(task, options),
  pending,
  send,
  status: showStatus,
});

// Read the project again and show it: only if the answer is about the opening
// shown, and not while another project is being opened (openProject shows
// that one).
async function reread() {
  const answer = await call("GET", "/api/project");
  if (opening === null && sameOpening(answer)) {
    applyAnswer(answer);
  }
}

// Each chart's drawing is fetched with the token ("Updating…" counts it until
// it arrives or no card awaits it), at a low priority: an edit waiting for a
// connection goes before the drawings waiting with it. One the server no
// longer keeps is asked for again after the project is read again (reread).
const charts = new ChartCards({
  fetch: (path, signal) =>
    dock.track(
      request("GET", path, { signal, priority: "low" }).then((response) => response.blob()),
      { chart: true },
    ),
  reread,
});

// --- Projects ---

async function showProjects() {
  const dialog = $("projects-dialog");
  $("projects-error").textContent = "";
  const listing = await call("GET", "/api/projects");
  $("projects-root").textContent = `Projects are saved in ${listing.root}`;
  $("projects-empty").hidden = listing.projects.length > 0;
  const list = $("project-list");
  list.replaceChildren();
  for (const entry of listing.projects) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.name;
    button.addEventListener("click", () =>
      openProject("/api/projects/open", entry.name, { name: entry.name }),
    );
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
let opening = null; // what is being opened ("Opening …" names it), or null

function setOpening(name) {
  opening = name;
  const dialog = $("projects-dialog");
  dialog.setAttribute("aria-busy", String(name !== null));
  const line = $("projects-state");
  line.textContent = name === null ? "" : `Opening ${name}…`;
  line.hidden = name === null;
}

// Show the project `answer` is about in place of the one shown: one this page
// created or opened (switchTo), or one opened in another tab (followOpening).
// Image ids repeat across projects (img-1 in each): drop everything shown.
// The edits still queued were made in the project shown before (an undo
// queued behind a refusal answered late): none runs now, since each would name
// the opening of this one and be made here (ordered checks its own).
function showOpened(answer) {
  proteinPanel.invalidateEdits();
  view.setImage(null, 0, 0);
  view.setOverlay([], [], null);
  state.shownImageId = null;
  for (const id of [...state.bitmaps.keys()]) {
    forgetBitmap(id);
  }
  state.originalColours.clear();
  proteinPanel.forgetTyped();
  laneTable.forgetTyped();
  charts.forget(); // its object URLs revoked: chart URLs repeat across projects too
  $("lane-picker").hidden = true; // its retry places a box in the project it asked about
  applyAnswer(answer, { choose: { imageId: null, proteinId: null, boxId: null } });
  lastBoxStep = null; // log numbers repeat across projects
}

// Create or open a project: POST `json` (no body if undefined) to `path`,
// then show the project answered in place of the one shown (showOpened).
// `name` is what "Opening …" says meanwhile. With `closes`, the request names
// that opening, which it closes (request()): refused if another project is
// open now (an import of images handed to Proteia, whose dialog says what it
// closes). Gives the answer once the project is shown; null if a create or
// open is already running here, or if the answer is older than the project
// shown by then. Rejects with the server's refusal, the project shown
// unchanged.
async function switchTo(path, name, json, { closes = null } = {}) {
  if (opening !== null) {
    return null;
  }
  setOpening(name);
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
    // Its answer is of the opening it makes: not taken for a newer one than
    // the request named (request()).
    const answer = await call("POST", path, json, { anyProject: true, closes });
    if (!isCurrent(answer.project)) {
      return null;
    }
    showOpened(answer);
    return answer;
  } finally {
    setOpening(null);
  }
}

// Create or open a project from the Projects dialog (switchTo), which closes
// once the project is shown; what went wrong is said in the dialog. Gives the
// answer once the project is shown, or null.
async function openProject(path, name, json) {
  if (opening !== null) {
    return null;
  }
  $("projects-error").textContent = "";
  try {
    const answer = await switchTo(path, name, json);
    if (answer === null) {
      $("projects-error").textContent = "Another project was opened meanwhile.";
      return null;
    }
    $("projects-dialog").close();
    showStatus("");
    return answer;
  } catch (error) {
    $("projects-error").textContent = error.message;
    return null;
  }
}

// --- A project opened in another tab ---

// A request refused because it named an opening no longer open (`error`,
// project_changed, whose detail names the open one). Unless this page is
// creating or opening a project itself (the reads it sent before, such as
// previews and charts, come back refused, and it shows the project it
// opened), or the refusal names an opening no newer than the one shown (a
// request sent before the page showed it, answered after: a read sent before
// this page's own switch, say), the page follows: it shows the project open
// now. The edits queued meanwhile are dropped, and with them what they would
// say: each would be refused. `method` and `path`: the request's, which says
// what was not done (NOT_DONE): nothing for a read, or a reveal (a folder not
// shown); otherwise by its path (REFUSED_AS), or a change. Nothing either if
// `answered`: the request was answered, about an opening newer than the one
// it named (request()), so it was not refused.
function projectChanged(error, method, path, { answered = false } = {}) {
  const now = error.detail && error.detail.open_id;
  const shown = shownOpening();
  if (opening !== null || !shown || !(now > shown)) {
    return;
  }
  proteinPanel.invalidateEdits();
  const refused =
    answered || method === "GET" || path === "/api/project/reveal"
      ? null
      : REFUSED_AS[path] || "change";
  followOpening({ refused });
}

// What a request refused as project_changed did not do, in the status line.
// An undo or redo lost no change: the one it would take back or make again was
// made, and is saved with the project.
const NOT_DONE = {
  change: "Your last change was not made.",
  undo: "Nothing was undone.",
  redo: "Nothing was redone.",
  export: "Nothing was exported.",
};

// The requests other than a change, by path: what their refusal did not do (a
// key of NOT_DONE).
const REFUSED_AS = {
  "/api/undo": "undo",
  "/api/redo": "redo",
  "/api/export": "export",
};

let following = null; // the follow under way: {refused, notes, done}, or null

// Show the project open now in place of the one shown, which is no longer
// open, and say why in the status line, and what the refused requests did not
// do (`refused`: a key of NOT_DONE, or null), then `note` (what else the page
// found, or null). First every edit made before has its answer (each is
// refused: none is made in the project open now); then the project is read,
// whichever it is, and shown with nothing kept of the one before
// (showOpened), unless this page has meanwhile started its own create or open
// (it shows that one) or the answer is no newer than what it shows. A call
// while one runs joins it. Settles once done, and never rejects.
function followOpening({ refused = null, note = null } = {}) {
  if (following) {
    if (refused) {
      following.refused.add(refused);
    }
    if (note) {
      following.notes.push(note);
    }
    return following.done;
  }
  const follow = {
    refused: new Set(refused ? [refused] : []),
    notes: note ? [note] : [],
    done: null,
  };
  following = follow;
  follow.done = (async () => {
    try {
      const before = state.project;
      await Promise.all([proteinPanel.settled(), pending()]);
      proteinPanel.invalidateEdits();
      const answer = await call("GET", "/api/project", undefined, { anyProject: true });
      const shown = shownOpening();
      if (opening !== null || !shown || !(answer.project.open_id > shown)) {
        return;
      }
      showOpened(answer);
      showStatus([followedText(before, answer.project, follow.refused), ...follow.notes].join(" "));
    } catch (error) {
      report(error);
    } finally {
      following = null;
    }
  })();
  return follow.done;
}

// Why the page now shows `project` in place of `before`, and what the
// requests `refused` did not do (keys of NOT_DONE).
function followedText(before, project, refused) {
  const now = isolate(project.name);
  const why =
    project.name === before.name
      ? `${now} was opened again, in another tab or after a change outside Proteia;` +
        " it is shown as it is now."
      : `Proteia now shows ${now}, opened in another tab; ${isolate(before.name)} is saved.`;
  return [why, ...[...refused].map((kind) => NOT_DONE[kind])].join(" ");
}

let checking = null; // the check under way (checkOpening), or null
let started = false; // the page has taken its first listing (start)

// Which project is open, and the images handed to Proteia waiting (GET
// /api/workspace): read by request(), as every answer is; about no project.
function readWorkspace() {
  return request("GET", "/api/workspace", { anyProject: true, answer: true });
}

// A page shown again (its tab chosen, its window focused) checks which project
// is open, and follows another opening (followOpening), so the user sees the
// project open now before editing. The header is what guarantees it: an edit
// sent before this answer is refused, not made in another project. It takes
// the images waiting too (takeListing): the Import dialog shows those found
// meanwhile, or closes once another tab imported or discarded the ones it
// shows. What the listing gives to say follows `note` (what the page just
// said, still in the status line) there, and both stay should it follow. One
// check at a time (a call meanwhile gives the one under way); none while this
// page creates or opens a project itself, its own import included. Settles
// once done, and never rejects.
function checkOpening({ note = null } = {}) {
  if (checking || !started || quitting || opening !== null) {
    return checking;
  }
  checking = readWorkspace()
    .then((workspace) => {
      const found = takeListing(workspace);
      const said = [note, found].filter(Boolean).join(" ") || null;
      if (found) {
        showStatus(said); // kept should the follow not show another project
      }
      const shown = shownOpening();
      if (opening === null && shown && workspace.open_id > shown) {
        return followOpening({ note: said });
      }
      return null;
    })
    .catch(() => {}) // Proteia stopped, say: the next action says so
    .finally(() => {
      checking = null;
    });
  return checking;
}

// Check again once the check under way (if any) is done: it may have been
// asked before what this page just did, whose answer it would not reflect.
function checkAgain(note = null) {
  return Promise.resolve(checking).then(() => checkOpening({ note }));
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    checkOpening();
  }
});
window.addEventListener("focus", () => checkOpening());

// --- Images handed to Proteia (#57) ---

// A launch given image files (`proteia a.tif b.tif`, or "Open with Proteia")
// hands them to the running Proteia, where they wait (a hand-off) until a
// page imports them into a new project or discards them: kind, membrane and
// polarity have no default, so the page asks, in the Import dialog
// (handoffs.js). The page finds them at start, on each check (a page shown
// again), and before Quit, and shows one at a time.

// How often the listing is read again while the hand-off shown may still grow.
const SETTLE_MS = 2000;
const GONE = "These images were imported or discarded in another tab.";
const BEING_IMPORTED = "These images are being imported in another tab.";
const WAITING_AT_QUIT = "Import or discard the images waiting in Proteia before quitting.";
const NOT_RESPONDING = "Proteia is not responding. Start it again to reopen this page.";

// The image hand-offs waiting, as last listed (takeListing).
let handoffs = [];
// Hand-offs closed here with Esc or Later: not shown again unasked ("N
// images waiting…" shows them).
const setAside = new Set();
// Hand-offs this page imported or discarded, and notices it said: a listing
// asked for before that answer may still list them.
const finished = new Set();
let settling = null; // the timer of the next read while the hand-off shown may grow
// The project open as last listed ({name, open_id}), or null for none: what an
// import closes while this page shows no project (importCloses).
let listedOpen = null;

const importDialog = new ImportDialog({
  accept: () => importHandoff(),
  discard: () => discardHandoff(),
  closed: (handoff) => {
    setAside.add(handoff.id);
    stopSettling();
    renderWaiting();
    showOpenOrProjects();
  },
});

// Whether a dialog is up: Projects, a chart enlarged, or Import.
function dialogUp() {
  return [...document.getElementsByTagName("dialog")].some((dialog) => dialog.open);
}

// What an import of images waiting closes, as the Import dialog says and its
// request names: the project shown; with none shown, the project open as last
// listed (another tab opened it), or null for none.
function importCloses() {
  if (state.project) {
    return { name: state.project.name, open_id: shownOpening() };
  }
  return listedOpen;
}

// Take a listing of the workspace (GET /api/workspace). The hand-off the
// Import dialog shows is refreshed as listed now, or, no longer listed
// (another tab imported or discarded it), the dialog closes, and this gives
// that to say; not while this page's own import or discard of it is awaited
// (its claim hides it). Its line saying what an import closes follows the
// project open (importCloses). Then, with no dialog up, the first hand-off
// not set aside is shown (`show` "auto"); asked for (Quit, "N images
// waiting…"), the first even if set aside, over another dialog ("asked").
// With no project shown and no dialog up, the page shows the project open, or
// the Projects dialog. Gives what to say, or null: the dialog closed, then the
// notices (sayNotices); the caller says it with its own message, never over
// it. Notices wait while this page's own import or discard is awaited, whose
// answer the status line says next: the check after it says them.
function takeListing(workspace, { show = "auto" } = {}) {
  const listed = workspace.handoffs.filter((handoff) => !finished.has(handoff.id));
  const notices = importDialog.busy ? [] : listed.filter((handoff) => handoff.kind === "notice");
  handoffs = listed.filter((handoff) => handoff.kind === "images");
  listedOpen =
    workspace.open_id === null ? null : { name: workspace.open, open_id: workspace.open_id };
  for (const id of [...setAside]) {
    if (!handoffs.some((handoff) => handoff.id === id)) {
      setAside.delete(id);
    }
  }
  const shown = importDialog.handoff;
  let gone = null;
  if (shown && !importDialog.busy) {
    const now = handoffs.find((handoff) => handoff.id === shown.id);
    if (now) {
      importDialog.refresh(now);
    } else {
      importDialog.hide();
      gone = GONE;
    }
  }
  if (importDialog.handoff) {
    importDialog.renderCloses(importCloses());
  }
  if (!importDialog.handoff && (show === "asked" || !dialogUp())) {
    const next = handoffs.find((handoff) => show === "asked" || !setAside.has(handoff.id));
    if (next) {
      setAside.delete(next.id);
      importDialog.show(next, importCloses());
    }
  }
  renderWaiting();
  keepSettling();
  if (!state.project && !dialogUp()) {
    showOpenOrProjects(workspace);
  }
  return [gone, sayNotices(notices)].filter(Boolean).join(" ") || null;
}

// Notices: the arguments launches could not open, and no image. Gives what
// they say, for the status line (null for none), and discards each as said
// (its count of entries), so no other page says it again: one that grew
// meanwhile is refused (handoff_changed) and said whole at the next check; one
// another page discarded first was said there.
function sayNotices(notices) {
  if (!notices.length) {
    return null;
  }
  const said = notices.map((notice) => refusedText(notice.refused, notice.more_refused));
  for (const notice of notices) {
    finished.add(notice.id); // said: not again, whatever the next listing
    request("POST", `/api/handoffs/${notice.id}/discard`, {
      anyProject: true,
      json: { files: [], refused: notice.refused.length + notice.more_refused },
    }).catch((error) => {
      if (error.code === "handoff_changed") {
        finished.delete(notice.id);
        checkAgain();
      }
    });
  }
  return `Not opened: ${said.join("; ")}.`;
}

// "N images waiting…", in the header and in the Projects dialog, while images
// wait that the Import dialog does not show; either shows them.
function renderWaiting() {
  const shown = importDialog.handoff;
  const count = handoffs
    .filter((handoff) => !shown || handoff.id !== shown.id)
    .reduce((sum, handoff) => sum + handoff.files.length, 0);
  for (const id of ["handoffs-waiting", "projects-handoffs"]) {
    const button = $(id);
    button.hidden = !count || quitting;
    button.textContent = `${counted(count, "image", "images")} waiting…`;
  }
}

// Show the images waiting, asked for: the listing is read again first.
async function showWaiting() {
  try {
    const found = takeListing(await readWorkspace(), { show: "asked" });
    if (found) {
      showStatus(found);
    }
  } catch (error) {
    report(error);
  }
}

$("handoffs-waiting").addEventListener("click", () => showWaiting());
$("projects-handoffs").addEventListener("click", () => showWaiting());

// While the hand-off shown may still grow (more_may_arrive: the launches of a
// selection opened with Proteia, one per file, are still handing theirs off),
// the listing is read again every SETTLE_MS until it may not; not while this
// page's own import or discard of it is awaited.
function keepSettling() {
  const shown = importDialog.handoff;
  if (settling !== null || !shown || !shown.more_may_arrive || importDialog.busy) {
    return;
  }
  settling = window.setTimeout(() => {
    settling = null;
    checkOpening();
  }, SETTLE_MS);
}

function stopSettling() {
  window.clearTimeout(settling);
  settling = null;
}

// With no project shown and no dialog up (the Import dialog closed, its
// images imported in another tab, say), show the project open now, as a
// start does, or the Projects dialog to open one. `workspace`: a listing just
// read, or null to read one.
async function showOpenOrProjects(workspace = null) {
  if (state.project || dialogUp() || opening !== null || quitting) {
    return;
  }
  try {
    const now = workspace || (await readWorkspace());
    if (state.project || dialogUp() || opening !== null) {
      return;
    }
    if (now.open) {
      applyAnswer(await call("GET", "/api/project"));
    } else {
      await showProjects();
    }
  } catch (error) {
    report(error);
  }
}

// "the image", "the 3 images": a hand-off's files, in a sentence.
function theImages(handoff) {
  const count = handoff.files.length;
  return count === 1 ? "the image" : `the ${count} images`;
}

// Whether the Import dialog still shows `handoff`.
function showsHandoff(handoff) {
  return importDialog.handoff !== null && importDialog.handoff.id === handoff.id;
}

// Import the hand-off the dialog shows (its Import): a new project with its
// images, shown in place of the project shown, as this page's own switch
// (switchTo: no word of another tab). The request names the opening the
// dialog says the import closes (importCloses): refused if another tab opened
// a project since, and the page follows before the user presses Import again.
// A page showing no project knows the project open only from its last
// listing, so it reads the listing again first, and asks again should a
// project be open that the dialog did not name. (With none open then, the
// request names none: only a project another tab opens between that read and
// the import is closed unnamed.)
async function importHandoff() {
  const handoff = importDialog.handoff;
  if (!handoff || !importDialog.ready() || opening !== null || exporting || quitting) {
    return;
  }
  const choices = importDialog.choices();
  const images = counted(handoff.files.length, "image", "images");
  stopSettling();
  importDialog.setBusy(`Importing ${images}…`);
  if (!state.project) {
    const named = importDialog.closes;
    await checkAgain();
    const now = importDialog.closes;
    if (now && (!named || now.open_id !== named.open_id)) {
      importDialog.setBusy(null);
      askAgain(now.name);
      return;
    }
  }
  const closes = importDialog.closes;
  let answer;
  try {
    const name = choices.name || handoff.suggested_name || "the new project";
    answer = await switchTo(`/api/handoffs/${handoff.id}/accept`, name, choices, {
      closes: closes ? closes.open_id : null,
    });
  } catch (error) {
    importDialog.setBusy(null);
    await importRefused(error, handoff);
    return;
  }
  finished.add(handoff.id);
  if (showsHandoff(handoff)) {
    importDialog.hide();
  }
  const projects = $("projects-dialog");
  if (projects.open) {
    projects.close();
  }
  const said = answer
    ? importedText(answer)
    : `Imported ${images} into a new project, but another project was opened meanwhile.`;
  showStatus(said);
  checkAgain(said); // the next images waiting, if any
}

// The Import dialog asks again: another tab opened `name`, which an import
// closes now (as its line says).
function askAgain(name) {
  importDialog.say(`Another tab opened ${isolate(name)}: check, then press Import again.`);
  keepSettling();
  if (focusLost()) {
    importDialog.focusBack();
  }
}

// What an import of a hand-off did, for the status line: the images imported
// and into which project; what each import found about its file, as a page
// import says it; the files not imported, and why; the arguments the launches
// could not open; and the notes (an image put on a new membrane because the
// one it was to join was not imported).
function importedText(answer) {
  const done = answer.handoff;
  const images = counted(done.imported.length, "image", "images");
  const parts = [`Imported ${images} into ${isolate(answer.project.name)}.`];
  for (const file of done.imported) {
    const image = answer.project.images.find((each) => each.id === file.image_id);
    const found = image ? image.warnings.map((warning) => warning.message) : [];
    if (found.length) {
      parts.push(`${isolate(file.name)}: ${found.join(" ")}`);
    }
  }
  if (done.refused.length) {
    parts.push(`Not imported: ${refusedText(done.refused)}.`);
  }
  if (done.launch_refused.length || done.more_refused) {
    parts.push(`Not opened: ${refusedText(done.launch_refused, done.more_refused)}.`);
  }
  parts.push(...done.notes.map((note) => `${note}.`)); // each starts with a file name, as it is
  return parts.join(" ");
}

// An import refused (`error`). The server changed nothing, unless the images
// went into a project after all, or none could be imported: then the
// hand-off is gone, the dialog closes, and the status line says what came of
// it. Otherwise the dialog stays up and says what to do (handoffRefused).
async function importRefused(error, handoff) {
  const detail = error.detail || {};
  if (error.code === "nothing_imported" || (error.code === "unsaved_changes" && detail.created)) {
    finished.add(handoff.id);
    if (showsHandoff(handoff)) {
      importDialog.hide();
    }
    const said =
      error.code === "nothing_imported"
        ? `Nothing was imported: ${refusedText(detail.refused || [])}.`
        : sentence(error.message);
    showStatus(said);
    checkAgain(said);
    return;
  }
  if (error.code === PROJECT_CHANGED) {
    // Another tab opened a project, maybe by importing these very images. The
    // page checks (checkOpening; this page's switch had ended, so the refusal
    // started no follow): it shows the project open now, and closes the
    // dialog if these images are gone. If they still wait, the dialog says
    // what an import closes now.
    await checkAgain();
    if (showsHandoff(handoff)) {
      askAgain(detail.open);
    }
    return;
  }
  if (error.code === "project_exists" || error.code === "invalid_project_name") {
    if (showsHandoff(handoff)) {
      importDialog.sayName(sentence(error.message));
      keepSettling();
      $("handoff-name").focus();
      return;
    }
  }
  handoffRefused(error, handoff, "Import");
}

// Discard the hand-off the dialog shows (its Discard): the server drops its
// copies of the images; the original files are never touched. It sends the
// files and refused entries shown, so one that grew since is refused
// (handoff_changed), and the dialog shows it grown.
async function discardHandoff() {
  const handoff = importDialog.handoff;
  if (!handoff || importDialog.busy) {
    return;
  }
  const images = counted(handoff.files.length, "image", "images");
  stopSettling();
  importDialog.setBusy(`Discarding ${images}…`);
  try {
    await request("POST", `/api/handoffs/${handoff.id}/discard`, {
      anyProject: true,
      json: importDialog.shownParts(),
    });
  } catch (error) {
    importDialog.setBusy(null);
    handoffRefused(error, handoff, "Discard");
    return;
  }
  finished.add(handoff.id);
  if (showsHandoff(handoff)) {
    importDialog.hide();
  }
  const said = `Discarded ${theImages(handoff)} waiting; the original files are unchanged.`;
  showStatus(said);
  checkAgain(said); // the next images waiting, if any
}

// An import or discard (`press`: its button) refused, nothing changed. Once
// another tab took the hand-off (handoff_claimed: it is importing it;
// handoff_not_found: imported or discarded), the dialog closes, says so, and
// the page checks again: the next images waiting, or the project that import
// opened. Once more joined it (handoff_changed), the dialog shows it as it is
// now, each new image with its bands unchosen, and says what joined (images,
// or only files a launch could not open), to press again. Anything else is
// said in the dialog, which stays up.
function handoffRefused(error, handoff, press) {
  if (error.code === "handoff_claimed" || error.code === "handoff_not_found") {
    if (error.code === "handoff_not_found") {
      finished.add(handoff.id);
    }
    if (showsHandoff(handoff)) {
      importDialog.hide();
    }
    const said = error.code === "handoff_claimed" ? BEING_IMPORTED : GONE;
    showStatus(said);
    checkAgain(said);
    return;
  }
  if (!showsHandoff(handoff)) {
    report(error); // closed meanwhile
    return;
  }
  if (error.code === "handoff_changed") {
    const joined = importDialog.refresh(error.detail);
    const what = joined.images
      ? "More images arrived: check them"
      : joined.refused
        ? "More files could not be opened: check the list"
        : "These images changed: check them";
    importDialog.say(`${what}, then press ${press} again.`);
  } else if (error instanceof ApiError) {
    importDialog.say(error.status === 401 ? NEEDS_LAUNCH : sentence(error.message));
  } else {
    importDialog.say(NOT_RESPONDING);
  }
  keepSettling();
  if (focusLost()) {
    importDialog.focusBack();
  }
}

// "Open sample project": a new project on the synthetic sample blot, with its
// lanes and proteins set up and each protein's row left to drag
// (sample_project.py on the server). The status line says what to do next.
// The first-use tour will start here: the answer's sample.rows gives each
// protein's row, top to bottom, as a drag over it, for the tour to point at.
async function openSample() {
  const answer = await openProject("/api/projects/sample", "the sample project");
  if (answer === null || !sameOpening(answer)) {
    return;
  }
  const names = answer.sample.rows
    .map((row) => answer.project.proteins.find((p) => p.id === row.protein_id))
    .filter(Boolean)
    .map((protein) => protein.name);
  showStatus(
    "For each protein, choose it on the left and drag across its row to box its bands" +
      ` (rows from the top: ${names.join(", ")}); the table and charts fill in.`,
  );
}

$("new-project").addEventListener("submit", (event) => {
  event.preventDefault();
  const name = $("new-project-name").value;
  openProject("/api/projects", name, { name });
});
$("open-sample").addEventListener("click", () => openSample());
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

// The file names of the images `ids` names (a refusal's), set apart for
// bidirectional text; one not in the project shown (an image an undo would
// bring back) is left out.
function imageNames(ids) {
  return ids
    .map((id) => state.project.images.find((image) => image.id === id))
    .filter(Boolean)
    .map((image) => isolate(image.original_name));
}

function render() {
  $("lane-picker").hidden = true; // its question was about the state before
  const project = state.project;
  importDialog.renderCloses(importCloses()); // what an import of images waiting closes
  $("workspace").hidden = !project;
  $("switch-project").hidden = false;
  $("reveal").hidden = !project;
  $("export").hidden = !project;
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
  proteinPanel.render(project, image, state.proteinId, state.results);
  renderBox(project);
  renderNotices(project);
  renderHint(project, image);
  renderColours(image);
  renderView(project);
  laneTable.render(project, state.results);
  charts.render(state.results);
  dock.render(state.results);
  renderRequantify(project);
  renderExport(project);
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

// A box's over-exposure: the check at the detector limit, or on an image it
// cannot trust (lossy, colour or converted), how near the limit its pixels come
// (#112).
function overExposureText(band) {
  if (band.clipped !== null) {
    return band.clipped
      ? "Yes: pixels at the detector limit, so the net is an under-estimate"
      : "No";
  }
  if (band.possibly_clipped === true) {
    return "Possibly: pixels near the detector limit, which this image cannot confirm; check the original capture";
  }
  if (band.possibly_clipped === false) {
    // Not "No": a colour channel saturated alone may never bring the grey mean
    // near the limit, so the heuristic seeing nothing does not clear the band.
    return "Not checked; no sign of it in the grey analysis image";
  }
  return "Not checked";
}

// What a box's label adds about its over-exposure; a question mark where it
// cannot be confirmed (#112).
function overExposureLabel(band) {
  if (band.clipped === true) {
    return " over-exposed";
  }
  return band.possibly_clipped === true ? " over-exposed?" : "";
}

function renderBox(project) {
  const found = state.boxId ? findBox(project, state.boxId) : null;
  $("box-panel").hidden = !found;
  if (!found) {
    return;
  }
  const { protein, band } = found;
  $("box-summary").textContent = `${protein.name}, ${laneName(project, band.lane_index)}`;
  $("box-clipped").textContent = overExposureText(band);
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
  // A notice of both sets (each keeps its charts' test notices) is listed once.
  const [applied] = state.results.sets;
  const said = (notice) => JSON.stringify([notice.code, notice.message]);
  const listed = new Set(applied.notices.map(said));
  for (const set of state.results.sets) {
    const prefix = set.id === "all_lanes" ? "All lanes: " : "";
    for (const notice of set.notices) {
      if (set !== applied && listed.has(said(notice))) {
        continue;
      }
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

// The preview of each colours: the grey analysis image, or the stored file's
// original colours (the server answers the grey one for a file without colour).
const PREVIEW_QUERY = { grey: "", original: "?colour=original" };

// One fetch per preview, shared by every render that waits for it.
function bitmapOf(imageId, colours) {
  if (!state.bitmaps.has(imageId)) {
    state.bitmaps.set(imageId, {});
  }
  const previews = state.bitmaps.get(imageId);
  if (!previews[colours]) {
    const pending = request("GET", `/api/images/${imageId}/preview${PREVIEW_QUERY[colours]}`)
      .then((response) => response.blob())
      .then((blob) => createImageBitmap(blob));
    pending.catch(() => {
      if (previews[colours] === pending) {
        delete previews[colours]; // the next render tries again
      }
    });
    previews[colours] = pending;
  }
  return previews[colours];
}

// The colours the view shows `image` in: "original" while its "Original
// colours" is on (offered only for a file with colour), else "grey".
function coloursOf(image) {
  return image.colour && state.originalColours.has(image.id) ? "original" : "grey";
}

function renderColours(image) {
  const button = $("original-colours");
  button.hidden = !image || !image.colour;
  button.setAttribute("aria-pressed", String(Boolean(image) && coloursOf(image) === "original"));
}

// The image whose colours were just switched: the view says which it shows
// once it shows them (for a screen reader: the switch may be the C key).
let switchedColours = null;

// "Original colours" (the button, or C): the chosen image in the stored file's
// own colours, or back to the grey analysis image. The view keeps its zoom.
function toggleColours() {
  const project = state.project;
  const image = project && project.images.find((i) => i.id === state.imageId);
  if (!image || !image.colour || $("workspace").hidden) {
    return;
  }
  if (!state.originalColours.delete(image.id)) {
    state.originalColours.add(image.id);
  }
  switchedColours = image.id;
  renderColours(image);
  renderView(project);
}

$("original-colours").addEventListener("click", () => toggleColours());

function sayColours(image, colours) {
  const name = isolate(image.original_name);
  $("view-colours-state").textContent =
    colours === "original"
      ? `${name} in its original colours, for display only`
      : `${name} as the grey analysis image, which the nets are measured on`;
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
    // The chosen protein's padded boxes show their fitted size inside. Not a
    // box at the image's edge: the padding may have shifted it inward, off
    // the fit's centre, and the outline would be drawn off the fit.
    const { across, along } = protein.box_padding;
    const inset = protein.id === state.proteinId && (across > 0 || along > 0);
    for (const band of protein.bands) {
      const [x0, y0, x1, y1] = band.rect;
      const atEdge = x0 <= 0 || y0 <= 0 || x1 >= image.width || y1 >= image.height;
      boxes.push({
        id: band.id,
        rect: band.rect,
        fitted: inset && !atEdge ? [x0 + across, y0 + along, x1 - across, y1 - along] : null,
        color,
        clipped: band.clipped === true,
        label: `${band.lane_index + 1}${overExposureLabel(band)}`,
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
  const colours = coloursOf(image);
  if (state.shownImageId === image.id) {
    // Shown already: its boxes are drawn now, over the bitmap shown, not once the
    // preview in these colours has loaded (both cover the same pixels), so the
    // view never shows, nor takes a click on, a box the server no longer has.
    view.setOverlay(boxes, ghosts, state.boxId, marks);
  }
  bitmapOf(image.id, colours)
    .then((bitmap) => {
      if (generation !== renderGeneration) {
        return; // a later render owns the view
      }
      if (state.shownImageId === image.id) {
        view.setBitmap(bitmap); // this image, maybe in other colours: the zoom stays
      } else {
        view.setImage(bitmap, image.width, image.height);
        view.setOverlay(boxes, ghosts, state.boxId, marks);
        state.shownImageId = image.id;
      }
      const switched = switchedColours === image.id;
      switchedColours = null;
      if (switched) {
        sayColours(image, colours);
      }
    })
    .catch((error) => {
      if (
        colours !== "original" ||
        generation !== renderGeneration ||
        (error instanceof ApiError && error.code === PROJECT_CHANGED)
      ) {
        report(error);
        return;
      }
      // Not shown in its colours (its file changed outside Proteia, say): the
      // switch goes back off, and the grey analysis image is shown.
      state.originalColours.delete(image.id);
      switchedColours = null;
      renderColours(image);
      renderView(state.project);
      if (!(error instanceof ApiError && error.status === 401)) {
        const name = isolate(image.original_name);
        showStatus(`${name} not shown in its original colours: ${sentence(error.message)}`);
      }
    });
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
    } else if (error.code === "overlap") {
      showBoxRefusal(error, "Box not placed", "click again");
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

// Why a row left a lane with neither a box nor an n.d. mark, by the
// detector's reason for the empty lane (LaneReason in core/rowdetect.py). A
// lane where no band reaches the detection limit (no_band) gets an n.d. mark,
// unless its place lies outside the row box, or rests on the one band the row
// found (the answer's unlocated_lanes, worded apart).
const NOT_MEASURED = {
  artefact: "a stain or streak",
  line: "a line or strip across the lanes",
  edge_signal: "only signal at the row box's top or bottom edge",
  side_signal: "only signal at the row box's left or right edge",
  unassigned: "signal that fits no lane",
  no_band: "outside the row box",
};

// The detector's warnings about a row it placed (WARNING_FLAGS in
// core/rowdetect.py), in words. `note`: words of the detector's note on it,
// which names its lanes first ("lane 5: …", "lanes 3, 7: …").
const ROW_WARNINGS = {
  doubtful_lanes: {
    note: null,
    words: () => "the bands' spacing does not fit the lanes read",
  },
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
  cut_by_row_box: {
    note: "cuts through the band",
    words: (lanes) => {
      if (!lanes) {
        return "the row box cuts through a band; include the whole band";
      }
      const bands = lanes.startsWith("lanes") ? "bands" : "band";
      return `the row box cuts through the ${bands} in ${lanes}; include the whole ${bands}`;
    },
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

// The other proteins on the image whose nets a row changed, in words, or
// null: every ring leaves out every box on its image, so a new row moves the
// local background, and the net, of the boxes already there.
function remeasuredText(answer) {
  const found = answer.remeasured
    .map((entry) => findBox(answer.project, entry.band_id))
    .filter(Boolean);
  if (!found.length) {
    return null;
  }
  const names = [...new Set(found.map((box) => box.protein.name))];
  const who =
    names.length === 1 ? `${names[0]} on this image was` : "other proteins on this image were";
  const largest = answer.largest_change;
  const where = largest && findBox(answer.project, largest.band_id);
  if (!where) {
    return `${who} re-measured`;
  }
  const percent = largest.change * 100;
  const size = percent < 1 ? "under 1%" : `${Math.round(percent)}%`;
  const whose = names.length === 1 ? "" : ` (${where.protein.name})`;
  return `${who} re-measured; largest change ${size} in lane ${where.band.lane_index + 1}${whose}`;
}

// What a row did, lane by lane, from its answer (lanes numbered from 1): the
// boxes placed, the lanes with no band (n.d.), those kept as they were, those
// not measured and why, the boxes an earlier row placed that went, the
// detector's warnings and the other proteins it re-measured. `before`: the
// state shown before, where those boxes are.
// Gives {text, check, unchanged}: `unchanged` when the row changed nothing (the
// same drag again), `check` when it changed the project and left signal that
// fits no lane, the mark of a row box over part of the row (its bands then
// read as several lanes each), or read lanes that do not fit the bands'
// spacing (doubtful_lanes, #111: a first row box that also covers a ladder,
// labels or another panel reads its lanes off by one or more): the text asks
// to check the lane numbers.
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
  const unlocated = new Set(answer.unlocated_lanes);
  const unmeasured = new Map(); // reason -> lanes
  for (const lane of answer.unmeasured_lanes.filter((index) => !unlocated.has(index))) {
    const reason = empty.has(lane) ? empty.get(lane).reason : "";
    unmeasured.set(reason, [...(unmeasured.get(reason) || []), lane]);
  }
  for (const [reason, lanes] of unmeasured) {
    parts.push(`${lanesPhrase(lanes)} not measured: ${NOT_MEASURED[reason] || reason}`);
  }
  if (unlocated.size) {
    parts.push(
      `${lanesPhrase([...unlocated])} not recorded: one band cannot show where the other` +
        " lanes lie",
    );
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
  const remeasured = remeasuredText(answer);
  if (remeasured) {
    parts.push(remeasured);
  }
  const partRow = unmeasured.has("unassigned");
  const check = !unchanged && (partRow || answer.flags.includes("doubtful_lanes"));
  if (check) {
    const all = answer.band_ids.length;
    parts.push(
      partRow
        ? "check the boxes' lane numbers: a row box over part of the row misreads the lanes" +
            ` (Undo, then drag across all ${all})`
        : "check the boxes' lane numbers: a row box that also covers a ladder, labels or" +
            ` another panel misreads the lanes (Undo, then drag over the ${all} lanes only)`,
    );
  } else if (unmeasured.size) {
    parts.push("click a dashed placeholder to box a lane by hand");
  }
  return { text: parts.join(" · "), check, unchanged };
}

// What to do about a refused row, by the refusal's code. A row the detector
// saw bands in (a refusal whose detail names another cause than no_band)
// already says what to do.
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

// The boxes a refusal names (an overlap's: those in the way) in words, protein
// by protein as the project holds them: "the box of GAPDH in lane 2", "the
// boxes of GAPDH in lanes 1, 2 and 3"; null when it names none shown.
function namedBoxes(error) {
  const lanes = new Map(); // protein -> the lane indices of its boxes named
  for (const id of error.ids) {
    const found = findBox(state.project, id);
    if (found) {
      lanes.set(found.protein, [...(lanes.get(found.protein) || []), found.band.lane_index]);
    }
  }
  const parts = [...lanes].map(([protein, indices]) => {
    const boxes = indices.length === 1 ? "box" : "boxes";
    return `the ${boxes} of ${protein.name} in ${lanesPhrase(indices.sort((a, b) => a - b))}`;
  });
  return parts.length ? inWords(parts) : null;
}

// Show a box on its image, selected: the user moves or deletes it from there.
function selectBox(boxId) {
  const found = state.project && findBox(state.project, boxId);
  if (!found) {
    return; // gone meanwhile (an undo)
  }
  state.imageId = found.protein.image_id;
  state.boxId = boxId;
  select();
  render();
}

// What to do about the boxes in the way of a box or a row (an overlap refusal,
// which names them: another box of the protein, or another protein's boxes it
// would overlap by more than half): move or delete them, then `again`. With an
// Undo of the last box change when it made one of them (it may be the
// mistake), else a Select of the first of them. Gives {text, action}; no text
// when it names none shown.
function boxesInTheWay(error, again) {
  const which = namedBoxes(error);
  if (!which) {
    return { text: null, action: null };
  }
  const text = `Move or delete ${which}, then ${again}.`;
  const step = state.project.history.undo;
  if (lastBoxStepNamed(error)) {
    const words = actionWords(step.action);
    return {
      text: `${text} If the last change (${words}) was the mistake, Undo takes it back.`,
      action: {
        label: "Undo",
        name: `Undo ${words}`,
        seq: step.seq,
        run: () => takeStep("undo", { seq: step.seq }),
      },
    };
  }
  const first = error.ids.map((id) => findBox(state.project, id)).find(Boolean);
  return {
    text,
    action: {
      label: "Select",
      name: `Select the box of ${first.protein.name} in lane ${first.band.lane_index + 1}`,
      run: () => selectBox(first.band.id),
    },
  };
}

// A box placed or moved refused in the status line: `head` (what was not
// done), the server's reason, and for boxes in its way what to do, with an
// Undo or Select of them (boxesInTheWay).
function showBoxRefusal(error, head, again) {
  if (saidElsewhere(error)) {
    return;
  }
  const sentences = [`${head}: ${sentence(error.message)}`];
  let action = null;
  if (error.code === "overlap") {
    const offer = boxesInTheWay(error, again);
    if (offer.text) {
      sentences.push(offer.text);
      action = offer.action;
    }
  }
  showStatus(sentences.join(" "), action);
}

// A refused row in the status line: the server's reason with the protein
// named, what to do, and, when the boxes already on the image did not let the
// row read its lanes just after a box was placed or changed, an Undo of that
// change; for boxes in its way, an Undo or Select of them (showBoxRefusal).
function showRowRefusal(error, name) {
  if (saidElsewhere(error)) {
    return;
  }
  if (error.code === "overlap") {
    showBoxRefusal(error, `Row box of ${name} not placed`, "drag again");
    return;
  }
  const sentences = [`Row box of ${name} not placed: ${sentence(error.message)}`];
  let action = null;
  if (error.code === "row_lanes_unclear") {
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
  } else if (ROW_HINTS[error.code] && !(error.detail && error.detail.cause !== "no_band")) {
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
      call("POST", `/api/images?${query}`, undefined, {
        body: file,
        contentType: "application/octet-stream",
      }),
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

// --- Requantifying: the local background, and looking for over-exposure ---

// The background method of a project quantified before the local background:
// each net above its image's median. It stays until the project is requantified.
const LEGACY_BACKGROUND = "global_median";

// What the offer says for each reason to requantify (requantifyReason): the
// legacy background, which moves every net, or boxes on a lossy, colour or
// CMYK image measured before Proteia looked for pixels near the detector limit
// there (#112). The note is its own tooltip too, whole where it is cut short.
const REQUANTIFY_OFFERS = {
  background: {
    label: "Requantify with local background",
    title:
      "The nets use the whole-image median background, as measured before the local background.",
    note: "Nets move to the local ring background; Undo takes it back.",
  },
  overExposure: {
    label: "Requantify",
    title:
      "Boxes on a lossy, colour or CMYK image were measured before Proteia looked for pixels" +
      " near the detector limit there.",
    note: "Looks for pixels near the detector limit; Undo takes it back.",
  },
};

let requantifying = false; // a press has no answer yet

// Whether any protein has a box on an image.
function hasBoxes(project) {
  return project.proteins.some((protein) => protein.bands.length);
}

// Why the project is offered a requantify (a REQUANTIFY_OFFERS key), or null:
// the legacy background while there are boxes to measure again (that
// requantify measures every box, so it looks near the limit too), else images
// whose boxes were never looked at near the limit (the server's
// unassessed_images, which a requantify assesses).
function requantifyReason(project) {
  if (project.background_method === LEGACY_BACKGROUND && hasBoxes(project)) {
    return "background";
  }
  return project.unassessed_images.length ? "overExposure" : null;
}

// The offer in the results' header, worded for its reason (the Checks say
// why); it goes once there is nothing left to requantify. In view whatever the
// side panel shows, next to the numbers it changes.
function renderRequantify(project) {
  const reason = requantifyReason(project);
  $("requantify-offer").hidden = reason === null;
  if (reason !== null) {
    const offer = REQUANTIFY_OFFERS[reason];
    const button = $("requantify");
    button.textContent = offer.label;
    button.title = offer.title;
    const note = $("requantify-note");
    note.textContent = offer.note;
    note.title = offer.note;
  }
  $("requantify").disabled = requantifying;
}

// Measure boxes again: every box against the local background, or those not
// yet looked at near the detector limit, in one change the status line offers
// to Undo. Like an undo or a clear, it runs in the panel's
// queue once the edits made before it have their answers (a box being placed
// is measured again too), so it never reaches a project opened after it was
// asked for, and the edits made after it wait for it. Pressed twice (a double
// click), it is sent once: the button stays disabled until the answer.
function requantify() {
  if (requantifying || !state.project || $("workspace").hidden) {
    return;
  }
  requantifying = true;
  $("requantify").disabled = true;
  const after = Promise.allSettled([pending(), proteinPanel.adding]);
  const asked = proteinPanel.queueEdit(
    async (current) => {
      const before = state.project;
      const opened = shownOpening();
      try {
        const answer = await send("POST", "/api/requantify");
        if (answer && current()) {
          showRequantified(answer, before);
        }
      } catch (error) {
        if (current() && opened === shownOpening()) {
          reportRequantify(error);
        }
      }
      return null;
    },
    { after },
  );
  const done = () => {
    requantifying = false;
    const button = $("requantify");
    button.disabled = false;
    if (!$("requantify-offer").hidden) {
      // Still offered (refused): the keyboard stays on it.
      if (focusLost()) {
        button.focus();
      }
    } else {
      // Gone: the keyboard is on the status line's Undo (showRequantified), or,
      // with none (nothing was done), goes on to Undo or Redo.
      keepFocus(button, "undo", null);
    }
  };
  asked.then(done, done);
}

$("requantify").addEventListener("click", requantify);

// What the requantify did, with an Undo of it that goes once the history moves
// on, worded for the offer the page showed. It did nothing if nothing was left
// to do: the revision shown did not move, or, once another tab requantified
// since this page showed the project, no image was measured again although it
// has boxes (a requantify of a legacy project measures every image with boxes,
// which leaves none to look at near the limit). Then the last change is not
// this one: no Undo.
function showRequantified(answer, before) {
  const project = answer.project;
  const count = answer.images.length;
  const background = before.background_method === LEGACY_BACKGROUND;
  if (
    (before.open_id === project.open_id && before.revision === project.revision) ||
    (!count && hasBoxes(project))
  ) {
    // Nothing logged.
    showStatus(
      background
        ? "The nets already use the local background."
        : "Every box has been looked at near the detector limit already.",
    );
    return;
  }
  const images = counted(count, "image", "images");
  const text = !background
    ? `Requantified ${images}: looked for pixels near the detector limit`
    : count
      ? `Requantified ${images} with the local background`
      : "Switched to the local background (no boxes to requantify)";
  const step = project.history.undo;
  const undo =
    step && step.action === "requantify"
      ? {
          label: "Undo",
          name: background ? "Undo requantifying with the local background" : "Undo requantifying",
          seq: step.seq,
          run: () => takeStep("undo", { seq: step.seq, back: () => $("requantify") }),
        }
      : null;
  const button = showStatus(text, undo);
  // The offer has gone: the keyboard goes on to that Undo. The hidden button
  // may still hold the focus until the browser moves it, as Clear boxes may.
  if (button && (focusLost() || document.activeElement === $("requantify"))) {
    button.focus();
  }
}

// A refusal changes nothing: the server's reason, with the images it names by
// their file names (a missing or changed image file).
function reportRequantify(error) {
  if (saidElsewhere(error)) {
    return;
  }
  const names = imageNames(error.ids || []);
  const which = names.length ? ` (${inWords(names)})` : "";
  showStatus(
    `Not requantified: ${sentence(`${error.message}${which}`)} The nets are as they were.`,
  );
}

// --- Exporting the results ---

const EXPORT_TITLE =
  "Write the charts, the lane tables and a record of how they were made into a new folder" +
  " under exports in the project folder";
const NO_LANES_TITLE = "Nothing to export yet: declare the lanes in Lanes & values";

let exporting = false; // a press has no answer yet

// The header's Export: disabled while an export runs, while no lanes are
// declared (its tooltip says so), and once Quit is pressed (renderQuit).
function renderExport(project) {
  const button = $("export");
  const lanes = project.lanes.length > 0;
  button.disabled = exporting || quitting || !lanes;
  button.title = exporting ? "Exporting…" : lanes ? EXPORT_TITLE : NO_LANES_TITLE;
}

// Write the results, as the page shows them, into a new folder under exports:
// each set's lane table and charts (in the server's default formats: none are
// asked for), a README and the record. Like a requantify, it runs in the
// panel's queue once the edits made before it have their answers (a lane just
// typed, a box being placed), so the export has them, and the edits made after
// it, and an open, wait for it. Not a change: nothing is logged, so no Undo;
// and no value changes, so it is not awaited as an edit is ("Updating…"): the
// status line says "Exporting…". Pressed twice (a double click), it is sent
// once: the button stays disabled until the answer. Quit waits for it too
// (renderQuit).
function exportResults() {
  if (exporting || quitting || !state.project || $("workspace").hidden) {
    return;
  }
  exporting = true;
  renderExport(state.project);
  renderQuit();
  showStatus("Exporting…");
  const after = Promise.allSettled([pending(), proteinPanel.adding]);
  const asked = proteinPanel.queueEdit(
    async (current) => {
      showStatus("Exporting…"); // again: the answer of an edit before it empties the line
      // Its answer is about the opening shown (request()). Refused as
      // project_changed, or answered about a newer opening, it starts the
      // follow instead (projectChanged), which turns current() false.
      try {
        const answer = await request("POST", "/api/export", { answer: true });
        applyAnswer(answer);
        if (current()) {
          showExported(answer);
        }
      } catch (error) {
        if (current()) {
          await reportExport(error);
        }
      }
      return null;
    },
    { after },
  );
  const done = () => {
    exporting = false;
    renderQuit();
    if (state.project) {
      renderExport(state.project);
    }
    const button = $("export");
    if (!button.disabled && !button.hidden) {
      // Refused: the keyboard stays on Export, which may have lost it while disabled.
      if (focusLost()) {
        button.focus();
      }
    } else {
      // Disabled for good: the lanes are gone (a no_lanes refusal shows the
      // project read again before this). The keyboard goes on to Undo or Redo,
      // as from Requantify once its offer has gone.
      keepFocus(button, "undo", null);
    }
  };
  asked.then(done, done);
}

$("export").addEventListener("click", exportResults);

// What the export wrote, with Show folder, which opens that folder in the
// system file manager. The keyboard goes on to it: Export again would make
// another folder.
function showExported(answer) {
  const folder = answer.folder;
  const button = showStatus(
    `Exported ${counted(answer.files.length, "file", "files")} to ${folder}`,
    { label: "Show folder", name: `Show the folder ${folder}`, run: () => revealExport(folder) },
  );
  if (button && (focusLost() || document.activeElement === $("export"))) {
    button.focus();
  }
}

// Show an export folder, as the export answered it, in the system file
// manager. One moved or deleted since is said, which takes the status line's
// Show folder away: the keyboard then goes back to Export. Settles once
// answered, and never rejects.
async function revealExport(folder) {
  const opened = shownOpening();
  try {
    await call("POST", "/api/project/reveal", { folder });
  } catch (error) {
    if (opened !== shownOpening()) {
      return;
    }
    if (error.code === "folder_not_found") {
      showStatus(
        `${folder} is not in the project folder any more: it was moved or deleted outside Proteia.`,
      );
    } else {
      report(error);
    }
    const button = $("export");
    if (focusLost() && !button.disabled && !button.hidden) {
      button.focus();
    }
  }
}

// A refused export writes nothing: what to do, by the refusal's code; the
// images a missing or changed file belongs to, by their file names. Settles
// once said (and, with no lanes, once the project is shown as it is now).
async function reportExport(error) {
  if (saidElsewhere(error)) {
    return;
  }
  let why = sentence(error.message);
  if (error.code === "no_lanes") {
    why = "No lanes are declared. Declare the lanes in Lanes & values, then export.";
  } else if (error.code === "image_file_changed") {
    const names = imageNames(error.ids);
    const one = names.length <= 1;
    const which = names.length
      ? `The ${one ? "file" : "files"} of ${inWords(names)} ${one ? "is" : "are"}`
      : "An image file is";
    why =
      `${which} missing from the project's images folder or ${one ? "was" : "were"} changed` +
      ` outside Proteia. Put the original ${one ? "file" : "files"} back, then export.`;
  } else if (error.code === "path_too_long") {
    // Projects live only in the projects folder (Projects… names it), so the
    // way out is a shorter name: the project's folder's, renamed while Proteia
    // is not running (it has no rename).
    why =
      "The project folder's path is too long for the file names an export writes. Give the" +
      " project a shorter name: quit Proteia, rename the project's folder (Projects… shows" +
      " where projects are saved), then start Proteia again and export.";
  }
  showStatus(`Not exported: ${why}`);
  if (error.code === "no_lanes") {
    // The page showed lanes (Export is disabled without): another tab removed
    // them since. Show the project as it is now.
    await reread().catch(() => {});
  }
}

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
  set_box_padding: "change box padding",
  clear_boxes: "clear boxes",
  detect_row_boxes: "detect row boxes",
  remove_undetected: "remove n.d. mark",
  requantify: "requantify",
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
    const names = imageNames(error.ids);
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
// is still the one logged as `seq` (the status line's Undo of a clear); `back`
// then gives the control that made it (Clear boxes by default), where the
// keyboard goes once that Undo has gone. It runs in the panel's queue and its
// answer goes through send(), so it never reaches or shows a project opened
// after it was asked for (see openProject); the edits made after it wait for
// it (ordered, and the queue).
function takeStep(direction, { seq = null, back = () => $("clear-boxes") } = {}) {
  if (!state.project || $("workspace").hidden) {
    return;
  }
  const had = document.activeElement;
  const from = seq !== null ? back : null; // taken from the status line
  const offered = state.project.history[direction]; // what the page showed when asked
  const after = Promise.allSettled([pending(), proteinPanel.adding]);
  proteinPanel.queueEdit(
    async (current) => {
      const step = state.project.history[direction];
      if (seq !== null && !(step && step.seq === seq)) {
        showStatus(
          "Not undone: other changes were made since. Undo at the top takes back the last one.",
        );
        keepFocus(had, direction, from);
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
        keepFocus(had, direction, from);
        return answer;
      } catch (error) {
        if (current() && opened === shownOpening()) {
          reportStep(direction, error);
          keepFocus(had, direction, from);
        }
        return null;
      }
    },
    { after },
  );
}

// The control a step was taken from (`had`, focused then) may be gone (the
// status line's Undo), hidden (Requantify once the project is on the local
// background, the box panel's Delete box once its box is gone: a redo of their
// change) or disabled (the last Undo): if the focus was still on it, the
// keyboard goes on to the next useful one: from the status line, the control
// whose change it took back (`from()`, if any), then Undo or Redo.
function keepFocus(had, direction, from) {
  const gone = had && (!had.isConnected || had.disabled || !had.getClientRects().length);
  if (!gone || !(focusLost() || document.activeElement === had)) {
    return;
  }
  const other = direction === "undo" ? "redo" : "undo";
  const targets = [...(from ? [from()] : []), $(direction), $(other)];
  const target = targets.find((t) => t && !t.disabled && t.getClientRects().length);
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
    document.querySelector("dialog[open]") ||
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
  } else if (event.key === "c" || event.key === "C") {
    toggleColours();
  }
});

// --- Quit ---

let quitting = false; // pressed: its answer is awaited, or Proteia has stopped

// Quit waits for an export: stopping the server under it would cut it off, and
// its answer would come after "Proteia has stopped" (the page shown again, or
// "Not exported" for a folder written in full). Disabled while one runs, its
// tooltip says why; once pressed, no export starts (renderExport). So does an
// import of images handed to Proteia, which closes the project shown.
function renderQuit() {
  const button = $("quit");
  button.disabled = quitting || exporting;
  button.title = exporting ? "Quit once the export is written" : "";
  importDialog.block(exporting ? "Import once the export is written." : null);
}

// Before it stops Proteia, Quit reads which images wait: a stop would drop
// them, so it shows them instead, to import or discard, and Proteia keeps
// running. One that cannot be read (Proteia stopped, say) is left to the quit
// to say.
$("quit").addEventListener("click", async () => {
  if (quitting || exporting) {
    return;
  }
  quitting = true;
  renderQuit();
  if (state.project) {
    renderExport(state.project);
  }
  const workspace = await readWorkspace().catch(() => null);
  const waiting = workspace
    ? workspace.handoffs.filter((handoff) => handoff.kind === "images" && !finished.has(handoff.id))
    : [];
  if (waiting.length) {
    quitting = false;
    renderQuit();
    if (state.project) {
      renderExport(state.project);
    }
    const found = takeListing(workspace, { show: "asked" });
    importDialog.say(WAITING_AT_QUIT);
    showStatus([WAITING_AT_QUIT, found].filter(Boolean).join(" "));
    return;
  }
  try {
    await request("POST", "/api/quit");
    showStatus("Proteia has stopped. You can close this tab.");
    $("workspace").hidden = true;
    $("quit").hidden = true;
    $("export").hidden = true;
    $("undo").hidden = true;
    $("redo").hidden = true;
    renderWaiting();
  } catch (error) {
    quitting = false;
    if (error instanceof ApiError && error.status === 401) {
      $("quit").hidden = true; // this tab cannot reach the running Proteia
    } else {
      showStatus(error.message || "Proteia did not stop. Try Quit again.");
    }
    renderQuit();
    if (state.project) {
      renderExport(state.project);
    }
  }
});

// --- Start ---

async function start() {
  if (!token) {
    showStatus(NEEDS_LAUNCH);
    return;
  }
  // The project open, if one is, then the images waiting on top of it
  // (takeListing: the first in the Import dialog); with neither, the Projects
  // dialog.
  try {
    const workspace = await readWorkspace();
    $("quit").hidden = false;
    showStatus("");
    if (workspace.open) {
      applyAnswer(await call("GET", "/api/project"));
    }
    started = true;
    const found = takeListing(workspace);
    if (found) {
      showStatus(found);
    }
  } catch (error) {
    if (!(error instanceof ApiError && error.status === 401)) {
      showStatus(NOT_RESPONDING);
    }
  }
}

start();
