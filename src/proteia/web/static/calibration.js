// SPDX-License-Identifier: Apache-2.0
// The "Molecular weight" section (#58), for the shown image's register group
// (the images linked to one marker image): the membrane's ladder, finding a
// ladder and adjusting it on a ruler before it is applied, marking bands one
// by one where no ladder stands out, the fit and the marks; and, in the Images
// section, the marker image a chemiluminescence image is linked to. It stores
// nothing itself: each change goes to the server through the app, and the
// section is drawn again from the state answered. The ruler is the page's
// draft until Apply stores its solid ticks, in one change and one undo step.
import { $, counted, inWords, isolate, rebuild, sentence, span, swatch } from "/static/dom.js";

// The core's thresholds (proteia.core.mwcal), which the fit line words.
const FIT_WARN = 0.2; // a ladder whose leave-one-out disagreement is above this
const LADDERS_WARN = 0.05; // two ladders that disagree beyond this, their offset taken out
const MIN_LADDER_POINTS = 2; // a ladder with fewer marks is not used
const MIN_SHARED_MWS = 2; // two ladders are combined only when they share this many MWs

// The arrow keys move a ruler tick this many image pixels (with Shift, the larger).
const NUDGE = 0.5;
const NUDGE_FAR = 5;
// A stretch never brings the ruler's end ticks closer than this (image pixels).
const MIN_SPAN = 4;

const TICK_COLOR = "#ffd54a";
const UNLABELLED_COLOR = "#9e9e9e"; // a band ▲▼ left with no label
const RIGHT_COLOR = "#80deea"; // the marks of the second ladder
// A reference band's colour on the image, by the first word its vendor names it with.
const REFERENCE_COLORS = { orange: "#ff9800", green: "#43a047", pink: "#ff5fa2", blue: "#4f8cff" };

// A ruler tick's state in words. Every state but "predicted" (a hollow tick,
// where the ladder's shape puts a band nobody found, snapped to or placed) is
// solid: Apply stores the solid ticks only.
const TICK_STATES = {
  found: "on a band found",
  predicted: "predicted, not stored",
  snapped: "snapped to its band",
  hand: "placed by hand",
  stored: "as stored",
};

// A calibration point's source, as the list of marks names it.
const SOURCE_WORDS = {
  visible_marker: "marker band",
  chemiluminescence_marker: "faint marker band",
  strip_edge: "strip edge",
};

const KDA = new Intl.NumberFormat("en", { maximumSignificantDigits: 4, useGrouping: false });
const RANGE = new Intl.NumberFormat("en", { maximumSignificantDigits: 3, useGrouping: false });

// An MW as the page writes it: "250", "61.5".
function kdaText(mw) {
  return KDA.format(mw);
}

function capital(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// Whether two MWs are one, as the server compares them (on log10).
function sameMw(a, b) {
  return Math.abs(Math.log10(a) - Math.log10(b)) < 1e-12;
}

// Whether two ladders' MWs (top to bottom) are one list.
function sameLadder(a, b) {
  return Array.isArray(a) && a.length === b.length && a.every((mw, i) => sameMw(mw, b[i]));
}

// A tick on a band (found, snapped to, placed or stored), not only predicted.
function solid(tick) {
  return tick.state !== "predicted";
}

// A tick with a ladder MW: ▲ or ▼ can move a label past either end of the
// ladder's list, and the tick then has none until moved back or relabelled.
function labelled(tick) {
  return tick.mw !== null;
}

// What Apply stores: a solid tick with a label.
function kept(tick) {
  return solid(tick) && labelled(tick);
}

// What the ruler shows: every solid tick (one without a label as "?"), and the
// predicted ticks that have a label.
function shown(tick) {
  return solid(tick) || labelled(tick);
}

// A tick in words: "70 kDa", or "the band with no label".
function tickName(tick) {
  return labelled(tick) ? `${kdaText(tick.mw)} kDa` : "the band with no label";
}

// A ladder side as the page names it.
function ladderName(side) {
  return side === "right" ? "the second ladder (right)" : "the ladder";
}

// A refusal of the ladder's marks in words, from its code and detail (the
// ladder side and the MWs the server names): never an id or a position; null
// for any other refusal, whose own message is shown.
function ladderRefusalWords(error) {
  const detail = error.detail || {};
  const ladder = ladderName(detail.side);
  if (error.code === "calibration_order" && detail.upper != null && detail.lower != null) {
    const [upper, lower] = [kdaText(detail.upper), kdaText(detail.lower)];
    return detail.reason === "same_height"
      ? `on ${ladder}, ${upper} and ${lower} kDa would lie at the same height, but each band` +
          " lies at its own: move one of them"
      : `on ${ladder}, ${lower} kDa would lie below ${upper} kDa, but heavier bands lie higher` +
          " up: move one of them back, or relabel it";
  }
  if (error.code === "duplicate_mw" && detail.mw != null) {
    return `${ladder} would hold ${kdaText(detail.mw)} kDa twice: relabel or remove one of them`;
  }
  if (error.code === "ladder_sides" && detail.reason === "strip_edge") {
    return (
      "a second ladder cannot be used with strip edges, which calibrate images with one ladder" +
      " only: remove the strip edges first"
    );
  }
  if (error.code === "ladder_sides" && detail.reason === "no_x" && detail.mw != null) {
    return (
      `a second ladder needs the lane of every mark, and the ${kdaText(detail.mw)} kDa mark of` +
      ` ${ladderName(detail.side)} has none (it was saved before marks kept it): remove it and` +
      " mark it again"
    );
  }
  if (error.code === "ladder_sides" && detail.reason === "not_right") {
    return "the second ladder must lie right of the first one: find or mark it right of that lane";
  }
  return null;
}

// A calibration refusal for the status line: `verb` ("Not applied") and its
// words (ladderRefusalWords), or the server's message.
function refusalText(error, verb) {
  const words = ladderRefusalWords(error);
  return words ? `${verb}: ${words}.` : sentence(error.message);
}

// Why a ruler's Apply was refused as calibration_changed (detail.changed):
// what it was opened with changed since, in another tab say.
const CHANGED_WORDS = {
  ladder_kda:
    "the membrane's ladder was changed (in another tab, say), and its labels are the ladder" +
    " before",
  group:
    "the image's marker link was changed (in another tab, say), and with it the marks a ruler" +
    " replaces",
};

// The titles of ▲ and ▼ (index.html's), while they can move the labels.
const SHIFT_TITLES = {
  "cal-up": "Move every label up one band: each band then reads the next lighter MW",
  "cal-down": "Move every label down one band: each band then reads the next heavier MW",
};

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function sortTicks(ticks) {
  return [...ticks].sort((a, b) => a.y - b.y);
}

// A share as a percentage for the fit line: "15%", "0.4%".
function percent(share) {
  const value = share * 100;
  return value < 1 ? `${Math.max(0.1, value).toFixed(1)}%` : `${Math.round(value)}%`;
}

// Where a ladder of marks (sorted down the image, each {y, mw}) puts `mw`:
// log10(MW) linear in y between neighbouring marks, the end segments
// extended, as the server fits a ladder; null with fewer than two marks.
function ladderY(marks, mw) {
  if (marks.length < 2) {
    return null;
  }
  const z = Math.log10(mw);
  let i = 0;
  while (i < marks.length - 2 && z < Math.log10(marks[i + 1].mw)) {
    i += 1;
  }
  const [a, b] = [marks[i], marks[i + 1]];
  const [za, zb] = [Math.log10(a.mw), Math.log10(b.mw)];
  return a.y + ((z - za) * (b.y - a.y)) / (zb - za);
}

// The MW a ladder of marks (sorted down the image) puts at `y`, the same way.
function ladderMw(marks, y) {
  if (marks.length < 2) {
    return null;
  }
  let i = 0;
  while (i < marks.length - 2 && y > marks[i + 1].y) {
    i += 1;
  }
  const [a, b] = [marks[i], marks[i + 1]];
  const [za, zb] = [Math.log10(a.mw), Math.log10(b.mw)];
  return 10 ** (za + ((y - a.y) * (zb - za)) / (b.y - a.y));
}

// Whether `mw` at `y` keeps every mark of `marks` in order: heavier above it,
// lighter below it, none at its MW.
function inOrder(marks, y, mw) {
  return marks.every((mark) => {
    if (sameMw(mark.mw, mw)) {
      return false;
    }
    return mark.y < y ? mark.mw > mw : mark.y > y && mark.mw < mw;
  });
}

// The ticks of a ruler with every hollow tick that is out of order with the
// labelled solid ones dropped (a relabel, or a mark moved past it).
function orderedHollow(ticks) {
  const anchors = ticks.filter(kept);
  return ticks.filter(
    (tick) => solid(tick) || !labelled(tick) || inOrder(anchors, tick.y, tick.mw),
  );
}

// A preset's product, short, as the fit line names it: "PageRuler Plus",
// "Precision Plus Dual Color".
function shortProduct(preset) {
  return preset.product
    .replace(/,.*$/, "")
    .replace(/\b(Prestained|Protein|Ladder|Standards)\b/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

let nextTickId = 1; // each ruler tick's key, kept through its edits (the list keeps its focus)

// A ruler tick: its MW and its place in the ladder's list `kda` (null for an
// MW the list does not hold), where it is, and how it got there.
function newTick(mw, y, state, kda) {
  const index = kda.findIndex((each) => sameMw(each, mw));
  return { id: nextTickId++, mw, index: index < 0 ? null : index, y, state, snappedX: null };
}

export class CalibrationPanel {
  // `view`: the image view, which draws the marks and the ruler (setLadderTicks,
  // setRuler) and takes the clicks of a pick tool (setPickTool); its handlers
  // pick, ruler and ladderPoint come here. handlers: read(method, path, json,
  // {anyProject}) gives a Promise of the answer of a route that changes nothing
  // (the presets, a proposal, a snap); edit(method, path, json, {refused,
  // sent}) sends a change in order with the page's other edits and gives its
  // answer once applied (null: about a project no longer shown), and rejects
  // with the refusal, which `refused` (by default the status line) shows;
  // `sent(project)` is given the state shown when it is sent; status(text,
  // action) shows a line with an optional action ({label, name, seq, run}) and
  // gives its button; report(error) shows a refusal there; undo(seq) takes back
  // the change logged as `seq` if it is still the last; showImage(id) shows
  // another image; changed() draws the page again (the view's hint and row
  // tool follow what this section is doing).
  constructor(view, handlers) {
    this.view = view;
    this.handlers = handlers;
    this.project = null;
    this.image = null;
    this.membrane = null; // the shown image's membrane, as the state has it
    this.group = null; // ... and its register group: {image_ids, points, fit}
    this.presets = null; // GET /api/ladders, once answered
    this.presetsAsked = false;
    this.ladderShown = null; // what the ladder select was built for
    this.markersShown = null; // what the marker select was built for
    // The ruler being adjusted: {imageId, openId, ladder, group, side, x,
    // foundAt, doubtful, extra, ticks: [{id, mw, index, y, state, snappedX}]
    // (newTick), error}, `ladder` and `group` what ladderScope() and
    // groupScope() were when it opened; `base` is the ruler as it was when a
    // drag of it began (Esc takes it back there).
    this.draft = null;
    this.base = null;
    this.focusId = null; // the tick whose button has the keyboard focus
    this.snapping = Promise.resolve(); // a dropped tick's snap under way
    this.applying = false;
    this.finding = false;
    // What a click on the image does now: {kind: "find", side} or {kind:
    // "mark", side, edges}, each with the openId and groupScope() it was armed
    // on (newTool); null: nothing of this section's.
    this.tool = null;
    // The open popup: {back, scope}, the control it returns the keyboard to,
    // and what it was opened on (menuScope).
    this.menu = null;
    this.customFor = null; // the opening and membrane the Custom ladder form was opened for
    this.dropped = null; // why render() last dropped a stale ruler, until taken (takeDropped)
    this.bind();
  }

  // --- The page's hooks ---

  // Another project is shown: nothing of the one before carries over.
  forget() {
    this.closeDraft();
    this.closeCustom(false);
    this.dropped = null;
    this.tool = null;
    this.ladderShown = null;
    this.markersShown = null;
  }

  // What the view's hint says while this section takes the clicks, or null.
  hint() {
    if (!this.image) {
      return null;
    }
    const tool = this.tool;
    if (tool && tool.kind === "find") {
      return tool.side === "right"
        ? "Click the second ladder's lane, right of the first: its bands are found and labelled" +
            " · Esc cancels"
        : "Click the ladder's lane: its bands are found and labelled · Esc cancels";
    }
    if (tool && tool.kind === "mark") {
      const ladder = tool.side === "right" ? "second ladder" : "ladder";
      return tool.edges
        ? "Click where the cut runs through a marker band, then choose that band's MW" +
            " · Esc ends marking"
        : `Click a band of the ${ladder}, then choose its MW · Esc ends marking`;
    }
    if (this.draft) {
      return (
        "Ruler: drag its line, a round grip (stretch) or a tick (it snaps; Alt: exact)" +
        " · click a tick to relabel it · Enter applies · Esc drops it"
      );
    }
    return null;
  }

  // Whether this section takes the image's clicks and drags (no row, no box).
  busy() {
    return Boolean(this.tool || this.draft);
  }

  // Whether the view's hint goes to the right: a ruler in the image's left
  // half would sit under it, top grip first.
  hintAway() {
    const draft = this.draft;
    const image = this.image;
    return Boolean(draft && image && draft.imageId === image.id && draft.x < image.width / 2);
  }

  // Esc: closes the popup, ends a tool, or drops the ruler (nothing is
  // stored). Only the drag, when Esc just cancelled one (`gestureCancelled`):
  // the ruler goes back to how it was before it. Whether it was taken.
  escape({ gestureCancelled = false } = {}) {
    if (this.menu) {
      this.closeMenu(true);
      return true;
    }
    if (gestureCancelled) {
      return this.busy();
    }
    if (this.tool) {
      const marking = this.tool.kind === "mark";
      this.tool = null;
      this.handlers.changed();
      this.handlers.status(marking ? "Marking ended." : "");
      return true;
    }
    if (this.draft) {
      if (this.applying) {
        return true; // being stored: its answer closes it
      }
      this.closeDraft();
      this.handlers.changed();
      this.handlers.status("Ruler dropped: nothing was stored.");
      this.keepFocus();
      return true;
    }
    return false;
  }

  // Enter (not on a button): applies the ruler. Whether it was taken.
  enter() {
    if (!this.draft || this.menu) {
      return false;
    }
    this.apply();
    return true;
  }

  // After an import of a visible-light marker or merged image: the one
  // chemiluminescence image of its membrane, of its size and not linked yet,
  // to offer to link it to ({imageId, name}); null for none or several.
  linkOffer(project, markerId) {
    const marker = project.images.find((image) => image.id === markerId);
    if (!marker || !(marker.kind === "visible_marker" || marker.kind === "merged")) {
      return null;
    }
    const unlinked = project.images.filter(
      (image) =>
        image.membrane_id === marker.membrane_id &&
        image.kind === "chemiluminescence" &&
        image.marker_image_id === null &&
        image.width === marker.width &&
        image.height === marker.height,
    );
    return unlinked.length === 1
      ? { imageId: unlinked[0].id, name: isolate(unlinked[0].original_name) }
      : null;
  }

  // --- Drawing ---

  render(project, image) {
    this.project = project;
    this.image = image;
    this.membrane = image
      ? project.membranes.find((membrane) => membrane.id === image.membrane_id) || null
      : null;
    this.group = this.membrane
      ? this.membrane.groups.find((group) => group.image_ids.includes(image.id)) || null
      : null;
    const draft = this.draft;
    const stale = draft ? this.staleRuler(draft) : null;
    if (stale !== null) {
      this.closeDraft();
      if (stale) {
        this.dropped = stale; // an Undo or Redo says it after its own words (takeDropped)
        this.handlers.status(stale);
      }
    }
    if (this.tool && !this.toolFits(this.tool)) {
      this.tool = null; // armed on another register group, or its button disabled since
    }
    if (this.customFor !== null && this.customFor !== this.customScope()) {
      // Typed for another membrane, or another project: it sets no ladder here.
      this.closeCustom($("cal-custom").contains(document.activeElement));
    }
    if (this.menu && this.menu.scope !== this.menuScope()) {
      // Opened on another image, ladder or register group (the image switched
      // by keyboard, an Undo): its MWs, and what a choice marks, were those.
      this.closeMenu($("cal-menu").contains(document.activeElement));
    }
    $("calibration").hidden = !this.group;
    this.renderMarkerLink();
    if (!this.group) {
      this.closeMenu(false);
      this.drawOnView();
      return;
    }
    this.askPresets();
    this.renderGroup();
    this.renderLadder();
    this.refresh();
  }

  // The membrane's ladder as it is now, and the register group of the shown
  // image (its membrane and images): a ruler's labels are that ladder's, and
  // its Apply replaces a ladder of that group's marks.
  ladderScope() {
    return JSON.stringify(this.membrane ? [this.membrane.ladder, this.membrane.ladder_kda] : null);
  }

  groupScope() {
    return JSON.stringify(this.group ? [this.membrane.id, [...this.group.image_ids].sort()] : null);
  }

  // The Custom ladder form's scope: the project opening and the membrane shown.
  customScope() {
    return JSON.stringify(this.membrane ? [this.project.open_id, this.membrane.id] : null);
  }

  // A popup's scope: the project opening, the image shown, its ladder and its
  // register group.
  menuScope() {
    const image = this.image ? this.image.id : null;
    return JSON.stringify([this.project.open_id, image, this.ladderScope(), this.groupScope()]);
  }

  // Why the last drawing dropped a ruler (staleRuler), once: an Undo or Redo
  // says it after its own words, which replace the status line. "" for none.
  takeDropped() {
    const dropped = this.dropped;
    this.dropped = null;
    return dropped || "";
  }

  // Why the ruler can no longer be applied as it is drawn, for the status line
  // ("": nothing to say, as when another project is shown), or null while it
  // can: it belongs to the image it was drawn on, the ladder its labels are
  // from, and the register group whose marks it replaces. However the ladder
  // or the group changed (another tab, Undo, Redo), its labels or what Apply
  // replaces are no longer what the ruler shows.
  staleRuler(draft) {
    const image = this.image;
    if (!image || draft.openId !== this.project.open_id) {
      return "";
    }
    if (draft.imageId !== image.id) {
      return "The ruler was dropped: another image is shown. Nothing was stored.";
    }
    if (draft.ladder !== this.ladderScope()) {
      return (
        "The ruler was dropped: the membrane's ladder changed, and its labels were the ladder" +
        ` before. Nothing was stored; ${this.findAgain()}.`
      );
    }
    if (draft.group !== this.groupScope()) {
      return (
        "The ruler was dropped: the image's marker link changed, and with it the marks a ruler" +
        " replaces. Nothing was stored."
      );
    }
    return null;
  }

  // A tool armed now, on the shown image's register group.
  newTool(fields) {
    return { ...fields, openId: this.project.open_id, group: this.groupScope() };
  }

  // Whether an armed tool still acts on what it was armed on: the same project
  // opening and register group (not an image of another membrane), and its
  // button not disabled since (the ladder taken back, say).
  toolFits(tool) {
    if (!this.group || tool.openId !== this.project.open_id || tool.group !== this.groupScope()) {
      return false;
    }
    return !(tool.kind === "find" ? this.findRefusal(tool.side) : this.markRefusal());
  }

  // Draw again what a change of the ruler, a tool or the presets changes.
  refresh() {
    if (!this.group) {
      return;
    }
    this.renderLocks();
    this.renderFind();
    this.renderDraft();
    this.renderFit();
    this.renderMarks();
    this.renderMarking();
    this.drawOnView();
  }

  drawOnView() {
    this.view.setLadderTicks(this.marksShown());
    this.view.setRuler(this.rulerShown());
    this.view.setPickTool(this.tool);
  }

  // The presets, asked for once: until they come the select offers Custom… only.
  askPresets() {
    if (this.presets !== null || this.presetsAsked) {
      return;
    }
    this.presetsAsked = true;
    this.handlers
      .read("GET", "/api/ladders", undefined, { anyProject: true })
      .then((answer) => {
        this.presets = answer.ladders;
        if (this.membrane) {
          this.renderLadder();
          this.refresh();
        }
      })
      .catch(() => {
        this.presetsAsked = false; // asked again at the next drawing
      });
  }

  preset() {
    const key = this.membrane && this.membrane.ladder;
    return (this.presets || []).find((preset) => preset.key === key) || null;
  }

  // The ladder's MWs, top to bottom, as the membrane stores them.
  ladderKda() {
    return this.membrane ? this.membrane.ladder_kda : [];
  }

  // MW (to 12 digits) -> the colour of that reference band of the preset.
  referenceColors() {
    const colors = new Map();
    const preset = this.preset();
    for (const band of preset ? preset.reference : []) {
      const color = REFERENCE_COLORS[band.colour.split(" ")[0]];
      if (color) {
        colors.set(band.kda.toPrecision(12), color);
      }
    }
    return colors;
  }

  referenceColor(mw) {
    const colors = this.referenceColors();
    const kda = [...colors.keys()].find((key) => sameMw(Number(key), mw));
    return kda === undefined ? null : colors.get(kda);
  }

  // The preset's reference bands in words: "70 and 25 kDa orange, 10 kDa green".
  referenceWords() {
    const preset = this.preset();
    const byColour = new Map();
    for (const band of preset ? preset.reference : []) {
      byColour.set(band.colour, [...(byColour.get(band.colour) || []), kdaText(band.kda)]);
    }
    return [...byColour].map(([colour, mws]) => `${inWords(mws)} kDa ${colour}`).join(", ");
  }

  // The group's marks on `side`, down the image.
  sidePoints(side) {
    const points = this.group ? this.group.points : [];
    return sortTicks(points.filter((point) => point.side === side));
  }

  // ... but its strip edges: the marks of bands, which a ruler can hold.
  bandPoints(side) {
    return this.sidePoints(side).filter((point) => point.source !== "strip_edge");
  }

  // The marks of the ruler's side that its Apply removes and it does not show:
  // Apply stores the ruler's ticks as that side's marks, as band marks of its
  // image at its x, in place of all it had; a strip edge, or a mark on another
  // image of the group, is none of these, and is left off the ruler (adjust).
  lostPoints(draft) {
    return this.sidePoints(draft.side).filter(
      (point) => point.source === "strip_edge" || point.image_id !== draft.imageId,
    );
  }

  // Those marks in words: "the 10 kDa strip edge and the 70 and 55 kDa marks
  // on ⁨b.tif⁩".
  lostWords(points) {
    const byWhat = new Map();
    for (const point of points) {
      const key = point.source === "strip_edge" ? "" : point.image_id;
      byWhat.set(key, [...(byWhat.get(key) || []), point]);
    }
    const parts = [...byWhat].map(([imageId, marks]) => {
      const mws = `${inWords(marks.map((mark) => kdaText(mark.mw)))} kDa`;
      if (!imageId) {
        return `the ${mws} ${marks.length === 1 ? "strip edge" : "strip edges"}`;
      }
      const image = this.project.images.find((each) => each.id === imageId);
      const name = isolate(image ? image.original_name : imageId);
      return `the ${mws} ${marks.length === 1 ? "mark" : "marks"} on ${name}`;
    });
    return inWords(parts);
  }

  // The marks the view draws: every mark of the group, but those of the side
  // the ruler replaces while it is adjusted; each known by its side and MW,
  // which name a mark of the group (a drag of one follows it by that).
  marksShown() {
    if (!this.group || !this.image) {
      return [];
    }
    const replaced = this.draft ? this.draft.side : null;
    return this.group.points
      .filter((point) => point.side !== replaced)
      .map((point) => ({
        point,
        id: JSON.stringify([point.side, point.mw]),
        x: point.x,
        y: point.y,
        label: `${kdaText(point.mw)} kDa`,
        color: point.side === "right" ? RIGHT_COLOR : TICK_COLOR,
        labelsLeft: (point.x === null ? 0 : point.x) < this.image.width / 2,
      }));
  }

  // The ruler as the view draws it: labels on the side away from the lanes.
  rulerShown() {
    const draft = this.draft;
    if (!draft || !this.image || draft.imageId !== this.image.id) {
      return null;
    }
    return {
      x: draft.x,
      labelsLeft: draft.x < this.image.width / 2,
      extra: draft.extra,
      ticks: draft.ticks.filter(shown).map((tick) => {
        const reference = labelled(tick) ? this.referenceColor(tick.mw) : null;
        return {
          id: tick.id,
          y: tick.y,
          label: labelled(tick) ? kdaText(tick.mw) : "?",
          color: labelled(tick) ? reference || TICK_COLOR : UNLABELLED_COLOR,
          reference: reference !== null,
          solid: solid(tick),
          focused: tick.id === this.focusId,
        };
      }),
    };
  }

  // The marker image of a chemiluminescence image (Images section): none, or
  // a marker or merged image of its membrane; one of another size is offered
  // disabled, its reason in its tooltip.
  renderMarkerLink() {
    const image = this.image;
    const field = $("marker-field");
    field.hidden = !image || image.kind !== "chemiluminescence";
    if (field.hidden) {
      $("marker-none").hidden = true;
      return;
    }
    const select = $("marker-image");
    const markers = this.project.images.filter(
      (other) =>
        other.membrane_id === image.membrane_id &&
        (other.kind === "visible_marker" || other.kind === "merged"),
    );
    const shown = JSON.stringify([
      this.project.open_id,
      image.id,
      markers.map((m) => [m.id, m.original_name, m.width, m.height]),
    ]);
    if (shown !== this.markersShown) {
      this.markersShown = shown;
      const options = [new Option("none", "")];
      for (const marker of markers) {
        const option = new Option(marker.original_name, marker.id);
        if (marker.width !== image.width || marker.height !== image.height) {
          option.disabled = true;
          option.title =
            `${marker.width}×${marker.height} px, not ${image.width}×${image.height}:` +
            " a marker image must match its image pixel for pixel";
        }
        options.push(option);
      }
      select.replaceChildren(...options);
    }
    select.value = image.marker_image_id || "";
    $("marker-none").hidden = markers.length > 0;
  }

  // Which images the section calibrates, and, for a chemiluminescence image
  // with no marks and no marker image while another image of its size on the
  // membrane has marks, an offer to link it to that one.
  renderGroup() {
    const names = this.group.image_ids
      .map((id) => this.project.images.find((image) => image.id === id))
      .filter(Boolean)
      .map((image) => isolate(image.original_name));
    const line = $("cal-group");
    line.hidden = names.length < 2;
    line.textContent = `One calibration for ${inWords(names)} (linked).`;
    const image = this.image;
    const offer = $("cal-link-offer");
    const unmarked =
      image.kind === "chemiluminescence" &&
      image.marker_image_id === null &&
      !this.group.points.length;
    const marked = unmarked
      ? this.membrane.groups
            .filter((group) => group !== this.group && group.points.length)
            .flatMap((group) => group.image_ids)
            .map((id) => this.project.images.find((other) => other.id === id))
            .filter(
              (other) =>
                other &&
                (other.kind === "visible_marker" || other.kind === "merged") &&
                other.width === image.width &&
                other.height === image.height,
            )
        : [];
    offer.hidden = !marked.length;
    if (marked.length) {
      const [marker] = marked;
      const name = isolate(marker.original_name);
      $("cal-link-text").textContent =
        `${name} has ladder marks; link this image to it to read MWs here.`;
      const button = $("cal-link");
      button.textContent = "Link";
      button.setAttribute("aria-label", `Link this image to its marker image ${name}`);
      button.onclick = () => this.link(image.id, marker.id);
    }
  }

  // The ladder select: "choose…" until one is chosen, then the preset (grouped
  // by product, labelled by buffer system) or the custom ladder; Custom… opens
  // the form for another. Built again only when its options change.
  renderLadder() {
    const select = $("cal-ladder");
    const ladder = this.membrane.ladder;
    const custom = ladder !== null && !ladder.includes("/");
    const shown = JSON.stringify([
      this.presets ? this.presets.length : null,
      custom ? ladder : null,
    ]);
    if (shown !== this.ladderShown) {
      this.ladderShown = shown;
      const placeholder = new Option("choose…", "");
      placeholder.disabled = true;
      const options = [placeholder];
      const products = new Map();
      for (const preset of this.presets || []) {
        if (!products.has(preset.product)) {
          const group = document.createElement("optgroup");
          group.label = `${preset.product} (#${preset.catalog_numbers.join(", #")})`;
          products.set(preset.product, group);
          options.push(group);
        }
        const option = new Option(preset.system, preset.key);
        option.title = preset.source;
        products.get(preset.product).append(option);
      }
      if (custom) {
        options.push(new Option(`Custom: ${ladder}`, "custom:stored"));
      }
      options.push(new Option("Custom…", "custom"));
      select.replaceChildren(...options);
    }
    select.value = ladder === null ? "" : custom ? "custom:stored" : ladder;
    const preset = this.preset();
    const kda = this.ladderKda();
    const mws = $("cal-ladder-mws");
    // The select shows the buffer system only: the product goes here.
    const product = preset ? `${shortProduct(preset)}: ` : "";
    mws.textContent = kda.length
      ? `${product}${kda.map(kdaText).join(", ")} kDa, top to bottom`
      : ladder !== null
        ? "No MWs listed: Find ladder needs them; mark each band by clicking it and typing its MW."
        : "";
    mws.hidden = !mws.textContent;
  }

  // While a ladder is being found or a ruler is open, the ladder and the
  // marker link stay as they are: the ruler's labels are that ladder's, and
  // its Apply replaces a ladder of that register group's marks. (Undo, Redo or
  // another tab can still change them: staleRuler then drops the ruler.)
  renderLocks() {
    const locked = Boolean(this.draft) || this.finding;
    const why = this.draft
      ? "Apply or drop the ruler first: its labels are this ladder's"
      : "The ladder is being found";
    const preset = this.preset();
    const ladder = $("cal-ladder");
    ladder.disabled = locked;
    ladder.title = locked ? why : preset ? preset.source : "";
    const marker = $("marker-image");
    marker.disabled = locked;
    marker.title = locked ? why : "";
    $("cal-link").disabled = locked;
    $("cal-custom").querySelector('button[type="submit"]').disabled = locked;
  }

  // Why Find ladder (`side` "left") or Find second ladder ("right") cannot be
  // used now, or "".
  findRefusal(side) {
    if (this.membrane.ladder === null) {
      return "Choose the ladder first";
    }
    if (!this.ladderKda().length) {
      return "The ladder has no MWs listed to find: mark its bands by clicking them";
    }
    if (side === "right" && this.sidePoints("left").length < MIN_LADDER_POINTS) {
      return "Find the first ladder first";
    }
    return "";
  }

  // Why the marking tools cannot be used now, or "".
  markRefusal() {
    return this.membrane.ladder === null ? "Choose the ladder first" : "";
  }

  renderFind() {
    for (const [id, side] of [
      ["cal-find", "left"],
      ["cal-find-right", "right"],
    ]) {
      const button = $(id);
      const why = this.findRefusal(side);
      const armed = Boolean(this.tool && this.tool.kind === "find" && this.tool.side === side);
      button.disabled = Boolean(why) || this.applying || this.finding;
      button.setAttribute("aria-pressed", String(armed));
      button.title =
        why ||
        (side === "left"
          ? "Then click the ladder's lane on the image: its bands are found and labelled on a" +
            " ruler to adjust"
          : "Then click the second ladder's lane, right of the first: two ladders follow a" +
            " tilted blot");
    }
  }

  renderMarking() {
    const why = this.markRefusal();
    for (const [id, edges] of [
      ["cal-mark", false],
      ["cal-mark-edges", true],
    ]) {
      const button = $(id);
      const armed = Boolean(this.tool && this.tool.kind === "mark" && this.tool.edges === edges);
      button.setAttribute("aria-pressed", String(armed));
      // Not while a ladder is being found: its answer opens a ruler.
      button.disabled = Boolean(why) || Boolean(this.draft) || this.applying || this.finding;
      button.title =
        why ||
        (edges
          ? "Mark an edge only where the cut runs through a marker band; enter that band's MW"
          : "Mark the ladder's bands one by one: click a band, then choose its MW (Esc ends)");
    }
  }

  // The ruler's panel: what it holds, the doubtful-labels warning, its tools,
  // a button per tick (the keyboard's way to it), and Apply.
  renderDraft() {
    const draft = this.draft;
    $("cal-draft").hidden = !draft;
    $("cal-find-tools").hidden = Boolean(draft);
    if (!draft) {
      return;
    }
    const stored = draft.ticks.filter(kept).length;
    const which = draft.side === "right" ? "Second ladder (right)" : "Ladder";
    const ticks = counted(draft.ticks.filter(shown).length, "tick", "ticks");
    $("cal-draft-head").textContent =
      `${which} at x = ${draft.x.toFixed(1)}: ${stored} of ${ticks} to store`;
    const doubt = $("cal-draft-doubt");
    doubt.hidden = !draft.doubtful;
    doubt.textContent = draft.doubtful ? this.doubtText() : "";
    // What Apply removes that the ruler does not show (Apply's description).
    const lost = this.lostPoints(draft);
    const warning = $("cal-draft-lost");
    warning.hidden = !lost.length;
    warning.textContent = lost.length
      ? `Apply replaces this ladder's marks: it removes ${this.lostWords(lost)}, which the` +
        ` ruler does not hold (Undo brings ${lost.length === 1 ? "it" : "them"} back).`
      : "";
    const busy = this.applying;
    // Why ▲▼ cannot move the labels, said beside them too: a disabled
    // button's tooltip is out of the keyboard's reach.
    const cannot = this.shiftRefusal();
    const why = $("cal-shift-why");
    why.textContent = cannot ? `${cannot}.` : "";
    why.hidden = !cannot;
    for (const [id, step] of [
      ["cal-up", 1],
      ["cal-down", -1],
    ]) {
      $(id).disabled = busy || !this.canShift(step);
      $(id).title = cannot || SHIFT_TITLES[id];
    }
    $("cal-snap").disabled = busy;
    const apply = $("cal-apply");
    apply.disabled = busy || !stored;
    apply.title = stored
      ? "Store the solid ticks as the ladder's marks, in one step (Enter)"
      : "No tick to store: snap the ruler (Snap all), or drag a tick onto a band";
    $("cal-drop").disabled = busy;
    const colors = this.referenceColors();
    rebuild($("cal-ticks"), () => {
      for (const tick of draft.ticks.filter(shown)) {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.key = String(tick.id);
        button.dataset.id = String(tick.id);
        button.className = kept(tick) ? "" : "hollow";
        button.setAttribute("aria-keyshortcuts", "ArrowUp ArrowDown Enter Delete");
        const state = labelled(tick) ? TICK_STATES[tick.state] : "no label, not stored";
        const name = labelled(tick) ? `${kdaText(tick.mw)} kDa` : "No label";
        button.setAttribute("aria-label", `${name}, ${state}, at y ${tick.y.toFixed(1)}`);
        const reference = labelled(tick)
          ? [...colors.keys()].find((key) => sameMw(Number(key), tick.mw))
          : undefined;
        // A reference band's colour; the others keep its room, so the MWs line up.
        button.append(swatch(reference === undefined ? "transparent" : colors.get(reference)));
        button.append(
          span("cal-tick-mw", labelled(tick) ? name : "?"),
          span("cal-tick-state", state),
        );
        button.addEventListener("click", () => this.relabelMenu(tick.id, button));
        const item = document.createElement("li");
        item.append(button);
        $("cal-ticks").append(item);
      }
    });
    const error = $("cal-draft-error");
    error.textContent = draft.error;
    error.hidden = !draft.error;
  }

  // The status line of a proposal whose labels may be one band off (its gap to
  // the next-best labelling is small).
  doubtText() {
    const references = this.referenceWords();
    return references
      ? `The labels may be one band off: check the coloured reference bands (${references});` +
          " move them with ▲▼."
      : "The labels may be one band off: check them against the ladder's bands; move them" +
          " with ▲▼.";
  }

  // The fit of the shown image's group, in words: the ladder, its marks, the
  // two ladders' tilt and agreement, how well each ladder's marks agree, the
  // sides not used, and the range read.
  renderFit() {
    const box = $("cal-fit");
    box.replaceChildren();
    for (const { text, warning } of this.fitLines()) {
      const line = document.createElement("p");
      line.className = warning ? "note warning" : "note";
      line.textContent = text;
      box.append(line);
    }
  }

  fitLines() {
    const preset = this.preset();
    const ladder = this.membrane.ladder;
    const name = preset
      ? `${shortProduct(preset)} · ${preset.system}`
      : ladder !== null
        ? ladder
        : "No ladder chosen";
    const fit = this.group.fit;
    const points = this.group.points;
    if (!fit) {
      const least = `mark at least ${MIN_LADDER_POINTS} on a ladder for a curve`;
      const marks = points.length
        ? `${counted(points.length, "mark", "marks")}: ${least}`
        : "no ladder marked yet";
      return [{ text: `${name} · ${marks}` }];
    }
    const lines = [];
    const two = fit.ladders.length === 2;
    const count = `${fit.ladders.map((one) => one.points).join(" + ")} points`;
    const head = [name, !two && fit.ladders[0].two_point ? `${count} · less reliable` : count];
    const disagree =
      two &&
      (fit.disagreement_infinite || (fit.disagreement !== null && fit.disagreement > LADDERS_WARN));
    if (two && fit.offset_px !== null) {
      const offset = Math.abs(fit.offset_px);
      head.push(
        offset < 0.05
          ? "left and right level"
          : `left and right differ by ${offset.toFixed(offset < 10 ? 1 : 0)} px, about` +
              ` ${Math.abs(fit.tilt_deg).toFixed(1)}°`,
      );
      if (!disagree) {
        head.push("the ladders agree");
      }
    }
    lines.push({ text: head.join(" · ") });
    if (disagree) {
      const near = fit.disagreement_mw === null ? "" : ` near ${kdaText(fit.disagreement_mw)} kDa`;
      const size = fit.disagreement_infinite ? "" : ` (${percent(fit.disagreement)})`;
      lines.push({
        text: `The left and right ladders disagree${near}${size}: check both ladders' labels.`,
        warning: true,
      });
    }
    // D2's leave-one-out check, worded as a test the reader can picture.
    const check = "take one ladder band away and predict it from the bands above and below it";
    const poor = (one) => one.quality_infinite || (one.quality !== null && one.quality > FIT_WARN);
    const agree = fit.ladders.filter((one) => !one.two_point && one.quality !== null && !poor(one));
    if (two && agree.length === 2) {
      const [left, right] = agree;
      lines.push({
        text:
          `${capital(check)}: they agree within ${percent(left.quality)} on the left ladder and` +
          ` ${percent(right.quality)} on the right`,
      });
    }
    for (const one of fit.ladders) {
      const who = two ? `${one.side === "left" ? "Left" : "Right"} ladder: ` : "";
      if (one.two_point) {
        if (two) {
          lines.push({ text: `${who}2 points · less reliable` });
        }
      } else if (poor(one)) {
        lines.push({
          text: capital(
            `${who}${kdaText(one.worst_mw)} kDa: the bands above and below it put it at` +
              ` ${kdaText(one.predicted_mw)} kDa; check its label`,
          ),
          warning: true,
        });
      } else if (one.quality !== null && !(two && agree.length === 2)) {
        lines.push({ text: capital(`${who}${check}: they agree within ${percent(one.quality)}`) });
      }
    }
    for (const [side, reason] of fit.ignored) {
      const which = side === "left" ? "Left ladder" : "Right ladder";
      const marks = points.filter((point) => point.side === side).length;
      lines.push({
        text:
          reason === "one_point"
            ? `${which}: ${counted(marks, "point", "points")}, not used;` +
              ` mark at least ${MIN_LADDER_POINTS}`
            : `${which}: shares fewer than ${MIN_SHARED_MWS} marked MWs with the left one,` +
              " not used; mark the same bands on both",
        warning: true,
      });
    }
    const [low, high] = fit.range;
    if (low !== null && high !== null) {
      const where = two ? ", where both ladders reach" : "";
      const range = `${RANGE.format(low)} to ${RANGE.format(high)} kDa`;
      lines.push({ text: `Reads MWs from ${range}${where}` });
    }
    return lines;
  }

  // The group's marks, ladder by ladder, down the image: MW, height, source
  // and image, each with Remove; then Adjust per ladder, and Clear.
  renderMarks() {
    const points = this.group.points;
    const left = this.sidePoints("left");
    const right = this.sidePoints("right");
    $("cal-marks").hidden = !points.length;
    $("cal-marks-count").textContent = right.length
      ? `Marks: ${left.length} + ${right.length}`
      : `Marks: ${left.length}`;
    // Not while a ladder is being found either: its answer opens a ruler.
    const busy = Boolean(this.draft) || this.applying || this.finding;
    rebuild($("cal-points"), () => {
      for (const [side, marks] of [
        ["left", left],
        ["right", right],
      ]) {
        if (!marks.length) {
          continue;
        }
        const head = document.createElement("li");
        head.className = "cal-points-head";
        head.textContent = side === "left" ? "Ladder" : "Second ladder (right)";
        $("cal-points").append(head);
        for (const point of marks) {
          const image = this.project.images.find((each) => each.id === point.image_id);
          const elsewhere = image && image.id !== this.image.id;
          const on = elsewhere ? ` · on ${isolate(image.original_name)}` : "";
          const source = SOURCE_WORDS[point.source] || point.source;
          const remove = document.createElement("button");
          remove.type = "button";
          remove.dataset.key = `${side}:${point.mw}`;
          remove.textContent = "Remove";
          const ladder = side === "left" ? "ladder" : "second ladder";
          const name = `Remove the ${kdaText(point.mw)} kDa mark of the ${ladder}`;
          remove.setAttribute("aria-label", name);
          remove.addEventListener("click", () => this.removePoint(point));
          remove.disabled = busy; // the ruler replaces the ladder's marks: Apply or drop it first
          const item = document.createElement("li");
          const text = `${kdaText(point.mw)} kDa · y ${point.y.toFixed(1)} · ${source}${on}`;
          item.append(span("", text), remove);
          $("cal-points").append(item);
        }
      }
    });
    // A ladder of strip edges only has no band mark to put on a ruler.
    $("cal-adjust").hidden = !this.bandPoints("left").length;
    $("cal-adjust").disabled = busy;
    $("cal-adjust-right").hidden = !this.bandPoints("right").length;
    $("cal-adjust-right").disabled = busy;
    $("cal-clear").disabled = busy;
  }

  // --- Controls ---

  bind() {
    $("marker-image").addEventListener("change", (event) => {
      if (this.image) {
        this.link(this.image.id, event.target.value || null);
      }
    });
    $("cal-ladder").addEventListener("change", (event) => {
      const value = event.target.value;
      if (value === "custom") {
        this.openCustom();
        this.renderLadder(); // the select shows the ladder stored until the form sets one
      } else if (value && value !== "custom:stored") {
        this.setLadder({ ladder: value });
      }
    });
    $("cal-custom").addEventListener("submit", (event) => {
      event.preventDefault();
      this.submitCustom();
    });
    $("cal-custom-cancel").addEventListener("click", () => this.closeCustom(true));
    $("cal-custom").addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        this.closeCustom(true);
      }
    });
    $("cal-find").addEventListener("click", () => this.arm({ kind: "find", side: "left" }));
    $("cal-find-right").addEventListener("click", () => this.arm({ kind: "find", side: "right" }));
    $("cal-mark").addEventListener("click", () =>
      this.arm({ kind: "mark", side: "left", edges: false }),
    );
    $("cal-mark-edges").addEventListener("click", () =>
      this.arm({ kind: "mark", side: "left", edges: true }),
    );
    $("cal-up").addEventListener("click", () => this.shiftLabels(1));
    $("cal-down").addEventListener("click", () => this.shiftLabels(-1));
    $("cal-snap").addEventListener("click", () => this.snapAll());
    $("cal-apply").addEventListener("click", () => this.apply());
    $("cal-drop").addEventListener("click", () => this.escape());
    $("cal-adjust").addEventListener("click", () => this.adjust("left"));
    $("cal-adjust-right").addEventListener("click", () => this.adjust("right"));
    $("cal-clear").addEventListener("click", () => this.clear());
    const ticks = $("cal-ticks");
    ticks.addEventListener("keydown", (event) => this.tickKey(event));
    ticks.addEventListener("focusin", (event) => {
      const button = event.target.closest("button[data-id]");
      this.focusId = button ? Number(button.dataset.id) : null;
      this.view.setRuler(this.rulerShown());
    });
    ticks.addEventListener("focusout", (event) => {
      if (!ticks.contains(event.relatedTarget)) {
        this.focusId = null;
        this.view.setRuler(this.rulerShown());
      }
    });
    const menu = $("cal-menu");
    menu.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        this.closeMenu(true);
      }
    });
    $("cal-menu-cancel").addEventListener("click", () => this.closeMenu(true));
    // A press anywhere else closes the popup, as a click away from a menu does.
    document.addEventListener(
      "pointerdown",
      (event) => {
        if (this.menu && !menu.contains(event.target)) {
          this.closeMenu(false);
        }
      },
      true,
    );
  }

  // A key on a tick's button: ↑↓ move the tick (Shift: further), Enter
  // applies the ruler, Delete takes the tick off it (not a ladder band). Tab
  // goes to the next tick, Space or a click relabels it (the button's own).
  tickKey(event) {
    const button = event.target.closest("button[data-id]");
    if (!button || !this.draft || event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
    const id = Number(button.dataset.id);
    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      const step = event.shiftKey ? NUDGE_FAR : NUDGE;
      this.nudge(id, event.key === "ArrowUp" ? -step : step);
    } else if (event.key === "Enter") {
      event.preventDefault(); // not a click: that relabels
      event.stopPropagation();
      this.apply();
    } else if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      event.stopPropagation(); // not the page's: that deletes the selected box
      this.notABand(id);
    }
  }

  // Arm a tool (Find ladder, Mark ladder bands, Mark strip edges): the next
  // click on the image finds or marks. Pressed again, it is put away.
  arm(tool) {
    if (this.finding || this.applying) {
      return; // the answer awaited opens a ruler, or closes one
    }
    const same =
      this.tool &&
      this.tool.kind === tool.kind &&
      this.tool.side === tool.side &&
      Boolean(this.tool.edges) === Boolean(tool.edges);
    this.closeMenu(false);
    this.tool = same ? null : this.newTool(tool);
    this.handlers.changed();
    if (this.tool && this.tool.kind === "mark" && this.tool.edges) {
      this.handlers.status(
        "Mark an edge only where the cut runs through a marker band; enter that band's MW.",
      );
    } else {
      this.handlers.status("");
    }
  }

  // Whether the view's image and project are still those of `image` and `openId`.
  shows(image, openId) {
    return Boolean(
      this.image && this.image.id === image.id && this.project && this.project.open_id === openId,
    );
  }

  // --- The view's handlers ---

  // A click on the image while a tool is armed.
  pick(x, y, where) {
    const tool = this.tool;
    if (!tool || !this.image) {
      return;
    }
    if (tool.kind === "find") {
      this.tool = null;
      this.handlers.changed();
      this.find(tool.side, x);
    } else {
      this.askMark(tool, x, y, where);
    }
  }

  // The ruler dragged or clicked (view.js).
  ruler(step) {
    const draft = this.draft;
    if (!draft || this.applying) {
      if (step.phase === "drop" || step.phase === "cancel") {
        // A drag ended while Apply was sent (Enter during it): its steps were
        // not taken, and the next drag starts from the ruler as it is.
        this.base = null;
      }
      return;
    }
    if (step.phase === "click") {
      if (step.part.kind === "tick") {
        this.relabelMenu(step.part.id, null, step);
      }
      return;
    }
    if (step.phase === "cancel") {
      if (this.base) {
        this.draft = this.base;
        this.base = null;
        this.drawOnView();
      }
      return;
    }
    if (!this.base) {
      this.base = draft;
      this.closeMenu(false);
    }
    const moved = this.moved(this.base, step);
    if (step.phase === "move") {
      this.draft = moved;
      this.view.setRuler(this.rulerShown());
      return;
    }
    this.base = null;
    this.draft = { ...moved, ticks: sortTicks(moved.ticks), error: "" };
    if (step.part.kind === "tick" && !step.alt) {
      this.snapTick(step.part.id);
    }
    if (step.part.kind === "body" && step.axis === "x") {
      this.handlers.changed(); // the view's hint may go to the other side
    } else {
      this.refresh();
    }
  }

  // The ruler `base` as a drag moves it: its line up and down (every tick) or
  // sideways (its x), one of its ends stretched (every tick moves linearly in
  // y between the other end, which stays, and the dragged one), or one tick.
  // Every tick stays on the image. A tick a drag of the line or a stretch
  // moves is no longer on its band: placed by hand, if it was solid.
  moved(base, { part, dx, dy, axis }) {
    const { width, height } = this.image;
    const onImage = (y) => Math.min(height, Math.max(0, y));
    const shifted = (state) => (state === "predicted" ? state : "hand");
    const ys = base.ticks.filter(shown).map((tick) => tick.y); // as the view draws its ends
    if (part.kind === "body" && axis === "x") {
      return { ...base, x: Math.min(width, Math.max(0, base.x + dx)) };
    }
    if (part.kind === "body") {
      const shift = Math.min(height - Math.max(...ys), Math.max(-Math.min(...ys), dy));
      return {
        ...base,
        ticks: base.ticks.map((tick) =>
          shift
            ? { ...tick, y: onImage(tick.y + shift), state: shifted(tick.state), snappedX: null }
            : tick,
        ),
      };
    }
    if (part.kind === "grip") {
      const top = Math.min(...ys);
      const bottom = Math.max(...ys);
      const [fixed, end] = part.end === "top" ? [bottom, top] : [top, bottom];
      if (end === fixed) {
        return base; // one tick: nothing to stretch
      }
      let dragged = onImage(end + dy);
      dragged =
        part.end === "top"
          ? Math.min(dragged, fixed - MIN_SPAN)
          : Math.max(dragged, fixed + MIN_SPAN);
      const factor = (dragged - fixed) / (end - fixed);
      return {
        ...base,
        ticks: base.ticks.map((tick) =>
          factor === 1
            ? tick
            : {
                ...tick,
                y: onImage(fixed + (tick.y - fixed) * factor),
                state: tick.y === fixed ? tick.state : shifted(tick.state),
                snappedX: tick.y === fixed ? tick.snappedX : null,
              },
        ),
      };
    }
    return {
      ...base,
      ticks: base.ticks.map((tick) =>
        tick.id === part.id
          ? { ...tick, y: onImage(tick.y + dy), state: "hand", snappedX: null }
          : tick,
      ),
    };
  }

  // A ladder mark dragged (moved, snapped to its band unless Alt is held: one
  // step each) or clicked (relabel or remove).
  ladderPoint({ phase, tick, y, alt, clientX, clientY }) {
    if (phase === "drop") {
      this.movePoint(tick.point, y, !alt);
    } else if (phase === "click") {
      this.pointMenu(tick.point, { clientX, clientY });
    }
  }

  // --- Finding a ladder ---

  // Find the ladder whose lane was clicked at `x` on the shown image: a ruler
  // of its proposal to adjust; where none stands out, marking by clicks. While
  // it is found, the controls that open a ruler or a popup, or change the
  // ladder, are disabled (renderLocks, renderMarking, renderMarks).
  async find(side, x) {
    const image = this.image;
    const openId = this.project.open_id;
    if (this.finding || this.draft) {
      return;
    }
    const scope = [this.ladderScope(), this.groupScope()];
    this.finding = true;
    this.refresh();
    this.handlers.status("Finding the ladder…");
    let answer = null;
    let refusal = null;
    try {
      answer = await this.handlers.read("POST", `/api/images/${image.id}/ladder-proposal`, {
        x,
        side,
      });
    } catch (error) {
      refusal = error;
    } finally {
      this.finding = false;
      if (this.membrane) {
        this.refresh();
      }
    }
    if (!this.shows(image, openId)) {
      return;
    }
    const changedHere = this.ladderScope() !== scope[0] || this.groupScope() !== scope[1];
    if (refusal) {
      // A ladder that lists no MWs, as the server holds it (detail.ladder_kda):
      // nothing can be found on it. Changed elsewhere (another tab chose one
      // without MWs, or took the ladder back) while this page listed some, it
      // is read again first; the refusal's own words name an id.
      const listed = refusal.detail && refusal.detail.ladder_kda;
      if (!Array.isArray(listed)) {
        this.handlers.report(refusal);
      } else if (changedHere || sameLadder(listed, this.ladderKda())) {
        this.handlers.status(
          "The ladder or the marker link changed while the ladder was being found:" +
            ` ${this.findAgain()}.`,
        );
      } else {
        this.sayOnceRead(
          openId,
          () => `The membrane's ladder was changed (in another tab, say): ${this.findAgain()}.`,
        );
      }
      return;
    }
    if (this.draft || changedHere) {
      // Undo, Redo or another tab changed the ladder or the marker link
      // meanwhile: the proposal's labels, or the marks it would replace, are
      // no longer those.
      this.handlers.status(
        "The ladder or the marker link changed while the ladder was being found:" +
          ` ${this.findAgain()}.`,
      );
      return;
    }
    // The ladder the proposal is labelled with, as the server held it: this
    // page's copy may be older (another tab chose another ladder since it was
    // read), and a ruler's labels must be the ladder the page shows.
    const labels = answer.ladder_kda;
    if (!sameLadder(labels, this.ladderKda())) {
      this.handlers.status(
        "The membrane's ladder was changed (in another tab, say): it is shown now. Find ladder" +
          " again.",
      );
      this.handlers.reread();
      return;
    }
    const proposal = answer.proposal;
    if (!proposal) {
      // Fewer than two bands stand out there: mark the ladder's bands by clicking them.
      this.tool = this.newTool({ kind: "mark", side, edges: false });
      this.handlers.changed();
      this.handlers.status(
        `No ladder stands out where you clicked (x = ${Math.round(x)}): fewer than two bands` +
          " there. Click each of the ladder's bands to mark it, or Find ladder again on its lane;" +
          " Esc ends marking.",
      );
      return;
    }
    this.openDraft({
      side,
      x: proposal.x,
      foundAt: proposal.x,
      doubtful: proposal.doubtful,
      extra: [...proposal.extra],
      ladderKda: [...labels],
      ticks: proposal.ticks.map((tick) =>
        newTick(tick.mw, tick.y, tick.found ? "found" : "predicted", labels),
      ),
    });
    this.handlers.status(this.proposalText(proposal));
  }

  // What a proposal holds, for the status line. A ruler that is not doubtful
  // is still only the best labelling of the peaks down the lane clicked, which
  // a lane of samples has too: the line asks to check it is the ladder.
  proposalText(proposal) {
    const found = proposal.ticks.filter((tick) => tick.found).length;
    const predicted = proposal.ticks.length - found;
    const parts = [`${counted(found, "label", "labels")} on peaks found down the lane`];
    if (predicted) {
      parts.push(`${predicted} predicted (hollow: stored only once snapped or placed)`);
    }
    if (proposal.extra.length) {
      parts.push(`${counted(proposal.extra.length, "peak", "peaks")} left unlabelled (grey)`);
    }
    const head = `Ruler at x = ${Math.round(proposal.x)}: ${parts.join(", ")}.`;
    if (proposal.doubtful) {
      return `${head} ${this.doubtText()}`;
    }
    return (
      `${head} Check that this lane is the ladder and that each label sits on its band, then` +
      " Apply (Enter); Esc drops the ruler."
    );
  }

  openDraft(fields) {
    this.closeMenu(false);
    this.tool = null;
    this.draft = {
      imageId: this.image.id,
      openId: this.project.open_id,
      ladder: this.ladderScope(),
      group: this.groupScope(),
      // The ladder MWs its labels are, which Apply names to the server
      // (calibration_changed): a proposal's (find), or else the page's.
      ladderKda: [...this.ladderKda()],
      groupIds: [...this.group.image_ids],
      error: "",
      ...fields,
      ticks: sortTicks(fields.ticks),
    };
    this.base = null;
    this.focusId = null;
    this.handlers.changed();
    // In view in the side panel: its Apply is the next thing to press.
    $("cal-draft").scrollIntoView({ block: "nearest" });
  }

  closeDraft() {
    this.draft = null;
    this.base = null;
    this.focusId = null;
    this.closeMenu(false);
  }

  // Reopen the ruler from a ladder's band marks, on the image holding most of
  // them (shown first if it is not), with a hollow tick where its marks put
  // each ladder MW not marked. Only the marks Apply would store as they are go
  // on the ruler, and set its x: those of bands on that image. A strip edge,
  // or a mark on another image of the group, would be stored as a band of
  // this image at the ruler's x: it is left off, and the panel says Apply
  // removes it (lostPoints).
  adjust(side) {
    const bands = this.bandPoints(side);
    if (!bands.length || this.draft || this.applying || this.finding) {
      return;
    }
    const on = new Map();
    for (const mark of bands) {
      on.set(mark.image_id, (on.get(mark.image_id) || 0) + 1);
    }
    const [imageId] = [...on].sort((a, b) => b[1] - a[1])[0];
    if (imageId !== this.image.id) {
      this.handlers.showImage(imageId);
    }
    const image = this.image;
    if (!image || image.id !== imageId) {
      return;
    }
    const marks = bands.filter((mark) => mark.image_id === imageId);
    const xs = marks.map((mark) => mark.x).filter((x) => x !== null);
    const x = xs.length ? median(xs) : side === "left" ? 0 : image.width;
    const kda = this.ladderKda();
    const ticks = marks.map((mark) => newTick(mark.mw, mark.y, "stored", kda));
    for (const mw of kda) {
      if (marks.some((mark) => sameMw(mark.mw, mw))) {
        continue;
      }
      const y = ladderY(marks, mw);
      if (y !== null && y >= 0 && y <= image.height) {
        ticks.push(newTick(mw, y, "predicted", kda));
      }
    }
    const ruler = orderedHollow(sortTicks(ticks));
    this.openDraft({ side, x, foundAt: null, doubtful: false, extra: [], ticks: ruler });
    if (this.focusLost()) {
      // Adjust is disabled while the ruler is open: the keyboard goes on to it.
      const first = $("cal-ticks").querySelector("button");
      (first || $("cal-apply")).focus();
    }
    const predicted = ruler.length - marks.length;
    const which = side === "right" ? "second ladder" : "ladder";
    const more = predicted ? `, and ${predicted} predicted (hollow)` : "";
    this.handlers.status(
      `Ruler from the ${which}'s ${counted(marks.length, "mark", "marks")}${more}: adjust it,` +
        " then Apply (Enter); Esc drops it, and the marks stay as they are.",
    );
  }

  // --- Adjusting the ruler ---

  // Move one tick by `dy` (the arrow keys): placed by hand.
  nudge(id, dy) {
    const draft = this.draft;
    if (!draft || this.applying) {
      return;
    }
    const height = this.image.height;
    this.draft = {
      ...draft,
      error: "",
      ticks: sortTicks(
        draft.ticks.map((tick) =>
          tick.id === id
            ? {
                ...tick,
                y: Math.min(height, Math.max(0, tick.y + dy)),
                state: "hand",
                snappedX: null,
              }
            : tick,
        ),
      ),
    };
    this.refresh();
  }

  // The solid ticks whose MW the ladder's list does not hold (a mark of
  // another MW, typed while marking): ▲▼ have no band to move their labels to.
  unlisted() {
    const draft = this.draft;
    return draft ? draft.ticks.filter((tick) => kept(tick) && tick.index === null) : [];
  }

  // Why ▲▼ have no band to move a label to, or "": the ladder lists no MWs,
  // or a tick's MW is not in its list (unlisted).
  shiftRefusal() {
    if (!this.ladderKda().length) {
      return (
        "The ladder lists no MWs, so ▲▼ have no band to move a label to: relabel a tick by" +
        " typing its MW (click it, or Space)"
      );
    }
    const unlisted = this.unlisted();
    if (!unlisted.length) {
      return "";
    }
    const be = unlisted.length === 1 ? "is" : "are";
    return (
      `${inWords(unlisted.map((tick) => kdaText(tick.mw)))} kDa ${be} not in the ladder's list,` +
      " so ▲▼ have no band to move a label to: relabel each tick instead (click it, or Space)"
    );
  }

  // Whether ▲ (1) or ▼ (−1) leaves a solid tick with a label; never while a
  // tick's MW is not in the ladder's list (unlisted): moved one band, the
  // other labels would pass it.
  canShift(step) {
    if (this.unlisted().length) {
      return false;
    }
    const count = this.ladderKda().length;
    const draft = this.draft;
    const moves = (tick) => {
      const index = tick.index === null ? -1 : tick.index + step;
      return solid(tick) && index >= 0 && index < count;
    };
    return Boolean(draft && draft.ticks.some(moves));
  }

  // ▲ (1): every label moves up one band, so each band reads the next lighter
  // MW; ▼ (−1): down, the next heavier. A band whose label moves past the end
  // of the ladder's list has none (and is not stored) until the labels move
  // back or it is relabelled: ▲ then ▼ gives the ruler back as it was.
  shiftLabels(step) {
    const draft = this.draft;
    if (!draft || this.applying || !this.canShift(step)) {
      return;
    }
    const kda = this.ladderKda();
    const ticks = draft.ticks.map((tick) => {
      if (tick.index === null) {
        return tick; // an MW the ladder's list does not hold
      }
      const index = tick.index + step;
      return { ...tick, index, mw: index >= 0 && index < kda.length ? kda[index] : null };
    });
    this.draft = { ...draft, ticks, error: "" };
    const top = ticks.find(kept);
    const bare = ticks.filter((tick) => solid(tick) && !labelled(tick)).length;
    const parts = [`Every label moved ${step > 0 ? "up" : "down"} one band`];
    if (top) {
      parts.push(`the top band reads ${kdaText(top.mw)} kDa now`);
    }
    if (bare) {
      parts.push(
        `${counted(bare, "band has", "bands have")} no label left (?): not stored unless` +
          " relabelled or the labels move back",
      );
    }
    this.handlers.status(`${parts.join("; ")}.`);
    this.refresh();
    if (this.focusLost()) {
      // Its button is disabled now (the labels are at the end of the list):
      // the keyboard goes on to the other one, or to Apply.
      const other = $(step > 0 ? "cal-down" : "cal-up");
      (other.disabled ? $("cal-apply") : other).focus();
    }
  }

  // The snap route on the ruler's ticks, as they are now: [{y, snapped}] per
  // tick, each within the gaps to the others; null if refused (said).
  async snapped(draft) {
    try {
      const answer = await this.handlers.read("POST", `/api/images/${draft.imageId}/ladder-snap`, {
        x: draft.x,
        ys: draft.ticks.map((tick) => tick.y),
      });
      return answer.points;
    } catch (error) {
      if (this.draft === draft) {
        this.handlers.report(error);
      }
      return null;
    }
  }

  // Snap all: each tick onto the band nearest it, at the ruler's x (a tick on
  // a band found stays where it was found). A hollow tick that snaps becomes
  // solid; one where nothing stands out stays as it was, and the status line
  // names the solid ones among those: Apply stores them where they are.
  async snapAll() {
    const draft = this.draft;
    if (!draft || this.applying) {
      return;
    }
    const results = await this.snapped(draft);
    if (!results || this.draft !== draft) {
      if (results && this.draft) {
        this.handlers.status("The ruler changed while it was snapped: Snap all again.");
      }
      return;
    }
    let moved = 0;
    const stayed = []; // solid ticks to store, not on a band found, that no band took
    const ticks = draft.ticks.map((tick, index) => {
      const result = results[index];
      if (tick.state === "found") {
        return tick;
      }
      if (!result.snapped) {
        if (kept(tick)) {
          stayed.push(tick);
        }
        return tick;
      }
      moved += 1;
      return { ...tick, y: result.y, state: "snapped", snappedX: draft.x };
    });
    this.draft = { ...draft, ticks: sortTicks(ticks), error: "" };
    const parts = [`Snapped ${counted(moved, "tick", "ticks")} onto the band nearest each`];
    if (stayed.length) {
      const one = stayed.length === 1;
      const mws = `${inWords(stayed.map((tick) => kdaText(tick.mw)))} kDa`;
      parts.push(
        `${mws} ${one ? "stays where it is" : "stay where they are"}: no band stands out near` +
          ` ${one ? "it" : "them"}, and Apply stores ${one ? "it" : "them"} there`,
      );
    }
    const hollow = ticks.filter((tick) => !solid(tick) && labelled(tick)).length;
    if (hollow) {
      parts.push(`${hollow} still hollow (no band stands out there): not stored`);
    }
    this.handlers.status(`${parts.join("; ")}.`);
    this.refresh();
  }

  // A tick dropped without Alt: onto the band nearest it, as Snap all would
  // put it; where none stands out it stays where it was dropped. The ruler may
  // have changed while the snap was asked for (another tick dragged, a nudge,
  // ▲▼, a relabel): the snap still lands on the tick, by its id, if the tick
  // is still where it was dropped on a ruler at the same x; a later move of
  // it (or of the ruler) stands.
  snapTick(id) {
    const draft = this.draft;
    const run = async () => {
      const results = await this.snapped(draft);
      const index = draft.ticks.findIndex((tick) => tick.id === id);
      if (!results || index < 0) {
        return;
      }
      const dropped = draft.ticks[index];
      const still = (ruler) => {
        const now =
          ruler && ruler.imageId === draft.imageId && ruler.x === draft.x
            ? ruler.ticks.find((tick) => tick.id === id)
            : null;
        return Boolean(now && now.y === dropped.y && now.state === dropped.state);
      };
      if (!still(this.draft)) {
        return; // moved again, or the ruler was dropped or applied: that stands
      }
      if (!results[index].snapped) {
        this.handlers.status(
          `${tickName(this.draft.ticks.find((tick) => tick.id === id))}: no band stands out` +
            " where you dropped it; it stays there (placed by hand).",
        );
        return;
      }
      const onBand = (ruler) => ({
        ...ruler,
        ticks: sortTicks(
          ruler.ticks.map((each) =>
            each.id === id
              ? { ...each, y: results[index].y, state: "snapped", snappedX: draft.x }
              : each,
          ),
        ),
      });
      this.draft = onBand(this.draft);
      if (still(this.base)) {
        this.base = onBand(this.base); // a drag under way goes on from the snapped tick
      }
      this.refresh();
    };
    this.snapping = this.snapping.then(run, run);
  }

  // The popup of a ruler tick: every ladder MW (those that would put a solid
  // tick out of order, or on another's MW, disabled), and "Not a ladder band".
  relabelMenu(id, back, where = null) {
    const draft = this.draft;
    const tick = draft && draft.ticks.find((each) => each.id === id);
    if (!tick || this.applying) {
      return;
    }
    const others = draft.ticks.filter((each) => each.id !== id && kept(each));
    const now = labelled(tick) ? `now ${kdaText(tick.mw)} kDa` : "no label now";
    const choices = this.ladderKda().map((mw) => ({
      mw,
      current: labelled(tick) && sameMw(mw, tick.mw),
      disabled: others.some((each) => sameMw(each.mw, mw))
        ? `${kdaText(mw)} kDa is on another band`
        : inOrder(others, tick.y, mw)
          ? ""
          : "out of order with the bands above and below it",
    }));
    this.openMenu({
      title: `Label the band at y = ${tick.y.toFixed(1)} (${now})`,
      choices,
      // With no MWs listed to choose from, one is typed (▲▼ say so).
      other: !choices.length,
      otherLabel: "Relabel",
      choose: (mw) => this.relabel(id, mw),
      extra: { label: "Not a ladder band", run: () => this.notABand(id) },
      where: where || this.view.clientOf(draft.x, tick.y),
      back,
    });
  }

  relabel(id, mw) {
    const draft = this.draft;
    if (!draft) {
      return;
    }
    // A hollow tick of that MW goes, and so does one now out of order.
    const index = this.ladderKda().findIndex((each) => sameMw(each, mw));
    const ticks = draft.ticks
      .map((tick) => (tick.id === id ? { ...tick, mw, index: index < 0 ? null : index } : tick))
      .filter((tick) => tick.id === id || solid(tick) || !labelled(tick) || !sameMw(tick.mw, mw));
    this.draft = { ...draft, ticks: orderedHollow(ticks), error: "" };
    this.refresh();
  }

  // Take a tick off the ruler: the peak it was on is no ladder band (a speck,
  // a smear); it stays drawn as a grey dot, a peak no label took. A tick
  // predicted or placed by hand was on no peak: it leaves no dot.
  notABand(id) {
    const draft = this.draft;
    const index = draft ? draft.ticks.findIndex((tick) => tick.id === id) : -1;
    if (index < 0 || this.applying) {
      return;
    }
    const tick = draft.ticks[index];
    const place = draft.ticks.filter(shown).findIndex((each) => each.id === id); // its button's
    const ticks = draft.ticks.filter((each) => each.id !== id);
    const onPeak = solid(tick) && tick.state !== "hand";
    const extra = onPeak ? [...draft.extra, tick.y].sort((a, b) => a - b) : draft.extra;
    this.draft = { ...draft, ticks, extra, error: "" };
    const hadFocus = this.focusId === id;
    this.refresh();
    this.handlers.status(`Took ${tickName(tick)} off the ruler: not a ladder band.`);
    if (hadFocus) {
      const buttons = [...$("cal-ticks").querySelectorAll("button")];
      const next = buttons[Math.min(place, buttons.length - 1)] || $("cal-apply");
      next.focus();
    }
  }

  // Apply: the solid ticks become the ladder's marks, all at the ruler's x, in
  // one change (one undo step); the hollow ones are not stored. A snapped tick
  // whose ruler moved sideways since is snapped again at the x it is stored
  // at, so the server finds it where a snap there puts it.
  async apply() {
    if (!this.draft || this.applying) {
      return;
    }
    const had = document.activeElement; // Apply, or a tick: Apply is disabled while it is sent
    this.applying = true;
    this.refresh();
    try {
      await this.snapping; // a dropped tick's snap lands first
      let draft = this.draft;
      const moved = (tick) => tick.state === "snapped" && tick.snappedX !== draft.x;
      if (draft && draft.ticks.some(moved)) {
        draft = await this.snapAgain(draft);
      }
      if (!draft) {
        return;
      }
      const ticks = draft.ticks.filter(kept);
      if (!ticks.length) {
        const error = "No tick to store: snap the ruler (Snap all), or drag a tick onto a band.";
        this.draft = { ...draft, error };
        return;
      }
      const path = `/api/images/${draft.imageId}/calibration/${draft.side}/ladder`;
      // With what the ruler was opened with: the server refuses it if either
      // changed since (calibration_changed), which this page may not know of.
      const body = {
        x: draft.x,
        points: ticks.map((tick) => ({ y: tick.y, mw: tick.mw })),
        found_at: draft.foundAt,
        ladder_kda: draft.ladderKda,
        group: draft.groupIds,
      };
      const lost = this.lostPoints(draft);
      let before = null; // the state shown when it was sent
      let answer = null;
      try {
        answer = await this.handlers.edit("PUT", path, body, {
          refused: (error) => this.applyRefused(error, draft),
          sent: (project) => {
            before = project;
          },
        });
      } catch {
        return; // refused: said, and the ruler stays to fix
      }
      if (!answer) {
        return;
      }
      this.closeDraft();
      this.applying = false;
      this.handlers.changed();
      const unchanged =
        before !== null &&
        before.open_id === answer.project.open_id &&
        before.revision === answer.project.revision;
      if (unchanged) {
        // The ruler holds the marks as they are stored (Adjust, then Apply):
        // nothing was logged, so there is nothing to undo, and the last change
        // is another one.
        const which = draft.side === "right" ? "second ladder's" : "ladder's";
        this.handlers.status(`No change: the ruler holds the ${which} marks as they are stored.`);
        this.keepFocus();
        return;
      }
      const undo = this.undoOf(answer, "set_ladder_points", "Undo applying the ladder");
      const button = this.handlers.status(this.appliedText(answer, draft, lost), undo);
      if (button && this.focusLost()) {
        button.focus();
      } else {
        this.keepFocus();
      }
    } finally {
      this.applying = false;
      if (this.membrane) {
        this.refresh();
      }
      if (this.focusLost()) {
        this.focusAfterApply(had);
      }
    }
  }

  // Where the keyboard goes once Apply is answered, the control it was on
  // (`had`) disabled or gone meanwhile. Not applied, the ruler kept to fix:
  // back to `had`, or Apply, or the ruler's first tick. The ruler dropped
  // (calibration_changed): on to Find ladder (keepFocus).
  focusAfterApply(had) {
    if (!this.draft) {
      this.keepFocus();
      return;
    }
    const usable = (control) =>
      control && control.isConnected && !control.disabled && control.getClientRects().length;
    const first = $("cal-ticks").querySelector("button");
    const target = [had, $("cal-apply"), first].find(usable);
    if (target) {
      target.focus();
    }
  }

  // A refusal of the ruler's Apply. What it was opened with changed since
  // (calibration_changed: another tab chose another ladder, or linked or
  // unlinked an image): the ruler is dropped, the page reads the project as
  // it is, then the status line says why, and what finds the ladder now
  // (findAgain: the ladder read may list no MWs). Otherwise the ruler stays
  // to fix, the refusal in words under it and in the status line.
  applyRefused(error, draft) {
    if (error.code === "calibration_changed") {
      if (this.draft !== draft) {
        // This page dropped the ruler already, and said why (staleRuler): an
        // Undo queued before Apply, say, changed its ladder first.
        return;
      }
      this.closeDraft(); // the keyboard goes on once it is answered (focusAfterApply)
      this.handlers.changed();
      const why = CHANGED_WORDS[error.detail && error.detail.changed];
      this.sayOnceRead(
        this.project.open_id,
        () =>
          why
            ? `The ruler was not applied: ${why}. Nothing was stored; ${this.findAgain()}.`
            : sentence(error.message),
        true,
      );
      return;
    }
    const text = refusalText(error, "Not applied");
    if (this.draft === draft) {
      this.draft = { ...draft, error: text };
    }
    this.handlers.report(error, text);
  }

  // The ruler with each snapped tick moved sideways since snapped again at
  // the ruler's x (placed by hand where nothing stands out there); null if the
  // ruler changed meanwhile or the snap was refused.
  async snapAgain(draft) {
    const results = await this.snapped(draft);
    if (!results || this.draft !== draft) {
      return null;
    }
    const ticks = draft.ticks.map((tick, index) => {
      if (tick.state !== "snapped" || tick.snappedX === draft.x) {
        return tick;
      }
      return results[index].snapped
        ? { ...tick, y: results[index].y, snappedX: draft.x }
        : { ...tick, state: "hand", snappedX: null };
    });
    this.draft = { ...draft, ticks: sortTicks(ticks) };
    return this.draft;
  }

  // What Apply stored, from its answer: the marks by how each was placed (the
  // server's own judgement), those relabelled, those of `lost` (lostPoints)
  // it removed, a swap of sides, and the tilt.
  appliedText(answer, draft, lost) {
    const placed = answer.points;
    const count = (how) => placed.filter((point) => point.placed === how).length;
    const parts = [
      [count("found"), "found"],
      [count("snapped"), "snapped"],
      [count("hand"), "placed by hand"],
    ]
      .filter(([n]) => n)
      .map(([n, how]) => `${n} ${how}`);
    const side = placed.length ? placed[0].side : draft.side;
    const which = side === "right" ? "second ladder (right)" : "ladder";
    const sentences = [
      `Applied the ${which}: ${counted(placed.length, "mark", "marks")} (${inWords(parts)}).`,
    ];
    const relabelled = placed.filter((point) => point.relabelled).length;
    if (relabelled) {
      sentences.push(`${counted(relabelled, "label", "labels")} changed from the proposal.`);
    }
    const held = answer.project.membranes.flatMap((membrane) =>
      membrane.groups.flatMap((group) => group.points),
    );
    const same = (a, b) =>
      a.image_id === b.image_id && a.source === b.source && a.y === b.y && sameMw(a.mw, b.mw);
    const removed = lost.filter((point) => !held.some((other) => same(point, other)));
    if (removed.length) {
      sentences.push(`Removed ${this.lostWords(removed)}, which the ruler did not hold.`);
    }
    const fit = answer.fit;
    const two = fit && fit.ladders.length === 2;
    if (answer.sides_swapped && two) {
      sentences.push(
        "It lies left of the other ladder, so it is the left ladder now and the other one the" +
          " right.",
      );
    }
    if (two && fit.offset_px !== null) {
      const offset = Math.abs(fit.offset_px);
      sentences.push(
        `Left and right differ by ${offset.toFixed(offset < 10 ? 1 : 0)} px, about` +
          ` ${Math.abs(fit.tilt_deg).toFixed(1)}°: MWs follow that tilt across the lanes.`,
      );
    }
    return sentences.join(" ");
  }

  // --- Marking by clicks ---

  // A click while marking: the popup asks which MW it is (a strip edge's, the
  // band the cut runs through); with two marks on the ladder already, the MW
  // they put there, nearest unmarked, is offered first. An MW chosen from the
  // ladder's list names that list to the server (mark), one typed none.
  askMark(tool, x, y, where) {
    const image = this.image; // a choice marks the image clicked
    const side = this.markSide(tool, x);
    const marks = this.sidePoints(side);
    const kda = this.ladderKda();
    let proposed = null;
    const at = ladderMw(marks, y);
    if (at !== null && kda.length) {
      const free = kda.filter((mw) => !marks.some((mark) => sameMw(mark.mw, mw)));
      proposed = free.length
        ? free.reduce((best, mw) =>
            Math.abs(Math.log10(mw / at)) < Math.abs(Math.log10(best / at)) ? mw : best,
          )
        : null;
    }
    const choices = kda.map((mw) => ({
      mw,
      current: false,
      disabled: marks.some((mark) => sameMw(mark.mw, mw))
        ? "already marked"
        : inOrder(marks, y, mw)
          ? ""
          : "out of order with the bands marked above and below",
    }));
    const ladder = side === "right" ? "second ladder" : "ladder";
    this.openMenu({
      title: tool.edges
        ? `The cut at y = ${y.toFixed(1)} runs through the marker band of`
        : `Which band of the ${ladder} is this (y = ${y.toFixed(1)})?`,
      choices,
      proposed,
      other: true,
      choose: (mw, listed) => this.mark(tool, side, image, x, y, mw, listed ? kda : null),
      where,
      back: null,
    });
  }

  // The ladder a click while marking marks: a strip edge's is the left one;
  // Mark ladder bands marks the second ladder where the click lies nearer the
  // image's right edge than the first ladder's lane (its marks' median x),
  // as Find second ladder would, not the first ladder at a far lane.
  markSide(tool, x) {
    if (tool.edges || tool.side === "right") {
      return tool.edges ? "left" : "right";
    }
    const xs = this.sidePoints("left")
      .filter((point) => point.source !== "strip_edge" && point.x !== null)
      .map((point) => point.x);
    if (!xs.length) {
      return "left";
    }
    const lane = median(xs);
    return x > lane + (this.image.width - lane) / 2 ? "right" : "left";
  }

  // Mark `mw` where the click was: `kda` is the ladder's list it was chosen
  // from (null: typed), which the server refuses once it is no longer the
  // membrane's (calibration_changed: another tab chose another ladder), so
  // no MW of a ladder before is stored.
  async mark(tool, side, image, x, y, mw, kda) {
    const source = tool.edges
      ? "strip_edge"
      : image.kind === "chemiluminescence"
        ? "chemiluminescence_marker"
        : "visible_marker";
    const openId = this.project.open_id;
    const body = { y, mw, source, x, snap: true, ...(kda ? { ladder_kda: kda } : {}) };
    let answer = null;
    try {
      answer = await this.handlers.edit(
        "POST",
        `/api/images/${image.id}/calibration/${side}/points`,
        body,
        {
          refused: (error) => {
            if (error.code !== "calibration_changed") {
              this.handlers.report(error, refusalText(error, "Not marked"));
            }
          },
        },
      );
    } catch (error) {
      if (error.code === "calibration_changed") {
        const what = tool.edges ? "edge" : "band";
        this.ladderChanged("Not marked", kda, image, openId, (listed) => {
          const armed = this.tool && this.tool.kind === "mark" && this.tool.edges === tool.edges;
          const button = tool.edges ? "Mark strip edges" : "Mark ladder bands";
          const click = armed ? `click the ${what} again` : `press ${button} and click the ${what}`;
          return listed === null
            ? `no ladder is chosen now: choose it, then mark the ${what} again`
            : listed
              ? `${click} to choose its MW from the ladder shown now`
              : `the ladder lists no MWs now: ${click} and type its MW`;
        });
      }
      return; // refused: said
    }
    if (!answer) {
      return;
    }
    const what = tool.edges ? "edge" : "band";
    const how = answer.point.snapped
      ? `snapped to the ${what}`
      : "kept where you clicked: nothing stands out there";
    const more = this.tool ? ` Click the next ${what}, or Esc to end marking.` : "";
    const on = side === "right" ? " on the second ladder" : "";
    this.handlers.status(
      `Marked ${kdaText(mw)} kDa${on} (${how}).${more}`,
      this.undoOf(answer, "add_calibration_point", `Undo marking ${kdaText(mw)} kDa`),
    );
  }

  // --- The marks ---

  // The popup of a mark on the image: relabel it (one step), or remove it. A
  // label chosen from the ladder's list names that list to the server
  // (editPoint), one typed none.
  pointMenu(point, where) {
    const image = this.image;
    const kda = this.ladderKda();
    const others = this.sidePoints(point.side).filter((mark) => !sameMw(mark.mw, point.mw));
    const choices = kda.map((mw) => ({
      mw,
      current: sameMw(mw, point.mw),
      disabled: sameMw(mw, point.mw)
        ? ""
        : others.some((mark) => sameMw(mark.mw, mw))
          ? "already marked"
          : inOrder(others, point.y, mw)
            ? ""
            : "out of order with the bands marked above and below",
    }));
    this.openMenu({
      title: `The ${kdaText(point.mw)} kDa mark: relabel it`,
      choices,
      other: !choices.length, // no MWs listed: one is typed
      otherLabel: "Relabel",
      choose: (mw, listed) => {
        if (!sameMw(mw, point.mw)) {
          const said = `Relabelled the ${kdaText(point.mw)} kDa mark ${kdaText(mw)} kDa.`;
          this.editPoint(point, listed ? { mw, ladder_kda: kda } : { mw }, said, image);
        }
      },
      extra: { label: "Remove this mark", run: () => this.removePoint(point) },
      where,
      back: null,
    });
  }

  pointPath(point) {
    const mw = encodeURIComponent(String(point.mw));
    return `/api/images/${point.image_id}/calibration/${point.side}/points/${mw}`;
  }

  // A mark dragged to `y`: moved there, or onto the band nearest it with
  // `snap` (one step).
  movePoint(point, y, snap) {
    const image = this.project.images.find((each) => each.id === point.image_id);
    const to = Math.min(image ? image.height : y, Math.max(0, y));
    this.editPoint(point, { y: to, snap }, null);
  }

  // Move or relabel a mark (`body`: {y, snap} or {mw, ladder_kda?}, as the
  // route takes it), `said` the status line of a relabel; `image` the image
  // shown when its popup opened.
  async editPoint(point, body, said, image = this.image) {
    const openId = this.project.open_id;
    const verb = body.mw === undefined ? "Not moved" : "Not relabelled";
    let answer = null;
    try {
      answer = await this.handlers.edit("PATCH", this.pointPath(point), body, {
        refused: (error) => {
          if (error.code !== "calibration_changed") {
            this.handlers.report(error, refusalText(error, verb));
          }
        },
      });
    } catch (error) {
      if (error.code === "calibration_changed" && image) {
        this.ladderChanged(verb, body.ladder_kda, image, openId, (listed) =>
          listed
            ? "click the mark again to relabel it from the ladder shown now"
            : `${listed === null ? "no ladder is chosen" : "the ladder lists no MWs"} now:` +
              " click the mark again and type its new MW",
        );
      }
      return; // refused: said, and the mark is drawn where it is stored
    }
    if (!answer) {
      return;
    }
    let text = said;
    const stays =
      said === null &&
      sameMw(answer.point.mw, point.mw) &&
      Math.abs(answer.point.y - point.y) < 1e-9;
    if (stays) {
      // A snap put it back where it was: nothing changed, and nothing was logged.
      this.handlers.status(
        `The ${kdaText(point.mw)} kDa mark stays on its band (Alt: exactly where you drop it).`,
      );
      return;
    }
    if (text === null) {
      const where = answer.point.snapped
        ? " onto its band"
        : body.snap
          ? " (no band stands out there: kept where you dropped it)"
          : "";
      text = `Moved the ${kdaText(point.mw)} kDa mark${where}.`;
    }
    const name = `Undo changing the ${kdaText(point.mw)} kDa mark`;
    this.handlers.status(text, this.undoOf(answer, "edit_calibration_point", name));
  }

  async removePoint(point) {
    let answer = null;
    try {
      answer = await this.handlers.edit("DELETE", this.pointPath(point));
    } catch {
      return;
    }
    if (answer) {
      const name = `Undo removing the ${kdaText(point.mw)} kDa mark`;
      this.handlers.status(
        `Removed the ${kdaText(point.mw)} kDa mark.`,
        this.undoOf(answer, "remove_calibration_point", name),
      );
      this.keepFocus();
    }
  }

  // Clear marks: every mark of the group, both ladders (the ladder chosen stays).
  async clear() {
    const image = this.image;
    if (!image || !this.group.points.length) {
      return;
    }
    let answer = null;
    try {
      answer = await this.handlers.edit("DELETE", `/api/images/${image.id}/calibration`);
    } catch {
      return;
    }
    if (answer) {
      const button = this.handlers.status(
        "Cleared the ladder marks.",
        this.undoOf(answer, "clear_calibration", "Undo clearing the ladder marks"),
      );
      if (button && this.focusLost()) {
        button.focus();
      }
    }
  }

  // --- The ladder and the marker link ---

  async setLadder(body) {
    let answer = null;
    try {
      answer = await this.handlers.edit(
        "PUT",
        `/api/membranes/${this.membrane.id}/calibration/ladder`,
        body,
        body.kda === undefined
          ? {}
          : {
              refused: (error) => {
                $("cal-custom-error").textContent = sentence(error.message);
              },
            },
      );
    } catch {
      this.ladderShown = null;
      if (this.membrane) {
        this.renderLadder(); // the ladder stored, back in the select
      }
      return false;
    }
    if (!answer) {
      return false;
    }
    const preset = (this.presets || []).find((each) => each.key === body.ladder);
    const name = preset ? `${shortProduct(preset)} · ${preset.system}` : body.ladder;
    this.handlers.status(
      `Ladder: ${name}.` +
        (this.ladderKda().length
          ? " Find ladder, then click the ladder's lane on the image."
          : " Mark its bands by clicking them (Mark ladder bands)."),
    );
    return true;
  }

  openCustom() {
    const form = $("cal-custom");
    form.hidden = false;
    this.customFor = this.customScope(); // closed once another membrane is shown (render)
    $("cal-custom-error").textContent = "";
    const ladder = this.membrane.ladder;
    if (ladder !== null && !ladder.includes("/")) {
      $("cal-custom-name").value = ladder;
      $("cal-custom-kda").value = this.ladderKda().map(kdaText).join(", ");
    }
    $("cal-custom-name").focus();
  }

  closeCustom(refocus) {
    const form = $("cal-custom");
    form.reset();
    form.hidden = true;
    this.customFor = null;
    $("cal-custom-error").textContent = "";
    if (refocus && $("cal-ladder").getClientRects().length) {
      $("cal-ladder").focus();
    }
  }

  async submitCustom() {
    if (this.customFor === null || this.customFor !== this.customScope()) {
      this.closeCustom(true); // typed for a membrane no longer shown: it sets no ladder
      return;
    }
    const name = $("cal-custom-name").value.trim();
    const typed = $("cal-custom-kda").value.trim();
    const words = typed ? typed.split(/[\s,;]+/).filter(Boolean) : [];
    const kda = words.map(Number);
    if (kda.some((mw) => !Number.isFinite(mw) || mw <= 0)) {
      $("cal-custom-error").textContent =
        "The MWs are positive numbers of kDa, top to bottom, separated by commas.";
      return;
    }
    const body = kda.length ? { ladder: name, kda } : { ladder: name, kda: [] };
    if (await this.setLadder(body)) {
      this.closeCustom(true);
    }
  }

  // Link a chemiluminescence image to its marker image (null: unlink): one
  // register group, one calibration, one step.
  async link(imageId, markerId) {
    let answer = null;
    try {
      answer = await this.handlers.edit(
        "PUT",
        `/api/images/${imageId}/marker`,
        { marker_image_id: markerId },
        { refused: (error) => this.handlers.report(error, refusalText(error, "Not linked")) },
      );
    } catch {
      this.markersShown = null;
      this.handlers.changed(); // the link stored, back in the select
      return;
    }
    if (!answer) {
      return;
    }
    const named = (id) => {
      const image = answer.project.images.find((each) => each.id === id);
      return isolate(image ? image.original_name : id);
    };
    const text = markerId
      ? `Linked ${named(imageId)} to its marker image ${named(markerId)}: they share one` +
        " calibration."
      : `Unlinked ${named(imageId)} from its marker image.`;
    const name = markerId ? "Undo linking the marker image" : "Undo unlinking";
    this.handlers.status(text, this.undoOf(answer, "set_marker_image", name));
  }

  // --- The ladder changed elsewhere ---

  // What finds the ladder now, for the status line: Find ladder again; or,
  // with no ladder chosen, or one that lists no MWs (Find ladder needs them),
  // what can be done instead.
  findAgain() {
    if (!this.membrane || this.membrane.ladder === null) {
      return "no ladder is chosen now; choose it, then Find ladder";
    }
    return this.ladderKda().length
      ? "Find ladder again"
      : "the ladder lists no MWs now, and Find ladder needs them; mark its bands by clicking" +
          " them (Mark ladder bands)";
  }

  // Once the project is read again (another tab changed what this page
  // shows), while it still shows the opening `openId`: `words()`, worded from
  // what it shows then, on the status line. With `refocus` (the keyboard was
  // in this section: Apply), the keyboard, if its control is disabled or
  // hidden now (Find ladder, with no MWs listed now), goes on (keepFocus).
  // Nothing about another project is said.
  async sayOnceRead(openId, words, refocus = false) {
    await this.handlers.reread();
    if (!this.project || this.project.open_id !== openId) {
      return;
    }
    this.handlers.status(words());
    if (refocus && this.membrane) {
      this.keepFocus();
    }
  }

  // A mark's MW, or a relabel's, chosen from the ladder's list `kda` on
  // `image`, refused as calibration_changed: the membrane's ladder is no
  // longer that list. Nothing was stored. Said once the project is read
  // again, as a change made elsewhere (another tab) unless this page showed
  // it already (an Undo of its own); then, while `image` is shown, what can
  // be done with the ladder shown now: `next(listed)`, `listed` true while it
  // lists MWs, false while it lists none, null with no ladder chosen.
  ladderChanged(verb, kda, image, openId, next) {
    const before = this.project.membranes.find((each) => each.id === image.membrane_id);
    const elsewhere = Boolean(before) && sameLadder(kda, before.ladder_kda);
    const how = elsewhere ? "was changed (in another tab, say)" : "changed meanwhile";
    const said =
      `${verb}: the membrane's ladder ${how}, and the MW was chosen from the ladder before.` +
      " Nothing was stored";
    this.sayOnceRead(openId, () => {
      if (!this.shows(image, openId) || !this.membrane) {
        return `${said}.`;
      }
      const listed = this.membrane.ladder === null ? null : this.ladderKda().length > 0;
      return `${said}; ${next(listed)}.`;
    });
  }

  // --- The popup ---

  // A popup beside `where` (client coordinates) on the image: a button per
  // ladder MW (`choices`: {mw, current, disabled: why, or ""}; the reference
  // bands with their colour, `proposed` first to the keyboard), a field for
  // another MW (`other`, its button `otherLabel`), and one more action
  // (`extra`: {label, run}).
  // `choose(mw, listed)` takes the MW chosen, `listed` whether it was one of
  // `choices` (not typed); the keyboard goes back to `back` when it closes.
  openMenu({
    title,
    choices,
    proposed = null,
    other = false,
    otherLabel = "Mark",
    choose,
    extra = null,
    where,
    back,
  }) {
    const menu = $("cal-menu");
    this.closeMenu(false);
    $("cal-menu-title").textContent = title;
    const colors = this.referenceColors();
    const buttons = [];
    const list = $("cal-menu-choices");
    list.replaceChildren();
    for (const choice of choices) {
      const button = document.createElement("button");
      button.type = "button";
      const reference = [...colors.keys()].find((key) => sameMw(Number(key), choice.mw));
      if (reference !== undefined) {
        button.append(swatch(colors.get(reference)));
      }
      button.append(`${kdaText(choice.mw)}`);
      button.setAttribute("aria-label", `${kdaText(choice.mw)} kDa`);
      button.setAttribute("aria-pressed", String(choice.current));
      if (proposed !== null && sameMw(choice.mw, proposed)) {
        button.classList.add("proposed");
      }
      button.disabled = Boolean(choice.disabled);
      button.title = choice.disabled || "";
      button.addEventListener("click", () => {
        this.closeMenu(true);
        choose(choice.mw, true);
      });
      list.append(button);
      buttons.push([choice, button]);
    }
    const form = $("cal-menu-other");
    form.hidden = !other;
    form.querySelector('button[type="submit"]').textContent = otherLabel;
    form.onsubmit = (event) => {
      event.preventDefault();
      const mw = Number($("cal-menu-mw").value);
      if (Number.isFinite(mw) && mw > 0) {
        this.closeMenu(true);
        choose(mw, false);
      }
    };
    $("cal-menu-mw").value = "";
    const more = $("cal-menu-extra");
    more.hidden = !extra;
    more.textContent = extra ? extra.label : "";
    more.onclick = extra
      ? () => {
          this.closeMenu(true);
          extra.run();
        }
      : null;
    menu.hidden = false;
    // Beside the point, kept inside the stage so every button can be reached.
    const stage = menu.parentElement.getBoundingClientRect();
    const left = Math.min(where.clientX - stage.left + 14, stage.width - menu.offsetWidth - 8);
    const top = Math.min(where.clientY - stage.top - 16, stage.height - menu.offsetHeight - 8);
    menu.style.left = `${Math.max(8, left)}px`;
    menu.style.top = `${Math.max(8, top)}px`;
    this.menu = { back, scope: this.menuScope() };
    const offered = ([choice, button]) =>
      !button.disabled && proposed !== null && sameMw(choice.mw, proposed);
    const first =
      buttons.find(offered) ||
      buttons.find(([choice, button]) => !button.disabled && choice.current) ||
      buttons.find(([, button]) => !button.disabled);
    (first ? first[1] : other ? $("cal-menu-mw") : $("cal-menu-cancel")).focus();
  }

  closeMenu(refocus) {
    const open = this.menu;
    this.menu = null;
    $("cal-menu").hidden = true;
    if (refocus && open && open.back && open.back.isConnected) {
      open.back.focus();
    }
  }

  // --- Helpers ---

  // The status line's Undo of the change an answer logged as `action`, while
  // it is the last one; null otherwise.
  undoOf(answer, action, name) {
    const step = answer.project.history.undo;
    return step && step.action === action
      ? { label: "Undo", name, seq: step.seq, run: () => this.handlers.undo(step.seq) }
      : null;
  }

  // Whether the keyboard focus was lost: on nothing, or on a control removed,
  // disabled or hidden. A ruler tick's button, or Drop, keeps the focus a
  // moment once the ruler's panel is hidden (Enter or Esc on it), until the
  // browser takes it away to nothing.
  focusLost() {
    const active = document.activeElement;
    return (
      !active ||
      active === document.body ||
      !active.isConnected ||
      Boolean(active.disabled) ||
      !active.getClientRects().length
    );
  }

  // The ruler's panel closed: the keyboard goes on to Find ladder (or, with no
  // MWs listed to find, to Mark ladder bands; with no ladder chosen, to the
  // ladder's select).
  keepFocus() {
    if (this.focusLost()) {
      const next = ["cal-find", "cal-mark", "cal-ladder"]
        .map((id) => $(id))
        .find((button) => !button.disabled && button.getClientRects().length);
      if (next) {
        next.focus();
      }
    }
  }
}
