// SPDX-License-Identifier: Apache-2.0
// The chart cards of the results dock: one card per series (a target over one
// of its loading controls), side by side, scrolled sideways. The server draws
// every chart (ADR 0002), and each answer names a series' drawing by a URL of
// its content (chart_url), so a chart that did not change keeps its URL. A
// drawing is fetched with the token, as a blob, and shown as an image from the
// blob's object URL, never as inline SVG; the object URL is made once per
// drawing, kept while a card shows or awaits it, and revoked after (a fetch
// no card awaits any more is cancelled). An image stays in place, dimmed,
// until the one that replaces it has loaded. With two result sets (excluded
// lanes hold values, #71) each card shows both charts side by side, each
// labelled with its set. Under each chart: its legend (the chart's statement:
// what the marks are, its test or why there is none, the conditions it leaves
// out), the set's notices about the series, and its values in a table. A
// series with no chart says why. In the dock a chart is as large as the dock
// allows, often too small to read the text drawn in it: a card's header counts
// its warnings, and Enlarge (or a click on a chart) shows the card in a
// dialog, its charts at their own size where the window allows, with their
// whole captions. The cards edit nothing.
import { $, counted, ratioText, sentence } from "/static/dom.js";

// The notices that say why a series has no chart.
const NO_CHART = new Set([
  "reference_all_excluded",
  "reference_unusable",
  "no_plotted_values",
  "no_values",
  "loading_control_ambiguous",
]);

// A bar's error as written, or null: a condition with no bar has none, and
// one sample has no SD (nor SEM), though the core stores 0 for it.
function errorText(bar) {
  return bar.mean !== null && bar.n > 1 ? ratioText(bar.error) : null;
}

// Why a condition draws no bar: replicates not detected, or no value at all.
function noBarText(chart, bar, i) {
  const undetected = bar.not_detected_lanes.length;
  if (!undetected) {
    return "no value";
  }
  const cover = chart.coverage[i];
  const replicates = cover ? cover.replicates : bar.n + undetected;
  return `${undetected} of ${replicates} not detected`;
}

// What the image shows, in words: each bar's mean ± error and n, or why a
// condition has none, then the chart's legend, each line of it a sentence.
function altText(title, label, chart) {
  const bars = chart.bars.map((bar, i) => {
    if (bar.mean === null) {
      return `${bar.label} no bar (${noBarText(chart, bar, i)})`;
    }
    const error = errorText(bar);
    const n = `n=${bar.n}${error === null ? `, no ${chart.error_type}` : ""}`;
    return `${bar.label} ${ratioText(bar.mean)}${error === null ? "" : ` ± ${error}`} (${n})`;
  });
  const set = label ? ` (${label})` : "";
  const legend = chart.statement.map(sentence).join(" ");
  return `${title}${set}, ${chart.y_label}: ${bars.join(", ")}. ${legend}`;
}

// A bar's replicates in lane order, as [lane, text]: each value, and "n.d."
// for each replicate not detected.
function replicateCells(bar) {
  const rows = bar.points.map((value, j) => [bar.lane_indices[j], ratioText(value)]);
  for (const lane of bar.not_detected_lanes) {
    rows.push([lane, "n.d."]);
  }
  return rows.sort((a, b) => a[0] - b[0]);
}

function cell(tag, text, className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) {
    element.className = className;
  }
  return element;
}

// The chart's values: per bar its n, mean, error, samples and their lanes
// (1-based); a replicate not detected is "n.d.", and a condition with no bar
// has no mean.
function valuesTable(chart) {
  const table = document.createElement("table");
  table.className = "chart-table";
  const head = document.createElement("tr");
  const lanes = cell("th", "Lanes");
  lanes.title = "Each sample's lane (the first, for technical repeats averaged)";
  head.append(
    cell("th", "Condition", "label"),
    cell("th", "n"),
    cell("th", "Mean"),
    cell("th", chart.error_type),
    cell("th", "Samples", "label"),
    lanes,
  );
  for (const th of head.children) {
    th.scope = "col";
  }
  const thead = document.createElement("thead");
  thead.append(head);
  const body = document.createElement("tbody");
  chart.bars.forEach((bar, i) => {
    const row = document.createElement("tr");
    const name = cell("th", bar.label, "label");
    name.scope = "row";
    const error = errorText(bar);
    const spread = cell("td", error === null ? "—" : error);
    const mean = cell("td", bar.mean === null ? "—" : ratioText(bar.mean));
    if (bar.mean === null) {
      mean.title = `No bar: ${noBarText(chart, bar, i)}`;
      spread.title = mean.title;
    } else if (error === null) {
      spread.title = `One sample: no ${chart.error_type}`;
    }
    const undetected = bar.not_detected_lanes.length;
    const cover = chart.coverage[i];
    const n = undetected
      ? `${cover ? cover.replicates : bar.n + undetected} (${undetected} n.d.)`
      : String(bar.n);
    const cells = replicateCells(bar);
    row.append(
      name,
      cell("td", n),
      mean,
      spread,
      cell("td", cells.map(([, text]) => text).join(", "), "label samples"),
      cell("td", cells.map(([lane]) => (lane === undefined ? "" : lane + 1)).join(", ")),
    );
    body.append(row);
  });
  table.append(thead, body);
  return table;
}

const noticeKey = (notice) => `${notice.code}\n${notice.protein_ids.join("\n")}`;

// The notices about one set only: its excluded reference lanes, and its charts'
// tests (each set tests its own charts, and keeps these notices even when the
// other set has the same).
const OWN_SET = new Set([
  "reference_all_excluded",
  "conditions_not_tested",
  "test_not_applicable",
  "log_scale_unavailable",
  "rank_test_cannot_reach_alpha",
]);

// The notices that apply to each set, by set id. The all-lanes set carries
// only those the applied set lacks: of the applied set's, those apply to it as
// well, but for one it has its own of the same kind about the same proteins (a
// clipped notice over more lanes), and those about the applied set only.
function setNotices(sets) {
  const [applied, every] = sets;
  const notices = new Map([[applied.id, applied.notices]]);
  if (every) {
    const own = new Set(every.notices.map(noticeKey));
    const shared = applied.notices.filter((n) => !OWN_SET.has(n.code) && !own.has(noticeKey(n)));
    notices.set(every.id, [...every.notices, ...shared]);
  }
  return notices;
}

function aboutSeries(notice, series) {
  return (
    notice.protein_ids.includes(series.target_id) || notice.protein_ids.includes(series.loading_id)
  );
}

// Why a series has no chart: its own notice that says so, else the set's.
function noChartReason(notices, series) {
  const reason =
    notices.find((n) => NO_CHART.has(n.code) && aboutSeries(n, series)) ||
    notices.find((n) => NO_CHART.has(n.code) && !n.protein_ids.length);
  return reason || null;
}

// The notices shown under a series' chart: those about its proteins, and the
// set's own about the conditions it draws (technical repeats, look-alike
// labels), warnings first. A drawn chart shows none of the reasons for no chart.
function captionNotices(notices, series, chart, reason) {
  const bars = new Set(chart ? chart.bars.map((bar) => bar.label) : []);
  const shown = notices.filter((n) => {
    if (n === reason || (chart && NO_CHART.has(n.code))) {
      return false;
    }
    if (n.protein_ids.length) {
      return aboutSeries(n, series);
    }
    return Boolean(chart) && n.conditions.some((c) => bars.has(c));
  });
  return [
    ...shown.filter((n) => n.level === "warning"),
    ...shown.filter((n) => n.level !== "warning"),
  ];
}

// The cards the results show: one per series of the applied set, in series
// order, with the same series of each set; then one per target with no series,
// as it has no loading control chosen among two or more.
function cardsOf(results) {
  const [applied] = results.sets;
  const cards = applied.series.map((series) => ({
    key: `${series.target_id}\n${series.loading_id}`,
    title: `${series.target} / ${series.loading}`,
    series: results.sets.map((set) =>
      set.series.find(
        (s) => s.target_id === series.target_id && s.loading_id === series.loading_id,
      ),
    ),
    notice: null,
  }));
  for (const notice of applied.notices) {
    if (notice.code !== "loading_control_ambiguous") {
      continue;
    }
    for (const id of notice.protein_ids) {
      const column = results.proteins.find((p) => p.protein_id === id);
      cards.push({ key: `${id}\n`, title: column ? column.name : id, series: [], notice });
    }
  }
  return cards;
}

// What the dock shows with no chart: the next step.
function emptyText(results) {
  const codes = new Set(results.sets[0].notices.map((n) => n.code));
  if (codes.has("no_target") && codes.has("no_loading_control")) {
    return (
      "No charts yet: add a target and a loading control (role: loading control)" +
      " to normalize."
    );
  }
  if (codes.has("no_loading_control")) {
    return "No charts yet: add a loading control (role: loading control) to normalize.";
  }
  if (codes.has("no_target")) {
    return "No charts yet: add a target.";
  }
  if (codes.has("no_lanes")) {
    return "No charts yet: declare the lanes under Lanes & values.";
  }
  return "No charts yet.";
}

// Put `elements` in `container` in this order, moving only those out of
// place: an element left in place keeps the keyboard focus.
function inOrder(container, elements) {
  elements.forEach((element, i) => {
    if (container.children[i] !== element) {
      container.insertBefore(element, container.children[i] || null);
    }
  });
}

let cardCount = 0; // gives each card's heading its own id

export class ChartCards {
  // handlers: fetch(path, signal) gives a Promise of the drawing at `path` as
  // a Blob, fetched with the token, and rejects with the server's refusal (its
  // code), or once `signal` aborts; reread() gives a Promise that settles once
  // the project has been read again (GET /api/project) and its answer shown,
  // which registers its charts anew.
  constructor(handlers) {
    this.handlers = handlers;
    this.results = null; // the results the cards show
    this.cards = new Map(); // card key -> {key, item, title, flag, list, slots: Map set id -> slot}
    this.zoom = null; // the card the dialog shows: as a card, with no item and no flag
    this.images = new Map(); // chart URL -> {objectUrl, failed, promise, fetching}
    this.rereading = null; // the read of the project that missing drawings await
    // Each chart is as tall as the strip of cards allows (app.css). Its border
    // box, not its content box: a scroll bar coming or going changes nothing.
    const strip = $("chart-cards");
    new ResizeObserver(() => {
      strip.style.setProperty("--strip-height", `${strip.offsetHeight}px`);
    }).observe(strip, { box: "border-box" });
    // Beside the table the charts take at least the widest card (app.css).
    this.sizes = new ResizeObserver(() => this.measure());
    $("chart-dialog-close").addEventListener("click", () => this.closeZoom());
    $("chart-dialog").addEventListener("close", () => this.closeZoom());
  }

  // Cancel every fetch, revoke every object URL and drop the cards: another
  // project is shown now.
  forget() {
    this.closeZoom();
    for (const entry of this.images.values()) {
      this.discard(entry);
    }
    this.images.clear();
    for (const card of this.cards.values()) {
      this.drop(card);
    }
    this.cards.clear();
    this.results = null;
    this.measure();
    $("charts-empty").hidden = true;
  }

  // --- Drawing ---

  render(results) {
    if (results === this.results) {
      return; // drawn already (a render for a box chosen, say)
    }
    this.results = results;
    const notices = setNotices(results.sets);
    const wanted = cardsOf(results);
    const keys = new Set(wanted.map((c) => c.key));
    for (const [key, card] of this.cards) {
      if (!keys.has(key)) {
        this.drop(card);
        this.cards.delete(key);
      }
    }
    const items = wanted.map((each) => {
      let card = this.cards.get(each.key);
      if (!card) {
        card = this.makeCard(each.key);
        this.cards.set(each.key, card);
      }
      this.fillCard(card, each, results, notices);
      return card.item;
    });
    inOrder($("chart-cards"), items);
    const empty = $("charts-empty");
    empty.hidden = items.length > 0;
    if (!items.length) {
      empty.textContent = emptyText(results);
    }
    if (this.zoom) {
      const each = wanted.find((c) => c.key === this.zoom.key);
      if (each) {
        this.fillCard(this.zoom, each, results, notices);
      } else {
        this.closeZoom(); // its series is gone
      }
    }
    this.measure();
    this.sweep();
  }

  // A card: its heading (the series, its warnings counted, Enlarge), then a
  // place per result set.
  makeCard(key) {
    const item = document.createElement("li");
    item.className = "chart-card";
    const head = document.createElement("div");
    head.className = "chart-head";
    const title = document.createElement("h4");
    title.className = "chart-title";
    title.id = `chart-title-${++cardCount}`;
    const flag = cell("span", "", "chart-flag");
    flag.hidden = true;
    const enlarge = cell("button", "Enlarge", "chart-enlarge");
    enlarge.type = "button";
    enlarge.title = "Show the charts large, with their whole captions";
    enlarge.setAttribute("aria-describedby", title.id);
    enlarge.addEventListener("click", () => this.enlarge(key));
    head.append(title, flag, enlarge);
    const list = document.createElement("div");
    list.className = "chart-sets";
    item.append(head, list);
    this.sizes.observe(item);
    return { key, item, title, flag, enlarge, list, slots: new Map() };
  }

  drop(card) {
    if (card.item) {
      this.sizes.unobserve(card.item);
      card.item.remove();
    }
    for (const slot of card.slots.values()) {
      slot.live = false;
    }
  }

  fillCard(card, each, results, notices) {
    card.title.textContent = each.title;
    const sets = each.notice ? [results.sets[0]] : results.sets;
    const ids = new Set(sets.map((set) => set.id));
    for (const [id, slot] of card.slots) {
      if (!ids.has(id)) {
        slot.figure.remove();
        slot.live = false;
        card.slots.delete(id);
      }
    }
    sets.forEach((set, i) => {
      let slot = card.slots.get(set.id);
      if (!slot) {
        slot = this.makeSlot(card);
        card.slots.set(set.id, slot);
      }
      const labelled = !each.notice && sets.length > 1;
      slot.label.hidden = !labelled;
      slot.label.textContent = labelled ? set.label || "" : "";
      if (each.notice) {
        this.fillMissing(slot, sentence(each.notice.message), []);
        slot.placeholder.append(" Choose its loading control under Proteins on this image.");
      } else {
        this.fillSeries(slot, each, set, each.series[i], notices.get(set.id));
      }
    });
    inOrder(card.list, sets.map((set) => card.slots.get(set.id).figure));
    card.list.classList.toggle("paired", sets.length > 1);
    if (card.flag) {
      // The caption under a chart in the dock is often cut: its warnings are
      // counted where they are always seen, and read in full under Enlarge.
      const warnings = new Set(sets.flatMap((set) => card.slots.get(set.id).warnings));
      card.flag.textContent = counted(warnings.size, "warning", "warnings");
      card.flag.title = [...warnings].join("\n");
      card.flag.hidden = !warnings.size;
    }
  }

  // A chart's place: its set's label, the image (or the reason there is none),
  // and its caption with the values table. A click on a chart in the dock
  // enlarges its card (Enlarge, for the keyboard).
  makeSlot(card) {
    const figure = document.createElement("figure");
    figure.className = "chart-set";
    const label = cell("p", "", "chart-set-label");
    label.id = `${card.title.id}-${card.slots.size + 1}`;
    figure.setAttribute("aria-labelledby", `${card.title.id} ${label.id}`);
    const frame = document.createElement("div");
    frame.className = "chart-frame";
    const img = document.createElement("img");
    img.className = "chart-image";
    img.hidden = true;
    if (card.item) {
      img.addEventListener("click", () => this.enlarge(card.key));
    }
    const placeholder = cell("p", "", "chart-placeholder");
    frame.append(img, placeholder);
    const caption = document.createElement("figcaption");
    caption.className = "chart-caption";
    const test = document.createElement("div");
    test.className = "chart-test";
    const bars = cell("p", "", "chart-bars");
    const notes = document.createElement("ul");
    notes.className = "chart-notices";
    const details = document.createElement("details");
    details.className = "chart-values";
    const summary = cell("summary", "Values");
    const table = document.createElement("div");
    table.className = "chart-table-scroll";
    details.append(summary, table);
    caption.append(bars, test, notes, details); // the legend: its marks line first
    figure.append(label, frame, caption);
    return {
      figure,
      label,
      frame,
      img,
      placeholder,
      test,
      bars,
      notes,
      details,
      table,
      warnings: [], // the warnings in its caption, as written
      shown: null, // the chart URL whose drawing the image shows
      wanted: null, // the chart URL the slot shows once it has loaded, or null
      alt: "",
      live: true,
    };
  }

  fillSeries(slot, each, set, series, notices) {
    const chart = series ? series.chart : null;
    const url = series ? series.chart_url : null;
    if (!chart || !url) {
      const reason = series ? noChartReason(notices, series) : null;
      const text = reason ? sentence(reason.message) : "No chart for this series.";
      this.fillMissing(slot, text, series ? captionNotices(notices, series, null, reason) : []);
      return;
    }
    // The legend, in its order: what the marks are, then the test (or why
    // there is none) and the conditions it leaves out.
    const [marks = "", ...statistics] = chart.statement;
    slot.bars.textContent = marks ? sentence(marks) : "";
    slot.bars.hidden = !marks;
    slot.test.replaceChildren(...statistics.map((line) => cell("p", sentence(line))));
    slot.test.hidden = !statistics.length;
    this.fillNotices(slot, captionNotices(notices, series, chart, null));
    slot.table.replaceChildren(valuesTable(chart));
    slot.details.hidden = false;
    this.showImage(slot, url, altText(each.title, set.label, chart));
  }

  // No chart in this place: the reason, in a dashed frame, instead.
  fillMissing(slot, text, notices) {
    slot.wanted = null;
    slot.shown = null;
    slot.img.hidden = true;
    slot.img.removeAttribute("src");
    slot.img.alt = "";
    slot.placeholder.textContent = text;
    slot.placeholder.className = "chart-placeholder";
    slot.placeholder.hidden = false;
    slot.frame.classList.remove("stale");
    slot.test.hidden = true;
    slot.bars.hidden = true;
    slot.details.hidden = true;
    this.fillNotices(slot, notices);
  }

  fillNotices(slot, notices) {
    const lines = notices.map((notice) => [sentence(notice.message), notice.level]);
    slot.notes.replaceChildren(...lines.map(([text, level]) => cell("li", text, level)));
    slot.notes.hidden = !lines.length;
    slot.warnings = lines.filter(([, level]) => level === "warning").map(([text]) => text);
  }

  // --- The enlarged card ---

  // Show the card of `key` in the dialog: its charts at their own size, where
  // the text drawn in them reads, or as near it as the window allows, with
  // their whole captions. It follows the answers shown while it is open, and
  // closes if its series goes.
  enlarge(key) {
    if (!this.results || this.zoom) {
      return;
    }
    const each = cardsOf(this.results).find((c) => c.key === key);
    if (!each) {
      return;
    }
    const list = document.createElement("div");
    list.className = "chart-sets";
    $("chart-dialog-body").replaceChildren(list);
    const title = $("chart-dialog-title");
    this.zoom = { key, item: null, title, flag: null, list, slots: new Map() };
    this.fillCard(this.zoom, each, this.results, setNotices(this.results.sets));
    $("chart-dialog").showModal();
  }

  // Close the dialog (if open) and let go of its card; the keyboard focus goes
  // back to the card's Enlarge.
  closeZoom() {
    if (!this.zoom) {
      return;
    }
    const card = this.cards.get(this.zoom.key);
    this.drop(this.zoom);
    this.zoom = null;
    const dialog = $("chart-dialog");
    if (dialog.open) {
      dialog.close();
    }
    $("chart-dialog-body").replaceChildren();
    if (card && card.item.isConnected) {
      card.enlarge.focus();
    }
    this.sweep();
  }

  // --- Images ---

  // Show the drawing at `url` in `slot` once it has loaded; the image shown
  // until then stays, dimmed, and a place with none says the chart is being
  // drawn (over the reason there was none, or that it could not be loaded).
  // `alt` describes it.
  showImage(slot, url, alt) {
    slot.wanted = url;
    slot.alt = alt;
    if (slot.shown === url) {
      slot.img.alt = alt;
      slot.frame.classList.remove("stale");
      return;
    }
    const entry = this.image(url);
    if (entry.objectUrl) {
      slot.img.src = entry.objectUrl; // decoded already (load()): shown at once
      slot.img.alt = alt;
      slot.img.hidden = false;
      slot.placeholder.hidden = true;
      slot.shown = url;
      slot.frame.classList.remove("stale");
      return;
    }
    if (entry.failed) {
      slot.shown = null;
      slot.img.hidden = true;
      slot.img.removeAttribute("src");
      const retry = document.createElement("button");
      retry.type = "button";
      retry.textContent = "Try again";
      retry.addEventListener("click", () => {
        if (this.images.get(url) === entry) {
          this.images.delete(url); // fetched anew
        }
        if (slot.live && slot.wanted === url) {
          this.showImage(slot, url, slot.alt);
        }
      });
      slot.placeholder.replaceChildren("The chart could not be loaded.", retry);
      slot.placeholder.className = "chart-placeholder";
      slot.placeholder.hidden = false;
      slot.frame.classList.remove("stale");
      return;
    }
    if (slot.shown === null) {
      slot.placeholder.textContent = "Drawing the chart…";
      slot.placeholder.className = "chart-placeholder loading";
      slot.placeholder.hidden = false;
      slot.frame.classList.remove("stale");
    } else {
      slot.frame.classList.add("stale");
    }
    entry.promise.then(() => {
      if (slot.live && slot.wanted === url) {
        this.showImage(slot, url, slot.alt);
        this.sweep();
      }
    });
  }

  // The drawing at `url`: fetched once, however many places wait for it.
  image(url) {
    let entry = this.images.get(url);
    if (!entry) {
      entry = { objectUrl: null, failed: false, promise: null, fetching: new AbortController() };
      this.images.set(url, entry);
      entry.promise = this.load(url, entry);
    }
    return entry;
  }

  // Fetch a drawing and make its object URL; `entry.failed` if it could not
  // be loaded. Never rejects.
  async load(url, entry) {
    let objectUrl = null;
    try {
      const blob = await this.fetchDrawing(url, entry.fetching.signal);
      if (this.images.get(url) !== entry) {
        return; // no place wants it now
      }
      objectUrl = URL.createObjectURL(blob);
      // Decoded before it is shown, so it replaces the image shown at once;
      // and a drawing the browser cannot show is found here.
      const probe = new Image();
      probe.src = objectUrl;
      await probe.decode();
      if (this.images.get(url) !== entry) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      entry.objectUrl = objectUrl;
    } catch (error) {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
      entry.failed = true;
    }
  }

  // The drawing at `url`. One the server no longer keeps (404 unknown_id: it
  // keeps the last few, and none of a project opened before) is asked for once
  // more after the project is read again, which registers its charts anew;
  // not if no place wants it by then (`signal` aborted: discard()).
  async fetchDrawing(url, signal) {
    try {
      return await this.handlers.fetch(url, signal);
    } catch (error) {
      if (error.code !== "unknown_id" || signal.aborted) {
        throw error;
      }
    }
    await this.reread();
    signal.throwIfAborted();
    return this.handlers.fetch(url, signal);
  }

  // One read of the project for all the drawings found missing meanwhile.
  reread() {
    if (!this.rereading) {
      const done = () => {
        this.rereading = null;
      };
      this.rereading = this.handlers.reread().then(done, done);
    }
    return this.rereading;
  }

  // Revoke the object URLs of the drawings no place shows or awaits, and
  // cancel their fetches.
  sweep() {
    const used = new Set();
    const cards = this.zoom ? [...this.cards.values(), this.zoom] : this.cards.values();
    for (const card of cards) {
      for (const slot of card.slots.values()) {
        used.add(slot.shown);
        used.add(slot.wanted);
      }
    }
    for (const [url, entry] of this.images) {
      if (!used.has(url)) {
        this.images.delete(url);
        this.discard(entry);
      }
    }
  }

  // Let go of a drawing no place wants: its fetch cancelled (so it no longer
  // counts as awaited, dock.js, nor holds a connection), its object URL revoked.
  discard(entry) {
    entry.fetching.abort();
    if (entry.objectUrl) {
      URL.revokeObjectURL(entry.objectUrl);
    }
  }

  // The widest card's width, the least the charts take beside the table
  // (app.css): one card is seen whole, its two sets side by side.
  measure() {
    let widest = 0;
    for (const card of this.cards.values()) {
      widest = Math.max(widest, card.item.offsetWidth);
    }
    $("dock-body").style.setProperty("--cards-width", `${widest}px`);
  }
}
