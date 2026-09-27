// SPDX-License-Identifier: Apache-2.0
// The "Lanes & values" table: one row per lane with its condition, sample and
// include flag, which the user edits, and each protein's net and each series'
// normalized value or fold change, read from the results of the same answer.
// Its toolbar chooses the reference condition, declares the lanes from a list
// of conditions, and adds or removes the last lane. It stores nothing itself:
// every change sends the whole table (PUT /api/lanes) through the app's edit
// queue, and the table is drawn again from the state the server answers.
import { $, counted, inWords, netText, ratioText, sentence, span, swatch } from "/static/dom.js";
import { colorOf } from "/static/proteins.js";

const FIELDS = ["condition", "sample", "included"];
const FOLD_CHANGE = "fold_change";

// Typed text as the server stores it: trimmed, its inner white space one space.
function tidy(text) {
  return (text || "").trim().split(/\s+/).join(" ");
}

// The key the server compares conditions by (text_key in core/names.py):
// format characters dropped, NFKC, white space collapsed, so look-alike
// spellings (the micro sign µ and the Greek μ) share it.
function textKey(text) {
  return tidy(text.replace(/\p{Cf}/gu, "").normalize("NFKC"));
}

// The lane table as the server stored it, as rows to edit and send.
function tableOf(project) {
  return project.lanes.map((lane) => ({
    condition: lane.condition,
    sample: lane.sample,
    included: lane.included,
  }));
}

function sameTable(a, b) {
  return (
    a.length === b.length &&
    a.every(
      (lane, i) =>
        tidy(lane.condition) === tidy(b[i].condition) &&
        tidy(lane.sample) === tidy(b[i].sample) &&
        lane.included === b[i].included,
    )
  );
}

// The reference condition's new name when the edit renames it: every lane of
// it gets one new name, and no lane keeps the old one (in any look-alike
// spelling). Otherwise null, and the edit leaves the reference out: the server
// keeps it while a lane still has it, with the lanes that have it after the
// edit (labels swapped between lanes keep it with its label), and clears it,
// and says so, once none has.
function followedReference(reference, before, after) {
  if (reference === null) {
    return null;
  }
  const key = textKey(reference);
  if (after.some((lane) => textKey(lane.condition) === key)) {
    return null;
  }
  const lanes = [...before.keys()].filter((i) => before[i].condition === reference);
  if (!lanes.length || lanes.some((i) => i >= after.length)) {
    return null;
  }
  const names = new Set(lanes.map((i) => tidy(after[i].condition)));
  const [name] = names;
  return names.size === 1 && name ? name : null;
}

// Put a value in a control. A text field that has the focus keeps its text
// all selected if it was, and otherwise has the caret at the end (as setting
// its value leaves it).
function show(control, value) {
  if (control.type === "checkbox") {
    control.checked = value;
    return;
  }
  if (control.value === value) {
    return;
  }
  const all =
    document.activeElement === control &&
    control.value.length > 0 &&
    control.selectionStart === 0 &&
    control.selectionEnd === control.value.length;
  control.value = value;
  if (all) {
    control.select();
  }
}

function lanesText(indices) {
  const numbers = indices.map((i) => i + 1);
  return `${numbers.length === 1 ? "Lane" : "Lanes"} ${inWords(numbers.map(String))}`;
}

function hidden(text) {
  return span("sr-only", text);
}

function mark(text, className, label, title) {
  const element = span(className, text);
  element.setAttribute("aria-hidden", "true");
  const group = document.createElement("span");
  group.title = title;
  group.append(element, hidden(label));
  return group;
}

// A name in a column header: cut with an ellipsis past the column's width
// (app.css), and in full in its tooltip and for screen readers.
function clipped(text) {
  const part = span("clip", text);
  part.title = text;
  return part;
}

// A series' name: the target, then "÷ its loading control" on the same line
// or, when they do not fit together, on the next.
function seriesName(series) {
  const name = span("series-name", "");
  name.append(clipped(series.target), " ", clipped(`÷ ${series.loading}`));
  return name;
}

export class LaneTable {
  // handlers: queueEdit(task, {after}) runs `task(current)` in the app's edit
  // queue (ProteinPanel.queueEdit): after the edits before it, and never once
  // another project is asked for; pending() gives a Promise that settles once
  // the edits made outside that queue so far have their answers; send(method,
  // path, json) gives a Promise of the server's answer once the app has
  // applied it (null if it is about a project opened before the one shown),
  // and rejects with the server's refusal (code, message, ids); status(text)
  // shows a line to the user.
  constructor(handlers) {
    this.handlers = handlers;
    this.project = null;
    this.results = null;
    this.rows = []; // per lane: {row, head, tags, condition, sample, included}
    this.stored = new Map(); // control key -> its field's value in the table as last drawn
    this.synced = new Map(); // control key -> the value it last matched the table with (fill)
    this.sent = new Map(); // control key -> {value}: its last commit, until answered
    this.refused = null; // the boxes a lanes_in_use message shown names (ids), or null
    this.bind();
    // A cell scrolled to (by the keyboard) is not left under the sticky header,
    // nor under the lane number and condition, which stay while it scrolls sideways.
    const scroller = $("lane-table").parentElement;
    new ResizeObserver(() => {
      scroller.style.scrollPaddingTop = `${$("lane-head").offsetHeight}px`;
    }).observe($("lane-head"));
    this.stickyColumns = new ResizeObserver(() => {
      const lane = $("lt-lane");
      const condition = $("lt-condition");
      if (lane && condition) {
        $("lane-table").style.setProperty("--lane-column", `${lane.offsetWidth}px`);
        scroller.style.scrollPaddingLeft = `${lane.offsetWidth + condition.offsetWidth}px`;
      }
    });
  }

  // Forget what was typed and what a refusal said: another project is shown now.
  forgetTyped() {
    this.stored.clear();
    this.synced.clear();
    this.sent.clear();
    for (const row of this.rows) {
      row.row.remove();
    }
    this.rows = [];
    $("bulk-conditions").value = "";
    this.showError("");
  }

  bind() {
    $("reference").addEventListener("change", (event) => this.chooseReference(event.target.value));
    $("bulk-lanes").addEventListener("submit", (event) => {
      event.preventDefault();
      this.setLanes($("bulk-conditions").value);
    });
    $("add-lane").addEventListener("click", () => this.addLane());
    $("remove-lane").addEventListener("click", () => this.removeLane());
  }

  // --- Drawing ---

  // `answered`: control key -> the value an edit just answered sent from it;
  // the control shows the server's value (its spelling) if it still shows
  // that, and keeps what was typed since otherwise.
  render(project, results, answered = new Map()) {
    this.project = project;
    this.results = results;
    this.renderTools(project, answered.has("reference"));
    const columns = this.columns(results);
    this.renderHead(columns);
    this.renderRows(project, results, columns, answered);
    this.renderRefusal(project);
  }

  // Put the server's value in a control unless the user has changed it since
  // it last matched the table: the stored value it showed, or, once answered,
  // the value its commit sent (`answered`: control key -> the value an edit
  // just answered sent from it). A control that still shows that value takes
  // the new one, the focused field too (an undo, a redo or another edit's
  // answer changed its lane). A commit awaiting its answer, or text typed
  // since, stays until it is sent (the field is left) or put back (Escape).
  fill(key, control, value, answered) {
    this.stored.set(key, value);
    if (this.sent.has(key)) {
      return;
    }
    const shown = control.type === "checkbox" ? control.checked : control.value;
    const last = answered.has(key) ? answered.get(key) : this.synced.get(key);
    if (last === undefined || shown === last) {
      show(control, value);
      this.synced.set(key, value);
    } else if (answered.has(key)) {
      this.synced.set(key, last); // typed after its commit: kept
    }
  }

  // The value a control's lane field will have once the edits sent have their
  // answers: its last commit's, or else the stored one.
  expected(key) {
    return this.sent.has(key) ? this.sent.get(key).value : this.stored.get(key);
  }

  // Put back in a text field the value its lane field has, or will have once
  // its commit is answered; with none awaiting, it matches the table again.
  // Gives whether the text changed.
  restore(key, input) {
    const expected = this.expected(key);
    if (expected === undefined) {
      return false;
    }
    const changed = input.value !== expected;
    show(input, expected);
    if (!this.sent.has(key)) {
      this.synced.set(key, expected);
    }
    return changed;
  }

  // A lanes_in_use message goes once the boxes it names are deleted (on the
  // image, or by an undo), and names only those left meanwhile.
  renderRefusal(project) {
    if (!this.refused) {
      return;
    }
    const ids = new Set(project.proteins.flatMap((p) => p.bands.map((b) => b.id)));
    const left = this.refused.filter((id) => ids.has(id));
    if (!left.length) {
      this.showError("");
    } else if (left.length < this.refused.length) {
      this.showError(this.explain({ code: "lanes_in_use", ids: left, message: "" }), left);
    }
  }

  renderTools(project, force) {
    const select = $("reference");
    const conditions = [...new Set(project.lanes.map((lane) => lane.condition))];
    const stored = project.reference_condition || "";
    const shown = select.value;
    const typed = this.stored.has("reference") && shown !== this.stored.get("reference");
    select.replaceChildren(
      new Option("None", ""),
      ...conditions.map((condition) => new Option(condition, condition)),
    );
    const value = typed && !force ? shown : stored;
    select.value = conditions.includes(value) ? value : "";
    this.stored.set("reference", stored);
    select.disabled = !conditions.length;
    $("add-lane").disabled = !project.lanes.length;
    $("add-lane").title = project.lanes.length
      ? "Add a lane after the last, with its condition"
      : "Declare the lanes with Conditions first";
    $("remove-lane").disabled = !project.lanes.length;
  }

  // The value columns: each protein's net, then each series, split into the
  // applied set and the all-lanes set where a fold change differs between them
  // (a normalized value is the same in both, so it takes one column).
  columns(results) {
    const [applied, every] = results.sets;
    const series = applied.series.map((one) => {
      const other = every
        ? every.series.find((s) => s.target_id === one.target_id && s.loading_id === one.loading_id)
        : null;
      const split =
        Boolean(other) && (one.value_kind === FOLD_CHANGE || other.value_kind === FOLD_CHANGE);
      return { applied: one, all: split ? other : null };
    });
    return { nets: results.proteins, series };
  }

  header(id, parts, { cols = 1, rows = 1, scope = "col", className = "" } = {}) {
    const cell = document.createElement("th");
    cell.id = id;
    cell.scope = scope;
    if (cols > 1) {
      cell.colSpan = cols;
    }
    if (rows > 1) {
      cell.rowSpan = rows;
    }
    cell.className = className;
    cell.append(...parts);
    return cell;
  }

  // What a series' values are, for its header: a normalized value, or a fold
  // change with its baseline (the reference's mean normalized value); the
  // tooltip says it in full. `short`: under a header that says "fold change"
  // over this column and the one beside it; `narrow`: in two lines, beside a
  // column of normalized values.
  kindNote(series, reference, { short = false, narrow = false } = {}) {
    if (series.value_kind !== FOLD_CHANGE || series.baseline === null) {
      const note = span("column-note", "normalized");
      note.title = "Normalized: the target's net ÷ the loading control's net in the lane";
      return note;
    }
    const baseline = ratioText(series.baseline);
    const note = span("column-note", short ? "" : "fold change");
    if (narrow) {
      note.append(hidden(" · "), document.createElement("br"));
    } else if (!short) {
      note.append(" · ");
    }
    note.append(`baseline ${baseline}`);
    note.title =
      `Fold change vs ${reference}: each normalized value ÷ ${baseline},` +
      ` the mean normalized value of ${reference}`;
    return note;
  }

  renderHead(columns) {
    const reference = this.results.reference_condition;
    const split = columns.series.some((s) => s.all);
    const twoRows = columns.nets.length > 0 || split;
    const rows = twoRows ? 2 : 1;
    const top = document.createElement("tr");
    const bottom = document.createElement("tr");
    top.append(
      this.header("lt-lane", ["#"], { rows, className: "lane-head" }),
      this.header("lt-condition", ["Condition"], { rows, className: "condition" }),
      this.header("lt-sample", ["Sample"], { rows }),
      this.header("lt-included", ["Include"], { rows }),
    );
    if (columns.nets.length) {
      const net = this.header("lt-nets", ["Net (signal − background)"], {
        cols: columns.nets.length,
        scope: "colgroup",
        className: "spanning",
      });
      net.title = "Each box's integrated signal less its background, in image counts";
      top.append(net);
      columns.nets.forEach((column, j) => {
        const name = span("column-name", "");
        name.append(swatch(colorOf(this.project, column.protein_id)), clipped(column.name));
        bottom.append(this.header(`lt-net-${j}`, [name], { className: "value" }));
      });
    }
    columns.series.forEach((series, k) => {
      const one = series.applied;
      if (!series.all) {
        top.append(
          this.header(`lt-series-${k}`, [seriesName(one), this.kindNote(one, reference)], {
            rows,
            className: "value",
          }),
        );
        return;
      }
      // Two sets side by side (#71). The header over both says "fold change"
      // only when both are: a set with no usable reference (its reference lanes
      // all excluded) has normalized values, and its own column says so.
      const both = one.value_kind === FOLD_CHANGE && series.all.value_kind === FOLD_CHANGE;
      const parts = [seriesName(one)];
      if (both) {
        parts.push(span("column-note", "fold change"));
      }
      top.append(
        this.header(`lt-series-${k}`, parts, {
          cols: 2,
          scope: "colgroup",
          className: "spanning",
        }),
      );
      const [appliedSet, allSet] = this.results.sets;
      for (const [id, text, each, set] of [
        ["applied", "applied", one, appliedSet],
        ["all", "all lanes", series.all, allSet],
      ]) {
        const cell = this.header(
          `lt-series-${k}-${id}`,
          [
            span("column-name", text),
            this.kindNote(each, reference, { short: both, narrow: !both }),
          ],
          { className: "value" },
        );
        cell.title = set.label || text;
        bottom.append(cell);
      }
    });
    const head = $("lane-head");
    head.replaceChildren(top, ...(twoRows ? [bottom] : []));
    this.stickyColumns.disconnect();
    this.stickyColumns.observe($("lt-lane"));
    this.stickyColumns.observe($("lt-condition"));
  }

  textInput(index, field) {
    const input = document.createElement("input");
    input.type = "text";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.dataset.key = `${index}:${field}`;
    input.addEventListener("blur", () => this.sendTyped(index, field));
    input.addEventListener("keydown", (event) => this.key(event, index, field));
    // Text cut at the field's width is in its tooltip.
    input.addEventListener("pointerenter", () => {
      input.title = input.scrollWidth > input.clientWidth ? input.value : "";
    });
    return input;
  }

  // A lane's row: its number and tags (the row's header), then its three
  // editable cells; the value cells follow, drawn at each render.
  makeRow(index) {
    const row = document.createElement("tr");
    const head = this.header(`lt-row-${index}`, [], { scope: "row", className: "lane-head" });
    const tags = span("lane-tags", "");
    head.append(span("lane-number", String(index + 1)), hidden(" "), tags);
    head.prepend(hidden("Lane "));
    const condition = this.textInput(index, "condition");
    condition.setAttribute("aria-label", `Condition, lane ${index + 1}`);
    condition.className = "lane-condition";
    const sample = this.textInput(index, "sample");
    sample.setAttribute("aria-label", `Sample, lane ${index + 1}`);
    sample.placeholder = "own sample";
    sample.className = "lane-sample";
    const included = document.createElement("input");
    included.type = "checkbox";
    included.dataset.key = `${index}:included`;
    included.setAttribute("aria-label", `Include lane ${index + 1}`);
    included.addEventListener("change", () => this.commit(index, "included"));
    included.addEventListener("keydown", (event) => this.key(event, index, "included"));
    const cells = [
      [condition, "lt-condition", "condition"],
      [sample, "lt-sample", ""],
      [included, "lt-included", "include"],
    ].map(([control, column, className]) => {
      const cell = document.createElement("td");
      cell.className = className;
      cell.setAttribute("headers", `${column} ${head.id}`);
      cell.append(control);
      return cell;
    });
    row.append(head, ...cells);
    return { row, head, tags, condition, sample, included };
  }

  // The lanes whose values were averaged as technical repeats: included, with
  // a value in a series that averaged their condition and sample. A repeat
  // with no value there (no box) was not averaged, and has no tag.
  averagedLanes(project, results) {
    const lanes = new Set();
    for (const series of results.sets[0].series) {
      const repeats = new Set(series.averaged.map((a) => `${a.condition}\n${a.sample}`));
      project.lanes.forEach((lane, i) => {
        const key = `${lane.condition}\n${lane.sample}`;
        if (
          lane.included &&
          lane.sample !== null &&
          series.normalized[i] !== null &&
          repeats.has(key)
        ) {
          lanes.add(i);
        }
      });
    }
    return lanes;
  }

  renderRows(project, results, columns, answered) {
    const body = $("lane-rows");
    const n = project.lanes.length;
    while (this.rows.length > n) {
      const gone = this.rows.pop();
      gone.row.remove();
      for (const field of FIELDS) {
        for (const map of [this.stored, this.synced, this.sent]) {
          map.delete(`${this.rows.length}:${field}`);
        }
      }
    }
    while (this.rows.length < n) {
      const row = this.makeRow(this.rows.length);
      body.append(row.row);
      this.rows.push(row);
    }
    this.renderEmpty(body, n, columns);
    // The condition and sample fields are as wide as the longest (within limits, app.css).
    const table = $("lane-table");
    const longest = (texts) => String(Math.max(0, ...texts.map((text) => (text || "").length)));
    table.style.setProperty("--condition-chars", longest(project.lanes.map((l) => l.condition)));
    table.style.setProperty("--sample-chars", longest(project.lanes.map((l) => l.sample)));
    const averaged = this.averagedLanes(project, results);
    project.lanes.forEach((lane, i) => {
      const row = this.rows[i];
      for (const field of FIELDS) {
        const value = field === "sample" ? lane.sample || "" : lane[field];
        this.fill(`${i}:${field}`, row[field], value, answered);
      }
      row.row.classList.toggle("excluded", !lane.included);
      const tags = [];
      if (!lane.included) {
        tags.push(span("tag excluded-tag", "excluded"));
      }
      if (lane.condition === results.reference_condition) {
        tags.push(span("tag ref-tag", "ref"));
      }
      if (averaged.has(i)) {
        tags.push(
          mark(
            "≈",
            "tag repeat-tag",
            "technical repeat, averaged",
            "A technical repeat: averaged with the lanes of the same condition and sample",
          ),
        );
      }
      row.tags.replaceChildren(...tags);
      const values = this.valueCells(i, columns);
      while (row.row.children.length > 4) {
        row.row.lastChild.remove();
      }
      row.row.append(...values);
    });
  }

  // No lanes yet: one row that says how to declare them.
  renderEmpty(body, n, columns) {
    let empty = $("lanes-empty");
    if (n) {
      if (empty) {
        empty.remove();
      }
      return;
    }
    if (!empty) {
      empty = document.createElement("tr");
      empty.id = "lanes-empty";
      const cell = document.createElement("td");
      cell.className = "hint";
      cell.textContent =
        "No lanes yet: type the conditions in lane order under Conditions, separated by" +
        " commas, then Set lanes.";
      empty.append(cell);
    }
    empty.firstChild.colSpan =
      4 + columns.nets.length + columns.series.reduce((sum, s) => sum + (s.all ? 2 : 1), 0);
    body.append(empty);
  }

  valueCell(text, headers, { title = "", className = "value" } = {}) {
    const cell = document.createElement("td");
    cell.className = className;
    cell.setAttribute("headers", headers);
    cell.append(...(Array.isArray(text) ? text : [text]));
    if (title) {
      cell.title = title;
    }
    return cell;
  }

  // Whether a protein was examined in the lane and found below the detection limit.
  notDetected(proteinId, lane) {
    const column = this.results.proteins.find((c) => c.protein_id === proteinId);
    return Boolean(column) && column.detected[lane] === false;
  }

  seriesCell(series, lane, headers) {
    const value =
      series.value_kind === FOLD_CHANGE ? series.fold_change[lane] : series.normalized[lane];
    if (value !== null) {
      return this.valueCell(ratioText(value), headers);
    }
    if (this.notDetected(series.target_id, lane) || this.notDetected(series.loading_id, lane)) {
      return this.valueCell("n.d.", headers, {
        title: "Not detected: a band below the detection limit, so no value",
        className: "value missing",
      });
    }
    return this.valueCell("—", headers, { title: "No value", className: "value missing" });
  }

  valueCells(lane, columns) {
    const row = `lt-row-${lane}`;
    const cells = columns.nets.map((column, j) => {
      const headers = `lt-nets lt-net-${j} ${row}`;
      if (column.detected[lane] === false) {
        return this.valueCell("n.d.", headers, {
          title: "Not detected: below the detection limit, so no value",
          className: "value missing",
        });
      }
      const net = column.nets[lane];
      if (net === null) {
        return this.valueCell("—", headers, { title: "No box", className: "value missing" });
      }
      if (column.clipped[lane] !== true) {
        return this.valueCell(netText(net), headers);
      }
      const why = "Over-exposed: pixels at the detector limit, so the net is an under-estimate";
      return this.valueCell(
        [mark("▲", "over", "over-exposed", why), " ", netText(net)],
        headers,
        { title: why, className: "value clipped" },
      );
    });
    columns.series.forEach((series, k) => {
      if (series.all) {
        cells.push(
          this.seriesCell(series.applied, lane, `lt-series-${k} lt-series-${k}-applied ${row}`),
          this.seriesCell(series.all, lane, `lt-series-${k} lt-series-${k}-all ${row}`),
        );
      } else {
        cells.push(this.seriesCell(series.applied, lane, `lt-series-${k} ${row}`));
      }
    });
    return cells;
  }

  // `refused`: the ids of the boxes a lanes_in_use message names.
  showError(text, refused = null) {
    this.refused = text ? refused : null;
    const line = $("lanes-error");
    if (line.textContent !== text) {
      line.textContent = text;
    }
    line.hidden = !text;
  }

  // --- The keyboard ---

  // Up and Down go to the same column in the row above or below; Enter commits
  // the text and goes down, as in a spreadsheet; Escape puts back what the
  // table has (or was sent) before the typing.
  key(event, index, field) {
    if (event.altKey || event.ctrlKey || event.metaKey || event.isComposing) {
      return;
    }
    const text = field !== "included";
    const step = { ArrowUp: -1, ArrowDown: 1 }[event.key];
    if (step !== undefined || (text && event.key === "Enter")) {
      event.preventDefault();
      const next = this.rows[index + (step === undefined ? 1 : step)];
      if (next) {
        next[field].focus(); // leaving the field sends what was typed (sendTyped())
        if (text) {
          next[field].select();
        }
      } else if (text && event.key === "Enter") {
        this.sendTyped(index, field);
      }
    } else if (text && event.key === "Escape") {
      if (this.restore(`${index}:${field}`, this.rows[index][field])) {
        event.preventDefault();
      }
    }
  }

  // Leaving a text field (or Enter in the last row) sends the text typed in it
  // since it last matched the table, or since its commit awaiting an answer.
  // Text that still equals what it last matched the table with is no change:
  // if the table has changed meanwhile (an undo, a redo or another edit,
  // answered while typing over it was kept, then taken back), the field shows
  // the table's value, so it neither drifts from the table nor sends the old
  // value back. Not the change event: it compares with the text at focus.
  sendTyped(index, field) {
    const row = this.rows[index];
    if (!row) {
      return;
    }
    const key = `${index}:${field}`;
    const input = row[field];
    const expected = this.expected(key);
    const typed = this.sent.has(key) || input.value !== this.synced.get(key);
    if (expected !== undefined && input.value !== expected && typed) {
      this.commit(index, field);
    } else {
      this.restore(key, input);
    }
  }

  // --- Edits ---

  // Send the lane table that `change(table)` makes of the stored one, once the
  // edits made before it have their answers (so it starts from what they
  // stored); `keys` are the controls it sends, which then show the server's
  // values. `done()` runs once it is stored, or when it changes nothing. Gives
  // the answer, or null.
  edit(change, { sent = new Map(), done = null } = {}) {
    const after = this.handlers.pending();
    // The commits this edit sends are answered (or dropped): their controls
    // may show the server's values again. Gives key -> the value sent.
    const answered = () => {
      const values = new Map();
      for (const [key, commit] of sent) {
        if (this.sent.get(key) === commit) {
          this.sent.delete(key);
        }
        values.set(key, commit.value);
      }
      return values;
    };
    const run = this.handlers.queueEdit(
      async (current) => {
        const project = this.project;
        const before = tableOf(project);
        const lanes = change(tableOf(project));
        if (!lanes || sameTable(before, lanes)) {
          this.render(project, this.results, answered());
          if (lanes && done) {
            done();
          }
          return null;
        }
        const body = {
          lanes: lanes.map((lane) => ({
            condition: lane.condition,
            sample: tidy(lane.sample) ? lane.sample : null,
            included: lane.included,
          })),
        };
        const reference = project.reference_condition;
        const renamed = followedReference(reference, before, lanes);
        if (renamed !== null) {
          body.reference_condition = renamed;
        }
        try {
          const answer = await this.handlers.send("PUT", "/api/lanes", body);
          if (!answer || !current()) {
            return null; // another project is shown now
          }
          this.showError("");
          this.render(this.project, this.results, answered());
          this.report(answer, reference);
          if (done) {
            done();
          }
          return answer;
        } catch (error) {
          if (!current()) {
            return null;
          }
          this.render(this.project, this.results, answered());
          const refused = error.code === "lanes_in_use" ? error.ids : null;
          this.showError(this.explain(error), refused);
          return null;
        }
      },
      { after },
    );
    // Not run (another project was asked for): nothing awaits an answer.
    run.then(answered, answered);
    return run;
  }

  // What the server did beyond storing the table, in the status line.
  report(answer, reference) {
    if (answer.reference_cleared) {
      this.handlers.status(
        `No lane has the reference condition ${reference} now, so there is no reference:` +
          " the values are normalized, not fold changes. Choose one under Reference.",
      );
    } else if (answer.respelled.length) {
      const which = lanesText(answer.respelled);
      this.handlers.status(
        `${which} took the spelling already in the table: look-alike characters` +
          " (µ and μ, for example) make one name.",
      );
    }
  }

  // A refusal in the user's terms: the boxes it names by protein and lane.
  explain(error) {
    if (error.code === "lanes_in_use") {
      const found = error.ids
        .map((id) => {
          for (const protein of this.project.proteins) {
            const band = protein.bands.find((b) => b.id === id);
            if (band) {
              return { protein, band };
            }
          }
          return null;
        })
        .filter(Boolean);
      if (found.length) {
        const lanes = [...new Set(found.map((f) => f.band.lane_index))].sort((a, b) => a - b);
        const proteins = [...new Set(found.map((f) => f.protein.name))];
        const hold = lanes.length === 1 ? "holds" : "hold";
        const it = found.length === 1 ? "it" : "them";
        const lane = lanes.length === 1 ? "the lane" : "those lanes";
        return (
          `${lanesText(lanes)} ${hold} ${counted(found.length, "box", "boxes")} of` +
          ` ${inWords(proteins)}: delete ${it} first to remove ${lane}.`
        );
      }
    }
    return sentence(error.message);
  }

  // A typed condition or sample, or a ticked include box, of lane `index`.
  commit(index, field) {
    const row = this.rows[index];
    if (!row) {
      return;
    }
    const key = `${index}:${field}`;
    const value = field === "included" ? row.included.checked : row[field].value;
    if (field === "condition" && !tidy(value)) {
      this.restore(key, row.condition);
      this.showError(`Lane ${index + 1} needs a condition: it was put back.`);
      return;
    }
    const commit = { value };
    this.sent.set(key, commit);
    this.edit(
      (table) => {
        if (index >= table.length) {
          return null; // that lane is gone
        }
        table[index][field] = value;
        return table;
      },
      { sent: new Map([[key, commit]]) },
    );
  }

  // Declare the lanes from typed conditions, one lane per comma-separated
  // label: the conditions of the lanes there are now are overwritten, and each
  // keeps its sample and include flag; new lanes are their own samples and
  // included; a shorter list drops the last lanes (refused while they hold boxes).
  setLanes(text) {
    const labels = text.split(",").map(tidy);
    if (labels.length > 1 && !labels[labels.length - 1]) {
      labels.pop(); // a comma at the end
    }
    if (labels.every((label) => !label)) {
      this.showError("Type the conditions in lane order, separated by commas.");
      return;
    }
    const blank = labels.findIndex((label) => !label);
    if (blank >= 0) {
      this.showError(
        `Lane ${blank + 1} has no condition: two commas in a row, or one at the start?`,
      );
      return;
    }
    this.edit(
      (table) =>
        labels.map((condition, i) => ({
          condition,
          sample: i < table.length ? table[i].sample : null,
          included: i < table.length ? table[i].included : true,
        })),
      {
        done: () => {
          if ($("bulk-conditions").value === text) {
            $("bulk-conditions").value = "";
          }
        },
      },
    );
  }

  // A lane after the last, with the last one's condition (a lane needs one)
  // and its own sample; its condition is chosen for typing over.
  addLane() {
    this.edit(
      (table) => {
        if (!table.length) {
          return null;
        }
        const last = table[table.length - 1];
        return [...table, { condition: last.condition, sample: null, included: true }];
      },
      {
        done: () => {
          const added = this.rows[this.rows.length - 1];
          const active = document.activeElement;
          if (added && (!active || active === document.body || active === $("add-lane"))) {
            added.condition.focus();
            added.condition.select();
          }
        },
      },
    );
  }

  removeLane() {
    this.edit((table) => (table.length ? table.slice(0, -1) : null));
  }

  chooseReference(value) {
    const after = this.handlers.pending();
    this.handlers.queueEdit(
      async (current) => {
        const condition = value === "" ? null : value;
        if (condition === this.project.reference_condition) {
          this.render(this.project, this.results, new Map([["reference", value]]));
          return null;
        }
        try {
          const answer = await this.handlers.send("PUT", "/api/reference", { condition });
          if (!answer || !current()) {
            return null;
          }
          this.showError("");
          this.render(this.project, this.results, new Map([["reference", value]]));
          return answer;
        } catch (error) {
          if (!current()) {
            return null;
          }
          this.render(this.project, this.results, new Map([["reference", value]]));
          this.showError(this.explain(error));
          return null;
        }
      },
      { after },
    );
  }
}
