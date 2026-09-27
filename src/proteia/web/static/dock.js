// SPDX-License-Identifier: Apache-2.0
// The results dock under the image: its header (the results' tier, and
// "Updating…" while an answer or a chart's drawing is awaited), the splitter
// that resizes or hides it, and, in a narrow window, the tabs between the lane
// table and the charts. It shows the results the app hands it and edits nothing.
import { $ } from "/static/dom.js";

const NARROW = "(max-width: 1100px)"; // the lane table and the charts share one place: tabs
const UPDATING_DELAY = 150; // ms an answer may take before "Updating…" shows
const SHARE = 40; // percent of the right side the dock takes at first
const STEP = 5; // percent per arrow key on the splitter
const STAGE_MIN_REM = 8; // the image keeps at least this much height
const DOCK_MIN_REM = 12;

const TIERS = {
  export_only: {
    text: "Nets only",
    title: "Nets only: add a target and a loading control to normalize.",
  },
  normalized: {
    text: "Normalized",
    title: "Target ÷ loading control. Choose a reference condition for fold changes.",
  },
  fold_change: {
    text: "Fold change",
    title: "Target ÷ loading control, relative to the reference condition.",
  },
};

function rem() {
  return parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
}

export class Dock {
  constructor() {
    this.wanted = SHARE; // the share the user chose, in percent of the right side
    this.share = SHARE; // the share the dock takes: the wanted one, within its limits
    this.collapsed = false;
    this.tab = "lanes-panel"; // the panel shown while the dock has tabs
    this.awaited = { answers: 0, charts: 0 }; // answers awaited, chart drawings being fetched
    this.late = false; // one has been awaited longer than UPDATING_DELAY
    this.timer = null;
    this.narrow = window.matchMedia(NARROW);
    this.bindSplitter();
    this.bindTabs();
    $("dock-toggle").addEventListener("click", () => this.collapse(!this.collapsed));
    this.narrow.addEventListener("change", () => this.layout());
    this.layout();
    this.resize(SHARE);
  }

  // --- The header ---

  render(results) {
    const tier = TIERS[results.sets[0].tier];
    const badge = $("tier");
    badge.hidden = !tier;
    if (tier) {
      badge.textContent = tier.text;
      badge.title = tier.title;
    }
  }

  // Count `promise` as awaited until it settles: an answer, or a chart's
  // drawing being fetched (`chart`). "Updating…" shows once one has taken
  // longer than UPDATING_DELAY, until none is awaited; meanwhile the values
  // and charts an answer may change are dimmed while one is awaited (a chart
  // being fetched dims only the image it replaces, charts.js). Gives `promise`.
  track(promise, { chart = false } = {}) {
    const kind = chart ? "charts" : "answers";
    this.awaited[kind] += 1;
    this.showUpdating();
    const done = () => {
      this.awaited[kind] -= 1;
      this.showUpdating();
    };
    promise.then(done, done);
    return promise;
  }

  showUpdating() {
    const { answers, charts } = this.awaited;
    if (!answers && !charts) {
      clearTimeout(this.timer);
      this.timer = null;
      this.late = false;
    } else if (this.timer === null && !this.late) {
      this.timer = setTimeout(() => {
        this.timer = null;
        this.late = true;
        this.showUpdating();
      }, UPDATING_DELAY);
    }
    $("updating").hidden = !this.late;
    $("dock").classList.toggle("waiting", this.late && answers > 0);
    $("dock-body").setAttribute("aria-busy", String(this.late));
  }

  // --- Size and collapse ---

  // The dock's share of the right side, in percent, kept between its minimum
  // height and the image's.
  limits() {
    const height = $("work-area").getBoundingClientRect().height;
    if (!height) {
      return [0, 100];
    }
    const splitter = $("dock-splitter").getBoundingClientRect().height;
    const low = ((DOCK_MIN_REM * rem()) / height) * 100;
    const high = 100 - ((STAGE_MIN_REM * rem() + splitter) / height) * 100;
    return [Math.min(low, high), Math.max(low, high)];
  }

  // Take `share` (the one the user chose), or as near it as the limits allow.
  resize(share) {
    const [low, high] = this.limits();
    this.wanted = share;
    this.share = Math.min(high, Math.max(low, share));
    $("dock").style.flexBasis = `${this.share}%`;
    const splitter = $("dock-splitter");
    splitter.setAttribute("aria-valuemin", String(Math.round(low)));
    splitter.setAttribute("aria-valuemax", String(Math.round(high)));
    splitter.setAttribute("aria-valuenow", String(Math.round(this.share)));
    const percent = Math.round(this.share);
    splitter.setAttribute("aria-valuetext", `Results take ${percent}% of the height`);
  }

  collapse(collapsed) {
    this.collapsed = collapsed;
    $("dock").classList.toggle("collapsed", collapsed);
    $("dock-body").hidden = collapsed;
    const toggle = $("dock-toggle");
    toggle.textContent = collapsed ? "Show" : "Hide";
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.title = collapsed ? "Show the results" : "Hide the results";
    if (collapsed) {
      $("dock-splitter").setAttribute("aria-valuetext", "Results hidden");
    } else {
      this.resize(this.wanted);
    }
  }

  bindSplitter() {
    const splitter = $("dock-splitter");
    let dragging = null; // the pointer id while the splitter is dragged
    const follow = (event) => {
      const area = $("work-area").getBoundingClientRect();
      if (this.collapsed) {
        this.collapse(false);
      }
      const [low, high] = this.limits();
      const share = ((area.bottom - event.clientY) / area.height) * 100;
      this.resize(Math.min(high, Math.max(low, share)));
    };
    splitter.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }
      event.preventDefault(); // no text selection while dragging
      splitter.setPointerCapture(event.pointerId);
      splitter.classList.add("dragging");
      dragging = event.pointerId;
    });
    splitter.addEventListener("pointermove", (event) => {
      if (dragging === event.pointerId) {
        follow(event);
      }
    });
    const stop = (event) => {
      if (dragging === event.pointerId) {
        dragging = null;
        splitter.classList.remove("dragging");
      }
    };
    splitter.addEventListener("pointerup", stop);
    splitter.addEventListener("pointercancel", stop);
    splitter.addEventListener("dblclick", () => this.collapse(!this.collapsed));
    splitter.addEventListener("keydown", (event) => {
      const [low, high] = this.limits();
      const sizes = {
        ArrowUp: Math.min(high, this.share + STEP),
        ArrowDown: Math.max(low, this.share - STEP),
        Home: low,
        End: high,
      };
      if (event.key === "Enter") {
        event.preventDefault();
        this.collapse(!this.collapsed);
      } else if (Object.hasOwn(sizes, event.key)) {
        event.preventDefault();
        if (this.collapsed) {
          this.collapse(false);
        }
        this.resize(sizes[event.key]);
      }
    });
    // A smaller window keeps the image its minimum height; a larger one gives
    // the dock back the share chosen.
    new ResizeObserver(() => {
      if (!this.collapsed) {
        this.resize(this.wanted);
      }
    }).observe($("work-area"));
  }

  // --- Tabs, in a narrow window ---

  tabs() {
    return [...$("dock-tabs").querySelectorAll("[role=tab]")];
  }

  bindTabs() {
    for (const tab of this.tabs()) {
      tab.addEventListener("click", () => this.choose(tab.getAttribute("aria-controls"), false));
    }
    $("dock-tabs").addEventListener("keydown", (event) => {
      const tabs = this.tabs();
      const at = tabs.findIndex((tab) => tab.getAttribute("aria-controls") === this.tab);
      const to = {
        ArrowLeft: (at + tabs.length - 1) % tabs.length,
        ArrowRight: (at + 1) % tabs.length,
        Home: 0,
        End: tabs.length - 1,
      }[event.key];
      if (to !== undefined) {
        event.preventDefault();
        this.choose(tabs[to].getAttribute("aria-controls"), true);
      }
    });
  }

  // A tab chosen while the dock is hidden shows the dock with it.
  choose(panelId, focus) {
    this.tab = panelId;
    if (this.collapsed) {
      this.collapse(false);
    }
    this.layout();
    if (focus) {
      this.tabs()
        .find((tab) => tab.getAttribute("aria-controls") === panelId)
        .focus();
    }
  }

  // Side by side in a wide window, each panel under its own heading; tabs in a
  // narrow one, the chosen panel shown.
  layout() {
    const tabbed = this.narrow.matches;
    $("dock").classList.toggle("tabbed", tabbed);
    $("dock-tabs").hidden = !tabbed;
    for (const tab of this.tabs()) {
      const panel = $(tab.getAttribute("aria-controls"));
      const chosen = panel.id === this.tab;
      tab.setAttribute("aria-selected", String(chosen));
      tab.tabIndex = chosen ? 0 : -1;
      panel.querySelector(".panel-title").hidden = tabbed;
      if (tabbed) {
        panel.setAttribute("role", "tabpanel");
        panel.setAttribute("aria-labelledby", tab.id);
        panel.hidden = !chosen;
      } else {
        panel.removeAttribute("role");
        panel.setAttribute("aria-labelledby", panel.querySelector(".panel-title").id);
        panel.hidden = false;
      }
    }
  }
}
