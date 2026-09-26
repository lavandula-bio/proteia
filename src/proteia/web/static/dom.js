// SPDX-License-Identifier: Apache-2.0
// Small DOM helpers shared by the panels. Text always goes in as text, never as
// markup.

export const $ = (id) => document.getElementById(id);

// A file name inside a sentence, set apart for bidirectional text (between
// U+2068 and U+2069): a direction mark in the name cannot reorder the words
// around it. Typed names need none: the server drops such marks from them.
export function isolate(text) {
  return `⁨${text}⁩`;
}

export function span(className, text) {
  const element = document.createElement("span");
  element.className = className;
  element.textContent = text;
  return element;
}

export function swatch(color) {
  const mark = document.createElement("span");
  mark.className = "swatch";
  mark.style.backgroundColor = color;
  return mark;
}

// Replace a list's children with what `build` adds, keeping the keyboard focus
// on the element that has the same `data-key` (an id) as the focused one had.
export function rebuild(container, build) {
  const active = document.activeElement;
  const key = active && container.contains(active) ? active.dataset.key : undefined;
  container.replaceChildren();
  build();
  if (key !== undefined) {
    const same = [...container.querySelectorAll("[data-key]")].find((e) => e.dataset.key === key);
    if (same) {
      same.focus();
    }
  }
}

// Whether the keyboard focus was lost: the focused element was removed or hidden.
export function focusLost() {
  const active = document.activeElement;
  return !active || active === document.body || !active.isConnected;
}
