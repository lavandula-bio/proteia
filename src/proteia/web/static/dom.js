// SPDX-License-Identifier: Apache-2.0
// Small DOM and wording helpers shared by the panels. Text always goes in as
// text, never as markup.

export const $ = (id) => document.getElementById(id);

// "1 box", "3 boxes".
export function counted(count, one, many) {
  return `${count} ${count === 1 ? one : many}`;
}

// "a", "a and b", "a, b and c".
export function inWords(items) {
  return items.length > 2
    ? `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`
    : items.join(" and ");
}

// A server message as a sentence: capitalized, with a full stop.
export function sentence(text) {
  const capital = text.charAt(0).toUpperCase() + text.slice(1);
  return /[.!?]$/.test(capital) ? capital : `${capital}.`;
}

// Numbers as the page writes them, the same everywhere: a net (integrated
// signal less background, in image counts) as a whole number with thousands
// separators, or to four significant digits below 100, since a float image's
// nets are small (0.42 must not read as 0); a ratio (a normalized value, a
// fold change, a baseline) to three significant digits, so a column of them
// lines up and 1.00 reads as measured.
const WHOLE = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });
const SMALL = new Intl.NumberFormat("en", { maximumSignificantDigits: 4 });
const RATIO = new Intl.NumberFormat("en", {
  minimumSignificantDigits: 3,
  maximumSignificantDigits: 3,
});

export function netText(value) {
  const text = Math.abs(value) >= 100 ? WHOLE.format(value) : SMALL.format(value);
  return text === "-0" ? "0" : text;
}

export function ratioText(value) {
  return RATIO.format(value);
}

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
