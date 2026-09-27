# SPDX-License-Identifier: Apache-2.0
"""File exports that do not need a GUI.

Text files are written as UTF-8. CSV uses ``utf-8-sig`` (UTF-8 with a byte-order
mark) so Excel detects the encoding: condition, sample, and protein names are
typed by the user and often contain characters such as µ, α, or β, which the
locale code page on Windows (e.g. cp950) either cannot encode or encodes in a
way other machines misread.

Two exports are built from these pieces (:mod:`proteia.core.operations`): the
lane table alone (:func:`lane_table_bytes`), and the export bundle (#53), a new
folder per export (:func:`bundle_folder_name`) with each result set's lane table
and charts and a README (:func:`bundle_files`), next to which the operation
writes the reproducibility record. Every file numbers lanes from 1, as the app
does (:func:`~proteia.core.model.lane_number`); the record's content keeps the
stored 0-based index.

A bundle's file names come from protein names and result-set labels, made safe
for every supported file system (:func:`file_stem`, which keeps µ, α and β),
cut to fit the room the folder's path leaves (:func:`bundle_files`), and told
apart when two would name one file (:func:`unique_stems`).

Provisional until the maintainer decides (#53): the chart formats a bundle
writes by default (:data:`DEFAULT_CHART_FORMATS`); cells that a spreadsheet
reads as a formula (text starting with ``=``, ``+``, ``-`` or ``@``, such as a
``+LPS`` condition), which are written as typed; and the folder's name in
local time (:func:`bundle_folder_name`), which does not always sort in time.
"""

from __future__ import annotations

import csv
import io
import math
import textwrap
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

import proteia
from proteia.core.analyze import LaneNets, ReduceMethod
from proteia.core.model import lane_number
from proteia.core.names import name_key
from proteia.core.plotspec import PlotSpec, ValueKind
from proteia.core.results import Results, SeriesResult

CSV_ENCODING = "utf-8-sig"
# The lane-identity columns that lead the lane table, before one column per protein.
LANE_COLUMNS: Final = ("lane", "condition", "sample", "include")
# Nets in the lane table are rounded to this many decimals (reported in export records).
LANE_TABLE_DECIMALS: Final = 3
# Normalized values and fold changes in the lane table are rounded to this many
# decimals (reported in export records): they are near 1, where 3 would lose data.
LANE_TABLE_RATIO_DECIMALS: Final = 6


class ChartFormat(StrEnum):
    """A file format a bundle's charts are written in. The values are stable for clients."""

    SVG = "svg"
    PNG = "png"
    PDF = "pdf"


# Provisional (#53): which chart formats a bundle writes by default is the
# maintainer's decision.
DEFAULT_CHART_FORMATS: Final = (ChartFormat.SVG, ChartFormat.PNG)
CHART_PNG_DPI: Final = 300  # reported in export records
# The two files of a bundle whose names are fixed: never taken from typed text.
BUNDLE_README_FILE: Final = "README.txt"
BUNDLE_RECORD_FILE: Final = "export.record.json"

# The most characters of a protein name and of a set label a file name shows
# (a chart's name stays near 120 characters); fewer when the folder's path
# leaves less room (:func:`bundle_files`).
_NAME_LIMIT: Final = 32
_LABEL_LIMIT: Final = 40
# The room, in ASCII characters, a file name of a bundle needs at least, or the
# export is refused: every fixed name and each lane table's name fits whole,
# and a chart's name keeps some characters of its protein names and its label.
MIN_NAME_ROOM: Final = 48
# Characters Windows refuses in a file name.
_FORBIDDEN: Final = frozenset('<>:"/\\|?*')
# Control, format (bidirectional overrides among them), surrogate and unassigned
# characters (noncharacters such as U+FFFF are unassigned): replaced in names.
_UNSAFE_CATEGORIES: Final = frozenset({"Cc", "Cf", "Cs", "Cn"})
_FORMAT_WORDS: Final = {
    ChartFormat.SVG: "SVG (vector)",
    ChartFormat.PNG: f"PNG, {CHART_PNG_DPI} dpi",
    ChartFormat.PDF: "PDF (vector)",
}
_WIDTH: Final = 78  # of the README's lines


def _cell(value: float | None, decimals: int) -> float | str:
    """A number as the lane table writes it: rounded, or an empty cell when there
    is no value or none that is finite."""
    return "" if value is None or not math.isfinite(value) else round(value, decimals)


def lane_table_bytes(
    conditions: Sequence[str],
    samples: Sequence[str | None],
    included: Sequence[bool],
    proteins: Sequence[tuple[str, LaneNets]],
    *,
    clipped: Mapping[str, Sequence[bool | None]] | None = None,
    series: Sequence[tuple[str, Sequence[float | None]]] = (),
) -> bytes:
    """The per-lane table as CSV bytes: lane identity plus each protein's net.

    One row per lane with ``lane, condition, sample, include``, the lane numbered
    from 1 as the app numbers it (row i is the lane with stored index i), and
    one column per protein, in the order given. A missing net (no box for that
    protein on that lane) is an empty cell; nets are rounded to
    :data:`LANE_TABLE_DECIMALS` decimals. With ``clipped`` (protein name to
    per-lane flags), each such protein's net column is followed by a ``<name>
    clipped`` column: ``yes`` for an over-exposed band, ``no``, or empty when
    there is no box or the band was not checked. ``series`` adds a column per
    (name, per-lane values) after the proteins', such as a series' normalized
    values or fold changes, rounded to :data:`LANE_TABLE_RATIO_DECIMALS`
    decimals; no value, or one that is not finite, is an empty cell. Text is
    written as typed. Encoded as :data:`CSV_ENCODING` with CRLF rows.
    """
    n = len(conditions)
    if len(samples) != n or len(included) != n:
        raise ValueError("conditions, samples, and included must have the same length")
    flags = clipped or {}
    for name, nets in proteins:
        if len(nets) != n:
            raise ValueError(f"protein {name!r} has {len(nets)} nets but there are {n} lanes")
        if name in flags and len(flags[name]) != n:
            raise ValueError(
                f"protein {name!r} has {len(flags[name])} clipping flags but there are {n} lanes"
            )
    for name, values in series:
        if len(values) != n:
            raise ValueError(f"column {name!r} has {len(values)} values but there are {n} lanes")
    unknown = set(flags) - {name for name, _ in proteins}
    if unknown:
        raise ValueError(f"clipping flags for proteins not in the table: {sorted(unknown)}")

    header = list(LANE_COLUMNS)
    for name, _ in proteins:
        header += [name, f"{name} clipped"] if name in flags else [name]
    header += [name for name, _ in series]
    if len(set(header)) != len(header):
        raise ValueError(f"two lane-table columns would share a name: {header}")

    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(header)
    for i in range(n):
        row = [lane_number(i), conditions[i], samples[i] or "", "yes" if included[i] else "no"]
        for name, nets in proteins:
            row.append(_cell(nets[i], LANE_TABLE_DECIMALS))
            if name in flags:
                flag = flags[name][i]
                row.append("" if flag is None else "yes" if flag else "no")
        row += [_cell(values[i], LANE_TABLE_RATIO_DECIMALS) for _, values in series]
        writer.writerow(row)
    return text.getvalue().encode(CSV_ENCODING)


def write_lane_table(
    path: str | Path,
    conditions: Sequence[str],
    samples: Sequence[str | None],
    included: Sequence[bool],
    proteins: Sequence[tuple[str, LaneNets]],
    *,
    clipped: Mapping[str, Sequence[bool | None]] | None = None,
) -> None:
    """Write :func:`lane_table_bytes` to ``path``. Every check runs before the
    file is opened."""
    data = lane_table_bytes(conditions, samples, included, proteins, clipped=clipped)
    Path(path).write_bytes(data)


# --- The export bundle ---


def _series_name(series: SeriesResult) -> str:
    """A series as the web UI's lane table names it: ``β-catenin ÷ α-tubulin``."""
    return f"{series.target} ÷ {series.loading}"


def result_set_table(results: Results) -> bytes:
    """The lane table of one result set (:func:`lane_table_bytes`), as the web
    UI's lane table shows it: the lanes with this set's include flags, each
    protein's net and clipping flags, then each series' normalized values and,
    when the set forms its fold changes, those."""
    columns: list[tuple[str, Sequence[float | None]]] = []
    for series in results.series:
        name = _series_name(series)
        columns.append((f"{name} normalized", series.normalized))
        if series.fold_change is not None:
            vs = f"fold change vs {results.reference_condition}"
            columns.append((f"{name} {vs}", series.fold_change))
    return lane_table_bytes(
        [lane.condition for lane in results.lanes],
        [lane.sample for lane in results.lanes],
        [lane.included for lane in results.lanes],
        [(column.name, column.nets) for column in results.proteins],
        clipped={column.name: column.clipped for column in results.proteins},
        series=columns,
    )


def file_stem(text: str, *, limit: int | None = None) -> str:
    """``text`` as part of a file name that every supported file system takes.

    Whitespace is collapsed to single spaces; each character Windows refuses in
    a name (``<>:"/\\|?*``) and each control, format, surrogate or unassigned
    character is replaced by ``_``; and a text longer than ``limit`` characters
    is cut, ending in ``…``. Letters such as µ, α and β are kept.
    """
    shown = " ".join(text.split())
    safe = "".join(
        "_" if c in _FORBIDDEN or unicodedata.category(c) in _UNSAFE_CATEGORIES else c
        for c in shown
    )
    if limit is not None and len(safe) > limit:
        safe = safe[: limit - 1].rstrip() + "…"
    return safe


def unique_stems(stems: Sequence[str]) -> list[str]:
    """``stems``, each that would name the same file as one before it with the
    lowest free number added: ``name (2)``, ``name (3)``. Two names are the same
    file when their caseless keys (:func:`~proteia.core.names.name_key`) are
    equal, as Windows and macOS compare names (and look-alikes too, to be safe)."""
    taken: set[str] = set()
    unique: list[str] = []
    for stem in stems:
        candidate, number = stem, 1
        while name_key(candidate) in taken:
            number += 1
            candidate = f"{stem} ({number})"
        taken.add(name_key(candidate))
        unique.append(candidate)
    return unique


def bundle_folder_name(moment: datetime) -> str:
    """The name of the folder of a bundle exported at ``moment`` (an aware time):
    the local date and time to the minute, ``2026-09-27 1532``, which holds no
    character any file system refuses. The record's ``exported_at`` gives the
    moment itself, in UTC.

    The names sort in time only while the local offset from UTC stays the same:
    a later export can sort first when it falls in the hour a daylight-saving
    clock repeats, or after the computer's time zone changes (a laptop taken
    abroad, a project shared across zones). Provisional (#53): local time is
    the time people look for; UTC would always sort, but reads as another time.
    """
    return moment.astimezone().strftime("%Y-%m-%d %H%M")


def _labelled(stem: str, label: str | None) -> str:
    """A file's stem with its result set's label (as shown), when there are two sets."""
    return stem if label is None else f"{stem} ({label})"


def _fitted(
    parts: Sequence[tuple[str, int]],
    build: Callable[[Sequence[str]], str],
    fits: Callable[[str], bool],
) -> str:
    """``build`` of ``parts``, each a text shown as :func:`file_stem` shows it in
    at most its number of characters, cut further, one character at a time from
    the longest part shown (the first of the longest), until ``fits`` takes the
    result. ``ValueError`` when it does not fit with every part cut to ``…``."""
    limits = [limit for _, limit in parts]
    while True:
        shown = [
            file_stem(text, limit=limit) for (text, _), limit in zip(parts, limits, strict=True)
        ]
        name = build(shown)
        if fits(name):
            return name
        longest = max(range(len(shown)), key=lambda i: len(shown[i]))
        if len(shown[longest]) <= 1:
            raise ValueError(f"no file name fits the room left: {name!r}")
        limits[longest] = len(shown[longest]) - 1


def _stem_fits(
    fits: Callable[[str], bool], extensions: Sequence[str], count: int
) -> Callable[[str], bool]:
    """Whether a stem, one of ``count`` that may collide, fits with each of
    ``extensions`` and the number :func:`unique_stems` may add to it."""
    reserve = f" ({count})" if count > 1 else ""
    return lambda stem: all(fits(f"{stem}{reserve}.{ext}") for ext in extensions)


def _table_stem(results: Results, fits: Callable[[str], bool]) -> str:
    """The stem of a result set's lane table: ``lane-table (All lanes)``."""
    label = results.label
    return _fitted(
        [(label or "", _LABEL_LIMIT)],
        lambda shown: _labelled("lane-table", None if label is None else shown[0]),
        fits,
    )


def _chart_stem(results: Results, series: SeriesResult, fits: Callable[[str], bool]) -> str:
    """The stem of a series' chart: ``chart β-catenin ÷ α-tubulin (All lanes)``."""
    label = results.label
    return _fitted(
        [(series.target, _NAME_LIMIT), (series.loading, _NAME_LIMIT), (label or "", _LABEL_LIMIT)],
        lambda shown: _labelled(
            f"chart {shown[0]} ÷ {shown[1]}", None if label is None else shown[2]
        ),
        fits,
    )


def _fits_anywhere(name: str) -> bool:
    return True


def chart_bytes(spec: PlotSpec, chart_format: ChartFormat) -> bytes:
    """``spec`` drawn as the screen draws it (:mod:`proteia.viz`), in ``chart_format``."""
    from proteia import viz  # matplotlib loads only when a chart is drawn

    if chart_format is ChartFormat.SVG:
        return viz.render_svg(spec)
    if chart_format is ChartFormat.PNG:
        return viz.render_png(spec, dpi=CHART_PNG_DPI)
    return viz.render_pdf(spec)


def bundle_files(
    results: Results,
    *,
    formats: Sequence[ChartFormat],
    exported_at: str,
    name_fits: Callable[[str], bool] = _fits_anywhere,
) -> dict[str, bytes]:
    """Every file of a bundle but its record, by name, in the order to write them.

    For each result set (the one that applies the exclusions, then the all-lanes
    set when there is one): its lane table (:func:`result_set_table`); then, set
    by set, each series' chart in each of ``formats``, in that order; then
    :data:`BUNDLE_README_FILE`, which names every file, the record
    (:data:`BUNDLE_RECORD_FILE`) too. With two sets, each file name carries its
    set's label, as each chart does in its subtitle: ``lane-table (All
    lanes).csv``, ``chart β-catenin ÷ α-tubulin (All lanes).svg``. A series with
    no chart (no value, or no usable reference) has no chart file.

    ``name_fits`` tells whether the folder takes a file name
    (:func:`~proteia.core.storage.name_fits` for the folder's path). The
    protein names and labels in the file names are cut further, the longest
    first, until every name fits, and are told apart after that. It must take a
    name of :data:`MIN_NAME_ROOM` ASCII characters (``ValueError`` otherwise),
    which leaves every name room enough.
    """
    if not name_fits("x" * MIN_NAME_ROOM):
        raise ValueError(f"the folder takes no file name of {MIN_NAME_ROOM} characters")
    sets = [results] if results.all_lanes is None else [results, results.all_lanes]
    files: dict[str, bytes] = {}
    about: dict[str, str] = {}
    table_fits = _stem_fits(name_fits, ["csv"], len(sets))
    table_stems = unique_stems([_table_stem(one, table_fits) for one in sets])
    for one, stem in zip(sets, table_stems, strict=True):
        name = f"{stem}.csv"
        files[name] = result_set_table(one)
        about[name] = "The lane table" + (f' of the set "{one.label}".' if one.label else ".")

    charted = [
        (one, series, series.chart)
        for one in sets
        for series in one.series
        if series.chart is not None
    ]
    chart_fits = _stem_fits(name_fits, [f.value for f in formats], len(charted))
    stems = unique_stems([_chart_stem(one, series, chart_fits) for one, series, _ in charted])
    for (one, series, chart), stem in zip(charted, stems, strict=True):
        if series.value_kind is ValueKind.FOLD_CHANGE:
            kind = f"fold change vs {one.reference_condition}"
        else:
            kind = "normalized"
        where = f', set "{one.label}"' if one.label else ""
        for chart_format in formats:
            name = f"{stem}.{chart_format.value}"
            files[name] = chart_bytes(chart, chart_format)
            about[name] = (
                f"Chart of {series.target} / {series.loading} ({kind}){where};"
                f" {_FORMAT_WORDS[chart_format]}."
            )
    if not charted:
        no_chart = "no result has a chart (a target normalized to a loading control, with values)"
    elif not formats:
        no_chart = "no chart format was chosen"
    else:
        no_chart = None

    about[BUNDLE_README_FILE] = "This file."
    about[BUNDLE_RECORD_FILE] = (
        "The reproducibility record (JSON): the project's content and its SHA-256,"
        " its action log, the software versions and settings, the compute settings,"
        " and the SHA-256 and size of every other file here."
    )
    files[BUNDLE_README_FILE] = _readme(
        results, sets, about, no_chart=no_chart, exported_at=exported_at
    )
    return files


def _paragraph(text: str) -> list[str]:
    return [*textwrap.wrap(text, _WIDTH, break_long_words=False, break_on_hyphens=False), ""]


def _readme(
    results: Results,
    sets: Sequence[Results],
    about: Mapping[str, str],
    *,
    no_chart: str | None,
    exported_at: str,
) -> bytes:
    """The README of a bundle: what it holds and how to read it, in plain text
    (UTF-8, LF). ``about`` describes each file, in order; ``no_chart`` says why
    there is no chart, if there is none."""
    lines = ["Proteia export", "==============", ""]
    lines += _paragraph(
        f"Exported at {exported_at} (UTC) by Proteia {proteia.__version__}. Every file"
        f" here was written together, from one state of the project. {BUNDLE_RECORD_FILE}"
        " is the reproducibility record of that state: it lists every other file in"
        " this folder with its SHA-256, so each copy can be checked against it."
    )
    if len(sets) == 2:
        applied, every = sets
        lines += _paragraph(
            "Lanes the lane table excludes hold values, so the results come in two"
            " sets, each with files of its own that can be shared on their own:"
            f' "{applied.label}" applies the lane table\'s include flags, and'
            f' "{every.label}" includes every lane.'
        )
    error = results.error_type.value
    if results.method is ReduceMethod.MEAN:
        repeats = "averaged"
    else:
        repeats = "represented by one lane each, the first"
    plotted = results.plot_conditions
    conditions = "all" if plotted is None else ", ".join(repr(c) for c in plotted)
    lines += _paragraph(
        f"Charts: each bar is the mean ± {error} of a condition's samples, and each"
        " point one sample; n counts samples. Technical repeats (lanes with the same"
        f" condition and sample) are {repeats}. Plotted conditions: {conditions}."
    )
    if no_chart is not None:
        lines += _paragraph(f"No chart was written: {no_chart}.")
    lines += _paragraph(
        "Lane tables: one row per lane, numbered from 1 as the app numbers them, with"
        " its condition, its sample and whether the set includes it; each protein's"
        f" net signal (signal minus background, rounded to {LANE_TABLE_DECIMALS}"
        " decimals) and whether its band is clipped (over-exposed: yes, no, or empty"
        " when not checked); then each series' normalized value (target net /"
        " loading-control net) and, when the set has a usable reference condition,"
        f" its fold change, rounded to {LANE_TABLE_RATIO_DECIMALS} decimals. An"
        " empty cell has no value. UTF-8 with a byte-order mark; text as typed."
    )
    lines += ["Files", "-----", ""]
    for name, text in about.items():
        lines.append(name)
        lines += textwrap.wrap(
            text,
            _WIDTH,
            initial_indent="    ",
            subsequent_indent="    ",
            break_long_words=False,
            break_on_hyphens=False,
        )
    return ("\n".join(lines) + "\n").encode("utf-8")
