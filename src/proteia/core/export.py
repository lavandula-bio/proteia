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
writes the reproducibility record. The README says how the charts were tested
and gives each chart's legend as its caption; the images draw a key of their
marks, and the legend under the chart only when asked. Every file numbers
lanes from 1, as the app does (:func:`~proteia.core.model.lane_number`); the
record's content keeps the stored 0-based index.

A bundle's file names come from protein names and result-set labels, made safe
for every supported file system (:func:`file_stem`, which keeps µ, α and β),
cut to fit the room the folder's path leaves (:func:`bundle_files`), and told
apart when two would name one file (:func:`unique_stems`). A lane table's
column names come from protein names too; where two columns would share a
name, one takes a number (:func:`lane_columns`), and a bundle's README says
which and why.

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
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, NamedTuple

import proteia
from proteia.core.analyze import (
    ALPHA,
    MANN_WHITNEY_PERMUTATIONS,
    LaneNets,
    ReduceMethod,
    StatisticsSetting,
    TestComparisons,
    TestFamily,
    TestScale,
    registered_test,
)
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


# --- Column names ---


class SeriesColumn(NamedTuple):
    """One column of a series in a lane table: its normalized values, or its
    fold changes vs ``reference``."""

    target_id: str
    loading_id: str
    reference: str | None = None  # None: the normalized values

    def header(self, target: str, loading: str) -> str:
        """The column's name, from the names its proteins' columns carry, as
        the web UI's lane table names it: ``β-catenin ÷ α-tubulin normalized``,
        ``β-catenin ÷ α-tubulin fold change vs vehicle``."""
        kind = "normalized" if self.reference is None else f"fold change vs {self.reference}"
        return f"{target} ÷ {loading} {kind}"

    def holds(self, target: str, loading: str) -> str:
        """What the column holds, with its proteins' names, for a README."""
        if self.reference is None:
            values = "normalized values"
        else:
            values = f'fold changes vs "{self.reference}"'
        return f'{values} of the target "{target}" over the loading control "{loading}"'


@dataclass(frozen=True)
class LaneColumns:
    """The names of a lane table's protein and series columns (:func:`lane_columns`)."""

    proteins: dict[str, str]  # by protein id: the name its net and clipping columns carry
    series: dict[SeriesColumn, str]  # the name of each series column
    renamed: tuple[str, ...]  # per renamed column, a sentence: what it holds, and why


# What each lane column holds, for a README.
_LANE_HOLDS: Final = {
    "lane": "the lane numbers",
    "condition": "the lanes' conditions",
    "sample": "the lanes' samples",
    "include": "the include flags",
}


def _protein_headers(name: str) -> tuple[str, str]:
    """The columns of a protein whose columns carry ``name``: its nets, then its
    clipping flags (as :func:`lane_table_bytes` names them)."""
    return name, f"{name} clipped"


def _numbered(text: str, free: Callable[[str], bool]) -> str:
    """``text`` with the lowest number from 2 that ``free`` takes: ``text (2)``."""
    number = 2
    while not free(f"{text} ({number})"):
        number += 1
    return f"{text} ({number})"


def lane_columns(
    proteins: Sequence[tuple[str, str]], series: Sequence[SeriesColumn] = ()
) -> LaneColumns:
    """The names of the columns of a lane table with ``proteins`` ((id, name),
    in table order) and ``series`` columns, each unique among the table's
    headers, as :func:`lane_table_bytes` requires; names are compared exactly,
    as it compares them.

    A protein's columns carry its name (its nets, then ``<name> clipped``)
    unless another column has the name of either: a lane column
    (:data:`LANE_COLUMNS`), a column of a protein before it, or a series column
    as the proteins' own names name it. Then they carry its name with the
    lowest number that frees both and that no protein's own columns have:
    ``GAPDH clipped (2)`` and ``GAPDH clipped (2) clipped`` for a protein
    ``GAPDH clipped`` after ``GAPDH``. A series column is named from its
    proteins' columns (:meth:`SeriesColumn.header`), so a renamed protein's
    series carry its number; one whose name a column before it has (two series
    whose names hold `` ÷ ``, say) takes a number the same way. A series column
    listed twice is one column. ``renamed`` gives a sentence per renamed
    protein or series column: what it holds, and which column has its name.

    The operations refuse a protein name that is a lane column or that would
    share a name with another protein's clipping column (ignoring case), but a
    ``project.json`` may hold one; and nothing refuses a name like a series
    column's.
    """
    names = dict(proteins)
    series = list(dict.fromkeys(series))
    # Each column named so far, with what it holds.
    held: dict[str, str] = {column: _LANE_HOLDS[column] for column in LANE_COLUMNS}
    # The series columns as the proteins' own names name them: proteins give way.
    usual: dict[str, str] = {}
    for column in series:
        target, loading = names[column.target_id], names[column.loading_id]
        usual.setdefault(column.header(target, loading), f"the {column.holds(target, loading)}")
    own = {header for _, name in proteins for header in _protein_headers(name)}
    in_series = {pid for column in series for pid in (column.target_id, column.loading_id)}

    def clash(name: str) -> str | None:
        """The first of the columns ``name`` gives a protein that another column has."""
        return next((h for h in _protein_headers(name) if h in held or h in usual), None)

    columns: dict[str, str] = {}
    renamed: list[str] = []
    for protein_id, name in proteins:
        column = name
        taken = clash(name)
        if taken is not None:
            column = _numbered(
                name, lambda c: clash(c) is None and own.isdisjoint(_protein_headers(c))
            )
            net, clipped = _protein_headers(column)
            where = ", and in those of its series" if protein_id in in_series else ""
            holder = held[taken] if taken in held else usual[taken]
            renamed.append(
                f'The protein "{name}" is named "{column}" in its columns, "{net}" and'
                f' "{clipped}"{where}: the column "{taken}" holds {holder}.'
            )
        columns[protein_id] = column
        net, clipped = _protein_headers(column)
        held[net] = f'the nets of the protein "{name}"'
        held[clipped] = f'the clipping flags of the protein "{name}"'

    headers: dict[SeriesColumn, str] = {}
    for column in series:
        holds = column.holds(names[column.target_id], names[column.loading_id])
        header = column.header(columns[column.target_id], columns[column.loading_id])
        if header in held:
            numbered = _numbered(header, lambda c: c not in held and c not in own)
            renamed.append(
                f'The {holds} are in the column "{numbered}": the column "{header}"'
                f" holds {held[header]}."
            )
            header = numbered
        headers[column] = header
        held[header] = f"the {holds}"
    return LaneColumns(columns, headers, tuple(renamed))


def _series_values(
    results: Results, series: SeriesResult
) -> list[tuple[SeriesColumn, Sequence[float | None]]]:
    """The columns of a series in a set's lane table, with their values: its
    normalized values and, when the set forms them, its fold changes."""
    columns: list[tuple[SeriesColumn, Sequence[float | None]]] = [
        (SeriesColumn(series.target_id, series.loading_id), series.normalized)
    ]
    if series.fold_change is not None:
        reference = results.reference_condition
        columns.append(
            (SeriesColumn(series.target_id, series.loading_id, reference), series.fold_change)
        )
    return columns


def result_columns(results: Results) -> LaneColumns:
    """The column names of the lane tables of ``results`` and of its all-lanes
    set (:func:`lane_columns`): one naming for both, so a protein's or a
    series' columns have one name in every table of a bundle."""
    sets = [results] if results.all_lanes is None else [results, results.all_lanes]
    return lane_columns(
        [(column.protein_id, column.name) for column in results.proteins],
        [
            column
            for one in sets
            for series in one.series
            for column, _ in _series_values(one, series)
        ],
    )


# --- The export bundle ---


def result_set_table(results: Results, columns: LaneColumns | None = None) -> bytes:
    """The lane table of one result set (:func:`lane_table_bytes`), as the web
    UI's lane table shows it: the lanes with this set's include flags, each
    protein's net and clipping flags, then each series' normalized values and,
    when the set forms its fold changes, those. ``columns`` names the columns
    (by default, :func:`result_columns` of ``results``)."""
    if columns is None:
        columns = result_columns(results)
    names = columns.proteins
    return lane_table_bytes(
        [lane.condition for lane in results.lanes],
        [lane.sample for lane in results.lanes],
        [lane.included for lane in results.lanes],
        [(names[column.protein_id], column.nets) for column in results.proteins],
        clipped={names[column.protein_id]: column.clipped for column in results.proteins},
        series=[
            (columns.series[column], values)
            for series in results.series
            for column, values in _series_values(results, series)
        ],
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


def chart_bytes(spec: PlotSpec, chart_format: ChartFormat, *, statement: bool = False) -> bytes:
    """``spec`` drawn as the screen draws it (:mod:`proteia.viz`), in
    ``chart_format``; with ``statement``, its legend text drawn under it."""
    from proteia import viz  # matplotlib loads only when a chart is drawn

    if chart_format is ChartFormat.SVG:
        return viz.render_svg(spec, statement=statement)
    if chart_format is ChartFormat.PNG:
        return viz.render_png(spec, dpi=CHART_PNG_DPI, statement=statement)
    return viz.render_pdf(spec, statement=statement)


def bundle_files(
    results: Results,
    *,
    formats: Sequence[ChartFormat],
    exported_at: str,
    name_fits: Callable[[str], bool] = _fits_anywhere,
    statement_in_charts: bool = False,
) -> dict[str, bytes]:
    """Every file of a bundle but its record, by name, in the order to write them.

    For each result set (the one that applies the exclusions, then the all-lanes
    set when there is one): its lane table (:func:`result_set_table`); then, set
    by set, each series' chart in each of ``formats``, in that order; then
    :data:`BUNDLE_README_FILE`, which names every file, the record
    (:data:`BUNDLE_RECORD_FILE`) too. With two sets, each file name carries its
    set's label, as each chart does in its subtitle: ``lane-table (All
    lanes).csv``, ``chart β-catenin ÷ α-tubulin (All lanes).svg``. A series with
    no chart (no value, or no usable reference) has no chart file. The README
    gives each chart's legend (its statement) as its caption; with
    ``statement_in_charts`` each image draws it under the chart too.

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
    columns = result_columns(results)
    table_fits = _stem_fits(name_fits, ["csv"], len(sets))
    table_stems = unique_stems([_table_stem(one, table_fits) for one in sets])
    for one, stem in zip(sets, table_stems, strict=True):
        name = f"{stem}.csv"
        files[name] = result_set_table(one, columns)
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
            files[name] = chart_bytes(chart, chart_format, statement=statement_in_charts)
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
        results,
        sets,
        about,
        no_chart=no_chart,
        exported_at=exported_at,
        renamed=columns.renamed,
        legends=[
            (stem, chart.statement) for (_, _, chart), stem in zip(charted, stems, strict=True)
        ],
        statement_in_charts=statement_in_charts,
    )
    return files


def _paragraph(text: str, indent: str = "") -> list[str]:
    lines = textwrap.wrap(
        text,
        _WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return [*lines, ""]


_DESIGN: Final = (
    "the design (the value kind, the replicates per condition, the reference, and"
    " whether every tested value is above 0)"
)


def _tests_of(family: TestFamily, comparisons: TestComparisons) -> str:
    """The registered tests of ``family`` that ``comparisons`` can run, by the
    number of conditions, in the names the charts' legends use."""
    two = registered_test(family, TestComparisons.ALL_PAIRS, 2).name
    pairs = registered_test(family, TestComparisons.ALL_PAIRS, 3).name
    each = (
        f"{registered_test(family, TestComparisons.VS_REFERENCE, 3).name} of each"
        " condition against the reference"
    )
    if comparisons is TestComparisons.ALL_PAIRS:
        return f"{two} for two conditions and {pairs} for three or more"
    if comparisons is TestComparisons.VS_REFERENCE:
        return f"{two} for two conditions and {each} for three or more"
    return (
        f"{two} for two conditions, {each} for three or more when the reference is"
        f" tested, and {pairs} otherwise"
    )


def _statistics_text(setting: StatisticsSetting) -> str:
    """How a bundle's charts were tested, for its README: the setting, and for
    each field left ``auto`` the rule that resolved it, naming the tests as the
    charts' legends do (:data:`~proteia.core.analyze.TESTS`)."""
    if setting.family is TestFamily.NONE:
        return "Statistics: none were computed (the test family was set to none)."
    fields = setting.model_dump(mode="json")
    listed = ", ".join(f"{name} {value}" for name, value in fields.items())
    if all(value == "auto" for value in fields.values()):
        chosen = (
            f"each chart's test was chosen automatically from {_DESIGN}, never from the"
            " shape of the values"
        )
    elif "auto" in fields.values():
        chosen = (
            f"the test setting was {listed}; the fields left auto were resolved from"
            f" {_DESIGN}, never from the shape of the values"
        )
    else:
        chosen = f"the test setting was {listed}"
    sentences = [f"Statistics: {chosen}."]
    if setting.family is TestFamily.RANK:
        sentences.append("Rank tests compare the order of the values, which no scale changes.")
    elif setting.scale is TestScale.LOG:
        sentences.append(
            "Values are tested on log values (natural log); a chart with a tested value of"
            " 0 or below has no test, and its legend says why."
        )
    elif setting.scale is TestScale.LINEAR:
        sentences.append("Values are tested on linear values.")
    else:
        sentences.append(
            "Ratios (normalized values and fold changes) are tested on log values (natural"
            " log), or on linear values when a tested value is 0 or below, which the chart's"
            " legend then says; raw values are tested on linear values."
        )
    comparisons = setting.comparisons
    if setting.family is TestFamily.AUTO:
        sentences.append(
            f"The tests: with equal n per condition, {_tests_of(TestFamily.POOLED, comparisons)};"
            f" with unequal n, {_tests_of(TestFamily.WELCH, comparisons)}."
        )
    else:
        sentences.append(
            f"The tests ({setting.family.value}): {_tests_of(setting.family, comparisons)}."
        )
    if comparisons is TestComparisons.VS_REFERENCE:
        sentences.append(
            "A chart whose reference is not tested has no test, and its legend says why."
        )
    if setting.family is TestFamily.RANK:
        sentences.append(
            "The Mann-Whitney U test is exact; with tied values and more than"
            f" {MANN_WHITNEY_PERMUTATIONS:,} ways to arrange them, it is the normal"
            " approximation with the tie correction, which the chart's legend then says."
            " The Kruskal-Wallis p is the chi-square approximation."
        )
    sentences.append(
        f"Every test is two-sided at alpha {ALPHA:g}. A test leaves out the conditions with"
        " fewer than 2 replicates or with no value, and those with a replicate below the"
        " detection limit. Each chart is tested on its own, with no correction across"
        f" charts. Brackets join the conditions whose adjusted p is below {ALPHA:g}"
        " (* p < 0.05, ** p < 0.01, *** p < 0.001). Each chart's legend states its test"
        " with its p-value (for comparisons with the reference, each one's), the"
        f" conditions it covers and those it leaves out; {BUNDLE_RECORD_FILE} pins what"
        " each chart's test resolved to and every p-value it gave, unrounded."
    )
    return " ".join(sentences)


def _sentence(line: str) -> str:
    return line if line.endswith((".", "!", "?")) else f"{line}."


def _readme(
    results: Results,
    sets: Sequence[Results],
    about: Mapping[str, str],
    *,
    no_chart: str | None,
    exported_at: str,
    renamed: Sequence[str] = (),
    legends: Sequence[tuple[str, Sequence[str]]] = (),
    statement_in_charts: bool = False,
) -> bytes:
    """The README of a bundle: what it holds and how to read it, in plain text
    (UTF-8, LF). ``about`` describes each file, in order; ``no_chart`` says why
    there is no chart, if there is none; ``renamed`` says which lane-table
    columns were renamed, and why (:func:`lane_columns`). ``legends`` gives each
    chart's file stem and its legend lines, which the README gives as the
    chart's caption; ``statement_in_charts`` says whether the images draw them
    too."""
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
    charts = [chart for one in sets for series in one.series if (chart := series.chart)]
    if any(bar.not_detected_lanes for chart in charts for bar in chart.bars):
        lines += _paragraph(
            "A condition with a replicate not detected (below the detection limit) keeps its"
            " place with no bar: its detected values are points, each replicate not detected"
            " is an open circle in the shaded n.d. row under the axis, and it is not tested."
            " Its axis label counts all its replicates and those not detected: (n=3, 2 n.d.)."
        )
    if charts:
        lines += _paragraph(_statistics_text(results.statistics))
        if statement_in_charts:
            lines += _paragraph(
                "Each image also draws its legend under the chart; the legends are at the"
                " end of this file too, under Legends, as captions."
            )
        else:
            lines += _paragraph(
                "The images draw a key of their marks, not the legend: each chart's legend"
                " is its caption, given at the end of this file under Legends."
            )
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
    if renamed:
        lines += _paragraph(
            "Renamed columns: no two columns of a lane table share a name. Where two"
            " would, one takes a number."
        )
        for sentence in renamed:
            lines += _paragraph(sentence, "    ")
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
    if legends:
        lines += ["", "Legends", "-------", ""]
        for stem, statement in legends:
            lines.append(stem)
            lines += _paragraph(" ".join(_sentence(line) for line in statement), "    ")
    return ("\n".join(lines).rstrip("\n") + "\n").encode("utf-8")
