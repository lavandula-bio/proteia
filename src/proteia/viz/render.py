# SPDX-License-Identifier: Apache-2.0
"""Draw a :class:`~proteia.core.plotspec.PlotSpec` as a bar chart with matplotlib.

A functional, legible draft — bars (mean), error bars, individual replicate
points, and significance brackets, under the chart's title and its set's label,
with its axis titles. A key under the axes says what the marks are (bar: mean ±
SD or SEM, filled point: replicate, open circle: not detected). The statistics
are the chart's legend text (:attr:`~proteia.core.plotspec.PlotSpec.statement`),
never its title: the web page shows that text under the image, and a drawing
takes it under its key when asked (``statement=True``). Publication styling
(fonts, palettes, layout) is a later phase and lives here when it comes;
nothing upstream depends on it.

A condition with a replicate not detected (below the detection limit) keeps its
place on the x axis with no bar: its detected values are points, and each
replicate not detected is an open circle in a shaded "n.d." row under the
baseline, in lane order; a condition with no value at all says so in its empty
slot. A chart with no replicate not detected has no such row.

One figure (:func:`render_figure`) is drawn to three formats, each the same
bytes for the same spec: SVG for the screen and exports (:func:`render_svg`),
and PNG and PDF for exports (:func:`render_png`, :func:`render_pdf`). Each
takes the chart style to draw (:mod:`proteia.viz.styles`); this module's
figure is the ``bar`` style's.

The drawn marks carry ids that name what they show, so a drawing can be read
back against its spec: ``bar-i`` is ``spec.bars[i]`` (drawn only for a bar with
a mean), ``point-i-j`` is ``spec.bars[i].points[j]``, ``nd-i-j`` the j-th
replicate of ``spec.bars[i]`` not detected (``not_detected_lanes[j]``; its y is
layout only, never a value), ``slot-i`` the text of an empty slot and
``bracket-k`` is ``spec.comparisons[k]``; the key is ``key`` and the drawn
statement ``statement``. Only an SVG shows them (:func:`render_svg`).

A name is drawn as typed: matplotlib's math syntax (``$...$``) is not read, so a
``$`` or ``\\`` in a name is drawn and no name can fail to draw, and a character
XML cannot hold is drawn as U+FFFD (:func:`_shown`).
"""

from __future__ import annotations

import io
import re
import threading
from collections.abc import Callable, Sequence
from typing import Final

import matplotlib
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoLocator

from proteia.core.plotspec import Bar, PlotSpec
from proteia.viz.styles import DEFAULT_STYLE, chart_style

_BAR_FACE = "#cbd5e1"
_BAR_EDGE = "#334155"
_POINT = "#0f172a"
_TITLE_MARGIN_PT = 4.0  # the gap a title line keeps from the figure's edges
_PLOT_HEIGHT_IN = 4.5  # the title, the axes and their labels
_KEY_ROW_IN = 0.3  # the key under the axes
_TEXT_PT = 9.0  # the key's and the statement's type size
_LINE_SPACING = 1.3
_STATEMENT_MARGIN_PT = 8.0  # the statement's gap from the figure's edges
# The n.d. row: its shading, its height on paper whatever the data's range, and
# its open circles (larger than the points, so they read apart when shrunk).
_ROW_FACE = "#f1f5f9"
_ROW_PT = 16.0
_ND_MARKER_SIZE = 5.5
_ND_EDGE_WIDTH = 1.2
# Any character outside XML 1.0's Char production.
_NOT_XML: Final = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")

# An SVG the same bytes for the same spec: ids from a fixed salt, not at random,
# and glyphs drawn as paths, so the drawing needs no font of the viewer's.
_SVG_SETTINGS: Final = {"svg.hashsalt": "proteia-chart", "svg.fonttype": "path"}
# No metadata block: no date, and none of the URLs matplotlib's metadata names.
_SVG_METADATA: Final = {"Date": None, "Creator": None, "Type": None, "Format": None}
# A PDF with TrueType (Type 42) fonts, as journals and vector editors take them
# (many refuse Type 3), and no date, creator or producer: the same bytes for
# the same spec.
_PDF_SETTINGS: Final = {"pdf.fonttype": 42}
_PDF_METADATA: Final = {"Creator": None, "Producer": None, "CreationDate": None}
_PNG_METADATA: Final = {"Software": None}  # no software tag
# rc_context changes matplotlib's global settings for as long as it lasts, and
# restores all of them when it ends, so drawings made at once would see (and
# undo) each other's. One at a time, in every format: drawing is CPU-bound
# under the GIL, so threads would not draw faster anyway.
_draw_lock = threading.Lock()


def _shown(text: str) -> str:
    """``text`` as a chart shows it: each character XML cannot hold (a control
    character other than tab, newline and carriage return, an unpaired surrogate,
    U+FFFE or U+FFFF) as the replacement character U+FFFD. A typed name may hold
    U+FFFE or U+FFFF, matplotlib writes every text into an SVG as it is (in a
    comment beside its glyphs), and a browser shows no SVG that is not well-formed
    XML."""
    return _NOT_XML.sub("�", text)


def _point_xs(center: float, n: int, spread: float = 0.18) -> list[float]:
    """Deterministic horizontal spread for individual points (no RNG)."""
    if n == 0:
        return []
    if n == 1:
        return [center]
    step = (2 * spread) / (n - 1)
    return [center - spread + i * step for i in range(n)]


def _replicate_xs(center: float, bar: Bar) -> tuple[list[float], list[float]]:
    """The x of each point and of each replicate not detected: all the bar's
    replicates spread in lane order, so each keeps its place whether it was
    detected or not."""
    if not bar.not_detected_lanes:
        return _point_xs(center, len(bar.points)), []
    lanes = sorted([*bar.lane_indices, *bar.not_detected_lanes])
    xs = dict(zip(lanes, _point_xs(center, len(lanes)), strict=True))
    return [xs[i] for i in bar.lane_indices], [xs[i] for i in bar.not_detected_lanes]


def _tick(spec: PlotSpec, i: int) -> str:
    """``KO\\n(n=3, 3 n.d.)`` for a condition with replicates not detected (n
    counts its replicates), else ``vehicle\\n(n=3)``."""
    bar = spec.bars[i]
    undetected = len(bar.not_detected_lanes)
    if not undetected:
        return f"{_shown(bar.label)}\n(n={bar.n})"
    replicates = spec.coverage[i].replicates if spec.coverage else bar.n + undetected
    return f"{_shown(bar.label)}\n(n={replicates}, {undetected} n.d.)"


def _key(spec: PlotSpec) -> list[tuple[object, str]]:
    """The key's entries, each a mark the chart draws and what it is."""
    entries: list[tuple[object, str]] = []
    if any(bar.mean is not None for bar in spec.bars):
        patch = Patch(facecolor=_BAR_FACE, edgecolor=_BAR_EDGE, linewidth=1.0)
        entries.append((patch, f"Mean ± {spec.error_type.value}"))
    if any(bar.points for bar in spec.bars):
        point = Line2D([], [], marker="o", color=_POINT, markersize=4, linestyle="none")
        entries.append((point, "Replicate"))
    if any(bar.not_detected_lanes for bar in spec.bars):
        circle = Line2D(
            [],
            [],
            marker="o",
            markersize=_ND_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=_POINT,
            markeredgewidth=_ND_EDGE_WIDTH,
            linestyle="none",
        )
        entries.append((circle, "Not detected"))
    return entries


def render_figure(spec: PlotSpec, *, statement: bool = False) -> Figure:
    """Render the spec to a matplotlib :class:`Figure` (no global pyplot state).

    The title's lines: the spec's title and its subtitle (the result set) when it
    has one. The axis titles are the spec's; under the axes, a key says what the
    marks are. With ``statement``, the spec's statement (what the marks are, the
    test or why there is none, the conditions it leaves out) is drawn under the
    key, as the chart's legend text: the figure grows to hold it, so the plot
    keeps its size. A line too wide for the figure, in the title or the
    statement, is broken over lines (:func:`_fit_title`, :func:`_wrap`), so a
    saved figure never crops it.
    """
    width = max(4.0, 1.3 * len(spec.bars) + 1.5)
    fig = Figure(figsize=(width, _PLOT_HEIGHT_IN))
    stated = _wrapped_statement(fig, spec.statement) if statement else []
    key = _key(spec)
    text_in = (len(stated) * _LINE_SPACING + 1) * _TEXT_PT / 72 if stated else 0.0
    below_in = (_KEY_ROW_IN if key else 0.0) + text_in
    height = _PLOT_HEIGHT_IN + below_in
    fig.set_size_inches(width, height)
    rect = (0.0, below_in / height, 1.0, 1.0)  # the plot's part of the figure
    ax = fig.subplots()

    xs = list(range(len(spec.bars)))
    drawn = [i for i, bar in enumerate(spec.bars) if bar.mean is not None]
    if drawn:
        patches = ax.bar(
            drawn,
            [spec.bars[i].mean for i in drawn],
            yerr=[spec.bars[i].error for i in drawn],
            capsize=5,
            width=0.6,
            color=_BAR_FACE,
            edgecolor=_BAR_EDGE,
            linewidth=1.0,
            zorder=1,
        )
        for i, patch in zip(drawn, patches.patches, strict=True):
            patch.set_gid(f"bar-{i}")
    undetected: list[tuple[int, int, float]] = []  # (bar, replicate, x) of each open circle
    for i, (x, bar) in enumerate(zip(xs, spec.bars, strict=True)):
        point_xs, circle_xs = _replicate_xs(x, bar)
        for j, (px, val) in enumerate(zip(point_xs, bar.points, strict=True)):
            ax.plot(px, val, "o", color=_POINT, markersize=4, zorder=3, gid=f"point-{i}-{j}")
        undetected += [(i, j, cx) for j, cx in enumerate(circle_xs)]

    ax.set_xticks(xs)
    ax.set_xticklabels([_tick(spec, i) for i in xs], parse_math=False)
    if len(drawn) < len(
        spec.bars
    ):  # the x range every bar would give: an empty slot keeps its width
        span = len(spec.bars) - 0.4
        ax.set_xlim(-0.3 - 0.05 * span, len(spec.bars) - 0.7 + 0.05 * span)
    ax.set_ylabel(_shown(spec.y_label), parse_math=False)
    if spec.x_label:
        ax.set_xlabel(_shown(spec.x_label), parse_math=False)
    lines = [spec.title or "Quantification"]
    if spec.subtitle:
        lines.append(spec.subtitle)
    lines = [_shown(line) for line in lines]
    ax.set_title("\n".join(lines), parse_math=False)
    ax.spines[["top", "right"]].set_visible(False)

    _draw_significance(ax, spec)
    row = None
    if undetected:
        row = _value_range(ax, spec.bars)
        lo, top = row
        ticks = [t for t in AutoLocator().tick_values(lo, top) if lo - 1e-9 <= t <= top + 1e-9]
        ax.set_yticks(ticks)
        _set_row_depth(fig, ax, lo, top)  # a first guess, so the layout sees the row's label
        ax.tick_params(axis="y", which="minor", length=0, pad=7.0, labelcolor=_BAR_EDGE)
    fig.tight_layout(rect=rect)
    _fit_title(fig, ax, lines, rect)
    if row is not None:
        _draw_row(fig, ax, undetected, *row)
    _draw_slots(ax, spec)
    if key:
        handles, labels = zip(*key, strict=True)
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, rect[1]),
            ncols=len(key),
            frameon=False,
            fontsize=_TEXT_PT,
            handlelength=1.4,
            columnspacing=1.4,
        )
        legend.set_gid("key")
    if stated:
        margin = _STATEMENT_MARGIN_PT / 72
        fig.text(
            margin / width,
            (text_in - 0.5 * _TEXT_PT / 72) / height,
            "\n".join(stated),
            ha="left",
            va="top",
            fontsize=_TEXT_PT,
            linespacing=_LINE_SPACING,
            parse_math=False,
            gid="statement",
        )
    return fig


def _wrapped_statement(fig: Figure, statement: Sequence[str]) -> list[str]:
    """The statement's lines, each broken at spaces to the figure's width less
    its margins."""
    renderer = RendererAgg(int(fig.bbox.width), int(fig.bbox.height), fig.dpi)
    font = FontProperties(size=_TEXT_PT)

    def width(text: str) -> float:
        return renderer.get_text_width_height_descent(text, font, ismath=False)[0]

    room = fig.bbox.width - 2 * _STATEMENT_MARGIN_PT * fig.dpi / 72
    return [piece for line in statement for piece in _wrap(_shown(line), width, room)]


def _value_range(ax, bars: list[Bar]) -> tuple[float, float]:
    """The y range the values take above the n.d. row: the top the axes have, and
    a bottom of 0, or below the lowest value drawn (a point, or a mean less its
    error) when that is negative, so no value is ever cropped."""
    top = ax.get_ylim()[1]
    lows = [value for bar in bars for value in bar.points]
    lows += [bar.mean - bar.error for bar in bars if bar.mean is not None]
    lowest = min(lows, default=0.0)
    lo = 0.0 if lowest >= 0 else lowest - 0.05 * (top - lowest)
    return lo, top


def _set_row_depth(fig: Figure, ax, lo: float, top: float) -> tuple[float, float]:
    """Extend the y range below ``lo`` so the n.d. row is :data:`_ROW_PT` tall on
    paper, and label the row's middle "n.d." with a minor tick, so the numeric
    ticks keep their format. Gives the row's depth and its middle."""
    axes_pt = ax.get_position().height * fig.get_figheight() * 72
    share = _ROW_PT / axes_pt
    depth = (top - lo) * share / (1 - share)
    ax.set_ylim(lo - depth, top)
    middle = lo - depth / 2
    # A minor tick where a major one is (y = 0) would be dropped: the middle never is.
    ax.set_yticks([middle], labels=["n.d."], minor=True)
    return depth, middle


def _draw_row(fig: Figure, ax, undetected: list[tuple[int, int, float]], lo: float, top: float):
    """Draw the n.d. row once the layout has fixed the axes' height: the shading,
    the baseline at 0, and an open circle per replicate not detected."""
    depth, middle = _set_row_depth(fig, ax, lo, top)
    ax.axhspan(lo - depth, lo, color=_ROW_FACE, lw=0, zorder=0)
    ax.axhline(0, color="black", lw=0.8, zorder=2)  # the baseline the bars stand on
    ax.spines["left"].set_bounds(lo, top)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6)
    for i, j, x in undetected:
        ax.plot(
            x,
            middle,
            "o",
            markersize=_ND_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=_POINT,
            markeredgewidth=_ND_EDGE_WIDTH,
            zorder=3,
            gid=f"nd-{i}-{j}",
        )


def _draw_slots(ax, spec: PlotSpec) -> None:
    """Name each empty slot (a condition with no value to draw) at the baseline:
    ``n.d.`` when every replicate was not detected, else ``no value`` (some
    replicates, or all, have no box)."""
    for i, bar in enumerate(spec.bars):
        if bar.mean is None and not bar.points:
            undetected = len(bar.not_detected_lanes)
            replicates = spec.coverage[i].replicates if spec.coverage else undetected
            ax.annotate(
                "n.d." if undetected and undetected == replicates else "no value",
                (i, 0),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color=_BAR_EDGE,
                zorder=3,
                gid=f"slot-{i}",
            )


def _fit_title(fig: Figure, ax, lines: list[str], rect: tuple[float, ...]) -> None:
    """Break each title line too wide for the figure at its spaces, and lay out again.

    The title is centred on the axes, so a line fits when it is no wider than
    twice the distance from the axes' centre to the nearer figure edge, less a
    small margin. tight_layout places the axes by the title's height alone,
    never its width, so that distance is known once the figure is laid out. A
    taller title can shift the axes a little (their tick labels may change), so
    the fit is checked again after each layout, a few times at most. ``rect`` is
    the part of the figure the layout fills.
    """
    renderer = RendererAgg(int(fig.bbox.width), int(fig.bbox.height), fig.dpi)
    font = ax.title.get_fontproperties()

    def width(text: str) -> float:
        return renderer.get_text_width_height_descent(text, font, ismath=False)[0]

    margin = _TITLE_MARGIN_PT * fig.dpi / 72
    shown = lines
    for _ in range(3):
        box = ax.get_position()
        centre = (box.x0 + box.x1) / 2 * fig.bbox.width
        room = 2 * (min(centre, fig.bbox.width - centre) - margin)
        fitted = [piece for line in lines for piece in _wrap(line, width, room)]
        if fitted == shown:
            return
        shown = fitted
        ax.set_title("\n".join(shown))  # the same title text: math syntax still not read
        fig.tight_layout(rect=rect)


def _wrap(line: str, width: Callable[[str], float], room: float) -> list[str]:
    """Break ``line`` at spaces into lines no wider than ``room``, filling each in turn.

    A line that fits comes back whole; a word too wide on its own gets a line to
    itself rather than being cut.
    """
    if width(line) <= room:
        return [line]
    pieces: list[str] = []
    current = ""
    for word in line.split(" "):
        trial = f"{current} {word}" if current else word
        if current and width(trial) > room:
            pieces.append(current.rstrip())
            current = word
        else:
            current = trial
    pieces.append(current)
    return pieces


def _draw_significance(ax, spec: PlotSpec) -> None:
    """Stack significance brackets above the bars, lowest comparisons first. A
    condition with no bar raises the brackets by its highest point."""
    if not spec.comparisons:
        return
    label_to_x = {b.label: i for i, b in enumerate(spec.bars)}
    tops = [b.mean + b.error for b in spec.bars if b.mean is not None]
    tops += [max(b.points) for b in spec.bars if b.mean is None and b.points]
    ceiling = max(tops) if tops else 1.0
    gap = ceiling * 0.08 or 0.08

    level = 1
    for k, comp in enumerate(spec.comparisons):
        if comp.group_a not in label_to_x or comp.group_b not in label_to_x:
            continue
        x1, x2 = sorted((label_to_x[comp.group_a], label_to_x[comp.group_b]))
        y = ceiling + gap * level
        ax.plot(
            [x1, x1, x2, x2],
            [y - gap * 0.3, y, y, y - gap * 0.3],
            color=_BAR_EDGE,
            lw=1.0,
            gid=f"bracket-{k}",
        )
        ax.text(
            (x1 + x2) / 2, y, comp.stars, ha="center", va="bottom", fontsize=10, parse_math=False
        )
        level += 1
    ax.set_ylim(top=ceiling + gap * (level + 0.5))


def _figure(spec: PlotSpec, style: str, statement: bool) -> Figure:
    """``spec`` drawn in ``style`` (:data:`~proteia.viz.styles.CHART_STYLES`);
    ``KeyError`` for a style there is none of."""
    return chart_style(style).render(spec, statement=statement)


def render_svg(spec: PlotSpec, *, style: str = DEFAULT_STYLE, statement: bool = False) -> bytes:
    """The spec's figure in ``style`` as SVG: the same bytes for the same spec,
    with the drawn marks' ids (see the module docstring). ``statement`` draws the
    statement under the chart (:func:`render_figure`).

    Nothing in it runs or loads anything: matplotlib writes no script, no event
    handler and no link, every reference (a clip path, a glyph) points into the
    SVG itself, and there is no metadata. The glyphs are paths, so there is no
    text element and no font to load.
    """
    with _draw_lock, matplotlib.rc_context(_SVG_SETTINGS):
        buffer = io.BytesIO()
        _figure(spec, style, statement).savefig(buffer, format="svg", metadata=_SVG_METADATA)
    return buffer.getvalue()


def render_png(
    spec: PlotSpec, *, dpi: int, style: str = DEFAULT_STYLE, statement: bool = False
) -> bytes:
    """The spec's figure in ``style`` as PNG at ``dpi`` dots per inch, which the
    file states: the same bytes for the same spec and dpi, with no software tag."""
    with _draw_lock:
        buffer = io.BytesIO()
        figure = _figure(spec, style, statement)
        figure.savefig(buffer, format="png", dpi=dpi, metadata=_PNG_METADATA)
    return buffer.getvalue()


def render_pdf(spec: PlotSpec, *, style: str = DEFAULT_STYLE, statement: bool = False) -> bytes:
    """The spec's figure in ``style`` as PDF: vector, with TrueType (Type 42)
    fonts, and the same bytes for the same spec (no date, creator or producer)."""
    with _draw_lock, matplotlib.rc_context(_PDF_SETTINGS):
        buffer = io.BytesIO()
        _figure(spec, style, statement).savefig(buffer, format="pdf", metadata=_PDF_METADATA)
    return buffer.getvalue()


def save_figure(spec: PlotSpec, path: str, *, dpi: int = 150, statement: bool = True) -> None:
    """Render and write the figure to ``path``, by default with its statement
    under it: a file saved on its own states its statistics."""
    render_figure(spec, statement=statement).savefig(path, dpi=dpi)
