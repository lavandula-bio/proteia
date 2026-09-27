# SPDX-License-Identifier: Apache-2.0
"""Draw a :class:`~proteia.core.plotspec.PlotSpec` as a bar chart with matplotlib.

A functional, legible draft — bars (mean), error bars, individual lane points,
and significance brackets. Publication styling (fonts, palettes, layout) is a
later phase and lives here when it comes; nothing upstream depends on it.

The drawn marks carry ids that name what they show, so a drawing can be read
back against its spec: ``bar-i`` is ``spec.bars[i]``, ``point-i-j`` is
``spec.bars[i].points[j]`` and ``bracket-k`` is ``spec.comparisons[k]``. Only an
SVG shows them (:func:`render_svg`).

A name is drawn as typed: matplotlib's math syntax (``$...$``) is not read, so a
``$`` or ``\\`` in a name is drawn and no name can fail to draw, and a character
XML cannot hold is drawn as U+FFFD (:func:`_shown`).
"""

from __future__ import annotations

import io
import re
import threading
from collections.abc import Callable
from typing import Final

import matplotlib
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure

from proteia.core.plotspec import PlotSpec

_BAR_FACE = "#cbd5e1"
_BAR_EDGE = "#334155"
_POINT = "#0f172a"
_TITLE_MARGIN_PT = 4.0  # the gap a title line keeps from the figure's edges
# The core's tests as a reader names them; another is named by its id, spaced.
_TEST_NAMES: Final = {"welch_t": "Welch t-test", "anova_oneway": "One-way ANOVA"}
# Any character outside XML 1.0's Char production.
_NOT_XML: Final = re.compile("[^\t\n\r\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")

# An SVG the same bytes for the same spec: ids from a fixed salt, not at random,
# and glyphs drawn as paths, so the drawing needs no font of the viewer's.
_SVG_SETTINGS: Final = {"svg.hashsalt": "proteia-chart", "svg.fonttype": "path"}
# No metadata block: no date, and none of the URLs matplotlib's metadata names.
_SVG_METADATA: Final = {"Date": None, "Creator": None, "Type": None, "Format": None}
# rc_context changes matplotlib's global settings for as long as it lasts, so
# drawings made at once would see each other's. One at a time: drawing is
# CPU-bound under the GIL, so threads would not draw faster anyway.
_svg_lock = threading.Lock()


def _shown(text: str) -> str:
    """``text`` as a chart shows it: each character XML cannot hold (a control
    character other than tab, newline and carriage return, an unpaired surrogate,
    U+FFFE or U+FFFF) as the replacement character U+FFFD. A typed name may hold
    U+FFFE or U+FFFF, matplotlib writes every text into an SVG as it is (in a
    comment beside its glyphs), and a browser shows no SVG that is not well-formed
    XML."""
    return _NOT_XML.sub("\ufffd", text)


def _point_xs(center: float, n: int, spread: float = 0.18) -> list[float]:
    """Deterministic horizontal spread for individual points (no RNG)."""
    if n == 0:
        return []
    if n == 1:
        return [center]
    step = (2 * spread) / (n - 1)
    return [center - spread + i * step for i in range(n)]


def render_figure(spec: PlotSpec) -> Figure:
    """Render the spec to a matplotlib :class:`Figure` (no global pyplot state).

    The title's lines: the spec's title, its subtitle (the result set) when it has
    one, the test (by the name a reader knows it by) and its p when a test ran,
    and the spec's note when it has one:
    why no test ran, or which groups the test that ran leaves out. So a chart
    never drops its test, or a group from it, without a word. A line too wide
    for the figure is broken over lines (:func:`_fit_title`), so a saved figure
    never crops it.
    """
    fig = Figure(figsize=(max(4.0, 1.3 * len(spec.bars) + 1.5), 4.5))
    ax = fig.subplots()

    xs = list(range(len(spec.bars)))
    means = [b.mean for b in spec.bars]
    errs = [b.error for b in spec.bars]

    bars = ax.bar(
        xs,
        means,
        yerr=errs,
        capsize=5,
        width=0.6,
        color=_BAR_FACE,
        edgecolor=_BAR_EDGE,
        linewidth=1.0,
        zorder=1,
    )
    for i, patch in enumerate(bars.patches):
        patch.set_gid(f"bar-{i}")
    for i, (x, bar) in enumerate(zip(xs, spec.bars, strict=True)):
        for j, (px, val) in enumerate(zip(_point_xs(x, len(bar.points)), bar.points, strict=True)):
            ax.plot(px, val, "o", color=_POINT, markersize=4, zorder=3, gid=f"point-{i}-{j}")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{_shown(b.label)}\n(n={b.n})" for b in spec.bars], parse_math=False)
    ax.set_ylabel(_shown(spec.y_label), parse_math=False)
    lines = [spec.title or "Quantification"]
    if spec.subtitle:
        lines.append(spec.subtitle)
    if spec.test_name and spec.test_p is not None:
        name = _TEST_NAMES.get(spec.test_name, spec.test_name.replace("_", " "))
        lines.append(f"{name}: p = {spec.test_p:.3g}")
    if spec.test_note:
        lines.append(spec.test_note)
    lines = [_shown(line) for line in lines]
    ax.set_title("\n".join(lines), parse_math=False)
    ax.spines[["top", "right"]].set_visible(False)

    _draw_significance(ax, spec)
    fig.tight_layout()
    _fit_title(fig, ax, lines)
    return fig


def _fit_title(fig: Figure, ax, lines: list[str]) -> None:
    """Break each title line too wide for the figure at its spaces, and lay out again.

    The title is centred on the axes, so a line fits when it is no wider than
    twice the distance from the axes' centre to the nearer figure edge, less a
    small margin. tight_layout places the axes by the title's height alone,
    never its width, so that distance is known once the figure is laid out. A
    taller title can shift the axes a little (their tick labels may change), so
    the fit is checked again after each layout, a few times at most.
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
        fig.tight_layout()


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
    """Stack significance brackets above the bars, lowest comparisons first."""
    if not spec.comparisons:
        return
    label_to_x = {b.label: i for i, b in enumerate(spec.bars)}
    tops = [b.mean + b.error for b in spec.bars]
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


def render_svg(spec: PlotSpec) -> bytes:
    """The spec's figure (:func:`render_figure`) as SVG: the same bytes for the
    same spec, with the drawn marks' ids (see the module docstring).

    Nothing in it runs or loads anything: matplotlib writes no script, no event
    handler and no link, every reference (a clip path, a glyph) points into the
    SVG itself, and there is no metadata. The glyphs are paths, so there is no
    text element and no font to load.
    """
    with _svg_lock, matplotlib.rc_context(_SVG_SETTINGS):
        buffer = io.BytesIO()
        render_figure(spec).savefig(buffer, format="svg", metadata=_SVG_METADATA)
    return buffer.getvalue()


def save_figure(spec: PlotSpec, path: str, *, dpi: int = 150) -> None:
    """Render and write the figure to ``path``."""
    render_figure(spec).savefig(path, dpi=dpi)
