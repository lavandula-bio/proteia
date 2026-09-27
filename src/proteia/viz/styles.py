# SPDX-License-Identifier: Apache-2.0
"""The chart styles a :class:`~proteia.core.plotspec.PlotSpec` can be drawn in.

One registry with stable ids, which clients, exports and the record use: a new
style is added here, under an id of its own, and the web builds its menu from
:data:`CHART_STYLES` (never a list of its own). A style says whether it can draw
a spec (``supports``: None, or the reason it cannot), what its marks are (the
first line of the chart's statement) and draws it.

Only ``bar`` is built: bars (mean ± SD or SEM) with each replicate as a point,
and an n.d. row for replicates not detected (:mod:`proteia.viz.render`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from proteia.core.plotspec import PlotSpec, marks_line

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class ChartStyle:
    """One chart style. ``render(spec, statement=...)`` draws ``spec``, with its
    statement under it when asked; ``marks_line(spec)`` says what its marks are."""

    id: str  # stable: clients, exports and the record use it
    name: str  # the menu's text
    supports: Callable[[PlotSpec], str | None]  # None if it can draw the spec, else why not
    marks_line: Callable[[PlotSpec], str]
    render: Callable[..., Figure]


def _draw_bars(spec: PlotSpec, *, statement: bool = False) -> Figure:
    from proteia.viz import render  # the renderer draws through this registry

    return render.render_figure(spec, statement=statement)


def _any_spec(spec: PlotSpec) -> str | None:
    return None


BAR: Final = ChartStyle(
    id="bar",
    name="Bars with points",
    supports=_any_spec,
    marks_line=marks_line,
    render=_draw_bars,
)
CHART_STYLES: Final[Mapping[str, ChartStyle]] = MappingProxyType({BAR.id: BAR})
DEFAULT_STYLE: Final = BAR.id


def chart_style(style_id: str) -> ChartStyle:
    """The style ``style_id`` names; ``KeyError`` for an id there is no style of."""
    try:
        return CHART_STYLES[style_id]
    except KeyError:
        known = ", ".join(repr(key) for key in CHART_STYLES)
        raise KeyError(f"no chart style {style_id!r}; the styles are {known}") from None


def legend_lines(spec: PlotSpec, style_id: str = DEFAULT_STYLE) -> list[str]:
    """The legend text of ``spec`` drawn in ``style_id``: what the style's marks
    are, then the statistics of the spec's statement."""
    return [chart_style(style_id).marks_line(spec), *spec.statement[1:]]
