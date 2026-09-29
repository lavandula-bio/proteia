# SPDX-License-Identifier: Apache-2.0
"""Presentation layer: render a :class:`~proteia.core.plotspec.PlotSpec` to a figure.

This is the only place visual styling lives. Everything upstream (analysis,
statistics, the plot spec) is style-free, so publication polish in a later phase
changes only this package. The chart styles are a registry
(:data:`~proteia.viz.styles.CHART_STYLES`).
"""

from proteia.viz.render import render_figure, render_pdf, render_png, render_svg
from proteia.viz.styles import CHART_STYLES, DEFAULT_STYLE, ChartStyle, chart_style

__all__ = [
    "CHART_STYLES",
    "DEFAULT_STYLE",
    "ChartStyle",
    "chart_style",
    "render_figure",
    "render_pdf",
    "render_png",
    "render_svg",
]
