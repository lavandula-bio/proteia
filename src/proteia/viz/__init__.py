# SPDX-License-Identifier: Apache-2.0
"""Presentation layer: render a :class:`~proteia.core.plotspec.PlotSpec` to a figure.

This is the only place visual styling lives. Everything upstream (analysis,
statistics, the plot spec) is style-free, so publication polish in a later phase
changes only this package.
"""

from proteia.viz.render import render_figure, save_figure

__all__ = ["render_figure", "save_figure"]
