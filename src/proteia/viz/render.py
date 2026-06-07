# SPDX-License-Identifier: Apache-2.0
"""Draw a :class:`~proteia.core.plotspec.PlotSpec` as a bar chart with matplotlib.

A functional, legible draft — bars (mean), error bars, individual lane points,
and significance brackets. Publication styling (fonts, palettes, layout) is a
later phase and lives here when it comes; nothing upstream depends on it.
"""

from __future__ import annotations

from matplotlib.figure import Figure

from proteia.core.plotspec import PlotSpec

_BAR_FACE = "#cbd5e1"
_BAR_EDGE = "#334155"
_POINT = "#0f172a"


def _point_xs(center: float, n: int, spread: float = 0.18) -> list[float]:
    """Deterministic horizontal spread for individual points (no RNG)."""
    if n <= 1:
        return [center]
    step = (2 * spread) / (n - 1)
    return [center - spread + i * step for i in range(n)]


def render_figure(spec: PlotSpec) -> Figure:
    """Render the spec to a matplotlib :class:`Figure` (no global pyplot state)."""
    fig = Figure(figsize=(max(4.0, 1.3 * len(spec.bars) + 1.5), 4.5))
    ax = fig.subplots()

    xs = list(range(len(spec.bars)))
    means = [b.mean for b in spec.bars]
    errs = [b.error for b in spec.bars]

    ax.bar(
        xs, means, yerr=errs, capsize=5, width=0.6,
        color=_BAR_FACE, edgecolor=_BAR_EDGE, linewidth=1.0, zorder=1,
    )
    for x, bar in zip(xs, spec.bars, strict=True):
        for px, val in zip(_point_xs(x, len(bar.points)), bar.points, strict=True):
            ax.plot(px, val, "o", color=_POINT, markersize=4, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b.label}\n(n={b.n})" for b in spec.bars])
    ax.set_ylabel(spec.y_label)
    title = spec.title or "Quantification"
    if spec.test_name and spec.test_p is not None:
        title += f"\n{spec.test_name}: p = {spec.test_p:.3g}"
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)

    _draw_significance(ax, spec)
    fig.tight_layout()
    return fig


def _draw_significance(ax, spec: PlotSpec) -> None:
    """Stack significance brackets above the bars, lowest comparisons first."""
    if not spec.comparisons:
        return
    label_to_x = {b.label: i for i, b in enumerate(spec.bars)}
    tops = [b.mean + b.error for b in spec.bars]
    ceiling = max(tops) if tops else 1.0
    gap = ceiling * 0.08 or 0.08

    level = 1
    for comp in spec.comparisons:
        if comp.group_a not in label_to_x or comp.group_b not in label_to_x:
            continue
        x1, x2 = sorted((label_to_x[comp.group_a], label_to_x[comp.group_b]))
        y = ceiling + gap * level
        ax.plot([x1, x1, x2, x2], [y - gap * 0.3, y, y, y - gap * 0.3], color="#334155", lw=1.0)
        ax.text((x1 + x2) / 2, y, comp.stars, ha="center", va="bottom", fontsize=10)
        level += 1
    ax.set_ylim(top=ceiling + gap * (level + 0.5))


def save_figure(spec: PlotSpec, path: str, *, dpi: int = 150) -> None:
    """Render and write the figure to ``path``."""
    render_figure(spec).savefig(path, dpi=dpi)
