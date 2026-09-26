# SPDX-License-Identifier: Apache-2.0
"""A plot is described as pure data, separate from how it is drawn.

``PlotSpec`` is the firewall between the analysis layer and visual presentation:
it says *what* to plot (group means, error, individual points, significance,
provenance) but nothing about *how* (fonts, colours, DPI). Future publication
styling changes only the renderer (:mod:`proteia.viz`), never this spec nor
anything upstream of it.

Each bar keeps ``lane_indices`` parallel to its ``points`` — the provenance link
back to the source lanes. The same link later feeds representative-image cropping
and the tamper-evident audit trail; we keep it now even though nothing consumes
it yet.
"""

from __future__ import annotations

import math
from enum import StrEnum

from pydantic import BaseModel, Field

from proteia.core.analyze import GroupStats, TestResult


class ValueKind(StrEnum):
    RAW = "raw"
    LOADING_NORMALIZED = "loading_normalized"
    FOLD_CHANGE = "fold_change"


class ErrorType(StrEnum):
    SD = "SD"
    SEM = "SEM"


_Y_LABEL = {
    ValueKind.RAW: "Net signal (a.u.)",
    ValueKind.LOADING_NORMALIZED: "Normalized signal (target / loading)",
    ValueKind.FOLD_CHANGE: "Fold change vs control",
}


class Bar(BaseModel):
    """One condition group: a mean, an error, and its individual data points."""

    label: str
    mean: float
    error: float
    n: int
    points: list[float]
    lane_indices: list[int] = Field(default_factory=list)  # provenance: point -> source lane


class Significance(BaseModel):
    """A pairwise comparison bracket between two bars."""

    group_a: str
    group_b: str
    p_value: float

    @property
    def stars(self) -> str:
        p = self.p_value
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"


class PlotSpec(BaseModel):
    """Everything needed to draw one bar chart, and nothing about styling.

    ``subtitle`` names the result set the chart belongs to (e.g. ``All lanes``);
    ``None`` when there is only one set. A test that ran gives ``test_name`` and a
    finite ``test_p`` and no ``test_note``, and it covers every bar; with no test
    both are ``None``, there are no ``comparisons``, and ``test_note`` says why.
    """

    title: str
    value_kind: ValueKind
    error_type: ErrorType
    y_label: str
    bars: list[Bar]
    comparisons: list[Significance] = Field(default_factory=list)
    test_name: str | None = None
    test_p: float | None = None
    subtitle: str | None = None
    test_note: str | None = None


NO_VARIATION = "no test: the values do not vary"
NO_VARIATION_WITHIN = "no test: the values within each condition do not vary"
NO_P_VALUE = "no test: the test gives no p-value"


def _no_test_reason(bars: list[Bar], test: TestResult) -> str | None:
    """Why the chart shows no test, or ``None`` when ``test`` stands for the chart.

    A test stands only when it covers every drawn bar, is defined, and has a
    finite p. The core leaves groups with n < 2 out of its test, so a test over
    the other bars would pass for the chart's own (a Welch's t over two of three
    bars, with no multiple-comparison correction). When no condition's values
    vary the statistic divides by zero: scipy's p is NaN when every value is the
    same, and 0 or rounding noise when only the means differ, so constancy is
    read from the points themselves, never from the SD or the p. The core's own
    note, when it has one, comes first.
    """
    few = [bar.label for bar in bars if bar.n < 2]
    p = test.p_value
    if few:
        listed = ", ".join(repr(label) for label in few)
        verb = "has" if len(few) == 1 else "have"
        reason = f"no test: {listed} {verb} fewer than 2 replicates"
    elif bars and all(len(set(bar.points)) == 1 for bar in bars):
        same = len({value for bar in bars for value in bar.points}) == 1
        reason = NO_VARIATION if same else NO_VARIATION_WITHIN
    elif p is None or not math.isfinite(p):
        reason = NO_P_VALUE
    else:
        return None
    return test.note or reason


def build_plotspec(
    groups: dict[str, list[float]],
    stats: list[GroupStats],
    test: TestResult,
    *,
    value_kind: ValueKind,
    error_type: ErrorType = ErrorType.SD,
    title: str = "",
    lane_indices: dict[str, list[int]] | None = None,
    first_label: str | None = None,
    subtitle: str | None = None,
) -> PlotSpec:
    """Assemble a :class:`PlotSpec` from grouped values and computed statistics.

    ``error_type`` selects which precomputed error to surface — never hardcoded.
    ``lane_indices`` optionally carries provenance (condition -> source lanes).
    ``first_label`` (e.g. the control condition) is moved leftmost, the rest keep
    their order. Only significant pairwise comparisons (p < 0.05) become brackets.

    The chart shows no test (no test name, no p, no brackets) when a drawn group
    has n < 2, when no condition's values vary, or when the p is missing or not
    finite; ``test_note`` then carries the test's own note, or says which of
    these it was (:data:`NO_VARIATION`, :data:`NO_VARIATION_WITHIN`,
    :data:`NO_P_VALUE`, or the groups with too few replicates). So a spec never
    holds a NaN p, and never a test that covers only some of its bars.
    """
    bars: list[Bar] = []
    for gs in stats:
        err = gs.sd if error_type is ErrorType.SD else gs.sem
        bars.append(
            Bar(
                label=gs.label,
                mean=gs.mean,
                error=err,
                n=gs.n,
                points=groups.get(gs.label, []),
                lane_indices=(lane_indices or {}).get(gs.label, []),
            )
        )

    if first_label is not None:
        bars.sort(key=lambda b: b.label != first_label)  # stable: first_label to front

    note = _no_test_reason(bars, test)
    tested = note is None
    comparisons = [
        Significance(group_a=pw.group_a, group_b=pw.group_b, p_value=pw.p_value)
        for pw in (test.pairwise if tested else [])
        if pw.p_value < 0.05
    ]
    return PlotSpec(
        title=title,
        value_kind=value_kind,
        error_type=error_type,
        y_label=_Y_LABEL[value_kind],
        bars=bars,
        comparisons=comparisons,
        test_name=test.test if tested else None,
        test_p=test.p_value if tested else None,
        subtitle=subtitle,
        test_note=note,
    )
