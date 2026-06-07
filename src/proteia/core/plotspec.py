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
    """Everything needed to draw one bar chart, and nothing about styling."""

    title: str
    value_kind: ValueKind
    error_type: ErrorType
    y_label: str
    bars: list[Bar]
    comparisons: list[Significance] = Field(default_factory=list)
    test_name: str | None = None
    test_p: float | None = None


def build_plotspec(
    groups: dict[str, list[float]],
    stats: list[GroupStats],
    test: TestResult,
    *,
    value_kind: ValueKind,
    error_type: ErrorType = ErrorType.SD,
    title: str = "",
    lane_indices: dict[str, list[int]] | None = None,
) -> PlotSpec:
    """Assemble a :class:`PlotSpec` from grouped values and computed statistics.

    ``error_type`` selects which precomputed error to surface — never hardcoded.
    ``lane_indices`` optionally carries provenance (condition -> source lanes).
    Only significant pairwise comparisons (p < 0.05) become brackets.
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

    comparisons = [
        Significance(group_a=pw.group_a, group_b=pw.group_b, p_value=pw.p_value)
        for pw in test.pairwise
        if pw.p_value < 0.05
    ]
    return PlotSpec(
        title=title,
        value_kind=value_kind,
        error_type=error_type,
        y_label=_Y_LABEL[value_kind],
        bars=bars,
        comparisons=comparisons,
        test_name=test.test if test.p_value is not None else None,
        test_p=test.p_value,
    )
