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
from typing import NamedTuple

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
    finite ``test_p``; it covers the bars with n >= 2, ``comparisons`` pair only
    those, and ``test_note`` names the bars it leaves out (``None`` when it covers
    every bar). With no test both are ``None``, there are no ``comparisons``, and
    ``test_note`` says why.
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
TOO_FEW_CONDITIONS = "no test: a test needs two conditions or more"

# The same reasons said of the conditions a test would cover, after the ones with
# too few replicates are named: the values of those are not part of the reason.
_OF_THE_OTHERS = {
    NO_VARIATION: "the values of the other conditions do not vary",
    NO_VARIATION_WITHIN: "the values within each of the other conditions do not vary",
    NO_P_VALUE: "the test of the other conditions gives no p-value",
}

# Values whose spread is at most this fraction of their magnitude do not vary: a
# difference in the last bits (0.1 + 0.2 against 0.3) is rounding, not data.
_ROUNDING = 1e-9


def _constant(values: list[float]) -> bool:
    """Whether ``values`` (at least one) do not vary, up to rounding: their spread
    is within :data:`_ROUNDING` of their largest magnitude, so values that are all
    0 must be exactly 0."""
    return bool(values) and max(values) - min(values) <= _ROUNDING * max(map(abs, values))


class _Verdict(NamedTuple):
    """What a chart shows of its test: the groups the shown test covers (empty
    when the chart shows no test), and the chart's ``test_note``."""

    tested: frozenset[str]
    note: str | None


def _listed(bars: list[Bar]) -> str:
    return ", ".join(repr(bar.label) for bar in bars)


def _too_few(short: list[Bar]) -> str:
    """``'α', 'δ' have fewer than 2 replicates``."""
    return f"{_listed(short)} {'has' if len(short) == 1 else 'have'} fewer than 2 replicates"


def _left_out(short: list[Bar]) -> str:
    """``'α', 'δ' (n = 1) are not in the test``; each group gets its own n when
    their n differ."""
    if len({bar.n for bar in short}) == 1:
        named = f"{_listed(short)} (n = {short[0].n})"
    else:
        named = ", ".join(f"{bar.label!r} (n = {bar.n})" for bar in short)
    return f"{named} {'is' if len(short) == 1 else 'are'} not in the test"


def _test_verdict(bars: list[Bar], test: TestResult) -> _Verdict:
    """Whether the chart shows ``test``, over which bars, and what its note says.

    This is the one place that decides a chart's test. The core
    (:func:`~proteia.core.analyze.compare`) tests only the groups with n >= 2, and
    so does a chart: when at least two groups can be tested, their test is shown
    and the note names the groups it leaves out, so a test over some of the bars
    never passes for one over all of them. With fewer than two such groups there
    is no test, and the note names the groups with too few replicates.

    A test over the testable groups stands only when their values vary and its p
    is finite. When no tested condition's values vary the statistic divides by
    zero: scipy's p is NaN when every value is the same, and 0 or rounding noise
    when only the means differ, so constancy is read from the tested points
    themselves (:func:`_constant`), never from the SD or the p. Groups with n < 2
    are always named; otherwise the core's own note, when it has one, comes
    before ours.
    """
    short = [bar for bar in bars if bar.n < 2]
    testable = [bar for bar in bars if bar.n >= 2]
    untested: frozenset[str] = frozenset()
    if len(testable) < 2:
        if short:
            return _Verdict(untested, f"no test: {_too_few(short)}")
        return _Verdict(untested, test.note or TOO_FEW_CONDITIONS)
    p = test.p_value
    if all(_constant(bar.points) for bar in testable):
        same = _constant([value for bar in testable for value in bar.points])
        reason = NO_VARIATION if same else NO_VARIATION_WITHIN
    elif p is None or not math.isfinite(p):
        reason = NO_P_VALUE
    else:
        tested = frozenset(bar.label for bar in testable)
        return _Verdict(tested, _left_out(short) if short else None)
    if short:
        return _Verdict(untested, f"no test: {_too_few(short)}, and {_OF_THE_OTHERS[reason]}")
    return _Verdict(untested, test.note or reason)


def build_plotspec(
    groups: dict[str, list[float]],
    stats: list[GroupStats],
    test: TestResult,
    *,
    value_kind: ValueKind,
    error_type: ErrorType | str = ErrorType.SD,
    title: str = "",
    lane_indices: dict[str, list[int]] | None = None,
    first_label: str | None = None,
    subtitle: str | None = None,
) -> PlotSpec:
    """Assemble a :class:`PlotSpec` from grouped values and computed statistics.

    ``error_type`` selects which precomputed error to surface — never hardcoded.
    Its raw value (``"SD"``) works too; an unknown value raises ``ValueError``.
    ``lane_indices`` optionally carries provenance (condition -> source lanes).
    ``first_label`` (e.g. the control condition) is moved leftmost, the rest keep
    their order. Only significant pairwise comparisons (p < 0.05) become brackets.

    ``test`` is the core's test of ``groups``, which covers only the groups with
    n >= 2 (:func:`~proteia.core.analyze.compare`). The chart shows it when at
    least two groups have n >= 2, the values of those vary (up to rounding), and
    its p is finite; its brackets then pair only tested groups, and ``test_note``
    names the drawn groups the test leaves out, e.g. ``'50 µM' (n = 1) is not in
    the test``. Otherwise the chart shows no test (no test name, no p, no
    brackets), and ``test_note`` says why: it names the groups with too few
    replicates, and then the reason the other groups are not tested either, or
    else carries the test's own note, or says which reason it was
    (:data:`NO_VARIATION`, :data:`NO_VARIATION_WITHIN`, :data:`NO_P_VALUE`,
    :data:`TOO_FEW_CONDITIONS`). So a spec never holds a NaN p, and never a test
    over some of its bars without naming the rest.
    """
    error_type = ErrorType(error_type)  # before the identity check below
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

    verdict = _test_verdict(bars, test)
    shown = bool(verdict.tested)
    comparisons = [
        Significance(group_a=pw.group_a, group_b=pw.group_b, p_value=pw.p_value)
        for pw in test.pairwise
        if pw.p_value < 0.05 and {pw.group_a, pw.group_b} <= verdict.tested
    ]
    return PlotSpec(
        title=title,
        value_kind=value_kind,
        error_type=error_type,
        y_label=_Y_LABEL[value_kind],
        bars=bars,
        comparisons=comparisons,
        test_name=test.test if shown else None,
        test_p=test.p_value if shown else None,
        subtitle=subtitle,
        test_note=verdict.note,
    )
