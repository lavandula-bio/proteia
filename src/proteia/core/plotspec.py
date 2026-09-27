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

A chart's statistics are stated in words, in one wording for every renderer and
the web (:func:`statement_lines`): what the marks are, the test that ran (or why
none did), which conditions it covers and which it leaves out, and why. The
statement is the chart's legend text, never part of its title; the title and
the axis titles are fields of their own, with defaults, so a user can edit them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Final, Literal, NamedTuple

from pydantic import BaseModel, Field, model_validator

from proteia.core.analyze import (
    ALPHA,
    TESTS,
    GroupStats,
    TestFamily,
    TestResult,
    min_attainable_p,
)


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
# The x-axis title a chart has until the user gives it another.
DEFAULT_X_LABEL: Final = "Condition"
# Why a plotted condition is not tested, in order of precedence: a replicate not
# detected (below the detection limit), no value at all, or fewer than 2 values.
LeftOut = Literal["not_detected", "no_value", "fewer_than_2"]


def default_y_label(value_kind: ValueKind) -> str:
    """The y-axis title a chart of ``value_kind`` has until the user gives it another."""
    return _Y_LABEL[ValueKind(value_kind)]


class Bar(BaseModel):
    """One plotted condition: a mean, an error, and its individual data points.

    ``mean`` and ``error`` are None when the condition draws no bar: it has no
    value, or a replicate of it was not detected (its detected values are still
    points). ``not_detected_lanes`` has one lane per replicate not detected, its
    first included lane (0-based, ascending), as ``lane_indices`` has each
    point's. The geometric mean and SD factor are what a log-scale test
    compares; None where there is no bar or a value is 0 or below.
    """

    label: str
    mean: float | None
    error: float | None
    n: int  # replicates with a value: the number of points
    points: list[float]
    lane_indices: list[int] = Field(default_factory=list)  # provenance: point -> source lane
    not_detected_lanes: list[int] = Field(default_factory=list)
    geometric_mean: float | None = None
    geometric_sd_factor: float | None = None


class Coverage(BaseModel, frozen=True):
    """Whether a plotted condition is tested, and if not, why (``left_out``).

    ``n`` counts its replicates with a value, ``replicates`` all its replicates
    in the set's included lanes, and ``not_detected`` those not detected.
    ``left_out`` is None for a condition that can be tested, whether or not a
    test ran."""

    label: str
    n: int
    replicates: int
    not_detected: int
    tested: bool
    left_out: LeftOut | None = None


class PairP(BaseModel, frozen=True):
    """One comparison of a chart's test: its adjusted p and its estimate (b vs a)."""

    group_a: str
    group_b: str
    p_value: float
    estimate: float | None = None


class ChartTest(BaseModel, frozen=True):
    """The test a chart shows, as its plan resolved it (:class:`~proteia.core.analyze.TestPlan`).

    ``p_value`` and ``statistic`` are the omnibus test's (named by ``omnibus``)
    or the single comparison's; None for comparisons with a reference among 3
    conditions or more. ``pairwise`` holds every comparison, significant or not.
    ``method`` says how a Mann-Whitney U test's p was computed (``exact``,
    ``exact permutation`` or ``normal approximation``); None for other tests.
    """

    id: str
    name: str
    family: str
    comparisons: str
    scale: str | None  # None for a rank test
    reference: str | None
    design: str = "independent"
    covered: list[str]
    omnibus: str | None = None
    p_value: float | None = None
    statistic: float | None = None
    pairwise: list[PairP]
    adjustment: str
    chosen: dict[str, Literal["auto", "user"]]
    reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    min_attainable_p: float | None = None
    method: str | None = None


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
    """Everything needed to draw one chart, and nothing about styling.

    ``title``, ``x_label`` and ``y_label`` are the chart's titles, with defaults
    a user may replace. ``subtitle`` names the result set the chart belongs to
    (e.g. ``All lanes``); ``None`` when there is only one set.

    ``bars`` holds every plotted condition, in x order, and ``coverage`` says of
    each whether it is tested, and if not why, in the same order. ``test`` is
    set if and only if the chart shows a test: it covers at least two bars,
    every other plotted condition has a ``left_out`` reason, and
    ``comparisons`` (the brackets) pair only covered bars whose adjusted p is
    below :data:`~proteia.core.analyze.ALPHA`. With no test there are no
    brackets and ``test_note`` says why. ``statement`` is the chart's legend
    text (:func:`statement_lines`), and ``rank_min_p`` the smallest p a rank
    test could give here, by comparisons (``all_pairs``, ``vs_reference``).

    ``test_name`` (the test's id), ``test_p`` (the omnibus or the single
    comparison's p) and ``test_note`` (why there is no test, or which
    conditions the test leaves out) are the fields charts had before ``test``
    and ``statement``; they are kept, derived, for the napari chart.
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
    x_label: str = DEFAULT_X_LABEL
    test: ChartTest | None = None
    coverage: list[Coverage] = Field(default_factory=list)
    statement: list[str] = Field(default_factory=list)
    rank_min_p: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coverage_matches_the_bars(self) -> PlotSpec:
        if self.coverage:
            if [c.label for c in self.coverage] != [b.label for b in self.bars]:
                raise ValueError("coverage and bars must name the same conditions, in order")
            for c, bar in zip(self.coverage, self.bars, strict=True):
                if c.not_detected != len(bar.not_detected_lanes) or c.n != bar.n:
                    raise ValueError(f"the coverage of {c.label!r} does not match its bar")
        return self


NO_VARIATION = "no test: the values do not vary"
NO_VARIATION_WITHIN = "no test: the values within each condition do not vary"
NO_P_VALUE = "no test: the test gives no p-value"
TOO_FEW_CONDITIONS = "no test: a test needs two conditions or more"

# The same reasons said of the conditions a test would cover, after the ones left
# out are named: the values of those are not part of the reason.
_OF_THE_OTHERS = {
    NO_VARIATION: "the values of the other conditions do not vary",
    NO_VARIATION_WITHIN: "the values within each of the other conditions do not vary",
    NO_P_VALUE: "the test of the other conditions gives no p-value",
}

# Values whose spread is at most this fraction of their magnitude do not vary: a
# difference in the last bits (0.1 + 0.2 against 0.3) is rounding, not data.
_ROUNDING = 1e-9


def p_text(p: float) -> str:
    """A p-value as charts and the page's captions write it: below 0.0001 as a
    bound, otherwise to 3 significant digits."""
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.3g}"


def _constant(values: list[float]) -> bool:
    """Whether ``values`` (at least one) do not vary, up to rounding: their spread
    is within :data:`_ROUNDING` of their largest magnitude, so values that are all
    0 must be exactly 0."""
    return bool(values) and max(values) - min(values) <= _ROUNDING * max(map(abs, values))


def _why(c: Coverage) -> str:
    if c.left_out == "not_detected":
        return f"{c.not_detected} of {c.replicates} not detected"
    if c.left_out == "no_value":
        return "no value"
    return f"n = {c.n}"


def left_out_text(coverage: Sequence[Coverage]) -> str:
    """The conditions of ``coverage`` left out of the test, each with why, in x
    order: ``'KO' (3 of 3 not detected), '50 µM' (n = 1)``; conditions that share
    a reason share it once: ``'α', 'δ' (n = 1)``. The one wording of the chart's
    statement, its note and the ``conditions_not_tested`` notice."""
    out = [c for c in coverage if c.left_out is not None]
    whys = [_why(c) for c in out]
    if len(set(whys)) == 1:
        return f"{', '.join(repr(c.label) for c in out)} ({whys[0]})"
    return ", ".join(f"{c.label!r} ({why})" for c, why in zip(out, whys, strict=True))


def not_in_the_test(coverage: Sequence[Coverage]) -> str:
    """``'50 µM' (n = 1) is not in the test``: the conditions a test leaves out."""
    count = sum(c.left_out is not None for c in coverage)
    return f"{left_out_text(coverage)} {'is' if count == 1 else 'are'} not in the test"


def _too_few(short: list[Coverage]) -> str:
    """``'α', 'δ' have fewer than 2 replicates``."""
    listed = ", ".join(repr(c.label) for c in short)
    return f"{listed} {'has' if len(short) == 1 else 'have'} fewer than 2 replicates"


def marks_line(spec: PlotSpec) -> str:
    """What the bar style's marks are: ``Bars: mean ± SD; points: replicates``,
    and the open circles when a replicate was not detected."""
    line = f"Bars: mean ± {spec.error_type.value}; points: replicates"
    if any(bar.not_detected_lanes for bar in spec.bars):
        line += "; open circles: not detected"
    return line


def describe_test(test: ChartTest) -> str:
    """The test that ran, on which values, and its p: ``Student's t-test on log
    values: p = 0.012``, ``One-way ANOVA + Tukey-Kramer on log values: ANOVA p =
    0.00014``; comparisons with the reference have no omnibus p, so each one's
    is given: ``Dunnett's test on log values, each condition vs 'vehicle':
    '10 µM' p = 0.0029, '50 µM' p = 0.0113``. A Mann-Whitney U test computed by
    the normal approximation says so."""
    name = test.name
    if test.method == "normal approximation":
        name += " (normal approximation)"
    if test.scale == "log":
        name += " on log values"
    if test.comparisons == "vs_reference" and len(test.covered) > 2:
        each = ", ".join(f"{pair.group_b!r} {p_text(pair.p_value)}" for pair in test.pairwise)
        return f"{name}, each condition vs {test.reference!r}: {each}"
    if test.p_value is None:
        return name
    if test.omnibus is not None:
        return f"{name}: {TESTS[test.id].omnibus_short} {p_text(test.p_value)}"
    return f"{name}: {p_text(test.p_value)}"


def coverage_line(coverage: Sequence[Coverage]) -> str:
    """``Tested: all 3 conditions``, or ``Not tested: 'KO' (3 of 3 not
    detected), '50 µM' (n = 1)``, which names only the conditions left out."""
    if all(c.tested for c in coverage):
        return (
            "Tested: both conditions"
            if len(coverage) == 2
            else f"Tested: all {len(coverage)} conditions"
        )
    return f"Not tested: {left_out_text(coverage)}"


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]


def statement_lines(spec: PlotSpec) -> list[str]:
    """The chart's legend text, one line each: what the marks are
    (:func:`marks_line`); the test that ran (:func:`describe_test`) and which
    conditions it covers (:func:`coverage_line`), then its notes; or, with no
    test, why there is none."""
    lines = [marks_line(spec)]
    if spec.test is None:
        if spec.test_note:
            lines.append(_sentence_case(spec.test_note))
        return lines
    return [*lines, describe_test(spec.test), coverage_line(spec.coverage), *spec.test.notes]


class _Verdict(NamedTuple):
    """What a chart shows of its test: the groups the shown test covers (empty
    when the chart shows no test), and the chart's ``test_note``."""

    tested: frozenset[str]
    note: str | None


def _has_p(test: TestResult) -> bool:
    """Whether ``test`` gives every p its kind gives, each finite: its omnibus
    (or single) p, unless it compares each condition with a reference, and every
    pairwise p."""
    spec = TESTS[test.test]
    needs_p = spec.omnibus is not None or spec.conditions == "two"
    ps = [p.p_value for p in test.pairwise]
    if needs_p:
        if test.p_value is None:
            return False
        ps.append(test.p_value)
    return bool(ps) and all(math.isfinite(p) for p in ps)


def _unweighable(test: TestResult, covered: list[Bar]) -> str | None:
    """Why Welch's test of 3 conditions or more has no p here, or None.

    Welch's tests take each condition's own variance, and a condition whose
    values do not vary (:func:`_constant`) has none: Welch's ANOVA weighs each
    condition by the inverse of its variance, and Welch's t-test of two such
    conditions has no standard error. scipy then gives NaN, or 0 and three stars
    when the values differ by rounding, so this is read from the covered points,
    as the constancy of them all is. Welch's t-test of one such condition with
    one that varies is sound; Welch's t of two conditions is ruled by the
    constancy of them all."""
    spec = TESTS[test.test]
    if spec.family is not TestFamily.WELCH or spec.conditions == "two":
        return None
    flat = [bar.label for bar in covered if _constant(bar.points)]
    if spec.omnibus is not None:
        named = flat
        why = "so Welch's ANOVA gives no p-value"
    else:
        pairs = [p for p in test.pairwise if p.group_a in flat and p.group_b in flat]
        named = [label for label in flat if any(label in (p.group_a, p.group_b) for p in pairs)]
        why = "so Welch's t-test gives no p-value between them"
    if not named:
        return None
    return f"the values within {', '.join(map(repr, named))} do not vary, {why}"


def _test_verdict(bars: list[Bar], coverage: list[Coverage], test: TestResult) -> _Verdict:
    """Whether the chart shows ``test``, over which bars, and what its note says.

    The plan (:func:`~proteia.core.analyze.plan_test`) decides which conditions
    are tested; ``coverage`` says why the others are not. A planned test stands
    only when the values of the conditions it covers vary and its p-values are
    finite. When none of the covered conditions' values vary the statistic
    divides by zero: scipy's p is NaN when every value is the same, and 0 or
    rounding noise when only the means differ, so constancy is read from the
    covered points themselves (:func:`_constant`), never from the SD or the p.
    Welch's tests of 3 conditions or more also need the values of some of the
    conditions to vary (:func:`_unweighable`). The conditions left out are
    always named: when each has too few replicates, in the words charts have
    used since #82.
    """
    plan = test.plan
    none: frozenset[str] = frozenset()
    out = [c for c in coverage if c.left_out is not None]
    only_short = all(c.left_out == "fewer_than_2" for c in out)

    def naming_the_rest(reason: str, of_the_others: str | None = None) -> str:
        """``no test: <reason>``, naming the conditions left out, with
        ``of_the_others`` the reason as said of the conditions besides them."""
        if not out:
            return f"no test: {reason}"
        other = of_the_others or reason
        if only_short:
            return f"no test: {_too_few(out)}, and {other}"
        return f"no test: {other}; not tested: {left_out_text(out)}"

    def with_the_rest(note: str) -> str:
        return naming_the_rest(note.removeprefix("no test: "), _OF_THE_OTHERS[note])

    if plan.test is None:
        if plan.no_test != "too_few":
            return _Verdict(none, plan.no_test_note)
        if not out:
            return _Verdict(none, TOO_FEW_CONDITIONS)
        if only_short:
            return _Verdict(none, f"no test: {_too_few(out)}")
        return _Verdict(
            none, f"no test: fewer than 2 conditions to test; not tested: {left_out_text(out)}"
        )
    covered = [bar for bar in bars if bar.label in plan.covered]
    if all(_constant(bar.points) for bar in covered):
        same = _constant([value for bar in covered for value in bar.points])
        return _Verdict(none, with_the_rest(NO_VARIATION if same else NO_VARIATION_WITHIN))
    unweighable = _unweighable(test, covered)
    if unweighable is not None:
        return _Verdict(none, naming_the_rest(unweighable))
    if not _has_p(test):
        return _Verdict(none, with_the_rest(NO_P_VALUE))
    return _Verdict(frozenset(plan.covered), not_in_the_test(out) if out else None)


def _left_out(bar: Bar, excluded: set[str]) -> LeftOut | None:
    if bar.not_detected_lanes:
        return "not_detected"
    if bar.n == 0:
        return "no_value"
    if bar.label in excluded:
        return "fewer_than_2"
    return None


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _chart_test(test: TestResult) -> ChartTest:
    plan = test.plan
    return ChartTest(
        id=test.test,
        name=plan.name,
        family=plan.family.value,
        comparisons=plan.comparisons.value,
        scale=None if plan.scale is None else plan.scale.value,
        reference=plan.reference,
        design=plan.design.value,
        covered=list(plan.covered),
        omnibus=test.omnibus,
        p_value=test.p_value,
        statistic=_finite(test.statistic),
        pairwise=[
            PairP(
                group_a=p.group_a,
                group_b=p.group_b,
                p_value=p.p_value,
                estimate=_finite(p.estimate),
            )
            for p in test.pairwise
        ],
        adjustment=TESTS[test.test].adjustment,
        chosen=dict(plan.chosen),
        reasons=list(plan.reasons),
        notes=list(plan.notes),
        min_attainable_p=plan.min_attainable_p,
        method=test.method,
    )


def _rank_floors(coverage: list[Coverage], reference: str | None) -> dict[str, float]:
    """The smallest p a rank test could give over the conditions that can be
    tested, by comparisons: for the web's menu, which offers rank tests."""
    testable = [c for c in coverage if c.left_out is None]
    k = len(testable)
    if k < 2:
        return {}
    sizes = [c.n for c in testable]
    if k == 2:
        floors = {"all_pairs": min_attainable_p(sizes, "mann_whitney", 1)}
    else:
        floors = {"all_pairs": min_attainable_p(sizes, "kruskal_dunn_holm", k * (k - 1) // 2)}
    labels = [c.label for c in testable]
    if reference in labels:
        first = labels.index(reference)
        ordered = [sizes[first], *sizes[:first], *sizes[first + 1 :]]
        test = "mann_whitney" if k == 2 else "dunn_holm"
        floors["vs_reference"] = min_attainable_p(ordered, test, k - 1)
    return floors


def build_plotspec(
    groups: Mapping[str, Sequence[float]],
    stats: list[GroupStats],
    test: TestResult,
    *,
    value_kind: ValueKind,
    error_type: ErrorType | str = ErrorType.SD,
    title: str = "",
    lane_indices: Mapping[str, list[int]] | None = None,
    first_label: str | None = None,
    subtitle: str | None = None,
    replicates: Mapping[str, int] | None = None,
    not_detected_lanes: Mapping[str, list[int]] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> PlotSpec:
    """Assemble a :class:`PlotSpec` from grouped values and computed statistics.

    ``error_type`` selects which precomputed error to surface — never hardcoded.
    Its raw value (``"SD"``) works too; an unknown value raises ``ValueError``.
    ``lane_indices`` optionally carries provenance (condition -> source lanes).
    ``first_label`` (e.g. the reference condition) is moved leftmost, the rest
    keep their order. ``x_label`` and ``y_label`` default to
    :data:`DEFAULT_X_LABEL` and the value kind's (:func:`default_y_label`).

    ``replicates`` gives every plotted condition (in x order) its number of
    replicates, including a condition with no value, which then keeps its slot
    with no bar; without it (the napari chart) the plotted conditions are those
    of ``stats``. ``not_detected_lanes`` gives the first lane of each replicate
    not detected: such a condition draws no bar and is not tested (its detected
    values are still points).

    ``test`` is the core's test (:func:`~proteia.core.analyze.compare`) of the
    plotted conditions less those with a replicate not detected; its plan
    decides which conditions it covers. The chart shows it when the covered
    values vary (up to rounding) and its p-values are finite: its brackets then
    pair only covered conditions whose adjusted p is below
    :data:`~proteia.core.analyze.ALPHA`, and the conditions it leaves out are
    named. Otherwise the chart shows no test (no ``test``, no p, no brackets)
    and ``test_note`` says why (:data:`NO_VARIATION`,
    :data:`NO_VARIATION_WITHIN`, :data:`NO_P_VALUE`,
    :data:`TOO_FEW_CONDITIONS`, or the plan's own note), naming the conditions
    left out. So a spec never holds a NaN p, and never a test over some of its
    bars without naming the rest. ``ValueError`` for a test with no plan, or
    one that covers a condition the chart cannot test.
    """
    error_type = ErrorType(error_type)  # before the identity check below
    plan = test.plan
    if plan is None:
        raise ValueError("the test has no plan: run it with analyze.compare or analyze.run_test")
    by_label = {gs.label: gs for gs in stats}
    order = list(replicates) if replicates is not None else list(by_label)
    missing = [label for label in (not_detected_lanes or {}) if label not in order]
    if missing:
        raise ValueError(f"not-detected lanes for conditions not plotted: {missing}")
    bars: list[Bar] = []
    for label in order:
        points = list(groups.get(label, []))
        undetected = sorted((not_detected_lanes or {}).get(label, []))
        gs = by_label.get(label)
        drawn = gs is not None and bool(points) and not undetected
        bars.append(
            Bar(
                label=label,
                mean=gs.mean if drawn else None,
                error=(gs.sd if error_type is ErrorType.SD else gs.sem) if drawn else None,
                n=len(points),
                points=points,
                lane_indices=(lane_indices or {}).get(label, []),
                not_detected_lanes=undetected,
                geometric_mean=gs.geometric_mean if drawn else None,
                geometric_sd_factor=gs.geometric_sd_factor if drawn else None,
            )
        )
    if first_label is not None:
        bars.sort(key=lambda b: b.label != first_label)  # stable: first_label to front

    excluded = {group.condition for group in plan.excluded}
    left = {bar.label: _left_out(bar, excluded) for bar in bars}
    testable = {label for label, why in left.items() if why is None}
    if plan.test is not None and set(plan.covered) != testable:
        raise ValueError(
            f"the test covers {sorted(plan.covered)}, but the chart can test {sorted(testable)}"
        )

    def coverage_of(tested: frozenset[str]) -> list[Coverage]:
        return [
            Coverage(
                label=bar.label,
                n=bar.n,
                replicates=(replicates or {}).get(bar.label, bar.n + len(bar.not_detected_lanes)),
                not_detected=len(bar.not_detected_lanes),
                tested=bar.label in tested,
                left_out=left[bar.label],
            )
            for bar in bars
        ]

    verdict = _test_verdict(bars, coverage_of(frozenset()), test)
    shown = bool(verdict.tested)
    coverage = coverage_of(verdict.tested)
    comparisons = [
        Significance(group_a=pw.group_a, group_b=pw.group_b, p_value=pw.p_value)
        for pw in test.pairwise
        if shown and pw.p_value < ALPHA and {pw.group_a, pw.group_b} <= verdict.tested
    ]
    spec = PlotSpec(
        title=title,
        value_kind=value_kind,
        error_type=error_type,
        y_label=default_y_label(value_kind) if y_label is None else y_label,
        x_label=DEFAULT_X_LABEL if x_label is None else x_label,
        bars=bars,
        comparisons=comparisons,
        test_name=test.test if shown else None,
        test_p=test.p_value if shown else None,
        subtitle=subtitle,
        test_note=verdict.note,
        test=_chart_test(test) if shown else None,
        coverage=coverage,
        rank_min_p=_rank_floors(coverage, first_label),
    )
    return spec.model_copy(update={"statement": statement_lines(spec)})
