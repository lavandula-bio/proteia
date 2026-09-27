# SPDX-License-Identifier: Apache-2.0
"""Batch-level analysis: normalize, group by condition, and run statistics.

GUI-independent. The atomic unit upstream is a :class:`~proteia.core.model.Protein`
(one protein on one image). A *batch* is one experiment: several proteins sharing
the same lane spine, where one protein is the loading control. This module turns a
batch into per-condition value groups and descriptive/inferential statistics.

The "computability ladder" (decided 2026-06-07) gates what a batch can produce:

* only a target (or only a loading control) -> *export only*: raw values, no
  normalization, no statistics, no pooling.
* target + loading control -> *loading-normalized* values per lane, grouped by
  condition, with statistics; poolable across batches.
* + a designated control condition -> additionally *fold-change* vs that group.

Replicates are lanes sharing one condition label. Normalization is per lane (per
sample, joined by lane position/index); pooling into replicate lists is per
condition. Normalize first, then group.

No value type is baked in: a group of values may be raw nets, loading-normalized
ratios, or fold-changes. Plotting treats them the same; the statistics ask only
whether they are ratios, which the automatic rule tests on log values.

The tests are a registry with stable ids (:data:`TESTS`); which one a chart runs
is planned from the design of the data and the user's statistics setting
(:func:`plan_test`), and run by :func:`run_test` (:func:`compare` does both).
"""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ValidationError
from scipy import stats

from proteia.core.model import Role

# A protein's net signal per lane, aligned to the shared lane spine. ``None`` marks
# a lane where this protein has no box (a gap). Length == number of lanes.
LaneNets = list[float | None]


class Tier(StrEnum):
    """How far up the computability ladder a batch can go."""

    EXPORT_ONLY = "export_only"  # missing target or loading control
    NORMALIZED = "normalized"  # target + loading control present
    FOLD_CHANGE = "fold_change"  # + a designated control condition


@dataclass(frozen=True)
class ProteinNets:
    """One protein's per-lane nets within a batch, plus its role.

    For a target, ``loadings`` names the loading control(s) it is normalized
    against (by name). Empty means "use the batch default" — the single loading
    control if there is exactly one. A target may name several loading controls
    (a deliberate comparison), yielding one normalized series per pairing.
    """

    name: str
    role: Role
    nets: LaneNets
    loadings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Batch:
    """One experiment: proteins sharing a lane spine of condition labels.

    ``conditions`` gives the condition label of each lane (repeats == replicates);
    every protein's ``nets`` is aligned to it by index. ``control_condition`` names
    the reference group for fold-change, if any.

    The stored form of a run is :class:`proteia.core.model.Batch`; the compute step
    builds this statistics input from it.
    """

    conditions: list[str]
    proteins: list[ProteinNets]
    control_condition: str | None = None

    def __post_init__(self) -> None:
        n = len(self.conditions)
        for p in self.proteins:
            if len(p.nets) != n:
                raise ValueError(
                    f"protein {p.name!r} has {len(p.nets)} nets but there are {n} lanes"
                )
        if self.control_condition is not None and self.control_condition not in self.conditions:
            raise ValueError(f"control condition {self.control_condition!r} is not a lane label")

    def targets(self) -> list[ProteinNets]:
        return [p for p in self.proteins if p.role is Role.TARGET]

    def loading_controls(self) -> list[ProteinNets]:
        return [p for p in self.proteins if p.role is Role.LOADING_CONTROL]

    def loading_control(self) -> ProteinNets | None:
        """The first loading control (convenience for the common single-control
        batch). Per-target resolution should use :meth:`resolve_loadings`."""
        lcs = self.loading_controls()
        return lcs[0] if lcs else None

    def resolve_loadings(self, target: ProteinNets) -> list[ProteinNets]:
        """Which loading control(s) a target is normalized against.

        Explicit ``target.loadings`` win (each must name an existing loading
        control). With none given, default to the batch's single loading control.
        Returns ``[]`` when unresolvable: no loading control at all, an ambiguous
        choice (2+ controls and no explicit pick), or a named control that does
        not exist / is not a loading control.
        """
        lcs = self.loading_controls()
        by_name = {p.name: p for p in lcs}
        if target.loadings:
            return [by_name[name] for name in target.loadings if name in by_name]
        return [lcs[0]] if len(lcs) == 1 else []


@dataclass(frozen=True)
class Compliance:
    """The ladder tier a batch reaches, plus any non-compliance warnings."""

    tier: Tier
    warnings: list[str] = field(default_factory=list)

    @property
    def can_pool(self) -> bool:
        return self.tier in (Tier.NORMALIZED, Tier.FOLD_CHANGE)


def assess(batch: Batch) -> Compliance:
    """Decide the computability tier and collect non-compliance warnings.

    Tolerant by design: a batch that cannot be normalized is still usable for
    raw export; we flag the shortfall rather than refuse it.
    """
    warnings: list[str] = []
    has_target = bool(batch.targets())
    has_loading = batch.loading_control() is not None

    if not has_target:
        warnings.append("no target protein: export only")
    if not has_loading:
        warnings.append("no loading control: cannot normalize; export only")
    if not (has_target and has_loading):
        return Compliance(Tier.EXPORT_ONLY, warnings)

    if batch.control_condition is None:
        return Compliance(Tier.NORMALIZED, warnings)
    return Compliance(Tier.FOLD_CHANGE, warnings)


def normalize_lane(target: LaneNets, loading: LaneNets) -> LaneNets:
    """Per-lane loading normalization: target net / loading-control net.

    A lane is ``None`` in the result if either net is missing or the loading
    signal is non-positive (cannot divide).
    """
    out: LaneNets = []
    for t, lo in zip(target, loading, strict=True):
        if t is None or lo is None or lo <= 0:
            out.append(None)
        else:
            out.append(t / lo)
    return out


@dataclass(frozen=True)
class NormalizedSeries:
    """One target normalized against one loading control: per-lane ratio.

    ``target``/``loading`` are protein names; ``values`` is the per-lane
    target/loading ratio (``None`` where either is missing). One series per
    (target, loading) pairing.
    """

    target: str
    loading: str
    values: LaneNets


def normalize_batch(batch: Batch) -> tuple[list[NormalizedSeries], list[str]]:
    """Exhaustive normalization: every target against each of its loading
    control(s), for all lanes — independent of which conditions get plotted later.

    Returns one :class:`NormalizedSeries` per (target, loading) pairing, plus
    warnings naming targets that could not be resolved to a loading control. This
    is the analysis layer's "compute everything" step; selecting a target /
    loading / condition subset to chart happens downstream.
    """
    series: list[NormalizedSeries] = []
    warnings: list[str] = []
    for t in batch.targets():
        loadings = batch.resolve_loadings(t)
        if not loadings:
            warnings.append(f"target {t.name!r} has no resolvable loading control")
            continue
        for lc in loadings:
            series.append(NormalizedSeries(t.name, lc.name, normalize_lane(t.nets, lc.nets)))
    return series, warnings


class ReduceMethod(StrEnum):
    MEAN = "mean"  # average technical repeats (force-merge; the safe default)
    REPRESENTATIVE = "representative"  # keep one repeat per sample, drop the rest


@dataclass(frozen=True)
class SampleReduction:
    """Per-condition lists of *sample* values, after collapsing technical repeats.

    ``groups`` feeds the statistics: each value is one biological sample, so its
    length is the correct n. ``lanes`` is parallel to ``groups``: the lanes behind
    each value (several for collapsed technical repeats; the first is the one a
    representative reduction keeps). ``averaged`` lists the ``(condition, sample)``
    keys that had more than one lane (i.e. were collapsed), for transparency.
    """

    groups: dict[str, list[float]]
    averaged: list[tuple[str, str]]
    warnings: list[str] = field(default_factory=list)
    lanes: dict[str, list[list[int]]] = field(default_factory=dict)


def repeats_message(count: int, method: ReduceMethod | str) -> str:
    """How :func:`reduce_samples` reports ``count`` samples with technical repeats.
    ``method`` may be its raw value (``"mean"``)."""
    what = "averaged" if ReduceMethod(method) is ReduceMethod.MEAN else "kept one lane of"
    return f"{what} {count} sample(s) with technical repeats (repeats do not count as n)"


def _replicate_key(condition: str, sample: str | None, lane: int) -> tuple[str, str | int]:
    """The replicate a lane belongs to: its ``(condition, sample)``. An unnamed lane
    (None or a blank name) is its own sample, keyed by its int position so it can
    never merge with a sample the user named with a digit (e.g. "2") or with
    another unnamed lane."""
    named = sample is not None and str(sample).strip() != ""
    return (condition, str(sample) if named else lane)


def replicate_lanes(
    conditions: Sequence[str],
    samples: Sequence[str | None] | None,
    included: Sequence[bool] | None = None,
) -> dict[str, list[list[int]]]:
    """Per condition, the lanes of each replicate (biological sample), keyed as
    :func:`reduce_samples` keys them: every included lane, whether it holds a
    value or not, conditions and replicates in the order of their first lane.

    The results count a condition's replicates from it, so a replicate with no
    value (not detected, or not measured) still counts, with the same keys the
    statistics use for the replicates that have one."""
    n = len(conditions)
    if samples is None:
        samples = [None] * n
    if len(samples) != n or (included is not None and len(included) != n):
        raise ValueError("samples and included must match the conditions' length")
    keyed: dict[tuple[str, str | int], list[int]] = {}
    for i in range(n):
        if included is None or included[i]:
            keyed.setdefault(_replicate_key(conditions[i], samples[i], i), []).append(i)
    lanes: dict[str, list[list[int]]] = {}
    for (condition, _), members in keyed.items():
        lanes.setdefault(condition, []).append(members)
    return lanes


def reduce_samples(
    values: LaneNets,
    conditions: list[str],
    samples: list[str | None] | None = None,
    *,
    included: list[bool] | None = None,
    method: ReduceMethod | str = ReduceMethod.MEAN,
) -> SampleReduction:
    """Collapse technical repeats to one value per biological sample, then group.

    Lanes sharing ``(condition, sample)`` are technical repeats of one sample;
    they are reduced to a single value (mean, or the first when picking a
    representative) *before* grouping, so n counts biological samples, not lanes —
    this is what prevents pseudoreplication. When ``samples`` is ``None`` every
    lane is treated as its own sample (the biological-replicate default).
    ``included=False`` drops presentation-only lanes. ``method`` may be its raw
    value (``"mean"``); an unknown value raises ``ValueError``.
    """
    method = ReduceMethod(method)  # before the identity checks below
    n = len(conditions)
    if len(values) != n:
        raise ValueError("values and conditions length mismatch")
    if samples is None:
        samples = [None] * n
    elif len(samples) != n:
        raise ValueError("samples and conditions length mismatch")
    if included is not None and len(included) != n:
        raise ValueError("included and conditions length mismatch")

    # Collect each (condition, sample)'s lane values, preserving first-seen order.
    buckets: dict[tuple[str, str | int], list[float]] = {}
    bucket_lanes: dict[tuple[str, str | int], list[int]] = {}
    order: list[tuple[str, str | int]] = []
    for i in range(n):
        if included is not None and not included[i]:
            continue
        if values[i] is None:
            continue
        key = _replicate_key(conditions[i], samples[i], i)
        if key not in buckets:
            buckets[key] = []
            bucket_lanes[key] = []
            order.append(key)
        buckets[key].append(values[i])
        bucket_lanes[key].append(i)

    groups: dict[str, list[float]] = {}
    lanes: dict[str, list[list[int]]] = {}
    averaged: list[tuple[str, str]] = []
    for key in order:
        cond, sample = key
        vals = buckets[key]
        reduced = float(np.mean(vals)) if method is ReduceMethod.MEAN else vals[0]
        groups.setdefault(cond, []).append(reduced)
        lanes.setdefault(cond, []).append(bucket_lanes[key])
        if len(vals) > 1:  # only named samples can span several lanes
            averaged.append((cond, str(sample)))

    warnings = [repeats_message(len(averaged), method)] if averaged else []
    return SampleReduction(groups=groups, averaged=averaged, warnings=warnings, lanes=lanes)


BaselineReason = Literal["no_value", "not_positive"]


class BaselineError(ValueError):
    """The control condition gives no usable fold-change baseline; ``reason`` says why."""

    def __init__(self, reason: BaselineReason, message: str) -> None:
        super().__init__(message)
        self.reason: BaselineReason = reason


def reference_baseline(groups: Mapping[str, Sequence[float]], control_condition: str) -> float:
    """The fold-change baseline: the mean of the control condition's reduced values.

    ``groups`` is a :class:`SampleReduction`'s ``groups``, so the baseline sees the
    same included samples, with technical repeats collapsed, as the statistics.
    Raises :class:`BaselineError` (a ``ValueError``) if the control condition has
    no value (``no_value``) or its mean is not positive (``not_positive``).
    """
    control_vals = groups.get(control_condition, [])
    if not control_vals:
        raise BaselineError(
            "no_value",
            f"control condition {control_condition!r} has no value in any included lane",
        )
    baseline = float(np.mean(control_vals))
    if baseline <= 0:
        raise BaselineError(
            "not_positive", "control condition mean is non-positive; cannot form fold-change"
        )
    return baseline


def fold_change_lane(
    values: LaneNets,
    conditions: list[str],
    control_condition: str,
    samples: list[str | None] | None = None,
    *,
    included: list[bool] | None = None,
    method: ReduceMethod | str = ReduceMethod.MEAN,
) -> LaneNets:
    """Express each lane as a fold-change vs the control condition's baseline.

    The baseline is the mean of the control condition's *sample* values, reduced
    exactly as the statistics see them (:func:`reduce_samples`): ``included=False``
    lanes are dropped and technical repeats collapse to one value per sample. So
    the control group's reduced fold-changes average to 1.0 and its n matches the
    statistics. Pass the lane table's ``included``, not a plot's condition subset:
    the baseline must not depend on which conditions are charted.

    Keeps the per-lane shape (so individual points survive), dividing every lane
    by the baseline. Returns ``None`` lanes unchanged. Raises :class:`BaselineError`
    when :func:`reference_baseline` finds no usable baseline.
    """
    reduction = reduce_samples(values, conditions, samples, included=included, method=method)
    baseline = reference_baseline(reduction.groups, control_condition)
    return [None if v is None else v / baseline for v in values]


@dataclass(frozen=True)
class GroupStats:
    """A group's descriptive statistics. The geometric mean and the geometric SD
    factor (``exp`` of the SD of the logs) describe what a log-scale test
    compares; they are None when a value is 0 or below (and the factor also
    when n < 2)."""

    label: str
    n: int
    mean: float
    sd: float
    sem: float
    geometric_mean: float | None = None
    geometric_sd_factor: float | None = None


def describe(groups: Mapping[str, Sequence[float]]) -> list[GroupStats]:
    """Mean, SD and SEM per group, and the geometric mean and SD factor. Both
    error types are always computed; the choice of which to display is a
    presentation parameter, never hardcoded here.
    """
    out: list[GroupStats] = []
    for label, vals in groups.items():
        n = len(vals)
        if n == 0:
            out.append(GroupStats(label, 0, float("nan"), float("nan"), float("nan")))
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if n > 1 else 0.0
        sem = sd / np.sqrt(n) if n > 1 else 0.0
        geometric = factor = None
        if bool(np.all(arr > 0)):
            logs = np.log(arr)
            geometric = float(np.exp(logs.mean()))
            factor = float(np.exp(logs.std(ddof=1))) if n > 1 else None
        out.append(GroupStats(label, n, mean, sd, sem, geometric, factor))
    return out


# --- Inferential statistics: the setting, the automatic rule and the tests ---
#
# A chart's test is chosen from the design of the data, never from the shape of
# its values: the value kind (raw, or a ratio), which conditions have n >= 2 and
# their n, and whether a reference is set and tested (:func:`plan_test`). No
# normality or variance pre-test is run. Each field of the setting is either
# ``auto`` (resolved by the rule) or the user's; a choice of the user's that
# cannot apply gives no test and says why, never another test. Every test is
# two-sided.

ALPHA: Final = 0.05
# scipy's Dunnett integrates by randomized quasi-Monte Carlo; the seed makes its p
# the same on every run.
DUNNETT_SEED: Final = 0
# Bumped whenever the automatic rule changes; the reproducibility record pins it.
AUTO_RULE_VERSION: Final = 1
# The Mann-Whitney U test is exact: scipy's exact distribution for distinct
# values, and with tied values the exact permutation distribution up to this
# many arrangements (a few tenths of a second), beyond which scipy's normal
# approximation with the tie correction is used. The result names the method
# (TestResult.method), and a chart says when it is the approximation.
MANN_WHITNEY_PERMUTATIONS: Final = 20_000
# The note a test gives when fewer than two conditions have 2 replicates or more.
TOO_FEW_NOTE: Final = "need >=2 groups with >=2 replicates for a test"


class TestFamily(StrEnum):
    """Which kind of test: the values are stable for clients, exports and the record."""

    AUTO = "auto"
    POOLED = "pooled"  # equal variances: Student's t, ANOVA + Tukey-Kramer, Dunnett
    WELCH = "welch"  # unequal variances: Welch's t, Welch's ANOVA + Games-Howell, Holm
    RANK = "rank"  # Mann-Whitney U, Kruskal-Wallis + Dunn
    NONE = "none"  # statistics off


class TestComparisons(StrEnum):
    AUTO = "auto"
    ALL_PAIRS = "all_pairs"
    VS_REFERENCE = "vs_reference"  # each condition vs the reference (many-to-one)


class TestScale(StrEnum):
    AUTO = "auto"
    LOG = "log"  # natural log: the estimate is a ratio of geometric means
    LINEAR = "linear"


class Design(StrEnum):
    INDEPENDENT = "independent"
    BLOCKED = "blocked"  # reserved: several batches pooled, each a block


Origin = Literal["auto", "user"]
# Why a plan has no test: fewer than 2 testable conditions, statistics turned off,
# a choice of the user's that cannot apply, or several batches (not supported yet).
NoTest = Literal["too_few", "statistics_off", "not_applicable", "several_batches"]
_FIELDS: Final = ("family", "comparisons", "scale")


class StatisticsSetting(BaseModel, frozen=True, extra="forbid"):
    """How a chart's test is chosen: a compute argument, like the error type."""

    family: TestFamily = TestFamily.AUTO
    comparisons: TestComparisons = TestComparisons.AUTO
    scale: TestScale = TestScale.AUTO


def statistics_setting(
    value: StatisticsSetting | Mapping[str, str] | None,
) -> StatisticsSetting:
    """``value`` as a :class:`StatisticsSetting`: None is all ``auto``, and a
    mapping may hold raw strings (``{"family": "welch"}``). An unknown key or
    value raises ``ValueError``."""
    if value is None:
        return StatisticsSetting()
    if isinstance(value, StatisticsSetting):
        return value
    try:
        return StatisticsSetting.model_validate(value)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'setting'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ValueError(f"invalid statistics setting: {problems}") from None


@dataclass(frozen=True)
class ExcludedGroup:
    """A condition a plan leaves out of its test, and why."""

    condition: str
    n: int
    reason: Literal["fewer_than_2"] = "fewer_than_2"


@dataclass(frozen=True)
class TestPlan:
    """Which test a chart runs, over which conditions, and why (:func:`plan_test`).

    ``test`` is an id of :data:`TESTS`, or None for no test (``no_test`` says why
    and ``no_test_note`` in words). ``family``, ``comparisons`` and ``scale`` are
    resolved, never ``auto``; ``scale`` is None for a rank test, which the scale
    does not change. ``covered`` lists the tested conditions, the reference first
    for comparisons with it, else in the order given. ``chosen`` says per field
    whether the user chose it or the rule did, ``reasons`` why the rule resolved
    its fields so, and ``notes`` what the chart says besides: the linear scale a
    ratio fell back to (``nonpositive`` names the conditions with a value of 0
    or below), or the smallest p a rank test can give here
    (``min_attainable_p``) when it cannot reach :data:`ALPHA`.
    """

    test: str | None
    name: str | None
    family: TestFamily | None
    comparisons: TestComparisons | None
    scale: TestScale | None
    reference: str | None  # set only for comparisons with the reference
    covered: tuple[str, ...]
    excluded: tuple[ExcludedGroup, ...]
    chosen: Mapping[str, Origin] = field(hash=False)
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    min_attainable_p: float | None = None
    design: Design = Design.INDEPENDENT
    no_test: NoTest | None = None
    no_test_note: str | None = None
    nonpositive: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairwiseResult:
    """One comparison. ``p_value`` is adjusted within the test's family of
    comparisons (raw when there is one). ``estimate`` is b vs a: the ratio of
    geometric means on log values, the difference of means on linear ones, and
    None for a rank test."""

    group_a: str
    group_b: str
    p_value: float
    estimate: float | None = None


@dataclass(frozen=True)
class TestResult:
    """Outcome of a test: an omnibus comparison, or one or several pairwise ones.

    ``test`` is the id of :data:`TESTS` that ran, or ``"none"`` (``note`` says
    why). ``p_value`` and ``statistic`` are the omnibus test's (``omnibus`` names
    it), or the single comparison's when two conditions are tested; None for
    comparisons with a reference among three or more conditions, whose p-values
    are only pairwise. ``pairwise`` holds every comparison. ``method`` says how
    the p was computed for a test that has several ways, the Mann-Whitney U test
    (``"exact"``, ``"exact permutation"`` or ``"normal approximation"``); None
    for the others.
    """

    test: str
    p_value: float | None
    statistic: float | None
    pairwise: list[PairwiseResult] = field(default_factory=list)
    note: str | None = None
    plan: TestPlan | None = None
    omnibus: str | None = None
    method: str | None = None


@dataclass(frozen=True)
class _Run:
    """What a runner gives: the omnibus (or single) statistic and p, each
    comparison as (index a, index b, adjusted p) into the covered conditions,
    and how the p was computed when the test has several ways."""

    statistic: float | None
    p_value: float | None
    pairs: list[tuple[int, int, float]]
    method: str | None = None


@dataclass(frozen=True)
class TestSpec:
    """One registered test. ``conditions`` is the number it takes (``"two"``, or
    ``"three_or_more"``); a test of two takes either comparisons, as with two
    conditions all pairs are the one pair with the reference. ``adjustment``
    names how its pairwise p-values are adjusted, ``omnibus`` its omnibus test
    and ``omnibus_short`` how a chart names that test's p."""

    id: str
    name: str
    family: TestFamily
    runner: Callable[[list[np.ndarray]], _Run] = field(repr=False, compare=False)
    comparisons: TestComparisons | None = None  # None: a test of two conditions
    conditions: Literal["two", "three_or_more"] = "two"
    adjustment: str = "none"
    omnibus: str | None = None
    omnibus_short: str | None = None


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment, in the order given. A NaN p makes every
    adjusted p NaN: the family then has no p-values to give."""
    m = len(p_values)
    if any(math.isnan(p) for p in p_values):
        return [math.nan] * m
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: p_values[i])):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def dunn(samples: Sequence[Sequence[float]], pairs: Sequence[tuple[int, int]]) -> list[float]:
    """Dunn's z-tests of the ``pairs`` (indices into ``samples``) on the ranks of
    all the samples together, with the tie correction, Holm-adjusted."""
    arrays = [np.asarray(sample, dtype=float) for sample in samples]
    data = np.concatenate(arrays)
    ranks = stats.rankdata(data)
    total = data.size
    _, counts = np.unique(data, return_counts=True)
    ties = float((counts**3 - counts).sum()) / (12 * (total - 1))
    edges = np.cumsum([0] + [array.size for array in arrays])
    means = [ranks[edges[g] : edges[g + 1]].mean() for g in range(len(arrays))]
    raw: list[float] = []
    for i, j in pairs:
        variance = (total * (total + 1) / 12 - ties) * (1 / arrays[i].size + 1 / arrays[j].size)
        z = abs(means[i] - means[j]) / math.sqrt(variance) if variance > 0 else math.nan
        raw.append(float(2 * stats.norm.sf(z)))
    return holm(raw)


def _all_pairs(k: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(k), 2))


def _reference_pairs(k: int) -> list[tuple[int, int]]:
    return [(0, j) for j in range(1, k)]


def _dunn_z_max(a: int, b: int, total: int) -> float:
    """The largest Dunn z between two groups of ``a`` and ``b`` values among
    ``total``: one group tied at the bottom, the other tied at the top and the
    rest tied between them (the ties shrink the variance, not the mean ranks)."""
    rest = total - a - b
    ties = sum(t**3 - t for t in (a, rest, b)) / (12 * (total - 1))
    variance = (total * (total + 1) / 12 - ties) * (1 / a + 1 / b)
    return (total - (a + b) / 2) / math.sqrt(variance)


def min_attainable_p(sizes: Sequence[int], test: str, n_comparisons: int) -> float:
    """The smallest p a rank test can give with these group sizes, over any
    values, ties included: no p it gives, adjusted, is below it.

    ``mann_whitney``: the exact two-sided p of complete separation. ``dunn_holm``
    (``sizes[0]`` is the reference) and ``kruskal_dunn_holm``: Holm's smallest
    adjusted p, ``n_comparisons`` times the p of the largest z any pair can have.
    ``KeyError`` for another test."""
    if test == "mann_whitney":
        a, b = sizes
        return min(1.0, 2 / math.comb(a + b, a))
    if test == "dunn_holm":
        pairs = _reference_pairs(len(sizes))
    elif test == "kruskal_dunn_holm":
        pairs = _all_pairs(len(sizes))
    else:
        raise KeyError(f"{test!r} is not a rank test")
    total = sum(sizes)
    z = max(_dunn_z_max(sizes[i], sizes[j], total) for i, j in pairs)
    return min(1.0, n_comparisons * float(2 * stats.norm.sf(z)))


def _t_test(equal_var: bool) -> Callable[[list[np.ndarray]], _Run]:
    def run(xs: list[np.ndarray]) -> _Run:
        res = stats.ttest_ind(xs[0], xs[1], equal_var=equal_var)
        p = float(res.pvalue)
        return _Run(float(res.statistic), p, [(0, 1, p)])

    return run


def _mann_whitney(xs: list[np.ndarray]) -> _Run:
    a, b = xs
    method: str | stats.PermutationMethod = "exact"
    name = "exact"
    if np.unique(np.concatenate(xs)).size < a.size + b.size:  # tied values
        if math.comb(a.size + b.size, a.size) <= MANN_WHITNEY_PERMUTATIONS:
            method, name = stats.PermutationMethod(n_resamples=np.inf), "exact permutation"
        else:
            method, name = "asymptotic", "normal approximation"
    res = stats.mannwhitneyu(a, b, method=method)
    p = float(res.pvalue)
    return _Run(float(res.statistic), p, [(0, 1, p)], name)


def _anova(equal_var: bool) -> Callable[[list[np.ndarray]], _Run]:
    def run(xs: list[np.ndarray]) -> _Run:
        omnibus = stats.f_oneway(*xs, equal_var=equal_var)
        tukey = stats.tukey_hsd(*xs, equal_var=equal_var)
        pairs = [(i, j, float(tukey.pvalue[i, j])) for i, j in _all_pairs(len(xs))]
        return _Run(float(omnibus.statistic), float(omnibus.pvalue), pairs)

    return run


def _dunnett(xs: list[np.ndarray]) -> _Run:
    res = stats.dunnett(*xs[1:], control=xs[0], rng=np.random.default_rng(DUNNETT_SEED))
    return _Run(None, None, [(0, j, float(res.pvalue[j - 1])) for j in range(1, len(xs))])


def _welch_holm(xs: list[np.ndarray]) -> _Run:
    pairs = _reference_pairs(len(xs))
    raw = [float(stats.ttest_ind(xs[i], xs[j], equal_var=False).pvalue) for i, j in pairs]
    return _Run(None, None, [(i, j, p) for (i, j), p in zip(pairs, holm(raw), strict=True)])


def _kruskal_dunn(xs: list[np.ndarray]) -> _Run:
    omnibus = stats.kruskal(*xs)
    pairs = _all_pairs(len(xs))
    adjusted = dunn(xs, pairs)
    return _Run(
        float(omnibus.statistic),
        float(omnibus.pvalue),
        [(i, j, p) for (i, j), p in zip(pairs, adjusted, strict=True)],
    )


def _dunn_reference(xs: list[np.ndarray]) -> _Run:
    pairs = _reference_pairs(len(xs))
    return _Run(None, None, [(i, j, p) for (i, j), p in zip(pairs, dunn(xs, pairs), strict=True)])


def _many(
    test_id: str,
    name: str,
    family: TestFamily,
    runner: Callable[[list[np.ndarray]], _Run],
    comparisons: TestComparisons,
    adjustment: str,
    omnibus: str | None = None,
    omnibus_short: str | None = None,
) -> TestSpec:
    """A registered test of three conditions or more."""
    return TestSpec(
        test_id,
        name,
        family,
        runner,
        comparisons,
        "three_or_more",
        adjustment,
        omnibus,
        omnibus_short,
    )


_F, _C = TestFamily, TestComparisons
# The one registry of tests, for the web, the exports and the record. A test is
# added here, under a new stable id, and the rule (plan_test) says when it runs.
TESTS: Final[Mapping[str, TestSpec]] = MappingProxyType(
    {
        spec.id: spec
        for spec in (
            TestSpec("student_t", "Student's t-test", _F.POOLED, _t_test(equal_var=True)),
            TestSpec("welch_t", "Welch's t-test", _F.WELCH, _t_test(equal_var=False)),
            TestSpec("mann_whitney", "Mann-Whitney U test", _F.RANK, _mann_whitney),
            _many(
                "anova_tukey",
                "One-way ANOVA + Tukey-Kramer",
                _F.POOLED,
                _anova(equal_var=True),
                _C.ALL_PAIRS,
                "Tukey-Kramer",
                "One-way ANOVA",
                "ANOVA",
            ),
            _many(
                "welch_anova_games_howell",
                "Welch's ANOVA + Games-Howell",
                _F.WELCH,
                _anova(equal_var=False),
                _C.ALL_PAIRS,
                "Games-Howell",
                "Welch's ANOVA",
                "Welch's ANOVA",
            ),
            _many(
                "kruskal_dunn_holm",
                "Kruskal-Wallis + Dunn's test (Holm)",
                _F.RANK,
                _kruskal_dunn,
                _C.ALL_PAIRS,
                "Holm",
                "Kruskal-Wallis",
                "Kruskal-Wallis",
            ),
            _many("dunnett", "Dunnett's test", _F.POOLED, _dunnett, _C.VS_REFERENCE, "Dunnett"),
            _many(
                "welch_t_holm",
                "Welch's t-tests (Holm)",
                _F.WELCH,
                _welch_holm,
                _C.VS_REFERENCE,
                "Holm",
            ),
            _many(
                "dunn_holm", "Dunn's test (Holm)", _F.RANK, _dunn_reference, _C.VS_REFERENCE, "Holm"
            ),
        )
    }
)


def registered_test(family: TestFamily, comparisons: TestComparisons, k: int) -> TestSpec:
    """The registered test of ``family`` for ``comparisons`` among ``k`` conditions
    (2, or 3 and more): what a plan resolved to those runs."""
    size = "two" if k == 2 else "three_or_more"
    return next(
        spec
        for spec in TESTS.values()
        if spec.family is family
        and spec.conditions == size
        and spec.comparisons in (None, comparisons)
    )


def rank_floor_note(name: str, floor: float) -> str:
    """What a chart says of the rank test ``name`` when the smallest p it can
    give here, ``floor`` (:func:`min_attainable_p`), is not below :data:`ALPHA`."""
    return (
        f"{name}: with these n no p can be below {floor:.2g},"
        f" so no comparison can reach p < {ALPHA:g}"
    )


def _listed(labels: Sequence[str]) -> str:
    """``'α', 'β' have``: the labels and the verb that agrees with them."""
    return f"{', '.join(map(repr, labels))} {'has' if len(labels) == 1 else 'have'}"


def plan_test(
    groups: Mapping[str, Sequence[float]],
    setting: StatisticsSetting | Mapping[str, str] | None,
    *,
    ratio: bool,
    reference: str | None,
    blocks: Mapping[str, Sequence[str]] | None = None,
) -> TestPlan:
    """Which test ``groups`` get under ``setting``: the one place that decides
    which conditions a chart tests. It reads the design only, never the shape
    of the values.

    Conditions with fewer than 2 values are left out (``excluded``); a test
    needs 2 conditions or more. The ``auto`` fields resolve by the rule (version
    :data:`AUTO_RULE_VERSION`):

    * scale: log values when ``ratio`` (a normalized value or a fold change) and
      every tested value is above 0; otherwise linear, with a note for a ratio;
    * family: pooled variance when every tested condition has the same n,
      Welch's otherwise (a rank test is never chosen automatically);
    * comparisons: each condition vs the ``reference`` when it is tested and 3
      conditions or more are, else all pairs.

    With equal n that gives Student's t (2 conditions), Dunnett's test (vs a
    tested reference) or ANOVA + Tukey-Kramer; with unequal n, Welch's t,
    Welch's t-tests vs the reference with Holm, or Welch's ANOVA +
    Games-Howell. A field the user chose is used as it is, and when it cannot
    apply (comparisons with a reference that is not tested, log values of a
    value of 0 or below) there is no test, and the note says why. ``blocks``
    (a batch per value, for pooled batches) gives no test yet.
    """
    setting = statistics_setting(setting)
    chosen: dict[str, Origin] = {
        name: "auto" if getattr(setting, name) == "auto" else "user" for name in _FIELDS
    }
    testable = [label for label, values in groups.items() if len(values) >= 2]
    excluded = tuple(
        ExcludedGroup(label, len(values)) for label, values in groups.items() if len(values) < 2
    )

    def no_test(reason: NoTest, note: str) -> TestPlan:
        return TestPlan(
            test=None,
            name=None,
            family=None,
            comparisons=None,
            scale=None,
            reference=None,
            covered=(),
            excluded=excluded,
            chosen=chosen,
            no_test=reason,
            no_test_note=note,
        )

    if blocks is not None:
        return no_test("several_batches", "no test: several batches are not supported yet")
    if setting.family is TestFamily.NONE:
        return no_test("statistics_off", "no test: statistics are turned off")
    k = len(testable)
    if k < 2:
        return no_test("too_few", TOO_FEW_NOTE)

    reference_tested = reference is not None and reference in testable
    if setting.comparisons is TestComparisons.AUTO:
        if reference_tested and k >= 3:
            comparisons = TestComparisons.VS_REFERENCE
            why_comparisons = (
                "a tested reference and 3 or more conditions: each condition vs the reference"
            )
        else:
            comparisons = TestComparisons.ALL_PAIRS
            why_comparisons = (
                "2 conditions: all pairs" if k == 2 else "no tested reference: all pairs"
            )
    else:
        comparisons, why_comparisons = setting.comparisons, None
        if comparisons is TestComparisons.VS_REFERENCE and not reference_tested:
            if reference is None:
                why = "no reference is set"
            elif reference in groups:
                why = f"the reference {reference!r} has fewer than 2 replicates"
            else:
                why = f"the reference {reference!r} is not among the conditions tested"
            return no_test(
                "not_applicable", f"no test: comparisons with the reference were chosen, but {why}"
            )

    if setting.family is TestFamily.AUTO:
        if len({len(groups[label]) for label in testable}) == 1:
            family, why_family = TestFamily.POOLED, "equal n: pooled variance"
        else:
            family, why_family = TestFamily.WELCH, "unequal n: Welch"
    else:
        family, why_family = setting.family, None

    below = tuple(label for label in testable if any(v <= 0 for v in groups[label]))
    notes: list[str] = []
    nonpositive: tuple[str, ...] = ()
    why_scale: str | None = None
    scale: TestScale | None
    if family is TestFamily.RANK:
        scale = None  # ranks are the same on any scale
    elif setting.scale is TestScale.AUTO:
        if not ratio:
            scale, why_scale = TestScale.LINEAR, "raw values: linear scale"
        elif below:
            scale, why_scale = TestScale.LINEAR, "ratios with a value of 0 or below: linear scale"
            nonpositive = below
            notes.append(f"Linear scale: {_listed(below)} a value of 0 or below")
        else:
            scale, why_scale = TestScale.LOG, "ratios: log scale"
    elif setting.scale is TestScale.LOG and below:
        return no_test(
            "not_applicable",
            f"no test: log values were chosen, but {_listed(below)} a value of 0 or below",
        )
    else:
        scale = setting.scale

    spec = registered_test(family, comparisons, k)
    covered = tuple(testable)
    if comparisons is TestComparisons.VS_REFERENCE:
        covered = (reference, *[label for label in testable if label != reference])
    floor: float | None = None
    if family is TestFamily.RANK:
        if k == 2:
            count = 1
        elif comparisons is TestComparisons.VS_REFERENCE:
            count = k - 1
        else:
            count = k * (k - 1) // 2
        floor = min_attainable_p([len(groups[label]) for label in covered], spec.id, count)
        if floor >= ALPHA:
            notes.append(rank_floor_note(spec.name, floor))
    reasons = tuple(why for why in (why_scale, why_family, why_comparisons) if why is not None)
    return TestPlan(
        test=spec.id,
        name=spec.name,
        family=family,
        comparisons=comparisons,
        scale=scale,
        reference=reference if comparisons is TestComparisons.VS_REFERENCE else None,
        covered=covered,
        excluded=excluded,
        chosen=chosen,
        reasons=reasons,
        notes=tuple(notes),
        min_attainable_p=floor,
        nonpositive=nonpositive,
    )


def run_test(groups: Mapping[str, Sequence[float]], plan: TestPlan) -> TestResult:
    """Run ``plan`` on ``groups``: its test over the conditions it covers, or no
    test with its note. Results are cached by the plan and the values."""
    if plan.test is None:
        return TestResult(
            test="none", p_value=None, statistic=None, note=plan.no_test_note, plan=plan
        )
    frozen = tuple((label, tuple(float(v) for v in groups[label])) for label in plan.covered)
    result = _run(plan, frozen)
    return replace(result, pairwise=list(result.pairwise))  # the cached list stays as it was


@functools.lru_cache(maxsize=256)
def _run(plan: TestPlan, frozen: tuple[tuple[str, tuple[float, ...]], ...]) -> TestResult:
    spec = TESTS[plan.test]
    labels = [label for label, _ in frozen]
    raw = [np.asarray(values, dtype=float) for _, values in frozen]
    xs = [np.log(values) for values in raw] if plan.scale is TestScale.LOG else raw
    run = spec.runner(xs)

    def estimate(i: int, j: int) -> float | None:
        if plan.family is TestFamily.RANK:
            return None
        difference = float(xs[j].mean() - xs[i].mean())
        return math.exp(difference) if plan.scale is TestScale.LOG else difference

    return TestResult(
        test=spec.id,
        p_value=run.p_value,
        statistic=run.statistic,
        pairwise=[PairwiseResult(labels[i], labels[j], p, estimate(i, j)) for i, j, p in run.pairs],
        plan=plan,
        omnibus=spec.omnibus,
        method=run.method,
    )


def clear_test_cache() -> None:
    """Forget every cached test result (:func:`run_test`)."""
    _run.cache_clear()


def compare(
    groups: Mapping[str, Sequence[float]],
    setting: StatisticsSetting | Mapping[str, str] | None,
    *,
    ratio: bool,
    reference: str | None,
    blocks: Mapping[str, Sequence[str]] | None = None,
) -> TestResult:
    """The test ``setting`` gives ``groups``: :func:`run_test` of :func:`plan_test`.

    ``ratio`` and ``reference`` have no defaults on purpose: every caller states
    the value kind and the reference, so no caller tests linear values over all
    pairs while the app runs Dunnett's test on log values.
    """
    return run_test(
        groups, plan_test(groups, setting, ratio=ratio, reference=reference, blocks=blocks)
    )
