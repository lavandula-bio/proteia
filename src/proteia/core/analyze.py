# SPDX-License-Identifier: Apache-2.0
"""Batch-level analysis: normalize, group by condition, and run statistics.

GUI-independent. The atomic unit upstream is an :class:`~proteia.core.model.Analysis`
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
ratios, or fold-changes. Statistics and plotting treat them the same; only the
``value_kind`` label distinguishes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy import stats

# A protein's net signal per lane, aligned to the shared lane spine. ``None`` marks
# a lane where this protein has no box (a gap). Length == number of lanes.
LaneNets = list[float | None]


class Role(StrEnum):
    TARGET = "target"
    LOADING_CONTROL = "loading control"


class Tier(StrEnum):
    """How far up the computability ladder a batch can go."""

    EXPORT_ONLY = "export_only"  # missing target or loading control
    NORMALIZED = "normalized"  # target + loading control present
    FOLD_CHANGE = "fold_change"  # + a designated control condition


@dataclass(frozen=True)
class ProteinNets:
    """One protein's per-lane nets within a batch, plus its role."""

    name: str
    role: Role
    nets: LaneNets


@dataclass(frozen=True)
class Batch:
    """One experiment: proteins sharing a lane spine of condition labels.

    ``conditions`` gives the condition label of each lane (repeats == replicates);
    every protein's ``nets`` is aligned to it by index. ``control_condition`` names
    the reference group for fold-change, if any.
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

    def loading_control(self) -> ProteinNets | None:
        for p in self.proteins:
            if p.role is Role.LOADING_CONTROL:
                return p
        return None


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


def group_by_condition(values: LaneNets, conditions: list[str]) -> dict[str, list[float]]:
    """Pool per-lane values into replicate lists keyed by condition label.

    Lanes with a ``None`` value are skipped. Insertion order of first appearance
    is preserved so groups read left-to-right like the gel.
    """
    groups: dict[str, list[float]] = {}
    for cond, val in zip(conditions, values, strict=True):
        groups.setdefault(cond, [])
        if val is not None:
            groups[cond].append(val)
    return groups


def fold_change_lane(values: LaneNets, conditions: list[str], control_condition: str) -> LaneNets:
    """Express each lane as a fold-change vs the control condition's mean.

    Keeps the per-lane shape (so individual points survive), dividing every lane
    by the mean of the control group. Returns ``None`` lanes unchanged.
    """
    control_vals = group_by_condition(values, conditions).get(control_condition, [])
    if not control_vals:
        raise ValueError(f"control condition {control_condition!r} has no values")
    baseline = float(np.mean(control_vals))
    if baseline <= 0:
        raise ValueError("control condition mean is non-positive; cannot form fold-change")
    return [None if v is None else v / baseline for v in values]


class ReduceMethod(StrEnum):
    MEAN = "mean"  # average technical repeats (force-merge; the safe default)
    REPRESENTATIVE = "representative"  # keep one repeat per sample, drop the rest


@dataclass(frozen=True)
class SampleReduction:
    """Per-condition lists of *sample* values, after collapsing technical repeats.

    ``groups`` feeds the statistics: each value is one biological sample, so its
    length is the correct n. ``averaged`` lists the ``(condition, sample)`` keys
    that had more than one lane (i.e. were collapsed), for transparency.
    """

    groups: dict[str, list[float]]
    averaged: list[tuple[str, str]]
    warnings: list[str] = field(default_factory=list)


def reduce_samples(
    values: LaneNets,
    conditions: list[str],
    samples: list[str | None] | None = None,
    *,
    included: list[bool] | None = None,
    method: ReduceMethod = ReduceMethod.MEAN,
) -> SampleReduction:
    """Collapse technical repeats to one value per biological sample, then group.

    Lanes sharing ``(condition, sample)`` are technical repeats of one sample;
    they are reduced to a single value (mean, or the first when picking a
    representative) *before* grouping, so n counts biological samples, not lanes —
    this is what prevents pseudoreplication. When ``samples`` is ``None`` every
    lane is treated as its own sample (the biological-replicate default).
    ``included=False`` drops presentation-only lanes.
    """
    n = len(conditions)
    if len(values) != n:
        raise ValueError("values and conditions length mismatch")
    if samples is None:
        samples = [str(i) for i in range(n)]
    elif len(samples) != n:
        raise ValueError("samples and conditions length mismatch")
    if included is not None and len(included) != n:
        raise ValueError("included and conditions length mismatch")

    # Collect each (condition, sample)'s lane values, preserving first-seen order.
    buckets: dict[tuple[str, str], list[float]] = {}
    order: list[tuple[str, str]] = []
    for i in range(n):
        if included is not None and not included[i]:
            continue
        if values[i] is None:
            continue
        key = (conditions[i], str(samples[i]) if samples[i] is not None else str(i))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(values[i])

    groups: dict[str, list[float]] = {}
    averaged: list[tuple[str, str]] = []
    for key in order:
        cond, _sample = key
        vals = buckets[key]
        reduced = float(np.mean(vals)) if method is ReduceMethod.MEAN else vals[0]
        groups.setdefault(cond, []).append(reduced)
        if len(vals) > 1:
            averaged.append(key)

    warnings: list[str] = []
    if averaged:
        what = "averaged" if method is ReduceMethod.MEAN else "kept one lane of"
        warnings.append(
            f"{what} {len(averaged)} sample(s) with technical repeats "
            "(repeats do not count as n)"
        )
    return SampleReduction(groups=groups, averaged=averaged, warnings=warnings)


@dataclass(frozen=True)
class GroupStats:
    label: str
    n: int
    mean: float
    sd: float
    sem: float


def describe(groups: dict[str, list[float]]) -> list[GroupStats]:
    """Mean, SD and SEM per group. Both error types are always computed; the
    choice of which to display is a presentation parameter, never hardcoded here.
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
        out.append(GroupStats(label, n, mean, sd, sem))
    return out


@dataclass(frozen=True)
class PairwiseResult:
    group_a: str
    group_b: str
    p_value: float


@dataclass(frozen=True)
class TestResult:
    """Outcome of an omnibus comparison, with optional pairwise post-hoc.

    A uniform shape so new tests register without changing callers or plotting.
    """

    test: str
    p_value: float | None
    statistic: float | None
    pairwise: list[PairwiseResult] = field(default_factory=list)
    note: str | None = None


def compare(groups: dict[str, list[float]]) -> TestResult:
    """Pick a sensible default test by group count and replicate availability.

    * 2 groups -> Welch's unpaired t-test.
    * 3+ groups -> one-way ANOVA + Tukey HSD post-hoc.

    Groups with n < 2 cannot enter a test; if too few qualify we return a result
    with ``p_value=None`` and a note rather than raising. The menu is deliberately
    tiny now; the registry-style shape lets more tests slot in later.
    """
    usable = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(usable) < 2:
        return TestResult(
            test="none",
            p_value=None,
            statistic=None,
            note="need >=2 groups with >=2 replicates for a test",
        )

    labels = list(usable)
    samples = [np.asarray(usable[k], dtype=float) for k in labels]

    if len(usable) == 2:
        res = stats.ttest_ind(samples[0], samples[1], equal_var=False)
        return TestResult(
            test="welch_t",
            p_value=float(res.pvalue),
            statistic=float(res.statistic),
            pairwise=[PairwiseResult(labels[0], labels[1], float(res.pvalue))],
        )

    omni = stats.f_oneway(*samples)
    tukey = stats.tukey_hsd(*samples)
    pairwise: list[PairwiseResult] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairwise.append(PairwiseResult(labels[i], labels[j], float(tukey.pvalue[i, j])))
    return TestResult(
        test="anova_oneway",
        p_value=float(omni.pvalue),
        statistic=float(omni.statistic),
        pairwise=pairwise,
    )