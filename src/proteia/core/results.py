# SPDX-License-Identifier: Apache-2.0
"""The single compute step: a stored batch in, every result the app shows out.

Pure: :func:`compute_results` takes a :class:`proteia.core.model.Batch` and reads
no file and no pixel, so live results and the export reuse it without a session.
One call gives:

* the lane rows and the raw per-lane table: each protein's stored nets joined to
  the lane table by the stored lane index, so a lane with no box is ``None`` and
  shifts nothing. A lane with a not-detected record has no value either, so the
  statistics leave it out as they do a lane with no box; its ``detected`` flag
  (:func:`lane_detected`) and a ``below_detection`` notice tell the two apart;
* one :class:`SeriesResult` per (target, resolved loading control) pairing, with
  its per-lane ratios, fold-changes, reduced groups and chart;
* typed :class:`Notice` objects, built from the model directly, never by parsing
  warning strings.

Within a result set, each series is reduced once
(:func:`~proteia.core.analyze.reduce_samples` with that set's include flags). The
fold-change baseline comes from that reduction and the chart's groups are that
reduction divided by the baseline, so the table and the chart of a series use
one loading control and one baseline, and the plotted subset of conditions,
applied afterwards, never moves the baseline. Dividing the reduced values
(rather than reducing per-lane fold values) is equal in exact arithmetic,
bit-identical for representative repeats and single lanes, and can differ in the
last ulp only for averaged technical repeats.

When excluded lanes (include=no) hold values, a second result set is computed
with every lane included (:attr:`Results.all_lanes`), so removing data points is
never hidden; lanes excluded without any value (a ladder, an empty lane) add no
second set. Each of the two sets is labelled, and every chart carries its set's
label as its subtitle, so a chart cannot be mistaken for the other set's.

Notice messages count lanes from 1, as the user does; every index field
(:attr:`Notice.lane_indices`, :attr:`Results.excluded_lanes`) stays 0-based.
"""

from __future__ import annotations

from collections.abc import Collection
from enum import StrEnum
from functools import partial

from pydantic import BaseModel, model_validator

from proteia.core import analyze, model
from proteia.core.analyze import (
    BaselineError,
    LaneNets,
    ProteinNets,
    ReduceMethod,
    Tier,
    assess,
    compare,
    describe,
    normalize_batch,
    reduce_samples,
    reference_baseline,
    repeats_message,
)
from proteia.core.model import Role
from proteia.core.names import name_key, resolve_label
from proteia.core.plotspec import ErrorType, PlotSpec, ValueKind, build_plotspec
from proteia.core.project import spine_axes


class NoticeCode(StrEnum):
    """What a :class:`Notice` is about. The values are stable for clients."""

    NO_LANES = "no_lanes"
    NO_TARGET = "no_target"
    NO_LOADING_CONTROL = "no_loading_control"
    # The model accepts these two on purpose; compute reports them.
    LOADING_CONTROL_AMBIGUOUS = "loading_control_ambiguous"  # none chosen among 2+
    REFERENCE_ALL_EXCLUDED = "reference_all_excluded"  # every reference lane is include=no
    REFERENCE_UNUSABLE = "reference_unusable"  # per series: no included value, or mean <= 0
    LOADING_NOT_POSITIVE = "loading_not_positive"  # target has a value, loading net <= 0
    NO_VALUES = "no_values"  # a series with no included value at all
    NO_PLOTTED_VALUES = "no_plotted_values"  # a series with no value in the plotted conditions
    TECHNICAL_REPEATS = "technical_repeats"  # repeats averaged, or one repeat kept per sample
    SIMILAR_CONDITIONS = "similar_conditions"  # labels equal except for case (name_key)
    UNKNOWN_PLOT_CONDITION = "unknown_plot_condition"  # ignored: a stale selection must not break
    REFERENCE_NOT_PLOTTED = "reference_not_plotted"  # the baseline still comes from it
    EXTRA_BANDS_IGNORED = "extra_bands_ignored"  # band_index > 0 is not quantified yet
    CLIPPED = "clipped"  # bands with pixels at the detector limit: over-exposed, still included
    BELOW_DETECTION = "below_detection"  # not-detected records in included lanes: no value


class Level(StrEnum):
    INFO = "info"
    WARNING = "warning"


_INFO_CODES = frozenset(
    {
        NoticeCode.TECHNICAL_REPEATS,
        NoticeCode.REFERENCE_NOT_PLOTTED,
        NoticeCode.EXTRA_BANDS_IGNORED,
    }
)


class Notice(BaseModel, frozen=True):
    """Something about the results the user should know, with the objects involved."""

    code: NoticeCode
    level: Level
    message: str
    protein_ids: tuple[str, ...] = ()
    lane_indices: tuple[int, ...] = ()
    conditions: tuple[str, ...] = ()


class LaneRow(BaseModel, frozen=True):
    """One row of the lane table, as the results show it."""

    index: int
    condition: str
    sample: str | None
    included: bool


class ProteinColumn(BaseModel, frozen=True):
    """One protein's column of the raw per-lane table (band index 0 only)."""

    protein_id: str
    name: str
    role: Role
    image_id: str
    nets: list[float | None]  # joined on the stored lane_index; None = no box
    band_ids: list[str | None]
    clipped: list[bool | None]  # per lane: over-exposed; None = no box, or not checked
    # Per lane: True = a box; False = not detected (below the detection limit, no
    # value); None = not measured.
    detected: list[bool | None]


class SeriesResult(BaseModel, frozen=True):
    """One target normalized against one resolved loading control."""

    target_id: str
    target: str
    loading_id: str
    loading: str
    normalized: list[float | None]  # per lane: target net / loading net
    value_kind: ValueKind  # FOLD_CHANGE with a usable baseline, else LOADING_NORMALIZED
    baseline: float | None  # mean of the reference's reduced samples
    fold_change: list[float | None] | None  # per lane: normalized / baseline
    groups: dict[str, list[float]]  # the one reduction, every condition, in value_kind units
    averaged: list[tuple[str, str]]  # (condition, sample) keys whose repeats were collapsed
    chart: PlotSpec | None  # groups restricted to the plotted conditions
    # Per condition, in lane order: the included lanes where the target was not
    # detected (the loading control's are in its below_detection notice).
    undetected: dict[str, list[int]]


class Results(BaseModel, frozen=True):
    """Everything :func:`compute_results` gives, with the arguments it used.

    ``excluded_lanes`` names the lanes this set leaves out (include=no); empty
    means every lane is in. When it is not empty, ``all_lanes`` holds the same
    results with every lane included, so what an exclusion changes is always
    visible, and each set can be shown or exported on its own.

    ``label`` names the set when there are two: ``Excluding lane 8`` or
    ``Excluding lanes 3, 7`` (1-based, ascending) for this one, ``All lanes`` for
    ``all_lanes``. It is ``None`` when there is only one set. It names only the
    excluded lanes that hold values: an excluded lane without any (a ladder)
    changes nothing, so it is in ``excluded_lanes`` but not in the label. Every
    chart of a set has the set's label as its subtitle.
    """

    lanes: list[LaneRow]
    tier: Tier
    reference_condition: str | None
    proteins: list[ProteinColumn]
    series: list[SeriesResult]  # target order, then each target's loading-control order
    notices: list[Notice]
    plot_conditions: list[str] | None  # resolved to stored labels; None = all
    error_type: ErrorType
    method: ReduceMethod
    excluded_lanes: list[int] = []
    label: str | None = None
    all_lanes: Results | None = None

    @model_validator(mode="after")
    def _one_level(self) -> Results:
        if self.all_lanes is not None and self.all_lanes.all_lanes is not None:
            raise ValueError("the all-lanes result set has no all-lanes set of its own")
        return self


def _field(bands: list[model.Band | None], attr: str) -> list:
    """One attribute of joined bands, None where a lane has no band."""
    return [None if band is None else getattr(band, attr) for band in bands]


def _join(protein: model.Protein, n: int) -> list[model.Band | None]:
    """A protein's band-index-0 band per lane, by stored lane index.

    The same join as :func:`~proteia.core.project.join_to_spine`: a lane with no box
    is ``None`` and shifts nothing. The model allows one band-index-0 band per lane.
    """
    by_lane = {band.lane_index: band for band in protein.bands if band.band_index == 0}
    return [by_lane.get(i) for i in range(n)]


def lane_detected(batch: model.Batch) -> dict[str, list[bool | None]]:
    """Each protein's detection state per lane (band index 0), keyed like
    :func:`lane_nets`: True where a box was measured, False where a detector found
    the band below its detection limit (a not-detected record: no value), None
    where the lane was not measured.

    The one join of boxes and records to the lane table, by stored lane index as
    in :func:`_join`; :func:`compute_results` takes the detection state from
    here. The model allows a lane a box or a record, never both.
    """
    n = len(batch.lanes)
    detected: dict[str, list[bool | None]] = {}
    for protein in batch.proteins:
        state = {u.lane_index: False for u in protein.undetected if u.band_index == 0}
        state.update((b.lane_index, True) for b in protein.bands if b.band_index == 0)
        detected[protein.id] = [state.get(i) for i in range(n)]
    return detected


def lane_nets(batch: model.Batch) -> dict[str, LaneNets]:
    """Each protein's stored nets per lane, keyed by protein id in model order.

    Nets are joined to the lane table by their stored lane index (band index 0
    only): a lane with no box is ``None``. With no lanes, every protein has ``[]``.
    """
    n = len(batch.lanes)
    return {protein.id: _field(_join(protein, n), "net") for protein in batch.proteins}


def lane_clipped(batch: model.Batch) -> dict[str, list[bool | None]]:
    """Each protein's clipping flags per lane, keyed like :func:`lane_nets`: None
    where there is no box, or where the band was not checked."""
    n = len(batch.lanes)
    return {protein.id: _field(_join(protein, n), "clipped") for protein in batch.proteins}


def _listed(values: Collection[object]) -> str:
    return ", ".join(repr(v) for v in values)


def _lanes(indices: Collection[int]) -> str:
    """0-based lane indices as the user counts lanes: ``lane 8``, ``lanes 3, 7``
    (1-based, ascending)."""
    numbers = sorted(i + 1 for i in indices)
    return ("lane " if len(numbers) == 1 else "lanes ") + ", ".join(map(str, numbers))


def _chart(
    groups: dict[str, list[float]],
    point_lanes: dict[str, list[list[int]]],
    *,
    chosen: list[str] | None,
    kind: ValueKind,
    error_type: ErrorType,
    title: str,
    reference: str | None,
    subtitle: str | None,
) -> PlotSpec | None:
    """The chart of one series: its groups restricted to the plotted conditions."""
    shown = {c: g for c, g in groups.items() if chosen is None or c in chosen}
    if not shown:
        return None
    # Provenance parallel to the points: each sample's first lane (the lane a
    # representative reduction keeps; technical repeats share one point).
    lane_indices = {c: [lanes[0] for lanes in point_lanes[c]] for c in shown}
    # Statistics run on the plotted subset, as the napari chart does.
    return build_plotspec(
        shown,
        describe(shown),
        compare(shown),
        value_kind=kind,
        error_type=error_type,
        title=title,
        lane_indices=lane_indices,
        first_label=reference,
        subtitle=subtitle,
    )


def compute_results(
    batch: model.Batch,
    *,
    plot_conditions: Collection[str] | None = None,
    error_type: ErrorType | str = ErrorType.SD,
    method: ReduceMethod | str = ReduceMethod.MEAN,
) -> Results:
    """Everything the results view shows, from the stored batch alone.

    See :func:`_compute` for one result set. When lanes the lane table excludes
    (include=no) hold a value for any protein, the returned set applies the
    exclusions and its ``all_lanes`` is the same computation with every lane
    included: removing data points is never hidden. Excluded lanes without any
    value (a ladder, an empty lane) change nothing, so they add no second set.
    Notices the two sets share are kept only in the first. Two sets are labelled
    (see :class:`Results`); one set has no label. A chart with a plotted group
    too small for a test (n < 2) is still drawn. When at least two other groups
    can be tested, the chart shows their test, with brackets only among them and
    a note naming the groups left out; otherwise it has no test result, no
    brackets, and a note saying why (see
    :func:`~proteia.core.plotspec.build_plotspec`).

    ``error_type`` and ``method`` may be their raw values (``"SEM"``, ``"mean"``):
    they become their enums here, before anything compares them by identity. An
    unknown value raises ``ValueError``.
    """
    one_set = partial(
        _compute,
        plot_conditions=plot_conditions,
        error_type=ErrorType(error_type),
        method=ReduceMethod(method),
    )
    per_protein = lane_nets(batch).values()
    removed = [  # the excluded lanes that hold a value: what the exclusion changes
        lane.index
        for lane in batch.lanes
        if not lane.included and any(nets[lane.index] is not None for nets in per_protein)
    ]
    if not removed:
        return one_set(batch, set_label=None)
    results = one_set(batch, set_label=f"Excluding {_lanes(removed)}")
    every_lane = batch.model_copy(
        update={"lanes": [lane.model_copy(update={"included": True}) for lane in batch.lanes]}
    )
    all_lanes = one_set(every_lane, set_label="All lanes")
    own = [notice for notice in all_lanes.notices if notice not in results.notices]
    return results.model_copy(update={"all_lanes": all_lanes.model_copy(update={"notices": own})})


def _compute(
    batch: model.Batch,
    *,
    plot_conditions: Collection[str] | None,
    error_type: ErrorType,
    method: ReduceMethod,
    set_label: str | None,
) -> Results:
    """One result set, over the lane table's included lanes.

    ``plot_conditions`` chooses the charted conditions (None or empty: all); each
    is resolved against the lane labels (:func:`~proteia.core.names.resolve_label`),
    and one that matches no lane is reported and ignored. ``method`` reduces
    technical repeats; ``error_type`` picks the charts' error bars. ``set_label``
    names the set (:attr:`Results.label`) and is every chart's subtitle.

    Series come from :func:`~proteia.core.analyze.normalize_batch`. Each is reduced
    once over the lane table's included lanes. With a reference condition and a
    usable baseline, the per-lane fold-changes and the groups are divided by it;
    an unusable baseline gives loading-normalized groups and no chart for that
    series. The plotted subset is applied last and changes only the chart.
    """
    if isinstance(plot_conditions, str):
        raise TypeError("plot_conditions is a collection of condition labels, not one label")
    n = len(batch.lanes)
    ref = batch.reference_condition
    notices: list[Notice] = []

    def note(code: NoticeCode, message: str, **objects: tuple) -> None:
        level = Level.INFO if code in _INFO_CODES else Level.WARNING
        notices.append(Notice(code=code, level=level, message=message, **objects))

    # 1. The lane rows and the raw table, and what the model lacks.
    lanes = [
        LaneRow(index=lane.index, condition=lane.label, sample=lane.sample, included=lane.included)
        for lane in batch.lanes
    ]
    joined = {protein.id: _join(protein, n) for protein in batch.proteins}
    nets = {pid: _field(bands, "net") for pid, bands in joined.items()}
    detected = lane_detected(batch)
    columns = [
        ProteinColumn(
            protein_id=p.id,
            name=p.name,
            role=p.role,
            image_id=p.image_id,
            nets=nets[p.id],
            band_ids=_field(joined[p.id], "id"),
            clipped=_field(joined[p.id], "clipped"),
            detected=detected[p.id],
        )
        for p in batch.proteins
    ]
    # Boxes only: a record beyond the first band has no value to ignore.
    extra = tuple(p.id for p in batch.proteins if any(b.band_index > 0 for b in p.bands))
    if extra:
        note(
            NoticeCode.EXTRA_BANDS_IGNORED,
            "extra bands (band index above 0) are not quantified yet;"
            " each lane uses its first band",
            protein_ids=extra,
        )
    targets = [p for p in batch.proteins if p.role is Role.TARGET]
    loading_controls = [p for p in batch.proteins if p.role is Role.LOADING_CONTROL]
    if not targets:
        note(NoticeCode.NO_TARGET, "no target protein: raw nets only")
    if not loading_controls:
        note(NoticeCode.NO_LOADING_CONTROL, "no loading control: cannot normalize; raw nets only")

    name_of = {p.id: p.name for p in batch.proteins}
    id_of = {p.name: p.id for p in batch.proteins}  # names are unique in the model
    protein_nets = [
        ProteinNets(
            p.name, p.role, nets[p.id], loadings=tuple(name_of[i] for i in p.loading_control_ids)
        )
        for p in batch.proteins
    ]

    # 2. No lanes: only the raw columns (all empty).
    if n == 0:
        note(NoticeCode.NO_LANES, "no lanes declared: declare the lane table")
        return Results(
            lanes=[],
            tier=assess(analyze.Batch([], protein_nets)).tier,
            reference_condition=ref,
            proteins=columns,
            series=[],
            notices=notices,
            plot_conditions=None,
            error_type=error_type,
            method=method,
        )

    # 3-4. The lane axes, and what the model accepts on purpose but the user should see.
    conditions, samples, included = spine_axes(batch.lanes)
    for column in columns:  # over-exposed bands in lanes this set includes
        over = tuple(i for i, flag in enumerate(column.clipped) if flag and included[i])
        if over:
            kept = "the lane stays" if len(over) == 1 else "the lanes stay"
            note(
                NoticeCode.CLIPPED,
                f"{column.name!r} is over-exposed in {_lanes(over)}: pixels at the detector"
                f" limit make its net an under-estimate; {kept} included",
                protein_ids=(column.protein_id,),
                lane_indices=over,
            )
    for column in columns:  # not-detected records in lanes this set includes
        below = tuple(i for i, flag in enumerate(column.detected) if flag is False and included[i])
        if not below:
            continue
        if column.role is Role.LOADING_CONTROL:
            effect = "targets normalized to it have no value there"
        elif len(below) == 1:
            effect = "that lane has no value and is left out of the statistics"
        else:
            effect = "those lanes have no value and are left out of the statistics"
        note(
            NoticeCode.BELOW_DETECTION,
            f"{column.name!r} was not detected in {_lanes(below)}"
            f" (below the detection limit): {effect}",
            protein_ids=(column.protein_id,),
            lane_indices=below,
            conditions=tuple(dict.fromkeys(conditions[i] for i in below)),
        )
    labels = list(dict.fromkeys(conditions))  # distinct, in lane order
    similar: dict[str, list[str]] = {}
    for label in labels:
        similar.setdefault(name_key(label), []).append(label)
    for group in similar.values():
        if len(group) > 1:
            note(
                NoticeCode.SIMILAR_CONDITIONS,
                f"conditions {_listed(group)} differ only in case or look alike;"
                " they are separate groups",
                conditions=tuple(group),
            )
    if len(loading_controls) > 1:
        for target in targets:
            if not target.loading_control_ids:
                note(
                    NoticeCode.LOADING_CONTROL_AMBIGUOUS,
                    f"{target.name!r} has no loading control chosen among"
                    f" {len(loading_controls)}: it is not normalized",
                    protein_ids=(target.id,),
                )
    reference_lanes = tuple(i for i in range(n) if conditions[i] == ref)
    reference_all_excluded = bool(reference_lanes) and not any(included[i] for i in reference_lanes)
    if reference_all_excluded:
        note(
            NoticeCode.REFERENCE_ALL_EXCLUDED,
            f"every lane of the reference condition {ref!r} is excluded:"
            " no fold-change can be formed from the included lanes",
            lane_indices=reference_lanes,
            conditions=(ref,),
        )

    # 5. The statistics input: loading-control ids become names.
    abatch = analyze.Batch(conditions, protein_nets, control_condition=ref)
    tier = assess(abatch).tier
    if tier is Tier.FOLD_CHANGE and reference_all_excluded:
        tier = Tier.NORMALIZED  # no series can form a fold-change

    # 6. The plotted subset.
    chosen: list[str] | None = None
    if plot_conditions:
        resolved: set[str] = set()
        unknown: dict[str, None] = {}  # an ordered set
        for entry in plot_conditions:
            label = resolve_label(entry, labels)
            if label is None:
                unknown[entry] = None
            else:
                resolved.add(label)
        if unknown:
            note(
                NoticeCode.UNKNOWN_PLOT_CONDITION,
                f"plot condition(s) {_listed(unknown)} match no lane; ignored",
                conditions=tuple(unknown),
            )
        if resolved:
            chosen = [label for label in labels if label in resolved]
    if tier is Tier.FOLD_CHANGE and chosen is not None and ref not in chosen:
        note(
            NoticeCode.REFERENCE_NOT_PLOTTED,
            f"the reference condition {ref!r} is not plotted;"
            " fold-changes are still relative to it",
            conditions=(ref,),
        )

    # 7. One series per (target, resolved loading control), each reduced once.
    series: list[SeriesResult] = []
    averaged: list[tuple[str, str]] = []
    if tier is not Tier.EXPORT_ONLY:
        pairs, _ = normalize_batch(abatch)
        for s in pairs:
            target_id, loading_id = id_of[s.target], id_of[s.loading]
            pair_ids = (target_id, loading_id)
            target_nets, loading_nets = nets[target_id], nets[loading_id]
            not_positive = tuple(
                i
                for i in range(n)
                if included[i]
                and target_nets[i] is not None
                and loading_nets[i] is not None
                and loading_nets[i] <= 0
            )
            if not_positive:
                note(
                    NoticeCode.LOADING_NOT_POSITIVE,
                    f"{s.loading!r} has a net of 0 in {_lanes(not_positive)}:"
                    f" {s.target!r} / {s.loading!r} has no value there",
                    protein_ids=pair_ids,
                    lane_indices=not_positive,
                )

            red = reduce_samples(s.values, conditions, samples, included=included, method=method)
            for key in red.averaged:
                if key not in averaged:
                    averaged.append(key)
            undetected: dict[str, list[int]] = {}
            for i, flag in enumerate(detected[target_id]):
                if flag is False and included[i]:
                    undetected.setdefault(conditions[i], []).append(i)
            kind = ValueKind.LOADING_NORMALIZED
            groups = red.groups
            baseline: float | None = None
            fold_change: LaneNets | None = None
            chartable = True
            if ref is not None:
                try:
                    baseline = reference_baseline(red.groups, ref)
                except BaselineError as exc:
                    chartable = False  # the napari chart skips such a series too
                    # Already explained when every reference lane is excluded, or when
                    # the series has no value at all (NO_VALUES below).
                    if not reference_all_excluded and red.groups:
                        reason = str(exc)
                        in_reference = [i for i in reference_lanes if included[i]]
                        # The target or its loading control (or both) was not
                        # detected in any included reference lane.
                        missing = [
                            name
                            for pid, name in ((target_id, s.target), (loading_id, s.loading))
                            if in_reference and all(detected[pid][i] is False for i in in_reference)
                        ]
                        if exc.reason == "no_value" and missing:
                            were = "was" if len(missing) == 1 else "were"
                            reason = (
                                f"{' and '.join(map(repr, missing))} {were} not detected in the"
                                f" reference condition {ref!r} (below the detection limit):"
                                " no fold-change can be formed"
                            )
                        note(
                            NoticeCode.REFERENCE_UNUSABLE,
                            f"{s.target!r} / {s.loading!r}: {reason}",
                            protein_ids=pair_ids,
                            conditions=(ref,),
                        )
                else:
                    fold_change = [None if v is None else v / baseline for v in s.values]
                    groups = {c: [v / baseline for v in vals] for c, vals in red.groups.items()}
                    kind = ValueKind.FOLD_CHANGE

            chart = None
            if not groups:
                note(
                    NoticeCode.NO_VALUES,
                    f"{s.target!r} / {s.loading!r} has no value in any included lane",
                    protein_ids=pair_ids,
                )
            elif chartable:
                if kind is ValueKind.FOLD_CHANGE:
                    title = f"{s.target} fold-change vs {ref}  (/{s.loading})"
                else:
                    title = f"{s.target} / {s.loading}"
                chart = _chart(
                    groups,
                    red.lanes,
                    chosen=chosen,
                    kind=kind,
                    error_type=error_type,
                    title=title,
                    reference=ref,
                    subtitle=set_label,
                )
                if chart is None:
                    note(
                        NoticeCode.NO_PLOTTED_VALUES,
                        f"{s.target!r} / {s.loading!r} has no value in the plotted conditions",
                        protein_ids=pair_ids,
                    )
            series.append(
                SeriesResult(
                    target_id=target_id,
                    target=s.target,
                    loading_id=loading_id,
                    loading=s.loading,
                    normalized=s.values,
                    value_kind=kind,
                    baseline=baseline,
                    fold_change=fold_change,
                    groups=groups,
                    averaged=red.averaged,
                    chart=chart,
                    undetected=undetected,
                )
            )

    # 8. Technical repeats, once for all series.
    if averaged:
        note(
            NoticeCode.TECHNICAL_REPEATS,
            repeats_message(len(averaged), method),
            conditions=tuple(dict.fromkeys(c for c, _ in averaged)),
        )

    return Results(
        lanes=lanes,
        tier=tier,
        reference_condition=ref,
        proteins=columns,
        series=series,
        notices=notices,
        plot_conditions=chosen,
        error_type=error_type,
        method=method,
        excluded_lanes=[i for i in range(n) if not included[i]],
        label=set_label,
    )
