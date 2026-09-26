# SPDX-License-Identifier: Apache-2.0
"""What the browser shows of the results: :class:`~proteia.core.results.Results`
as JSON. A translation only; every number comes from the compute step.

* Strict JSON: the results go through ``model_dump_json``, which writes NaN and
  inf as ``null``, so an answer never fails to encode and the browser never
  parses a bare ``NaN``.
* Mappings keyed by condition become ordered lists (``groups`` as
  ``[{condition, values}]``, ``undetected`` as ``[{condition, lanes}]``, in lane
  order), since a JS object puts keys that look like numbers ("5", "10") first,
  in numeric order. ``averaged`` becomes ``[{condition, sample}]``.
* Each chart comparison carries its ``stars``.
* The result sets become a ``sets`` list: ``applied`` (the lane table's
  include flags), then ``all_lanes`` when excluded lanes hold values
  (:attr:`~proteia.core.results.Results.all_lanes`). Each set has its ``label``,
  ``excluded_lanes`` (0-based), ``tier``, ``notices`` and ``series``. The
  ``all_lanes`` set's notices are only those the applied set does not already
  carry, as in the core results; what applies to it is both lists. The lanes,
  the proteins' per-lane columns and the settings are the same in both sets and
  appear once, with the lane table's own include flags. Element i of every
  per-lane list belongs to ``lanes[i]``, excluded lanes included.
* Each series has a ``chart_url``: the chart store's URL for its chart, or
  ``null`` without a store or a chart.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import JsonValue

from proteia.core.plotspec import PlotSpec
from proteia.core.results import Results, SeriesResult

ChartStore = Callable[[PlotSpec], str]  # registers a chart; gives the URL it is drawn at


def results_payload(
    results: Results, *, open_id: int, revision: int, charts: ChartStore | None = None
) -> dict[str, JsonValue]:
    """``results`` as the web UI reads them, stamped with the ``open_id`` and
    ``revision`` of the project they were computed from."""
    data = json.loads(results.model_dump_json())  # NaN and inf become null
    sets = [_result_set("applied", results, data, charts)]
    if results.all_lanes is not None:
        sets.append(_result_set("all_lanes", results.all_lanes, data["all_lanes"], charts))
    return {
        "open_id": open_id,
        "revision": revision,
        "settings": {
            "error_type": data["error_type"],
            "plot_conditions": data["plot_conditions"],
            "method": data["method"],
        },
        "reference_condition": data["reference_condition"],
        "lanes": data["lanes"],
        "proteins": data["proteins"],
        "sets": sets,
    }


def _result_set(
    set_id: str, results: Results, data: dict[str, Any], charts: ChartStore | None
) -> dict[str, JsonValue]:
    return {
        "id": set_id,
        "label": data["label"],
        "excluded_lanes": data["excluded_lanes"],
        "tier": data["tier"],
        "notices": data["notices"],
        "series": [
            _series(series, dumped, charts)
            for series, dumped in zip(results.series, data["series"], strict=True)
        ],
    }


def _series(
    series: SeriesResult, data: dict[str, Any], charts: ChartStore | None
) -> dict[str, JsonValue]:
    """One series; its model gives the order of the condition keys and the stars."""
    chart = data["chart"]
    if series.chart is not None:
        chart["comparisons"] = [
            {**dumped, "stars": comparison.stars}
            for comparison, dumped in zip(
                series.chart.comparisons, chart["comparisons"], strict=True
            )
        ]
    return {
        **data,
        "groups": [
            {"condition": condition, "values": data["groups"][condition]}
            for condition in series.groups
        ],
        "averaged": [
            {"condition": condition, "sample": sample} for condition, sample in series.averaged
        ],
        "undetected": [
            {"condition": condition, "lanes": data["undetected"][condition]}
            for condition in series.undetected
        ],
        "chart": chart,
        "chart_url": None if charts is None or series.chart is None else charts(series.chart),
    }
