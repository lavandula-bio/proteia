# SPDX-License-Identifier: Apache-2.0
"""The reproducibility record written next to every export.

GUI-independent: the standard library plus the core. A record is derived when an
export happens and never stored in the project. It says what the exported
numbers came from and how:

* ``content``: the project content exactly as :func:`~proteia.core.storage.content_hash`
  hashes it: the background method, each image's SHA-256, median and polarity,
  each protein's box size, each band's net and background, the lane table with
  its include flags and the reference condition;
* ``content_hash``: the SHA-256 of that content, so a record verifies itself
  (``sha256(canonical_json(record["content"])) == record["content_hash"]``) with
  no Proteia and no knowledge of which keys the hash leaves out;
* ``log``: the project's action log (:class:`~proteia.core.model.LogEntry`),
  and ``history_issues``: what the log cannot vouch for;
* ``software`` and ``settings``: the versions and the code-level constants of
  the build that exported (each log entry's ``version`` names the build that
  made that change), among them how its lane tables number lanes and round
  values and at what resolution its PNG charts are drawn;
* ``files``: the SHA-256 and size of each exported file (an export bundle's
  record lists every other file of its folder, see
  :func:`~proteia.core.operations.export_bundle`);
* ``results``: the compute settings of the results the export used, or None,
  with what each chart's test resolved to (:func:`results_statistics`): the
  automatic rule may change in a later version, and the record keeps what this
  one chose.

Nothing is sent anywhere, and no platform, host name, user or path is recorded.
A record serializes canonically (sorted keys, no NaN), so a later signature can
cover ``canonical_json(record)`` without a format change.

Undo and redo are changes in the log like any other (actions ``undo`` and
``redo``), so an undone change stays in it: which changes are in effect is
read from the ``undone_seq`` and ``redone_seq`` entries. An undo or redo
restores a state whole, and its entry's ``content_hash`` equals that of the
entry it names by ``returns_to_seq``. Its params list the ids and the
not-detected record keys (protein id, lane index, band index) that went or came
back, not field changes: an object in both states whose fields changed (a moved
box, a replaced record) is audited through ``returns_to_seq``, that entry and
the content hash.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import platform
from collections.abc import Mapping
from typing import Any, Final

from pydantic import JsonValue, TypeAdapter

import proteia
from proteia.core import analyze, export, grow, quantify, rowdetect, storage
from proteia.core.model import LogEntry, Project, Timestamp, lane_number
from proteia.core.results import Results

RECORD_FORMAT: Final = 1
# The actions of entries that restore a state of the undo history.
_RESTORING_ACTIONS: Final = frozenset({"undo", "redo"})
# The libraries core computes or decodes pixels with, by distribution name.
_DISTRIBUTIONS: Final = ("numpy", "scipy", "scikit-image", "pillow", "tifffile")
_TIMESTAMP = TypeAdapter(Timestamp)


def software_versions() -> dict[str, str | None]:
    """Proteia's version, Python's, and each pixel library's, read locally;
    None for a library that is not installed."""
    versions: dict[str, str | None] = {
        "proteia": proteia.__version__,
        "python": platform.python_version(),
    }
    for dist in _DISTRIBUTIONS:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def settings() -> dict[str, JsonValue]:
    """The code-level settings behind the stored numbers and the exported tables
    and charts.

    ``background`` is the local background method this build quantifies with
    (:func:`~proteia.core.quantify.background_settings`); a record also names
    the method the project's stored nets used (:func:`build_record`)."""
    return {
        "background": quantify.background_settings(),
        # What place_box passes to grow_box: no width or height cap.
        "grow_box": {
            "rel_threshold": grow.REL_THRESHOLD,
            "noise_k": grow.NOISE_K,
            "max_width": None,
            "max_height": None,
        },
        "clipped_pixels_threshold": quantify.CLIPPED_PIXELS_THRESHOLD,
        "lane_table_decimals": export.LANE_TABLE_DECIMALS,
        "lane_table_ratio_decimals": export.LANE_TABLE_RATIO_DECIMALS,
        # The number the lane table gives the lane of stored index 0 (0 before #53).
        "lane_table_first_lane": lane_number(0),
        "chart_png_dpi": export.CHART_PNG_DPI,
        # What detect_row_boxes runs (rowdetect.detect_row with its defaults).
        "detect_row": rowdetect.settings(),
        # How the charts' tests compute: every test is two-sided.
        "statistics": {
            "alpha": analyze.ALPHA,
            "dunnett_rng_seed": analyze.DUNNETT_SEED,
            # Each chart's record names the method its test used ("method").
            "mann_whitney": (
                "exact; with tied values, the exact permutation distribution up to"
                " mann_whitney_tied_permutations arrangements, else the normal"
                " approximation with the tie correction"
            ),
            "mann_whitney_tied_permutations": analyze.MANN_WHITNEY_PERMUTATIONS,
            "kruskal_wallis_p": "chi-square approximation",
            "dunn_adjustment": "holm",
            "log_base": "e",
            "auto_rule": analyze.AUTO_RULE_VERSION,
            "tests": sorted(analyze.TESTS),
        },
    }


def results_settings(results: Results) -> dict[str, JsonValue]:
    """The compute arguments behind ``results``: they are not project state, so
    each export records the ones its numbers came from, with what each chart's
    test resolved to (:func:`results_statistics`)."""
    plotted = results.plot_conditions
    return {
        "method": results.method.value,
        "error_type": results.error_type.value,
        "plot_conditions": None if plotted is None else list(plotted),  # resolved labels
        "excluded_lanes": list(results.excluded_lanes),
        "statistics": results_statistics(results),
    }


def results_statistics(results: Results) -> dict[str, JsonValue]:
    """The statistics setting of ``results`` and, for each chart of it and of its
    all-lanes set, what the setting resolved to: the test (None when the chart
    shows none, and ``note`` says why), its family, comparisons, scale and
    reference, the conditions it covers and those it leaves out, with why, and
    which fields the user chose and which the automatic rule did (``chosen``,
    ``reasons``). This pins what ``auto`` gave even when a later version changes
    the rule. The test's p-values are here unrounded, as the chart's legend
    gives them rounded: ``p_value`` and ``statistic`` (the omnibus test's or the
    single comparison's; None for comparisons with the reference among 3
    conditions or more), every comparison in ``pairwise`` with its adjusted p
    and estimate, and ``method``, how a Mann-Whitney U test's p was computed."""
    setting = results.statistics
    chosen = {
        name: "auto" if value == "auto" else "user"
        for name, value in setting.model_dump(mode="json").items()
    }
    sets = [results] if results.all_lanes is None else [results, results.all_lanes]
    charts: list[JsonValue] = []
    for one in sets:
        for series in one.series:
            chart = series.chart
            if chart is None:
                continue
            test = chart.test
            charts.append(
                {
                    "result_set": one.label,
                    "target_id": series.target_id,
                    "loading_id": series.loading_id,
                    "test": None if test is None else test.id,
                    "family": None if test is None else test.family,
                    "comparisons": None if test is None else test.comparisons,
                    "scale": None if test is None else test.scale,
                    "reference": None if test is None else test.reference,
                    "design": None if test is None else test.design,
                    "covered": [] if test is None else list(test.covered),
                    "not_tested": [
                        {
                            "condition": c.label,
                            "n": c.n,
                            "replicates": c.replicates,
                            "not_detected": c.not_detected,
                            "reason": c.left_out,
                        }
                        for c in chart.coverage
                        if c.left_out is not None
                    ],
                    "chosen": chosen if test is None else dict(test.chosen),
                    "reasons": [] if test is None else list(test.reasons),
                    "note": None if test is not None else chart.test_note,
                    "p_value": None if test is None else test.p_value,
                    "statistic": None if test is None else test.statistic,
                    "pairwise": []
                    if test is None
                    else [pair.model_dump(mode="json") for pair in test.pairwise],
                    "method": None if test is None else test.method,
                }
            )
    return {"setting": setting.model_dump(mode="json"), "charts": charts}


def _undo_mismatch(log: tuple[LogEntry, ...]) -> bool:
    """Whether an undo or redo entry names no earlier entry by ``returns_to_seq``,
    or left other content than the entry it names."""
    left: dict[int, str] = {}  # seq -> content hash, of the entries so far
    for entry in log:
        if entry.action in _RESTORING_ACTIONS:
            seq = entry.params.get("returns_to_seq")
            if type(seq) is not int or left.get(seq) != entry.content_hash:
                return True
        left[entry.seq] = entry.content_hash
    return False


def _changed_before_migration(log: tuple[LogEntry, ...]) -> bool:
    """Whether a ``migrate`` entry started from other content than the entry
    before it left: the file was changed outside the log before it was migrated
    (:func:`proteia.core.storage.project_from_json`)."""
    return any(
        entry.action == "migrate" and entry.params.get("from_content_hash") != before.content_hash
        for before, entry in itertools.pairwise(log)
    )


def _history_issues(project: Project, digest: str) -> list[str]:
    if not project.log:
        return ["no_history"]
    issues = []
    if project.log[0].action != "new_project":
        issues.append("history_starts_late")
    if project.log[-1].content_hash != digest or _changed_before_migration(project.log):
        issues.append("content_changed_outside_log")
    if _undo_mismatch(project.log):
        issues.append("undo_mismatch")
    return issues


def history_issues(project: Project) -> list[str]:
    """What the log cannot vouch for, as stable codes (empty when it can):

    * ``no_history``: the project has no log (made before the log existed);
    * ``history_starts_late``: the log does not begin with the project's creation;
    * ``content_changed_outside_log``: the content differs from what the last
      entry left (``project.json`` edited by hand, or an unlogged change), or a
      ``migrate`` entry started from other content than the entry before it
      left (the file was edited before it was migrated to a newer schema);
    * ``undo_mismatch``: an undo or redo restored content that no earlier entry
      left, as its ``returns_to_seq`` should name (``project.json`` was edited
      by hand before the session whose undo went back past the edit).
    """
    return _history_issues(project, storage.content_hash(project))


def build_record(
    project: Project,
    *,
    exported_at: str,
    files: Mapping[str, bytes],
    results: Results | None = None,
) -> dict[str, Any]:
    """The record of an export of ``project`` (see the module docstring).

    Pure: the time is passed in (a ``Timestamp``, as ``ProjectSession.timestamp``
    gives it). ``files`` maps each exported file's fixed name to its bytes.
    """
    content = storage.content_document(project)
    digest = storage.document_hash(content)  # equals storage.content_hash(project)
    return {
        "record_format": RECORD_FORMAT,
        "exported_at": _TIMESTAMP.validate_python(exported_at, strict=True),
        "software": software_versions(),
        # The method the stored nets used: ring_median_v1 (settings()["background"])
        # or, until the project is requantified, the legacy global_median.
        "settings": {**settings(), "background_method": project.background_method},
        "content": content,
        "content_hash": digest,
        "log": [entry.model_dump(mode="json") for entry in project.log],
        "history_issues": _history_issues(project, digest),
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in files.items()
        },
        "results": None if results is None else results_settings(results),
    }


def record_bytes(record: Mapping[str, Any]) -> bytes:
    """The record file's bytes: the same form as ``project.json``."""
    return storage.document_bytes(record)
