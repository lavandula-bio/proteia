# SPDX-License-Identifier: Apache-2.0
"""File exports that do not need a GUI.

Text files are written as UTF-8. CSV uses ``utf-8-sig`` (UTF-8 with a byte-order
mark) so Excel detects the encoding: condition, sample, and protein names are
typed by the user and often contain characters such as µ, α, or β, which the
locale code page on Windows (e.g. cp950) either cannot encode or encodes in a
way other machines misread.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from proteia.core.analyze import LaneNets

CSV_ENCODING = "utf-8-sig"
# The lane-identity columns that lead the lane table, before one column per protein.
LANE_COLUMNS: Final = ("lane", "condition", "sample", "include")


def write_lane_table(
    path: str | Path,
    conditions: Sequence[str],
    samples: Sequence[str | None],
    included: Sequence[bool],
    proteins: Sequence[tuple[str, LaneNets]],
    *,
    clipped: Mapping[str, Sequence[bool | None]] | None = None,
) -> None:
    """Write the raw per-lane table: lane identity plus each protein's net.

    One row per lane with ``lane, condition, sample, include`` and one column per
    protein, in the order given. A missing net (no box for that protein on that
    lane) is an empty cell; nets are rounded to three decimals. With ``clipped``
    (protein name to per-lane flags), each such protein's net column is followed
    by a ``<name> clipped`` column: ``yes`` for an over-exposed band, ``no``, or
    empty when there is no box or the band was not checked.
    """
    n = len(conditions)
    if len(samples) != n or len(included) != n:
        raise ValueError("conditions, samples, and included must have the same length")
    flags = clipped or {}
    for name, nets in proteins:
        if len(nets) != n:
            raise ValueError(f"protein {name!r} has {len(nets)} nets but there are {n} lanes")
        if name in flags and len(flags[name]) != n:
            raise ValueError(
                f"protein {name!r} has {len(flags[name])} clipping flags but there are {n} lanes"
            )
    unknown = set(flags) - {name for name, _ in proteins}
    if unknown:
        raise ValueError(f"clipping flags for proteins not in the table: {sorted(unknown)}")

    with Path(path).open("w", encoding=CSV_ENCODING, newline="") as fh:
        writer = csv.writer(fh)
        header = list(LANE_COLUMNS)
        for name, _ in proteins:
            header += [name, f"{name} clipped"] if name in flags else [name]
        if len(set(header)) != len(header):
            raise ValueError(f"two lane-table columns would share a name: {header}")
        writer.writerow(header)
        for i in range(n):
            row = [i, conditions[i], samples[i] or "", "yes" if included[i] else "no"]
            for name, nets in proteins:
                row.append("" if nets[i] is None else round(nets[i], 3))
                if name in flags:
                    flag = flags[name][i]
                    row.append("" if flag is None else "yes" if flag else "no")
            writer.writerow(row)
