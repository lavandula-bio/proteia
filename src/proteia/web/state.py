# SPDX-License-Identifier: Apache-2.0
"""What the browser draws: the open project as JSON, and image previews.

Coordinates are image pixels: a box is ``[x0, y0, x1, y1]`` with the end
exclusive, as :meth:`~proteia.core.model.Box.rect` gives it, so a box the browser
draws at any zoom lands on the same pixels the server quantifies.

Each protein's ``undetected`` lists its not-detected records
(:class:`~proteia.core.model.UndetectedBand`), in lane order, for the view to
mark: ``lane_index``, ``band_index``, ``reason``, ``snr`` and ``threshold`` (the
detector's statistic and the limit it stayed below), ``region`` (the slot the
detector measured, ``[x0, y0, x1, y1]`` like a box) and ``source`` (the
detector). A lane with a first-band record was examined, so it is not offered as
a missing box.
"""

from __future__ import annotations

import io
import statistics

import numpy as np
from PIL import Image
from pydantic import JsonValue

from proteia.core.imaging import preview
from proteia.core.model import Batch, Project, Protein
from proteia.core.project import lane_anchors, lane_positions
from proteia.core.session import ProjectSession


def _missing_lanes(
    batch: Batch, protein: Protein, anchors: list[tuple[float, int]]
) -> list[JsonValue]:
    """The declared lanes where ``protein`` has neither a first-band box nor a
    first-band not-detected record (that lane was examined), each with where its
    box is expected: the lane's centre x from ``anchors``, the boxes already on
    its image (:func:`~proteia.core.project.lane_positions`), and the protein's
    row (the median centre y of its first bands); None where that cannot be
    known yet."""
    first = [band for band in protein.bands if band.band_index == 0]
    has = {band.lane_index for band in first}
    has.update(record.lane_index for record in protein.undetected if record.band_index == 0)
    lanes = [lane.index for lane in batch.lanes if lane.index not in has]
    if not lanes:
        return []
    xs = lane_positions(anchors, lanes)
    centres = [band.box.y + protein.box_size.height / 2 for band in first]
    y = statistics.median(centres) if centres else None
    return [{"lane_index": lane, "x": xs.get(lane), "y": y} for lane in lanes]


def project_state(name: str, session: ProjectSession) -> dict[str, JsonValue]:
    """The open project ``name`` as the web UI draws it."""
    project: Project = session.project  # one snapshot for the whole answer
    batch = project.batch
    images: list[JsonValue] = [
        {
            "id": image.id,
            "membrane_id": membrane.id,
            "original_name": image.original_name,
            "kind": image.kind.value,
            "polarity": image.polarity.value,
            "width": image.width,
            "height": image.height,
            "bit_depth": image.bit_depth,
            "warnings": [
                {"code": warning.code, "message": warning.message}
                for warning in image.import_warnings
            ],
        }
        for membrane in batch.membranes
        for image in membrane.images
    ]
    anchors = {image.id: lane_anchors(batch, image) for image in batch.iter_images()}
    proteins: list[JsonValue] = []
    for protein in batch.proteins:
        size = protein.box_size
        proteins.append(
            {
                "id": protein.id,
                "name": protein.name,
                "role": protein.role.value,
                "image_id": protein.image_id,
                "box_size": {"width": size.width, "height": size.height},
                "bands": [
                    {
                        "id": band.id,
                        "lane_index": band.lane_index,
                        "band_index": band.band_index,
                        "rect": list(band.box.rect(size)),
                        "clipped": band.clipped,
                        "manually_edited": band.manually_edited,
                    }
                    for band in protein.bands
                ],
                "undetected": [
                    {
                        "lane_index": record.lane_index,
                        "band_index": record.band_index,
                        "reason": record.reason.value,
                        "snr": record.snr,
                        "threshold": record.threshold,
                        "region": list(record.region.rect()),
                        "source": record.source.value,
                    }
                    for record in protein.undetected
                ],
                "missing_lanes": _missing_lanes(batch, protein, anchors[protein.image_id]),
            }
        )
    return {
        "name": name,
        "lanes": [
            {
                "index": lane.index,
                "condition": lane.label,
                "sample": lane.sample,
                "included": lane.included,
            }
            for lane in batch.lanes
        ],
        "reference_condition": batch.reference_condition,
        "images": images,
        "proteins": proteins,
        "saved": not session.dirty,
        "save_error": None if session.save_error is None else str(session.save_error),
    }


def preview_png(array: np.ndarray) -> bytes:
    """A grayscale PNG of an analysis array for display
    (:func:`~proteia.core.imaging.preview`: 16-bit levels stretched to 8 bits)."""
    buffer = io.BytesIO()
    Image.fromarray(preview(array)).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()
