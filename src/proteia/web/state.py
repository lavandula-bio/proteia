# SPDX-License-Identifier: Apache-2.0
"""What the browser draws: the open project as JSON, and image previews.

The state is built from one snapshot of the committed project, the one its
results were computed from (:func:`~proteia.core.operations.compute_view`), and
carries ``open_id`` and ``revision``: which opening of a project it belongs to
(the workspace counts every create, and every open of a project not already
open: reopening the open one answers its own, unless it reads a
``project.json`` changed outside Proteia again) and the ``seq`` of the project's
last log entry (0 with an empty log). Every commit appends one entry, so within
one opening the revision names one state, and a client can keep the newest
answer by the pair and drop an older one that arrives late. The revision alone
would not do: another project may be opened with a shorter log.

Coordinates are image pixels: a box is ``[x0, y0, x1, y1]`` with the end
exclusive, as :meth:`~proteia.core.model.Box.rect` gives it, so a box the browser
draws at any zoom lands on the same pixels the server quantifies.

Each protein's ``box_size`` is the size of every one of its boxes, the one
quantified and drawn, and ``fitted_size`` the size its clicks, rows or typing
asked for, which the boxes extend beyond by the protein's padding on each side
(:attr:`~proteia.core.model.Protein.fitted_size`), both as ``{width, height}``;
``box_padding`` is that padding, as ``{across, along}``: whole pixels left and
right, and above and below. The page's size fields show the fitted size, which
``PUT /api/proteins/{id}/box-size`` takes, and its padding fields the padding,
which ``PUT /api/proteins/{id}/box-padding`` takes.

``background_method`` names how the stored nets' backgrounds were measured
(``ring_median_v1``, or ``global_median`` for a project quantified before #83,
until it is requantified), and each band's ``background_mode`` how its own was:
``symmetric``, ``asymmetric`` or ``image`` (a ring cut short), or
``global_median``. Each band's ``clipped`` says whether it is over-exposed
(null: not checked), and ``possibly_clipped`` whether it looks so on an image
that check cannot trust (null: not assessed; #112). ``unassessed_images`` lists
the images whose bands were measured before Proteia looked for pixels near the
detector limit, which ``POST /api/requantify`` assesses
(:func:`~proteia.core.operations.unassessed_images`).

Each protein's ``undetected`` lists its not-detected records
(:class:`~proteia.core.model.UndetectedBand`), in lane order, for the view to
mark: ``lane_index``, ``band_index``, ``reason``, ``snr`` and ``threshold`` (the
detector's statistic and the limit it stayed below), ``region`` (the slot the
detector measured, ``[x0, y0, x1, y1]`` like a box) and ``source`` (the
detector). A lane with a first-band record was examined, so it is not offered as
a missing box.

Each image's ``colour`` says whether its stored file has colour that its gray
analysis view does not show (:func:`has_colour`): then the page offers a view of
it in its original colours (:func:`original_png`), for display only. Only that
reads a stored file here: a header, once per image (TIFF, or a colour file).

Molecular weights (#58). Each image's ``marker_image_id`` names the marker image
a chemiluminescence image is linked to (null: none), and each band's
``apparent_mw`` its MW read from its image's calibration at the box's centre
(null: no curve there, or outside its range). ``membranes`` lists each
membrane's ``id``, its ``image_ids``, its ``ladder`` (a preset key or a custom
name, or null) and ``ladder_kda`` (the ladder's MWs, top to bottom), and its
register groups (``groups``), in the order of their first image: each group's
``image_ids``, its calibration ``points`` (``image_id``, ``y``, ``mw``,
``source``, ``x`` and ``side``, in continuous coordinates of the analysis
array) and its ``fit`` (:meth:`~proteia.core.operations.CalibrationFit.as_json`,
null without a curve).
"""

from __future__ import annotations

import io
import statistics

import numpy as np
from PIL import Image
from pydantic import JsonValue

from proteia.core.imaging import TIFF_SUFFIXES, display_rgb, preview
from proteia.core.model import Batch, ImageRef, Membrane, Project, Protein
from proteia.core.operations import calibration_fit, unassessed_images
from proteia.core.project import lane_anchors, lane_positions
from proteia.core.session import HistoryStep, ProjectSession


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


def _membrane(membrane: Membrane) -> JsonValue:
    """A membrane's ladder and its register groups, each with its calibration
    points and its fit."""
    calibration = membrane.calibration
    groups: list[JsonValue] = []
    for group in membrane.register_groups():
        image_ids = [image.id for image in membrane.images if image.id in group]
        fit = calibration_fit(membrane, image_ids[0])
        groups.append(
            {
                "image_ids": list(image_ids),
                "points": [
                    {
                        "image_id": point.image_id,
                        "y": point.y,
                        "mw": point.mw,
                        "source": point.source.value,
                        "x": point.x,
                        "side": point.side.value,
                    }
                    for point in calibration.points
                    if point.image_id in group
                ],
                "fit": None if fit is None else fit.as_json(),
            }
        )
    return {
        "id": membrane.id,
        "image_ids": [image.id for image in membrane.images],
        "ladder": calibration.ladder,
        "ladder_kda": list(calibration.ladder_kda),
        "groups": groups,
    }


def revision(project: Project) -> int:
    """The ``seq`` of the project's last log entry; 0 with an empty log."""
    return project.log[-1].seq if project.log else 0


def _step(step: HistoryStep | None) -> JsonValue:
    return None if step is None else {"seq": step.seq, "action": step.action}


def project_state(
    name: str, session: ProjectSession, project: Project, *, open_id: int
) -> dict[str, JsonValue]:
    """The open project ``name`` as the web UI draws it, from the snapshot
    ``project`` of ``session``; ``open_id`` names this opening of it. Only the
    save status and the undo history are read from the session itself, without
    waiting for a running operation (so in a race they may be of a later commit
    than the snapshot): ``history`` names the change undo would take back and
    the one redo would make again, each as its log entry's ``seq`` and
    ``action`` (null when there is none)."""
    undo, redo = session.history_steps
    batch = project.batch
    images: list[JsonValue] = [
        {
            "id": image.id,
            "membrane_id": membrane.id,
            "original_name": image.original_name,
            "kind": image.kind.value,
            "polarity": image.polarity.value,
            "marker_image_id": image.marker_image_id,
            "width": image.width,
            "height": image.height,
            "bit_depth": image.bit_depth,
            "colour": has_colour(session, image),
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
        size, fitted, padding = protein.box_size, protein.fitted_size, protein.box_padding
        proteins.append(
            {
                "id": protein.id,
                "name": protein.name,
                "role": protein.role.value,
                "image_id": protein.image_id,
                "loading_control_ids": list(protein.loading_control_ids),  # the series order
                "expected_mw": protein.expected_mw,
                "box_size": {"width": size.width, "height": size.height},
                "fitted_size": {"width": fitted.width, "height": fitted.height},
                "box_padding": {"across": padding.across, "along": padding.along},
                "bands": [
                    {
                        "id": band.id,
                        "lane_index": band.lane_index,
                        "band_index": band.band_index,
                        "rect": list(band.box.rect(size)),
                        "apparent_mw": band.apparent_mw,
                        "clipped": band.clipped,
                        "possibly_clipped": band.possibly_clipped,
                        "background_mode": band.background_mode,
                        "source": band.source.value,
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
        "open_id": open_id,
        "revision": revision(project),
        "history": {"undo": _step(undo), "redo": _step(redo)},
        "background_method": project.background_method,
        "unassessed_images": unassessed_images(batch),
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
        "membranes": [_membrane(membrane) for membrane in batch.membranes],
        "proteins": proteins,
        "saved": not session.dirty,
        "save_error": None if session.save_error is None else str(session.save_error),
    }


def has_colour(session: ProjectSession, image: ImageRef) -> bool:
    """Whether the image's stored file has colour its gray analysis array does not
    show, so the page offers its original colours:

    * red, green and blue that differ, which the import records as the
      ``color_channels_differ`` warning (the gray is their mean); a palette PNG
      reads as its colours, so a colour palette counts;
    * a palette TIFF's colour map that is not gray: its pixels read as the
      indices (the gray levels, as an ImageJ lookup table colours them).

    A CMYK file is read converted to red, green and blue, which its gray is
    the mean of, so it is offered on the same terms, in those colours
    (approximate, as its ``cmyk_converted`` warning says); files in other
    colour spaces (CIELAB, YCbCr) are refused on import. Equal channels are
    gray, and so is gray with alpha. The stored file's header is read for a
    TIFF, or a file with the warning
    (:meth:`~proteia.core.session.ProjectSession.file_colours`, once each)."""
    differ = any(warning.code == "color_channels_differ" for warning in image.import_warnings)
    if not differ and not image.file.endswith(TIFF_SUFFIXES):
        return False  # gray: only a TIFF has colours its pixels do not read as (a palette)
    colours = session.file_colours(image)
    return colours == "palette" or (colours == "rgb" and differ)


def _png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def preview_png(array: np.ndarray) -> bytes:
    """A grayscale PNG of an analysis array for display
    (:func:`~proteia.core.imaging.preview`: 16-bit levels stretched to 8 bits)."""
    return _png(preview(array))


def original_png(pixels: np.ndarray) -> bytes:
    """An RGB PNG of an image's pixels in its file's own colours
    (:func:`~proteia.core.imaging.read_colours`), for display only
    (:func:`~proteia.core.imaging.display_rgb`: alpha dropped, levels
    other than 8-bit stretched to 8 bits over all three channels at once, so
    the hues keep their balance). Pixel for pixel the analysis array's grid:
    same size, same orientation, so boxes drawn on it stay in place."""
    return _png(display_rgb(pixels))
