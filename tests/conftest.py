# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: a sample project that exercises every part of the data model.

:func:`make_project` builds it from plain dicts, the way ``project.json`` arrives,
so the fixture itself passes through every validator. The non-ASCII names (µ, α,
β) are on purpose: they travel through validation, the saved file and the hash.

Layout: two membranes. ``mem-1`` holds a chemiluminescence image paired with its
visible-light marker and a reprobe of the same blot; ``mem-5`` holds one JPEG on
light-on-dark. The target β-catenin (``img-2``) normalizes against α-tubulin on
the other membrane; GAPDH is measured on the reprobe. Lane 2 has no β-catenin
band, and band lists are given out of order. It has no not-detected record, so
its saved form has no ``undetected`` key; :func:`make_project_with_undetected`
adds some. :func:`sample_doc_v1` is the same project as schema 1 saved it,
before #83, for the migration tests.

Image helpers for tests that read real pixels: :func:`write_tiff` and
:func:`synthetic_blot`. :class:`FakeClock` gives a session predictable log times.
:func:`assert_strict_json` checks that a model's JSON is strict and loses nothing.
"""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import tifffile
from pydantic import BaseModel

from proteia.core.model import Project, apply_change
from proteia.core.storage import image_path

MEMBRANE_LEVEL = 50000.0  # the flat membrane of synthetic_blot, in 16-bit units


class FakeClock:
    """Aware UTC time from 2026-09-26T08:00:00Z, one second later on each call."""

    def __init__(
        self,
        start: datetime = datetime(2026, 9, 26, 8, 0, tzinfo=UTC),
        step: timedelta = timedelta(seconds=1),
    ) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> datetime:
        now, self.now = self.now, self.now + self.step
        return now


def _not_json(constant: str) -> float:
    raise ValueError(f"{constant} is not JSON")


def assert_strict_json(obj: BaseModel) -> None:
    """Strict JSON with nothing lost. Pydantic writes NaN and inf as null, which
    strict JSON accepts, so only the round trip shows that none got into ``obj``."""
    dump = obj.model_dump_json()
    json.loads(dump, parse_constant=_not_json)
    assert type(obj).model_validate_json(dump) == obj


def image_bytes(image_id: str) -> bytes:
    """The stand-in pixels of a fixture image; the project records their sha256."""
    return f"pixels of {image_id}".encode()


def _image(image_id: str, suffix: str, original_name: str, kind: str, **fields) -> dict:
    return {
        "id": image_id,
        "file": image_id + suffix,
        "original_name": original_name,
        "kind": kind,
        "sha256": hashlib.sha256(image_bytes(image_id)).hexdigest(),
        **fields,
    }


def _band(band_id: str, lane: int, x: int, y: int, net: float, source: str, **fields) -> dict:
    return {
        "id": band_id,
        "lane_index": lane,
        "box": {"x": x, "y": y},
        "net": net,
        "source": source,
        **fields,
    }


def make_project() -> Project:
    """A valid project with ids 1-18 in use and ``next_id`` 19."""
    return Project.model_validate(_sample_doc())


# The content hash schema 1 gave the sample project (sample_doc_v1): what the
# log of a project saved before #83 holds.
V1_CONTENT_HASH = "28dfdbcbca235bb7359164954bf76a6d743a2c4448e51049172f42cc00b3df6a"


def as_legacy(project: Project) -> Project:
    """``project`` as a project quantified before #83 holds it (a migrated
    schema-1 project): the ``global_median`` method, each band's level at its
    image's median, with that mode and no spread. The nets are left as they are."""

    def change(draft: Project) -> None:
        draft.background_method = "global_median"
        for protein in draft.batch.proteins:
            level = draft.batch.find_image(protein.image_id).background
            for band in protein.bands:
                band.background_level = level
                band.background_mode = "global_median"
                band.background_spread = 0.0

    return apply_change(project, change)[0]


def sample_doc_v1() -> dict:
    """The sample project as a schema-1 ``project.json`` held it, before #83: no
    background method and no band background fields. Its nets are the image
    median's (global_median), and its content hash was :data:`V1_CONTENT_HASH`."""
    return _sample_doc_v1()


def _sample_doc() -> dict:
    """The sample project in the current schema: the local background method,
    each band's level at its image's median (a stand-in: the images are not
    real pixels), measured on a whole ring, with no spread."""
    doc = _sample_doc_v1()
    doc["schema_version"] = 2
    doc["background_method"] = "ring_median_v1"
    batch = doc["batch"]
    medians = {
        image["id"]: image["background"]
        for membrane in batch["membranes"]
        for image in membrane["images"]
    }
    for protein in batch["proteins"]:
        for band in protein["bands"]:
            band["background_level"] = medians[protein["image_id"]]
            band["background_mode"] = "symmetric"
            band["background_spread"] = 0.0
    return doc


def _undetected(lane: int, snr: float, region: tuple[int, int, int, int], **fields) -> dict:
    x0, y0, x1, y1 = region
    return {
        "lane_index": lane,
        "reason": "below_detection_limit",
        "snr": snr,
        "threshold": 6.0,
        "region": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "source": "row_box",
        **fields,
    }


def make_project_with_undetected() -> Project:
    """:func:`make_project` with not-detected records: β-catenin in lane 2, where
    it has no band, and GAPDH in lanes 2 and 3, given out of order (the model
    sorts them). GAPDH's lane-3 record comes from molecular-weight-guided
    detection, and its lane-2 record has a negative SNR."""
    doc = _sample_doc()
    beta, _, gapdh = doc["batch"]["proteins"]
    beta["undetected"] = [_undetected(2, 2.5, (98, 36, 128, 64))]
    gapdh["undetected"] = [
        _undetected(3, 4.125, (138, 93, 168, 121), source="mw_guided"),
        _undetected(2, -0.75, (98, 93, 128, 121)),
    ]
    return Project.model_validate(doc)


def _sample_doc_v1() -> dict:
    """The sample project as a schema-1 ``project.json`` held it (see
    :func:`sample_doc_v1`)."""
    lanes = [
        {"index": 0, "label": "vehicle", "sample": "v1"},
        {"index": 1, "label": "vehicle", "sample": "v2"},
        {"index": 2, "label": "10 µM", "sample": "a1"},
        {
            "index": 3,
            "label": "10 µM",
            "sample": "a2",
            "included": False,
            "metadata": {"dose": "10 µM", "day": "1"},
        },
    ]
    signal = {"width": 340, "height": 150, "bit_depth": 16, "polarity": "dark_on_light"}
    mem_1 = {
        "id": "mem-1",
        "images": [
            _image(
                "img-2",
                ".tif",
                "β-actin 10 µM.tif",
                "chemiluminescence",
                **signal,
                background=199.88251668003335,
                marker_image_id="img-3",
            ),
            _image(
                "img-3",
                ".png",
                "marker α.png",
                "visible_marker",
                **{**signal, "bit_depth": 8},
                background=187.25,
            ),
            _image(
                "img-4", ".tif", "reprobe β.tif", "chemiluminescence", **signal, background=201.5
            ),
        ],
        "calibration": {
            "ladder": "PageRuler Plus Prestained",
            "points": [
                {"image_id": "img-3", "y": 20.0, "mw": 250, "source": "visible_marker"},
                {"image_id": "img-3", "y": 61.5, "mw": 100, "source": "visible_marker"},
                {"image_id": "img-2", "y": 95.25, "mw": 55, "source": "chemiluminescence_marker"},
            ],
            "fit_quality": 0.998,
        },
    }
    mem_5 = {
        "id": "mem-5",
        "images": [
            _image(
                "img-6",
                ".jpg",
                "α-tubulin µ.jpg",
                "chemiluminescence",
                width=200,
                height=100,
                polarity="light_on_dark",
                background=55.11748331996667,
                import_warnings=[
                    {"code": "lossy_format", "message": "JPEG compression changes pixel values"}
                ],
            )
        ],
        # Given out of order: the model sorts points by y.
        "calibration": {
            "points": [
                {"image_id": "img-6", "y": 100, "mw": 75, "source": "strip_edge"},
                {"image_id": "img-6", "y": 0, "mw": 100, "source": "strip_edge"},
            ]
        },
    }
    proteins = [
        {
            "id": "prot-7",
            "name": "β-catenin",
            "role": "target",
            "image_id": "img-2",
            "loading_control_ids": ["prot-8"],
            "expected_mw": 92,
            "box_size": {"width": 24, "height": 14},
            # Out of order, and lane 2 has no band.
            "bands": [
                _band(
                    "band-12",
                    3,
                    138,
                    43,
                    2485.68288495034,
                    "mw_guided",
                    apparent_mw=90.5,
                    clipped=False,
                    manually_edited=True,
                ),
                _band("band-10", 0, 18, 43, 4279.740326695199, "click"),
                _band("band-11", 1, 58, 43, 4832.277498000875, "row_box"),
            ],
        },
        {
            "id": "prot-8",
            "name": "α-tubulin",
            "role": "loading control",
            "image_id": "img-6",
            "box_size": {"width": 20, "height": 10},
            # The light-on-dark α-tubulin nets of the regression baseline, shuffled.
            "bands": [
                _band("band-15", 2, 100, 40, 7822.343450967059, "row_box"),
                _band("band-13", 0, 10, 40, 7389.877572928557, "row_box"),
                _band("band-16", 3, 145, 40, 7434.837197658317, "row_box"),
                _band("band-14", 1, 55, 40, 6996.5326190520545, "row_box"),
            ],
        },
        {
            "id": "prot-9",
            "name": "GAPDH",
            "role": "loading control",
            "image_id": "img-4",
            "box_size": {"width": 24, "height": 14},
            "bands": [
                _band("band-17", 0, 18, 100, 5120.5, "click"),
                _band("band-18", 1, 58, 100, 4987.25, "click"),
            ],
        },
    ]
    return {
        "schema_version": 1,
        "next_id": 19,
        "batch": {
            "lanes": lanes,
            "reference_condition": "vehicle",
            "membranes": [mem_1, mem_5],
            "proteins": proteins,
        },
    }


@pytest.fixture
def project() -> Project:
    return make_project()


def write_image_files(folder: Path, project: Project) -> None:
    """Write each image's stand-in pixels to ``images/<file>`` under ``folder``."""
    for image in project.batch.iter_images():
        path = image_path(folder, image)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_bytes(image.id))


# --- Real pixels ---


def write_tiff(path: Path, pixels: np.ndarray) -> Path:
    """Write ``pixels`` as an uncompressed single-image TIFF (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, pixels)
    return path


def synthetic_blot(
    shape: tuple[int, int],
    bands: Sequence[tuple[float, float, float, float, float]],
    *,
    dtype: type = np.uint16,
) -> np.ndarray:
    """A flat membrane at :data:`MEMBRANE_LEVEL` darkened by Gaussian bands.

    ``shape`` is ``(height, width)``; each band is ``(cx, cy, sx, sy, depth)``:
    its centre, its horizontal and vertical 1/e half-widths in pixels, and how
    much darker its centre is. Integer types are rounded and clipped to their range.
    """
    height, width = shape
    y, x = np.mgrid[0:height, 0:width].astype(float)
    image = np.full(shape, MEMBRANE_LEVEL)
    for cx, cy, sx, sy, depth in bands:
        image -= depth * np.exp(-(((x - cx) / sx) ** 2) - ((y - cy) / sy) ** 2)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        image = np.clip(np.round(image), info.min, info.max)
    return image.astype(dtype)
