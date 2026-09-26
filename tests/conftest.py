# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: a sample project that exercises every part of the data model.

:func:`make_project` builds it from plain dicts, the way ``project.json`` arrives,
so the fixture itself passes through every validator. The non-ASCII names (µ, α,
β) are on purpose: they travel through validation, the saved file and the hash.

Layout: two membranes. ``mem-1`` holds a chemiluminescence image paired with its
visible-light marker and a reprobe of the same blot; ``mem-5`` holds one JPEG on
light-on-dark. The target β-catenin (``img-2``) normalizes against α-tubulin on
the other membrane; GAPDH is measured on the reprobe. Lane 2 has no β-catenin
band, and band lists are given out of order.

Image helpers for tests that read real pixels: :func:`write_tiff` and
:func:`synthetic_blot`. :class:`FakeClock` gives a session predictable log times.
"""

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import tifffile

from proteia.core.model import Project
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
    return Project.model_validate(
        {
            "schema_version": 1,
            "next_id": 19,
            "batch": {
                "lanes": lanes,
                "reference_condition": "vehicle",
                "membranes": [mem_1, mem_5],
                "proteins": proteins,
            },
        }
    )


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
