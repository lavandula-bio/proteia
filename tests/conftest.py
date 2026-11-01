# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: a sample project that exercises every part of the data model.

:func:`make_project` builds it from plain dicts, the way ``project.json`` arrives,
so the fixture itself passes through every validator. The dicts are those of a
schema-2 file (:func:`sample_doc`), converted as an older file is when it is
read (:func:`~proteia.core.storage.migrate`), so every test of the sample also
runs the conversion. The non-ASCII names (µ, α, β) are on purpose: they travel
through validation, the saved file and the hash.

Layout: two membranes. ``mem-1`` holds a chemiluminescence image paired with its
visible-light marker and a reprobe of the same blot; ``mem-5`` holds one JPEG on
light-on-dark. The target β-catenin (``img-2``) normalizes against α-tubulin on
the other membrane; GAPDH is measured on the reprobe. Lane 2 has no β-catenin
band, and band lists are given out of order. It has no not-detected record, so
its saved form has no ``undetected`` key; :func:`make_project_with_undetected`
adds some. :func:`sample_doc_v1` is the same project as schema 1 saved it,
before #83, for the migration tests.

Image helpers for tests that read real pixels: :func:`write_tiff`,
:func:`write_lzw_tiff` (an LZW TIFF of any layout, which tifffile cannot
write without imagecodecs) and :func:`synthetic_blot`. :class:`FakeClock`
gives a session predictable log times.
:func:`assert_strict_json` checks that a model's JSON is strict and loses nothing.
"""

import functools
import hashlib
import inspect
import json
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import tifffile
from pydantic import BaseModel

from proteia.core import imaging, operations
from proteia.core.model import Project, apply_change
from proteia.core.session import ProjectSession
from proteia.core.storage import image_path, migrate


class _NoBrowser:
    """A browser that fails the test that asks for it."""

    def open(self, url, new=0, autoraise=True):
        raise AssertionError(f"a test tried to open the real browser at {url!r}")

    open_new = open_new_tab = open


@pytest.fixture(autouse=True)
def _keep_tests_off_the_desktop(monkeypatch, tmp_path_factory):
    """No test opens the user's browser or writes the user's state folder.

    ``launch.start`` takes ``webbrowser.open`` as its default opener, bound when
    the module loads, so the default browser is replaced where
    ``webbrowser.open`` looks it up: its order of browsers holds only one that
    fails the test. A browser named by its command (``webbrowser.get(command)``,
    as the packaging checks use) is still read as before. The state folder
    (``launch.state_dir``) follows ``LOCALAPPDATA`` on Windows and
    ``XDG_STATE_HOME`` elsewhere: both point at a folder of the test's own. A
    test that needs a browser passes its own opener; one that sets either
    variable itself still does."""
    import webbrowser

    guard = "proteia-test-guard"
    browsers = {**getattr(webbrowser, "_browsers", {}), guard: [None, _NoBrowser()]}
    monkeypatch.setattr(webbrowser, "_browsers", browsers)
    monkeypatch.setattr(webbrowser, "_tryorder", [guard])
    state = tmp_path_factory.mktemp("user-state")
    monkeypatch.setenv("LOCALAPPDATA", str(state))
    monkeypatch.setenv("XDG_STATE_HOME", str(state))


# The name of an image's stored copy in images/ (img-1.tif), whatever the case.
_STORED_COPY = re.compile(r"img-\d+\.(?:tiff?|png|jpe?g)", re.IGNORECASE)
# The start of a Windows path (C:\ or C:/), as written or escaped in a repr.
_DRIVE = re.compile(r"\b[A-Za-z]:[\\/]")


def stored_copy_or_path_in(message: str, original_name: str, folder: Path) -> bool:
    """Whether ``message``, a refusal of reading a file offered as
    ``original_name`` into the project at ``folder`` (on its import, or as
    an image already in the project), names an image's stored copy
    (``img-1.tif``) or shows a local path: the project folder, the folder it
    is in, or any Windows path (#234). The offered name itself may hold
    anything."""
    rest = message.replace(repr(original_name), "").replace(original_name, "")
    if _STORED_COPY.search(rest) or _DRIVE.search(rest):
        return True
    paths = {folder, folder.resolve(), folder.parent, folder.resolve().parent}
    forms = {form for path in paths for form in (str(path), repr(str(path))[1:-1])}
    return any(form in rest for form in forms)


_IMPORT_IMAGE = inspect.signature(operations.import_image)


# Words of the libraries that read a file (tifffile, Pillow, scikit-image),
# which no refusal holds (#234): their errors are logged instead.
_LIBRARY_WORDS = (
    "header=",
    "cannot identify image file",
    "unpack requires",
    "image file is truncated",
    "broken data stream",
    "failed to read",
    "suspicious number of tags",
    "corrupted IFD",
    "not a TIFF file",
    "decompression bomb",
    "can not be opened",
)


def library_words_in(error: operations.OperationError) -> bool:
    """Whether the refusal ``error`` holds a library's words: any of
    :data:`_LIBRARY_WORDS`; or, of a refusal of reading the file, the words
    of an error of its cause chain that is not Proteia's own (a system
    error's reason alone may stay, which names no path)."""
    message = str(error)
    if any(words in message for words in _LIBRARY_WORDS):
        return True
    if error.code is not operations.ErrorCode.UNREADABLE_IMAGE:
        return False  # storing the file, in Proteia's words (an empty file)
    cause = error.__cause__
    while cause is not None:
        own = isinstance(cause, operations.OperationError | imaging.ReadingRefusal)
        if not own and str(cause) and str(cause) in message:
            return True
        cause = cause.__cause__
    return False


def _check_refusal(what: str, error: operations.OperationError, name: object, folder) -> None:
    """Fail ``what``, the refusal ``error`` of reading a file offered as
    ``name`` into the project at ``folder``, if it names a stored copy,
    shows a path or holds a library's words."""
    if isinstance(name, str) and folder is not None:
        where = stored_copy_or_path_in(str(error), name, folder)
        assert not where, f"{what} names a stored copy or a path: {error}"
    assert not library_words_in(error), f"{what} holds a library's words: {error}"


@pytest.fixture(autouse=True)
def _import_refusals_name_the_file_as_chosen(monkeypatch):
    """No refusal of reading a file that any test meets names the image's
    stored copy, shows a local path or holds a library's words (#234),
    whatever else the test checks: each refusal of
    :func:`~proteia.core.operations.import_image`, and since #246 each of
    reading an image already in the project
    (:meth:`~proteia.core.session.ProjectSession._read_checked`: its pixels,
    on any channel, and its colours), is checked as it is raised."""
    real = operations.import_image

    @functools.wraps(real)
    def import_image(*args, **kwargs):
        try:
            return real(*args, **kwargs)
        except operations.OperationError as exc:
            call = _IMPORT_IMAGE.bind(*args, **kwargs)
            session = call.arguments["session"]
            folder = getattr(session, "folder", None)
            _check_refusal("an import refusal", exc, call.arguments["original_name"], folder)
            raise

    real_read = ProjectSession._read_checked

    @functools.wraps(real_read)
    def read_checked(self, image, read):
        try:
            return real_read(self, image, read)
        except operations.OperationError as exc:
            what = "a refusal of reading an imported image"
            _check_refusal(what, exc, image.original_name, self.folder)
            raise

    monkeypatch.setattr(operations, "import_image", import_image)
    monkeypatch.setattr(ProjectSession, "_read_checked", read_checked)


def guards_in_place() -> bool:
    """Whether the guards of the autouse fixtures above are all in place, as
    a test that takes its own patches back checks it took back only its own
    (``monkeypatch.undo()`` takes these back too): the refusal checks wrap
    the import and the session's reads, the browser that fails the test is
    the only one, and the state folder is the test's own."""
    import webbrowser

    state = [os.environ.get(name, "") for name in ("LOCALAPPDATA", "XDG_STATE_HOME")]
    return (
        hasattr(operations.import_image, "__wrapped__")
        and hasattr(ProjectSession._read_checked, "__wrapped__")
        and getattr(webbrowser, "_tryorder", None) == ["proteia-test-guard"]
        and all(Path(folder).name.startswith("user-state") for folder in state)
    )


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


def converted(doc: dict) -> Project:
    """The project an older ``project.json`` holding ``doc`` opens as: converted
    to the current schema, then validated."""
    return Project.model_validate(migrate(doc))


def make_project() -> Project:
    """The sample project (:func:`sample_doc`), converted: ids 1-18 as the
    schema-2 file held them, then its lane table ``lt-19``, the membrane areas
    ``arena-20`` to ``arena-23`` (one per image) and the rows ``row-24`` to
    ``row-26`` (one per protein); ``next_id`` 27."""
    return converted(_sample_doc())


def sample_doc() -> dict:
    """The sample project as a schema-2 ``project.json`` holds it: ids 1-18 in
    use and ``next_id`` 19."""
    return _sample_doc()


# The content hash schema 1 gave the sample project (sample_doc_v1): what the
# log of a project saved before #83 holds.
V1_CONTENT_HASH = "28dfdbcbca235bb7359164954bf76a6d743a2c4448e51049172f42cc00b3df6a"


def as_legacy(project: Project) -> Project:
    """``project`` as a project quantified before #83 holds it (a migrated
    schema-1 project): the ``global_median`` method, each band's level at its
    image's median, with that mode and no spread. The nets are left as they are."""

    def change(draft: Project) -> None:
        draft.background_method = "global_median"
        batch = draft.batch
        for protein in batch.proteins:
            level = batch.image_of(protein).background
            for band in batch.boxes_of(protein).bands:
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
    """:func:`make_project` with not-detected records (:func:`undetected_doc`)."""
    return converted(undetected_doc())


def undetected_doc() -> dict:
    """:func:`sample_doc` with not-detected records: β-catenin in lane 2, where
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
    return doc


def make_project_with_clashing_names() -> Project:
    """:func:`make_project` with clashing names (:func:`clashing_names_doc`)."""
    return converted(clashing_names_doc())


def clashing_names_doc() -> dict:
    """:func:`sample_doc` with protein names whose lane-table columns would
    share a name, as a ``project.json`` may hold them (the operations refuse
    them): β-catenin named ``GAPDH`` and GAPDH named ``GAPDH clipped``, a target
    of α-tubulin, whose net column the first one's clipping column would name.
    Every band of both was checked for clipping: ``GAPDH`` is clipped in lane
    1 (index 0), ``GAPDH clipped`` in lane 2 (index 1)."""
    doc = _sample_doc()
    first, _, second = doc["batch"]["proteins"]
    first["name"] = "GAPDH"
    second.update(name="GAPDH clipped", role="target", loading_control_ids=["prot-8"])
    for protein, clipped_lane in ((first, 0), (second, 1)):
        for band in protein["bands"]:
            band["clipped"] = band["lane_index"] == clipped_lane
    return doc


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


_LZW_CLEAR, _LZW_END, _LZW_FIRST = 256, 257, 258


def lzw_encode(data: bytes) -> bytes:
    """``data`` compressed as TIFF LZW, in 9-bit codes only: a Clear code
    starts the table again before it reaches 511 entries, where a decoder
    would widen its codes to 10 bits."""
    codes = [_LZW_CLEAR]
    table: dict[bytes, int] = {}
    prefix = b""
    for value in data:
        joined = prefix + bytes((value,))
        if not prefix or joined in table:
            prefix = joined
            continue
        codes.append(table[prefix] if len(prefix) > 1 else prefix[0])
        table[joined] = _LZW_FIRST + len(table)
        prefix = joined[-1:]
        if _LZW_FIRST + len(table) == 511:
            codes.append(_LZW_CLEAR)
            table.clear()
    if prefix:
        codes.append(table[prefix] if len(prefix) > 1 else prefix[0])
    codes.append(_LZW_END)
    bits = "".join(format(code, "09b") for code in codes)
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def write_lzw_tiff(
    path: Path,
    pixels: np.ndarray,
    *,
    planar: bool = False,
    byteorder: str = "<",
    rowsperstrip: int | None = None,
    tile: tuple[int, int] | None = None,
    predictor: bool = False,
    compression: int = tifffile.COMPRESSION.LZW,
    **options,
) -> Path:
    """Write ``pixels`` (2-D, or channels last) as an LZW-compressed TIFF, in
    any layout tifffile writes uncompressed with the same arguments (``options``
    such as ``photometric`` and ``extrasamples``): planar, of either byte order,
    in strips or tiles, with the horizontal predictor. tifffile cannot encode
    LZW without imagecodecs, so it writes the strips (or tiles) encoded here as
    Deflate, and the Compression tag is then set to LZW, or to another
    ``compression`` for a test of what the header alone decides."""
    data = np.asarray(pixels)
    data = data.astype(data.dtype.newbyteorder(byteorder))
    stored = np.moveaxis(data, -1, 0) if planar else data
    planes = list(stored) if planar else [stored]
    height, width = planes[0].shape[:2]
    rows = rowsperstrip or height
    chunks = []
    for plane in planes:
        if tile:
            across, down = -(-width // tile[1]) * tile[1], -(-height // tile[0]) * tile[0]
            padded = np.zeros((down, across, *plane.shape[2:]), plane.dtype)
            padded[:height, :width] = plane
            pieces = [
                padded[y : y + tile[0], x : x + tile[1]]
                for y in range(0, down, tile[0])
                for x in range(0, across, tile[1])
            ]
        else:
            pieces = [plane[top : top + rows] for top in range(0, height, rows)]
        for piece in pieces:
            if predictor:  # each sample less the same sample of the pixel on its left
                original, piece = piece, piece.copy()
                piece[:, 1:] -= original[:, :-1]  # wrapping round, as unsigned integers do
            chunks.append(lzw_encode(np.ascontiguousarray(piece).tobytes()))
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        iter(chunks),
        shape=stored.shape,
        dtype=stored.dtype,
        byteorder=byteorder,
        compression="zlib",
        predictor=predictor or None,
        planarconfig="separate" if planar else None,
        rowsperstrip=None if tile else rows,
        tile=tile,
        **options,
    )
    with tifffile.TiffFile(path) as tif:
        offset = tif.pages.first.tags["Compression"].valueoffset
    with path.open("r+b") as file:
        file.seek(offset)
        file.write(int(compression).to_bytes(2, "little" if byteorder == "<" else "big"))
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
