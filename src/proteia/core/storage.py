# SPDX-License-Identifier: Apache-2.0
"""The project folder: save, load, the content hash and the image store.

GUI-independent: the standard library plus the model (and pydantic, through its
types). A project folder, which may have any name, holds::

    project.json        the model (see "Canonical form" below)
    images/img-2.tif    byte-identical copies of the imported files, named by image id
    exports/            created by save_project, for exported results

Stored names come only from validated ids plus a whitelisted suffix; an image's
``original_name`` never becomes a path. A crash can leave orphan files (a temp
file, or an image no saved project references) but never a dangling reference:
an image is stored before the model references it, and a file is deleted only
after a saved ``project.json`` no longer references it.

Canonical form. ``project.json`` is ``json.dumps`` of the re-validated model's
JSON-mode dump with sorted keys, ``indent=1``, ``ensure_ascii=False`` and
``allow_nan=False``, plus a trailing newline, encoded as UTF-8 with no BOM and
written in binary, so every platform gets LF and raw UTF-8 for µ, α and β. The
content hash is the lowercase hex SHA-256 of :func:`canonical_json` (the same
dump written compactly) without the keys in :data:`HASH_EXCLUDE`. In detail:

* Floats are written by the standard library's ``float.__repr__``, the shortest
  form that round-trips exactly. pydantic-core's serializer writes some floats
  differently (``1e-7`` where Python writes ``1e-07``), so ``model_dump_json`` is
  never used: a library upgrade must not move the hash.
* Sorted keys make the bytes independent of field order and of the insertion
  order of ``Lane.metadata``. The model sorts bands and calibration points; the
  order of membranes, images and proteins is content and changes the hash.
* The hash covers ``schema_version``, every id, the lane table, the image records
  (and so the pixels, through each ``sha256``), the calibrations, and the
  proteins and bands with their stored nets and flags. It excludes ``next_id``
  and the file's formatting, and it is not stored in the project.
* Loading a file Proteia wrote and saving it again gives the same bytes, because
  the strict load keeps every type exactly.

The canonical form follows Python's float repr, not RFC 8785 (JSON
Canonicalization Scheme); switch, with a schema bump, before hashes must be
reproduced outside Python.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Final, NoReturn

from pydantic import TypeAdapter, ValidationError

from proteia.core.model import (
    IMAGE_SUFFIXES,
    SCHEMA_VERSION,
    ImageId,
    ImageRef,
    OriginalName,
    Project,
    revalidate,
)

PROJECT_FILE: Final = "project.json"
IMAGES_DIR: Final = "images"
EXPORTS_DIR: Final = "exports"
# Top-level keys left out of the content hash: bookkeeping, not content.
HASH_EXCLUDE: Final = frozenset({"next_id"})
# A reader holding project.json (antivirus, indexer, OneDrive) makes the replace
# fail on Windows: retry with a doubling delay (about 0.75 s in all).
REPLACE_ATTEMPTS = 5
REPLACE_DELAY = 0.05  # seconds; tests set it to 0
_CHUNK: Final = 1 << 20  # store_image copies 1 MiB at a time

_IMAGE_ID = TypeAdapter(ImageId)
_ORIGINAL_NAME = TypeAdapter(OriginalName)


class ProjectError(Exception):
    """Base for project-folder content problems."""


class ProjectFormatError(ProjectError):
    """project.json is not UTF-8, not JSON, has a duplicate key or NaN/Infinity, is
    not an object, or fails validation (the pydantic ValidationError is __cause__)."""


class SchemaVersionError(ProjectFormatError):
    """project.json has a schema version this Proteia cannot read."""

    def __init__(self, message: str, *, found: object, supported: int) -> None:
        super().__init__(message)
        self.found = found
        self.supported = supported


class MissingImageError(ProjectError):
    """Image files the project references are missing from ``images/``."""

    def __init__(self, image_ids: list[str]) -> None:
        super().__init__(f"missing image files for {', '.join(image_ids)}")
        self.image_ids = list(image_ids)


class ImageTooLargeError(ValueError):
    """store_image: the stream exceeded max_bytes."""


@dataclass(frozen=True)
class StoredImage:
    """Where :func:`store_image` put a file, and what it holds."""

    file: str  # name inside images/
    sha256: str
    size: int  # bytes


# --- Serialization and the content hash ---


def canonical_json(obj: object) -> bytes:
    """The compact canonical form: the input to every hash."""
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def project_to_json(project: Project) -> bytes:
    """The exact ``project.json`` bytes. Raises ``ValidationError`` if an in-place
    edit made the project invalid."""
    doc = revalidate(project).model_dump(mode="json")
    text = json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False)
    return (text + "\n").encode("utf-8")


def content_hash(project: Project) -> str:
    """Lowercase hex SHA-256 of the project's content (see the module docstring)."""
    doc = revalidate(project).model_dump(mode="json", exclude=set(HASH_EXCLUDE))
    return hashlib.sha256(canonical_json(doc)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not allowed")


def project_from_json(data: bytes) -> Project:
    """Parse and validate ``project.json`` bytes, migrating an older schema.

    A BOM (from a Windows editor) and CRLF are tolerated; duplicate keys,
    NaN/Infinity, wrong types and unknown keys are not.
    """
    try:
        raw = json.loads(
            data.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ValueError as exc:  # includes UnicodeDecodeError and JSONDecodeError
        raise ProjectFormatError(f"project.json is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProjectFormatError("project.json must hold a JSON object")

    version = raw.get("schema_version")
    if type(version) is not int or version < 1:  # a bool is not a version
        raise SchemaVersionError(
            f"project.json has no valid schema_version (found {version!r})",
            found=version,
            supported=SCHEMA_VERSION,
        )
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"project.json was saved by a newer Proteia (schema {version});"
            f" this version reads schema <= {SCHEMA_VERSION}",
            found=version,
            supported=SCHEMA_VERSION,
        )
    if version < SCHEMA_VERSION:
        raw = migrate(raw)

    # The model knows only the latest schema; strict JSON mode keeps every type
    # exactly (no "340" for an int, no 0 for a bool), so load then save is a fixed point.
    try:
        return Project.model_validate_json(json.dumps(raw, ensure_ascii=False), strict=True)
    except ValidationError as exc:
        raise ProjectFormatError(f"project.json is not a valid project: {exc}") from exc


# --- Schema versions and migrations ---

Migration = Callable[[dict[str, Any]], dict[str, Any]]
# {n: a step from schema n to n + 1}. Empty while SCHEMA_VERSION is 1. From the v0.1
# tag on, every change to the saved form bumps SCHEMA_VERSION and registers a step
# here (an additive change registers a step that only bumps the number).
MIGRATIONS: Mapping[int, Migration] = {}


def migrate(
    doc: dict[str, Any],
    *,
    target: int = SCHEMA_VERSION,
    migrations: Mapping[int, Migration] = MIGRATIONS,
) -> dict[str, Any]:
    """Run the migration steps from ``doc``'s schema up to ``target`` on a copy."""
    out, version = copy.deepcopy(doc), doc["schema_version"]
    while version < target:
        step = migrations.get(version)
        if step is None:
            raise SchemaVersionError(
                f"no migration from schema {version}", found=version, supported=target
            )
        out = step(out)
        if out.get("schema_version") != version + 1:
            raise RuntimeError(f"migration from schema {version} did not produce {version + 1}")
        version += 1
    return out


# --- The project folder ---


def image_path(folder: str | os.PathLike[str], image: ImageRef) -> Path:
    """Where an image's pixels are stored."""
    return Path(folder) / IMAGES_DIR / image.file


def _missing_images(project: Project, folder: str | os.PathLike[str]) -> list[str]:
    return [
        image.id for image in project.batch.iter_images() if not image_path(folder, image).is_file()
    ]


def _replace(src: str, dst: Path) -> None:
    """``os.replace``, retried while Windows reports a sharing violation."""
    delay = REPLACE_DELAY
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _fsync_dir(directory: Path) -> None:
    if os.name == "nt":  # Windows cannot open a directory to fsync it
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` so a reader sees either the old or the new bytes."""
    # The same directory means the same volume, so the replace is atomic.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        # Closed before the replace: Windows cannot replace with an open file.
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
        # Best effort: the replace already succeeded, and some filesystems (network
        # or FUSE mounts) refuse to fsync a directory.
        with contextlib.suppress(OSError):
            _fsync_dir(path.parent)
    finally:
        with contextlib.suppress(OSError):  # already gone after a successful replace
            os.unlink(tmp)


def save_project(project: Project, folder: str | os.PathLike[str]) -> Path:
    """Write ``project.json`` into ``folder`` atomically and return its path.

    Validates first (a ``ValidationError`` means an in-process bug, and nothing is
    written) and refuses, before creating anything, when an image file is missing.
    Creates the folder, ``images/`` and ``exports/``; never copies or deletes images.
    A ``PermissionError`` that outlasts the retries propagates and leaves the
    previous ``project.json`` intact.
    """
    data = project_to_json(project)
    missing = _missing_images(project, folder)
    if missing:
        raise MissingImageError(missing)
    folder = Path(folder)
    for directory in (folder, folder / IMAGES_DIR, folder / EXPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    path = folder / PROJECT_FILE
    _atomic_write(path, data)
    return path


def load_project(folder: str | os.PathLike[str], *, require_images: bool = True) -> Project:
    """Read and validate ``folder/project.json``.

    ``FileNotFoundError`` propagates. With ``require_images``, a missing image file
    raises :class:`MissingImageError`. Image hashes are not re-checked here; see
    :func:`verify_images`.
    """
    project = project_from_json((Path(folder) / PROJECT_FILE).read_bytes())
    if require_images:
        missing = _missing_images(project, folder)
        if missing:
            raise MissingImageError(missing)
    return project


def store_image(
    folder: str | os.PathLike[str],
    image_id: str,
    original_name: str,
    source: BinaryIO,
    *,
    max_bytes: int | None = None,
) -> StoredImage:
    """Copy an image stream into ``images/<image_id><suffix>`` without decoding it.

    The suffix is the lowercased suffix of ``original_name`` and must be in
    ``IMAGE_SUFFIXES``. Refuses to overwrite (``FileExistsError``), an empty stream
    (``ValueError``) and a stream longer than ``max_bytes`` (:class:`ImageTooLargeError`),
    leaving no file behind. Width, height, bit depth, background and warnings are
    the caller's (the image loader's) to fill in.
    """
    try:
        _IMAGE_ID.validate_python(image_id, strict=True)
        _ORIGINAL_NAME.validate_python(original_name, strict=True)
    except ValidationError as exc:
        raise ValueError(f"cannot store image {image_id!r} from {original_name!r}: {exc}") from exc
    suffix = PurePosixPath(original_name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image type {suffix!r} of {original_name!r}")

    images = Path(folder) / IMAGES_DIR
    file = image_id + suffix
    target = images / file
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    images.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=images, prefix=f".{file}.", suffix=".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := source.read(_CHUNK):
                size += len(chunk)
                if max_bytes is not None and size > max_bytes:
                    raise ImageTooLargeError(f"image {original_name!r} exceeds {max_bytes} bytes")
                digest.update(chunk)
                out.write(chunk)
            if size == 0:
                raise ValueError(f"image {original_name!r} is empty")
            out.flush()
            os.fsync(out.fileno())
        _replace(tmp, target)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
    return StoredImage(file=file, sha256=digest.hexdigest(), size=size)


def verify_images(project: Project, folder: str | os.PathLike[str]) -> list[str]:
    """Ids of images whose file is missing or whose bytes no longer match ``sha256``."""
    bad: list[str] = []
    for image in project.batch.iter_images():
        path = image_path(folder, image)
        if not path.is_file():
            bad.append(image.id)
            continue
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != image.sha256:
                bad.append(image.id)
    return bad
