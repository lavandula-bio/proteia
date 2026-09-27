# SPDX-License-Identifier: Apache-2.0
"""The project folder: save, load, the content hash and the image store.

GUI-independent: the standard library plus the model (and pydantic, through its
types). A project folder, which may have any name, holds::

    project.json                    the model (see "Canonical form" below)
    project.schema1.json            project.json as it was before a migration from schema 1
                                    rewrote it (keep_backup; numbered if taken:
                                    project.schema1 (2).json)
    images/img-2.tif                byte-identical copies of the imported files, named by id
    exports/                        created by save_project, for exported results:
    exports/lane-table.csv          the per-lane table
    exports/lane-table.record.json  its reproducibility record (proteia.core.record)
    exports/2026-09-27 1532/        an export folder, one per export: lane tables,
                                    charts, README.txt and export.record.json
                                    (proteia.core.operations.export_bundle)

Stored names come only from validated ids plus a whitelisted suffix; an image's
``original_name`` never becomes a path. A crash can leave orphan files (a temp
file, or an image no saved project references) but never a dangling reference:
an image is stored before the model references it, and a file is deleted only
after a saved ``project.json`` no longer references it. An orphan image can hold
an id that a failed or unsaved import gave back, so importing removes
:func:`orphan_files` before storing a new image: all but those the session
keeps (:class:`~proteia.core.session.ProjectSession`), none of which holds the
id the import takes.

Canonical form. ``project.json`` is ``json.dumps`` of the re-validated model's
JSON-mode dump with sorted keys, ``indent=1``, ``ensure_ascii=False`` and
``allow_nan=False``, plus a trailing newline, encoded as UTF-8 with no BOM and
written in binary, so every platform gets LF and raw UTF-8 for µ, α and β
(:func:`document_bytes`, which the export records share). The content hash is
the lowercase hex SHA-256 of :func:`canonical_json` (the same dump written
compactly) without the keys in :data:`HASH_EXCLUDE`. In detail:

* Floats are written by the standard library's ``float.__repr__``, the shortest
  form that round-trips exactly. pydantic-core's serializer writes some floats
  differently (``1e-7`` where Python writes ``1e-07``), so ``model_dump_json`` is
  never used: a library upgrade must not move the hash.
* Sorted keys make the bytes independent of field order and of the insertion
  order of ``Lane.metadata``. The model sorts bands, not-detected records and
  calibration points; the order of membranes, images and proteins is content
  and changes the hash.
* An optional field added before v0.1 is left out of every dump (``project.json``,
  :func:`content_document` and so the export record's ``content``) while its value
  is empty or its default, so adding it changes no project's bytes or hash. The
  first case is ``Protein.undetected``: Proteia never writes ``"undetected": []``,
  and a hand-edited file that holds it loads, hashes as if the key were absent and
  is saved without it. The second is ``Protein.box_padding``, left out while it is
  ``{"across": 0, "along": 0}``, likewise. The third is ``Band.possibly_clipped``
  (#112), left out while it is ``null``: only bands on a lossy, colour or CMYK
  image of known bit depth have it. Writing the empty value instead would
  move every existing project's hash, so every export would report
  ``content_changed_outside_log``.
* The hash covers ``schema_version``, the background method, every id, the lane
  table, the image records (and so the pixels, through each ``sha256``), the
  calibrations, and the proteins with their bands (stored nets, backgrounds and
  flags) and their not-detected records. It excludes ``next_id`` and the log
  (the history: equal content made at other times must hash equal), and the
  file's formatting. The project never stores its own hash; each log entry
  stores the hash of the content it left.
* Loading a file Proteia wrote and saving it again gives the same bytes, because
  the strict load keeps every type exactly. A file of an older schema is
  migrated as it loads (:func:`migrate`, :data:`MIGRATIONS`), in memory, and
  gets one ``migrate`` log entry; it is written in the new schema at its next
  save (which :func:`~proteia.core.session.open_project` runs at once, through
  the session's autosave hook). Before that, opening it keeps the file as it
  was read in a backup beside it (:func:`keep_backup`, #140), which the entry
  names: renamed back to ``project.json``, it is the project as it was. The
  files of a backup's images are never deleted as orphans
  (:func:`backup_references`), so a migration that lost an image reference
  cannot lose the image.

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
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Final, NoReturn

from pydantic import TypeAdapter, ValidationError

import proteia
from proteia.core.model import (
    IMAGE_SUFFIXES,
    LEGACY_BACKGROUND_METHOD,
    SCHEMA_VERSION,
    ImageId,
    ImageRef,
    LogEntry,
    OriginalName,
    Project,
    format_timestamp,
    revalidate,
)

PROJECT_FILE: Final = "project.json"
# project.schema<N>.json: a project.json of schema N kept before its migration.
BACKUP_PREFIX: Final = "project.schema"
IMAGES_DIR: Final = "images"
EXPORTS_DIR: Final = "exports"
# Top-level keys left out of the content hash: bookkeeping and history, not content.
HASH_EXCLUDE: Final = frozenset({"next_id", "log"})
# A reader holding project.json (antivirus, indexer, OneDrive) makes the replace
# fail on Windows: retry with a doubling delay (about 0.75 s in all).
REPLACE_ATTEMPTS = 5
REPLACE_DELAY = 0.05  # seconds; tests set it to 0
# The longest file name most file systems take, in UTF-8 bytes (ext4, APFS;
# NTFS counts 255 UTF-16 code units, never more than a name's UTF-8 bytes).
NAME_LIMIT_BYTES = 255  # tests lower it
# The longest path Windows takes while long paths are off, its default:
# MAX_PATH (260) less its terminating NUL, in UTF-16 code units. None where no
# such limit applies.
PATH_LIMIT: int | None = 259 if os.name == "nt" else None  # tests set it
_CHUNK: Final = 1 << 20  # store_image copies 1 MiB at a time

_IMAGE_ID = TypeAdapter(ImageId)
_ORIGINAL_NAME = TypeAdapter(OriginalName)
# A stored image's name (img-N and an image suffix) anywhere in a file's bytes,
# in any case, ending where the suffix does (img-3.tiff names img-3.tiff, not img-3.tif).
_IMAGE_NAME = re.compile(
    rb"img-[0-9]+\.(?:"
    + b"|".join(re.escape(suffix.removeprefix(".").encode()) for suffix in IMAGE_SUFFIXES)
    + rb")(?![a-z0-9])",
    re.IGNORECASE,
)


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


class BackupError(OSError):
    """An older-schema ``project.json`` could not be kept in a backup before its
    migration (:func:`keep_backup`; the ``OSError`` is ``__cause__``), so it was
    not migrated and nothing was written. The message names the project by its
    folder's name, never its path, and says why."""


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


def document_bytes(doc: object) -> bytes:
    """A JSON document in the file form (see "Canonical form"): ``project.json``
    and the export records share it."""
    text = json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False)
    return (text + "\n").encode("utf-8")


def project_to_json(project: Project) -> bytes:
    """The exact ``project.json`` bytes. Raises ``ValidationError`` if an in-place
    edit made the project invalid; the log is validated too, so a file that
    cannot load is never written."""
    return document_bytes(revalidate(project).model_dump(mode="json"))


def content_document(project: Project) -> dict[str, Any]:
    """The re-validated content that :func:`content_hash` hashes: the JSON-mode
    dump without the keys in :data:`HASH_EXCLUDE`."""
    return revalidate(project, log=False).model_dump(mode="json", exclude=set(HASH_EXCLUDE))


def content_bytes(project: Project) -> bytes:
    """``canonical_json(content_document(project))``: the bytes :func:`content_hash`
    hashes, and the form in which a session's undo history keeps each state
    (:func:`project_from_content` reads it back)."""
    return canonical_json(content_document(project))


def document_hash(doc: Any) -> str:
    """Lowercase hex SHA-256 of ``canonical_json(doc)``: the one hash recipe."""
    return hashlib.sha256(canonical_json(doc)).hexdigest()


def content_hash(project: Project) -> str:
    """Lowercase hex SHA-256 of the project's content (see the module docstring):
    of :func:`content_bytes`. Its cost does not grow with the log."""
    return hashlib.sha256(content_bytes(project)).hexdigest()


def project_from_content(data: bytes, *, next_id: int, log: tuple[LogEntry, ...]) -> Project:
    """The project whose content is ``data`` (:func:`content_bytes`), with
    ``next_id`` and ``log``: what an undo restores.

    The content is validated strictly, as :func:`project_from_json` validates a
    file (``ValueError`` for bytes that are not JSON or repeat a key, a pydantic
    ``ValidationError`` for content that is not a valid project, an id at or
    above ``next_id`` included). ``log`` is attached as it is, neither copied nor
    validated again, so a session sees the project as prepared from its committed
    one. No migration runs: the history lives only as long as its session.
    """
    doc = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if not isinstance(doc, dict):
        raise ValueError("the content must be a JSON object")
    project = Project.model_validate_json(
        json.dumps({**doc, "next_id": next_id}, ensure_ascii=False), strict=True
    )
    return project.model_copy(update={"log": log})


def _utc_now() -> datetime:
    """The time of a ``migrate`` entry when no session clock is given."""
    return datetime.now(UTC)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not allowed")


def project_from_json(data: bytes, *, clock: Callable[[], datetime] = _utc_now) -> Project:
    """Parse and validate ``project.json`` bytes, migrating an older schema.

    A BOM (from a Windows editor) and CRLF are tolerated; duplicate keys,
    NaN/Infinity, wrong types and unknown keys are not. A migrated project gets
    one ``migrate`` log entry (:func:`_migrated`), timed by ``clock``.
    """
    return _read_json(data, clock)[0]


def _read_json(data: bytes, clock: Callable[[], datetime]) -> tuple[Project, bool]:
    """:func:`project_from_json`, and whether it migrated the project."""
    project, found = _parse(data)
    if found == SCHEMA_VERSION:
        return project, False
    return _migrated(project, found, clock), True


def _parse(data: bytes) -> tuple[Project, int]:
    """The project ``project.json``'s bytes ``data`` hold, its content migrated
    to the latest schema (without the ``migrate`` entry :func:`_migrated` adds),
    and the schema the file had."""
    try:
        raw = json.loads(
            data.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    # ValueError includes UnicodeDecodeError and JSONDecodeError; RecursionError
    # comes from absurdly deep nesting in a corrupt file.
    except (ValueError, RecursionError) as exc:
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
        project = Project.model_validate_json(json.dumps(raw, ensure_ascii=False), strict=True)
    except ValidationError as exc:
        raise ProjectFormatError(f"project.json is not a valid project: {exc}") from exc
    return project, version


# --- Schema versions and migrations ---

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def _dicts(value: object) -> list[dict[str, Any]]:
    """The objects in a JSON list, for a migration step: a part of an unvalidated
    file that is not the shape it should be is skipped, for validation to refuse."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _v1_to_v2(doc: dict[str, Any]) -> dict[str, Any]:
    """Schema 1 to 2 (#83): the project's background method and each band's
    background fields.

    A schema-1 net was measured above its image's median, so the project keeps
    that method (``global_median``) and each band records its image's
    ``background`` as its level, with mode ``global_median`` and no spread:
    every net stays exact, and stays what the pixels give under that method.
    Pure JSON, on the copy :func:`migrate` passes.
    """
    doc["schema_version"] = 2
    doc["background_method"] = LEGACY_BACKGROUND_METHOD
    batch = doc.get("batch")
    if not isinstance(batch, dict):
        return doc
    medians = {
        image["id"]: image.get("background")
        for membrane in _dicts(batch.get("membranes"))
        for image in _dicts(membrane.get("images"))
        if isinstance(image.get("id"), str)
    }
    for protein in _dicts(batch.get("proteins")):
        image_id = protein.get("image_id")
        if not isinstance(image_id, str) or image_id not in medians:
            continue  # an unknown image: validation refuses the protein
        for band in _dicts(protein.get("bands")):
            band["background_level"] = medians[image_id]
            band["background_mode"] = LEGACY_BACKGROUND_METHOD
            band["background_spread"] = 0.0
    return doc


def _v1_content(content: dict[str, Any]) -> dict[str, Any]:
    """A schema-2 content document (:func:`content_document`) as schema 1 held
    the same content: without the background method and the band background
    fields. Exact for a migrated project, since the step only adds them."""
    old = {key: value for key, value in content.items() if key != "background_method"}
    old["schema_version"] = 1
    old["batch"] = batch = dict(content["batch"])
    added = {"background_level", "background_mode", "background_spread"}
    batch["proteins"] = [
        {
            **protein,
            "bands": [
                {key: value for key, value in band.items() if key not in added}
                for band in protein["bands"]
            ],
        }
        for protein in batch["proteins"]
    ]
    return old


# {n: a step from schema n to n + 1}. Before v0.1, an additive change (a new
# optional field left out while empty, or a new enum value) does not bump; from
# v0.1 on, every saved-form change bumps and registers a migration: it bumps
# SCHEMA_VERSION and registers a step here (an additive change registers a step
# that only bumps the number), and registers in _EARLIER_CONTENT how the content
# of a project migrated from each older schema read in that schema.
MIGRATIONS: Mapping[int, Migration] = {1: _v1_to_v2}
# {n: a latest-schema content document as schema n held it}: the migrate entry's
# from_content_hash.
_EARLIER_CONTENT: Mapping[int, Callable[[dict[str, Any]], dict[str, Any]]] = {1: _v1_content}


def _migrated(
    project: Project, found: int, clock: Callable[[], datetime], *, backup: str | None = None
) -> Project:
    """``project``, migrated from schema ``found``, with a ``migrate`` log entry
    appended. Its params name the schemas it went from and to and the hash the
    content it started from had under schema ``found`` (``from_content_hash``:
    the old log's last hash, unless the file was changed outside it, which
    :func:`proteia.core.record.history_issues` reports), and, when given, the
    ``backup`` that holds the file it was read from (:func:`keep_backup`); its
    ``content_hash`` is, as for every entry, that of the content it left. The
    log is not content, so appending the entry does not change that hash.
    """
    content = content_document(project)
    params: dict[str, Any] = {
        "from_schema": found,
        "to_schema": SCHEMA_VERSION,
        "from_content_hash": document_hash(_EARLIER_CONTENT[found](content)),
    }
    if backup is not None:
        params["backup"] = backup
    entry = LogEntry(
        seq=len(project.log) + 1,
        time=format_timestamp(clock()),
        action="migrate",
        version=proteia.__version__,
        params=params,
        content_hash=document_hash(content),
    )
    return project.model_copy(update={"log": (*project.log, entry)})


def migrate(
    doc: dict[str, Any],
    *,
    target: int = SCHEMA_VERSION,
    migrations: Mapping[int, Migration] = MIGRATIONS,
) -> dict[str, Any]:
    """Run the migration steps from ``doc``'s schema up to ``target`` on a copy.

    The steps change content only. :func:`project_from_json` then appends one
    ``migrate`` log entry for the whole run, with the hash of the validated
    content it left; without it, every export record of a migrated project
    would report ``content_changed_outside_log``.
    """
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


def _replace(src: Path, dst: Path) -> None:
    """``Path.replace``, retried while Windows reports a sharing violation."""
    delay = REPLACE_DELAY
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _fsync_dir(directory: Path) -> None:
    """Best effort: make a rename in ``directory`` durable (POSIX only)."""
    if os.name == "nt":  # Windows cannot open a directory to fsync it
        return
    # The rename already succeeded, and some filesystems (network or FUSE mounts)
    # refuse to fsync a directory.
    with contextlib.suppress(OSError):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _file_mode(target: Path) -> int | None:
    """The permission bits a new file should get on POSIX (``None`` on Windows).

    ``mkstemp`` creates owner-only (0600) files, which would lock other users out
    of a project folder on a shared volume: keep the replaced file's mode, or use
    the default mode for new files under the current umask.
    """
    if os.name == "nt":
        return None
    with contextlib.suppress(OSError):
        return target.stat().st_mode & 0o777
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _temp_file(directory: Path, name: str, suffix: str) -> tuple[int, Path]:
    # The same directory means the same volume, so the later replace is atomic.
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{name}.", suffix=suffix)
    return fd, Path(tmp)


def name_fits(folder: Path, name: str) -> bool:
    """Whether :func:`write_atomic` can write a file named ``name`` into
    ``folder`` (which need not exist yet) within the file system's length limits.

    It writes the file through a temp file with a longer name,
    ``.<name>.<8 random characters>.tmp`` (:func:`tempfile.mkstemp`'s names),
    whose name must hold at most :data:`NAME_LIMIT_BYTES` UTF-8 bytes and, on
    Windows, whose absolute path must hold at most :data:`PATH_LIMIT` UTF-16
    code units (a character outside the Basic Multilingual Plane counts two).
    """
    temp = f".{name}.{'x' * 8}.tmp"
    if len(temp.encode("utf-8", "surrogatepass")) > NAME_LIMIT_BYTES:
        return False
    if PATH_LIMIT is None:
        return True
    path = os.path.join(os.path.abspath(folder), temp)
    return len(path.encode("utf-16-le", "surrogatepass")) // 2 <= PATH_LIMIT


def write_atomic(path: Path, data: bytes, *, private: bool = False) -> None:
    """Replace ``path`` with ``data`` so a reader sees either the old or the new bytes.

    The folder must exist. A ``PermissionError`` that outlasts the retries (a
    reader holding the file on Windows) propagates and leaves the old file intact.
    ``private`` makes the file owner-only (0600 on POSIX) whatever the old one's
    mode; on Windows the folder's permissions apply either way.
    """
    mode = None if private else _file_mode(path)
    fd, tmp = _temp_file(path.parent, path.name, ".tmp")
    try:
        # Closed before the replace: Windows cannot replace with an open file.
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            tmp.chmod(mode)
        _replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        with contextlib.suppress(OSError):  # already gone after a successful replace
            tmp.unlink(missing_ok=True)


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
    write_atomic(path, data)
    return path


def load_project(
    folder: str | os.PathLike[str],
    *,
    require_images: bool = True,
    clock: Callable[[], datetime] = _utc_now,
) -> Project:
    """Read and validate ``folder/project.json``, migrating an older schema in
    memory (:func:`project_from_json`, whose ``migrate`` entry ``clock`` times);
    the file is not rewritten.

    ``FileNotFoundError`` propagates. With ``require_images``, a missing image file
    raises :class:`MissingImageError`. Image hashes are not re-checked here; see
    :func:`verify_images`.
    """
    return read_project(folder, require_images=require_images, clock=clock)[0]


def read_project(
    folder: str | os.PathLike[str],
    *,
    require_images: bool = True,
    clock: Callable[[], datetime] = _utc_now,
    backup: bool = False,
) -> tuple[Project, bool]:
    """:func:`load_project`, and whether the project was migrated: then
    ``project.json``, of an older schema, does not hold it (nor its ``migrate``
    entry) until it is saved.

    With ``backup``, a project is migrated only once the bytes of the
    ``project.json`` it was read from are kept in a backup beside it
    (:func:`keep_backup`), which its ``migrate`` entry names (its ``backup``
    param): what opening a project does before it saves the migration. A
    backup that cannot be written raises :class:`BackupError`; a missing image
    file raises before it is written. Nothing else is written."""
    folder = Path(folder)
    data = (folder / PROJECT_FILE).read_bytes()
    project, found = _parse(data)
    if require_images:
        missing = _missing_images(project, folder)
        if missing:
            raise MissingImageError(missing)
    if found == SCHEMA_VERSION:
        return project, False
    kept = keep_backup(folder, data, found) if backup else None
    return _migrated(project, found, clock, backup=kept), True


def backup_name(schema: int, number: int = 1) -> str:
    """The name of the ``number``-th backup of a ``project.json`` of schema
    ``schema``: ``project.schema1.json``, then ``project.schema1 (2).json`` and
    so on, numbered as export folders are."""
    stem = f"{BACKUP_PREFIX}{schema}"
    return f"{stem}.json" if number == 1 else f"{stem} ({number}).json"


def is_backup(name: str) -> bool:
    """Whether a file named ``name`` in a project folder is taken for a backup
    of an older ``project.json`` (``project.schema*.json``, in any case, as
    Windows compares names): more than :func:`backup_name` gives, so that no
    backup is missed."""
    lower = name.lower()
    return lower.startswith(BACKUP_PREFIX) and lower.endswith(".json")


def keep_backup(folder: str | os.PathLike[str], data: bytes, schema: int) -> str:
    """Keep ``data``, the bytes of ``folder``'s ``project.json`` of schema
    ``schema``, in a backup beside it before a migration rewrites the file;
    return the backup's name.

    The backup is the first :func:`backup_name` whose file holds ``data``
    already, or that nothing has, where ``data`` is then written: a file is
    never replaced, so a backup kept at an earlier open stays as it was. The
    bytes are written as they were read (a BOM and CRLF included, which loading
    tolerates), and made durable before this returns, so a migrated
    ``project.json`` never replaces the only copy of the file. An ``OSError``
    raises :class:`BackupError` and leaves no file behind (one it began to
    write is removed).
    """
    folder = Path(folder)
    number = 1
    try:
        while True:
            name = backup_name(schema, number)
            path = folder / name
            if _write_new(path, data) or _holds(path, data):
                return name
            number += 1
    except OSError as exc:
        why = exc.strerror or type(exc).__name__
        raise BackupError(
            f"{folder.name!r} was saved by an older Proteia (schema {schema}), and a copy"
            f" of its {PROJECT_FILE} could not be kept before updating it ({why});"
            " nothing was changed"
        ) from exc


def _write_new(path: Path, data: bytes) -> bool:
    """Write ``data`` into a new file ``path``, durably; False, writing nothing,
    if something of that name exists. An ``OSError`` removes what was written."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, 0o666)  # the default mode for new files, under the umask
    except FileExistsError:
        return False
    except PermissionError:
        if path.exists():  # Windows refuses so where a folder has the name
            return False
        raise
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    _fsync_dir(path.parent)
    return True


def _holds(path: Path, data: bytes) -> bool:
    """Whether ``path`` is a file that holds ``data`` (False if it cannot be read)."""
    try:
        return path.is_file() and path.stat().st_size == len(data) and path.read_bytes() == data
    except OSError:
        return False


def backup_references(folder: str | os.PathLike[str]) -> frozenset[str] | None:
    """The names of the stored image files (``img-N`` and an image suffix, in
    the case found) that the backups in ``folder`` (:func:`is_backup`) name
    anywhere in their bytes, whatever their schema, even one that no longer
    loads: the files a restored backup would need. None when the folder cannot
    be listed or a backup cannot be read, as what they name is then unknown.

    The orphan cleanup deletes none of them whose id is below the project's
    ``next_id``, and while this is None it keeps every such file
    (:class:`~proteia.core.session.ProjectSession`). A backup's images hold
    only such ids, so a migration that lost an image reference cannot lose the
    image while the backup is kept; and a name found elsewhere in a backup (an
    image's original name, say) never keeps the file of the id an import takes
    next.
    """
    names: set[str] = set()
    try:
        for path in Path(folder).iterdir():
            if is_backup(path.name) and path.is_file():
                data = path.read_bytes()
                names.update(match.decode("ascii") for match in _IMAGE_NAME.findall(data))
    except OSError:
        return None
    return frozenset(names)


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
        raise FileExistsError(f"{target} already exists (remove orphan_files first)")
    images.mkdir(parents=True, exist_ok=True)
    mode = _file_mode(target)
    fd, tmp = _temp_file(images, file, ".part")
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
        if mode is not None:
            tmp.chmod(mode)
        _replace(tmp, target)
        # The image must be durable before a saved project.json can reference it.
        _fsync_dir(images)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return StoredImage(file=file, sha256=digest.hexdigest(), size=size)


def orphan_files(project: Project, folder: str | os.PathLike[str]) -> list[Path]:
    """Files in ``images/`` that ``project`` does not reference, sorted by name.

    Some are left by a crash or by an import that was never saved: stale
    ``.part``/``.tmp`` files, and images whose id the project may hand out again.
    Others are the files of images removed or undone in the open session, which
    it keeps while its undo history can bring them back, files the saved
    ``project.json`` still references, or files with an id below ``next_id``
    that a backup of an older ``project.json`` names
    (:func:`backup_references`). The session deletes
    only the rest, before an import and after a save
    (:class:`~proteia.core.session.ProjectSession`).
    """
    images = Path(folder) / IMAGES_DIR
    if not images.is_dir():
        return []
    referenced = {image.file for image in project.batch.iter_images()}
    return sorted(
        path for path in images.iterdir() if path.is_file() and path.name not in referenced
    )


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
