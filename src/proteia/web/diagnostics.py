# SPDX-License-Identifier: Apache-2.0
"""The diagnostic file (#138): one zip file a tester attaches to a bug report.

The page first shows what the file will hold (:func:`plan`, ``GET
/api/diagnostics``), then writes it (:func:`write`, ``POST /api/diagnostics``).
It holds:

* ``environment.json``: the versions of Proteia and Python, the operating
  system (:func:`platform.platform`, and its system, release, version and
  machine), whether Proteia runs as the installed build, and the libraries'
  versions as the analysis record reads them
  (:func:`~proteia.core.record.software_versions`); no computer name, user name
  or path;
* ``logs/``: the newest files of the session log (:mod:`proteia.web.logs`),
  which already hides tokens and folder paths but names the projects,
  proteins, conditions and image files worked on, newest first and at most
  :data:`LOG_BYTES` in all: the current file always, cut to its last
  :data:`LOG_BYTES` from the start of a line when it is longer (a rotation
  another program kept blocking lets it grow), then each older one while the
  total stays within;
* ``project/``, when a project is open: its ``project.json`` as last saved
  (read when listed, so the file holds the one whose images were listed), the
  reproducibility records of its exports (``exports/*.record.json`` and
  ``exports/<folder>/*.record.json``, the :data:`RECORDS` newest), and, only
  when asked for, the image files its images are stored in (``images/``):
  those of the project as open, and those ``project.json`` names, which differ
  while its latest changes are not saved (an image removed keeps its file
  until a save). The folder is then a copy of the project Proteia can open, as
  last saved (the images are not capped: the page shows their size before the
  file is written, and PNG and JPEG files are stored as they are, the rest
  compressed). ``project.json`` and the records hold the names of the
  project, its proteins, conditions, samples and image files;
* ``README.txt``: what is inside, and that Proteia sends it nowhere;
* ``manifest.json``: every other file, with its size and SHA-256, and each file
  left out, with why.

Only the files named above are taken, so the file never holds the access
token or a file that holds it (``instance.json``, ``open-proteia.html``), the
lock file, or images handed to Proteia but not imported (``incoming/``).
Symbolic links and junctions among the export folders are not followed. A
file that cannot be read (removed meanwhile) is left out, and the manifest
says why; the README then says whether ``project/`` still opens as the project.

What the page listed is what is written: the list carries a digest of the
project's files it names (:meth:`Plan.digest`), and a write whose list, made
again, names others (an export made or an image imported meanwhile, in another
tab) is refused, so the page lists again.

The file is written in the ``diagnostics`` folder of the per-user state folder
(:func:`proteia.web.launch.state_dir`), whether a project is open or not: the
session log it holds is about every project, so it does not belong in one
project's folder, whose files go wherever the project is copied or shared; and
the state folder stays on this computer, where the launcher keeps the log. It is named
``Proteia diagnostics <local date and time>.zip``, numbered ``(2)`` and on when
the name is taken, and written under a temporary name first, so a failure
leaves no partial file. One is written at a time, and the :data:`KEPT` newest
are kept: each one written deletes older ones (images can make them large, and
the folder is out of sight).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import textwrap
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import proteia
from proteia.core import storage
from proteia.core.export import bundle_folder_name
from proteia.core.model import format_timestamp
from proteia.core.record import software_versions
from proteia.core.session import ProjectSession
from proteia.web import logs

DIAGNOSTICS_DIR: Final = "diagnostics"  # in the per-user state folder
FILE_STEM: Final = "Proteia diagnostics"
LOG_BYTES: Final = 6 * 1024 * 1024  # of the session log, the newest files first
RECORDS: Final = 10  # export records, the newest first
KEPT: Final = 5  # diagnostic files kept in the diagnostics folder
MAX_NUMBERED: Final = 1000  # names tried: the plain one, then numbered
README_FILE: Final = "README.txt"
MANIFEST_FILE: Final = "manifest.json"
ENVIRONMENT_FILE: Final = "environment.json"
MANIFEST_FORMAT: Final = 1
_PART: Final = ".part"  # the suffix of a file being written
_STORED: Final = frozenset({".png", ".jpg", ".jpeg"})  # compressed already: stored as they are
_CHUNK: Final = 1024 * 1024
_WIDTH: Final = 78  # of the README's lines
_LOG_NAME: Final = re.compile(re.escape(logs.LOG_FILE) + r"(?:\.([0-9]+))?")
_RECORD_SUFFIX: Final = ".record.json"
_PROJECT: Final = "project/"  # the folder of the project's files in the file

_writing = threading.Lock()  # one diagnostic file at a time


@dataclass(frozen=True)
class Item:
    """A file of the diagnostic file: its ``name`` there and its ``size`` when
    listed (the current log grows until it is written). Read from ``source``
    when written, or made here (``data``); with ``tail``, only its last
    :data:`LOG_BYTES` are taken."""

    name: str
    size: int
    source: Path | None = None
    data: bytes | None = None
    tail: bool = False


@dataclass(frozen=True)
class LeftOut:
    """A file not in the diagnostic file: its name there, its size (None if not
    known) and why."""

    name: str
    size: int | None
    reason: str


@dataclass(frozen=True)
class Plan:
    """What a diagnostic file holds (:func:`plan`): ``project`` (the open one's
    name, or None) and whether its changes are saved (``saved``, None without
    one); the files taken (``items``), the project's image files, taken only
    when asked for (``images``), and the files left out. ``needs``: the image
    files (named as in the file) that the ``project.json`` taken names, which
    ``project/`` needs to open as the project; None without a ``project.json``
    taken that reads as one."""

    project: str | None
    saved: bool | None
    items: tuple[Item, ...]
    images: tuple[Item, ...]
    left_out: tuple[LeftOut, ...]
    needs: tuple[str, ...] | None

    def digest(self) -> str:
        """A SHA-256 of the names of the project's files listed: those taken
        and its image files, taken or not. A write checks that its list, made
        again, has the same, so it holds no project file the page did not show.
        The log's files are not in it: its newest part is taken as it is when
        written, and it grows meanwhile."""
        names = {item.name for item in self.items if item.name.startswith(_PROJECT)}
        names.update(item.name for item in self.images)
        return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()

    def files(self, images: bool) -> tuple[Item, ...]:
        return self.items + (self.images if images else ())

    def omitted(self, images: bool) -> tuple[LeftOut, ...]:
        if images:
            return self.left_out
        asked = "images are included only when asked for"
        return self.left_out + tuple(LeftOut(item.name, item.size, asked) for item in self.images)


@dataclass(frozen=True)
class Written:
    """A diagnostic file written: where, how many files it holds (the manifest
    included), its size in bytes, and how many files were left out."""

    path: Path
    files: int
    size: int
    left_out: int


def environment() -> dict[str, Any]:
    """What ``environment.json`` holds (see the module docstring)."""
    return {
        "proteia": proteia.__version__,
        "build": "installed" if getattr(sys, "frozen", False) else "source",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "software": software_versions(),
    }


def _json(document: object) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _size_text(size: int) -> str:
    return f"{size / (1024 * 1024):g} MiB" if size >= 1024 * 1024 else f"{size} bytes"


def _regular(path: Path) -> int | None:
    """The size of ``path`` if it is a regular file, not a link to one; None otherwise."""
    try:
        if path.is_symlink():
            return None
        info = path.stat()
    except OSError:
        return None
    return info.st_size if path.is_file() else None


def _log_files(folder: Path) -> tuple[list[Item], list[LeftOut]]:
    """The session log's files in ``folder`` taken (see the module docstring),
    and those left out."""
    found: list[tuple[int, Path, int]] = []
    with contextlib.suppress(OSError):
        for path in folder.iterdir():
            match = _LOG_NAME.fullmatch(path.name)
            size = None if match is None else _regular(path)
            if match is not None and size is not None:
                found.append((int(match[1] or 0), path, size))
    found.sort()
    items: list[Item] = []
    left: list[LeftOut] = []
    total = 0
    for _, path, size in found:
        name = f"logs/{path.name}"
        if not items:
            tail = size > LOG_BYTES
            items.append(Item(name, min(size, LOG_BYTES), source=path, tail=tail))
            total = items[0].size
        elif not left and total + size <= LOG_BYTES:
            items.append(Item(name, size, source=path))
            total += size
        else:
            left.append(
                LeftOut(name, size, f"older than the newest {_size_text(LOG_BYTES)} of the log")
            )
    return items, left


def _records(exports: Path) -> tuple[list[Item], list[LeftOut]]:
    """The export records in ``exports`` taken, the newest first, and those left
    out; links and junctions are not followed."""
    found: list[tuple[float, str, Path, int]] = []

    def take(path: Path, name: str) -> None:
        size = _regular(path) if path.name.endswith(_RECORD_SUFFIX) else None
        if size is not None:
            with contextlib.suppress(OSError):
                found.append((path.stat().st_mtime, name, path, size))

    with contextlib.suppress(OSError):
        for child in exports.iterdir():
            name = f"{_PROJECT}{storage.EXPORTS_DIR}/{child.name}"
            if child.is_symlink() or child.is_junction():
                continue
            if child.is_dir():
                with contextlib.suppress(OSError):
                    for grandchild in child.iterdir():
                        take(grandchild, f"{name}/{grandchild.name}")
            else:
                take(child, name)
    found.sort(key=lambda entry: (-entry[0], entry[1]))
    items = [Item(name, size, source=path) for _, name, path, size in found[:RECORDS]]
    left = [
        LeftOut(name, size, f"older than the newest {RECORDS} export records")
        for _, name, path, size in found[RECORDS:]
    ]
    return items, left


def _saved_images(data: bytes | None) -> list[str] | None:
    """The image files ``project.json``'s bytes ``data`` names; None without
    them, or when they do not read as a project."""
    if data is None:
        return None
    try:
        project = storage.project_from_json(data)
    except storage.ProjectError:
        return None
    return [image.file for image in project.batch.iter_images()]


def plan(state: Path, session: ProjectSession | None) -> Plan:
    """What a diagnostic file written now would hold, for the per-user state
    folder ``state`` (its ``logs`` folder) and the open project's ``session``
    (None: no project is open). Mostly lists: nothing is read but
    ``environment`` and the project's ``project.json``, which is taken as read
    here, so it names the image files listed."""
    data = _json(environment())
    items = [Item(ENVIRONMENT_FILE, len(data), data=data)]
    log_items, left = _log_files(state / logs.LOG_DIR)
    items += log_items
    images: list[Item] = []
    if session is None:
        return Plan(None, None, tuple(items), (), tuple(left), None)
    folder = session.folder
    name = f"{_PROJECT}{storage.PROJECT_FILE}"
    source = folder / storage.PROJECT_FILE
    saved = None  # project.json's bytes, as taken
    if _regular(source) is None:
        left.append(LeftOut(name, None, "not found in the project folder"))
    else:
        try:
            saved = source.read_bytes()
        except OSError as exc:
            left.append(LeftOut(name, None, f"could not be read: {exc.strerror or exc}"))
        else:
            items.append(Item(name, len(saved), data=saved))
    records, older = _records(folder / storage.EXPORTS_DIR)
    items += records
    left += older
    # The image files of the project as open, then those only project.json
    # names: an image removed while saving fails keeps its file until a save.
    named = _saved_images(saved)
    files = [image.file for image in session.project.batch.iter_images()] + (named or [])
    for file in dict.fromkeys(files):
        path = folder / storage.IMAGES_DIR / file
        name = f"{_PROJECT}{storage.IMAGES_DIR}/{file}"
        size = _regular(path)
        if size is None:
            left.append(LeftOut(name, None, "not found as a file in the project's images folder"))
        else:
            images.append(Item(name, size, source=path))
    needs = None if named is None else tuple(f"{_PROJECT}{storage.IMAGES_DIR}/{f}" for f in named)
    return Plan(folder.name, not session.dirty, tuple(items), tuple(images), tuple(left), needs)


def _info(name: str, moment: datetime) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=moment.astimezone().timetuple()[:6])
    info.external_attr = 0o600 << 16  # as zipfile writes text itself: the owner only
    stored = os.path.splitext(name)[1].lower() in _STORED
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    return info


def _add_bytes(archive: zipfile.ZipFile, name: str, data: bytes, moment: datetime) -> dict:
    archive.writestr(_info(name, moment), data)
    return {"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _add(archive: zipfile.ZipFile, item: Item, moment: datetime) -> dict | LeftOut:
    """Write ``item`` into ``archive``: its manifest entry, or why it was left
    out (it could not be opened). An error once it is open propagates."""
    if item.data is not None:
        return _add_bytes(archive, item.name, item.data, moment)
    assert item.source is not None
    try:
        source = item.source.open("rb")
    except OSError as exc:
        return LeftOut(item.name, item.size, f"could not be read: {exc.strerror or exc}")
    with source:
        size = os.fstat(source.fileno()).st_size
        if item.tail and size > LOG_BYTES:
            source.seek(size - LOG_BYTES)
            data = source.read(LOG_BYTES)
            data = data[data.find(b"\n") + 1 :]  # from the start of a line
            entry = _add_bytes(archive, item.name, data, moment)
            entry["note"] = f"the last {len(data)} bytes of a file of {size}"
            return entry
        info = _info(item.name, moment)
        info.file_size = size  # so a file past 2 GiB is written in the ZIP64 format
        digest = hashlib.sha256()
        count = 0
        with archive.open(info, "w") as target:
            while chunk := source.read(_CHUNK):
                digest.update(chunk)
                target.write(chunk)
                count += len(chunk)
    return {"name": item.name, "size": count, "sha256": digest.hexdigest()}


def _counted(count: int, one: str, many: str) -> str:
    return f"{'no' if count == 0 else count} {one if count == 1 else many}"


def _readme(
    plan: Plan, listed: list[dict], left_out: list[LeftOut], images: bool, moment: datetime
) -> str:
    """``README.txt``: what the file holds, as written (``listed``, ``left_out``)."""
    names = [entry["name"] for entry in listed]
    log_files = sum(name.startswith("logs/") for name in names)
    records = sum(name.endswith(_RECORD_SUFFIX) for name in names)
    when = moment.astimezone().strftime("%Y-%m-%d %H:%M (UTC%z)")
    items = [
        f"{ENVIRONMENT_FILE}: the versions of Proteia, Python, the operating system and the"
        " libraries that read and measure images. No computer name, user name or folder.",
        f"logs/: the newest part of Proteia's session log ({_counted(log_files, 'file', 'files')}):"
        " what Proteia did, refused and failed at, and when, naming the projects, proteins,"
        " conditions and image files it worked on. Access tokens are hidden in it, and"
        " folders are written as <projects>, <state>, <temp> or ~."
        if log_files
        else "No session log was found.",
    ]
    if plan.project is None:
        items.append("No project was open: no project files are included.")
    else:
        saved = "" if plan.saved else " (its latest changes could not be saved)"
        if not images:
            taken = "; not its images"
        elif plan.needs is not None and set(plan.needs) <= set(names):
            taken = (
                " and its image files: with them, project/ is a copy of the project that"
                " Proteia can open"
            )
        else:
            taken = (
                " and its image files, but project/ is not a copy of the project that Proteia"
                " can open: not every file it needs could be taken"
            )
        items.append(
            f'project/: the project that was open, "{plan.project}": its {storage.PROJECT_FILE}'
            f" as last saved{saved}, {_counted(records, 'export record', 'export records')}"
            f"{taken}. {storage.PROJECT_FILE} and the records name the project, its proteins,"
            " conditions, samples and image files."
        )
    items.append(
        f"{MANIFEST_FILE}: every other file with its size and SHA-256, and each file left"
        f" out and why ({_counted(len(left_out), 'file', 'files')} left out)."
    )
    lines = [
        "Proteia diagnostic file",
        "",
        textwrap.fill(
            f"Written by Proteia {proteia.__version__} on {when}, to be attached to a bug"
            " report. Proteia sends nothing anywhere: this file goes only where it is"
            " attached by hand.",
            _WIDTH,
        ),
        "",
        "What is inside:",
        "",
        *(
            textwrap.fill(item, _WIDTH, initial_indent="- ", subsequent_indent="  ")
            for item in items
        ),
        "",
        textwrap.fill(
            "Never included: the access token, Proteia's instance files, and images handed to"
            " Proteia but not imported.",
            _WIDTH,
        ),
        "",
    ]
    return "\n".join(lines)


def _free_name(folder: Path, moment: datetime) -> Path:
    stem = f"{FILE_STEM} {bundle_folder_name(moment)}"
    for number in range(1, MAX_NUMBERED + 1):
        path = folder / (f"{stem}.zip" if number == 1 else f"{stem} ({number}).zip")
        if not os.path.lexists(path):
            return path
    raise FileExistsError(f"no free name for a diagnostic file: {MAX_NUMBERED} are taken")


def make_folder(folder: Path) -> None:
    """Make the diagnostics ``folder`` if need be, owner-only on POSIX."""
    folder.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(folder, 0o700)


def _prune(folder: Path, written: Path) -> None:
    """Delete the diagnostic files past the :data:`KEPT` newest, never
    ``written``, and what an earlier write left unfinished (one runs at a time)."""
    found: list[tuple[float, str, Path]] = []
    with contextlib.suppress(OSError):  # the file is written: a failure here is not its
        for path in folder.iterdir():
            with contextlib.suppress(OSError):
                if path.name.endswith(_PART) and path.name.startswith("."):
                    path.unlink()
                elif path.name.startswith(f"{FILE_STEM} ") and path.suffix == ".zip":
                    found.append((path.stat().st_mtime, path.name, path))
    found.sort(reverse=True)
    for _, _, path in found[KEPT:]:
        if path != written:
            with contextlib.suppress(OSError):
                path.unlink()


def write(plan: Plan, folder: Path, *, images: bool, moment: datetime) -> Written:
    """Write the diagnostic file ``plan`` lists into ``folder`` (made if need
    be), with the project's images if ``images``; ``moment`` (an aware time)
    names it and dates its files. A file that cannot be opened is left out and
    the manifest says why; any other error propagates, and nothing is left.
    With no project, there are no images to take, and ``images`` is ignored."""
    images = images and plan.project is not None
    with _writing:
        make_folder(folder)
        fd, temp = tempfile.mkstemp(dir=folder, prefix=".", suffix=f".zip{_PART}")
        try:
            with os.fdopen(fd, "wb") as raw:
                with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
                    listed: list[dict] = []
                    left_out = list(plan.omitted(images))
                    for item in plan.files(images):
                        entry = _add(archive, item, moment)
                        if isinstance(entry, LeftOut):
                            left_out.append(entry)
                        else:
                            listed.append(entry)
                    readme = _readme(plan, listed, left_out, images, moment)
                    listed.insert(0, _add_bytes(archive, README_FILE, readme.encode(), moment))
                    manifest = {
                        "format": MANIFEST_FORMAT,
                        "written": format_timestamp(moment),
                        "proteia": proteia.__version__,
                        "project": plan.project,
                        "project_saved": plan.saved,
                        "images_included": images,
                        "files": listed,
                        "left_out": [
                            {"name": item.name, "size": item.size, "reason": item.reason}
                            for item in left_out
                        ],
                    }
                    _add_bytes(archive, MANIFEST_FILE, _json(manifest), moment)
                raw.flush()
                os.fsync(raw.fileno())
            path = _free_name(folder, moment)
            os.replace(temp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(temp)
            raise
        _prune(folder, path)
        return Written(path, len(listed) + 1, path.stat().st_size, len(left_out))
