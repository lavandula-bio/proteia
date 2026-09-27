# SPDX-License-Identifier: Apache-2.0
"""Where the web app keeps projects: one folder per project in an app-managed
projects root, ``<Documents>/Proteia`` (the maintainer's choice for v0.1, #52).

A project's name is its folder's name. Paths come only from the server: a name
from the client must be one plain folder name that every supported file system
accepts (:func:`project_name`), and it is only ever joined to the projects root.
Names are unique ignoring case and look-alike spellings, as file systems on
Windows and macOS compare them. A project made from an image file is named
after it (:func:`name_from_file`), numbered if that name is taken
(:func:`free_name`). A project set up as it is created (the sample project, a
hand-off's images) is created by :func:`create_set_up`, which removes its folder
if the setup fails.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

from proteia.core import storage
from proteia.core.model import format_timestamp
from proteia.core.names import TextError, clean_text, name_key
from proteia.core.session import Clock, ProjectSession, new_project, utc_now

ROOT_NAME: Final = "Proteia"
MAX_NAME: Final = 100  # characters; well inside every file system's limit
MAX_NUMBERED: Final = 1000  # names free_name tries: the plain one, then numbered
# The longest name taken from a file: room left for free_name's " (1000)".
MAX_FILE_NAME: Final = MAX_NAME - len(f" ({MAX_NUMBERED})")
# The name of a project made from files whose names give none.
FALLBACK_NAME: Final = "Imported images"
# Characters Windows refuses in a file name.
_FORBIDDEN: Final = frozenset('<>:"/\\|?*')
# Device names Windows reserves, with or without an extension: COM and LPT with
# 0-9 and the superscript digits 1-3 as well.
_RESERVED: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{p}{d}" for p in ("COM", "LPT") for d in "0123456789¹²³"}
)


class ProjectNameError(ValueError):
    """A name that cannot be a project folder's name."""


class ProjectExistsError(ValueError):
    """A project with this name, ignoring case and look-alikes, already exists."""


class ProjectNotFoundError(LookupError):
    """No project with this name in the projects root."""


class FolderLeftError(OSError):
    """A new project's setup failed (:func:`create_set_up`), and its folder
    could not all be removed after: it is left in the projects root,
    unfinished."""


def _windows_documents() -> Path | None:
    """The Documents known folder, wherever it is redirected (e.g. to OneDrive)."""
    import ctypes
    import uuid
    from ctypes import wintypes

    class _Guid(ctypes.Structure):
        _fields_ = [
            ("data1", wintypes.DWORD),
            ("data2", wintypes.WORD),
            ("data3", wintypes.WORD),
            ("data4", wintypes.BYTE * 8),
        ]

    documents = uuid.UUID("{FDD39AD0-238F-46AF-ADB4-6C85480369C7}")  # FOLDERID_Documents
    guid = _Guid.from_buffer_copy(documents.bytes_le)
    path = ctypes.c_wchar_p()
    try:
        result = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(guid), 0, None, ctypes.byref(path)
        )
        return Path(path.value) if result == 0 and path.value else None
    except (OSError, AttributeError):
        return None
    finally:
        ctypes.windll.ole32.CoTaskMemFree(path)


def projects_root() -> Path:
    """``<Documents>/Proteia``; it is created when the first project is."""
    documents = _windows_documents() if os.name == "nt" else None
    return (documents or Path.home() / "Documents") / ROOT_NAME


def project_name(name: object) -> str:
    """``name`` as stored (:func:`~proteia.core.names.clean_text`) if it can name a
    project folder, else :class:`ProjectNameError`."""
    if not isinstance(name, str):
        raise ProjectNameError(f"a project name must be text, not {name!r}")
    try:
        text = clean_text(name)
    except TextError as exc:
        raise ProjectNameError(str(exc)) from exc
    if len(text) > MAX_NAME:
        raise ProjectNameError(f"a project name has at most {MAX_NAME} characters")
    bad = sorted(set(text) & _FORBIDDEN)
    if bad:
        raise ProjectNameError(f"a project name cannot contain {' '.join(bad)}")
    if text.endswith(".") or text in (".", ".."):
        raise ProjectNameError("a project name cannot end with a dot")
    if text.split(".")[0].rstrip().upper() in _RESERVED:
        raise ProjectNameError(f"{text!r} is a name Windows reserves for devices")
    return text


def name_from_file(original_name: str) -> str:
    """A project name for images imported from the file ``original_name``: its
    stem as stored (:func:`~proteia.core.names.clean_text`), with each character
    Windows refuses in a file name (``<>:"/\\|?*``) made a space, cut to
    :data:`MAX_FILE_NAME` characters (so :func:`free_name` can still number
    it) and without spaces and dots at the end; :data:`FALLBACK_NAME` when
    nothing is left or :func:`project_name` still refuses it (a device name
    such as ``CON``, or a control character). ``β-actin 10 µM.tif`` gives
    ``β-actin 10 µM``, ``a:b.png`` ``a b``, ``blot..tif`` ``blot``."""
    try:
        text = clean_text(PurePosixPath(original_name).stem)
    except TextError:
        return FALLBACK_NAME
    text = " ".join("".join(" " if c in _FORBIDDEN else c for c in text).split())
    text = text[:MAX_FILE_NAME].rstrip(" .")
    try:
        return project_name(text) if text else FALLBACK_NAME
    except ProjectNameError:
        return FALLBACK_NAME


@dataclass(frozen=True)
class ProjectEntry:
    """A project in the projects root."""

    name: str
    modified: str  # when project.json last changed (UTC, the log's timestamp form)


def list_projects(root: Path) -> list[ProjectEntry]:
    """The projects in ``root``, most recently changed first. Folders without a
    ``project.json``, or whose names :func:`project_name` refuses, are left out."""
    if not root.is_dir():
        return []
    entries = []
    for child in root.iterdir():
        project_file = child / storage.PROJECT_FILE
        try:
            project_name(child.name)  # a name the app could have made
            if not project_file.is_file():
                continue
            changed = datetime.fromtimestamp(project_file.stat().st_mtime, UTC)
        except (ProjectNameError, OSError):
            continue
        entries.append(ProjectEntry(name=child.name, modified=format_timestamp(changed)))
    return sorted(entries, key=lambda entry: (entry.modified, entry.name), reverse=True)


def _existing(root: Path, name: str) -> Path | None:
    """The folder in ``root`` named exactly ``name``, else the first whose name
    matches it ignoring case and look-alike spellings (on a case-sensitive file
    system two such folders can exist; the exact one wins)."""
    if not root.is_dir():
        return None
    children = sorted(root.iterdir())
    for child in children:
        if child.name == name:
            return child
    key = name_key(name)
    for child in children:
        if name_key(child.name) == key:
            return child
    return None


def names_in(root: Path) -> list[str]:
    """The names of the files and folders in ``root``, listed once; none if it
    is not a folder (yet)."""
    if not root.is_dir():
        return []
    return [child.name for child in root.iterdir()]


def free_name(root: Path, name: object, *, existing: Iterable[str] | None = None) -> str:
    """``name`` as stored (:func:`project_name`), or ``name (2)``, ``name (3)``
    and so on: the first that no file or folder in ``root`` has, ignoring case
    and look-alike spellings (as :func:`_existing` compares them), as export
    folders are numbered. ``root`` is listed once (:func:`names_in`), or not at
    all when ``existing`` gives that listing: for a caller that names several
    projects at once. :class:`ProjectExistsError` once :data:`MAX_NUMBERED`
    are taken."""
    text = project_name(name)
    taken = {name_key(other) for other in (names_in(root) if existing is None else existing)}
    for number in range(1, MAX_NUMBERED + 1):
        candidate = text if number == 1 else project_name(f"{text} ({number})")
        if name_key(candidate) not in taken:
            return candidate
    raise ProjectExistsError(f"no free project name for {text!r}: {MAX_NUMBERED} are taken")


def create_project(root: Path, name: object, *, clock: Clock = utc_now) -> ProjectSession:
    """Create the project ``name`` in ``root`` and open it."""
    text = project_name(name)
    if _existing(root, text) is not None:
        raise ProjectExistsError(f"a project named {text!r} already exists")
    root.mkdir(parents=True, exist_ok=True)
    return new_project(root / text, clock=clock)


def create_set_up(
    root: Path,
    name: object,
    set_up: Callable[[ProjectSession], None],
    *,
    clock: Clock = utc_now,
    what: str = "the new project",
) -> ProjectSession:
    """Create the project ``name`` in ``root`` (:func:`create_project`), set it
    up with ``set_up`` and open it; saved before it is returned, so an
    ``OSError`` or :class:`~proteia.core.storage.ProjectError` while saving
    propagates too. If ``set_up`` or that save fails, the folder is removed
    with everything written into it, and the error propagates. If some of it
    cannot be removed (a file in it held open by another program, which
    Windows refuses to delete), the folder stays, without a ``project.json``:
    the Projects dialog does not list it, but its name is taken.
    :class:`FolderLeftError` then says so, naming ``what`` and the folder, in
    place of the error."""
    session = create_project(root, name, clock=clock)
    try:
        set_up(session)
        if session.dirty:  # an autosave failed: fail now, not when it is next edited
            session.save()
    except BaseException as exc:
        shutil.rmtree(session.folder, ignore_errors=True)
        if isinstance(exc, Exception) and session.folder.exists():
            raise FolderLeftError(
                f"{what} could not be set up ({exc}), and its unfinished folder"
                f" {session.folder.name!r} could not be removed: delete it from the projects"
                " folder"
            ) from exc
        raise
    return session


def project_folder(root: Path, name: object) -> Path:
    """The folder of the project ``name`` in ``root``, to open: named exactly
    ``name`` as stored (:func:`project_name`), else ignoring case and look-alike
    spellings (:func:`_existing`); :class:`ProjectNotFoundError` if it holds no
    project."""
    folder = _existing(root, project_name(name))
    if folder is None or not (folder / storage.PROJECT_FILE).is_file():
        raise ProjectNotFoundError(f"no project named {name!r}")
    return folder


def reveal(folder: Path) -> None:
    """Show ``folder`` in the system file manager."""
    if os.name == "nt":
        os.startfile(folder)  # a folder the server chose
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder)], check=False)
    else:
        subprocess.run(["xdg-open", str(folder)], check=False)
