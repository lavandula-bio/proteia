# SPDX-License-Identifier: Apache-2.0
"""Where the web app keeps projects: one folder per project in an app-managed
projects root, ``<Documents>/Proteia`` (the maintainer's choice for v0.1, #52).

A project's name is its folder's name. Paths come only from the server: a name
from the client must be one plain folder name that every supported file system
accepts (:func:`project_name`), and it is only ever joined to the projects root.
Names are unique ignoring case and look-alike spellings, as file systems on
Windows and macOS compare them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from proteia.core import storage
from proteia.core.model import format_timestamp
from proteia.core.names import TextError, clean_text, name_key
from proteia.core.session import Clock, ProjectSession, new_project, open_project, utc_now

ROOT_NAME: Final = "Proteia"
MAX_NAME: Final = 100  # characters; well inside every file system's limit
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


def create_project(root: Path, name: object, *, clock: Clock = utc_now) -> ProjectSession:
    """Create the project ``name`` in ``root`` and open it."""
    text = project_name(name)
    if _existing(root, text) is not None:
        raise ProjectExistsError(f"a project named {text!r} already exists")
    root.mkdir(parents=True, exist_ok=True)
    return new_project(root / text, clock=clock)


def open_named(root: Path, name: object, *, clock: Clock = utc_now) -> ProjectSession:
    """Open the project ``name`` in ``root``; storage errors propagate."""
    folder = _existing(root, project_name(name))
    if folder is None or not (folder / storage.PROJECT_FILE).is_file():
        raise ProjectNotFoundError(f"no project named {name!r}")
    return open_project(folder, clock=clock)


def reveal(folder: Path) -> None:
    """Show ``folder`` in the system file manager."""
    if os.name == "nt":
        os.startfile(folder)  # a folder the server chose
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder)], check=False)
    else:
        subprocess.run(["xdg-open", str(folder)], check=False)
