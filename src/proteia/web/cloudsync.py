# SPDX-License-Identifier: Apache-2.0
"""Whether the projects root lies in a folder a cloud sync service uploads (#139).

The projects root is ``<Documents>/Proteia`` (:mod:`proteia.web.projects`), and
the Documents folder is often one a sync service uploads: OneDrive's backup of
it moves it into the OneDrive folder, say. Proteia itself sends nothing, but
every project's files, the images imported included, then go to that service,
so the page tells the user so, once (``GET /api/notices``,
:mod:`proteia.web.api`), until the notice is dismissed for good
(:func:`dismiss`).

The folders recognised (:func:`synced_folders`), each named by its service:

* OneDrive, on Windows: the folders in the ``OneDrive``, ``OneDriveConsumer``
  and ``OneDriveCommercial`` environment variables, which OneDrive sets, and
  every account's ``UserFolder`` under
  ``HKEY_CURRENT_USER\\Software\\Microsoft\\OneDrive\\Accounts`` (the personal
  account's and each work or school account's);
* Dropbox: every account's ``path`` in Dropbox's ``info.json``, in
  ``%APPDATA%\\Dropbox`` and ``%LOCALAPPDATA%\\Dropbox`` on Windows and in
  ``~/.dropbox`` elsewhere;
* iCloud Drive: its folder, ``%USERPROFILE%\\iCloudDrive`` on Windows and
  ``~/Library/Mobile Documents`` on macOS;
* on macOS, every folder in ``~/Library/CloudStorage``, where OneDrive, Google
  Drive, Dropbox and Box keep what they sync, named by the service its name
  begins with;
* Google Drive for desktop, on Windows: a drive labelled ``Google Drive``, the
  drive it streams My Drive on.

Not recognised: a folder a service backs up where it is rather than in a
folder of its own (Google Drive's backup of folders on the computer, say),
Google Drive mirrored to a folder, a SharePoint library OneDrive syncs outside
the OneDrive folder, and other services. The check only reads these settings
and folders, on this computer, and changes nothing.

The projects root and each folder are compared resolved (links followed, as far
as the folders exist), as their system compares paths: on Windows ignoring case
and either slash (:func:`containing`). The check never fails for a setting or
file it cannot read: it skips it.

The notices dismissed are recorded in ``notices.json`` in the per-user state
folder (:func:`proteia.web.launch.state_dir`), never in a project: the notice is
about where every project is kept.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Final

from proteia.core.storage import write_atomic

ONEDRIVE: Final = "OneDrive"
DROPBOX: Final = "Dropbox"
ICLOUD: Final = "iCloud Drive"
GOOGLE_DRIVE: Final = "Google Drive"
# The environment variables OneDrive sets to its folders.
ONEDRIVE_VARIABLES: Final = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
ONEDRIVE_ACCOUNTS: Final = r"Software\Microsoft\OneDrive\Accounts"  # in HKEY_CURRENT_USER
GOOGLE_DRIVE_LABEL: Final = "Google Drive"  # the volume label of the drive it streams on
# A folder in ~/Library/CloudStorage is named "<service>-<account>".
CLOUD_STORAGE_SERVICES: Final = {
    "OneDrive": ONEDRIVE,
    "GoogleDrive": GOOGLE_DRIVE,
    "Dropbox": DROPBOX,
    "Box": "Box",
}
OTHER_SERVICE: Final = "a cloud storage service"
MAX_INFO_BYTES: Final = 1024 * 1024  # of Dropbox's info.json read; it holds a few lines
MAX_ACCOUNTS: Final = 100  # OneDrive accounts read from the registry

NOTICES_FILE: Final = "notices.json"  # in the per-user state folder
CLOUD_SYNC: Final = "cloud_sync"  # the notice that the projects folder is synced


@dataclass(frozen=True)
class SyncedFolder:
    """A folder a sync service uploads: ``service`` names the service, as the
    page shows it; ``path`` is the folder as found; ``source`` says where it
    was found, for the session log, and names no path."""

    service: str
    path: str
    source: str


@dataclass(frozen=True)
class Computer:
    """What the check reads of a computer (:func:`this_computer`); a test gives
    its own. ``system`` is ``windows``, ``macos`` or ``posix``; paths are text,
    as that system spells them. ``onedrive_accounts`` gives each OneDrive
    account's name and folder from the registry; ``read_text`` a file's text,
    ``list_folder`` the names in a folder and ``volume_label`` a drive's label
    (``C:\\``; None if it has none), each raising ``OSError`` or
    ``ValueError`` when it cannot; ``resolve`` a path with its links followed."""

    system: str
    environ: Mapping[str, str]
    home: str
    onedrive_accounts: Callable[[], Iterable[tuple[str, str]]]
    read_text: Callable[[str], str]
    list_folder: Callable[[str], Iterable[str]]
    volume_label: Callable[[str], str | None]
    resolve: Callable[[str], str]


def _account(name: str) -> str:
    """An account's name for the log, if it is one plain word (``Personal``,
    ``Business1``): the log repeats nothing else a setting holds."""
    plain = name.isascii() and name.isalnum() and len(name) <= 32
    return f"account {name}" if plain else "an account"


def onedrive_folders(
    environ: Mapping[str, str], accounts: Iterable[tuple[str, str]]
) -> list[SyncedFolder]:
    """OneDrive's folders: those its environment variables name
    (:data:`ONEDRIVE_VARIABLES`), then those of ``accounts``, the registry's
    account names and ``UserFolder`` values; an empty one is left out."""
    found = [
        SyncedFolder(ONEDRIVE, environ[name], f"the {name} environment variable")
        for name in ONEDRIVE_VARIABLES
        if environ.get(name)
    ]
    found += [
        SyncedFolder(ONEDRIVE, folder, f"the registry, {_account(name)}")
        for name, folder in accounts
        if folder
    ]
    return found


def dropbox_folders(texts: Iterable[str]) -> list[SyncedFolder]:
    """Dropbox's folders: every account's ``path`` in ``texts``, the text of
    its ``info.json`` files (``{"personal": {"path": ...}, "business": ...}``).
    A text that is not such JSON, or an account without a path, is skipped."""
    found = []
    for text in texts:
        try:
            info = json.loads(text)
        except ValueError:
            continue
        if not isinstance(info, dict):
            continue
        for name, account in info.items():
            path = account.get("path") if isinstance(account, dict) else None
            if isinstance(path, str) and path:
                found.append(SyncedFolder(DROPBOX, path, f"Dropbox's info.json, {_account(name)}"))
    return found


def cloud_storage_folders(parent: str, names: Iterable[str]) -> list[SyncedFolder]:
    """The folders in macOS's ``~/Library/CloudStorage`` (``parent``), from their
    ``names``: each ``<service>-<account>``, named by its service
    (:data:`CLOUD_STORAGE_SERVICES`, else :data:`OTHER_SERVICE`). A hidden
    name (``.DS_Store``) is skipped."""
    return [
        SyncedFolder(
            CLOUD_STORAGE_SERVICES.get(name.split("-", 1)[0], OTHER_SERVICE),
            str(PurePosixPath(parent, name)),
            "a folder in ~/Library/CloudStorage",
        )
        for name in names
        if name and not name.startswith(".")
    ]


def _flavour(windows: bool) -> type[PurePath]:
    return PureWindowsPath if windows else PurePosixPath


def containing(root: str, folders: Iterable[SyncedFolder], *, windows: bool) -> SyncedFolder | None:
    """The folder of ``folders`` that ``root`` is or lies in: the innermost if
    it lies in several, the first found of those that are one folder; None if
    none. Compared as Windows compares paths if ``windows`` (ignoring case,
    either slash and trailing separators), else as spelled; a relative folder
    never matches. Pass the paths resolved (:func:`check` does)."""
    flavour = _flavour(windows)
    target = flavour(root)
    best: tuple[int, SyncedFolder] | None = None
    for folder in folders:
        path = flavour(folder.path)
        if not path.is_absolute() or not target.is_relative_to(path):
            continue
        depth = len(path.parts)
        if best is None or depth > best[0]:
            best = (depth, folder)
    return None if best is None else best[1]


def _read_all(computer: Computer, paths: Iterable[str]) -> list[str]:
    """The text of each file of ``paths`` that can be read."""
    texts = []
    for path in paths:
        try:
            texts.append(computer.read_text(path))
        except (OSError, ValueError):
            continue
    return texts


def synced_folders(root: str, computer: Computer) -> list[SyncedFolder]:
    """The folders a sync service uploads on ``computer``, as the module
    docstring lists them, found where the settings say, not yet resolved. The
    Google Drive drive is looked for only as the drive of ``root``."""
    windows = computer.system == "windows"
    flavour = _flavour(windows)
    home = flavour(computer.home)
    found: list[SyncedFolder] = []
    if windows:
        try:
            accounts = list(computer.onedrive_accounts())
        except (OSError, ValueError):
            accounts = []
        found += onedrive_folders(computer.environ, accounts)
        bases = [computer.environ.get(name) for name in ("APPDATA", "LOCALAPPDATA")]
        infos = [str(flavour(base, "Dropbox", "info.json")) for base in bases if base]
    else:
        infos = [str(home / ".dropbox" / "info.json")]
    found += dropbox_folders(_read_all(computer, infos))
    if windows:
        found.append(SyncedFolder(ICLOUD, str(home / "iCloudDrive"), "iCloud Drive's folder"))
        drive = PureWindowsPath(root).anchor
        try:
            label = computer.volume_label(drive) if drive else None
        except (OSError, ValueError):
            label = None
        if label == GOOGLE_DRIVE_LABEL:
            found.append(SyncedFolder(GOOGLE_DRIVE, drive, "the drive's volume label"))
    elif computer.system == "macos":
        library = home / "Library"
        found.append(
            SyncedFolder(ICLOUD, str(library / "Mobile Documents"), "iCloud Drive's folder")
        )
        storage = str(library / "CloudStorage")
        try:
            names = list(computer.list_folder(storage))
        except (OSError, ValueError):
            names = []
        found += cloud_storage_folders(storage, names)
    return found


def check(root: str | os.PathLike[str], computer: Computer | None = None) -> SyncedFolder | None:
    """The folder a sync service uploads that ``root`` lies in, on ``computer``
    (default: :func:`this_computer`), both resolved (:func:`containing`); None
    if it lies in none that is recognised. A folder that cannot be resolved is
    compared as found."""
    computer = this_computer() if computer is None else computer
    windows = computer.system == "windows"
    flavour = _flavour(windows)
    given = os.fspath(root)
    try:
        resolved = computer.resolve(given)
    except (OSError, ValueError):
        resolved = given
    folders = []
    for folder in synced_folders(resolved, computer):
        if not flavour(folder.path).is_absolute():
            continue  # resolving would make it absolute, relative to the working folder
        try:
            folders.append(replace(folder, path=computer.resolve(folder.path)))
        except (OSError, ValueError):
            folders.append(folder)
    return containing(resolved, folders, windows=windows)


# --- This computer ---


def this_computer() -> Computer:
    """What the check reads of this computer, now."""
    if os.name == "nt":
        system = "windows"
    elif sys.platform == "darwin":
        system = "macos"
    else:
        system = "posix"
    return Computer(
        system=system,
        environ=os.environ,  # on Windows it looks names up ignoring case, as Windows does
        home=str(Path.home()),
        onedrive_accounts=_onedrive_accounts,
        read_text=_read_text,
        list_folder=lambda path: [entry.name for entry in os.scandir(path)],
        volume_label=_volume_label,
        resolve=lambda path: str(Path(path).resolve()),
    )


def _read_text(path: str) -> str:
    with open(path, "rb") as file:
        return file.read(MAX_INFO_BYTES).decode("utf-8")


def _onedrive_accounts() -> list[tuple[str, str]]:
    """Each OneDrive account's name and ``UserFolder`` in the registry; none on
    a computer without OneDrive, or if the key cannot be read."""
    if os.name != "nt":
        return []
    import winreg

    found = []
    try:
        accounts = winreg.OpenKey(winreg.HKEY_CURRENT_USER, ONEDRIVE_ACCOUNTS)
    except OSError:
        return []
    with accounts:
        for index in range(MAX_ACCOUNTS):
            try:
                name = winreg.EnumKey(accounts, index)
            except OSError:
                break  # no more
            try:
                with winreg.OpenKey(accounts, name) as account:
                    value, kind = winreg.QueryValueEx(account, "UserFolder")
            except OSError:
                continue  # an account with no folder (signed out)
            if kind == winreg.REG_EXPAND_SZ and isinstance(value, str):
                value = winreg.ExpandEnvironmentStrings(value)
            if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) and isinstance(value, str):
                found.append((name, value))
    return found


def _volume_label(drive: str) -> str | None:
    """The volume label of ``drive`` (``G:\\``), or None if it has none or it
    cannot be read."""
    if os.name != "nt":
        return None
    import ctypes

    label = ctypes.create_unicode_buffer(261)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(drive), label, len(label), None, None, None, None, 0
    )
    return (label.value or None) if ok else None


# --- The notices dismissed ---


def dismissed(state: Path) -> frozenset[str]:
    """The notices dismissed for good, as ``notices.json`` in the per-user
    state folder ``state`` records them: none if it is missing or unreadable."""
    try:
        doc = json.loads((state / NOTICES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()
    names = doc.get("dismissed") if isinstance(doc, dict) else None
    if not isinstance(names, list):
        return frozenset()
    return frozenset(name for name in names if isinstance(name, str))


def dismiss(state: Path, notice: str) -> None:
    """Record ``notice`` as dismissed for good in ``notices.json`` in the state
    folder ``state``, with those dismissed before; ``OSError`` if it cannot be
    written."""
    names = sorted(dismissed(state) | {notice})
    state.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"dismissed": names}).encode("utf-8")
    write_atomic(state / NOTICES_FILE, data, private=True)
