# SPDX-License-Identifier: Apache-2.0
"""The notice that the projects folder lies in a folder a sync service uploads
(#139): what the check recognises, read from stand-in settings (the
environment, the registry, Dropbox's info.json, the folders), how it compares
paths, the dismissal remembered in the per-user state folder, the routes and
the page."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from conftest import FakeClock
from proteia.web import api, cloudsync, launch, server
from proteia.web.cloudsync import Computer, SyncedFolder
from test_web import _code, _function, _Tags
from test_web_api import Client

ONEDRIVE = r"D:\Sync\OneDrive"
ROOT = ONEDRIVE + r"\Documents\Proteia"


def _given[T](given: dict[str, T | Exception], key: str) -> T:
    """What ``given`` holds for ``key``, raised if it is an error; a missing
    file's error if it holds nothing."""
    if key not in given:
        raise FileNotFoundError(key)
    value = given[key]
    if isinstance(value, Exception):
        raise value
    return value


@dataclass
class FakeComputer:
    """Stand-in settings and files for :class:`Computer`, paths spelled as
    ``system`` spells them; an error given in place of a value is raised."""

    system: str = "windows"
    environ: dict[str, str] = field(default_factory=dict)
    home: str = r"C:\Users\ann"
    accounts: list[tuple[str, str]] | Exception = field(default_factory=list)
    files: dict[str, str | Exception] = field(default_factory=dict)
    folders: dict[str, list[str] | Exception] = field(default_factory=dict)
    labels: dict[str, str | Exception] = field(default_factory=dict)  # a drive's volume label
    links: dict[str, str | Exception] = field(default_factory=dict)  # a path -> where it leads
    read: list[str] = field(default_factory=list)  # the files asked for, in order

    def onedrive_accounts(self) -> list[tuple[str, str]]:
        if isinstance(self.accounts, Exception):
            raise self.accounts
        return self.accounts

    def read_text(self, path: str) -> str:
        self.read.append(path)
        return _given(self.files, path)

    def list_folder(self, path: str) -> list[str]:
        return _given(self.folders, path)

    def volume_label(self, drive: str) -> str | None:
        return _given(self.labels, drive) if drive in self.labels else None

    def resolve(self, path: str) -> str:
        return _given(self.links, path) if path in self.links else path

    def computer(self) -> Computer:
        return Computer(
            system=self.system,
            environ=self.environ,
            home=self.home,
            onedrive_accounts=self.onedrive_accounts,
            read_text=self.read_text,
            list_folder=self.list_folder,
            volume_label=self.volume_label,
            resolve=self.resolve,
        )


def found(folders: Iterable[SyncedFolder]) -> list[tuple[str, str, str]]:
    return [(folder.service, folder.path, folder.source) for folder in folders]


# --- What is recognised ---


def test_onedrive_folders_come_from_its_environment_variables_and_accounts():
    lab = r"C:\Users\ann\OneDrive - Lab"
    environ = {
        "OneDrive": ONEDRIVE,
        "OneDriveConsumer": ONEDRIVE,
        "OneDriveCommercial": lab,
        "Path": r"C:\Windows",
        "OneDriveOther": r"E:\Else",  # not one OneDrive sets
    }
    accounts = [("Personal", ONEDRIVE), ("Business1", lab), ("Business2", "")]  # 2: no folder
    assert found(cloudsync.onedrive_folders(environ, accounts)) == [
        ("OneDrive", ONEDRIVE, "the OneDrive environment variable"),
        ("OneDrive", ONEDRIVE, "the OneDriveConsumer environment variable"),
        ("OneDrive", lab, "the OneDriveCommercial environment variable"),
        ("OneDrive", ONEDRIVE, "the registry, account Personal"),
        ("OneDrive", lab, "the registry, account Business1"),
    ]
    assert cloudsync.onedrive_folders({"OneDrive": ""}, []) == []


def test_an_account_name_the_log_should_not_repeat_is_left_out():
    # The source goes to the session log: an account's name that is not one
    # plain word is not repeated there.
    (folder,) = cloudsync.onedrive_folders({}, [("Business1 x@y", ONEDRIVE)])
    assert folder.source == "the registry, an account"


def test_dropbox_folders_come_from_each_account_in_its_info_file():
    info = {
        "personal": {"path": r"D:\Sync\Dropbox", "host": 1, "is_team": False},
        "business": {"path": r"D:\Sync\Dropbox (Lab)", "host": 2, "is_team": True},
        "odd": {"path": 3},
        "other": "text",
    }
    texts = [json.dumps(info), "not JSON", "[1, 2]", json.dumps({"personal": {"path": ""}})]
    assert found(cloudsync.dropbox_folders(texts)) == [
        ("Dropbox", r"D:\Sync\Dropbox", "Dropbox's info.json, account personal"),
        ("Dropbox", r"D:\Sync\Dropbox (Lab)", "Dropbox's info.json, account business"),
    ]


def test_the_macos_cloud_storage_folders_are_named_by_their_service():
    parent = "/Users/ann/Library/CloudStorage"
    names = [
        "OneDrive-Personal",
        "OneDrive-SharedLibraries-Lab",
        "GoogleDrive-ann@example.org",
        "Dropbox",
        "Box-Box",
        "pCloud Drive",
        ".DS_Store",
    ]
    folders = cloudsync.cloud_storage_folders(parent, names)
    assert [(f.service, f.path) for f in folders] == [
        ("OneDrive", f"{parent}/OneDrive-Personal"),
        ("OneDrive", f"{parent}/OneDrive-SharedLibraries-Lab"),
        ("Google Drive", f"{parent}/GoogleDrive-ann@example.org"),
        ("Dropbox", f"{parent}/Dropbox"),
        ("Box", f"{parent}/Box-Box"),
        ("a cloud storage service", f"{parent}/pCloud Drive"),
    ]
    # The log names the service only: never the folder, which can name the account.
    assert {f.source for f in folders} == {"a folder in ~/Library/CloudStorage"}


# --- Comparing paths ---


def folder(path: str, service: str = "OneDrive") -> SyncedFolder:
    return SyncedFolder(service, path, "a test")


@pytest.mark.parametrize(
    ("synced", "root", "inside"),
    [
        (ONEDRIVE, ROOT, True),
        (ONEDRIVE, ONEDRIVE, True),  # the synced folder itself
        (r"d:\sync\onedrive", ROOT, True),  # case ignored
        (ONEDRIVE, r"D:\SYNC\ONEDRIVE\DOCUMENTS\PROTEIA", True),
        ("C:\\Users\\Ärzte\\OneDrive", "c:\\users\\ärzte\\onedrive\\Proteia", True),
        (ONEDRIVE + "\\", ROOT, True),  # trailing separators
        (ONEDRIVE + "\\\\", ROOT + "\\", True),
        ("D:/Sync/OneDrive/", ROOT, True),  # either slash
        (r"D:\Sync\OneDrive - Lab", ROOT, False),  # a name that only begins alike
        (ONEDRIVE, r"D:\Sync\OneDrive - Lab\Proteia", False),
        (ONEDRIVE, r"C:\Sync\OneDrive\Documents\Proteia", False),  # another drive
        (ROOT + r"\Blot", ROOT, False),  # a folder inside the root
        ("OneDrive", ROOT, False),  # relative: never matched
        ("D:Sync", ROOT, False),  # relative to the drive's current folder
        ("", ROOT, False),
        (r"\\server\share", r"\\SERVER\Share\Documents\Proteia", True),
        ("G:\\", r"G:\My Drive\Proteia", True),  # a whole drive
    ],
)
def test_a_root_is_inside_a_synced_folder_as_windows_compares_paths(synced, root, inside):
    match = cloudsync.containing(root, [folder(synced)], windows=True)
    assert (match is not None) == inside


@pytest.mark.parametrize(
    ("synced", "root", "inside"),
    [
        ("/Users/ann/Dropbox", "/Users/ann/Dropbox/Documents/Proteia", True),
        ("/Users/ann/Dropbox/", "/Users/ann/Dropbox/Documents/Proteia/", True),
        ("/Users/ann/Dropbox", "/Users/ann/dropbox/Documents/Proteia", False),  # case kept
        ("/Users/ann/Dropbox", "/Users/ann/Dropbox (Lab)/Proteia", False),
        ("Dropbox", "/Users/ann/Dropbox/Proteia", False),
    ],
)
def test_elsewhere_paths_are_compared_as_they_are_spelled(synced, root, inside):
    match = cloudsync.containing(root, [folder(synced, "Dropbox")], windows=False)
    assert (match is not None) == inside


def test_the_innermost_synced_folder_names_the_service():
    folders = [
        folder(r"D:\Sync", "Dropbox"),
        folder(ONEDRIVE, "OneDrive"),
        folder(r"D:\Sync\OneDrive\Documents", "OneDrive (again)"),
        folder(r"D:\Sync\OneDrive\Documents", "OneDrive (a third time)"),
    ]
    match = cloudsync.containing(ROOT, folders, windows=True)
    assert match is not None and match.service == "OneDrive (again)"  # the first of equals
    assert cloudsync.containing(r"E:\Proteia", folders, windows=True) is None


# --- The check, on stand-in computers ---


def test_the_check_finds_documents_redirected_into_onedrive():
    fake = FakeComputer(
        environ={"OneDrive": ONEDRIVE}, accounts=[("Personal", ONEDRIVE)], labels={"D:\\": "Data"}
    )
    match = cloudsync.check(ROOT, fake.computer())
    assert match == SyncedFolder("OneDrive", ONEDRIVE, "the OneDrive environment variable")


def test_the_check_reads_the_registry_when_the_environment_has_nothing():
    lab = r"C:\Users\ann\OneDrive - Lab"
    fake = FakeComputer(accounts=[("Business1", lab)])
    match = cloudsync.check(lab + r"\Documents\Proteia", fake.computer())
    assert match is not None
    assert (match.service, match.source) == ("OneDrive", "the registry, account Business1")


def test_the_check_compares_the_paths_resolved():
    # Documents is a link into the OneDrive folder, and the variable spells the
    # folder otherwise: both are compared where they lead.
    documents = r"C:\Users\ann\Documents\Proteia"
    fake = FakeComputer(
        environ={"OneDrive": r"C:\Users\ann\OD"},
        links={documents: ROOT, r"C:\Users\ann\OD": ONEDRIVE},
    )
    match = cloudsync.check(documents, fake.computer())
    assert match is not None and match.path == ONEDRIVE
    fake.links.clear()
    assert cloudsync.check(documents, fake.computer()) is None


def test_the_check_finds_dropbox_icloud_and_a_google_drive_drive_on_windows():
    fake = FakeComputer(
        environ={"APPDATA": r"C:\Users\ann\AppData\Roaming", "LOCALAPPDATA": r"C:\Local"},
        files={r"C:\Local\Dropbox\info.json": json.dumps({"personal": {"path": r"E:\Dropbox"}})},
        labels={"G:\\": "Google Drive"},
    )
    computer = fake.computer()
    dropbox = cloudsync.check(r"E:\Dropbox\Documents\Proteia", computer)
    assert dropbox is not None and dropbox.service == "Dropbox"
    assert fake.read == [
        r"C:\Users\ann\AppData\Roaming\Dropbox\info.json",  # missing: skipped
        r"C:\Local\Dropbox\info.json",
    ]
    icloud = cloudsync.check(r"C:\Users\ann\iCloudDrive\Proteia", computer)
    assert icloud is not None and icloud.service == "iCloud Drive"
    drive = cloudsync.check(r"G:\My Drive\Documents\Proteia", computer)
    assert drive == SyncedFolder("Google Drive", "G:\\", "the drive's volume label")
    assert cloudsync.check(r"C:\Users\ann\Documents\Proteia", computer) is None


def test_the_check_on_macos_finds_icloud_drive_and_the_cloud_storage_folders():
    fake = FakeComputer(
        system="macos",
        home="/Users/ann",
        folders={"/Users/ann/Library/CloudStorage": ["GoogleDrive-ann@example.org"]},
        files={"/Users/ann/.dropbox/info.json": json.dumps({"personal": {"path": "/Volumes/D"}})},
    )
    computer = fake.computer()
    icloud = cloudsync.check(
        "/Users/ann/Library/Mobile Documents/com~apple~CloudDocs/Documents/Proteia", computer
    )
    assert icloud is not None and icloud.service == "iCloud Drive"
    google = cloudsync.check(
        "/Users/ann/Library/CloudStorage/GoogleDrive-ann@example.org/My Drive/Proteia", computer
    )
    assert google is not None and google.service == "Google Drive"
    dropbox = cloudsync.check("/Volumes/D/Proteia", computer)
    assert dropbox is not None and dropbox.service == "Dropbox"
    assert cloudsync.check("/Users/ann/Documents/Proteia", computer) is None


def test_the_check_elsewhere_finds_dropbox_only():
    fake = FakeComputer(
        system="posix",
        home="/home/ann",
        environ={"OneDrive": "/home/ann/OneDrive"},  # OneDrive's variables: Windows only
        files={"/home/ann/.dropbox/info.json": json.dumps({"personal": {"path": "/home/ann/Db"}})},
    )
    computer = fake.computer()
    match = cloudsync.check("/home/ann/Db/Documents/Proteia", computer)
    assert match is not None and match.service == "Dropbox"
    assert cloudsync.check("/home/ann/OneDrive/Proteia", computer) is None


@pytest.mark.parametrize(
    "broken",
    [
        {"accounts": PermissionError("registry")},
        {"files": {r"C:\Local\Dropbox\info.json": UnicodeDecodeError("utf-8", b"\xff", 0, 1, "")}},
        {"files": {r"C:\Local\Dropbox\info.json": PermissionError("info.json")}},
        {"labels": {"D:\\": OSError("no drive")}},
        {"links": {r"C:\Users\ann\iCloudDrive": ValueError("embedded null character")}},
    ],
)
def test_a_setting_that_cannot_be_read_is_skipped(broken):
    fake = FakeComputer(environ={"OneDrive": ONEDRIVE, "LOCALAPPDATA": r"C:\Local"}, **broken)
    match = cloudsync.check(ROOT, fake.computer())
    assert match is not None and match.source == "the OneDrive environment variable"


def test_a_folder_on_macos_that_cannot_be_listed_is_skipped():
    fake = FakeComputer(
        system="macos",
        home="/Users/ann",
        folders={"/Users/ann/Library/CloudStorage": PermissionError("not allowed")},
    )
    assert cloudsync.check("/Users/ann/Documents/Proteia", fake.computer()) is None


def test_the_check_reads_this_computer(tmp_path, monkeypatch):
    # The real settings: a synced folder made in a scratch folder, named where
    # the service names it on this system.
    synced = tmp_path / "Synced µ"
    root = synced / "Documents" / "Proteia"
    root.mkdir(parents=True)
    if os.name == "nt":
        # Spelled otherwise: ASCII letters in capitals, and a trailing separator.
        spelled = "".join(c.upper() if c.isascii() else c for c in str(synced)) + "\\"
        monkeypatch.setenv("OneDrive", spelled)
        service = "OneDrive"
    else:
        monkeypatch.setenv("HOME", str(tmp_path))
        info = tmp_path / ".dropbox" / "info.json"
        info.parent.mkdir()
        info.write_text(json.dumps({"personal": {"path": str(synced)}}), encoding="utf-8")
        service = "Dropbox"
    match = cloudsync.check(root)
    assert match is not None and match.service == service
    assert cloudsync.check(tmp_path / "elsewhere" / "Proteia") is None


@pytest.mark.skipif(sys.platform != "win32", reason="the registry is Windows'")
def test_the_registry_is_read_without_failing():
    accounts = cloudsync.this_computer().onedrive_accounts()
    assert all(isinstance(name, str) and isinstance(path, str) for name, path in accounts)


# --- The dismissal, in the per-user state folder ---


def test_a_dismissed_notice_is_remembered_in_the_state_folder(tmp_path):
    state = tmp_path / "state µ"
    assert cloudsync.dismissed(state) == frozenset()  # no folder yet
    cloudsync.dismiss(state, cloudsync.CLOUD_SYNC)
    assert cloudsync.dismissed(state) == {cloudsync.CLOUD_SYNC}
    saved = json.loads((state / cloudsync.NOTICES_FILE).read_bytes())
    assert saved == {"dismissed": [cloudsync.CLOUD_SYNC]}
    cloudsync.dismiss(state, "another")
    cloudsync.dismiss(state, cloudsync.CLOUD_SYNC)
    assert cloudsync.dismissed(state) == {cloudsync.CLOUD_SYNC, "another"}


@pytest.mark.parametrize("text", ["not JSON", "[]", '{"dismissed": "cloud_sync"}', '{"x": 1}'])
def test_an_unreadable_notices_file_dismisses_nothing_and_is_replaced(tmp_path, text):
    (tmp_path / cloudsync.NOTICES_FILE).write_text(text, encoding="utf-8")
    assert cloudsync.dismissed(tmp_path) == frozenset()
    cloudsync.dismiss(tmp_path, cloudsync.CLOUD_SYNC)
    assert cloudsync.dismissed(tmp_path) == {cloudsync.CLOUD_SYNC}


# --- The routes ---


class Check:
    """Stands in for the check: answers ``answer`` (raises it, if an error)
    and records the roots it is asked about."""

    def __init__(self, answer: SyncedFolder | Exception | None) -> None:
        self.answer = answer
        self.roots: list[Path] = []

    def __call__(self, root: Path) -> SyncedFolder | None:
        self.roots.append(root)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


SYNCED = SyncedFolder("OneDrive", ONEDRIVE, "the OneDrive environment variable")


@pytest.fixture
def serve(tmp_path) -> Iterator[Callable[[Check], Client]]:
    """Starts Proteia with a check, on one state folder in ``tmp_path``: each
    start stops the one before, as a user quits and starts Proteia again."""
    running: list[tuple[launch.Instance, threading.Thread]] = []

    def stop() -> None:
        while running:
            instance, thread = running.pop()
            instance.stop()
            thread.join(10)

    def start(check: Check) -> Client:
        stop()
        root = tmp_path / "projects"
        workspace = api.Workspace(
            root, reveal=lambda folder: None, clock=FakeClock(), sync_check=check
        )
        state = tmp_path / "state µ"
        instance = launch.start(folder=state, opener=lambda url: True, workspace=workspace)
        assert instance is not None
        thread = threading.Thread(target=instance.serve, daemon=True)
        thread.start()
        running.append((instance, thread))
        deadline = time.monotonic() + 10
        while not instance.server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        return Client(instance.port, instance.token, root, [], workspace)

    yield start
    stop()


def test_the_notice_is_shown_until_dismissed_for_good(serve, tmp_path):
    check = Check(SYNCED)
    client = serve(check)
    assert client.ok("GET", "/api/notices") == {"cloud_sync": {"service": "OneDrive"}}
    assert client.ok("GET", "/api/notices") == {"cloud_sync": {"service": "OneDrive"}}
    assert check.roots == [client.root]  # checked once
    assert client.call("POST", "/api/notices/cloud_sync/dismiss")[0] == 204
    assert client.ok("GET", "/api/notices") == {"cloud_sync": None}
    assert (tmp_path / "state µ" / cloudsync.NOTICES_FILE).is_file()
    assert not client.root.exists()  # nothing in the projects folder

    again = serve(Check(SYNCED))  # the next start, with the same state folder
    assert again.ok("GET", "/api/notices") == {"cloud_sync": None}


def test_no_notice_when_the_projects_folder_is_not_synced(serve):
    client = serve(Check(None))
    assert client.ok("GET", "/api/notices") == {"cloud_sync": None}


def test_the_notice_names_the_service_and_no_path(serve):
    client = serve(Check(SYNCED))
    status, answer = client.call("GET", "/api/notices")
    assert status == 200
    text = json.dumps(answer)
    assert "Sync" not in text and "\\" not in text and "/" not in text


def test_the_notice_routes_need_the_token(serve):
    client = serve(Check(SYNCED))
    wrong = {"Authorization": "Bearer " + "x" * 43}
    assert client.call("GET", "/api/notices", headers=wrong)[0] == 401
    assert client.call("POST", "/api/notices/cloud_sync/dismiss", headers=wrong)[0] == 401
    assert client.ok("GET", "/api/notices")["cloud_sync"] is not None  # not dismissed


def test_a_check_that_fails_shows_no_notice_and_is_logged(serve, caplog):
    client = serve(Check(RuntimeError("a bug in the check")))
    with caplog.at_level(logging.INFO, logger=api.__name__):
        assert client.ok("GET", "/api/notices") == {"cloud_sync": None}
        client.ok("GET", "/api/projects")  # the app goes on
    logged = [r for r in caplog.records if r.name == api.__name__]
    assert [r.getMessage() for r in logged] == [
        "could not check whether the projects folder is synced to the cloud"
    ]
    assert logged[0].levelno == logging.WARNING and logged[0].exc_info


def test_without_a_state_folder_the_notice_is_never_offered(serve):
    # Its dismissal could not be remembered: it would be shown at every start.
    client = serve(Check(SYNCED))
    client.workspace.state = None
    assert client.ok("GET", "/api/notices") == {"cloud_sync": None}
    assert client.refused("POST", "/api/notices/cloud_sync/dismiss")[:2] == (
        409,
        "no_state_folder",
    )


def test_a_dismissal_that_cannot_be_written_names_no_path(serve, monkeypatch, tmp_path):
    # The page shows the answer's message: a folder that cannot be written is
    # said without the per-user path.
    client = serve(Check(SYNCED))
    where = str(tmp_path / "state µ" / ".notices.json.tmp")

    def refuse(folder, notice):
        raise PermissionError(13, "Permission denied", where)

    monkeypatch.setattr(cloudsync, "dismiss", refuse)
    status, answer = client.call("POST", "/api/notices/cloud_sync/dismiss")
    assert status == 500
    assert answer["message"] == "the notice could not be dismissed: Permission denied"
    assert str(tmp_path) not in json.dumps(answer)
    assert client.ok("GET", "/api/notices")["cloud_sync"] is not None  # not dismissed


def test_the_check_and_the_dismissal_are_logged_by_label(serve, caplog):
    client = serve(Check(SYNCED))
    with caplog.at_level(logging.INFO, logger=api.__name__):
        client.ok("GET", "/api/notices")
        client.call("POST", "/api/notices/cloud_sync/dismiss")
    logged = [r.getMessage() for r in caplog.records if r.name == api.__name__]
    assert logged == [
        "the projects folder is in a folder OneDrive uploads (found from the OneDrive"
        " environment variable)",
        "the notice that the projects folder is synced to the cloud was dismissed for good",
    ]
    assert not any("Sync" in line or "\\" in line for line in logged)  # no path


def test_a_projects_folder_that_is_not_synced_is_logged(serve, caplog):
    client = serve(Check(None))
    with caplog.at_level(logging.INFO, logger=api.__name__):
        client.ok("GET", "/api/notices")
    logged = [r.getMessage() for r in caplog.records if r.name == api.__name__]
    assert logged == ["the projects folder is in no folder a sync service Proteia recognises"]


# --- The page ---


def test_the_page_has_a_notice_dialog_that_names_the_service_and_its_settings():
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    start = html.index('<dialog id="sync-notice"')
    dialog = html[start : html.index("</dialog>", start)]
    assert 'aria-labelledby="sync-notice-title"' in dialog
    assert dialog.count('class="sync-service"') >= 2  # filled with the service's name
    words = " ".join(dialog.split())
    for said in (
        # The service uploads, named: never "it", which could be read as Proteia.
        '<span class="sync-service"></span> uploads every project',
        "the images you import",
        "Proteia itself sends nothing",
        "unpublished",
        "own settings",
        "Proteia does not change",
        "choose another projects folder",
        "not shown again",
    ):
        assert said in words, said
    parser = _Tags()
    parser.feed(html)
    assert parser.by_id["sync-notice-ok"]["type"] == "button"


def test_the_page_shows_the_notice_before_any_other_dialog_and_dismisses_it_once():
    script = _code("app.js")
    _, notice = _function(script, "async function showSyncNotice(")
    assert '"/api/notices"' in notice and "anyProject: true" in notice
    assert "showModal()" in notice
    assert ".textContent = " in notice  # the service's name, as text
    assert "root" not in notice  # no path on this notice
    # However it is closed (OK, Escape), it is dismissed, once, and then it
    # gives that it was shown; with none to show, at once that it was not.
    assert re.search(r'addEventListener\(\s*"close"', notice) and "once: true" in notice
    assert '"/api/notices/cloud_sync/dismiss"' in notice
    assert notice.index("showModal()") < notice.index("await closed") < notice.index("return true;")
    assert notice.index("return false;") < notice.index("showModal()")
    # At start only, over the project open if one is, and closed before the
    # page takes its first listing, which may show the Import or the Projects
    # dialog: two modal dialogs opened without a click close together on one
    # Escape. No check takes a listing before (checkOpening waits for
    # started); once it is closed the page checks again, since what it listed
    # before may have changed meanwhile.
    assert len(re.findall(r"(?<!function )showSyncNotice\(", script)) == 1
    _, start = _function(script, "async function start(")
    shown = start.index("await showSyncNotice()")
    assert start.index('call("GET", "/api/project")') < shown < start.index("started = true;")
    assert shown < start.index("await checkOpening()")
    assert shown < start.index("const found = takeListing(workspace)")
    assert "!started" in _function(script, "function checkOpening(")[1]
