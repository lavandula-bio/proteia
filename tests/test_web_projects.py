# SPDX-License-Identifier: Apache-2.0
"""The app-managed projects root (#52 decision): names, listing, create and open."""

from __future__ import annotations

import os

import pytest

from conftest import FakeClock
from proteia.core import storage
from proteia.web import projects


@pytest.mark.parametrize(
    ("name", "stored"),
    [
        ("Blot 1", "Blot 1"),
        ("  Blot\t 1 ", "Blot 1"),
        ("β-actin µM", "β-actin µM"),
        ("實驗 2026-09", "實驗 2026-09"),  # CJK text
        ("v0.1 run", "v0.1 run"),
        ("CONSOLE", "CONSOLE"),  # only the exact device names are reserved
    ],
)
def test_a_project_name_is_stored_cleaned(name, stored):
    assert projects.project_name(name) == stored


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        "a/b",
        "a\\b",
        "a:b",
        "a*b",
        "a?b",
        'a"b',
        "a<b",
        "a|b",
        ".",
        "..",
        "ends.",
        "CON",
        "COM0",
        "COM¹",
        "lpt³.txt",
        "CONIN$",
        "conout$",
        "con",
        "Aux.txt",
        "COM1",
        "lpt9.log",
        "x" * 101,
        "a\u0007b",
        7,
        None,
    ],
)
def test_a_name_that_cannot_be_a_folder_is_refused(name):
    with pytest.raises(projects.ProjectNameError):
        projects.project_name(name)


def test_projects_are_listed_newest_first_and_strays_are_left_out(tmp_path):
    root = tmp_path / "root"
    assert projects.list_projects(root) == []  # no root yet
    old = projects.create_project(root, "Old", clock=FakeClock())
    new = projects.create_project(root, "New", clock=FakeClock())
    os.utime(old.folder / storage.PROJECT_FILE, (1_000_000_000, 1_000_000_000))
    (root / "not a project").mkdir()
    (root / "loose file.txt").write_text("x", encoding="utf-8")
    listed = projects.list_projects(root)
    assert [entry.name for entry in listed] == ["New", "Old"]
    assert listed[1].modified == "2001-09-09T01:46:40.000Z"
    assert new.folder.parent == root


def test_names_are_unique_ignoring_case_and_look_alikes(tmp_path):
    root = tmp_path / "root"
    projects.create_project(root, "µ Blot", clock=FakeClock())  # micro sign
    for twin in ("µ BLOT", "μ blot"):  # upper case; Greek mu
        with pytest.raises(projects.ProjectExistsError):
            projects.create_project(root, twin, clock=FakeClock())
    for spelling in ("μ BLOT", " µ blot "):
        folder = projects.project_folder(root, spelling)
        assert (folder.parent, folder.name) == (root, "µ Blot")  # the folder's own spelling
    with pytest.raises(projects.ProjectNotFoundError):
        projects.project_folder(root, "Other")


@pytest.mark.skipif(os.name != "nt", reason="the Windows Documents known folder")
def test_the_projects_root_is_in_the_documents_folder():
    root = projects.projects_root()
    assert root.name == "Proteia" and root.parent.is_dir()


def _case_sensitive(folder) -> bool:
    (folder / "probe").mkdir()
    return not (folder / "PROBE").exists()


def test_the_exact_name_wins_over_a_look_alike(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    if not _case_sensitive(tmp_path):
        pytest.skip("two names differing only in case need a case-sensitive file system")
    for name in ("blot", "Blot"):
        folder = root / name
        folder.mkdir()
        (folder / storage.PROJECT_FILE).write_bytes(b"")
    upper = projects._existing(root, "Blot")
    lower = projects._existing(root, "blot")
    assert (upper.name, lower.name) == ("Blot", "blot")


def test_a_free_name_is_the_name_or_the_first_free_number(tmp_path):
    root = tmp_path / "root"
    assert projects.free_name(root, "  Sample  blot ") == "Sample blot"  # no root yet; cleaned
    projects.create_project(root, "Sample blot", clock=FakeClock())
    (root / "SAMPLE BLOT (2)").mkdir()  # any folder takes its name, ignoring case
    (root / "sample blot （3）").write_bytes(b"")  # and look-alikes: fullwidth parentheses
    assert projects.free_name(root, "Sample blot") == "Sample blot (4)"
    with pytest.raises(projects.ProjectNameError):
        projects.free_name(root, "a/b")


def test_no_free_name_is_refused_as_existing(tmp_path, monkeypatch):
    root = tmp_path / "root"
    monkeypatch.setattr(projects, "MAX_NUMBERED", 2)
    for name in ("Blot", "Blot (2)"):
        projects.create_project(root, name, clock=FakeClock())
    with pytest.raises(projects.ProjectExistsError):
        projects.free_name(root, "Blot")


# --- A project named after a file (#57, N3) ---


@pytest.mark.parametrize(
    ("file", "name"),
    [
        ("β-actin 10 µM.tif", "β-actin 10 µM"),
        ("blot..tif", "blot"),
        ("a:b.png", "a b"),
        ('a<b>"c|d?e*f\\g.jpg', "a b c d e f g"),
        ("  lots   of  space .TIF", "lots of space"),
        ("實驗 2026-09.tiff", "實驗 2026-09"),
        ("CON.tif", projects.FALLBACK_NAME),
        ("lpt1.tif", projects.FALLBACK_NAME),
        ("....tif", projects.FALLBACK_NAME),
        (" .tif", projects.FALLBACK_NAME),
        ("a\u0007b.tif", projects.FALLBACK_NAME),  # a control character
        ("half\ud800.tif", projects.FALLBACK_NAME),
        ("x" * 120 + ".tif", "x" * 93),
        ("x" * 92 + " .....y.tif", "x" * 92),  # cut to 93, then its end trimmed
    ],
)
def test_a_project_is_named_after_a_file(file, name):
    assert projects.name_from_file(file) == name
    assert projects.project_name(name) == name


def test_a_name_from_a_file_can_always_be_numbered(tmp_path):
    root = tmp_path / "root"
    name = projects.name_from_file("x" * 120 + ".tif")
    projects.create_project(root, name, clock=FakeClock())
    numbered = projects.free_name(root, name)
    assert numbered == "x" * 93 + " (2)" and len(numbered) == 97
    assert projects.MAX_FILE_NAME + len(f" ({projects.MAX_NUMBERED})") == projects.MAX_NAME


def test_a_free_name_can_be_found_from_a_listing_read_once(tmp_path, monkeypatch):
    root = tmp_path / "root"
    for name in ("Blot", "blot (2)", "ＢＬＯＴ (3)"):  # any case; full-width look-alikes
        projects.create_project(root, name, clock=FakeClock())
    listed = []
    iterdir = type(root).iterdir

    def counting(self):
        listed.append(self)
        return iterdir(self)

    monkeypatch.setattr(type(root), "iterdir", counting)
    assert projects.free_name(root, "BLOT") == "BLOT (4)"
    assert listed == [root]  # once, not once per number tried
    existing = projects.names_in(root)
    listed.clear()
    assert projects.free_name(root, "blot", existing=existing) == "blot (4)"
    assert projects.free_name(tmp_path / "none", "blot", existing=["BLOT"]) == "blot (2)"
    assert listed == []
    assert projects.names_in(tmp_path / "none") == []


def test_a_project_set_up_as_it_is_created_is_saved_or_leaves_no_folder(tmp_path):
    root = tmp_path / "root"
    seen = []
    session = projects.create_set_up(root, "Set up µ", seen.append, clock=FakeClock())
    assert seen == [session] and (root / "Set up µ" / storage.PROJECT_FILE).is_file()

    def fail(session):
        (session.folder / "notes.txt").write_text("written", encoding="utf-8")
        raise OSError(28, "No space left on device")

    with pytest.raises(OSError, match="No space left"):
        projects.create_set_up(root, "Failed", fail, clock=FakeClock())
    assert sorted(p.name for p in root.iterdir()) == ["Set up µ"]
    with pytest.raises(projects.ProjectExistsError):
        projects.create_set_up(root, "set up µ", seen.append, clock=FakeClock())
    assert len(seen) == 1  # never set up


def test_a_folder_that_cannot_be_removed_after_a_failed_set_up_is_named(tmp_path, monkeypatch):
    root = tmp_path / "root"

    def fail(session):
        raise ValueError("refused")

    monkeypatch.setattr(projects.shutil, "rmtree", lambda path, ignore_errors=False: None)
    with pytest.raises(projects.FolderLeftError) as left:
        projects.create_set_up(root, "Left", fail, clock=FakeClock(), what="the imported images")
    assert str(left.value) == (
        "the imported images could not be set up (refused), and its unfinished folder 'Left'"
        " could not be removed: delete it from the projects folder"
    )
    assert isinstance(left.value, OSError)
