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
    session = projects.open_named(root, "μ BLOT", clock=FakeClock())
    assert session.folder.name == "µ Blot"
    with pytest.raises(projects.ProjectNotFoundError):
        projects.open_named(root, "Other", clock=FakeClock())


@pytest.mark.skipif(os.name != "nt", reason="the Windows Documents known folder")
def test_the_projects_root_is_in_the_documents_folder():
    root = projects.projects_root()
    assert root.name == "Proteia" and root.parent.is_dir()
