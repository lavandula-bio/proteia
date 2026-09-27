# SPDX-License-Identifier: Apache-2.0
"""The ``proteia`` command line (#57, N3): its grammar, and the checks of each
PATH in the launching process (:mod:`proteia.web.cli`). Pure: files in
``tmp_path``, no server and no launch; the launches that hand the files over
are tested in test_web.py."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

import proteia
from proteia.web import cli, launch
from proteia.web.cli import CommandLine, RefusedPath
from proteia.web.handoff import Refusal

WINDOWS = os.name == "nt"


def image(folder: Path, name: str, data: bytes = b"pixels") -> Path:
    """A file that passes for an image: the launcher never decodes one."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(data)
    return path


def refused(argv: list[str]) -> list[tuple[str, str]]:
    """Each refused PATH's name and code."""
    return [(path.name, path.code) for path in cli.parse(argv).refused]


# --- The grammar ---


def test_no_arguments_is_a_plain_launch():
    command_line = cli.parse([])
    assert command_line == CommandLine() and not command_line.paths


def test_images_are_taken_in_argument_order(tmp_path):
    paths = [image(tmp_path, name) for name in ("b.tif", "a.png", "c.jpeg", "d.tiff", "e.jpg")]
    command_line = cli.parse([str(path) for path in paths])
    assert [file.name for file in command_line.files] == [path.name for path in paths]
    assert command_line.refused == () and command_line.paths
    first = command_line.files[0]
    assert (first.argument, first.path, first.size) == (str(paths[0]), paths[0].resolve(), 6)
    with first.open() as stream:
        assert stream.read() == b"pixels"


def test_the_same_file_given_twice_is_taken_once(tmp_path, monkeypatch):
    a, b = image(tmp_path, "a.tif"), image(tmp_path, "b.tif")
    monkeypatch.chdir(tmp_path)
    argv = ["a.tif", "b.tif", str(a), os.path.join(".", "a.tif"), "b.tif"]
    if WINDOWS:
        argv.append(str(a).upper())  # Windows names ignore case
    command_line = cli.parse(argv)
    assert [file.path for file in command_line.files] == [a.resolve(), b.resolve()]
    assert command_line.refused == ()


def test_a_relative_path_resolves_in_the_launching_folder(tmp_path, monkeypatch):
    here = tmp_path / "sub α"
    wanted = image(here, "β-actin blot.tif")
    image(tmp_path, "β-actin blot.tif", b"another")  # the same name, one folder up
    monkeypatch.chdir(here)
    (file,) = cli.parse(["β-actin blot.tif"]).files
    assert file.path == wanted.resolve() and file.path.is_absolute()
    (up,) = cli.parse([os.path.join("..", "β-actin blot.tif")]).files
    assert up.path == (tmp_path / "β-actin blot.tif").resolve() and up.size == 7


def test_a_name_is_kept_as_it_is_and_suffixes_in_any_case_are_images(tmp_path):
    names = ["β-actin 10 µM.TIF", "Marker α.Png", "x.JPEG"]
    command_line = cli.parse([str(image(tmp_path, name)) for name in names])
    assert [file.name for file in command_line.files] == names


def test_an_option_is_refused_before_anything_is_checked(capsys):
    with pytest.raises(SystemExit) as exit:
        cli.parse(["--bogus", "a.tif"])
    assert exit.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: proteia") and "unrecognized arguments: --bogus" in err


def test_a_path_that_begins_with_a_dash_goes_after_the_double_dash(tmp_path, monkeypatch, capsys):
    image(tmp_path, "-x.tif")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit:
        cli.parse(["-x.tif"])
    assert exit.value.code == 2
    capsys.readouterr()
    (file,) = cli.parse(["--", "-x.tif"]).files
    assert file.name == "-x.tif"


def test_help_and_version_are_printed(capsys):
    with pytest.raises(SystemExit) as exit:
        cli.parse(["--help"])
    assert exit.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: proteia [-h] [--version] [PATH ...]")
    words = " ".join(out.split())  # as wrapped to the console's width
    assert ".tif, .tiff, .png, .jpg, .jpeg" in words and "dark or light" in words
    assert "Exit status: 0 opened; 1 not opened" in words and "proteia --self-test" in words
    with pytest.raises(SystemExit) as exit:
        cli.parse(["--version"])
    assert exit.value.code == 0
    assert capsys.readouterr().out == f"Proteia {proteia.__version__}\n"


@pytest.mark.parametrize(
    ("argv", "status", "stream", "text"),
    [
        (["--bogus"], 2, "err", "unrecognized arguments: --bogus"),
        (["a.tif", "--self-test"], 2, "err", "unrecognized arguments: --self-test"),
        (["--help"], 0, "out", "Exit status:"),
        (["--version"], 0, "out", proteia.__version__),
    ],
)
def test_the_command_starts_nothing_for_wrong_arguments_help_or_version(
    monkeypatch, capsys, argv, status, stream, text
):
    monkeypatch.setattr(launch, "start", lambda **kwargs: pytest.fail("nothing may start"))
    monkeypatch.setattr(launch, "state_dir", lambda: pytest.fail("no session log"))
    assert launch.main(argv) == status
    assert text in getattr(capsys.readouterr(), stream)


# --- Each PATH checked ---


def test_every_path_is_checked_on_its_own_and_the_others_proceed(tmp_path):
    folder = tmp_path / "blot"
    folder.mkdir()
    good = image(tmp_path, "good.tif")
    argv = [
        str(tmp_path / "missing.tif"),
        str(image(tmp_path, "photo.bmp")),
        str(good),
        str(folder),
        str(image(folder, "project.json", b"{}")),
        str(image(tmp_path, "empty.png", b"")),
    ]
    command_line = cli.parse(argv)
    assert [file.name for file in command_line.files] == ["good.tif"]
    assert [(path.name, path.code) for path in command_line.refused] == [
        ("missing.tif", "missing"),
        ("photo.bmp", "unsupported_type"),
        ("blot", "folder"),
        ("project.json", "project_file"),
        ("empty.png", "empty"),
    ]
    # The console shows each in full; the page gets its name, code and message only.
    missing = command_line.refused[0]
    assert missing.argument == argv[0] and missing.shown == str(
        (tmp_path / "missing.tif").resolve()
    )
    assert missing.message == "no such file or folder"
    assert missing.refusal() == Refusal("missing.tif", "missing", "no such file or folder")
    assert command_line.refused[2].message == (
        "a folder; open projects from Proteia's Projects dialog"
    )
    assert command_line.refused[3].message == (
        "a Proteia project file; open projects from Proteia's Projects dialog"
    )


@pytest.mark.parametrize("name", ["photo.bmp", "slide.scn", "notes.json", "blot", "blot.tif.txt"])
def test_a_file_of_another_type_is_refused(tmp_path, name):
    (entry,) = cli.parse([str(image(tmp_path, name))]).refused
    assert (entry.name, entry.code) == (name, "unsupported_type")
    assert entry.message == "not an image type Proteia imports (.tif, .tiff, .png, .jpg, .jpeg)"
    assert refused([str(image(tmp_path, "project.json.tif"))]) == []  # a .tif all the same


@pytest.mark.parametrize("name", ["project.json", "PROJECT.JSON", "Project.Json"])
def test_a_project_file_is_refused_in_any_case(tmp_path, name):
    assert refused([str(image(tmp_path / "Blot", name, b"{}"))]) == [(name, "project_file")]


def test_a_folder_is_refused_whatever_its_name(tmp_path):
    (tmp_path / "scan.tif").mkdir()
    assert refused([str(tmp_path / "scan.tif")]) == [("scan.tif", "folder")]


def test_a_path_through_a_file_is_missing(tmp_path):
    parent = image(tmp_path, "a.tif")
    assert refused([str(parent / "b.tif")]) == [("b.tif", "missing")]


def test_a_file_over_the_size_limit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "MAX_UPLOAD_BYTES", 5)
    (entry,) = cli.parse([str(image(tmp_path, "big.tif", b"pixels"))]).refused
    assert (entry.code, entry.message) == ("too_large", "larger than 5 bytes")
    monkeypatch.setattr(cli, "MAX_UPLOAD_BYTES", 6)
    assert len(cli.parse([str(tmp_path / "big.tif")]).files) == 1


def test_the_size_limit_is_the_upload_limit():
    assert cli.MAX_UPLOAD_BYTES == 512 * 1024**2
    assert cli._size(cli.MAX_UPLOAD_BYTES) == "512 MiB"


def test_a_name_longer_than_a_project_can_store_is_refused(tmp_path, monkeypatch):
    # No file system here holds a 256-character name: the checks are shown one.
    real = image(tmp_path, "real.tif")
    monkeypatch.setattr(cli, "_stat", lambda location: os.stat(real))
    monkeypatch.setattr(cli, "_readable", lambda location: None)
    name = "x" * 252 + ".tif"
    (entry,) = cli.parse([str(tmp_path / name)]).refused
    assert (entry.name, entry.code) == (name, "bad_name")
    assert entry.message == "its name cannot be stored; rename the file"
    assert len(cli.parse([str(tmp_path / ("x" * 251 + ".tif"))]).files) == 1


def test_a_name_with_a_lone_surrogate_is_refused(tmp_path):
    # Windows names can hold one; on POSIX an undecodable byte arrives as one.
    name = "\udc80blot.tif" if WINDOWS else "\udcffblot.tif"
    try:
        path = image(tmp_path, name)
    except (OSError, UnicodeEncodeError):
        pytest.skip("this file system takes no such name")
    (entry,) = cli.parse([str(path)]).refused
    assert (entry.name, entry.code) == (name, "bad_name")
    assert entry.refusal().name == "\ufffdblot.tif"  # what a JSON body can carry


def test_an_empty_argument_is_refused():
    (entry,) = cli.parse([""]).refused
    assert (entry.name, entry.code, entry.message) == ('""', "bad_argument", "an empty argument")


@pytest.mark.skipif(WINDOWS or os.geteuid() == 0, reason="POSIX permissions, not as root")
def test_a_file_that_cannot_be_read_is_refused(tmp_path):
    path = image(tmp_path, "locked.tif")
    path.chmod(0)
    try:
        (entry,) = cli.parse([str(path)]).refused
    finally:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert entry.code == "unreadable" and entry.message.startswith("cannot be read: ")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes in the file system")
def test_a_file_that_is_not_a_regular_file_is_refused_unopened(tmp_path):
    os.mkfifo(tmp_path / "pipe.tif")  # opening it would wait for a writer
    (entry,) = cli.parse([str(tmp_path / "pipe.tif")]).refused
    assert (entry.code, entry.message) == ("unreadable", "cannot be read: not a regular file")


@pytest.mark.skipif(not WINDOWS, reason="Windows argument quoting")
def test_an_argument_with_a_quote_mark_is_refused_on_windows(tmp_path):
    # "C:\x\blot\" reaches argv as C:\x\blot" : its closing quote is lost.
    (entry,) = cli.parse([str(tmp_path / "blot") + '"']).refused
    assert (entry.name, entry.code) == ('blot"', "bad_argument")
    assert "quote mark" in entry.message


@pytest.mark.skipif(not WINDOWS, reason="Windows file names")
def test_a_name_windows_refuses_is_unreadable_not_missing(tmp_path):
    (entry,) = cli.parse([str(tmp_path / "a?b.tif")]).refused
    assert (entry.name, entry.code) == ("a?b.tif", "unreadable")
    assert entry.message.startswith("cannot be read: ") and len(entry.message) > 16


@pytest.mark.skipif(not WINDOWS, reason="Windows path length limit")
def test_a_path_longer_than_windows_allows_is_read_through_its_long_form(tmp_path):
    folder = tmp_path / ("d" * 120) / ("e" * 120)
    os.makedirs("\\\\?\\" + str(folder))
    with open("\\\\?\\" + str(folder / "blot α.tif"), "wb") as out:
        out.write(b"pixels")
    path = folder / "blot α.tif"
    # Too long to reach plainly unless the system has long paths enabled (as
    # CI runners do): read through the long form either way.
    assert len(str(path)) > 300
    (file,) = cli.parse([str(path)]).files
    assert (file.name, file.size, file.path) == ("blot α.tif", 6, path)
    assert str(file.location).startswith("\\\\?\\")
    with file.open() as stream:
        assert stream.read() == b"pixels"


@pytest.mark.skipif(not WINDOWS, reason="Windows path forms")
def test_the_long_form_of_a_windows_path():
    assert str(cli.long_form(Path("C:\\x\\a.tif"))) == "\\\\?\\C:\\x\\a.tif"
    assert str(cli.long_form(Path("\\\\server\\share\\x\\a.tif"))) == (
        "\\\\?\\UNC\\server\\share\\x\\a.tif"
    )
    assert str(cli.long_form(Path("\\\\?\\C:\\x\\a.tif"))) == "\\\\?\\C:\\x\\a.tif"


@pytest.mark.skipif(WINDOWS, reason="POSIX paths")
def test_a_posix_path_is_used_as_it_is():
    assert cli.long_form(Path("/x/a.tif")) == Path("/x/a.tif")


def test_a_refused_entry_carries_no_lone_surrogate():
    path = RefusedPath("a", "a", "\udc80.tif", "bad_name", "why \ud800")
    assert path.refusal() == Refusal("\ufffd.tif", "bad_name", "why \ufffd")


def test_the_console_command_reads_its_own_arguments(monkeypatch):
    # Should main not read them, it must not launch for real either.
    monkeypatch.setattr(launch, "start", lambda **kwargs: pytest.fail("nothing may start"))
    monkeypatch.setattr(launch, "state_dir", lambda: pytest.fail("no session log"))
    seen = []
    monkeypatch.setattr(sys, "argv", ["proteia", "--version"])
    monkeypatch.setattr(cli, "parse", lambda argv: seen.append(argv) or sys.exit(0))
    assert launch.main() == 0
    assert seen == [["--version"]]
