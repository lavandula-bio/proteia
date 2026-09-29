# SPDX-License-Identifier: Apache-2.0
"""The ``proteia`` command line (#57, N3): ``proteia [-h] [--version] [--] [PATH ...]``.

Each PATH names an image file to import into a new project, named after the
first one. The command line never imports anything itself: polarity has no
default, so the files are handed to the running Proteia
(:mod:`proteia.web.launch`, :mod:`proteia.web.handoff`), and its page asks how to
import them. ``--`` ends the options, so a file named ``-x.tif`` can be given.

:func:`parse` checks every PATH on its own, in the launching process, before
anything starts: it reads the file system and nothing else, and needs no running
Proteia. A PATH it cannot take is refused with a code and a reason
(:class:`RefusedPath`), and the others proceed, in argument order; the same file
given twice is taken once. The checks, in order (the first that applies refuses
it):

* ``bad_argument``: an empty argument; on Windows, one that holds ``"``, which
  no file name can (a quoted path that ends in ``\\`` loses its closing quote,
  and may swallow the next argument);
* ``unreadable``: the path cannot be resolved;
* ``missing``: nothing is there;
* ``unreadable``: the operating system refuses to look (an invalid name, a
  folder that cannot be listed), with its reason, in its language;
* ``folder``, ``project_file``: a folder, or a Proteia ``project.json``: projects
  are opened from Proteia's Projects dialog;
* ``unreadable``: not a regular file (a device or a pipe);
* ``unsupported_type``: not one of :data:`~proteia.core.model.IMAGE_SUFFIXES`
  (in any case);
* ``empty``; ``too_large``: over :data:`~proteia.web.api.MAX_UPLOAD_BYTES`;
* ``bad_name``: a name no project can store as the image's original name
  (longer than 255 characters, or holding a lone surrogate);
* ``unreadable``: it cannot be opened for reading.

Images are not decoded here: a file that is not the image its name says is found
when it is imported, and the page says so.

Paths: a relative path is resolved in the launching process's working folder
(the server's is unrelated), with :meth:`~pathlib.Path.resolve`, which also
expands Windows 8.3 short names and follows links (so a link's target's name is
the one stored). On Windows every file-system call on an argument goes through
the ``\\\\?\\`` form of the resolved path (``\\\\?\\UNC\\…`` for a network share), so
a path longer than 260 characters works whether or not long paths are enabled;
the console and the page show the plain path and name.
"""

from __future__ import annotations

import argparse
import os
import stat
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import BinaryIO, Final

import proteia
from proteia.core.model import IMAGE_SUFFIXES
from proteia.core.session import OperationError
from proteia.web.api import MAX_UPLOAD_BYTES
from proteia.web.handoff import Refusal, check_name

PROG: Final = "proteia"
PROJECT_FILE: Final = "project.json"
DESCRIPTION: Final = (
    "Start Proteia, the Western blot quantification app, in your web browser. If\n"
    "Proteia is already running, open it instead."
)
PATH_HELP: Final = (
    f"Image files ({', '.join(IMAGE_SUFFIXES)}) to import into a new project named after"
    " the first one. The browser asks how to import them (kind, membrane, and whether"
    " bands are dark or light) before anything is imported."
)
EPILOG: Final = (
    "A path that begins with - goes after --, as in: proteia -- -blot.tif\n"
    "proteia --self-test checks the installation instead (proteia --self-test --help).\n"
    "\n"
    "Exit status: 0 opened; 1 not opened, or the files were not handed to Proteia;\n"
    "2 wrong arguments (nothing was done); 3 opened, but some paths were not (they\n"
    "are listed here and in the browser)."
)
_PROJECTS_DIALOG: Final = "open projects from Proteia's Projects dialog"
_LONG_PREFIX: Final = "\\\\?\\"  # \\?\ : the Windows path form with no length limit
_DEVICE_PREFIX: Final = "\\\\.\\"


@dataclass(frozen=True)
class RefusedPath:
    """A PATH not taken, and why. ``shown`` is the path as the console shows it
    (the resolved one if it could be resolved); ``name`` its last part, the only
    part the page sees."""

    argument: str
    shown: str
    name: str
    code: str
    message: str

    def refusal(self) -> Refusal:
        """The entry a launch hands to Proteia: the name, code and message, each
        lone surrogate (which no JSON parser need take) made U+FFFD. Proteia
        bounds it further (:meth:`~proteia.web.handoff.Refusal.bounded`)."""
        return Refusal(_no_surrogates(self.name), self.code, _no_surrogates(self.message))


@dataclass(frozen=True)
class ImageFile:
    """A PATH taken: an image file to hand to Proteia. ``path`` is the resolved
    path, as the console shows it; ``name`` its last part, which the project
    stores as the image's original name; ``location`` the path every file-system
    call uses (on Windows, ``path`` in its ``\\\\?\\`` form); ``size`` as it was
    checked."""

    argument: str
    path: Path
    name: str
    size: int
    location: Path

    def open(self) -> BinaryIO:
        return open(self.location, "rb")

    def refused(self, code: str, message: str) -> RefusedPath:
        """This file as refused after all (by the running Proteia, say)."""
        return RefusedPath(self.argument, str(self.path), self.name, code, message)


@dataclass(frozen=True)
class CommandLine:
    """The PATHs of a command line: the files taken, in argument order, and those
    refused."""

    files: tuple[ImageFile, ...] = ()
    refused: tuple[RefusedPath, ...] = ()

    @property
    def paths(self) -> bool:
        """Whether any PATH was given."""
        return bool(self.files or self.refused)


class _Parser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__(
            prog=PROG,
            description=DESCRIPTION,
            epilog=EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            allow_abbrev=False,
        )
        self.add_argument(
            "--version",
            action="version",
            version=f"Proteia {proteia.__version__}",
            help="show Proteia's version and exit",
        )
        self.add_argument("paths", nargs="*", metavar="PATH", help=PATH_HELP)


def parser() -> argparse.ArgumentParser:
    """The argument parser, whose ``--help`` describes the command line."""
    return _Parser()


def parse(argv: Sequence[str]) -> CommandLine:
    """The command line ``argv`` (without the program name), each PATH checked
    (see the module docstring). ``SystemExit`` as :mod:`argparse` raises it: 2
    for a usage error, with its message on standard error; 0 after ``--help`` or
    ``--version``, printed on standard output."""
    args = parser().parse_args(list(argv))
    files: list[ImageFile] = []
    refused: list[RefusedPath] = []
    seen: set[Hashable] = set()
    for argument in args.paths:
        checked = check(argument)
        if isinstance(checked, RefusedPath):
            refused.append(checked)
            continue
        file, identity = checked
        if identity not in seen:
            seen.add(identity)
            files.append(file)
    return CommandLine(tuple(files), tuple(refused))


def check(argument: str) -> RefusedPath | tuple[ImageFile, Hashable]:
    """One PATH checked: refused, or taken with its identity (the same file,
    however named, has the same)."""
    fallback = PurePath(argument).name or argument or '""'

    def refuse(code: str, message: str, resolved: Path | None = None) -> RefusedPath:
        name = resolved.name if resolved is not None and resolved.name else fallback
        shown = argument if resolved is None else str(resolved)
        return RefusedPath(argument, shown, name, code, message)

    if not argument:
        return refuse("bad_argument", "an empty argument")
    if os.name == "nt" and '"' in argument:
        return refuse(
            "bad_argument",
            "contains a quote mark; a path that ends in \\ before its closing quote loses"
            " that quote",
        )
    try:
        resolved = Path(argument).resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        return refuse("unreadable", f"cannot be read: {reason(exc)}")
    location = long_form(resolved)
    try:
        found = _stat(location)
    except (FileNotFoundError, NotADirectoryError):
        return refuse("missing", "no such file or folder", resolved)
    except (OSError, ValueError) as exc:
        return refuse("unreadable", f"cannot be read: {reason(exc)}", resolved)
    name = resolved.name
    if stat.S_ISDIR(found.st_mode):
        return refuse("folder", f"a folder; {_PROJECTS_DIALOG}", resolved)
    if name.lower() == PROJECT_FILE:
        return refuse("project_file", f"a Proteia project file; {_PROJECTS_DIALOG}", resolved)
    if not stat.S_ISREG(found.st_mode):
        return refuse("unreadable", "cannot be read: not a regular file", resolved)
    if PurePath(name).suffix.lower() not in IMAGE_SUFFIXES:
        types = ", ".join(IMAGE_SUFFIXES)
        return refuse("unsupported_type", f"not an image type Proteia imports ({types})", resolved)
    if found.st_size == 0:
        return refuse("empty", "the file is empty", resolved)
    if found.st_size > MAX_UPLOAD_BYTES:
        return refuse("too_large", f"larger than {_size(MAX_UPLOAD_BYTES)}", resolved)
    try:
        check_name(name)
    except OperationError:
        return refuse("bad_name", "its name cannot be stored; rename the file", resolved)
    try:
        _readable(location)
    except OSError as exc:
        return refuse("unreadable", f"cannot be read: {reason(exc)}", resolved)
    file = ImageFile(argument, resolved, name, found.st_size, location)
    identity: Hashable = (found.st_dev, found.st_ino)
    if not found.st_ino:  # a file system with no file ids
        identity = os.path.normcase(str(resolved))
    return file, identity


def long_form(path: Path) -> Path:
    """``path`` (absolute) as file-system calls take it: on Windows in its
    ``\\\\?\\`` form, which has no length limit (``\\\\?\\UNC\\server\\share\\…`` for
    ``\\\\server\\share\\…``); unchanged elsewhere, or when it has that form, or
    names a device, already."""
    if os.name != "nt":
        return path
    text = str(path)
    if text.startswith((_LONG_PREFIX, _DEVICE_PREFIX)):
        return path
    if text.startswith("\\\\"):
        return Path(f"{_LONG_PREFIX}UNC\\{text[2:]}")
    return Path(_LONG_PREFIX + text)


def _stat(location: Path) -> os.stat_result:
    return os.stat(location)


def _readable(location: Path) -> None:
    with open(location, "rb"):
        pass


def reason(exc: BaseException) -> str:
    """Why a call failed, as the operating system says it (in its language),
    without the path."""
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    return str(exc) or type(exc).__name__


def _size(limit: int) -> str:
    mib, rest = divmod(limit, 1024**2)
    return f"{limit} bytes" if rest else f"{mib} MiB"


def _no_surrogates(text: str) -> str:
    return "".join("\ufffd" if 0xD800 <= ord(c) <= 0xDFFF else c for c in text)
