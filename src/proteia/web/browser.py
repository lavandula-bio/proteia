# SPDX-License-Identifier: Apache-2.0
"""Open a launch's redirect page in a web browser (#251).

On Windows the page goes to the browser Windows names for web links (http): the
program and the command of the ``open`` verb the shell finds for the ``http``
protocol (``AssocQueryStringW``), which is the user's choice of default browser
(else the system's). That command runs with the page's ``file:`` URI in place of
its ``%1``, as the shell runs it with a web address. Opening the file itself, as
``webbrowser.open`` does on Windows, goes through the ``.html`` file association
instead, which may be an editor or another program that never follows the
page's redirect to Proteia's address.

The URI is the redirect page's, never Proteia's address, so the token is on no
command line (ADR 0002). :meth:`pathlib.Path.as_uri` percent-encodes it: it holds
no space, quote or backslash, so it is put in the command as it is, whether the
command quotes its ``%1`` or not. (Chromium-based browsers take all that follows
``--single-argument`` as one argument, quotes included.)

``webbrowser.open`` opens the page instead, as before #251: on other systems;
when ``BROWSER`` names a browser, an explicit choice that ``webbrowser`` tries
first (the packaging checks name one there that only records the address); and
when Windows names no browser for http, its command has no place for the page,
or it cannot be started. In those last cases, and whenever ``webbrowser.open``
fails, :func:`open_page` answers :class:`NotOpened`: the console then says why,
and which file to open by hand.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Final

from proteia.web import cli


@dataclass(frozen=True)
class Handler:
    """The browser Windows names for http: its ``program`` and the
    ``command`` of its ``open`` verb (which holds ``%1`` for the address)."""

    program: str
    command: str


@dataclass(frozen=True)
class NotOpened:
    """The page ``page`` may not have reached a browser: ``reason``, a
    sentence for the console and the session log, says why and what was done
    instead. It names no folder."""

    page: Path
    reason: str


# ASSOCF_IS_PROTOCOL (the user's default for a protocol, from Windows 8) and
# ASSOCF_NOTRUNCATE (an error, not a cut string, if the buffer is too small).
_ASSOC_FLAGS: Final = 0x1000 | 0x0020
_ASSOCSTR_COMMAND: Final = 1
_ASSOCSTR_EXECUTABLE: Final = 2
_S_OK: Final = 0
_S_FALSE: Final = 1  # no buffer given: the size is told
# The shell's insert sequences in a command: %1 and %L stand for the item
# opened (here the page), %* and %2 to %9 for further arguments (none here).
_INSERT: Final = re.compile(r"%([1Ll*2-9])")

Query = Callable[..., int]  # AssocQueryStringW's shape: an HRESULT


def open_page(
    uri: str,
    *,
    windows: bool | None = None,
    environ: Mapping[str, str] | None = None,
    handler: Callable[[], Handler | None] | None = None,
    start: Callable[[str, str], object] | None = None,
    fallback: Callable[[str], object] | None = None,
) -> bool | NotOpened:
    """Open the page at ``uri``, a ``file:`` URI as :meth:`~pathlib.Path.as_uri`
    makes it, in a web browser: True if a browser took it, else
    :class:`NotOpened`. The keywords stand in for what it uses (tests): whether
    this is Windows, the environment, the browser Windows names for http
    (:func:`_http_handler`), the start of a program with a command line
    (:func:`_start`) and ``webbrowser.open``."""
    windows = os.name == "nt" if windows is None else windows
    environ = os.environ if environ is None else environ
    handler = _http_handler if handler is None else handler
    start = _start if start is None else start
    fallback = webbrowser.open if fallback is None else fallback
    if not windows or environ.get("BROWSER"):
        if fallback(uri):
            return True
        return NotOpened(Path.from_uri(uri), "No web browser could be started for the page.")
    found = handler()
    if found is None:
        why = "Windows names no browser for web links (http)"
    else:
        named = f"The browser Windows names for web links (http), {_name(found.program)},"
        line = command_line(found.command, uri)
        if line is None:
            why = f"{named} cannot be given the page"
        else:
            try:
                start(found.program, line)
            except OSError as exc:
                why = f"{named} could not be started ({cli.reason(exc)})"
            else:
                return True
    if fallback(uri):
        did = "so the page was opened as a file instead, which may not reach a browser"
    else:
        did = "and the page could not be opened as a file either"
    return NotOpened(Path.from_uri(uri), f"{why}, {did}.")


def command_line(command: str, uri: str) -> str | None:
    """``command`` with ``uri`` in place of each ``%1`` or ``%L``, as the shell
    puts the item it opens there, and no further arguments (``%*``, ``%2`` to
    ``%9``); None if it has no place for ``uri``. One pass: ``uri`` is never
    read for insert sequences."""
    placed = False

    def insert(match: re.Match[str]) -> str:
        nonlocal placed
        item = match.group(1) in "1Ll"
        placed = placed or item
        return uri if item else ""

    line = _INSERT.sub(insert, command)
    return line if placed else None


def _name(program: str) -> str:
    """The file name of ``program``, a Windows path (on any system)."""
    return PureWindowsPath(program).name


def handler_named_by(query: Query) -> Handler | None:
    """The browser that ``query`` (``AssocQueryStringW``) names for http, the
    command and the program of its ``open`` verb; None unless it gives both
    whole."""
    command = _assoc_string(query, _ASSOCSTR_COMMAND)
    program = _assoc_string(query, _ASSOCSTR_EXECUTABLE)
    if command is None or program is None:
        return None
    return Handler(program=program, command=command)


def _assoc_string(query: Query, which: int) -> str | None:
    """The string ``which`` (an ASSOCSTR) of the http protocol's open verb, as
    the shell gives it: first its size (S_FALSE, with no buffer), then the
    string (S_OK); None if it gives either otherwise, or an empty string."""
    size = ctypes.c_uint32()
    if query(_ASSOC_FLAGS, which, "http", "open", None, ctypes.pointer(size)) != _S_FALSE:
        return None
    out = ctypes.create_unicode_buffer(size.value)
    if query(_ASSOC_FLAGS, which, "http", "open", out, ctypes.pointer(size)) != _S_OK:
        return None
    return out.value or None


def _http_handler() -> Handler | None:
    """The browser Windows names for http (:func:`handler_named_by`, asking
    the shell); None if it names none, or where there is no such shell."""
    try:  # WINFUNCTYPE and windll are Windows's only
        prototype = ctypes.WINFUNCTYPE(
            ctypes.c_long,  # HRESULT
            ctypes.c_uint,  # ASSOCF
            ctypes.c_int,  # ASSOCSTR
            ctypes.c_wchar_p,  # pszAssoc
            ctypes.c_wchar_p,  # pszExtra
            ctypes.c_wchar_p,  # pszOut
            ctypes.POINTER(ctypes.c_uint32),  # pcchOut
        )
        query = prototype(("AssocQueryStringW", ctypes.windll.shlwapi))
    except (AttributeError, OSError):
        return None
    return handler_named_by(query)


def _start(program: str, line: str) -> None:
    """Start ``program`` with the command line ``line`` as it is (no shell, no
    quoting: CreateProcess), and leave it running on its own: in a process
    group of its own, without the console. Windows only."""
    subprocess.Popen(
        line,
        executable=program,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
