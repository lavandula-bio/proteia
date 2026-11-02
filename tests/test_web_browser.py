# SPDX-License-Identifier: Apache-2.0
"""Opening the launch's redirect page (#251): on Windows in the browser Windows
names for web links (http), not through the ``.html`` file association, and
else as before, with the reason the console prints when the page may not have
reached a browser.

Every test gives :func:`browser.open_page` stand-ins for the shell's answer,
the start of the browser and ``webbrowser.open`` (here an ``.html``
association that is not a browser): none reads the registry or starts a
program (conftest guards both)."""

import ctypes
import dataclasses
import inspect
import os
from pathlib import Path, PureWindowsPath

import pytest

from conftest import DESKTOP_CALLS
from proteia.web import browser, launch
from proteia.web.browser import Handler, NotOpened
from proteia.web.launch import REDIRECT_FILE

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BRAVE = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
FIREFOX = r"C:\Program Files\Mozilla Firefox\firefox.exe"
IE = r"C:\Program Files\Internet Explorer\iexplore.exe"
UNQUOTED = r"C:\Program Files\Some Browser\BROWSER.EXE"
RUNDLL32 = r"C:\Windows\System32\rundll32.exe"
URL_DLL = r"C:\Windows\System32\url.dll"

# A redirect page in a profile folder whose name holds spaces and µ α β, as
# the launcher hands it over (Path.as_uri), spelled out.
PAGE = PureWindowsPath(r"C:\Users\µ α β\AppData\Local\Proteia\open-proteia.html")
URI = "file:///C:/Users/%C2%B5%20%CE%B1%20%CE%B2/AppData/Local/Proteia/open-proteia.html"

AS_FILE = "so the page was opened as a file instead, which may not reach a browser."
# What Proteia knows when the shell names it no .exe to start (#262 review):
# no handler, a packaged (Store) app, a missing program, access denied, a DLL.
NO_PROGRAM = "Proteia found no program it could start for web links (http)"
NO_HANDLER = f"{NO_PROGRAM}, {AS_FILE}"


class NotABrowser:
    """Stands in for webbrowser.open with an ``.html`` association that is an
    editor: it takes the file (answers ``took``) and never follows its
    redirect."""

    def __init__(self, took: bool = True) -> None:
        self.took = took
        self.uris: list[str] = []

    def __call__(self, uri: str) -> bool:
        self.uris.append(uri)
        return self.took


class Starts:
    """Stands in for the start of a program: records the program and the
    command line, or raises ``error``."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def __call__(self, program: str, line: str) -> None:
        self.calls.append((program, line))
        if self.error is not None:
            raise self.error


def never() -> Handler:
    raise AssertionError("the browser Windows names for http was asked for")


def test_the_page_uri_holds_no_space_quote_or_backslash():
    # The shell puts a web address in the command as it is: the page's URI
    # must hold nothing the command line would split on.
    assert PAGE.as_uri() == URI


@pytest.mark.parametrize(
    ("program", "command", "line"),
    [
        pytest.param(
            EDGE, f'"{EDGE}" --single-argument %1', f'"{EDGE}" --single-argument {URI}', id="edge"
        ),
        pytest.param(
            CHROME,
            f'"{CHROME}" --single-argument %1',
            f'"{CHROME}" --single-argument {URI}',
            id="chrome",
        ),
        pytest.param(
            BRAVE,
            f'"{BRAVE}" --single-argument %1',
            f'"{BRAVE}" --single-argument {URI}',
            id="brave",
        ),
        pytest.param(
            FIREFOX,
            f'"{FIREFOX}" -osint -url "%1"',
            f'"{FIREFOX}" -osint -url "{URI}"',
            id="firefox",
        ),
        pytest.param(CHROME, f'"{CHROME}" -- "%1"', f'"{CHROME}" -- "{URI}"', id="older-chrome"),
        pytest.param(IE, f'"{IE}" %1', f'"{IE}" {URI}', id="unquoted"),
        pytest.param(EDGE, f'"{EDGE}" "%1"', f'"{EDGE}" "{URI}"', id="system-default"),
        # An unquoted program path with spaces: the shell's program is the
        # one started, not the command's first word ("C:\Program").
        pytest.param(UNQUOTED, f'{UNQUOTED} -url "%1"', f'{UNQUOTED} -url "{URI}"', id="spaces"),
    ],
)
def test_on_windows_the_page_goes_to_the_browser_named_for_http(program, command, line):
    # #251: not through the .html association, which may be an editor that
    # never follows the page's redirect. The program started is the one the
    # shell names, given the command line it would build.
    starts, association = Starts(), NotABrowser()
    opened = browser.open_page(
        URI,
        windows=True,
        environ={},
        handler=lambda: Handler(program, command),
        start=starts,
        fallback=association,
    )
    assert opened is True
    assert starts.calls == [(program, line)]
    assert association.uris == []


@pytest.mark.parametrize(
    ("command", "line"),
    [
        pytest.param("b.exe %1", "b.exe page", id="1"),
        pytest.param("b.exe %0", "b.exe page", id="0"),
        pytest.param('b.exe "%0" %2', 'b.exe "page" ', id="0-and-others"),
        pytest.param("b.exe %L", "b.exe page", id="L"),
        pytest.param("b.exe %l", "b.exe page", id="l"),
        pytest.param('b.exe "%1" %*', 'b.exe "page" ', id="star-dropped"),
        pytest.param("b.exe %1 %2 %9", "b.exe page  ", id="others-dropped"),
        pytest.param("b.exe %1 --again %1", "b.exe page --again page", id="twice"),
        pytest.param("b.exe %I %1", "b.exe %I page", id="unknown-kept"),
        pytest.param("b.exe --new-window", None, id="no-place"),
        pytest.param("b.exe %2 %*", None, id="no-place-but-others"),
    ],
)
def test_the_page_takes_the_place_of_the_item_in_the_command(command, line):
    assert browser.command_line(command, "page") == line


def test_a_page_uri_is_put_in_the_command_once_never_read_again():
    # A folder named "a%1 %2 %L" is percent-encoded to "a%251%20%252%20%25L":
    # read again, its "%2" and "%L" would be taken for the command's own.
    uri = PureWindowsPath(r"C:\a%1 %2 %L\open-proteia.html").as_uri()
    assert uri == "file:///C:/a%251%20%252%20%25L/open-proteia.html"
    assert browser.command_line("b.exe %1 %*", uri) == f"b.exe {uri} "


def test_a_page_in_a_folder_with_spaces_and_non_ascii_names_reaches_the_browser_whole(tmp_path):
    folder = tmp_path / "µ α β 狀態" / "a b"
    page = folder / REDIRECT_FILE
    starts = Starts()
    opened = browser.open_page(
        page.as_uri(),
        windows=True,
        environ={},
        handler=lambda: Handler(CHROME, f'"{CHROME}" --single-argument %1'),
        start=starts,
        fallback=NotABrowser(),
    )
    assert opened is True
    ((_, line),) = starts.calls
    # Chromium takes all that follows --single-argument as one argument, raw.
    given = line.split(" --single-argument ", 1)[1]
    assert given.isascii() and " " not in given and '"' not in given
    assert Path.from_uri(given) == page


def test_with_no_browser_named_for_http_the_page_is_opened_as_before(tmp_path):
    # #251: the fallback is today's webbrowser.open, and the console is told
    # which file to open by hand.
    page = tmp_path / "state µ" / REDIRECT_FILE
    starts, association = Starts(), NotABrowser()
    opened = browser.open_page(
        page.as_uri(), windows=True, environ={}, handler=lambda: None, start=starts,
        fallback=association,
    )  # fmt: skip
    assert opened == NotOpened(page, NO_HANDLER)
    assert association.uris == [page.as_uri()] and starts.calls == []


def test_when_the_page_cannot_be_opened_as_a_file_either_the_reason_says_so(tmp_path):
    page = tmp_path / REDIRECT_FILE
    opened = browser.open_page(
        page.as_uri(), windows=True, environ={}, handler=lambda: None, start=Starts(),
        fallback=NotABrowser(took=False),
    )  # fmt: skip
    assert opened == NotOpened(
        page, f"{NO_PROGRAM}, and the page could not be opened as a file either."
    )


# What the shell answers (#262 review, read-only on Windows 11): a packaged
# (Store) app as the http default, or a program that is gone, gives no
# command or no program; a command run through rundll32 names its DLL.
CO_E_APPNOTFOUND = 0x800401F5 - 2**32
E_ACCESSDENIED = 0x80070005 - 2**32
WITH_CHROME = {1: f'"{CHROME}" --single-argument %1', 2: CHROME}


@pytest.mark.parametrize(
    "named",
    [
        pytest.param(lambda: browser.handler_named_by(Shell({})), id="store-app"),
        pytest.param(
            lambda: browser.handler_named_by(Shell(WITH_CHROME, {(2, 1): CO_E_APPNOTFOUND})),
            id="program-gone",
        ),
        pytest.param(
            lambda: browser.handler_named_by(Shell(WITH_CHROME, {(2, 1): E_ACCESSDENIED})),
            id="access-denied",
        ),
        pytest.param(
            lambda: Handler(URL_DLL, f'"{RUNDLL32}" {URL_DLL},FileProtocolHandler %1'),
            id="rundll32",
        ),
        pytest.param(lambda: Handler(r"C:\Tools\browser", r"C:\Tools\browser %1"), id="no-suffix"),
        pytest.param(lambda: Handler(r"C:\Tools\open.cmd", r"C:\Tools\open.cmd %1"), id="cmd"),
    ],
)
def test_with_no_program_proteia_can_start_the_reason_says_only_that(tmp_path, named):
    # #262 review m1, m2: Proteia cannot tell these apart, and none is "no
    # browser"; it starts only an .exe (a DLL is never called the browser).
    page = tmp_path / REDIRECT_FILE
    starts, association = Starts(), NotABrowser()
    opened = browser.open_page(
        page.as_uri(), windows=True, environ={}, handler=named, start=starts,
        fallback=association,
    )  # fmt: skip
    assert opened == NotOpened(page, NO_HANDLER)
    assert starts.calls == [] and association.uris == [page.as_uri()]


@pytest.mark.parametrize(
    ("handler", "error", "why"),
    [
        pytest.param(
            Handler(CHROME, f'"{CHROME}" --single-argument %1'),
            None,
            "Proteia could not ask Windows which program opens web links (http) (RuntimeError)",
            id="lookup",
        ),
        pytest.param(
            Handler(CHROME, f'"{CHROME}" --single-argument %1'),
            ValueError("embedded null character"),
            "The browser Windows names for web links (http), chrome.exe, could not be started"
            " (ValueError)",
            id="start",
        ),
    ],
)
def test_an_unexpected_failure_of_the_lookup_or_the_start_falls_back(tmp_path, handler, error, why):
    # #262 review n1: a failure of either only degrades the launch; its type
    # names it (its text might hold a path).
    page = tmp_path / REDIRECT_FILE
    association = NotABrowser()

    def lookup() -> Handler:
        if error is None:
            raise RuntimeError(r"C:\Users\someone\secret")
        return handler

    opened = browser.open_page(
        page.as_uri(), windows=True, environ={}, handler=lookup, start=Starts(error),
        fallback=association,
    )  # fmt: skip
    assert opened == NotOpened(page, f"{why}, {AS_FILE}")
    assert association.uris == [page.as_uri()]


@pytest.mark.parametrize("where", ["lookup", "start"])
def test_an_assertion_is_never_taken_for_the_system_s_answer(tmp_path, where):
    # A failed assertion is a broken invariant, or a test's guard (conftest):
    # it goes on up, and the page is not opened some other way.
    association = NotABrowser()

    def lookup() -> Handler:
        if where == "lookup":
            raise AssertionError("guard")
        return Handler(CHROME, f'"{CHROME}" --single-argument %1')

    with pytest.raises(AssertionError, match="^guard$"):
        browser.open_page(
            (tmp_path / REDIRECT_FILE).as_uri(), windows=True, environ={}, handler=lookup,
            start=Starts(AssertionError("guard")), fallback=association,
        )  # fmt: skip
    assert association.uris == []


@pytest.mark.parametrize(
    ("handler", "said"),
    [
        pytest.param(None, "asked Windows for the browser", id="lookup"),
        pytest.param(
            lambda: Handler(CHROME, CHROME_COMMAND), "tried to start the browser", id="start"
        ),
    ],
)
def test_the_desktop_guard_still_stops_a_test_that_would_reach_the_browser(tmp_path, handler, said):
    # conftest's stand-ins for the shell lookup and the start: their
    # AssertionError still stops the test, with a fallback of its own too.
    association = NotABrowser()
    with pytest.raises(AssertionError, match=said):
        browser.open_page(
            (tmp_path / REDIRECT_FILE).as_uri(), windows=True, environ={}, handler=handler,
            fallback=association,
        )  # fmt: skip
    assert association.uris == []


def test_a_browser_that_cannot_be_started_is_named_without_its_path(tmp_path):
    # The reason goes to the session log too: the program's name, never its
    # folder (a per-user install is in the user's profile).
    program = r"C:\Users\someone µ\AppData\Local\Google\Chrome\Application\chrome.exe"
    page = tmp_path / REDIRECT_FILE
    missing = FileNotFoundError(2, "The system cannot find the file specified")
    starts, association = Starts(missing), NotABrowser()
    opened = browser.open_page(
        page.as_uri(),
        windows=True,
        environ={},
        handler=lambda: Handler(program, f'"{program}" --single-argument %1'),
        start=starts,
        fallback=association,
    )
    assert opened == NotOpened(
        page,
        "The browser Windows names for web links (http), chrome.exe, could not be started"
        f" (The system cannot find the file specified), {AS_FILE}",
    )
    assert "someone" not in opened.reason
    assert len(starts.calls) == 1 and association.uris == [page.as_uri()]


def test_a_command_with_no_place_for_the_page_is_not_run(tmp_path):
    page = tmp_path / REDIRECT_FILE
    starts, association = Starts(), NotABrowser()
    opened = browser.open_page(
        page.as_uri(),
        windows=True,
        environ={},
        handler=lambda: Handler(r"C:\Tools\b.exe", r'"C:\Tools\b.exe" --new-window'),
        start=starts,
        fallback=association,
    )
    assert opened == NotOpened(
        page,
        "The browser Windows names for web links (http), b.exe, cannot be given the page,"
        f" {AS_FILE}",
    )
    assert starts.calls == [] and association.uris == [page.as_uri()]


@pytest.mark.parametrize("took", [True, False])
@pytest.mark.parametrize(
    ("windows", "environ"),
    [
        pytest.param(False, {}, id="not-windows"),
        pytest.param(True, {"BROWSER": "stand-in %s"}, id="browser-set"),
        pytest.param(False, {"BROWSER": "stand-in %s"}, id="not-windows-browser-set"),
    ],
)
def test_elsewhere_or_with_a_browser_set_webbrowser_opens_the_page(
    tmp_path, windows, environ, took
):
    # BROWSER is an explicit choice, which webbrowser tries first: the
    # packaging checks name a stand-in there that only records the address.
    page = tmp_path / REDIRECT_FILE
    association = NotABrowser(took)
    opened = browser.open_page(
        page.as_uri(), windows=windows, environ=environ, handler=never, start=Starts(),
        fallback=association,
    )  # fmt: skip
    if took:
        assert opened is True
    else:
        assert opened == NotOpened(page, "No web browser could be started for the page.")
    assert association.uris == [page.as_uri()]


def test_an_empty_browser_variable_is_no_choice(tmp_path):
    starts = Starts()
    opened = browser.open_page(
        (tmp_path / REDIRECT_FILE).as_uri(),
        windows=True,
        environ={"BROWSER": ""},
        handler=lambda: Handler(EDGE, f'"{EDGE}" --single-argument %1'),
        start=starts,
        fallback=NotABrowser(),
    )
    assert opened is True and len(starts.calls) == 1


# --- What the shell names for http (AssocQueryStringW) ---

S_OK, S_FALSE = 0, 1
NO_ASSOCIATION = 0x80070483 - 2**32  # HRESULT_FROM_WIN32(ERROR_NO_ASSOCIATION), as a LONG
E_POINTER = 0x80004003 - 2**32


class Shell:
    """Stands in for AssocQueryStringW: ``strings[which]`` for the http open
    verb, as the shell answers (no buffer: S_FALSE and the size, with its
    terminating null; then S_OK), or the HRESULT of ``fails[(which, call)]``
    (``call`` 1 for the size, 2 for the string), having written as much of
    the string as fits, as the shell does without ASSOCF_NOTRUNCATE."""

    def __init__(self, strings: dict[int, str], fails: dict | None = None) -> None:
        self.strings = strings
        self.fails = fails or {}
        self.calls: list[tuple[int, int, str, str, int]] = []

    def __call__(self, flags, which, assoc, extra, out, size) -> int:
        call = 1 if out is None else 2
        self.calls.append((flags, which, assoc, extra, call))
        text = self.strings.get(which)
        if text is None:
            return NO_ASSOCIATION
        if (which, call) in self.fails:
            if out is not None:
                out.value = text[: max(size.contents.value - 1, 0)][:4]
            return self.fails[(which, call)]
        needed = len(text) + 1
        if out is None:
            size.contents.value = needed
            return S_FALSE
        if size.contents.value < needed:
            return E_POINTER
        out.value = text
        size.contents.value = needed
        return S_OK


COMMAND, EXECUTABLE = 1, 2  # ASSOCSTR_COMMAND, ASSOCSTR_EXECUTABLE
CHROME_COMMAND = f'"{CHROME}" --single-argument %1'


def test_the_shell_is_asked_for_the_http_open_command_and_its_program():
    shell = Shell({COMMAND: CHROME_COMMAND, EXECUTABLE: CHROME})
    assert browser.handler_named_by(shell) == Handler(CHROME, CHROME_COMMAND)
    # ASSOCF_IS_PROTOCOL | ASSOCF_NOTRUNCATE: the user's default for a
    # protocol, and an error rather than a cut command.
    assert shell.calls == [
        (0x1020, COMMAND, "http", "open", 1),
        (0x1020, COMMAND, "http", "open", 2),
        (0x1020, EXECUTABLE, "http", "open", 1),
        (0x1020, EXECUTABLE, "http", "open", 2),
    ]


@pytest.mark.parametrize(
    ("strings", "fails"),
    [
        pytest.param({}, {}, id="no-association"),
        pytest.param({EXECUTABLE: CHROME}, {}, id="no-command"),
        pytest.param({COMMAND: CHROME_COMMAND}, {}, id="no-executable"),
        pytest.param({COMMAND: "", EXECUTABLE: CHROME}, {}, id="empty-command"),
        pytest.param({COMMAND: CHROME_COMMAND, EXECUTABLE: ""}, {}, id="empty-executable"),
        pytest.param(
            {COMMAND: CHROME_COMMAND, EXECUTABLE: CHROME},
            {(COMMAND, 1): S_OK},
            id="size-not-told",
        ),
        pytest.param(
            {COMMAND: CHROME_COMMAND, EXECUTABLE: CHROME},
            {(COMMAND, 2): E_POINTER},
            id="command-cut",
        ),
        pytest.param(
            {COMMAND: CHROME_COMMAND, EXECUTABLE: CHROME},
            {(EXECUTABLE, 2): S_FALSE},
            id="executable-not-given",
        ),
    ],
)
def test_no_handler_unless_the_shell_gives_both_whole(strings, fails):
    assert browser.handler_named_by(Shell(strings, fails)) is None


def test_the_shell_s_answers_fit_the_buffer_it_asked_for():
    # The size the shell tells is the buffer given: one character less, and
    # the shell would refuse it.
    sizes: list[int] = []
    shell = Shell({COMMAND: CHROME_COMMAND, EXECUTABLE: CHROME})

    def told(flags, which, assoc, extra, out, size):
        if out is not None:
            sizes.append(ctypes.sizeof(out) // ctypes.sizeof(ctypes.c_wchar))
        return shell(flags, which, assoc, extra, out, size)

    assert browser.handler_named_by(told) == Handler(CHROME, CHROME_COMMAND)
    assert sizes == [len(CHROME_COMMAND) + 1, len(CHROME) + 1]


def test_the_shell_is_asked_through_assoc_query_string_w(monkeypatch):
    # Proteia's own lookup, with the function it binds stood in for (the
    # shell's own would read the user's registry). Elsewhere there is no
    # shell library to bind it from.
    shell = Shell({COMMAND: CHROME_COMMAND, EXECUTABLE: CHROME})
    bound: list[object] = []

    def prototype(*types):
        bound.append(types)

        def function(spec):
            bound.append(spec[0])
            return shell

        return function

    with monkeypatch.context() as patch:
        patch.setattr(ctypes, "WINFUNCTYPE", prototype, raising=False)
        found = DESKTOP_CALLS["_http_handler"]()
    if os.name != "nt":
        assert found is None
        return
    assert found == Handler(CHROME, CHROME_COMMAND)
    wchar_p = ctypes.c_wchar_p
    assert bound == [
        (
            ctypes.c_long,
            ctypes.c_uint,
            ctypes.c_int,
            wchar_p,
            wchar_p,
            wchar_p,
            ctypes.POINTER(ctypes.c_uint32),
        ),
        "AssocQueryStringW",
    ]


def test_with_no_shell_to_ask_no_browser_is_named(monkeypatch):
    with monkeypatch.context() as patch:
        patch.delattr(ctypes, "WINFUNCTYPE", raising=False)
        assert DESKTOP_CALLS["_http_handler"]() is None


@pytest.mark.skipif(os.name != "nt", reason="a Windows command line, started on Windows only")
def test_the_browser_is_started_with_the_command_line_as_it_is(monkeypatch):
    # Proteia's own start, with the process start stood in for: the program
    # the shell named, and the line as built (no shell, no quoting), on its
    # own (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP).
    calls: list[tuple[tuple, dict]] = []
    with monkeypatch.context() as patch:
        patch.setattr(
            browser.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs))
        )
        DESKTOP_CALLS["_start"](CHROME, f'"{CHROME}" --single-argument {URI}')
    devnull = browser.subprocess.DEVNULL
    assert calls == [
        (
            (f'"{CHROME}" --single-argument {URI}',),
            {
                "executable": CHROME,
                "stdin": devnull,
                "stdout": devnull,
                "stderr": devnull,
                "close_fds": True,
                "creationflags": 0x00000008 | 0x00000200,
            },
        )
    ]


def test_what_open_page_says_is_kept_as_it_was_said(tmp_path):
    # Kept on a launch's frozen records (launch.Opened), so frozen too.
    named = browser.handler_named_by(Shell({COMMAND: CHROME_COMMAND, EXECUTABLE: CHROME}))
    said = browser.open_page(
        (tmp_path / REDIRECT_FILE).as_uri(), windows=True, environ={}, handler=lambda: None,
        start=Starts(), fallback=NotABrowser(),
    )  # fmt: skip
    for record in (named, said):
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.reason = "else"
        assert hash(record) == hash(dataclasses.replace(record))
    assert hash(launch.Opened(unopened=said)) == hash(launch.Opened(unopened=said))


# --- The launch ---


def test_a_launch_opens_its_page_this_way_unless_told_otherwise():
    assert inspect.signature(launch.start).parameters["opener"].default is browser.open_page


def test_the_browser_is_given_the_redirect_file_never_the_token(tmp_path):
    # ADR 0002: the token stays in the private redirect file, off every
    # command line other local users can list.
    folder = tmp_path / "state µ α β 狀態" / "a b"
    starts, association = Starts(), NotABrowser()

    def opener(uri: str) -> bool | NotOpened:
        return browser.open_page(
            uri,
            windows=True,
            environ={},
            handler=lambda: Handler(CHROME, CHROME_COMMAND),
            start=starts,
            fallback=association,
        )

    instance = launch.start(folder=folder, opener=opener)
    try:
        page = folder / REDIRECT_FILE
        assert starts.calls == [(CHROME, f'"{CHROME}" --single-argument {page.as_uri()}')]
        assert instance.token not in starts.calls[0][1] and "#token" not in starts.calls[0][1]
        assert instance.token in page.read_text(encoding="utf-8")
        assert association.uris == [] and instance.unopened is None
    finally:
        instance.close()


@pytest.mark.skipif(os.name != "nt", reason="the browser Windows names for http")
def test_a_launch_on_windows_starts_the_browser_named_for_http(tmp_path, monkeypatch):
    # #251: the launch's own opener, with the shell's answer and the start
    # stood in for; the .html association (webbrowser, which conftest makes
    # fail the test) is never asked.
    folder = tmp_path / "state µ"
    starts = Starts()
    with monkeypatch.context() as patch:
        patch.delenv("BROWSER", raising=False)
        patch.setattr(browser, "_http_handler", lambda: Handler(CHROME, CHROME_COMMAND))
        patch.setattr(browser, "_start", starts)
        instance = launch.start(folder=folder)
    try:
        page = folder / REDIRECT_FILE
        assert starts.calls == [(CHROME, f'"{CHROME}" --single-argument {page.as_uri()}')]
        assert instance.unopened is None
    finally:
        instance.close()
