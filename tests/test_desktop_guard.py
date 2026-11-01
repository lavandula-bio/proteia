# SPDX-License-Identifier: Apache-2.0
"""The shared guards (conftest ``_keep_tests_off_the_desktop`` and the refusal
checks): a test that would launch Proteia for real neither opens the user's
browser nor writes the user's state folder; :func:`conftest.guards_in_place`
tells whether every guard is in place, and no test ends with one taken back
(#250)."""

import shutil
import webbrowser
from pathlib import Path

import pytest

from conftest import guards_in_place
from proteia.core import operations
from proteia.core.session import ProjectSession
from proteia.web import launch


def test_the_real_browser_cannot_be_opened_from_a_test():
    with pytest.raises(AssertionError, match="real browser"):
        webbrowser.open("http://127.0.0.1:1/")


def test_the_state_folder_is_the_test_s_own():
    assert "user-state" in str(launch.state_dir())


def _unwrapped(owner: object, name: str):
    """The change that puts back what the refusal check on ``owner.name``
    wraps: Proteia's own, unchecked."""
    return lambda patch, own, base: patch.setattr(owner, name, getattr(owner, name).__wrapped__)


def _held(owner: object, name: str):
    """The change that puts a test's own patch over the refusal check on
    ``owner.name``, one that calls it (as test_web_api holds an import)."""

    def change(patch, own, base):
        checked = getattr(owner, name)
        patch.setattr(owner, name, lambda *args, **kwargs: checked(*args, **kwargs))

    return change


def _browsers(order: list[str] | None):
    """The change that makes ``order`` the order webbrowser tries browsers in."""
    return lambda patch, own, base: patch.setattr(webbrowser, "_tryorder", order)


def _state(variable: str, folder):
    """The change that sets ``variable`` to ``folder(own, base)``: of the
    test's own folder and the folder pytest makes this run's temporary
    folders in; deleted when ``folder`` is None."""
    if folder is None:
        return lambda patch, own, base: patch.delenv(variable)
    return lambda patch, own, base: patch.setenv(variable, str(folder(own, base)))


@pytest.mark.parametrize(
    ("change", "in_place"),
    [
        pytest.param(_unwrapped(operations, "import_image"), False, id="import-check-gone"),
        pytest.param(_unwrapped(ProjectSession, "_read_checked"), False, id="read-check-gone"),
        pytest.param(_held(operations, "import_image"), True, id="import-held"),
        pytest.param(_held(ProjectSession, "_read_checked"), True, id="read-held"),
        pytest.param(_browsers(None), False, id="browsers-not-looked-up"),
        pytest.param(_browsers(["firefox", "proteia-test-guard"]), False, id="real-browser-first"),
        pytest.param(_state("LOCALAPPDATA", lambda own, base: base.parent), False, id="local-out"),
        pytest.param(
            _state("XDG_STATE_HOME", lambda own, base: base.parent / f"{base.name}0"),
            False,
            id="xdg-beside-of-same-prefix",
        ),
        pytest.param(_state("XDG_STATE_HOME", None), False, id="xdg-unset"),
        pytest.param(_state("LOCALAPPDATA", lambda own, base: ""), False, id="local-empty"),
        pytest.param(_state("LOCALAPPDATA", lambda own, base: own), True, id="local-own"),
        pytest.param(
            _state("XDG_STATE_HOME", lambda own, base: own / "state µ"), True, id="xdg-own"
        ),
    ],
)
def test_the_guards_are_told_in_place_only_while_each_one_is(
    monkeypatch, tmp_path, tmp_path_factory, change, in_place
):
    # #233, #248 review, #250: monkeypatch.undo() takes back every guard at
    # once; each one taken back alone is told (the import's own __wrapped__,
    # its lock's, told none). A test's own patch over a refusal check, or a
    # state folder of its own (in pytest's temporary folders, as
    # test_packaging's are), leaves it guarded; the user's folder or any
    # beside them does not, nor none.
    base = tmp_path_factory.getbasetemp()
    assert guards_in_place()
    with monkeypatch.context() as patch:
        change(patch, tmp_path, base)
        assert guards_in_place() is in_place
    assert guards_in_place()


# Tests run by a pytest of their own on conftest.py as it is (ASCII only: the
# inner run's output is read back as UTF-8, and Windows writes it as cp950).
# Two end with every guard taken back, one of them expected to fail; three
# end with the guards in place.
_INNER = """
import sys

import pytest

LEVEL = 1


def test_takes_back_every_patch(monkeypatch):
    monkeypatch.undo()


@pytest.mark.xfail(strict=True)
def test_expected_to_fail_takes_back_every_patch(monkeypatch):
    monkeypatch.undo()
    raise AssertionError("as expected")


def test_takes_back_its_own_patch(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(sys.modules[__name__], "LEVEL", 2)
        assert LEVEL == 2
    assert LEVEL == 1


def test_leaves_its_own_patch_to_monkeypatch(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "LEVEL", 3)


def test_gives_itself_a_state_folder_of_its_own(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
"""


def test_a_test_that_ends_with_the_guards_taken_back_fails(pytester):
    # #250: monkeypatch.undo() takes back conftest's guards too, so whatever
    # a test did after it ran unguarded; conftest's check fails each test
    # that ends so, at its teardown (an expected failure too), and only those.
    # The inner run inherits this test's state folder, so even its undo()
    # never reaches the user's.
    pytester.makeini("[pytest]\n")
    shutil.copyfile(Path(__file__).with_name("conftest.py"), pytester.path / "conftest.py")
    pytester.makepyfile(test_inner=_INNER)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", timeout=300)
    result.assert_outcomes(passed=4, xfailed=1, errors=2)
    taken_back = "E *GuardsTakenBack: the test ended with the guards of conftest.py taken back*"
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_takes_back_every_patch*",
            taken_back,
            "*ERROR at teardown of test_expected_to_fail_takes_back_every_patch*",
            taken_back,
            # The short summary, cut to the terminal's width.
            "ERROR test_inner.py::test_takes_back_every_patch*",
            "ERROR test_inner.py::test_expected_to_fail_takes_back_every_patch*",
        ]
    )
