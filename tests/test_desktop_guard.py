# SPDX-License-Identifier: Apache-2.0
"""The shared guard (conftest ``_keep_tests_off_the_desktop``): a test that
would launch Proteia for real neither opens the user's browser nor writes the
user's state folder."""

import webbrowser

import pytest

from proteia.web import launch


def test_the_real_browser_cannot_be_opened_from_a_test():
    with pytest.raises(AssertionError, match="real browser"):
        webbrowser.open("http://127.0.0.1:1/")


def test_the_state_folder_is_the_test_s_own():
    assert "user-state" in str(launch.state_dir())
