# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the package and its subpackages import and expose a version."""

import importlib.metadata

import proteia
from proteia import core, gui  # noqa: F401  (import-only check)


def test_version():
    assert proteia.__version__ == "0.1.0.dev0"


def test_version_matches_the_installed_metadata():
    # Records report proteia.__version__; it must not drift from pyproject.toml.
    assert proteia.__version__ == importlib.metadata.version("proteia")
