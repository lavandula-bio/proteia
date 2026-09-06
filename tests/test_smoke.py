# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the package and its subpackages import and expose a version."""

import proteia
from proteia import core, gui  # noqa: F401  (import-only check)


def test_version():
    assert proteia.__version__ == "0.1.0"
