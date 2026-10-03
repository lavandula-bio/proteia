# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the package and its subpackages import and expose a version, and
no source imports a desktop GUI toolkit (the interface is the local web app,
ADR 0002)."""

import ast
import importlib.metadata
from pathlib import Path

import proteia
from proteia import core, viz, web  # noqa: F401  (import-only check)

ROOT = Path(__file__).resolve().parents[1]
# The napari app and the Qt it ran on, removed in #57.
GUI_TOOLKITS = ("napari", "magicgui", "qtpy", "PySide6", "shiboken6", "PyQt5", "PyQt6")
# matplotlib's Qt backends (backend_qtagg, backend_qtcairo, ...) and the modules
# that import a Qt binding for them (qt_compat, qt_editor).
QT_BACKENDS = ("matplotlib.backends.backend_qt", "matplotlib.backends.qt_")


def test_version():
    assert proteia.__version__ == "0.1.0.dev0"


def test_version_matches_the_installed_metadata():
    # Records report proteia.__version__; it must not drift from pyproject.toml.
    assert proteia.__version__ == importlib.metadata.version("proteia")


def _imported(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names += [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
    return names


def _is_gui(name: str) -> bool:
    return name.split(".")[0] in GUI_TOOLKITS or name.startswith(QT_BACKENDS)


def test_no_source_imports_a_gui_toolkit():
    found = [
        f"{path.relative_to(ROOT).as_posix()}: {name}"
        for folder in ("src", "tests")
        for path in sorted((ROOT / folder).rglob("*.py"))
        for name in _imported(path)
        if _is_gui(name)
    ]
    assert not found


def test_the_check_catches_every_way_into_qt(tmp_path):
    lines = (
        "import napari",
        "from PySide6.QtWidgets import QApplication",
        "from matplotlib.backends import backend_qtagg",
        "from matplotlib.backends import qt_compat",
        "from matplotlib.backends.qt_compat import QtWidgets",
        "import matplotlib.backends.qt_editor.figureoptions",
        "import matplotlib.pyplot",
        "from matplotlib.backends import backend_agg",
    )
    source = tmp_path / "probe.py"
    source.write_text("\n".join(lines), encoding="utf-8")
    assert [name for name in _imported(source) if _is_gui(name)] == [
        "napari",
        "PySide6.QtWidgets",
        "PySide6.QtWidgets.QApplication",
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.qt_compat",
        "matplotlib.backends.qt_compat",
        "matplotlib.backends.qt_compat.QtWidgets",
        "matplotlib.backends.qt_editor.figureoptions",
    ]
