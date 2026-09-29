# SPDX-License-Identifier: Apache-2.0
"""What Proteia's Windows bundle holds (ADR 0003): the lists the PyInstaller
spec (``proteia.spec``) builds from, and the rules the spec, the build
(``build.py``), the notices (``notices.py``) and the tests share.

The standard library only, so the tests run it where PyInstaller is not
installed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath, PureWindowsPath
from typing import Final

APP_NAME: Final = "Proteia"
EXE_NAME: Final = "Proteia.exe"
# The name the installer and the version resource give as the publisher. Provisional
# for v0.1: ADR 0003 leaves it open until the first public release.
PUBLISHER: Final = "Lavandula"
APP_URL: Final = "https://github.com/lavandula-bio/proteia"
# The ``proteia`` console command, which the frozen entry script runs.
ENTRY_POINT: Final = "proteia.web.launch:main"
# Proteia's own distribution: its license goes in as LICENSE.txt, not through its
# metadata, which names the folder it was installed from.
OWN_DISTRIBUTION: Final = "proteia"

# Proteia's package data: the web client, served from the package folder.
PACKAGE_DATA: Final = ("web/static/**/*",)
# The distributions whose versions the reproducibility record reads with
# importlib.metadata (proteia.core.record); the self-test checks them in the bundle.
RECORD_DISTRIBUTIONS: Final = ("numpy", "scipy", "scikit-image", "pillow", "tifffile")
# The matplotlib backends the app draws with: PNG previews through Agg, SVG
# charts, PDF exports. Without this PyInstaller's hook picks an interactive
# backend and takes its toolkit along (QtAgg took Qt while PySide6 was installed).
MATPLOTLIB_BACKENDS: Final = ("Agg", "SVG", "PDF")

# Modules the bundle leaves out although something may import them. napari
# left the dependencies in #57 and no locked package needs Qt, so the build
# environment comes without them (build.py checks); naming them here is a second
# guard. tkinter comes with Python, and setuptools and pkg_resources are
# PyInstaller's build requirements: nothing the app runs needs them.
EXCLUDES: Final = (
    "napari",
    "PySide6",
    "shiboken6",
    "PyQt5",
    "PyQt6",
    "qtpy",
    "tkinter",
    "_tkinter",
    "setuptools",
    "pkg_resources",
    "_distutils_hack",
)
# Names no file in a finished bundle may have: napari and Qt (ADR 0002's LGPL
# duties arise only when Qt ships). Matched case-insensitively against each
# part of every path in the bundle.
FORBIDDEN_NAMES: Final = re.compile(
    r"^(napari.*|pyside6.*|shiboken6.*|pyqt[56].*|qt[56].*|vispy.*|_?tcl.*|_?tk\d.*|_tk_data)$",
    re.IGNORECASE,
)

# Files at the top of the bundle's ``_internal`` folder that no package claims:
# the Python runtime and the libraries its standard modules load, and the
# Visual C++ runtime. notices.py gives each of them a notice.
PYTHON_RUNTIME_FILES: Final = re.compile(
    r"^(python3\d*\.dll|libcrypto-3(-x64)?\.dll|libssl-3(-x64)?\.dll|libffi-\d+\.dll"
    r"|sqlite3\.dll|base_library\.zip)$",
    re.IGNORECASE,
)
VC_RUNTIME_FILES: Final = re.compile(r"^(vcruntime140(_1)?|msvcp140(_\w+)?)\.dll$", re.IGNORECASE)
# PyInstaller's own modules in the bundle (its runtime hooks and their helpers).
PYINSTALLER_MODULES: Final = re.compile(r"^(pyi_rth_\w+|_pyi_rth_utils|pyimod\d\d_\w+)$")


def is_universal_crt(name: str) -> bool:
    """Whether ``name`` (a file name or a path in the bundle) is a Universal CRT
    DLL, which Windows 10 and later provide: PyInstaller takes them from the
    build machine's ``PATH`` when it finds them there, so the bundle leaves them
    out (ADR 0003)."""
    base = PureWindowsPath(name).name.lower()
    return base == "ucrtbase.dll" or (base.startswith("api-ms-win-") and base.endswith(".dll"))


def numeric_version(version: str) -> tuple[int, int, int, int]:
    """The four numbers of a Windows version resource for a Python version:
    its release part, padded with zeros (``0.1.0.dev0`` gives ``(0, 1, 0, 0)``);
    a pre-, post- or development release is not numbered. ``ValueError`` for a
    version that does not start with a number, or a part above 65535."""
    match = re.match(r"^v?(\d+(?:\.\d+)*)", version.strip())
    if match is None:
        raise ValueError(f"{version!r} does not start with a release number")
    parts = [int(part) for part in match.group(1).split(".")][:4]
    if any(part > 0xFFFF for part in parts):
        raise ValueError(f"{version!r}: a part is above 65535")
    padded = (parts + [0, 0, 0, 0])[:4]
    return (padded[0], padded[1], padded[2], padded[3])


def installer_name(version: str) -> str:
    """The installer's file name, without ``.exe``: Inno Setup's
    ``OutputBaseFilename``."""
    if not re.fullmatch(r"[0-9A-Za-z.+-]+", version):
        raise ValueError(f"{version!r} cannot be part of a file name")
    return f"{APP_NAME}-{version}-setup"


def normalize(name: str) -> str:
    """A distribution name as PEP 503 compares them."""
    return re.sub(r"[-_.]+", "-", name).lower()


def top_level(path: str) -> str:
    """The top-level import name a path in the bundle (``numpy/_core/x.pyd``) or
    a module name (``numpy._core``) belongs to: its first part, and for a
    file at the top the name before its first dot (``_ssl.pyd`` gives
    ``_ssl``). A vendored-library folder (``numpy.libs``) belongs to its
    package."""
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if len(parts) > 1:  # a path in a folder
        first = parts[0]
        return first[: -len(".libs")] if first.endswith(".libs") else first
    return parts[0].split(".")[0] if parts else ""


def bundled_distributions(
    modules: Iterable[str],
    files: Iterable[str],
    packages: Mapping[str, Sequence[str]],
    stdlib: Iterable[str],
) -> tuple[dict[str, set[str]], set[str]]:
    """Which installed distributions the bundle takes code or data from.

    ``modules`` are the bundled modules' names and ``files`` the paths of the
    other bundled files (binaries and data), relative to the bundle's folder;
    ``packages`` maps a top-level import name to the distributions that
    provide it (``importlib.metadata.packages_distributions()``), and
    ``stdlib`` names the standard library's modules. Returns each distribution
    with the top-level names it provides, and the top-level names nothing
    accounts for (neither a distribution, the standard library, the Python or
    Visual C++ runtime, PyInstaller's own modules, nor metadata). The build
    stops on the latter: every distribution in the bundle needs its metadata
    and its license texts in the notices.
    """
    stdlib = set(stdlib)
    names: set[str] = {top_level(name) for name in modules}
    for path in files:
        first = PurePosixPath(path.replace("\\", "/")).parts[0] if path else ""
        if first.endswith((".dist-info", ".egg-info")):
            continue
        if PYTHON_RUNTIME_FILES.match(first) or VC_RUNTIME_FILES.match(first):
            continue
        names.add(top_level(path))
    found: dict[str, set[str]] = {}
    unknown: set[str] = set()
    for name in sorted(names - {""}):
        if name in packages:
            for dist in packages[name]:
                found.setdefault(dist, set()).add(name)
        elif name not in stdlib and not PYINSTALLER_MODULES.match(name):
            unknown.add(name)
    return found, unknown


def missing_record_distributions(bundled: Iterable[str]) -> list[str]:
    """The distributions the record reads that the bundle does not hold."""
    have = {normalize(name) for name in bundled}
    return [name for name in RECORD_DISTRIBUTIONS if normalize(name) not in have]


def forbidden_paths(paths: Iterable[str]) -> list[str]:
    """The paths (relative, either separator) with a part :data:`FORBIDDEN_NAMES`
    matches."""
    return sorted(
        path
        for path in paths
        if any(FORBIDDEN_NAMES.match(part) for part in PurePosixPath(path.replace("\\", "/")).parts)
    )


def iss_app_id(iss: str) -> str:
    """The application ID ``proteia.iss`` (its text) fixes, without braces."""
    match = re.search(r"^AppId=\{\{([0-9A-Fa-f-]{36})\}\s*$", iss, re.MULTILINE)
    if match is None:
        raise ValueError("proteia.iss fixes no AppId")
    return match.group(1)
