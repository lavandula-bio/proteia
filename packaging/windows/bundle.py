# SPDX-License-Identifier: Apache-2.0
"""What Proteia's Windows bundle holds (ADR 0003): the lists the PyInstaller
spec (``proteia.spec``) builds from, and the rules the spec, the build
(``build.py``), the notices (``notices.py``) and the tests share.

The standard library only, so the tests run it where PyInstaller is not
installed; reading import tables (:func:`pe_imports`) needs pefile, which
PyInstaller requires on Windows, so the build runs ``python bundle.py imports``
with the build environment's Python.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
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
# The extension modules that import MSVCP140.dll by that name (the import tables
# of the scikit-image 0.26.0 wheel), which the bundle does not carry (#54).
# Nothing the app runs loads them: scikit-image loads skimage.feature's modules
# only when used, and _remap only inside skimage.util.map_array, which Proteia
# does not call. Left out, a later use fails the same way on every machine
# (ModuleNotFoundError in the frozen app, which the self-test reports), not only
# on a Windows without the Visual C++ Redistributable; the build fails if a
# bundled module imports a Visual C++ runtime file the bundle lacks (build.py).
MSVCP140_IMPORTERS: Final = (
    "skimage.feature._cascade",
    "skimage.feature._haar",
    "skimage.feature._hessian_det_appx",
    "skimage.feature.brief_cy",
    "skimage.util._remap",
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
# The Visual C++ runtime (Visual Studio 2015 and later) under Microsoft's file
# names: what the Visual C++ Redistributable installs into System32, and what
# PyInstaller collects from there for a module that imports it (the names in its
# depend/dylib.py). Every rule about these files starts from this list.
_VC_RUNTIME_NAMES: Final = (
    r"concrt140|msvcp140(_1|_2|_atomic_wait|_codecvt_ids)?|vccorlib140"
    r"|vcruntime140(_1|_threads)?|vcamp140|vcomp140"
)
VC_RUNTIME_FILES: Final = re.compile(rf"^({_VC_RUNTIME_NAMES})\.dll$", re.IGNORECASE)
# The same files under Microsoft's names or under one a package's wheel gave its
# own copy (numpy.libs/msvcp140-<hash>.dll, as delvewheel names them).
VC_RUNTIME_COPIES: Final = re.compile(rf"^({_VC_RUNTIME_NAMES})([-.][^\\/]*)?\.dll$", re.IGNORECASE)
# Of those, the ones the Python runtime's Windows build includes, which its
# LICENSE.txt covers (notices.py, section 1). Besides these the bundle carries
# only packages' own copies, from their folders: Microsoft lets only Visual
# Studio licensees redistribute the files, so the spec leaves out the copies
# PyInstaller finds elsewhere (System32's MSVCP140.dll, #54).
PYTHON_VC_RUNTIME: Final = frozenset({"vcruntime140.dll", "vcruntime140_1.dll"})
# The bundle's folder next to Proteia.exe, which PyInstaller puts everything in.
CONTENTS: Final = "_internal"
# PyInstaller's own modules in the bundle (its runtime hooks and their helpers).
PYINSTALLER_MODULES: Final = re.compile(r"^(pyi_rth_\w+|_pyi_rth_utils|pyimod\d\d_\w+)$")


def is_universal_crt(name: str) -> bool:
    """Whether ``name`` (a file name or a path in the bundle) is a Universal CRT
    DLL, which Windows 10 and later provide: PyInstaller takes them from the
    build machine's ``PATH`` when it finds them there, so the bundle leaves them
    out (ADR 0003)."""
    base = PureWindowsPath(name).name.lower()
    return base == "ucrtbase.dll" or (base.startswith("api-ms-win-") and base.endswith(".dll"))


def is_vc_runtime(name: str) -> bool:
    """Whether ``name`` (a file name or a path, either separator) is a Visual C++
    runtime DLL under Microsoft's name (:data:`VC_RUNTIME_FILES`)."""
    return VC_RUNTIME_FILES.match(PureWindowsPath(name).name) is not None


def _is_python_vc_runtime(name: str) -> bool:
    return PureWindowsPath(name).name.lower() in PYTHON_VC_RUNTIME


def from_package_folder(source: str, package_roots: Iterable[str]) -> bool:
    """Whether ``source`` (a file's path on the build machine) lies in a folder
    inside one of ``package_roots`` (the build environment's site-packages
    folders): a file of an installed package, not one PyInstaller found elsewhere
    (System32, the Python installation)."""
    path = PureWindowsPath(source)
    for root in map(PureWindowsPath, package_roots):
        if path.is_relative_to(root) and len(path.relative_to(root).parts) > 1:
            return True
    return False


def leave_out(
    binaries: Iterable[tuple[str, str, str]], package_roots: Iterable[str]
) -> tuple[list[tuple[str, str, str]], dict[str, list[str]]]:
    """PyInstaller's ``binaries`` (``(name in the bundle, source, kind)``) without
    the ones the bundle leaves out, and the names of those, by why: the Universal
    CRT DLLs (:func:`is_universal_crt`), and the Visual C++ runtime files that are
    neither the Python build's nor a package's own (:func:`from_package_folder`
    with ``package_roots``), such as System32's ``MSVCP140.dll`` (#54)."""
    roots = list(package_roots)
    kept: list[tuple[str, str, str]] = []
    left_out: dict[str, list[str]] = {"universal_crt": [], "vc_runtime": []}
    for entry in binaries:
        name, source = entry[0], entry[1]
        if is_universal_crt(name):
            left_out["universal_crt"].append(name)
        elif (
            is_vc_runtime(name)
            and not _is_python_vc_runtime(name)
            and not from_package_folder(source, roots)
        ):
            left_out["vc_runtime"].append(name)
        else:
            kept.append(entry)
    return kept, {reason: sorted(names) for reason, names in left_out.items()}


def _at_top(path: str) -> bool:
    """Whether ``path`` (in the finished bundle folder) is directly in it or in
    its :data:`CONTENTS` folder, where PyInstaller puts what no package holds."""
    parts = PurePosixPath(path.replace("\\", "/")).parts
    return len(parts) == 1 or (len(parts) == 2 and parts[0] == CONTENTS)


def is_left_out_vc_runtime(path: str) -> bool:
    """Whether ``path`` (a file's path in the finished bundle folder) is a Visual
    C++ runtime file the bundle leaves out: one at its top (:func:`_at_top`),
    which no package provides, other than the Python build's."""
    return _at_top(path) and is_vc_runtime(path) and not _is_python_vc_runtime(path)


def vc_import_problems(imports: Mapping[str, Iterable[str]], files: Iterable[str]) -> list[str]:
    """What is wrong with the DLLs the bundled binaries import: ``imports`` maps
    each binary (its path in the bundle folder) to the DLL names it imports
    (:func:`pe_imports`), and ``files`` are the paths of every file in the
    bundle. A Visual C++ runtime DLL imported by name must be one the bundle
    provides: a copy it holds and does not leave out (:func:`is_left_out_vc_runtime`).
    Otherwise Windows loads it from System32, where only the Visual C++
    Redistributable puts it (#54)."""
    provided = {
        PureWindowsPath(path).name.lower()
        for path in files
        if is_vc_runtime(path) and not is_left_out_vc_runtime(path)
    }
    return [
        f"{binary} imports {dll}, a Visual C++ runtime file the bundle does not carry: it"
        " loads only where the Visual C++ Redistributable is installed; leave the module out"
        " (bundle.py) or bundle a copy its package provides (#54)"
        for binary in sorted(imports)
        for dll in sorted(set(imports[binary]), key=str.lower)
        if is_vc_runtime(dll) and dll.lower() not in provided
    ]


def pe_imports(path: Path) -> list[str]:
    """The DLL names the PE file ``path`` imports, at load time or delay-loaded,
    as its import tables spell them. Needs pefile (the build environment's)."""
    import pefile

    pe = pefile.PE(str(path), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
            ],
            import_dllnames_only=True,
        )
        entries = [
            *getattr(pe, "DIRECTORY_ENTRY_IMPORT", []),
            *getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []),
        ]
        return sorted({entry.dll.decode("ascii", "replace") for entry in entries}, key=str.lower)
    finally:
        pe.close()


def bundle_imports(folder: Path) -> dict[str, list[str]]:
    """Every PE file in the bundle ``folder`` (``.pyd``, ``.dll`` or ``.exe``, in
    any case), by its path in it, with the DLL names it imports."""
    return {
        path.relative_to(folder).as_posix(): pe_imports(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in (".pyd", ".dll", ".exe")
    }


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


def main(argv: Sequence[str] | None = None) -> int:
    """``python bundle.py imports FOLDER OUT``: :func:`bundle_imports` of the
    bundle ``FOLDER`` into the JSON file ``OUT``; build.py runs it with the build
    environment's Python, which has pefile."""
    parser = argparse.ArgumentParser(description="The DLLs a bundle's binaries import.")
    parser.add_argument("command", choices=["imports"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    imports = bundle_imports(args.folder)
    args.out.write_text(json.dumps(imports, indent=1), encoding="utf-8")
    print(f"bundle: the imports of {len(imports)} binaries in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
