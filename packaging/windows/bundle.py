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
import enum
import json
import os
import posixpath
import re
import struct
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

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
# Of those, the ones the Python runtime's Windows build includes, in its own
# folder, which its LICENSE.txt covers (notices.py, section 1). Besides these the
# bundle carries only packages' own copies, from their folders: Microsoft lets
# only Visual Studio licensees redistribute the files, so the spec leaves out the
# copies PyInstaller finds elsewhere (System32's MSVCP140.dll, #54).
PYTHON_VC_RUNTIME: Final = frozenset({"vcruntime140.dll", "vcruntime140_1.dll"})
# The bundle's folder next to Proteia.exe, which PyInstaller puts everything in.
CONTENTS: Final = "_internal"
# The suffixes of the files whose import tables the build reads.
BINARY_SUFFIXES: Final = (".pyd", ".dll", ".exe")
# The bundle manifest's key for the spec's classification of the Visual C++
# runtime files (:func:`read_vc_runtime`).
VC_RUNTIME_KEY: Final = "vc_runtime"
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


def is_vc_runtime_copy(name: str) -> bool:
    """Whether ``name`` (a file name or a path, either separator) is a Visual C++
    runtime DLL under Microsoft's name or a package's own name for its copy
    (:data:`VC_RUNTIME_COPIES`)."""
    return VC_RUNTIME_COPIES.match(PureWindowsPath(name).name) is not None


class VcKind(enum.StrEnum):
    """Where a Visual C++ runtime file PyInstaller collected came from: the
    spec's classification (:func:`classify_vc_runtime`), which the bundle check
    and the notices re-check (#54)."""

    PYTHON = "python"  # the Python build's own copy, from its folder
    WHEEL = "wheel"  # a package's own copy, from its folder in site-packages
    ELSEWHERE = "elsewhere"  # found anywhere else (System32, say): left out


# The kinds whose copies the bundle carries.
CARRIED_KINDS: Final = frozenset({VcKind.PYTHON, VcKind.WHEEL})


@dataclass(frozen=True)
class VcRuntime:
    """The spec's classification of one Visual C++ runtime file."""

    path: str  # its path in the bundle folder: _internal/numpy.libs/msvcp140-<hash>.dll
    kind: VcKind
    folder: str | None = None  # WHEEL: the package's folder in site-packages (numpy.libs)
    source: str | None = None  # ELSEWHERE: where PyInstaller found it

    def to_json(self) -> dict[str, str]:
        entry = {"path": self.path, "kind": str(self.kind)}
        if self.folder is not None:
            entry["folder"] = self.folder
        if self.source is not None:
            entry["source"] = self.source
        return entry

    @classmethod
    def from_json(cls, entry: Mapping[str, Any]) -> VcRuntime:
        """The entry :meth:`to_json` wrote; ``ValueError`` for an unknown kind."""
        return cls(entry["path"], VcKind(entry["kind"]), entry.get("folder"), entry.get("source"))


def read_vc_runtime(manifest: Mapping[str, Any]) -> list[VcRuntime]:
    """The classification the bundle ``manifest`` records (none when it records
    none, so that every Visual C++ runtime file in the bundle is unclassified)."""
    return [VcRuntime.from_json(entry) for entry in manifest.get(VC_RUNTIME_KEY, [])]


# Turns a path on the build machine into the one it stands for, links followed:
# os.path.realpath, which on Windows also resolves a substituted drive and a
# short (8.3) name.
Resolve = Callable[[str], str]


def _resolved(path: str, resolve: Resolve) -> PureWindowsPath:
    return PureWindowsPath(resolve(str(path)))


def package_folder(
    source: str, package_roots: Iterable[str], resolve: Resolve = os.path.realpath
) -> str | None:
    """The folder of an installed package that ``source`` (a file's path on the
    build machine) lies in, if any: its first part below one of ``package_roots``
    (the build environment's site-packages folders), such as ``numpy.libs``. A
    file directly in a root is no package's. Decided on resolved paths, so a
    junction, a substituted drive or a short name gives the same answer."""
    path = _resolved(source, resolve)
    for root in package_roots:
        root_path = _resolved(root, resolve)
        if path.is_relative_to(root_path):
            parts = path.relative_to(root_path).parts
            if len(parts) > 1:
                return parts[0]
    return None


def classify_vc_runtime(
    name: str,
    source: str,
    package_roots: Iterable[str],
    python_dir: str,
    resolve: Resolve = os.path.realpath,
) -> VcRuntime:
    """The kind of the Visual C++ runtime file PyInstaller would put at ``name``
    (in :data:`CONTENTS`) from ``source``: a package's own copy when ``source``
    lies in a package's folder (:func:`package_folder`); the Python build's when
    it is ``vcruntime140.dll`` or ``vcruntime140_1.dll`` directly in
    ``python_dir``, the Python installation's folder (``sys.base_prefix``), both
    resolved; otherwise from elsewhere, such as System32, which the bundle does
    not carry (#54)."""
    path = f"{CONTENTS}/{PureWindowsPath(name).as_posix()}"
    folder = package_folder(source, package_roots, resolve)
    if folder is not None:
        return VcRuntime(path, VcKind.WHEEL, folder=folder)
    if PureWindowsPath(name).name.lower() in PYTHON_VC_RUNTIME and _resolved(
        source, resolve
    ).parent == _resolved(python_dir, resolve):
        return VcRuntime(path, VcKind.PYTHON)
    return VcRuntime(path, VcKind.ELSEWHERE, source=str(source))


def leave_out(
    binaries: Iterable[tuple[str, str, str]],
    package_roots: Iterable[str],
    python_dir: str,
    resolve: Resolve = os.path.realpath,
) -> tuple[list[tuple[str, str, str]], dict[str, list[str]], list[VcRuntime]]:
    """PyInstaller's ``binaries`` (``(name in the bundle, source, kind)``) without
    the ones the bundle leaves out; the names of those, by why: the Universal CRT
    DLLs (:func:`is_universal_crt`), and the Visual C++ runtime files from
    elsewhere than the Python build or a package (such as System32's
    ``MSVCP140.dll``, #54); and the classification of every Visual C++ runtime
    file (:func:`classify_vc_runtime`), kept or not, by path."""
    roots = list(package_roots)
    kept: list[tuple[str, str, str]] = []
    left_out: dict[str, list[str]] = {"universal_crt": [], "vc_runtime": []}
    classified: list[VcRuntime] = []
    for entry in binaries:
        name, source = entry[0], entry[1]
        if is_universal_crt(name):
            left_out["universal_crt"].append(name)
            continue
        if is_vc_runtime_copy(name):
            vc = classify_vc_runtime(name, source, roots, python_dir, resolve)
            classified.append(vc)
            if vc.kind not in CARRIED_KINDS:
                left_out["vc_runtime"].append(name)
                continue
        kept.append(entry)
    left_out = {reason: sorted(names) for reason, names in left_out.items()}
    return kept, left_out, sorted(classified, key=lambda vc: vc.path.lower())


def dll_search_folders(binary: str) -> list[str]:
    """The folders of the bundle (paths in it) where Windows looks for the DLLs
    the binary at ``binary`` (its path in the bundle folder) imports by name:
    :data:`CONTENTS`, which the bootloader adds (``SetDllDirectoryW``); the
    binary's own folder, which Python adds when it loads an extension module
    (``LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR``); and, for a binary of a package, the
    package's ``<name>.libs`` folder, which delvewheel's patch to the package's
    ``__init__.py`` adds with ``os.add_dll_directory`` (NumPy's and SciPy's). A
    DLL elsewhere in the bundle is found only if something else loaded it first."""
    own = posixpath.dirname(binary) or "."
    folders = [CONTENTS, own]
    parts = PurePosixPath(binary).parts
    if len(parts) > 2 and parts[0] == CONTENTS:
        folders.append(f"{CONTENTS}/{top_level('/'.join(parts[1:]))}.libs")
    return list(dict.fromkeys(folders))


def vc_runtime_problems(
    imports: Mapping[str, Sequence[str]], classified: Iterable[VcRuntime]
) -> list[str]:
    """What is wrong with the Visual C++ runtime in the bundle: ``imports`` maps
    each of its binaries (its path in the bundle folder) to the DLL names it
    imports (:func:`bundle_imports`), and ``classified`` is the spec's
    classification (:func:`read_vc_runtime`). Each Visual C++ runtime file among
    the binaries must be one the spec classified as the Python build's or a
    package's own copy; and each such DLL a binary imports by name must be one of
    those in a folder Windows looks in for that binary (:func:`dll_search_folders`).
    Otherwise Windows looks for it in System32, where only the Visual C++
    Redistributable puts Microsoft's names (#54)."""
    by_path = {vc.path.lower(): vc for vc in classified}
    problems = []
    carried = []
    for path in sorted(imports):
        if not is_vc_runtime_copy(path):
            continue
        vc = by_path.get(path.lower())
        if vc is None:
            problems.append(
                f"{path}: a Visual C++ runtime file the spec's classification (the bundle"
                " manifest) does not list, so nothing says where it came from; the bundle"
                " carries only the Python build's copies and packages' own (#54)"
            )
        elif vc.kind not in CARRIED_KINDS:
            problems.append(
                f"{path}: a Visual C++ runtime file from {vc.source}, which is neither the"
                " Python build's folder nor a package's; the bundle leaves such files out (#54)"
            )
        else:
            carried.append(PurePosixPath(path.lower()))
    for binary in sorted(imports):
        folders = dll_search_folders(binary)
        searched = {folder.lower() for folder in folders}
        provided = {path.name for path in carried if str(path.parent) in searched}
        for dll in sorted(set(imports[binary]), key=str.lower):
            if not is_vc_runtime_copy(dll) or dll.lower() in provided:
                continue
            why = (
                "it loads only where the Visual C++ Redistributable is installed"
                if is_vc_runtime(dll)
                else "no Windows provides it"
            )
            problems.append(
                f"{binary} imports {dll}, a Visual C++ runtime file the bundle does not carry"
                f" where Windows looks for it ({', '.join(folders)}): {why}; leave the module"
                " out (bundle.py) or bundle a copy its package provides (#54)"
            )
    return problems


class NotPE(Exception):
    """A file named like a binary whose import tables cannot be read: not a PE
    file, or a damaged one."""


_PE_SIGNATURE: Final = b"PE" + bytes(2)


def pe_header_problem(path: Path) -> str | None:
    """Why the file at ``path`` is not a PE file whose import tables can be read,
    as far as its headers tell (the standard library only): None when it starts
    with a DOS header pointing at a PE signature, and its section table and every
    section it lists lie within the file. pefile reads a file cut short without
    an error, and finds no imports in it."""
    size = path.stat().st_size
    if size == 0:
        return "the file is empty"
    with path.open("rb") as f:
        dos = f.read(64)
        if len(dos) < 64 or dos[:2] != b"MZ":
            return "it does not start with a DOS header"
        (offset,) = struct.unpack_from("<I", dos, 0x3C)
        f.seek(offset)
        coff = f.read(24)
        if coff[:4] != _PE_SIGNATURE:
            return "there is no PE signature where its DOS header points"
        if len(coff) < 24:
            return "its PE header runs past the end of the file"
        (count,) = struct.unpack_from("<H", coff, 6)
        (optional,) = struct.unpack_from("<H", coff, 20)
        f.seek(offset + 24 + optional)
        table = f.read(40 * count)
    if len(table) < 40 * count:
        return "its section table runs past the end of the file"
    for start in range(0, 40 * count, 40):
        raw_size, raw_offset = struct.unpack_from("<II", table, start + 16)
        if raw_size and raw_offset + raw_size > size:
            section = table[start : start + 8].rstrip(bytes(1)).decode("ascii", "replace")
            return f"its section {section!r} runs past the end of the file, which is cut short"
    return None


def pe_imports(path: Path) -> list[str]:
    """The DLL names the PE file ``path`` imports, at load time or delay-loaded,
    as its import tables spell them. Needs pefile (the build environment's).
    ``NotPE`` when its headers say it is no PE file (:func:`pe_header_problem`),
    pefile refuses it, or an import table its headers declare cannot be read."""
    problem = pe_header_problem(path)
    if problem is not None:
        raise NotPE(problem)
    import pefile

    entry = pefile.DIRECTORY_ENTRY
    tables = {  # data directory: pefile's attribute, the table's name
        entry["IMAGE_DIRECTORY_ENTRY_IMPORT"]: ("DIRECTORY_ENTRY_IMPORT", "import table"),
        entry["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]: (
            "DIRECTORY_ENTRY_DELAY_IMPORT",
            "delay-import table",
        ),
    }
    try:
        pe = pefile.PE(str(path), fast_load=True)
    except pefile.PEFormatError as exc:
        raise NotPE(f"pefile cannot read it: {exc}") from exc
    try:
        try:
            pe.parse_data_directories(directories=list(tables), import_dllnames_only=True)
        except pefile.PEFormatError as exc:
            raise NotPE(f"pefile cannot read its import tables: {exc}") from exc
        declared = pe.OPTIONAL_HEADER.DATA_DIRECTORY
        names = set()
        for index, (attribute, table) in tables.items():
            entries = getattr(pe, attribute, None)
            if not entries and index < len(declared) and declared[index].VirtualAddress:
                raise NotPE(f"pefile finds no entry in the {table} its headers declare")
            names |= {item.dll.decode("ascii", "replace") for item in entries or []}
        return sorted(names, key=str.lower)
    finally:
        pe.close()


def bundle_imports(folder: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Every file in the bundle ``folder`` named like a binary (``.pyd``, ``.dll``
    or ``.exe``, in any case), by its path in it: the DLL names each PE file
    imports, and why each other one is not a PE file (:class:`NotPE`)."""
    imports: dict[str, list[str]] = {}
    not_pe: dict[str, str] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in BINARY_SUFFIXES:
            name = path.relative_to(folder).as_posix()
            try:
                imports[name] = pe_imports(path)
            except NotPE as exc:
                not_pe[name] = str(exc)
    return imports, not_pe


def write_imports(
    out: Path, imports: Mapping[str, Sequence[str]], not_pe: Mapping[str, str]
) -> None:
    """Write :func:`bundle_imports`'s answer to ``out`` as JSON."""
    data = {"binaries": dict(imports), "not_pe": dict(not_pe)}
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")


def read_imports(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """What :func:`write_imports` wrote to ``path``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["binaries"], data["not_pe"]


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
    bundle ``FOLDER`` into the JSON file ``OUT`` (:func:`write_imports`), files
    that are not PE files included; build.py runs it with the build environment's
    Python, which has pefile, and reads it with :func:`read_imports`."""
    parser = argparse.ArgumentParser(description="The DLLs a bundle's binaries import.")
    parser.add_argument("command", choices=["imports"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    imports, not_pe = bundle_imports(args.folder)
    write_imports(args.out, imports, not_pe)
    others = f"; {len(not_pe)} files named like binaries are not PE files" if not_pe else ""
    print(f"bundle: the imports of {len(imports)} binaries in {args.out}{others}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
