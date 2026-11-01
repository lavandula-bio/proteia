# SPDX-License-Identifier: Apache-2.0
"""The Windows installer's build files (#54, ADR 0003), in the parts that run
without PyInstaller or Inno Setup: the spec's lists against the code they
describe, the installer script against the launcher, the build's version and
bundle checks, the notices generator on a stand-in bundle, and the launch
check's stand-in browser. A test that needs PyInstaller (the ``build`` group)
skips without it."""

import ast
import importlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tomllib
import types
import webbrowser
from pathlib import Path

import pytest

import proteia
from proteia import selftest
from proteia.core import record
from proteia.web import launch, logs, server

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "packaging" / "windows"
PACKAGE = ROOT / "src" / "proteia"
ISS = (WINDOWS / "proteia.iss").read_text(encoding="utf-8")


def _load(*names: str) -> list:
    """The build scripts, imported as build.py imports its neighbours."""
    sys.path.insert(0, str(WINDOWS))
    try:
        return [importlib.import_module(name) for name in names]
    finally:
        sys.path.remove(str(WINDOWS))


bundle, notices, build, smoke = _load("bundle", "notices", "build", "smoke")


# --- The spec's lists ---


def test_the_lists_name_what_the_code_uses():
    assert bundle.RECORD_DISTRIBUTIONS == record._DISTRIBUTIONS
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["proteia"] == bundle.ENTRY_POINT
    module, _, function = bundle.ENTRY_POINT.partition(":")
    entry = (WINDOWS / "proteia_entry.py").read_text(encoding="utf-8")
    assert f"from {module} import {function}" in entry
    assert bundle.OWN_DISTRIBUTION == pyproject["project"]["name"]


def test_the_package_data_takes_every_file_of_the_web_client():
    static = PACKAGE / "web" / "static"
    wanted = {
        path for path in static.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    }
    taken = {path for pattern in bundle.PACKAGE_DATA for path in PACKAGE.glob(pattern)}
    assert wanted and wanted <= taken


def _imported_top_levels(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def test_nothing_the_app_imports_is_left_out():
    excluded = set(bundle.EXCLUDES) | set(bundle.MSVCP140_IMPORTERS)
    for path in PACKAGE.rglob("*.py"):
        for name in _imported_top_levels(path):
            parts = name.split(".")
            prefixes = {".".join(parts[: n + 1]) for n in range(len(parts))}
            assert not prefixes & excluded, (
                f"{path.name} imports {name}, which the bundle leaves out"
            )


def test_napari_qt_and_tkinter_are_left_out():
    assert set(selftest.LEFT_OUT) <= set(bundle.EXCLUDES)
    assert {"setuptools", "pkg_resources"} <= set(bundle.EXCLUDES)  # build tools only


@pytest.mark.parametrize(
    ("name", "crt"),
    [
        ("api-ms-win-crt-runtime-l1-1-0.dll", True),
        ("API-MS-WIN-CORE-FILE-L1-2-0.DLL", True),
        ("ucrtbase.dll", True),
        ("sub\\ucrtbase.dll", True),
        ("VCRUNTIME140.dll", False),
        ("MSVCP140.dll", False),
        ("numpy.libs/msvcp140-abc.dll", False),
        ("api-ms-win-crt-runtime-l1-1-0.pyd", False),
    ],
)
def test_universal_crt_dlls_are_recognized(name, crt):
    assert bundle.is_universal_crt(name) is crt


@pytest.mark.parametrize(
    ("name", "vc"),
    [
        # The Visual C++ runtime under Microsoft's names, as PyInstaller would copy
        # it from the build machine's System32 (#54).
        ("MSVCP140.dll", True),
        ("msvcp140.DLL", True),
        ("sub\\MSVCP140.dll", True),
        ("sub/msvcp140.dll", True),
        ("MSVCP140_1.dll", True),
        ("msvcp140_2.dll", True),
        ("msvcp140_atomic_wait.dll", True),
        ("msvcp140_codecvt_ids.dll", True),
        ("concrt140.dll", True),
        ("vccorlib140.dll", True),
        ("vcomp140.dll", True),
        ("vcamp140.dll", True),
        ("vcruntime140_threads.dll", True),
        ("VCRUNTIME140.dll", True),
        ("vcruntime140_1.dll", True),
        # A package's renamed copy is not under Microsoft's name.
        ("numpy.libs/msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll", False),
        # Windows' own C++ library and the .NET Framework's copies are not these.
        ("msvcp_win.dll", False),
        ("msvcp140_clr0400.dll", False),
        ("msvcp140.pyd", False),
        ("ucrtbase.dll", False),
    ],
)
def test_the_vc_runtime_is_recognized_by_microsofts_names(name, vc):
    assert bundle.is_vc_runtime(name) is vc


def test_a_packages_renamed_copy_of_the_vc_runtime_is_recognized_too():
    for name in (
        "msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll",
        "VCOMP140-0123abcd.DLL",
        "concrt140.dll",
        "msvcp140_atomic_wait.dll",
    ):
        assert bundle.VC_RUNTIME_COPIES.match(name), name
        assert bundle.is_vc_runtime_copy(f"_internal\\pkg.libs\\{name}"), name
    for name in ("msvcp140_clr0400.dll", "msvcp_win.dll", "vcruntime1400.dll", "msvcp140.pyd"):
        assert not bundle.VC_RUNTIME_COPIES.match(name), name
        assert not bundle.is_vc_runtime_copy(f"_internal/pkg.libs/{name}"), name


_SITE = r"C:\b\env\Lib\site-packages"
_PY = r"C:\py"


def _as_is(path: str) -> str:
    """The resolver for made-up paths, which name nothing on this machine."""
    return path


def _vc(path: str, kind: str, **more: str):
    """A classified Visual C++ runtime file, as the bundle manifest records it."""
    return bundle.VcRuntime.from_json({"path": path, "kind": kind, **more})


def test_the_spec_classifies_each_vc_runtime_file_by_where_it_came_from():
    entries = [
        ("python313.dll", rf"{_PY}\python313.dll", "BINARY"),
        # the Python build's own copies, from its folder, stay
        ("VCRUNTIME140.dll", rf"{_PY}\vcruntime140.dll", "BINARY"),
        ("vcruntime140_1.dll", rf"{_PY}\VCRUNTIME140_1.dll", "BINARY"),
        # what PyInstaller found in System32 or a Windows Kits folder goes
        ("MSVCP140.dll", r"C:\Windows\System32\msvcp140.dll", "BINARY"),
        ("concrt140.dll", r"C:\Windows\System32\concrt140.dll", "BINARY"),
        ("api-ms-win-crt-heap-l1-1-0.dll", r"C:\Kits\api-ms-win-crt-heap-l1-1-0.dll", "BINARY"),
        ("ucrtbase.dll", r"C:\Kits\ucrtbase.dll", "BINARY"),
        # a package's own copies, renamed or not, in its folders, stay (any case)
        ("numpy.libs\\msvcp140-abc.dll", rf"{_SITE}\numpy.libs\msvcp140-abc.dll", "BINARY"),
        ("pkg.libs\\vcomp140.dll", rf"{_SITE.upper()}\pkg.libs\vcomp140.dll", "BINARY"),
        ("pkg\\sub\\MSVCP140_1.dll", r"D:\plat\pkg\sub\MSVCP140_1.dll", "BINARY"),
        # in site-packages itself, or in a folder beside it, is no package's folder
        ("vcomp140.dll", rf"{_SITE}\vcomp140.dll", "BINARY"),
        ("vccorlib140.dll", rf"{_SITE}-old\pkg\vccorlib140.dll", "BINARY"),
        # a renamed copy found outside every package's folder goes too
        ("msvcp140-def.dll", r"C:\other\numpy.libs\msvcp140-def.dll", "BINARY"),
    ]
    kept, left_out, classified = bundle.leave_out(entries, [_SITE, r"D:\plat"], _PY, _as_is)
    assert [entry[0] for entry in kept] == [
        "python313.dll",
        "VCRUNTIME140.dll",
        "vcruntime140_1.dll",
        "numpy.libs\\msvcp140-abc.dll",
        "pkg.libs\\vcomp140.dll",
        "pkg\\sub\\MSVCP140_1.dll",
    ]
    assert left_out == {
        "universal_crt": ["api-ms-win-crt-heap-l1-1-0.dll", "ucrtbase.dll"],
        "vc_runtime": [
            "MSVCP140.dll",
            "concrt140.dll",
            "msvcp140-def.dll",
            "vccorlib140.dll",
            "vcomp140.dll",
        ],
    }
    # Every Visual C++ runtime file, kept or not, by its path in the bundle: what
    # the manifest records and the bundle check and the notices re-check.
    assert [vc.to_json() for vc in classified] == [
        {
            "path": "_internal/concrt140.dll",
            "kind": "elsewhere",
            "source": r"C:\Windows\System32\concrt140.dll",
        },
        {
            "path": "_internal/msvcp140-def.dll",
            "kind": "elsewhere",
            "source": r"C:\other\numpy.libs\msvcp140-def.dll",
        },
        {
            "path": "_internal/MSVCP140.dll",
            "kind": "elsewhere",
            "source": r"C:\Windows\System32\msvcp140.dll",
        },
        {
            "path": "_internal/numpy.libs/msvcp140-abc.dll",
            "kind": "wheel",
            "folder": "numpy.libs",
        },
        {"path": "_internal/pkg.libs/vcomp140.dll", "kind": "wheel", "folder": "pkg.libs"},
        {"path": "_internal/pkg/sub/MSVCP140_1.dll", "kind": "wheel", "folder": "pkg"},
        {
            "path": "_internal/vccorlib140.dll",
            "kind": "elsewhere",
            "source": r"C:\b\env\Lib\site-packages-old\pkg\vccorlib140.dll",
        },
        {
            "path": "_internal/vcomp140.dll",
            "kind": "elsewhere",
            "source": r"C:\b\env\Lib\site-packages\vcomp140.dll",
        },
        {"path": "_internal/VCRUNTIME140.dll", "kind": "python"},
        {"path": "_internal/vcruntime140_1.dll", "kind": "python"},
    ]
    # With no package folder given, every Visual C++ runtime file but Python's goes.
    kept, left_out, classified = bundle.leave_out(entries, [], _PY, _as_is)
    assert len(left_out["vc_runtime"]) == 8
    assert [vc.kind for vc in classified].count("python") == 2
    spec = (WINDOWS / "proteia.spec").read_text(encoding="utf-8")
    assert (
        "a.binaries, left_out, vc_runtime = bundle.leave_out(a.binaries, package_roots,"
        " sys.base_prefix)"
    ) in spec.replace("\n    ", " ")
    assert "bundle.VC_RUNTIME_KEY: [vc.to_json() for vc in vc_runtime]," in spec


def test_the_python_builds_two_files_are_kept_only_from_its_folder():
    def kind(name: str, source: str) -> str:
        return bundle.classify_vc_runtime(name, source, [_SITE], _PY, _as_is).kind

    assert kind("VCRUNTIME140.dll", rf"{_PY}\vcruntime140.dll") == "python"
    assert kind("vcruntime140_1.dll", r"c:\PY\VCRUNTIME140_1.DLL") == "python"
    # The same names from System32, another program's folder, or a folder below
    # the Python build's are not its copies (#239).
    for source in (
        r"C:\Windows\System32\vcruntime140.dll",
        r"C:\Program Files\App\vcruntime140.dll",
        rf"{_PY}\DLLs\vcruntime140.dll",
    ):
        assert kind("VCRUNTIME140.dll", source) == "elsewhere", source
    # Another of Microsoft's names in the Python build's folder is not its own either.
    assert kind("MSVCP140.dll", rf"{_PY}\msvcp140.dll") == "elsewhere"
    # A package's own copy under the Python build's name is the package's.
    assert kind("pkg.libs\\vcruntime140.dll", rf"{_SITE}\pkg.libs\vcruntime140.dll") == "wheel"
    system = [("VCRUNTIME140.dll", r"C:\Windows\System32\vcruntime140.dll", "BINARY")]
    kept, left_out, _ = bundle.leave_out(system, [_SITE], _PY, _as_is)
    assert kept == [] and left_out["vc_runtime"] == ["VCRUNTIME140.dll"]


def test_where_a_file_came_from_is_decided_on_resolved_paths():
    # One folder under other names: a junction, a substituted drive, short names.
    aliases = {
        r"C:\link": _SITE,
        "S:": r"C:\b\env\Lib",
        r"C:\PROGRA~1\ENV": r"C:\Program Files\env",
        r"C:\pylink": r"C:\py313",
    }

    def resolve(path: str) -> str:
        for alias, target in aliases.items():
            if path.lower() == alias.lower() or path.lower().startswith(alias.lower() + "\\"):
                return target + path[len(alias) :]
        return path

    cases = [
        (r"C:\link\numpy.libs\msvcp140-abc.dll", [_SITE]),
        (rf"{_SITE}\numpy.libs\msvcp140-abc.dll", [r"S:\site-packages"]),
        (
            r"C:\PROGRA~1\ENV\Lib\SITE-P~1\numpy.libs\msvcp140-abc.dll",
            [r"C:\Program Files\env\Lib\SITE-P~1"],
        ),
    ]
    for source, roots in cases:
        assert bundle.package_folder(source, roots, resolve) == "numpy.libs", source
        # Compared as written, the same paths name no package's folder.
        assert bundle.package_folder(source, roots, _as_is) is None, source
    # The Python build's folder and its files, each named through a link.
    for source, python_dir in (
        (r"C:\py313\vcruntime140.dll", r"C:\pylink"),
        (r"C:\pylink\vcruntime140.dll", r"C:\py313"),
    ):
        vc = bundle.classify_vc_runtime("VCRUNTIME140.dll", source, [], python_dir, resolve)
        assert vc.kind == "python", source
        vc = bundle.classify_vc_runtime("VCRUNTIME140.dll", source, [], python_dir, _as_is)
        assert vc.kind == "elsewhere", source


def test_a_package_folder_reached_through_a_link_is_the_same_folder(tmp_path):
    site = tmp_path / "env" / "site-packages"
    dll = _file(site / "numpy.libs" / "msvcp140-abc.dll")
    link = tmp_path / "link"
    if os.name == "nt":  # a junction, which needs no privilege
        import _winapi

        _winapi.CreateJunction(str(site), str(link))
    else:
        link.symlink_to(site, target_is_directory=True)
    try:
        through = str(link / "numpy.libs" / dll.name)
        assert bundle.package_folder(through, [str(site)]) == "numpy.libs"
        assert bundle.package_folder(str(dll), [str(link)]) == "numpy.libs"
        assert bundle.package_folder(through, [str(link / "numpy.libs")]) is None
    finally:
        os.rmdir(link) if os.name == "nt" else link.unlink()


@pytest.mark.skipif(os.name != "nt", reason="short (8.3) names are Windows'")
def test_a_package_folder_given_by_its_short_name_is_the_same_folder(tmp_path):
    import ctypes

    site = tmp_path / "a long folder name" / "site-packages"
    dll = _file(site / "numpy.libs" / "msvcp140-abc.dll")
    buffer = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetShortPathNameW(str(site), buffer, len(buffer))
    short = buffer.value
    if not short or short.lower() == str(site).lower():
        pytest.skip("this volume makes no short names")
    assert bundle.package_folder(str(dll), [short]) == "numpy.libs"
    assert bundle.package_folder(os.path.join(short, "numpy.libs", dll.name), [str(site)]) == (
        "numpy.libs"
    )


def test_the_modules_that_import_msvcp140_are_left_out():
    # The only bundled modules that import MSVCP140.dll by that name (dumpbin and
    # pefile on the scikit-image 0.26.0 wheel); the app never loads them (#54).
    assert bundle.MSVCP140_IMPORTERS == (
        "skimage.feature._cascade",
        "skimage.feature._haar",
        "skimage.feature._hessian_det_appx",
        "skimage.feature.brief_cy",
        "skimage.util._remap",
    )
    # A misspelt name would leave nothing out: each one is a module that exists.
    for name in bundle.MSVCP140_IMPORTERS:
        assert importlib.util.find_spec(name) is not None, name
    spec = (WINDOWS / "proteia.spec").read_text(encoding="utf-8")
    assert "excludes=list(bundle.EXCLUDES + bundle.MSVCP140_IMPORTERS)," in spec


# What a fresh interpreter prints after importing argv[1:-1], then every module
# of proteia: those modules, and the loaded ones among argv[-1]'s "excluded"
# (JSON). It runs without the site module (-S), on argv[-1]'s "path": no .pth
# file runs first (setuptools' loads _distutils_hack where setuptools is installed).
_IMPORT_EVERYTHING = """
import importlib, json, pkgutil, sys
given = json.loads(sys.argv[-1])
sys.path[:0] = given["path"]
excluded = given["excluded"]
for name in sys.argv[1:-1]:
    importlib.import_module(name)
import proteia
names = sorted(info.name for info in pkgutil.walk_packages(proteia.__path__, "proteia."))
for name in names:
    importlib.import_module(name)
loaded = sorted(
    module for module in sys.modules
    if any(module == name or module.startswith(name + ".") for name in excluded)
)
print(json.dumps({"modules": names, "loaded": loaded}))
"""


def _import_everything(folder: Path, *first: str) -> dict:
    """The modules of proteia, and those the bundle leaves out that importing
    ``first`` and then all of them loads, in a fresh interpreter run in ``folder``
    on this one's import path (the state-folder variables still the test's own)."""
    given = {"path": sys.path, "excluded": [*bundle.EXCLUDES, *bundle.MSVCP140_IMPORTERS]}
    done = subprocess.run(
        [sys.executable, "-S", "-c", _IMPORT_EVERYTHING, *first, json.dumps(given)],
        capture_output=True,
        text=True,
        cwd=folder,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def test_importing_every_module_of_proteia_loads_none_the_bundle_leaves_out(tmp_path):
    # A module imported at the top of another (skimage.feature.corner imports
    # _hessian_det_appx) slips past test_nothing_the_app_imports_is_left_out:
    # here everything is imported, and what got loaded is looked at (#239).
    # Its limit: only the modules' top levels run, so an import inside a function
    # (imaging's `from skimage import io`, say) is not followed; the frozen
    # self-test is the check for those paths.
    found = _import_everything(tmp_path)
    modules = {
        ".".join(path.relative_to(PACKAGE.parent).with_suffix("").parts).removesuffix(".__init__")
        for path in PACKAGE.rglob("*.py")
    }
    assert set(found["modules"]) == modules - {"proteia"}
    assert found["loaded"] == []


def test_the_import_check_finds_an_excluded_module_another_imports_at_its_top(tmp_path):
    # The same check, had proteia imported skimage.feature.corner; on Linux the
    # module is there too, so what counts is that it is loaded.
    found = _import_everything(tmp_path, "skimage.feature.corner")
    assert found["loaded"] == ["skimage.feature._hessian_det_appx"]


_REDISTRIBUTABLE = "it loads only where the Visual C++ Redistributable is installed"


_LEAVE_OUT = "leave the module out (bundle.py) or bundle a copy its package provides"
_NEXT_TO_IT = (
    "an .exe finds it only in its own folder, before anything adds another, so bundle a copy"
    " next to it or leave the program out (proteia.spec)"
)


def _not_found(
    binary: str, dll: str, folders: str, why: str = _REDISTRIBUTABLE, step: str = _LEAVE_OUT
) -> str:
    return (
        f"{binary} imports {dll}, a Visual C++ runtime file the bundle does not carry where"
        f" Windows looks for it ({folders}): {why}; {step} (#54)"
    )


def test_a_vc_runtime_file_counts_only_where_windows_looks_for_it():
    classified = [
        _vc("_internal/VCRUNTIME140.dll", "python"),
        _vc("_internal/alpha.libs/vcomp140.dll", "wheel", folder="alpha.libs"),
        _vc("_internal/alpha/sub/concrt140.dll", "wheel", folder="alpha"),
        _vc("_internal/numpy.libs/msvcp140-abc.dll", "wheel", folder="numpy.libs"),
        _vc("_internal/alpha/bin/vcomp140.dll", "wheel", folder="alpha"),
    ]
    imports = {
        **{vc.path: [] for vc in classified},
        # found in _internal, in the binary's own folder, in its package's .libs folder
        "_internal/python313.dll": ["vcruntime140.DLL", "KERNEL32.dll"],
        "_internal/alpha/x.pyd": ["vcomp140.dll", "VCRUNTIME140.dll"],
        "_internal/alpha/sub/y.pyd": ["concrt140.dll", "VCOMP140.dll"],
        "_internal/numpy/_core/z.pyd": ["msvcp140-abc.dll", "python313.dll"],
        "_internal/numpy.libs/openblas.dll": ["msvcp140-abc.dll"],
        # an .exe's own folder only: its imports are found when its process starts,
        # before the bootloader adds _internal, and no package adds a folder for it
        "_internal/alpha/bin/helper.exe": ["vcomp140.dll"],
        # not: in another package's folder, in a folder below its own, in neither
        "_internal/beta/w.pyd": ["vcomp140.dll", "msvcp140-abc.dll", "vcomp140.dll"],
        "_internal/alpha/v.pyd": ["concrt140.dll"],
        "_internal/python3.dll": ["vcomp140.dll"],
        # an .exe, from _internal or its package's .libs folder (#239 review)
        "Proteia.exe": ["VCRUNTIME140.dll", "KERNEL32.dll"],
        "_internal/alpha/tool.exe": ["VCRUNTIME140.dll", "vcomp140.dll"],
    }
    assert bundle.vc_runtime_problems(imports, classified) == [
        _not_found("Proteia.exe", "VCRUNTIME140.dll", "the bundle folder", step=_NEXT_TO_IT),
        _not_found("_internal/alpha/tool.exe", "vcomp140.dll", "_internal/alpha", step=_NEXT_TO_IT),
        _not_found(
            "_internal/alpha/tool.exe", "VCRUNTIME140.dll", "_internal/alpha", step=_NEXT_TO_IT
        ),
        _not_found(
            "_internal/alpha/v.pyd",
            "concrt140.dll",
            "_internal, _internal/alpha, _internal/alpha.libs",
        ),
        _not_found(
            "_internal/beta/w.pyd",
            "msvcp140-abc.dll",
            "_internal, _internal/beta, _internal/beta.libs",
            "no Windows provides it",
        ),
        _not_found(
            "_internal/beta/w.pyd", "vcomp140.dll", "_internal, _internal/beta, _internal/beta.libs"
        ),
        _not_found("_internal/python3.dll", "vcomp140.dll", "_internal"),
    ]
    assert bundle.dll_search_folders("Proteia.exe") == ["."]
    assert bundle.dll_search_folders("_internal/pkg/bin/helper.EXE") == ["_internal/pkg/bin"]
    assert bundle.dll_search_folders("_internal/PIL/_imaging.pyd") == [
        "_internal",
        "_internal/PIL",
        "_internal/PIL.libs",
    ]


_UNCLASSIFIED = (
    ": a Visual C++ runtime file the spec's classification (the bundle manifest) does not"
    " list, so nothing says where it came from; the bundle carries only the Python build's"
    " copies and packages' own. Collect it as a binary, which the spec classifies, or leave"
    " it out (proteia.spec) (#54)"
)


def test_the_bundle_check_rechecks_the_specs_classification():
    classified = [
        _vc("_internal/VCRUNTIME140.dll", "python"),
        _vc("_internal/numpy.libs/msvcp140-abc.dll", "wheel", folder="numpy.libs"),
        _vc("_internal/MSVCP140.dll", "elsewhere", source=r"C:\Windows\System32\msvcp140.dll"),
    ]
    imports = {
        "_internal/vcruntime140.dll": [],  # classified, in another case
        "_internal/numpy.libs/msvcp140-abc.dll": [],
        "_internal/MSVCP140.dll": [],  # classified as System32's, yet in the bundle
        "_internal/pkg/sub/vcomp140.dll": [],  # never classified: collected as data, say
        "_internal/pkg/sub/x.pyd": ["vcomp140.dll", "VCRUNTIME140.dll"],
        "_internal/numpy/y.pyd": ["MSVCP140.dll", "msvcp140-abc.dll"],
    }
    assert bundle.vc_runtime_problems(imports, classified) == [
        "_internal/MSVCP140.dll: a Visual C++ runtime file from"
        " C:\\Windows\\System32\\msvcp140.dll, which is neither the Python build's folder nor"
        " a package's; the bundle leaves such files out, so find what added it after the spec"
        " left it out (a data entry of the same name, say) (#54)",
        "_internal/pkg/sub/vcomp140.dll" + _UNCLASSIFIED,
        _not_found(
            "_internal/numpy/y.pyd",
            "MSVCP140.dll",
            "_internal, _internal/numpy, _internal/numpy.libs",
        ),
        _not_found(
            "_internal/pkg/sub/x.pyd",
            "vcomp140.dll",
            "_internal, _internal/pkg/sub, _internal/pkg.libs",
        ),
    ]
    # A manifest from before the classification classifies nothing.
    assert bundle.read_vc_runtime({"proteia": "0.1.0.dev0"}) == []
    assert bundle.vc_runtime_problems({"_internal/VCRUNTIME140.dll": []}, []) == [
        "_internal/VCRUNTIME140.dll" + _UNCLASSIFIED
    ]


def _python_left_out(path: str, source: str, importers: int) -> str:
    return (
        f"{path}: the Python build's own Visual C++ runtime file, which the spec left out:"
        f" PyInstaller found it in {source}, not directly in the Python installation's folder"
        f" (sys.base_prefix); bundled binaries that import it: {importers}. Build with a"
        " Python whose own folder holds vcruntime140.dll and vcruntime140_1.dll, such as an"
        " installed or uv-managed CPython (#54)"
    )


def test_the_python_builds_file_left_out_is_reported_as_the_cause():
    # Before the #239 review the build printed only one line per importer (110 in
    # a real build), each with advice that cannot work, and never this cause.
    imports = {
        "_internal/python313.dll": ["VCRUNTIME140.dll", "KERNEL32.dll"],
        "_internal/_ssl.pyd": ["vcruntime140.dll", "VCRUNTIME140_1.dll"],
        "_internal/skimage/util/_remap.pyd": ["MSVCP140.dll", "VCRUNTIME140_1.dll"],
        "_internal/VCRUNTIME140_1.dll": [],
    }
    system = r"C:\Windows\System32"
    classified = [
        _vc("_internal/VCRUNTIME140.dll", "elsewhere", source=rf"{system}\VCRUNTIME140.dll"),
        _vc("_internal/VCRUNTIME140_1.dll", "python"),
        # System32's MSVCP140.dll left out is what #54 wants: its importers are the cause.
        _vc("_internal/MSVCP140.dll", "elsewhere", source=rf"{system}\msvcp140.dll"),
    ]
    assert bundle.vc_runtime_problems(imports, classified) == [
        _python_left_out("_internal/VCRUNTIME140.dll", rf"{system}\VCRUNTIME140.dll", 2),
        _not_found(
            "_internal/skimage/util/_remap.pyd",
            "MSVCP140.dll",
            "_internal, _internal/skimage/util, _internal/skimage.libs",
        ),
    ]
    # Both of them, from a Python run from its source tree; nothing imports the second.
    tree = r"C:\cpython\PCbuild\amd64"
    classified = [
        _vc("_internal/VCRUNTIME140.dll", "elsewhere", source=rf"{tree}\vcruntime140.dll"),
        _vc("_internal/vcruntime140_1.dll", "elsewhere", source=rf"{tree}\vcruntime140_1.dll"),
    ]
    imports = {"_internal/python313.dll": ["VCRUNTIME140.dll"]}
    assert bundle.vc_runtime_problems(imports, classified) == [
        _python_left_out("_internal/VCRUNTIME140.dll", rf"{tree}\vcruntime140.dll", 1),
        _python_left_out("_internal/vcruntime140_1.dll", rf"{tree}\vcruntime140_1.dll", 0),
    ]
    # In the bundle after all (a data entry, say): what the bundle holds is the problem.
    imports["_internal/VCRUNTIME140.dll"] = []
    problems = bundle.vc_runtime_problems(imports, classified)
    assert problems[0] == _python_left_out(
        "_internal/vcruntime140_1.dll", rf"{tree}\vcruntime140_1.dll", 0
    )
    assert problems[1].startswith(
        rf"_internal/VCRUNTIME140.dll: a Visual C++ runtime file from {tree}\vcruntime140.dll,"
    )
    assert len(problems) == 3 and problems[2].startswith("_internal/python313.dll imports")


def test_a_renamed_copy_from_outside_every_package_is_a_problem():
    # Main failed the build on it (its notices: "source not identified"); left out
    # and imported by nothing, it would go missing quietly (#239 review).
    entries = [("msvcp140-def.dll", r"C:\other\bin\msvcp140-def.dll", "BINARY")]
    kept, left_out, classified = bundle.leave_out(entries, [_SITE], _PY, _as_is)
    assert kept == [] and left_out["vc_runtime"] == ["msvcp140-def.dll"]
    renamed = (
        "_internal/msvcp140-def.dll: a package's renamed copy of the Visual C++ runtime, which"
        " the spec left out: PyInstaller found it in C:\\other\\bin\\msvcp140-def.dll, outside"
        " every package folder of the build environment, and no Windows provides it under that"
        " name; bundled binaries that import it: {}. Install the package that ships it in the"
        " build environment (uv.lock), or leave out what needs it (bundle.py) (#54)"
    )
    assert bundle.vc_runtime_problems({"_internal/python313.dll": []}, classified) == [
        renamed.format(0)
    ]
    imports = {"_internal/pkg/x.pyd": ["MSVCP140-DEF.dll"], "_internal/pkg/y.pyd": []}
    assert bundle.vc_runtime_problems(imports, classified) == [renamed.format(1)]


def test_the_classification_goes_through_the_manifest_unchanged():
    classified = [
        _vc("_internal/VCRUNTIME140.dll", "python"),
        _vc("_internal/numpy.libs/msvcp140-abc.dll", "wheel", folder="numpy.libs"),
        _vc("_internal/MSVCP140.dll", "elsewhere", source=r"C:\Windows\System32\msvcp140.dll"),
    ]
    written = json.dumps({bundle.VC_RUNTIME_KEY: [vc.to_json() for vc in classified]})
    manifest = json.loads(written)
    assert manifest["vc_runtime"][1] == {
        "path": "_internal/numpy.libs/msvcp140-abc.dll",
        "kind": "wheel",
        "folder": "numpy.libs",
    }
    read = bundle.read_vc_runtime(manifest)
    assert read == classified
    assert [type(vc.kind) for vc in read] == [bundle.VcKind] * 3
    assert read[2].source == r"C:\Windows\System32\msvcp140.dll" and read[2].folder is None
    with pytest.raises(ValueError, match="'system'"):
        bundle.read_vc_runtime({"vc_runtime": [{"path": "_internal/x.dll", "kind": "system"}]})
    with pytest.raises(AttributeError):  # one classification: no entry changes afterwards
        read[0].kind = bundle.VcKind.WHEEL


def _check_libs_folders(roots) -> list[str]:
    """The ``<name>.libs`` folders in the site-packages folders ``roots``, each
    asserted to be added by the package ``<name>`` (its ``__init__.py`` calls
    ``os.add_dll_directory`` with it), as delvewheel's patch does."""
    folders = sorted(path for root in roots for path in root.glob("*.libs") if path.is_dir())
    for folder in folders:
        init = folder.with_name(folder.name.removesuffix(".libs")) / "__init__.py"
        assert init.is_file(), f"{folder.name}: no package {init.parent.name} to add it"
        text = init.read_text(encoding="utf-8")
        assert "os.add_dll_directory(" in text and f"'{folder.name}'" in text, folder.name
    return [folder.name for folder in folders]


@pytest.mark.skipif(os.name != "nt", reason="delvewheel patches the Windows wheels")
def test_each_libs_folder_is_added_by_the_package_it_is_named_after():
    # dll_search_folders counts a package's <name>.libs folder as searched for the
    # package's binaries: delvewheel's patch to its __init__.py adds it with
    # os.add_dll_directory. A wheel that named it otherwise would fail here.
    import sysconfig

    roots = {Path(sysconfig.get_path(name)) for name in ("purelib", "platlib")}
    assert {"numpy.libs", "scipy.libs"} <= set(_check_libs_folders(roots))


def test_a_libs_folder_named_otherwise_fails_the_check_by_its_name(tmp_path):
    # pyzmq's wheel names it after the distribution (pyzmq.libs, for zmq): the
    # check says which folder, rather than failing to read a file.
    site = tmp_path / "site-packages"
    patch = "os.add_dll_directory(os.path.join(here, os.pardir, 'numpy.libs'))\n"
    _file(site / "numpy" / "__init__.py", patch)
    (site / "numpy.libs").mkdir()
    assert _check_libs_folders([site]) == ["numpy.libs"]
    (site / "pyzmq.libs").mkdir()
    _file(site / "zmq" / "__init__.py", "os.add_dll_directory(libs)\n")
    with pytest.raises(AssertionError, match=r"^pyzmq\.libs: no package pyzmq to add it"):
        _check_libs_folders([site])


def _pe(
    sections=((0x400, 0x200),),
    size: int | None = None,
    at: int = 0x80,
    *,
    magic: int = 0x20B,
    optional: int = 0xF0,
    directories: int = 16,
    names=(b".text", b".data", b".rsrc"),
) -> bytes:
    """The headers of a PE file: a DOS header pointing at the PE signature at
    ``at``, a COFF header, an optional header of ``optional`` bytes (PE32+ by
    ``magic``, with ``directories`` data directories) and a section table with
    ``sections`` ((offset, size) of each section's data, named by ``names``), and
    zeros up to ``size`` bytes (by default, to the end of the last section)."""
    dos = bytearray(at)
    dos[:2] = b"MZ"
    dos[0x3C:0x40] = struct.pack("<I", at)
    coff = struct.pack("<HHIIIHH", 0x8664, len(sections), 0, 0, 0, optional, 0x2022)
    header = bytearray(optional + 4)
    header[:4] = struct.pack("<HBB", magic, 14, 38)  # and the linker's version, as MSVC's
    count_at = {0x10B: 92, 0x20B: 108}.get(magic, 108)
    header[count_at : count_at + 4] = struct.pack("<I", directories)
    table = b"".join(
        struct.pack("<8sIIIIIIHHI", name, raw + 0x11, 0x1000, raw, offset, 0, 0, 0, 0, 0x60000020)
        for name, (offset, raw) in zip(names, sections, strict=False)
    )
    data = bytes(dos) + b"PE" + bytes(2) + coff + bytes(header[:optional]) + table
    end = max([len(data), *(offset + raw for offset, raw in sections)])
    total = end if size is None else size
    return (data + bytes(max(0, total - len(data))))[:total]


_NOT_PE = {
    "_internal/pkg/cut.dll": (
        "its section '.text' runs past the end of the file, which is cut short"
    ),
    "_internal/pkg/cut2.dll": (
        "its section '.data' runs past the end of the file, which is cut short"
    ),
    "_internal/pkg/empty.pyd": "the file is empty",
    "_internal/pkg/far.DLL": "there is no PE signature where its DOS header points",
    "_internal/pkg/mz.exe": "it does not start with a DOS header",
    "_internal/pkg/nosig.dll": "there is no PE signature where its DOS header points",
    "_internal/pkg/notes.dll": "it does not start with a DOS header",
    "_internal/pkg/short.dll": "its PE header runs past the end of the file",
    "_internal/pkg/table.dll": "its section table runs past the end of the file",
    "_internal/pkg/table2.dll": "its section table runs past the end of the file",
}


def _not_pe_bundle(folder: Path) -> None:
    """Files named like binaries that are not PE files, each as :data:`_NOT_PE` says."""
    pe = _pe()
    for name, data in {
        "cut.dll": _pe(size=0x500),
        "cut2.dll": _pe(((0x400, 0x200), (0x600, 0x200)), size=0x7FF),  # the second section
        "empty.pyd": b"",
        "far.DLL": b"MZ" + bytes(58) + struct.pack("<I", 4096),  # 64 bytes, pointing past them
        "mz.exe": b"MZ",
        "nosig.dll": pe[:0x80] + b"NE" + pe[0x82:],
        "notes.dll": b"not a binary\n",
        "short.dll": pe[: 0x80 + 10],  # the signature, and part of the COFF header
        "table.dll": pe[: 0x80 + 24 + 0xF0 + 20],
        "table2.dll": _pe(((0x400, 0x200), (0x600, 0x200)))[: 0x80 + 24 + 0xF0 + 50],
        "notes.txt": b"MZ",  # not named like a binary
    }.items():
        path = folder / "_internal" / "pkg" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (folder / "_internal" / "pkg" / "folder.dll").mkdir()  # a folder named like one


def test_a_file_named_like_a_binary_that_is_not_a_pe_file_is_a_bundle_problem(tmp_path, capsys):
    # Before #239 pefile's error stopped the build with a traceback, and a file cut
    # short gave no error at all: pefile found no imports in it.
    folder = tmp_path / "Proteia"
    _not_pe_bundle(folder)
    assert bundle.pe_header_problem(Path(_file(tmp_path / "ok.dll"))) == "the file is empty"
    # Whole files: the last section ends at the end; a section without data (an
    # uninitialized one) may point anywhere; no section at all.
    for data in (_pe(), _pe(((0x400, 0x200), (0x9000, 0))), _pe(()), _pe(at=0x40)):
        (tmp_path / "ok.dll").write_bytes(data)
        assert bundle.pe_header_problem(tmp_path / "ok.dll") is None
    # None of them gets as far as pefile, which the test environment may lack.
    assert bundle.bundle_imports(folder) == ({}, _NOT_PE)
    out = tmp_path / "pe-imports.json"
    assert bundle.main(["imports", str(folder), str(out)]) == 0
    assert bundle.read_imports(out) == ({}, _NOT_PE)
    assert capsys.readouterr().out == (
        f"bundle: the imports of 0 binaries in {out}; 10 files named like binaries are not"
        " PE files\n"
    )
    _file(folder / bundle.EXE_NAME)
    problems = build.bundle_problems(folder, {}, _NOT_PE, [])
    next_step = (
        "; find the package that ships it, then leave the file out (proteia.spec) or reinstall"
        " that package if the file is damaged"
    )
    assert problems[0] == (
        "_internal/pkg/cut.dll: not a valid PE file (its section '.text' runs past the end of"
        " the file, which is cut short), so the DLLs it imports cannot be checked" + next_step
    )
    assert problems[2] == (
        "_internal/pkg/empty.pyd: not a valid PE file (the file is empty), so the DLLs it"
        " imports cannot be checked" + next_step
    )
    assert len([p for p in problems if "not a valid PE file" in p]) == 10
    # A file that is not one is no runtime the bundle carries either.
    imports = {"_internal/pkg/x.pyd": ["MSVCP140.dll"]}
    not_pe = {"_internal/MSVCP140.dll": "it does not start with a DOS header"}
    problems = build.bundle_problems(folder, imports, not_pe, [_vc(*not_pe, "python")])
    assert (
        _not_found(
            "_internal/pkg/x.pyd", "MSVCP140.dll", "_internal, _internal/pkg, _internal/pkg.libs"
        )
        in problems
    )


def test_a_pe_header_that_contradicts_itself_names_no_section_it_does_not_have(tmp_path):
    # With a wrong optional header size the section table is read from the wrong
    # place, and main named a garbage "section" ('\x06'; #239 review).
    path = tmp_path / "x.dll"
    for data, why in (
        (_pe(optional=0), "its optional header is neither PE32 nor PE32+"),
        (_pe(magic=0x107), "its optional header is neither PE32 nor PE32+"),
        (_pe(optional=96), "its optional header (96 bytes) is shorter than its PE32+ fields"),
        (
            _pe(magic=0x10B, optional=95, directories=0),
            "its optional header (95 bytes) is shorter than its PE32 fields",
        ),
        (
            _pe(optional=112),
            "its optional header (112 bytes) cannot hold its 16 data directories",
        ),
        (
            _pe(optional=0xF0, directories=17),
            "its optional header (240 bytes) cannot hold its 17 data directories",
        ),
        (_pe()[: 0x80 + 24 + 100], "its optional header runs past the end of the file"),
        (
            _pe(((0x400, 0x200),), size=0x500, names=(bytes([6]),)),
            "its section number 1 runs past the end of the file, which is cut short",
        ),
        (
            _pe(((0x400, 0x100), (0x500, 0x200)), size=0x600, names=(b".text", b"")),
            "its section number 2 runs past the end of the file, which is cut short",
        ),
    ):
        path.write_bytes(data)
        assert bundle.pe_header_problem(path) == why
    # The real layouts: PE32 (224 bytes) and PE32+ (240), with 16 data directories,
    # and a header with fewer of them.
    for data in (
        _pe(magic=0x10B, optional=224),
        _pe(),
        _pe(optional=112, directories=0),
        _pe(magic=0x10B, optional=96, directories=0),
        _pe(magic=0x10B, optional=96 + 8, directories=1),
    ):
        path.write_bytes(data)
        assert bundle.pe_header_problem(path) is None


def _fake_pefile(monkeypatch, *, refuse: str = "", refuse_tables: str = "", declared=(), **tables):
    """A stand-in for pefile in ``sys.modules``: ``refuse`` makes it refuse the
    file, ``refuse_tables`` the import tables; ``declared`` are the
    VirtualAddress fields of the data directories, ``tables`` the import tables
    it reads (DIRECTORY_ENTRY_IMPORT=[...]). Returns the list each closed file
    is appended to."""
    closed: list[str] = []

    class PEFormatError(Exception):
        pass

    class PE:
        def __init__(self, name, fast_load):
            assert fast_load
            if refuse:
                raise PEFormatError(refuse)
            self.name = name
            self.OPTIONAL_HEADER = types.SimpleNamespace(
                DATA_DIRECTORY=[types.SimpleNamespace(VirtualAddress=va) for va in declared]
            )

        def parse_data_directories(self, directories, import_dllnames_only):
            assert directories == [1, 13] and import_dllnames_only
            if refuse_tables:
                raise PEFormatError(refuse_tables)
            for attribute, dlls in tables.items():
                entries = [types.SimpleNamespace(dll=dll.encode("ascii")) for dll in dlls]
                setattr(self, attribute, entries)

        def close(self):
            closed.append(self.name)

    module = types.ModuleType("pefile")
    module.PE, module.PEFormatError = PE, PEFormatError
    module.DIRECTORY_ENTRY = {
        "IMAGE_DIRECTORY_ENTRY_IMPORT": 1,
        "IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT": 13,
    }
    monkeypatch.setitem(sys.modules, "pefile", module)
    return closed


def test_pefiles_refusal_and_an_import_table_it_cannot_read_are_reported(tmp_path, monkeypatch):
    path = tmp_path / "x.dll"
    path.write_bytes(_pe())
    declared = [0] * 16
    with monkeypatch.context() as patch:
        _fake_pefile(patch, refuse="'Invalid NT Headers signature.'")
        with pytest.raises(bundle.NotPE, match=r"^pefile cannot read it: 'Invalid NT"):
            bundle.pe_imports(path)
    with monkeypatch.context() as patch:
        closed = _fake_pefile(patch, refuse_tables="'bad'", declared=declared)
        with pytest.raises(bundle.NotPE, match=r"^pefile cannot read its import tables: 'bad'$"):
            bundle.pe_imports(path)
        assert closed == [str(path)]
    import_declared = [0, 0x2000] + [0] * 14
    with monkeypatch.context() as patch:
        closed = _fake_pefile(patch, declared=import_declared)
        with pytest.raises(bundle.NotPE, match="^pefile finds no entry in the import table"):
            bundle.pe_imports(path)
        assert closed == [str(path)]
    delay_declared = [0] * 13 + [0x3000, 0, 0]
    with monkeypatch.context() as patch:
        _fake_pefile(patch, declared=delay_declared, DIRECTORY_ENTRY_IMPORT=["KERNEL32.dll"])
        with pytest.raises(bundle.NotPE, match="^pefile finds no entry in the delay-import"):
            bundle.pe_imports(path)
    # Both tables read: their names, once each, in any case order.
    with monkeypatch.context() as patch:
        closed = _fake_pefile(
            patch,
            declared=[0, 0x2000] + [0] * 11 + [0x3000, 0, 0],
            DIRECTORY_ENTRY_IMPORT=["VCRUNTIME140.dll", "KERNEL32.dll"],
            DIRECTORY_ENTRY_DELAY_IMPORT=["advapi32.dll", "MSVCP140.dll", "VCRUNTIME140.dll"],
        )
        assert bundle.pe_imports(path) == [
            "advapi32.dll",
            "KERNEL32.dll",
            "MSVCP140.dll",
            "VCRUNTIME140.dll",
        ]
        assert closed == [str(path)]
    # Fewer data directories than the delay-import table's index, or none declared.
    for declared in ([0, 0x2000], [0, 0x2000] + [0] * 11, [0] * 16):
        with monkeypatch.context() as patch:
            _fake_pefile(patch, declared=declared, DIRECTORY_ENTRY_IMPORT=["KERNEL32.dll"])
            assert bundle.pe_imports(path) == ["KERNEL32.dll"]
    with monkeypatch.context() as patch:
        _fake_pefile(patch, declared=[0] * 16)
        assert bundle.pe_imports(path) == []
    # In a bundle, pefile's refusal is one more file that is not a PE file.
    folder = tmp_path / "Proteia"
    _file(folder / "_internal" / "a.pyd").write_bytes(_pe())
    _file(folder / "_internal" / "b.dll").write_bytes(b"text")
    with monkeypatch.context() as patch:
        _fake_pefile(patch, refuse="'DOS Header magic not found.'")
        assert bundle.bundle_imports(folder) == (
            {},
            {
                "_internal/a.pyd": "pefile cannot read it: 'DOS Header magic not found.'",
                "_internal/b.dll": "it does not start with a DOS header",
            },
        )


@pytest.mark.skipif(os.name != "nt", reason="the Windows runtime's DLLs")
def test_the_import_tables_are_read_with_pefile(tmp_path):
    pefile = pytest.importorskip("pefile", reason="the build group is not installed")
    python_dll = Path(sys.base_prefix) / f"python{sys.version_info[0]}{sys.version_info[1]}.dll"
    names = {name.lower() for name in bundle.pe_imports(python_dll)}
    assert {"kernel32.dll", "vcruntime140.dll"} <= names
    folder = tmp_path / "Proteia"
    shutil.copy(python_dll, _file(folder / "_internal" / python_dll.name))
    _file(folder / "_internal" / "data.txt", "not a binary")
    _file(folder / "_internal" / "notes.dll", "not a binary")
    cut = _file(folder / "_internal" / "cut.pyd")
    cut.write_bytes(python_dll.read_bytes()[:4096])
    # pefile alone reads the file cut short without an error, and no imports.
    pe = pefile.PE(str(cut), fast_load=True)
    tables = ["IMAGE_DIRECTORY_ENTRY_IMPORT", "IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY[name] for name in tables], import_dllnames_only=True
    )
    assert getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) == []
    pe.close()
    imports, not_pe = bundle.bundle_imports(folder)
    assert list(imports) == [f"_internal/{python_dll.name}"]
    assert list(not_pe) == ["_internal/cut.pyd", "_internal/notes.dll"]
    assert "runs past the end of the file" in not_pe["_internal/cut.pyd"]


@pytest.mark.parametrize(
    ("version", "numbers"),
    [
        ("0.1.0.dev0", (0, 1, 0, 0)),
        ("0.1.0", (0, 1, 0, 0)),
        ("1.2", (1, 2, 0, 0)),
        ("2.10.3rc1", (2, 10, 3, 0)),
        ("1.2.3.4.5", (1, 2, 3, 4)),
        ("v3.0.1.post2", (3, 0, 1, 0)),
    ],
)
def test_the_version_resource_numbers_the_release(version, numbers):
    assert bundle.numeric_version(version) == numbers


@pytest.mark.parametrize("version", ["dev", "", "1.70000.0"])
def test_a_version_without_usable_numbers_is_refused(version):
    with pytest.raises(ValueError):
        bundle.numeric_version(version)


def test_the_installer_is_named_after_the_version():
    assert bundle.installer_name("0.1.0.dev0") == "Proteia-0.1.0.dev0-setup"
    with pytest.raises(ValueError):
        bundle.installer_name("0.1 beta")


def test_bundled_modules_and_files_are_traced_to_their_distributions():
    packages = {"numpy": ["numpy"], "PIL": ["pillow"], "proteia": ["proteia"], "yaml": ["PyYAML"]}
    modules = ["numpy.linalg", "os.path", "proteia.web", "pyi_rth_inspect", "_pyi_rth_utils", "x.y"]
    files = [
        "numpy.libs/libscipy_openblas.dll",
        "PIL\\_imaging.cp313-win_amd64.pyd",
        "_ssl.pyd",
        "python313.dll",
        "VCRUNTIME140.dll",
        "libcrypto-3-x64.dll",
        "numpy-2.4.6.dist-info/METADATA",
        "base_library.zip",
        "strange.dll",
    ]
    found, unknown = bundle.bundled_distributions(modules, files, packages, {"os", "_ssl"})
    assert found == {"numpy": {"numpy"}, "pillow": {"PIL"}, "proteia": {"proteia"}}
    assert unknown == {"x", "strange"}


def test_forbidden_names_are_found_in_any_part_of_a_path():
    paths = [
        "_internal/PySide6/QtCore.pyd",
        "_internal/napari/__init__.pyc",
        "_internal/_tcl_data/init.tcl",
        "_internal/matplotlib/mpl-data/images/qt4_editor_options.svg",
        "_internal/numpy/core.pyd",
        "Proteia.exe",
    ]
    assert bundle.forbidden_paths(paths) == sorted(paths[:3])


def test_the_record_distributions_are_matched_by_normalized_name():
    names = ["numpy", "scipy", "scikit_image", "Pillow", "tifffile", "matplotlib"]
    assert bundle.missing_record_distributions(names) == []
    assert bundle.missing_record_distributions(names[:3]) == ["pillow", "tifffile"]


def test_the_spec_is_valid_python():
    compile((WINDOWS / "proteia.spec").read_text(encoding="utf-8"), "proteia.spec", "exec")


# --- The installer script ---


def test_the_application_id_never_changes():
    # Upgrades and the uninstall entry depend on it: changing it makes a second product.
    assert bundle.iss_app_id(ISS) == "B32AD498-6981-42C9-9528-00A05682ECD0"


def test_the_build_passes_every_define_the_script_requires(tmp_path):
    required = set(re.findall(r"^#ifndef (\w+)$", ISS, re.MULTILINE))
    defines = build.iscc_defines("0.1.0.dev0", tmp_path)
    assert set(defines) == required
    assert defines["NumericVersion"] == "0.1.0.0"
    assert defines["AppPublisher"] == bundle.PUBLISHER
    [pinned] = re.findall(r'^#define RequiredInnoSetup "([\d.]+)"$', ISS, re.MULTILINE)
    assert pinned == build.INNO_SETUP_VERSION


def _iss_constant(name: str) -> str:
    [value] = re.findall(rf"^\s*{name} = '([^']*)';$", ISS, re.MULTILINE)
    return value


def test_the_script_finds_the_launcher_files_where_the_launcher_writes_them(tmp_path, monkeypatch):
    assert _iss_constant("LockFileName") == launch.LOCK_FILE
    assert _iss_constant("InstanceFileName") == launch.INSTANCE_FILE
    assert _iss_constant("RedirectFileName") == launch.REDIRECT_FILE
    assert "GetEnv('LOCALAPPDATA')" in ISS  # the variable first, as state_dir reads it
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert launch.state_dir() == tmp_path / _iss_constant("StateFolderName")


def test_the_script_accepts_the_tokens_the_launcher_makes():
    low, high = re.search(r"\{(\d+),(\d+)\}", server.TOKEN_PATTERN.pattern).groups()
    assert f"(Length(S) >= {low}) and (Length(S) <= {high})" in ISS
    assert "'/api/quit'" in ISS and "'Bearer '" in ISS


def _pascal_routine(name: str) -> str:
    """The text of the [Code] routine ``name``, up to the next routine."""
    match = re.search(
        rf"^(?:function|procedure) {name}\b.*?(?=^(?:function|procedure) |\Z)",
        ISS,
        re.MULTILINE | re.DOTALL,
    )
    assert match, name
    return match.group(0)


def test_nothing_is_deleted_while_a_proteia_runs_the_installed_program():
    # Quit, then the installed Proteia.exe must be free; an installer or
    # uninstaller that could not stop Proteia stops itself (install_check step 4
    # runs these paths).
    problem = _pascal_routine("ProteiaProblem")
    assert "StopRunningProteia" in problem and "ProgramInUse" in problem
    assert "Result := ProteiaProblem;" in _pascal_routine("PrepareToInstall")
    in_use = _pascal_routine("ProgramInUse")
    assert "ExpandConstant('{app}\\{#ExeName}')" in in_use
    # A Proteia that does not answer Quit is refused, unless its process id is
    # stale (another program has it now).
    stop = _pascal_routine("StopRunningProteia")
    no_answer = stop[stop.index("if Status = 0 then") : stop.index("else if Status = 409")]
    assert "CompareText(Name, '{#ExeName}') = 0" in no_answer and "Result :=" in no_answer


def test_the_uninstaller_stops_proteia_only_after_the_user_confirmed():
    # InitializeUninstall runs before the "Are you sure" prompt; usAppMutexCheck
    # after it and before anything is removed, and Abort ends the uninstall there.
    assert "InitializeUninstall" not in ISS
    steps = _pascal_routine("CurUninstallStepChanged")
    check = steps[steps.index("usAppMutexCheck") : steps.index("usPostUninstall")]
    assert "ProteiaProblem" in check and "Abort;" in check


def test_an_uninstall_deletes_only_the_launcher_files():
    deleted = re.findall(r"DeleteFile\(Folder \+ '\\' \+ (\w+)\);", ISS)
    assert sorted(deleted) == ["InstanceFileName", "LockFileName", "RedirectFileName"]
    assert "[UninstallDelete]" not in ISS  # nothing else of the state folder goes
    assert "DelTree" not in ISS  # the session log's folder among it
    assert "RemoveDir(Folder)" in ISS  # which removes it only when empty
    code = [line.lower() for line in ISS.splitlines() if not line.startswith(";")]
    assert not [line for line in code if "userdocs" in line or "documents" in line]


# --- The build ---


def test_the_versions_agree():
    assert build.read_version(ROOT) == proteia.__version__
    assert build.project_version('[project]\nversion = "1.2"\n') == "1.2"
    assert build.package_version('"""x"""\n__version__ = "1.2.dev3"\n') == "1.2.dev3"
    with pytest.raises(build.BuildError):
        build.package_version("VERSION = 1\n")


def test_the_build_tools_are_a_pinned_group_of_their_own():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = pyproject["dependency-groups"]["build"]
    assert pins == ["pyinstaller==6.22.3", "pyinstaller-hooks-contrib==2026.7"]
    assert "build" not in pyproject.get("tool", {}).get("uv", {}).get("default-groups", [])
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {package["name"]: package["version"] for package in lock["package"]}
    for pin in pins:
        name, _, version = pin.partition("==")
        assert versions[name] == version


def test_numbers_are_compared_key_by_key():
    assert build.compare_numbers({"a": 1.0, "b": [1, 2]}, {"b": [1, 2], "a": 1.0}) == []
    assert build.compare_numbers({"a": 1.0, "c": 3}, {"a": 1.5}) == [
        "a: frozen 1.0, unfrozen 1.5",
        "c: frozen 3, unfrozen None",
    ]


def test_inno_setup_is_looked_for_in_the_variable_then_its_folders(tmp_path):
    env = {
        "ISCC": str(tmp_path / "mine" / "ISCC.exe"),
        "PATH": "",
        "ProgramFiles(x86)": str(tmp_path / "x86"),
        "LOCALAPPDATA": str(tmp_path / "local"),
    }
    assert build.iscc_candidates(env) == [
        tmp_path / "mine" / "ISCC.exe",
        tmp_path / "x86" / "Inno Setup 7" / "ISCC.exe",
        tmp_path / "local" / "Programs" / "Inno Setup 7" / "ISCC.exe",
    ]


def _file(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def test_inno_setups_license_is_found_next_to_the_real_compiler(tmp_path):
    iscc = _file(tmp_path / "Inno Setup 7" / "ISCC.exe")
    license_file = _file(tmp_path / "Inno Setup 7" / "license.txt", "Inno Setup License")
    assert build.inno_setup_license(iscc) == license_file.resolve()
    # A shim (Scoop's, say) runs the compiler from elsewhere: the build stops
    # before it starts, with a message instead of a traceback in the notices step.
    shim = _file(tmp_path / "scoop" / "shims" / "ISCC.exe")
    with pytest.raises(build.BuildError, match=r"license\.txt is not next to .*--iscc"):
        build.inno_setup_license(shim)


def test_the_finished_bundle_is_checked(tmp_path):
    dist = tmp_path / "Proteia"
    _file(dist / bundle.EXE_NAME)
    _file(dist / "_internal" / "proteia" / "web" / "static" / "index.html")
    for name in ("numpy-2.4.6", "scipy-1.17.1", "scikit_image-0.26.0", "pillow-12.2.0"):
        _file(dist / "_internal" / f"{name}.dist-info" / "METADATA")
    assert build.bundle_problems(dist, {}, {}, []) == [
        "no metadata of tifffile, whose version the record reads"
    ]
    _file(dist / "_internal" / "tifffile-2026.5.15.dist-info" / "METADATA")
    assert build.bundle_problems(dist, {}, {}, []) == []
    # The Python build's Visual C++ runtime and a package's own copies, renamed
    # or not, as the spec classified them, may stay; so may modules that import
    # only those, where Windows finds them.
    classified = [
        _vc("_internal/VCRUNTIME140.dll", "python"),
        _vc("_internal/VCRUNTIME140_1.dll", "python"),
        _vc("_internal/numpy.libs/msvcp140-0123abcd.dll", "wheel", folder="numpy.libs"),
        _vc("_internal/beta/concrt140.dll", "wheel", folder="beta"),
    ]
    # The checks of DLLs take the files from the import tables' map, not the folder.
    imports = {vc.path: [] for vc in classified}
    imports["_internal/beta/x.pyd"] = ["concrt140.dll", "VCRUNTIME140.dll"]
    assert build.bundle_problems(dist, imports, {}, classified) == []
    _file(dist / "_internal" / "PySide6" / "QtCore.pyd")
    imports["_internal/api-ms-win-crt-heap-l1-1-0.dll"] = []
    imports["_internal/MSVCP140.dll"] = []
    imports["vcomp140.dll"] = []
    imports["_internal/skimage/util/_remap.pyd"] = ["MSVCP140.dll"]
    not_pe = {"_internal/pkg/ucrtbase.dll": "it does not start with a DOS header"}
    assert build.bundle_problems(dist, imports, not_pe, classified) == [
        "_internal/PySide6 must not be in the bundle",
        "_internal/PySide6/QtCore.pyd must not be in the bundle",
        "_internal/pkg/ucrtbase.dll: not a valid PE file (it does not start with a DOS"
        " header), so the DLLs it imports cannot be checked; find the package that ships it,"
        " then leave the file out (proteia.spec) or reinstall that package if the file is"
        " damaged",
        "_internal/api-ms-win-crt-heap-l1-1-0.dll: a Universal CRT DLL",
        "_internal/pkg/ucrtbase.dll: a Universal CRT DLL",
        "_internal/MSVCP140.dll" + _UNCLASSIFIED,
        "vcomp140.dll" + _UNCLASSIFIED,
        _not_found(
            "_internal/skimage/util/_remap.pyd",
            "MSVCP140.dll",
            "_internal, _internal/skimage/util, _internal/skimage.libs",
        ),
    ]


def test_the_bundle_step_checks_the_bundle_against_the_specs_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("UV", "uv")
    work = tmp_path / "work"
    maker = build.Build(work, tmp_path / "out")
    manifest = work / "pyinstaller" / "proteia" / "bundle-manifest.json"
    imports = {"_internal/VCRUNTIME140.dll": [], "_internal/numpy/x.pyd": ["VCRUNTIME140.dll"]}
    classified = [_vc("_internal/VCRUNTIME140.dll", "python")]
    calls: dict[str, list[str]] = {}

    def run(name, argv, **kwargs):  # PyInstaller, bundle.py imports and notices.py
        calls[name] = [str(arg) for arg in argv]
        if name == "pyinstaller":
            _file(maker.dist / bundle.EXE_NAME)
            _file(maker.dist / "_internal" / "proteia" / "web" / "static" / "index.html")
            for dist in bundle.RECORD_DISTRIBUTIONS:
                _file(
                    maker.dist / "_internal" / f"{dist.replace('-', '_')}-1.dist-info" / "METADATA"
                )
            entries = [vc.to_json() for vc in classified]
            _file(manifest, json.dumps({"proteia": "0.1", bundle.VC_RUNTIME_KEY: entries}))
        elif name == "pe-imports":
            bundle.write_imports(Path(argv[-1]), imports, {})
        return ""

    monkeypatch.setattr(maker, "run", run)
    maker.freeze("0.1.0.dev0", tmp_path / "license.txt")
    assert calls["notices"][-4:] == [
        "--manifest",
        str(manifest),
        "--inno-license",
        str(tmp_path / "license.txt"),
    ]
    assert maker.summary["bundle_manifest"]["vc_runtime"] == [
        {"path": "_internal/VCRUNTIME140.dll", "kind": "python"}
    ]
    # Without that classification the build stops before the notices.
    classified, calls = [], {}
    with pytest.raises(build.BuildError, match="VCRUNTIME140.dll: a Visual C.+does not list"):
        maker.freeze("0.1.0.dev0", None)
    assert "notices" not in calls and maker.summary["bundle_manifest"]["vc_runtime"] == []


# --- The build environment and --reuse-env ---

_PY313 = {"path": r"c:\uv\cpython-3.13.14\python.exe", "version": "3.13.14 (main) [MSC v.1944]"}
_PY312 = {"path": r"c:\uv\cpython-3.12.13\python.exe", "version": "3.12.13 (main) [MSC v.1944]"}
_LOCKED = b"numpy==2.4.6 \\\n    --hash=sha256:aa\n"


def test_the_build_environment_is_reused_only_when_its_stamp_matches():
    wanted = build.env_stamp(_LOCKED, _PY313)
    assert build.env_reuse_problem(wanted, wanted) is None
    # No stamp: the install never finished, or the environment predates stamps.
    assert "no stamp" in build.env_reuse_problem(None, wanted)
    assert "no stamp" in build.env_reuse_problem(["not", "a", "stamp"], wanted)
    # A stale stamp: the lock changed since.
    stale = build.env_stamp(b"numpy==2.4.5 \\\n    --hash=sha256:bb\n", _PY313)
    assert "requirements" in build.env_reuse_problem(stale, wanted)
    # Another interpreter: --python asks for 3.12, the environment has 3.13.
    problem = build.env_reuse_problem(wanted, build.env_stamp(_LOCKED, _PY312))
    assert "3.13.14" in problem and "3.12.13" in problem
    elsewhere = build.env_stamp(_LOCKED, dict(_PY313, path=r"c:\other\python.exe"))
    assert build.env_reuse_problem(wanted, elsewhere) is not None
    # An interpreter that cannot be identified is never taken for the same one.
    unknown = build.env_stamp(_LOCKED, None)
    assert "not be identified" in build.env_reuse_problem(unknown, unknown)


def test_an_environment_whose_install_stopped_is_made_again(tmp_path, monkeypatch):
    monkeypatch.setenv("UV", "uv")
    work = tmp_path / "work"
    maker = build.Build(work, tmp_path / "out")
    requirements = _file(work / "requirements.txt", _LOCKED.decode())
    py313, py312 = _file(tmp_path / "3.13" / "python.exe"), _file(tmp_path / "3.12" / "python.exe")
    made_from: dict[str, Path] = {}
    calls: list[str] = []
    stop_install = [True]

    def run(name, argv, **kwargs):
        calls.append(name)
        if name == "uv-venv":  # --clear: the folder is made anew
            shutil.rmtree(work / "env", ignore_errors=True)
            _file(maker.env_python)
            made_from["python"] = Path(argv[argv.index("--python") + 1])
        elif name == "uv-install" and stop_install:
            stop_install.clear()
            raise build.BuildError("uv-install failed (2)")
        return ""

    def identify(python):
        python = Path(python)
        base = made_from["python"] if python == maker.env_python else python
        return {py313: _PY313, py312: _PY312}[base]

    monkeypatch.setattr(maker, "run", run)
    monkeypatch.setattr(build, "identify_interpreter", identify)
    with pytest.raises(build.BuildError):
        maker.make_environment(str(py313), requirements, reuse=True)
    assert maker.env_python.is_file()  # half made, and unstamped
    assert maker.make_environment(str(py313), requirements, reuse=True).startswith("made")
    assert calls == ["uv-venv", "uv-install"] * 2
    assert maker.make_environment(str(py313), requirements, reuse=True) == "reused"
    # --python names another interpreter: the environment is made on it.
    assert maker.make_environment(str(py312), requirements, reuse=True).startswith("made")
    assert made_from["python"] == py312
    assert maker.make_environment(str(py312), requirements, reuse=True) == "reused"
    _file(requirements, "numpy==2.4.5\n")
    assert "requirements" in maker.make_environment(str(py312), requirements, reuse=True)
    assert maker.make_environment(str(py312), requirements, reuse=False).startswith("made")
    assert calls.count("uv-venv") == 5


# --- The notices ---


def test_the_notices_table_names_every_direct_runtime_dependency():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    direct = {
        bundle.normalize(re.match(r"[A-Za-z0-9._-]+", requirement).group())
        for requirement in pyproject["project"]["dependencies"]
    }
    table = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| ([^|]+?) \| [^|]+ \|$", table, flags=re.MULTILINE)
    assert rows[:2] == ["Package", "---"]
    assert direct - {bundle.normalize(name) for name in rows[2:]} == set()


def _dist_info(contents: Path, name: str, version: str, meta: str, files: dict[str, str]) -> None:
    folder = contents / f"{name}-{version}.dist-info"
    _file(folder / "METADATA", f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n{meta}")
    record_rows = [f"{path},sha256=x,1" for path in files] + [f"{folder.name}/METADATA,,"]
    _file(folder / "RECORD", "\n".join(record_rows) + "\n")
    for path, text in files.items():
        if path.startswith(folder.name):
            _file(contents / path, text)


@pytest.fixture
def stand_in_bundle(tmp_path) -> Path:
    """A bundle folder shaped like PyInstaller's, with three packages whose
    metadata gives their licenses in the three usual ways."""
    root = tmp_path / "Proteia µ"
    contents = root / "_internal"
    _dist_info(
        contents,
        "alpha",
        "1.0",
        "License-Expression: MIT\nProject-URL: Source, https://example.org/alpha\n",
        {"alpha/__init__.py": "", "alpha-1.0.dist-info/licenses/LICENSE": "alpha license µ"},
    )
    _dist_info(
        contents,
        "beta",
        "2.0",
        "Classifier: License :: OSI Approved :: BSD License\n",
        {"beta/x.pyd": "", "beta.libs/lib.dll": "", "beta-2.0.dist-info/LICENSE.txt": "beta text"},
    )
    _dist_info(
        contents,
        "gamma",
        "0.1",
        "License: Gamma License\n        whole text, line two\n",
        {"gamma/__init__.py": ""},
    )
    for path in ("alpha/_core.pyd", "beta/x.pyd", "beta.libs/lib.dll", "proteia/web/app.pyc"):
        _file(contents / path)
    for name in ("_ssl.pyd", "python313.dll", "VCRUNTIME140.dll", "base_library.zip"):
        _file(contents / name)
    _file(contents / "beta.libs" / "msvcp140-0123abcd.dll")
    fonts = contents / "matplotlib" / "mpl-data" / "fonts"
    _file(fonts / "ttf" / "DejaVuSans.ttf")
    _file(fonts / "ttf" / "LICENSE_DEJAVU", "DejaVu license")
    _file(fonts / "pdfcorefonts" / "readme.txt", "Adobe AFM terms")
    _file(fonts / "pdfcorefonts" / "Courier.afm")
    _file(root / bundle.EXE_NAME)
    # matplotlib's metadata, so the audit accounts for its folder.
    _dist_info(contents, "matplotlib", "3.10.9", "License-Expression: PSF-2.0\n", {
        "matplotlib/__init__.py": "", "matplotlib-3.10.9.dist-info/LICENSE": "mpl license"
    })  # fmt: skip
    return root


def test_each_package_is_read_with_its_license_texts(stand_in_bundle):
    packages = {p.name: p for p in notices.read_packages(stand_in_bundle / "_internal")}
    assert list(packages) == ["alpha", "beta", "gamma", "matplotlib"]
    alpha, beta, gamma = packages["alpha"], packages["beta"], packages["gamma"]
    assert (alpha.license, alpha.urls, alpha.texts) == (
        "MIT",
        ("https://example.org/alpha",),
        (("licenses/LICENSE", "alpha license µ"),),
    )
    assert alpha.top_level == {"alpha"}
    assert beta.license == "BSD License" and beta.texts == (("LICENSE.txt", "beta text"),)
    assert beta.top_level == {"beta", "beta.libs"}
    # A license given as a whole text in the metadata is that text.
    assert gamma.license == "see the license text below"
    assert gamma.texts == (("METADATA (License)", "Gamma License\n        whole text, line two"),)


def test_the_audit_accepts_a_bundle_every_entry_of_which_is_accounted_for(stand_in_bundle):
    contents = stand_in_bundle / "_internal"
    packages = notices.read_packages(contents)
    assert notices.audit(contents, packages, {"_ssl"}) == []
    _file(contents / "mystery" / "x.pyd")
    _file(contents / "odd.dll")
    assert notices.audit(contents, packages, {"_ssl"}) == [
        "_internal/mystery is accounted for by no package or runtime",
        "_internal/odd.dll is accounted for by no package or runtime",
    ]


def test_the_audit_refuses_a_package_without_a_license_text(stand_in_bundle):
    contents = stand_in_bundle / "_internal"
    _dist_info(contents, "delta", "1", "Summary: none\n", {"delta/__init__.py": ""})
    _file(contents / "delta" / "__init__.py")
    problems = notices.audit(contents, notices.read_packages(contents), {"_ssl"})
    assert problems == ["delta 1 carries no license text"]


def test_the_notices_hold_every_section_and_text(stand_in_bundle):
    contents = stand_in_bundle / "_internal"
    runtime = notices.Runtime(
        version="3.13.14",
        libraries=(("OpenSSL", "3.5.7"), ("libffi", None)),
        license_text="PSF LICENSE TEXT",
        supplements=(("cpython-3.13-incorporated.txt", "INCORPORATED"),),
    )
    vc_files = notices.vc_runtime_files(stand_in_bundle)
    assert vc_files == [
        "_internal/VCRUNTIME140.dll",
        "_internal/beta.libs/msvcp140-0123abcd.dll",
    ]
    vc = [notices.VcSource(name, bundle.VcKind.PYTHON, f"source of {name}") for name in vc_files]
    fonts = notices.font_notes(contents)
    assert fonts[1] == [
        ("matplotlib/mpl-data/fonts/pdfcorefonts/readme.txt", "Adobe AFM terms"),
        ("matplotlib/mpl-data/fonts/ttf/LICENSE_DEJAVU", "DejaVu license"),
    ]
    arguments = dict(
        proteia_version="0.1.0.dev0",
        packages=notices.read_packages(contents),
        runtime=runtime,
        vc_files=vc,
        fonts=fonts,
        pyinstaller=("6.22.3", "PYINSTALLER COPYING"),
        freetype=["Pillow (FreeType 2.14.3)"],
    )
    text = notices.render(**arguments, inno_license="INNO LICENSE")
    for expected in (
        "Third-party notices for Proteia 0.1.0.dev0 for Windows",
        "CPython 3.13.14 (PSF-2.0), with OpenSSL 3.5.7, libffi.",
        "PSF LICENSE TEXT",
        "INCORPORATED",
        "_internal/beta.libs/msvcp140-0123abcd.dll",
        "alpha 1.0  ",
        "alpha license µ",
        "beta text",
        "whole text, line two",
        "DejaVu license",
        "Adobe AFM terms",
        "PYINSTALLER COPYING",
        "6. Inno Setup",
        "INNO LICENSE",
        "FreeType is built into Pillow (FreeType 2.14.3)",
    ):
        assert expected in text, expected
    assert "Inno Setup" not in notices.render(**arguments)
    assert all(len(line) <= 78 for line in text.splitlines()[:30])


def test_each_vc_runtime_file_is_given_where_the_spec_found_it(stand_in_bundle):
    # Only the Python build's own copies fall under its LICENSE.txt's conditions,
    # and a package's under its wheel's terms; a file the spec found elsewhere
    # (System32's MSVCP140.dll) or did not classify is refused (#54). The spec's
    # classification decides, not the file's place or bytes.
    contents = stand_in_bundle / "_internal"
    _file(contents / "MSVCP140.dll")  # from System32, though the spec left it out
    _file(contents / "MSVCP140_1.dll")  # never classified
    _file(contents / "gamma.libs" / "vcomp140.dll")  # by a folder no package's RECORD lists
    classified = [
        _vc("_internal/vcruntime140.dll", "python"),  # in another case than the file
        _vc("_internal/beta.libs/msvcp140-0123abcd.dll", "wheel", folder="beta.libs"),
        _vc("_internal/MSVCP140.dll", "elsewhere", source=r"C:\Windows\System32\msvcp140.dll"),
        _vc("_internal/gamma.libs/vcomp140.dll", "wheel", folder="gamma.libs"),
    ]
    packages = notices.read_packages(contents)
    files = notices.vc_runtime_files(stand_in_bundle)
    vc, problems = notices.vc_runtime_sources(files, classified, packages)
    assert [(entry.name, entry.kind, entry.text) for entry in vc] == [
        (
            "_internal/VCRUNTIME140.dll",
            "python",
            "from the Python runtime's Windows build (section 1)",
        ),
        (
            "_internal/beta.libs/msvcp140-0123abcd.dll",
            "wheel",
            "from the beta 2.0 wheel (section 3)",
        ),
    ]
    assert [type(entry.kind) for entry in vc] == [bundle.VcKind] * 2
    assert problems == [
        "_internal/MSVCP140.dll: from C:\\Windows\\System32\\msvcp140.dll, which is neither"
        " the Python build's folder nor a package's" + _NOT_CARRIED,
        "_internal/MSVCP140_1.dll: the spec did not classify it" + _NOT_CARRIED,
        "_internal/gamma.libs/vcomp140.dll: from the package folder gamma.libs, which no"
        " package's RECORD in the bundle lists" + _NOT_CARRIED,
    ]
    # A wheel's copy whose folder a package's RECORD lists is that package's.
    gamma_dll = ["_internal/gamma.libs/vcomp140.dll"]
    for folder in ("beta.libs", "Beta.LIBS"):  # the folder in any case its RECORD spells it
        gamma = [_vc(gamma_dll[0], "wheel", folder=folder)]
        vc, problems = notices.vc_runtime_sources(gamma_dll, gamma, packages)
        assert (vc[0].text, problems) == ("from the beta 2.0 wheel (section 3)", []), folder
    # A wheel's entry without its folder (a damaged manifest) is refused, not a crash.
    vc, problems = notices.vc_runtime_sources(gamma_dll, [_vc(gamma_dll[0], "wheel")], packages)
    assert vc == [] and "from the package folder None, which no package's RECORD" in problems[0]
    text = notices.render(
        proteia_version="0.1.0.dev0",
        packages=packages,
        runtime=notices.Runtime("3.13.14", (), "PSF", ()),
        vc_files=notices.vc_runtime_sources(files, classified, packages)[0],
        fonts=notices.font_notes(contents),
        pyinstaller=("6.22.3", "COPYING"),
        freetype=["Pillow"],
    )
    start = text.index("2. Microsoft Visual C++ runtime\n=")
    section = text[start : text.index("3. Python packages (", start)]
    assert notices.VC_REDIST_TERMS in section and '"Distributable Code"' in section
    assert "under the\nconditions in section 1" not in section
    lines = section.splitlines()
    vcruntime = lines.index("  _internal/VCRUNTIME140.dll")
    assert "Python runtime's Windows build" in lines[vcruntime + 1]
    assert "  _internal/beta.libs/msvcp140-0123abcd.dll" in lines
    assert "MSVCP140.dll" not in section and "System32" not in section
    assert all(len(line) <= 78 for line in lines)


_NOT_CARRIED = (
    "; the bundle carries only the Python runtime's and the packages' own copies"
    " of the Visual C++ runtime (#54)"
)


def _manifest(folder: Path, *classified) -> Path:
    """A bundle manifest in ``folder`` that records ``classified``."""
    entries = [vc.to_json() for vc in classified]
    return _file(folder / "bundle-manifest.json", json.dumps({bundle.VC_RUNTIME_KEY: entries}))


@pytest.mark.parametrize(
    "name", ["MSVCP140.dll", "concrt140.dll", "vcomp140.dll", "vcruntime140_threads.dll"]
)
def test_the_notices_refuse_a_vc_runtime_file_the_spec_left_out(
    stand_in_bundle, tmp_path, capsys, name
):
    # On its own: the notices step refuses it even if the bundle check let it by.
    _file(stand_in_bundle / "_internal" / name, "the redistributable's")
    manifest = _manifest(
        tmp_path,
        _vc("_internal/VCRUNTIME140.dll", "python"),
        _vc("_internal/beta.libs/msvcp140-0123abcd.dll", "wheel", folder="beta.libs"),
        _vc(f"_internal/{name}", "elsewhere", source=rf"C:\Windows\System32\{name.lower()}"),
    )
    argv = [str(stand_in_bundle), "--version", "0.1", "--manifest", str(manifest)]
    assert notices.main(argv) == 1
    err = capsys.readouterr().err
    assert "notices: the bundle cannot be covered:" in err
    assert f"_internal/{name}: from C:\\Windows\\System32\\{name.lower()}, which is" in err
    assert "_internal/VCRUNTIME140.dll" not in err
    assert not (stand_in_bundle / notices.NOTICES_FILE).exists()


def test_the_notices_need_the_bundle_manifest(stand_in_bundle, tmp_path, capsys):
    manifest = tmp_path / "bundle-manifest.json"
    argv = [str(stand_in_bundle), "--version", "0.1", "--manifest", str(manifest)]
    for text, why in (
        (None, "No such file"),
        ("{not json", "Expecting property name"),
        ('{"vc_runtime": [{"path": "_internal/x.dll", "kind": "system"}]}', "'system'"),
        ('{"vc_runtime": [{"kind": "python"}]}', "'path'"),  # an entry without its path
        ('{"vc_runtime": ["_internal/x.dll"]}', "string indices"),  # not an entry
        ("[]", "'list' object has no attribute 'get'"),  # not a manifest
    ):
        if text is not None:
            _file(manifest, text)
        assert notices.main(argv) == 1
        err = capsys.readouterr().err
        assert err.startswith(f"notices: the bundle manifest {manifest} cannot be read: "), err
        assert why in err
        assert not (stand_in_bundle / notices.NOTICES_FILE).exists()
    with pytest.raises(SystemExit) as stop:
        notices.main([str(stand_in_bundle), "--version", "0.1"])
    assert stop.value.code == 2
    assert "--manifest" in capsys.readouterr().err


def test_the_notices_list_the_vc_runtime_by_the_names_the_bundle_knows(tmp_path):
    root = tmp_path / "Proteia"
    listed = [
        "_internal/concrt140.dll",
        "_internal/pkg.libs/msvcp140-0a1b.dll",
        "_internal/pkg/VCOMP140.DLL",
        "_internal/vcamp140.dll",
        "_internal/vccorlib140.dll",
        "_internal/vcruntime140_threads.dll",
    ]
    for path in [*listed, "_internal/msvcp140_clr0400.dll", "_internal/msvcp_win.dll"]:
        _file(root / path)
    _file(root / "_internal" / "msvcp140.dll.txt")
    (root / "_internal" / "odd" / "vcomp140.dll").mkdir(parents=True)  # a folder
    assert notices.vc_runtime_files(root) == listed


def test_a_missing_inno_setup_license_is_reported_without_a_traceback(
    stand_in_bundle, tmp_path, capsys
):
    missing = tmp_path / "shims" / "license.txt"
    manifest = _manifest(tmp_path)
    argv = [str(stand_in_bundle), "--version", "0.1", "--inno-license", str(missing)]
    assert notices.main([*argv, "--manifest", str(manifest)]) == 1
    assert "Inno Setup's license was not found" in capsys.readouterr().err
    assert not (stand_in_bundle / notices.NOTICES_FILE).exists()


def test_the_runtime_supplements_name_the_libraries_built_into_python():
    incorporated = (notices.SUPPLEMENTS / "cpython-3.13-incorporated.txt").read_text("utf-8")
    for section in ("OpenSSL", "expat", "libffi", "zlib", "libmpdec", "Mersenne Twister"):
        assert re.search(rf"^{section}\n-+$", incorporated, re.MULTILINE), section
    xz = (notices.SUPPLEMENTS / "xz-liblzma.txt").read_text("utf-8")
    assert "Permission to use, copy, modify, and/or distribute this" in xz


@pytest.mark.skipif(os.name != "nt", reason="the Windows runtime, which the bundle carries")
def test_the_runtime_is_this_interpreter():
    runtime = notices.build_runtime()
    assert runtime.version == ".".join(map(str, sys.version_info[:3]))
    assert "PYTHON SOFTWARE FOUNDATION LICENSE" in runtime.license_text
    assert "Microsoft Distributable Code" in runtime.license_text


def test_pyinstallers_license_is_read_from_its_metadata():
    pytest.importorskip("PyInstaller", reason="the build group is not installed")
    version, text = notices.pyinstaller_license()
    assert version == "6.22.3"
    assert "bootloader" in text.lower()


# --- The install check's scratch folder ---


@pytest.fixture
def install_check(monkeypatch):
    """install_check.py with its checks replaced by a stand-in that records the
    folder each run was given; nothing is installed."""
    if os.name != "nt":
        pytest.skip("install_check.py runs on Windows (winreg)")
    [module] = _load("install_check")
    runs: list[Path] = []

    class StandInCheck:
        def __init__(self, setup: Path, folder: Path) -> None:
            runs.append(folder)

        def run(self, expected):
            return {"ok": True, "failures": [], "expected": expected}

    monkeypatch.setattr(module, "Check", StandInCheck)
    return module, runs


def test_the_install_check_never_empties_a_folder_it_did_not_make(install_check, tmp_path):
    module, runs = install_check
    setup = _file(tmp_path / "dist" / "Proteia-0.1.0.dev0-setup.exe")
    # The build's work folder, with the report --expect-numbers reads.
    work = tmp_path / "build" / "windows"
    report = _file(work / "selftest-frozen" / "report.json", json.dumps({"numbers": {"a": 1}}))
    _file(work / "dist" / "Proteia" / bundle.EXE_NAME)
    with pytest.raises(SystemExit):
        module.main([str(setup), str(work), "--expect-numbers", str(report)])
    # Any other folder that holds something, a user's say.
    documents = tmp_path / "documents µ"
    thesis = _file(documents / "thesis.docx", "years of work")
    with pytest.raises(SystemExit):
        module.main([str(setup), str(documents)])
    assert runs == []
    assert report.is_file() and (work / "dist" / "Proteia" / bundle.EXE_NAME).is_file()
    assert thesis.read_text(encoding="utf-8") == "years of work"


def test_the_install_check_empties_only_its_own_folder(install_check, tmp_path):
    module, runs = install_check
    setup = _file(tmp_path / "Proteia-0.1.0.dev0-setup.exe")
    report = _file(tmp_path / "report.json", json.dumps({"numbers": {"a": 1}}))
    scratch = tmp_path / "scratch"
    assert module.main([str(setup), str(scratch), "--expect-numbers", str(report)]) == 0
    assert (scratch / module.MARKER).is_file()
    left = _file(scratch / "app" / "left from the last run.txt")
    # A missing installer or an unreadable report stops the check before the
    # folder is touched.
    for argv in (
        [str(tmp_path / "no-setup.exe"), str(scratch)],
        [str(setup), str(scratch), "--expect-numbers", str(tmp_path / "no-report.json")],
        [str(setup), str(scratch), "--expect-numbers", str(setup)],
        [str(left), str(scratch)],  # an installer inside the folder it would empty
    ):
        with pytest.raises(SystemExit):
            module.main(argv)
        assert left.is_file(), argv
    assert module.main([str(setup), str(scratch)]) == 0
    assert not left.exists() and (scratch / module.MARKER).is_file()
    # An empty folder is used as it is.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert module.main([str(setup), str(empty)]) == 0
    assert runs == [scratch, scratch, empty]


# --- The install check's clean-up ---


class _Process:
    """A stand-in for a Proteia the check starts: it runs until it is killed."""

    def __init__(self, pid: int) -> None:
        self.pid, self.returncode = pid, None

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.returncode = 1

    def wait(self, timeout=None):
        return self.returncode


class _Windows:
    """What the install check runs, faked: the installer and the uninstaller only
    make and remove files and the per-user uninstall entry, the self-test passes,
    and a started Proteia never answers."""

    def __init__(self, programs: Path) -> None:
        self.shortcut = programs / f"{bundle.APP_NAME}.lnk"
        self.registry: dict[str, dict[str, str]] = {}
        self.runs: list[list[str]] = []
        self.started: list[_Process] = []
        self.install_error: Exception | None = None
        self.uninstall_leaves_shortcut = False

    def run(self, argv, **kwargs):
        argv = [str(arg) for arg in argv]
        self.runs.append(argv)
        exe = Path(argv[0])
        if exe.name == "unins000.exe":
            shutil.rmtree(exe.parent)
            if not self.uninstall_leaves_shortcut:
                self.shortcut.unlink(missing_ok=True)
            self.registry.clear()
        elif "--self-test" in argv:
            report = {"ok": True, "steps": [], "numbers": {}}
            _file(Path(argv[argv.index("--json") + 1]), json.dumps(report))
        else:  # the installer
            if self.install_error is not None:
                raise self.install_error
            app = Path(next(arg for arg in argv if arg.startswith("/DIR="))[len("/DIR=") :])
            for name in (bundle.EXE_NAME, "LICENSE.txt", "THIRD_PARTY_NOTICES.txt", "unins000.exe"):
                _file(app / name)
            _file(self.shortcut)
            self.registry["entry"] = {"DisplayName": bundle.APP_NAME, "InstallLocation": str(app)}
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def popen(self, argv, **kwargs) -> _Process:
        self.started.append(_Process(7000 + len(self.started)))
        return self.started[-1]


@pytest.fixture
def faked_check(tmp_path, monkeypatch):
    """install_check.py on a faked Windows (see _Windows): nothing is installed,
    nothing runs."""
    if os.name != "nt":
        pytest.skip("install_check.py runs on Windows (winreg)")
    [module] = _load("install_check")
    windows = _Windows(tmp_path / "scratch" / "programs")
    fake = types.SimpleNamespace(
        run=windows.run,
        Popen=windows.popen,
        DEVNULL=subprocess.DEVNULL,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    monkeypatch.setattr(module, "subprocess", fake)
    monkeypatch.setattr(module, "uninstall_entry", lambda app_id: windows.registry.get("entry"))
    monkeypatch.setattr(
        module,
        "known_folders",
        lambda env=None: {
            name: tmp_path / ("scratch" if env else "real") / name
            for name in ("documents", "programs", "desktop")
        },
    )
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "real" / "localappdata"))

    def no_answer(proc, state, start):
        raise RuntimeError("no answer from /api/status in 90 s")

    monkeypatch.setattr(smoke, "check_stand_in", lambda browser, log, env: None)
    monkeypatch.setattr(smoke, "run", lambda exe, folder, runs: {"ok": True, "launches": []})
    monkeypatch.setattr(smoke, "wait_for_status", no_answer)
    folder = tmp_path / "check µ"
    module.prepare_folder(folder)
    return module, windows, module.Check(tmp_path / "Proteia-0.1.0.dev0-setup.exe", folder)


def test_a_check_that_stops_midway_ends_what_it_started_and_uninstalls(faked_check):
    module, windows, check = faked_check
    report = check.run(None)  # step 3's Proteia never answers
    assert report["ok"] is False
    assert report["stopped"] == {
        "step": "3. upgrade while running",
        "error": "RuntimeError: no answer from /api/status in 90 s",
        "not_run": [
            "4. refused while unreachable",
            "5. uninstall while running",
            "6. uninstall after a crash",
        ],
    }
    assert any("3. upgrade while running" in failure for failure in report["failures"])
    [proc] = windows.started
    assert proc.poll() is not None
    uninstaller = windows.runs[-1]
    assert Path(uninstaller[0]) == check.app / "unins000.exe"
    assert set(module.SILENT) <= set(uninstaller)
    assert report["cleanup"]["stopped"] == [proc.pid]
    assert report["cleanup"]["uninstalled"]["exit"] == 0
    assert report["cleanup"]["left"] == []
    assert windows.registry == {} and not windows.shortcut.exists()
    assert not [failure for failure in report["failures"] if failure.startswith("clean-up")]
    json.dumps(report, default=str)  # the report is still written


def test_the_clean_up_says_what_it_could_not_remove(faked_check):
    module, windows, check = faked_check
    windows.uninstall_leaves_shortcut = True
    report = check.run(None)
    assert report["cleanup"]["left"] == [str(windows.shortcut)]
    assert any(
        failure.startswith("clean-up") and str(windows.shortcut) in failure
        for failure in report["failures"]
    )


def test_the_clean_up_runs_no_uninstaller_the_check_did_not_install(faked_check):
    module, windows, check = faked_check
    # The installer timed out having installed nothing: nothing to clean up.
    windows.install_error = subprocess.TimeoutExpired("setup", 900)
    report = check.run(None)
    assert report["stopped"]["step"] == "1. install"
    assert report["stopped"]["error"].startswith("TimeoutExpired: ")
    assert len(report["stopped"]["not_run"]) == 5
    assert [Path(argv[0]) for argv in windows.runs] == [check.setup]
    assert report["cleanup"] == {"stopped": [], "uninstalled": None, "left": []}
    # Proteia is installed for this account already: the check refuses before
    # it installs anything, and runs no uninstaller.
    windows.registry["entry"] = {"DisplayName": bundle.APP_NAME}
    with pytest.raises(SystemExit, match="uninstall it first"):
        module.Check(check.setup, check.folder).run(None)
    assert len(windows.runs) == 1 and windows.registry


# --- The launch check's stand-in browser ---


def test_the_stand_in_browser_survives_webbrowsers_split(tmp_path):
    log = tmp_path / "a folder µ" / "browser.log"
    log.parent.mkdir()
    value = smoke.stand_in_browser(log)
    assert "%s" in value and "\\" not in value
    command = webbrowser.get(value)
    assert isinstance(command, webbrowser.GenericBrowser)
    assert [command.name, *command.args] == [
        Path(sys.executable).as_posix(),
        Path(smoke.__file__).as_posix(),
        smoke.STAND_IN_FLAG,
        log.as_posix(),
        "%s",
    ]


def test_the_stand_in_browser_records_the_address_and_never_fails(tmp_path):
    log = tmp_path / "browser µ.log"
    smoke.check_stand_in(smoke.stand_in_browser(log), log, dict(os.environ))
    assert not log.exists()  # the check's own line is removed
    assert smoke.main([smoke.STAND_IN_FLAG, str(log), "file:///x/open-proteia.html"]) == 0
    assert log.read_text(encoding="utf-8") == "file:///x/open-proteia.html\n"
    assert smoke.main([smoke.STAND_IN_FLAG, str(tmp_path / "no" / "such" / "log"), "u"]) == 0


def test_the_launch_check_reads_the_session_log_the_launcher_writes(tmp_path, monkeypatch):
    assert (smoke.LOG_DIR, smoke.LOG_FILE) == (logs.LOG_DIR, logs.LOG_FILE)
    state = tmp_path / "state µ"
    monkeypatch.setattr(launch, "state_dir", lambda: state)
    token = "t" * 43

    class Served:  # the first launch, which serves until Quit
        port = 1234
        redirect_path = state / launch.REDIRECT_FILE
        taken = 0
        unread = ()

        def serve(self) -> None:
            pass

    monkeypatch.setattr(launch, "start", lambda **kwargs: Served())
    assert launch.main([]) == 0
    # the second opens the first
    monkeypatch.setattr(launch, "start", lambda **kwargs: launch.Opened())
    assert launch.main([]) == 0
    assert [path.name for path in state.iterdir()] == [smoke.LOG_DIR]
    assert smoke.session_log_problems(state, token, sessions=2) == []
    assert smoke.session_log_problems(state, token, sessions=3) == [
        "the session log records 'session started' 2 times, not 3",
        "the session log records 'session ended' 2 times, not 3",
    ]
    with (state / smoke.LOG_DIR / smoke.LOG_FILE).open("a", encoding="utf-8") as f:
        f.write(f"#token={token}\n")
    assert smoke.session_log_problems(state, token, sessions=2) == [
        "the session log holds the access token"
    ]
    [missing] = smoke.session_log_problems(tmp_path / "no state", token, sessions=2)
    assert missing.startswith("the session log cannot be read: FileNotFoundError")


def test_a_launch_gets_a_restricted_environment(tmp_path):
    env = smoke.restricted_env(tmp_path, "stand-in %s")
    assert env["BROWSER"] == "stand-in %s"
    assert env["LOCALAPPDATA"] == str(tmp_path / "localappdata")
    assert env["TEMP"] == env["TMP"] and Path(env["TEMP"]).is_dir()
    assert all("python" not in part.lower() for part in env["PATH"].split(";"))
    assert "BROWSER" not in smoke.restricted_env(tmp_path)
    json.dumps(env)
