# SPDX-License-Identifier: Apache-2.0
"""The Windows installer's build files (#54, ADR 0003), in the parts that run
without PyInstaller or Inno Setup: the spec's lists against the code they
describe, the installer script against the launcher, the build's version and
bundle checks, the notices generator on a stand-in bundle, and the launch
check's stand-in browser. A test that needs PyInstaller (the ``build`` group)
skips without it."""

import ast
import importlib
import json
import os
import re
import shutil
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
    excluded = set(bundle.EXCLUDES)
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
    assert build.bundle_problems(dist) == [
        "no metadata of tifffile, whose version the record reads"
    ]
    _file(dist / "_internal" / "tifffile-2026.5.15.dist-info" / "METADATA")
    assert build.bundle_problems(dist) == []
    _file(dist / "_internal" / "PySide6" / "QtCore.pyd")
    _file(dist / "_internal" / "api-ms-win-crt-heap-l1-1-0.dll")
    assert build.bundle_problems(dist) == [
        "_internal/PySide6 must not be in the bundle",
        "_internal/PySide6/QtCore.pyd must not be in the bundle",
        "_internal/api-ms-win-crt-heap-l1-1-0.dll: a Universal CRT DLL",
    ]


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
    vc = [(name, f"source of {name}") for name in vc_files]
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


def test_each_vc_runtime_file_is_given_where_it_came_from(stand_in_bundle, tmp_path):
    # Only the Python build's own copies fall under its LICENSE.txt's conditions;
    # MSVCP140.dll comes from the build machine, NumPy's copy from its wheel.
    contents = stand_in_bundle / "_internal"
    python_dir, system_dir = tmp_path / "python", tmp_path / "System32"
    _file(contents / "VCRUNTIME140.dll", "the python build's")
    _file(python_dir / "vcruntime140.dll", "the python build's")
    _file(system_dir / "vcruntime140.dll", "a newer one")
    _file(contents / "MSVCP140.dll", "the redistributable's")
    _file(system_dir / "msvcp140.dll", "the redistributable's")
    _file(contents / "MSVCP140_1.dll", "from nowhere known")
    packages = notices.read_packages(contents)
    files = notices.vc_runtime_files(stand_in_bundle)
    vc = notices.vc_runtime_sources(stand_in_bundle, files, packages, python_dir, system_dir)
    sources = dict(vc)
    assert "Python runtime's Windows build" in sources["_internal/VCRUNTIME140.dll"]
    assert "System32" in sources["_internal/MSVCP140.dll"]
    assert "Python build does not include it" in sources["_internal/MSVCP140.dll"]
    assert (
        sources["_internal/beta.libs/msvcp140-0123abcd.dll"]
        == "from the beta 2.0 wheel (section 3)"
    )
    assert sources["_internal/MSVCP140_1.dll"] == "source not identified"
    text = notices.render(
        proteia_version="0.1.0.dev0",
        packages=packages,
        runtime=notices.Runtime("3.13.14", (), "PSF", ()),
        vc_files=vc,
        fonts=notices.font_notes(contents),
        pyinstaller=("6.22.3", "COPYING"),
        freetype=["Pillow"],
    )
    start = text.index("2. Microsoft Visual C++ runtime\n=")
    section = text[start : text.index("3. Python packages (", start)]
    assert notices.VC_REDIST_TERMS in section and '"Distributable Code"' in section
    assert "under the\nconditions in section 1" not in section
    lines = section.splitlines()
    msvcp = lines.index("  _internal/MSVCP140.dll")
    assert "System32" in lines[msvcp + 1]
    assert all(len(line) <= 78 for line in lines)


def test_a_missing_inno_setup_license_is_reported_without_a_traceback(
    stand_in_bundle, tmp_path, capsys
):
    missing = tmp_path / "shims" / "license.txt"
    argv = [str(stand_in_bundle), "--version", "0.1", "--inno-license", str(missing)]
    assert notices.main(argv) == 1
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
