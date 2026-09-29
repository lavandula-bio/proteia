# SPDX-License-Identifier: Apache-2.0
"""Build Proteia's Windows installer (#54, ADR 0003).

From the repository root, on 64-bit Windows 10 or later::

    uv run --no-project --python 3.13 python packaging/windows/build.py

This script needs only the standard library, ``uv`` and Inno Setup 7.1.0; it
builds everything else from ``uv.lock``. The steps, each stopping the build when
it fails:

1. **Versions.** ``pyproject.toml``'s version must be ``proteia.__version__``;
   the installer is named after it.
2. **Build environment.** ``uv export --locked --no-dev --group build`` writes
   the locked requirements with the ``build`` group (PyInstaller, pinned); a new
   virtual environment ``WORK/env`` gets exactly those (``--no-deps``) and then
   Proteia itself, not editable, and may hold no napari and no Qt. The default
   ``uv sync`` never installs the ``build`` group. ``--python`` chooses the
   interpreter the bundle carries (default: the one running this script). Once
   those requirements are installed, a stamp (``WORK/env/build-env.json``)
   records their hash and the interpreter (real path and version);
   ``--reuse-env`` keeps ``WORK/env`` only when that stamp matches the
   requirements just exported and the interpreter ``--python`` names, so an
   install that stopped halfway, or another interpreter, makes it anew.
3. **Bundle.** PyInstaller runs ``proteia.spec`` into ``WORK/dist/Proteia``
   (log: ``WORK/pyinstaller.log``). The bundle may hold no napari, Qt or Tcl
   file and no Universal CRT DLL, and must hold the web client and the
   metadata the record reads. ``LICENSE.txt`` and ``THIRD_PARTY_NOTICES.txt``
   (``notices.py``) are added to it.
4. **Self-test.** ``Proteia.exe --self-test`` (:mod:`proteia.selftest`) runs with
   ``PATH`` limited to the Windows folders and per-user folders in ``WORK``, and
   the same test runs unfrozen in the build environment; every number must
   agree.
5. **Launch check.** ``smoke.py`` launches the bundle ``--smoke-runs`` times
   (start, token, redirect file, second launch, quit) with a stand-in browser:
   no browser opens.
6. **Installer.** Inno Setup compiles ``proteia.iss`` into ``OUT``, and
   ``OUT/<installer>.sha256`` records the installer's SHA-256 for testers.
   Inno Setup is found through ``--iscc``, the ``ISCC`` variable, ``PATH``, or
   its default folders, and its ``license.txt`` (for the notices) must be next
   to that ``ISCC.exe``, which a shim's folder lacks; the build checks both
   before it starts. Inno Setup is not installed by this script.

``WORK`` (``--work``, default ``build/windows``) and ``OUT`` (``--out``, default
``dist``) are ignored by git. Keep ``WORK`` short (under about 100 characters):
the self-test writes an export folder below it, and Windows refuses paths over
259 characters. ``WORK/build-summary.json`` records the versions, sizes, times
and results. ``install_check.py`` then installs the installer into a scratch
folder (a new or empty one, or one it made before), checks it, and uninstalls
it::

    uv run --no-project --python 3.13 python packaging/windows/install_check.py
        dist/Proteia-<version>-setup.exe <scratch folder>
        --expect-numbers build/windows/selftest-frozen/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import bundle
import smoke

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = HERE / "proteia.spec"
ISS = HERE / "proteia.iss"
NOTICES = HERE / "notices.py"
INNO_SETUP_VERSION = "7.1.0"  # proteia.iss refuses another
# Modules the build environment may not hold: napari left the dependencies in
# #57, and Qt would bring the duties of conveying it (ADR 0002).
ABSENT = ("napari", "PySide6", "qtpy")
# In WORK/env, written once its locked requirements are installed.
ENV_STAMP = "build-env.json"
# Prints which interpreter runs it, or which one a virtual environment's comes
# from: its real path (links such as uv's minor-version folders followed) and
# full version, as JSON (ASCII, whatever the console's code page).
_IDENTIFY = (
    "import json, os, sys; print(json.dumps({'path': os.path.normcase(os.path.realpath("
    "getattr(sys, '_base_executable', sys.executable))), 'version': sys.version}))"
)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class BuildError(Exception):
    """A build step failed; the message says which and why."""


def project_version(pyproject: str) -> str:
    """The version ``pyproject.toml`` (its text) declares."""
    return tomllib.loads(pyproject)["project"]["version"]


def package_version(init: str) -> str:
    """``__version__`` as ``proteia/__init__.py`` (its text) assigns it."""
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.MULTILINE)
    if match is None:
        raise BuildError("proteia/__init__.py assigns no __version__")
    return match.group(1)


def read_version(root: Path = ROOT) -> str:
    declared = project_version((root / "pyproject.toml").read_text(encoding="utf-8"))
    package = package_version((root / "src" / "proteia" / "__init__.py").read_text("utf-8"))
    if declared != package:
        raise BuildError(f"pyproject.toml says {declared}, proteia.__version__ {package}")
    return declared


def iscc_defines(version: str, dist: Path) -> dict[str, str]:
    """The ``/D`` defines ``proteia.iss`` requires."""
    return {
        "AppName": bundle.APP_NAME,
        "ExeName": bundle.EXE_NAME,
        "AppVersion": version,
        "NumericVersion": ".".join(map(str, bundle.numeric_version(version))),
        "AppPublisher": bundle.PUBLISHER,
        "AppURL": bundle.APP_URL,
        "DistDir": str(dist),
    }


def iscc_candidates(env: Mapping[str, str]) -> list[Path]:
    """Where Inno Setup 7's compiler is looked for after ``--iscc``: the ``ISCC``
    variable, ``PATH``, then the folders its installer uses by default."""
    found = [Path(env["ISCC"])] if env.get("ISCC") else []
    which = shutil.which("ISCC", path=env.get("PATH"))
    if which:
        found.append(Path(which))
    for variable, below in (
        ("ProgramFiles(x86)", ("Inno Setup 7",)),
        ("ProgramFiles", ("Inno Setup 7",)),
        ("LOCALAPPDATA", ("Programs", "Inno Setup 7")),
    ):
        if env.get(variable):
            found.append(Path(env[variable]).joinpath(*below, "ISCC.exe"))
    return found


def find_iscc(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else iscc_candidates(os.environ)
    for path in candidates:
        if path.is_file():
            return path
    raise BuildError(
        f"Inno Setup {INNO_SETUP_VERSION} was not found (looked at: "
        + ", ".join(str(path) for path in candidates)
        + "). Install it, then pass --iscc PATH\\ISCC.exe or set ISCC; or build"
        " without the installer (--no-installer)."
    )


def inno_setup_license(iscc: Path) -> Path:
    """Inno Setup's ``license.txt``, next to the real ``ISCC.exe`` (a link is
    followed); ``BuildError`` when it is not there, as next to a shim that runs
    the compiler from elsewhere (Scoop's, say), so that the build stops before it
    starts rather than in the notices step."""
    license_file = iscc.resolve().parent / "license.txt"
    if not license_file.is_file():
        raise BuildError(
            f"Inno Setup's license.txt is not next to {iscc} (looked for {license_file});"
            " the notices carry it. If that ISCC.exe is a shim, pass --iscc with the"
            " ISCC.exe in Inno Setup's own folder."
        )
    return license_file


def compare_numbers(frozen: Mapping[str, Any], unfrozen: Mapping[str, Any]) -> list[str]:
    """Where the self-test's numbers differ between the two runs."""
    keys = sorted(set(frozen) | set(unfrozen))
    return [
        f"{key}: frozen {frozen.get(key)!r}, unfrozen {unfrozen.get(key)!r}"
        for key in keys
        if frozen.get(key) != unfrozen.get(key)
    ]


def folder_size(folder: Path) -> tuple[int, int]:
    """The bytes and the number of files under ``folder``."""
    files = [path for path in folder.rglob("*") if path.is_file()]
    return sum(path.stat().st_size for path in files), len(files)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_problems(dist: Path) -> list[str]:
    """What the finished bundle folder must not lack or hold."""
    contents = dist / "_internal"
    paths = [path.relative_to(dist).as_posix() for path in dist.rglob("*")]
    problems = [f"{path} must not be in the bundle" for path in bundle.forbidden_paths(paths)]
    problems += [f"{path}: a Universal CRT DLL" for path in paths if bundle.is_universal_crt(path)]
    if not (contents / "proteia" / "web" / "static" / "index.html").is_file():
        problems.append("the web client (proteia/web/static) is missing")
    metadata = [path.name.split("-")[0] for path in contents.glob("*.dist-info")]
    problems += [
        f"no metadata of {name}, whose version the record reads"
        for name in bundle.missing_record_distributions(metadata)
    ]
    if not (dist / bundle.EXE_NAME).is_file():
        problems.append(f"no {bundle.EXE_NAME}")
    return problems


def identify_interpreter(python: str | Path) -> dict[str, str] | None:
    """The interpreter ``python`` is, or the one a virtual environment's comes
    from: ``{"path", "version"}``; None when it does not run."""
    try:
        done = subprocess.run(
            [str(python), "-c", _IDENTIFY],
            capture_output=True,
            creationflags=_NO_WINDOW,
            timeout=60,
        )
        identity = json.loads(done.stdout) if done.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return identity if isinstance(identity, dict) else None


def env_stamp(requirements: bytes, interpreter: Mapping[str, str] | None) -> dict[str, Any]:
    """The stamp of a build environment with the locked ``requirements`` (the
    exported file's bytes) installed on ``interpreter`` (None: unknown)."""
    return {
        "requirements_sha256": hashlib.sha256(requirements).hexdigest(),
        "interpreter": None if interpreter is None else dict(interpreter),
    }


def env_reuse_problem(stamp: Any, wanted: Mapping[str, Any]) -> str | None:
    """Why the build environment stamped ``stamp`` (None: it has no readable
    stamp) cannot serve a build that wants ``wanted`` (:func:`env_stamp`), or
    None when it can."""
    if not isinstance(stamp, Mapping):
        return "it has no stamp (its install did not finish, or it predates stamps)"
    if stamp.get("requirements_sha256") != wanted["requirements_sha256"]:
        return "the locked requirements changed since it was made"
    if wanted["interpreter"] is None:
        return "the interpreter --python names could not be identified"
    if stamp.get("interpreter") != wanted["interpreter"]:
        return f"it was made on {stamp.get('interpreter')}, not {wanted['interpreter']}"
    return None


def read_env_stamp(path: Path) -> Any:
    """The stamp at ``path``, or None when it is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Build:
    """One build: the paths, the log of each step, and the summary."""

    def __init__(self, work: Path, out: Path) -> None:
        self.work = work
        self.out = out
        self.summary: dict[str, Any] = {"steps": {}}
        self.uv = os.environ.get("UV") or shutil.which("uv")
        if not self.uv:
            raise BuildError("uv is not on PATH")

    @property
    def env_python(self) -> Path:
        return self.work / "env" / "Scripts" / "python.exe"

    @property
    def dist(self) -> Path:
        return self.work / "dist" / bundle.APP_NAME

    def run(self, name: str, argv: Sequence[str | Path], **kwargs: Any) -> str:
        """Run ``argv``, its output into ``WORK/<name>.log``; ``BuildError`` when
        it fails. Returns the output."""
        log = self.work / f"{name}.log"
        done = subprocess.run(
            [str(arg) for arg in argv],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=_NO_WINDOW,
            **kwargs,
        )
        output = done.stdout.decode("utf-8", "replace")
        log.write_text(output, encoding="utf-8")
        if done.returncode != 0:
            tail = "\n".join(output.splitlines()[-25:])
            raise BuildError(f"{name} failed ({done.returncode}); see {log}:\n{tail}")
        return output

    def step(self, name: str, action: Any) -> Any:
        print(f"== {name}", flush=True)
        start = time.perf_counter()
        result = action()
        self.summary["steps"][name] = round(time.perf_counter() - start, 1)
        return result

    @property
    def env_stamp_file(self) -> Path:
        return self.work / "env" / ENV_STAMP

    def requested_interpreter(self, python: str) -> dict[str, str] | None:
        """The interpreter ``--python`` names, found as ``uv venv`` finds it: a
        file is itself; any other request (``3.12``, say) is resolved among the
        installed interpreters, not virtual environments. None when not found."""
        if not Path(python).is_file():
            done = subprocess.run(
                [self.uv, "python", "find", "--system", "--no-project", python],
                capture_output=True,
                creationflags=_NO_WINDOW,
            )
            if done.returncode != 0:
                return None
            python = done.stdout.decode("utf-8", "replace").strip()
        return identify_interpreter(python)

    def make_environment(self, python: str, requirements: Path, reuse: bool) -> str:
        """Make ``WORK/env`` on ``python`` with the locked ``requirements``, or with
        ``reuse`` keep the one there when its stamp matches both. Says which."""
        locked = requirements.read_bytes()
        problem = "--reuse-env was not given"
        if reuse:
            wanted = env_stamp(locked, self.requested_interpreter(python))
            stamp = read_env_stamp(self.env_stamp_file) if self.env_python.is_file() else None
            problem = env_reuse_problem(stamp, wanted)
            if problem is None:
                return "reused"
            print(f"   making {self.work / 'env'} anew: {problem}", flush=True)
        # uv venv --clear removes the stamp too; this covers its failing first.
        self.env_stamp_file.unlink(missing_ok=True)
        self.run("uv-venv", [self.uv, "venv", "--clear", self.work / "env", "--python", python])
        self.run(
            "uv-install",
            [self.uv, "pip", "install", "--python", self.env_python, "--no-deps",
             "--require-hashes", "--requirement", requirements],
        )  # fmt: skip
        # Only now, with every requirement installed, can the environment be reused.
        made_on = identify_interpreter(self.env_python)
        if made_on is None:
            raise BuildError(f"{self.env_python} does not run")
        text = json.dumps(env_stamp(locked, made_on), indent=1)
        self.env_stamp_file.write_text(text, encoding="utf-8")
        return f"made ({problem})"

    def environment(self, python: str, reuse: bool) -> None:
        requirements = self.work / "requirements.txt"
        self.run(
            "uv-export",
            [self.uv, "export", "--locked", "--no-dev", "--group", "build", "--no-emit-project",
             "--format", "requirements.txt", "--output-file", requirements],
            cwd=ROOT,
        )  # fmt: skip
        self.summary["environment"] = self.make_environment(python, requirements, reuse)
        self.run(
            "uv-install-proteia",
            [self.uv, "pip", "install", "--python", self.env_python, "--no-deps",
             "--reinstall-package", "proteia", ROOT],
        )  # fmt: skip
        check = (
            "import importlib.util, sys; sys.exit(any(importlib.util.find_spec(n)"
            f" for n in {list(ABSENT)!r}))"
        )
        if subprocess.run([self.env_python, "-c", check], creationflags=_NO_WINDOW).returncode:
            raise BuildError("the build environment holds napari or Qt")
        self.summary["python"] = self.run(
            "python-version", [self.env_python, "-c", "import sys; print(sys.version)"]
        ).strip()

    def freeze(self, version: str, inno_license: Path | None) -> None:
        shutil.rmtree(self.work / "dist", ignore_errors=True)
        self.run(
            "pyinstaller",
            [self.env_python, "-m", "PyInstaller", "--noconfirm", "--clean",
             "--distpath", self.work / "dist", "--workpath", self.work / "pyinstaller", SPEC],
            cwd=HERE,
        )  # fmt: skip
        problems = bundle_problems(self.dist)
        if problems:
            raise BuildError("the bundle is not right:\n  " + "\n  ".join(problems))
        shutil.copyfile(ROOT / "LICENSE", self.dist / "LICENSE.txt")
        argv: list[str | Path] = [self.env_python, NOTICES, self.dist, "--version", version]
        if inno_license is not None:
            argv += ["--inno-license", inno_license]
        self.run("notices", argv)
        manifest = self.work / "pyinstaller" / "proteia" / "bundle-manifest.json"
        self.summary["bundle_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        size, files = folder_size(self.dist)
        self.summary["bundle"] = {"bytes": size, "files": files}

    def self_test(self) -> None:
        reports = {}
        for name, argv in (
            ("frozen", [self.dist / bundle.EXE_NAME, "--self-test"]),
            ("unfrozen", [self.env_python, "-m", "proteia.selftest"]),
        ):
            folder = self.work / f"selftest-{name}"
            shutil.rmtree(folder, ignore_errors=True)
            report = folder / "report.json"
            env = smoke.restricted_env(folder)
            start = time.perf_counter()
            done = subprocess.run(
                [*map(str, argv), "--json", str(report)],
                env=env,
                capture_output=True,
                creationflags=_NO_WINDOW,
                timeout=600,
            )
            seconds = round(time.perf_counter() - start, 2)
            (folder / "output.txt").write_bytes(done.stdout + b"\n" + done.stderr)
            if not report.exists():
                raise BuildError(f"the {name} self-test wrote no report ({done.returncode})")
            reports[name] = json.loads(report.read_text(encoding="utf-8"))
            failed = [step for step in reports[name]["steps"] if not step["ok"]]
            self.summary[f"selftest_{name}"] = {
                "exit": done.returncode,
                "seconds": seconds,
                "steps": {step["name"]: step["ok"] for step in reports[name]["steps"]},
            }
            if done.returncode != 0 or failed:
                errors = "; ".join(f"{step['name']}: {step['error']}" for step in failed)
                raise BuildError(f"the {name} self-test failed ({done.returncode}): {errors}")
        differences = compare_numbers(reports["frozen"]["numbers"], reports["unfrozen"]["numbers"])
        if differences:
            raise BuildError("frozen and unfrozen numbers differ:\n  " + "\n  ".join(differences))
        self.summary["selftest_numbers_agree"] = len(reports["frozen"]["numbers"])

    def smoke(self, runs: int) -> None:
        folder = self.work / "smoke"
        shutil.rmtree(folder, ignore_errors=True)
        try:
            result = smoke.run(self.dist / bundle.EXE_NAME, folder, runs)
        except RuntimeError as exc:  # the stand-in browser failed: nothing was launched
            raise BuildError(str(exc)) from exc
        (self.work / "smoke.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        self.summary["smoke"] = [
            {key: launch.get(key) for key in ("first_status_s", "quit_to_exit_s", "failures")}
            for launch in result["launches"]
        ]
        if not result["ok"]:
            raise BuildError(f"the launch check failed: {self.summary['smoke']}")

    def installer(self, iscc: Path, version: str) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        defines = [f"/D{key}={value}" for key, value in iscc_defines(version, self.dist).items()]
        self.run("iscc", [iscc, *defines, f"/O{self.out}", ISS], cwd=HERE)
        setup = self.out / f"{bundle.installer_name(version)}.exe"
        digest = sha256_file(setup)
        checksum = setup.with_name(setup.name + ".sha256")
        checksum.write_bytes(f"{digest}  {setup.name}\n".encode("ascii"))
        self.summary["installer"] = {
            "path": str(setup),
            "bytes": setup.stat().st_size,
            "sha256": digest,
        }
        return setup


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", type=Path, default=ROOT / "build" / "windows")
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    parser.add_argument("--python", default=getattr(sys, "_base_executable", sys.executable))
    parser.add_argument("--iscc", type=Path, help="Inno Setup 7.1.0's ISCC.exe")
    parser.add_argument("--no-installer", action="store_true", help="stop after the checks")
    parser.add_argument("--smoke-runs", type=int, default=3)
    parser.add_argument(
        "--reuse-env",
        action="store_true",
        help="keep WORK/env when its stamp shows the same locked requirements, fully"
        " installed, on the interpreter --python names",
    )
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        print("build.py builds the Windows installer, on Windows", file=sys.stderr)
        return 2
    work, out = args.work.absolute(), args.out.absolute()
    work.mkdir(parents=True, exist_ok=True)
    try:
        build = Build(work, out)
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    try:
        version = read_version()
        build.summary["proteia"] = version
        iscc = None if args.no_installer else find_iscc(args.iscc)
        inno_license = None if iscc is None else inno_setup_license(iscc)
        build.step("build environment", lambda: build.environment(args.python, args.reuse_env))
        build.step("bundle", lambda: build.freeze(version, inno_license))
        build.step("self-test", build.self_test)
        build.step("launch check", lambda: build.smoke(args.smoke_runs))
        if iscc is not None:
            build.step("installer", lambda: build.installer(iscc, version))
    except BuildError as exc:
        build.summary["error"] = str(exc)
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    finally:
        text = json.dumps(build.summary, indent=1)
        (work / "build-summary.json").write_text(text, encoding="utf-8")
    shown = {key: value for key, value in build.summary.items() if key != "bundle_manifest"}
    print(json.dumps(shown, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
