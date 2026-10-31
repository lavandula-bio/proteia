# SPDX-License-Identifier: Apache-2.0
"""Install, check and uninstall a built Proteia installer (#54, ADR 0003).

``python install_check.py SETUP FOLDER [--expect-numbers REPORT] [--json FILE]``
runs the installer silently for the current user into ``FOLDER/app`` and:

1. checks what it installed (the program, LICENSE.txt, THIRD_PARTY_NOTICES.txt,
   the uninstaller, the Start Menu shortcut, the uninstall entry);
2. launches the installed app three times (``smoke.py``: the first launch of the
   new files is the cold start) and runs its self-test, whose numbers must equal
   those of ``REPORT`` (the build's frozen self-test) when given;
3. runs the installer again while Proteia runs: it must stop Proteia through
   Quit and install;
4. runs the installer (told not to close applications) and the uninstaller
   while Proteia runs but cannot be reached with Quit: its instance file names
   a port nothing listens on (as for a hung Proteia), or there is none (as for
   another account's Proteia). Both must refuse, the installer at its Preparing
   step (exit code 7), and leave Proteia running and every file in place. A
   stale instance file, whose process id another program now has, must not
   stop the installer;
5. uninstalls while Proteia runs, with other files in the state folder: the
   uninstaller must stop Proteia, remove the program, the shortcut, the
   uninstall entry and the launcher's instance files, and keep the other files:
   the session log the app wrote there (``logs/proteia.log``, #137), a rotated
   log from an earlier session, and a file of the user's;
6. installs again, ends Proteia without Quit (its instance files stay behind,
   as after a crash) and uninstalls: the uninstaller removes those files.

From step 3 on, the state folder also holds the session log the app writes
there, and each step expects it.

Every process runs with ``LOCALAPPDATA``, ``APPDATA``, ``USERPROFILE`` and
``TEMP`` in ``FOLDER`` and a stand-in browser (no browser opens, no window
shows). What Inno Setup takes from Windows' known folders and the registry
rather than the environment is real, and the report lists it: the Start Menu
shortcut and the uninstall entry under ``HKEY_CURRENT_USER`` (both created and
removed). ``Documents/Proteia`` and the real ``%LOCALAPPDATA%/Proteia`` are
compared before and after: no Proteia the check starts may write there, its
session log included (a Proteia run from source meanwhile writes its session
log there, so quit it first). The check refuses to run when this account already
has Proteia installed. A step that raises (a Proteia that never answers, a
timeout) is a failure that ends the run; however the run ends, the check then
ends the Proteia processes it started and runs the scratch installation's
uninstaller silently when it is there, so that the account is left without
Proteia, and the report says what it cleaned up. ``FOLDER`` must be new or
empty, or one an earlier run made (it holds the file :data:`MARKER`), which is
emptied first; any other folder is refused, never emptied. Exit status 0 when
every check passed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import winreg
from pathlib import Path
from typing import Any

import bundle
import smoke
from build import compare_numbers, folder_size

HERE = Path(__file__).resolve().parent
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
SILENT = ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART")
INSTANCE_FILES = ("instance.lock", "instance.json", "open-proteia.html")
# The session log the app writes in the state folder, as state_files lists it.
SESSION_LOG = (smoke.LOG_DIR, f"{smoke.LOG_DIR}/{smoke.LOG_FILE}")
# What step 5 adds to the state folder, besides the app's files: a log an
# earlier session rotated, and a file of the user's.
OTHER_STATE_FILES = (f"{smoke.LOG_DIR}/{smoke.LOG_FILE}.1", "notes µ.txt")
# The file that marks FOLDER as this check's own: the next run may empty it.
MARKER = ".install-check"
# Setup's exit code when its Preparing step (PrepareToInstall) refused.
PREPARE_REFUSED = 7
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def uninstall_entry(app_id: str) -> dict[str, str] | None:
    """The per-user uninstall entry Inno Setup writes for ``app_id``, or None."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"{UNINSTALL_KEY}\{{{app_id}}}_is1") as key:
            names = ("DisplayName", "DisplayVersion", "Publisher", "InstallLocation")
            entry = {}
            for name in names:
                with contextlib.suppress(OSError):
                    entry[name] = str(winreg.QueryValueEx(key, name)[0])
            return entry
    except OSError:
        return None


def known_folders(env: dict[str, str] | None = None) -> dict[str, Path]:
    """The Documents, Start Menu programs and Desktop folders (Windows' known
    folders) a process with ``env`` gets. Those Windows keeps as paths in the
    user profile follow ``USERPROFILE``: run with a scratch one, the installer
    puts its shortcut in the scratch folder. One redirected elsewhere
    (Documents to OneDrive, say) stays the real one."""
    powershell = shutil.which("powershell") or "powershell"
    script = (
        "[Environment]::GetFolderPath('MyDocuments', 'DoNotVerify');"
        "[Environment]::GetFolderPath('Programs', 'DoNotVerify');"
        "[Environment]::GetFolderPath('DesktopDirectory', 'DoNotVerify')"
    )
    done = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        env=env,
        creationflags=_NO_WINDOW,
        timeout=60,
        check=True,
    )
    documents, programs, desktop = done.stdout.strip().splitlines()
    return {"documents": Path(documents), "programs": Path(programs), "desktop": Path(desktop)}


def remove_tree(folder: Path) -> None:
    """Remove ``folder``, read-only entries too (Windows makes the Start Menu
    folders the installer created read-only)."""

    def writable(function: Any, path: str, error: BaseException) -> None:
        os.chmod(path, stat.S_IWRITE)
        function(path)

    shutil.rmtree(folder, onexc=writable)


def prepare_folder(folder: Path) -> None:
    """Make ``folder`` this check's scratch folder, marked with :data:`MARKER`.
    A new or empty folder is used as it is, and one an earlier run made is
    emptied first; ``ValueError`` for any other folder that holds something,
    which is left as it is."""
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"{folder} is not a folder")
    if folder.is_dir() and any(folder.iterdir()):
        if not (folder / MARKER).is_file():
            raise ValueError(
                f"{folder} holds files this check did not make; give a new or empty folder"
            )
        remove_tree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    text = "The scratch folder of install_check.py: its next run empties it.\n"
    (folder / MARKER).write_bytes(text.encode("utf-8"))


def snapshot(path: Path) -> dict[str, Any]:
    """Whether ``path`` exists, and when it and everything in it last changed."""
    if not path.exists():
        return {"exists": False}
    times = [path.stat().st_mtime_ns] + [p.stat().st_mtime_ns for p in path.rglob("*")]
    return {"exists": True, "entries": len(times) - 1, "latest_change_ns": max(times)}


class Check:
    """The checks of one run, and their results."""

    def __init__(self, setup: Path, folder: Path) -> None:
        self.setup = setup
        self.folder = folder
        self.app = folder / "app"
        self.user = folder / "user"
        self.log = self.user / "browser.log"
        self.env = smoke.restricted_env(self.user, smoke.stand_in_browser(self.log))
        self.state = self.user / "localappdata" / "Proteia"
        self.app_id = bundle.iss_app_id((HERE / "proteia.iss").read_text(encoding="utf-8"))
        self.report: dict[str, Any] = {"failures": [], "touched_outside_folder": []}
        self.started: list[subprocess.Popen] = []  # every Proteia start_app started
        self.shortcut: Path | None = None  # the Start Menu shortcut, once known

    def expect(self, condition: object, message: str) -> None:
        if not condition:
            self.report["failures"].append(message)

    def install(
        self, label: str, extra: tuple[str, ...] = (), expect_exit: int = 0
    ) -> dict[str, Any]:
        start = time.perf_counter()
        done = subprocess.run(
            [str(self.setup), *SILENT, *extra, "/CURRENTUSER", f"/DIR={self.app}",
             f"/LOG={self.folder / f'{label}.log'}"],
            env=self.env, timeout=900,
        )  # fmt: skip
        result = {"exit": done.returncode, "seconds": round(time.perf_counter() - start, 2)}
        self.expect(
            done.returncode == expect_exit,
            f"{label}: the installer exited {done.returncode}, not {expect_exit}",
        )
        return result

    def uninstall(self, label: str, refused: bool = False) -> dict[str, Any]:
        """Run the uninstaller and wait until it has finished: it runs a copy of
        itself from TEMP, so its own exit is not the end. ``refused``: it must
        stop before it removes anything, with a non-zero exit code."""
        start = time.perf_counter()
        uninstaller = self.app / "unins000.exe"
        done = subprocess.run(
            [str(uninstaller), *SILENT, f"/LOG={self.folder / f'{label}.log'}"],
            env=self.env,
            timeout=900,
        )
        deadline = time.monotonic() + 180
        while (
            not refused
            and time.monotonic() < deadline
            and (uninstaller.exists() or uninstall_entry(self.app_id) is not None)
        ):
            time.sleep(0.2)
        result = {"exit": done.returncode, "seconds": round(time.perf_counter() - start, 2)}
        if refused:
            self.expect(done.returncode != 0, f"{label}: the uninstaller did not refuse")
        else:
            self.expect(done.returncode == 0, f"{label}: the uninstaller exited {done.returncode}")
        return result

    def start_app(self) -> subprocess.Popen:
        """Start the installed app and wait for its first answer. The process is
        remembered first, so that :meth:`clean_up` ends it if it never answers."""
        proc = subprocess.Popen(
            [str(self.app / bundle.EXE_NAME)],
            env=self.env,
            cwd=self.user,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
        )
        self.started.append(proc)
        smoke.wait_for_status(proc, self.state, time.perf_counter())
        return proc

    def state_files(self) -> list[str]:
        if not self.state.exists():
            return []
        return sorted(p.relative_to(self.state).as_posix() for p in self.state.rglob("*"))

    def app_files(self) -> list[tuple[str, int]]:
        """The installation's files and their sizes."""
        return sorted(
            (p.relative_to(self.app).as_posix(), p.stat().st_size)
            for p in self.app.rglob("*")
            if p.is_file()
        )

    def refusals(self) -> dict[str, Any]:
        """Step 4: Proteia runs, but the installer's Quit cannot stop it. Its
        instance file names a port nothing listens on, as for a hung Proteia, or
        there is none, as for another account's Proteia. The installer, told not
        to close applications (as when the user chooses so), and the uninstaller
        must refuse and leave Proteia running and every file in place. Then,
        with Proteia quit, an instance file whose process id another program
        now has must not stop the installer."""
        report: dict[str, Any] = {}
        proc = self.start_app()
        instance = self.state / "instance.json"
        original = instance.read_bytes()
        info = json.loads(original)
        files = self.app_files()
        for label, wrong_port in (
            ("install-no-answer", True),
            ("install-no-instance-file", False),
            ("uninstall-no-answer", True),
            ("uninstall-no-instance-file", False),
        ):
            if wrong_port:
                instance.write_text(json.dumps(dict(info, port=1)), encoding="utf-8")
            else:
                instance.unlink(missing_ok=True)
            if label.startswith("install"):
                result = self.install(label, ("/NOCLOSEAPPLICATIONS",), PREPARE_REFUSED)
            else:
                result = self.uninstall(label, refused=True)
            result["app_running"] = proc.poll() is None
            self.expect(proc.poll() is None, f"{label}: Proteia ended ({proc.poll()})")
            self.expect(self.app_files() == files, f"{label}: the installed files changed")
            self.expect(uninstall_entry(self.app_id) is not None, f"{label}: no uninstall entry")
            with contextlib.suppress(OSError):
                result["static_app_js"] = smoke._request(info["port"], "/static/app.js")[0]
            self.expect(result.get("static_app_js") == 200, f"{label}: app.js is not served")
            left = self.state_files()
            wanted = sorted(
                ["instance.lock", "open-proteia.html", *SESSION_LOG]
                + ["instance.json"] * wrong_port
            )
            self.expect(left == wanted, f"{label}: the state folder holds {left}, not {wanted}")
            report[label] = result
        instance.write_bytes(original)
        with contextlib.suppress(OSError):
            report["quit"] = smoke._request(info["port"], "/api/quit", info["token"], "POST")[0]
        try:
            report["app_exit"] = proc.wait(smoke.QUIT_WAIT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(30)
        self.expect(report.get("app_exit") == 0, f"after the refusals: Proteia {report}")
        # A stale instance file whose process id now belongs to another program
        # (this script) is no reason to refuse.
        stale = dict(info, pid=os.getpid(), port=1)
        instance.write_text(json.dumps(stale), encoding="utf-8")
        report["install-stale-instance-file"] = self.install("install-stale-instance-file")
        instance.unlink(missing_ok=True)
        return report

    def run(self, expect_numbers: dict[str, Any] | None) -> dict[str, Any]:
        """Every check, in order. A step that raises is a failure that ends the
        run; whatever happens, :meth:`clean_up` then ends the processes the check
        started and uninstalls what it installed, so the account is left as the
        check found it."""
        report = self.report
        real, scratch = known_folders(), known_folders(self.env)
        report["known_folders"] = {"real": real, "with_scratch_env": scratch}
        shortcut = self.shortcut = scratch["programs"] / f"{bundle.APP_NAME}.lnk"
        desktop = scratch["desktop"] / f"{bundle.APP_NAME}.lnk"
        outside = {
            "documents": real["documents"] / bundle.APP_NAME,
            "real_state": Path(os.environ["LOCALAPPDATA"]) / bundle.APP_NAME,
            "real_start_menu": real["programs"] / f"{bundle.APP_NAME}.lnk",
            "real_desktop": real["desktop"] / f"{bundle.APP_NAME}.lnk",
        }
        before = {name: snapshot(path) for name, path in outside.items()}
        # Refused before anything is installed: nothing to clean up.
        if uninstall_entry(self.app_id) is not None or before["real_start_menu"]["exists"]:
            raise SystemExit("Proteia is installed for this account: uninstall it first")
        smoke.check_stand_in(self.env["BROWSER"], self.log, self.env)

        steps = (
            ("1. install", lambda: self.check_install(shortcut, desktop)),
            ("2. launches and self-test", lambda: self.check_launches(expect_numbers)),
            ("3. upgrade while running", self.check_upgrade),
            ("4. refused while unreachable", self.check_refusals),
            ("5. uninstall while running", self.check_uninstall),
            ("6. uninstall after a crash", self.check_uninstall_after_crash),
        )
        try:
            for index, (name, action) in enumerate(steps):
                try:
                    action()
                except Exception as exc:  # recorded; the finally below cleans up
                    error = f"{type(exc).__name__}: {exc}"
                    not_run = [later for later, _ in steps[index + 1 :]]
                    report["stopped"] = {"step": name, "error": error, "not_run": not_run}
                    self.expect(False, f"step {name} stopped with {error}")
                    break
        finally:
            report["cleanup"] = self.clean_up()

        after = {name: snapshot(path) for name, path in outside.items()}
        report["outside"] = {name: [str(outside[name]), before[name]] for name in outside}
        self.expect(before == after, f"changed outside the scratch folder: {before} -> {after}")
        report["ok"] = not report["failures"]
        return report

    def clean_up(self) -> dict[str, Any]:
        """End every Proteia the check started that still runs, then run the
        scratch installation's uninstaller silently if it is there (a run that
        stopped midway leaves it installed for this account, with its uninstall
        entry and Start Menu shortcut, and the next run refuses to start).
        Returns what it did; what it could not remove is a failure. Never
        raises, so that the report is written."""
        done: dict[str, Any] = {"stopped": [], "uninstalled": None, "left": []}
        for proc in self.started:
            if proc.poll() is None:
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(30)
                done["stopped"].append(proc.pid)
        if (self.app / "unins000.exe").is_file():
            try:
                done["uninstalled"] = self.uninstall("clean-up-uninstall")
            except Exception as exc:
                self.expect(False, f"clean-up: the uninstaller failed: {type(exc).__name__}: {exc}")
        if uninstall_entry(self.app_id) is not None:
            done["left"].append(f"HKEY_CURRENT_USER\\{UNINSTALL_KEY}\\{{{self.app_id}}}_is1")
        if self.shortcut is not None and self.shortcut.exists():
            done["left"].append(str(self.shortcut))
        self.expect(
            not done["left"],
            "clean-up: still there, remove by hand before the next run: " + "; ".join(done["left"]),
        )
        return done

    def check_install(self, shortcut: Path, desktop: Path) -> None:
        """Step 1: install, and what it installed."""
        report = self.report
        report["install"] = self.install("install")
        for name in (bundle.EXE_NAME, "LICENSE.txt", "THIRD_PARTY_NOTICES.txt", "unins000.exe"):
            self.expect((self.app / name).is_file(), f"no {name} in the installation")
        size, files = folder_size(self.app)
        report["installed"] = {"bytes": size, "files": files}
        entry = uninstall_entry(self.app_id)
        report["uninstall_entry"] = entry
        self.expect(entry is not None, "no uninstall entry")
        self.expect(shortcut.exists(), f"no Start Menu shortcut at {shortcut}")
        self.expect(not desktop.exists(), "a desktop shortcut nobody asked for")
        report["touched_outside_folder"].append(
            f"HKEY_CURRENT_USER\\{UNINSTALL_KEY}\\{{{self.app_id}}}_is1"
        )
        if not shortcut.is_relative_to(self.folder):
            report["touched_outside_folder"].append(str(shortcut))

    def check_launches(self, expect_numbers: dict[str, Any] | None) -> None:
        """Step 2: launches (the first is the cold start) and the self-test."""
        report = self.report
        launches = smoke.run(self.app / bundle.EXE_NAME, self.folder / "smoke", runs=3)
        report["launches"] = [
            {key: launch.get(key) for key in ("first_status_s", "second_launch_s", "failures")}
            for launch in launches["launches"]
        ]
        self.expect(launches["ok"], "a launch of the installed app failed its checks")
        selftest = self.folder / "selftest.json"
        done = subprocess.run(
            [str(self.app / bundle.EXE_NAME), "--self-test", "--json", str(selftest)],
            env=self.env, capture_output=True, creationflags=_NO_WINDOW, timeout=600,
        )  # fmt: skip
        result = json.loads(selftest.read_text(encoding="utf-8")) if selftest.exists() else {}
        report["selftest"] = {
            "exit": done.returncode,
            "steps": {step["name"]: step["ok"] for step in result.get("steps", [])},
        }
        self.expect(done.returncode == 0 and result.get("ok"), "the installed self-test failed")
        if expect_numbers is not None:
            differences = compare_numbers(result.get("numbers", {}), expect_numbers)
            report["selftest"]["numbers_as_built"] = not differences
            self.expect(not differences, f"installed numbers differ: {differences[:3]}")

    def check_upgrade(self) -> None:
        """Step 3: the installer again while Proteia runs."""
        proc = self.start_app()
        result = self.report["upgrade_while_running"] = self.install("upgrade")
        result["app_exit"] = proc.poll()
        self.expect(proc.poll() == 0, f"Proteia exited {proc.poll()} when the installer ran")
        if proc.poll() is None:
            proc.kill()
        wanted = ["instance.lock", *SESSION_LOG]
        left = self.state_files()
        self.expect(left == wanted, f"upgrade: the state folder holds {left}, not {wanted}")

    def check_refusals(self) -> None:
        """Step 4: the installer and the uninstaller while Quit cannot reach Proteia."""
        self.report["refused_while_unreachable"] = self.refusals()

    def check_uninstall(self) -> None:
        """Step 5: uninstall while Proteia runs, with other files in the state folder."""
        for name in OTHER_STATE_FILES:
            path = self.state / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{name}: kept\n".encode())
        proc = self.start_app()
        result = self.report["uninstall_while_running"] = self.uninstall("uninstall")
        result["app_exit"] = proc.poll()
        self.expect(proc.poll() == 0, f"Proteia exited {proc.poll()} when the uninstaller ran")
        if proc.poll() is None:
            proc.kill()
        self.after_uninstall("uninstall")

    def check_uninstall_after_crash(self) -> None:
        """Step 6: instance files left by a Proteia that did not quit."""
        self.install("reinstall")
        proc = self.start_app()
        proc.kill()
        proc.wait(30)
        left = self.state_files()
        self.expect(set(INSTANCE_FILES) <= set(left), f"after a kill the state holds {left}")
        self.report["uninstall_after_crash"] = self.uninstall("uninstall-after-crash")
        self.after_uninstall("uninstall-after-crash")

    def after_uninstall(self, label: str) -> None:
        """What an uninstall must remove, and what of the state folder it must
        keep: the session log, and the other files step 5 adds."""
        kept = sorted([*SESSION_LOG, *OTHER_STATE_FILES])
        left = self.state_files()
        self.report[f"{label}_state_left"] = left
        self.expect(left == kept, f"{label}: the state folder holds {left}, not {kept}")
        self.expect(not self.app.exists() or not any(self.app.iterdir()), f"{label}: files left")
        shortcut = self.shortcut
        self.expect(
            shortcut is not None and not shortcut.exists(),
            f"{label}: the Start Menu shortcut is still there",
        )
        self.expect(uninstall_entry(self.app_id) is None, f"{label}: the uninstall entry stayed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("setup", type=Path, help="the installer")
    parser.add_argument(
        "folder", type=Path, help="a new or empty scratch folder, or one an earlier run made"
    )
    parser.add_argument("--expect-numbers", type=Path, help="the build's frozen self-test report")
    parser.add_argument("--json", type=Path, help="write the report to this file")
    args = parser.parse_args(argv)
    # Everything is checked before the folder is touched.
    folder, setup = args.folder.absolute(), args.setup.absolute()
    if not setup.is_file():
        parser.error(f"no installer at {setup}")
    expected = None
    if args.expect_numbers:
        try:
            expected = json.loads(args.expect_numbers.read_text(encoding="utf-8"))["numbers"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"{args.expect_numbers} is not a self-test report: {exc!r}")
    for given in (setup, args.expect_numbers):
        if given is not None and given.resolve().is_relative_to(folder.resolve()):
            parser.error(f"{given} is inside {folder}, which the check empties")
    try:
        prepare_folder(folder)
    except ValueError as exc:
        parser.error(str(exc))
    report = Check(setup, folder).run(expected)
    if args.json:
        text = json.dumps(report, indent=1, ensure_ascii=False, default=str)
        args.json.write_text(text, encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))  # ASCII, for any console
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
