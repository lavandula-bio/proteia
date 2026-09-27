# SPDX-License-Identifier: Apache-2.0
"""The launch check of a built or installed ``Proteia.exe`` (ADR 0003): start,
token, redirect file, second launch, quit.

``python smoke.py EXE FOLDER [--runs N] [--json FILE]`` launches ``EXE`` as the
Start Menu would, ``N`` times in a row (the first launch of new files is the
cold start, the others warm), each in a fresh state folder under ``FOLDER``,
and checks each launch:

* ``/api/status`` answers 200 with the token from ``instance.json`` and 401
  without it; the page shell and ``/static/app.js`` are served;
* the redirect file holds the address with that token, and the browser was
  given the redirect file;
* a second launch opens the running instance (exit 0) instead of starting a
  server;
* every module the process loaded comes from the bundle or Windows;
* ``POST /api/quit`` ends the process with exit code 0 and removes the instance
  and redirect files (the lock file and the session log's folder stay);
* the session log (``logs/proteia.log`` in the state folder, #137) records the
  start and the end of both launches, and never the access token.

It times each launch from the start of the process to the first answer of
``/api/status``. The process runs without a console window, with ``PATH``
limited to the Windows folders and ``LOCALAPPDATA``, ``APPDATA``,
``USERPROFILE`` and ``TEMP`` in ``FOLDER``. No browser opens: ``BROWSER``
names a stand-in (this script with ``--stand-in-browser``), which Python's
``webbrowser`` tries first and which only records the address; the stand-in is
run once before any launch, exactly as the app will run it, and the check stops
if it fails, since ``webbrowser`` would then fall back to the real browser.
Exit status 0 when every launch passed every check.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shlex
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

STAND_IN_FLAG = "--stand-in-browser"
START_WAIT = 90.0  # seconds from the start of the process to the first answer
QUIT_WAIT = 30.0
# The session log in the state folder (proteia.web.logs.LOG_DIR and LOG_FILE):
# the app makes the folder, and uninstalling keeps it (ADR 0003).
LOG_DIR = "logs"
LOG_FILE = "proteia.log"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_SYSTEM_ROOT = os.environ.get("SystemRoot", r"C:\Windows")


def restricted_env(folder: Path, browser: str | None = None) -> dict[str, str]:
    """The environment a launch gets: ``PATH`` limited to the Windows folders, the
    per-user folders in ``folder`` (created), and ``BROWSER`` when given."""
    env = {
        "PATH": rf"{_SYSTEM_ROOT}\System32;{_SYSTEM_ROOT}",
        "SYSTEMROOT": _SYSTEM_ROOT,
        "WINDIR": _SYSTEM_ROOT,
        "SYSTEMDRIVE": os.environ.get("SystemDrive", "C:"),
    }
    if browser is not None:
        env["BROWSER"] = browser
    for name, sub in (
        ("LOCALAPPDATA", "localappdata"),
        ("APPDATA", "appdata"),
        ("USERPROFILE", "home"),
        ("TEMP", "temp"),
        ("TMP", "temp"),
    ):
        (folder / sub).mkdir(parents=True, exist_ok=True)
        env[name] = str(folder / sub)
    return env


def stand_in_browser(log: Path, python: str | None = None) -> str:
    """A ``BROWSER`` value that runs this script as a stand-in, recording each
    address in ``log``. It contains ``%s``, so ``webbrowser`` splits it as a
    shell command line (``webbrowser.get``); forward slashes keep the paths
    whole through that split."""
    parts = [Path(python or sys.executable).as_posix(), Path(__file__).as_posix()]
    return shlex.join([*parts, STAND_IN_FLAG, log.as_posix(), "%s"])


def check_stand_in(browser: str, log: Path, env: dict[str, str]) -> None:
    """Run the stand-in once as the app will (``webbrowser.get`` splits it, then
    runs it with the launch's environment); ``RuntimeError`` unless it records
    the address and exits 0."""
    command = webbrowser.get(browser)
    if not isinstance(command, webbrowser.GenericBrowser):
        raise RuntimeError(f"webbrowser reads {browser!r} as {command!r}, not a command")
    probe = "proteia-smoke:check"
    argv = [command.name] + [arg.replace("%s", probe) for arg in command.args]
    done = subprocess.run(argv, env=env, capture_output=True, creationflags=_NO_WINDOW, timeout=60)
    recorded = log.read_text(encoding="utf-8") if log.exists() else ""
    if done.returncode != 0 or probe not in recorded:
        raise RuntimeError(
            f"the stand-in browser does not work ({done.returncode}, {done.stderr[-500:]!r});"
            " a launch would open the real browser"
        )
    log.unlink()


def _request(port: int, path: str, token: str | None = None, method: str = "GET") -> tuple:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def _instance(state: Path) -> dict[str, Any] | None:
    try:
        return json.loads((state / "instance.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def wait_for_status(proc: subprocess.Popen, state: Path, start: float) -> tuple[dict, float]:
    """The instance file's contents and the seconds from ``start`` to the first
    200 from ``/api/status``; ``RuntimeError`` if the process ends or does not
    answer in :data:`START_WAIT`."""
    info = None
    while time.perf_counter() - start < START_WAIT:
        if proc.poll() is not None:
            raise RuntimeError(f"the process ended with {proc.returncode} before answering")
        info = info or _instance(state)
        if info is not None:
            try:
                if _request(info["port"], "/api/status", info["token"])[0] == 200:
                    return info, time.perf_counter() - start
            except OSError:
                pass
        time.sleep(0.01)
    raise RuntimeError(f"no answer from /api/status in {START_WAIT:.0f} s")


def session_log_problems(state: Path, token: str, sessions: int) -> list[str]:
    """What is wrong with the session log in the state folder ``state`` after
    ``sessions`` launches that each ended: it must be there, record each one's
    start and end, and never hold the access token ``token``."""
    path = state / LOG_DIR / LOG_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        return [f"the session log cannot be read: {type(exc).__name__}: {exc}"]
    problems = []
    for event in ("session started", "session ended"):
        count = text.count(f" proteia.web.launch: {event}")
        if count != sessions:
            problems.append(f"the session log records {event!r} {count} times, not {sessions}")
    if token in text:
        problems.append("the session log holds the access token")
    return problems


def _allowed(exe: Path) -> tuple[str, ...]:
    """Where a module the app loads may come from: the bundle, Windows, and
    Microsoft Defender's platform folder, whose modules Windows loads into
    processes it scans (its MpOav.dll was seen in Proteia's)."""
    program_data = os.environ.get("ProgramData", r"C:\ProgramData")
    return tuple(
        str(folder).lower().rstrip("\\") + "\\"
        for folder in (exe.parent, _SYSTEM_ROOT, rf"{program_data}\Microsoft\Windows Defender")
    )


def _process_info(pid: int) -> tuple[list[str], int | None]:
    """The files of the modules process ``pid`` loaded, and its working set."""
    script = f"$p = Get-Process -Id {pid}; $p.WorkingSet64; $p.Modules.FileName"
    done = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        creationflags=_NO_WINDOW,
        timeout=60,
    )
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    working_set = int(lines[0]) if lines and lines[0].isdigit() else None
    return lines[1:], working_set


def launch_once(exe: Path, folder: Path, python: str | None = None) -> dict[str, Any]:
    """One launch of ``exe`` with its state in ``folder``: what it measured and
    which checks failed."""
    log = folder / "browser.log"
    browser = stand_in_browser(log, python)
    env = restricted_env(folder, browser)
    check_stand_in(browser, log, env)
    state = folder / "localappdata" / "Proteia"
    failures: list[str] = []
    result: dict[str, Any] = {"failures": failures}

    def expect(condition: object, message: str) -> None:
        if not condition:
            failures.append(message)

    with (folder / "stdout.txt").open("wb") as out, (folder / "stderr.txt").open("wb") as err:
        start = time.perf_counter()
        proc = subprocess.Popen(
            [str(exe)], env=env, cwd=folder, stdout=out, stderr=err, creationflags=_NO_WINDOW
        )
        try:
            info, first = wait_for_status(proc, state, start)
            result["first_status_s"] = round(first, 3)
            port, token = info["port"], info["token"]
            status, body = _request(port, "/api/status", token)
            expect(json.loads(body).get("app") == "proteia", f"/api/status answered {body!r}")
            expect(_request(port, "/api/status")[0] == 401, "/api/status without the token")
            status, shell = _request(port, "/")
            expect(status == 200 and b"/static/app.js" in shell, f"the page shell: {status}")
            expect(_request(port, "/static/app.js")[0] == 200, "/static/app.js is not served")
            redirect = (state / "open-proteia.html").read_text(encoding="utf-8")
            url = f"http://127.0.0.1:{port}/#token={token}"
            expect(url in redirect, "the redirect file does not hold the address with the token")
            opened = log.read_text(encoding="utf-8").split() if log.exists() else []
            redirect_uri = (state / "open-proteia.html").as_uri()
            expect(opened == [redirect_uri], f"the browser was given {opened}")

            second_start = time.perf_counter()
            second = subprocess.run(
                [str(exe)], env=env, cwd=folder, capture_output=True, timeout=60,
                creationflags=_NO_WINDOW,
            )  # fmt: skip
            result["second_launch_s"] = round(time.perf_counter() - second_start, 3)
            said = second.stdout.decode("utf-8", "replace")
            expect(second.returncode == 0, f"the second launch exited {second.returncode}")
            expect("already running" in said, f"the second launch said {said!r}")
            opened = log.read_text(encoding="utf-8").split()
            expect(opened == [redirect_uri] * 2, f"after the second launch: {opened}")

            modules, working_set = _process_info(proc.pid)
            outside = [m for m in modules if not m.lower().startswith(_allowed(exe))]
            expect(modules, "the loaded modules could not be listed")
            expect(not outside, f"modules from outside the bundle: {outside}")
            result["modules_loaded"] = len(modules)
            result["working_set_mb"] = None if working_set is None else round(working_set / 2**20)

            quit_start = time.perf_counter()
            status, _ = _request(port, "/api/quit", token, method="POST")
            expect(status == 202, f"POST /api/quit answered {status}")
            code = proc.wait(QUIT_WAIT)
            result["quit_to_exit_s"] = round(time.perf_counter() - quit_start, 3)
            expect(code == 0, f"the process exited with {code} after Quit")
            left = sorted(path.name for path in state.iterdir())
            wanted = ["instance.lock", LOG_DIR]
            expect(left == wanted, f"the state folder holds {left} after Quit, not {wanted}")
            failures.extend(session_log_problems(state, token, sessions=2))
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(QUIT_WAIT)
    result["stdout"] = (folder / "stdout.txt").read_text("utf-8", "replace").splitlines()
    result["stderr"] = (folder / "stderr.txt").read_text("utf-8", "replace")[-2000:]
    return result


def run(exe: Path, folder: Path, runs: int = 3, python: str | None = None) -> dict[str, Any]:
    """``runs`` launches of ``exe``, each in ``folder/launch-<n>``."""
    launches = [launch_once(exe, folder / f"launch-{n}", python) for n in range(1, runs + 1)]
    return {
        "ok": all(not launch["failures"] for launch in launches),
        "exe": str(exe),
        "launches": launches,
    }


def _record_address(log: str, url: str) -> int:
    """The stand-in browser: record ``url`` in ``log``; always exit 0, so that
    ``webbrowser`` never falls back to the real browser."""
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write(url + "\n")
    except OSError:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == [STAND_IN_FLAG]:
        return _record_address(argv[1], argv[2]) if len(argv) == 3 else 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("exe", type=Path)
    parser.add_argument("folder", type=Path, help="a new folder for the launches' state")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json", type=Path, help="write the results to this file")
    args = parser.parse_args(argv)
    result = run(args.exe.absolute(), args.folder.absolute(), args.runs)
    text = json.dumps(result, indent=1, ensure_ascii=False)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    for n, launch in enumerate(result["launches"], 1):
        timing = launch.get("first_status_s")
        print(f"launch {n}: first /api/status after {timing} s; failures: {launch['failures']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
