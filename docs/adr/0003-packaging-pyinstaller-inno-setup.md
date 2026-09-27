# ADR 0003: Packaging — Windows installer with PyInstaller and Inno Setup

- **Status**: Accepted
- **Date**: 2026-09-27
- **Deciders**: Roger Huang

## Context

ADR 0001 deferred packaging and named PyInstaller and Briefcase as possible tools.
ADR 0002 replaced napari with a local web application and left the choice of a
packaging tool to a separate ADR. Issue #54 asks for a single installer file that
installs Proteia on a Windows machine without Python and launches the app: the
server starts and the browser opens. A macOS installer and code signing are out of
scope for that issue.

**What the installer carries.** The `proteia` command (`proteia.web.launch:main`)
starts the local server, writes the per-launch token to files in the per-user state
folder (`%LOCALAPPDATA%\Proteia`), and hands the browser a redirect file. Projects
are kept in `Documents\Proteia`. An installer therefore carries a Python 3.13
runtime, the scientific stack (NumPy, SciPy, scikit-image, matplotlib, Pillow,
tifffile), the web stack (FastAPI, Starlette, Uvicorn, and their requirements), and
the web client's static files as package data. It needs no GUI toolkit: the user's
browser renders the interface.

**Package metadata is part of the result.** The analysis record stores the versions
of NumPy, SciPy, scikit-image, Pillow, and tifffile, which it reads with
`importlib.metadata`. A bundle that leaves out the packages' metadata still runs,
but its records lack those versions, and no error reports it.

**napari and PySide6 are still dependencies.** Until #57 removes them,
`napari[pyside6]` is a runtime dependency, and its tree accounts for 94 of the 116
locked packages (ADR 0002), including PySide6 (LGPL-3.0). The web application
imports neither. An installer built before #57 should leave them out, both for size
and to avoid the duties of conveying Qt that ADR 0002 describes.

**Guarantees to keep.** The installed app keeps ADR 0002's guarantees: it works
offline, sends no telemetry, and a launch makes no network connection other than the
check of its own loopback port. A normal install should not need administrator
rights, and uninstalling must not delete the user's projects.

## Candidates tried

The two tools ADR 0001 named were built and run on this app on 2026-09-27:

- **PyInstaller (one-folder mode) with Inno Setup.** PyInstaller 6.22.3, with
  pyinstaller-hooks-contrib 2026.7, freezes the app into a folder: an executable
  launcher, the Python runtime, the modules' bytecode in an archive, and the
  packages' native libraries and data. Inno Setup 7.1.0 wraps that folder in a setup
  program (`.exe`).
- **Briefcase with WiX.** Briefcase 0.4.5 assembles a Windows app from the Python
  embeddable package (3.13.15), a compiled stub launcher, and the installed packages
  as ordinary source files, and packages it as an MSI (Windows Installer package)
  with WiX 5.0.2.

**Method.** Both were built from the same source, `main` at commit 43bea77 exported
with `git archive`, with package versions from `uv.lock`, on one Windows 11 Pro x64
machine with Microsoft Defender real-time protection on. Each built app was run with
`PATH` limited to the Windows system folders, a temporary `LOCALAPPDATA`, and a
stand-in browser (set through the `BROWSER` variable, which Python's `webbrowser`
module tries first) that only records the URL it receives. A run passed when:

- the server answered `GET /api/status` with 200 when given the token and 401
  without it, and served the page shell;
- the launcher handed the browser the redirect file, and the token in that file
  matched the running instance;
- a second launch opened the running instance instead of starting a second server;
- `POST /api/quit` stopped the process with exit code 0 and removed the instance and
  redirect files.

Timings are wall-clock times from two to five runs on that one machine. They compare
the candidates; they are not benchmarks.

**Results.** Both candidates passed every run check.

| | PyInstaller + Inno Setup | Briefcase + WiX |
| --- | --- | --- |
| Installer file | 58.5 MB (`.exe`) | 79.0 MB (`.msi`) |
| Installed app | 190.0 MB in 1,160 files | 263.6 MB in 5,281 files |
| Clean build | 52 s to freeze + 22 s for the installer | 87 s, of which 80 s is WiX compressing the MSI |
| Launch to first answer, repeat launches | 1.24–1.44 s | 3.02–3.21 s |
| Launch to first answer, first launch of newly written files | 5.3–5.6 s | 12.2 s |
| Memory at idle (working set) | about 162 MB | 165–170 MB |
| `POST /api/quit` to process exit | 0.34–0.38 s | 0.31–0.51 s |
| Real install and uninstall | Tested (per-user; silent install 4.4–5.0 s) | Not tested; the MSI was only unpacked with `msiexec /a` |
| Upgrade over an older install | Tested, including while the app runs (fails, see below) | Not tested |

The extra time on a first launch is most likely Defender scanning the new native
libraries; neither trial isolated it. PyInstaller's first-launch figure comes from
two newly built variants without a console window, Briefcase's from the files
unpacked from its MSI. Defender reported no threat in either build.

### PyInstaller + Inno Setup

- **Defaults failed.** With PyInstaller's defaults the build stopped after 22 s on a
  path longer than Windows' 260-character limit (long paths are disabled on the
  build machine): IPython, reached through napari's dependencies, pulled in jedi's
  bundled type stubs. A near-default configuration that left out only IPython and
  jedi failed after 154 s with the same kind of error. By then matplotlib's hook had
  chosen the Qt backend (QtAgg) because PySide6 was installed, and the partial output
  was already 301 MB and contained PySide6, pandas, and dask.
- **Working configuration.** A spec file that collects the web client's static files
  as package data; copies the metadata of the five packages the record reads (the
  frozen app recorded all five versions); limits matplotlib's backends to Agg, SVG,
  and PDF (previews through Agg, charts as SVG); excludes `proteia.gui`, napari and
  the packages only it needs (PySide6, shiboken6, qtpy, superqt, vispy, magicgui,
  app-model, IPython and Jupyter, dask, pandas, zarr, pint) as well as tkinter; turns
  off UPX compression; and adds a Windows version resource. The result contained no
  Qt or napari file, and the running app loaded no module from outside its own
  folder and `C:\Windows`.
- **Frozen self-test.** A second executable built from the same spec ran the whole
  analysis inside the frozen environment: a sample image in a folder with a
  non-ASCII name, row detection, statistics, an SVG chart, and the lane table, then
  the same steps through the HTTP API with a project named `Web µ α β`. Its numbers
  were identical to an unfrozen run.
- **Installer.** Per-user by default (`%LOCALAPPDATA%\Programs\Proteia`, no UAC
  prompt), with a per-machine option for administrators; a Start Menu shortcut; a
  fixed application ID for upgrades. Uninstalling removed the program files, the
  shortcut, the registry entries, and the launcher state in `%LOCALAPPDATA%\Proteia`,
  and left `Documents\Proteia` alone.
- **Upgrade while running.** A silent install over a running Proteia gave up after
  31.7 s with exit code 5, because Windows' Restart Manager could not close the app.
  The running app and its data were not affected.
- **Windowed build.** A build without a console window never answered when started
  without standard streams, as from Explorer: Uvicorn's log formatter calls
  `sys.stdout.isatty()` on a missing stream and fails while Uvicorn is being
  configured (reproduced without freezing). A startup hook that points missing
  streams at `os.devnull` fixed it, at the cost of discarding all output.
- **One-file mode.** A single 84.6 MB executable, but it unpacks about 190 MB to the
  temporary folder on every launch, took 5.2–5.7 s on every start, and failed to
  unpack under a very long `TEMP` path.
- **Build machine leaks into the output.** The Universal CRT DLLs came from a
  Windows Kits folder on the build machine's `PATH`, and `MSVCP140.dll` from
  `System32`. The same spec can produce different bundles on different machines.
- **Notices.** The bundle carried license files for 8 packages and the fonts only;
  the license texts of Python, OpenSSL, and the other packages were missing.
- **Package set.** Import analysis decides what goes in. Because the build
  environment still contained napari's tree, 13 packages and parts of pywin32 that
  the app does not need came along through optional imports (see License check).
  Conversely, networkx, a requirement of scikit-image, was left out because nothing
  the app runs imports it; the self-test passed without it.
- A build warning reported that the hidden import `scipy.special._cdflib` was not
  found (a stale hook entry for SciPy 1.17); the self-test was not affected.

### Briefcase + WiX

- **Excluding napari.** Briefcase installs everything in `[project].dependencies`
  and cannot subtract from it. With `env_manager = "uv"`, a constraints file exported
  from `uv.lock`, and uv's `--excludes` naming napari, napari's whole tree never
  installed: 31 packages instead of 116, the same set that remains after #57. With
  napari included, the app was 1.09 GB in 25,426 files (PySide6 alone 664.5 MB), the
  MSI 323 MB, and the build took 361 s.
- **Project changes.** Briefcase needed `license-files = ["LICENSE"]` in the project
  metadata (it refuses to build without a declared license file), a
  `src/proteia/__main__.py` (its launcher runs `python -m proteia`), a
  `[tool.briefcase]` section, and the constraints file, which has to be regenerated
  whenever `uv.lock` changes.
- **Build failures.** The first three attempts failed on path length: cloning the
  template, rendering it, and Briefcase resolving a substituted drive back to the
  long path. They were worked around for the build process only and probably come
  from the deep build folder rather than from Briefcase; a short path was not tested.
- **Bytecode.** Briefcase ships `.py` files only, so every launch compiles every
  imported module, which explains most of the slower start. Precompiling with
  `compileall`, a custom step outside Briefcase, cut repeat launches to 1.17 s and
  the first launch to 5.4 s, and added 73 MB.
- **PATH.** For a console app the MSI adds the install folder to the user's or the
  system `PATH`, which puts `python313.dll`, the OpenSSL DLLs, `sqlite3.dll`, and the
  VC++ runtime into every program's DLL search path. Avoiding this needs Briefcase's
  GUI launcher (no console window, output discarded, stopped only through Quit on
  the page) or a custom template.
- **Other.** Briefcase signs the launcher and the MSI with `signtool` when given a
  certificate thumbprint. The MSI offers a per-user or a per-machine install
  (per-user by default; a silent install needs no administrator rights). The
  `author` field becomes the manufacturer, the install folder name, and the registry
  key. WiX needs a plain numeric version, so `0.1.0.dev0` becomes `0.1.0`.

### Common to both

- The installers, the launchers, and most third-party native libraries are unsigned.
  Files built locally carry no download mark, so the SmartScreen prompt was not
  reproduced; an unsigned installer downloaded with a browser is expected to show the
  "unknown publisher" warning.
- Neither tool has an updater: a newer installer replaces the old version.
- Neither build writes a log file; the console build prints three lines to its
  console window.
- What closing the console window while the server runs does to saving and cleanup
  was not tested with either.

## Decision

Package Proteia for Windows with **PyInstaller in one-folder mode** and install that
folder with an **Inno Setup** installer (`.exe`). Briefcase with WiX (MSI) is not
used, for the reasons below.

**Bundle**

- One-folder mode, not one-file.
- The spec file and the installer script live in the repository (for example under
  `packaging/windows/`). The spec collects the static files, copies the metadata of
  every bundled distribution (the record needs it, and the license files come with
  it), limits matplotlib to the Agg, SVG, and PDF backends, and uses no UPX.
- The build environment is installed from `uv.lock` without napari's tree, the way
  the Briefcase trial did it (`uv export` and `--excludes napari`), and the spec
  keeps its exclusion list as a second guard. After #57 the plain locked environment
  is enough and the exclusions go. This combination has not been built yet: the
  trial built from the full environment and relied on the spec's exclusions.
- The v0.1 app keeps its console window. The window shows that Proteia is running
  and what it reports, and Ctrl+C or Quit on the page stops it. A log file for every
  session (#137) and a diagnostic bundle for bug reports (#138) keep those messages
  after the window closes. A build without a console window, which needs a startup
  hook for the missing standard streams, is left to a later decision.

**Build tools**

- PyInstaller, pyinstaller-hooks-contrib, and their requirements (altgraph, pefile,
  pywin32-ctypes, setuptools) go into a separate uv dependency group, locked in
  `uv.lock`. They are not runtime dependencies: the default `uv sync` and the CI test
  jobs do not install them.
- PyInstaller's license (GPL-2.0-or-later with the Bootloader Exception) and the
  dual licensing of pyinstaller-hooks-contrib (GPL-2.0-or-later for its build hooks,
  Apache-2.0 for its runtime hooks) are accepted for build tools. No GPL-licensed code
  becomes part of Proteia's own code. What ships from these tools is PyInstaller's
  bootloader inside `Proteia.exe`, which the Bootloader Exception allows to be
  distributed with the app under any terms, and the Apache-2.0 runtime hooks.
- Installers are built by a CI workflow on a Windows runner, separate from the test
  jobs. It runs the checks below and keeps the built installer as a workflow
  artifact for one day.

**Installer**

- A per-user install by default, with a per-machine option for administrators; a
  fixed application ID; a Start Menu shortcut and an optional desktop shortcut.
- Uninstalling removes the program files and the launcher's own instance
  files in the uninstalling user's `%LOCALAPPDATA%\Proteia` (`instance.lock`,
  `instance.json`, `open-proteia.html`). It keeps the rest of that folder, where
  the session logs are written (#137), so a reinstall made to work around a
  problem does not delete the logs a bug report needs. It never touches
  `Documents\Proteia`.
- A per-machine uninstall runs as the administrator, so it reaches only that
  account's `%LOCALAPPDATA%`; other users' state folders stay. They hold no
  project data, and a later install reuses them.
- Before it replaces files, the installer stops a running Proteia, for example by
  sending Quit to the running instance (`POST /api/quit` with the token from
  `instance.json`; Proteia saves first and answers 409 if it cannot), and stops with
  a clear message if that fails. The trial did not implement this.
- The installer carries a notices file for the binary distribution: the Python
  runtime and the libraries built into it, the VC++ runtime terms, every bundled
  package, and the fonts.

**v0.1 (closed beta)**

- The installer is not code-signed. Testers receive its SHA-256 checksum and
  instructions for the Windows SmartScreen prompt. Code signing is decided before
  the first public release.
- There is no update check: a newer installer replaces the old version. An update
  check is planned for the first public release as a separate change (#143). It
  would be the app's first outbound connection, so it comes with an amendment to
  ADR 0002.

**Checks for every installer build**

- A smoke test of the built app with a restricted `PATH` and a stand-in browser:
  start, token, redirect file, second launch, quit.
- A frozen self-test that runs the analysis inside the bundle, compares its numbers
  with an unfrozen run, and fails if the record lacks any library version.
- For #54, an install, launch, and uninstall on a clean Windows machine without
  Python.

**Reasons**

- **Faster start.** Repeat launches answer in 1.24–1.44 s against 3.02–3.21 s, and
  the first launch of newly installed files in 5.3–5.6 s against 12.2 s. Briefcase
  comes close only with a custom bytecode step that adds 73 MB.
- **Smaller.** A 58.5 MB installer and 190 MB in 1,160 files once installed, against
  79.0 MB and 264 MB in 5,281 files. Fewer files probably also means less to scan
  on the first launch.
- **Verified lifecycle.** Install, uninstall, and upgrade were run for real,
  including the upgrade-while-running failure that this decision addresses. The
  Briefcase MSI was never installed.
- **No change to `PATH`.** The Briefcase console build puts the runtime's DLLs on
  every program's search path; avoiding that costs the console window or a custom
  template.
- **Project metadata unchanged.** Packaging lives in two files beside the code;
  Briefcase needs changes to `pyproject.toml`, an extra entry module, and a
  constraints file kept in step with `uv.lock`.

What Briefcase does better, and why it does not decide the matter:

- It installs packages as ordinary files with their metadata, so nothing can go
  missing through import analysis. For PyInstaller the frozen self-test is the
  guard; it passed in the trial, and it has to grow with the app.
- Its napari exclusion works at install time and is clean. The same uv step can
  prepare PyInstaller's build environment.
- It produces an MSI and signs through its own options. Inno Setup can call
  `signtool` as well, and an MSI can still be built later if institutions require
  one (see Reversibility).

## Consequences

**Positive**

- A user installs one 58.5 MB file, without Python and without administrator
  rights, and starts Proteia from the Start Menu; it answers in about 1.3 s.
- No Qt is bundled, so the LGPL duties that ADR 0002 describes do not arise, even
  before #57.
- Every installer build is checked by running it: the smoke test and the frozen
  self-test catch what the bundle leaves out.
- Uninstalling leaves the user's projects in place.
- The build tools stay out of the runtime and test environments, and `uv.lock` pins
  their versions for every build.

**Trade-offs**

- The spec needs care. PyInstaller bundles what its import analysis finds, so a new
  optional or lazily loaded import, or a package whose metadata the app reads, can
  be missing from the bundle without any build error. The frozen self-test must
  cover every analysis path and every metadata lookup.
- The output depends on the build machine (the Universal CRT DLLs and
  `MSVCP140.dll` above). Installers therefore come from the CI build, and the
  Universal CRT DLLs, which Windows 10 and later provide, are left out.
- PyInstaller's launcher is a known source of antivirus false positives in general,
  although none occurred here. One-folder mode, no UPX, a version resource, and
  later a signature reduce the risk; a false positive has to be reported to the
  antivirus vendor.
- Until the installer is signed, a downloaded installer shows the SmartScreen
  warning, and testers need the instructions to get past it. A checksum shows that
  the file arrived intact; unlike a signature, it does not show who built it.
- A console window stays open while Proteia runs, and closing it stops the server.
  What that does to saving has to be tested.
- Updates are manual in v0.1: testers learn of a new version outside the app and
  run the newer installer.
- CI keeps a built installer for one day, so an installer meant for testers has to
  be downloaded within that day or built again.
- Inno Setup's authors request a commercial license from commercial users (see
  License check).
- Two build tools, PyInstaller with its hooks and Inno Setup, have to be kept up to
  date, and a new release of either can change the bundle. Inno Setup is not a
  Python package, so `uv.lock` does not pin it; the build has to pin its version.

**Reversibility**

- The application code does not depend on the packaging tool; the entry point is
  the existing `proteia.web.launch:main`. Switching to Briefcase later means adding
  the configuration described above.
- Changing the installer format from Inno Setup to MSI, which some managed
  deployment tools require, is harder once people have Proteia installed: the two
  installers do not know about each other, so the MSI would have to find and remove
  the Inno Setup installation, or users would have to uninstall it first. The format
  is cheapest to change before the first wide distribution.

## License check

Licenses were read from each package's metadata and license files and, for the
tools, from their license texts and websites, on 2026-09-27.

**Build tools (run at build time, not shipped)**

| Tool | License | Notes |
| --- | --- | --- |
| PyInstaller 6.22.3 | GPL-2.0-or-later with the Bootloader Exception; Apache-2.0 for the runtime hooks | Accepted as a build tool. Its documentation states that the bundles it builds can be shipped under any license that complies with the licenses of their dependencies; only modifications to PyInstaller itself fall under the GPL. |
| pyinstaller-hooks-contrib 2026.7 | GPL-2.0-or-later (build hooks); Apache-2.0 (runtime hooks) | Accepted as a build tool. The build hooks run only at build time. |
| altgraph, pefile, pywin32-ctypes, setuptools | MIT, MIT, BSD-3-Clause, MIT | PyInstaller's requirements, installed with it in the build group. setuptools also reached the trial bundle (see below). |
| Inno Setup 7.1.0 | Inno Setup License (permissive; commercial use allowed) | Binary redistributions must keep its copyright notices and web addresses in place. Its authors request that all commercial users buy a commercial license, regardless of the version used; they define commercial users as for-profit organizations, and individuals doing for-profit work, with annual revenue above USD 5,000, and count donations to open-source or freeware software toward that threshold. They state that this is not strictly required. The request is to be reviewed before a commercial release. |

**Shipped from the tools**

- PyInstaller's bootloader, inside `Proteia.exe`: GPL-2.0-or-later with the
  Bootloader Exception, which allows it to be distributed with the built application
  without restriction. The runtime hooks in the bundle (`pyi_rth_inspect`,
  `pyi_rth_mplconfig`, `pyi_rth_multiprocessing`, `pyi_rth_pkgutil`,
  `pyi_rth_setuptools`, `_pyi_rth_utils`) are Apache-2.0.
- Inno Setup's setup and uninstall programs, inside the installer: Inno Setup
  License. The third-party components they contain (such as the LZMA decompressor)
  were not reviewed.

**Bundled runtime**

- CPython 3.13.14, the python-build-standalone build that uv installs: PSF-2.0. It
  includes OpenSSL 3 (Apache-2.0), libffi (MIT), and the compression, decimal, and
  XML libraries built into standard-library modules (bzip2, xz, zlib, mpdecimal,
  expat), all under permissive licenses. The bundled interpreter is the one in the
  build environment, so releases pin it.
- The VC++ runtime: `VCRUNTIME140.dll` and `VCRUNTIME140_1.dll` (from the Python
  build) and `MSVCP140.dll` (copied from the build machine), under Microsoft's
  redistribution terms for the Visual C++ runtime. NumPy and SciPy wheels also carry
  their own copies of `msvcp140`.
- The Universal CRT DLLs picked up from the build machine: not needed on Windows 10
  and later, to be left out.
- matplotlib's fonts, DejaVu and STIX, with their license files.

**Bundled Python packages**

The 31 packages that remain once napari is excluded. This is the set the Briefcase
build installed; the PyInstaller bundle held the same packages except networkx.

| Package | Version | License |
| --- | --- | --- |
| annotated-doc | 0.0.4 | MIT |
| annotated-types | 0.7.0 | MIT |
| anyio | 4.14.2 | MIT |
| click | 8.4.1 | BSD-3-Clause |
| colorama | 0.4.6 | BSD-3-Clause |
| contourpy | 1.3.3 | BSD-3-Clause |
| cycler | 0.12.1 | BSD-3-Clause |
| fastapi | 0.141.1 | MIT |
| fonttools | 4.63.0 | MIT |
| h11 | 0.16.0 | MIT |
| idna | 3.17 | BSD-3-Clause |
| imageio | 2.37.3 | BSD-2-Clause |
| kiwisolver | 1.5.0 | BSD-3-Clause |
| lazy-loader | 0.5 | BSD-3-Clause |
| matplotlib | 3.10.9 | Matplotlib License (PSF-based) |
| networkx | 3.6.1 | BSD-3-Clause |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; bundles OpenBLAS (see below) |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause |
| pillow | 12.2.0 | MIT-CMU; bundles image libraries (see below) |
| pydantic | 2.13.4 | MIT |
| pydantic-core | 2.46.4 | MIT |
| pyparsing | 3.3.2 | MIT |
| python-dateutil | 2.9.0.post0 | BSD-3-Clause; newer contributions also Apache-2.0 |
| scikit-image | 0.26.0 | BSD-3-Clause |
| scipy | 1.17.1 | BSD-3-Clause; bundles OpenBLAS (see below) |
| six | 1.17.0 | MIT |
| starlette | 1.7.0 | BSD-3-Clause |
| tifffile | 2026.5.15 | BSD-3-Clause |
| typing-extensions | 4.15.0 | PSF-2.0 |
| typing-inspection | 0.4.2 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |

In the PyInstaller trial the build environment still held napari's tree, and 13 more
packages came into the bundle through optional imports: attrs (MIT), certifi
(MPL-2.0), charset-normalizer (MIT), markdown-it-py (MIT), mdurl (MIT), psutil
(BSD-3-Clause), pydantic-extra-types (MIT), Pygments (BSD-2-Clause), python-dotenv
(BSD-3-Clause), PyYAML (MIT), rich (MIT), setuptools (MIT), and tzdata (Apache-2.0),
along with modules of pywin32 (PSF-2.0). Building from an environment without
napari is expected to leave them out; that has not been verified.

**Findings**

- Nothing in the runtime bundle is under the AGPL or a non-commercial license. The
  only GPL-licensed code is PyInstaller's bootloader and the GCC runtime inside
  OpenBLAS, and both come with exceptions that allow distribution under any terms:
  - **PyInstaller's bootloader** is GPL-2.0-or-later with the Bootloader Exception.
  - **GCC runtime in OpenBLAS.** NumPy and SciPy bundle OpenBLAS, which contains the
    GCC runtime under GPL-3.0-or-later with the GCC Runtime Library Exception 3.1.
    This applies to any bundler.
- **certifi** (MPL-2.0) came in only through napari's tree and is not needed;
  building without napari leaves it out.
- **Pillow** bundles image libraries that its license file lists. FreeType is
  offered under the FreeType License or GPL-2.0; Proteia uses it under the FreeType
  License, which asks for a credit in the documentation.
- **PySide6 (LGPL-3.0)** and Qt are not in the bundle. #57 removes them from the
  dependencies, and `THIRD_PARTY_NOTICES.md` drops them then.
- **WiX** (MS-RL) matters only if an MSI is built later: the Briefcase MSI embedded
  WiX's installer bitmaps, and WiX 6 and later add a maintenance fee for users who
  earn revenue with it. Not reviewed further.
- **Notices.** The binary distribution has to carry the license texts the trial
  bundle lacked: the Python runtime and the libraries built into it, the VC++
  runtime terms, every bundled package (including the notices for the libraries
  bundled inside NumPy, SciPy, and Pillow), and the fonts. `THIRD_PARTY_NOTICES.md`
  already promises a generated inventory before the first release; the installer
  build generates it.

## Open questions

1. **Code signing.** How the installer, the uninstaller, and `Proteia.exe` are
   signed, for example with an organization-validated certificate, a cloud signing
   service, or a signing program for open-source projects; eligibility was not
   checked. Third-party native libraries in the bundle stay unsigned either way.
   Decided before the first public release.
2. **Publisher name.** The name shown in Windows' installed-apps list, in the
   installer, and in the version resource. Once the installer is signed, SmartScreen
   shows the certificate's subject, so the two should match; with a fixed
   application ID, changing the name later does not affect upgrades. Decided before
   the first public release.
3. **Inno Setup's commercial license request.** Reviewed before a commercial
   release (see License check).

## Alternatives considered

**Briefcase with WiX (MSI)**

- Tried; see above. It gives an MSI, installs packages as ordinary files with their
  metadata, and signs through its own options.
- Not chosen because it starts more slowly and is larger, its console build changes
  `PATH`, it needs changes to the project metadata, and its MSI was not installed
  for real in the trial. It would be the better choice if an MSI and built-in
  signing came to outweigh start time and size.

**PyInstaller in one-file mode**

- Tried. One file, but it unpacks about 190 MB on every launch, took 5.2–5.7 s on
  every start, and failed under a very long `TEMP` path.

**PyInstaller with an MSI authored in WiX**

- Not tried. It would pair PyInstaller's start time with an MSI, at the cost of
  writing the WiX authoring by hand and of WiX's licensing (see License check).

**Other freezers and installer builders** (for example Nuitka, cx_Freeze, pynsist)

- Not tried: both tools that ADR 0001 named produced a working installer, so no
  further tool was evaluated.
