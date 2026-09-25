# ADR 0002: GUI foundation — local web application

- **Status**: Accepted
- **Date**: 2026-09-25
- **Deciders**: Roger Huang
- **Supersedes**: [ADR 0001](0001-gui-foundation-napari.md)

## Context

ADR 0001 chose napari as the GUI foundation for an audience of technical users who
can install a Python package. It rejected a browser-based interface for that
audience and noted that the choice could be revisited if non-technical adoption
required it. Proteia is meant for wet-lab researchers, most of whom do not manage
Python environments, so that condition now applies. Building on napari has also
exposed limits in interaction and in distribution.

**Interaction.** napari is a general-purpose n-dimensional viewer. Its window
carries viewer chrome (layer list, layer controls, dimension and view buttons) that
plays no part in quantifying a blot and that a non-technical user has to learn to
ignore. Proteia moves and edits boxes through the Shapes layer's selection mode,
which fits poorly with how Proteia manages boxes: because the layer is rebuilt
after each edit, clicks in selection mode can raise errors inside napari's
selection handling (napari catches them and no data is lost), dragging a box is not
smooth, and a placed box is awkward to fine-tune. A reliable fix means replacing
selection mode with custom mouse handling, which is most of the work a purpose-built
interface would need anyway.

**Distribution.** A one-step installer that works on a machine without Python is
now a requirement, and the interface must be usable without knowing Python. napari
needs a Qt binding; Proteia uses PySide6, which is available under LGPL-3.0. napari
and its Qt binding dominate the dependency tree: in the Windows development
environment the installed PySide6 package alone takes about 640 MB, and 94 of the
116 runtime packages in the lockfile (counted across all platforms) are there only
because of `napari[pyside6]`. An installer that bundles Qt also conveys
LGPL-licensed libraries, with duties that every release must meet: ship the GPL
and LGPL texts with a prominent notice, make the corresponding source of the
bundled Qt and PySide6 libraries available, let users replace those libraries with
modified versions (in practice, by shipping them as replaceable shared libraries),
and not restrict reverse engineering for debugging such modifications. The
`pyside6` package that napari's extra installs also includes Qt add-on modules that
are available as open source only under GPL-3.0 (for example, Qt Charts and Qt Data
Visualization), so an installer would also have to leave them out.

**What does not change.** The application stays local-first and works offline: no
telemetry, and data stays on the user's machine. ADR 0001 met this with no server
at all; this ADR keeps the same guarantees with a server that is reachable only
from the same machine. The analysis core is already GUI-independent, as ADR 0001
required: `proteia.core` does not import napari or Qt. Application state and the
orchestration of user actions, however, still live in the napari module
(`proteia.gui.app`), so they cannot be tested or reused without it.

## Decision

Replace napari with a **local web application**: a Python server on the user's
machine does all the work, and the user's own browser displays the interface.

**Server**

- A Python server (FastAPI on Uvicorn) binds to `127.0.0.1` only, on a random free
  port chosen at startup. It makes no outbound network connections.
- Each session gets its own access token: every launch generates a new random
  token, the launcher opens the default browser at `http://127.0.0.1:<port>/` with
  the token in the URL, and the server rejects requests without a valid token.
- Other web pages must not be able to use the token. The client presents it
  explicitly, for example in a request header, rather than relying on anything the
  browser attaches on its own. If a cookie is used as well, it is HttpOnly and
  SameSite=Strict, and the server rejects state-changing requests whose `Origin`
  is not the application's own.
- The server also rejects requests whose `Host` header does not name the loopback
  address and port it is serving, which blocks DNS-rebinding attacks from web
  pages. Requests that other pages send straight to the loopback port carry a valid
  `Host` header; the token is what stops them.
- FastAPI's interactive API documentation pages (`/docs` and `/redoc`) are turned
  off, since they load their scripts and styles from a CDN.
- All computation and image rendering happen on the server: quantification,
  statistics, charts, and display previews of the scans (including mapping 16-bit
  data to a displayable range). Analysis always uses the full-resolution image
  data, never the display previews.

**Project layer**

- Every user action (import an image, declare lanes, add a protein, place, move or
  remove a box, compute, export, save) is a function on a GUI-independent project
  layer. The HTTP routes are thin adapters over these functions, and the web UI acts
  only through them. The project layer is tested directly, without a browser.
- A project is stored as a project folder: a `project.json` file with the project
  data, plus the imported images and exports.
- The browser sends each imported image to the server as a raw request body, one
  file per request, and the server copies it into the project folder, so no
  multipart form parser is needed.

**Client**

- A thin browser client in plain HTML, CSS, and JavaScript, served by the local
  server as static files. It draws server-rendered images and box overlays on an
  HTML canvas and handles pointer input (pan, zoom, drawing and moving boxes).
- No Node.js build step, no npm dependencies, and no front-end framework: the files
  in the repository are the files the browser runs.
- The client loads nothing from the network (no CDN scripts or web fonts), so the
  application works offline.

**napari and PySide6**

- The napari GUI stays available during the transition. napari and PySide6 leave
  the runtime dependencies once the web UI reaches parity with it.

**New runtime dependencies**

| Package | Role | License |
| --- | --- | --- |
| FastAPI | HTTP routes and request validation (uses the existing pydantic) | MIT |
| Starlette (via FastAPI) | ASGI toolkit: routing, responses, static files | BSD-3-Clause |
| Uvicorn | ASGI server | BSD-3-Clause |
| AnyIO (via Starlette) | Asynchronous I/O layer | MIT |
| h11 (via Uvicorn) | HTTP/1.1 protocol implementation | MIT |

Licenses were checked against each project's PyPI metadata and LICENSE file.
Uvicorn is installed without its `standard` extra; its default asyncio event loop
and h11 parser are sufficient for a single local user. None of these five packages
is in `uv.lock` yet; everything else the stack requires is already locked. Three of
those requirements are currently locked only through napari and stay when napari
leaves: annotated-doc (MIT), click (BSD-3-Clause), and idna (BSD-3-Clause). On
Windows, the locked version of click also requires colorama (BSD-3-Clause). Exact
versions are recorded in `uv.lock` when the packages are added, and
`THIRD_PARTY_NOTICES.md` is updated at that point and again when napari and
PySide6 are removed.

Choosing a tool to package the installer is out of scope and will be recorded in a
separate ADR.

## Consequences

**Positive**

- The interaction is designed for the task (drawing a box around a row of bands,
  moving and resizing boxes, flags drawn on the image) rather than fitted into a
  general viewer's modes and chrome.
- Qt leaves the product. Once napari and PySide6 are removed, the locked runtime
  dependency set shrinks from 116 packages to about 30 (across all platforms,
  including the new web stack), and the redistribution duties that come with
  bundling Qt no longer apply. All new dependencies are under permissive licenses
  compatible with Apache-2.0.
- The installer only needs to carry Python and Python packages; the browser the
  user already has renders the interface.
- One repository and one toolchain (uv) cover both server and client; the client
  needs no build step.
- Every user action is a tested project-layer function, so what each action does is
  covered by pytest without a GUI, and results, saving, and exports follow one code
  path whatever the front-end.
- The offline requirement holds: the server listens only on loopback, the client
  loads nothing from the network, and there is no telemetry.

**Trade-offs**

- Image display and box interaction are written from scratch: pan, zoom,
  hit-testing, drawing, moving, and resizing boxes, and keyboard shortcuts, all of
  which napari provided.
- A second language (JavaScript) enters the codebase. The client needs its own
  review and testing approach; CI currently runs only Python checks (ruff, pytest).
- Every edit is a request to the local server, and so is any view change that
  needs a new preview. Previews of large 16-bit scans must be sized so that
  interaction stays responsive.
- The browser does not give the server a file-system path: an imported image
  arrives as file contents only, and choosing where a project folder lives needs
  its own design (for example, a projects directory managed by the application, or
  a folder picker that the application serves).
- A local server is an attack surface on the user's machine. Loopback binding, the
  per-session token, and the Host-header check are required, and tests must show
  that requests without the token, with a foreign `Host` header, or sent cross-site
  by another page are rejected. Exposing the server beyond loopback would need a
  new decision.
- The application lifecycle needs explicit handling: closing the browser tab does
  not stop the server, so the app needs a clear way to quit, and launching it again
  while it is running must behave predictably.
- Until the web UI reaches parity, two front-ends coexist and napari remains a
  dependency.

**Reversibility**

- The analysis core and the project layer do not depend on the front-end, and the
  HTTP layer is a thin adapter over the project layer. Another front-end (a Qt
  application, or a native window that hosts the same client) can be built on the
  project layer later without rewriting the analysis code.
- Removing napari can be undone through version control, but bringing it back would
  also bring back Qt's size and redistribution duties.

## Alternatives considered

**Keep napari (with a trimmed interface)**

- Lowest immediate cost: the current GUI works, and napari supplies image display
  and a shapes layer.
- Not chosen because the needed fixes (custom mouse handling for boxes, hiding
  general viewer controls) amount to working around napari rather than using it,
  and the installer would still carry Qt's size and redistribution duties.

**Custom Qt application (PySide6)**

- Full control over the interface, a native window, and a single language (Python).
- Not chosen because it keeps what this decision removes from the product: Qt's
  size and redistribution duties. Qt's Graphics View framework would supply pan,
  zoom, hit-testing, and movable items, which saves part of the web client's work,
  but the box rules (shared size, resizing, no overlap) would still be custom code.

**Electron or Tauri shell**

- A native window around a web interface, without opening the user's browser.
- Not chosen because it adds a second toolchain and runtime on top of the Python
  server, which would still have to be bundled: Electron ships its own Chromium and
  Node.js runtime and is built with the npm toolchain; Tauri relies on the system
  webview and needs a Rust toolchain. A dedicated window does not justify that cost,
  and the same client can still be hosted in a native window later.

**Pyodide (Python in the browser via WebAssembly)**

- No analysis server: the analysis code runs inside the browser (the page and its
  WebAssembly packages still have to be served over HTTP, locally or from a host).
- Not chosen because the scientific stack (NumPy, SciPy, scikit-image, matplotlib)
  would have to be shipped and loaded as WebAssembly packages; computation runs
  slower and in a memory-limited 32-bit WebAssembly environment, which matters for
  large 16-bit scans; and reading and writing a project folder on disk depends on
  browser file-system APIs that are not available in every browser. The analysis
  would also run in a runtime that the test suite does not exercise.
