# ADR 0001: GUI foundation — napari desktop application

- **Status**: Accepted
- **Date**: 2026-05-31
- **Deciders**: Roger Huang

## Context

Proteia's core workflow requires interactive visualization. Users must view a blot
image and define or adjust regions of interest (ROIs) for lane and band
quantification. Automatic detection cannot be assumed to be 100% correct, so a
human-in-the-loop interface for reviewing and correcting ROIs is a core function,
not an optional add-on.

The tool is local-first and offline by design (no server, no telemetry) and is
implemented in Python to use the scientific imaging ecosystem (scikit-image,
OpenCV, NumPy/SciPy).

Two foundations were considered for the interactive interface:

- A Python desktop application built on **napari**.
- A **browser-based (web)** interface.

## Decision

Use **napari** (BSD-3-Clause) as the GUI foundation for the v0.1 MVP, running as a
local desktop application.

The quantification and QC logic will be implemented as a standalone **core
library**; the GUI is a thin layer that calls into it. The core must not depend on
the GUI.

## Consequences

**Positive**

- napari provides GPU-accelerated image display and an interactive shapes layer
  (ROI editing) out of the box — the fastest path to a working interactive
  quantification loop.
- Single-language stack (Python), matching the quantification and QC code, which
  lowers the maintenance burden for a small team.
- A desktop application is offline by nature, satisfying the local-first constraint
  with no extra work.
- License is permissive (BSD-3-Clause) and compatible with Apache-2.0.

**Trade-offs**

- Desktop use requires a Python environment (pip/conda). This is acceptable for the
  technical alpha audience but less convenient for non-technical users. Packaging
  into a standalone app (e.g. PyInstaller or briefcase) is deferred to a later
  phase.
- UI customization is bounded by napari/Qt conventions. Interaction models that do
  not fit an "image viewer + control panels" paradigm would be harder to build.

**Reversibility**

- Because the core logic is decoupled from the GUI, an alternative front-end (web or
  CLI) can be added later against the same core without rewriting the analysis code.
  This decision sets the near-term build path and UX baseline, not a permanent
  constraint.

## Alternatives considered

**Web (browser-based) interface**

- Offers a higher ceiling for bespoke UI design and the lowest install friction (no
  install), which is best for reaching non-technical users.
- Rejected for v0.1 because it requires building image and ROI interaction from
  scratch, introduces a second language (JavaScript) alongside Python, and
  complicates the offline/local-first requirement (a local server or in-browser
  Python runtime). These costs are not justified while the v0.1 audience is
  technical alpha users who can install a Python package. May be revisited in a
  later phase if non-technical adoption requires it.
