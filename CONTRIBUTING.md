# Contributing to Proteia

Thanks for your interest in Proteia. This guide covers local setup and the
development workflow.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

## Setup

```bash
# Create the environment and install all dependencies (including dev tools)
uv sync

# Run the test suite
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```

To launch napari locally (requires a display):

```bash
uv run napari
```

## Project layout

```
src/proteia/        Python package
  core/             GUI-independent data model, quantification, QC
  gui/              napari GUI layer (calls into core; never the reverse)
tests/              test suite
docs/adr/           architecture decision records
```

The separation between `core` and `gui` is intentional (see
[ADR 0001](docs/adr/0001-gui-foundation-napari.md)): the core must not depend
on the GUI, so an alternative front-end can be added later without rewriting
the analysis code.

## Workflow

Proteia uses trunk-based development: `main` is the only long-lived branch and
every change reaches it through a short-lived branch and a pull request.

- Start from an issue. Open one (or pick an existing one) before you branch, so
  the work has a number to reference.
- Branch from `main`: `<type>/<issue>-<short-description>` in kebab-case, where
  `<type>` is the Conventional Commits type of the work (`feat`, `fix`, `docs`,
  `chore`, `refactor`, `test`). Example: `feat/42-csv-export`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) in English.
  Keep commits atomic.
- Open a pull request against `main` and reference the issue in the description
  (`Closes #42`). `main` is protected and does not accept direct pushes.
- Delete the branch after the pull request is merged.
- Releases are tagged on `main` using SemVer.

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0. New source files should carry the header:

```python
# SPDX-License-Identifier: Apache-2.0
```

Prefer permissively licensed dependencies (MIT/BSD/Apache/LGPL); avoid
GPL/AGPL or non-commercial licenses.
