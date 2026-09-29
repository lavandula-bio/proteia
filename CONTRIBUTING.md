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

To run the app locally:

```bash
uv run proteia
```

This starts a local server on `127.0.0.1` and opens Proteia in your default
browser. Quit Proteia on the page, or press Ctrl+C in the terminal, to stop
it. Running the command again while Proteia runs opens the running app.

With image files, `uv run proteia PATH...` opens Proteia with those images
offered for import into a new project (the page asks each image's kind,
membrane and polarity), and works while Proteia is already running.

## Project layout

```
src/proteia/        Python package
  core/             GUI-independent data model, quantification, QC
  viz/              chart rendering (PNG, SVG, PDF)
  web/              local web app: server and browser client (calls into core;
                    never the reverse)
tests/              test suite
docs/adr/           architecture decision records
```

The separation between `core` and `web` is intentional (see
[ADR 0002](docs/adr/0002-gui-local-web-app.md)): the core must not depend on
the front end, so another front end can be added later without rewriting the
analysis code.

## Workflow

Proteia uses trunk-based development: `main` is the only long-lived branch and
every change reaches it through a short-lived branch and a pull request.

- Start from an issue. Open one with the task or bug report template (or pick
  an existing one) before you branch, so the work has a number to reference.
  The maintainer labels it (`bug`, `feature`, `docs`, `refactor`, `chore`,
  `test`, plus `priority:*`) and assigns it to the milestone of the next
  release.
- Branch from `main`: `<type>/<issue>-<short-description>` in kebab-case, where
  `<type>` is the Conventional Commits type of the work (`feat`, `fix`, `docs`,
  `chore`, `refactor`, `test`). Example: `feat/42-csv-export`. The type
  follows the work, not the label: a `feature` issue becomes a `feat/` branch,
  a `bug` issue becomes a `fix/` branch.
- One logical change per pull request. Write branch commits, the pull request
  title, and the pull request body in English, using
  [Conventional Commits](https://www.conventionalcommits.org/) for commit
  messages; keep branch commits atomic.
- Open a pull request against `main` (it is protected; no direct pushes) and
  fill in every section of the pull request template. Reference the issue
  with `Closes #42`, or `Refs #42` when the issue stays open for further
  pull requests.
- Pull requests are squash-merged into one commit on `main`: the PR title
  (plus GitHub's ` (#N)` suffix) becomes the commit subject, so it must be a
  valid Conventional Commits subject such as
  `feat(export): add per-lane CSV export`; the PR body becomes the commit
  body, so write it for `git log`: prose and short lines, since GitHub
  reflows it at 72 columns, and delete the template's comment prompts.
- The maintainer merges only after the `ci` check is green and an
  independent review (a person, or an automated review pass run by the
  maintainer) whose findings have been resolved.
- Branches in this repository are deleted automatically on merge; delete
  branches in your own fork yourself.
- Releases are tagged on `main` using SemVer.

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0. New source files should carry the header:

```python
# SPDX-License-Identifier: Apache-2.0
```

Prefer permissively licensed dependencies (MIT, BSD, Apache-2.0, ISC). Ask
before adding an LGPL, MPL or dual-licensed one; avoid GPL, AGPL, SSPL and
non-commercial licenses.
