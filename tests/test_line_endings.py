# SPDX-License-Identifier: Apache-2.0
"""The repository's text files use LF line endings and UTF-8 without a BOM, as
on the Linux machines the project started on. Nothing else enforces it: the
repository has no ``.gitattributes``, and ``ruff format`` keeps a file's own
line endings, so a file written with CRLF on Windows would otherwise pass."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = frozenset(
    {".py", ".js", ".css", ".html", ".md", ".toml", ".yml", ".yaml", ".json", ".txt", ".lock"}
)
TEXT_NAMES = frozenset({"LICENSE", ".gitignore", ".python-version"})
# Not the repository's own files: environments, caches, and the private and
# tool folders that git ignores.
SKIPPED = frozenset(
    {".git", ".venv", ".claude", "_local", "__pycache__", ".pytest_cache", ".ruff_cache", "dist"}
)


def _text_files() -> list[Path]:
    found = []
    stack = [ROOT]
    while stack:
        folder = stack.pop()
        for path in folder.iterdir():
            if path.is_dir():
                if path.name not in SKIPPED:
                    stack.append(path)
            elif path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
                found.append(path)
    return sorted(found)


def test_the_text_files_are_found():
    names = {path.relative_to(ROOT).as_posix() for path in _text_files()}
    assert {"pyproject.toml", "tests/test_line_endings.py", "src/proteia/__init__.py"} <= names


@pytest.mark.parametrize("path", _text_files(), ids=lambda path: path.relative_to(ROOT).as_posix())
def test_a_text_file_has_lf_line_endings_and_no_bom(path):
    data = path.read_bytes()
    assert b"\r" not in data, "CR found: write the file with LF line endings"
    assert not data.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM found"
