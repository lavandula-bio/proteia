# SPDX-License-Identifier: Apache-2.0
"""The repository stores its text files with LF line endings and in UTF-8
without a BOM, as on the Linux machines the project started on.
``.gitattributes`` has git store and check out every text file with LF (#174);
this test is the second line: it catches a file stored with CRLF before that or
outside it, and a BOM, which nothing else rejects (``ruff format`` keeps a
file's own line endings and BOM).

What is checked is the content git stores (``git ls-files --eol``), not the
working tree, which a checkout may convert."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _stored_eols() -> dict[str, str]:
    """Each tracked file's stored line endings as git reports them: ``lf``,
    ``crlf``, ``mixed``, ``none``, or ``-text`` for a binary file."""
    try:
        listing = subprocess.run(
            ["git", "ls-files", "--eol"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    eols = {}
    for line in listing.splitlines():
        info, _, path = line.partition("\t")
        eols[path] = info.split()[0].removeprefix("i/")
    return eols


def test_every_text_file_is_stored_with_lf_line_endings():
    eols = _stored_eols()
    assert "tests/test_line_endings.py" in eols
    assert {path for path, eol in eols.items() if eol in ("crlf", "mixed")} == set()


def test_no_text_file_starts_with_a_utf8_bom():
    eols = _stored_eols()
    text = [path for path, eol in eols.items() if eol != "-text"]
    assert "pyproject.toml" in text
    marked = [path for path in text if (ROOT / path).read_bytes().startswith(b"\xef\xbb\xbf")]
    assert marked == []
