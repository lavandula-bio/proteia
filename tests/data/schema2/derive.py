# SPDX-License-Identifier: Apache-2.0
"""Print a golden schema-2 fixture's derived output as canonical JSON (#221).

    uv run python tests/data/schema2/derive.py f1 [--out FILE]

FIXTURE is a case (``f1``, ``f2``, ``f3``) or the path of a fixture folder,
named anything: its ``stored.json`` names its case. The output is what
``make_fixtures.derived_output`` gives with the build that runs it, sorted and
indented the same way every time, so that two builds (a pull request's base
and its head) can be compared byte for byte on one machine. The fixture is
read, never written: it is copied to a temporary folder first.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import make_fixtures  # this script's folder is first on sys.path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("fixture", help="f1, f2, f3, or a fixture folder")
    parser.add_argument("--out", type=Path, help="write here instead of to stdout")
    args = parser.parse_args(argv)
    if args.fixture in make_fixtures.CASES:
        folder = make_fixtures.HERE / args.fixture
    else:
        folder = Path(args.fixture).resolve()
    try:
        case = make_fixtures.case_of(folder)
    except ValueError as exc:
        print(f"derive.py: {exc}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="schema2-", ignore_cleanup_errors=True) as tmp:
        derived = make_fixtures.derived_output(folder, Path(tmp), case)
    data = make_fixtures.canonical_json(derived)
    if args.out is None:
        sys.stdout.buffer.write(data)
    else:
        args.out.write_bytes(data)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")  # a console of any code page
    sys.exit(main())
