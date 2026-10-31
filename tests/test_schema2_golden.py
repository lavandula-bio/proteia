# SPDX-License-Identifier: Apache-2.0
"""Golden schema-2 projects (#221): what projects built at 05b4af5 store, and
what the current build derives from them.

``tests/data/schema2/<case>/`` holds three small projects built through the
operations on synthetic images (``make_fixtures.py`` says what each holds):
``project.json`` and ``images/`` as 05b4af5 wrote them; ``stored.json``, every
stored value they hold, written once at that commit and never again; and
``derived.json``, what the current build derives from them (the results with
their notices, the exported lane tables, the calibration fits as the page
shows them, the history check, and the stored values and log of a fresh build
of the same project from the same image files).

Stored values are copied, never recomputed, so they match exactly on every
platform. Derived floats pass through logarithms and statistics functions that
may differ in the last bits between platforms: they match within a relative
1e-12, and a lane-table cell may be one unit off in its last decimal only where
the golden marks its value as within 1e-12 of a rounding midpoint. Text,
integers, booleans, None, keys and list order match exactly.

A pull request that means to change the derived output refreshes it with
``make_fixtures.REFRESH_COMMAND`` and says why in its description; the frozen
files never change.
"""

import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from proteia.core import operations as ops

DATA = Path(__file__).parent / "data" / "schema2"
FROZEN = "05b4af5b08e51b450436e9446651f5650bff48e4"
# How far the frozen JPEG's pixels may lie from those drawn: its compression
# (measured: at most 8 levels, 1.64 on average), not another drawing (another
# noise seed: 12 levels, 2.46-2.51 on average).
JPEG_MAX_ERROR, JPEG_MEAN_ERROR = 10, 1.9


def _load(name: str):
    """A script of ``tests/data/schema2`` as a module of its own name."""
    spec = importlib.util.spec_from_file_location(f"schema2_{name}", DATA / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fixtures = _load("make_fixtures")
CASES = ("f1", "f2", "f3")


def _golden(case: str, name: str) -> dict:
    return json.loads((DATA / case / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def derived(tmp_path_factory) -> dict[str, dict]:
    """Each case's derived output from the current build, computed once."""
    work = tmp_path_factory.mktemp("schema2-derived")
    return {case: fixtures.derived_output(case, work / case, root=DATA) for case in CASES}


def _opened(case: str, tmp_path: Path):
    """A copy of the frozen project, opened without autosave."""
    folder = tmp_path / case
    fixtures.copy_project(DATA / case, folder)
    return ops.open_project(folder, autosave=None)


# --- The frozen files ---


def test_the_cases_are_what_the_generator_builds():
    assert fixtures.CASES == CASES
    assert fixtures.FROZEN_COMMIT == FROZEN


@pytest.mark.parametrize("case", CASES)
def test_frozen_fixtures_are_built_at_05b4af5(case):
    stored = _golden(case, "stored.json")
    assert stored["built_at"] == FROZEN
    assert stored["version"] == "0.1.0.dev0"
    project = json.loads((DATA / case / "project.json").read_text(encoding="utf-8"))
    assert project["schema_version"] == 2
    assert {entry["version"] for entry in project["log"]} == {"0.1.0.dev0"}


@pytest.mark.parametrize("case", CASES)
def test_stored_golden_matches(case, tmp_path):
    session = _opened(case, tmp_path)
    golden = _golden(case, "stored.json")
    expected = {k: v for k, v in golden.items() if k not in ("built_at", "version")}
    assert fixtures.differences(expected, fixtures.stored_values(session.project)) == []


@pytest.mark.parametrize("case", CASES)
def test_derived_golden_matches(case, derived):
    fixtures.check_derived(_golden(case, "derived.json"), derived[case], case)


def test_derived_failure_prints_the_refresh_command(derived):
    golden = _golden("f1", "derived.json")
    degraded = copy.deepcopy(derived["f1"])
    degraded["history_issues"].append("undo_mismatch")
    with pytest.raises(AssertionError) as failed:
        fixtures.check_derived(golden, degraded, "f1")
    message = str(failed.value)
    assert fixtures.REFRESH_COMMAND in message
    assert "\n  $.history_issues: 0 items != 1\n" in message  # the golden's, then the build's
    assert fixtures.REFRESH_COMMAND == (
        "uv run python tests/data/schema2/make_fixtures.py --refresh-derived"
    )


# --- The comparison catches each degraded case (#221) ---


def _first_float(values: list) -> int:
    return next(i for i, v in enumerate(values) if isinstance(v, float))


def test_comparator_catches_stored_ulp():
    golden = _golden("f1", "stored.json")
    actual = copy.deepcopy(golden)
    band = actual["proteins"][0]["bands"][0]
    band["net"] = math.nextafter(band["net"], math.inf)
    assert band["net"] != golden["proteins"][0]["bands"][0]["net"]
    found = fixtures.differences(golden, actual)
    assert found == [
        f"$.proteins[0].bands[0].net: {golden['proteins'][0]['bands'][0]['net']!r}"
        f" != {band['net']!r}"
    ]
    # Within the derived tolerance it would pass: exactness is what catches it.
    assert fixtures.differences(golden, actual, rel_tol=1e-12) == []


def test_stored_check_catches_a_net_one_ulp_off_in_the_file(tmp_path):
    folder = tmp_path / "f1"
    fixtures.copy_project(DATA / "f1", folder)
    path = folder / "project.json"
    text = path.read_text(encoding="utf-8")
    old = json.loads(text)["batch"]["proteins"][0]["bands"][0]["net"]
    new = math.nextafter(old, math.inf)
    assert text.count(f'"net": {old!r}') == 1
    changed = text.replace(f'"net": {old!r}', f'"net": {new!r}')
    path.write_text(changed, encoding="utf-8", newline="\n")
    golden = _golden("f1", "stored.json")
    expected = {k: v for k, v in golden.items() if k not in ("built_at", "version")}
    project = ops.open_project(folder, autosave=None).project
    found = fixtures.differences(expected, fixtures.stored_values(project))
    assert [f.split(":")[0] for f in found] == ["$.content_hash", "$.proteins[0].bands[0].net"]


def test_comparator_catches_derived_1e9(derived):
    golden = _golden("f1", "derived.json")
    actual = copy.deepcopy(derived["f1"])
    values = actual["results"]["series"][0]["normalized"]
    i = _first_float(values)
    values[i] *= 1.0 + 1e-9
    found = fixtures.derived_differences(golden, actual)
    assert [f.split(":")[0] for f in found] == [f"$.results.series[0].normalized[{i}]"]
    # A change of 1e-13 is within the tolerance.
    actual = copy.deepcopy(derived["f1"])
    actual["results"]["series"][0]["normalized"][i] *= 1.0 + 1e-13
    assert fixtures.derived_differences(golden, actual) == []
    assert fixtures.differences(golden["results"], actual["results"]) != []


def test_comparator_catches_notice_removed(derived):
    golden = _golden("f1", "derived.json")
    actual = copy.deepcopy(derived["f1"])
    removed = actual["results"]["notices"].pop(0)
    assert removed["code"]
    found = fixtures.derived_differences(golden, actual)
    assert found and all(f.startswith("$.results.notices") for f in found)


def test_comparator_catches_notice_text(derived):
    golden = _golden("f1", "derived.json")
    actual = copy.deepcopy(derived["f1"])
    notice = actual["results"]["notices"][-1]
    text = notice["message"]
    notice["message"] = text[:-1] + ("," if text[-1] != "," else ";")
    found = fixtures.derived_differences(golden, actual)
    last = len(actual["results"]["notices"]) - 1
    assert [f.split(":")[0] for f in found] == [f"$.results.notices[{last}].message"]


def test_comparator_catches_series_swap(derived):
    golden = _golden("f1", "derived.json")
    actual = copy.deepcopy(derived["f1"])
    series = actual["results"]["series"]
    assert len(series) == 2 and series[0]["target"] != series[1]["target"]
    series[0], series[1] = series[1], series[0]
    found = fixtures.derived_differences(golden, actual)
    assert any(f.startswith("$.results.series[0].target:") for f in found)
    assert any(f.startswith("$.results.series[1].target:") for f in found)


def test_comparator_catches_table_cell(derived):
    golden = _golden("f1", "derived.json")
    for change in ("number", "text"):
        actual = copy.deepcopy(derived["f1"])
        table = actual["lane_tables"][0]
        rows = fixtures.table_rows(table["csv"])
        if change == "number":
            # The first net of the first protein, one unit up in its last decimal.
            column = 4
            rows[1][column] = repr(round(float(rows[1][column]) + 0.001, 3))
        else:
            column = 2
            rows[1][column] = rows[1][column] + "x"
        table["csv"] = fixtures.table_text(rows)
        found = fixtures.derived_differences(golden, actual)
        assert [f.split(":")[0] for f in found] == [f"$.lane_tables[0].csv[1][{column}]"], change
    # Golden first, then the build's, in every line.
    old = fixtures.table_rows(golden["lane_tables"][0]["csv"])[1][2]
    assert found == [f"$.lane_tables[0].csv[1][2]: {old!r} != {old + 'x'!r}"]


def test_comparator_catches_lane_tables_missing_or_dropped(derived):
    golden = _golden("f1", "derived.json")
    actual = copy.deepcopy(derived["f1"])
    actual["lane_tables"].pop()
    assert fixtures.derived_differences(golden, actual) == ["$.lane_tables: 2 items != 1"]
    del actual["lane_tables"]
    found = fixtures.derived_differences(golden, actual)
    assert (
        len(found) == 1
        and found[0].startswith("$.lane_tables: [{")
        and found[0].endswith(" != None")
    )


def _table(csv: str, near: list | None = None, label: str | None = None) -> dict:
    return {"label": label, "csv": csv, "near_midpoint": near or []}


def test_table_cell_one_unit_off_only_at_a_midpoint():
    golden = _table("lane,a\r\n1,2.345\r\n")
    one_up = _table("lane,a\r\n1,2.346\r\n")
    two_up = _table("lane,a\r\n1,2.347\r\n")
    assert fixtures.table_differences(golden, one_up, "$") == ["$.csv[1][1]: '2.345' != '2.346'"]
    golden["near_midpoint"] = [[1, 1, 3]]
    assert fixtures.table_differences(golden, one_up, "$") == []
    assert fixtures.table_differences(one_up, _table("lane,a\r\n1,2.345\r\n", [[1, 1, 3]]), "$")
    assert (
        fixtures.table_differences(_table("lane,a\r\n1,2.346\r\n", [[1, 1, 3]]), golden, "$") == []
    )
    assert fixtures.table_differences(golden, two_up, "$") == ["$.csv[1][1]: '2.345' != '2.347'"]
    assert fixtures.table_differences(golden, _table("lane,a\r\n1,x\r\n"), "$") != []
    # One unit of a 6-decimal column is not one unit of a 3-decimal one.
    golden["near_midpoint"] = [[1, 1, 6]]
    assert fixtures.table_differences(golden, one_up, "$") != []
    # Only the cell marked.
    golden["near_midpoint"] = [[1, 0, 3]]
    assert fixtures.table_differences(golden, one_up, "$") != []
    # Rows and columns must match in number, and the label exactly.
    longer = _table("lane,a\r\n1,2.345\r\n2,1.0\r\n")
    assert fixtures.table_differences(golden, longer, "$") == ["$.csv: 2 rows != 3"]
    wider = _table("lane,a,b\r\n1,2.345,\r\n")
    assert fixtures.table_differences(golden, wider, "$") == [
        "$.csv[0]: 2 cells != 3",
        "$.csv[1]: 2 cells != 3",
    ]
    labelled = _table("lane,a\r\n1,2.345\r\n", label="All lanes")
    assert fixtures.table_differences(golden, labelled, "$") == ["$.label: None != 'All lanes'"]


def test_near_midpoint():
    assert fixtures.near_midpoint(1.2345, 3)  # 1234.5 on a 3-decimal grid
    assert not fixtures.near_midpoint(1.2346, 3)
    assert not fixtures.near_midpoint(1.2345, 6)
    assert fixtures.near_midpoint(0.000015, 5)
    assert not fixtures.near_midpoint(0.0000125, 5)
    assert fixtures.near_midpoint(-2.5, 0)
    assert not fixtures.near_midpoint(2.5 + 1e-9, 0)
    assert fixtures.near_midpoint(2.5, 0, rel=0.0)  # on the midpoint itself
    assert fixtures.near_midpoint(2.5 * (1 + 0.9e-12), 0)
    assert not fixtures.near_midpoint(2.5 * (1 + 1.1e-12), 0)


def test_midpoint_cells():
    text = "lane,net,ratio\r\n1,2.345,0.25\r\n2,2.346,0.25\r\n"
    unrounded = {
        3: "lane,net,ratio\r\n1,2.3445,0.25\r\n2,2.3461,0.25\r\n",
        6: "lane,net,ratio\r\n1,2.345,0.2500004999999\r\n2,2.346,0.25000049\r\n",
    }
    assert fixtures.midpoint_cells(text, unrounded) == [[1, 1, 3], [1, 2, 6]]
    assert fixtures.midpoint_cells(text, {3: text}) == []


def test_comparator_is_strict_about_types_keys_and_order():
    d = fixtures.differences
    assert fixtures.REL_TOL == 1e-12
    assert d({"a": 1}, {"a": 1}) == []
    assert d({"a": 1}, {"a": 1.0}) == ["$.a: 1 != 1.0"]
    assert d({"a": 1.0}, {"a": 1}) != []
    assert d({"a": True}, {"a": 1}) != []
    assert d({"a": None}, {"a": 0}) != []
    assert d({"a": "β"}, {"a": "β "}) != []
    assert d({"a": 1}, {"a": 1, "b": 2}) == ["$.b: not in the golden"]
    assert d({"a": 1, "b": 2}, {"a": 1}) == ["$.b: missing"]
    assert d([{"a": 1}], [{"a": 1, "b": 2}]) == ["$[0].b: not in the golden"]
    assert d([1, 2], [2, 1]) == ["$[0]: 1 != 2", "$[1]: 2 != 1"]
    assert d([1, 2], [1, 2, 3]) == ["$: 2 items != 3"]
    assert d(0.0, -0.0) == ["$: 0.0 != -0.0"]
    # Floats: exact unless a tolerance is given, and then relative only.
    assert d(1.0, 1.0 + 1e-15) != []
    assert d(1.0, 1.0 + 1e-15, rel_tol=1e-12) == []
    assert d(0.0, 1e-300, rel_tol=1e-12) != []
    assert d(1e6, 1e6 * (1 + 2e-12), rel_tol=1e-12) != []
    # The tolerance reaches floats nested anywhere.
    assert d({"x": [[1.0]]}, {"x": [[1.0 + 1e-14]]}, rel_tol=1e-12) == []
    assert d({"x": [[1.0]]}, {"x": [[1.0 + 1e-14]]}) == ["$.x[0][0]: 1.0 != 1.00000000000001"]
    # The derived rule: 1.2e-12 relative is too far.
    assert fixtures.derived_differences({"x": 1.0}, {"x": 1.0 + 1.2e-12}) != []
    assert fixtures.derived_differences({"x": 1.0}, {"x": 1.0 + 0.8e-12}) == []


def test_derived_failure_lists_at_most_40_differences():
    with pytest.raises(AssertionError) as failed:
        fixtures.check_derived({"x": [0] * 45}, {"x": [1] * 45}, "f9")
    lines = str(failed.value).splitlines()
    assert lines[0] == "the derived output of f9 differs from f9/derived.json:"
    assert lines[1] == "  $.x[0]: 0 != 1"
    assert lines[40] == "  $.x[39]: 0 != 1"
    assert lines[41] == "  ... and 5 more"
    with pytest.raises(AssertionError) as failed:
        fixtures.check_derived({"x": [0] * 41}, {"x": [1] * 41}, "f9")
    assert str(failed.value).splitlines()[40:42] == ["  $.x[39]: 0 != 1", "  ... and 1 more"]
    with pytest.raises(AssertionError) as failed:
        fixtures.check_derived({"x": [0] * 40}, {"x": [1] * 40}, "f9")
    assert "more" not in str(failed.value)


def test_canonical_json():
    value = {"b": [1, 2.5, None, "µ", True], "a": {"c": [{"d": 0.1}], "e": {}}, "f": []}
    assert fixtures.canonical_json(value).decode("utf-8") == (
        "{\n"
        ' "a": {\n'
        '  "c": [\n'
        "   {\n"
        '    "d": 0.1\n'
        "   }\n"
        "  ],\n"
        '  "e": {}\n'
        " },\n"
        ' "b": [1, 2.5, null, "µ", true],\n'
        ' "f": []\n'
        "}\n"
    )
    with pytest.raises(TypeError, match="JSON keys are text"):
        fixtures.canonical_json({"a": {1: 2}})
    for bad in ({"a": {1, 2}}, [b"x"]):
        with pytest.raises(TypeError):
            fixtures.canonical_json(bad)
    for bad in ({"a": float("nan")}, [float("inf")]):
        with pytest.raises(ValueError):
            fixtures.canonical_json(bad)


@pytest.mark.parametrize("case", CASES)
def test_goldens_are_canonical(case):
    for name in ("stored.json", "derived.json"):
        data = (DATA / case / name).read_bytes()
        assert data == fixtures.canonical_json(json.loads(data)), name


# --- The commands ---


def test_refresh_derived_leaves_frozen_files(tmp_path, monkeypatch, capsys):
    root = tmp_path / "schema2"
    for case in CASES:
        fixtures.copy_project(DATA / case, root / case)
        for name in ("stored.json", "derived.json"):
            (root / case / name).write_bytes((DATA / case / name).read_bytes())
    (root / "f1" / "derived.json").write_text("{}\n", encoding="utf-8")  # stale: rewritten

    def listing() -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "folder"
            )
            for path in sorted(root.rglob("*"))
        }

    before = listing()
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))  # the copies it makes go here
    assert fixtures.main(["--refresh-derived", "--out", str(root)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"refreshed {case}/derived.json" for case in CASES
    ]
    assert list(temp.iterdir()) == []  # and are gone
    after = listing()
    assert set(after) == set(before)
    changed = {name for name in after if after[name] != before[name]}
    assert changed <= {f"{case}/derived.json" for case in CASES}
    assert "f1/derived.json" in changed
    for case in CASES:
        written = (root / case / "derived.json").read_bytes()
        assert written == fixtures.canonical_json(json.loads(written))
        fixtures.check_derived(_golden(case, "derived.json"), json.loads(written), case)


def test_frozen_refusal():
    toplevel = Path("checkout")
    module = toplevel / "src" / "proteia" / "__init__.py"
    ok = {"head": FROZEN, "changes": "", "module": module, "toplevel": toplevel}
    assert fixtures.frozen_refusal(**ok) is None
    other = fixtures.frozen_refusal(**{**ok, "head": "1" * 40})
    assert other is not None and FROZEN in other and "1" * 40 in other
    changed = fixtures.frozen_refusal(**{**ok, "changes": " M src/proteia/core/model.py\n"})
    assert changed is not None and "src/proteia/core/model.py" in changed
    elsewhere = Path("elsewhere") / "proteia" / "__init__.py"
    imported = fixtures.frozen_refusal(**{**ok, "module": elsewhere})
    assert imported is not None and "elsewhere" in imported


def _head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=DATA,
            capture_output=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def test_write_frozen_refuses_at_this_commit(tmp_path, capsys):
    # This file exists only after 05b4af5, so this checkout is never at it;
    # where git cannot name the commit, it cannot tell, and refuses too.
    head = _head()
    out = tmp_path / "out"
    assert fixtures.main(["--write-frozen", "--out", str(out)]) == 2
    assert not out.exists()
    err = capsys.readouterr().err
    if head is None:
        assert err.startswith(f"refused: cannot tell whether this checkout is at {FROZEN}: ")
    else:
        assert err.startswith(f"refused: this checkout is at {head}, not at {FROZEN}: ")
    # And as a script.
    script = DATA / "make_fixtures.py"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(
        [sys.executable, str(script), "--write-frozen", "--out", str(out)],
        capture_output=True,
        encoding="utf-8",
        env=env,
    )
    assert done.returncode == 2 and done.stderr.startswith("refused: ")
    assert not out.exists()


def test_write_frozen_writes_a_case_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "_refusal_here", lambda: None)  # as if at 05b4af5
    (tmp_path / "f2").mkdir()
    with pytest.raises(fixtures.Refused, match="^f2 exist in "):
        fixtures.write_frozen(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f2"]


def test_write_frozen_writes_every_case(tmp_path, monkeypatch, capsys):
    # As at 05b4af5 (a stand-in commit), from the frozen image files: what it
    # writes holds the frozen values, as far as this platform's floats allow.
    monkeypatch.setattr(fixtures, "_refusal_here", lambda: None)
    real_git = fixtures._git
    monkeypatch.setattr(
        fixtures,
        "_git",
        lambda *args: "f" * 40 + "\n" if args == ("rev-parse", "HEAD") else real_git(*args),
    )
    monkeypatch.setattr(fixtures, "sources", lambda case: fixtures.frozen_sources(DATA / case))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    assert fixtures.main(["--write-frozen", "--out", str(out)]) == 0
    assert capsys.readouterr().out.splitlines() == [f"wrote {case}" for case in CASES]
    for case in CASES:
        stored = json.loads((out / case / "stored.json").read_text(encoding="utf-8"))
        frozen = _golden(case, "stored.json")
        assert stored.pop("built_at") == "f" * 40 and frozen.pop("built_at") == FROZEN
        for values in (stored, frozen):
            del values["content_hash"]  # changes whole with the last bit of any float
        assert fixtures.differences(frozen, stored, rel_tol=1e-12) == []
        written = json.loads((out / case / "derived.json").read_text(encoding="utf-8"))
        fixtures.check_derived(_golden(case, "derived.json"), written, case)
        assert (out / case / "project.json").is_file()
        assert sorted(p.name for p in (out / case / "images").iterdir()) == sorted(
            p.name for p in (DATA / case / "images").iterdir()
        )


def test_fixtures_stay_under_400_kb():
    # Every file of the three cases: the projects, their images and goldens.
    files = [p for case in CASES for p in (DATA / case).rglob("*") if p.is_file()]
    assert {p.name for p in files} >= {"project.json", "stored.json", "derived.json"}
    assert {p.suffix for p in files} == {".json", ".tif", ".png", ".jpg"}
    assert sum(p.stat().st_size for p in files) < 400_000


@pytest.mark.parametrize("case", CASES)
def test_generator_draws_the_frozen_images(case):
    # The lossless files hold the pixels drawn; the JPEG holds them up to its
    # compression.
    drawn = fixtures.drawn(case)
    frozen = fixtures.frozen_sources(DATA / case)
    assert set(drawn) == set(frozen)
    for name, pixels in drawn.items():
        stored = fixtures.decoded(frozen[name])
        assert (stored.shape, stored.dtype) == (pixels.shape, pixels.dtype), name
        if name.endswith(".jpg"):
            error = np.abs(stored.astype(int) - pixels.astype(int))
            assert error.max() <= JPEG_MAX_ERROR and error.mean() < JPEG_MEAN_ERROR, name
        else:
            assert np.array_equal(stored, pixels), name
    with pytest.raises(ValueError, match="unknown case 'f9'"):
        fixtures.drawn("f9")


def test_generator_writes_the_formats_named():
    magic = {".tif": "49492a00", ".png": "89504e470d0a1a0a", ".jpg": "ffd8ff"}
    for case in CASES:
        drawn = fixtures.drawn(case)
        for name, data in fixtures.sources(case).items():
            suffix = name[-4:]
            assert data.startswith(bytes.fromhex(magic[suffix])), name
            if suffix != ".jpg":
                assert np.array_equal(fixtures.decoded(data), drawn[name]), name


def test_build_imports_the_files_given(tmp_path):
    files = fixtures.frozen_sources(DATA / "f3")
    marker = fixtures.decoded(files[fixtures.N])
    files[fixtures.N] = fixtures._encoded(marker, "PNG", compress_level=1)  # same pixels
    assert files[fixtures.N] != fixtures.frozen_sources(DATA / "f3")[fixtures.N]
    project = fixtures.build(tmp_path / "f3", "f3", files=files)
    stored = {image.original_name: image.sha256 for image in project.batch.iter_images()}
    assert stored == {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def test_derive_writes_a_case(tmp_path, monkeypatch, capsysbinary, derived):
    monkeypatch.syspath_prepend(str(DATA))  # as when it runs as a script
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    derive = _load("derive")
    out = tmp_path / "f3.json"
    assert derive.main(["f3", "--out", str(out)]) == 0
    assert out.read_bytes() == fixtures.canonical_json(derived["f3"])
    assert [p.name for p in tmp_path.iterdir()] == ["f3.json"]  # its copies are gone
    assert capsysbinary.readouterr().out == b""
    assert derive.main([str(DATA / "f3")]) == 0  # a case folder, to stdout
    assert capsysbinary.readouterr().out == out.read_bytes()


def test_derive_script_prints_canonical_json(tmp_path, derived):
    env = {**os.environ, "TMP": str(tmp_path), "TEMP": str(tmp_path), "TMPDIR": str(tmp_path)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = DATA / "derive.py"
    printed = subprocess.run(
        [sys.executable, str(script), str(DATA / "f2")], check=True, env=env, capture_output=True
    ).stdout
    assert printed == fixtures.canonical_json(derived["f2"])


# --- What each project holds (the issue's list) ---


def _project(case: str) -> dict:
    return json.loads((DATA / case / "project.json").read_text(encoding="utf-8"))


def _notice_codes(case: str) -> set[str]:
    results = _golden(case, "derived.json")["results"]
    return {n["code"] for n in results["notices"]}


def test_f1_holds_two_membranes_a_marker_two_groups_two_ladders_and_strip_edges():
    batch = _project("f1")["batch"]
    a, b = batch["membranes"]
    kinds = [
        (i["original_name"][:2], i["kind"], i["bit_depth"], i["polarity"]) for i in a["images"]
    ]
    assert kinds == [
        ("E1", "chemiluminescence", 16, "dark_on_light"),
        ("E2", "chemiluminescence", 16, "dark_on_light"),
        ("M ", "visible_marker", 8, "dark_on_light"),
    ]
    e1, e2, m = a["images"]
    assert e1["marker_image_id"] == m["id"] and e2["marker_image_id"] is None
    assert [i["file"][-4:] for i in a["images"]] == [".tif", ".tif", ".png"]
    points = a["calibration"]["points"]
    by = {(p["image_id"], p["source"], p.get("side", "left")) for p in points}
    assert by == {
        (m["id"], "visible_marker", "left"),
        (m["id"], "visible_marker", "right"),
        (e2["id"], "chemiluminescence_marker", "left"),
    }
    assert sum(p["image_id"] == m["id"] for p in points) == 6
    assert a["calibration"]["ladder"] == "pageruler_plus/tris_glycine"
    (e3,) = b["images"]
    assert e3["polarity"] == "light_on_dark" and e3["bit_depth"] == 16
    assert (e3["width"], e3["height"]) != (e1["width"], e1["height"])
    assert [p["source"] for p in b["calibration"]["points"]] == ["strip_edge", "strip_edge"]
    calibration = _golden("f1", "derived.json")["calibration"]
    assert [len(m["groups"]) for m in calibration] == [2, 1]
    assert calibration[0]["groups"][0]["image_ids"] == [e1["id"], m["id"]]
    two_ladders = calibration[0]["groups"][0]["fit"]
    assert [ladder["side"] for ladder in two_ladders["ladders"]] == ["left", "right"]
    assert two_ladders["tilt_deg"] > 0
    assert calibration[0]["groups"][1]["fit"]["ladders"][0]["two_point"] is True


def test_f1_holds_its_lanes_and_four_ways_of_placing_boxes():
    batch = _project("f1")["batch"]
    lanes = [(x["label"], x["sample"], x["included"]) for x in batch["lanes"]]
    assert lanes == [
        ("vehicle", "V1", True),
        ("vehicle", "V1", True),
        ("vehicle", "V2", True),
        ("10 µM", "A1", True),
        ("10 µM", "A2", True),
        ("10 µM", "A3", False),
        ("50 µM", "B1", True),
        ("50 µM", "B2", True),
    ]
    assert batch["reference_condition"] == "vehicle"
    proteins = {p["name"]: p for p in batch["proteins"]}
    assert list(proteins) == ["β-catenin", "α-tubulin", "GAPDH", "p53 µ"]
    images = {i["original_name"][:2]: i["id"] for m in batch["membranes"] for i in m["images"]}
    on = {name: p["image_id"] for name, p in proteins.items()}
    assert on == {
        "β-catenin": images["E1"],
        "α-tubulin": images["E3"],
        "GAPDH": images["E1"],
        "p53 µ": images["E2"],
    }
    sources = {name: {b["source"] for b in p["bands"]} for name, p in proteins.items()}
    assert sources == {
        "β-catenin": {"row_box"},
        "α-tubulin": {"click"},
        "GAPDH": {"manual"},
        "p53 µ": {"mw_guided"},
    }
    assert all("bands_found" in b for b in proteins["β-catenin"]["bands"])
    assert [u["source"] for u in proteins["β-catenin"]["undetected"]] == ["row_box"]
    assert [u["source"] for u in proteins["p53 µ"]["undetected"]] == ["mw_guided"]
    assert proteins["α-tubulin"]["box_padding"] == {"across": 2, "along": 1}
    assert [b["clipped"] for b in proteins["GAPDH"]["bands"]].count(True) == 1
    ids = {name: p["id"] for name, p in proteins.items()}
    assert proteins["β-catenin"]["loading_control_ids"] == [ids["α-tubulin"]]
    assert proteins["p53 µ"]["loading_control_ids"] == [ids["GAPDH"]]
    assert proteins["β-catenin"]["expected_mw"] == 92.0
    assert proteins["p53 µ"]["expected_mw"] == 53.0
    results = _golden("f1", "derived.json")["results"]
    assert results["all_lanes"] is not None  # the excluded lane holds values
    assert len(_golden("f1", "derived.json")["lane_tables"]) == 2
    assert {"technical_repeats", "clipped", "below_detection", "calibration_two_points"} <= (
        _notice_codes("f1")
    )


def test_f2_holds_a_lossy_and_a_colour_image_with_overlapping_boxes():
    batch = _project("f2")["batch"]
    images = [i for m in batch["membranes"] for i in m["images"]]
    assert [i["file"][-4:] for i in images] == [".jpg", ".png"]
    warnings = [[w["code"] for w in i["import_warnings"]] for i in images]
    assert "lossy_format" in warnings[0] and "color_channels_differ" in warnings[1]
    bands = {p["name"]: p["bands"] for p in batch["proteins"]}
    assert any(b.get("possibly_clipped") for b in bands["β-catenin"])
    assert {"boxes_overlap", "possibly_clipped", "clipping_not_checked"} <= _notice_codes("f2")


def test_f3_holds_an_undo_and_redo_history_with_a_removed_image():
    project = _project("f3")
    log = project["log"]
    assert [entry["action"] for entry in log[-4:]] == ["remove_image", "undo", "redo", "undo"]
    removal, undo, redo, last = log[-4:]
    before = log[-5]
    assert undo["params"]["returns_to_seq"] == before["seq"]
    assert redo["params"]["returns_to_seq"] == removal["seq"]
    assert last["params"]["returns_to_seq"] == before["seq"]
    assert last["content_hash"] == before["content_hash"] != removal["content_hash"]
    assert _golden("f3", "stored.json")["content_hash"] == before["content_hash"]
    assert _golden("f3", "derived.json")["history_issues"] == []
    # The removal took the marker link, the ladder and the apparent MWs along;
    # the last undo brought them back.
    batch = project["batch"]
    signal, marker = batch["membranes"][0]["images"]
    assert signal["marker_image_id"] == marker["id"]
    assert removal["params"]["image_id"] == marker["id"]
    assert len(batch["membranes"][0]["calibration"]["points"]) == 3
    beta = batch["proteins"][0]
    assert beta["name"] == "β-catenin" and all(b["apparent_mw"] for b in beta["bands"])
