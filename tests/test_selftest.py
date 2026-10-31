# SPDX-License-Identifier: Apache-2.0
"""The installation check (#54): ``proteia --self-test`` passes here, unfrozen,
writes its report, and names a step that fails without stopping the others.
The installer build runs the same check inside the frozen bundle."""

import json

import pytest

import proteia
from proteia import selftest
from proteia.web import launch, logs


def _no_state_folder_and_no_session_log(patch: pytest.MonkeyPatch) -> None:
    """Make any use of the per-user state folder, or a session log set up (in
    it or anywhere), fail: the self-test writes only in its temporary folder."""

    def refuse(*args, **kwargs):
        raise AssertionError("the self-test used the state folder or set up a session log")

    patch.setattr(launch, "state_dir", refuse)
    patch.setattr(logs, "setup", refuse)


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    """One run of every step, shared by the tests that only read it."""
    with pytest.MonkeyPatch.context() as patch:
        _no_state_folder_and_no_session_log(patch)
        return selftest.run(tmp_path_factory.mktemp("self test"))


def test_every_step_passes_here(report):
    failed = {step["name"]: step.get("error") for step in report["steps"] if not step["ok"]}
    assert failed == {}
    assert report["ok"]
    assert [step["name"] for step in report["steps"]] == [name for name, _ in selftest.STEPS]
    assert report["proteia"] == proteia.__version__
    assert report["frozen"] is False


def test_the_report_names_every_library_version_the_record_reads(report):
    from proteia.core import record

    assert report["versions"] == record.software_versions()
    assert None not in report["versions"].values()


def test_the_numbers_cover_each_image_kind_the_sample_and_every_test(report):
    from proteia.core import analyze

    numbers = report["numbers"]
    images = {key for key in numbers if key.startswith("image ")}
    assert len(images) == 8
    assert "cmyk_converted" in numbers["image tiff cmyk"]["warnings"]
    sample = numbers["sample"]
    assert len(sample["fold_change"]) == 8 and sample["test_p"] < 0.05
    tests = {key.split()[1] for key in numbers if key.startswith("test ")}
    assert tests == set(analyze.TESTS)
    # The Mann-Whitney U test runs both ways: exact, and by permutation on ties.
    assert {key for key in numbers if key.startswith("test mann_whitney")} == {
        "test mann_whitney (exact)",
        "test mann_whitney (exact permutation)",
    }
    json.dumps(numbers, allow_nan=False)  # the build compares them as JSON


def test_the_web_step_loads_every_static_file_the_page_needs(report):
    from proteia.web.server import STATIC_DIR

    served = set(report["info"]["static_files"])
    assert "/static/app.js" in served and "/static/app.css" in served
    assert served <= {f"/static/{path.name}" for path in STATIC_DIR.iterdir()}


def test_the_web_step_names_another_instance_that_answered_in_its_folder(tmp_path, monkeypatch):
    _no_state_folder_and_no_session_log(monkeypatch)
    # A launch that found an instance answering opened it: it started none.
    monkeypatch.setattr(launch, "start", lambda **kwargs: launch.Opened())
    with pytest.raises(selftest.CheckError, match="another instance holds the lock"):
        selftest._web_app(tmp_path, {})


def test_a_failing_step_is_named_and_the_others_still_run(tmp_path, monkeypatch):
    def broken(folder, report):
        raise selftest.CheckError("left out of the bundle")

    steps = [("versions", broken), ("statistics", selftest._statistics)]
    monkeypatch.setattr(selftest, "STEPS", tuple(steps))
    result = selftest.run(tmp_path)
    assert not result["ok"]
    assert [(step["name"], step["ok"]) for step in result["steps"]] == [
        ("versions", False),
        ("statistics", True),
    ]
    assert "left out of the bundle" in result["steps"][0]["error"]


def test_main_writes_the_report_and_exits_0_or_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(selftest, "STEPS", (("statistics", selftest._statistics),))
    monkeypatch.setattr(selftest.tempfile, "tempdir", str(tmp_path))
    out = tmp_path / "report µ.json"
    assert selftest.main(["--json", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["ok"] and written["steps"][0]["name"] == "statistics"
    assert "passed" in capsys.readouterr().out
    assert [path.name for path in tmp_path.iterdir()] == [out.name]  # its folder is removed

    def broken(folder, report):
        raise selftest.CheckError("no")

    monkeypatch.setattr(selftest, "STEPS", (("versions", broken),))
    assert selftest.main([]) == 1
    captured = capsys.readouterr()
    assert "FAIL versions" in captured.out and "CheckError: no" in captured.err


def test_the_console_command_runs_it_with_self_test(monkeypatch):
    calls = []
    monkeypatch.setattr(selftest, "main", lambda argv: calls.append(argv) or 7)
    monkeypatch.setattr(launch, "start", lambda: pytest.fail("the app must not start"))
    _no_state_folder_and_no_session_log(monkeypatch)
    assert launch.main(["--self-test", "--json", "r.json"]) == 7
    assert calls == [["--json", "r.json"]]
