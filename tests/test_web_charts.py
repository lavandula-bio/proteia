# SPDX-License-Identifier: Apache-2.0
"""The chart store (#52): a chart's URL is named by its spec, it is drawn when
first fetched (once for the fetches that come while it is drawn) and then kept,
a key it does not keep is unknown, and a create or open empties it."""

from __future__ import annotations

import hashlib
import threading

import pytest

from proteia.core.analyze import StatisticsSetting, compare, describe
from proteia.core.model import UnknownIdError
from proteia.core.plotspec import PlotSpec, ValueKind, build_plotspec
from proteia.viz import render_svg
from proteia.web import charts

GROUPS = {"vehicle": [1.0, 1.1, 0.9], "10 µM": [2.0, 2.1, 1.9]}  # 10 micro-molar


def spec(title: str = "β-catenin / α-tubulin") -> PlotSpec:
    test = compare(GROUPS, StatisticsSetting(), ratio=True, reference="vehicle")
    return build_plotspec(
        GROUPS, describe(GROUPS), test, value_kind=ValueKind.FOLD_CHANGE, title=title
    )


@pytest.fixture
def drawn(monkeypatch) -> list[str]:
    """The title of every chart the store draws, in order."""
    titles: list[str] = []

    def counting(chart: PlotSpec) -> bytes:
        titles.append(chart.title)
        return render_svg(chart)

    monkeypatch.setattr(charts, "render_svg", counting)
    return titles


def key_of(url: str) -> str:
    prefix, suffix = "/api/charts/", ".svg"
    assert url.startswith(prefix) and url.endswith(suffix)
    return url[len(prefix) : -len(suffix)]


def test_the_key_is_a_hash_of_the_spec():
    chart = spec()
    # v5: the statistics as legend text, a key of the marks, the n.d. row, the
    # axis titles; and the style the chart is drawn in.
    drawing = b"render-v5\nbar\n" + chart.model_dump_json().encode()
    expected = hashlib.sha256(drawing).hexdigest()
    assert charts.chart_key(chart) == expected[:32]
    assert charts.chart_url(chart) == f"/api/charts/{expected[:32]}.svg"
    assert charts.chart_key(spec()) == charts.chart_key(chart)  # an equal spec, the same key
    assert charts.chart_key(chart.model_copy(update={"subtitle": "All lanes"})) != (
        charts.chart_key(chart)
    )


def test_the_key_differs_by_style():
    # A drawing in another style is another drawing: it never takes this one's key.
    chart = spec()
    assert charts.chart_key(chart, style="bar") == charts.chart_key(chart)
    assert charts.chart_key(chart, style="log-points") != charts.chart_key(chart)


def test_a_chart_is_drawn_when_first_fetched_and_then_kept(drawn):
    store = charts.ChartStore()
    store.reset(1)
    url = store.register(spec(), open_id=1)
    assert url == charts.chart_url(spec()) and drawn == []  # registering draws nothing
    svg = store.svg(key_of(url))
    assert svg == render_svg(spec()) and len(drawn) == 1
    assert store.svg(key_of(url)) == svg and len(drawn) == 1
    store.register(spec(), open_id=1)  # an unchanged chart, in a later answer
    assert store.svg(key_of(url)) == svg and len(drawn) == 1


@pytest.mark.parametrize(
    "key",
    [
        "0" * 32,  # well formed, never given
        "A" * 32,  # upper case
        "0" * 31,
        "0" * 33,
        "g" * 32,
        "0" * 32 + "\n",
        "",
    ],
)
def test_a_key_the_store_never_gave_is_unknown(key):
    store = charts.ChartStore()
    store.reset(1)
    store.register(spec(), open_id=1)
    with pytest.raises(UnknownIdError):
        store.svg(key)


@pytest.mark.parametrize("key", ["A" * 32, "0" * 31, "../" + "0" * 29, "0" * 10_000])
def test_a_malformed_key_is_refused_before_the_store_is_read(key):
    class Untouchable:
        def __enter__(self):
            raise AssertionError("the store was read")

        def __exit__(self, *exc):
            return False

    store = charts.ChartStore()
    store._lock = Untouchable()  # type: ignore[assignment]
    with pytest.raises(UnknownIdError) as refused:
        store.svg(key)
    assert key not in str(refused.value)  # the answer does not echo what was sent


def test_a_reset_forgets_every_chart_and_later_registrations_of_the_old_open_id(drawn):
    store = charts.ChartStore()
    store.reset(1)
    url = store.register(spec(), open_id=1)
    store.svg(key_of(url))
    store.reset(2)  # another project opened
    with pytest.raises(UnknownIdError):
        store.svg(key_of(url))
    # A late answer about the earlier opening registers nothing.
    assert store.register(spec(), open_id=1) == url
    with pytest.raises(UnknownIdError):
        store.svg(key_of(url))
    store.register(spec(), open_id=2)  # the same chart in the new project
    assert store.svg(key_of(url)) == render_svg(spec())
    assert len(drawn) == 2  # drawn again: nothing drawn before the reset is kept


def test_the_specs_registered_last_are_kept(drawn):
    store = charts.ChartStore(specs_kept=2, drawings_kept=2)
    store.reset(1)
    first, second = (store.register(spec(title), open_id=1) for title in ("A", "B"))
    store.svg(key_of(first))  # a fetch keeps it too
    third = store.register(spec("C"), open_id=1)
    with pytest.raises(UnknownIdError):
        store.svg(key_of(second))  # the least recently used
    store.svg(key_of(first))
    store.svg(key_of(third))
    assert drawn == ["A", "C"]  # A was kept drawn

    store.register(spec("D"), open_id=1)  # evicts A, and its drawing with it
    with pytest.raises(UnknownIdError):
        store.svg(key_of(first))
    store.register(spec("A"), open_id=1)  # a read of the project registers it again
    store.svg(key_of(first))
    assert drawn == ["A", "C", "A"]


def test_a_chart_registered_again_is_kept_as_one_registered_last(drawn):
    store = charts.ChartStore(specs_kept=2)
    store.reset(1)
    first, second = (store.register(spec(title), open_id=1) for title in ("A", "B"))
    store.register(spec("A"), open_id=1)  # unchanged, in a later answer, and not fetched
    store.register(spec("C"), open_id=1)  # evicts B, the least recently registered
    assert store.svg(key_of(first)) == render_svg(spec("A"))  # A's URL, in that later answer
    with pytest.raises(UnknownIdError):
        store.svg(key_of(second))


def test_the_drawings_fetched_last_are_kept(drawn):
    store = charts.ChartStore(specs_kept=8, drawings_kept=2)
    store.reset(1)
    a, b, c = (key_of(store.register(spec(title), open_id=1)) for title in "ABC")
    for key in (a, b, a, c):  # C evicts B, the drawing least recently fetched
        store.svg(key)
    assert drawn == ["A", "B", "C"]
    store.svg(a)
    store.svg(b)  # still registered: drawn again
    assert drawn == ["A", "B", "C", "B"]


def test_a_drawing_that_finishes_after_a_reset_is_not_kept(monkeypatch):
    store = charts.ChartStore()
    store.reset(1)
    url = store.register(spec(), open_id=1)

    def reset_while_drawing(chart: PlotSpec) -> bytes:
        store.reset(2)  # another project opened while the chart was drawn
        return render_svg(chart)

    monkeypatch.setattr(charts, "render_svg", reset_while_drawing)
    assert store.svg(key_of(url)) == render_svg(spec())  # the request still gets it
    store.register(spec(), open_id=2)
    drawn: list[PlotSpec] = []
    monkeypatch.setattr(charts, "render_svg", lambda chart: drawn.append(chart) or b"<svg/>")
    store.svg(key_of(url))
    assert drawn == [spec()]  # the drawing from before the reset was not kept


def fetch_at_once(store: charts.ChartStore, key: str, n: int) -> list[bytes | BaseException]:
    """What each of ``n`` fetches of ``key``, made at once, got."""
    got: list[bytes | BaseException] = [b""] * n
    start = threading.Barrier(n)

    def fetch(i: int) -> None:
        start.wait(10)
        try:
            got[i] = store.svg(key)
        except Exception as exc:
            got[i] = exc

    threads = [threading.Thread(target=fetch, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    return got


def test_fetches_of_a_chart_being_drawn_wait_for_that_drawing(monkeypatch):
    store = charts.ChartStore()
    store.reset(1)
    key = key_of(store.register(spec(), open_id=1))
    begun: list[PlotSpec] = []
    another = threading.Event()

    def slow(chart: PlotSpec) -> bytes:
        begun.append(chart)
        if len(begun) == 1:
            another.wait(0.5)  # the other fetches come meanwhile; a second drawing ends it
        else:
            another.set()
        return render_svg(chart)

    monkeypatch.setattr(charts, "render_svg", slow)
    assert fetch_at_once(store, key, 6) == [render_svg(spec())] * 6
    assert begun == [spec()]  # drawn once for them all
    store.svg(key)
    assert begun == [spec()]  # and kept


def test_a_drawing_begun_before_a_reset_is_not_waited_for(monkeypatch):
    store = charts.ChartStore()
    store.reset(1)
    key = key_of(store.register(spec(), open_id=1))
    begun: list[int] = []
    drawing = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]

    def held(chart: PlotSpec) -> bytes:
        n = len(begun)
        begun.append(n)
        if n < len(drawing):  # the first two drawings wait to be released
            drawing[n].set()
            release[n].wait(10)
        return render_svg(chart)

    monkeypatch.setattr(charts, "render_svg", held)
    fetches = [threading.Thread(target=store.svg, args=(key,)) for _ in range(3)]
    try:
        fetches[0].start()
        assert drawing[0].wait(10)
        store.reset(2)  # the project opened again while its chart was drawn
        store.register(spec(), open_id=2)
        fetches[1].start()
        assert drawing[1].wait(2)  # drawn anew, not waiting for the drawing begun before
        release[0].set()
        fetches[0].join(10)  # that drawing ends while the new one is under way
        fetches[2].start()
        fetches[2].join(0.2)  # time enough to begin a drawing, were it not waiting
        assert begun == [0, 1]  # it waits for the new drawing
    finally:
        for event in release:
            event.set()
        for fetch in fetches:
            if fetch.ident is not None:
                fetch.join(10)
    assert begun == [0, 1]
    store.svg(key)
    assert begun == [0, 1]  # the new drawing was kept


def test_a_drawing_that_fails_is_not_shared_or_kept(monkeypatch):
    store = charts.ChartStore()
    store.reset(1)
    key = key_of(store.register(spec(), open_id=1))
    begun: list[PlotSpec] = []
    another = threading.Event()

    def failing_first(chart: PlotSpec) -> bytes:
        begun.append(chart)
        if len(begun) == 1:
            another.wait(0.5)  # the other fetch comes meanwhile
            raise ValueError("the first drawing fails")
        another.set()
        return render_svg(chart)

    monkeypatch.setattr(charts, "render_svg", failing_first)
    got = fetch_at_once(store, key, 2)
    # One fetch drew and failed; the other then drew the chart for itself.
    assert {type(g) for g in got} == {ValueError, bytes}
    assert render_svg(spec()) in got and len(begun) == 2
    store.svg(key)
    assert len(begun) == 2  # the drawing that did not fail was kept
