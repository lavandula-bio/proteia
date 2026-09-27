# SPDX-License-Identifier: Apache-2.0
"""The SVG a chart is served as (#52): the same bytes for the same spec, an id on
every drawn bar, point and bracket, and nothing in it a browser would run or
fetch."""

from __future__ import annotations

import re
import threading
import xml.etree.ElementTree as ET

import matplotlib
import pytest

from proteia.core.analyze import compare, describe
from proteia.core.plotspec import PlotSpec, ValueKind, build_plotspec
from proteia.viz import render, render_pdf, render_png, render_svg

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"
# The only URLs an SVG may hold: its namespace names and the SVG 1.1 DTD's
# identifier in matplotlib's prolog. None of them is ever fetched.
_NAMES_NOT_FETCHED = {_SVG_NS, _XLINK_NS, "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"}
# Elements that run code, load a resource, link away or change attributes over time.
_ACTIVE = {"script", "foreignObject", "iframe", "object", "embed", "image", "a", "animate", "set"}
_ACTIVE |= {"animateTransform", "animateMotion", "handler", "listener", "feImage"}

VEHICLE, LOW, HIGH = "vehicle", "10 µM", "50 µM"  # 10 and 50 micro-molar


def _spec(groups: dict[str, list[float]], **update: object) -> PlotSpec:
    lanes, start = {}, 0
    for condition, values in groups.items():
        lanes[condition] = list(range(start, start + len(values)))
        start += len(values)
    spec = build_plotspec(
        groups,
        describe(groups),
        compare(groups),
        value_kind=ValueKind.FOLD_CHANGE,
        title="β-catenin / α-tubulin",  # beta-catenin over alpha-tubulin
        lane_indices=lanes,
        first_label=VEHICLE,
    )
    return spec.model_copy(update=update)


TESTED = {VEHICLE: [1.0, 1.1, 0.9], LOW: [2.0, 2.1, 1.9], HIGH: [0.5, 0.6]}


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _ids(svg: bytes, pattern: str) -> set[str]:
    root = ET.fromstring(svg)
    return {el.get("id") for el in root.iter() if re.fullmatch(pattern, el.get("id") or "")}


def test_a_spec_is_drawn_as_the_same_bytes_every_time():
    spec, other = _spec(TESTED), _spec(TESTED, subtitle="All lanes")
    first = render_svg(spec)
    drawn_between = render_svg(other)
    assert render_svg(spec) == first  # no counter, date or random id carries over
    assert drawn_between != first
    assert first.startswith(b"<?xml")


def test_charts_drawn_at_once_are_the_bytes_drawn_one_at_a_time():
    specs = [_spec(TESTED), _spec(TESTED, subtitle="All lanes")] * 3
    alone = [render_svg(spec) for spec in specs]
    drawn: list[bytes | None] = [None] * len(specs)
    start = threading.Barrier(len(specs))

    def draw(i: int) -> None:
        start.wait(10)
        drawn[i] = render_svg(specs[i])

    threads = [threading.Thread(target=draw, args=(i,)) for i in range(len(specs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert drawn == alone


def _png(spec: PlotSpec) -> bytes:
    return render_png(spec, dpi=72)


@pytest.mark.parametrize(
    ("draw_first", "draw_second"),
    [
        (render_svg, render_svg),
        (render_svg, render_pdf),
        (render_pdf, render_svg),
        (render_pdf, render_pdf),
        (render_svg, _png),
        (_png, render_svg),
    ],
    ids=["svg-svg", "svg-pdf", "pdf-svg", "pdf-pdf", "svg-png", "png-svg"],
)
def test_a_drawing_waits_until_the_one_before_it_is_saved(monkeypatch, draw_first, draw_second):
    # The lock spans a whole drawing, from the figure to the saved bytes, in
    # every format, as the settings a drawing uses are matplotlib's global ones:
    # an SVG and a PDF (an export's, drawn while the screen draws its SVGs)
    # would each undo the other's.
    figure = render.render_figure
    begun: list[str | None] = []
    saving, release = threading.Event(), threading.Event()

    def held_while_saved(spec: PlotSpec):
        begun.append(spec.subtitle)
        drawn = figure(spec)
        if len(begun) == 1:
            save = drawn.savefig

            def held(*args, **kwargs):
                saving.set()
                release.wait(10)
                return save(*args, **kwargs)

            drawn.savefig = held
        return drawn

    monkeypatch.setattr(render, "render_figure", held_while_saved)
    first = threading.Thread(target=draw_first, args=(_spec(TESTED, subtitle="first"),))
    second = threading.Thread(target=draw_second, args=(_spec(TESTED, subtitle="second"),))
    first.start()
    try:
        assert saving.wait(10)
        second.start()
        second.join(0.2)  # time enough for it to begin, were it not held back
        assert begun == ["first"]
    finally:
        release.set()
        first.join(10)
        if second.ident is not None:
            second.join(10)
    assert begun == ["first", "second"]


def test_drawing_leaves_the_global_settings_alone():
    keys = ("svg.hashsalt", "svg.fonttype", "pdf.fonttype")
    before = {key: matplotlib.rcParams[key] for key in keys}
    render_svg(_spec(TESTED))
    render_pdf(_spec(TESTED))
    assert {key: matplotlib.rcParams[key] for key in before} == before


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # scipy, on values that do not vary
@pytest.mark.parametrize(
    "groups",
    [
        TESTED,  # a test ran, with brackets
        {VEHICLE: [1.0], LOW: [2.0, 2.1]},  # a group of one: no test, no brackets
        {VEHICLE: [1.0, 1.0], LOW: [1.0, 1.0]},  # the values do not vary
        {VEHICLE: [1.0, 1.1], LOW: [2.0, 2.1], HIGH: []},  # a condition with no value
    ],
)
def test_every_bar_point_and_bracket_has_its_own_id(groups):
    spec = _spec(groups)
    svg = render_svg(spec)
    assert _ids(svg, r"bar-\d+") == {f"bar-{i}" for i in range(len(spec.bars))}
    assert _ids(svg, r"point-\d+-\d+") == {
        f"point-{i}-{j}" for i, bar in enumerate(spec.bars) for j in range(len(bar.points))
    }
    assert _ids(svg, r"bracket-\d+") == {f"bracket-{k}" for k in range(len(spec.comparisons))}
    if groups is TESTED:
        assert spec.comparisons  # the brackets above were drawn


def test_a_bar_id_names_the_bar_of_that_index():
    spec = _spec(TESTED)
    root = ET.fromstring(render_svg(spec))
    tops = []
    for i in range(len(spec.bars)):
        (group,) = [el for el in root.iter() if el.get("id") == f"bar-{i}"]
        (path,) = group.iter(f"{{{_SVG_NS}}}path")
        ys = [float(y) for y in re.findall(r"[\d.]+ ([\d.]+)", path.get("d"))]
        tops.append(min(ys))  # SVG y grows downwards: the top of the bar
    by_height = sorted(range(len(spec.bars)), key=lambda i: -spec.bars[i].mean)
    assert sorted(range(len(tops)), key=lambda i: tops[i]) == by_height


def test_a_point_id_names_the_point_of_that_index():
    # Every point a value of its own, in no order, so each id has one right place.
    spec = _spec({VEHICLE: [1.0, 1.5, 0.5], LOW: [2.2, 1.8, 2.6], HIGH: [0.3, 0.9]})
    root = ET.fromstring(render_svg(spec))
    values, ys = {}, {}
    for i, bar in enumerate(spec.bars):
        for j, value in enumerate(bar.points):
            (group,) = [el for el in root.iter() if el.get("id") == f"point-{i}-{j}"]
            (mark,) = group.iter(f"{{{_SVG_NS}}}use")
            values[(i, j)], ys[(i, j)] = value, float(mark.get("y"))
    assert len(set(values.values())) == len(values)
    # SVG y grows downwards: from the top, the points of the largest values first.
    assert sorted(ys, key=ys.__getitem__) == sorted(values, key=lambda ij: -values[ij])


def _named(name: str) -> PlotSpec:
    """A chart with ``name`` for its title, its subtitle, a condition and its y label."""
    groups = {VEHICLE: [1.0, 1.1, 0.9], name: [2.0, 2.1, 1.9]}
    return _spec(groups, title=name, subtitle=name, y_label=name)


# Names as typed (or as a project file holds them), each with the text a chart
# shows for it: matplotlib's math syntax as typed, and a character that XML
# cannot hold as the replacement character, U+FFFD.
_WRAPPED = " ".join([r"$\beta$-catenin"] * 8)  # a title broken over lines
NAMES = {
    "math": (r"$\alpha$-tubulin", r"$\alpha$-tubulin"),
    "not-math": (r"$\notacommand$", r"$\notacommand$"),  # math matplotlib cannot draw
    "math-wrapped": (_WRAPPED, _WRAPPED),
    "uffff": ("GAPDH\uffff", "GAPDH\ufffd"),  # noncharacters, which a typed name may hold
    "ufffe": ("GAPDH\ufffe", "GAPDH\ufffd"),
    "control": ("bell\x07", "bell\ufffd"),  # a control character and an unpaired
    "surrogate": ("half\ud800", "half\ufffd"),  # surrogate, which no typed name holds
    "markup": ("<script>--]]>&amp;", "<script>--]]>&amp;"),
}
# A name that long leaves its tick label and y label too wide to lay out.
_TOO_WIDE = pytest.mark.filterwarnings("ignore:Tight layout not applied:UserWarning")


@_TOO_WIDE
@pytest.mark.parametrize(("name", "shown"), list(NAMES.values()), ids=list(NAMES))
def test_a_name_is_drawn_as_typed_in_a_well_formed_svg(name, shown):
    root = ET.fromstring(render_svg(_named(name)))  # well-formed XML
    glyphs = {el.get("id") for el in root.iter() if (el.get("id") or "").startswith("DejaVu")}
    assert {f"DejaVuSans-{ord(c):x}" for c in shown} <= glyphs  # each character drawn
    assert not any("Oblique" in glyph for glyph in glyphs)  # none as math's italics


@_TOO_WIDE
@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(_spec(TESTED), id="plain"),
        pytest.param(_spec(TESTED, subtitle="Excluding lanes 3, 7"), id="subtitled"),
        *(pytest.param(_named(name), id=label) for label, (name, _) in NAMES.items()),
    ],
)
def test_the_svg_runs_nothing_and_fetches_nothing(spec):
    svg = render_svg(spec)
    root = ET.fromstring(svg)  # well-formed XML; ElementTree fetches no DTD
    assert root.tag == f"{{{_SVG_NS}}}svg"
    styles = []
    for element in root.iter():
        assert _local(element.tag) not in _ACTIVE
        for name, value in element.attrib.items():
            assert not _local(name).lower().startswith("on"), name  # no event handler
            if _local(name) == "href":
                assert value.startswith("#"), value  # only a part of this SVG
            targets = re.findall(r"url\(\s*['\"]?(.)", value)
            assert targets == ["#"] * value.count("url("), value  # only parts of this SVG
        if _local(element.tag) == "style":
            styles.append(element.text or "")
    for style in styles:
        assert "@import" not in style and "url(" not in style
    assert set(re.findall(rb"https?://[^\s\"'<>)]+", svg)) <= {
        name.encode() for name in _NAMES_NOT_FETCHED
    }
    assert b"<script" not in svg.lower()


def test_the_svg_has_no_date_and_draws_its_text_as_paths(monkeypatch):
    # Whatever the global settings say (a user's matplotlibrc, say).
    monkeypatch.setitem(matplotlib.rcParams, "svg.fonttype", "none")
    svg = render_svg(_spec(TESTED, subtitle="All lanes"))
    root = ET.fromstring(svg)
    tags = {_local(element.tag) for element in root.iter()}
    assert "metadata" not in tags and b"dc:date" not in svg  # nothing that differs per run
    # Glyphs are paths: the chart looks the same without Proteia's fonts, and the
    # title's µ, α and β need none of the viewer's.
    assert "text" not in tags and b"font-family" not in svg
