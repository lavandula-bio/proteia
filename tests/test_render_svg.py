# SPDX-License-Identifier: Apache-2.0
"""The SVG a chart is served as (#52): the same bytes for the same spec, an id on
every drawn bar, point and bracket, and nothing in it a browser would run or
fetch. Also the n.d. row of replicates not detected, and the style registry."""

from __future__ import annotations

import re
import threading
import xml.etree.ElementTree as ET

import matplotlib
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from proteia.core.analyze import StatisticsSetting, compare, describe
from proteia.core.plotspec import PlotSpec, ValueKind, build_plotspec
from proteia.viz import (
    CHART_STYLES,
    DEFAULT_STYLE,
    chart_style,
    render,
    render_figure,
    render_pdf,
    render_png,
    render_svg,
)
from proteia.viz.styles import legend_lines

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"
# The only URLs an SVG may hold: its namespace names and the SVG 1.1 DTD's
# identifier in matplotlib's prolog. None of them is ever fetched.
_NAMES_NOT_FETCHED = {_SVG_NS, _XLINK_NS, "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"}
# Elements that run code, load a resource, link away or change attributes over time.
_ACTIVE = {"script", "foreignObject", "iframe", "object", "embed", "image", "a", "animate", "set"}
_ACTIVE |= {"animateTransform", "animateMotion", "handler", "listener", "feImage"}

VEHICLE, LOW, HIGH = "vehicle", "10 µM", "50 µM"  # 10 and 50 micro-molar


def _spec(
    groups: dict[str, list[float]],
    *,
    undetected: dict[str, int] | None = None,
    **update: object,
) -> PlotSpec:
    """The chart of ``groups`` (lanes in order, each condition's detected values
    first), with ``undetected`` replicates not detected per condition."""
    lanes, nd, replicates, start = {}, {}, {}, 0
    for condition, values in groups.items():
        missing = (undetected or {}).get(condition, 0)
        lanes[condition] = list(range(start, start + len(values)))
        nd[condition] = list(range(start + len(values), start + len(values) + missing))
        replicates[condition] = len(values) + missing
        start += len(values) + missing
    shown = {c: v for c, v in groups.items() if v}
    test = compare(
        {c: v for c, v in shown.items() if not nd[c]},
        StatisticsSetting(),
        ratio=True,
        reference=VEHICLE,
    )
    spec = build_plotspec(
        shown,
        describe(shown),
        test,
        value_kind=ValueKind.FOLD_CHANGE,
        title="β-catenin / α-tubulin",  # beta-catenin over alpha-tubulin
        lane_indices=lanes,
        first_label=VEHICLE,
        replicates=replicates,
        not_detected_lanes={c: lanes for c, lanes in nd.items() if lanes},
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

    def held_while_saved(spec: PlotSpec, **kwargs):
        begun.append(spec.subtitle)
        drawn = figure(spec, **kwargs)
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
    drawn = [i for i, bar in enumerate(spec.bars) if bar.mean is not None]
    assert _ids(svg, r"bar-\d+") == {f"bar-{i}" for i in drawn}
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


# --- Replicates not detected: the n.d. row (option A) ---

WITH_ND = {VEHICLE: [1.0, 1.1, 0.9], LOW: [2.0, 2.1, 1.9], HIGH: [0.08], "KO": []}
UNDETECTED = {HIGH: 2, "KO": 3}


def _after_save(spec: PlotSpec):
    """The figure of ``spec``, laid out and drawn as a saved file draws it."""
    fig = render_figure(spec)
    FigureCanvasAgg(fig).draw()
    return fig


def test_a_chart_with_no_replicate_not_detected_has_no_row():
    spec = _spec(TESTED)
    svg = render_svg(spec)
    assert not _ids(svg, r"nd-\d+-\d+") and not _ids(svg, r"slot-\d+")
    ax = _after_save(spec).axes[0]
    assert ax.get_ylim()[0] == 0  # the bars stand on the axis
    assert ax.spines["bottom"].get_visible()
    assert ax.get_autoscalex_on()  # the x limits as matplotlib gives them: no empty slot
    assert not [t for t in ax.get_yticklabels(minor=True) if t.get_text() == "n.d."]


def test_each_replicate_not_detected_is_an_open_circle_in_the_row():
    spec = _spec(WITH_ND, undetected=UNDETECTED)
    svg = render_svg(spec)
    assert _ids(svg, r"nd-\d+-\d+") == {"nd-2-0", "nd-2-1", "nd-3-0", "nd-3-1", "nd-3-2"}
    assert _ids(svg, r"bar-\d+") == {"bar-0", "bar-1"}  # no bar where a replicate is n.d.
    assert _ids(svg, r"point-\d+-\d+") >= {"point-2-0"}  # KO's detected value is still drawn
    assert _ids(svg, r"slot-\d+") == {"slot-3"}  # the empty slot is named
    ax = _after_save(spec).axes[0]
    [slot] = [t for t in ax.texts if t.get_gid() == "slot-3"]
    assert slot.get_text() == "n.d."
    labels = [t.get_text() for t in ax.get_yticklabels(minor=True)]
    assert labels == ["n.d."]
    assert [t.get_text() for t in ax.get_xticklabels()][2:] == [
        f"{HIGH}\n(n=3, 2 n.d.)",
        "KO\n(n=3, 3 n.d.)",
    ]


def test_the_row_is_16_points_tall_on_paper():
    spec = _spec(WITH_ND, undetected=UNDETECTED)
    fig = _after_save(spec)
    ax = fig.axes[0]
    bottom, _ = ax.get_ylim()
    lo = ax.spines["left"].get_bounds()[0]
    to_display = ax.transData.transform
    height_px = to_display((0, lo))[1] - to_display((0, bottom))[1]
    assert height_px * 72 / fig.dpi == pytest.approx(16, abs=0.5)
    assert lo == 0  # every value is 0 or above: the numbers start at 0
    assert not ax.spines["bottom"].get_visible()


def test_a_negative_value_lies_inside_the_numbers_above_the_row():
    spec = _spec({VEHICLE: [1.0, 1.1, 0.9], LOW: [-0.4, 0.2, 0.1], "KO": []}, undetected={"KO": 3})
    ax = _after_save(spec).axes[0]
    lo = ax.spines["left"].get_bounds()[0]
    assert lo < -0.4 < ax.get_ylim()[1]  # the lowest point is in the numeric range
    assert ax.get_ylim()[0] < lo  # the row below it


def test_a_crowded_chart_crops_nothing():
    groups = {VEHICLE: [1.0, 1.1, 0.9], LOW: [2.0, 2.1, 1.9], HIGH: [3.0, 3.2, 3.1]}
    groups |= {"50 µM rapamycin + 10 nM bafilomycin A1": [2.0], "KO": []}
    spec = _spec(groups, undetected={"KO": 3}, subtitle="Excluding lanes 4, 8")
    fig = render_figure(spec, statement=True)
    renderer = FigureCanvasAgg(fig).get_renderer()
    boxes = [fig.axes[0].title.get_window_extent(renderer)]
    boxes += [t.get_window_extent(renderer) for t in fig.texts if t.get_gid() == "statement"]
    boxes += [legend.get_window_extent(renderer) for legend in fig.legends]
    assert len(boxes) == 3
    for box in boxes:
        assert 0 <= box.x0 and box.x1 <= fig.bbox.width
        assert 0 <= box.y0 and box.y1 <= fig.bbox.height


def _chart(groups, lanes, undetected, replicates):
    """The chart of ``groups`` with each condition's point lanes, lanes not
    detected and replicates given as they are."""
    shown = {c: v for c, v in groups.items() if v}
    test = compare(
        {c: v for c, v in shown.items() if not undetected.get(c)},
        StatisticsSetting(),
        ratio=True,
        reference=VEHICLE,
    )
    return build_plotspec(
        shown,
        describe(shown),
        test,
        value_kind=ValueKind.FOLD_CHANGE,
        title="β-catenin / α-tubulin",
        lane_indices=lanes,
        first_label=VEHICLE,
        replicates=replicates,
        not_detected_lanes=undetected,
    )


def _slot_texts(spec: PlotSpec) -> dict[str, str]:
    ax = _after_save(spec).axes[0]
    return {t.get_gid(): t.get_text() for t in ax.texts if (t.get_gid() or "").startswith("slot-")}


def test_an_empty_slot_reads_nd_only_when_every_replicate_was_not_detected():
    groups = {VEHICLE: [1.0, 1.1, 0.9], LOW: [2.0, 2.1, 1.9], "KO": []}
    lanes = {VEHICLE: [0, 1, 2], LOW: [3, 4, 5]}
    # KO has 2 replicates: lane 6 not detected, lane 7 with no box (no value).
    partly = _chart(groups, lanes, {"KO": [6]}, {VEHICLE: 3, LOW: 3, "KO": 2})
    assert partly.coverage[2].left_out == "not_detected"
    assert _slot_texts(partly) == {"slot-2": "no value"}
    every = _chart(groups, lanes, {"KO": [6, 7]}, {VEHICLE: 3, LOW: 3, "KO": 2})
    assert _slot_texts(every) == {"slot-2": "n.d."}


def test_replicates_not_detected_keep_their_lane_order_among_the_points():
    # KO: lane 6 not detected, lanes 7 and 8 detected: the open circle is leftmost.
    groups = {VEHICLE: [1.0, 1.1, 0.9], LOW: [2.0, 2.1, 1.9], "KO": [0.3, 0.4]}
    lanes = {VEHICLE: [0, 1, 2], LOW: [3, 4, 5], "KO": [7, 8]}
    spec = _chart(groups, lanes, {"KO": [6]}, {VEHICLE: 3, LOW: 3, "KO": 3})
    ax = _after_save(spec).axes[0]
    xs = {line.get_gid(): line.get_xdata()[0] for line in ax.lines if line.get_gid()}
    assert xs["nd-2-0"] < xs["point-2-0"] < xs["point-2-1"]


def test_an_empty_slot_keeps_the_width_every_bar_would_give():
    full = _spec(TESTED)
    empty = _spec({**TESTED, HIGH: []}, undetected={HIGH: 2})
    assert empty.bars[2].mean is None and not empty.bars[2].points
    full_ax, empty_ax = _after_save(full).axes[0], _after_save(empty).axes[0]
    assert not empty_ax.get_autoscalex_on()  # fixed, not the autoscale of the bars drawn
    assert empty_ax.get_xlim() == pytest.approx(full_ax.get_xlim())


# --- The style registry ---


def test_the_bar_style_is_the_registered_default():
    assert DEFAULT_STYLE == "bar" and CHART_STYLES["bar"] is chart_style("bar")
    style = chart_style("bar")
    spec = _spec(WITH_ND, undetected=UNDETECTED)
    assert style.supports(spec) is None
    assert legend_lines(spec) == spec.statement
    assert style.marks_line(spec) == spec.statement[0]
    assert render_svg(spec, style="bar") == render_svg(spec)


def test_an_unknown_style_is_a_key_error():
    with pytest.raises(KeyError, match="no chart style 'bar-broken-y'"):
        chart_style("bar-broken-y")
    with pytest.raises(KeyError):
        render_svg(_spec(TESTED), style="log-points")


def test_the_statement_is_drawn_only_when_asked():
    spec = _spec(TESTED)
    assert render_svg(spec, statement=True) != render_svg(spec)
    assert render_pdf(spec, statement=True) != render_pdf(spec)
    # The last line, short enough to be drawn unbroken: "Tested: all 3 conditions".
    assert spec.statement[-1].encode() in render_svg(spec, statement=True)
    assert spec.statement[-1].encode() not in render_svg(spec)
