# SPDX-License-Identifier: Apache-2.0
"""Tests for the plot spec builder and the matplotlib renderer."""

from proteia.core.analyze import compare, describe
from proteia.core.plotspec import ErrorType, ValueKind, build_plotspec
from proteia.viz import render_figure, save_figure


def _spec(error_type=ErrorType.SD):
    groups = {"ctl": [1.0, 1.1, 0.9], "A": [2.0, 2.1, 1.9], "B": [0.5, 0.6]}
    return build_plotspec(
        groups,
        describe(groups),
        compare(groups),
        value_kind=ValueKind.LOADING_NORMALIZED,
        error_type=error_type,
        title="test",
        lane_indices={"ctl": [0, 1, 2], "A": [3, 4, 5], "B": [6, 7]},
    )


def test_build_plotspec_carries_points_and_provenance():
    spec = _spec()
    bar = next(b for b in spec.bars if b.label == "A")
    assert bar.n == 3
    assert bar.points == [2.0, 2.1, 1.9]
    assert bar.lane_indices == [3, 4, 5]  # provenance link survives


def test_error_type_selects_sd_vs_sem():
    sd_bar = next(b for b in _spec(ErrorType.SD).bars if b.label == "ctl")
    sem_bar = next(b for b in _spec(ErrorType.SEM).bars if b.label == "ctl")
    assert sem_bar.error < sd_bar.error  # SEM = SD / sqrt(n)


def test_only_significant_comparisons_become_brackets():
    spec = _spec()
    for comp in spec.comparisons:
        assert comp.p_value < 0.05


def test_significance_stars():
    spec = _spec()
    for comp in spec.comparisons:
        assert comp.stars in {"*", "**", "***"}


def test_value_kind_sets_y_label():
    spec = _spec()
    assert "target / loading" in spec.y_label


def test_render_figure_has_one_axes():
    fig = render_figure(_spec())
    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylabel()


def test_save_figure_writes_file(tmp_path):
    out = tmp_path / "chart.png"
    save_figure(_spec(), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_first_label_moves_to_front():
    groups = {"A": [1.0, 1.1], "ctl": [1.0, 0.9], "B": [2.0, 2.1]}
    spec = build_plotspec(
        groups, describe(groups), compare(groups),
        value_kind=ValueKind.FOLD_CHANGE, first_label="ctl",
    )
    assert [b.label for b in spec.bars] == ["ctl", "A", "B"]  # control leftmost, rest in order


def test_render_handles_empty_group_without_crashing():
    # A condition with no values (all-None lanes) yields a zero-point bar; the
    # renderer must not raise on it (regression for the strict-zip crash).
    groups = {"a": [1.0, 2.0], "empty": []}
    spec = build_plotspec(
        groups, describe(groups), compare(groups), value_kind=ValueKind.LOADING_NORMALIZED
    )
    fig = render_figure(spec)
    assert len(fig.axes) == 1
