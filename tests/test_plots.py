"""Tests for rul.evaluation.plots."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render off-screen; must run before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from rul.evaluation.metrics import evaluate  # noqa: E402
from rul.evaluation.plots import (  # noqa: E402
    FIGURES_DIR,
    SERIES,
    plot_pred_vs_true,
    plot_stage_errors,
    plot_trajectory,
    pred_vs_true_limits,
    save_figure,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_pred_vs_true_draws_one_point_per_engine_and_a_perfect_prediction_line():
    ax = plot_pred_vs_true([10, 50, 120], [15, 40, 130])

    assert ax.collections[0].get_offsets().tolist() == [[10, 15], [50, 40], [120, 130]]
    xs, ys = ax.lines[0].get_data()
    assert list(xs) == list(ys)  # the y = x reference
    assert xs[0] == 0 and xs[-1] >= 130  # and it spans every point
    assert "true" in ax.get_xlabel().lower()
    assert "predicted" in ax.get_ylabel().lower()


def test_pred_vs_true_can_draw_points_in_another_models_colour():
    # Lets each model keep its series colour when panels sit side by side.
    ax = plot_pred_vs_true([10, 50], [15, 40], color=SERIES[1])

    assert tuple(ax.collections[0].get_facecolor()[0]) == to_rgba(SERIES[1])


def test_pred_vs_true_keeps_negative_predictions_in_view():
    # Linear models can predict a negative RUL; hiding those points would flatter them.
    ax = plot_pred_vs_true([10, 50], [-20, 40])

    bottom, top = ax.get_ylim()
    assert bottom < -20 and top > 50
    assert ax.get_xlim() == (bottom, top)  # square, so y = x stays at 45 degrees


def test_pred_vs_true_panels_can_share_axis_limits():
    ax = plot_pred_vs_true([10, 50], [15, 40], limits=(-50, 250))

    assert ax.get_xlim() == ax.get_ylim() == (-50, 250)
    xs, ys = ax.lines[0].get_data()
    assert (xs[0], xs[-1]) == (ys[0], ys[-1]) == (-50, 250)


@pytest.mark.parametrize("values, expected", [
    (([10, 50], [15, 40]), (0, 52.5)),  # 50 plus a 5% margin; the axis starts at 0
    (([10, 50], [-20, 40], [0, 100]), (-26, 106)),  # span -20..100, margin 6 on each side
])
def test_shared_limits_cover_every_panels_values(values, expected):
    assert pred_vs_true_limits(*values) == pytest.approx(expected)


def test_pred_vs_true_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="3 targets"):
        plot_pred_vs_true([10, 50, 120], [15, 40])


def test_trajectory_draws_true_and_predicted_rul_over_the_engines_cycles():
    ax = plot_trajectory([1, 2, 3], [114, 113, 112], [100, 105, 111])

    lines = {line.get_label(): line.get_data() for line in ax.get_lines()}
    assert [list(v) for v in lines["True RUL"]] == [[1, 2, 3], [114, 113, 112]]
    assert [list(v) for v in lines["Predicted RUL"]] == [[1, 2, 3], [100, 105, 111]]
    assert ax.get_legend() is not None
    assert "cycle" in ax.get_xlabel().lower()


def test_trajectory_can_draw_the_prediction_in_another_models_colour():
    ax = plot_trajectory([1, 2], [114, 113], [100, 105], color=SERIES[2])

    lines = {line.get_label(): line for line in ax.get_lines()}
    assert to_rgba(lines["Predicted RUL"].get_color()) == to_rgba(SERIES[2])


def test_trajectory_rejects_cycles_that_do_not_match_the_targets():
    with pytest.raises(ValueError, match="2 cycles for 3 targets"):
        plot_trajectory([1, 2], [114, 113, 112], [100, 105, 111])


# One engine per stage (true RUL 150 / 75 / 20), so each stage's RMSE is just
# that engine's absolute error.
RF = evaluate([150, 75, 20], [160, 70, 24])  # errors 10, 5, 4
XGB = evaluate([150, 75, 20], [150, 80, 30])  # errors 0, 5, 10


def test_stage_errors_draw_one_bar_group_per_stage_and_one_colour_per_model():
    ax = plot_stage_errors({"Random Forest": RF, "XGBoost": XGB})

    rf_bars, xgb_bars = ax.containers
    assert [b.get_height() for b in rf_bars] == [10, 5, 4]
    assert [b.get_height() for b in xgb_bars] == [0, 5, 10]
    assert [rf_bars.get_label(), xgb_bars.get_label()] == ["Random Forest", "XGBoost"]
    assert [t.get_text().split()[0] for t in ax.get_xticklabels()] == ["Early", "Mid", "Late"]
    # Colour follows the model's position: first model slot 1, second slot 2.
    assert rf_bars[0].get_facecolor() == to_rgba(SERIES[0])
    assert xgb_bars[0].get_facecolor() == to_rgba(SERIES[1])
    assert ax.get_legend() is not None


def test_stage_errors_can_show_mae_instead_of_rmse():
    rf = evaluate([150, 150, 75], [160, 150, 70])  # early errors 10 and 0: RMSE 7.07, MAE 5

    ax = plot_stage_errors({"Random Forest": rf}, metric="mae")

    assert [b.get_height() for b in ax.containers[0]][:2] == [5, 5]
    assert "MAE" in ax.get_ylabel()


def test_stage_errors_leave_an_empty_stage_blank_instead_of_failing():
    only_early = evaluate([150], [160])  # no mid or late engines

    ax = plot_stage_errors({"Random Forest": only_early})

    heights = [b.get_height() for b in ax.containers[0]]
    assert heights[0] == 10
    assert all(math.isnan(h) for h in heights[1:])


def test_stage_errors_reject_an_unknown_metric():
    with pytest.raises(ValueError, match="rmse, mae, nasa_score"):
        plot_stage_errors({"Random Forest": RF}, metric="r2")


def test_stage_errors_refuse_more_models_than_distinct_colours():
    too_many = {f"model {i}": RF for i in range(9)}

    with pytest.raises(ValueError, match="at most 8 models"):
        plot_stage_errors(too_many)


def test_stage_errors_refuse_results_scored_with_different_stage_bins():
    other_bins = evaluate([150, 75, 20], [160, 70, 24], stage_bins=(30, 60))

    with pytest.raises(ValueError, match="same stage bins"):
        plot_stage_errors({"Random Forest": RF, "XGBoost": other_bins})


def test_save_figure_writes_a_png_and_creates_the_folder(tmp_path):
    ax = plot_pred_vs_true([10, 50], [12, 45])

    path = save_figure(ax.figure, "fd001_rf_pred_vs_true", out_dir=tmp_path / "figures")

    assert path == tmp_path / "figures" / "fd001_rf_pred_vs_true.png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figures_default_to_results_figures_in_the_repo():
    repo_root = Path(__file__).resolve().parents[1]

    assert FIGURES_DIR == repo_root / "results" / "figures"
