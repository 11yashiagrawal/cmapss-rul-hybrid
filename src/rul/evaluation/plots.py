"""Evaluation plots shared by every model, sized and styled for the IEEE report.

Each function draws on an existing ``ax`` when given one (so notebooks can build
subplot grids), otherwise on a new figure, and returns the Axes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ArrayLike = Sequence[float] | np.ndarray

# src/rul/evaluation/plots.py -> repo root is three folders up from this file's folder.
FIGURES_DIR = Path(__file__).resolve().parents[3] / "results" / "figures"

# Figures go into a white-page PDF, so the surface is plain white.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Categorical slots in a fixed, colour-blind-checked order. A model keeps its
# slot in every figure, so "XGBoost is orange" holds across the whole report.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

STAGES = ("early", "mid", "late")
METRIC_LABELS = {"rmse": "RMSE (cycles)", "mae": "MAE (cycles)", "nasa_score": "NASA score"}


def plot_pred_vs_true(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    ax: Axes | None = None,
    title: str | None = None,
    color: str = SERIES[0],
    limits: tuple[float, float] | None = None,
) -> Axes:
    """Scatter of predicted against true RUL, one point per engine.

    Points on the ``y = x`` line are perfect. Above it the model
    predicts too much life left (late, the costly side); below it, too little.
    Pass the model's ``SERIES`` colour so it matches the other figures, and the
    same ``limits`` to every panel of a figure so they can be compared.
    """
    true, pred = _paired(y_true, y_pred)
    ax = _axes(ax)
    low, high = limits if limits is not None else pred_vs_true_limits(true, pred)

    ax.plot([low, high], [low, high], color=INK_MUTED, linewidth=1, zorder=2)
    # Label runs along the line, just under it, where predictions are rarest.
    at = low + 0.82 * (high - low)
    ax.annotate("perfect prediction", (at, at), xytext=(5, -5),
                textcoords="offset points", rotation=45, rotation_mode="anchor",
                ha="center", va="top", fontsize=8, color=INK_MUTED)
    ax.scatter(true, pred, s=36, color=color, edgecolors=SURFACE, linewidths=1, zorder=3)

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    _title(ax, title)
    return ax


def pred_vs_true_limits(*values: ArrayLike) -> tuple[float, float]:
    """Square axis limits that show every value: from 0 (or below, for negative
    predictions) to the largest value, with a 5% margin."""
    flat = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    low, high = min(0.0, float(flat.min())), float(flat.max())
    margin = 0.05 * (high - low)
    return (low - margin if low < 0 else 0.0), high + margin


def plot_trajectory(
    cycle: ArrayLike,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    ax: Axes | None = None,
    title: str | None = None,
    color: str = SERIES[0],
) -> Axes:
    """True and predicted RUL of one engine across its recorded cycles.

    The true curve is the neutral reference; the prediction carries the
    colour, which should be the model's ``SERIES`` colour.
    """
    true, pred = _paired(y_true, y_pred)
    cycles = np.asarray(cycle)
    if cycles.shape != true.shape:
        raise ValueError(f"got {cycles.size} cycles for {true.size} targets")
    ax = _axes(ax)

    ax.plot(cycles, true, color=INK_SECONDARY, linewidth=2, label="True RUL", zorder=2)
    ax.plot(cycles, pred, color=color, linewidth=2, label="Predicted RUL", zorder=3)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    _legend(ax)
    _title(ax, title)
    return ax


def plot_stage_errors(
    results: Mapping[str, dict],
    metric: str = "rmse",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Grouped bars: each model's error in the early, mid and late stage.

    Args:
        results: ``{model name: evaluate(...) result}``. Dict order sets each
            model's colour, so pass models in the same order in every figure.
        metric: ``"rmse"``, ``"mae"`` or ``"nasa_score"``.
    """
    if metric not in METRIC_LABELS:
        raise ValueError(f"metric must be one of {', '.join(METRIC_LABELS)}, got {metric!r}")
    if len(results) > len(SERIES):
        raise ValueError(
            f"at most {len(SERIES)} models per chart (got {len(results)}); split them across figures"
        )
    bins = {tuple(result["stage_bins"]) for result in results.values()}
    if len(bins) > 1:
        raise ValueError(f"all results must use the same stage bins, got {sorted(bins)}")
    ax = _axes(ax)

    x = np.arange(len(STAGES))
    width = min(0.15, 0.8 / len(results))
    for i, (name, result) in enumerate(results.items()):
        heights = [_or_nan(result["by_stage"][stage][metric]) for stage in STAGES]
        offset = (i - (len(results) - 1) / 2) * width
        ax.bar(x + offset, heights, width=width, color=SERIES[i], edgecolor=SURFACE,
               linewidth=1.5, label=name, zorder=3)

    ((late_max, mid_max),) = bins
    ax.set_xticks(x, [f"Early\n(RUL > {mid_max:g})", f"Mid\n({late_max:g} < RUL ≤ {mid_max:g})",
                      f"Late\n(RUL ≤ {late_max:g})"])
    ax.grid(False, axis="x")
    # Headroom above the tallest bar keeps the legend row clear of the data.
    tallest = max((bar.get_height() for c in ax.containers for bar in c
                   if not np.isnan(bar.get_height())), default=1.0)
    ax.set_ylim(0, tallest * 1.3)
    ax.set_ylabel(METRIC_LABELS[metric])
    _legend(ax, loc="upper left", ncols=min(len(results), 4))
    _title(ax, title)
    return ax


def save_figure(fig: Figure, name: str, out_dir: str | Path = FIGURES_DIR) -> Path:
    """Save ``fig`` as ``<out_dir>/<name>.png`` at print resolution; return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return path


def _or_nan(value: float | None) -> float:
    """Empty stages report ``None``; plot them as a missing bar, not a zero."""
    return np.nan if value is None else value


def _paired(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"got {true.size} targets and {pred.size} predictions")
    return true, pred


def _axes(ax: Axes | None) -> Axes:
    """Return ``ax``, or a new styled figure's Axes, with the shared chart chrome."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    return ax


def _legend(ax: Axes, loc: str = "upper right", ncols: int = 1) -> None:
    legend = ax.legend(frameon=False, fontsize=9, loc=loc, ncols=ncols)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def _title(ax: Axes, title: str | None) -> None:
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
