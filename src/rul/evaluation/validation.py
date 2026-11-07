"""Decide whether differences between models are real, not just run-to-run noise.

Every function works on per-engine predictions, scored once per engine at its
last cycle. Predictions may be one run (``[engines]``) or several seeds
(``[seeds, engines]``); with several seeds, RMSE is the mean of the per-seed
RMSEs and per-engine absolute error is averaged over seeds.

The value test (tracker #83) is fixed in advance. A hybrid adds value over its
best single branch only if at least one of these holds:

1. The paired bootstrap 95% CI on the RMSE difference (hybrid - branch) lies
   entirely below zero.
2. The Holm-corrected Wilcoxon signed-rank test on per-engine absolute error
   gives p < 0.05, with the hybrid's median absolute error lower.
3. RMSE varies less across seeds, with a mean RMSE that is no worse.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

from rul.evaluation.metrics import STAGES

METRICS = ("rmse", "mae", "nasa_score")


def aggregate_seeds(runs: Sequence[dict]) -> dict:
    """Mean and standard deviation of :func:`rul.evaluation.metrics.evaluate` results.

    Args:
        runs: One ``evaluate`` dict per seed, all on the same engines.

    Returns:
        ``{"n_seeds", metric: {"mean", "std"}, "by_stage": {stage: {metric: ...}}}``.
        The standard deviation uses ``ddof=1`` (sample spread across seeds) and
        is 0 for a single seed.
    """
    if not runs:
        raise ValueError("need at least one run")

    def summary(values: list) -> dict:
        if any(v is None for v in values):
            return {"mean": None, "std": None}
        arr = np.asarray(values, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}

    result: dict = {"n_seeds": len(runs)}
    result.update({m: summary([r[m] for r in runs]) for m in METRICS})
    result["by_stage"] = {
        stage: {m: summary([r["by_stage"][stage][m] for r in runs]) for m in METRICS}
        for stage in STAGES
    }
    return result


def _as_runs(y_true: np.ndarray, y_pred) -> np.ndarray:
    """Errors shaped ``[seeds, engines]``."""
    pred = np.atleast_2d(np.asarray(y_pred, dtype=float))
    if pred.shape[1] != y_true.size:
        raise ValueError(f"got {y_true.size} engines and predictions shaped {pred.shape}")
    return pred - y_true


def _metric(errors: np.ndarray, metric: str) -> np.ndarray:
    """Seed-averaged metric over the last axis; errors are ``[seeds, ..., engines]``."""
    if metric == "rmse":
        per_seed = np.sqrt(np.mean(errors**2, axis=-1))
    elif metric == "mae":
        per_seed = np.mean(np.abs(errors), axis=-1)
    else:
        raise ValueError(f"metric must be 'rmse' or 'mae', got {metric!r}")
    return per_seed.mean(axis=0)


def _resamples(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _interval(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def bootstrap_ci(
    y_true,
    y_pred,
    metric: str = "rmse",
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for RMSE or MAE, resampling engines with replacement.

    Returns:
        ``{"estimate", "low", "high"}`` in cycles.
    """
    true = np.asarray(y_true, dtype=float)
    errors = _as_runs(true, y_pred)
    idx = _resamples(true.size, n_boot, seed)
    samples = _metric(errors[:, idx], metric)
    low, high = _interval(samples, alpha)
    return {"estimate": float(_metric(errors, metric)), "low": low, "high": high}


def paired_bootstrap_diff(
    y_true,
    pred_a,
    pred_b,
    metric: str = "rmse",
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for ``metric(a) - metric(b)``, resampling the same engines for both.

    A negative difference means model ``a`` is better.

    Returns:
        ``{"difference", "low", "high", "excludes_zero"}``.
    """
    true = np.asarray(y_true, dtype=float)
    err_a, err_b = _as_runs(true, pred_a), _as_runs(true, pred_b)
    idx = _resamples(true.size, n_boot, seed)
    samples = _metric(err_a[:, idx], metric) - _metric(err_b[:, idx], metric)
    low, high = _interval(samples, alpha)
    return {
        "difference": float(_metric(err_a, metric) - _metric(err_b, metric)),
        "low": low,
        "high": high,
        "excludes_zero": bool(low > 0 or high < 0),
    }
