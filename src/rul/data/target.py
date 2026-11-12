"""Choose the RUL training cap from training engines only (issue #55).

Why a cap at all: a C-MAPSS engine starts healthy and its sensors only begin to
drift some time before failure. While it is healthy, an engine with 300 cycles
left looks the same as one with 200 left, so the uncapped RUL is not learnable
there. The usual fix clips the training target at a cap (flat while healthy,
then falling to 0). This module decides *which* cap, without looking at the
official test set.

The rule has three steps, all fitted on the training fold only:

1. :class:`HealthIndex` turns the retained sensors into one number per cycle
   that rises as the engine wears (z-score, align signs, average).
2. :func:`hinge_onsets` fits "flat, then linear" to each engine's index and
   records the RUL at the bend, i.e. where degradation becomes visible.
3. :func:`estimate_k` takes a high percentile of those onset RULs. Cycles with
   true RUL at or below ``K`` are the ones where degradation is visible for
   most engines, and :func:`selection_rmse` scores candidate caps only there.

Scoring on every held-out cycle instead would reward no cap at all, because the
held-out training engines contain very high RUL cycles that the official test
set (cut off well before failure) never asks about.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rul.data.load import SENSOR_COLS
from rul.evaluation.metrics import ArrayLike, rmse
from rul.features.cleaning import select_sensors

DEFAULT_PERCENTILE = 90.0
MIN_SEGMENT = 5


class HealthIndex:
    """One-number wear indicator, fitted on training engines only.

    ``fit`` learns, from the training rows, which sensors to keep, their mean
    and standard deviation, and the sign that makes each one rise toward
    failure. ``transform`` applies those numbers unchanged to any rows, so a
    held-out engine never influences its own index.

    Args:
        min_abs_corr: Sensors whose absolute correlation with the cycles to
            failure is below this are left out. They pass the variance check in
            :func:`select_sensors` but carry no wear trend (``s_6`` on FD001),
            so averaging them in only adds noise.
    """

    def __init__(self, min_abs_corr: float = 0.1) -> None:
        self.min_abs_corr = min_abs_corr

    def fit(self, train: pd.DataFrame) -> "HealthIndex":
        """Learn sensors, scaling and signs from run-to-failure ``train`` rows."""
        cycles_left = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
        corr = {
            sensor: np.corrcoef(train[sensor], -cycles_left)[0, 1]
            for sensor in select_sensors(train, [c for c in SENSOR_COLS if c in train])
        }
        kept = [s for s, c in corr.items() if np.isfinite(c) and abs(c) >= self.min_abs_corr]
        if not kept:
            raise ValueError("no sensor trends with the cycles to failure")
        self.sensors_ = kept
        self.mean_ = train[kept].mean()
        self.std_ = train[kept].std(ddof=0)
        self.sign_ = pd.Series({s: np.sign(corr[s]) for s in kept})
        return self

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        """Health index of each row: mean of the sign-aligned z-scores."""
        z = (frame[self.sensors_] - self.mean_) / self.std_
        return (z * self.sign_).mean(axis=1).rename("health_index")


def fit_hinge(cycles: ArrayLike, values: ArrayLike, min_segment: int = MIN_SEGMENT) -> dict:
    """Least-squares fit of ``a`` (flat) up to cycle ``tau``, then ``a + b (t - tau)``.

    Every observed cycle that leaves at least ``min_segment`` points on each
    side is tried as ``tau``. For a fixed ``tau`` the model is linear in
    ``(a, b)``, so it is solved in closed form, and the ``tau`` with the
    smallest squared error wins. Only rising fits (``b > 0``) count as an
    onset, because the health index is built to rise toward failure.

    Returns:
        ``{"onset_cycle", "level", "slope", "sse"}`` for the best ``tau``.
    """
    t = np.asarray(cycles, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError("cycles and values must be 1-D arrays of the same length")
    order = np.argsort(t)
    t, y = t[order], y[order]
    if t.size < 2 * min_segment + 1:
        raise ValueError(f"need at least {2 * min_segment + 1} cycles, got {t.size}")

    taus = t[min_segment - 1 : t.size - min_segment]
    x = np.maximum(t[None, :] - taus[:, None], 0.0)  # one row per candidate tau
    n = t.size
    sx, sy = x.sum(1), y.sum()
    sxx, sxy = (x * x).sum(1), (x * y).sum(1)
    slope = (n * sxy - sx * sy) / (n * sxx - sx**2)
    level = (sy - slope * sx) / n
    sse = ((y[None, :] - level[:, None] - slope[:, None] * x) ** 2).sum(1)
    sse = np.where(slope > 0, sse, np.inf)
    if not np.isfinite(sse).any():
        raise ValueError("no rising hinge fits these values")
    best = int(np.argmin(sse))
    return {
        "onset_cycle": float(taus[best]),
        "level": float(level[best]),
        "slope": float(slope[best]),
        "sse": float(sse[best]),
    }


def hinge_onsets(
    train: pd.DataFrame, health: pd.Series, min_segment: int = MIN_SEGMENT
) -> pd.DataFrame:
    """Fit a hinge to each training engine and record the RUL at its onset.

    Training engines run to failure, so the RUL at the onset cycle is simply
    ``last cycle - onset cycle``.

    Returns:
        One row per ``unit``: ``onset_cycle``, ``last_cycle``, ``onset_rul``,
        ``level`` and ``slope``.
    """
    rows = []
    for unit, engine in train.assign(health=health.to_numpy()).groupby("unit"):
        fit = fit_hinge(engine["cycle"], engine["health"], min_segment)
        last = float(engine["cycle"].max())
        rows.append({"unit": unit, **fit, "last_cycle": last,
                     "onset_rul": last - fit["onset_cycle"]})
    return pd.DataFrame(rows).set_index("unit")


def estimate_k(
    train: pd.DataFrame,
    percentile: float | Sequence[float] = DEFAULT_PERCENTILE,
    min_abs_corr: float = 0.1,
) -> float | dict[float, float]:
    """K for one training fold: a percentile of the per-engine onset RULs.

    Everything (health index, hinge fits, percentile) uses ``train`` only.
    Pass several percentiles to get ``{percentile: K}`` from one set of fits.
    """
    index = HealthIndex(min_abs_corr).fit(train)
    onset = hinge_onsets(train, index.transform(train))["onset_rul"]
    if np.ndim(percentile) == 0:
        return float(np.percentile(onset, percentile))
    return {float(q): float(np.percentile(onset, q)) for q in percentile}


def selection_rmse(y_true: ArrayLike, y_pred: ArrayLike, k: float) -> float:
    """Selection metric for caps and tuning: RMSE on cycles with true RUL <= ``k``.

    ``y_true`` must be the true, uncapped RUL of held-out cycles. Cycles above
    ``k`` are dropped because, for most engines, no degradation is visible
    there yet, and the official test set rarely asks about them.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"got {true.size} targets and {pred.size} predictions")
    keep = true <= k
    if not keep.any():
        raise ValueError(f"no held-out cycle has true RUL <= {k}")
    return rmse(true[keep], pred[keep])
