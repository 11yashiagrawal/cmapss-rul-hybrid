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
