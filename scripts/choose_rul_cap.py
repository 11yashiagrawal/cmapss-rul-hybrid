"""Choose the RUL training cap on FD001 training engines only (issue #55).

Extends ``scripts/capped_rul_ablation.py``. That prototype scored caps on the
official test set; this script never opens the test or RUL files. For every
seed and engine-grouped fold it:

1. estimates K on the training part of the fold (:func:`rul.data.target.estimate_k`)
   at several percentiles, so the percentile sensitivity comes for free;
2. trains each model on the fold's training engines with RUL clipped at each
   candidate cap (``None`` = uncapped);
3. scores the held-out engines against their true, uncapped RUL, both on every
   cycle and with the selection metric (cycles with true RUL <= K).

The cap with the lowest fold-mean selection RMSE wins for each seed and model.
Hyperparameters are fixed at the Phase 1 engineered-feature choices, so only
the cap changes between runs.

Run from the repository root after ``scripts/download_data.py``:

    python scripts/choose_rul_cap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from rul.data.labels import add_train_rul
from rul.data.load import load_training
from rul.data.split import group_kfold_splits
from rul.data.target import DEFAULT_PERCENTILE, estimate_k, selection_rmse
from rul.evaluation.metrics import rmse
from rul.models.experiment import make_comparison_models

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "metrics"
SUBSET, N_SPLITS = "FD001", 3
SEEDS = (42, 1, 2, 3, 4)
CAPS = (None, 100, 125, 150)
PERCENTILES = (50, 60, 70, 75, 80, 85, 90, 95)
# Phase 1 engineered-feature choices (results/metrics/fd001_test.json).
PARAMS = {
    "random_forest": {"n_estimators": 100, "min_samples_leaf": 1, "n_jobs": -1},
    "xgboost": {"n_estimators": 200, "max_depth": 5, "n_jobs": -1},
    "elastic_net": {"model__alpha": 1.0, "model__l1_ratio": 0.2},
}


def cap_label(cap: float | None) -> str:
    return "none" if cap is None else str(int(cap))


def run_fold(train: pd.DataFrame, train_idx, val_idx, seed: int, fold: int) -> list[dict]:
    """Score every model and cap on one fold."""
    fit_rows, val_rows = train.iloc[train_idx], train.iloc[val_idx]
    ks = estimate_k(fit_rows.drop(columns="rul"), PERCENTILES)
    pipelines = make_comparison_models(engineered=True, seed=seed)
    features = clone(pipelines["random_forest"].named_steps["features"])
    X_fit = features.fit_transform(fit_rows.drop(columns="rul"))
    X_val = features.transform(val_rows.drop(columns="rul"))
    y_val = val_rows["rul"].to_numpy()

    rows = []
    for cap in CAPS:
        y_fit = fit_rows["rul"].clip(upper=cap) if cap is not None else fit_rows["rul"]
        for name, pipeline in pipelines.items():
            model = clone(pipeline.named_steps["model"]).set_params(**PARAMS[name])
            pred = model.fit(X_fit, y_fit).predict(X_val)
            every = rmse(y_val, pred)
            at_k90 = selection_rmse(y_val, pred, ks[DEFAULT_PERCENTILE])
            for q, k in ks.items():
                rows.append({
                    "seed": seed, "fold": fold, "model": name, "cap": cap_label(cap),
                    "percentile": q, "k": k, "n_selected": int((y_val <= k).sum()),
                    "rmse_selected": selection_rmse(y_val, pred, k), "rmse_all": every,
                })
            print(f"seed {seed} fold {fold} cap {cap_label(cap):>4} {name:>13}: "
                  f"all {every:6.2f}, <=K90 {at_k90:6.2f}", flush=True)
    return rows


def choose(scores: pd.DataFrame) -> pd.DataFrame:
    """Winning cap per seed, model and percentile (lowest fold-mean selection RMSE)."""
    mean = (scores.groupby(["percentile", "seed", "model", "cap"])["rmse_selected"]
            .mean().reset_index())
    best = mean.loc[mean.groupby(["percentile", "seed", "model"])["rmse_selected"].idxmin()]
    return best.rename(columns={"cap": "chosen_cap"}).reset_index(drop=True)


def main() -> None:
    train = add_train_rul(load_training(SUBSET, data_dir=ROOT / "data/raw/CMAPSSData"))
    rows = []
    for seed in SEEDS:
        for fold, (tr, va) in enumerate(group_kfold_splits(train, N_SPLITS, seed)):
            rows += run_fold(train, tr, va, seed, fold)
    scores = pd.DataFrame(rows)
    choices = choose(scores)

    main_q = choices[choices["percentile"] == DEFAULT_PERCENTILE]
    sensitivity = {
        str(int(q)): {
            "k_mean": float(scores.loc[scores["percentile"] == q, "k"].mean()),
            "chosen_cap_counts": part["chosen_cap"].value_counts().to_dict(),
        }
        for q, part in choices.groupby("percentile")
    }
    at_q = scores[scores["percentile"] == DEFAULT_PERCENTILE]
    summary = {
        "subset": SUBSET,
        "issue": 55,
        "protocol": {
            "data": f"train_{SUBSET}.txt only; official test and RUL files are not opened",
            "folds": f"{N_SPLITS} engine-grouped folds per seed",
            "seeds": list(SEEDS),
            "caps": [cap_label(c) for c in CAPS],
            "models": PARAMS,
            "features": "Phase 1 engineered features, fitted per training fold",
            "selection_metric": "RMSE on held-out cycles with true (uncapped) RUL <= K",
            "k_rule": (f"{DEFAULT_PERCENTILE:g}th percentile of per-engine hinge-onset RUL, "
                       "estimated on the training part of each fold"),
        },
        "k_by_fold": (at_q.groupby(["seed", "fold"])["k"].first()
                      .reset_index().to_dict(orient="records")),
        "chosen_cap_counts": main_q["chosen_cap"].value_counts().to_dict(),
        "mean_rmse_by_cap": {
            metric: at_q.groupby(["model", "cap"])[metric].mean().unstack().round(3).to_dict(orient="index")
            for metric in ("rmse_selected", "rmse_all")
        },
        "percentile_sensitivity": sensitivity,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT / "fd001_rul_cap_scores.csv", index=False)
    choices.to_csv(OUT / "fd001_rul_cap_choices.csv", index=False)
    (OUT / "fd001_rul_cap_choice.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps({k: summary[k] for k in ("chosen_cap_counts", "mean_rmse_by_cap")}, indent=2))


if __name__ == "__main__":
    main()
