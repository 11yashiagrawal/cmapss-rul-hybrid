"""Phase 2 ML baselines under the chosen RUL cap (issue #61).

Generalizes ``scripts/capped_rul_ablation.py`` from one model to Random Forest,
XGBoost and Elastic Net, on raw and engineered inputs, so that ML and DL are
compared on the same target and the same selection rule:

* the 3 engine-grouped folds of Phase 1 (seed 42);
* training RUL capped at ``--cap`` (125, from #55), with uncapped runs beside
  it as the protocol asks;
* candidates ranked by the selection RMSE from #55: held-out cycles whose true,
  uncapped RUL is at most ``--k`` (130, as in ``configs/dl_base.yaml``);
* the XGBoost learning rate is now part of its grid (Phase 1 left it at 0.3).

Cross-validation opens only ``train_FDxxx.txt``. ``--score-test`` refits each
chosen setting on all training engines and scores the official test once, at
each engine's last cycle against the uncapped RUL. Only #71 runs that, after
the model freeze.

Run from the repository root after ``scripts/download_data.py``:

    python scripts/run_ml_baselines.py               # cross-validation only
    python scripts/run_ml_baselines.py --score-test  # #71 only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from rul.data.labels import add_train_rul
from rul.data.load import load_subset, load_training
from rul.data.split import group_kfold_splits
from rul.data.target import selection_rmse
from rul.evaluation.metrics import rmse
from rul.evaluation.official import predict_test_cycles, save_results, score_last_cycle
from rul.models.experiment import COMPARISON_GRIDS, EngineFeatures, make_comparison_models

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("raw", "engineered")
MODELS = ("random_forest", "xgboost", "elastic_net")

# Phase 1 grids, addressed to the model step (the features are fitted once per fold).
GRIDS = {name: {key.removeprefix("model__"): values for key, values in grid.items()}
         for name, grid in COMPARISON_GRIDS.items()}
GRIDS["xgboost"]["learning_rate"] = [0.05, 0.1, 0.3]
# One small candidate per model, for the test of this script.
QUICK_GRIDS = {
    "random_forest": {"n_estimators": [5], "min_samples_leaf": [3]},
    "xgboost": {"n_estimators": [10], "max_depth": [2], "learning_rate": [0.3]},
    "elastic_net": {"model__alpha": [1.0], "model__l1_ratio": [0.5]},
}
# Thread settings only: they change speed, not results.
SPEED = {"random_forest": {"n_jobs": -1}, "xgboost": {"n_jobs": -1}, "elastic_net": {}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--cap", type=float, default=125.0, help="training RUL cap from #55")
    parser.add_argument("--k", type=float, default=130.0,
                        help="selection threshold on the true RUL (configs/dl_base.yaml)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/CMAPSSData")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/metrics")
    parser.add_argument("--quick", action="store_true", help="tiny grids, for tests")
    parser.add_argument("--score-test", action="store_true",
                        help="refit the chosen settings and score the official test (#71 only)")
    return parser.parse_args(argv)


def cap_label(cap: float | None) -> str:
    return "none" if cap is None else f"{cap:g}"


def capped(rul: pd.Series, cap: float | None) -> pd.Series:
    return rul if cap is None else rul.clip(upper=cap)


def cross_validate(train: pd.DataFrame, caps, grids: dict, args) -> pd.DataFrame:
    """Score every candidate on every fold; one row per (variant, cap, model, params, fold)."""
    rows = []
    folds = list(group_kfold_splits(train, n_splits=args.n_splits, seed=args.seed))
    X = train.drop(columns="rul")
    for variant in VARIANTS:
        models = make_comparison_models(engineered=variant == "engineered", seed=args.seed)
        for fold, (fit_idx, val_idx) in enumerate(folds):
            features = EngineFeatures(engineered=variant == "engineered", seed=args.seed)
            X_fit = features.fit_transform(X.iloc[fit_idx])
            X_val = features.transform(X.iloc[val_idx])
            y_fit, y_val = train["rul"].iloc[fit_idx], train["rul"].iloc[val_idx].to_numpy()
            for cap in caps:
                for name in MODELS:
                    for params in ParameterGrid(grids[name]):
                        model = clone(models[name].named_steps["model"])
                        model.set_params(**params, **SPEED[name])
                        pred = model.fit(X_fit, capped(y_fit, cap)).predict(X_val)
                        rows.append({
                            "variant": variant, "cap": cap_label(cap), "model": name,
                            "params": json.dumps(params, sort_keys=True), "fold": fold,
                            "selection_rmse": selection_rmse(y_val, pred, args.k),
                            "all_cycle_rmse": rmse(y_val, pred),
                        })
                print(f"{variant:>10} fold {fold} cap {cap_label(cap):>4}: done", flush=True)
    return pd.DataFrame(rows)


def choose(grid: pd.DataFrame) -> pd.DataFrame:
    """Best candidate (lowest fold-mean selection RMSE) per variant, cap and model."""
    keys = ["variant", "cap", "model", "params"]
    summary = grid.groupby(keys, sort=False).agg(
        selection_rmse=("selection_rmse", "mean"),
        selection_std=("selection_rmse", "std"),
        all_cycle_rmse=("all_cycle_rmse", "mean"),
    ).reset_index()
    best = summary.loc[summary.groupby(keys[:3], sort=False)["selection_rmse"].idxmin()]
    return best.reset_index(drop=True)


def final_model(variant: str, name: str, params: dict, seed: int) -> Pipeline:
    """Feature pipeline plus the chosen model, ready to refit on all training engines."""
    model = clone(make_comparison_models(engineered=variant == "engineered", seed=seed)[name]
                  .named_steps["model"]).set_params(**params, **SPEED[name])
    return Pipeline([("features", EngineFeatures(engineered=variant == "engineered", seed=seed)),
                     ("model", model)])


def score_test(chosen: pd.DataFrame, args) -> dict:
    """Refit each chosen setting on all training engines and score the official test once."""
    train, test, rul_test = load_subset(args.subset, data_dir=args.data_dir)
    train = add_train_rul(train)
    X = train.drop(columns="rul")
    scores, predictions = {}, None
    for row in chosen.itertuples(index=False):
        cap = None if row.cap == "none" else float(row.cap)
        model = final_model(row.variant, row.model, json.loads(row.params), args.seed)
        model.fit(X, capped(train["rul"], cap))
        last, metrics = score_last_cycle(predict_test_cycles(model, test, rul_test))
        scores.setdefault(row.variant, {}).setdefault(row.cap, {})[row.model] = metrics
        if predictions is None:
            predictions = last[["unit", "cycle", "rul"]].reset_index(drop=True)
        predictions[f"{row.variant}_{row.model}_cap{row.cap}"] = last["prediction"].to_numpy()
    name = f"{args.subset.lower()}_ml_capped_test"
    save_results(name, {"subset": args.subset, "scored_at": "last cycle, uncapped RUL",
                        "models": scores}, predictions, out_dir=args.out_dir)
    return scores


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    caps = (args.cap, None)
    train = add_train_rul(load_training(args.subset, data_dir=args.data_dir))
    grid = cross_validate(train, caps, QUICK_GRIDS if args.quick else GRIDS, args)
    chosen = choose(grid)

    report = {
        "subset": args.subset,
        "issue": 61,
        "protocol": {
            "data": f"train_{args.subset}.txt only",
            "folds": f"{args.n_splits} engine-grouped folds, seed {args.seed}",
            "target": f"training RUL capped at {cap_label(args.cap)}; uncapped shown beside it",
            "selection_metric": f"RMSE on held-out cycles with true (uncapped) RUL <= {args.k:g}",
            "choice": "lowest fold-mean selection RMSE per variant, cap and model",
            "grids": QUICK_GRIDS if args.quick else GRIDS,
        },
        "chosen": [
            {**{k: getattr(r, k) for k in ("variant", "cap", "model")},
             "params": json.loads(r.params),
             "selection_rmse": {"mean": r.selection_rmse, "std": r.selection_std},
             "all_cycle_rmse": r.all_cycle_rmse}
            for r in chosen.itertuples(index=False)
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.subset.lower()}_ml_capped"
    grid.to_csv(args.out_dir / f"{stem}_grid.csv", index=False)
    (args.out_dir / f"{stem}_cv.json").write_text(json.dumps(report, indent=2) + "\n")
    print(chosen.round(2).to_string(index=False))

    if args.score_test:
        report["test"] = score_test(chosen, args)
    return report


if __name__ == "__main__":
    main()
