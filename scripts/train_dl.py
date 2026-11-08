"""Cross-validate a sequence model from a YAML config, and optionally refit it.

Run from the repository root after ``scripts/download_data.py``:

    python scripts/train_dl.py --config configs/lstm.yaml
    python scripts/train_dl.py --config configs/lstm.yaml --set target.cap=null --name lstm_uncapped
    python scripts/train_dl.py --config configs/lstm.yaml --refit --seed 43

Cross-validation writes ``results/metrics/<subset>_cv_<name>.json`` (scores,
folds, config) and ``..._predictions.csv`` (every held-out cycle). With
``--refit`` the model is then trained on every training engine for the
median best epoch and saved to ``results/models/<subset>_<name>_seed<seed>.pt``
(gitignored).

This script never reads the official test set. That runs once, after the
model freeze, in #71.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rul.data.load import load_subset
from rul.evaluation.official import METRICS_DIR
from rul.training.config import apply_overrides, load_config, run_cv, run_parts, run_refit
from rul.training.runs import save_cv

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="model config, e.g. configs/lstm.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. training.epochs=5")
    parser.add_argument("--name", help="run name (default: the config's name)")
    parser.add_argument("--seed", type=int, help="training seed; added to the run name")
    parser.add_argument("--refit", action="store_true",
                        help="after CV, train on all engines and save a checkpoint")
    parser.add_argument("--data-dir", default=ROOT / "data" / "raw" / "CMAPSSData")
    parser.add_argument("--metrics-dir", default=METRICS_DIR)
    parser.add_argument("--models-dir", default=ROOT / "results" / "models")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run cross-validation (and the refit) described by the arguments."""
    args = parse_args(argv)
    overrides = list(args.set) + ([f"training.seed={args.seed}"] if args.seed is not None else [])
    config = apply_overrides(load_config(args.config), overrides)
    subset = config.get("subset", "FD001")
    name = args.name or config["name"]
    if args.seed is not None:
        name = f"{name}_seed{args.seed}"
    parts = run_parts(config)
    seed = parts.train_config.seed

    train = load_subset(subset, data_dir=args.data_dir).train
    cv = run_cv(config, train)
    meta = {"name": name, "subset": subset, "seed": seed, "config": config}
    paths = save_cv(f"{subset.lower()}_cv_{name}", cv, meta=meta, out_dir=args.metrics_dir)

    summary = cv.summary()
    print(f"{name}: selection RMSE {summary['selection_rmse']['mean']:.2f} "
          f"± {summary['selection_rmse']['std']:.2f}, all-cycle RMSE {summary['rmse']['mean']:.2f}, "
          f"best epochs {[f.best_epoch for f in cv.folds]}, "
          f"{sum(f.seconds for f in cv.folds):.0f} s")
    print(f"saved {paths['metrics']}")

    if args.refit:
        epochs = cv.median_best_epoch
        model, seconds = run_refit(config, train, epochs)
        base_name = args.name or config["name"]
        path = Path(args.models_dir) / f"{subset.lower()}_{base_name}_seed{seed}.pt"
        model.save(path, meta={**meta, "epochs": epochs, "seconds": seconds})
        print(f"refit in {seconds:.0f} s (networks: {epochs} epochs), saved {path}")


if __name__ == "__main__":
    main()
