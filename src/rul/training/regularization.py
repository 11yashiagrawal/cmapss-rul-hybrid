"""One-factor LSTM regularization study through the shared trainer (#66).

Only run-to-failure training cycles enter this module. Completed CV runs are
cached with their config, input/code hashes and runtime versions; predictions
are retained so #55 can change the selection rule without retraining.
"""

from __future__ import annotations

import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import pandas as pd
import torch

from rul.training.config import apply_overrides, run_cv

FACTORS = (
    "model.dropout", "training.weight_decay", "training.input_noise_std",
    "inputs.stride", "training.patience",
)


def regularization_variants(config: dict) -> dict[str, dict]:
    """Expand one-factor changes, skipping duplicates and the baseline value."""
    baseline = copy.deepcopy(config)
    study = baseline.pop("study")
    variants = {"baseline": baseline}
    for factor, values in study["variations"].items():
        if factor not in FACTORS:
            raise ValueError(f"unknown regularization factor: {factor}")
        if not isinstance(values, list):
            raise ValueError(f"values for {factor} must be a list")
        section, key = factor.split(".")
        seen = [baseline[section][key]]
        for value in values:
            if value not in seen:
                variants[f"{key}_{value}"] = apply_overrides(baseline, [f"{factor}={value}"])
                seen.append(value)
    return variants


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _context(train: pd.DataFrame) -> dict:
    package = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        source.update(str(path.relative_to(package)).encode())
        source.update(path.read_bytes())
    return {
        "training_data_sha256": _sha256(train.to_csv(index=False).encode()),
        "source_sha256": source.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {name: version(name) for name in (
            "torch", "numpy", "pandas", "scikit-learn", "scipy",
        )},
        "torch_threads": torch.get_num_threads(),
    }


def run_regularization(
    train: pd.DataFrame, config: dict, cache_dir: str | Path,
) -> tuple[list[dict], pd.DataFrame]:
    """Run each setting on the same engine folds, with capped and uncapped RUL.

    Cache hits require identical data, source, environment and resolved config,
    plus an intact predictions CSV. Each completed run is saved immediately,
    so an interrupted notebook resumes instead of repeating the whole grid.
    """
    variants = regularization_variants(config)
    if config["target"]["cap"] is None or not config["target"]["cap"] > 0:
        raise ValueError("the study baseline needs a positive training cap")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    context = _context(train)
    records, predictions = [], []
    for name, variant in variants.items():
        for target in ("capped", "uncapped"):
            resolved = copy.deepcopy(variant)
            if target == "uncapped":
                resolved["target"]["cap"] = None
            record_path = cache_dir / f"{name}_{target}.json"
            prediction_path = cache_dir / f"{name}_{target}_predictions.csv"
            record = None
            if record_path.exists() and prediction_path.exists():
                try:
                    candidate = json.loads(record_path.read_text())
                    if (candidate["context"] == context and candidate["config"] == resolved
                            and candidate["predictions_sha256"] == _sha256(prediction_path.read_bytes())):
                        record = candidate
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass  # A partial or malformed completion marker is a cache miss.
            if record is None:
                cv = run_cv(resolved, train)
                cv.predictions.to_csv(prediction_path, index=False)
                record = {
                    "variant": name, "target": target, "config": resolved,
                    "context": context,
                    "predictions_sha256": _sha256(prediction_path.read_bytes()),
                    **cv.to_dict(),
                }
                # JSON is the completion marker; only write it after the CSV.
                temporary = record_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                temporary.replace(record_path)
            records.append(record)
            predictions.append(pd.read_csv(prediction_path).assign(variant=name, target=target))
            score = record["summary"]["selection_rmse"]
            print(f"{name}, {target}: {score['mean']:.3f} ± {score['std']:.3f}", flush=True)
    return records, pd.concat(predictions, ignore_index=True)


def regularization_table(records: list[dict]) -> pd.DataFrame:
    """Pair capped/uncapped scores, ranked by mean capped selection RMSE."""
    paired = {}
    for record in records:
        name, target = record["variant"], record["target"]
        row = paired.setdefault(name, {"variant": name})
        summary = record["summary"]
        for metric in ("selection_rmse", "rmse", "mae", "nasa_score"):
            row[f"{target}_{metric}"] = summary[metric]["mean"]
        row[f"{target}_selection_std"] = summary["selection_rmse"]["std"]
        row[f"{target}_best_epochs"] = [fold["best_epoch"] for fold in record["folds"]]
        row[f"{target}_seconds"] = record["seconds"]
    table = pd.DataFrame(paired.values())
    baseline = table.loc[table.variant == "baseline", "capped_selection_rmse"].item()
    table["delta_from_baseline"] = table["capped_selection_rmse"] - baseline
    return table.sort_values(["capped_selection_rmse", "variant"]).reset_index(drop=True)


def plot_regularization(records: list[dict]) -> plt.Figure:
    """Learning losses for all folds of the best/worst capped settings.

    Training is online MSE with dropout/noise; validation is the clean inner
    engine holdout, against capped labels. The outer selection score chooses
    the best/worst rows, and is not used for early stopping.
    """
    capped = {r["variant"]: r for r in records if r["target"] == "capped"}
    table = regularization_table(records)
    extremes = [("Best", table.iloc[0]), ("Worst", table.iloc[-1])]
    n_folds = len(next(iter(capped.values()))["folds"])
    fig, axes = plt.subplots(2, n_folds, figsize=(12, 6), squeeze=False, sharey=True)
    for row, (label, setting) in enumerate(extremes):
        record = capped[setting["variant"]]
        scale = record["config"]["target"]["cap"]
        for ax, fold in zip(axes[row], record["folds"]):
            history = pd.DataFrame(fold["history"])
            ax.plot(history.epoch, history.train_loss, label="training (dropout/noise on)")
            ax.plot(history.epoch, (history.val_rmse / scale) ** 2,
                    label="inner holdout (clean)")
            ax.axvline(fold["best_epoch"], color="grey", ls=":", label="best epoch")
            ax.set(title=f"{label}: {setting['variant']} · fold {fold['fold']}",
                   xlabel="epoch", yscale="log")
        axes[row, 0].set_ylabel(f"Scaled MSE (RUL / {scale:g})²")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("LSTM regularization: inner-holdout learning curves (training engines only)")
    fig.tight_layout()
    return fig
