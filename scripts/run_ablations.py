"""Run every ablation variant in a YAML file through the shared trainer.

Run from the repository root after ``scripts/download_data.py``:

    python scripts/run_ablations.py                          # every variant in configs/ablations.yaml
    python scripts/run_ablations.py --only window cap_none   # one group and one variant
    python scripts/run_ablations.py --smoke                  # one fold, two epochs (a quick check)

Each variant is cross-validated on the Phase 1 folds (``run_cv``) and cached as
soon as it finishes under ``results/models/ablations/`` (gitignored), keyed by a
hash of its resolved config. A restart skips every variant whose cache exists,
and variants that resolve to the same config are trained once.

After each variant, the summary ``results/metrics/<subset>_ablations.csv`` is
rewritten: one row per variant with its settings, the selection RMSE of every
fold, their mean and standard deviation, and the selection-band RMSE by life
stage. Smoke runs write ``<subset>_ablations_smoke.csv`` instead.

This script never reads the official test set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rul.data.load import load_subset
from rul.evaluation.official import METRICS_DIR
from rul.training.config import apply_overrides, load_config, run_cv
from rul.training.runs import save_cv

ROOT = Path(__file__).resolve().parents[1]
SMOKE = {"training.epochs": 2, "cv.only_folds": [1]}
STAGES = ("late", "mid", "early")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=ROOT / "configs" / "ablations.yaml",
                        help="ablation file with `base` and `groups`")
    parser.add_argument("--only", nargs="*", default=None, metavar="NAME",
                        help="run only these groups or variants")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override applied to every variant, e.g. training.seed=43")
    parser.add_argument("--smoke", action="store_true", help="one fold, two epochs")
    parser.add_argument("--data-dir", default=ROOT / "data" / "raw" / "CMAPSSData")
    parser.add_argument("--cache-dir", default=ROOT / "results" / "models" / "ablations")
    parser.add_argument("--metrics-dir", default=METRICS_DIR)
    return parser.parse_args(argv)


def set_values(config: dict, overrides: dict) -> dict:
    """Apply ``{"section.key": value}`` overrides to a copy of ``config``."""
    config = copy.deepcopy(config)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = copy.deepcopy(value)
    return config


def load_variants(path: str | Path) -> tuple[dict, list[dict]]:
    """Read the ablation file: the base config and one entry per variant."""
    path = Path(path)
    spec = yaml.safe_load(path.read_text()) or {}
    if "base" not in spec or not spec.get("groups"):
        raise ValueError(f"{path} needs `base` and `groups`")
    base = load_config(path.parent / spec["base"])
    variants, seen = [], set()
    for group, members in spec["groups"].items():
        for name, overrides in (members or {}).items():
            if name in seen:
                raise ValueError(f"variant name {name!r} is used twice")
            seen.add(name)
            variants.append({"group": group, "variant": name, "overrides": overrides or {}})
    return base, variants


def config_key(config: dict) -> str:
    """Short, stable hash of a resolved config (its cache key)."""
    text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def stage_rmse(predictions: pd.DataFrame, max_true_rul: float | None) -> dict:
    """Selection-band RMSE by true-RUL stage: late <= 50 < mid <= 100 < early."""
    band = predictions if max_true_rul is None else predictions[predictions["rul"] <= max_true_rul]
    stage = pd.cut(band["rul"], [-np.inf, 50, 100, np.inf], labels=list(STAGES))
    err = ((band["prediction"] - band["rul"]) ** 2).groupby(stage, observed=False).mean() ** 0.5
    return {f"{s}_rmse": (None if pd.isna(err[s]) else float(err[s])) for s in STAGES}


def run_variant(config: dict, train: pd.DataFrame, cache_dir: Path) -> tuple[dict, pd.DataFrame, bool]:
    """Cross-validate ``config``, or load it from the cache. Returns (record, predictions, trained)."""
    key = config_key(config)
    path = cache_dir / f"{key}.json"
    trained = not path.exists()
    if trained:
        cv = run_cv(config, train)
        save_cv(key, cv, meta={"config": config, "key": key}, out_dir=cache_dir)
    record = json.loads(path.read_text())
    return record, pd.read_csv(cache_dir / f"{key}_predictions.csv"), trained


def summary_row(entry: dict, record: dict, predictions: pd.DataFrame, max_true_rul) -> dict:
    """One results row: variant, settings, fold scores, mean and spread, stages."""
    folds = {f"fold_{f['fold']}": f["scores"]["selection_rmse"] for f in record["folds"]}
    summary = record["summary"]
    return {
        "group": entry["group"],
        "variant": entry["variant"],
        "settings": json.dumps(entry["overrides"], sort_keys=True),
        "key": record["meta"]["key"],
        **folds,
        "selection_rmse_mean": summary["selection_rmse"]["mean"],
        "selection_rmse_std": summary["selection_rmse"]["std"],
        **stage_rmse(predictions, max_true_rul),
        "all_cycle_rmse_mean": summary["rmse"]["mean"],
        "median_best_epoch": record["median_best_epoch"],
        "seconds": record["seconds"],
    }


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Run the selected variants and write the summary CSV; returns the summary."""
    args = parse_args(argv)
    base, variants = load_variants(args.config)
    if args.only is not None:
        wanted = set(args.only)
        unknown = wanted - {v["group"] for v in variants} - {v["variant"] for v in variants}
        if unknown:
            raise ValueError(f"unknown groups or variants: {sorted(unknown)}")
        variants = [v for v in variants if v["group"] in wanted or v["variant"] in wanted]

    common = apply_overrides(base, args.set)
    subset = common.get("subset", "FD001")
    train = load_subset(subset, data_dir=args.data_dir).train
    cache_dir = Path(args.cache_dir)
    out = Path(args.metrics_dir) / f"{subset.lower()}_ablations{'_smoke' if args.smoke else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, entry in enumerate(variants, start=1):
        config = set_values(common, entry["overrides"])
        if args.smoke:
            config = set_values(config, SMOKE)
        record, predictions, trained = run_variant(config, train, cache_dir)
        max_rul = (config.get("selection") or {}).get("max_true_rul")
        rows.append(summary_row(entry, record, predictions, max_rul))
        summary = pd.DataFrame(rows)
        summary.to_csv(out, index=False)  # rewritten after every variant
        status = "trained" if trained else "cached"
        print(f"[{i}/{len(variants)}] {entry['variant']}: selection RMSE "
              f"{rows[-1]['selection_rmse_mean']:.2f} ± {rows[-1]['selection_rmse_std']:.2f} ({status})")
    print(f"saved {out}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
