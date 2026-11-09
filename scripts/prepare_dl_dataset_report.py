"""Reproduce issue #74's tables from training cycles and the frozen #66 CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn

from rul.data.load import COLUMNS
from rul.data.split import group_kfold_splits, holdout_split
from rul.features.cleaning import select_sensors
from rul.features.windows import make_windows

ROOT = Path(__file__).resolve().parents[1]
STUDY_COMMIT = "598ecb8dce4d101d118b23dd6ef6d2a1790f2615"
STUDY_FILES = {
    "results/metrics/fd001_regularization.csv":
        "54a5e2b578afbf1a64213141fdd96fbedfa7fbe9dead9f21f3f65dc0f138d8e0",
    "results/figures/fd001_regularization.png":
        "fe24983c99bf912d0d9a17468f1ade0827556f3d0a7dfa6d9a52acc2287b0dc1",
}
LABELS = {
    "baseline": r"\textbf{Baseline}",
    "dropout_0.1": r"Dropout $p=0.1$",
    "dropout_0.5": r"Dropout $p=0.5$",
    "weight_decay_0.0001": r"Weight decay $10^{-4}$",
    "weight_decay_0.001": r"Weight decay $10^{-3}$",
    "input_noise_std_0.01": r"Noise $\sigma=0.01$",
    "input_noise_std_0.05": r"Noise $\sigma=0.05$",
    "stride_2": r"Stride $s=2$",
    "stride_5": r"Stride $s=5$",
    "patience_5": r"Patience 5",
    "patience_20": r"Patience 20",
}


def count_windows(frame: pd.DataFrame, stride: int) -> int:
    """Check the padded count formula against the actual window builder."""
    expected = int(np.ceil(frame.groupby("unit").size() / stride).sum())
    actual = len(make_windows(frame, ["s_2"], window=30, stride=stride,
                              target_column=None, pad=True).X)
    if actual != expected:
        raise ValueError("window construction changed; review the report's count formula")
    return actual


def prepare_tables() -> dict[str, str]:
    """Return generated table/data/provenance files; never read official test data."""
    for name, expected in STUDY_FILES.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} differs from the recorded #66 snapshot")
    source = ROOT / "data/raw/CMAPSSData/train_FD001.txt"
    train = pd.read_csv(source, sep=r"\s+", header=None, names=COLUMNS)
    if len(train) != 20631 or train.unit.nunique() != 100:
        raise ValueError("expected the 100-engine, 20,631-cycle FD001 training file")
    counts, folds = [], []

    def add_partition(label: str, part: pd.DataFrame, fitting: bool) -> None:
        counts.append({
            "partition": label, "engines": part.unit.nunique(),
            "stride_1": count_windows(part, 1),
            "stride_2": count_windows(part, 2) if fitting else None,
            "stride_5": count_windows(part, 5) if fitting else None,
        })

    add_partition("All training", train, True)
    for k, (train_idx, val_idx) in enumerate(group_kfold_splits(train, 3, 42), 1):
        fold = train.iloc[train_idx]
        fit_idx, hold_idx = holdout_split(fold, 0.2, 42)
        fit, hold, scored = fold.iloc[fit_idx], fold.iloc[hold_idx], train.iloc[val_idx]
        units = [sorted(int(u) for u in part.unit.unique()) for part in (fit, hold, scored)]
        if any(set(units[a]) & set(units[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("engine leakage in the reported folds")
        sensors = select_sensors(fit)
        if len(sensors) != 15:
            raise ValueError("sensor selection changed; review the report")
        folds.append({"fold": k, "fit_units": units[0], "early_stop_units": units[1],
                      "scored_units": units[2], "sensors": sensors})
        for label, part, fitting in (("fit", fit, True), ("ES", hold, False),
                                     ("CV", scored, False)):
            add_partition(f"Fold {k} {label}", part, fitting)

    frame = pd.DataFrame(counts).astype({c: "Int64" for c in (
        "engines", "stride_1", "stride_2", "stride_5",
    )})
    window_rows = [r"\begin{tabular}{@{}lrrrr@{}}", r"\hline",
                   r"Partition & Engines & $s=1$ & $s=2$ & $s=5$ \\", r"\hline"]
    for row in frame.to_dict("records"):
        values = ["--" if pd.isna(row[c]) else f"{int(row[c]):,}" for c in (
            "engines", "stride_1", "stride_2", "stride_5",
        )]
        window_rows.append(row["partition"] + " & " + " & ".join(values) + r" \\")
    window_rows += [r"\hline", r"\end{tabular}"]

    study = pd.read_csv(ROOT / "results/metrics/fd001_regularization.csv").set_index("variant")
    if not study.index.is_unique or set(study.index) != set(LABELS):
        raise ValueError("the study grid changed; review the report's result labels")
    result_rows = [r"\begin{tabular}{@{}lcc@{}}", r"\hline",
                   r"Setting & Capped & Uncapped \\", r"\hline"]
    for name, label in LABELS.items():
        row = study.loc[name]
        scores = [f"${row[f'{t}_selection_rmse']:.2f} \\pm "
                  f"{row[f'{t}_selection_std']:.2f}$" for t in ("capped", "uncapped")]
        result_rows.append(label + " & " + " & ".join(scores) + r" \\")
    result_rows += [r"\hline", r"\end{tabular}"]
    manifest = {
        "issue": 74, "subset": "FD001", "window": 30, "padding": "repeat first cycle",
        "split_seed": 42, "training_seed": 42, "inner_holdout_fraction": 0.2,
        "training_file": source.name,
        "training_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "study_commit": STUDY_COMMIT, "study_file_sha256": STUDY_FILES,
        "selection_status": "provisional until #55: training cap 125, true RUL <= 130",
        "official_test_used": False,
        "all_training_unpadded_stride_1": int((train.groupby("unit").size() - 29).clip(lower=0).sum()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "folds": folds,
    }
    return {
        "reports/phase2/tables/dl_window_counts.tex": "\n".join(window_rows) + "\n",
        "reports/phase2/tables/dl_regularization.tex": "\n".join(result_rows) + "\n",
        "reports/phase2/data/dl_window_counts.csv": frame.to_csv(index=False),
        "reports/phase2/data/dl_dataset_provenance.json": json.dumps(manifest, indent=2) + "\n",
    }


def main() -> None:
    """Regenerate report tables, or verify committed files with ``--check``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check without writing files")
    args = parser.parse_args()
    for name, content in prepare_tables().items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise ValueError(f"{name} is stale; regenerate the report tables")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print("Verified report tables, window counts and #66 asset checksums."
          if args.check else "Generated report tables and training-only provenance.")


if __name__ == "__main__":
    main()
