"""Smoke-mode tests of scripts/run_ablations.py on a tiny synthetic subset."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn

from rul.data.load import COLUMNS, SENSOR_COLS, SETTING_COLS
from rul.models.base import SequenceEncoder
from rul.models.registry import MODELS, register_model
from rul.training.config import run_parts

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ablations.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_ablations", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LastStepEncoder(SequenceEncoder):
    def __init__(self, n_features: int, output_dim: int = 4) -> None:
        super().__init__(output_dim)
        self.proj = nn.Linear(n_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x[:, -1, :])


@pytest.fixture(autouse=True)
def stand_in_model():
    register_model("last_step", LastStepEncoder)
    yield
    MODELS.pop("last_step", None)


def write_subset(folder: Path, n_units: int = 6, seed: int = 0) -> None:
    """train/test/RUL files in the raw C-MAPSS format."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, int(rng.integers(30, 45)) + 1):
            row = {"unit": unit, "cycle": cycle, **dict.fromkeys(SETTING_COLS, 0.0)}
            row.update(dict.fromkeys(SENSOR_COLS, 5.0))
            row["s_2"] = 600 + 0.2 * cycle + rng.normal(0, 0.3)
            rows.append(row)
    frame = pd.DataFrame(rows)[COLUMNS]
    folder.mkdir(parents=True)
    frame.to_csv(folder / "train_FD001.txt", sep=" ", header=False, index=False)
    frame[frame["unit"] <= 2].to_csv(folder / "test_FD001.txt", sep=" ", header=False, index=False)
    (folder / "RUL_FD001.txt").write_text("10\n20\n")


@pytest.fixture
def setup(tmp_path):
    write_subset(tmp_path / "data")
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({
        "subset": "FD001",
        "name": "stand_in",
        "target": {"cap": 30},
        "selection": {"max_true_rul": 25},
        "cv": {"n_splits": 3, "split_seed": 42, "holdout_size": 0.25},
        "inputs": {"window": 5},
        "training": {"epochs": 20, "batch_size": 32, "seed": 0},
        "model": {"type": "last_step"},
    }))
    ablations = tmp_path / "ablations.yaml"
    ablations.write_text(yaml.safe_dump({
        "base": "base.yaml",
        "groups": {
            "window": {"window_3": {"inputs.window": 3}, "window_5": {}},
            "cap": {"cap_none": {"target.cap": None}, "cap_30": {}},
        },
    }, sort_keys=False))
    args = [
        "--config", str(ablations), "--smoke",
        "--data-dir", str(tmp_path / "data"),
        "--cache-dir", str(tmp_path / "cache"),
        "--metrics-dir", str(tmp_path / "metrics"),
    ]
    return tmp_path, args


def test_smoke_run_writes_one_row_per_variant_with_one_fold(setup):
    tmp_path, args = setup
    summary = load_script().main(args)

    saved = pd.read_csv(tmp_path / "metrics" / "fd001_ablations_smoke.csv")
    assert saved["variant"].tolist() == ["window_3", "window_5", "cap_none", "cap_30"]
    assert saved["group"].tolist() == ["window", "window", "cap", "cap"]
    assert [c for c in saved.columns if c.startswith("fold_")] == ["fold_1"]
    assert saved["selection_rmse_mean"].equals(saved["fold_1"])
    assert json.loads(saved.loc[0, "settings"]) == {"inputs.window": 3}
    assert {"late_rmse", "mid_rmse", "early_rmse"} <= set(saved.columns)
    assert len(summary) == 4


def test_smoke_mode_trains_two_epochs_on_one_fold(setup):
    tmp_path, args = setup
    load_script().main(args + ["--only", "window_3"])
    record = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert [f["fold"] for f in record["folds"]] == [1]
    assert len(record["folds"][0]["history"]) <= 2
    assert record["meta"]["config"]["inputs"]["window"] == 3


def test_identical_configs_train_once_and_restarts_skip_finished_variants(setup, monkeypatch):
    tmp_path, args = setup
    script = load_script()
    calls = []
    real_run_cv = script.run_cv
    monkeypatch.setattr(script, "run_cv", lambda c, t: calls.append(c) or real_run_cv(c, t))

    script.main(args)
    assert len(calls) == 3  # window_5 and cap_30 are both the base setting

    script.main(args)
    assert len(calls) == 3  # everything cached


def test_changed_settings_are_trained_again(setup):
    tmp_path, args = setup
    script = load_script()
    script.main(args + ["--only", "window_3"])
    script.main(args + ["--only", "window_3", "--set", "training.seed=1"])
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2


def test_only_selects_groups_and_rejects_unknown_names(setup):
    tmp_path, args = setup
    script = load_script()
    assert script.main(args + ["--only", "cap"])["variant"].tolist() == ["cap_none", "cap_30"]
    with pytest.raises(ValueError, match="unknown"):
        script.main(args + ["--only", "window_99"])


def test_repeated_variant_names_are_rejected(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"model": {"type": "last_step"}}))
    path = tmp_path / "ablations.yaml"
    path.write_text(yaml.safe_dump({"base": "base.yaml", "groups": {"a": {"v": {}}, "b": {"v": {}}}}))
    with pytest.raises(ValueError, match="twice"):
        load_script().load_variants(path)


def test_project_ablation_file_resolves_to_valid_runs():
    script = load_script()
    base, variants = script.load_variants(ROOT / "configs" / "ablations.yaml")
    assert {"window", "cap", "features", "cycle", "branches"} <= {v["group"] for v in variants}
    for entry in variants:
        parts = run_parts(script.set_values(base, entry["overrides"]))
        parts.make_inputs()  # unknown input settings raise here
