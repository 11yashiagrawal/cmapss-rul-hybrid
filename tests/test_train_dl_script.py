"""End-to-end test of scripts/train_dl.py on a tiny synthetic subset."""

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
from rul.training.runs import TrainedModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_dl.py"


def load_script():
    spec = importlib.util.spec_from_file_location("train_dl", SCRIPT)
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
    """train/test/RUL files in the raw C-MAPSS format; the test file reuses engines 1 and 2."""
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


def test_script_cross_validates_and_refits(tmp_path):
    write_subset(tmp_path / "data")
    config = tmp_path / "stand_in.yaml"
    config.write_text(yaml.safe_dump({
        "subset": "FD001",
        "name": "stand_in",
        "target": {"cap": 30},
        "selection": {"max_true_rul": 25},
        "cv": {"n_splits": 3, "split_seed": 42, "holdout_size": 0.25},
        "inputs": {"window": 5},
        "training": {"epochs": 3, "batch_size": 32, "seed": 0},
        "model": {"type": "last_step"},
    }))
    load_script().main([
        "--config", str(config), "--set", "training.epochs=2", "--refit",
        "--data-dir", str(tmp_path / "data"),
        "--metrics-dir", str(tmp_path / "metrics"),
        "--models-dir", str(tmp_path / "models"),
    ])

    record = json.loads((tmp_path / "metrics" / "fd001_cv_stand_in.json").read_text())
    assert record["meta"]["config"]["training"]["epochs"] == 2
    assert len(record["folds"]) == 3
    assert (tmp_path / "metrics" / "fd001_cv_stand_in_predictions.csv").is_file()

    checkpoint = tmp_path / "models" / "fd001_stand_in_seed0.pt"
    trained = TrainedModel.load(checkpoint)
    assert trained.meta["epochs"] == record["median_best_epoch"]
    assert sorted(trained.train_units) == [1, 2, 3, 4, 5, 6]


def test_seed_flag_changes_the_run_name(tmp_path):
    write_subset(tmp_path / "data")
    config = tmp_path / "stand_in.yaml"
    config.write_text(yaml.safe_dump({
        "name": "stand_in",
        "target": {"cap": 30},
        "cv": {"n_splits": 3, "split_seed": 42, "holdout_size": 0.25},
        "inputs": {"window": 5},
        "training": {"epochs": 1, "batch_size": 32},
        "model": {"type": "last_step"},
    }))
    load_script().main([
        "--config", str(config), "--seed", "7",
        "--data-dir", str(tmp_path / "data"), "--metrics-dir", str(tmp_path / "metrics"),
    ])
    assert (tmp_path / "metrics" / "fd001_cv_stand_in_seed7.json").is_file()
