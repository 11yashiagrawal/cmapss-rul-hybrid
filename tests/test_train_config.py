"""Tests for config loading, the model registry and config-driven runs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn

from rul.data.load import SENSOR_COLS, SETTING_COLS
from rul.models.base import RegressionHead, SequenceEncoder, SequenceRegressor
from rul.models.registry import MODELS, build_model, register_model
from rul.training.config import apply_overrides, load_config, run_cv, run_parts
from rul.training.inputs import WindowInputs
from rul.training.trainer import TrainConfig

ROOT = Path(__file__).resolve().parents[1]


class LastStepEncoder(SequenceEncoder):
    """Stand-in encoder: a linear layer on the last time step."""

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


def write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_child_config_overrides_its_base(tmp_path):
    write(tmp_path / "base.yaml", {"subset": "FD001", "training": {"epochs": 100, "learning_rate": 0.001}})
    child = write(tmp_path / "child.yaml", {"base": "base.yaml", "name": "x", "training": {"epochs": 5}})
    config = load_config(child)
    assert config == {"subset": "FD001", "name": "x", "training": {"epochs": 5, "learning_rate": 0.001}}


def test_overrides_parse_yaml_values():
    config = apply_overrides(
        {"training": {"grad_clip": 1.0}, "inputs": {"window": 30}},
        ["training.grad_clip=null", "inputs.window=40", "target.cap=125"],
    )
    assert config == {"training": {"grad_clip": None}, "inputs": {"window": 40}, "target": {"cap": 125}}
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["training.epochs"])


def test_base_config_is_valid():
    config = load_config(ROOT / "configs" / "dl_base.yaml")
    TrainConfig.from_dict(config["training"])
    assert config["cv"] == {"n_splits": 3, "split_seed": 42, "holdout_size": 0.2}
    assert config["training"]["device"] == "cpu"


def test_build_model_wraps_an_encoder_with_the_configured_head():
    model = build_model(
        {"type": "last_step", "output_dim": 6, "head": {"hidden_dim": 8, "dropout": 0.1}},
        {"n_features": 3},
    )
    assert isinstance(model, SequenceRegressor)
    assert model.encoder.output_dim == 6
    assert isinstance(model.head, RegressionHead) and model.head.input_dim == 6
    assert model(torch.randn(2, 5, 3)).shape == (2,)


def test_build_model_rejects_unknown_types():
    with pytest.raises(KeyError, match="last_step"):
        build_model({"type": "nope"}, {"n_features": 3})


def test_run_parts_follow_the_config():
    config = {
        "target": {"cap": 125},
        "selection": {"max_true_rul": 130},
        "cv": {"n_splits": 3, "split_seed": 42, "holdout_size": 0.2},
        "inputs": {"window": 20, "stride": 2},
        "training": {"epochs": 7},
        "model": {"type": "last_step"},
    }
    parts = run_parts(config)
    assert parts.train_config == TrainConfig(epochs=7)
    assert parts.cv_kwargs == {
        "cap": 125, "selection_max_rul": 130, "n_splits": 3, "split_seed": 42, "holdout_size": 0.2,
    }
    inputs = parts.make_inputs()
    assert isinstance(inputs, WindowInputs) and (inputs.window, inputs.stride) == (20, 2)


def engines(n_units: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(30, 50))
        for cycle in range(1, life + 1):
            row = {"unit": unit, "cycle": cycle, **dict.fromkeys(SETTING_COLS, 0.0)}
            row.update(dict.fromkeys(SENSOR_COLS, 5.0))
            row["s_2"] = 600 + 0.2 * cycle + rng.normal(0, 0.3)
            rows.append(row)
    return pd.DataFrame(rows)


def test_run_cv_from_a_config():
    config = {
        "target": {"cap": 30},
        "selection": {"max_true_rul": 25},
        "cv": {"n_splits": 3, "split_seed": 42, "holdout_size": 0.25},
        "inputs": {"window": 5},
        "training": {"epochs": 2, "batch_size": 32, "seed": 0},
        "model": {"type": "last_step"},
    }
    cv = run_cv(config, engines())
    assert len(cv.folds) == 3
    assert cv.folds[0].scores["max_true_rul"] == 25
