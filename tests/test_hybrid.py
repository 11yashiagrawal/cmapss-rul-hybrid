"""Tests for the hybrid models."""

import numpy as np
import pandas as pd
import pytest
import torch

from rul.data.labels import add_train_rul
from rul.data.load import SENSOR_COLS, SETTING_COLS
from rul.models.hybrid import FUSIONS, GatedHybrid
from rul.models.lstm import LSTMEncoder
from rul.models.registry import build_model
from rul.training.config import run_parts
from rul.training.inputs import HybridInputs

BATCH, WINDOW, FEATURES, ENGINEERED = 4, 30, 15, 20


@pytest.fixture
def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(BATCH, WINDOW, FEATURES), torch.randn(BATCH, ENGINEERED)


def hybrid(fusion: str = "gated") -> GatedHybrid:
    return GatedHybrid(LSTMEncoder(FEATURES, hidden_size=16), ENGINEERED, fusion=fusion)


@pytest.mark.parametrize("fusion", FUSIONS)
def test_every_fusion_returns_one_value_per_window(inputs, fusion):
    assert hybrid(fusion)(*inputs).shape == (BATCH,)


def test_gate_stays_between_zero_and_one(inputs):
    model = hybrid().eval()
    gate = model.gate(*inputs)
    assert gate.shape == (BATCH, 16)
    assert ((gate >= 0) & (gate <= 1)).all()


def test_gated_output_mixes_the_two_branches(inputs):
    model = hybrid().eval()
    h_seq, h_feat = model.branches(*inputs)
    g = model.gate(*inputs)
    expected = model.head(g * h_seq + (1 - g) * h_feat)
    torch.testing.assert_close(model(*inputs), expected)


@pytest.mark.parametrize("fusion", ["gated", "concat"])
def test_gradients_reach_both_branches(inputs, fusion):
    model = hybrid(fusion)
    model(*inputs).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert any(n.startswith("encoder.") for n, _ in model.named_parameters())
    assert any(n.startswith("feature_net.") for n, _ in model.named_parameters())


def test_switching_off_a_branch_ignores_its_input(inputs):
    x_seq, x_feat = inputs
    seq_only, feat_only = hybrid("sequence").eval(), hybrid("features").eval()
    torch.testing.assert_close(seq_only(x_seq, x_feat), seq_only(x_seq, x_feat + 10))
    torch.testing.assert_close(feat_only(x_seq, x_feat), feat_only(x_seq + 10, x_feat))
    assert not any(n.startswith("feature_net.") for n, _ in seq_only.named_parameters())
    assert not any(n.startswith("encoder.") for n, _ in feat_only.named_parameters())
    with pytest.raises(ValueError, match="gate"):
        seq_only.gate(x_seq, x_feat)


def test_config_builds_a_gated_hybrid(inputs):
    model = build_model(
        {"type": "hybrid_gated", "encoder": {"type": "lstm", "hidden_size": 16}, "fusion": "concat",
         "feature_hidden": 32, "head": {"hidden_dim": 8}},
        {"n_features": FEATURES, "n_engineered": ENGINEERED},
    )
    assert isinstance(model, GatedHybrid) and model.fusion == "concat"
    assert model.head.input_dim == 32  # concatenation doubles the embedding
    assert model(*inputs).shape == (BATCH,)


def test_rejects_unknown_fusion():
    with pytest.raises(ValueError, match="fusion"):
        hybrid("sum")


def engines(n_units: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, int(rng.integers(30, 45)) + 1):
            row = {"unit": unit, "cycle": cycle, **dict.fromkeys(SETTING_COLS, 0.0)}
            row.update(dict.fromkeys(SENSOR_COLS, 5.0))
            row["s_2"] = 600 + 0.2 * cycle + rng.normal(0, 0.3)
            row["s_3"] = 1500 - 0.3 * cycle + rng.normal(0, 0.3)
            rows.append(row)
    return add_train_rul(pd.DataFrame(rows))


def test_hybrid_inputs_line_features_up_with_window_ends():
    frame = engines()
    inputs = HybridInputs(window=5, feature_windows=(2, 4)).fit(frame)
    batch = inputs.transform(frame, target="rul")
    windows, features = batch.inputs
    assert windows.shape == (len(frame), 5, 2)
    assert features.dtype == np.float32 and features.flags.writeable  # torch warns on read-only arrays
    assert features.shape == (len(frame), inputs.input_dims["n_engineered"])
    assert inputs.input_dims == {"n_features": 2, "n_engineered": 1 + 2 + 2 * 2 * 3}
    expected = inputs.features_.transform(frame).set_index(["unit", "cycle"])
    keys = list(zip(batch.index["unit"], batch.index["cycle"]))
    np.testing.assert_allclose(features, expected.loc[keys].to_numpy(), rtol=1e-6)


def test_hybrid_inputs_from_config_keep_feature_groups():
    parts = run_parts({"inputs": {"type": "hybrid", "window": 5, "groups": ["age", "sensors"],
                                  "feature_windows": [2]},
                       "model": {"type": "hybrid_gated", "encoder": {"type": "lstm"}}})
    inputs = parts.make_inputs().fit(engines())
    assert inputs.features_.output_columns_ == ["age", "s_2", "s_3"]
