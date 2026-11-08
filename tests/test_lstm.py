"""Tests for the LSTM encoder."""

import pytest
import torch

from rul.models.base import SequenceEncoder, SequenceRegressor
from rul.models.lstm import LSTMEncoder
from rul.models.registry import build_model

BATCH, WINDOW, FEATURES = 4, 30, 15


@pytest.fixture
def windows() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, WINDOW, FEATURES)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_encoder_returns_one_embedding_per_window(windows, num_layers):
    encoder = LSTMEncoder(FEATURES, hidden_size=32, num_layers=num_layers)
    assert isinstance(encoder, SequenceEncoder)
    assert encoder.output_dim == 32
    assert encoder(windows).shape == (BATCH, 32)


def test_config_builds_a_regressor(windows):
    model = build_model({"type": "lstm", "hidden_size": 16, "num_layers": 2, "dropout": 0.2},
                        {"n_features": FEATURES})
    assert isinstance(model, SequenceRegressor) and isinstance(model.encoder, LSTMEncoder)
    assert model(windows).shape == (BATCH,)


def test_backward_pass_reaches_every_parameter(windows):
    model = SequenceRegressor(LSTMEncoder(FEATURES, hidden_size=16, num_layers=2, dropout=0.1))
    model(windows).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_order_of_cycles_matters(windows):
    encoder = LSTMEncoder(FEATURES, hidden_size=16).eval()
    reversed_time = windows.flip(dims=[1])
    assert not torch.allclose(encoder(windows), encoder(reversed_time))


def test_embedding_comes_from_the_last_cycle_after_reading_the_whole_window(windows):
    encoder = LSTMEncoder(FEATURES, hidden_size=16).eval()
    changed_first = windows.clone()
    changed_first[:, 0] += 5
    assert not torch.allclose(encoder(windows), encoder(changed_first))


def test_rejects_bad_settings():
    with pytest.raises(ValueError):
        LSTMEncoder(FEATURES, hidden_size=16, dropout=1.0)
    with pytest.raises(ValueError):
        LSTMEncoder(FEATURES, hidden_size=16, num_layers=0)
