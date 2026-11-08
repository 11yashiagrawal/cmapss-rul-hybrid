"""Tests for the Transformer encoder."""

import pytest
import torch

from rul.models.base import SequenceEncoder, SequenceRegressor
from rul.models.registry import build_model
from rul.models.transformer import TransformerEncoder

BATCH, WINDOW, FEATURES = 4, 30, 15


@pytest.fixture
def windows() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, WINDOW, FEATURES)


@pytest.mark.parametrize("pooling", ["mean", "last"])
def test_encoder_returns_one_embedding_per_window(windows, pooling):
    encoder = TransformerEncoder(FEATURES, d_model=16, n_heads=2, num_layers=2, pooling=pooling)
    assert isinstance(encoder, SequenceEncoder)
    assert encoder.output_dim == 16
    assert encoder(windows).shape == (BATCH, 16)


def test_config_builds_a_regressor(windows):
    model = build_model({"type": "transformer", "d_model": 16, "n_heads": 4, "num_layers": 1},
                        {"n_features": FEATURES})
    assert isinstance(model, SequenceRegressor) and isinstance(model.encoder, TransformerEncoder)
    assert model(windows).shape == (BATCH,)


def test_backward_pass_reaches_every_parameter(windows):
    model = SequenceRegressor(TransformerEncoder(FEATURES, d_model=16, n_heads=2, num_layers=2))
    model(windows).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_positional_encoding_makes_order_matter(windows):
    encoder = TransformerEncoder(FEATURES, d_model=16, n_heads=2, num_layers=1, pooling="mean").eval()
    shuffled = windows[:, torch.randperm(WINDOW)]
    assert not torch.allclose(encoder(windows), encoder(shuffled), atol=1e-5)


def test_attention_maps_are_available_for_every_layer(windows):
    encoder = TransformerEncoder(FEATURES, d_model=16, n_heads=4, num_layers=2).eval()
    before = encoder(windows)
    maps = encoder.attention(windows)
    assert len(maps) == 2
    for attn in maps:
        assert attn.shape == (BATCH, 4, WINDOW, WINDOW)
        torch.testing.assert_close(attn.sum(dim=-1), torch.ones(BATCH, 4, WINDOW))
    # reading the maps doesn't change the model
    torch.testing.assert_close(encoder(windows), before)


def test_rejects_bad_settings():
    with pytest.raises(ValueError, match="divisible"):
        TransformerEncoder(FEATURES, d_model=10, n_heads=4)
    with pytest.raises(ValueError, match="pooling"):
        TransformerEncoder(FEATURES, d_model=16, n_heads=2, pooling="max")
    with pytest.raises(ValueError):
        TransformerEncoder(FEATURES, d_model=16, n_heads=2, dropout=1.0)
    with pytest.raises(ValueError, match="window"):
        TransformerEncoder(FEATURES, d_model=16, n_heads=2, max_len=20)(torch.randn(1, 30, FEATURES))
