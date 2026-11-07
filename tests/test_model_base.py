import pytest
import torch
from torch import nn

from rul.models.base import RegressionHead, SequenceEncoder, SequenceRegressor

BATCH, WINDOW, FEATURES, DIM = 4, 30, 15, 8


class MeanEncoder(SequenceEncoder):
    """Tiny stand-in encoder: average over time, then one linear layer."""

    def __init__(self, n_features: int, output_dim: int) -> None:
        super().__init__(output_dim)
        self.proj = nn.Linear(n_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=1))


@pytest.fixture
def windows() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, WINDOW, FEATURES)


def test_encoder_maps_windows_to_embeddings(windows):
    encoder = MeanEncoder(FEATURES, DIM)
    assert encoder(windows).shape == (BATCH, DIM)
    assert encoder.output_dim == DIM


@pytest.mark.parametrize("hidden_dim", [None, 16])
def test_regressor_returns_one_value_per_window(windows, hidden_dim):
    model = SequenceRegressor(
        MeanEncoder(FEATURES, DIM), RegressionHead(DIM, hidden_dim=hidden_dim, dropout=0.1)
    )
    assert model(windows).shape == (BATCH,)


def test_default_head_matches_encoder(windows):
    model = SequenceRegressor(MeanEncoder(FEATURES, DIM))
    assert model.head.input_dim == DIM
    assert model(windows).shape == (BATCH,)


def test_backward_pass_reaches_every_parameter(windows):
    model = SequenceRegressor(MeanEncoder(FEATURES, DIM), RegressionHead(DIM, hidden_dim=16))
    loss = nn.functional.mse_loss(model(windows), torch.rand(BATCH) * 100)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_base_encoder_requires_forward(windows):
    with pytest.raises(NotImplementedError):
        SequenceEncoder(DIM)(windows)


def test_rejects_mismatched_head():
    with pytest.raises(ValueError, match="head expects"):
        SequenceRegressor(MeanEncoder(FEATURES, DIM), RegressionHead(DIM + 1))


def test_rejects_non_encoder():
    with pytest.raises(TypeError):
        SequenceRegressor(nn.Linear(FEATURES, DIM))


def test_rejects_input_without_time_axis():
    model = SequenceRegressor(MeanEncoder(FEATURES, DIM))
    with pytest.raises(ValueError, match=r"\[batch, window, features\]"):
        model(torch.randn(BATCH, FEATURES))


def test_rejects_encoder_with_wrong_output_shape(windows):
    class WrongEncoder(SequenceEncoder):
        def forward(self, x):
            return x[:, -1, :]  # FEATURES columns, not output_dim

    with pytest.raises(ValueError, match="encoder returned"):
        SequenceRegressor(WrongEncoder(DIM))(windows)


@pytest.mark.parametrize("bad", [0, -3])
def test_rejects_non_positive_dims(bad):
    with pytest.raises(ValueError):
        SequenceEncoder(bad)
    with pytest.raises(ValueError):
        RegressionHead(bad)
