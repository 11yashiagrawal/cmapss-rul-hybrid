import pytest
import torch
from torch import nn

from rul.models.base import SequenceEncoder

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


def test_base_encoder_requires_forward(windows):
    with pytest.raises(NotImplementedError):
        SequenceEncoder(DIM)(windows)
