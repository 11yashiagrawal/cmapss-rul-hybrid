"""Shared contract for every Phase 2 sequence model.

Each architecture (LSTM, temporal CNN, Transformer, hybrid branches) is a
:class:`SequenceEncoder` that maps a batch of sensor windows to one embedding
per window. A :class:`RegressionHead` turns that embedding into an RUL
estimate, and :class:`SequenceRegressor` joins the two. Keeping this one
interface lets the trainer, the ablations and the hybrids treat every encoder
the same way.
"""

from __future__ import annotations

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    """Base class for encoders: ``[batch, window, features] -> [batch, output_dim]``.

    Subclasses call ``super().__init__(output_dim)`` and implement
    :meth:`forward`.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode windows shaped ``[batch, window, features]`` into ``[batch, output_dim]``."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")


class RegressionHead(nn.Module):
    """Map an embedding ``[batch, input_dim]`` to one RUL value per window, ``[batch]``.

    With ``hidden_dim=None`` the head is a single linear layer; otherwise it is
    one hidden ReLU layer with dropout.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.input_dim = int(input_dim)
        if hidden_dim is None:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return RUL estimates shaped ``[batch]``."""
        return self.net(z).squeeze(-1)
