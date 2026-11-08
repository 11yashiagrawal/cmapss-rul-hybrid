"""LSTM encoder (Hochreiter and Schmidhuber, 1997), the reference sequence model.

The window is read one cycle at a time. Input, forget and output gates decide
what the cell state keeps, so slow degradation can be carried across the whole
window instead of only the last few cycles a rolling feature sees. The
embedding is the top layer's hidden state after the window's last cycle.
"""

from __future__ import annotations

import torch
from torch import nn

from rul.models.base import SequenceEncoder


class LSTMEncoder(SequenceEncoder):
    """Stacked LSTM: ``[batch, window, features] -> [batch, hidden_size]``.

    Args:
        n_features: Inputs per cycle.
        hidden_size: Units per layer, which is also the embedding size.
        num_layers: Stacked LSTM layers.
        dropout: Dropout between layers and on the output embedding.
    """

    def __init__(
        self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__(hidden_size)
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden state of the top layer after the last cycle of each window."""
        _, (hidden, _) = self.lstm(x)
        return self.dropout(hidden[-1])
