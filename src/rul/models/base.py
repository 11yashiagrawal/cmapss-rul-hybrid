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
