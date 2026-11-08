"""Transformer encoder (Vaswani et al., 2017), kept small for FD001.

Each cycle in the window is projected to ``d_model`` and given a sinusoidal
position, so the model knows the order of cycles. Self-attention then lets
every cycle look at every other cycle in one step, instead of passing
information along the sequence like the LSTM, or through a fixed field like
the temporal CNN. FD001 has only about 18,000 training windows, so the model
stays small: one or two pre-norm layers of 32-64 units.

:meth:`TransformerEncoder.attention` returns each layer's attention weights,
for the interpretability work in #73.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from rul.models.base import SequenceEncoder

POOLINGS = ("mean", "last")


class SinusoidalPositions(nn.Module):
    """Adds the fixed sine/cosine position code to ``[batch, window, d_model]``."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        code = torch.zeros(max_len, d_model)
        code[:, 0::2] = torch.sin(position * frequency)
        code[:, 1::2] = torch.cos(position * frequency[: d_model // 2])
        self.register_buffer("code", code, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.code.shape[0]:
            raise ValueError(f"window of {x.shape[1]} cycles is longer than max_len {self.code.shape[0]}")
        return x + self.code[: x.shape[1]]


class EncoderLayer(nn.Module):
    """Pre-norm layer: self-attention then a feed-forward block, each with a residual."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(x)
        attended, weights = self.attn(
            h, h, h, need_weights=need_weights, average_attn_weights=False
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, weights


class TransformerEncoder(SequenceEncoder):
    """Small Transformer: ``[batch, window, features] -> [batch, d_model]``.

    Args:
        n_features: Inputs per cycle.
        d_model: Width of every layer, which is also the embedding size.
        n_heads: Attention heads; must divide ``d_model``.
        num_layers: Encoder layers.
        dim_feedforward: Hidden width of the feed-forward block (default ``2 * d_model``).
        dropout: Dropout on attention weights, residual branches and the embedding.
        pooling: ``"mean"`` averages all cycles; ``"last"`` takes the last cycle.
        max_len: Longest window the position code supports.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        pooling: str = "mean",
        max_len: int = 512,
    ) -> None:
        super().__init__(d_model)
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.pooling = pooling
        self.project = nn.Linear(n_features, d_model)
        self.positions = SinusoidalPositions(d_model, max_len)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, dim_feedforward or 2 * d_model, dropout)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _encode(self, x: torch.Tensor, need_weights: bool) -> tuple[torch.Tensor, list]:
        h = self.positions(self.project(x))
        maps = []
        for layer in self.layers:
            h, weights = layer(h, need_weights=need_weights)
            maps.append(weights)
        return self.norm(h), maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled embedding of each window."""
        h, _ = self._encode(x, need_weights=False)
        pooled = h.mean(dim=1) if self.pooling == "mean" else h[:, -1]
        return self.dropout(pooled)

    @torch.no_grad()
    def attention(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Attention weights per layer, each ``[batch, heads, window, window]``.

        Row ``i`` of a map says how much cycle ``i`` attends to each cycle.
        Call it in eval mode, so dropout doesn't change the weights.
        """
        _, maps = self._encode(x, need_weights=True)
        return maps
