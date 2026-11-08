"""Temporal convolutional encoder (Bai, Kolter and Koltun, 2018).

Stacks of dilated causal 1-D convolutions in residual blocks. Causal means
the output at cycle ``t`` only uses cycles up to ``t``. Dilation doubles at
each level, so the receptive field (the cycles the last output can see)
grows exponentially with depth while the parameter count grows linearly.
Unlike the LSTM, the field is explicit: :func:`receptive_field` gives it
exactly, and a setting whose field is shorter than the window ignores the
oldest cycles.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rul.models.base import SequenceEncoder


def receptive_field(kernel_size: int, levels: int) -> int:
    """Cycles the last output can see: ``1 + 2 (k - 1) (2^levels - 1)``.

    Each level has two convolutions with dilation ``2^level``.
    """
    return 1 + 2 * (kernel_size - 1) * (2**levels - 1)


class CausalConv1d(nn.Module):
    """1-D convolution padded on the left only, so no output sees a later step."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with ReLU and dropout, plus a residual path."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + self.residual(x))


class TCNEncoder(SequenceEncoder):
    """Temporal CNN: ``[batch, window, features] -> [batch, channels]``.

    The embedding is the top level's output at the window's last cycle.

    Args:
        n_features: Inputs per cycle.
        channels: Channels in every block, which is also the embedding size.
        kernel_size: Convolution width (at least 2).
        levels: Residual blocks; block ``i`` uses dilation ``2^i``.
        dropout: Dropout after each convolution.
    """

    def __init__(
        self,
        n_features: int,
        channels: int = 32,
        kernel_size: int = 3,
        levels: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(channels)
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be at least 2, got {kernel_size}")
        if levels < 1:
            raise ValueError(f"levels must be positive, got {levels}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.kernel_size = kernel_size
        self.levels = levels
        self.blocks = nn.Sequential(
            *(
                TemporalBlock(n_features if i == 0 else channels, channels, kernel_size, 2**i, dropout)
                for i in range(levels)
            )
        )

    @property
    def receptive_field(self) -> int:
        """Cycles the embedding can see, counting the last one."""
        return receptive_field(self.kernel_size, self.levels)

    def covers(self, window: int) -> bool:
        """Whether the embedding sees every cycle of a window this long."""
        return self.receptive_field >= window

    def sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Per-cycle outputs ``[batch, channels, window]``; cycle ``t`` only sees cycles up to ``t``."""
        return self.blocks(x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output at the last cycle of each window."""
        return self.sequence(x)[..., -1]
