"""Hybrid models that combine a sequence encoder with the engineered features.

**Hybrid A, gated fusion.** Rolling features summarize at most 20 cycles; the
sequence encoder reads the whole window in order. Each branch is embedded to
the same size ``d`` and mixed by a learned, per-dimension gate (a gated
multimodal unit, Arevalo et al. 2017)::

    g = sigmoid(W [h_seq, h_feat] + b)
    z = g * h_seq + (1 - g) * h_feat
    RUL = head(z)

``g`` near 1 means the model relies on the sequence branch, near 0 on the
features, so its value by life stage shows which branch matters when. Prior
work fuses handcrafted features with an LSTM by concatenation (Chen et al.
2021; Jin et al. 2022). ``fusion`` also offers that, and each branch alone,
for the ablations.
"""

from __future__ import annotations

import torch
from torch import nn

from rul.models.base import RegressionHead, SequenceEncoder

FUSIONS = ("gated", "concat", "sequence", "features")


class GatedHybrid(nn.Module):
    """Sequence branch plus engineered-feature branch: ``(x_seq, x_feat) -> [batch]``.

    Args:
        encoder: Sequence encoder for ``x_seq`` (``[batch, window, features]``).
            Its ``output_dim`` sets the embedding size ``d`` of both branches.
        n_engineered: Engineered features per window (``x_feat`` is ``[batch, n_engineered]``).
        fusion: ``"gated"``, ``"concat"``, or one branch alone (``"sequence"``,
            ``"features"``); the unused branch isn't built.
        feature_hidden: Hidden units of the feature branch's MLP.
        dropout: Dropout in the feature branch.
        head: Regression head; defaults to a linear layer of the right size.
    """

    def __init__(
        self,
        encoder: SequenceEncoder,
        n_engineered: int,
        *,
        fusion: str = "gated",
        feature_hidden: int = 64,
        dropout: float = 0.1,
        head: RegressionHead | None = None,
    ) -> None:
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
        self.fusion = fusion
        d = encoder.output_dim
        self.embedding_dim = d
        if fusion != "features":
            self.encoder = encoder
        if fusion != "sequence":
            self.feature_net = nn.Sequential(
                nn.Linear(n_engineered, feature_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_hidden, d),
                nn.Tanh(),
            )
        if fusion == "gated":
            self.gate_net = nn.Linear(2 * d, d)
        head_dim = 2 * d if fusion == "concat" else d
        self.head = head if head is not None else RegressionHead(head_dim)
        if self.head.input_dim != head_dim:
            raise ValueError(f"head expects {self.head.input_dim} inputs, fusion gives {head_dim}")

    def branches(
        self, x_seq: torch.Tensor, x_feat: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Embeddings ``h_seq`` and ``h_feat`` (``None`` for a branch that's off)."""
        h_seq = self.encoder(x_seq) if self.fusion != "features" else None
        h_feat = self.feature_net(x_feat) if self.fusion != "sequence" else None
        return h_seq, h_feat

    def _gate(self, h_seq: torch.Tensor, h_feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_net(torch.cat([h_seq, h_feat], dim=1)))

    def gate(self, x_seq: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Gate values ``[batch, d]`` in [0, 1]; 1 means all sequence branch."""
        if self.fusion != "gated":
            raise ValueError(f"only the gated fusion has a gate, this model uses {self.fusion!r}")
        return self._gate(*self.branches(x_seq, x_feat))

    def forward(self, x_seq: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        h_seq, h_feat = self.branches(x_seq, x_feat)
        if self.fusion == "gated":
            g = self._gate(h_seq, h_feat)
            z = g * h_seq + (1 - g) * h_feat
        elif self.fusion == "concat":
            z = torch.cat([h_seq, h_feat], dim=1)
        else:
            z = h_seq if h_seq is not None else h_feat
        return self.head(z)


def build_gated_hybrid(
    n_features: int,
    n_engineered: int,
    *,
    encoder: dict,
    fusion: str = "gated",
    feature_hidden: int = 64,
    dropout: float = 0.1,
    head: dict | None = None,
) -> GatedHybrid:
    """Build Hybrid A from a config: ``model: {type: hybrid_gated, encoder: {type: lstm, ...}}``."""
    from rul.models.registry import build_encoder

    seq_encoder = build_encoder(encoder, n_features)
    head_dim = 2 * seq_encoder.output_dim if fusion == "concat" else seq_encoder.output_dim
    return GatedHybrid(
        seq_encoder, n_engineered, fusion=fusion, feature_hidden=feature_hidden,
        dropout=dropout, head=RegressionHead(head_dim, **(head or {})),
    )
