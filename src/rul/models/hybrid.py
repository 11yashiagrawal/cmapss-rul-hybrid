"""Hybrid models that combine a sequence encoder with the engineered features.

Two hybrids: **A** fuses the branches inside one network with a learned gate;
**B** feeds cross-fitted encoder embeddings and the features to XGBoost.

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

import numpy as np
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


USES = ("both", "features", "embeddings")
XGBOOST_DEFAULTS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
}


class TwoStageHybrid:
    """Hybrid B: XGBoost on engineered features plus cross-fitted encoder embeddings.

    Stage 1 learns sequence embeddings; stage 2 is gradient-boosted trees
    (a literal ML + DL combination). The embeddings XGBoost trains on must not
    come from an encoder that saw the same engine, or it would learn to trust
    embeddings that are better than they will be on new engines. So the
    training engines are split into ``inner_folds`` groups by engine, and each
    group is embedded by an encoder trained on the other groups
    (cross-fitting). Prediction uses one more encoder, trained on every
    training engine.

    The engineered features (#57) and the encoders' window inputs are fitted
    on the training engines given to :meth:`fit` only, so during
    cross-validation the held-out fold is unseen by both stages.

    Args:
        encoder: Encoder config, e.g. ``{"type": "lstm", "hidden_size": 64}``;
            it is trained with a linear head on the capped RUL.
        train_config: Optimisation settings for every encoder.
        inputs: :class:`~rul.training.inputs.WindowInputs` options.
        use: ``"both"``, or ``"features"`` / ``"embeddings"`` alone (ablations).
        inner_folds: Engine groups for cross-fitting.
        holdout_size: Engine holdout for each encoder's early stopping.
        split_seed: Seed for the inner groups and holdouts.
        feature_windows, groups, n_regimes: Engineered-feature options (#57).
        xgboost: XGBoost parameters, over :data:`XGBOOST_DEFAULTS`.
        seed: XGBoost seed.
    """

    def __init__(
        self,
        encoder: dict,
        train_config,
        *,
        inputs: dict | None = None,
        use: str = "both",
        inner_folds: int = 3,
        holdout_size: float = 0.2,
        split_seed: int = 42,
        feature_windows: tuple[int, ...] = (5, 10, 20),
        groups: tuple[str, ...] | None = None,
        n_regimes: int = 1,
        xgboost: dict | None = None,
        seed: int = 42,
    ) -> None:
        if use not in USES:
            raise ValueError(f"use must be one of {USES}, got {use!r}")
        self.encoder_spec = dict(encoder)
        self.train_config = train_config
        self.inputs = dict(inputs or {})
        self.use = use
        self.inner_folds = inner_folds
        self.holdout_size = holdout_size
        self.split_seed = split_seed
        self.feature_windows = tuple(feature_windows)
        self.groups = groups
        self.n_regimes = n_regimes
        self.xgboost = {**XGBOOST_DEFAULTS, **(xgboost or {})}
        self.seed = seed

    def make_inputs(self):
        from rul.training.inputs import WindowInputs

        return WindowInputs(**self.inputs)

    def _make_model(self, input_dims: dict) -> nn.Module:
        from rul.models.registry import build_model

        return build_model(self.encoder_spec, input_dims)

    def _train_encoder(self, frame):
        from rul.training.runs import train_on_engines

        return train_on_engines(
            frame, self._make_model, self.make_inputs, self.train_config,
            cap=None, holdout_size=self.holdout_size, split_seed=self.split_seed,
        )

    def fit(self, labelled) -> TwoStageHybrid:
        """Fit both stages on ``labelled`` cycles (``rul`` holds the training label)."""
        import time

        from xgboost import XGBRegressor

        from rul.data.split import group_kfold_splits
        from rul.features.matrix import FEATURE_GROUPS, HybridFeatures

        start = time.perf_counter()
        units = sorted(int(u) for u in labelled["unit"].unique())
        self.features_ = HybridFeatures(
            n_regimes=self.n_regimes, windows=self.feature_windows,
            groups=tuple(self.groups or FEATURE_GROUPS), seed=self.seed,
        ).fit(labelled)
        parts, names = [], []
        if self.use != "embeddings":
            parts.append(self._features(labelled))
            names += list(self.features_.output_columns_)

        self.crossfit_, self.encoder_, self.final_holdout_units_ = [], None, []
        self.best_epoch_ = 0
        if self.use != "features":
            embedded = None
            for fit_idx, emb_idx in group_kfold_splits(
                labelled, n_splits=self.inner_folds, seed=self.split_seed
            ):
                trained, result, holdout_units = self._train_encoder(labelled.iloc[fit_idx])
                part = labelled.iloc[emb_idx]
                emb = trained.embed(part)
                if embedded is None:
                    embedded = np.full((len(labelled), emb.shape[1]), np.nan, dtype=np.float32)
                embedded[emb_idx] = emb
                self.crossfit_.append({
                    "train_units": trained.train_units,
                    "holdout_units": holdout_units,
                    "embedded_units": sorted(int(u) for u in part["unit"].unique()),
                    "best_epoch": result.best_epoch,
                })
            assert not np.isnan(embedded).any(), "every training row needs a cross-fitted embedding"
            self.encoder_, result, self.final_holdout_units_ = self._train_encoder(labelled)
            self.best_epoch_ = result.best_epoch
            parts.append(embedded)
            names += [f"emb_{i}" for i in range(embedded.shape[1])]

        X = np.hstack(parts)
        self.feature_names_ = names
        self.train_shape_ = X.shape
        self.booster_ = XGBRegressor(**self.xgboost, random_state=self.seed).fit(
            X, labelled["rul"].to_numpy(dtype=float)
        )
        self.train_units_ = units
        self.fit_seconds_ = time.perf_counter() - start
        return self

    def _features(self, frame):
        return self.features_.transform(frame)[self.features_.output_columns_].to_numpy(dtype=np.float32)

    def predict(self, frame):
        """RUL for every row of ``frame``, in its row order."""
        parts = []
        if self.use != "embeddings":
            parts.append(self._features(frame))
        if self.use != "features":
            parts.append(self.encoder_.embed(frame))
        return self.booster_.predict(np.hstack(parts)).astype(float)

    def save(self, path, meta: dict | None = None):
        """Save both stages and ``meta`` to one file (under the gitignored ``results/models/``)."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = meta or {}
        if self.encoder_ is not None:
            self.encoder_.model.to("cpu")
        torch.save(self, path)
        return path

    @staticmethod
    def load(path) -> TwoStageHybrid:
        """Load a model written by :meth:`save` (only load files you trust)."""
        return torch.load(path, map_location="cpu", weights_only=False)
