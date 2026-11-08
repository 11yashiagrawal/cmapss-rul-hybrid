"""Cross-validation and the final refit, shared by every sequence model.

Protocol (CONTRIBUTING.md section 8):

* The same 3 engine-grouped folds as Phase 1 (``group_kfold_splits``, seed 42).
* Inside each fold's training engines, an engine-level holdout
  (``holdout_split``) is used only for early stopping. Input scaling is fitted
  on the remaining engines, the ones the network trains on.
* The fold's held-out engines are scored at every cycle against the true,
  uncapped RUL. The selection metric is the RMSE on cycles whose true RUL is at
  most ``selection_max_rul`` (set from #55).
* The final refit trains on all training engines for the median best epoch.

Nothing here touches the official test set; that runs once, in #71.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from rul.data.labels import add_train_rul
from rul.data.split import group_kfold_splits, holdout_split
from rul.evaluation.metrics import evaluate, rmse
from rul.evaluation.official import METRICS_DIR, save_results
from rul.training.inputs import WindowInputs
from rul.training.trainer import FitResult, TrainConfig, fit, predict, set_seed

ModelFactory = Callable[[dict], nn.Module]
InputsFactory = Callable[[], WindowInputs]


@dataclass
class TrainedModel:
    """A trained network with the input builder it was fitted with.

    ``predict(frame)`` returns one RUL per row of ``frame``, in its row order,
    so it plugs into ``rul.evaluation.official`` like the Phase 1 models.
    """

    model: nn.Module
    inputs: WindowInputs
    target_scale: float
    train_units: list[int] = field(default_factory=list)
    device: str = "cpu"
    meta: dict = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict RUL at every row of ``frame`` (each row's own and earlier cycles)."""
        if not frame.index.is_unique:
            raise ValueError("frame needs a unique index to return predictions in row order")
        batch = self.inputs.transform(frame, target=None, stride=1)
        out = predict(self.model, batch.inputs, target_scale=self.target_scale, device=self.device)
        return pd.Series(out, index=batch.index["row"].to_numpy()).loc[frame.index].to_numpy()

    def outputs(self, frame: pd.DataFrame, fn: Callable[..., torch.Tensor]) -> np.ndarray:
        """``fn(*inputs)`` for every row of ``frame``, in its row order.

        ``fn`` takes the same input tensors as the model, for example
        ``model.encoder`` (embeddings) or a hybrid's ``gate``. It runs in
        evaluation mode without gradients.
        """
        if not frame.index.is_unique:
            raise ValueError("frame needs a unique index to return outputs in row order")
        batch = self.inputs.transform(frame, target=None, stride=1)
        self.model.to(self.device).eval()
        with torch.no_grad():
            out = torch.cat([
                fn(*(torch.as_tensor(a[i : i + 1024]).to(self.device) for a in batch.inputs)).cpu()
                for i in range(0, len(batch.index), 1024)
            ]).numpy()
        rows = pd.Series(np.arange(len(out)), index=batch.index["row"].to_numpy())
        return out[rows.loc[frame.index].to_numpy()]

    def embed(self, frame: pd.DataFrame) -> np.ndarray:
        """Encoder embeddings ``[rows, output_dim]`` for every row of ``frame``, in its row order.

        Needs a model with an ``encoder`` (a :class:`~rul.models.base.SequenceRegressor`).
        """
        return self.outputs(frame, self.model.encoder)

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        """Save the network, its fitted inputs and ``meta`` (e.g. the run config).

        Checkpoints belong under ``results/models/``, which git ignores.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if meta is not None:
            self.meta = meta
        self.model.to("cpu")
        torch.save(
            {
                "model": self.model,
                "inputs": self.inputs,
                "target_scale": self.target_scale,
                "train_units": self.train_units,
                "meta": self.meta,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> TrainedModel:
        """Load a checkpoint written by :meth:`save` (only load files you trust)."""
        state = torch.load(path, map_location=device, weights_only=False)
        return cls(
            state["model"], state["inputs"], state["target_scale"],
            state["train_units"], device, state["meta"],
        )


@dataclass
class FoldResult:
    """One cross-validation fold."""

    fold: int
    best_epoch: int
    seconds: float
    train_units: list[int]
    holdout_units: list[int]
    heldout_units: list[int]
    scores: dict
    history: list[dict]

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CVResult:
    """All folds, plus every held-out cycle's prediction."""

    folds: list[FoldResult]
    predictions: pd.DataFrame  # unit, cycle, rul (true, uncapped), fold, prediction

    @property
    def median_best_epoch(self) -> int:
        """Epochs for the final refit: the median of the folds' best epochs."""
        return int(np.median([f.best_epoch for f in self.folds]))

    def summary(self) -> dict:
        """Mean and standard deviation across folds of the main scores."""
        values = {
            "selection_rmse": [f.scores["selection_rmse"] for f in self.folds],
            **{
                m: [f.scores["all_cycles"][m] for f in self.folds]
                for m in ("rmse", "mae", "nasa_score")
            },
        }
        return {
            name: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
            for name, v in values.items()
        }

    def to_dict(self) -> dict:
        """JSON-ready record (without the per-cycle predictions)."""
        return {
            "summary": self.summary(),
            "median_best_epoch": self.median_best_epoch,
            "seconds": float(sum(f.seconds for f in self.folds)),
            "folds": [f.to_dict() for f in self.folds],
        }


def save_cv(
    name: str, cv: CVResult, meta: dict | None = None, out_dir: str | Path = METRICS_DIR
) -> dict[str, Path]:
    """Write ``<name>.json`` (scores, folds, ``meta``) and ``<name>_predictions.csv``.

    The CSV keeps every held-out cycle, so the selection metric can be
    recomputed later (for example with the threshold from #55) without
    retraining.
    """
    record = {"meta": meta or {}, **cv.to_dict()}
    return save_results(name, record, cv.predictions, out_dir=out_dir)


def score_heldout(y_true, y_pred, max_true_rul: float | None) -> dict:
    """Selection RMSE (cycles with true RUL <= ``max_true_rul``) and all-cycle scores.

    ``max_true_rul=None`` scores every cycle.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.ones(true.shape, bool) if max_true_rul is None else true <= max_true_rul
    return {
        "selection_rmse": rmse(true[mask], pred[mask]),
        "selection_cycles": int(mask.sum()),
        "max_true_rul": max_true_rul,
        "all_cycles": evaluate(true, pred),
    }


def _units(frame: pd.DataFrame) -> list[int]:
    return sorted(int(u) for u in frame["unit"].unique())


def train_on_engines(
    frame: pd.DataFrame,
    make_model: ModelFactory,
    make_inputs: InputsFactory,
    config: TrainConfig,
    *,
    cap: float | None,
    holdout_size: float | None,
    split_seed: int,
    epochs: int | None = None,
) -> tuple[TrainedModel, FitResult, list[int]]:
    """Fit inputs and a fresh model on ``frame`` (labelled with ``rul``).

    With ``holdout_size``, an engine-level holdout is split off for early
    stopping. Returns the trained model, its fit history and the holdout units.
    """
    if holdout_size:
        fit_idx, hold_idx = holdout_split(frame, holdout_size, split_seed)
        fit_part, hold_part = frame.iloc[fit_idx], frame.iloc[hold_idx]
    else:
        fit_part, hold_part = frame, None

    inputs = make_inputs().fit(fit_part)
    batch = inputs.transform(fit_part, target="rul")
    validation = None
    if hold_part is not None:
        held = inputs.transform(hold_part, target="rul", stride=1)
        validation = (held.inputs, held.y)

    scale = float(cap) if cap else float(batch.y.max())
    set_seed(config.seed)
    model = make_model(inputs.input_dims)
    result = fit(
        model, batch.inputs, batch.y, config,
        validation=validation, epochs=epochs, target_scale=scale,
    )
    trained = TrainedModel(model, inputs, scale, _units(fit_part), config.device)
    return trained, result, [] if hold_part is None else _units(hold_part)


def cross_validate(
    train: pd.DataFrame,
    make_model: ModelFactory,
    make_inputs: InputsFactory,
    config: TrainConfig,
    *,
    cap: float | None = 125,
    n_splits: int = 3,
    split_seed: int = 42,
    holdout_size: float = 0.2,
    selection_max_rul: float | None = None,
    on_fold: Callable[[int, TrainedModel, pd.DataFrame], None] | None = None,
) -> CVResult:
    """Score a model on the Phase 1 engine folds.

    Args:
        train: Training cycles (``unit``, ``cycle``, settings, sensors).
        make_model: Builds a fresh model from the inputs' ``input_dims``.
        make_inputs: Builds a fresh, unfitted input builder.
        config: Optimisation settings (its ``seed`` seeds every fold).
        cap: Training-label cap; ``None`` trains on the uncapped RUL.
        n_splits, split_seed: Phase 1 folds (3 and 42).
        holdout_size: Share of each fold's training engines used for early stopping.
        selection_max_rul: Threshold for the selection RMSE (from #55).
        on_fold: Called as ``on_fold(fold, trained, heldout)`` after each fold,
            with the fold's model and its held-out cycles (``rul`` is the
            training label, ``rul_true`` the uncapped RUL). Use it to inspect
            the fold models, e.g. a hybrid's gate on unseen engines.
    """
    labelled = add_train_rul(train, cap=cap).assign(rul_true=add_train_rul(train)["rul"])
    folds, predictions = [], []
    for k, (train_idx, val_idx) in enumerate(
        group_kfold_splits(labelled, n_splits=n_splits, seed=split_seed), start=1
    ):
        fold_train, fold_val = labelled.iloc[train_idx], labelled.iloc[val_idx]
        trained, result, holdout_units = train_on_engines(
            fold_train, make_model, make_inputs, config,
            cap=cap, holdout_size=holdout_size, split_seed=split_seed,
        )
        pred = trained.predict(fold_val)
        if on_fold is not None:
            on_fold(k, trained, fold_val)
        folds.append(
            FoldResult(
                fold=k,
                best_epoch=result.best_epoch,
                seconds=result.seconds,
                train_units=trained.train_units,
                holdout_units=holdout_units,
                heldout_units=_units(fold_val),
                scores=score_heldout(fold_val["rul_true"], pred, selection_max_rul),
                history=result.history,
            )
        )
        predictions.append(
            pd.DataFrame(
                {
                    "unit": fold_val["unit"].to_numpy(),
                    "cycle": fold_val["cycle"].to_numpy(),
                    "rul": fold_val["rul_true"].to_numpy(),
                    "fold": k,
                    "prediction": pred,
                }
            )
        )
    return CVResult(folds, pd.concat(predictions, ignore_index=True))


def cross_validate_estimator(
    train: pd.DataFrame,
    make_estimator: Callable[[], object],
    *,
    cap: float | None = 125,
    n_splits: int = 3,
    split_seed: int = 42,
    selection_max_rul: float | None = None,
) -> CVResult:
    """Like :func:`cross_validate`, for models that fit on frames themselves.

    ``make_estimator()`` returns an object with ``fit(labelled_frame)`` and
    ``predict(frame)``; after fitting it exposes ``train_units_`` (every
    engine it learned from), ``best_epoch_`` and ``fit_seconds_``. The
    two-stage hybrid (#65) runs through here, so it is scored exactly like
    the networks.
    """
    labelled = add_train_rul(train, cap=cap).assign(rul_true=add_train_rul(train)["rul"])
    folds, predictions = [], []
    for k, (train_idx, val_idx) in enumerate(
        group_kfold_splits(labelled, n_splits=n_splits, seed=split_seed), start=1
    ):
        fold_train, fold_val = labelled.iloc[train_idx], labelled.iloc[val_idx]
        estimator = make_estimator().fit(fold_train.drop(columns="rul_true"))
        pred = estimator.predict(fold_val)
        folds.append(
            FoldResult(
                fold=k,
                best_epoch=int(estimator.best_epoch_),
                seconds=float(estimator.fit_seconds_),
                train_units=list(estimator.train_units_),
                holdout_units=[],
                heldout_units=_units(fold_val),
                scores=score_heldout(fold_val["rul_true"], pred, selection_max_rul),
                history=[],
            )
        )
        predictions.append(
            pd.DataFrame(
                {
                    "unit": fold_val["unit"].to_numpy(),
                    "cycle": fold_val["cycle"].to_numpy(),
                    "rul": fold_val["rul_true"].to_numpy(),
                    "fold": k,
                    "prediction": pred,
                }
            )
        )
    return CVResult(folds, pd.concat(predictions, ignore_index=True))


def refit(
    train: pd.DataFrame,
    make_model: ModelFactory,
    make_inputs: InputsFactory,
    config: TrainConfig,
    *,
    cap: float | None = 125,
    epochs: int,
) -> tuple[TrainedModel, FitResult]:
    """Train on every training engine for a fixed number of epochs (no holdout)."""
    labelled = add_train_rul(train, cap=cap)
    trained, result, _ = train_on_engines(
        labelled, make_model, make_inputs, config,
        cap=cap, holdout_size=None, split_seed=0, epochs=epochs,
    )
    return trained, result
