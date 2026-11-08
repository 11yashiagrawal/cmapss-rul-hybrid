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

import numpy as np
import pandas as pd
from torch import nn

from rul.data.labels import add_train_rul
from rul.data.split import group_kfold_splits, holdout_split
from rul.evaluation.metrics import evaluate, rmse
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

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict RUL at every row of ``frame`` (each row's own and earlier cycles)."""
        if not frame.index.is_unique:
            raise ValueError("frame needs a unique index to return predictions in row order")
        batch = self.inputs.transform(frame, target=None, stride=1)
        out = predict(self.model, batch.inputs, target_scale=self.target_scale, device=self.device)
        return pd.Series(out, index=batch.index["row"].to_numpy()).loc[frame.index].to_numpy()


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


def _train(
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
    """Fit inputs and a fresh model on ``frame`` (labelled with ``rul``)."""
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
    """
    labelled = add_train_rul(train, cap=cap).assign(rul_true=add_train_rul(train)["rul"])
    folds, predictions = [], []
    for k, (train_idx, val_idx) in enumerate(
        group_kfold_splits(labelled, n_splits=n_splits, seed=split_seed), start=1
    ):
        fold_train, fold_val = labelled.iloc[train_idx], labelled.iloc[val_idx]
        trained, result, holdout_units = _train(
            fold_train, make_model, make_inputs, config,
            cap=cap, holdout_size=holdout_size, split_seed=split_seed,
        )
        pred = trained.predict(fold_val)
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
    trained, result, _ = _train(
        labelled, make_model, make_inputs, config,
        cap=cap, holdout_size=None, split_seed=0, epochs=epochs,
    )
    return trained, result
