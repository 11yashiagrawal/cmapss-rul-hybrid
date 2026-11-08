"""Tests for cross-validation, the final refit and the frame-level predictor."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rul.data.labels import add_train_rul
from rul.data.load import SENSOR_COLS, SETTING_COLS
from rul.data.split import group_kfold_splits
from rul.evaluation.official import predict_test_cycles, score_last_cycle
from rul.models.base import SequenceEncoder, SequenceRegressor
from rul.training.inputs import WindowInputs
from rul.training.runs import (
    TrainedModel,
    cross_validate,
    refit,
    save_cv,
    score_heldout,
)
from rul.training.trainer import TrainConfig


class MeanEncoder(SequenceEncoder):
    def __init__(self, n_features: int, output_dim: int = 8) -> None:
        super().__init__(output_dim)
        self.proj = nn.Sequential(nn.Linear(n_features, output_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=1))


def make_model(dims: dict) -> SequenceRegressor:
    return SequenceRegressor(MeanEncoder(dims["n_features"]))


def make_inputs() -> WindowInputs:
    return WindowInputs(window=5)


def engines(n_units: int = 9, seed: int = 0) -> pd.DataFrame:
    """Run-to-failure engines whose two live sensors drift as failure nears."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(45, 80))
        for cycle in range(1, life + 1):
            health = np.exp(-(life - cycle) / 20)
            row = {"unit": unit, "cycle": cycle, **dict.fromkeys(SETTING_COLS, 0.0)}
            row.update(dict.fromkeys(SENSOR_COLS, 5.0))
            row["s_2"] = 600 + 10 * health + rng.normal(0, 0.3)
            row["s_3"] = 1500 - 15 * health + rng.normal(0, 0.3)
            rows.append(row)
    return pd.DataFrame(rows)


CONFIG = TrainConfig(epochs=6, batch_size=64, learning_rate=1e-2, patience=2, seed=0)
CAP = 40


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return engines()


@pytest.fixture(scope="module")
def cv(train):
    return cross_validate(
        train, make_model, make_inputs, CONFIG, cap=CAP, n_splits=3, holdout_size=0.25,
        selection_max_rul=30,
    )


def test_window_inputs_give_one_window_per_cycle(train):
    inputs = make_inputs().fit(train)
    batch = inputs.transform(add_train_rul(train), target="rul")
    assert inputs.input_dims == {"n_features": 2}
    assert len(batch.inputs) == 1
    assert batch.inputs[0].shape == (len(train), 5, 2)
    assert batch.y.shape == (len(train),)


def test_cv_uses_the_phase1_folds(train, cv):
    expected = [set(train["unit"].iloc[val]) for _, val in group_kfold_splits(train, 3, 42)]
    assert [set(f.heldout_units) for f in cv.folds] == expected


def test_cv_keeps_training_holdout_and_scored_engines_apart(train, cv):
    for fold in cv.folds:
        fit_units, hold, scored = map(set, (fold.train_units, fold.holdout_units, fold.heldout_units))
        assert hold and not (fit_units & hold or fit_units & scored or hold & scored)
        assert fit_units | hold | scored == set(train["unit"])


def test_cv_scores_every_cycle_against_the_uncapped_rul(train, cv):
    pred = cv.predictions.sort_values(["unit", "cycle"]).reset_index(drop=True)
    truth = add_train_rul(train).sort_values(["unit", "cycle"]).reset_index(drop=True)
    assert len(pred) == len(train)
    np.testing.assert_array_equal(pred["rul"], truth["rul"])
    assert pred["rul"].max() > CAP
    fold_of = {u: f.fold for f in cv.folds for u in f.heldout_units}
    assert (pred["fold"] == pred["unit"].map(fold_of)).all()


def test_cv_records_best_epochs_and_their_median(cv):
    best = [f.best_epoch for f in cv.folds]
    assert all(1 <= b <= CONFIG.epochs for b in best)
    assert cv.median_best_epoch == int(np.median(best))


def test_cv_summary_and_dict_are_json_ready(cv):
    summary = cv.summary()
    assert set(summary) >= {"selection_rmse", "rmse", "mae", "nasa_score"}
    assert summary["selection_rmse"]["mean"] == pytest.approx(
        np.mean([f.scores["selection_rmse"] for f in cv.folds])
    )
    record = cv.to_dict()
    assert record["median_best_epoch"] == cv.median_best_epoch
    assert len(record["folds"]) == 3 and "history" in record["folds"][0]


def test_cv_is_reproducible(train, cv):
    again = cross_validate(
        train, make_model, make_inputs, CONFIG, cap=CAP, n_splits=3, holdout_size=0.25,
        selection_max_rul=30,
    )
    pd.testing.assert_frame_equal(cv.predictions, again.predictions)


def test_selection_metric_keeps_only_cycles_near_failure():
    scores = score_heldout([10, 20, 40, 80], [12, 20, 40, 0], max_true_rul=40)
    assert scores["selection_rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert scores["selection_cycles"] == 3
    assert scores["all_cycles"]["rmse"] == pytest.approx(np.sqrt((4 + 6400) / 4))
    assert score_heldout([10, 80], [10, 70], max_true_rul=None)["selection_cycles"] == 2


def test_refit_trains_on_every_engine_for_the_given_epochs(train):
    trained, result = refit(train, make_model, make_inputs, CONFIG, cap=CAP, epochs=3)
    assert [h["epoch"] for h in result.history] == [1, 2, 3]
    assert set(trained.train_units) == set(train["unit"])
    assert trained.target_scale == CAP


def test_uncapped_target_scales_by_the_longest_life(train):
    trained, _ = refit(train, make_model, make_inputs, CONFIG, cap=None, epochs=1)
    assert trained.target_scale == add_train_rul(train)["rul"].max()


def test_trained_model_predicts_in_frame_row_order(train):
    trained, _ = refit(train, make_model, make_inputs, CONFIG, cap=CAP, epochs=2)
    shuffled = train.sample(frac=1, random_state=3)
    ordered = pd.Series(trained.predict(train), index=train.index)
    np.testing.assert_allclose(trained.predict(shuffled), ordered.loc[shuffled.index])


def test_trained_model_plugs_into_official_scoring(train):
    trained, _ = refit(train, make_model, make_inputs, CONFIG, cap=CAP, epochs=2)
    test = engines(4, seed=5)
    test = test[test["cycle"] <= 30]  # test engines stop before failure
    rul_test = pd.Series([12.0, 7.0, 20.0, 3.0], index=pd.Index([1, 2, 3, 4], name="unit"))
    last, scores = score_last_cycle(predict_test_cycles(trained, test, rul_test))
    assert list(last["unit"]) == [1, 2, 3, 4]
    assert np.isfinite(scores["rmse"])


def test_trained_model_rejects_duplicate_row_labels(train):
    trained, _ = refit(train, make_model, make_inputs, CONFIG, cap=CAP, epochs=1)
    with pytest.raises(ValueError, match="unique index"):
        trained.predict(pd.concat([train.head(5), train.head(5)]))


def test_trained_model_round_trips_through_a_checkpoint(train, tmp_path):
    trained, _ = refit(train, make_model, make_inputs, CONFIG, cap=CAP, epochs=2)
    path = trained.save(tmp_path / "models" / "run.pt", meta={"name": "run"})
    loaded = TrainedModel.load(path)
    np.testing.assert_array_equal(loaded.predict(train), trained.predict(train))
    assert loaded.target_scale == trained.target_scale
    assert loaded.meta == {"name": "run"}


def test_save_cv_writes_metrics_json_and_cycle_predictions(train, cv, tmp_path):
    paths = save_cv("fd001_cv_stand_in", cv, meta={"model": {"type": "mean"}}, out_dir=tmp_path)
    record = json.loads(paths["metrics"].read_text())
    assert record["meta"] == {"model": {"type": "mean"}}
    assert record["median_best_epoch"] == cv.median_best_epoch
    assert record["summary"]["selection_rmse"]["mean"] == pytest.approx(
        cv.summary()["selection_rmse"]["mean"]
    )
    saved = pd.read_csv(paths["predictions"])
    assert list(saved.columns) == ["unit", "cycle", "rul", "fold", "prediction"]
    assert len(saved) == len(train)
