"""Tests for Hybrid B: XGBoost on engineered features plus cross-fitted embeddings."""

import numpy as np
import pandas as pd
import pytest

from rul.data.labels import add_train_rul
from rul.data.load import SENSOR_COLS, SETTING_COLS
from rul.models.hybrid import TwoStageHybrid
from rul.training.config import run_cv
from rul.training.inputs import WindowInputs
from rul.training.runs import TrainedModel, cross_validate_estimator
from rul.training.trainer import TrainConfig

TINY = TrainConfig(epochs=2, batch_size=64, learning_rate=1e-2, patience=1, seed=0)
ENCODER = {"type": "lstm", "hidden_size": 8, "num_layers": 1}
XGB = {"n_estimators": 10, "max_depth": 2}


def engines(n_units: int = 9, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(30, 45))
        for cycle in range(1, life + 1):
            health = np.exp(-(life - cycle) / 15)
            row = {"unit": unit, "cycle": cycle, **dict.fromkeys(SETTING_COLS, 0.0)}
            row.update(dict.fromkeys(SENSOR_COLS, 5.0))
            row["s_2"] = 600 + 10 * health + rng.normal(0, 0.3)
            row["s_3"] = 1500 - 15 * health + rng.normal(0, 0.3)
            rows.append(row)
    return pd.DataFrame(rows)


def two_stage(**kwargs) -> TwoStageHybrid:
    options = {"encoder": ENCODER, "train_config": TINY, "inputs": {"window": 5},
               "feature_windows": (2, 4), "inner_folds": 3, "holdout_size": 0.25, "xgboost": XGB}
    return TwoStageHybrid(**{**options, **kwargs})


@pytest.fixture(scope="module")
def fitted() -> tuple[TwoStageHybrid, pd.DataFrame]:
    labelled = add_train_rul(engines(), cap=30)
    return two_stage().fit(labelled), labelled


def test_every_training_engine_is_embedded_by_an_encoder_that_never_saw_it(fitted):
    model, labelled = fitted
    embedded = []
    for record in model.crossfit_:
        seen = set(record["train_units"]) | set(record["holdout_units"])
        assert seen.isdisjoint(record["embedded_units"])
        embedded += record["embedded_units"]
    assert sorted(embedded) == sorted(labelled["unit"].unique())
    assert len(model.crossfit_) == 3


def test_final_encoder_trains_on_every_engine(fitted):
    model, labelled = fitted
    assert isinstance(model.encoder_, TrainedModel)
    seen = set(model.encoder_.train_units) | set(model.final_holdout_units_)
    assert seen == set(labelled["unit"])


def test_booster_input_is_features_plus_embeddings(fitted):
    model, labelled = fitted
    n_features = len(model.features_.output_columns_)
    assert model.train_shape_ == (len(labelled), n_features + 8)
    assert model.feature_names_[-8:] == [f"emb_{i}" for i in range(8)]


def test_predicts_every_row_in_frame_order(fitted):
    model, labelled = fitted
    shuffled = labelled.sample(frac=1, random_state=1)
    ordered = pd.Series(model.predict(labelled), index=labelled.index)
    np.testing.assert_allclose(model.predict(shuffled), ordered.loc[shuffled.index])


@pytest.mark.parametrize(("use", "width"), [("features", None), ("embeddings", 8)])
def test_either_input_can_be_switched_off(use, width):
    labelled = add_train_rul(engines(), cap=30)
    model = two_stage(use=use).fit(labelled)
    expected = width or len(model.features_.output_columns_)
    assert model.train_shape_ == (len(labelled), expected)
    assert model.predict(labelled).shape == (len(labelled),)
    if use == "features":
        assert model.crossfit_ == [] and model.encoder_ is None


def test_cross_validation_keeps_held_out_engines_unseen():
    cv = cross_validate_estimator(engines(), lambda: two_stage(), cap=30, n_splits=3,
                                  selection_max_rul=25)
    for fold in cv.folds:
        assert set(fold.train_units).isdisjoint(fold.heldout_units)
    assert len(cv.predictions) == len(engines())


def test_config_runs_the_two_stage_hybrid():
    config = {
        "target": {"cap": 30}, "selection": {"max_true_rul": 25},
        "cv": {"n_splits": 3, "split_seed": 42, "holdout_size": 0.25},
        "inputs": {"window": 5}, "training": TINY.to_dict(),
        "model": {"type": "hybrid_two_stage", "encoder": ENCODER, "inner_folds": 3,
                  "feature_windows": [2, 4], "xgboost": XGB},
    }
    cv = run_cv(config, engines())
    assert len(cv.folds) == 3 and cv.folds[0].scores["max_true_rul"] == 25


def test_window_inputs_are_the_default_encoder_inputs():
    assert isinstance(two_stage().make_inputs(), WindowInputs)


def test_refit_from_config_saves_and_loads(tmp_path):
    from rul.training.config import run_refit

    config = {
        "target": {"cap": 30}, "cv": {"n_splits": 3, "split_seed": 42, "holdout_size": 0.25},
        "inputs": {"window": 5}, "training": TINY.to_dict(),
        "model": {"type": "hybrid_two_stage", "encoder": ENCODER, "inner_folds": 3,
                  "feature_windows": [2, 4], "xgboost": XGB},
    }
    train = engines()
    model, seconds = run_refit(config, train, epochs=99)  # epochs only matter for networks
    assert isinstance(model, TwoStageHybrid) and seconds > 0
    path = model.save(tmp_path / "two_stage.pt", meta={"name": "x"})
    loaded = TwoStageHybrid.load(path)
    np.testing.assert_allclose(loaded.predict(train), model.predict(train))
    assert loaded.meta == {"name": "x"}
