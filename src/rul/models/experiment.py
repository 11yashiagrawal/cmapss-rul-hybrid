"""Fold-fitted feature pipelines for the raw/engineered ML comparison."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from rul.data.load import SENSOR_COLS, SETTING_COLS
from rul.features.cleaning import select_sensors
from rul.features.regimes import RegimeNormalizer
from rul.features.rolling import add_rolling_features
from rul.models.ml import make_models


COMPARISON_GRIDS = {
    "random_forest": {"model__n_estimators": [50, 100], "model__min_samples_leaf": [1, 3]},
    "xgboost": {"model__n_estimators": [100, 200], "model__max_depth": [3, 5]},
    "elastic_net": {"model__model__alpha": [0.1, 1.0], "model__model__l1_ratio": [0.2, 0.8]},
}


class EngineFeatures(TransformerMixin, BaseEstimator):
    """Keep engine metadata for causal transforms, then return predictors only."""

    def __init__(
        self, engineered: bool = False, n_regimes: int = 1,
        windows: tuple[int, ...] = (5, 10, 20), seed: int = 42,
    ) -> None:
        self.engineered = engineered
        self.n_regimes = n_regimes
        self.windows = windows
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: object = None) -> "EngineFeatures":
        """Learn selected sensors and regime statistics from this training fold."""
        if self.engineered:
            self.sensors_ = select_sensors(X)
            self.normalizer_ = RegimeNormalizer(self.n_regimes, self.seed).fit(X)
            self.feature_names_ = ["cycle", *SETTING_COLS, "regime", *self.sensors_]
            self.feature_names_ += [
                f"{sensor}_{feature}_{window}"
                for window in self.windows
                for sensor in self.sensors_
                for feature in ("mean", "slope", "health")
            ]
        else:
            self.feature_names_ = ["cycle", *SETTING_COLS, *SENSOR_COLS]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform each engine using training statistics and observed history."""
        check_is_fitted(self, "feature_names_")
        if self.engineered:
            X = add_rolling_features(
                self.normalizer_.transform(X), self.sensors_, self.windows
            )
        return X.loc[:, self.feature_names_].copy()

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the predictor names in their model-input order."""
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)


def make_comparison_models(engineered: bool, seed: int = 42) -> dict[str, Pipeline]:
    """Wrap seeded baselines with fold-fitted raw or engineered predictors."""
    models = make_models(seed)
    models["random_forest"].set_params(max_features=0.5)
    models["elastic_net"].set_params(model__max_iter=20000)
    return {
        name: Pipeline([
            ("features", EngineFeatures(engineered=engineered, seed=seed)),
            ("model", model),
        ])
        for name, model in models.items()
    }
