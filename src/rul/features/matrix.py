"""Assemble final, training-fitted C-MAPSS feature matrices."""

from collections.abc import Sequence
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from rul.data.labels import add_train_rul
from rul.data.load import SETTING_COLS
from rul.features.cleaning import select_sensors
from rul.features.regimes import RegimeNormalizer
from rul.features.rolling import add_rolling_features
from rul.models.experiment import EngineFeatures


class FeatureMatrices(NamedTuple):
    """Aligned train/test matrices and their permitted predictor columns."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    sensors: list[str]
    n_regimes: int
    windows: tuple[int, ...]
    seed: int


def build_feature_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_regimes: int,
    windows: Sequence[int] = (5, 10, 20),
    seed: int = 42,
) -> FeatureMatrices:
    """Fit preprocessing on official training rows and transform both sets."""
    windows = tuple(windows)
    sensors = select_sensors(train)
    normalizer = RegimeNormalizer(n_regimes=n_regimes, random_state=seed).fit(train)
    train_scaled = normalizer.transform(add_train_rul(train))
    test_scaled = normalizer.transform(test)
    train_rolled = add_rolling_features(train_scaled, sensors=sensors, windows=windows)
    test_rolled = add_rolling_features(test_scaled, sensors=sensors, windows=windows)
    feature_columns = ["cycle", *SETTING_COLS, "regime", *sensors]
    feature_columns += [
        f"{sensor}_{feature}_{window}"
        for window in windows
        for sensor in sensors
        for feature in ("mean", "slope", "health")
    ]
    return FeatureMatrices(
        train_rolled.loc[:, ["unit", *feature_columns, "rul"]],
        test_rolled.loc[:, ["unit", *feature_columns]],
        feature_columns,
        sensors,
        n_regimes,
        windows,
        seed,
    )


def save_feature_matrices(
    matrices: FeatureMatrices, output_dir: str | Path, *, subset: str
) -> dict[str, Path]:
    """Write ignored CSV matrices and a JSON manifest for downstream runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / f"{subset}_train_features.csv",
        "test": output_dir / f"{subset}_test_features.csv",
        "manifest": output_dir / f"{subset}_feature_manifest.json",
    }
    matrices.train.to_csv(paths["train"], index=False)
    matrices.test.to_csv(paths["test"], index=False)
    manifest = {
        "subset": subset,
        "fit_scope": "all official training engines; refit preprocessing inside CV folds",
        "seed": matrices.seed,
        "n_regimes": matrices.n_regimes,
        "windows": list(matrices.windows),
        "sensors": matrices.sensors,
        "feature_columns": matrices.feature_columns,
        "train_rows": len(matrices.train),
        "test_rows": len(matrices.test),
        "train_engines": int(matrices.train["unit"].nunique()),
        "test_engines": int(matrices.test["unit"].nunique()),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


FEATURE_GROUPS = ("age", "sensors", "mean", "slope", "health")


def feature_group_columns(
    feature_names: Sequence[str], sensors: Sequence[str], groups: Sequence[str]
) -> list[str]:
    """Pick the engineered columns that belong to the requested feature groups.

    Groups: ``age`` (``cycle``), ``sensors`` (current scaled readings), and the
    rolling ``mean``, ``slope`` and ``health`` features. Operating settings and
    the regime label are not included: regime scaling already removes their
    effect from the sensors.
    """
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown or not groups:
        raise ValueError(f"groups must be a non-empty subset of {FEATURE_GROUPS}, got {list(groups)}")
    sensors = set(sensors)

    def group_of(name: str) -> str | None:
        if name == "cycle":
            return "age"
        if name in sensors:
            return "sensors"
        parts = name.rsplit("_", 2)
        if len(parts) == 3 and parts[0] in sensors and parts[1] in ("mean", "slope", "health"):
            return parts[1]
        return None

    return [name for name in feature_names if group_of(name) in groups]
