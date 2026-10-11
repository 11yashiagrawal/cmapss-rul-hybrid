"""Engine-aware preprocessing is fitted independently by each model pipeline."""

import numpy as np
import pandas as pd

from rul.data.load import SENSOR_COLS, SETTING_COLS
from sklearn.model_selection import GridSearchCV

from rul.data.split import group_kfold_splits
from rul.models.experiment import EngineFeatures, make_comparison_models, COMPARISON_GRIDS


def sample_cycles(offset=0.0):
    data = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2]})
    for name in SETTING_COLS:
        data[name] = 0.0
    for name in SENSOR_COLS:
        data[name] = 1.0
    data["s_2"] = np.array([1, 2, 3, 4]) + offset
    data["rul"] = [1, 0, 1, 0]
    return data


def test_engineered_transform_uses_training_statistics_and_sensor_selection():
    train = sample_cycles()
    validation = sample_cycles(100)
    validation["s_3"] = [1, 2, 3, 4]
    transformer = EngineFeatures(engineered=True, windows=(2,)).fit(train)

    transformed = transformer.transform(validation)

    assert "s_3" not in transformed
    assert transformed["s_2"].min() > 50
    assert {"unit", "rul"}.isdisjoint(transformed.columns)
    assert (transformed.loc[[0, 2], "s_2_health_2"] == 0).all()
    assert list(transformer.get_feature_names_out()) == list(transformed.columns)


def test_raw_transform_preserves_sensor_values_without_metadata_or_target():
    data = sample_cycles()
    result = EngineFeatures(engineered=False).fit_transform(data)

    assert result["s_2"].tolist() == data["s_2"].tolist()
    assert list(result.columns) == ["cycle", *SETTING_COLS, *SENSOR_COLS]


def test_comparison_pipelines_support_grouped_search_and_nested_elastic_scaling():
    data = sample_cycles()
    models = make_comparison_models(engineered=True, seed=9)
    assert set(models) == {"random_forest", "xgboost", "elastic_net"}
    assert models["elastic_net"].named_steps["model"].named_steps["scaler"] is not None
    for name, pipeline in models.items():
        pipeline.set_params(**{key: values[0] for key, values in COMPARISON_GRIDS[name].items()})
    search = GridSearchCV(models["random_forest"], {"model__n_estimators": [2]},
                          cv=list(group_kfold_splits(data, 2, seed=9)),
                          scoring="neg_root_mean_squared_error")
    search.fit(data, data["rul"])
    assert np.isfinite(search.predict(data)).all()
    assert search.best_estimator_.named_steps["features"].sensors_ == ["s_2"]
