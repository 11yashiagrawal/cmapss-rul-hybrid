"""Final feature matrices keep test data out of fitted preprocessing."""

import json

import numpy as np
import pandas as pd
import pytest

from rul.data.load import SENSOR_COLS, SETTING_COLS
from rul.features.matrix import (
    FEATURE_GROUPS,
    HybridFeatures,
    align_to_windows,
    build_feature_matrices,
    feature_group_columns,
    save_feature_matrices,
)


def cycles(offset: float = 0.0, length: int = 3) -> pd.DataFrame:
    rows = []
    for unit in (1, 2):
        for cycle in range(1, length + 1):
            row = {"unit": unit, "cycle": cycle, **dict.fromkeys(SETTING_COLS, 0.0)}
            row.update(dict.fromkeys(SENSOR_COLS, 1.0))
            row["s_2"] = cycle + unit + offset
            rows.append(row)
    return pd.DataFrame(rows)


def test_final_matrices_fit_on_train_and_keep_test_targets_out():
    matrices = build_feature_matrices(cycles(), cycles(offset=100, length=2), n_regimes=1, windows=(2,))

    assert matrices.sensors == ["s_2"]
    assert matrices.train.groupby("unit")["rul"].apply(list).tolist() == [[2, 1, 0], [2, 1, 0]]
    assert "rul" not in matrices.test
    assert "unit" not in matrices.feature_columns
    assert "rul" not in matrices.feature_columns
    assert list(matrices.train.columns) == ["unit", *matrices.feature_columns, "rul"]
    assert list(matrices.test.columns) == ["unit", *matrices.feature_columns]
    assert matrices.test["s_2"].min() > 50  # Uses train statistics, not test statistics.
    assert (matrices.train.groupby("unit")["s_2_health_2"].first() == 0).all()
    assert (matrices.test.groupby("unit")["s_2_health_2"].first() == 0).all()


def test_save_matrices_writes_reproducible_manifest(tmp_path):
    matrices = build_feature_matrices(cycles(), cycles(length=2), n_regimes=1, windows=(2,))

    paths = save_feature_matrices(
        matrices, tmp_path / "processed", subset="FD001",
    )

    saved_train = pd.read_csv(paths["train"])
    saved_test = pd.read_csv(paths["test"])
    manifest = json.loads(paths["manifest"].read_text())
    assert list(saved_train.columns) == list(matrices.train.columns)
    assert list(saved_test.columns) == list(matrices.test.columns)
    assert "rul" not in saved_test
    assert manifest["feature_columns"] == matrices.feature_columns
    assert manifest["sensors"] == ["s_2"]
    assert manifest["n_regimes"] == 1
    assert manifest["windows"] == [2]
    assert manifest["train_rows"] == 6
    assert manifest["test_rows"] == 4


def engines(lengths: dict[int, int], offset: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Engines with two drifting sensors; the other sensors are flat."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, n in lengths.items():
        for cycle in range(1, n + 1):
            row = {"unit": unit, "cycle": cycle, **dict.fromkeys(SETTING_COLS, 0.0)}
            row.update(dict.fromkeys(SENSOR_COLS, 1.0))
            row["s_2"] = 600 + 0.3 * cycle + rng.normal() + offset
            row["s_7"] = 550 - 0.2 * cycle + rng.normal() + offset
            rows.append(row)
    return pd.DataFrame(rows)


WINDOWS = (2, 4)


def test_hybrid_features_keep_input_rows_and_order():
    train = engines({1: 12, 2: 10})
    shuffled = train.sample(frac=1, random_state=1)
    out = HybridFeatures(windows=WINDOWS).fit(train).transform(shuffled)
    assert out.index.equals(shuffled.index)
    assert out[["unit", "cycle"]].equals(shuffled[["unit", "cycle"]])


def test_hybrid_features_are_scaled_with_training_statistics():
    train, test = engines({1: 15, 2: 12}), engines({9: 8}, offset=500, seed=3)
    model = HybridFeatures(windows=WINDOWS).fit(train)
    train_out = model.transform(train).drop(columns=["unit", "cycle"])
    np.testing.assert_allclose(train_out.mean(), 0, atol=1e-9)
    assert model.transform(test)["s_2"].min() > 50  # test drift is not scaled away


def test_hybrid_features_use_only_past_cycles():
    full = engines({1: 15})
    future_changed = full.copy()
    future_changed.loc[full["cycle"] > 8, ["s_2", "s_7"]] += 1000
    model = HybridFeatures(windows=WINDOWS).fit(engines({1: 15, 2: 12}))
    a = model.transform(full)[full["cycle"] <= 8]
    b = model.transform(future_changed)[future_changed["cycle"] <= 8]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        (("age",), ["age"]),
        (("sensors",), ["s_2", "s_7"]),
        (("mean",), ["s_2_mean_2", "s_7_mean_2", "s_2_mean_4", "s_7_mean_4"]),
        (("slope", "health"), [
            "s_2_slope_2", "s_2_health_2", "s_7_slope_2", "s_7_health_2",
            "s_2_slope_4", "s_2_health_4", "s_7_slope_4", "s_7_health_4",
        ]),
    ],
)
def test_each_group_returns_its_columns(groups, expected):
    model = HybridFeatures(windows=WINDOWS, groups=groups).fit(engines({1: 10, 2: 9}))
    assert model.output_columns_ == expected
    assert list(model.transform(engines({3: 5})).columns) == ["unit", "cycle", *expected]


def test_all_groups_cover_every_degradation_feature():
    model = HybridFeatures(windows=WINDOWS).fit(engines({1: 10, 2: 9}))
    assert len(model.output_columns_) == 1 + 2 + 2 * 2 * 3  # age, sensors, 3 rolling x 2 windows
    assert not {"op_1", "op_2", "op_3", "regime"} & set(model.output_columns_)


def test_rejects_unknown_or_empty_groups():
    with pytest.raises(ValueError):
        feature_group_columns(["cycle"], [], ("age", "settings"))
    with pytest.raises(ValueError):
        feature_group_columns(["cycle"], [], ())
    assert set(FEATURE_GROUPS) == {"age", "sensors", "mean", "slope", "health"}


def test_align_to_windows_follows_the_window_order():
    train = engines({1: 10, 2: 9})
    features = HybridFeatures(windows=WINDOWS).fit(train).transform(train)
    index = pd.DataFrame({"unit": [2, 1, 1], "cycle": [9, 3, 10]})
    rows = align_to_windows(features, index)
    keyed = features.set_index(["unit", "cycle"])
    assert rows.shape == (3, features.shape[1] - 2) and rows.dtype == np.float32
    np.testing.assert_allclose(rows[0], keyed.loc[(2, 9)].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(rows[1], keyed.loc[(1, 3)].to_numpy(), rtol=1e-6)


def test_align_to_windows_rejects_missing_window_ends():
    train = engines({1: 5})
    features = HybridFeatures(windows=WINDOWS).fit(train).transform(train)
    with pytest.raises(ValueError, match="no features"):
        align_to_windows(features, pd.DataFrame({"unit": [1], "cycle": [99]}))
