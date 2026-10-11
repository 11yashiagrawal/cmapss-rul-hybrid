"""Final feature matrices keep test data out of fitted preprocessing."""

import json

import pandas as pd

from rul.data.load import SENSOR_COLS, SETTING_COLS
from rul.features.matrix import build_feature_matrices, save_feature_matrices


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
