"""Sensor selection follows the fitted regimes in every preprocessing path."""

import numpy as np
import pandas as pd
import pytest

from rul.data.load import DATA_DIR, SENSOR_COLS, SETTING_COLS, load_subset
from rul.features.matrix import build_feature_matrices
from rul.features.windows import WindowPreprocessor
from rul.models.experiment import EngineFeatures

REGIME_FLAT = ["s_1", "s_5", "s_18", "s_19"]
FD001_SENSORS = ["s_2", "s_3", "s_4", "s_6", "s_7", "s_8", "s_9", "s_11",
                 "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]


def two_regime_engines() -> pd.DataFrame:
    """s_1 jumps with the operating setting only; s_2 degrades within each regime."""
    rng = np.random.default_rng(0)
    rows = []
    for unit in (1, 2, 3, 4):
        for cycle in range(1, 21):
            setting = float(cycle % 2) * 20  # alternate between two operating points
            row = {"unit": unit, "cycle": cycle, **dict.fromkeys(SETTING_COLS, setting)}
            row.update(dict.fromkeys(SENSOR_COLS, 1.0))
            row["s_1"] = 100.0 + 5 * setting
            row["s_2"] = 600 + 0.5 * cycle + setting + rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def test_every_preprocessing_path_drops_a_regime_only_sensor():
    train = two_regime_engines()
    assert WindowPreprocessor(n_regimes=2).fit(train).sensors_ == ["s_2"]
    assert EngineFeatures(engineered=True, n_regimes=2).fit(train).sensors_ == ["s_2"]
    assert build_feature_matrices(train, train, n_regimes=2, windows=(3,)).sensors == ["s_2"]


def test_global_scaling_keeps_the_regime_only_sensor():
    assert WindowPreprocessor(n_regimes=1).fit(two_regime_engines()).sensors_ == ["s_1", "s_2"]


needs_data = pytest.mark.skipif(
    not (DATA_DIR / "train_FD002.txt").is_file(), reason="raw C-MAPSS data not downloaded"
)


@needs_data
def test_fd001_selection_is_unchanged():
    train = load_subset("FD001").train
    assert WindowPreprocessor(n_regimes=1).fit(train).sensors_ == FD001_SENSORS


@needs_data
@pytest.mark.parametrize("subset", ["FD002", "FD004"])
def test_multi_regime_subsets_drop_the_four_regime_flat_sensors(subset):
    sensors = WindowPreprocessor(n_regimes=6).fit(load_subset(subset).train).sensors_
    assert not set(REGIME_FLAT) & set(sensors)
    assert sensors == [s for s in SENSOR_COLS if s not in REGIME_FLAT]
