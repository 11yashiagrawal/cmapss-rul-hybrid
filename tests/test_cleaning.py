"""Training-only selection of useful sensor columns."""

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from rul.features.cleaning import select_sensors


def test_select_sensors_drops_only_training_constants():
    train = pd.DataFrame({"s_1": [5.0, 5.0, 5.0], "s_2": [1.0, 2.0, 3.0]})

    assert select_sensors(train, ["s_1", "s_2"]) == ["s_2"]


def test_select_sensors_can_drop_nearly_flat_training_signals():
    train = pd.DataFrame(
        {
            "s_1": [999.99, 1000.0, 1000.01],
            "s_2": [1.0, 2.0, 3.0],
        }
    )

    assert select_sensors(train, ["s_1", "s_2"], min_relative_std=0.001) == ["s_2"]


@pytest.mark.parametrize("cutoff", [-0.1, float("nan"), float("inf")])
def test_select_sensors_rejects_invalid_cutoff(cutoff):
    train = pd.DataFrame({"s_1": [1.0, 2.0]})

    with pytest.raises(ValueError, match="min_relative_std"):
        select_sensors(train, ["s_1"], min_relative_std=cutoff)


def test_select_sensors_rejects_training_fold_without_useful_sensors():
    train = pd.DataFrame({"s_1": [5.0, 5.0]})

    with pytest.raises(ValueError, match="No sensors"):
        select_sensors(train, ["s_1"])


def test_select_sensors_drops_float_constant_despite_roundoff():
    train = pd.DataFrame(
        {"s_1": [14.62] * 7, "s_2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
    )

    assert select_sensors(train, ["s_1", "s_2"]) == ["s_2"]


def test_raw_training_choice_is_reused_on_scaled_validation_rows():
    train = pd.DataFrame({"s_1": [999.99, 1000.0, 1000.01], "s_2": [1.0, 2.0, 3.0]})
    validation = pd.DataFrame({"s_1": [800.0, 1200.0], "s_2": [4.0, 5.0]})
    kept = select_sensors(train, ["s_1", "s_2"], min_relative_std=0.001)
    scaler = StandardScaler().fit(train)
    scaled_validation = pd.DataFrame(scaler.transform(validation), columns=train.columns)

    assert kept == ["s_2"]
    assert scaled_validation[kept].columns.tolist() == kept


def test_regime_flat_sensor_is_dropped_only_when_regimes_are_given():
    # s_1 only tracks the operating condition: one value per regime, nothing within it.
    train = pd.DataFrame(
        {"s_1": [10.0, 10.0, 50.0, 50.0], "s_2": [1.0, 2.0, 7.0, 9.0]}
    )
    regimes = [0, 0, 1, 1]

    assert select_sensors(train, ["s_1", "s_2"]) == ["s_1", "s_2"]
    assert select_sensors(train, ["s_1", "s_2"], regimes=regimes) == ["s_2"]


def test_sensor_that_varies_in_any_regime_is_kept():
    train = pd.DataFrame({"s_1": [10.0, 10.0, 50.0, 51.0]})

    assert select_sensors(train, ["s_1"], regimes=[0, 0, 1, 1]) == ["s_1"]


def test_one_regime_matches_the_plain_selection():
    train = pd.DataFrame(
        {"s_1": [5.0, 5.0, 5.0], "s_2": [999.99, 1000.0, 1000.01], "s_3": [1.0, 2.0, 3.0]}
    )
    sensors = ["s_1", "s_2", "s_3"]

    for cutoff in (0.0, 0.001):
        assert select_sensors(train, sensors, cutoff, regimes=[0, 0, 0]) == select_sensors(
            train, sensors, cutoff
        )


def test_regime_labels_must_match_training_rows():
    train = pd.DataFrame({"s_1": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="regime labels"):
        select_sensors(train, ["s_1"], regimes=[0, 1])
