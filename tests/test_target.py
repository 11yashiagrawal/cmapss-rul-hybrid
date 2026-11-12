"""Tests for rul.data.target (choosing the RUL cap, issue #55)."""

import numpy as np
import pandas as pd
import pytest

from rul.data.load import DATA_DIR
from rul.data.target import (
    HealthIndex,
    estimate_k,
    fit_hinge,
    hinge_onsets,
    selection_rmse,
)


def hinge_engine(unit: int, life: int, onset_rul: int, rng, noise: float = 0.0) -> pd.DataFrame:
    """Synthetic run-to-failure engine whose wear starts ``onset_rul`` cycles before failure.

    ``s_2`` rises with wear and ``s_7`` falls (as on FD001); ``s_1`` stays flat.
    """
    cycle = np.arange(1, life + 1)
    wear = np.maximum(cycle - (life - onset_rul), 0) / onset_rul
    return pd.DataFrame({
        "unit": unit,
        "cycle": cycle,
        "s_1": 518.67,
        "s_2": 642.0 + 1.5 * wear + rng.normal(0, noise, life),
        "s_7": 554.0 - 4.0 * wear + rng.normal(0, noise, life),
    })


def fleet(onsets, life: int = 200, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.concat(
        [hinge_engine(u, life, o, rng, noise) for u, o in enumerate(onsets, start=1)],
        ignore_index=True,
    )


def test_hinge_recovers_a_noiseless_onset_exactly():
    t = np.arange(1, 101)
    y = 2.0 + 0.5 * np.maximum(t - 60, 0)

    fit = fit_hinge(t, y)

    assert fit["onset_cycle"] == 60
    assert fit["level"] == pytest.approx(2.0)
    assert fit["slope"] == pytest.approx(0.5)


def test_hinge_does_not_depend_on_row_order():
    t = np.arange(1, 81)
    y = np.maximum(t - 50, 0.0)
    shuffled = np.random.default_rng(1).permutation(t.size)

    assert fit_hinge(t[shuffled], y[shuffled])["onset_cycle"] == 50


def test_hinge_rejects_falling_series_and_short_engines():
    with pytest.raises(ValueError, match="rising"):
        fit_hinge(np.arange(1, 50), -np.arange(1, 50.0))
    with pytest.raises(ValueError, match="at least"):
        fit_hinge(np.arange(1, 6), np.arange(1, 6.0))


def test_health_index_rises_toward_failure_whatever_the_sensor_direction():
    train = fleet([80, 100])
    index = HealthIndex().fit(train)

    assert index.sign_["s_2"] == 1 and index.sign_["s_7"] == -1
    assert "s_1" not in index.sensors_  # constant, so dropped
    health = index.transform(train)
    for _, engine in health.groupby(train["unit"]):
        assert engine.iloc[-1] > engine.iloc[0]


def test_health_index_uses_only_the_rows_it_was_fitted_on():
    train = fleet([80, 100])
    index = HealthIndex().fit(train)
    before = index.mean_.copy()

    index.transform(fleet([150], seed=5).assign(s_2=1e6))

    pd.testing.assert_series_equal(index.mean_, before)


def test_onset_rul_is_last_cycle_minus_onset_cycle():
    train = fleet([60, 90, 120], noise=0.05)

    onsets = hinge_onsets(train, HealthIndex().fit(train).transform(train))

    assert onsets["onset_rul"].tolist() == pytest.approx([60, 90, 120], abs=5)
    assert (onsets["onset_rul"] == onsets["last_cycle"] - onsets["onset_cycle"]).all()


def test_k_is_the_requested_percentile_of_onset_ruls():
    train = fleet([60, 70, 80, 90, 100, 110, 120, 130, 140, 150])

    assert estimate_k(train, 50) == pytest.approx(105, abs=2)
    ks = estimate_k(train, [50, 90])
    assert set(ks) == {50.0, 90.0} and ks[90.0] > ks[50.0]


def test_selection_rmse_ignores_cycles_above_k():
    y_true = [10, 50, 100, 300]
    y_pred = [13, 46, 100, 125]  # the huge miss is at RUL 300

    assert selection_rmse(y_true, y_pred, k=100) == pytest.approx(np.sqrt((9 + 16 + 0) / 3))
    assert selection_rmse(y_true, y_pred, k=50) == pytest.approx(np.sqrt((9 + 16) / 2))


def test_selection_rmse_keeps_k_itself_and_validates_inputs():
    assert selection_rmse([100], [90], k=100) == pytest.approx(10)
    with pytest.raises(ValueError, match="no held-out"):
        selection_rmse([200, 300], [0, 0], k=100)
    with pytest.raises(ValueError, match="targets"):
        selection_rmse([1, 2], [1], k=10)


@pytest.mark.skipif(not (DATA_DIR / "train_FD001.txt").is_file(), reason="FD001 not downloaded")
def test_fd001_fold_k_matches_the_issue_prototype():
    from rul.data.load import load_training
    from rul.data.split import group_kfold_splits

    train = load_training("FD001")
    ks = [estimate_k(train.iloc[idx]) for idx, _ in group_kfold_splits(train, 3, 42)]

    assert ks == pytest.approx([134, 132, 124], abs=1.5)
