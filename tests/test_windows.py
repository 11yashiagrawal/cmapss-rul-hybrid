"""Tests for engine-bounded sequence windows."""

import numpy as np
import pandas as pd
import pytest
import torch

from rul.data.labels import add_train_rul
from rul.data.load import SENSOR_COLS, SETTING_COLS
from rul.features.windows import WindowDataset, WindowPreprocessor, make_windows


def engines(lengths: dict[int, int], seed: int = 0) -> pd.DataFrame:
    """C-MAPSS-shaped frame: {unit: number of cycles}, with a few varying sensors."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, n in lengths.items():
        for cycle in range(1, n + 1):
            row = {"unit": unit, "cycle": cycle, **{c: 0.0 for c in SETTING_COLS}}
            row.update({s: 5.0 for s in SENSOR_COLS})  # flat sensors get dropped
            row["s_2"] = 600 + cycle * 0.1 + rng.normal()
            row["s_3"] = 1500 - cycle * 0.2 + rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def tagged(lengths: dict[int, int]) -> pd.DataFrame:
    """Frame whose features are the unit and cycle, so windows can be read back."""
    frame = engines(lengths)
    return add_train_rul(frame.assign(unit_f=frame["unit"], cycle_f=frame["cycle"]))


FEATURES = ["unit_f", "cycle_f"]


def test_no_window_spans_two_engines():
    w = make_windows(tagged({1: 40, 2: 35, 3: 12}), FEATURES, window=30)
    assert (w.X[:, :, 0] == w.index["unit"].to_numpy()[:, None]).all()


def test_labels_come_from_the_last_cycle_of_each_window():
    frame = tagged({1: 40, 2: 35})
    w = make_windows(frame, FEATURES, window=10)
    expected = frame.set_index(["unit", "cycle"]).loc[
        list(zip(w.index["unit"], w.index["cycle"])), "rul"
    ]
    np.testing.assert_array_equal(w.y, expected.to_numpy(dtype=np.float32))
    np.testing.assert_array_equal(w.X[:, -1, 1], w.index["cycle"])


def test_windows_never_see_future_cycles():
    w = make_windows(tagged({1: 50, 2: 20}), FEATURES, window=15)
    cycles = w.X[:, :, 1]
    assert (cycles.max(axis=1) == w.index["cycle"].to_numpy()).all()
    assert (np.diff(cycles, axis=1) >= 0).all()


def test_short_engine_is_left_padded_with_its_first_cycle():
    w = make_windows(tagged({7: 3}), FEATURES, window=5)
    np.testing.assert_array_equal(w.X[0, :, 1], [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(w.X[2, :, 1], [1, 1, 1, 2, 3])
    assert len(w.X) == 3


def test_one_window_per_cycle_by_default_and_last_window_ends_at_last_cycle():
    frame = tagged({1: 40, 2: 4, 3: 31})
    w = make_windows(frame, FEATURES, window=30)
    assert w.X.shape == (len(frame), 30, 2)
    last = w.index.groupby("unit")["cycle"].max()
    assert last.to_dict() == {1: 40, 2: 4, 3: 31}


def test_stride_counts_back_from_the_last_cycle():
    w = make_windows(tagged({1: 10}), FEATURES, window=4, stride=3)
    assert w.index["cycle"].tolist() == [1, 4, 7, 10]


def test_without_padding_short_histories_are_skipped():
    w = make_windows(tagged({1: 10, 2: 3}), FEATURES, window=4, pad=False)
    assert w.index["unit"].unique().tolist() == [1]
    assert w.index["cycle"].tolist() == list(range(4, 11))
    np.testing.assert_array_equal(w.X[0, :, 1], [1, 2, 3, 4])


def test_input_row_order_does_not_matter():
    frame = tagged({1: 12, 2: 9})
    shuffled = frame.sample(frac=1, random_state=3)
    a = make_windows(frame, FEATURES, window=5)
    b = make_windows(shuffled, FEATURES, window=5)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)


def test_rejects_bad_window_settings():
    with pytest.raises(ValueError):
        make_windows(tagged({1: 5}), FEATURES, window=0)
    with pytest.raises(ValueError):
        make_windows(tagged({1: 5}), FEATURES, stride=0)


def test_dataset_returns_window_and_target_tensors():
    data = WindowDataset(make_windows(tagged({1: 20, 2: 15}), FEATURES, window=8))
    x, y = data[0]
    assert len(data) == 35
    assert x.shape == (8, 2) and x.dtype == torch.float32
    assert y.shape == () and y.dtype == torch.float32


def test_dataset_needs_labels():
    with pytest.raises(ValueError):
        WindowDataset(make_windows(tagged({1: 5}), FEATURES, target_column=None))


def test_preprocessor_uses_training_engines_only():
    train, test = engines({1: 60, 2: 50}), engines({9: 30}, seed=1)
    test[["s_2", "s_3"]] += 1000  # test drift must not leak into the fitted scaling
    prep = WindowPreprocessor().fit(train)
    assert prep.feature_columns_ == ["s_2", "s_3"]
    out = prep.transform(test)
    mean, std = train["s_2"].mean(), train["s_2"].std(ddof=0)
    np.testing.assert_allclose(out["s_2"], (test["s_2"] - mean) / std)


def test_preprocessor_can_add_a_scaled_cycle_input():
    train = engines({1: 60, 2: 50})
    prep = WindowPreprocessor(include_cycle=True).fit(train)
    out = prep.transform(train)
    assert prep.feature_columns_[-1] == "cycle_z"
    assert abs(out["cycle_z"].mean()) < 1e-9
