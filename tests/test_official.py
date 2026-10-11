"""Tests for rul.evaluation.official."""

import json
from pathlib import Path

import pandas as pd
import pytest

from rul.data.load import DATA_DIR, load_subset
from rul.evaluation.official import METRICS_DIR, predict_test_cycles, save_results, score_last_cycle


def cycles(spec: dict) -> pd.DataFrame:
    """Frame with one row per (unit, cycle): {unit: [cycles in row order]}."""
    rows = [(unit, c) for unit, cs in spec.items() for c in cs]
    return pd.DataFrame(rows, columns=["unit", "cycle"])


def rul_file(values: dict) -> pd.Series:
    """True RUL at the last test cycle, shaped like load_subset's rul_test."""
    return pd.Series(values, name="rul").rename_axis("unit")


class CountdownModel:
    """Predicts 120 - cycle and remembers the frame it was asked about."""

    def predict(self, frame: pd.DataFrame) -> list[float]:
        self.seen = frame
        return (120 - frame["cycle"]).tolist()


def test_every_test_cycle_gets_its_prediction_next_to_its_true_rul():
    # Engine 1 stops at cycle 3 with 112 cycles left; engine 2 at cycle 2 with 98.
    test = cycles({1: [1, 2, 3], 2: [1, 2]})

    result = predict_test_cycles(CountdownModel(), test, rul_file({1: 112, 2: 98}))

    assert result.values.tolist() == [
        [1, 1, 114, 119],
        [1, 2, 113, 118],
        [1, 3, 112, 117],
        [2, 1, 99, 119],
        [2, 2, 98, 118],
    ]
    assert list(result.columns) == ["unit", "cycle", "rul", "prediction"]


def test_the_model_never_sees_the_true_rul():
    model = CountdownModel()

    predict_test_cycles(model, cycles({1: [1, 2, 3]}), rul_file({1: 112}))

    assert "rul" not in model.seen.columns


def test_the_model_sees_each_engines_whole_history_not_just_its_last_cycle():
    # Rolling features at the last cycle need the cycles before it.
    model = CountdownModel()

    predict_test_cycles(model, cycles({1: [1, 2, 3], 2: [1, 2]}), rul_file({1: 112, 2: 98}))

    assert model.seen[["unit", "cycle"]].values.tolist() == [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2]]


def test_each_engine_is_scored_once_at_its_last_cycle():
    predictions = pd.DataFrame(
        [(1, 1, 114, 100), (1, 2, 113, 100), (1, 3, 112, 117), (2, 1, 99, 0), (2, 2, 98, 118)],
        columns=["unit", "cycle", "rul", "prediction"],
    )

    last, metrics = score_last_cycle(predictions)

    # Only (112 -> 117) and (98 -> 118) count: errors 5 and 20 cycles.
    assert last[["unit", "cycle", "rul", "prediction"]].values.tolist() == [[1, 3, 112, 117], [2, 2, 98, 118]]
    assert metrics["n"] == 2
    assert metrics["mae"] == 12.5
    assert metrics["rmse"] == pytest.approx(14.5774, abs=1e-4)  # sqrt((25 + 400) / 2)
    assert metrics["by_stage"]["early"]["n"] == metrics["by_stage"]["mid"]["n"] == 1  # 112 early, 98 mid


class ConstantModel:
    """Predicts the same RUL for every cycle."""

    def predict(self, frame: pd.DataFrame) -> list[float]:
        return [100.0] * len(frame)


# Expected errors were computed from RUL_FD001.txt with awk, not by rul.
@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="raw C-MAPSS data not downloaded")
def test_scoring_on_real_fd001_uses_one_row_per_test_engine():
    _, test, rul_test = load_subset("FD001")

    last, metrics = score_last_cycle(predict_test_cycles(ConstantModel(), test, rul_test))

    assert last["unit"].tolist() == list(range(1, 101))
    assert last["rul"].tolist() == rul_test.tolist()
    assert metrics["n"] == 100
    assert metrics["rmse"] == pytest.approx(48.230074, abs=1e-6)
    assert metrics["mae"] == pytest.approx(38.06)


def test_results_are_saved_as_json_metrics_and_a_per_engine_csv(tmp_path):
    metrics = {"random_forest": {"rmse": 14.5, "by_stage": {"late": {"n": 0, "rmse": None}}}}
    predictions = pd.DataFrame({"unit": [1, 2], "rul": [112, 98], "random_forest": [117.0, 118.0]})

    paths = save_results("fd001_test", metrics, predictions, out_dir=tmp_path / "metrics")

    assert paths == {
        "metrics": tmp_path / "metrics" / "fd001_test.json",
        "predictions": tmp_path / "metrics" / "fd001_test_predictions.csv",
    }
    assert json.loads(paths["metrics"].read_text()) == metrics
    saved = pd.read_csv(paths["predictions"])
    assert saved.columns.tolist() == ["unit", "rul", "random_forest"]
    assert saved.values.tolist() == [[1, 112, 117.0], [2, 98, 118.0]]


def test_results_default_to_results_metrics_in_the_repo():
    repo_root = Path(__file__).resolve().parents[1]

    assert METRICS_DIR == repo_root / "results" / "metrics"
