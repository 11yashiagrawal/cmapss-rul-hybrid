"""Small end-to-end runs of scripts/run_ml_baselines.py on synthetic C-MAPSS files."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rul.data.load import COLUMNS, SENSOR_COLS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ml_baselines.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("run_ml_baselines", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engines(lives, rng, cut=None) -> pd.DataFrame:
    """Run-to-failure engines whose sensors drift as they wear (cut short if ``cut``)."""
    rows = []
    for unit, life in enumerate(lives, start=1):
        stop = life if cut is None else cut[unit - 1]
        for cycle in range(1, stop + 1):
            wear = cycle / life
            sensors = 1.0 + 0.05 * rng.standard_normal(len(SENSOR_COLS)) + wear * np.linspace(-1, 1, 21)
            rows.append([unit, cycle, 0.0, 0.0, 100.0, *sensors])
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture()
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    write = lambda frame, name: frame.to_csv(tmp_path / name, sep=" ", header=False, index=False)
    write(engines([40, 45, 50, 55, 60, 65], rng), "train_FD001.txt")
    lives, cut = [50, 60, 70], [30, 35, 40]
    write(engines(lives, rng, cut), "test_FD001.txt")
    (tmp_path / "RUL_FD001.txt").write_text("".join(f"{l - c}\n" for l, c in zip(lives, cut)))
    return tmp_path


def run(script, data_dir, *extra):
    out = data_dir / "out"
    report = script.main(["--data-dir", str(data_dir), "--out-dir", str(out),
                          "--quick", "--n-splits", "2", *extra])
    return report, out


def test_cross_validation_scores_every_model_variant_and_cap(script, data_dir):
    report, out = run(script, data_dir)

    chosen = {(c["variant"], c["cap"], c["model"]) for c in report["chosen"]}
    assert chosen == {(v, cap, m) for v in ("raw", "engineered") for cap in ("125", "none")
                      for m in ("random_forest", "xgboost", "elastic_net")}
    grid = pd.read_csv(out / "fd001_ml_capped_grid.csv")
    assert len(grid) == 2 * 2 * 2 * 3  # variants x folds x caps x one candidate per model
    saved = json.loads((out / "fd001_ml_capped_cv.json").read_text())
    assert saved["chosen"] == json.loads(json.dumps(report["chosen"]))
    assert not (out / "fd001_ml_capped_test.json").exists()


def test_cross_validation_never_opens_the_official_test_files(script, data_dir):
    (data_dir / "test_FD001.txt").unlink()
    (data_dir / "RUL_FD001.txt").unlink()

    report, _ = run(script, data_dir)

    assert report["chosen"]


def test_a_low_cap_scores_worse_on_every_cycle_than_no_cap(script, data_dir):
    _, out = run(script, data_dir, "--cap", "20")
    grid = pd.read_csv(out / "fd001_ml_capped_grid.csv", dtype={"cap": str})

    capped_rf = grid[(grid["cap"] == "20") & (grid["model"] == "random_forest")]
    uncapped_rf = grid[(grid["cap"] == "none") & (grid["model"] == "random_forest")]
    # Held-out RUL reaches 64, so a forest trained on targets <= 20 misses the high cycles.
    assert capped_rf["all_cycle_rmse"].mean() > uncapped_rf["all_cycle_rmse"].mean()


def test_score_test_saves_one_prediction_per_test_engine(script, data_dir):
    report, out = run(script, data_dir, "--score-test")

    predictions = pd.read_csv(out / "fd001_ml_capped_test_predictions.csv")
    assert len(predictions) == 3
    assert predictions["rul"].tolist() == [20, 25, 30]  # uncapped truth from RUL_FD001.txt
    assert predictions.shape[1] == 3 + 12  # unit, cycle, rul + one column per chosen model
    assert set(report["test"]) == {"raw", "engineered"}
