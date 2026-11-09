"""Contracts for one-factor experiments and resumable, training-only CV."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rul.training.config import apply_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]


def small_config():
    config = apply_overrides(load_config(ROOT / "configs/regularization.yaml"), [
        "training.epochs=2", "training.batch_size=64", "training.patience=1",
        "model.hidden_size=4", "inputs.window=5", "target.cap=10",
        "selection.max_true_rul=15",
    ])
    config["study"]["variations"] = {"training.input_noise_std": [0.05], "inputs.stride": [2]}
    return config


def engines():
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(1, 10):
        for cycle in range(1, 22 + unit):
            rows.append({
                "unit": unit, "cycle": cycle, "op_1": 0.0, "op_2": 0.0, "op_3": 0.0,
                **{f"s_{i}": 5.0 for i in range(1, 22)},
                "s_2": cycle + rng.normal(), "s_3": -cycle + rng.normal(),
            })
    return pd.DataFrame(rows)


def test_variants_change_exactly_one_factor_and_leave_config_untouched():
    from rul.training.regularization import regularization_variants

    config = load_config(ROOT / "configs/regularization.yaml")
    original = copy.deepcopy(config)
    variants = regularization_variants(config)
    assert len(variants) == 11
    baseline = variants.pop("baseline")
    assert "study" not in baseline
    assert baseline["model"] == load_config(ROOT / "configs/lstm.yaml")["model"]
    for variant in variants.values():
        differences = [
            f"{section}.{key}" for section in ("model", "training", "inputs")
            for key in baseline[section] if baseline[section][key] != variant[section][key]
        ]
        assert len(differences) == 1
        assert variant["cv"] == baseline["cv"]
    assert config == original


def test_variants_reject_unknown_factors_and_deduplicate_baseline():
    from rul.training.regularization import regularization_variants

    config = small_config()
    config["study"]["variations"] = {"training.patience": [1, 1, 2, 2]}
    assert len(regularization_variants(config)) == 2
    config["study"]["variations"] = {"training.seed": [1]}
    with pytest.raises(ValueError, match="factor"):
        regularization_variants(config)


def test_study_preserves_engine_folds_and_uncapped_truth_and_resumes(tmp_path, monkeypatch):
    import rul.training.regularization as study

    train, config = engines(), small_config()
    records, predictions = study.run_regularization(train, config, tmp_path)
    assert len(records) == 6  # three variants, two targets each
    assert len(predictions) == len(train) * 6
    assert predictions["rul"].max() > config["target"]["cap"]
    reference = [f["heldout_units"] for f in records[0]["folds"]]
    for record in records:
        assert [f["heldout_units"] for f in record["folds"]] == reference
        assert record["context"]["training_data_sha256"]
        for fold in record["folds"]:
            fit, hold, scored = map(set, (fold["train_units"], fold["holdout_units"], fold["heldout_units"]))
            assert fit and hold and scored
            assert not (fit & hold or fit & scored or hold & scored)
        part = predictions[(predictions.variant == record["variant"]) &
                           (predictions.target == record["target"])]
        assert not part.duplicated(["unit", "cycle"]).any()
        assert len(part) == len(train)
    table = study.regularization_table(records)
    assert len(table) == 3
    assert np.isfinite(table[["capped_selection_rmse", "uncapped_selection_rmse"]]).all().all()
    assert table.loc[table.variant == "baseline", "delta_from_baseline"].item() == 0

    def no_retraining(*args):
        pytest.fail("matching cached runs should not train again")

    monkeypatch.setattr(study, "run_cv", no_retraining)
    cached, cached_predictions = study.run_regularization(train, config, tmp_path)
    assert records == cached
    pd.testing.assert_frame_equal(predictions, cached_predictions, check_exact=False, rtol=1e-14)


def test_cache_key_changes_with_data_and_config(tmp_path, monkeypatch):
    import rul.training.regularization as study

    train, config = engines(), small_config()
    config["study"]["variations"] = {}
    records, _ = study.run_regularization(train, config, tmp_path)
    changed = train.copy()
    changed.loc[0, "s_2"] += 1
    changed_records, _ = study.run_regularization(changed, config, tmp_path)
    assert records[0]["context"] != changed_records[0]["context"]
    config["training"]["epochs"] = 1
    final, _ = study.run_regularization(changed, config, tmp_path)
    assert final[0]["config"]["training"]["epochs"] == 1
    assert all(len(f["history"]) == 1 for f in final[0]["folds"])

    # Interrupted or edited prediction files must not be trusted as cache hits.
    path = tmp_path / "baseline_capped_predictions.csv"
    path.write_text("broken")
    repaired, predictions = study.run_regularization(changed, config, tmp_path)
    assert len(predictions) == len(train) * 2
    assert repaired[0]["predictions_sha256"] != "broken"
    (tmp_path / "baseline_capped.json").write_text("{interrupted")
    recovered, _ = study.run_regularization(changed, config, tmp_path)
    assert len(recovered[0]["folds"]) == 3
