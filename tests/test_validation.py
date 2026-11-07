"""Tests for seed aggregation, bootstrap CIs, paired tests and the value test."""

import numpy as np
import pytest

from rul.evaluation.metrics import evaluate
from rul.evaluation.validation import (
    aggregate_seeds,
    bootstrap_ci,
    paired_bootstrap_diff,
)

TRUE = np.linspace(10, 150, 60)


def test_aggregate_seeds_gives_mean_and_sample_std():
    runs = [evaluate(TRUE, TRUE + offset) for offset in (2.0, 4.0, 6.0)]
    summary = aggregate_seeds(runs)
    assert summary["n_seeds"] == 3
    assert summary["rmse"]["mean"] == pytest.approx(4.0)
    assert summary["rmse"]["std"] == pytest.approx(2.0)
    assert summary["by_stage"]["late"]["mae"]["mean"] == pytest.approx(4.0)


def test_aggregate_single_seed_has_zero_std():
    assert aggregate_seeds([evaluate(TRUE, TRUE + 3)])["mae"] == {"mean": 3.0, "std": 0.0}


def test_bootstrap_ci_is_exact_for_constant_error():
    ci = bootstrap_ci(TRUE, TRUE + 5, n_boot=500)
    assert ci == pytest.approx({"estimate": 5.0, "low": 5.0, "high": 5.0})


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    ci = bootstrap_ci(TRUE, TRUE + rng.normal(0, 10, TRUE.size), "mae", n_boot=2000)
    assert ci["low"] < ci["estimate"] < ci["high"]


def test_bootstrap_averages_rmse_over_seeds():
    preds = np.stack([TRUE + 2, TRUE + 6])
    assert bootstrap_ci(TRUE, preds, n_boot=200)["estimate"] == pytest.approx(4.0)


def test_paired_difference_with_known_answer():
    diff = paired_bootstrap_diff(TRUE, TRUE + 1, TRUE - 5, n_boot=500)
    assert diff["difference"] == pytest.approx(-4.0)
    assert diff["low"] == pytest.approx(-4.0) and diff["high"] == pytest.approx(-4.0)
    assert diff["excludes_zero"]


def test_paired_difference_of_identical_models_is_zero():
    pred = TRUE + np.random.default_rng(1).normal(0, 8, TRUE.size)
    diff = paired_bootstrap_diff(TRUE, pred, pred, n_boot=500)
    assert diff["difference"] == 0 and not diff["excludes_zero"]


def test_rejects_mismatched_engine_counts():
    with pytest.raises(ValueError):
        bootstrap_ci(TRUE, TRUE[:-1])
