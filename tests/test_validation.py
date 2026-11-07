"""Tests for seed aggregation, bootstrap CIs, paired tests and the value test."""

import numpy as np
import pytest

from rul.evaluation.metrics import evaluate
from rul.evaluation.validation import aggregate_seeds

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
