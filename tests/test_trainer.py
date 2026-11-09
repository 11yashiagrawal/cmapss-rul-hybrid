"""Tests for the shared training loop."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from rul.models.base import SequenceEncoder, SequenceRegressor
from rul.training.trainer import FitResult, TrainConfig, fit, predict, set_seed

WINDOW, FEATURES = 10, 3


class MeanEncoder(SequenceEncoder):
    """Tiny stand-in encoder: average over time, then one linear layer with dropout."""

    def __init__(self, n_features: int = FEATURES, output_dim: int = 8) -> None:
        super().__init__(output_dim)
        self.proj = nn.Sequential(nn.Linear(n_features, output_dim), nn.Tanh(), nn.Dropout(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=1))


def make_model(seed: int) -> SequenceRegressor:
    set_seed(seed)
    return SequenceRegressor(MeanEncoder())


def synthetic(n: int = 256, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Windows whose target is a linear function of the time-averaged features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, WINDOW, FEATURES)).astype(np.float32)
    mean = X.mean(axis=1)
    y = (60 + 40 * mean[:, 0] - 25 * mean[:, 1]).astype(np.float32)
    return X, y


FAST = TrainConfig(epochs=8, batch_size=32, learning_rate=1e-2, patience=3, seed=0)


def test_same_seed_gives_the_same_model():
    X, y = synthetic()
    runs = []
    for _ in range(2):
        model = make_model(0)
        result = fit(model, X, y, FAST, target_scale=100)
        runs.append((result.history, predict(model, X, target_scale=100)))
    assert runs[0][0] == runs[1][0]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


def test_different_seed_gives_a_different_model():
    X, y = synthetic()
    predictions = []
    for seed in (0, 1):
        model = make_model(seed)
        fit(model, X, y, TrainConfig(**{**FAST.__dict__, "seed": seed}), target_scale=100)
        predictions.append(predict(model, X, target_scale=100))
    assert not np.allclose(predictions[0], predictions[1])


def test_loss_falls_on_a_small_synthetic_problem():
    X, y = synthetic(512)
    model = make_model(0)
    result = fit(model, X, y, TrainConfig(epochs=30, batch_size=32, learning_rate=1e-2, seed=0),
                 target_scale=100)
    losses = [h["train_loss"] for h in result.history]
    assert losses[-1] < 0.2 * losses[0]
    rmse = float(np.sqrt(np.mean((predict(model, X, target_scale=100) - y) ** 2)))
    assert rmse < y.std() / 2


def test_without_validation_trains_for_the_requested_epochs():
    X, y = synthetic()
    result = fit(make_model(0), X, y, FAST, epochs=5)
    assert isinstance(result, FitResult)
    assert [h["epoch"] for h in result.history] == [1, 2, 3, 4, 5]
    assert result.best_epoch == 5
    assert all(h["val_rmse"] is None for h in result.history)
    assert result.seconds > 0


def test_early_stopping_keeps_the_best_epoch():
    X, y = synthetic(256, seed=0)
    X_val, y_val = synthetic(128, seed=1)
    model = make_model(0)
    config = TrainConfig(epochs=40, batch_size=32, learning_rate=5e-2, patience=2, seed=0)
    result = fit(model, X, y, config, validation=(X_val, y_val), target_scale=100)

    val = [h["val_rmse"] for h in result.history]
    assert result.best_epoch == int(np.argmin(val)) + 1
    # stops `patience` epochs after the best one, unless it ran out of epochs first
    assert len(val) in (result.best_epoch + config.patience, config.epochs)
    # the returned weights are the best epoch's, not the last epoch's
    rmse = float(np.sqrt(np.mean((predict(model, X_val, target_scale=100) - y_val) ** 2)))
    assert rmse == pytest.approx(min(val), rel=1e-5)


def test_validation_rmse_is_reported_in_cycles():
    X, y = synthetic()
    model = make_model(0)
    result = fit(model, X, y, FAST, validation=(X, y), target_scale=100, epochs=1)
    rmse = float(np.sqrt(np.mean((predict(model, X, target_scale=100) - y) ** 2)))
    assert result.history[0]["val_rmse"] == pytest.approx(rmse, rel=1e-5)


def test_accepts_several_input_arrays():
    class TwoInputs(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seq = nn.Linear(FEATURES, 1)
            self.feat = nn.Linear(2, 1)

        def forward(self, x, f):
            return (self.seq(x.mean(dim=1)) + self.feat(f)).squeeze(-1)

    X, y = synthetic()
    extra = np.random.default_rng(0).normal(size=(len(X), 2)).astype(np.float32)
    set_seed(0)
    model = TwoInputs()
    fit(model, (X, extra), y, FAST, epochs=2)
    assert predict(model, (X, extra)).shape == (len(X),)


def test_rejects_mismatched_lengths_and_unknown_options():
    X, y = synthetic()
    with pytest.raises(ValueError, match="rows"):
        fit(make_model(0), X, y[:-1], FAST)
    with pytest.raises(ValueError, match="scheduler"):
        TrainConfig(scheduler="sometimes")
    with pytest.raises(ValueError, match="loss"):
        TrainConfig(loss="hinge")
    with pytest.raises(TypeError):
        TrainConfig.from_dict({"epochs": 3, "learning_rat": 0.1})


def test_config_from_dict_keeps_values():
    config = TrainConfig.from_dict({"epochs": 3, "learning_rate": 0.01, "grad_clip": None})
    assert (config.epochs, config.learning_rate, config.grad_clip) == (3, 0.01, None)


@pytest.mark.parametrize("noise", [-0.1, float("nan"), float("inf")])
def test_input_noise_rejects_invalid_standard_deviation(noise):
    with pytest.raises(ValueError, match="input_noise_std"):
        TrainConfig(input_noise_std=noise)


def test_input_noise_is_seeded_and_does_not_change_the_source_arrays():
    X, y = synthetic()
    original = X.copy()
    config = replace(FAST, input_noise_std=0.1)
    predictions = []
    for _ in range(2):
        model = make_model(0)
        fit(model, X, y, config, target_scale=100)
        predictions.append(predict(model, X, target_scale=100))
    np.testing.assert_array_equal(predictions[0], predictions[1])
    np.testing.assert_array_equal(X, original)
    clean = make_model(0)
    fit(clean, X, y, FAST, target_scale=100)
    assert not np.allclose(predictions[0], predict(clean, X, target_scale=100))


def test_input_noise_reaches_training_only_and_supports_multiple_inputs():
    class RecordInputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.seen = []

        def forward(self, windows, extra):
            self.seen.append((self.training, windows.detach().clone(), extra.detach().clone()))
            return (windows.mean((1, 2)) + extra.mean(1)) * self.weight

    X = np.zeros((64, WINDOW, FEATURES), dtype=np.float32)
    extra = np.zeros((64, 8), dtype=np.float32)
    model = RecordInputs()
    config = TrainConfig(epochs=1, batch_size=64, input_noise_std=0.2)
    fit(model, (X, extra), np.ones(64), config,
        validation=((X + 2, extra + 3), np.ones(64)))
    training, noisy_windows, noisy_extra = model.seen[0]
    assert training
    assert noisy_windows.std().item() == pytest.approx(0.2, rel=0.1)
    assert noisy_extra.std().item() == pytest.approx(0.2, rel=0.1)
    for training, windows, features in model.seen[1:]:
        assert not training
        assert torch.all(windows == 2) and torch.all(features == 3)
    assert not X.any() and not extra.any()
