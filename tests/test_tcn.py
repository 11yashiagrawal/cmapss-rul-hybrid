"""Tests for the temporal convolutional encoder."""

import pytest
import torch

from rul.models.base import SequenceEncoder, SequenceRegressor
from rul.models.registry import build_model
from rul.models.tcn import TCNEncoder, receptive_field

BATCH, WINDOW, FEATURES = 4, 30, 15


@pytest.fixture
def windows() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, WINDOW, FEATURES)


def test_encoder_returns_one_embedding_per_window(windows):
    encoder = TCNEncoder(FEATURES, channels=16, kernel_size=3, levels=4)
    assert isinstance(encoder, SequenceEncoder)
    assert encoder.output_dim == 16
    assert encoder(windows).shape == (BATCH, 16)


def test_config_builds_a_regressor(windows):
    model = build_model({"type": "tcn", "channels": 8, "kernel_size": 3, "levels": 4},
                        {"n_features": FEATURES})
    assert isinstance(model, SequenceRegressor) and isinstance(model.encoder, TCNEncoder)
    assert model(windows).shape == (BATCH,)


def test_backward_pass_reaches_every_parameter(windows):
    model = SequenceRegressor(TCNEncoder(FEATURES, channels=8, kernel_size=3, levels=4, dropout=0.1))
    model(windows).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_convolutions_are_causal(windows):
    encoder = TCNEncoder(FEATURES, channels=8, kernel_size=3, levels=4).eval()
    t = 12
    changed_future = windows.clone()
    changed_future[:, t + 1 :] += 10.0
    before, after = encoder.sequence(windows), encoder.sequence(changed_future)
    assert before.shape == (BATCH, 8, WINDOW)
    torch.testing.assert_close(before[..., : t + 1], after[..., : t + 1])
    assert not torch.allclose(before[..., t + 1 :], after[..., t + 1 :])


@pytest.mark.parametrize(("kernel_size", "levels", "expected"), [(3, 3, 29), (3, 4, 61), (5, 3, 57), (2, 1, 3)])
def test_receptive_field_formula(kernel_size, levels, expected):
    assert receptive_field(kernel_size, levels) == expected
    assert TCNEncoder(FEATURES, channels=4, kernel_size=kernel_size, levels=levels).receptive_field == expected


def test_receptive_field_matches_what_the_last_output_can_see():
    encoder = TCNEncoder(FEATURES, channels=4, kernel_size=3, levels=3).eval()  # field 29
    x = torch.randn(1, 40, FEATURES)
    last = encoder(x)
    outside, inside = x.clone(), x.clone()
    outside[:, 40 - 29 - 1] += 10.0  # one step older than the field
    inside[:, 40 - 29] += 10.0  # the oldest step inside the field
    torch.testing.assert_close(encoder(outside), last)
    assert not torch.allclose(encoder(inside), last)


def test_covers_window():
    assert TCNEncoder(FEATURES, kernel_size=3, levels=4).covers(30)
    assert not TCNEncoder(FEATURES, kernel_size=3, levels=3).covers(30)


def test_rejects_bad_settings():
    for bad in ({"kernel_size": 1}, {"levels": 0}, {"dropout": 1.0}, {"channels": 0}):
        with pytest.raises(ValueError):
            TCNEncoder(FEATURES, **bad)


@pytest.mark.parametrize("dilation", [1, 2, 4])
def test_causal_conv_matches_a_left_padded_convolution(dilation):
    from torch.nn.functional import conv1d, pad

    from rul.models.tcn import CausalConv1d

    layer = CausalConv1d(5, 7, kernel_size=3, dilation=dilation)
    x = torch.randn(2, 5, 30)
    expected = conv1d(pad(x, (2 * dilation, 0)), layer.conv.weight, layer.conv.bias, dilation=dilation)
    torch.testing.assert_close(layer(x), expected)


def test_pointwise_residual_matches_a_1x1_convolution():
    from torch.nn.functional import conv1d

    from rul.models.tcn import TemporalBlock

    block = TemporalBlock(5, 7, kernel_size=3, dilation=1, dropout=0.0)
    x = torch.randn(2, 5, 30)
    expected = conv1d(x, block.residual.weight, block.residual.bias)
    torch.testing.assert_close(block.project_residual(x), expected)
