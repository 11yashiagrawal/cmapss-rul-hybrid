"""Build sequence models by name, so runs can be described in YAML configs.

``MODELS`` maps a config ``type`` to either an import path (``"module:Name"``)
or the object itself:

* a :class:`~rul.models.base.SequenceEncoder` subclass is built with
  ``n_features`` plus its config parameters and wrapped in a
  :class:`~rul.models.base.SequenceRegressor` with the config's ``head``;
* anything else is a builder called with the input sizes, the parameters and
  ``head``, returning a complete model (used by the hybrids).

Each model module adds itself here in the PR that introduces it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from torch import nn

from rul.models.base import RegressionHead, SequenceEncoder, SequenceRegressor

MODELS: dict[str, str | type | Callable[..., nn.Module]] = {
    "lstm": "rul.models.lstm:LSTMEncoder",
    "tcn": "rul.models.tcn:TCNEncoder",
}


def register_model(name: str, target: str | type | Callable[..., nn.Module]) -> None:
    """Make ``target`` available as config ``type: name``."""
    MODELS[name] = target


def _resolve(target):
    if isinstance(target, str):
        module, _, attr = target.partition(":")
        return getattr(importlib.import_module(module), attr)
    return target


def build_model(spec: dict, input_dims: dict[str, int]) -> nn.Module:
    """Build a fresh model from a config ``model`` section.

    Args:
        spec: ``{"type": name, "head": {...}, **params}``; ``head`` holds the
            :class:`RegressionHead` options (``hidden_dim``, ``dropout``).
        input_dims: Sizes from the fitted input builder, e.g. ``{"n_features": 15}``.
    """
    params = dict(spec)
    name = params.pop("type")
    head = params.pop("head", None) or {}
    if name not in MODELS:
        raise KeyError(f"unknown model type {name!r}; known: {sorted(MODELS)}")
    target = _resolve(MODELS[name])
    if isinstance(target, type) and issubclass(target, SequenceEncoder):
        encoder = target(n_features=input_dims["n_features"], **params)
        return SequenceRegressor(encoder, RegressionHead(encoder.output_dim, **head))
    return target(**input_dims, **params, head=head)
