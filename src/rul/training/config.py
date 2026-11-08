"""YAML run configs: load, override and turn into cross-validation runs.

A model config starts with ``base: dl_base.yaml`` and adds its ``name`` and
``model`` section; everything else (target cap, folds, inputs, optimiser)
comes from the shared base, so every model is trained the same way.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

from rul.data.labels import add_train_rul
from rul.models.registry import build_model
from rul.training.inputs import HybridInputs, WindowInputs
from rul.models.registry import _resolve
from rul.training.runs import CVResult, cross_validate, cross_validate_estimator, refit
from rul.training.trainer import TrainConfig

INPUTS: dict[str, Callable[..., WindowInputs]] = {"windows": WindowInputs, "hybrid": HybridInputs}

# Model types that fit on frames themselves instead of through the network trainer.
ESTIMATORS = {"hybrid_two_stage": "rul.models.hybrid:TwoStageHybrid"}


def _merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict:
    """Read a YAML config, merging in its ``base`` file (relative to it) first."""
    path = Path(path)
    config = yaml.safe_load(path.read_text()) or {}
    base = config.pop("base", None)
    if base is None:
        return config
    return _merge(load_config(path.parent / base), config)


def apply_overrides(config: dict, overrides: Iterable[str]) -> dict:
    """Apply ``section.key=value`` overrides; values are parsed as YAML."""
    config = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"overrides look like key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        *parents, leaf = dotted.split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return config


@dataclass
class RunParts:
    """Everything :func:`rul.training.runs.cross_validate` and ``refit`` need."""

    make_model: Callable
    make_inputs: Callable[[], WindowInputs]
    train_config: TrainConfig
    cv_kwargs: dict


def run_parts(config: dict) -> RunParts:
    """Turn a loaded config into model/input factories and run settings."""
    inputs = dict(config.get("inputs") or {})
    kind = inputs.pop("type", "windows")
    cv = config.get("cv") or {}
    return RunParts(
        make_model=partial(build_model, config["model"]),
        make_inputs=partial(INPUTS[kind], **inputs),
        train_config=TrainConfig.from_dict(config.get("training") or {}),
        cv_kwargs={
            "cap": (config.get("target") or {}).get("cap"),
            "selection_max_rul": (config.get("selection") or {}).get("max_true_rul"),
            **cv,
        },
    )


def is_estimator(config: dict) -> bool:
    """Whether the config's model fits on frames itself (e.g. the two-stage hybrid)."""
    return config["model"]["type"] in ESTIMATORS


def make_estimator(config: dict) -> Callable[[], object]:
    """Factory for a frame-level model, sharing the config's trainer, inputs and folds."""
    params = dict(config["model"])
    cls = _resolve(ESTIMATORS[params.pop("type")])
    inputs = dict(config.get("inputs") or {})
    inputs.pop("type", None)
    cv = config.get("cv") or {}
    train_config = TrainConfig.from_dict(config.get("training") or {})
    return partial(
        cls, train_config=train_config, inputs=inputs, seed=train_config.seed,
        holdout_size=cv.get("holdout_size", 0.2), split_seed=cv.get("split_seed", 42), **params,
    )


def run_cv(config: dict, train: pd.DataFrame) -> CVResult:
    """Cross-validate the configured model on ``train``."""
    parts = run_parts(config)
    if is_estimator(config):
        kwargs = {k: v for k, v in parts.cv_kwargs.items() if k != "holdout_size"}
        return cross_validate_estimator(train, make_estimator(config), **kwargs)
    return cross_validate(
        train, parts.make_model, parts.make_inputs, parts.train_config, **parts.cv_kwargs
    )


def run_refit(config: dict, train: pd.DataFrame, epochs: int):
    """Train the configured model on every training engine.

    Networks train for ``epochs``; frame-level models (the two-stage hybrid)
    fit their own way. Returns ``(model, seconds)``; the model has ``predict(frame)``.
    """
    parts = run_parts(config)
    cap = parts.cv_kwargs.get("cap")
    if is_estimator(config):
        model = make_estimator(config)().fit(add_train_rul(train, cap=cap))
        return model, model.fit_seconds_
    trained, result = refit(
        train, parts.make_model, parts.make_inputs, parts.train_config, cap=cap, epochs=epochs
    )
    return trained, result.seconds
