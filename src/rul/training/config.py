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

from rul.models.registry import build_model
from rul.training.inputs import WindowInputs
from rul.training.runs import CVResult, cross_validate
from rul.training.trainer import TrainConfig

INPUTS: dict[str, Callable[..., WindowInputs]] = {"windows": WindowInputs}


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


def run_cv(config: dict, train: pd.DataFrame) -> CVResult:
    """Cross-validate the configured model on ``train``."""
    parts = run_parts(config)
    return cross_validate(
        train, parts.make_model, parts.make_inputs, parts.train_config, **parts.cv_kwargs
    )
