"""Turn cycles into model inputs, fitted on training engines only.

An input builder is fitted on one set of training engines and then turns any
frame of cycles into an :class:`InputBatch`: one array per model input, the
labels (if asked for) and the ``unit``/``cycle``/``row`` each window ends at.
The trainer only relies on ``fit``, ``transform`` and ``input_dims``, so the
hybrid can add its engineered-feature input without changing the trainer.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from rul.features.matrix import FEATURE_GROUPS, HybridFeatures, align_to_windows
from rul.features.windows import WindowPreprocessor, make_windows


class InputBatch(NamedTuple):
    """Model inputs for a frame of cycles."""

    inputs: tuple[np.ndarray, ...]  # one array per model input, all with the same rows
    y: np.ndarray | None  # labels at each window's last cycle
    index: pd.DataFrame  # unit, cycle and row (label in the input frame) per window


class WindowInputs:
    """Sensor windows: fold-fitted scaling, then fixed-length windows per engine.

    Args:
        window: Cycles per window.
        stride: Step between training windows. Prediction always uses stride 1,
            so every cycle gets a prediction.
        n_regimes: Operating regimes for regime scaling (1 on FD001).
        include_cycle: Add the z-scored cycle index as an input feature.
        seed: Seed for the regime clustering.
    """

    def __init__(
        self,
        window: int = 30,
        stride: int = 1,
        n_regimes: int = 1,
        include_cycle: bool = False,
        seed: int = 42,
    ) -> None:
        self.window = window
        self.stride = stride
        self.n_regimes = n_regimes
        self.include_cycle = include_cycle
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> WindowInputs:
        """Learn sensor selection and scaling from training engines."""
        self.preprocessor_ = WindowPreprocessor(
            n_regimes=self.n_regimes, include_cycle=self.include_cycle, seed=self.seed
        ).fit(train)
        return self

    @property
    def input_dims(self) -> dict[str, int]:
        """Sizes the model needs: ``{"n_features": ...}``."""
        return {"n_features": len(self.preprocessor_.feature_columns_)}

    def transform(
        self, frame: pd.DataFrame, *, target: str | None = None, stride: int | None = None
    ) -> InputBatch:
        """Windows for ``frame``; ``target`` names the label column (``None`` to predict)."""
        windows = make_windows(
            self.preprocessor_.transform(frame),
            self.preprocessor_.feature_columns_,
            window=self.window,
            stride=self.stride if stride is None else stride,
            target_column=target,
        )
        return InputBatch((windows.X,), windows.y, windows.index)


class HybridInputs(WindowInputs):
    """Sensor windows plus the engineered features at each window's last cycle.

    Both are fitted on the same training engines. The second input is the
    Phase 1 feature set from :class:`rul.features.matrix.HybridFeatures`
    (#57), lined up with the windows by ``(unit, cycle)``.

    Args:
        groups: Feature groups to keep (ablations in #69).
        feature_windows: Rolling windows for the engineered features.
        Other arguments as for :class:`WindowInputs`.
    """

    def __init__(
        self,
        window: int = 30,
        stride: int = 1,
        n_regimes: int = 1,
        include_cycle: bool = False,
        seed: int = 42,
        groups: tuple[str, ...] = FEATURE_GROUPS,
        feature_windows: tuple[int, ...] = (5, 10, 20),
    ) -> None:
        super().__init__(window, stride, n_regimes, include_cycle, seed)
        self.groups = tuple(groups)
        self.feature_windows = tuple(feature_windows)

    def fit(self, train: pd.DataFrame) -> HybridInputs:
        super().fit(train)
        self.features_ = HybridFeatures(
            n_regimes=self.n_regimes, windows=self.feature_windows, groups=self.groups, seed=self.seed
        ).fit(train)
        return self

    @property
    def input_dims(self) -> dict[str, int]:
        return {**super().input_dims, "n_engineered": len(self.features_.output_columns_)}

    def transform(
        self, frame: pd.DataFrame, *, target: str | None = None, stride: int | None = None
    ) -> InputBatch:
        batch = super().transform(frame, target=target, stride=stride)
        features = align_to_windows(self.features_.transform(frame), batch.index)
        return InputBatch((*batch.inputs, features), batch.y, batch.index)
