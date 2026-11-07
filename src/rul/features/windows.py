"""Fixed-length sensor windows for sequence models, built inside each engine.

A window ending at cycle ``t`` holds the ``window`` cycles up to and including
``t`` of one engine, so it never sees the future or another engine. Its label
is the RUL at ``t``. Cycles with too little history are left-padded by
repeating the engine's first cycle, so every cycle (including the last cycle of
a short test engine) gets a window.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset

from rul.features.cleaning import select_sensors
from rul.features.regimes import RegimeNormalizer


class SequenceWindows(NamedTuple):
    """Windows, their labels and the cycle each window ends at."""

    X: np.ndarray  # [n_windows, window, features], float32
    y: np.ndarray | None  # [n_windows], float32, or None without a target
    index: pd.DataFrame  # unit, cycle (window end) and row (label in the input frame)


def make_windows(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    window: int = 30,
    stride: int = 1,
    target_column: str | None = "rul",
    pad: bool = True,
) -> SequenceWindows:
    """Cut each engine into fixed-length windows that end at its own cycles.

    Args:
        frame: Cycles with ``unit``, ``cycle`` and ``feature_columns``.
        feature_columns: Columns to put in each window, in this order.
        window: Number of cycles per window.
        stride: Step between window ends. Steps are counted back from each
            engine's last cycle, so the last cycle always gets a window.
        target_column: Label column, read at each window's last cycle; ``None``
            builds windows without labels (prediction).
        pad: Left-pad short histories by repeating the engine's first cycle.
            With ``False``, cycles with fewer than ``window`` cycles of history
            get no window.

    Returns:
        :class:`SequenceWindows`, ordered by ``unit`` then ``cycle``.
    """
    if window < 1 or stride < 1:
        raise ValueError(f"window and stride must be positive, got {window} and {stride}")
    feature_columns = list(feature_columns)
    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    windows, labels, index = [], [], []
    for unit, engine in ordered.groupby("unit", sort=False):
        values = engine[feature_columns].to_numpy(dtype=np.float32)
        n = len(values)
        ends = np.arange(n - 1, -1, -stride)[::-1]
        if pad:
            values = np.concatenate([np.repeat(values[:1], window - 1, axis=0), values])
        else:
            ends = ends[ends >= window - 1] - (window - 1)
        if ends.size == 0:
            continue
        # Row i of the view is the window that ends at padded row i + window - 1.
        view = sliding_window_view(values, window, axis=0).transpose(0, 2, 1)
        windows.append(view[ends])
        positions = ends if pad else ends + window - 1
        if target_column is not None:
            labels.append(engine[target_column].to_numpy(dtype=np.float32)[positions])
        index.append(
            pd.DataFrame(
                {
                    "unit": unit,
                    "cycle": engine["cycle"].to_numpy()[positions],
                    "row": engine.index.to_numpy()[positions],
                }
            )
        )
    if not windows:
        raise ValueError("no engine has enough cycles for one window; use pad=True")
    X = np.ascontiguousarray(np.concatenate(windows), dtype=np.float32)
    y = np.concatenate(labels) if target_column is not None else None
    return SequenceWindows(X, y, pd.concat(index, ignore_index=True))


class WindowDataset(Dataset):
    """PyTorch dataset of ``(window, target)`` pairs, windows shaped ``[window, features]``."""

    def __init__(self, windows: SequenceWindows) -> None:
        if windows.y is None:
            raise ValueError("WindowDataset needs labelled windows (target_column set)")
        self.X = torch.from_numpy(windows.X)
        self.y = torch.from_numpy(windows.y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


class WindowPreprocessor(TransformerMixin, BaseEstimator):
    """Fold-safe sensor selection and regime scaling for window inputs.

    Fit on one fold's training engines only; transform any frame with those
    statistics.

    Args:
        n_regimes: Operating regimes for :class:`RegimeNormalizer` (1 on FD001).
        include_cycle: Add the cycle index as an input, z-scored with
            training statistics (an ablation in #69).
        seed: Seed for the regime clustering.
    """

    def __init__(self, n_regimes: int = 1, include_cycle: bool = False, seed: int = 42) -> None:
        self.n_regimes = n_regimes
        self.include_cycle = include_cycle
        self.seed = seed

    def fit(self, train: pd.DataFrame, y: object = None) -> WindowPreprocessor:
        """Learn the kept sensors and their per-regime statistics from training rows."""
        self.sensors_ = select_sensors(train)
        self.normalizer_ = RegimeNormalizer(self.n_regimes, self.seed).fit(train)
        self.cycle_mean_ = float(train["cycle"].mean())
        self.cycle_std_ = float(train["cycle"].std(ddof=0)) or 1.0
        self.feature_columns_ = [*self.sensors_, *(["cycle_z"] if self.include_cycle else [])]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` with scaled sensors (and ``cycle_z`` if enabled) added."""
        check_is_fitted(self, "feature_columns_")
        scaled = self.normalizer_.transform(frame)
        if self.include_cycle:
            scaled["cycle_z"] = (scaled["cycle"] - self.cycle_mean_) / self.cycle_std_
        return scaled


class SequencePredictor:
    """Adapter with ``predict(frame)``, so sequence models plug into ``predict_test_cycles``.

    Builds one padded window per cycle (stride 1) and returns predictions in
    the row order of ``frame``.

    Args:
        model: A ``torch.nn.Module`` mapping ``[batch, window, features]`` to ``[batch]``.
        preprocessor: A fitted :class:`WindowPreprocessor`.
        window: The window length the model was trained with.
        batch_size: Windows per forward pass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor: WindowPreprocessor,
        window: int = 30,
        batch_size: int = 1024,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.window = window
        self.batch_size = batch_size

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict RUL for every row of ``frame``, in its row order."""
        if not frame.index.is_unique:
            raise ValueError("frame needs a unique index to return predictions in row order")
        windows = make_windows(
            self.preprocessor.transform(frame),
            self.preprocessor.feature_columns_,
            window=self.window,
            target_column=None,
        )
        self.model.eval()
        with torch.no_grad():
            X = torch.from_numpy(windows.X)
            out = torch.cat(
                [self.model(X[i : i + self.batch_size]) for i in range(0, len(X), self.batch_size)]
            )
        predictions = pd.Series(out.numpy().astype(float), index=windows.index["row"].to_numpy())
        return predictions.loc[frame.index].to_numpy()
