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
