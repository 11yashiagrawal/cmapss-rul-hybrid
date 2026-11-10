"""Select sensors using training rows only."""

from collections.abc import Sequence
from math import isfinite

import numpy as np
import pandas as pd

from rul.data.load import SENSOR_COLS


def select_sensors(
    train: pd.DataFrame,
    sensors: Sequence[str] = SENSOR_COLS,
    min_relative_std: float = 0.0,
    regimes: Sequence | np.ndarray | None = None,
) -> list[str]:
    """Return training sensors whose relative standard deviation exceeds the cutoff.

    Args:
        train: Training rows.
        sensors: Candidate sensor columns.
        min_relative_std: Cutoff on ``std / max(|mean|, 1)``.
        regimes: Optional operating-regime label for each row of ``train``. A
            sensor is then kept only if it varies *within* at least one regime.
            On multi-regime subsets (FD002, FD004) some sensors change only
            with the operating condition and are flat inside every regime, so
            they carry no degradation signal once each regime is scaled on its
            own. With one regime (FD001) this gives the same choice as
            ``regimes=None``.
    """
    if not isfinite(min_relative_std) or min_relative_std < 0:
        raise ValueError("min_relative_std must be finite and nonnegative")
    if regimes is None:
        groups = [train]
    else:
        labels = np.asarray(regimes)
        if labels.shape != (len(train),):
            raise ValueError(f"got {labels.size} regime labels for {len(train)} training rows")
        groups = [part for _, part in train.groupby(labels, sort=True)]

    def varies(values: pd.Series) -> bool:
        return (
            values.nunique(dropna=False) > 1
            and values.std(ddof=0) / max(abs(values.mean()), 1.0) > min_relative_std
        )

    kept = [sensor for sensor in sensors if any(varies(part[sensor]) for part in groups)]
    if not kept:
        raise ValueError("No sensors exceeded the training variability cutoff")
    return kept
