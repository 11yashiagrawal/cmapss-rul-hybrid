"""Load the NASA C-MAPSS turbofan degradation files into DataFrames.

Each subset (FD001-FD004) ships as three whitespace-separated text files:
``train_FDxxx.txt`` (engines run to failure), ``test_FDxxx.txt`` (engines cut
off some time before failure) and ``RUL_FDxxx.txt`` (the true remaining useful
life at the last recorded cycle of each test engine).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

SETTING_COLS = [f"op_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]

# src/rul/data/load.py -> repo root is three folders up from this file's folder.
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "CMAPSSData"


class CMAPSSData(NamedTuple):
    """One C-MAPSS subset. Unpacks as ``train, test, rul_test = ...``."""

    train: pd.DataFrame
    test: pd.DataFrame
    rul_test: pd.Series


def load_training(subset: str, data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load run-to-failure cycles without opening official test or RUL files."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {', '.join(SUBSETS)}")
    return _read_cycles(Path(data_dir) / f"train_{subset}.txt")


def load_subset(subset: str, data_dir: str | Path = DATA_DIR) -> CMAPSSData:
    """Load the train, test and true-RUL files of one C-MAPSS subset.

    Args:
        subset: One of :data:`SUBSETS`, e.g. ``"FD001"``.
        data_dir: Folder holding the raw ``.txt`` files.

    Returns:
        ``CMAPSSData(train, test, rul_test)``. ``train`` and ``test`` have one
        row per engine cycle, with the columns in :data:`COLUMNS`. ``rul_test``
        is the true RUL at each test engine's last cycle, indexed by ``unit``.
    """
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {', '.join(SUBSETS)}")
    data_dir = Path(data_dir)
    train = _read_cycles(data_dir / f"train_{subset}.txt")
    test = _read_cycles(data_dir / f"test_{subset}.txt")
    rul_test = _read_rul(data_dir / f"RUL_{subset}.txt")

    # RUL_FDxxx.txt has one line per test engine, in unit order.
    test_units = sorted(test["unit"].unique())
    if len(rul_test) != len(test_units):
        raise ValueError(
            f"RUL_{subset}.txt has {len(rul_test)} RUL values for {len(test_units)} test engines"
        )
    rul_test.index = pd.Index(test_units, name="unit")
    return CMAPSSData(train, test, rul_test)


def _read_cycles(path: Path) -> pd.DataFrame:
    """Read one train/test file: 26 space-separated columns, no header."""
    df = pd.read_csv(_require_file(path), sep=r"\s+", header=None, names=COLUMNS)
    return df.astype({"unit": "int64", "cycle": "int64"})


def _read_rul(path: Path) -> pd.Series:
    """Read an RUL file: one integer per line, one line per test engine."""
    return pd.read_csv(_require_file(path), header=None, names=["rul"])["rul"].astype("int64")


def _require_file(path: Path) -> Path:
    """Fail early with instructions when a raw data file is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. The raw data is not committed to git: download "
            f"the NASA C-MAPSS dataset and extract its .txt files into {path.parent}"
        )
    return path
