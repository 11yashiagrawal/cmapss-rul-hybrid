"""Tests for rul.data.load."""

from pathlib import Path

import pytest

from rul.data.load import DATA_DIR, load_subset

# First row of train_FD001.txt after the unit and cycle columns, including the
# trailing double space the raw files end every line with.
RAW_TAIL = (
    "-0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 "
    "2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 "
    "100.00 39.06 23.4190  "
)
EXPECTED_COLUMNS = ["unit", "cycle", "op_1", "op_2", "op_3"] + [f"s_{i}" for i in range(1, 22)]


def write_subset(folder: Path, train_cycles: dict, test_cycles: dict, rul: list, name: str = "FD001") -> Path:
    """Write a tiny fake subset: {unit: n_cycles} per split, plus a RUL file."""

    def lines(cycles: dict) -> str:
        return "".join(f"{u} {c} {RAW_TAIL}\n" for u, n in cycles.items() for c in range(1, n + 1))

    (folder / f"train_{name}.txt").write_text(lines(train_cycles))
    (folder / f"test_{name}.txt").write_text(lines(test_cycles))
    # Real RUL files also end each line with a trailing space.
    (folder / f"RUL_{name}.txt").write_text("".join(f"{r} \n" for r in rul))
    return folder


def test_cycles_are_parsed_into_named_columns(tmp_path):
    write_subset(tmp_path, train_cycles={1: 3, 2: 2}, test_cycles={1: 2, 2: 1}, rul=[112, 98])

    train, test, _ = load_subset("FD001", data_dir=tmp_path)

    assert list(train.columns) == EXPECTED_COLUMNS
    assert list(test.columns) == EXPECTED_COLUMNS
    assert train["unit"].tolist() == [1, 1, 1, 2, 2]
    assert train["cycle"].tolist() == [1, 2, 3, 1, 2]
    assert train["unit"].dtype == "int64" and train["cycle"].dtype == "int64"
    assert train.loc[0, "op_3"] == 100.0
    assert train.loc[0, "s_1"] == 518.67
    assert train.loc[0, "s_21"] == pytest.approx(23.419)
    assert len(test) == 3


def test_rul_targets_are_indexed_by_test_unit(tmp_path):
    write_subset(tmp_path, train_cycles={1: 3, 2: 2}, test_cycles={1: 2, 2: 1}, rul=[112, 98])

    _, _, rul_test = load_subset("FD001", data_dir=tmp_path)

    assert rul_test.to_dict() == {1: 112, 2: 98}
    assert rul_test.index.name == "unit"


@pytest.mark.parametrize("bad_name", ["FD005", "fd001", "train_FD001", ""])
def test_unknown_subset_name_is_rejected(tmp_path, bad_name):
    with pytest.raises(ValueError, match="FD001"):
        load_subset(bad_name, data_dir=tmp_path)


def test_missing_data_file_error_names_the_file_and_the_fix(tmp_path):
    write_subset(tmp_path, train_cycles={1: 3}, test_cycles={1: 2}, rul=[112])
    (tmp_path / "test_FD001.txt").unlink()

    with pytest.raises(FileNotFoundError, match="test_FD001.txt") as excinfo:
        load_subset("FD001", data_dir=tmp_path)

    assert "download" in str(excinfo.value).lower()


def test_rul_file_must_have_one_value_per_test_engine(tmp_path):
    write_subset(tmp_path, train_cycles={1: 3}, test_cycles={1: 2, 2: 1, 3: 4}, rul=[112, 98])

    with pytest.raises(ValueError, match="2 RUL values for 3 test engines"):
        load_subset("FD001", data_dir=tmp_path)


def test_load_training_does_not_require_or_read_official_test_files(tmp_path):
    from rul.data.load import load_training

    write_subset(tmp_path, train_cycles={1: 3}, test_cycles={1: 2}, rul=[112])
    (tmp_path / "test_FD001.txt").unlink()
    (tmp_path / "RUL_FD001.txt").unlink()
    train = load_training("FD001", data_dir=tmp_path)
    assert list(train.columns) == EXPECTED_COLUMNS
    assert train["cycle"].tolist() == [1, 2, 3]


def test_load_training_rejects_unknown_subset(tmp_path):
    from rul.data.load import load_training

    with pytest.raises(ValueError, match="FD001"):
        load_training("fd001", data_dir=tmp_path)


# Counted from the raw files with `wc -l` and `awk '{print $1}' | sort -u`.
# Note: the dataset readme swaps the FD004 engine counts; the files have
# 249 training and 248 test engines.
REAL_COUNTS = {
    # subset: (train rows, train engines, test rows, test engines)
    "FD001": (20631, 100, 13096, 100),
    "FD002": (53759, 260, 33991, 259),
    "FD003": (24720, 100, 16596, 100),
    "FD004": (61249, 249, 41214, 248),
}


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="raw C-MAPSS data not downloaded")
@pytest.mark.parametrize("subset", REAL_COUNTS)
def test_real_subsets_match_raw_file_counts(subset):
    train_rows, train_units, test_rows, test_units = REAL_COUNTS[subset]

    train, test, rul_test = load_subset(subset)

    assert (len(train), train["unit"].nunique()) == (train_rows, train_units)
    assert (len(test), test["unit"].nunique()) == (test_rows, test_units)
    assert len(rul_test) == test_units
    assert not train.isna().any().any() and not test.isna().any().any()
