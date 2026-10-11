# NASA C-MAPSS data

Source: A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation
Simulation Data Set*, NASA Ames Prognostics Center of Excellence.
See section 6 of the [NASA data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
and its [download archive](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip).

## Download

From the repository root, run:

```bash
python3 scripts/download_data.py
```

The download contains a nested `CMAPSSData.zip` archive. The script uses only
Python's standard library and extracts the 12 data files
into `data/raw/CMAPSSData/`, the default path used by `rul.data.load.load_subset`.
It skips downloading when all 12 files already exist. If any file is missing,
it downloads and extracts the dataset again. Paths are resolved relative to
the script, so it also works when launched from another directory.
Raw and processed data are gitignored; only this README is committed.

## Files

Each subset (`FD001`, `FD002`, `FD003`, `FD004`) contains:

| File | Contents |
|---|---|
| `train_FDxxx.txt` | Engine trajectories that run to failure, one row per cycle. |
| `test_FDxxx.txt` | Engine trajectories stopped before failure, one row per cycle. |
| `RUL_FDxxx.txt` | One remaining-life value in cycles per test engine, in ascending unit order, at its last recorded cycle. |

`xxx` is `001` through `004`. Unit IDs are local to each subset and split.
FD001 is the project's primary subset; FD002–FD004 are extensions.
Training RUL is the engine's final cycle minus the current cycle.
The separate RUL files supply test targets; RUL is not one of the 26 input columns.

## The 26 columns

Train and test files have no header and use whitespace separators (including
trailing spaces). The names below match `rul.data.load.COLUMNS`.

| Position (1-based) | Column | Meaning |
|---|---|---|
| 1 | `unit` | Engine ID |
| 2 | `cycle` | Operating cycle |
| 3 | `op_1` | Operating setting 1 |
| 4 | `op_2` | Operating setting 2 |
| 5 | `op_3` | Operating setting 3 |
| 6 | `s_1` | Sensor measurement 1 |
| 7 | `s_2` | Sensor measurement 2 |
| 8 | `s_3` | Sensor measurement 3 |
| 9 | `s_4` | Sensor measurement 4 |
| 10 | `s_5` | Sensor measurement 5 |
| 11 | `s_6` | Sensor measurement 6 |
| 12 | `s_7` | Sensor measurement 7 |
| 13 | `s_8` | Sensor measurement 8 |
| 14 | `s_9` | Sensor measurement 9 |
| 15 | `s_10` | Sensor measurement 10 |
| 16 | `s_11` | Sensor measurement 11 |
| 17 | `s_12` | Sensor measurement 12 |
| 18 | `s_13` | Sensor measurement 13 |
| 19 | `s_14` | Sensor measurement 14 |
| 20 | `s_15` | Sensor measurement 15 |
| 21 | `s_16` | Sensor measurement 16 |
| 22 | `s_17` | Sensor measurement 17 |
| 23 | `s_18` | Sensor measurement 18 |
| 24 | `s_19` | Sensor measurement 19 |
| 25 | `s_20` | Sensor measurement 20 |
| 26 | `s_21` | Sensor measurement 21 |
