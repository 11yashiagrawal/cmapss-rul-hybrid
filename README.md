# cmapss-rul-hybrid

Remaining Useful Life (RUL) prediction for turbofan engines on the NASA C-MAPSS dataset. We compare regression on engineered degradation features (ML, Phase 1) against sequence models that learn directly from raw sensor windows (DL, Phase 2). A hybrid model then fuses the two.

## Problem

Each engine in C-MAPSS is a multivariate time series of 3 operating settings and 21 sensors, recorded once per cycle, that runs until failure. The task is to predict how many cycles a test engine has left, given its history up to some point before failure.

Fixed rolling-window features may miss long degradation trajectories and interactions whose timing differs across engines. This project tests whether sequence models close that gap. The full brief is in [`docs/project_brief.md`](docs/project_brief.md).

## Approach

| Phase | Method | Models |
|---|---|---|
| 1: ML | Engineered features: cycle, rolling mean/slope, operating regime, sensor health | Random Forest, XGBoost, Elastic Net |
| 2: DL | Fixed-length sensor windows | LSTM, Temporal CNN, Transformer encoder |
| 2: Hybrid | Sequence encoder + engineered-feature branch, fused into one regression head | Fusion model |

**Fair comparison.** All models share the same engine-level splits (by `unit` id, never by row) and the same metrics: RMSE and MAE on RUL, NASA score, and error by degradation stage (early / mid / late).

Operating-regime normalization uses `rul.features.regimes.RegimeNormalizer`. Fit it on each training fold, then reuse that fitted object for validation and test data:

```python
from rul.features.regimes import RegimeNormalizer

normalizer = RegimeNormalizer(n_regimes=6, random_state=42)  # FD002 / FD004
train_scaled = normalizer.fit(train_fold).transform(train_fold)
valid_scaled = normalizer.transform(validation_fold)
test_scaled = normalizer.transform(test)
```

Use `n_regimes=1` for FD001 / FD003. The transform preserves metadata columns, replaces the 21 sensor columns with per-regime z-scores, and adds a `regime` column. Never fit it on validation or test rows.

Select sensors separately from the **raw training fold**, before regime normalization. Apply the resulting list to normalized train, validation, and test rows:

```python
from rul.features.cleaning import select_sensors

kept_sensors = select_sensors(train_fold)
train_features = train_scaled[kept_sensors]
valid_features = valid_scaled[kept_sensors]
test_features = test_scaled[kept_sensors]
```

The default rule drops exactly constant training sensors; an optional `min_relative_std` cutoff can also remove nearly flat signals. Relative standard deviation is `std(ddof=0) / max(abs(mean), 1)` on **raw training rows**. Choose any nonzero cutoff using training data only, then reuse the returned sensor list for validation and test. Do not estimate relative variability after z-score normalization, which gives varying sensors roughly unit standard deviation. The [EDA notebook](notebooks/phase1/01_eda.ipynb) found six constant sensors in pooled FD001 training data (`s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19`) and five in FD003 (`s_1`, `s_5`, `s_16`, `s_18`, `s_19`); FD002/FD004 had none when pooled across regimes. Its near-constant candidates are exploratory, so the default rule does not remove them automatically.

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation (Saxena et al., 2008). FD001 is the primary subset; FD002–FD004 are extensions.

The raw data is not committed. Download instructions are under Setup.

## Repository structure

```
├── configs/          experiment configs (YAML)
├── data/             raw + processed data (gitignored)
├── docs/             project brief, literature notes
├── notebooks/
│   ├── phase1/       EDA, feature engineering, ML models, ML results
│   └── phase2/       LSTM, TCN, Transformer, ablations, validation, hybrid, final comparison
├── presentations/    phase slide decks
├── reports/          IEEE LaTeX reports (one section file per owner)
├── results/          figures, metrics (committed); model checkpoints (gitignored)
├── scripts/          data download, training, ablation runners
├── src/rul/          shared Python package
│   ├── data/         loading, RUL labels, engine-level splits
│   ├── features/     cleaning, regimes, rolling features, windows
│   ├── models/       ML baselines, sequence models, hybrid
│   ├── training/     shared PyTorch training loop
│   └── evaluation/   metrics and plots
└── tests/            unit tests (labels, splits, windows)
```

## Setup & reproduce

Use Python **3.10 or newer**. Clone `dev`, which contains the current Phase 1
notebooks, then create and activate a virtual environment:

```sh
git clone --branch dev https://github.com/vks-g/cmapss-rul-hybrid.git
cd cmapss-rul-hybrid
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, use `python -m venv .venv`, then activate with
`.venv\Scripts\activate` in Command Prompt or `.venv\Scripts\Activate.ps1`
in PowerShell instead of the `source` command.

Install dependencies and the local package, then download the dataset:

```sh
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py
```

In Windows PowerShell 5.1, run the two `pip install` commands separately,
continuing only after the first succeeds (`&&` requires PowerShell 7 or newer).

The editable install makes `import rul` work in notebooks and tests.
Data is saved to `data/raw/CMAPSSData/`. If all 12 dataset files already exist,
the download is skipped. See [`data/README.md`](data/README.md) for the source,
files, and 26-column format.

Run the tests (all should pass after downloading the data), then open JupyterLab:

```sh
pytest
jupyter lab notebooks/phase1/
```

Use the virtual environment's Python kernel and **Restart Kernel and Run All
Cells** in each Phase 1 notebook, in this order:

1. [`01_eda.ipynb`](notebooks/phase1/01_eda.ipynb)
2. [`02_feature_engineering.ipynb`](notebooks/phase1/02_feature_engineering.ipynb)
   — writes feature files to `data/processed/`.
3. [`03_ml_models.ipynb`](notebooks/phase1/03_ml_models.ipynb) — about 8 minutes.
4. [`04_ml_results.ipynb`](notebooks/phase1/04_ml_results.ipynb) — about 4 minutes;
   writes metrics to `results/metrics/` and figures to `results/figures/`.

Runtimes depend on your hardware.

## Results

FD001 official test set: 100 engines, each scored once at its last recorded cycle against `RUL_FD001.txt`, with the uncapped RUL target. Lower is better. Phase 2 rows are filled in as experiments complete.

| Model | RMSE | MAE | NASA score |
|---|---|---|---|
| Random Forest | 27.55 | 17.67 | 145,410 |
| XGBoost | 30.50 | 19.82 | 61,202 |
| Elastic Net | 28.39 | 23.57 | 7,777 |
| LSTM | – | – | – |
| Temporal CNN | – | – | – |
| Transformer | – | – | – |
| Hybrid | – | – | – |

Phase 1 rows use the engineered features, with hyperparameters chosen by engine-grouped cross-validation on the training engines only. The NASA score adds one exponential penalty per engine, so a single badly late prediction can dominate it: one test engine accounts for 92% of Random Forest's score. With raw sensor inputs instead, RMSE is 26.15 (Random Forest), 26.84 (XGBoost) and 31.08 (Elastic Net). Stage-wise errors, figures and per-engine predictions are in [`04_ml_results.ipynb`](notebooks/phase1/04_ml_results.ipynb) and [`results/metrics/`](results/metrics/).

## Team

| Member | GitHub |
|---|---|
| M1 (team lead) | [@vks-g](https://github.com/vks-g) |
| M2 | – |
| M3 | – |
| M4 | – |
| M5 | – |

Git workflow is in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## References

- A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," *PHM08*, 2008.
- S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997.
