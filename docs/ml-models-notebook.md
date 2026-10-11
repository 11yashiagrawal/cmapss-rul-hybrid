# FD001 raw and engineered ML comparison

[`03_ml_models.ipynb`](../notebooks/phase1/03_ml_models.ipynb) implements #18.
Install `requirements.txt` and the editable package as described in
`CONTRIBUTING.md`, put the FD001 train/test/RUL files in
`data/raw/CMAPSSData/`, then open the notebook and **Restart & Run All**.
Run from the repository root or the notebook directory. The notebook locates
the repository root and uses no machine-specific paths.

## Protocol

- Uncapped training RUL; seed 42; 80 training and 20 held-out engines.
- The same three engine-grouped CV folds and four candidates per model for
  both raw and engineered features. See `COMPARISON_GRIDS` in
  `src/rul/models/experiment.py`; these are bounded notebook grids, separate
  from #17's baseline configuration.
- Raw predictors: cycle, three operating settings, all 21 sensors.
- Engineered predictors: cycle, settings, operating regime, nonconstant
  sensors, and causal mean/slope/health features at windows 5, 10, and 20.
- Sensor selection and normalization fit inside every CV training fold.
  Elastic Net scaling also fits inside CV. The notebook therefore starts
  from raw trajectories, rather than the all-training-fitted matrices from #15.
- Single-worker model searches and a temporary preprocessing cache bound
  resource use. Random Forest uses `max_features=0.5` in both variants;
  Elastic Net allows 20,000 iterations.
- Holdout RMSE, MAE, NASA score, and errors by true-RUL stage use the shared
  evaluation module. These scores cover every held-out cycle. Official
  last-cycle test-set comparison belongs to #6.

## Explanations

Tree SHAP explains the engineered Random Forest and XGBoost predictions on
a seeded sample of 200 held-out cycles. Complete engine trajectories are
transformed before sampling, preserving causal rolling history. The notebook
checks that SHAP contributions plus their base values reconstruct predictions.
Mean absolute SHAP ranks importance; beeswarm plots show contribution direction.
Correlated sensor and rolling features can share attribution. These are model
explanations, not evidence of a causal effect on engine life.

The executed notebook contains selected parameters, every CV candidate,
the comparison plot, stage metrics, SHAP tables, and plots. It does not export
model checkpoints or modify the baseline YAML configuration.

## Observed internal validation

| Model | Raw CV RMSE | Engineered CV RMSE | Raw holdout RMSE | Engineered holdout RMSE |
|---|---:|---:|---:|---:|
| Random Forest | 39.158 | 39.265 | 30.759 | 26.472 |
| XGBoost | 39.892 | 40.871 | 30.904 | 30.330 |
| Elastic Net | 42.651 | 42.882 | 31.740 | 33.628 |

The trees improve on this holdout, while Elastic Net worsens. None improves
mean CV RMSE, so the holdout gains do not establish a general advantage for
engineered features. Engineered XGBoost also has a higher NASA score despite
its slightly lower holdout RMSE; the metrics measure different error costs.

Both variants select Random Forest with 100 trees and minimum leaf size 3,
and XGBoost with 100 trees and maximum depth 3. Raw Elastic Net selects
alpha 0.1 and l1 ratio 0.8; engineered Elastic Net selects alpha 1.0 and
l1 ratio 0.2. The notebook uses the additional fixed model settings documented
above and the baseline factory's defaults.
