# Implementation Progress Against `method.md`

## Done

- Leakage-safe preprocessing fitted on train only.
- Temporal train/validation/test split and official split support.
- Fixed-count and fixed-time window helpers.
- SWDS with quantile-grid approximation and equal-size subsampling mode.
- Baseline drift scores: KS, PSI, MMD-RBF, energy distance, C2ST, optional Sinkhorn/POT.
- Classification/regression model evaluation and primary metric selection.
- Main H1 pipeline: drift scores per window, downstream quality per window, Spearman/Kendall correlation.
- Validation-period threshold calibration.
- H3 retraining simulation: no retraining, periodic, score-triggered, oracle-drop, oracle-every-window; all-history and rolling-history modes.
- YAML experiment configs and CLI `run-config`.
- TabReD processed-format loader and template configs for eight TabReD datasets.
- TabReD preparation command that clones/runs the official preprocessing scripts and links outputs into `data/raw/tabred`.
- External CSV template configs for IEEE-CIS Fraud, Rossmann Store Sales, and Bike Sharing Demand.
- Smoke tests and unit tests.
- H2 controlled drift injection with mean, variance, correlation, categorical-prior, missingness, local-subpopulation, concept-like negative-control, and mixed scenarios.
- H2 detection summary with AUROC, AUPRC, detection delay, and pre-drift false alarm rate.
- Ablation runner for SWDS projection count, window size, threshold quantile, reference mode, and drift representation.
- Train and recent reference modes.
- Fixed-count and fixed-time window modes in the main pipeline.
- Drift representation modes: raw leakage-safe features, PCA-compressed features, and model output.
- Aggregate YAML batch runner with manifest and dataset exclusion report.
- Paper-table/report builder for saved result directories.
- First-class optional model names for LightGBM, XGBoost, and CatBoost.

## Partially Done

- Hyperparameter tuning is intentionally minimal; the framework exposes model families but does not yet run random search budgets.
- Sinkhorn divergence exists as an optional score, but it is not included in default configs.
- Figure generation covers core plots, but final paper styling still needs manual curation.

## Still Needed

- Download/prepare the real TabReD and external datasets under `data/raw/`.
- Add Kaggle credentials and accept required Kaggle rules before running `swds prepare-tabred`.
- Run batch experiments on the full dataset list.
- Inspect `dataset_exclusions.csv` and adjust configs for datasets whose local filenames differ.
- Curate final paper plots and write the manuscript text.
- Add optional random-search tuning if runtime budget allows.

## Next Priority

Run the full configured benchmark once the real datasets are available:

```bash
uv run swds run-batch \
  --configs configs/datasets/tabred/*.yaml configs/datasets/external/*.yaml \
  --output results/batch
```

Then build the paper-facing report tables and figures with `uv run swds build-report`.
