# Sliced Wasserstein Drift Score for Temporal Tabular ML

This repository implements the experimental core described in `method.md`:
leakage-safe tabular preprocessing, temporal splits, windowed monitoring,
Sliced Wasserstein Drift Score (SWDS), baseline drift indicators, downstream
quality tracking, threshold calibration, and saved result tables.

## Quick start

```bash
uv sync
uv run swds run-config --config configs/experiments/synthetic.yaml
```

For status-heavy runs, enable structured logs. CLI logging options can be
placed before or after the subcommand:

```bash
uv run swds --log-level INFO --log-file results/run.log \
  run-config --config configs/experiments/synthetic.yaml
```

`SWDS_LOG_LEVEL` and `SWDS_LOG_FILE` provide the same controls. `INFO` logs
dataset loading, preprocessing, training, scoring windows, thresholding,
retraining simulation, downloads/preparation, and saved artifacts; `DEBUG`
also records matrix shapes, per-method scores, and per-window metrics.

The command writes:

- `window_scores.csv`: one row per test window and drift method;
- `validation_scores.csv`: validation-window scores used for thresholds;
- `correlations.csv`: Spearman/Kendall drift-quality correlations;
- `dataset_summary.csv`: reproducibility metadata;
- `config.json`: the exact experiment settings;
- `retraining_policy_windows.csv`: H3 policy simulation by monitoring window;
- `retraining_policy_summary.csv`: regret, mean/worst quality, retrain counts,
  and false-retrain rates.

## Run on a temporal CSV

```bash
uv run swds run-csv \
  --path data/raw/my_dataset.csv \
  --target target \
  --time timestamp \
  --task classification \
  --window-size 1000 \
  --output results/my_dataset
```

If `--task` is omitted, the loader infers classification vs regression from
the target column. The timestamp column is used only for sorting and temporal
splitting. Preprocessing is fitted only on the train split.

## Run from YAML

```bash
uv run swds run-config --config configs/datasets/external/ieee-cis-fraud.yaml
```

YAML configs support three dataset types:

- `synthetic`: generated temporal sanity-check data;
- `csv`: public/Kaggle-style temporal CSV files;
- `tabred`: preprocessed TabReD directories with `X_num.npy`, `X_bin.npy`,
  `X_cat.npy`, `Y.npy`, `info.json`, and `split-default/*_idx.npy`.

TabReD template configs live in `configs/datasets/tabred/`. External CSV
templates currently cover IEEE-CIS Fraud Detection, Rossmann Store Sales, and
Bike Sharing Demand.

## Prepare TabReD

The official TabReD repository downloads raw data from Kaggle and writes the
processed arrays expected by this project. You need Kaggle credentials and must
accept the relevant competition/dataset rules on Kaggle first.

```bash
uv sync --extra tabred
uv run --extra tabred swds prepare-tabred --datasets all
uv run --extra tabred swds prepare-tabred --validate-only
```

By default this uses `data/raw/tabred_repo` for the official repository and
links processed datasets into `data/raw/tabred/<dataset>`, matching the YAML
configs in `configs/datasets/tabred/`.

## Controlled Drift Injection

```bash
uv run swds run-synthetic-drift --config configs/experiments/synthetic_drift.yaml
```

This implements the H2 protocol: future monitoring windows are modified with a
known drift start, while labels are not used by the score. Outputs:

- `synthetic_drift_windows.csv`: method scores, drift labels, thresholds, and
  triggers per scenario/window;
- `synthetic_drift_summary.csv`: AUROC, AUPRC, detection delay, and pre-drift
  false alarm rate;
- `synthetic_drift_validation_scores.csv`: validation scores used for threshold
  calibration.

## Ablations

```bash
uv run swds run-ablation --config configs/experiments/synthetic.yaml --output results/ablation
```

The ablation runner covers:

- SWDS projection counts, using dynamic method names like `swds_k64`;
- fixed-count window sizes;
- threshold quantiles;
- train vs recent reference distributions;
- drift representation modes: `raw`, `pca`, and `model_output`.

Outputs include `ablation_correlations.csv`, `ablation_runtimes.csv`,
`ablation_retraining.csv` when threshold policies are evaluated, and
`ablation_runs.csv`.

## Batch Runs And Reports

```bash
uv run swds run-batch \
  --configs configs/datasets/tabred/*.yaml configs/datasets/external/*.yaml \
  --output results/batch \
  --log-level INFO --log-file results/batch/run.log

uv run swds build-report \
  --results results/batch/* results/synthetic_drift results/ablation \
  --output results/report
```

The batch runner writes `run_manifest.csv`, `dataset_exclusions.csv`, and
`environment_manifest.json`. The report builder assembles paper-facing CSV
tables and figures from saved experiment directories.

Batch runs are resumable by default: configs whose `window_scores.csv`,
`validation_scores.csv`, `correlations.csv`, `dataset_summary.csv`, and
`config.json` already exist are recorded as completed and skipped. Add
`--rerun-completed` to force recomputation.

The dataset YAMLs keep the full method set and H3 retraining protocol, but set
`n_jobs: -1` so drift scoring uses all CPU cores. You can override at runtime:

```bash
uv run swds run-batch \
  --configs configs/datasets/tabred/*.yaml configs/datasets/external/*.yaml \
  --output results/batch \
  --n-jobs -1 \
  --swds-backend auto
```

`swds_backend: auto` uses a CUDA-enabled Torch installation for SWDS when
`torch.cuda.is_available()` is true and falls back to NumPy otherwise. The
sklearn downstream models and baselines such as KS, PSI, MMD, energy, and C2ST
remain CPU-based; GPU acceleration applies to SWDS projection and quantile
computation. If using Colab, make sure the environment actually contains a
CUDA Torch wheel before launching `uv run`, then pass
`--swds-backend torch --swds-device cuda` to fail fast if CUDA is not visible.
On CPU machines `auto` avoids importing Torch unless an NVIDIA device is visible
or `SWDS_AUTO_TORCH=1` is set.

For fixed train-reference monitoring, the runner prepares shared reference
state once per dataset for SWDS, KS, PSI, MMD, energy distance, and C2ST.
`mean_ks`/`max_ks` and `mean_psi`/`max_psi` share the same per-feature
statistics inside each window, and H3 retraining policies with identical
trigger schedules share one simulated model trajectory. Runtime knobs are also
available from the CLI, for example `--ks-max-features`, `--psi-max-features`,
`--mmd-max-samples`, `--energy-max-samples`, `--c2st-max-samples`,
`--c2st-n-splits`, and `--c2st-n-jobs`.

## Implemented drift methods

- `swds`: quantile-grid Sliced Wasserstein distance;
- `mean_ks`, `max_ks`;
- `mean_psi`, `max_psi`;
- `mmd_rbf`;
- `energy`;
- `c2st`: classifier two-sample test;
- `sinkhorn`: optional POT-based Sinkhorn divergence, available after
  `uv sync --extra ot`.

GBDT libraries and POT/Sinkhorn are declared as optional extras because the
baseline package should run on a plain CPU environment:

```bash
uv sync --extra gbdt --extra ot
```

First-class model names are `linear`, `hist_gbdt`, `lightgbm`, `xgboost`, and
`catboost`. The default configs use `linear` so smoke runs stay lightweight.

## Retraining policies

The H3 simulation compares:

- `no_retraining`;
- `periodic_pN`;
- one score-triggered policy per drift method;
- `oracle_drop`;
- `oracle_every_window`.

Each policy is evaluated with `all` history and/or `rolling` history according
to the YAML config. Triggers are applied after observing a window, so retraining
affects future windows rather than retroactively improving the current one.
