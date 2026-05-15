from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from swds.data.schema import TabularDataset
from swds.experiments.pipeline import ExperimentConfig, ExperimentResult, run_temporal_experiment


@dataclass(frozen=True)
class AblationConfig:
    projection_counts: tuple[int, ...] = (16, 32, 64, 128, 256)
    window_sizes: tuple[int, ...] = (500, 1000, 2000, 5000)
    threshold_quantiles: tuple[float, ...] = (0.90, 0.95, 0.975)
    reference_modes: tuple[str, ...] = ("train", "recent")
    representation_modes: tuple[str, ...] = ("raw", "pca", "model_output")
    run_projection_count: bool = True
    run_window_size: bool = True
    run_threshold: bool = True
    run_reference_mode: bool = True
    run_representation: bool = True


@dataclass(frozen=True)
class AblationResult:
    correlations: pd.DataFrame
    runtimes: pd.DataFrame
    retraining: pd.DataFrame
    run_index: pd.DataFrame


def run_ablation_experiment(
    dataset: TabularDataset,
    *,
    base_config: ExperimentConfig,
    config: AblationConfig | None = None,
    output_dir: str | Path | None = None,
) -> AblationResult:
    config = config or AblationConfig()
    runs: list[tuple[str, str, ExperimentConfig]] = []

    if config.run_projection_count:
        methods = tuple(f"swds_k{k}" for k in config.projection_counts)
        runs.append(
            (
                "projection_count",
                ",".join(map(str, config.projection_counts)),
                replace(base_config, methods=methods, run_retraining=False),
            )
        )

    if config.run_window_size:
        for window_size in config.window_sizes:
            runs.append(
                (
                    "window_size",
                    str(window_size),
                    replace(base_config, window_size=window_size, run_retraining=False),
                )
            )

    if config.run_threshold:
        for quantile in config.threshold_quantiles:
            runs.append(
                (
                    "threshold_quantile",
                    str(quantile),
                    replace(base_config, threshold_quantile=quantile, run_retraining=True),
                )
            )

    if config.run_reference_mode:
        for mode in config.reference_modes:
            runs.append(
                (
                    "reference_mode",
                    mode,
                    replace(base_config, reference_mode=mode, run_retraining=False),
                )
            )

    if config.run_representation:
        for mode in config.representation_modes:
            runs.append(
                (
                    "drift_representation",
                    mode,
                    replace(base_config, drift_representation=mode, run_retraining=False),
                )
            )

    correlation_tables = []
    runtime_tables = []
    retraining_tables = []
    run_rows = []

    for run_id, (ablation_type, ablation_value, run_config) in enumerate(runs):
        result = run_temporal_experiment(dataset, config=run_config)
        correlation_tables.append(_annotate(result.correlations, run_id, ablation_type, ablation_value))
        runtime_tables.append(
            _runtime_summary(
                result.window_scores,
                run_id=run_id,
                ablation_type=ablation_type,
                ablation_value=ablation_value,
            )
        )
        if not result.retraining_summary.empty:
            retraining_tables.append(_annotate(result.retraining_summary, run_id, ablation_type, ablation_value))
        run_rows.append(
            {
                "run_id": run_id,
                "ablation_type": ablation_type,
                "ablation_value": ablation_value,
                "dataset": dataset.name,
                **asdict(run_config),
            }
        )

    output = AblationResult(
        correlations=_concat(correlation_tables),
        runtimes=_concat(runtime_tables),
        retraining=_concat(retraining_tables),
        run_index=pd.DataFrame(run_rows),
    )
    if output_dir is not None:
        save_ablation_result(output, output_dir=output_dir, config=config)
    return output


def save_ablation_result(result: AblationResult, *, output_dir: str | Path, config: AblationConfig) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.correlations.to_csv(output / "ablation_correlations.csv", index=False)
    result.runtimes.to_csv(output / "ablation_runtimes.csv", index=False)
    if not result.retraining.empty:
        result.retraining.to_csv(output / "ablation_retraining.csv", index=False)
    result.run_index.to_csv(output / "ablation_runs.csv", index=False)
    pd.DataFrame([asdict(config)]).to_json(output / "ablation_config.json", orient="records", indent=2)


def _annotate(frame: pd.DataFrame, run_id: int, ablation_type: str, ablation_value: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "run_id", run_id)
    out.insert(1, "ablation_type", ablation_type)
    out.insert(2, "ablation_value", ablation_value)
    return out


def _runtime_summary(
    window_scores: pd.DataFrame,
    *,
    run_id: int,
    ablation_type: str,
    ablation_value: str,
) -> pd.DataFrame:
    grouped = (
        window_scores.groupby(["dataset", "model", "method"], sort=True)["runtime_seconds"]
        .agg(runtime_median="median", runtime_iqr=lambda x: x.quantile(0.75) - x.quantile(0.25))
        .reset_index()
    )
    return _annotate(grouped, run_id, ablation_type, ablation_value)


def _concat(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)
