from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

from swds.analysis.stats import drift_quality_correlations
from swds.data.preprocessing import build_preprocessor, transform_to_float32
from swds.data.schema import TabularDataset, TaskType
from swds.data.temporal_split import official_split, temporal_split
from swds.drift.registry import DEFAULT_METHODS, compute_drift_scores
from swds.drift.utils import matrix_vstack
from swds.models.evaluate import evaluate_model, primary_metric, quality_drop
from swds.models.train import train_model
from swds.monitoring.retraining_policy import build_policy_triggers, simulate_retraining_policies
from swds.monitoring.thresholds import quantile_threshold
from swds.monitoring.windows import fixed_count_windows, fixed_time_windows


@dataclass(frozen=True)
class ExperimentConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    window_mode: str = "count"
    window_size: int = 500
    window_time_freq: str = "M"
    min_window_size: int = 2
    reference_mode: str = "train"
    recent_reference_windows: int = 4
    drift_representation: str = "raw"
    pca_components: int = 32
    model_name: str = "linear"
    methods: tuple[str, ...] = DEFAULT_METHODS
    seed: int = 42
    max_onehot_cardinality: int = 32
    hash_features: int = 256
    threshold_quantile: float = 0.95
    run_retraining: bool = True
    retraining_periods: tuple[int, ...] = (4,)
    retraining_history_modes: tuple[str, ...] = ("all", "rolling")
    rolling_history_windows: int = 4
    oracle_min_quality_drop: float = 0.02


@dataclass(frozen=True)
class ExperimentResult:
    window_scores: pd.DataFrame
    correlations: pd.DataFrame
    dataset_summary: pd.DataFrame
    validation_scores: pd.DataFrame
    retraining_windows: pd.DataFrame
    retraining_summary: pd.DataFrame


def run_temporal_experiment(
    dataset: TabularDataset,
    *,
    config: ExperimentConfig | None = None,
    output_dir: str | Path | None = None,
) -> ExperimentResult:
    config = config or ExperimentConfig()
    split = (
        official_split(dataset.frame, split_col=dataset.split_col)
        if dataset.split_col
        else temporal_split(
            dataset.frame,
            time_col=dataset.time_col,
            train_frac=config.train_frac,
            val_frac=config.val_frac,
            test_frac=config.test_frac,
        )
    )

    feature_cols = dataset.feature_columns
    preprocessor, feature_summary = build_preprocessor(
        split.train,
        feature_cols,
        max_onehot_cardinality=config.max_onehot_cardinality,
        hash_features=config.hash_features,
    )
    preprocessor.fit(split.train[feature_cols])

    X_train = transform_to_float32(preprocessor, split.train[feature_cols])
    X_val = transform_to_float32(preprocessor, split.val[feature_cols])
    X_test = transform_to_float32(preprocessor, split.test[feature_cols])
    y_train = split.train[dataset.target_col].to_numpy()
    y_val = split.val[dataset.target_col].to_numpy()
    y_test = split.test[dataset.target_col].to_numpy()

    model = train_model(
        X_train,
        y_train,
        task_type=dataset.task_type,
        model_name=config.model_name,
        seed=config.seed,
    )
    val_metrics = evaluate_model(model, X_val, y_val, task_type=dataset.task_type)
    metric_name = primary_metric(dataset.task_type, val_metrics)
    ref_quality = float(val_metrics.get(metric_name, np.nan))
    X_train_drift, X_val_drift, X_test_drift = _make_drift_representation(
        model=model,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        config=config,
    )

    val_windows = _make_windows(split.val, dataset.time_col, config=config, prefix="V")
    test_windows = _make_windows(split.test, dataset.time_col, config=config, prefix="W")

    validation_scores = _score_windows(
        X_ref=X_train_drift,
        X_stream=X_val_drift,
        X_model_stream=X_val,
        y_stream=y_val,
        windows=val_windows,
        model=model,
        dataset=dataset,
        config=config,
        metric_name=metric_name,
        reference_quality=ref_quality,
        stream_name="validation",
    )
    thresholds = _calibrate_thresholds(validation_scores, config.threshold_quantile)
    test_recent_reference = _initial_recent_reference(X_val_drift, val_windows, config=config)

    window_scores = _score_windows(
        X_ref=X_train_drift,
        X_stream=X_test_drift,
        X_model_stream=X_test,
        y_stream=y_test,
        windows=test_windows,
        model=model,
        dataset=dataset,
        config=config,
        metric_name=metric_name,
        reference_quality=ref_quality,
        stream_name="test",
        thresholds=thresholds,
        initial_recent_reference=test_recent_reference,
    )
    correlations = drift_quality_correlations(window_scores)
    correlations.insert(0, "dataset", dataset.name)
    correlations.insert(1, "model", config.model_name)

    retraining_windows = pd.DataFrame()
    retraining_summary = pd.DataFrame()
    if config.run_retraining and len(test_windows):
        retraining = _run_retraining_simulation(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            test_windows=test_windows,
            window_scores=window_scores,
            dataset=dataset,
            config=config,
            metric_name=metric_name,
        )
        retraining_windows = retraining.windows
        retraining_summary = retraining.summary

    summary = _dataset_summary(dataset, split, feature_summary, test_windows, metric_name)
    summary["window_mode"] = config.window_mode
    summary["reference_mode"] = config.reference_mode
    summary["drift_representation"] = config.drift_representation
    result = ExperimentResult(
        window_scores=window_scores,
        correlations=correlations,
        dataset_summary=summary,
        validation_scores=validation_scores,
        retraining_windows=retraining_windows,
        retraining_summary=retraining_summary,
    )
    if output_dir is not None:
        save_experiment_result(result, output_dir=output_dir, config=config)
    return result


def save_experiment_result(result: ExperimentResult, *, output_dir: str | Path, config: ExperimentConfig) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.window_scores.to_csv(output / "window_scores.csv", index=False)
    result.validation_scores.to_csv(output / "validation_scores.csv", index=False)
    result.correlations.to_csv(output / "correlations.csv", index=False)
    result.dataset_summary.to_csv(output / "dataset_summary.csv", index=False)
    if not result.retraining_windows.empty:
        result.retraining_windows.to_csv(output / "retraining_policy_windows.csv", index=False)
    if not result.retraining_summary.empty:
        result.retraining_summary.to_csv(output / "retraining_policy_summary.csv", index=False)
    pd.DataFrame([asdict(config)]).to_json(output / "config.json", orient="records", indent=2)


def _score_windows(
    *,
    X_ref,
    X_stream,
    X_model_stream,
    y_stream,
    windows,
    model,
    dataset: TabularDataset,
    config: ExperimentConfig,
    metric_name: str,
    reference_quality: float,
    stream_name: str,
    thresholds: dict[str, float] | None = None,
    initial_recent_reference: list | None = None,
) -> pd.DataFrame:
    rows = []
    recent_reference = list(initial_recent_reference) if initial_recent_reference is not None else [X_ref]
    for window in windows:
        X_cur = X_stream[window.start : window.end]
        X_model_cur = X_model_stream[window.start : window.end]
        y_cur = y_stream[window.start : window.end]
        X_ref_cur = _reference_for_window(
            fixed_reference=X_ref,
            recent_reference=recent_reference,
            config=config,
        )
        metrics = evaluate_model(model, X_model_cur, y_cur, task_type=dataset.task_type)
        current_quality = float(metrics.get(metric_name, np.nan))
        drop = quality_drop(reference_quality, current_quality, metric_name=metric_name)
        drift_scores = compute_drift_scores(X_ref_cur, X_cur, methods=config.methods, seed=config.seed)
        for drift_score in drift_scores:
            threshold = None if thresholds is None else thresholds.get(drift_score.method)
            rows.append(
                {
                    "dataset": dataset.name,
                    "model": config.model_name,
                    "stream": stream_name,
                    "window_index": window.index,
                    "window_label": window.label,
                    "window_start": window.start,
                    "window_end": window.end,
                    "window_size": window.size,
                    "method": drift_score.method,
                    "score": drift_score.score,
                    "runtime_seconds": drift_score.runtime_seconds,
                    "reference_mode": config.reference_mode,
                    "drift_representation": config.drift_representation,
                    "reference_size": int(X_ref_cur.shape[0]),
                    "threshold": threshold,
                    "triggered": bool(threshold is not None and drift_score.score >= threshold),
                    "primary_metric": metric_name,
                    "reference_quality": reference_quality,
                    "quality": current_quality,
                    "quality_drop": drop,
                    **{f"metric_{key}": value for key, value in metrics.items()},
                }
            )
        if config.reference_mode == "recent":
            recent_reference.append(X_cur)
            recent_reference = recent_reference[-max(config.recent_reference_windows, 1) :]
    return pd.DataFrame(rows)


def _run_retraining_simulation(
    *,
    X_train,
    y_train,
    X_test,
    y_test,
    test_windows,
    window_scores: pd.DataFrame,
    dataset: TabularDataset,
    config: ExperimentConfig,
    metric_name: str,
):
    X_windows = [X_test[window.start : window.end] for window in test_windows]
    y_windows = [y_test[window.start : window.end] for window in test_windows]
    per_window = (
        window_scores.drop_duplicates("window_index")
        .sort_values("window_index")
        .reset_index(drop=True)
    )
    triggers = build_policy_triggers(
        window_scores,
        methods=tuple(config.methods),
        periods=tuple(config.retraining_periods),
        oracle_min_quality_drop=config.oracle_min_quality_drop,
    )
    simulation = simulate_retraining_policies(
        X_initial=X_train,
        y_initial=y_train,
        X_windows=X_windows,
        y_windows=y_windows,
        task_type=dataset.task_type,
        model_name=config.model_name,
        primary_metric=metric_name,
        policy_triggers=triggers,
        history_modes=tuple(config.retraining_history_modes),
        rolling_history_windows=config.rolling_history_windows,
        seed=config.seed,
        base_quality_drop=per_window["quality_drop"].to_numpy(dtype=float),
        false_retrain_min_quality_drop=config.oracle_min_quality_drop,
    )
    simulation.windows.insert(0, "dataset", dataset.name)
    simulation.windows.insert(1, "model", config.model_name)
    simulation.summary.insert(0, "dataset", dataset.name)
    simulation.summary.insert(1, "model", config.model_name)
    return simulation


def _calibrate_thresholds(validation_scores: pd.DataFrame, quantile: float) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for method, group in validation_scores.groupby("method"):
        scores = group["score"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(scores):
            thresholds[method] = quantile_threshold(scores, quantile=quantile)
    return thresholds


def _make_drift_representation(*, model, X_train, X_val, X_test, config: ExperimentConfig):
    mode = config.drift_representation
    if mode == "raw":
        return X_train, X_val, X_test
    if mode == "pca":
        n_components = min(config.pca_components, X_train.shape[1], max(X_train.shape[0] - 1, 1))
        if n_components < 1:
            return X_train, X_val, X_test
        reducer = TruncatedSVD(n_components=n_components, random_state=config.seed) if sparse.issparse(X_train) else PCA(n_components=n_components, random_state=config.seed)
        reducer.fit(X_train)
        return reducer.transform(X_train), reducer.transform(X_val), reducer.transform(X_test)
    if mode in {"model_output", "model_aware"}:
        return (
            _model_output_representation(model, X_train),
            _model_output_representation(model, X_val),
            _model_output_representation(model, X_test),
        )
    raise ValueError("drift_representation must be one of: raw, pca, model_output")


def _model_output_representation(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        output = model.predict_proba(X)
    else:
        output = np.asarray(model.predict(X)).reshape(-1, 1)
    return np.asarray(output, dtype=np.float64)


def _make_windows(frame: pd.DataFrame, time_col: str, *, config: ExperimentConfig, prefix: str):
    if config.window_mode == "count":
        return fixed_count_windows(
            len(frame),
            window_size=config.window_size,
            min_size=config.min_window_size,
            prefix=prefix,
        )
    if config.window_mode == "time":
        return fixed_time_windows(
            frame,
            time_col=time_col,
            freq=config.window_time_freq,
            min_size=config.min_window_size,
            prefix=prefix,
        )
    raise ValueError("window_mode must be either 'count' or 'time'")


def _initial_recent_reference(X_val, val_windows, *, config: ExperimentConfig) -> list:
    if config.reference_mode != "recent":
        return []
    n_recent = max(config.recent_reference_windows, 1)
    selected_windows = val_windows[-n_recent:]
    return [X_val[window.start : window.end] for window in selected_windows]


def _reference_for_window(*, fixed_reference, recent_reference: list, config: ExperimentConfig):
    if config.reference_mode == "train":
        return fixed_reference
    if config.reference_mode == "recent":
        parts = recent_reference[-max(config.recent_reference_windows, 1) :]
        if not parts:
            return fixed_reference
        return matrix_vstack(parts)
    raise ValueError("reference_mode must be either 'train' or 'recent'")


def _dataset_summary(dataset, split, feature_summary, test_windows, metric_name: str) -> pd.DataFrame:
    frame = dataset.frame
    features = dataset.feature_columns
    missing_rate = float(frame[features].isna().mean().mean()) if features else 0.0
    time_values = frame[dataset.time_col]
    return pd.DataFrame(
        [
            {
                "dataset": dataset.name,
                "source": dataset.source,
                "task": TaskType(dataset.task_type).value,
                "n_samples": len(frame),
                "n_features": len(features),
                "n_numeric": len(feature_summary.numeric),
                "n_categorical": len(feature_summary.categorical),
                "n_high_cardinality_categorical": len(feature_summary.high_cardinality_categorical),
                "missing_rate": missing_rate,
                "time_min": time_values.min(),
                "time_max": time_values.max(),
                "n_train": len(split.train),
                "n_val": len(split.val),
                "n_test": len(split.test),
                "n_test_windows": len(test_windows),
                "primary_metric": metric_name,
            }
        ]
    )
