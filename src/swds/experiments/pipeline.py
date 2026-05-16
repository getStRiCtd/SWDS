from __future__ import annotations

import logging
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
from swds.drift.registry import DEFAULT_METHODS, DriftRuntimeConfig, DriftScorer
from swds.drift.utils import matrix_vstack
from swds.models.evaluate import evaluate_model, primary_metric, quality_drop
from swds.models.train import train_model
from swds.monitoring.retraining_policy import build_policy_triggers, simulate_retraining_policies
from swds.monitoring.thresholds import quantile_threshold
from swds.monitoring.windows import fixed_count_windows, fixed_time_windows


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    window_mode: str = "count"
    window_size: int = 500
    window_time_freq: str = "M"
    min_window_size: int = 2
    min_test_windows: int = 20
    enforce_min_test_windows: bool = False
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
    n_jobs: int = 1
    swds_backend: str = "auto"
    swds_device: str | None = None
    ks_max_features: int | None = 2048
    psi_max_features: int | None = 2048
    psi_n_bins: int = 10
    mmd_max_samples: int = 1000
    energy_max_samples: int = 1000
    c2st_max_samples: int = 4000
    c2st_n_splits: int = 5
    c2st_n_jobs: int | None = None
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
    psi_diagnostics: pd.DataFrame
    retraining_windows: pd.DataFrame
    retraining_summary: pd.DataFrame


def run_temporal_experiment(
    dataset: TabularDataset,
    *,
    config: ExperimentConfig | None = None,
    output_dir: str | Path | None = None,
) -> ExperimentResult:
    config = config or ExperimentConfig()
    LOGGER.info(
        "experiment started dataset=%s task=%s rows=%d features=%d model=%s output_dir=%s",
        dataset.name,
        TaskType(dataset.task_type).value,
        len(dataset.frame),
        len(dataset.feature_columns),
        config.model_name,
        output_dir,
    )
    LOGGER.debug("experiment config: %s", asdict(config))
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
    LOGGER.info(
        "temporal split ready dataset=%s train=%d val=%d test=%d split_col=%s",
        dataset.name,
        len(split.train),
        len(split.val),
        len(split.test),
        dataset.split_col or "",
    )

    feature_cols = dataset.feature_columns
    LOGGER.info("building preprocessor dataset=%s feature_cols=%d", dataset.name, len(feature_cols))
    preprocessor, feature_summary = build_preprocessor(
        split.train,
        feature_cols,
        max_onehot_cardinality=config.max_onehot_cardinality,
        hash_features=config.hash_features,
    )
    LOGGER.info(
        "feature types dataset=%s numeric=%d low_card_categorical=%d high_card_categorical=%d",
        dataset.name,
        len(feature_summary.numeric),
        len(feature_summary.low_cardinality_categorical),
        len(feature_summary.high_cardinality_categorical),
    )
    LOGGER.info("fitting preprocessor dataset=%s train_rows=%d", dataset.name, len(split.train))
    preprocessor.fit(split.train[feature_cols])

    LOGGER.info("transforming train/validation/test splits dataset=%s", dataset.name)
    X_train = transform_to_float32(preprocessor, split.train[feature_cols])
    X_val = transform_to_float32(preprocessor, split.val[feature_cols])
    X_test = transform_to_float32(preprocessor, split.test[feature_cols])
    LOGGER.debug(
        "transformed matrix shapes dataset=%s train=%s val=%s test=%s",
        dataset.name,
        _shape(X_train),
        _shape(X_val),
        _shape(X_test),
    )
    y_train = split.train[dataset.target_col].to_numpy()
    y_val = split.val[dataset.target_col].to_numpy()
    y_test = split.test[dataset.target_col].to_numpy()

    LOGGER.info(
        "training model dataset=%s model=%s task=%s train_rows=%d",
        dataset.name,
        config.model_name,
        TaskType(dataset.task_type).value,
        len(y_train),
    )
    model = train_model(
        X_train,
        y_train,
        task_type=dataset.task_type,
        model_name=config.model_name,
        seed=config.seed,
    )
    LOGGER.info("evaluating validation split dataset=%s val_rows=%d", dataset.name, len(y_val))
    val_metrics = evaluate_model(model, X_val, y_val, task_type=dataset.task_type)
    metric_name = primary_metric(dataset.task_type, val_metrics)
    ref_quality = float(val_metrics.get(metric_name, np.nan))
    LOGGER.info(
        "validation quality dataset=%s metric=%s value=%.6f metrics=%s",
        dataset.name,
        metric_name,
        ref_quality,
        val_metrics,
    )
    LOGGER.info("building drift representation dataset=%s mode=%s", dataset.name, config.drift_representation)
    X_train_drift, X_val_drift, X_test_drift = _make_drift_representation(
        model=model,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        config=config,
    )
    LOGGER.debug(
        "drift representation shapes dataset=%s train=%s val=%s test=%s",
        dataset.name,
        _shape(X_train_drift),
        _shape(X_val_drift),
        _shape(X_test_drift),
    )

    val_windows = _make_windows(split.val, dataset.time_col, config=config, prefix="V")
    test_windows = _make_windows(split.test, dataset.time_col, config=config, prefix="W")
    LOGGER.info(
        "monitoring windows ready dataset=%s validation_windows=%d test_windows=%d mode=%s",
        dataset.name,
        len(val_windows),
        len(test_windows),
        config.window_mode,
    )
    if config.enforce_min_test_windows and len(test_windows) < config.min_test_windows:
        LOGGER.error(
            "dataset excluded because test windows below minimum dataset=%s test_windows=%d min_test_windows=%d window_size=%d",
            dataset.name,
            len(test_windows),
            config.min_test_windows,
            config.window_size,
        )
        raise ValueError(
            f"dataset {dataset.name!r} has {len(test_windows)} test windows; "
            f"minimum for main protocol is {config.min_test_windows}"
        )
    fixed_drift_scorer = None
    if config.reference_mode == "train":
        fixed_drift_scorer = DriftScorer(methods=config.methods, config=_drift_runtime_config(config))
        LOGGER.info("preparing shared fixed drift reference dataset=%s", dataset.name)
        fixed_drift_scorer.prepare_reference(X_train_drift)

    validation_scores, validation_psi_diagnostics = _score_windows(
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
        scorer=fixed_drift_scorer,
    )
    LOGGER.info(
        "calibrating thresholds dataset=%s quantile=%.4f validation_rows=%d",
        dataset.name,
        config.threshold_quantile,
        len(validation_scores),
    )
    thresholds = _calibrate_thresholds(validation_scores, config.threshold_quantile)
    LOGGER.info("threshold calibration completed dataset=%s methods=%d", dataset.name, len(thresholds))
    LOGGER.debug("thresholds dataset=%s values=%s", dataset.name, thresholds)
    test_recent_reference = _initial_recent_reference(X_val_drift, val_windows, config=config)
    LOGGER.debug("initial recent reference blocks dataset=%s count=%d", dataset.name, len(test_recent_reference))

    window_scores, test_psi_diagnostics = _score_windows(
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
        scorer=fixed_drift_scorer,
    )
    psi_diagnostics = pd.concat(
        [validation_psi_diagnostics, test_psi_diagnostics],
        ignore_index=True,
    )
    LOGGER.info("computing drift-quality correlations dataset=%s rows=%d", dataset.name, len(window_scores))
    correlations = drift_quality_correlations(window_scores)
    correlations.insert(0, "dataset", dataset.name)
    correlations.insert(1, "model", config.model_name)
    LOGGER.info("correlations completed dataset=%s rows=%d", dataset.name, len(correlations))

    retraining_windows = pd.DataFrame()
    retraining_summary = pd.DataFrame()
    if config.run_retraining and len(test_windows):
        LOGGER.info("retraining simulation enabled dataset=%s test_windows=%d", dataset.name, len(test_windows))
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
        LOGGER.info(
            "retraining simulation completed dataset=%s window_rows=%d summary_rows=%d",
            dataset.name,
            len(retraining_windows),
            len(retraining_summary),
        )
    else:
        LOGGER.info(
            "retraining simulation skipped dataset=%s run_retraining=%s test_windows=%d",
            dataset.name,
            config.run_retraining,
            len(test_windows),
        )

    LOGGER.info("building dataset summary dataset=%s", dataset.name)
    summary = _dataset_summary(dataset, split, feature_summary, test_windows, metric_name)
    summary["window_mode"] = config.window_mode
    summary["reference_mode"] = config.reference_mode
    summary["drift_representation"] = config.drift_representation
    result = ExperimentResult(
        window_scores=window_scores,
        correlations=correlations,
        dataset_summary=summary,
        validation_scores=validation_scores,
        psi_diagnostics=psi_diagnostics,
        retraining_windows=retraining_windows,
        retraining_summary=retraining_summary,
    )
    if output_dir is not None:
        save_experiment_result(result, output_dir=output_dir, config=config)
    LOGGER.info("experiment completed dataset=%s output_dir=%s", dataset.name, output_dir)
    return result


def save_experiment_result(result: ExperimentResult, *, output_dir: str | Path, config: ExperimentConfig) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    LOGGER.info("saving experiment result output_dir=%s", output)
    _write_csv(result.window_scores, output / "window_scores.csv")
    _write_csv(result.validation_scores, output / "validation_scores.csv")
    if not result.psi_diagnostics.empty:
        _write_csv(result.psi_diagnostics, output / "psi_diagnostics.csv")
    _write_csv(result.correlations, output / "correlations.csv")
    _write_csv(result.dataset_summary, output / "dataset_summary.csv")
    if not result.retraining_windows.empty:
        _write_csv(result.retraining_windows, output / "retraining_policy_windows.csv")
    if not result.retraining_summary.empty:
        _write_csv(result.retraining_summary, output / "retraining_policy_summary.csv")
    config_path = output / "config.json"
    pd.DataFrame([asdict(config)]).to_json(config_path, orient="records", indent=2)
    LOGGER.info("wrote config JSON path=%s", config_path)


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
    scorer: DriftScorer | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info(
        "scoring windows dataset=%s stream=%s windows=%d methods=%s reference_mode=%s",
        dataset.name,
        stream_name,
        len(windows),
        ",".join(config.methods),
        config.reference_mode,
    )
    rows = []
    psi_rows = []
    recent_reference = list(initial_recent_reference) if initial_recent_reference is not None else [X_ref]
    scorer_was_provided = scorer is not None
    scorer = scorer or DriftScorer(methods=config.methods, config=_drift_runtime_config(config))
    if config.reference_mode == "train" and not scorer_was_provided:
        LOGGER.info("preparing fixed drift reference dataset=%s stream=%s", dataset.name, stream_name)
        scorer.prepare_reference(X_ref)
    for window in windows:
        X_cur = X_stream[window.start : window.end]
        X_model_cur = X_model_stream[window.start : window.end]
        y_cur = y_stream[window.start : window.end]
        X_ref_cur = _reference_for_window(
            fixed_reference=X_ref,
            recent_reference=recent_reference,
            config=config,
        )
        LOGGER.info(
            "scoring window dataset=%s stream=%s label=%s index=%d rows=%d reference_rows=%d",
            dataset.name,
            stream_name,
            window.label,
            window.index,
            window.size,
            int(X_ref_cur.shape[0]),
        )
        metrics = evaluate_model(model, X_model_cur, y_cur, task_type=dataset.task_type)
        current_quality = float(metrics.get(metric_name, np.nan))
        drop = quality_drop(reference_quality, current_quality, metric_name=metric_name)
        LOGGER.debug(
            "window quality dataset=%s stream=%s label=%s metric=%s value=%.6f drop=%.6f metrics=%s",
            dataset.name,
            stream_name,
            window.label,
            metric_name,
            current_quality,
            drop,
            metrics,
        )
        if config.reference_mode != "train":
            scorer = DriftScorer(methods=config.methods, config=_drift_runtime_config(config))
        drift_scores = scorer.compute(X_ref_cur, X_cur)
        psi_diagnostics = scorer.last_psi_diagnostics()
        if psi_diagnostics is not None:
            zero_expected_share = psi_diagnostics.zero_expected_bins / max(psi_diagnostics.n_bins_total, 1)
            zero_actual_share = psi_diagnostics.zero_actual_bins / max(psi_diagnostics.n_bins_total, 1)
            clipped_expected_share = psi_diagnostics.clipped_expected_bins / max(psi_diagnostics.n_bins_total, 1)
            clipped_actual_share = psi_diagnostics.clipped_actual_bins / max(psi_diagnostics.n_bins_total, 1)
            LOGGER.info(
                "PSI diagnostics dataset=%s stream=%s label=%s features=%d bins=%d zero_expected_share=%.4f zero_actual_share=%.4f clipped_expected_share=%.4f clipped_actual_share=%.4f top_feature=%s top_psi=%s",
                dataset.name,
                stream_name,
                window.label,
                psi_diagnostics.n_features,
                psi_diagnostics.n_bins_total,
                zero_expected_share,
                zero_actual_share,
                clipped_expected_share,
                clipped_actual_share,
                psi_diagnostics.top_features[0].feature_id if psi_diagnostics.top_features else "",
                f"{psi_diagnostics.top_features[0].psi:.6f}" if psi_diagnostics.top_features else "",
            )
            for rank, feature in enumerate(psi_diagnostics.top_features, start=1):
                psi_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.model_name,
                        "stream": stream_name,
                        "window_index": window.index,
                        "window_label": window.label,
                        "window_start": window.start,
                        "window_end": window.end,
                        "window_size": window.size,
                        "reference_mode": config.reference_mode,
                        "drift_representation": config.drift_representation,
                        "reference_size": int(X_ref_cur.shape[0]),
                        "rank": rank,
                        "feature_id": feature.feature_id,
                        "psi": feature.psi,
                        "n_bins": feature.n_bins,
                        "zero_expected_bins": feature.zero_expected_bins,
                        "zero_actual_bins": feature.zero_actual_bins,
                        "clipped_expected_bins": feature.clipped_expected_bins,
                        "clipped_actual_bins": feature.clipped_actual_bins,
                        "n_features": psi_diagnostics.n_features,
                        "n_bins_total": psi_diagnostics.n_bins_total,
                        "zero_expected_share": zero_expected_share,
                        "zero_actual_share": zero_actual_share,
                        "clipped_expected_share": clipped_expected_share,
                        "clipped_actual_share": clipped_actual_share,
                    }
                )
        for drift_score in drift_scores:
            threshold = None if thresholds is None else thresholds.get(drift_score.method)
            LOGGER.debug(
                "drift score dataset=%s stream=%s label=%s method=%s score=%.8f threshold=%s runtime=%.4fs",
                dataset.name,
                stream_name,
                window.label,
                drift_score.method,
                drift_score.score,
                threshold,
                drift_score.runtime_seconds,
            )
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
            LOGGER.debug(
                "recent reference updated dataset=%s stream=%s blocks=%d",
                dataset.name,
                stream_name,
                len(recent_reference),
            )
    out = pd.DataFrame(rows)
    psi_out = pd.DataFrame(psi_rows)
    LOGGER.info(
        "scoring windows completed dataset=%s stream=%s rows=%d psi_diagnostic_rows=%d",
        dataset.name,
        stream_name,
        len(out),
        len(psi_out),
    )
    return out, psi_out


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
    LOGGER.info(
        "building retraining inputs dataset=%s windows=%d methods=%s periods=%s histories=%s",
        dataset.name,
        len(test_windows),
        ",".join(config.methods),
        config.retraining_periods,
        config.retraining_history_modes,
    )
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
    trigger_counts = {name: int(values.sum()) for name, values in triggers.items()}
    LOGGER.info("retraining triggers ready dataset=%s trigger_counts=%s", dataset.name, trigger_counts)
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


def _drift_runtime_config(config: ExperimentConfig) -> DriftRuntimeConfig:
    return DriftRuntimeConfig(
        seed=config.seed,
        n_jobs=config.n_jobs,
        swds_backend=config.swds_backend,
        swds_device=config.swds_device,
        ks_max_features=config.ks_max_features,
        psi_max_features=config.psi_max_features,
        psi_n_bins=config.psi_n_bins,
        mmd_max_samples=config.mmd_max_samples,
        energy_max_samples=config.energy_max_samples,
        c2st_max_samples=config.c2st_max_samples,
        c2st_n_splits=config.c2st_n_splits,
        c2st_n_jobs=config.c2st_n_jobs,
    )


def _calibrate_thresholds(validation_scores: pd.DataFrame, quantile: float) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for method, group in validation_scores.groupby("method"):
        scores = group["score"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(scores):
            thresholds[method] = quantile_threshold(scores, quantile=quantile)
            LOGGER.info(
                "threshold calibrated method=%s quantile=%.4f value=%.8f n_scores=%d",
                method,
                quantile,
                thresholds[method],
                len(scores),
            )
        else:
            LOGGER.warning("threshold skipped method=%s reason=no finite validation scores", method)
    return thresholds


def _make_drift_representation(*, model, X_train, X_val, X_test, config: ExperimentConfig):
    mode = config.drift_representation
    if mode == "raw":
        LOGGER.info("using raw drift representation")
        return X_train, X_val, X_test
    if mode == "pca":
        n_components = min(config.pca_components, X_train.shape[1], max(X_train.shape[0] - 1, 1))
        if n_components < 1:
            LOGGER.warning("PCA representation skipped because n_components < 1")
            return X_train, X_val, X_test
        LOGGER.info("fitting PCA/SVD drift representation components=%d", n_components)
        reducer = (
            TruncatedSVD(n_components=n_components, random_state=config.seed)
            if sparse.issparse(X_train)
            else PCA(n_components=n_components, random_state=config.seed)
        )
        reducer.fit(X_train)
        return reducer.transform(X_train), reducer.transform(X_val), reducer.transform(X_test)
    if mode in {"model_output", "model_aware"}:
        LOGGER.info("building model-output drift representation")
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
        windows = fixed_count_windows(
            len(frame),
            window_size=config.window_size,
            min_size=config.min_window_size,
            prefix=prefix,
        )
        LOGGER.info(
            "fixed-count windows built prefix=%s rows=%d window_size=%d min_size=%d count=%d",
            prefix,
            len(frame),
            config.window_size,
            config.min_window_size,
            len(windows),
        )
        return windows
    if config.window_mode == "time":
        windows = fixed_time_windows(
            frame,
            time_col=time_col,
            freq=config.window_time_freq,
            min_size=config.min_window_size,
            prefix=prefix,
        )
        LOGGER.info(
            "fixed-time windows built prefix=%s rows=%d freq=%s min_size=%d count=%d",
            prefix,
            len(frame),
            config.window_time_freq,
            config.min_window_size,
            len(windows),
        )
        return windows
    raise ValueError("window_mode must be either 'count' or 'time'")


def _initial_recent_reference(X_val, val_windows, *, config: ExperimentConfig) -> list:
    if config.reference_mode != "recent":
        return []
    n_recent = max(config.recent_reference_windows, 1)
    selected_windows = val_windows[-n_recent:]
    LOGGER.info("initial recent reference selected windows=%d", len(selected_windows))
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
    LOGGER.info(
        "dataset summary ready dataset=%s n_samples=%d n_features=%d missing_rate=%.6f",
        dataset.name,
        len(frame),
        len(features),
        missing_rate,
    )
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


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    LOGGER.info("wrote CSV path=%s rows=%d columns=%d", path, len(frame), len(frame.columns))


def _shape(matrix) -> tuple[int, ...]:
    return tuple(getattr(matrix, "shape", ()))
