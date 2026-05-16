from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from swds.data.preprocessing import build_preprocessor, infer_feature_types, transform_to_float32
from swds.data.schema import TabularDataset
from swds.data.temporal_split import official_split, temporal_split
from swds.drift.registry import DEFAULT_METHODS, DriftRuntimeConfig, DriftScorer
from swds.monitoring.thresholds import quantile_threshold
from swds.monitoring.windows import fixed_count_windows


LOGGER = logging.getLogger(__name__)


DEFAULT_SCENARIOS = (
    "mean_shift",
    "variance_shift",
    "correlation_shift",
    "categorical_prior_shift",
    "missingness_shift",
    "local_subpopulation_shift",
    "concept_like_shift",
    "mixed_shift",
)


@dataclass(frozen=True)
class SyntheticDriftExperimentConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    window_size: int = 500
    drift_start_window: int = 4
    methods: tuple[str, ...] = DEFAULT_METHODS
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
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
    numeric_fraction: float = 0.3
    categorical_fraction: float = 0.5
    mean_delta: float = 0.75
    variance_multiplier: float = 1.5
    correlation_angle_degrees: float = 40.0
    missing_probability: float = 0.2
    local_fraction: float = 0.25


@dataclass(frozen=True)
class SyntheticDriftResult:
    window_scores: pd.DataFrame
    summary: pd.DataFrame
    validation_scores: pd.DataFrame


@dataclass(frozen=True)
class DriftStats:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_mean: pd.Series
    numeric_std: pd.Series
    numeric_median: pd.Series
    categorical_values: dict[str, np.ndarray]
    categorical_probabilities: dict[str, np.ndarray]


def run_synthetic_drift_experiment(
    dataset: TabularDataset,
    *,
    config: SyntheticDriftExperimentConfig | None = None,
    output_dir: str | Path | None = None,
) -> SyntheticDriftResult:
    config = config or SyntheticDriftExperimentConfig()
    LOGGER.info(
        "synthetic drift experiment started dataset=%s rows=%d scenarios=%s methods=%s output_dir=%s",
        dataset.name,
        len(dataset.frame),
        config.scenarios,
        config.methods,
        output_dir,
    )
    LOGGER.debug("synthetic drift config: %s", asdict(config))
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
        "synthetic drift split ready dataset=%s train=%d val=%d test=%d",
        dataset.name,
        len(split.train),
        len(split.val),
        len(split.test),
    )
    feature_cols = dataset.feature_columns
    LOGGER.info("synthetic drift building preprocessor dataset=%s features=%d", dataset.name, len(feature_cols))
    preprocessor, _ = build_preprocessor(
        split.train,
        feature_cols,
        max_onehot_cardinality=config.max_onehot_cardinality,
        hash_features=config.hash_features,
    )
    LOGGER.info("synthetic drift fitting preprocessor dataset=%s", dataset.name)
    preprocessor.fit(split.train[feature_cols])
    X_ref = transform_to_float32(preprocessor, split.train[feature_cols])
    LOGGER.debug("synthetic drift reference matrix shape=%s", tuple(getattr(X_ref, "shape", ())))
    stats = _fit_drift_stats(
        split.train,
        feature_cols,
        max_onehot_cardinality=config.max_onehot_cardinality,
    )

    val_windows = fixed_count_windows(len(split.val), window_size=config.window_size, prefix="V")
    test_windows = fixed_count_windows(len(split.test), window_size=config.window_size, prefix="W")
    LOGGER.info(
        "synthetic drift windows ready validation=%d test=%d window_size=%d drift_start_window=%d",
        len(val_windows),
        len(test_windows),
        config.window_size,
        config.drift_start_window,
    )
    scorer = DriftScorer(methods=config.methods, config=_drift_runtime_config(config))
    scorer.prepare_reference(X_ref)
    validation_scores = _score_clean_windows(
        X_ref=X_ref,
        frame=split.val,
        feature_cols=feature_cols,
        preprocessor=preprocessor,
        windows=val_windows,
        methods=config.methods,
        dataset_name=dataset.name,
        config=config,
        scorer=scorer,
    )
    LOGGER.info("synthetic drift calibrating thresholds validation_rows=%d", len(validation_scores))
    thresholds = _calibrate_thresholds(validation_scores, config.threshold_quantile)
    LOGGER.info("synthetic drift thresholds ready methods=%d", len(thresholds))
    LOGGER.debug("synthetic drift thresholds=%s", thresholds)

    rows = []
    for scenario in config.scenarios:
        if scenario not in DEFAULT_SCENARIOS:
            raise ValueError(f"unknown synthetic drift scenario: {scenario!r}")
        LOGGER.info("synthetic drift scenario started scenario=%s windows=%d", scenario, len(test_windows))
        for window in test_windows:
            drift_label = int(window.index >= config.drift_start_window)
            raw_window = split.test.iloc[window.start : window.end].copy()
            LOGGER.info(
                "synthetic drift scoring window scenario=%s label=%s index=%d rows=%d drift_label=%d",
                scenario,
                window.label,
                window.index,
                window.size,
                drift_label,
            )
            if drift_label:
                raw_window = inject_controlled_drift(
                    raw_window,
                    feature_cols=feature_cols,
                    stats=stats,
                    scenario=scenario,
                    config=config,
                    seed=config.seed + 1009 * window.index,
                )
                LOGGER.debug("synthetic drift injected scenario=%s window=%s", scenario, window.label)
            X_cur = transform_to_float32(preprocessor, raw_window[feature_cols])
            scores = scorer.compute(X_ref, X_cur)
            for score in scores:
                threshold = thresholds.get(score.method, np.nan)
                LOGGER.debug(
                    "synthetic drift score scenario=%s window=%s method=%s score=%.8f threshold=%s runtime=%.4fs",
                    scenario,
                    window.label,
                    score.method,
                    score.score,
                    threshold,
                    score.runtime_seconds,
                )
                rows.append(
                    {
                        "dataset": dataset.name,
                        "scenario": scenario,
                        "window_index": window.index,
                        "window_label": window.label,
                        "window_start": window.start,
                        "window_end": window.end,
                        "window_size": window.size,
                        "drift_start_window": config.drift_start_window,
                        "drift_label": drift_label,
                        "method": score.method,
                        "score": score.score,
                        "runtime_seconds": score.runtime_seconds,
                        "threshold": threshold,
                        "triggered": bool(np.isfinite(threshold) and score.score >= threshold),
                    }
                )
        LOGGER.info("synthetic drift scenario completed scenario=%s", scenario)

    window_scores = pd.DataFrame(rows)
    LOGGER.info("synthetic drift summarizing detection rows=%d", len(window_scores))
    summary = summarize_detection(window_scores)
    result = SyntheticDriftResult(window_scores=window_scores, summary=summary, validation_scores=validation_scores)
    if output_dir is not None:
        save_synthetic_drift_result(result, output_dir=output_dir, config=config)
    LOGGER.info("synthetic drift experiment completed dataset=%s output_dir=%s", dataset.name, output_dir)
    return result


def inject_controlled_drift(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    stats: DriftStats,
    scenario: str,
    config: SyntheticDriftExperimentConfig,
    seed: int,
) -> pd.DataFrame:
    LOGGER.debug(
        "injecting controlled drift scenario=%s rows=%d seed=%d",
        scenario,
        len(frame),
        seed,
    )
    rng = np.random.default_rng(seed)
    shifted = frame.copy()
    numeric = _select_columns(stats.numeric_columns, config.numeric_fraction, rng)
    categorical = _select_columns(stats.categorical_columns, config.categorical_fraction, rng)
    LOGGER.debug("controlled drift selected numeric=%s categorical=%s", numeric, categorical)

    if scenario == "mean_shift":
        _apply_mean_shift(shifted, numeric, stats, config.mean_delta)
    elif scenario == "variance_shift":
        _apply_variance_shift(shifted, numeric, stats, config.variance_multiplier)
    elif scenario == "correlation_shift":
        _apply_correlation_shift(shifted, numeric, stats, config.correlation_angle_degrees)
    elif scenario == "categorical_prior_shift":
        _apply_categorical_prior_shift(shifted, categorical, stats, rng)
    elif scenario == "missingness_shift":
        _apply_missingness_shift(shifted, numeric + categorical, config.missing_probability, rng)
    elif scenario == "local_subpopulation_shift":
        row_mask = _local_row_mask(shifted, numeric, stats, config.local_fraction, rng)
        _apply_mean_shift(shifted, numeric, stats, config.mean_delta, row_mask=row_mask)
        _apply_missingness_shift(shifted, categorical, config.missing_probability, rng, row_mask=row_mask)
    elif scenario == "concept_like_shift":
        # Negative-control scenario for X-only drift monitors: concept drift changes
        # P(y|X), while the observed covariate distribution is intentionally unchanged.
        pass
    elif scenario == "mixed_shift":
        _apply_mean_shift(shifted, numeric, stats, config.mean_delta * 0.5)
        _apply_variance_shift(shifted, numeric, stats, 1.0 + (config.variance_multiplier - 1.0) * 0.5)
        _apply_categorical_prior_shift(shifted, categorical, stats, rng)
        _apply_missingness_shift(shifted, numeric + categorical, config.missing_probability * 0.5, rng)
    else:
        raise ValueError(f"unknown synthetic drift scenario: {scenario!r}")
    return shifted


def summarize_detection(window_scores: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("summarizing synthetic detection groups=%d", window_scores.groupby(["dataset", "scenario", "method"]).ngroups)
    rows = []
    for (dataset, scenario, method), group in window_scores.groupby(["dataset", "scenario", "method"], sort=True):
        ordered = group.sort_values("window_index")
        labels = ordered["drift_label"].to_numpy(dtype=int)
        scores = ordered["score"].to_numpy(dtype=float)
        triggers = ordered["triggered"].to_numpy(dtype=bool)
        drift_start = int(ordered["drift_start_window"].iloc[0])
        if len(np.unique(labels)) < 2:
            auroc = auprc = float("nan")
        else:
            auroc = _safe_metric(roc_auc_score, labels, scores)
            auprc = _safe_metric(average_precision_score, labels, scores)
        pre = ordered.loc[ordered["window_index"] < drift_start]
        post = ordered.loc[ordered["window_index"] >= drift_start]
        false_alarm_rate = float(pre["triggered"].mean()) if len(pre) else float("nan")
        first_detection = post.loc[post["triggered"], "window_index"].min() if len(post) else np.nan
        detection_delay = (
            float(first_detection - drift_start)
            if pd.notna(first_detection)
            else float("nan")
        )
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "method": method,
                "auroc": auroc,
                "auprc": auprc,
                "detection_delay_windows": detection_delay,
                "false_alarm_rate": false_alarm_rate,
                "n_windows": len(ordered),
                "n_drift_windows": int(labels.sum()),
                "threshold": float(ordered["threshold"].dropna().iloc[0]) if ordered["threshold"].notna().any() else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["scenario", "auroc"], ascending=[True, False])
    LOGGER.info("synthetic detection summary ready rows=%d", len(out))
    return out


def save_synthetic_drift_result(
    result: SyntheticDriftResult,
    *,
    output_dir: str | Path,
    config: SyntheticDriftExperimentConfig,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    LOGGER.info("saving synthetic drift result output_dir=%s", output)
    _write_csv(result.window_scores, output / "synthetic_drift_windows.csv")
    _write_csv(result.summary, output / "synthetic_drift_summary.csv")
    _write_csv(result.validation_scores, output / "synthetic_drift_validation_scores.csv")
    config_path = output / "synthetic_drift_config.json"
    pd.DataFrame([asdict(config)]).to_json(config_path, orient="records", indent=2)
    LOGGER.info("wrote synthetic drift config path=%s", config_path)


def _score_clean_windows(
    *,
    X_ref,
    frame: pd.DataFrame,
    feature_cols: list[str],
    preprocessor,
    windows,
    methods: tuple[str, ...],
    dataset_name: str,
    config: SyntheticDriftExperimentConfig,
    scorer: DriftScorer | None = None,
) -> pd.DataFrame:
    LOGGER.info("scoring clean validation windows dataset=%s windows=%d methods=%s", dataset_name, len(windows), methods)
    rows = []
    if scorer is None:
        scorer = DriftScorer(methods=methods, config=_drift_runtime_config(config))
        scorer.prepare_reference(X_ref)
    for window in windows:
        LOGGER.info("scoring clean window dataset=%s label=%s index=%d rows=%d", dataset_name, window.label, window.index, window.size)
        X_cur = transform_to_float32(preprocessor, frame.iloc[window.start : window.end][feature_cols])
        for score in scorer.compute(X_ref, X_cur):
            LOGGER.debug(
                "clean validation score dataset=%s window=%s method=%s score=%.8f runtime=%.4fs",
                dataset_name,
                window.label,
                score.method,
                score.score,
                score.runtime_seconds,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "window_index": window.index,
                    "window_label": window.label,
                    "method": score.method,
                    "score": score.score,
                    "runtime_seconds": score.runtime_seconds,
                }
            )
    out = pd.DataFrame(rows)
    LOGGER.info("clean validation scoring completed dataset=%s rows=%d", dataset_name, len(out))
    return out


def _calibrate_thresholds(validation_scores: pd.DataFrame, quantile: float) -> dict[str, float]:
    thresholds = {}
    for method, group in validation_scores.groupby("method"):
        scores = group["score"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(scores):
            thresholds[method] = quantile_threshold(scores, quantile=quantile)
            LOGGER.info(
                "synthetic drift threshold calibrated method=%s quantile=%.4f value=%.8f n_scores=%d",
                method,
                quantile,
                thresholds[method],
                len(scores),
            )
        else:
            LOGGER.warning("synthetic drift threshold skipped method=%s reason=no finite scores", method)
    return thresholds


def _drift_runtime_config(config: SyntheticDriftExperimentConfig) -> DriftRuntimeConfig:
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


def _fit_drift_stats(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    max_onehot_cardinality: int,
) -> DriftStats:
    LOGGER.info("fitting drift injection stats train_rows=%d feature_cols=%d", len(train_frame), len(feature_cols))
    feature_types = infer_feature_types(
        train_frame,
        feature_cols,
        max_onehot_cardinality=max_onehot_cardinality,
    )
    numeric = feature_types.numeric
    categorical = feature_types.categorical
    numeric_mean = train_frame[numeric].mean(numeric_only=True) if numeric else pd.Series(dtype=float)
    numeric_std = train_frame[numeric].std(numeric_only=True).replace(0.0, 1.0).fillna(1.0) if numeric else pd.Series(dtype=float)
    numeric_median = train_frame[numeric].median(numeric_only=True) if numeric else pd.Series(dtype=float)
    cat_values: dict[str, np.ndarray] = {}
    cat_probs: dict[str, np.ndarray] = {}
    for col in categorical:
        counts = train_frame[col].astype("string").fillna("__missing__").value_counts(normalize=True, dropna=False)
        if counts.empty:
            cat_values[col] = np.array(["__missing__"], dtype=object)
            cat_probs[col] = np.array([1.0], dtype=float)
        else:
            cat_values[col] = counts.index.astype(object).to_numpy()
            cat_probs[col] = counts.to_numpy(dtype=float)
    stats = DriftStats(
        numeric_columns=numeric,
        categorical_columns=categorical,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        numeric_median=numeric_median,
        categorical_values=cat_values,
        categorical_probabilities=cat_probs,
    )
    LOGGER.info("drift injection stats ready numeric=%d categorical=%d", len(numeric), len(categorical))
    return stats


def _select_columns(columns: list[str], fraction: float, rng: np.random.Generator) -> list[str]:
    if not columns:
        return []
    n = max(1, int(np.ceil(len(columns) * fraction)))
    n = min(n, len(columns))
    return sorted(rng.choice(columns, size=n, replace=False).tolist())


def _apply_mean_shift(
    frame: pd.DataFrame,
    columns: list[str],
    stats: DriftStats,
    delta: float,
    *,
    row_mask=None,
) -> None:
    if not columns:
        return
    rows = _rows(frame, row_mask)
    for col in columns:
        frame.loc[rows, col] = pd.to_numeric(frame.loc[rows, col], errors="coerce") + delta * stats.numeric_std[col]


def _apply_variance_shift(frame: pd.DataFrame, columns: list[str], stats: DriftStats, multiplier: float) -> None:
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        mean = stats.numeric_mean[col]
        frame[col] = mean + multiplier * (values - mean)


def _apply_correlation_shift(
    frame: pd.DataFrame,
    columns: list[str],
    stats: DriftStats,
    angle_degrees: float,
) -> None:
    if len(columns) < 2:
        return
    angle = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    paired = columns[: len(columns) - (len(columns) % 2)]
    for left, right in zip(paired[0::2], paired[1::2], strict=True):
        values = frame[[left, right]].apply(pd.to_numeric, errors="coerce")
        means = np.array([stats.numeric_mean[left], stats.numeric_mean[right]], dtype=float)
        stds = np.array([stats.numeric_std[left], stats.numeric_std[right]], dtype=float)
        standardized = (values.to_numpy(dtype=float) - means) / np.maximum(stds, 1e-12)
        shifted = standardized @ rotation
        frame[[left, right]] = shifted * stds + means


def _apply_categorical_prior_shift(
    frame: pd.DataFrame,
    columns: list[str],
    stats: DriftStats,
    rng: np.random.Generator,
) -> None:
    for col in columns:
        values = stats.categorical_values.get(col)
        probs = stats.categorical_probabilities.get(col)
        if values is None or probs is None or len(values) == 0:
            continue
        shifted_probs = probs[::-1].copy()
        shifted_probs = shifted_probs / shifted_probs.sum()
        frame[col] = rng.choice(values[::-1], size=len(frame), replace=True, p=shifted_probs)


def _apply_missingness_shift(
    frame: pd.DataFrame,
    columns: list[str],
    probability: float,
    rng: np.random.Generator,
    *,
    row_mask=None,
) -> None:
    if not columns:
        return
    rows = _rows(frame, row_mask)
    selected_index = frame.index[rows]
    for col in columns:
        mask = rng.uniform(size=len(selected_index)) < probability
        frame.loc[selected_index[mask], col] = np.nan


def _local_row_mask(
    frame: pd.DataFrame,
    columns: list[str],
    stats: DriftStats,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(frame)
    if n == 0:
        return np.zeros(0, dtype=bool)
    target = max(1, int(np.ceil(n * fraction)))
    if columns:
        col = columns[0]
        values = pd.to_numeric(frame[col], errors="coerce")
        threshold = stats.numeric_median[col]
        mask = (values >= threshold).fillna(False).to_numpy(dtype=bool)
        if mask.sum() >= target:
            chosen = rng.choice(np.flatnonzero(mask), size=target, replace=False)
            out = np.zeros(n, dtype=bool)
            out[chosen] = True
            return out
    out = np.zeros(n, dtype=bool)
    out[rng.choice(n, size=target, replace=False)] = True
    return out


def _rows(frame: pd.DataFrame, row_mask) -> np.ndarray:
    if row_mask is None:
        return np.ones(len(frame), dtype=bool)
    return np.asarray(row_mask, dtype=bool)


def _safe_metric(func, *args, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except ValueError:
        return float("nan")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
    LOGGER.info("wrote CSV path=%s rows=%d columns=%d", path, len(frame), len(frame.columns))
