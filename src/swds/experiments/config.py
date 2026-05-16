from __future__ import annotations

import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from swds.data.loaders import load_dataset_from_spec
from swds.data.schema import TaskType, normalize_task_type
from swds.drift.registry import DEFAULT_METHODS
from swds.experiments.ablation import AblationConfig
from swds.experiments.pipeline import ExperimentConfig
from swds.experiments.synthetic_drift import SyntheticDriftExperimentConfig
from swds.experiments.synthetic import SyntheticSpec, make_synthetic_temporal_dataset


LOGGER = logging.getLogger(__name__)


def load_experiment_yaml(path: str | Path) -> dict[str, Any]:
    LOGGER.info("reading experiment YAML path=%s", path)
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        LOGGER.error("experiment config is not a mapping path=%s", path)
        raise ValueError(f"experiment config must be a mapping: {path}")
    LOGGER.info("experiment YAML loaded path=%s keys=%s", path, sorted(config))
    return config


def load_dataset_from_experiment_config(config: dict[str, Any]):
    dataset_spec = dict(config.get("dataset", {}))
    dataset_type = str(dataset_spec.get("type", "csv")).lower()
    LOGGER.info("loading dataset from experiment config type=%s", dataset_type)
    if dataset_type == "synthetic":
        task_type = normalize_task_type(dataset_spec.get("task_type", dataset_spec.get("task", "classification")))
        LOGGER.info(
            "generating configured synthetic dataset n_samples=%s task=%s seed=%s",
            dataset_spec.get("n_samples", 12000),
            TaskType(task_type).value,
            dataset_spec.get("seed", config.get("seed", 42)),
        )
        dataset, _ = make_synthetic_temporal_dataset(
            SyntheticSpec(
                n_samples=int(dataset_spec.get("n_samples", 12000)),
                n_numeric=int(dataset_spec.get("n_numeric", 10)),
                n_categorical=int(dataset_spec.get("n_categorical", 3)),
                drift_start_frac=float(dataset_spec.get("drift_start_frac", 0.75)),
                task_type=TaskType(task_type),
                seed=int(dataset_spec.get("seed", config.get("seed", 42))),
            )
        )
        if "name" in dataset_spec:
            LOGGER.info("renaming configured synthetic dataset name=%s", dataset_spec["name"])
            dataset = type(dataset)(
                name=dataset_spec["name"],
                frame=dataset.frame,
                target_col=dataset.target_col,
                time_col=dataset.time_col,
                task_type=dataset.task_type,
                split_col=dataset.split_col,
                source=dataset.source,
            )
        LOGGER.info("configured synthetic dataset ready name=%s rows=%d", dataset.name, dataset.n_samples)
        return dataset
    return load_dataset_from_spec(dataset_spec)


def experiment_config_from_mapping(config: dict[str, Any]) -> ExperimentConfig:
    experiment = dict(config.get("experiment", config))
    retraining = dict(experiment.pop("retraining", {}))
    if retraining:
        experiment.setdefault("run_retraining", retraining.get("enabled", True))
        experiment.setdefault("retraining_periods", retraining.get("periods", (4,)))
        experiment.setdefault("retraining_history_modes", retraining.get("history_modes", ("all", "rolling")))
        experiment.setdefault("rolling_history_windows", retraining.get("rolling_history_windows", 4))
        experiment.setdefault("oracle_min_quality_drop", retraining.get("oracle_min_quality_drop", 0.02))

    if "methods" not in experiment:
        experiment["methods"] = DEFAULT_METHODS

    allowed = {field.name for field in fields(ExperimentConfig)}
    values = {key: value for key, value in experiment.items() if key in allowed}
    for key in ("methods", "retraining_periods", "retraining_history_modes"):
        if key in values and not isinstance(values[key], tuple):
            values[key] = tuple(values[key])
    out = ExperimentConfig(**values)
    LOGGER.info("experiment config parsed model=%s methods=%s window_size=%d", out.model_name, out.methods, out.window_size)
    LOGGER.debug("experiment config values=%s", out)
    return out


def synthetic_drift_config_from_mapping(config: dict[str, Any]) -> SyntheticDriftExperimentConfig:
    experiment = dict(config.get("experiment", {}))
    drift = dict(config.get("synthetic_drift", {}))
    for key in (
        "train_frac",
        "val_frac",
        "test_frac",
        "window_size",
        "methods",
        "seed",
        "max_onehot_cardinality",
        "hash_features",
        "threshold_quantile",
    ):
        if key in experiment and key not in drift:
            drift[key] = experiment[key]
    allowed = {field.name for field in fields(SyntheticDriftExperimentConfig)}
    values = {key: value for key, value in drift.items() if key in allowed}
    for key in ("methods", "scenarios"):
        if key in values and not isinstance(values[key], tuple):
            values[key] = tuple(values[key])
    out = SyntheticDriftExperimentConfig(**values)
    LOGGER.info("synthetic drift config parsed scenarios=%s methods=%s", out.scenarios, out.methods)
    LOGGER.debug("synthetic drift config values=%s", out)
    return out


def ablation_config_from_mapping(config: dict[str, Any]) -> AblationConfig:
    raw = dict(config.get("ablation", {}))
    allowed = {field.name for field in fields(AblationConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    for key in ("projection_counts", "window_sizes", "threshold_quantiles", "reference_modes", "representation_modes"):
        if key in values and not isinstance(values[key], tuple):
            values[key] = tuple(values[key])
    out = AblationConfig(**values)
    LOGGER.info("ablation config parsed")
    LOGGER.debug("ablation config values=%s", out)
    return out


def output_dir_from_config(config: dict[str, Any], default: str = "results/config_run") -> str:
    return str(config.get("output_dir") or config.get("output") or default)


def synthetic_drift_output_dir_from_config(config: dict[str, Any]) -> str:
    if config.get("synthetic_drift_output_dir"):
        return str(config["synthetic_drift_output_dir"])
    base = Path(output_dir_from_config(config))
    return str(base.with_name(f"{base.name}_synthetic_drift"))
