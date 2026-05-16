from __future__ import annotations

import hashlib
import json
import logging
import platform
import traceback
from pathlib import Path

import pandas as pd

from swds import __version__
from swds.experiments.config import (
    experiment_config_from_mapping,
    load_dataset_from_experiment_config,
    load_experiment_yaml,
    output_dir_from_config,
)
from swds.experiments.pipeline import run_temporal_experiment


LOGGER = logging.getLogger(__name__)


def run_config_batch(
    config_paths: list[str],
    *,
    output_dir: str | Path,
    continue_on_error: bool = True,
    runtime_overrides: dict[str, object] | None = None,
    skip_completed: bool = True,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "config batch started configs=%d output_dir=%s continue_on_error=%s",
        len(config_paths),
        output,
        continue_on_error,
    )
    runtime_overrides = dict(runtime_overrides or {})
    if runtime_overrides:
        LOGGER.info("batch runtime overrides=%s", runtime_overrides)
    rows = []

    for run_index, path_str in enumerate(config_paths):
        path = Path(path_str)
        LOGGER.info("batch config started index=%d path=%s", run_index, path)
        try:
            raw_config = load_experiment_yaml(path)
            run_output = Path(output_dir_from_config(raw_config, default=str(output / path.stem)))
            if not run_output.is_absolute():
                run_output = output / path.stem
            if skip_completed and _completed_run_exists(run_output):
                dataset_name, n_samples = _completed_dataset_metadata(run_output)
                LOGGER.info("batch config skipped because result already exists index=%d path=%s output=%s", run_index, path, run_output)
                rows.append(
                    {
                        "config_path": str(path),
                        "config_sha256": _sha256(path),
                        "status": "completed",
                        "dataset": dataset_name,
                        "output_dir": str(run_output),
                        "n_samples": n_samples,
                        "reason": "already completed",
                        "traceback": "",
                    }
                )
                continue
            dataset = load_dataset_from_experiment_config(raw_config)
            experiment_config = experiment_config_from_mapping(raw_config)
            if runtime_overrides:
                experiment_config = type(experiment_config)(**{**vars(experiment_config), **runtime_overrides})
            LOGGER.info(
                "batch experiment running index=%d dataset=%s output=%s",
                run_index,
                dataset.name,
                run_output,
            )
            run_temporal_experiment(dataset, config=experiment_config, output_dir=run_output)
            rows.append(
                {
                    "config_path": str(path),
                    "config_sha256": _sha256(path),
                    "status": "completed",
                    "dataset": dataset.name,
                    "output_dir": str(run_output),
                    "n_samples": dataset.n_samples,
                    "reason": "",
                    "traceback": "",
                }
            )
            LOGGER.info("batch config completed index=%d path=%s output=%s", run_index, path, run_output)
        except Exception as exc:
            LOGGER.exception("batch config excluded index=%d path=%s", run_index, path)
            rows.append(
                {
                    "config_path": str(path),
                    "config_sha256": _sha256(path) if path.exists() else "",
                    "status": "excluded",
                    "dataset": "",
                    "output_dir": "",
                    "n_samples": "",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            if not continue_on_error:
                LOGGER.warning("batch stopping after failure index=%d path=%s", run_index, path)
                break

    manifest = pd.DataFrame(rows)
    manifest_path = output / "run_manifest.csv"
    exclusions_path = output / "dataset_exclusions.csv"
    manifest.to_csv(manifest_path, index=False)
    manifest.loc[manifest["status"] != "completed"].to_csv(exclusions_path, index=False)
    LOGGER.info("batch manifest written path=%s rows=%d", manifest_path, len(manifest))
    LOGGER.info("batch exclusions written path=%s rows=%d", exclusions_path, int((manifest["status"] != "completed").sum()))
    _write_environment_manifest(output / "environment_manifest.json")
    LOGGER.info("config batch completed output_dir=%s rows=%d", output, len(manifest))
    return manifest


def _write_environment_manifest(path: Path) -> None:
    payload = {
        "swds_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("environment manifest written path=%s", path)


def _completed_run_exists(path: Path) -> bool:
    required = (
        "window_scores.csv",
        "validation_scores.csv",
        "correlations.csv",
        "dataset_summary.csv",
        "config.json",
    )
    return all((path / name).exists() for name in required)


def _completed_dataset_metadata(path: Path) -> tuple[str, int | str]:
    summary_path = path / "dataset_summary.csv"
    if not summary_path.exists():
        return "", ""
    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        LOGGER.warning("failed to read completed dataset summary path=%s", summary_path)
        return "", ""
    if summary.empty:
        return "", ""
    row = summary.iloc[0]
    return str(row.get("dataset", "")), row.get("n_samples", "")


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
