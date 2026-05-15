from __future__ import annotations

import hashlib
import json
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


def run_config_batch(
    config_paths: list[str],
    *,
    output_dir: str | Path,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for path_str in config_paths:
        path = Path(path_str)
        try:
            raw_config = load_experiment_yaml(path)
            dataset = load_dataset_from_experiment_config(raw_config)
            experiment_config = experiment_config_from_mapping(raw_config)
            run_output = Path(output_dir_from_config(raw_config, default=str(output / path.stem)))
            if not run_output.is_absolute():
                run_output = output / path.stem
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
        except Exception as exc:
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
                break

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output / "run_manifest.csv", index=False)
    manifest.loc[manifest["status"] != "completed"].to_csv(output / "dataset_exclusions.csv", index=False)
    _write_environment_manifest(output / "environment_manifest.json")
    return manifest


def _write_environment_manifest(path: Path) -> None:
    payload = {
        "swds_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
