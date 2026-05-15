from __future__ import annotations

import argparse
from pathlib import Path

from swds.data.loaders import load_csv_dataset
from swds.data.schema import TaskType, normalize_task_type
from swds.drift.registry import DEFAULT_METHODS
from swds.experiments.config import (
    ablation_config_from_mapping,
    experiment_config_from_mapping,
    load_dataset_from_experiment_config,
    load_experiment_yaml,
    output_dir_from_config,
    synthetic_drift_config_from_mapping,
    synthetic_drift_output_dir_from_config,
)
from swds.experiments.ablation import run_ablation_experiment
from swds.experiments.aggregate import run_config_batch
from swds.experiments.pipeline import ExperimentConfig, run_temporal_experiment
from swds.experiments.reporting import build_report_from_results
from swds.experiments.synthetic import SyntheticSpec, make_synthetic_temporal_dataset
from swds.experiments.synthetic_drift import run_synthetic_drift_experiment
from swds.experiments.tabred_prepare import prepare_tabred, validate_tabred_root


MODEL_CHOICES = ["linear", "hist_gbdt", "gbdt", "lightgbm", "xgboost", "catboost"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swds", description="Temporal tabular drift experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("run-synthetic", help="run the full pipeline on generated temporal data")
    synthetic.add_argument("--output", default="results/synthetic", help="directory for result CSV files")
    synthetic.add_argument("--n-samples", type=int, default=12000)
    synthetic.add_argument("--window-size", type=int, default=500)
    synthetic.add_argument("--task", choices=[t.value for t in TaskType], default=TaskType.CLASSIFICATION.value)
    synthetic.add_argument("--model", default="linear", choices=MODEL_CHOICES)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))

    make_syn = subparsers.add_parser("make-synthetic", help="write generated temporal data to CSV")
    make_syn.add_argument("--output", default="data/processed/synthetic_temporal.csv")
    make_syn.add_argument("--n-samples", type=int, default=12000)
    make_syn.add_argument("--task", choices=[t.value for t in TaskType], default=TaskType.CLASSIFICATION.value)
    make_syn.add_argument("--seed", type=int, default=42)

    csv = subparsers.add_parser("run-csv", help="run the pipeline on a temporal CSV dataset")
    csv.add_argument("--path", required=True)
    csv.add_argument("--target", required=True)
    csv.add_argument("--time", required=True)
    csv.add_argument("--task", choices=[t.value for t in TaskType], default=None)
    csv.add_argument("--name", default=None)
    csv.add_argument("--split-col", default=None)
    csv.add_argument("--output", default="results/csv_run")
    csv.add_argument("--window-size", type=int, default=1000)
    csv.add_argument("--model", default="linear", choices=MODEL_CHOICES)
    csv.add_argument("--seed", type=int, default=42)
    csv.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))

    run_config = subparsers.add_parser("run-config", help="run an experiment from a YAML config")
    run_config.add_argument("--config", required=True)
    run_config.add_argument("--output", default=None, help="override output_dir from config")

    run_drift = subparsers.add_parser("run-synthetic-drift", help="run controlled drift injection from YAML config")
    run_drift.add_argument("--config", required=True)
    run_drift.add_argument("--output", default=None, help="override synthetic drift output directory")

    run_ablation = subparsers.add_parser("run-ablation", help="run configured ablations from a YAML config")
    run_ablation.add_argument("--config", required=True)
    run_ablation.add_argument("--output", default=None, help="override ablation output directory")

    run_batch = subparsers.add_parser("run-batch", help="run many YAML configs and record exclusions")
    run_batch.add_argument("--configs", nargs="+", required=True)
    run_batch.add_argument("--output", default="results/batch")
    run_batch.add_argument("--fail-fast", action="store_true", help="stop after the first failed config")

    report = subparsers.add_parser("build-report", help="build paper tables and figures from result directories")
    report.add_argument("--results", nargs="+", required=True)
    report.add_argument("--output", default="results/report")

    tabred = subparsers.add_parser("prepare-tabred", help="clone, preprocess, and link TabReD datasets")
    tabred.add_argument("--repo", default="data/raw/tabred_repo")
    tabred.add_argument("--output-root", default="data/raw/tabred")
    tabred.add_argument("--datasets", nargs="+", default=["all"])
    tabred.add_argument("--no-clone", action="store_true")
    tabred.add_argument("--copy", action="store_true", help="copy datasets instead of symlinking")
    tabred.add_argument("--validate-only", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run-synthetic":
        task = normalize_task_type(args.task)
        dataset, drift_start = make_synthetic_temporal_dataset(
            SyntheticSpec(n_samples=args.n_samples, task_type=task, seed=args.seed)
        )
        config = ExperimentConfig(
            window_size=args.window_size,
            model_name=args.model,
            methods=tuple(args.methods),
            seed=args.seed,
        )
        run_temporal_experiment(dataset, config=config, output_dir=args.output)
        print(f"synthetic drift starts at row {drift_start}; results written to {args.output}")
        return 0

    if args.command == "make-synthetic":
        task = normalize_task_type(args.task)
        dataset, drift_start = make_synthetic_temporal_dataset(
            SyntheticSpec(n_samples=args.n_samples, task_type=task, seed=args.seed)
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        dataset.frame.to_csv(output, index=False)
        print(f"wrote {len(dataset.frame)} rows to {output}; drift starts at row {drift_start}")
        return 0

    if args.command == "run-csv":
        dataset = load_csv_dataset(
            args.path,
            target_col=args.target,
            time_col=args.time,
            task_type=args.task,
            name=args.name,
            split_col=args.split_col,
        )
        config = ExperimentConfig(
            window_size=args.window_size,
            model_name=args.model,
            methods=tuple(args.methods),
            seed=args.seed,
        )
        run_temporal_experiment(dataset, config=config, output_dir=args.output)
        print(f"results written to {args.output}")
        return 0

    if args.command == "run-config":
        raw_config = load_experiment_yaml(args.config)
        dataset = load_dataset_from_experiment_config(raw_config)
        config = experiment_config_from_mapping(raw_config)
        output = args.output or output_dir_from_config(raw_config)
        run_temporal_experiment(dataset, config=config, output_dir=output)
        print(f"results written to {output}")
        return 0

    if args.command == "run-synthetic-drift":
        raw_config = load_experiment_yaml(args.config)
        dataset = load_dataset_from_experiment_config(raw_config)
        config = synthetic_drift_config_from_mapping(raw_config)
        output = args.output or synthetic_drift_output_dir_from_config(raw_config)
        run_synthetic_drift_experiment(dataset, config=config, output_dir=output)
        print(f"synthetic drift results written to {output}")
        return 0

    if args.command == "run-ablation":
        raw_config = load_experiment_yaml(args.config)
        dataset = load_dataset_from_experiment_config(raw_config)
        base_config = experiment_config_from_mapping(raw_config)
        ablation_config = ablation_config_from_mapping(raw_config)
        output = args.output or str(Path(output_dir_from_config(raw_config)).with_name("ablation"))
        run_ablation_experiment(dataset, base_config=base_config, config=ablation_config, output_dir=output)
        print(f"ablation results written to {output}")
        return 0

    if args.command == "run-batch":
        run_config_batch(args.configs, output_dir=args.output, continue_on_error=not args.fail_fast)
        print(f"batch results written to {args.output}")
        return 0

    if args.command == "build-report":
        build_report_from_results(args.results, output_dir=args.output)
        print(f"report written to {args.output}")
        return 0

    if args.command == "prepare-tabred":
        try:
            if args.validate_only:
                results = validate_tabred_root(args.output_root)
            else:
                results = prepare_tabred(
                    repo_dir=args.repo,
                    output_root=args.output_root,
                    datasets=tuple(args.datasets),
                    clone_if_missing=not args.no_clone,
                    link=not args.copy,
                )
        except (FileNotFoundError, ImportError, ValueError) as exc:
            print(f"prepare-tabred failed: {exc}")
            return 2
        for item in results:
            print(f"{item.dataset}: {item.status} -> {item.target_path} ({item.message})")
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
