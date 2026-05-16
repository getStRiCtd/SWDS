from __future__ import annotations

import argparse
import logging
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
from swds.logging_utils import configure_logging


MODEL_CHOICES = ["linear", "hist_gbdt", "gbdt", "lightgbm", "xgboost", "catboost"]
LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swds", description="Temporal tabular drift experiments")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging level; can also be set with SWDS_LOG_LEVEL",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="optional log file path; can also be set with SWDS_LOG_FILE",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("run-synthetic", help="run the full pipeline on generated temporal data")
    synthetic.add_argument("--output", default="results/synthetic", help="directory for result CSV files")
    synthetic.add_argument("--n-samples", type=int, default=12000)
    synthetic.add_argument("--window-size", type=int, default=500)
    synthetic.add_argument("--task", choices=[t.value for t in TaskType], default=TaskType.CLASSIFICATION.value)
    synthetic.add_argument("--model", default="linear", choices=MODEL_CHOICES)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    _add_runtime_args(synthetic)

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
    _add_runtime_args(csv)

    run_config = subparsers.add_parser("run-config", help="run an experiment from a YAML config")
    run_config.add_argument("--config", required=True)
    run_config.add_argument("--output", default=None, help="override output_dir from config")
    _add_runtime_args(run_config)

    run_drift = subparsers.add_parser("run-synthetic-drift", help="run controlled drift injection from YAML config")
    run_drift.add_argument("--config", required=True)
    run_drift.add_argument("--output", default=None, help="override synthetic drift output directory")
    _add_runtime_args(run_drift)

    run_ablation = subparsers.add_parser("run-ablation", help="run configured ablations from a YAML config")
    run_ablation.add_argument("--config", required=True)
    run_ablation.add_argument("--output", default=None, help="override ablation output directory")
    _add_runtime_args(run_ablation)

    run_batch = subparsers.add_parser("run-batch", help="run many YAML configs and record exclusions")
    run_batch.add_argument("--configs", nargs="+", required=True)
    run_batch.add_argument("--output", default="results/batch")
    run_batch.add_argument("--fail-fast", action="store_true", help="stop after the first failed config")
    run_batch.add_argument("--rerun-completed", action="store_true", help="recompute configs whose result files already exist")
    _add_runtime_args(run_batch)

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

    for command_parser in (synthetic, make_syn, csv, run_config, run_drift, run_ablation, run_batch, report, tabred):
        _add_subcommand_logging_args(command_parser)

    args = parser.parse_args(argv)
    configure_logging(level=args.log_level, log_file=args.log_file)
    LOGGER.info("command started: %s", args.command)
    LOGGER.debug("parsed arguments: %s", vars(args))
    try:
        exit_code = _run_command(args)
    except Exception:
        LOGGER.exception("command failed: %s", args.command)
        raise
    LOGGER.info("command finished: %s exit_code=%s", args.command, exit_code)
    return exit_code


def _run_command(args: argparse.Namespace) -> int:
    if args.command == "run-synthetic":
        task = normalize_task_type(args.task)
        LOGGER.info(
            "generating synthetic dataset n_samples=%d task=%s seed=%d",
            args.n_samples,
            task.value,
            args.seed,
        )
        dataset, drift_start = make_synthetic_temporal_dataset(
            SyntheticSpec(n_samples=args.n_samples, task_type=task, seed=args.seed)
        )
        config = ExperimentConfig(
            window_size=args.window_size,
            model_name=args.model,
            methods=tuple(args.methods),
            seed=args.seed,
        )
        config = _with_runtime_overrides(config, args)
        run_temporal_experiment(dataset, config=config, output_dir=args.output)
        LOGGER.info("synthetic run completed output=%s drift_start_row=%d", args.output, drift_start)
        print(f"synthetic drift starts at row {drift_start}; results written to {args.output}")
        return 0

    if args.command == "make-synthetic":
        task = normalize_task_type(args.task)
        LOGGER.info(
            "generating synthetic CSV n_samples=%d task=%s seed=%d output=%s",
            args.n_samples,
            task.value,
            args.seed,
            args.output,
        )
        dataset, drift_start = make_synthetic_temporal_dataset(
            SyntheticSpec(n_samples=args.n_samples, task_type=task, seed=args.seed)
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        dataset.frame.to_csv(output, index=False)
        LOGGER.info("synthetic CSV written rows=%d output=%s", len(dataset.frame), output)
        print(f"wrote {len(dataset.frame)} rows to {output}; drift starts at row {drift_start}")
        return 0

    if args.command == "run-csv":
        LOGGER.info("loading CSV dataset path=%s target=%s time=%s", args.path, args.target, args.time)
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
        config = _with_runtime_overrides(config, args)
        run_temporal_experiment(dataset, config=config, output_dir=args.output)
        LOGGER.info("CSV run completed output=%s", args.output)
        print(f"results written to {args.output}")
        return 0

    if args.command == "run-config":
        LOGGER.info("loading experiment config path=%s", args.config)
        raw_config = load_experiment_yaml(args.config)
        dataset = load_dataset_from_experiment_config(raw_config)
        config = experiment_config_from_mapping(raw_config)
        config = _with_runtime_overrides(config, args)
        output = args.output or output_dir_from_config(raw_config)
        run_temporal_experiment(dataset, config=config, output_dir=output)
        LOGGER.info("config run completed config=%s output=%s", args.config, output)
        print(f"results written to {output}")
        return 0

    if args.command == "run-synthetic-drift":
        LOGGER.info("loading synthetic drift config path=%s", args.config)
        raw_config = load_experiment_yaml(args.config)
        dataset = load_dataset_from_experiment_config(raw_config)
        config = synthetic_drift_config_from_mapping(raw_config)
        config = _with_runtime_overrides(config, args)
        output = args.output or synthetic_drift_output_dir_from_config(raw_config)
        run_synthetic_drift_experiment(dataset, config=config, output_dir=output)
        LOGGER.info("synthetic drift run completed config=%s output=%s", args.config, output)
        print(f"synthetic drift results written to {output}")
        return 0

    if args.command == "run-ablation":
        LOGGER.info("loading ablation config path=%s", args.config)
        raw_config = load_experiment_yaml(args.config)
        dataset = load_dataset_from_experiment_config(raw_config)
        base_config = experiment_config_from_mapping(raw_config)
        base_config = _with_runtime_overrides(base_config, args)
        ablation_config = ablation_config_from_mapping(raw_config)
        output = args.output or str(Path(output_dir_from_config(raw_config)).with_name("ablation"))
        run_ablation_experiment(dataset, base_config=base_config, config=ablation_config, output_dir=output)
        LOGGER.info("ablation run completed config=%s output=%s", args.config, output)
        print(f"ablation results written to {output}")
        return 0

    if args.command == "run-batch":
        LOGGER.info("batch run started configs=%d output=%s fail_fast=%s", len(args.configs), args.output, args.fail_fast)
        run_config_batch(
            args.configs,
            output_dir=args.output,
            continue_on_error=not args.fail_fast,
            runtime_overrides=_runtime_overrides(args),
            skip_completed=not args.rerun_completed,
        )
        LOGGER.info("batch run completed output=%s", args.output)
        print(f"batch results written to {args.output}")
        return 0

    if args.command == "build-report":
        LOGGER.info("report build started result_dirs=%d output=%s", len(args.results), args.output)
        build_report_from_results(args.results, output_dir=args.output)
        LOGGER.info("report build completed output=%s", args.output)
        print(f"report written to {args.output}")
        return 0

    if args.command == "prepare-tabred":
        LOGGER.info(
            "TabReD command started repo=%s output_root=%s datasets=%s validate_only=%s copy=%s no_clone=%s",
            args.repo,
            args.output_root,
            args.datasets,
            args.validate_only,
            args.copy,
            args.no_clone,
        )
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
            LOGGER.exception("prepare-tabred failed")
            print(f"prepare-tabred failed: {exc}")
            return 2
        for item in results:
            LOGGER.info(
                "TabReD dataset status dataset=%s status=%s target=%s message=%s",
                item.dataset,
                item.status,
                item.target_path,
                item.message,
            )
            print(f"{item.dataset}: {item.status} -> {item.target_path} ({item.message})")
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


def _add_subcommand_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default=argparse.SUPPRESS,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging level; can also be set before the subcommand or with SWDS_LOG_LEVEL",
    )
    parser.add_argument(
        "--log-file",
        default=argparse.SUPPRESS,
        help="optional log file path; can also be set before the subcommand or with SWDS_LOG_FILE",
    )


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-jobs", type=int, default=None, help="parallel jobs for drift scoring; -1 uses all cores")
    parser.add_argument("--swds-backend", choices=["auto", "numpy", "torch"], default=None)
    parser.add_argument("--swds-device", default=None, help="torch device for SWDS, for example cuda or cuda:0")
    parser.add_argument("--ks-max-features", type=int, default=None, help="maximum features sampled for KS baselines")
    parser.add_argument("--psi-max-features", type=int, default=None, help="maximum features sampled for PSI baselines")
    parser.add_argument("--psi-n-bins", type=int, default=None, help="number of PSI bins")
    parser.add_argument("--mmd-max-samples", type=int, default=None, help="maximum rows per side for MMD")
    parser.add_argument("--energy-max-samples", type=int, default=None, help="maximum rows per side for energy distance")
    parser.add_argument("--c2st-max-samples", type=int, default=None, help="maximum total rows for C2ST")
    parser.add_argument("--c2st-n-splits", type=int, default=None, help="cross-validation splits for C2ST")
    parser.add_argument("--c2st-n-jobs", type=int, default=None, help="parallel jobs inside C2ST cross-validation")


def _runtime_overrides(args: argparse.Namespace) -> dict[str, object]:
    values = {}
    for attr in (
        "n_jobs",
        "swds_backend",
        "swds_device",
        "ks_max_features",
        "psi_max_features",
        "psi_n_bins",
        "mmd_max_samples",
        "energy_max_samples",
        "c2st_max_samples",
        "c2st_n_splits",
        "c2st_n_jobs",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            values[attr] = value
    return values


def _with_runtime_overrides(config, args: argparse.Namespace):
    overrides = _runtime_overrides(args)
    if not overrides:
        return config
    return type(config)(**{**vars(config), **overrides})


if __name__ == "__main__":
    raise SystemExit(main())
