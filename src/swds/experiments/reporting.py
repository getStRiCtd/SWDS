from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from swds.analysis.plots import correlation_boxplot, timeline_plot
from swds.analysis.stats import friedman_test_by_pair, pairwise_wilcoxon_vs_reference

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt


def build_report_from_results(result_dirs: list[str], *, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    loaded = _load_result_tables([Path(path) for path in result_dirs])
    written: dict[str, Path] = {}

    if "dataset_summary" in loaded:
        written["dataset_summary"] = _write(loaded["dataset_summary"], tables / "table_1_dataset_summary.csv")
    if "window_scores" in loaded:
        written["downstream_quality"] = _write(
            _downstream_quality_table(loaded["window_scores"]),
            tables / "table_2_downstream_quality.csv",
        )
    if "correlations" in loaded:
        corr = loaded["correlations"].copy()
        corr["dataset_model"] = corr["dataset"].astype(str) + "::" + corr["model"].astype(str)
        summary = _correlation_summary(corr)
        written["drift_quality_correlation"] = _write(summary, tables / "table_3_drift_quality_correlation.csv")
        written["friedman_test"] = _write(
            friedman_test_by_pair(corr, pair_col="dataset_model"),
            tables / "table_3b_friedman_test.csv",
        )
        written["correlation_boxplot"] = correlation_boxplot(corr, output_path=figures / "correlation_boxplot.png")
    if "synthetic_drift_summary" in loaded:
        written["synthetic_drift_detection"] = _write(
            _synthetic_detection_summary(loaded["synthetic_drift_summary"]),
            tables / "table_4_synthetic_drift_detection.csv",
        )
        written["detection_delay_plot"] = _detection_delay_plot(
            loaded["synthetic_drift_summary"],
            figures / "detection_delay.png",
        )
    if "retraining_policy_summary" in loaded:
        written["retraining_policy"] = _write(
            _retraining_summary(loaded["retraining_policy_summary"]),
            tables / "table_5_retraining_policy.csv",
        )
    if "ablation_correlations" in loaded:
        written["ablation"] = _write(
            _ablation_summary(loaded["ablation_correlations"], loaded.get("ablation_runtimes")),
            tables / "table_6_ablation.csv",
        )
        if "ablation_runtimes" in loaded:
            written["runtime_plot"] = _runtime_plot(loaded["ablation_runtimes"], figures / "runtime_vs_ablation.png")
        written["ablation_plot"] = _ablation_plot(loaded["ablation_correlations"], figures / "ablation_correlations.png")
    if "window_scores" in loaded:
        first = loaded["window_scores"].dropna(subset=["score", "quality"]).head(10_000)
        if not first.empty:
            method = "swds" if "swds" in set(first["method"]) else str(first["method"].iloc[0])
            written["timeline_plot"] = timeline_plot(first, method=method, output_path=figures / "timeline.png")
            written["scatter_plot"] = _scatter_plot(first, method=method, output_path=figures / "swds_vs_quality_drop.png")
    if "retraining_policy_summary" in loaded:
        written["retraining_regret_plot"] = _retraining_regret_plot(
            loaded["retraining_policy_summary"],
            figures / "retraining_regret.png",
        )

    return written


def _load_result_tables(result_dirs: list[Path]) -> dict[str, pd.DataFrame]:
    names = {
        "window_scores": "window_scores.csv",
        "validation_scores": "validation_scores.csv",
        "correlations": "correlations.csv",
        "dataset_summary": "dataset_summary.csv",
        "retraining_policy_summary": "retraining_policy_summary.csv",
        "synthetic_drift_summary": "synthetic_drift_summary.csv",
        "ablation_correlations": "ablation_correlations.csv",
        "ablation_runtimes": "ablation_runtimes.csv",
    }
    tables: dict[str, list[pd.DataFrame]] = {key: [] for key in names}
    for result_dir in result_dirs:
        for key, filename in names.items():
            path = result_dir / filename
            if path.exists():
                frame = pd.read_csv(path)
                frame.insert(0, "result_dir", str(result_dir))
                tables[key].append(frame)
    return {key: pd.concat(parts, ignore_index=True) for key, parts in tables.items() if parts}


def _correlation_summary(correlations: pd.DataFrame) -> pd.DataFrame:
    ranks = correlations.copy()
    ranks["rank"] = ranks.groupby("dataset_model")["spearman"].rank(ascending=False, method="min")
    summary = (
        ranks.groupby("method", sort=True)
        .agg(
            median_spearman=("spearman", "median"),
            median_spearman_ci_low=("spearman_ci_low", "median") if "spearman_ci_low" in ranks else ("spearman", "median"),
            median_spearman_ci_high=("spearman_ci_high", "median") if "spearman_ci_high" in ranks else ("spearman", "median"),
            mean_rank=("rank", "mean"),
            wins=("rank", lambda x: int((x == 1).sum())),
            n_pairs=("dataset_model", "nunique"),
        )
        .reset_index()
        .sort_values(["median_spearman", "wins"], ascending=[False, False])
    )
    try:
        tests = pairwise_wilcoxon_vs_reference(ranks, reference_method="swds", pair_col="dataset_model")
        summary = summary.merge(tests[["method", "p_value", "p_value_bh", "reject_bh"]], on="method", how="left")
    except ValueError:
        summary["p_value"] = pd.NA
        summary["p_value_bh"] = pd.NA
        summary["reject_bh"] = pd.NA
    return summary


def _downstream_quality_table(window_scores: pd.DataFrame) -> pd.DataFrame:
    base = (
        window_scores.drop_duplicates(["result_dir", "dataset", "model", "window_index"])
        .sort_values(["result_dir", "dataset", "model", "window_index"])
    )
    rows = []
    for (result_dir, dataset, model), group in base.groupby(["result_dir", "dataset", "model"], sort=True):
        first = group.iloc[0]
        final = group.iloc[-1]
        rows.append(
            {
                "result_dir": result_dir,
                "dataset": dataset,
                "model": model,
                "primary_metric": first["primary_metric"],
                "validation_quality": first["reference_quality"],
                "test_initial_window_quality": first["quality"],
                "test_final_window_quality": final["quality"],
                "quality_drop_final": final["quality_drop"],
            }
        )
    return pd.DataFrame(rows)


def _synthetic_detection_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("method", sort=True)
        .agg(
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            median_detection_delay=("detection_delay_windows", "median"),
            mean_false_alarm_rate=("false_alarm_rate", "mean"),
            n_scenarios=("scenario", "nunique"),
        )
        .reset_index()
        .sort_values(["mean_auroc", "mean_auprc"], ascending=[False, False])
    )


def _retraining_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["policy", "history_mode"], sort=True)
        .agg(
            median_regret=("cumulative_regret", "median"),
            mean_quality=("mean_quality", "mean"),
            worst_window_quality=("worst_window_quality", "mean"),
            mean_retrains=("n_retrains", "mean"),
            false_retrain_share=("false_retrain_share", "mean"),
        )
        .reset_index()
        .sort_values("median_regret")
    )


def _ablation_summary(correlations: pd.DataFrame, runtimes: pd.DataFrame | None) -> pd.DataFrame:
    summary = (
        correlations.groupby(["ablation_type", "ablation_value", "method"], sort=True)
        .agg(median_spearman=("spearman", "median"), n_runs=("run_id", "nunique"))
        .reset_index()
    )
    if runtimes is not None and not runtimes.empty:
        rt = (
            runtimes.groupby(["ablation_type", "ablation_value", "method"], sort=True)
            .agg(runtime_median=("runtime_median", "median"), runtime_iqr=("runtime_iqr", "median"))
            .reset_index()
        )
        summary = summary.merge(rt, on=["ablation_type", "ablation_value", "method"], how="left")
    return summary.sort_values(["ablation_type", "median_spearman"], ascending=[True, False])


def _detection_delay_plot(summary: pd.DataFrame, output_path: Path) -> Path:
    data = summary.dropna(subset=["detection_delay_windows"])
    fig, ax = plt.subplots(figsize=(8, 4))
    if not data.empty:
        data.boxplot(column="detection_delay_windows", by="method", ax=ax, rot=45)
    ax.set_title("Detection Delay")
    ax.figure.suptitle("")
    ax.set_xlabel("method")
    ax.set_ylabel("windows")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _scatter_plot(window_scores: pd.DataFrame, *, method: str, output_path: Path) -> Path:
    data = window_scores.loc[window_scores["method"] == method].dropna(subset=["score", "quality_drop"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(data["score"], data["quality_drop"], s=18, alpha=0.7)
    if len(data) >= 2:
        coef = pd.Series(data["quality_drop"]).corr(pd.Series(data["score"]), method="spearman")
        ax.set_title(f"{method} vs quality drop (rho={coef:.2f})")
    ax.set_xlabel("drift score")
    ax.set_ylabel("quality drop")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _runtime_plot(runtimes: pd.DataFrame, output_path: Path) -> Path:
    data = runtimes.copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    data.boxplot(column="runtime_median", by="method", ax=ax, rot=45)
    ax.set_title("Runtime by Method")
    ax.figure.suptitle("")
    ax.set_xlabel("method")
    ax.set_ylabel("median runtime, seconds")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _ablation_plot(correlations: pd.DataFrame, output_path: Path) -> Path:
    data = correlations.dropna(subset=["spearman"])
    fig, ax = plt.subplots(figsize=(9, 4))
    if not data.empty:
        data.boxplot(column="spearman", by="ablation_type", ax=ax, rot=30)
    ax.set_title("Ablation Spearman Correlations")
    ax.figure.suptitle("")
    ax.set_xlabel("ablation")
    ax.set_ylabel("Spearman rho")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _retraining_regret_plot(summary: pd.DataFrame, output_path: Path) -> Path:
    data = summary.copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    if not data.empty:
        data.boxplot(column="cumulative_regret", by="policy", ax=ax, rot=45)
    ax.set_title("Retraining Policy Regret")
    ax.figure.suptitle("")
    ax.set_xlabel("policy")
    ax.set_ylabel("cumulative regret")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path
