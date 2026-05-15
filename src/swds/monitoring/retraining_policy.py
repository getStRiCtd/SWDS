from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swds.drift.utils import matrix_vstack
from swds.models.evaluate import evaluate_model, higher_is_better as metric_higher_is_better
from swds.models.train import train_model


def periodic_triggers(n_windows: int, *, period: int) -> np.ndarray:
    if period <= 0:
        raise ValueError("period must be positive")
    triggers = np.zeros(n_windows, dtype=bool)
    triggers[period - 1 :: period] = True
    return triggers


def score_threshold_triggers(scores, *, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) >= threshold


def oracle_triggers(quality_drop, *, min_drop: float) -> np.ndarray:
    return np.asarray(quality_drop, dtype=float) >= min_drop


def cumulative_regret(policy_quality, oracle_quality, *, higher_is_better: bool = True) -> float:
    policy = np.asarray(policy_quality, dtype=float)
    oracle = np.asarray(oracle_quality, dtype=float)
    mask = np.isfinite(policy) & np.isfinite(oracle)
    if higher_is_better:
        regret = oracle[mask] - policy[mask]
    else:
        regret = policy[mask] - oracle[mask]
    return float(np.sum(np.maximum(regret, 0.0)))


@dataclass(frozen=True)
class RetrainingSimulationConfig:
    periods: tuple[int, ...] = (4,)
    history_modes: tuple[str, ...] = ("all", "rolling")
    rolling_history_windows: int = 4
    oracle_min_quality_drop: float = 0.02


@dataclass(frozen=True)
class PolicySimulationResult:
    windows: object
    summary: object


def build_policy_triggers(
    window_scores,
    *,
    methods: tuple[str, ...],
    periods: tuple[int, ...],
    oracle_min_quality_drop: float,
) -> dict[str, np.ndarray]:
    by_window = (
        window_scores.drop_duplicates("window_index")
        .sort_values("window_index")
        .reset_index(drop=True)
    )
    n_windows = len(by_window)
    triggers: dict[str, np.ndarray] = {"no_retraining": np.zeros(n_windows, dtype=bool)}

    for period in periods:
        triggers[f"periodic_p{period}"] = periodic_triggers(n_windows, period=period)

    for method in methods:
        method_rows = (
            window_scores.loc[window_scores["method"] == method]
            .sort_values("window_index")
            .reset_index(drop=True)
        )
        if len(method_rows) == n_windows and "triggered" in method_rows:
            triggers[f"{method}_trigger"] = method_rows["triggered"].to_numpy(dtype=bool)

    triggers["oracle_drop"] = oracle_triggers(
        by_window["quality_drop"].to_numpy(dtype=float),
        min_drop=oracle_min_quality_drop,
    )
    triggers["oracle_every_window"] = np.ones(n_windows, dtype=bool)
    return triggers


def simulate_retraining_policies(
    *,
    X_initial,
    y_initial,
    X_windows: list,
    y_windows: list[np.ndarray],
    task_type,
    model_name: str,
    primary_metric: str,
    policy_triggers: dict[str, np.ndarray],
    history_modes: tuple[str, ...] = ("all", "rolling"),
    rolling_history_windows: int = 4,
    seed: int = 42,
    base_quality_drop: np.ndarray | None = None,
    false_retrain_min_quality_drop: float = 0.02,
    oracle_policy: str = "oracle_every_window",
):
    import pandas as pd

    rows = []
    n_windows = len(X_windows)
    higher = metric_higher_is_better(primary_metric)
    base_quality_drop = (
        np.asarray(base_quality_drop, dtype=float)
        if base_quality_drop is not None
        else np.full(n_windows, np.nan)
    )

    for history_mode in history_modes:
        if history_mode not in {"all", "rolling"}:
            raise ValueError("history_mode must be 'all' or 'rolling'")
        for policy_name, triggers in policy_triggers.items():
            if len(triggers) != n_windows:
                raise ValueError(f"trigger length mismatch for policy {policy_name!r}")
            model = train_model(
                X_initial,
                y_initial,
                task_type=task_type,
                model_name=model_name,
                seed=seed,
            )
            seen_X: list = []
            seen_y: list[np.ndarray] = []
            retrains = 0

            for window_index, (X_cur, y_cur) in enumerate(zip(X_windows, y_windows, strict=True)):
                metrics = evaluate_model(model, X_cur, y_cur, task_type=task_type)
                quality = float(metrics.get(primary_metric, np.nan))
                trigger = bool(triggers[window_index])
                rows.append(
                    {
                        "policy": policy_name,
                        "history_mode": history_mode,
                        "window_index": window_index,
                        "quality": quality,
                        "triggered_after_window": trigger,
                        "n_retrains_so_far": retrains,
                        "primary_metric": primary_metric,
                        **{f"metric_{key}": value for key, value in metrics.items()},
                    }
                )

                seen_X.append(X_cur)
                seen_y.append(np.asarray(y_cur))
                if trigger:
                    X_fit, y_fit = _training_history(
                        X_initial=X_initial,
                        y_initial=y_initial,
                        seen_X=seen_X,
                        seen_y=seen_y,
                        history_mode=history_mode,
                        rolling_history_windows=rolling_history_windows,
                    )
                    model = train_model(
                        X_fit,
                        y_fit,
                        task_type=task_type,
                        model_name=model_name,
                        seed=seed + retrains + 1,
                    )
                    retrains += 1

    window_table = pd.DataFrame(rows)
    oracle_key = f"{oracle_policy}:all"
    oracle_series = (
        window_table.loc[
            (window_table["policy"] == oracle_policy) & (window_table["history_mode"] == "all"),
            ["window_index", "quality"],
        ]
        .set_index("window_index")["quality"]
        .sort_index()
    )
    if len(oracle_series) != n_windows:
        oracle_series = window_table.groupby("window_index")["quality"].max() if higher else window_table.groupby("window_index")["quality"].min()
        oracle_key = "best_observed"

    no_retraining_quality = {}
    for history_mode, group in window_table.loc[window_table["policy"] == "no_retraining"].groupby("history_mode"):
        no_retraining_quality[history_mode] = (
            group.sort_values("window_index")["quality"].to_numpy(dtype=float)
        )

    summary_rows = []
    for (policy, history_mode), group in window_table.groupby(["policy", "history_mode"], sort=True):
        quality = group.sort_values("window_index")["quality"].to_numpy(dtype=float)
        triggers = group.sort_values("window_index")["triggered_after_window"].to_numpy(dtype=bool)
        oracle_quality = oracle_series.reindex(range(n_windows)).to_numpy(dtype=float)
        n_retrains = int(triggers.sum())
        false_retrains = int(np.sum(triggers & (base_quality_drop < false_retrain_min_quality_drop)))
        baseline_quality = no_retraining_quality.get(history_mode)
        if baseline_quality is not None and len(baseline_quality) == len(quality):
            gain = quality - baseline_quality if higher else baseline_quality - quality
            mean_gain = float(np.nanmean(gain))
        else:
            mean_gain = float("nan")
        summary_rows.append(
            {
                "policy": policy,
                "history_mode": history_mode,
                "oracle_reference": oracle_key,
                "mean_quality": float(np.nanmean(quality)),
                "worst_window_quality": float(np.nanmin(quality) if higher else np.nanmax(quality)),
                "n_retrains": n_retrains,
                "mean_quality_gain_vs_no_retraining": mean_gain,
                "quality_gain_per_retrain": float(mean_gain / max(n_retrains, 1)),
                "cumulative_regret": cumulative_regret(quality, oracle_quality, higher_is_better=higher),
                "false_retrains": false_retrains,
                "false_retrain_share": float(false_retrains / max(n_retrains, 1)),
            }
        )

    return PolicySimulationResult(
        windows=window_table,
        summary=pd.DataFrame(summary_rows).sort_values("cumulative_regret"),
    )


def _training_history(
    *,
    X_initial,
    y_initial,
    seen_X: list,
    seen_y: list[np.ndarray],
    history_mode: str,
    rolling_history_windows: int,
):
    if history_mode == "all":
        return matrix_vstack([X_initial, *seen_X]), np.concatenate([np.asarray(y_initial), *seen_y])
    if rolling_history_windows <= 0:
        raise ValueError("rolling_history_windows must be positive")
    selected_X = seen_X[-rolling_history_windows:]
    selected_y = seen_y[-rolling_history_windows:]
    if not selected_X:
        return X_initial, np.asarray(y_initial)
    return matrix_vstack(selected_X), np.concatenate(selected_y)
