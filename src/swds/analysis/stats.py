from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, friedmanchisquare, kendalltau, spearmanr, wilcoxon


def drift_quality_correlations(window_scores: pd.DataFrame, *, bootstrap_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    rows = []
    for method, group in window_scores.groupby("method", sort=True):
        clean = group[["window_index", "score", "quality_drop"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3:
            spearman = kendall = float("nan")
            ci_low = ci_high = float("nan")
            lag1 = lag2 = float("nan")
            n = len(clean)
        else:
            spearman = _safe_spearman(clean["score"], clean["quality_drop"])
            kendall = float(kendalltau(clean["score"], clean["quality_drop"]).statistic)
            ci_low, ci_high = spearman_bootstrap_ci(
                clean["score"].to_numpy(dtype=float),
                clean["quality_drop"].to_numpy(dtype=float),
                n_bootstrap=bootstrap_samples,
                seed=seed,
            )
            lag1 = lagged_spearman(clean, lag=1)
            lag2 = lagged_spearman(clean, lag=2)
            n = len(clean)
        rows.append(
            {
                "method": method,
                "spearman": spearman,
                "spearman_ci_low": ci_low,
                "spearman_ci_high": ci_high,
                "kendall_tau": kendall,
                "spearman_lag1": lag1,
                "spearman_lag2": lag2,
                "n_windows": n,
            }
        )
    return pd.DataFrame(rows).sort_values("spearman", ascending=False, na_position="last")


def spearman_bootstrap_ci(x, y, *, n_bootstrap: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=len(x))
        stat = _safe_spearman(x[idx], y[idx])
        if np.isfinite(stat):
            values.append(float(stat))
    if not values:
        return float("nan"), float("nan")
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def lagged_spearman(clean: pd.DataFrame, *, lag: int) -> float:
    ordered = clean.sort_values("window_index").copy()
    ordered["future_quality_drop"] = ordered["quality_drop"].shift(-lag)
    pair = ordered[["score", "future_quality_drop"]].dropna()
    if len(pair) < 3:
        return float("nan")
    return _safe_spearman(pair["score"], pair["future_quality_drop"])


def _safe_spearman(x, y) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        stat = spearmanr(x, y).statistic
    return float(stat) if np.isfinite(stat) else float("nan")


def benjamini_hochberg(p_values, *, alpha: float = 0.05) -> pd.DataFrame:
    arr = np.asarray(p_values, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    n = len(arr)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        running = min(running, ranked[idx] * n / (idx + 1))
        adjusted[order[idx]] = running
    return pd.DataFrame(
        {
            "p_value": arr,
            "p_value_bh": np.clip(adjusted, 0.0, 1.0),
            "reject_bh": np.clip(adjusted, 0.0, 1.0) <= alpha,
        }
    )


def pairwise_wilcoxon_vs_reference(
    paired_scores: pd.DataFrame,
    *,
    reference_method: str = "swds",
    value_col: str = "spearman",
    pair_col: str = "dataset_model",
) -> pd.DataFrame:
    pivot = paired_scores.pivot(index=pair_col, columns="method", values=value_col)
    if reference_method not in pivot.columns:
        raise ValueError(f"reference method {reference_method!r} is absent")

    rows = []
    ref = pivot[reference_method]
    for method in pivot.columns:
        if method == reference_method:
            continue
        clean = pd.concat([ref, pivot[method]], axis=1).dropna()
        clean.columns = [reference_method, method]
        if len(clean) < 3:
            stat = p_value = float("nan")
        else:
            stat, p_value = wilcoxon(clean[reference_method], clean[method], zero_method="wilcox")
        rows.append({"method": method, "wilcoxon_stat": stat, "p_value": p_value, "n_pairs": len(clean)})
    out = pd.DataFrame(rows)
    if len(out):
        correction = benjamini_hochberg(out["p_value"].fillna(1.0).to_numpy())
        out["p_value_bh"] = correction["p_value_bh"]
        out["reject_bh"] = correction["reject_bh"]
    return out


def friedman_test_by_pair(
    paired_scores: pd.DataFrame,
    *,
    value_col: str = "spearman",
    pair_col: str = "dataset_model",
) -> pd.DataFrame:
    pivot = paired_scores.pivot(index=pair_col, columns="method", values=value_col).dropna()
    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return pd.DataFrame([{"friedman_stat": np.nan, "p_value": np.nan, "n_pairs": pivot.shape[0], "n_methods": pivot.shape[1]}])
    stat, p_value = friedmanchisquare(*[pivot[col].to_numpy(dtype=float) for col in pivot.columns])
    return pd.DataFrame(
        [
            {
                "friedman_stat": float(stat),
                "p_value": float(p_value),
                "n_pairs": int(pivot.shape[0]),
                "n_methods": int(pivot.shape[1]),
            }
        ]
    )
