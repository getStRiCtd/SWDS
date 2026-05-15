from __future__ import annotations

import numpy as np

from swds.drift.utils import dense_column, finite_1d


def psi_statistics(
    X_ref,
    X_cur,
    *,
    n_bins: int = 10,
    eps: float = 1e-6,
    max_features: int | None = 2048,
    seed: int = 42,
) -> np.ndarray:
    n_features = X_ref.shape[1]
    feature_ids = np.arange(n_features)
    if max_features is not None and n_features > max_features:
        rng = np.random.default_rng(seed)
        feature_ids = np.sort(rng.choice(n_features, size=max_features, replace=False))

    values = []
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in feature_ids:
        ref = finite_1d(dense_column(X_ref, int(idx)))
        cur = finite_1d(dense_column(X_cur, int(idx)))
        if len(ref) == 0 or len(cur) == 0:
            continue

        edges = np.unique(np.quantile(ref, qs))
        if len(edges) < 3:
            lo = min(np.min(ref), np.min(cur))
            hi = max(np.max(ref), np.max(cur))
            if lo == hi:
                values.append(0.0)
                continue
            edges = np.linspace(lo, hi, n_bins + 1)

        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
        ref_pct = ref_counts / max(ref_counts.sum(), 1)
        cur_pct = cur_counts / max(cur_counts.sum(), 1)
        ref_pct = np.clip(ref_pct, eps, None)
        cur_pct = np.clip(cur_pct, eps, None)
        values.append(float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))))
    return np.asarray(values, dtype=np.float64)


def mean_psi_score(X_ref, X_cur, *, n_bins: int = 10, max_features: int | None = 2048, seed: int = 42) -> float:
    stats = psi_statistics(X_ref, X_cur, n_bins=n_bins, max_features=max_features, seed=seed)
    return float(np.nanmean(stats)) if len(stats) else float("nan")


def max_psi_score(X_ref, X_cur, *, n_bins: int = 10, max_features: int | None = 2048, seed: int = 42) -> float:
    stats = psi_statistics(X_ref, X_cur, n_bins=n_bins, max_features=max_features, seed=seed)
    return float(np.nanmax(stats)) if len(stats) else float("nan")
