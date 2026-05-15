from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from swds.drift.utils import dense_column, finite_1d


def ks_statistics(X_ref, X_cur, *, max_features: int | None = 2048, seed: int = 42) -> np.ndarray:
    n_features = X_ref.shape[1]
    feature_ids = np.arange(n_features)
    if max_features is not None and n_features > max_features:
        rng = np.random.default_rng(seed)
        feature_ids = np.sort(rng.choice(n_features, size=max_features, replace=False))

    stats = []
    for idx in feature_ids:
        ref = finite_1d(dense_column(X_ref, int(idx)))
        cur = finite_1d(dense_column(X_cur, int(idx)))
        if len(ref) == 0 or len(cur) == 0:
            continue
        stats.append(float(ks_2samp(ref, cur, alternative="two-sided", mode="auto").statistic))
    return np.asarray(stats, dtype=np.float64)


def mean_ks_score(X_ref, X_cur, *, max_features: int | None = 2048, seed: int = 42) -> float:
    stats = ks_statistics(X_ref, X_cur, max_features=max_features, seed=seed)
    return float(np.nanmean(stats)) if len(stats) else float("nan")


def max_ks_score(X_ref, X_cur, *, max_features: int | None = 2048, seed: int = 42) -> float:
    stats = ks_statistics(X_ref, X_cur, max_features=max_features, seed=seed)
    return float(np.nanmax(stats)) if len(stats) else float("nan")
