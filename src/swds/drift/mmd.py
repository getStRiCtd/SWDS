from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

from swds.drift.utils import subsample_rows, to_dense


def mmd_rbf_score(
    X_ref,
    X_cur,
    *,
    max_samples: int = 1000,
    seed: int = 42,
    gamma: float | None = None,
) -> float:
    X_ref_sub, _ = subsample_rows(X_ref, max_samples, seed=seed)
    X_cur_sub, _ = subsample_rows(X_cur, max_samples, seed=seed + 1)
    x = to_dense(X_ref_sub)
    y = to_dense(X_cur_sub)

    if gamma is None:
        gamma = median_heuristic_gamma(x, y)

    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)
    mmd2 = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return float(np.sqrt(max(mmd2, 0.0)))


def median_heuristic_gamma(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-12) -> float:
    joined = np.vstack([x, y])
    if len(joined) > 2000:
        rng = np.random.default_rng(0)
        joined = joined[np.sort(rng.choice(len(joined), size=2000, replace=False))]
    distances = pairwise_distances(joined, metric="sqeuclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    upper = upper[np.isfinite(upper) & (upper > 0)]
    if len(upper) == 0:
        return 1.0
    median_sq_dist = float(np.median(upper))
    return 1.0 / max(2.0 * median_sq_dist, eps)
