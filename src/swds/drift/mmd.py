from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import pairwise_distances

from swds.drift.utils import subsample_rows, to_dense


@dataclass(frozen=True)
class PreparedMMDReference:
    x_ref: np.ndarray
    ref_sq_distances: np.ndarray
    ref_upper_sq_distances: np.ndarray


def prepare_mmd_reference(X_ref, *, max_samples: int = 1000, seed: int = 42) -> PreparedMMDReference:
    X_ref_sub, _ = subsample_rows(X_ref, max_samples, seed=seed)
    x = to_dense(X_ref_sub)
    ref_sq_distances = pairwise_distances(x, x, metric="sqeuclidean")
    ref_upper = ref_sq_distances[np.triu_indices_from(ref_sq_distances, k=1)]
    return PreparedMMDReference(
        x_ref=x,
        ref_sq_distances=ref_sq_distances,
        ref_upper_sq_distances=ref_upper,
    )


def mmd_rbf_score_prepared(
    prepared: PreparedMMDReference,
    X_cur,
    *,
    max_samples: int = 1000,
    seed: int = 42,
    gamma: float | None = None,
) -> float:
    X_cur_sub, _ = subsample_rows(X_cur, max_samples, seed=seed + 1)
    y = to_dense(X_cur_sub)
    d_yy = pairwise_distances(y, y, metric="sqeuclidean")
    d_xy = pairwise_distances(prepared.x_ref, y, metric="sqeuclidean")

    if gamma is None:
        if prepared.x_ref.shape[0] + y.shape[0] > 2000:
            gamma = median_heuristic_gamma(prepared.x_ref, y)
        else:
            gamma = median_heuristic_gamma_from_distances(
                prepared.ref_upper_sq_distances,
                d_yy,
                d_xy,
            )

    k_xx = np.exp(-gamma * prepared.ref_sq_distances).mean()
    k_yy = np.exp(-gamma * d_yy).mean()
    k_xy = np.exp(-gamma * d_xy).mean()
    mmd2 = float(k_xx + k_yy - 2.0 * k_xy)
    return float(np.sqrt(max(mmd2, 0.0)))


def mmd_rbf_score(
    X_ref,
    X_cur,
    *,
    max_samples: int = 1000,
    seed: int = 42,
    gamma: float | None = None,
) -> float:
    prepared = prepare_mmd_reference(X_ref, max_samples=max_samples, seed=seed)
    return mmd_rbf_score_prepared(prepared, X_cur, max_samples=max_samples, seed=seed, gamma=gamma)


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


def median_heuristic_gamma_from_distances(
    ref_upper_sq_distances: np.ndarray,
    cur_sq_distances: np.ndarray,
    cross_sq_distances: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    cur_upper = cur_sq_distances[np.triu_indices_from(cur_sq_distances, k=1)]
    upper = np.concatenate([ref_upper_sq_distances, cross_sq_distances.ravel(), cur_upper])
    upper = upper[np.isfinite(upper) & (upper > 0)]
    if len(upper) == 0:
        return 1.0
    median_sq_dist = float(np.median(upper))
    return 1.0 / max(2.0 * median_sq_dist, eps)
