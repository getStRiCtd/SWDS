from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from swds.drift.utils import subsample_rows, to_dense


def energy_distance_score(
    X_ref,
    X_cur,
    *,
    max_samples: int = 1000,
    seed: int = 42,
) -> float:
    X_ref_sub, _ = subsample_rows(X_ref, max_samples, seed=seed)
    X_cur_sub, _ = subsample_rows(X_cur, max_samples, seed=seed + 1)
    x = to_dense(X_ref_sub)
    y = to_dense(X_cur_sub)

    d_xy = pairwise_distances(x, y).mean()
    d_xx = pairwise_distances(x, x).mean()
    d_yy = pairwise_distances(y, y).mean()
    value = 2.0 * d_xy - d_xx - d_yy
    return float(np.sqrt(max(value, 0.0)))
