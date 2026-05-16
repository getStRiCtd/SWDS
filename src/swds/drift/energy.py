from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import pairwise_distances

from swds.drift.utils import subsample_rows, to_dense


@dataclass(frozen=True)
class PreparedEnergyReference:
    x_ref: np.ndarray
    d_xx_mean: float


def prepare_energy_reference(X_ref, *, max_samples: int = 1000, seed: int = 42) -> PreparedEnergyReference:
    X_ref_sub, _ = subsample_rows(X_ref, max_samples, seed=seed)
    x = to_dense(X_ref_sub)
    return PreparedEnergyReference(
        x_ref=x,
        d_xx_mean=float(pairwise_distances(x, x).mean()),
    )


def energy_distance_score_prepared(
    prepared: PreparedEnergyReference,
    X_cur,
    *,
    max_samples: int = 1000,
    seed: int = 42,
) -> float:
    X_cur_sub, _ = subsample_rows(X_cur, max_samples, seed=seed + 1)
    y = to_dense(X_cur_sub)

    d_xy = pairwise_distances(prepared.x_ref, y).mean()
    d_yy = pairwise_distances(y, y).mean()
    value = 2.0 * d_xy - prepared.d_xx_mean - d_yy
    return float(np.sqrt(max(value, 0.0)))


def energy_distance_score(
    X_ref,
    X_cur,
    *,
    max_samples: int = 1000,
    seed: int = 42,
) -> float:
    prepared = prepare_energy_reference(X_ref, max_samples=max_samples, seed=seed)
    return energy_distance_score_prepared(prepared, X_cur, max_samples=max_samples, seed=seed)
