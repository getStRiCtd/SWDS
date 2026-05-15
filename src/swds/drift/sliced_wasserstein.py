from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swds.drift.utils import n_rows, project, take_rows


@dataclass(frozen=True)
class SlicedWassersteinResult:
    score: float
    per_projection: np.ndarray


def random_directions(n_features: int, n_projections: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_features, n_projections))
    norms = np.linalg.norm(directions, axis=0, keepdims=True)
    return directions / np.maximum(norms, 1e-12)


def sliced_wasserstein_score(
    X_ref,
    X_cur,
    *,
    n_projections: int = 128,
    n_quantiles: int = 512,
    seed: int = 42,
    mode: str = "quantile",
    subsample_size: int | None = None,
    n_subsample_seeds: int = 5,
    return_details: bool = False,
) -> float | SlicedWassersteinResult:
    if X_ref.shape[1] != X_cur.shape[1]:
        raise ValueError(f"feature dimension mismatch: {X_ref.shape[1]} != {X_cur.shape[1]}")
    if n_rows(X_ref) < 2 or n_rows(X_cur) < 2:
        raise ValueError("SWDS requires at least two rows in each sample")

    directions = random_directions(X_ref.shape[1], n_projections, seed=seed)
    if mode == "quantile":
        result = _quantile_grid_score(
            X_ref,
            X_cur,
            directions=directions,
            n_quantiles=n_quantiles,
        )
    elif mode == "subsample":
        result = _equal_size_subsample_score(
            X_ref,
            X_cur,
            directions=directions,
            seed=seed,
            subsample_size=subsample_size,
            n_subsample_seeds=n_subsample_seeds,
        )
    else:
        raise ValueError("mode must be either 'quantile' or 'subsample'")

    if return_details:
        return result
    return result.score


def _quantile_grid_score(X_ref, X_cur, *, directions: np.ndarray, n_quantiles: int) -> SlicedWassersteinResult:
    z_ref = project(X_ref, directions)
    z_cur = project(X_cur, directions)
    qs = np.linspace(0.0, 1.0, num=n_quantiles)
    q_ref = np.quantile(z_ref, qs, axis=0)
    q_cur = np.quantile(z_cur, qs, axis=0)
    per_projection = np.mean((q_ref - q_cur) ** 2, axis=0)
    return SlicedWassersteinResult(
        score=float(np.sqrt(np.mean(per_projection))),
        per_projection=per_projection,
    )


def _equal_size_subsample_score(
    X_ref,
    X_cur,
    *,
    directions: np.ndarray,
    seed: int,
    subsample_size: int | None,
    n_subsample_seeds: int,
) -> SlicedWassersteinResult:
    rng = np.random.default_rng(seed)
    size = min(n_rows(X_ref), n_rows(X_cur))
    if subsample_size is not None:
        size = min(size, subsample_size)
    if size < 2:
        raise ValueError("equal-size subsampling requires sample size >= 2")

    per_seed = []
    for _ in range(n_subsample_seeds):
        ref_idx = np.sort(rng.choice(n_rows(X_ref), size=size, replace=False))
        cur_idx = np.sort(rng.choice(n_rows(X_cur), size=size, replace=False))
        z_ref = np.sort(project(take_rows(X_ref, ref_idx), directions), axis=0)
        z_cur = np.sort(project(take_rows(X_cur, cur_idx), directions), axis=0)
        per_seed.append(np.mean((z_ref - z_cur) ** 2, axis=0))

    per_projection = np.mean(np.vstack(per_seed), axis=0)
    return SlicedWassersteinResult(
        score=float(np.sqrt(np.mean(per_projection))),
        per_projection=per_projection,
    )
