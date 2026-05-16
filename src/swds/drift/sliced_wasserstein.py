from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
from scipy import sparse

from swds.drift.utils import n_rows, project, take_rows


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlicedWassersteinResult:
    score: float
    per_projection: np.ndarray


@dataclass(frozen=True)
class SlicedWassersteinReference:
    directions: Any
    reference_quantiles: Any
    quantile_grid: Any
    backend: str
    device: str | None


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
    backend: str = "auto",
    device: str | None = None,
    return_details: bool = False,
) -> float | SlicedWassersteinResult:
    if X_ref.shape[1] != X_cur.shape[1]:
        raise ValueError(f"feature dimension mismatch: {X_ref.shape[1]} != {X_cur.shape[1]}")
    if n_rows(X_ref) < 2 or n_rows(X_cur) < 2:
        raise ValueError("SWDS requires at least two rows in each sample")

    directions = random_directions(X_ref.shape[1], n_projections, seed=seed)
    if mode == "quantile":
        prepared = prepare_sliced_wasserstein_reference(
            X_ref,
            directions=directions,
            n_quantiles=n_quantiles,
            backend=backend,
            device=device,
        )
        result = score_sliced_wasserstein_prepared(
            prepared,
            X_cur,
            return_details=True,
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


def prepare_sliced_wasserstein_reference(
    X_ref,
    *,
    directions: np.ndarray,
    n_quantiles: int,
    backend: str = "auto",
    device: str | None = None,
) -> SlicedWassersteinReference:
    resolved_backend, resolved_device = _resolve_backend(backend, device, X_ref)
    LOGGER.debug(
        "preparing SWDS reference backend=%s device=%s rows=%d features=%d projections=%d quantiles=%d",
        resolved_backend,
        resolved_device,
        n_rows(X_ref),
        X_ref.shape[1],
        directions.shape[1],
        n_quantiles,
    )
    if resolved_backend == "torch":
        return _prepare_torch_reference(
            X_ref,
            directions=directions,
            n_quantiles=n_quantiles,
            device=resolved_device,
        )
    qs = np.linspace(0.0, 1.0, num=n_quantiles)
    z_ref = project(X_ref, directions)
    q_ref = np.quantile(z_ref, qs, axis=0)
    LOGGER.info(
        "SWDS reference prepared on numpy rows=%d projections=%d quantiles=%d",
        n_rows(X_ref),
        directions.shape[1],
        n_quantiles,
    )
    return SlicedWassersteinReference(
        directions=directions,
        reference_quantiles=q_ref,
        quantile_grid=qs,
        backend="numpy",
        device=None,
    )


def score_sliced_wasserstein_prepared(
    prepared: SlicedWassersteinReference,
    X_cur,
    *,
    return_details: bool = False,
) -> float | SlicedWassersteinResult:
    if prepared.backend == "torch":
        result = _score_torch_prepared(prepared, X_cur)
    else:
        result = _score_numpy_prepared(prepared, X_cur)
    if return_details:
        return result
    return result.score


def _score_numpy_prepared(prepared: SlicedWassersteinReference, X_cur) -> SlicedWassersteinResult:
    z_cur = project(X_cur, prepared.directions)
    q_cur = np.quantile(z_cur, prepared.quantile_grid, axis=0)
    q_ref = np.asarray(prepared.reference_quantiles)
    per_projection = np.mean((q_ref - q_cur) ** 2, axis=0)
    return SlicedWassersteinResult(
        score=float(np.sqrt(np.mean(per_projection))),
        per_projection=per_projection,
    )


def _resolve_backend(backend: str, device: str | None, X) -> tuple[str, str | None]:
    backend = backend.lower()
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be one of: auto, numpy, torch")
    if backend == "numpy":
        return "numpy", None
    if backend == "auto" and sparse.issparse(X):
        return "numpy", None
    try:
        import torch
    except ImportError:
        if backend == "torch":
            raise
        return "numpy", None
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if backend == "auto" and resolved_device == "cpu":
        return "numpy", None
    if str(resolved_device).startswith("cuda") and not torch.cuda.is_available():
        if backend == "torch":
            raise RuntimeError("SWDS torch backend requested CUDA, but torch.cuda.is_available() is false")
        return "numpy", None
    return "torch", str(resolved_device)


def _prepare_torch_reference(
    X_ref,
    *,
    directions: np.ndarray,
    n_quantiles: int,
    device: str | None,
) -> SlicedWassersteinReference:
    import torch

    x_ref = _to_torch_tensor(X_ref, device=device)
    dirs = torch.as_tensor(directions, dtype=x_ref.dtype, device=x_ref.device)
    qs = torch.linspace(0.0, 1.0, steps=n_quantiles, dtype=x_ref.dtype, device=x_ref.device)
    q_ref = torch.quantile(x_ref.matmul(dirs), qs, dim=0)
    LOGGER.info(
        "SWDS reference prepared on torch device=%s rows=%d projections=%d quantiles=%d",
        x_ref.device,
        x_ref.shape[0],
        dirs.shape[1],
        n_quantiles,
    )
    return SlicedWassersteinReference(
        directions=dirs,
        reference_quantiles=q_ref,
        quantile_grid=qs,
        backend="torch",
        device=str(x_ref.device),
    )


def _score_torch_prepared(prepared: SlicedWassersteinReference, X_cur) -> SlicedWassersteinResult:
    import torch

    x_cur = _to_torch_tensor(X_cur, device=prepared.device)
    q_cur = torch.quantile(x_cur.matmul(prepared.directions), prepared.quantile_grid, dim=0)
    per_projection_t = torch.mean((prepared.reference_quantiles - q_cur) ** 2, dim=0)
    score = torch.sqrt(torch.mean(per_projection_t)).detach().cpu().item()
    per_projection = per_projection_t.detach().cpu().numpy()
    return SlicedWassersteinResult(score=float(score), per_projection=per_projection)


def _to_torch_tensor(X, *, device: str | None):
    import torch

    if sparse.issparse(X):
        array = X.toarray()
    else:
        array = np.asarray(X)
    return torch.as_tensor(array, dtype=torch.float32, device=device)


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
