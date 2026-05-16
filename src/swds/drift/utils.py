from __future__ import annotations

import numpy as np
from scipy import sparse


def n_rows(X) -> int:
    return int(X.shape[0])


def to_dense(X, *, dtype=np.float64) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray().astype(dtype, copy=False)
    return np.asarray(X, dtype=dtype)


def matrix_vstack(parts):
    if any(sparse.issparse(part) for part in parts):
        return sparse.vstack(parts, format="csr")
    return np.vstack(parts)


def take_rows(X, idx: np.ndarray):
    return X[idx]


def subsample_rows(X, max_samples: int | None, *, seed: int) -> tuple[object, np.ndarray]:
    n = n_rows(X)
    if max_samples is None or n <= max_samples:
        idx = np.arange(n)
        return X, idx
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_samples, replace=False))
    return take_rows(X, idx), idx


def project(X, directions: np.ndarray) -> np.ndarray:
    projected = X @ directions
    if sparse.issparse(projected):
        projected = projected.toarray()
    return np.asarray(projected, dtype=np.float64)


def dense_column(X, column_idx: int) -> np.ndarray:
    if sparse.issparse(X):
        return X[:, column_idx].toarray().ravel()
    return np.asarray(X[:, column_idx]).ravel()


def binary_column_means_if_binary(X, column_ids: list[int] | tuple[int, ...]) -> np.ndarray | None:
    if not column_ids or X.shape[0] == 0:
        return None
    if sparse.issparse(X):
        subset = X[:, list(column_ids)]
        data = subset.data
        if len(data) and not np.all((data == 0.0) | (data == 1.0)):
            return None
        return np.asarray(subset.mean(axis=0)).ravel().astype(np.float64, copy=False)

    subset = np.asarray(X[:, list(column_ids)], dtype=np.float64)
    if subset.ndim == 1:
        subset = subset.reshape(-1, 1)
    if not np.all(np.isfinite(subset)):
        return None
    if not np.all((subset == 0.0) | (subset == 1.0)):
        return None
    return np.mean(subset, axis=0)


def finite_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]
