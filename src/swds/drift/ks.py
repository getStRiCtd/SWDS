from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swds.drift.utils import binary_column_means_if_binary, dense_column, finite_1d


@dataclass(frozen=True)
class PreparedKSReference:
    feature_ids: tuple[int, ...]
    sorted_reference: tuple[np.ndarray | None, ...]
    binary_reference_mean: tuple[float, ...]
    is_binary_reference: tuple[bool, ...]


def prepare_ks_reference(X_ref, *, max_features: int | None = 2048, seed: int = 42) -> PreparedKSReference:
    feature_ids = _selected_feature_ids(X_ref.shape[1], max_features=max_features, seed=seed)
    kept_ids: list[int] = []
    sorted_reference: list[np.ndarray | None] = []
    binary_reference_mean: list[float] = []
    is_binary_reference: list[bool] = []

    for idx in feature_ids:
        ref = finite_1d(dense_column(X_ref, int(idx)))
        if len(ref) == 0:
            continue
        ref_sorted = np.sort(ref)
        is_binary = _is_binary_values(ref_sorted)
        kept_ids.append(int(idx))
        sorted_reference.append(None if is_binary else ref_sorted)
        binary_reference_mean.append(float(np.mean(ref_sorted)) if is_binary else float("nan"))
        is_binary_reference.append(is_binary)

    return PreparedKSReference(
        feature_ids=tuple(kept_ids),
        sorted_reference=tuple(sorted_reference),
        binary_reference_mean=tuple(binary_reference_mean),
        is_binary_reference=tuple(is_binary_reference),
    )


def ks_statistics_prepared(prepared: PreparedKSReference, X_cur) -> np.ndarray:
    stats: list[float | None] = [None] * len(prepared.feature_ids)
    binary_positions = [pos for pos, is_binary in enumerate(prepared.is_binary_reference) if is_binary]
    binary_feature_ids = [prepared.feature_ids[pos] for pos in binary_positions]
    binary_cur_means = binary_column_means_if_binary(X_cur, binary_feature_ids)
    if binary_cur_means is not None:
        for pos, cur_mean in zip(binary_positions, binary_cur_means, strict=True):
            stats[pos] = abs(prepared.binary_reference_mean[pos] - float(cur_mean))

    for pos, (idx, ref_sorted, ref_mean, ref_is_binary) in enumerate(zip(
        prepared.feature_ids,
        prepared.sorted_reference,
        prepared.binary_reference_mean,
        prepared.is_binary_reference,
        strict=True,
    )):
        if stats[pos] is not None:
            continue
        cur = finite_1d(dense_column(X_cur, idx))
        if len(cur) == 0:
            continue
        if ref_is_binary and _is_binary_values(cur):
            stats[pos] = abs(ref_mean - float(np.mean(cur)))
            continue
        if ref_is_binary:
            stats[pos] = _ks_statistic_binary_reference(ref_mean, np.sort(cur))
            continue
        if ref_sorted is None:
            continue
        stats[pos] = _ks_statistic_from_sorted(ref_sorted, np.sort(cur))
    return np.asarray([value for value in stats if value is not None], dtype=np.float64)


def ks_statistics(X_ref, X_cur, *, max_features: int | None = 2048, seed: int = 42) -> np.ndarray:
    prepared = prepare_ks_reference(X_ref, max_features=max_features, seed=seed)
    return ks_statistics_prepared(prepared, X_cur)


def mean_ks_score(X_ref, X_cur, *, max_features: int | None = 2048, seed: int = 42) -> float:
    stats = ks_statistics(X_ref, X_cur, max_features=max_features, seed=seed)
    return float(np.nanmean(stats)) if len(stats) else float("nan")


def max_ks_score(X_ref, X_cur, *, max_features: int | None = 2048, seed: int = 42) -> float:
    stats = ks_statistics(X_ref, X_cur, max_features=max_features, seed=seed)
    return float(np.nanmax(stats)) if len(stats) else float("nan")


def _selected_feature_ids(n_features: int, *, max_features: int | None, seed: int) -> np.ndarray:
    feature_ids = np.arange(n_features)
    if max_features is not None and n_features > max_features:
        rng = np.random.default_rng(seed)
        feature_ids = np.sort(rng.choice(n_features, size=max_features, replace=False))
    return feature_ids


def _is_binary_values(values: np.ndarray) -> bool:
    if len(values) == 0:
        return False
    return bool(np.all((values == 0.0) | (values == 1.0)))


def _ks_statistic_from_sorted(ref_sorted: np.ndarray, cur_sorted: np.ndarray) -> float:
    if len(ref_sorted) == 0 or len(cur_sorted) == 0:
        return float("nan")
    support = np.concatenate([ref_sorted, cur_sorted])
    support = np.unique(support)
    ref_cdf = np.searchsorted(ref_sorted, support, side="right") / len(ref_sorted)
    cur_cdf = np.searchsorted(cur_sorted, support, side="right") / len(cur_sorted)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def _ks_statistic_binary_reference(ref_mean: float, cur_sorted: np.ndarray) -> float:
    if len(cur_sorted) == 0:
        return float("nan")
    support = np.concatenate([cur_sorted, np.asarray([0.0, 1.0])])
    support = np.unique(support)
    ref_cdf = np.where(support < 0.0, 0.0, np.where(support < 1.0, 1.0 - ref_mean, 1.0))
    cur_cdf = np.searchsorted(cur_sorted, support, side="right") / len(cur_sorted)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))
