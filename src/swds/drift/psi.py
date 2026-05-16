from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swds.drift.utils import binary_column_means_if_binary, dense_column, finite_1d


@dataclass(frozen=True)
class PreparedPSIFeature:
    feature_id: int
    edges: np.ndarray | None
    reference_pct: np.ndarray | None
    reference_values: np.ndarray | None
    reference_counts: np.ndarray | None


@dataclass(frozen=True)
class PreparedPSIReference:
    features: tuple[PreparedPSIFeature, ...]
    n_bins: int
    eps: float


def prepare_psi_reference(
    X_ref,
    *,
    n_bins: int = 10,
    eps: float = 1e-6,
    max_features: int | None = 2048,
    seed: int = 42,
) -> PreparedPSIReference:
    feature_ids = _selected_feature_ids(X_ref.shape[1], max_features=max_features, seed=seed)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    features: list[PreparedPSIFeature] = []

    for idx in feature_ids:
        ref = finite_1d(dense_column(X_ref, int(idx)))
        if len(ref) == 0:
            continue

        edges = np.unique(np.quantile(ref, qs))
        if len(edges) < 3:
            values, counts = np.unique(ref, return_counts=True)
            features.append(
                PreparedPSIFeature(
                    feature_id=int(idx),
                    edges=None,
                    reference_pct=None,
                    reference_values=values.astype(np.float64, copy=False),
                    reference_counts=counts.astype(np.float64, copy=False),
                )
            )
            continue

        edges = edges.astype(np.float64, copy=True)
        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_counts, _ = np.histogram(ref, bins=edges)
        ref_pct = _clipped_pct(ref_counts, eps=eps)
        features.append(
            PreparedPSIFeature(
                feature_id=int(idx),
                edges=edges,
                reference_pct=ref_pct,
                reference_values=None,
                reference_counts=None,
            )
        )

    return PreparedPSIReference(features=tuple(features), n_bins=n_bins, eps=eps)


def psi_statistics_prepared(prepared: PreparedPSIReference, X_cur) -> np.ndarray:
    values: list[float | None] = [None] * len(prepared.features)
    binary_positions = [
        pos
        for pos, feature in enumerate(prepared.features)
        if _is_binary_fallback_feature(feature)
    ]
    binary_feature_ids = [prepared.features[pos].feature_id for pos in binary_positions]
    binary_cur_means = binary_column_means_if_binary(X_cur, binary_feature_ids)
    if binary_cur_means is not None:
        for pos, cur_mean in zip(binary_positions, binary_cur_means, strict=True):
            ref_mean = _binary_reference_mean(prepared.features[pos])
            values[pos] = _binary_psi(ref_mean, float(cur_mean), eps=prepared.eps)

    for pos, feature in enumerate(prepared.features):
        if values[pos] is not None:
            continue
        cur = finite_1d(dense_column(X_cur, feature.feature_id))
        if len(cur) == 0:
            continue
        if feature.edges is None:
            edges, ref_pct = _fallback_edges_and_reference_pct(feature, cur, n_bins=prepared.n_bins, eps=prepared.eps)
            if edges is None:
                values[pos] = 0.0
                continue
        else:
            edges = feature.edges
            ref_pct = feature.reference_pct
            if ref_pct is None:
                continue

        cur_counts, _ = np.histogram(cur, bins=edges)
        cur_pct = _clipped_pct(cur_counts, eps=prepared.eps)
        values[pos] = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return np.asarray([value for value in values if value is not None], dtype=np.float64)


def psi_statistics(
    X_ref,
    X_cur,
    *,
    n_bins: int = 10,
    eps: float = 1e-6,
    max_features: int | None = 2048,
    seed: int = 42,
) -> np.ndarray:
    prepared = prepare_psi_reference(X_ref, n_bins=n_bins, eps=eps, max_features=max_features, seed=seed)
    return psi_statistics_prepared(prepared, X_cur)


def mean_psi_score(X_ref, X_cur, *, n_bins: int = 10, max_features: int | None = 2048, seed: int = 42) -> float:
    stats = psi_statistics(X_ref, X_cur, n_bins=n_bins, max_features=max_features, seed=seed)
    return float(np.nanmean(stats)) if len(stats) else float("nan")


def max_psi_score(X_ref, X_cur, *, n_bins: int = 10, max_features: int | None = 2048, seed: int = 42) -> float:
    stats = psi_statistics(X_ref, X_cur, n_bins=n_bins, max_features=max_features, seed=seed)
    return float(np.nanmax(stats)) if len(stats) else float("nan")


def _selected_feature_ids(n_features: int, *, max_features: int | None, seed: int) -> np.ndarray:
    feature_ids = np.arange(n_features)
    if max_features is not None and n_features > max_features:
        rng = np.random.default_rng(seed)
        feature_ids = np.sort(rng.choice(n_features, size=max_features, replace=False))
    return feature_ids


def _clipped_pct(counts: np.ndarray, *, eps: float) -> np.ndarray:
    pct = np.asarray(counts, dtype=np.float64) / max(float(np.sum(counts)), 1.0)
    return np.clip(pct, eps, None)


def _fallback_edges_and_reference_pct(
    feature: PreparedPSIFeature,
    cur: np.ndarray,
    *,
    n_bins: int,
    eps: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    if feature.reference_values is None or feature.reference_counts is None:
        return None, np.asarray([], dtype=np.float64)
    lo = min(float(np.min(feature.reference_values)), float(np.min(cur)))
    hi = max(float(np.max(feature.reference_values)), float(np.max(cur)))
    if lo == hi:
        return None, np.asarray([], dtype=np.float64)
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(
        feature.reference_values,
        bins=edges,
        weights=feature.reference_counts,
    )
    return edges, _clipped_pct(ref_counts, eps=eps)


def _is_binary_fallback_feature(feature: PreparedPSIFeature) -> bool:
    if feature.edges is not None or feature.reference_values is None:
        return False
    return bool(np.all((feature.reference_values == 0.0) | (feature.reference_values == 1.0)))


def _binary_reference_mean(feature: PreparedPSIFeature) -> float:
    if feature.reference_values is None or feature.reference_counts is None:
        return float("nan")
    total = max(float(np.sum(feature.reference_counts)), 1.0)
    return float(np.sum(feature.reference_values * feature.reference_counts) / total)


def _binary_psi(ref_mean: float, cur_mean: float, *, eps: float) -> float:
    ref_pct = np.clip(np.asarray([1.0 - ref_mean, ref_mean], dtype=np.float64), eps, None)
    cur_pct = np.clip(np.asarray([1.0 - cur_mean, cur_mean], dtype=np.float64), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
