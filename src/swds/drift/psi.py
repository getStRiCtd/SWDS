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


@dataclass(frozen=True)
class PSIFeatureDiagnostic:
    feature_id: int
    psi: float
    n_bins: int
    zero_expected_bins: int
    zero_actual_bins: int
    clipped_expected_bins: int
    clipped_actual_bins: int


@dataclass(frozen=True)
class PSIDiagnostics:
    n_features: int
    n_bins_total: int
    zero_expected_bins: int
    zero_actual_bins: int
    clipped_expected_bins: int
    clipped_actual_bins: int
    top_features: tuple[PSIFeatureDiagnostic, ...]


@dataclass(frozen=True)
class PSIStatisticsResult:
    values: np.ndarray
    diagnostics: PSIDiagnostics


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
                reference_counts=ref_counts.astype(np.float64, copy=False),
            )
        )

    return PreparedPSIReference(features=tuple(features), n_bins=n_bins, eps=eps)


def psi_statistics_prepared(prepared: PreparedPSIReference, X_cur) -> np.ndarray:
    return psi_statistics_with_diagnostics_prepared(prepared, X_cur).values


def psi_statistics_with_diagnostics_prepared(
    prepared: PreparedPSIReference,
    X_cur,
    *,
    top_k: int = 10,
) -> PSIStatisticsResult:
    values: list[float | None] = [None] * len(prepared.features)
    feature_diagnostics: list[PSIFeatureDiagnostic] = []
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
            ref_counts = np.asarray([1.0 - ref_mean, ref_mean], dtype=np.float64)
            cur_counts = np.asarray([1.0 - float(cur_mean), float(cur_mean)], dtype=np.float64)
            psi, diagnostic = _psi_from_counts(
                ref_counts,
                cur_counts,
                eps=prepared.eps,
                feature_id=prepared.features[pos].feature_id,
            )
            values[pos] = psi
            feature_diagnostics.append(diagnostic)

    for pos, feature in enumerate(prepared.features):
        if values[pos] is not None:
            continue
        cur = finite_1d(dense_column(X_cur, feature.feature_id))
        if len(cur) == 0:
            continue
        if feature.edges is None:
            edges, ref_counts = _fallback_edges_and_reference_counts(feature, cur, n_bins=prepared.n_bins)
            if edges is None:
                values[pos] = 0.0
                feature_diagnostics.append(
                    PSIFeatureDiagnostic(
                        feature_id=feature.feature_id,
                        psi=0.0,
                        n_bins=1,
                        zero_expected_bins=0,
                        zero_actual_bins=0,
                        clipped_expected_bins=0,
                        clipped_actual_bins=0,
                    )
                )
                continue
        else:
            edges = feature.edges
            ref_counts = feature.reference_counts
            if ref_counts is None:
                continue

        cur_counts, _ = np.histogram(cur, bins=edges)
        psi, diagnostic = _psi_from_counts(
            ref_counts,
            cur_counts,
            eps=prepared.eps,
            feature_id=feature.feature_id,
        )
        values[pos] = psi
        feature_diagnostics.append(diagnostic)
    stats = np.asarray([value for value in values if value is not None], dtype=np.float64)
    diagnostics = _summarize_feature_diagnostics(feature_diagnostics, top_k=top_k)
    return PSIStatisticsResult(values=stats, diagnostics=diagnostics)


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


def topk_psi_score(
    X_ref,
    X_cur,
    *,
    top_k: int = 5,
    n_bins: int = 10,
    max_features: int | None = 2048,
    seed: int = 42,
) -> float:
    stats = psi_statistics(X_ref, X_cur, n_bins=n_bins, max_features=max_features, seed=seed)
    return topk_psi_from_stats(stats, top_k=top_k)


def p95_psi_score(X_ref, X_cur, *, n_bins: int = 10, max_features: int | None = 2048, seed: int = 42) -> float:
    stats = psi_statistics(X_ref, X_cur, n_bins=n_bins, max_features=max_features, seed=seed)
    return float(np.nanpercentile(stats, 95)) if len(stats) else float("nan")


def topk_psi_from_stats(stats: np.ndarray, *, top_k: int = 5) -> float:
    finite = np.asarray(stats, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    k = min(max(int(top_k), 1), len(finite))
    return float(np.mean(np.partition(finite, -k)[-k:]))


def _selected_feature_ids(n_features: int, *, max_features: int | None, seed: int) -> np.ndarray:
    feature_ids = np.arange(n_features)
    if max_features is not None and n_features > max_features:
        rng = np.random.default_rng(seed)
        feature_ids = np.sort(rng.choice(n_features, size=max_features, replace=False))
    return feature_ids


def _clipped_pct(counts: np.ndarray, *, eps: float) -> np.ndarray:
    pct = np.asarray(counts, dtype=np.float64) / max(float(np.sum(counts)), 1.0)
    return np.clip(pct, eps, None)


def _fallback_edges_and_reference_counts(
    feature: PreparedPSIFeature,
    cur: np.ndarray,
    *,
    n_bins: int,
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
    return edges, ref_counts


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


def _psi_from_counts(
    ref_counts,
    cur_counts,
    *,
    eps: float,
    feature_id: int,
) -> tuple[float, PSIFeatureDiagnostic]:
    ref_counts = np.asarray(ref_counts, dtype=np.float64)
    cur_counts = np.asarray(cur_counts, dtype=np.float64)
    ref_raw = ref_counts / max(float(np.sum(ref_counts)), 1.0)
    cur_raw = cur_counts / max(float(np.sum(cur_counts)), 1.0)
    ref_pct = np.clip(ref_raw, eps, None)
    cur_pct = np.clip(cur_raw, eps, None)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    diagnostic = PSIFeatureDiagnostic(
        feature_id=feature_id,
        psi=psi,
        n_bins=len(ref_counts),
        zero_expected_bins=int(np.sum(ref_counts == 0.0)),
        zero_actual_bins=int(np.sum(cur_counts == 0.0)),
        clipped_expected_bins=int(np.sum(ref_raw < eps)),
        clipped_actual_bins=int(np.sum(cur_raw < eps)),
    )
    return psi, diagnostic


def _summarize_feature_diagnostics(
    feature_diagnostics: list[PSIFeatureDiagnostic],
    *,
    top_k: int,
) -> PSIDiagnostics:
    top = sorted(feature_diagnostics, key=lambda item: item.psi, reverse=True)[: max(top_k, 0)]
    return PSIDiagnostics(
        n_features=len(feature_diagnostics),
        n_bins_total=int(sum(item.n_bins for item in feature_diagnostics)),
        zero_expected_bins=int(sum(item.zero_expected_bins for item in feature_diagnostics)),
        zero_actual_bins=int(sum(item.zero_actual_bins for item in feature_diagnostics)),
        clipped_expected_bins=int(sum(item.clipped_expected_bins for item in feature_diagnostics)),
        clipped_actual_bins=int(sum(item.clipped_actual_bins for item in feature_diagnostics)),
        top_features=tuple(top),
    )
