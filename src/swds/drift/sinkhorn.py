from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from swds.drift.utils import subsample_rows, to_dense


def sinkhorn_divergence_score(
    X_ref,
    X_cur,
    *,
    max_samples: int = 1000,
    reg: float = 0.05,
    seed: int = 42,
) -> float:
    try:
        import ot
    except ImportError as exc:
        raise ImportError("sinkhorn drift score requires POT: install with `uv sync --extra ot`") from exc

    X_ref_sub, _ = subsample_rows(X_ref, max_samples, seed=seed)
    X_cur_sub, _ = subsample_rows(X_cur, max_samples, seed=seed + 1)
    x = to_dense(X_ref_sub)
    y = to_dense(X_cur_sub)
    a = np.full(len(x), 1.0 / len(x))
    b = np.full(len(y), 1.0 / len(y))

    m_xy = pairwise_distances(x, y, metric="sqeuclidean")
    m_xx = pairwise_distances(x, x, metric="sqeuclidean")
    m_yy = pairwise_distances(y, y, metric="sqeuclidean")
    scale = _positive_median(m_xy)
    m_xy = m_xy / scale
    m_xx = m_xx / scale
    m_yy = m_yy / scale

    xy = float(ot.sinkhorn2(a, b, m_xy, reg=reg, method="sinkhorn_log"))
    xx = float(ot.sinkhorn2(a, a, m_xx, reg=reg, method="sinkhorn_log"))
    yy = float(ot.sinkhorn2(b, b, m_yy, reg=reg, method="sinkhorn_log"))
    divergence = max((xy - 0.5 * xx - 0.5 * yy) * scale, 0.0)
    return float(np.sqrt(divergence))


def _positive_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values) & (values > 0)]
    if len(finite) == 0:
        return 1.0
    return max(float(np.median(finite)), 1e-12)
