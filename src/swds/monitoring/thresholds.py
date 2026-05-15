from __future__ import annotations

import numpy as np


def quantile_threshold(scores, *, quantile: float = 0.95) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        raise ValueError("cannot calibrate threshold from empty scores")
    return float(np.quantile(arr, quantile))


def apply_threshold(scores, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64) >= threshold
