from __future__ import annotations

import logging

import numpy as np


LOGGER = logging.getLogger(__name__)


def quantile_threshold(scores, *, quantile: float = 0.95) -> float:
    if not 0.0 < quantile < 1.0:
        LOGGER.error("invalid threshold quantile quantile=%.6f", quantile)
        raise ValueError("quantile must be in (0, 1)")
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        LOGGER.error("cannot calibrate threshold from empty scores")
        raise ValueError("cannot calibrate threshold from empty scores")
    threshold = float(np.quantile(arr, quantile))
    LOGGER.debug("quantile threshold calibrated quantile=%.4f n_scores=%d threshold=%.8f", quantile, len(arr), threshold)
    return threshold


def apply_threshold(scores, threshold: float) -> np.ndarray:
    out = np.asarray(scores, dtype=np.float64) >= threshold
    LOGGER.debug("threshold applied threshold=%.8f n_scores=%d n_triggered=%d", threshold, len(out), int(out.sum()))
    return out
