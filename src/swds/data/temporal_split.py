from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def sort_by_time(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in frame.columns:
        LOGGER.error("time column absent time_col=%s columns=%s", time_col, list(frame.columns))
        raise ValueError(f"time column {time_col!r} is absent")
    LOGGER.info("sorting frame by time column time_col=%s rows=%d", time_col, len(frame))
    return frame.sort_values(time_col, kind="mergesort").reset_index(drop=True)


def temporal_split(
    frame: pd.DataFrame,
    *,
    time_col: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> TemporalSplit:
    LOGGER.info(
        "temporal split requested rows=%d train_frac=%.4f val_frac=%.4f test_frac=%.4f",
        len(frame),
        train_frac,
        val_frac,
        test_frac,
    )
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {total:.6f}")
    if min(train_frac, val_frac, test_frac) <= 0:
        raise ValueError("split fractions must be positive")

    ordered = sort_by_time(frame, time_col)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    if train_end == 0 or val_end <= train_end or val_end >= n:
        LOGGER.error("not enough rows for temporal split rows=%d train_end=%d val_end=%d", n, train_end, val_end)
        raise ValueError("not enough rows for a non-empty train/val/test split")

    split = TemporalSplit(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        val=ordered.iloc[train_end:val_end].reset_index(drop=True),
        test=ordered.iloc[val_end:].reset_index(drop=True),
    )
    LOGGER.info("temporal split completed train=%d val=%d test=%d", len(split.train), len(split.val), len(split.test))
    return split


def official_split(frame: pd.DataFrame, *, split_col: str) -> TemporalSplit:
    LOGGER.info("official split requested split_col=%s rows=%d", split_col, len(frame))
    values = frame[split_col].astype(str).str.lower()
    train = frame.loc[values == "train"].reset_index(drop=True)
    val = frame.loc[values.isin(["val", "valid", "validation"])].reset_index(drop=True)
    test = frame.loc[values == "test"].reset_index(drop=True)
    if min(len(train), len(val), len(test)) == 0:
        LOGGER.error("official split missing partition split_col=%s train=%d val=%d test=%d", split_col, len(train), len(val), len(test))
        raise ValueError(f"split column {split_col!r} must contain train/val/test rows")
    LOGGER.info("official split completed train=%d val=%d test=%d", len(train), len(val), len(test))
    return TemporalSplit(train=train, val=val, test=test)
