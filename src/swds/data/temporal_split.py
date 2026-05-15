from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def sort_by_time(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in frame.columns:
        raise ValueError(f"time column {time_col!r} is absent")
    return frame.sort_values(time_col, kind="mergesort").reset_index(drop=True)


def temporal_split(
    frame: pd.DataFrame,
    *,
    time_col: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> TemporalSplit:
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
        raise ValueError("not enough rows for a non-empty train/val/test split")

    return TemporalSplit(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        val=ordered.iloc[train_end:val_end].reset_index(drop=True),
        test=ordered.iloc[val_end:].reset_index(drop=True),
    )


def official_split(frame: pd.DataFrame, *, split_col: str) -> TemporalSplit:
    values = frame[split_col].astype(str).str.lower()
    train = frame.loc[values == "train"].reset_index(drop=True)
    val = frame.loc[values.isin(["val", "valid", "validation"])].reset_index(drop=True)
    test = frame.loc[values == "test"].reset_index(drop=True)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"split column {split_col!r} must contain train/val/test rows")
    return TemporalSplit(train=train, val=val, test=test)
