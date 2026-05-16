from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Window:
    index: int
    start: int
    end: int
    label: str

    @property
    def size(self) -> int:
        return self.end - self.start


def fixed_count_windows(
    n_rows: int,
    *,
    window_size: int,
    min_size: int | None = None,
    drop_remainder: bool = False,
    prefix: str = "W",
) -> list[Window]:
    if window_size <= 0:
        LOGGER.error("invalid fixed-count window size window_size=%d", window_size)
        raise ValueError("window_size must be positive")
    LOGGER.debug(
        "building fixed-count windows rows=%d window_size=%d min_size=%s drop_remainder=%s prefix=%s",
        n_rows,
        window_size,
        min_size,
        drop_remainder,
        prefix,
    )
    min_size = min_size or max(2, window_size // 2)
    windows: list[Window] = []
    index = 0
    for start in range(0, n_rows, window_size):
        end = min(start + window_size, n_rows)
        if end - start < min_size:
            if drop_remainder:
                continue
            if windows:
                last = windows.pop()
                windows.append(Window(last.index, last.start, end, last.label))
                break
        windows.append(Window(index=index, start=start, end=end, label=f"{prefix}{index + 1}"))
        index += 1
    LOGGER.info("fixed-count windows built rows=%d count=%d prefix=%s", n_rows, len(windows), prefix)
    return windows


def fixed_time_windows(
    frame: pd.DataFrame,
    *,
    time_col: str,
    freq: str,
    min_size: int = 2,
    prefix: str = "T",
) -> list[Window]:
    LOGGER.debug("building fixed-time windows rows=%d time_col=%s freq=%s min_size=%d prefix=%s", len(frame), time_col, freq, min_size, prefix)
    ordered = frame.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    dt = pd.to_datetime(ordered[time_col])
    groups = ordered.groupby(dt.dt.to_period(freq), sort=True).indices
    windows: list[Window] = []
    for idx, (_, row_ids) in enumerate(groups.items()):
        ids = sorted(int(i) for i in row_ids)
        if len(ids) < min_size:
            LOGGER.debug("fixed-time window skipped prefix=%s period_index=%d rows=%d", prefix, idx, len(ids))
            continue
        windows.append(Window(index=len(windows), start=ids[0], end=ids[-1] + 1, label=f"{prefix}{idx + 1}"))
    LOGGER.info("fixed-time windows built rows=%d count=%d prefix=%s", len(frame), len(windows), prefix)
    return windows
