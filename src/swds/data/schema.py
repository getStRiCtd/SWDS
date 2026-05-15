from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


TaskLike = Literal["classification", "regression"] | TaskType


@dataclass(frozen=True)
class TabularDataset:
    name: str
    frame: pd.DataFrame
    target_col: str
    time_col: str
    task_type: TaskType
    split_col: str | None = None
    source: str | None = None

    @property
    def feature_columns(self) -> list[str]:
        excluded = {self.target_col, self.time_col}
        if self.split_col is not None:
            excluded.add(self.split_col)
        return [col for col in self.frame.columns if col not in excluded]

    @property
    def n_samples(self) -> int:
        return int(len(self.frame))


def normalize_task_type(task_type: TaskLike | None, y: pd.Series | None = None) -> TaskType:
    if task_type is not None:
        return TaskType(str(task_type))
    if y is None:
        raise ValueError("task_type must be provided when y is unavailable")

    non_null = y.dropna()
    if non_null.empty:
        raise ValueError("cannot infer task type from an empty target")

    if pd.api.types.is_bool_dtype(non_null):
        return TaskType.CLASSIFICATION
    if (
        pd.api.types.is_object_dtype(non_null)
        or pd.api.types.is_string_dtype(non_null)
        or isinstance(non_null.dtype, pd.CategoricalDtype)
    ):
        return TaskType.CLASSIFICATION
    if not pd.api.types.is_numeric_dtype(non_null):
        return TaskType.CLASSIFICATION

    unique_count = int(non_null.nunique())
    is_integer_like = pd.api.types.is_integer_dtype(non_null)
    if is_integer_like and unique_count <= max(20, int(0.05 * len(non_null))):
        return TaskType.CLASSIFICATION
    return TaskType.REGRESSION


def dataset_name_from_path(path: str | Path) -> str:
    return Path(path).stem.replace(" ", "_").lower()
