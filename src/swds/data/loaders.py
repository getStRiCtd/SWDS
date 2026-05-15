from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from swds.data.schema import TabularDataset, TaskLike, TaskType, dataset_name_from_path, normalize_task_type


def load_csv_dataset(
    path: str | Path,
    *,
    target_col: str,
    time_col: str,
    task_type: TaskLike | None = None,
    name: str | None = None,
    split_col: str | None = None,
    **read_csv_kwargs,
) -> TabularDataset:
    frame = pd.read_csv(path, **read_csv_kwargs)
    missing = {target_col, time_col} - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns in {path}: {sorted(missing)}")

    normalized_task = normalize_task_type(task_type, frame[target_col])
    return TabularDataset(
        name=name or dataset_name_from_path(path),
        frame=frame,
        target_col=target_col,
        time_col=time_col,
        task_type=normalized_task,
        split_col=split_col,
        source=str(Path(path)),
    )


def load_tabred_dataset(
    path: str | Path,
    *,
    split: str = "default",
    name: str | None = None,
) -> TabularDataset:
    """Load a TabReD preprocessed dataset directory.

    Expected format follows the public yandex-research/tabred repository:
    `X_num.npy`, optional `X_bin.npy`, optional `X_cat.npy`, `Y.npy`,
    `info.json`, and `split-<split>/{train,val,test}_idx.npy`.
    """

    import json

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"TabReD dataset directory does not exist: {root}")

    split_dir = root / f"split-{split}"
    if not split_dir.exists():
        raise FileNotFoundError(f"TabReD split directory does not exist: {split_dir}")

    info_path = root / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    task_type = _tabred_task_type(info.get("task_type"))

    y = np.load(root / "Y.npy", allow_pickle=False)
    arrays = _load_tabred_feature_arrays(root)
    parts = []
    for part in ("train", "val", "test"):
        idx_path = split_dir / f"{part}_idx.npy"
        if not idx_path.exists():
            raise FileNotFoundError(f"missing TabReD split index: {idx_path}")
        idx = np.load(idx_path, allow_pickle=False)
        part_frame = _tabred_part_frame(arrays, idx)
        part_frame["target"] = y[idx]
        part_frame["time"] = np.arange(len(part_frame), dtype=np.int64)
        part_frame["split"] = part
        parts.append(part_frame)

    frame = pd.concat(parts, ignore_index=True)
    frame["time"] = np.arange(len(frame), dtype=np.int64)
    return TabularDataset(
        name=name or root.name.replace("_", "-"),
        frame=frame,
        target_col="target",
        time_col="time",
        task_type=task_type,
        split_col="split",
        source=str(root),
    )


def load_dataset_from_spec(spec: dict) -> TabularDataset:
    dataset_type = str(spec.get("type", "csv")).lower()
    if dataset_type == "csv":
        return load_csv_dataset(
            spec["path"],
            target_col=spec["target_col"],
            time_col=spec["time_col"],
            task_type=spec.get("task_type"),
            name=spec.get("name"),
            split_col=spec.get("split_col"),
            **dict(spec.get("read_csv_kwargs", {})),
        )
    if dataset_type == "tabred":
        return load_tabred_dataset(
            spec["path"],
            split=spec.get("split", "default"),
            name=spec.get("name"),
        )
    raise ValueError(f"unsupported dataset type: {dataset_type!r}")


def _load_tabred_feature_arrays(root: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in ("X_num", "X_bin", "X_cat", "X_meta"):
        path = root / f"{key}.npy"
        if path.exists():
            arrays[key] = np.load(path, allow_pickle=False)
    if not arrays:
        raise ValueError(f"no TabReD feature arrays found in {root}")
    return arrays


def _tabred_part_frame(arrays: dict[str, np.ndarray], idx: np.ndarray) -> pd.DataFrame:
    columns = {}
    for key, arr in arrays.items():
        part = arr[idx]
        if part.ndim == 1:
            part = part.reshape(-1, 1)
        prefix = {
            "X_num": "num",
            "X_bin": "bin",
            "X_cat": "cat",
            "X_meta": "meta",
        }[key]
        for col_idx in range(part.shape[1]):
            values = part[:, col_idx]
            col_name = f"{prefix}_{col_idx}"
            if key in {"X_cat", "X_meta"}:
                columns[col_name] = pd.Series(values, dtype="string")
            else:
                columns[col_name] = values
    return pd.DataFrame(columns)


def _tabred_task_type(value: str | None) -> TaskType:
    if value is None:
        return TaskType.CLASSIFICATION
    value = str(value).lower()
    if value in {"regression", "reg"}:
        return TaskType.REGRESSION
    if value in {"binclass", "multiclass", "classification", "class"}:
        return TaskType.CLASSIFICATION
    return normalize_task_type(value)
