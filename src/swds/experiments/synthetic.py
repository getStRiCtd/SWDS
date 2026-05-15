from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swds.data.schema import TabularDataset, TaskType


@dataclass(frozen=True)
class SyntheticSpec:
    n_samples: int = 12000
    n_numeric: int = 10
    n_categorical: int = 3
    drift_start_frac: float = 0.75
    task_type: TaskType = TaskType.CLASSIFICATION
    seed: int = 42


def make_synthetic_temporal_dataset(spec: SyntheticSpec | None = None) -> tuple[TabularDataset, int]:
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n = spec.n_samples
    drift_start = int(n * spec.drift_start_frac)
    n_num = spec.n_numeric

    base_cov = np.eye(n_num)
    for i in range(min(4, n_num)):
        for j in range(i + 1, min(4, n_num)):
            base_cov[i, j] = base_cov[j, i] = 0.25

    x = rng.multivariate_normal(np.zeros(n_num), base_cov, size=n)
    drift_mask = np.arange(n) >= drift_start

    if n_num >= 2:
        x[drift_mask, 0] += 0.75
        x[drift_mask, 1] -= 0.50
    if n_num >= 4:
        rotation = np.array([[0.0, 1.0], [1.0, 0.0]])
        x[drift_mask, 2:4] = x[drift_mask, 2:4] @ rotation
    if n_num >= 6:
        x[drift_mask, 4:6] *= 1.45

    frame = pd.DataFrame(x, columns=[f"x_num_{i}" for i in range(n_num)])
    for j in range(spec.n_categorical):
        pre_probs = np.array([0.60, 0.25, 0.10, 0.05])
        post_probs = np.array([0.25, 0.25, 0.25, 0.25])
        choices = np.array(["a", "b", "c", "d"], dtype=object)
        values = rng.choice(choices, size=n, p=pre_probs)
        values[drift_mask] = rng.choice(choices, size=int(drift_mask.sum()), p=post_probs)
        frame[f"x_cat_{j}"] = values

    if n_num:
        missing_feature = "x_num_0"
        miss = rng.uniform(size=n) < np.where(drift_mask, 0.18, 0.03)
        frame.loc[miss, missing_feature] = np.nan

    beta = rng.normal(size=n_num)
    beta /= np.linalg.norm(beta) + 1e-12
    linear = np.nan_to_num(x, nan=0.0) @ beta
    linear += 0.35 * np.sin(np.linspace(0, 10 * np.pi, n))
    if spec.task_type == TaskType.CLASSIFICATION:
        logits = linear + rng.normal(scale=0.7, size=n)
        prob = 1.0 / (1.0 + np.exp(-logits))
        target = rng.binomial(1, prob)
    else:
        target = linear + rng.normal(scale=0.5, size=n)

    frame["time"] = np.arange(n)
    frame["target"] = target

    dataset = TabularDataset(
        name="synthetic_temporal",
        frame=frame,
        target_col="target",
        time_col="time",
        task_type=spec.task_type,
        source="generated",
    )
    return dataset, drift_start
