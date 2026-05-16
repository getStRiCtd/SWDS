from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from swds.drift.utils import matrix_vstack, subsample_rows


@dataclass(frozen=True)
class PreparedC2STReference:
    X_ref_sub: object


def prepare_c2st_reference(X_ref, *, max_samples: int = 4000, seed: int = 42) -> PreparedC2STReference:
    X_ref_sub, _ = subsample_rows(X_ref, max_samples // 2, seed=seed)
    return PreparedC2STReference(X_ref_sub=X_ref_sub)


def classifier_two_sample_score_prepared(
    prepared: PreparedC2STReference,
    X_cur,
    *,
    max_samples: int = 4000,
    seed: int = 42,
    n_splits: int = 5,
    n_jobs: int | None = None,
) -> float:
    X_cur_sub, _ = subsample_rows(X_cur, max_samples // 2, seed=seed + 1)
    return _classifier_two_sample_score_from_samples(
        prepared.X_ref_sub,
        X_cur_sub,
        seed=seed,
        n_splits=n_splits,
        n_jobs=n_jobs,
    )


def classifier_two_sample_score(
    X_ref,
    X_cur,
    *,
    max_samples: int = 4000,
    seed: int = 42,
    n_splits: int = 5,
    n_jobs: int | None = None,
) -> float:
    prepared = prepare_c2st_reference(X_ref, max_samples=max_samples, seed=seed)
    X_cur_sub, _ = subsample_rows(X_cur, max_samples // 2, seed=seed + 1)
    return _classifier_two_sample_score_from_samples(
        prepared.X_ref_sub,
        X_cur_sub,
        seed=seed,
        n_splits=n_splits,
        n_jobs=n_jobs,
    )


def _classifier_two_sample_score_from_samples(
    X_ref_sub,
    X_cur_sub,
    *,
    seed: int,
    n_splits: int,
    n_jobs: int | None,
) -> float:
    x = matrix_vstack([X_ref_sub, X_cur_sub])
    y = np.concatenate([np.zeros(X_ref_sub.shape[0]), np.ones(X_cur_sub.shape[0])])

    min_class_count = int(min(np.bincount(y.astype(int))))
    if min_class_count < 2:
        return float("nan")

    n_splits = min(n_splits, min_class_count)
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,
        random_state=seed,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, x, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
    return float(np.mean(scores))
