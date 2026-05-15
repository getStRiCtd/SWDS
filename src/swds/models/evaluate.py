from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from swds.data.schema import TaskType


def evaluate_model(model, X, y_true, *, task_type: TaskType) -> dict[str, float]:
    if task_type == TaskType.CLASSIFICATION:
        return classification_metrics(model, X, y_true)
    return regression_metrics(model, X, y_true)


def classification_metrics(model, X, y_true) -> dict[str, float]:
    y_true = np.asarray(y_true)
    labels = np.unique(y_true)
    pred = model.predict(X)
    metrics = {"accuracy": float(accuracy_score(y_true, pred))}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        try:
            metrics["logloss"] = float(log_loss(y_true, proba, labels=labels))
        except ValueError:
            metrics["logloss"] = float("nan")
        if proba.shape[1] == 2 and len(labels) == 2:
            score = proba[:, 1]
            metrics["roc_auc"] = _safe_metric(roc_auc_score, y_true, score)
            metrics["pr_auc"] = _safe_metric(average_precision_score, y_true, score)
            metrics["brier"] = _safe_metric(brier_score_loss, y_true, score)
        elif len(labels) > 2:
            metrics["roc_auc_ovr"] = _safe_metric(
                roc_auc_score,
                y_true,
                proba,
                multi_class="ovr",
                average="macro",
            )
    return metrics


def regression_metrics(model, X, y_true) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(model.predict(X), dtype=float)
    mse = mean_squared_error(y_true, pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": _safe_metric(r2_score, y_true, pred),
    }


def primary_metric(task_type: TaskType, metrics: dict[str, float]) -> str:
    if task_type == TaskType.CLASSIFICATION:
        for key in ("roc_auc", "roc_auc_ovr", "accuracy"):
            if key in metrics and np.isfinite(metrics[key]):
                return key
        return "accuracy"
    return "rmse"


def higher_is_better(metric_name: str) -> bool:
    return metric_name not in {"logloss", "brier", "rmse", "mae"}


def quality_drop(reference: float, current: float, *, metric_name: str) -> float:
    if not np.isfinite(reference) or not np.isfinite(current):
        return float("nan")
    if higher_is_better(metric_name):
        return float(reference - current)
    return float(current - reference)


def _safe_metric(func, *args, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except ValueError:
        return float("nan")
