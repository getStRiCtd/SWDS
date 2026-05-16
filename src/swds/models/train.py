from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from swds.data.schema import TaskType


LOGGER = logging.getLogger(__name__)


class DenseInputEstimator(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(_to_dense(X), y)
        return self

    def predict(self, X):
        return self.estimator.predict(_to_dense(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(_to_dense(X))

    def decision_function(self, X):
        return self.estimator.decision_function(_to_dense(X))


class DenseInputClassifier(DenseInputEstimator, ClassifierMixin):
    pass


class DenseInputRegressor(DenseInputEstimator, RegressorMixin):
    pass


def make_model(task_type: TaskType, *, model_name: str = "linear", seed: int = 42):
    model_name = model_name.lower()
    LOGGER.info("creating model name=%s task=%s seed=%d", model_name, TaskType(task_type).value, seed)
    if task_type == TaskType.CLASSIFICATION:
        if model_name == "linear":
            return LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
        if model_name in {"hist_gbdt", "gbdt"}:
            return DenseInputClassifier(
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    l2_regularization=1e-3,
                    random_state=seed,
                )
            )
        if model_name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError("LightGBM model requires `uv sync --extra gbdt`") from exc
            return LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=seed,
                verbosity=-1,
            )
        if model_name == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError("XGBoost model requires `uv sync --extra gbdt`") from exc
            return XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                eval_metric="logloss",
                random_state=seed,
            )
        if model_name == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:
                raise ImportError("CatBoost model requires `uv sync --extra gbdt`") from exc
            return DenseInputClassifier(
                CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="Logloss",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                )
            )
    elif task_type == TaskType.REGRESSION:
        if model_name == "linear":
            return Ridge(alpha=1.0, random_state=seed)
        if model_name in {"hist_gbdt", "gbdt"}:
            return DenseInputRegressor(
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=300,
                    l2_regularization=1e-3,
                    random_state=seed,
                )
            )
        if model_name == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError("LightGBM model requires `uv sync --extra gbdt`") from exc
            return LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                verbosity=-1,
            )
        if model_name == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError("XGBoost model requires `uv sync --extra gbdt`") from exc
            return XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                random_state=seed,
            )
        if model_name == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:
                raise ImportError("CatBoost model requires `uv sync --extra gbdt`") from exc
            return DenseInputRegressor(
                CatBoostRegressor(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="RMSE",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                )
            )
    raise ValueError(f"unsupported model_name={model_name!r} for task_type={task_type!r}")


def train_model(X_train, y_train, *, task_type: TaskType, model_name: str = "linear", seed: int = 42):
    LOGGER.info(
        "model training started name=%s task=%s rows=%d shape=%s",
        model_name,
        TaskType(task_type).value,
        len(y_train),
        tuple(getattr(X_train, "shape", ())),
    )
    model = make_model(task_type, model_name=model_name, seed=seed)
    model.fit(X_train, y_train)
    LOGGER.info("model training completed name=%s rows=%d", model_name, len(y_train))
    return model


def _to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)
