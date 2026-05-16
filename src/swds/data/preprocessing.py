from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureTypeSummary:
    numeric: list[str]
    low_cardinality_categorical: list[str]
    high_cardinality_categorical: list[str]

    @property
    def categorical(self) -> list[str]:
        return self.low_cardinality_categorical + self.high_cardinality_categorical


class HashingCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Deterministic stateless hashing for high-cardinality categorical columns."""

    def __init__(
        self,
        n_features: int = 256,
        *,
        alternate_sign: bool = False,
        missing_value: str = "__missing__",
    ) -> None:
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        self.missing_value = missing_value

    def fit(self, X, y=None):
        self.feature_names_in_ = _feature_names(X)
        return self

    def transform(self, X):
        names = getattr(self, "feature_names_in_", _feature_names(X))
        arr = _as_2d_object_array(X)
        samples = []
        for row in arr:
            tokens = []
            for name, value in zip(names, row, strict=False):
                if pd.isna(value):
                    value = self.missing_value
                tokens.append(f"{name}={value}")
            samples.append(tokens)
        hasher = FeatureHasher(
            n_features=self.n_features,
            input_type="string",
            alternate_sign=self.alternate_sign,
        )
        return hasher.transform(samples)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"hash_{i}" for i in range(self.n_features)], dtype=object)


def infer_feature_types(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    max_onehot_cardinality: int = 32,
) -> FeatureTypeSummary:
    columns = list(columns)
    LOGGER.info("inferring feature types columns=%d max_onehot_cardinality=%d", len(columns), max_onehot_cardinality)
    numeric: list[str] = []
    low_card: list[str] = []
    high_card: list[str] = []

    for col in columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric.append(col)
            continue

        nunique = int(series.nunique(dropna=True))
        if nunique <= max_onehot_cardinality:
            low_card.append(col)
        else:
            high_card.append(col)

    summary = FeatureTypeSummary(
        numeric=numeric,
        low_cardinality_categorical=low_card,
        high_cardinality_categorical=high_card,
    )
    LOGGER.info(
        "feature type inference completed numeric=%d low_card_categorical=%d high_card_categorical=%d",
        len(summary.numeric),
        len(summary.low_cardinality_categorical),
        len(summary.high_cardinality_categorical),
    )
    return summary


def build_preprocessor(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    max_onehot_cardinality: int = 32,
    hash_features: int = 256,
) -> tuple[ColumnTransformer, FeatureTypeSummary]:
    LOGGER.info(
        "building preprocessor train_rows=%d feature_columns=%d hash_features=%d",
        len(train_frame),
        len(feature_columns),
        hash_features,
    )
    summary = infer_feature_types(
        train_frame,
        feature_columns,
        max_onehot_cardinality=max_onehot_cardinality,
    )
    transformers = []

    if summary.numeric:
        LOGGER.info("adding numeric preprocessing columns=%d", len(summary.numeric))
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, summary.numeric))

    if summary.low_cardinality_categorical:
        LOGGER.info("adding low-cardinality categorical preprocessing columns=%d", len(summary.low_cardinality_categorical))
        low_card_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("categorical_low", low_card_pipeline, summary.low_cardinality_categorical))

    if summary.high_cardinality_categorical:
        LOGGER.info("adding high-cardinality hashing preprocessing columns=%d", len(summary.high_cardinality_categorical))
        high_card_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("hash", HashingCategoricalEncoder(n_features=hash_features)),
            ]
        )
        transformers.append(("categorical_high", high_card_pipeline, summary.high_cardinality_categorical))

    if not transformers:
        LOGGER.error("no usable feature columns were found")
        raise ValueError("no usable feature columns were found")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
    LOGGER.info("preprocessor built transformers=%s", [name for name, _, _ in transformers])
    return preprocessor, summary


def transform_to_float32(preprocessor: ColumnTransformer, frame: pd.DataFrame):
    LOGGER.debug("transforming frame to float32 rows=%d columns=%d", len(frame), len(frame.columns))
    transformed = preprocessor.transform(frame)
    if sparse.issparse(transformed):
        out = transformed.astype(np.float32)
        LOGGER.debug("transformed sparse matrix shape=%s nnz=%d dtype=%s", out.shape, out.nnz, out.dtype)
        return out
    out = np.asarray(transformed, dtype=np.float32)
    LOGGER.debug("transformed dense matrix shape=%s dtype=%s", out.shape, out.dtype)
    return out


def _feature_names(X) -> list[str]:
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    arr = _as_2d_object_array(X)
    return [f"feature_{i}" for i in range(arr.shape[1])]


def _as_2d_object_array(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=object)
    else:
        arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr
