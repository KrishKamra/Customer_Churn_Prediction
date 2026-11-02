"""Custom scikit-learn transformers.

Both transformers are DataFrame-in / DataFrame-out so that column names survive all the
way to the estimator, which keeps SHAP explanations human-readable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.features.engineer import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    add_engineered_features,
)
from src.features.schema import FEATURE_COLUMNS, validate_feature_frame

DEFAULT_INTERACTIONS: tuple[tuple[str, str], ...] = (
    ("InternetService", "Contract"),
    ("Contract", "PaymentMethod"),
    ("tenure_cohort", "Contract"),
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Select canonical features and append stateless engineered features.

    Extra columns (e.g. ``customerID`` or ``Churn`` in a batch upload) are dropped, which
    is the first line of defence against target leakage at inference time.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        validate_feature_frame(X)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        validate_feature_frame(X)
        return add_engineered_features(X[FEATURE_COLUMNS])

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        return np.asarray(
            [*FEATURE_COLUMNS, *ENGINEERED_NUMERIC, *ENGINEERED_CATEGORICAL], dtype=object
        )


class RiskRatioEncoder(BaseEstimator, TransformerMixin):
    """Smoothed risk-ratio encoding of high-churn categorical intersections.

    For every pair ``(a, b)`` the encoder learns, on the *training fold only*::

        ratio(a=i, b=j) = [(n_ij * rate_ij + m * prior) / (n_ij + m)] / prior

    where ``m`` is the smoothing strength. A ratio of 2.0 means the segment churns at twice
    the base rate. Unseen combinations encode to 1.0 (no evidence -> base rate).
    """

    def __init__(
        self,
        interactions: Sequence[tuple[str, str]] = DEFAULT_INTERACTIONS,
        smoothing: float = 20.0,
    ) -> None:
        self.interactions = interactions
        self.smoothing = smoothing

    @staticmethod
    def _key(X: pd.DataFrame, a: str, b: str) -> pd.Series:
        return X[a].astype(str) + "|" + X[b].astype(str)

    @staticmethod
    def output_name(a: str, b: str) -> str:
        return f"risk_ratio__{a}_x_{b}"

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> RiskRatioEncoder:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.shape[0] != X.shape[0]:
            raise ValueError("X and y have inconsistent lengths.")
        self.prior_ = float(y_arr.mean())
        self.mappings_: dict[tuple[str, str], dict[str, float]] = {}
        target = pd.Series(y_arr, index=X.index)
        for a, b in self.interactions:
            stats = target.groupby(self._key(X, a, b)).agg(["sum", "count"])
            smoothed = (stats["sum"] + self.smoothing * self.prior_) / (
                stats["count"] + self.smoothing
            )
            self.mappings_[(a, b)] = (smoothed / self.prior_).to_dict()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "mappings_")
        out = X.copy()
        for (a, b), mapping in self.mappings_.items():
            out[self.output_name(a, b)] = (
                self._key(X, a, b).map(mapping).astype(float).fillna(1.0).to_numpy()
            )
        return out

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        check_is_fitted(self, "mappings_")
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray(
            [*base, *(self.output_name(a, b) for a, b in self.interactions)], dtype=object
        )

    @property
    def output_columns(self) -> list[str]:
        return [self.output_name(a, b) for a, b in self.interactions]
