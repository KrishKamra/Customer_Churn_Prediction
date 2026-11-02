"""Data ingestion, cleaning and stateless feature generation.

All functions here are *stateless* (they never look at the target), so they are safe to
apply before the train/test split. Anything that learns from data (imputation medians,
target-derived risk ratios, scaling) lives in :mod:`src.features.transformers` and is fit
inside the scikit-learn pipeline, i.e. strictly on training folds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_settings
from src.core.exceptions import DataValidationError
from src.features.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    RAW_COLUMNS,
    TARGET_COLUMN,
)

EPSILON = 1e-6

TENURE_BINS: list[float] = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS: list[str] = ["0-6m", "7-12m", "1-2y", "2-4y", "4y+"]

ADDON_SERVICES: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ENGINEERED_NUMERIC: list[str] = [
    "log_total_charges",
    "log_monthly_charges",
    "charge_velocity_ratio",
    "avg_monthly_spend",
    "num_addon_services",
    "is_fiber_month_to_month",
    "is_auto_payment",
]
ENGINEERED_CATEGORICAL: list[str] = ["tenure_cohort"]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw Telco CSV snapshot and assert the column contract."""
    path = path or get_settings().raw_data_path
    df = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Raw data missing columns: {missing}", missing_columns=missing)
    return df


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise types and fix known data-quality issues.

    * ``TotalCharges`` is stored as text with 11 blank strings, all for ``tenure == 0``
      customers who have not been billed yet. They are set to ``0.0`` (the true value),
      rather than imputed, which would fabricate spend.
    * Categorical columns are whitespace-stripped strings.
    * ``Churn`` is mapped to ``{0, 1}`` when present.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    unbilled = out["TotalCharges"].isna() & (out["tenure"] == 0)
    out.loc[unbilled, "TotalCharges"] = 0.0
    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip().astype(object)
    for col in ("SeniorCitizen", "tenure"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    if TARGET_COLUMN in out.columns and out[TARGET_COLUMN].dtype != np.int64:
        mapped = out[TARGET_COLUMN].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
        if mapped.isna().any():
            raise DataValidationError("Target column contains values other than Yes/No.")
        out[TARGET_COLUMN] = mapped.astype(np.int64)
    if ID_COLUMN in out.columns:
        out[ID_COLUMN] = out[ID_COLUMN].astype(str)
    return out


def load_clean(path: Path | None = None) -> pd.DataFrame:
    """Load the processed parquet if it exists, otherwise clean the raw CSV."""
    settings = get_settings()
    parquet = path or settings.processed_data_path
    if parquet.exists():
        return pd.read_parquet(parquet)
    return clean_raw(load_raw())


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with only the canonical model features in ``X``."""
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].astype(int)


def tenure_cohort(tenure: pd.Series) -> pd.Series:
    """Bucket tenure (months) into behavioural cohorts; ``0-6m`` is the danger zone."""
    return pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append deterministic, target-free features to a copy of ``df``.

    * ``charge_velocity_ratio`` = TotalCharges / (tenure * MonthlyCharges + eps): an
      RFM-style proxy; values far from 1 indicate plan changes or discounts over time.
    * ``log_*`` = log1p of right-skewed financial scales.
    * ``avg_monthly_spend`` = TotalCharges / max(tenure, 1).
    * ``num_addon_services`` = count of subscribed add-ons (stickiness proxy).
    * ``is_fiber_month_to_month`` = the single highest-churn intersection in the data.
    * ``is_auto_payment`` = automatic bank / card payment.
    * ``tenure_cohort`` = categorical lifecycle bucket.
    """
    out = df.copy()
    tenure = out["tenure"].astype(float)
    monthly = out["MonthlyCharges"].astype(float)
    total = out["TotalCharges"].astype(float)

    out["log_total_charges"] = np.log1p(total.clip(lower=0))
    out["log_monthly_charges"] = np.log1p(monthly.clip(lower=0))
    out["charge_velocity_ratio"] = total / (tenure * monthly + EPSILON)
    out.loc[tenure == 0, "charge_velocity_ratio"] = 1.0
    out["charge_velocity_ratio"] = out["charge_velocity_ratio"].clip(upper=5.0)
    out["avg_monthly_spend"] = total / tenure.clip(lower=1)
    out["num_addon_services"] = (out[ADDON_SERVICES] == "Yes").sum(axis=1).astype(float)
    out["is_fiber_month_to_month"] = (
        (out["InternetService"] == "Fiber optic") & (out["Contract"] == "Month-to-month")
    ).astype(float)
    out["is_auto_payment"] = (
        out["PaymentMethod"].astype(str).str.contains("automatic").astype(float)
    )
    out["tenure_cohort"] = tenure_cohort(tenure)
    return out
