"""Canonical data contract for the IBM Telco Customer Churn dataset.

This module is the single source of truth for column names, categorical levels and the
Pydantic validation schema applied to every inbound customer payload (UI form, batch CSV,
database row) before it reaches the model.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.core.exceptions import DataValidationError

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

YES_NO = ("No", "Yes")
INTERNET_ADDON = ("No", "Yes", "No internet service")

CATEGORICAL_LEVELS: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male"),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ("No", "Yes", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}

CATEGORICAL_COLUMNS: list[str] = list(CATEGORICAL_LEVELS)
NUMERIC_COLUMNS: list[str] = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLUMNS: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
RAW_COLUMNS: list[str] = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]

AddOn = Literal["No", "Yes", "No internet service"]
YesNo = Literal["No", "Yes"]


class CustomerFeatures(BaseModel):
    """Validated model input for a single customer (raw Telco column names)."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    gender: Literal["Female", "Male"] = "Female"
    SeniorCitizen: int = Field(default=0, ge=0, le=1)
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    tenure: int = Field(default=12, ge=0, le=120)
    PhoneService: YesNo = "Yes"
    MultipleLines: Literal["No", "Yes", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: AddOn = "No"
    OnlineBackup: AddOn = "No"
    DeviceProtection: AddOn = "No"
    TechSupport: AddOn = "No"
    StreamingTV: AddOn = "No"
    StreamingMovies: AddOn = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: YesNo = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    MonthlyCharges: float = Field(default=70.0, ge=0.0, le=500.0)
    TotalCharges: float = Field(default=840.0, ge=0.0, le=100_000.0)

    def to_frame(self) -> pd.DataFrame:
        """Return a one-row DataFrame in canonical column order."""
        return pd.DataFrame([self.model_dump()], columns=FEATURE_COLUMNS)


def validate_feature_frame(df: pd.DataFrame, *, required: Iterable[str] = FEATURE_COLUMNS) -> None:
    """Raise :class:`DataValidationError` if mandatory feature columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Input is missing {len(missing)} required column(s): {', '.join(missing)}",
            missing_columns=missing,
        )
