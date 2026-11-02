"""SQLModel schema definitions (portable across SQLite and PostgreSQL)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column, DateTime
from sqlmodel import Field, SQLModel

# Mapping between ORM attribute names (snake_case) and canonical dataset columns.
CUSTOMER_COLUMN_MAP: dict[str, str] = {
    "customer_id": "customerID",
    "gender": "gender",
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "tenure",
    "phone_service": "PhoneService",
    "multiple_lines": "MultipleLines",
    "internet_service": "InternetService",
    "online_security": "OnlineSecurity",
    "online_backup": "OnlineBackup",
    "device_protection": "DeviceProtection",
    "tech_support": "TechSupport",
    "streaming_tv": "StreamingTV",
    "streaming_movies": "StreamingMovies",
    "contract_type": "Contract",
    "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod",
    "monthly_charges": "MonthlyCharges",
    "total_charges": "TotalCharges",
    "churned": "Churn",
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Customer(SQLModel, table=True):
    """Raw demographic, service and billing attributes for one subscriber."""

    __tablename__ = "customers"

    customer_id: str = Field(primary_key=True, max_length=32)
    gender: str = Field(max_length=16)
    senior_citizen: int = 0
    partner: str = Field(max_length=8)
    dependents: str = Field(max_length=8)
    tenure: int = Field(ge=0, index=True)
    phone_service: str = Field(max_length=8)
    multiple_lines: str = Field(max_length=24)
    internet_service: str = Field(max_length=16)
    online_security: str = Field(max_length=24)
    online_backup: str = Field(max_length=24)
    device_protection: str = Field(max_length=24)
    tech_support: str = Field(max_length=24)
    streaming_tv: str = Field(max_length=24)
    streaming_movies: str = Field(max_length=24)
    contract_type: str = Field(max_length=24, index=True)
    paperless_billing: str = Field(max_length=8)
    payment_method: str = Field(max_length=32)
    monthly_charges: float
    total_charges: float
    churned: int | None = Field(default=None, description="Ground-truth label if known.")
    created_at: datetime = Field(
        default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False)
    )

    def to_feature_dict(self) -> dict[str, Any]:
        """Return model-ready features keyed by canonical dataset column names."""
        return {
            CUSTOMER_COLUMN_MAP[k]: getattr(self, k)
            for k in CUSTOMER_COLUMN_MAP
            if k not in ("customer_id", "churned")
        }


class PredictionLog(SQLModel, table=True):
    """Immutable audit record of every score produced by the platform."""

    __tablename__ = "prediction_logs"

    id: int | None = Field(default=None, primary_key=True)
    customer_id: str | None = Field(
        default=None, foreign_key="customers.customer_id", index=True, max_length=32
    )
    model_version: str = Field(max_length=64)
    churn_probability: float = Field(ge=0.0, le=1.0)
    predicted_label: int
    risk_tier: str = Field(max_length=16, index=True)
    source: str = Field(default="ui", max_length=16, description="ui | batch | api")
    shap_summary: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
