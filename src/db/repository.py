"""CRUD data-access layer. The UI and services never touch SQL directly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import Engine, func
from sqlmodel import col, select

from src.db.database import get_engine, session_scope
from src.db.models import CUSTOMER_COLUMN_MAP, Customer, PredictionLog
from src.features.schema import ID_COLUMN

_REVERSE_MAP = {v: k for k, v in CUSTOMER_COLUMN_MAP.items()}


class CustomerRepository:
    """Customer and prediction-audit persistence operations."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or get_engine()

    # --- Customers -------------------------------------------------------------------------

    def seed_from_dataframe(self, df: pd.DataFrame, chunk_size: int = 1000) -> int:
        """Idempotently insert customers from a cleaned dataset frame.

        Rows whose ``customerID`` already exists are skipped, so the seed can be re-run
        safely on every app start. Returns the number of newly inserted rows.
        """
        frame = df.rename(columns=_REVERSE_MAP)
        frame = frame[[c for c in CUSTOMER_COLUMN_MAP if c in frame.columns]]
        with session_scope(self.engine) as session:
            existing = set(session.exec(select(Customer.customer_id)).all())
        new_rows = frame[~frame["customer_id"].isin(existing)]
        new_rows = new_rows.drop_duplicates(subset="customer_id")
        records = new_rows.to_dict(orient="records")
        for start in range(0, len(records), chunk_size):
            with session_scope(self.engine) as session:
                session.add_all(
                    Customer(**{k: _py(v) for k, v in r.items()})
                    for r in records[start : start + chunk_size]
                )
        return len(records)

    def count_customers(self) -> int:
        with session_scope(self.engine) as session:
            return int(session.exec(select(func.count()).select_from(Customer)).one())

    def get_customer(self, customer_id: str) -> Customer | None:
        with session_scope(self.engine) as session:
            return session.get(Customer, customer_id.strip())

    def search_customer_ids(self, prefix: str, limit: int = 20) -> list[str]:
        """Case-insensitive prefix autocomplete over customer IDs."""
        pattern = f"{prefix.strip().upper()}%"
        with session_scope(self.engine) as session:
            stmt = (
                select(Customer.customer_id)
                .where(func.upper(Customer.customer_id).like(pattern))
                .order_by(Customer.customer_id)
                .limit(limit)
            )
            return list(session.exec(stmt).all())

    def upsert_customer(self, data: dict[str, Any]) -> Customer:
        """Insert or update a customer from canonical-column ``data``."""
        attrs = {_REVERSE_MAP.get(k, k): _py(v) for k, v in data.items() if k in _REVERSE_MAP}
        with session_scope(self.engine) as session:
            customer = session.get(Customer, attrs["customer_id"])
            if customer is None:
                customer = Customer(**attrs)
            else:
                for k, v in attrs.items():
                    setattr(customer, k, v)
            session.add(customer)
            session.flush()
            session.refresh(customer)
            return customer

    # --- Prediction audit -----------------------------------------------------------------

    def log_prediction(
        self,
        *,
        customer_id: str | None,
        model_version: str,
        churn_probability: float,
        predicted_label: int,
        risk_tier: str,
        shap_summary: dict[str, Any] | None = None,
        source: str = "ui",
    ) -> PredictionLog:
        entry = PredictionLog(
            customer_id=customer_id,
            model_version=model_version,
            churn_probability=float(churn_probability),
            predicted_label=int(predicted_label),
            risk_tier=risk_tier,
            shap_summary=shap_summary,
            source=source,
        )
        with session_scope(self.engine) as session:
            session.add(entry)
            session.flush()
            session.refresh(entry)
            return entry

    def log_predictions_bulk(self, entries: Sequence[dict[str, Any]]) -> int:
        with session_scope(self.engine) as session:
            session.add_all(PredictionLog(**e) for e in entries)
        return len(entries)

    def prediction_history(self, customer_id: str, limit: int = 200) -> list[PredictionLog]:
        """Chronological audit trail for one customer."""
        with session_scope(self.engine) as session:
            stmt = (
                select(PredictionLog)
                .where(PredictionLog.customer_id == customer_id)
                .order_by(col(PredictionLog.created_at).asc())
                .limit(limit)
            )
            return list(session.exec(stmt).all())

    def recent_predictions(self, limit: int = 50) -> list[PredictionLog]:
        with session_scope(self.engine) as session:
            stmt = select(PredictionLog).order_by(col(PredictionLog.created_at).desc()).limit(limit)
            return list(session.exec(stmt).all())

    def history_frame(self, customer_id: str) -> pd.DataFrame:
        rows = self.prediction_history(customer_id)
        return pd.DataFrame([r.model_dump() for r in rows])


def customer_to_frame(customer: Customer) -> pd.DataFrame:
    """One-row canonical-feature frame for model scoring."""
    return pd.DataFrame([{ID_COLUMN: customer.customer_id, **customer.to_feature_dict()}])


def _py(value: Any) -> Any:
    """Convert numpy / pandas scalars into plain Python for the DB driver."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value
