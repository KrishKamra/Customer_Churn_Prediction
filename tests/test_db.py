"""Database CRUD and session-persistence tests (in-memory SQLite)."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import Engine

from src.core.exceptions import DatabaseError
from src.db.database import _normalise_url, session_scope
from src.db.models import Customer
from src.db.repository import CustomerRepository, customer_to_frame
from src.features.schema import FEATURE_COLUMNS


def test_seed_is_idempotent(memory_engine: Engine, clean_df: pd.DataFrame) -> None:
    repo = CustomerRepository(memory_engine)
    subset = clean_df.head(300)
    assert repo.seed_from_dataframe(subset) == 300
    assert repo.seed_from_dataframe(subset) == 0
    assert repo.count_customers() == 300


def test_seed_skips_duplicate_ids_within_frame(
    memory_engine: Engine, clean_df: pd.DataFrame
) -> None:
    repo = CustomerRepository(memory_engine)
    dup = pd.concat([clean_df.head(5), clean_df.head(5)])
    assert repo.seed_from_dataframe(dup) == 5


def test_prefix_search_is_case_insensitive(memory_engine: Engine, clean_df: pd.DataFrame) -> None:
    repo = CustomerRepository(memory_engine)
    repo.seed_from_dataframe(clean_df.head(500))
    target = clean_df.iloc[10]["customerID"]
    hits = repo.search_customer_ids(target[:6].lower())
    assert target in hits
    assert all(h.upper().startswith(target[:6].upper()) for h in hits)


def test_customer_round_trip_to_model_frame(memory_engine: Engine, clean_df: pd.DataFrame) -> None:
    repo = CustomerRepository(memory_engine)
    repo.seed_from_dataframe(clean_df.head(1))
    row = clean_df.iloc[0]
    customer = repo.get_customer(row["customerID"])
    assert customer is not None
    frame = customer_to_frame(customer)
    for col in FEATURE_COLUMNS:
        assert frame.loc[0, col] == row[col], col


def test_prediction_log_history_is_chronological(
    memory_engine: Engine, clean_df: pd.DataFrame
) -> None:
    repo = CustomerRepository(memory_engine)
    repo.seed_from_dataframe(clean_df.head(1))
    cid = clean_df.iloc[0]["customerID"]
    for p in (0.2, 0.5, 0.8):
        repo.log_prediction(
            customer_id=cid,
            model_version="test",
            churn_probability=p,
            predicted_label=int(p > 0.4),
            risk_tier="High",
            shap_summary={"Contract": 0.4},
        )
    history = repo.prediction_history(cid)
    assert [h.churn_probability for h in history] == [0.2, 0.5, 0.8]
    assert history[0].shap_summary == {"Contract": 0.4}
    assert repo.history_frame(cid).shape[0] == 3


def test_ad_hoc_prediction_without_customer(memory_engine: Engine) -> None:
    entry = CustomerRepository(memory_engine).log_prediction(
        customer_id=None,
        model_version="t",
        churn_probability=0.1,
        predicted_label=0,
        risk_tier="Low",
    )
    assert entry.id is not None and entry.customer_id is None


def test_foreign_key_is_enforced(memory_engine: Engine) -> None:
    with pytest.raises(DatabaseError):
        CustomerRepository(memory_engine).log_prediction(
            customer_id="DOES-NOT-EXIST",
            model_version="t",
            churn_probability=0.1,
            predicted_label=0,
            risk_tier="Low",
        )


def test_session_scope_rolls_back_on_error(memory_engine: Engine, clean_df: pd.DataFrame) -> None:
    repo = CustomerRepository(memory_engine)
    with pytest.raises(RuntimeError), session_scope(memory_engine) as session:
        session.add(
            Customer(
                **{
                    "customer_id": "TMP-1",
                    "gender": "Male",
                    "partner": "No",
                    "dependents": "No",
                    "tenure": 1,
                    "phone_service": "Yes",
                    "multiple_lines": "No",
                    "internet_service": "DSL",
                    "online_security": "No",
                    "online_backup": "No",
                    "device_protection": "No",
                    "tech_support": "No",
                    "streaming_tv": "No",
                    "streaming_movies": "No",
                    "contract_type": "Month-to-month",
                    "paperless_billing": "Yes",
                    "payment_method": "Mailed check",
                    "monthly_charges": 10.0,
                    "total_charges": 10.0,
                }
            )
        )
        raise RuntimeError("boom")
    assert repo.get_customer("TMP-1") is None


def test_upsert_updates_existing(memory_engine: Engine, clean_df: pd.DataFrame) -> None:
    repo = CustomerRepository(memory_engine)
    row = clean_df.iloc[0].to_dict()
    repo.upsert_customer(row)
    repo.upsert_customer({**row, "tenure": 99})
    customer = repo.get_customer(row["customerID"])
    assert customer is not None and customer.tenure == 99


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgres://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_database_url_normalisation(url: str, expected: str) -> None:
    assert _normalise_url(url) == expected
