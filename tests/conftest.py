"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import Engine

from src.db.database import build_engine, init_db
from src.features.engineer import clean_raw, load_raw


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean_raw(raw_df)


@pytest.fixture(scope="session")
def sample_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Small stratified sample for fast pipeline tests."""
    return (
        clean_df.groupby("Churn", group_keys=False)
        .sample(frac=0.12, random_state=0)
        .reset_index(drop=True)
    )


@pytest.fixture()
def memory_engine() -> Engine:
    return init_db(build_engine("sqlite://"))
