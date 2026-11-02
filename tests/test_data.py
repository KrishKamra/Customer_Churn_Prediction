"""Data ingestion and schema-validation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.exceptions import DataValidationError
from src.features.engineer import clean_raw, split_features_target, tenure_cohort
from src.features.schema import (
    CATEGORICAL_LEVELS,
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    TARGET_COLUMN,
    CustomerFeatures,
    validate_feature_frame,
)


def test_raw_snapshot_shape_and_columns(raw_df: pd.DataFrame) -> None:
    assert raw_df.shape == (7043, 21)
    assert list(raw_df.columns) == RAW_COLUMNS
    assert raw_df["customerID"].is_unique


def test_class_balance_matches_reference(clean_df: pd.DataFrame) -> None:
    assert clean_df[TARGET_COLUMN].mean() == pytest.approx(0.2654, abs=1e-3)


def test_blank_total_charges_are_zero_for_new_customers(raw_df: pd.DataFrame) -> None:
    blanks = pd.to_numeric(raw_df["TotalCharges"], errors="coerce").isna()
    assert blanks.sum() == 11
    assert (raw_df.loc[blanks, "tenure"] == 0).all()
    cleaned = clean_raw(raw_df)
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert (cleaned.loc[blanks, "TotalCharges"] == 0).all()


def test_categorical_levels_match_contract(clean_df: pd.DataFrame) -> None:
    for col, levels in CATEGORICAL_LEVELS.items():
        assert set(clean_df[col].unique()) == set(levels), col


def test_target_is_binary_int(clean_df: pd.DataFrame) -> None:
    assert set(clean_df[TARGET_COLUMN].unique()) == {0, 1}


def test_split_excludes_target_and_id(clean_df: pd.DataFrame) -> None:
    X, y = split_features_target(clean_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert TARGET_COLUMN not in X.columns and "customerID" not in X.columns
    assert len(X) == len(y)


def test_invalid_target_raises() -> None:
    df = pd.DataFrame(
        {"TotalCharges": ["1"], "tenure": [1], "MonthlyCharges": [1.0], TARGET_COLUMN: ["Maybe"]}
    )
    with pytest.raises(DataValidationError):
        clean_raw(df)


def test_validate_feature_frame_reports_missing_columns() -> None:
    with pytest.raises(DataValidationError) as info:
        validate_feature_frame(pd.DataFrame({"tenure": [1]}))
    assert "Contract" in info.value.missing_columns


def test_customer_schema_rejects_out_of_domain_values() -> None:
    with pytest.raises(ValueError):
        CustomerFeatures(Contract="Lifetime")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CustomerFeatures(tenure=-3)


def test_customer_schema_round_trips_to_frame() -> None:
    frame = CustomerFeatures(tenure=5).to_frame()
    assert list(frame.columns) == FEATURE_COLUMNS and frame.loc[0, "tenure"] == 5


def test_tenure_cohorts_are_exhaustive(clean_df: pd.DataFrame) -> None:
    cohorts = tenure_cohort(clean_df["tenure"])
    assert cohorts.isin(["0-6m", "7-12m", "1-2y", "2-4y", "4y+"]).all()
    danger = clean_df.loc[cohorts == "0-6m", TARGET_COLUMN].mean()
    assert danger > 0.5
