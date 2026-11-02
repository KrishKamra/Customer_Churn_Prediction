"""Inferential-statistics tests (values cross-checked against SciPy directly)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analytics.statistics import (
    CATEGORICAL_TEST_COLUMNS,
    chi_square_suite,
    cohort_churn,
    continuous_tests,
    cramers_v,
    render_statistical_report,
)


def test_sixteen_categorical_variables_are_tested(clean_df: pd.DataFrame) -> None:
    assert len(CATEGORICAL_TEST_COLUMNS) == 16
    assert len(chi_square_suite(clean_df)) == 16


def test_contract_is_primary_driver_and_gender_is_independent(clean_df: pd.DataFrame) -> None:
    chi = chi_square_suite(clean_df).set_index("feature")
    assert chi["cramers_v"].idxmax() == "Contract"
    assert not chi.loc["gender", "significant"]
    assert not chi.loc["PhoneService", "significant"]
    assert chi.loc["gender", "p_value"] == pytest.approx(0.47, abs=0.01)


def test_welch_matches_scipy(clean_df: pd.DataFrame) -> None:
    res = continuous_tests(clean_df).set_index("feature")
    churn = clean_df.loc[clean_df.Churn == 1, "tenure"]
    stay = clean_df.loc[clean_df.Churn == 0, "tenure"]
    expected = stats.ttest_ind(churn, stay, equal_var=False).pvalue
    assert res.loc["tenure", "welch_p"] == pytest.approx(expected)
    assert (res["welch_p"] < 1e-10).all() and (res["mannwhitney_p"] < 1e-10).all()


def test_cramers_v_bounds() -> None:
    independent = np.array([[50, 50], [50, 50]])
    perfect = np.array([[100, 0], [0, 100]])
    assert cramers_v(independent) == pytest.approx(0.0, abs=1e-9)
    assert cramers_v(perfect, bias_correction=False) == pytest.approx(1.0)


def test_danger_zone_cohort(clean_df: pd.DataFrame) -> None:
    cohorts = cohort_churn(clean_df).set_index("tenure_cohort")
    assert cohorts.loc["0-6m", "churn_rate"] > 0.5
    assert cohorts["churn_rate"].is_monotonic_decreasing
    assert cohorts["customers"].sum() == len(clean_df)


def test_report_renders_all_sections(clean_df: pd.DataFrame) -> None:
    md = render_statistical_report(clean_df)
    for heading in (
        "Continuous variables",
        "Collinearity",
        "Categorical association",
        "danger zone",
    ):
        assert heading in md
