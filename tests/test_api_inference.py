"""Deterministic inference tests against the committed production pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import get_settings
from src.core.exceptions import DataValidationError, ModelNotFoundError
from src.features.schema import FEATURE_COLUMNS, CustomerFeatures
from src.models.explainability import ChurnExplainer, generate_counterfactuals
from src.models.inference import ChurnPredictor, risk_tier

pytestmark = pytest.mark.skipif(
    not get_settings().model_path.exists(), reason="production pipeline not trained"
)

HIGH_RISK = CustomerFeatures(
    tenure=2,
    Contract="Month-to-month",
    InternetService="Fiber optic",
    MonthlyCharges=95.0,
    TotalCharges=190.0,
    PaymentMethod="Electronic check",
    TechSupport="No",
    OnlineSecurity="No",
)
LOW_RISK = CustomerFeatures(
    tenure=66,
    Contract="Two year",
    InternetService="DSL",
    MonthlyCharges=60.0,
    TotalCharges=3960.0,
    PaymentMethod="Bank transfer (automatic)",
    TechSupport="Yes",
    OnlineSecurity="Yes",
    Partner="Yes",
    Dependents="Yes",
    PaperlessBilling="No",
)


@pytest.fixture(scope="module")
def predictor() -> ChurnPredictor:
    return ChurnPredictor.load()


def test_metadata_is_complete(predictor: ChurnPredictor) -> None:
    for key in ("model_version", "champion", "decision_threshold", "holdout", "feature_columns"):
        assert key in predictor.metadata
    assert predictor.metadata["feature_columns"] == FEATURE_COLUMNS
    assert 0.0 < predictor.threshold < 1.0


def test_inference_is_deterministic(predictor: ChurnPredictor) -> None:
    a = predictor.predict_one(HIGH_RISK)
    b = predictor.predict_one(HIGH_RISK.model_dump())
    assert a == b


def test_risk_ordering_is_sensible(predictor: ChurnPredictor) -> None:
    high = predictor.predict_one(HIGH_RISK)
    low = predictor.predict_one(LOW_RISK)
    assert high.churn_probability > 0.5 > low.churn_probability
    assert high.risk_tier in ("High", "Critical") and low.risk_tier == "Low"
    assert high.predicted_label == 1 and low.predicted_label == 0


def test_batch_scores_match_single_scores(
    predictor: ChurnPredictor, clean_df: pd.DataFrame
) -> None:
    batch = clean_df.head(20)
    scored = predictor.score_frame(batch)
    single = [
        predictor.predict_one(r[FEATURE_COLUMNS].to_dict()).churn_probability
        for _, r in batch.iterrows()
    ]
    np.testing.assert_allclose(scored["churn_probability"].to_numpy(), single, rtol=1e-9)


def test_invalid_payload_raises_domain_error(predictor: ChurnPredictor) -> None:
    with pytest.raises(DataValidationError):
        predictor.predict_one({"tenure": "forever"})


def test_missing_model_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ModelNotFoundError):
        ChurnPredictor.load(tmp_path / "missing.joblib", tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("p", "tier"), [(0.05, "Low"), (0.3, "Medium"), (0.6, "High"), (0.95, "Critical")]
)
def test_risk_tier_boundaries(p: float, tier: str) -> None:
    assert risk_tier(p) == tier


def test_counterfactual_contract_migration_reduces_risk(predictor: ChurnPredictor) -> None:
    cfs = generate_counterfactuals(predictor.predict_proba, HIGH_RISK.to_frame(), top_k=10)
    assert cfs, "expected at least one risk-reducing intervention"
    assert all(c.new_probability < c.baseline_probability for c in cfs)
    assert any("contract" in c.action.lower() for c in cfs)


def test_local_shap_explanation_is_additive(
    predictor: ChurnPredictor, clean_df: pd.DataFrame
) -> None:
    explainer = ChurnExplainer(predictor.model, clean_df, max_background=100)
    exp = explainer.explain_instance(HIGH_RISK.to_frame())
    assert not exp.contributions.empty
    # Aggregating one-hot columns must preserve the total attribution.
    assert exp.contributions["shap"].sum() == pytest.approx(exp.raw_shap.sum(), rel=1e-6)
    assert exp.top(3).iloc[0]["shap"] != 0
