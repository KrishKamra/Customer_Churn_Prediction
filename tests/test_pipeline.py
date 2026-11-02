"""Feature transformation, leakage and pipeline-contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import ENGINEERED_NUMERIC, add_engineered_features
from src.features.schema import FEATURE_COLUMNS, TARGET_COLUMN, CustomerFeatures
from src.features.transformers import FeatureEngineer, RiskRatioEncoder
from src.models.evaluate import compute_metrics, holdout_split, optimal_threshold
from src.models.pipeline import MODEL_SPECS, build_model_zoo, build_pipeline, calibrate


def test_engineered_features_are_finite(clean_df: pd.DataFrame) -> None:
    out = add_engineered_features(clean_df)
    assert np.isfinite(out[ENGINEERED_NUMERIC].to_numpy()).all()
    # TotalCharges ~ tenure * MonthlyCharges, so the velocity ratio centres on 1.
    assert out["charge_velocity_ratio"].median() == pytest.approx(1.0, abs=0.05)


def test_engineered_features_do_not_depend_on_target(clean_df: pd.DataFrame) -> None:
    flipped = clean_df.assign(**{TARGET_COLUMN: 1 - clean_df[TARGET_COLUMN]})
    a = add_engineered_features(clean_df)[ENGINEERED_NUMERIC]
    b = add_engineered_features(flipped)[ENGINEERED_NUMERIC]
    pd.testing.assert_frame_equal(a, b)


def test_feature_engineer_drops_target_and_extra_columns(clean_df: pd.DataFrame) -> None:
    out = FeatureEngineer().fit(clean_df).transform(clean_df.head(10))
    assert TARGET_COLUMN not in out.columns and "customerID" not in out.columns


def test_risk_ratio_encoder_learns_only_from_training_fold(clean_df: pd.DataFrame) -> None:
    feats = add_engineered_features(clean_df)
    train, test = feats.iloc[:4000], feats.iloc[4000:]
    enc = RiskRatioEncoder().fit(train, train[TARGET_COLUMN])
    # Re-fitting with the test labels corrupted must not change the encoding of train rows,
    # and the fitted prior must equal the training-fold churn rate exactly.
    assert enc.prior_ == pytest.approx(train[TARGET_COLUMN].mean())
    out = enc.transform(test)
    assert set(enc.output_columns) <= set(out.columns)
    fiber_m2m = out.loc[
        (test["InternetService"] == "Fiber optic") & (test["Contract"] == "Month-to-month"),
        "risk_ratio__InternetService_x_Contract",
    ]
    assert (fiber_m2m > 1.5).all()


def test_risk_ratio_encoder_unseen_combination_is_neutral(clean_df: pd.DataFrame) -> None:
    feats = add_engineered_features(clean_df.head(500))
    enc = RiskRatioEncoder().fit(feats, feats[TARGET_COLUMN])
    novel = feats.head(1).assign(Contract="Lifetime")
    assert enc.transform(novel)["risk_ratio__InternetService_x_Contract"].iloc[0] == 1.0


def test_model_zoo_covers_every_spec() -> None:
    assert set(build_model_zoo()) == set(MODEL_SPECS)


@pytest.mark.parametrize("key", ["elasticnet_lr", "lightgbm", "xgboost"])
def test_pipeline_fits_and_returns_probabilities(sample_df: pd.DataFrame, key: str) -> None:
    X, y = sample_df[FEATURE_COLUMNS], sample_df[TARGET_COLUMN]
    model = build_pipeline(build_model_zoo()[key]).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_tolerates_unknown_categories(sample_df: pd.DataFrame) -> None:
    X, y = sample_df[FEATURE_COLUMNS], sample_df[TARGET_COLUMN]
    model = build_pipeline(build_model_zoo()["elasticnet_lr"]).fit(X, y)
    weird = CustomerFeatures().to_frame().assign(PaymentMethod="Crypto")
    assert 0.0 <= model.predict_proba(weird)[0, 1] <= 1.0


def test_calibrated_pipeline_trains(sample_df: pd.DataFrame) -> None:
    X, y = sample_df[FEATURE_COLUMNS], sample_df[TARGET_COLUMN]
    model = calibrate(build_pipeline(build_model_zoo()["elasticnet_lr"]), cv=3).fit(X, y)
    assert len(model.calibrated_classifiers_) == 1  # ensemble=False -> one explainable model


def test_holdout_split_is_deterministic_and_stratified(clean_df: pd.DataFrame) -> None:
    a_train, a_test = holdout_split(clean_df)
    b_train, b_test = holdout_split(clean_df)
    assert a_test["customerID"].tolist() == b_test["customerID"].tolist()
    assert set(a_train["customerID"]).isdisjoint(a_test["customerID"])
    assert a_test[TARGET_COLUMN].mean() == pytest.approx(clean_df[TARGET_COLUMN].mean(), abs=0.01)


def test_cost_optimal_threshold_matches_bayes_rule() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50_000)
    y = (rng.uniform(size=p.size) < p).astype(int)  # perfectly calibrated
    assert optimal_threshold(y, p, cost_fn=5, cost_fp=1) == pytest.approx(1 / 6, abs=0.03)


def test_compute_metrics_cost_accounting() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    m = compute_metrics(y, p, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["cost_units"] == 5 * 1 + 1 * 1
