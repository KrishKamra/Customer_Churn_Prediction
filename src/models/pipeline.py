"""End-to-end scikit-learn pipeline and model-zoo definitions.

Every candidate model is wrapped in the *same* pipeline::

    raw frame -> FeatureEngineer -> RiskRatioEncoder -> ColumnTransformer -> estimator

so the tournament compares algorithms, not preprocessing. Learned preprocessing (medians,
risk ratios, scaling, one-hot vocabularies) is re-fit inside every CV fold, eliminating
train/validation leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.fixes import parse_version
from xgboost import XGBClassifier

from src.core.hardware import Device
from src.features.engineer import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC
from src.features.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.features.transformers import DEFAULT_INTERACTIONS, FeatureEngineer, RiskRatioEncoder

RISK_RATIO_COLUMNS = [RiskRatioEncoder.output_name(a, b) for a, b in DEFAULT_INTERACTIONS]
MODEL_NUMERIC = [*NUMERIC_COLUMNS, *ENGINEERED_NUMERIC, *RISK_RATIO_COLUMNS]
MODEL_CATEGORICAL = [*CATEGORICAL_COLUMNS, *ENGINEERED_CATEGORICAL]


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry describing a tournament contender."""

    key: str
    display_name: str
    family: str  # "linear" | "tree" | "ensemble"


MODEL_SPECS: dict[str, ModelSpec] = {
    "elasticnet_lr": ModelSpec("elasticnet_lr", "ElasticNet Logistic Regression", "linear"),
    "balanced_rf": ModelSpec("balanced_rf", "Balanced Random Forest", "tree"),
    "lightgbm": ModelSpec("lightgbm", "LightGBM", "tree"),
    "xgboost": ModelSpec("xgboost", "XGBoost", "tree"),
    "catboost": ModelSpec("catboost", "CatBoost", "tree"),
    "stacking": ModelSpec("stacking", "Stacking (CatBoost + XGBoost + LR)", "ensemble"),
}


def build_preprocessor() -> ColumnTransformer:
    """Impute, scale and one-hot encode the engineered frame."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, MODEL_NUMERIC), ("cat", categorical, MODEL_CATEGORICAL)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# scikit-learn >= 1.8 infers ElasticNet from ``l1_ratio`` and deprecates ``penalty``; older
# releases (e.g. Colab's preinstalled build) silently fall back to L2 without it.
_ELASTICNET_KWARGS: dict[str, Any] = (
    {} if parse_version(sklearn.__version__) >= parse_version("1.8") else {"penalty": "elasticnet"}
)

# Estimators that train on the GPU when device == "cuda" (CV folds then run sequentially).
GPU_MODELS: frozenset[str] = frozenset({"xgboost", "catboost", "stacking"})


def _logistic(seed: int) -> LogisticRegression:
    return LogisticRegression(
        **_ELASTICNET_KWARGS,
        l1_ratio=0.5,
        C=0.5,
        solver="saga",
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    )


def _xgboost(seed: int, device: Device, pos_weight: float, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        scale_pos_weight=pos_weight,
        tree_method="hist",
        device=device,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=n_jobs,
    )


def _catboost(seed: int, device: Device, n_jobs: int) -> CatBoostClassifier:
    params: dict[str, Any] = {
        "iterations": 600,
        "learning_rate": 0.04,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "auto_class_weights": "Balanced",
        "random_seed": seed,
        "verbose": 0,
        "allow_writing_files": False,
    }
    if device == "cuda":
        params.update(task_type="GPU", devices="0")
    else:
        params["thread_count"] = n_jobs
    return CatBoostClassifier(**params)


def build_model_zoo(
    seed: int = 42,
    device: Device = "cpu",
    pos_weight: float = 2.77,
    n_jobs: int = 1,
) -> dict[str, BaseEstimator]:
    """Instantiate every contender with sensible, regularised defaults.

    Args:
        seed: Random seed propagated to every stochastic estimator.
        device: ``"cuda"`` enables GPU training for XGBoost and CatBoost.
        pos_weight: Negative/positive ratio used by XGBoost/LightGBM for class balance.
        n_jobs: Threads per estimator (keep at 1 when CV folds already run in parallel).
    """
    return {
        "elasticnet_lr": _logistic(seed),
        "balanced_rf": BalancedRandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            scale_pos_weight=pos_weight,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        ),
        "xgboost": _xgboost(seed, device, pos_weight, n_jobs),
        "catboost": _catboost(seed, device, n_jobs),
        "stacking": StackingClassifier(
            estimators=[
                ("catboost", _catboost(seed, device, n_jobs)),
                ("xgboost", _xgboost(seed, device, pos_weight, n_jobs)),
                ("logreg", _logistic(seed)),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000),
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            stack_method="predict_proba",
            n_jobs=1,
        ),
    }


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Wrap an estimator in the shared feature pipeline."""
    pipe = Pipeline(
        [
            ("features", FeatureEngineer()),
            ("risk", RiskRatioEncoder()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
    pipe.set_output(transform="pandas")
    return pipe


def calibrate(
    pipeline: Pipeline, *, method: str = "isotonic", cv: int = 5, seed: int = 42
) -> CalibratedClassifierCV:
    """Wrap a full pipeline in probability calibration.

    ``ensemble=False`` fits a *single* pipeline on all training data and learns the
    calibration map from out-of-fold predictions, so the inner model stays directly
    explainable with SHAP.
    """
    return CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
        ensemble=False,
    )


def search_space(key: str) -> dict[str, Any]:
    """Randomised hyper-parameter search distributions (``model__`` prefixed)."""
    spaces: dict[str, dict[str, Any]] = {
        "elasticnet_lr": {"model__C": loguniform(1e-2, 10), "model__l1_ratio": uniform(0, 1)},
        "balanced_rf": {
            "model__n_estimators": randint(200, 800),
            "model__min_samples_leaf": randint(1, 10),
            "model__max_depth": [None, 8, 12, 16],
        },
        "lightgbm": {
            "model__n_estimators": randint(200, 1200),
            "model__learning_rate": loguniform(5e-3, 0.1),
            "model__num_leaves": randint(7, 63),
            "model__min_child_samples": randint(10, 80),
            "model__reg_lambda": loguniform(1e-2, 20),
        },
        "xgboost": {
            "model__n_estimators": randint(200, 1200),
            "model__learning_rate": loguniform(5e-3, 0.1),
            "model__max_depth": randint(2, 7),
            "model__min_child_weight": randint(1, 10),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_lambda": loguniform(1e-1, 20),
        },
        "catboost": {
            "model__iterations": randint(300, 1500),
            "model__learning_rate": loguniform(1e-2, 0.1),
            "model__depth": randint(3, 8),
            "model__l2_leaf_reg": loguniform(1, 20),
        },
        "stacking": {"model__final_estimator__C": loguniform(1e-2, 10)},
    }
    return spaces[key]


def to_cpu(estimator: Any) -> Any:
    """Recursively switch GPU-trained XGBoost boosters to CPU inference (portable pickle)."""
    if isinstance(estimator, XGBClassifier):
        estimator.set_params(device="cpu")
        return estimator
    for attr in ("estimator", "calibrated_classifiers_", "steps", "estimators_", "named_steps"):
        child = getattr(estimator, attr, None)
        if child is None:
            continue
        if isinstance(child, dict):
            child = list(child.values())
        if isinstance(child, list | tuple):
            for item in child:
                to_cpu(item[1] if isinstance(item, tuple) else item)
        else:
            to_cpu(child)
    return estimator
