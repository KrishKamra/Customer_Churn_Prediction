"""Multi-tier explainability: global SHAP, local waterfalls and counterfactual prescriptions.

SHAP values are computed on the *inner* (pre-calibration) pipeline. For tree and linear
champions they are exact TreeSHAP / LinearSHAP values in log-odds space; for the stacking
ensemble a model-agnostic permutation explainer is used in probability space. Calibration
is a monotone map, so feature rankings and signs carry over to the calibrated output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.schema import FEATURE_COLUMNS
from src.models.pipeline import MODEL_CATEGORICAL

TREE_TYPES = (XGBClassifier, LGBMClassifier, CatBoostClassifier, RandomForestClassifier)

PRETTY_NAMES: dict[str, str] = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "SeniorCitizen": "Senior citizen",
    "log_total_charges": "log(Total charges)",
    "log_monthly_charges": "log(Monthly charges)",
    "charge_velocity_ratio": "Charge velocity ratio",
    "avg_monthly_spend": "Avg. monthly spend",
    "num_addon_services": "# add-on services",
    "is_fiber_month_to_month": "Fiber + month-to-month",
    "is_auto_payment": "Automatic payment",
    "risk_ratio__InternetService_x_Contract": "Risk ratio: Internet x Contract",
    "risk_ratio__Contract_x_PaymentMethod": "Risk ratio: Contract x Payment",
    "risk_ratio__tenure_cohort_x_Contract": "Risk ratio: Cohort x Contract",
    "tenure_cohort": "Tenure cohort",
}


def unwrap_pipeline(model: Any) -> Pipeline:
    """Return the fitted inner pipeline from a (possibly calibrated) production model."""
    if hasattr(model, "calibrated_classifiers_"):
        inner = model.calibrated_classifiers_[0].estimator
    else:
        inner = model
    if not isinstance(inner, Pipeline):
        raise TypeError(f"Expected a sklearn Pipeline, got {type(inner).__name__}")
    return inner


def feature_group(transformed_name: str) -> str:
    """Map a transformed column (e.g. ``Contract_Two year``) to its source feature."""
    for col in sorted(MODEL_CATEGORICAL, key=len, reverse=True):
        if transformed_name.startswith(f"{col}_"):
            return col
    return transformed_name


@dataclass
class LocalExplanation:
    """Per-instance attribution aggregated to human-level features."""

    base_value: float
    contributions: pd.DataFrame  # columns: feature, label, value, shap
    output_space: str
    raw_shap: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def top(self, k: int = 10) -> pd.DataFrame:
        order = self.contributions["shap"].abs().sort_values(ascending=False).index
        return self.contributions.loc[order].head(k)


class ChurnExplainer:
    """Unified SHAP facade over every champion family."""

    def __init__(self, model: Any, background: pd.DataFrame, max_background: int = 200) -> None:
        self.pipeline = unwrap_pipeline(model)
        self.transformer = self.pipeline[:-1]
        self.estimator = self.pipeline[-1]
        bg_raw = background[FEATURE_COLUMNS].sample(
            min(max_background, len(background)), random_state=0
        )
        self.background = self._transform(bg_raw)
        self.feature_names = list(self.background.columns)
        self.output_space, self._explainer = self._build()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self.transformer.transform(X[FEATURE_COLUMNS])
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)

    def _build(self) -> tuple[str, Any]:
        est = self.estimator
        if isinstance(est, TREE_TYPES):
            return "log-odds", shap.TreeExplainer(est)
        if isinstance(est, LogisticRegression):
            masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
            return "log-odds", shap.LinearExplainer(est, masker)
        names = self.feature_names

        def predict(Z: np.ndarray) -> np.ndarray:
            return np.asarray(est.predict_proba(pd.DataFrame(Z, columns=names))[:, 1])

        masker = shap.maskers.Independent(self.background, max_samples=100)
        return "probability", shap.Explainer(predict, masker, algorithm="permutation")

    @staticmethod
    def _positive_class(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        return arr[..., 1] if arr.ndim == 3 else arr

    def shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """SHAP explanation for raw rows (positive class only)."""
        Xt = self._transform(X)
        if self.output_space == "probability":
            exp = self._explainer(Xt.to_numpy(), max_evals=2 * Xt.shape[1] + 1)
        else:
            exp = self._explainer(Xt)
        values = self._positive_class(exp.values)
        base = np.asarray(exp.base_values)
        base = base[..., 1] if base.ndim == 2 else base
        return shap.Explanation(
            values=values,
            base_values=base,
            data=Xt.to_numpy(),
            feature_names=self.feature_names,
        )

    def explain_instance(self, row: pd.DataFrame) -> LocalExplanation:
        """Aggregate one-hot SHAP columns back to their source features."""
        exp = self.shap_values(row.iloc[[0]])
        vals = np.asarray(exp.values)[0]
        groups: dict[str, float] = {}
        for name, v in zip(self.feature_names, vals, strict=True):
            g = feature_group(name)
            groups[g] = groups.get(g, 0.0) + float(v)
        # Unscaled engineered values (features + risk ratios, before imputation/scaling).
        unscaled = self.transformer[:2].transform(row[FEATURE_COLUMNS].iloc[[0]]).iloc[0]
        records = []
        for g, v in groups.items():
            value = unscaled[g] if g in unscaled.index else ""
            label = PRETTY_NAMES.get(g, g)
            if isinstance(value, float | np.floating):
                label = f"{label} = {value:,.2f}"
            else:
                label = f"{label} = {value}"
            records.append({"feature": g, "label": label, "value": value, "shap": v})
        base = float(np.ravel(exp.base_values)[0])
        return LocalExplanation(base, pd.DataFrame(records), self.output_space, vals)

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per source feature over ``X``."""
        exp = self.shap_values(X)
        df = pd.DataFrame(np.abs(exp.values), columns=self.feature_names)
        grouped = df.T.groupby([feature_group(c) for c in df.columns]).sum().T
        imp = grouped.mean().sort_values(ascending=False)
        return imp.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})

    def plot_summary(self, X: pd.DataFrame, path: Path, max_display: int = 20) -> Path:
        """Save a global SHAP beeswarm plot."""
        import matplotlib.pyplot as plt

        from src.analytics.plotting import apply_theme

        apply_theme()
        exp = self.shap_values(X)
        exp.feature_names = [PRETTY_NAMES.get(n, n) for n in self.feature_names]
        plt.figure(figsize=(10, 8), dpi=150)
        shap.plots.beeswarm(exp, max_display=max_display, show=False)
        plt.title(f"Global SHAP summary ({self.output_space})", fontsize=13, weight="bold")
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close(plt.gcf())
        return path


# --------------------------------------------------------------------------------------
# Counterfactual prescriptions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Counterfactual:
    """A single actionable what-if intervention and its modelled effect."""

    action: str
    description: str
    baseline_probability: float
    new_probability: float

    @property
    def delta(self) -> float:
        return self.new_probability - self.baseline_probability


Intervention = Callable[[pd.Series], pd.Series | None]


def _set(**changes: Any) -> Intervention:
    def apply(row: pd.Series) -> pd.Series | None:
        if all(row.get(k) == v for k, v in changes.items()):
            return None
        out = row.copy()
        for k, v in changes.items():
            out[k] = v
        return out

    return apply


def _requires_internet(inner: Intervention) -> Intervention:
    def apply(row: pd.Series) -> pd.Series | None:
        return None if row["InternetService"] == "No" else inner(row)

    return apply


def _discount(pct: float) -> Intervention:
    def apply(row: pd.Series) -> pd.Series | None:
        out = row.copy()
        out["MonthlyCharges"] = round(float(row["MonthlyCharges"]) * (1 - pct), 2)
        return out

    return apply


def _bundle(row: pd.Series) -> pd.Series | None:
    if row["Contract"] != "Month-to-month":
        return None
    out = row.copy()
    out["Contract"] = "One year"
    out["PaymentMethod"] = "Credit card (automatic)"
    if row["InternetService"] != "No":
        out["TechSupport"] = "Yes"
    return out


INTERVENTIONS: dict[str, Intervention] = {
    "Migrate to a one-year contract": lambda r: (
        _set(Contract="One year")(r) if r["Contract"] == "Month-to-month" else None
    ),
    "Migrate to a two-year contract": _set(Contract="Two year"),
    "Switch to automatic credit-card payment": lambda r: (
        None if "automatic" in str(r["PaymentMethod"]) else
        _set(PaymentMethod="Credit card (automatic)")(r)
    ),
    "Bundle Tech Support": _requires_internet(_set(TechSupport="Yes")),
    "Bundle Online Security": _requires_internet(_set(OnlineSecurity="Yes")),
    "Apply a 15% loyalty discount": _discount(0.15),
    "Retention bundle (annual + auto-pay + support)": _bundle,
}


def generate_counterfactuals(
    predict_proba: Callable[[pd.DataFrame], np.ndarray],
    customer: pd.DataFrame,
    top_k: int = 4,
    min_improvement: float = 0.005,
) -> list[Counterfactual]:
    """Score actionable interventions for one customer and rank by risk reduction.

    These are *model-based* what-ifs, not causal estimates: they answer "what would the
    model predict for an otherwise identical customer with this change".
    """
    row = customer[FEATURE_COLUMNS].iloc[0]
    baseline = float(predict_proba(customer[FEATURE_COLUMNS].iloc[[0]])[0])
    candidates: list[tuple[str, pd.Series]] = []
    for action, fn in INTERVENTIONS.items():
        changed = fn(row)
        if changed is not None:
            candidates.append((action, changed))
    if not candidates:
        return []
    frame = pd.DataFrame([c for _, c in candidates], columns=FEATURE_COLUMNS)
    probs = predict_proba(frame)
    results = [
        Counterfactual(
            action=action,
            description=f"{action} reduces churn risk from {baseline:.0%} to {p:.0%}",
            baseline_probability=baseline,
            new_probability=float(p),
        )
        for (action, _), p in zip(candidates, probs, strict=True)
        if baseline - float(p) >= min_improvement
    ]
    return sorted(results, key=lambda c: c.new_probability)[:top_k]
