"""Deterministic inference API around the serialized production pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.core.exceptions import DataValidationError, ModelNotFoundError
from src.features.schema import FEATURE_COLUMNS, CustomerFeatures, validate_feature_frame

RiskTier = Literal["Low", "Medium", "High", "Critical"]
RISK_TIERS: tuple[RiskTier, ...] = ("Low", "Medium", "High", "Critical")


def risk_tier(probability: float, settings: Settings | None = None) -> RiskTier:
    """Map a calibrated churn probability onto the four business risk tiers."""
    s = settings or get_settings()
    if probability < s.tier_low_max:
        return "Low"
    if probability < s.tier_medium_max:
        return "Medium"
    if probability < s.tier_high_max:
        return "High"
    return "Critical"


@dataclass(frozen=True)
class PredictionResult:
    """Single-customer scoring outcome."""

    churn_probability: float
    predicted_label: int
    risk_tier: RiskTier
    threshold: float
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChurnPredictor:
    """Loads the calibrated production pipeline plus metadata and scores customers."""

    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        self.model = model
        self.metadata = metadata

    @classmethod
    def load(cls, model_path: Path | None = None, metadata_path: Path | None = None) -> ChurnPredictor:
        s = get_settings()
        model_path = model_path or s.model_path
        metadata_path = metadata_path or s.metadata_path
        if not model_path.exists():
            raise ModelNotFoundError(
                f"No production pipeline at {model_path}. Train one with "
                "`uv run python -m src.models.train`."
            )
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(joblib.load(model_path), metadata)

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", 0.5))

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unversioned"))

    @property
    def champion_name(self) -> str:
        return str(self.metadata.get("champion_display_name", "Unknown model"))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated churn probabilities for a frame with canonical raw columns."""
        validate_feature_frame(X)
        return np.asarray(self.model.predict_proba(X[FEATURE_COLUMNS])[:, 1], dtype=float)

    def predict_one(self, customer: CustomerFeatures | dict[str, Any]) -> PredictionResult:
        """Validate and score one customer payload."""
        try:
            payload = (
                customer
                if isinstance(customer, CustomerFeatures)
                else CustomerFeatures.model_validate(customer)
            )
        except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
            raise DataValidationError(f"Invalid customer payload: {exc}") from exc
        prob = float(self.predict_proba(payload.to_frame())[0])
        return PredictionResult(
            churn_probability=prob,
            predicted_label=int(prob >= self.threshold),
            risk_tier=risk_tier(prob),
            threshold=self.threshold,
            model_version=self.version,
        )

    def score_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Batch scoring: returns probability, label and tier columns aligned to ``X``."""
        proba = self.predict_proba(X)
        return pd.DataFrame(
            {
                "churn_probability": proba,
                "predicted_label": (proba >= self.threshold).astype(int),
                "risk_tier": [risk_tier(p) for p in proba],
            },
            index=X.index,
        )
