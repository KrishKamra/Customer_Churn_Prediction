"""Centralised, environment-driven configuration.

Every tunable knob in ChurnAI lives here. Values can be overridden with environment
variables prefixed ``CHURNAI_`` (e.g. ``CHURNAI_RANDOM_STATE=7``) or a ``.env`` file.
``DATABASE_URL`` is read without a prefix to match platform conventions (Heroku, Render,
Railway, Supabase, ...).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        env_prefix="CHURNAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---------------------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "Telco_Customer_Churn.csv"
    processed_data_path: Path = PROJECT_ROOT / "data" / "processed" / "churn_clean.parquet"
    model_path: Path = PROJECT_ROOT / "models" / "production_pipeline.joblib"
    metadata_path: Path = PROJECT_ROOT / "models" / "model_metadata.json"
    docs_dir: Path = PROJECT_ROOT / "docs"
    figures_dir: Path = PROJECT_ROOT / "docs" / "figures"
    evaluations_dir: Path = PROJECT_ROOT / "docs" / "evaluations"

    # --- Persistence -----------------------------------------------------------------------
    database_url: str = Field(
        default=f"sqlite:///{(PROJECT_ROOT / 'data' / 'churn.db').as_posix()}",
        validation_alias=AliasChoices("DATABASE_URL", "CHURNAI_DATABASE_URL"),
    )
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # --- Reproducibility & validation -----------------------------------------------------
    random_state: int = 42
    test_size: float = Field(default=0.20, gt=0.0, lt=0.5)
    cv_folds: int = Field(default=5, ge=2)
    cv_repeats: int = Field(default=2, ge=1)

    # --- Compute ---------------------------------------------------------------------------
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # --- Business cost model ---------------------------------------------------------------
    # Cost = cost_fn * FN + cost_fp * FP  (a missed churner is 5x worse than a wasted offer).
    cost_fn: float = 5.0
    cost_fp: float = 1.0
    # Dollar value of one cost unit (i.e. one wasted retention offer). Assumption, not data.
    cost_unit_usd: float = 50.0

    # --- Risk tiers (calibrated probability upper bounds) -----------------------------------
    tier_low_max: float = 0.20
    tier_medium_max: float = 0.45
    tier_high_max: float = 0.70

    # --- Model validation gate (CI) --------------------------------------------------------
    gate_min_roc_auc: float = 0.82
    gate_min_pr_auc: float = 0.60
    gate_min_recall: float = 0.75
    gate_max_brier: float = 0.16


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
