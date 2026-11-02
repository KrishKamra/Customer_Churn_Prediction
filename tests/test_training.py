"""End-to-end tournament, reporting and validation-gate tests (small, fast settings)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.config import get_settings
from src.core.exceptions import ModelValidationError
from src.models.evaluate import validate_thresholds
from src.models.reporting import render_benchmarks_markdown
from src.models.train import TournamentResult, run_tournament


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[TournamentResult, Path]:
    out = tmp_path_factory.mktemp("artifacts")
    (out / "docs").mkdir()
    settings = get_settings().model_copy(
        update={
            "cv_folds": 2,
            "cv_repeats": 1,
            "model_path": out / "model.joblib",
            "metadata_path": out / "meta.json",
            "processed_data_path": out / "clean.parquet",
            "docs_dir": out / "docs",
            "figures_dir": out / "docs" / "figures",
            "evaluations_dir": out / "docs" / "evaluations",
        }
    )
    result = run_tournament(settings, models=["elasticnet_lr"], make_figures=False)
    return result, out


def test_tournament_writes_all_artifacts(tiny_run: tuple[TournamentResult, Path]) -> None:
    result, out = tiny_run
    assert result.champion == "elasticnet_lr"
    for rel in (
        "model.joblib",
        "meta.json",
        "clean.parquet",
        "docs/benchmarks.md",
        "docs/evaluations/model_benchmarks.json",
    ):
        assert (out / rel).exists(), rel
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_train"] + meta["n_holdout"] == 7043
    assert len(meta["holdout_fingerprint"]) == 64


def test_tournament_metrics_are_in_expected_band(tiny_run: tuple[TournamentResult, Path]) -> None:
    r = tiny_run[0].results["elasticnet_lr"]
    assert 0.80 < r.holdout["roc_auc"] < 0.90
    assert 0.55 < r.cv_mean["pr_auc"] < 0.75
    assert r.oof_threshold < 0.5  # 5:1 cost matrix pushes the threshold down


def test_benchmark_markdown_contains_every_model(tiny_run: tuple[TournamentResult, Path]) -> None:
    result, _ = tiny_run
    md = render_benchmarks_markdown(result.results, result.metadata)
    assert "ElasticNet Logistic Regression" in md and "Cost analysis" in md


def test_gate_passes_and_detects_regression(tiny_run: tuple[TournamentResult, Path]) -> None:
    _, out = tiny_run
    s = get_settings().model_copy(
        update={"model_path": out / "model.joblib", "metadata_path": out / "meta.json"}
    )
    assert validate_thresholds(s)["roc_auc"] > s.gate_min_roc_auc
    with pytest.raises(ModelValidationError):
        validate_thresholds(s.model_copy(update={"gate_min_roc_auc": 0.99}))


def test_gate_detects_holdout_drift(
    tiny_run: tuple[TournamentResult, Path], tmp_path: Path
) -> None:
    _, out = tiny_run
    shutil.copy(out / "model.joblib", tmp_path / "model.joblib")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    meta["holdout_fingerprint"] = "0" * 64
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    s = get_settings().model_copy(
        update={"model_path": tmp_path / "model.joblib", "metadata_path": tmp_path / "meta.json"}
    )
    with pytest.raises(ModelValidationError, match="fingerprint"):
        validate_thresholds(s)
