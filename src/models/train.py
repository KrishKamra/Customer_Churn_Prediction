"""Reproducible, CLI-driven model tournament and champion serialization.

Examples::

    uv run python -m src.models.train                       # full tournament
    uv run python -m src.models.train --quick               # smoke run (2 models, 1 repeat)
    uv run python -m src.models.train --tune-iter 30         # + champion hyper-parameter search

Protocol:
    1. Clean the raw snapshot and persist ``data/processed/churn_clean.parquet``.
    2. Stratified 80/20 train/holdout split (fixed seed, fingerprinted).
    3. Every contender -- wrapped in the shared pipeline *and* probability calibration --
       is scored with Repeated Stratified K-Fold CV on the training split only.
    4. The champion (highest mean CV PR-AUC) is optionally tuned, refit on the full
       training split, and its cost-optimal threshold is chosen from out-of-fold
       predictions (never from the holdout).
    5. All contenders are scored once on the untouched holdout for the benchmark report.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version as pkg_version
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from src import __version__
from src.config import Settings, get_settings
from src.core.hardware import machine_info, machine_summary
from src.core.logger import get_logger
from src.features.engineer import clean_raw, load_raw, split_features_target
from src.features.schema import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from src.models.evaluate import (
    compute_metrics,
    holdout_split,
    ids_fingerprint,
    measure_latency_ms,
    optimal_threshold,
    plot_roc_pr,
)
from src.models.pipeline import (
    MODEL_SPECS,
    build_model_zoo,
    build_pipeline,
    calibrate,
    search_space,
)

log = get_logger("train")

CV_METRICS = ("pr_auc", "roc_auc", "brier", "recall", "precision", "f2")
QUICK_MODELS = ("elasticnet_lr", "lightgbm")


def _silence_warnings() -> None:
    warnings.filterwarnings("ignore", message=".*feature names.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class ModelResult:
    """Everything measured for one contender."""

    key: str
    display_name: str
    cv_mean: dict[str, float] = field(default_factory=dict)
    cv_std: dict[str, float] = field(default_factory=dict)
    cv_cost_per_customer: float = float("nan")
    oof_threshold: float = 0.5
    fit_time_s: float = float("nan")
    latency_ms: float = float("nan")
    holdout: dict[str, float] = field(default_factory=dict)
    tuned_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TournamentResult:
    champion: str
    results: dict[str, ModelResult]
    metadata: dict[str, Any]


def _fit_fold(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    _silence_warnings()
    model = calibrate(build_pipeline(clone(estimator)), cv=3, seed=seed)
    start = time.perf_counter()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    fit_time = time.perf_counter() - start
    return val_idx, model.predict_proba(X.iloc[val_idx])[:, 1], fit_time


def cross_validate_model(
    key: str,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    settings: Settings,
    n_jobs: int,
) -> ModelResult:
    """Repeated stratified CV of the calibrated pipeline; also derives the OOF threshold."""
    cv = RepeatedStratifiedKFold(
        n_splits=settings.cv_folds, n_repeats=settings.cv_repeats, random_state=settings.random_state
    )
    splits = list(cv.split(X, y))
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(estimator, X, y, tr, va, settings.random_state) for tr, va in splits
    )
    fold_metrics: list[dict[str, float]] = []
    oof = np.zeros((settings.cv_repeats, len(y)))
    fit_times = []
    for i, (val_idx, proba, fit_time) in enumerate(outputs):
        oof[i // settings.cv_folds, val_idx] = proba
        fold_metrics.append(compute_metrics(y.iloc[val_idx], proba, 0.5, settings))
        fit_times.append(fit_time)

    thresholds = [optimal_threshold(y, oof[r], settings.cost_fn, settings.cost_fp) for r in range(settings.cv_repeats)]
    threshold = float(np.median(thresholds))
    costs = [
        compute_metrics(y, oof[r], threshold, settings)["cost_per_customer"]
        for r in range(settings.cv_repeats)
    ]
    frame = pd.DataFrame(fold_metrics)
    spec = MODEL_SPECS[key]
    return ModelResult(
        key=key,
        display_name=spec.display_name,
        cv_mean={m: float(frame[m].mean()) for m in CV_METRICS},
        cv_std={m: float(frame[m].std(ddof=1)) if len(frame) > 1 else 0.0 for m in CV_METRICS},
        cv_cost_per_customer=float(np.mean(costs)),
        oof_threshold=threshold,
        fit_time_s=float(np.mean(fit_times)),
    )


def tune_estimator(
    key: str,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    settings: Settings,
    n_iter: int,
    n_jobs: int,
) -> tuple[BaseEstimator, dict[str, Any], float]:
    """Randomised search on the *uncalibrated* pipeline, optimising PR-AUC."""
    search = RandomizedSearchCV(
        build_pipeline(clone(estimator)),
        search_space(key),
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(settings.cv_folds, shuffle=True, random_state=settings.random_state),
        random_state=settings.random_state,
        n_jobs=n_jobs,
        refit=False,
    )
    search.fit(X, y)
    params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    tuned = clone(estimator).set_params(**params)
    return tuned, params, float(search.best_score_)


def _library_versions() -> dict[str, str]:
    libs = ["scikit-learn", "xgboost", "lightgbm", "catboost", "shap", "pandas", "numpy"]
    return {lib: pkg_version(lib) for lib in libs}


def run_tournament(
    settings: Settings | None = None,
    models: list[str] | None = None,
    tune_iter: int = 0,
    write_artifacts: bool = True,
    make_figures: bool = True,
) -> TournamentResult:
    """Execute the full tournament and (optionally) persist every artifact."""
    _silence_warnings()
    s = settings or get_settings()
    keys = models or list(MODEL_SPECS)
    unknown = sorted(set(keys) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Choose from {list(MODEL_SPECS)}")

    # Single-threaded estimators, with CV folds parallelised across all CPU cores.
    fold_jobs = -1
    log.info(
        "Starting tournament",
        extra={"ctx": {"host": machine_summary(), "models": ",".join(keys),
                       "folds": s.cv_folds, "repeats": s.cv_repeats}},
    )

    df = clean_raw(load_raw(s.raw_data_path))
    if write_artifacts:
        s.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(s.processed_data_path, index=False)
    train_df, test_df = holdout_split(df, s)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    zoo = build_model_zoo(s.random_state, pos_weight, n_jobs=1)

    results: dict[str, ModelResult] = {}
    for key in keys:
        t0 = time.perf_counter()
        results[key] = cross_validate_model(key, zoo[key], X_train, y_train, s, fold_jobs)
        r = results[key]
        log.info(
            "CV %s", r.display_name,
            extra={"ctx": {"pr_auc": round(r.cv_mean["pr_auc"], 4),
                           "roc_auc": round(r.cv_mean["roc_auc"], 4),
                           "brier": round(r.cv_mean["brier"], 4),
                           "secs": round(time.perf_counter() - t0, 1)}},
        )

    champion = max(results, key=lambda k: (results[k].cv_mean["pr_auc"], -results[k].cv_cost_per_customer))
    log.info("Champion selected", extra={"ctx": {"champion": champion}})

    if tune_iter > 0:
        tuned, params, score = tune_estimator(
            champion, zoo[champion], X_train, y_train, s, tune_iter, fold_jobs
        )
        tuned_result = cross_validate_model(
            champion, tuned, X_train, y_train, s, fold_jobs
        )
        if tuned_result.cv_mean["pr_auc"] > results[champion].cv_mean["pr_auc"]:
            tuned_result.tuned_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
            results[champion] = tuned_result
            zoo[champion] = tuned
            log.info("Tuning improved champion", extra={"ctx": {"search_pr_auc": round(score, 4)}})
        else:
            log.info("Tuning did not beat defaults; keeping default hyper-parameters.")

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted: dict[str, Any] = {}
    for key in keys:
        model = calibrate(build_pipeline(clone(zoo[key])), cv=5, seed=s.random_state)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - start
        proba = model.predict_proba(X_test)[:, 1]
        r = results[key]
        r.holdout = compute_metrics(y_test, proba, r.oof_threshold, s)
        r.holdout["fit_time_full_s"] = fit_s
        r.latency_ms = measure_latency_ms(model, X_test)
        curves[r.display_name] = (y_test.to_numpy(), proba)
        fitted[key] = model

    champ = results[champion]
    trained_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "model_version": f"{__version__}+{trained_at:%Y%m%d}.{champion}",
        "package_version": __version__,
        "champion": champion,
        "champion_display_name": champ.display_name,
        "trained_at": trained_at.isoformat(),
        "decision_threshold": champ.oof_threshold,
        "threshold_strategy": "argmin cost_fn*FN + cost_fp*FP on out-of-fold calibrated probabilities",
        "cost_matrix": {"fn": s.cost_fn, "fp": s.cost_fp, "unit_usd": s.cost_unit_usd},
        "risk_tiers": {"low_max": s.tier_low_max, "medium_max": s.tier_medium_max, "high_max": s.tier_high_max},
        "calibration": "isotonic (CalibratedClassifierCV, cv=5, ensemble=False)",
        "cv": {"folds": s.cv_folds, "repeats": s.cv_repeats, "mean": champ.cv_mean, "std": champ.cv_std},
        "holdout": champ.holdout,
        "holdout_fingerprint": ids_fingerprint(test_df[ID_COLUMN]),
        "n_train": len(train_df),
        "n_holdout": len(test_df),
        "train_churn_rate": float(y_train.mean()),
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "tuned_params": champ.tuned_params,
        "random_state": s.random_state,
        "hardware": machine_info(),
        "python": platform.python_version(),
        "libraries": _library_versions(),
    }

    if write_artifacts:
        from src.models.reporting import write_benchmark_reports

        s.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[champion], s.model_path, compress=3)
        s.metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        write_benchmark_reports(results, metadata, s)
        if make_figures:
            plot_roc_pr(curves, s.figures_dir / "roc_pr_curves.png")
            from src.models.explainability import ChurnExplainer

            explainer = ChurnExplainer(fitted[champion], X_train)
            n = 150 if MODEL_SPECS[champion].family == "ensemble" else 800
            explainer.plot_summary(X_test.head(n), s.figures_dir / "shap_summary.png")
        log.info("Artifacts written", extra={"ctx": {"model": str(s.model_path)}})

    return TournamentResult(champion, results, metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the ChurnAI model tournament.")
    parser.add_argument("--models", default="all", help="Comma-separated keys or 'all'. "
                        f"Available: {','.join(MODEL_SPECS)}")
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--tune-iter", type=int, default=0, help="RandomizedSearch iterations "
                        "for the champion (0 disables tuning).")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run: 2 models, 1 repeat.")
    parser.add_argument("--no-artifacts", action="store_true", help="Do not write any files.")
    parser.add_argument("--no-figures", action="store_true", help="Skip ROC/PR and SHAP figures.")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.cv_folds:
        overrides["cv_folds"] = args.cv_folds
    if args.cv_repeats:
        overrides["cv_repeats"] = args.cv_repeats
    if args.quick:
        overrides["cv_repeats"] = 1
    settings = get_settings().model_copy(update=overrides)
    models = list(QUICK_MODELS) if args.quick else (
        None if args.models == "all" else [m.strip() for m in args.models.split(",")]
    )
    result = run_tournament(
        settings,
        models=models,
        tune_iter=args.tune_iter,
        write_artifacts=not args.no_artifacts,
        make_figures=not args.no_figures,
    )
    summary = {
        k: {"cv_pr_auc": round(r.cv_mean["pr_auc"], 4), "holdout_pr_auc": round(r.holdout["pr_auc"], 4)}
        for k, r in result.results.items()
    }
    print(json.dumps({"champion": result.champion, "models": summary}, indent=2))
    return 0


def result_to_dict(result: ModelResult) -> dict[str, Any]:
    """JSON-serialisable view of a :class:`ModelResult`."""
    return asdict(result)


if __name__ == "__main__":
    sys.exit(main())
