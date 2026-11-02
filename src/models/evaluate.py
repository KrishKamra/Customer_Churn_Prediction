"""Cost-sensitive and probabilistic evaluation, plus the CI model-validation gate.

Run the gate with::

    uv run python -m src.models.evaluate --validate-thresholds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import Settings, get_settings
from src.core.exceptions import ModelValidationError
from src.core.logger import get_logger
from src.features.schema import ID_COLUMN, TARGET_COLUMN

log = get_logger("evaluate")


def holdout_split(
    df: pd.DataFrame, settings: Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified train/holdout split shared by training and the CI gate."""
    s = settings or get_settings()
    train, test = train_test_split(
        df, test_size=s.test_size, stratify=df[TARGET_COLUMN], random_state=s.random_state
    )
    return train, test


def ids_fingerprint(ids: pd.Series) -> str:
    """Order-independent SHA-256 of customer IDs, used to prove the holdout is unchanged."""
    joined = "\n".join(sorted(ids.astype(str)))
    return hashlib.sha256(joined.encode()).hexdigest()


def business_cost(tn_fp_fn_tp: tuple[int, int, int, int], cost_fn: float, cost_fp: float) -> float:
    """``cost_fn * FN + cost_fp * FP`` in abstract cost units."""
    _, fp, fn, _ = tn_fp_fn_tp
    return cost_fn * fn + cost_fp * fp


def optimal_threshold(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    grid: np.ndarray | None = None,
) -> float:
    """Return the decision threshold minimising expected business cost.

    For perfectly calibrated probabilities the Bayes-optimal threshold is
    ``cost_fp / (cost_fp + cost_fn)`` (= 1/6 for a 5:1 matrix); the empirical scan corrects
    for residual miscalibration.
    """
    y = np.asarray(y_true)
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs = []
    for t in grid:
        pred = (proba >= t).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        costs.append(cost_fn * fn + cost_fp * fp)
    return float(grid[int(np.argmin(costs))])


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    threshold: float = 0.5,
    settings: Settings | None = None,
) -> dict[str, float]:
    """Full evaluation hierarchy for one set of probabilistic predictions."""
    s = settings or get_settings()
    y = np.asarray(y_true).astype(int)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    cost = business_cost((tn, fp, fn, tp), s.cost_fn, s.cost_fp)
    return {
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "cost_units": float(cost),
        "cost_per_customer": float(cost / len(y)),
        "expected_loss_usd": float(cost * s.cost_unit_usd),
    }


def measure_latency_ms(model: Any, X: pd.DataFrame, n_rows: int = 100) -> float:
    """Median single-row ``predict_proba`` latency in milliseconds."""
    sample = X.head(n_rows)
    model.predict_proba(sample.iloc[[0]])  # warm-up
    timings = []
    for i in range(len(sample)):
        start = time.perf_counter()
        model.predict_proba(sample.iloc[[i]])
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.median(timings))


def plot_roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    """Save side-by-side ROC and Precision-Recall curves for several models."""
    from matplotlib.figure import Figure
    from sklearn.metrics import precision_recall_curve, roc_curve

    from src.analytics.plotting import PALETTE, apply_theme

    apply_theme()
    # Backend-free Figure: safe on headless CI and does not hijack a Jupyter backend.
    fig = Figure(figsize=(13, 5.5), dpi=150)
    ax_roc, ax_pr = fig.subplots(1, 2)
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        ax_pr.plot(rec, prec, lw=2, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    base_rate = float(np.mean(next(iter(curves.values()))[0]))
    ax_roc.plot([0, 1], [0, 1], ls="--", c=PALETTE["muted"], lw=1)
    ax_pr.axhline(base_rate, ls="--", c=PALETTE["muted"], lw=1, label=f"Base rate ({base_rate:.2f})")
    for ax, title, xl, yl in (
        (ax_roc, "ROC curve (holdout)", "False positive rate", "True positive rate"),
        (ax_pr, "Precision-Recall curve (holdout)", "Recall", "Precision"),
    ):
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8, loc="lower right" if ax is ax_roc else "upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------------------
# Model validation gate
# --------------------------------------------------------------------------------------


def validate_thresholds(settings: Settings | None = None) -> dict[str, float]:
    """Re-score the committed production pipeline on the holdout and enforce minimums.

    Raises:
        ModelValidationError: if any metric violates its gate or the holdout drifted.
    """
    from src.features.engineer import clean_raw, load_raw, split_features_target
    from src.models.inference import ChurnPredictor

    s = settings or get_settings()
    predictor = ChurnPredictor.load(s.model_path, s.metadata_path)
    _, holdout = holdout_split(clean_raw(load_raw(s.raw_data_path)), s)

    expected_fp = predictor.metadata.get("holdout_fingerprint")
    if expected_fp and expected_fp != ids_fingerprint(holdout[ID_COLUMN]):
        raise ModelValidationError("Holdout split fingerprint mismatch: data or seed changed.")

    X_test, y_test = split_features_target(holdout)
    proba = predictor.predict_proba(X_test)
    metrics = compute_metrics(y_test, proba, predictor.threshold, s)

    gates = {
        "roc_auc": (metrics["roc_auc"], s.gate_min_roc_auc, ">="),
        "pr_auc": (metrics["pr_auc"], s.gate_min_pr_auc, ">="),
        "recall": (metrics["recall"], s.gate_min_recall, ">="),
        "brier": (metrics["brier"], s.gate_max_brier, "<="),
    }
    failures = []
    for name, (value, bound, op) in gates.items():
        ok = value >= bound if op == ">=" else value <= bound
        log.info(
            "gate %s", name, extra={"ctx": {"value": round(value, 4), "bound": f"{op}{bound}"}}
        )
        if not ok:
            failures.append(f"{name}={value:.4f} (required {op} {bound})")
    if failures:
        raise ModelValidationError("Model validation gate failed: " + "; ".join(failures))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChurnAI model evaluation utilities.")
    parser.add_argument(
        "--validate-thresholds",
        action="store_true",
        help="Score the production pipeline on the holdout and enforce quality gates.",
    )
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    args = parser.parse_args(argv)
    if not args.validate_thresholds:
        parser.print_help()
        return 0
    try:
        metrics = validate_thresholds()
    except ModelValidationError as exc:
        log.error(str(exc))
        return 1
    if args.json:
        print(json.dumps(metrics, indent=2))
    log.info("Model validation gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
