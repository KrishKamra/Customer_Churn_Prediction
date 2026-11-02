<div align="center">

# 📡 ChurnAI

### Production-grade customer churn prediction & statistical intelligence

[![CI](https://github.com/KrishKamra/Customer_Churn_Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/KrishKamra/Customer_Churn_Prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/packaging-uv-DE5FE9)
![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64)
![mypy](https://img.shields.io/badge/types-mypy-2A6DB2)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**Calibrated churn scores · SHAP explanations · counterfactual retention offers · statistical proof**

[Quick start](#-quick-start) · [Results](#-results) · [Architecture](docs/architecture.md) · [Model card](docs/model_card.md) · [Benchmarks](docs/benchmarks.md) · [Design system](docs/design_system.md)

</div>

---

## ✨ What is ChurnAI?

ChurnAI turns the IBM Telco customer base (7,043 subscribers) into ranked, explained and
costed retention decisions:

* **🔍 Real-time risk engine.** Edit any attribute and get back a *calibrated* churn
  probability, a local SHAP waterfall explaining it, and counterfactual prescriptions such as
  *"Migrate to a two-year contract reduces churn risk from 80 % to 23 %"*. A timestamped
  executive report downloads with one click.
* **🗄️ Live customer database.** ID autocomplete over SQLite (zero setup) or PostgreSQL
  (`DATABASE_URL`), one-click score-and-log, and a full prediction audit timeline per customer.
* **📊 Executive analytics with statistical proof.** Cohort dashboards including the 0–6 month
  danger zone, backed by Welch's t-tests, Mann-Whitney U, χ² tests and bias-corrected
  Cramér's V, all computed live.
* **📂 Batch inference.** Drag-and-drop a CSV. Rows are schema-validated, scored in chunks,
  ranked by risk, exported, and optionally written to the audit trail.

Underneath sits a reproducible ML platform: a six-model tournament under repeated stratified
CV, isotonic calibration, a threshold chosen to minimise business cost, a fingerprinted
holdout, and a CI gate that blocks any model regression.

![System architecture](docs/figures/system_architecture.png)

## 🏆 Results

Holdout set of 1,409 customers, never used for any modelling decision. Full tables are in
[`docs/benchmarks.md`](docs/benchmarks.md).

| Model | CV PR-AUC (2×5-fold) | Holdout ROC-AUC | Holdout PR-AUC | Recall @ cost-opt. threshold | Brier ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ElasticNet Logistic Regression** 🏆 | **0.655 ± 0.018** | **0.848** | **0.651** | 86.1 % | **0.134** |
| Stacking (CatBoost + XGBoost + LR) | 0.652 | 0.849 | 0.647 | 86.6 % | 0.134 |
| XGBoost | 0.649 | 0.843 | 0.637 | 90.6 % | 0.136 |
| CatBoost | 0.636 | 0.842 | 0.643 | 92.0 % | 0.137 |
| Balanced Random Forest | 0.637 | 0.841 | 0.632 | 89.0 % | 0.138 |
| LightGBM | 0.636 | 0.838 | 0.631 | 87.4 % | 0.138 |

**Takeaways**

* The well-regularised linear model **wins**. On this dataset the non-leaky ceiling is about
  ROC-AUC 0.85, and every contender lands within roughly one CV standard deviation of it. The
  tournament shows that more complexity buys nothing here, and the champion is the most
  interpretable and fastest option. A 30-iteration randomised hyper-parameter search did not
  beat the regularised defaults either, so the defaults ship.
* The deployed threshold is **0.185**, not 0.5, because a missed churner is assumed to cost 5×
  a wasted retention offer. At that threshold the champion catches **322 of 374** holdout
  churners. Expected loss falls from **$93k** (do nothing) and **$52k** (offer everyone) to
  **$31k**, using an illustrative $50 per offer ([assumptions](docs/benchmarks.md#protocol)).
* `Contract` is the strongest driver (Cramér's V = 0.41). Customers in their first six months
  churn at **53 %**. `gender` is statistically independent of churn (χ² p = 0.47).

<p align="center">
  <img src="docs/figures/roc_pr_curves.png" width="88%" alt="ROC and PR curves"/>
</p>
<p align="center">
  <img src="docs/figures/danger_zone_cohorts.png" width="49%" alt="Danger zone"/>
  <img src="docs/figures/shap_summary.png" width="42%" alt="SHAP summary"/>
</p>

## 🚀 Quick start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs the right
Python for you).

```bash
git clone https://github.com/KrishKamra/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
uv sync --all-extras                      # reproducible env from uv.lock
uv run streamlit run src/ui/app.py        # → http://localhost:8501
```

The trained production model is committed, so the app works immediately. On first start it
creates `data/churn.db` and seeds it with all 7,043 customers.

### Common tasks

| Task | Command |
| :--- | :--- |
| Run the full tournament (~5 min on a laptop CPU) | `uv run python -m src.models.train` |
| Tournament + champion hyper-parameter search | `uv run python -m src.models.train --tune-iter 30` |
| Fast smoke run | `uv run python -m src.models.train --quick` |
| Re-execute the notebooks | `uv run jupyter nbconvert --execute --inplace notebooks/*.ipynb` |
| CI model gate | `uv run python -m src.models.evaluate --validate-thresholds` |
| Tests + coverage | `uv run pytest --cov=src` |
| Lint / format / types | `uv run ruff check . && uv run ruff format --check . && uv run mypy src` |
| Use PostgreSQL | `uv sync --extra postgres` then `DATABASE_URL=postgresql://user:pass@host/db` |
| Re-render the architecture figure | `uv run python scripts/render_architecture.py` |

All settings, including seeds, CV folds, cost matrix, risk-tier cut-offs and gate thresholds,
can be overridden with `CHURNAI_*` environment variables or a `.env` file. See
[`src/config.py`](src/config.py).

## 📓 Notebooks

| Notebook | Purpose | Local run time |
| :--- | :--- | :---: |
| [01 · Exploratory data analysis](notebooks/01_exploratory_data_analysis.ipynb) | Imbalance, skew, collinearity, danger-zone cohorts → parquet + figures | ~15 s |
| [02 · Inferential statistics](notebooks/02_inferential_statistics.ipynb) | Welch, Mann-Whitney, 16× χ² + Cramér's V → `statistical_report.md` | ~10 s |
| [03 · Model tournament & tuning](notebooks/03_model_benchmarking_tuning.ipynb) | 6-model tournament, 30-iteration tuning, calibration, cost curve, SHAP, CI gate | ~10 min |

The notebooks are committed with their outputs and write their artifacts straight into the
repository (`data/processed/`, `docs/`, `models/`).

### 💻 Reproducibility: everything ran locally

Every number, figure and model in this repository was produced on one laptop CPU. No cloud
compute or GPU was used, and the host is recorded in `models/model_metadata.json` with each
training run.

| Component | Spec |
| :--- | :--- |
| Machine | ASUS Zenbook 14 (UX3405CA) |
| CPU | Intel Core Ultra 9 285H, 16 cores / 16 threads |
| Memory | 32 GB |
| OS | Windows 11 (build 26300) |
| Python | 3.12 via `uv`, dependencies pinned in `uv.lock` |

Cross-validation folds run in parallel across all cores. On 7,043 rows a GPU would add
nothing, so the training code is CPU-only and runs the same way on any machine.

## 🧱 Project layout

```text
├── .github/              CI workflow, Dependabot, issue forms, PR template
├── data/
│   ├── raw/              Telco_Customer_Churn.csv (immutable snapshot)
│   └── processed/        churn_clean.parquet (versioned)
├── docs/                 architecture · design system · model card · benchmarks · figures · evaluations
├── models/               production_pipeline.joblib + model_metadata.json
├── notebooks/            01 EDA · 02 inferential statistics · 03 tournament
├── scripts/              render_architecture.py
├── src/
│   ├── config.py         pydantic-settings configuration
│   ├── core/             logging · exceptions · host hardware fingerprint
│   ├── features/         schema contract · stateless features · custom transformers
│   ├── models/           pipeline · train · evaluate · inference · explainability · reporting
│   ├── analytics/        inferential statistics · figure theme
│   ├── db/               SQLModel schema · engine/session factory · repository
│   └── ui/               Streamlit portal · glass components · CSS motion system
└── tests/                data · pipeline/leakage · db · inference · analytics · training/gate
```

## 🧪 Engineering quality

* **Leakage-proof.** Every learned transform, including a target-derived risk-ratio encoder, is
  re-fit inside each CV fold. Tests cover target independence of the stateless features and
  fold-only fitting.
* **Honest evaluation.** The holdout is SHA-256 fingerprinted, and CI fails if it changes.
* **Typed and linted.** `mypy` is clean on `src/`, and `ruff` enforces rules `E F I N W UP`.
* **Tested.** 60+ tests and over 80 % coverage of non-UI code, enforced in CI on Python 3.11 and 3.12.
* **Reproducible.** Seeds are fixed, dependencies are locked, and every training run records
  its library versions and host hardware in `models/model_metadata.json`.

## 📚 Documentation

| Document | Contents |
| :--- | :--- |
| [Architecture](docs/architecture.md) | C4 context/container diagrams, pipeline, data flow, ERD, failure modes |
| [Design system](docs/design_system.md) | Tokens, glassmorphism spec, motion and keyframes, component rules |
| [Model card](docs/model_card.md) | Intended use, metrics, fairness analysis, limitations |
| [Benchmarks](docs/benchmarks.md) | Cross-model scorecards, latency, cost analysis (auto-generated) |
| [Statistical report](docs/evaluations/statistical_report.md) | Hypothesis tests and effect sizes (auto-generated) |

## 🤝 Contributing & security

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), which covers the uv
setup, branch naming, pre-commit checks and the PR checklist, and the
[Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities privately as described in
[SECURITY.md](SECURITY.md).

## 👤 Author

**Krish Kamra**: [GitHub @KrishKamra](https://github.com/KrishKamra)

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE). The IBM Telco Customer Churn dataset is
distributed by IBM as a public sample dataset.
