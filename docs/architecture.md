# ChurnAI Architecture

> Version 2.0 · Audience: engineers extending, deploying or auditing ChurnAI.
> Diagrams follow the [C4 model](https://c4model.com) and render natively on GitHub (Mermaid).

![System architecture](figures/system_architecture.png)

## 1. Design principles

| Principle | How it shows up in the code |
| :--- | :--- |
| **One pipeline, everywhere** | Training, CV folds, the dashboard, batch scoring and the CI gate all call the *same* serialized scikit-learn object. There is no re-implemented preprocessing in the UI. |
| **Leakage-proof by construction** | Stateless features (`src/features/engineer.py`) never see the target; anything learned (medians, risk ratios, scalers, one-hot vocabularies, calibration) is fit *inside* the pipeline, so it is re-fit on every CV training fold. |
| **Honest evaluation** | A fingerprinted holdout is never used for selection, tuning, calibration or threshold choice. |
| **Decisions in business units** | The deployed threshold minimises `5·FN + 1·FP`, not accuracy. |
| **Config over constants** | Every knob lives in `src/config.py` (`pydantic-settings`, `CHURNAI_*` env vars). |
| **Swap infrastructure, not code** | SQLite ↔ PostgreSQL by changing `DATABASE_URL`. |

## 2. System context (C4 level 1)

```mermaid
C4Context
    title System context — ChurnAI
    Person(analyst, "Retention manager", "Scores customers, explores what-if retention offers")
    Person(exec, "Executive", "Reviews churn drivers and expected business loss")
    Person(mle, "ML engineer", "Retrains, benchmarks and ships models")
    System(churnai, "ChurnAI", "Churn scoring, explainability and statistical evidence platform")
    SystemDb_Ext(dw, "Customer data source", "CRM / billing export (IBM Telco schema)")
    System_Ext(gh, "GitHub Actions", "Lint, type-check, tests, model validation gate")

    Rel(analyst, churnai, "Uses", "HTTPS / Streamlit")
    Rel(exec, churnai, "Reads dashboards & reports")
    Rel(mle, churnai, "Trains via CLI / notebooks")
    Rel(dw, churnai, "CSV snapshots / batch files")
    Rel(gh, churnai, "Validates every push / PR")
```

## 3. Containers (C4 level 2)

```mermaid
C4Container
    title Containers — ChurnAI
    Person(user, "Business user")
    Person(mle, "ML engineer")

    System_Boundary(c1, "ChurnAI") {
        Container(ui, "Streamlit portal", "Python, Streamlit, Plotly", "Risk engine, customer lookup, analytics, batch scoring (src/ui)")
        Container(core, "Inference & explainability core", "scikit-learn, SHAP", "ChurnPredictor, ChurnExplainer, counterfactuals (src/models)")
        Container(train, "Training orchestrator", "Python CLI", "Model tournament, calibration, reports (src/models/train.py)")
        Container(nb, "Notebooks", "Jupyter (local)", "EDA, inferential statistics, tournament (notebooks/)")
        ContainerDb(db, "Operational database", "SQLite or PostgreSQL", "customers, prediction_logs (src/db)")
        ContainerDb(art, "Model registry (filesystem)", "joblib + JSON", "models/production_pipeline.joblib, model_metadata.json")
        ContainerDb(docs, "Evaluation store", "Markdown / JSON / PNG", "docs/benchmarks.md, docs/evaluations/*, docs/figures/*")
    }

    Rel(user, ui, "Interacts", "HTTPS")
    Rel(ui, core, "predict_one / score_frame / explain_instance")
    Rel(ui, db, "Search, upsert, audit log", "SQLAlchemy session")
    Rel(core, art, "Loads", "joblib")
    Rel(mle, train, "uv run python -m src.models.train")
    Rel(mle, nb, "Runs")
    Rel(nb, train, "run_tournament()")
    Rel(train, art, "Serializes champion")
    Rel(train, docs, "Writes benchmarks & figures")
```

## 4. Components (C4 level 3)

| Package | Module | Responsibility |
| :--- | :--- | :--- |
| `src.config` | `config.py` | Typed `Settings` (paths, DB URL, seeds, CV, cost matrix, risk tiers, CI gates). |
| `src.core` | `logger.py` | Structured logging (console or one-line JSON via `CHURNAI_LOG_FORMAT=json`). |
| | `exceptions.py` | `ChurnAIError` → `DataValidationError`, `ModelNotFoundError`, `ModelValidationError`, `DatabaseError`. |
| | `hardware.py` | Host fingerprint (CPU, threads, RAM, OS) recorded with every training run. |
| `src.features` | `schema.py` | Column contract, categorical vocabularies, Pydantic `CustomerFeatures`. |
| | `engineer.py` | Ingestion, cleaning, stateless engineered features, tenure cohorts. |
| | `transformers.py` | `FeatureEngineer` and target-aware `RiskRatioEncoder` (sklearn API). |
| `src.models` | `pipeline.py` | Shared preprocessing, model zoo, calibration wrapper, search spaces. |
| | `train.py` | Tournament orchestration + artifact writing (CLI). |
| | `evaluate.py` | Metrics, cost-optimal threshold, latency, ROC/PR plots, **CI validation gate** (CLI). |
| | `inference.py` | `ChurnPredictor`, `PredictionResult`, risk tiers. |
| | `explainability.py` | `ChurnExplainer` (Tree/Linear/Permutation SHAP), counterfactual prescriptions. |
| | `reporting.py` | Renders `docs/benchmarks.md` and `model_benchmarks.json`. |
| `src.analytics` | `statistics.py` | Welch / Mann-Whitney / χ² / Cramér's V / cohorts + Markdown report. |
| | `plotting.py` | Matplotlib theme aligned with the design system. |
| `src.db` | `models.py` | `Customer`, `PredictionLog` (SQLModel). |
| | `database.py` | Engine factory, SQLite pragmas, pooled PostgreSQL, `session_scope`, `scoped_session`. |
| | `repository.py` | Idempotent seeding, autocomplete search, audit queries. |
| `src.ui` | `styles.py` / `components.py` / `app.py` | Design tokens & CSS motion, glass components, four-page portal. |

## 5. The model pipeline

```mermaid
flowchart LR
    A[Raw frame<br/>19 canonical columns<br/>+ any extras] --> B[FeatureEngineer<br/>select columns, drop extras<br/>+7 numeric, +1 categorical]
    B --> C[RiskRatioEncoder<br/>smoothed churn-rate ratios<br/>3 categorical intersections]
    C --> D{ColumnTransformer}
    D -->|numeric ×14| E[SimpleImputer median<br/>→ StandardScaler]
    D -->|categorical ×16| F[SimpleImputer mode<br/>→ OneHotEncoder<br/>handle_unknown=ignore]
    E --> G[Estimator<br/>champion of the tournament]
    F --> G
    G --> H[CalibratedClassifierCV<br/>isotonic, cv=5, ensemble=False]
    H --> I[p churn calibrated]
    I --> J{p ≥ threshold?<br/>argmin 5·FN + 1·FP}
    J --> K[Label + risk tier<br/>Low / Medium / High / Critical]
```

`ensemble=False` fits a single pipeline on all training data and learns the isotonic map from
out-of-fold predictions. This keeps exactly one inner model, which SHAP can explain
directly.

## 6. End-to-end data flow

How one customer payload travels through the system, from the UI or a batch upload to the
audit log:

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as Streamlit (src/ui/app.py)
    participant V as Validation (schema.py)
    participant P as ChurnPredictor
    participant PL as Pipeline (joblib)
    participant X as ChurnExplainer
    participant R as CustomerRepository
    participant DB as SQLite / PostgreSQL

    U->>UI: edit inputs / upload CSV / search customer ID
    UI->>R: search_customer_ids(prefix) (lookup page)
    R->>DB: SELECT … WHERE upper(customer_id) LIKE 'PREFIX%'
    UI->>V: CustomerFeatures.model_validate(payload)
    V-->>UI: typed payload or DataValidationError
    UI->>P: predict_one(payload) / score_frame(df)
    P->>PL: predict_proba(X[FEATURE_COLUMNS])
    PL-->>P: calibrated probability
    P-->>UI: PredictionResult(prob, label, tier, threshold, version)
    UI->>X: explain_instance(row)
    X-->>UI: aggregated SHAP contributions
    UI->>P: counterfactual what-ifs (vectorised predict_proba)
    U->>UI: "Log to audit trail" / "Write to database"
    UI->>R: log_prediction(...) / log_predictions_bulk(...)
    R->>DB: INSERT prediction_logs (one transaction)
    DB-->>UI: committed → toast + Plotly timeline refresh
```

* **Ingestion:** the UI form, batch CSV upload (`pd.read_csv` → `clean_raw`) or a DB row
  (`Customer.to_feature_dict`). Today there is no HTTP API. `ChurnPredictor` is the seam where a
  FastAPI service would plug in, using the same `CustomerFeatures` schema as its request model.
* **Validation:** Pydantic enforces types, ranges and categorical vocabularies for single
  payloads. Batch frames are checked for required columns (`validate_feature_frame`).
* **Inference:** one `predict_proba` call on the calibrated pipeline. Rows are scored in
  chunks in the UI so the progress bar stays live.
* **Persistence:** each write happens in `session_scope`, which commits on success and rolls
  back on failure, wrapping the error as `DatabaseError`.
* **UI state:** Streamlit caches the predictor, explainer, dataset and repository with
  `st.cache_resource` / `st.cache_data`, so each rerun only does inference.

## 7. Training & release flow

```mermaid
flowchart TB
    raw[(data/raw/Telco_Customer_Churn.csv)] --> clean[clean_raw<br/>fix 11 blank TotalCharges]
    clean --> pq[(data/processed/churn_clean.parquet)]
    clean --> split{Stratified 80/20<br/>seed 42}
    split -->|5,634| cv[Repeated Stratified 5-Fold CV<br/>× 6 contenders<br/>pipeline + calibration per fold]
    split -->|1,409 holdout<br/>SHA-256 fingerprint| hold[Holdout scoring]
    cv --> champ[Champion = max mean PR-AUC]
    champ --> tune[RandomizedSearchCV<br/>30 iterations, local CPU]
    tune --> thr[Threshold = argmin cost<br/>on out-of-fold probabilities]
    thr --> fit[Refit calibrated champion<br/>on full training split]
    fit --> hold
    fit --> art[(models/production_pipeline.joblib<br/>models/model_metadata.json)]
    hold --> rep[(docs/benchmarks.md<br/>docs/evaluations/model_benchmarks.json<br/>docs/figures/*.png)]
    art --> ci[CI: validate-thresholds<br/>ROC-AUC ≥ .82 · PR-AUC ≥ .60<br/>Recall ≥ .75 · Brier ≤ .16]
```

## 8. Persistence model

```mermaid
erDiagram
    customers ||--o{ prediction_logs : "is scored by"
    customers {
        string customer_id PK
        string gender
        int senior_citizen
        int tenure "indexed"
        string contract_type "indexed"
        string internet_service
        string payment_method
        float monthly_charges
        float total_charges
        int churned "nullable ground truth"
        datetime created_at
    }
    prediction_logs {
        int id PK
        string customer_id FK "nullable for ad-hoc what-ifs"
        string model_version
        float churn_probability
        int predicted_label
        string risk_tier "indexed"
        string source "ui | batch | api"
        json shap_summary
        datetime created_at "indexed"
    }
```

| Concern | SQLite (default) | PostgreSQL (production) |
| :--- | :--- | :--- |
| URL | `sqlite:///data/churn.db` | `postgresql+psycopg://user:pass@host/db` (`postgres://` is auto-normalised) |
| Concurrency | `check_same_thread=False`, WAL journal | pooled (`pool_size=5`, `max_overflow=10`) |
| Integrity | `PRAGMA foreign_keys=ON` | native FK constraints |
| Resilience | — | `pool_pre_ping=True`, `pool_recycle=1800` |
| Install | built in | `uv sync --extra postgres` |

## 9. Failure modes & defences

| Failure mode | Where it is caught | Behaviour |
| :--- | :--- | :--- |
| **Missing columns** in a batch file | `validate_feature_frame` (UI and `FeatureEngineer`) | `DataValidationError` listing every missing column; nothing is scored. |
| **Extra columns** (`customerID`, `Churn`, …) | `FeatureEngineer.transform` | Silently dropped, which also blocks target leakage at inference. |
| **Unseen categorical level** (`PaymentMethod="Crypto"`) | `OneHotEncoder(handle_unknown="ignore")`, `RiskRatioEncoder` | All-zero one-hot vector and a neutral risk ratio of 1.0. Prediction still succeeds (tested). |
| **Out-of-range / mistyped single payload** | `CustomerFeatures` (Pydantic) | `DataValidationError` with field-level messages. |
| **Blank / non-numeric charges** | `clean_raw` coercion + median imputer | Zero for unbilled customers, otherwise median-imputed. |
| **Schema drift in the training snapshot** | `load_raw` column assertion, holdout fingerprint in the CI gate | Training refuses to start; CI fails on a changed holdout. |
| **Model regression** | `python -m src.models.evaluate --validate-thresholds` | Non-zero exit blocks the merge. |
| **Missing model artifact** | `ChurnPredictor.load` | `ModelNotFoundError` with the retrain command; the analytics page still works. |
| **Database connection dropout** | `pool_pre_ping`, `session_scope` | Stale connections are recycled transparently; a failed transaction rolls back and surfaces a `DatabaseError` in the UI instead of crashing. |
| **FK violation** (logging an unknown customer) | DB constraint → `DatabaseError` | Ad-hoc what-ifs are logged with `customer_id = NULL` by design. |

## 10. Deployment topologies

| Topology | Command / notes |
| :--- | :--- |
| Local | `uv sync && uv run streamlit run src/ui/app.py` (SQLite auto-created and seeded). |
| Streamlit Community Cloud | Entry point `src/ui/app.py`; the app bootstraps `sys.path`. Use `uv export --no-hashes > requirements.txt` if the platform needs a requirements file. |
| Container / PaaS | Set `DATABASE_URL` to managed PostgreSQL and `CHURNAI_LOG_FORMAT=json`. Put the app behind an authenticating proxy (see `SECURITY.md`). |
| Training | Notebook 03 or `uv run python -m src.models.train --tune-iter 30` on any CPU. All published artifacts were trained locally on an Intel Core Ultra 9 285H (16 threads, 32 GB RAM) in about ten minutes, tuning included. |

## 11. Evolution path

* **Serving API:** a thin FastAPI layer over `ChurnPredictor` using `CustomerFeatures` as the
  request schema, with `source="api"` in `prediction_logs`.
* **Temporal data:** migrate to the KKBox churn dataset (2.8M users, longitudinal logs).
  Feature generation would move to windowed aggregations, and the `customers` table would gain
  a snapshot date.
* **Drift monitoring:** compare `prediction_logs` score distributions (PSI) and per-feature
  distributions against the training snapshot on a schedule.
