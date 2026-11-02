# Contributing to ChurnAI

Thanks for helping make ChurnAI better! This guide gets you from clone to merged PR.
By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## 1. Bootstrap a development environment

ChurnAI uses [**uv**](https://docs.astral.sh/uv/) for Python, dependency and lockfile
management. Install it once:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then:

```bash
git clone https://github.com/KrishKamra/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
uv sync --all-extras          # creates .venv from uv.lock (Python pinned by .python-version)
uv run pytest -q              # sanity check — should be all green
uv run streamlit run src/ui/app.py
```

| Extra | Installs | When you need it |
| :--- | :--- | :--- |
| `dev` | ruff, mypy, pytest, pytest-cov | always |
| `notebooks` | seaborn, ipykernel, nbconvert | editing / executing `notebooks/` |
| `postgres` | psycopg 3 | running against PostgreSQL (`DATABASE_URL=postgresql://…`) |

Adding a dependency: `uv add <pkg>` (or `uv add --optional dev <pkg>`) and commit both
`pyproject.toml` **and** `uv.lock`.

## 2. Branching convention

Branch from `main` using a typed prefix:

| Prefix | Use for | Example |
| :--- | :--- | :--- |
| `feat/` | new user-facing functionality | `feat/kkbox-schema-migration` |
| `fix/` | bug fixes | `fix/batch-upload-unicode` |
| `docs/` | documentation only | `docs/model-card-fairness` |
| `refactor/` | behaviour-preserving restructuring | `refactor/explainer-facade` |
| `perf/` | performance work | `perf/vectorised-counterfactuals` |
| `ci/` | workflows & tooling | `ci/python-3.13-matrix` |

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat(ui): add cohort heatmap`). Keep PRs focused — one logical change each.

## 3. Pre-commit verification

Run exactly what CI runs before you push:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest tests/ --cov=src --cov-report=term-missing
uv run python -m src.models.evaluate --validate-thresholds
```

`uv run ruff format .` and `uv run ruff check --fix .` fix most findings automatically.

## 4. Changing the model

* Never tune on, threshold on, or otherwise peek at the **holdout** split. The CI gate
  verifies its SHA-256 fingerprint.
* Retrain with `uv run python -m src.models.train --tune-iter 30` or by executing notebook 03.
  Both run on CPU in a few minutes. This regenerates `models/`, `docs/benchmarks.md`,
  `docs/evaluations/model_benchmarks.json` and figures — commit all of them together.
* If metrics move, update the narrative in `docs/model_card.md` and explain the change in
  the PR. A drop below the gates in `src/config.py` (`gate_*`) fails CI by design.
* Fill in the **model regression** issue template if you observe drift or a discrepancy.

## 5. Pull-request checklist

* [ ] Branch name and commit messages follow the conventions above.
* [ ] `ruff`, `mypy` and `pytest` pass locally; coverage of `src/` (excluding `src/ui`)
      stays **≥ 80 %**.
* [ ] New public functions/classes have type hints and Google-style docstrings.
* [ ] New behaviour is covered by tests (`tests/test_*.py`); bug fixes include a
      regression test.
* [ ] Docs updated (`README.md`, `docs/*`) when behaviour, configuration or metrics change.
* [ ] UI changes include a screenshot or short clip in the PR description and respect the
      tokens in [`docs/design_system.md`](docs/design_system.md).
* [ ] No secrets, customer data or large binaries (other than the versioned model) added.

## 6. Code style

* Python ≥ 3.11, line length 100, `ruff` rules `E F I N W UP`.
* Type everything in `src/`; `mypy src` must be clean.
* Logging via `src.core.logger.get_logger`, never `print` in library code.
* Raise subclasses of `src.core.exceptions.ChurnAIError` for domain errors.
* Configuration belongs in `src/config.py` (overridable via `CHURNAI_*` env vars), not in
  literals scattered through the code.

## 7. Reporting bugs & requesting features

Use the [issue templates](https://github.com/KrishKamra/Customer_Churn_Prediction/issues/new/choose).
For security issues, follow [SECURITY.md](SECURITY.md) instead.
