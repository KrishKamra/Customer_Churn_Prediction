## Summary

<!-- What does this PR change and why? Link issues with "Closes #123". -->

## Type of change

- [ ] `feat` — new functionality
- [ ] `fix` — bug fix
- [ ] `docs` — documentation only
- [ ] `refactor` / `perf` — no behaviour change
- [ ] `model` — retrained or changed the production model
- [ ] `ci` — workflows / tooling

## Validation

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy src` passes
- [ ] `uv run pytest tests/ --cov=src` passes; coverage of `src/` stays ≥ 80 %
- [ ] New / changed behaviour is covered by tests
- [ ] `uv run python -m src.models.evaluate --validate-thresholds` passes

## Model changes (if applicable)

| Metric (holdout) | Before | After |
| :--- | :---: | :---: |
| PR-AUC | | |
| ROC-AUC | | |
| Recall @ deployed threshold | | |
| Brier | | |
| Expected loss ($) | | |

- [ ] Holdout split untouched (fingerprint check passes)
- [ ] `models/`, `docs/benchmarks.md`, `docs/evaluations/model_benchmarks.json` and figures regenerated together
- [ ] `docs/model_card.md` narrative updated

## Documentation & UI

- [ ] `README.md` / `docs/` updated where behaviour or configuration changed
- [ ] UI changes follow [`docs/design_system.md`](../docs/design_system.md); screenshot attached below

## Screenshots / notes

<!-- Optional -->
