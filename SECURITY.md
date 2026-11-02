# Security Policy

## Supported versions

| Version | Supported | Notes |
| :--- | :---: | :--- |
| 2.x (`main`) | ✅ | Active development; security fixes released as patch versions. |
| 1.x (single-notebook prototype) | ❌ | Superseded; please upgrade to 2.x. |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Report privately through either channel:

1. **GitHub private vulnerability reporting** (preferred):
   [Security → Report a vulnerability](https://github.com/KrishKamra/Customer_Churn_Prediction/security/advisories/new)
2. **Email:** `krishkamra.dev@gmail.com` with the subject `[SECURITY] ChurnAI – <short summary>`.

Please include:

* the affected component (`src/db`, `src/ui`, model artifact loading, CI, …) and version / commit SHA;
* a description of the issue and its impact;
* minimal reproduction steps or a proof of concept (no destructive payloads, please);
* any suggested remediation.

## Response targets

| Stage | Target |
| :--- | :--- |
| Acknowledgement of report | **within 48 hours** |
| Initial assessment & severity (CVSS v3.1) | within 5 business days |
| Fix or mitigation for Critical / High | within 14 days |
| Fix or mitigation for Medium / Low | next scheduled release |

We will keep you informed throughout, credit you in the release notes (unless you prefer to
remain anonymous) and coordinate a disclosure date with you. We ask that you give us a
reasonable window — **90 days by default** — before any public disclosure.

## Scope and known trust boundaries

* **Model artifacts are pickles.** `models/production_pipeline.joblib` is loaded with
  `joblib.load`, which can execute arbitrary code. Only load artifacts produced by this
  repository's own training pipeline or CI. Never point `CHURNAI_MODEL_PATH` at an
  untrusted file.
* **Database credentials** are read from `DATABASE_URL`. Never commit `.env` files; the
  repository's `.gitignore` excludes them.
* **Batch uploads** in the dashboard are parsed with pandas and validated against the
  schema in `src/features/schema.py`; they are scored, never executed.
* The Streamlit app has **no built-in authentication**. Deploy it behind an authenticating
  reverse proxy or a platform with SSO before exposing real customer data.

## Out of scope

* Vulnerabilities in third-party dependencies with no demonstrated impact on ChurnAI
  (please report them upstream; we track advisories via Dependabot).
* Findings that require a compromised developer machine or CI runner.
