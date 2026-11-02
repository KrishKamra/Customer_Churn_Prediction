# Model Card — ChurnAI Production Churn Classifier

> Format: Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).
> Machine-readable metadata: [`models/model_metadata.json`](../models/model_metadata.json).
> Benchmarks: [`benchmarks.md`](benchmarks.md) · Statistics: [`evaluations/statistical_report.md`](evaluations/statistical_report.md)

## 1. Model details

| Field | Value |
| :--- | :--- |
| Model version | `2.0.0+20261102.elasticnet_lr` |
| Algorithm | ElasticNet-penalised logistic regression (`l1_ratio=0.5`, `C=0.5`, `saga`, `class_weight="balanced"`), wrapped in isotonic `CalibratedClassifierCV(cv=5, ensemble=False)` |
| Selected by | Six-model tournament (ElasticNet LR, Balanced RF, LightGBM, XGBoost, CatBoost, Stacking). Highest mean PR-AUC under 2× repeated stratified 5-fold CV. A 30-iteration randomised search over `C` and `l1_ratio` did not improve on the defaults, so they were kept |
| Inputs | 19 raw customer attributes (demographics, services, contract, billing), expanded to 30 pipeline features: 14 numeric (with 7 engineered and 3 smoothed risk ratios) and 16 one-hot-encoded categoricals |
| Output | Calibrated probability that the customer churns, plus a binary *intervene* decision at threshold **0.185** and a risk tier (Low < 0.20 ≤ Medium < 0.45 ≤ High < 0.70 ≤ Critical) |
| Decision rule | Threshold minimises `5·FN + 1·FP` on out-of-fold training predictions |
| Training data | 5,634 customers (80 % stratified split, seed 42) |
| Training environment | Trained and evaluated locally on CPU: ASUS Zenbook 14 (UX3405CA), Intel Core Ultra 9 285H (16 threads), 32 GB RAM, Windows 11, Python 3.12, dependencies pinned in `uv.lock`. No GPU or cloud compute was used. |
| Developed by / contact | Krish Kamra. Issues via the repository's *model regression* template |
| License | Apache-2.0 |

## 2. Intended use

**Primary use.** Rank existing subscribers of a telecom-style subscription business by
churn risk, so that a *human* retention team can prioritise outreach and choose an offer.
The SHAP explanations and counterfactual what-ifs support that conversation.

**Primary users.** Retention managers, CRM analysts, and data scientists who need a
transparent baseline.

**Out-of-scope uses**

* Automated adverse actions against customers (price increases, service downgrades, credit or
  collections decisions).
* Any decision about individuals in regulated domains (credit, insurance, employment, housing).
* Populations or products that differ materially from the training data (other countries,
  B2B accounts, prepaid plans, or after major pricing changes) without re-validation.
* Causal claims. Counterfactual prescriptions describe how the *model* would score an otherwise
  identical customer. They are not estimates of what an intervention would actually do.
  Validate offers with A/B tests.

## 3. Training data & factors

| Property | Value |
| :--- | :--- |
| Source | IBM Telco Customer Churn sample dataset (public), one snapshot |
| Size | 7,043 customers × 21 attributes. 11 blank `TotalCharges` (all `tenure = 0`) set to 0 |
| Label | `Churn` = left within the last month. Prevalence **26.54 %** (imbalance ≈ 1 : 2.8) |
| Split | 5,634 train / 1,409 holdout, stratified. The holdout is SHA-256 fingerprinted and never used for any modelling decision |

**Demographic composition.** Gender is balanced (Female 3,488 / Male 3,555). Senior citizens
are 16.2 % (1,142). 48.3 % have a partner and 30.0 % have dependents. There are no attributes
for race, ethnicity, age (beyond the senior flag), income or geography, so the model cannot
be audited on those dimensions.

**Relevant factors.** Contract type, tenure lifecycle stage, internet service type, payment
method and support add-ons dominate churn behaviour (see the statistical report). Base rates
differ sharply by subgroup, for example 41.7 % for seniors vs 23.6 % for non-seniors.

## 4. Metrics

### 4.1 Quantitative performance

| Metric | Repeated stratified 5-fold CV (train, mean ± SD) | Holdout (n = 1,409) |
| :--- | :---: | :---: |
| PR-AUC (average precision), *primary* | 0.655 ± 0.018 | **0.651** |
| ROC-AUC | 0.849 ± 0.012 | **0.848** |
| Brier score ↓ | 0.133 ± 0.005 | **0.134** |
| Recall on churners @ 0.185 | — | **86.1 %** (322 / 374) |
| Precision @ 0.185 | — | 47.7 % |
| F2 @ 0.185 | — | 0.742 |
| Expected loss @ $50 per cost unit | — | $30,650, compared with $93,471 for doing nothing and $51,756 for offering everyone |
| Median single-row latency (full pipeline) | — | ≈ 21 ms (laptop CPU, see training environment) |

CV recall and precision are reported at 0.5 in `benchmarks.md` for cross-model comparability
only. Production uses the cost-optimal threshold.

**Why these metrics.** With 26.5 % positives, accuracy rewards ignoring churners. PR-AUC
measures ranking quality where it matters, Brier measures calibration (the probabilities are
shown to users and multiplied into revenue-at-risk), and the cost matrix turns errors into
business units.

### 4.2 Decision threshold & uncertainty

Under perfect calibration the Bayes-optimal threshold for a 5:1 cost ratio is 1/6 ≈ 0.167. The
empirical out-of-fold optimum, 0.185, is close to it, which suggests the calibration is sound.
CV fold-to-fold SD is about 0.02 PR-AUC. Differences smaller than that between models, or
between retrains, should not be read as real improvements.

## 5. Ethical considerations & fairness

**Gender.** Churn is statistically independent of gender (χ² = 0.52, p = 0.470, bias-corrected
Cramér's V = 0.000). Gender is kept as an input for transparency. Its mean |SHAP| is 0.013 log-odds, ranking 25th
of 30 features, against 0.71 for the top feature. Holdout performance is similar across genders:

| Group (holdout) | n | Churn rate | Mean predicted p | Recall | Precision | FPR | ROC-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Female | 687 | 28.1 % | 26.7 % | 83.4 % | 49.7 % | 33.0 % | 0.845 |
| Male | 722 | 25.1 % | 26.8 % | 89.0 % | 45.9 % | 35.1 % | 0.852 |
| Not senior | 1,187 | 23.3 % | 23.9 % | 82.6 % | 44.9 % | 30.7 % | 0.849 |
| **Senior citizen** | 222 | 44.1 % | 41.7 % | 95.9 % | 56.3 % | **58.9 %** | **0.784** |
| No partner | 736 | 33.7 % | 33.6 % | 89.1 % | 50.3 % | 44.7 % | 0.829 |
| Partner | 673 | 18.7 % | 19.3 % | 80.2 % | 42.8 % | 24.7 % | 0.852 |
| No dependents | 978 | 31.0 % | 31.9 % | 89.1 % | 48.3 % | 42.8 % | 0.836 |
| Dependents | 431 | 16.5 % | 15.1 % | 73.2 % | 44.8 % | 17.8 % | 0.848 |

**Findings**

* **Calibration holds within groups.** Mean predicted probability tracks the observed churn
  rate within about 2.5 points for every group, so scores mean the same thing across groups.
* **Seniors are flagged far more often.** Their false-positive rate (58.9 %) is about twice
  that of non-seniors, and ranking quality is lower for them (ROC-AUC 0.784). This follows from
  their higher base rate meeting a single global threshold. It is acceptable for *offering
  benefits*, because a false positive means an unneeded retention offer. It would **not** be
  acceptable for any adverse action. If equal opportunity matters, use group-specific
  thresholds or monitor FPR parity.
* **Customers with dependents** have lower recall (73 %), so retention outreach may
  under-serve them.
* **Proxy risk.** `SeniorCitizen`, `Partner` and `Dependents` are personal-status attributes.
  They are significant (p < 1e-35) but have modest effects (V ≈ 0.15–0.16). Removing them is a
  valid policy choice. Expect a small PR-AUC cost, and re-run the tournament to measure it.

**Privacy.** The training data is public and contains pseudonymous IDs. In production,
`prediction_logs` store customer IDs, scores and SHAP summaries, and should be governed as
personal data (retention limits, access control). See `SECURITY.md` for deployment hardening.

## 6. Caveats & recommendations

* **Single snapshot, no time dimension.** The model sees state, not trajectory (usage decline,
  support tickets, price changes). The labels describe one month, and seasonality and concept
  drift are not represented. Plan migration to longitudinal data, such as the KKBox schema.
* **Within-contract ranking is weak.** On the holdout, ROC-AUC is 0.76 for month-to-month,
  0.75 for one-year and 0.72 for two-year contracts. Much of the global AUC comes from
  separating contract types. At the deployed threshold, **none of the 9 two-year-contract
  churners in the holdout are flagged**, and only 42 % of one-year churners are. Do not rely
  on the model alone for long-contract customers.
* **Counterfactuals are associational.** "Move to a two-year contract" lowers the *score*
  partly because long-contract customers are self-selected loyalists. Run controlled
  experiments before attributing uplift.
* **Cost assumptions are illustrative.** The 5:1 ratio and $50 unit are configurable
  (`CHURNAI_COST_FN`, `CHURNAI_COST_FP`, `CHURNAI_COST_UNIT_USD`). Re-derive the threshold with
  real offer costs and customer lifetime values.
* **Monitoring.** Re-run `python -m src.models.evaluate --validate-thresholds` on fresh labelled
  data at least monthly. Track the score distribution in `prediction_logs` (PSI) and per-group
  FPR. Retrain when PR-AUC falls more than 0.03 below the documented value.
* **Pickle trust boundary.** Load only artifacts produced by this repository's pipeline.
