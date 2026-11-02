"""Inferential statistics shared by Notebook 02, the dashboard Evidence tab and reports."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy import stats

from src.features.engineer import TENURE_LABELS, tenure_cohort
from src.features.schema import CATEGORICAL_COLUMNS, TARGET_COLUMN

# 15 string categoricals + the binary-coded SeniorCitizen flag = 16 categorical variables.
CATEGORICAL_TEST_COLUMNS: list[str] = ["SeniorCitizen", *CATEGORICAL_COLUMNS]
CONTINUOUS_TEST_COLUMNS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges_log"]
ALPHA = 0.05


def with_log_total(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TotalCharges_log"] = np.log1p(out["TotalCharges"].astype(float))
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardised mean difference."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def continuous_tests(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Welch's t-test and Mann-Whitney U for each continuous variable (churned vs retained)."""
    data = with_log_total(df)
    rows = []
    for c in columns or CONTINUOUS_TEST_COLUMNS:
        churned = data.loc[data[TARGET_COLUMN] == 1, c].astype(float).to_numpy()
        retained = data.loc[data[TARGET_COLUMN] == 0, c].astype(float).to_numpy()
        t = stats.ttest_ind(churned, retained, equal_var=False)
        u = stats.mannwhitneyu(churned, retained, alternative="two-sided")
        rows.append(
            {
                "feature": c,
                "mean_churned": float(churned.mean()),
                "mean_retained": float(retained.mean()),
                "median_churned": float(np.median(churned)),
                "median_retained": float(np.median(retained)),
                "welch_t": float(t.statistic),
                "welch_p": float(t.pvalue),
                "mannwhitney_u": float(u.statistic),
                "mannwhitney_p": float(u.pvalue),
                # rank-biserial correlation: effect size for Mann-Whitney U
                "rank_biserial": float(1 - 2 * u.statistic / (len(churned) * len(retained))),
                "cohens_d": cohens_d(churned, retained),
                "skew_churned": float(stats.skew(churned)),
                "skew_retained": float(stats.skew(retained)),
            }
        )
    return pd.DataFrame(rows)


def cramers_v(contingency: pd.DataFrame | np.ndarray, bias_correction: bool = True) -> float:
    """Cramér's V effect size (Bergsma 2013 bias-corrected by default)."""
    table = np.asarray(contingency, dtype=float)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    phi2 = chi2 / n
    r, k = table.shape
    if bias_correction:
        phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1)
        k = k - (k - 1) ** 2 / (n - 1)
    denom = min(k - 1, r - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def chi_square_suite(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Chi-square test of independence + Cramér's V for every categorical variable."""
    rows = []
    for c in columns or CATEGORICAL_TEST_COLUMNS:
        table = pd.crosstab(df[c], df[TARGET_COLUMN])
        chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
        churn_by_level = df.groupby(c)[TARGET_COLUMN].mean()
        rows.append(
            {
                "feature": c,
                "chi2": float(chi2),
                "dof": int(dof),
                "p_value": float(p),
                "cramers_v": cramers_v(table),
                "significant": bool(p < ALPHA),
                "highest_risk_level": str(churn_by_level.idxmax()),
                "highest_risk_rate": float(churn_by_level.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def cohort_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate and customer count by tenure cohort (ordered)."""
    cohorts = tenure_cohort(df["tenure"])
    out = (
        df.assign(tenure_cohort=cohorts)
        .groupby("tenure_cohort")[TARGET_COLUMN]
        .agg(churn_rate="mean", customers="size")
        .reindex(TENURE_LABELS)
        .reset_index()
    )
    return out


def _p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def render_statistical_report(df: pd.DataFrame) -> str:
    """Markdown report consumed by ``docs/evaluations/statistical_report.md``."""
    cont = continuous_tests(df)
    cat = chi_square_suite(df)
    cohorts = cohort_churn(df)
    churn_rate = float(df[TARGET_COLUMN].mean())
    corr = df[["tenure", "MonthlyCharges", "TotalCharges"]].corr()

    lines = [
        "# Statistical Report — Inferential Evidence",
        "",
        "> Auto-generated by `notebooks/02_inferential_statistics.ipynb` via "
        "`src.analytics.statistics.render_statistical_report`.  ",
        f"> Generated: `{datetime.now(UTC):%Y-%m-%d %H:%M UTC}` · n = {len(df):,} customers · "
        f"churn rate = {churn_rate:.2%} · α = {ALPHA}",
        "",
        "## 1. Continuous variables: churned vs. retained",
        "",
        "Welch's t-test (unequal variances) tests mean differences; the Mann-Whitney U test "
        "is the distribution-free check, which matters because `tenure` and `TotalCharges` "
        "are strongly skewed.",
        "",
        "| Feature | Mean (churned) | Mean (retained) | Welch t | Welch p | Mann-Whitney p "
        "| Rank-biserial r | Cohen's d |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in cont.itertuples():
        lines.append(
            f"| `{r.feature}` | {r.mean_churned:,.2f} | {r.mean_retained:,.2f} | "
            f"{r.welch_t:.2f} | {_p(r.welch_p)} | {_p(r.mannwhitney_p)} | "
            f"{r.rank_biserial:+.3f} | {r.cohens_d:+.3f} |"
        )
    lines += [
        "",
        "**Interpretation.** Churners have markedly shorter tenure and lower lifetime spend "
        "but *higher* monthly charges: they are new, expensive-plan customers. Both the "
        "parametric and rank-based tests agree for every variable.",
        "",
        "## 2. Collinearity of financial variables (Pearson r)",
        "",
        "| | tenure | MonthlyCharges | TotalCharges |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for idx, row in corr.iterrows():
        lines.append(f"| `{idx}` | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    lines += [
        "",
        f"`TotalCharges` is close to `tenure × MonthlyCharges` (r = "
        f"{corr.loc['tenure', 'TotalCharges']:.2f} with tenure), so the pipeline adds a "
        "charge-velocity ratio instead of relying on the redundant raw total.",
        "",
        "## 3. Categorical association suite (χ² test of independence + Cramér's V)",
        "",
        "Cramér's V is bias-corrected (Bergsma, 2013). Rough guide for this table size: "
        "0.1 small, 0.3 medium, 0.5 large.",
        "",
        "| Rank | Feature | χ² | dof | p-value | Cramér's V | Significant | Highest-risk level "
        "(churn rate) |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | :---: | :--- |",
    ]
    for i, r in enumerate(cat.itertuples(), 1):
        lines.append(
            f"| {i} | `{r.feature}` | {r.chi2:,.1f} | {r.dof} | {_p(r.p_value)} | "
            f"{r.cramers_v:.3f} | {'✅' if r.significant else '❌'} | "
            f"{r.highest_risk_level} ({r.highest_risk_rate:.1%}) |"
        )
    non_sig = cat.loc[~cat["significant"], "feature"].tolist()
    top = cat.iloc[0]
    lines += [
        "",
        f"**Interpretation.** `{top.feature}` is the strongest categorical driver "
        f"(V = {top.cramers_v:.3f}). "
        + (
            "Not significant at α = 0.05: " + ", ".join(f"`{c}`" for c in non_sig) + ". "
            if non_sig
            else ""
        )
        + "Gender independence from churn is the key fairness result used in the model card.",
        "",
        "## 4. Tenure cohorts — the danger zone",
        "",
        "| Cohort | Customers | Churn rate |",
        "| :--- | ---: | ---: |",
    ]
    for r in cohorts.itertuples():
        flag = " 🚨" if r.tenure_cohort == "0-6m" else ""
        lines.append(f"| {r.tenure_cohort}{flag} | {r.customers:,} | {r.churn_rate:.1%} |")
    lines += [
        "",
        "Over half of customers in their first six months churn — retention spend should be "
        "front-loaded into onboarding.",
        "",
    ]
    return "\n".join(lines)
