"""Generate every data-driven part of the LaTeX paper and package it for Overleaf.

Writes ``docs/paper/generated/*.tex`` (macros, tables, pgfplots coordinates) and
``docs/paper/figures/*`` from the committed model, metadata and dataset, then zips
``docs/paper`` into ``docs/paper/churnai_paper_overleaf.zip``. No number in the paper is
typed by hand, so the paper cannot drift from the repository.

Run: uv run python scripts/build_paper.py
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.analytics.statistics import chi_square_suite, cohort_churn, continuous_tests
from src.config import get_settings
from src.features.engineer import clean_raw, load_raw, split_features_target
from src.features.schema import CustomerFeatures
from src.models.evaluate import compute_metrics, holdout_split
from src.models.explainability import PRETTY_NAMES, ChurnExplainer, generate_counterfactuals
from src.models.inference import ChurnPredictor

S = get_settings()
PAPER = S.docs_dir / "paper"
GEN = PAPER / "generated"
FIG = PAPER / "figures"

# The "High-risk newcomer" preset shown in the dashboard screenshots.
EXAMPLE = {
    "tenure": 2, "Contract": "Month-to-month", "InternetService": "Fiber optic",
    "MonthlyCharges": 94.5, "TotalCharges": 189.0, "PaymentMethod": "Electronic check",
    "TechSupport": "No", "OnlineSecurity": "No", "PaperlessBilling": "Yes", "SeniorCitizen": 1,
    "Partner": "No", "Dependents": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
}  # fmt: skip
SHORT_NAMES = {
    "elasticnet_lr": "ElasticNet LR",
    "balanced_rf": "Balanced RF",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "stacking": "Stacking",
}


def tex(s: object) -> str:
    """Escape text for LaTeX."""
    out = str(s)
    for a, b in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"), ("×", r"$\times$"), ("—", "---"), ("–", "--"),
        ("→", r"$\rightarrow$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
    ]:  # fmt: skip
        out = out.replace(a, b)
    return out


def pval(p: float) -> str:
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return rf"${mant}\times10^{{{int(exp)}}}$"
    return f"{p:.3f}"


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}\\%"


def write(name: str, body: str) -> None:
    (GEN / name).write_text(body.strip() + "\n", encoding="utf-8")


def macro(name: str, value: object) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main() -> None:  # noqa: PLR0915 (linear report assembly)
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    meta = json.loads(S.metadata_path.read_text(encoding="utf-8"))
    bench = json.loads((S.evaluations_dir / "model_benchmarks.json").read_text(encoding="utf-8"))
    df = clean_raw(load_raw(S.raw_data_path))
    train_df, test_df = holdout_split(df, S)
    X_train, _ = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    predictor = ChurnPredictor.load()
    proba = predictor.predict_proba(X_test)
    h, champ = meta["holdout"], meta["champion"]
    hw = meta["hardware"]
    cont, chi, cohorts = continuous_tests(df), chi_square_suite(df), cohort_churn(df)
    corr = df[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
    gender = chi.set_index("feature").loc["gender"]
    contract = chi.set_index("feature").loc["Contract"]
    rate = float(df["Churn"].mean())
    unit = meta["cost_matrix"]["unit_usd"]
    naive_none = len(test_df) * meta["train_churn_rate"] * meta["cost_matrix"]["fn"] * unit
    naive_all = len(test_df) * (1 - meta["train_churn_rate"]) * meta["cost_matrix"]["fp"] * unit

    # ------------------------------------------------------------------ macros
    m = [
        macro("NCustomers", f"{len(df):,}".replace(",", "{,}")),
        macro("NTrain", f"{meta['n_train']:,}".replace(",", "{,}")),
        macro("NHoldout", f"{meta['n_holdout']:,}".replace(",", "{,}")),
        macro("ChurnRate", pct(rate, 2)),
        macro("NChurners", f"{int(df['Churn'].sum()):,}".replace(",", "{,}")),
        macro("Champion", tex(meta["champion_display_name"])),
        macro("Threshold", f"{meta['decision_threshold']:.3f}"),
        macro("HoldoutROC", f"{h['roc_auc']:.3f}"),
        macro("HoldoutPR", f"{h['pr_auc']:.3f}"),
        macro("HoldoutRecall", pct(h["recall"])),
        macro("HoldoutPrecision", pct(h["precision"])),
        macro("HoldoutFtwo", f"{h['f2']:.3f}"),
        macro("HoldoutBrier", f"{h['brier']:.3f}"),
        macro("HoldoutTP", h["tp"]),
        macro("HoldoutFN", h["fn"]),
        macro("HoldoutFP", h["fp"]),
        macro("HoldoutTN", h["tn"]),
        macro("HoldoutPositives", h["tp"] + h["fn"]),
        macro("CVPR", f"{meta['cv']['mean']['pr_auc']:.3f}"),
        macro("CVPRsd", f"{meta['cv']['std']['pr_auc']:.3f}"),
        macro("CVROC", f"{meta['cv']['mean']['roc_auc']:.3f}"),
        macro("CVBrier", f"{meta['cv']['mean']['brier']:.3f}"),
        macro("CVFolds", meta["cv"]["folds"]),
        macro("CVRepeats", meta["cv"]["repeats"]),
        macro("LossChampion", f"\\${h['expected_loss_usd']:,.0f}".replace(",", "{,}")),
        macro("LossNone", f"\\${naive_none:,.0f}".replace(",", "{,}")),
        macro("LossAll", f"\\${naive_all:,.0f}".replace(",", "{,}")),
        macro("LossSavingNone", pct(1 - h["expected_loss_usd"] / naive_none, 0)),
        macro("LossSavingAll", pct(1 - h["expected_loss_usd"] / naive_all, 0)),
        macro("CostUnit", f"\\${unit:.0f}"),
        macro("DangerChurn", pct(float(cohorts.iloc[0]["churn_rate"]), 1)),
        macro("LoyalChurn", pct(float(cohorts.iloc[-1]["churn_rate"]), 1)),
        macro("GenderP", f"{gender['p_value']:.3f}"),
        macro("GenderChi", f"{gender['chi2']:.2f}"),
        macro("ContractV", f"{contract['cramers_v']:.3f}"),
        macro("RTenureTotal", f"{corr.loc['tenure', 'TotalCharges']:.3f}"),
        macro("RMonthlyTotal", f"{corr.loc['MonthlyCharges', 'TotalCharges']:.3f}"),
        macro("Latency", f"{bench['models'][champ]['latency_ms']:.0f}"),
        macro("HWcpu", tex(hw["cpu"].replace("(R)", "").replace("(TM)", ""))),
        macro("HWthreads", hw["logical_cores"]),
        macro("HWram", f"{math.ceil(hw['ram_gb'])}"),  # nominal capacity
        macro("HWos", tex(hw["os"].split(" (")[0])),
        macro("PyVersion", tex(meta["python"])),
        macro("SklearnVersion", tex(meta["libraries"]["scikit-learn"])),
        macro("ModelVersion", tex(meta["model_version"])),
    ]
    for lib, name in [("xgboost", "XGBVersion"), ("lightgbm", "LGBMVersion"),
                      ("catboost", "CatBoostVersion"), ("shap", "ShapVersion"),
                      ("pandas", "PandasVersion"), ("numpy", "NumpyVersion")]:  # fmt: skip
        m.append(macro(name, tex(meta["libraries"][lib])))
    write(
        "macros.tex", "% Auto-generated by scripts/build_paper.py -- do not edit.\n" + "\n".join(m)
    )

    # ------------------------------------------------------------------ tables
    rows = []
    for r in cont.itertuples():
        name = {"TotalCharges_log": r"$\log(1+\mathit{TotalCharges})$"}.get(
            r.feature, rf"\texttt{{{tex(r.feature)}}}"
        )
        rows.append(
            f"{name} & {r.mean_churned:,.1f} & {r.mean_retained:,.1f} & {r.welch_t:.1f} & "
            f"{pval(r.welch_p)} & {pval(r.mannwhitney_p)} & {r.rank_biserial:+.2f} \\\\"
        )
    write("tab_continuous.tex", "\n".join(rows))

    rows = []
    for i, r in enumerate(chi.itertuples(), 1):
        sig = r"\cmark" if r.significant else r"\xmark"
        rows.append(
            f"{i} & \\texttt{{{tex(r.feature)}}} & {r.chi2:,.1f} & {r.dof} & {pval(r.p_value)} & "
            f"\\vbar{{{r.cramers_v:.3f}}} & {sig} & {tex(r.highest_risk_level)} "
            f"({pct(r.highest_risk_rate, 0)}) \\\\"
        )
    write("tab_chisq.tex", "\n".join(rows))

    models = sorted(bench["models"].values(), key=lambda r: -r["cv_mean"]["pr_auc"])
    rows = []
    for r in models:
        best = r["key"] == champ
        name = rf"\textbf{{{tex(r['display_name'])}}}$^\star$" if best else tex(r["display_name"])
        c, sd = r["cv_mean"], r["cv_std"]
        rows.append(
            f"{name} & {c['pr_auc']:.3f}\\,\\pm\\,{sd['pr_auc']:.3f} & "
            f"{c['roc_auc']:.3f}\\,\\pm\\,{sd['roc_auc']:.3f} & {c['brier']:.3f} & "
            f"{r['cv_cost_per_customer']:.3f} & {r['fit_time_s']:.1f} \\\\"
        )
    write(
        "tab_cv.tex", "\n".join(rows).replace("\\pm", "$\\pm$").replace("\\,$\\pm$\\,", " $\\pm$ ")
    )

    rows = []
    for r in sorted(bench["models"].values(), key=lambda r: r["holdout"]["expected_loss_usd"]):
        hh = r["holdout"]
        best = r["key"] == champ
        name = rf"\textbf{{{tex(r['display_name'])}}}$^\star$" if best else tex(r["display_name"])
        rows.append(
            f"{name} & {r['oof_threshold']:.3f} & {hh['roc_auc']:.3f} & {hh['pr_auc']:.3f} & "
            f"{hh['recall']:.3f} & {hh['precision']:.3f} & {hh['f2']:.3f} & {hh['brier']:.3f} & "
            f"{r['latency_ms']:.0f} & {hh['expected_loss_usd']:,.0f} \\\\".replace(",", "{,}")
        )
    write("tab_holdout.tex", "\n".join(rows))

    rows = []
    deployed = meta["decision_threshold"]
    for t in sorted({0.10, 0.15, deployed, 0.25, 0.30, 0.40, 0.50}):
        mt = compute_metrics(y_test, proba, t, S)
        label = rf"\textbf{{{t:.3f}}}$^\dagger$" if t == deployed else f"{t:.2f}"
        rows.append(
            f"{label} & {mt['recall']:.3f} & {mt['precision']:.3f} & {mt['tp']} & {mt['fp']} & "
            f"{mt['fn']} & {mt['expected_loss_usd']:,.0f} \\\\".replace(",", "{,}")
        )
    write("tab_threshold.tex", "\n".join(rows))

    rows = []
    for col, labels in [("gender", {"Female": "Female", "Male": "Male"}),
                        ("SeniorCitizen", {0: "Not senior", 1: "Senior citizen"}),
                        ("Partner", {"No": "No partner", "Yes": "Partner"}),
                        ("Dependents", {"No": "No dependents", "Yes": "Dependents"})]:  # fmt: skip
        for level, label in labels.items():
            mask = (test_df[col] == level).to_numpy()
            mt = compute_metrics(y_test[mask], proba[mask], deployed, S)
            fpr = mt["fp"] / max(1, mt["fp"] + mt["tn"])
            rows.append(
                f"{label} & {int(mask.sum())} & {pct(float(y_test[mask].mean()))} & "
                f"{pct(float(proba[mask].mean()))} & {pct(mt['recall'])} & {pct(fpr)} & "
                f"{mt['roc_auc']:.3f} \\\\"
            )
        rows.append(r"\addlinespace[2pt]")
    write("tab_fairness.tex", "\n".join(rows[:-1]))

    explainer = ChurnExplainer(predictor.model, X_train)
    imp = explainer.global_importance(X_test.head(800)).head(10)
    rows = [
        f"{i} & {tex(PRETTY_NAMES.get(r.feature, r.feature))} & \\impbar{{{r.mean_abs_shap:.3f}}} \\\\"
        for i, r in enumerate(imp.itertuples(), 1)
    ]
    write("tab_importance.tex", "\n".join(rows))
    write("macros_importance.tex", macro("ImpMax", f"{imp['mean_abs_shap'].max():.3f}"))

    example = CustomerFeatures(**EXAMPLE)
    frame = example.to_frame()
    p0 = float(predictor.predict_proba(frame)[0])
    local = explainer.explain_instance(frame).top(7)
    rows = [f"{tex(r.label)} & {r.shap:+.3f} \\\\" for r in local.itertuples()]
    write("tab_local_shap.tex", "\n".join(rows))
    cfs = generate_counterfactuals(predictor.predict_proba, frame, top_k=6)
    rows = [
        f"{tex(c.action)} & {pct(c.new_probability, 0)} & "
        f"$-${(c.baseline_probability - c.new_probability) * 100:.0f} \\\\"
        for c in cfs
    ]
    write("tab_counterfactual.tex", "\n".join(rows))
    write("macros_example.tex", macro("ExampleP", pct(p0, 0)))

    _charts(df, cohorts, chi, bench, meta, y_test, proba)

    # ------------------------------------------------------------------ raster figures
    _shap_figure(explainer, X_test.head(800))
    _correlation_figure(df)
    for name in ("01_risk_engine.png", "02_explainability.png"):
        shutil.copy(S.docs_dir / "screenshots" / name, FIG / name)

    _zip()


def _coords(pairs: list[tuple[object, float]], fmt: str = ".4f") -> str:
    return " ".join(f"({x},{y:{fmt}})" for x, y in pairs)


def _charts(
    df: pd.DataFrame,
    cohorts: pd.DataFrame,
    chi: pd.DataFrame,
    bench: dict,
    meta: dict,
    y_test: pd.Series,
    proba: np.ndarray,
) -> None:
    """Write complete, self-contained pgfplots pictures (styles live in main.tex)."""
    rate = float(df["Churn"].mean()) * 100
    danger = cohorts.iloc[0]
    rest = cohorts.iloc[1:]
    labels = ",".join(cohorts["tenure_cohort"])
    write(
        "fig_cohort.tex",
        rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ci base, width=\columnwidth, height=4.5cm, symbolic x coords={{{labels}}},
  xtick={{{labels}}}, ymin=0, ymax=64, ylabel={{Churn rate (\%)}}, xlabel={{Tenure cohort}},
  enlarge x limits=0.14, ymajorgrids,
  extra y ticks={{{rate:.2f}}}, extra y tick labels={{}},
  extra y tick style={{grid=major, major grid style={{ciAmber, dashed, thick}}}}]
\addplot[ybar, bar width=18pt, bar shift=0pt, fill=ciCrimson, draw=none, ci labels]
  coordinates {{({danger.tenure_cohort},{danger.churn_rate * 100:.2f})}};
\addplot[ybar, bar width=18pt, bar shift=0pt, fill=ciViolet, draw=none, ci labels]
  coordinates {{{_coords([(r.tenure_cohort, r.churn_rate * 100) for r in rest.itertuples()], ".2f")}}};
\node[font=\scriptsize\sffamily, text=ciAmber, anchor=south east]
  at (rel axis cs:0.985,{rate / 64:.4f}) {{overall {rate:.1f}\%}};
\end{{axis}}
\end{{tikzpicture}}""",
    )

    ranked = sorted(bench["models"].values(), key=lambda r: -r["cv_mean"]["pr_auc"])
    names = ",".join(SHORT_NAMES[r["key"]] for r in ranked)
    champ = meta["champion"]

    def bars(rows: list[dict], color: str) -> str:
        pts = " ".join(
            f"({SHORT_NAMES[r['key']]},{r['cv_mean']['pr_auc']:.4f}) +- (0,{r['cv_std']['pr_auc']:.4f})"
            for r in rows
        )
        return (
            rf"\addplot[ybar, bar width=15pt, bar shift=0pt, fill={color}, draw=none, area legend,"
            rf" error bars/.cd, y dir=both, y explicit, error bar style={{ciInk, thick}}]"
            "\n  coordinates {" + pts + "};"
        )

    write(
        "fig_models.tex",
        rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ci base, width=\columnwidth, height=5cm, symbolic x coords={{{names}}},
  xtick={{{names}}}, x tick label style={{rotate=28, anchor=north east, font=\scriptsize\sffamily}},
  ymin=0.60, ymax=0.68, ylabel={{PR-AUC}}, enlarge x limits=0.1, ymajorgrids,
  y tick label style={{/pgf/number format/fixed, /pgf/number format/precision=2}},
  legend style={{at={{(0.99,0.98)}}, anchor=north east, font=\scriptsize\sffamily,
  draw=none, fill=white, fill opacity=0.85, text opacity=1}}, legend cell align=left]
{bars([r for r in ranked if r["key"] == champ], "ciEmerald")}
\addlegendentry{{CV mean $\pm$ SD (champion)}}
{bars([r for r in ranked if r["key"] != champ], "ciViolet!80")}
\addlegendentry{{CV mean $\pm$ SD}}
\addplot[only marks, mark=diamond*, mark size=3pt, color=ciAmber]
  coordinates {{{" ".join(f"({SHORT_NAMES[r['key']]},{r['holdout']['pr_auc']:.4f})" for r in ranked)}}};
\addlegendentry{{Holdout}}
\end{{axis}}
\end{{tikzpicture}}""",
    )

    fpr, tpr, _ = roc_curve(y_test, proba)
    grid = np.linspace(0, 1, 70)
    roc = list(zip(grid, np.interp(grid, fpr, tpr), strict=True))
    auc = meta["holdout"]["roc_auc"]
    write(
        "fig_roc.tex",
        rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ci base, ci square, xlabel={{False positive rate}}, ylabel={{True positive rate}}]
\addplot[draw=ciViolet, very thick, fill=ciViolet, fill opacity=0.12] coordinates {{{_coords(roc)}}} \closedcycle;
\addplot[ciSlate, dashed] coordinates {{(0,0) (1,1)}};
\node[ci note] at (rel axis cs:0.62,0.18) {{AUC $= {auc:.3f}$}};
\end{{axis}}
\end{{tikzpicture}}""",
    )

    prec, rec, _ = precision_recall_curve(y_test, proba)
    grid_r = np.linspace(0.01, 1, 70)
    prec_i = [float(prec[rec >= g].max()) for g in grid_r]  # interpolated precision
    base = float(y_test.mean())
    write(
        "fig_pr.tex",
        rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ci base, ci square, xlabel={{Recall}}, ylabel={{Precision}}]
\addplot[draw=ciCyan!85!black, very thick] coordinates {{{_coords(list(zip(grid_r, prec_i, strict=True)))}}};
\addplot[ciSlate, dashed] coordinates {{(0,{base:.4f}) (1,{base:.4f})}};
\node[ci note] at (rel axis cs:0.6,0.86) {{AP $= {meta["holdout"]["pr_auc"]:.3f}$}};
\node[font=\tiny\sffamily, text=ciSlate, anchor=south west] at (axis cs:0.02,{base:.4f}) {{base rate {base:.2f}}};
\end{{axis}}
\end{{tikzpicture}}""",
    )

    frac, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    write(
        "fig_reliability.tex",
        rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ci base, ci square, xlabel={{Mean predicted probability}}, ylabel={{Observed churn rate}}]
\addplot[ciSlate, dashed] coordinates {{(0,0) (1,1)}};
\addplot[ciEmerald, very thick, mark=*, mark size=1.6pt] coordinates {{{_coords(list(zip(mean_pred, frac, strict=True)))}}};
\node[ci note] at (rel axis cs:0.62,0.18) {{Brier $= {meta["holdout"]["brier"]:.3f}$}};
\end{{axis}}
\end{{tikzpicture}}""",
    )

    ts = np.round(np.arange(0.02, 0.81, 0.01), 2)
    losses = [compute_metrics(y_test, proba, t, S)["expected_loss_usd"] / 1000 for t in ts]
    lo, hi = min(losses), max(losses)
    thr = meta["decision_threshold"]
    deployed = compute_metrics(y_test, proba, thr, S)["expected_loss_usd"] / 1000
    ymin, ymax = np.floor(lo / 5) * 5, np.ceil(hi / 5) * 5

    def vline(x: float, style: str) -> str:
        return rf"\draw[{style}] (axis cs:{x:.4f},{ymin}) -- (axis cs:{x:.4f},{ymax});"

    write(
        "fig_cost.tex",
        rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ci base, width=\columnwidth, height=4.8cm, xmin=0, xmax=0.8, ymin={ymin}, ymax={ymax},
  xlabel={{Decision threshold $t$}}, ylabel={{Expected loss (k\$)}}, ymajorgrids, xmajorgrids,
  legend style={{at={{(0.02,0.98)}}, anchor=north west, font=\scriptsize\sffamily, draw=none,
  fill=white, fill opacity=0.85, text opacity=1}}, legend cell align=left]
\addplot[ciViolet, very thick] coordinates {{{_coords(list(zip(ts, losses, strict=True)), ".3f")}}};
\addlegendentry{{Holdout expected loss}}
\addlegendimage{{ciEmerald, thick, dashed}}\addlegendentry{{Deployed $t={thr:.3f}$ (OOF)}}
\addlegendimage{{ciAmber, thick, dotted}}\addlegendentry{{Bayes rule $t^\ast=1/6$}}
\addlegendimage{{ciCrimson, thick, dotted}}\addlegendentry{{Na\"ive $t=0.5$}}
{vline(thr, "ciEmerald, thick, dashed")}
{vline(1 / 6, "ciAmber, thick, dotted")}
{vline(0.5, "ciCrimson, thick, dotted")}
\addplot[only marks, mark=*, mark size=2.2pt, ciEmerald] coordinates {{({thr:.4f},{deployed:.3f})}};
\end{{axis}}
\end{{tikzpicture}}""",
    )


def _light_rc() -> None:
    mpl.rc_file_defaults()
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 200,
        "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False,
    })  # fmt: skip


def _shap_figure(explainer: ChurnExplainer, X: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import shap

    _light_rc()
    exp = explainer.shap_values(X)
    exp.feature_names = [PRETTY_NAMES.get(n, n) for n in explainer.feature_names]
    plt.figure(figsize=(3.45, 3.3))  # native column width: no downscaled text
    shap.plots.beeswarm(exp, max_display=12, show=False, plot_size=None)
    # shap hard-codes 13pt labels; scale every axis (incl. the colour bar) to print size.
    for ax in plt.gcf().axes:
        ax.tick_params(labelsize=6.5)
        ax.xaxis.label.set_size(7.5)
        ax.yaxis.label.set_size(7.5)
    for coll in plt.gca().collections:
        coll.set_sizes([6])
    plt.xlabel("SHAP value (impact on log-odds of churn)")
    plt.savefig(FIG / "shap_summary_light.png")
    plt.close("all")


def _correlation_figure(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    _light_rc()
    cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]
    corr = df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(
        range(len(cols)), [c.replace("Charges", "Chg.") for c in cols], rotation=35, ha="right"
    )
    ax.set_yticks(range(len(cols)), [c.replace("Charges", "Chg.") for c in cols])
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.55 else "black")  # fmt: skip
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIG / "correlation_light.png")
    plt.close(fig)


def _zip() -> None:
    out = PAPER / "churnai_paper_overleaf.zip"
    out.unlink(missing_ok=True)
    keep = [
        p
        for p in PAPER.rglob("*")
        if p.is_file() and p.suffix in {".tex", ".bib", ".png", ".cls", ".sty"}
    ]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(keep):
            zf.write(p, p.relative_to(PAPER).as_posix())
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(keep)} files)")


if __name__ == "__main__":
    main()
