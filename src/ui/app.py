"""ChurnAI multi-page Streamlit portal.

Run with::

    uv run streamlit run src/ui/app.py
"""

from __future__ import annotations

import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow `streamlit run src/ui/app.py` without an editable install (e.g. Streamlit Cloud).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.analytics.statistics import chi_square_suite, cohort_churn, continuous_tests  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.core.exceptions import ChurnAIError, DataValidationError  # noqa: E402
from src.db.database import get_engine, init_db  # noqa: E402
from src.db.repository import CustomerRepository, customer_to_frame  # noqa: E402
from src.features.engineer import clean_raw, load_clean  # noqa: E402
from src.features.schema import (  # noqa: E402
    CATEGORICAL_LEVELS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    CustomerFeatures,
)
from src.models.explainability import ChurnExplainer, LocalExplanation, generate_counterfactuals  # noqa: E402
from src.models.inference import ChurnPredictor, PredictionResult  # noqa: E402
from src.ui import components as ui  # noqa: E402
from src.ui.styles import TIER_COLORS, TOKENS, inject_styles  # noqa: E402

st.set_page_config(
    page_title="ChurnAI · Retention Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

PAGES = {
    "risk": ":material/query_stats: Real-Time Risk Engine",
    "db": ":material/database: Customer Database",
    "analytics": ":material/insights: Executive Analytics",
    "batch": ":material/upload_file: Batch Inference",
}

# Widget keys for the risk-engine form (shared with "open in risk engine" and presets).
FIELD_KEYS = {c: f"f_{c}" for c in FEATURE_COLUMNS}

PRESETS: dict[str, dict[str, Any]] = {
    "High-risk newcomer": {
        "tenure": 2,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "MonthlyCharges": 94.5,
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
    },
    "Loyal veteran": {
        "tenure": 64,
        "Contract": "Two year",
        "InternetService": "DSL",
        "MonthlyCharges": 64.0,
        "PaymentMethod": "Bank transfer (automatic)",
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "PaperlessBilling": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "SeniorCitizen": 0,
    },
    "Mid-tenure family": {
        "tenure": 22,
        "Contract": "One year",
        "InternetService": "Fiber optic",
        "MonthlyCharges": 82.0,
        "PaymentMethod": "Credit card (automatic)",
        "TechSupport": "No",
        "OnlineSecurity": "Yes",
        "Partner": "Yes",
        "Dependents": "Yes",
        "SeniorCitizen": 0,
        "PaperlessBilling": "Yes",
    },
}


# --------------------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor.load()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_clean()


@st.cache_resource(show_spinner=False)
def get_explainer() -> ChurnExplainer:
    return ChurnExplainer(get_predictor().model, get_dataset())


@st.cache_resource(show_spinner=False)
def get_repository() -> CustomerRepository:
    engine = init_db(get_engine())
    repo = CustomerRepository(engine)
    if repo.count_customers() == 0:
        repo.seed_from_dataframe(get_dataset())
    return repo


@st.cache_data(show_spinner=False)
def get_statistics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = get_dataset()
    return continuous_tests(df), chi_square_suite(df), cohort_churn(df)


@st.cache_data(show_spinner=False)
def get_benchmarks() -> dict[str, Any] | None:
    path = get_settings().evaluations_dir / "model_benchmarks.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def shap_summary(exp: LocalExplanation, k: int = 6) -> dict[str, float]:
    return {row.feature: round(float(row.shap), 4) for row in exp.top(k).itertuples()}


def apply_values_to_form(values: dict[str, Any]) -> None:
    for col, val in values.items():
        if col in FIELD_KEYS:
            st.session_state[FIELD_KEYS[col]] = val
    if "TotalCharges" not in values and "tenure" in values and "MonthlyCharges" in values:
        st.session_state["derive_total"] = True


def on_preset_change() -> None:
    preset = st.session_state.get("preset")
    if preset in PRESETS:
        defaults = CustomerFeatures().model_dump()
        apply_values_to_form({**defaults, **PRESETS[preset]})
        st.session_state["derive_total"] = True


def open_in_risk_engine(customer_id: str) -> None:
    customer = get_repository().get_customer(customer_id)
    if customer is None:
        return
    apply_values_to_form(customer.to_feature_dict())
    st.session_state["derive_total"] = False
    st.session_state["preset"] = "Custom"
    st.session_state["nav"] = PAGES["risk"]


def build_report(
    features: dict[str, Any],
    result: PredictionResult,
    exp: LocalExplanation,
    cfs: list[Any],
    champion: str,
) -> tuple[str, str]:
    now = datetime.now(UTC)
    payload = {
        "report": "ChurnAI Executive Risk Assessment",
        "generated_at": now.isoformat(),
        "model": {
            "name": champion,
            "version": result.model_version,
            "decision_threshold": result.threshold,
        },
        "prediction": result.to_dict(),
        "customer_profile": features,
        "top_drivers": [
            {"feature": r.label, "shap": round(float(r.shap), 4)} for r in exp.top(8).itertuples()
        ],
        "recommended_actions": [
            {
                "action": c.action,
                "new_probability": round(c.new_probability, 4),
                "risk_reduction_pts": round((c.baseline_probability - c.new_probability) * 100, 1),
            }
            for c in cfs
        ],
        "disclaimer": "Model-based what-if estimates; not causal guarantees.",
    }
    drivers = "\n".join(
        f"  {'▲' if r.shap > 0 else '▼'} {r.label:<45} {r.shap:+.3f}"
        for r in exp.top(8).itertuples()
    )
    actions = (
        "\n".join(f"  • {c.description}" for c in cfs)
        or "  • No single action lowers risk materially."
    )
    decision = "INTERVENE" if result.predicted_label else "MONITOR"
    text = f"""CHURNAI — EXECUTIVE RISK ASSESSMENT
Generated : {now:%Y-%m-%d %H:%M:%S} UTC
Model     : {champion} ({result.model_version})
==================================================================
RESULT
  Churn probability : {result.churn_probability:.1%}
  Risk tier         : {result.risk_tier.upper()}
  Decision          : {decision} (threshold {result.threshold:.2f})

CUSTOMER SNAPSHOT
  Tenure            : {features["tenure"]} months ({features["Contract"]})
  Internet          : {features["InternetService"]}
  Payment           : {features["PaymentMethod"]}
  Monthly / Total   : ${features["MonthlyCharges"]:,.2f} / ${features["TotalCharges"]:,.2f}

TOP RISK DRIVERS (SHAP, {exp.output_space})
{drivers}

RECOMMENDED RETENTION ACTIONS
{actions}
==================================================================
Model-based what-if estimates, not causal guarantees. CONFIDENTIAL — internal use only.
"""
    return text, json.dumps(payload, indent=2, default=str)


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------


def sidebar(predictor: ChurnPredictor | None) -> str:
    with st.sidebar:
        ui.render(
            f"""<div style="display:flex;align-items:center;gap:.7rem;margin:.2rem 0 1.2rem 0">
            <div style="width:42px;height:42px;border-radius:12px;display:grid;place-items:center;
            background:linear-gradient(135deg,#6366F1,#22D3EE);
            box-shadow:0 8px 24px rgba(99,102,241,.4)" class="brand-mark">
            {ui.icon("cell_tower")}</div>
            <div><div style="font-weight:800;font-size:1.25rem;letter-spacing:-.02em">ChurnAI</div>
            <div class="faint" style="font-size:.75rem">Retention Intelligence · v2.0</div>
            </div></div>"""
        )
        choice = st.radio(
            "Navigation", list(PAGES.values()), key="nav", label_visibility="collapsed"
        )
        st.divider()
        if predictor is not None:
            h = predictor.metadata.get("holdout", {})
            ui.render(
                f"""<div class="glass" style="padding:.85rem 1rem">
                <p class="eyebrow" style="margin-bottom:.5rem">Production model</p>
                <div style="font-weight:700">{ui.esc(predictor.champion_name)}</div>
                <div class="faint mono" style="font-size:.72rem;margin:.2rem 0 .6rem 0">
                {ui.esc(predictor.version)}</div>
                <div class="kv" style="grid-template-columns:1fr 1fr">
                <div><b>PR-AUC</b><span>{h.get("pr_auc", float("nan")):.3f}</span></div>
                <div><b>ROC-AUC</b><span>{h.get("roc_auc", float("nan")):.3f}</span></div>
                <div><b>Recall</b><span>{h.get("recall", float("nan")):.1%}</span></div>
                <div><b>Threshold</b><span>{predictor.threshold:.2f}</span></div></div></div>"""
            )
        backend = get_settings().database_url.split(":", 1)[0].split("+")[0]
        ui.render(
            f'<div class="faint" style="font-size:.75rem;margin-top:.4rem">'
            f"{ui.icon('database')} Database backend: "
            f'<span class="mono">{ui.esc(backend)}</span></div>'
        )
    return choice


# --------------------------------------------------------------------------------------
# Page 1 — Real-time risk engine
# --------------------------------------------------------------------------------------


def _ensure_form_defaults() -> None:
    if "form_initialised" in st.session_state:
        return
    apply_values_to_form({**CustomerFeatures().model_dump(), **PRESETS["High-risk newcomer"]})
    st.session_state["preset"] = "High-risk newcomer"
    st.session_state["derive_total"] = True
    st.session_state["form_initialised"] = True


def _select(col: str, label: str) -> None:
    st.selectbox(label, CATEGORICAL_LEVELS[col], key=FIELD_KEYS[col])


def risk_engine_form() -> dict[str, Any]:
    _ensure_form_defaults()
    st.selectbox(
        "Scenario preset",
        ["Custom", *PRESETS],
        key="preset",
        on_change=on_preset_change,
        help="Load a representative customer profile, then fine-tune any input.",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Tenure (months)", 0, 72, key=FIELD_KEYS["tenure"])
        _select("Contract", "Contract")
        _select("InternetService", "Internet service")
    with c2:
        st.slider("Monthly charges ($)", 18.0, 120.0, step=0.5, key=FIELD_KEYS["MonthlyCharges"])
        _select("PaymentMethod", "Payment method")
        st.checkbox("Derive total charges from tenure × monthly", key="derive_total")
    tenure = st.session_state[FIELD_KEYS["tenure"]]
    monthly = st.session_state[FIELD_KEYS["MonthlyCharges"]]
    if st.session_state["derive_total"]:
        st.session_state[FIELD_KEYS["TotalCharges"]] = round(float(tenure * monthly), 2)
    st.number_input(
        "Total charges ($)",
        0.0,
        10_000.0,
        step=10.0,
        key=FIELD_KEYS["TotalCharges"],
        disabled=st.session_state["derive_total"],
    )

    with st.expander("Services & demographics", expanded=False, icon=":material/tune:"):
        a, b = st.columns(2)
        with a:
            _select("gender", "Gender")
            st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda v: "Yes" if v else "No",
                key=FIELD_KEYS["SeniorCitizen"],
            )
            _select("Partner", "Partner")
            _select("Dependents", "Dependents")
            _select("PhoneService", "Phone service")
            _select("MultipleLines", "Multiple lines")
            _select("PaperlessBilling", "Paperless billing")
        with b:
            for col, label in [
                ("OnlineSecurity", "Online security"),
                ("OnlineBackup", "Online backup"),
                ("DeviceProtection", "Device protection"),
                ("TechSupport", "Tech support"),
                ("StreamingTV", "Streaming TV"),
                ("StreamingMovies", "Streaming movies"),
            ]:
                _select(col, label)
    return {c: st.session_state[k] for c, k in FIELD_KEYS.items()}


def page_risk_engine(predictor: ChurnPredictor) -> None:
    ui.page_header(
        "Real-time risk engine",
        "Customer churn risk, explained.",
        "Adjust any attribute and the calibrated probability, SHAP attribution and retention "
        "prescriptions update instantly.",
    )
    left, right = st.columns([1.05, 1], gap="large")
    with left, ui.glass_container("form"):
        ui.section_title("badge", "Customer profile")
        raw = risk_engine_form()

    try:
        customer = CustomerFeatures.model_validate(raw)
    except ValueError as exc:
        right.error(f"Invalid input: {exc}")
        return
    frame = customer.to_frame()
    result = predictor.predict_one(customer)
    explanation = get_explainer().explain_instance(frame)
    cfs = generate_counterfactuals(predictor.predict_proba, frame, top_k=4)

    with right:
        ui.risk_gauge(result.churn_probability, result.risk_tier)
        decision = "Intervene" if result.predicted_label else "Monitor"
        ui.metric_row(
            [
                ui.metric_card(
                    "Decision",
                    decision,
                    caption=f"threshold {result.threshold:.2f}",
                    accent=TIER_COLORS[result.risk_tier],
                    delay_ms=120,
                ),
                ui.metric_card(
                    "Annual revenue at risk",
                    f"${customer.MonthlyCharges * 12 * result.churn_probability:,.0f}",
                    caption="12 × monthly × p(churn)",
                    accent=TOKENS["amber"],
                    delay_ms=200,
                ),
            ]
        )

    st.write("")
    w1, w2 = st.columns([1.25, 1], gap="large")
    with w1, ui.glass_container("shap"):
        ui.section_title("psychology", "Why this score? (local SHAP)")
        st.plotly_chart(
            ui.shap_waterfall(explanation), width="stretch", config={"displayModeBar": False}
        )
        st.caption(
            "Red bars push risk up, green bars pull it down. Values are on the inner "
            f"model's {explanation.output_space} scale; the gauge shows the calibrated "
            "probability."
        )
    with w2:
        ui.section_title("lightbulb", "Counterfactual retention prescriptions")
        if cfs:
            for i, cf in enumerate(cfs):
                ui.counterfactual_card(cf, delay_ms=120 + i * 100)
        else:
            ui.render(
                '<div class="glass enter">This customer is already low-risk; no single '
                "intervention lowers the score materially.</div>"
            )
        st.caption(
            "Model-based what-ifs for an otherwise identical customer — not causal estimates."
        )

        text, payload = build_report(raw, result, explanation, cfs, predictor.champion_name)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        d1, d2 = st.columns(2)
        d1.download_button(
            "Report (.txt)",
            text,
            f"churnai_risk_report_{stamp}.txt",
            "text/plain",
            width="stretch",
            icon=":material/download:",
        )
        d2.download_button(
            "Report (.json)",
            payload,
            f"churnai_risk_report_{stamp}.json",
            "application/json",
            width="stretch",
            icon=":material/data_object:",
        )
        if st.button(
            "Log assessment to audit trail",
            width="stretch",
            type="secondary",
            icon=":material/inventory_2:",
        ):
            try:
                get_repository().log_prediction(
                    customer_id=None,
                    model_version=result.model_version,
                    churn_probability=result.churn_probability,
                    predicted_label=result.predicted_label,
                    risk_tier=result.risk_tier,
                    shap_summary=shap_summary(explanation),
                    source="ui",
                )
                st.toast("Assessment logged to the audit trail.", icon=":material/check_circle:")
            except ChurnAIError as exc:
                st.error(f"Could not write to the database: {exc}")


# --------------------------------------------------------------------------------------
# Page 2 — Customer database
# --------------------------------------------------------------------------------------


def page_customer_db(predictor: ChurnPredictor) -> None:
    ui.page_header(
        "Live customer database",
        "Look up, score and audit any subscriber.",
        "Search the customer table by ID, score the stored profile with the production model "
        "and review the full prediction history.",
    )
    try:
        repo = get_repository()
        total = repo.count_customers()
    except ChurnAIError as exc:
        st.error(f"Database unavailable: {exc}")
        return

    with ui.glass_container("search"):
        c1, c2 = st.columns([1, 1.3])
        with c1:
            query = st.text_input(
                "Search by customer ID", placeholder="e.g. 7590 or 9237-HQITU", key="db_query"
            )
        matches = repo.search_customer_ids(query, limit=25) if query.strip() else []
        with c2:
            options = matches or repo.search_customer_ids("", limit=25)
            selected = st.selectbox(
                f"{len(matches)} match(es)" if query.strip() else f"Browse {total:,} customers",
                options,
                key="db_selected",
            )
    if not selected:
        st.info("No customer matches that ID.")
        return
    customer = repo.get_customer(selected)
    if customer is None:
        st.warning("Customer not found.")
        return

    features = customer.to_feature_dict()
    st.write("")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        ui.section_title("person", f"Profile · {customer.customer_id}")
        truth = {None: "unknown", 1: "Churned", 0: "Retained"}[customer.churned]
        ui.profile_grid(
            {
                "Tenure": f"{customer.tenure} month{'' if customer.tenure == 1 else 's'}",
                "Contract": customer.contract_type,
                "Internet": customer.internet_service,
                "Payment": customer.payment_method,
                "Monthly": f"${customer.monthly_charges:,.2f}",
                "Total": f"${customer.total_charges:,.2f}",
                "Tech support": customer.tech_support,
                "Security": customer.online_security,
                "Senior": "Yes" if customer.senior_citizen else "No",
                "Partner": customer.partner,
                "Dependents": customer.dependents,
                "Ground truth": truth,
            }
        )
    frame = customer_to_frame(customer)
    result = predictor.predict_one(features)
    with right:
        ui.risk_gauge(
            result.churn_probability, result.risk_tier, caption="Live score from the stored profile"
        )
        b1, b2 = st.columns(2)
        if b1.button("Score & log", width="stretch", icon=":material/save:"):
            exp = get_explainer().explain_instance(frame)
            repo.log_prediction(
                customer_id=customer.customer_id,
                model_version=result.model_version,
                churn_probability=result.churn_probability,
                predicted_label=result.predicted_label,
                risk_tier=result.risk_tier,
                shap_summary=shap_summary(exp),
                source="ui",
            )
            st.toast(f"Logged score for {customer.customer_id}", icon=":material/check_circle:")
        b2.button(
            "Open in risk engine",
            icon=":material/open_in_new:",
            width="stretch",
            type="secondary",
            on_click=open_in_risk_engine,
            args=(customer.customer_id,),
        )

    st.write("")
    with ui.glass_container("history"):
        ui.section_title("history", "Prediction audit history")
        history = repo.history_frame(customer.customer_id)
        if history.empty:
            ui.render(
                '<p class="muted">No logged predictions for this customer yet — use '
                "<b>Score &amp; log</b> to create the first audit record.</p>"
            )
        else:
            st.plotly_chart(
                ui.history_timeline(history, predictor.threshold),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.dataframe(
                history[
                    ["created_at", "churn_probability", "risk_tier", "model_version", "source"]
                ].sort_values("created_at", ascending=False),
                hide_index=True,
                width="stretch",
                column_config={
                    "churn_probability": st.column_config.ProgressColumn(
                        "Churn probability", format="percent", min_value=0.0, max_value=1.0
                    )
                },
            )

    with st.expander("Latest predictions across all customers", icon=":material/receipt_long:"):
        recent = pd.DataFrame(
            [r.model_dump(exclude={"shap_summary"}) for r in repo.recent_predictions(50)]
        )
        if recent.empty:
            st.caption("The audit trail is empty.")
        else:
            st.dataframe(recent, hide_index=True, width="stretch")


# --------------------------------------------------------------------------------------
# Page 3 — Executive analytics & statistical proof
# --------------------------------------------------------------------------------------


def page_analytics() -> None:
    ui.page_header(
        "Executive analytics",
        "Where churn comes from — with statistical proof.",
        "Interactive cohort analytics backed by Welch's t-tests, Mann-Whitney U, χ² tests and "
        "Cramér's V effect sizes computed live on the full dataset.",
    )
    df = get_dataset()
    cont, chi, cohorts = get_statistics()
    churn_rate = df[TARGET_COLUMN].mean()
    danger = cohorts.loc[cohorts["tenure_cohort"] == "0-6m", "churn_rate"].iloc[0]
    mrr_lost = df.loc[df[TARGET_COLUMN] == 1, "MonthlyCharges"].sum()
    ui.metric_row(
        [
            ui.metric_card(
                "Customers",
                ui.countup(len(df)),
                caption="IBM Telco snapshot",
                accent=TOKENS["violet"],
                delay_ms=0,
            ),
            ui.metric_card(
                "Churn rate",
                ui.countup(churn_rate * 100, "%"),
                caption="class imbalance 1 : 2.8",
                accent=TOKENS["amber"],
                delay_ms=90,
            ),
            ui.metric_card(
                "Danger-zone churn",
                ui.countup(danger * 100, "%"),
                caption="tenure 0–6 months",
                accent=TOKENS["crimson"],
                delay_ms=180,
                effect="pulse",
            ),
            ui.metric_card(
                "Monthly revenue lost",
                f"${mrr_lost / 1000:,.0f}k",
                caption="MRR of churned customers",
                accent=TOKENS["emerald"],
                delay_ms=270,
                effect="pulse",
            ),
        ]
    )

    tab_overview, tab_evidence, tab_models = st.tabs(
        [
            ":material/trending_up: Business overview",
            ":material/science: Statistical evidence",
            ":material/emoji_events: Model benchmarks",
        ]
    )
    with tab_overview:
        c1, c2 = st.columns(2, gap="large")
        with c1, ui.glass_container("cohorts"):
            ui.section_title("hourglass_bottom", "Churn by tenure cohort — the danger zone")
            colors = [
                TOKENS["crimson"] if c == "0-6m" else TOKENS["violet"]
                for c in cohorts["tenure_cohort"]
            ]
            fig = go.Figure(
                go.Bar(
                    x=cohorts["tenure_cohort"],
                    y=cohorts["churn_rate"],
                    marker_color=colors,
                    customdata=cohorts["customers"],
                    text=[f"{v:.0%}" for v in cohorts["churn_rate"]],
                    textposition="outside",
                    hovertemplate=(
                        "%{x}: <b>%{y:.1%}</b> churn<br>%{customdata:,} customers<extra></extra>"
                    ),
                )
            )
            fig.update_yaxes(tickformat=".0%", range=[0, max(cohorts["churn_rate"]) * 1.25])
            st.plotly_chart(
                ui.style_figure(fig, legend=False),
                width="stretch",
                config={"displayModeBar": False},
            )
        with c2, ui.glass_container("contracts"):
            ui.section_title("stacked_bar_chart", "Churn by contract × internet service")
            grid = df.groupby(["Contract", "InternetService"])[TARGET_COLUMN].mean().reset_index()
            fig = px.bar(
                grid,
                x="Contract",
                y=TARGET_COLUMN,
                color="InternetService",
                barmode="group",
                color_discrete_sequence=[TOKENS["cyan"], TOKENS["crimson"], TOKENS["emerald"]],
                labels={TARGET_COLUMN: "Churn rate"},
            )
            fig.update_traces(
                hovertemplate="%{x} · %{fullData.name}<br><b>%{y:.1%}</b><extra></extra>"
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(ui.style_figure(fig), width="stretch", config={"displayModeBar": False})
        with ui.glass_container("charges"):
            ui.section_title("payments", "Monthly charge distribution by outcome")
            plot_df = df.assign(Outcome=np.where(df[TARGET_COLUMN] == 1, "Churned", "Retained"))
            fig = px.histogram(
                plot_df,
                x="MonthlyCharges",
                color="Outcome",
                nbins=50,
                barmode="overlay",
                opacity=0.72,
                histnorm="probability density",
                color_discrete_map={"Churned": TOKENS["crimson"], "Retained": TOKENS["emerald"]},
            )
            st.plotly_chart(
                ui.style_figure(fig, height=320), width="stretch", config={"displayModeBar": False}
            )

    with tab_evidence:
        with st.expander(
            "Continuous variables — Welch's t-test & Mann-Whitney U",
            expanded=True,
            icon=":material/functions:",
        ):
            show = cont[
                [
                    "feature",
                    "mean_churned",
                    "mean_retained",
                    "welch_t",
                    "welch_p",
                    "mannwhitney_p",
                    "rank_biserial",
                    "cohens_d",
                ]
            ]
            st.dataframe(
                show,
                hide_index=True,
                width="stretch",
                column_config={
                    "welch_p": st.column_config.NumberColumn("Welch p", format="%.2e"),
                    "mannwhitney_p": st.column_config.NumberColumn("Mann-Whitney p", format="%.2e"),
                    "mean_churned": st.column_config.NumberColumn(format="%.2f"),
                    "mean_retained": st.column_config.NumberColumn(format="%.2f"),
                    "welch_t": st.column_config.NumberColumn(format="%.2f"),
                    "rank_biserial": st.column_config.NumberColumn(format="%.3f"),
                    "cohens_d": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            st.caption(
                "Churners: shorter tenure, lower lifetime spend, higher monthly bills. "
                "Parametric and rank-based tests agree (p ≪ 0.001)."
            )
        with st.expander(
            "Categorical drivers — χ² independence & Cramér's V",
            expanded=True,
            icon=":material/hub:",
        ):
            fig = go.Figure(
                go.Bar(
                    y=chi["feature"][::-1],
                    x=chi["cramers_v"][::-1],
                    orientation="h",
                    marker_color=[
                        TOKENS["violet"] if s else TOKENS["text_faint"]
                        for s in chi["significant"][::-1]
                    ],
                    customdata=np.stack(
                        [chi["p_value"][::-1], chi["highest_risk_level"][::-1]], axis=1
                    ),
                    hovertemplate="%{y}: V = <b>%{x:.3f}</b><br>p = %{customdata[0]:.2e}"
                    "<br>riskiest level: %{customdata[1]}<extra></extra>",
                )
            )
            fig.update_layout(xaxis_title="Cramér's V (bias-corrected)")
            st.plotly_chart(
                ui.style_figure(fig, height=520, legend=False),
                width="stretch",
                config={"displayModeBar": False},
            )
            non_sig = ", ".join(chi.loc[~chi["significant"], "feature"]) or "none"
            st.caption(
                f"Grey bars are not significant at α = 0.05 ({non_sig}). "
                f"`{chi.iloc[0]['feature']}` is the primary driver."
            )
        with st.expander("Full statistical report", icon=":material/article:"):
            report = get_settings().evaluations_dir / "statistical_report.md"
            if report.exists():
                st.markdown(report.read_text(encoding="utf-8"))
            else:
                st.caption("Run notebook 02 to generate docs/evaluations/statistical_report.md.")

    with tab_models:
        bench = get_benchmarks()
        if bench is None:
            st.info("No benchmark file yet — run `uv run python -m src.models.train`.")
        else:
            rows = []
            for key, r in bench["models"].items():
                rows.append(
                    {
                        "Model": r["display_name"] + (" 🏆" if key == bench["champion"] else ""),
                        "CV PR-AUC": r["cv_mean"]["pr_auc"],
                        "±": r["cv_std"]["pr_auc"],
                        "Holdout ROC-AUC": r["holdout"]["roc_auc"],
                        "Holdout PR-AUC": r["holdout"]["pr_auc"],
                        "Recall": r["holdout"]["recall"],
                        "Brier": r["holdout"]["brier"],
                        "Latency (ms)": r["latency_ms"],
                        "Expected loss ($)": r["holdout"]["expected_loss_usd"],
                    }
                )
            table = pd.DataFrame(rows).sort_values("CV PR-AUC", ascending=False)
            with ui.glass_container("bench"):
                ui.section_title("leaderboard", "Tournament results")
                fig = go.Figure(
                    go.Bar(
                        x=table["Model"],
                        y=table["CV PR-AUC"],
                        error_y={
                            "type": "data",
                            "array": table["±"],
                            "color": TOKENS["text_muted"],
                        },
                        marker_color=[
                            TOKENS["emerald"] if "🏆" in m else TOKENS["violet"]
                            for m in table["Model"]
                        ],
                        hovertemplate="%{x}<br>PR-AUC %{y:.4f}<extra></extra>",
                    )
                )
                fig.update_yaxes(
                    range=[table["CV PR-AUC"].min() - 0.05, table["CV PR-AUC"].max() + 0.03]
                )
                st.plotly_chart(
                    ui.style_figure(fig, height=330, legend=False),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                st.dataframe(
                    table,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        c: st.column_config.NumberColumn(format="%.3f")
                        for c in [
                            "CV PR-AUC",
                            "±",
                            "Holdout ROC-AUC",
                            "Holdout PR-AUC",
                            "Recall",
                            "Brier",
                        ]
                    }
                    | {
                        "Latency (ms)": st.column_config.NumberColumn(format="%.1f"),
                        "Expected loss ($)": st.column_config.NumberColumn(format="$%d"),
                    },
                )
            fig_path = get_settings().figures_dir / "roc_pr_curves.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Holdout ROC and precision-recall curves")


# --------------------------------------------------------------------------------------
# Page 4 — Batch inference
# --------------------------------------------------------------------------------------


def _template_csv() -> bytes:
    sample = get_dataset().drop(columns=[TARGET_COLUMN]).sample(25, random_state=7)
    return sample.to_csv(index=False).encode()


def page_batch(predictor: ChurnPredictor) -> None:
    ui.page_header(
        "High-throughput batch inference",
        "Score an entire customer file in seconds.",
        "Drop a CSV with the Telco schema. Rows are validated, scored in chunks and ranked by "
        "risk; results can be exported or written to the audit trail.",
    )
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        upload = st.file_uploader("Drag & drop a customer CSV", type=["csv"], key="batch_upload")
    with c2, ui.glass_container("template"):
        ui.section_title("description", "Need a file?")
        st.caption("Download a 25-row sample in the expected schema (customerID optional).")
        st.download_button(
            "Sample CSV",
            _template_csv(),
            "churnai_batch_template.csv",
            "text/csv",
            width="stretch",
            icon=":material/download:",
        )
        persist = st.checkbox("Save customers & predictions to the database", value=False)

    if upload is None:
        ui.skeleton(4)
        st.caption("Awaiting upload…")
        return

    try:
        raw = pd.read_csv(io.BytesIO(upload.getvalue()))
        clean = clean_raw(raw)
        missing = [c for c in FEATURE_COLUMNS if c not in clean.columns]
        if missing:
            raise DataValidationError(
                f"Missing columns: {', '.join(missing)}", missing_columns=missing
            )
    except (DataValidationError, ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc), icon=":material/error:")
        return

    n = len(clean)
    progress = st.progress(0.0, text="Scoring…")
    chunk = max(250, n // 20)
    parts = []
    for start in range(0, n, chunk):
        parts.append(predictor.score_frame(clean.iloc[start : start + chunk]))
        progress.progress(
            min(1.0, (start + chunk) / n), text=f"Scored {min(n, start + chunk):,} / {n:,} rows"
        )
    progress.empty()
    scored = pd.concat(parts)
    id_col = (
        clean[ID_COLUMN]
        if ID_COLUMN in clean.columns
        else pd.Series([f"ROW-{i + 1:05d}" for i in range(n)], index=clean.index)
    )
    out = pd.concat([id_col.rename(ID_COLUMN), scored, clean[FEATURE_COLUMNS]], axis=1)
    out = out.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    counts = out["risk_tier"].value_counts()
    ui.metric_row(
        [
            ui.metric_card(
                f"{t} risk",
                ui.countup(int(counts.get(t, 0))),
                caption=f"{counts.get(t, 0) / n:.0%} of file",
                accent=TIER_COLORS[t],
                delay_ms=i * 90,
                effect="pulse" if t == "Critical" else "shimmer",
            )
            for i, t in enumerate(["Critical", "High", "Medium", "Low"])
        ]
    )
    with ui.glass_container("batch-results"):
        ui.section_title("format_list_numbered", f"{n:,} customers ranked by churn risk")
        tiers = st.multiselect(
            "Filter tiers",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"],
        )
        view = out[out["risk_tier"].isin(tiers)]
        st.dataframe(
            view,
            hide_index=True,
            width="stretch",
            height=460,
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", format="percent", min_value=0.0, max_value=1.0
                ),
                "predicted_label": st.column_config.CheckboxColumn("Intervene"),
            },
        )
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        st.download_button(
            f"Export {len(view):,} scored rows (.csv)",
            view.to_csv(index=False).encode(),
            f"churnai_batch_scores_{stamp}.csv",
            "text/csv",
            icon=":material/file_download:",
        )

    if persist and st.button("Write to database", type="secondary", icon=":material/save:"):
        repo = get_repository()
        has_ids = ID_COLUMN in clean.columns
        if has_ids:
            repo.seed_from_dataframe(clean)
        logged = repo.log_predictions_bulk(
            [
                {
                    "customer_id": r[ID_COLUMN] if has_ids else None,
                    "model_version": predictor.version,
                    "churn_probability": float(r["churn_probability"]),
                    "predicted_label": int(r["predicted_label"]),
                    "risk_tier": r["risk_tier"],
                    "source": "batch",
                }
                for r in out.to_dict(orient="records")
            ]
        )
        st.success(f"Logged {logged:,} predictions to the audit trail.")


# --------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------


def main() -> None:
    load_error = ""
    try:
        predictor: ChurnPredictor | None = get_predictor()
    except ChurnAIError as exc:
        predictor = None
        load_error = str(exc)
    choice = sidebar(predictor)
    key = next(k for k, v in PAGES.items() if v == choice)
    with st.container(key=f"page-{key}"):
        if key == "analytics":
            page_analytics()
            return
        if predictor is None:
            st.error(load_error, icon=":material/block:")
            return
        if key == "risk":
            page_risk_engine(predictor)
        elif key == "db":
            page_customer_db(predictor)
        else:
            page_batch(predictor)


main()
