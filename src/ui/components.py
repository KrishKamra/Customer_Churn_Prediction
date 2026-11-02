"""Reusable glassmorphic UI components with motion.

Components return or render small HTML fragments styled by :mod:`src.ui.styles`. HTML is
emitted on a single line because Markdown treats indented lines as code blocks.
"""

from __future__ import annotations

import html as _html
from collections.abc import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.explainability import Counterfactual, LocalExplanation
from src.ui.styles import FONT_FAMILY, TIER_COLORS, TOKENS


def _compact(fragment: str) -> str:
    return "".join(line.strip() for line in fragment.splitlines())


def render(fragment: str) -> None:
    """Render an HTML fragment inline."""
    st.markdown(_compact(fragment), unsafe_allow_html=True)


def esc(value: object) -> str:
    return _html.escape(str(value))


# --------------------------------------------------------------------------------------
# Layout & typography
# --------------------------------------------------------------------------------------


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    render(
        f"""<div class="enter" style="--d:0ms">
        <p class="eyebrow">{esc(eyebrow)}</p>
        <h1 class="page-title">{esc(title)}</h1>
        <p class="page-sub">{esc(subtitle)}</p></div>"""
    )


def section_title(icon: str, text: str) -> None:
    render(f'<div class="section-title"><span>{icon}</span><span>{esc(text)}</span></div>')


def glass_container(key: str, border: bool = False) -> st.delta_generator.DeltaGenerator:
    """A real Streamlit container (can hold widgets) styled as a glass card."""
    return st.container(border=border, key=f"glass-{key}")


def skeleton(lines: int = 3) -> None:
    render(
        "".join(
            f'<div class="skeleton" style="width:{w}%"></div>' for w in [92, 78, 85, 64, 70][:lines]
        )
    )


# --------------------------------------------------------------------------------------
# Metrics, badges, gauge
# --------------------------------------------------------------------------------------


def countup(value: float, suffix: str = "") -> str:
    """CSS-only animated integer count-up (falls back to the final value if unsupported)."""
    target = int(round(value))
    return (
        f'<span class="countup" style="--target:{target};--suffix:\'{esc(suffix)}\'" '
        f'aria-label="{target}{esc(suffix)}"><span class="v">{target}{esc(suffix)}</span></span>'
    )


def metric_card(
    label: str,
    value: str,
    *,
    caption: str = "",
    accent: str = TOKENS["violet"],
    delay_ms: int = 0,
    effect: str = "shimmer",
) -> str:
    """Return the HTML for one KPI card (combine several with :func:`metric_row`)."""
    return (
        f'<div class="glass metric enter {effect}" style="--accent:{accent};--d:{delay_ms}ms">'
        f'<div class="label">{esc(label)}</div><div class="value">{value}</div>'
        f'<div class="caption">{esc(caption)}</div></div>'
    )


def metric_row(cards: Iterable[str]) -> None:
    cards = list(cards)
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards, strict=True):
        with col:
            render(card)


def tier_badge(tier: str) -> str:
    color = TIER_COLORS.get(tier, TOKENS["violet"])
    extra = " glow pulse" if tier in ("High", "Critical") else " pulse" if tier == "Low" else ""
    return (
        f'<span class="badge{extra}" style="--accent:{color}"><span class="dot"></span>'
        f"{esc(tier.upper())} RISK</span>"
    )


def risk_gauge(
    probability: float, tier: str, caption: str = "Calibrated churn probability"
) -> None:
    """Animated semicircular SVG gauge with count-up value."""
    color = TIER_COLORS.get(tier, TOKENS["violet"])
    pct = max(0.0, min(1.0, probability)) * 100
    gid = f"g{int(pct * 100)}"
    render(
        f"""<div class="glass enter gauge-wrap" style="--accent:{color};--d:60ms">
        <svg viewBox="0 0 220 130" width="100%" style="max-width:300px">
          <defs><linearGradient id="{gid}" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stop-color="{TOKENS["emerald"]}"/>
            <stop offset="45%" stop-color="{TOKENS["amber"]}"/>
            <stop offset="100%" stop-color="{TOKENS["crimson"]}"/></linearGradient></defs>
          <path d="M20 115 A90 90 0 0 1 200 115" fill="none" stroke="rgba(255,255,255,.07)"
                stroke-width="16" stroke-linecap="round"/>
          <path class="gauge-arc" pathLength="100" d="M20 115 A90 90 0 0 1 200 115" fill="none"
                stroke="url(#{gid})" stroke-width="16" stroke-linecap="round"
                style="--gauge-target:{100 - pct:.2f}"/>
        </svg>
        <div class="gauge-value">{countup(pct, "%")}</div>
        <div class="gauge-caption">{esc(caption)}</div>
        <div style="margin-top:.7rem">{tier_badge(tier)}</div></div>"""
    )


def counterfactual_card(cf: Counterfactual, delay_ms: int = 0) -> None:
    reduction = cf.baseline_probability - cf.new_probability
    width = 0 if cf.baseline_probability <= 0 else reduction / cf.baseline_probability * 100
    render(
        f"""<div class="glass enter" style="--d:{delay_ms}ms">
        <div class="cf"><div><div class="title">{esc(cf.action)}</div>
        <div class="desc">Churn risk {cf.baseline_probability:.0%} →&nbsp;
        <b>{cf.new_probability:.0%}</b></div></div>
        <div class="delta">−{reduction * 100:.0f} pts</div></div>
        <div class="cf"><div class="bar" style="flex:1">
        <span style="--w:{width:.0f}%"></span></div></div>
        </div>"""
    )


def profile_grid(items: dict[str, object]) -> None:
    cells = "".join(f"<div><b>{esc(k)}</b><span>{esc(v)}</span></div>" for k, v in items.items())
    render(f'<div class="glass enter kv" style="--d:80ms">{cells}</div>')


# --------------------------------------------------------------------------------------
# Plotly theming & charts
# --------------------------------------------------------------------------------------


def style_figure(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "color": TOKENS["text"], "size": 13},
        margin={"l": 10, "r": 10, "t": 64, "b": 10},
        hoverlabel={
            "bgcolor": TOKENS["card"],
            "bordercolor": TOKENS["violet"],
            "font": {"family": FONT_FAMILY, "color": TOKENS["text"]},
        },
        showlegend=legend,
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "orientation": "h",
            "y": 1.02,
            "x": 0,
            "yanchor": "bottom",
        },
        transition={"duration": 400, "easing": "cubic-in-out"},
    )
    grid = "rgba(255,255,255,0.06)"
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid)
    return fig


def shap_waterfall(explanation: LocalExplanation, top_k: int = 10) -> go.Figure:
    """Horizontal Plotly waterfall of the largest local SHAP contributions."""
    top = explanation.top(top_k)
    rest = explanation.contributions["shap"].sum() - top["shap"].sum()
    labels = [*top["label"].tolist()[::-1]]
    values = [*top["shap"].tolist()[::-1]]
    if abs(rest) > 1e-9:
        labels.insert(0, f"{len(explanation.contributions) - len(top)} other features")
        values.insert(0, float(rest))
    unit = "log-odds" if explanation.output_space == "log-odds" else "probability"
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=["Baseline (avg. customer)", *labels, "Model output"],
            x=[explanation.base_value, *values, 0],
            measure=["absolute", *["relative"] * len(values), "total"],
            increasing={"marker": {"color": TOKENS["crimson"]}},
            decreasing={"marker": {"color": TOKENS["emerald"]}},
            totals={"marker": {"color": TOKENS["violet"]}},
            connector={"line": {"color": "rgba(255,255,255,0.15)", "width": 1}},
            hovertemplate="%{y}<br>contribution: %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(title={"text": f"Local SHAP attribution ({unit})", "font": {"size": 14}})
    return style_figure(fig, height=60 + 34 * (len(labels) + 2), legend=False)


def tier_color_map() -> dict[str, str]:
    return dict(TIER_COLORS)


def history_timeline(history: pd.DataFrame, threshold: float) -> go.Figure:
    """Interactive timeline of a customer's scored risk over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["created_at"],
            y=history["churn_probability"],
            mode="lines+markers",
            line={"color": TOKENS["violet_soft"], "width": 2.5, "shape": "spline"},
            marker={
                "size": 11,
                "color": [TIER_COLORS.get(t, TOKENS["violet"]) for t in history["risk_tier"]],
                "line": {"width": 2, "color": TOKENS["bg_0"]},
            },
            customdata=history[["risk_tier", "model_version", "source"]].to_numpy(),
            hovertemplate=(
                "<b>%{y:.1%}</b> · %{customdata[0]}<br>%{x|%Y-%m-%d %H:%M:%S}"
                "<br>model %{customdata[1]} · via %{customdata[2]}<extra></extra>"
            ),
            name="Churn probability",
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color=TOKENS["amber"],
        annotation_text=f"decision threshold {threshold:.2f}",
        annotation_font_color=TOKENS["amber"],
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return style_figure(fig, height=340, legend=False)
