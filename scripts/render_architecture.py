"""Render docs/figures/system_architecture.png.

Run: uv run python scripts/render_architecture.py
"""

from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.analytics.plotting import PALETTE, apply_theme
from src.config import get_settings

BOX_W, BOX_H = 2.35, 1.05


def box(ax, x, y, title, lines, color):  # type: ignore[no-untyped-def]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            BOX_W,
            BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#161B2E",
            edgecolor=color,
            linewidth=1.6,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + BOX_H - 0.08),
            BOX_W,
            0.08,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=color,
            edgecolor="none",
        )
    )
    ax.text(x + 0.14, y + BOX_H - 0.27, title, fontsize=10.5, weight="bold", color=PALETTE["text"])
    for i, line in enumerate(lines):
        ax.text(x + 0.14, y + BOX_H - 0.52 - i * 0.2, line, fontsize=7.8, color=PALETTE["muted"])


def arrow(ax, start, end, color=PALETTE["muted"], style="-|>", ls="-"):  # type: ignore[no-untyped-def]
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            color=color,
            lw=1.3,
            linestyle=ls,
            connectionstyle="arc3,rad=0.0",
        )
    )


def main() -> None:
    apply_theme()
    fig = Figure(figsize=(15, 7.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.text(
        0.4,
        6.75,
        "ChurnAI — system architecture",
        fontsize=17,
        weight="bold",
        color=PALETTE["text"],
    )
    ax.text(
        0.4,
        6.45,
        "One leakage-free pipeline shared by training, CI, the dashboard and batch scoring",
        fontsize=10,
        color=PALETTE["muted"],
    )

    # Lanes
    for y, label in ((3.55, "SERVING"), (0.35, "TRAINING & RELEASE")):
        ax.add_patch(
            FancyBboxPatch(
                (0.25, y),
                14.5,
                2.55,
                boxstyle="round,pad=0,rounding_size=0.2",
                facecolor="#10131D",
                edgecolor="#1F2537",
                linewidth=1,
            )
        )
        ax.text(
            0.45, y + 2.3, label, fontsize=8.5, weight="bold", color=PALETTE["violet"], alpha=0.9
        )

    v, c, e, a, r = (PALETTE[k] for k in ("violet", "cyan", "emerald", "amber", "crimson"))
    # Serving lane
    box(ax, 0.5, 4.3, "Inputs", ["UI form / presets", "Batch CSV upload", "Customer DB lookup"], c)
    box(
        ax,
        3.3,
        4.3,
        "Validation",
        ["Pydantic CustomerFeatures", "column contract", "clean_raw()"],
        v,
    )
    box(
        ax,
        6.1,
        4.3,
        "Calibrated pipeline",
        ["FeatureEngineer → RiskRatio", "Impute · Scale · OneHot", "champion + isotonic"],
        e,
    )
    box(
        ax,
        8.9,
        4.3,
        "Decision & XAI",
        ["cost-optimal threshold", "risk tiers · SHAP", "counterfactuals"],
        a,
    )
    box(
        ax,
        11.7,
        4.3,
        "Streamlit portal",
        ["Risk engine · Lookup", "Analytics · Batch", "glass UI + motion"],
        r,
    )
    for x in (2.85, 5.65, 8.45, 11.25):
        arrow(ax, (x, 4.82), (x + 0.45, 4.82))

    # Training lane
    box(ax, 0.5, 1.1, "Raw snapshot", ["Telco CSV (7,043 × 21)", "→ churn_clean.parquet"], c)
    box(ax, 3.3, 1.1, "Notebooks 01 · 02", ["EDA, danger-zone cohorts", "Welch · MWU · χ² · V"], v)
    box(
        ax,
        6.1,
        1.1,
        "Tournament (03 / CLI)",
        ["6 models · repeated 5-fold", "tuning · local CPU"],
        e,
    )
    box(ax, 8.9, 1.1, "Artifacts", ["production_pipeline.joblib", "model_metadata.json · docs/"], a)
    box(ax, 11.7, 1.1, "CI quality gate", ["ruff · mypy · pytest", "validate-thresholds"], r)
    for x in (2.85, 5.65, 8.45, 11.25):
        arrow(ax, (x, 1.62), (x + 0.45, 1.62))

    # Cross-lane links
    arrow(ax, (10.07, 2.15), (7.3, 4.3), color=e, ls="--")
    ax.text(8.0, 3.2, "loads", fontsize=8, color=e)
    ax.add_patch(
        FancyBboxPatch(
            (12.25, 3.62),
            1.8,
            0.5,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="#161B2E",
            edgecolor=v,
            linewidth=1.2,
        )
    )
    ax.text(
        13.15,
        3.87,
        "SQLite / PostgreSQL",
        fontsize=8,
        ha="center",
        va="center",
        color=PALETTE["text"],
        weight="bold",
    )
    arrow(ax, (12.87, 4.3), (12.87, 4.13), color=v, style="<|-|>")
    ax.text(14.15, 3.83, "audit log", fontsize=7.5, color=PALETTE["muted"])

    out = get_settings().figures_dir / "system_architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=PALETTE["bg"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
