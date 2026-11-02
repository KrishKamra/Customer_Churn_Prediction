"""Matplotlib / seaborn theme matching the dashboard design tokens (for exported figures)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PALETTE: dict[str, str] = {
    "bg": "#0B0D13",
    "panel": "#121622",
    "grid": "#262C3F",
    "text": "#E6E8F0",
    "muted": "#9AA3B8",
    "violet": "#6366F1",
    "cyan": "#22D3EE",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "crimson": "#EF4444",
}
CHURN_COLORS: dict[Any, str] = {
    0: PALETTE["emerald"],
    1: PALETTE["crimson"],
    "Retained": PALETTE["emerald"],
    "Churned": PALETTE["crimson"],
}
SEQUENCE: list[str] = [
    PALETTE["violet"],
    PALETTE["cyan"],
    PALETTE["emerald"],
    PALETTE["amber"],
    PALETTE["crimson"],
]


def apply_theme() -> None:
    """Apply the Obsidian figure theme globally."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["panel"],
            "savefig.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.6,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["text"],
            "legend.facecolor": PALETTE["panel"],
            "legend.edgecolor": PALETTE["grid"],
            "legend.labelcolor": PALETTE["text"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Plus Jakarta Sans", "Inter", "Segoe UI", "DejaVu Sans"],
            "axes.prop_cycle": cycler(color=SEQUENCE),
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def save(fig: Any, path: Path) -> Path:
    """Save a figure, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
