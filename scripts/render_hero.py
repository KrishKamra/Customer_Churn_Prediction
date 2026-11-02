"""Render the animated README hero (docs/assets/hero.svg) from real model artifacts.

A terminal-window card that combines ASCII art (logo, cohort bars) with live SVG motion
(gauge sweep, self-drawing holdout ROC curve, typed prompt, blinking cursor). Pure SVG +
CSS/SMIL, so it animates inside GitHub's ``<img>`` sandbox without JavaScript.

Run: uv run python scripts/render_hero.py
"""

from __future__ import annotations

import json
from html import escape

import numpy as np
from sklearn.metrics import roc_curve

from src.analytics.statistics import cohort_churn
from src.config import get_settings
from src.features.engineer import clean_raw, load_raw, split_features_target
from src.models.evaluate import holdout_split
from src.models.inference import ChurnPredictor

LOGO = [
    " ██████╗██╗  ██╗██╗   ██╗██████╗ ███╗   ██╗ █████╗ ██╗",
    "██╔════╝██║  ██║██║   ██║██╔══██╗████╗  ██║██╔══██╗██║",
    "██║     ███████║██║   ██║██████╔╝██╔██╗ ██║███████║██║",
    "██║     ██╔══██║██║   ██║██╔══██╗██║╚██╗██║██╔══██║██║",
    "╚██████╗██║  ██║╚██████╔╝██║  ██║██║ ╚████║██║  ██║██║",
    " ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝",
]
W, H = 1200, 470
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Helvetica, Arial, sans-serif"
CYCLE = 9  # seconds per animation loop


def _roc_points(n: int = 48) -> tuple[list[tuple[float, float]], float]:
    s = get_settings()
    predictor = ChurnPredictor.load()
    _, test = holdout_split(clean_raw(load_raw(s.raw_data_path)), s)
    X, y = split_features_target(test)
    fpr, tpr, _ = roc_curve(y, predictor.predict_proba(X))
    grid = np.linspace(0, 1, n)
    return list(zip(grid, np.interp(grid, fpr, tpr), strict=True)), float(
        predictor.metadata["holdout"]["roc_auc"]
    )


def _vector_logo(x0: float, y0: float, cw: float = 11.3, ch: float = 20.5) -> str:
    """Render the ANSI-Shadow ASCII art as crisp vectors, one glyph cell at a time.

    ``█`` cells become filled squares (slightly overlapping, so no seams) and the
    double-line box glyphs (``═ ║ ╗ ╔ ╚ ╝``) become double strokes, preserving the look
    of the terminal font without depending on which monospace font the viewer has.
    """
    gap = 2.2  # half-distance between the two strokes of a double line
    blocks: list[str] = []
    strokes: list[str] = []
    for r, line in enumerate(LOGO):
        # Merge each horizontal run of █ into one rect so no anti-aliasing seams appear.
        c = 0
        while c < len(line):
            if line[c] == "█":
                end = c
                while end < len(line) and line[end] == "█":
                    end += 1
                blocks.append(
                    f'<rect x="{x0 + c * cw:.2f}" y="{y0 + r * ch:.2f}" '
                    f'width="{(end - c) * cw + 0.4:.2f}" height="{ch + 0.4:.2f}"/>'
                )
                c = end
            else:
                c += 1
        for c, char in enumerate(line):
            x, y = x0 + c * cw, y0 + r * ch
            mx, my = x + cw / 2, y + ch / 2
            if char == "═":
                strokes += [f"M{x:.2f} {my + d:.2f}H{x + cw:.2f}" for d in (-gap, gap)]
            elif char == "║":
                strokes += [f"M{mx + d:.2f} {y:.2f}V{y + ch:.2f}" for d in (-gap, gap)]
            elif char in "╗╔╚╝":
                sx = 1 if char in "╔╚" else -1  # horizontal arm points right (+1) / left (-1)
                sy = 1 if char in "╗╔" else -1  # vertical arm points down (+1) / up (-1)
                ex, ey = mx + sx * cw / 2, my + sy * ch / 2
                strokes.append(f"M{ex:.2f} {my - sy * gap:.2f}H{mx - sx * gap:.2f}V{ey:.2f}")
                strokes.append(f"M{ex:.2f} {my + sy * gap:.2f}H{mx + sx * gap:.2f}V{ey:.2f}")
    width = max(len(line) for line in LOGO) * cw
    grad = (
        f'<linearGradient id="logoU" gradientUnits="userSpaceOnUse" x1="{x0}" y1="0" '
        f'x2="{x0 + width:.1f}" y2="0" spreadMethod="reflect">'
        '<stop offset="0" stop-color="#C7D2FE"/><stop offset=".35" stop-color="#818CF8"/>'
        '<stop offset=".7" stop-color="#6366F1"/><stop offset="1" stop-color="#22D3EE"/>'
        f'<animateTransform attributeName="gradientTransform" type="translate" '
        f'values="0 0;{width:.1f} 0" dur="6s" repeatCount="indefinite"/></linearGradient>'
    )
    return (
        f"<defs>{grad}</defs>"
        '<g opacity="0">'
        f'<animate attributeName="opacity" values="0;1;1;0" {_keyframes(0.27)} dur="{CYCLE}s" '
        'repeatCount="indefinite"/>'
        f'<path d="{"".join(strokes)}" fill="none" stroke="#818CF8" stroke-opacity=".6" '
        'stroke-width="1.15" stroke-linecap="square"/>'
        f'<g fill="url(#logoU)">{"".join(blocks)}</g></g>'
    )


def _keyframes(show_at: float, hide_at: float = 0.94) -> str:
    """keyTimes/values helper for a looping reveal (0 → 1 → hold → 0)."""
    return f'keyTimes="0;{show_at:.3f};{hide_at:.3f};1"'


def build() -> str:
    s = get_settings()
    meta = json.loads(s.metadata_path.read_text(encoding="utf-8"))
    h = meta["holdout"]
    cohorts = cohort_churn(clean_raw(load_raw(s.raw_data_path)))
    roc, auc = _roc_points()

    parts: list[str] = []
    add = parts.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'role="img" aria-labelledby="t d">'
        '<title id="t">ChurnAI</title>'
        '<desc id="d">Animated terminal card: ChurnAI ASCII logo, churn by tenure cohort, '
        f"risk gauge and holdout ROC curve (AUC {auc:.3f}).</desc>"
    )
    add(f"""<style>
  .mono {{ font-family: {MONO}; }}
  .sans {{ font-family: {SANS}; }}
  .faint {{ fill: #646D82; }} .muted {{ fill: #9AA3B8; }} .txt {{ fill: #E6E8F0; }}
  .cursor {{ animation: blink 1.05s steps(1) infinite; }}
  @keyframes blink {{ 50% {{ opacity: 0; }} }}
  .blob1 {{ animation: drift1 18s ease-in-out infinite alternate; }}
  .blob2 {{ animation: drift2 22s ease-in-out infinite alternate; }}
  @keyframes drift1 {{ to {{ transform: translate(120px, 40px); }} }}
  @keyframes drift2 {{ to {{ transform: translate(-140px, -30px); }} }}
  .pulse {{ animation: pulse 2.2s ease-in-out infinite; transform-origin: center;
            transform-box: fill-box; }}
  @keyframes pulse {{ 50% {{ opacity: .55; }} }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>""")
    add("""<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#0B0D13"/><stop offset="1" stop-color="#121622"/></linearGradient>
  <linearGradient id="logo" x1="0" x2="1" y1="0" y2="0" gradientUnits="objectBoundingBox">
    <stop offset="0" stop-color="#A5B4FC"/><stop offset=".45" stop-color="#6366F1"/>
    <stop offset=".75" stop-color="#22D3EE"/><stop offset="1" stop-color="#A5B4FC"/>
    <animateTransform attributeName="gradientTransform" type="translate" values="-1 0;1 0"
      dur="7s" repeatCount="indefinite"/></linearGradient>
  <linearGradient id="gauge" x1="0" x2="1"><stop offset="0" stop-color="#10B981"/>
    <stop offset=".5" stop-color="#F59E0B"/><stop offset="1" stop-color="#EF4444"/></linearGradient>
  <linearGradient id="rocfill" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0" stop-color="#6366F1" stop-opacity=".35"/>
    <stop offset="1" stop-color="#6366F1" stop-opacity="0"/></linearGradient>
  <radialGradient id="glowA"><stop offset="0" stop-color="#6366F1" stop-opacity=".28"/>
    <stop offset="1" stop-color="#6366F1" stop-opacity="0"/></radialGradient>
  <radialGradient id="glowB"><stop offset="0" stop-color="#22D3EE" stop-opacity=".16"/>
    <stop offset="1" stop-color="#22D3EE" stop-opacity="0"/></radialGradient>
  <clipPath id="frame"><rect x="0" y="0" width="1200" height="470" rx="18"/></clipPath>
  <filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
    <feGaussianBlur stdDeviation="4"/></filter>
</defs>""")

    # --- window chrome & ambient light -------------------------------------------------
    add('<g clip-path="url(#frame)">')
    add(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
    add('<ellipse class="blob1" cx="260" cy="120" rx="360" ry="220" fill="url(#glowA)"/>')
    add('<ellipse class="blob2" cx="980" cy="380" rx="340" ry="200" fill="url(#glowB)"/>')
    add('<rect width="1200" height="40" fill="#0E111A" opacity=".9"/>')
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        add(f'<circle cx="{26 + i * 20}" cy="20" r="6" fill="{c}"/>')
    add(
        '<text x="600" y="25" text-anchor="middle" class="mono faint" font-size="12.5">'
        "churnai — ~/Customer_Churn_Prediction</text>"
    )
    add('<line x1="0" y1="40" x2="1200" y2="40" stroke="#FFFFFF" stroke-opacity=".07"/>')
    add("</g>")
    add(
        f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="18" fill="none" '
        'stroke="#FFFFFF" stroke-opacity=".09"/>'
    )

    # --- left: terminal ----------------------------------------------------------------
    x0 = 40
    cmd = "uv run streamlit run src/ui/app.py"
    cw = 8.4  # char advance at 14px
    # typing effect: the command is revealed through a clip that widens each loop
    typed_w = (len(cmd) + 1) * cw
    add(
        f'<clipPath id="typed"><rect x="{x0 + 95}" y="58" height="26" width="0">'
        f'<animate attributeName="width" values="0;{typed_w};{typed_w};0" {_keyframes(0.22, 0.97)} '
        f'dur="{CYCLE}s" repeatCount="indefinite"/></rect></clipPath>'
    )
    add(
        f'<text x="{x0}" y="76" class="mono" font-size="14"><tspan fill="#10B981">➜</tspan>'
        '<tspan fill="#22D3EE" dx="8">churnai</tspan></text>'
    )
    add(
        f'<text x="{x0 + 95}" y="76" class="mono txt" font-size="14" clip-path="url(#typed)">'
        f"{escape(cmd)}</text>"
    )
    add(_vector_logo(x0, 92))
    add(
        f'<text x="{x0}" y="257" class="sans muted" font-size="15">Calibrated churn risk · '
        "SHAP explanations · counterfactual retention offers · statistical proof</text>"
    )

    # ASCII cohort bars that grow each loop
    add(
        f'<text x="{x0}" y="294" class="mono faint" font-size="12.5" xml:space="preserve">'
        "tenure cohort   churn rate (n = 7,043)</text>"
    )
    max_rate = float(cohorts["churn_rate"].max())
    for i, r in enumerate(cohorts.itertuples()):
        y = 318 + i * 23
        n_blocks = max(1, round(r.churn_rate / max_rate * 30))
        bar = "█" * n_blocks
        color = "#EF4444" if r.tenure_cohort == "0-6m" else "#818CF8"
        bar_w = n_blocks * 8.4
        cid = f"bar{i}"
        start = 0.36 + i * 0.035
        add(
            f'<clipPath id="{cid}"><rect x="{x0 + 110}" y="{y - 14}" height="18" width="0">'
            f'<animate attributeName="width" values="0;{bar_w + 60};{bar_w + 60};0" '
            f'{_keyframes(start + 0.08)} dur="{CYCLE}s" repeatCount="indefinite"/></rect></clipPath>'
        )
        add(
            f'<text x="{x0}" y="{y}" class="mono muted" font-size="14" xml:space="preserve">'
            f"{r.tenure_cohort:<8}</text>"
        )
        add(
            f'<text x="{x0 + 110}" y="{y}" class="mono" font-size="14" clip-path="url(#{cid})">'
            f'<tspan fill="{color}">{bar}</tspan>'
            f'<tspan dx="10" class="txt" font-weight="700">{r.churn_rate:.0%}</tspan></text>'
        )
    add(
        f'<text x="{x0}" y="444" class="mono" font-size="14"><tspan fill="#10B981">➜</tspan>'
        '<tspan fill="#22D3EE" dx="8">churnai</tspan>'
        '<tspan class="cursor" dx="8" fill="#E6E8F0">█</tspan></text>'
    )

    # --- right: glass card --------------------------------------------------------------
    cx, cy, cwid, chei = 770, 62, 392, 380
    add(
        f'<rect x="{cx}" y="{cy}" width="{cwid}" height="{chei}" rx="14" fill="#161B2E" '
        'fill-opacity=".72" stroke="#FFFFFF" stroke-opacity=".09"/>'
    )
    add(
        f'<text x="{cx + 20}" y="{cy + 30}" class="sans" font-size="11" font-weight="700" '
        'letter-spacing="2" fill="#818CF8">LIVE RISK SCORE</text>'
    )
    # gauge
    gx, gy, gr = cx + 110, cy + 130, 74
    arc = f"M{gx - gr} {gy} A{gr} {gr} 0 0 1 {gx + gr} {gy}"
    target = 100 - 93
    add(
        f'<path d="{arc}" fill="none" stroke="#FFFFFF" stroke-opacity=".08" stroke-width="13" '
        'stroke-linecap="round"/>'
    )
    add(
        f'<path d="{arc}" pathLength="100" fill="none" stroke="url(#gauge)" stroke-width="13" '
        'stroke-linecap="round" stroke-dasharray="100" stroke-dashoffset="100" filter="url(#soft)" '
        'opacity=".55">'
        f'<animate attributeName="stroke-dashoffset" values="100;{target};{target};100" '
        f'{_keyframes(0.3)} dur="{CYCLE}s" repeatCount="indefinite"/></path>'
    )
    add(
        f'<path d="{arc}" pathLength="100" fill="none" stroke="url(#gauge)" stroke-width="13" '
        'stroke-linecap="round" stroke-dasharray="100" stroke-dashoffset="100">'
        f'<animate attributeName="stroke-dashoffset" values="100;{target};{target};100" '
        f'{_keyframes(0.3)} dur="{CYCLE}s" repeatCount="indefinite"/></path>'
    )
    add(
        f'<text x="{gx}" y="{gy - 8}" text-anchor="middle" class="sans txt" font-size="34" '
        'font-weight="800">93%</text>'
    )
    add(
        f'<text x="{gx}" y="{gy + 16}" text-anchor="middle" class="sans muted" font-size="11">'
        "p(churn) · high-risk newcomer</text>"
    )
    add(
        f'<g class="pulse"><rect x="{gx - 52}" y="{gy + 28}" width="104" height="22" rx="11" '
        'fill="#EF4444" fill-opacity=".14" stroke="#EF4444" stroke-opacity=".55"/>'
        f'<text x="{gx}" y="{gy + 43}" text-anchor="middle" class="sans" font-size="10.5" '
        'font-weight="800" letter-spacing="1.2" fill="#EF4444">CRITICAL RISK</text></g>'
    )
    # counterfactual hint
    add(
        f'<text x="{cx + 222}" y="{cy + 92}" class="sans muted" font-size="11">Best retention offer</text>'
    )
    add(
        f'<text x="{cx + 222}" y="{cy + 114}" class="sans txt" font-size="13.5" font-weight="700">'
        "Two-year contract</text>"
    )
    add(
        f'<text x="{cx + 222}" y="{cy + 140}" class="sans" font-size="22" font-weight="800" '
        'fill="#10B981">−53 pts</text>'
    )
    add(
        f'<text x="{cx + 222}" y="{cy + 160}" class="sans faint" font-size="11">93% → 40% risk</text>'
    )

    # ROC curve (real holdout data) drawing itself
    rx0, ry0, rw, rh = cx + 22, cy + 205, 210, 120
    pts = " ".join(f"{rx0 + f * rw:.1f},{ry0 + rh - t * rh:.1f}" for f, t in roc)
    add(
        f'<rect x="{rx0}" y="{ry0}" width="{rw}" height="{rh}" fill="none" stroke="#FFFFFF" '
        'stroke-opacity=".08"/>'
    )
    add(
        f'<line x1="{rx0}" y1="{ry0 + rh}" x2="{rx0 + rw}" y2="{ry0}" stroke="#646D82" '
        'stroke-dasharray="4 4" stroke-width="1"/>'
    )
    add(
        f'<polygon points="{rx0},{ry0 + rh} {pts} {rx0 + rw},{ry0 + rh}" fill="url(#rocfill)" '
        'opacity="0">'
        f'<animate attributeName="opacity" values="0;1;1;0" {_keyframes(0.62)} dur="{CYCLE}s" '
        'repeatCount="indefinite"/></polygon>'
    )
    add(
        f'<polyline points="{pts}" fill="none" stroke="#818CF8" stroke-width="2.4" '
        'stroke-linejoin="round" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">'
        f'<animate attributeName="stroke-dashoffset" values="1;0;0;1" {_keyframes(0.6)} '
        f'dur="{CYCLE}s" repeatCount="indefinite"/></polyline>'
    )
    add(
        f'<text x="{rx0}" y="{ry0 - 10}" class="sans muted" font-size="11">Holdout ROC · '
        f'<tspan class="txt" font-weight="700">AUC {auc:.3f}</tspan></text>'
    )
    # metric chips
    chips = [
        ("PR-AUC", f"{h['pr_auc']:.3f}", "#818CF8"),
        ("Recall", f"{h['recall']:.1%}", "#10B981"),
        ("Brier", f"{h['brier']:.3f}", "#22D3EE"),
        ("Threshold", f"{meta['decision_threshold']:.2f}", "#F59E0B"),
    ]
    for i, (label, value, color) in enumerate(chips):
        x = rx0 + rw + 16
        y = ry0 - 4 + i * 32
        add(
            f'<rect x="{x}" y="{y}" width="120" height="26" rx="8" fill="{color}" '
            f'fill-opacity=".10" stroke="{color}" stroke-opacity=".35"/>'
            f'<text x="{x + 10}" y="{y + 17}" class="sans" font-size="10.5" fill="#9AA3B8">{label}'
            f'<tspan dx="6" font-weight="800" font-size="12" fill="{color}">{value}</tspan></text>'
        )
    add("</svg>")
    return "\n".join(parts)


def main() -> None:
    out = get_settings().docs_dir / "assets" / "hero.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
