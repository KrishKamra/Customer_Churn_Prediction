"""Obsidian / frosted-glass design system for the Streamlit portal.

Pure CSS, no JavaScript: Streamlit strips ``<script>`` tags, so every motion effect
(entrance, hover lift, shimmer, pulse, count-up, gauge fill, page fade) is implemented with
CSS keyframes, transitions and registered custom properties (``@property``). All tokens
are documented in ``docs/design_system.md``.
"""

from __future__ import annotations

import streamlit as st

TOKENS: dict[str, str] = {
    "bg_0": "#0B0D13",
    "bg_1": "#121622",
    "card": "#161B2E",
    "glass": "rgba(22, 27, 46, 0.65)",
    "glass_strong": "rgba(22, 27, 46, 0.85)",
    "stroke": "rgba(255, 255, 255, 0.08)",
    "stroke_hover": "rgba(129, 140, 248, 0.45)",
    "text": "#E6E8F0",
    "text_muted": "#9AA3B8",
    "text_faint": "#646D82",
    "violet": "#6366F1",
    "violet_soft": "#818CF8",
    "cyan": "#22D3EE",
    "emerald": "#10B981",
    "amber": "#F59E0B",
    "orange": "#F97316",
    "crimson": "#EF4444",
    "ease": "cubic-bezier(0.4, 0, 0.2, 1)",
    "radius": "12px",
    "blur": "16px",
}

TIER_COLORS: dict[str, str] = {
    "Low": TOKENS["emerald"],
    "Medium": TOKENS["amber"],
    "High": TOKENS["orange"],
    "Critical": TOKENS["crimson"],
}

FONT_FAMILY = "'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, sans-serif"

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..600,0..1,0&display=block');

:root {
  --bg-0: %(bg_0)s; --bg-1: %(bg_1)s; --card: %(card)s;
  --glass: %(glass)s; --glass-strong: %(glass_strong)s;
  --stroke: %(stroke)s; --stroke-hover: %(stroke_hover)s;
  --text: %(text)s; --text-muted: %(text_muted)s; --text-faint: %(text_faint)s;
  --violet: %(violet)s; --violet-soft: %(violet_soft)s; --cyan: %(cyan)s;
  --emerald: %(emerald)s; --amber: %(amber)s; --orange: %(orange)s; --crimson: %(crimson)s;
  --ease: %(ease)s; --dur: 250ms; --radius: %(radius)s; --blur: %(blur)s;
  --lift: -4px;
  --shadow-1: 0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.25);
  --shadow-2: 0 2px 4px rgba(0,0,0,.4), 0 18px 40px rgba(0,0,0,.45);
}

/* ---------- Registered properties (enable numeric animation) ---------- */
@property --num { syntax: '<integer>'; initial-value: 0; inherits: false; }
@property --gauge { syntax: '<number>'; initial-value: 100; inherits: false; }

/* ---------- Keyframes ---------- */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(14px); }
                      to   { opacity: 1; transform: translateY(0); } }
@keyframes pageFade { from { opacity: 0; } to { opacity: 1; } }
@keyframes shimmer  { 0% { background-position: -468px 0; } 100% { background-position: 468px 0; } }
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent, var(--emerald)) 45%%, transparent); }
  50%      { box-shadow: 0 0 0 10px color-mix(in srgb, var(--accent, var(--emerald)) 0%%, transparent); } }
@keyframes ambientShift { 0% { filter: hue-rotate(0deg); opacity: .85; }
                          50% { filter: hue-rotate(35deg); opacity: 1; }
                          100% { filter: hue-rotate(0deg); opacity: .85; } }
@keyframes countUp { from { --num: 0; } to { --num: var(--target); } }
@keyframes gaugeFill { from { --gauge: 100; } to { --gauge: var(--gauge-target); } }
@keyframes dropBounce { 0% { transform: scale(1); } 40% { transform: scale(1.015); } 100% { transform: scale(1); } }

/* ---------- Shell ---------- */
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
  font-family: %(font)s !important;
}
.stApp {
  background: radial-gradient(1200px 600px at 10%% -10%%, rgba(99,102,241,.14), transparent 60%%),
              radial-gradient(900px 500px at 110%% 10%%, rgba(34,211,238,.08), transparent 60%%),
              linear-gradient(160deg, var(--bg-0) 0%%, var(--bg-1) 100%%);
  background-attachment: fixed;
  color: var(--text);
}
.stApp::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background: radial-gradient(700px 420px at 75%% 85%%, rgba(99,102,241,.10), transparent 70%%),
              radial-gradient(600px 380px at 20%% 70%%, rgba(16,185,129,.06), transparent 70%%);
  animation: ambientShift 24s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 1ms !important; animation-iteration-count: 1 !important;
                           transition-duration: 1ms !important; }
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1400px; }
h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -0.02em; }
p, label, span, li { color: var(--text); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--text-muted) !important; }
code { font-family: 'JetBrains Mono', monospace !important; }

/* Page / view switch fade (keyed containers remount on navigation) */
[class*="st-key-page-"] { animation: pageFade 280ms var(--ease) both; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(14,17,28,.92), rgba(11,13,19,.96));
  border-right: 1px solid var(--stroke);
  backdrop-filter: blur(var(--blur));
}
section[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: .55rem .75rem; border-radius: 10px; margin-bottom: .25rem; width: 100%%;
  border: 1px solid transparent;
  transition: all var(--dur) var(--ease);
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(99,102,241,.10); border-color: var(--stroke); transform: translateX(3px);
}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(99,102,241,.22), rgba(99,102,241,.04));
  border-color: var(--stroke-hover);
}

/* ---------- Glass primitives ---------- */
.glass, [class*="st-key-glass"] {
  background: var(--glass);
  backdrop-filter: blur(var(--blur)); -webkit-backdrop-filter: blur(var(--blur));
  border: 1px solid var(--stroke) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-1), inset 0 1px 0 rgba(255,255,255,.04);
  transition: transform var(--dur) var(--ease), box-shadow var(--dur) var(--ease),
              border-color var(--dur) var(--ease);
}
.glass { padding: 1.1rem 1.25rem; margin-bottom: .9rem; }
[class*="st-key-glass"] { padding: 1rem 1.1rem 1.1rem 1.1rem; }
.glass:hover, [class*="st-key-glass"]:hover {
  transform: translateY(var(--lift));
  border-color: var(--stroke-hover) !important;
  box-shadow: var(--shadow-2), inset 0 0 24px rgba(99,102,241,.08), inset 0 1px 0 rgba(255,255,255,.06);
}
.enter { opacity: 0; animation: fadeInUp 520ms var(--ease) forwards; animation-delay: var(--d, 0ms); }
[class*="st-key-glass"] { animation: fadeInUp 520ms var(--ease) both; }

/* ---------- Typography helpers ---------- */
.eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
           color: var(--violet-soft); margin: 0 0 .35rem 0; }
.page-title { font-size: 2.05rem; font-weight: 800; margin: 0; line-height: 1.15;
  background: linear-gradient(90deg, #fff 0%%, #c7d2fe 55%%, #67e8f9 100%%);
  -webkit-background-clip: text; background-clip: text; color: transparent !important; }
.page-sub { color: var(--text-muted); margin: .4rem 0 1.4rem 0; font-size: .98rem; max-width: 820px; }
.section-title { font-size: 1.02rem; font-weight: 700; margin: 0 0 .6rem 0; display: flex;
                 align-items: center; gap: .55rem; }
/* Material Symbols (same icon family Streamlit uses for widget icons). */
.msi { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal;
       font-size: 1.25em; line-height: 1; letter-spacing: normal; text-transform: none;
       display: inline-block; white-space: nowrap; direction: ltr; vertical-align: middle;
       font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
       font-variation-settings: 'FILL' 0, 'wght' 450, 'GRAD' 0, 'opsz' 24; }
.section-title .msi { color: var(--violet-soft); padding: .3rem; border-radius: 8px;
                      background: rgba(99,102,241,.12); border: 1px solid rgba(99,102,241,.22);
                      font-size: 1.15rem; }
.brand-mark .msi { color: #fff; font-size: 1.45rem; font-variation-settings: 'FILL' 1, 'wght' 500; }
.muted { color: var(--text-muted) !important; } .faint { color: var(--text-faint) !important; }
.mono { font-family: 'JetBrains Mono', monospace; }

/* ---------- Metric cards ---------- */
.metric { position: relative; overflow: hidden; }
.metric::after { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
                 background: var(--accent, var(--violet)); opacity: .9; }
.metric .label { font-size: .78rem; color: var(--text-muted); font-weight: 600;
                 text-transform: uppercase; letter-spacing: .08em; }
.metric .value { font-size: 1.9rem; font-weight: 800; margin-top: .25rem; color: var(--text);
                 font-variant-numeric: tabular-nums; }
.metric .caption { font-size: .8rem; color: var(--text-faint); margin-top: .15rem; }
.metric.pulse { animation: fadeInUp 520ms var(--ease) both, pulseGlow 2.4s var(--ease) 600ms 3; }
.metric.shimmer::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(90deg, transparent 0%%, rgba(255,255,255,.06) 50%%, transparent 100%%);
  background-size: 936px 100%%; animation: shimmer 1.6s linear 2;
}

/* ---------- Count-up numbers (CSS only) ---------- */
.countup { animation: countUp 1.1s var(--ease) forwards; counter-reset: num var(--num); }
.countup::after { content: counter(num) var(--suffix, ""); }
/* The literal value stays in the markup (accessible fallback); hide it once CSS is active. */
.countup > .v { display: none; }

/* ---------- Gauge ---------- */
.gauge-wrap { display: flex; flex-direction: column; align-items: center; }
.gauge-arc { stroke-dasharray: 100px; stroke-dashoffset: calc(var(--gauge) * 1px);
             animation: gaugeFill 1.1s var(--ease) forwards;
             filter: drop-shadow(0 0 8px var(--accent)); }
.gauge-value { font-size: 2.6rem; font-weight: 800; font-variant-numeric: tabular-nums;
               margin-top: -3.6rem; color: var(--text); }
.gauge-caption { color: var(--text-muted); font-size: .8rem; margin-top: .1rem; }

/* ---------- Badges ---------- */
.badge { display: inline-flex; align-items: center; gap: .4rem; padding: .28rem .7rem;
  border-radius: 999px; font-size: .78rem; font-weight: 700; letter-spacing: .04em;
  color: var(--accent); background: color-mix(in srgb, var(--accent) 14%%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent) 40%%, transparent); }
.badge .dot { width: 7px; height: 7px; border-radius: 50%%; background: var(--accent); }
.badge.glow { box-shadow: 0 0 18px color-mix(in srgb, var(--accent) 45%%, transparent); }
.badge.pulse .dot { animation: pulseGlow 1.8s var(--ease) infinite; }

/* ---------- Counterfactual cards ---------- */
.cf { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.cf .title { font-weight: 700; font-size: .95rem; }
.cf .desc { color: var(--text-muted); font-size: .85rem; margin-top: .15rem; }
.cf .delta { font-weight: 800; font-size: 1.25rem; color: var(--emerald); white-space: nowrap; }
.cf .bar { height: 4px; border-radius: 4px; background: rgba(255,255,255,.06); margin-top: .55rem; overflow: hidden; }
.cf .bar > span { display: block; height: 100%%; border-radius: 4px;
                  background: linear-gradient(90deg, var(--emerald), var(--cyan));
                  width: var(--w); animation: fadeInUp 600ms var(--ease) both; }

/* ---------- Skeleton loader ---------- */
.skeleton { height: 14px; border-radius: 6px; margin: .5rem 0;
  background: linear-gradient(90deg, rgba(255,255,255,.04) 8%%, rgba(255,255,255,.10) 18%%, rgba(255,255,255,.04) 33%%);
  background-size: 936px 100%%; animation: shimmer 1.3s linear infinite; }

/* ---------- Key-value profile grid ---------- */
.kv { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: .6rem; }
.kv div { background: rgba(255,255,255,.025); border: 1px solid var(--stroke); border-radius: 10px;
          padding: .55rem .7rem; transition: border-color var(--dur) var(--ease); }
.kv div:hover { border-color: var(--stroke-hover); }
.kv b { display: block; font-size: .7rem; color: var(--text-faint); text-transform: uppercase;
        letter-spacing: .08em; font-weight: 700; }
.kv span { font-weight: 600; font-size: .92rem; }

/* ---------- Native widgets ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 10px; font-weight: 700; letter-spacing: .01em;
  border: 1px solid var(--stroke-hover);
  background: linear-gradient(135deg, var(--violet) 0%%, #4F46E5 100%%); color: #fff !important;
  box-shadow: 0 6px 18px rgba(99,102,241,.25);
  transition: all var(--dur) var(--ease);
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  transform: scale(1.02); box-shadow: 0 10px 28px rgba(99,102,241,.40); border-color: var(--violet-soft);
}
.stButton > button:active, .stDownloadButton > button:active, .stFormSubmitButton > button:active {
  transform: scale(0.98);
}
.stButton > button[kind="secondary"] { background: rgba(255,255,255,.04); box-shadow: none;
                                       border-color: var(--stroke); }
.stButton > button[kind="secondary"]:hover { background: rgba(99,102,241,.12); }
.stDownloadButton > button { background: transparent !important; color: var(--emerald) !important;
                             border: 1.5px solid var(--emerald) !important; box-shadow: none; }
.stDownloadButton > button:hover { background: var(--emerald) !important; color: #06281D !important;
  transform: scale(1.02); box-shadow: 0 8px 26px rgba(16,185,129,.35); }

[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"],
.stNumberInput [data-baseweb="input"] {
  background: rgba(255,255,255,.035) !important; border-radius: 10px !important;
  border: 1px solid var(--stroke) !important; transition: all var(--dur) var(--ease);
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color: var(--violet-soft) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,.22), 0 0 22px rgba(99,102,241,.18) !important;
}
[data-baseweb="slider"] [role="slider"] { transition: transform var(--dur) var(--ease), box-shadow var(--dur) var(--ease); }
[data-baseweb="slider"] [role="slider"]:hover { transform: scale(1.15); box-shadow: 0 0 0 6px rgba(99,102,241,.2); }

.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--stroke); }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: .55rem 1rem;
  transition: all var(--dur) var(--ease); color: var(--text-muted); }
.stTabs [data-baseweb="tab"]:hover { background: rgba(99,102,241,.08); color: var(--text); }
.stTabs [aria-selected="true"] { color: var(--text) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--violet), var(--cyan)); }
.stTabs [data-baseweb="tab-panel"] { animation: pageFade 260ms var(--ease) both; }

[data-testid="stExpander"] details {
  background: var(--glass); border: 1px solid var(--stroke) !important; border-radius: var(--radius) !important;
  backdrop-filter: blur(var(--blur)); transition: all var(--dur) var(--ease);
}
[data-testid="stExpander"] details:hover { transform: translateY(var(--lift)); border-color: var(--stroke-hover) !important;
  box-shadow: var(--shadow-2); }

[data-testid="stFileUploaderDropzone"] {
  background: rgba(99,102,241,.05); border: 1.5px dashed rgba(129,140,248,.45) !important;
  border-radius: 14px; padding: 2rem 1rem; transition: all var(--dur) var(--ease);
}
[data-testid="stFileUploaderDropzone"]:hover {
  background: rgba(99,102,241,.12); border-color: var(--violet-soft) !important;
  box-shadow: 0 0 0 4px rgba(99,102,241,.12), 0 0 32px rgba(99,102,241,.18);
}
[data-testid="stFileUploaderFile"] { animation: dropBounce 420ms var(--ease); }

[data-testid="stProgress"] > div > div > div > div {
  background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
}
[data-testid="stDataFrame"] { border: 1px solid var(--stroke); border-radius: var(--radius); overflow: hidden; }
[data-testid="stAlert"] { border-radius: var(--radius); backdrop-filter: blur(8px); }
hr { border-color: var(--stroke) !important; }
"""


def build_css() -> str:
    """Return the fully token-substituted stylesheet."""
    css = _CSS
    for key, value in {**TOKENS, "font": FONT_FAMILY}.items():
        css = css.replace(f"%({key})s", value)
    return css.replace("%%", "%")


def inject_styles() -> None:
    """Inject the global stylesheet into the current Streamlit page."""
    # st.markdown keeps a raw <style> block intact; st.html's sanitiser drops a leading one.
    st.markdown(f"<style>{build_css()}</style>", unsafe_allow_html=True)
