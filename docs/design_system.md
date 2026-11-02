# ChurnAI Design System: Obsidian Glass

> Source of truth: `src/ui/styles.py` (tokens + CSS) and `src/ui/components.py` (components).
> Exported figures use the same palette via `src/analytics/plotting.py`.
> The native Streamlit theme in `.streamlit/config.toml` (`base = "dark"`) uses the same tokens,
> so built-in widgets, tables and Plotly charts match before any custom CSS is applied.

The interface should feel calm and precise. It has a deep obsidian canvas, translucent
"frosted glass" surfaces, and colour that is reserved for meaning (risk). Motion is used to
show hierarchy and causality, never as decoration. All effects are **pure CSS**, because
Streamlit strips `<script>`.

## 1. Design tokens

### 1.1 Colour

| Token | Hex / value | Usage |
| :--- | :--- | :--- |
| `--bg-0` Obsidian | `#0B0D13` | App background (gradient start), figure background |
| `--bg-1` Obsidian raised | `#121622` | Background gradient end, chart panels |
| `--card` Slate card | `#161B2E` | Solid card fallback, tooltip background |
| `--glass` | `rgba(22, 27, 46, 0.65)` | Glass surface fill |
| `--glass-strong` | `rgba(22, 27, 46, 0.85)` | Glass on busy backgrounds |
| `--stroke` | `rgba(255, 255, 255, 0.08)` | Hairline borders |
| `--stroke-hover` | `rgba(129, 140, 248, 0.45)` | Border on hover / focus |
| `--text` | `#E6E8F0` | Primary text |
| `--text-muted` | `#9AA3B8` | Secondary text, captions |
| `--text-faint` | `#646D82` | Labels, meta |
| `--violet` Accent | `#6366F1` | Primary actions, focus, brand |
| `--violet-soft` | `#818CF8` | Eyebrows, hover strokes |
| `--cyan` | `#22D3EE` | Secondary accent, gradients |

### 1.2 Semantic risk colours

| Tier | Token | Hex | Probability band | Treatment |
| :--- | :--- | :--- | :--- | :--- |
| Low | `--emerald` | `#10B981` | `p < 0.20` | Soft pulse on the badge dot |
| Medium | `--amber` | `#F59E0B` | `0.20 ≤ p < 0.45` | Static |
| High | `--orange` | `#F97316` | `0.45 ≤ p < 0.70` | Glow + pulse |
| Critical | `--crimson` | `#EF4444` | `p ≥ 0.70` | Glow + pulse |

Band edges come from `Settings.tier_*_max` and can be changed without touching the UI. Colour is
never the only signal: every badge also carries the tier name in text (WCAG 1.4.1).

**Contrast:** `--text` on `--bg-0` is ≈ 15:1 and `--text-muted` on `--bg-0` is ≈ 7.5:1, both
above WCAG AA for body text.

### 1.3 Typography

| Role | Font | Size / weight | Notes |
| :--- | :--- | :--- | :--- |
| Display (`.page-title`) | Plus Jakarta Sans | 2.05rem / 800, `-0.02em` | White → indigo → cyan gradient text |
| Eyebrow (`.eyebrow`) | Plus Jakarta Sans | 0.72rem / 700, `0.14em`, uppercase | `--violet-soft` |
| Section (`.section-title`) | Plus Jakarta Sans | 1.02rem / 700 | Icon + label |
| Body | Plus Jakarta Sans | 0.95–1rem / 400–500 | |
| Metric value | Plus Jakarta Sans | 1.9rem / 800, `tabular-nums` | Numbers stay aligned while counting |
| Code / IDs (`.mono`) | JetBrains Mono | 0.72–0.85rem | Model versions, customer IDs |

Fallback stack: `'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, sans-serif`.
Both fonts load from Google Fonts with `display=swap`.

### 1.4 Spacing, radius, elevation

| Token | Value |
| :--- | :--- |
| `--radius` | `12px` (cards, panels, expanders). Inputs and buttons use `10px`; badges use `999px`. |
| Card padding | `1.1rem 1.25rem` (HTML glass), `1rem 1.1rem` (widget containers) |
| `--shadow-1` (rest) | `0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.25)` |
| `--shadow-2` (hover) | `0 2px 4px rgba(0,0,0,.4), 0 18px 40px rgba(0,0,0,.45)` |
| Content max-width | `1400px` |

## 2. Glassmorphism spec

```css
.glass {
  background: rgba(22, 27, 46, 0.65);          /* --glass */
  backdrop-filter: blur(16px);                  /* --blur */
  border: 1px solid rgba(255, 255, 255, 0.08);  /* --stroke */
  border-radius: 12px;                          /* --radius */
  box-shadow: var(--shadow-1), inset 0 1px 0 rgba(255,255,255,.04);  /* top light edge */
}
.glass:hover {
  transform: translateY(-4px);                  /* --lift */
  border-color: rgba(129, 140, 248, 0.45);      /* --stroke-hover */
  box-shadow: var(--shadow-2), inset 0 0 24px rgba(99,102,241,.08);  /* soft inner glow */
}
```

* **Two kinds of glass.**
  * `.glass` is for static HTML fragments: metric cards, the gauge, counterfactuals and profiles.
  * `st.container(key="glass-…")` is for panels that hold real Streamlit widgets.
    Streamlit adds the class `st-key-glass-…`, and the same rules target
    `[class*="st-key-glass"]`.
* **Ambient background.** Two radial gradients sit over a 160° linear obsidian gradient. A fixed
  `::before` layer shifts hue by 35° over a **24 s** loop (`ambientShift`). It is subtle enough
  that it only registers subconsciously.

## 3. Motion & animation spec

### 3.1 Curves & durations

| Token | Value | Used for |
| :--- | :--- | :--- |
| `--ease` | `cubic-bezier(0.4, 0, 0.2, 1)` | Every transition and entrance |
| `--dur` | `250ms` | Hover / focus / press transitions |
| Entrance | `520ms` | `fadeInUp` on cards |
| Stagger | `80–120ms` per item, via the `--d` custom property | Metric rows, counterfactual cards |
| Page / tab switch | `260–280ms` opacity fade | `[class*="st-key-page-"]`, tab panels |
| Count-up / gauge fill | `1.1s` | Numbers and the risk arc |

### 3.2 Keyframes

| Name | Definition | Applied to |
| :--- | :--- | :--- |
| `fadeInUp` | `opacity 0 → 1`, `translateY(14px) → 0` | `.enter`, glass containers |
| `pageFade` | `opacity 0 → 1` | Page container, tab panels |
| `shimmer` | Highlight sweeps `background-position −468px → 468px` | KPI first-load sheen (2 passes), `.skeleton` loaders (infinite) |
| `pulseGlow` | Ring `box-shadow` 0 → 10px, fading out | Critical/emerald KPIs (3 pulses), tier badge dot |
| `ambientShift` | `hue-rotate 0 → 35deg → 0` | Background layer, 24 s infinite |
| `countUp` | Registered `@property --num <integer>` animates `0 → var(--target)` | `.countup` numbers (`counter()` renders the value) |
| `gaugeFill` | Registered `@property --gauge <number>` animates `100 → var(--gauge-target)` | SVG arc `stroke-dashoffset` (with `pathLength="100"`) |
| `dropBounce` | `scale 1 → 1.015 → 1` | File chip after a CSV is dropped |

### 3.3 Micro-interactions

| Element | Rest | Hover | Active / focus |
| :--- | :--- | :--- | :--- |
| Primary button | Violet gradient, `0 6px 18px` violet shadow | `scale(1.02)`, stronger shadow | `scale(0.98)` |
| Secondary button | Glass fill, hairline stroke | Violet tint | `scale(0.98)` |
| Download button | Emerald outline | Fills emerald, dark text, `scale(1.02)` | `scale(0.98)` |
| Text / select input | Translucent fill, hairline | — | Violet border + `0 0 0 3px` focus ring + `22px` glow |
| Slider thumb | — | `scale(1.15)` + 6px halo | — |
| Sidebar nav item | Transparent | Violet tint, `translateX(3px)` | Selected: gradient fill + violet stroke |
| Tabs | Muted text | Tinted background | Violet → cyan underline |
| Expander | Glass | Lift `-4px` + shadow | — |
| Upload dropzone | Dashed violet border | Brighter fill, 4px ring + glow | Dropped file chip bounces |

### 3.4 Accessibility

* `@media (prefers-reduced-motion: reduce)` reduces every animation and transition to 1 ms with a
  single iteration, so all content still appears in its final state.
* Count-up numbers set `aria-label` to the final value so screen readers never hear
  intermediate digits.
* Without `@property` support, `--num` and `--gauge` fall back to their initial values. For
  that reason the risk value is also always shown in text (the badge, tooltips and report).

## 4. Component guidelines

| Component | API (`src/ui/components.py`) | Rules |
| :--- | :--- | :--- |
| Page header | `page_header(eyebrow, title, subtitle)` | One per page. Subtitle at most two lines (`max-width: 820px`). |
| Metric badge / KPI | `metric_card(label, value, caption=, accent=, delay_ms=, effect=)` + `metric_row([...])` | 2–4 per row. Accent colour carries meaning (tier or category). `effect="pulse"` only for the single most important KPI. Values use `countup()` where they are integers. |
| Risk gauge | `risk_gauge(probability, tier, caption)` | Semicircle, emerald → amber → crimson gradient, with the tier badge below. Exactly one per scored customer. |
| Tier badge | `tier_badge(tier)` | Always uppercase text plus a coloured dot. Glows only for High/Critical. |
| Counterfactual card | `counterfactual_card(cf, delay_ms)` | Title is the action. Shows `before → after` and `−N pts`, and the bar width is relative reduction. Stagger by 100 ms. Always followed by the "not causal" caption. |
| Glass panel | `glass_container(key)` | Wrap groups of related widgets. Keys must be unique per page. |
| Profile grid | `profile_grid({label: value})` | Auto-fill, minimum 170px per cell. 8–12 fields. |
| SHAP waterfall | `shap_waterfall(explanation, top_k=10)` | Red pushes risk up, green pulls it down, violet is the total. Remaining features are collapsed into one bar. |
| Timeline | `history_timeline(df, threshold)` | Spline line, markers coloured by tier, dotted amber threshold. |
| Skeleton | `skeleton(lines)` | Empty states that are waiting for user input. |

### Layout & responsiveness

* Page grid: inputs on the left (`~1.05fr`), results on the right (`1fr`), `gap="large"`.
  Streamlit stacks the columns on narrow screens.
* Sliders hold continuous drivers (tenure, monthly charges). Selects hold categorical
  attributes. Secondary demographics and services go inside an expander so the primary drivers
  stay visible.
* Forms follow a two-column internal grid. Never put more than 6 controls in one column without
  a group heading.
* Data tables use `width="stretch"`, progress columns for probabilities and a fixed height
  (460px) for long lists.

### Charts (Plotly)

`style_figure()` applies a transparent paper and plot background, the design font, 6% white
gridlines, glass tooltips (`--card` background, violet border) and a 400 ms
`cubic-in-out` transition between data updates. The categorical sequence is violet → cyan →
emerald → amber → crimson. Red and green are reserved for churned/retained and risk direction.
