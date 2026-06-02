# SPROUTS Visualization Redesign — Design Spec

**Status:** approved (verbal) 2026-06-01. Known issue #1 in `docs/KNOWN_ISSUES.md`.

**Problem:** the embedded Dash dashboards render and function correctly, but the layout is bad — content sits in a centered, padded, max-width Bootstrap (`dbc`) column, so the chart is boxed into a narrow strip while the Qt panel has width to spare; controls eat a tall card; the Growth-Lines hover images sit *below* the chart, unreachable without page scroll (and the embedded page does not scroll usefully). Functionally fine, visually cramped.

**Goal:** rebuild each dashboard's layout to the SPROUTS `VizView` mockup — a thin header, a collapsible control strip, thin metric chips, and a **dominant full-width chart** that **auto-fits the available panel width** as it changes; put Growth-Lines images **beside** the chart.

**Hard guardrail:** layout-tree + CSS only. NO change to computed data, trace values, `customdata` arrays/order, callback `@Output`/`Input`/`State`, component `id`s, or option `value`s. This is approach **A1** (Dash layout rebuild on the SPROUTS card system).

---

## Architecture

Both dashboards keep their current embedding: a Dash app served by a local Werkzeug thread, loaded into a `QWebEngineView` that `visualization_manager` swaps into the shell's Visualize stage (`right_panel` idx 2). Only each app's `layout()` (and `app/assets/style.css`) changes. `dash_visualizations.py` chart bodies keep their data; only `update_layout` size/`autosize` flags and figure `height`/`width` are adjusted per §"Sizing".

### Page anatomy (top → bottom), full-bleed flex column at `100vh`

```
┌────────────────────────────────────────────────────────────┐
│ INTERACTIVE DASHBOARD            [Stacked|Time|Lines|Faceted]│  header band (~52px, fixed)
│ Root Length — trial overview                        [Export] │
├────────────────────────────────────────────────────────────┤
│ ▸ Tube & date range   (3 tubes · L1–L20)            collapsed│  control strip (~40px; expands)
├────────────────────────────────────────────────────────────┤
│  [Images measured 248] [Mean 42.1 mm] [Timepoints 6]         │  metric chips (~56px, fixed)
├────────────────────────────────────────────────────────────┤
│                                                              │
│                    CHART CARD  (flex:1, min-height:0)        │  fills remaining space,
│                    full panel WIDTH                          │  full-bleed width
│                                                              │
└────────────────────────────────────────────────────────────┘
```

The page root is `display:flex; flex-direction:column; height:100vh` (the WebEngine viewport == the Qt panel). Header / control-strip / chips take natural height; the chart card is `flex:1; min-height:0` and **full panel width** (no `mx-auto`, no max-width cap, no side padding).

---

## Components

New CSS classes ported into `app/assets/style.css` (from the design `styles.css`), token-based:
- `.viz-page` — the 100vh flex column, full-bleed.
- `.viz-header` — eyebrow (`.viz-eyebrow`, mono uppercase faint) + title (`.viz-title`) left; view segmented control + Export right.
- `.viz-controls` — the collapsible strip: a one-line toggle row (`▸ Tube & date range` + selection summary) over a `dbc.Collapse` holding the EXISTING controls (tube multiselect, From/To inputs, Apply). Collapsed by default.
- `.viz-metrics` / `.metric` — thin horizontal metric chips (mono value + faint label).
- `.viz-chart-card` — `flex:1; min-height:0`, full width, holds the `dcc.Graph`.
- `.viz-lines-row` — Growth-Lines two-pane: `display:flex`; chart pane `flex:1; min-width:0`; image pane `.viz-image-panel` fixed `~300px`, `overflow-y:auto`.

All existing component `id`s are preserved verbatim: `view-selector` (now in the header, still `dbc.RadioItems` with the same values), `main-graph`, `image-container`, `selected-tubes-display`, `tube-date-availability`, `faceted-selection-info`, the From/To inputs, Apply button, etc. Only their **wrapping `html.Div` structure/classes** change.

---

## Sizing (the core fix — width-first, fluid)

1. **Full-bleed chart card.** Drop the centered/max-width/padded Bootstrap column; the chart card spans 100% of the panel width.
2. **No pinned figure `width`.** `dcc.Graph` style `{width:"100%"}` + existing `config responsive:True`. Plotly auto-fits width to the card.
3. **Auto-resize for free.** Qt window/splitter resize → WebEngine viewport resize → page `resize` event → `responsive:True` re-fits chart width to the new panel size.
4. **Clientside refit on in-page toggles.** A `clientside_callback` fires `Plotly.Plots.resize(document.getElementById('main-graph'))` (or `window.dispatchEvent(new Event('resize'))`) when the control strip collapse state or `view-selector` changes — so layout changes that aren't window resizes still refit. New clientside callback only; it writes a dummy/no-op `Output` (e.g. a hidden `Store`'s `data`) and touches NO figure `Output`.
5. **Height stays comfortable/stable** — the recovered space is WIDTH. Keep a sensible chart height (e.g. the card's height or a stable large value); do not also stretch height.
6. **Faceted depth profile:** keep its computed tall height for subplot readability; the extra **width** (full-bleed) is the win for its side-by-side subplots. If its height exceeds the card, the card scrolls internally (`overflow-y:auto`) — width still auto-fits.

---

## Growth-Lines image adjacency

When `view-selector == "lines"` (Length dashboard only) and images exist, render `.viz-lines-row`: **chart pane left (`flex:1`)** + **`image-container` right (`~300px`, internal scroll)** so hover/selected images sit adjacent and always visible without page scroll. `image-container` keeps its `id` and the `display_hover_images` callback `Output` (children/style) unchanged — it is only relocated into the right column and shown only for the Lines view. Other views: full-width chart, no side panel. The Area dashboard has no image strip → always full-width.

---

## Per-dashboard differences

- **Length (`dash_app.py`):** views Stacked / Time / Lines / Faceted. Lines → side image panel; Faceted → full-width + internal scroll.
- **Area (`dash_app_area.py`):** views Stacked / Time / Profile. No image strip → always full-width chart.
- Shared CSS in `app/assets/style.css`; the layout skeleton is duplicated per app (they are separate Dash apps) but follows the same class structure.

---

## Error / empty states

Keep the existing empty/error figures and "No Data" annotations (already retinted to tokens in PR1). They render inside the same full-width `.viz-chart-card`. No new error handling.

---

## Testing

- `pytest -q` must stay green (currently 70) after every change — the existing viz tests assert the data/callback contract.
- Add NO assertions that pin layout pixels. If feasible, add a smoke test importing each Dash app and asserting the key `id`s still exist in `app.layout` (string scan) and `view-selector` options/values unchanged.
- Human GUI verify (only confirmable in the running app): chart uses full panel width, widens when the window/splitter widens, control strip collapses, Growth-Lines images sit beside the chart, faceted uses the full width.

---

## Delegation plan (implementation)

Sized to modification level:
- **CSS port** (`app/assets/style.css` — add the `.viz-*` classes): mechanical → `cavecrew-builder` or a single general-purpose task.
- **`dash_app.py` layout rebuild** (header + collapsible strip + chips + full-bleed chart + Lines two-pane): multi-section, judgment → `general-purpose` implementer (one agent, sequential, pytest-gated, ids/callbacks frozen).
- **`dash_app_area.py` layout rebuild** (mirror, no image strip): `general-purpose` implementer.
- **`dash_visualizations.py` figure sizing** (drop pinned width, autosize flags; faceted height/scroll): small, surgical → `cavecrew-builder`.
- **clientside resize callback**: small → folded into the `dash_app.py` agent.

Detailed bite-sized tasks → `writing-plans` after this spec is approved.
