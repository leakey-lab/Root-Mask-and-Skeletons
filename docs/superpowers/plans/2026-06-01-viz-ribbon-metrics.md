# Viz Redesign + Ribbon Split + Metrics Bar — Implementation Plan

> REQUIRED SUB-SKILL: subagent-driven-development. Presentation only; `pytest -q` green after every task (currently 70).

**Goal:** (V) rebuild both Dash dashboards to the SPROUTS VizView — full-bleed, width-fluid chart, collapsible controls, Growth-Lines images beside the chart; (R) split the ribbon into 8 explicit stages; (M) add the display metrics bar.

**Guardrails:** NO change to computed data, Qt signals/slots, `main_window.<attr>` names, `QStackedWidget` indices, or Dash `Output`/`id`/`value`/`customdata`. Specs: `docs/superpowers/specs/2026-06-01-viz-redesign-design.md`.

---

## Track V — Viz redesign (files: `app/assets/style.css`, `app/visualization/dash_app.py`, `dash_app_area.py`, `dash_visualizations.py`)

- [ ] **V1 — CSS port.** Add token-based `.viz-page`/`.viz-header`/`.viz-eyebrow`/`.viz-title`/`.viz-controls`/`.viz-metrics`/`.metric`/`.viz-chart-card`/`.viz-lines-row`/`.viz-image-panel` classes to `app/assets/style.css`. `pytest -q`.
- [ ] **V2 — `dash_app.py` layout rebuild.** Replace the centered `dbc` card tree in `layout()` with the `.viz-page` flex column: `.viz-header` (eyebrow + title + `view-selector` RadioItems + Export), `.viz-controls` collapsible strip wrapping the EXISTING tube multiselect / From-To / Apply (collapsed default), `.viz-metrics` chips, `.viz-chart-card` (`flex:1; min-height:0`, full width) holding `dcc.Loading > dcc.Graph(id="main-graph", style width:100%)`. For `view-selector=="lines"` render `.viz-lines-row` = chart pane + `image-container` (id unchanged) right panel. KEEP every `id`, callback `Output`/`Input`/`State`, option value. Add a `clientside_callback` firing `Plotly.Plots.resize` on control-strip toggle + `view-selector` change (writes a hidden `dcc.Store`, no figure Output). `pytest -q`.
- [ ] **V3 — `dash_app_area.py` layout rebuild.** Mirror V2 (views Stacked/Time/Profile, NO image strip → always full-width chart). KEEP ids/values. `pytest -q`.
- [ ] **V4 — figure sizing.** In `dash_visualizations.py` + `dash_app_area.py` chart builders: remove pinned figure `width`; for stacked/time/lines drop explicit `height` (let autosize fill the card); keep faceted's computed tall height (card scrolls). Keep `config responsive:True`. KEEP all data/customdata. `pytest -q`.
- [ ] **V-verify:** human GUI check — chart full width, widens with panel/splitter, control strip collapses, Growth-Lines images beside chart, faceted full width.

## Track R — Ribbon 8 stages (files: `app/gui/shell_chrome.py`, `app/gui/main_window.py`)

- [ ] **R1 — 8 stages.** `build_ribbon`: Library(image)/Generate Mask(cpu)/Trace(brush)/Generate Skeleton(skeleton)/Correct(node)/Measure(ruler)/Visualize(chart). Click handlers: Library/GenMask/GenSkel/Measure/Visualize→`switch_right_panel('display')`; Trace→`toggle_mask_tracing`; Correct→`toggle_skeleton_correction`. `pytest -q`.
- [ ] **R2 — action-bar dispatch.** Update `_populate_action_bar`/`_activate_action_stage` for the 8 labels: GenMask→`generate_mask_button`; Trace→`mask_tracing_interface.clear_button`; GenSkel→`generate_button`; Measure→`calculate_length_button`+`calculate_area_button`; Visualize→Length/Area seg; Library/Correct→none. Per-stage hint text. KEEP all connects. `pytest -q`; contract smoke test still green.
- [ ] **R-verify:** human GUI — all 8 stages switch correctly, each action-bar fires its handler.

## Track M — Metrics bar (files: new `app/gui/metrics_bar.py`, `app/gui/ui_panels.py`, `app/gui/main_window.py`)

- [ ] **M1 — `MetricsBar` widget.** New `app/gui/metrics_bar.py`: token-styled thin strip, 4 cells (Root length accent / Root area / FOV 18×13 / Status), `set_metrics(length, area, status)` showing `—` when None. Import smoke test. `pytest -q`.
- [ ] **M2 — mount + refresh.** Add the bar below the display canvas (in `create_right_panel`'s display page), store `main_window.metrics_bar`; refresh it from the existing image-selection / `load_results` path (read the selected image's measured length/area + status — read-only, no compute). `pytest -q`.
- [ ] **M-verify:** human GUI — selecting a measured image populates the metrics; `—` before measurement.

---

## Delegation (by modification size)
- V1, V4, M1 → `cavecrew:cavecrew-builder` (surgical, 1-2 files).
- V2, V3, R1+R2, M2 → `general-purpose` implementer (multi-section / cross-file).
- Sequential where files overlap (R + M both touch `main_window.py`; V touches disjoint `app/visualization` + `style.css`). pytest-gated, commit per task, pause at each track's verify boundary.
