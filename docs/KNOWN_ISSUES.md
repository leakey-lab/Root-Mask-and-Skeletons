# SPROUTS redesign — known issues / to reconsider

Tracked during the `feat/sprouts-redesign` work. These are open items, not regressions introduced by the redesign unless noted.

## 1. Visualization is still broken — needs reconsideration

The embedded Dash/Plotly dashboards still do not render/behave correctly and the
sizing/layout approach needs to be rethought, not just enlarged. The PR4 change
(`fix(viz): enlarge Dash chart drawing area`) raised the drawing area (85vh main,
90vh faceted, taller per-row faceted height) but the visualization as a whole is
**still broken and must be reconsidered** — the chart area, faceted depth profile
layout, and the QWebEngineView embedding all need a fresh design pass rather than
incremental size bumps. Do not treat the current viz as final.

## 2. Ribbon needs separate buttons for skeleton generation vs tracing

The 6-stage ribbon currently conflates actions: skeleton **generation**
(`skeleton_handler.generate_skeleton`, today an action-bar button under the
Skeleton stage) and mask **tracing** / skeleton **correction** editing are not
clearly separated as their own ribbon entries. Skeleton generation and tracing
should each get their **own dedicated ribbon button** so the pipeline reads
clearly (generate vs manually edit are distinct user intents). Revisit the ribbon
stage model when doing the editor PRs (PR5/PR6).
