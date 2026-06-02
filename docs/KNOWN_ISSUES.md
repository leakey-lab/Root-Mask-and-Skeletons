# SPROUTS redesign — known issues / to reconsider

Tracked during the `feat/sprouts-redesign` work. These are open items, not regressions introduced by the redesign unless noted.

## 1. Visualization layout — RESOLVED (redesigned 2026-06-01)

Rebuilt both Dash dashboards to the SPROUTS VizView (spec:
`docs/superpowers/specs/2026-06-01-viz-redesign-design.md`): full-bleed,
width-fluid chart that auto-fits the panel; collapsible control strip; metric
chips; Growth-Lines hover images now sit **beside** the chart (300px right
column), not below. No data/callback/id/customdata changes. Verify in the
running app that the chart fills width and the faceted profile scrolls.

## 2. Ribbon split — RESOLVED (2026-06-01)

Ribbon is now 7 explicit stages: Library · Generate Mask · Trace ·
Generate Skeleton · Correct · Measure · Visualize. Generation (auto) and manual
editing (Trace/Correct editors) are distinct entries, each wired to the existing
handlers.

## 3. Metrics bar — length/area have no read-only source (OPEN)

The new display `MetricsBar` shows Status + FOV correctly, but **Root length /
Root area show `—`**: measured values are produced only by batch threads
(`skeleton_handler.calculate_root_length/area` → calculator threads) and written
to CSV outputs; they are never loaded back into per-image state. Populating them
read-only would need parsing those CSVs (or caching the measurements in
`image_manager`). Deferred — do not fabricate values.
