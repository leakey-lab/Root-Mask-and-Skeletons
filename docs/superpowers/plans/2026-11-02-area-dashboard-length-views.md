# Area dashboard on the length views — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The root-area dashboard shows the length dashboard's views (Growth Over Time, Depth Profile, Depth Profile Grid) with the same controls and toggles, charting area; only Totals by Tube (the field-level stacked bar chart, already shared) stays as it is.

**Architecture:** The length dashboard's page, view model and callbacks are already metric-agnostic except for a few strings, and its figures (`DashVisualizations`) already chart any metric. One main class, `MetricDashApp`, takes over the body of today's `DashApp`; `DashApp` (length) and `DashAppArea` (area) become subclasses that only name their shell, page title and export slug. The three `dash_length_*` modules are renamed `dash_metric_*`. The area's own figures, callbacks, floating legend button and position-means export are deleted.

**Tech Stack:** Python 3.14, Dash 4.4.1, plotly, pandas, PyQt6 (QWebEngine embedding), pytest.

**Spec:** the owner's request of 2026-11-02 (no separate spec): "refer to the length visualization that we have and except for the field level stacked bar chart I would like for you to change all the other visualization to the one from the length visualization with same control and toggles they should just work with the area parameter … we can create a main class with these visualization and then just refer to them from there."

## Global Constraints

- Run everything with `./venv/Scripts/python.exe`; Qt tests need `$env:QT_QPA_PLATFORM = 'offscreen'`.
- Lint/format: ruff only (`ruff check --fix .`, `ruff format .`), line length 100. The code uses Python 3.14 `except A, B:` syntax; keep it.
- **The length dashboard must not change**: `tests/test_viz_dash_split_golden.py` (served layout SHA-256, callback wiring, clientside JS SHA-256) must pass with its golden values untouched.
- Component ids and callback Outputs are a contract (renderer, embedding window): area uses exactly the length ids.
- View values for both dashboards: `stacked`, `time`, `lines`, `faceted` (labels "Totals by Tube", "Growth Over Time", "Depth Profile", "Depth Profile Grid"). The area value `profile` is gone.
- No CSV value or output pixel changes: nothing goes in CHANGELOG "Scientific output changes".
- Implementers never change git state (no `git add/commit/mv/rm/stash/checkout`); the lead commits each task after review.

## Review Focus

1. Area values are 100-1000x length values (mm² vs mm): Depth Profile x range and the grid's x ticks must stay readable — Task 1 test `test_area_grid_ticks_stay_few_for_large_areas`.
2. An area CSV with one tube: Growth Over Time with field average and variance must draw (SD is NaN, no band) — Task 1 test `test_area_time_view_with_one_tube_draws_without_a_band`.
3. Area dashboard without matching photos (image manager None or empty): hovering the Depth Profile shows "No images available for this interval", no error — Task 1 test `test_area_hover_without_images_says_so`.
4. Both dashboards built in one process: each serves its own thumbnails from its own server — Task 1 test `test_area_hover_thumbnail_is_served_by_the_area_app`.
5. Length unchanged: the existing goldens (`tests/test_viz_dash_split_golden.py`) — no new test, must stay green.

---

## Task 0 (lead, before dispatch): renames

- [ ] `git mv app/visualization/dash_length_layout.py app/visualization/dash_metric_layout.py`
- [ ] `git mv app/visualization/dash_length_views.py app/visualization/dash_metric_views.py`
- [ ] `git mv app/visualization/dash_length_callbacks.py app/visualization/dash_metric_callbacks.py`
- [ ] `git mv tests/test_viz_dash_length_views.py tests/test_viz_dash_metric_views.py`

Imports are fixed in Task 1. Nothing is committed until Task 1 passes.

---

## Task 1: `MetricDashApp` main class; area dashboard built from it

**Files:**
- Create: `app/visualization/dash_metric_app.py`
- Modify: `app/visualization/dash_app.py`, `app/visualization/dash_app_area.py`, `app/visualization/dash_metric_layout.py`, `app/visualization/dash_metric_views.py`, `app/visualization/dash_metric_callbacks.py`, `app/visualization/dash_export.py`, `app/visualization/dash_shell.py`, `app/visualization/__init__.py`
- Modify (imports / names only): `tests/test_import_smoke.py`, `tests/test_integrator_cross_module.py`, `tests/test_viz_dash_export_failure.py`, `tests/test_viz_dash_metric_views.py`, `tests/test_viz_dash_split_golden.py` (docstring only, goldens untouched), `tests/test_viz_dash_split_surface.py`, `tests/test_viz_embed_window.py` (`test_prewarm_imports_loads_the_dashboard_stack` module list), `tests/viz_dash_helpers.py`
- Create test: `tests/test_viz_dash_metric_app.py`

**Interfaces:**
- Produces:
  - `dash_metric_layout.CONTROLS_LABEL = "Tube & date range"`; `metric_shell(metric_label: str, plot_filename: str) -> ShellSpec` (title `f"Root {metric_label} — trial overview"`, the four views in the order and with the `has_controls` flags `LENGTH_SHELL` has today, `tube_view="lines"`); `LENGTH_SHELL = metric_shell("Length", "root_length_plot")`; `AREA_SHELL = metric_shell("Area", "root_area_plot")`.
  - `dash_metric_app.MetricDashApp(DashVisualizations)` with `ClassVar`s `shell_spec: ShellSpec`, `page_title: str`, `metric_slug: str`; `__init__(self, data_processor, save_directory: str, image_manager=None)`; methods `_parse_images_after_this_response`, `_setup_layout`, `_setup_callbacks` (today's `DashApp` bodies). It calls `create_dash_app(type(self).__module__, title=self.page_title)` so the length app keeps its import name `app.visualization.dash_app`. `theme.use("sprouts")` at its module top.
  - `dash_app.DashApp(MetricDashApp)`: `shell_spec = LENGTH_SHELL`, `page_title = "SPROUTS — Root Length"`, `metric_slug = "root_length"`; nothing else.
  - `dash_app_area.DashAppArea(MetricDashApp)`: `shell_spec = AREA_SHELL`, `page_title = "SPROUTS — Root Area"`, `metric_slug = "root_area"`; nothing else. The module still exports `AREA_SHELL` (imported from `dash_metric_layout`).
  - `dash_metric_callbacks.register_callbacks(dash_app)` passes `metric_slug=dash_app.metric_slug` to `register_export` (registration order unchanged).
  - `dash_metric_views`: protocol renamed `MetricDashboard` (adds `data_processor`); `export_table` descriptions use `noun = dashboard.data_processor.metric_label.lower()`: `f"Total root {noun} per tube and measurement date (sum over positions)"`, `f"Root {noun} over time: field average of per-tube totals (dates within 3 days merged; SD across tubes) and per-tube totals per measurement date"`, `f"Root {noun} statistics per {GROWTH_LINES_INTERVAL}-position depth interval and measurement date for one tube"`, `f"Mean root {noun} per depth interval: each tube next to the field average (all tubes) of the same interval and date"` (the length strings stay byte-identical). `faceted_outputs` uses `CONTROLS_LABEL` instead of `LENGTH_SHELL.controls_label`.
  - `tests/viz_dash_helpers.build_area_app(tmp_path, image_manager=None)` -> `DashAppArea(DataProcessorArea(csv), str(tmp_path), image_manager or FakeImageManager())`.
  - `__init__._EXPORTS["MetricDashApp"] = ".dash_metric_app"`.
- Deletes: everything else in `dash_app_area.py` (its layout, callbacks, figures, `_LEGEND_*`, `_CHART_BOX`); `dash_export.position_means`; `DashboardShell.register_legend_toggle` (only the area used it). Keep `LEGEND_STORE_ID`, `LEGEND_TOGGLE_ID`, `legend_shown`, `flip_legend`, `legend_patch` in `dash_shell`.

- [ ] **Step 1: Write the failing tests** in `tests/test_viz_dash_metric_app.py` (use `build_length_app`, `build_area_app`, `call_callback`, `component_ids`, `FakeImageManager`, `write_metric_csv`, `AREA_COLUMN` from `tests/viz_dash_helpers.py`; look at `tests/test_viz_dash_callbacks.py` and `tests/test_viz_dash_export.py` for how hover and export callbacks are driven):
  - `test_both_dashboards_are_the_main_class`: `issubclass(DashApp, MetricDashApp)` and `issubclass(DashAppArea, MetricDashApp)`; `{n for n, v in vars(DashAppArea).items() if callable(v) and not n.startswith("__")} == set()`.
  - `test_area_shell_has_the_length_views`: `[(v.label, v.value, v.has_controls) for v in AREA_SHELL.views] == [(v.label, v.value, v.has_controls) for v in LENGTH_SHELL.views]`; `AREA_SHELL.title == "Root Area — trial overview"`; `AREA_SHELL.plot_filename == "root_area_plot"`; `AREA_SHELL.tube_view == "lines"`; `AREA_SHELL.controls_label == LENGTH_SHELL.controls_label`.
  - `test_area_page_has_every_length_component`: served `/_dash-layout` component ids of area == of length.
  - `test_area_callbacks_are_wired_like_length`: the `(output, inputs, states, prevent_initial_call, clientside)` tuples of `/_dash-dependencies` are equal for both apps (same helper shape as `tests/test_viz_dash_split_golden.py`).
  - `test_area_figures_chart_area`: area `show_growth_lines(1).layout.xaxis.title.text == "Root Area (mm²)"`; `show_growth_over_time(selected_tubes=[1]).layout.title.text == "Root Area Growth Over Time"` and y title `"Total Area (mm²)"`; `create_faceted_depth_profile([1, 2])` has a trace whose `hovertemplate` contains `"Avg Area"`.
  - `test_area_time_export_names_area`: Export CSV of the area `time` view with `selected-tubes-store.data == [1]`: filename starts `root_area_time_`, content contains `Root area over time` and the column `Total Area (mm²)`.
  - `test_area_grid_ticks_stay_few_for_large_areas`: area CSV with values around 5000; the faceted figure's `xaxis.tickvals` has at most 13 values.
  - `test_area_time_view_with_one_tube_draws_without_a_band`: area CSV with one tube; `show_growth_over_time(selected_tubes=[1], show_field_average=True, show_field_variance=True)` has no trace named `"Field Variance (±1 SD)"` and has `"Field Average"`.
  - `test_area_hover_without_images_says_so`: area app with `FakeImageManager()`; the `image-container.children` callback with view `lines`, tube 1 and a hover point `{"y": 10, "customdata": "2024.07.18"}` returns the "No images available for this interval" hint.
  - `test_area_hover_thumbnail_is_served_by_the_area_app`: area app with `FakeImageManager({"T001_L005_2024.07.18.png": <a real small PNG written under tmp_path>})`; the hover callback returns an `Img` whose `src` the area app's Flask test client serves with status 200.
- [ ] **Step 2: Run them; they fail** (`AREA_SHELL` has 3 views, `MetricDashApp` missing).
  Run: `./venv/Scripts/python.exe -m pytest tests/test_viz_dash_metric_app.py -q -p no:cacheprovider`
- [ ] **Step 3: Implement** the Interfaces above. Module docstrings of the renamed modules say "metric dashboards (length and area)"; keep every comment that still holds.
- [ ] **Step 4: Fix the renamed imports** in the test files listed under Files (`dash_length_layout` -> `dash_metric_layout`, `dash_length_views` -> `dash_metric_views`, `dash_length_callbacks` -> `dash_metric_callbacks`). In `tests/test_viz_dash_split_surface.py`: `SPLIT_MODULES` gets the new names plus `app.visualization.dash_metric_app`; the "own attributes" test checks that `MetricDashApp` owns the three methods with today's kinds and parameters and that `DashApp` owns only `shell_spec`, `page_title`, `metric_slug`; the removed-delegator checks stay. In `tests/test_viz_embed_window.py` add `app.visualization.dash_metric_app` to the prewarm module list (only if `viz_window.prewarm_imports` loads it; it does through `dash_app`).
- [ ] **Step 5: Verify.**
  Run: `./venv/Scripts/python.exe -m pytest tests/test_viz_dash_metric_app.py tests/test_viz_dash_split_golden.py tests/test_viz_dash_split_surface.py tests/test_viz_dash_metric_views.py tests/test_import_smoke.py tests/test_integrator_cross_module.py -q -p no:cacheprovider`
  Expected: all pass, goldens untouched. Then `ruff check` and `ruff format --check` on the changed files: clean.
  Known and expected to fail until Tasks 3-4: the area-pinning tests listed there (do not edit them in this task).
- [ ] **Step 6: Lead commits** `refactor(viz): one MetricDashApp for both dashboards; area gets the length views`.

---

## Task 2: area window gets the image manager

**Files:**
- Modify: `app/visualization/root_area_visualization.py`, `app/gui/visualization_manager.py`
- Modify tests: `tests/test_viz_embed_manager.py`; any stub standing in for `RootAreaVisualization` that cannot take `(csv_path, image_manager)` (`tests/test_window_library_focus.py`, `tests/test_window_width_budget.py`: check, change only if needed)

**Interfaces:**
- Consumes: `DashAppArea(data_processor, save_directory, image_manager=None)` (Task 1).
- Produces: `build_area_dashboard(csv_path: str, save_directory: str, image_manager)`; `RootAreaVisualization.__init__(self, csv_path: str, image_manager, parent=None)` mirroring `RootLengthVisualization` (sets `self.image_manager` before `super().__init__`); `DashboardSpec` loses `needs_image_manager`, and `_build` always calls `cls(csv_path, mw.image_manager)`.

- [ ] **Step 1: Failing test** in `tests/test_viz_embed_manager.py`: `test_area_dashboard_gets_the_image_manager` — show the area dashboard; `StubArea.made[0].image_manager is` the window's image manager (`StubArea.__init__(self, csv_path, image_manager)` like `StubLength`).
- [ ] **Step 2: Run it; it fails.** `./venv/Scripts/python.exe -m pytest tests/test_viz_embed_manager.py -q -p no:cacheprovider`
- [ ] **Step 3: Implement** the Interfaces.
- [ ] **Step 4: Verify**: `./venv/Scripts/python.exe -m pytest tests/test_viz_embed_manager.py tests/test_viz_embed_imports.py tests/test_viz_embed_window.py tests/test_window_library_focus.py tests/test_window_width_budget.py -q -p no:cacheprovider` — pass (a test that builds a real area dashboard may need Task 1; report it). Ruff clean.
- [ ] **Step 5: Lead commits** `feat(viz): the area dashboard gets the photos for Depth Profile hover`.

---

## Task 3: figure-level area tests follow the new views

**Files (tests only):** delete `tests/test_viz_area_legend_toggle.py`; modify `tests/test_viz_data_time_view.py`, `tests/test_viz_data_bar_labels.py`, `tests/test_viz_theme_bar_label_contrast.py`, `tests/test_viz_dash_regressions.py`, `tests/test_viz_dash_plotly_bundle.py`.

**Interfaces:** Consumes Task 1: `DashAppArea` has the `DashVisualizations` builders (`show_growth_over_time(selected_tubes=..., show_field_average=..., show_field_variance=..., show_legend=...)`, `show_growth_lines(tube)`, `create_faceted_depth_profile(tubes, dates)`, `create_stacked_bar_chart()`); `show_area_profile` and the area-only time figure are gone.

Rule for each failing test: if it pins behaviour that no longer exists (grouped-bar area profile, all-tubes area time view, the floating legend button, `position_means`), delete it, unless the same property still holds for the new area view, in which case point it at the new builder (example: the bar-label contrast test keeps the area stacked bars; `test_viz_data_time_view` compares the area time view's unvalidated JSON with a validated build the way it does for length). Keep tests of the stacked bars as they are. Do not weaken a length assertion.

- [ ] **Step 1:** Run the five files; list the failures.
- [ ] **Step 2:** Apply the rule; delete `tests/test_viz_area_legend_toggle.py` (the length toggle tests cover the shared toggle; parametrize one of them over `build_area_app` if none covers area).
- [ ] **Step 3: Verify**: the five files (minus the deleted one) pass; ruff clean.
- [ ] **Step 4: Lead commits** `test(viz): area figure tests follow the length views`.

---

## Task 4: Dash-level area tests follow the new views

**Files (tests only):** `tests/test_viz_dash_shell.py`, `tests/test_viz_dash_callbacks.py`, `tests/test_viz_dash_export.py`, `tests/test_viz_dash_view_names.py`, `tests/test_viz_dash_uirevision.py`, `tests/test_viz_dash_autosize_guard.py`, `tests/test_viz_dash_offline_assets.py`, `tests/test_viz_dash_export_failure.py`.

**Interfaces:** Consumes Task 1: area page ids == `LENGTH_IDS`, area callbacks == `LENGTH_CALLBACKS`; area views `stacked/time/lines/faceted`; area export filenames `root_area_<view>_...`; the `browser_figure` used by callbacks lives in `dash_metric_callbacks` (the area has no module of its own to monkeypatch).

Same rule as Task 3. Concretely: `AREA_IDS` / `AREA_CALLBACKS` become the length sets (or the parametrization uses `LENGTH_*` for both); `profile` cases become `lines`; `test_area_profile_*` become `lines` tests or are deleted if `lines` is covered for length by a test now parametrized over both builders; the area profile export test becomes a `lines` export test (depth-interval columns, `root_area_lines_` filename); `test_viz_dash_view_names` asserts both dashboards have the same view values and labels and that the area titles read "Root Area Depth Profile - Tube 1", "Root Area Growth Over Time", "Root Area by Tube and Date".

- [ ] **Step 1:** Run the eight files; list the failures.
- [ ] **Step 2:** Apply the rule.
- [ ] **Step 3: Verify**: the eight files pass; ruff clean.
- [ ] **Step 4: Lead commits** `test(viz): area Dash tests follow the length views`.

---

## Task 5: docs

**Files:** `CLAUDE.md` (Visualization row: `dash_metric_app.py` `MetricDashApp` composes `dash_metric_layout.py` / `dash_metric_views.py` / `dash_metric_callbacks.py`; `DashApp` and `DashAppArea` are thin subclasses; the golden still pins the length page), `docs/SPROUTS_DASH_THEME.md` (file table, the view-name table: both dashboards `stacked/time/lines/faceted`; drop the area-only profile/legend-button notes), `CHANGELOG.md` (Unreleased, a "Changed" entry: the area dashboard has the length views and controls — tube selection tabs, field average and variance, legend toggle, Depth Profile lines with photo hover, Depth Profile Grid; its Export CSV of Growth Over Time and Depth Profile now writes the length dashboard's tables; no CSV value or pixel changes, so nothing under Scientific output changes), `docs/KNOWN_ISSUES.md` (only if it describes the area dashboard's views as current behaviour; resolved history rows stay).

- [ ] **Step 1:** Edit; keep every existing statement that still holds; plain sentences, no new jargon.
- [ ] **Step 2: Verify**: `grep -rn "dash_length_\|show_area_profile\|position_means" CLAUDE.md docs/SPROUTS_DASH_THEME.md` finds only historical mentions you chose to keep.
- [ ] **Step 3: Lead commits** `docs: the area dashboard uses the length views`.

---

## Wrap-up (lead)

- [ ] Full suite: `$env:QT_QPA_PLATFORM = 'offscreen'; ./venv/Scripts/python.exe -m pytest -n auto -m "not wallclock"` then `-m wallclock`.
- [ ] `ruff check .` and `ruff format --check .` clean.
- [ ] Whole-branch review (one reviewer agent) of the five commits.
- [ ] `graphify update .` (uv tool venv python, memory `graphify-cli-invocation`).
- [ ] Memory + vault note.

## Execution waves

- Wave 1 (parallel, disjoint files): Task 1, Task 2, Task 5.
- Wave 2 (parallel, disjoint files, after Task 1 is committed): Task 3, Task 4.
