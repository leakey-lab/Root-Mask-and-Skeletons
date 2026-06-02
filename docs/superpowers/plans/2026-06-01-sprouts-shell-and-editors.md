# SPROUTS Shell & Editors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish porting the SPROUTS refined-dark redesign into the PyQt6 app — the Guided shell chrome (titlebar / 6-stage ribbon / action-bar / status line), the Welcome + Loading onboarding screens, and the modern floating Trace & Skeleton editors — reusing the already-built `app/gui/widgets/` primitives and changing presentation only.

**Architecture:** The app stays a PyQt6 desktop shell with Dash charts embedded in `QWebEngineView`. The redesign is a *new front-end over existing methods*: the 6 ribbon stages drive the existing `switch_right_panel` 4-index `QStackedWidget` (display 0 / mask-tracing 1 / viz 2 / skeleton 3) and call the existing handler methods; the editors keep every internal signal and only move their controls out of a bottom `QGroupBox` into floating overlays repositioned in `resizeEvent`. No data, signal, index, or Dash-contract changes.

**Tech Stack:** PyQt6 (QSS is a CSS2 subset — no box-shadow/text-shadow/opacity/backdrop-filter; shadows via `QGraphicsDropShadowEffect`, filled sliders via QSS `::sub-page`/`::add-page`), the in-repo `app/gui/widgets/` design system, IBM Plex Sans/Mono.

---

## Design tokens (single source of truth: `app/gui/widgets/tokens.py` / `resources/themes/dark_theme.qss`)

```
bg-0 #15161c · bg-1 #1b1c24 · bg-2 #21232e · bg-3 #282a37 · hover #2f3242 · active #363a4d
border #2a2c39 · border-strong #373a4c · text #eceef5 · muted #9498ad · faint #686c82
ACCENT #c39af6 (purple) · accent-press #ab87d8 · accent-ink #1a1322 · accent-soft rgba(195,154,246,.14)
SEL #b794f6 · sel-soft rgba(183,148,246,.16) · MASK #5fd6a0 (green) · SKEL #f0a868 (orange)
warn #e8b25e · info #79c0e8 · danger #ec6a78 · radius 10/7/14 · comfy row-h 40 / pad 18 / ui 14.5
```
Purple = accent + selection + chrome. Green = mask overlay only. Orange = skeleton only.

## Already landed (do not redo)

- **PR1 Dash theme** — `theme.use("sprouts")`, DARKLY, IBM Plex `index_string`, `dbc.RadioItems` view-selector (id/values kept), `dcc.Loading`, collapsed `update_layout`, `theme.hover`. Customdata/Outputs preserved.
- **PR2 QSS + palette** — `resources/themes/dark_theme.qss` + `main.py` palette on tokens; filled sliders; purple `#primaryButton`; killed neon scrollbars/splitter.
- **Widgets package** `app/gui/widgets/` (61 tests green): `tokens`, `icons` (`load_icon`/`load_pixmap` recolor `currentColor` SVGs), `effects` (`drop_shadow`/`pop_shadow`), `controls` (`SegmentedControl(valueChanged str)`, `IconButton`), `overlays` (`ToolRail`, `FloatingDock`, `EnhancePopover`, `Toast`, `ToastManager`, `ProgressOverlay`). 50 SVG icons at `resources/icons/sprouts/`.
- **PR3 onboarding/feedback (partial)** — `empty_state.py` + `task_progress.py` on tokens; `main_window` has `ToastManager` + `notify()` + `resizeEvent`; success messages routed through toasts; splitter neon purged.

## Global guardrails (every task)

- **Presentation only.** Never change computed data, Qt signals/slots, `main_window.<attr>` names, `QStackedWidget` indices, Dash `Output`/`id`/`value`/`customdata`.
- **Reparenting MUST NOT touch `_connect_signals` `connect()` calls.** Move widgets between layouts/parents; leave the wiring block byte-identical.
- `view_mode_combo` stays a live `QComboBox` (indices 0/1/2 Single/Overlay/Side-by-Side); any `SegmentedControl` is a cosmetic forwarder that calls `view_mode_combo.setCurrentIndex(...)`.
- Run `./venv/Scripts/python.exe -m pytest -q` after **every** task; `tests/test_import_smoke.py` + `tests/test_regression.py` + `tests/test_widgets_smoke.py` must stay green.
- After each PR, launch `./venv/Scripts/python.exe main.py` and walk the flow against the matching bundle screenshot (`09-guided-final`, `_polish_welcome`, `08-loading`, `05-trace`, `10-polyline`, `11-enhance`).
- `graphify update .` after code edits.

## Stage → method mapping (the 6 ribbon stages over 4 panel indices)

| Stage | switch_right_panel | Action-bar action → existing method |
|---|---|---|
| Library | display (0) | Load Images → `main_window.load_images` |
| Mask | display (0) | Generate ML Masks → `mask_generation_handler.generate_masks` |
| Trace | mask-tracing (1) | Clear Mask → `mask_tracing_interface.clear_mask` |
| Skeleton | skeleton (3) | Generate Skeleton → `skeleton_handler.generate_skeleton` |
| Measure | display (0) | Calculate Length/Area → `calculate_root_length` / `calculate_root_area` |
| Visualize | viz (2) | Length/Area `SegmentedControl` → `toggle_root_length/area_visualization` |

---

## File Structure

**New files**

| File | Responsibility |
|---|---|
| `app/gui/welcome_screen.py` | `WelcomeWidget(on_get_started: Callable)` — full-window first-run page: leaf logo + "SPR**OU**TS" wordmark, tagline, dashed dropzone + "Browse…" primary button, recent-projects list (via `QSettings`), 6-stage pipeline chip row, GPU/model footer. Button + dropzone call `on_get_started` (= `main_window.load_images`). Pure chrome, no app state. |
| `app/gui/shell_chrome.py` | `build_titlebar(mw)`, `build_ribbon(mw)`, `build_action_bar(mw)`, `build_statusline(mw)` → each returns a `QWidget`. They **re-parent existing `mw.<attr>` buttons by reference** (or assign new ribbon buttons to the exact attr names) and wire stage clicks to existing methods. No new wiring logic. |
| `app/gui/loading_overlay.py` | `LoadingOverlay(parent)` — thin subclass/wrapper of `app.gui.widgets.ProgressOverlay` adding the pulsing leaf logo + filename/count labels from the design. API: `start()`, `set_progress(v)`, `hide()`, `reposition()`. Child of the shell, hidden by default. |
| `tests/test_shell_chrome_smoke.py` | Instantiate a `MainWindow` offscreen; assert every attr in the connect-map still exists, `view_mode_combo.count()==3`, `right_panel.count()==4`, and the new shell widgets import + build. |

**Modified files**

| File | Change (line ranges approximate — locate by content; this session already added toasts/`resizeEvent`/`notify` near the status-bar block) |
|---|---|
| `app/gui/main_window.py` | `init_ui` → extract `_build_body()` (the splitter + `right_panel`, order preserved so `switch_right_panel` idx 0–3 stay valid) and `_build_shell()`; wrap in `self.app_stack` (`QStackedWidget`: 0=Welcome, 1=shell), `setCentralWidget(self.app_stack)`. `load_images` → `app_stack.setCurrentIndex(1)` + `loading_overlay.start()`. `update_loading_progress` → `loading_overlay.set_progress`. `on_loading_finished` → `loading_overlay.hide()` (leave the `display_page_stack.setCurrentIndex(0)` flip intact). Keep the single `self.task_progress` instance. |
| `app/gui/ui_panels.py` | `create_left_panel` → reskin: re-parent the existing buttons + `file_list` into ribbon/action-bar/library containers; **keep every attr name + `.clicked.connect`**; drop the 4 bordered `*_widget` section frames. `create_right_panel` unchanged (display_page_stack stays at `right_panel` idx0); MetricsBar + view-mode `SegmentedControl` added here. |
| `app/gui/mask_tracing_interface.py` | Reparent tools→`ToolRail`, actions+sliders→`FloatingDock`, `norm_controls`→`EnhancePopover`; delete the `_create_control_panel` `QGroupBox` chrome; add `_build_overlays` + `resizeEvent`. **`_connect_signals` body untouched** (call site moves to end of `_build_overlays`). |
| `app/gui/skeleton_correction_interface.py` | Same reparent into rail/dock/popover + top-center polyline prompt; recolor scene markers to tokens (`QColor(57,255,20)`→`#f0a868`, handles/selection→`#c39af6`); delete `QGroupBox` chrome; add `_build_overlays` + `resizeEvent`. **Inline `connect()` block untouched.** |

---

## Tasks

> TDD note: this is a presentation port — runtime logic is frozen, so the "test" is the **guardrail suite** (`pytest -q`) plus a per-PR smoke-launch. Each PR ends with a verification task. New widget classes get an import/instantiation test first (true TDD where it applies).

### PR4 — Guided shell + Welcome/Loading

- [ ] **T4.1 — Failing import test for WelcomeWidget.** Add to `tests/test_shell_chrome_smoke.py`: `from app.gui.welcome_screen import WelcomeWidget`. Run `./venv/Scripts/python.exe -m pytest tests/test_shell_chrome_smoke.py -q` → FAIL (module missing).
- [ ] **T4.2 — Create `app/gui/welcome_screen.py`.** `WelcomeWidget(QWidget)` with `__init__(self, on_get_started, parent=None)`; leaf logo via `widgets.load_pixmap("sprouts_logo", tokens.ACCENT, 58)`; "SPROUTS" title (accent on "OU"); dashed dropzone (`QFrame`, `setAcceptDrops`, `dragEnterEvent`/`dropEvent` → `on_get_started`); "Browse…" `#primaryButton` → `on_get_started`; recent list from `QSettings("LeakeyLab","SPROUTS")`; 6 stage chips; footer label. Verify: `pytest tests/test_shell_chrome_smoke.py -q` → PASS.
- [ ] **T4.3 — Create `app/gui/loading_overlay.py`.** `LoadingOverlay(ProgressOverlay)` (or wrapper) adding a pulsing logo + `set_filename(str)`/`set_count(scanned,total)`. Verify: `./venv/Scripts/python.exe -c "from app.gui.loading_overlay import LoadingOverlay"`.
- [ ] **T4.4 — Extract `_build_body()` in `main_window.py`.** Move the splitter + `left_panel`/`right_panel` assembly out of `init_ui` into `_build_body(self)->QWidget` returning the splitter container; **preserve the 4 `right_panel.addWidget` order**. `init_ui` calls it. Verify: `pytest -q` green; `./venv/Scripts/python.exe main.py` launches unchanged.
- [ ] **T4.5 — Add `app_stack` + Welcome page.** In `init_ui`: `self.app_stack = QStackedWidget()`; `self.welcome = WelcomeWidget(on_get_started=self.load_images)`; `app_stack.addWidget(self.welcome)` (idx0); shell container at idx1; `setCentralWidget(self.app_stack)`; `app_stack.setCurrentIndex(0)`. Verify: launch shows Welcome first.
- [ ] **T4.6 — Flip to shell on load.** In `load_images`, inside `if dir_name:`, add `self.app_stack.setCurrentIndex(1)` and `self.loading_overlay.start()`. Verify: launch → Browse → shell appears.
- [ ] **T4.7 — `_build_shell()` chrome scaffolding.** Create `app/gui/shell_chrome.py` with `build_titlebar/ribbon/action_bar/statusline`. `_build_shell(self)` lays them vertically over `_build_body()` (stretch) with statusline footer; returns the shell container used at `app_stack` idx1. Titlebar logo + "Load Images" → `self.load_images`. Verify: launch, shell renders with all four bands; `pytest -q`.
- [ ] **T4.8 — Wire the 6 ribbon stages to existing methods.** In `build_ribbon`, create 6 stage buttons (icons: image/cpu/brush/skeleton/ruler/chart). Route: Library→`switch_right_panel('display')`; Mask→`switch_right_panel('display')`; Trace→`self.toggle_mask_tracing`; Skeleton→`self.toggle_skeleton_correction`; Measure→`switch_right_panel('display')`; Visualize→`switch_right_panel('display')` + show the Length/Area `SegmentedControl`. **Trace/Skeleton MUST go through the toggle methods** (they manage `_mask_tracing_signals_connected` + OpenGL), never raw index. Verify: each stage switches `right_panel`; trace/skeleton toggles connect/disconnect cleanly (no duplicate-signal warnings); `pytest -q`.
- [ ] **T4.9 — Action-bar contextual actions.** `build_action_bar` shows a hint label + the stage's existing buttons by reference: Library→`load_images_button`; Mask→`generate_mask_button`; Trace→`clear_button` (from `mask_tracing_interface`); Skeleton→`generate_button`; Measure→`calculate_length_button`+`calculate_area_button`; Visualize→Length/Area `SegmentedControl`→`toggle_root_length/area_visualization`. Verify: every action fires its original handler; `pytest -q`.
- [ ] **T4.10 — Relocate status bar into the statusline; keep `task_progress`.** Move `status_bar`/`task_progress`/`ToastManager` into the `build_statusline` band (live dot + message + "N images · GPU · CUDA 12.8"). **Keep the single `self.task_progress` instance + attr name** (the `_ProgressBarShim`/`loading_progress_bar` property depend on it). Verify: generate a mask → progress still updates via the shim; toasts still appear; `pytest -q`.
- [ ] **T4.11 — Loading overlay wiring.** `load_images`→`loading_overlay.start()`; `update_loading_progress(value)`→`loading_overlay.set_progress(value)`; `on_loading_finished`→`loading_overlay.hide()`. **Do not remove the `display_page_stack.setCurrentIndex(0)` flip.** Add `loading_overlay.reposition()` to `MainWindow.resizeEvent`. Verify: load a real dir → overlay shows, advances, hides; display populates.
- [ ] **T4.12 — Reskin `ui_panels.create_left_panel`.** Re-parent existing buttons + `file_list` + expand/collapse into the ribbon/action-bar/library; drop the bordered section frames; add a search field filtering `file_list`; mono `done/total` rollup + status dots on tree rows. **Keep all attr names + `.clicked.connect` lines.** Verify: expand/collapse lambdas, every stage button, tree `itemClicked` all fire; `pytest -q`.
- [ ] **T4.13 — Replace view-mode combo chrome with a forwarder SegmentedControl.** Keep `view_mode_combo` live but `setVisible(False)`; add `SegmentedControl(["single","overlay","split"])` whose `valueChanged` calls `view_mode_combo.setCurrentIndex(0|1|2)`; sync combo→seg on `currentIndexChanged`. Verify: switching the seg drives `display_controller.update_display_mode`; `load_results`' `setCurrentText` path still works; `pytest -q`.
- [ ] **T4.14 — MetricsBar.** Below the display canvas, add a 4-metric strip (Root length accent / Root area / FOV 18×13 / Status) fed from the selected image/results. Read-only. Verify: select a measured image → values populate; `pytest -q`.
- [ ] **T4.15 — PR4 verification pass.** Assert (in `tests/test_shell_chrome_smoke.py`) every connect-map attr exists on `MainWindow`, `view_mode_combo.count()==3`, `right_panel.count()==4`. Smoke-launch: Welcome → Browse → Loading → shell; walk all 6 stages; expand/collapse; open a Dash dashboard. `graphify update .`. Commit.

### PR5 — Modern Trace editor (`mask_tracing_interface.py`)

- [ ] **T5.1 — Import overlays.** Add `from .widgets import ToolRail, FloatingDock, EnhancePopover, IconButton, load_icon, tokens`. Verify: `pytest -q`.
- [ ] **T5.2 — `_build_overlays(self)` skeleton.** Create `self.tool_rail`/`self.dock`/`self.enhance_popover` parented to the interface widget; called from `initUI` after `setLayout`. Verify: import OK; `pytest -q`.
- [ ] **T5.3 — Tools → ToolRail.** Move `brush_button`/`eraser_button`/`fill_button` (+ `tool_button_group`) into `tool_rail`; keep attr names + group membership. Verify: launch trace mode, tools visible + exclusive.
- [ ] **T5.4 — `mode_toggle` → rail.** Move its construction into `_build_overlays`; keep `self.mode_toggle.clicked.connect(self.toggle_mode)`. Verify: Draw/Pan toggle works.
- [ ] **T5.5 — Actions + sliders → FloatingDock.** Move `size_slider`/`opacity_slider` (filled), `undo`/`redo`/`clear_button` (icon buttons), primary `save_button` into `dock`. Keep attr names. Verify: undo/redo/clear/save + sliders fire.
- [ ] **T5.6 — `norm_controls` → EnhancePopover.** `enhance_popover.set_content(self.norm_controls)`; add a contrast `IconButton` on the rail → `enhance_popover.toggle`. Verify: popover opens; `apply_button` (`apply_normalization`) + `restore_defaults_button` fire.
- [ ] **T5.7 — Call `_connect_signals()` at end of `_build_overlays`.** Move only the call site; **leave the `_connect_signals` body byte-identical**. Verify: `pytest -q`.
- [ ] **T5.8 — Delete `QGroupBox` chrome.** Remove `_create_control_panel` panel/stylesheet/`control_layout`/`return` and the `addWidget(control_panel)`; keep the tool/action/adjustment factory methods (drop only the QGroupBox wrappers). Verify: launch, no leftover bottom panel.
- [ ] **T5.9 — `resizeEvent` + initial reposition.** `tool_rail.reposition()+raise_()`, `dock.reposition()+raise_()`, `enhance_popover.reposition()`; one reposition after first `showEvent`. Verify: overlays positioned (not at 0,0), clickable.
- [ ] **T5.10 — PR5 verification.** B-key `wheelEvent` adjusts `size_slider`; Ctrl-wheel adjusts `zoom_slider`; `keyPressEvent` emits `b_key_status_changed`; trace a mask (brush/erase/fill, undo/redo, save) + Enhance + zoom. `pytest -q`; `graphify update .`; commit.

### PR6 — Modern Skeleton editor (`skeleton_correction_interface.py`)

- [ ] **T6.1 — Recolor scene markers (values only).** `QColor(57,255,20)` overlay → `QColor("#f0a868")` (`--skel`); endpoints → `#f0a868`; handles/selection/previews → `#c39af6` (`--accent`). **No `connect()` touched.** Verify: `pytest -q`; launch, markers recolored.
- [ ] **T6.2 — Import overlays + `_build_overlays` skeleton.** Same as T5.2, parented to `graphics_view`; called from `_build_ui`. Verify: import OK.
- [ ] **T6.3 — Tools → ToolRail.** Reparent `select`/`eraser`/`polyline`/`connect_button` + `mode_toggle` + `smooth_polyline_toggle`; keep `tool_group.addButton` loop + `select_button.setChecked(True)`. Verify: tool switching works (`buttonClicked`→`_on_tool_changed`).
- [ ] **T6.4 — Actions + overlay slider → FloatingDock.** Reparent `load_skeleton_button`/`undo`/`redo`/`clear_button`/primary `save_skeleton_button` + `opacity_slider`. Verify: all fire.
- [ ] **T6.5 — Conditional eraser slider.** Reparent `eraser_slider` into the dock; visibility toggled in `_on_tool_changed` when eraser active. Verify: slider shows only in eraser mode.
- [ ] **T6.6 — Top-center polyline prompt.** Inline overlay with `finish_polyline_button` + `cancel_polyline_button`, shown only mid-polyline (drive from `_update_polyline_buttons_enabled`). Keep their `.clicked`. Verify: polyline mode shows "Polyline · N points" + Finish/Cancel; Enter/Esc work.
- [ ] **T6.7 — `norm_controls` → EnhancePopover.** Reparent + toggle button; **keep `apply_button.clicked.connect` exactly**. Verify: normalization applies.
- [ ] **T6.8 — `status_label` relocate.** Move the hint into the action bar / a thin overlay label; preserve the object (`_update_status_label`). Verify: status text updates.
- [ ] **T6.9 — Delete `QGroupBox` chrome + `resizeEvent`.** Remove the 5 wrappers + stylesheet; drop `QGroupBox` import if unused; rewire `_build_ui` to parent overlays to `graphics_view`; add `resizeEvent` reposition+raise_ + initial reposition. Verify: launch, no leftover panel; overlays positioned.
- [ ] **T6.10 — PR6 verification.** `undo_shortcut`/`redo_shortcut` fire; programmatic `polyline_button.setChecked(True)` path works; full skeleton edit session (select/erase/polyline-finish/connect, smooth, save). `pytest -q`; `graphify update .`; commit.

---

## Cross-cutting risks + guards

1. **Three QStackedWidgets** — `app_stack` (welcome/shell) ≠ `right_panel` (4 fixed stage indices) ≠ `display_page_stack` (nested in `right_panel` idx0). Guard: `on_loading_finished` flips `display_page_stack` only; never redirect it to `app_stack`; keep the empty→display flip.
2. **`task_progress` single instance** — `_ProgressBarShim` + `loading_progress_bar` property + load flow bind one `TaskProgressWidget`. Guard: statusline refactor keeps the `self.task_progress` attr + does not recreate it.
3. **Ribbon routes through toggle methods** — `toggle_mask_tracing` manages `_mask_tracing_signals_connected` (F-015); viz stages must call `toggle_root_*_visualization` (Dash thread + OpenGL lifecycle), not raw `switch_right_panel('display')` (which leaks the Dash server). Guard: ribbon→toggle methods; raw display switch only for plain Library/Mask/Measure.
4. **Button-attr setters stay valid** — code `setText`s `toggle_*_button` and `visualize_*_button`. Guard: assign ribbon buttons to those exact attr names (or re-parent originals).
5. **`view_mode_combo` is source of truth** — `currentIndexChanged→update_display_mode`. Guard: hide, never delete; seg forwards via `setCurrentIndex` (not a parallel connect → avoids double-fire).
6. **Parented overlays render at (0,0)** unless repositioned; `EnhancePopover` starts hidden. Guard: every overlay gets `resizeEvent→reposition()+raise_()` + one initial reposition; wire a control to `enhance_popover.toggle`.
7. **Reparenting must not touch `connect()` lines** — move construction + `addWidget` only; every `self.<attr>` keeps its name; `tool_group.addButton`/`tool_button_group` membership must survive (exclusivity + programmatic `setChecked`).
8. **Dynamic viz widgets + Dash thread lifecycle** — viz widgets appended past `right_panel` idx3, removed via `removeWidget+deleteLater`; `closeEvent` closes both Dash servers. Guard: new shell/welcome must not destroy `right_panel` or its children before `closeEvent` runs (avoids "QThread destroyed while running").
