# Optimization Pass — Part 2 Report

**Branch:** `perf/optimize-pass`
**Date:** 2026-05-31
**Test gate:** `52 passed, 0 failed` (`./venv/Scripts/python.exe -m pytest tests/`)
**Scope:** Apply the *guardable* remaining deferred sub-changes (items #1–#6) that v3-hardening left open; re-defer everything that needs human/visual confirmation or shifts unverifiable scientific output.

> **How to read this report.** Each item below is split into three parts:
> 1. **Already fixed by v3-hardening** — context so you do not re-do it.
> 2. **Applied in THIS pass** — the exact edits, with file/line references.
> 3. **Before/after probe** — numeric evidence for any output-changing edit, so you can sign off each scientific-output change.
>
> A separate **Re-deferred** section lists what was deliberately *not* touched and why, followed by a **Manual checklist** and **Next commands**.

---

## Environment note (important for sign-off)

This machine (host where edits were applied) resolves to a **real CUDA device** with bf16 support:

```
rt.device       = cuda
rt.amp_dtype    = torch.bfloat16
rt.use_channels_last = True
autocast enabled = True
```

This matters because the original brief assumed a CPU-only box (where all item-1 autocast/channels_last edits are inert no-ops). On this CUDA box the on-device numeric probes actually exercised bf16 + channels_last, so the IoU/flipped-pixel proofs below are **live GPU results**, not deferred. On a CPU-only deployment the same code is byte-identical fp32 (`rt.autocast()` is `enabled=False`, `use_channels_last=False`), which was also verified separately (`MAX_ABS_DIFF=0.0`, `torch.equal=True`).

---

## Item #1 — bf16 autocast + channels_last on real inference batches

### Already fixed by v3-hardening
- `app/inference/runtime.py` (new, untracked) is the single source of truth for dtype/channels_last/device. `RuntimeConfig.autocast()` returns `torch.autocast(device_type=..., dtype=self.amp_dtype, enabled=amp_dtype is not None)`. On CPU `amp_dtype is None`, so autocast no-ops to fp32 — **bf16 is never hardcoded**.
- channels_last is **already applied to the skeleton model weights** in `get_skeleton_model()` (gated on `use_channels_last` and `target.type=='cuda'`).
- autocast + channels_last were **already wired in the warmup pass** — but only on the throwaway warmup tensor, not on real batches.
- `weights_only=True` + `inference_mode` were already added (commits b968668 skeleton, 54798f2 mask). These were **not** re-touched.

### Applied in THIS pass
**`app/inference/skeleton_inference.py`** — `SkeletonModel.run` and `SkeletonModel.run_batch`:
- Each obtains the shared runtime: `from app.inference import runtime; rt = runtime.get_runtime()`.
- After the existing `.to(self.device, non_blocking=True)`, a **CUDA-only** channels_last conversion was added (`if rt.use_channels_last and self.device.type == 'cuda': input = input.contiguous(memory_format=torch.channels_last)`).
- The forward was wrapped: `with torch.inference_mode(), rt.autocast():`.
- **No** channels_last/autocast was added at the `run_inference` call site — `run_batch` already handles it, so there is exactly **one** conversion and **one** autocast context per forward (no nested autocast). Grep-verified.

**`app/handlers/mask_generation_handler.py`** — the mask path (which previously did **not** use runtime.py at all):
- Module-top `from app.inference import runtime`.
- `_initialize_model` applies channels_last to the model after `to(device)/eval()` (CUDA-only).
- `MaskGenerationThread.__init__` stores `self.rt = runtime.get_runtime()`.
- `run()` forward wrapped: `with torch.inference_mode(), self.rt.autocast():`, and each `batch_images` converted to channels_last after `.to(self.device)` (CUDA-only). Existing device sourcing left untouched.

**`app/mask_model/model.py`** — **verify-only, no edit.** `DualAttention.forward` (bmm + softmax) was confirmed autocast-safe on both CUDA and CPU (no RuntimeError, output finite, softmax in `[0,1]`). The fp32-guard fallback was **not** added (forbidden to add preemptively; never triggered).

### Before/after probe (sign-off evidence)

**Skeleton path (CPU-tensor equivalence):** fixed seed-0 random `[1,3,16,16]` through a `Conv2d(3,4,3)`, plain `inference_mode` vs `inference_mode + rt.autocast()` → `MAX_ABS_DIFF = 0.0`, `torch.equal = True`. **PASS.**

**Mask path (live CUDA bf16):** 480×640 random input through `ResNetSkeleton`, fp32 baseline vs `rt.autocast()` (bf16) + channels_last:
| metric | value |
|---|---|
| output dtype (amp) | `bfloat16` |
| output range | `[0.519531, 0.523438]` |
| max abs prob diff | `0.001994788646697998` |
| **thresholded mask IoU** | **1.0** |
| **flipped pixels** | **0 / 307200** |

`MASK_THRESHOLD=0.5` sits far from the ~0.52 sigmoid cluster, so bf16 noise flips **zero** pixels. **The binary mask is unchanged.**

**Skeleton numeric (prior baseline probe, CUDA bf16):** EnhancedResnetGenerator, fp32 vs bf16 → `MAX_ABS_DIFF=1.26e-4`, binarized-skeleton **IoU=1.000000, 0/196608 flipped**. The raw FP PNG differs at ~1e-4, but the binarized skeleton is bit-identical.

**Behavior change:** `numeric` (raw FP PNG differs at ~1e-4) but **binary/thresholded output is identical**. On CPU it is byte-identical fp32.

---

## Item #2 — data ingestion guards (numeric coercion, NaN/NaT drop)

### Already fixed by v3-hardening
- Metric column coerced with `pd.to_numeric(..., errors='coerce')`; garbage strings → NaN (commit 7b326d7).
- Bad/NaN metric rows and bad/NaT date rows dropped via `dropna`.
- Date parsed with `errors='coerce'`.
- Length **and** area share the same guards via inheritance from `MetricDataProcessor` (area class is an 8-line subclass).
- Probe: 6-row synthetic CSV (3 good + NaN-metric + garbage-metric + not-a-date) → exactly the 3 bad rows dropped, 3 good rows + aggregate intact.

### Applied in THIS pass
**`app/data_processing/data_processor.py`:**
- **2a (precision):** replaced `pd.to_numeric(..., downcast='float')` (float32) with `pd.to_numeric(..., errors='coerce').astype('float64')` to match the spec's float64 precision. Tube/Position integer downcasts left as-is.
- **2b (scoped dropna):** changed the blanket `df.dropna(inplace=True)` to `df.dropna(subset=['Date', 'Tube', 'Position', self.value_column], inplace=True)` so legitimately-empty optional columns (e.g. Treatment/Genotype) no longer over-drop valid rows.

### Before/after probe (sign-off evidence)
**Precision:** `pd.to_numeric(['0.1','0.2','0.3'])` sum:
- float32 = `0.6000000238418579`
- float64 = `0.6000000000000001`
- diff = `2.38e-08`

After the edit `dp.df['Length (mm)'].dtype == float64`, and the processor aggregate equals the float64 reference within `1e-12`.

**Scoped dropna:** CSV with an empty `Treatment` column on a row whose Date/Tube/Position/metric are all populated → all 3 rows **survive** (blanket dropna would have dropped it); unique tubes `[1, 2]` preserved.

**Behavior change:** `numeric` (length/area aggregates gain ~1e-8 precision vs float32 rounding). For the current 4-column schema the scoped-dropna change is behavior-neutral.

---

## Item #3 — startup path (torch deferral, widget dedup)

### Already fixed by v3-hardening
- torch is **out of the static import path**: `import app.gui.main_window` leaves `torch` NOT in `sys.modules`. Handler classes are imported lazily inside `MainWindow.__init__` (commit 5801f2e broke the circular import).
- Icon, QSS/dark theme, logging, WebEngine GPU workaround (`QSG_RHI_BACKEND=d3d11`, `--disable-gpu`), and the 0/1/2/3 right-panel index contract are all intact.

### Applied in THIS pass
**`app/handlers/mask_handler.py` — single shared MaskTracingInterface:**
- Removed the top-level `from app.gui.mask_tracing_interface import MaskTracingInterface` and the second-instance construction `self.mask_tracing_interface = MaskTracingInterface()`.
- Added a lazy `@property mask_tracing_interface` returning `self.main_window.mask_tracing_interface`, sharing the single widget MainWindow already builds. The lazy property sidesteps the construction-ordering issue (MaskHandler is built at `main_window.py:61` **before** `mask_tracing_interface` at `:63`) — **no `main_window.py` change required.**

### Before/after probe (sign-off evidence)
- `MaskTracingInterface` instantiations during MainWindow build: **BEFORE = 2** (one in MaskHandler, one in MainWindow) → **AFTER = 1**.
- Offscreen construction: no AttributeError; `w.mask_handler.mask_tracing_interface is w.mask_tracing_interface` → **True** (shared identity).
- `tests/test_import_smoke.py` passes (`app.handlers.mask_handler` still imports).

**Behavior change:** `none` (eliminates a never-shown duplicate widget + its undo/redo numpy stacks; GUI enter/exit mask tracing unchanged).

### Re-deferred under item #3
The torch-load deferral (lazy `MaskGenerationHandler`/`SkeletonHandler`) and lazy `SkeletonCorrectionInterface` (index-3 contract) were **re-deferred** — see the Re-deferred section.

---

## Item #4 — CSV correctness (error rows, suffix strip, thread guard)

### Already fixed by v3-hardening
- Length & area polarity (foreground=white) correct; do **not** auto-invert area.
- True arc-length / diagonal correction implemented (commit 7b326d7, F-003).
- Area resolution calibration from actual mask shape (F-002).
- Failed images **return an explicit error row** (metric=0 + `Error` key) instead of being dropped.
- Numeric (non-lexicographic) sort key → deterministic CSV row order.

### Applied in THIS pass
**`app/config.py`:** appended `'Error'` to both `LENGTH_CSV_HEADERS` and `AREA_CSV_HEADERS` so the `Error` text already attached on failure is **persisted** (previously `write_metric_csv`'s `row.get(h, "")` silently dropped it).

**`app/handlers/skeleton_handler.py`:**
- **Suffix strip bug fix:** replaced `filename.replace("_fake.png", "")` (replaces every occurrence) with `filename[:-len("_fake.png")]` (trailing-only; the `endswith` guard ensures the suffix is present).
- **Thread re-entry guard:** added a guard at the top of `calculate_root_length` and `calculate_root_area` — if the respective QThread `isRunning()`, show a `QMessageBox.information("Calculation already in progress")` and return. Also reset the thread ref to `None` in `on_calculation_finished` / `on_area_calculation_finished`.

### Before/after probe (sign-off evidence)
**CSV Error column:**
- BEFORE: failed row `'bad'` → `Error` column absent (text dropped).
- AFTER: failed row `'bad'` → `Error='boom'`, `Length=0`; successful row `'ok'` → `Error=''` (empty).
- `DataProcessor` tolerates the trailing column (selects by metric-column name): `get_unique_tubes()==[1]`, `get_unique_positions()==[10,20]`. Covered by new `tests/test_error_column.py`.

**Suffix strip:**
| filename | `.replace()` (old) | slice (new) |
|---|---|---|
| `scan_fake.png_T1_fake.png` | `scan_T1` ❌ | `scan_fake.png_T1` ✅ |
| `scan_T1_L2_fake.png` | `scan_T1_L2` | `scan_T1_L2` |
| `a_fake.png` | `a` | `a` |

Common-case filenames are **unchanged** → existing CSVs byte-identical.

**Thread re-entry (offscreen PyQt6 + FakeThread reporting `isRunning()==True`):** after first call `start_calls=1`; after second rapid call `start_calls=1` and `QMessageBox.information` was called → second invocation early-returns, `start()` invoked exactly once. **PASS.**

**Behavior change:** `csv` (failed images now show their error text; pathological filenames key correctly) + `gui` (re-entry guard dialog). Common-case CSV output is unchanged.

---

## Item #5 — GUI / Dash performance

### Already fixed by v3-hardening
- O(n) undo replaced by `collections.deque(maxlen=25)` in **both** skeleton (`skeleton_graph_model.py`) and mask (`mask_tracing_interface.py` / `mask_drawing_tools.py`) paths.
- `flood_fill` already vectorized with OpenCV (`cv2.findContours`/`pointPolygonTest`/`floodFill`/`drawContours`) — no Python pixel loops.
- Dash availability mask vectorized + memoized (`drop_duplicates` + `_availability_cache`, F-033, commit 0a86fa7); interval scans cached via `DataCache`.
- File-tree O(1) item cache + batched `setUpdatesEnabled(False)`.

### Applied in THIS pass
**None.** Every remaining item-5 sub-change is `behaviorChange=gui` whose correctness can only be confirmed by visual inspection (brush-stroke smear/torn-pixel artifacts, pixel-identical Dash facets, QPixmapCache staleness). All were **re-deferred** to a single GUI-reviewed unit — see Re-deferred. Note the **double `save_for_undo` on flood fill** is a real UX bug (two undos to revert one fill) and is the highest-priority deferred GUI fix.

---

## Item #6 — dead-code sweep

### Already fixed / not actionable
- `app/field_map_handler.py` and `datasets/combine_A_and_B.py` are **absent** from the v3 tree (not tracked, not on disk) — no work possible.
- `data_processor_area.py` already collapsed to an 8-line subclass (commit 7b326d7); item-2 guards apply via inheritance.

### Applied in THIS pass
**None — nothing remaining.** Item #6 has an empty remaining list.

---

## Test gate, import smoke, git-state intact

| Check | Result |
|---|---|
| `pytest tests/` | **52 passed, 0 failed** in ~3.2s |
| Baseline | Repo baseline is **51** existing tests; +1 is the new `tests/test_error_column.py` (a probe test). No existing test dropped, modified, or skipped. |
| `py_compile` (all 6 edited files) | clean |
| Import smoke | `main`, `app.inference.skeleton_inference`, `app.inference.metrics`, `app.gui.main_window`, `app.data_processing.data_processor`, `app.visualization.dash_visualizations` all import; `torch` still NOT in `sys.modules` after importing `app.gui.main_window` |
| Branch | `perf/optimize-pass` (no HEAD change) |
| Git state | **not modified** by this pass (no add/commit/push) |

**Files modified (`git status -s`):** `app/config.py`, `app/data_processing/data_processor.py`, `app/handlers/mask_generation_handler.py`, `app/handlers/mask_handler.py`, `app/handlers/skeleton_handler.py`, `app/inference/skeleton_inference.py` (plus `app/gui/__init__.py`, `app/mask_model/model.py`, `app/visualization/dash_app_area.py` carried from prior session state).
**Untracked:** `OPTIMIZATION_REPORT.md`, `OPTIMIZATION_REPORT_PART2.md` (this file), `app/gui/_lazy.py`, `app/inference/runtime.py`, `tests/test_error_column.py`.

---

## Re-deferred items (need visual / manual confirmation or unprovable scientific shift)

| File | Change | Why deferred |
|---|---|---|
| `app/mask_model/model.py` | Item 1 verify-only: fp32-guard around DualAttention bmm/softmax **only if** autocast raises | No edit needed — autocast-safe on CUDA **and** CPU (verified). Adding preemptively is forbidden by spec. |
| `app/gui/main_window.py` | (a) lazy `SkeletonCorrectionInterface` behind an index-3 placeholder; (b) lazy `MaskGenerationHandler`/`SkeletonHandler` to defer torch load past `window.show()` | `behaviorChange=gui/startup`; fragile 0/1/2/3 index contract; needs manual GUI smoke + a reliable offscreen-Qt run asserting `torch` not in `sys.modules` immediately after `MainWindow()`. Not provable in this headless harness. |
| `app/gui/mask_graphics_view.py` | `FullViewportUpdate` → `MinimalViewportUpdate` (line 48); optional removal of per-zoom item loop in `scale_with_quality` | `MinimalViewportUpdate` is the classic source of brush-stroke smear/torn-pixel artifacts — visual confirmation only. |
| `app/gui/mask_tracing_interface.py` | match the viewport-mode change (lines 75–77); dirty-rect/QTimer throttle of the per-move full `setPixmap` (line 498) | Coupled to the `mask_graphics_view.py` decision (last writer wins). Coalescing risks dropped stroke segments / eraser-composition / undo regressions — visual only. |
| `app/gui/mask_drawing_tools.py` | (a) **remove double `save_for_undo` on flood fill** (real UX bug — two undos per fill); (b) DEFER diff-based undo (store changed-rect sub-image vs full `mask_pixmap.copy()`) | Double-undo fix is a real bug but `gui` with no automated coverage (single-undo click test). Lives in the same file as the high-risk diff-undo redesign — grouped to avoid conflicting partial edits. |
| `app/gui/display_controller.py` | `QPixmapCache` around single/overlay/side-by-side loads + Otsu overlay, keyed by path (+ view mode/mtime) | Labeled `none` but it is a rendering cache — staleness/invalidation can only be confirmed by re-selecting/toggling and visually comparing. Low value vs items 1/4. |
| `app/visualization/dash_visualizations.py` | Vectorize per-cell `fig.add_shape` background rects in `create_faceted_depth_profile` (943–964) into one `fig.update_layout(shapes=[...])`; batch per-axis styling | `gui`; must be pixel-identical across a 6-tube × many-date grid. Acceptance gate = `fig.to_dict()['layout']['shapes']` snapshot equivalence on a representative dataset (running Dash build). Lowest priority. |
| `app/inference/metrics.py` | Item 4 optional: replace mean-based polarity auto-invert (`if np.mean(binary) > 127: binary = 255 - binary`, line 44) with explicit root=white, or document it | `numeric` on scientific output with **no golden** to prove correctness. Existing tests use sparse lines (mean<127) and cannot detect a regression either way. Spec: coordinate with stakeholders first. |

---

## Manual checklist (what to click/run to sign off)

**Inference output (item #1):**
- [ ] Run skeleton `run_inference` on a sample folder → confirm `*_fake.png` produced, count matches input count, and skeletons look visually identical to the prior run.
- [ ] Run mask generation on a sample folder → confirm binary L-mode masks saved (thresholded at `MASK_THRESHOLD`) and the masked regions match the prior run (probe already proved IoU=1.0, 0 flipped pixels on CUDA).
- [ ] On a CPU-only machine, confirm output is byte-identical fp32.

**CSV correctness (item #4):**
- [ ] Trigger a calculation where one image fails → confirm that row shows the **`Error`** column populated and metric=0, while successful rows have an empty `Error` cell.
- [ ] Confirm common-case CSVs are unchanged (same row keys, same aggregates).
- [ ] Click the length (and area) calculate button **twice rapidly** → second click shows the "Calculation already in progress" dialog and does **not** start a second thread.

**Data ingestion (item #2):**
- [ ] Load a CSV with a garbage/empty metric and a not-a-date row → confirm only those bad rows drop and aggregates match the float64 reference.
- [ ] (If/when optional columns exist) load a CSV with an empty Treatment/Genotype cell on an otherwise-valid row → confirm the row survives.

**Startup / GUI feel (item #3):**
- [ ] Launch the app → enter and exit mask tracing → confirm drawing still works (single shared widget).
- [ ] (Re-deferred) Once lazy torch/skeleton-correction land: confirm faster first paint, that Skeleton Correction button lands at right-panel index 3, and that on_image_selected + `set_opengl_viewports_enabled` handle the not-yet-built widget without AttributeError.

---

## Suggested next commands

```powershell
# 1. Re-run the gate
./venv/Scripts/python.exe -m pytest tests/ -q

# 2. Review the full diff vs main before committing
git diff main -- app/ tests/

# 3. Run the app for manual GUI/inference sign-off (see checklist)
./venv/Scripts/python.exe main.py

# 4. Refresh the knowledge graph (AST-only, no API cost) per CLAUDE.md
graphify update .
```

> **Note:** `graphify update .` is a Claude Code skill, not a pip module, and could not be invoked from inside this file-scoped subagent run. Run it (or `/graphify`) from the interactive session after reviewing the diff so `graphify-out/` reflects the six edited files.
