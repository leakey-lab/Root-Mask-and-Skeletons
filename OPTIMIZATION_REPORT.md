# Optimization Report — `perf/optimize-pass`

Repo: `D:/Root-Mask-and-Skeletons` · Branch: `perf/optimize-pass`

This pass applied a set of **safe, behavior-preserving** optimizations and added two shared infrastructure modules. Higher-risk wins that change numeric output, CSV artifacts, or GUI behavior were deliberately **deferred** for your review (Section 3). All changed files compile and import cleanly; GUI and numeric correctness still require a **manual run** (Section 4).

---

## 1. What changed & expected impact

### New shared modules
| Module | Purpose |
|---|---|
| `app/inference/runtime.py` | Single runtime auto-tuner. `get_runtime()` (`lru_cache(maxsize=1)`) is the one source of truth for device / amp_dtype / channels_last / pin_memory / num_workers / should_compile. Auto-detects **CUDA vs CPU with full CPU fallback, never hardcodes a GPU**. Also hosts the cached `get_skeleton_model(device)` singleton + `warmup()`. |
| `app/gui/_lazy.py` | PEP 562 lazy-import helper for the GUI package — maps each re-exported name to its submodule and imports on first attribute access. |

### Inference / GPU
- **Skeleton model singleton cache** (`skeleton_inference.py`): the model was previously **rebuilt + `torch.load` + `load_state_dict` + GPU-uploaded on every `run_inference()` call**. Now built once per process via `runtime.get_skeleton_model()`. Expected impact: eliminates a full model rebuild+reload (hundreds of ms to seconds) on *every* inference invocation after the first.
- **Removed per-batch `torch.cuda.empty_cache()`** on both inference paths (skeleton steady-state loop; mask handler `finally`). These forced `cudaFree`/`cudaMalloc` churn each batch, defeating the caching allocator. `del output_batch` retained so fixed-size buffers are reused. Recovery `empty_cache()` kept only in the OOM `except` path. Expected impact: removes a per-batch GPU sync stall; smoother, faster steady-state throughput.
- **Single-GPU: no DataParallel.** `move_net_to_device` now does plain `net.to(device)` and only wraps in `DataParallel` when `len(gpu_ids) > 1`, removing scatter/gather overhead on the common 1-GPU case.
- **cudnn.benchmark + warmup**: `get_runtime()` sets `torch.backends.cudnn.benchmark=True` and `torch.set_float32_matmul_precision('high')` once on CUDA; `warmup()` runs a dummy 256×256 forward so cuDNN autotune + first-kernel JIT is amortized off the user's first real batch.
- **`non_blocking=True`** on H2D copies in `SkeletonModel.run()` / `run_batch()`, pairing with pinned memory for transfer/compute overlap.
- **`pretrained=False` default** for `ResNetSkeleton` (`model.py:55`): removes a **dead ImageNet ResNet18 download/load** when constructed with defaults (production caller already passed `pretrained=False`).

### Startup
- **Lazy `app/gui/__init__.py`** (PEP 562 delegation to `_lazy.py`): importing `app.gui` (which happens transitively at launch) **no longer eagerly pulls cv2 / scipy / skimage / dash / plotly / pandas**. Verified: post-change, `import app.gui` loads **none** of those heavy modules eagerly. Expected impact: materially faster cold start; heavy libs load only when the feature that needs them is touched.

### UI / bugs / dedup
- **Throttled progress signal** (`mask_generation_handler.py`): `progress.emit` now fires only when the integer percent changes, instead of every batch — fewer cross-thread Qt signal hops and UI repaints.
- **Typed `finished` signal**: replaced the brittle `"...|FAILURES:"` string-packed payload with `finished = pyqtSignal(str, list)` emitting `(output_dir, failed_images)`; slot decodes a structured list.
- **`generate_masks()` re-entry guard**: returns early with a status message if a generation thread is already running (prevents double-launch / racing threads).

### Dash
- **Vectorized stacked-bar** (`dash_app_area.py`): replaced the O(dates×tubes) per-cell boolean filter with a single `pivot(index='Tube', columns='Date', values='Area (mm²)').reindex(...).fillna(0)`; per-date traces are now `pivot[date].to_numpy()`. Numerically identical (missing cells → 0 as before; tube order preserved). Expected impact: chart build drops from quadratic to a single grouped pivot.
- **`compress=True`** added to the area Dash app (matching the length app) to gzip callback responses.

### Data
- **Vectorized label columns** (`data_processor.py`): removed both `df.apply(axis=1)` passes for `tube_date` / `tube_position`; now `df['Date'].dt.strftime` + Series string concatenation. Verified **byte-identical** output vs the row-wise version.
- **Cached uniques**: `get_unique_*` accessors no longer re-run `sorted(unique())` on every call — computed once via `_compute_unique_caches()` in `__init__`, guarded for the empty-DataFrame failure path.

---

## 2. Verification status

| Check | Result |
|---|---|
| `py_compile` on all 8 changed files | **PASS** (exit 0, no output) |
| Whole-app import smoke (`skeleton_inference`, `gui.main_window`, `dash_visualizations`) — no GUI/server launched | **PASS** (exit 0, no ImportError/SyntaxError) |
| Lazy-import effectiveness | **PASS** — `import app.gui` pulls none of cv2/scipy/skimage/dash/plotly/pandas eagerly; `__all__` preserves all 15 names |
| Runtime auto-detect (this machine) | device=cuda, amp_dtype=bfloat16, channels_last=True, pin_memory=True, num_workers=0, should_compile=False (triton absent on Windows → graceful) |
| `app/runtime/device.py` check | **N/A** — no such module was created (runtime lives at `app/inference/runtime.py`) |

> **Still requires YOUR manual run.** There are **no GUI tests and no numeric/golden-output tests** in this repo. Compile + import smoke prove nothing crashed on load, but they **do not** confirm: (a) the GUI screens still render and behave correctly, or (b) generated masks/skeletons and length/area CSVs are still correct. Please run the manual checklist in Section 4 before merging. (Note: all auto-applied inference changes are designed to be numerically identical — same weights, same `eval()` model, single-GPU forward, `empty_cache` removal does not affect results — but please still spot-check one output.)

---

## 3. Deferred higher-risk proposals (awaiting your approval)

Each is a **real win** but touches numeric output, CSV artifacts, or GUI behavior, so it was **not** auto-applied.

1. **autocast / bf16 + channels_last on the inference forward** (`skeleton_inference.py` + mask forward) — *top GPU win.* Tradeoff: changes exact FP output of generated PNGs. Thresholded result should be effectively unchanged but **not bit-identical**; must be visually/numerically verified on a bf16 GPU. Infrastructure (`runtime.amp_dtype`) is already in place behind it.
2. **Data ingestion-correctness guards** (`data_processor.py`: `float64` metric, `dropna(subset=...)`, date-coercion) — Tradeoff: changes *which rows survive* and aggregate precision → changes scientific output. Must be validated against real CSV exports.
3. **Startup restructure** (`main.py` / `main_window.py` / `mask_handler.py`: deferred torch import, lazy skeleton-correction widget, shared mask-tracing widget — *currently built twice* at `main_window.py:63` + `mask_handler.py:8* — icon/QSS/logging, WebEngine GPU flag) — Tradeoff: reorders construction, right-panel widget indexing, button-enable lifecycle, and the WebEngine GPU workaround. GUI-sensitive; needs an app run to confirm no black WebEngine surface / missing widgets / init AttributeErrors. Group as one reviewed pass.
4. **CSV-artifact correctness fixes** (`metrics.py`, `root_length/area_inference_handler.py`, `skeleton_handler.py`: sort key, mask polarity, error rows, `_fake.png` matching, thread re-entry) — Tradeoff: each changes a produced artifact (row order, computed length, CSV schema/content, which files aggregate). Real fixes but alter scientific output / user-visible behavior.
5. **Large secondary perf cluster** (GUI hot-path repaint `FullViewportUpdate` + per-move `update_display`, undo-diff, flood_fill, `QPixmapCache`, file-tree thumbnails, Dash faceted O(n²) `add_shape` + per-interval mask scans / growth vectorization) — Tradeoff: each is a sizeable behavior-touching refactor (viewport update mode, scene reuse, cache invalidation, groupby rewrites) needing its own focused change + visual verification.

**Not actionable:** `app/data_processing/field_map_handler.py` and `datasets/combine_A_and_B.py` from the brief **do not exist** in this repo (only `data_processor.py` / `data_processor_area.py`). Paths appear stale or from another branch — confirm intended location before any work.

---

## 4. Manual-verification checklist

Run the app and confirm each path works **and feels faster**:

**Skeleton inference**
- [ ] Trigger skeleton inference on a small image set. First run may pause briefly (warmup); confirm it completes and writes skeleton PNGs.
- [ ] Run inference **again** in the same session — confirm it starts immediately (model is now cached, no rebuild/reload pause).
- [ ] Open one output skeleton PNG and confirm it looks correct (single-GPU path, no DataParallel).

**Mask generation**
- [ ] Start mask generation; confirm the progress bar advances smoothly (throttled) and masks are written.
- [ ] Confirm the completion handler fires with the new typed signal — finished status shows, and any failed images are listed correctly.
- [ ] While a generation is running, click the generate button again — confirm it **does not** launch a second run (re-entry guard message).

**Mask tracing (interactive)**
- [ ] Open the mask tracing interface; draw/erase strokes; confirm rendering and undo work (unchanged behavior — repaint optimization was deferred).

**Area inference + Dash views**
- [ ] Run area inference; confirm area CSV is produced.
- [ ] Open the **area** Dash dashboard → stacked-bar chart: confirm bars/values/tube ordering look correct and the page loads fast (vectorized + gzip).
- [ ] Open the **length** Dash dashboard: confirm it still renders normally.
- [ ] In each dashboard, exercise the date/tube/treatment/genotype filters: confirm labels like `Tube 7 (2024-01-05)` and positions like `Tube 7_L3` render correctly (vectorized label build) and filtering is responsive.

**Cold start**
- [ ] Launch the app from cold; subjectively confirm startup feels quicker (heavy libs no longer eagerly imported via `app.gui`).

---

## 5. Suggested next commands

```powershell
# Review everything this branch changed vs main
git diff main...perf/optimize-pass

# Launch the app for the manual checklist above
./venv/Scripts/python.exe main.py

# After confirming, keep the knowledge graph current (AST-only, no API cost)
graphify update .
```

If the manual checklist passes, this branch is ready to commit/merge. To pursue the deferred wins, approve them individually (Section 3) — start with **#1 (bf16 autocast)** for the biggest remaining GPU gain and **#3 (startup restructure)** for the biggest remaining cold-start gain.
