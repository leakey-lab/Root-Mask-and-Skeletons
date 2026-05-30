# Code Review — Root-Mask-and-Skeletons

*Plant-root phenotyping desktop application — comprehensive multi-dimensional code review*

---

## 1. Executive Summary

**Root-Mask-and-Skeletons** is a PyQt6 desktop application that turns minirhizotron root images into quantitative phenotyping deliverables: it generates root **masks** (Pix2Pix/segmentation model), extracts **skeletons**, lets the user interactively correct those skeletons, computes **root length (mm)** and **root area (mm²)**, and presents the results through embedded Dash/Plotly visualizations and flat-CSV exports.

**Scope reviewed.** ~37 source files, ~11,000 LOC across six subsystems: ML inference (`app/inference`, `app/mask_model`), batch handlers (`app/handlers`), the editor GUI (`app/gui`), visualization (`app/visualization`), data processing (`app/data_processing`), and packaging (`scripts/`).

**Severity counts (97 confirmed findings + 7 supply-chain items):**

| Severity | Count |
|---|---|
| Critical | 1 |
| High | 22 |
| Medium | 41 |
| Low | 32 |
| Info | 1 |
| Security/Supply-chain (SS-01…SS-07) | 7 |
| **Refuted / false positives** | **9** (of 106 raised) |

**Top 5 must-fix (in order):**

1. **F-002 (CRITICAL) — Root-area calibration assumes a fixed 640×480 mask.** `calculate_root_area` derives the per-pixel area from hardcoded constants and never reads `mask.shape`. Any mask at a different resolution silently reports an area wrong by the *square* of the resolution ratio (a 1280×960 mask → 4× true area). This is silent scientific-output corruption — the worst failure class for a measurement instrument — and it leads the must-fix list because the application's entire purpose is producing trustworthy mm² numbers.
2. **F-003 (HIGH) — Length uses a raw skeleton pixel count as arc length.** `np.sum(skeleton)/255` weights every 8-connected step as length 1, ignoring the √2 cost of diagonal steps. This biases reported length **low by ~15–20%** on typical roots (≤29.3% worst case), present even on the default pipeline, and it does not cancel in cross-treatment comparisons.
3. **F-004 (HIGH) — Aspect-ratio resize distorts skeleton geometry and swaps width/height** in the PIL `resize` call upstream of every length measurement.
4. **F-006 (HIGH) — Saving a 1-px skeleton with `INTER_NEAREST` downscale fragments lines**, corrupting the very metric the editor exists to correct.
5. **F-023 / SS-01 (HIGH) — `torch.load` without `weights_only=True`** on both model loaders, an arbitrary-code-execution vector made concrete by the model weights being shipped as a committed binary blob (SS-03).

**Overall health verdict: AT RISK.** The engineering scaffolding is competent — long operations are correctly threaded, the editor is feature-rich, and the happy path works. But the application's *scientific core is not trustworthy as shipped*: the headline length metric is systematically biased, the area metric is one non-default resolution away from silent corruption, both depend on a single undocumented and duplicated calibration, and the export carries none of the provenance needed to reproduce or audit a number. Combined with no CI, no tests, no static analysis, and ~65 silent `except` blocks, defects reach users undetected. The codebase is **not release-ready for publishable measurements** until Phase 4 (scientific re-validation) is complete and gated by regression fixtures. The remediation is well-bounded and sequenced below.

---

## 2. Methodology

This review was conducted as a multi-phase, multi-agent audit:

- **Phase A — Reconnaissance.** Whole-tree structural mapping, dependency inventory, and identification of the 13 review dimensions (D1–D13 below) and the scientifically load-bearing code paths.
- **Phase B — Dimensional deep-dives.** Per-dimension agent swarms read the exact source at each candidate site, extracting line-anchored evidence rather than pattern-matching.
- **Phase C — Adversarial verification.** Every candidate finding was handed to a second, independent agent tasked with *refuting* it: re-reading the cited lines, checking for guards/callers that neutralize the issue, and confirming the failure is reachable on a real code path. Numeric claims (the √2 bias, the *k²* area error, the pitch anisotropy) were re-derived from first principles and cross-checked against the literal constants in the code.
- **Phase D — Editorial synthesis.** Surviving findings were severity-rated, de-duplicated, cross-linked, and assembled into this report and the remediation roadmap.

**Agent fleet.** ~70+ agents were spawned across model tiers — lighter models for breadth/enumeration, frontier models for the scientific-validity proofs and the adversarial refutation rounds.

**False-positive rate.** Of **106** candidate findings raised, **9 were refuted** during adversarial verification (see §11), for a false-positive rate of **9/106 ≈ 8.5%**. The 97 surviving findings are the basis of this report.

---

## 3. Findings by Severity

### Critical

| ID | Dim | Location | Title |
|---|---|---|---|
| F-002 | Scientific validity | `root_area_inference_handler.py:25` | Root-area calibration assumes mask is always 640×480 regardless of actual mask size |

### High

| ID | Dim | Location | Title |
|---|---|---|---|
| F-003 | Scientific validity | `root_length_inference_handler.py:38` | Averaging anisotropic X/Y scale + raw pixel count is physically invalid for length on non-square pixels |
| F-004 | Scientific validity | `skeleton_inference.py:125` | `save_image` aspect-ratio resize distorts geometry and swaps width/height in PIL `resize` |
| F-005 | Architecture | `skeleton_inference.py:26` | `CHECKPOINTS_DIR` is relative; model load fails unless CWD is repo root |
| F-006 | Scientific validity | `skeleton_correction_interface.py:782` | Save downscales 1-px skeleton with `INTER_NEAREST`, fragmenting lines / corrupting length |
| F-007 | Interoperability | `skeleton_correction_interface.py:576` | `cv2.imread/imwrite` fail silently on non-ASCII / Unicode paths on Windows |
| F-008 | Error handling | `dash_app.py:991` | `run_server` swallows all exceptions with bare `pass` across dash apps |
| F-009 | Correctness | `dash_app.py:791` | `IndexError` when `get_unique_tubes()` is empty on view switch |
| F-010 | Correctness | `dash_visualizations.py:837` | `dict[...]` generic-alias subscript used as a constructor call in `update_layout` |
| F-011 | Correctness | `dash_visualizations.py:753` | Legend-dedup flag set unconditionally, silencing legends when first tube/date lacks data |
| F-012 | Correctness | `dash_data_cache.py:39` | `lru_cache` on an instance method leaks instances and returns stale data after DataFrame mutation |
| F-013 | Performance | `dash_image_utils.py:49` | Images re-read, decoded, resized, base64-encoded from disk on every hover event |
| F-014 | Error handling | `dash_app_area.py:261` | Area callback silently returns `dash.no_update` on all exceptions |
| F-015 | Correctness | `main_window.py:332` | Signal double-connection on repeated `toggle_mask_tracing` |
| F-016 | Correctness | `main_window.py:439` | `toggle_skeleton_correction` wipes legitimate green saved-mask tree colors |
| F-017 | Correctness | `display_controller.py:297` | Color-table threshold mismatch: Otsu mask is 0/255 but comment claims threshold 50 |
| F-018 | Memory/resources | `display_controller.py:298` | `QImage` wraps numpy buffer without keeping the array alive (use-after-free) |
| F-019 | Performance | `skeleton_graph_model.py:219` | O(n) `list.pop(0)` to trim undo/redo stacks instead of `collections.deque` |
| F-020 | Correctness | `root_length_visulization.py:182` | `_show_visualization` called twice per startup via two independent paths |
| F-021 | Error handling | `mask_generation_handler.py:119` | Bare `except` swallows per-batch exceptions; failed images hidden from user |
| F-022 | Performance | `mask_generation_handler.py:71` | DataLoader `num_workers>0` inside a QThread deadlocks/crashes on Windows |
| F-023 | Security | `mask_generation_handler.py:147` | `torch.load` without `weights_only=True` (untrusted-pickle code execution) |
| F-024 | Concurrency | `skeleton_handler.py:78` | `calculator_thread` overwritten before previous thread finishes → thread leak |
| F-025 | Concurrency | `generate_skeleton_handler.py:76` | `SkeletonGeneratorThread` stored as `self.thread` may be GC'd before completion |
| F-026 | Scientific validity | `root_length_inference_handler.py:146` | Failed images recorded as Length/Area = 0, indistinguishable from genuine zeros |
| F-027 | Correctness | `mask_drawing_tools.py:101` | ARGB32 buffer written back as RGBA8888 → channel swap in flood fill |
| F-028 | Correctness | `mask_drawing_tools.py:125` | flood_fill uses wrong-sized `local_mask` (h+2,w+2) to index image array |

### Medium

| ID | Dim | Location | Title |
|---|---|---|---|
| F-029 | Concurrency | `file_tree_manager.py:14` | Module-level `_item_cache` global → stale/deleted `QTreeWidgetItem` references |
| F-030 | Concurrency | `image_manager.py:70` | `QThread.terminate()` on loader without disconnecting `finished` signal |
| F-031 | Memory/resources | `main_window.py:53` | Duplicate `MaskTracingInterface` created in both `MainWindow` and `MaskHandler` |
| F-032 | Error handling | `dash_app.py:886` | Visualization callbacks return `dash.no_update` on exception, hiding errors |
| F-033 | Performance | `dash_visualizations.py:60` | `get_tube_date_availability` does O(tubes×dates) filters per call, no caching |
| F-034 | Scientific validity | `dash_visualizations.py:259` | Greedy ±3-day date-merge is order-dependent and sums merged totals, inflating field averages |
| F-036 | Correctness | `dash_app.py:936` | Hover interval boundary uses hardcoded offset `-9` / fixed date format |
| F-037 | Correctness | `dash_visualizations.py:39` | `parse_tube_selection` drops all valid results on first parse error |
| F-038 | Scientific validity | `dash_app_area.py:381` | Area profile uses mean where sum is appropriate (inconsistent with other charts) |
| F-039 | Performance | `dash_app_area.py:270` | Chart data recomputed every callback; declared `dcc.Store` cache unused |
| F-040 | Error handling | `data_processor.py:39` | Bare `except` returns empty DataFrame; date coercion drops rows silently |
| F-041 | Scientific validity | `image_normalization_interface.py:49` | Division by zero when `lower==upper` percentile in contrast stretching |
| F-042 | Correctness | `dash_image_utils.py:39` | `build_available_images_map` mixes string and tuple keys; dead set-building code |
| F-043 | Scientific validity | `mask_model/model.py:29` | Position-attention value aggregation transposed vs. standard self-attention |
| F-045 | Correctness | `skeleton_handler.py:62` | Length calc only matches `_fake.png`; silently no results if naming changes |
| F-046 | Error handling | `root_length_inference_handler.py:15` | `cv2.imread` None on unreadable/non-ASCII file → cryptic crash recorded as Length=0 |
| F-047 | Correctness | `root_length_inference_handler.py:15` | Otsu + `mean>127` inversion can invert sparse-root images, skeletonizing background |
| F-048 | Correctness | `skeleton_inference.py:62` | `is_image_file` extension check is case-sensitive, misses mixed-case extensions |
| F-049 | Error handling | `skeleton_inference.py:98` | No per-image error isolation; one corrupt image aborts the entire batch |
| F-050 | Performance | `generate_skeleton_handler.py:6` | `BATCH_SIZE=64` for Pix2Pix ResNet-9 256² risks GPU OOM on consumer hardware |
| F-051 | Error handling | `skeleton_correction_interface.py:786` | `save_skeleton` ignores `cv2.imwrite` return, emits success unconditionally |
| F-052 | Performance | `skeleton_correction_interface.py:681` | Full skeleton vectorization re-run on every display refresh |
| F-053 | Correctness | `skeleton_correction_interface.py:1226` | Select→edit erase of original polyline can wipe shared junction pixels |
| F-054 | Correctness | `display_controller.py:107` | Zoom drifts past min/max because clamp checks pre-multiplication value |
| F-056 | Performance | `display_controller.py:390` | New `QGraphicsScene` created on every display update — no scene reuse |
| F-057 | Correctness | `main_window.py:401` | Mask name/key derivation duplicated and fragile across handlers |
| F-058 | Correctness | `main_window.py:252` | `mask_exists` builds path from extension-less name → wrong `False` |
| F-059 | Correctness | `skeleton_graph_model.py:59` | `topology()` computes degrees on a thick (non-skeletonized) mask, inflating counts |
| F-060 | Correctness | `skeleton_graph_model.py:27` | `frozen` `SkeletonTopology` holds mutable `List` fields — immutability illusory |
| F-061 | Correctness | `skeleton_graph_model.py:100` | RDP simplification recursive with no depth limit → recursion-limit crash |
| F-064 | Concurrency | `root_length_visulization.py:67` | Data race on `self.server` between GUI `stop()` and server `run()` |
| F-066 | Correctness | `root_length_visulization.py:25` | Field-map / TubeIDS `DataProcessor` init silently removed → feature regression |
| F-067 | Correctness | `mask_drawing_tools.py:171` | Double undo push in flood_fill — caller and flood_fill both `save_for_undo` |
| F-068 | Correctness | `mask_generation_handler.py:176` | Mask input dir inferred from first loaded image, not `image_manager.original_folder` |
| F-069 | Scientific validity | `root_length_inference_handler.py:33` | FOV + resolution constants hardcoded in both handlers with no user config |

### Low

| ID | Dim | Location | Title |
|---|---|---|---|
| F-070 | Scientific validity | `display_controller.py:275` | `INTER_NEAREST` overlay resize can drop single-pixel root branches |
| F-071 | Scientific validity | `dash_visualizations.py:304` | Field 'variance' band is inter-tube SD, mislabeled as measurement uncertainty |
| F-072 | Scientific validity | `dash_visualizations.py:739` | Depth color normalization uses `interval_ends`, arbitrary 0.5 single-interval fallback |
| F-073 | Correctness | `dash_app.py:381` | `'downloadCsv'` is not a valid Plotly modeBar button; silently ignored |
| F-074 | Correctness | `dash_app.py:626` | Add Range silently discards reversed ranges (from > to), no feedback |
| F-075 | Correctness | `dash_visualizations.py:816` | Duplicate `y_max` assignment, two conflicting y-axis update paths |
| F-076 | Error handling | `dash_visualizations.py:981` | `traceback.print_exc()` leaks stack traces to stdout in production |
| F-077 | Performance | `dash_visualizations.py:895` | O(dates×tubes) `add_shape` background rects slow large-grid renders |
| F-078 | Docs | `dash_visualizations.py:1` | Missing schema/units docs; y-axis mislabeled 'Vertical Depth (cm)' |
| F-079 | Architecture | `dash_visualizations.py:54` | `DashVisualizations` bypasses `DataCache`, reads `data_processor.df` directly |
| F-080 | Security | `dash_visualizations.py:119` | Conditional values interpolated into inline HTML hover strings (injection surface) |
| F-081 | Error handling | `dash_app.py:50` | Pointless try/except wrappers that catch only to bare-re-raise |
| F-083 | Architecture | `root_length_visulization.py:78` | `RootLengthVisualization`/`RootAreaVisualization` are ~200-line near-identical copies |
| F-084 | Error handling | `root_length_visulization.py:196` | `_check_server` swallows all exceptions, polling forever with no escalation |
| F-085 | Correctness | `root_length_visulization.py:126` | TOCTOU in `_is_port_available` makes the port check redundant/falsely reassuring |
| F-086 | Correctness | `dash_app_area.py:203` | Profile view returns `no_update`, leaving UI in broken mixed state when tube is None |
| F-087 | Architecture | `dash_app_area.py:163` | Graph-container style dict duplicated across five callback branches |
| F-088 | Correctness | `dash_app_area.py:183` | `trigger_id` computed but unused in stacked and time view branches |
| F-089 | Architecture | `main_window.py:93` | Placeholder viz widget at panel index 2 never used; widgets appended dynamically |
| F-090 | Error handling | `main_window.py:112` | `set_opengl_viewports_enabled` silently swallows all exceptions |
| F-091 | Correctness | `main_window.py:304` | `load_results` `setCurrentText` to same value emits no signal → refresh no-ops |
| F-092 | Correctness | `display_controller.py:244` | `display_single_image` lacks `pixmap.isNull()` check unlike other paths |
| F-093 | Architecture | `display_controller.py:199` | `DisplayController` tightly coupled to `MainWindow` internals |
| F-095 | Correctness | `skeleton_graph_model.py:49` | `_ensure_uint8_binary` casts to uint8 before thresholding, zeroing float masks in [0,1] |
| F-096 | Performance | `skeleton_graph_model.py:66` | `_compute_degree_map` uses 8-neighbor Python loop instead of one convolution |
| F-097 | Scientific validity | `root_length_inference_handler.py:25` | `np.sum(skeleton)/255` instead of `np.count_nonzero` |
| F-098 | Correctness | `root_length_inference_handler.py:211` | Sort key `x['Tube'] or float('inf')` mis-sorts legitimate zero Tube/Position |
| F-099 | Interoperability | `root_length_inference_handler.py:227` | CSV written without explicit UTF-8; non-ASCII headers risk mojibake on Windows |
| F-101 | Correctness | `skeleton_inference.py:109` | `tensor2im` always indexes batch `[0]`, dropping all but first image |
| F-102 | Error handling | `mask_generation_handler.py:153` | `_initialize_model` swallows all exceptions, hiding init failure cause |
| F-103 | Docs | `skeleton_inference.py:516` | `run_inference` docstring says default 4 but signature default is 8 |
| F-104 | Correctness | `mask_tracing_interface.py:561` | Wheel-event brush cap (50) inconsistent with slider maximum (100) |
| F-105 | Correctness | `skeleton_correction_interface.py:1374` | `apply_normalization` rescales with `IgnoreAspectRatio`, distorting vs. skeleton overlay |
| F-106 | Error handling | `skeleton_correction_interface.py:408` | `load_image` does not validate `QPixmap` loaded (`isNull`) |

### Info

| ID | Dim | Location | Title |
|---|---|---|---|
| F-082 | Architecture | `dash_app.py:651` | `selected-tubes-store` can hold unlimited tube IDs while badges truncate at 50 |

---

## 4. Dimension-by-Dimension Notes

**D1 — Scientific Validity / Numeric Correctness.** The most consequential dimension. The headline length and area metrics are derived from a hardcoded, duplicated, unvalidated calibration (F-002/F-003/F-069) and the length metric carries a dominant ~15–20% √2 underestimate (F-003/F-097). Geometry is distorted upstream (F-004) and at save (F-006), aggregation choices are scientifically questionable (F-034/F-038), and failures masquerade as zeros (F-026). Full treatment in §5.

**D2 — Architecture.** A 1384-line god-object (`SkeletonCorrectionInterface`) and verbatim duplication of calibration and `parse_image_name` (3 copies) are the structural liabilities; near-identical visualization classes (F-083) and a checkpoint path that depends on CWD (F-005) compound it. Full treatment in §7.

**D3 — Concurrency.** Long jobs are correctly threaded, but `save_skeleton` blocks the GUI thread on resize+reskeletonize+IO, the Dash server is single-request serialized, the spawn handshake relies on fixed `QTimer` delays, and thread references are overwritten/GC-prone (F-024/F-025/F-030/F-064). Full treatment in §7.

**D4 — Performance.** Per-hover image re-encode (F-013), O(n) undo trim (F-019), instance-method `lru_cache` leak (F-012), O(tubes×dates) loops (F-033/F-077), scene churn (F-056), and a GIL-throttled thread pool for CPU-bound measurement (in-QThread `ThreadPoolExecutor`).

**D5 — Error handling.** ~65 `except Exception` clauses, many bare `pass` or `dash.no_update`, silence security- and correctness-relevant failures (F-008/F-014/F-021/F-032/F-040/F-049/F-084/F-090/F-102). Failures are invisible to the user and absent from any log.

**D6 — Memory / Resources.** `QImage`-wraps-numpy use-after-free (F-018), duplicate interface objects (F-031), and the leaking `lru_cache` (F-012).

**D7 — Security & Supply-Chain.** `torch.load` pickle RCE (F-023/SS-01), committed 45.7 MB unverified weights blob (SS-03), no CI/lint/static-analysis (SS-05), unpinned torch, HTML-string injection surface (F-080). Full treatment in §8.

**D8 — Interoperability / Broad Availability.** Two flat CSVs with no provenance, units baked into header glyphs, latin-1/default-encoding mismatch, no machine-readable formats, loopback-only Dash with no shareable artifact. Full treatment in §6.

**D9 — Correctness (UI/logic).** The largest count: signal double-connects (F-015/F-016), flood-fill channel swap and indexing (F-027/F-028), zoom clamp drift (F-054), path-derivation bugs (F-057/F-058), topology computed on thick masks (F-059), and many view-state edge cases.

**D10 — Reproducibility.** Broken at the export boundary: no software version, model hash, calibration record, or failure flag travels with the numbers (see §6).

**D11 — Maintainability.** Triplicated `parse_image_name`, duplicated calibration and visualization classes, dead code (`run_server`, unused `dcc.Store`, `trigger_id`), and a god-object with 137 instance attributes.

**D12 — Robustness on real workloads.** Windows DataLoader deadlock (F-022), batch OOM (F-050), no per-image isolation (F-049), Unicode path failures (F-007). See §9 Phase 5.

**D13 — Documentation.** Missing schema/units docs (F-078), docstring/signature mismatch (F-103), mislabeled axes and bands (F-071/F-078).

---

## 5. Scientific Validity

This section audits the quantitative pipeline that turns binary masks and skeletons into the two scientific deliverables of the application: **root length (mm)** and **root area (mm²)**. The conclusion is that *both metrics are derived from a hardcoded, unvalidated calibration*, and that the length metric carries a second, larger, systematic underestimate from counting skeleton pixels as unit-length steps. The area metric is currently numerically correct on the default pipeline but is silently fragile; the length metric is biased even on the default pipeline.

### 5.1 The fixed calibration: 640×480 px ≡ 18×13 mm

Both handlers encode the same physical model as four bare literals (F-002, F-003):

```python
# root_area_inference_handler.py:25-30        # root_length_inference_handler.py:28-35
image_width_px   = 640                         original_width_px  = 640
image_height_px  = 480                         original_height_px = 480
physical_width_mm  = 18                         original_width_mm  = 18
physical_height_mm = 13                         original_height_mm = 13
```

This asserts a sensor field of view of 18 mm × 13 mm sampled onto a 640 × 480 grid, giving anisotropic physical pixel pitches:

- pixel_size_x = 18 / 640 = **0.028125 mm/px**
- pixel_size_y = 13 / 480 = **0.0270833 mm/px**

These pitches differ by `0.028125 / 0.0270833 = 1.0385`, i.e. the pixels are physically **non-square by 3.85%** (the FOV aspect ratio 18:13 = 1.3846 is not equal to the grid aspect ratio 640:480 = 1.3333). That non-squareness is the root cause of the anisotropy problems in §5.3.

**Where the calibration breaks.** The calibration is a *constant*, never compared against the actual array (`mask.shape` / `skeleton.shape` are never read). Three failure modes follow:

1. **Resolution drift (F-002, area).** `calculate_root_area` computes `area_per_pixel = pixel_size_x * pixel_size_y` from the 640×480 constants and then multiplies it by `root_pixels = np.sum(mask == 255)`, the white-pixel count of the *real* mask. The pixel count scales with the true resolution, but the per-pixel area is frozen. A mask delivered at any other resolution yields an area wrong by exactly the ratio of pixel densities. For a mask resized to *k*× linear resolution the reported area is wrong by a factor *k²* (a 1280×960 mask reports **4× the true area**). This is currently latent only because `mask_generation_handler.py:57` forces `transforms.Resize((480, 640))`; but `skeleton_handler.py` ingests *any* `.png` in the masks directory with no shape check, so a manually drawn mask, a different camera, or a future model output size silently corrupts the mm² output with no error. This is a **silent scientific-output corruption**, not a crash — the worst failure class for a measurement tool.

2. **No single source of truth.** The 18/13/640/480 quad is duplicated verbatim across the two handlers. Any recalibration (new lens, new sensor) requires editing two files in lockstep; divergence produces length and area numbers that disagree on the physical scale of the same image.

3. **The FOV itself is an unvalidated assumption.** There is no provenance for 18 mm × 13 mm — no EXIF read, no per-image calibration target, no config. Every reported millimetre depends on this one undocumented pair of constants being exactly right for every image ever processed. If the true FOV is *F_x × F_y*, every length scales linearly and every area scales as *F_x·F_y*; a 5% FOV error becomes a 5% length error and a ~10% area error directly in the published numbers.

**Recommended fix.** Derive pitch from the array and centralize calibration:

```python
PHYS_W_MM, PHYS_H_MM = 18.0, 13.0          # single shared config / dataclass
h, w = mask.shape[:2]
pixel_size_x = PHYS_W_MM / w
pixel_size_y = PHYS_H_MM / h
```

plus an explicit `assert mask.shape[:2] == (480, 640)` (or a logged warning) so an unexpected resolution is surfaced rather than silently rescaled. The FOV constants should live in one module-level config object consumed by both handlers, ideally overridable per-acquisition.

### 5.2 Skeleton pixel count as arc length: the √2 underestimate (PROVEN)

The length handler's length basis is (`root_length_inference_handler.py:25`):

```python
pixel_length = np.sum(skeleton) / 255   # = number of foreground skeleton pixels
```

This treats **every skeleton pixel as one unit of arc length**, independent of how the skeleton threads through the 8-connected grid. This is provably biased low, and the bias is quantifiable.

**Proof.** A 1-px-wide skeleton is a chain of 8-connected pixels. Between two consecutive skeleton pixels the centre-to-centre Euclidean displacement is:

- **1.0 px** for a horizontal/vertical (orthogonal) step, and
- **√2 ≈ 1.41421 px** for a diagonal step.

Counting pixels approximates the true polyline length as the *number of steps* (≈ number of pixels − 1 along a simple path), assigning weight **1** to every step. Diagonal steps therefore each contribute a length deficit of √2 − 1 = **0.41421 px**. The count is exact only for a perfectly axis-aligned path (all orthogonal steps) and maximally wrong for a 45° path (all diagonal). For a path with a fraction *d* of diagonal steps:

```
L_count  = N_orth + N_diag                       (each step weighted 1)
L_true   = N_orth · 1 + N_diag · √2
ratio    = L_count / L_true = 1 / (1 + d·(√2 − 1))
bias     = 1 − L_count/L_true = d·(√2 − 1) / (1 + d·(√2 − 1))
```

**Expected bias magnitude.**

| diagonal fraction *d* | L_count / L_true | underestimate of true length |
|---|---|---|
| 0.0 (axis-aligned) | 1.000 | 0% |
| 0.25 | 0.906 | 9.4% |
| 0.50 (mixed/typical roots) | 0.828 | **17.2%** |
| 0.75 | 0.763 | 23.7% |
| 1.00 (45° line) | 0.707 | 29.3% |

For curvilinear root skeletons whose local orientation is roughly uniform, *d* averages near 0.5, so the expected systematic **underestimate is ~15–20%**, bounded at 29.3% for the worst case. This is the *dominant* error in the length pipeline — far larger than the anisotropy term in §5.3 — and it is present even on the default 640×480 path. It biases every length monotonically in the same direction, so it does **not** cancel in cross-treatment comparisons; it rescales them.

**Corrected formula.** Replace the scalar count with a connectivity-aware polyline integral. Trace the skeleton (or its branch graph) and sum the *physical* displacement of each inter-pixel step, applying the anisotropic pitch per axis (this simultaneously fixes §5.3):

```
L = Σ_steps  sqrt( (Δx · pixel_size_x)²  +  (Δy · pixel_size_y)² )
```

where (Δx, Δy) ∈ {(1,0),(0,1),(1,1),…} are the integer pixel offsets between adjacent skeleton pixels. A standard, well-tested implementation is **`skan.Skeleton(...).path_lengths()`** (or `summarize`), which builds the skeleton graph and integrates branch lengths with exact diagonal weighting and supports per-axis spacing via `spacing=(pixel_size_y, pixel_size_x)`. A cheaper closed-form approximation that removes most of the bias without graph tracing is the **Kulpa / two-weight estimator**:

```
L ≈ 0.948·N_orth + 1.343·N_diag        (Kulpa weights; ~2–4% residual error)
```

Either eliminates the √2 deficit; the graph-integral form is preferred because it also carries the anisotropic pitch correctly and lets length and area share one calibration.

### 5.3 X/Y anisotropy and the scalar-mean collapse (PARTIALLY VALID — secondary)

The length handler collapses both the resampling scale and the physical pitch into scalar means (`lines 40, 47`):

```python
scaling_factor   = (scaling_factor_x + scaling_factor_y) / 2   # 0.5328, 0.5333  → avg
average_pixel_size = (pixel_size_x + pixel_size_y) / 2          # 0.028125, 0.027083 → avg
```

It then maps the 341×256 skeleton count back to the 640×480 frame (`pixel_length / scaling_factor`) and multiplies by `average_pixel_size`. Two observations:

- **Resampling-scale averaging is negligible.** `scaling_factor_x = 341/640 = 0.53281`, `scaling_factor_y = 256/480 = 0.53333`; they differ by 0.097%. Averaging them introduces well under 0.1% error. (This resample path also confirms a separate concern: the skeleton is computed at 341×256, an aspect ratio 1.332 that is *not* the FOV's 18:13 = 1.385 — see F-004 on the unconditional aspect-ratio stretch in `skeleton_inference.py`, which warps geometry before length is ever measured.)
- **Pitch averaging is real but mild.** Using `average_pixel_size = 0.0276042` for a directionless count instead of the correct per-axis pitches biases length by up to the ±1.9% half-spread of the 3.85% pitch anisotropy, in a direction that depends on the root's mean orientation. This is a genuine but **second-order** error: roughly an order of magnitude smaller than the √2 deficit in §5.2.

The correct treatment is *not* to average. Because the underlying pixels are physically non-square, length must be integrated with both pitches inside the Euclidean step (the corrected formula in §5.2 already does this). Averaging is only defensible if the pixels are *intended* to be square — in which case the right fix is to correct the FOV constants so that 18/640 = 13/480, not to paper over the discrepancy with a mean.

### 5.4 /255 thresholding and length-vs-area parity

Two thresholding conventions coexist, and they are **not equivalent**:

- **Area** (`load_mask`, `root_area_inference_handler.py:14-15`): `cv2.threshold(img, 127, 255, THRESH_BINARY)` — a fixed mid-grey cut. The subsequent `np.sum(mask == 255)` counts only pixels *exactly* 255; since `THRESH_BINARY` emits exactly 0 or 255 this is consistent here, but brittle: any path producing intermediate values (anti-aliased/soft masks) would silently drop foreground pixels in 1–254 from the area count. (This mirrors the threshold-mismatch fragility flagged in the overlay path, F-017.)
- **Length** (`preprocess_image`, `root_length_inference_handler.py:16-19`): `THRESH_BINARY + THRESH_OTSU` (adaptive), an auto-inversion `if np.mean(binary) > 127: binary = 255 - binary`, then `skeletonize(binary / 255)` and `(skeleton * 255).astype(uint8)`, finally divided back by 255.

**Parity problem.** Area uses a *fixed* threshold of 127; length uses *Otsu* plus a polarity guess. On the same image these can disagree about which pixels are root, so the two metrics are not guaranteed to measure the same foreground. The Otsu+auto-invert heuristic is also a hidden assumption: on a near-empty image Otsu's split is meaningless and the `np.mean > 127` flip can invert foreground and background, producing a spuriously large or inverted skeleton (F-047). The `/255` round-trips are harmless arithmetic, but they obscure that two different definitions of "root pixel" are in play.

**Recommendation.** Binarize once with a single documented rule, feed the *same* binary mask to both the area count and the skeletonization, and drop the mean-based auto-invert in favour of an explicit, documented foreground polarity.

### 5.5 Net bias budget (default pipeline)

On the current default 640×480 path, stacking the verified effects on **length**:

| source | direction | magnitude |
|---|---|---|
| skeleton pixel count vs arc length (§5.2) | underestimate | **~15–20%** (≤29.3%) |
| pitch averaging anisotropy (§5.3) | orientation-dependent | ≤ ±1.9% |
| resample-scale averaging (§5.3) | either | < 0.1% |
| FOV constant error (§5.1, if any) | linear in FOV | unknown, unbounded |

The length metric is therefore expected to read **biased low by roughly 15–20%** today, dominated entirely by the √2 effect. **Area** has no √2 analogue (a direct pixel count × per-pixel area) and is currently correct *on the default path*, but is exposed to the unbounded *k²* resolution error of §5.1 the moment a non-640×480 mask enters.

### 5.6 Recommended ground-truth validation experiment

1. **Physical calibration target.** Image a precision graticule / stage micrometer (known mm grid) filling the FOV at the production working distance. Read off the *true* FOV (replacing the assumed 18×13) and confirm or refute pixel non-squareness. Pins §5.1. Repeat per camera/lens.
2. **Synthetic phantoms (closed-form ground truth).** Generate masks of analytically known length and area: straight segments at 0°/30°/45°/60°/90°, arcs, sinusoids; filled disks/rectangles. The 45° segment must expose the full 29.3% deficit and the disk area must match πr²·(pixel area) — a unit test that *proves* the √2 fix and the calibration fix numerically.
3. **Resolution-invariance test.** Resize one mask to 0.5×/1×/2×; confirm reported **area is invariant** (currently scales as *k²* — F-002) and **length is invariant** after the §5.2 fix.
4. **Physical root standards.** Image ≥30 real roots / wire mimics of independently measured length and known projected area spanning the size range. Regress pipeline vs. ground truth; report slope (target ~1.0 post-fix), intercept, R², Bland–Altman limits. Pre-fix, the length regression slope should land near 0.80–0.85, confirming the predicted underestimate.
5. **Method cross-check.** Validate corrected length against `skan` branch-length summation or WinRHIZO/RhizoVision on the same skeletons.

**Acceptance criteria:** post-fix length and area regressions with slope within [0.97, 1.03], no resolution dependence, and identical foreground definition feeding both metrics.

**Verified source locations.** `root_area_inference_handler.py:14-16` (threshold), `:22` (count), `:25-30` (calibration), `:33-40` (per-pixel area, no `mask.shape`) — F-002. `root_length_inference_handler.py:16-19` (Otsu+invert+skeletonize), `:25` (√2 underestimate basis), `:28-35` (duplicated constants), `:38-40`/`:45-47` (scalar-mean scaling/pitch) — F-003. `skeleton_inference.py` aspect stretch (F-004); `display_controller.py` soft-mask threshold analogue (F-017).

---

## 6. Broad Availability and Interoperability

### 6.1 Summary

The application's scientific outputs reach the outside world through exactly one channel: two flat CSV files (`root_lengths.csv`, `root_areas.csv`) written with `csv.DictWriter`, plus an in-process Dash web view bound to `localhost`. No finding in this dimension was independently confirmed as a defect-class bug, because the export path does not *crash* — it works. The interoperability problem is not correctness but **impoverishment**: the export schema discards almost everything a downstream consumer needs to trust or reuse the numbers. Measured against FAIR principles, the pipeline is locally accessible but neither interoperable nor reusable, and the most consequential cross-dimension findings (F-002, F-003, F-004, F-006) mean the *values* in those CSVs are silently resolution- and geometry-dependent — making true reproducibility impossible from the export alone.

### 6.2 CSV schema vs. FAIR principles

Both handlers write the same six-column shape:

- Length: `["Image", "Tube", "Position", "Date", "Time", "Length (mm)"]` (`root_length_inference_handler.py:225`)
- Area: `["Image", "Tube", "Position", "Date", "Time", "Area (mm²)"]` (`root_area_inference_handler.py:212`)

- **Findable — Weak.** Hardcoded basenames, no run identifier, no DOI/UUID, no manifest. Two runs overwrite each other.
- **Accessible — Partial.** CSV is universally readable (the one genuine strength), but length and area are split across two unjoined files keyed only on the composite `Image/Tube/Position/Date/Time`. The `Error` field the handlers populate on failure is *deliberately stripped* from the CSV (`row = {header: result.get(header, "") for header in headers}`, line 219/233): a failed measurement is written as `0` with **no error column and no flag** — indistinguishable from a true zero. A silent data-integrity hole in the shared artifact.
- **Interoperable — Poor.** No controlled vocabulary, no ontology mapping (Planteome/MIAPPE), no machine-declared units (the unit is baked into a header string `"Area (mm²)"` embedding a non-ASCII superscript). `Date` is `YYYY.MM.DD` text and `Time` is `HH:MM:SS` text, not ISO-8601, so ingestion requires custom parsing (the app itself re-parses with `format="%Y.%m.%d"` in `data_processor.py:15`).
- **Reusable — Very poor.** **No provenance**: no software version, no model checkpoint name/hash, no calibration record, no acquisition device, no operator, no analysis timestamp.

### 6.3 Units, provenance, and calibration metadata in exports

1. **Calibration is invisible and unverifiable from the export.** The mm/pixel calibration lives only as hardcoded constants in code (`root_area_inference_handler.py:29-33`, duplicated in the length handler). Per F-002 the area calc never consults `mask.shape`; per F-003 the length calc averages anisotropic scale and counts pixels without diagonal weighting; per F-004/F-006 the skeleton raster is stretched/downsampled. The net effect: **the exported `Length (mm)` and `Area (mm²)` are functions of pipeline-internal resolution choices that are not recorded anywhere in the export.** A downstream user who recomputes from the same masks with correct calibration gets different numbers and cannot tell why.
2. **No single source of truth for calibration.** The 18×13 / 640×480 quad is duplicated with no shared config. A collaborator with a different lens/sensor must edit source code, and nothing in the export declares which calibration was used.
3. **Units are documentary, not machine-readable.** Encoding the unit in the header glyph (`Area (mm²)`) means consumers must string-match a Unicode superscript. Combined with the reader using `encoding="latin-1"` (`data_processor.py:13`) while the writer uses the platform default, the `²` glyph is a latent mojibake/round-trip hazard (F-099).

### 6.4 Missing machine-readable formats (JSON / Parquet)

There is no JSON, Parquet, NetCDF, Excel, or sidecar metadata file anywhere. The only serialization is `csv.DictWriter`. The absence of a **typed columnar format** (Parquet) means types are re-inferred on every read (`pd.to_numeric(..., errors="coerce")`, `data_processor.py:18-24`) and coercion failures are silently dropped (`df.dropna(inplace=True)`, line 27) — rows can vanish between export and analysis with no warning. The absence of a **structured format** (JSON) means there is nowhere to attach per-run provenance without changing the flat schema.

### 6.5 The Dash web-visualization sharing model

The "web" framing overstates the sharing model. Both Dash servers bind to loopback only — `make_server("127.0.0.1", self.port, ...)` (`root_length_visulization.py:52`, `root_area_visualization.py:51`), ports 8050/8051, embedded in a `QWebEngineView`. This is **a local rendering surface, not a shareable web deployment**: nothing is reachable off-machine, no shareable URLs, no static HTML bundle. Sharing a *view* is impossible; the user can only screenshot, or rely on the Plotly modebar PNG export and a single (invalid — F-073) `downloadCsv` button (`dash_app.py:381`), plus a `QWebEngineProfile.downloadRequested` handler. The loopback binding is the correct *security* default, but it offers no sharing pathway beyond hand-carrying the two CSVs.

### 6.6 Reproducibility of exported numbers

A recipient of `root_areas.csv`/`root_lengths.csv` **cannot reproduce or independently validate the numbers**. They lack (a) the calibration used, (b) the mask/skeleton resolution actually processed (which, per F-002/F-004/F-006, changes the result), (c) the model checkpoint and software version, (d) any flag distinguishing failed measurements (written as `0`) from real zeros, and (e) acquisition metadata beyond what is regex-parsed from the filename. The reproducibility chain is broken at the export boundary.

### 6.7 Recommendations

1. **Emit a provenance sidecar** (`<results>_metadata.json`) with software version/git commit, model checkpoint name + SHA-256, the calibration block (`fov_width_mm`, `fov_height_mm`, `reference_width_px/height_px`, actual `mask_width_px/height_px`), skeletonization method/params, UTC ISO-8601 analysis timestamp, operator. Highest-value fix; closes the F-002/F-003/F-004/F-006 reproducibility gap at the sharing boundary.
2. **Machine-readable units.** Rename columns to `area_mm2`/`length_mm` (ASCII snake_case), declare units in the sidecar, write CSV explicitly as UTF-8 and read as UTF-8 (drops the `²` round-trip hazard).
3. **Stop coding failures as `0`.** Add a `status`/`error` column preserving the `Error` string the handlers already compute but discard at `:219`/`:233`.
4. **Offer Parquet (and JSON-Lines) alongside CSV** to remove the lossy re-inference / `dropna` row loss.
5. **ISO-8601 timestamps.** Combine `Date`+`Time` into one `acquired_at`; keep raw filename for traceability.
6. **Centralize calibration** as one config object consumed by both handlers and *serialized into the sidecar*.
7. **Provide a real share artifact** from the Dash view (`fig.write_html(..., include_plotlyjs="cdn")` or PDF). Keep the server on `127.0.0.1`.
8. **Adopt a phenotyping metadata standard** (MIAPPE + Planteome trait mapping).

**Relevant files.** `root_area_inference_handler.py` (schema + `save_to_csv` 210-220; calibration 29-40; error stripped 219); `root_length_inference_handler.py` (schema + `save_to_csv` 223-233; calibration 34-53; parsing 57-105); `data_processor.py` (latin-1 read, coercion + `dropna` 13-27); `root_length_visulization.py`/`root_area_visualization.py` (loopback 52/51; download handlers 254/251); `dash_app.py` (only `downloadCsv` button, 381).

---

## 7. Architecture & Concurrency

The application is a PyQt6 desktop front-end orchestrating three subsystems — an interactive editor, a batch ML/measurement pipeline, and an embedded Dash/Plotly web visualization — without clear seams between them. The boundaries that *should* separate UI, domain logic, threading, and rendering are collapsed into a handful of oversized modules. The two structural problems that dominate are a 1384-line god-object in the editor and a pervasive copy-paste of the scientific calibration/measurement logic. Layered on top are concrete concurrency hazards: synchronous CPU/IO on the Qt main thread, a serialized single-request Dash server thread, and a fragile QThread↔QTimer↔Dash handoff.

### 7.1 The 1384-LOC god-object: `SkeletonCorrectionInterface`

`app/gui/skeleton_correction_interface.py` defines a single `QWidget` subclass (line 48) spanning the entire 1384-line file. Its `__init__` establishes **137 instance attributes** — the clearest signal of a class with no coherent single responsibility. It is simultaneously a controller, tool state machine, geometry library, renderer, persistence layer, and path resolver. A method census:

- **UI construction**: `_build_ui` (131), `_create_control_panel` (157, ~220 lines inline rather than composing from `ui_panels.py`).
- **Tool/mode state machine**: `_on_tool_changed` (437), `_on_mode_toggle` (470), `_update_polyline_buttons_enabled` (483), `_update_status_label` (494).
- **Raw input event handling**: `on_tool_mouse_press` (790, ~90 lines), `_move` (881), `_release` (919), `_double_click` (955), `on_key_press` (962).
- **Computational geometry** (no business in a widget): `_nearest_endpoint` (693), `_nearest_skeleton_pixel` (1029), `_nearest_polyline_handle_index` (1101), `_nearest_polyline_segment_index` (1119), `_build_polyline_path` (1161), `_nearest_topology_polyline` (1245), `_sample_smooth_polyline_points` (1295) — pure algorithms, untestable in isolation because bound to `self`.
- **Rendering/scene management**: `_update_skeleton_display` (589) + throttled twin (625), endpoint item lifecycle.
- **Persistence and path policy**: `_default_skeletons_dir` (546), `_default_correction_path` (556), `save_skeleton` (771), `load_image` (408).

The QImage-wraps-numpy use-after-free (F-018) and the INTER_NEAREST save-resize corruption (F-006) both live here precisely because rendering and persistence were never separated from editing.

### 7.2 Length/area duplication: no single source of truth

The scientific core is physically duplicated across the two inference handlers and partially triplicated into the editor's graph model. The calibration constants are copied verbatim (`root_length_inference_handler.py:31-35`, `root_area_inference_handler.py:25-30`). This is the root cause of **F-002**: per-pixel size derived from hardcoded constants rather than `mask.shape`. A fix applied to one handler will not propagate to the other — the two scientific outputs can drift apart.

`parse_image_name` exists in **three** independent copies: `root_area_inference_handler.py:44`, `root_length_inference_handler.py:57`, `image_manager.py:150`. A filename-convention change requires editing three locations in two layers with no test to catch a miss. Skeletonization logic is likewise spread across `skeleton_inference.py`, `root_length_inference_handler.py:preprocess_image`, and `skeleton_graph_model.py:_skeletonize_uint8`, with differing thresholds (Otsu vs fixed) — the editor and batch pipeline can produce *different skeletons from the same mask*. The natural home — a `Calibration` dataclass, one `parse_image_name`, one `skeletonize` — does not exist.

### 7.3 GUI-thread blocking

The codebase correctly threads the *long* operations (`MaskGenerationThread`, `SkeletonGeneratorThread`, `RootLengthCalculatorThread`, `RootAreaCalculatorThread`). The problem is what was *not* moved off-thread. `save_skeleton` (`skeleton_correction_interface.py:771`) runs on the GUI thread and synchronously calls `render_to_size` (line 783) → a full `cv2.resize(..., INTER_NEAREST)` + `_skeletonize_uint8` + `cv2.imwrite` (line 786) before the event loop regains control — the same path implicated in F-006. Image *loading* (`load_image` → `cv2.imread`/`QPixmap`) is likewise synchronous, despite `image_manager.py`'s `ImageLoaderThread` existing precisely to avoid this.

### 7.4 Dash-thread vs Qt-loop coupling

**(a) The Dash server thread is serialized.** `DashServerThread.run` (`root_length_visulization.py:50-56`) drives a single-request loop:

```python
self.server.timeout = 0.5
while not self._stop_event.is_set():
    self.server.handle_request()
```

`handle_request()` processes exactly **one** request per iteration. Despite the dead `run_server` passing `threaded=True` (F-008), the actual path is strictly serial. The hover callback (F-013) re-reads/decodes/resizes/base64-encodes up to 10 images per hover; on a lab share this turns hover latency into seconds. Serialization + no thumbnail cache are an architectural pair.

**(b) The spawn handshake is timing-fragile.** `visualization_manager.py` sequences startup with `QTimer.singleShot(500/100, ...)` fixed delays (lines 244, 418, 101, 138, 215-216) substituting for readiness checks. The correct signals (`port_assigned`, `error`) exist but are bypassed, so a slow machine races the 500 ms window and a fast teardown/re-open collides with a still-shutting-down socket. Combined with F-009 (empty-tube `IndexError`) and F-014 (callback swallowing into `no_update`), a startup race produces a silently blank panel.

### 7.5 Performance and memory hotspots

- **O(n) undo stack** (F-019): `list.pop(0)` for a 25-deep bound; use `collections.deque(maxlen=25)`.
- **Per-hover image re-encode** (F-013): no caching on `get_encoded_image`; precompute/memoize thumbnails.
- **`lru_cache` on an instance method** (F-012): keys on `self`, pins the `DataCache` instance for the process lifetime and returns stale data after the DataFrame mutates.
- **Nested concurrency**: each batch handler spawns a `ThreadPoolExecutor(max_workers=min(cpu*2, 32, n))` *inside* a `QThread`; for CPU-bound `skeletonize`/`np.sum` this is GIL-throttled. A `ProcessPoolExecutor` is the throughput-correct choice for the measurement stage.
- **Use-after-free** (F-018): `QImage(binary_mask.data, ...)` aliasing an unretained numpy buffer.

### 7.6 Target decomposition

1. **`domain/calibration.py`** — one `Calibration` dataclass with `pixel_size(shape)` deriving mm/px from the *actual* `mask.shape`. Kills F-002's root cause.
2. **`domain/measurement.py`** — one `parse_image_name`, one `skeletonize`, one diagonal-aware `skeleton_length` (F-003), one `root_area`. Both handlers and the editor import from here.
3. **Split `SkeletonCorrectionInterface`** into `SkeletonEditorWidget` (UI), `EditorController` (state machine/events), `SkeletonGeometry` (pure functions — Qt-free, unit-testable), `SkeletonRenderer` (scene/QImage lifecycle, owning buffer lifetime → closes F-018), `SkeletonStore` (load/save, render+write on a worker).
4. **`services/`** — a `UnicodeImageIO` wrapper (`cv2.imdecode(np.fromfile(...))` / `cv2.imencode(...).tofile(...)`) everywhere (F-007), and a single `resource_path` anchor (F-005).

### 7.7 High-throughput desktop design

- **Worker pattern, uniformly applied** via a single `JobRunner` (`QThreadPool` + `QRunnable`); `save_skeleton` snapshots on the GUI thread and hands render+write to a worker.
- **Process pool for the measurement stage** so the GIL stops serializing compute; keep a thread pool only for IO `imread` prefetch.
- **A real Dash server** (`make_server(..., threaded=True)` + `serve_forever`, or waitress) with a startup thumbnail cache; delete dead `run_server` (F-008).
- **Event-driven lifecycle** off `port_assigned`/`error` + an HTTP readiness probe instead of `QTimer.singleShot(500, ...)` guesses.
- **Immutable precomputed structures** behind the visualization; drop the leaking instance-method `lru_cache` (F-012).

**Relevant files.** `skeleton_correction_interface.py` (god-object), `skeleton_graph_model.py` (render_to_size + O(n) undo), `root_length_inference_handler.py`/`root_area_inference_handler.py` (calibration/parse duplication), `image_manager.py` (third parse copy), `root_length_visulization.py`/`root_area_visualization.py` (serialized server, QTimer coupling), `visualization_manager.py` (timer handshake), `dash_image_utils.py`/`dash_data_cache.py` (per-hover encode, leaking cache).

---

## 8. Security & Supply-Chain

> **Dimension verdict:** No findings here were designated verified *primary* security findings, but the codebase exhibits a cluster of concrete, reproducible security and supply-chain defects substantiated by direct code evidence. Each is tied to a specific file and line.

### SS-01 — `torch.load` without `weights_only=True` on both model loaders (HIGH)

**Files:** `skeleton_inference.py:459`, `mask_generation_handler.py:147`

```python
state_dict = torch.load(load_path, map_location=str(self.device))   # skeleton_inference.py:459
torch.load(weights_path, map_location=self.device)                  # mask_generation_handler.py:147
```

PyTorch's pickle-based deserializer executes arbitrary Python during unpickling. Without `weights_only=True`, loading a tampered `.pth` is equivalent to code execution as the running user. Not theoretical: the weights ship as `checkpoints.zip` committed to the repo (SS-03); replacing that archive yields RCE on every workstation. **Fix:** pass `weights_only=True` to both calls; enumerate any non-tensor objects with `torch.serialization.add_safe_globals`.

### SS-02 — Dead `run_server` methods bind to all interfaces with no auth (MEDIUM)

**Files:** `root_length_visulization.py:52`, `dash_app.py:994`, `dash_app_area.py:500`

The live path binds to `127.0.0.1` correctly. But the two dead `run_server` methods call `self.app.run_server(debug=False, port=8050, threaded=True)` with **no `host`** — Flask/Dash can default to `0.0.0.0`, exposing the unauthenticated Dash app on all interfaces. Currently dead code (F-008) but one refactor from active. Neither Dash app has authentication regardless of bind address. **Fix:** add explicit `host="127.0.0.1"`; delete the dead methods; document loopback-only + no-auth.

### SS-03 — 45.7 MB binary blob committed to git history (HIGH)

```
commit 2075568  reqs  (Oct 18 2024)
  checkpoints.zip  | Bin 0 -> 47935331 bytes
```

`checkpoints.zip` is permanently in history; `.gitignore` now lists it but the blob is already in the pack. Consequences: repository bloat (every clone transfers 46 MB), **no integrity verification** (no hash/signature; combined with SS-01 → RCE on replacement), **no provenance** (no training run/dataset record). **Fix:** `git filter-repo --path checkpoints.zip --invert-paths`; host weights on a release asset / object store / model registry; verify SHA-256 at startup before `torch.load`; document provenance.

### SS-04 — Full-freeze pinning with no hash verification; torch unpinned (MEDIUM)

**File:** `requirements.txt` (114 `==`-pinned entries)

Version-only pinning does not stop a compromised re-upload or dependency-confusion attack; there is no `--hash` lock file and no `pip install --require-hashes` in `scripts/build.ps1`. Worse, `torch`/`torchvision` — the highest-risk deps given SS-01 — are **absent from the file entirely**, so `pip install -r requirements.txt` pulls whatever PyPI currently serves. **Fix:** pin torch/torchvision; generate a `pip-compile --generate-hashes` lock; install with `--require-hashes` in the build script.

### SS-05 — No CI, no lint, no static analysis (HIGH — process)

**Evidence:** no `.github/`, no `pyproject.toml`/`setup.cfg`/`.flake8`/`.pylintrc`/`mypy.ini`. The only automation is `scripts/build.ps1` (PyInstaller). Many findings here (F-002…F-019) would be caught by `mypy`/a linter on first commit; the `dict[...](...)` call (F-010) is a direct result of no linter; `bandit` would flag `torch.load`-without-`weights_only` and bare `except` with zero config; the committed blob (SS-03) would be caught by `detect-secrets`/a file-size hook. **Minimum remediation:** a GitHub Actions workflow running `flake8`, `mypy --ignore-missing-imports`, `bandit -r app/` on every PR; a `pre-commit` config with `detect-secrets` + blob-size guard.

### SS-06 — Pervasive bare `except` swallows security-relevant failures (MEDIUM)

**Scope:** 65+ `except Exception` clauses. Representative:

| File | Line | Silenced |
|---|---|---|
| `mask_generation_handler.py` | 153 | Failed model load (e.g. tampered weights raising on `load_state_dict`) |
| `generate_skeleton_handler.py` | 39 | Any torch deserialization error incl. pickle exploits |
| `data_processor.py` | 39 | CSV parse failure → empty frame |
| `dash_app.py` | 995 | Server startup failure |
| `root_area_inference_handler.py` | 85 | Silent mask-processing failure, no output |

A crafted `.pth` raising mid-unpickling after partial side effects is silenced with no audit trail. **Fix:** replace bare `pass` with `logger.exception("context")` at minimum; propagate model-load failure to the UI rather than silently setting `self.model = None`.

### SS-07 — No path-boundary validation on user-supplied dataset paths (LOW–MEDIUM)

**Files:** `skeleton_correction_interface.py:576`, `root_area_inference_handler.py:14`, `root_length_inference_handler.py:15`

`cv2.imread(path, ...)` is called on paths from a `QTreeWidget` populated by `os.walk` with no check that the resolved path stays within the selected root. A symlink to outside the dataset is silently followed. Immediate impact is a silent processing error, but the pattern shows no trust boundary between dataset folder and filesystem. **Fix:** `os.path.realpath` every loaded path and assert it is under the selected root.

### Summary Table

| ID | Severity | Title |
|---|---|---|
| SS-01 | High | `torch.load` without `weights_only=True` — pickle RCE vector |
| SS-02 | Medium | `run_server` dead-code binds to all interfaces, no auth |
| SS-03 | High | 45.7 MB weights committed to git; no hash verification |
| SS-04 | Medium | torch/torchvision unpinned; no hash-verified lock |
| SS-05 | High | No CI/lint/static-analysis — process gap enabling all other findings |
| SS-06 | Medium | 65+ bare `except` silence security-relevant failures |
| SS-07 | Low–Medium | No path-boundary validation on dataset paths |

---

## 9. Prioritized Remediation Roadmap

Effort key: **S** ≈ <1h, **M** ≈ 1–4h, **L** ≈ 4–8h, **XL** ≈ >1 day. Phases ordered; within a phase, ordered by dependency.

### Phase 1 — Quick Wins (low-risk, isolated, high signal-to-effort)

| Order | Finding(s) | Title | Effort | Notes |
|---|---|---|---|---|
| 1.1 | F-010 | Replace `dict[...](...)` with `dict(...)` in `dash_visualizations.py:837` | S | Removes latent <3.9 crash + linter noise |
| 1.2 | F-019 | `list.pop(0)` undo/redo → `collections.deque(maxlen=...)` | S | Isolated perf |
| 1.3 | F-073, F-074 | Remove invalid `downloadCsv` button; add reversed-range feedback/auto-swap | S | UI correctness |
| 1.4 | F-076, F-081 | Remove `traceback.print_exc()`/inline import; delete catch-and-bare-raise | S | Cleanup |
| 1.5 | F-075 | Remove duplicate `y_max`; consolidate y-axis config | S | — |
| 1.6 | F-008 | Re-raise (after log) or delete dead `run_server` | S | Prefer deletion |
| 1.7 | F-048 | Case-insensitive extension check in `is_image_file` | S | Unblocks mixed-case images |
| 1.8 | F-067 | Remove double `save_for_undo()` in flood_fill | S | — |
| 1.9 | F-041 | Guard divide-by-zero (`upper==lower`) in contrast stretch | S | — |
| 1.10 | F-054 | Clamp zoom *after* multiplication | S | — |
| 1.11 | F-078 | Document schema/units; fix mislabeled "Vertical Depth (cm)" | S | Docs |

### Phase 2 — Correctness & Performance

| Order | Finding(s) | Title | Effort | Notes |
|---|---|---|---|---|
| 2.1 | F-009 | Guard empty `get_unique_tubes()` before `[0]` | S | Prevents IndexError on view switch |
| 2.2 | F-037 | Move try/except inside parse loop; handle ranged tokens | S | — |
| 2.3 | F-058 | Build `mask_path` from `image_path` (with extension) | S | Fixes missing green highlight |
| 2.4 | F-068 | Use `image_manager.original_folder` for mask input dir | S | After 2.x |
| 2.5 | F-045 | Configurable skeleton suffix / fallback beyond `_fake.png` | S | — |
| 2.6 | F-028 | Slice floodFill `local_mask[1:h+1,1:w+1]` before indexing | S | Real IndexError |
| 2.7 | F-027 | Reconstruct flood-fill result as `Format_ARGB32` | M | After 2.6 |
| 2.8 | F-015, F-016 | Fix toggle signal double-connect; stop skeleton-toggle wiping saved-mask colors | M | F-016 after F-015 |
| 2.9 | F-017 | Reconcile Otsu (0/255) vs color-table threshold-50 mismatch | S | — |
| 2.10 | F-051 | Check `cv2.imwrite` return; emit success only on real write | S | Pairs with F-007 |
| 2.11 | F-059, F-060 | Skeletonize inside `topology()` before vectorize; make lists tuples | M | Together |
| 2.12 | F-061 | Convert recursive RDP to iterative stack | M | After 2.11 |
| 2.13 | F-053 | Erase original polyline exact-pixel, thickness 1–2, protect junctions | M | After 2.11/2.12 |
| 2.14 | F-052 | Cache topology; invalidate on mask change; debounce endpoint recompute | M | After 2.11 |
| 2.15 | F-013 | `lru_cache`/precompute base64 thumbnails for hover | S | Hot path |
| 2.16 | F-056 | Reuse single `QGraphicsScene` (`scene.clear()`) | M | UI responsiveness |
| 2.17 | F-033, F-039 | Build (tube,date) availability from set/cache; populate or remove dead `dcc.Store` | M | Precursor to F-012/F-079 |
| 2.18 | F-077 | Replace O(dates×tubes) `add_shape` with per-axis bgcolor | S | — |
| 2.19 | F-082 | Cap `selected-tubes-store` or store selection-mode + exclusions | S | Info-level perf |
| 2.20 | F-036 | Derive interval from `interval_size`; robust date parse | S | After 2.17 |

### Phase 3 — Structural / Dedup + OOP

| Order | Finding(s) | Title | Effort | Notes |
|---|---|---|---|---|
| 3.1 | F-005 | Anchor `CHECKPOINTS_DIR` via `get_resource_path()`/`__file__` | S | Foundational for packaged builds |
| 3.2 | F-057 | Centralize image-name/key extraction into one utility | M | After 2.3/2.4 |
| 3.3 | F-031 | Remove duplicate `MaskTracingInterface` in `MaskHandler` | M | After F-015/F-016 |
| 3.4 | F-029 | Instance-scope `_item_cache` (or `WeakValueDictionary`); validate before return | M | After 3.2 |
| 3.5 | F-012, F-079 | Replace instance-method `lru_cache` with per-instance/precomputed cache; route all df access via DataCache; return copies | L | After 2.17 |
| 3.6 | F-024, F-025 | Guard `isRunning()` before re-creating workers; connect `finished→deleteLater` | M | Shared lifecycle |
| 3.7 | F-030, F-064 | Cooperative cancel + disconnect for loader; lock/event-guard `self.server` | M | After 3.6 |
| 3.8 | F-018 | Keep numpy buffer alive for `QImage` | S | Independent |
| 3.9 | F-020 | `_visualization_shown` idempotency guard; stop poll timer on port-assigned | S | Precursor to 3.10 |
| 3.10 | F-083, F-084, F-085 | Extract `BaseVisualization(QMainWindow)`; fold F-084 escalation + drop TOCTOU F-085 into base | L | **After 3.9, 3.7, 1.6** |

### Phase 4 — Scientific Re-validation (needs domain review)

> Changes reported measurements. Coordinated workstream with a domain owner. **After Phase 3.1.**

| Order | Finding(s) | Title | Effort | Notes |
|---|---|---|---|---|
| 4.1 | F-069, F-002 | Extract calibration (FOV mm, resolution, mm/px) into one shared config; validate against `mask.shape`/`img.shape` | L | **Single source of truth — blocks all Phase-4 items** |
| 4.2 | F-004 | Save model output at native size / correct aspect; fix swapped `(w,h)` in PIL `resize` | M | After 4.1 |
| 4.3 | F-006 | Save corrected skeleton at editor resolution (or dilate→resize→re-skeletonize) | M | After 4.2 |
| 4.4 | F-003 | Diagonal-aware skeleton length (graph traversal, per-axis pixel size) | L | After 4.1, 4.2 |
| 4.5 | F-070 | Overlay resize INTER_LINEAR+re-threshold or dilate-before-resize | S | After 4.3 |
| 4.6 | F-047 | Explicit foreground polarity; skip Otsu+mean-inversion for binary inputs | M | After 4.1/4.2 |
| 4.7 | F-043 | Correct position-attention aggregation to `A@V` (DANet form) | M | **Requires model retraining/eval** |
| 4.8 | F-038, F-034 | Sum (not mean) for area profile; single-linkage date clustering w/ documented threshold | M | After 4.1 |
| 4.9 | F-026 | Write NaN/empty + Error/Status column for failed records | M | **After F-046 (Phase 5)** |
| 4.10 | F-071, F-072, F-080 | Relabel "Field Variance" as inter-tube SD; fix depth-color normalization; pre-extract HTML hover values | S | Last in phase |

> **Phase-4 exit gate:** re-run a reference dataset, diff length/area vs pre-change, domain-owner sign-off. Capture as regression fixtures for Phase 6.

### Phase 5 — Memory & High-Throughput Hardening

| Order | Finding(s) | Title | Effort | Notes |
|---|---|---|---|---|
| 5.1 | F-022 | Force `num_workers=0` on Windows (or build DataLoader on main thread) | M | **High impact — prevents QThread/spawn deadlock** |
| 5.2 | F-050 | Reduce `BATCH_SIZE` (VRAM-aware); overlap PNG encode/CPU transfer | M | After 5.1 |
| 5.3 | F-023 | Add `weights_only=True` to both `torch.load` calls | S | Pairs with F-005 |
| 5.4 | F-007 | Unicode-safe cv2 I/O everywhere | M | Precondition for F-046 |
| 5.5 | F-046 | Guard `cv2.imread` None → explicit error | S | **After 5.4; blocks F-026** |
| 5.6 | F-021 | Log + accumulate per-batch failures; surface in finished report | M | After 5.1/5.2 |
| 5.7 | F-049 | Per-image try/except in dataset load+save; skip and report | M | After 5.6 |
| 5.8 | F-014, F-032, F-040, F-066 | Surface viz/data-load errors (figure annotation + `logging.exception`) instead of silent `no_update`/empty df; restore field-map | M | After data path stable |

### Phase 6 — Deps-to-Latest + CI/Tests

| Order | Item | Effort | Notes |
|---|---|---|---|
| 6.1 | CI (GitHub Actions, Windows runner) + smoke test: import app, load checkpoints (F-005), run 1-image inference | M | After F-005, F-022, F-023 land |
| 6.2 | Unit tests for pure logic in Phases 1–3 (F-037 parse, F-058/F-068 path, F-019 deque, F-054 zoom, F-061 RDP, F-009/F-074 guards) | L | After Phases 1–3 |
| 6.3 | **Scientific regression fixtures** from Phase-4 gate (length/area/profile golden values) — F-002/F-003/F-004/F-006/F-038 | M | **After Phase 4 sign-off** |
| 6.4 | Dependency upgrades to latest stable (`opencv-python` for F-007, torch, dash, numpy/pandas/scikit-image, PyQt6) with CI gating each bump | XL | **Last** — CI+tests must exist first; verify `torch.load`/Dash callback API stability |

### Cross-Phase Dependency Highlights

- **Calibration config (F-069/F-002)** is the Phase-4 keystone: F-002, F-003, F-038, F-034 all consume it.
- **F-005 (checkpoint path)** lands early (3.1) because Phase-4 reruns and Phase-6 CI depend on reliable model loading.
- **F-022 (Windows DataLoader)** gates realistic batch testing → before F-050/F-021/F-049 and CI smoke tests.
- **F-007 (Unicode I/O) → F-046 (None handling) → F-026 (failed-record encoding).**
- **F-008/F-020/F-084/F-085** corrections precede the **F-083 base-class extraction** (3.10).
- Phase-4 reference outputs feed Phase-6.3 fixtures; 6.4 dep upgrades run last, gated by the full suite.

---

## 10. File-by-File Risk Appendix

Risk = aggregate severity + count of findings anchored to the file.

| File | Risk | Findings | Notes |
|---|---|---|---|
| `app/inference/root_area_inference_handler.py` | **CRITICAL** | F-002 | The lone critical: silent area corruption off the default resolution. Calibration duplication + error-stripping on CSV. |
| `app/inference/root_length_inference_handler.py` | **HIGH** | F-003, F-026, F-046, F-047, F-069, F-097, F-098, F-099 | Dominant √2 length bias, duplicated calibration, failures coded as 0, encoding. Single highest-density scientific-risk file. |
| `app/gui/skeleton_correction_interface.py` | **HIGH** | F-006, F-007, F-051, F-052, F-053, F-105, F-106 | 1384-line god-object; save-resize corruption, Unicode IO, use-after-free origin. |
| `app/inference/skeleton_inference.py` | **HIGH** | F-004, F-005, F-048, F-049, F-101, F-103, SS-01 | Geometry distortion, relative checkpoint path, batch fragility, pickle RCE. |
| `app/handlers/mask_generation_handler.py` | **HIGH** | F-021, F-022, F-023, F-068, F-102, SS-01, SS-06 | Windows DataLoader deadlock, pickle RCE, silent batch failures. |
| `app/visualization/dash_app.py` | **HIGH** | F-008, F-009, F-032, F-036, F-073, F-074, F-081, F-082, SS-02 | Dead bind-all server, IndexError, swallowed errors. |
| `app/visualization/dash_visualizations.py` | **HIGH** | F-010, F-011, F-034, F-037, F-071, F-072, F-075, F-076, F-077, F-078, F-079, F-080 | Highest finding count; alias-call crash, scientific aggregation, injection surface. |
| `app/gui/main_window.py` | **HIGH** | F-015, F-016, F-031, F-057, F-058, F-089, F-090, F-091 | Signal double-connect, color wipe, path-derivation bugs. |
| `app/gui/display_controller.py` | **HIGH** | F-017, F-018, F-054, F-056, F-070, F-092, F-093 | Use-after-free, threshold mismatch, scene churn, tight coupling. |
| `app/gui/skeleton_graph_model.py` | **HIGH** | F-019, F-059, F-060, F-061, F-095, F-096 | O(n) undo, topology-on-thick-mask, recursive RDP, false immutability. |
| `app/gui/mask_drawing_tools.py` | **MEDIUM** | F-027, F-028, F-067 | Flood-fill channel swap + indexing + double undo. |
| `app/visualization/dash_app_area.py` | **MEDIUM** | F-014, F-038, F-039, F-086, F-087, F-088, SS-02 | Silent error masking, mean-vs-sum, dup style dicts. |
| `app/visualization/root_length_visulization.py` / `root_area_visualization.py` | **MEDIUM** | F-020, F-064, F-066, F-083, F-084, F-085 | Serialized server, data race, near-identical copies. |
| `app/handlers/skeleton_handler.py` | **MEDIUM** | F-024, F-045 | Thread leak, `_fake.png`-only matching. |
| `app/handlers/generate_skeleton_handler.py` | **MEDIUM** | F-025, F-050 | GC-prone thread, batch OOM. |
| `app/visualization/dash_data_cache.py` | **HIGH** | F-012 | Instance-method `lru_cache` leak + stale data. |
| `app/visualization/dash_image_utils.py` | **HIGH** | F-013, F-042 | Per-hover re-encode, mixed-key map. |
| `app/data_processing/data_processor.py` | **MEDIUM** | F-040, latin-1/dropna (§6.4) | Silent empty-frame + row loss. |
| `app/gui/image_manager.py` | **MEDIUM** | F-030, third `parse_image_name` (§7.2) | Loader terminate, duplicated parsing. |
| `app/gui/file_tree_manager.py` | **MEDIUM** | F-029 | Module-level cache → stale items. |
| `app/gui/image_normalization_interface.py` | **MEDIUM** | F-041 | Divide-by-zero. |
| `app/mask_model/model.py` | **MEDIUM** | F-043 | Transposed position-attention (needs retrain to fix). |
| `app/gui/mask_tracing_interface.py` | **LOW** | F-104 | Brush-cap inconsistency. |
| `requirements.txt` / build / repo | **HIGH (process)** | SS-03, SS-04, SS-05 | Committed weights blob, unpinned torch, no CI/lint. |

---

## 11. Refuted / False-Positive Findings (9)

During adversarial verification, 9 of the 106 candidate findings were withdrawn after re-reading the cited code and confirming the issue was either unreachable, already guarded, or based on a misreading. They are recorded here for auditability (false-positive rate 9/106 ≈ 8.5%):

1. **F-001** — *Calibration "wrong constant" mis-identification.* The original draft flagged a specific arithmetic error in the calibration value itself; verification showed the constants are internally consistent and the real defect is the *fixed-resolution assumption* (re-issued as F-002) — the original numeric claim was withdrawn.
2. **F-035** — Claimed an off-by-one in date-interval bucketing; the cited boundary was inclusive by design and matched the documented interval semantics. No defect.
3. **F-044** — Alleged a channel-order bug in mask model preprocessing; the transform already matched the training normalization. Misread.
4. **F-055** — Claimed pan offset was not clamped; a guard exists one method up the call chain. Already handled.
5. **F-062** — Suspected a memory leak in endpoint `QGraphicsItem` creation; `_clear_endpoint_items` does remove them before recreation. Not reachable.
6. **F-063** — Alleged a race in the undo stack between paint and save; both run on the GUI thread, so no concurrency. Misattributed.
7. **F-065** — Claimed the Dash port was hardcoded and would collide; the live path assigns a free port and emits `port_assigned`. Refuted.
8. **F-094** — Suspected a divide-by-zero in depth normalization; a single-interval fallback (the 0.5 in F-072) already prevents it. The remaining issue is cosmetic (re-scoped into F-072), not a crash.
9. **F-100** — Alleged CSV rows could be written out of order corrupting downstream joins; ordering is non-load-bearing because consumers join on the composite key. Not a defect (the *sort-key* edge case is the separate, valid F-098).

---

*End of report.*