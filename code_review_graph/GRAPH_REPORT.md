# Graph Report - code_review_findings.json  (2026-05-29)

## Corpus Check
- 97 files · ~50,000 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 135 nodes · 354 edges · 38 communities detected
- Extraction: 27% EXTRACTED · 73% INFERRED · 0% AMBIGUOUS · INFERRED: 257 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_dash_visualizations.py|dash_visualizations.py]]
- [[_COMMUNITY_main_window.py|main_window.py]]
- [[_COMMUNITY_dash_app.py|dash_app.py]]
- [[_COMMUNITY_skeleton_correction_interface.py|skeleton_correction_interface.py]]
- [[_COMMUNITY_display_controller.py|display_controller.py]]
- [[_COMMUNITY_dash_app_area.py|dash_app_area.py]]
- [[_COMMUNITY_skeleton_graph_model.py|skeleton_graph_model.py]]
- [[_COMMUNITY_root_length_visulization.py|root_length_visulization.py]]
- [[_COMMUNITY_root_length_inference_handler.py|root_length_inference_handler.py]]
- [[_COMMUNITY_mask_generation_handler.py|mask_generation_handler.py]]
- [[_COMMUNITY_skeleton_inference.py|skeleton_inference.py]]
- [[_COMMUNITY_root_length_inference_handler.py|root_length_inference_handler.py]]
- [[_COMMUNITY_mask_drawing_tools.py|mask_drawing_tools.py]]
- [[_COMMUNITY_skeleton_inference.py|skeleton_inference.py]]
- [[_COMMUNITY_dash_image_utils.py|dash_image_utils.py]]
- [[_COMMUNITY_skeleton_handler.py|skeleton_handler.py]]
- [[_COMMUNITY_generate_skeleton_handler.py|generate_skeleton_handler.py]]
- [[_COMMUNITY_root_area_inference_handler.py|root_area_inference_handler.py]]
- [[_COMMUNITY_dash_data_cache.py|dash_data_cache.py]]
- [[_COMMUNITY_file_tree_manager.py|file_tree_manager.py]]
- [[_COMMUNITY_image_manager.py|image_manager.py]]
- [[_COMMUNITY_data_processor.py|data_processor.py]]
- [[_COMMUNITY_image_normalization_interface.py|image_normalization_interface.py]]
- [[_COMMUNITY_model.py|model.py]]
- [[_COMMUNITY_mask_tracing_interface.py|mask_tracing_interface.py]]
- [[_COMMUNITY_Scientific validity|Scientific validity]]
- [[_COMMUNITY_ML inference|ML inference]]
- [[_COMMUNITY_Correctness|Correctness]]
- [[_COMMUNITY_Performance|Performance]]
- [[_COMMUNITY_Memoryresources|Memory/resources]]
- [[_COMMUNITY_Error handling|Error handling]]
- [[_COMMUNITY_Security|Security]]
- [[_COMMUNITY_Architecture|Architecture]]
- [[_COMMUNITY_Concurrency|Concurrency]]
- [[_COMMUNITY_Interoperability|Interoperability]]
- [[_COMMUNITY_Reproducibility|Reproducibility]]
- [[_COMMUNITY_Docs|Docs]]
- [[_COMMUNITY_Repo hygiene|Repo hygiene]]

## God Nodes (most connected - your core abstractions)
1. `[high] dict[] generic-alias subscript used as a constructor call in fig.updat` - 13 edges
2. `[high] Legend dedup flag set unconditionally, silencing legends when first tu` - 13 edges
3. `[medium] get_tube_date_availability does O(tubes*dates) DataFrame filters per c` - 13 edges
4. `[medium] Greedy ±3-day date-merging is order-dependent and sums (not averages) ` - 13 edges
5. `[medium] parse_tube_selection drops all valid results on first parse error` - 13 edges
6. `[low] Field 'variance' band is inter-tube SD of total length, mislabeled as ` - 13 edges
7. `[low] Depth color normalization uses interval_ends (not filtered y_positions` - 13 edges
8. `[low] Duplicate y_max assignment and two conflicting y-axis update paths` - 13 edges
9. `[low] traceback.print_exc() in except block leaks stack traces to stdout in ` - 13 edges
10. `[low] O(dates*tubes) fig.add_shape background rects slow renders for large g` - 13 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities

### Community 0 - "dash_visualizations.py"
Cohesion: 1.0
Nodes (13): [high] dict[] generic-alias subscript used as a constructor call in fig.updat, [high] Legend dedup flag set unconditionally, silencing legends when first tu, [medium] get_tube_date_availability does O(tubes*dates) DataFrame filters per c, [medium] Greedy ±3-day date-merging is order-dependent and sums (not averages) , [medium] parse_tube_selection drops all valid results on first parse error, [low] Field 'variance' band is inter-tube SD of total length, mislabeled as , [low] Depth color normalization uses interval_ends (not filtered y_positions, [low] Duplicate y_max assignment and two conflicting y-axis update paths (+5 more)

### Community 1 - "main_window.py"
Cohesion: 1.0
Nodes (8): [high] Signal double-connection on repeated toggle_mask_tracing calls, [high] toggle_skeleton_correction calls toggle_mask_tracing, wiping legitimat, [medium] Duplicate MaskTracingInterface created in both MainWindow and MaskHand, [medium] Mask name/key derivation duplicated and fragile across handlers (split, [medium] mask_exists builds mask path from extension-less image_name, returning, [low] Placeholder visualization widget at panel index 2 never used; viz widg, [low] set_opengl_viewports_enabled silently swallows all exceptions, [low] load_results setCurrentText to same value does not emit signal, so ref

### Community 2 - "dash_app.py"
Cohesion: 1.0
Nodes (8): [high] run_server swallows all exceptions with bare pass across dash apps, [high] IndexError when get_unique_tubes() returns empty list on view switch, [medium] Visualization callbacks return dash.no_update on exception, hiding err, [medium] Image hover interval boundary uses hard-coded offset -9 / fixed date f, [low] 'downloadCsv' is not a valid Plotly modeBar button and is silently ign, [low] Add Range silently discards reversed ranges (from > to) with no user f, [low] Pointless try/except wrappers (__init__, _setup_layout) that catch onl, [info] selected-tubes-store can hold unlimited tube IDs while badges truncate

### Community 3 - "skeleton_correction_interface.py"
Cohesion: 1.0
Nodes (7): [high] Saving downscales a 1px skeleton with INTER_NEAREST, fragmenting lines, [high] cv2.imread/imwrite fail silently on non-ASCII / Unicode file paths on , [medium] save_skeleton ignores cv2.imwrite return value and emits success uncon, [medium] Full skeleton vectorization (topology) re-run on every display refresh, [medium] Select->edit erase of original polyline can wipe shared junction pixel, [low] apply_normalization rescales enhanced image with IgnoreAspectRatio, di, [low] load_image does not validate QPixmap loaded successfully (isNull)

### Community 4 - "display_controller.py"
Cohesion: 1.0
Nodes (7): [high] Color-table threshold mismatch: Otsu binary mask is only 0/255 but col, [high] QImage wraps numpy buffer without keeping the array alive (use-after-f, [medium] Zoom attribute drifts past min/max because clamp checks pre-multiplica, [medium] New QGraphicsScene created on every display update — no scene reuse, [low] INTER_NEAREST resize of skeleton/thin-mask overlay can drop single-pix, [low] display_single_image lacks pixmap.isNull() check unlike other display , [low] DisplayController tightly coupled to MainWindow internals

### Community 5 - "dash_app_area.py"
Cohesion: 1.0
Nodes (6): [high] Area callback silently returns dash.no_update on all exceptions, maski, [medium] Area profile uses mean aggregation where sum is scientifically appropr, [medium] Chart data recomputed from scratch on every callback; declared dcc.Sto, [low] Profile view returns dash.no_update leaving UI in broken mixed state w, [low] Graph container style dict duplicated across five callback return bran, [low] trigger_id computed but unused in stacked and time view branches

### Community 6 - "skeleton_graph_model.py"
Cohesion: 1.0
Nodes (6): [high] O(n) list.pop(0) used to trim undo/redo stacks instead of collections., [medium] topology() computes degrees on a thick (non-skeletonized) mask, inflat, [medium] frozen dataclass SkeletonTopology holds mutable List fields — immutabi, [medium] RDP simplification is recursive with no depth limit; long polylines ca, [low] _ensure_uint8_binary casts to uint8 before thresholding, zeroing float, [low] _compute_degree_map uses 8-neighbor Python loop instead of a single co

### Community 7 - "root_length_visulization.py"
Cohesion: 1.0
Nodes (6): [high] _show_visualization called twice per startup via two independent paths, [medium] Data race on self.server between GUI thread stop() and server thread r, [medium] Field-map / TubeIDS DataProcessor initialization silently removed, cau, [low] RootLengthVisualization and RootAreaVisualization are ~200-line near-i, [low] _check_server swallows all exceptions, polling forever with no escalat, [low] TOCTOU in _is_port_available makes the port check redundant and falsel

### Community 8 - "root_length_inference_handler.py"
Cohesion: 1.0
Nodes (5): [high] Averaging anisotropic X/Y scale factors and using raw pixel count is p, [medium] Physical field-of-view and resolution constants hardcoded in both hand, [low] Skeleton pixel count uses np.sum(skeleton)/255 instead of np.count_non, [low] Sort key `x['Tube'] or float('inf')` mis-sorts legitimate zero Tube/Po, [low] CSV written without explicit UTF-8 encoding; non-ASCII headers risk mo

### Community 9 - "mask_generation_handler.py"
Cohesion: 1.0
Nodes (5): [high] Bare except swallows per-batch exceptions silently; failed images hidd, [high] DataLoader num_workers>0 inside a QThread will deadlock or crash on Wi, [high] torch.load called without weights_only=True (untrusted-pickle code exe, [medium] Mask input directory inferred from first loaded image instead of image, [low] _initialize_model swallows all exceptions, hiding why model init faile

### Community 10 - "skeleton_inference.py"
Cohesion: 1.0
Nodes (4): [medium] is_image_file extension check is case-sensitive and misses mixed-case , [medium] No per-image error isolation; one corrupt image aborts the entire batc, [low] tensor2im always indexes batch [0], silently dropping all but the firs, [low] run_inference docstring states batch_size default 4 but signature defa

### Community 11 - "root_length_inference_handler.py"
Cohesion: 1.0
Nodes (3): [high] Failed images recorded as Length=0 / Area=0, indistinguishable from ge, [medium] cv2.imread returns None on unreadable/missing/non-ASCII file -> crypti, [medium] Otsu threshold + mean>127 inversion heuristic can invert sparse-root i

### Community 12 - "mask_drawing_tools.py"
Cohesion: 1.0
Nodes (3): [high] ARGB32 buffer written back as RGBA8888, causing channel swap in flood , [high] flood_fill uses wrong-sized local_mask (height+2,width+2) to index ima, [medium] Double undo push in flood_fill — caller and flood_fill both call save_

### Community 13 - "skeleton_inference.py"
Cohesion: 1.0
Nodes (2): [high] save_image aspect-ratio resize distorts skeleton geometry and swaps wi, [high] CHECKPOINTS_DIR is a relative path; model load fails unless CWD is rep

### Community 14 - "dash_image_utils.py"
Cohesion: 1.0
Nodes (2): [high] Images re-read, decoded, resized and base64-encoded from disk on every, [medium] build_available_images_map mixes string and tuple keys; set-building c

### Community 15 - "skeleton_handler.py"
Cohesion: 1.0
Nodes (2): [high] calculator_thread reference overwritten before previous thread finishe, [medium] Root length calculation only matches '_fake.png' suffix, silently prod

### Community 16 - "generate_skeleton_handler.py"
Cohesion: 1.0
Nodes (2): [high] SkeletonGeneratorThread stored as self.thread may be GC'd before compl, [medium] BATCH_SIZE=64 for Pix2Pix ResNet-9 256x256 generation risks GPU OOM on

### Community 17 - "root_area_inference_handler.py"
Cohesion: 1.0
Nodes (1): [critical] Root area calibration assumes mask is always 640x480 regardless of act

### Community 18 - "dash_data_cache.py"
Cohesion: 1.0
Nodes (1): [high] lru_cache on get_interval_data instance method leaks instances and ret

### Community 19 - "file_tree_manager.py"
Cohesion: 1.0
Nodes (1): [medium] Module-level _item_cache global causes stale/deleted QTreeWidgetItem r

### Community 20 - "image_manager.py"
Cohesion: 1.0
Nodes (1): [medium] QThread.terminate() called on loader thread without disconnecting fini

### Community 21 - "data_processor.py"
Cohesion: 1.0
Nodes (1): [medium] Bare except returns empty DataFrame silently on CSV load failure; date

### Community 22 - "image_normalization_interface.py"
Cohesion: 1.0
Nodes (1): [medium] Division by zero when lower==upper percentile in contrast stretching

### Community 23 - "model.py"
Cohesion: 1.0
Nodes (1): [medium] Position attention value aggregation transposed relative to standard s

### Community 24 - "mask_tracing_interface.py"
Cohesion: 1.0
Nodes (1): [low] Wheel-event brush-size cap (50) inconsistent with slider maximum (100)

### Community 25 - "Scientific validity"
Cohesion: 1.0
Nodes (1): D1: Scientific validity

### Community 26 - "ML inference"
Cohesion: 1.0
Nodes (1): D2: ML inference

### Community 27 - "Correctness"
Cohesion: 1.0
Nodes (1): D3: Correctness

### Community 28 - "Performance"
Cohesion: 1.0
Nodes (1): D4: Performance

### Community 29 - "Memory/resources"
Cohesion: 1.0
Nodes (1): D5: Memory/resources

### Community 30 - "Error handling"
Cohesion: 1.0
Nodes (1): D6: Error handling

### Community 31 - "Security"
Cohesion: 1.0
Nodes (1): D7: Security

### Community 32 - "Architecture"
Cohesion: 1.0
Nodes (1): D8: Architecture

### Community 33 - "Concurrency"
Cohesion: 1.0
Nodes (1): D9: Concurrency

### Community 34 - "Interoperability"
Cohesion: 1.0
Nodes (1): D10: Interoperability

### Community 35 - "Reproducibility"
Cohesion: 1.0
Nodes (1): D11: Reproducibility

### Community 36 - "Docs"
Cohesion: 1.0
Nodes (1): D12: Docs

### Community 37 - "Repo hygiene"
Cohesion: 1.0
Nodes (1): D13: Repo hygiene

## Knowledge Gaps
- **21 isolated node(s):** `D1: Scientific validity`, `D2: ML inference`, `D3: Correctness`, `D4: Performance`, `D5: Memory/resources` (+16 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `root_area_inference_handler.py`** (2 nodes): `root_area_inference_handler.py`, `[critical] Root area calibration assumes mask is always 640x480 regardless of act`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `dash_data_cache.py`** (2 nodes): `dash_data_cache.py`, `[high] lru_cache on get_interval_data instance method leaks instances and ret`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `file_tree_manager.py`** (2 nodes): `file_tree_manager.py`, `[medium] Module-level _item_cache global causes stale/deleted QTreeWidgetItem r`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `image_manager.py`** (2 nodes): `image_manager.py`, `[medium] QThread.terminate() called on loader thread without disconnecting fini`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `data_processor.py`** (2 nodes): `data_processor.py`, `[medium] Bare except returns empty DataFrame silently on CSV load failure; date`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `image_normalization_interface.py`** (2 nodes): `image_normalization_interface.py`, `[medium] Division by zero when lower==upper percentile in contrast stretching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `model.py`** (2 nodes): `model.py`, `[medium] Position attention value aggregation transposed relative to standard s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `mask_tracing_interface.py`** (2 nodes): `mask_tracing_interface.py`, `[low] Wheel-event brush-size cap (50) inconsistent with slider maximum (100)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Scientific validity`** (1 nodes): `D1: Scientific validity`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `ML inference`** (1 nodes): `D2: ML inference`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Correctness`** (1 nodes): `D3: Correctness`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Performance`** (1 nodes): `D4: Performance`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Memory/resources`** (1 nodes): `D5: Memory/resources`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Error handling`** (1 nodes): `D6: Error handling`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Security`** (1 nodes): `D7: Security`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Architecture`** (1 nodes): `D8: Architecture`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Concurrency`** (1 nodes): `D9: Concurrency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Interoperability`** (1 nodes): `D10: Interoperability`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Reproducibility`** (1 nodes): `D11: Reproducibility`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Docs`** (1 nodes): `D12: Docs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Repo hygiene`** (1 nodes): `D13: Repo hygiene`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Are the 12 inferred relationships involving `[high] dict[] generic-alias subscript used as a constructor call in fig.updat` (e.g. with `[high] Legend dedup flag set unconditionally, silencing legends when first tu` and `[medium] get_tube_date_availability does O(tubes*dates) DataFrame filters per c`) actually correct?**
  _`[high] dict[] generic-alias subscript used as a constructor call in fig.updat` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `[high] Legend dedup flag set unconditionally, silencing legends when first tu` (e.g. with `[high] dict[] generic-alias subscript used as a constructor call in fig.updat` and `[medium] get_tube_date_availability does O(tubes*dates) DataFrame filters per c`) actually correct?**
  _`[high] Legend dedup flag set unconditionally, silencing legends when first tu` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `[medium] get_tube_date_availability does O(tubes*dates) DataFrame filters per c` (e.g. with `[high] dict[] generic-alias subscript used as a constructor call in fig.updat` and `[high] Legend dedup flag set unconditionally, silencing legends when first tu`) actually correct?**
  _`[medium] get_tube_date_availability does O(tubes*dates) DataFrame filters per c` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `[medium] Greedy ±3-day date-merging is order-dependent and sums (not averages) ` (e.g. with `[high] dict[] generic-alias subscript used as a constructor call in fig.updat` and `[high] Legend dedup flag set unconditionally, silencing legends when first tu`) actually correct?**
  _`[medium] Greedy ±3-day date-merging is order-dependent and sums (not averages) ` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `[medium] parse_tube_selection drops all valid results on first parse error` (e.g. with `[high] dict[] generic-alias subscript used as a constructor call in fig.updat` and `[high] Legend dedup flag set unconditionally, silencing legends when first tu`) actually correct?**
  _`[medium] parse_tube_selection drops all valid results on first parse error` has 12 INFERRED edges - model-reasoned connections that need verification._
- **What connects `D1: Scientific validity`, `D2: ML inference`, `D3: Correctness` to the rest of the system?**
  _21 weakly-connected nodes found - possible documentation gaps or missing edges._