# RootQuant-V2 vs RootSegResNet vs Pix2PixRoot Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how close RootQuant-V2 (RQv2, image-level trait regressor), RootSegResNet (SPROUTS mask model -> area) and Pix2PixRoot (SPROUTS skeleton model -> length) come to ground truth, on (Part 1) the RootQuant trait test set and (Part 2) the 726-image pixel-labelled set that Pix2PixRoot was tested on, where RQv2 also gets pixel masks from its patch-token PCA.

**Architecture:** One standalone benchmark folder `E:\RootQuantV2\benchmark_sprouts\` holds a small Python package `bench/` (paths, metrics, pixel tools, GPU lock, SPROUTS runner, RQv2 runner), numbered scripts that produce CSVs, and a final report. SPROUTS models and SPROUTS's own measurement code (`app.inference.metrics`, 18 x 13 mm field of view) turn masks into area and skeletons into length. RQv2 runs from the release package; its existing NCSA predictions on the filtered test set are reused for Part 1 and spot-checked locally.

**Tech Stack:** Python 3.10 in the SPROUTS venv (`torch 2.8.0.dev+cu128`, RTX 5070 Ti 16 GB), numpy, pandas, scipy, scikit-image 0.25, OpenCV, matplotlib, pytest.

**Spec:** the user's request (2026-10-25 session) plus these decisions: data lives in `E:\RootQuant`; the 247 mislabelled test images listed in `E:\RootQuantV2\images_to_exclude_from_test.csv` are removed (test n = 17,313), and RQv2's results on that set are in `E:\RootQuantV2\runs\inference_eval\`.

## Global Constraints

- Python: always `D:\Root-Mask-and-Skeletons\venv\Scripts\python.exe` (absolute path). Never bare `python`. Never `pip install` into that venv.
- Extra packages RQv2's torch.hub DINOv3 code may need (`termcolor`, `torchmetrics`, `lightning-utilities`): install only with `pip install --no-deps --target E:\RootQuantV2\benchmark_sprouts\_site <pkg>`; `bench/paths.py` prepends `_site` to `sys.path`.
- Read-only: `D:\Root-Mask-and-Skeletons\**`, `E:\RootQuantV2-release\**`, `E:\RootQuantV2\**` (except the new `benchmark_sprouts\`), `E:\RootQuant\**`, `E:\pix2pix_skeleton_gen\**`. Write only under `E:\RootQuantV2\benchmark_sprouts\`.
- `E:\RootQuantV2` is not a git repo. Do not `git init`. Instead of commits, each task appends a dated section to `E:\RootQuantV2\benchmark_sprouts\PROGRESS.md` (what ran, row counts, headline numbers, problems).
- Never fabricate or impute a value. A failed image gets an `error` cell and is counted and reported, not dropped silently and not set to 0.
- Calibration is SPROUTS's: `Calibration(fov_width_mm=18.0, fov_height_mm=13.0)` mapped onto each array's own resolution. Area = `app.inference.metrics.root_area_mm2(mask0_255)`. Length = `app.inference.metrics.root_length_mm(skel0_255)` (arc length). Import these; do not reimplement them.
- SPROUTS model weights: `D:\Root-Mask-and-Skeletons\checkpoints\mask_weights\best_mask_model_V5.pth` (RootSegResNet), `D:\Root-Mask-and-Skeletons\checkpoints\skeletonizer\latest_net_G.pth` (Pix2PixRoot). Mask threshold 0.5 (`app.config.MASK_THRESHOLD`).
- RQv2: package root `E:\RootQuantV2-release\rootquant-v2` (insert on `sys.path`, then `import RootQuantV2`); checkpoint `E:\RootQuantV2-release\drive-upload\rootquant-v2-weights.pt` (SHA-256 `f55ca3fb7d85fef0b894d919e561c64105ce5528530ea2a76e46204c5e3a55f2`, same 95,561,807 bytes as `E:\RootQuantV2\runs\checkpoints\v2_dinov3_mona768_v3\best.pt`); env `DINOV3_CHECKPOINT_PATH=E:\RootQuantV2\dinov3\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`; torch.hub code already cached at `C:\Users\IGB\.cache\torch\hub\facebookresearch_dinov3_main`. Precision fp32; TTA = the checkpoint's 8-view D4 (`tta_predict_mean`).
- GPU: one 16 GB card shared by parallel agents. Every GPU job runs inside `bench.gpu_lock.gpu_lock(name)`.
- Long jobs (> 5 min): run in the background with stdout/stderr to `E:\RootQuantV2\benchmark_sprouts\work\logs\<script>.log`; scripts must resume (skip outputs that already exist).
- Tests: `$env:QT_QPA_PLATFORM='offscreen'; & D:\Root-Mask-and-Skeletons\venv\Scripts\python.exe -m pytest E:\RootQuantV2\benchmark_sprouts\tests -v --rootdir E:\RootQuantV2\benchmark_sprouts -p no:cacheprovider`.
- Regression metric definitions match `E:\RootQuantV2\fullset_metrics.py::_metrics` (R2 = 1 - SS_res/SS_tot about the GT mean; RMSE; MAE).

## Data (verified 2026-10-25)

| Item | Path | Notes |
|---|---|---|
| Part 1 GT | `E:\RootQuant\dataset\test_data.csv` | 17,560 rows: `ImageName, AliveLength(mm), AliveSurfArea(mm2)`; ~72 % rows are 0,0 |
| Part 1 images | `E:\RootQuant\dataset\images\<ImageName>` | 640 x 480 RGB jpg |
| Exclusion list | `E:\RootQuantV2\images_to_exclude_from_test.csv` | column `image_name`, 247 rows (reasons in `runs\inference_eval\removed_247_images.csv`) |
| RQv2 Part 1 preds | `E:\RootQuantV2\runs\inference_eval\preds\v2_dinov3_mona768_v3.csv` | 17,560 rows: `model,image,species,pred_length_vanilla,pred_area_vanilla,pred_length_tta,pred_area_tta,true_length,true_area`; species `maize`/`soy` |
| RQv2 reference numbers | `E:\RootQuantV2\runs\inference_eval\metrics_summary_fullset_filtered.csv` | `v2_dinov3_mona768_v3, filtered_17313, all, tta`: length R2 0.94956 RMSE 2.0252 MAE 0.7723; area R2 0.93002 RMSE 3.1497 MAE 1.0473 |
| Part 2 images / GT | `E:\RootQuant\Publication Material\LABELED_ROOT_IMAGERY\DATA\{IMG\*.jpg, MASK\*.png, SKELETON\*.png}` | 726 stems (identical to `E:\pix2pix_skeleton_gen\DATA`). IMG/MASK 640 x {480, 475, 460, ...}; SKELETON mostly 640 x 480, at least one 256 x 256 |
| Part 2 held-out split | stems of `E:\pix2pix_skeleton_gen\Data_Split\val\B\*.png` | 73 stems = Pix2PixRoot's test split (`test73`); the other 653 were its training set (`cal653`) |
| Part 2 reference GT | `E:\RootQuant\Publication Material\LABELED_ROOT_IMAGERY\predictions_ms_img.csv` | `image_name, gt_length, gt_area` from the RootQuant pipeline (mask -> spur-pruned skeleton); cross-check only |

Known caveats the report must state: RootSegResNet's training data is undocumented (`docs/MODELS.md`), so the 726 set may overlap it; Pix2PixRoot trained on `cal653`, so only `test73` is a fair Pix2PixRoot test; RQv2's frozen-backbone PCA is documented in `RootQuantV2/viz_root_panel.py` as task-agnostic (it also lights up soil texture); GT `AliveSurfArea(mm2)` may not be the same quantity as SPROUTS projected area (Task 6 measures the ratio, never rescales).

## File structure

```
E:\RootQuantV2\benchmark_sprouts\
  pytest.ini, conftest.py          # conftest puts the folder on sys.path
  PROGRESS.md                      # task log (replaces commits)
  bench\__init__.py
  bench\paths.py                   # every path constant + sys.path / env bootstrap
  bench\gpu_lock.py                # cross-process GPU lock
  bench\stats.py                   # regression + agreement metrics, bootstrap CIs
  bench\pixel.py                   # mask/skeleton binarisation, IoU family, tolerant skeleton F1, map->mask
  bench\sprouts_run.py             # stage images, run SPROUTS mask+skeleton models, measure
  bench\rqv2_run.py                # load RQv2, traits (vanilla/TTA), per-patch maps
  scripts\p1_manifest.py           # Part 1 manifest (n = 17,313)
  scripts\p1_sprouts.py            # Part 1 RootSegResNet area + Pix2PixRoot length
  scripts\p1_rqv2_spotcheck.py     # local RQv2 == NCSA preds on 32 images
  scripts\p2_manifest_gt.py        # Part 2 manifest + GT traits via SPROUTS
  scripts\p2_sprouts.py            # Part 2 SPROUTS outputs + traits
  scripts\p2_rqv2.py               # Part 2 RQv2 traits + maps (.npz)
  scripts\p2_rqv2_masks.py         # thresholds on cal653, masks, pixel + skeleton metrics
  scripts\evaluate.py              # all metric tables
  scripts\make_report.py           # figures + results\REPORT.md
  tests\test_stats.py, test_pixel.py, test_p1_manifest.py, test_p2_manifest.py, test_rqv2_geometry.py
  work\  (staging hardlinks, model outputs, maps, logs, gpu.lock)   results\  (CSVs, JSON, figures, REPORT.md)
```

## Review Focus

1. Name joins across `.jpg` / `.png` / `_fake.png` / `_combined` forms silently dropping rows: every join asserts its expected count (17,313; 726; 73) and fails loudly.
2. Letterbox geometry off by a pixel or a patch in RQv2 map -> image projection, shifting masks and collapsing IoU: a synthetic round-trip test pins it (Task 3).
3. GT polarity and odd sizes (white-on-black vs black-on-white, the 256 x 256 skeleton, 640 x 460 images) giving wrong GT: binarisation tests with both polarities and a resize case (Task 1).
4. The 247-image exclusion not applied, or applied with mismatched names: manifest test asserts n = 17,313 and zero overlap (Task 2), and Task 6 reproduces the RQv2 reference R2 to 4 decimals.
5. Parallel agents oversubscribing the GPU (OOM or a hung lock): every GPU job holds `gpu_lock`; the lock records owner + pid and is released in `finally`.

---

### Task 1: Foundations (`bench/` core + tests)

**Files:**
- Create: `pytest.ini`, `conftest.py`, `PROGRESS.md`, `bench/__init__.py`, `bench/paths.py`, `bench/gpu_lock.py`, `bench/stats.py`, `bench/pixel.py`
- Test: `tests/test_stats.py`, `tests/test_pixel.py`

**Interfaces:**
- Produces:
  - `bench.paths`: constants `ROOT, WORK, RESULTS, LOGS, SITE, SPROUTS_ROOT, RQV2_PKG_ROOT, RQV2_CKPT, RQV2_CKPT_SHA256, DINOV3_WEIGHTS, P1_GT_CSV, P1_IMAGES, P1_EXCLUDE_CSV, P1_RQV2_PREDS, P1_RQV2_REF_METRICS, P2_IMG, P2_MASK, P2_SKEL, P2_SPLIT_DIR, P2_REF_CSV` (all `pathlib.Path`, values from the Data table); `bootstrap() -> None` (idempotent: prepends `SITE`, `SPROUTS_ROOT`, `RQV2_PKG_ROOT` to `sys.path`, sets `DINOV3_CHECKPOINT_PATH` if unset, creates `WORK/RESULTS/LOGS`).
  - `bench.gpu_lock.gpu_lock(name: str, poll_s: float = 15.0)` context manager: lock file `WORK/gpu.lock` created with `os.O_CREAT | os.O_EXCL`, content `name pid iso-time`; waits (prints owner every poll) while it exists; removes it in `finally`.
  - `bench.stats.regression_metrics(pred: np.ndarray, gt: np.ndarray) -> dict` with keys `n, r2, rmse, mae, bias, pearson_r, ccc, slope, intercept, median_ratio` (`bias = mean(pred-gt)`; `ccc` = Lin's concordance; `slope/intercept` = OLS of pred on gt; `median_ratio = median(pred/gt)` over gt > 1.0, NaN if none). NaN-safe: rows with NaN pred or gt are excluded and `n` reflects that.
  - `bench.stats.bootstrap_ci(pred, gt, key: str, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]` (2.5 / 97.5 percentiles of `regression_metrics(...)[key]` over row resamples).
  - `bench.stats.empty_fp(pred: np.ndarray, gt: np.ndarray, thr: float) -> dict` keys `n_empty, fp_rate, mean_pred_on_empty` (empty = gt == 0; FP = pred > thr).
  - `bench.pixel.to_binary(img: np.ndarray, *, invert_if_majority: bool = True, otsu: bool = False) -> np.ndarray[bool]` (threshold 127 unless `otsu`; inverts when foreground > 50 % of pixels, same rule as SPROUTS `_mostly_white`).
  - `bench.pixel.to_u8(mask: np.ndarray[bool]) -> np.ndarray[uint8]` (0/255).
  - `bench.pixel.resize_nearest(mask: np.ndarray, w: int, h: int) -> np.ndarray` and `resize_linear(arr: np.ndarray, w: int, h: int) -> np.ndarray` (cv2 `INTER_NEAREST` / `INTER_LINEAR`, float32 for linear).
  - `bench.pixel.thin(mask: np.ndarray[bool]) -> np.ndarray[bool]` (`skimage.morphology.skeletonize`).
  - `bench.pixel.mask_metrics(pred: bool[H,W], gt: bool[H,W]) -> dict` keys `iou, dice, precision, recall, tp, fp, fn` (shapes must match, else `ValueError`; empty-pred precision = NaN; both empty -> iou = dice = 1.0).
  - `bench.pixel.skeleton_f1(pred_skel: bool[H,W], gt_skel: bool[H,W], tol_px: float = 3.0) -> dict` keys `precision, recall, f1` (precision = share of pred skeleton pixels with Euclidean distance <= tol to a GT skeleton pixel, via `scipy.ndimage.distance_transform_edt(~gt_skel)`; recall symmetric; f1 = 0 when both are 0; empty pred -> precision NaN, recall 0, f1 0).

- [ ] **Step 1: Write failing tests** in `tests/test_stats.py`:
  - `test_perfect_prediction`: `pred = gt = [0,1,2,3,4]` -> `r2 == 1`, `rmse == 0`, `mae == 0`, `ccc == 1`, `slope == 1`, `intercept == 0`.
  - `test_matches_rqv2_definition`: random `gt` (seed 0, n=200), `pred = gt + noise`; `r2`, `rmse`, `mae` equal `1 - sum((p-g)^2)/sum((g-g.mean())^2)`, `sqrt(mean((p-g)^2))`, `mean(|p-g|)` to 1e-12.
  - `test_scale_bias_visible`: `pred = 3.14159 * gt` (gt in 1..50) -> `median_ratio == approx(3.14159)`, `slope == approx(3.14159)`, `pearson_r == approx(1)`, `ccc < 0.5`.
  - `test_nan_rows_dropped`: one NaN pred in 10 rows -> `n == 9`.
  - `test_bootstrap_ci_brackets_point`: lower <= point r2 <= upper; same seed -> same CI.
  - `test_empty_fp`: gt `[0,0,0,5]`, pred `[0.0,2.0,0.5,5]`, thr 1.0 -> `n_empty 3`, `fp_rate == 1/3`.
  In `tests/test_pixel.py`:
  - `test_to_binary_both_polarities`: a 20 x 20 white-on-black square and its inverse give the same bool mask.
  - `test_mask_metrics_known`: pred = 10 px, gt = 10 px, overlap 5 -> `iou == 5/15`, `dice == 0.5`, `precision == 0.5`, `recall == 0.5`.
  - `test_mask_metrics_shape_mismatch_raises`.
  - `test_skeleton_f1_shifted_line`: horizontal line vs the same line shifted 2 px -> f1 == 1 at tol 3; shifted 5 px -> f1 == 0.
  - `test_skeleton_f1_empty_pred`: precision NaN, recall 0, f1 0.
  - `test_resize_nearest_keeps_binary`: a 256 x 256 0/255 mask resized to 640 x 480 holds only {0, 255}.
- [ ] **Step 2: Run tests** (command in Global Constraints). Expected: FAIL (import errors).
- [ ] **Step 3: Implement** `bench/paths.py`, `bench/gpu_lock.py`, `bench/stats.py`, `bench/pixel.py`, `pytest.ini` (`[pytest]` + `testpaths = tests`), `conftest.py` (insert benchmark root on `sys.path`), `PROGRESS.md` (title + plan path).
- [ ] **Step 4: Run tests.** Expected: all PASS. Also run `& $PY -c "import sys; sys.path.insert(0, r'E:\RootQuantV2\benchmark_sprouts'); from bench import paths; paths.bootstrap(); from app.inference.metrics import root_area_mm2, root_length_mm; import RootQuantV2; print('ok')"` -> prints `ok`.
- [ ] **Step 5: Log** a Task 1 section in `PROGRESS.md`.

### Task 2: Part 1 — manifest + SPROUTS traits on the RootQuant test set

**Files:**
- Create: `bench/sprouts_run.py`, `scripts/p1_manifest.py`, `scripts/p1_sprouts.py`
- Test: `tests/test_p1_manifest.py`

**Interfaces:**
- Consumes: `bench.paths`, `bench.gpu_lock.gpu_lock`.
- Produces:
  - `results/p1_manifest.csv` columns `image, species, gt_length_mm, gt_area_mm2, present, rqv2_length_vanilla, rqv2_area_vanilla, rqv2_length_tta, rqv2_area_tta` (17,313 rows; `present = gt_length_mm > 0 and gt_area_mm2 > 0`; GT from `test_data.csv`, RQv2 columns + species from the preds CSV).
  - `bench.sprouts_run.stage(names: list[str], src_dir: Path, dst_dir: Path) -> int` (hardlink with `os.link`, copy if linking fails; skip existing; returns count present).
  - `bench.sprouts_run.run_masks(in_dir: Path, out_dir: Path, batch_size: int = 8) -> BatchReport` (wraps `app.inference.mask_inference.generate_masks`; skip stems whose `<stem>.png` exists) and `run_skeletons(in_dir: Path, out_dir: Path, batch_size: int = 16) -> BatchReport` (wraps `app.inference.skeleton_inference.run_skeleton_generation`). Both inside `gpu_lock`.
  - `bench.sprouts_run.measure_area(mask_png: Path) -> float` = `root_area_mm2(load_mask(path))`; `measure_length(fake_png: Path) -> float` = `root_length_mm(preprocess_image(path))` (both from `app.inference.metrics`).
  - `results/p1_sprouts_traits.csv` columns `image, resnet_area_mm2, pix2pix_length_mm, error` (17,313 rows).
  - Model outputs: `work/p1/mask/<stem>.png`, `work/p1/skeletons/<stem>_fake.png`; staged inputs `work/p1/images/`.

- [ ] **Step 1: Write failing tests** in `tests/test_p1_manifest.py` (they read `results/p1_manifest.csv`, so they fail until Step 3 runs):
  - `test_row_count`: 17,313 rows, `image` unique.
  - `test_exclusion_applied`: no overlap with `P1_EXCLUDE_CSV.image_name`; `17,313 + 247 == 17,560`.
  - `test_gt_matches_rqv2_csv`: `|gt_length_mm - true_length| < 1e-3` and same for area on every row (fail if any differ; print the first 5).
  - `test_present_count_logged`: `present.sum()` is between 4,600 and 4,630 (paper visible-root subset is 4,618).
  - `test_species_values`: subset of `{maize, soy}`, no nulls.
- [ ] **Step 2: Implement `scripts/p1_manifest.py`**, run it, then run the tests. Expected: PASS. Every `image` must exist in `P1_IMAGES` (assert; list missing names if not).
- [ ] **Step 3: Implement `bench/sprouts_run.py`.** Dry run first: stage 5 manifest images into `work/p1_dry/images`, run masks + skeletons + measure, check outputs are 640 x 480 masks (0/255) and 341 x 256 `_fake.png`, values finite. Delete `work/p1_dry`.
- [ ] **Step 4: Implement `scripts/p1_sprouts.py`** (stage all 17,313 -> masks -> skeletons -> measure on a thread pool, `max_workers=8`) and run it in the background, log `work/logs/p1_sprouts.log`.
- [ ] **Step 5: Verify:** `results/p1_sprouts_traits.csv` has 17,313 rows, `error` empty for all (else list them), `work/p1/mask` and `work/p1/skeletons` each hold 17,313 images. Print `describe()` of both trait columns split by `present`.
- [ ] **Step 6: Log** Task 2 in `PROGRESS.md` (timings, row counts, failures).

### Task 3: RQv2 runner — reproduction spot-check + Part 2 traits and maps

**Files:**
- Create: `bench/rqv2_run.py`, `scripts/p1_rqv2_spotcheck.py`, `scripts/p2_rqv2.py`
- Test: `tests/test_rqv2_geometry.py`

**Interfaces:**
- Consumes: `bench.paths`, `bench.gpu_lock`, `results/p1_manifest.csv` (Task 2 Step 2; if absent, read `P1_RQV2_PREDS` directly for the spot-check).
- Produces:
  - `bench.rqv2_run.load_model(device: str = "cuda") -> RQ` where `RQ` is a dataclass `model, cfg, stats, transform, density_head, blend, device` (mirror `RootQuantV2/predict.py` lines 123-156 for loading; mirror `RootQuantV2/viz_root_panel.py::main` for `density_head` and `blend`). Verify the checkpoint SHA-256 equals `RQV2_CKPT_SHA256` before loading.
  - `bench.rqv2_run.predict_traits(rq: RQ, paths: list[Path], tta: bool, batch_size: int = 2) -> np.ndarray[float32, (n, 2)]` (length_mm, area_mm2; inverted with the checkpoint stats, clamped >= 0, exactly as `predict.py`).
  - `bench.rqv2_run.Geometry` dataclass `orig_w, orig_h, target, patch, grid, scale, new_w, new_h, left, top` and `letterbox_geometry(w: int, h: int, target: int, patch: int = 16) -> Geometry` (same arithmetic as `RootQuantV2/data/transforms.py::LetterboxToSquare.letterbox`: `scale = target / max(w, h)`, `new_w = round(w*scale)`, `left = (target-new_w)//2`, same for h/top).
  - `bench.rqv2_run.patch_maps(rq: RQ, path: Path) -> dict` keys `pc1, pca3, dens_len, dens_area` (each `(grid, grid) float32`, NaN outside valid patches), `valid (grid, grid) bool`, plus the `Geometry` fields. Port `viz_root_panel.analyze` (adapted tokens from `forward_features`, `pca_feats`, density `softplus(mlp) * gain`); skip the frozen-model branch. `pc1` uses this orientation rule instead of the border rule: after `_norm01`, flip to `1 - pc1` when more than half of the valid patches are > 0.5 (roots are the minority class).
  - `bench.rqv2_run.map_to_image(m: np.ndarray, geo: Geometry) -> np.ndarray[float32, (orig_h, orig_w)]`: replace NaN with the minimum of the valid values; resize the `grid x grid` map to `target x target` with `cv2.INTER_LINEAR` (each patch = `patch` px); crop `[top:top+new_h, left:left+new_w]`; resize to `(orig_w, orig_h)` with `cv2.INTER_LINEAR`.
  - `results/p1_rqv2_spotcheck.csv` columns `image, csv_length_vanilla, local_length_vanilla, csv_area_vanilla, local_area_vanilla, csv_length_tta, local_length_tta, csv_area_tta, local_area_tta`.
  - `results/p2_rqv2_traits.csv` columns `stem, rqv2_length_vanilla, rqv2_area_vanilla, rqv2_length_tta, rqv2_area_tta` (726 rows).
  - `work/p2/rqv2_maps/<stem>.npz` with the `patch_maps` keys (726 files).

- [ ] **Step 1: Write failing tests** in `tests/test_rqv2_geometry.py` (pure numpy, no GPU):
  - `test_geometry_matches_release_letterbox`: for `(w, h)` in `[(640,480), (640,460), (635,450), (480,640)]` and `target=896`, `letterbox_geometry` equals `(new_w, new_h, left, top)` from `RootQuantV2.data.transforms.LetterboxToSquare(896).letterbox(Image.new("RGB",(w,h)))`.
  - `test_map_roundtrip_corner_patch`: 640 x 480, target 896, a map that is 1.0 only on the valid patch covering the content's top-left corner -> after `map_to_image`, the pixel `(0, 0)` > 0.5 and the image centre < 0.1.
  - `test_valid_mask_matches_release`: `valid` derived from geometry equals `content_patch_mask_2d(new_w, new_h, left, top, 896, 16)`.
- [ ] **Step 2: Run tests.** Expected: FAIL. **Step 3:** implement `bench/rqv2_run.py`; tests PASS.
- [ ] **Step 4: Environment check:** `load_model()` on CUDA. If the hub import raises `ModuleNotFoundError`, install that module into `_site` (Global Constraints) and retry. Record peak GPU memory for `batch_size=2` with TTA.
- [ ] **Step 5: Spot-check** `scripts/p1_rqv2_spotcheck.py`: 32 manifest images (seed 0: 16 with `present`, 16 without), vanilla and TTA. Pass rule per value: `|local - csv| <= 0.05 + 0.02 * |csv|`. Expected: all 128 values pass. If any fail, stop and report the table (do not continue to Step 6).
- [ ] **Step 6: Run `scripts/p2_rqv2.py`** in the background (log `work/logs/p2_rqv2.log`): traits vanilla + TTA and `patch_maps` for all 726 images in `P2_IMG`. Verify 726 rows / 726 `.npz`, all finite traits, every map has >= 1 valid patch.
- [ ] **Step 7: Log** Task 3 in `PROGRESS.md` (spot-check max abs diff, timings, peak memory).

### Task 4: Part 2 — manifest, GT traits, SPROUTS outputs on the 726 set

**Files:**
- Create: `scripts/p2_manifest_gt.py`, `scripts/p2_sprouts.py`
- Test: `tests/test_p2_manifest.py`

**Interfaces:**
- Consumes: `bench.paths`, `bench.pixel`, `bench.sprouts_run` (Task 2; if Task 2 is not finished, wait for `bench/sprouts_run.py` to exist rather than writing a second copy).
- Produces:
  - `results/p2_manifest.csv` columns `stem, split, img_path, mask_path, skel_path, img_w, img_h, mask_w, mask_h, skel_w, skel_h, gt_area_mm2, gt_length_mm, ref_gt_area_mm2, ref_gt_length_mm` (726 rows; `split` is `test73` or `cal653`).
  - GT rules: `gt_area_mm2 = root_area_mm2(to_u8(to_binary(MASK)))` at the mask's own resolution. `gt_length_mm = root_length_mm(to_u8(thin(to_binary(SKELETON))))` at the skeleton's own resolution (a skeleton thinned again is unchanged where already 1 px). `ref_*` from `P2_REF_CSV` (`gt_length`, `gt_area`, joined on stem).
  - `work/p2/gt_skel/<stem>.png`: GT skeleton at the GT mask's resolution (`resize_nearest` to mask size when sizes differ, then `thin`), 0/255. Used by Task 5 for skeleton F1.
  - `results/p2_sprouts_traits.csv` columns `stem, resnet_area_mm2, pix2pix_length_mm, error` (726 rows); outputs in `work/p2/mask/<stem>.png`, `work/p2/skeletons/<stem>_fake.png`.

- [ ] **Step 1: Write failing tests** in `tests/test_p2_manifest.py`:
  - `test_counts`: 726 rows, unique stems; `split` counts `test73 == 73`, `cal653 == 653`; every test73 stem comes from `P2_SPLIT_DIR` (`*.png` stems).
  - `test_mask_matches_image_size`: `mask_w == img_w` and `mask_h == img_h` on every row (if not, the test lists them; Task 5 then resizes predictions to the mask size).
  - `test_gt_positive`: `gt_area_mm2 > 0` and `gt_length_mm > 0` on every row (the set has no empty images).
  - `test_gt_agrees_with_reference`: Pearson r between `gt_area_mm2` and `ref_gt_area_mm2` > 0.99 and median ratio within [0.95, 1.05]; for length, r > 0.9 (reference is spur-pruned from masks; ours is from the GT skeleton images). Print the ratios either way.
- [ ] **Step 2: Implement `scripts/p2_manifest_gt.py`**, run, run tests. Expected: PASS (if `test_gt_agrees_with_reference` fails, keep going but record numbers in `PROGRESS.md`; it is a cross-check, not a gate).
- [ ] **Step 3: Implement + run `scripts/p2_sprouts.py`**: stage the 726 IMG files to `work/p2/images`, `run_masks`, `run_skeletons`, measure. Verify 726 rows, `error` empty, mask files 640 x 480 0/255.
- [ ] **Step 4: Log** Task 4 in `PROGRESS.md` (GT vs reference ratios, odd-size files found).

### Task 5: Part 2 — RQv2 masks from PCA, pixel and skeleton metrics

**Files:**
- Create: `scripts/p2_rqv2_masks.py`
- Test: extend `tests/test_pixel.py` with `test_threshold_selection`

**Interfaces:**
- Consumes: `results/p2_manifest.csv`, `work/p2/gt_skel/`, `work/p2/mask/`, `work/p2/skeletons/` (Task 4); `work/p2/rqv2_maps/*.npz`, `bench.rqv2_run.map_to_image`, `Geometry` (Task 3); `bench.pixel`, `bench.sprouts_run.measure_area`.
- Produces:
  - `bench.pixel.select_threshold(scores: list[np.ndarray], gts: list[np.ndarray[bool]], taus: np.ndarray) -> tuple[float, float]` returning `(best_tau, best_mean_iou)`; mean per-image IoU; ties -> smaller tau.
  - RQv2 mask variants (all at the GT mask resolution, via `map_to_image`):
    - `rqv2_pca3_cal`: map `pca3`, tau chosen on `cal653` from `np.round(np.arange(0.02, 0.99, 0.02), 2)`.
    - `rqv2_pca3_otsu`: map `pca3`, per-image `skimage.filters.threshold_otsu` over the image's projected values.
    - `rqv2_pc1_cal`: map `pc1` (minority-oriented), tau chosen on `cal653`, same grid.
    - `rqv2_dens_cal`: map `dens_area` (absolute units, the model's own per-patch area density; label it "density head, not PCA"), tau grid = 50 evenly spaced quantiles (50th to 99.5th percentile) of the pooled projected `cal653` values.
    - `rqv2_pca_best`: whichever of `rqv2_pca3_cal`, `rqv2_pca3_otsu`, `rqv2_pc1_cal` has the highest mean IoU on `cal653` (decided on `cal653` only, before any `test73` number is computed).
  - `results/p2_rqv2_thresholds.json`: per variant `{map, rule, tau, cal_mean_iou, chosen_as_pca_best}`.
  - `work/p2/rqv2_masks/<variant>/<stem>.png` (0/255).
  - `results/p2_pixel_metrics.csv` columns `stem, split, method, iou, dice, precision, recall, area_mm2, length_mm` for `method` in `resnet` (SPROUTS mask resized nearest to GT size), `rqv2_pca3_cal`, `rqv2_pca3_otsu`, `rqv2_pc1_cal`, `rqv2_dens_cal`, `rqv2_pca_best`. `area_mm2 = root_area_mm2` of that mask; `length_mm = root_length_mm(thin(mask))` (mask-derived length; for `resnet` too, as a reference only).
  - `results/p2_skeleton_metrics.csv` columns `stem, split, method, precision, recall, f1` at `tol_px = 3.0` against `work/p2/gt_skel` for `method` in `pix2pix` (`_fake.png` grayscale resized with `resize_linear` to the GT mask size, then `to_binary(otsu=True)`, then `thin`) and `rqv2_pca_best_skel` (`thin` of the `rqv2_pca_best` mask).

- [ ] **Step 1: Write failing test** `test_threshold_selection`: two synthetic 10 x 10 score maps built only from the values `{0.10, 0.59, 0.61, 0.95}` (each value present in each map), GT = `score > 0.6`; `select_threshold(..., taus=np.round(np.arange(0.02,0.99,0.02),2))` returns `best_tau == 0.6` and `best_mean_iou == 1.0`. The prediction rule everywhere is strict: `mask = score > tau`.
- [ ] **Step 2: Run** -> FAIL; implement `select_threshold` in `bench/pixel.py`; run -> PASS.
- [ ] **Step 3: Implement + run `scripts/p2_rqv2_masks.py`** (thresholds from `cal653` first, written to JSON, then masks and metrics for all 726 rows).
- [ ] **Step 4: Verify:** `p2_pixel_metrics.csv` has 726 x 6 rows, `p2_skeleton_metrics.csv` 726 x 2 rows; no NaN IoU; print mean IoU per method for `cal653` and `test73`. Save a 6-image visual check `results/figures/p2_mask_examples.png` (3 test73 + 3 cal653; columns image | GT | ResNet | rqv2_pca_best | rqv2_dens_cal) and look at it: masks must sit on the roots, not shifted.
- [ ] **Step 5: Log** Task 5 in `PROGRESS.md` (chosen taus, cal IoU per variant).

### Task 6: Evaluation tables, figures, report

**Files:**
- Create: `scripts/evaluate.py`, `scripts/make_report.py`
- Test: add `test_reproduces_rqv2_reference` to `tests/test_stats.py`

**Interfaces:**
- Consumes: every `results/*.csv` above; `bench.stats`.
- Produces:
  - `results/p1_metrics.csv` columns `method, trait, subset, n, r2, r2_lo, r2_hi, rmse, mae, mae_lo, mae_hi, bias, pearson_r, ccc, slope, intercept, median_ratio`. Methods: `rqv2_tta` (headline), `rqv2_vanilla`, `rootsegresnet` (area only), `pix2pixroot` (length only). Subsets: `all` (17,313), `present`, `maize`, `soy`.
  - `results/p1_empty_fp.csv`: per method x trait `empty_fp` with `thr = 1.0` (mm or mm2).
  - `results/p2_trait_metrics.csv`: same columns, subsets `test73`, `all726`. Methods: `rqv2_tta`, `rqv2_vanilla`, `rqv2_pca_best_mask` (area and mask-derived length), `rqv2_dens_cal_mask`, `rootsegresnet` (area; plus mask-derived length labelled `rootsegresnet_mask_len`), `pix2pixroot` (length). GT = `gt_area_mm2`, `gt_length_mm`.
  - `results/p2_pixel_summary.csv`: per `method x split` mean +- sd of IoU, Dice, precision, recall, plus pooled IoU (sum tp / sum(tp+fp+fn)); skeleton P/R/F1 the same way.
  - Figures in `results/figures/`: `p1_scatter.png` (pred vs GT, one panel per method x trait, log1p axes, y = x line, n and R2 in the title), `p2_scatter_test73.png`, `p2_iou_box.png` (per-method IoU on test73).
  - `results/REPORT.md`: data and exclusions, method notes, the three tables (Part 1 traits, Part 2 traits, Part 2 pixels), units check (`median_ratio`, `slope` per method/trait with a one-line reading), every caveat from the Data section, failures/errors, and how to re-run (script order).

- [ ] **Step 1: Write failing test** `test_reproduces_rqv2_reference`: from `results/p1_manifest.csv`, `regression_metrics(rqv2_length_tta, gt_length_mm)` gives `r2 = 0.94956`, `rmse = 2.0252`, `mae = 0.7723` and area gives `r2 = 0.93002`, `rmse = 3.1497`, `mae = 1.0473`, each within 5e-5 (rounding of the reference file).
- [ ] **Step 2: Run** -> PASS expected once Task 2 Step 2 is done (if it fails, the manifest join is wrong: stop and fix Task 2's manifest, do not loosen the tolerance).
- [ ] **Step 3: Implement + run `scripts/evaluate.py`**, then `scripts/make_report.py`. Look at each figure.
- [ ] **Step 4: Verify** the report's numbers against the CSVs by recomputing 3 cells by hand (pandas one-liners) and printing them next to the table values.
- [ ] **Step 5: Log** Task 6 in `PROGRESS.md` with the headline table.

## Execution waves (Sonnet agents, high effort)

| Wave | Tasks | Why |
|---|---|---|
| 1 | Task 1 | Shared package every later task imports |
| 2 | Tasks 2, 3, 4 in parallel | Disjoint files; GPU jobs serialize on `gpu_lock`; Task 4 waits for `bench/sprouts_run.py` from Task 2 Step 3 before its Step 3 |
| 3 | Task 5 | Needs Task 3 maps + Task 4 GT/outputs |
| 4 | Task 6 | Needs everything |
| 5 | Lead review | Re-run the full test suite, spot-check numbers, sanity-read the report |
