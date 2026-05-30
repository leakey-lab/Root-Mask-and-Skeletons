# Remediation Report — Root-Mask-and-Skeletons (v3 hardening)

Branch: `refactor/v3-hardening` (off `v3`). Driven by the verified findings in
[`CODE_REVIEW.md`](CODE_REVIEW.md) / `code_review_findings.json`. Every group is
gated by the regression + import-smoke suite (`tests/`).

## Verification status

- **51 tests pass** (regression + import-smoke), headless (`QT_QPA_PLATFORM=offscreen`).
- **MainWindow constructs** end-to-end under offscreen Qt (full widget tree wired).
- **CUDA torch preserved**: numpy upgraded to 2.x while the GPU torch build keeps
  working (`torch.cuda.is_available()` → True; numpy bridge verified).
- `v3` is untouched; all work is isolated on `refactor/v3-hardening`.

## What changed, by phase

### R0 — Regression safety net
Added `tests/` pinning the scientifically load-bearing paths (length/area
calibration, skeleton vectorization topology, undo/redo bounds, CSV schema,
filename parsing, DataProcessor) plus an import-smoke over all 31 app modules.
Added `pyproject.toml` (pytest/ruff/black/isort config) and `requirements-dev.txt`.

### R1 — Dependency upgrade (torch pinned)
Live env upgraded to latest stable, **CUDA torch/torchvision left untouched** to
protect GPU inference:

| | before | after |
|---|---|---|
| numpy | 1.26.3 | 2.2.6 |
| scipy | 1.14.1 | 1.15.3 |
| scikit-image | 0.24.0 | 0.25.2 |
| opencv-python | 4.10.0 | 4.13.0 |
| pillow | 10.2.0 | 12.2.0 |
| pandas | 2.2.3 | 2.3.3 |
| dash | 2.18.1 | 4.1.0 |
| plotly | 5.24.1 | 6.7.0 |
| Flask / Werkzeug | 3.0.x | 3.1.x |
| PyQt6 / WebEngine | 6.7.x | 6.11.0 |

Breakage fixed: `Dash.run_server` is obsolete in dash 4 → switched to `Dash.run`.
`requirements.txt` rewritten as a curated UTF-8 runtime list (was a 113-package
full-env freeze with jupyter/wandb/ipython/sentry bloat); torch documented for a
separate CUDA-index install so pip cannot silently pull a CPU wheel.

### R4 + R5 (scientific core) — calibration fix + dedup
The review's #1 and #2 findings.

- **F-002 (CRITICAL) — area silent corruption.** Area is now calibrated from the
  mask's **actual** resolution × the physical field of view, not a hardcoded
  640×480. A full mask measures the whole FOV (18×13 = **234 mm²**) at any
  resolution; previously a non-640×480 mask was wrong by the square of the
  resolution ratio.
- **F-003 (HIGH) — length undercount.** Replaced the raw skeleton pixel **count**
  with a true arc-length estimator: orthogonal steps cost one pixel pitch,
  diagonal steps cost the Euclidean diagonal, using the per-axis (anisotropic)
  pitch. The old code treated √2 diagonals as 1, biasing length low ~15–20%. The
  fragile fixed 341×256 "processed/original" scaling hack is gone (derived from
  actual shape).
- **Dedup.** New `app/config.py` (single source of calibration, ports, checkpoint
  paths, batch sizes, thresholds, CSV schema) and `app/inference/metrics.py`
  (shared length/area math, parsing, CSV writer, thread-pool driver). The two
  ~220-line inference handlers are now thin wrappers; `data_processor` +
  `data_processor_area` collapse onto one `MetricDataProcessor`; the two Dash
  server threads share `_viz_server._DashServerThreadBase`.

**Golden values for length/area were intentionally updated** to the corrected
behavior, with resolution-invariance tests pinning both fixes (full mask → 234 mm²
at any resolution; full-width line → ~FOV width at any width; diagonal uses the
Euclidean step).

### R2 / R3 / R5 (modules) — memory, performance, OOP, error handling
Applied across 6 disjoint module groups (worktree-isolated, each gated on the
suite), addressing ~40 verified findings. Highlights:

- **Memory:** QImage backing-array kept alive for its lifetime (F-018
  use-after-free); undo/redo stacks moved to `deque(maxlen=…)` (F-019, also a
  perf win); Dash server threads joined on stop; CUDA cache freed between
  inference batches; per-event QImage reallocation in the eraser removed.
- **Performance:** degree-map computation vectorized via `scipy.ndimage.convolve`
  (was an 8-pass Python loop); hover thumbnails precomputed/cached instead of
  re-read+decoded+base64-encoded on every hover (F-013); availability lookups
  memoized (F-033, was O(tubes×dates) per callback); `torch.inference_mode()`.
- **Correctness:** `dict[...]` generic-alias used as a constructor (F-010);
  legend dedup silencing legends (F-011); `lru_cache` on an instance method
  leaking instances / returning stale data (F-012); empty-tube IndexError
  (F-009); signal double-connection (F-015); saved-mask tree colors wiped
  (F-016); color-table threshold mismatch (F-017); skeleton save width/height
  swap (F-004).
- **Robustness / security:** `torch.load(weights_only=True)` (F-023/SS-01);
  Unicode-safe cv2 image IO on Windows paths (F-007); ~115 bare/broad
  `except`/`except: pass` replaced with specific exceptions + module-level
  `logging` (no more silently-swallowed failures or metrics silently set to 0).
- **Architecture:** the `mask_handler → mask_tracing_interface → main_window`
  circular import broken (handlers imported lazily in `MainWindow.__init__`); all
  31 modules now import cleanly standalone.

### R6 — scale hardening, CI, docs
- High-throughput desktop runtime work (streaming batches, model loaded once,
  `inference_mode`, CUDA `empty_cache`, worker tuning) landed with R2/R3.
- `.github/workflows/ci.yml` (CPU-torch install + deps + ruff advisory + suite,
  headless), `.pre-commit-config.yaml` (ruff/format + hygiene hooks), and a new
  `README.md` (the repo had none) documenting the separate CUDA-torch install,
  the corrected calibration, and run/test instructions.

## Residual risks / follow-ups

- **Numeric re-validation against ground truth.** The arc-length and area fixes
  are mathematically correct and unit-tested, but the corrected absolute mm
  values should be validated against physical reference roots before publishing
  measurements. Historical CSVs produced before this branch carry the old biased
  length and the resolution-dependent area — re-run measurements rather than
  comparing across the fix boundary.
- **Dash 4 / plotly 6 runtime UI.** Imports and server startup are fixed and
  smoke-tested; the interactive callbacks were not exercised against a live
  browser session here. A manual pass over each view is recommended.
- **God-object.** `skeleton_correction_interface.py` was hardened and partially
  factored but not fully decomposed (kept conservative to avoid breaking Qt
  wiring); a deeper split remains available as future work.
- **Lint.** CI runs `ruff` in advisory mode; a dedicated cleanup pass can make it
  blocking once the legacy backlog is triaged.
- **Repo hygiene.** `checkpoints.zip` / large binaries in history were out of
  scope for this branch (history rewrite); `.gitignore` already excludes them
  going forward.

## How to proceed

```bash
# review the branch
git log --oneline v3..refactor/v3-hardening
# run the suite
QT_QPA_PLATFORM=offscreen pytest
# open a PR when ready (not done automatically)
```
