"""
Shared root-metric primitives (length & area).

This module unifies logic that was previously duplicated, near-identically,
across ``root_length_inference_handler`` and ``root_area_inference_handler``
(filename parsing, the per-image worker, CSV writing, the QThread driver) and
fixes the calibration science:

* Area (``root_area_mm2``) is computed from the mask's *actual* resolution and
  the physical field of view, so a mask that is not 640x480 no longer reports an
  area wrong by the square of the resolution ratio (F-002).

* Length (``root_length_mm``) uses a true arc-length estimator: orthogonal pixel
  steps cost one pixel pitch, diagonal steps cost the Euclidean diagonal. The
  previous code summed the raw skeleton pixel *count*, which undercounts every
  diagonal step (treating sqrt(2) as 1) and biased reported length low by
  ~15-20% (F-003). The estimator also uses the per-axis pixel pitch, so the
  near-isotropic (<4%) x/y difference is handled exactly rather than averaged.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

logger = logging.getLogger(__name__)

import cv2
import numpy as np
from skimage.morphology import skeletonize

from app.config import CALIBRATION


# --------------------------- image -> binary -------------------------------

def preprocess_image(image_path: str) -> np.ndarray:
    """Read an image and return its 1px skeleton (uint8 0/255)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = 255 - binary
    skeleton = skeletonize(binary / 255)
    return (skeleton * 255).astype(np.uint8)


def load_mask(image_path: str) -> np.ndarray:
    """Read a mask image and return a binary (0/255) array."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read mask: {image_path}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


# ----------------------------- metrics -------------------------------------

def skeleton_arc_length_px(skeleton: np.ndarray) -> tuple[float, float, float]:
    """Return (orthogonal_x_steps, orthogonal_y_steps, diagonal_steps) counted as
    edges in the 8-connected pixel graph of the skeleton."""
    s = skeleton > 0
    orth_x = int(np.count_nonzero(s[:, :-1] & s[:, 1:]))   # horizontal neighbors
    orth_y = int(np.count_nonzero(s[:-1, :] & s[1:, :]))   # vertical neighbors
    diag = (int(np.count_nonzero(s[:-1, :-1] & s[1:, 1:]))
            + int(np.count_nonzero(s[:-1, 1:] & s[1:, :-1])))
    return orth_x, orth_y, diag


def root_length_mm(skeleton: np.ndarray) -> float:
    """Root length in mm from a skeleton, using arc-length + per-axis calibration."""
    h, w = skeleton.shape[:2]
    mx, my = CALIBRATION.mm_per_px(w, h)
    orth_x, orth_y, diag = skeleton_arc_length_px(skeleton)
    diag_mm = math.hypot(mx, my)
    return orth_x * mx + orth_y * my + diag * diag_mm


def root_area_mm2(mask: np.ndarray) -> float:
    """Root area in mm^2 from a binary mask, calibrated to the mask's resolution."""
    h, w = mask.shape[:2]
    root_pixels = int(np.count_nonzero(mask == 255))
    return root_pixels * CALIBRATION.area_per_px_mm2(w, h)


# Backwards-compatible aliases (kept so existing imports keep working).
def calculate_root_length(skeleton: np.ndarray) -> float:
    return root_length_mm(skeleton)


def calculate_root_area(mask: np.ndarray) -> float:
    return root_area_mm2(mask)


# ----------------------------- filename ------------------------------------

def parse_image_name(name: str) -> dict:
    """Parse ``..._T{tube}_L{pos}_YYYY.MM.DD_HHMMSS_...`` into its fields."""
    info = {
        "original_name": name,
        "tube_number": None,
        "length_position": None,
        "date": None,
        "time": None,
    }
    try:
        if (m := re.search(r"T(\d+)", name)):
            info["tube_number"] = int(m.group(1))
        if (m := re.search(r"L(\d+)", name)):
            info["length_position"] = int(m.group(1))
        if (m := re.search(r"(\d{4})\.(\d{2})\.(\d{2})", name)):
            y, mo, d = m.groups()
            info["date"] = f"{y}.{mo}.{d}"
        if (m := re.search(r"_(\d{6})(?:_|$)", name)):
            t = m.group(1)
            info["time"] = f"{t[:2]}:{t[2:4]}:{t[4:]}"
    except (ValueError, AttributeError) as exc:
        logger.debug("parse_image_name could not parse %r: %s", name, exc)
    return info


# ----------------------------- CSV -----------------------------------------

def write_metric_csv(results: list[dict], filename: str, headers: list[str]) -> None:
    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in results:
                writer.writerow({h: row.get(h, "") for h in headers})
        logger.debug("Wrote %d rows to %s", len(results), filename)
    except OSError as exc:
        logger.error("Failed to write metric CSV %s: %s", filename, exc)
        raise


def _sort_key(row: dict):
    return (
        row.get("Tube") or float("inf"),
        row.get("Date") or "",
        row.get("Time") or "",
        row.get("Position") or float("inf"),
    )


def run_metric_pool(
    images: dict[str, str],
    worker: Callable[[str, str], dict],
    progress_cb: Callable[[int], None] | None = None,
) -> list[dict]:
    """Run ``worker(name, path)`` over ``images`` on a thread pool, return rows."""
    results: list[dict] = []
    total = len(images)
    if total == 0:
        return results
    # Cap at half of logical cores (min 2, max 8) so the Qt main thread
    # is not starved by workers saturating every CPU core.
    max_workers = min(max(2, (os.cpu_count() or 4) // 2), 8, total)
    last_pct = -1
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, name, path): name for name, path in images.items()}
        for i, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if progress_cb is not None:
                pct = int(i / total * 100)
                if pct != last_pct:
                    last_pct = pct
                    progress_cb(pct)
    return sorted(results, key=_sort_key)
