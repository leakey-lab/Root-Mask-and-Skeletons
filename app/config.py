"""
Central configuration for Root-Mask-and-Skeletons.

Single source of truth for the values that were previously hardcoded and
duplicated across the inference handlers, visualization apps, and GUI
(calibration constants, network ports, model-checkpoint paths, batch sizes,
thresholds). Importing from here keeps the scientific calibration in exactly
one place — see CALIBRATION below, which the root-length/area metrics depend on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# --- Repository / resource paths -------------------------------------------
# Resolve relative to this file so behavior does not depend on the current
# working directory (fixes F-005: relative CHECKPOINTS_DIR).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MASK_WEIGHTS_PATH = os.path.join(CHECKPOINTS_DIR, "mask_weights", "best_mask_model_V5.pth")
SKELETONIZER_DIR = os.path.join(CHECKPOINTS_DIR, "skeletonizer")


@dataclass(frozen=True)
class Calibration:
    """
    Physical calibration of the imaging system.

    The camera captures a fixed physical field of view (FOV). The pixel<->mm
    conversion is derived from the FOV and the *actual* image/mask resolution,
    NOT from any hardcoded pixel dimension — this is what makes the metrics
    resolution-independent (fixes F-002, where a non-640x480 mask silently
    reported an area wrong by the square of the resolution ratio).
    """

    fov_width_mm: float = 18.0
    fov_height_mm: float = 13.0

    def mm_per_px(self, width_px: int, height_px: int) -> tuple[float, float]:
        """Return (x, y) mm-per-pixel for an image of the given resolution."""
        return self.fov_width_mm / float(width_px), self.fov_height_mm / float(height_px)

    def area_per_px_mm2(self, width_px: int, height_px: int) -> float:
        """Physical area represented by one pixel (mm^2) at the given resolution."""
        mx, my = self.mm_per_px(width_px, height_px)
        return mx * my

    def mean_mm_per_px(self, width_px: int, height_px: int) -> float:
        """Mean isotropic mm-per-pixel (the imaging system is near-isotropic:
        18/640 vs 13/480 differ by < 4%)."""
        mx, my = self.mm_per_px(width_px, height_px)
        return (mx + my) / 2.0


CALIBRATION = Calibration()

# --- Visualization servers --------------------------------------------------
DASH_LENGTH_PORT = 8050
DASH_AREA_PORT = 8051

# --- Inference / batching ---------------------------------------------------
MASK_BATCH_SIZE = 16
SKELETON_BATCH_SIZE = 64
MASK_THRESHOLD = 0.5  # sigmoid threshold for binarizing the mask model output

# --- Output filenames / CSV schema -----------------------------------------
ROOT_LENGTHS_CSV = "root_lengths.csv"
ROOT_AREAS_CSV = "root_areas.csv"
LENGTH_CSV_HEADERS = ["Image", "Tube", "Position", "Date", "Time", "Length (mm)", "Error"]
AREA_CSV_HEADERS = ["Image", "Tube", "Position", "Date", "Time", "Area (mm²)", "Error"]
