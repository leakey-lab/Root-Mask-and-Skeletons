"""
Baseline regression suite for Root-Mask-and-Skeletons.

These tests PIN the *current* behavior of the scientifically load-bearing code
paths so that the refactor/upgrade work (deps, memory, performance, dedup, OOP)
can proceed without silently changing measured outputs.

IMPORTANT: The calibration constants encoded as golden values below reflect the
behavior at the start of the hardening branch. The R5 scientific-correctness fix
(arc-length calibration) will INTENTIONALLY change the length golden values; when
that happens, update LEN_PER_PX here together with a documented justification in
REMEDIATION_REPORT.md. Until then, every other phase must keep these green.
"""
import csv
import os

import numpy as np
import pytest

from app.inference.root_length_inference_handler import (
    calculate_root_length,
    parse_image_name,
    process_single_image,
)
from app.inference.root_area_inference_handler import calculate_root_area
from app.gui.skeleton_graph_model import vectorize_skeleton, SkeletonCorrectionModel
from app.data_processing.data_processor import DataProcessor

# --- Golden constants captured from baseline (v3) ---
LEN_PER_PX = 0.05178309721543722   # mm per skeleton pixel
AREA_PER_PX = 0.000761718750         # mm^2 per root pixel (18/640 * 13/480)


def _row_skeleton(n_px, h=20, w=120):
    sk = np.zeros((h, w), dtype=np.uint8)
    sk[h // 2, 0:n_px] = 255
    return sk


# ---------------- Length ----------------

@pytest.mark.parametrize("n_px", [1, 10, 25, 100])
def test_calculate_root_length_scales_with_pixel_count(n_px):
    sk = _row_skeleton(n_px)
    assert calculate_root_length(sk) == pytest.approx(n_px * LEN_PER_PX, rel=1e-9)


def test_calculate_root_length_empty_is_zero():
    assert calculate_root_length(np.zeros((10, 10), dtype=np.uint8)) == 0.0


# ---------------- Area ----------------

@pytest.mark.parametrize("n_px", [1, 100, 200, 1000])
def test_calculate_root_area_scales_with_pixel_count(n_px):
    m = np.zeros((40, 40), dtype=np.uint8)
    flat = m.ravel()
    flat[:n_px] = 255
    m = flat.reshape(40, 40)
    assert calculate_root_area(m) == pytest.approx(n_px * AREA_PER_PX, rel=1e-9)


# ---------------- Filename parsing ----------------

def test_parse_image_name_full():
    info = parse_image_name("scan_T005_L030_2024.03.15_143022_fake.png")
    assert info == {
        "original_name": "scan_T005_L030_2024.03.15_143022_fake.png",
        "tube_number": 5,
        "length_position": 30,
        "date": "2024.03.15",
        "time": "14:30:22",
    }


def test_parse_image_name_missing_fields():
    info = parse_image_name("nothing_here.png")
    assert info["tube_number"] is None
    assert info["date"] is None


# ---------------- Skeleton vectorization topology ----------------

def test_vectorize_cross_topology_endpoints():
    cr = np.zeros((11, 11), dtype=np.uint8)
    cr[5, 1:10] = 255
    cr[1:10, 5] = 255
    topo = vectorize_skeleton(cr, simplify_epsilon=1.0)
    assert sorted(topo.endpoints) == [(1, 5), (5, 1), (5, 9), (9, 5)]
    assert len(topo.polylines) >= 4


def test_skeleton_model_undo_redo_roundtrip():
    model = SkeletonCorrectionModel()
    model.set_empty((30, 30))
    model.push_undo()
    model.draw_polyline([(2, 2), (20, 20)], thickness=2)
    drawn = int((model.mask == 255).sum())
    assert drawn > 0
    assert model.undo() is True
    assert int((model.mask == 255).sum()) == 0
    assert model.redo() is True
    assert int((model.mask == 255).sum()) == drawn


def test_undo_stack_is_bounded():
    model = SkeletonCorrectionModel()
    model.set_empty((10, 10))
    for _ in range(model.max_stack_size + 20):
        model.push_undo()
    assert len(model._undo_stack) <= model.max_stack_size


# ---------------- CSV schema (length pipeline) ----------------

def test_process_single_image_row_schema(tmp_path):
    import cv2
    sk = _row_skeleton(40)
    # process_single_image reads from disk; write a binary image it will threshold/skeletonize
    img = np.zeros((20, 120), dtype=np.uint8)
    img[10, 0:40] = 255
    p = tmp_path / "x_T1_L2_2024.01.02_010203_fake.png"
    cv2.imwrite(str(p), img)
    row = process_single_image(p.name, str(p))
    assert set(row.keys()) >= {"Image", "Tube", "Position", "Date", "Time", "Length (mm)"}
    assert row["Tube"] == 1 and row["Position"] == 2


# ---------------- DataProcessor ----------------

def test_data_processor_loads_and_filters(tmp_path):
    csv_path = tmp_path / "root_lengths.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Image", "Tube", "Position", "Date", "Time", "Length (mm)"])
        w.writeheader()
        w.writerow({"Image": "a", "Tube": 1, "Position": 10, "Date": "2024.01.01", "Time": "00:00:01", "Length (mm)": 5.5})
        w.writerow({"Image": "b", "Tube": 2, "Position": 20, "Date": "2024.01.02", "Time": "00:00:02", "Length (mm)": 7.0})
    dp = DataProcessor(str(csv_path))
    assert list(dp.get_unique_tubes()) == [1, 2]
    assert len(dp.get_unique_positions()) == 2


def test_data_processor_bad_path_returns_empty():
    dp = DataProcessor("does_not_exist_12345.csv")
    assert dp.df.empty
