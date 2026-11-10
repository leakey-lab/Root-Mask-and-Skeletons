"""The skeleton corpus of the ROI-skeletonize tests: three skeletons built in memory, a model
that opens one, and the skeleton tools' model calls as seeded random edits.

Moved here from ``test_skeleton_roi.py``; ``test_skeleton_graph_model_render.py`` replays the
same edits. ``_full_frame`` is scikit-image on the whole frame, bound when this module is
imported (before any test patches ``sgm._skeletonize_uint8``).
"""

from __future__ import annotations

import weakref
from itertools import pairwise

import cv2
import numpy as np

import app.gui.skeleton_graph_model as sgm
from app.gui.skeleton_graph_model import SkeletonCorrectionModel

_full_frame = sgm._skeletonize_uint8  # the reference: scikit-image on the whole frame


# --------------------------------------------------------------------------------------
# Cases (deterministic, built in memory)
# --------------------------------------------------------------------------------------


def _synthetic_640() -> np.ndarray:
    """The committed golden case's generated skeleton (341x256, grey 1-3 px lines) opened
    at 640x480, as the editor opens it."""
    raster = np.zeros((256, 341), np.uint8)
    pts = np.array([[20, 30], [120, 90], [200, 80], [320, 200]], np.int32)
    cv2.polylines(raster, [pts.reshape(-1, 1, 2)], False, 191, 2)
    cv2.line(raster, (60, 240), (300, 20), 191, 1)
    cv2.line(raster, (150, 150), (150, 250), 191, 3)
    model = SkeletonCorrectionModel()
    model.load_from_raster(raster, target_size=(640, 480))
    return model.mask.copy()


def _dense_640() -> np.ndarray:
    """A dense 640x480 skeleton: 260 seeded random walks, 3 px wide, thinned."""
    rng = np.random.default_rng(20260929)
    canvas = np.zeros((480, 640), np.uint8)
    for _ in range(260):
        x, y = rng.uniform(0, 640), rng.uniform(0, 480)
        angle = rng.uniform(0, 2 * np.pi)
        pts = []
        for _ in range(int(rng.integers(4, 16))):
            angle += rng.normal(0, 0.6)
            x = float(np.clip(x + 14 * np.cos(angle), 0, 639))
            y = float(np.clip(y + 14 * np.sin(angle), 0, 479))
            pts.append((int(x), int(y)))
        cv2.polylines(canvas, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, 255, 3)
    return _full_frame(canvas)


def _upscale_2048() -> np.ndarray:
    """The dense skeleton opened on a 2048x1536 image (nearest upscale, then thinned)."""
    model = SkeletonCorrectionModel()
    model.load_from_raster(_dense_640(), target_size=(2048, 1536))
    return model.mask.copy()


CASES = {"synthetic 640": _synthetic_640, "dense 640": _dense_640, "upscale 2048": _upscale_2048}
SEEDS = {"synthetic 640": 11, "dense 640": 12, "upscale 2048": 13}


def _opened(base: np.ndarray) -> SkeletonCorrectionModel:
    """A model that opened ``base`` like a skeleton file of the image's size."""
    model = SkeletonCorrectionModel()
    model.load_from_raster(base, target_size=(base.shape[1], base.shape[0]))
    assert np.array_equal(model.mask, base)
    return model


# --------------------------------------------------------------------------------------
# Editor-like random edits (the skeleton tools' model calls)
# --------------------------------------------------------------------------------------


def _walk(rng, w: int, h: int, n: int, step: int) -> list[tuple[int, int]]:
    x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
    pts = [(x, y)]
    for _ in range(n - 1):
        x = int(np.clip(x + rng.integers(-step, step + 1), 0, w - 1))
        y = int(np.clip(y + rng.integers(-step, step + 1), 0, h - 1))
        pts.append((x, y))
    return pts


_BRANCHES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _branch(model: SkeletonCorrectionModel, rng) -> list[tuple[int, int]]:
    """A branch's own pixels, as Select hands them to the tools (S03-01). Taken from the
    topology of the skeleton as opened, once per model: later edits may have changed some
    of those pixels, which still makes an erase of skeleton pixels along a branch."""
    branches = _BRANCHES.get(model)
    if branches is None:
        branches = _BRANCHES[model] = [own for own in model.topology().own if own]
    return branches[int(rng.integers(0, len(branches)))]


def _polyline(model: SkeletonCorrectionModel, rng) -> None:
    """PolylineTool.commit: a new line, or a picked branch redrawn."""
    h, w = model.mask.shape
    model.push_undo()
    if rng.random() < 0.3:
        model.erase_pixels(_branch(model, rng))
    model.draw_polyline(_walk(rng, w, h, int(rng.integers(2, 9)), 60), thickness=3)


def _eraser(model: SkeletonCorrectionModel, rng) -> None:
    """EraserTool: press, then a segment per mouse move (the release thins)."""
    h, w = model.mask.shape
    radius = int(rng.integers(1, 21))
    pts = _walk(rng, w, h, int(rng.integers(1, 13)), 25)
    model.push_undo()
    model.erase_segment(pts[0], pts[0], radius)
    for p0, p1 in pairwise(pts):
        model.erase_segment(p0, p1, radius)


def _connect(model: SkeletonCorrectionModel, rng) -> None:
    """ConnectTool.release: a 3 px line from an endpoint to a nearby point."""
    h, w = model.mask.shape
    ends = model.endpoints()
    if ends:
        x, y = ends[int(rng.integers(0, len(ends)))]
    else:
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
    to = (
        int(np.clip(x + rng.integers(-40, 41), 0, w - 1)),
        int(np.clip(y + rng.integers(-40, 41), 0, h - 1)),
    )
    model.push_undo()
    model.draw_polyline([(x, y), to], thickness=3)


def _delete_branch(model: SkeletonCorrectionModel, rng) -> None:
    """PolylineTool.delete_branch: erase a branch's own pixels."""
    model.push_undo()
    model.erase_pixels(_branch(model, rng))


EDITS = (_polyline, _eraser, _connect, _delete_branch)
