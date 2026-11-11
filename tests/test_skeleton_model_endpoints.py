"""The skeleton model's endpoints (``SkeletonCorrectionModel.endpoints`` / ``endpoint_array``).

The tips are the set pixels with exactly one set 8-neighbour, as (x, y), row-major: the
markers Connect grabs. They equal ``vectorize_skeleton``'s endpoints, come from one filter
pass (never the full vectorization), are cached until the next edit, and are recomputed only
around the pixels that changed (S2): after every step of seeded random edit sequences (the
tools' calls, undo / redo, writes in place, a replaced mask) they equal a whole-frame
recomputation, set and order.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

import app.gui.skeleton_graph_model as sgm
from app.gui.skeleton_graph_model import (
    SkeletonCorrectionModel,
    _compute_degree_map,
    _ensure_uint8_binary,
    vectorize_skeleton,
)


def _model_with_line(size=(60, 40)) -> SkeletonCorrectionModel:
    model = SkeletonCorrectionModel()
    model.set_empty(size)
    model.draw_polyline([(5, 20), (50, 20)], thickness=1)
    model.mark_saved()
    return model


def _cross() -> np.ndarray:
    cr = np.zeros((11, 11), np.uint8)
    cr[5, 1:10] = 255
    cr[1:10, 5] = 255
    return cr


# ------------------------------------------------------------------- the tips themselves
def test_endpoints_of_the_cross_are_its_four_arm_tips():
    model = SkeletonCorrectionModel()
    model.mask = _cross()
    assert sorted(model.endpoints()) == [(1, 5), (5, 1), (5, 9), (9, 5)]


def test_endpoints_equal_the_vectorized_endpoints_on_random_skeletons():
    rng = np.random.default_rng(3)
    for _ in range(25):
        model = SkeletonCorrectionModel()
        model.set_empty((160, 120))
        for _ in range(int(rng.integers(1, 8))):
            x, y, ang = rng.uniform(0, 160), rng.uniform(0, 120), rng.uniform(0, 2 * math.pi)
            pts = []
            for _ in range(int(rng.integers(3, 20))):
                ang += rng.normal(0, 0.5)
                x, y = x + 6 * math.cos(ang), y + 6 * math.sin(ang)
                pts.append((int(x), int(y)))
            model.draw_polyline(pts, thickness=3)
        model.skeletonize()
        assert set(model.endpoints()) == set(vectorize_skeleton(model.mask).endpoints)


def test_endpoints_never_run_the_full_vectorization(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("endpoints() must not vectorize the skeleton")

    monkeypatch.setattr(sgm, "vectorize_skeleton", forbidden)
    model = _model_with_line()
    assert sorted(model.endpoints()) == [(5, 20), (50, 20)]


def test_endpoints_are_cached_until_the_next_edit():
    model = _model_with_line()
    first = model.endpoints()
    assert model.endpoints() is first
    model.push_undo()
    model.erase_circle((50, 20), 2)
    assert model.endpoints() is not first


# ---------------------------------------------------------------- windowed endpoints
def _full_endpoints(mask: np.ndarray) -> list[tuple[int, int]]:
    """The endpoints as computed before S2: the whole frame, row-major."""
    skel = _ensure_uint8_binary(mask) > 0
    ys, xs = np.nonzero(skel & (_compute_degree_map(skel) == 1))
    return list(zip(xs.tolist(), ys.tolist(), strict=True))


def _walk(rng, w, h, n, step):
    x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
    points = [(x, y)]
    for _ in range(n - 1):
        x = int(np.clip(x + rng.integers(-step, step + 1), 0, w - 1))
        y = int(np.clip(y + rng.integers(-step, step + 1), 0, h - 1))
        points.append((x, y))
    return points


def _random_step(model: SkeletonCorrectionModel, rng) -> None:
    h, w = model.mask.shape
    roll = rng.random()
    if roll < 0.45:  # a tool action
        model.push_undo()
        if rng.random() < 0.15:  # the legacy protocol: a write in place right after
            x, y = int(rng.integers(0, w - 6)), int(rng.integers(0, h - 6))
            model.mask[y : y + 3, x : x + 6] = 255
        kind = int(rng.integers(0, 4))
        if kind == 0:
            model.draw_polyline(_walk(rng, w, h, int(rng.integers(2, 7)), 40), thickness=3)
        elif kind == 1:
            pts = _walk(rng, w, h, 4, 25)
            for p0, p1 in zip(pts, pts[1:], strict=False):
                model.erase_segment(p0, p1, int(rng.integers(1, 9)))
        elif kind == 2:
            model.erase_polyline(_walk(rng, w, h, 3, 30), 9)
        else:
            model.draw_polyline(_walk(rng, w, h, 2, 60), thickness=1)
        if rng.random() < 0.8:
            model.skeletonize()
    elif roll < 0.65:
        model.undo()
    elif roll < 0.80:
        model.redo()
    elif roll < 0.86:
        model.erase_circle(_walk(rng, w, h, 1, 0)[0], int(rng.integers(1, 6)))  # no push
    elif roll < 0.90:
        model.push_undo()
        model.clear()
    elif roll < 0.93:
        model.mask = np.ascontiguousarray(model.mask.copy())  # the array replaced
    else:
        model.skeletonize()


@pytest.mark.parametrize(("seed", "size"), [(1, (160, 120)), (2, (320, 240)), (3, (641, 479))])
def test_windowed_endpoints_equal_a_whole_frame_recompute(seed, size):
    rng = np.random.default_rng(seed)
    model = SkeletonCorrectionModel()
    w, h = size
    start = np.zeros((h, w), np.uint8)
    for _ in range(30):
        pts = np.array(_walk(rng, w, h, 6, 30), np.int32).reshape(-1, 1, 2)
        cv2.polylines(start, [pts], False, 255, 3)
    model.load_from_raster(start, target_size=size)
    for step in range(250):
        _random_step(model, rng)
        assert model.endpoints() == _full_endpoints(model.mask), (seed, step)
        xy = model.endpoint_array()
        assert xy.shape == (len(model.endpoints()), 2)
        assert [tuple(p) for p in xy.tolist()] == model.endpoints()


def test_endpoints_of_a_one_pixel_and_an_empty_image():
    model = SkeletonCorrectionModel()
    model.set_empty((1, 1))
    assert model.endpoints() == []
    model.push_undo()
    model.mask[0, 0] = 255
    assert model.endpoints() == []  # an isolated pixel has no neighbour: not a tip
    model.set_empty((2, 1))
    model.push_undo()
    model.draw_polyline([(0, 0), (1, 0)], thickness=1)
    assert model.endpoints() == [(0, 0), (1, 0)]
    assert model.undo() and model.endpoints() == []


def test_a_small_edit_recomputes_only_its_window(monkeypatch):
    import app.gui.skeleton_graph_model as sgm

    model = SkeletonCorrectionModel()
    model.set_empty((2048, 1536))
    model.push_undo()
    for y in range(20, 1500, 40):
        model.draw_polyline([(10, y), (2000, y + 10)], thickness=1)
    model.skeletonize()  # a known fixed point: the next action thins a window
    model.endpoints()
    boxes = []
    real = sgm._tip_indices

    def spy(mask, box=None):
        boxes.append(box)
        return real(mask, box)

    monkeypatch.setattr(sgm, "_tip_indices", spy)
    model.push_undo()
    model.draw_polyline([(1000, 700), (1040, 720)], thickness=3)
    model.skeletonize()
    assert model.endpoints() == _full_endpoints(model.mask)
    assert len(boxes) == 1 and boxes[0] is not None
    x0, y0, x1, y1 = boxes[0]
    assert (x1 - x0) * (y1 - y0) < 100 * 100  # the edit's neighbourhood, not the frame
