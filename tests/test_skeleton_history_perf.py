"""Whole-frame skeleton edits at large sizes (KNOWN_ISSUES P3): record, undo and redo time.

The skeleton model (``app.gui.skeleton_graph_model``) records an action as the box it
changed, before and after, through ``app.gui.patch_history``: a 0/255 side is stored as its
packed bits, deflated, and the all-0 side of a Clear as nothing. The skeleton here is a dense
synthetic one over the whole frame (1 px random walks, about 2 % of the pixels: denser than
the real 640 x 480 data, so the boxes span the frame), and the edits are the two that
change the whole frame: Clear and a replaced skeleton (Load prediction, Load file). The
record is timed as the next action's ``push_undo()`` makes it, as on the page.

Strict budgets, only with ``SPROUTS_PERF_TESTS=1`` (medians of 7, this machine, about 1.5x
the medians measured alone): at 6000 x 4000 a Clear records in <= 33 ms, undoes in <= 24 ms
and redoes in <= 15 ms; a replaced skeleton records in <= 50 ms and undoes and redoes in
<= 24 ms. Measured 2026-10-29 with other jobs on the CPU: 21-22, 15-16, 8-9, 32-34 and
15-16 ms (31 and 43 ms for the records once, after the Correct perf tests in the same
process); byte for byte deflated (before) 32-37, 19-25, 10-13, 46-55 and 19-25 ms, with 25 %
more memory. At 2048 x 1536 the records take <= 5 ms (2.3-3.4; before 3.2-5.4) and the undos
and redos <= 3 ms (1.0-2.0; before 1.0-2.4). The 640 x 480 data takes under 0.5 ms either
way; ``tests/test_skeleton_patch_history.py`` pins the round trip and the stored size.
"""

from __future__ import annotations

import os
import statistics
import time

import cv2
import numpy as np
import pytest

from app.gui.skeleton_graph_model import SkeletonCorrectionModel

STRICT = os.environ.get("SPROUTS_PERF_TESTS") == "1"
pytestmark = [
    pytest.mark.wallclock,
    pytest.mark.skipif(not STRICT, reason="wall-clock budget: set SPROUTS_PERF_TESTS=1"),
]


def dense_skeleton(size: tuple[int, int], seed: int) -> np.ndarray:
    """1 px random walks over the whole frame, 0/255 (400 walks at 2048 x 1536, by area)."""
    w, h = size
    rng = np.random.default_rng(seed)
    canvas = np.zeros((h, w), np.uint8)
    for _ in range(max(20, 400 * w * h // (2048 * 1536))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        points = [(x, y)]
        for _ in range(7):
            x = int(np.clip(x + rng.integers(-40, 41), 0, w - 1))
            y = int(np.clip(y + rng.integers(-40, 41), 0, h - 1))
            points.append((x, y))
        cv2.polylines(canvas, [np.array(points, np.int32).reshape(-1, 1, 2)], False, 255, 1)
    return canvas


def timed(call) -> float:
    t0 = time.perf_counter()
    assert call() is not False
    return (time.perf_counter() - t0) * 1000


def whole_frame_edits(size: tuple[int, int], count: int = 7) -> dict[str, float]:
    """Median ms of recording, undoing and redoing a Clear and a replaced skeleton."""
    first, second = dense_skeleton(size, 1), dense_skeleton(size, 2)
    times: dict[str, list[float]] = {}
    for run in range(count + 1):  # the first run warms up
        model = SkeletonCorrectionModel()
        model.load_thinned(first.copy())
        model.push_undo()
        model.clear()
        record = timed(model.push_undo)  # records the Clear, opens an empty action
        model.undo()  # the empty action
        clear = {"clear record": record, "clear undo": timed(model.undo)}
        clear["clear redo"] = timed(model.redo)
        assert not model.mask.any()
        model.undo()
        assert np.array_equal(model.mask, first)
        model.push_undo()
        model.mask = second.copy()  # Load prediction / Load file
        record = timed(model.push_undo)
        model.undo()
        replace = {"replace record": record, "replace undo": timed(model.undo)}
        assert np.array_equal(model.mask, first)
        replace["replace redo"] = timed(model.redo)
        assert np.array_equal(model.mask, second)
        if run:
            for key, value in (clear | replace).items():
                times.setdefault(key, []).append(value)
    return {key: statistics.median(values) for key, values in times.items()}


def test_whole_frame_skeleton_edits_at_6000_x_4000():
    got = whole_frame_edits((6000, 4000))
    budget = {
        "clear record": 33,
        "clear undo": 24,
        "clear redo": 15,
        "replace record": 50,
        "replace undo": 24,
        "replace redo": 24,
    }
    assert {key: got[key] for key in budget if got[key] > budget[key]} == {}, got


def test_whole_frame_skeleton_edits_at_2048_x_1536():
    got = whole_frame_edits((2048, 1536))
    budget = {key: 5 if key.endswith("record") else 3 for key in got}
    assert {key: got[key] for key in budget if got[key] > budget[key]} == {}, got
