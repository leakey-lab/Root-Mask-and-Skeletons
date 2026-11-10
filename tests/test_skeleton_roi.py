"""ROI skeletonize: thinning only around an edit gives exactly the full-frame result.

``SkeletonCorrectionModel.skeletonize()`` thins a window around the pixels that changed
since a known skeletonize fixed point (the snapshot of the current action) instead of
the whole frame (plan step P5a-3, spec D24). These tests replay seeded random edits the
way the skeleton tools make them (Polyline commit, eraser drag, Connect, branch delete,
undo / redo) on three skeletons and compare every result with scikit-image run on the
whole edited frame: there must be 0 mismatches, and the window path must be the rule.

Always on: about 50 edits per case. With ``SPROUTS_PERF_TESTS=1``: 300 edits per case
and a 2,800-edit run over the three cases.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import app.gui.skeleton_graph_model as sgm
from app.gui.skeleton_graph_model import SkeletonCorrectionModel
from tests.skeleton_corpus_support import (
    CASES,
    EDITS,
    SEEDS,
    _full_frame,
    _opened,
)

PERF = os.environ.get("SPROUTS_PERF_TESTS") == "1"


@pytest.fixture(scope="module")
def bases() -> dict[str, np.ndarray]:
    return {name: build() for name, build in CASES.items()}


@pytest.fixture
def thinned(monkeypatch) -> list[tuple[int, int]]:
    """Shapes the model hands to scikit-image (a window, or the whole frame)."""
    shapes: list[tuple[int, int]] = []

    def spy(mask: np.ndarray) -> np.ndarray:
        shapes.append(mask.shape)
        return _full_frame(mask)

    monkeypatch.setattr(sgm, "_skeletonize_uint8", spy)
    return shapes


def test_the_cases_are_real_skeletons(bases):
    assert int((bases["synthetic 640"] > 0).sum()) > 1000
    assert int((bases["dense 640"] > 0).sum()) >= 15_000
    assert bases["upscale 2048"].shape == (1536, 2048)
    for base in bases.values():
        assert np.array_equal(_full_frame(base), base)  # thinning a skeleton changes nothing


def _replay(model: SkeletonCorrectionModel, seed: int, edits: int, thinned: list) -> dict:
    """Run ``edits`` random actions; compare each thinning with the full-frame one."""
    rng = np.random.default_rng(seed)
    frame = model.mask.shape
    stats = {"edits": 0, "mismatches": [], "window": 0, "nothing": 0, "frame": 0}
    step = 0
    while stats["edits"] < edits:
        step += 1
        roll = rng.random()
        if roll < 0.06 and model.can_undo:
            model.undo()
            continue
        if roll < 0.09 and model.can_redo:
            model.redo()
            continue
        EDITS[step % len(EDITS)](model, rng)
        edited = model.mask.copy()
        thinned.clear()
        model.skeletonize()
        if not np.array_equal(model.mask, _full_frame(edited)):
            stats["mismatches"].append(step)
        if not thinned:
            stats["nothing"] += 1  # the edit changed no pixel: nothing to thin
        elif frame in thinned:
            stats["frame"] += 1
        else:
            stats["window"] += 1
        stats["edits"] += 1
    return stats


def _assert_exact_and_local(stats: dict) -> None:
    assert stats["mismatches"] == [], stats
    # The window is the rule: the whole frame is thinned for at most 1 edit in 10.
    assert stats["frame"] <= stats["edits"] // 10, stats


@pytest.mark.parametrize("case", sorted(CASES))
def test_window_skeletonize_equals_the_full_frame_after_random_edits(case, bases, thinned):
    stats = _replay(_opened(bases[case]), SEEDS[case], 50, thinned)
    _assert_exact_and_local(stats)


@pytest.mark.skipif(not PERF, reason="long run: set SPROUTS_PERF_TESTS=1")
@pytest.mark.parametrize("case", sorted(CASES))
def test_window_skeletonize_300_random_edits_per_case(case, bases, thinned):
    stats = _replay(_opened(bases[case]), 100 + SEEDS[case], 300, thinned)
    _assert_exact_and_local(stats)


@pytest.mark.skipif(not PERF, reason="long run: set SPROUTS_PERF_TESTS=1")
def test_window_skeletonize_2800_random_edits_over_the_three_cases(bases, thinned):
    runs = {"synthetic 640": 1100, "dense 640": 1100, "upscale 2048": 600}
    for case, edits in runs.items():
        stats = _replay(_opened(bases[case]), 1000 + SEEDS[case], edits, thinned)
        _assert_exact_and_local(stats)


# --------------------------------------------------------------------------------------
# When the window is (not) used
# --------------------------------------------------------------------------------------


def test_a_base_that_is_not_a_fixed_point_takes_the_full_frame_path(thinned):
    model = SkeletonCorrectionModel()
    model.set_empty((120, 90))
    block = np.zeros((90, 120), np.uint8)
    block[20:60, 20:100] = 255  # a solid block: thinning changes it everywhere
    model.mask = block
    model.push_undo()
    model.draw_polyline([(5, 80), (110, 85)], thickness=3)
    edited = model.mask.copy()
    thinned.clear()
    model.skeletonize()
    assert thinned == [(90, 120)]
    assert np.array_equal(model.mask, _full_frame(edited))


def test_without_a_snapshot_the_whole_frame_is_thinned(bases, thinned):
    model = _opened(bases["synthetic 640"])
    model.draw_polyline([(100, 400), (300, 380)], thickness=3)  # no push_undo()
    edited = model.mask.copy()
    thinned.clear()
    model.skeletonize()
    assert thinned == [(480, 640)]
    assert np.array_equal(model.mask, _full_frame(edited))


def test_a_small_edit_is_thinned_in_a_small_window(bases, thinned):
    model = _opened(bases["upscale 2048"])
    model.push_undo()
    model.draw_polyline([(1000, 700), (1040, 720)], thickness=3)
    edited = model.mask.copy()
    thinned.clear()
    model.skeletonize()
    assert len(thinned) == 1
    h, w = thinned[0]
    assert h <= 21 + 5 + 32 and w <= 41 + 5 + 32  # the edit box + 16 px on each side
    assert np.array_equal(model.mask, _full_frame(edited))


def test_an_edit_that_changes_no_pixel_thins_nothing(bases, thinned):
    model = _opened(bases["synthetic 640"])
    before = model.mask.copy()
    model.push_undo()
    model.erase_circle((630, 10), 4)  # empty background
    thinned.clear()
    model.skeletonize()
    assert thinned == []
    assert np.array_equal(model.mask, before)


def test_an_in_place_write_right_after_push_undo_stays_exact(bases, thinned):
    # The legacy protocol: push_undo(), then write model.mask in place.
    model = _opened(bases["dense 640"])
    model.push_undo()
    model.mask[200:230, 300:360] = 255
    model.mask[0:12, 0:40] = 1  # any non-zero value is foreground
    edited = model.mask.copy()
    model.skeletonize()
    assert np.array_equal(model.mask, _full_frame(edited))
    assert model.mask.dtype == np.uint8 and set(np.unique(model.mask).tolist()) <= {0, 255}


def test_an_in_place_write_pushed_again_before_thinning_stays_exact(bases, thinned):
    model = _opened(bases["dense 640"])
    model.push_undo()
    model.mask[100:160, 100:220] = 255  # in place, right after push_undo(): a thick block
    model.push_undo()  # the next action starts before anything thinned the block
    model.draw_polyline([(300, 300), (340, 320)], thickness=3)
    edited = model.mask.copy()
    model.skeletonize()
    assert np.array_equal(model.mask, _full_frame(edited))


def test_edits_at_the_image_border_stay_exact(bases, thinned):
    model = _opened(bases["dense 640"])
    edits = [
        lambda m: m.erase_circle((0, 0), 15),
        lambda m: m.draw_polyline([(0, 479), (639, 470)], thickness=3),
        lambda m: m.erase_segment((639, 0), (639, 479), 6),
        lambda m: m.draw_polyline([(630, 5), (639, 0), (600, 0)], thickness=9),
    ]
    for edit in edits:
        model.push_undo()
        edit(model)
        edited = model.mask.copy()
        model.skeletonize()
        assert np.array_equal(model.mask, _full_frame(edited))


def test_undo_redo_then_edit_stays_exact_on_the_window_path(bases, thinned):
    model = _opened(bases["dense 640"])
    for x in (100, 300, 500):
        model.push_undo()
        model.draw_polyline([(x, 100), (x + 40, 300)], thickness=3)
        model.skeletonize()
    model.undo()
    model.undo()
    model.redo()
    model.push_undo()
    model.draw_polyline([(50, 400), (400, 420)], thickness=3)
    edited = model.mask.copy()
    thinned.clear()
    model.skeletonize()
    assert thinned and (480, 640) not in thinned
    assert np.array_equal(model.mask, _full_frame(edited))


def test_the_changed_box_covers_every_pixel_the_window_thinning_changed(bases):
    model = _opened(bases["synthetic 640"])
    model.push_undo()
    model.draw_polyline([(100, 300), (180, 330), (260, 310)], thickness=9)
    model.take_changed_box()  # the drawn stroke, reported to the display already
    before = model.mask.copy()
    model.skeletonize()
    x, y, w, h = model.take_changed_box()
    ys, xs = np.nonzero(before != model.mask)
    assert xs.size > 0
    assert x <= xs.min() and y <= ys.min() and x + w > xs.max() and y + h > ys.max()
    assert w <= 260 - 100 + 2 * 12 and h <= 330 - 300 + 2 * 12  # local to the stroke
