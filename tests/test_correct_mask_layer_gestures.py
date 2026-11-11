"""MaskLayer: the mid-gesture rule (C2). A stroke is a gesture until its release or a
forced finish (Esc, a tool or layer switch, PgDn, a document action); a forced finish
commits exactly what the release would, and the rest of that drag changes nothing and
reaches no tool. Oracle: a direct ``PaintCanvas`` replay (``correct_mask_layer_support``).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QRectF

from app.gui.correct import mask_tools
from app.gui.correct.layer import (
    SaveOutcome,
)
from app.masks import write_mask_file
from tests.correct_mask_layer_support import (
    NO_MODS,
    H,
    Replay,
    W,
    boxes_plane,
    canvas_shows_the_plane,
    count,
    gt_file,
    image_path,
    open_layer,
    plane_of,
    read_rgba,
    ring_rect,
    stroke,
)
from tests.correct_mask_layer_support import scene as scene  # fixture

pytestmark = pytest.mark.usefixtures("no_prompts")


# --------------------------------------------------------------------------- gestures
def test_a_stroke_is_a_gesture_until_finish_gesture_commits_it(scene, tmp_path):
    layer, host = open_layer(scene, tmp_path)
    assert layer.gesture_active() is False

    layer.press(QPoint(10, 10), NO_MODS)
    assert layer.gesture_active() is True
    layer.move(QPoint(50, 30), True)
    layer.finish_gesture()

    assert layer.gesture_active() is False
    assert count(host, "layer_edited") == 1
    oracle = Replay(np.zeros((H, W), np.uint8))
    oracle.stroke([(10, 10), (50, 30)], 8)
    assert np.array_equal(plane_of(layer), oracle.plane)


def test_a_fill_press_is_no_gesture(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path, boxes_plane())
    layer.set_tool("fill")

    layer.press(QPoint(150, 110), NO_MODS)

    assert layer.gesture_active() is False


TRANSITIONS = {
    "esc": lambda layer: layer.cancel_transient(),
    "fill": lambda layer: layer.set_tool("fill"),
    "eraser": lambda layer: layer.set_tool("eraser"),
    "same_tool": lambda layer: layer.set_tool("brush"),  # a B tap mid-stroke
    "layer_switch": lambda layer: (layer.finish_gesture(), layer.set_active(False)),
    "made_reference": lambda layer: layer.set_active(False),
    "pgdn": lambda layer: (layer.finish_gesture(), layer.save()),
}


@pytest.mark.parametrize("transition", list(TRANSITIONS))
def test_a_transition_mid_stroke_commits_it_and_the_rest_of_the_drag_changes_nothing(
    scene, tmp_path, transition
):
    start = boxes_plane()
    layer, host = open_layer(scene, tmp_path, start)
    layer.set_size(7)
    host.calls.clear()

    layer.press(QPoint(20, 90), NO_MODS)
    layer.move(QPoint(40, 60), True)
    layer.move(QPoint(70, 50), True)  # into the first box
    TRANSITIONS[transition](layer)
    assert layer.gesture_active() is False
    calls_after_transition = list(host.calls)
    layer.move(QPoint(90, 60), True)  # across the box: an eraser would change it
    layer.move(QPoint(130, 100), True)
    layer.release(QPoint(130, 100))

    assert host.calls == calls_after_transition  # the rest of the drag did nothing
    assert count(host, "layer_edited") == 1  # the forced finish
    oracle = Replay(start)
    oracle.stroke([(20, 90), (40, 60), (70, 50)], 7)
    assert np.array_equal(plane_of(layer), oracle.plane)
    assert canvas_shows_the_plane(layer)  # the canvas holds nothing the plane lacks

    outcome = layer.save()
    assert outcome.ok
    replay_file = os.path.join(str(tmp_path), "replay.png")
    write_mask_file(replay_file, oracle.plane > 0)
    assert np.array_equal(read_rgba(gt_file(tmp_path)), read_rgba(replay_file))

    ring_follows = transition in ("esc", "eraser", "same_tool", "pgdn")
    assert layer.ring.is_visible() is ring_follows
    if ring_follows:  # only the ring moved: it sits on the pointer's last pixel
        assert ring_rect(layer).center() == QRectF(130, 100, 1, 1).center()


def _recording(cls, log):
    """``cls`` (a real mask tool) that also logs the drag input it receives."""

    class Recording(cls):
        def move(self, pos):
            log.append((self.tool_id, "move"))
            super().move(pos)

        def release(self, pos):
            log.append((self.tool_id, "release"))
            super().release(pos)

    return Recording


def test_a_drag_reaches_only_the_tool_that_began_it_and_nothing_after_a_finish(
    scene, tmp_path, monkeypatch
):
    log: list[tuple[str, str]] = []
    tools = tuple(_recording(cls, log) for cls in mask_tools.TOOL_CLASSES)
    monkeypatch.setattr(mask_tools, "TOOL_CLASSES", tools)
    layer, _ = open_layer(scene, tmp_path)

    stroke(layer, [(10, 10), (20, 10)])
    assert log == [("brush", "move"), ("brush", "release")]

    log.clear()
    layer.press(QPoint(10, 40), NO_MODS)
    layer.move(QPoint(20, 40), True)
    layer.set_tool("eraser")  # a forced finish
    layer.move(QPoint(30, 40), True)
    layer.release(QPoint(30, 40))
    assert log == [("brush", "move")]  # neither the eraser nor the brush saw the rest

    log.clear()
    stroke(layer, [(10, 70), (20, 70)])
    assert log == [("eraser", "move"), ("eraser", "release")]


@pytest.mark.parametrize("action", ["save", "revert", "close", "open", "hide", "clear", "undo"])
def test_document_actions_commit_a_stroke_in_progress_first(scene, tmp_path, action):
    start = boxes_plane()
    layer, host = open_layer(scene, tmp_path, start)
    layer.press(QPoint(10, 100), NO_MODS)
    layer.move(QPoint(60, 110), True)
    oracle = Replay(start)
    oracle.stroke([(10, 100), (60, 110)], 8)

    if action == "save":
        outcome = layer.save()
        saved = read_rgba(gt_file(tmp_path))[:, :, 3] == 255
        assert np.array_equal(saved, oracle.plane > 0)
        assert outcome == SaveOutcome("mask", ok=True, path=outcome.path)
        assert layer.is_dirty() is False
    elif action == "revert":
        assert layer.revert() == []
        assert np.array_equal(plane_of(layer), start)
        assert (layer.can_undo(), layer.can_redo()) == (False, False)
    elif action == "close":
        layer.close()
        assert layer.is_open() is False
    elif action == "open":
        other = image_path(tmp_path, "other_T001_L001_2024.07.18_111530_001_SMM")
        layer.open(other, (W, H), None)
        assert not plane_of(layer).any()
        assert (layer.can_undo(), layer.is_dirty()) == (False, False)
    elif action == "hide":
        layer.set_visible(False)
        assert np.array_equal(plane_of(layer), oracle.plane)
    elif action == "clear":
        layer.clear()
        layer.undo()  # back to the committed stroke
        assert np.array_equal(plane_of(layer), oracle.plane)
    else:  # undo: the stroke becomes an action, which is the one undone
        assert layer.undo() is True
        assert np.array_equal(plane_of(layer), start)

    assert layer.gesture_active() is False
    assert host.calls.index(("layer_edited", "mask")) == 0  # the stroke was committed first
    if layer.is_open():
        assert canvas_shows_the_plane(layer)
