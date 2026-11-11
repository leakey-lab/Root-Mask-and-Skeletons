"""Polyline and eraser gestures of the skeleton tools (``app/gui/skeleton_tools.py``).

Through a real ``SkeletonLayer`` (``tests/skeleton_edit_support.py``):

- Ctrl+click on a segment of the draft adds a bend there: the point goes between that
  segment's two ends, and Enter draws the line through it;
- Shift+drag on a segment moves the whole line by the drag, each point clamped to the image;
- Esc during an eraser drag keeps the erasure as one undo step, thins it and ends the drag:
  a later hover erases nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt

from tests.skeleton_edit_support import H, W, click, edits, key, lines, open_layer, thin

pytestmark = pytest.mark.usefixtures("app", "no_prompts")

CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
BACKGROUND = [(100, 420), (500, 420)]  # a skeleton to open, away from the drawn lines


def _draft(layer, *points) -> None:
    layer.set_tool("polyline")
    for x, y in points:
        click(layer, x, y)


def test_a_ctrl_click_on_a_segment_adds_a_bend_between_its_ends(tmp_path):
    layer, host = open_layer(tmp_path, lines(BACKGROUND))
    _draft(layer, (100, 100), (300, 100), (300, 300))

    layer.press(QPoint(200, 106), CTRL)  # 6 px off the first segment, far from its points
    layer.release(QPoint(200, 106))
    layer.press(QPoint(305, 200), CTRL)  # off the second segment
    layer.release(QPoint(305, 200))

    assert layer.draft.xy() == [(100, 100), (200, 106), (300, 100), (305, 200), (300, 300)]
    assert key(layer, Qt.Key.Key_Return) is True
    mask = layer.model.mask
    assert edits(host) == 1
    assert mask[104:109, 198:203].any(), "the line runs through the bend"
    assert not mask[98:103, 198:203].any(), "not along the old straight segment"
    layer.cancel_scheduled_render()


def test_a_shift_drag_moves_the_whole_line_and_clamps_it_to_the_image(tmp_path):
    layer, _host = open_layer(tmp_path, lines(BACKGROUND))
    _draft(layer, (100, 100), (300, 100), (300, 300))

    layer.press(QPoint(200, 102), SHIFT)  # on the first segment
    layer.move(QPoint(230, 142), True)
    layer.release(QPoint(230, 142))

    assert layer.draft.xy() == [(130, 140), (330, 140), (330, 340)]
    assert not layer.gesture_active()

    layer.press(QPoint(230, 142), SHIFT)  # a second move, far past the right edge
    layer.move(QPoint(630, 142), True)
    layer.release(QPoint(630, 142))

    assert layer.draft.xy() == [(530, 140), (W - 1, 140), (W - 1, 340)]
    layer.cancel_scheduled_render()


def test_esc_during_an_eraser_drag_keeps_the_erasure_thinned_and_ends_the_drag(tmp_path):
    skeleton = lines([(100, 240), (539, 240)])
    layer, host = open_layer(tmp_path, skeleton)
    layer.set_tool("eraser")
    layer.set_size(6)
    layer.press(QPoint(200, 240), Qt.KeyboardModifier.NoModifier)
    layer.move(QPoint(230, 240), True)

    layer.cancel_transient()  # Esc
    layer.move(QPoint(400, 240), False)  # a hover after the drag

    mask = layer.model.mask
    assert not layer.gesture_active()
    assert not mask[238:243, 200:231].any(), "the erasure is kept"
    assert mask[240, 380:421].all(), "a hover after Esc erases nothing"
    assert np.array_equal(mask, thin(mask)), "the skeleton is thinned"
    assert mask.shape == (H, W) and edits(host) == 1
    assert layer.undo() is True
    assert np.array_equal(layer.model.mask, skeleton), "the drag undoes as one step"
    layer.cancel_scheduled_render()
