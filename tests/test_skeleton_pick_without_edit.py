"""A branch only picked, not changed, is left alone by Enter, Finish and a double-click (S03-02).

Select hands the picked branch to the Polyline tool. Enter, the prompt's Finish line and a
double-click used to commit it even when nothing moved: the branch was erased and redrawn
through its simplified points (straight segments), and a Select double-click also drew a
chord from the branch's end back to the click. The layer then read as edited, so paging on
saved it as the image's GT skeleton.

Now an unchanged pick changes no pixel and makes no undo step: Enter and Finish drop the
pick, a double-click keeps it open. A pick whose points moved still commits, and Enter or
Finish with Smooth on still applies the smoothed curve to a branch of 3+ points (the
documented pick, S, Enter).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt

from tests.skeleton_edit_support import (
    click,
    double_click,
    drag,
    edits,
    key,
    lines,
    open_layer,
    pick,
)

pytestmark = pytest.mark.usefixtures("app", "no_prompts")

STRAIGHT = [[(100, 240), (539, 240)]]
Y = [[(320, 240), (320, 80)], [(320, 240), (200, 360)], [(320, 240), (440, 360)]]
L = [[(100, 100), (300, 100), (300, 300)]]  # one branch of three points
CURVE = [[(100, 300), (180, 200), (260, 170), (340, 200), (420, 300)]]  # one bent branch
SHAPES = {
    "straight": (STRAIGHT, (300, 242)),
    "Y arm": (Y, (320, 160)),
    "curve": (CURVE, (260, 172)),
}


@pytest.fixture(params=sorted(SHAPES))
def picked(request, tmp_path):
    """A layer with a branch picked by Select and left as it was."""
    shape, at = SHAPES[request.param]
    skeleton = lines(*shape)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, *at)
    assert not layer.is_dirty()
    yield layer, host, skeleton
    layer.cancel_scheduled_render()


def _off_the_branch(mask, skeleton) -> bool:
    """Whether ``mask`` has pixels more than 1 px away from every pixel of ``skeleton``."""
    near = cv2.dilate(skeleton, np.ones((3, 3), np.uint8))
    return bool(((mask > 0) & (near == 0)).any())


def _unchanged(layer, host, skeleton) -> None:
    assert np.array_equal(layer.model.mask, skeleton), "the skeleton changed"
    assert not layer.is_dirty()
    assert not layer.can_undo() and edits(host) == 0, "an undo step was made"


@pytest.mark.parametrize("finish", ["enter", "keypad enter", "finish button"])
def test_enter_or_finish_on_a_plain_pick_changes_nothing_and_drops_the_pick(picked, finish):
    layer, host, skeleton = picked

    if finish == "finish button":
        layer.finish_polyline()
    else:
        qt_key = Qt.Key.Key_Return if finish == "enter" else Qt.Key.Key_Enter
        assert key(layer, qt_key) is True

    _unchanged(layer, host, skeleton)
    assert layer.draft.points == [] and layer.draft.edit_original is None


def test_a_select_double_click_on_a_branch_draws_no_chord_and_keeps_the_pick(tmp_path):
    for name, (shape, (x, y)) in SHAPES.items():
        skeleton = lines(*shape)
        layer, host = open_layer(tmp_path / name, skeleton)
        layer.set_tool("select")

        double_click(layer, x, y)  # the first press picks, the double-click reaches Polyline

        _unchanged(layer, host, skeleton)
        assert layer.draft.edit_original is not None, "the pick stays open"
        assert layer.current_tool == "polyline"
        layer.cancel_scheduled_render()


def test_a_double_click_on_a_handle_of_a_plain_pick_changes_nothing(picked):
    layer, host, skeleton = picked
    first = layer.draft.points[0]

    double_click(layer, first.x(), first.y())

    _unchanged(layer, host, skeleton)
    assert layer.draft.edit_original is not None


def test_with_smooth_on_before_the_pick_a_double_click_still_changes_nothing(tmp_path):
    skeleton = lines(*CURVE)
    layer, host = open_layer(tmp_path, skeleton)
    layer.set_tool("polyline")
    layer.set_smooth(True)  # Smooth is sticky: it stays on for the next pick
    layer.set_tool("select")

    double_click(layer, 260, 172)

    _unchanged(layer, host, skeleton)
    assert layer.draft.edit_original is not None
    layer.cancel_scheduled_render()


@pytest.mark.parametrize("finish", ["enter", "finish button"])
def test_pick_s_enter_still_applies_the_smoothed_branch(tmp_path, finish):
    skeleton = lines(*L)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, 200, 102)
    assert len(layer.draft.points) == 3
    assert key(layer, Qt.Key.Key_S) is True

    if finish == "enter":
        key(layer, Qt.Key.Key_Return)
    else:
        layer.finish_polyline()

    assert _off_the_branch(layer.model.mask, skeleton), "the curve was applied"
    assert edits(host) == 1 and layer.can_undo()
    assert layer.draft.points == []
    layer.cancel_scheduled_render()


def test_enter_with_smooth_already_on_still_applies_it_to_a_branch_of_three_points(tmp_path):
    skeleton = lines(*L)
    layer, host = open_layer(tmp_path, skeleton)
    layer.set_tool("polyline")
    layer.set_smooth(True)
    pick(layer, 200, 102)

    key(layer, Qt.Key.Key_Return)

    assert _off_the_branch(layer.model.mask, skeleton)
    assert edits(host) == 1
    layer.cancel_scheduled_render()


def test_smooth_does_not_redraw_a_plain_pick_of_two_points(tmp_path):
    skeleton = lines(*STRAIGHT)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, 300, 242)
    assert len(layer.draft.points) == 2
    key(layer, Qt.Key.Key_S)  # Smooth needs 3+ points: nothing to apply

    key(layer, Qt.Key.Key_Return)

    _unchanged(layer, host, skeleton)


def test_a_moved_pick_still_commits_on_enter(picked):
    layer, host, skeleton = picked
    first = layer.draft.points[0]
    drag(layer, (first.x(), first.y()), (first.x(), first.y() + 15))
    assert layer.is_dirty()

    key(layer, Qt.Key.Key_Return)

    assert not np.array_equal(layer.model.mask, skeleton)
    assert edits(host) == 1 and layer.draft.points == []


def test_a_moved_pick_commits_on_a_double_click_without_a_chord(tmp_path):
    skeleton = lines(*L)
    on_enter = []
    for finish in ("enter", "double-click"):
        layer, host = open_layer(tmp_path / finish, skeleton)
        pick(layer, 200, 102)
        corner = layer.draft.points[1]  # the L's corner, its middle point
        drag(layer, (corner.x(), corner.y()), (320, 80))
        if finish == "enter":
            key(layer, Qt.Key.Key_Return)
        else:
            first = layer.draft.points[0]  # a double-click on a handle: no new point
            double_click(layer, first.x(), first.y())
        assert edits(host) == 1 and layer.draft.points == []
        on_enter.append(layer.model.mask.copy())
        layer.cancel_scheduled_render()
    assert np.array_equal(on_enter[0], on_enter[1]), "the double-click added a chord"


# Off-handle points on each picked branch: a press there is no grab, so it adds a point.
OFF_HANDLE = {
    "straight": (STRAIGHT, (300, 242), (300, 240)),
    "Y arm": (Y, (320, 160), (320, 160)),
    "curve": (CURVE, (260, 172), (140, 250)),
    "L": (L, (200, 102), (200, 100)),
}


def _assert_off_the_handles(layer, x, y) -> None:
    from app.gui.skeleton_tools import HANDLE_HIT_PX

    for p in layer.draft.points:
        assert max(abs(p.x() - x), abs(p.y() - y)) > HANDLE_HIT_PX, "the test point is a handle"


@pytest.mark.parametrize("name", sorted(OFF_HANDLE))
def test_a_double_click_off_the_handles_of_a_plain_pick_draws_no_chord(tmp_path, name):
    # The double-click's first press reaches the Polyline tool and adds the click as a new
    # last point; the double-click must take it back instead of committing a chord to it.
    shape, at, (x, y) = OFF_HANDLE[name]
    skeleton = lines(*shape)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, *at)
    picked = list(layer.draft.edit_original)
    _assert_off_the_handles(layer, x, y)

    double_click(layer, x, y)

    _unchanged(layer, host, skeleton)
    assert layer.draft.xy() == picked, "the pick stays open as it was picked"
    assert layer.current_tool == "polyline"
    layer.cancel_scheduled_render()


def test_a_double_click_whose_second_press_drifts_still_draws_no_chord(tmp_path):
    # Qt's double-click may land a pixel or two from its first press.
    shape, at, (x, y) = OFF_HANDLE["curve"]
    skeleton = lines(*shape)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, *at)
    picked = list(layer.draft.edit_original)

    layer.press(QPoint(x, y), Qt.KeyboardModifier.NoModifier)
    layer.release(QPoint(x, y))
    layer.double_click(QPoint(x + 2, y - 1))
    layer.release(QPoint(x + 2, y - 1))

    _unchanged(layer, host, skeleton)
    assert layer.draft.xy() == picked
    layer.cancel_scheduled_render()


# Double-clicks beyond Select's reach of a plain pick (off the picked branch): the click is
# a chord only because the pick is unchanged, not because it lies on the branch.
FAR_OFF = {
    "straight, below": (STRAIGHT, (300, 242), (300, 330)),
    "straight, far corner": (STRAIGHT, (300, 242), (600, 400)),
    "Y arm, on another arm": (Y, (320, 160), (220, 340)),
}


@pytest.mark.parametrize("name", sorted(FAR_OFF))
def test_a_double_click_far_off_a_plain_pick_draws_no_chord(tmp_path, name):
    from app.editing.geometry import nearest_segment_index
    from app.gui.skeleton_tools import PICK_HIT_PX, reach

    shape, at, (x, y) = FAR_OFF[name]
    skeleton = lines(*shape)
    layer, host = open_layer(tmp_path, skeleton)
    pick(layer, *at)
    picked = list(layer.draft.edit_original)
    _assert_off_the_handles(layer, x, y)
    far = nearest_segment_index(picked, (x, y), reach(layer, PICK_HIT_PX)) is None
    assert far, "the test point lies on the picked branch"

    double_click(layer, x, y)

    _unchanged(layer, host, skeleton)
    assert layer.draft.xy() == picked, "the pick stays open as it was picked"
    assert layer.current_tool == "polyline"
    layer.cancel_scheduled_render()


@pytest.mark.parametrize("name", ["L", "curve"])
def test_a_moved_pick_and_an_off_handle_double_click_commit_as_enter_does(tmp_path, name):
    shape, at, (x, y) = OFF_HANDLE[name]
    skeleton = lines(*shape)
    masks = []
    for finish in ("enter", "double-click"):
        layer, host = open_layer(tmp_path / finish, skeleton)
        pick(layer, *at)
        middle = layer.draft.points[1]
        drag(layer, (middle.x(), middle.y()), (middle.x() + 20, middle.y() - 20))
        _assert_off_the_handles(layer, x, y)
        if finish == "enter":
            key(layer, Qt.Key.Key_Return)
        else:
            double_click(layer, x, y)
        assert edits(host) == 1 and layer.draft.points == []
        masks.append(layer.model.mask.copy())
        layer.cancel_scheduled_render()
    assert np.array_equal(masks[0], masks[1]), "the double-click drew a chord to the click"


def test_a_click_that_extends_a_pick_is_kept_by_a_later_double_click(tmp_path):
    # Only the double-click's own point is taken back: a point added by an earlier click is
    # an edit, so the double-click commits the branch with it, as Enter would.
    skeleton = lines(*STRAIGHT)
    masks = []
    for finish in ("enter", "double-click"):
        layer, host = open_layer(tmp_path / finish, skeleton)
        pick(layer, 300, 242)
        click(layer, 539, 400)  # a new last point below the right end
        assert layer.is_dirty()
        if finish == "enter":
            key(layer, Qt.Key.Key_Return)
        else:
            double_click(layer, 300, 240)
        assert edits(host) == 1 and layer.draft.points == []
        masks.append(layer.model.mask.copy())
        layer.cancel_scheduled_render()
    assert np.array_equal(masks[0], masks[1])
    assert not np.array_equal(masks[0], skeleton), "the extension was drawn"


def test_a_double_click_at_a_new_end_of_an_extended_pick_keeps_that_point(tmp_path):
    # A pick the user already extended, finished with a double-click off the branch: the
    # double-click's point is the line's new end (no chord), so it stays, as on a new line.
    skeleton = lines(*STRAIGHT)
    masks = []
    for finish in ("clicks and enter", "double-click"):
        layer, host = open_layer(tmp_path / finish, skeleton)
        pick(layer, 300, 242)
        click(layer, 520, 280)
        if finish == "clicks and enter":
            click(layer, 560, 330)
            key(layer, Qt.Key.Key_Return)
        else:
            double_click(layer, 560, 330)
        assert edits(host) == 1 and layer.draft.points == []
        masks.append(layer.model.mask.copy())
        layer.cancel_scheduled_render()
    assert masks[1][328:333, 558:563].any(), "the segment to the double-click was dropped"
    assert np.array_equal(masks[0], masks[1])


def test_a_double_click_that_drifts_from_its_press_ends_an_extended_pick_where_it_lands(
    tmp_path,
):
    # Qt delivers a double-click as press(A), release, double-click(B), and B can land a few
    # screen px from A. On an extended pick finished off its line, A stays (no chord) and B
    # is appended too, as HEAD and a new line do: the saved line ends at B.
    skeleton = lines(*STRAIGHT)
    masks = []
    for finish in ("enter", "double-click"):
        layer, host = open_layer(tmp_path / finish, skeleton)
        pick(layer, 300, 242)
        click(layer, 520, 280)
        if finish == "enter":
            click(layer, 560, 330)
            # Appended directly: a click at (562, 331) would grab the handle at (560, 330).
            layer.draft.points.append(QPoint(562, 331))
            key(layer, Qt.Key.Key_Return)
        else:
            layer.press(QPoint(560, 330), Qt.KeyboardModifier.NoModifier)
            layer.release(QPoint(560, 330))
            layer.double_click(QPoint(562, 331))
            layer.release(QPoint(562, 331))
        assert edits(host) == 1 and layer.draft.points == []
        masks.append(layer.model.mask.copy())
        layer.cancel_scheduled_render()
    assert np.array_equal(masks[0], masks[1])


def test_a_new_line_still_takes_the_double_click_as_its_last_point(tmp_path):
    skeleton = lines(*STRAIGHT)
    layer, host = open_layer(tmp_path, skeleton)
    layer.set_tool("polyline")
    click(layer, 100, 400)

    double_click(layer, 500, 400)

    assert edits(host) == 1 and layer.draft.points == []
    assert layer.model.mask[400, 300] > 0 or layer.model.mask[399:402, 300].any()
    layer.cancel_scheduled_render()


def test_the_hint_offers_enter_only_when_enter_would_commit(tmp_path):
    skeleton = lines(*L)
    layer, _host = open_layer(tmp_path, skeleton)
    pick(layer, 200, 102)
    assert "Enter commits" not in layer.tool_hint()[0], "Enter only drops a plain pick"

    key(layer, Qt.Key.Key_S)  # Smooth bends the 3-point pick: Enter applies it
    assert "Enter commits" in layer.tool_hint()[0]
    key(layer, Qt.Key.Key_S)
    assert "Enter commits" not in layer.tool_hint()[0]

    corner = layer.draft.points[1]
    drag(layer, (corner.x(), corner.y()), (320, 80))
    assert "Enter commits" in layer.tool_hint()[0]
    layer.cancel_scheduled_render()
