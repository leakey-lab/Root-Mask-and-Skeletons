"""SkeletonLayer: drags, the reference view and what the canvas item shows (C14).

The mid-gesture rule (an eraser, Connect or polyline drag is finished as its release would
finish it; the rest of that drag reaches no tool), the reference layer (dimmed, markers
and previews hidden, the draft kept), the z-order under the decorations, and the item
updated with the model's changed boxes (``take_changed_box``'s only caller).
"""

from __future__ import annotations

import ast
import os

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QGraphicsItem,
)

from app.gui.canvas_layers.skeleton import SkeletonLayerItem
from app.gui.widgets import tokens
from tests.correct_skeleton_layer_support import (
    NO_MOD,
    ROOT,
    H,
    W,
    _click,
    _commit,
    _count,
    _drag,
    _open,
    _paint,
    _render,
    _thin,
    _wait_for,
    _write_raster,
)
from tests.correct_skeleton_layer_support import gap_layer as gap_layer  # fixture
from tests.correct_skeleton_layer_support import host as host  # fixture
from tests.correct_skeleton_layer_support import layer as layer  # fixture
from tests.correct_skeleton_layer_support import project as project  # fixture
from tests.correct_skeleton_layer_support import scene as scene  # fixture
from tests.correct_skeleton_layer_support import tool_calls as tool_calls  # fixture

pytestmark = pytest.mark.usefixtures("no_prompts")


# ---------------------------------------------------------------- active and visibility
def test_endpoint_markers_are_device_cached_and_shown_only_while_active(layer):
    markers = layer.decorations.endpoints_item
    assert markers.cacheMode() == QGraphicsItem.CacheMode.DeviceCoordinateCache
    assert markers.isVisible() and not markers.path().isEmpty()

    layer.set_active(False)
    assert not markers.isVisible()
    _commit(layer, (50, 50), (200, 100))
    layer.undo()  # a full render of the reference layer keeps them hidden
    assert not markers.isVisible()

    layer.set_active(True)
    assert markers.isVisible()


def test_going_inactive_hides_the_preview_but_keeps_the_draft(layer):
    deco = layer.decorations
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    _click(layer, 200, 60)

    layer.set_active(False)

    assert not deco.polyline_item.isVisible() and not deco.handles_item.isVisible()
    assert len(layer.draft.points) == 2 and not layer.is_active()

    layer.set_active(True)

    assert deco.polyline_item.isVisible() and deco.handles_item.isVisible()
    assert not deco.polyline_item.path().isEmpty()
    assert layer.is_active()


def test_the_reference_layer_is_drawn_at_half_its_opacity(layer):
    assert layer.opacity() == pytest.approx(0.85)
    assert layer.item.opacity() == pytest.approx(0.85)

    layer.set_active(False)
    assert layer.item.opacity() == pytest.approx(0.425)
    assert layer.opacity() == pytest.approx(0.85)

    layer.set_opacity(0.5)
    assert layer.item.opacity() == pytest.approx(0.25)
    layer.set_active(True)
    assert layer.item.opacity() == pytest.approx(0.5)

    layer.set_opacity(2.0)
    assert layer.opacity() == 1.0


def test_hiding_the_layer_hides_the_item_and_its_decorations(layer):
    deco = layer.decorations
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    _click(layer, 200, 60)

    layer.set_visible(False)

    assert not layer.is_visible()
    assert not layer.item.isVisible()
    assert not deco.endpoints_item.isVisible()
    assert not deco.polyline_item.isVisible() and not deco.handles_item.isVisible()

    layer.set_visible(True)

    assert layer.item.isVisible() and deco.endpoints_item.isVisible()
    assert deco.polyline_item.isVisible()


# ------------------------------------------------------------------------------ gestures
def test_gesture_active_during_an_eraser_drag(layer):
    layer.set_tool("eraser")
    assert not layer.gesture_active()
    layer.press(QPoint(300, 240), NO_MOD)
    layer.move(QPoint(320, 240), True)
    assert layer.gesture_active()
    layer.release(QPoint(320, 240))
    assert not layer.gesture_active()


def test_gesture_active_during_a_connect_drag_from_an_endpoint(gap_layer):
    gap_layer.set_tool("connect")
    gap_layer.press(QPoint(150, 300), NO_MOD)  # no endpoint here: no drag
    assert not gap_layer.gesture_active()
    gap_layer.release(QPoint(150, 300))

    gap_layer.press(QPoint(199, 100), NO_MOD)
    assert gap_layer.gesture_active()
    gap_layer.release(QPoint(260, 100))
    assert not gap_layer.gesture_active()


def test_gesture_active_while_a_polyline_point_or_line_is_dragged(layer):
    layer.set_tool("polyline")
    layer.press(QPoint(50, 50), NO_MOD)
    assert not layer.gesture_active()  # a click that adds a point drags nothing
    layer.release(QPoint(50, 50))
    _click(layer, 200, 60)

    layer.press(QPoint(200, 60), NO_MOD)  # on a point: drag it
    assert layer.gesture_active()
    layer.release(QPoint(210, 70))
    assert not layer.gesture_active()

    layer.press(QPoint(125, 55), Qt.KeyboardModifier.ShiftModifier)  # move the line
    assert layer.gesture_active()
    layer.release(QPoint(130, 60))
    assert not layer.gesture_active()


def test_finish_gesture_releases_an_eraser_drag_once_and_ignores_its_rest(layer, host, tool_calls):
    band = np.zeros((H, W), np.uint8)
    band[238:243, 100:540] = 255
    layer.document.model.mask = band
    layer.render_skeleton()
    layer.set_tool("eraser")
    layer.set_size(6)
    host.calls.clear()
    layer.press(QPoint(150, 240), NO_MOD)
    layer.move(QPoint(170, 240), True)

    layer.finish_gesture()

    mask = layer.document.model.mask
    assert np.array_equal(mask, _thin(mask))  # the release thinned it
    assert not mask[236:245, 147:174].any()
    assert not layer.gesture_active()
    assert _count(host, "layer_edited") == 1
    after = mask.copy()
    tool_calls.clear()

    layer.move(QPoint(400, 240), True)
    layer.release(QPoint(400, 240))

    assert tool_calls == []  # the rest of the drag reaches no tool
    assert np.array_equal(layer.document.model.mask, after)
    assert layer.eraser_ring.rect().center() == QPointF(400.5, 240.5)  # only the ring moved
    assert _count(host, "layer_edited") == 1


def test_finish_gesture_without_a_drag_does_nothing(layer, host):
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    host.calls.clear()

    layer.finish_gesture()

    assert host.calls == [] and len(layer.draft.points) == 1


def test_a_tool_change_mid_eraser_drag_releases_it_and_the_rest_reaches_no_tool(
    layer, host, tool_calls
):
    layer.set_tool("eraser")
    layer.press(QPoint(150, 240), NO_MOD)
    layer.move(QPoint(170, 240), True)

    layer.set_tool("polyline")

    assert layer.current_tool == "polyline" and not layer.gesture_active()
    assert not layer.document.model.mask[240, 145:176].any()
    after = layer.document.model.mask.copy()
    tool_calls.clear()

    layer.move(QPoint(250, 240), True)
    layer.move(QPoint(300, 240), True)
    layer.release(QPoint(300, 240))

    assert tool_calls == []
    assert layer.draft.points == []
    assert np.array_equal(layer.document.model.mask, after)
    assert _count(host, "layer_edited") == 1


def test_finish_gesture_joins_a_connect_drag_at_the_last_position(gap_layer, host, tool_calls):
    gap_layer.set_tool("connect")
    gap_layer.press(QPoint(199, 100), NO_MOD)
    gap_layer.move(QPoint(230, 100), True)
    gap_layer.move(QPoint(260, 100), True)

    gap_layer.finish_gesture()

    assert gap_layer.document.model.mask[100, 200:260].all()
    assert _count(host, "layer_edited") == 1
    assert not gap_layer.decorations.connect_item.isVisible()
    tool_calls.clear()
    gap_layer.release(QPoint(300, 100))
    assert tool_calls == []
    assert _count(host, "layer_edited") == 1


def test_escape_during_a_connect_drag_drops_it(gap_layer, host, tool_calls):
    before = gap_layer.document.model.mask.copy()
    gap_layer.set_tool("connect")
    gap_layer.press(QPoint(199, 100), NO_MOD)
    gap_layer.move(QPoint(230, 100), True)

    gap_layer.cancel_transient()

    assert np.array_equal(gap_layer.document.model.mask, before)
    assert _count(host, "layer_edited") == 0
    assert not gap_layer.decorations.connect_item.isVisible()
    tool_calls.clear()
    gap_layer.release(QPoint(260, 100))
    assert tool_calls == [] and _count(host, "layer_edited") == 0


def test_a_dragged_polyline_point_stays_where_it_is_when_the_drag_is_finished(layer):
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    _click(layer, 200, 60)
    layer.press(QPoint(200, 60), NO_MOD)
    layer.move(QPoint(220, 90), True)

    layer.finish_gesture()

    assert layer.draft.points[1] == QPoint(220, 90)
    layer.move(QPoint(300, 300), True)
    layer.release(QPoint(300, 300))
    assert layer.draft.points[1] == QPoint(220, 90)


@pytest.mark.parametrize("transition", ["set_tool", "hide", "save", "revert", "open", "close"])
def test_the_layers_own_transitions_finish_a_drag_first(gap_layer, host, project, transition):
    root, image = project
    gap_layer.set_tool("connect")
    gap_layer.press(QPoint(199, 100), NO_MOD)
    gap_layer.move(QPoint(260, 100), True)

    {
        "set_tool": lambda: gap_layer.set_tool("select"),
        "hide": lambda: gap_layer.set_visible(False),
        "save": gap_layer.save,
        "revert": gap_layer.revert,
        "open": lambda: gap_layer.open(image, (W, H), root),
        "close": gap_layer.close,
    }[transition]()

    assert not gap_layer.gesture_active()
    assert _count(host, "layer_edited") == 1  # the drag was joined, not dropped
    if transition == "save":
        saved = cv2.imread(os.path.join(root, "gt", "skeletons", "a.png"), 0)
        assert saved[100, 200:260].all()


def test_a_new_press_first_finishes_a_drag_whose_release_was_lost(gap_layer, host):
    gap_layer.set_tool("connect")
    gap_layer.press(QPoint(199, 100), NO_MOD)
    gap_layer.move(QPoint(230, 100), True)

    gap_layer.press(QPoint(150, 300), NO_MOD)  # the first drag's release never came
    gap_layer.release(QPoint(150, 300))

    mask = gap_layer.document.model.mask
    assert mask[100, 200:226].all()  # joined toward where that drag last was (230, 100)
    assert not mask[250:310, 140:175].any()  # not dragged on to the second press
    assert _count(host, "layer_edited") == 1


# ------------------------------------------------------------------------------ z-order
def test_a_polyline_draft_paints_above_the_skeleton_line_at_800_percent(host, scene, project):
    layer = _open(host, scene, project, [(100, 300, 100)])
    layer.item.EXTRACT_BUDGET_MS = float("inf")  # extract every tile in the first paint
    source = QRectF(180, 80, 40, 40)  # the crossing (200.5, 100.5) at (164, 164) of the render
    before = QColor(_render(scene, source, 8).pixel(164, 164))
    assert layer.item.mode_painted() == "vector"
    assert before.red() > before.blue()  # the skeleton colour: the line alone

    layer.set_tool("polyline")
    _click(layer, 200, 60)
    _click(layer, 200, 140)
    image = _render(scene, source, 8)

    assert QColor(image.pixel(164, 164)) == QColor(tokens.ACCENT)  # the preview is on top
    beside = QColor(image.pixel(240, 164))  # x = 210: the line only
    assert beside.red() > beside.blue()
    layer.cancel_scheduled_render()


# ------------------------------------------------------------------------- item update
def test_only_the_skeleton_layer_calls_take_changed_box():
    callers = set()
    for path in (ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "take_changed_box":
                callers.add(path.relative_to(ROOT).as_posix())

    assert callers == {"app/gui/correct/skeleton_layer.py"}


def test_the_item_is_updated_with_the_models_changed_boxes(layer, monkeypatch):
    model = layer.document.model
    taken, given = [], []
    real_take = model.take_changed_box
    real_set_mask = layer.item.set_mask

    def take():
        box = real_take()
        taken.append(box)
        return box

    def set_mask(mask, **kwargs):
        given.append((mask is model.mask, kwargs.get("changed", "missing")))
        real_set_mask(mask, **kwargs)

    monkeypatch.setattr(model, "take_changed_box", take)
    monkeypatch.setattr(layer.item, "set_mask", set_mask)
    layer.set_tool("eraser")

    layer.press(QPoint(300, 240), NO_MOD)

    assert given == [(True, taken[0])]
    x, y, w, h = taken[0]  # the erased disc of radius 10 around (300, 240)
    assert x <= 290 and y <= 230 and x + w >= 311 and y + h >= 251

    layer.release(QPoint(300, 240))
    layer.undo()  # an undo writes the action's box back in place: the item gets that box

    assert given[-1] == (True, taken[-1]) and taken[-1] not in (None, (0, 0, 0, 0))
    assert len(given) == len(taken)


def test_the_item_always_shows_the_models_pixels(gap_layer, project, monkeypatch):
    layer = gap_layer
    model = layer.document.model
    steps = [
        lambda: _commit(layer, (50, 300), (200, 360)),
        lambda: layer.set_tool("eraser"),
        lambda: _drag(layer, [(120, 100), (140, 100), (160, 100)]),
        lambda: layer.set_tool("connect"),
        lambda: _drag(layer, [(260, 100), (300, 200)]),
        layer.undo,
        layer.redo,
        layer.clear,
        layer.undo,
    ]
    for step in steps:
        step()
        reference = SkeletonLayerItem()
        reference.set_mask(model.mask.copy())
        assert _paint(layer.item) == _paint(reference)

    frames = []
    real_set_mask = layer.item.set_mask

    def counted(mask, **kwargs):
        frames.append(kwargs.get("changed"))
        real_set_mask(mask, **kwargs)

    monkeypatch.setattr(layer.item, "set_mask", counted)
    layer.set_tool("eraser")
    layer.press(QPoint(120, 328), NO_MOD)  # across the committed polyline
    layer.move(QPoint(140, 336), True)
    assert _wait_for(lambda: len(frames) == 2)  # the coalesced frame of the move drew
    reference = SkeletonLayerItem()
    reference.set_mask(model.mask.copy())
    assert _paint(layer.item) == _paint(reference)
    layer.release(QPoint(140, 336))

    other = _write_raster(project[0], "elsewhere/other.png", [(30, 90, 400)])
    for step in (layer.revert, lambda: layer.load_file(other)):
        step()
        reference = SkeletonLayerItem()
        reference.set_mask(model.mask.copy())
        assert _paint(layer.item) == _paint(reference)
