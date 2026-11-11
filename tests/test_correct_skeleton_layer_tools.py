"""SkeletonLayer: its tools through the layer (C14). Select hands a picked branch to
Polyline; the Eraser (footprint ring, coalesced repaints, size steps); Delete of a picked
branch; Polyline (Enter, double-click, Smooth, draft points); Connect endpoints (the
endpoint nearest the press); Clear. Offscreen, with a recording ``FakeHost``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QPainterPath, QTransform
from PyQt6.QtWidgets import (
    QGraphicsView,
)

from app.gui.correct.skeleton_layer import DraftState, SkeletonLayer
from app.gui.key_text import platform_keys
from app.gui.widgets import tokens
from tests.correct_skeleton_layer_support import (
    NO_MOD,
    H,
    W,
    _click,
    _count,
    _drag,
    _key,
    _open,
    _thin,
    _wait_for,
)
from tests.correct_skeleton_layer_support import gap_layer as gap_layer  # fixture
from tests.correct_skeleton_layer_support import host as host  # fixture
from tests.correct_skeleton_layer_support import layer as layer  # fixture
from tests.correct_skeleton_layer_support import project as project  # fixture
from tests.correct_skeleton_layer_support import scene as scene  # fixture
from tests.correct_skeleton_layer_support import tool_calls as tool_calls  # fixture

pytestmark = pytest.mark.usefixtures("no_prompts")


# ------------------------------------------------------------------------------ tools
def test_set_tool_deactivates_the_old_tool_and_resets_the_draft(layer, host):
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    _click(layer, 150, 60)
    host.calls.clear()

    layer.set_tool("connect")

    assert layer.current_tool == "connect"
    assert layer.draft.points == []
    assert layer.decorations.polyline_item.path().isEmpty()
    assert _count(host, "layer_changed") == 1


def test_select_hands_the_picked_branch_to_polyline_keeping_the_draft(layer, host):
    layer.set_tool("select")
    host.calls.clear()

    _click(layer, 300, 242)

    assert layer.current_tool == "polyline"
    assert layer.draft.edit_original is not None
    assert sorted((p.x(), p.y()) for p in layer.draft.points) == [(100, 240), (539, 240)]
    assert layer.draft_state() == DraftState(True, 2, False)
    assert ("layer_changed", "skeleton") in host.calls
    path = layer.decorations.polyline_item.path()  # the click's release reached Polyline,
    assert path.elementCount() == 3  # as today: a rubber band from the branch to the pointer
    assert (path.elementAt(2).x, path.elementAt(2).y) == (300.5, 242.5)  # its pixel centre


def test_the_tool_host_geometry_follows_the_open_skeleton(layer, host, scene):
    assert layer.image_size() == (W, H)
    assert layer.clamp_to_image(QPoint(-5, 900)) == QPoint(0, H - 1)
    assert layer.clamp_to_image(QPoint(700, 20)) == QPoint(W - 1, 20)
    closed = SkeletonLayer(host, scene)
    assert closed.image_size() == (0, 0)
    assert closed.clamp_to_image(QPoint(5, 5)) == QPoint(0, 0)


def test_hover_moves_reach_the_current_tool(layer):
    layer.set_tool("polyline")
    _click(layer, 50, 50)

    layer.move(QPoint(120, 90), False)

    path = layer.decorations.polyline_item.path()
    assert path.elementCount() == 2
    assert (path.elementAt(1).x, path.elementAt(1).y) == (120.5, 90.5)  # the pixel's centre


# ----------------------------------------------------------------------------- eraser
def test_a_fast_eraser_drag_leaves_no_fragments(layer):
    layer.set_tool("eraser")
    layer.set_size(6)

    _drag(layer, [(120, 240), *((x, 240) for x in range(145, 521, 25))])

    assert int((layer.document.model.mask[240, 120:521] > 0).sum()) == 0


def test_eraser_drag_repaints_once_per_frame_and_its_release_thins_and_renders(
    layer, host, monkeypatch
):
    band = np.zeros((H, W), np.uint8)
    band[238:243, 100:540] = 255  # 5 px thick: the release has something to thin
    layer.document.model.mask = band
    layer.render_skeleton()
    renders = []
    real_set_mask = layer.item.set_mask

    def spy(mask, **kwargs):
        renders.append(kwargs.get("changed"))
        real_set_mask(mask, **kwargs)

    monkeypatch.setattr(layer.item, "set_mask", spy)
    markers = layer.decorations.endpoints_item
    layer.set_tool("eraser")
    layer.set_size(6)
    host.calls.clear()

    layer.press(QPoint(120, 240), NO_MOD)
    for x in range(125, 300, 5):
        layer.move(QPoint(x, 240), True)

    assert len(renders) == 1  # the press draws at once, the moves only schedule a frame
    assert not markers.isVisible()  # hidden while erasing, as today
    assert _wait_for(lambda: len(renders) == 2)  # the last moves show without a release
    assert len(renders) == 2  # one frame for all the moves
    assert _count(host, "layer_changed") == 0  # drag frames never reach the host

    layer.release(QPoint(295, 240))

    assert len(renders) == 3
    assert _count(host, "layer_changed") == 1
    mask = layer.document.model.mask
    assert np.array_equal(mask, _thin(mask))
    assert markers.isVisible()
    assert _count(host, "layer_edited") == 1


def test_the_eraser_ring_is_the_footprint_at_one_and_four_times(layer, scene):
    view = QGraphicsView(scene)
    view.resize(900, 700)
    layer.set_tool("eraser")
    layer.set_size(20)
    for zoom in (1.0, 4.0):
        view.setTransform(QTransform.fromScale(zoom, zoom))
        layer.move(QPoint(300, 200), False)
        ring = layer.eraser_ring
        assert ring.is_visible()
        assert ring.rect().width() == pytest.approx(20)
        assert ring.rect().center() == QPointF(300.5, 200.5)
        on_screen = view.mapFromScene(ring.rect()).boundingRect().width()
        assert on_screen == pytest.approx(20 * zoom, abs=2)
    assert layer.eraser_footprint_radius() == 10


def test_the_eraser_ring_hides_for_other_tools_while_panning_and_when_inactive(layer, host):
    ring = layer.eraser_ring
    layer.set_tool("eraser")
    layer.move(QPoint(300, 200), False)
    assert ring.is_visible()

    host.panning = True
    layer.hover(QPoint(310, 200))
    assert not ring.is_visible()
    host.panning = False
    layer.hover(QPoint(310, 200))
    assert ring.is_visible()

    layer.set_active(False)
    assert not ring.is_visible()
    layer.hover(QPoint(300, 200))
    assert not ring.is_visible()
    layer.set_active(True)

    layer.hover(QPoint(300, 200))
    layer.hover(None)
    assert not ring.is_visible()

    layer.set_tool("polyline")
    layer.move(QPoint(300, 200), False)
    assert not ring.is_visible()


def test_the_eraser_ring_is_dashed_in_the_skeleton_colour(layer):
    pen = layer.eraser_ring.inner.pen()
    assert pen.style() == Qt.PenStyle.DashLine
    assert pen.color() == QColor(tokens.SKEL)


def test_eraser_size_steps_by_twelve_percent_at_least_one_px_and_clamps(layer, host):
    assert layer.size() == 20
    layer.step_size(1)
    assert layer.size() == 22
    layer.set_size(20)
    layer.step_size(-1)
    assert layer.size() == 18
    layer.set_size(3)
    layer.step_size(1)
    assert layer.size() == 4
    layer.set_size(50)
    layer.step_size(1)
    assert layer.size() == 56
    layer.set_size(0)
    assert layer.size() == 1
    layer.step_size(-1)
    assert layer.size() == 1
    layer.set_size(500)
    assert layer.size() == 80
    host.calls.clear()
    layer.step_size(1)
    assert layer.size() == 80
    assert _count(host, "layer_changed") == 0  # nothing changed


def test_the_ring_and_the_footprint_follow_the_size(layer, host):
    layer.set_tool("eraser")
    layer.hover(QPoint(300, 200))
    host.calls.clear()

    layer.set_size(30)

    assert layer.eraser_ring.rect().width() == pytest.approx(30)
    assert layer.eraser_footprint_radius() == 15
    assert _count(host, "layer_changed") == 1  # the dock slider follows


# ------------------------------------------------------------------- select and delete
def test_delete_removes_the_picked_branch_and_undo_restores_it(layer, host):
    before = layer.document.model.mask.copy()
    layer.set_tool("select")
    _click(layer, 300, 242)

    assert layer.key(_key(Qt.Key.Key_Delete)) is True

    assert not layer.document.model.mask[240, 110:530].any()
    assert layer.draft.points == []
    assert ("notify", platform_keys("Branch deleted · Ctrl+Z to undo"), "info") in host.calls
    assert layer.undo() is True
    assert np.array_equal(layer.document.model.mask, before)


def test_delete_with_nothing_picked_is_a_no_op(layer):
    before = layer.document.model.mask.copy()

    assert layer.key(_key(Qt.Key.Key_Delete)) is False
    assert np.array_equal(layer.document.model.mask, before)


# ---------------------------------------------------------------------------- polyline
def _new_line_drawn(layer):
    return bool(layer.document.model.mask[45:66, 50:201].any())


def test_enter_commits_the_polyline(layer):
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    _click(layer, 200, 60)
    assert not _new_line_drawn(layer)

    assert layer.key(_key(Qt.Key.Key_Return)) is True

    assert _new_line_drawn(layer)
    assert layer.draft.points == []


def test_a_double_click_commits_the_polyline(layer):
    layer.set_tool("polyline")
    _click(layer, 50, 50)

    layer.double_click(QPoint(200, 60))

    assert _new_line_drawn(layer)
    assert layer.draft.points == []


def test_finish_polyline_needs_the_polyline_tool_and_two_points(layer):
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    layer.finish_polyline()
    assert not _new_line_drawn(layer) and len(layer.draft.points) == 1

    _click(layer, 200, 60)
    layer.finish_polyline()

    assert _new_line_drawn(layer) and layer.draft.points == []


@pytest.mark.parametrize("cancel", ["cancel_transient", "cancel_polyline"])
def test_escape_and_cancel_drop_the_draft(layer, cancel):
    before = layer.document.model.mask.copy()
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    _click(layer, 200, 60)

    getattr(layer, cancel)()

    assert layer.draft.points == []
    assert layer.decorations.polyline_item.path().isEmpty()
    assert np.array_equal(layer.document.model.mask, before)


def test_s_toggles_smooth_only_in_polyline(layer):
    layer.set_tool("select")
    assert layer.key(_key(Qt.Key.Key_S)) is False
    assert not layer.draft.smooth

    layer.set_tool("polyline")
    assert layer.key(_key(Qt.Key.Key_S)) is True
    assert layer.draft.smooth and layer.draft_state().smooth
    assert layer.key(_key(Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)) is False
    assert layer.draft.smooth


def test_set_smooth_redraws_the_preview_and_reports_the_draft(layer, host):
    layer.set_tool("polyline")
    for x, y in ((50, 50), (120, 90), (200, 60)):
        _click(layer, x, y)
    host.calls.clear()

    layer.set_smooth(True)

    assert layer.draft_state() == DraftState(True, 3, True)
    path = layer.decorations.polyline_item.path()
    kinds = {path.elementAt(i).type for i in range(path.elementCount())}
    assert QPainterPath.ElementType.CurveToElement in kinds
    assert _count(host, "layer_changed") == 1


def test_undo_in_gesture_pops_draft_points_and_nothing_else(layer):
    before = layer.document.model.mask.copy()
    layer.set_tool("polyline")
    for x, y in ((50, 50), (120, 90), (200, 60)):
        _click(layer, x, y)

    assert layer.undo_in_gesture() is True
    assert [(p.x(), p.y()) for p in layer.draft.points] == [(50, 50), (120, 90)]
    assert layer.undo_in_gesture() and layer.undo_in_gesture()
    assert layer.draft.points == []
    assert layer.undo_in_gesture() is False
    assert np.array_equal(layer.document.model.mask, before)
    assert not layer.can_undo()


def test_draft_state_reports_the_tool_the_points_and_smooth(layer):
    assert layer.draft_state() == DraftState(False, 0, False)
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    assert layer.draft_state() == DraftState(True, 1, False)


# ------------------------------------------------------------------------------ connect
def test_a_connect_drag_joins_two_endpoints(gap_layer, host):
    gap_layer.set_tool("connect")

    _drag(gap_layer, [(199, 100), (230, 100), (260, 100)])

    assert gap_layer.document.model.mask[100, 200:260].all()
    assert _count(host, "layer_edited") == 1


def connect_preview_start(layer) -> QPointF:
    """Where the Connect preview line the user sees starts (scene px, a pixel's centre)."""
    item = layer.decorations.connect_item
    assert item.isVisible(), "no Connect preview is shown"
    return item.line().p1()


def test_connect_starts_at_the_endpoint_nearest_the_press(host, scene, project):
    # Two tips both within reach of the press: the nearer one, not the first in row-major
    # order, is where the line starts (a guard of the owner bug report of 2026-11-02).
    layer = _open(host, scene, project, [(100, 200, 100), (207, 300, 108)])
    assert {(199, 100), (207, 108)} <= set(layer.model.endpoints())
    layer.set_tool("connect")

    layer.press(QPoint(205, 106), NO_MOD)  # 2.8 px from (207, 108), 8.5 px from (199, 100)

    assert connect_preview_start(layer) == QPointF(207.5, 108.5)
    layer.release(QPoint(205, 106))
    layer.cancel_scheduled_render()


def test_connect_breaks_a_tie_between_endpoints_by_the_lower_y_then_x(host, scene, project):
    layer = _open(host, scene, project, [(100, 200, 100), (100, 200, 110)])
    assert {(199, 100), (199, 110)} <= set(layer.model.endpoints())
    layer.set_tool("connect")

    layer.press(QPoint(199, 105), NO_MOD)  # 5 px from both

    assert connect_preview_start(layer) == QPointF(199.5, 100.5)
    layer.release(QPoint(199, 105))
    layer.cancel_scheduled_render()


# -------------------------------------------------------------------------------- clear
def test_clear_is_undoable_and_posts_the_cleared_notice(layer, host):
    assert layer.clear() is True

    assert not layer.document.model.mask.any()
    assert ("notify", platform_keys("Skeleton cleared. Ctrl+Z to undo."), "info") in host.calls
    assert layer.undo() is True
    assert layer.document.model.mask[240, 100:540].all()
