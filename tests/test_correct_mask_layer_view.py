"""MaskLayer: what the user sees of the layer and how input reaches it (C13).

The brush ring (follows the pointer in the mask colour, dashed for the eraser, none for
Fill, hidden while panning, closed, hidden or the reference layer), the user's opacity and
the dimmed reference, the brush size (clamped 1..100, 12 % steps, one slider for brush and
eraser), the tool hint, and the whole path EditorCanvasView -> ToolRouter -> MaskLayer.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import (
    QColor,
    QMouseEvent,
    QWheelEvent,
)
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QGraphicsScene

from app.gui.correct.layer import (
    LayerStatus,
)
from app.gui.correct.mask_layer import MaskLayer
from app.gui.correct.tool_router import ToolRouter
from app.gui.editor_canvas import EditorCanvasView
from app.gui.widgets import tokens
from tests.correct_fakes import FakeHost, FakeLayer
from tests.correct_mask_layer_support import (
    NO_MODS,
    H,
    Replay,
    W,
    click,
    count,
    image_path,
    open_layer,
    plane_of,
    ring_rect,
)
from tests.correct_mask_layer_support import scene as scene  # fixture

pytestmark = pytest.mark.usefixtures("no_prompts")


# --------------------------------------------------------------------------- view state
def test_the_ring_follows_the_pointer_in_the_mask_colour(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    layer.set_size(8)

    layer.hover(QPoint(30, 40))

    assert layer.ring.is_visible()
    assert ring_rect(layer) == QRectF(26.5, 36.5, 8, 8)  # centred on the pixel centre
    assert layer.ring.inner.pen().color() == QColor(tokens.MASK)
    assert layer.ring.inner.pen().style() == Qt.PenStyle.SolidLine

    layer.move(QPoint(31, 41), False)
    assert ring_rect(layer) == QRectF(27.5, 37.5, 8, 8)
    layer.hover(None)
    assert not layer.ring.is_visible()


def test_the_eraser_ring_is_dashed_and_fill_has_none(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    layer.hover(QPoint(30, 40))

    layer.set_tool("eraser")
    assert layer.ring.is_visible()
    assert layer.ring.inner.pen().color() == QColor(tokens.MASK)
    assert layer.ring.inner.pen().style() == Qt.PenStyle.DashLine

    layer.set_tool("fill")
    assert not layer.ring.is_visible()
    layer.move(QPoint(35, 45), False)
    assert not layer.ring.is_visible()

    layer.set_tool("brush")
    assert layer.ring.is_visible()
    assert layer.ring.inner.pen().style() == Qt.PenStyle.SolidLine


def test_the_ring_hides_while_the_view_pans(scene, tmp_path):
    host = FakeHost(panning=True)
    layer, _ = open_layer(scene, tmp_path, host=host)

    layer.hover(QPoint(30, 40))
    assert not layer.ring.is_visible()

    host.panning = False
    layer.move(QPoint(31, 40), False)
    assert layer.ring.is_visible()


def test_the_ring_diameter_is_the_brush_size(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    layer.hover(QPoint(50, 50))

    layer.set_size(21)

    assert ring_rect(layer) == QRectF(40.0, 40.0, 21, 21)


def test_a_closed_layer_shows_no_ring_and_ignores_input(scene):
    host = FakeHost()
    layer = MaskLayer(host, scene)

    layer.hover(QPoint(3, 3))
    layer.press(QPoint(3, 3), NO_MODS)
    layer.move(QPoint(9, 9), True)
    layer.release(QPoint(9, 9))

    assert not layer.ring.is_visible()
    assert layer.gesture_active() is False
    assert (layer.is_open(), layer.file_stamps(), layer.status()) == (
        False,
        (),
        LayerStatus("none"),
    )
    assert host.calls == []
    assert layer.save().ok is False


def test_a_fill_click_on_a_closed_layer_says_nothing(scene):
    host = FakeHost()
    layer = MaskLayer(host, scene)
    layer.set_tool("fill")
    host.calls.clear()

    click(layer, (3, 3))

    assert host.calls == []


def test_a_closed_layer_hides_its_ring_even_where_the_pointer_is(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    layer.hover(QPoint(30, 40))
    assert layer.ring.is_visible()

    layer.close()
    assert not layer.ring.is_visible()
    layer.hover(QPoint(31, 40))
    assert not layer.ring.is_visible()


def test_the_active_state_dims_the_layer_and_hides_the_ring(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    layer.hover(QPoint(30, 40))
    assert layer.is_active() is True
    assert layer.item.opacity() == pytest.approx(0.60)

    layer.set_active(False)
    assert layer.is_active() is False
    assert layer.item.opacity() == pytest.approx(0.30)
    assert not layer.ring.is_visible()
    layer.hover(QPoint(31, 40))
    assert not layer.ring.is_visible()

    layer.set_active(True)
    assert layer.item.opacity() == pytest.approx(0.60)
    assert not layer.ring.is_visible()  # shown again where the pointer next is
    layer.hover(QPoint(32, 40))
    assert layer.ring.is_visible()


def test_the_opacity_is_the_users_and_the_reference_is_half_of_it(scene):
    layer = MaskLayer(FakeHost(), scene)

    layer.set_opacity(0.8)
    assert layer.opacity() == pytest.approx(0.8)
    assert layer.item.opacity() == pytest.approx(0.8)
    layer.set_active(False)
    assert layer.opacity() == pytest.approx(0.8)
    assert layer.item.opacity() == pytest.approx(0.4)
    layer.set_opacity(1.7)
    assert layer.opacity() == pytest.approx(1.0)
    assert layer.item.opacity() == pytest.approx(0.5)


def test_a_hidden_layer_hides_the_item_and_shows_no_hover_ring(scene, tmp_path):
    layer, _ = open_layer(scene, tmp_path)
    layer.hover(QPoint(30, 40))

    layer.set_visible(False)
    assert (layer.is_visible(), layer.item.isVisible(), layer.ring.is_visible()) == (
        False,
        False,
        False,
    )
    layer.hover(QPoint(31, 40))
    assert not layer.ring.is_visible()

    layer.set_visible(True)
    assert layer.item.isVisible()
    layer.hover(QPoint(32, 40))
    assert layer.ring.is_visible()


def test_a_press_shows_the_ring_where_the_stroke_starts(scene, tmp_path):
    # The router shows a hidden layer, then presses: the ring must follow the stroke.
    layer, _ = open_layer(scene, tmp_path)
    layer.set_visible(False)
    layer.set_visible(True)

    layer.press(QPoint(60, 60), NO_MODS)
    assert layer.ring.is_visible()
    layer.move(QPoint(70, 60), True)
    assert ring_rect(layer).center() == QRectF(70, 60, 1, 1).center()
    layer.release(QPoint(70, 60))


# --------------------------------------------------------------------------- sizes
@pytest.mark.parametrize(
    ("start", "direction", "expected"),
    [
        (8, 1, 9),
        (8, -1, 7),
        (50, 1, 56),
        (56, -1, 49),
        (100, 1, 100),
        (95, 1, 100),
        (1, -1, 1),
        (4, 1, 5),  # 12 % of 4 rounds to 0: the step is still 1 px
        (4, -1, 3),
        (2, 1, 3),
    ],
)
def test_step_size_moves_twelve_percent_at_least_one_pixel_within_range(
    scene, start, direction, expected
):
    layer = MaskLayer(FakeHost(), scene)
    layer.set_size(start)

    layer.step_size(direction)

    assert layer.size() == expected


@pytest.mark.parametrize(("value", "expected"), [(0, 1), (-5, 1), (1, 1), (100, 100), (500, 100)])
def test_set_size_clamps_to_one_to_a_hundred(scene, value, expected):
    layer = MaskLayer(FakeHost(), scene)

    layer.set_size(value)

    assert layer.size() == expected


def test_brush_and_eraser_share_one_size_slider_and_fill_has_none(scene):
    layer = MaskLayer(FakeHost(), scene)

    brush = layer.size_spec()
    layer.set_tool("eraser")
    eraser = layer.size_spec()
    layer.set_tool("fill")

    assert brush is not None and brush == eraser
    assert (brush.minimum, brush.maximum) == (1, 100)
    assert layer.size_spec() is None


def test_the_hint_names_the_current_tool(scene):
    layer = MaskLayer(FakeHost(), scene)
    hints = {}
    for tool, label in (("brush", "Brush"), ("eraser", "Eraser"), ("fill", "Fill")):
        layer.set_tool(tool)
        text, tooltip = layer.tool_hint()
        assert text.startswith(label), text
        assert tooltip
        hints[tool] = text

    assert len(set(hints.values())) == 3


# --------------------------------------------------------------------------- the input path
def _mouse(view, kind, scene_xy, button, buttons) -> None:
    vp = view.viewport()
    pos = QPointF(view.mapFromScene(QPointF(*scene_xy)))
    global_pos = QPointF(vp.mapToGlobal(pos.toPoint()))
    QApplication.sendEvent(vp, QMouseEvent(kind, pos, global_pos, button, buttons, NO_MODS))


def test_the_canvas_view_and_router_drive_the_layer_end_to_end(qapp, tmp_path):
    # The wiring P7b builds: EditorCanvasView -> ToolRouter (the sink) -> MaskLayer.
    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    view = EditorCanvasView()
    scene = QGraphicsScene(view)
    view.setScene(scene)
    host = FakeHost()
    layer = MaskLayer(host, scene)
    layer.open(image_path(tmp_path), (W, H), None)
    router = ToolRouter(
        {"mask": layer, "skeleton": FakeLayer("skeleton")},
        canvas_size=lambda: QSize(W, H),
        on_tool_request=lambda layer_id, tool_id: None,
        on_b_hold=lambda held: None,
        on_layer_shown=lambda layer_id: None,
    )
    view.set_sink(router)
    view.set_content_rect(QRectF(0, 0, W, H))
    view.resize(400, 300)
    view.show()
    qapp.processEvents()
    view.set_zoom(2.0)
    view.centerOn(W / 2, H / 2)
    oracle = Replay(np.zeros((H, W), np.uint8))
    try:
        # a stroke: image pixel (x, y) is hit at its centre (x + 0.5, y + 0.5)
        _mouse(view, QEvent.Type.MouseButtonPress, (40.5, 30.5), left, left)
        _mouse(view, QEvent.Type.MouseMove, (80.5, 50.5), none, left)
        _mouse(view, QEvent.Type.MouseMove, (100.5, 90.5), none, left)
        _mouse(view, QEvent.Type.MouseButtonRelease, (100.5, 90.5), left, none)
        oracle.stroke([(40, 30), (80, 50), (100, 90)], 8)
        assert np.array_equal(plane_of(layer), oracle.plane)
        assert ring_rect(layer).center() == QRectF(100, 90, 1, 1).center()

        # Esc mid-drag commits the stroke; the rest of that drag paints nothing
        _mouse(view, QEvent.Type.MouseButtonPress, (40.5, 100.5), left, left)
        _mouse(view, QEvent.Type.MouseMove, (70.5, 100.5), none, left)
        QTest.keyClick(view, Qt.Key.Key_Escape)
        _mouse(view, QEvent.Type.MouseMove, (120.5, 60.5), none, left)
        _mouse(view, QEvent.Type.MouseButtonRelease, (120.5, 60.5), left, none)
        oracle.stroke([(40, 100), (70, 100)], 8)
        assert np.array_equal(plane_of(layer), oracle.plane)
        assert count(host, "layer_edited") == 2

        # hold B + wheel: the brush grows by 12 % (at least 1 px) and the UI hears of it
        host.calls.clear()
        QTest.keyPress(view, Qt.Key.Key_B)
        vp = view.viewport()
        centre = QPointF(200, 150)
        QApplication.sendEvent(
            vp,
            QWheelEvent(
                centre,
                QPointF(vp.mapToGlobal(centre.toPoint())),
                QPoint(0, 0),
                QPoint(0, 120),
                none,
                NO_MODS,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            ),
        )
        QTest.keyRelease(view, Qt.Key.Key_B)
        assert layer.size() == 9
        assert ("layer_changed", "mask") in host.calls
    finally:
        view.close()
