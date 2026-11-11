"""The zoom model of the canvas views (app/gui/canvas_view.py): the ladder, the zoom-out
limit and trackpad gestures.

Keys and buttons step along a ladder of round zoom stops (80 % and 125 % around 100 %,
integer stops from 2x); the wheel stays continuous (1.2 per notch).

Interactive zoom-out (Ctrl+wheel, Ctrl+- and the zoom-out button) stops at the zoom-out
limit, ``min(100 %, half the fit zoom)``: the image never shrinks below half its fitted
size, and a small image can always go down to actual pixels. Without an image (or before
the view is laid out) the absolute ``MIN_ZOOM`` applies. A fit always fits, however large
the image: the fit zoom is never clamped up to ``MIN_ZOOM``.

macOS and Linux trackpads send ``QNativeGestureEvent`` instead of Ctrl+wheel. A
``ZoomNativeGesture`` of value v multiplies the zoom by ``1 + v`` about the fingers, at once
(no glide), within the zoom-out limit and ``MAX_ZOOM``; a ``SmartZoomNativeGesture``
(two-finger double tap) toggles fit and 100 %. Both are ignored mid-stroke. Every canvas has
it, the editor canvas without asking its sink.
"""

import pytest
from PyQt6.QtCore import QMargins, QPoint, QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QNativeGestureEvent, QPixmap, QPointingDevice, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene

from app.gui import canvas_gl
from app.gui.canvas_view import CanvasViewMixin
from app.gui.editor_canvas import EditorCanvasView
from tests.canvas_view_support import View as _View
from tests.qt_teardown import delete_widgets, flush_deleted

CTRL = Qt.KeyboardModifier.ControlModifier
IMAGE_W, IMAGE_H = 800, 600
ZOOM = Qt.NativeGestureType.ZoomNativeGesture
SMART_ZOOM = Qt.NativeGestureType.SmartZoomNativeGesture
FINGERS = QPointF(150, 120)

pytestmark = pytest.mark.usefixtures("no_prompts")


@pytest.fixture
def view(app, monkeypatch):
    """A shown 400x300 raster view over an 800x600 image at 100 %."""
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    view = _View()
    scene = QGraphicsScene(view)
    pm = QPixmap(IMAGE_W, IMAGE_H)
    pm.fill(QColor(90, 90, 90))
    scene.addItem(QGraphicsPixmapItem(pm))
    view.setScene(scene)
    view.set_content_rect(QRectF(0, 0, IMAGE_W, IMAGE_H))
    view.resize(400, 300)
    view.show()
    app.processEvents()
    view.set_zoom(1.0)
    yield view
    view.close()
    view.deleteLater()
    del view, scene
    flush_deleted()


class _WheelEatingSink:
    """An InputSink that consumes every wheel turn and records whether it was asked."""

    def __init__(self):
        self.wheels = 0

    def canvas_size(self) -> QSize:
        return QSize(800, 600)

    def canvas_press(self, pos, event) -> bool:
        return True

    def canvas_move(self, pos, event, dragging) -> None:
        pass

    def canvas_release(self, pos, event) -> None:
        pass

    def canvas_double_click(self, pos, event) -> None:
        pass

    def canvas_wheel(self, event) -> bool:
        self.wheels += 1
        return True

    def canvas_key_press(self, event) -> bool:
        return False

    def canvas_key_release(self, event) -> bool:
        return False

    def canvas_hover_left(self) -> None:
        pass

    def canvas_pan_changed(self, panning) -> None:
        pass

    def canvas_focus_lost(self) -> None:
        pass


@pytest.fixture
def make_view(app, monkeypatch):
    """A shown ``size`` view over a ``w`` x ``h`` content rect (a small pixmap stands in)."""
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    made = []

    def make(w=800, h=600, size=(400, 300), cls=_View):
        view = cls()
        scene = QGraphicsScene(view)
        pm = QPixmap(64, 48)
        pm.fill(QColor(90, 90, 90))
        scene.addItem(QGraphicsPixmapItem(pm))
        view.setScene(scene)
        view.set_content_rect(QRectF(0, 0, w, h))
        view.resize(*size)
        view.show()
        app.processEvents()
        made.append(view)
        return view

    yield make
    delete_widgets(made)


def fit_zoom(view, w, h, margin=CanvasViewMixin.FIT_MARGIN) -> float:
    vp = view.viewport().rect()
    return min((vp.width() - 2 * margin) / w, (vp.height() - 2 * margin) / h)


def wheel(view, delta: int, pos: QPointF | None = None) -> None:
    pos = QPointF(150, 120) if pos is None else pos
    vp = view.viewport()
    event = QWheelEvent(
        pos,
        QPointF(vp.mapToGlobal(pos.toPoint())),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        CTRL,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(vp, event)


def gesture(view, kind, value: float = 0.0, pos: QPointF = FINGERS) -> bool:
    """Send a native gesture of ``kind`` at viewport position ``pos``; whether it was taken."""
    vp = view.viewport()
    event = QNativeGestureEvent(
        kind,
        QPointingDevice.primaryPointingDevice(),
        2,
        pos,
        pos,
        QPointF(vp.mapToGlobal(pos.toPoint())),
        value,
        QPointF(0, 0),
    )
    return QApplication.sendEvent(vp, event)


# --------------------------------------------------------------------------- ladder
def test_zoom_stops_are_sorted_round_numbers_within_the_limits():
    stops = CanvasViewMixin.ZOOM_STOPS
    assert list(stops) == sorted(stops)
    assert stops[0] == CanvasViewMixin.MIN_ZOOM == 0.1
    assert stops[-1] == CanvasViewMixin.MAX_ZOOM == 20.0
    for stop in (0.8, 1.0, 1.25, 2.0, 3.0, 4.0, 5.0, 6.0):
        assert stop in stops


@pytest.mark.parametrize(
    ("start", "direction", "expected"),
    [
        (1.0, +1, 1.25),
        (1.0, -1, 0.8),
        (1.63, +1, 2.0),
        (1.63, -1, 1.5),
        (2.0, +1, 3.0),
        (3.0, -1, 2.0),
        (0.9995, +1, 1.25),  # within the 1e-3 tolerance of 100 %: the next stop up
        (1.0005, -1, 0.8),
        (20.0, +1, 20.0),  # clamped at MAX_ZOOM
        # Below the zoom-out limit (half the fit zoom, about 0.21 here) a step down stays.
        (0.1, -1, 0.1),
        (0.11, -1, 0.11),
        (19.0, +1, 20.0),
    ],
)
def test_zoom_step_moves_to_the_neighbouring_stop(view, start, direction, expected):
    view.set_zoom(start)
    view.zoom_step(direction)
    assert view.zoom_factor == pytest.approx(expected)


def test_zoom_step_keeps_the_anchor_point_in_place(view):
    anchor = QPointF(120, 80)
    before = view.mapToScene(anchor.toPoint())
    view.zoom_step(+1, anchor)
    after = view.mapToScene(anchor.toPoint())
    assert view.zoom_factor == pytest.approx(1.25)
    assert abs(after.x() - before.x()) <= 1.0
    assert abs(after.y() - before.y()) <= 1.0


def test_zoom_step_reports_the_new_zoom(view):
    seen = []
    view.zoom_changed.connect(seen.append)
    view.zoom_step(+1)
    assert seen == [pytest.approx(1.25)]


@pytest.mark.parametrize(
    ("key", "start", "expected"),
    [
        (Qt.Key.Key_Equal, 1.63, 2.0),
        (Qt.Key.Key_Plus, 1.63, 2.0),
        (Qt.Key.Key_Minus, 1.63, 1.5),
        (Qt.Key.Key_Equal, 4.0, 5.0),
        (Qt.Key.Key_Minus, 0.5, 0.333),
    ],
)
def test_ctrl_zoom_keys_follow_the_ladder(view, key, start, expected):
    view.set_zoom(start)
    QTest.keyClick(view, key, CTRL)
    assert view.zoom_factor == pytest.approx(expected)


# ------------------------------------------------------------------ zoom-out limit


def test_the_zoom_out_limit_is_half_the_fit_zoom(make_view):
    view = make_view()
    assert view.zoom_out_limit == pytest.approx(fit_zoom(view, 800, 600) / 2)


def test_zoom_steps_down_stop_at_the_limit(make_view):
    view = make_view()
    view.fit_to_view()
    for _ in range(12):
        view.zoom_step(-1)
    assert view.zoom_factor == pytest.approx(view.zoom_out_limit)
    view.zoom_step(-1)
    assert view.zoom_factor == pytest.approx(view.zoom_out_limit)


def test_ctrl_wheel_out_stops_at_the_limit(make_view):
    view = make_view()
    view.fit_to_view()
    for _ in range(20):
        wheel(view, -120)
    assert view.zoom_factor == pytest.approx(view.zoom_out_limit)


def test_a_small_image_can_always_zoom_out_to_actual_pixels(make_view):
    view = make_view(64, 48)  # fits at about 5x
    assert fit_zoom(view, 64, 48) / 2 > 1.0
    assert view.zoom_out_limit == 1.0
    view.fit_to_view()
    for _ in range(12):
        view.zoom_step(-1)
    assert view.zoom_factor == pytest.approx(1.0)


def test_below_the_limit_a_step_down_stays_and_a_step_up_zooms_in(make_view):
    view = make_view()
    view.set_zoom(0.11)  # programmatic: below the interactive limit
    view.zoom_step(-1)
    assert view.zoom_factor == pytest.approx(0.11)
    view.zoom_step(+1)
    assert view.zoom_factor > 0.11


def test_the_limit_follows_the_viewport_size(app, make_view):
    view = make_view()
    small = view.zoom_out_limit
    view.resize(800, 600)
    app.processEvents()
    assert view.zoom_out_limit == pytest.approx(fit_zoom(view, 800, 600) / 2)
    assert view.zoom_out_limit > small


def test_without_content_the_absolute_minimum_applies(make_view):
    view = make_view()
    view.clear_content_rect()
    assert view.zoom_out_limit == CanvasViewMixin.MIN_ZOOM


def test_a_huge_image_fits_below_the_absolute_minimum(make_view):
    """A 6000 x 4000 photo in a 600 px canvas fits at about 0.09x: Ctrl+0 must show all of it
    (the fit used to be clamped up to MIN_ZOOM = 0.1, cutting the image off)."""
    view = make_view(6000, 4000, size=(600, 400))
    expected = fit_zoom(view, 6000, 4000)
    assert expected < CanvasViewMixin.MIN_ZOOM

    view.fit_to_view()

    assert view.zoom_factor == pytest.approx(expected)
    shown = view.viewportTransform().mapRect(QRectF(0, 0, 6000, 4000))
    vp = view.viewport().rect()
    assert shown.left() >= -1 and shown.right() <= vp.width() + 1
    assert shown.top() >= -1 and shown.bottom() <= vp.height() + 1
    view.zoom_step(-1)  # and zooming out from there stops at half of it
    assert view.zoom_out_limit == pytest.approx(expected / 2)
    assert view.zoom_factor < expected


def test_a_fit_with_insets_bigger_than_the_margin_keeps_the_limit_at_half_of_it(make_view):
    view = make_view()
    insets = QMargins(10, 20, 150, 40)
    view.set_fit_insets_provider(lambda: insets)
    vp = view.viewport().rect()
    fit = min((vp.width() - 160) / 800, (vp.height() - 60) / 600)
    assert view.zoom_out_limit == pytest.approx(fit / 2)


# ------------------------------------------------------------------ trackpad gestures


def test_pinch_multiplies_the_zoom_and_keeps_the_anchor(make_view):
    view = make_view()
    view.set_zoom(0.5)
    before = view.mapToScene(FINGERS.toPoint())
    assert gesture(view, ZOOM, 0.25)
    assert view.zoom_factor == pytest.approx(0.625)
    after = view.mapToScene(FINGERS.toPoint())
    assert after.x() == pytest.approx(before.x(), abs=0.5)
    assert after.y() == pytest.approx(before.y(), abs=0.5)


def test_pinch_out_stops_at_the_zoom_out_limit(make_view):
    view = make_view()
    for _ in range(60):
        gesture(view, ZOOM, -0.10)
    assert view.zoom_factor == pytest.approx(view.zoom_out_limit)


def test_pinch_in_stops_at_max_zoom(make_view):
    view = make_view()
    for _ in range(80):
        gesture(view, ZOOM, 0.20)
    assert view.zoom_factor == pytest.approx(view.MAX_ZOOM)


def test_pinch_is_ignored_while_drawing(make_view):
    view = make_view()
    view.set_zoom(0.5)
    view.drawing = True
    gesture(view, ZOOM, 0.25)
    assert view.zoom_factor == pytest.approx(0.5)


def test_smart_zoom_toggles_fit_and_actual_pixels(make_view):
    view = make_view()
    view.fit_to_view()
    fitted = view.zoom_factor
    assert fitted != pytest.approx(1.0)
    gesture(view, SMART_ZOOM)
    assert view.zoom_factor == pytest.approx(1.0)
    gesture(view, SMART_ZOOM)
    assert view.zoom_factor == pytest.approx(fitted)


def test_pinch_ends_a_glide(make_view):
    view = make_view()
    view.set_zoom(1.0)
    view.set_zoom_animation(120)
    view.zoom_step(+1)  # glides to 125 %
    assert view.zoom_animating()
    gesture(view, ZOOM, 0.25)
    assert not view.zoom_animating()
    assert view.zoom_factor == pytest.approx(1.25 * 1.25)  # on from the glide's target


def test_pinch_reaches_the_editor_canvas(make_view):
    view = make_view(cls=EditorCanvasView)
    sink = _WheelEatingSink()
    view.set_sink(sink)
    view.set_zoom(0.5)
    gesture(view, ZOOM, 0.25)
    assert view.zoom_factor == pytest.approx(0.625)
    assert sink.wheels == 0
