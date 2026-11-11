"""Where a Ctrl+wheel zoom anchors (app/gui/canvas_view.py).

The point under the cursor stays under it through a run of Ctrl+wheel notches. The view
scrolls in whole pixels, so after each zoom the anchored scene point sits up to half a pixel
off the cursor. Re-reading the point under the cursor at the next notch took that error as
the new target, and a run of notches into a deep zoom magnified the earlier errors: the image
slid by up to 5 screen px under a still cursor (270 % -> 1160 %). While nothing else moves
the view, each zoom now aims at the point the run started from.

A zoom the view could not anchor does not steer the next one. Near the zoom-out limit the
padded scene is no larger than the viewport: the view centres it and cannot scroll, so the
anchored point is not under the anchor any more. The next zoom at the same anchor must then
start from the point shown there, not from the one the earlier zooms aimed at (the view
jumped back to that one: ~500 px at 300 % in the probe below).
"""

import pytest
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene

from app.gui import canvas_gl
from tests.canvas_view_support import FramelessView
from tests.canvas_view_support import View as _View
from tests.qt_teardown import flush_deleted

CTRL = Qt.KeyboardModifier.ControlModifier
W, H = 400, 300  # the small image of the clamped cases


@pytest.fixture
def view(app, monkeypatch):
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    view = _View()
    scene = QGraphicsScene(view)
    pm = QPixmap(2000, 1500)
    pm.fill(QColor(90, 90, 90))
    scene.addItem(QGraphicsPixmapItem(pm))
    view.setScene(scene)
    view.set_content_rect(QRectF(0, 0, 2000, 1500))
    view.resize(900, 700)
    view.show()
    app.processEvents()
    yield view
    view.close()
    view.deleteLater()
    del view, scene
    flush_deleted()


def wheel(view, anchor: QPointF, delta: int) -> None:
    vp = view.viewport()
    event = QWheelEvent(
        anchor,
        QPointF(vp.mapToGlobal(anchor.toPoint())),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        CTRL,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(vp, event)


def screen_drift(view, anchor: QPointF, start: QPointF) -> float:
    """How far (screen px) the scene point ``start`` now is from the cursor pixel."""
    now = view.mapFromScene(start)
    pixel = anchor.toPoint()
    return max(abs(now.x() - pixel.x()), abs(now.y() - pixel.y()))


@pytest.fixture
def small_view(app, monkeypatch):
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    view = FramelessView()
    scene = QGraphicsScene(view)
    pm = QPixmap(W, H)
    pm.fill(QColor(90, 90, 90))
    item = QGraphicsPixmapItem(pm)
    scene.addItem(item)
    view.setScene(scene)
    view.set_content_rect(QRectF(0, 0, W, H))
    view.resize(W, H)
    view.show()
    app.processEvents()
    view.fit_to_view()
    yield view
    scene.removeItem(item)
    view.close()
    view.deleteLater()
    del view, scene
    flush_deleted()


def _off_anchor(view, scene_pt: QPointF, anchor: QPoint) -> int:
    shown = view.mapFromScene(scene_pt)
    return max(abs(shown.x() - anchor.x()), abs(shown.y() - anchor.y()))


def _zoom_out_to_the_limit(view, anchor: QPointF) -> None:
    for _ in range(12):  # Ctrl+wheel notches out, well past the limit
        wheel(view, anchor, -120)
    assert view.zoom_factor == pytest.approx(view.zoom_out_limit)


@pytest.mark.parametrize("zoom", [0.3, 1.0, 2.7])
@pytest.mark.parametrize("anchor", [QPointF(123.4, 321.7), QPointF(700.2, 90.6), QPointF(450, 350)])
def test_a_run_of_wheel_notches_keeps_the_point_under_the_cursor(view, zoom, anchor):
    view.set_zoom(zoom)
    view.centerOn(1000, 750)
    start = view.mapToScene(anchor.toPoint())
    worst = 0.0
    for delta in [120] * 8 + [-120] * 8:
        wheel(view, anchor, delta)
        worst = max(worst, screen_drift(view, anchor, start))
    assert worst <= 1.0
    assert view.zoom_factor == pytest.approx(zoom)


def test_a_pan_between_notches_anchors_on_the_new_point(view):
    anchor = QPointF(300.3, 200.6)
    view.set_zoom(2.0)
    wheel(view, anchor, 120)
    view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() + 57)  # a pan
    start = view.mapToScene(anchor.toPoint())
    wheel(view, anchor, 120)
    assert screen_drift(view, anchor, start) <= 1.0


def test_another_cursor_position_anchors_on_the_point_under_it(view):
    view.set_zoom(2.0)
    wheel(view, QPointF(100, 100), 120)
    anchor = QPointF(600.4, 500.2)
    start = view.mapToScene(anchor.toPoint())
    wheel(view, anchor, 120)
    assert screen_drift(view, anchor, start) <= 1.0


# ------------------------------------------------------------- a clamped zoom-out


def test_the_setup_clamps_the_zoom_out(small_view):
    """At the limit the view centres the scene: the anchored point is off the cursor."""
    view = small_view
    anchor = QPointF(60, 50)
    view.set_zoom(2.0, anchor)
    aimed = view.mapToScene(anchor.toPoint())
    _zoom_out_to_the_limit(view, anchor)
    assert _off_anchor(view, aimed, anchor.toPoint()) > 50


def test_a_zoom_after_a_clamped_zoom_out_keeps_the_point_shown_under_the_cursor(small_view):
    view = small_view
    anchor = QPointF(60, 50)
    view.set_zoom(2.0, anchor)
    _zoom_out_to_the_limit(view, anchor)
    shown = view.mapToScene(anchor.toPoint())  # what the still cursor is over now

    view.set_zoom(3.0, anchor)

    assert _off_anchor(view, shown, anchor.toPoint()) <= 1


def test_a_wheel_notch_after_a_clamped_zoom_out_starts_from_the_point_shown(small_view):
    view = small_view
    anchor = QPointF(60, 50)
    view.set_zoom(2.0, anchor)
    _zoom_out_to_the_limit(view, anchor)
    view.set_zoom(0.9, anchor)  # a zoom the view can anchor again (scrollable)
    shown = view.mapToScene(anchor.toPoint())

    wheel(view, anchor, 4 * 120)  # four notches in, one Ctrl+wheel event (animations are off)

    assert _off_anchor(view, shown, anchor.toPoint()) <= 1


def test_keyboard_steps_in_from_the_limit_zoom_about_the_centre_shown(small_view):
    """Ctrl+- to the limit, then Ctrl+=: the zoom goes on about what the centre shows."""
    view = small_view
    view.set_zoom(4.0)
    view.centerOn(40.0, 30.0)  # zoomed into the top-left corner
    for _ in range(10):
        view.zoom_step(-1)
    assert view.zoom_factor == pytest.approx(view.zoom_out_limit)
    centre = view.viewport().rect().center()
    shown = view.mapToScene(centre)

    view.set_zoom(3.0)

    assert _off_anchor(view, shown, centre) <= 1
