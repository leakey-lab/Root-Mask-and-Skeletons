"""The optional pixel grid of the canvas views (app/gui/canvas_view.py).

Off by default. When on, it shows from ``PIXEL_GRID_MIN_ZOOM`` (800 %) up: a one-device-pixel
line on the first device column (row) of every image pixel, at the device edges where the
exact-pixel layers start each pixel block, so the lines sit on the block edges at any zoom.
Display only: it changes no item and no image. ``toggle_pixel_grid`` is what the shortcut
calls; ``pixel_grid_changed`` reports the new state.

Inside a window, a raster viewport paints into the window's backing store, moved there by
its offset in the window (Qt's paint redirection). The grid lines must still sit on the
device columns where the exact-pixel layer starts each pixel block, at every
devicePixelRatio: the grid must not add that offset a second time (it did: every line
landed ``offset x dpr`` device px to the right and below, with no grid over the top-left).
"""

import math

import numpy as np
import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from app.gui import canvas_gl
from app.gui.canvas_layers.raster import RasterLayerItem
from app.gui.canvas_view import CanvasViewMixin, draw_pixel_grid
from tests.canvas_view_support import FramelessView
from tests.qt_teardown import delete_widgets, flush_deleted
from tests.render_metrics import qimage_to_rgb

W, H = 64, 48
FLAT = QColor(90, 90, 90)


class _View(CanvasViewMixin, QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_canvas()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)


@pytest.fixture
def make_view(app, monkeypatch):
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    made = []

    def make(item=None):
        view = _View()
        scene = QGraphicsScene(view)
        if item is None:
            pm = QPixmap(W, H)
            pm.fill(FLAT)
            item = QGraphicsPixmapItem(pm)
        scene.addItem(item)
        view.setScene(scene)
        view.set_content_rect(QRectF(0, 0, W, H))
        view.resize(320, 240)
        view.show()
        app.processEvents()
        made.append((view, item))
        return view

    yield make
    views = []
    for view, item in made:
        view.scene().removeItem(item)
        views.append(view)
    made.clear()
    delete_widgets(views)


def grab(view, zoom: float, centre=(W / 2, H / 2)) -> np.ndarray:
    view.set_zoom(zoom)
    view.centerOn(*centre)
    view.viewport().repaint()
    QApplication.processEvents()
    return qimage_to_rgb(view.viewport().grab().toImage()).astype(int)


def inside(view, rgb: np.ndarray) -> np.ndarray:
    """The part of ``rgb`` inside the image, 2 device px in from its edges."""
    rect = view.mapFromScene(QRectF(0, 0, W, H)).boundingRect()
    y0, x0 = max(0, rect.top() + 2), max(0, rect.left() + 2)
    y1, x1 = min(rgb.shape[0], rect.bottom() - 2), min(rgb.shape[1], rect.right() - 2)
    return rgb[y0:y1, x0:x1]


def off_flat(block: np.ndarray) -> np.ndarray:
    """Per pixel: differs from the flat grey."""
    return np.abs(block - np.array([90, 90, 90])).sum(axis=2) > 6


def test_the_grid_is_off_by_default(make_view):
    view = make_view()
    assert view.pixel_grid() is False
    assert not off_flat(inside(view, grab(view, 10.0))).any()


def test_the_grid_marks_the_first_device_column_and_row_of_every_pixel(make_view):
    view = make_view()
    seen = []
    view.pixel_grid_changed.connect(seen.append)
    assert view.toggle_pixel_grid() is True and seen == [True]

    rgb = grab(view, 10.0)
    ox, oy = view.mapFromScene(0.0, 0.0).x(), view.mapFromScene(0.0, 0.0).y()
    marked = off_flat(rgb)
    rows = range(max(0, oy) + 1, min(rgb.shape[0], oy + H * 10) - 1)
    cols = range(max(0, ox) + 1, min(rgb.shape[1], ox + W * 10) - 1)
    for x in cols:
        on_line = (x - ox) % 10 == 0
        column = marked[rows.start : rows.stop, x]
        if on_line:
            assert column.all(), x
        else:  # only where a horizontal line crosses
            assert column.sum() == sum((y - oy) % 10 == 0 for y in rows), x


def test_below_800_percent_the_grid_does_not_show(make_view):
    view = make_view()
    view.set_pixel_grid(True)
    assert not off_flat(inside(view, grab(view, 6.0))).any()
    assert off_flat(inside(view, grab(view, CanvasViewMixin.PIXEL_GRID_MIN_ZOOM))).any()


def test_the_grid_is_display_only(make_view):
    view = make_view()
    items = list(view.scene().items())
    view.set_pixel_grid(True)
    grab(view, 12.0)
    view.set_pixel_grid(False)
    assert list(view.scene().items()) == items
    assert not off_flat(inside(view, grab(view, 12.0))).any()
    assert view.toggle_pixel_grid() is True and view.toggle_pixel_grid() is False


def _checkerboard_canvas() -> QImage:
    yy, xx = np.mgrid[0:H, 0:W]
    on = (xx + yy) % 2 == 0
    bgra = np.zeros((H, W, 4), np.uint8)
    bgra[on] = (40, 200, 40, 255)
    bgra[~on] = (200, 40, 200, 255)
    image = QImage(bgra.data, W, H, W * 4, QImage.Format.Format_ARGB32_Premultiplied)
    return image.copy()


@pytest.mark.parametrize("zoom", [8.0, 10.0, 12.0, 16.0, 8.7, 11.3])
def test_grid_lines_sit_on_the_block_edges_of_an_exact_pixel_layer(make_view, zoom):
    """Exact at the ladder's integer stops (every stop from 200 %). At a wheel zoom the
    raster engine's fixed-point nearest walk can move a block edge by one device column,
    so there each line is within one column of an edge."""
    canvas = _checkerboard_canvas()
    item = RasterLayerItem(lambda: canvas)
    view = make_view(item)
    plain = grab(view, zoom, centre=(W / 2 + 0.3, H / 2 + 0.2))
    view.set_pixel_grid(True)
    gridded = grab(view, zoom, centre=(W / 2 + 0.3, H / 2 + 0.2))

    # Along the middle of an image row: block edges are the device columns where the
    # checkerboard colour changes; grid lines are the columns the grid changed.
    y = view.mapFromScene(0.0, 20.5).y()
    x0 = max(1, view.mapFromScene(0.0, 0.0).x() + 1)
    x1 = min(plain.shape[1], view.mapFromScene(float(W), 0.0).x())
    row, lined = plain[y], gridded[y]
    edges = {x for x in range(x0, x1) if (row[x] != row[x - 1]).any()}
    lines = {x for x in range(x0, x1) if np.abs(lined[x] - row[x]).sum() > 6}
    assert len(edges) > 10
    if zoom.is_integer():
        assert lines == edges
    else:
        assert len(lines) == len(edges)
        assert all(min(abs(x - e) for e in edges) <= 1 for x in lines)


@pytest.mark.parametrize("dpr", [1.0, 2.0, 1.5])
def test_the_lines_are_one_device_pixel_wide_at_any_device_pixel_ratio(app, dpr):
    """At zoom 8 on a devicePixelRatio ``dpr`` device a pixel is ``8 * dpr`` device px: one
    grid column and row per pixel, one device pixel each, on the pixel's first column."""
    size = 400
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(FLAT)
    painter = QPainter(image)
    painter.scale(8, 8)
    colour = QColor(255, 255, 255, 255)
    draw_pixel_grid(painter, QRectF(0, 0, 10, 10), colour)
    painter.end()

    rgb = qimage_to_rgb(image).astype(int)
    white = (rgb == 255).all(axis=2)
    step = 8 * dpr
    expected = {math.ceil(k * step - 0.5) for k in range(10)}  # the 10th edge ends the area
    middle = int(step / 2)  # a device row inside the first pixel row
    assert {x for x in range(size) if white[middle, x]} == expected
    assert {y for y in range(size) if white[y, middle]} == expected


# ------------------------------------------------------------- inside a window

OFFSET = (37, 23)  # the view's position in its window
VIEW = (320, 240)


@pytest.fixture
def window(app, monkeypatch):
    """A window holding the view at ``OFFSET``, the checkerboard in exact pixels."""
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    canvas = _checkerboard_canvas()
    host = QWidget()
    host.resize(500, 400)
    view = FramelessView(host)
    view.setGeometry(*OFFSET, *VIEW)
    scene = QGraphicsScene(view)
    item = RasterLayerItem(lambda: canvas)
    scene.addItem(item)
    view.setScene(scene)
    view.set_content_rect(QRectF(0, 0, W, H))
    host.show()
    app.processEvents()
    yield host, view
    scene.removeItem(item)
    host.close()
    host.deleteLater()
    del host, view, scene
    flush_deleted()


def _render(app, host, view, dpr: float, zoom: float, grid: bool) -> np.ndarray:
    """The whole window at ``dpr`` (the view painted through the window's redirection)."""
    view.set_pixel_grid(grid)
    view.set_zoom(zoom)
    view.centerOn(W / 2 + 0.3, H / 2 + 0.2)
    app.processEvents()
    image = QImage(
        round(host.width() * dpr), round(host.height() * dpr), QImage.Format.Format_ARGB32
    )
    image.setDevicePixelRatio(dpr)
    image.fill(QColor(0, 0, 0))
    host.render(image)
    return qimage_to_rgb(image).astype(int)


@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
@pytest.mark.parametrize("zoom", [8.0, 10.0, 12.0])
def test_grid_lines_sit_on_the_block_edges_inside_a_window(app, window, dpr, zoom):
    host, view = window
    plain = _render(app, host, view, dpr, zoom, grid=False)
    lined = _render(app, host, view, dpr, zoom, grid=True)

    # A device row through the middle of image row 20; the view's columns, 2 px in.
    row_y = math.floor((OFFSET[1] + view.mapFromScene(0.0, 20.5).y() + 0.25) * dpr)
    first = math.ceil((OFFSET[0] + 2) * dpr)
    last = math.floor((OFFSET[0] + VIEW[0] - 2) * dpr)
    row, gridded = plain[row_y], lined[row_y]
    edges = {x for x in range(first, last) if (row[x] != row[x - 1]).any()}
    lines = {x for x in range(first, last) if np.abs(gridded[x] - row[x]).sum() > 6}

    assert len(edges) > 10
    if (zoom * dpr).is_integer():
        assert lines == edges
    else:  # the raster engine's nearest walk may move a block edge by one column
        assert len(lines) == len(edges)
        assert all(min(abs(x - e) for e in edges) <= 1 for x in lines)


def test_the_grid_covers_the_top_left_of_the_view_inside_a_window(app, window):
    """No band without grid lines at the view's top-left (the doubled offset clipped it)."""
    host, view = window
    lined = _render(app, host, view, 1.0, 10.0, grid=True)
    plain = _render(app, host, view, 1.0, 10.0, grid=False)
    changed = np.abs(lined - plain).sum(axis=2) > 6
    x0, y0 = OFFSET
    band = changed[y0 : y0 + 20, x0 : x0 + VIEW[0]]  # the view's top 20 device rows
    assert band.any(axis=1).all()  # every row of the band crosses vertical grid lines
