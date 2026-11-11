"""The skeleton overlay helpers (``app/gui/skeleton_overlay.py``), display only.

``overlay_indices`` (the line plus its one-pixel halo, UXE-13) feeds the skeleton layer
item, and ``SkeletonDecorations`` draws the endpoint markers of the Correct workspace:
markers and handles stop growing past 100 %; the markers item (``DotsItem``) repaints only
the markers that appeared or went (S2), and after such a partial repaint of its device cache
the screen equals a fresh item's; markers, handles and previews sit on pixel centres;
QPixmapCache holds the markers' device cache on a large viewport (its 10 MiB default is
smaller than one 2600 x 1500 pixmap). The cache mode is in
``test_skeleton_overlay_marker_cache.py``.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmapCache, QTransform
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QStyleOptionGraphicsItem,
)

from app.gui.skeleton_overlay import DotsItem, SkeletonDecorations
from app.gui.widgets import tokens
from tests.qt_teardown import delete_widgets


def test_overlay_halo_surrounds_the_line_only():
    from app.gui.skeleton_overlay import HALO_INDEX, LINE_INDEX, overlay_indices

    mask = np.zeros((9, 9), np.uint8)
    mask[4, 2:7] = 255
    idx = overlay_indices(mask)
    assert (idx == LINE_INDEX).sum() == 5 and np.array_equal(idx == LINE_INDEX, mask > 0)
    halo = idx == HALO_INDEX
    ring = cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0
    assert np.array_equal(halo, ring & (mask == 0))
    assert mask[4, 2] == 255  # input untouched


def test_touching_endpoint_markers_stay_filled(app):
    from PyQt6.QtWidgets import QGraphicsScene

    from app.gui.skeleton_overlay import SkeletonDecorations

    deco = SkeletonDecorations(QGraphicsScene())
    deco.set_endpoints([(10, 10), (12, 10)])
    assert deco.endpoints_item.path().contains(QPointF(11, 10))


def _painted(item, scale: float) -> int:
    """How many device pixels ``item`` covers when painted at zoom ``scale`` (centred)."""
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtWidgets import QStyleOptionGraphicsItem

    image = QImage(400, 400, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    painter.translate(200, 200)
    painter.scale(scale, scale)
    painter.translate(-10, -10)  # the dot at (10, 10) lands in the middle
    item.paint(painter, QStyleOptionGraphicsItem(), None)
    painter.end()
    alpha = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), np.uint8)
    return int((alpha.reshape(400, 400, 4)[:, :, 3] > 0).sum())


@pytest.mark.parametrize("which", ["endpoints", "highlight", "handles"])
def test_markers_and_handles_stop_growing_past_100_percent(app, which):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QGraphicsScene

    from app.gui.skeleton_overlay import SkeletonDecorations

    deco = SkeletonDecorations(QGraphicsScene())
    deco.set_endpoints([(10, 10)])
    deco.highlight_endpoint(QPoint(10, 10))
    deco.show_polyline([QPoint(10, 10)], None, smooth=False)
    item = getattr(deco, f"{which}_item")
    at_100, at_1600 = _painted(item, 1.0), _painted(item, 16.0)
    at_50 = _painted(item, 0.5)

    assert at_100 > 0 and at_1600 <= at_100 * 1.2  # a 128 px disc at 1600 % before
    assert at_50 < at_100  # below 100 % the dots shrink with the image, as before
    assert item.boundingRect().contains(item.path().boundingRect())


# ------------------------------------------------------------------- markers item
def _view(scene: QGraphicsScene, zoom: float, size=(400, 300)) -> QGraphicsView:
    view = QGraphicsView(scene)
    view.setFrameShape(QFrame.Shape.NoFrame)  # the scene fits exactly: nothing scrolls
    view.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view.setRenderHints(QPainter.RenderHint.Antialiasing)
    view.setSceneRect(QRectF(0, 0, size[0] / zoom, size[1] / zoom))
    view.setTransform(QTransform.fromScale(zoom, zoom))
    view.resize(*size)
    view.show()
    return view


def _screen(view: QGraphicsView) -> QImage:
    view.viewport().repaint()
    return view.viewport().grab().toImage()


def _markers(scene: QGraphicsScene) -> DotsItem:
    deco = SkeletonDecorations(scene)
    return deco, deco.endpoints_item


def _close(*views: QGraphicsView) -> None:
    delete_widgets(list(views))


@pytest.mark.parametrize("zoom", [0.5, 1.0, 3.0])
def test_a_partial_marker_repaint_equals_a_fresh_render(app, zoom):
    rng = np.random.default_rng(7)
    extent = (400 / zoom, 300 / zoom)
    first = np.column_stack(
        (rng.integers(0, int(extent[0]), 300), rng.integers(0, int(extent[1]), 300))
    )
    first = np.unique(first, axis=0)
    second = np.concatenate((first[3:], [[int(extent[0] // 2), int(extent[1] // 3)]]))

    scene = QGraphicsScene()
    deco, markers = _markers(scene)
    assert markers.cacheMode() == QGraphicsItem.CacheMode.DeviceCoordinateCache
    deco.set_endpoints(first)
    view = _view(scene, zoom)
    _screen(view)  # the device cache holds every marker now
    updates = []
    real_update = markers.update
    markers.update = lambda *rect: (updates.append(rect), real_update(*rect))[1]

    deco.set_endpoints(second)

    assert updates and all(rect for rect in updates)  # only rects, never the whole item
    edited = _screen(view)
    fresh_scene = QGraphicsScene()
    fresh_deco, _ = _markers(fresh_scene)
    fresh_deco.set_endpoints(second)
    fresh = _view(fresh_scene, zoom)
    assert edited == _screen(fresh)
    _close(view, fresh)


def test_unchanged_markers_repaint_nothing(app):
    scene = QGraphicsScene()
    deco, markers = _markers(scene)
    deco.set_endpoints([(10, 10), (40, 40)])
    updates = []
    markers.update = lambda *rect: updates.append(rect)
    deco.set_endpoints(np.array([[10, 10], [40, 40]]))
    assert updates == []


def test_markers_outside_the_bounds_grow_them(app):
    scene = QGraphicsScene()
    deco, markers = _markers(scene)
    deco.set_endpoints([(10, 10)])
    deco.set_endpoints([(10, 10), (300, 200)])
    assert markers.boundingRect().contains(markers.path().boundingRect())
    deco.set_endpoints([])
    assert markers.path().isEmpty() and markers.boundingRect().isEmpty()


def test_markers_handles_and_previews_sit_on_pixel_centres(app):
    scene = QGraphicsScene()
    deco = SkeletonDecorations(scene)
    deco.set_endpoints([(10, 20)])
    deco.highlight_endpoint(QPoint(10, 20))
    deco.show_polyline([QPoint(10, 20), QPoint(30, 20)], QPoint(30, 40), smooth=False)
    deco.show_connect(QPoint(10, 20), QPoint(50, 60))

    for item in (deco.endpoints_item, deco.highlight_item):
        assert item.path().boundingRect().center() == QPointF(10.5, 20.5)
    assert deco.handles_item.path().boundingRect().center() == QPointF(20.5, 20.5)
    path = deco.polyline_item.path()
    assert [path.elementAt(i) for i in range(path.elementCount())][0].x == 10.5
    assert path.currentPosition() == QPointF(30.5, 40.5)  # the rubber band ends on the cursor
    line = deco.connect_item.line()
    assert (line.x1(), line.y1(), line.x2(), line.y2()) == (10.5, 20.5, 50.5, 60.5)


def test_a_big_viewport_keeps_the_marker_cache(app):
    scene = QGraphicsScene()
    deco, _ = _markers(scene)
    deco.set_endpoints([(x, y) for x in range(5, 2600, 97) for y in range(5, 1500, 89)])
    before = QPixmapCache.cacheLimit()
    view = _view(scene, 1.0, size=(2600, 1500))
    _screen(view)
    try:
        assert QPixmapCache.cacheLimit() * 1024 >= 2600 * 1500 * 4
    finally:
        QPixmapCache.setCacheLimit(before)
        _close(view)


def _covered(item: DotsItem, scale: float, exposed: QRectF | None = None) -> int:
    image = QImage(200, 200, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.scale(scale, scale)
    option = QStyleOptionGraphicsItem()
    if exposed is not None:
        option.exposedRect = exposed
    item.paint(painter, option, None)
    painter.end()
    alpha = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), np.uint8)
    return int((alpha.reshape(200, 200, 4)[:, :, 3] > 0).sum())


def test_a_paint_draws_only_the_dots_it_exposes(app):
    item = DotsItem(4)
    item.setPen(QPen(QColor(tokens.BG_0), 1))
    item.set_points([(20.5, 20.5), (150.5, 150.5)])
    both = _covered(item, 1.0)
    one = _covered(item, 1.0, QRectF(0, 0, 60, 60))
    assert 0 < one < both and abs(both - 2 * one) <= 4


class _CountingPainter(QPainter):
    """Counts the dot sprites a paint blits."""

    def __init__(self, device):
        super().__init__(device)
        self.blits = 0

    def drawPixmap(self, *args):  # noqa: N802 - Qt name
        self.blits += 1
        super().drawPixmap(*args)


def test_a_paint_blits_only_the_dots_on_the_device(app):
    item = DotsItem(4)
    item.set_points([(20.5, 20.5), (150.5, 150.5), (900.5, 40.5), (40.5, -300.5)])
    image = QImage(200, 200, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = _CountingPainter(image)
    item.paint(painter, QStyleOptionGraphicsItem(), None)
    painter.end()
    assert painter.blits == 2  # the two dots off the image are skipped
