"""The photo warms the pyramid levels one zoom step away when the view is idle
(app/gui/canvas_layers/photo.py).

After a paint, what a zoom step (``WARM_STEP``, the ladder's largest ratio, in or out) would
draw is warmed once the view has been still for ``WARM_DELAY_MS``: the tiles of the levels
drawn at that zoom, over the area such a step can show (inside the view for a step in; the
view grown by half its size on every side for a step out, whatever the anchor), converted
in idle slices of ``WARM_BUDGET_MS``. On an OpenGL paint engine the next frames draw those tiles
once, invisibly, so their textures are uploaded before the first frame that shows them.
A new image drops the warm-up of the old one.

Those upload frames come wherever the view is: a view repaints only for an update that
meets its viewport, so the request names a pixel of the photo on screen (the top-left
pixel it named is off screen once the view is zoomed into another part, and then no
frame came and every tile made ahead waited for the first frame that showed it).
"""

import time

import numpy as np
import pytest
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView, QStyleOptionGraphicsItem

from app.gui import canvas_gl
from app.gui.canvas_layers import photo as photo_module
from app.gui.canvas_layers.photo import PhotoLayerItem, TiledLevels
from tests.canvas_view_support import FramelessView
from tests.qt_teardown import flush_deleted

pytestmark = pytest.mark.usefixtures("app")


def photo(w: int = 1400, h: int = 1000, tile: int = 64) -> PhotoLayerItem:
    """An item whose levels are tiled at ``tile`` px (level 0 to 2 here)."""
    rng = np.random.default_rng(1)
    item = PhotoLayerItem()
    item.TILE = tile
    item.set_image(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    return item


def paint(item: PhotoLayerItem, zoom: float, exposed: QRectF) -> None:
    """Paint ``exposed`` (item px) at ``zoom`` into a raster image, as a view would."""
    size = (int(exposed.width() * zoom) + 2, int(exposed.height() * zoom) + 2)
    image = QImage(*size, QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(image)
    painter.scale(zoom, zoom)
    painter.translate(-exposed.left(), -exposed.top())
    option = QStyleOptionGraphicsItem()
    option.exposedRect = exposed
    item.paint(painter, option, None)
    painter.end()


def levels_built(item: PhotoLayerItem) -> set[int]:
    return {key[0] for key in item.built_tiles()}


def warm_all(item: PhotoLayerItem) -> None:
    for _ in range(1000):
        if not item.warm_levels(budget_ms=1000.0):
            return
    raise AssertionError("the warm-up never ended")


def test_a_step_into_a_coarser_level_is_warmed_ahead():
    item = photo()
    area = QRectF(300, 200, 400, 300)
    paint(item, 0.9, area)  # level 0 (the blend starts at 0.76)
    assert levels_built(item) == {0}
    warm_all(item)
    assert levels_built(item) == {0, 1}  # 0.9 / 1.5 = 0.6: level 0 blended with level 1
    before = item.built_tiles()
    paint(item, 0.6, QRectF(250, 160, 500, 380))  # zoomed out about the view: inside the grown area
    assert item.built_tiles() == before  # nothing converted in the frame


def test_a_step_into_a_finer_level_is_warmed_ahead():
    item = photo()
    area = QRectF(300, 200, 600, 400)
    paint(item, 0.4, area)  # level 1
    assert levels_built(item) == {1}
    warm_all(item)
    assert levels_built(item) == {0, 1}  # 0.4 * 1.5 = 0.6: level 0 blended with level 1
    before = item.built_tiles()
    paint(item, 0.6, QRectF(400, 250, 400, 300))  # zoomed in: inside the view
    paint(item, 0.4 / 1.5, QRectF(0, 0, 900, 600))  # zoomed out 1.5x about the view's corner
    assert item.built_tiles() == before


def test_nothing_is_warmed_when_a_step_stays_on_the_same_levels():
    item = photo(640, 480)
    paint(item, 2.0, QRectF(0, 0, 640, 480))  # 1.33 and 3.0: level 0 only
    assert item.warm_levels(budget_ms=1000.0) is False
    assert levels_built(item) == {0}


def test_the_warm_up_runs_by_itself_once_the_view_is_idle():
    item = photo()
    scene = QGraphicsScene()
    scene.addItem(item)
    paint(item, 0.9, QRectF(300, 200, 400, 300))
    assert levels_built(item) == {0}
    deadline = time.perf_counter() + 3.0
    while levels_built(item) != {0, 1} and time.perf_counter() < deadline:
        QTest.qWait(20)
    assert levels_built(item) == {0, 1}
    scene.removeItem(item)


@pytest.mark.wallclock
def test_each_idle_slice_keeps_to_its_budget():
    item = photo(4000, 3000, tile=64)  # many small tiles to warm
    paint(item, 0.9, QRectF(0, 0, 4000, 3000))
    t0 = time.perf_counter()
    assert item.warm_levels(budget_ms=2.0) is True  # more remains
    assert (time.perf_counter() - t0) * 1000 < 50  # one slice, not the whole warm-up


def test_a_new_image_drops_the_warm_up_of_the_old_one():
    item = photo()
    paint(item, 0.9, QRectF(300, 200, 400, 300))
    item.set_image(np.zeros((480, 640, 3), np.uint8))
    assert item.warm_levels(budget_ms=1000.0) is False
    assert item.built_tiles() == {}


def test_tiles_made_ahead_are_uploaded_once_on_a_gl_engine_only():
    levels = TiledLevels(_tile_maker(), tile=64, margin=2, single_pixels=64 * 64)
    size, image = (300, 200), (300, 200)
    assert levels.warm(0, size, image, QRectF(0, 0, 300, 200), time.perf_counter() + 10) is False
    made = levels.built()
    assert len(made) == 5 * 4 and levels.pending_uploads() == len(made)

    image_ = QImage(10, 10, QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(image_)
    assert levels.upload(painter, 6) is False  # a raster engine uploads nothing
    assert levels.pending_uploads() == len(made)
    painter.end()

    painter = QPainter(image_)
    levels.draw(painter, 0, size, image, QRectF(0, 0, 60, 60))  # tiles drawn for real
    painter.end()
    assert levels.pending_uploads() < len(made)


def _tile_maker():
    from PyQt6.QtGui import QPixmap

    def make(level, x0, y0, x1, y1):
        pixmap = QPixmap(x1 - x0, y1 - y0)
        pixmap.fill()
        return pixmap

    return make


# ------------------------------------------------------------- upload frames in view

W, H = 1400, 1000


@pytest.fixture
def page(app, monkeypatch):
    """A 400 x 300 view over a 1400 x 1000 photo tiled at 64 px, updated as a GL viewport
    is (the whole viewport per frame); engines count as GL."""
    monkeypatch.setenv(canvas_gl.ENV_VAR, "0")
    monkeypatch.setattr(photo_module, "_is_gl", lambda painter: True)
    rng = np.random.default_rng(3)
    item = PhotoLayerItem()
    item.TILE = 64
    item.set_image(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
    view = FramelessView()
    view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
    scene = QGraphicsScene(view)
    scene.addItem(item)
    view.setScene(scene)
    view.set_content_rect(QRectF(0, 0, W, H))
    view.resize(400, 300)
    yield view, item
    scene.removeItem(item)
    view.close()
    view.deleteLater()
    del view, scene
    flush_deleted()


def _show(app, view, zoom: float, centre: tuple[float, float]) -> None:
    view.set_zoom(zoom)
    view.centerOn(QPointF(*centre))
    view.show()
    app.processEvents()
    view.centerOn(QPointF(*centre))
    view.viewport().repaint()


def _settle(item: PhotoLayerItem, shown: int, timeout_s: float = 5.0) -> None:
    """Wait until the warm-up made tiles beyond the ``shown`` ones and none waits."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        QTest.qWait(20)
        if len(item.built_tiles()) > shown and item._levels.pending_uploads() == 0:
            return


def test_tiles_made_ahead_are_uploaded_when_the_image_corner_is_off_screen(app, page):
    view, item = page
    _show(app, view, 0.9, (W / 2, H / 2))  # the corner (0, 0) is far off screen
    assert not view.mapToScene(view.viewport().rect()).boundingRect().contains(0.5, 0.5)
    shown = len(item.built_tiles())
    _settle(item, shown)
    assert len(item.built_tiles()) > shown  # the warm-up made tiles ahead
    assert item._levels.pending_uploads() == 0  # and idle frames uploaded them


def test_tiles_made_ahead_are_uploaded_when_the_corner_is_on_screen(app, page):
    view, item = page
    _show(app, view, 0.25, (W / 2, H / 2))  # the whole photo in view
    shown = len(item.built_tiles())
    _settle(item, shown)
    assert len(item.built_tiles()) > shown
    assert item._levels.pending_uploads() == 0
