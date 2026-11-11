"""The photo layer (app/gui/canvas_layers/photo.py, contract C6, spec section 10).

``PhotoLayerItem`` draws the photo smooth at every zoom: an area pyramid level chosen from the
device scale (zoom x devicePixelRatio, ``app.gui.canvas_layers.device_scale``), drawn
bilinear, exposed rect only, in seam-free tiles made only where shown. Offscreen raster
viewport; thresholds from the zoom-render and contracts scouts.
"""

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QStyleOptionGraphicsItem

from app.gui.canvas_layers import device_scale
from app.gui.canvas_layers.photo import PhotoLayerItem
from tests.canvas_layers_support import (
    _blank,
    _inner,
    _paint_item,
    _photo_bgr,
)
from tests.render_metrics import (
    area_reference,
    equal_neighbour_fraction,
    mean_run,
    psnr,
    qimage_to_rgb,
    remap_reference,
    render_view,
    ring_std,
    stripes_image,
    zone_plate,
)

# Render tests measure the raster engine (offscreen has no GL anyway).
pytestmark = pytest.mark.usefixtures("app", "no_prompts", "gl_off")


def test_device_scale_counts_device_pixels_per_item_pixel():
    for dpr, scale, expected in ((1.0, 0.25, 0.25), (2.0, 0.25, 0.5), (2.0, 3.0, 6.0)):
        image = _blank(10, 10, dpr)
        painter = QPainter(image)
        painter.scale(scale, scale)
        assert device_scale(painter) == pytest.approx(expected)
        painter.rotate(30)
        assert device_scale(painter) == pytest.approx(expected)
        painter.end()


# ---------------------------------------------------------------------------- photo
def test_photo_at_800_percent_is_bilinear_not_blocks():
    bgr = _photo_bgr()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    item = PhotoLayerItem()
    item.set_image(bgr)
    render = render_view([item], 8.0, (640, 480))
    x0, y0, x1, y1 = _inner(render, 640, 480)
    shown = render.rgb[y0:y1, x0:x1]
    args = (render.zoom, render.dx, render.dy, 640, 480)
    bilinear = remap_reference(rgb, *args, cv2.INTER_LINEAR)[y0:y1, x0:x1]
    nearest = remap_reference(rgb, *args, cv2.INTER_NEAREST)[y0:y1, x0:x1]
    assert psnr(shown, bilinear) >= 40.0
    assert psnr(shown, bilinear) >= psnr(shown, nearest) + 5.0
    assert equal_neighbour_fraction(shown) < 0.4
    assert mean_run(shown) < 3.0
    assert item.level_painted() == 0


def test_photo_at_100_percent_equals_the_source():
    bgr = _photo_bgr()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    item = PhotoLayerItem()
    item.set_image(bgr)
    render = render_view([item], 1.0, (500, 400), centre=(300.0, 250.0))
    assert render.dx == int(render.dx) and render.dy == int(render.dy)  # integer scroll
    x0, y0, x1, y1 = _inner(render, 640, 480, margin=0)
    expected = remap_reference(rgb, 1.0, render.dx, render.dy, 500, 400, cv2.INTER_NEAREST)
    diff = np.abs(render.rgb[y0:y1, x0:x1].astype(int) - expected[y0:y1, x0:x1].astype(int))
    assert diff.max() <= 1


def test_photo_at_25_percent_is_as_alias_free_as_the_area_ideal():
    plate = zone_plate(1200)
    item = PhotoLayerItem()
    item.set_image(plate)
    render = render_view([item], 0.25, (400, 400))
    origin = (render.dx, render.dy)
    ideal = area_reference(plate, 0.25, render.dx, render.dy, 400, 400)
    assert ring_std(render.rgb, 0.25, origin) <= 1.2 * ring_std(ideal, 0.25, origin)
    assert item.level_painted() == 2


@pytest.mark.parametrize("zoom", [0.25, 0.33, 0.5])
def test_photo_minified_stripes_average_to_mid_grey(zoom):
    item = PhotoLayerItem()
    item.set_image(stripes_image(640, 480))
    render = render_view([item], zoom, (400, 300))
    x0, y0, x1, y1 = _inner(render, 640, 480)
    shown = render.rgb[y0:y1, x0:x1].astype(float)
    assert shown.mean() == pytest.approx(127.5, abs=12)
    assert shown.std() < 25  # no full-contrast moire bands


@pytest.mark.parametrize(("dpr", "level"), [(1.0, 2), (2.0, 1)])
def test_photo_level_follows_the_device_scale_including_hidpi(dpr, level):
    item = PhotoLayerItem()
    item.set_image(_photo_bgr())
    _paint_item(item, _blank(400, 300, dpr), 0.25)
    assert item.level_painted() == level


def test_photo_pixmap_is_level_zero_with_the_source_pixels():
    bgr = _photo_bgr(97, 61)
    item = PhotoLayerItem()
    item.set_image(bgr)
    pixmap = item.pixmap()
    assert (pixmap.width(), pixmap.height()) == (97, 61)
    assert np.array_equal(qimage_to_rgb(pixmap.toImage()), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    gray = bgr[:, :, 1].copy()
    item.set_image(gray)
    assert np.array_equal(qimage_to_rgb(item.pixmap().toImage())[:, :, 0], gray)
    assert item.image_size() == (97, 61)


def test_photo_keeps_the_array_by_reference_and_never_writes_it():
    bgr = _photo_bgr(64, 48)
    before = bgr.copy()
    bgr.setflags(write=False)
    item = PhotoLayerItem()
    item.set_image(bgr)
    for zoom in (0.2, 0.5, 1.0, 3.0):
        render_view([item], zoom, (120, 100))
    assert np.array_equal(bgr, before)


def test_photo_set_image_none_empties_the_item():
    item = PhotoLayerItem()
    item.set_image(_photo_bgr(64, 48))
    item.set_image(None)
    assert item.image_size() == (0, 0)
    assert item.boundingRect().isEmpty()
    assert item.pixmap().isNull()
    target = _blank(50, 50)
    _paint_item(item, target, 1.0)
    assert target == _blank(50, 50)  # nothing painted: still fully transparent


def test_photo_new_image_leaves_no_stale_level_at_25_percent():
    item = PhotoLayerItem()
    item.set_image(np.full((480, 640, 3), 255, np.uint8))
    first = render_view([item], 0.25, (300, 240))
    assert first.rgb.max() == 255 and item.level_painted() == 2
    item.set_image(np.zeros((480, 640, 3), np.uint8))
    second = render_view([item], 0.25, (300, 240))
    assert second.rgb.max() <= 1


@pytest.mark.parametrize("zoom", [0.37, 0.6, 3.3])
def test_photo_partial_repaint_matches_a_full_repaint(zoom):
    item = PhotoLayerItem()
    item.set_image(_photo_bgr(320, 240))
    size = (int(320 * zoom) + 2, int(240 * zoom) + 2)
    full = _blank(*size)
    _paint_item(item, full, zoom)
    part = _blank(*size)
    _paint_item(item, part, zoom)
    exposed = QRectF(101.3, 67.7, 41.0, 23.0)  # item px, like a moved brush ring
    _paint_item(item, part, zoom, exposed)
    a = qimage_to_rgb(full).astype(int)
    b = qimage_to_rgb(part).astype(int)
    assert np.abs(a - b).max() <= 1  # no seam at the exposed rect's border


def test_photo_restores_the_painter_hints():
    item = PhotoLayerItem()
    item.set_image(_photo_bgr(64, 48))
    image = _blank(80, 60)
    painter = QPainter(image)
    hints = QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
    painter.setRenderHints(hints)
    painter.scale(0.7, 0.7)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = item.boundingRect()
    item.paint(painter, option, None)
    assert painter.renderHints() == hints
    painter.end()


def _photo(image: np.ndarray, tile: int) -> PhotoLayerItem:
    item = PhotoLayerItem()
    item.TILE = tile  # 64: levels over 256 x 256 px are drawn in 64 px tiles
    item.set_image(image)
    return item


@pytest.mark.parametrize("zoom", [0.23, 0.37, 0.6, 1.0, 2.35, 5.0])
def test_photo_tiles_render_like_one_image(zoom):
    image = _photo_bgr(660, 500)  # levels 0 and 1 tiled, level 2 one pixmap
    size = (int(660 * zoom) + 2, int(500 * zoom) + 2)
    tiled, whole = _blank(*size), _blank(*size)
    _paint_item(_photo(image, 64), tiled, zoom)
    _paint_item(_photo(image, 4096), whole, zoom)
    diff = np.abs(qimage_to_rgb(tiled).astype(int) - qimage_to_rgb(whole).astype(int))
    assert diff.max() <= 1  # no seam, gap or overlap at tile borders


@pytest.mark.parametrize("zoom", [0.37, 3.3])
def test_photo_tiled_partial_repaint_matches_a_full_repaint(zoom):
    item = _photo(_photo_bgr(660, 500), 64)
    size = (int(660 * zoom) + 2, int(500 * zoom) + 2)
    full, part = _blank(*size), _blank(*size)
    _paint_item(item, full, zoom)
    _paint_item(item, part, zoom)
    _paint_item(item, part, zoom, QRectF(50.3, 40.7, 41.0, 49.0))  # across tile borders
    assert np.abs(qimage_to_rgb(full).astype(int) - qimage_to_rgb(part).astype(int)).max() <= 1


def test_photo_converts_only_the_tiles_it_shows():
    item = _photo(_photo_bgr(660, 500), 64)
    assert item.built_tiles() == {}  # set_image converts nothing
    render_view([item], 2.0, (100, 80), centre=(32.0, 32.0))
    assert set(item.built_tiles()) == {(0, 0, 0)}  # (level, tx, ty)
    render_view([item], 0.5, (100, 80), centre=(32.0, 32.0))
    level_one = {key for key in item.built_tiles() if key[0] == 1}
    assert (1, 0, 0) in level_one and len(level_one) < 6 * 4  # of the 6 x 4 level-1 tiles
