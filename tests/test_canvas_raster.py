"""The editable raster layer (app/gui/canvas_layers/raster.py, contract C6, spec section 10).

``RasterLayerItem`` draws an editable ARGB canvas: exact pixels at device scale >= 1 from
256 px tiles (a stroke re-keys only the tiles it touched), area coverage below 1 (a 1 px
stroke fades instead of vanishing). Offscreen raster viewport; thresholds from the
zoom-render and contracts scouts.
"""

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QGraphicsScene, QStyleOptionGraphicsItem

from app.gui.canvas_layers.photo import PhotoLayerItem
from app.gui.canvas_layers.raster import RasterLayerItem
from app.gui.widgets import tokens
from tests.canvas_layers_support import (
    SEED,
    _blank,
    _inner,
    _paint_item,
    _photo_bgr,
    _premultiplied,
)
from tests.render_metrics import (
    area_reference,
    equal_neighbour_fraction,
    mean_run,
    psnr,
    remap_reference,
    render_view,
)

# Render tests measure the raster engine (offscreen has no GL anyway).
pytestmark = pytest.mark.usefixtures("app", "no_prompts", "gl_off")


# --------------------------------------------------------------------------- raster
class _Canvas:
    """Holds the canvas QImage the way PaintCanvas does (the item reads it at paint time)."""

    def __init__(self, image: QImage | None):
        self.image = image


def _bgra_to_image(bgra: np.ndarray) -> QImage:
    """An owning ARGB32_Premultiplied QImage (the PaintCanvas format) of straight BGRA bytes."""
    bgra = np.ascontiguousarray(bgra)
    h, w = bgra.shape[:2]
    straight = QImage(bgra.data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()
    return straight.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)


def _encoded_canvas(w=700, h=600) -> tuple[np.ndarray, QImage]:
    """Opaque canvas whose colour encodes its own pixel position (R, B: x, y mod 256; G: the
    256 px tile of x and y), so a grab can be decoded back to image coordinates."""
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    bgra = np.zeros((h, w, 4), np.uint8)
    bgra[..., 0] = ys % 256
    bgra[..., 1] = (xs // 256) * 64 + (ys // 256) * 16
    bgra[..., 2] = xs % 256
    bgra[..., 3] = 255
    return bgra, _bgra_to_image(bgra)


def _decode(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    x = (g // 64) * 256 + r
    y = ((g % 64) // 16) * 256 + b
    return x, y, g % 16 == 0


def _mask_canvas(w=640, h=480) -> np.ndarray:
    """Straight BGRA mask canvas: root-like strokes in the mask colour, transparent elsewhere."""
    plane = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(SEED)
    for _ in range(14):
        pts = rng.integers(0, (w, h), size=(4, 2)).astype(np.int32)
        cv2.polylines(plane, [pts], False, 1, int(rng.integers(1, 9)))
    cv2.circle(plane, (w // 3, h // 2), 40, 1, -1)
    colour = QColor(tokens.MASK)
    bgra = np.zeros((h, w, 4), np.uint8)
    bgra[plane > 0] = (colour.blue(), colour.green(), colour.red(), 255)
    return bgra


def _runs(seq: np.ndarray) -> np.ndarray:
    change = np.flatnonzero(np.diff(seq)) + 1
    bounds = np.concatenate([[0], change, [len(seq)]])
    return np.diff(bounds)[1:-1]  # drop the partial first / last runs


def test_raster_pixels_are_exact_at_integer_zooms():
    bgra, image = _encoded_canvas()
    item = RasterLayerItem(lambda: image)
    rgb = np.ascontiguousarray(bgra[..., 2::-1])
    for zoom, centre in ((1.0, (350.0, 300.0)), (4.0, (256.0, 256.0)), (20.0, (511.7, 255.2))):
        render = render_view([item], zoom, (640, 480), centre=centre)
        x0, y0, x1, y1 = _inner(render, 700, 600, margin=0)
        expected = remap_reference(rgb, zoom, render.dx, render.dy, 640, 480, cv2.INTER_NEAREST)
        assert np.array_equal(render.rgb[y0:y1, x0:x1], expected[y0:y1, x0:x1]), zoom


def test_raster_zoomed_pixels_stay_crisp_under_a_smooth_painter():
    """Nearest-neighbour even when the painter asks for smooth pixmap scaling."""
    src = QImage(2, 1, QImage.Format.Format_ARGB32_Premultiplied)
    src.setPixelColor(QPoint(0, 0), QColor(0, 0, 0))
    src.setPixelColor(QPoint(1, 0), QColor(255, 255, 255))
    scene = QGraphicsScene()
    scene.addItem(RasterLayerItem(lambda: src))
    img = QImage(16, 8, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    scene.render(painter, QRectF(0, 0, 16, 8), QRectF(0, 0, 2, 1))
    after = painter.testRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.end()
    assert img.pixelColor(7, 4) == QColor(0, 0, 0)  # last column of the black pixel
    assert img.pixelColor(8, 4) == QColor(255, 255, 255)
    assert after  # the painter's own hint is restored for later items


@pytest.mark.parametrize("zoom", [1.288, 2.35, 3.7])
@pytest.mark.parametrize("centre", [(256.0, 256.0), (512.3, 300.7)])
def test_raster_fractional_zooms_show_every_pixel_in_order_without_blending(zoom, centre):
    bgra, image = _encoded_canvas()
    item = RasterLayerItem(lambda: image)
    render = render_view([item], zoom, (900, 700), centre=centre)  # Antialiasing view
    x0, y0, x1, y1 = _inner(render, 700, 600, margin=1)
    xs, ys, clean = _decode(render.rgb[y0:y1, x0:x1])
    assert clean.all()  # no blended colour anywhere, tile borders included
    shown = bgra[ys, xs][..., 2::-1]
    assert np.array_equal(shown, render.rgb[y0:y1, x0:x1])  # each pixel is a canvas pixel
    lo, hi = int(np.floor(zoom)), int(np.ceil(zoom))
    for seq in (xs[len(xs) // 2], ys[:, ys.shape[1] // 2], xs[3], ys[:, -4]):
        steps = np.diff(seq)
        assert set(np.unique(steps).tolist()) <= {0, 1}  # monotonic, no image pixel skipped
        runs = _runs(seq)
        assert runs.min() >= lo and runs.max() <= hi
    # the columns / rows either side of every tile border are shown
    for border in (255, 256, 511, 512):
        if xs.min() < border < xs.max():
            assert border in xs[len(xs) // 2]
        if ys.min() < border < ys.max():
            assert border in ys[:, ys.shape[1] // 2]


def test_raster_minified_coverage_matches_the_area_reference():
    bgra = _mask_canvas()
    image = _bgra_to_image(bgra)
    item = RasterLayerItem(lambda: image)
    render = render_view([item], 0.25, (300, 240))
    over_black = np.ascontiguousarray(cv2.cvtColor(bgra[..., :3], cv2.COLOR_BGR2RGB))
    reference = area_reference(over_black, 0.25, render.dx, render.dy, 300, 240)
    x0, y0, x1, y1 = _inner(render, 640, 480, margin=2)
    assert psnr(render.rgb[y0:y1, x0:x1], reference[y0:y1, x0:x1]) >= 60.0


@pytest.mark.parametrize("row", [240, 241, 242, 243])
@pytest.mark.parametrize("zoom", [0.25, 0.33, 0.5, 0.75])
def test_raster_thin_line_stays_visible_when_minified(row, zoom):
    bgra = np.zeros((480, 640, 4), np.uint8)
    colour = QColor(tokens.MASK)
    bgra[row, 20:620] = (colour.blue(), colour.green(), colour.red(), 255)
    image = _bgra_to_image(bgra)
    item = RasterLayerItem(lambda: image)
    shown = render_view([item], zoom, (500, 400), centre=(320.0, 240.0))
    item.setVisible(False)
    hidden = render_view([item], zoom, (500, 400), centre=(320.0, 240.0))
    diff = np.abs(shown.rgb.astype(int) - hidden.rgb.astype(int)).max(axis=2)
    y_dev = int(row * zoom + shown.dy)
    band = diff[max(0, y_dev - 2) : y_dev + 3]
    cols = range(int(np.ceil(40 * zoom + shown.dx)), int(np.floor(600 * zoom + shown.dx)))
    visible = [band[:, c].max() >= 1 for c in cols]
    assert np.mean(visible) >= 0.95


def test_raster_canvas_changed_rekeys_only_the_touched_tiles():
    _bgra, image = _encoded_canvas()
    canvas = _Canvas(image)
    item = RasterLayerItem(lambda: canvas.image)
    render_view([item], 1.0, (900, 700), centre=(350.0, 300.0))  # every tile shown
    before = item.built_tiles()
    assert set(before) == {(tx, ty) for tx in range(3) for ty in range(3)}
    painter = QPainter(canvas.image)
    painter.fillRect(QRect(250, 10, 10, 5), QColor(255, 0, 255))
    painter.end()
    item.canvas_changed(QRect(250, 10, 10, 5))  # crosses the border of tiles (0, 0) / (1, 0)
    after_render = render_view([item], 1.0, (900, 700), centre=(350.0, 300.0))
    after = item.built_tiles()
    changed = {key for key in before if after.get(key) != before[key]}
    assert changed == {(0, 0), (1, 0)}
    x, y = int(252 + after_render.dx), int(12 + after_render.dy)
    assert tuple(after_render.rgb[y, x]) == (255, 0, 255)  # the new pixels are shown


def test_raster_canvas_replaced_rebuilds_every_tile_and_follows_the_size():
    _bgra, image = _encoded_canvas()
    canvas = _Canvas(image)
    item = RasterLayerItem(lambda: canvas.image)
    render_view([item], 1.0, (900, 700), centre=(350.0, 300.0))
    before = item.built_tiles()
    canvas.image = _bgra_to_image(np.full((600, 700, 4), 255, np.uint8))
    item.canvas_replaced()
    render = render_view([item], 1.0, (900, 700), centre=(350.0, 300.0))
    after = item.built_tiles()
    assert set(after) == set(before) and all(after[k] != before[k] for k in before)
    assert render.rgb[int(render.dy + 300), int(render.dx + 350)].tolist() == [255, 255, 255]
    canvas.image = _bgra_to_image(np.zeros((20, 30, 4), np.uint8))
    item.canvas_replaced()
    assert item.boundingRect() == QRectF(0, 0, 30, 20)
    canvas.image = None
    item.canvas_replaced()
    assert item.boundingRect().isEmpty()
    assert item.built_tiles() == {}


def test_raster_tiles_are_only_built_where_shown():
    _bgra, image = _encoded_canvas()
    item = RasterLayerItem(lambda: image)
    render_view([item], 4.0, (400, 300), centre=(100.0, 100.0))  # inside tile (0, 0)
    assert set(item.built_tiles()) == {(0, 0)}


def test_raster_painting_the_source_after_display_does_not_copy_it():
    _bgra, image = _encoded_canvas(64, 48)
    canvas = _Canvas(image)
    item = RasterLayerItem(lambda: canvas.image)
    for zoom in (1.0, 0.4):  # nearest tiles and minified coverage
        render_view([item], zoom, (100, 80))
    serial = canvas.image.cacheKey() >> 32
    for i in range(5):
        painter = QPainter(canvas.image)
        painter.fillRect(i, i, 2, 2, QColor(0, 255, 0))
        painter.end()
        item.canvas_changed(QRect(i, i, 2, 2))
    assert canvas.image.cacheKey() >> 32 == serial  # same data, no detach copy
    render = render_view([item], 1.0, (100, 80), centre=(32.0, 24.0))
    assert tuple(render.rgb[int(render.dy + 4), int(render.dx + 4)]) == (0, 255, 0)


def test_raster_minified_render_follows_canvas_changes():
    bgra = _mask_canvas()
    canvas = _Canvas(_bgra_to_image(bgra))
    item = RasterLayerItem(lambda: canvas.image)
    render_view([item], 0.3, (300, 240))  # builds the coverage
    painter = QPainter(canvas.image)
    painter.fillRect(QRect(401, 97, 57, 33), QColor(tokens.SKEL))
    painter.end()
    item.canvas_changed(QRect(401, 97, 57, 33))
    updated = render_view([item], 0.3, (300, 240))
    fresh = render_view([RasterLayerItem(lambda: canvas.image)], 0.3, (300, 240))
    assert np.array_equal(updated.rgb, fresh.rgb)


def _raster(image: QImage, cover_tile: int) -> RasterLayerItem:
    item = RasterLayerItem(lambda: image)
    item.COVER_TILE = cover_tile  # 64: coverage levels over 128 x 128 px are tiled
    return item


@pytest.mark.parametrize("zoom", [0.3, 0.45, 0.6, 0.9])
def test_raster_tiled_coverage_renders_like_one_image(zoom):
    image = _bgra_to_image(_mask_canvas(330, 250))
    size = (int(330 * zoom) + 2, int(250 * zoom) + 2)
    tiled, whole = _blank(*size), _blank(*size)
    _paint_item(_raster(image, 64), tiled, zoom)
    _paint_item(_raster(image, 4096), whole, zoom)
    assert np.abs(_premultiplied(tiled) - _premultiplied(whole)).max() <= 1


@pytest.mark.parametrize("zoom", [0.3, 0.7, 2.0])
def test_raster_straight_alpha_canvas_renders_like_premultiplied(zoom):
    bgra = _mask_canvas(330, 250)
    bgra[::7, :, 3] = 90  # some translucent rows, so the two formats really differ
    premultiplied = _bgra_to_image(bgra)
    straight = premultiplied.convertToFormat(QImage.Format.Format_ARGB32)
    a = render_view([RasterLayerItem(lambda: premultiplied)], zoom, (300, 240))
    b = render_view([RasterLayerItem(lambda: straight)], zoom, (300, 240))
    assert np.abs(a.rgb.astype(int) - b.rgb.astype(int)).max() <= 1


def test_raster_stroke_while_minified_remakes_only_the_touched_cover_tiles():
    canvas = _Canvas(_bgra_to_image(_mask_canvas(330, 250)))
    item = RasterLayerItem(lambda: canvas.image)
    item.COVER_TILE = 64
    render_view([item], 0.7, (300, 240))  # level 0 coverage, 6 x 4 tiles of 64 px
    before = item.coverage_tiles()
    assert {key[0] for key in before} == {0} and len(before) == 24
    painter = QPainter(canvas.image)
    painter.fillRect(QRect(100, 100, 10, 10), QColor(tokens.SKEL))  # inside tile (1, 1)
    painter.end()
    item.canvas_changed(QRect(100, 100, 10, 10))
    after_render = render_view([item], 0.7, (300, 240))
    after = item.coverage_tiles()
    assert {key for key in before if after.get(key) != before[key]} == {(0, 1, 1)}
    fresh = render_view([RasterLayerItem(lambda: canvas.image)], 0.7, (300, 240))
    assert np.abs(after_render.rgb.astype(int) - fresh.rgb.astype(int)).max() <= 1


@pytest.mark.parametrize("zoom", [0.3, 0.6, 2.35])
def test_raster_partial_repaint_matches_a_full_repaint(zoom):
    image = _bgra_to_image(_mask_canvas(320, 240))
    item = RasterLayerItem(lambda: image)
    size = (int(320 * zoom) + 2, int(240 * zoom) + 2)
    full = _blank(*size)
    _paint_item(item, full, zoom)
    part = _blank(*size)
    _paint_item(item, part, zoom)
    _paint_item(item, part, zoom, QRectF(101.3, 67.7, 41.0, 23.0))
    assert np.abs(_premultiplied(full) - _premultiplied(part)).max() <= 1


def test_raster_paint_draws_only_the_exposed_rect():
    image = QImage(40, 40, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(255, 0, 0))
    item = RasterLayerItem(lambda: image)
    target = QImage(40, 40, QImage.Format.Format_ARGB32)
    target.fill(Qt.GlobalColor.transparent)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = QRectF(0, 0, 10, 10)
    painter = QPainter(target)
    item.paint(painter, option, None)
    painter.end()
    assert target.pixelColor(5, 5) == QColor(255, 0, 0)
    assert target.pixelColor(20, 20).alpha() == 0


def test_raster_paint_without_a_source_draws_nothing():
    item = RasterLayerItem(lambda: None)
    assert item.boundingRect().isEmpty()
    target = _blank(10, 10)
    _paint_item(item, target, 1.0, QRectF(0, 0, 10, 10))
    assert target.pixelColor(5, 5).alpha() == 0


@pytest.mark.parametrize(("dpr", "tiles"), [(1.0, False), (2.0, True)])
def test_raster_exact_or_coverage_follows_the_device_scale_including_hidpi(dpr, tiles):
    _bgra, image = _encoded_canvas(300, 200)
    item = RasterLayerItem(lambda: image)
    _paint_item(item, _blank(300, 200, dpr), 0.75)  # device scale 0.75 or 1.5
    assert bool(item.built_tiles()) is tiles


def test_raster_restores_the_painter_hints():
    _bgra, image = _encoded_canvas(64, 48)
    item = RasterLayerItem(lambda: image)
    for scale in (3.0, 0.4):
        target = _blank(200, 150)
        painter = QPainter(target)
        hints = (
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        painter.setRenderHints(hints)
        painter.scale(scale, scale)
        option = QStyleOptionGraphicsItem()
        option.exposedRect = item.boundingRect()
        item.paint(painter, option, None)
        assert painter.renderHints() == hints
        painter.end()


def test_smooth_photo_and_crisp_layer_share_one_scene():
    photo = PhotoLayerItem()
    photo.set_image(_photo_bgr(200, 150))
    photo.setZValue(0)
    _bgra, image = _encoded_canvas(100, 150)
    layer = RasterLayerItem(lambda: image)
    layer.setZValue(1)  # covers the left half of the photo
    top = PhotoLayerItem()
    top.set_image(_photo_bgr(50, 40, seed=SEED + 1))
    top.setPos(140, 100)
    top.setZValue(2)  # painted after the layer: must still be smooth
    render = render_view([photo, layer, top], 4.0, (800, 600), centre=(100.0, 75.0))
    ox, oy = render.dx, render.dy
    left = render.rgb[int(oy + 10 * 4) : int(oy + 140 * 4), int(ox + 10 * 4) : int(ox + 90 * 4)]
    _xs, _ys, clean = _decode(left)
    assert clean.all() and mean_run(left) >= 3.5  # the crisp layer: 4 px blocks
    right = render.rgb[int(oy + 10 * 4) : int(oy + 90 * 4), int(ox + 110 * 4) : int(ox + 135 * 4)]
    assert equal_neighbour_fraction(right) < 0.4  # the photo, right of the layer
    corner = render.rgb[
        int(oy + 105 * 4) : int(oy + 135 * 4), int(ox + 145 * 4) : int(ox + 185 * 4)
    ]
    assert equal_neighbour_fraction(corner) < 0.4  # the top photo, painted after the layer


def test_raster_opacity_and_visibility_use_the_standard_item_api():
    image = QImage(4, 4, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(255, 0, 0))
    scene = QGraphicsScene()
    item = RasterLayerItem(lambda: image)
    scene.addItem(item)

    def render() -> QImage:
        out = QImage(4, 4, QImage.Format.Format_ARGB32)
        out.fill(Qt.GlobalColor.transparent)
        painter = QPainter(out)
        scene.render(painter, QRectF(0, 0, 4, 4), QRectF(0, 0, 4, 4))
        painter.end()
        return out

    item.setOpacity(0.5)
    assert 100 <= render().pixelColor(1, 1).alpha() <= 155
    item.setVisible(False)
    assert render().pixelColor(1, 1).alpha() == 0


def test_minified_coverage_follows_edits_whose_rect_ends_on_an_odd_pixel():
    """The coverage is patched in whole 2 x 2 blocks around each edit: an edit whose right or
    bottom end is odd, anywhere on an odd-sized canvas (its last column and row included),
    shows zoomed out exactly as a fresh layer of the same canvas draws it."""
    rng = np.random.default_rng(SEED)
    h, w = 251, 331
    canvas = _Canvas(_bgra_to_image(np.zeros((h, w, 4), np.uint8)))
    item = RasterLayerItem(lambda: canvas.image)
    for zoom in (0.3, 0.6):  # coverage levels 1 and 0 built
        _paint_item(item, _blank(120, 90), zoom)
    for _ in range(30):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        rect = QRect(x, y, int(rng.integers(1, 9)), int(rng.integers(1, 9)))
        rect = rect.intersected(QRect(0, 0, w, h))
        painter = QPainter(canvas.image)
        painter.fillRect(rect, QColor(tokens.MASK))
        painter.end()
        item.canvas_changed(rect)
        for zoom in (0.3, 0.6):
            got, fresh = _blank(120, 90), _blank(120, 90)
            _paint_item(item, got, zoom)
            _paint_item(RasterLayerItem(lambda: canvas.image), fresh, zoom)
            assert got == fresh, (rect, zoom)
