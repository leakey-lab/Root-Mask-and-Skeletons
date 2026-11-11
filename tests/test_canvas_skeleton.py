"""SkeletonLayerItem: what it draws (app/gui/canvas_layers/skeleton.py, contract C6, spec 10.3).

The 1 px skeleton raster is drawn as a line at every zoom: below device scale 1 the line +
halo overlay through a max-pool pyramid, from 1 to 2 the overlay at level 0 (nearest), and
from 2 device px per skeleton px m-adjacency vector links between pixel centres, extracted
lazily per 256 px tile at paint time. ``vectors=False`` (the Library's 2 px generated file)
never draws vectors or nearest pixels: bilinear from scale 1. The vector extraction itself
is tested in test_skeleton_vectors.py.
"""

from __future__ import annotations

import os

os.environ["ROOT_VIEWER_ENABLE_OPENGL_VIEWPORT"] = "0"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage, QPainter, QTransform  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QStyleOptionGraphicsItem,
)

from app.gui.canvas_layers.skeleton import SkeletonLayerItem  # noqa: E402
from app.gui.skeleton_overlay import overlay_indices  # noqa: E402
from app.gui.widgets import tokens  # noqa: E402
from tests.canvas_skeleton_support import (  # noqa: E402
    GREY,
    HALO_ALPHA,
    SMOOTH_VIEW_HINTS,
    VIEW_HINTS,
    _components,
    _dots,
    _drop_scenes,  # noqa: F401 - the module's autouse fixture
    _in_scene,
    _line_mask,
    _mask,
    _max_run,
    _near,
    _over,
    _render,
    _rgb,
    _rgb_of,
    _three_curves,
    _unbudgeted,
)

pytestmark = pytest.mark.usefixtures("no_prompts")


# ------------------------------------------------------------------------ step 1: API
def test_zoom_one_draws_the_line_in_the_skeleton_colour_over_a_dark_halo(app):
    mask = _line_mask()
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(mask)

    rgb = _render(item, 1.0, (40, 30))

    assert item.mode_painted() == "raster"
    line = mask > 0
    assert (rgb[line] == _rgb_of(tokens.SKEL)).all()
    halo_px = rgb[9, 12]  # directly above the horizontal line
    assert np.abs(halo_px - _over(tokens.BG_0, HALO_ALPHA)).max() <= 2
    assert tuple(rgb[2, 2]) == _rgb_of(GREY)  # far from the line: untouched


def test_halo_false_draws_the_line_only(app):
    mask = _line_mask()
    item = _in_scene(SkeletonLayerItem(halo=False))
    item.set_mask(mask)

    rgb = _render(item, 1.0, (40, 30))

    assert (rgb[mask > 0] == _rgb_of(tokens.SKEL)).all()
    assert (rgb[mask == 0] == _rgb_of(GREY)).all()


def test_the_line_colour_follows_the_color_argument(app):
    mask = _line_mask()
    item = _in_scene(SkeletonLayerItem(color=tokens.MASK, halo=False))
    item.set_mask(mask)

    rgb = _render(item, 1.0, (40, 30))

    assert (rgb[mask > 0] == _rgb_of(tokens.MASK)).all()


def test_scale_becomes_the_item_transform_and_bounds_stay_in_skeleton_pixels(app):
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(np.zeros((256, 341), np.uint8), scale=(640 / 341, 480 / 256))

    assert item.boundingRect() == QRectF(0, 0, 341, 256)
    assert item.transform() == QTransform.fromScale(640 / 341, 480 / 256)
    scene_rect = item.sceneBoundingRect()
    assert scene_rect.width() == pytest.approx(640) and scene_rect.height() == pytest.approx(480)

    item.set_mask(np.zeros((256, 341), np.uint8))  # default scale: identity again
    assert item.transform() == QTransform()


def test_revision_counts_every_set_mask_call(app):
    item = SkeletonLayerItem()
    assert item.revision() == 0
    item.set_mask(_line_mask())
    item.set_mask(_line_mask(), changed=(1, 1, 2, 2))
    item.set_mask(None)
    assert item.revision() == 3


def test_no_mask_means_an_empty_item_that_paints_nothing(app):
    item = _in_scene(SkeletonLayerItem())
    assert item.mode_painted() == ""
    item.set_mask(_line_mask())
    item.set_mask(None)

    assert item.boundingRect().isEmpty()
    rgb = _render(item, 1.0, (40, 30))
    assert (rgb == _rgb_of(GREY)).all()
    assert item.mode_painted() == ""


def test_paint_draws_only_the_exposed_rect(app):
    mask = _line_mask()
    item = SkeletonLayerItem(halo=False)
    item.set_mask(mask)
    target = QImage(40, 30, QImage.Format.Format_ARGB32_Premultiplied)
    target.fill(Qt.GlobalColor.transparent)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = QRectF(0, 0, 12, 30)  # the left part of the horizontal line only

    painter = QPainter(target)
    item.paint(painter, option, None)
    painter.end()

    assert target.pixelColor(8, 10) == QColor(tokens.SKEL)
    assert target.pixelColor(25, 10).alpha() == 0  # line pixel outside the exposed rect


def test_painting_restores_the_painters_render_hints(app):
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(_line_mask())
    wanted = QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
    img = QImage(40, 30, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(GREY)
    painter = QPainter(img)
    painter.setRenderHints(wanted)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = QRectF(0, 0, 40, 30)

    item.paint(painter, option, None)
    after = painter.renderHints()
    painter.end()

    assert after & wanted == wanted


def test_no_vector_tiles_are_extracted_below_the_vector_scale(app):
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(_line_mask())
    _render(item, 1.0, (40, 30))
    assert item.vector_tiles() == {}


def test_800_percent_draws_a_thin_line_not_eight_pixel_blocks(app):
    mask = np.zeros((40, 60), np.uint8)
    mask[10, 5:46] = 255  # horizontal
    for i in range(20):
        mask[15 + i, 10 + i] = 255  # diagonal
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(mask)

    rgb = _render(item, 8.0, (480, 320), hints=VIEW_HINTS)

    assert item.mode_painted() == "vector"
    skel = _near(rgb, tokens.SKEL)
    across_horizontal = [_max_run(skel[40:130, c]) for c in range(200, 360, 7)]
    across_diagonal = [_max_run(skel[r, 60:300]) for r in range(150, 270, 7)]
    assert max(across_horizontal) < 4 and min(across_horizontal) >= 1
    assert max(across_diagonal) < 4 and min(across_diagonal) >= 1


@pytest.mark.parametrize("zoom", [0.25, 0.3, 0.45, 0.6, 0.75, 0.9])
def test_minified_lines_never_break_into_dashes(app, zoom):
    mask = _three_curves()
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(mask)
    size = (int(np.ceil(400 * zoom)), int(np.ceil(300 * zoom)))

    rgb = _render(item, zoom, size, hints=VIEW_HINTS)

    assert item.mode_painted() == "raster-lod"
    assert _components(_near(rgb, tokens.SKEL, tol=0)) == _components(mask > 0) == 3


def test_a_one_pixel_line_on_any_row_stays_visible_when_minified(app):
    for row in (100, 101, 102, 103):
        mask = np.zeros((200, 300), np.uint8)
        mask[row, 20:280] = 255
        item = _in_scene(SkeletonLayerItem(halo=False))
        item.set_mask(mask)
        for zoom in (0.3, 0.6, 0.75):
            rgb = _render(item, zoom, (int(300 * zoom), int(200 * zoom)))
            skel = _near(rgb, tokens.SKEL, tol=0)
            covered = skel.any(axis=0)[int(25 * zoom) : int(275 * zoom)]
            assert covered.all(), (row, zoom)


@pytest.mark.parametrize(
    ("zoom", "mode"), [(0.5, "raster-lod"), (1.0, "raster"), (1.5, "raster"), (4.0, "vector")]
)
def test_the_tier_follows_the_device_scale(app, zoom, mode):
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(_line_mask())
    _render(item, zoom, (int(40 * zoom), int(30 * zoom)))
    assert item.mode_painted() == mode


def test_devicepixelratio_two_at_zoom_one_is_the_vector_tier(app):
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(_line_mask())
    _render(item, 1.0, (80, 60), dpr=2.0)
    assert item.mode_painted() == "vector"


@pytest.mark.parametrize(
    ("zoom", "mode"), [(0.4, "raster-lod"), (0.5, "raster"), (0.75, "raster"), (1.0, "vector")]
)
def test_tiers_count_device_pixels_at_devicepixelratio_two(app, zoom, mode):
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(_line_mask())
    _render(item, zoom, (int(80 * zoom), int(60 * zoom)), dpr=2.0)
    assert item.mode_painted() == mode


def test_the_list_fallback_for_other_qreal_layouts_draws_the_same_pixels(app, monkeypatch):
    import app.gui.canvas_layers.skeleton as module

    mask = _three_curves()
    mask[5, 5] = 255  # a dot
    item = _unbudgeted(_in_scene(SkeletonLayerItem()))  # a busy CPU must not split the tiles
    item.set_mask(mask)
    fast = _render(item, 4.0, (1600, 1200), hints=VIEW_HINTS)
    monkeypatch.setattr(module, "_SHARES_NUMPY_LAYOUT", {QLineF: False, QPointF: False})

    assert np.array_equal(_render(item, 4.0, (1600, 1200), hints=VIEW_HINTS), fast)


@pytest.mark.parametrize(
    ("count", "mode"), [(300, "vector"), (1000, "vector-thin"), (3000, "raster")]
)
def test_raster_engine_budgets_pick_thinner_lines_then_the_pixel_overlay(app, count, mode):
    mask = _dots(count)
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(mask)
    h, w = mask.shape
    _render(item, 4.0, (4 * w, 4 * h))
    assert item.mode_painted() == mode


@pytest.mark.parametrize("zoom", [0.5, 1.5, 4.0])
def test_an_isolated_pixel_is_drawn_at_every_tier(app, zoom):
    item = _in_scene(SkeletonLayerItem(halo=False))
    item.set_mask(_mask(20, 20, [(9, 9)]))
    rgb = _render(item, zoom, (int(20 * zoom), int(20 * zoom)))
    assert _near(rgb, tokens.SKEL, tol=60).any()


@pytest.mark.parametrize("zoom", [0.5, 1.5, 4.0])
def test_halo_false_never_darkens_the_background(app, zoom):
    item = _in_scene(SkeletonLayerItem(halo=False))
    item.set_mask(_three_curves())
    rgb = _render(item, zoom, (int(400 * zoom), int(300 * zoom)), hints=VIEW_HINTS)
    assert rgb.min() >= min(_rgb_of(GREY)) - 1  # blends of the line and grey only


def test_the_halo_darkens_both_sides_of_a_vector_line(app):
    mask = np.zeros((20, 40), np.uint8)
    mask[10, 5:35] = 255
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(mask)
    rgb = _render(item, 8.0, (320, 160), hints=VIEW_HINTS)
    column = rgb[:, 160].astype(int).sum(axis=1)
    centre = 84  # device row of the line centre (10.5 * 8)
    grey = sum(_rgb_of(GREY))
    assert column[centre - 3 : centre - 1].min() < grey - 60  # dark rim above the line
    assert column[centre + 1 : centre + 3].min() < grey - 60  # and below it


@pytest.mark.parametrize(
    ("zoom", "mode"), [(0.5, "raster-lod"), (1.5, "raster-smooth"), (4.0, "raster-smooth")]
)
def test_library_items_never_draw_vectors(app, zoom, mode):
    item = _in_scene(SkeletonLayerItem(halo=False, vectors=False))
    item.set_mask(_line_mask())
    _render(item, zoom, (int(40 * zoom), int(30 * zoom)))
    assert item.mode_painted() == mode
    assert item.vector_tiles() == {}


def test_library_two_pixel_lines_have_soft_edges_at_4x(app):
    mask = np.zeros((30, 40), np.uint8)
    mask[10:12, 5:35] = 255  # a generated line, about 2 px wide
    item = _in_scene(SkeletonLayerItem(halo=False, vectors=False))
    item.set_mask(mask)

    rgb = _render(item, 4.0, (160, 120), hints=VIEW_HINTS)

    profile = rgb[30:60, 80, 0].astype(int)  # red channel across the line's upper edge
    grey, skel = _rgb_of(GREY)[0], _rgb_of(tokens.SKEL)[0]
    between = (profile > grey + 10) & (profile < skel - 10)
    assert between.any()  # bilinear: intermediate values, no hard pixel-block edge
    assert _max_run(_near(rgb[:, 80], tokens.SKEL, tol=4)) <= 5  # no 8 px flat block


def test_library_scale_covers_the_whole_photo(app):
    mask = np.zeros((256, 341), np.uint8)
    mask[0, :] = mask[-1, :] = 255
    mask[:, 0] = mask[:, -1] = 255
    item = _in_scene(SkeletonLayerItem(halo=False, vectors=False))
    item.set_mask(mask, scale=(640 / 341, 480 / 256))

    rgb = _render(item, 1.0, (660, 500))

    ys, xs = np.nonzero(_near(rgb, tokens.SKEL, tol=100))
    assert xs.min() <= 1 and ys.min() <= 1
    assert 638 <= xs.max() <= 640 and 478 <= ys.max() <= 480


@pytest.mark.parametrize("zoom", [0.3, 1.0, 1.37, 1.5, 1.93])
def test_nearest_tiers_never_blend_even_under_a_smoothing_view(app, zoom):
    mask = np.zeros((600, 600), np.uint8)
    cv2.line(mask, (0, 255), (599, 257), 255, 1)  # crosses the x = 256 and x = 512 borders
    cv2.line(mask, (255, 0), (257, 599), 255, 1)  # crosses the y = 256 and y = 512 borders
    cv2.circle(mask, (256, 256), 40, 255, 1)
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(mask)

    rgb = _render(item, zoom, (int(600 * zoom), int(600 * zoom)), hints=SMOOTH_VIEW_HINTS)

    halo = np.round(_over(tokens.BG_0, HALO_ALPHA)).astype(int)
    exact = _near(rgb, tokens.SKEL, tol=0) | _near(rgb, GREY, tol=0)
    exact |= (np.abs(rgb.astype(int) - halo) <= 1).all(axis=-1)
    assert exact.all()  # line, halo or background: no blended pixel anywhere


@pytest.mark.parametrize("zoom", [1.0, 1.37, 1.93])
def test_nearest_tier_tiles_leave_no_gap_in_a_line_crossing_their_border(app, zoom):
    mask = np.zeros((520, 600), np.uint8)
    mask[300, :] = 255  # crosses both vertical tile borders
    mask[:, 300] = 255  # crosses the horizontal tile border
    item = _in_scene(SkeletonLayerItem(halo=False))
    item.set_mask(mask)

    rgb = _render(item, zoom, (int(600 * zoom), int(520 * zoom)))

    skel = _near(rgb, tokens.SKEL, tol=0)
    assert skel.any(axis=0)[: int(599 * zoom)].all()  # every device column shows the row
    assert skel.any(axis=1)[: int(519 * zoom)].all()  # every device row shows the column


@pytest.mark.parametrize(
    "scale, offset", [(1.5, 1.5), (1.5, 2.5), (1.5, 0.5), (1.5, 0.0), (1.25, 0.5), (1.75, 0.5)]
)
def test_nearest_tier_tiles_sit_on_the_device_pixels_of_the_mask_layer_tiles(app, scale, offset):
    """At a .5 tie of the device mapping (common at 125-175 % display scaling) the skeleton's
    pixel tiles meet on the same device columns and rows as the mask layer's tiles: both
    layers show the same pixels identically."""
    from app.gui.canvas_layers.raster import RasterLayerItem
    from app.gui.canvas_layers.skeleton import _argb

    mask = np.zeros((300, 600), np.uint8)
    mask[:, [254, 255, 256, 257, 511, 512]] = 255  # lines on both sides of the tile borders
    mask[[255, 256], :] = 255
    item = SkeletonLayerItem(halo=False)
    item.set_mask(mask)
    canvas = _argb(overlay_indices(mask), item._table)  # the same pixels as a mask canvas
    layer = RasterLayerItem(lambda: canvas)

    def paint(target) -> np.ndarray:
        size = (int(600 * scale + offset) + 2, int(300 * scale + offset) + 2)
        img = QImage(*size, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(GREY)
        painter = QPainter(img)
        painter.translate(offset, offset)
        painter.scale(scale, scale)
        option = QStyleOptionGraphicsItem()
        option.exposedRect = target.boundingRect()
        target.paint(painter, option, None)
        painter.end()
        return _rgb(img)

    skeleton = paint(item)
    assert item.mode_painted() == "raster"
    assert np.array_equal(skeleton, paint(layer))
