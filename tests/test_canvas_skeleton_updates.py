"""SkeletonLayerItem: lazy tiles and O(box) updates (app/gui/canvas_layers/skeleton.py).

``set_mask(mask, changed=box)`` patches the overlay around the box only, drops exactly the
vector tiles and pixel tiles its padded box touches, and patches (or, past half of a level,
drops) the cached LOD level images when they are next drawn; the render always equals a
full update. A paint extracts only the exposed tiles, within its budget, and asks for more
paints until every tile on screen is done.
"""

from __future__ import annotations

import os

os.environ["ROOT_VIEWER_ENABLE_OPENGL_VIEWPORT"] = "0"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QGraphicsScene,
    QGraphicsView,
)

from app.gui.canvas_layers.skeleton import SkeletonLayerItem  # noqa: E402
from app.gui.skeleton_overlay import overlay_indices  # noqa: E402
from tests.canvas_skeleton_support import (  # noqa: E402
    _busy_mask,
    _dots,
    _drop_scenes,  # noqa: F401 - the module's autouse fixture
    _edit,
    _in_scene,
    _render,
    _unbudgeted,
)
from tests.qt_teardown import delete_widgets  # noqa: E402

pytestmark = pytest.mark.usefixtures("no_prompts")

EDIT_BOXES = [(300, 200, 32, 32), (250, 250, 12, 12), (0, 0, 8, 8), (1016, 760, 8, 8)]


@pytest.mark.parametrize("box", EDIT_BOXES)
@pytest.mark.parametrize("new_array", [False, True])
@pytest.mark.parametrize("zoom", [0.5, 1.5, 3.0])
def test_a_box_update_renders_the_same_pixels_as_a_full_update(app, box, new_array, zoom):
    mask = _busy_mask()
    item = _unbudgeted(_in_scene(SkeletonLayerItem()))
    item.set_mask(mask)
    _render(item, zoom, (int(1024 * zoom), int(768 * zoom)))  # every cache built
    if new_array:
        mask = mask.copy()
    _edit(mask, box)

    item.set_mask(mask, changed=box)
    fresh = _unbudgeted(_in_scene(SkeletonLayerItem()))
    fresh.set_mask(mask.copy())

    size = (int(1024 * zoom), int(768 * zoom))
    assert np.array_equal(_render(item, zoom, size), _render(fresh, zoom, size))


ODD_BOXES = [(300, 200, 32, 32), (0, 0, 9, 7), (990, 770, 11, 7), (511, 255, 3, 3)]


@pytest.mark.parametrize("box", ODD_BOXES)
def test_a_box_update_patches_every_cached_level_to_a_full_rebuild(app, box):
    mask = _busy_mask(1001, 777)  # odd sides: each level replicates its last row and column
    item = _unbudgeted(_in_scene(SkeletonLayerItem()))
    item.set_mask(mask)
    zooms = (0.5, 0.24, 1.0 - 1e-7)  # levels 1 and 3, and level 0 of the LOD tier
    for zoom in zooms:
        _render(item, zoom, (int(1001 * zoom), int(777 * zoom)))
    assert set(item._level_images) == {0, 1, 3}
    for step in range(3):  # edits one after the other, each patched in place
        x, y, w, h = box
        _edit(mask, (max(0, x - 5 * step), y, w, h))
        item.set_mask(mask, changed=(max(0, x - 5 * step), y, w, h))
    fresh = _unbudgeted(_in_scene(SkeletonLayerItem()))
    fresh.set_mask(mask.copy())

    for zoom in zooms:
        size = (int(1001 * zoom), int(777 * zoom))
        assert np.array_equal(_render(item, zoom, size), _render(fresh, zoom, size)), zoom


def test_a_box_update_never_rebuilds_a_whole_level_and_moves_its_cache_key(app, monkeypatch):
    import app.gui.canvas_layers.skeleton as module

    mask = _busy_mask(2048, 1536)
    item = _unbudgeted(_in_scene(SkeletonLayerItem()))
    item.set_mask(mask)
    for zoom in (0.5, 0.24):
        _render(item, zoom, (int(2048 * zoom), int(1536 * zoom)))
    keys = {k: image.cacheKey() for k, image in item._level_images.items()}
    shapes = []
    monkeypatch.setattr(
        module, "overlay_indices", lambda m: shapes.append(m.shape) or overlay_indices(m)
    )

    _edit(mask, (1000, 690, 32, 32))
    item.set_mask(mask, changed=(1000, 690, 32, 32))
    _render(item, 0.5, (1024, 768))
    _render(item, 0.24, (int(2048 * 0.24), int(1536 * 0.24)))

    assert max(h * w for h, w in shapes) <= 36 * 36  # the box and its context, never a level
    assert set(item._level_images) == set(keys)  # patched, not dropped
    for k, image in item._level_images.items():
        assert image.cacheKey() != keys[k], k  # an OpenGL texture of it is uploaded again


def test_a_box_update_patches_a_cached_level_only_when_that_level_is_drawn(app, monkeypatch):
    import app.gui.canvas_layers.skeleton as module

    mask = _busy_mask(2048, 1536)
    item = _unbudgeted(_in_scene(SkeletonLayerItem()))
    item.set_mask(mask)
    for zoom in (0.5, 0.24):  # levels 1 and 3 cached
        _render(item, zoom, (int(2048 * zoom), int(1536 * zoom)))
    keys = {k: image.cacheKey() for k, image in item._level_images.items()}
    shapes = []
    monkeypatch.setattr(
        module, "overlay_indices", lambda m: shapes.append(m.shape) or overlay_indices(m)
    )

    _edit(mask, (1000, 690, 32, 32))
    item.set_mask(mask, changed=(1000, 690, 32, 32))
    assert shapes == [(36, 36)]  # the level-0 overlay only: the levels wait for a draw
    assert {k: image.cacheKey() for k, image in item._level_images.items()} == keys
    _render(item, 0.5, (1024, 768))  # level 1 drawn and patched, level 3 still waits
    assert len(shapes) == 2 and shapes[1][0] * shapes[1][1] <= 20 * 20
    assert item._level_images[1].cacheKey() != keys[1]
    assert item._level_images[3].cacheKey() == keys[3]
    _render(item, 0.24, (int(2048 * 0.24), int(1536 * 0.24)))
    assert len(shapes) == 3 and item._level_images[3].cacheKey() != keys[3]


def test_a_frame_sized_box_drops_the_cached_levels_instead_of_patching_them(app, monkeypatch):
    import app.gui.canvas_layers.skeleton as module

    mask = _busy_mask(1001, 777)
    full = mask.copy()
    frame = (0, 0, 1001, 777)
    item = _unbudgeted(_in_scene(SkeletonLayerItem()))
    item.set_mask(mask)
    zooms = (0.5, 0.24)  # levels 1 and 3
    for zoom in zooms:
        _render(item, zoom, (int(1001 * zoom), int(777 * zoom)))
    assert set(item._level_images) == {1, 3}
    words, patched = module.pixel_words, []
    monkeypatch.setattr(module, "pixel_words", lambda image: patched.append(image) or words(image))

    for content in (0, full):  # Clear, then its undo
        mask[:] = content
        item.set_mask(mask, changed=frame)
        assert item._level_images == {}  # rebuilt when drawn, as cheap as a patch
        fresh = _unbudgeted(_in_scene(SkeletonLayerItem()))
        fresh.set_mask(mask.copy())
        for zoom in zooms:
            size = (int(1001 * zoom), int(777 * zoom))
            assert np.array_equal(_render(item, zoom, size), _render(fresh, zoom, size)), zoom

    assert patched == []  # no level was patched


def test_a_cached_level_merges_its_stale_boxes_and_is_dropped_past_half_of_it(app):
    mask = _busy_mask(1001, 777)
    item = _unbudgeted(_in_scene(SkeletonLayerItem()))
    item.set_mask(mask)
    zooms = (0.5, 0.24)  # levels 1 and 3 cached; level 3 is the one drawn below
    for zoom in zooms:
        _render(item, zoom, (int(1001 * zoom), int(777 * zoom)))

    for box in ((0, 0, 8, 8), (990, 770, 11, 7)):  # opposite corners
        _edit(mask, box)
        item.set_mask(mask, changed=box)
        _render(item, 0.24, (int(1001 * 0.24), int(777 * 0.24)))

    # level 1 waited: one stale box over both corners, the whole level, so it was dropped
    assert set(item._level_images) == {3}
    fresh = _unbudgeted(_in_scene(SkeletonLayerItem()))
    fresh.set_mask(mask.copy())
    for zoom in zooms:
        size = (int(1001 * zoom), int(777 * zoom))
        assert np.array_equal(_render(item, zoom, size), _render(fresh, zoom, size)), zoom


def test_a_box_update_recomputes_the_overlay_around_the_box_only_and_extracts_nothing(
    app, monkeypatch
):
    import app.gui.canvas_layers.skeleton as module

    mask = np.zeros((1536, 2048), np.uint8)
    mask[700, 100:1900] = 255
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(mask)
    shapes, extracted = [], []
    monkeypatch.setattr(
        module, "overlay_indices", lambda m: shapes.append(m.shape) or overlay_indices(m)
    )
    real_segments = module.tile_segments
    monkeypatch.setattr(
        module, "tile_segments", lambda *a: extracted.append(a) or real_segments(*a)
    )

    mask[690:722, 1000:1032] = 0
    item.set_mask(mask, changed=(1000, 690, 32, 32))
    item.set_mask(mask, changed=(0, 0, 3, 3))  # clipped at the image corner

    # the overlay of the box padded by 1 needs 1 more pixel of context (the 3x3 halo)
    assert shapes == [(36, 36), (5, 5)]
    assert extracted == []
    # The spy does see an extraction: a vector paint over the line needs one.
    _render(item, 4.0, (64, 64), origin=(1100.0, 690.0))
    assert extracted and item.mode_painted() == "vector"


def test_a_box_update_drops_exactly_the_vector_tiles_its_padded_box_touches(app):
    mask = _busy_mask()
    item = _unbudgeted(_in_scene(SkeletonLayerItem()))
    item.set_mask(mask)
    _render(item, 2.0, (2048, 1536))  # s = 2: every tile extracted
    before = item.vector_tiles()
    assert set(before) == {(tx, ty) for tx in range(4) for ty in range(3)}

    item.set_mask(mask, changed=(300, 300, 10, 10))  # inside tile (1, 1)
    after_inside = item.vector_tiles()
    item.set_mask(mask, changed=(256, 20, 4, 4))  # padded box reaches x = 255: tile (0, 0)
    after_border = item.vector_tiles()

    assert set(before) - set(after_inside) == {(1, 1)}
    assert set(after_inside) - set(after_border) == {(0, 0), (1, 0)}
    assert set(after_border) <= set(before)  # an update never extracts


def test_a_vector_paint_extracts_only_the_exposed_tiles(app):
    item = _unbudgeted(_in_scene(SkeletonLayerItem()))
    item.set_mask(_busy_mask())

    _render(item, 4.0, (400, 400), origin=(300, 300))  # scene 300..400: inside tile (1, 1)
    assert set(item.vector_tiles()) == {(1, 1)}
    _render(item, 4.0, (800, 400), origin=(300, 300))  # scene x 300..500: still tile (1, 1)
    assert set(item.vector_tiles()) == {(1, 1)}
    _render(item, 4.0, (1200, 400), origin=(300, 300))  # scene x 300..600: into tile (2, 1)
    assert set(item.vector_tiles()) == {(1, 1), (2, 1)}


def test_without_an_extraction_budget_pending_tiles_show_the_pixel_overlay(app):
    mask = _busy_mask()
    item = _in_scene(SkeletonLayerItem())
    item.set_mask(mask)
    item.EXTRACT_BUDGET_MS = 0.0  # one tile per paint
    pixels = _in_scene(SkeletonLayerItem())
    pixels.set_mask(mask)
    pixels.VECTOR_MIN_SCALE = 100.0  # the nearest level-0 overlay at every zoom >= 1

    first = _render(item, 2.0, (1024, 1024), origin=(128, 128))  # tiles x 0..2, y 0..2
    reference = _render(pixels, 2.0, (1024, 1024), origin=(128, 128))

    assert len(item.vector_tiles()) == 1
    (done,) = item.vector_tiles()
    for tx in range(3):
        for ty in range(3):
            if (tx, ty) == done:
                continue
            x0, y0 = max(0, 2 * (tx * 256 - 128)) + 8, max(0, 2 * (ty * 256 - 128)) + 8
            x1, y1 = (
                min(1024, 2 * ((tx + 1) * 256 - 128)) - 8,
                min(1024, 2 * ((ty + 1) * 256 - 128)) - 8,
            )
            assert np.array_equal(first[y0:y1, x0:x1], reference[y0:y1, x0:x1]), (tx, ty)


@pytest.fixture
def shown_views():
    """Views a test shows, closed and deleted afterwards even when the test fails."""
    views: list[QGraphicsView] = []
    yield views
    delete_widgets(views)


def _raster_view(
    views: list[QGraphicsView],
    item: SkeletonLayerItem,
    size: tuple[int, int],
    zoom: float,
    centre: QPointF | None = None,
) -> QGraphicsView:
    """A shown raster-viewport view (partial updates) holding ``item`` at ``zoom``."""
    view = QGraphicsView()
    views.append(view)
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scene = QGraphicsScene(view)
    view.setScene(scene)
    scene.addItem(item)
    view.resize(*size)
    view.scale(zoom, zoom)
    if centre is not None:
        view.centerOn(centre)
    view.show()
    for _ in range(5):
        QApplication.processEvents()
    return view


def test_pending_tiles_schedule_paints_until_the_render_equals_an_unbudgeted_one(app, shown_views):
    mask = _busy_mask()
    item = SkeletonLayerItem()
    item.set_mask(mask)
    item.EXTRACT_BUDGET_MS = 0.0  # one tile per paint
    view = _raster_view(shown_views, item, (900, 700), 2.0, QPointF(500, 380))
    visible = view.mapToScene(view.viewport().rect()).boundingRect()
    wanted = {
        (tx, ty)
        for tx in range(int(visible.left()) // 256, int(visible.right()) // 256 + 1)
        for ty in range(int(visible.top()) // 256, int(visible.bottom()) // 256 + 1)
    }
    assert len(wanted) > 3

    for _ in range(200):  # the item asks for the follow-up paints itself
        app.processEvents()
        if set(item.vector_tiles()) >= wanted:
            break
    assert set(item.vector_tiles()) >= wanted

    unbudgeted = _unbudgeted(_in_scene(SkeletonLayerItem()))
    unbudgeted.set_mask(mask)
    origin = (visible.left(), visible.top())
    assert np.array_equal(
        _render(item, 2.0, (600, 400), origin=origin),
        _render(unbudgeted, 2.0, (600, 400), origin=origin),
    )


def test_a_small_repaint_keeps_the_tier_of_the_whole_view(app, shown_views):
    item = _unbudgeted(SkeletonLayerItem())
    item.set_mask(_dots(3000))  # 3000 segments on screen: over the raster budgets
    view = _raster_view(shown_views, item, (560, 640), 4.0)
    assert item.mode_painted() == "raster"

    painted = []
    item.paint = lambda p, o, w=None: (
        painted.append(o.exposedRect) or type(item).paint(item, p, o, w)
    )
    corner = view.mapFromScene(QPointF(20.0, 20.0))
    view.viewport().update(corner.x(), corner.y(), 12, 12)  # a hover-sized repaint
    for _ in range(5):
        app.processEvents()

    assert painted and max(r.width() for r in painted) < 10  # a few dots were repainted
    assert item.mode_painted() == "raster"  # not vectors in a patch of the pixel view
