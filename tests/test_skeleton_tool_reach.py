"""Grab and snap distances of the skeleton tools follow the zoom (KNOWN_ISSUES #6, S4).

The distances (grab a polyline point, add a bend, move the line, pick a branch, grab an
endpoint) are screen px, as the markers are: at 1600 % an endpoint no longer grabs from
160 screen px while its marker is 4 px wide. In image px they are clamped to at least one
pixel (the pointer is floored to a pixel) and at most four times the 100 % distance. At
100 %, and without a view, they are exactly the image px distances of before.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QTransform
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView

from app.gui import skeleton_tools
from app.gui.correct.skeleton_layer import SkeletonLayer
from tests.correct_fakes import FakeHost
from tests.qt_teardown import delete_widgets

W, H = 640, 480
NO_MOD = Qt.KeyboardModifier.NoModifier


@pytest.fixture
def make_layer(app, tmp_path):
    views: list[QGraphicsView] = []

    def make(zoom: float | None, rows=((100, 300, 200),)) -> SkeletonLayer:
        image = str(tmp_path / "a.png")
        cv2.imwrite(image, np.full((H, W, 3), 90, np.uint8))
        raster = np.zeros((H, W), np.uint8)
        for x0, x1, y in rows:
            raster[y, x0:x1] = 255
        os.makedirs(tmp_path / "skeletons", exist_ok=True)
        cv2.imwrite(str(tmp_path / "skeletons" / "a_fake.png"), raster)
        scene = QGraphicsScene()
        layer = SkeletonLayer(FakeHost(), scene)
        layer.open(image, (W, H), str(tmp_path))
        if zoom is not None:
            view = QGraphicsView(scene)
            view.setTransform(QTransform.fromScale(zoom, zoom))
            views.append(view)
        layer._scene_ref = scene  # keep the scene alive with the layer
        return layer

    yield make
    delete_widgets(views)


def _click(layer, x, y, modifiers=NO_MOD):
    layer.press(QPoint(x, y), modifiers)
    layer.release(QPoint(x, y))


@pytest.mark.parametrize("zoom", [None, 1.0])
@pytest.mark.parametrize(
    "name",
    ["HANDLE_HIT_PX", "BEND_HIT_PX", "MOVE_HIT_PX", "PICK_HIT_PX", "ENDPOINT_HIT_PX"],
)
def test_at_100_percent_and_without_a_view_the_distances_are_unchanged(make_layer, zoom, name):
    layer = make_layer(zoom)
    base = getattr(skeleton_tools, name)
    assert skeleton_tools.reach(layer, base) == base


@pytest.mark.parametrize(
    ("zoom", "expected"),
    [(16.0, 1.0), (4.0, 2.5), (2.0, 5.0), (0.5, 20.0), (0.25, 40.0), (0.1, 40.0)],
)
def test_the_reach_is_constant_on_screen_within_its_clamps(make_layer, zoom, expected):
    layer = make_layer(zoom)
    assert skeleton_tools.reach(layer, 10) == pytest.approx(expected)


def _connect_drag(layer, start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    """A Connect drag from ``start`` to ``end``; the skeleton afterwards (a copy)."""
    layer.press(QPoint(*start), NO_MOD)
    layer.move(QPoint(*end), True)
    layer.release(QPoint(*end))
    return layer.model.mask.copy()


def _joined_from_the_tip(mask: np.ndarray, end_x: int) -> bool:
    """A line runs from the tip (299, 200) to the release at (``end_x``, 260): every row in
    between has a pixel within one of the straight line (it is thinned after drawing)."""
    rows = range(205, 256)
    return all(mask[y, round(299 + (end_x - 299) * (y - 200) / 60) - 1 :][:3].any() for y in rows)


def test_at_1600_percent_connect_grabs_an_endpoint_only_at_its_pixel(make_layer):
    layer = make_layer(16.0)  # the line's tips: (100, 200) and (299, 200)
    layer.set_tool("connect")
    before = layer.model.mask.copy()

    after = _connect_drag(layer, (303, 200), (303, 260))  # 4 image px = 64 screen px away

    assert np.array_equal(after, before) and not layer.model.can_undo, "nothing was grabbed"
    after = _connect_drag(layer, (300, 200), (300, 260))  # the next pixel
    assert _joined_from_the_tip(after, 300) and layer.model.can_undo


def test_at_100_percent_connect_still_grabs_from_ten_pixels(make_layer):
    layer = make_layer(1.0)
    layer.set_tool("connect")

    after = _connect_drag(layer, (309, 200), (309, 260))

    assert _joined_from_the_tip(after, 309) and layer.model.can_undo


def test_at_1600_percent_a_click_beside_a_handle_adds_a_point(make_layer):
    layer = make_layer(16.0)
    layer.set_tool("polyline")
    _click(layer, 50, 50)
    _click(layer, 60, 50)

    layer.press(QPoint(62, 52), NO_MOD)  # 2.8 image px from the last handle: a new point
    layer.release(QPoint(62, 52))

    assert [(p.x(), p.y()) for p in layer.draft.points] == [(50, 50), (60, 50), (62, 52)]


def test_at_25_percent_select_picks_a_branch_from_twelve_screen_px(make_layer):
    layer = make_layer(0.25)
    layer.set_tool("select")
    _click(layer, 200, 240)  # 40 image px = 10 screen px below the line at y 200
    assert layer.current_tool == "polyline" and layer.draft.edit_original is not None


def test_at_100_percent_select_misses_a_branch_twenty_px_away(make_layer):
    layer = make_layer(1.0)
    layer.set_tool("select")
    _click(layer, 200, 220)
    assert layer.current_tool == "select" and layer.draft.edit_original is None
