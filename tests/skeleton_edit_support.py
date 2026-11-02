"""Helpers of the skeleton editing tests at junctions, on picks and on rings.

Builds 1 px skeletons from straight lines (thinned by scikit-image, so they are what the
editor opens), opens them in a real ``SkeletonLayer`` on an offscreen scene with a recording
``FakeHost``, and sends it clicks, double-clicks and keys as the canvas does (press,
release, double-click, release). Used by ``test_skeleton_junction_edits.py``,
``test_skeleton_pick_without_edit.py`` and ``test_skeleton_ring_pick.py``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QGraphicsScene
from skimage.morphology import skeletonize

from app.gui.correct.skeleton_layer import SkeletonLayer
from tests.correct_fakes import FakeHost

W, H = 640, 480
NO_MOD = Qt.KeyboardModifier.NoModifier


def thin(mask: np.ndarray) -> np.ndarray:
    """scikit-image's thinning of ``mask`` as uint8 0/255 (independent of the model)."""
    return skeletonize(mask > 0).astype(np.uint8) * 255


def lines(*polylines, w: int = W, h: int = H) -> np.ndarray:
    """A thinned 1 px skeleton (uint8 0/255) of the open ``polylines`` of ``(x, y)`` points."""
    raster = np.zeros((h, w), np.uint8)
    for points in polylines:
        pts = np.array(points, np.int32).reshape(-1, 1, 2)
        cv2.polylines(raster, [pts], False, 255, 1)
    return thin(raster)


def components(mask: np.ndarray) -> int:
    """The number of 8-connected pieces of ``mask``."""
    count, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return count - 1


def tips(mask: np.ndarray) -> int:
    """The number of skeleton tips (set pixels with exactly one set 8-neighbour)."""
    on = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    degree = cv2.filter2D(on, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return int(np.count_nonzero(on & (degree == 1)))


def open_layer(tmp_path, skeleton: np.ndarray) -> tuple[SkeletonLayer, FakeHost]:
    """A ``SkeletonLayer`` with ``skeleton`` open as the generated skeleton of image a.

    Asserts that the model opened exactly ``skeleton`` (it is a thinning fixed point). The
    caller ends with ``layer.cancel_scheduled_render()``.
    """
    root = str(tmp_path)
    os.makedirs(root, exist_ok=True)
    image = os.path.join(root, "a.png")
    assert cv2.imwrite(image, np.full((H, W, 3), 90, np.uint8))
    os.makedirs(os.path.join(root, "skeletons"), exist_ok=True)
    assert cv2.imwrite(os.path.join(root, "skeletons", "a_fake.png"), skeleton)
    host = FakeHost()
    scene = QGraphicsScene()
    layer = SkeletonLayer(host, scene)
    layer._test_scene = scene  # keep the scene alive as long as the layer
    layer.open(image, (W, H), root)
    assert np.array_equal(layer.model.mask, skeleton), "the test skeleton is not thin"
    host.calls.clear()
    return layer, host


def key(layer: SkeletonLayer, qt_key, modifiers=NO_MOD) -> bool:
    """A key press sent to the layer; True when the layer used it."""
    return layer.key(QKeyEvent(QEvent.Type.KeyPress, qt_key, modifiers))


def click(layer: SkeletonLayer, x: int, y: int) -> None:
    """A left click at image pixel ``(x, y)``: press, then release."""
    layer.press(QPoint(x, y), NO_MOD)
    layer.release(QPoint(x, y))


def double_click(layer: SkeletonLayer, x: int, y: int) -> None:
    """A fast double-click as Qt delivers it: press, release, double-click, release."""
    layer.press(QPoint(x, y), NO_MOD)
    layer.release(QPoint(x, y))
    layer.double_click(QPoint(x, y))
    layer.release(QPoint(x, y))


def drag(layer: SkeletonLayer, start, end) -> None:
    """A left drag from ``start`` to ``end`` (image pixels)."""
    layer.press(QPoint(*start), NO_MOD)
    layer.move(QPoint(*end), True)
    layer.release(QPoint(*end))


def pick(layer: SkeletonLayer, x: int, y: int) -> None:
    """Pick the branch nearest ``(x, y)`` with the Select tool (it opens in Polyline)."""
    layer.set_tool("select")
    click(layer, x, y)
    assert layer.draft.edit_original is not None, f"nothing picked at {(x, y)}"


def edits(host: FakeHost) -> int:
    """How many undoable actions the layer reported."""
    return sum(1 for call in host.calls if call[0] == "layer_edited")
