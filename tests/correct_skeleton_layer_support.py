"""Shared parts of the SkeletonLayer tests (``test_correct_skeleton_layer*.py``).

A layer on an offscreen scene with a recording ``FakeHost`` and the image's generated
skeleton drawn from straight segments (``_open``; fixtures ``layer``: one line, ``gap_layer``:
two segments whose endpoints face each other), input helpers, scikit-image's thinning as an
independent reference (``_thin``), item paints and scene renders, and the ``tool_calls``
spy. A test module takes the fixtures by importing them by name.
Moved here from ``test_correct_skeleton_layer.py`` when it was split by aspect.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QPainter
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QStyleOptionGraphicsItem,
)
from skimage.morphology import skeletonize

from app.gui import skeleton_tools
from app.gui.correct.skeleton_layer import SkeletonLayer
from app.gui.widgets import tokens
from tests.correct_fakes import FakeHost

ROOT = Path(__file__).resolve().parents[1]
NO_MOD = Qt.KeyboardModifier.NoModifier
W, H = 640, 480
LINE = [(100, 540, 240)]  # (x0, x1, y): today's test line, row 240, x 100..539
GAP = [(100, 200, 100), (260, 360, 100)]  # endpoints (199, 100) and (260, 100) face each other
EMPTY_HINT = (
    "No skeleton yet · Generate, Load one, or draw with Polyline (P)",
    "Generate skeletons first, open a file with Load (dock), or draw the roots with the "
    "Polyline tool.",
)


# ---------------------------------------------------------------------------- fixtures


@pytest.fixture()
def scene(app):
    return QGraphicsScene()


@pytest.fixture()
def host():
    return FakeHost()


@pytest.fixture()
def project(tmp_path):
    image = str(tmp_path / "a.png")
    assert cv2.imwrite(image, np.full((H, W, 3), 90, np.uint8))
    return str(tmp_path), image


def _write_raster(root, relpath, segments=()):
    raster = np.zeros((H, W), np.uint8)
    for x0, x1, y in segments:
        raster[y, x0:x1] = 255
    path = os.path.join(root, *relpath.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert cv2.imwrite(path, raster)
    return path


def _open(host, scene, project, segments=(), **kwargs) -> SkeletonLayer:
    """A layer on ``scene`` with the image of ``project`` open (its generated skeleton =
    ``segments``); the host's call log starts empty."""
    root, image = project
    if segments:
        _write_raster(root, "skeletons/a_fake.png", segments)
    layer = SkeletonLayer(host, scene, **kwargs)
    layer.open(image, (W, H), root)
    host.calls.clear()
    return layer


@pytest.fixture()
def layer(host, scene, project):
    layer = _open(host, scene, project, LINE)
    yield layer
    layer.cancel_scheduled_render()


@pytest.fixture()
def gap_layer(host, scene, project):
    layer = _open(host, scene, project, GAP)
    yield layer
    layer.cancel_scheduled_render()


# ----------------------------------------------------------------------------- helpers
def _click(layer, x, y, modifiers=NO_MOD):
    layer.press(QPoint(x, y), modifiers)
    layer.release(QPoint(x, y))


def _drag(layer, points, modifiers=NO_MOD):
    first, *rest = [QPoint(x, y) for x, y in points]
    layer.press(first, modifiers)
    for point in rest:
        layer.move(point, True)
    layer.release(rest[-1] if rest else first)


def _key(key, modifiers=NO_MOD):
    return QKeyEvent(QEvent.Type.KeyPress, key, modifiers)


def _commit(layer, a, b):
    """One Polyline action from ``a`` to ``b`` (image pixels)."""
    layer.set_tool("polyline")
    _click(layer, *a)
    _click(layer, *b)
    layer.finish_polyline()


def _count(host, name):
    return sum(1 for call in host.calls if call[0] == name)


def _thin(mask):
    """scikit-image's thinning of ``mask`` as uint8 0/255 (independent of the model)."""
    return skeletonize(mask > 0).astype(np.uint8) * 255


def _wait_for(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        QTest.qWait(5)
    return condition()


def _paint(item):
    """``item`` painted 1:1 over transparency (its exact pixels at device scale 1)."""
    rect = item.boundingRect()
    image = QImage(int(rect.width()), int(rect.height()), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = rect
    item.paint(painter, option, None)
    painter.end()
    return image


def _render(scene, source, zoom):
    """``source`` (scene px) of ``scene`` rendered ``zoom`` times over the canvas colour."""
    image = QImage(
        int(source.width() * zoom), int(source.height() * zoom), QImage.Format.Format_RGB32
    )
    image.fill(QColor(tokens.BG_0))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    scene.render(painter, QRectF(image.rect()), source)
    painter.end()
    return image


@pytest.fixture()
def tool_calls(monkeypatch):
    """Every move / release any skeleton tool receives, as ``(tool_id, method)``."""
    calls = []
    for cls in (
        skeleton_tools.SelectTool,
        skeleton_tools.EraserTool,
        skeleton_tools.PolylineTool,
        skeleton_tools.ConnectTool,
    ):
        for name in ("move", "release"):
            real = getattr(cls, name)

            def spy(self, pos, _real=real, _name=name):
                calls.append((self.tool_id, _name))
                return _real(self, pos)

            monkeypatch.setattr(cls, name, spy)
    return calls
