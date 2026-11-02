"""The raster halo is drawn on whole device pixels (sheet #15, X2-07).

From 200 % on a canvas without OpenGL (Remote Desktop, a software renderer, Linux while a
dashboard lives) the skeleton's halo is drawn opaque into its own image and blended once at
HALO_ALPHA. That image was aligned to whole logical pixels and blitted at that position: at
125 / 150 / 175 % display scaling a logical pixel is a fractional number of device pixels, so
the halo sat up to half a device pixel off and, as each repaint (a brush ring's hover) put
its image elsewhere, jumped along the ring. The image is now aligned to whole device pixels
and blitted unscaled. Display only.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QRectF
from PyQt6.QtGui import QColor, QImage, QPainter, QTransform
from PyQt6.QtWidgets import QStyleOptionGraphicsItem, QWidget

from app.gui.canvas_layers.skeleton import SkeletonLayerItem
from tests.qt_teardown import delete_widgets

W, H = 120, 90
GREY = QColor(96, 96, 96)


@pytest.fixture
def viewport():
    """A widget larger than every render: the item's on-screen part is all of it, so each
    repaint strokes the full frame's line list (X2-06) and only the halo can move."""
    widget = QWidget()
    widget.resize(3000, 3000)
    yield widget
    delete_widgets([widget])


def _skeleton() -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    cv2.line(mask, (5, 10), (110, 70), 255, 1)
    cv2.circle(mask, (60, 45), 25, 255, 1)
    mask[80, 10:100] = 255
    mask[5:85, 100] = 255
    return mask


def _item() -> SkeletonLayerItem:
    item = SkeletonLayerItem()
    item.EXTRACT_BUDGET_MS = float("inf")
    item.set_mask(_skeleton())
    return item


def _paint(
    item: SkeletonLayerItem,
    widget: QWidget,
    dpr: float,
    scale: float,
    offset: float,
    exposed: QRectF | None = None,
    painter_class: type[QPainter] = QPainter,
) -> np.ndarray:
    """ARGB words of ``item`` painted at device ``scale`` and device ``offset`` on a ``dpr``
    image: the same device transform at every ``dpr``."""
    size = (math.ceil(W * scale + offset) + 2, math.ceil(H * scale + offset) + 2)
    image = QImage(*size, QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(GREY)
    painter = painter_class(image)
    painter.translate(offset / dpr, offset / dpr)
    painter.scale(scale / dpr, scale / dpr)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = exposed if exposed is not None else item.boundingRect()
    item.paint(painter, option, widget)
    painter.end()
    data = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), np.uint32)
    return data.reshape(image.height(), image.bytesPerLine() // 4)[:, : image.width()].copy()


@pytest.mark.parametrize("dpr", [1.25, 1.5, 1.75])
@pytest.mark.parametrize(("scale", "offset"), [(2.5, 0.6), (3.0, 0.5), (4.0, 0.5)])
def test_fractional_scaling_draws_the_halo_of_100_percent(app, viewport, dpr, scale, offset):
    """At 125-175 % the frame equals the one at 100 % with the same device transform
    (cosmetic pens are device pixels wide): no half-pixel shift."""
    item = _item()
    reference = _paint(item, viewport, 1.0, scale, offset)

    frame = _paint(item, viewport, dpr, scale, offset)

    assert item.mode_painted() == "vector"
    differ = frame != reference
    assert not differ.any(), f"{int(differ.sum())} device px differ"


@pytest.mark.parametrize("dpr", [1.25, 1.5, 1.75])
def test_a_partial_repaint_draws_the_full_frames_halo(app, viewport, dpr):
    """A repaint rect moved by a pixel at a time (a brush ring's hover) draws the full
    frame's halo inside it: the halo no longer jumps along the ring."""
    item = _item()
    scale = 2.5
    full = _paint(item, viewport, dpr, scale, 0.0)
    for step in range(8):
        exposed = QRectF(20.0 + step / scale, 15.0 + 0.7 * step / scale, 60.0, 50.0)
        part = _paint(item, viewport, dpr, scale, 0.0, exposed)
        x0 = math.ceil(exposed.left() * scale) + 1  # device px wholly inside the repaint
        x1 = math.floor(exposed.right() * scale) - 1
        y0 = math.ceil(exposed.top() * scale) + 1
        y1 = math.floor(exposed.bottom() * scale) - 1
        differ = part[y0:y1, x0:x1] != full[y0:y1, x0:x1]
        assert not differ.any(), f"step {step}: {int(differ.sum())} device px differ"


def test_the_halo_image_is_blitted_unscaled_on_whole_device_pixels(app, viewport):
    """The blit runs with an identity device transform from a whole device pixel, the
    raster engine's plain blend (about 1 ms per frame faster than the scaled blit)."""
    blits: list[tuple[QTransform.TransformationType, float, QPoint]] = []

    class Spy(QPainter):
        def drawImage(self, *args):  # noqa: N802 - Qt name
            if isinstance(args[0], QPoint):  # the halo image (pixel tiles pass a QRectF)
                blits.append((self.deviceTransform().type(), args[1].devicePixelRatio(), args[0]))
            super().drawImage(*args)

    item = _item()
    _paint(item, viewport, 1.25, 2.5, 0.0, painter_class=Spy)

    assert item.mode_painted() == "vector"
    assert [(kind, ratio) for kind, ratio, _point in blits] == [
        (QTransform.TransformationType.TxNone, 1.0)
    ]
