"""Shared helpers of the photo and raster layer tests (test_canvas_photo.py,
test_canvas_raster.py): seeded photo-like images, item paints into images, device rects."""

import cv2
import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QStyleOptionGraphicsItem

SEED = 20260929


def _photo_bgr(w=640, h=480, seed=SEED) -> np.ndarray:
    """A photo-like BGR image: seeded noise blurred to realistic local contrast."""
    noise = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 1.2)


def _inner(render, w, h, margin=3):
    """Device rect (x0, y0, x1, y1) inside both the viewport and the image, shrunk by margin."""
    height, width = render.rgb.shape[:2]
    x0 = max(0, int(np.ceil(render.dx)) + margin)
    y0 = max(0, int(np.ceil(render.dy)) + margin)
    x1 = min(width, int(np.floor(render.dx + w * render.zoom)) - margin)
    y1 = min(height, int(np.floor(render.dy + h * render.zoom)) - margin)
    return x0, y0, x1, y1


def _paint_item(item, image: QImage, scale: float, exposed: QRectF | None = None) -> None:
    """Paint ``item`` into ``image`` at ``scale``; with ``exposed``, repaint only that part
    the way a view does: clip to it and clear it before the item paints again."""
    painter = QPainter(image)
    painter.scale(scale, scale)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = item.boundingRect() if exposed is None else exposed
    if exposed is not None:
        painter.setClipRect(exposed)  # what the view's paint-event region does
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(exposed, Qt.GlobalColor.transparent)  # the background redraw
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    item.paint(painter, option, None)
    painter.end()


def _blank(w, h, dpr=1.0) -> QImage:
    image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    return image


def _premultiplied(image: QImage) -> np.ndarray:
    """``(h, w, 4)`` int premultiplied B, G, R, A: what a translucent layer composites
    with (un-premultiplied RGB would blow a +-1 alpha step up into a large colour step)."""
    image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    rows = np.frombuffer(ptr, np.uint8).reshape(image.height(), image.bytesPerLine())
    return rows[:, : image.width() * 4].reshape(image.height(), image.width(), 4).astype(int)
