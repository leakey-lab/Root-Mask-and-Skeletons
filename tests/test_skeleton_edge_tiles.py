"""The skeleton's last partial tiles sit on the mask layer's device pixels (sheet #40).

From device scale 1 to 2 the skeleton is drawn as pixels in 256 px tiles, nearest, as the
mask layer (:class:`RasterLayerItem`) draws its canvas. The mask pads its last tile column
and row to 256 px and lets the clip end them at the image edge; the skeleton cut them at the
edge and snapped that edge on its own. Qt maps device pixels to image pixels from each
tile's own size, so wherever the edges rounded apart (a fit zoom at 100 % scaling too, on
OpenGL and raster canvases alike) the skeleton's right and bottom tiles sat up to 1 device
pixel off the mask's. The skeleton now pads them as the mask does. Display only.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QStyleOptionGraphicsItem

from app.gui.canvas_layers.raster import RasterLayerItem
from app.gui.canvas_layers.skeleton import SkeletonLayerItem, _argb
from app.gui.skeleton_overlay import overlay_indices

W, H = 640, 480  # the owner's photos: tile columns 0-511 full, 512-639 partial; rows 256-479
GREY = QColor(96, 96, 96)


def _skeleton() -> np.ndarray:
    """Lines over every tile, both partial edges and the image's last column and row."""
    rng = np.random.default_rng(40)
    mask = np.zeros((H, W), np.uint8)
    for _ in range(30):
        p = tuple(int(v) for v in rng.integers(0, (W, H)))
        q = tuple(int(v) for v in rng.integers(0, (W, H)))
        cv2.line(mask, p, q, 255, 1)
    mask[:, [511, 512, 600, W - 1]] = 255
    mask[[255, 256, 400, H - 1], :] = 255
    return mask


def _paint(item, dpr: float, scale: float, offset: float, exposed: QRectF) -> np.ndarray:
    """ARGB words of ``item`` painted at device ``scale`` and device ``offset`` on a
    ``dpr`` image (the view's zoom is ``scale / dpr``)."""
    size = (int(W * scale + offset) + 3, int(H * scale + offset) + 3)
    image = QImage(*size, QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(GREY)
    painter = QPainter(image)
    painter.translate(offset / dpr, offset / dpr)
    painter.scale(scale / dpr, scale / dpr)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = exposed
    item.paint(painter, option, None)
    painter.end()
    data = np.frombuffer(image.constBits().asstring(image.sizeInBytes()), np.uint32)
    return data.reshape(image.height(), image.bytesPerLine() // 4)[:, : image.width()].copy()


@pytest.mark.parametrize(
    ("dpr", "scale", "offset"),
    [
        (1.0, 1.1, 0.0),  # 100 % scaling, a fit zoom
        (1.0, 1.4, 0.0),
        (1.0, 1.9, 0.5),
        (1.25, 1.2, 0.5),
        (1.25, 1.37, 0.0),
        (1.5, 1.66, 0.0),
        (1.75, 1.4, 0.3),
    ],
)
def test_the_last_partial_tiles_draw_the_mask_layers_device_pixels(app, dpr, scale, offset):
    mask = _skeleton()
    item = SkeletonLayerItem()
    item.set_mask(mask)
    canvas = _argb(overlay_indices(mask), item._table)  # the same pixels as a mask canvas
    layer = RasterLayerItem(lambda: canvas)
    whole = item.boundingRect()

    skeleton = _paint(item, dpr, scale, offset, whole)

    assert item.mode_painted() == "raster"
    differ = skeleton != _paint(layer, dpr, scale, offset, whole)
    assert not differ.any(), f"{int(differ.sum())} device px differ"


def test_a_repaint_of_the_corner_tile_draws_the_mask_layers_device_pixels(app):
    """A partial repaint (a brush ring's hover) over the corner tile, right and bottom
    partial at once: the same pixels as the mask layer's repaint."""
    mask = _skeleton()
    item = SkeletonLayerItem()
    item.set_mask(mask)
    canvas = _argb(overlay_indices(mask), item._table)
    layer = RasterLayerItem(lambda: canvas)
    corner = QRectF(590.0, 430.0, 50.0, 50.0)

    skeleton = _paint(item, 1.0, 1.37, 0.0, corner)

    assert item.mode_painted() == "raster"
    differ = skeleton != _paint(layer, 1.0, 1.37, 0.0, corner)
    assert not differ.any(), f"{int(differ.sum())} device px differ"
