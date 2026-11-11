"""Shared helpers of the skeleton layer tests (test_canvas_skeleton.py,
test_canvas_skeleton_updates.py): scenes kept for the module, renders at a zoom and a
devicePixelRatio, colour checks and test skeletons."""

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QGraphicsScene

from app.gui.canvas_layers.skeleton import SkeletonLayerItem

GREY = QColor(96, 96, 96)
HALO_ALPHA = 170

# ------------------------------------------------------------------------------ helpers
_SCENES: list[QGraphicsScene] = []  # keep every scene referenced for the module


@pytest.fixture(scope="module", autouse=True)
def _drop_scenes():
    """The scenes go with the module (with their items' warm-up timers, which would fire in
    a later test file)."""
    yield
    _SCENES.clear()


def _in_scene(item: SkeletonLayerItem) -> SkeletonLayerItem:
    scene = QGraphicsScene()
    scene.addItem(item)
    _SCENES.append(scene)
    return item


def _unbudgeted(item: SkeletonLayerItem) -> SkeletonLayerItem:
    """Every exposed tile is extracted in the first paint (renders become deterministic)."""
    item.EXTRACT_BUDGET_MS = float("inf")
    return item


def _render(
    item: SkeletonLayerItem,
    zoom: float,
    size: tuple[int, int],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    dpr: float = 1.0,
    hints: QPainter.RenderHint | None = None,
    background: QColor = GREY,
) -> np.ndarray:
    """RGB (h, w, 3) of the item's scene rendered at ``zoom`` into ``size`` device px.

    ``origin`` is the scene point at the image's top-left; the device scale the item sees
    is ``zoom * dpr`` (times its own ``scale`` transform).
    """
    w, h = size
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(dpr)
    img.fill(background)
    painter = QPainter(img)
    if hints is not None:
        painter.setRenderHints(hints)
    lw, lh = w / dpr, h / dpr
    item.scene().render(
        painter, QRectF(0, 0, lw, lh), QRectF(origin[0], origin[1], lw / zoom, lh / zoom)
    )
    painter.end()
    return _rgb(img)


def _rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    rows = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())
    return rows[:, : img.width() * 3].reshape(img.height(), img.width(), 3).copy()


def _rgb_of(color: str | QColor) -> tuple[int, int, int]:
    c = QColor(color)
    return (c.red(), c.green(), c.blue())


def _over(color: str, alpha: int, background: QColor = GREY) -> np.ndarray:
    """``color`` at ``alpha`` composited over an opaque ``background`` (hand formula)."""
    a = alpha / 255.0
    fg = np.array(_rgb_of(color), float)
    bg = np.array(_rgb_of(background), float)
    return fg * a + bg * (1.0 - a)


def _mask(w: int, h: int, pixels) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for x, y in pixels:
        m[y, x] = 255
    return m


def _line_mask() -> np.ndarray:
    """40x30: a horizontal line on row 10 (x 5..30) and a vertical one on column 20."""
    m = np.zeros((30, 40), np.uint8)
    m[10, 5:31] = 255
    m[11:25, 20] = 255
    return m


VIEW_HINTS = QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
SMOOTH_VIEW_HINTS = VIEW_HINTS | QPainter.RenderHint.SmoothPixmapTransform


def _near(rgb: np.ndarray, color: str | QColor, tol: int = 30) -> np.ndarray:
    """Pixels within ``tol`` of ``color`` on every channel."""
    return (np.abs(rgb.astype(int) - np.array(_rgb_of(color))) <= tol).all(axis=-1)


def _max_run(flags: np.ndarray) -> int:
    """Longest run of True in a 1-D array."""
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _components(binary: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return count - 1


def _three_curves() -> np.ndarray:
    """400x300: a circle, a diagonal and a sine curve, 8-connected, far apart."""
    mask = np.zeros((300, 400), np.uint8)
    cv2.circle(mask, (80, 80), 50, 255, 1)
    cv2.line(mask, (200, 20), (380, 150), 255, 1)
    xs = np.arange(20, 380)
    pts = np.stack([xs, (230 + 30 * np.sin(xs / 25.0)).astype(int)], axis=1)
    cv2.polylines(mask, [pts.astype(np.int32)], False, 255, 1)
    return mask


def _dots(count: int, columns: int = 50) -> np.ndarray:
    """``count`` isolated pixels on a 2 px grid (one vector segment each)."""
    rows = -(-count // columns)
    mask = np.zeros((2 * rows + 2, 2 * columns + 2), np.uint8)
    for i in range(count):
        mask[1 + 2 * (i // columns), 1 + 2 * (i % columns)] = 255
    return mask


def _busy_mask(w: int = 1024, h: int = 768) -> np.ndarray:
    """Lines in every 256 px tile, some crossing tile borders."""
    mask = np.zeros((h, w), np.uint8)
    for y in range(40, h, 96):
        cv2.line(mask, (0, y), (w - 1, y + 30), 255, 1)
    for x in range(30, w, 128):
        cv2.line(mask, (x, 0), (x + 40, h - 1), 255, 1)
    return mask


def _edit(mask: np.ndarray, box) -> None:
    """Erase and draw inside ``box`` (x, y, w, h), in place."""
    x, y, w, h = box
    mask[y : y + h, x : x + w] = 0
    cv2.line(mask, (x, y + h // 2), (x + w - 1, y + h - 1), 255, 1)
    mask[y, x] = 255  # a pixel on the box corner: its halo reaches outside the box
