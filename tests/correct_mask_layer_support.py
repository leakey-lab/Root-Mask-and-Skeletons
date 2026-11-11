"""Shared parts of the MaskLayer tests (``test_correct_mask_layer*.py``).

The oracle for every pixel: a direct ``PaintCanvas`` replay with today's lookup table,
canvas-to-plane and fill-source rules written out by hand (``Replay``); a layer opened on
a W x H image with a recording ``FakeHost`` (``open_layer``); strokes, clicks, the paths of
the prediction and the GT mask, and the ``scene`` fixture (import it by name).
Moved here from ``test_correct_mask_layer.py`` when it was split by aspect.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
)
from PyQt6.QtWidgets import QGraphicsScene

from app.gui.correct.layer import (
    LayerStatus,
)
from app.gui.correct.mask_layer import MaskLayer
from app.gui.key_text import platform_keys
from app.gui.paint_canvas import PaintCanvas, pixmap_to_argb_array
from app.gui.patch_history import PatchHistory
from app.gui.widgets import tokens
from app.io_utils import imread_unicode, imwrite_unicode
from tests.correct_fakes import FakeHost

STEM = "plot_T001_L001_2024.07.18_111530_001_SMM"
W, H = 160, 120
NO_MODS = Qt.KeyboardModifier.NoModifier
MASK_COLOUR = QColor(tokens.MASK)
CLEARED = platform_keys("Mask cleared. Ctrl+Z to undo; Save writes an empty GT mask.")
FILL_MISS = "Fill works inside a closed outline. Close the loop first."
NOT_SHOWN_SIZE_TEXT = (
    "Saved mask size does not match the image: it is not shown and is kept unless you save over it"
)


@pytest.fixture
def scene(qapp):
    return QGraphicsScene()


# --------------------------------------------------------------------------- the oracle
def mask_lut() -> np.ndarray:
    """Today's _plane_lut: every value above 0 opaque in the mask colour (B, G, R, A)."""
    lut = np.zeros((256, 4), np.uint8)
    lut[1:] = (MASK_COLOUR.blue(), MASK_COLOUR.green(), MASK_COLOUR.red(), 255)
    return lut


def to_plane(argb: np.ndarray, rect) -> np.ndarray:
    """Today's _argb_to_plane: painted (alpha > 0) is 1."""
    return (argb[:, :, 3] > 0).astype(np.uint8)


def fill_source(plane: np.ndarray) -> tuple[np.ndarray, int]:
    """Today's _fill_source: the mask outline, filled with 1."""
    return (plane > 0).astype(np.uint8) * 255, 1


class Replay:
    """Today's semantics straight on a PaintCanvas over an equal plane."""

    def __init__(self, plane: np.ndarray) -> None:
        self.history = PatchHistory()
        self.history.reset(plane.copy())
        self.canvas = PaintCanvas(
            self.history,
            lut=mask_lut,
            to_plane=to_plane,
            fill_source=fill_source,
            on_region=lambda rect: None,
            on_replaced=lambda: None,
        )
        self.canvas.rebuild()

    @property
    def plane(self) -> np.ndarray:
        return self.history.plane

    def stroke(self, points, size: int, *, erase: bool = False) -> None:
        self.canvas.begin(QPoint(*points[0]), MASK_COLOUR, size, erase=erase)
        for p in points[1:]:
            self.canvas.extend(QPoint(*p))
        self.canvas.commit()

    def fill(self, p) -> None:
        self.canvas.fill(QPoint(*p))


# --------------------------------------------------------------------------- helpers
def image_path(tmp_path, stem: str = STEM) -> str:
    """The photo's path: the layer never decodes it (the editor passes its size)."""
    return os.path.join(str(tmp_path), f"{stem}.jpg")


def mask_file(tmp_path, stem: str = STEM, ext: str = ".png") -> str:
    """The model's mask (the prediction): never written by the layer."""
    return os.path.join(str(tmp_path), "mask", f"{stem}{ext}")


def gt_file(tmp_path, stem: str = STEM) -> str:
    """The GT mask: the one file a save writes (WG-T2)."""
    return os.path.join(str(tmp_path), "gt", "mask", f"{stem}.png")


#: The status of a layer showing the prediction it opened.
PREDICTION = LayerStatus(
    "none",
    "Prediction",
    f"mask/{STEM}.png — the model's output. Save after an edit writes gt/mask/{STEM}.png",
)


def write_model_mask(tmp_path, plane: np.ndarray, stem: str = STEM) -> None:
    """A model mask as the generator writes it: 8-bit grey, 255 where ``plane`` is set."""
    path = mask_file(tmp_path, stem)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imwrite_unicode(path, (plane > 0).astype(np.uint8) * 255, durable=False)


def boxes_plane() -> np.ndarray:
    """A saved mask with two filled boxes (the eraser has something to erase)."""
    plane = np.zeros((H, W), np.uint8)
    plane[40:80, 60:120] = 1
    plane[5:15, 5:25] = 1
    return plane


def open_layer(scene, tmp_path, plane: np.ndarray | None = None, host: FakeHost | None = None):
    """A MaskLayer opened on a W x H image; ``plane`` becomes its saved mask file."""
    host = host or FakeHost()
    if plane is not None:
        write_model_mask(tmp_path, plane)
    layer = MaskLayer(host, scene)
    notices = layer.open(image_path(tmp_path), (W, H), None)
    assert notices == []
    host.calls.clear()
    return layer, host


def plane_of(layer) -> np.ndarray:
    return layer.document.history.plane


def count(host: FakeHost, name: str) -> int:
    return sum(1 for call in host.calls if call[0] == name)


def stroke(layer, points) -> None:
    """Press at the first point, a dragging move per further point, release at the last."""
    layer.press(QPoint(*points[0]), NO_MODS)
    for p in points[1:]:
        layer.move(QPoint(*p), True)
    layer.release(QPoint(*points[-1]))


def click(layer, p) -> None:
    layer.press(QPoint(*p), NO_MODS)
    layer.release(QPoint(*p))


def canvas_shows_the_plane(layer) -> bool:
    return np.array_equal(pixmap_to_argb_array(layer.canvas.image), mask_lut()[plane_of(layer)])


def ring_rect(layer) -> QRectF:
    return layer.ring.rect()


def read_rgba(path: str) -> np.ndarray:
    return imread_unicode(path, cv2.IMREAD_UNCHANGED)
