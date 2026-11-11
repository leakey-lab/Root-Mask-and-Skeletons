"""Helpers and fixtures of the ``ClassesLayer`` tests (``test_correct_classes_layer.py``,
``test_correct_classes_layer_unsaved.py``): an 80 x 60 photo with a full-white ``mask/`` (never
read by the class labels), the goldens' 640 x 480 photo, an invalid ``gt/classes.json``, a layer
or a real ``CorrectSession`` over one layer on a recording ``FakeHost``, strokes, and readings
of the plane, the canvas and the notices.

Fixtures are imported by name (``from tests.correct_classes_layer_support import scene as scene
# fixture``).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QGraphicsScene

from app.gt import (
    Tracing,
    default_registry,
    gt_path,
    registry_path,
    write_tracing,
)
from app.gui.correct.classes_layer import ClassesLayer
from app.gui.correct.session import CorrectSession
from app.gui.paint_canvas import pixmap_to_argb_array
from tests.correct_fakes import FakeHost, FakePrompts
from tests.gt_goldens_support import make_gt_project

H, W = 60, 80  # the ported tests' photo
NO_MODIFIER = Qt.KeyboardModifier.NoModifier


@pytest.fixture
def scene(qapp):
    return QGraphicsScene()


@pytest.fixture
def project(tmp_path):
    """``plot_001.png`` (80 x 60) and a full-white ``mask/plot_001.png``: the class labels
    never read (or write) ``mask/``, so every test starts from empty labels."""
    folder = tmp_path / "project"
    (folder / "mask").mkdir(parents=True)
    image = str(folder / "plot_001.png")
    cv2.imwrite(image, np.full((H, W, 3), 90, np.uint8))
    mask = str(folder / "mask" / "plot_001.png")
    cv2.imwrite(mask, np.full((H, W), 255, np.uint8))
    return SimpleNamespace(folder=str(folder), image=image, mask=mask, labels=gt_path(image))


def _gt_project(tmp_path, case: str) -> SimpleNamespace:
    image, folder = make_gt_project(str(tmp_path), case)
    return SimpleNamespace(folder=folder, image=image, labels=gt_path(image))


@pytest.fixture
def gt_project(tmp_path):
    """The goldens' 640 x 480 photo, nothing under ``gt/``."""
    return _gt_project(tmp_path, "two_classes_crossing")


@pytest.fixture
def gt_project_bad_registry(tmp_path):
    """The goldens' photo with an invalid ``gt/classes.json``."""
    project = _gt_project(tmp_path, "empty_save")
    os.makedirs(os.path.join(project.folder, "gt"))
    with open(registry_path(project.folder), "w", encoding="utf-8") as fh:
        fh.write("{")
    return project


def open_layer(scene, image: str, size=(W, H)):
    """A ClassesLayer opened on ``image``; ``(layer, host, the open's notices)``."""
    host = FakeHost()
    layer = ClassesLayer(host, scene)
    notices = layer.open(image, size, os.path.dirname(image))
    return layer, host, notices


def open_session(scene, image: str, prompts: FakePrompts | None = None, size=(W, H)):
    """A real CorrectSession over one ClassesLayer, opened on ``image``."""
    host = FakeHost()
    layer = ClassesLayer(host, scene)
    out = SimpleNamespace(layer=layer, host=host, prompts=prompts or FakePrompts())
    out.dirty, out.notices = [], []
    out.session = CorrectSession(
        [layer],
        prompts=out.prompts,
        parent=lambda: None,
        on_dirty_changed=out.dirty.append,
        on_notice=lambda message, kind: out.notices.append((message, kind)),
        on_history_changed=lambda: None,
        on_undone=lambda *_args: None,
        on_saved=lambda _outcome: None,
    )
    host.session = out.session
    out.session.open(image, size, os.path.dirname(image))
    return out


def stroke(layer, *points, size=None) -> None:
    """Press at the first point, a dragging move per further point, release at the last."""
    if size is not None:
        layer.set_size(size)
    layer.press(QPoint(*points[0]), NO_MODIFIER)
    for point in points[1:]:
        layer.move(QPoint(*point), True)
    layer.release(QPoint(*points[-1]))


def click(layer, point) -> None:
    layer.press(QPoint(*point), NO_MODIFIER)
    layer.release(QPoint(*point))


def ids(layer) -> np.ndarray:
    return layer.document.history.plane


def shows_the_plane(layer) -> bool:
    """The canvas is exactly the plane in the registry's colours."""
    argb = pixmap_to_argb_array(layer.canvas.image)
    return np.array_equal(argb, layer.document.lut()[ids(layer)])


def notices_of(host: FakeHost) -> list[tuple[str, str]]:
    return [(call[1], call[2]) for call in host.calls if call[0] == "notify"]


def locked_text(layer) -> str:
    return (
        "Editing is off: gt/classes.json is invalid ("
        + layer.document.registry_error
        + "). Fix the file, then reopen the image."
    )


def file_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def write_with_unknown_key(image: str) -> bytes:
    """A labels file from another project: id 4 = 'hypha', unknown here (becomes Ignore)."""
    other = default_registry().add(key="hypha", name="Hypha", color="#3fa7ff")
    plane = np.zeros((H, W), np.uint8)
    plane[5:15, 5:15] = 4
    plane[30:40, 30:40] = 1
    write_tracing(gt_path(image), Tracing(plane), other)
    return file_bytes(gt_path(image))
