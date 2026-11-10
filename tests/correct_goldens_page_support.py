"""The driver of the Correct pixel goldens: the golden op scripts through one whole
``CorrectEditor`` (the rail's tool buttons, the dock's size sliders, the polyline prompt,
Clear, Save, Undo / Redo and the canvas input), the page's fixtures, and the checks of the
saved files against ``tests/data/correct_goldens.json``.

Moved here from ``test_correct_goldens.py``; ``test_correct_goldens_real.py`` and
``test_enhance_display_only.py`` replay the same scripts. A test module takes the fixtures
``app``, ``guards`` (autouse) and ``make_editor`` by importing them by name. The op
semantics are in ``tests/correct_goldens_support.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QKeyEvent

from app.gui.correct.layer import MASK, SKELETON
from app.gui.correct.mask_layer import MaskLayer
from app.gui.correct.skeleton_layer import SkeletonLayer
from app.project_layout import ProjectLayout
from tests.correct_goldens_support import (
    SYNTH_STEM,
    describe_mask_file,
    describe_skeleton_file,
    load_goldens,
    sha,
)
from tests.correct_support import (
    Mouse,
    drag,
    editor_factory,
    install_guards,
    qapp,
    view_drag,
)

GOLDENS = load_goldens()
OPS = GOLDENS["ops"]
KEYS = {"Delete": Qt.Key.Key_Delete}
MASK_INFORMATIONAL = {"file_sha256_informational"}  # encoder bytes may differ; pixels may not
SKELETON_SAVED = ("exists", "shape", "dtype", "colour_type", "pixels_sha256", "on_count", "values")


@pytest.fixture(scope="module")
def app():
    return qapp()


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    yield from install_guards(monkeypatch, tmp_path)


@pytest.fixture
def make_editor(app):
    yield from editor_factory()


# ============================================================================== the driver
def router_stroke(editor, points) -> None:
    """Press, a dragging move per further point, release (C18), on the canvas input."""
    drag(editor, *points)


def mouse_stroke(editor, points) -> None:
    """The same stroke as real mouse events through the canvas view."""
    view_drag(editor.graphics_view, [tuple(point) for point in points])


def apply(editor, op: list, stroke=router_stroke):
    """One golden op through the page (C18 semantics); ``editor.save()`` for ``["save"]``."""
    kind, controls = op[0], editor.controls
    if kind in ("brush", "eraser"):  # the mask's; the skeleton's eraser is ["tool", "eraser"]
        controls.tool_buttons[(MASK, kind)].click()
        controls.size_sliders[MaskLayer.SIZE_SPEC].slider.setValue(op[1])
    elif kind == "fill":
        controls.tool_buttons[(MASK, "fill")].click()
    elif kind == "tool":
        controls.tool_buttons[(SKELETON, op[1])].click()
    elif kind == "smooth":
        if controls.smooth_button.isChecked() != bool(op[1]):
            controls.smooth_button.click()
    elif kind == "eraser_size":
        controls.size_sliders[SkeletonLayer.ERASER_SIZE].slider.setValue(op[1])
    elif kind in ("stroke", "drag"):
        stroke(editor, op[1])
    elif kind == "click":
        stroke(editor, [op[1]])
    elif kind == "connect_endpoints":
        ends = sorted(editor.layers[SKELETON].document.model.endpoints())
        if len(ends) > max(op[1], op[2]):
            stroke(editor, [ends[op[1]], ends[op[2]]])
    elif kind == "double_click":
        editor.router.canvas_double_click(QPoint(*op[1]), Mouse())
    elif kind == "key":
        key = QKeyEvent(QEvent.Type.KeyPress, KEYS[op[1]], Qt.KeyboardModifier.NoModifier)
        editor.router.canvas_key_press(key)
    elif kind == "finish":
        controls.finish_button.click()
    elif kind == "clear":
        editor.clear_active_layer()
    elif kind == "undo":
        editor.undo()
    elif kind == "redo":
        editor.redo()
    elif kind == "save":  # O2: nothing unsaved writes nothing, and fails nothing
        nothing_unsaved = not editor.has_unsaved_changes()
        return editor.save() or nothing_unsaved
    else:
        raise ValueError(f"unknown golden op {op!r}")
    return None


def state_of(editor, layer_id: str) -> np.ndarray:
    """The mask plane (0 / 1) or the skeleton raster (0 / 255) in memory."""
    if layer_id == MASK:
        return editor.layers[MASK].document.history.plane
    return editor.layers[SKELETON].document.model.mask


def replay(editor, layer_id: str, script: str, expected: list[str], stroke=router_stroke):
    """Run ``script`` on ``layer_id`` (made the edited layer first), each op checked against
    its golden sha; returns what the script's save returned."""
    editor.set_active_layer(layer_id)
    saved = None
    for index, op in enumerate(OPS[script]):
        result = apply(editor, op, stroke)
        if op[0] == "save":
            saved = result
        assert sha(state_of(editor, layer_id)) == expected[index], f"op {index} {op} diverges"
    return saved


def files(folder: str, stem: str = SYNTH_STEM) -> tuple[str, str]:
    """The saved (GT) mask and the saved (GT) skeleton of image ``stem`` (default: the
    synthetic image)."""
    layout = ProjectLayout(folder)
    return layout.gt_mask_path(stem), layout.gt_skeleton_path(stem)


def assert_mask_file(path: str, golden: dict) -> None:
    contract = {k: v for k, v in golden.items() if k not in MASK_INFORMATIONAL}
    described = describe_mask_file(path)
    assert {key: described.get(key) for key in contract} == contract


def assert_skeleton_file(path: str, golden: dict) -> None:
    described = describe_skeleton_file(path)
    assert {k: described.get(k) for k in SKELETON_SAVED} == {
        k: golden.get(k) for k in SKELETON_SAVED
    }


def skeleton_start(editor) -> dict:
    doc = editor.layers[SKELETON].document
    return {
        "skeleton_source": doc.source,
        "raster_sha256": sha(doc.model.mask),
        "raster_shape": list(doc.model.mask.shape),
        "save_size": list(doc.save_size) if doc.save_size else None,
    }
