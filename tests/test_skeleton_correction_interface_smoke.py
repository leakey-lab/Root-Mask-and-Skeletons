"""Smoke test for the modern (PR6) SkeletonCorrectionInterface overlay layout.

Instantiates the interface under an offscreen QApplication and asserts that the
floating overlays exist and that the underlying control widgets/attrs survived
the reparenting into the ToolRail / FloatingDock / EnhancePopover / polyline
prompt, plus that key signal paths (undo/redo shortcuts, programmatic tool
selection) still function.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.gui.skeleton_correction_interface import SkeletonCorrectionInterface  # noqa: E402
from app.gui.widgets import ToolRail, FloatingDock, EnhancePopover  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def test_overlays_present(app):
    iface = SkeletonCorrectionInterface()

    # Floating overlays exist and are the expected types.
    assert isinstance(iface.tool_rail, ToolRail)
    assert isinstance(iface.dock, FloatingDock)
    assert isinstance(iface.enhance_popover, EnhancePopover)
    assert iface.polyline_prompt is not None

    # Tool buttons preserved and still in the exclusive group.
    for btn in (
        iface.select_button,
        iface.eraser_button,
        iface.polyline_button,
        iface.connect_button,
    ):
        assert btn is not None
        assert btn in iface.tool_group.buttons()

    # Toggles / actions / sliders / polyline-prompt buttons survived.
    assert iface.mode_toggle is not None
    assert iface.smooth_polyline_toggle is not None
    assert iface.save_skeleton_button is not None
    assert iface.load_skeleton_button is not None
    assert iface.undo_button is not None
    assert iface.redo_button is not None
    assert iface.clear_button is not None
    assert iface.eraser_slider is not None
    assert iface.opacity_slider is not None
    assert iface.finish_polyline_button is not None
    assert iface.cancel_polyline_button is not None
    assert iface.status_label is not None
    assert iface.norm_controls is not None
    assert iface.enhance_button is not None

    # Default tool is Select; eraser-only slider hidden initially.
    # isHidden() tracks explicit show/hide even when the host isn't on-screen.
    assert iface.select_button.isChecked()
    assert iface.eraser_container.isHidden()


def test_tool_selection_and_shortcuts(app):
    iface = SkeletonCorrectionInterface()

    # Programmatic polyline tool selection drives _on_tool_changed.
    iface.polyline_button.setChecked(True)
    iface._on_tool_changed(iface.polyline_button)
    assert iface.current_tool == iface.TOOL_POLYLINE
    assert iface.smooth_polyline_toggle.isEnabled()

    # Eraser tool makes the eraser slider container visible.
    iface.eraser_button.setChecked(True)
    iface._on_tool_changed(iface.eraser_button)
    assert iface.current_tool == iface.TOOL_ERASER
    assert not iface.eraser_container.isHidden()

    # Undo/redo shortcuts are wired and fire without error (no skeleton loaded).
    iface.undo_shortcut.activated.emit()
    iface.redo_shortcut.activated.emit()

    # Resize triggers overlay reposition without error.
    iface.resize(900, 700)
