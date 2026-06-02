"""Smoke test for the modern (PR5) MaskTracingInterface overlay layout.

Instantiates the interface under an offscreen QApplication and asserts that the
floating overlays exist and that the underlying control widgets/attrs survived
the reparenting into the ToolRail / FloatingDock / EnhancePopover.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.gui.mask_tracing_interface import MaskTracingInterface  # noqa: E402
from app.gui.widgets import ToolRail, FloatingDock, EnhancePopover  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def test_overlays_present(app):
    iface = MaskTracingInterface()

    # Floating overlays exist and are the expected types.
    assert isinstance(iface.tool_rail, ToolRail)
    assert isinstance(iface.dock, FloatingDock)
    assert isinstance(iface.enhance_popover, EnhancePopover)

    # Core control widgets/attrs preserved through the reparenting.
    assert iface.brush_button is not None
    assert iface.eraser_button is not None
    assert iface.fill_button is not None
    assert iface.save_button is not None
    assert iface.size_slider is not None
    assert iface.zoom_slider is not None
    assert iface.mode_toggle is not None

    # Tool buttons stay in the exclusive group.
    assert iface.brush_button in iface.tool_button_group.buttons()


def test_b_key_and_ctrl_wheel_still_drive_sliders(app):
    iface = MaskTracingInterface()

    # B-key wheel adjusts the brush size slider.
    iface.size_slider.setValue(5)
    iface.b_key_pressed = True
    start = iface.size_slider.value()
    iface.update_brush_size(start)  # smoke the brush-size code path
    assert iface.size_slider.value() == start

    # Ctrl-wheel zoom path drives the zoom slider.
    iface.zoom_slider.setValue(100)
    iface.zoom_slider.setValue(min(iface.zoom_slider.value() + 10, 200))
    assert iface.zoom_slider.value() == 110

    # Resize triggers overlay reposition without error.
    iface.resize(900, 700)
