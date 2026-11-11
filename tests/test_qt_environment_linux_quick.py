"""Dashboards on Linux with X11: native child windows with an OpenGL surface (KNOWN_ISSUES D8).

Measured on 2026-10-29 in WSL Ubuntu 24.04 (XWayland and Xvfb, ``tests/test_dashboard_native_
platform.py`` asks the same on CI): with raster canvases the first dashboard recreated the main
window once per app start; as a native OpenGL child it never does, and OpenGL canvases keep their
viewports. These tests pin the decision on the suite's offscreen platform.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QSurface

from tests.qt_teardown import delete_widgets

# The backend rule these tests rely on (Linux draws Qt Quick through OpenGL unless the user
# chose otherwise) is tested in test_qt_environment.py.


def test_x11_dashboards_embed_natively_and_skip_the_canvas_swap(monkeypatch):
    import app.gui.visualization_manager as vm
    import app.visualization.viz_window as vw

    real = vw.native_child_surface
    monkeypatch.setattr(
        vw, "native_child_surface", lambda platform=None, environ=None: real("xcb", {})
    )
    assert vm.dashboards_embed_natively() is True
    monkeypatch.setattr(
        vw, "native_child_surface", lambda platform=None, environ=None: real("wayland", {})
    )
    assert vm.dashboards_embed_natively() is False


def test_an_x11_dashboard_gets_its_own_opengl_window(qapp, tmp_path, monkeypatch):
    import app.visualization.viz_window as vw
    from tests.viz_window_support import FakeProfile, _Dashboard

    opengl = QSurface.SurfaceType.OpenGLSurface
    real = vw.native_child_surface
    monkeypatch.setattr(
        vw, "native_child_surface", lambda platform=None, environ=None: real("xcb", {})
    )
    csv_path = tmp_path / "root_lengths.csv"
    csv_path.write_text("Image,Length (mm)\n", encoding="utf-8")

    window = _Dashboard(str(csv_path), FakeProfile())
    try:
        assert window.testAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        assert window.windowHandle() is not None
        assert window.windowHandle().surfaceType() == opengl
    finally:
        delete_widgets([window], close=False, prepare=lambda board: board.cleanup_server())
