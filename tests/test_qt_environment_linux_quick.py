"""Dashboards on Linux with X11: native child windows with an OpenGL surface (KNOWN_ISSUES D8).

Measured on 2026-10-29 in WSL Ubuntu 24.04 (XWayland and Xvfb, ``tests/test_dashboard_native_
platform.py`` asks the same on CI): with raster canvases the first dashboard recreated the main
window once per app start; as a native OpenGL child it never does, and OpenGL canvases keep their
viewports. These tests pin the decision on the suite's offscreen platform.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QSurface

from app import qt_environment as qe
from tests.qt_teardown import delete_widgets


@pytest.mark.parametrize(
    ("environ", "opengl"),
    [
        ({}, True),  # Qt's Linux default: configure_qt_environment sets no backend there
        ({"QSG_RHI_BACKEND": "opengl"}, True),
        ({"QSG_RHI_BACKEND": " GL "}, True),
        ({"QSG_RHI_BACKEND": "vulkan"}, False),
        ({"QSG_RHI_BACKEND": "d3d11"}, False),
        ({"QT_QUICK_BACKEND": "rhi"}, True),  # Qt Quick's default backend
        ({"QT_QUICK_BACKEND": "software"}, False),  # no RHI at all
        ({"QSG_RHI_BACKEND": "opengl", "QT_QUICK_BACKEND": "software"}, False),
    ],
)
def test_qt_quick_on_linux_draws_through_opengl_unless_the_user_chose_otherwise(environ, opengl):
    assert qe.linux_quick_uses_opengl(environ) is opengl


def test_the_windows_rule_is_unchanged_by_the_linux_one():
    assert qe.windows_quick_uses_direct3d({}) is True
    assert qe.windows_quick_uses_direct3d({"QSG_RHI_BACKEND": "opengl"}) is False
    assert qe.linux_quick_uses_opengl({"QSG_RHI_BACKEND": "d3d11"}) is False


def test_linux_start_up_sets_no_qt_quick_backend():
    """The X11 route relies on Qt's own default (OpenGL): nothing may pin another one."""
    env: dict[str, str] = {}
    qe.configure_qt_environment(env, platform="linux", euid=1000)
    assert "QSG_RHI_BACKEND" not in env and "QT_QUICK_BACKEND" not in env
    assert qe.linux_quick_uses_opengl(env)


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
