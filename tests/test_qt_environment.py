"""Qt / Chromium start-up environment per OS (``app/qt_environment.py``, owner decision O21).

Windows pins the D3D11 RHI and the D3D11 ANGLE backend; Linux and macOS take Qt's own
defaults (OpenGL and Metal), because a D3D11 backend there leaves the dashboards black;
Chromium's sandbox is off only for root; macOS keeps text at its designed size. A value the
user exported wins. The dashboards' surface follows the Qt Quick backend in use. The module
is stdlib only: ``main.py`` imports it before Qt.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from app import qt_environment as qe

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_windows_sets_d3d11_and_angle_once():
    env = {"QTWEBENGINE_CHROMIUM_FLAGS": "--foo"}
    qe.configure_qt_environment(env, "win32")
    qe.configure_qt_environment(env, "win32")
    assert env["QSG_RHI_BACKEND"] == "d3d11"
    assert env["QTWEBENGINE_CHROMIUM_FLAGS"] == "--foo --use-angle=d3d11"


def test_windows_returns_what_it_set():
    env = {}
    assert qe.configure_qt_environment(env, "win32") == {
        "QSG_RHI_BACKEND": "d3d11",
        "QTWEBENGINE_CHROMIUM_FLAGS": "--use-angle=d3d11",
    }
    assert qe.configure_qt_environment(env, "win32") == {}


def test_windows_keeps_a_users_backend():
    env = {"QSG_RHI_BACKEND": "opengl"}
    qe.configure_qt_environment(env, "win32")
    assert env["QSG_RHI_BACKEND"] == "opengl"


def test_linux_takes_qt_defaults():
    env = {}
    assert qe.configure_qt_environment(env, "linux", euid=1000) == {}
    assert env == {}


def test_linux_root_disables_the_chromium_sandbox():
    env = {}
    qe.configure_qt_environment(env, "linux", euid=0)
    assert env == {"QTWEBENGINE_DISABLE_SANDBOX": "1"}


def test_darwin_takes_metal_and_96_dpi_fonts():
    env = {}
    qe.configure_qt_environment(env, "darwin", euid=0)
    assert env == {"QT_FONT_DPI": "96"}


@pytest.mark.parametrize(
    ("environ", "d3d11"),
    [
        ({}, True),  # what configure_qt_environment sets on Windows, and Qt's Windows default
        ({"QSG_RHI_BACKEND": "d3d11"}, True),
        ({"QSG_RHI_BACKEND": " D3D11 "}, True),
        ({"QSG_RHI_BACKEND": "opengl"}, False),
        ({"QSG_RHI_BACKEND": "vulkan"}, False),
        ({"QT_QUICK_BACKEND": "rhi"}, True),  # Qt Quick's default backend
        ({"QT_QUICK_BACKEND": "software"}, False),  # no RHI at all
        ({"QSG_RHI_BACKEND": "d3d11", "QT_QUICK_BACKEND": "software"}, False),
    ],
)
def test_qt_quick_on_windows_draws_through_d3d11_unless_the_user_chose_otherwise(environ, d3d11):
    assert qe.windows_quick_uses_direct3d(environ) is d3d11


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


def test_style_is_native_on_windows_and_fusion_elsewhere():
    assert qe.qt_style_for("win32") is None
    assert qe.qt_style_for("linux") == qe.qt_style_for("darwin") == "Fusion"


def test_desktop_file_name_matches_the_script_name():
    """Wayland compositors match the window to its icon through the desktop file name: the
    name of the script ``pip install .`` makes for ``main:main``."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        scripts = tomllib.load(fh)["project"]["scripts"]
    assert scripts.get(qe.DESKTOP_FILE_NAME) == "main:main", scripts


def test_the_module_is_qt_free():
    src = Path(qe.__file__).read_text(encoding="utf-8")
    assert "PyQt6" not in src and "torch" not in src
