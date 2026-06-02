"""Smoke + wiring assertions for the PR4 SPROUTS guided shell.

Runs under an offscreen QApplication. Covers the Welcome screen, the loading
overlay, the shell chrome, and (T4.15) the MainWindow connect-map contract.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# QtWebEngine (imported transitively by the visualization modules that
# MainWindow pulls in) requires either an early QtWebEngineWidgets import or
# AA_ShareOpenGLContexts set before the QApplication exists. Do that here so the
# MainWindow contract test (T4.15) can construct the window offscreen.
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def test_welcome_widget_imports(app):
    from app.gui.welcome_screen import WelcomeWidget

    calls = []
    w = WelcomeWidget(on_get_started=lambda: calls.append(1))
    assert w is not None


def test_welcome_get_started_callback(app):
    from app.gui.welcome_screen import WelcomeWidget

    calls = []
    w = WelcomeWidget(on_get_started=lambda: calls.append(1))
    # Find the primary Browse button and click it.
    browse = w.findChild(QPushButton, "primaryButton")
    assert browse is not None
    browse.click()
    assert calls == [1]


def test_loading_overlay_imports(app):
    from app.gui.loading_overlay import LoadingOverlay

    ov = LoadingOverlay()
    ov.set_filename("foo.png")
    ov.set_count(3, 10)
    ov.set_progress(50)
    ov.start()
    ov.hide()


# --------------------------------------------------------------------------- #
#  T4.15 — MainWindow connect-map contract
# --------------------------------------------------------------------------- #
_CONNECT_MAP_ATTRS = [
    "load_images_button",
    "generate_mask_button",
    "generate_button",
    "toggle_skeleton_correction_button",
    "toggle_mask_tracing_button",
    "calculate_length_button",
    "calculate_area_button",
    "visualize_root_length_button",
    "visualize_root_area_button",
    "view_mode_combo",
    "expand_all_button",
    "collapse_all_button",
    "file_list",
]


def test_main_window_contract(app):
    from app.gui.main_window import MainWindow

    if not hasattr(MainWindow, "_pr4_shell_ready"):
        pytest.skip("shell (app_stack) not wired yet — enabled at T4.5/T4.15")

    mw = MainWindow()
    try:
        for attr in _CONNECT_MAP_ATTRS:
            assert hasattr(mw, attr), f"missing {attr}"
        assert mw.view_mode_combo.count() == 3
        assert mw.right_panel.count() == 4
        assert mw.app_stack.count() >= 2
    finally:
        mw.close()
        mw.deleteLater()
