import os
import sys

# NOTE (Qt6 / Windows):
# - QWebEngineView is backed by a QQuickWidget internally.
# - QQuickWidget requires a QRhi-backed composition path (D3D11 by default on Windows).
# - Forcing Qt to use OpenGL (e.g. via QT_OPENGL=software) or mixing in QOpenGLWidget
#   in the same top-level window can lead to a black WebEngine surface and warnings like:
#   "OpenGL is not compatible with this QQuickWidget" / "Failed to get a QRhi".
#
# We prefer the default D3D11 RHI path for stability.
os.environ.setdefault("QSG_RHI_BACKEND", "d3d11")

# If the user hasn't configured Chromium flags, add a conservative GPU-disable set.
# (This helps on some Windows GPU driver combos without forcing OpenGL.)
_chromium_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
for _flag in ("--disable-gpu", "--disable-gpu-compositing"):
    if _flag not in _chromium_flags:
        _chromium_flags = f"{_chromium_flags} {_flag}".strip()
if _chromium_flags:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = _chromium_flags

from PyQt6.QtGui import QColor, QIcon, QPalette  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.gui.main_window import MainWindow  # noqa: E402


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def apply_stylesheet(app):
    theme_path = resource_path("resources/themes/dark_theme.qss")
    with open(theme_path, "r") as f:
        stylesheet = f.read()

    app.setStyleSheet(stylesheet)

    # Set the application palette for a consistent dark theme
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#282a36"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#f8f8f2"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#282a36"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#3d4251"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#282a36"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f8f8f2"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#f8f8f2"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#44475a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f8f8f2"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#8be9fd"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#bd93f9"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))

    app.setPalette(palette)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application icon
    icon_path = resource_path("resources/app_icon.ico")
    if os.path.exists(icon_path):
        icon = QIcon(icon_path)
        app.setWindowIcon(icon)
        # Set icon for all windows
        QIcon.setThemeSearchPaths([os.path.dirname(icon_path)])
    
    apply_stylesheet(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
