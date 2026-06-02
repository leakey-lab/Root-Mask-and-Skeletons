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

# Hardware WebGL path: let Chromium use its default ANGLE renderer, translating
# GLES -> Direct3D 11 (matches QSG_RHI_BACKEND=d3d11 above). We previously forced
# --disable-gpu here, which disabled ANGLE and dropped WebEngine to software GLES2
# ("Failed to create GLES3 context, fallback to GLES2") and slow plotly WebGL.
# Re-enabling the GPU + pinning the D3D11 ANGLE backend restores hardware GLES3.
# If this regresses to a black WebEngine surface on some GPU/driver combos, the
# fallback is to disable WebGL at the Plotly layer (render_mode="svg") rather
# than re-disabling the whole GPU.
_chromium_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
for _flag in ("--use-angle=d3d11",):
    if _flag not in _chromium_flags:
        _chromium_flags = f"{_chromium_flags} {_flag}".strip()
if _chromium_flags:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = _chromium_flags

# Logging must be configured before any app module is imported so that
# module-level loggers (e.g. config._auto_skeleton_batch) are captured.
from app.logging_config import setup_logging  # noqa: E402
setup_logging()
import logging  # noqa: E402
logger = logging.getLogger(__name__)

from PyQt6.QtGui import QColor, QIcon, QPalette  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.gui.main_window import MainWindow  # noqa: E402


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        logger.debug("Running from PyInstaller bundle: _MEIPASS=%s", base_path)
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def apply_stylesheet(app):
    theme_path = resource_path("resources/themes/dark_theme.qss")
    with open(theme_path, "r") as f:
        stylesheet = f.read()

    app.setStyleSheet(stylesheet)

    # Application palette — SPROUTS refined-dark tokens (styles.css :root),
    # purple accent #c39af6 primary, purple #b794f6 selection.
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#15161c"))           # --bg-0
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#eceef5"))       # --text
    palette.setColor(QPalette.ColorRole.Base, QColor("#1b1c24"))             # --bg-1
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#21232e"))    # --bg-2
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1b1c24"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#eceef5"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#eceef5"))
    palette.setColor(QPalette.ColorRole.Disabled, QPalette.ColorRole.Text, QColor("#686c82"))  # --text-faint
    palette.setColor(QPalette.ColorRole.Button, QColor("#282a37"))           # --bg-3
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#eceef5"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#79c0e8"))             # --info
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#b794f6"))        # --sel
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#15161c"))

    app.setPalette(palette)


if __name__ == "__main__":
    logger.info("Starting Root-Mask-and-Skeletons (QSG_RHI_BACKEND=%s)", os.environ.get("QSG_RHI_BACKEND"))
    app = QApplication(sys.argv)

    # Set application icon
    icon_path = resource_path("resources/app_icon.ico")
    logger.debug("Icon path: %s (exists=%s)", icon_path, os.path.exists(icon_path))
    if os.path.exists(icon_path):
        icon = QIcon(icon_path)
        app.setWindowIcon(icon)
        # Set icon for all windows
        QIcon.setThemeSearchPaths([os.path.dirname(icon_path)])

    apply_stylesheet(app)
    logger.info("Creating MainWindow")
    window = MainWindow()
    window.show()
    logger.info("Entering Qt event loop")
    sys.exit(app.exec())
