import sys
import logging
from PyQt6.QtWidgets import QApplication
from app.gui.main_window import MainWindow
from PyQt6.QtGui import QPalette, QColor, QIcon
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualization_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


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
    os.environ["QT_ENABLE_DIRECTWRITE"] = "0"
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
