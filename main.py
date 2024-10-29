import sys
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow
from PyQt6.QtGui import QPalette, QColor
import os


def apply_stylesheet(app):
    with open("./themes/dark_theme.qss", "r") as f:
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
    apply_stylesheet(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
