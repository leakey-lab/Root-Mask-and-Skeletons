from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QColorDialog,
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
import os


class MaskTracingInterface(QWidget):
    mask_saved = pyqtSignal(str)  # Signal to emit when a mask is saved
    mask_cleared = pyqtSignal(str)  # Signal to emit when a mask is cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = ""
        self.mask_directory = ""
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Image display area
        self.image_container = QWidget()
        self.image_container.setFixedSize(800, 600)
        self.image_label = QLabel(self.image_container)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_label = QLabel(self.image_container)
        self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_container)

        # Controls
        controls_layout = QHBoxLayout()

        self.brush_button = QPushButton("Brush")
        self.brush_button.setCheckable(True)
        self.brush_button.setChecked(True)
        controls_layout.addWidget(self.brush_button)

        self.eraser_button = QPushButton("Eraser")
        self.eraser_button.setCheckable(True)
        controls_layout.addWidget(self.eraser_button)

        self.clear_button = QPushButton("Clear Mask")
        self.clear_button.clicked.connect(self.clear_mask)
        controls_layout.addWidget(self.clear_button)

        self.save_button = QPushButton("Save Mask")
        self.save_button.clicked.connect(self.save_mask)
        controls_layout.addWidget(self.save_button)

        self.color_button = QPushButton("Brush Color")
        self.color_button.clicked.connect(self.choose_color)
        controls_layout.addWidget(self.color_button)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 50)
        self.size_slider.setValue(5)
        controls_layout.addWidget(self.size_slider)

        self.size_label = QLabel("Brush Size: 5")
        self.size_slider.valueChanged.connect(self.update_brush_size)
        controls_layout.addWidget(self.size_label)

        layout.addLayout(controls_layout)
        self.setLayout(layout)

        # Drawing attributes
        self.last_point = QPoint()
        self.drawing = False
        self.brush_color = QColor(Qt.GlobalColor.white)
        self.brush_size = 5
        self.mask_pixmap = None

    def load_image(self, image_path):
        self.current_image_path = image_path
        self.mask_directory = os.path.join(os.path.dirname(image_path), "mask")

        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setFixedSize(pixmap.size())
        self.mask_label.setFixedSize(pixmap.size())
        self.mask_pixmap = QPixmap(pixmap.size())
        self.mask_pixmap.fill(Qt.GlobalColor.transparent)

        # Load existing mask if it exists
        mask_path = os.path.join(self.mask_directory, os.path.basename(image_path))
        if os.path.exists(mask_path):
            self.mask_pixmap = QPixmap(mask_path)

        self.mask_label.setPixmap(self.mask_pixmap)
        self.image_container.setFixedSize(pixmap.size())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.mask_pixmap:
            self.drawing = True
            self.last_point = event.pos() - self.mask_label.pos()

    def mouseMoveEvent(self, event):
        if (
            event.buttons()
            and Qt.MouseButton.LeftButton
            and self.drawing
            and self.mask_pixmap
        ):
            painter = QPainter(self.mask_pixmap)
            painter.setPen(
                QPen(
                    self.brush_color,
                    self.brush_size,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            if self.eraser_button.isChecked():
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_Clear
                )
            current_point = event.pos() - self.mask_label.pos()
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.mask_label.setPixmap(self.mask_pixmap)
            painter.end()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.brush_color = color

    def update_brush_size(self, value):
        self.brush_size = value
        self.size_label.setText(f"Brush Size: {value}")

    def save_mask(self):
        if self.mask_pixmap and self.current_image_path:
            os.makedirs(self.mask_directory, exist_ok=True)
            save_path = os.path.normpath(
                os.path.join(
                    self.mask_directory, os.path.basename(self.current_image_path)
                )
            )
            self.mask_pixmap.save(save_path, "JPG")
            self.mask_saved.emit(
                self.current_image_path
            )  # Emit signal when mask is saved

    def clear_mask(self):
        if self.mask_pixmap:
            self.mask_pixmap.fill(Qt.GlobalColor.transparent)
            self.mask_label.setPixmap(self.mask_pixmap)

            # Delete the saved mask file if it exists
            mask_path = os.path.join(
                self.mask_directory, os.path.basename(self.current_image_path)
            )
            if os.path.exists(mask_path):
                os.remove(mask_path)

            # Emit the mask_cleared signal
            self.mask_cleared.emit(self.current_image_path)
