from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsView,
    QGraphicsScene,
    QWidget,
    QPushButton,
    QSlider,
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QWheelEvent
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QMessageBox
import cv2
import os


class MagnifyingGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setOptimizationFlags(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing
        )
        self.zoom = 1

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            factor = 1.25
            self.zoom *= factor
        else:
            factor = 0.8
            self.zoom *= factor
        self.scale(factor, factor)


class BasicViewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.web_view = QWebEngineView()
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(25, 500)  # 25% to 500% zoom
        self.zoom_slider.setValue(100)  # 100% zoom by default
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(25)
        self.zoom_slider.valueChanged.connect(self.update_zoom)

        self.zoom_in_button = QPushButton("+")
        self.zoom_out_button = QPushButton("-")
        self.zoom_reset_button = QPushButton("Reset Zoom")

        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_reset_button.clicked.connect(self.reset_zoom)

        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.zoom_reset_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.web_view)
        layout.addLayout(zoom_layout)

    def load_url(self, url):
        self.web_view.load(QUrl.fromLocalFile(url))

    def update_zoom(self):
        zoom_factor = self.zoom_slider.value() / 100
        self.web_view.setZoomFactor(zoom_factor)

    def zoom_in(self):
        self.zoom_slider.setValue(self.zoom_slider.value() + 25)

    def zoom_out(self):
        self.zoom_slider.setValue(self.zoom_slider.value() - 25)

    def reset_zoom(self):
        self.zoom_slider.setValue(100)


class DisplayController:
    def __init__(self, main_window):
        self.main_window = main_window
        self.magnifying_view = None
        self.basic_view_widget = None
        self.current_image = None
        self.current_fake_image = None
        self.html_path = None

    def setup_display_area(self, parent):
        layout = QVBoxLayout(parent)

        self.magnifying_view = MagnifyingGraphicsView()
        layout.addWidget(self.magnifying_view)

        self.basic_view_widget = BasicViewWidget()
        layout.addWidget(self.basic_view_widget)

        self.basic_view_widget.hide()

    def display_selected_image(self, item):
        name = item.text()
        self.current_image = self.main_window.image_manager.get_image_path(name)
        self.current_fake_image = self.main_window.image_manager.get_fake_image_path(
            name
        )
        self.update_display()

    def update_display_mode(self):
        self.update_display()

    def update_display(self):
        view_mode = self.main_window.view_mode_combo.currentText()

        # Hide both views initially
        self.magnifying_view.hide()
        self.basic_view_widget.hide()

        if view_mode == "Basic View":
            self.display_basic_view()
        elif self.current_image:
            self.magnifying_view.show()
            if view_mode == "Single Image":
                self.display_single_image()
            elif view_mode == "Overlay":
                self.display_overlay_image()
            elif view_mode == "Side by Side":
                self.display_side_by_side_images()
        else:
            self.clear_magnifying_view()

    def display_basic_view(self):
        img_manager = self.main_window.image_manager
        if img_manager.has_fake_real_pairs and img_manager.html_path:
            self.basic_view_widget.show()
            self.basic_view_widget.load_url(img_manager.html_path)
            print(
                f"DEBUG: Basic View displayed using HTML file: {img_manager.html_path}"
            )
        else:
            print("DEBUG: Basic View not available. Displaying single image instead.")
            self.display_single_image()

    def display_single_image(self):
        if self.current_image:
            pixmap = QPixmap(self.current_image)
            self.set_magnifying_view_image(pixmap)
        else:
            self.clear_magnifying_view()

    def display_overlay_image(self):
        if not self.current_image or not self.current_fake_image:
            print("DEBUG: Missing real or fake image path")
            return

        real_pixmap = QPixmap(self.current_image)
        fake_image_gray = cv2.imread(self.current_fake_image, cv2.IMREAD_GRAYSCALE)
        if fake_image_gray is None:
            print("DEBUG: Failed to load fake image")
            return

        _, binary_mask = cv2.threshold(
            fake_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        real_image = real_pixmap.toImage()

        if real_image.size() != (binary_mask.shape[1], binary_mask.shape[0]):
            print("DEBUG: Resizing binary mask to match real image.")
            binary_mask = cv2.resize(
                binary_mask,
                (real_image.width(), real_image.height()),
                interpolation=cv2.INTER_NEAREST,
            )

        result_image = QImage(
            real_image.size(), QImage.Format.Format_ARGB32_Premultiplied
        )
        result_image.fill(Qt.GlobalColor.transparent)

        painter = QPainter(result_image)
        painter.drawImage(0, 0, real_image)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        for x in range(real_image.width()):
            for y in range(real_image.height()):
                if binary_mask[y, x] >= 50:
                    result_image.setPixelColor(x, y, QColor(Qt.GlobalColor.black))

        painter.end()

        result_pixmap = QPixmap.fromImage(result_image)
        self.set_magnifying_view_image(result_pixmap)
        print(f"DEBUG: Overlay image displayed. Size: {result_pixmap.size()}")

    def display_side_by_side_images(self):
        real_pixmap = QPixmap(self.current_image)
        if self.current_fake_image:
            fake_pixmap = QPixmap(self.current_fake_image)
            combined_pixmap = QPixmap(real_pixmap.width() * 2, real_pixmap.height())
            combined_pixmap.fill(Qt.GlobalColor.white)
            painter = QPainter(combined_pixmap)
            painter.drawPixmap(0, 0, real_pixmap)
            painter.drawPixmap(real_pixmap.width(), 0, fake_pixmap)
            painter.end()
        else:
            combined_pixmap = real_pixmap

        scaled_pixmap = combined_pixmap.scaled(
            800, 600, Qt.AspectRatioMode.KeepAspectRatio
        )
        self.set_magnifying_view_image(scaled_pixmap)
        print(f"DEBUG: Side-by-side images displayed. Size: {scaled_pixmap.size()}")

    def set_magnifying_view_image(self, pixmap):
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.magnifying_view.setScene(scene)
        self.magnifying_view.fitInView(
            scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def clear_magnifying_view(self):
        scene = QGraphicsScene()
        scene.addText("Select an image to display")
        self.magnifying_view.setScene(scene)
        self.magnifying_view.show()

    def set_html_path(self, html_path):
        self.html_path = html_path
        print(f"DEBUG: HTML path set to {self.html_path}")
