from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QColorDialog,
    QScrollArea,
)
from PyQt6.QtGui import (
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QCursor,
    QImage,
    QBrush,
    QTransform,
    qRgba,
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
import os
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    binary_closing,
    gaussian_filter,
)
import cv2


def is_point_in_closed_area(arr, pos):
    # Convert the image to grayscale and apply thresholding to binary
    gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if the point is inside any contour
    for contour in contours:
        if cv2.pointPolygonTest(contour, (pos.x(), pos.y()), False) >= 0:
            return True
    return False


class MaskTracingInterface(QWidget):
    mask_saved = pyqtSignal(str)
    mask_cleared = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = ""
        self.mask_directory = ""
        self.zoom_factor = 1.0
        self.undo_stack = []
        self.redo_stack = []
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Scroll area for image and mask
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_container = QWidget()
        self.scroll_area.setWidget(self.image_container)
        main_layout.addWidget(self.scroll_area)

        image_layout = QVBoxLayout(self.image_container)
        image_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_mask_label = QLabel()
        self.image_mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(self.image_mask_label)

        # Controls
        controls_layout = QHBoxLayout()

        self.brush_button = QPushButton("Brush")
        self.brush_button.setCheckable(True)
        self.brush_button.setChecked(True)
        controls_layout.addWidget(self.brush_button)

        self.eraser_button = QPushButton("Eraser")
        self.eraser_button.setCheckable(True)
        controls_layout.addWidget(self.eraser_button)

        self.fill_button = QPushButton("Fill")
        self.fill_button.setCheckable(True)
        controls_layout.addWidget(self.fill_button)

        self.clear_button = QPushButton("Clear Mask")
        self.clear_button.clicked.connect(self.clear_mask)
        controls_layout.addWidget(self.clear_button)

        self.save_button = QPushButton("Save Mask")
        self.save_button.clicked.connect(self.save_mask)
        controls_layout.addWidget(self.save_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        controls_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        controls_layout.addWidget(self.redo_button)

        self.size_label = QLabel("Brush Size: 5")
        controls_layout.addWidget(self.size_label)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 50)
        self.size_slider.setValue(5)
        self.size_slider.valueChanged.connect(self.update_brush_size)
        controls_layout.addWidget(self.size_slider)

        self.opacity_label = QLabel("Opacity: 0%")
        controls_layout.addWidget(self.opacity_label)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(0)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        controls_layout.addWidget(self.opacity_slider)

        self.zoom_label = QLabel("Zoom: 100%")
        controls_layout.addWidget(self.zoom_label)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        controls_layout.addWidget(self.zoom_slider)

        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

        # Drawing attributes
        self.last_point = QPoint()
        self.drawing = False
        self.brush_color = QColor(Qt.GlobalColor.white)
        self.brush_size = 5
        self.brush_opacity = 0.5
        self.mask_pixmap = None
        self.image_pixmap = None

        # Set custom cursor
        self.setCursor(self.create_cursor(self.brush_size))

    def create_cursor(self, size):
        cursor_size = max(32, size * 2)
        cursor_pixmap = QPixmap(cursor_size, cursor_size)
        cursor_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(cursor_pixmap)
        painter.setPen(QPen(Qt.GlobalColor.white, 1.0, Qt.PenStyle.SolidLine))

        painter.drawEllipse(
            cursor_size // 2 - size // 2, cursor_size // 2 - size // 2, size, size
        )
        painter.setPen(QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine))
        painter.drawEllipse(
            cursor_size // 2 - size // 2 - 1,
            cursor_size // 2 - size // 2 - 1,
            size + 2,
            size + 2,
        )

        # Draw crosshair
        painter.drawLine(cursor_size // 2, 0, cursor_size // 2, cursor_size)
        painter.drawLine(0, cursor_size // 2, cursor_size, cursor_size // 2)

        painter.end()

        return QCursor(cursor_pixmap, cursor_size // 2, cursor_size // 2)

    def load_image(self, image_path):
        self.current_image_path = image_path
        self.mask_directory = os.path.join(os.path.dirname(image_path), "mask")

        self.image_pixmap = QPixmap(image_path)
        self.mask_pixmap = QPixmap(self.image_pixmap.size())
        self.mask_pixmap.fill(Qt.GlobalColor.transparent)

        # Load existing mask if it exists
        mask_path = os.path.join(self.mask_directory, os.path.basename(image_path))
        if os.path.exists(mask_path):
            self.mask_pixmap = QPixmap(mask_path)

        self.update_display()

    def update_display(self):
        if self.image_pixmap and self.mask_pixmap:
            combined_pixmap = QPixmap(self.image_pixmap.size())
            combined_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(combined_pixmap)
            painter.drawPixmap(0, 0, self.image_pixmap)
            painter.setOpacity(self.brush_opacity)
            painter.drawPixmap(0, 0, self.mask_pixmap)
            painter.end()

            scaled_pixmap = combined_pixmap.scaled(
                combined_pixmap.size() * self.zoom_factor,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_mask_label.setPixmap(scaled_pixmap)
            self.image_mask_label.setFixedSize(scaled_pixmap.size())

            # Center the image in the scroll area
            self.image_container.setFixedSize(scaled_pixmap.size())
            self.scroll_area.setWidgetResizable(False)
            self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # def create_cursor(self, size):
    #     cursor_size = max(32, size * 2)
    #     cursor_pixmap = QPixmap(cursor_size, cursor_size)
    #     cursor_pixmap.fill(Qt.GlobalColor.transparent)

    #     painter = QPainter(cursor_pixmap)
    #     painter.setPen(QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.SolidLine))
    #     painter.drawEllipse(0, 0, size, size)
    #     painter.setPen(QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine))
    #     painter.drawEllipse(0, 0, size, size)

    #     # Draw crosshair
    #     painter.drawLine(size // 2, 0, size // 2, size)
    #     painter.drawLine(0, size // 2, size, size // 2)

    #     painter.end()

    #     return QCursor(cursor_pixmap, size // 2, size // 2)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.mask_pixmap:
            self.drawing = True
            self.save_for_undo()
            pos = self.map_to_image(event.pos())
            if self.fill_button.isChecked():
                self.flood_fill(pos)
            else:
                self.draw_point(pos)

    def mouseMoveEvent(self, event):
        if (
            event.buttons()
            and Qt.MouseButton.LeftButton
            and self.drawing
            and self.mask_pixmap
        ):
            self.draw_point(self.map_to_image(event.pos()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.last_point = None

    def map_to_image(self, pos):
        # Calculate the offset of the image within the scroll area
        image_rect = self.image_mask_label.rect()
        scroll_rect = self.scroll_area.viewport().rect()
        offset = QPoint(
            max(0, (scroll_rect.width() - image_rect.width()) // 2),
            max(0, (scroll_rect.height() - image_rect.height()) // 2),
        )

        # Adjust the position based on scroll area's viewport position and image offset
        adjusted_pos = (
            pos
            - self.image_mask_label.pos()
            - offset
            + self.scroll_area.viewport().pos()
        )

        # Scale the position based on zoom factor
        scaled_pos = adjusted_pos / self.zoom_factor

        # Ensure the position is within the image bounds
        scaled_pos.setX(max(0, min(scaled_pos.x(), self.mask_pixmap.width() - 1)))
        scaled_pos.setY(max(0, min(scaled_pos.y(), self.mask_pixmap.height() - 1)))

        # Adjust for the cursor hotspot
        cursor_offset = QPoint(self.brush_size // 2, self.brush_size // 2)
        return scaled_pos - cursor_offset

    def draw_point(self, pos):
        if not self.mask_pixmap:
            return

        painter = QPainter(self.mask_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Create a temporary pixmap for the current stroke
        temp_pixmap = QPixmap(self.mask_pixmap.size())
        temp_pixmap.fill(Qt.GlobalColor.transparent)
        temp_painter = QPainter(temp_pixmap)
        temp_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Set up the pen for full opacity
        pen = QPen(
            self.brush_color,
            self.brush_size,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        pen.setWidthF(self.brush_size)
        temp_painter.setPen(pen)
        temp_painter.setOpacity(self.brush_opacity)

        # Draw the stroke with full opacity
        if self.last_point:
            temp_painter.drawLine(self.last_point, pos)
        else:
            temp_painter.drawPoint(pos)

        temp_painter.end()

        # # Apply the stroke to the main mask pixmap with the desired opacity
        # painter.setOpacity(self.brush_opacity)

        if self.eraser_button.isChecked():
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_DestinationOut
            )
        else:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )

        painter.drawPixmap(0, 0, temp_pixmap)
        painter.end()

        self.last_point = pos
        self.update_display()

    def update_brush_size(self, value):
        self.brush_size = value
        self.size_label.setText(f"Brush Size: {value}")
        self.setCursor(self.create_cursor(self.brush_size))

    def update_opacity(self, value):
        # Map the 0-100 range to 51-100
        actual_opacity = (value / 100) * 49 + 51
        self.brush_opacity = actual_opacity / 100
        self.opacity_label.setText(
            f"Opacity: {value}%"
        )  # Display the original 0-100 value
        self.update_display()

    def update_zoom(self, value):
        self.zoom_factor = value / 100
        self.zoom_label.setText(f"Zoom: {value}%")
        self.update_display()

    # def save_mask(self):
    #     if self.mask_pixmap and self.current_image_path:
    #         os.makedirs(self.mask_directory, exist_ok=True)
    #         save_path = os.path.normpath(
    #             os.path.join(
    #                 self.mask_directory, os.path.basename(self.current_image_path)
    #             )
    #         )

    #         binary_mask = self.mask_pixmap.toImage().convertToFormat(
    #             QImage.Format.Format_Mono
    #         )
    #         binary_mask.save(save_path, "PNG")

    #         self.mask_saved.emit(self.current_image_path)

    def save_mask(self):
        if self.mask_pixmap and self.current_image_path:
            os.makedirs(self.mask_directory, exist_ok=True)
            save_path = os.path.normpath(
                os.path.join(
                    self.mask_directory, os.path.basename(self.current_image_path)
                )
            )

            # Convert QPixmap to QImage
            mask_image = self.mask_pixmap.toImage()

            # Convert QImage to numpy array
            width = mask_image.width()
            height = mask_image.height()
            ptr = mask_image.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

            # Extract the alpha channel (assumed to be the mask)
            alpha = arr[:, :, 3]

            # Apply threshold to convert to binary (0, 1)
            threshold_value = 1  # Capture any non-zero alpha values
            binary_mask = (alpha > threshold_value).astype(np.uint8)

            # Apply morphological operations
            structure = np.ones((3, 3))
            opened_mask = binary_opening(binary_mask, structure=structure)
            smoothed_mask = binary_closing(opened_mask, structure=structure)

            # Create a new QImage with the binary mask
            binary_image = QImage(width, height, QImage.Format.Format_Mono)
            for y in range(height):
                for x in range(width):
                    value = 1 if smoothed_mask[y, x] else 0
                    binary_image.setPixel(x, y, value)

            # Save the binary mask
            binary_image.save(save_path, "PNG")

            # Emit signal indicating the mask has been saved
            self.mask_saved.emit(self.current_image_path)

            print(f"Mask saved to {save_path}")  # Debug print
            print(f"Alpha channel min: {alpha.min()}, max: {alpha.max()}")
            print(f"Binary mask sum: {binary_mask.sum()}")
            print(f"Smoothed mask sum: {smoothed_mask.sum()}")

    ############################################
    # Code to Svae as GrayScale
    ###############################################

    # def save_mask(self):
    #     if self.mask_pixmap and self.current_image_path:
    #         os.makedirs(self.mask_directory, exist_ok=True)
    #         save_path = os.path.normpath(
    #             os.path.join(
    #                 self.mask_directory, os.path.basename(self.current_image_path)
    #             )
    #         )

    #         # Convert QPixmap to QImage
    #         mask_image = self.mask_pixmap.toImage()

    #         # Convert QImage to numpy array
    #         width = mask_image.width()
    #         height = mask_image.height()
    #         ptr = mask_image.bits()
    #         ptr.setsize(height * width * 4)
    #         arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

    #         # Extract the alpha channel (assumed to be the mask)
    #         alpha = arr[:, :, 3]

    #         # Use a very low threshold to capture any drawn pixels
    #         threshold_value = 1  # Capture any non-zero alpha values
    #         binary_mask = (alpha > threshold_value).astype(np.uint8)

    #         # Apply morphological opening (smooths jagged edges, removes noise)
    #         structure = np.ones(
    #             (3, 3)
    #         )  # Structuring element for morphological operation
    #         opened_mask = binary_opening(binary_mask, structure=structure)

    #         # Apply morphological closing (fills small holes, smooths edges)
    #         smoothed_mask = binary_closing(opened_mask, structure=structure)

    #         # Convert back to a 255-scale mask for saving
    #         final_mask = (smoothed_mask * 255).astype(np.uint8)

    #         # Create a new QImage with the smoothed mask
    #         binary_image = QImage(width, height, QImage.Format.Format_Grayscale8)
    #         for y in range(height):
    #             for x in range(width):
    #                 value = int(final_mask[y, x])  # Convert numpy.uint8 to int
    #                 binary_image.setPixel(x, y, value)

    #         # Save the binary mask

    #         binary_image.save(save_path, "PNG")

    #         # Emit signal indicating the mask has been saved
    #         self.mask_saved.emit(self.current_image_path)

    #         print(f"Mask saved to {save_path}")  # Debug print
    #         print(f"Alpha channel min: {alpha.min()}, max: {alpha.max()}")
    #         print(f"Binary mask sum: {binary_mask.sum()}")
    #         print(f"Final mask sum: {final_mask.sum()}")

    def clear_mask(self):
        if self.mask_pixmap:
            self.save_for_undo()
            self.mask_pixmap.fill(Qt.GlobalColor.transparent)
            self.update_display()

            mask_path = os.path.join(
                self.mask_directory, os.path.basename(self.current_image_path)
            )
            if os.path.exists(mask_path):
                os.remove(mask_path)

            self.mask_cleared.emit(self.current_image_path)

    def save_for_undo(self):
        if self.mask_pixmap:
            self.undo_stack.append(self.mask_pixmap.copy())
            self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.mask_pixmap.copy())
            self.mask_pixmap = self.undo_stack.pop()
            self.update_display()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.mask_pixmap.copy())
            self.mask_pixmap = self.redo_stack.pop()
            self.update_display()

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_slider.setValue(min(self.zoom_slider.value() + 10, 200))
            else:
                self.zoom_slider.setValue(max(self.zoom_slider.value() - 10, 10))
            event.accept()
        else:
            super().wheelEvent(event)

    def flood_fill(self, pos):
        if not self.mask_pixmap:
            return

        # Convert QPixmap to QImage for pixel manipulation
        image = self.mask_pixmap.toImage()
        width, height = image.width(), image.height()

        # Create a numpy array from the image for faster processing
        buffer = image.bits().asstring(width * height * 4)
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4)).copy()

        # Define the fill color (white with current opacity)
        fill_color = QColor(Qt.GlobalColor.white)
        fill_color.setAlpha(int(self.brush_opacity * 255))
        fill_color_rgba = fill_color.getRgb()

        # Get the color of the clicked pixel
        target_color = tuple(arr[pos.y(), pos.x()])

        if target_color == fill_color_rgba:
            return  # If the target color is already the fill color, return

        # Tolerance threshold for comparing colors to avoid tiny differences (due to anti-aliasing)
        def color_match(c1, c2, tolerance=10):
            return all(abs(c1[i] - c2[i]) <= tolerance for i in range(3))

        # Morphological operation to close small gaps in the boundary
        mask = np.all(arr[:, :, :3] != fill_color_rgba[:3], axis=-1)
        closed_mask = binary_dilation(
            binary_erosion(mask)
        )  # Close gaps in the boundary

        # Check if the point is in a closed area before filling
        if not is_point_in_closed_area(arr, pos):
            print("Flood fill denied: point is outside a closed area.")
            return

        # Stack-based flood fill algorithm with color tolerance check
        def stack_based_fill(start_x, start_y):
            stack = [(start_x, start_y)]
            while stack:
                x, y = stack.pop()
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                # Ensure the pixel matches the target color with a tolerance
                if color_match(arr[y, x], target_color) and closed_mask[y, x]:
                    arr[y, x] = fill_color_rgba  # Fill the pixel
                    # Add neighboring pixels to the stack
                    stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

        # Start the fill process from the clicked position
        stack_based_fill(pos.x(), pos.y())

        # Convert the numpy array back to QImage
        bytes_per_line = width * 4
        result_image = QImage(
            arr.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888
        )

        # Update the mask pixmap
        self.mask_pixmap = QPixmap.fromImage(result_image)
        self.update_display()
