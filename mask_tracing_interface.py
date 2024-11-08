from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QScrollArea,
    QButtonGroup,
    QGroupBox,
)
from PyQt6.QtGui import (
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QCursor,
    QImage,
    QBrush,
    QKeySequence,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QRect, QEvent, QPointF
import os
import numpy as np
import cv2
from image_normalization_interface import ImageNormalization, NormalizationControls


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
        self.b_key_pressed = False
        self.scroll_area = None
        self.size_slider = None
        self.zoom_slider = None
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # Reduce overall margins

        # Scroll area for image and mask
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.viewport().installEventFilter(self)

        self.image_mask_label = QLabel()
        self.image_mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_mask_label)
        main_layout.addWidget(self.scroll_area)

        # Bottom control panel with dark background
        control_panel = QWidget()
        control_panel.setStyleSheet(
            """
            QWidget {
                background-color: #1e1e1e;
            }
            QGroupBox {
                border: 1px solid #333333;
                border-radius: 4px;
                margin-top: 4px;
                padding-top: 12px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 0px 5px 0px 5px;
            }
        """
        )

        control_layout = QHBoxLayout(control_panel)
        control_layout.setSpacing(4)  # Reduce spacing between groups
        control_layout.setContentsMargins(2, 1, 2, 1)  # Reduce vertical margins

        # Tools Group
        tools_group = QGroupBox("Tools")
        tools_group.setFixedWidth(100)  # Make tools group wider
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(2)  # Reduce spacing between buttons
        tools_layout.setContentsMargins(4, 2, 4, 2)  # Reduce margins inside group

        # Tool buttons style
        tool_button_style = """
            QPushButton {
                background-color: #2d2d2d;
                border: none;
                border-radius: 4px;
                padding: 4px;
                color: white;
                min-width: 40px;
                max-width: 100px;
                min-height: 28px;
                max-height: 28px;
                font-size: 16px;
            }
            QPushButton:checked {
                background-color: #404040;
            }
            QPushButton:hover {
                background-color: #404040;
            }
        """

        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.setExclusive(True)

        self.brush_button = QPushButton("🖌️ Brush")
        self.eraser_button = QPushButton("🧽 Eraser")
        self.fill_button = QPushButton("🪣 Fill")

        for button in [self.brush_button, self.eraser_button, self.fill_button]:
            button.setStyleSheet(tool_button_style)
            button.setCheckable(True)
            self.tool_button_group.addButton(button)
            tools_layout.addWidget(button)

        self.brush_button.setChecked(True)
        tools_group.setLayout(tools_layout)
        control_layout.addWidget(tools_group)

        # Actions Group
        actions_group = QGroupBox("Actions")
        actions_group.setFixedWidth(100)  # Make actions group wider
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(2)  # Reduce spacing between buttons
        actions_layout.setContentsMargins(4, 2, 4, 2)  # Reduce margins inside group

        action_button_style = """
            QPushButton {
                background-color: #2d2d2d;
                border: none;
                border-radius: 4px;
                padding: 4px;
                color: white;
                min-width: 40px;
                max-width: 100px;
                min-height: 28px;
                max-height: 28px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #404040;
            }
        """

        self.clear_button = QPushButton("Clear Mask")
        self.save_button = QPushButton("Save Mask")
        self.undo_button = QPushButton("⬅️")
        self.redo_button = QPushButton("➡️")

        for button in [
            self.clear_button,
            self.save_button,
            self.undo_button,
            self.redo_button,
        ]:
            button.setStyleSheet(action_button_style)
            actions_layout.addWidget(button)

        actions_group.setLayout(actions_layout)
        control_layout.addWidget(actions_group)

        # Adjustments Group
        adjustments_group = QGroupBox("Adjustments")
        adjustments_layout = QVBoxLayout()
        adjustments_layout.setSpacing(2)  # Reduce spacing between sliders
        adjustments_layout.setContentsMargins(8, 2, 8, 2)  # Reduce vertical margins

        # Slider style
        slider_style = """
            QSlider {
                max-height: 20px;
            }
            QSlider::groove:horizontal {
                border: none;
                height: 4px;
                background: #404040;
                margin: 0px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ff79c6;
                border: none;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #ff92d0;
            }
            QLabel {
                color: white;
                font-size: 11px;
                max-height: 15px;
            }
        """

        # Create sliders with labels
        sliders_data = [
            ("Brush Size", 3, 100, 5),
            ("Opacity", 0, 100, 100),
            ("Zoom", 10, 200, 100),
        ]

        for name, min_val, max_val, default in sliders_data:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(1)  # Minimal spacing between label and slider
            container_layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel(f"{name}: {default}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setStyleSheet(slider_style)
            slider.setRange(min_val, max_val)
            slider.setValue(default)

            container_layout.addWidget(label)
            container_layout.addWidget(slider)
            adjustments_layout.addWidget(container)

            if name == "Brush Size":
                self.size_slider = slider
                self.size_label = label
            elif name == "Opacity":
                self.opacity_slider = slider
                self.opacity_label = label
            else:
                self.zoom_slider = slider
                self.zoom_label = label

        adjustments_group.setLayout(adjustments_layout)
        control_layout.addWidget(adjustments_group)

        # Image Enhancement Controls
        self.norm_controls = NormalizationControls(self)
        control_layout.addWidget(self.norm_controls)

        # Add the control panel to the main layout
        main_layout.addWidget(control_panel)

        # Connect signals
        self.size_slider.valueChanged.connect(self.update_brush_size)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        self.clear_button.clicked.connect(self.clear_mask)
        self.save_button.clicked.connect(self.save_mask)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.norm_controls.apply_button.clicked.connect(self.apply_normalization)

        # Add shortcuts
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.undo)
        self.redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.activated.connect(self.redo)

        # Set the main layout
        self.setLayout(main_layout)

        # Initialize drawing attributes
        self.last_point = QPoint()
        self.drawing = False
        self.brush_color = QColor(Qt.GlobalColor.white)
        self.brush_size = 5
        self.brush_opacity = 1.0
        self.mask_pixmap = None
        self.image_pixmap = None

        # Set custom cursor
        self.setCursor(self.create_cursor(self.brush_size))

    def create_cursor(self, size):
        cursor_size = max(size * 2, 32)  # Ensure the cursor is at least 32x32 pixels
        cursor_pixmap = QPixmap(cursor_size, cursor_size)
        cursor_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(cursor_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the outer circle (white)
        painter.setPen(QPen(Qt.GlobalColor.white, 2, Qt.PenStyle.SolidLine))
        painter.drawEllipse(1, 1, cursor_size - 2, cursor_size - 2)

        # Draw the inner circle (black)
        painter.setPen(QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine))
        painter.drawEllipse(2, 2, cursor_size - 4, cursor_size - 4)

        # Draw the brush size circle
        painter.setPen(QPen(Qt.GlobalColor.red, 1, Qt.PenStyle.DotLine))
        brush_circle_size = min(size, cursor_size - 4)
        offset = (cursor_size - brush_circle_size) // 2
        painter.drawEllipse(offset, offset, brush_circle_size, brush_circle_size)

        # Draw crosshair
        painter.setPen(QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine))
        mid = cursor_size // 2
        painter.drawLine(mid, 0, mid, cursor_size)
        painter.drawLine(0, mid, cursor_size, mid)

        painter.end()

        # Set the hotspot to the center of the cursor
        hotspot = QPoint(cursor_size // 2, cursor_size // 2)

        return QCursor(cursor_pixmap, hotspot.x(), hotspot.y())

    def load_image(self, image_path):
        # Clear previous image and mask data
        self.current_image_path = image_path
        self.mask_directory = os.path.join(os.path.dirname(image_path), "mask")

        # Clear undo/redo stacks
        self.undo_stack.clear()
        self.redo_stack.clear()

        # Load the new image
        self.image_pixmap = QPixmap(image_path)

        # Create a fresh transparent mask
        self.mask_pixmap = QPixmap(self.image_pixmap.size())
        self.mask_pixmap.fill(Qt.GlobalColor.transparent)

        # Reset drawing state
        self.drawing = False
        self.last_point = None

        # Reset zoom and opacity to default values
        if self.zoom_slider:
            self.zoom_slider.setValue(100)
        if self.opacity_slider:
            self.opacity_slider.setValue(0)

        # Load existing mask if it exists
        mask_path = os.path.join(self.mask_directory, os.path.basename(image_path))
        if os.path.exists(mask_path):
            self.mask_pixmap = QPixmap(mask_path)

        # Force update of the display
        self.update_display()

    def update_display(self):
        if self.image_pixmap and self.mask_pixmap:
            # Create a new pixmap for the combined display
            combined_pixmap = QPixmap(self.image_pixmap.size())
            combined_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(combined_pixmap)

            # Always draw the background image at full opacity
            painter.drawPixmap(0, 0, self.image_pixmap)

            # Draw the mask with current opacity
            painter.setOpacity(self.brush_opacity)

            # When erasing, reduce the mask opacity further
            if self.eraser_button.isChecked():
                painter.setOpacity(max(0.3, self.brush_opacity))

            painter.drawPixmap(0, 0, self.mask_pixmap)
            painter.end()

            # Scale the combined image according to zoom factor
            scaled_pixmap = combined_pixmap.scaled(
                combined_pixmap.size() * self.zoom_factor,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            self.image_mask_label.setPixmap(scaled_pixmap)
            self.image_mask_label.setFixedSize(scaled_pixmap.size())
            # Remove this line since image_container no longer exists:
            # self.image_container.setFixedSize(scaled_pixmap.size())
            self.scroll_area.setWidgetResizable(False)
            self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.mask_pixmap:
            self.drawing = True
            self.save_for_undo()
            pos = self.map_to_image(event.pos())

            if self.fill_button.isChecked():
                self.flood_fill(pos)
            elif self.brush_button.isChecked():
                self.draw_point(pos)

    def mouseMoveEvent(self, event):
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self.drawing
            and self.mask_pixmap
        ):
            pos = self.map_to_image(event.pos())
            self.draw_point(pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.last_point = None

    def map_to_image(self, pos):
        """Maps window coordinates to image coordinates accounting for scroll position and widget alignment."""

        # Get scroll bar visibility and scroll values
        h_scroll_visible = self.scroll_area.horizontalScrollBar().isVisible()
        v_scroll_visible = self.scroll_area.verticalScrollBar().isVisible()
        h_scroll = self.scroll_area.horizontalScrollBar().value()
        v_scroll = self.scroll_area.verticalScrollBar().value()

        # Get viewport and widget geometries
        viewport_rect = self.scroll_area.viewport().rect()
        label_rect = self.image_mask_label.rect()

        # Calculate centering offsets if the widget is smaller than viewport
        h_offset = max(0, (viewport_rect.width() - label_rect.width()) // 2)
        v_offset = max(0, (viewport_rect.height() - label_rect.height()) // 2)

        # Convert input position to viewport coordinates
        viewport_pos = self.scroll_area.viewport().mapFrom(self, pos)

        # Adjust for scrolling and centering
        if h_scroll_visible:
            x = viewport_pos.x() + h_scroll - h_offset
        else:
            x = viewport_pos.x() - h_offset

        if v_scroll_visible:
            y = viewport_pos.y() + v_scroll - v_offset
        else:
            y = viewport_pos.y() - v_offset

        # Scale coordinates according to zoom factor
        image_x = int(x / self.zoom_factor)
        image_y = int(y / self.zoom_factor)

        # Ensure coordinates are within image boundaries
        final_x = max(0, min(image_x, self.mask_pixmap.width() - 1))
        final_y = max(0, min(image_y, self.mask_pixmap.height() - 1))

        return QPoint(final_x, final_y)

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

        # Set up the pen and brush
        pen = QPen(
            self.brush_color,
            1,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        temp_painter.setPen(pen)
        temp_painter.setBrush(QBrush(self.brush_color))

        # Draw the stroke
        if self.last_point:
            temp_painter.setPen(
                QPen(
                    self.brush_color,
                    self.brush_size,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            temp_painter.drawLine(self.last_point, pos)
        else:
            diameter = self.brush_size
            top_left = QPoint(pos.x() - diameter // 2, pos.y() - diameter // 2)
            temp_painter.drawEllipse(top_left.x(), top_left.y(), diameter, diameter)

        temp_painter.end()

        # Apply the stroke to the mask
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
        # Map the 0-100 range to 0.3-1.0 range to maintain minimum visibility
        self.brush_opacity = 0.3 + (value / 100) * 0.7
        self.opacity_label.setText(
            f"Opacity: {value}%"
        )  # Display the original 0-100 value
        self.update_display()

    def update_zoom(self, value):
        self.zoom_factor = value / 100
        self.zoom_label.setText(f"Zoom: {value}%")
        self.update_display()

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
            ptr.setsize(height * width * 4)  # Assuming 4 channels (RGBA)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

            # Extract RGB and alpha channels
            rgb = arr[:, :, :3]  # Only RGB channels
            alpha = arr[:, :, 3].astype(float) / 255.0  # Normalize alpha to 0-1 range

            # Create a mask for additions (white areas with any level of opacity)
            additions_mask = np.max(rgb, axis=2) * alpha

            # Create a mask for erasures (dark areas with any level of opacity)
            erasures_mask = np.max(rgb, axis=2) < 254  # Binary mask for erasures

            # Check if a mask already exists
            if os.path.exists(save_path):
                # Load the existing mask in greyscale
                existing_mask = (
                    cv2.imread(save_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
                )

                # Ensure the existing mask has the same dimensions
                if existing_mask.shape != (height, width):
                    existing_mask = cv2.resize(existing_mask, (width, height))

                # Merge the existing mask with new additions and erasures
                merged_mask = existing_mask - erasures_mask + additions_mask
                merged_mask = np.clip(
                    merged_mask, 0, 1
                )  # Ensure values are in 0-1 range
            else:
                # If no existing mask, use the additions mask and apply erasures
                merged_mask = additions_mask - erasures_mask
                merged_mask = np.clip(
                    merged_mask, 0, 1
                )  # Ensure values are in 0-1 range

            # Convert to 8-bit grayscale
            gray_mask = (merged_mask * 255).astype(np.uint8)

            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            opened_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
            final_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

            # Apply threshold to create the final binary mask
            _, binary_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

            # Save the binary mask with high quality
            cv2.imwrite(save_path, binary_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            print(f"DEBUG: Mask saved to {save_path}")

            # Emit signal indicating the mask has been saved
            self.mask_saved.emit(self.current_image_path)

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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_B:
            # Toggle the state of b_key_pressed when the B key is pressed
            self.b_key_pressed = not self.b_key_pressed
            if self.b_key_pressed:
                print("B KEY PRESSED")
            else:
                print("B KEY UNPRESSED")
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        if self.b_key_pressed and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            # Adjust brush size
            delta = event.angleDelta().y()
            if delta > 0:
                self.size_slider.setValue(min(self.size_slider.value() + 1, 50))
            else:
                self.size_slider.setValue(max(self.size_slider.value() - 1, 1))
            event.accept()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Zoom functionality
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_slider.setValue(min(self.zoom_slider.value() + 10, 200))
            else:
                self.zoom_slider.setValue(max(self.zoom_slider.value() - 10, 10))
            event.accept()
        else:
            # Allow normal scrolling
            event.ignore()
            super().wheelEvent(event)

    def eventFilter(self, obj, event):
        if obj == self.scroll_area.viewport() and event.type() == QEvent.Type.Wheel:
            # Convert QEvent to QWheelEvent
            wheel_event = QWheelEvent(
                QPointF(event.position()),  # Convert QPoint to QPointF
                QPointF(event.globalPosition()),  # Convert QPoint to QPointF
                event.pixelDelta(),
                event.angleDelta(),
                event.buttons(),
                event.modifiers(),
                Qt.ScrollPhase.NoScrollPhase,  # Default scroll phase
                False,  # Not inverted
                Qt.MouseEventSource.MouseEventNotSynthesized,  # Default source
            )

            # Send the wheel event to our wheelEvent method
            self.wheelEvent(wheel_event)

            # If the event was accepted by our wheelEvent, we're done
            if wheel_event.isAccepted():
                return True

        # For all other cases, including unhandled wheel events
        return super().eventFilter(obj, event)

    def flood_fill(self, pos):
        if not self.mask_pixmap:
            return

        # Convert QPixmap to QImage for pixel manipulation
        image = self.mask_pixmap.toImage()
        width, height = image.width(), image.height()

        # Create a numpy array from the image for faster processing
        buffer = image.bits().asstring(width * height * 4)
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4)).copy()

        # Convert to grayscale for boundary detection
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)

        # Find contours in the image
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find which contour contains the click point
        point = (pos.x(), pos.y())
        target_contour = None
        for contour in contours:
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                target_contour = contour
                break

        # If no containing contour found, return
        if target_contour is None:
            return

        # Create a mask for the specific contour
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [target_contour], 0, 255, -1)

        # Define the fill color (white with full opacity)
        fill_color = np.array([255, 255, 255, 255], dtype=np.uint8)

        # Create a mask of existing content within the contour
        existing_content = (arr[..., 3] > 0) & (mask > 0)

        # Fill only the area within the contour that isn't already filled
        fill_area = (mask > 0) & ~existing_content

        # Apply the fill
        arr[fill_area] = fill_color

        # Convert the numpy array back to QImage
        bytes_per_line = width * 4
        result_image = QImage(
            arr.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888
        )

        # Save for undo and update the mask pixmap
        self.save_for_undo()
        self.mask_pixmap = QPixmap.fromImage(result_image)
        self.update_display()

    def apply_normalization(self):
        if not self.current_image_path:
            return

        method = self.norm_controls.method_combo.currentText()
        img = cv2.imread(self.current_image_path)

        if method == "CLAHE":
            clip_limit = self.norm_controls.clip_slider.value() / 10.0
            tile_size = self.norm_controls.tile_slider.value()
            enhanced = ImageNormalization.apply_clahe(
                img, clip_limit=clip_limit, tile_size=(tile_size, tile_size)
            )
        else:  # Contrast Stretching
            lower = self.norm_controls.lower_slider.value()
            upper = self.norm_controls.upper_slider.value()
            enhanced = ImageNormalization.apply_contrast_stretching(
                img, lower_percentile=lower, upper_percentile=upper
            )

        # Convert to QPixmap and update display
        rgb_img = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_img.shape
        q_img = QImage(
            rgb_img.data, width, height, width * 3, QImage.Format.Format_RGB888
        )
        self.image_pixmap = QPixmap.fromImage(q_img)
        self.update_display()
