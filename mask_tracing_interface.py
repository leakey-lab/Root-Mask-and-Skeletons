from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QScrollArea,
    QButtonGroup,
    QApplication,
)
from PyQt6.QtGui import (
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QCursor,
    QImage,
    QBrush,
    QFont,
    QKeySequence,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QRect, QEvent, QPointF
import os
import numpy as np
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
        self.b_key_pressed = False
        self.scroll_area = None
        self.size_slider = None
        self.zoom_slider = None
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Scroll area for image and mask
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Install event filter on scroll area's viewport
        self.scroll_area.viewport().installEventFilter(self)

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

        # Create a button group for mutually exclusive selection
        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.setExclusive(True)

        self.brush_button = QPushButton("🖌️")
        self.brush_button.setCheckable(True)
        self.brush_button.setChecked(True)
        self.brush_button.setFont(QFont("Roboto", 30))
        self.tool_button_group.addButton(self.brush_button)
        controls_layout.addWidget(self.brush_button)

        self.eraser_button = QPushButton("🧽")
        self.eraser_button.setCheckable(True)
        self.eraser_button.setFont(QFont("Roboto", 30))
        self.tool_button_group.addButton(self.eraser_button)
        controls_layout.addWidget(self.eraser_button)

        self.fill_button = QPushButton("🪣")
        self.fill_button.setCheckable(True)
        self.fill_button.setFont(QFont("Roboto", 30))
        self.tool_button_group.addButton(self.fill_button)
        controls_layout.addWidget(self.fill_button)

        self.clear_button = QPushButton("Clear Mask")
        self.clear_button.clicked.connect(self.clear_mask)
        controls_layout.addWidget(self.clear_button)

        self.save_button = QPushButton("Save Mask")
        self.save_button.clicked.connect(self.save_mask)
        controls_layout.addWidget(self.save_button)

        self.undo_button = QPushButton("⬅️")
        self.undo_button.setFont(QFont("Roboto", 30))
        self.undo_button.clicked.connect(self.undo)
        controls_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("➡️")
        self.redo_button.setFont(QFont("Roboto", 30))
        self.redo_button.clicked.connect(self.redo)
        controls_layout.addWidget(self.redo_button)

        self.size_label = QLabel("Brush Size: 5")
        controls_layout.addWidget(self.size_label)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 50)
        self.size_slider.setValue(5)
        self.size_slider.valueChanged.connect(self.update_brush_size)
        controls_layout.addWidget(self.size_slider)

        self.opacity_label = QLabel("Opacity: 100%")
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

        # Add shortcuts for undo and redo
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.undo)

        self.redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.activated.connect(self.redo)

        # Drawing attributes
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

        print(
            f"DEBUG: Cursor created with size: {cursor_size}, hotspot: ({hotspot.x()}, {hotspot.y()}), Brush size: {size}"
        )

        return QCursor(cursor_pixmap, hotspot.x(), hotspot.y())

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

    #########################################################
    # map_to_image for MINIZOTRON CAMERA
    #########################################################

    # def map_to_image(self, pos):
    #     # Calculate the offset of the image within the scroll area
    #     image_rect = self.image_mask_label.rect()
    #     scroll_rect = self.scroll_area.viewport().rect()
    #     offset = QPoint(
    #         max(0, (scroll_rect.width() - image_rect.width()) // 2),
    #         max(0, (scroll_rect.height() - image_rect.height()) // 2),
    #     )

    #     # Adjust the position based on scroll area's viewport position and image offset
    #     adjusted_pos = (
    #         pos
    #         - self.image_mask_label.pos()
    #         - offset
    #         + self.scroll_area.viewport().pos()
    #     )

    #     # Scale the position based on zoom factor
    #     scaled_pos = adjusted_pos / self.zoom_factor

    #     # Ensure the position is within the image bounds
    #     scaled_pos.setX(max(0, min(scaled_pos.x(), self.mask_pixmap.width() - 1)))
    #     scaled_pos.setY(max(0, min(scaled_pos.y(), self.mask_pixmap.height() - 1)))

    #     # Adjust for the cursor hotspot
    #     cursor_offset = QPoint(self.brush_size // 2 - 2, self.brush_size // 2 - 2)
    #     final_pos = scaled_pos - cursor_offset

    #     return final_pos

    #####################################################
    # Correct Mapping for new Camera
    #####################################################

    def map_to_image(self, pos):
        # Check if scrollbars are visible
        h_scrollbar_visible = self.scroll_area.horizontalScrollBar().isVisible()
        v_scrollbar_visible = self.scroll_area.verticalScrollBar().isVisible()

        if h_scrollbar_visible or v_scrollbar_visible:
            # Scrollbars are present, use the scroll-based approach
            scroll_x = self.scroll_area.horizontalScrollBar().value()
            scroll_y = self.scroll_area.verticalScrollBar().value()

            # Calculate the position relative to the top-left of the image in the scroll area
            adjusted_pos = QPoint(pos.x() + scroll_x, pos.y() + scroll_y)

            # Calculate the position relative to the image mask label
            relative_x = adjusted_pos.x() - self.image_mask_label.pos().x()
            relative_y = adjusted_pos.y() - self.image_mask_label.pos().y()

            # Adjust for zoom factor
            scaled_x = relative_x / self.zoom_factor
            scaled_y = relative_y / self.zoom_factor

            # Get the cursor's hotspot (center point)
            cursor_hotspot = self.cursor().hotSpot()

            # Adjust the position to account for the cursor hotspot
            final_x = max(
                0,
                min(
                    scaled_x - cursor_hotspot.x() + self.brush_size,
                    self.mask_pixmap.width(),
                ),
            )
            final_y = max(
                0,
                min(
                    scaled_y - cursor_hotspot.y() + self.brush_size,
                    self.mask_pixmap.height(),
                ),
            )
        else:
            # No scrollbars, use the offset-based approach
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
            scaled_pos.setX(max(0, min(scaled_pos.x(), self.mask_pixmap.width())))
            scaled_pos.setY(max(0, min(scaled_pos.y(), self.mask_pixmap.height())))

            # Adjust for the cursor hotspot
            cursor_hotspot = self.cursor().hotSpot()
            final_x = scaled_pos.x() - cursor_hotspot.x() + self.brush_size // 2
            final_y = scaled_pos.y() - cursor_hotspot.y() + self.brush_size // 2

        # Ensure final position is within image bounds
        final_x = max(0, min(final_x, self.mask_pixmap.width()))
        final_y = max(0, min(final_y, self.mask_pixmap.height()))

        final_pos = QPoint(int(final_x), int(final_y))

        return final_pos

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
            1,  # Set pen width to 1 for precise control
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        temp_painter.setPen(pen)
        temp_painter.setBrush(QBrush(self.brush_color))  # Set brush for filling
        temp_painter.setOpacity(self.brush_opacity)

        # Draw the stroke with full opacity
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
            print(
                f"DEBUG: Drawing line from ({self.last_point.x()}, {self.last_point.y()}) to ({pos.x()}, {pos.y()})"
            )
        else:
            # Draw a filled circle with diameter equal to brush size
            diameter = self.brush_size
            top_left = QPoint(pos.x() - diameter // 2, pos.y() - diameter // 2)
            temp_painter.drawEllipse(top_left.x(), top_left.y(), diameter, diameter)
            print(
                f"DEBUG: Drawing point at ({pos.x()}, {pos.y()}) with brush size {self.brush_size}"
            )

        temp_painter.end()

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
        self.brush_opacity = value / 100
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
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

            # Extract RGB and alpha channels
            rgb = arr[:, :, :3]
            alpha = arr[:, :, 3].astype(float) / 255.0  # Normalize alpha to 0-1 range

            # Create a mask for additions (white areas with any level of opacity)
            additions_mask = np.max(rgb, axis=2) * alpha

            # Create a mask for erasures (dark areas with any level of opacity)
            erasures_mask = np.max(rgb, axis=2) < 254  # This makes it binary

            # Check if a mask already exists
            if os.path.exists(save_path):
                # Load the existing mask
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

            # Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            opened_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
            smoothed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

            # Apply threshold to create final binary mask
            _, binary_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

            # Save the binary mask
            cv2.imwrite(save_path, binary_mask)

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

        # Define the fill color (white with full opacity)
        fill_color = QColor(Qt.GlobalColor.white)
        fill_color.setAlpha(255)  # Ensure full opacity for boundaries
        fill_color_rgba = fill_color.getRgb()

        # Get the color of the clicked pixel
        target_color = tuple(arr[pos.y(), pos.x()])

        if target_color == fill_color_rgba:
            return  # If the target color is already the fill color, return

        # Tolerance threshold for comparing colors to avoid tiny differences (due to anti-aliasing)
        def color_match(c1, c2, tolerance=10):
            return all(abs(c1[i] - c2[i]) <= tolerance for i in range(3))

        # Check if the point is in a closed area before filling
        if not is_point_in_closed_area(arr, pos):

            return

        # Stack-based flood fill algorithm with color tolerance check
        def stack_based_fill(start_x, start_y):
            stack = [(start_x, start_y)]
            while stack:
                x, y = stack.pop()
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                # Ensure the pixel matches the target color with a tolerance
                if color_match(arr[y, x], target_color):
                    # Protect boundary pixels by checking alpha
                    if arr[y, x, 3] == 255:
                        continue  # Skip boundary pixels
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
