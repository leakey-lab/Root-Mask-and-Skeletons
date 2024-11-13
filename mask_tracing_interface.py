from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QButtonGroup,
    QGroupBox,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsView,
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
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QRectF, QEvent, QPointF
import os
import numpy as np
import cv2
from image_normalization_interface import ImageNormalization, NormalizationControls
from PyQt6.QtGui import QMouseEvent


class MaskTracingGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mask_tracing_interface = parent
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.pan_mode = False
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.drawing = False
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0

    def wheelEvent(self, event: QWheelEvent):
        if self.drawing:
            event.ignore()
            return

        # Check if B key is pressed in the parent interface
        if self.mask_tracing_interface.b_key_pressed:
            # Ignore the wheel event here, let the parent handle brush size
            event.ignore()
            return

        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Zoom
            zoom_in = event.angleDelta().y() > 0
            if zoom_in and self.zoom_factor < self.max_zoom:
                factor = 1.25
                self.zoom_factor *= factor
            elif not zoom_in and self.zoom_factor > self.min_zoom:
                factor = 0.8
                self.zoom_factor *= factor
            else:
                event.ignore()
                return

            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.scale(factor, factor)

            # Update the mask interface zoom slider
            if hasattr(self.mask_tracing_interface, "zoom_slider"):
                zoom_percent = int(round(self.zoom_factor * 100))
                self.mask_tracing_interface.zoom_slider.setValue(zoom_percent)
                if hasattr(self.mask_tracing_interface, "zoom_label"):
                    self.mask_tracing_interface.zoom_label.setText(
                        f"Zoom: {zoom_percent}%"
                    )

            event.accept()
        elif (
            not self.mask_tracing_interface.b_key_pressed
        ):  # Only allow scrolling when B is not pressed
            # Allow normal scrolling in pan mode, prevent in draw mode
            if not self.drawing:
                super().wheelEvent(event)

    def scrollContentsBy(self, dx: int, dy: int):
        """Override scroll behavior to prevent scrolling during drawing"""
        if not self.drawing:
            super().scrollContentsBy(dx, dy)

    def mousePressEvent(self, event: QMouseEvent):
        if self.pan_mode:
            # In pan mode, handle only panning
            if event.button() in [
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.MiddleButton,
            ]:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                super().mousePressEvent(event)
        else:
            # In draw mode, handle only drawing
            if (
                event.button() == Qt.MouseButton.LeftButton
                and self.mask_tracing_interface.mask_pixmap
            ):
                self.drawing = True
                self.viewport().setProperty(
                    "cursor", self.mask_tracing_interface.brush_cursor
                )
                self.mask_tracing_interface.drawing = True
                self.mask_tracing_interface.save_for_undo()
                pos = self.map_to_image(event.position())

                if self.mask_tracing_interface.fill_button.isChecked():
                    self.mask_tracing_interface.flood_fill(pos)
                elif self.mask_tracing_interface.brush_button.isChecked():
                    self.mask_tracing_interface.draw_point(pos)

                event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.pan_mode:
            # In pan mode, handle only panning
            if event.button() in [
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.MiddleButton,
            ]:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                super().mouseReleaseEvent(event)
        else:
            # In draw mode, handle only drawing
            if event.button() == Qt.MouseButton.LeftButton:
                self.drawing = False
                self.mask_tracing_interface.drawing = False
                self.mask_tracing_interface.last_point = None
                event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.pan_mode:
            # In pan mode, handle only panning
            super().mouseMoveEvent(event)
        else:
            # In draw mode, handle only drawing
            if (
                event.buttons() & Qt.MouseButton.LeftButton
                and self.mask_tracing_interface.drawing
                and self.mask_tracing_interface.mask_pixmap
            ):
                pos = self.map_to_image(event.position())
                self.mask_tracing_interface.draw_point(pos)
                event.accept()

    def set_mode(self, pan_mode: bool):
        """Switch between pan and draw modes"""
        self.pan_mode = pan_mode
        self.drawing = False
        if pan_mode:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        else:
            self.setCursor(self.mask_tracing_interface.brush_cursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)

    def map_to_image(self, pos: QPointF):
        scene_pos = self.mapToScene(pos.toPoint())
        image_x = int(scene_pos.x())
        image_y = int(scene_pos.y())

        final_x = max(
            0, min(image_x, self.mask_tracing_interface.mask_pixmap.width() - 1)
        )
        final_y = max(
            0, min(image_y, self.mask_tracing_interface.mask_pixmap.height() - 1)
        )

        return QPoint(final_x, final_y)

    def ensureVisible(self, *args, **kwargs):
        """Override to prevent automatic scrolling during drawing"""
        if not self.drawing:
            super().ensureVisible(*args, **kwargs)

    def set_zoom(self, factor):
        """Set zoom level directly (called from zoom slider)"""
        if self.min_zoom <= factor / 100 <= self.max_zoom:
            scale = (factor / 100) / self.zoom_factor
            self.zoom_factor = factor / 100
            self.scale(scale, scale)


class MaskTracingInterface(QWidget):
    mask_saved = pyqtSignal(str)
    mask_cleared = pyqtSignal(str)
    b_key_status_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = ""
        self.mask_directory = ""
        self.zoom_factor = 1.0
        self.undo_stack = []
        self.redo_stack = []
        self.max_stack_size = 25  # Set maximum stack size
        self.b_key_pressed = False
        self.size_slider = None
        self.zoom_slider = None
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # Reduce overall margins

        # Graphics View and Scene for image and mask
        self.graphics_view = MaskTracingGraphicsView(self)  # Use the subclassed view
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.graphics_view.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        self.graphics_view.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.graphics_view.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        main_layout.addWidget(self.graphics_view)

        # Initialize pixmap items for image and mask
        self.image_item = QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self.scene.addItem(self.image_item)

        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setOpacity(1.0)  # Initial opacity
        self.mask_item.setZValue(1)
        self.scene.addItem(self.mask_item)

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
            ("Brush Size", 1, 100, 5),
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

        toggle_button_style = """
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
            QPushButton:checked {
                background-color: #404040;
            }
            QPushButton:hover {
                background-color: #404040;
            }
        """

        self.mode_toggle = QPushButton("🔒 Draw")
        self.mode_toggle.setStyleSheet(toggle_button_style)
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.setChecked(True)  # Start in draw mode
        self.mode_toggle.clicked.connect(self.toggle_mode)
        tools_layout.addWidget(self.mode_toggle)

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

        # Create custom cursors
        self.create_cursors()

        # Set the initial cursor
        self.setCursor(self.brush_cursor)

    def toggle_mode(self):
        if self.mode_toggle.isChecked():
            self.mode_toggle.setText("🔒 Draw")
            self.graphics_view.set_mode(False)  # Draw mode
        else:
            self.mode_toggle.setText("✋ Pan")
            self.graphics_view.set_mode(True)  # Pan mode

    def create_cursors(self):
        """Creates custom cursors for drawing and panning."""
        # Brush Cursor
        self.brush_cursor = self.create_brush_cursor(self.brush_size)

        # Panning Cursor (Open Hand)
        self.panning_cursor = QCursor(Qt.CursorShape.OpenHandCursor)

    def create_brush_cursor(self, size):
        """Creates a custom brush cursor based on the brush size."""
        cursor_size = max(size * 2, 32)  # Ensure the cursor is at least 32x32 pixels
        cursor_pixmap = QPixmap(cursor_size, cursor_size)
        cursor_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(cursor_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

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
        """Initialize a new image for editing"""
        # Clear previous image and mask data
        self.current_image_path = image_path
        self.mask_directory = os.path.join(os.path.dirname(image_path), "mask")

        # Clear undo/redo stacks
        self.undo_stack.clear()
        self.redo_stack.clear()

        # Create a new scene and items
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        # Create new graphics items
        self.image_item = QGraphicsPixmapItem()
        self.mask_item = QGraphicsPixmapItem()

        # Add items to scene immediately
        self.scene.addItem(self.image_item)
        self.scene.addItem(self.mask_item)

        # Set z-order
        self.image_item.setZValue(0)
        self.mask_item.setZValue(1)

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
            self.opacity_slider.setValue(100)

        # Load existing mask if it exists
        mask_path = os.path.join(self.mask_directory, os.path.basename(image_path))
        if os.path.exists(mask_path):
            # Read the mask using OpenCV
            mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_cv is not None:
                # Create RGBA image with transparency
                height, width = mask_cv.shape
                rgba = np.zeros((height, width, 4), dtype=np.uint8)
                # Set white color only where mask is definitely white (255)
                mask = mask_cv == 255
                rgba[mask] = [255, 255, 255, 255]  # White with full opacity
                # Create QImage from numpy array
                mask_qimage = QImage(
                    rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888
                )
                self.mask_pixmap = QPixmap.fromImage(mask_qimage)

        # Update the display
        self.update_display()

    def update_display(self):
        if not self.image_pixmap or not self.mask_pixmap:
            return

        # Only recreate items if they're not in the scene
        if not self.image_item.scene():
            self.scene.addItem(self.image_item)
            self.image_item.setZValue(0)

        if not self.mask_item.scene():
            self.scene.addItem(self.mask_item)
            self.mask_item.setZValue(1)

        # Update the pixmaps directly
        self.image_item.setPixmap(self.image_pixmap)
        self.mask_item.setPixmap(self.mask_pixmap)

        # Update scene rect only if needed
        if self.scene.sceneRect() != QRectF(self.image_pixmap.rect()):
            self.scene.setSceneRect(QRectF(self.image_pixmap.rect()))

    def draw_point(self, pos):
        if not self.mask_pixmap:
            return

        painter = QPainter(self.mask_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Create a temporary pixmap for the current stroke
        temp_pixmap = QPixmap(self.mask_pixmap.size())
        temp_pixmap.fill(Qt.GlobalColor.transparent)
        temp_painter = QPainter(temp_pixmap)
        temp_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        temp_painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

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
        self.brush_cursor = self.create_brush_cursor(self.brush_size)
        if self.mode_toggle.isChecked() or self.b_key_pressed:
            self.graphics_view.viewport().setCursor(self.brush_cursor)

    def update_opacity(self, value):
        # Map the 0-100 range to 0.3-1.0 range to maintain minimum visibility
        self.brush_opacity = 0.3 + (value / 100) * 0.7
        self.opacity_label.setText(
            f"Opacity: {value}%"
        )  # Display the original 0-100 value
        self.mask_item.setOpacity(self.brush_opacity)

    def update_zoom(self, value):
        self.zoom_factor = value / 100
        self.zoom_label.setText(f"Zoom: {value}%")
        self.graphics_view.resetTransform()
        self.graphics_view.scale(self.zoom_factor, self.zoom_factor)

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

            # Ensure the image has an alpha channel
            if mask_image.format() != QImage.Format.Format_ARGB32:
                mask_image = mask_image.convertToFormat(QImage.Format.Format_ARGB32)

            # Save the image with high-quality settings
            mask_image.save(save_path, "PNG", 100)

            print(f"Mask saved to {save_path}")
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
        """Save current state for undo with stack size limit."""
        if self.mask_pixmap:
            # Add current state to undo stack
            self.undo_stack.append(self.mask_pixmap.copy())

            # Remove oldest state if stack exceeds maximum size
            if len(self.undo_stack) > self.max_stack_size:
                self.undo_stack.pop(0)  # Remove oldest item

            # Clear redo stack as new action invalidates redo history
            self.redo_stack.clear()

    def undo(self):
        """Undo last action with stack size limit."""
        if self.undo_stack:
            # Save current state to redo stack
            self.redo_stack.append(self.mask_pixmap.copy())

            # Remove oldest redo state if stack exceeds maximum size
            if len(self.redo_stack) > self.max_stack_size:
                self.redo_stack.pop(0)  # Remove oldest item

            # Restore previous state from undo stack
            self.mask_pixmap = self.undo_stack.pop()
            self.update_display()

    def redo(self):
        """Redo last undone action with stack size limit."""
        if self.redo_stack:
            # Save current state to undo stack
            self.undo_stack.append(self.mask_pixmap.copy())

            # Remove oldest undo state if stack exceeds maximum size
            if len(self.undo_stack) > self.max_stack_size:
                self.undo_stack.pop(0)  # Remove oldest item

            # Restore next state from redo stack
            self.mask_pixmap = self.redo_stack.pop()
            self.update_display()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_B:
            # Toggle the state of b_key_pressed when the B key is pressed
            self.b_key_pressed = not self.b_key_pressed
            # Emit signal when state changes
            self.b_key_status_changed.emit(self.b_key_pressed)
            if self.b_key_pressed:
                print("B KEY PRESSED")
            else:
                print("B KEY UNPRESSED")
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self.b_key_pressed:
            # Adjust brush size when B is pressed
            delta = event.angleDelta().y()
            if delta > 0:
                new_value = min(self.size_slider.value() + 1, 50)
            else:
                new_value = max(self.size_slider.value() - 1, 1)

            # Update the slider value and force immediate cursor update
            self.size_slider.setValue(new_value)
            self.update_brush_size(new_value)  # This will update the cursor immediately
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

    def flood_fill(self, pos):
        """
        Enhanced flood fill that uses the current brush color.
        """
        if not self.mask_pixmap:
            return

        # Convert QPixmap to QImage for pixel manipulation
        image = self.mask_pixmap.toImage()
        width, height = image.width(), image.height()

        # Convert QImage to numpy array
        image = image.convertToFormat(QImage.Format.Format_ARGB32)
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))

        # Get the brush color components
        brush_color = self.brush_color
        r, g, b = brush_color.red(), brush_color.green(), brush_color.blue()

        # Extract the alpha channel
        alpha_channel = arr[:, :, 3].copy()  # Make a copy to preserve original

        # Create a binary mask from alpha channel
        binary_mask = (alpha_channel > 0).astype(np.uint8) * 255

        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Seed point for flood fill
        seed_x, seed_y = pos.x(), pos.y()

        # Create a mask for the local region around the clicked point
        local_mask = np.zeros((height + 2, width + 2), np.uint8)

        # Find which contour contains the seed point (if any)
        target_contour = None
        for contour in contours:
            if cv2.pointPolygonTest(contour, (seed_x, seed_y), False) >= 0:
                target_contour = contour
                break

        # If no contour contains the point, create a new fill region
        if target_contour is None:
            # Create a mask for floodFill
            flood_mask = np.zeros((height + 2, width + 2), np.uint8)

            # Perform flood fill on the alpha channel with connectivity=8
            flags = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
            _, local_mask, _, _ = cv2.floodFill(
                alpha_channel.copy(),  # Use a copy to avoid modifying original
                flood_mask,
                (seed_x, seed_y),
                255,
                loDiff=0,
                upDiff=0,
                flags=flags,
            )
        else:
            # Create a mask from the target contour
            local_mask = np.zeros((height, width), np.uint8)
            cv2.drawContours(local_mask, [target_contour], -1, 255, -1)

        # Fill the region with the brush color
        if self.eraser_button.isChecked():
            # For eraser, set alpha to 0
            arr[local_mask == 255, 3] = 0
        else:
            # For brush, set the color and alpha
            arr[local_mask == 255, 0] = b  # Blue component
            arr[local_mask == 255, 1] = g  # Green component
            arr[local_mask == 255, 2] = r  # Red component
            arr[local_mask == 255, 3] = 255  # Alpha

        # Convert the numpy array back to QImage
        result_image = QImage(
            arr.data, width, height, width * 4, QImage.Format.Format_RGBA8888
        )

        # Save for undo and update the mask pixmap
        self.save_for_undo()
        self.mask_pixmap = QPixmap.fromImage(result_image.copy())
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
