"""
Mask tracing interface for drawing and editing masks.
Provides tools for brush, eraser, and flood fill operations.
"""

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
)
from PyQt6.QtGui import (
    QPixmap,
    QPainter,
    QColor,
    QImage,
    QKeySequence,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QRectF
from collections import deque
import logging
import os
import numpy as np
import cv2

logger = logging.getLogger(__name__)

from .mask_graphics_view import MaskTracingGraphicsView
from .mask_cursor_utils import create_brush_cursor, create_panning_cursor
from .mask_drawing_tools import MaskDrawingMixin
from .image_normalization_interface import ImageNormalization, NormalizationControls
from app.gui.widgets import (
    ToolRail,
    FloatingDock,
    EnhancePopover,
    IconButton,
    load_icon,
    tokens,
)


class MaskTracingInterface(QWidget, MaskDrawingMixin):
    """
    Main widget for mask tracing with brush, eraser, and flood fill tools.
    Inherits drawing functionality from MaskDrawingMixin.
    """
    
    mask_saved = pyqtSignal(str)
    mask_cleared = pyqtSignal(str)
    b_key_status_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_path = ""
        self.mask_directory = ""
        self.zoom_factor = 1.0
        # Bounded undo/redo history. deque(maxlen=...) auto-evicts the oldest
        # entry on append in O(1), replacing the previous O(n) list.pop(0) trim.
        self.max_stack_size = 25
        self.undo_stack = deque(maxlen=self.max_stack_size)
        self.redo_stack = deque(maxlen=self.max_stack_size)
        self.b_key_pressed = False
        self.size_slider = None
        self.zoom_slider = None
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Graphics View and Scene for image and mask
        self.graphics_view = MaskTracingGraphicsView(self)
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.graphics_view.setViewportUpdateMode(
            self.graphics_view.ViewportUpdateMode.FullViewportUpdate
        )
        self.graphics_view.setTransformationAnchor(
            self.graphics_view.ViewportAnchor.AnchorUnderMouse
        )
        self.graphics_view.setResizeAnchor(
            self.graphics_view.ViewportAnchor.AnchorUnderMouse
        )

        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        main_layout.addWidget(self.graphics_view)

        # Initialize pixmap items for image and mask
        self.image_item = QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self.scene.addItem(self.image_item)

        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setOpacity(1.0)
        self.mask_item.setZValue(1)
        self.scene.addItem(self.mask_item)

        # Bottom control panel
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        # Set the main layout
        self.setLayout(main_layout)

        # Build floating overlays (tool rail, dock, enhance popover)
        self._build_overlays()

        # Initialize drawing attributes
        self.last_point = QPoint()
        self.drawing = False
        self.brush_color = QColor(Qt.GlobalColor.white)
        self.brush_size = 5
        self.brush_opacity = 1.0
        self.mask_pixmap = None
        self.image_pixmap = None

        # Create custom cursors
        self.brush_cursor = create_brush_cursor(self.brush_size)
        self.panning_cursor = create_panning_cursor()

        # Set the initial cursor
        self.setCursor(self.brush_cursor)

    def _build_overlays(self):
        """Build the floating in-window overlays (presentation only).

        Creates a left vertical ``ToolRail``, a bottom-centre ``FloatingDock``
        and a top-right ``EnhancePopover``, all parented to this interface
        widget (matching the SPROUTS canvas layout).
        """
        self.tool_rail = ToolRail(self)
        self.dock = FloatingDock(self)
        self.enhance_popover = EnhancePopover(self)

        # Tools -> rail (same button objects, still in tool_button_group).
        self.tool_rail.add_widget(self.brush_button)
        self.tool_rail.add_widget(self.eraser_button)
        self.tool_rail.add_widget(self.fill_button)

        # Mode toggle (Draw/Pan) -> rail, after a separator.
        self.tool_rail.add_separator()
        self.mode_toggle = QPushButton("🔒 Draw")
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.setChecked(True)
        self.mode_toggle.clicked.connect(self.toggle_mode)
        self.tool_rail.add_widget(self.mode_toggle)

        # Sliders + actions -> bottom dock.
        self.size_container.setFixedWidth(150)
        self.opacity_container.setFixedWidth(150)
        self.dock.add_widget(self.size_container)
        self.dock.add_widget(self.opacity_container)
        self.dock.add_separator()
        self.dock.add_widget(self.undo_button)
        self.dock.add_widget(self.redo_button)
        self.dock.add_widget(self.clear_button)
        self.dock.add_separator()
        self.dock.add_widget(self.save_button)

        # Image enhancement controls -> top-right popover, toggled from rail.
        self.enhance_popover.set_content(self.norm_controls)
        self.enhance_button = IconButton(
            "contrast", "Image enhancement", checkable=False
        )
        self.enhance_button.clicked.connect(self.enhance_popover.toggle)
        self.tool_rail.add_separator()
        self.tool_rail.add_widget(self.enhance_button)

    def _create_control_panel(self):
        """Create the bottom control panel with tools and adjustments."""
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
        control_layout.setSpacing(4)
        control_layout.setContentsMargins(2, 1, 2, 1)

        # Tools Group
        tools_group, tools_layout = self._create_tools_group()
        control_layout.addWidget(tools_group)

        # Actions Group
        actions_group = self._create_actions_group()
        control_layout.addWidget(actions_group)

        # Adjustments Group
        adjustments_group = self._create_adjustments_group()
        control_layout.addWidget(adjustments_group)

        # Image Enhancement Controls
        self.norm_controls = NormalizationControls(self)
        control_layout.addWidget(self.norm_controls)

        # Connect signals
        self._connect_signals()

        return control_panel

    def _create_tools_group(self):
        """Create the tools group with brush, eraser, and fill buttons."""
        tools_group = QGroupBox("Tools")
        tools_group.setFixedWidth(100)
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(2)
        tools_layout.setContentsMargins(4, 2, 4, 2)

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
        
        return tools_group, tools_layout

    def _create_actions_group(self):
        """Create the actions group with clear, save, undo, redo buttons."""
        actions_group = QGroupBox("Actions")
        actions_group.setFixedWidth(100)
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(2)
        actions_layout.setContentsMargins(4, 2, 4, 2)

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
        return actions_group

    def _create_adjustments_group(self):
        """Create the adjustments group with sliders."""
        adjustments_group = QGroupBox("Adjustments")
        adjustments_layout = QVBoxLayout()
        adjustments_layout.setSpacing(2)
        adjustments_layout.setContentsMargins(8, 2, 8, 2)

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

        sliders_data = [
            ("Brush Size", 1, 100, 5),
            ("Opacity", 0, 100, 100),
            ("Zoom", 10, 200, 100),
        ]

        for name, min_val, max_val, default in sliders_data:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(1)
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
                self.size_container = container
            elif name == "Opacity":
                self.opacity_slider = slider
                self.opacity_label = label
                self.opacity_container = container
            else:
                self.zoom_slider = slider
                self.zoom_label = label
                self.zoom_container = container

        adjustments_group.setLayout(adjustments_layout)
        return adjustments_group

    def _connect_signals(self):
        """Connect all signals to their handlers."""
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

    def toggle_mode(self):
        """Toggle between draw and pan modes."""
        if self.mode_toggle.isChecked():
            self.mode_toggle.setText("🔒 Draw")
            self.graphics_view.set_mode(False)
        else:
            self.mode_toggle.setText("✋ Pan")
            self.graphics_view.set_mode(True)

    def find_mask_path(self, image_path):
        """Find the corresponding mask file, with enhanced debugging."""
        base_name = os.path.splitext(os.path.basename(os.path.normpath(image_path)))[0]

        extensions = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]

        for ext in extensions:
            mask_path = os.path.normpath(
                os.path.join(self.mask_directory, base_name + ext)
            )
            if os.path.exists(mask_path):
                return mask_path
        return None

    def load_image(self, image_path):
        """Initialize a new image for editing with enhanced mask loading."""
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
        mask_path = self.find_mask_path(image_path)
        if mask_path:
            try:
                mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_cv is not None:
                    height, width = mask_cv.shape
                    rgba = np.zeros((height, width, 4), dtype=np.uint8)

                    mask = mask_cv > 200
                    rgba[mask] = [255, 255, 255, 255]

                    # QImage wraps the rgba buffer without copying; the very
                    # next line (QPixmap.fromImage) deep-copies the pixels, so
                    # rgba stays alive for the QImage's entire lifetime (cf.
                    # F-018).
                    mask_qimage = QImage(
                        rgba.data,
                        width,
                        height,
                        width * 4,
                        QImage.Format.Format_RGBA8888,
                    )
                    self.mask_pixmap = QPixmap.fromImage(mask_qimage)
            except (cv2.error, ValueError) as e:
                logger.warning("Failed to load existing mask %s: %s", mask_path, e)

        # Update the display
        self.update_display()

    def update_display(self):
        """Update the graphics scene with current image and mask."""
        if not self.image_pixmap or not self.mask_pixmap:
            return

        # Only recreate items if they're not in the scene
        if not self.image_item.scene():
            self.scene.addItem(self.image_item)
            self.image_item.setZValue(0)

        if not self.mask_item.scene():
            self.scene.addItem(self.mask_item)
            self.mask_item.setZValue(1)

        # Configure high quality settings for items
        self.image_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_item.setCacheMode(
            QGraphicsPixmapItem.CacheMode.DeviceCoordinateCache
        )

        self.mask_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.mask_item.setCacheMode(QGraphicsPixmapItem.CacheMode.DeviceCoordinateCache)

        # Update the pixmaps directly
        self.image_item.setPixmap(self.image_pixmap)
        self.mask_item.setPixmap(self.mask_pixmap)

        # Update scene rect only if needed
        if self.scene.sceneRect() != QRectF(self.image_pixmap.rect()):
            self.scene.setSceneRect(QRectF(self.image_pixmap.rect()))

    def update_brush_size(self, value):
        """Update brush size and cursor."""
        self.brush_size = value
        self.size_label.setText(f"Brush Size: {value}")
        self.brush_cursor = create_brush_cursor(self.brush_size)
        if self.mode_toggle.isChecked() or self.b_key_pressed:
            self.graphics_view.viewport().setCursor(self.brush_cursor)

    def update_opacity(self, value):
        """Update mask opacity."""
        self.brush_opacity = 0.3 + (value / 100) * 0.7
        self.opacity_label.setText(f"Opacity: {value}%")
        self.mask_item.setOpacity(self.brush_opacity)

    def update_zoom(self, value):
        """Update zoom level."""
        self.zoom_factor = value / 100
        self.zoom_label.setText(f"Zoom: {value}%")
        self.graphics_view.resetTransform()
        self.graphics_view.scale(self.zoom_factor, self.zoom_factor)

    def save_mask(self):
        """Save the current mask to disk."""
        if self.mask_pixmap and self.current_image_path:
            os.makedirs(self.mask_directory, exist_ok=True)
            save_path = os.path.normpath(
                os.path.join(
                    self.mask_directory,
                    os.path.splitext(os.path.basename(self.current_image_path))[0]
                    + ".png",
                )
            )

            mask_image = self.mask_pixmap.toImage()

            if mask_image.format() != QImage.Format.Format_ARGB32:
                mask_image = mask_image.convertToFormat(QImage.Format.Format_ARGB32)

            mask_image.save(save_path, "PNG", 100)

            self.mask_saved.emit(self.current_image_path)

    def clear_mask(self):
        """Clear the current mask."""
        if self.mask_pixmap:
            self.save_for_undo()
            self.mask_pixmap.fill(Qt.GlobalColor.transparent)
            self.update_display()

            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]

            extensions = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]
            for ext in extensions:
                mask_path = os.path.join(self.mask_directory, base_name + ext)
                if os.path.exists(mask_path):
                    os.remove(mask_path)
                    break

            self.mask_cleared.emit(self.current_image_path)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_B:
            self.b_key_pressed = not self.b_key_pressed
            self.b_key_status_changed.emit(self.b_key_pressed)
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Handle wheel events for brush size adjustment."""
        if self.b_key_pressed:
            delta = event.angleDelta().y()
            if delta > 0:
                new_value = min(self.size_slider.value() + 1, 50)
            else:
                new_value = max(self.size_slider.value() - 1, 1)

            self.size_slider.setValue(new_value)
            self.update_brush_size(new_value)
            event.accept()
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_slider.setValue(min(self.zoom_slider.value() + 10, 200))
            else:
                self.zoom_slider.setValue(max(self.zoom_slider.value() - 10, 10))
            event.accept()
        else:
            event.ignore()
            super().wheelEvent(event)

    def apply_normalization(self):
        """Apply image normalization/enhancement."""
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
