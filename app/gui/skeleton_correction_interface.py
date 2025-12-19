"""
Skeleton correction interface (polyline + eraser) kept isolated from the app's
existing skeleton generation and display streams.

Key properties:
- Users explicitly load a skeleton raster via file dialog.
- Editing happens on an internal 1px skeleton mask (raster), with polyline draw
  and eraser/cut tools.
- Corrected skeletons are saved to `<images_folder>/skeleton_corrections/<base>_skeleton.png`.
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QPixmap, QKeySequence, QShortcut, QPainterPath
from PyQt6.QtGui import QPen, QBrush
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QButtonGroup,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsLineItem,
)

from .skeleton_correction_graphics_view import SkeletonCorrectionGraphicsView
from .skeleton_graph_model import SkeletonCorrectionModel
from .image_normalization_interface import ImageNormalization, NormalizationControls
from .mask_cursor_utils import create_brush_cursor


class SkeletonCorrectionInterface(QWidget):
    """Main widget for skeleton correction editing."""

    skeleton_saved = pyqtSignal(str)

    TOOL_SELECT = "select"
    TOOL_ERASER = "eraser"
    TOOL_POLYLINE = "polyline"
    TOOL_CONNECT = "connect"

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_image_path: str = ""
        self.current_image_name: str = ""
        self.images_base_folder: Optional[str] = None

        self._save_size: Optional[Tuple[int, int]] = None  # (w,h) for _skeleton.png
        self._loaded_skeleton_path: Optional[str] = None
        self._original_image_pixmap: Optional[QPixmap] = None

        self.model = SkeletonCorrectionModel()

        # UI state
        self.current_tool = self.TOOL_SELECT
        self.pan_mode = False

        # Polyline drawing state
        self._polyline_points: List[QPoint] = []
        self._polyline_preview_item: Optional[QGraphicsPathItem] = None
        self._polyline_handle_items: List[QGraphicsEllipseItem] = []
        self._polyline_dragging: bool = False
        self._polyline_drag_index: Optional[int] = None  # dragging a control point
        self._polyline_translate_dragging: bool = False  # shift-drag to move the whole line
        self._polyline_translate_anchor: Optional[QPoint] = None
        self._polyline_translate_orig_points: List[QPoint] = []
        self._polyline_smooth: bool = False
        # When "Select" is used to pick an existing skeleton polyline to edit,
        # we store the original polyline points here so we can erase it on commit.
        self._edit_original_polyline: Optional[List[Tuple[int, int]]] = None

        # Connect tool state (drag-from-endpoint line drawing)
        self._connect_first_endpoint: Optional[QPoint] = None
        self._connect_dragging: bool = False  # True while dragging a line from endpoint
        self._connect_line_preview_item: Optional[QGraphicsLineItem] = None

        # Select tool state (for precise Delete-key cuts)
        self._selected_point: Optional[QPoint] = None
        self._selection_item: Optional[QGraphicsEllipseItem] = None

        # Endpoint markers
        self._endpoint_items: List[QGraphicsEllipseItem] = []

        # Controls
        # NOTE: For UX consistency with the mask editor, this slider value is treated as
        # the *diameter* (in image pixels) of the eraser footprint shown by the red cursor ring.
        # The actual erase operation uses radius = diameter / 2.
        self.eraser_radius = 20
        self.draw_thickness = 3
        self.overlay_opacity = 0.85
        
        # Cursor state
        # Zoom-aware cursor: cursor pixmap is in screen pixels, while eraser works in image pixels.
        self._zoom_factor: float = 1.0
        self.brush_cursor = create_brush_cursor(self._cursor_diameter_screen_px())

        # Eraser stroke tracking
        self._eraser_active = False

        # Performance: throttling display updates during eraser strokes
        self._last_display_update_time: float = 0.0
        self._display_update_interval: float = 0.030  # 30ms throttle
        self._endpoints_hidden: bool = False

        self._build_ui()

        # Shortcuts (match common expectations)
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.undo)
        self.redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.activated.connect(self.redo)

    # -------------------- UI --------------------
    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.graphics_view = SkeletonCorrectionGraphicsView(self)
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        main_layout.addWidget(self.graphics_view)

        # Scene items
        self.image_item = QGraphicsPixmapItem()
        self.image_item.setZValue(0)
        self.scene.addItem(self.image_item)

        self.skeleton_item = QGraphicsPixmapItem()
        self.skeleton_item.setZValue(1)
        self.skeleton_item.setOpacity(self.overlay_opacity)
        self.scene.addItem(self.skeleton_item)

        # Bottom controls
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        self.setLayout(main_layout)

    def _create_control_panel(self) -> QWidget:
        control_panel = QWidget()
        control_panel.setStyleSheet(
            """
            QWidget { background-color: #1e1e1e; }
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

        layout = QHBoxLayout(control_panel)
        layout.setSpacing(4)
        layout.setContentsMargins(2, 1, 2, 1)

        tools_group = QGroupBox("Tools")
        tools_group.setFixedWidth(140)
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(2)
        tools_layout.setContentsMargins(4, 2, 4, 2)

        btn_style = """
            QPushButton {
                background-color: #2d2d2d;
                border: none;
                border-radius: 4px;
                padding: 4px;
                color: white;
                min-height: 28px;
                font-size: 12px;
            }
            QPushButton:checked { background-color: #404040; }
            QPushButton:hover { background-color: #404040; }
        """

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)

        self.select_button = QPushButton("🖱️ Select")
        self.eraser_button = QPushButton("🧽 Eraser")
        self.polyline_button = QPushButton("📏 Polyline")
        self.connect_button = QPushButton("🔗 Connect")

        for b in [
            self.select_button,
            self.eraser_button,
            self.polyline_button,
            self.connect_button,
        ]:
            b.setStyleSheet(btn_style)
            b.setCheckable(True)
            self.tool_group.addButton(b)
            tools_layout.addWidget(b)

        self.select_button.setChecked(True)

        # Add mode toggle button to tools group
        self.mode_toggle = QPushButton("🔒 Draw")
        self.mode_toggle.setStyleSheet(btn_style)
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.setChecked(True)
        tools_layout.addWidget(self.mode_toggle)

        # Polyline smoothing toggle (for curvable/bendable lines)
        self.smooth_polyline_toggle = QPushButton("〰 Smooth")
        self.smooth_polyline_toggle.setStyleSheet(btn_style)
        self.smooth_polyline_toggle.setCheckable(True)
        self.smooth_polyline_toggle.setChecked(False)
        self.smooth_polyline_toggle.setEnabled(False)  # enabled only in Polyline tool
        tools_layout.addWidget(self.smooth_polyline_toggle)

        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)

        actions_group = QGroupBox("Actions")
        actions_group.setFixedWidth(140)
        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(2)
        actions_layout.setContentsMargins(4, 2, 4, 2)

        self.load_skeleton_button = QPushButton("📎 Load Skeleton…")
        self.save_skeleton_button = QPushButton("💾 Save Skeleton")
        self.undo_button = QPushButton("⬅️ Undo (Ctrl+Z)")
        self.redo_button = QPushButton("➡️ Redo (Ctrl+Y)")
        self.clear_button = QPushButton("🗑️ Clear All")
        
        # Polyline control buttons with distinctive styling
        polyline_btn_style = """
            QPushButton {
                background-color: #50fa7b;
                border: none;
                border-radius: 4px;
                padding: 4px;
                color: #282a36;
                min-height: 28px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:disabled { background-color: #44475a; color: #6272a4; }
            QPushButton:hover:enabled { background-color: #69ff94; }
        """
        cancel_btn_style = """
            QPushButton {
                background-color: #ff5555;
                border: none;
                border-radius: 4px;
                padding: 4px;
                color: white;
                min-height: 28px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:disabled { background-color: #44475a; color: #6272a4; }
            QPushButton:hover:enabled { background-color: #ff6e6e; }
        """
        self.finish_polyline_button = QPushButton("✓ Finish Line (Enter)")
        self.cancel_polyline_button = QPushButton("✗ Cancel Line (Esc)")

        for b in [
            self.load_skeleton_button,
            self.save_skeleton_button,
            self.undo_button,
            self.redo_button,
            self.clear_button,
        ]:
            b.setStyleSheet(btn_style)
            actions_layout.addWidget(b)
        
        # Add polyline buttons with special styling
        self.finish_polyline_button.setStyleSheet(polyline_btn_style)
        self.cancel_polyline_button.setStyleSheet(cancel_btn_style)
        actions_layout.addWidget(self.finish_polyline_button)
        actions_layout.addWidget(self.cancel_polyline_button)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        adj_group = QGroupBox("Adjustments")
        adj_layout = QVBoxLayout()
        adj_layout.setSpacing(2)
        adj_layout.setContentsMargins(8, 2, 8, 2)

        slider_style = """
            QSlider { max-height: 20px; }
            QSlider::groove:horizontal {
                border: none; height: 4px; background: #404040;
                margin: 0px; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ff79c6; border: none; width: 16px;
                margin: -6px 0; border-radius: 8px;
            }
            QSlider::handle:horizontal:hover { background: #ff92d0; }
            QLabel { color: white; font-size: 11px; max-height: 15px; }
        """

        self.eraser_label = QLabel(f"Eraser: {self.eraser_radius}px")
        self.eraser_slider = QSlider(Qt.Orientation.Horizontal)
        self.eraser_slider.setStyleSheet(slider_style)
        self.eraser_slider.setRange(1, 80)
        self.eraser_slider.setValue(self.eraser_radius)

        self.opacity_label = QLabel(f"Overlay: {int(self.overlay_opacity * 100)}%")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setStyleSheet(slider_style)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(int(self.overlay_opacity * 100))

        adj_layout.addWidget(self.eraser_label)
        adj_layout.addWidget(self.eraser_slider)
        adj_layout.addWidget(self.opacity_label)
        adj_layout.addWidget(self.opacity_slider)

        adj_group.setLayout(adj_layout)
        layout.addWidget(adj_group)

        # Signals
        self.tool_group.buttonClicked.connect(self._on_tool_changed)
        self.load_skeleton_button.clicked.connect(self.load_skeleton_via_dialog)
        self.save_skeleton_button.clicked.connect(self.save_skeleton)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.clear_button.clicked.connect(self.clear_skeleton)
        self.finish_polyline_button.clicked.connect(self.finish_polyline)
        self.cancel_polyline_button.clicked.connect(self.cancel_polyline)
        self.eraser_slider.valueChanged.connect(self._on_eraser_radius_changed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.mode_toggle.clicked.connect(self._on_mode_toggle)
        self.smooth_polyline_toggle.toggled.connect(self._on_smooth_polyline_toggled)

        # Image enhancement controls (same as mask tracing)
        self.norm_controls = NormalizationControls(self)
        self.norm_controls.apply_button.clicked.connect(self.apply_normalization)
        layout.addWidget(self.norm_controls)

        # Status/hint label
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        status_layout.setSpacing(2)
        status_layout.setContentsMargins(4, 2, 4, 2)
        self.status_label = QLabel("Select an image, then load a skeleton to edit.")
        self.status_label.setStyleSheet("color: #8be9fd; font-size: 11px;")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        return control_panel

    # -------------------- wiring --------------------
    def clamp_to_image(self, pt: QPoint) -> QPoint:
        if not self.image_item.pixmap() or self.image_item.pixmap().isNull():
            return QPoint(0, 0)
        w = self.image_item.pixmap().width()
        h = self.image_item.pixmap().height()
        return QPoint(max(0, min(pt.x(), w - 1)), max(0, min(pt.y(), h - 1)))

    def on_zoom_changed(self, _zoom_factor: float) -> None:
        # Keep the eraser cursor footprint accurate while zooming.
        try:
            self._zoom_factor = float(_zoom_factor)
        except Exception:
            self._zoom_factor = 1.0
        self._rebuild_brush_cursor()

    def _cursor_diameter_screen_px(self) -> int:
        """Red-ring diameter in *screen pixels* (accounts for zoom)."""
        z = max(0.01, float(getattr(self, "_zoom_factor", 1.0)))
        # Slider is diameter in image pixels; convert to screen pixels with zoom.
        return max(1, int(round(float(self.eraser_radius) * z)))

    def _rebuild_brush_cursor(self) -> None:
        """Rebuild cursor so red ring matches actual eraser footprint on screen."""
        self.brush_cursor = create_brush_cursor(self._cursor_diameter_screen_px())
        if (
            hasattr(self, "graphics_view")
            and self.current_tool == self.TOOL_ERASER
            and not self.pan_mode
        ):
            self.graphics_view.update_cursor()

    # -------------------- public API --------------------
    def load_image(self, image_path: str, images_base_folder: Optional[str] = None) -> None:
        """Load the base image and (if present) the last saved correction skeleton."""
        self.current_image_path = image_path
        self.current_image_name = os.path.splitext(os.path.basename(image_path))[0]
        self.images_base_folder = images_base_folder or os.path.dirname(image_path)

        pix = QPixmap(image_path)
        self._original_image_pixmap = pix.copy()
        self.image_item.setPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))

        # Reset editor state
        self._polyline_points.clear()
        self._connect_first_endpoint = None
        self._set_selected_point(None)
        self.model.set_empty((pix.width(), pix.height()))
        self._save_size = None
        self._loaded_skeleton_path = None

        # Try auto-load last correction for convenience (still isolated stream)
        correction_path = self._default_correction_path()
        if correction_path and os.path.exists(correction_path):
            self._load_skeleton_from_path(correction_path)
        else:
            self._update_skeleton_display()
        
        self._update_status_label()

    # -------------------- tool selection --------------------
    def _on_tool_changed(self, button: QPushButton) -> None:
        if button == self.select_button:
            self.current_tool = self.TOOL_SELECT
        elif button == self.eraser_button:
            self.current_tool = self.TOOL_ERASER
        elif button == self.polyline_button:
            self.current_tool = self.TOOL_POLYLINE
        elif button == self.connect_button:
            self.current_tool = self.TOOL_CONNECT

        # Update cursor in view
        if hasattr(self, 'graphics_view'):
            self.graphics_view.update_cursor()

        # Reset transient state when switching tools
        self._polyline_points.clear()
        self._polyline_dragging = False
        self._polyline_drag_index = None
        self._polyline_translate_dragging = False
        self._polyline_translate_anchor = None
        self._polyline_translate_orig_points = []
        self._connect_first_endpoint = None
        self._connect_dragging = False
        self._set_selected_point(None)
        self._clear_polyline_preview()
        self._clear_polyline_handles()
        self._clear_connect_line_preview()
        self._clear_endpoint_highlights()
        self._eraser_active = False

        self._update_polyline_buttons_enabled()
        self._update_status_label()

    def _on_mode_toggle(self) -> None:
        if self.mode_toggle.isChecked():
            self.mode_toggle.setText("🔒 Draw")
            self.pan_mode = False
            self.graphics_view.set_pan_mode(False)
        else:
            self.mode_toggle.setText("✋ Pan")
            self.pan_mode = True
            self.graphics_view.set_pan_mode(True)
        
        # Ensure cursor is correct after mode toggle
        self.graphics_view.update_cursor()

    def _update_polyline_buttons_enabled(self) -> None:
        in_polyline = self.current_tool == self.TOOL_POLYLINE
        has_pts = len(self._polyline_points) >= 2
        has_any_pts = len(self._polyline_points) > 0
        self.finish_polyline_button.setEnabled(in_polyline and has_pts)
        self.cancel_polyline_button.setEnabled(in_polyline and has_any_pts)
        self.smooth_polyline_toggle.setEnabled(in_polyline)
        
        # Update status label with helpful context
        self._update_status_label()
    
    def _update_status_label(self) -> None:
        """Update the status label with context-sensitive hints."""
        if self.model.mask is None:
            self.status_label.setText("Select an image, then load a skeleton to edit.")
            return
        
        tool = self.current_tool
        if tool == self.TOOL_ERASER:
            self.status_label.setText("🧽 ERASER: Click+drag to erase skeleton pixels. Release to finish stroke.")
        elif tool == self.TOOL_POLYLINE:
            n = len(self._polyline_points)
            smooth_hint = ""
            if self._polyline_smooth and n < 3:
                smooth_hint = " (Smooth needs 3+ points: Ctrl+click a segment to add a bend point)"
            if n == 0:
                self.status_label.setText("📏 POLYLINE: Click to place points. Drag points to adjust. Ctrl+click a segment to add a bend. Shift+drag a segment to move the whole line. Enter/double-click to finish.")
            elif n == 1:
                self.status_label.setText(f"📏 POLYLINE: {n} point. Click to add more. Drag points to adjust. Ctrl+click segment to add a bend. Enter/double-click to finish, Esc to cancel.{smooth_hint}")
            else:
                self.status_label.setText(f"📏 POLYLINE: {n} points. Drag points to adjust. Ctrl+click segment to add a bend. Shift+drag segment to move line. Enter/double-click/Finish to commit, Esc/Cancel to discard.{smooth_hint}")
        elif tool == self.TOOL_CONNECT:
            if self._connect_first_endpoint is None:
                self.status_label.setText("🔗 CONNECT: Click near an endpoint (orange dot) to select first point.")
            else:
                self.status_label.setText("🔗 CONNECT: Click near another endpoint to draw a connecting line.")
        elif tool == self.TOOL_SELECT:
            self.status_label.setText("🖱️ SELECT: Click near an existing skeleton line to load it for editing (then drag points / Smooth / Enter to commit).")
        else:
            self.status_label.setText("Ready to edit skeleton.")

    def _on_eraser_radius_changed(self, v: int) -> None:
        self.eraser_radius = int(v)
        self.eraser_label.setText(f"Eraser: {v}px")
        self._rebuild_brush_cursor()

    def _eraser_effective_radius(self) -> int:
        """Effective eraser radius in image pixels (matches red cursor ring)."""
        return max(1, int(round(float(self.eraser_radius) / 2.0)))

    def _on_smooth_polyline_toggled(self, checked: bool) -> None:
        self._polyline_smooth = bool(checked)
        # If we already have points, update the preview to reflect smoothing
        if self.current_tool == self.TOOL_POLYLINE and self._polyline_points:
            self._draw_polyline_preview(self._polyline_points[-1])
            self._update_status_label()

    def _on_opacity_changed(self, v: int) -> None:
        self.overlay_opacity = float(v) / 100.0
        self.opacity_label.setText(f"Overlay: {v}%")
        self.skeleton_item.setOpacity(self.overlay_opacity)

    # -------------------- skeleton IO --------------------
    def _default_skeletons_dir(self) -> Optional[str]:
        if not self.images_base_folder:
            return None
        return os.path.join(self.images_base_folder, "skeletons")

    def _default_corrections_dir(self) -> Optional[str]:
        if not self.images_base_folder:
            return None
        return os.path.join(self.images_base_folder, "skeleton_corrections")

    def _default_correction_path(self) -> Optional[str]:
        if not self.images_base_folder or not self.current_image_name:
            return None
        d = self._default_corrections_dir()
        return os.path.join(d, f"{self.current_image_name}_skeleton.png")

    def load_skeleton_via_dialog(self) -> None:
        start_dir = self._default_skeletons_dir() or (self.images_base_folder or "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Skeleton File", start_dir, "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        self._load_skeleton_from_path(path)

    def _load_skeleton_from_path(self, skeleton_path: str) -> None:
        if not self.image_item.pixmap() or self.image_item.pixmap().isNull():
            QMessageBox.warning(self, "Warning", "Load an image first.")
            return

        img_gray = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            QMessageBox.critical(self, "Error", f"Failed to load skeleton: {skeleton_path}")
            return

        # Track original size as preferred save size (plan requirement)
        self._save_size = (int(img_gray.shape[1]), int(img_gray.shape[0]))
        self._loaded_skeleton_path = skeleton_path

        target_size = (self.image_item.pixmap().width(), self.image_item.pixmap().height())
        self.model.load_from_raster(img_gray, target_size=target_size)
        self._update_skeleton_display()

    def _update_skeleton_display(self, force_endpoints: bool = True) -> None:
        """Render current mask to overlay pixmap and optionally refresh endpoint markers.
        Optimized to use QImage.Format_Indexed8 to avoid expensive RGBA allocations.
        """
        if self.model.mask is None:
            self.skeleton_item.setPixmap(QPixmap())
            return

        mask = self.model.mask
        h, w = mask.shape
        
        # Ensure mask is contiguous for QImage
        if not mask.flags['C_CONTIGUOUS']:
            mask = np.ascontiguousarray(mask)

        # Create QImage pointing directly to the numpy buffer
        # Format_Indexed8: 1 byte per pixel
        qimg = QImage(mask.data, w, h, w, QImage.Format.Format_Indexed8)
        
        # Define color table
        # Index 0 = Transparent
        # Index 255 = Neon Green (R=57, G=255, B=20)
        color_table = [0] * 256
        color_table[255] = QColor(57, 255, 20).rgba()
        qimg.setColorTable(color_table)

        # Convert to QPixmap (converts to display format efficiently in C++)
        pix = QPixmap.fromImage(qimg)
        self.skeleton_item.setPixmap(pix)
        self.skeleton_item.setOpacity(self.overlay_opacity)

        if force_endpoints:
            self._show_endpoints()
            self._refresh_endpoints()
        self._update_polyline_buttons_enabled()

    def _update_skeleton_display_throttled(self) -> None:
        """Throttled version of display update for use during continuous drawing (e.g., eraser).
        
        Updates at most every _display_update_interval seconds.
        Hides endpoints during continuous strokes for better performance.
        """
        now = time.time()
        if now - self._last_display_update_time < self._display_update_interval:
            return
        self._last_display_update_time = now

        # Hide endpoints during continuous eraser stroke for performance
        if not self._endpoints_hidden:
            self._hide_endpoints()

        self._update_skeleton_display(force_endpoints=False)

    def _hide_endpoints(self) -> None:
        """Hide endpoint markers temporarily for performance during strokes."""
        for it in self._endpoint_items:
            it.setVisible(False)
        self._endpoints_hidden = True

    def _show_endpoints(self) -> None:
        """Show endpoint markers again."""
        for it in self._endpoint_items:
            it.setVisible(True)
        self._endpoints_hidden = False

    # -------------------- endpoints / connect --------------------
    def _clear_endpoint_items(self) -> None:
        for it in self._endpoint_items:
            self.scene.removeItem(it)
        self._endpoint_items.clear()

    def _set_selected_point(self, pt: Optional[QPoint]) -> None:
        """Set/clear the current selection marker (select tool)."""
        if self._selection_item is not None:
            self.scene.removeItem(self._selection_item)
            self._selection_item = None
        self._selected_point = pt
        if pt is None:
            return

        r = 5
        item = QGraphicsEllipseItem(pt.x() - r, pt.y() - r, r * 2, r * 2)
        item.setPen(QPen(QColor("#8be9fd")))
        item.setBrush(QBrush(QColor(0, 0, 0, 0)))
        item.setZValue(4)
        self.scene.addItem(item)
        self._selection_item = item

    def _clear_endpoint_highlights(self) -> None:
        for it in self._endpoint_items:
            it.setBrush(QBrush(QColor("#ffb86c")))  # orange

    def _refresh_endpoints(self) -> None:
        self._clear_endpoint_items()
        topo = self.model.topology(simplify_epsilon=1.5)
        for (x, y) in topo.endpoints:
            r = 4
            item = QGraphicsEllipseItem(x - r, y - r, r * 2, r * 2)
            item.setPen(QPen(QColor("#282a36")))
            item.setBrush(QBrush(QColor("#ffb86c")))
            item.setZValue(3)
            self.scene.addItem(item)
            self._endpoint_items.append(item)

    def _nearest_endpoint(self, pos: QPoint, max_dist: int = 10) -> Optional[QPoint]:
        topo = self.model.topology(simplify_epsilon=1.5)
        if not topo.endpoints:
            return None
        best = None
        best_d2 = None
        for (x, y) in topo.endpoints:
            dx = x - pos.x()
            dy = y - pos.y()
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = QPoint(x, y)
        if best is None or best_d2 is None:
            return None
        if best_d2 <= (max_dist * max_dist):
            return best
        return None

    def _highlight_endpoint(self, pt: QPoint) -> None:
        # naive: highlight nearest ellipse
        for it in self._endpoint_items:
            c = it.rect().center()
            if abs(c.x() - pt.x()) <= 1 and abs(c.y() - pt.y()) <= 1:
                it.setBrush(QBrush(QColor("#50fa7b")))

    # -------------------- actions --------------------
    def clear_skeleton(self) -> None:
        if self.model.mask is None:
            return
        self.model.push_undo()
        self.model.mask.fill(0)
        self._polyline_points.clear()
        self._connect_first_endpoint = None
        self._connect_dragging = False
        self._set_selected_point(None)
        self._clear_polyline_preview()
        self._clear_polyline_handles()
        self._clear_connect_line_preview()
        self._update_skeleton_display()

    def undo(self) -> None:
        # If user is mid-polyline, undo removes last point (UI expectation)
        if self.current_tool == self.TOOL_POLYLINE and self._polyline_points:
            self._polyline_points.pop()
            if self._polyline_points:
                self._refresh_polyline_handles()
                self._draw_polyline_preview(self._polyline_points[-1])
            else:
                self._clear_polyline_preview()
                self._clear_polyline_handles()
            self._update_polyline_buttons_enabled()
            self._update_status_label()
            return

        if self.model.undo():
            self._polyline_points.clear()
            self._connect_first_endpoint = None
            self._connect_dragging = False
            self._set_selected_point(None)
            self._clear_polyline_preview()
            self._clear_polyline_handles()
            self._clear_connect_line_preview()
            self._update_skeleton_display()
            self._update_status_label()

    def redo(self) -> None:
        if self.model.redo():
            self._polyline_points.clear()
            self._connect_first_endpoint = None
            self._connect_dragging = False
            self._set_selected_point(None)
            self._clear_polyline_preview()
            self._clear_polyline_handles()
            self._clear_connect_line_preview()
            self._update_skeleton_display()
            self._update_status_label()

    def save_skeleton(self) -> None:
        if self.model.mask is None or not self.images_base_folder or not self.current_image_name:
            return

        out_dir = self._default_corrections_dir()
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.current_image_name}_skeleton.png")

        # Determine save size: prefer loaded skeleton raster size, else default to ML-ish size.
        save_size = self._save_size or (341, 256)
        rendered = self.model.render_to_size(save_size)

        # Save white-on-black
        cv2.imwrite(out_path, rendered)
        self.skeleton_saved.emit(out_path)

    # -------------------- event delegation from view --------------------
    def on_tool_mouse_press(self, pos: QPoint, _event) -> None:
        if self.model.mask is None:
            return

        if self.current_tool == self.TOOL_ERASER:
            # Push undo state BEFORE any modifications
            self.model.push_undo()
            self._eraser_active = True
            self.model.erase_circle((pos.x(), pos.y()), self._eraser_effective_radius())
            self._update_skeleton_display()
            return

        if self.current_tool == self.TOOL_POLYLINE:
            # Polyline supports editing:
            # - Drag a point to move it
            # - Ctrl+click a segment to insert a new point (bend)
            # - Shift+drag a segment to translate the whole polyline
            event = _event
            modifiers = event.modifiers() if event is not None else Qt.KeyboardModifier.NoModifier

            # 1) If user clicked near an existing control point -> start dragging it
            hit_idx = self._nearest_polyline_handle_index(pos, max_dist=8)
            if hit_idx is not None:
                self._polyline_dragging = True
                self._polyline_drag_index = hit_idx
                self._polyline_translate_dragging = False
                self._polyline_translate_anchor = None
                self._polyline_translate_orig_points = []
                return

            # 2) Ctrl+click near a segment -> insert new bend point
            if modifiers & Qt.KeyboardModifier.ControlModifier and len(self._polyline_points) >= 2:
                seg_idx = self._nearest_polyline_segment_index(pos, max_dist=8)
                if seg_idx is not None:
                    self._polyline_points.insert(seg_idx + 1, pos)
                    self._refresh_polyline_handles()
                    self._draw_polyline_preview(pos)
                    self._update_polyline_buttons_enabled()
                    self._update_status_label()
                    return

            # 3) Shift+click near a segment -> start translating whole polyline
            if modifiers & Qt.KeyboardModifier.ShiftModifier and len(self._polyline_points) >= 2:
                seg_idx = self._nearest_polyline_segment_index(pos, max_dist=10)
                if seg_idx is not None:
                    self._polyline_translate_dragging = True
                    self._polyline_translate_anchor = pos
                    self._polyline_translate_orig_points = [QPoint(p) for p in self._polyline_points]
                    self._polyline_dragging = False
                    self._polyline_drag_index = None
                    return

            # 4) Default behavior: add a new point
            self._polyline_points.append(pos)
            self._refresh_polyline_handles()
            self._draw_polyline_preview(pos)
            self._update_polyline_buttons_enabled()
            self._update_status_label()
            return

        if self.current_tool == self.TOOL_CONNECT:
            ep = self._nearest_endpoint(pos)
            if ep:
                # Start drag from endpoint
                self._connect_first_endpoint = ep
                self._connect_dragging = True
                self._clear_endpoint_highlights()
                self._highlight_endpoint(ep)
                self._draw_connect_line_preview(ep, pos)
                self._update_status_label()
            return

        if self.current_tool == self.TOOL_SELECT:
            # "Select" is for picking an existing skeleton polyline to edit.
            poly = self._nearest_topology_polyline(pos, max_dist=12)
            if not poly:
                self._update_status_label()
                return

            self._edit_original_polyline = poly
            self._polyline_points = [QPoint(int(x), int(y)) for (x, y) in poly]
            # Switch to Polyline tool for interactive editing (without triggering tool reset)
            self.current_tool = self.TOOL_POLYLINE
            self.polyline_button.setChecked(True)
            self._set_selected_point(None)
            self._refresh_polyline_handles()
            self._draw_polyline_preview(self._polyline_points[-1])
            self._update_polyline_buttons_enabled()
            self._update_status_label()
            return

    def on_tool_mouse_move(self, pos: QPoint, _event) -> None:
        if self.model.mask is None:
            return

        if self.current_tool == self.TOOL_ERASER and self._eraser_active:
            self.model.erase_circle((pos.x(), pos.y()), self._eraser_effective_radius())
            self._update_skeleton_display_throttled()
            return

        if self.current_tool == self.TOOL_POLYLINE and self._polyline_points:
            if self._polyline_dragging and self._polyline_drag_index is not None:
                # Move the selected control point
                self._polyline_points[self._polyline_drag_index] = pos
                self._refresh_polyline_handles()
                self._draw_polyline_preview(pos)
                self._update_polyline_buttons_enabled()
                return

            if self._polyline_translate_dragging and self._polyline_translate_anchor is not None and self._polyline_translate_orig_points:
                dx = pos.x() - self._polyline_translate_anchor.x()
                dy = pos.y() - self._polyline_translate_anchor.y()
                new_pts: List[QPoint] = []
                for p0 in self._polyline_translate_orig_points:
                    new_pts.append(self.clamp_to_image(QPoint(p0.x() + dx, p0.y() + dy)))
                self._polyline_points = new_pts
                self._refresh_polyline_handles()
                self._draw_polyline_preview(pos)
                self._update_polyline_buttons_enabled()
                return

            # Hover rubber-banding preview
            self._draw_polyline_preview(pos)
            return

        if self.current_tool == self.TOOL_CONNECT and self._connect_dragging and self._connect_first_endpoint:
            self._draw_connect_line_preview(self._connect_first_endpoint, pos)
            return

    def on_tool_mouse_release(self, pos: QPoint, _event) -> None:
        if self.current_tool == self.TOOL_ERASER and self._eraser_active:
            self._eraser_active = False
            # Clean to 1px at end of stroke for stable endpoints
            self.model.skeletonize()
            # Full display update with endpoints on release
            self._update_skeleton_display(force_endpoints=True)
            return

        if self.current_tool == self.TOOL_POLYLINE:
            # End dragging modes
            self._polyline_dragging = False
            self._polyline_drag_index = None
            self._polyline_translate_dragging = False
            self._polyline_translate_anchor = None
            self._polyline_translate_orig_points = []
            # Keep preview consistent
            if self._polyline_points:
                self._draw_polyline_preview(pos)
            return

        if self.current_tool == self.TOOL_CONNECT and self._connect_dragging and self._connect_first_endpoint:
            # Commit the line from endpoint to release position
            first = self._connect_first_endpoint
            self._connect_dragging = False
            self._connect_first_endpoint = None
            self._clear_connect_line_preview()
            self._clear_endpoint_highlights()

            self.model.push_undo()
            self.model.draw_polyline([(first.x(), first.y()), (pos.x(), pos.y())], thickness=self.draw_thickness)
            self.model.skeletonize()
            self._update_skeleton_display()
            self._update_status_label()
            return

    def on_tool_mouse_double_click(self, pos: QPoint, _event) -> None:
        if self.current_tool == self.TOOL_POLYLINE and self._polyline_points:
            # Finish polyline by adding final point (if different) and committing
            if pos != self._polyline_points[-1]:
                self._polyline_points.append(pos)
            self._commit_polyline()

    def on_key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key.Key_Escape:
            self._polyline_points.clear()
            self._polyline_dragging = False
            self._polyline_drag_index = None
            self._polyline_translate_dragging = False
            self._polyline_translate_anchor = None
            self._polyline_translate_orig_points = []
            self._connect_first_endpoint = None
            self._connect_dragging = False
            self._set_selected_point(None)
            self._clear_polyline_preview()
            self._clear_polyline_handles()
            self._clear_connect_line_preview()
            self._clear_endpoint_highlights()
            self._update_polyline_buttons_enabled()
            self._update_status_label()
            return True

        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.current_tool == self.TOOL_POLYLINE and len(self._polyline_points) >= 2:
                self._commit_polyline()
                return True

        if event.key() == Qt.Key.Key_Delete:
            if self.current_tool == self.TOOL_SELECT and self._selected_point is not None:
                self.model.push_undo()
                self.model.erase_circle(
                    (self._selected_point.x(), self._selected_point.y()),
                    max(2, self._eraser_effective_radius()),
                )
                self.model.skeletonize()
                self._set_selected_point(None)
                self._update_skeleton_display()
                self._update_status_label()
                return True

            # Otherwise: clear connect selection (non-destructive)
            self._connect_first_endpoint = None
            self._clear_endpoint_highlights()
            self._update_status_label()
            return True

        return False

    def finish_polyline(self) -> None:
        """Explicit UI affordance to finish the current polyline and start a new one."""
        if self.current_tool != self.TOOL_POLYLINE:
            return
        if len(self._polyline_points) >= 2:
            self._commit_polyline()
        self._update_status_label()

    def cancel_polyline(self) -> None:
        """Cancel current polyline creation without modifying the skeleton mask."""
        self._polyline_points.clear()
        self._clear_polyline_preview()
        self._clear_polyline_handles()
        self._edit_original_polyline = None
        self._polyline_dragging = False
        self._polyline_drag_index = None
        self._polyline_translate_dragging = False
        self._polyline_translate_anchor = None
        self._polyline_translate_orig_points = []
        self._update_polyline_buttons_enabled()
        self._update_status_label()

    def _nearest_skeleton_pixel(self, pos: QPoint, max_dist: int = 10) -> Optional[QPoint]:
        """Find a skeleton pixel near pos for selection, or None if none within radius."""
        if self.model.mask is None:
            return None
        mask = self.model.mask
        h, w = mask.shape
        x0, y0 = pos.x(), pos.y()
        r = int(max_dist)

        x_min = max(0, x0 - r)
        x_max = min(w - 1, x0 + r)
        y_min = max(0, y0 - r)
        y_max = min(h - 1, y0 + r)

        region = mask[y_min : y_max + 1, x_min : x_max + 1]
        ys, xs = np.where(region > 0)

        best = None
        best_d2 = None
        for dy, dx in zip(ys, xs):
            x = x_min + int(dx)
            y = y_min + int(dy)
            ddx = x - x0
            ddy = y - y0
            d2 = ddx * ddx + ddy * ddy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = QPoint(x, y)

        return best

    # -------------------- connect tool line preview --------------------
    def _clear_connect_line_preview(self) -> None:
        if self._connect_line_preview_item is not None:
            self.scene.removeItem(self._connect_line_preview_item)
            self._connect_line_preview_item = None

    def _draw_connect_line_preview(self, start: QPoint, end: QPoint) -> None:
        """Draw a transient preview line for the Connect tool drag using vector graphics."""
        if self._connect_line_preview_item is None:
            self._connect_line_preview_item = QGraphicsLineItem()
            pen = QPen(QColor("#50fa7b"))  # green
            pen.setWidth(2)
            pen.setCosmetic(True)  # Keep width constant regardless of zoom
            self._connect_line_preview_item.setPen(pen)
            self._connect_line_preview_item.setZValue(2)
            self.scene.addItem(self._connect_line_preview_item)
        
        self._connect_line_preview_item.setLine(float(start.x()), float(start.y()), float(end.x()), float(end.y()))

    # -------------------- polyline preview/commit --------------------
    def _clear_polyline_handles(self) -> None:
        for it in self._polyline_handle_items:
            self.scene.removeItem(it)
        self._polyline_handle_items.clear()

    def _refresh_polyline_handles(self) -> None:
        """Create/update draggable control point markers for the current polyline."""
        self._clear_polyline_handles()
        if self.current_tool != self.TOOL_POLYLINE or not self._polyline_points:
            return
        for p in self._polyline_points:
            r = 4
            item = QGraphicsEllipseItem(p.x() - r, p.y() - r, r * 2, r * 2)
            pen = QPen(QColor("#ff79c6"))  # pink
            pen.setWidth(2)
            item.setPen(pen)
            item.setBrush(QBrush(QColor(0, 0, 0, 0)))
            item.setZValue(3)
            self.scene.addItem(item)
            self._polyline_handle_items.append(item)

    def _nearest_polyline_handle_index(self, pos: QPoint, max_dist: int = 8) -> Optional[int]:
        if not self._polyline_points:
            return None
        best_i = None
        best_d2 = None
        for i, p in enumerate(self._polyline_points):
            dx = p.x() - pos.x()
            dy = p.y() - pos.y()
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_i = i
        if best_d2 is None or best_i is None:
            return None
        if best_d2 <= max_dist * max_dist:
            return best_i
        return None

    def _nearest_polyline_segment_index(self, pos: QPoint, max_dist: int = 8) -> Optional[int]:
        """Return index i such that segment (i -> i+1) is closest to pos."""
        if len(self._polyline_points) < 2:
            return None

        def dist2_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
            abx = bx - ax
            aby = by - ay
            apx = px - ax
            apy = py - ay
            denom = abx * abx + aby * aby
            if denom <= 1e-6:
                return apx * apx + apy * apy
            t = (apx * abx + apy * aby) / denom
            t = max(0.0, min(1.0, t))
            cx = ax + t * abx
            cy = ay + t * aby
            dx = px - cx
            dy = py - cy
            return dx * dx + dy * dy

        px, py = float(pos.x()), float(pos.y())
        best_i = None
        best_d2 = None
        for i in range(len(self._polyline_points) - 1):
            a = self._polyline_points[i]
            b = self._polyline_points[i + 1]
            d2 = dist2_point_to_segment(px, py, float(a.x()), float(a.y()), float(b.x()), float(b.y()))
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_i = i
        if best_d2 is None or best_i is None:
            return None
        if best_d2 <= float(max_dist * max_dist):
            return best_i
        return None

    def _clear_polyline_preview(self) -> None:
        if self._polyline_preview_item is not None:
            self.scene.removeItem(self._polyline_preview_item)
            self._polyline_preview_item = None

    def _build_polyline_path(self, points: List[QPoint], cursor_pos: Optional[QPoint]) -> QPainterPath:
        """Build a QPainterPath for the preview; supports smooth (curved) mode."""
        pts = list(points)
        if cursor_pos is not None and (not pts or cursor_pos != pts[-1]):
            pts.append(cursor_pos)
        path = QPainterPath()
        if not pts:
            return path

        p0 = pts[0]
        path.moveTo(float(p0.x()), float(p0.y()))

        if (not self._polyline_smooth) or len(pts) < 3:
            for p in pts[1:]:
                path.lineTo(float(p.x()), float(p.y()))
            return path

        # Catmull–Rom to cubic Bezier for smooth curve through points
        for i in range(len(pts) - 1):
            P0 = pts[i - 1] if i > 0 else pts[i]
            P1 = pts[i]
            P2 = pts[i + 1]
            P3 = pts[i + 2] if (i + 2) < len(pts) else pts[i + 1]

            c1x = float(P1.x()) + (float(P2.x()) - float(P0.x())) / 6.0
            c1y = float(P1.y()) + (float(P2.y()) - float(P0.y())) / 6.0
            c2x = float(P2.x()) - (float(P3.x()) - float(P1.x())) / 6.0
            c2y = float(P2.y()) - (float(P3.y()) - float(P1.y())) / 6.0
            path.cubicTo(c1x, c1y, c2x, c2y, float(P2.x()), float(P2.y()))

        return path

    def _draw_polyline_preview(self, cursor_pos: QPoint) -> None:
        """Draw a transient preview polyline on top of the scene using vector graphics."""
        if not self._polyline_points:
            return

        if self._polyline_preview_item is None:
            self._polyline_preview_item = QGraphicsPathItem()
            pen = QPen(QColor("#8be9fd"))  # cyan
            pen.setWidth(2)
            pen.setCosmetic(True)
            self._polyline_preview_item.setPen(pen)
            self._polyline_preview_item.setZValue(2)
            self.scene.addItem(self._polyline_preview_item)

        self._polyline_preview_item.setPath(self._build_polyline_path(self._polyline_points, cursor_pos))

    def _commit_polyline(self) -> None:
        if self.model.mask is None or len(self._polyline_points) < 2:
            self._polyline_points.clear()
            self._clear_polyline_preview()
            self._clear_polyline_handles()
            self._edit_original_polyline = None
            self._update_polyline_buttons_enabled()
            return

        pts: List[Tuple[int, int]]
        if self._polyline_smooth and len(self._polyline_points) >= 3:
            pts = self._sample_smooth_polyline_points(self._polyline_points, step_px=0.75)
        else:
            pts = [(p.x(), p.y()) for p in self._polyline_points]

        self.model.push_undo()
        # If we are editing an existing skeleton polyline (picked via Select), erase it first.
        if self._edit_original_polyline and self.model.mask is not None and len(self._edit_original_polyline) >= 2:
            try:
                erase_pts = np.array(self._edit_original_polyline, dtype=np.int32).reshape((-1, 1, 2))
                erase_thickness = max(5, int(self.draw_thickness) + 6)
                cv2.polylines(self.model.mask, [erase_pts], isClosed=False, color=0, thickness=erase_thickness)
            except Exception:
                # Best-effort erase; if it fails we still draw the new polyline.
                pass
            self._edit_original_polyline = None

        self.model.draw_polyline(pts, thickness=self.draw_thickness)
        self.model.skeletonize()
        self._polyline_points.clear()
        self._clear_polyline_preview()
        self._clear_polyline_handles()
        self._update_skeleton_display()
        self._update_polyline_buttons_enabled()
        self._update_status_label()

    def _nearest_topology_polyline(self, pos: QPoint, max_dist: int = 12) -> Optional[List[Tuple[int, int]]]:
        """Pick the nearest vectorized skeleton polyline (for Select->Edit)."""
        if self.model.mask is None:
            return None
        topo = self.model.topology(simplify_epsilon=1.5)
        if not topo.polylines:
            return None

        px, py = float(pos.x()), float(pos.y())
        best_poly: Optional[List[Tuple[int, int]]] = None
        best_d2: Optional[float] = None

        def dist2_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
            abx = bx - ax
            aby = by - ay
            apx = px - ax
            apy = py - ay
            denom = abx * abx + aby * aby
            if denom <= 1e-6:
                return apx * apx + apy * apy
            t = (apx * abx + apy * aby) / denom
            t = max(0.0, min(1.0, t))
            cx = ax + t * abx
            cy = ay + t * aby
            dx = px - cx
            dy = py - cy
            return dx * dx + dy * dy

        for poly in topo.polylines:
            if len(poly) < 2:
                continue
            local_best: Optional[float] = None
            for i in range(len(poly) - 1):
                ax, ay = float(poly[i][0]), float(poly[i][1])
                bx, by = float(poly[i + 1][0]), float(poly[i + 1][1])
                d2 = dist2_point_to_segment(px, py, ax, ay, bx, by)
                if local_best is None or d2 < local_best:
                    local_best = d2
            if local_best is None:
                continue
            if best_d2 is None or local_best < best_d2:
                best_d2 = local_best
                best_poly = [(int(x), int(y)) for (x, y) in poly]

        if best_d2 is None or best_poly is None:
            return None
        if best_d2 <= float(max_dist * max_dist):
            return best_poly
        return None

    def _sample_smooth_polyline_points(self, points: List[QPoint], *, step_px: float = 0.75) -> List[Tuple[int, int]]:
        """Sample a Catmull–Rom spline through points into integer pixels for raster drawing."""
        if len(points) < 2:
            return []

        # If too few points, treat as straight
        if len(points) < 3:
            return [(p.x(), p.y()) for p in points]

        out: List[Tuple[int, int]] = []

        def add_pt(ix: int, iy: int) -> None:
            pt = self.clamp_to_image(QPoint(ix, iy))
            tup = (pt.x(), pt.y())
            if not out or out[-1] != tup:
                out.append(tup)

        for i in range(len(points) - 1):
            P0 = points[i - 1] if i > 0 else points[i]
            P1 = points[i]
            P2 = points[i + 1]
            P3 = points[i + 2] if (i + 2) < len(points) else points[i + 1]

            # Determine sampling density based on segment length
            dx = float(P2.x() - P1.x())
            dy = float(P2.y() - P1.y())
            seg_len = math.hypot(dx, dy)
            steps = max(2, int(seg_len / max(step_px, 0.25)))

            for s in range(steps + 1):
                t = float(s) / float(steps)
                t2 = t * t
                t3 = t2 * t

                # Catmull–Rom spline (uniform)
                x = 0.5 * (
                    (2.0 * float(P1.x()))
                    + (-float(P0.x()) + float(P2.x())) * t
                    + (2.0 * float(P0.x()) - 5.0 * float(P1.x()) + 4.0 * float(P2.x()) - float(P3.x())) * t2
                    + (-float(P0.x()) + 3.0 * float(P1.x()) - 3.0 * float(P2.x()) + float(P3.x())) * t3
                )
                y = 0.5 * (
                    (2.0 * float(P1.y()))
                    + (-float(P0.y()) + float(P2.y())) * t
                    + (2.0 * float(P0.y()) - 5.0 * float(P1.y()) + 4.0 * float(P2.y()) - float(P3.y())) * t2
                    + (-float(P0.y()) + 3.0 * float(P1.y()) - 3.0 * float(P2.y()) + float(P3.y())) * t3
                )
                add_pt(int(round(x)), int(round(y)))

        return out

    def apply_normalization(self) -> None:
        """Apply CLAHE or contrast stretching to the displayed image (non-destructive)."""
        if not self.current_image_path:
            return
        method = self.norm_controls.method_combo.currentText()
        img = cv2.imread(self.current_image_path)
        if img is None:
            return

        if method == "CLAHE":
            clip_limit = self.norm_controls.clip_slider.value() / 10.0
            tile = self.norm_controls.tile_slider.value()
            enhanced = ImageNormalization.apply_clahe(
                img, clip_limit=clip_limit, tile_size=(tile, tile)
            )
        else:
            lower = self.norm_controls.lower_slider.value()
            upper = self.norm_controls.upper_slider.value()
            enhanced = ImageNormalization.apply_contrast_stretching(
                img, lower_percentile=lower, upper_percentile=upper
            )

        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        h, w, _c = rgb.shape
        q_img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(q_img)

        # Preserve original size/scene rect; scale if needed
        if self._original_image_pixmap and (pix.size() != self._original_image_pixmap.size()):
            pix = pix.scaled(
                self._original_image_pixmap.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.image_item.setPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))


