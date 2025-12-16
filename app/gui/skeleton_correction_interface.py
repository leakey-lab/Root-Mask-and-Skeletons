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

import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPixmap, QKeySequence, QShortcut
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
)

from .skeleton_correction_graphics_view import SkeletonCorrectionGraphicsView
from .skeleton_graph_model import SkeletonCorrectionModel
from .image_normalization_interface import ImageNormalization, NormalizationControls


class SkeletonCorrectionInterface(QWidget):
    """Main widget for skeleton correction editing."""

    skeleton_saved = pyqtSignal(str)

    TOOL_SELECT = "select"
    TOOL_ERASER = "eraser"
    TOOL_POLYLINE = "polyline"
    TOOL_CONNECT = "connect"
    TOOL_SPLIT = "split"

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
        self._polyline_preview_item: Optional[QGraphicsPixmapItem] = None

        # Connect tool state (drag-from-endpoint line drawing)
        self._connect_first_endpoint: Optional[QPoint] = None
        self._connect_dragging: bool = False  # True while dragging a line from endpoint
        self._connect_line_preview_item: Optional[QGraphicsPixmapItem] = None

        # Select tool state (for precise Delete-key cuts)
        self._selected_point: Optional[QPoint] = None
        self._selection_item: Optional[QGraphicsEllipseItem] = None

        # Endpoint markers
        self._endpoint_items: List[QGraphicsEllipseItem] = []

        # Controls
        self.eraser_radius = 10
        self.draw_thickness = 3
        self.overlay_opacity = 0.85
        
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
        self.split_button = QPushButton("✂️ Split")

        for b in [
            self.select_button,
            self.eraser_button,
            self.polyline_button,
            self.connect_button,
            self.split_button,
        ]:
            b.setStyleSheet(btn_style)
            b.setCheckable(True)
            self.tool_group.addButton(b)
            tools_layout.addWidget(b)

        self.select_button.setChecked(True)

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

        self.mode_toggle = QPushButton("🔒 Draw")
        self.mode_toggle.setStyleSheet(btn_style)
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.setChecked(True)
        layout.addWidget(self.mode_toggle)

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
        # Placeholder hook for future zoom UI
        return

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
        elif button == self.split_button:
            self.current_tool = self.TOOL_SPLIT

        # Reset transient state when switching tools
        self._polyline_points.clear()
        self._connect_first_endpoint = None
        self._connect_dragging = False
        self._set_selected_point(None)
        self._clear_polyline_preview()
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

    def _update_polyline_buttons_enabled(self) -> None:
        in_polyline = self.current_tool == self.TOOL_POLYLINE
        has_pts = len(self._polyline_points) >= 2
        has_any_pts = len(self._polyline_points) > 0
        self.finish_polyline_button.setEnabled(in_polyline and has_pts)
        self.cancel_polyline_button.setEnabled(in_polyline and has_any_pts)
        
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
            if n == 0:
                self.status_label.setText("📏 POLYLINE: Click to place first point. Double-click or Enter to finish.")
            elif n == 1:
                self.status_label.setText(f"📏 POLYLINE: {n} point. Click to add more. Enter/double-click to finish, Esc to cancel.")
            else:
                self.status_label.setText(f"📏 POLYLINE: {n} points. Enter/double-click/Finish to commit, Esc/Cancel to discard.")
        elif tool == self.TOOL_CONNECT:
            if self._connect_first_endpoint is None:
                self.status_label.setText("🔗 CONNECT: Click near an endpoint (orange dot) to select first point.")
            else:
                self.status_label.setText("🔗 CONNECT: Click near another endpoint to draw a connecting line.")
        elif tool == self.TOOL_SPLIT:
            self.status_label.setText("✂️ SPLIT: Click on skeleton to cut/split at that point.")
        elif tool == self.TOOL_SELECT:
            if self._selected_point is None:
                self.status_label.setText("🖱️ SELECT: Click near skeleton to select. Press Delete to cut at selection.")
            else:
                self.status_label.setText("🖱️ SELECT: Point selected. Press Delete to cut, or click elsewhere.")
        else:
            self.status_label.setText("Ready to edit skeleton.")

    def _on_eraser_radius_changed(self, v: int) -> None:
        self.eraser_radius = int(v)
        self.eraser_label.setText(f"Eraser: {v}px")

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
        """Render current mask to overlay pixmap and optionally refresh endpoint markers."""
        if self.model.mask is None:
            self.skeleton_item.setPixmap(QPixmap())
            return

        mask = self.model.mask
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        on = mask > 0
        # Neon-ish green (RGBA)
        rgba[on] = [57, 255, 20, 255]

        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg.copy())
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
        self._clear_connect_line_preview()
        self._update_skeleton_display()

    def undo(self) -> None:
        # If user is mid-polyline, undo removes last point (UI expectation)
        if self.current_tool == self.TOOL_POLYLINE and self._polyline_points:
            self._polyline_points.pop()
            if self._polyline_points:
                self._draw_polyline_preview(self._polyline_points[-1])
            else:
                self._clear_polyline_preview()
            self._update_polyline_buttons_enabled()
            self._update_status_label()
            return

        if self.model.undo():
            self._polyline_points.clear()
            self._connect_first_endpoint = None
            self._connect_dragging = False
            self._set_selected_point(None)
            self._clear_polyline_preview()
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
            self.model.erase_circle((pos.x(), pos.y()), self.eraser_radius)
            self._update_skeleton_display()
            return

        if self.current_tool == self.TOOL_POLYLINE:
            self._polyline_points.append(pos)
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

        if self.current_tool == self.TOOL_SPLIT:
            # split is a small cut at click location
            self.model.push_undo()
            self.model.erase_circle((pos.x(), pos.y()), max(2, self.eraser_radius // 2))
            self.model.skeletonize()
            self._update_skeleton_display()
            return

        if self.current_tool == self.TOOL_SELECT:
            skel_pt = self._nearest_skeleton_pixel(pos, max_dist=10)
            self._set_selected_point(skel_pt)
            self._update_status_label()
            return

    def on_tool_mouse_move(self, pos: QPoint, _event) -> None:
        if self.model.mask is None:
            return

        if self.current_tool == self.TOOL_ERASER and self._eraser_active:
            self.model.erase_circle((pos.x(), pos.y()), self.eraser_radius)
            self._update_skeleton_display_throttled()
            return

        if self.current_tool == self.TOOL_POLYLINE and self._polyline_points:
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
            self._connect_first_endpoint = None
            self._connect_dragging = False
            self._set_selected_point(None)
            self._clear_polyline_preview()
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
                    max(2, self.eraser_radius // 2),
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
        """Draw a transient preview line for the Connect tool drag."""
        base_pix = self.image_item.pixmap()
        if base_pix.isNull():
            return
        preview = QPixmap(base_pix.size())
        preview.fill(Qt.GlobalColor.transparent)

        painter = QPainter(preview)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("#50fa7b"))  # green
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(start, end)
        painter.end()

        if self._connect_line_preview_item is None:
            self._connect_line_preview_item = QGraphicsPixmapItem()
            self._connect_line_preview_item.setZValue(2)
            self.scene.addItem(self._connect_line_preview_item)
        self._connect_line_preview_item.setPixmap(preview)

    # -------------------- polyline preview/commit --------------------
    def _clear_polyline_preview(self) -> None:
        if self._polyline_preview_item is not None:
            self.scene.removeItem(self._polyline_preview_item)
            self._polyline_preview_item = None

    def _draw_polyline_preview(self, cursor_pos: QPoint) -> None:
        """Draw a transient preview polyline on top of the scene."""
        if not self._polyline_points:
            return

        # Render preview into a transparent pixmap the size of the image
        base_pix = self.image_item.pixmap()
        if base_pix.isNull():
            return
        preview = QPixmap(base_pix.size())
        preview.fill(Qt.GlobalColor.transparent)

        painter = QPainter(preview)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = painter.pen()
        pen.setColor(QColor("#8be9fd"))  # cyan
        pen.setWidth(2)
        painter.setPen(pen)

        pts = self._polyline_points + [cursor_pos]
        for i in range(1, len(pts)):
            painter.drawLine(pts[i - 1], pts[i])
        painter.end()

        if self._polyline_preview_item is None:
            self._polyline_preview_item = QGraphicsPixmapItem()
            self._polyline_preview_item.setZValue(2)
            self.scene.addItem(self._polyline_preview_item)
        self._polyline_preview_item.setPixmap(preview)

    def _commit_polyline(self) -> None:
        if self.model.mask is None or len(self._polyline_points) < 2:
            self._polyline_points.clear()
            self._clear_polyline_preview()
            self._update_polyline_buttons_enabled()
            return

        pts = [(p.x(), p.y()) for p in self._polyline_points]
        self.model.push_undo()
        self.model.draw_polyline(pts, thickness=self.draw_thickness)
        self.model.skeletonize()
        self._polyline_points.clear()
        self._clear_polyline_preview()
        self._update_skeleton_display()
        self._update_polyline_buttons_enabled()
        self._update_status_label()

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


