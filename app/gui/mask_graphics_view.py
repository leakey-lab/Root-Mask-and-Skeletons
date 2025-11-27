"""
Graphics view for the mask tracing interface.
Handles zooming, panning, and drawing event routing.
"""

from PyQt6.QtWidgets import QGraphicsView, QGraphicsPixmapItem
from PyQt6.QtGui import QPainter, QWheelEvent, QMouseEvent
from PyQt6.QtCore import Qt, QPointF, QPoint


class MaskTracingGraphicsView(QGraphicsView):
    """
    Custom QGraphicsView for mask tracing with zoom, pan, and draw modes.
    """
    
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

    def setup_high_quality_rendering(self):
        """Configure high-quality rendering settings"""
        # Enable all quality render hints
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.LosslessImageRendering
            | QPainter.RenderHint.TextAntialiasing
        )

        # Set transformation anchor for better zoom quality
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Enable caching for better performance
        self.setCacheMode(QGraphicsView.CacheMode.CacheBackground)

        # Optimize for interactive use
        self.setOptimizationFlags(
            QGraphicsView.OptimizationFlag.DontSavePainterState
        )

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
            # Zoom with high quality
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

            # Apply high-quality scaling
            self.scale_with_quality(factor)

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
        """Map viewport position to image coordinates"""
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

    def scale_with_quality(self, factor):
        """Apply scaling with high quality settings"""
        # Ensure smooth transformation for all items before scaling
        if self.scene():
            for item in self.scene().items():
                if isinstance(item, QGraphicsPixmapItem):
                    item.setTransformationMode(
                        Qt.TransformationMode.SmoothTransformation
                    )
                    item.setCacheMode(
                        QGraphicsPixmapItem.CacheMode.DeviceCoordinateCache
                    )

        # Apply the scaling
        self.scale(factor, factor)

        # Force a high-quality update
        self.viewport().update()

    def set_zoom(self, factor):
        """Set zoom level directly with high quality (called from zoom slider)"""
        if self.min_zoom <= factor / 100 <= self.max_zoom:
            # Reset transform first
            self.resetTransform()

            # Set new zoom factor
            new_zoom = factor / 100
            self.zoom_factor = new_zoom

            # Apply high-quality scaling
            self.scale_with_quality(new_zoom)

