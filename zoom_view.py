import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QPainter, QTransform
from PyQt6.QtCore import Qt, QPoint, QRectF, QPointF


class EnhancedZoomView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )

        # Enable mouse tracking for smooth pan
        self.setMouseTracking(True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        # Initialize zoom parameters
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step = 0.1

        # Pan parameters
        self.is_panning = False
        self.last_mouse_pos = None

        # Create scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

    def wheelEvent(self, event):
        """Handle zoom with mouse wheel"""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # Get the position before zoom
            old_pos = self.mapToScene(event.position().toPoint())

            # Calculate zoom factor
            zoom_in = event.angleDelta().y() > 0
            if zoom_in:
                factor = 1.0 + self.zoom_step
            else:
                factor = 1.0 - self.zoom_step

            # Check zoom bounds
            resulting_zoom = self.zoom_factor * factor
            if self.min_zoom <= resulting_zoom <= self.max_zoom:
                self.zoom_factor = resulting_zoom
                self.scale(factor, factor)

                # Adjust scene position to keep mouse position fixed
                new_pos = self.mapToScene(event.position().toPoint())
                delta = new_pos - old_pos
                self.translate(delta.x(), delta.y())

            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Handle pan start"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle pan end"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """Handle panning"""
        if self.is_panning and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self.last_mouse_pos = event.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def reset_zoom(self):
        """Reset zoom to original scale"""
        self.resetTransform()
        self.zoom_factor = 1.0

    def fit_in_view(self):
        """Fit content in view while maintaining aspect ratio"""
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_factor = 1.0  # Reset zoom factor after fitting

    def set_zoom(self, factor):
        """Set absolute zoom level"""
        if self.min_zoom <= factor <= self.max_zoom:
            # Calculate relative factor
            relative_factor = factor / self.zoom_factor
            self.scale(relative_factor, relative_factor)
            self.zoom_factor = factor
