"""
QGraphicsView for the skeleton correction editor.

This mirrors the interaction patterns used in the existing mask tracing view:
- Ctrl + mouse wheel zoom
- Toggle pan mode (hand drag)
- Delegate tool-specific mouse handling to the parent interface
- GPU-accelerated rendering via QOpenGLWidget viewport
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QPointF, QPoint
from PyQt6.QtGui import QPainter, QWheelEvent, QMouseEvent, QKeyEvent
from PyQt6.QtWidgets import QGraphicsView

# Try to use OpenGL viewport for GPU-accelerated rendering
try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False


class SkeletonCorrectionGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.skeleton_interface = parent

        # Use OpenGL viewport for GPU-accelerated compositing
        if HAS_OPENGL:
            try:
                gl_widget = QOpenGLWidget()
                self.setViewport(gl_widget)
            except Exception:
                pass  # Fall back to software rendering

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.LosslessImageRendering
            | QPainter.RenderHint.TextAntialiasing
        )

        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.pan_mode = False
        self.drawing = False
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_pan_mode(self, enabled: bool) -> None:
        self.pan_mode = enabled
        self.drawing = False
        if enabled:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
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

            self.scale(factor, factor)
            if hasattr(self.skeleton_interface, "on_zoom_changed"):
                self.skeleton_interface.on_zoom_changed(self.zoom_factor)
            event.accept()
            return

        super().wheelEvent(event)

    def map_to_image(self, pos: QPointF) -> QPoint:
        scene_pos = self.mapToScene(pos.toPoint())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        return self.skeleton_interface.clamp_to_image(QPoint(x, y))

    def mousePressEvent(self, event: QMouseEvent):
        if self.pan_mode:
            if event.button() in [Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton]:
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.map_to_image(event.position())
            self.drawing = True
            self.skeleton_interface.on_tool_mouse_press(pos, event)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.pan_mode:
            super().mouseMoveEvent(event)
            return

        if self.drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = self.map_to_image(event.position())
            self.skeleton_interface.on_tool_mouse_move(pos, event)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.pan_mode:
            if event.button() in [Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton]:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.map_to_image(event.position())
            self.drawing = False
            self.skeleton_interface.on_tool_mouse_release(pos, event)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.pan_mode:
            super().mouseDoubleClickEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.map_to_image(event.position())
            self.skeleton_interface.on_tool_mouse_double_click(pos, event)
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if hasattr(self.skeleton_interface, "on_key_press"):
            handled = self.skeleton_interface.on_key_press(event)
            if handled:
                event.accept()
                return
        super().keyPressEvent(event)


