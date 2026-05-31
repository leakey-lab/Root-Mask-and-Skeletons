"""
Drawing tools for the mask tracing interface.
Provides mixin class with drawing, flood fill, and undo/redo functionality.
"""

from PyQt6.QtGui import QPixmap, QPainter, QPen, QBrush, QImage
from PyQt6.QtCore import Qt, QPoint
import numpy as np
import cv2


class MaskDrawingMixin:
    """
    Mixin class providing drawing functionality for mask tracing.
    Should be used with a class that has:
    - self.mask_pixmap: QPixmap for the mask
    - self.brush_color: QColor for the brush
    - self.brush_size: int for brush diameter
    - self.eraser_button: QPushButton for eraser mode
    - self.last_point: QPoint or None for last drawn point
    - self.undo_stack: bounded deque(maxlen=max_stack_size) for undo history
    - self.redo_stack: bounded deque(maxlen=max_stack_size) for redo history
    - self.max_stack_size: int for maximum undo/redo stack size
    - self.update_display(): method to refresh the display
    """

    def draw_point(self, pos):
        """
        Draw a point or line segment on the mask.
        Optimized to draw directly on the mask pixmap.
        
        Args:
            pos: QPoint position to draw at
        """
        if not self.mask_pixmap:
            return

        painter = QPainter(self.mask_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Set composition mode
        if self.eraser_button.isChecked():
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_DestinationOut
            )
        else:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )

        # Set up the pen and brush
        pen = QPen(
            self.brush_color,
            self.brush_size if self.last_point else 1, # Use brush size for line, 1 for ellipse outline (filled)
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        
        # For dots (no last point), we draw a filled ellipse
        # For lines, the Pen width handles the filling
        painter.setPen(pen)
        painter.setBrush(QBrush(self.brush_color))

        # Draw the stroke
        if self.last_point:
            # Drawing a line segment with a thick pen
            painter.drawLine(self.last_point, pos)
        else:
            # Drawing a single dot
            diameter = self.brush_size
            # We need to turn off the pen for the ellipse fill if we want exact size, 
            # or just use the pen width?
            # Original code used a 1px pen and filled ellipse.
            # Let's match original appearance.
            painter.setPen(Qt.PenStyle.NoPen)
            top_left = QPoint(pos.x() - diameter // 2, pos.y() - diameter // 2)
            painter.drawEllipse(top_left.x(), top_left.y(), diameter, diameter)

        painter.end()

        self.last_point = pos
        self.update_display()

    def flood_fill(self, pos):
        """
        Enhanced flood fill that uses the current brush color.
        
        Args:
            pos: QPoint position to start the fill
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
        alpha_channel = arr[:, :, 3].copy()

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
                alpha_channel.copy(),
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

        # Update the mask pixmap. The undo snapshot is taken by the caller
        # (mask_graphics_view.mousePressEvent) before dispatching to flood_fill,
        # so we must NOT save again here or a single fill would cost two undos.
        self.mask_pixmap = QPixmap.fromImage(result_image.copy())
        self.update_display()

    def save_for_undo(self):
        """Save current state for undo with stack size limit."""
        if self.mask_pixmap:
            # Add current state to undo stack. undo_stack is a bounded deque,
            # so appending past max_stack_size auto-evicts the oldest entry.
            self.undo_stack.append(self.mask_pixmap.copy())

            # Clear redo stack as new action invalidates redo history
            self.redo_stack.clear()

    def undo(self):
        """Undo last action with stack size limit."""
        if self.undo_stack:
            # Save current state to redo stack (bounded deque auto-evicts).
            self.redo_stack.append(self.mask_pixmap.copy())

            # Restore previous state from undo stack
            self.mask_pixmap = self.undo_stack.pop()
            self.update_display()

    def redo(self):
        """Redo last undone action with stack size limit."""
        if self.redo_stack:
            # Save current state to undo stack (bounded deque auto-evicts).
            self.undo_stack.append(self.mask_pixmap.copy())

            # Restore next state from redo stack
            self.mask_pixmap = self.redo_stack.pop()
            self.update_display()

