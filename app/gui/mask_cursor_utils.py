"""
Cursor utilities for the mask tracing interface.
Handles brush cursor creation for drawing operations.
"""

from PyQt6.QtGui import QPixmap, QPainter, QPen, QCursor
from PyQt6.QtCore import Qt, QPoint


def create_brush_cursor(size):
    """
    Creates a custom brush cursor based on the brush size.
    
    Args:
        size: Brush diameter in pixels
        
    Returns:
        QCursor: Custom cursor for drawing
    """
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


def create_panning_cursor():
    """
    Creates a panning cursor (open hand).
    
    Returns:
        QCursor: Open hand cursor for panning
    """
    return QCursor(Qt.CursorShape.OpenHandCursor)

