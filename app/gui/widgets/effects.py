"""Depth helpers — QSS can't do box-shadow, so use QGraphicsDropShadowEffect."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QWidget


def drop_shadow(widget: QWidget, *, blur: int = 24, dy: int = 8, dx: int = 0,
                alpha: int = 140) -> QGraphicsDropShadowEffect:
    """Attach a soft drop shadow to ``widget`` and return the effect."""
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur)
    eff.setXOffset(dx)
    eff.setYOffset(dy)
    eff.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(eff)
    return eff


def pop_shadow(widget: QWidget) -> QGraphicsDropShadowEffect:
    """Stronger shadow for floating popovers/docks (--shadow-pop)."""
    return drop_shadow(widget, blur=50, dy=18, alpha=180)
