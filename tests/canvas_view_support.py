"""The smallest canvas views for tests of the shared canvas behaviour (app/gui/canvas_view.py).

``View`` is ``CanvasViewMixin`` on a plain ``QGraphicsView``; ``FramelessView`` is the same
without the frame, so its viewport starts at the widget's origin (device-pixel tests).
"""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QGraphicsView

from app.gui.canvas_view import CanvasViewMixin


class View(CanvasViewMixin, QGraphicsView):
    """The mixin on a plain QGraphicsView (zoom, fit, pan, the viewport policy)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_canvas()


class FramelessView(View):
    """``View`` without a frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
