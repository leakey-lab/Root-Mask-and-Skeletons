"""Read-only metrics strip shown beneath the display canvas.

``MetricsBar`` is a thin horizontal token-styled widget with four cells:

    ROOT LENGTH  ·  ROOT AREA  ·  FOV  ·  STATUS

It is *pure presentation*: it surfaces values the app has already computed and
never calculates anything itself. ``set_metrics`` formats whatever it is handed
and falls back to an em-dash (``—``) for any value that is ``None``.

The FOV cell is a fixed informational constant ("18×13 mm").
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.gui.widgets import tokens

_DASH = "—"
_FOV_VALUE = "18×13 mm"


class MetricsBar(QWidget):
    """A thin read-only strip of measured metrics under the display canvas."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("metricsBar")
        self.setFixedHeight(56)
        self.setStyleSheet(
            f"QWidget#metricsBar {{ background-color: {tokens.BG_1}; "
            f"border-top: 1px solid {tokens.BORDER}; }}"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        # Each cell is built once; the value labels are kept for live updates.
        self._length_value = self._add_cell(row, "ROOT LENGTH", accent=True, first=True)
        self._area_value = self._add_cell(row, "ROOT AREA")
        fov_value = self._add_cell(row, "FOV")
        fov_value.setText(_FOV_VALUE)
        self._status_value = self._add_cell(row, "STATUS")

        row.addStretch(1)

        # Default to all-empty until fed real values.
        self.set_metrics()

    def _add_cell(
        self,
        row: QHBoxLayout,
        label_text: str,
        accent: bool = False,
        first: bool = False,
    ) -> QLabel:
        """Add one label/value cell to ``row`` and return its value QLabel."""
        cell = QFrame()
        cell.setObjectName("metricsCell")
        border = "none" if first else f"border-left: 1px solid {tokens.BORDER};"
        cell.setStyleSheet(
            f"QFrame#metricsCell {{ background-color: transparent; {border} }}"
        )

        col = QVBoxLayout(cell)
        col.setContentsMargins(18, 8, 18, 8)
        col.setSpacing(2)

        label = QLabel(label_text)
        label.setStyleSheet(
            f"color: {tokens.TEXT_FAINT}; font-family: {tokens.MONO}; "
            f"font-size: 8pt; font-weight: 600; letter-spacing: 1px; border: none;"
        )

        value_color = tokens.ACCENT if accent else tokens.TEXT
        value = QLabel(_DASH)
        value.setStyleSheet(
            f"color: {value_color}; font-family: {tokens.MONO}; "
            f"font-size: 11pt; font-weight: 600; border: none;"
        )

        col.addWidget(label)
        col.addWidget(value)
        col.addStretch(1)

        row.addWidget(cell)
        return value

    def set_metrics(
        self,
        length: Optional[float] = None,
        area: Optional[float] = None,
        status: Optional[str] = None,
    ) -> None:
        """Update the displayed values (read-only); ``None`` shows ``—``."""
        self._length_value.setText(
            f"{length:.2f} mm" if length is not None else _DASH
        )
        self._area_value.setText(
            f"{area:.2f} mm²" if area is not None else _DASH
        )
        self._status_value.setText(status if status else _DASH)
