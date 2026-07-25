"""Read-only metrics strip shown beneath the display canvas.

Was a 56px two-line strip whose four cells, in practice, read
``— · — · 18×13 mm · Mask ready`` — root length and area are only produced by
the batch threads and written to CSV (KNOWN_ISSUES #3), so two of the four cells
are permanently em-dashes and a third is a constant. 56px for one live value.

Now a 26px single-line strip that only shows cells it can actually fill: the
selected image name, then LENGTH / AREA *when measured values exist*, then the
fixed FOV, then STATUS. Cells with nothing to say are hidden rather than shown
as ``—``, so the strip stays quiet until it has content.

Still pure presentation: ``set_metrics`` formats what it is handed and never
computes anything. The value QLabels keep their ``—`` fallback text (and their
``_length_value`` / ``_area_value`` / ``_status_value`` names) so existing
callers and tests are unaffected.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from app.gui.widgets import tokens

_DASH = "—"
_FOV_VALUE = "18×13 mm"


class MetricsBar(QWidget):
    """A thin read-only strip of measured metrics under the display canvas."""

    HEIGHT = 26

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("metricsBar")
        self.setFixedHeight(self.HEIGHT)
        self.setStyleSheet(
            f"QWidget#metricsBar {{ background-color: {tokens.BG_1}; "
            f"border-top: 1px solid {tokens.BORDER}; }}"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self._name_cell, self._name_value = self._add_cell(row, None, first=True)
        self._name_value.setStyleSheet(
            f"color: {tokens.TEXT}; font-size: 11.5px; font-weight: 600; border: none;"
        )
        self._length_cell, self._length_value = self._add_cell(row, "LENGTH", accent=True)
        self._area_cell, self._area_value = self._add_cell(row, "AREA")
        _fov_cell, fov_value = self._add_cell(row, "FOV")
        fov_value.setText(_FOV_VALUE)
        self._status_cell, self._status_value = self._add_cell(row, "STATUS")

        row.addStretch(1)

        self.set_metrics()

    def _add_cell(
        self,
        row: QHBoxLayout,
        label_text: str | None,
        accent: bool = False,
        first: bool = False,
    ) -> tuple[QFrame, QLabel]:
        """Add one inline label/value cell; return (cell, value QLabel)."""
        cell = QFrame()
        cell.setObjectName("metricsCell")
        border = "none" if first else f"border-left: 1px solid {tokens.BORDER};"
        cell.setStyleSheet(f"QFrame#metricsCell {{ background-color: transparent; {border} }}")

        line = QHBoxLayout(cell)
        line.setContentsMargins(13, 0, 13, 0)
        line.setSpacing(7)

        if label_text:
            label = QLabel(label_text)
            label.setStyleSheet(
                f"color: {tokens.TEXT_FAINT}; font-family: {tokens.MONO}; "
                f"font-size: 7.5pt; font-weight: 600; letter-spacing: 1px; border: none;"
            )
            line.addWidget(label)

        value = QLabel(_DASH)
        value.setStyleSheet(
            f"color: {tokens.ACCENT if accent else tokens.TEXT}; "
            f"font-family: {tokens.MONO}; font-size: 9.5pt; font-weight: 600; border: none;"
        )
        line.addWidget(value)

        row.addWidget(cell)
        return cell, value

    def set_metrics(
        self,
        length: float | None = None,
        area: float | None = None,
        status: str | None = None,
        name: str | None = None,
    ) -> None:
        """Update the displayed values (read-only). Empty cells are hidden."""
        self._length_value.setText(f"{length:.2f} mm" if length is not None else _DASH)
        self._area_value.setText(f"{area:.2f} mm²" if area is not None else _DASH)
        self._status_value.setText(status if status else _DASH)
        self._name_value.setText(name if name else _DASH)

        self._length_cell.setVisible(length is not None)
        self._area_cell.setVisible(area is not None)
        self._status_cell.setVisible(bool(status))
        self._name_cell.setVisible(bool(name))
