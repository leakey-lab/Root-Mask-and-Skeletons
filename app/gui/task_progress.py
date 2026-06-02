"""
Task progress widget — a status-bar-mounted progress indicator.

Fixes the long-standing cramped UX where progress text was painted *on* the bar
and where root length/area calculations ran silently. Here the operation name and
percent live in a label ROW ABOVE the bar (``setTextVisible(False)`` on the bar).

This widget knows nothing about handlers. It is mounted into the status bar by
``MainWindow`` (``status_bar.addPermanentWidget(...)`` + ``main_window.task_progress``)
and driven via its small public API:

    start(operation, cancelable=False)
    set_progress(value: int)
    set_operation(operation: str)
    finish(message: str | None = None)
    fail(message: str)

It is hidden by default; ``start`` shows it, ``finish``/``fail`` hide it.

Styled with inline QSS matching the app's Dracula palette (self-contained — no
external theme dependency).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# SPROUTS refined-dark tokens (mirrors app.gui.widgets.tokens / dark_theme.qss).
_BG_ALT = "#21232e"       # --bg-2
_SURFACE = "#282a37"      # --bg-3
_SURFACE_HI = "#373a4c"   # --border-strong
_BORDER = "#2a2c39"       # --border
_TEXT = "#eceef5"         # --text
_ACCENT = "#c39af6"       # --accent (purple)
_DANGER = "#ec6a78"       # --danger (cancel hover)

_LABEL_QSS = f"color: {_TEXT}; font-size: 9pt; font-weight: 600;"

_PROGRESS_QSS = f"""
QProgressBar {{
    background-color: {_BG_ALT};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {_ACCENT};
    border-radius: 3px;
}}
"""

_CANCEL_QSS = f"""
QToolButton {{
    background-color: {_SURFACE};
    color: {_TEXT};
    border: 1px solid {_SURFACE_HI};
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 8pt;
}}
QToolButton:hover {{
    background-color: {_BORDER};
    border: 1px solid {_DANGER};
    color: {_DANGER};
}}
"""


class TaskProgressWidget(QWidget):
    """Compact progress widget: ``<operation> — <pct>%`` label above a styled bar.

    Signals:
        canceled: emitted when the user clicks the (optional) Cancel button.
    """

    canceled = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._operation = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 2, 8, 2)
        outer.setSpacing(2)

        # --- Label row: "<operation> — <pct>%"  + optional Cancel ---
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self.label = QLabel("")
        self.label.setStyleSheet(_LABEL_QSS)
        row.addWidget(self.label, 1)

        self.cancel_button = QToolButton()
        self.cancel_button.setText("Cancel")
        self.cancel_button.setToolTip("Cancel the current operation")
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.setStyleSheet(_CANCEL_QSS)
        self.cancel_button.clicked.connect(self.canceled.emit)
        self.cancel_button.hide()
        row.addWidget(self.cancel_button, 0)

        outer.addLayout(row)

        # --- Progress bar (text OFF — text lives in the label above) ---
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setMinimumWidth(360)
        self.bar.setStyleSheet(_PROGRESS_QSS)
        outer.addWidget(self.bar)

        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self, operation: str, cancelable: bool = False) -> None:
        """Begin a new operation: reset to 0%, show the widget (and Cancel if asked)."""
        self._operation = operation or ""
        self.bar.setValue(0)
        self.cancel_button.setVisible(bool(cancelable))
        self._render(0)
        self.show()

    def set_progress(self, value: int) -> None:
        """Update the percentage (clamped to 0..100) and refresh the label."""
        try:
            v = int(value)
        except (TypeError, ValueError):
            return
        v = max(0, min(100, v))
        if not self.isVisible():
            self.show()
        self.bar.setValue(v)
        self._render(v)

    def set_operation(self, operation: str) -> None:
        """Change the operation name shown in the label (percent unchanged)."""
        self._operation = operation or ""
        self._render(self.bar.value())

    def finish(self, message: str | None = None) -> None:
        """Complete the operation: snap to 100%, then hide.

        If ``message`` is given it is briefly shown in the label before hiding so
        a glance confirms success; otherwise the widget hides immediately.
        """
        self.bar.setValue(100)
        self.cancel_button.hide()
        if message:
            self.label.setText(message)
            QTimer.singleShot(1500, self._hide_and_reset)
        else:
            self._hide_and_reset()

    def fail(self, message: str) -> None:
        """Mark the operation as failed: show ``message`` briefly, then hide."""
        self.cancel_button.hide()
        self.label.setText(message or "Failed")
        QTimer.singleShot(2500, self._hide_and_reset)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _render(self, pct: int) -> None:
        if self._operation:
            self.label.setText(f"{self._operation} — {pct}%")
        else:
            self.label.setText(f"{pct}%")

    def _hide_and_reset(self) -> None:
        self.hide()
        self.bar.setValue(0)
        self.label.setText("")
        self._operation = ""
