"""Full-window loading overlay shown while a directory is being scanned/loaded.

Thin subclass of the design-system :class:`ProgressOverlay` that adds optional
``set_filename`` / ``set_count`` detail labels. ``start`` / ``set_progress`` /
``hide`` / ``reposition`` are inherited.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.gui.widgets import tokens
from app.gui.widgets.overlays import ProgressOverlay


class LoadingOverlay(ProgressOverlay):
    """Loading progress card with filename + count detail lines."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, width=360)

        # Append two small detail labels under the inherited progress bar.
        detail = QVBoxLayout()
        detail.setContentsMargins(16, 0, 16, 12)
        detail.setSpacing(2)
        self._filename_label = QLabel("")
        self._filename_label.setStyleSheet(
            f"color: {tokens.TEXT_MUTED}; background: transparent; font-size: 11px;"
        )
        self._filename_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._count_label = QLabel("")
        self._count_label.setStyleSheet(
            f"color: {tokens.TEXT_FAINT}; background: transparent; "
            f"font-family: {tokens.MONO}; font-size: 10.5px;"
        )
        detail.addWidget(self._filename_label)
        detail.addWidget(self._count_label)

        # The inherited frame is the first child of self's top-level layout.
        frame = self.findChild(QWidget, "progFrame")
        if frame is not None and frame.layout() is not None:
            frame.layout().addLayout(detail)

    def set_filename(self, name: str) -> None:
        self._filename_label.setText(str(name) if name else "")

    def set_count(self, scanned: int, total: int) -> None:
        if total:
            self._count_label.setText(f"{scanned} / {total}")
        else:
            self._count_label.setText("")
