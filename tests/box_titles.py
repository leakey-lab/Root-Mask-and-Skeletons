"""The title the app gave a message box, on every platform.

Qt's macOS build drops a ``QMessageBox``'s window title (Apple's alerts have none:
``QMessageBox::setWindowTitle`` does nothing there), so ``box.windowTitle()`` is ``""`` on
macOS and the tests that pin a prompt's title failed there (CI run 40970595195). While
:func:`record_titles` is active, a title the app sets with ``box.setWindowTitle(...)`` is
also kept on the box, and :func:`title_of` reads it back (the box's own title otherwise).

Import it as ``from tests.box_titles import record_titles, title_of``.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox

__all__ = ["record_titles", "title_of"]

_TITLE = "sprouts_test_title"


def record_titles(monkeypatch) -> None:
    """Keep each message box title set from Python on the box (until ``monkeypatch`` undoes)."""
    set_title = QMessageBox.setWindowTitle

    def recording(box: QMessageBox, title: str) -> None:
        box.setProperty(_TITLE, title)
        set_title(box, title)

    monkeypatch.setattr(QMessageBox, "setWindowTitle", recording)


def title_of(box: QMessageBox) -> str:
    """The title the app set on ``box`` (recorded), else the box's window title."""
    title = box.property(_TITLE)
    return box.windowTitle() if title is None else title
