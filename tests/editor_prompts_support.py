"""A scripted answer to the canvas editors' message boxes (``app.gui.editor_prompts``).

Moved here from ``test_editor_prompts.py``, whose ``ESCAPE`` and ``Answer`` the Correct
page's tests borrowed.
"""

from PyQt6.QtWidgets import QMessageBox

from tests.box_titles import record_titles, title_of

ESCAPE = "escape"  # the scripted answer "the user pressed Esc"


class Answer:
    """Stands in for ``QMessageBox.exec``: records what the box shows, returns an answer."""

    def __init__(self, monkeypatch, answer):
        self.answer = answer
        self.seen: dict = {}
        record_titles(monkeypatch)
        monkeypatch.setattr(QMessageBox, "exec", lambda box: self._exec(box))

    def _exec(self, box):
        self.seen = {
            "icon": box.icon(),
            "title": title_of(box),
            "text": box.text(),
            "info": box.informativeText(),
            "buttons": {box.standardButton(w): w.text() for w in box.buttons()},
            "default": box.standardButton(box.defaultButton()),
            "escape": box.standardButton(box.escapeButton()),
        }
        if self.answer == ESCAPE:  # Esc (or closing the box) answers with the escape button
            return int(box.standardButton(box.escapeButton()).value)
        return int(getattr(self.answer, "value", self.answer))
