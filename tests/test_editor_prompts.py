"""editor_prompts (C11): the four boxes of the canvas editor that decide whether edits are saved,
kept or lost.

The Save / Discard / Cancel box, the failed-save box (Discard / Keep editing), the box before a
save replaces a file the editor could not show (Replace / Keep it) and the box before a save
overwrites a file changed on disk (Overwrite / Reload (lose my edits) / Cancel): their wording
word for word, every answer mapped to a plain string, Esc and closing the box the safe answer
(also with the real ``exec()``), and each box freed after use.
"""

import pytest
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from app.gui import editor_prompts
from app.gui.editor_prompts import (
    CANCEL,
    DISCARD,
    KEEP,
    SAVE,
    ask_after_failed_save,
    ask_changed_on_disk,
    ask_replace_unloaded,
    ask_unsaved,
    join_nouns,
)
from tests.editor_prompts_support import ESCAPE, Answer
from tests.qt_teardown import flush_deleted

Button = QMessageBox.StandardButton
REPLACE_PROMPT = (
    "Replace the saved mask?",
    "gt/mask/a.png exists but could not be shown. Replace it?",
)
CHANGED_PROMPT = ("Mask changed on disk", "gt/mask/a.png changed on disk after you opened it.")
CHANGED_INFO = "Overwrite it with your edits, reload it from disk (your edits are lost), or cancel."


@pytest.fixture
def parent(app):
    widget = QWidget()
    yield widget
    widget.deleteLater()
    del widget
    flush_deleted()


NOUNS = ["mask", "skeleton", "mask and skeleton", "GT"]


# --------------------------------------------------------------------------- the unsaved box
@pytest.mark.parametrize("noun", NOUNS)
def test_the_unsaved_box_keeps_todays_wording(monkeypatch, parent, noun):
    answer = Answer(monkeypatch, Button.Save)
    assert ask_unsaved(parent, noun, "plot_7.jpg") == SAVE
    assert answer.seen["icon"] == QMessageBox.Icon.Question
    assert answer.seen["title"] == f"Unsaved {noun}"
    assert answer.seen["text"] == f"Save changes to the {noun} for plot_7.jpg?"
    assert answer.seen["info"] == "Discarded edits cannot be recovered."
    assert set(answer.seen["buttons"]) == {Button.Save, Button.Discard, Button.Cancel}
    assert answer.seen["default"] == Button.Save
    assert answer.seen["escape"] == Button.Cancel


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (Button.Save, SAVE),
        (Button.Discard, DISCARD),
        (Button.Cancel, CANCEL),
        (ESCAPE, CANCEL),
        (0, CANCEL),  # closed some other way
    ],
)
def test_the_unsaved_box_maps_every_answer(monkeypatch, parent, answer, expected):
    Answer(monkeypatch, answer)
    assert ask_unsaved(parent, "mask", "a.jpg") == expected


# --------------------------------------------------------------------------- the failed box
@pytest.mark.parametrize("noun", NOUNS)
def test_the_failed_save_box_keeps_todays_wording(monkeypatch, parent, noun):
    answer = Answer(monkeypatch, Button.Discard)
    assert ask_after_failed_save(parent, noun, "plot_7.jpg") == DISCARD
    assert answer.seen["icon"] == QMessageBox.Icon.Warning
    assert answer.seen["title"] == f"{noun[:1].upper()}{noun[1:]} not saved"
    assert answer.seen["text"] == f"The {noun} for plot_7.jpg could not be saved."
    assert answer.seen["info"] == "Discard the edits, or keep editing? Discarded edits are lost."
    assert set(answer.seen["buttons"]) == {Button.Discard, Button.Cancel}
    assert answer.seen["buttons"][Button.Cancel] == "Keep editing"
    assert answer.seen["default"] == Button.Cancel  # Keep editing
    assert answer.seen["escape"] == Button.Cancel


def test_failed_box_titles_capitalise_only_the_first_letter(monkeypatch, parent):
    answer = Answer(monkeypatch, Button.Discard)
    ask_after_failed_save(parent, "mask and skeleton", "a.jpg")
    assert answer.seen["title"] == "Mask and skeleton not saved"
    ask_after_failed_save(parent, "GT", "a.jpg")
    assert answer.seen["title"] == "GT not saved"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [(Button.Discard, DISCARD), (Button.Cancel, KEEP), (ESCAPE, KEEP), (0, KEEP)],
)
def test_the_failed_save_box_maps_every_answer(monkeypatch, parent, answer, expected):
    Answer(monkeypatch, answer)
    assert ask_after_failed_save(parent, "skeleton", "a.jpg") == expected


# ------------------------------------------------------------- the replace-unloaded box
@pytest.mark.parametrize(
    ("pressed", "expected"),
    [
        (QMessageBox.StandardButton.Save, editor_prompts.REPLACE),
        (QMessageBox.StandardButton.Cancel, editor_prompts.CANCEL),
        (ESCAPE, editor_prompts.CANCEL),
        (0, editor_prompts.CANCEL),  # closed some other way
    ],
)
def test_the_replace_box_names_the_file_and_keeps_it_by_default(
    app, monkeypatch, pressed, expected
):
    answer = Answer(monkeypatch, pressed)

    assert editor_prompts.ask_replace_unloaded(None, "mask", "gt/mask/a.png") == expected

    seen, button = answer.seen, QMessageBox.StandardButton
    assert (seen["title"], seen["text"]) == REPLACE_PROMPT
    assert seen["info"] == "Your edits replace the whole file. Keep it leaves the file as it is."
    assert seen["buttons"] == {button.Save: "Replace", button.Cancel: "Keep it"}
    assert seen["default"] == seen["escape"] == button.Cancel


# ---------------------------------------------------------------- the changed-on-disk box
@pytest.mark.parametrize(
    ("pressed", "expected"),
    [
        (QMessageBox.StandardButton.Save, editor_prompts.OVERWRITE),
        (QMessageBox.StandardButton.Discard, editor_prompts.RELOAD),
        (QMessageBox.StandardButton.Cancel, editor_prompts.CANCEL),
        (ESCAPE, editor_prompts.CANCEL),
        (0, editor_prompts.CANCEL),  # closed some other way: the edits are kept
    ],
)
def test_the_changed_on_disk_box_names_the_file_and_cancels_by_default(
    app, monkeypatch, pressed, expected
):
    answer = Answer(monkeypatch, pressed)

    assert editor_prompts.ask_changed_on_disk(None, "mask", "gt/mask/a.png") == expected

    seen, button = answer.seen, QMessageBox.StandardButton
    assert (seen["title"], seen["text"]) == CHANGED_PROMPT
    assert seen["info"] == CHANGED_INFO
    assert set(seen["buttons"]) == {button.Save, button.Discard, button.Cancel}
    assert seen["buttons"][button.Save] == "Overwrite"
    assert seen["buttons"][button.Discard] == "Reload (lose my edits)"
    assert seen["default"] == seen["escape"] == button.Cancel


# --------------------------------------------------------------- the real box, not a stub
#: Each box and its safe answer: what Esc or closing the window must give.
SAFE = {
    "unsaved": (ask_unsaved, CANCEL),
    "failed save": (ask_after_failed_save, KEEP),
    "replace unloaded": (ask_replace_unloaded, editor_prompts.CANCEL),
    "changed on disk": (ask_changed_on_disk, editor_prompts.CANCEL),
}


def _when_shown(answer) -> None:
    """Run ``answer(box)`` on the modal box once the real ``exec()`` shows it."""

    def act() -> None:
        box = QApplication.activeModalWidget()
        if not isinstance(box, QMessageBox):
            QTimer.singleShot(10, act)
            return
        answer(box)

    QTimer.singleShot(0, act)


@pytest.mark.parametrize("how", ["Esc", "close"])
@pytest.mark.parametrize("name", sorted(SAFE))
def test_esc_or_closing_the_real_box_gives_the_safe_answer(parent, name, how):
    ask, safe = SAFE[name]
    if how == "Esc":
        _when_shown(lambda box: QTest.keyClick(box, Qt.Key.Key_Escape))
    else:
        _when_shown(lambda box: box.close())  # the window's close button

    assert ask(parent, "mask", "gt/mask/a.png") == safe


# --------------------------------------------------------------------------- helpers
@pytest.mark.parametrize(
    ("nouns", "joined"),
    [
        ([], ""),
        (["mask"], "mask"),
        (["mask", "skeleton"], "mask and skeleton"),
        (("mask", "skeleton", "GT"), "mask, skeleton and GT"),
    ],
)
def test_join_nouns(nouns, joined):
    assert join_nouns(nouns) == joined


@pytest.mark.parametrize("ask", [ask_unsaved, ask_after_failed_save])
def test_every_box_is_freed_after_use(monkeypatch, parent, ask):
    Answer(monkeypatch, Button.Discard)
    for _ in range(3):
        ask(parent, "mask", "a.jpg")
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert parent.findChildren(QMessageBox) == []
    assert not [w for w in QApplication.topLevelWidgets() if isinstance(w, QMessageBox)]
