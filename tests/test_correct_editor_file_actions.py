"""The Correct page's file actions (``app/gui/correct/actions.py``): Load skeleton
file asks about the unsaved skeleton only, after reading the chosen file; Remove
GT… is refused while edits are unsaved.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QFileDialog

from app.gui.correct.layer import MASK, SKELETON
from tests.correct_support import (
    CANCEL,
    DISCARD,
    SAVE,
    box_plane,
    draw_line,
    paint,
    plane,
    raster,
    record,
    same_path,
    write_mask,
    write_skeleton,
)

pytestmark = pytest.mark.usefixtures("correct_guards")


def test_load_skeleton_file_asks_about_the_skeleton_only_then_opens_the_file(
    make_editor, project, monkeypatch, answers
):
    chosen = write_skeleton(os.path.join(project.folder, "elsewhere", "x.png"), row=20)
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (60, 100))
    draw_line(editor)
    starts = []

    def dialog(parent, title, start, filters):
        starts.append(start)
        return chosen, ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", dialog)
    answers.will(DISCARD)
    editor.controls.load_button.click()

    assert answers.seen == [("Unsaved skeleton", "Save changes to the skeleton for a.png?")]
    assert len(starts) == 1 and same_path(starts[0], project.generated())
    assert raster(editor)[20].any() and editor.layers[SKELETON].document.source == "loaded"
    assert editor.session.journal.undo_entries == (MASK,)  # the skeleton's entries are gone
    assert editor.layers[MASK].is_dirty()


def test_a_loaded_skeleton_file_leaves_no_skeleton_step_to_undo(make_editor, project, monkeypatch):
    chosen = write_skeleton(os.path.join(project.folder, "elsewhere", "x.png"), row=20)
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    assert editor.save()  # clean again, its action still in the timeline
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (chosen, ""))

    editor.controls.load_button.click()  # nothing unsaved: no prompt

    assert raster(editor)[20].any() and editor.session.journal.undo_entries == ()


def test_an_unreadable_skeleton_file_keeps_the_skeleton_timeline(make_editor, project, monkeypatch):
    bad = os.path.join(project.folder, "elsewhere", "broken.png")
    os.makedirs(os.path.dirname(bad), exist_ok=True)
    with open(bad, "wb") as fh:
        fh.write(b"not an image")
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    assert editor.save()  # clean again, its action still in the timeline
    notices = record(editor.notice)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (bad, ""))

    editor.controls.load_button.click()

    assert notices == [("Could not read skeleton broken.png", "danger")]
    assert editor.session.journal.undo_entries == (SKELETON,)
    editor.undo()
    assert not raster(editor)[130].any()  # the line's undo still works
    assert not editor.undo_button.isEnabled()


def test_loading_a_skeleton_file_needs_an_open_image(make_editor):
    editor = make_editor()
    notices = record(editor.notice)

    editor.controls.load_requested.emit()  # the guard fails on any file dialog

    assert notices == [("Open an image first", "warn")]


def test_a_cancelled_load_dialog_changes_nothing(make_editor, project, monkeypatch):
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    before = raster(editor).copy()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))

    editor.controls.load_button.click()  # the guard fails on any prompt

    assert np.array_equal(raster(editor), before)
    assert editor.session.journal.undo_entries == (SKELETON,)


# ------------------------------------------------------------- Load skeleton file: the prompt
def chosen_file(project, monkeypatch) -> str:
    """The Load dialog answers with another skeleton file (one line on row 20)."""
    chosen = write_skeleton(os.path.join(project.folder, "elsewhere", "x.png"), row=20)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (chosen, ""))
    return chosen


def test_cancel_at_the_load_prompt_loads_nothing_and_keeps_the_skeleton_edits(
    make_editor, project, monkeypatch, answers
):
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    edited = raster(editor).copy()
    chosen_file(project, monkeypatch)
    answers.will(CANCEL)

    editor.controls.load_button.click()

    assert answers.seen == [("Unsaved skeleton", "Save changes to the skeleton for a.png?")]
    assert np.array_equal(raster(editor), edited) and not raster(editor)[20].any()
    skeleton = editor.layers[SKELETON]
    assert skeleton.is_dirty() and skeleton.document.source == "prediction"
    assert editor.session.journal.undo_entries == (SKELETON,)  # the line can still be undone
    assert not os.path.exists(project.correction())


def test_save_at_the_load_prompt_writes_the_gt_skeleton_then_opens_the_file(
    make_editor, project, monkeypatch, answers
):
    editor = make_editor(project.a, project.folder)
    draw_line(editor)  # from (40, 125) to (160, 135)
    chosen_file(project, monkeypatch)
    answers.will(SAVE)

    editor.controls.load_button.click()

    saved = cv2.imread(project.correction(), cv2.IMREAD_GRAYSCALE)
    assert saved is not None and saved[124:137, 40:161].any()  # the edit is GT now
    assert raster(editor)[20].any() and editor.layers[SKELETON].document.source == "loaded"
    assert editor.session.journal.undo_entries == ()


def test_an_unreadable_file_costs_no_edits_and_asks_nothing(
    make_editor, project, monkeypatch, answers
):
    bad = os.path.join(project.folder, "elsewhere", "broken.png")
    os.makedirs(os.path.dirname(bad), exist_ok=True)
    with open(bad, "wb") as fh:
        fh.write(b"not an image")
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    edited = raster(editor).copy()
    notices = record(editor.notice)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (bad, ""))

    editor.controls.load_button.click()  # answers has no answer queued: a prompt fails

    assert notices == [("Could not read skeleton broken.png", "danger")] and answers.seen == []
    assert np.array_equal(raster(editor), edited) and editor.layers[SKELETON].is_dirty()
    assert editor.session.journal.undo_entries == (SKELETON,)


# ------------------------------------------------------------------------------- Remove GT…
def test_remove_gt_with_unsaved_edits_deletes_nothing_and_keeps_them(make_editor, project):
    """The guard behind the Remove GT… button, which is off while edits are unsaved
    (test_correct_classes_editor.py checks the button)."""
    write_mask(project.gt_mask(), box_plane())  # a GT mask Remove GT would delete
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (60, 100))
    edited = plane(editor).copy()
    asked = []
    editor.actions.confirm_remove = lambda *args: asked.append(args) or True

    editor.remove_gt()

    assert asked == [] and os.path.isfile(project.gt_mask())
    assert np.array_equal(plane(editor), edited) and editor.has_unsaved_changes()
