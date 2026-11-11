"""Leaving an image with a skeleton draft or a pick on the Correct page (PgDn, a tree
click, the prompt's Discard).

Paging never saves a polyline draft that could be finished: the page stays and says
"Finish the line (Enter) or cancel it (Esc) first", and brings a hidden draft back (the
skeleton becomes the edited layer, shown if it was hidden) so the keys it names reach it.
A branch picked with Select and left as it was is no unsaved work (S5): PgDn drops it and
moves on, and nothing announces a dropped draft; a picked branch whose points moved, or a
new line of two points, keeps the page where it is. With the skeleton as saved and only a
draft pending, Discard drops the draft and keeps the skeleton's history; a skeleton that
differs from its file is read again.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QKeyEvent

import app.gui.correct.skeleton_document as skeleton_document
from app.gui.correct.layer import MASK, SKELETON
from app.gui.correct.skeleton_layer import SkeletonLayer
from tests.correct_support import DISCARD, Mouse, click, paint, plane, raster, record

NO_MODS = Qt.KeyboardModifier.NoModifier

pytestmark = pytest.mark.usefixtures("correct_guards")


def pick(editor) -> None:
    """Select the generated line of image a (row 110) with the Select tool."""
    editor.controls.tool_buttons[(SKELETON, "select")].click()
    click(editor, (100, 111))
    assert editor.layers[SKELETON].draft.edit_original is not None


def move_first_point(editor, dy: int = 12) -> None:
    """Drag the picked branch's first point ``dy`` px down."""
    first = editor.layers[SKELETON].draft.points[0]
    router, mouse = editor.router, Mouse()
    router.canvas_press(first, mouse)
    router.canvas_move(first + QPoint(0, dy), mouse, True)
    router.canvas_release(first + QPoint(0, dy), mouse)


def draft(editor, *points) -> None:
    """A polyline draft of ``points``, not finished."""
    editor.controls.tool_buttons[(SKELETON, "polyline")].click()
    for point in points:
        click(editor, point)


def committed_line(editor) -> None:
    """A polyline drawn and finished on the skeleton (one undoable action), then saved."""
    draft(editor, (40, 60), (150, 70))
    editor.layers[SKELETON].finish_polyline()
    assert editor.save() is True
    assert not editor.has_unsaved_changes()


# ---------------------------------------------------------------------------- PgDn, a pick
def test_paging_after_a_plain_pick_drops_the_pick_and_moves_on(make_editor, project):
    editor = make_editor(project.a, project.folder)
    pick(editor)
    moved, notices = record(editor.navigate_requested), record(editor.notice)

    editor.navigate(1)

    assert moved == [1] and notices == []
    skeleton = editor.layers[SKELETON]
    assert skeleton.draft.points == [] and skeleton.draft.edit_original is None
    assert not os.path.exists(project.correction())  # nothing was saved
    assert not editor.has_unsaved_changes()


def test_the_next_image_opens_without_a_dropped_draft_notice(make_editor, project):
    editor = make_editor(project.a, project.folder)
    pick(editor)
    editor.navigate(1)
    notices = record(editor.notice)

    editor.load_image(project.b, project.folder)  # what MainWindow does next

    assert (SkeletonLayer.DRAFT_DISCARDED_NOTICE, "info") not in notices
    assert editor.current_image_path == project.b


def test_a_tree_click_after_a_plain_pick_says_nothing_about_a_draft(make_editor, project):
    editor = make_editor(project.a, project.folder)
    pick(editor)
    notices = record(editor.notice)

    editor.load_image(project.b, project.folder)

    assert (SkeletonLayer.DRAFT_DISCARDED_NOTICE, "info") not in notices
    assert editor.layers[SKELETON].draft.points == []


def test_a_plain_pick_on_the_reference_layer_does_not_hold_paging(make_editor, project):
    editor = make_editor(project.a, project.folder)
    pick(editor)
    editor.set_active_layer(MASK)  # the pick is kept, out of sight
    moved, notices = record(editor.navigate_requested), record(editor.notice)

    editor.navigate(1)

    assert moved == [1] and notices == []
    assert editor.active_layer_id == MASK


def test_paging_after_a_pick_whose_points_moved_still_stays(make_editor, project):
    editor = make_editor(project.a, project.folder)
    pick(editor)
    move_first_point(editor)
    moved, notices = record(editor.navigate_requested), record(editor.notice)

    editor.navigate(1)

    assert moved == [] and notices == [(editor.FINISH_LINE_FIRST, "warn")]
    assert editor.layers[SKELETON].draft.edit_original is not None  # still being edited


def test_paging_with_a_new_two_point_line_still_stays(make_editor, project):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))
    moved, notices = record(editor.navigate_requested), record(editor.notice)

    editor.navigate(1)

    assert moved == [] and notices == [(editor.FINISH_LINE_FIRST, "warn")]
    assert editor.layers[SKELETON].draft_state().points == 2


# ------------------------------------- Enter or Finish on a plain pick, then PgDn (S03-02)
@pytest.mark.parametrize("finish", ["enter", "finish button"])
def test_enter_or_finish_on_a_plain_pick_then_paging_saves_no_gt(make_editor, project, finish):
    editor = make_editor(project.a, project.folder)
    before = raster(editor).copy()
    pick(editor)

    if finish == "enter":
        enter = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        assert editor.router.canvas_key_press(enter) is True
    else:
        editor.controls.finish_button.click()

    assert np.array_equal(raster(editor), before) and not editor.has_unsaved_changes()
    moved = record(editor.navigate_requested)
    editor.navigate(1)
    assert moved == [1]
    assert not os.path.exists(project.correction())  # the pick wrote no GT skeleton


# ------------------------------------------------- a smoothed pick that is dropped (O20)
SMOOTH_TOAST = ("Smoothed branch not applied · pick it and press Enter to apply", "info")


def test_a_tree_click_after_pick_and_smooth_toasts_once_and_asks_nothing(
    make_editor, project, answers
):
    editor = make_editor(project.a, project.folder)
    pick(editor)
    editor.layers[SKELETON].set_smooth(True)  # what the S key does
    notices = record(editor.notice)
    editor.load_image(project.b, project.folder)  # what MainWindow does on a tree click
    assert notices.count(SMOOTH_TOAST) == 1 and answers.seen == []


def test_paging_after_pick_and_smooth_toasts_once_and_moves_on(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    pick(editor)
    editor.layers[SKELETON].set_smooth(True)
    moved, notices = record(editor.navigate_requested), record(editor.notice)
    editor.navigate(1)
    assert moved == [1] and notices == [SMOOTH_TOAST] and answers.seen == []
    assert not editor.has_unsaved_changes()  # a dropped smoothed pick was never unsaved work
    editor.load_image(project.b, project.folder)  # the pick is gone: no second toast
    assert notices.count(SMOOTH_TOAST) == 1


def test_a_pick_with_smooth_already_on_says_nothing(make_editor, project):
    editor = make_editor(project.a, project.folder)
    editor.layers[SKELETON].set_smooth(True)  # sticky, before the pick
    pick(editor)
    notices = record(editor.notice)
    editor.navigate(1)
    assert notices == []


def test_smooth_switched_on_and_off_again_says_nothing(make_editor, project):
    editor = make_editor(project.a, project.folder)
    pick(editor)
    skeleton = editor.layers[SKELETON]
    skeleton.set_smooth(True)
    skeleton.set_smooth(False)
    notices = record(editor.notice)
    editor.navigate(1)
    assert notices == []


def test_escape_after_pick_and_smooth_says_nothing(make_editor, project):
    editor = make_editor(project.a, project.folder)
    pick(editor)
    editor.layers[SKELETON].set_smooth(True)
    notices = record(editor.notice)
    editor.layers[SKELETON].cancel_transient()  # Esc
    editor.navigate(1)
    assert notices == []


# ----------------------------------------------------------------- Discard, a draft only
def test_discard_with_only_a_draft_pending_keeps_the_history(
    make_editor, project, answers, monkeypatch
):
    editor = make_editor(project.a, project.folder)
    committed_line(editor)
    before = raster(editor).copy()
    draft(editor, (30, 20), (120, 30))
    reads = []
    real_read = skeleton_document.imread_unicode
    monkeypatch.setattr(
        skeleton_document,
        "imread_unicode",
        lambda path, *args: reads.append(path) or real_read(path, *args),
    )
    answers.will(DISCARD)

    assert editor.maybe_save_changes() is True  # e.g. leaving the stage

    skeleton = editor.layers[SKELETON]
    assert skeleton.draft_state().points == 0
    assert np.array_equal(raster(editor), before)  # the saved skeleton, as it was
    assert reads == []  # the file was not read again
    assert editor.session.journal.undo_entries == (SKELETON,)
    assert editor.session.can_undo() and skeleton.can_undo()
    assert not editor.has_unsaved_changes()
    editor.undo()  # the line drawn before the draft can still be undone
    assert not raster(editor)[60:71, 40:151].any()


def test_discard_says_the_draft_went(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    committed_line(editor)
    draft(editor, (30, 20), (120, 30))
    notices = record(editor.notice)
    answers.will(DISCARD)

    assert editor.maybe_save_changes() is True

    assert (SkeletonLayer.DRAFT_DISCARDED_NOTICE, "info") in notices


def test_discard_with_the_mask_unsaved_and_a_skeleton_draft(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    committed_line(editor)
    paint(editor, (20, 100), (80, 110))
    draft(editor, (30, 20), (120, 30))
    answers.will(DISCARD)

    assert editor.maybe_save_changes() is True

    assert editor.session.journal.undo_entries == (SKELETON,)  # the mask's went
    assert not editor.layers[MASK].can_undo() and editor.layers[SKELETON].can_undo()
    assert not plane(editor)[100:111, 20:81].any()  # the mask is as saved again
    assert editor.layers[SKELETON].draft_state().points == 0


def test_discard_of_an_unsaved_skeleton_still_reads_the_file_again(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))
    editor.layers[SKELETON].finish_polyline()  # an unsaved edit of the skeleton itself
    draft(editor, (30, 20), (120, 30))
    answers.will(DISCARD)

    assert editor.maybe_save_changes() is True

    skeleton = editor.layers[SKELETON]
    assert not raster(editor)[60:71, 40:151].any()  # back to the generated skeleton
    assert raster(editor)[110, 20:180].any()
    assert not skeleton.can_undo() and editor.session.journal.undo_entries == ()
    assert skeleton.draft_state().points == 0


def key(editor, which) -> bool:
    """A key press on the canvas input (what the canvas does with the keys it gets)."""
    return editor.router.canvas_key_press(QKeyEvent(QEvent.Type.KeyPress, which, NO_MODS))


def test_paging_with_a_hidden_draft_brings_it_back_so_enter_finishes_it(make_editor, project):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))
    editor.set_active_layer(MASK)  # the draft is kept, out of sight
    moved, notices = record(editor.navigate_requested), record(editor.notice)

    editor.navigate(1)

    assert moved == [] and notices == [(editor.FINISH_LINE_FIRST, "warn")]
    assert editor.active_layer_id == SKELETON
    assert editor.controls.prompt.isVisible()  # Finish line (Enter) / Cancel (Esc)
    assert key(editor, Qt.Key.Key_Return)  # the advice works: Enter finishes the line
    assert editor.layers[SKELETON].draft_state().points == 0
    assert raster(editor)[60:71, 40:151].any()
    editor.navigate(1)
    assert moved == [1]  # the finished line was saved and the page moved on


def test_esc_then_cancels_the_hidden_draft(make_editor, project):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))
    editor.set_active_layer(MASK)
    editor.navigate(1)

    assert key(editor, Qt.Key.Key_Escape)

    assert editor.layers[SKELETON].draft_state().points == 0
    assert not editor.has_unsaved_changes()


def test_a_draft_on_a_hidden_skeleton_is_shown_again(make_editor, project):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))
    editor.set_active_layer(MASK)
    editor.set_layer_visible(SKELETON, False)

    editor.navigate(1)

    assert editor.layers[SKELETON].is_visible()
    assert editor.layer_panel.rows[SKELETON].eye_button.isChecked()
    assert editor.active_layer_id == SKELETON


def test_a_one_point_draft_does_not_hold_paging_or_switch_the_layer(make_editor, project):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60))
    editor.set_active_layer(MASK)
    moved = record(editor.navigate_requested)

    editor.navigate(1)

    assert moved == [1] and editor.active_layer_id == MASK
