"""What counts as unsaved on the skeleton layer, and the one prompt that covers it.

Two kinds of pending skeleton work are unsaved besides an edit of the model:

- a polyline draft that could be finished (KNOWN_ISSUES #6, S5): two or more points (a new
  line, or a picked branch whose points moved). Save commits the draft, then writes the
  correction; Discard drops it; Cancel stays with it. PgDn still stays and says how to end
  the draft. A single point, or a picked branch left as it was, is not unsaved;
- a skeleton opened with "Load skeleton file..." (wave 2, E1). Load used to reset the saved
  baseline, so the loaded skeleton read clean ("Loaded file") and the next image, a stage
  switch or PgDn dropped it without a word. The baseline now stays the skeleton that was
  opened or last saved: a loaded file that differs from it is an unsaved edit. The one
  prompt asks about it (Save writes it as the correction, Discard re-reads the start file,
  Cancel stays), PgDn saves it as it saves every unsaved layer, and a file identical to the
  open skeleton stays clean.

The model's own rule (a raster loaded as an edit keeps the saved baseline) is in
``test_skeleton_model_editing.py``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QFileDialog

from app.gui.correct.layer import SKELETON, LayerStatus
from app.gui.correct.skeleton_document import (
    GT,
    LOADED,
    PREDICTION,
    SkeletonDocument,
)
from tests.correct_support import (
    CANCEL,
    DISCARD,
    SAVE,
    Mouse,
    click,
    raster,
    record,
    write_skeleton,
)

pytestmark = pytest.mark.usefixtures("correct_guards")

W, H = 200, 150


def draft(editor, *points) -> None:
    """A polyline draft of ``points``, not finished."""
    editor.controls.tool_buttons[(SKELETON, "polyline")].click()
    for point in points:
        click(editor, point)


def _correction(project) -> np.ndarray | None:
    path = project.correction()
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None


# ================================================================== a polyline draft
def test_a_draft_of_two_points_is_unsaved_and_one_point_is_not(make_editor, project):
    editor = make_editor(project.a, project.folder)
    skeleton = editor.layers[SKELETON]
    dirty = record(editor.dirty_changed)

    draft(editor, (40, 60))
    assert not skeleton.is_dirty() and not editor.has_unsaved_changes()
    click(editor, (150, 70))

    assert skeleton.is_dirty() and editor.has_unsaved_changes()
    assert skeleton.status() == LayerStatus("dirty")
    assert dirty == [True]
    editor.controls.cancel_button.click()  # Esc / Cancel drops it: clean again
    assert not skeleton.is_dirty() and dirty == [True, False]


def test_a_picked_branch_is_unsaved_only_once_its_points_move(make_editor, project):
    editor = make_editor(project.a, project.folder)
    skeleton = editor.layers[SKELETON]
    editor.controls.tool_buttons[(SKELETON, "select")].click()
    click(editor, (100, 111))  # the generated line at row 110
    assert skeleton.draft.edit_original is not None
    assert not skeleton.is_dirty()

    first = skeleton.draft.points[0]
    editor.router.canvas_press(first, _mouse())
    editor.router.canvas_move(first + _offset(0, 12), _mouse(), True)
    editor.router.canvas_release(first + _offset(0, 12), _mouse())

    assert skeleton.is_dirty()


def test_ctrl_s_commits_the_draft_and_writes_it(make_editor, project):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))

    assert editor.save() is True

    saved = _correction(project)
    assert saved is not None and saved[60:71, 40:151].any()  # the line is in the file
    assert editor.layers[SKELETON].draft_state().points == 0
    assert not editor.has_unsaved_changes()
    assert editor.session.journal.undo_entries == (SKELETON,)  # the commit is undoable


def test_a_tree_click_with_a_draft_asks_and_save_keeps_the_line(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))
    answers.will(SAVE)

    editor.load_image(project.b, project.folder)

    assert len(answers.seen) == 1 and "skeleton" in answers.seen[0][1]
    saved = _correction(project)
    assert saved is not None and saved[60:71, 40:151].any()
    assert os.path.basename(editor.current_image_path) == "b.png"


def test_a_tree_click_with_a_draft_answered_discard_drops_it(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))
    answers.will(DISCARD)

    editor.load_image(project.b, project.folder)

    assert len(answers.seen) == 1
    assert _correction(project) is None  # nothing written
    assert os.path.basename(editor.current_image_path) == "b.png"
    assert editor.layers[SKELETON].draft_state().points == 0


def test_a_tree_click_with_a_draft_answered_cancel_stays_with_it(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))
    answers.will(CANCEL)

    editor.load_image(project.b, project.folder)

    assert os.path.basename(editor.current_image_path) == "a.png"
    assert editor.layers[SKELETON].draft_state().points == 2
    assert not raster(editor)[60:71, 40:151].any()  # still a draft, not committed


def test_leaving_the_stage_with_a_draft_asks_first(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))

    answers.will(CANCEL)
    assert editor.maybe_save_changes() is False  # the stage switch is refused
    answers.will(SAVE)
    assert editor.maybe_save_changes() is True

    saved = _correction(project)
    assert saved is not None and saved[60:71, 40:151].any()


def test_a_draft_on_the_reference_layer_is_still_asked_about(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))
    editor.set_active_layer("mask")  # the draft is kept while the mask is edited
    answers.will(DISCARD)

    assert editor.maybe_save_changes() is True
    assert len(answers.seen) == 1 and editor.layers[SKELETON].draft_state().points == 0


def test_paging_with_a_draft_still_stays_and_says_how_to_end_it(make_editor, project):
    editor = make_editor(project.a, project.folder)
    draft(editor, (40, 60), (150, 70))
    moved, notices = record(editor.navigate_requested), record(editor.notice)

    editor.navigate(1)

    assert moved == [] and notices == [(editor.FINISH_LINE_FIRST, "warn")]
    assert _correction(project) is None


def _mouse():
    return Mouse()


def _offset(dx: int, dy: int):
    return QPoint(dx, dy)


# ========================================================= a loaded file: the document
def _line(row: int, size: tuple[int, int] = (W, H)) -> np.ndarray:
    w, h = size
    raster_ = np.zeros((h, w), np.uint8)
    raster_[row, 20 : w - 20] = 255
    return raster_


def _file(path, row: int) -> str:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    assert cv2.imwrite(str(path), _line(row))
    return str(path)


@pytest.fixture
def folder(tmp_path):
    image = tmp_path / "a.png"
    assert cv2.imwrite(str(image), np.full((H, W, 3), 90, np.uint8))
    return tmp_path, str(image)


def _document(root, image) -> SkeletonDocument:
    doc = SkeletonDocument()
    doc.open(image, (W, H), str(root))
    return doc


def test_a_loaded_file_that_differs_is_unsaved_until_saved(folder):
    root, image = folder
    _file(root / "skeletons" / "a_fake.png", row=30)
    other = _file(root / "elsewhere" / "x.png", row=90)
    doc = _document(root, image)

    assert doc.load_file(other) == []

    assert doc.source == LOADED and doc.is_dirty()
    assert doc.status() == LayerStatus("dirty")
    result = doc.save()
    assert result.ok and not doc.is_dirty() and doc.source == GT
    written = cv2.imread(result.path, cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(written, doc.model.mask) and written[90].any()


def test_discard_after_a_load_reads_the_start_file_again(folder):
    root, image = folder
    _file(root / "skeletons" / "a_fake.png", row=30)
    doc = _document(root, image)
    opened = doc.model.mask.copy()
    doc.load_file(_file(root / "elsewhere" / "x.png", row=90))

    doc.reopen()  # the prompt's Discard

    assert not doc.is_dirty() and doc.source == PREDICTION
    assert np.array_equal(doc.model.mask, opened)


def test_a_loaded_copy_of_the_open_skeleton_stays_clean(folder):
    root, image = folder
    generated = _file(root / "skeletons" / "a_fake.png", row=30)
    doc = _document(root, image)

    doc.load_file(_file(root / "elsewhere" / "copy.png", row=30))
    assert not doc.is_dirty() and doc.status() == LayerStatus("none", "Loaded file")
    doc.load_file(generated)
    assert not doc.is_dirty() and doc.source == PREDICTION


def test_a_file_loaded_over_no_skeleton_is_unsaved(folder):
    root, image = folder
    doc = _document(root, image)  # nothing on disk: an empty skeleton
    assert doc.source is None and not doc.is_dirty()

    doc.load_file(_file(root / "elsewhere" / "x.png", row=90))

    assert doc.is_dirty()


def test_an_unreadable_file_leaves_a_clean_skeleton_clean(folder):
    root, image = folder
    _file(root / "skeletons" / "a_fake.png", row=30)
    doc = _document(root, image)
    bad = root / "bad.png"
    bad.write_bytes(b"not an image")

    doc.load_file(str(bad))

    assert not doc.last_load_ok and not doc.is_dirty() and doc.source == PREDICTION


def test_check_file_reads_without_changing_the_document(folder):
    root, image = folder
    _file(root / "skeletons" / "a_fake.png", row=30)
    doc = _document(root, image)
    opened = doc.model.mask.copy()
    bad = root / "bad.png"
    bad.write_bytes(b"not an image")

    assert doc.check_file(_file(root / "elsewhere" / "x.png", row=90)) == []
    assert [(n.message, n.kind) for n in doc.check_file(str(bad))] == [
        ("Could not read skeleton bad.png", "danger")
    ]
    assert np.array_equal(doc.model.mask, opened) and not doc.is_dirty()
    assert doc.source == PREDICTION


# =========================================================== a loaded file: the editor
def _load(editor, monkeypatch, path: str) -> None:
    """Load skeleton file... answered with ``path``."""
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (path, ""))
    editor.controls.load_button.click()


def _other(project, name: str = "x.png", row: int = 20) -> str:
    return write_skeleton(os.path.join(project.folder, "elsewhere", name), row=row)


def test_a_loaded_skeleton_makes_the_page_unsaved(make_editor, project, monkeypatch):
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(SKELETON)
    dirty = record(editor.dirty_changed)

    _load(editor, monkeypatch, _other(project))

    assert raster(editor)[20].any() and editor.has_unsaved_changes()
    assert dirty == [True] and editor.layers[SKELETON].status() == LayerStatus("dirty")


def test_a_tree_click_after_a_load_asks_and_save_writes_the_loaded_skeleton(
    make_editor, project, monkeypatch, answers
):
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(SKELETON)
    _load(editor, monkeypatch, _other(project))
    answers.will(SAVE)

    editor.load_image(project.b, project.folder)

    assert answers.seen == [("Unsaved skeleton", "Save changes to the skeleton for a.png?")]
    saved = _correction(project)
    assert saved is not None and saved[20].any() and not saved[110].any()
    assert os.path.basename(editor.current_image_path) == "b.png"


def test_a_tree_click_after_a_load_answered_discard_writes_nothing(
    make_editor, project, monkeypatch, answers
):
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(SKELETON)
    _load(editor, monkeypatch, _other(project))
    answers.will(DISCARD)

    editor.load_image(project.b, project.folder)

    assert len(answers.seen) == 1 and _correction(project) is None
    assert os.path.basename(editor.current_image_path) == "b.png"


def test_a_tree_click_after_a_load_answered_cancel_stays_with_it(
    make_editor, project, monkeypatch, answers
):
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(SKELETON)
    _load(editor, monkeypatch, _other(project))
    answers.will(CANCEL)

    editor.load_image(project.b, project.folder)

    assert os.path.basename(editor.current_image_path) == "a.png"
    assert raster(editor)[20].any() and editor.has_unsaved_changes()


def test_leaving_the_stage_after_a_load_asks_first(make_editor, project, monkeypatch, answers):
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(SKELETON)
    _load(editor, monkeypatch, _other(project))

    answers.will(CANCEL)
    assert editor.maybe_save_changes() is False
    answers.will(DISCARD)
    assert editor.maybe_save_changes() is True

    assert _correction(project) is None and not editor.has_unsaved_changes()
    assert raster(editor)[110].any() and not raster(editor)[20].any()  # the generated file


def test_paging_after_a_load_saves_it_as_the_correction(make_editor, project, monkeypatch):
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(SKELETON)
    _load(editor, monkeypatch, _other(project))
    moved = record(editor.navigate_requested)

    editor.navigate(1)  # PgDn saves every unsaved layer, no prompt

    assert moved == [1]
    saved = _correction(project)
    assert saved is not None and saved[20].any()


def test_load_with_a_polyline_draft_asks_first_and_discard_drops_the_draft(
    make_editor, project, monkeypatch, answers
):
    editor = make_editor(project.a, project.folder)
    editor.controls.tool_buttons[(SKELETON, "polyline")].click()
    for point in ((40, 60), (150, 70)):
        editor.router.canvas_press(QPoint(*point), Mouse())
        editor.router.canvas_release(QPoint(*point), Mouse())
    answers.will(DISCARD)

    _load(editor, monkeypatch, _other(project))

    assert answers.seen == [("Unsaved skeleton", "Save changes to the skeleton for a.png?")]
    assert editor.layers[SKELETON].draft_state().points == 0
    assert raster(editor)[20].any() and not raster(editor)[60:71, 40:151].any()


def test_loading_again_asks_about_the_loaded_skeleton(make_editor, project, monkeypatch, answers):
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(SKELETON)
    _load(editor, monkeypatch, _other(project))
    answers.will(DISCARD)

    _load(editor, monkeypatch, _other(project, "y.png", row=60))

    assert answers.seen == [("Unsaved skeleton", "Save changes to the skeleton for a.png?")]
    assert raster(editor)[60].any() and not raster(editor)[20].any()
    assert editor.has_unsaved_changes()


def test_an_unreadable_file_is_refused_before_the_prompt_and_the_edits_stay(
    make_editor, project, monkeypatch, answers
):
    # C6-03: the file is read before Save / Discard is asked, so a bad file costs no edits.
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(SKELETON)
    _load(editor, monkeypatch, _other(project))  # an unsaved skeleton (row 20)
    bad = os.path.join(project.folder, "elsewhere", "broken.png")
    with open(bad, "wb") as fh:
        fh.write(b"not an image")
    answers.will(DISCARD)
    notices = record(editor.notice)

    _load(editor, monkeypatch, bad)

    assert answers.seen == [] and notices == [("Could not read skeleton broken.png", "danger")]
    assert raster(editor)[20].any() and editor.has_unsaved_changes()


def test_a_save_at_the_prompt_is_read_back_when_the_gt_file_is_loaded(
    make_editor, project, monkeypatch, answers
):
    # The file is read again after the prompt: a Save there rewrites the GT file.
    gt_file = write_skeleton(project.correction(), row=60)
    editor = make_editor(project.a, project.folder)  # opens the GT file (row 60)
    editor.set_active_layer(SKELETON)
    _load(editor, monkeypatch, _other(project))  # an unsaved skeleton (row 20)
    answers.will(SAVE)

    _load(editor, monkeypatch, gt_file)

    assert len(answers.seen) == 1
    assert raster(editor)[20].any() and not raster(editor)[60].any()  # what the Save wrote
    assert not editor.has_unsaved_changes() and editor.layers[SKELETON].document.source == GT
