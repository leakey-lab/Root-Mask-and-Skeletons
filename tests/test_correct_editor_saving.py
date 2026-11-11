"""Saving on the Correct page: what Ctrl+S writes (GT only, never a prediction, nothing
when nothing is unsaved), the save reports and toasts, PgDn's save, and the one
Save / Discard / Cancel prompt on every way of leaving an edited image, failed saves
included. The Qt-free decisions: ``test_correct_session.py``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from app.gui.correct.layer import MASK, SKELETON
from app.gui.correct.mask_layer import MaskLayer
from app.gui.correct.skeleton_layer import SkeletonLayer
from app.gui.key_text import platform_keys
from app.masks import MASK_THRESHOLD
from app.project_layout import ProjectLayout
from tests.correct_goldens_support import png_colour_type
from tests.correct_support import (
    CANCEL,
    DISCARD,
    SAVE,
    H,
    W,
    alpha,
    box_plane,
    click,
    drag,
    draw_line,
    listing,
    paint,
    plane,
    raster,
    record,
    same_path,
    write_image,
    write_mask,
    write_skeleton,
)

pytestmark = pytest.mark.usefixtures("correct_guards")


# ================================================================================= saving
def test_a_mask_edit_saves_only_the_mask(make_editor, project):
    editor = make_editor(project.a, project.folder)
    masks, skeletons = record(editor.mask_saved), record(editor.skeleton_saved)
    paint(editor, (20, 100), (80, 110))
    notices = record(editor.notice)

    assert editor.save() is True

    assert masks == [project.a] and skeletons == []
    assert notices == [("Saved mask · a.png", "success")]
    assert not os.path.exists(project.correction())
    assert png_colour_type(project.gt_mask()) == 6 and not editor.has_unsaved_changes()
    assert png_colour_type(project.mask()) == 0  # the model mask is never rewritten


def test_a_skeleton_edit_saves_only_the_gt_skeleton(make_editor, project):
    with open(project.mask(), "rb") as fh:
        mask_bytes = fh.read()
    editor = make_editor(project.a, project.folder)
    masks, skeletons = record(editor.mask_saved), record(editor.skeleton_saved)
    draw_line(editor)
    notices = record(editor.notice)

    assert editor.save() is True

    assert masks == [] and len(skeletons) == 1 and same_path(skeletons[0], project.correction())
    assert notices == [("Saved skeleton · a.png", "success")]
    with open(project.mask(), "rb") as fh:
        assert fh.read() == mask_bytes
    assert png_colour_type(project.mask()) == 0  # the model mask is never rewritten


# The skeleton half: tests/test_skeleton_save_only_when_changed.py (O2: an untouched skeleton
# is not written).
@pytest.mark.parametrize("layer_id", [MASK, SKELETON])
def test_saving_with_nothing_unsaved_writes_nothing(make_editor, project, layer_id):  # O2
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(layer_id)
    notices = record(editor.notice)
    before = listing(project.folder)

    assert editor.save() is False

    assert listing(project.folder) == before
    assert notices == [("Nothing unsaved", "info")]


def test_saving_both_layers_says_so_in_one_toast(make_editor, project):
    editor = make_editor(project.a, project.folder)
    masks, skeletons = record(editor.mask_saved), record(editor.skeleton_saved)
    paint(editor, (20, 100), (80, 110))
    draw_line(editor)
    notices = record(editor.notice)

    assert editor.save() is True

    assert notices == [("Saved mask and skeleton · a.png", "success")]
    assert masks == [project.a] and len(skeletons) == 1


def test_a_failed_skeleton_save_is_named_stays_unsaved_and_the_mask_is_saved(
    make_editor, project, monkeypatch
):
    import app.gui.correct.skeleton_document as skeleton_document

    editor = make_editor(project.a, project.folder)
    masks, skeletons = record(editor.mask_saved), record(editor.skeleton_saved)
    paint(editor, (20, 100), (80, 110))
    draw_line(editor)
    notices = record(editor.notice)

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(skeleton_document, "imwrite_unicode", fail)
    assert editor.save() is False

    assert notices == [
        ("Saved mask · a.png", "success"),
        ("Could not save gt/skeletons/a.png: disk full", "danger"),
    ]
    assert masks == [project.a] and skeletons == []
    assert editor.layers[SKELETON].is_dirty() and not editor.layers[MASK].is_dirty()
    assert editor.layer_panel.rows[SKELETON].state_label.text() == "Not saved"


def test_a_failed_skeleton_save_keeps_the_old_file(make_editor, project, monkeypatch):
    old = write_skeleton(project.correction(), row=40)
    with open(old, "rb") as fh:
        before = fh.read()
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    notices = record(editor.notice)

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail)
        assert editor.save() is False

    with open(old, "rb") as fh:
        assert fh.read() == before
    assert not [f for f in os.listdir(os.path.dirname(old)) if f.endswith(".tmp")]
    assert [kind for _message, kind in notices] == ["danger"]
    assert editor.has_unsaved_changes() and editor.status_chip.state() == "error"


@pytest.mark.parametrize("layer_id", [MASK, SKELETON])
def test_an_unwritable_folder_is_a_notice_not_a_crash(make_editor, project, monkeypatch, layer_id):
    editor = make_editor(project.a, project.folder)
    if layer_id == MASK:
        paint(editor, (20, 100), (80, 110))
    else:
        draw_line(editor)
    notices = record(editor.notice)

    def denied(*_args, **_kwargs):
        raise PermissionError("read-only share")

    monkeypatch.setattr(os, "makedirs", denied)
    assert editor.save() is False

    assert [kind for _message, kind in notices] == ["danger"] and editor.has_unsaved_changes()


def test_the_skeleton_saves_at_image_resolution_whatever_size_was_read(make_editor, project):
    write_skeleton(project.generated(), row=55, w=W // 2, h=H // 2)
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    assert editor.save()
    saved = cv2.imread(project.correction(), cv2.IMREAD_GRAYSCALE)
    assert (saved.shape[1], saved.shape[0]) == (W, H)
    assert set(np.unique(saved)) <= {0, 255}

    editor.load_image(project.b, project.folder)  # nothing read: image size too
    draw_line(editor)
    assert editor.save()
    saved = cv2.imread(project.correction("b"), cv2.IMREAD_GRAYSCALE)
    assert (saved.shape[1], saved.shape[0]) == (W, H)


def test_an_emptied_mask_saves_an_empty_gt_mask_and_reports_it_once(make_editor, project):
    editor = make_editor(project.a, project.folder)
    cleared, masks, notices = (
        record(editor.mask_cleared),
        record(editor.mask_saved),
        record(editor.notice),
    )

    editor.clear_active_layer()
    assert not os.path.exists(project.gt_mask())  # Clear alone writes nothing
    editor.undo()
    assert plane(editor).any()
    editor.clear_active_layer()
    assert editor.save() is True

    assert os.path.exists(project.mask())  # the prediction stays
    assert png_colour_type(project.gt_mask()) == 6  # O9: an empty GT mask
    assert not cv2.imread(project.gt_mask(), cv2.IMREAD_UNCHANGED).any()
    assert masks == [project.a] and cleared == []
    cleared = platform_keys("Mask cleared. Ctrl+Z to undo; Save writes an empty GT mask.")
    assert (cleared, "info") in notices
    assert notices[-2:] == [
        ("Saved mask · a.png", "success"),
        ("Empty mask saved as GT for a.png: root area 0", "info"),
    ]


def test_the_saved_mask_is_white_on_transparent_and_pixel_identical(make_editor, tmp_path):
    photo = write_image(str(tmp_path / "big" / "plot_001.png"), w=640, h=480)
    editor = make_editor(photo)
    editor.controls.size_sliders[MaskLayer.SIZE_SPEC].slider.setValue(9)
    paint(editor, (10, 10), (600, 400))
    expected = alpha(editor) > 0
    assert editor.save()

    path = str(tmp_path / "big" / "gt" / "mask" / "plot_001.png")
    assert os.path.getsize(path) < 64 * 1024
    rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert np.array_equal(rgba[:, :, 3] == 255, expected)
    assert set(np.unique(rgba)) <= {0, 255}
    assert np.array_equal(rgba[:, :, :3].max(axis=2) == 255, expected)
    assert os.listdir(tmp_path / "big" / "gt" / "mask") == ["plot_001.png"]


def test_saving_writes_the_gt_mask_and_never_the_class_labels(make_editor, project):
    editor = make_editor(project.b, project.folder)
    editor.controls.size_sliders[MaskLayer.SIZE_SPEC].slider.setValue(6)
    paint(editor, (10, 10), (40, 10))
    draw_line(editor)
    assert editor.save()

    layout = ProjectLayout(project.folder)
    gt_mask = layout.gt_mask_path(project.b)
    assert os.path.isfile(gt_mask) and not os.path.exists(layout.gt_path(project.b))
    assert not os.path.exists(project.mask("b"))  # no prediction is ever written
    saved = cv2.imread(gt_mask, cv2.IMREAD_GRAYSCALE)
    assert saved.shape == (H, W) and set(np.unique(saved)) <= {0, 255}
    assert saved[10, 25] == 255 and saved[140, 190] == 0

    editor.load_image(project.b, project.folder)  # the saved files read back
    assert np.array_equal(alpha(editor) > 0, saved > MASK_THRESHOLD)


def test_a_session_without_edits_writes_nothing(make_editor, project):
    editor = make_editor(project.a, project.folder)
    view, chrome = editor.graphics_view, editor.chrome
    written = [
        record(editor.mask_saved),
        record(editor.mask_cleared),
        record(editor.skeleton_saved),
    ]
    moved = record(editor.navigate_requested)
    before = listing(project.folder)

    editor.set_active_layer(SKELETON)
    editor.controls.opacity_slider.slider.setValue(25)
    editor.set_active_layer(MASK)
    for eye in (row.eye_button for row in editor.layer_panel.rows.values()):
        eye.click()
        eye.click()
    view.set_zoom(4.0)
    view.zoom_step(-1)
    view.fit_to_view()
    chrome.pan_button.click()
    chrome.pan_button.click()
    chrome.enhance_button.click()
    chrome.enhance_button.click()
    editor.navigate(1)

    assert listing(project.folder) == before and png_colour_type(project.mask()) == 0
    assert written == [[], [], []] and moved == [1]


# ================================================================== save reports, prompt
def test_the_prompts_save_reports_each_written_file_once(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (80, 110))
    draw_line(editor)
    masks, skeletons = record(editor.mask_saved), record(editor.skeleton_saved)
    answers.will(SAVE)

    assert editor.maybe_save_changes() is True

    assert masks == [project.a] and len(skeletons) == 1


def test_an_emptied_mask_saved_from_the_prompt_reports_mask_saved_once(
    make_editor, project, answers
):
    editor = make_editor(project.a, project.folder)
    editor.clear_active_layer()
    cleared, masks = record(editor.mask_cleared), record(editor.mask_saved)
    answers.will(SAVE)

    assert editor.maybe_save_changes() is True

    assert masks == [project.a] and cleared == [] and os.path.exists(project.mask())
    assert os.path.isfile(project.gt_mask())  # O9: an empty GT mask


def test_a_mask_back_to_the_prediction_removed_from_the_prompt_reports_mask_cleared_once(
    make_editor, project, answers
):
    write_mask(project.gt_mask(), np.zeros((H, W), np.uint8))
    editor = make_editor(project.a, project.folder)
    editor.load_prediction()
    cleared, masks = record(editor.mask_cleared), record(editor.mask_saved)
    answers.will(SAVE)

    assert editor.maybe_save_changes() is True

    assert cleared == [project.a] and masks == [] and not os.path.exists(project.gt_mask())


def test_a_tree_click_answered_save_reports_the_saved_mask_once(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (80, 110))
    masks = record(editor.mask_saved)
    answers.will(SAVE)

    editor.load_image(project.b, project.folder)

    assert masks == [project.a] and same_path(editor.current_image_path, project.b)


def test_paging_saves_without_a_prompt_reports_each_file_once_and_asks_to_move(
    make_editor, project
):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (80, 110))
    draw_line(editor)
    masks, skeletons, moved = (
        record(editor.mask_saved),
        record(editor.skeleton_saved),
        record(editor.navigate_requested),
    )

    editor.navigate(1)  # the guard fails on any prompt

    assert masks == [project.a] and len(skeletons) == 1 and moved == [1]
    assert not editor.has_unsaved_changes()


def test_a_failed_save_reports_nothing_and_paging_stays(make_editor, project, monkeypatch):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (80, 110))
    masks, moved = record(editor.mask_saved), record(editor.navigate_requested)

    def denied(*_args, **_kwargs):
        raise PermissionError("read-only share")

    monkeypatch.setattr(os, "makedirs", denied)
    editor.navigate(1)
    assert editor.save() is False

    assert masks == [] and moved == [] and editor.has_unsaved_changes()


def test_paging_with_a_polyline_draft_stays_and_says_how_to_end_it(make_editor, project):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (80, 110))
    editor.controls.tool_buttons[(SKELETON, "polyline")].click()
    click(editor, (40, 125))
    click(editor, (100, 130))
    moved, notices = record(editor.navigate_requested), record(editor.notice)

    editor.navigate(1)

    assert moved == [] and notices == [(editor.FINISH_LINE_FIRST, "warn")]
    assert editor.layers[SKELETON].draft_state().points == 2
    assert editor.has_unsaved_changes()  # nothing was saved either


def test_opening_another_image_announces_a_dropped_polyline_draft(make_editor, project):
    editor = make_editor(project.a, project.folder)
    editor.controls.tool_buttons[(SKELETON, "polyline")].click()
    click(editor, (40, 125))
    notices = record(editor.notice)

    editor.load_image(project.b, project.folder)

    assert (SkeletonLayer.DRAFT_DISCARDED_NOTICE, "info") in notices
    assert editor.layers[SKELETON].draft_state().points == 0


def test_paging_without_edits_only_asks_for_the_image(make_editor, project):
    editor = make_editor(project.a, project.folder)
    moved = record(editor.navigate_requested)
    before = listing(project.folder)

    editor.navigate(-1)

    assert moved == [-1] and listing(project.folder) == before


def test_one_prompt_names_every_unsaved_layer(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (80, 110))
    answers.will(CANCEL, CANCEL)

    editor.maybe_save_changes()
    draw_line(editor)
    editor.maybe_save_changes()

    assert answers.seen == [
        ("Unsaved mask", "Save changes to the mask for a.png?"),
        ("Unsaved mask and skeleton", "Save changes to the mask and skeleton for a.png?"),
    ]


def test_save_in_the_prompt_saves_then_opens_the_next_image(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    answers.will(SAVE)

    editor.load_image(project.b, project.folder)

    assert os.path.isfile(project.correction())
    assert same_path(editor.current_image_path, project.b) and not editor.has_unsaved_changes()


def test_discard_reverts_both_layers_to_their_files(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    start = raster(editor).copy()
    paint(editor, (20, 100), (80, 110))
    draw_line(editor)
    answers.will(DISCARD)

    assert editor.maybe_save_changes() is True

    assert np.array_equal(plane(editor), box_plane())
    assert np.array_equal(alpha(editor) > 0, box_plane() > 0)
    assert np.array_equal(raster(editor), start)
    assert not editor.has_unsaved_changes() and editor.session.journal.undo_entries == ()
    assert not os.path.exists(project.correction())


def test_discard_on_a_tree_click_opens_the_next_image_unsaved(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    answers.will(DISCARD)

    editor.load_image(project.b, project.folder)

    assert len(answers.seen) == 1 and "a.png" in answers.seen[0][1]
    assert same_path(editor.current_image_path, project.b)
    assert not os.path.exists(project.correction())


def test_cancel_keeps_the_image_its_edits_and_their_history(make_editor, project, answers):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (80, 110))
    draw_line(editor)
    edited = plane(editor).copy(), raster(editor).copy()
    answers.will(CANCEL, CANCEL, CANCEL)

    editor.load_image(project.b, project.folder)
    assert editor.maybe_save_changes() is False
    assert editor.reset() is False

    assert same_path(editor.current_image_path, project.a) and editor.has_unsaved_changes()
    assert np.array_equal(plane(editor), edited[0]) and np.array_equal(raster(editor), edited[1])
    assert editor.session.journal.undo_entries == (MASK, SKELETON)


def test_a_failed_save_on_a_tree_click_keeps_the_open_image(make_editor, project, answers):
    editor = make_editor(project.b, project.folder)
    paint(editor, (10, 10), (40, 10))
    edited = plane(editor).copy()
    os.makedirs(project.gt_mask("b"))  # a folder where the file must go
    answers.will(SAVE, CANCEL)  # Save, then "Keep editing"

    editor.load_image(project.a, project.folder)

    assert [title for title, _text in answers.seen] == ["Unsaved mask", "Mask not saved"]
    assert same_path(editor.current_image_path, project.b) and editor.has_unsaved_changes()
    assert np.array_equal(plane(editor), edited)


def test_a_failed_save_in_the_prompt_asks_again_keep_editing_stays_discard_goes(
    make_editor, project, answers, monkeypatch
):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (80, 110))

    def denied(*_args, **_kwargs):
        raise PermissionError("read-only share")

    monkeypatch.setattr(os, "makedirs", denied)
    answers.will(SAVE, CANCEL)
    assert editor.maybe_save_changes() is False
    assert [title for title, _text in answers.seen] == ["Unsaved mask", "Mask not saved"]
    assert editor.has_unsaved_changes()

    answers.will(SAVE, DISCARD)
    assert editor.maybe_save_changes() is True
    assert not editor.has_unsaved_changes() and np.array_equal(plane(editor), box_plane())


def test_the_clean_page_never_prompts_and_reset_empties_it(make_editor, project):
    editor = make_editor(project.a, project.folder)
    editor.load_image(project.b, project.folder)

    assert editor.maybe_save_changes() is True
    assert editor.reset() is True

    assert editor.current_image_path == "" and editor.status_chip.text() == "No image"
    assert editor.photo_item.image_size() == (0, 0)
    assert not any(layer.is_open() for layer in editor.layers.values())
    drag(editor, (10, 10), (40, 30))
    assert editor.session.journal.undo_entries == ()
