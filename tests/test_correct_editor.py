"""CorrectEditor (plan P7b, contract C17): the Correct page, one canvas for the mask, the
skeleton and the class labels of an image.

The editor only composes the parts: the canvas (``EditorCanvasView``, input to the
``ToolRouter``), the photo, the layers, the ``CorrectSession``, the ``EditorChrome``, the
``CorrectControls`` and the Layers panel. These tests drive the page on the real parts, as
MainWindow and the user do: the parts MainWindow uses, opening an image and coming back to it
(spec 9), and the layers (spec 4). The page's other aspects have their own files:
``test_correct_editor_controls.py``, ``_saving.py`` (saves and the prompt), ``_file_actions.py``,
``_view.py``, ``_keys.py``, ``_history.py`` (undo and gestures), ``_paging.py`` and
``_drafts.py``; the shared helpers are in ``tests/correct_support.py``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF
from PyQt6.QtTest import QTest

from app.gui import shortcuts
from app.gui.correct.layer import CLASSES, MASK, SKELETON
from app.gui.correct.mask_layer import MaskLayer
from app.gui.correct.skeleton_layer import DraftState, SkeletonLayer
from app.gui.editor_canvas import EditorCanvasView
from app.gui.key_text import platform_keys
from app.gui.skeleton_graph_model import SkeletonCorrectionModel
from app.gui.widgets import (
    EnhancePopover,
    FloatingDock,
    ToolRail,
    ZoomCluster,
)
from tests.correct_support import (
    LEFT,
    H,
    Mouse,
    W,
    alpha,
    box_plane,
    bump,
    click,
    count_opens,
    drag,
    draw_line,
    paint,
    plane,
    raster,
    record,
    same_path,
    stroke_plane,
    write_image,
    write_mask,
    write_skeleton,
)

pytestmark = pytest.mark.usefixtures("correct_guards")


def test_the_page_names_the_parts_mainwindow_and_the_shell_use(make_editor):
    editor = make_editor(show=False)
    chrome, controls = editor.chrome, editor.controls

    assert editor.EDITOR_NOUN == "mask, skeleton and class labels"
    assert editor.SHORTCUT_SCOPE == shortcuts.CORRECT
    assert isinstance(editor.graphics_view, EditorCanvasView)
    assert editor.scene.parent() is editor.graphics_view  # a parented scene
    assert isinstance(editor.dock, FloatingDock) and editor.dock is chrome.dock
    assert isinstance(editor.tool_rail, ToolRail) and editor.tool_rail is chrome.rail
    assert isinstance(editor.enhance_popover, EnhancePopover)
    assert editor.enhance_popover is chrome.popover
    assert isinstance(editor.zoom_cluster, ZoomCluster) and editor.zoom_cluster is chrome.zoom
    assert editor.status_chip is chrome.chip
    for name in ("save_button", "undo_button", "redo_button"):
        assert getattr(editor, name) is getattr(controls, name)
    assert isinstance(editor.layers[MASK], MaskLayer)
    assert isinstance(editor.layers[SKELETON], SkeletonLayer)
    assert list(editor.layers) == [MASK, SKELETON, CLASSES]
    assert editor.session.layers == editor.layers
    assert set(editor.layer_panel.rows) == {MASK, SKELETON, CLASSES}
    assert editor.photo_item.scene() is editor.scene and editor.photo_item.zValue() == 0
    for layer in editor.layers.values():
        assert layer.item.scene() is editor.scene


RAIL_TIPS = {
    (MASK, "brush"): "Mask brush (B) · hold B + scroll to resize",
    (MASK, "eraser"): "Mask eraser (E) · [ and ] resize",
    (MASK, "fill"): "Fill a closed outline in the mask (F)",
    (SKELETON, "select"): "Select (V) · pick a branch, Delete removes it",
    (SKELETON, "eraser"): "Skeleton eraser (E) · [ and ] resize",
    (SKELETON, "polyline"): "Polyline (P) · Enter finish · Esc cancel",
    (SKELETON, "connect"): "Connect endpoints (C)",
    (CLASSES, "brush"): "Class labels brush (B) · hold B + scroll to resize",
    (CLASSES, "eraser"): "Class labels eraser (E) · [ and ] resize",
    (CLASSES, "fill"): "Fill a closed outline in the class labels (F)",
}


def test_the_rail_and_the_dock_name_each_tool_with_its_key(make_editor):
    editor = make_editor(show=False)

    assert {key: b.toolTip() for key, b in editor.controls.tool_buttons.items()} == RAIL_TIPS
    assert "Space" in editor.chrome.pan_button.toolTip()
    assert "(S)" in editor.controls.smooth_button.toolTip()
    assert platform_keys("Ctrl+Y") in editor.redo_button.toolTip()
    assert platform_keys("Ctrl+Z") in editor.undo_button.toolTip()
    assert editor.enhance_button is editor.chrome.enhance_button
    assert editor.pan_button is editor.chrome.pan_button


def test_a_new_page_has_no_image_and_edits_the_mask(make_editor):
    editor = make_editor()

    assert editor.current_image_path == "" and editor.active_layer_id == MASK
    assert editor.layers[MASK].current_tool == "brush"
    assert editor.layers[SKELETON].current_tool == "select"
    assert editor.status_chip.text() == "No image" and editor.tool_rail.width() <= 60
    assert not editor.has_unsaved_changes() and editor.maybe_save_changes() is True
    assert editor.save() is False  # nothing to save
    editor.undo()
    editor.redo()
    drag(editor, (10, 10), (40, 30))  # nothing open: no input is taken
    assert editor.session.journal.undo_entries == ()


# ================================================================================ opening
def test_load_image_opens_both_layers_and_fits_the_photo(make_editor, project):
    editor = make_editor(project.a, project.folder)
    view = editor.graphics_view

    assert same_path(editor.current_image_path, project.a)
    assert np.array_equal(plane(editor), box_plane())
    assert editor.layers[SKELETON].document.source == "prediction"
    assert raster(editor)[110, 20:180].all()
    assert editor.photo_item.image_size() == (W, H)
    assert editor.status_chip.text() == "a.png" and not editor.has_unsaved_changes()
    fitted = view.zoom_factor
    assert fitted > 2.0  # a 200 px photo in a 1100 px page
    view.fit_to_view()
    assert view.zoom_factor == pytest.approx(fitted)


def test_load_image_decodes_the_photo_once(make_editor, project, monkeypatch):
    import app.enhancement as enhancement

    real, decoded = enhancement.imread_unicode, []

    def spy(path, *args, **kwargs):
        decoded.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(enhancement, "imread_unicode", spy)
    editor = make_editor(project.a, project.folder)
    assert len(decoded) == 1 and same_path(decoded[0], project.a)

    editor.apply_normalization()  # the Enhance render reuses the decoded photo
    editor.load_image(project.b, project.folder)
    assert len(decoded) == 2 and same_path(decoded[1], project.b)


def test_an_undecodable_image_is_reported_and_nothing_changes(make_editor, project, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (60, 100))  # unsaved: yet no prompt for an image that cannot open
    edited = plane(editor).copy()
    notices = record(editor.notice)

    editor.load_image(str(bad), str(tmp_path))

    assert notices == [("Could not open bad.png", "danger")]
    assert same_path(editor.current_image_path, project.a)
    assert np.array_equal(plane(editor), edited) and editor.has_unsaved_changes()


def test_an_undecodable_first_image_leaves_the_page_empty(make_editor, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    editor = make_editor()
    notices = record(editor.notice)

    editor.load_image(str(bad), str(tmp_path))

    assert [kind for _message, kind in notices] == ["danger"]
    assert editor.current_image_path == "" and not editor.layers[MASK].is_open()


def test_the_gt_skeleton_wins_over_the_prediction(make_editor, project):
    write_skeleton(project.correction(), row=40)
    editor = make_editor(project.a, project.folder)

    assert editor.layers[SKELETON].document.source == "gt"
    assert raster(editor)[40].any() and not raster(editor)[110].any()


def test_the_injected_resolver_finds_the_generated_skeleton(make_editor, project):
    elsewhere = write_skeleton(os.path.join(project.folder, "out", "b_fake.png"), row=60)
    editor = make_editor()
    editor.set_generated_skeleton_resolver(lambda image_path: elsewhere)

    editor.load_image(project.b, project.folder)

    assert editor.layers[SKELETON].document.source == "prediction" and raster(editor)[60].any()


def test_an_image_without_mask_or_skeleton_says_so(make_editor, project):
    editor = make_editor(project.b, project.folder)
    rows = editor.layer_panel.rows

    assert rows[MASK].state_label.text() == "No mask"
    assert rows[SKELETON].state_label.text() == "No skeleton"
    editor.set_active_layer(SKELETON)
    assert "No skeleton yet" in editor.status_chip.hint_label.text()


# ================================================================================= re-entry
def centre_of(view) -> QPointF:
    return view.mapToScene(view.viewport().rect().center())


def test_the_same_image_with_unsaved_edits_keeps_them_and_their_history(make_editor, project):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (60, 100))
    edited = plane(editor).copy()

    editor.load_image(project.a, project.folder)  # the guard fails on any prompt

    assert np.array_equal(plane(editor), edited) and editor.session.can_undo()


def test_the_same_image_with_unsaved_edits_is_kept_when_its_files_change(make_editor, project):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (60, 100))
    edited = plane(editor).copy()
    write_skeleton(project.generated(), row=30)  # a Generate run meanwhile
    bump(project.generated())

    editor.load_image(project.a, project.folder)

    assert np.array_equal(plane(editor), edited) and editor.session.can_undo()
    assert raster(editor)[110].any() and not raster(editor)[30].any()


def test_the_same_clean_image_keeps_documents_history_and_view(make_editor, project, monkeypatch):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (60, 100))
    editor.undo()  # clean again, with a step to redo
    view = editor.graphics_view
    view.set_zoom(3.0)
    view.centerOn(QPointF(50, 40))
    centre = centre_of(view)
    opened = count_opens(editor, monkeypatch)

    editor.load_image(project.a, project.folder)

    assert opened == [] and editor.session.can_redo()
    assert view.zoom_factor == pytest.approx(3.0) and centre_of(view) == centre


def test_a_changed_mask_file_reloads_the_documents_in_the_same_view(make_editor, project):
    editor = make_editor(project.a, project.folder)
    view = editor.graphics_view
    view.set_zoom(3.0)
    view.centerOn(QPointF(50, 40))
    centre = centre_of(view)
    other = np.zeros((H, W), np.uint8)
    other[5:25, 5:45] = 1
    write_mask(project.mask(), other)
    bump(project.mask())

    editor.load_image(project.a, project.folder)

    assert np.array_equal(plane(editor), other) and np.array_equal(alpha(editor) > 0, other > 0)
    assert view.zoom_factor == pytest.approx(3.0) and centre_of(view) == centre


def test_a_rewritten_generated_skeleton_reloads_the_skeleton(make_editor, project):
    editor = make_editor(project.a, project.folder)
    write_skeleton(project.generated(), row=30)  # a Generate run
    bump(project.generated())

    editor.load_image(project.a, project.folder)

    assert raster(editor)[30].any() and not raster(editor)[110].any()


def test_a_changed_photo_opens_fitted_again(make_editor, project):
    editor = make_editor(project.a, project.folder)
    view = editor.graphics_view
    view.set_zoom(7.0)
    write_image(project.a, value=120)
    bump(project.a)

    editor.load_image(project.a, project.folder)

    fitted = view.zoom_factor
    assert fitted != pytest.approx(7.0)
    view.fit_to_view()
    assert view.zoom_factor == pytest.approx(fitted)


def test_another_image_opens_fitted(make_editor, project):
    editor = make_editor(project.a, project.folder)
    view = editor.graphics_view
    fitted = view.zoom_factor
    view.set_zoom(7.0)

    editor.load_image(project.b, project.folder)

    assert view.zoom_factor == pytest.approx(fitted, rel=0.05)
    assert editor.status_chip.text() == "b.png"


def test_re_entry_after_our_own_skeleton_save_reloads_the_saved_correction(make_editor, project):
    # A generated skeleton at half size: the correction is saved at image resolution, so
    # re-entry reads back exactly the raster that was in memory (Q1 fixed).
    write_skeleton(project.generated(), row=55, w=W // 2, h=H // 2)
    editor = make_editor(project.a, project.folder)
    draw_line(editor)
    assert editor.save()
    in_memory = raster(editor).copy()

    editor.load_image(project.a, project.folder)  # leaving and re-entering Correct

    saved = cv2.imread(project.correction(), cv2.IMREAD_GRAYSCALE)
    fresh = SkeletonCorrectionModel()
    fresh.load_from_raster(saved, target_size=(W, H))
    assert np.array_equal(raster(editor), fresh.mask)
    assert np.array_equal(raster(editor), in_memory)
    assert editor.layers[SKELETON].document.source == "gt"


# ============================================================================ the two layers
def test_set_active_layer_switches_the_edited_layer_once(make_editor, project):
    editor = make_editor(project.a, project.folder)
    switched = record(editor.active_layer_changed)

    assert editor.set_active_layer(SKELETON) is True
    assert editor.set_active_layer(SKELETON) is False
    assert switched == [SKELETON] and editor.active_layer_id == SKELETON
    assert editor.layers[SKELETON].is_active() and not editor.layers[MASK].is_active()
    rows = editor.layer_panel.rows
    assert rows[SKELETON].is_active() and not rows[MASK].is_active()


def test_a_rail_tool_of_the_other_layer_switches_to_it_with_that_tool(make_editor, project):
    editor = make_editor(project.a, project.folder)
    buttons = editor.controls.tool_buttons

    buttons[(SKELETON, "connect")].click()
    assert (editor.active_layer_id, editor.layers[SKELETON].current_tool) == (SKELETON, "connect")
    buttons[(MASK, "fill")].click()
    assert (editor.active_layer_id, editor.layers[MASK].current_tool) == (MASK, "fill")
    assert buttons[(MASK, "fill")].isChecked()


def test_a_click_on_a_layers_row_edits_that_layer(make_editor, project):
    editor = make_editor(project.a, project.folder)

    QTest.mouseClick(editor.layer_panel.rows[SKELETON], LEFT)

    assert editor.active_layer_id == SKELETON
    assert editor.layer_panel.rows[SKELETON].is_active()


def test_a_switch_mid_drag_commits_the_stroke_and_the_rest_paints_nothing(make_editor, project):
    editor = make_editor(project.b, project.folder)
    router, event = editor.router, Mouse()
    router.canvas_press(QPoint(20, 20), event)
    router.canvas_move(QPoint(60, 20), event, True)

    QTest.mouseClick(editor.layer_panel.rows[SKELETON], LEFT)  # the skeleton row, mid-drag
    router.canvas_move(QPoint(60, 90), event, True)
    router.canvas_release(QPoint(60, 90), event)

    assert np.array_equal(
        plane(editor), stroke_plane(np.zeros((H, W), np.uint8), [(20, 20), (60, 20)])
    )
    assert editor.session.journal.undo_entries == (MASK,)
    assert not raster(editor).any()


def test_a_polyline_draft_survives_a_visit_to_the_mask(make_editor, project):
    editor = make_editor(project.a, project.folder)
    skeleton = editor.layers[SKELETON]
    editor.controls.tool_buttons[(SKELETON, "polyline")].click()
    click(editor, (40, 125))
    click(editor, (100, 130))

    editor.set_active_layer(MASK)
    assert skeleton.draft_state() == DraftState(True, 2, False)
    assert not skeleton.decorations.polyline_item.isVisible()
    assert editor.controls.prompt.isHidden()

    editor.set_active_layer(SKELETON)
    assert skeleton.decorations.polyline_item.isVisible()
    assert not editor.controls.prompt.isHidden()
    click(editor, (160, 135))
    editor.controls.finish_button.click()
    assert editor.session.journal.undo_entries == (SKELETON,)
    assert skeleton.draft_state().points == 0


def test_the_tool_of_the_edited_layer_asked_for_again_starts_over(make_editor, project):
    editor = make_editor(project.a, project.folder)
    skeleton, polyline = (
        editor.layers[SKELETON],
        editor.controls.tool_buttons[(SKELETON, "polyline")],
    )
    polyline.click()
    click(editor, (40, 125))
    click(editor, (100, 130))

    polyline.click()  # as today's rail and P key: the tool is taken again, the draft goes

    assert skeleton.draft_state() == DraftState(True, 0, False) and polyline.isChecked()
    assert editor.session.journal.undo_entries == ()


def test_the_other_layer_is_drawn_at_half_its_opacity(make_editor, project):
    editor = make_editor(project.a, project.folder)
    mask, skeleton = editor.layers[MASK].item, editor.layers[SKELETON].item

    assert (mask.opacity(), skeleton.opacity()) == pytest.approx((0.60, 0.425))
    editor.set_active_layer(SKELETON)
    assert (mask.opacity(), skeleton.opacity()) == pytest.approx((0.30, 0.85))


def test_a_layers_eye_hides_and_shows_its_layer_without_activating_it(make_editor, project):
    editor = make_editor(project.a, project.folder)
    skeleton, eye = editor.layers[SKELETON], editor.layer_panel.rows[SKELETON].eye_button

    eye.click()
    assert not skeleton.is_visible() and not skeleton.item.isVisible()
    assert editor.active_layer_id == MASK
    eye.click()
    assert skeleton.is_visible() and skeleton.item.isVisible()


def test_editing_a_hidden_active_layer_shows_it_first_with_a_toast(make_editor, project):
    editor = make_editor(project.a, project.folder)
    mask = editor.layers[MASK]
    editor.layer_panel.rows[MASK].eye_button.click()
    assert not mask.is_visible()
    notices = record(editor.notice)

    paint(editor, (20, 100), (60, 100))

    assert mask.is_visible() and mask.item.isVisible()
    assert editor.layer_panel.rows[MASK].eye_button.isChecked()
    assert notices == [("Mask shown", "info")] and plane(editor)[100, 40] == 1


def test_clearing_a_hidden_layer_shows_it_first(make_editor, project):
    editor = make_editor(project.a, project.folder)
    mask = editor.layers[MASK]
    editor.layer_panel.rows[MASK].eye_button.click()
    notices = record(editor.notice)

    editor.clear_active_layer()

    assert mask.is_visible() and editor.layer_panel.rows[MASK].eye_button.isChecked()
    assert notices[0] == ("Mask shown", "info") and not plane(editor).any()


def test_the_layer_settings_stay_for_the_next_image(make_editor, project):
    editor = make_editor(project.b, project.folder)
    editor.controls.opacity_slider.slider.setValue(30)  # the mask, 30 %
    editor.controls.tool_buttons[(SKELETON, "eraser")].click()
    editor.layers[SKELETON].set_size(30)
    editor.layer_panel.rows[MASK].eye_button.click()

    editor.load_image(project.a, project.folder)  # a has a skeleton: the eraser hint shows

    mask, skeleton = editor.layers[MASK], editor.layers[SKELETON]
    assert editor.active_layer_id == SKELETON and skeleton.current_tool == "eraser"
    assert skeleton.size() == 30 and mask.opacity() == pytest.approx(0.30)
    assert not mask.is_visible() and not editor.layer_panel.rows[MASK].eye_button.isChecked()
    assert editor.layer_panel.rows[SKELETON].is_active()
    assert editor.controls.tool_buttons[(SKELETON, "eraser")].isChecked()
    assert editor.status_chip.hint_label.text().startswith("Skeleton · Eraser")


def test_endpoint_markers_show_only_while_the_skeleton_is_edited(make_editor, project):
    editor = make_editor(project.a, project.folder)
    markers = editor.layers[SKELETON].decorations.endpoints_item

    assert not markers.path().isEmpty() and not markers.isVisible()
    editor.set_active_layer(SKELETON)
    assert markers.isVisible()
    editor.set_active_layer(MASK)
    assert not markers.isVisible()
