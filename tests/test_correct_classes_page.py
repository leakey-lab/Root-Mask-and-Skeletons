"""The class labels as the third layer of the Correct page (WG-T5, Design 3.2 - 3.5).

The page edits the mask, the skeleton and the class labels of an image on one canvas. Here:
the three layers and their rows, the side card that grows with the class panel, the layer keys
(M / L / G, X back to the previous one), the class digits (only on the class labels), B / F and
the wheel on the edited raster layer, each class's own brush size, Shift+H, the dock and the rail
on the class labels, a one-pixel dab, the row's saved state, and the layer's opacity and
visibility kept across pages. Helpers and fixtures: ``tests/correct_classes_support.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtTest import QTest

from app.gt import (
    default_registry,
)
from app.gui import shortcuts
from app.gui.class_panel import ClassPanel
from app.gui.correct.editor import CorrectEditor
from app.gui.correct.layer import CLASSES, MASK, SKELETON
from app.gui.correct.side_panels import SidePanels
from tests.correct_classes_support import (
    box,
    ids,
    paint,
    press,
    row_text,
    tap_b,
)
from tests.correct_classes_support import confirm as confirm  # fixture
from tests.correct_classes_support import layout as layout  # fixture
from tests.correct_classes_support import made as made  # fixture
from tests.correct_classes_support import make_editor as make_editor  # fixture
from tests.correct_classes_support import notices as notices  # fixture
from tests.correct_classes_support import page as page  # fixture
from tests.correct_classes_support import project_bad_registry as project_bad_registry  # fixture
from tests.correct_classes_support import project_lossy as project_lossy  # fixture
from tests.correct_classes_support import prompts as prompts  # fixture
from tests.correct_classes_support import reads as reads  # fixture
from tests.correct_classes_support import signals as signals  # fixture
from tests.correct_classes_support import themed as themed  # fixture
from tests.correct_classes_support import window_with_project as window_with_project  # fixture
from tests.correct_support import (
    activate,
    drag,
    settle,
)
from tests.qt_teardown import delete_widgets

pytestmark = pytest.mark.usefixtures("correct_guards")


def test_the_page_has_three_layers_in_panel_order_and_the_class_panel_shows_only_when_active(
    make_editor, project
):
    editor = make_editor(project.a, project.folder)

    assert list(editor.layers) == [MASK, SKELETON, CLASSES]
    assert list(editor.layer_panel.rows) == [MASK, SKELETON, CLASSES]
    assert list(editor.session.layers) == [MASK, SKELETON, CLASSES]
    assert isinstance(editor.side_panels, SidePanels) and isinstance(editor.class_panel, ClassPanel)
    assert editor.side_panels.isAncestorOf(editor.layer_panel)
    assert editor.side_panels.isAncestorOf(editor.class_panel)
    assert not editor.class_panel.isVisible()
    editor.set_active_layer(CLASSES)
    settle(editor)
    assert editor.class_panel.isVisible()
    assert editor.layer_panel.rows[CLASSES].is_active()
    editor.set_active_layer(SKELETON)
    settle(editor)
    assert not editor.class_panel.isVisible()


def test_the_classes_row_names_its_key_and_layer(make_editor, project):
    row = make_editor(project.a, project.folder).layer_panel.rows[CLASSES]

    assert row.name_label.text() == "Classes" and row.key_label.text() == "G"
    assert row.toolTip() == "Edit the class labels (G)"
    assert "Hide the class labels" in row.eye_button.toolTip()
    assert "H: every layer" in row.eye_button.toolTip()
    assert "Shift+H: the others" in row.eye_button.toolTip()


def test_the_side_card_grows_with_the_class_panel_and_stays_on_the_right(make_editor, project):
    editor = make_editor(project.a, project.folder)
    card = editor.side_panels
    settle(editor)
    short = card.height()

    editor.set_active_layer(CLASSES)
    settle(editor)

    assert card.height() > short + editor.class_panel._header.sizeHint().height()
    assert card.geometry().right() < editor.width()
    assert editor.class_panel.geometry().bottom() <= card.height()
    assert editor.layer_panel.graphicsEffect() is None  # one card, one shadow
    assert editor.class_panel.graphicsEffect() is None and card.graphicsEffect() is not None


def test_keys_m_l_g_switch_layers_and_x_returns_to_the_previous_one(make_editor, project):
    editor = make_editor(project.a, project.folder)
    activate(editor)

    press(editor, "L")
    assert editor.active_layer_id == SKELETON
    press(editor, "G")
    assert editor.active_layer_id == CLASSES
    press(editor, "X")
    assert editor.active_layer_id == SKELETON
    press(editor, "M")
    press(editor, "X")
    assert editor.active_layer_id == SKELETON and editor.previous_layer_id == MASK
    assert shortcuts.find_conflicts() == []


def test_x_on_a_fresh_page_switches_between_the_mask_and_the_skeleton(make_editor, project):
    editor = make_editor(project.a, project.folder)
    activate(editor)
    assert editor.previous_layer_id is None

    press(editor, "X")
    assert editor.active_layer_id == SKELETON
    press(editor, "X")
    assert editor.active_layer_id == MASK


def test_the_correct_rows_name_the_layer_keys():
    rows = {spec.action: spec for spec in shortcuts.specs_for(shortcuts.CORRECT)}

    assert [rows[a].keys for a in ("mask_layer", "skeleton_layer", "classes_layer")] == [
        ("M",),
        ("L",),
        ("G",),
    ]
    assert rows["classes_layer"].description == "Edit the class labels (GT for training)"
    assert rows["swap_layer"].description == "Switch to the previous layer"
    assert rows["select_class"].keys == ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
    assert rows["select_class"].label == "1–9, 0"
    assert rows["select_class"].description == (
        "Pick the n-th class; 0 = Ignore (class labels layer only)"
    )
    assert rows["toggle_layers"].description == "Show or hide every layer (see the photo)"
    assert rows["toggle_reference"].description == "Show or hide the other layers"


def test_digits_pick_classes_only_while_the_classes_layer_is_active(make_editor, project):
    editor = make_editor(project.a, project.folder)
    activate(editor)

    press(editor, "2")
    assert editor.active_layer_id == MASK and editor.layers[CLASSES].active_class() == "root"
    press(editor, "G")
    press(editor, "2")
    assert editor.layers[CLASSES].active_class() == "nodule"
    assert editor.class_panel.active_key() == "nodule"
    press(editor, "0")
    assert editor.layers[CLASSES].active_class() == "ignore"
    press(editor, "9")  # no ninth class: nothing changes
    assert editor.layers[CLASSES].active_class() == "ignore"


def test_keypad_digits_pick_classes_too(page):
    keypad = Qt.KeyboardModifier.KeypadModifier
    press(page, "2", keypad)
    assert page.class_panel.active_key() == "nodule"
    press(page, "0", keypad)
    assert page.class_panel.active_key() == "ignore"
    press(page, "3", keypad)
    assert page.class_panel.active_key() == "root_tip"


def test_b_tap_requests_the_brush_of_the_active_raster_layer(make_editor, project):
    editor = make_editor(project.a, project.folder)
    activate(editor)
    editor.set_active_layer(CLASSES)
    editor.request_tool(CLASSES, "eraser")

    tap_b(editor)
    assert (editor.active_layer_id, editor.layers[CLASSES].current_tool) == (CLASSES, "brush")
    editor.set_active_layer(SKELETON)
    tap_b(editor)
    assert editor.active_layer_id == MASK and editor.layers[MASK].current_tool == "brush"


def test_f_takes_the_fill_of_the_classes_layer_and_the_masks_from_the_skeleton(
    make_editor, project
):
    editor = make_editor(project.a, project.folder)
    activate(editor)

    press(editor, "G")
    press(editor, "F")
    assert (editor.active_layer_id, editor.layers[CLASSES].current_tool) == (CLASSES, "fill")
    press(editor, "L")
    press(editor, "F")
    assert (editor.active_layer_id, editor.layers[MASK].current_tool) == (MASK, "fill")


def test_b_and_the_wheel_size_the_brush_of_the_classes_layer(page):
    from tests.correct_support import wheel

    mask_size = page.layers[MASK].size()
    QTest.keyPress(page.graphics_view, Qt.Key.Key_B)
    wheel(page.graphics_view, 120)
    QTest.keyRelease(page.graphics_view, Qt.Key.Key_B)

    assert page.active_layer_id == CLASSES and page.layers[CLASSES].size() == 9
    assert page.layers[MASK].size() == mask_size


def test_each_class_keeps_its_own_brush_size_on_the_dock(page):
    page.controls.size_changed.emit(20)  # root
    press(page, "2")
    assert page.layers[CLASSES].size() == 8
    page.controls.size_changed.emit(30)
    press(page, "1")
    slider = next(s for s in page.controls.size_sliders.values() if s.isVisibleTo(page.dock))
    assert page.layers[CLASSES].size() == 20 and slider.slider.value() == 20


def test_shift_h_hides_and_shows_every_layer_but_the_edited_one(page):
    page.toggle_reference()
    assert [page.layers[i].is_visible() for i in (MASK, SKELETON, CLASSES)] == [False, False, True]
    page.toggle_reference()
    assert all(layer.is_visible() for layer in page.layers.values())
    page.set_layer_visible(MASK, False)
    page.toggle_reference()  # one other layer is shown: hide the others
    assert not page.layers[SKELETON].is_visible() and page.layers[CLASSES].is_visible()


def test_the_controls_on_the_classes_layer(page):
    controls = page.controls

    assert not controls.load_button.isEnabled()
    assert not controls.prediction_button.isEnabled()
    assert controls.prediction_button.toolTip() == "Load prediction (not for the class labels)"
    assert controls.clear_icon_button.toolTip() == (
        "Clear class labels (undoable) · Save then writes an empty file"
    )
    assert controls.tool_buttons[(CLASSES, "fill")].toolTip() == (
        "Fill a closed outline in the class labels (F)"
    )
    assert controls.tool_buttons[(MASK, "brush")].toolTip() == (
        "Mask brush (B) · hold B + scroll to resize"
    )
    assert controls.tool_buttons[(CLASSES, "brush")].toolTip() == (
        "Class labels brush (B) · hold B + scroll to resize"
    )


def test_the_class_labels_tools_show_on_the_rail_only_while_edited(make_editor, project):
    """As the class panel: three more rail tools for every user did not fit beside the
    polyline prompt in a 700 x 520 canvas (``test_the_overlays_never_collide``)."""
    editor = make_editor(project.a, project.folder)
    buttons = editor.controls.tool_buttons
    classes = [buttons[(CLASSES, tool)] for tool in ("brush", "eraser", "fill")]
    assert not any(button.isVisible() for button in classes)
    assert buttons[(SKELETON, "select")].isVisible()  # another layer's tools stay, dimmed

    editor.set_active_layer(CLASSES)
    settle(editor)
    assert all(button.isVisible() for button in classes)
    assert buttons[(CLASSES, "brush")].isChecked()

    editor.request_tool(MASK, "fill")
    settle(editor)
    assert not any(button.isVisible() for button in classes)
    assert editor.tool_rail.geometry().bottom() < editor.height()


def test_the_tool_hints_name_the_edited_layer(page):
    """The Brush and Eraser hints said "the mask" on every raster layer (WG-T4 review)."""
    chip = page.status_chip
    assert chip.hint_label.text() == "Class labels · Brush · drag to paint"
    assert (
        page.layers[CLASSES].tool_hint()[1].startswith("Click and drag to paint the class labels")
    )
    page.request_tool(CLASSES, "eraser")
    assert (
        page.layers[CLASSES].tool_hint()[1].startswith("Click and drag to erase the class labels")
    )
    page.set_active_layer(MASK)
    assert page.layers[MASK].tool_hint()[1].startswith("Click and drag to paint the mask")


def test_a_one_pixel_click_paints_its_class_pixel(page):
    page.controls.size_changed.emit(1)
    drag(page, (30, 20))

    assert np.count_nonzero(ids(page)) == 1
    assert ids(page)[20, 30] == default_registry().by_key("root").id


def test_the_classes_row_says_what_is_saved(page):
    assert row_text(page, CLASSES) == "No class labels"
    paint(page, (20, 100), (80, 110))
    assert row_text(page, CLASSES) == "Unsaved"
    assert page.save()
    assert row_text(page, CLASSES) == "GT"  # as the mask and the skeleton
    box(page, "nodule").click()  # picks a class: nothing new to save
    assert row_text(page, CLASSES) == "GT" and not page.has_unsaved_changes()


def test_opacity_and_visibility_of_the_classes_layer_persist(app, tmp_path):
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    made = []
    try:
        first = CorrectEditor(settings=settings)
        made.append(first)
        first.show()
        first.layers[CLASSES].set_opacity(0.3)
        first.set_layer_visible(CLASSES, False)
        first.hide()  # the page stores its layers' prefs when it is hidden
        assert float(settings.value("correct/classes/opacity")) == pytest.approx(0.3)

        second = CorrectEditor(settings=settings)
        made.append(second)

        assert second.layers[CLASSES].opacity() == pytest.approx(0.3)
        assert not second.layers[CLASSES].is_visible()
        assert not second.layer_panel.rows[CLASSES].eye_button.isChecked()
    finally:
        delete_widgets(made, prepare=lambda e: e.layers[SKELETON].cancel_scheduled_render())
