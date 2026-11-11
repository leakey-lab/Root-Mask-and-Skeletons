"""The Correct page's controls wired to its layers: the rail and the size and opacity
sliders follow the edited layer, the dock keeps its place, the polyline prompt, the
status chip and the Layers rows show the layers' state, Save and Clear follow the
open image and its unsaved state. The controls alone: ``test_correct_controls.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPoint

from app.gui.correct.layer import MASK, SKELETON
from app.gui.correct.mask_layer import MaskLayer
from app.gui.correct.skeleton_layer import DraftState, SkeletonLayer
from app.gui.widgets import (
    tokens,
)
from tests.correct_support import (
    box_plane,
    click,
    draw_line,
    drawn_in,
    new_project,
    paint,
    plane,
    raster,
    settle,
)

pytestmark = pytest.mark.usefixtures("correct_guards")


# ======================================================================= controls and chip
def test_the_rail_fades_the_tools_of_the_other_layer(make_editor, project):
    editor = make_editor(project.a, project.folder)
    buttons = editor.controls.tool_buttons

    assert drawn_in(buttons[(MASK, "eraser")], tokens.TEXT_MUTED)
    assert drawn_in(buttons[(SKELETON, "eraser")], tokens.TEXT_DISABLED)
    editor.set_active_layer(SKELETON)
    assert drawn_in(buttons[(MASK, "eraser")], tokens.TEXT_DISABLED)
    assert drawn_in(buttons[(SKELETON, "eraser")], tokens.TEXT_MUTED)
    assert all(button.isEnabled() for button in buttons.values())


def test_the_size_slider_follows_the_tool_of_the_edited_layer(make_editor, project):
    editor = make_editor(project.a, project.folder)
    controls = editor.controls
    brush, eraser = MaskLayer.SIZE_SPEC, SkeletonLayer.ERASER_SIZE

    def shown():
        return [spec for spec, s in controls.size_sliders.items() if s.isVisibleTo(editor.dock)]

    assert shown() == [brush] and controls.size_sliders[brush].slider.value() == 8
    for key, expected in (
        ((MASK, "eraser"), [brush]),
        ((MASK, "fill"), []),
        ((SKELETON, "select"), []),
        ((SKELETON, "eraser"), [eraser]),
        ((SKELETON, "polyline"), []),
        ((SKELETON, "connect"), []),
    ):
        controls.tool_buttons[key].click()
        assert shown() == expected, key
    controls.tool_buttons[(SKELETON, "eraser")].click()
    assert controls.size_sliders[eraser].slider.value() == 20


def test_the_size_slider_sizes_the_tool_of_the_edited_layer(make_editor, project):
    editor = make_editor(project.a, project.folder)
    controls = editor.controls

    controls.size_sliders[MaskLayer.SIZE_SPEC].slider.setValue(30)
    controls.tool_buttons[(SKELETON, "eraser")].click()
    controls.size_sliders[SkeletonLayer.ERASER_SIZE].slider.setValue(11)

    assert (editor.layers[MASK].size(), editor.layers[SKELETON].size()) == (30, 11)


def test_the_opacity_slider_shows_and_sets_the_edited_layer(make_editor, project):
    editor = make_editor(project.a, project.folder)
    slider = editor.controls.opacity_slider.slider
    mask, skeleton = editor.layers[MASK], editor.layers[SKELETON]

    assert slider.value() == 60
    slider.setValue(30)
    assert mask.opacity() == pytest.approx(0.30) and mask.item.opacity() == pytest.approx(0.30)
    editor.set_active_layer(SKELETON)
    assert slider.value() == 85
    slider.setValue(40)
    assert editor.controls.opacity_slider.value_label.text() == "40%"
    assert skeleton.opacity() == pytest.approx(0.40) and mask.opacity() == pytest.approx(0.30)
    assert mask.item.opacity() == pytest.approx(0.15)


def test_load_skeleton_file_is_offered_only_while_the_skeleton_is_edited(make_editor, project):
    editor = make_editor(project.a, project.folder)
    load = editor.controls.load_button

    assert not load.isEnabled()
    editor.set_active_layer(SKELETON)
    assert load.isEnabled()
    editor.set_active_layer(MASK)
    assert not load.isEnabled() and not load.isHidden()


def test_the_dock_keeps_its_place_across_tools_and_layers(make_editor, project):
    editor = make_editor(project.a, project.folder, size=(1280, 800))
    save = editor.controls.save_button
    places = set()
    for key in editor.controls.tool_buttons:
        editor.controls.tool_buttons[key].click()
        settle(editor)
        places.add((editor.dock.geometry().getRect(), save.mapTo(editor, QPoint(0, 0)).x()))
    editor.controls.tool_buttons[(SKELETON, "eraser")].click()
    editor.controls.size_sliders[SkeletonLayer.ERASER_SIZE].slider.setValue(80)  # a wider value
    settle(editor)
    places.add((editor.dock.geometry().getRect(), save.mapTo(editor, QPoint(0, 0)).x()))

    assert len(places) == 1, places


def test_the_polyline_prompt_shows_only_for_a_skeleton_draft(make_editor, project):
    editor = make_editor(project.a, project.folder)
    prompt, finish = editor.controls.prompt, editor.controls.finish_button
    editor.controls.tool_buttons[(SKELETON, "polyline")].click()
    assert prompt.isHidden()

    click(editor, (40, 125))
    assert not prompt.isHidden() and not finish.isEnabled()
    click(editor, (100, 130))
    assert finish.isEnabled()
    editor.set_active_layer(MASK)
    assert prompt.isHidden()
    editor.set_active_layer(SKELETON)
    assert not prompt.isHidden()
    finish.click()
    assert prompt.isHidden() and editor.session.journal.undo_entries == (SKELETON,)


def test_smooth_and_cancel_in_the_prompt_act_on_the_draft(make_editor, project):
    editor = make_editor(project.a, project.folder)
    skeleton, controls = editor.layers[SKELETON], editor.controls
    before = raster(editor).copy()
    controls.tool_buttons[(SKELETON, "polyline")].click()
    for point in ((40, 125), (100, 140), (160, 125)):
        click(editor, point)

    controls.smooth_button.click()
    assert skeleton.draft_state() == DraftState(True, 3, True)
    controls.cancel_button.click()
    assert skeleton.draft_state().points == 0 and controls.prompt.isHidden()
    assert np.array_equal(raster(editor), before)


def test_the_chip_shows_the_state_of_both_layers(make_editor, project, monkeypatch):
    import app.gui.correct.skeleton_document as skeleton_document

    editor = make_editor(project.b, project.folder)
    chip = editor.status_chip
    assert chip.state() == "none"  # no mask, no skeleton
    editor.load_image(project.a, project.folder)
    assert chip.state() == "none"  # the model's mask and a generated skeleton: nothing saved
    paint(editor, (20, 100), (60, 100))
    assert chip.state() == "dirty"
    editor.undo()
    assert chip.state() == "none"

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(skeleton_document, "imwrite_unicode", fail)
    paint(editor, (20, 100), (60, 100))
    assert editor.save()
    assert chip.state() == "clean"  # a saved (GT) mask and a generated skeleton
    draw_line(editor)
    assert chip.state() == "dirty"  # an unsaved skeleton beside a saved mask
    editor.save()
    assert chip.state() == "error" and "disk full" in chip.state_label.toolTip()
    assert editor.layer_panel.rows[MASK].state_label.text() == "GT"


def test_the_chip_hint_names_the_edited_layer_and_its_tool_in_its_colour(make_editor, project):
    editor = make_editor(project.a, project.folder)
    chip = editor.status_chip

    assert chip.hint_label.text() == "Mask · Brush · drag to paint"
    assert not chip.hint_dot.isHidden() and tokens.MASK in chip.hint_dot.styleSheet()
    assert not editor.dock.isAncestorOf(chip.hint_label)
    editor.controls.tool_buttons[(MASK, "fill")].click()
    assert chip.hint_label.text() == "Mask · Fill · click inside a closed outline"
    editor.set_active_layer(SKELETON)
    assert chip.hint_label.text() == "Skeleton · Select · click a skeleton line to edit it"
    assert tokens.SKEL in chip.hint_dot.styleSheet()


def test_the_layers_rows_show_the_save_state_of_each_layer(make_editor, project):
    editor = make_editor(project.a, project.folder)
    rows = editor.layer_panel.rows

    def states():
        return rows[MASK].state_label.text(), rows[SKELETON].state_label.text()

    assert states() == ("Prediction", "Prediction")
    paint(editor, (20, 100), (60, 100))
    assert states() == ("Unsaved", "Prediction")
    draw_line(editor)
    assert states() == ("Unsaved", "Unsaved")
    editor.save()
    assert states() == ("GT", "GT")


def test_the_dock_clear_button_follows_and_clears_the_edited_layer(make_editor, project):
    editor = make_editor(project.a, project.folder)
    button = editor.controls.clear_icon_button

    assert "mask" in button.toolTip()
    editor.set_active_layer(SKELETON)
    assert "skeleton" in button.toolTip()
    button.click()

    assert not raster(editor).any() and np.array_equal(plane(editor), box_plane())
    assert editor.session.journal.undo_entries == (SKELETON,)


def test_save_is_enabled_only_while_something_is_unsaved(make_editor, project):
    editor = make_editor()  # no image
    assert not editor.save_button.isEnabled() and not editor.controls.clear_icon_button.isEnabled()
    editor.load_image(project.a, project.folder)
    assert not editor.save_button.isEnabled() and editor.controls.clear_icon_button.isEnabled()
    paint(editor, (20, 100), (80, 110))
    assert editor.save_button.isEnabled()
    editor.undo()  # back to the saved state
    assert not editor.save_button.isEnabled()
    paint(editor, (20, 100), (80, 110))
    assert editor.save() is True
    assert not editor.save_button.isEnabled()
    assert editor.reset() is True  # a new folder
    assert not editor.save_button.isEnabled() and not editor.controls.clear_icon_button.isEnabled()


def test_a_polyline_of_two_points_enables_save_and_a_cancel_disables_it(make_editor, project):
    editor = make_editor(project.a, project.folder)
    editor.controls.tool_buttons[(SKELETON, "polyline")].click()
    click(editor, (40, 60))
    assert not editor.save_button.isEnabled()  # one point is not a line yet
    click(editor, (150, 70))
    assert editor.save_button.isEnabled()
    editor.layers[SKELETON].cancel_polyline()
    assert not editor.save_button.isEnabled()


def test_clearing_the_mask_makes_save_available(make_editor, project):
    editor = make_editor(project.a, project.folder)
    editor.clear_active_layer()
    assert editor.save_button.isEnabled()  # an empty mask is a change (O9)


def test_the_dock_keeps_its_width_and_place_in_every_save_state(make_editor, project):
    editor = make_editor(size=(1280, 800))
    save, places = editor.controls.save_button, []

    def place():
        settle(editor)
        return (editor.dock.geometry().getRect(), save.mapTo(editor, QPoint(0, 0)).x())

    places.append(place())  # no image
    editor.load_image(project.a, project.folder)
    places.append(place())  # clean
    paint(editor, (20, 100), (80, 110))
    places.append(place())  # unsaved
    assert len(set(places)) == 1, places


# ------------------------------------------------------------------ the page
def test_the_page_shows_the_library_position_its_provider_gives(make_editor, tmp_path):
    project = new_project(tmp_path)
    editor = make_editor()
    order = [project.a, project.b]

    editor.set_position_provider(lambda path: (order.index(path) + 1, len(order)))
    editor.load_image(project.b, project.folder)

    chip = editor.status_chip
    assert chip.position_label.text() == "Image 2 / 2" and not chip.position_label.isHidden()
    editor.load_image(project.a, project.folder)
    assert chip.position_label.text() == "Image 1 / 2"


def test_without_a_provider_or_a_place_the_page_shows_no_position(make_editor, tmp_path, caplog):
    project = new_project(tmp_path)
    editor = make_editor(project.a, project.folder)
    assert editor.status_chip.position_label.isHidden()

    editor.set_position_provider(lambda path: None)  # not in the library
    assert editor.status_chip.position_label.isHidden()

    def broken(path):
        raise KeyError(path)

    editor.set_position_provider(broken)
    editor.load_image(project.b, project.folder)
    assert editor.status_chip.position_label.isHidden()
    assert "library position" in caplog.text


def test_correct_with_no_image_says_select_an_image(make_editor):
    editor = make_editor()
    assert editor.status_chip.hint_label.text() == "Select an image in the Library"
