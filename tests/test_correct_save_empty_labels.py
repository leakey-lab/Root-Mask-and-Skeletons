"""Mark empty on the Correct page: empty class labels (owner decision 2026-11-02: "yes
empty label file is necessary").

Since the class tick became the active class (2026-11-01), only painted pixels make the
class labels unsaved, so an image that shows none of the classes never got a labels file and
Export dataset left it out. **Mark empty** (the top bar's Correct page, next to Remove GT…)
saves ``gt/<stem>.png`` with nothing labelled (id 0 everywhere): "checked, none of the
classes", a negative example for training. It saves through the page's normal save
(``gt/classes.json`` first when missing, ``gt_saved``, the "GT" state, the toast) and only
the class labels: other layers' edits stay unsaved and do not block it. It is on only while
an image is open with untouched class labels, a valid ``gt/classes.json`` and no labels
file saved or kept for the image, so it never replaces saved labels.
"""

from __future__ import annotations

import csv
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QPushButton

from app.gt import (
    Tracing,
    default_registry,
    load_registry,
    read_tracing,
    registry_path,
    save_registry,
    write_tracing,
)
from app.gt.export import ExportOptions, export_dataset
from app.gui.correct.actions import EMPTY_NO_IMAGE_TIP, EMPTY_TIP
from app.gui.correct.layer import CLASSES, MASK, SKELETON
from app.project_layout import ProjectLayout
from tests.correct_support import H, W, drag, draw_line, new_project, paint, record, settle
from tests.qt_teardown import delete_widgets
from tests.window_support import build_window, open_folder, quiet_window

pytestmark = pytest.mark.usefixtures("correct_guards")


# ============================================================================== helpers
def norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def tree(folder: str) -> dict[str, tuple[int, int]]:
    """Every file under ``folder``: normalised path -> (mtime_ns, size)."""
    files = {}
    for root, _dirs, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            stat = os.stat(path)
            files[norm(path)] = (stat.st_mtime_ns, stat.st_size)
    return files


def changed(before: dict, after: dict) -> set[str]:
    """The files written, made or removed between two :func:`tree` listings."""
    return {path for path in before.keys() | after.keys() if before.get(path) != after.get(path)}


def button(editor) -> QPushButton:
    return editor.actions.empty_button


def row_text(editor, layer_id: str) -> str:
    return editor.layer_panel.rows[layer_id].state_label.text()


def paint_classes(editor, *points) -> None:
    """A stroke of the class labels' brush (they become the edited layer)."""
    editor.set_active_layer(CLASSES)
    drag(editor, *points)


def labels_file(project) -> str:
    return ProjectLayout.for_image(project.a).gt_path("a")


@pytest.fixture
def page(make_editor, project):
    """The page on image a (a mask prediction and a generated skeleton, no GT)."""
    return make_editor(project.a, project.folder)


# ============================================================================ the button
def test_the_button_is_off_with_no_image(make_editor):
    editor = make_editor()

    assert button(editor).text() == "Mark empty"
    assert not button(editor).isEnabled() and button(editor).toolTip() == EMPTY_NO_IMAGE_TIP


def test_the_button_is_on_for_an_untouched_image_without_class_labels(page):
    assert button(page).isEnabled() and button(page).toolTip() == EMPTY_TIP
    assert "none of the classes" in EMPTY_TIP


def test_the_button_leaves_the_keyboard_on_the_canvas_and_tab_reaches_it(page):
    assert button(page).focusPolicy() == Qt.FocusPolicy.TabFocus


def test_the_button_is_on_whichever_layer_is_edited(page):
    for layer_id in (MASK, SKELETON, CLASSES):
        page.set_active_layer(layer_id)
        assert button(page).isEnabled(), layer_id


# ============================================================================== the save
def test_a_click_saves_class_labels_with_nothing_labelled(page, project):
    saved = record(page.gt_saved)
    notices = record(page.notice)
    before = tree(project.folder)
    assert not os.path.exists(registry_path(project.folder))

    button(page).click()

    expected = {norm(labels_file(project)), norm(registry_path(project.folder))}
    assert changed(before, tree(project.folder)) == expected  # no mask or skeleton file
    tracing = read_tracing(labels_file(project))
    assert tracing.ids.shape == (H, W) and not tracing.ids.any()  # id 0 (background) only
    assert load_registry(project.folder) == default_registry()
    assert saved == [project.a]
    assert ("Saved class labels · a.png", "success") in notices
    assert row_text(page, CLASSES) == "GT" and page.status_chip.state() == "clean"
    assert not page.has_unsaved_changes()
    assert not button(page).isEnabled()
    assert "already has class labels" in button(page).toolTip()


def test_the_saved_empty_labels_open_as_saved_class_labels(page, project):
    button(page).click()

    page.load_image(project.b, project.folder)
    page.load_image(project.a, project.folder)

    classes = page.layers[CLASSES]
    assert classes.status().label == "GT" and not classes.is_dirty()
    assert not button(page).isEnabled()


def test_an_existing_gt_classes_json_is_kept(page, project):
    registry = default_registry().add(key="hypha", name="Hypha", color="#3fa7ff")
    save_registry(project.folder, registry)
    page.load_image(project.b, project.folder)
    page.load_image(project.a, project.folder)  # the page reads the new registry
    before = tree(project.folder)

    button(page).click()

    assert changed(before, tree(project.folder)) == {norm(labels_file(project))}
    assert load_registry(project.folder) == registry


def test_a_failed_write_reports_it_and_keeps_the_button_on(page, project, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("app.gt.write_tracing", refuse)
    saved = record(page.gt_saved)
    notices = record(page.notice)

    button(page).click()

    assert saved == [] and not os.path.exists(labels_file(project))
    assert any(kind == "danger" and "Could not save class labels" in text for text, kind in notices)
    assert row_text(page, CLASSES) != "GT"
    assert button(page).isEnabled()  # nothing was saved: the user can try again


def test_a_labels_file_written_since_the_open_is_read_not_overwritten(page, project):
    """Another program saved labels after the open: the save gate reads them instead."""
    plane = np.zeros((H, W), np.uint8)
    plane[10:20, 10:20] = 1
    write_tracing(labels_file(project), Tracing(plane), default_registry())
    with open(labels_file(project), "rb") as fh:
        theirs = fh.read()
    notices = record(page.notice)
    saved = record(page.gt_saved)

    button(page).click()

    with open(labels_file(project), "rb") as fh:
        assert fh.read() == theirs
    assert ("Class labels reloaded: the file changed on disk", "info") in notices
    assert saved == [] and row_text(page, CLASSES) == "GT"
    assert not button(page).isEnabled()


# ============================================================================ when it is off
def test_painted_class_labels_turn_it_off_and_undoing_them_turns_it_on(page):
    paint_classes(page, (20, 100), (80, 110))

    assert not button(page).isEnabled() and "painted class labels" in button(page).toolTip()

    page.undo()

    assert not page.has_unsaved_changes()
    assert button(page).isEnabled() and button(page).toolTip() == EMPTY_TIP


def test_painted_class_labels_are_never_saved_by_it(page, project):
    paint_classes(page, (20, 100), (80, 110))

    page.actions.save_empty_labels()  # the button is off: nothing happens

    assert not os.path.exists(labels_file(project))
    assert page.layers[CLASSES].is_dirty()


def test_an_existing_labels_file_turns_it_off(make_editor, project):
    write_tracing(labels_file(project), Tracing(np.zeros((H, W), np.uint8)), default_registry())
    editor = make_editor(project.a, project.folder)

    assert not button(editor).isEnabled()
    assert button(editor).toolTip() == "This image already has class labels"


def test_an_invalid_registry_turns_it_off(make_editor, project):
    os.makedirs(os.path.join(project.folder, "gt"), exist_ok=True)
    with open(registry_path(project.folder), "w", encoding="utf-8") as fh:
        fh.write("{")
    editor = make_editor(project.a, project.folder)

    assert not button(editor).isEnabled()
    assert "gt/classes.json is invalid" in button(editor).toolTip()

    editor.actions.save_empty_labels()

    assert not os.path.exists(labels_file(project))


@pytest.mark.parametrize("kept", ["unreadable", "size"])
def test_a_kept_labels_file_that_could_not_be_shown_turns_it_off(make_editor, project, kept):
    path = labels_file(project)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if kept == "unreadable":
        with open(path, "wb") as fh:
            fh.write(b"not a png")
    else:
        write_tracing(path, Tracing(np.ones((10, 10), np.uint8)), default_registry())
    with open(path, "rb") as fh:
        original = fh.read()
    editor = make_editor(project.a, project.folder)

    assert not button(editor).isEnabled()
    assert "could not be shown" in button(editor).toolTip()

    editor.actions.save_empty_labels()

    with open(path, "rb") as fh:
        assert fh.read() == original  # never replaced


# ============================================================== the other layers' edits
def test_other_layers_unsaved_edits_stay_unsaved_and_do_not_block_it(page, project):
    layout = ProjectLayout.for_image(project.a)
    paint(page, (20, 100), (60, 100))  # the mask
    page.set_active_layer(SKELETON)
    draw_line(page)
    assert page.session.dirty_layers() == [MASK, SKELETON]
    assert button(page).isEnabled()

    button(page).click()

    assert os.path.isfile(labels_file(project))
    assert not os.path.exists(layout.gt_mask_path("a"))
    assert not os.path.exists(layout.gt_skeleton_path("a"))
    assert page.session.dirty_layers() == [MASK, SKELETON]
    assert page.status_chip.state() == "dirty"  # the unsaved layers still show


# ================================================================ Remove GT… and Export
def test_remove_gt_removes_the_empty_labels_and_turns_it_on_again(page, project):
    button(page).click()
    page.actions.confirm_remove = lambda parent, name, detail: True

    page.remove_gt()

    assert not os.path.exists(labels_file(project))
    assert button(page).isEnabled() and row_text(page, CLASSES) == "No class labels"


def test_export_dataset_exports_the_image_with_empty_labels(page, project, tmp_path):
    button(page).click()
    out = tmp_path / "dataset"
    images = {"a": project.a, "b": project.b}

    result = export_dataset(
        project.folder, images, load_registry(project.folder), ExportOptions(str(out))
    )

    assert result.exported == 1 and not result.errors  # b has no labels file: left out
    mask = cv2.imread(str(out / "masks" / "a.png"), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (H, W) and not mask.any()
    with open(out / "manifest.csv", encoding="utf-8", newline="") as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["stem"] == "a" and row["classes_present"] == ""
    assert int(row["px_background"]) == W * H


# ======================================================================= the main window
@pytest.fixture
def window_with_project(app, tmp_path, no_temp_sweep):
    """A shown, active MainWindow on a project with image a open in the Correct stage."""
    project = new_project(tmp_path)
    window = build_window(tmp_path / "settings.ini")
    made = [window]
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window, 3000)
    open_folder(window, project.folder, 2)
    window.file_list.expandAll()
    item = window.library_index.item_for("a")
    window.file_list.setCurrentItem(item)
    window.on_tree_item_clicked(item, 0)
    settle()
    window.ribbon_buttons["Correct"].click()
    settle()
    yield SimpleNamespace(window=window, project=project, editor=window.correct_editor)
    del window
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def tab_walk(start, steps: int) -> list:
    start.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    seen = [QApplication.focusWidget()]
    for _ in range(steps):
        QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_Tab)
        QApplication.processEvents()
        seen.append(QApplication.focusWidget())
    return seen


def test_the_top_bar_correct_page_holds_it_before_remove_gt(window_with_project):
    window, editor = window_with_project.window, window_with_project.editor
    page = window.action_bar_stack.widget(window._action_bar_pages["Correct"])

    assert window.gt_empty_button is editor.actions.empty_button
    assert window.gt_empty_button.parentWidget() is page
    assert window.gt_empty_button.isVisible() and window.gt_empty_button.isEnabled()


def test_tab_follows_the_correct_page_row(window_with_project):
    window = window_with_project.window

    walk = tab_walk(window.ribbon_buttons["Correct"], 5)

    assert walk == [
        window.ribbon_buttons["Correct"],
        window.gt_classes_button,
        window.gt_export_button,
        window.gt_empty_button,
        window.gt_remove_button,
        window.topbar_load_button,
    ]


def test_a_click_in_the_window_marks_the_image_labelled_in_the_library(window_with_project):
    window, project = window_with_project.window, window_with_project.project
    assert not window.image_manager.has_gt_labels("a")

    window.gt_empty_button.click()
    settle()

    assert os.path.isfile(labels_file(project))
    assert window.image_manager.has_gt_labels("a")
    assert not window.gt_empty_button.isEnabled()
