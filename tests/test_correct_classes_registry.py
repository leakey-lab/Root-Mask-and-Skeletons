"""The project's classes on the Correct page: an invalid ``gt/classes.json`` and Classes….

While ``gt/classes.json`` is invalid the class labels alone are read-only: edits are refused
with one danger notice per image, the row and (while edited) the chip say "Read-only", the other
layers' states still show, a box click only picks a class, and a fixed file is read again on
reselect. Classes… (the gear or the top bar) saves the registry, recolours the labels (ids and
history kept), counts as no change on disk, applies a restored class, is deleted after use, and
the class panel follows the new classes with one box ticked.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import (
    QApplication,
)

from app.gt import (
    default_registry,
    load_registry,
    read_tracing,
    registry_path,
    save_registry,
)
from app.gui.class_editor_dialog import ClassEditorDialog
from app.gui.correct.layer import CLASSES, MASK, SKELETON
from app.gui.paint_canvas import pixmap_to_argb_array
from tests.correct_classes_support import (
    assert_one_tick,
    box,
    fake_class_editor,
    ids,
    paint,
    press,
    ticked,
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
    settle,
)

pytestmark = pytest.mark.usefixtures("correct_guards")


def test_an_invalid_registry_locks_only_the_classes_layer(
    make_editor, project_bad_registry, notices
):
    folder, a = project_bad_registry.folder, project_bad_registry.a
    editor = make_editor(a, folder)
    paint(editor, (20, 100), (60, 100))  # the mask
    assert editor.save() is True

    editor.set_active_layer(CLASSES)
    paint(editor, (20, 100), (80, 110))
    paint(editor, (20, 120), (80, 120))
    editor.clear_active_layer()
    box(editor, "nodule").click()  # picks the class: no edit, so not refused
    editor.edit_classes()  # refused too, without a dialog (the guard fails on one)

    classes = editor.layers[CLASSES]
    assert classes.status().label == "Read-only" and not classes.is_dirty()
    assert not ids(editor).any() and ticked(editor) == ["nodule"]
    refusals = [n for n in notices if n[0].startswith("Editing is off: gt/classes.json is invalid")]
    assert len(refusals) == 1 and refusals[0][1] == "danger"
    # One danger notice per image: the open's read-only notice is not shown as well.
    assert [kind for _message, kind in notices].count("danger") == 1
    assert not [m for m, _k in notices if m.startswith("gt/classes.json is invalid")]


def test_an_invalid_registry_shows_read_only_on_the_row_and_on_the_chip_while_edited(
    make_editor, project_bad_registry
):
    """The lock is the class labels' own (``LayerStatus.local``): the chip says "Read-only"
    only while they are edited; the other layers' states always show (review WG-T5 r1)."""
    editor = make_editor(project_bad_registry.a, project_bad_registry.folder)
    row, chip = editor.layer_panel.rows[CLASSES], editor.status_chip
    assert row.state_label.text() == "Read-only"
    assert "Expecting property name" in row.state_label.toolTip()
    assert str(project_bad_registry.folder) not in row.state_label.toolTip()
    assert editor.layers[CLASSES].status().local
    assert chip.state() == "none" and "Read-only" not in chip.state_label.text()

    editor.set_active_layer(CLASSES)
    assert chip.state() == "error" and "Read-only" in chip.state_label.text()
    assert "gt/classes.json is invalid" in chip.state_label.toolTip()

    editor.set_active_layer(MASK)
    assert chip.state() == "none"


def test_a_read_only_class_labels_layer_never_hides_another_layers_unsaved_edits(
    make_editor, project_bad_registry
):
    editor = make_editor(project_bad_registry.a, project_bad_registry.folder)
    chip = editor.status_chip

    paint(editor, (20, 100), (60, 100))  # the mask
    assert chip.state() == "dirty" and editor.layer_panel.rows[MASK].state_label.text() == "Unsaved"
    editor.set_active_layer(SKELETON)
    assert chip.state() == "dirty"  # another layer's unsaved edits still count
    editor.set_active_layer(CLASSES)
    assert chip.state() == "error" and "Read-only" in chip.state_label.text()
    assert editor.layer_panel.rows[CLASSES].state_label.text() == "Read-only"


def test_a_box_click_while_read_only_picks_the_class_and_edits_nothing(
    make_editor, project_bad_registry
):
    editor = make_editor(project_bad_registry.a, project_bad_registry.folder)
    editor.set_active_layer(CLASSES)

    box(editor, "nodule").click()

    assert_one_tick(editor, "nodule")
    assert not editor.layers[CLASSES].is_dirty() and not ids(editor).any()


def test_an_invalid_registry_is_read_again_on_reselect(make_editor, project_bad_registry):
    folder, a = project_bad_registry.folder, project_bad_registry.a
    editor = make_editor(a, folder)
    assert editor.layers[CLASSES].is_read_only()

    save_registry(folder, default_registry())  # fixed
    editor.load_image(a, folder)

    assert not editor.layers[CLASSES].is_read_only()
    editor.set_active_layer(CLASSES)
    paint(editor, (20, 100), (80, 110))
    assert editor.save()


def test_edit_classes_saves_the_registry_and_recolours(page, project, monkeypatch, notices):
    paint(page, (10, 30), (70, 30))  # root
    press(page, "2")
    paint(page, (40, 5), (40, 55))  # nodule
    before = ids(page).copy()
    layer = page.layers[CLASSES]
    new_registry = layer.registry().update("root", color="#123456", name="Main root")
    seen: list = []
    fake_class_editor(monkeypatch, new_registry, seen)

    page.edit_classes()

    assert seen == [default_registry()] and layer.registry() == new_registry
    assert np.array_equal(ids(page), before)
    argb = pixmap_to_argb_array(layer.canvas.image)
    assert tuple(argb[30, 15, :3]) == (0x56, 0x34, 0x12)  # BGR of #123456
    assert (
        os.path.exists(registry_path(project.folder)) and ("Classes updated", "success") in notices
    )
    assert page.class_panel._rows["root"]._name_label.text() == "Main root"
    page.undo()  # the history holds ids: undoing the nodule stroke keeps the root ids
    assert ids(page)[30, 15] == 1 and ids(page)[30, 40] == 1 and ids(page)[10, 40] == 0
    assert page.save()
    assert read_tracing(page.layers[CLASSES].document.gt_path).ids[30, 15] == 1


def test_the_gear_and_the_topbar_open_the_same_class_editor(page, monkeypatch):
    seen: list = []
    fake_class_editor(monkeypatch, None, seen)

    page.class_panel.edit_requested.emit()

    assert len(seen) == 1


def test_classes_saved_by_the_page_are_no_change_on_disk(page, project, monkeypatch, reads):
    """The registry write re-stamps the class labels: re-selecting the image keeps them,
    their history and asks nothing (WG-T4 review, carried concern 2)."""
    paint(page, (10, 30), (70, 30))
    assert page.save()
    paint(page, (10, 40), (70, 40))
    page.undo()  # back to the saved labels, with one undo and one redo
    bigger = page.layers[CLASSES].registry().add(key="hypha", name="Hypha", color="#3fa7ff")
    fake_class_editor(monkeypatch, bigger)
    page.edit_classes()
    reads.clear()

    page.load_image(project.a, project.folder)  # the guard fails on any prompt

    assert reads == [] and not page.session.files_changed()
    assert page.layers[CLASSES].can_undo() and page.layers[CLASSES].can_redo()


def test_a_restored_class_is_applied_saved_and_no_change_on_disk(
    make_editor, project, monkeypatch, notices
):
    """WG-T9: Restore in the class editor ends in the page's usual Classes… path."""
    retired = default_registry().retire("nodule")
    save_registry(project.folder, retired)
    editor = make_editor(project.a, project.folder)
    layer = editor.layers[CLASSES]
    assert layer.registry().by_key("nodule").retired
    fake_class_editor(monkeypatch, retired.restore("nodule"))

    editor.edit_classes()

    assert layer.registry().by_key("nodule").retired is False
    assert load_registry(project.folder).by_key("nodule").retired is False
    assert not editor.session.files_changed()
    assert ("Classes updated", "success") in notices
    assert "nodule" in editor.class_panel._rows


def test_classes_without_an_image_says_where_classes_are_kept(make_editor, notices):
    editor = make_editor()

    editor.edit_classes()  # the guard fails on any dialog

    assert notices == [("Open an image first; classes are saved with its folder.", "info")]


def test_the_class_editor_dialog_is_deleted_after_use(page, monkeypatch):
    from PyQt6 import sip

    opened = []
    monkeypatch.setattr(ClassEditorDialog, "exec", lambda dialog: opened.append(dialog) or 0)

    page.edit_classes()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert len(opened) == 1 and sip.isdeleted(opened[0])


def test_the_class_panel_resizes_after_classes_are_added(page, monkeypatch):
    bigger = (
        page.layers[CLASSES]
        .registry()
        .add(key="dead_root", name="Dead root", color="#7f5af0")
        .add(key="hypha", name="Hypha", color="#3fa7ff")
    )
    fake_class_editor(monkeypatch, bigger)

    page.edit_classes()
    settle(page)

    panel = page.class_panel
    assert set(panel._rows) == {c.key for c in bigger.active()}
    assert panel._scroll.height() >= panel._rows_container.sizeHint().height()
    assert page.side_panels.geometry().right() < page.width()


def test_one_box_stays_ticked_when_the_classes_change(page, monkeypatch):
    press(page, "2")
    bigger = page.layers[CLASSES].registry().add(key="hypha", name="Hypha", color="#3fa7ff")
    fake_class_editor(monkeypatch, bigger)
    page.edit_classes()
    QApplication.processEvents()
    assert_one_tick(page, "nodule")  # a new class: the active one keeps its tick

    fake_class_editor(monkeypatch, bigger.retire("nodule"))
    page.edit_classes()
    QApplication.processEvents()
    assert_one_tick(page, "root")  # the active class retired: the first class takes the tick
    assert "nodule" not in page.class_panel._rows

    press(page, "3")  # root, root_tip, hypha
    assert_one_tick(page, "hypha")


def test_classes_that_cannot_be_saved_change_nothing_and_say_why(page, monkeypatch, notices):
    """Classes… when ``gt/classes.json`` cannot be written (a locked or read-only file): one
    danger notice names it, the labels keep their classes and colours, and nothing counts as
    changed on disk."""
    import app.gt as gt_package

    def denied(project_dir, registry):
        raise PermissionError(13, "Permission denied", "gt/classes.json")

    layer = page.layers[CLASSES]
    before = layer.registry()
    fake_class_editor(monkeypatch, before.update("root", color="#123456"))
    monkeypatch.setattr(gt_package, "save_registry", denied)
    notices.clear()

    page.edit_classes()

    danger = [m for m, kind in notices if kind == "danger"]
    assert len(danger) == 1 and danger[0].startswith("Could not save gt/classes.json:")
    assert layer.registry() == before and ("Classes updated", "success") not in notices
    assert layer.brush_color.name() == before.by_key("root").color
    assert not page.session.files_changed()
