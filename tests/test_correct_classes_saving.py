"""Saving the class labels on the Correct page, and coming back to them.

Save writes only ``gt/<stem>.png`` (and ``gt/classes.json`` when missing) and only for unsaved
labels (O2); each layer writes only its own file; the one prompt names the class labels; PgDn
saves silently unless a lossy remap must be asked about; Discard puts back the saved labels and
keeps the class picked; reselecting a clean image keeps its labels, history and view, while a
labels file changed on disk is read again; one undo timeline runs through the three layers; the
labels are read ahead and handed back; a labels file another program holds open is reported,
not raised. Helpers and fixtures: ``tests/correct_classes_support.py``.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from app import io_utils
from app.gt import (
    Registry,
    Tracing,
    default_registry,
    gt_path,
    read_tracing,
    registry_path,
    save_registry,
    write_tracing,
)
from app.gui.correct.editor import CorrectEditor
from app.gui.correct.layer import CLASSES, MASK, SKELETON
from app.gui.editor_prompts import DISCARD, SAVE
from app.gui.paint_canvas import pixmap_to_argb_array
from tests.correct_classes_support import (
    assert_one_tick,
    box,
    changed,
    ids,
    norm,
    paint,
    press,
    row_text,
    tree,
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
    H,
    W,
    drag,
    draw_line,
    record,
)
from tests.gt_goldens_support import write_legacy_tracing
from tests.qt_teardown import delete_widgets
from tests.qt_wait import wait_until as _spin_until

pytestmark = pytest.mark.usefixtures("correct_guards")

STEM = "24EFCAM1_T001_L045_2024.07.18_111618_001_SMM"


def test_a_classes_save_writes_only_gt_stem_png_and_emits_gt_saved(
    make_editor, project, layout, signals, notices
):
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(CLASSES)
    paint(editor, (20, 100), (80, 110))
    before = tree(project.folder)

    assert editor.save() is True

    expected = {norm(layout.gt_path("a")), norm(registry_path(project.folder))}
    assert changed(before, tree(project.folder)) == expected
    assert signals["gt_saved"] == [project.a] and signals["mask_saved"] == []
    assert ("Saved class labels · a.png", "success") in notices
    assert read_tracing(layout.gt_path("a")).ids[105, 50] == 1


def test_each_layer_writes_only_its_own_file(make_editor, project, layout):
    """The guard of Design 3.7: one layer, one file."""
    save_registry(project.folder, default_registry())  # so the labels save writes one file
    editor = make_editor(project.a, project.folder)
    edits = {
        MASK: lambda: paint(editor, (20, 100), (60, 100)),
        SKELETON: lambda: draw_line(editor),
        CLASSES: lambda: paint(editor, (20, 100), (80, 110)),
    }
    files = {
        MASK: layout.gt_mask_path("a"),
        SKELETON: layout.gt_skeleton_path("a"),
        CLASSES: layout.gt_path("a"),
    }
    for layer_id in (MASK, SKELETON, CLASSES):
        editor.set_active_layer(layer_id)
        edits[layer_id]()
        before = tree(project.folder)
        assert editor.save(), layer_id
        assert changed(before, tree(project.folder)) == {norm(files[layer_id])}, layer_id


def test_prompts_name_the_class_labels(make_editor, project, prompts):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (60, 100))  # the mask
    editor.set_active_layer(CLASSES)
    paint(editor, (20, 120), (80, 120))
    prompts.answer = DISCARD

    assert editor.maybe_save_changes() is True

    assert prompts.calls == [("ask_unsaved", "mask and class labels", "a.png")]
    assert not editor.has_unsaved_changes() and not ids(editor).any()


def test_an_untouched_image_writes_no_class_labels(make_editor, project, layout, notices):
    """O2: Save writes only the unsaved layers (Trace GT wrote an empty file here)."""
    editor = make_editor(project.a, project.folder)
    editor.set_active_layer(CLASSES)

    assert editor.save() is False
    assert not os.path.exists(layout.gt_path("a")) and ("Nothing unsaved", "info") in notices


def test_a_box_click_alone_is_no_edit(page, layout):
    """The box was a "checked" tick, an edit saved into the file, until 2026-11-01: now it
    picks the class, and Save has nothing to write."""
    box(page, "nodule").click()
    assert_one_tick(page, "nodule")
    assert not page.has_unsaved_changes()

    assert page.save() is False and not os.path.exists(layout.gt_path("a"))


def test_pgdn_prompts_after_a_lossy_remap_and_saves_silently_otherwise(
    make_editor, project_lossy, layout, prompts, notices
):
    a = project_lossy.a
    original = (Path(layout.gt_path("a"))).read_bytes()
    editor = make_editor(a, project_lossy.folder)
    moves = record(editor.navigate_requested)
    assert editor.layers[CLASSES].is_dirty()
    assert any(kind == "warn" and "hypha" in message for message, kind in notices)

    editor.navigate(1)  # lossy: the prompt, answered Cancel
    assert prompts.calls == [("ask_unsaved", "class labels", "a.png")] and moves == []
    assert Path(layout.gt_path("a")).read_bytes() == original

    prompts.answer = SAVE
    editor.navigate(1)
    assert moves == [1] and (read_tracing(layout.gt_path("a")).ids[5:15, 5:15] == 255).all()

    editor.set_active_layer(CLASSES)
    paint(editor, (20, 100), (80, 110))  # no remap pending now: a silent save
    prompts.calls.clear()
    editor.navigate(1)
    assert prompts.calls == [] and moves == [1, 1]
    assert read_tracing(layout.gt_path("a")).ids[105, 50] == 1


def test_pgdn_saves_class_labels_and_asks_for_the_next_image(page, layout):
    moves = record(page.navigate_requested)
    paint(page, (20, 100), (80, 110))

    press(page, "PgDown")

    assert moves == [1] and os.path.isfile(layout.gt_path("a"))
    assert not page.has_unsaved_changes()


def test_pgdn_saves_a_lossless_remap_silently(make_editor, project, layout, notices):
    """Ids that differ only by number (same keys) are remapped and saved without a prompt."""
    swapped = default_registry().to_dict()
    swapped["classes"][0]["id"], swapped["classes"][1]["id"] = 2, 1  # root = 2, nodule = 1
    plane = np.zeros((H, W), np.uint8)
    plane[5:15, 5:15] = 2  # root in the other project's ids
    write_tracing(layout.gt_path("a"), Tracing(plane), Registry.from_dict(swapped))
    editor = make_editor(project.a, project.folder)
    moves = record(editor.navigate_requested)
    assert editor.layers[CLASSES].is_dirty() and not [n for n in notices if n[1] == "warn"]

    editor.navigate(1)  # the guard fails on any prompt

    assert moves == [1] and (read_tracing(layout.gt_path("a")).ids[5:15, 5:15] == 1).all()


def test_discard_restores_the_saved_labels_and_keeps_the_class_picked(page, prompts):
    paint(page, (10, 30), (70, 30))  # root
    assert page.save()
    saved = ids(page).copy()
    press(page, "2")
    paint(page, (40, 5), (40, 55))  # nodule
    box(page, "root_tip").click()  # picks Tip: no edit
    prompts.answer = DISCARD

    assert page.maybe_save_changes() is True

    assert prompts.calls == [("ask_unsaved", "class labels", "a.png")]
    assert not page.has_unsaved_changes() and np.array_equal(ids(page), saved)
    argb = pixmap_to_argb_array(page.layers[CLASSES].canvas.image)
    assert np.array_equal(argb, page.layers[CLASSES].document.lut()[saved])
    assert_one_tick(page, "root_tip")  # Discard puts back pixels, never the class picked
    assert not page.layers[CLASSES].can_undo() and not page.undo_button.isEnabled()
    assert row_text(page, CLASSES) == "GT"


def test_an_edit_after_discard_never_saves_the_discarded_strokes(page, prompts, layout):
    paint(page, (10, 30), (70, 30))
    assert page.save()
    press(page, "2")
    paint(page, (40, 5), (40, 55))  # to be discarded
    prompts.answer = DISCARD
    assert page.maybe_save_changes() is True
    paint(page, (5, 50), (20, 50))  # a new nodule stroke, kept

    assert page.save()

    saved = read_tracing(layout.gt_path("a")).ids
    assert saved[50, 10] == 2 and saved[10, 40] == 0 and saved[30, 40] == 1


def test_discard_without_a_labels_file_empties_the_canvas(page, prompts, layout):
    paint(page, (10, 10), (40, 10))
    box(page, "nodule").click()
    prompts.answer = DISCARD

    assert page.maybe_save_changes() is True

    assert not ids(page).any()
    assert_one_tick(page, "nodule")
    assert row_text(page, CLASSES) == "No class labels"
    assert not os.path.exists(layout.gt_path("a"))


def test_reselecting_the_clean_open_image_keeps_labels_history_and_view(page, project, reads):
    paint(page, (10, 10), (40, 10))
    assert page.save()
    paint(page, (10, 40), (40, 40))
    page.undo()  # back at the saved labels; one undo and one redo left
    view = page.graphics_view
    view.set_zoom(4.0)
    view.centerOn(20, 20)
    zoom, center = view.zoom_factor, view.mapToScene(view.viewport().rect().center())
    reads.clear()

    page.load_image(project.a, project.folder)

    assert reads == []  # the labels file was not read again
    assert page.layers[CLASSES].can_undo() and page.layers[CLASSES].can_redo()
    assert view.zoom_factor == zoom
    assert view.mapToScene(view.viewport().rect().center()) == center
    assert not page.has_unsaved_changes()


def test_a_labels_file_changed_on_disk_reloads_in_the_same_view(page, project, layout, reads):
    from tests.correct_support import bump

    paint(page, (10, 10), (40, 10))
    assert page.save()
    view = page.graphics_view
    view.set_zoom(3.0)
    zoom = view.zoom_factor
    other = np.zeros((H, W), np.uint8)
    other[30:40, 30:40] = 2  # someone else's labels, saved by an older SPROUTS
    write_legacy_tracing(layout.gt_path("a"), Tracing(other), default_registry(), {"nodule"})
    bump(layout.gt_path("a"))
    reads.clear()

    page.load_image(project.a, project.folder)

    assert [norm(path) for path in reads] == [norm(layout.gt_path("a"))]
    assert np.array_equal(ids(page), other)  # its old "reviewed" list is ignored
    assert_one_tick(page, "root")
    assert not page.layers[CLASSES].can_undo() and view.zoom_factor == zoom
    assert not page.has_unsaved_changes()
    os.remove(layout.gt_path("a"))  # gone: empty labels
    page.load_image(project.a, project.folder)
    assert not ids(page).any() and row_text(page, CLASSES) == "No class labels"


def test_one_undo_timeline_runs_through_the_three_layers(make_editor, project, notices):
    editor = make_editor(project.a, project.folder)
    paint(editor, (20, 100), (60, 100))  # the mask
    editor.set_active_layer(SKELETON)
    draw_line(editor)
    editor.set_active_layer(CLASSES)
    paint(editor, (20, 120), (80, 120))
    editor.set_active_layer(MASK)
    assert editor.session.journal.undo_entries == (MASK, SKELETON, CLASSES)
    notices.clear()

    editor.undo()  # the newest action: the class labels, without switching layers
    assert ("Undo · class labels", "info") in notices and editor.active_layer_id == MASK
    assert not editor.layers[CLASSES].is_dirty() and editor.layers[MASK].is_dirty()
    editor.undo()
    assert ("Undo · skeleton", "info") in notices and not editor.layers[SKELETON].is_dirty()
    editor.undo()
    assert not editor.has_unsaved_changes()
    editor.redo()
    assert editor.layers[MASK].is_dirty() and not editor.layers[CLASSES].is_dirty()


def test_the_class_labels_are_read_ahead_and_handed_back(make_editor, project, monkeypatch):
    """PgDn opens the neighbour's class labels from what the worker read; going back takes
    the image left as it is in memory (also after a class-label save): no read either way."""
    from app.gui.correct.classes_layer import ClassesLayer

    opened: list[bool] = []
    real_open = ClassesLayer.open

    def spy(layer, *args, ahead=None, **kwargs):
        opened.append(ahead is not None)
        return real_open(layer, *args, ahead=ahead, **kwargs)

    monkeypatch.setattr(ClassesLayer, "open", spy)
    editor = make_editor(project.a, project.folder)
    other = {norm(project.a): project.b, norm(project.b): project.a}
    editor.set_neighbour_provider(lambda path: [other[norm(path)]])
    editor.load_image(project.a, project.folder)  # read b ahead
    assert _spin_until(lambda: editor.paging.cached_photo(project.b) is not None)
    opened.clear()

    editor.load_image(project.b, project.folder)
    assert opened == [True]  # b from the worker's read
    editor.load_image(project.a, project.folder)
    assert opened == [True, True]  # a handed back

    editor.set_active_layer(CLASSES)
    paint(editor, (20, 100), (80, 110))
    assert editor.save()
    editor.load_image(project.b, project.folder)
    opened.clear()
    editor.load_image(project.a, project.folder)
    assert opened == [True] and ids(editor)[105, 50] == 1  # the saved labels, handed back


def _sharing_violation(path) -> PermissionError:
    exc = PermissionError(errno.EACCES, "The process cannot access the file", str(path))
    exc.winerror = 32  # ERROR_SHARING_VIOLATION
    return exc


@pytest.mark.usefixtures("no_prompts", "gl_off")
def test_the_class_labels_report_a_locked_gt_file(app, tmp_path, monkeypatch):
    image = str(tmp_path / f"{STEM}.png")
    cv2.imwrite(image, np.full((30, 40, 3), 90, np.uint8))
    notices = []
    editor = CorrectEditor()
    try:
        editor.resize(900, 700)
        editor.notice.connect(lambda msg, kind: notices.append((msg, kind)))
        editor.load_image(image, images_base_folder=str(tmp_path))
        editor.set_active_layer(CLASSES)
        drag(editor, (5, 5), (20, 5))

        def locked(src, dst):
            raise _sharing_violation(dst)

        monkeypatch.setattr(io_utils.os, "replace", locked)
        monkeypatch.setattr(io_utils.time, "sleep", lambda s: None)
        notices.clear()

        assert editor.save() is False  # reported, not raised
        assert any(kind == "danger" and "open in another program" in msg for msg, kind in notices)
        assert editor.has_unsaved_changes()
        assert read_tracing(gt_path(image)) is None
    finally:
        delete_widgets([editor])
