"""Remove GT… on the Correct page: the open image's GT files, and nothing else.

The button is off while nothing is open or edits are unsaved; it asks first (Cancel is the
default and Esc); confirmed, it deletes ``gt/mask/<stem>.png``, ``gt/skeletons/<stem>.png`` and
``gt/<stem>.png`` (never a prediction, ``gt/classes.json`` or another image's GT), reads the
image again from what is left, forgets its read-ahead and names a file it could not delete.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PyQt6.QtWidgets import (
    QMessageBox,
)

from app.gt import (
    Tracing,
    default_registry,
    registry_path,
    save_registry,
    write_tracing,
)
from app.gui.correct.layer import CLASSES, MASK, SKELETON
from app.masks import write_mask_file
from app.project_layout import ProjectLayout
from tests.box_titles import record_titles, title_of
from tests.correct_classes_support import (
    REMOVE_DETAIL,
    ids,
    norm,
    paint,
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
    draw_line,
    record,
)

pytestmark = pytest.mark.usefixtures("correct_guards")


def _all_gt(editor, layout, folder) -> None:
    """Every layer edited and saved: the three GT files exist."""
    save_registry(folder, default_registry())
    paint(editor, (20, 100), (60, 100))
    editor.set_active_layer(SKELETON)
    draw_line(editor)
    editor.set_active_layer(CLASSES)
    paint(editor, (20, 120), (80, 120))
    assert editor.save()
    for path in (layout.gt_mask_path("a"), layout.gt_skeleton_path("a"), layout.gt_path("a")):
        assert os.path.isfile(path), path


def test_remove_gt_is_disabled_while_edits_are_unsaved_and_asks_before_it_deletes(
    window_with_project, confirm
):
    window, editor, project = (
        window_with_project.window,
        window_with_project.editor,
        window_with_project.project,
    )
    layout = ProjectLayout.for_image(project.a)
    gt_mask, gt_skel, gt_labels = (
        layout.gt_mask_path("a"),
        layout.gt_skeleton_path("a"),
        layout.gt_path("a"),
    )
    removed = record(editor.gt_removed)
    button = window.gt_remove_button
    save_registry(project.folder, default_registry())
    paint(editor, (20, 100), (60, 100))
    assert not button.isEnabled() and "Save or discard" in button.toolTip()
    editor.set_active_layer(SKELETON)
    draw_line(editor)
    editor.set_active_layer(CLASSES)
    paint(editor, (20, 120), (80, 120))
    assert editor.save()
    assert button.isEnabled() and window.image_manager.has_gt_mask("a")

    confirm.answer = False
    button.click()
    assert os.path.isfile(gt_mask) and removed == []

    confirm.answer = True
    button.click()

    assert not any(map(os.path.exists, (gt_mask, gt_skel, gt_labels)))
    assert os.path.isfile(project.mask()) and os.path.isfile(project.generated())
    assert removed == [project.a] and editor.layers[MASK].status().label == "Prediction"
    assert editor.layers[SKELETON].status().label == "Prediction"
    assert editor.layers[CLASSES].status().label == "No class labels"
    assert not window.image_manager.has_gt_mask("a")
    assert not window.image_manager.has_gt_skeleton("a")
    assert not window.image_manager.has_gt_labels("a")
    assert os.path.isfile(registry_path(project.folder))  # the project's classes stay
    assert confirm.calls[-1] == ("a.png", REMOVE_DETAIL)


def test_remove_gt_needs_an_open_image(make_editor):
    editor = make_editor()
    button = editor.actions.remove_button

    assert not button.isEnabled() and button.text() == "Remove GT…"


def test_remove_gt_toasts_what_it_removed(make_editor, project, layout, confirm, notices):
    editor = make_editor(project.a, project.folder)
    editor.actions.confirm_remove = confirm
    _all_gt(editor, layout, project.folder)
    confirm.answer = True
    notices.clear()

    editor.remove_gt()

    assert ("Removed GT · a.png", "success") in notices
    assert not editor.has_unsaved_changes() and not ids(editor).any()


def test_remove_gt_with_no_gt_asks_nothing(make_editor, project, confirm, notices):
    editor = make_editor(project.a, project.folder)
    editor.actions.confirm_remove = confirm

    editor.remove_gt()

    assert confirm.calls == [] and notices == [("No GT to remove for a.png", "info")]


def test_remove_gt_forgets_the_read_ahead_of_that_image(make_editor, project, layout, confirm):
    editor = make_editor(project.a, project.folder)
    editor.actions.confirm_remove = confirm
    _all_gt(editor, layout, project.folder)
    editor.set_neighbour_provider(lambda path: [project.b])
    editor.load_image(project.b, project.folder)
    editor.load_image(project.a, project.folder)  # b is kept as read; a was read back
    confirm.answer = True
    forgotten = []
    real = editor.paging.forget
    editor.paging.forget = lambda path=None: forgotten.append(path) or real(path)

    editor.remove_gt()

    assert forgotten and all(path == project.a for path in forgotten)
    assert norm(project.a) not in {norm(p) for p in editor.paging.pending()}
    cached = editor.paging.cached_photo(project.a)
    assert cached is not None  # the photo stays shared; the documents were read again
    assert editor.layers[MASK].status().label == "Prediction"


def test_remove_gt_reports_what_it_could_not_remove(
    make_editor, project, layout, confirm, notices, monkeypatch
):
    """Must-do (d): a file that stays is named, the rest is removed and the image reopens."""
    editor = make_editor(project.a, project.folder)
    editor.actions.confirm_remove = confirm
    _all_gt(editor, layout, project.folder)
    removed = record(editor.gt_removed)
    stuck = layout.gt_skeleton_path("a")
    real_remove = os.remove

    def remove(path):
        if norm(path) == norm(stuck):
            raise PermissionError(13, "Permission denied")
        real_remove(path)

    monkeypatch.setattr(os, "remove", remove)
    confirm.answer = True
    notices.clear()

    editor.remove_gt()

    assert os.path.isfile(stuck) and not os.path.exists(layout.gt_mask_path("a"))
    assert removed == [project.a]
    assert editor.layers[MASK].status().label == "Prediction"  # reopened from what is left
    assert editor.layers[SKELETON].status().label == "GT"
    assert notices == [
        (
            "Remove GT · a.png: removed gt/mask/a.png, gt/a.png; could not remove"
            " gt/skeletons/a.png (Permission denied)",
            "danger",
        )
    ]


def test_the_remove_gt_box_defaults_to_cancel(monkeypatch):
    """Enter and Esc both keep the GT: Cancel is the default and the Esc button."""
    from app.gui.correct.actions import ask_remove_gt

    seen: list[tuple] = []

    def exec_(box):
        cancel = box.button(QMessageBox.StandardButton.Cancel)
        assert box.defaultButton() is cancel and box.escapeButton() is cancel
        seen.append((title_of(box), box.text(), box.informativeText()))
        answer = seen_answers.pop(0)
        if answer is not None:
            next(b for b in box.buttons() if b.text() == answer).click()
        return 0

    seen_answers: list[str | None] = [None, "Cancel", "Remove GT"]
    record_titles(monkeypatch)
    monkeypatch.setattr(QMessageBox, "exec", exec_)

    assert ask_remove_gt(None, "a.png", "detail") is False  # closed: Esc
    assert ask_remove_gt(None, "a.png", "detail") is False  # Cancel
    assert ask_remove_gt(None, "a.png", "detail") is True
    assert seen[0] == ("Remove GT", "Remove the GT of a.png?", "detail")


def every_byte(folder: str) -> dict[str, bytes]:
    """``relative path -> bytes`` of every file under ``folder``."""
    files = {}
    for dirpath, _dirs, names in os.walk(folder):
        for name in names:
            path = os.path.join(dirpath, name)
            with open(path, "rb") as fh:
                files[os.path.relpath(path, folder).replace(os.sep, "/")] = fh.read()
    return files


def test_remove_gt_changes_exactly_the_open_images_three_gt_files(
    make_editor, project, layout, confirm
):
    """Image a and image b both have a GT mask, a GT skeleton and class labels, and both
    have predictions: Remove GT… on a removes a's three files and no other byte changes
    (b's GT, the predictions, ``gt/classes.json`` and the folders stay)."""
    b_layout = ProjectLayout.for_image(project.b)
    on = np.zeros((H, W), bool)
    on[5:15, 5:25] = True
    write_mask_file(b_layout.gt_mask_path("b"), on, durable=False)
    write_mask_file(b_layout.gt_skeleton_path("b"), on, durable=False)
    write_tracing(b_layout.gt_path("b"), Tracing(np.ones((H, W), np.uint8)), default_registry())
    editor = make_editor(project.a, project.folder)
    editor.actions.confirm_remove = confirm
    _all_gt(editor, layout, project.folder)
    before = every_byte(project.folder)
    confirm.answer = True

    editor.remove_gt()

    after = every_byte(project.folder)
    gone = {"gt/mask/a.png", "gt/skeletons/a.png", "gt/a.png"}
    assert {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)} == gone
    assert gone <= before.keys() and not gone & after.keys()
    assert {"gt/mask/b.png", "gt/skeletons/b.png", "gt/b.png", "gt/classes.json"} <= after.keys()
