"""The class labels' actions in the window: the top bar's Correct page and the export.

The Correct page of the top bar holds Classes…, Export dataset…, Mark empty and Remove GT…
(the editor's own buttons). Export dataset… opens the dialog on the folder's own registry, read
before the unsaved-changes prompt (an invalid file costs no edits) and again after it, saves the
open edits first, and the dialog is deleted after use. A saved or removed labels file moves the
Library's marks and the metrics bar's "Classes" status, with one toast.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import (
    QApplication,
    QPushButton,
)

from app.gt import (
    default_registry,
    registry_path,
    save_registry,
)
from app.gui import editor_prompts
from app.gui.correct.layer import CLASSES
from app.gui.editor_prompts import CANCEL, DISCARD, SAVE
from app.project_layout import ProjectLayout
from tests.correct_classes_support import confirm as confirm  # fixture
from tests.correct_classes_support import (
    fake_class_editor,
    paint,
)
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
from tests.qt_teardown import delete_widgets

pytestmark = pytest.mark.usefixtures("correct_guards")


def test_the_correct_page_holds_the_gt_actions(window_with_project):
    window = window_with_project.window
    topbar = window.ribbon
    page = window.action_bar_stack.widget(topbar._pages["Correct"])

    assert [b.text() for b in page.findChildren(QPushButton)] == [
        # O18: no Clear here, the page's one Clear is in the dock
        "Classes…",
        "Export dataset…",
        "Mark empty",
        "Remove GT…",
    ]
    assert window.gt_remove_button is window.correct_editor.actions.remove_button
    assert window.gt_empty_button is window.correct_editor.actions.empty_button
    assert "Trace GT" not in topbar._pages  # the stage and its page are gone (WG-T11)


def test_the_topbar_classes_button_opens_the_page_class_editor(window_with_project, monkeypatch):
    seen: list = []
    fake_class_editor(monkeypatch, None, seen)

    window_with_project.window.gt_classes_button.click()

    assert len(seen) == 1


def test_export_gt_dataset_saves_the_correct_editor_first(window_with_project, monkeypatch):
    import app.gui.export_dataset_dialog as edd

    window, editor = window_with_project.window, window_with_project.editor
    layout = ProjectLayout.for_image(window_with_project.project.a)
    opened = []
    monkeypatch.setattr(edd.ExportDatasetDialog, "exec", lambda dialog: opened.append(dialog) or 0)
    editor.set_active_layer(CLASSES)
    paint(editor, (20, 100), (80, 110))
    asked = []
    monkeypatch.setattr(editor_prompts, "ask_unsaved", lambda *args: asked.append(args) or SAVE)

    window.export_gt_dataset()

    assert len(asked) == 1 and os.path.isfile(layout.gt_path("a")) and len(opened) == 1

    paint(editor, (20, 120), (80, 120))
    monkeypatch.setattr(editor_prompts, "ask_unsaved", lambda *args: CANCEL)
    window.export_gt_dataset()
    assert len(opened) == 1  # Cancel: no export without the edits


def test_export_gt_dataset_refuses_an_invalid_registry_before_the_prompt(
    window_with_project, monkeypatch
):
    # C6-03: gt/classes.json is read before Save / Discard is asked, so a bad file costs no
    # edits; it is read again after the prompt, as a Save there can write it.
    import app.gui.export_dataset_dialog as edd

    window, editor = window_with_project.window, window_with_project.editor
    folder = ProjectLayout.for_image(window_with_project.project.a).root
    opened, asked, toasts = [], [], []
    monkeypatch.setattr(
        edd.ExportDatasetDialog, "exec", lambda dialog: opened.append(dialog.registry) or 0
    )
    monkeypatch.setattr(window, "notify", lambda message, kind="success", **_k: toasts.append(kind))
    editor.set_active_layer(CLASSES)
    paint(editor, (20, 100), (80, 110))
    Path(registry_path(folder)).parent.mkdir(parents=True, exist_ok=True)
    Path(registry_path(folder)).write_text("{", encoding="utf-8")
    monkeypatch.setattr(editor_prompts, "ask_unsaved", lambda *args: asked.append(args) or DISCARD)

    window.export_gt_dataset()

    assert asked == [] and opened == [] and toasts == ["danger"]
    assert editor.has_unsaved_changes()  # the edits stay

    custom = default_registry().add(key="dead_root", name="Dead root", color="#7f5af0")
    save_registry(folder, default_registry())

    def rewrite_at_the_prompt(*args):
        save_registry(folder, custom)  # as a Save at the prompt can
        return DISCARD

    monkeypatch.setattr(editor_prompts, "ask_unsaved", rewrite_at_the_prompt)
    window.export_gt_dataset()
    assert opened == [custom]


def test_the_window_handlers_follow_the_gt_files(window_with_project, monkeypatch):
    """``on_gt_saved`` marks the class labels; ``on_gt_removed`` sets every GT mark from
    the disk (a file that could not be deleted keeps its mark: must-do d);
    ``on_skeleton_cleared`` drops the skeleton's mark. Each repaints the row."""
    from app.gui.window import editors as window_editors

    window, project = window_with_project.window, window_with_project.project
    layout = ProjectLayout.for_image(project.a)
    manager = window.image_manager
    repainted: list[str] = []
    real = window.library_index.refresh_status
    monkeypatch.setattr(
        window.library_index,
        "refresh_status",
        lambda name=None: repainted.append(name) or real(name),
    )

    window_editors.on_gt_saved(window, project.a)
    assert manager.has_gt_labels("a") and repainted == ["a"]

    os.makedirs(os.path.dirname(layout.gt_skeleton_path("a")), exist_ok=True)
    Path(layout.gt_skeleton_path("a")).write_bytes(b"stays")  # the file that stayed
    manager.set_gt("a", "mask", True)
    window_editors.on_gt_removed(window, project.a)
    assert (manager.has_gt_mask("a"), manager.has_gt_skeleton("a")) == (False, True)
    assert not manager.has_gt_labels("a") and repainted == ["a", "a"]

    window_editors.on_skeleton_cleared(window, project.a)
    assert not manager.has_gt_skeleton("a") and repainted == ["a", "a", "a"]


def test_a_class_label_save_marks_the_library_row(window_with_project):
    window, editor = window_with_project.window, window_with_project.editor
    assert not window.image_manager.has_gt_labels("a")
    editor.set_active_layer(CLASSES)
    paint(editor, (20, 100), (80, 110))

    assert editor.save()

    assert window.image_manager.has_gt_labels("a")
    assert window.metrics_bar._status_value.text().endswith("Classes: traced")


def test_a_class_label_save_shows_one_toast_the_pages_summary(window_with_project, monkeypatch):
    window, editor = window_with_project.window, window_with_project.editor
    toasts: list[str] = []
    real = window.notify
    monkeypatch.setattr(
        window, "notify", lambda message, *a, **k: toasts.append(message) or real(message, *a, **k)
    )
    editor.set_active_layer(CLASSES)
    paint(editor, (20, 100), (80, 110))

    assert editor.save()

    assert toasts == [f"Saved class labels · {os.path.basename(window_with_project.project.a)}"]


@pytest.fixture
def main_window(app, monkeypatch):
    from app.gui.main_window import MainWindow

    made = [MainWindow()]
    # No editor prompt may block a headless run.
    monkeypatch.setattr(made[0].correct_editor, "maybe_save_changes", lambda: None)
    yield made[0]
    delete_widgets(made)


def test_export_gt_dataset_opens_dialog(main_window, monkeypatch, tmp_path):

    from app.gt import default_registry

    mw = main_window
    toasts = []
    monkeypatch.setattr(mw, "notify", lambda msg, kind="success", **_k: toasts.append(kind))

    opened = []

    class _FakeDialog:
        def __init__(self, project_dir, images, registry, parent=None):
            opened.append((project_dir, images, registry, parent))

        def exec(self):
            return 0

        def deleteLater(self):  # noqa: N802 - Qt API
            pass

        def testAttribute(self, _attribute):  # noqa: N802 - Qt API (run_modal asks)
            return False

    fake = types.ModuleType("app.gui.export_dataset_dialog")
    fake.ExportDatasetDialog = _FakeDialog
    monkeypatch.setitem(sys.modules, "app.gui.export_dataset_dialog", fake)

    mw.image_manager.original_folder = None
    mw.export_gt_dataset()
    assert opened == [] and toasts == ["warn"]

    mw.image_manager.original_folder = str(tmp_path)
    mw.image_manager.images = {"a": str(tmp_path / "a.png")}
    mw.export_gt_dataset()
    assert len(opened) == 1
    project_dir, images, registry, parent = opened[0]
    assert project_dir == str(tmp_path)
    assert images == {"a": str(tmp_path / "a.png")}
    assert images is not mw.image_manager.images  # a copy
    assert registry == default_registry()  # the folder has no gt/classes.json yet
    assert parent is mw


def test_export_uses_the_folders_registry(main_window, monkeypatch, tmp_path):
    """Export must use the folder's gt/classes.json even if Correct never opened
    an image there (the editor's registry is still the default then)."""

    from app.gt import default_registry, save_registry

    mw = main_window
    toasts = []
    monkeypatch.setattr(mw, "notify", lambda msg, kind="success", **_k: toasts.append(kind))
    opened = []

    class _FakeDialog:
        def __init__(self, project_dir, images, registry, parent=None):
            opened.append(registry)

        def exec(self):
            return 0

        def deleteLater(self):  # noqa: N802 - Qt API
            pass

        def testAttribute(self, _attribute):  # noqa: N802 - Qt API (run_modal asks)
            return False

    fake = types.ModuleType("app.gui.export_dataset_dialog")
    fake.ExportDatasetDialog = _FakeDialog
    monkeypatch.setitem(sys.modules, "app.gui.export_dataset_dialog", fake)

    custom = default_registry().add(key="dead_root", name="Dead root", color="#7f5af0")
    save_registry(str(tmp_path), custom)
    mw.image_manager.original_folder = str(tmp_path)
    mw.image_manager.images = {}
    mw.export_gt_dataset()
    assert opened == [custom]

    # An invalid registry blocks the export instead of silently using defaults.
    (tmp_path / "gt" / "classes.json").write_text("{", encoding="utf-8")
    mw.export_gt_dataset()
    assert opened == [custom] and toasts == ["danger"]


def test_export_dialog_is_deleted_after_use(main_window, monkeypatch, tmp_path):
    from PyQt6 import sip

    import app.gui.export_dataset_dialog as edd

    mw = main_window
    mw.image_manager.original_folder = str(tmp_path)
    mw.image_manager.images = {}
    opened = []
    monkeypatch.setattr(edd.ExportDatasetDialog, "exec", lambda d: opened.append(d) or 0)
    mw.export_gt_dataset()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(opened) == 1 and sip.isdeleted(opened[0])
