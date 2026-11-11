"""MainWindow wiring of units outside the window's frame: the Welcome screen and the loading
card, library navigation, opening a project and the recents, the jobs, the shortcut sheet
and About, the GT dataset export.

Offscreen; projects are tmp_path folders of tiny synthetic JPGs; every window gets its own
INI QSettings, so the user's real settings are never read or written.

(Parked here by A10 in wave 1, bodies unchanged; the lead decides the final homes at
integration. The GT-export tests are W2-labels': they stay until wave 2.)
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QPushButton

from tests.qt_teardown import delete_widgets
from tests.window_support import app as app  # fixture
from tests.window_support import make_window as make_window  # fixture
from tests.window_support import mw as mw  # fixture
from tests.window_support import no_delayed_temp_sweep as no_delayed_temp_sweep  # fixture
from tests.window_support import open_folder, write_photos
from tests.window_support import settings as settings  # fixture

USER = Qt.ItemDataRole.UserRole
STEMS = [f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in (11, 12, 13)]


def test_welcome_widget_imports(app):
    from app.gui.welcome_screen import WelcomeWidget

    calls = []
    opened = []
    w = WelcomeWidget(
        on_get_started=lambda: calls.append(1),
        on_open_recent=lambda p: opened.append(p),
    )
    assert w is not None
    # The open-recent forwarder routes to the supplied callback.
    w._handle_open_recent("C:/some/dir")
    assert opened == ["C:/some/dir"]


def test_welcome_get_started_callback(app):
    from app.gui.welcome_screen import WelcomeWidget

    calls = []
    w = WelcomeWidget(on_get_started=lambda: calls.append(1))
    # Find the primary Browse button and click it.
    browse = w.findChild(QPushButton, "primaryButton")
    assert browse is not None
    browse.click()
    assert calls == [1]


def test_loading_overlay_imports(app):
    from app.gui.loading_overlay import LoadingOverlay

    ov = LoadingOverlay()
    ov.set_filename("foo.png")
    ov.set_count(3, 10)
    ov.set_progress(50)
    ov.start()
    ov.hide()


def test_filter_file_list(app):
    """FIX5: case-insensitive contains filter, empty query shows all; parents of
    matching leaves stay visible."""
    from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem

    from app.gui.ui_panels import filter_file_list

    class _MW:
        pass

    mw = _MW()
    tree = QTreeWidget()
    parent = QTreeWidgetItem(["FieldA"])
    leaf1 = QTreeWidgetItem(["root_001.png"])
    leaf2 = QTreeWidgetItem(["other_002.png"])
    parent.addChild(leaf1)
    parent.addChild(leaf2)
    tree.addTopLevelItem(parent)
    mw.file_list = tree

    filter_file_list(mw, "ROOT")
    assert not leaf1.isHidden()
    assert leaf2.isHidden()
    assert not parent.isHidden()  # kept for the matching descendant

    filter_file_list(mw, "")
    assert not leaf1.isHidden()
    assert not leaf2.isHidden()
    assert not parent.isHidden()


def test_loading_records_a_recent_project(mw, tmp_path):
    folder = write_photos(tmp_path / "proj", STEMS)
    open_folder(mw, folder, len(STEMS))
    entries = mw.recent_projects.entries()
    assert entries and os.path.normpath(entries[0].path) == os.path.normpath(str(folder))
    assert entries[0].count == len(STEMS)


def test_missing_folder_changes_nothing(mw, tmp_path):
    seen = []
    mw.notify = lambda msg, kind="success", **_k: seen.append((kind, msg))
    mw.open_path(str(tmp_path / "nope"))
    assert seen and seen[0][0] == "danger"
    assert mw.app_stack.currentIndex() == 0


def test_open_is_cancelled_by_an_editor_that_keeps_its_edits(mw, tmp_path):
    folder = write_photos(tmp_path / "proj", STEMS)
    calls = []
    mw.image_manager.load_images = calls.append
    mw.correct_editor.maybe_save_changes = lambda: False
    mw.open_path(str(folder))
    assert calls == [] and mw.app_stack.currentIndex() == 0
    mw.correct_editor.maybe_save_changes = lambda: True
    mw.open_path(str(folder))
    assert calls == [str(folder)]


def test_selection_follows_an_editor_that_kept_its_image(mw, tmp_path):
    folder = write_photos(tmp_path / "proj", STEMS)
    open_folder(mw, folder, len(STEMS))
    leaves = mw._image_leaves()
    kept = mw.image_manager.get_image_path(leaves[0].data(0, USER))
    editor = mw.correct_editor
    mw.right_panel.setCurrentWidget(editor)
    editor.load_image(kept, str(folder))
    editor.load_image = lambda *_a, **_k: None  # it keeps its image (the prompt said Cancel)
    mw.on_image_selected(leaves[2])
    assert mw.file_list.currentItem() is leaves[0]


@pytest.fixture
def main_window(app, monkeypatch):
    from app.gui.main_window import MainWindow

    made = [MainWindow()]
    # No editor prompt may block a headless run.
    monkeypatch.setattr(made[0].correct_editor, "maybe_save_changes", lambda: None)
    yield made[0]
    delete_widgets(made)


def test_navigate_image_moves_tree_selection(main_window, monkeypatch):
    from PyQt6.QtWidgets import QTreeWidgetItem

    mw = main_window
    mw.file_list.clear()
    folder = QTreeWidgetItem(["Field"])
    leaves = {}
    for name in ("a", "b", "c"):
        leaf = QTreeWidgetItem([f"{name}.png"])
        leaf.setData(0, Qt.ItemDataRole.UserRole, name)
        folder.addChild(leaf)
        leaves[name] = leaf
    mw.file_list.addTopLevelItem(folder)
    leaves["c"].setHidden(True)  # filtered out of the library
    mw.image_manager.images = {n: f"/project/{n}.png" for n in leaves}

    loaded = []
    monkeypatch.setattr(mw.correct_editor, "load_image", lambda path, **_k: loaded.append(path))
    mw.switch_right_panel("correct")
    mw.file_list.setCurrentItem(leaves["a"])

    mw.navigate_image(1)
    assert mw.file_list.currentItem() is leaves["b"]
    assert loaded == ["/project/b.png"]
    mw.navigate_image(1)  # "c" is hidden: stay on the last visible image
    assert mw.file_list.currentItem() is leaves["b"]
    assert loaded == ["/project/b.png"]
    mw.navigate_image(-1)
    assert mw.file_list.currentItem() is leaves["a"]
    assert loaded == ["/project/b.png", "/project/a.png"]
    mw.navigate_image(-1)  # already first
    assert mw.file_list.currentItem() is leaves["a"]

    # PgDn/PgUp in the editor arrive as navigate_requested(+-1).
    mw.correct_editor.navigate_requested.emit(1)
    assert mw.file_list.currentItem() is leaves["b"]


def test_navigate_starts_from_the_image_open_in_correct(main_window, monkeypatch):
    """Arrow keys move the tree selection without loading; PgDn must still step
    from the image that is open in the editor."""
    from PyQt6.QtWidgets import QTreeWidgetItem

    mw = main_window
    editor = mw.correct_editor
    gt = SimpleNamespace(current_image_path="")  # what the editor has open
    monkeypatch.setattr(
        type(editor), "current_image_path", property(lambda _self: gt.current_image_path)
    )
    mw.file_list.clear()
    folder = QTreeWidgetItem(["Field"])
    leaves = {}
    for name in ("a", "b", "c", "d"):
        leaf = QTreeWidgetItem([f"{name}.png"])
        leaf.setData(0, Qt.ItemDataRole.UserRole, name)
        folder.addChild(leaf)
        leaves[name] = leaf
    mw.file_list.addTopLevelItem(folder)
    mw.image_manager.images = {n: f"/project/{n}.png" for n in leaves}

    loaded = []

    def _load(path, **_kwargs):
        loaded.append(path)
        gt.current_image_path = path

    monkeypatch.setattr(editor, "load_image", _load)
    mw.switch_right_panel("correct")
    gt.current_image_path = "/project/a.png"
    mw.file_list.setCurrentItem(leaves["c"])  # moved by arrow keys, nothing loaded

    mw.navigate_image(1)
    assert loaded == ["/project/b.png"]
    assert mw.file_list.currentItem() is leaves["b"]

    # Nothing open in the editor: fall back to the tree selection.
    gt.current_image_path = ""
    mw.file_list.setCurrentItem(leaves["c"])
    mw.navigate_image(1)
    assert loaded[-1] == "/project/d.png"
    gt.current_image_path = ""


def test_load_results_adds_no_toast_of_its_own(mw, monkeypatch, tmp_path):
    # S11-09: the skeleton run's own toast is the one message ("Results loaded" is gone).
    toasts = []
    monkeypatch.setattr(mw, "notify", lambda message, *a, **k: toasts.append(message))
    refreshed = []
    monkeypatch.setattr(mw, "on_outputs_changed", lambda: refreshed.append(True))

    mw.load_results(str(tmp_path))

    assert refreshed == [True] and toasts == []


def test_job_buttons_are_disabled_while_a_job_runs(mw):
    mw.generate_all_button.setToolTip("Generate masks and skeletons")
    mw.jobs._name = "Generating masks"
    mw._on_jobs_busy_changed(True)
    assert not mw.generate_all_button.isEnabled()
    assert "Generating masks is running" in mw.calculate_length_button.toolTip()
    mw.jobs._name = None
    mw._on_jobs_busy_changed(False)
    assert mw.generate_all_button.isEnabled()
    assert mw.generate_all_button.toolTip() == "Generate masks and skeletons"


def test_global_shortcuts_come_from_the_registry(mw):
    from app.gui import shell_chrome, shortcuts

    keys = {
        action: [s.key().toString() for s in bound]
        for action, bound in mw._global_shortcuts.items()
    }
    assert keys["open_folder"] == list(shortcuts.keys_for("open_folder", shortcuts.GLOBAL))
    assert len(mw.ribbon_buttons) == len(shell_chrome.STAGES)
    assert len(keys["go_to_stage"]) == len(mw.ribbon_buttons) * len(shortcuts.stage_keys()[0])
    assert mw._help_shortcut_f1.key().toString() == "F1"
    for bound in mw._global_shortcuts.values():
        assert all(s.context() == Qt.ShortcutContext.WindowShortcut for s in bound)
    # Paging keys are never window-wide (the editors bind their own): no ambiguity.
    editors = set(mw._editors().values())
    assert all(s.parent() not in editors for s in mw._paging_shortcuts)
    assert all(
        s.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut for s in mw._paging_shortcuts
    )


def test_stage_shortcut_needs_a_loaded_folder(mw, tmp_path):
    from app.gui import shell_chrome

    measure = [key for key, *_ in shell_chrome.STAGES].index("Measure")
    slot = mw._global_shortcuts["go_to_stage"][measure]
    slot.activated.emit()
    assert not mw.ribbon_buttons["Measure"].isChecked()  # Welcome: nothing happens
    open_folder(mw, write_photos(tmp_path / "proj", STEMS), len(STEMS))
    slot.activated.emit()
    assert mw.ribbon_buttons["Measure"].isChecked()


def test_about_and_shortcut_sheet_open(mw, monkeypatch):
    from app.gui import about_dialog, main_window

    opened = []
    monkeypatch.setattr(about_dialog.AboutDialog, "exec", lambda self: opened.append("about"))
    monkeypatch.setattr(main_window.ShortcutsDialog, "exec", lambda self: opened.append("keys"))
    mw.show_about()
    mw.show_shortcuts()
    assert opened == ["about", "keys"]


def _reject_the_modal_dialog_soon():
    from PyQt6.QtCore import QTimer

    def attempt():
        dialog = QApplication.activeModalWidget()
        if dialog is None:
            QTimer.singleShot(10, attempt)
        else:
            dialog.reject()

    QTimer.singleShot(0, attempt)


def test_about_and_shortcut_sheet_close_with_the_real_exec(mw):
    # RB-16: both dialogs set WA_DeleteOnClose, so QDialog.exec() deletes them before it
    # returns; a deleteLater() after it raised "wrapped C/C++ object ... has been deleted"
    # on every close (the stubbed exec above cannot see that).
    from PyQt6.QtCore import QCoreApplication, QEvent

    from app.gui import about_dialog, main_window

    for show in (mw.show_shortcuts, mw.show_about):
        _reject_the_modal_dialog_soon()
        show()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert mw.findChildren(main_window.ShortcutsDialog) == []
    assert mw.findChildren(about_dialog.AboutDialog) == []
