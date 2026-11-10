"""Unmark's toast is the editors' one save toast: it replaces Mark empty's in place (V6).

``app.gui.window.editors.on_editor_notice`` keys a save's summary ("Saved … · X"), Remove
GT's ("Removed GT · X") and Unmark's ("Unmarked · X") as ``SAVED_TOAST``, so Mark empty then
Unmark show one toast that changes its text, never two stacked toasts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.gui.window.editors import SAVED_TOAST
from tests.correct_support import new_project, settle
from tests.qt_teardown import delete_widgets
from tests.window_support import build_window, open_folder, quiet_window

pytestmark = pytest.mark.usefixtures("correct_guards")


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
    yield SimpleNamespace(window=window, project=project)
    del window
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def test_unmark_replaces_the_mark_empty_toast_in_place(window_with_project, monkeypatch):
    window = window_with_project.window
    toasts: list[tuple[str, object]] = []
    real = window.notify

    def notify(message, kind="info", *args, key=None, **kwargs):
        toasts.append((message, key))
        return real(message, kind, *args, key=key, **kwargs)

    monkeypatch.setattr(window, "notify", notify)

    window.gt_empty_button.click()  # Mark empty
    settle()
    window.gt_empty_button.click()  # Unmark
    settle()

    assert toasts == [
        ("Saved class labels · a.png", SAVED_TOAST),
        ("Unmarked · a.png", SAVED_TOAST),
    ]
