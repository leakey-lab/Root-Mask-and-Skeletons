"""Leaving the editor page (``app/gui/window/editors.py`` ``switch_right_panel``).

An editor that keeps its unsaved edits refuses the switch and stays on screen (an unknown
page changes nothing). Back on the Library page, the display shows the library's selection
(review finding): the Correct page moves the library selection when they page (PgDn) or open
a clicked row, and the metrics bar follows the selection on every stage, but the display
page kept its last image, so back in the Library (or Generate / Measure) the photo of one
image sat over the name and measured values of another. The display page now shows the
selected image whenever it comes back on screen, with the metrics bar read again (a skeleton
correction saved meanwhile shows too).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app.gui.panels import Panel
from app.gui.window import editors
from tests.correct_support import same_path
from tests.metrics_bar_support import text
from tests.qt_teardown import delete_widgets
from tests.qt_wait import settle
from tests.window_project_support import STEMS, enter, new_project, open_project, select
from tests.window_support import build_window, quiet_window
from tests.window_support import make_window as make_window  # fixture
from tests.window_support import mw as mw  # fixture
from tests.window_support import settings as settings  # fixture

USER = Qt.ItemDataRole.UserRole

pytestmark = pytest.mark.usefixtures("window_guards")


@pytest.fixture
def window(app, tmp_path):
    made = [build_window(tmp_path / "s.ini")]
    project = new_project(tmp_path)
    open_project(made[0], project)
    made[0].project = project
    yield made[0]
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def _selected(window) -> str:
    return window.file_list.currentItem().data(0, USER)


@pytest.mark.parametrize("back", ["Library", "Measure", "Generate"])
def test_after_paging_in_the_editor_the_display_shows_the_selected_image(window, back):
    select(window, 0)
    enter(window, "Correct")
    editor = window.correct_editor
    editor.navigate(1)  # PgDn: saves nothing (clean), opens image 1, moves the selection
    settle()
    assert same_path(editor.current_image_path, window.project.image(1))
    assert _selected(window) == STEMS[1]

    enter(window, back)

    assert window.right_panel.currentIndex() == Panel.DISPLAY
    assert window.display_controller.current_name == STEMS[1]
    assert text(window.metrics_bar, "name") == STEMS[1]


def test_back_from_an_editor_on_the_same_image_reads_the_metrics_again(window, tmp_path):
    select(window, 1)
    assert "Skeleton: GT" not in text(window.metrics_bar, "status")
    enter(window, "Correct")
    gt = Path(window.project.folder) / "gt" / "skeletons" / f"{STEMS[1]}.png"  # saved meanwhile
    gt.parent.mkdir(parents=True, exist_ok=True)
    gt.write_bytes(b"s")

    enter(window, "Library")

    assert window.display_controller.current_name == STEMS[1]
    assert "Skeleton: GT" in text(window.metrics_bar, "status")


def test_leaving_an_editor_that_refuses_stays_put(mw):
    from app.gui.panels import Panel

    editor = mw.correct_editor
    mw.right_panel.setCurrentWidget(editor)
    editor.maybe_save_changes = lambda: False
    assert mw.switch_right_panel(Panel.DISPLAY) is False
    assert mw.right_panel.currentWidget() is editor
    editor.maybe_save_changes = lambda: None  # legacy stub: not a refusal
    assert mw.switch_right_panel(Panel.DISPLAY) is True


def switch_window(calls: list, *, keeps_edits: bool) -> SimpleNamespace:
    editor = SimpleNamespace(maybe_save_changes=lambda: not keeps_edits)
    window = SimpleNamespace(
        right_panel=SimpleNamespace(
            currentIndex=lambda: Panel.CORRECT, setCurrentIndex=calls.append
        )
    )
    window._editors = lambda: {Panel.CORRECT: editor}
    return window


def test_an_unknown_page_is_logged_and_changes_nothing(caplog):
    calls = []
    with caplog.at_level(logging.WARNING, logger=editors.logger.name):
        assert editors.switch_right_panel(switch_window(calls, keeps_edits=False), "x") is True
    assert calls == [] and "Unknown right panel" in caplog.text
