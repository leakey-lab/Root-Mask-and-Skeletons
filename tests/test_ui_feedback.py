"""The Qt feedback port (``app/gui/ui_feedback.py`` ``QtUiFeedback``) and the window's use of it.

Promises: with a total, progress counts items; without one it is a percent; an explicit
total wins; Cancel reaches the handler; a failure shows on the row and as a danger toast;
``on_task_end`` runs after every finished or failed task. In the window, ``feedback`` drives
the one status surface, and a handler's notice reaches the toasts with its button.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QPushButton, QStatusBar

from app.feedback import NoticeAction, UiFeedback
from app.gui.task_progress import TaskProgressWidget
from app.gui.ui_feedback import QtUiFeedback
from app.gui.widgets.toasts import DEFAULT_TIMEOUT_MS
from tests.qt_teardown import delete_widgets
from tests.window_support import make_window as make_window  # fixture
from tests.window_support import mw as mw  # fixture
from tests.window_support import no_delayed_temp_sweep as no_delayed_temp_sweep  # fixture
from tests.window_support import settings as settings  # fixture


@pytest.fixture
def parts(app):
    tp = TaskProgressWidget()
    bar = QStatusBar()
    toasts = []
    refreshed = []
    fb = QtUiFeedback(
        tp,
        status_bar=bar,
        notifier=lambda msg, kind: toasts.append((msg, kind)),
        on_task_end=lambda: refreshed.append(1),
    )
    return fb, tp, bar, toasts, refreshed


def test_is_a_ui_feedback(parts):
    assert isinstance(parts[0], UiFeedback)


def test_counts_mode(parts):
    fb, tp, *_ = parts
    fb.task_started("Generating masks", total=46_178)
    fb.task_progress(1_234)
    assert tp.label.text().startswith("Generating masks — 1,234 / 46,178")
    assert tp.bar.value() == 2


def test_percent_mode(parts):
    fb, tp, *_ = parts
    fb.task_started("Calculating root length")
    fb.task_progress(42)
    assert tp.label.text() == "Calculating root length — 42%"


def test_explicit_total_wins(parts):
    fb, tp, *_ = parts
    fb.task_started("x", total=10)
    fb.task_progress(5, 20)
    assert tp.bar.value() == 25


def test_cancel_reaches_the_handler(parts):
    fb, tp, *_ = parts
    cancelled = []
    fb.task_started("Generating masks", cancelable=True, on_cancel=lambda: cancelled.append(1))
    tp.cancel_button.click()
    assert cancelled == [1]


def test_failure_is_shown_and_toasted(parts):
    fb, tp, _bar, toasts, refreshed = parts
    fb.task_started("x")
    fb.task_failed("Model weights missing")
    assert tp.label.text() == "Model weights missing"
    assert toasts == [("Model weights missing", "danger")]
    assert refreshed == [1]


def test_finish_status_and_notify(parts):
    fb, tp, bar, toasts, refreshed = parts
    fb.task_started("x")
    fb.task_finished("Done")
    assert tp.label.text() == "Done"
    fb.status("Saved", 1000)
    assert bar.currentMessage() == "Saved"
    fb.notify("Hi", "info")
    assert toasts == [("Hi", "info")]
    assert refreshed == [1]


def test_feedback_drives_the_one_status_surface(mw):
    from app.gui.ui_feedback import QtUiFeedback

    assert isinstance(mw.feedback, QtUiFeedback)
    assert mw.status_bar.isHidden()  # the statusline is the only band
    mw.feedback.status("Hello from a handler")
    assert mw.statusline_message.text() == "Hello from a handler"
    mw.feedback.task_started("Generating masks", total=10)
    mw.feedback.task_progress(4)
    assert "4 / 10" in mw.task_progress.label.text()
    mw.feedback.task_finished("Done")


@pytest.fixture
def window(app, tmp_path, no_prompts, no_file_dialogs, no_temp_sweep, gl_off):
    from tests.window_support import build_window, quiet_window

    made = [build_window(tmp_path / "s.ini")]
    yield made[0]
    delete_widgets(made, prepare=quiet_window)


def test_the_window_toasts_last_the_default_lifetime(window):
    window.notify("Loaded 40 images")
    assert window.toasts.toasts()[-1].lifetime() == DEFAULT_TIMEOUT_MS


def test_a_handler_notice_reaches_the_window_with_its_button(window):
    ran = []
    window.feedback.notify(
        "Skeletons saved to the skeletons folder",
        "success",
        action=NoticeAction("Open folder", lambda: ran.append(True)),
    )
    toast = window.toasts.toasts()[-1]
    button = toast.findChild(QPushButton, "toastAction")
    assert button is not None and button.text() == "Open folder"
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert ran == [True]
