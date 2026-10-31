"""The capsule toast (2026-10-31 owner choice, Sonner + Coloplast).

A pill with a countdown ring around the kind glyph, the title and (for a save) the file name
in mono on one line, a close button on every toast and at most one action button. Newer
toasts stack in front of older ones, which tuck behind it until the pointer is over the
stack; hovering or leaving the application pauses every countdown.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QEnterEvent, QGuiApplication
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from app.feedback import NoticeAction
from app.gui.widgets.toasts import (
    DEFAULT_TIMEOUT_MS,
    PEEK,
    ToastManager,
    split_message,
    toast_timeout,
)
from tests.qt_teardown import delete_widgets
from tests.qt_wait import settle, wait_until

NAME = "24EFCAM1_T001_L044_2024.07.18_111530_001_SMM.jpg"


@pytest.fixture
def host(app):
    made = [QWidget()]
    made[0].resize(1024, 640)
    made[0].show()
    QTest.qWaitForWindowExposed(made[0])
    yield made[0]
    delete_widgets(made)


def _title(toast) -> QLabel:
    return toast.findChild(QLabel, "toastTitle")


def _detail(toast) -> QLabel:
    return toast.findChild(QLabel, "toastDetail")


def _hover(widget, inside: bool) -> None:
    if inside:
        QApplication.sendEvent(widget, QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
    else:
        QApplication.sendEvent(widget, QEvent(QEvent.Type.Leave))


# --------------------------------------------------------------------- the message
def test_a_save_message_splits_into_title_and_file_name():
    assert split_message(f"Saved mask · {NAME}") == ("Saved mask", NAME)


@pytest.mark.parametrize(
    "message",
    ["Loaded 40 images", "Branch deleted · Ctrl+Z to undo", " · L44.jpg", "Saved mask · "],
)
def test_other_messages_stay_whole(message):
    assert split_message(message) == (message, None)


def test_successes_last_four_seconds():
    assert DEFAULT_TIMEOUT_MS == 4000
    assert toast_timeout("success", None) == 4000
    assert toast_timeout("warn", None) == 6000
    assert toast_timeout("danger", 50) is None


# ------------------------------------------------------------------------ the look
def test_a_save_shows_its_title_and_file_name_on_one_line(host):
    toast = ToastManager(host).show("Saved mask · L44.jpg")
    settle()

    assert _title(toast).text() == "Saved mask"
    assert _detail(toast).text() == "L44.jpg" and _detail(toast).isVisibleTo(toast)
    assert not _title(toast).wordWrap()
    assert toast.height() == 40  # one line: a pill
    assert toast.message() == toast.toolTip() == toast.accessibleName() == "Saved mask · L44.jpg"


# File-name fitting: tests/test_small_texts_2026_10_toasts.py (KNOWN_ISSUES #7).


@pytest.mark.parametrize("kind", ["success", "info", "warn", "danger"])
def test_every_toast_has_a_close_button(host, kind):
    manager = ToastManager(host)
    toast = manager.show("Loaded 40 images", kind)
    assert toast.close_button is not None and toast.close_button.isVisibleTo(toast)
    QTest.mouseClick(toast.close_button, Qt.MouseButton.LeftButton)
    assert toast not in manager.toasts()


# ------------------------------------------------------------------- the countdown
def test_the_ring_counts_down_the_toast_lifetime(host):
    toast = ToastManager(host).show("Loaded 40 images", timeout=1000)
    assert toast.ring.progress() == pytest.approx(1.0)
    QTest.qWait(350)
    assert 0.0 < toast.ring.progress() < 0.85


def test_an_error_ring_shows_no_countdown(host):
    toast = ToastManager(host).show("Could not save mask", "danger")
    QTest.qWait(100)
    assert toast.ring.progress() is None


def test_hovering_the_stack_pauses_the_countdown(host):
    manager = ToastManager(host)
    toast = manager.show("Loaded 40 images", timeout=150)
    _hover(toast, True)
    QTest.qWait(400)
    assert toast in manager.toasts()

    _hover(toast, False)
    assert wait_until(lambda: toast not in manager.toasts(), 2.0)


def test_leaving_the_application_pauses_the_countdown(host):
    manager = ToastManager(host)
    toast = manager.show("Loaded 40 images", timeout=150)
    signal = QGuiApplication.instance().applicationStateChanged
    try:
        signal.emit(Qt.ApplicationState.ApplicationInactive)
        QTest.qWait(400)
        assert toast in manager.toasts()
    finally:
        signal.emit(Qt.ApplicationState.ApplicationActive)
    assert wait_until(lambda: toast not in manager.toasts(), 2.0)


def test_an_update_in_place_restarts_the_countdown(host):
    manager = ToastManager(host)
    toast = manager.show("Saved mask · a.jpg", timeout=1000, key="saved")
    QTest.qWait(400)
    assert manager.show("Saved mask · b.jpg", timeout=1000, key="saved") is toast
    assert toast.ring.progress() == pytest.approx(1.0)
    assert _detail(toast).text() == "b.jpg"


# ----------------------------------------------------------------------- the stack
def test_older_toasts_tuck_behind_the_newest(host):
    manager = ToastManager(host)
    older, old, front = (manager.show(text) for text in ("one", "two", "Loaded 40 images"))
    settle()

    assert [t.tucked() for t in (older, old, front)] == [True, True, False]
    f = front.geometry()
    for depth, toast in ((1, old), (2, older)):
        g = toast.geometry()
        assert g.top() == f.top() - PEEK * depth
        assert g.height() == f.height()
        assert g.width() < f.width() and f.left() < g.left() and g.right() < f.right()


def test_hovering_the_stack_fans_it_out(host):
    manager = ToastManager(host)
    toasts = [manager.show(text) for text in ("one", "two", "Loaded 40 images")]
    settle()
    _hover(toasts[-1], True)
    settle()

    assert not any(t.tucked() for t in toasts)
    boxes = [t.geometry() for t in toasts]
    assert all(not a.intersects(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    assert boxes[0].bottom() < boxes[1].top() and boxes[1].bottom() < boxes[2].top()

    _hover(toasts[-1], False)
    assert wait_until(lambda: toasts[0].tucked(), 2.0)


def test_the_stack_height_is_the_fanned_out_height(host):
    """Clearance keeps room for the stack as it is on hover, so it does not jump."""
    manager = ToastManager(host, gap=10)
    toasts = [manager.show(text) for text in ("one", "two", "Loaded 40 images")]
    settle()
    assert manager.stack_height() == sum(t.natural_size().height() for t in toasts) + 2 * 10


def test_a_new_toast_slides_up_into_place(host):
    manager = ToastManager(host, motion=True)
    toast = manager.show("Loaded 40 images")
    QApplication.processEvents()
    start = toast.geometry()
    assert wait_until(lambda: toast.geometry().bottom() == host.height() - 16 - 1, 2.0)
    assert start.top() > toast.geometry().top()


def test_without_motion_a_toast_is_placed_at_once(host):
    toast = ToastManager(host, motion=False).show("Loaded 40 images")
    assert toast.geometry().bottom() == host.height() - 16 - 1


# ---------------------------------------------------------------------- the action
def test_the_action_button_runs_once_and_closes_the_toast(host):
    ran = []
    manager = ToastManager(host)
    toast = manager.show(
        "Skeletons saved to the skeletons folder",
        action=NoticeAction("Open folder", lambda: ran.append(True)),
    )
    button = toast.findChild(QPushButton, "toastAction")
    assert button is not None and button.text() == "Open folder"
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert ran == [True]
    assert toast not in manager.toasts()


def test_a_toast_without_an_action_has_no_action_button(host):
    toast = ToastManager(host).show("Loaded 40 images")
    assert toast.findChild(QPushButton, "toastAction") is None


# --------------------------------------------------------------- the feedback port
def test_the_feedback_port_hands_a_notice_its_action(app):
    from app.gui.task_progress import TaskProgressWidget
    from app.gui.ui_feedback import QtUiFeedback

    seen = []
    feedback = QtUiFeedback(
        TaskProgressWidget(),
        notifier=lambda message, kind, action=None: seen.append((message, kind, action)),
    )
    action = NoticeAction("Open folder", lambda: None)
    feedback.notify("Skeletons saved to the skeletons folder", "success", action=action)
    assert seen == [("Skeletons saved to the skeletons folder", "success", action)]


def test_a_notifier_without_actions_still_gets_plain_notices(app):
    from app.gui.task_progress import TaskProgressWidget
    from app.gui.ui_feedback import QtUiFeedback

    seen = []
    feedback = QtUiFeedback(
        TaskProgressWidget(), notifier=lambda message, kind: seen.append((message, kind))
    )
    feedback.notify("Loaded 40 images")
    assert seen == [("Loaded 40 images", "success")]


def test_the_recording_double_keeps_the_action():
    from app.feedback import RecordingFeedback

    feedback = RecordingFeedback()
    action = NoticeAction("Open folder", lambda: None)
    feedback.notify("Saved", "success", action=action)
    feedback.notify("Loaded")
    assert feedback.calls[0].kwargs == {"action": action}
    assert feedback.calls[1].kwargs == {}


# ------------------------------------------------------------------- the main window
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


def test_the_lifetime_follows_the_kind(host):
    manager = ToastManager(host)
    assert manager.show("Loaded 40 images").lifetime() == DEFAULT_TIMEOUT_MS
    assert manager.show("GPU unavailable", "warn", timeout=50).lifetime() == 6000
    assert manager.show("Could not save mask", "danger").lifetime() is None
