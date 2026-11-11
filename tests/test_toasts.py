"""Toasts (``app.gui.widgets.toasts``): the capsule, its lifetime and the stack.

Lifetimes by kind: a success lasts 4 s, a warning at least 6 s, an error until closed (its
ring shows no countdown). The ring counts the lifetime down; hovering the stack or leaving the
application pauses every countdown, a toast shown meanwhile included. At most three toasts, a
danger one outlives newer successes; a toast with the same key updates in place and restarts.
Every toast closes from its button or a click; an action button runs once and closes it. The
stack keeps above a bottom offset, refits to a narrower host, and a toast without a manager
places itself bottom-right. Before (UXS-07): 14 "GT saved" toasts covered the window and an
error vanished after 3.2 s. Toasts nest no graphics effect (Qt draws nested effects wrongly).
Text fitting: ``tests/test_toasts_text.py``; the main window's toasts: area A10.
"""

from __future__ import annotations

import time

import pytest
from PyQt6.QtCore import QEvent, QPointF, QRect, Qt, qInstallMessageHandler
from PyQt6.QtGui import QEnterEvent, QGuiApplication
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from app.feedback import NoticeAction
from app.gui.widgets.toasts import (
    DEFAULT_TIMEOUT_MS,
    PEEK,
    Toast,
    ToastManager,
    toast_timeout,
)
from tests.qt_teardown import delete_widgets
from tests.qt_wait import settle, wait_until

LONG_ERROR = (
    "Cannot export: gt/classes.json is invalid ("
    + "class 'root_tip' colour #ffd166 duplicates class 'nodule'; " * 4
    + ")"
)


@pytest.fixture
def host(app):
    made = [QWidget()]
    made[0].resize(1024, 640)
    made[0].show()
    QTest.qWaitForWindowExposed(made[0])
    yield made[0]
    delete_widgets(made)


def _detail(toast) -> QLabel:
    return toast.findChild(QLabel, "toastDetail")


def _text(toast) -> str:
    return toast.findChild(QLabel, "toastTitle").text()


def _hover(widget, inside: bool) -> None:
    if inside:
        QApplication.sendEvent(widget, QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
    else:
        QApplication.sendEvent(widget, QEvent(QEvent.Type.Leave))


# ------------------------------------------------------------------------ the lifetime
def test_successes_last_four_seconds():
    assert DEFAULT_TIMEOUT_MS == 4000
    assert toast_timeout("success", None) == 4000
    assert toast_timeout("warn", None) == 6000
    assert toast_timeout("danger", 50) is None


def test_errors_stay_until_closed_while_successes_expire(host):
    manager = ToastManager(host)
    danger = manager.show("Could not save GT: disk full", "danger", timeout=50)
    success = manager.show("Mask saved", "success", timeout=50)
    QTest.qWait(600)
    assert danger in manager.toasts() and success not in manager.toasts()

    assert danger.close_button is not None and danger.close_button.isVisible()
    QTest.mouseClick(danger.close_button, Qt.MouseButton.LeftButton)
    assert danger not in manager.toasts()


def test_warnings_last_longer_and_can_be_closed(host):
    manager = ToastManager(host)
    warning = manager.show("GPU unavailable — using CPU", "warn", timeout=50)
    QTest.qWait(600)
    assert warning in manager.toasts()
    assert warning.close_button is not None


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


def test_a_toast_shown_while_the_application_is_in_the_background_waits_for_it(host):
    manager = ToastManager(host)
    signal = QGuiApplication.instance().applicationStateChanged
    try:
        signal.emit(Qt.ApplicationState.ApplicationInactive)
        toast = manager.show("Loaded 40 images", timeout=150)
        QTest.qWait(400)
        assert toast in manager.toasts(), "it counted down while nobody could see it"
    finally:
        signal.emit(Qt.ApplicationState.ApplicationActive)
    assert wait_until(lambda: toast not in manager.toasts(), 2.0)


def test_a_toast_shown_while_the_pointer_is_over_the_stack_waits_for_it_to_leave(host):
    manager = ToastManager(host)
    first = manager.show("Loaded 40 images", timeout=60000)
    _hover(first, True)
    toast = manager.show("Mask saved", timeout=150)
    QTest.qWait(400)
    assert toast in manager.toasts(), "it counted down under the pointer"

    _hover(first, False)
    assert wait_until(lambda: toast not in manager.toasts(), 2.0)


def test_an_update_in_place_restarts_the_countdown(host):
    manager = ToastManager(host)
    toast = manager.show("Saved mask · a.jpg", timeout=1000, key="saved")
    QTest.qWait(400)
    assert manager.show("Saved mask · b.jpg", timeout=1000, key="saved") is toast
    assert toast.ring.progress() == pytest.approx(1.0)
    assert _detail(toast).text() == "b.jpg"


# ----------------------------------------------------------------------- the stack
def test_a_burst_keeps_only_the_three_newest_toasts(host):
    manager = ToastManager(host)
    shown = [manager.show(f"GT saved for T001_L0{40 + i}.jpg") for i in range(14)]
    QApplication.processEvents()
    assert manager.toasts() == shown[-3:]


def test_a_danger_toast_outlives_newer_successes_when_the_stack_is_full(host):
    manager = ToastManager(host)
    danger = manager.show("Could not save GT: disk full", "danger")
    successes = [manager.show(f"Mask saved for {i}.jpg") for i in range(3)]
    assert manager.toasts() == [danger, *successes[1:]]


def test_toasts_with_the_same_key_update_in_place(host):
    manager = ToastManager(host)
    first = manager.show("GT saved for a.jpg", key="gt-saved")
    second = manager.show("GT saved for b.jpg", key="gt-saved")
    assert second is first
    assert manager.toasts() == [first]
    assert _text(first) == "GT saved for b.jpg"


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


def test_toasts_stay_above_the_bottom_offset(host):
    manager = ToastManager(host)
    manager.set_bottom_offset(74)  # e.g. the status bands of the main window
    toast = manager.show("Loaded 40 images")
    assert toast.geometry().bottom() < host.height() - 74

    offset = [200]
    manager.set_bottom_offset(lambda: offset[0])  # e.g. above the current editor's dock
    assert toast.geometry().bottom() < host.height() - 200


def test_a_toast_refits_when_the_host_narrows(host):
    manager = ToastManager(host)
    toast = manager.show(LONG_ERROR, "danger")
    settle()
    assert toast.width() > 300

    host.resize(320, 640)
    manager.reposition()  # what the window does on every resize
    settle()

    assert toast.width() <= 320 - 2 * manager.margin()
    assert host.rect().contains(toast.geometry())


def test_a_lone_toast_places_itself_bottom_right(host):
    toast = Toast(host, "Mask saved", kind="success", margin=16)
    toast.show_toast()
    box = toast.geometry()
    assert (box.right() + 1, box.bottom() + 1) == (host.width() - 16, host.height() - 16)


# ---------------------------------------------------------------- closing and the action
@pytest.mark.parametrize("kind", ["success", "info", "warn", "danger"])
def test_every_toast_has_a_close_button(host, kind):
    manager = ToastManager(host)
    toast = manager.show("Loaded 40 images", kind)
    assert toast.close_button is not None and toast.close_button.isVisibleTo(toast)
    QTest.mouseClick(toast.close_button, Qt.MouseButton.LeftButton)
    assert toast not in manager.toasts()


def test_a_click_on_a_toast_dismisses_it(host):
    manager = ToastManager(host)
    toast = manager.show("Loaded 40 images")
    QTest.mouseClick(toast, Qt.MouseButton.LeftButton)
    assert toast not in manager.toasts()


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


# ---------------------------------------------------------------- no nested graphics effect
def _pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()


def test_toast_frame_has_no_effect_under_the_fade():
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    toast = Toast(host, "Mask saved", "success")
    effects = [w.graphicsEffect() for w in toast.findChildren(QWidget) if w.graphicsEffect()]
    assert effects == [], "a child effect under the toast's opacity effect breaks painting"
    assert toast.graphicsEffect() is not None  # the fade itself is kept
    host.deleteLater()
    app.processEvents()


def test_partial_repaint_of_a_toast_over_a_sibling_logs_no_painter_warnings():
    app = QApplication.instance() or QApplication([])
    messages: list[str] = []
    previous = qInstallMessageHandler(lambda _mode, _ctx, msg: messages.append(msg))
    try:
        host = QWidget()
        host.resize(800, 600)
        under = QLabel("x" * 200, host)
        under.setGeometry(0, 0, 800, 600)
        host.show()
        _pump(app, 0.2)
        toast = ToastManager(host).show("hello", kind="info", timeout=5000)
        _pump(app, 0.4)
        g = toast.geometry()
        under.update(QRect(g.x() + 5, g.y() + 5, 20, 10))
        toast.update()
        _pump(app, 0.3)
        painter_msgs = [m for m in messages if "ainter" in m]
        assert painter_msgs == []
        host.close()
        host.deleteLater()
        app.processEvents()
    finally:
        qInstallMessageHandler(previous)
