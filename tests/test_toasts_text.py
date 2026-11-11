"""How a toast fits its text (``app.gui.widgets.toasts``).

A save's message "<what> · <file name>" splits into a title and the file name (mono), shown on
one line with the name elided in the middle (its start and its extension stay); only a name
left under MIN_NAME_PX beside its title takes a line of its own (KNOWN_ISSUES #7: "Saved mask
· <long name>" wrapped onto two lines). Any other message wraps at 460 px inside the window;
an unbreakable path is elided in the middle. The full text is always the tooltip and the
accessible name.
"""

from __future__ import annotations

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from app.gui.widgets import ToastManager
from app.gui.widgets.toasts import split_message
from tests.qt_teardown import delete_widgets

NAME = "24EFCAM1_T001_L044_2024.07.18_111530_001_SMM.jpg"
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


def _title(toast) -> QLabel:
    return toast.findChild(QLabel, "toastTitle")


def _name(toast) -> QLabel:
    return toast.findChild(QLabel, "toastDetail")


def _text(toast) -> str:
    return _title(toast).text()


def _one_line(toast) -> bool:
    title, name = _title(toast), _name(toast)
    title_box, name_box = title.geometry(), name.geometry()
    return (
        not title.wordWrap()
        and name_box.left() > title_box.right()
        and name_box.top() < title_box.bottom()  # the same row (centred, smaller font)
    )


# --------------------------------------------------------------------- the message
def test_a_save_message_splits_into_title_and_file_name():
    assert split_message(f"Saved mask · {NAME}") == ("Saved mask", NAME)


@pytest.mark.parametrize(
    "message",
    ["Loaded 40 images", "Branch deleted · Ctrl+Z to undo", " · L44.jpg", "Saved mask · "],
)
def test_other_messages_stay_whole(message):
    assert split_message(message) == (message, None)


# ------------------------------------------------------------------- a save's file name
# "mask and skeleton" is left out: the offscreen platform's wide font leaves its name
# under MIN_NAME_PX beside its title (the app's font leaves it about 230 px).
@pytest.mark.parametrize("what", ["mask", "GT"])
def test_a_long_file_name_is_elided_in_the_middle_on_one_line(host, what):
    message = f"Saved {what} · {NAME}"
    toast = ToastManager(host).show(message)
    QApplication.processEvents()

    assert _title(toast).text() == f"Saved {what}"
    name = _name(toast).text()
    assert name.startswith("24EF") and name.endswith(".jpg") and "…" in name
    assert _one_line(toast)
    assert toast.width() <= 460
    assert host.rect().contains(toast.geometry())
    assert toast.toolTip() == message
    assert toast.accessibleName() == message


def test_a_short_name_is_left_whole(host):
    toast = ToastManager(host).show("Saved mask · L44.jpg")
    QApplication.processEvents()

    assert (_title(toast).text(), _name(toast).text()) == ("Saved mask", "L44.jpg")
    assert _one_line(toast)


def test_a_name_with_no_room_left_beside_its_title_takes_its_own_line(host):
    message = f"Saved the mask, the skeleton and the ground truth tracing of this image · {NAME}"
    toast = ToastManager(host).show(message)
    QApplication.processEvents()

    assert _title(toast).text().startswith("Saved the mask, the skeleton")
    assert _name(toast).geometry().top() >= _title(toast).geometry().bottom()
    assert _name(toast).text().endswith(".jpg")


def test_an_update_in_place_refits_the_name(host):
    manager = ToastManager(host)
    toast = manager.show("Saved mask · L44.jpg", key="saved")
    QApplication.processEvents()
    assert manager.show(f"Saved mask · {NAME}", key="saved") is toast
    QApplication.processEvents()

    assert _one_line(toast)
    assert _name(toast).text().endswith(".jpg") and "…" in _name(toast).text()


# ------------------------------------------------------------------- other messages
def test_a_sentence_still_wraps(host):
    message = f"Could not save mask for {NAME}: disk full"
    toast = ToastManager(host).show(message, "danger")
    QApplication.processEvents()

    assert _title(toast).wordWrap()
    assert _title(toast).text().endswith(": disk full")
    assert not _name(toast).isVisibleTo(toast)


def test_several_words_after_the_dot_still_wrap(host):
    message = "Branch deleted · " + "press Ctrl+Z to undo the deletion " * 3
    toast = ToastManager(host).show(message.strip(), "info")
    QApplication.processEvents()

    assert _title(toast).wordWrap()
    assert "…" not in _title(toast).text()
    assert not _name(toast).isVisibleTo(toast)


def test_long_messages_wrap_inside_the_window(host):
    manager = ToastManager(host)
    toast = manager.show(LONG_ERROR, "danger")
    QApplication.processEvents()
    assert host.rect().contains(toast.geometry())
    assert toast.width() <= 460
    assert toast.accessibleName() == LONG_ERROR


def test_unbreakable_paths_are_elided_with_the_full_text_in_the_tooltip(host):
    manager = ToastManager(host)
    path = "D:/" + "/".join(["a_very_long_folder_name"] * 12) + "/mask/T001_L044.png"
    message = f"Could not save mask to {path}"
    toast = manager.show(message, "danger")
    QApplication.processEvents()
    assert host.rect().contains(toast.geometry())
    assert "…" in _text(toast)
    assert toast.toolTip() == message
