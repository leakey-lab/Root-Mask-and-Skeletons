"""A save toast with a long file name stays on one line (KNOWN_ISSUES #7).

"Saved mask · <file name>" wrapped onto two lines when the name was long: the name fitted
on a line of its own, so it was not shortened, but the whole line did not fit. Now a
"<what> · <name>" toast shows its title and then the name (mono), elided in the middle (its
start and its extension stay) on the same line; only a name left under MIN_NAME_PX beside
its title takes a line of its own (2026-10-31 capsule). The full text stays in the tooltip.
Other messages wrap as before.
"""

from __future__ import annotations

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from app.gui.widgets import ToastManager
from tests.qt_teardown import delete_widgets

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


def _name(toast) -> QLabel:
    return toast.findChild(QLabel, "toastDetail")


def _one_line(toast) -> bool:
    title, name = _title(toast), _name(toast)
    title_box, name_box = title.geometry(), name.geometry()
    return (
        not title.wordWrap()
        and name_box.left() > title_box.right()
        and name_box.top() < title_box.bottom()  # the same row (centred, smaller font)
    )


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


def test_an_update_in_place_refits_the_name(host):
    manager = ToastManager(host)
    toast = manager.show("Saved mask · L44.jpg", key="saved")
    QApplication.processEvents()
    assert manager.show(f"Saved mask · {NAME}", key="saved") is toast
    QApplication.processEvents()

    assert _one_line(toast)
    assert _name(toast).text().endswith(".jpg") and "…" in _name(toast).text()
