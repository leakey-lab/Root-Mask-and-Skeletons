"""What the window says about the open project: its title, the status line and the toasts
(once, alike for every layer of the editor; V3, V6).

- V3: a Measure run leaves "Saved <its CSV path>" on the status line for 15 s and a toast
  for a few seconds. Both survived opening another folder: the new project's status line
  named the previous project's CSV (while its Visualize page said nothing was measured)
  and its own Measure summary stayed hidden until the 15 s ran out. Opening a folder now
  drops the previous project's messages; a folder that is not opened changes nothing.
- V6: Correct's save toasts had no key, so each Ctrl+S stacked another identical toast,
  and Trace GT worded its own differently ("GT saved for X.jpg"). A save's toast now
  updates the live one in place, worded "Saved <what> · <image>" (the class labels too,
  since Trace GT became a layer of Correct); a failure's toast is never replaced by a
  later save's.
- The title bar said "Root Viewer" (the app's old name) whatever was open; it names the
  app and the open project.

Offscreen; every project is a tmp_path folder of four photos (tests.test_correct_main_window).
"""

from __future__ import annotations

import os

import pytest
from PyQt6.QtWidgets import QApplication

from app.config import ROOT_LENGTHS_CSV
from app.gui.correct.layer import CLASSES
from app.gui.panels import Panel
from app.handlers.deps import SAVED_PATH_STATUS_MS
from tests.correct_support import drag, paint
from tests.qt_teardown import delete_widgets
from tests.qt_wait import settle
from tests.qt_wait import wait_until as _wait
from tests.qt_wait import wait_until as wait
from tests.window_project_support import STEMS, enter, new_project, open_project, select
from tests.window_support import build_window, quiet_window, write_photos

pytestmark = pytest.mark.usefixtures("window_guards")

#: Toast lifetime in these tests: no toast expires while a test runs on a busy machine
#: (an expired toast would read as dropped, or a re-created one as stacked).
LONG_TOAST_MS = 120_000


@pytest.fixture
def window(app, tmp_path):
    """A shown 1280x800 window on ``tmp_path/project`` (``window.project``); its toasts
    keep their keys and stacking but last :data:`LONG_TOAST_MS`."""
    made = [build_window(tmp_path / "s.ini", show=False)]
    shown = made[0]
    real_show = shown.toasts.show
    shown.toasts.show = lambda message, kind="success", timeout=None, *, key=None, action=None: (
        real_show(message, kind, LONG_TOAST_MS, key=key, action=action)
    )
    shown.show()
    settle()
    shown.project = new_project(tmp_path)
    open_project(shown, shown.project)
    del shown
    yield made[0]
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def toast_texts(window) -> list[str]:
    """The live toasts' messages, oldest first."""
    return [toast.message() for toast in window.toasts.toasts()]


def saved_toasts(window) -> list:
    """The live toasts that report a save."""
    return [toast for toast in window.toasts.toasts() if toast.message().startswith("Saved")]


# ======================================================================== V3: stale messages
def measure_ran(window) -> str:
    """What a finished Measure run reports (``SkeletonHandler._on_finished``)."""
    csv_path = os.path.join(window.project.folder, "skeletons", ROOT_LENGTHS_CSV)
    window.feedback.status(f"Saved {csv_path}", SAVED_PATH_STATUS_MS)
    window.feedback.notify(f"Saved {ROOT_LENGTHS_CSV} in the skeletons folder", "success")
    window.feedback.notify("2 images could not be measured: see the Error column", "warn")
    settle()
    assert window.statusline_message.text() == f"Saved {csv_path}"
    return csv_path


def other_project(tmp_path):
    return new_project(tmp_path / "other")


def test_opening_another_folder_drops_the_previous_projects_messages(window, tmp_path):
    measure_ran(window)

    open_project(window, other_project(tmp_path))

    assert window.status_bar.currentMessage() == ""
    assert window.statusline_message.text() == f"Ready · {len(STEMS)} images"
    assert toast_texts(window) == [f"Loaded {len(STEMS)} images"]


def test_the_new_projects_measure_summary_shows_at_once(window, tmp_path):
    enter(window, "Measure")
    measure_ran(window)

    open_project(window, other_project(tmp_path))

    assert window.ribbon_buttons["Measure"].isChecked()
    assert wait(lambda: "not measured yet" in window.statusline_message.text())
    assert "Saved" not in window.statusline_message.text()


def test_the_loading_card_never_shows_the_previous_projects_path(window, tmp_path):
    """The messages go when the open starts, not when the scan ends: a large or network
    folder can take a while, and the status line under the card is visible meanwhile."""
    measure_ran(window)
    folder = other_project(tmp_path).folder
    window.image_manager.load_images = lambda _folder: None  # the scan has not ended yet

    window.open_path(folder)
    settle()

    assert window.status_bar.currentMessage() == ""
    assert "Saved" not in window.statusline_message.text()
    assert toast_texts(window) == []


def test_opening_a_folder_never_recounts_the_project_being_left(window, tmp_path):
    """Clearing the old message must not make the status line recount the old project for
    its idle text (on Generate that walks every image name: about 20 ms with 46k images,
    V8); it shows the load's progress instead, as before."""
    enter(window, "Generate")
    measure_ran(window)
    counted = []
    real = window.stage_status.text
    window.stage_status.text = lambda *a, **k: counted.append(a) or real(*a, **k)
    window.image_manager.load_images = lambda _folder: None  # the scan has not ended yet

    window.open_path(other_project(tmp_path).folder)

    assert counted == []
    assert window.status_bar.currentMessage() == ""


def test_a_folder_that_is_not_opened_changes_nothing(window, tmp_path):
    csv_path = measure_ran(window)
    shown = [text for text in toast_texts(window) if not text.startswith("Loaded")]

    window.open_path(str(tmp_path / "gone"))  # no such folder: reported, nothing else
    settle()

    assert window.status_bar.currentMessage() == f"Saved {csv_path}"
    # The load's own toast says it instead ("Loaded …" is replaced, W13); the rest stay.
    assert toast_texts(window) == [*shown, f"Folder not found: {tmp_path / 'gone'}"]


def test_an_editor_that_keeps_its_edits_keeps_the_messages_too(window, tmp_path):
    csv_path = measure_ran(window)
    shown = toast_texts(window)
    window.correct_editor.maybe_save_changes = lambda: False  # the user chose Cancel

    window.open_path(other_project(tmp_path).folder)
    settle()

    assert window.status_bar.currentMessage() == f"Saved {csv_path}"
    assert toast_texts(window) == shown


# ======================================================================== V6: save toasts
def in_correct(window, image: int):
    select(window, image)
    enter(window, "Correct")
    assert window.right_panel.currentIndex() == Panel.CORRECT
    return window.correct_editor


def test_saving_again_in_correct_updates_the_save_toast_in_place(window):
    editor = in_correct(window, 1)  # image 1 has no mask yet
    name = f"{STEMS[1]}.jpg"
    paint(editor, (20, 100), (60, 100))
    assert editor.save()
    settle()
    [first] = saved_toasts(window)
    assert first.message() == f"Saved mask · {name}"

    paint(editor, (20, 120), (60, 120))  # another edit: Ctrl+S saves only unsaved layers
    assert editor.save()
    settle()

    assert saved_toasts(window) == [first]
    assert first.message() == f"Saved mask · {name}"


def test_paging_on_after_each_edit_keeps_one_save_toast_naming_the_last_image(window):
    editor = in_correct(window, 1)
    paint(editor, (20, 100), (60, 100))
    editor.navigate(1)  # PgDn: saves, then opens image 2
    assert wait(lambda: editor.current_image_path.endswith(f"{STEMS[2]}.jpg"))
    paint(editor, (20, 120), (60, 120))
    editor.navigate(1)
    assert wait(lambda: editor.current_image_path.endswith(f"{STEMS[3]}.jpg"))
    settle()

    assert [toast.message() for toast in saved_toasts(window)] == [f"Saved mask · {STEMS[2]}.jpg"]


def test_a_class_labels_save_toast_is_worded_alike_and_updates_in_place(window):
    editor = in_correct(window, 1)
    editor.set_active_layer(CLASSES)
    drag(editor, (20, 100), (60, 100))
    assert editor.save()
    settle()
    [first] = saved_toasts(window)

    drag(editor, (20, 120), (60, 120))
    assert editor.save()
    settle()

    assert saved_toasts(window) == [first]
    assert first.message() == f"Saved class labels · {STEMS[1]}.jpg"


def test_a_failed_save_stays_beside_a_later_saves_toast(window):
    editor = in_correct(window, 1)
    editor.notice.emit("Could not save mask to X: disk full", "danger")
    paint(editor, (20, 100), (60, 100))

    assert editor.save()
    settle()

    assert toast_texts(window)[-2:] == [
        "Could not save mask to X: disk full",
        f"Saved mask · {STEMS[1]}.jpg",
    ]


def test_other_editor_notices_still_stack(window):
    """Only a save's summary is keyed: distinct notices (here two of them) each show."""
    window.correct_editor.notice.emit("Mask shown", "info")
    window.correct_editor.notice.emit("Skeleton shown", "info")
    settle()

    assert toast_texts(window)[-2:] == ["Mask shown", "Skeleton shown"]


# ============================================================================== the title


def _folder(root, name: str, count: int) -> str:
    stems = [f"24EFCAM1_T001_L0{10 + i}_2024.07.18_1115{10 + i}_001_SMM" for i in range(count)]
    return str(write_photos(root / name, stems))


def test_the_window_title_names_the_app_and_the_project(app, tmp_path, no_prompts):
    """The title bar said "Root Viewer" (the app's old name) whatever was open."""
    window = build_window(tmp_path / "t.ini", size=None, show=False)
    try:
        assert window.windowTitle() == "SPROUTS[*]"
        window.open_path(_folder(tmp_path, "trial_B", 2))
        assert _wait(lambda: len(window.library_index.leaves()) == 2)
        assert window.windowTitle() == "trial_B — SPROUTS[*]"
    finally:
        delete_widgets([window])
