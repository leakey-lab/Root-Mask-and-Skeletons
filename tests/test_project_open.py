"""Opening a project folder (``app/gui/window/project_loader.py``): what is refused, what a
load shows, and how an empty, failed or partly failed load ends.

A missing folder, a running batch job (R7, O15) and an editor that keeps its edits change
nothing; a folder without images keeps what was loaded (and returns to Welcome when nothing
was); one keyed "Loaded N images" toast per load, a second one for the files the Library
does not list (O6, O7); a scan superseded by a newer one never delivers.

Offscreen; every project is a tmp_path folder of tiny synthetic photos.
"""

from __future__ import annotations

import os
import sys

import pytest
from PIL import Image
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.handlers.deps import HandlerDependencies
from app.project_layout import ImageScan
from tests.qt_teardown import delete_widgets
from tests.qt_wait import wait_until as _wait
from tests.window_project_support import mw as mw  # fixture
from tests.window_support import build_window, open_folder, quiet_window, write_photos

pytestmark = pytest.mark.usefixtures("no_temp_sweep")

STEMS = [f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in (11, 12, 13, 14, 15)]


def _load(mw, folder):
    """Open ``folder`` (all of STEMS) and wait until the load has ended."""
    open_folder(mw, folder, len(STEMS))
    return folder


def test_missing_folder_keeps_the_library(mw, tmp_path, monkeypatch):
    _load(mw, write_photos(tmp_path / "p", STEMS))
    loads = []
    monkeypatch.setattr(mw.image_manager, "load_images", loads.append)
    mw.open_path(str(tmp_path / "moved"))
    assert loads == []
    assert len(mw._image_leaves()) == len(STEMS)
    assert mw.toasts_seen[-1][0] == "danger" and "not found" in mw.toasts_seen[-1][1]


def test_image_less_folder_warns_and_keeps_the_library(mw, tmp_path):
    _load(mw, write_photos(tmp_path / "p", STEMS))
    empty = tmp_path / "empty"
    empty.mkdir()
    mw.open_path(str(empty))
    assert _wait(lambda: mw.toasts_seen and mw.toasts_seen[-1][0] == "warn")
    assert len(mw._image_leaves()) == len(STEMS)
    assert os.path.normcase(mw.image_manager.original_folder) == os.path.normcase(
        str(tmp_path / "p")
    )
    assert not any(kind == "success" and "0 image" in msg for kind, msg in mw.toasts_seen)


def test_image_less_first_folder_returns_to_welcome(mw, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    mw.open_path(str(empty))
    assert _wait(lambda: mw.toasts_seen and mw.toasts_seen[-1][0] == "warn")
    assert mw.app_stack.currentIndex() == 0


def test_load_success_is_one_formatted_message(mw, tmp_path):
    _load(mw, write_photos(tmp_path / "p", STEMS))
    assert [m for k, m in mw.toasts_seen if k == "success"] == ["Loaded 5 images"]


def test_reopening_never_terminates_and_drops_the_stale_scan(mw, tmp_path, monkeypatch):
    a = write_photos(tmp_path / "a", STEMS[:2])
    b = write_photos(tmp_path / "b", STEMS[2:])
    monkeypatch.setattr(QThread, "terminate", lambda self: pytest.fail("terminate() called"))
    finished = []
    real = mw.on_loading_finished
    monkeypatch.setattr(mw, "on_loading_finished", lambda *a: finished.append(1) or real(*a))
    mw.image_manager.load_images(str(a))
    mw.image_manager.load_images(str(b))
    assert _wait(lambda: finished)
    QTest.qWait(150)
    assert finished == [1]
    assert sorted(mw.image_manager.images) == sorted(STEMS[2:])


def test_open_path_is_refused_while_a_job_runs(mw, tmp_path, monkeypatch):
    _load(mw, write_photos(tmp_path / "a", STEMS))
    # The busy registry is undone before the fixture closes the window: closing with a
    # job running asks, in a modal box, whether to stop it.
    with monkeypatch.context() as patch:
        patch.setattr(mw.jobs, "is_busy", lambda: True)
        patch.setattr(mw.jobs, "running_name", lambda: "Generating masks")
        mw.task_progress.start("Generating masks", on_cancel=lambda: None)
        mw.open_path(str(write_photos(tmp_path / "b", STEMS[:2])))
        QTest.qWait(50)
    assert (
        "warn",
        "Generating masks is running. Cancel it or wait before opening another folder.",
    ) in mw.toasts_seen
    assert os.path.normcase(mw.image_manager.original_folder) == os.path.normcase(
        str(tmp_path / "a")
    )
    assert mw.task_progress._operation == "Generating masks"
    assert not mw.task_progress.cancel_button.isHidden()


def test_the_folder_dialog_is_not_opened_while_a_job_runs(mw, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    with monkeypatch.context() as patch:  # undone before the window closes (as above)
        patch.setattr(mw.jobs, "is_busy", lambda: True)
        patch.setattr(mw.jobs, "running_name", lambda: "Generating masks")
        patch.setattr(
            QFileDialog,
            "getExistingDirectory",
            lambda *a, **k: pytest.fail("the folder dialog opened during a job"),
        )
        mw.load_images()
    assert mw.toasts_seen[-1][0] == "warn"


@pytest.fixture
def window(app, tmp_path):
    made = [build_window(tmp_path / "s.ini", show=False)]
    made[0].show()
    yield made[0]
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def _open(window, folder, count) -> None:
    open_folder(window, folder, count)
    QApplication.processEvents()


def _load_toasts(window) -> list[str]:
    return [t.message() for t in window.toasts.toasts() if "Loaded" in t.message()]


def test_opening_the_same_folder_twice_shows_one_loaded_toast(window, tmp_path):
    folder = write_photos(tmp_path / "p", STEMS[:3])
    _open(window, folder, 3)
    _open(window, folder, 3)

    assert _load_toasts(window) == ["Loaded 3 images"]


def test_a_second_folder_updates_the_loaded_toast_in_place(window, tmp_path):
    _open(window, write_photos(tmp_path / "p", STEMS[:3]), 3)
    _open(window, write_photos(tmp_path / "q", STEMS[:1]), 1)

    assert _load_toasts(window) == ["Loaded 1 image"]


def _images(folder, spec):
    """``spec``: ``{file name: grey level}``; every image 64x48 RGB."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, level in spec.items():
        Image.new("RGB", (64, 48), (level, level, level)).save(folder / name)
    return str(folder)


def test_the_load_toast_names_hidden_and_unsupported_files(mw, tmp_path):  # noqa: F811 - fixture
    folder = tmp_path / "p"
    _images(folder, {"A.png": 200, "a.jpg": 0, "b.png": 100, "t.tif": 50})

    mw.open_path(str(folder))
    assert _wait(lambda: any(kind == "warn" for kind, _ in mw.toasts_seen))
    assert (
        "warn",
        "1 image is not listed: a.jpg has the same name as A.png. 1 file (.tif) is not "
        "supported and is not listed; convert it to PNG or JPEG",
    ) in mw.toasts_seen
    assert [m for k, m in mw.toasts_seen if k == "success"] == ["Loaded 2 images"]
    assert len(mw._image_leaves()) == 2


def test_a_folder_of_tiffs_says_why_nothing_loaded(mw, tmp_path):  # noqa: F811 - fixture
    _images(tmp_path / "tifs", {"x.tif": 0, "y.bmp": 0})

    mw.open_path(str(tmp_path / "tifs"))
    assert _wait(lambda: mw.toasts_seen and mw.toasts_seen[-1][0] == "warn")
    assert mw.toasts_seen[-1] == (
        "warn",
        "No .png/.jpg/.jpeg images in tifs — nothing loaded. 2 files (.bmp, .tif) are not "
        "supported and are not listed; convert them to PNG or JPEG",
    )


def _boom(*_args, **_kwargs):
    raise RuntimeError("adoption failed")


@pytest.mark.parametrize("where", ["manager", "tree"])
def test_a_failed_adoption_loads_nothing_and_leaves_jobs_startable(
    mw,  # noqa: F811 - fixture
    tmp_path,
    monkeypatch,
    where,
):
    # Since W1-T1 a folder load refuses every Generate / Measure run until it ends; an
    # exception while the manager adopts the scan, or while the tree is built, must still
    # end it, and never leave the Library on one project and the jobs' folder on another.
    from app.gui.image_manager import ImageManager, project_layout_of

    _images(tmp_path / "old", {"a.png": 90, "b.png": 90})
    mw.open_path(str(tmp_path / "old"))
    assert _wait(lambda: len(mw._image_leaves()) == 2)
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    if where == "manager":
        monkeypatch.setattr(ImageManager, "on_loading_finished", _boom)
    else:  # the new folder's tree fails; an empty one still builds
        real = mw.populate_file_list
        monkeypatch.setattr(
            mw, "populate_file_list", lambda: _boom() if mw.image_manager.images else real()
        )
    _images(tmp_path / "new", {"c.png": 90})

    mw.open_path(str(tmp_path / "new"))
    assert _wait(lambda: any(kind == "danger" for kind, _ in mw.toasts_seen))
    assert project_layout_of(mw.image_manager) is None and mw.image_manager.images == {}
    assert mw._image_leaves() == [] and mw.metrics.index is None
    assert mw.app_stack.currentIndex() == 0  # nothing loaded: back to Welcome
    assert mw._loading_project is False
    assert HandlerDependencies(mw).refuse_while_loading() is False
    assert errors == []


def test_a_failure_after_the_library_is_shown_keeps_the_folder_loaded(
    mw,  # noqa: F811 - fixture
    tmp_path,
    monkeypatch,
):
    from app.gui.image_manager import project_layout_of

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    monkeypatch.setattr(mw.metrics, "prewarm", _boom)
    folder = tmp_path / "p"
    _images(folder, {"a.png": 90})

    mw.open_path(str(folder))
    assert _wait(lambda: any(kind == "success" for kind, _ in mw.toasts_seen))
    assert [m for k, m in mw.toasts_seen if k == "success"] == ["Loaded 1 image"]
    assert not any(kind == "danger" for kind, _ in mw.toasts_seen)
    layout = project_layout_of(mw.image_manager)
    assert os.path.normcase(layout.root) == os.path.normcase(str(folder))
    assert len(mw._image_leaves()) == 1 and not mw.loading_overlay.isVisible()
    assert mw._loading_project is False and errors == []


def test_the_skipped_text_names_at_most_three_hidden_files():
    from app.gui.window.project_loader import skipped_text

    three = (("a.jpg", "A.png"), ("b.JPG", "B.png"), ("c.jpeg", "C.png"))
    assert skipped_text(ImageScan({}, hidden=three)) == (
        "3 images are not listed: each has the same name as another image (a.jpg, b.JPG, c.jpeg)"
    )
    four = (*three, ("d.jpg", "D.png"))
    assert skipped_text(ImageScan({}, hidden=four)).endswith("(a.jpg, b.JPG, c.jpeg, …)")
    assert skipped_text(ImageScan({}, unsupported=("a.TIF", "b.bmp", "c.tif"))) == (
        "3 files (.bmp, .tif) are not supported and are not listed; convert them to PNG or JPEG"
    )
    assert skipped_text(ImageScan({"a": "a.png"})) == ""


# ============================================================== a second project replaces it
def test_a_second_project_shows_none_of_the_first_projects_values(mw, tmp_path):
    """A stem both projects hold opens the second project's photo, and the metrics bar shows
    the second project's measured length for it, never the first's (the bar read the index of
    the project that was open before)."""
    from pathlib import Path

    from app.config import LENGTH_CSV_HEADERS
    from tests.measure_support import _row, _write_lengths

    shared, only_a = STEMS[0], STEMS[1]
    a = write_photos(tmp_path / "a", [shared, only_a])
    b = write_photos(tmp_path / "b", [shared])
    _write_lengths(Path(a), [_row(shared, LENGTH_CSV_HEADERS[5], 11.25)])
    _write_lengths(Path(b), [_row(shared, LENGTH_CSV_HEADERS[5], 22.5)])
    length = mw.metrics_bar._length_value  # the bar's LENGTH value (no public reader)

    def show(stem):
        item = mw.library_index.item_for(stem)
        mw.file_list.setCurrentItem(item)
        mw.on_image_selected(item)

    open_folder(mw, a, 2)
    show(shared)
    assert _wait(lambda: "11.25" in length.text()), length.text()

    open_folder(mw, b, 1)
    assert [leaf.data(0, Qt.ItemDataRole.UserRole) for leaf in mw.library_index.leaves()] == [
        shared
    ]
    show(shared)
    assert os.path.normcase(mw.display_controller.current_image) == os.path.normcase(
        str(b / f"{shared}.jpg")
    )
    assert _wait(lambda: "22.5" in length.text()), length.text()
    assert "11.25" not in length.text()
    assert mw.windowTitle().startswith("b — ")


# ============================================================== the folder dialog (Ctrl+O)
@pytest.fixture
def persisting_window(app, tmp_path):
    """A window with its own INI settings (it remembers the last folder)."""
    made = [build_window(tmp_path / "s.ini", show=False)]
    yield made[0]
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def test_the_folder_dialog_starts_at_the_last_folder_and_opens_the_choice(
    persisting_window, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QFileDialog

    window = persisting_window
    last = write_photos(tmp_path / "last", STEMS[:1])
    chosen = write_photos(tmp_path / "chosen", STEMS[:2])
    window.window_state.set_last_folder(str(last))
    asked = []
    answer = [str(chosen)]
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda parent, title, start: asked.append(start) or answer[0],
    )
    opened = []
    monkeypatch.setattr(window, "open_path", opened.append)

    window.load_images()
    answer[0] = ""  # Cancel
    window.load_images()

    assert [os.path.normcase(p) for p in asked] == [os.path.normcase(str(last))] * 2
    assert opened == [str(chosen)]


# ============================================================== a recent project that is gone
def test_a_recent_project_deleted_since_the_welcome_was_built_says_so(persisting_window, tmp_path):
    import shutil

    window = persisting_window
    gone = write_photos(tmp_path / "gone", STEMS[:1])
    window.recent_projects.add(str(gone), 1)
    window.welcome.refresh_recents()
    (row,) = window.welcome.recent_rows()
    assert _wait(lambda: not row.is_missing() and row.open_button.isEnabled())
    shutil.rmtree(gone)

    row.open_button.click()

    assert window.app_stack.currentIndex() == 0  # Welcome stays
    messages = [toast.message() for toast in window.toasts.toasts()]
    assert f"Folder not found: {gone}" in messages
    assert _wait(lambda: window.welcome.recent_rows()[0].is_missing())


# ============================================================== a row without an image
def test_a_row_whose_image_left_the_project_opens_nothing(mw, tmp_path):
    _load(mw, write_photos(tmp_path / "p", STEMS))
    first = mw.library_index.item_for(STEMS[0])
    mw.file_list.setCurrentItem(first)
    mw.on_image_selected(first)
    assert mw.display_controller.current_name == STEMS[0]
    del mw.image_manager.images[STEMS[1]]  # its file left the project (a rescan is pending)

    mw.on_image_selected(mw.library_index.item_for(STEMS[1]))

    assert mw.display_controller.current_name == STEMS[0]
    assert mw.metrics_bar._name_value.text() == STEMS[0]
