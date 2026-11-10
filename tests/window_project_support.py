"""MainWindow tests on a project folder: the four-photo project of the Correct-stage window
tests, opening it, the library rows, the topbar's stages and the windows of those tests.

Moved here from ``test_correct_main_window.py`` (the project, the stage and row helpers,
``make_window`` and ``in_correct``), ``test_shell_keys_after_tree_focus.py`` (``click_row``,
``in_stage``) and ``test_library_main_window.py`` (its fixture ``mw``), which other test
modules borrowed. A test module takes a fixture by importing it by name
(``# noqa: F401 - fixture``).

Every project is a tmp_path folder of four 200x150 photos, the first with a model mask and a
generated skeleton; every window has its own INI settings, so the user's settings are never
read or written.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.gui.panels import Panel
from tests.correct_support import box_plane, same_path, write_image, write_mask, write_skeleton
from tests.qt_teardown import delete_widgets, flush_deleted
from tests.qt_wait import settle
from tests.qt_wait import wait_until as wait
from tests.window_support import build_window, open_folder, quiet_window

STEMS = tuple(f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in (21, 22, 23, 24))
STAGE_NAMES = ["Library", "Generate", "Correct", "Measure", "Visualize"]


# ============================================================================== helpers
@dataclass(frozen=True)
class Project:
    """A project folder: four photos; the first has a model mask and a generated skeleton."""

    folder: str

    def image(self, i: int) -> str:
        """The photo of image ``i``."""
        return os.path.join(self.folder, f"{STEMS[i]}.jpg")

    def mask(self, i: int) -> str:
        """The model's mask (the prediction) of image ``i``."""
        return os.path.join(self.folder, "mask", f"{STEMS[i]}.png")

    def gt_mask(self, i: int) -> str:
        """The saved (GT) mask of image ``i``: what a mask save writes."""
        return os.path.join(self.folder, "gt", "mask", f"{STEMS[i]}.png")

    def correction(self, i: int) -> str:
        """The saved (GT) skeleton of image ``i``: what a skeleton save writes."""
        return os.path.join(self.folder, "gt", "skeletons", f"{STEMS[i]}.png")

    def generated(self, i: int) -> str:
        """The generated skeleton of image ``i``."""
        return os.path.join(self.folder, "skeletons", f"{STEMS[i]}_fake.png")


def new_project(root) -> Project:
    """Four photos in ``root/project``; image 0 with a model mask and a generated skeleton."""
    project = Project(str(root / "project"))
    for i in range(len(STEMS)):
        write_image(project.image(i), value=60 + 20 * i)
    write_mask(project.mask(0), box_plane())
    write_skeleton(project.generated(0))
    return project


def open_project(window, project: Project) -> None:
    """Open ``project`` in ``window`` (the folder scan runs in a thread), wait until that
    load has ended, and expand the tree."""
    open_folder(window, project.folder, len(STEMS))
    window.file_list.expandAll()
    settle()


def leaf(window, i: int):
    """The library row of image ``i``."""
    return window.library_index.item_for(STEMS[i])


def select(window, i: int) -> None:
    """A click on the library row of image ``i``."""
    item = leaf(window, i)
    window.file_list.setCurrentItem(item)
    window.on_tree_item_clicked(item, 0)
    settle()


def enter(window, stage: str) -> None:
    """A click on the topbar's ``stage`` button."""
    window.ribbon_buttons[stage].click()
    settle()


def activate(window) -> None:
    """Make ``window`` the active window (keyboard shortcuts need it)."""
    window.show()
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window, 3000)
    settle()


# ============================================================================= fixtures
@pytest.fixture
def make_window(app, tmp_path):
    """``make(*, show=True)`` builds a MainWindow (1280x800) with its own INI settings;
    every window made is closed without prompts and deleted afterwards."""
    made = []

    def make(*, show: bool = True):
        window = build_window(tmp_path / f"settings_{len(made)}.ini", show=False)
        made.append(window)
        if show:
            window.show()
            settle()
        return window

    yield make
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def in_correct(make_window, tmp_path, image: int = 0, project: Project | None = None):
    """A shown window on a new project with image ``image`` open in the Correct stage."""
    project = project or new_project(tmp_path)
    window = make_window()
    open_project(window, project)
    select(window, image)
    enter(window, "Correct")
    assert window.right_panel.currentIndex() == Panel.CORRECT
    return window, project


# ======================================================= the library tree has the keyboard
def click_row(window, i: int) -> None:
    """A mouse click on the library row of image ``i`` (the tree takes the keyboard)."""
    tree = window.file_list
    item = leaf(window, i)
    tree.scrollToItem(item)
    settle()
    QTest.mouseClick(
        tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualItemRect(item).center()
    )
    settle()


def in_stage(window, stage: str, image: int = 1):
    """Enter ``stage``, click the image before ``image`` in the library, then step to
    ``image`` with the Down arrow: it opens, and the tree keeps the keyboard (by design,
    so the arrows keep browsing)."""
    enter(window, stage)
    activate(window)
    click_row(window, image - 1)
    window.file_list.setFocus()
    settle()
    QTest.keyClick(window.file_list, Qt.Key.Key_Down)
    editor = window._editors().get(window.right_panel.currentIndex())
    if editor is not None:
        path = window.project.image(image)
        assert wait(lambda: same_path(editor.current_image_path, path), 2.0)
    else:
        assert wait(lambda: window.display_controller.current_name == STEMS[image], 2.0)
    settle()
    assert window.file_list.hasFocus()
    assert window.file_list.currentItem() is leaf(window, image)
    return editor


# ============================================================ the Library tests' bare window
@pytest.fixture
def mw(app, monkeypatch):
    from app.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_record_recent_project", lambda self, count: None)
    w = MainWindow()
    monkeypatch.setattr(w.correct_editor, "maybe_save_changes", lambda: None)
    toasts = []
    monkeypatch.setattr(w, "notify", lambda msg, kind="success", **_k: toasts.append((kind, msg)))
    w.toasts_seen = toasts
    yield w
    w.close()
    w.deleteLater()
    del w
    flush_deleted()
