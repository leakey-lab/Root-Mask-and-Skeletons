"""Browsing the Library in the window: arrows, PgDn / PgUp, paging across groups, and the
order the Correct page gets from it.

The arrows load the image they come to rest on; PgDn / PgUp step from the image on screen
(after a click on a tube, date or field row: the image the display shows, X6-01) and reveal
the opened row even while the tree animates an expand (H1-01). The Correct page reads its
neighbours and its "Image i / n" place from the library order.

Offscreen; every project is a tmp_path folder of tiny photos.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QAbstractItemView, QApplication

from app.gui.library_index import USER_ROLE
from tests.qt_teardown import delete_widgets
from tests.qt_wait import wait_until as _wait
from tests.window_project_support import mw as mw  # fixture
from tests.window_support import build_window, open_folder, quiet_window, write_photos

pytestmark = pytest.mark.usefixtures("no_temp_sweep")

USER = Qt.ItemDataRole.UserRole
#: The five photos of the bare-window (``mw``) tests.
FIVE_STEMS = [f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in (11, 12, 13, 14, 15)]


def _load(mw, folder):
    """Open ``folder`` (all of FIVE_STEMS) and wait until the load has ended."""
    open_folder(mw, folder, len(FIVE_STEMS))
    return folder


def _select(mw, stem):
    leaf = mw.find_tree_item_by_image_name(stem)
    mw.file_list.setCurrentItem(leaf)
    mw.on_image_selected(leaf)
    return leaf


def test_view_mode_switch_keeps_tree_selection_and_filter(mw, tmp_path, monkeypatch):
    import app.gui.file_tree_manager as ftm

    _load(mw, write_photos(tmp_path / "p", FIVE_STEMS))
    leaf = _select(mw, FIVE_STEMS[3])
    mw.file_list.expandAll()
    calls = []
    monkeypatch.setattr(ftm, "populate_file_list", lambda m: calls.append(m))
    for i in (1, 2, 0):
        mw.view_mode_combo.setCurrentIndex(i)
    assert calls == []
    assert mw.file_list.currentItem() is leaf
    assert leaf.parent().isExpanded()


def test_down_arrow_in_library_shows_next_image(mw, tmp_path):
    _load(mw, write_photos(tmp_path / "p", FIVE_STEMS))
    mw.show()
    mw.file_list.expandAll()
    _select(mw, FIVE_STEMS[0])
    mw.file_list.setFocus()
    QTest.keyClick(mw.file_list, Qt.Key.Key_Down)
    QTest.keyClick(mw.file_list, Qt.Key.Key_Down)
    assert _wait(lambda: mw.display_controller.current_name == FIVE_STEMS[2], 2.0)


def test_pgdn_in_correct_opens_next_image(mw, tmp_path, monkeypatch):
    folder = _load(mw, write_photos(tmp_path / "p", FIVE_STEMS))
    loaded = []
    monkeypatch.setattr(
        mw.correct_editor, "load_image", lambda path, images_base_folder=None: loaded.append(path)
    )
    _select(mw, FIVE_STEMS[0])
    mw.toggle_correct()
    mw.show()
    mw.activateWindow()
    assert QTest.qWaitForWindowActive(mw, 3000)
    view = mw.correct_editor.graphics_view
    view.setFocus()
    QTest.keyClick(view, Qt.Key.Key_PageDown)
    assert loaded[-1] == str(folder / f"{FIVE_STEMS[1]}.jpg")
    assert mw.file_list.currentItem().data(0, USER) == FIVE_STEMS[1]
    QTest.keyClick(mw.file_list, Qt.Key.Key_PageUp)
    assert mw.file_list.currentItem().data(0, USER) == FIVE_STEMS[0]


STEMS = [f"24EFCAM1_T001_L0{d}_2024.07.18_1115{d}_001_SMM" for d in range(11, 17)]


@pytest.fixture
def window(app, tmp_path, no_prompts):
    folder = tmp_path / "project"
    folder.mkdir()
    for i, stem in enumerate(STEMS):
        image = np.random.default_rng(i).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        assert cv2.imwrite(str(folder / f"{stem}.jpg"), image)
    made = [build_window(tmp_path / "settings.ini", show=False)]
    made[0].show()
    open_folder(made[0], folder, len(STEMS))
    yield made[0]
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def _in_correct(mw, stem):
    """Open ``stem`` from the library on the Correct page; returns its chip's position text."""
    item = mw.library_index.item_for(stem)
    mw.file_list.setCurrentItem(item)
    mw.on_image_selected(item)
    mw.toggle_correct()
    QApplication.processEvents()
    return mw.correct_editor.status_chip.position_label.toolTip()


def test_the_correct_page_gets_the_library_order(window):
    mw = window
    editor = mw.correct_editor
    path = mw.image_manager.get_image_path(STEMS[2])

    assert _in_correct(mw, STEMS[2]) == f"Image 3 of {len(STEMS)} in the library order"
    assert editor.paging.position(path) == (3, len(STEMS))
    # The neighbour provider has no public getter: its one read shows the library's order.
    assert list(editor.paging._neighbours(path)) == [
        mw.image_manager.get_image_path(STEMS[3]),
        mw.image_manager.get_image_path(STEMS[1]),
    ]


def test_a_generation_run_drops_what_the_correct_page_read_ahead(window, monkeypatch):
    mw = window
    calls = []
    monkeypatch.setattr(mw.correct_editor, "discard_read_ahead", lambda: calls.append(1))
    mw.on_outputs_changed()
    assert calls == [1]


def test_a_search_moves_the_correct_pages_position(window):
    mw = window
    chip = mw.correct_editor.status_chip.position_label
    assert _in_correct(mw, STEMS[1]) == f"Image 2 of {len(STEMS)} in the library order"

    mw.library_search.setText("_L012_")
    mw.library_search_controller.apply_now()
    QApplication.processEvents()
    assert chip.toolTip() == "Image 1 of 1 in the library order"

    mw.library_search.setText("")  # clearing applies at once
    QApplication.processEvents()
    assert chip.toolTip() == f"Image 2 of {len(STEMS)} in the library order"


# One image in tubes 1 and 2, two in tube 3: each PgDn from tube 1 enters another tube.
GROUPED = [
    "24EFCAM1_T001_L011_2024.07.18_111511_001_SMM",
    "24EFCAM1_T002_L011_2024.07.18_111512_001_SMM",
    "24EFCAM1_T003_L011_2024.07.18_111513_001_SMM",
    "24EFCAM1_T003_L012_2024.07.18_111514_001_SMM",
]


@pytest.fixture
def grouped_window(app, tmp_path, no_prompts):
    folder = tmp_path / "grouped"
    folder.mkdir()
    for i, stem in enumerate(GROUPED):
        image = np.random.default_rng(i).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        assert cv2.imwrite(str(folder / f"{stem}.jpg"), image)
    made = [build_window(tmp_path / "settings.ini", show=False)]
    made[0].show()
    open_folder(made[0], folder, len(GROUPED))
    yield made[0]
    delete_widgets(made, prepare=quiet_window)
    QApplication.processEvents()


def _shown_in_tree(tree, item) -> bool:
    """The row is in the tree's layout: every group above it is open."""
    parent = item.parent()
    while parent is not None:
        if not parent.isExpanded():
            return False
        parent = parent.parent()
    return tree.visualItemRect(item).isValid()


def test_fast_paging_across_collapsed_groups_reveals_each_opened_row(grouped_window):
    # The tree animates opening a group (250 ms), and Qt's scrollToItem opens no group
    # while an animation runs: the second fast PgDn used to leave its row hidden (H1-01).
    mw = grouped_window
    tree = mw.file_list
    assert tree.isAnimated()
    tree.collapseAll()
    leaves = mw.library_index.leaves()
    for expected in leaves[:3]:  # three tube changes, no pause between them
        mw._page_image(1)
        assert mw.display_controller.current_name == expected.data(0, USER_ROLE)
        assert tree.currentItem() is expected
        assert _shown_in_tree(tree, expected)

    # The next arrow key steps on from the opened row (it used to jump to the top).
    tree.setFocus()
    QTest.keyClick(tree, Qt.Key.Key_Down)
    assert tree.currentItem() is leaves[3]


def _settled(tree) -> None:
    """Wait out the tree's expand animation (a click during it can miss its row)."""
    assert _wait(lambda: tree.state() == QAbstractItemView.State.NoState)


def _click_row(tree, item) -> None:
    parent = item.parent()
    while parent is not None:
        parent.setExpanded(True)
        parent = parent.parent()
    _settled(tree)
    tree.scrollToItem(item)
    _settled(tree)
    rect = tree.visualItemRect(item)
    assert rect.isValid()
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    QApplication.processEvents()


def _open(mw, item) -> None:
    mw.file_list.setCurrentItem(item)
    mw.on_image_selected(item)


def test_pgdn_after_a_group_click_goes_on_from_the_image_on_screen(grouped_window):
    # X6-01 (owner: A, as Correct does): PgDn used to open tube 1's image.
    mw = grouped_window
    leaves = mw.library_index.leaves()
    _open(mw, leaves[1])
    tube = leaves[2].parent().parent()
    _click_row(mw.file_list, tube)
    assert mw.file_list.currentItem() is tube

    mw._page_image(1)

    assert mw.display_controller.current_name == GROUPED[2]
    assert mw.file_list.currentItem() is leaves[2]


def test_pgup_after_a_group_click_goes_on_from_the_image_on_screen(grouped_window):
    # X6-01: PgUp used to do nothing from a group row.
    mw = grouped_window
    leaves = mw.library_index.leaves()
    _open(mw, leaves[2])
    for row in (leaves[0].parent(), leaves[0].parent().parent()):  # a date, then a tube row
        _click_row(mw.file_list, row)
        assert mw.file_list.currentItem() is row

        mw._page_image(-1)

        assert mw.display_controller.current_name == GROUPED[1]
        assert mw.file_list.currentItem() is leaves[1]
        _open(mw, leaves[2])


def test_paging_from_a_group_row_with_nothing_on_screen_opens_the_first_image(grouped_window):
    # Nothing shown yet: PgDn starts at the first image, as it does from no selection.
    mw = grouped_window
    tube = mw.library_index.leaves()[2].parent().parent()
    _click_row(mw.file_list, tube)
    assert mw.display_controller.current_name is None

    mw._page_image(1)

    assert mw.display_controller.current_name == GROUPED[0]
