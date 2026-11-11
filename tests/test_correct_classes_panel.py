"""The class panel on the Correct page: its boxes pick the class the brush paints.

Owner decision 2026-11-01: the one ticked box is the active class, on the highlighted row, in
the brush's colour. A click (row or box) never takes the keyboard from the canvas, so Space
still pans; Tab reaches the boxes and Space on a focused box picks its class and keeps the
focus; a screen reader's toggle picks the class too; the class keys work from a focused box;
painting never ticks or locks a box. The Tab order of the page is ``test_correct_tab_order.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
)

from app.gt import (
    default_registry,
)
from app.gui.correct.layer import CLASSES
from tests.correct_classes_support import (
    KEYS,
    Focus,
    assert_one_tick,
    box,
    ids,
    paint,
    pixels,
    press,
)
from tests.correct_classes_support import confirm as confirm  # fixture
from tests.correct_classes_support import layout as layout  # fixture
from tests.correct_classes_support import made as made  # fixture
from tests.correct_classes_support import make_editor as make_editor  # fixture
from tests.correct_classes_support import notices as notices  # fixture
from tests.correct_classes_support import page as page  # fixture
from tests.correct_classes_support import project_bad_registry as project_bad_registry  # fixture
from tests.correct_classes_support import project_lossy as project_lossy  # fixture
from tests.correct_classes_support import prompts as prompts  # fixture
from tests.correct_classes_support import reads as reads  # fixture
from tests.correct_classes_support import signals as signals  # fixture
from tests.correct_classes_support import themed as themed  # fixture
from tests.correct_classes_support import window_with_project as window_with_project  # fixture
from tests.correct_support import (
    drag,
    settle,
)

pytestmark = pytest.mark.usefixtures("correct_guards")


def test_a_click_on_the_class_panel_never_takes_the_focus(page):
    panel = page.class_panel
    for widget in (panel, *panel.findChildren(QWidget)):
        assert not widget.focusPolicy() & Focus.ClickFocus, widget
    assert panel._gear.focusPolicy() == Focus.TabFocus
    for key in (*KEYS, "ignore"):
        assert box(page, key).focusPolicy() == Focus.TabFocus
        assert panel._rows[key].focusPolicy() == Focus.NoFocus


def test_space_on_a_focused_box_picks_its_class_and_the_focus_stays_for_the_next_one(page):
    nodule, root_tip = box(page, "nodule"), box(page, "root_tip")
    nodule.setFocus(Qt.FocusReason.TabFocusReason)

    QTest.keyClick(nodule, Qt.Key.Key_Space)

    assert_one_tick(page, "nodule")
    assert not page.has_unsaved_changes() and nodule.hasFocus()
    QTest.keyClick(nodule, Qt.Key.Key_Tab)
    assert root_tip.hasFocus()
    QTest.keyClick(root_tip, Qt.Key.Key_Space)
    assert_one_tick(page, "root_tip")
    QTest.keyClick(root_tip, Qt.Key.Key_Space)  # the ticked box again: it stays ticked
    assert_one_tick(page, "root_tip")
    assert root_tip.hasFocus()


def test_a_click_on_a_box_hands_the_keys_back_to_the_canvas(page):
    view = page.graphics_view
    box(page, "root_tip").setFocus(Qt.FocusReason.TabFocusReason)
    nodule = box(page, "nodule")

    QTest.mouseClick(nodule, Qt.MouseButton.LeftButton)

    assert_one_tick(page, "nodule")
    assert QApplication.focusWidget() is view
    QTest.keyPress(view, Qt.Key.Key_Space)  # Space pans; it never presses a box
    assert view.space_pan
    QTest.keyRelease(view, Qt.Key.Key_Space)
    assert_one_tick(page, "nodule")


def test_a_mouse_click_on_the_focused_box_hands_the_keys_back_to_the_canvas(page):
    """A click is a click even on the box that holds the keys: the next Space pans."""
    view = page.graphics_view
    nodule = box(page, "nodule")
    nodule.setFocus(Qt.FocusReason.TabFocusReason)

    QTest.mouseClick(nodule, Qt.MouseButton.LeftButton)

    assert_one_tick(page, "nodule")
    assert QApplication.focusWidget() is view
    QTest.keyPress(view, Qt.Key.Key_Space)
    assert view.space_pan
    QTest.keyRelease(view, Qt.Key.Key_Space)
    root_tip = box(page, "root_tip")
    root_tip.setFocus(Qt.FocusReason.TabFocusReason)
    QTest.mouseClick(page.class_panel._rows["root_tip"], Qt.MouseButton.LeftButton)  # its row
    assert_one_tick(page, "root_tip")
    assert QApplication.focusWidget() is view


def test_a_toggle_of_a_box_picks_its_class_and_leaves_the_keys(page):
    """A screen reader's Toggle action (``toggle()``): the class is picked as by Space, the
    keys stay where the reader has them, and the active box is never unticked."""
    nodule = box(page, "nodule")
    nodule.setFocus(Qt.FocusReason.TabFocusReason)

    nodule.toggle()

    assert_one_tick(page, "nodule")
    assert QApplication.focusWidget() is nodule
    nodule.toggle()  # the active box: put back
    assert_one_tick(page, "nodule")
    assert not page.has_unsaved_changes()


def test_a_row_click_moves_the_tick_and_the_brush_together(page):
    """Owner bug 2026-11-01: Root nodule was ticked, Tip's row was clicked, and the brush
    painted Tip while the tick stayed on Root nodule. One state drives both now."""
    panel = page.class_panel
    QTest.mouseClick(box(page, "nodule"), Qt.MouseButton.LeftButton)
    assert_one_tick(page, "nodule")

    QTest.mouseClick(panel._rows["root_tip"], Qt.MouseButton.LeftButton)

    assert_one_tick(page, "root_tip")
    paint(page, (20, 40), (120, 40))
    assert set(np.unique(ids(page)).tolist()) == {0, default_registry().by_key("root_tip").id}


def test_a_click_on_a_box_paints_with_its_class(page):
    assert_one_tick(page, "root")  # the first class, as the page opens

    QTest.mouseClick(box(page, "nodule"), Qt.MouseButton.LeftButton)

    assert_one_tick(page, "nodule")
    paint(page, (20, 40), (120, 40))
    assert set(np.unique(ids(page)).tolist()) == {0, default_registry().by_key("nodule").id}


def test_a_click_on_the_ticked_box_keeps_it_ticked(page):
    root = box(page, "root")
    assert_one_tick(page, "root")

    for _ in range(2):  # not an off switch
        QTest.mouseClick(root, Qt.MouseButton.LeftButton)
        assert_one_tick(page, "root")
    assert not page.has_unsaved_changes()


def test_the_class_keys_move_the_tick(page):
    for key, class_key in (("2", "nodule"), ("3", "root_tip"), ("0", "ignore"), ("1", "root")):
        press(page, key)
        assert_one_tick(page, class_key)  # key 0: the Ignore row's box


@pytest.mark.parametrize("checked", [False, True], ids=["unchecked", "checked"])
def test_the_focused_box_draws_a_focus_ring(themed, page, checked):
    nodule = box(page, "nodule")
    if checked:
        QTest.mouseClick(nodule, Qt.MouseButton.LeftButton)
        assert nodule.isChecked()
    page.graphics_view.setFocus()
    before = pixels(nodule)

    nodule.setFocus(Qt.FocusReason.TabFocusReason)
    assert nodule.hasFocus()
    after = pixels(nodule)

    assert (before != after).any(axis=2).mean() > 0.1


def test_a_focused_box_stays_usable_when_its_class_gets_pixels(page):
    root, nodule = box(page, "root"), box(page, "nodule")
    nodule.setFocus(Qt.FocusReason.TabFocusReason)

    drag(page, (20, 20))  # a root dab: the boxes neither move nor lock

    assert_one_tick(page, "root")
    assert root.isEnabled() and nodule.isEnabled()
    assert QApplication.focusWidget() is nodule
    QTest.keyClick(nodule, Qt.Key.Key_Space)
    assert_one_tick(page, "nodule")


def test_the_class_keys_still_work_from_a_focused_box(page):
    nodule = box(page, "nodule")
    nodule.setFocus(Qt.FocusReason.TabFocusReason)

    QTest.keyClick(nodule, Qt.Key.Key_3)

    assert_one_tick(page, "root_tip")
    assert not nodule.isChecked()


def test_the_tick_follows_the_class_painted_and_painting_locks_no_box(page):
    """Painting never ticks or locks a box (owner bug report 2026-10-30: every painted class
    was ticked at once): the one tick is the class the brush paints (owner decision
    2026-11-01)."""
    rows = page.class_panel._rows
    for key, y in (("root", 40), ("root_tip", 80)):
        page.layers[CLASSES].set_active_class(key)
        drag(page, (20, y), (120, y))
        settle(page)
        assert_one_tick(page, key)

    assert all(row.checkbox.isEnabled() for row in rows.values())
    rows["nodule"].checkbox.click()
    assert_one_tick(page, "nodule")
