"""Tests for the GT class panel and class editor widgets (Task 3).

Covers ``app.gui.class_panel.ClassPanel`` (row rendering, active selection, the one box
ticked on the active class, keyboard-shortcut lookup, scrolling/height cap) and
``app.gui.class_editor_dialog.ClassEditorDialog`` (add/rename/recolour/retire
against the real ``Registry`` API, inline validation errors, accept/cancel).

Runs under an offscreen QApplication (``QT_QPA_PLATFORM=offscreen``).
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QFrame,
    QScrollArea,
    QWidget,
)

from app.gt.registry import default_registry
from app.gui.class_editor_dialog import ClassEditorDialog
from app.gui.class_panel import ClassPanel
from app.gui.widgets import tokens
from tests.qt_teardown import delete_widgets

_MADE: list[QWidget] = []  # the hosts and dialogs a test made: deleted after it


@pytest.fixture(autouse=True)
def _delete_test_widgets(app):
    """Close and delete the test's hosts and dialogs, and drop every dead wrapper, before
    the next test runs (see tests/qt_teardown.py)."""
    yield
    delete_widgets(_MADE)


def _host(w=400, h=600):
    host = QWidget()
    _MADE.append(host)
    host.resize(w, h)
    return host


def _dialog(registry):
    dialog = ClassEditorDialog(registry)
    _MADE.append(dialog)
    return dialog


def _registry_with_many_classes(n_labelled: int = 12):
    reg = default_registry()
    for i in range(n_labelled - len(reg.labelled())):
        reg = reg.add(key=f"extra_{i}", name=f"Extra {i}", color=f"#{100 + i:06x}")
    return reg


# --------------------------------------------------------------------------- #
#  ClassPanel
# --------------------------------------------------------------------------- #
def test_panel_shows_one_row_per_active_class_in_order(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())

    assert list(panel._rows.keys()) == ["root", "nodule", "root_tip", "ignore"]
    # Every row has a box, Ignore too (it can be the active class: key 0).
    tips = {key: row.checkbox.toolTip() for key, row in panel._rows.items()}
    assert tips == {
        "root": "Paint with this class (key 1)",
        "nodule": "Paint with this class (key 2)",
        "root_tip": "Paint with this class (key 3)",
        "ignore": "Paint with this class (key 0)",
    }


def test_clicking_a_row_emits_active_changed_and_highlights(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())

    got = []
    panel.active_changed.connect(got.append)
    QTest.mouseClick(panel._rows["nodule"], Qt.MouseButton.LeftButton)

    assert got == ["nodule"]
    assert panel.active_key() == "nodule"
    assert panel._rows["nodule"]._active is True

    # set_active() must not emit.
    got.clear()
    panel.set_active("root")
    assert got == []
    assert panel.active_key() == "root"
    assert panel._rows["root"]._active is True
    assert panel._rows["nodule"]._active is False


def _highlighted(panel) -> list[str]:
    return [key for key, row in panel._rows.items() if row._active]


def test_the_tick_and_the_highlight_are_the_active_class_alone(app):
    """Owner bug 2026-11-01: Root nodule was ticked while Tip was highlighted (and painted).
    The tick and the highlight now come from the active key alone."""
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    assert panel.ticked_keys() == [] == _highlighted(panel)  # no class set yet

    panel.set_active("nodule")
    assert panel.ticked_keys() == ["nodule"] == _highlighted(panel)

    got = []
    panel.active_changed.connect(got.append)
    QTest.mouseClick(panel._rows["root_tip"], Qt.MouseButton.LeftButton)  # Tip's row

    assert got == ["root_tip"] and panel.active_key() == "root_tip"
    assert panel.ticked_keys() == ["root_tip"] == _highlighted(panel)


def test_a_click_on_a_box_picks_its_class_and_is_the_only_tick(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("root")
    got = []
    panel.active_changed.connect(got.append)

    QTest.mouseClick(panel._rows["nodule"].checkbox, Qt.MouseButton.LeftButton)

    assert got == ["nodule"] and panel.active_key() == "nodule"
    assert panel.ticked_keys() == ["nodule"] == _highlighted(panel)
    assert panel._rows["nodule"].checkbox.isEnabled() is True  # a box is never locked


def test_a_click_on_the_ticked_box_keeps_it_ticked(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("nodule")
    got = []
    panel.active_changed.connect(got.append)

    for _ in range(2):  # not an off switch: a second click changes nothing either
        QTest.mouseClick(panel._rows["nodule"].checkbox, Qt.MouseButton.LeftButton)
        assert panel.ticked_keys() == ["nodule"] == _highlighted(panel)
    assert got == ["nodule", "nodule"]


def test_the_ignore_box_can_be_the_tick(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("root")

    QTest.mouseClick(panel._rows["ignore"].checkbox, Qt.MouseButton.LeftButton)

    assert panel.active_key() == "ignore"
    assert panel.ticked_keys() == ["ignore"] == _highlighted(panel)


def test_set_active_moves_the_one_tick_silently(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    got = []
    panel.active_changed.connect(got.append)

    for key in ("root", "ignore", "root_tip", "root_tip"):
        panel.set_active(key)
        assert panel.ticked_keys() == [key] == _highlighted(panel)
    assert got == []


def test_set_active_of_the_shown_class_changes_nothing(app):
    """Every stroke and undo puts the active class on the panel again: the box it shows
    must not untick and tick again (two toggles, two restyles each time)."""
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("nodule")
    toggles = []
    for row in panel._rows.values():
        row.checkbox.toggled.connect(lambda on, k=row.key: toggles.append((k, on)))

    panel.set_active("nodule")
    assert toggles == []

    panel.set_active("root")
    assert sorted(toggles) == [("nodule", False), ("root", True)]


def test_a_toggle_never_changes_a_box_and_a_toggle_on_picks_its_class(app):
    """Qt's accessible Toggle action (Narrator, NVDA) calls toggle(), which skips the
    click path: the active box stays ticked, and another box picks its class."""
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("root")
    got = []
    panel.active_changed.connect(lambda k: got.append((k, panel.picked_by_key())))

    panel._rows["root"].checkbox.toggle()  # off: put back, nothing asked
    assert panel.ticked_keys() == ["root"] == _highlighted(panel) and got == []
    panel._rows["root"].checkbox.setChecked(False)
    assert panel.ticked_keys() == ["root"] and got == []

    panel._rows["nodule"].checkbox.toggle()  # on: asks for the class, which ticks it alone

    assert got == [("nodule", True)] and panel.active_key() == "nodule"
    assert panel.ticked_keys() == ["nodule"] == _highlighted(panel)
    assert panel.picked_by_key() is False  # only while the pick is handled


def test_picked_by_key_tells_space_from_a_click(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    panel.set_active("root")
    got = []
    panel.active_changed.connect(lambda k: got.append((k, panel.picked_by_key())))
    nodule, root_tip = panel._rows["nodule"].checkbox, panel._rows["root_tip"].checkbox

    QTest.mouseClick(nodule, Qt.MouseButton.LeftButton)
    QTest.keyClick(root_tip, Qt.Key.Key_Space)
    QTest.mouseClick(panel._rows["root"], Qt.MouseButton.LeftButton)

    assert got == [("nodule", False), ("root_tip", True), ("root", False)]
    assert panel.ticked_keys() == ["root"]


def test_one_tick_after_the_classes_change(app):
    """A rebuild keeps the active class ticked; a retired active class leaves no stale tick
    (the page then sets the class it falls back to)."""
    host = _host()
    panel = ClassPanel(host)
    registry = default_registry()
    panel.set_registry(registry)
    panel.set_active("nodule")

    registry = registry.add(key="hypha", name="Hypha", color="#3fa7ff")
    panel.set_registry(registry)
    assert panel.ticked_keys() == ["nodule"] == _highlighted(panel)

    panel.set_registry(registry.retire("nodule"))
    assert panel.ticked_keys() == [] == _highlighted(panel)
    panel.set_active("root")
    assert panel.ticked_keys() == ["root"] == _highlighted(panel)


def test_key_for_shortcut_defaults(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())

    assert panel.key_for_shortcut(1) == "root"
    assert panel.key_for_shortcut(3) == "root_tip"
    assert panel.key_for_shortcut(0) == "ignore"
    assert panel.key_for_shortcut(9) is None


def test_panel_many_classes_scrolls(app):
    host = _host(400, 600)
    panel = ClassPanel(host)
    # Enough rows that the natural (unscrolled) content height genuinely
    # exceeds the 60%-of-600 = 360px cap, so this exercises real clipping,
    # not just an assertion that happens to hold for a short list.
    reg = _registry_with_many_classes(25)
    panel.set_registry(reg)

    assert isinstance(panel._scroll, QScrollArea)
    assert panel._scroll.widget() is panel._rows_container
    assert len(panel._rows) == 26  # 25 labelled + Ignore

    panel.reposition()
    cap = int(host.height() * 0.6)
    assert panel.height() <= cap
    assert panel._rows_container.sizeHint().height() > panel._scroll.height(), (
        "content must be taller than the visible scroll viewport for this to be a real test "
        "of scrolling/clipping, not a no-op cap"
    )
    assert panel.key_for_shortcut(9) == reg.labelled()[8].key


def test_panel_shrinks_back_after_fewer_classes(app):
    """Regression: QScrollArea.sizeHint() caches the contained widget's size
    and does not reliably invalidate without a real show/resize event cycle,
    so a naive adjustSize()-based reposition() can get stuck at the capped
    height even after the registry shrinks back down (e.g. retiring classes
    via the editor). reposition() must measure the rows container directly.
    """
    host = _host(400, 600)
    panel = ClassPanel(host)
    panel.set_registry(_registry_with_many_classes(25))
    panel.reposition()
    grown_height = panel.height()

    panel.set_registry(default_registry())
    panel.reposition()
    assert panel.height() < grown_height
    assert panel.height() <= int(host.height() * 0.6)


def _relative_luminance(color: QColor) -> float:
    def channel(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(color.red())
        + 0.7152 * channel(color.green())
        + 0.0722 * channel(color.blue())
    )


def _contrast(a: QColor, b: QColor) -> float:
    la, lb = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_contrast_helper_matches_wcag_reference_values():
    assert round(_contrast(QColor("#000000"), QColor("#ffffff")), 1) == 21.0
    assert round(_contrast(QColor("#777777"), QColor("#ffffff")), 2) == 4.48


def test_class_panel_paints_its_own_background(app):
    """UXE-04: the rows used to be drawn straight on the soil image (no panel fill)."""
    host = _host(900, 600)
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    host.show()
    try:
        QTest.qWaitForWindowExposed(host)
        panel.reposition()
        # A pixel inside the 6 px content margin, clear of the rounded corners.
        pixel = panel.grab().toImage().pixelColor(3, panel.height() // 2)
        assert pixel.alpha() == 255
        assert pixel.name() == QColor(tokens.BG_1).name()
    finally:
        host.close()


def test_class_key_hints_are_readable_on_the_panel(app):
    """UXE-04: the 1-9/0 hints need WCAG AA contrast (4.5:1) on the panel fill."""
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    hint = panel._rows["root"]._number_label
    hint.ensurePolished()
    color = hint.palette().color(QPalette.ColorRole.WindowText)
    assert _contrast(color, QColor(tokens.BG_1)) >= 4.5


def test_class_panel_prefers_all_rows_up_to_60_percent_of_the_host(app):
    """``preferred_height`` is the height the standalone placement uses."""
    host = _host(400, 600)
    panel = ClassPanel(host)
    panel.set_registry(default_registry())
    short = panel.preferred_height(600)
    panel.reposition()
    assert panel.height() == short < 360

    panel.set_registry(_registry_with_many_classes(25))
    assert panel.preferred_height(600) == 360
    assert panel.preferred_height(400) == 240


def test_class_panel_fits_a_smaller_height_and_scrolls(app):
    """An open Enhance popover leaves the panel less room; its rows scroll (UXE-09)."""
    host = _host(400, 600)
    panel = ClassPanel(host)
    panel.set_registry(_registry_with_many_classes(25))
    panel.fit_height(200)
    assert panel.height() == 200
    assert panel._rows_container.sizeHint().height() > panel._scroll.height()


def test_a_long_class_name_is_elided_at_its_end(app):
    """The start of the name stays (it is what tells the classes apart); the full name is
    the tooltip."""
    host = _host(900, 600)
    name = "Mycorrhizal hyphae and fine roots"
    panel = ClassPanel(host)
    panel.set_registry(default_registry().add(key="long", name=name, color="#123456"))
    host.show()
    panel.reposition()
    QApplication.processEvents()
    label = panel._rows["long"]._name_label
    shown = label.elided_text()
    assert shown.endswith("…") and name.startswith(shown[:-1])
    assert label.is_elided() and label.toolTip() == name and label.text() == name


def test_gear_button_emits_edit_requested(app):
    host = _host()
    panel = ClassPanel(host)
    panel.set_registry(default_registry())

    got = []
    panel.edit_requested.connect(lambda: got.append(1))
    panel._gear.click()
    assert got == [1]


# --------------------------------------------------------------------------- #
#  ClassEditorDialog
# --------------------------------------------------------------------------- #
def test_add_class_then_accept(app):
    dialog = _dialog(default_registry())
    dialog._add_name_edit.setText("Dead root")
    dialog._set_add_color("#111111")
    assert dialog._add_key_edit.text() == "dead_root"

    dialog._on_add_clicked()
    assert dialog._error is None

    dialog._on_accept()
    reg = dialog.result_registry()
    assert reg is not None
    assert reg.by_key("dead_root").id == 4
    assert reg.by_key("dead_root").name == "Dead root"
    assert reg.by_key("dead_root").color == "#111111"


def test_rename_nodule_reflected(app):
    dialog = _dialog(default_registry())
    row = dialog._rows["nodule"]
    row.name_edit.setText("Nodule X")
    row.name_edit.editingFinished.emit()
    assert dialog._error is None

    dialog._on_accept()
    reg = dialog.result_registry()
    assert reg is not None
    assert reg.by_key("nodule").name == "Nodule X"


def test_retire_nodule_bypassing_confirm(app, monkeypatch):
    import app.gui.class_editor_dialog as mod

    monkeypatch.setattr(
        mod.QMessageBox, "question", lambda *a, **k: mod.QMessageBox.StandardButton.Yes
    )

    dialog = _dialog(default_registry())
    dialog._on_retire_clicked("nodule")
    assert "nodule" not in dialog._rows
    assert dialog._candidate.by_key("nodule").retired is True

    dialog._on_accept()
    reg = dialog.result_registry()
    assert reg is not None
    assert reg.by_key("nodule").retired is True
    assert "nodule" not in [c.key for c in reg.active()]


def test_ignore_row_has_no_retire_button(app):
    dialog = _dialog(default_registry())
    assert dialog._rows["ignore"].retire_button is None
    for key in ("root", "nodule", "root_tip"):
        assert dialog._rows[key].retire_button is not None


def test_duplicate_colour_shows_inline_error_and_disables_ok(app):
    dialog = _dialog(default_registry())
    ok_button = dialog._button_box.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_button.isEnabled()

    dialog._add_name_edit.setText("Copy Root")
    dialog._set_add_color(default_registry().by_key("root").color)
    dialog._on_add_clicked()

    assert dialog._error is not None
    assert dialog._error_label.text() != ""
    # The dialog is never shown in this test, so use the explicit-visibility
    # flag (isHidden) rather than isVisible (which requires a shown ancestor).
    assert not dialog._error_label.isHidden()
    assert not ok_button.isEnabled()

    # Fixing the colour clears the error and re-enables OK.
    dialog._set_add_color("#222222")
    dialog._on_add_clicked()
    assert dialog._error is None
    assert ok_button.isEnabled()


def test_cancel_returns_none(app):
    dialog = _dialog(default_registry())
    dialog._add_name_edit.setText("Dead root")
    dialog._set_add_color("#333333")
    dialog._on_add_clicked()

    dialog.reject()
    assert dialog.result_registry() is None


def test_unchanged_registry_returns_none(app):
    dialog = _dialog(default_registry())
    dialog._on_accept()
    assert dialog.result_registry() is None


# --------------------------------------------------------------------------- #
#  Fix wave: stale rows, sticky errors, default colour, scrolling
# --------------------------------------------------------------------------- #
def test_set_registry_drops_old_rows_at_once(app):
    """Rebuilt rows replace the old ones immediately (no stale highlighted row)."""
    host = _host()
    panel = ClassPanel(host)
    host.show()
    try:
        panel.set_registry(default_registry())
        panel.set_active("root")
        QApplication.processEvents()
        old_rows = list(panel._rows.values())

        panel.set_registry(default_registry())
        rows = panel.findChildren(QFrame, "classRow")
        assert len(rows) == 4, len(rows)
        assert all(r.isHidden() and r.parent() is None for r in old_rows)

        QApplication.processEvents()
        visible = [r for r in panel.findChildren(QFrame, "classRow") if r.isVisible()]
        assert len(visible) == 4
        assert [r for r in visible if r._active] == [panel._rows["root"]]
    finally:
        host.close()


def _ok(dialog):
    return dialog._button_box.button(QDialogButtonBox.StandardButton.Ok)


def test_restoring_a_blanked_name_clears_the_error(app):
    dialog = _dialog(default_registry())
    row = dialog._rows["nodule"]
    row.name_edit.setText("")
    row.name_edit.editingFinished.emit()
    assert dialog._error is not None and not _ok(dialog).isEnabled()

    row.name_edit.setText("Nodule")  # back to the saved name
    row.name_edit.editingFinished.emit()
    assert dialog._error is None and _ok(dialog).isEnabled()


def test_failed_add_error_clears_when_the_form_is_edited(app):
    root_color = default_registry().by_key("root").color
    for edit in ("name", "key", "colour"):
        dialog = _dialog(default_registry())
        dialog._add_name_edit.setText("Copy Root")
        dialog._set_add_color(root_color)
        dialog._on_add_clicked()
        assert dialog._error is not None and not _ok(dialog).isEnabled(), edit

        if edit == "name":
            dialog._add_name_edit.setText("")  # the user abandons the form
        elif edit == "key":
            QTest.keyClicks(dialog._add_key_edit, "x")
        else:
            dialog._set_add_color("#0a0b0c")
        assert dialog._error is None and _ok(dialog).isEnabled(), edit
        dialog._on_accept()
        # OK adds a class still typed in the form (it used to drop it, G2 2026-09-30).
        added = dialog.result_registry()
        if edit == "name":
            assert added is None, edit  # the form was abandoned: nothing to add
        elif edit == "key":
            assert added is None and dialog._error is not None, edit  # still the root's colour
        else:
            assert added is not None and "copy_root" in {c.key for c in added.classes}, edit


def test_row_error_survives_add_form_edits(app):
    """Editing the add form must not hide a still-invalid (blank) class name."""
    dialog = _dialog(default_registry())
    row = dialog._rows["nodule"]
    row.name_edit.setText("")
    row.name_edit.editingFinished.emit()
    dialog._add_name_edit.setText("Dead root")
    assert dialog._error is not None and not _ok(dialog).isEnabled()


def test_default_add_colour_is_never_taken(app):
    retired = default_registry().add(key="old", name="Old", color="#76ff03").retire("old")
    for registry in (default_registry(), retired):
        dialog = _dialog(registry)
        for name in ("First", "Second", "Third"):
            used = {c.color.lower() for c in dialog._candidate.classes}
            assert dialog._add_color.lower() not in used
            dialog._add_name_edit.setText(name)
            dialog._on_add_clicked()  # no colour chosen: the default must be free
            assert dialog._error is None, dialog._error
        colours = [c.color.lower() for c in dialog._candidate.classes]
        assert len(set(colours)) == len(colours)


def test_default_add_colour_moves_off_a_colour_just_given_to_a_class(app):
    dialog = _dialog(default_registry())
    dialog._apply_color("nodule", dialog._add_color)
    assert dialog._error is None
    dialog._add_name_edit.setText("New")
    dialog._on_add_clicked()
    assert dialog._error is None


def test_editor_rows_scroll_when_there_are_many_classes(app):
    dialog = _dialog(_registry_with_many_classes(40))
    scroll = dialog._rows_scroll
    assert isinstance(scroll, QScrollArea)
    assert scroll.widget() is dialog._rows_container
    assert len(dialog._rows) == 41
    assert dialog.sizeHint().height() <= 640, dialog.sizeHint()
    assert scroll.maximumHeight() < dialog._rows_container.sizeHint().height()


def test_retire_by_clicking_its_own_button(app, monkeypatch):
    """The row holding the clicked Retire button is rebuilt inside that click."""
    import app.gui.class_editor_dialog as mod

    monkeypatch.setattr(
        mod.QMessageBox, "question", lambda *a, **k: mod.QMessageBox.StandardButton.Yes
    )
    dialog = _dialog(default_registry())
    dialog._rows["nodule"].retire_button.click()
    QApplication.processEvents()
    assert "nodule" not in dialog._rows
    assert dialog._candidate.by_key("nodule").retired is True


def test_restore_a_retired_class_in_the_editor(app):
    dialog = _dialog(default_registry().retire("nodule"))
    assert "nodule" not in dialog._rows and list(dialog.restore_buttons) == ["nodule"]

    dialog.restore_buttons["nodule"].click()
    assert "nodule" in dialog._rows and dialog.restore_buttons == {}

    dialog._on_accept()
    assert dialog.result_registry() == default_registry()
