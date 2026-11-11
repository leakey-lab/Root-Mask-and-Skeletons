"""``ClassEditorDialog`` (``app/gui/class_editor_dialog.py``): the project's classes.

Add (key slugged from the name, a free default colour), rename, recolour, retire (confirmed;
Ignore has none) and restore; every edit is checked by the registry and an error shows inline;
OK is off while a field holds invalid input, and adds a class typed but not added yet; Enter in
the form adds the class and keeps the dialog open; Cancel or no change gives no registry; the
rows scroll once there are many classes; the Tab order follows the screen.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QWidget,
)

from app.gt.registry import default_registry
from app.gui.class_editor_dialog import ClassEditorDialog
from tests.qt_teardown import delete_widgets

Focus = Qt.FocusPolicy

_MADE: list[QWidget] = []  # the hosts and dialogs a test made: deleted after it


# ClassEditorDialog has no public accessor for its form fields, rows and error line, and the
# tests drive some slots directly (the colour picker and the Retire question are modal). These
# helpers are the one place that reads its private parts (accessors recorded for the owner).
def add_name(dialog) -> QLineEdit:
    return dialog._add_name_edit


def add_key(dialog) -> QLineEdit:
    return dialog._add_key_edit


def add_swatch(dialog) -> QPushButton:
    return dialog._add_swatch


def add_button(dialog) -> QPushButton:
    return dialog._add_button


def add_colour(dialog) -> str:
    return dialog._add_color


def rows_of(dialog) -> dict:
    """The active classes' rows, by key."""
    return dialog._rows


def row_of(dialog, key):
    return dialog._rows[key]


def ok_of(dialog) -> QPushButton:
    return dialog._button_box.button(QDialogButtonBox.StandardButton.Ok)


def cancel_of(dialog) -> QPushButton:
    return dialog._button_box.button(QDialogButtonBox.StandardButton.Cancel)


def error_label(dialog) -> QLabel:
    return dialog._error_label


def blocking_error(dialog) -> str | None:
    """The error that keeps OK off (None: OK is on)."""
    return dialog._error


def candidate(dialog):
    """The registry as edited so far (before OK)."""
    return dialog._candidate


def rows_scroll(dialog) -> QScrollArea:
    return dialog._rows_scroll


def rows_container(dialog) -> QWidget:
    return dialog._rows_container


def accept(dialog) -> None:
    """What OK runs (``accept()`` alone would not record the result)."""
    dialog._on_accept()


def click_add(dialog) -> None:
    dialog._on_add_clicked()


def pick_add_colour(dialog, colour: str) -> None:
    """The add form's colour, as if picked in the colour dialog."""
    dialog._set_add_color(colour)


def recolour(dialog, key: str, colour: str) -> None:
    """Class ``key``'s colour, as if picked in the colour dialog."""
    dialog._apply_color(key, colour)


def commit_name(dialog, key: str) -> None:
    dialog._on_name_committed(key)


def retire(dialog, key: str) -> None:
    """Retire ``key`` as its button does (the question answered by the test)."""
    dialog._on_retire_clicked(key)


@pytest.fixture(autouse=True)
def _delete_test_widgets(app):
    """Close and delete the test's hosts and dialogs, and drop every dead wrapper, before
    the next test runs (see tests/qt_teardown.py)."""
    yield
    delete_widgets(_MADE)


def _dialog(registry):
    dialog = ClassEditorDialog(registry)
    _MADE.append(dialog)
    return dialog


def _registry_with_many_classes(n_labelled: int = 12):
    reg = default_registry()
    for i in range(n_labelled - len(reg.labelled())):
        reg = reg.add(key=f"extra_{i}", name=f"Extra {i}", color=f"#{100 + i:06x}")
    return reg


def _ok(dialog):
    return ok_of(dialog)


@pytest.fixture
def made(app):
    """Widgets closed and deleted deterministically after the test."""
    widgets: list = []
    yield widgets
    delete_widgets(widgets)


def test_add_class_then_accept(app):
    dialog = _dialog(default_registry())
    add_name(dialog).setText("Dead root")
    pick_add_colour(dialog, "#111111")
    assert add_key(dialog).text() == "dead_root"

    click_add(dialog)
    assert blocking_error(dialog) is None

    accept(dialog)
    reg = dialog.result_registry()
    assert reg is not None
    assert reg.by_key("dead_root").id == 4
    assert reg.by_key("dead_root").name == "Dead root"
    assert reg.by_key("dead_root").color == "#111111"


def test_rename_nodule_reflected(app):
    dialog = _dialog(default_registry())
    row = row_of(dialog, "nodule")
    row.name_edit.setText("Nodule X")
    row.name_edit.editingFinished.emit()
    assert blocking_error(dialog) is None

    accept(dialog)
    reg = dialog.result_registry()
    assert reg is not None
    assert reg.by_key("nodule").name == "Nodule X"


def test_retire_nodule_bypassing_confirm(app, monkeypatch):
    import app.gui.class_editor_dialog as mod

    monkeypatch.setattr(
        mod.QMessageBox, "question", lambda *a, **k: mod.QMessageBox.StandardButton.Yes
    )

    dialog = _dialog(default_registry())
    retire(dialog, "nodule")
    assert "nodule" not in rows_of(dialog)
    assert candidate(dialog).by_key("nodule").retired is True

    accept(dialog)
    reg = dialog.result_registry()
    assert reg is not None
    assert reg.by_key("nodule").retired is True
    assert "nodule" not in [c.key for c in reg.active()]


def test_retire_by_clicking_its_own_button(app, monkeypatch):
    """The row holding the clicked Retire button is rebuilt inside that click."""
    import app.gui.class_editor_dialog as mod

    monkeypatch.setattr(
        mod.QMessageBox, "question", lambda *a, **k: mod.QMessageBox.StandardButton.Yes
    )
    dialog = _dialog(default_registry())
    row_of(dialog, "nodule").retire_button.click()
    QApplication.processEvents()
    assert "nodule" not in rows_of(dialog)
    assert candidate(dialog).by_key("nodule").retired is True


def test_ignore_row_has_no_retire_button(app):
    dialog = _dialog(default_registry())
    assert row_of(dialog, "ignore").retire_button is None
    for key in ("root", "nodule", "root_tip"):
        assert row_of(dialog, key).retire_button is not None


def test_duplicate_colour_shows_inline_error_and_disables_ok(app):
    dialog = _dialog(default_registry())
    ok_button = ok_of(dialog)
    assert ok_button.isEnabled()

    add_name(dialog).setText("Copy Root")
    pick_add_colour(dialog, default_registry().by_key("root").color)
    click_add(dialog)

    assert blocking_error(dialog) is not None
    assert error_label(dialog).text() != ""
    # The dialog is never shown in this test, so use the explicit-visibility
    # flag (isHidden) rather than isVisible (which requires a shown ancestor).
    assert not error_label(dialog).isHidden()
    assert not ok_button.isEnabled()

    # Fixing the colour clears the error and re-enables OK.
    pick_add_colour(dialog, "#222222")
    click_add(dialog)
    assert blocking_error(dialog) is None
    assert ok_button.isEnabled()


def test_cancel_returns_none(app):
    dialog = _dialog(default_registry())
    add_name(dialog).setText("Dead root")
    pick_add_colour(dialog, "#333333")
    click_add(dialog)

    dialog.reject()
    assert dialog.result_registry() is None


def test_unchanged_registry_returns_none(app):
    dialog = _dialog(default_registry())
    accept(dialog)
    assert dialog.result_registry() is None


def test_restoring_a_blanked_name_clears_the_error(app):
    dialog = _dialog(default_registry())
    row = row_of(dialog, "nodule")
    row.name_edit.setText("")
    row.name_edit.editingFinished.emit()
    assert blocking_error(dialog) is not None and not _ok(dialog).isEnabled()

    row.name_edit.setText("Nodule")  # back to the saved name
    row.name_edit.editingFinished.emit()
    assert blocking_error(dialog) is None and _ok(dialog).isEnabled()


def test_failed_add_error_clears_when_the_form_is_edited(app):
    root_color = default_registry().by_key("root").color
    for edit in ("name", "key", "colour"):
        dialog = _dialog(default_registry())
        add_name(dialog).setText("Copy Root")
        pick_add_colour(dialog, root_color)
        click_add(dialog)
        assert blocking_error(dialog) is not None and not _ok(dialog).isEnabled(), edit

        if edit == "name":
            add_name(dialog).setText("")  # the user abandons the form
        elif edit == "key":
            QTest.keyClicks(add_key(dialog), "x")
        else:
            pick_add_colour(dialog, "#0a0b0c")
        assert blocking_error(dialog) is None and _ok(dialog).isEnabled(), edit
        accept(dialog)
        # OK adds a class still typed in the form (it used to drop it, G2 2026-09-30).
        added = dialog.result_registry()
        if edit == "name":
            assert added is None, edit  # the form was abandoned: nothing to add
        elif edit == "key":
            assert added is None and blocking_error(dialog) is not None, (
                edit
            )  # still the root's colour
        else:
            assert added is not None and "copy_root" in {c.key for c in added.classes}, edit


def test_row_error_survives_add_form_edits(app):
    """Editing the add form must not hide a still-invalid (blank) class name."""
    dialog = _dialog(default_registry())
    row = row_of(dialog, "nodule")
    row.name_edit.setText("")
    row.name_edit.editingFinished.emit()
    add_name(dialog).setText("Dead root")
    assert blocking_error(dialog) is not None and not _ok(dialog).isEnabled()


def test_default_add_colour_is_never_taken(app):
    retired = default_registry().add(key="old", name="Old", color="#76ff03").retire("old")
    for registry in (default_registry(), retired):
        dialog = _dialog(registry)
        for name in ("First", "Second", "Third"):
            used = {c.color.lower() for c in candidate(dialog).classes}
            assert add_colour(dialog).lower() not in used
            add_name(dialog).setText(name)
            click_add(dialog)  # no colour chosen: the default must be free
            assert blocking_error(dialog) is None, blocking_error(dialog)
        colours = [c.color.lower() for c in candidate(dialog).classes]
        assert len(set(colours)) == len(colours)


def test_default_add_colour_moves_off_a_colour_just_given_to_a_class(app):
    dialog = _dialog(default_registry())
    recolour(dialog, "nodule", add_colour(dialog))
    assert blocking_error(dialog) is None
    add_name(dialog).setText("New")
    click_add(dialog)
    assert blocking_error(dialog) is None


def test_editor_rows_scroll_when_there_are_many_classes(app):
    dialog = _dialog(_registry_with_many_classes(40))
    scroll = rows_scroll(dialog)
    assert isinstance(scroll, QScrollArea)
    assert scroll.widget() is rows_container(dialog)
    assert len(rows_of(dialog)) == 41
    assert dialog.sizeHint().height() <= 640, dialog.sizeHint()
    assert scroll.maximumHeight() < rows_container(dialog).sizeHint().height()


def test_restore_a_retired_class_in_the_editor(app):
    dialog = _dialog(default_registry().retire("nodule"))
    assert "nodule" not in rows_of(dialog) and list(dialog.restore_buttons) == ["nodule"]

    dialog.restore_buttons["nodule"].click()
    assert "nodule" in rows_of(dialog) and dialog.restore_buttons == {}

    accept(dialog)
    assert dialog.result_registry() == default_registry()


def test_the_class_editor_has_no_restore_button_without_retired_classes(app):
    dialog = ClassEditorDialog(default_registry())
    _MADE.append(dialog)
    assert dialog.restore_buttons == {}


@pytest.mark.parametrize("taken", ["nodule", "retired"])
def test_a_colour_another_class_uses_is_refused_and_never_blocks_ok(app, monkeypatch, taken):
    """Recolouring Root, through its swatch, to Nodule's colour or a retired class's: the
    registry refuses (colours are unique, retired included), the error shows, OK stays on, and
    OK gives the registry without that recolour (one-off errors never block OK)."""
    import app.gui.class_editor_dialog as mod

    registry = default_registry().add(key="old", name="Old", color="#123456").retire("old")
    colour = registry.by_key("nodule").color if taken == "nodule" else "#123456"
    monkeypatch.setattr(mod.QColorDialog, "getColor", lambda *a, **k: QColor(colour))
    dialog = ClassEditorDialog(registry)
    _MADE.append(dialog)

    row_of(dialog, "root").swatch_button.click()

    assert "Duplicate class colour" in error_label(dialog).text()
    assert blocking_error(dialog) is None and ok_of(dialog).isEnabled()
    add_name(dialog).setText("Hypha")  # a later edit that succeeds clears the message
    click_add(dialog)
    assert error_label(dialog).text() == ""
    accept(dialog)
    result = dialog.result_registry()
    assert result.by_key("root").color == registry.by_key("root").color
    assert "hypha" in {c.key for c in result.classes}


def test_enter_in_the_add_form_adds_the_class_and_keeps_the_dialog_open(made):
    dlg = ClassEditorDialog(default_registry())
    made.append(dlg)
    dlg.show()
    QTest.qWaitForWindowExposed(dlg)
    add_name(dlg).setText("Dead root")
    add_name(dlg).setFocus()
    QTest.keyClick(add_name(dlg), Qt.Key.Key_Return)
    assert dlg.isVisible() and dlg.result() == 0  # not accepted
    assert "dead_root" in rows_of(dlg) and add_name(dlg).text() == ""

    add_name(dlg).setText("Hypha")
    add_key(dlg).setFocus()
    QTest.keyClick(add_key(dlg), Qt.Key.Key_Enter)
    assert "hypha" in rows_of(dlg) and dlg.isVisible()

    ok_of(dlg).click()
    assert not dlg.isVisible()
    assert {"dead_root", "hypha"} <= {c.key for c in dlg.result_registry().classes}


def test_class_editor_keyboard_flow(made):
    """It opens in the new-class field; Tab cycles through the rows, the form, OK and Cancel
    in drawing order; Enter with nothing typed is OK."""
    dlg = ClassEditorDialog(default_registry())
    made.append(dlg)
    dlg.show()
    dlg.activateWindow()
    QTest.qWaitForWindowActive(dlg, 2000)
    assert dlg.focusWidget() is add_name(dlg)

    expected = []
    for class_row in (row_of(dlg, k) for k in ("root", "nodule", "root_tip", "ignore")):
        expected += [class_row.swatch_button, class_row.name_edit]
        if class_row.retire_button is not None:
            expected.append(class_row.retire_button)
    ok = ok_of(dlg)
    cancel = cancel_of(dlg)
    expected += [
        add_name(dlg),
        add_key(dlg),
        add_swatch(dlg),
        add_button(dlg),
        ok,
        cancel,
    ]
    chain, widget = [], expected[0]
    while True:
        if widget.focusPolicy() & Focus.TabFocus and widget.isVisible() and widget.isEnabled():
            chain.append(widget)
        widget = widget.nextInFocusChain()
        if widget is expected[0] or len(chain) > 40:
            break
    assert chain == expected

    QTest.keyClick(add_name(dlg), Qt.Key.Key_Return)  # nothing typed: Enter = OK
    assert not dlg.isVisible() and dlg.result() == 1


def test_ok_adds_a_typed_class_that_was_not_added_yet(made):
    dlg = ClassEditorDialog(default_registry())
    made.append(dlg)
    add_name(dlg).setText("Dead root")
    accept(dlg)
    assert dlg.result_registry() is not None
    assert "dead_root" in {c.key for c in dlg.result_registry().classes}


def test_ok_with_an_invalid_typed_class_stays_open_with_its_error(made):
    dlg = ClassEditorDialog(default_registry())
    made.append(dlg)
    dlg.show()
    add_name(dlg).setText("Second root")
    add_key(dlg).setText("root")  # key taken
    accept(dlg)
    assert dlg.isVisible() and dlg.result_registry() is None
    assert error_label(dlg).isVisible() and "Duplicate" in error_label(dlg).text()


def test_a_blank_name_says_a_class_needs_a_name(made):
    dlg = ClassEditorDialog(default_registry())
    made.append(dlg)
    row_of(dlg, "root").name_edit.setText("   ")
    commit_name(dlg, "root")
    assert error_label(dlg).text() == "A class needs a name."
    row_of(dlg, "root").name_edit.setText("Root")
    commit_name(dlg, "root")
    add_name(dlg).setText("  ")
    click_add(dlg)
    assert error_label(dlg).text() == "A class needs a name."


def test_a_retire_answered_no_changes_nothing(app, monkeypatch):
    import app.gui.class_editor_dialog as mod

    monkeypatch.setattr(
        mod.QMessageBox, "question", lambda *a, **k: mod.QMessageBox.StandardButton.No
    )
    dialog = _dialog(default_registry())

    row_of(dialog, "nodule").retire_button.click()

    assert "nodule" in rows_of(dialog) and not candidate(dialog).by_key("nodule").retired
    accept(dialog)
    assert dialog.result_registry() is None  # unchanged
