"""Declarative shortcut registry (UXS-10 / UXE-07 / UI-21) and the F1 sheet built from it."""

from __future__ import annotations

import sys

import pytest
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from app.gui import shortcuts as sc
from app.gui.key_text import platform_keys
from tests.qt_teardown import delete_widgets


def _portable(key: str) -> str:
    return QKeySequence(key).toString(QKeySequence.SequenceFormat.PortableText)


def _live_keys(widget: QWidget) -> set[str]:
    return {
        s.key().toString(QKeySequence.SequenceFormat.PortableText)
        for s in widget.findChildren(QShortcut)
        if s.parent() is widget
    }


def _top_level_labels() -> int:
    """How many labels are windows of their own (a label made without a parent and never
    put into a layout is one)."""
    return sum(isinstance(widget, QLabel) for widget in QApplication.topLevelWidgets())


# ------------------------------------------------------------------ Qt-free core
def test_registry_module_is_qt_free():
    import subprocess
    import sys

    code = "import sys, app.gui.shortcuts; print('PyQt6' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_registry_has_no_conflicting_keys():
    assert sc.find_conflicts() == []


def test_duplicate_key_in_one_scope_is_reported():
    """The Ctrl+Y double binding (UXE-07) is exactly what this catches."""
    dup = sc.REGISTRY + (sc.ShortcutSpec("redo_again", ("Ctrl+Y",), "Redo", sc.CORRECT),)
    conflicts = sc.find_conflicts(dup)
    assert [c[0] for c in conflicts] == ["CTRL+Y"]


def test_global_keys_conflict_with_every_editor_scope():
    dup = sc.REGISTRY + (sc.ShortcutSpec("oops", ("F1",), "x", sc.CORRECT),)
    assert [c[0] for c in sc.find_conflicts(dup)] == ["F1"]


def test_redo_is_ctrl_y_and_ctrl_shift_z_in_the_editor():
    assert sc.keys_for("redo", sc.CORRECT) == ("Ctrl+Y", "Ctrl+Shift+Z")
    assert sc.keys_for("undo", sc.CORRECT) == ("Ctrl+Z",)


def test_global_keys_the_integrator_wires():
    assert sc.keys_for("help", sc.GLOBAL) == ("F1", "?")
    assert sc.keys_for("open_folder", sc.GLOBAL) == ("Ctrl+O",)
    assert sc.keys_for("next_image", sc.GLOBAL) == ("PgDown",)
    assert sc.keys_for("previous_image", sc.GLOBAL) == ("PgUp",)
    assert sc.STAGE_KEYS == tuple(f"Alt+{i}" for i in range(1, 6))
    assert sc.keys_for("go_to_stage", sc.GLOBAL) == tuple(
        key for column in zip(*sc.stage_keys(), strict=True) for key in column
    )


def test_correct_sheet_lists_both_layers_keys_and_gestures():
    specs = sc.specs_for(sc.CORRECT)
    labels = " ".join(s.label for s in specs)
    for key in ("Enter", "Esc", "Delete", "Ctrl+click", "Shift+drag", "Double-click"):
        assert platform_keys(key) in labels
    keys = {key for spec in specs for key in spec.keys}
    mask_keys, skeleton_keys = {"B", "F"}, {"V", "P", "C", "S"}
    layer_keys = {"1", "2", "X", "H", "Shift+H", "E", "[", "]", "Ctrl+S"}
    assert mask_keys | skeleton_keys | layer_keys <= keys
    # A key is either a page QShortcut or a canvas key, never both (both would fight).
    shortcut_keys = {k.upper() for s in specs if s.kind == sc.SHORTCUT for k in s.keys}
    canvas_keys = {k.upper() for s in specs if s.kind == sc.KEY for k in s.keys}
    assert not shortcut_keys & canvas_keys
    assert {"1", "2", "X", "E", "F", "V", "P", "C", "CTRL+S"} <= shortcut_keys
    assert {"B", "S", "ENTER", "ESC", "DELETE", "BACKSPACE"} <= canvas_keys


def test_correct_sheet_lists_class_digits():
    spec = sc.find_spec("select_class", sc.CORRECT)
    assert spec is not None
    assert set(spec.keys) == {str(n) for n in range(10)}
    assert "Ignore" in spec.description


def test_tooltip_appends_the_first_key():
    assert sc.tooltip("Eraser", "eraser", sc.CORRECT) == "Eraser (E)"
    assert sc.tooltip("Redo", "redo", sc.CORRECT) == platform_keys("Redo (Ctrl+Y)")
    assert sc.tooltip("Nothing", "no_such_action", sc.CORRECT) == "Nothing"


def test_scope_for_stage():
    assert sc.scope_for_stage("Trace GT") is None  # the stage is gone (WG-T11)
    assert sc.scope_for_stage("Library") == sc.LIBRARY
    assert sc.scope_for_stage("Correct") == sc.CORRECT
    assert sc.scope_for_stage("Generate") is None


# ------------------------------------------------------------------ Qt adapter
def test_install_shortcuts_binds_every_key_once(app):
    w = QWidget()
    hits = []
    installed = sc.install_shortcuts(
        w, sc.CORRECT, {"redo": lambda: hits.append("redo"), "undo": lambda: None}
    )
    assert sorted(installed) == ["redo", "undo"]
    assert [s.key().toString() for s in installed["redo"]] == ["Ctrl+Y", "Ctrl+Shift+Z"]
    assert _live_keys(w) == {"Ctrl+Y", "Ctrl+Shift+Z", "Ctrl+Z"}
    installed["redo"][1].activated.emit()
    assert hits == ["redo"]


def test_install_shortcuts_rejects_unknown_or_non_shortcut_actions(app):
    w = QWidget()
    with pytest.raises(KeyError):
        sc.install_shortcuts(w, sc.CORRECT, {"no_such_action": lambda: None})
    with pytest.raises(ValueError):
        sc.install_shortcuts(w, sc.CORRECT, {"finish_line": lambda: None})


def test_every_shortcut_on_the_sheet_is_live_and_every_live_one_is_on_it(app, tmp_path):
    """The F1 sheet and the window's QShortcuts are one set, per scope: a key on the sheet that
    does nothing, or a working key the sheet does not list, fails. The owners: the window and
    the PgDn / PgUp owners (Global), the display canvas (Library), the Correct page (Correct,
    which binds PgDn / PgUp itself)."""
    from tests.window_support import build_window, quiet_window

    window = build_window(tmp_path / "s.ini", show=False)
    try:
        correct = window.correct_editor
        paging = {s.parent() for s in window._paging_shortcuts}
        live = {sc.GLOBAL: set(), sc.LIBRARY: set(), sc.CORRECT: set()}
        for shortcut in window.findChildren(QShortcut):
            owner = shortcut.parent()
            key = shortcut.key().toString(QKeySequence.SequenceFormat.PortableText)
            if owner is window or owner in paging:
                live[sc.GLOBAL].add(key)
            elif owner is window.display_area:
                live[sc.LIBRARY].add(key)
            elif owner is correct or correct.isAncestorOf(owner):
                live[sc.CORRECT].add(key)
            else:
                pytest.fail(f"{key} is bound on {type(owner).__name__}, which no scope names")

        def documented(scope):
            return {
                _portable(k) for s in sc.specs_for(scope) if s.kind == sc.SHORTCUT for k in s.keys
            }

        paging_keys = {
            _portable(k)
            for a in ("next_image", "previous_image")
            for k in sc.keys_for(a, sc.GLOBAL)
        }
        assert live[sc.GLOBAL] == documented(sc.GLOBAL)
        assert live[sc.LIBRARY] == documented(sc.LIBRARY)
        assert live[sc.CORRECT] == documented(sc.CORRECT) | paging_keys
    finally:
        delete_widgets([window], prepare=quiet_window)


# ------------------------------------------------------------------ F1 sheet
def test_shortcuts_dialog_has_one_row_per_spec(app):
    from app.gui.empty_state import ShortcutsDialog

    dlg = ShortcutsDialog()
    rows = dlg.findChildren(QLabel, "shortcutKeys")
    assert len(rows) == len(sc.REGISTRY)
    text = " ".join(r.text() for r in rows)
    assert "PgDn" in text or "PgDown" in text
    assert "Enter" in text


def test_shortcuts_dialog_puts_current_scope_first(app):
    from app.gui.empty_state import ShortcutsDialog

    dlg = ShortcutsDialog(current_scope=sc.CORRECT)
    assert dlg.section_titles()[0] == sc.CORRECT
    assert sorted(dlg.section_titles()) == sorted(sc.SCOPES)


def test_shortcuts_dialog_offers_about_when_wired(app):
    from app.gui.empty_state import ShortcutsDialog

    opened = []
    dlg = ShortcutsDialog(on_about=lambda: opened.append(True))
    buttons = [b for b in dlg.findChildren(QPushButton) if "About" in b.text()]
    assert len(buttons) == 1
    buttons[0].click()
    assert opened == [True]
    plain = ShortcutsDialog()
    assert not [b for b in plain.findChildren(QPushButton) if "About" in b.text()]


def test_shortcuts_dialog_keeps_its_focus_chain_and_row_order(app):
    """The sheet is built parent-first (S09-02): Tab order and rows stay the registry's."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialogButtonBox

    from app.gui.empty_state import ShortcutsDialog
    from tests.qt_teardown import delete_widgets

    strays = _top_level_labels()
    dlg = ShortcutsDialog(current_scope=sc.LIBRARY, on_about=lambda: None)
    try:
        close = dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Close)
        focusable, widget = [], dlg.nextInFocusChain()
        while widget is not dlg:
            if widget.focusPolicy() != Qt.FocusPolicy.NoFocus:
                focusable.append(widget.text() if isinstance(widget, QPushButton) else widget)
            widget = widget.nextInFocusChain()
        scroll = dlg.findChild(QWidget, "shortcutScroll")
        assert focusable == [scroll, "About SPROUTS…", close.text()]
        order = [sc.LIBRARY, sc.GLOBAL, sc.CORRECT]
        assert dlg.section_titles() == order
        rows = [label.text() for label in dlg.findChildren(QLabel, "shortcutKeys")]
        assert rows == [spec.label for scope in order for spec in sc.specs_for(scope)]
        assert _top_level_labels() == strays  # no label was left out of the sheet
    finally:
        delete_widgets([dlg])


def test_install_shortcuts_accepts_one_slot_per_key(app):
    """Alt+1 … Alt+5 each go to their own stage."""
    w = QWidget()
    hits = []
    slots = [lambda i=i: hits.append(i) for i in range(len(sc.keys_for("go_to_stage", sc.GLOBAL)))]
    installed = sc.install_shortcuts(w, sc.GLOBAL, {"go_to_stage": slots})
    installed["go_to_stage"][2].activated.emit()
    assert hits == [2]
    with pytest.raises(ValueError):
        sc.install_shortcuts(w, sc.GLOBAL, {"go_to_stage": slots[:3]})


def test_correct_sheet_documents_every_skeleton_tool_key():
    from app.gui import shortcuts, skeleton_tools

    scope = shortcuts.CORRECT
    documented = {k for spec in shortcuts.specs_for(scope) for k in spec.keys}
    assert set(skeleton_tools.TOOL_KEYS.values()) <= documented
    assert {skeleton_tools.SMOOTH_KEY, "Ctrl+S", "[", "]", "Delete", "Backspace"} <= documented
    assert shortcuts.find_spec("delete_point", scope) is None
    assert shortcuts.keys_for("delete_branch", scope) == ("Delete", "Backspace")
    actions = [spec.action for spec in shortcuts.specs_for(scope)]
    assert len(actions) == len(set(actions))


# ------------------------------------------------------------------ macOS key text


def test_the_undo_row_of_the_sheet_shows_cmd_z_on_macos(monkeypatch):
    undo = sc.find_spec("undo", sc.CORRECT)
    redo = sc.find_spec("redo", sc.CORRECT)
    assert undo is not None and redo is not None
    monkeypatch.setattr(sys, "platform", "win32")
    assert (undo.label, redo.label) == ("Ctrl+Z", "Ctrl+Y / Ctrl+Shift+Z")

    monkeypatch.setattr(sys, "platform", "darwin")

    assert (undo.label, redo.label) == ("Cmd+Z", "Cmd+Y / Cmd+Shift+Z")
    assert sc.find_spec("go_to_stage", sc.GLOBAL).label == "Option+1 … Option+5"


def test_tooltips_show_the_platform_key_text(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    assert sc.tooltip("Redo", "redo", sc.CORRECT) == "Redo (Cmd+Y)"
    assert sc.tooltip("Eraser", "eraser", sc.CORRECT) == "Eraser (E)"


def test_the_sheet_dialog_shows_cmd_on_macos(app, monkeypatch):
    from PyQt6.QtWidgets import QLabel

    from app.gui.empty_state import ShortcutsDialog

    monkeypatch.setattr(sys, "platform", "darwin")

    dialog = ShortcutsDialog(current_scope=sc.CORRECT)
    try:
        text = " ".join(label.text() for label in dialog.findChildren(QLabel, "shortcutKeys"))
    finally:
        dialog.close()

    assert "Cmd+Z" in text and "Cmd+0" in text and "Ctrl+" not in text and "Alt+" not in text


def test_stage_keys_stay_alt_n_and_add_cmd_option_n_on_macos():
    assert sc.STAGE_KEYS == tuple(f"Alt+{n}" for n in range(1, 6))
    assert sc.stage_keys("win32") == tuple((f"Alt+{n}",) for n in range(1, 6))
    assert sc.stage_keys("linux") == sc.stage_keys("win32")
    mac = sc.stage_keys("darwin")
    assert len(mac) == len(sc.STAGE_KEYS)
    assert mac[0] == ("Alt+1", "Ctrl+Alt+1")
    assert mac[1] == ("Alt+2", "Ctrl+Alt+2")


def test_stage_keys_default_to_sys_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert sc.stage_keys()[1] == ("Alt+2", "Ctrl+Alt+2")


def test_the_macos_registry_binds_both_stage_keys_without_a_conflict():
    registry = sc.build_registry("darwin")
    row = next(s for s in registry if s.action == "go_to_stage")

    # Column-major: key i is stage i for i < 5, so the window's slot list is five slots twice.
    assert row.keys == (*sc.STAGE_KEYS, *(f"Ctrl+Alt+{n}" for n in range(1, 6)))
    assert sc.find_conflicts(registry) == []
    assert sc.find_conflicts(sc.build_registry("win32")) == []
    plain = next(s for s in sc.build_registry("win32") if s.action == "go_to_stage")
    assert plain.keys == sc.STAGE_KEYS


def test_the_canvas_scopes_list_the_pinch_gesture():
    row = ("zoom_pinch", ("Pinch",), "Zoom about the fingers", sc.GESTURE)
    for scope in (sc.LIBRARY, sc.CORRECT):
        specs = [
            (s.action, s.keys, s.description, s.kind)
            for s in sc.specs_for(scope)
            if s.action == "zoom_pinch"
        ]
        assert specs == [row], scope
    assert sc.find_conflicts() == []
