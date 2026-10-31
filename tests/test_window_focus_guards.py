"""Focus guards of the window (Qt toolkit item 4): the click-focus audit of the Correct page
and of every button, and the focus-ring rule of widget stylesheets. Allowed offenders are
listed by issue id (none since C5-03 and K1b-01 were fixed; their tests stay as regression
guards).

Click focus (UXE-03). The canvas holds the hold keys (Space pans, B sizes the brush) and the
gesture keys; a control with click focus takes them, and Space then clicks it again. Qt gives
each control its own focus policy (buttons, sliders and combo boxes default to StrongFocus)
and has no focus scope. Every button of the app is Tab-only (owner item #44,
``widgets.focus.use_tab_only_buttons``, installed by ``main.main`` and the suite's
QApplication), and every other control over the canvas and on the top bar's Correct page must
be Tab-only too (``widgets.keep_canvas_focus``). Allowed: the Enhance popover's content (its
combo box and sliders keep click focus on purpose) and text fields.

Focus rings (UXS-08). A widget's own stylesheet beats the application's whatever the
specificity, so a widget stylesheet that sets ``border`` on a focusable control hides the
app's ``:focus`` ring unless it carries its own ``:focus`` rule (``tokens.focus_rule``, added
by hand in 12 stylesheets). Every widget stylesheet of the window and its dialogs is read:
each border rule that reaches a focusable control needs a ``:focus`` rule for that control.

The window has the first example open on the Correct page, with every layer made active and
every tool picked, so the controls made late (size sliders, class rows) are there.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)

from tests.qt_teardown import delete_widgets
from tests.qt_wait import settle
from tests.window_support import build_window, open_folder, quiet_window

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
TEXT_FIELDS = (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)
#: Top-bar controls of the Correct page allowed to take click focus, by issue (none today).
KNOWN_CLICK_FOCUS: dict[str, str] = {}
#: Focusable controls allowed a widget-stylesheet border without a ``:focus`` rule (none today).
KNOWN_NO_RING: dict[str, str] = {}
#: The status line's Cancel button as the ring rule names it (K1b-01).
CANCEL_BUTTON = "TaskProgressWidget > QToolButton 'Cancel'"


@pytest.fixture(scope="module")
def window(app, tmp_path_factory):
    """The window with the first example open on the Correct page, every layer and tool
    visited."""
    tmp_path = tmp_path_factory.mktemp("focus_guards")
    project = shutil.copytree(EXAMPLES, tmp_path / "project")
    made = [build_window(tmp_path / "settings.ini", detect_device=False, warm_up=False)]
    window = made[0]
    open_folder(window, project, 3)
    leaf = window._image_leaves()[0]
    window.file_list.setCurrentItem(leaf)
    window.on_image_selected(leaf)
    window.toggle_correct()
    settle()
    editor = window.correct_editor
    assert editor.session.image_path  # the image is open on the Correct page
    for layer_id in editor.layers:
        editor.set_active_layer(layer_id)
        settle()
        for (owner, _tool), button in editor.controls.tool_buttons.items():
            if owner == layer_id and button.isEnabled():
                button.click()
                settle()
    yield window
    del window
    delete_widgets(made, prepare=quiet_window)


# --------------------------------------------------------------------------- click focus
def _takes_click_focus(widget: QWidget) -> bool:
    return bool(widget.focusPolicy() & Qt.FocusPolicy.ClickFocus)


def _attr_name(window, widget: QWidget) -> str:
    """The window attribute naming ``widget``, else its class, object name and text."""
    for attr in ("gt_classes_button", "gt_export_button", "gt_remove_button"):
        if getattr(window, attr, None) is widget:
            return attr
    text = getattr(widget, "text", lambda: "")()
    return f"{type(widget).__name__}#{widget.objectName()}:{text}"


def _canvas_click_focus(window) -> list[str]:
    """Controls over the Correct canvas (every overlay role) that take click focus, outside
    the Enhance content and text fields. Popups (their own windows) are left out."""
    chrome = window.correct_editor.chrome
    popover = chrome.popover
    offenders = []
    for role, overlay in chrome._overlays.items():
        for widget in (overlay, *overlay.findChildren(QWidget)):
            if not widget.isWidgetType() or widget.window() is not overlay.window():
                continue
            in_enhance = popover.isAncestorOf(widget) and widget is not popover
            if (
                _takes_click_focus(widget)
                and not in_enhance
                and not isinstance(widget, TEXT_FIELDS)
            ):
                offenders.append(f"{role}: {_attr_name(window, widget)}")
    return offenders


def _topbar_click_focus(window) -> list[str]:
    """Controls on the top bar's Correct page that take click focus."""
    page = window.action_bar_stack.widget(window._action_bar_pages["Correct"])
    return [
        _attr_name(window, widget)
        for widget in page.findChildren(QWidget)
        if _takes_click_focus(widget) and not isinstance(widget, TEXT_FIELDS)
    ]


def test_the_audit_sees_the_controls_over_the_canvas(window):
    overlays = window.correct_editor.chrome._overlays
    assert {"rail", "dock", "popover", "zoom_cluster", "side_panel"} <= set(overlays)
    tab_only = [
        widget
        for overlay in overlays.values()
        for widget in overlay.findChildren(QWidget)
        if widget.focusPolicy() == Qt.FocusPolicy.TabFocus
    ]
    assert len(tab_only) >= 20  # rail, dock, popover buttons, zoom cluster, side card


def test_no_control_over_the_correct_canvas_takes_the_keyboard_on_a_click(window):
    assert _canvas_click_focus(window) == []


def test_no_new_correct_top_bar_control_takes_the_keyboard_on_a_click(window):
    offenders = _topbar_click_focus(window)
    assert set(offenders) - set(KNOWN_CLICK_FOCUS) == set()
    assert "gt_remove_button" not in offenders  # Remove GT… leaves the keys on the canvas


def test_classes_and_export_dataset_leave_the_keyboard_on_the_canvas(window):
    # C5-03 (owner question #9): after a click, Space must not open the dialog again.
    assert _topbar_click_focus(window) == []
    for button in (window.gt_classes_button, window.gt_export_button):
        assert button.focusPolicy() == Qt.FocusPolicy.TabFocus  # Tab still reaches it


def _buttons_with_click_focus(root: QWidget) -> list[str]:
    """Buttons under ``root`` that take click focus once polished (as before they are first
    shown; buttons on pages never shown are not polished yet)."""
    offenders = []
    for button in root.findChildren(QAbstractButton):
        if not button.isWidgetType():  # a stale wrapper (see widgets.focus.keep_canvas_focus)
            continue
        button.ensurePolished()
        if _takes_click_focus(button):
            offenders.append(_described(button))
    return offenders


def test_no_button_of_the_window_or_its_dialogs_takes_the_keyboard_on_a_click(window, dialogs):
    # Owner item #44: every button is Tab-only, not only those over the canvas.
    buttons = window.findChildren(QAbstractButton)
    assert len(buttons) > 50  # the audit sees the whole window
    offenders = _buttons_with_click_focus(window)
    for dialog in dialogs:
        offenders += _buttons_with_click_focus(dialog)
    assert offenders == []


# --------------------------------------------------------------------------- focus rings
_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_RULE = re.compile(r"([^{}]*)\{([^{}]*)\}")
_SETS_BORDER = re.compile(r"(?:^|;)\s*border[\w-]*\s*:", re.I)
_COMPOUND = re.compile(r"(?P<type>[A-Za-z_][\w]*|\*)?(?P<id>#[\w-]+)?")
_PROPERTY = re.compile(r'\[(?P<name>[\w-]+)(?:="?(?P<value>[^"\]]*)"?)?\]')


def _rules(sheet: str) -> list[tuple[str, str]]:
    """``(selector, declarations)`` of a stylesheet; bare declarations get the selector ""
    (they style the widget itself)."""
    text = _COMMENT.sub("", sheet)
    rules = [(selector.strip(), body) for selector, body in _RULE.findall(text)]
    bare = _RULE.sub("", text).strip()
    return rules + ([("", bare)] if bare else [])


def _matches(widget: QWidget, selector: str) -> bool:
    """Whether the last compound of ``selector`` (type, ``#name``, ``[property]``) selects
    ``widget``; pseudo-states and sub-controls are left out."""
    words = selector.split()
    if not words:
        return True
    last = words[-1]
    found = _COMPOUND.match(re.split(r"::?|\[", last, maxsplit=1)[0])
    kind, name = found.group("type"), (found.group("id") or "")[1:]
    if kind and kind != "*" and not widget.inherits(kind):
        return False
    if name and widget.objectName() != name:
        return False
    for prop in _PROPERTY.finditer(last):
        value = widget.property(prop.group("name"))
        wanted = prop.group("value")
        if value is None or (wanted is not None and str(value).lower() != wanted.lower()):
            return False
    return True


def _focusable(widget: QWidget) -> bool:
    return widget.isWidgetType() and bool(widget.focusPolicy() & Qt.FocusPolicy.TabFocus)


def _described(widget: QWidget) -> str:
    """``Parent > Class 'text'`` (the text only when the control has one)."""
    parent = widget.parentWidget()
    text = getattr(widget, "text", lambda: "")()
    owner = f"{type(parent).__name__} > " if parent is not None else ""
    return f"{owner}{type(widget).__name__}" + (f" {text!r}" if text else "")


def _rings_missing(root: QWidget) -> set[str]:
    """Focusable controls under ``root`` whose border is set by a widget stylesheet that has
    no ``:focus`` rule for them. A border only for ``:disabled`` cannot hide a ring."""
    missing = set()
    for owner in (root, *root.findChildren(QWidget)):
        if not owner.isWidgetType() or not owner.styleSheet():
            continue
        rules = _rules(owner.styleSheet())
        ringed = [s.strip() for sel, _ in rules for s in sel.split(",") if ":focus" in s]
        reach = [w for w in (owner, *owner.findChildren(QWidget)) if _focusable(w)]
        for selector, body in rules:
            if not _SETS_BORDER.search(body):
                continue
            for part in (s.strip() for s in selector.split(",")):
                if ":focus" in part or ":disabled" in part:
                    continue
                targets = [owner] if not part else [w for w in reach if _matches(w, part)]
                for target in targets:
                    if _focusable(target) and not any(_matches(target, s) for s in ringed):
                        missing.add(_described(target))
    return missing


@pytest.fixture(scope="module")
def dialogs(window, tmp_path_factory):
    """The dialogs with widget stylesheets of their own (built, never shown)."""
    from app.gt import default_registry
    from app.gui.class_editor_dialog import ClassEditorDialog
    from app.gui.empty_state import ShortcutsDialog
    from app.gui.export_dataset_dialog import ExportDatasetDialog

    folder = str(tmp_path_factory.mktemp("dialogs"))
    made = [
        ClassEditorDialog(default_registry(), window),
        ShortcutsDialog(window),
        ExportDatasetDialog(folder, {}, default_registry(), window),
    ]
    yield list(made)
    delete_widgets(made, close=False)


def test_the_ring_rule_reads_the_widget_stylesheets(window):
    # The rule sees the 11 hand-made rings (e.g. the stage buttons, the dock's buttons).
    sheets = [w for w in window.findChildren(QWidget) if w.styleSheet()]
    assert len(sheets) > 20
    ringed = [w for w in sheets if ":focus" in w.styleSheet()]
    assert len(ringed) >= 5


def test_every_focusable_control_with_a_widget_border_keeps_a_focus_ring(window, dialogs):
    missing = set(_rings_missing(window))
    for dialog in dialogs:
        missing |= _rings_missing(dialog)
    assert missing - set(KNOWN_NO_RING) == set()


def test_the_status_line_cancel_button_keeps_a_focus_ring(window):
    # K1b-01 (owner question #50): its stylesheet sets a border, so it carries the ring too.
    assert window.task_progress.cancel_button.focusPolicy() & Qt.FocusPolicy.TabFocus
    assert CANCEL_BUTTON not in _rings_missing(window)
