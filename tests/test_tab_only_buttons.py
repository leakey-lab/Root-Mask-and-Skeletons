"""Buttons are Tab-only app-wide (owner item #44): a click leaves the keyboard where it was,
Tab still reaches every button.

Qt gives push buttons, check boxes and radio buttons ``StrongFocus``, so a click took the
keyboard from the canvas and Space then clicked the button again (UXE-03, C5-03 / #9). The
point fixes (``widgets.keep_canvas_focus``) covered one control at a time;
``widgets.focus.use_tab_only_buttons`` installs a style that restricts every button to Tab
focus. ``main.main`` installs it, and so does the suite's QApplication (``conftest``), so
every test runs with it. Sliders, combo boxes and text fields keep their policy: the Enhance
popover's sliders and combo box keep click focus on purpose.

One exception keeps forms working: a click on a button that Qt gave click focus (push
buttons, check boxes, radio buttons) takes the keyboard from a text field of its window, as
before, because a text field commits what was typed in it when it loses the keyboard (a class
renamed in the Classes editor was dropped by a click on OK). Buttons that never had click
focus never take it: tool buttons (Qt gives them Tab focus; the top bar's stage buttons and
Load leave the keyboard in the library's search field), buttons made Tab-only by hand, and
the controls of a canvas overlay (``keep_canvas_focus``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QWidget,
)

from app.gt.registry import default_registry
from app.gui.class_editor_dialog import ClassEditorDialog
from app.gui.widgets.focus import keep_canvas_focus
from tests.qt_teardown import delete_widgets

REPO_ROOT = Path(__file__).resolve().parents[1]
TAB = Qt.FocusPolicy.TabFocus


class _Canvas(QWidget):
    """A canvas that holds the keyboard, as the editors' canvas does."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 120)


@pytest.fixture
def page(app):
    """A plain page (no overlay, no point fix): a canvas, then a push button, a check box, a
    radio button and a tool button, in a row. The canvas has the keyboard."""
    host = QWidget()
    row = QHBoxLayout(host)
    canvas = _Canvas()
    save, check, radio, tool = (
        QPushButton("Save"),
        QCheckBox("Grid"),
        QRadioButton("Mask"),
        QToolButton(),
    )
    tool.setText("Tool")
    for widget in (canvas, save, check, radio, tool):
        row.addWidget(widget)
    host.show()
    host.activateWindow()
    assert QTest.qWaitForWindowActive(host, 3000)
    canvas.setFocus()
    yield host, canvas, (save, check, radio, tool)
    delete_widgets([host])


def test_buttons_are_tab_only_and_other_controls_keep_their_policy(app):
    widgets = {
        "push": QPushButton("Save"),
        "check": QCheckBox("Grid"),
        "radio": QRadioButton("Mask"),
        "tool": QToolButton(),
        "slider": QSlider(Qt.Orientation.Horizontal),
        "combo": QComboBox(),
        "line": QLineEdit(),
        "spin": QSpinBox(),
    }
    for widget in widgets.values():
        widget.ensurePolished()  # the policy is set when a widget is polished
    try:
        policies = {name: widget.focusPolicy() for name, widget in widgets.items()}
        assert policies == {
            "push": TAB,
            "check": TAB,
            "radio": TAB,
            "tool": TAB,
            "slider": Qt.FocusPolicy.StrongFocus,  # the Enhance sliders keep click focus
            "combo": Qt.FocusPolicy.WheelFocus,
            "line": Qt.FocusPolicy.StrongFocus,
            "spin": Qt.FocusPolicy.WheelFocus,
        }
    finally:
        delete_widgets(list(widgets.values()), close=False)


def test_a_button_given_no_focus_keeps_no_focus(app):
    button = QPushButton("Remove")
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # e.g. a toast's close button
    button.ensurePolished()
    try:
        assert button.focusPolicy() == Qt.FocusPolicy.NoFocus
    finally:
        delete_widgets([button], close=False)


def test_a_button_restyled_later_stays_tab_only(app):
    button = QPushButton("Export")
    button.ensurePolished()
    button.setStyleSheet("QPushButton { border: 1px solid red; }")  # polished again
    try:
        assert button.focusPolicy() == TAB
    finally:
        delete_widgets([button], close=False)


@pytest.mark.parametrize("index", range(4), ids=["push", "check", "radio", "tool"])
def test_a_click_on_any_button_leaves_the_keyboard_on_the_canvas(page, index):
    _host, canvas, buttons = page
    QTest.mouseClick(buttons[index], Qt.MouseButton.LeftButton)
    assert QApplication.focusWidget() is canvas


def test_space_after_clicking_a_button_does_not_click_it_again(page):
    _host, canvas, (save, *_rest) = page
    clicks = []
    save.clicked.connect(lambda: clicks.append(1))
    QTest.mouseClick(save, Qt.MouseButton.LeftButton)
    QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_Space)
    assert clicks == [1]


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS: Tab follows the system setting")
def test_tab_still_reaches_every_button(page):
    # On macOS, Tab reaches buttons only when "Keyboard navigation" is on in the system
    # settings, as in every native app (Qt's tabFocusBehavior).
    _host, canvas, buttons = page
    reached = []
    for _ in buttons:
        QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_Tab)
        reached.append(QApplication.focusWidget())
    assert reached == list(buttons)


def _show_active(window: QWidget) -> None:
    window.show()
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window, 3000)


def test_a_rename_typed_in_the_classes_editor_survives_a_click_on_ok(app):
    """The Classes editor commits a row's new name when its field loses the keyboard; a click
    on OK must still take it, or the rename is dropped without a word (review of #44)."""
    registry = default_registry()
    dialog = ClassEditorDialog(registry)
    try:
        _show_active(dialog)
        name = registry.by_key("nodule").name
        [field] = [f for f in dialog.findChildren(QLineEdit) if f.text() == name]
        field.setFocus()
        field.selectAll()
        QTest.keyClicks(field, "Nodule X")
        ok = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)
        QTest.mouseClick(ok, Qt.MouseButton.LeftButton)
        registry = dialog.result_registry()
        assert registry is not None
        assert registry.by_key("nodule").name == "Nodule X"
    finally:
        delete_widgets([dialog])


@pytest.fixture
def form(app):
    """A form: a text field (it has the keyboard), a spin box that reads what was typed only
    when it loses the keyboard, and an Apply button."""
    host = QWidget()
    row = QHBoxLayout(host)
    field, spin, apply = QLineEdit(), QSpinBox(), QPushButton("Apply")
    spin.setKeyboardTracking(False)
    spin.setRange(0, 100)
    for widget in (field, spin, apply):
        row.addWidget(widget)
    _show_active(host)
    field.setFocus()
    yield host, field, spin, apply
    delete_widgets([host])


def test_a_click_takes_the_keyboard_from_a_text_field_and_commits_the_typing(form):
    _host, field, _spin, apply = form
    events = []
    field.editingFinished.connect(lambda: events.append(("committed", field.text())))
    apply.clicked.connect(lambda: events.append(("clicked", field.text())))
    QTest.keyClicks(field, "nodule")
    QTest.mouseClick(apply, Qt.MouseButton.LeftButton)
    assert events == [("committed", "nodule"), ("clicked", "nodule")]
    assert QApplication.focusWidget() is apply


def test_a_click_reads_the_number_typed_in_a_spin_box(form):
    _host, _field, spin, apply = form
    seen = []
    apply.clicked.connect(lambda: seen.append(spin.value()))
    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, "42")
    QTest.mouseClick(apply, Qt.MouseButton.LeftButton)
    assert seen == [42]


@pytest.mark.parametrize("case", ["read-only field", "overlay button", "no-focus button"])
def test_a_click_leaves_the_keyboard_in_the_field_when_nothing_is_typed_or_on_an_overlay(
    form, case
):
    host, field, _spin, apply = form
    if case == "read-only field":
        field.setReadOnly(True)  # nothing typed there to commit
    elif case == "overlay button":
        overlay = QWidget(host)  # a canvas overlay: its controls never take the keyboard
        keep_canvas_focus(overlay)
        apply = QPushButton("Save", overlay)  # added later, as the overlays' buttons can be
        overlay.show()
    else:
        apply.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    field.setFocus()
    QTest.mouseClick(apply, Qt.MouseButton.LeftButton)
    assert QApplication.focusWidget() is field


@pytest.fixture
def typing_row(app):
    """``typing_row(button)``: a line edit with the keyboard, ``button`` beside it."""
    hosts = []

    def make(button: QAbstractButton) -> QLineEdit:
        host = QWidget()
        row = QHBoxLayout(host)
        field = QLineEdit()
        row.addWidget(field)
        row.addWidget(button)
        hosts.append(host)
        _show_active(host)
        field.setFocus()
        return field

    yield make
    delete_widgets(hosts)


@pytest.mark.parametrize("kind", ["push", "check", "radio", "push restyled"])
def test_a_button_qt_gave_click_focus_still_takes_the_keyboard_from_a_text_field(typing_row, kind):
    if kind == "push restyled":  # polished again: Tab-only by then, it still commits
        button = QPushButton("Apply")
        button.ensurePolished()
        button.setStyleSheet("QPushButton { border: 1px solid red; }")
    else:
        button = {
            "push": QPushButton("Apply"),
            "check": QCheckBox("Grid"),
            "radio": QRadioButton("Mask"),
        }[kind]
    field = typing_row(button)
    committed = []
    field.editingFinished.connect(lambda: committed.append(field.text()))
    QTest.keyClicks(field, "nodule")
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert committed == ["nodule"]
    assert QApplication.focusWidget() is button


@pytest.mark.parametrize("case", ["tool button", "push button made Tab-only by hand"])
def test_a_button_that_never_had_click_focus_leaves_the_keyboard_in_the_text_field(
    typing_row, case
):
    """Qt gives tool buttons Tab focus, so a click never took the keyboard from the library's
    search field (the top bar's stage buttons, Load); the Tab-only style must not start to."""
    if case == "tool button":
        button = QToolButton()  # Qt's default policy: TabFocus
        button.setText("Load")
    else:
        button = QPushButton("Load")
        button.setFocusPolicy(TAB)  # before it is polished, as the point fixes did
    field = typing_row(button)
    clicks = []
    button.clicked.connect(lambda: clicks.append(1))
    QTest.keyClicks(field, "nod")
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert clicks == [1]
    assert QApplication.focusWidget() is field
    assert field.text() == "nod"


def test_main_makes_buttons_tab_only_before_the_window_exists(tmp_path):
    """``main.main`` installs the style (over the style it sets) before the window is made."""
    driver = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        import main
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication, QCheckBox, QMainWindow, QPushButton, QSlider

        class StubWindow(QMainWindow):
            def __init__(self):
                super().__init__()
                made = [QPushButton("Save", self), QCheckBox("Grid", self), QSlider(self)]
                for widget in made:
                    widget.ensurePolished()
                print("POLICIES", *(widget.focusPolicy().name for widget in made))
                QTimer.singleShot(50, QApplication.instance().quit)

        main.MainWindow = StubWindow
        print("RC", main.main(["sprouts"]))
        """
    )
    env = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        PYTHONDONTWRITEBYTECODE="1",
        SPROUTS_LOG_DIR=str(tmp_path / "logs"),
        SPROUTS_CONFIG_DIR=str(tmp_path / "config"),
    )
    proc = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.splitlines()
    assert "RC 0" in out, proc.stdout + proc.stderr
    assert "POLICIES TabFocus TabFocus StrongFocus" in out, proc.stdout
