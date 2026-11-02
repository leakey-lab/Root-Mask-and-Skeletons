"""Keyboard-focus policy: a click never takes the keyboard from the canvas.

The canvas editors read hold-keys (Space = pan, B = brush) from the focused canvas. A control
with Qt's default ``StrongFocus`` takes focus when clicked, so after clicking Save a Space
press clicked Save again and a B tap no longer selected the brush (audit UXE-03).

- :func:`use_tab_only_buttons` (called by ``main.main`` and by the test suite's QApplication)
  makes every button of the app Tab-only (owner item #44): a click leaves focus where it
  was, Tab still reaches the button. Except in a form: a click on a button whose click focus
  the style took away (push buttons, check boxes, radio buttons) still takes the keyboard from
  a text field of its window, so the field commits what was typed in it. Tool buttons never
  had click focus (Qt gives them Tab focus) and still never take the keyboard on a click.
- :func:`keep_canvas_focus` does the same for every control of an overlay, sliders and combo
  boxes too, or takes focus from them entirely.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QApplication, QProxyStyle, QWidget

#: Text fields that commit what was typed in them when they lose the keyboard: a line edit
#: emits ``editingFinished``, a spin box reads its number, an item view saves its open editor.
_TEXT_FIELDS = ("QLineEdit", "QAbstractSpinBox", "QTextEdit", "QPlainTextEdit")
#: Set by :func:`keep_canvas_focus` on its roots: no control below them takes the keyboard.
_CANVAS_CONTROLS = "sproutsKeepCanvasFocus"
#: Set by the style on a button whose click focus it took away: only such a button takes
#: the keyboard from a text field on a click. On the C++ object, so it outlives a stale
#: wrapper and a new style.
_CLICK_FOCUS_TAKEN = "sproutsClickFocusTaken"


def _restrict(widget: QWidget, allowed: Qt.FocusPolicy) -> bool:
    """Drop the focus reasons of ``widget`` outside ``allowed`` (never adds one).

    True when it dropped one.
    """
    policy = widget.focusPolicy()
    restricted = Qt.FocusPolicy(policy & allowed)
    if restricted == policy:
        return False
    widget.setFocusPolicy(restricted)
    return True


def _holds_typing(widget: QWidget | None) -> bool:
    """True when ``widget`` is a text field one can type in (not a read-only one)."""
    # QObject calls only (inherits, property): safe on a wrapper of the wrong type.
    return (
        widget is not None
        and any(widget.inherits(name) for name in _TEXT_FIELDS)
        and not widget.property("readOnly")
    )


def _in_canvas_controls(widget: QWidget | None) -> bool:
    while widget is not None:
        if widget.property(_CANVAS_CONTROLS):
            return True
        widget = widget.parentWidget()
    return False


class _CommitTyping(QObject):
    """Event filter of every button: a left press on a button whose click focus the style
    took away takes the keyboard from a text field of the button's window, before the button
    acts, as Qt's click focus did.

    The field then commits what was typed in it (the Classes editor commits a rename only
    then; without this, a click on OK dropped it). Anywhere else (the canvas, a list) the
    keyboard stays where it was. Buttons that never had click focus never take it: tool
    buttons (the top bar's stage buttons and Load leave it in the library's search field), a
    button made Tab-only or given no focus by hand, and the controls of a canvas overlay.
    """

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        if (
            event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and watched.isWidgetType()
            and watched.property(_CLICK_FOCUS_TAKEN)
            and watched.isEnabled()
            and watched.focusPolicy() & Qt.FocusPolicy.TabFocus
            and _holds_typing(watched.window().focusWidget())
            and not _in_canvas_controls(watched)
        ):
            watched.setFocus(Qt.FocusReason.MouseFocusReason)
        return False  # the button handles the press as always


class TabOnlyButtonStyle(QProxyStyle):
    """The application's style, except that no button takes focus on a click.

    Qt gives push buttons, check boxes and radio buttons ``StrongFocus`` (tool buttons have
    ``TabFocus``). Each button is restricted to Tab focus when it is polished, before it is
    first shown, and again whenever it is polished again (a new stylesheet); a button made
    ``NoFocus`` stays so. Sliders, combo boxes and text fields keep their policy: the Enhance
    popover's sliders and combo box keep click focus on purpose. (Not the style hint
    ``SH_Button_FocusPolicy``: Qt's sliders read it too.) A press on a button whose click
    focus was taken away still takes the keyboard from a text field (:class:`_CommitTyping`).
    """

    def __init__(self, key: str):
        super().__init__(key)
        self._commit_typing = _CommitTyping(self)

    def polish(self, target):  # QWidget, QPalette or QApplication: one Python override
        result = super().polish(target)
        # inherits() asks the live C++ object, not the Python wrapper: sip can hand back a
        # stale wrapper of another type at a reused address (see keep_canvas_focus).
        if isinstance(target, QWidget) and target.inherits("QAbstractButton"):
            if _restrict(target, Qt.FocusPolicy.TabFocus):
                target.setProperty(_CLICK_FOCUS_TAKEN, True)  # kept when polished again
            target.installEventFilter(self._commit_typing)  # once: Qt drops a repeat
        return result


def use_tab_only_buttons(app: QApplication, style_name: str | None = None) -> QProxyStyle:
    """Make every button of ``app`` Tab-only; call it before the stylesheet and any widget.

    ``style_name`` is the Qt style to keep drawing with (``qt_environment.qt_style_for()``:
    "Fusion" off Windows); None keeps the application's own (the native one on Windows).
    Wrapping changes no pixel. Returns the installed style (the same one when called again
    without ``style_name``).
    """
    current = app.style()
    if style_name is None and isinstance(current, TabOnlyButtonStyle):
        return current
    style = TabOnlyButtonStyle(style_name or current.name())
    app.setStyle(style)  # the application owns it now
    return style


def keep_canvas_focus(*widgets: QWidget, tab_reachable: bool = True) -> None:
    """Stop clicks on ``widgets`` and all their descendants from taking keyboard focus.

    Every focusable widget in those trees keeps only Tab focus (``tab_reachable=True``) or
    loses focus entirely (``tab_reachable=False``, for panels whose controls must never hold
    focus, such as the GT class panel). Widgets that never take focus (labels) are untouched.
    Their buttons, added later ones too, never take the keyboard from a text field either.
    """
    allowed = Qt.FocusPolicy.TabFocus if tab_reachable else Qt.FocusPolicy.NoFocus
    for root in widgets:
        root.setProperty(_CANVAS_CONTROLS, True)
        for widget in (root, *root.findChildren(QWidget)):
            # findChildren can hand back a stale wrapper typed as a widget for another object
            # now at its address (a C++-made child of a widget whose dead Python wrapper is
            # still held): touch widgets only.
            if not widget.isWidgetType():
                continue
            _restrict(widget, allowed)
