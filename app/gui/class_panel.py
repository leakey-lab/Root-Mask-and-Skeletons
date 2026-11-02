"""Class panel of the class labels on the Correct page (spec Sec 3.3).

A vertical floating panel, pinned to the right edge of the host canvas and
vertically centred, showing one row per active class in registry order:
a number-key hint, a colour swatch, the class name and a box. The box is ticked on
the active paint class only, Ignore included (owner decision 2026-11-01: the tick is
the class you paint with; until then it meant "fully annotated on this image" and
could disagree with the highlight). The highlight and the tick come from one state,
the active key (:meth:`ClassPanel.set_active`), so they never disagree. Clicking a
row or its box asks for that class (``active_changed``); a box never changes
itself, so a click on the ticked box keeps it ticked. Neither does a toggle (what a
screen reader's Toggle action calls): it is put back, and a toggle on asks for the
class like a click. The gear button opens the class editor dialog.

A click never takes keyboard focus: a click on a row, box or the gear leaves
focus on the canvas, so Space still pans. Tab does reach the gear and then each
box, in class order (also after the classes change); the focused box draws the
theme's focus ring and Space picks its class (walkthrough W18: the boxes could be
reached only with the mouse). :meth:`ClassPanel.picked_by_key` tells the two apart
while ``active_changed`` is handled, so the page hands the keys back to the canvas
after a click only.

Long class names are elided at the right (the full name is the name's tooltip), so
the panel keeps a bounded width and the boxes never scroll out of sight; the rows
scroll vertically only.

This module never touches ``app/gui/widgets/*`` — it only consumes the
existing ``_FloatingPanel``/``IconButton``/``keep_canvas_focus``/``ElidedLabel`` primitives.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.gt.registry import ClassDef, Registry
from app.gui.widgets import tokens
from app.gui.widgets.controls import IconButton
from app.gui.widgets.focus import keep_canvas_focus
from app.gui.widgets.overlays import _FloatingPanel
from app.gui.widgets.text_fit import ElidedLabel, label_advance

_MARGIN = 14
_MAX_HEIGHT_FRACTION = 0.6
_BOX_TOOLTIP = "Paint with this class"
# A class name shows at most this wide (longer names are elided), and can shrink to
# the minimum when the rows scroll and the scroll bar takes its share.
_NAME_MAX_PX = 140
_NAME_MIN_PX = 36


class _NameLabel(ElidedLabel):
    """A class name elided at the right to the label's width; the full name is the
    tooltip while it is elided. ``text()`` is always the full name."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent, mode=Qt.TextElideMode.ElideRight)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        hint = super().sizeHint()
        return QSize(min(hint.width(), _NAME_MAX_PX), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(_NAME_MIN_PX, super().minimumSizeHint().height())

    def is_elided(self) -> bool:
        """True when the name does not fit the label's width whole (the painted rule, by the
        fractional width rounded up: a cut name always has its tooltip)."""
        return label_advance(self.fontMetrics(), self.text()) > self.contentsRect().width()

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        super().setText(text)
        self._sync_tooltip()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._sync_tooltip()

    def _sync_tooltip(self) -> None:
        self.setToolTip(self.text() if self.is_elided() else "")


class _ActiveBox(QCheckBox):
    """A row's box: ticked while its class is the active one (:meth:`show_active`), and
    never changed by the user. A click or Space emits ``clicked`` and leaves the box as it
    is, so the ticked box stays ticked when clicked again. ``toggle()`` / ``setChecked()``
    from elsewhere (Qt's accessible Toggle action, used by Narrator and NVDA) is put back at
    once; a toggle on emits ``toggle_requested``, which asks for the class like a click."""

    toggle_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._on = False  # what the box shows: its class is the active one
        self.by_key = False  # True while Space on the focused box clicks it
        self.toggled.connect(self._put_back)

    def show_active(self, on: bool) -> None:
        """Tick the box (its class is the active one) or untick it; nothing else does."""
        self._on = on
        self.setChecked(on)  # never emits clicked; toggled only when the state changes

    def nextCheckState(self) -> None:  # noqa: N802 - Qt override
        """No toggle of its own on a click: the panel ticks the active class's box."""

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Space clicks the box on its release: that click comes from the keyboard."""
        self.by_key = True
        try:
            super().keyReleaseEvent(event)
        finally:
            self.by_key = False

    def _put_back(self, checked: bool) -> None:
        """A change that :meth:`show_active` did not make is undone; a toggle on asks for
        the box's class (which then ticks it)."""
        if checked == self._on:
            return
        self.setChecked(self._on)  # toggled again with the shown state: returns above
        if checked:
            self.toggle_requested.emit()


class _ClassRow(QFrame):
    """One row of the class panel.

    A left click anywhere in the row, a click (or Space) on its box, or a toggle on of the
    box emits ``activated`` with ``by_key``: True for Space on the box and for a toggle (an
    assistive technology's action, where the keys are), False for a mouse click. The box
    shows ``set_active``'s state and nothing else.
    """

    activated = pyqtSignal(bool)

    def __init__(self, class_def: ClassDef, number: int | None, parent=None):
        super().__init__(parent)
        self.key = class_def.key
        self.setObjectName("classRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(8)

        hint = "0" if class_def.is_ignore else ("" if number is None else str(number))
        self._number_label = QLabel(hint)
        self._number_label.setFixedWidth(12)
        self._number_label.setStyleSheet(
            f"color: {tokens.TEXT_FAINT}; font-family: {tokens.MONO}; "
            "font-size: 11px; background: transparent; border: none;"
        )
        row.addWidget(self._number_label)

        self._swatch = QLabel()
        self._swatch.setFixedSize(12, 12)
        row.addWidget(self._swatch)

        self._name_label = _NameLabel()
        self._name_label.setStyleSheet(
            f"color: {tokens.TEXT}; font-size: 12.5px; background: transparent; border: none;"
        )
        row.addWidget(self._name_label, 1)

        self.checkbox = _ActiveBox()
        self.checkbox.setToolTip(_BOX_TOOLTIP + (f" (key {hint})" if hint else ""))
        self.checkbox.clicked.connect(lambda: self.activated.emit(self.checkbox.by_key))
        self.checkbox.toggle_requested.connect(lambda: self.activated.emit(True))
        row.addWidget(self.checkbox)
        # A click falls through to the nearest focusable ancestor, so the row
        # itself (not only the checkbox) must refuse focus.
        keep_canvas_focus(self, tab_reachable=False)
        # Tab reaches the box (Space picks its class); a click still leaves the focus alone.
        self.checkbox.setFocusPolicy(Qt.FocusPolicy.TabFocus)

        self._active = False
        self.update_class(class_def)
        self._apply_style()

    def update_class(self, class_def: ClassDef) -> None:
        self._name_label.setText(class_def.name)
        self._swatch.setStyleSheet(
            f"background-color: {class_def.color}; border-radius: 2px; border: none;"
        )

    def set_active(self, active: bool) -> None:
        """Highlight the row and tick its box (or neither): one state, never one alone."""
        if self._active == active:
            return
        self.checkbox.show_active(active)
        self._active = active
        self._apply_style()

    def _apply_style(self) -> None:
        if self._active:
            self.setStyleSheet(
                f"QFrame#classRow {{ background-color: {tokens.SEL_SOFT}; "
                f"border-left: 2px solid {tokens.SEL}; border-radius: 4px; }}"
            )
        else:
            self.setStyleSheet(
                "QFrame#classRow { background: transparent; "
                "border-left: 2px solid transparent; border-radius: 4px; }"
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(False)
        super().mousePressEvent(event)


class ClassPanel(_FloatingPanel):
    """Floating class panel, right edge of the host, vertically centred."""

    active_changed = pyqtSignal(str)
    edit_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, horizontal=False)
        self.setObjectName("classPanel")

        header = QWidget(self)
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(6, 2, 2, 2)
        header_row.setSpacing(6)
        title = QLabel("Classes")
        title.setStyleSheet(
            f"color: {tokens.TEXT_MUTED}; font-size: 11.5px; font-weight: 600; "
            "background: transparent; border: none;"
        )
        header_row.addWidget(title, 1)
        self._gear = IconButton("settings", "Edit classes")
        self._gear.clicked.connect(self.edit_requested.emit)
        header_row.addWidget(self._gear)
        self.add_widget(header)
        self._header = header

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Names elide instead: a sideways scroll would hide the boxes.
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("background: transparent; border: none;")
        self.add_widget(self._scroll)

        self._rows_container = QWidget()
        self._rows_container.setStyleSheet("background: transparent;")
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch(1)
        self._scroll.setWidget(self._rows_container)
        self._scroll.viewport().setStyleSheet("background: transparent;")
        # A click falls through to the nearest focusable ancestor, so every
        # layer (not only the checkbox) must refuse focus.
        keep_canvas_focus(self, tab_reachable=False)
        self._gear.setFocusPolicy(Qt.FocusPolicy.TabFocus)  # Tab only, like the boxes

        self._registry: Registry | None = None
        self._rows: dict[str, _ClassRow] = {}
        self._active_key: str | None = None
        self._by_key = False  # the pick being handled came from the keyboard

    # ------------------------------------------------------------------ #
    def set_registry(self, registry: Registry) -> None:
        """Rebuild rows from ``registry.active()``, in registry order."""
        self._registry = registry

        while self._rows_layout.count() > 1:
            item = self._rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                # Off-screen and out of the tree now: deleteLater alone leaves
                # the old row visible (and findable) until the event loop runs.
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._rows.clear()

        active_classes = registry.active()
        labelled_keys = [c.key for c in active_classes if not c.is_ignore]
        number_for_key = {key: i + 1 for i, key in enumerate(labelled_keys[:9])}

        for class_def in active_classes:
            number = number_for_key.get(class_def.key)
            row = _ClassRow(class_def, number)
            row.activated.connect(
                lambda by_key, k=class_def.key: self._handle_row_activated(k, by_key)
            )
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
            self._rows[class_def.key] = row

        # Tab: the gear, then each box in class order. New rows join the end of the
        # window's focus chain: they came after the whole window once the classes changed.
        previous = self._gear
        for row in self._rows.values():
            QWidget.setTabOrder(previous, row.checkbox)
            previous = row.checkbox

        if self._active_key is not None and self._active_key in self._rows:
            self._rows[self._active_key].set_active(True)

    def set_active(self, key: str) -> None:
        """Highlight ``key``'s row and tick its box, and only its box (nothing changes when
        it is already shown). Never emits ``active_changed``."""
        previous = self._active_key
        if previous is not None and previous != key and previous in self._rows:
            self._rows[previous].set_active(False)
        self._active_key = key
        if key in self._rows:
            self._rows[key].set_active(True)

    def active_key(self) -> str:
        return self._active_key

    def picked_by_key(self) -> bool:
        """True while ``active_changed`` is handled for a pick from the keyboard (Space on a
        focused box, or a toggle), False for a mouse click or outside a pick."""
        return self._by_key

    def ticked_keys(self) -> list[str]:
        """The classes whose box is ticked: the active class alone (none before one is set)."""
        return [key for key, row in self._rows.items() if row.checkbox.isChecked()]

    def key_for_shortcut(self, n: int) -> str | None:
        """``n=1..9`` -> n-th labelled class key; ``0`` -> ``"ignore"``."""
        if self._registry is None:
            return None
        if n == 0:
            return "ignore"
        labelled = self._registry.labelled()
        if 1 <= n <= len(labelled):
            return labelled[n - 1].key
        return None

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        """The width of the rows as they are now, plus the scroll bar's when they scroll.

        ``QScrollArea.sizeHint`` keeps the rows' first measurement (see
        :meth:`_content_height`), so after a registry change with longer names the panel
        stayed as narrow as before and squeezed them.
        """
        hint = super().sizeHint()
        header = self._header.sizeHint().width()
        # Margins and frame: what the layout adds around its widest item.
        chrome = hint.width() - max(header, self._scroll.sizeHint().width())
        width = max(header, self._rows_container.sizeHint().width())
        if self._will_scroll():
            width += self._scroll.verticalScrollBar().sizeHint().width()
        return QSize(chrome + width, hint.height())

    def _will_scroll(self) -> bool:
        """True when the rows are taller than the panel's cap in its host (they scroll)."""
        parent = self.parent()
        if parent is None:
            return False
        cap = int(parent.height() * _MAX_HEIGHT_FRACTION) - self._chrome_height()
        return self._content_height() > cap

    def preferred_height(self, host_height: int) -> int:
        """Height showing every row, capped at 60% of ``host_height`` (then it scrolls)."""
        chrome = self._chrome_height()
        cap = max(0, int(host_height * _MAX_HEIGHT_FRACTION) - chrome)
        return chrome + min(self._content_height(), cap)

    def fit_height(self, height: int) -> None:
        """Resize to ``height`` (never below the header); the rows scroll inside."""
        self.adjustSize()  # settles the natural width from the header row
        chrome = self._chrome_height()
        scroll_height = min(self._content_height(), max(0, height - chrome))
        self._scroll.setFixedHeight(scroll_height)
        self.resize(self.width(), chrome + scroll_height)

    def _place(self) -> None:
        """Standalone placement (no OverlayLayout): right edge, vertically centred."""
        parent_width = self.parent().width()
        parent_height = self.parent().height()
        self.fit_height(self.preferred_height(parent_height))
        x = parent_width - self.width() - _MARGIN
        y = max(0, (parent_height - self.height()) // 2)
        self.move(x, y)

    def _chrome_height(self) -> int:
        """Everything but the rows: margins, header and the gap below it."""
        margins = self._box.contentsMargins()
        return (
            margins.top()
            + margins.bottom()
            + self._box.spacing()
            + self._header.sizeHint().height()
        )

    def _content_height(self) -> int:
        # NOTE: QScrollArea.sizeHint() caches the contained widget's size the
        # first time it is computed and does not reliably invalidate that
        # cache without a real show/resize event cycle (which this in-window
        # overlay never gets while offscreen/headless). Measuring
        # ``_rows_container``'s own sizeHint() directly sidesteps that stale
        # cache so the panel shrinks correctly when the registry shrinks
        # (e.g. after retiring classes), not just when it grows.
        return self._rows_container.sizeHint().height()

    def _handle_row_activated(self, key: str, by_key: bool) -> None:
        """A row or its box was picked: tick and highlight it, then ask for its class
        (:meth:`picked_by_key` says how it was picked meanwhile)."""
        self.set_active(key)
        self._by_key = by_key
        try:
            self.active_changed.emit(key)
        finally:
            self._by_key = False
