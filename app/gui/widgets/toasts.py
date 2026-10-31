"""Toasts: the capsule (owner choice of 2026-10-31: Sonner's stack, Coloplast's countdown).

- A toast is a pill: a countdown ring around the kind's glyph, the text, at most one action
  button (:class:`~app.feedback.NoticeAction`; the toast closes after it runs) and a close
  button on every kind. A click anywhere else on a toast dismisses it too.
- A save's message, "<what> · <file name>", shows its title and then the file name in mono
  on one line, the name elided in the middle (its start and extension stay); when the name
  has no room beside its title it takes a line of its own. Other messages wrap at
  ``min(460, host width - 2 * margin)``, an unbreakable token such as a path elided in the
  middle. A toast taller than one line keeps rounded corners instead of round ends. The full
  message is in the tooltip and the accessible name.
- Lifetimes by kind: success / info expire (4 s by default), warn lasts at least 6 s, danger
  stays until closed (its ring shows no countdown). The ring counts the lifetime down. Every
  countdown pauses while the pointer is over the stack and while the application is in the
  background, and goes on from where it stopped.
- :class:`ToastManager` stacks at most ``max_visible`` toasts bottom-right of a host; a new
  one pushes out the oldest, but a ``danger`` toast outlives lower-severity ones. The newest
  is in front and the older ones tuck behind it (``PEEK`` px higher and ``TUCK_SHRINK``
  narrower per step, their text hidden) until the pointer is over the stack, which fans it
  out. ``show(..., key=...)`` updates a live toast with the same key in place (the GT
  autosave on every PgDn), restarts its countdown and flashes it.
- ``set_bottom_offset`` keeps the stack above bottom chrome (status bars, a dock) and
  ``set_right_offset`` left of right-hand chrome (an editor's side panel);
  :meth:`ToastManager.column_left` / :meth:`ToastManager.stack_height` tell the host which
  chrome the stack would cover (its fanned-out height, so it does not jump when it fans out).
- Motion: a new toast slides up into place, the stack glides when it changes and a dismissed
  toast sinks as it fades. On the offscreen platform (tests, scripts) toasts are placed at
  once (:func:`motion_default`); the fades stay.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import (
    QEasingCurve,
    QElapsedTimer,
    QObject,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QBoxLayout,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from app.feedback import NoticeAction

from . import tokens
from .controls import IconButton
from .text_fit import elide_long_words, label_advance, wrap_label

DEFAULT_TIMEOUT_MS = 4000
WARN_MIN_TIMEOUT_MS = 6000
MAX_TOAST_WIDTH = 460
#: A save's toast names its file after this: "Saved mask · <file name>".
NAME_SEPARATOR = " · "
#: The least width a file name keeps beside its title; with less it takes its own line.
MIN_NAME_PX = 80
#: How far each older toast of a collapsed stack shows above the one in front of it.
PEEK = 14
#: How much narrower each older toast of a collapsed stack is (a fraction of the front's).
TUCK_SHRINK = 0.05
#: A toast up to this tall is a pill (round ends); a taller one gets RADIUS_LG corners.
PILL_MAX_HEIGHT = 48
MOTION_MS = 400
FADE_MS = 200
FLASH_MS = 420
#: How often a running countdown redraws its ring.
TICK_MS = 33
#: How long the pointer may be off the stack (between two toasts) before it collapses.
HOVER_GRACE_MS = 80
#: Gap between the title and the file name on one line.
TEXT_GAP = 8

_KIND_COLOR = {
    "success": tokens.OK,
    "info": tokens.INFO,
    "warn": tokens.WARN,
    "danger": tokens.DANGER,
}
# Glyph strokes in a 20 x 20 box: check, i, !, x.
_GLYPHS: dict[str, tuple[tuple[tuple[float, float], ...], ...]] = {
    "success": (((5, 10.5), (8.2, 13.7), (15, 6.8)),),
    "info": (((10, 9.2), (10, 14.2)), ((10, 6.1), (10, 6.2))),
    "warn": (((10, 5.6), (10, 11)), ((10, 14.2), (10, 14.3))),
    "danger": (((6.6, 6.6), (13.4, 13.4)), ((13.4, 6.6), (6.6, 13.4))),
}
# Set on the toast: its own widgets stay transparent whatever a host's stylesheet says (the
# pill is painted by Toast.paintEvent).
_TOAST_QSS = f"""
    QWidget#toast, QWidget#toastContent, QWidget#toastTextBox {{
        background: transparent; border: none;
    }}
    QLabel#toastTitle {{
        color: {tokens.TEXT}; background: transparent; font-size: 13px; font-weight: 500;
    }}
    QLabel#toastDetail {{
        color: {tokens.TEXT_MUTED}; background: transparent;
        font-family: {tokens.MONO}; font-size: 11.5px;
    }}
    QPushButton#toastAction {{
        color: {tokens.ACCENT}; background: transparent; border: none; border-radius: 12px;
        padding: 0 8px; min-height: 24px; max-height: 24px; font-size: 12.5px; font-weight: 500;
    }}
    QPushButton#toastAction:hover {{ background-color: {tokens.ACCENT_SOFT}; }}
    QPushButton#toastAction:pressed {{ color: {tokens.ACCENT_PRESS}; }}
"""
_CLOSE_QSS = f"""
    QToolButton#iconBtn {{ background: transparent; border: none; border-radius: 11px; }}
    QToolButton#iconBtn:hover {{ background-color: {tokens.BG_HOVER}; }}
"""


def motion_default() -> bool:
    """Toasts move unless Qt runs offscreen (tests, scripts), where each is placed at once."""
    return QGuiApplication.platformName() != "offscreen"


def split_message(message: str) -> tuple[str, str | None]:
    """``(title, file name)`` of a save's "<what> · <file name>" message, else
    ``(message, None)``: the part after the last " · " must be one word after a title."""
    lead, separator, name = message.rpartition(NAME_SEPARATOR)
    if not separator or not lead.strip() or not name or any(char.isspace() for char in name):
        return message, None
    return lead, name


def toast_timeout(kind: str, requested: int | None) -> int | None:
    """Lifetime of a ``kind`` toast in ms; ``None`` = stays until closed."""
    if kind == "danger":
        return None
    timeout = DEFAULT_TIMEOUT_MS if requested is None else int(requested)
    return max(timeout, WARN_MIN_TIMEOUT_MS) if kind == "warn" else timeout


def _fixed_width(widget: QWidget) -> int:
    """The width a layout gives ``widget``: its fixed width, else its size hint's."""
    if widget.minimumWidth() == widget.maximumWidth():
        return widget.minimumWidth()
    return widget.sizeHint().width()


class CountdownRing(QWidget):
    """The kind's glyph in a ring that empties, clockwise from the top, as the toast's
    lifetime runs out. ``progress()`` is the share left (``None``: no countdown)."""

    SIZE = 26

    def __init__(self, kind: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self._color = QColor(_KIND_COLOR.get(kind, tokens.OK))
        self._strokes = _GLYPHS.get(kind, _GLYPHS["info"])
        self._progress: float | None = 1.0

    def progress(self) -> float | None:
        return self._progress

    def set_progress(self, value: float | None) -> None:
        value = None if value is None else min(1.0, max(0.0, float(value)))
        if value != self._progress:
            self._progress = value
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        box = QRectF(2.5, 2.5, 21, 21)
        track = QColor(self._color)
        track.setAlphaF(0.24)
        painter.setPen(QPen(track, 2))
        painter.drawEllipse(box)
        if self._progress:
            pen = QPen(self._color, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(box, 90 * 16, -round(self._progress * 360 * 16))
        pen = QPen(self._color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        for stroke in self._strokes:  # the 20 px glyph box drawn at 14 px, centred
            painter.drawPolyline(QPolygonF([QPointF(6 + x * 0.7, 6 + y * 0.7) for x, y in stroke]))
        painter.end()


class Toast(QWidget):
    """One capsule toast (see the module docstring).

    It paints its own pill; the text and buttons sit on a content child, which a tucked
    toast hides. It fades through a QGraphicsOpacityEffect (windowOpacity is unreliable on
    some Qt platforms) and carries no drop shadow: Qt cannot nest graphics effects. A dark
    outline under the border keeps the pill apart from a bright photo instead.
    ``dismissed`` fires when it starts fading out, ``closed`` when it is gone and
    ``hover_changed`` when the pointer enters or leaves it.
    """

    closed = pyqtSignal()
    dismissed = pyqtSignal()
    hover_changed = pyqtSignal(bool)

    def __init__(
        self,
        parent,
        message: str,
        kind: str = "success",
        timeout: int | None = DEFAULT_TIMEOUT_MS,
        *,
        margin: int = 16,
        max_width: int = MAX_TOAST_WIDTH,
        action: NoticeAction | None = None,
        motion: bool | None = None,
    ):
        super().__init__(parent)
        self._margin = margin
        self._max_width = max_width
        self._kind = kind
        self._message = ""
        self._motion = motion_default() if motion is None else bool(motion)
        self._duration = toast_timeout(kind, timeout)
        self._remaining = float(self._duration or 0)
        self._pauses: set[str] = set()
        self._running = False
        self._shown = False
        self._dismissing = False
        self._placed = False
        self._tucked = False
        self._natural = QSize(0, 0)
        self._flash = 0.0
        self._action = action
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setObjectName("toast")
        self.setStyleSheet(_TOAST_QSS)

        content = QWidget(self)
        content.setObjectName("toastContent")
        row = QHBoxLayout(content)
        row.setContentsMargins(7, 6, 8, 6)
        row.setSpacing(10)
        self.ring = CountdownRing(kind)
        row.addWidget(self.ring, 0, Qt.AlignmentFlag.AlignVCenter)
        text = QWidget()
        text.setObjectName("toastTextBox")
        self._text_box = QBoxLayout(QBoxLayout.Direction.LeftToRight, text)
        self._text_box.setContentsMargins(0, 0, 0, 0)
        self._text_box.setSpacing(TEXT_GAP)
        self._title = QLabel()
        self._title.setObjectName("toastTitle")
        self._title.setTextFormat(Qt.TextFormat.PlainText)
        self._detail = QLabel()
        self._detail.setObjectName("toastDetail")
        self._detail.setTextFormat(Qt.TextFormat.PlainText)
        self._detail.hide()
        self._text_box.addWidget(self._title, 0, Qt.AlignmentFlag.AlignVCenter)
        self._text_box.addWidget(self._detail, 0, Qt.AlignmentFlag.AlignVCenter)
        self._text_box.addStretch(1)
        row.addWidget(text, 1)
        self.action_button: QPushButton | None = None
        if action is not None:
            self.action_button = QPushButton(action.label)
            self.action_button.setObjectName("toastAction")
            self.action_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.action_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.action_button.clicked.connect(self._run_action)
            row.addWidget(self.action_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.close_button = IconButton("xmark", "Dismiss", size=22, icon_px=12)
        self.close_button.setStyleSheet(_CLOSE_QSS)
        self.close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_button.clicked.connect(self.dismiss)
        row.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self._content, self._row = content, row

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)
        self._fade = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._glide = QPropertyAnimation(self, b"geometry", self)
        self._glide.setDuration(MOTION_MS)
        self._glide.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._flasher = QPropertyAnimation(self, b"flash", self)
        self._flasher.setDuration(FLASH_MS)
        self._flasher.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._clock = QElapsedTimer()
        self._ticker = QTimer(self)
        self._ticker.setInterval(TICK_MS)
        self._ticker.timeout.connect(self._tick)
        self.ring.set_progress(None if self._duration is None else 1.0)
        self.set_message(message)

    # ------------------------------------------------------------------ API
    def kind(self) -> str:
        return self._kind

    def message(self) -> str:
        return self._message

    def set_message(self, message: str) -> None:
        """Show ``message`` (see the module docstring); full text in tooltip and name."""
        self._message = message
        self.setToolTip(message)
        self.setAccessibleName(message)
        self._layout_text()

    def set_max_width(self, width: int) -> None:
        width = max(120, int(width))
        if width != self._max_width:
            self._max_width = width
            self._layout_text()

    def lifetime(self) -> int | None:
        """The toast's lifetime in ms; ``None`` = stays until closed."""
        return self._duration

    def natural_size(self) -> QSize:
        """The size the toast takes with its text shown (a tucked toast is smaller)."""
        return QSize(self._natural)

    def tucked(self) -> bool:
        """True while the toast is tucked behind a newer one (its text hidden)."""
        return self._tucked

    def restart(self, timeout: int | None = None) -> None:
        """Restart the lifetime (``timeout`` as in the constructor; ``None`` keeps it)."""
        if timeout is not None:
            self._duration = toast_timeout(self._kind, timeout)
        self._stop_clock()
        self._remaining = float(self._duration or 0)
        self.ring.set_progress(None if self._duration is None else 1.0)
        self._sync_clock()

    def pause(self, reason: str) -> None:
        """Hold the countdown for ``reason`` (hover, application in the background)."""
        self._pauses.add(reason)
        self._sync_clock()

    def resume(self, reason: str) -> None:
        """Release the hold of ``reason``; the countdown goes on once no hold is left."""
        self._pauses.discard(reason)
        self._sync_clock()

    def flash_once(self) -> None:
        """Brighten the pill for a moment (an update in place)."""
        self._flasher.stop()
        self._flasher.setStartValue(1.0)
        self._flasher.setEndValue(0.0)
        self._flasher.start()

    def show_toast(self):
        if not self._placed:
            self.reposition()
        self.raise_()
        self.show()
        self._fade.stop()
        self._fade.setDuration(MOTION_MS if self._motion else FADE_MS)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()
        self._shown = True
        self.restart()

    def reposition(self):
        """Bottom-right of the parent (a ToastManager stacks several instead)."""
        parent = self.parentWidget()
        if parent is None:
            return
        size = self._natural
        self.place(
            QRect(
                parent.width() - size.width() - self._margin,
                parent.height() - size.height() - self._margin,
                size.width(),
                size.height(),
            )
        )

    def place(
        self, rect: QRect, *, tucked: bool = False, animate: bool = False, entering: bool = False
    ) -> None:
        """Move to ``rect``: gliding when ``animate`` (from below when ``entering``), else at
        once (a glide under way is sent to ``rect`` instead). ``tucked`` hides the text."""
        self._placed = True
        self._tucked = tucked
        if tucked:
            self._content.hide()
        gliding = self._glide.state() == QPropertyAnimation.State.Running
        if not self._motion or not (animate or gliding):
            self._glide.stop()
            self.setGeometry(rect)
            self._show_content()
            return
        if gliding and not animate:
            self._glide.setEndValue(rect)
            return
        start = rect.translated(0, rect.height()) if entering else self.geometry()
        self._glide.stop()
        if start == rect:
            self.setGeometry(rect)
            self._show_content()
            return
        self.setGeometry(start)
        self._glide.setDuration(MOTION_MS)
        self._glide.setStartValue(start)
        self._glide.setEndValue(rect)
        self._show_content()  # at its full size: the growing pill uncovers it
        self._glide.start()

    def dismiss(self):
        if self._dismissing:
            return
        self._dismissing = True
        self._sync_clock()
        self.dismissed.emit()
        if self._motion and self._placed:
            box = self.geometry()
            self._glide.stop()
            self._glide.setDuration(FADE_MS)
            self._glide.setStartValue(box)
            self._glide.setEndValue(box.translated(0, round(box.height() * 0.4)))
            self._glide.start()
        self._fade.stop()
        self._fade.setDuration(FADE_MS)
        self._fade.setStartValue(self._opacity.opacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self._finish)
        self._fade.start()

    # ------------------------------------------------------------- internals
    def _get_flash(self) -> float:
        return self._flash

    def _set_flash(self, value: float) -> None:
        self._flash = float(value)
        self.update()

    flash = pyqtProperty(float, _get_flash, _set_flash)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = QRectF(self.rect())
        radius = box.height() / 2 if box.height() <= PILL_MAX_HEIGHT else tokens.RADIUS_LG
        painter.setPen(Qt.PenStyle.NoPen)
        outline = QColor(tokens.BG_0)  # parts the pill from a bright photo
        outline.setAlphaF(0.5)
        painter.setBrush(outline)
        painter.drawRoundedRect(box, radius, radius)
        inner = box.adjusted(1.5, 1.5, -1.5, -1.5)
        inner_radius = max(0.0, radius - 1.5)
        painter.setPen(QPen(QColor(tokens.BORDER_STRONG), 1))
        painter.setBrush(QColor(tokens.BG_3))
        painter.drawRoundedRect(inner, inner_radius, inner_radius)
        if self._flash > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            glow = QColor(tokens.TEXT)
            glow.setAlphaF(0.11 * self._flash)
            painter.setBrush(glow)
            painter.drawRoundedRect(inner, inner_radius, inner_radius)
        painter.end()

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.dismiss()
        super().mousePressEvent(event)

    def enterEvent(self, event):  # noqa: N802 - Qt override
        self.hover_changed.emit(True)
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802 - Qt override
        self.hover_changed.emit(False)
        super().leaveEvent(event)

    def _run_action(self) -> None:
        try:
            self._action.run()
        finally:
            self.dismiss()

    # countdown
    def _sync_clock(self) -> None:
        """Run the countdown while the toast is shown, expires, is not held and not leaving."""
        run = (
            self._shown and self._duration is not None and not self._pauses and not self._dismissing
        )
        if run and not self._running:
            self._clock.start()
            self._ticker.start()
            self._running = True
        elif not run:
            self._stop_clock()

    def _stop_clock(self) -> None:
        if self._running:
            self._remaining -= self._clock.elapsed()
            self._ticker.stop()
            self._running = False

    def _tick(self) -> None:
        left = self._remaining - self._clock.elapsed()
        if left <= 0:
            self.dismiss()
            return
        self.ring.set_progress(left / self._duration)

    # text
    def _show_content(self) -> None:
        if self._tucked:
            return
        self._content.setGeometry(1, 1, self._natural.width() - 2, self._natural.height() - 2)
        self._content.show()

    def _text_room(self) -> int:
        """Width left for the text once the border, margins, ring and buttons fit."""
        self._content.ensurePolished()
        margins = self._row.contentsMargins()
        chrome = 2 + margins.left() + margins.right()
        for widget in (self.ring, self.action_button, self.close_button):
            if widget is not None:
                chrome += _fixed_width(widget) + self._row.spacing()
        return max(60, self._max_width - chrome)

    def _layout_text(self) -> None:
        """Fit the text into ``max_width``. The chrome estimate can miss a few px of
        style-dependent padding, so the measured overshoot is taken off in a second pass."""
        room = self._text_room()
        self._fit_text(room)
        overshoot = self._natural.width() - self._max_width
        if overshoot > 0:
            self._fit_text(max(60, room - overshoot))
        if self._content.isVisibleTo(self):
            self._show_content()
        self.update()

    def _fit_text(self, room: int) -> None:
        title, name = split_message(self._message)
        self._title.ensurePolished()
        self._detail.ensurePolished()
        title_metrics, name_metrics = self._title.fontMetrics(), self._detail.fontMetrics()
        beside = None
        if name is not None:
            left = room - label_advance(title_metrics, title) - TEXT_GAP
            if left >= min(label_advance(name_metrics, name), MIN_NAME_PX):
                beside = left
        self._title.setText(
            elide_long_words(
                title,
                lambda word: title_metrics.horizontalAdvance(word) <= room,
                lambda word: title_metrics.elidedText(word, Qt.TextElideMode.ElideMiddle, room),
            )
        )
        wrap_label(self._title, room)
        if name is None:
            self._detail.hide()
            self._set_direction(QBoxLayout.Direction.LeftToRight)
        else:
            width = room if beside is None else beside
            self._detail.setText(name_metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, width))
            self._detail.show()
            self._set_direction(
                QBoxLayout.Direction.LeftToRight
                if beside is not None
                else QBoxLayout.Direction.TopToBottom
            )
        self._text_box.activate()
        self._row.activate()
        self._content.adjustSize()
        self._natural = self._content.size() + QSize(2, 2)

    def _set_direction(self, direction: QBoxLayout.Direction) -> None:
        if self._text_box.direction() != direction:
            self._text_box.setDirection(direction)
            self._text_box.setSpacing(
                TEXT_GAP if direction == QBoxLayout.Direction.LeftToRight else 1
            )

    def _finish(self):
        self.closed.emit()
        self.deleteLater()


class _HoverPad(QWidget):
    """Covers a fanned-out stack, gaps included, so the pointer can cross a gap between two
    toasts without the stack collapsing."""

    hover_changed = pyqtSignal(bool)

    def enterEvent(self, event):  # noqa: N802 - Qt override
        self.hover_changed.emit(True)
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802 - Qt override
        self.hover_changed.emit(False)
        super().leaveEvent(event)


class ToastManager(QObject):
    """Owns the stack of toasts anchored bottom-right of a host widget.

    Wire it on ``MainWindow`` and route transient feedback through :meth:`show`; hard
    errors that need a decision keep using ``QMessageBox``. ``motion`` (default:
    :func:`motion_default`) makes the toasts slide and glide.
    """

    def __init__(
        self,
        host: QWidget,
        *,
        gap: int = 10,
        margin: int = 16,
        max_visible: int = 3,
        motion: bool | None = None,
    ):
        super().__init__(host)
        self._host = host
        self._gap = gap
        self._margin = margin
        self._max_visible = max(1, max_visible)
        self._motion = motion_default() if motion is None else bool(motion)
        self._toasts: list[Toast] = []
        self._keys: dict[str, Toast] = {}
        self._bottom_offset: Callable[[], int] = lambda: 0
        self._right_offset: Callable[[], int] = lambda: 0
        self._expanded = False
        self._inactive = False
        self._under: set[QWidget] = set()
        self._pad: _HoverPad | None = None
        self._hover_check = QTimer(self)
        self._hover_check.setSingleShot(True)
        self._hover_check.setInterval(HOVER_GRACE_MS)
        self._hover_check.timeout.connect(self._check_hover)
        app = QGuiApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_application_state)

    def show(
        self,
        message: str,
        kind: str = "success",
        timeout: int | None = None,
        *,
        key: str | None = None,
        action: NoticeAction | None = None,
    ) -> Toast:
        """Show ``message``; ``kind`` in success / info / warn / danger (see module doc)."""
        live = self._keys.get(key) if key is not None else None
        if live is not None and live in self._toasts and live.kind() == kind:
            live.set_message(message)
            live.restart(timeout)
            live.flash_once()
            self._toasts.remove(live)
            self._toasts.append(live)  # newest again: the front of the stack
            self._layout(animate=True)
            self._restack()
            return live
        if live is not None:
            live.dismiss()

        toast = Toast(
            self._host,
            message,
            kind=kind,
            timeout=timeout,
            margin=self._margin,
            max_width=self._max_width(max(0, int(self._right_offset()))),
            action=action,
            motion=self._motion,
        )
        toast.dismissed.connect(lambda t=toast: self._remove(t))
        toast.hover_changed.connect(lambda inside, t=toast: self._hover(t, inside))
        self._toasts.append(toast)
        if key is not None:
            self._keys[key] = toast
        if self._expanded:
            toast.pause("hover")
        if self._inactive:
            toast.pause("inactive")
        self._evict()
        self._layout(animate=True, entering=toast)
        toast.show_toast()
        self._restack()
        return toast

    def toasts(self) -> list[Toast]:
        """The live toasts, oldest first."""
        return list(self._toasts)

    def margin(self) -> int:
        """Gap between the stack and the host's edges (before any offset)."""
        return self._margin

    def column_left(self, right_offset: int = 0) -> int:
        """x (host coordinates) of the left edge of the column the widest toast can take,
        with the stack ``right_offset`` px left of its corner."""
        right = max(0, int(right_offset))
        return self._host.width() - self._margin - right - self._max_width(right)

    def stack_height(self) -> int:
        """Height of the live stack fanned out (toasts and the gaps between them): what it
        takes when the pointer is over it, so the chrome it clears does not change then."""
        heights = [toast.natural_size().height() for toast in self._toasts]
        return sum(heights) + self._gap * max(0, len(heights) - 1)

    def set_bottom_offset(self, offset: int | Callable[[], int]) -> None:
        """Keep the stack ``offset`` px (or ``offset()`` px, read on every reposition)
        above the host's bottom edge, e.g. clear of status bars or an editor's dock."""
        self._bottom_offset = offset if callable(offset) else (lambda: int(offset))
        self.reposition()

    def set_right_offset(self, offset: int | Callable[[], int]) -> None:
        """Keep the stack ``offset`` px (or ``offset()`` px, read on every reposition)
        further left than its corner, e.g. left of an editor's side panel."""
        self._right_offset = offset if callable(offset) else (lambda: int(offset))
        self.reposition()

    def reposition(self):
        """Place the stack for the host's current size and offsets (no glide)."""
        self._layout()

    # ------------------------------------------------------------- internals
    def _max_width(self, right_offset: int = 0) -> int:
        return min(MAX_TOAST_WIDTH, self._host.width() - 2 * self._margin - right_offset)

    def _layout(self, *, animate: bool = False, entering: Toast | None = None) -> None:
        """The newest toast in the corner; the older ones above it, fanned out while the
        pointer is over the stack, else tucked behind it."""
        right = max(0, int(self._right_offset()))
        bottom = self._host.height() - self._margin - max(0, int(self._bottom_offset()))
        column_right = self._host.width() - self._margin - right
        newest_first = list(reversed(self._toasts))
        for toast in newest_first:
            toast.set_max_width(self._max_width(right))
        fanned = self._expanded or len(newest_first) < 2
        boxes: list[QRect] = []
        for depth, toast in enumerate(newest_first):
            size = toast.natural_size()
            if fanned:
                top = (boxes[-1].top() - self._gap if boxes else bottom) - size.height()
                box = QRect(column_right - size.width(), top, size.width(), size.height())
            elif depth == 0:
                box = QRect(
                    column_right - size.width(), bottom - size.height(), size.width(), size.height()
                )
            else:
                front = boxes[0]
                width = max(1, round(front.width() * (1 - TUCK_SHRINK * depth)))
                left = front.left() + (front.width() - width) // 2
                box = QRect(left, front.top() - PEEK * depth, width, front.height())
            toast.place(
                box,
                tucked=not fanned and depth > 0,
                animate=animate,
                entering=toast is entering,
            )
            boxes.append(box)
        self._place_pad(boxes if fanned and len(boxes) > 1 else [])

    def _place_pad(self, boxes: list[QRect]) -> None:
        if not boxes:
            if self._pad is not None:
                self._pad.hide()
            return
        if self._pad is None:
            self._pad = _HoverPad(self._host)
            self._pad.hover_changed.connect(lambda inside: self._hover(self._pad, inside))
        area = boxes[0]
        for box in boxes[1:]:
            area = area.united(box)
        self._pad.setGeometry(area)
        if not self._pad.isVisible():
            self._pad.show()
            self._restack()

    def _restack(self) -> None:
        """The pad under the toasts, the newest toast on top."""
        if self._pad is not None and self._pad.isVisible():
            self._pad.raise_()
        for toast in self._toasts:
            toast.raise_()

    def _hover(self, widget: QWidget, inside: bool) -> None:
        if inside:
            self._under.add(widget)
            self._hover_check.stop()
            self._set_expanded(True)
        else:
            self._under.discard(widget)
            self._hover_check.start()

    def _check_hover(self) -> None:
        if not self._under:
            self._set_expanded(False)

    def _set_expanded(self, expanded: bool) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        for toast in self._toasts:
            if expanded:
                toast.pause("hover")
            else:
                toast.resume("hover")
        self._layout(animate=True)
        self._restack()

    def _on_application_state(self, state: Qt.ApplicationState) -> None:
        inactive = state != Qt.ApplicationState.ApplicationActive
        if inactive == self._inactive:
            return
        self._inactive = inactive
        for toast in self._toasts:
            if inactive:
                toast.pause("inactive")
            else:
                toast.resume("inactive")

    def _evict(self) -> None:
        """Keep ``max_visible`` toasts. The newest always stays; of the older ones the
        oldest non-danger toast goes first, else the oldest."""
        while len(self._toasts) > self._max_visible:
            older = self._toasts[:-1]
            victim = next((t for t in older if t.kind() != "danger"), older[0])
            victim.dismiss()

    def _remove(self, toast: Toast):
        if toast in self._toasts:
            self._toasts.remove(toast)
        for key, keyed in list(self._keys.items()):
            if keyed is toast:
                del self._keys[key]
        self._under.discard(toast)
        if not self._under:
            self._hover_check.start()
        self._layout(animate=True)
