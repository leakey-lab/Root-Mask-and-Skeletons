"""SPROUTS guided-shell chrome builders (presentation-only).

Each ``build_*`` function returns a styled QWidget band. They reparent / wire to
the EXISTING ``MainWindow`` handlers and attrs — no computed data, signals, or
QStackedWidget contracts change here. The ribbon stage buttons forward to the
same toggle/switch handlers the left panel already uses.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.gui.widgets import SegmentedControl, tokens
from app.gui.widgets.icons import load_icon, load_pixmap


def _band(object_name: str, *, height: int | None = None) -> QFrame:
    band = QFrame()
    band.setObjectName(object_name)
    band.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    band.setStyleSheet(f"""
        QFrame#{object_name} {{
            background-color: {tokens.BG_1};
            border: none;
            border-bottom: 1px solid {tokens.BORDER};
        }}
    """)
    if height is not None:
        band.setFixedHeight(height)
    return band


# --------------------------------------------------------------------------- #
def build_titlebar(mw) -> QWidget:
    bar = _band("shellTitlebar", height=48)
    row = QHBoxLayout(bar)
    row.setContentsMargins(16, 0, 16, 0)
    row.setSpacing(10)

    logo = QLabel()
    logo.setPixmap(load_pixmap("sprouts_logo", tokens.ACCENT, 22))
    row.addWidget(logo)

    wordmark = QLabel(
        f'<span style="color:{tokens.TEXT}">SPR</span>'
        f'<span style="color:{tokens.ACCENT}">OU</span>'
        f'<span style="color:{tokens.TEXT}">TS</span>'
    )
    wordmark.setTextFormat(Qt.TextFormat.RichText)
    wordmark.setStyleSheet("font-size: 15px; font-weight: 800; letter-spacing: 1px;")
    row.addWidget(wordmark)

    row.addStretch(1)

    load_btn = QPushButton("Load Images")
    load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    load_btn.setIcon(load_icon("load", tokens.ACCENT_INK, 15))
    load_btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {tokens.ACCENT};
            color: {tokens.ACCENT_INK};
            border: none;
            border-radius: 7px;
            padding: 6px 16px;
            font-weight: 600;
            font-size: 12.5px;
        }}
        QPushButton:hover {{ background-color: {tokens.ACCENT_PRESS}; }}
    """)
    load_btn.clicked.connect(mw.load_images)
    mw.titlebar_load_button = load_btn
    row.addWidget(load_btn)

    return bar


# --------------------------------------------------------------------------- #
_RIBBON_STAGES = [
    ("Library", "image"),
    ("Mask", "cpu"),
    ("Trace", "brush"),
    ("Skeleton", "skeleton"),
    ("Measure", "ruler"),
    ("Visualize", "chart"),
]


def build_ribbon(mw) -> QWidget:
    bar = _band("shellRibbon", height=52)
    row = QHBoxLayout(bar)
    row.setContentsMargins(16, 8, 16, 8)
    row.setSpacing(6)

    group = QButtonGroup(bar)
    group.setExclusive(True)
    mw.ribbon = bar
    mw.ribbon_buttons = {}

    handlers = {
        "Library": lambda: mw.switch_right_panel("display"),
        "Mask": lambda: mw.switch_right_panel("display"),
        "Trace": mw.toggle_mask_tracing,
        "Skeleton": mw.toggle_skeleton_correction,
        "Measure": lambda: mw.switch_right_panel("display"),
        "Visualize": lambda: mw.switch_right_panel("display"),
    }

    for label, icon in _RIBBON_STAGES:
        btn = QToolButton(bar)
        btn.setText(label)
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setIcon(load_icon(icon, tokens.TEXT_MUTED, 16))
        btn.setIconSize(QSize(16, 16))
        btn.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                color: {tokens.TEXT_MUTED};
                padding: 5px 12px;
                font-size: 12.5px;
                font-weight: 500;
            }}
            QToolButton:hover {{ background-color: {tokens.BG_3}; color: {tokens.TEXT}; }}
            QToolButton:checked {{
                background-color: {tokens.ACCENT_SOFT};
                color: {tokens.TEXT};
                border: 1px solid {tokens.ACCENT_LINE};
            }}
        """)
        btn.clicked.connect(lambda _=False, h=handlers[label]: h())
        group.addButton(btn)
        row.addWidget(btn)
        mw.ribbon_buttons[label] = btn

    # Default selection: Library.
    mw.ribbon_buttons["Library"].setChecked(True)
    row.addStretch(1)
    return bar


# --------------------------------------------------------------------------- #
def build_action_bar(mw) -> QWidget:
    bar = _band("shellActionBar", height=52)
    row = QHBoxLayout(bar)
    row.setContentsMargins(16, 8, 16, 8)
    row.setSpacing(10)

    hint = QLabel("Select a stage above, then act on the loaded images.")
    hint.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 12px;")
    mw.action_bar_hint = hint
    row.addWidget(hint)

    row.addStretch(1)

    # Visualize stage: Length/Area segmented control forwarding to the same
    # visualization toggle handlers the left panel uses.
    viz_seg = SegmentedControl(
        [("length", "Length"), ("area", "Area")], value="length"
    )

    def _on_viz(value: str):
        if value == "length":
            mw.toggle_root_length_visualization()
        elif value == "area":
            mw.toggle_root_area_visualization()

    viz_seg.valueChanged.connect(_on_viz)
    mw.action_bar_viz_seg = viz_seg
    row.addWidget(viz_seg)

    return bar


# --------------------------------------------------------------------------- #
def build_statusline(mw) -> QWidget:
    band = QFrame()
    band.setObjectName("shellStatusline")
    band.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    band.setFixedHeight(30)
    band.setStyleSheet(f"""
        QFrame#shellStatusline {{
            background-color: {tokens.BG_1};
            border: none;
            border-top: 1px solid {tokens.BORDER};
        }}
        QLabel {{ background: transparent; }}
    """)
    row = QHBoxLayout(band)
    row.setContentsMargins(14, 0, 14, 0)
    row.setSpacing(8)

    dot = QLabel()
    dot.setPixmap(load_pixmap("info", tokens.OK, 8))
    dot.setFixedWidth(10)
    row.addWidget(dot)

    msg = QLabel("Ready")
    msg.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 11.5px;")
    mw.statusline_message = msg
    row.addWidget(msg)

    row.addStretch(1)

    info = QLabel("GPU · CUDA 12.8")
    info.setStyleSheet(
        f"color: {tokens.TEXT_FAINT}; font-family: {tokens.MONO}; font-size: 11px;"
    )
    mw.statusline_info = info
    row.addWidget(info)

    return band
