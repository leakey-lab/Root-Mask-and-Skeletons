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
    QStackedWidget,
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

    return bar


# --------------------------------------------------------------------------- #
_RIBBON_STAGES = [
    ("Library", "image"),
    ("Generate Mask", "cpu"),
    ("Trace", "brush"),
    ("Generate Skeleton", "skeleton"),
    ("Correct", "node"),
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

    base_handlers = {
        "Library": lambda: mw.switch_right_panel("display"),
        "Generate Mask": lambda: mw.switch_right_panel("display"),
        "Trace": mw.toggle_mask_tracing,
        "Generate Skeleton": lambda: mw.switch_right_panel("display"),
        "Correct": mw.toggle_skeleton_correction,
        "Measure": lambda: mw.switch_right_panel("display"),
        "Visualize": lambda: mw.switch_right_panel("display"),
    }

    def _make_handler(stage_label, base):
        def _h():
            base()
            # Swap the action-bar page + hint for this stage (FIX4). Safe no-op
            # until the action bar is populated.
            activate = getattr(mw, "_activate_action_stage", None)
            if callable(activate):
                activate(stage_label)
        return _h

    handlers = {
        label: _make_handler(label, base) for label, base in base_handlers.items()
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
                background-color: {tokens.BG_2};
                border: 1px solid {tokens.BORDER};
                border-radius: 9px;
                color: {tokens.TEXT_MUTED};
                padding: 5px 12px;
                font-size: 12.5px;
                font-weight: 500;
            }}
            QToolButton:hover {{
                background-color: {tokens.BG_3};
                color: {tokens.TEXT};
                border: 1px solid {tokens.BORDER_STRONG};
            }}
            QToolButton:checked {{
                background-color: {tokens.rgba(tokens.ACCENT, 0.14)};
                color: {tokens.ACCENT};
                border: 1px solid {tokens.rgba(tokens.ACCENT, 0.30)};
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
# Per-stage hint text shown at the left of the action bar.
_STAGE_HINTS = {
    "Library": "<b>Library.</b> Browse and select images from the tree.",
    "Generate Mask": "<b>Generate Mask.</b> Segment roots from images with the ML model.",
    "Trace": "<b>Trace.</b> Manually paint or refine the root mask.",
    "Generate Skeleton": "<b>Generate Skeleton.</b> Extract the medial-axis skeleton from masks.",
    "Correct": "<b>Correct.</b> Manually edit and repair the root skeleton.",
    "Measure": "<b>Measure.</b> Compute root length and area.",
    "Visualize": "<b>Visualize.</b> Explore length / area dashboards.",
}


def build_action_bar(mw) -> QWidget:
    bar = _band("shellActionBar", height=52)
    row = QHBoxLayout(bar)
    row.setContentsMargins(16, 8, 16, 8)
    row.setSpacing(10)

    hint = QLabel(_STAGE_HINTS["Library"])
    hint.setTextFormat(Qt.TextFormat.RichText)
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

    # Stage-aware action widgets live in a QStackedWidget on the right of the
    # action bar. Pages are built once the real buttons exist (after the body is
    # constructed); build_action_bar only erects the skeleton + the populate
    # callable, which _build_shell invokes post-body.
    stack = QStackedWidget()
    mw.action_bar_stack = stack
    row.addWidget(stack)

    def _page(*widgets) -> QWidget:
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        for w in widgets:
            if w is not None:
                w.setParent(page)
                lay.addWidget(w)
        return page

    def _populate():
        # Display-mode cosmetic segmented control (forwards into the hidden live
        # view_mode_combo; reverse-synced to avoid drift).
        view_seg = SegmentedControl(
            [("single", "Single"), ("overlay", "Overlay"), ("split", "Side by side")],
            value="single",
        )
        mw.action_bar_view_seg = view_seg
        _seg_index = {"single": 0, "overlay": 1, "split": 2}
        _index_seg = {0: "single", 1: "overlay", 2: "split"}

        def _on_view(value: str):
            idx = _seg_index.get(value)
            if idx is not None and mw.view_mode_combo.currentIndex() != idx:
                mw.view_mode_combo.setCurrentIndex(idx)

        view_seg.valueChanged.connect(_on_view)

        def _on_combo(idx: int):
            val = _index_seg.get(idx)
            if val is not None and view_seg.value() != val:
                view_seg.setValue(val, emit=False)

        mw.view_mode_combo.currentIndexChanged.connect(_on_combo)

        # Trace stage: reuse the mask-tracing clear button if exposed.
        trace_clear = getattr(
            getattr(mw, "mask_tracing_interface", None), "clear_button", None
        )

        pages = {
            "Library": _page(view_seg),
            "Generate Mask": _page(mw.generate_mask_button),
            "Trace": _page(trace_clear),
            "Generate Skeleton": _page(mw.generate_button),
            "Correct": _page(),
            "Measure": _page(
                mw.calculate_length_button, mw.calculate_area_button
            ),
            "Visualize": _page(viz_seg),
        }
        mw._action_bar_pages = {}
        for label, page in pages.items():
            mw._action_bar_pages[label] = stack.addWidget(page)

    def _activate_action_stage(label: str):
        hint.setText(_STAGE_HINTS.get(label, ""))
        idx = getattr(mw, "_action_bar_pages", {}).get(label)
        if idx is not None:
            stack.setCurrentIndex(idx)

    mw._populate_action_bar = _populate
    mw._activate_action_stage = _activate_action_stage

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
