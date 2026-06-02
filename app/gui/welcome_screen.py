"""SPROUTS welcome / get-started screen.

Presentation-only landing page shown at ``app_stack`` index 0. The single entry
point is ``on_get_started`` (a 0-arg callable, wired by ``MainWindow`` to
``load_images``); the dropzone and the Browse button both invoke it. The actual
directory dialog lives in ``load_images`` itself, so this screen never opens a
dialog directly.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.gui.widgets import tokens
from app.gui.widgets.icons import load_pixmap, load_icon

_STAGES = ["Library", "Mask", "Trace", "Skeleton", "Measure", "Visualize"]


class _Dropzone(QFrame):
    """Dashed drag-and-drop target that fires ``on_drop`` when files are dropped."""

    def __init__(self, on_drop: Callable[[], None], parent=None):
        super().__init__(parent)
        self._on_drop = on_drop
        self.setObjectName("dropzone")
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(150)
        self.setStyleSheet(f"""
            QFrame#dropzone {{
                background-color: {tokens.BG_2};
                border: 1.5px dashed {tokens.BORDER_STRONG};
                border-radius: {tokens.RADIUS_LG}px;
            }}
        """)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            if callable(self._on_drop):
                self._on_drop()


class WelcomeWidget(QWidget):
    """Get-started landing page. ``on_get_started`` is a 0-arg callable."""

    def __init__(self, on_get_started: Callable[[], None],
                 on_open_recent: Optional[Callable[[str], None]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._on_get_started = on_get_started
        self._on_open_recent = on_open_recent
        self.setObjectName("welcome")
        self.setStyleSheet(f"QWidget#welcome {{ background-color: {tokens.BG_0}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 40, 40, 28)
        outer.setSpacing(14)
        outer.addStretch(1)

        # --- Leaf logo ---
        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setPixmap(load_pixmap("sprouts_logo", tokens.ACCENT, 58))
        outer.addWidget(logo, 0, Qt.AlignmentFlag.AlignHCenter)

        # --- Title (style the "OU" in accent) ---
        title = QLabel(
            f'<span style="color:{tokens.TEXT}">SPR</span>'
            f'<span style="color:{tokens.ACCENT}">OU</span>'
            f'<span style="color:{tokens.TEXT}">TS</span>'
        )
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: 800; letter-spacing: 2px;")
        outer.addWidget(title)

        # --- Tagline ---
        tagline = QLabel("Plant-root phenotyping from minirhizotron images")
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tagline.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 13px;")
        outer.addWidget(tagline)

        # --- Dropzone ---
        self.dropzone = _Dropzone(self._handle_get_started)
        dz_box = QVBoxLayout(self.dropzone)
        dz_box.setContentsMargins(20, 20, 20, 20)
        dz_box.setSpacing(12)
        dz_box.addStretch(1)

        dz_hint = QLabel("Drop a folder of root images here")
        dz_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dz_hint.setStyleSheet(f"color: {tokens.TEXT_MUTED}; font-size: 12.5px; border: none; background: transparent;")
        dz_box.addWidget(dz_hint)

        self.browse_button = QPushButton("Browse…")
        self.browse_button.setObjectName("primaryButton")
        self.browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_button.setStyleSheet(f"""
            QPushButton#primaryButton {{
                background-color: {tokens.ACCENT};
                color: {tokens.ACCENT_INK};
                border: none;
                border-radius: 8px;
                padding: 9px 26px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton#primaryButton:hover {{ background-color: {tokens.ACCENT_PRESS}; }}
        """)
        self.browse_button.clicked.connect(self._handle_get_started)
        browse_row = QHBoxLayout()
        browse_row.addStretch(1)
        browse_row.addWidget(self.browse_button)
        browse_row.addStretch(1)
        dz_box.addLayout(browse_row)
        dz_box.addStretch(1)

        outer.addWidget(self.dropzone)

        # --- Recent projects (hidden when empty) ---
        self.recent_section = self._build_recent_section()
        if self.recent_section is not None:
            outer.addWidget(self.recent_section)

        # --- Stage chips ---
        outer.addWidget(self._build_stage_chips(), 0, Qt.AlignmentFlag.AlignHCenter)

        outer.addStretch(1)

        # --- Footer ---
        footer = QLabel(
            "GPU · CUDA 12.8 · resnet_mask_v5 · pix2pix_skeletonizer · v2.0"
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet(
            f"color: {tokens.TEXT_FAINT}; font-family: {tokens.MONO}; font-size: 11px;"
        )
        outer.addWidget(footer)

    # --------------------------------------------------------------------- #
    def _load_recent_entries(self) -> List[dict]:
        """Read the recent_projects MRU list from QSettings (JSON list of dicts)."""
        import json

        settings = QSettings("LeakeyLab", "SPROUTS")
        raw = settings.value("recent_projects", "[]")
        try:
            entries = json.loads(raw) if isinstance(raw, str) else list(raw)
        except (ValueError, TypeError):
            entries = []
        if not isinstance(entries, list):
            return []
        return [e for e in entries if isinstance(e, dict) and e.get("path")]

    @staticmethod
    def _relative_date(ts: str) -> str:
        """ISO timestamp -> short relative-ish label (best effort)."""
        from datetime import datetime

        try:
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return ""
        delta = datetime.now() - dt
        secs = delta.total_seconds()
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{int(secs // 60)}m ago"
        if secs < 86400:
            return f"{int(secs // 3600)}h ago"
        if secs < 7 * 86400:
            return f"{int(secs // 86400)}d ago"
        return dt.strftime("%Y-%m-%d")

    def _build_recent_section(self) -> Optional[QWidget]:
        entries = self._load_recent_entries()
        if not entries:
            return None

        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        heading = QLabel("Recent projects")
        heading.setStyleSheet(f"color: {tokens.TEXT_FAINT}; font-size: 11px;")
        col.addWidget(heading)
        for entry in entries[:8]:
            col.addWidget(self._build_recent_row(entry))
        return wrap

    def _build_recent_row(self, entry: dict) -> QWidget:
        path = str(entry.get("path", ""))
        name = str(entry.get("name") or path)
        count = entry.get("count")
        ts = entry.get("ts", "")

        row_btn = QPushButton()
        row_btn.setObjectName("recentRow")
        row_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        row_btn.setIcon(load_icon("image", tokens.TEXT_MUTED, 15))
        row_btn.setStyleSheet(f"""
            QPushButton#recentRow {{
                background-color: {tokens.BG_2};
                border: 1px solid {tokens.BORDER};
                border-radius: {tokens.RADIUS_SM}px;
                padding: 8px 12px;
                text-align: left;
            }}
            QPushButton#recentRow:hover {{
                background-color: {tokens.BG_3};
                border: 1px solid {tokens.BORDER_STRONG};
            }}
        """)

        inner = QHBoxLayout(row_btn)
        inner.setContentsMargins(8, 0, 8, 0)
        inner.setSpacing(10)
        inner.addSpacing(18)  # leave room for the button icon

        meta = QVBoxLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        meta.setSpacing(1)
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(
            f"color: {tokens.TEXT}; font-size: 12.5px; font-weight: 600; "
            f"background: transparent; border: none;"
        )
        path_lbl = QLabel(path)
        path_lbl.setStyleSheet(
            f"color: {tokens.TEXT_FAINT}; font-family: {tokens.MONO}; "
            f"font-size: 10.5px; background: transparent; border: none;"
        )
        meta.addWidget(name_lbl)
        meta.addWidget(path_lbl)
        inner.addLayout(meta, 1)

        bits = []
        if count is not None:
            bits.append(f"{count} images")
        rel = self._relative_date(ts)
        if rel:
            bits.append(rel)
        if bits:
            info = QLabel("  ·  ".join(bits))
            info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            info.setStyleSheet(
                f"color: {tokens.TEXT_MUTED}; font-size: 11px; "
                f"background: transparent; border: none;"
            )
            inner.addWidget(info)

        row_btn.clicked.connect(lambda _=False, p=path: self._handle_open_recent(p))
        return row_btn

    def _handle_open_recent(self, path: str) -> None:
        if path and callable(self._on_open_recent):
            self._on_open_recent(path)

    def _build_stage_chips(self) -> QWidget:
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        for stage in _STAGES:
            chip = QLabel(stage)
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setStyleSheet(f"""
                QLabel {{
                    background-color: {tokens.BG_2};
                    color: {tokens.TEXT_MUTED};
                    border: 1px solid {tokens.BORDER};
                    border-radius: 12px;
                    padding: 4px 12px;
                    font-size: 11.5px;
                }}
            """)
            row.addWidget(chip)
        return wrap

    def _handle_get_started(self, *_args) -> None:
        if callable(self._on_get_started):
            self._on_get_started()
