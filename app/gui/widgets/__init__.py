"""SPROUTS design-system widgets — reusable, presentation-only primitives.

Built for the Guided + comfy + purple-accent direction. All widgets wrap
standard Qt and expose plain signals so callers forward to existing handlers
without behaviour changes. See ``tokens`` for the colour/density values.
"""

from __future__ import annotations

from . import tokens
from .icons import load_icon, load_pixmap, has_icon, icon_path
from .effects import drop_shadow, pop_shadow
from .controls import SegmentedControl, IconButton
from .overlays import (
    ToolRail, FloatingDock, EnhancePopover, Toast, ToastManager, ProgressOverlay,
)

__all__ = [
    "tokens",
    "load_icon", "load_pixmap", "has_icon", "icon_path",
    "drop_shadow", "pop_shadow",
    "SegmentedControl", "IconButton",
    "ToolRail", "FloatingDock", "EnhancePopover", "Toast", "ToastManager",
    "ProgressOverlay",
]
