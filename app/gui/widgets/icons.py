"""SVG icon loader for the SPROUTS icon set.

Icons live in ``resources/icons/sprouts/<name>.svg`` and are authored with
``stroke="currentColor"``. Qt's :class:`QIcon` does not honour ``currentColor``,
so :func:`load_icon` substitutes the requested colour into the SVG text and
renders it with :class:`QSvgRenderer`. Results are cached per (name, colour, px).
"""

from __future__ import annotations

import os
from functools import lru_cache

from PyQt6.QtCore import QByteArray, Qt, QRectF
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from . import tokens

_ICON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "resources", "icons", "sprouts",
)


def icon_path(name: str) -> str:
    return os.path.join(_ICON_DIR, f"{name}.svg")


def has_icon(name: str) -> bool:
    return os.path.exists(icon_path(name))


@lru_cache(maxsize=512)
def _render(name: str, color: str, px: int) -> QPixmap:
    path = icon_path(name)
    if not os.path.exists(path):
        return QPixmap()
    with open(path, "r", encoding="utf-8") as f:
        svg = f.read()
    svg = svg.replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    renderer.render(painter, QRectF(0, 0, px, px))
    painter.end()
    return pm


def load_pixmap(name: str, color: str = tokens.TEXT_MUTED, px: int = 18) -> QPixmap:
    return _render(name, color, int(px))


def load_icon(name: str, color: str = tokens.TEXT_MUTED, px: int = 18) -> QIcon:
    """Return a recoloured :class:`QIcon` for ``name``."""
    pm = _render(name, color, int(px))
    return QIcon(pm)
