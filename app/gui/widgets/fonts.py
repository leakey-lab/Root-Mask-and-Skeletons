"""The UI fonts: IBM Plex Sans and IBM Plex Mono, shipped as TTF files (owner item #46).

The application stylesheet and the widget stylesheets name "IBM Plex Sans" and "IBM Plex
Mono" first (``tokens.SANS``, ``tokens.MONO``). Qt cannot read the ``.woff2`` copies in
``app/assets/fonts/`` (the dashboards' fonts), so until these TTFs shipped each OS drew its
own fallback font, with its own text widths. ``main.apply_stylesheet`` registers them
before it loads the stylesheet.

Only the weights the UI uses ship, unmodified from IBM's release (SIL OFL 1.1, the licence
beside them; versions and SHA-256 in ``app/assets/THIRD_PARTY.md``): Sans 400, 500, 600 and
700, Mono 400 and 600. A heavier weight (the 800 of two titles) draws in Bold, Plex's
heaviest here; Mono 500 draws in Regular.
"""

from __future__ import annotations

import logging
import os

from PyQt6.QtGui import QFontDatabase

from resources.resource_utils import resource_path

logger = logging.getLogger(__name__)

#: The font files in ``resources/fonts/``.
FONT_FILES: tuple[str, ...] = (
    "IBMPlexSans-Regular.ttf",
    "IBMPlexSans-Medium.ttf",
    "IBMPlexSans-SemiBold.ttf",
    "IBMPlexSans-Bold.ttf",
    "IBMPlexMono-Regular.ttf",
    "IBMPlexMono-SemiBold.ttf",
)
#: The families the stylesheets name first.
FAMILIES: tuple[str, ...] = ("IBM Plex Sans", "IBM Plex Mono")

_font_ids: dict[str, int] = {}  # font file -> its id in Qt's application fonts


def font_dir() -> str:
    """The folder of the shipped fonts (inside the bundle when frozen)."""
    return resource_path("resources", "fonts")


def register_fonts() -> tuple[str, ...]:
    """Add the shipped fonts to Qt's font database; a QGuiApplication must exist.

    Returns every family name Qt read from them, sorted (on Windows also the legacy names,
    e.g. "IBM Plex Sans Medm"). A file that is missing or that Qt rejects is logged and
    skipped: the stylesheet's next family is used instead. Calling it again adds nothing.
    """
    families: set[str] = set()
    for name in FONT_FILES:
        font_id = _font_ids.get(name, -1)
        found = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        if not found:  # not added yet (or the application that had it is gone)
            path = os.path.join(font_dir(), name)
            font_id = QFontDatabase.addApplicationFont(path)
            found = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
            if not found:
                logger.warning("UI font not loaded, the system font is used instead: %s", path)
                continue
            _font_ids[name] = font_id
        families.update(found)
    return tuple(sorted(families))
