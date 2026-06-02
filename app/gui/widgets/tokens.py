"""SPROUTS design tokens for use from Python code.

Mirrors ``resources/themes/dark_theme.qss`` and the design ``styles.css :root``.
Direction: Guided + comfy + purple accent. Mask stays green, skeleton orange;
purple is accent + selection only.
"""

from __future__ import annotations

# --- surfaces ---
BG_0 = "#15161c"          # deepest — canvas / app behind
BG_1 = "#1b1c24"          # chrome / rails
BG_2 = "#21232e"          # panels
BG_3 = "#282a37"          # raised cards / inputs
BG_HOVER = "#2f3242"
BG_ACTIVE = "#363a4d"

BORDER = "#2a2c39"
BORDER_STRONG = "#373a4c"

# --- text ---
TEXT = "#eceef5"
TEXT_MUTED = "#9498ad"
TEXT_FAINT = "#686c82"

# --- accent (purple, chosen direction) ---
ACCENT = "#c39af6"
ACCENT_PRESS = "#ab87d8"
ACCENT_INK = "#1a1322"            # dark text on accent fills
ACCENT_SOFT = "rgba(195, 154, 246, 0.14)"
ACCENT_LINE = "rgba(195, 154, 246, 0.30)"

# --- selection (purple) ---
SEL = "#b794f6"
SEL_SOFT = "rgba(183, 148, 246, 0.16)"

# --- semantic / status ---
WARN = "#e8b25e"
INFO = "#79c0e8"
DANGER = "#ec6a78"
MASK = "#5fd6a0"          # mask overlay = green
SKEL = "#f0a868"          # skeleton = warm orange
OK = "#5fd6a0"

# --- radii ---
RADIUS = 10
RADIUS_SM = 7
RADIUS_LG = 14

# --- comfy density ---
ROW_H = 40
PAD = 18
UI = 14.5

# --- fonts ---
SANS = '"IBM Plex Sans", "Segoe UI", sans-serif'
MONO = '"IBM Plex Mono", "Consolas", monospace'


def rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` + alpha -> ``rgba(r, g, b, a)`` string for QSS."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"
