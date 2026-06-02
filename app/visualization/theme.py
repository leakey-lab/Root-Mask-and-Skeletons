"""
SPROUTS visualization theme — a single source of truth for the Plotly look.

Drop-in module. Importing it registers two Plotly templates:

    "sprouts"        — dark, matches the SPROUTS desktop shell (default)
    "sprouts_light"  — light variant (same geometry, light surfaces)

and exposes small helpers so chart code stops re-declaring 40-line layout
dicts in every method.

Usage
-----
    from app.visualization import theme           # registers templates on import
    theme.use("sprouts")                           # make it the global default

    fig = go.Figure(...)
    theme.style(fig, title="Root Length by Tube")  # apply title + sane margins

    # build hover text without emoji / fragile inline <span> colours:
    hovertemplate = theme.hover("Tube %{x}", [
        ("Date",   "%{customdata[0]}"),
        ("Length", "%{y:.2f} mm"),
    ])

Design tokens mirror the SPROUTS CSS custom properties (styles.css :root) so the
embedded dashboard and the desktop chrome read as one product.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Design tokens — keep in sync with styles.css :root
# ---------------------------------------------------------------------------

# Dark surfaces (deep, desaturated)
BG_0 = "#15161c"   # deepest — app behind / paper
BG_1 = "#1b1c24"   # chrome
BG_2 = "#21232e"   # panels / plot area
BG_3 = "#282a37"   # raised cards / inputs

BORDER = "#2a2c39"
BORDER_STRONG = "#373a4c"

TEXT = "#eceef5"
TEXT_MUTED = "#9498ad"
TEXT_FAINT = "#686c82"

# Brand accents
ACCENT = "#5fd6a0"   # SPROUTS leaf green (primary)
SEL = "#b794f6"      # purple — reserved for selection / highlight
WARN = "#e8b25e"
INFO = "#79c0e8"
DANGER = "#ec6a78"
SKEL = "#f0a868"     # warm — skeleton

# Fonts (loaded via assets/, falls back to system)
FONT_SANS = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"
FONT_MONO = "IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace"

# Categorical palette for multi-series (tubes / dates). Ten hues, all chosen to
# stay legible on a dark plot ground and to stay distinct from each other.
# Replaces px.colors.qualitative.Plotly (the default rainbow).
COLORWAY = [
    "#5fd6a0",  # green   (accent)
    "#79c0e8",  # blue
    "#b794f6",  # purple
    "#e8b25e",  # amber
    "#f0a868",  # orange
    "#ec6a78",  # red
    "#4fd1c5",  # teal
    "#f6a5c0",  # pink
    "#c3e88d",  # lime
    "#9aa5ff",  # periwinkle
]

# Sequential scale for ordered data (depth profiles). Plasma reads well on dark
# and is perceptually uniform — keep it as the ordered default.
SEQUENTIAL = "Plasma"

# Light-variant surface overrides (geometry is shared with the dark template).
_LIGHT = {
    "paper": "#ffffff",
    "plot": "#ffffff",
    "text": "#1f2430",
    "muted": "#5b6172",
    "grid": "#e7e9f0",
    "zero": "#c7cad6",
    "line": "#aeb2c2",
    "hover_bg": "#ffffff",
    "hover_border": "#c7cad6",
    "legend_bg": "rgba(255,255,255,0.92)",
}


# ---------------------------------------------------------------------------
# Template construction
# ---------------------------------------------------------------------------

def _axis(grid: str, zero: str, line: str, tick: str) -> dict:
    return dict(
        showgrid=True,
        gridcolor=grid,
        gridwidth=1,
        zeroline=True,
        zerolinecolor=zero,
        zerolinewidth=1,
        showline=True,
        linecolor=line,
        linewidth=1,
        ticks="outside",
        ticklen=5,
        tickcolor=line,
        tickfont=dict(size=13, color=tick),
        title=dict(font=dict(size=15, color=tick)),
        automargin=True,
    )


def _build(dark: bool) -> go.layout.Template:
    if dark:
        paper, plot = BG_0, BG_2
        text, muted = TEXT, TEXT_MUTED
        grid, zero, line = "rgba(255,255,255,0.06)", BORDER_STRONG, BORDER
        hover_bg, hover_border = BG_3, BORDER_STRONG
        legend_bg = "rgba(33,35,46,0.92)"
    else:
        paper, plot = _LIGHT["paper"], _LIGHT["plot"]
        text, muted = _LIGHT["text"], _LIGHT["muted"]
        grid, zero, line = _LIGHT["grid"], _LIGHT["zero"], _LIGHT["line"]
        hover_bg, hover_border = _LIGHT["hover_bg"], _LIGHT["hover_border"]
        legend_bg = _LIGHT["legend_bg"]

    template = go.layout.Template()

    template.layout = go.Layout(
        font=dict(family=FONT_SANS, size=14, color=text),
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        colorway=COLORWAY,
        title=dict(
            font=dict(family=FONT_SANS, size=20, color=text),
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
        ),
        margin=dict(l=64, r=180, t=64, b=48),
        xaxis=_axis(grid, zero, line, muted),
        yaxis=_axis(grid, zero, line, muted),
        legend=dict(
            font=dict(size=13, color=muted),
            bgcolor=legend_bg,
            bordercolor=line,
            borderwidth=1,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor=hover_bg,
            bordercolor=hover_border,
            font=dict(family=FONT_SANS, size=13, color=text),
            align="left",
        ),
        colorscale=dict(sequential=SEQUENTIAL),
        bargap=0.2,
        bargroupgap=0.08,
        # subtle transition so callback-driven figure swaps animate
        transition=dict(duration=250, easing="cubic-in-out"),
    )

    # Per-trace defaults
    template.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    template.data.scatter = [
        go.Scatter(line=dict(width=2), marker=dict(size=7, line=dict(width=0)))
    ]

    return template


# Register on import so a bare `import theme` is enough.
pio.templates["sprouts"] = _build(dark=True)
pio.templates["sprouts_light"] = _build(dark=False)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def use(name: str = "sprouts") -> None:
    """Make a SPROUTS template the global Plotly default.

    Call once at app startup (after `import theme`). Every `go.Figure()` created
    afterwards inherits the theme, so individual charts only set title + data.
    """
    pio.templates.default = name


def style(fig: go.Figure, *, title: str | None = None, height: int | None = None,
          template: str = "sprouts", **layout_kwargs) -> go.Figure:
    """Apply the theme to a figure in one call.

    Use for figures where you want to be explicit instead of relying on the
    global default (e.g. figures built before `use()` runs, or subplots from
    make_subplots which do not inherit the default template cleanly).
    """
    fig.update_layout(template=template, **layout_kwargs)
    if title is not None:
        fig.update_layout(title_text=title)
    if height is not None:
        fig.update_layout(height=height)
    return fig


def hover(title: str, rows: list[tuple[str, str]], *, with_extra: bool = True) -> str:
    """Build a clean, consistent hovertemplate — no emoji, no inline colours.

    Colour, font and border come from the template's `hoverlabel`, so the hover
    box matches every other chart automatically.

        hover("Tube %{x}", [("Date", "%{customdata[0]}"), ("Length", "%{y:.2f} mm")])

    Args:
        title: bold first line (may contain Plotly format refs like %{x}).
        rows:  (label, value) pairs; value may contain %{...} format refs.
        with_extra: append "<extra></extra>" to suppress the secondary box.
    """
    lines = [f"<b>{title}</b>"]
    lines += [f"{label}\u2002<b>{value}</b>" for label, value in rows]
    out = "<br>".join(lines)
    return out + "<extra></extra>" if with_extra else out


def color(index: int) -> str:
    """Categorical colour by series index (wraps the colorway)."""
    return COLORWAY[index % len(COLORWAY)]
