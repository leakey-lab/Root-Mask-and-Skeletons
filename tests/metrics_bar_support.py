"""What a metrics-bar cell shows, for the tests.

The bar has no public accessor for its cells; these helpers are the one place the tests read
its private labels. The day the bar has one (the owner's decision: ``MetricsBar.cell(name)``),
only this module changes. ``name`` is one of ``name``, ``length``, ``area``, ``fov``,
``status``, ``zoom``.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QLabel

from app.gui.metrics_bar import MetricsBar


def cell(bar: MetricsBar, name: str) -> QFrame:
    """The cell ``name`` (its tooltip, its visibility, its separator)."""
    return getattr(bar, f"_{name}_cell")


def value(bar: MetricsBar, name: str) -> QLabel:
    """The value label of the cell ``name``."""
    return getattr(bar, f"_{name}_value")


def shown(bar: MetricsBar, name: str) -> bool:
    """True when the cell ``name`` is on the bar (a cell with nothing to show is hidden)."""
    return not cell(bar, name).isHidden()


def text(bar: MetricsBar, name: str) -> str:
    """The full text of the cell ``name`` (an elided label keeps its whole text)."""
    return value(bar, name).text()


def tooltip(bar: MetricsBar, name: str) -> str:
    """The tooltip of the cell ``name``."""
    return cell(bar, name).toolTip()


def stale(bar: MetricsBar, name: str):
    """The value's ``stale`` property: True while it is shown outdated (warning colour)."""
    return value(bar, name).property("stale")


def style(bar: MetricsBar, name: str) -> str:
    """The value label's style sheet (its colour)."""
    return value(bar, name).styleSheet()
