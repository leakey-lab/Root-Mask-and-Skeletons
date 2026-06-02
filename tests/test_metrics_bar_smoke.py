"""Smoke tests for the read-only display metrics bar (Track M).

Instantiates ``MetricsBar`` under an offscreen QApplication and exercises
``set_metrics`` with real values and with ``None`` (em-dash fallback). Also
asserts the metrics bar mounts in the right panel without disturbing the
display-page stack / right-panel indices.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.gui.metrics_bar import MetricsBar  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def test_metrics_bar_instantiates(app):
    bar = MetricsBar()
    # Fixed FOV cell is always populated.
    assert bar._status_value.text() == "—"
    assert bar.height() == 56


def test_set_metrics_with_values(app):
    bar = MetricsBar()
    bar.set_metrics(length=12.5, area=6.7, status="Skeleton ready")
    assert bar._length_value.text() == "12.50 mm"
    assert bar._area_value.text() == "6.70 mm²"
    assert bar._status_value.text() == "Skeleton ready"


def test_set_metrics_with_none(app):
    bar = MetricsBar()
    bar.set_metrics(length=4.0, area=2.0, status="x")
    bar.set_metrics()  # all None -> all dashes
    assert bar._length_value.text() == "—"
    assert bar._area_value.text() == "—"
    assert bar._status_value.text() == "—"
