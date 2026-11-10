"""One main class for both dashboards: the root-area dashboard is the root-length one.

:class:`~app.visualization.dash_metric_app.MetricDashApp` builds the page, the callbacks
and the figures of a metric dashboard; ``DashApp`` (length) and ``DashAppArea`` (area)
only name their shell, page title and export file prefix. The area dashboard therefore
has every view, control and toggle of the length dashboard, charting area.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image

from app.visualization.dash_app import DashApp
from app.visualization.dash_app_area import AREA_SHELL, DashAppArea
from app.visualization.dash_metric_app import MetricDashApp
from app.visualization.dash_metric_layout import LENGTH_SHELL
from tests.viz_dash_helpers import (
    AREA_COLUMN,
    FakeImageManager,
    build_area_app,
    build_length_app,
    call_callback,
    component_ids,
    write_metric_csv,
)

NO_IMAGES_HINT = "No images available for this interval"


def _served(dash, path: str):
    resp = dash.server.test_client().get(path)
    assert resp.status_code == 200
    return json.loads(resp.data)


def _props(items) -> tuple[str, ...]:
    return tuple(f"{item['id']}.{item['property']}" for item in items)


def _wiring(dash) -> tuple:
    """(output, inputs, states, prevent_initial_call, runs in the browser) of every
    served callback, in registration order (as tests/test_viz_dash_split_golden.py)."""
    return tuple(
        (
            callback["output"],
            _props(callback["inputs"]),
            _props(callback["state"]),
            callback["prevent_initial_call"],
            callback["clientside_function"] is not None,
        )
        for callback in _served(dash, "/_dash-dependencies")
    )


def _area_app(folder, rows, image_manager=None):
    """A DashAppArea over a root_areas.csv of ``rows`` written in ``folder``."""
    from app.data_processing.data_processor_area import DataProcessorArea

    os.makedirs(folder, exist_ok=True)
    csv = write_metric_csv(os.path.join(folder, "root_areas.csv"), AREA_COLUMN, rows)
    return DashAppArea(DataProcessorArea(csv), str(folder), image_manager or FakeImageManager())


def _hover(dash, y=10):
    return call_callback(
        dash,
        "image-container.children",
        {"main-graph.hoverData": {"points": [{"y": y, "customdata": "2024.07.18"}]},
         "tube-selector.value": 1, "view-selector.value": "lines"},
        changed=["main-graph.hoverData"],
    )  # fmt: skip


def _json_nodes(node):
    if isinstance(node, list):
        for child in node:
            yield from _json_nodes(child)
    elif isinstance(node, dict) and "type" in node and "props" in node:
        yield node
        yield from _json_nodes(node["props"].get("children"))


@pytest.fixture(scope="module")
def apps(tmp_path_factory):
    length = build_length_app(tmp_path_factory.mktemp("metric_length"))
    area = build_area_app(tmp_path_factory.mktemp("metric_area"))
    return length, area


# ---- one class --------------------------------------------------------------------


def test_both_dashboards_are_the_main_class():
    assert issubclass(DashApp, MetricDashApp)
    assert issubclass(DashAppArea, MetricDashApp)
    own_methods = {
        n for n, v in vars(DashAppArea).items() if callable(v) and not n.startswith("__")
    }
    assert own_methods == set()


def test_area_shell_has_the_length_views():
    def views(spec):
        return [(v.label, v.value, v.has_controls) for v in spec.views]

    assert views(AREA_SHELL) == views(LENGTH_SHELL)
    assert AREA_SHELL.title == "Root Area — trial overview"
    assert AREA_SHELL.plot_filename == "root_area_plot"
    assert AREA_SHELL.tube_view == "lines"
    assert AREA_SHELL.controls_label == LENGTH_SHELL.controls_label


def test_area_page_has_every_length_component(apps):
    length, area = apps
    assert component_ids(_served(area.app, "/_dash-layout")) == component_ids(
        _served(length.app, "/_dash-layout")
    )


def test_area_callbacks_are_wired_like_length(apps):
    length, area = apps
    assert _wiring(area.app) == _wiring(length.app)


# ---- area figures and export ------------------------------------------------------------


def test_area_figures_chart_area(apps):
    _, area = apps
    assert area.show_growth_lines(1).layout.xaxis.title.text == "Root Area (mm²)"
    time = area.show_growth_over_time(selected_tubes=[1])
    assert time.layout.title.text == "Root Area Growth Over Time"
    assert time.layout.yaxis.title.text == "Total Area (mm²)"
    grid = area.create_faceted_depth_profile([1, 2])
    assert any("Avg Area" in (trace.hovertemplate or "") for trace in grid.data)


def test_area_time_export_names_area(apps):
    _, area = apps
    status, out = call_callback(
        area.app, "viz-export-download.data",
        {"viz-export-btn.n_clicks": 1, "view-selector.value": "time",
         "selected-tubes-store.data": [1]},
        changed=["viz-export-btn.n_clicks"],
    )  # fmt: skip
    assert status == 200
    download = out["viz-export-download"]["data"]
    assert download["filename"].startswith("root_area_time_")
    assert "Root area over time" in download["content"]
    assert "Total Area (mm²)" in download["content"]


def test_area_grid_ticks_stay_few_for_large_areas(tmp_path):
    rows = (
        ("T001_L005_2024.07.18", 1, 5, "2024.07.18", 4800.0),
        ("T001_L015_2024.07.18", 1, 15, "2024.07.18", 5100.0),
        ("T002_L005_2024.07.18", 2, 5, "2024.07.18", 4950.0),
        ("T002_L012_2024.08.07", 2, 12, "2024.08.07", 5200.0),
    )
    area = _area_app(tmp_path / "large", rows)
    figure = area.create_faceted_depth_profile([1, 2])
    assert 0 < len(figure.layout.xaxis.tickvals) <= 13


def test_area_time_view_with_one_tube_draws_without_a_band(tmp_path):
    rows = (
        ("T001_L005_2024.07.18", 1, 5, "2024.07.18", 1.5),
        ("T001_L015_2024.07.18", 1, 15, "2024.07.18", 2.5),
        ("T001_L005_2024.08.07", 1, 5, "2024.08.07", 4.0),
    )
    area = _area_app(tmp_path / "one", rows)
    figure = area.show_growth_over_time(
        selected_tubes=[1], show_field_average=True, show_field_variance=True
    )
    names = [trace.name for trace in figure.data]
    assert "Field Mean ±1 SD" not in names
    assert "Field Average" in names


# ---- area hover images --------------------------------------------------------------


def test_area_hover_without_images_says_so(tmp_path):
    area = build_area_app(tmp_path, FakeImageManager())
    status, out = _hover(area.app)
    assert status == 200
    assert NO_IMAGES_HINT in json.dumps(out["image-container"]["children"])


def test_area_hover_thumbnail_is_served_by_the_area_app(tmp_path):
    path = tmp_path / "T001_L005_2024.07.18.png"
    Image.fromarray(np.full((48, 64, 3), 90, np.uint8)).save(path)
    area = build_area_app(tmp_path, FakeImageManager({"T001_L005_2024.07.18.png": str(path)}))
    status, out = _hover(area.app)
    assert status == 200
    sources = [
        node["props"]["src"]
        for node in _json_nodes(out["image-container"]["children"])
        if node["type"] == "Img"
    ]
    assert len(sources) == 1
    resp = area.app.server.test_client().get(sources[0])
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg" and resp.data  # a served JPEG, not an empty body
