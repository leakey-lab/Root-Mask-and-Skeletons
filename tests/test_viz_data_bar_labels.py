"""The Totals by Tube bars of both dashboards: their values, their gaps and their labels.

Values: each bar is a tube's total on one date; a date a tube was not imaged on is no bar (a
JSON null, never a 0.00 bar, which read as "no roots that day" and, in a typed array, restarted
the tube's later dates at 0).

Labels (PKG-VIZ, V2): every bar of the stacked and profile charts carried its value as a text
label. With the owner's ~170 tubes those labels shrink to unreadable specks, and each one costs
Plotly a text measurement: on a 46k-row synthetic trial the stacked chart's first render took
795 ms with labels and 314 ms without (median of 5, QtWebEngine). Labels are kept when at most
``theme.MAX_LABELLED_BARS`` bars share the axis; the values stay in the hover either way. A
label takes plotly.js 4's text colour (owner decision #51, 2026-10-31): ``theme.bar_labels`` sets
none (``test_viz_dash_plotly_bundle.py`` checks that the bundle picks it by contrast).
"""

from __future__ import annotations

import os

import pytest

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.data_processing.data_processor import DataProcessor
from app.data_processing.data_processor_area import DataProcessorArea
from app.visualization import theme
from app.visualization.dash_app_area import DashAppArea
from app.visualization.dash_data_cache import DataCache
from app.visualization.dash_visualizations import stacked_bar_figure
from tests.viz_dash_helpers import (
    AREA_COLUMN,
    LENGTH_COLUMN,
    build_length_app,
    call_callback,
    write_metric_csv,
)
from tests.viz_data_support import write_metric_csv as write_rows_csv

# ---- values ---------------------------------------------------------------------------


def _stacked_values(figure) -> dict:
    return {trace.name: dict(zip(trace.x, trace.y, strict=True)) for trace in figure.data}


def test_stacked_bars_leave_out_dates_a_tube_was_not_measured_on(tmp_path):
    # Tube 3 was imaged on 2024.08.07 only. Its 2024.07.18 bar was a 0.00 bar that
    # the unified hover listed as "0.00 mm", as if the tube had no roots that day.
    import math

    from app.data_processing.data_processor_area import DataProcessorArea
    from app.visualization.dash_app_area import DashAppArea

    csv = write_metric_csv(os.path.join(tmp_path, "root_areas.csv"), AREA_COLUMN)
    for dashboard in (
        build_length_app(tmp_path),
        DashAppArea(DataProcessorArea(csv), str(tmp_path)),
    ):
        values = _stacked_values(dashboard.create_stacked_bar_chart())

        assert math.isnan(values["2024-07-18"]["Tube 3"])
        assert values["2024-08-07"]["Tube 3"] == 0.75
        assert values["2024-07-18"]["Tube 1"] == 4.0  # 1.5 + 2.5


def test_stacked_bars_reach_the_browser_with_null_for_a_date_a_tube_missed(tmp_path):
    # Tube 1 was imaged on the first and third dates, not the second. The area chart sent
    # its bars as a typed array; plotly.js stacks the NaN inside one and restarted the
    # tube's later dates at 0 (Totals by Tube drew 162 of 170 real tubes too short). A
    # JSON null is left out of the stack.
    from app.data_processing.data_processor import DataProcessor
    from app.data_processing.data_processor_area import DataProcessorArea
    from app.visualization.dash_app import DashApp
    from app.visualization.dash_app_area import DashAppArea
    from tests.viz_dash_helpers import LENGTH_COLUMN, FakeImageManager

    rows = (
        ("T001_L005_2024.07.18", 1, 5, "2024.07.18", 1.5),
        ("T001_L005_2024.08.27", 1, 5, "2024.08.27", 2.5),
        ("T002_L005_2024.07.18", 2, 5, "2024.07.18", 1.0),
        ("T002_L005_2024.08.07", 2, 5, "2024.08.07", 2.0),
        ("T002_L005_2024.08.27", 2, 5, "2024.08.27", 3.0),
    )
    length_csv = write_metric_csv(os.path.join(tmp_path, "root_lengths.csv"), LENGTH_COLUMN, rows)
    area_csv = write_metric_csv(os.path.join(tmp_path, "root_areas.csv"), AREA_COLUMN, rows)
    dashboards = {
        "length": DashApp(DataProcessor(length_csv), str(tmp_path), FakeImageManager()),
        "area": DashAppArea(DataProcessorArea(area_csv), str(tmp_path)),
    }
    for name, dashboard in dashboards.items():
        _, out = call_callback(
            dashboard.app,
            "main-graph.figure",
            {"view-selector.value": "stacked"},
            changed=["view-selector.value"],
        )
        bars = {trace["name"]: trace["y"] for trace in out["main-graph"]["figure"]["data"]}

        assert bars == {
            "2024-07-18": [1.5, 1.0],
            "2024-08-07": [None, 2.0],
            "2024-08-27": [2.5, 3.0],
        }, name


# ---- labels ---------------------------------------------------------------------------


def _rows(tubes: int, positions: int = 1, dates=("2024.07.18",)):
    return tuple(
        (f"T{t:03d}_L{p:03d}_{d}", t, p, d, float(t + p))
        for t in range(1, tubes + 1)
        for p in range(1, positions + 1)
        for d in dates
    )


def _length_charts(tmp_path, rows):
    from app.data_processing.data_processor import DataProcessor
    from app.visualization.dash_app import DashApp
    from tests.viz_dash_helpers import FakeImageManager

    csv = write_metric_csv(os.path.join(tmp_path, "root_lengths.csv"), LENGTH_COLUMN, rows)
    return DashApp(DataProcessor(csv), str(tmp_path), FakeImageManager())


def _area_charts(tmp_path, rows):
    from app.data_processing.data_processor_area import DataProcessorArea
    from app.visualization.dash_app_area import DashAppArea

    csv = write_metric_csv(os.path.join(tmp_path, "root_areas.csv"), AREA_COLUMN, rows)
    return DashAppArea(DataProcessorArea(csv), str(tmp_path))


def _labelled(figure) -> bool:
    bars = [trace for trace in figure.data if trace.type == "bar"]
    assert bars
    shown = {trace.textposition != "none" and trace.text is not None for trace in bars}
    assert len(shown) == 1, "all bars of a chart are labelled alike"
    return shown.pop()


@pytest.mark.parametrize("build", [_length_charts, _area_charts], ids=["length", "area"])
def test_stacked_bars_keep_their_labels_for_a_few_tubes(tmp_path, build):
    figure = build(tmp_path, _rows(theme.MAX_LABELLED_BARS)).create_stacked_bar_chart()
    assert _labelled(figure)
    assert figure.data[0].text[0] == "2.00"  # tube 1, L1: 1 + 1


@pytest.mark.parametrize("build", [_length_charts, _area_charts], ids=["length", "area"])
def test_stacked_bars_drop_labels_that_cannot_be_read(tmp_path, build):
    figure = build(tmp_path, _rows(theme.MAX_LABELLED_BARS + 1)).create_stacked_bar_chart()
    assert not _labelled(figure)


def test_bar_labels_set_no_text_colour():
    assert theme.bar_labels(["1.00"], 1) == {"text": ["1.00"], "textposition": "auto"}
    assert theme.bar_labels(["1.00"], theme.MAX_LABELLED_BARS + 1) == {"textposition": "none"}
    assert not hasattr(theme, "inside_label_colour")  # the plotly.js 3 override is gone


def _five_date_csv(tmp_path, headers):
    dates = ("2024.06.03", "2024.06.24", "2024.07.16", "2024.08.07", "2024.08.28")
    rows = [(tube, 15, date, 1.5 + tube + d) for tube in (1, 2) for d, date in enumerate(dates)]
    return write_rows_csv(tmp_path / "metric.csv", headers, rows)


def test_the_dashboards_bars_leave_the_label_colour_to_plotly_js(tmp_path):
    # Five dates: the second one is #4f97d1, whose labels plotly.js 4 draws white.
    cache = DataCache(DataProcessor(_five_date_csv(tmp_path, LENGTH_CSV_HEADERS)))
    area = DashAppArea(DataProcessorArea(_five_date_csv(tmp_path, AREA_CSV_HEADERS)), "")
    for figure in (stacked_bar_figure(cache, "Length", "mm"), area.create_stacked_bar_chart()):
        traces = figure.to_plotly_json()["data"]
        assert traces[1]["marker"]["color"] == "#4f97d1"
        for trace in traces:
            assert trace["textposition"] == "auto" and trace["text"] is not None
            assert "insidetextfont" not in trace and "textfont" not in trace
