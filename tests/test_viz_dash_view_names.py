"""O27 / D7: both dashboards name their views, titles and legends with one set of words,
and a measurement date has one colour (a place on the blue ramp) in every view."""

from __future__ import annotations

import json

import pytest

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.data_processing.data_processor import DataProcessor
from app.data_processing.data_processor_area import DataProcessorArea
from app.visualization import theme
from app.visualization.dash_app_area import AREA_SHELL, DashAppArea
from app.visualization.dash_metric_layout import LENGTH_SHELL
from tests.viz_dash_helpers import build_area_app, build_length_app
from tests.viz_data_support import ROWS, write_metric_csv
from tests.viz_data_support import Charts as _Charts

BUILDERS = [build_length_app, build_area_app]
METRICS = ["length", "area"]


@pytest.fixture
def length(tmp_path):
    return _Charts(DataProcessor(write_metric_csv(tmp_path / "l.csv", LENGTH_CSV_HEADERS)))


@pytest.fixture
def area(tmp_path):
    return DashAppArea(
        DataProcessorArea(write_metric_csv(tmp_path / "a.csv", AREA_CSV_HEADERS)), str(tmp_path)
    )


def test_both_dashboards_label_their_views_with_the_same_words():
    length = {v.value: v.label for v in LENGTH_SHELL.views}
    area = {v.value: v.label for v in AREA_SHELL.views}
    assert length == {
        "stacked": "Totals by Tube",
        "time": "Growth Over Time",
        "lines": "Depth Profile",
        "faceted": "Depth Profile Grid",
    }
    assert area == length
    assert AREA_SHELL.tube_view == LENGTH_SHELL.tube_view


def test_chart_titles_use_one_scheme(length, area):
    assert length.create_stacked_bar_chart().layout.title.text == "Root Length by Tube and Date"
    assert area.create_stacked_bar_chart().layout.title.text == "Root Area by Tube and Date"
    time = length.show_growth_over_time([1], True, True, True)
    assert time.layout.title.text == "Root Length Growth Over Time"
    assert area.show_growth_over_time().layout.title.text == "Root Area Growth Over Time"
    assert length.show_growth_lines(1).layout.title.text == "Root Length Depth Profile - Tube 1"
    assert area.show_growth_lines(1).layout.title.text == "Root Area Depth Profile - Tube 1"
    for dashboard, label in ((length, "Length"), (area, "Area")):
        faceted = dashboard.create_faceted_depth_profile([1, 2])
        assert faceted.layout.title.text == (
            f"Root {label} Depth Profile Grid - Field Average (Left) vs Tube (Right)"
        )


def test_legend_title_and_depth_axes_agree(length, area):
    for dashboard in (length, area):
        for fig in (dashboard.create_stacked_bar_chart(), dashboard.show_growth_lines(1)):
            assert fig.layout.legend.title.text == "Measurement date"
        assert dashboard.show_growth_lines(1).layout.yaxis.title.text == "Position (L)"
        grid = dashboard.create_faceted_depth_profile([1, 2])
        assert grid.layout.yaxis.title.text == "Position (L)"


def test_hover_dates_are_iso(length):
    for fig in (
        length.create_stacked_bar_chart(),
        length.show_growth_over_time([1, 2], True, True),
    ):
        assert "%b" not in fig.to_json()


def _two_tube_rows(extra_dates=()):
    """ROWS (dates A, B) plus tube-1-only rows on later dates; tube 2 misses the first date."""
    rows = [row for row in ROWS if not (row[0] == 2 and row[2] == "2024.07.18")]
    rows += [(1, 5, date, 1.0) for date in extra_dates]
    return rows


def test_a_date_has_one_colour_in_every_view_and_every_tube(tmp_path):
    rows = _two_tube_rows(["2024.09.01"])
    length = _Charts(DataProcessor(write_metric_csv(tmp_path / "l.csv", LENGTH_CSV_HEADERS, rows)))
    area = DashAppArea(
        DataProcessorArea(write_metric_csv(tmp_path / "a.csv", AREA_CSV_HEADERS, rows)),
        str(tmp_path),
    )
    for dashboard in (length, area):
        cache = dashboard.data_cache
        ramp = theme.date_colors(len(cache.dates))
        names = [d.strftime("%Y-%m-%d") for d in cache.dates]
        stacked = {t.name: t.marker.color for t in dashboard.create_stacked_bar_chart().data}
        assert stacked == dict(zip(names, ramp, strict=True))
        lines = {
            tube: {t.name: t.line.color for t in dashboard.show_growth_lines(tube).data}
            for tube in (1, 2)
        }
        assert lines[1] == dict(zip(names, ramp, strict=True))
        assert set(lines[2]) == {"2024-08-07"}  # tube 2 has no row on the first and last date
        assert lines[2]["2024-08-07"] == lines[1]["2024-08-07"] == ramp[1]


def test_more_than_13_dates_stay_on_the_ramp(tmp_path):
    days = [f"2024.09.{day:02d}" for day in range(1, 16)]
    rows = [(1, 5, day, 1.0) for day in days]
    charts = _Charts(DataProcessor(write_metric_csv(tmp_path / "l.csv", LENGTH_CSV_HEADERS, rows)))
    colours = [t.marker.color for t in charts.create_stacked_bar_chart().data]
    assert len(colours) == 15 and set(colours) <= set(theme.date_colors(15))
    assert colours[0] != colours[-1]


# ---- the Depth Profile Grid and the Growth Over Time band name what they show ------------


@pytest.mark.parametrize(
    "builder, label, unit", [(build_length_app, "Length", "mm"), (build_area_app, "Area", "mm²")],
    ids=METRICS,
)  # fmt: skip
def test_the_depth_profile_grid_names_its_metric_unit_and_positions(tmp_path, builder, label, unit):
    dashboard = builder(tmp_path)
    layout = dashboard.create_faceted_depth_profile([1, 2]).layout  # 2 dates x 2 tubes

    assert layout.title.text == (
        f"Root {label} Depth Profile Grid - Field Average (Left) vs Tube (Right)"
    )
    assert layout.yaxis.title.text == "Position (L)"  # positions, as the Depth Profile view
    assert layout.yaxis.title.text == dashboard.show_growth_lines(1).layout.yaxis.title.text
    # The value axis is titled once, under the grid (FR-7): no panel's x axis carries it.
    value = f"Mean {label} ({unit})"
    assert [a.text for a in layout.annotations].count(value) == 1
    assert not any(layout[f"xaxis{s}"].title.text for s in ("", "2", "3", "4"))


@pytest.mark.parametrize("builder", BUILDERS, ids=METRICS)
def test_the_sd_band_is_not_called_a_variance(tmp_path, builder):
    dashboard = builder(tmp_path)
    figure = dashboard.show_growth_over_time([1, 2], True, True, True)
    band = next(trace for trace in figure.data if trace.fill == "tonexty")

    assert band.name == "Field Mean ±1 SD"
    assert band.hovertemplate.startswith("<b>Field ±1 SD Range</b>")
    layout = json.loads(dashboard.app.server.test_client().get("/_dash-layout").data)
    served = json.dumps(layout, ensure_ascii=False)
    assert '"label": "Show Field ±1 SD Shading", "value": "show_field_var"' in served
    assert "Variance" not in served
