"""Dashboards: the open project's field of view in exports, failed figures shown, grid texts.

The export's calibration line names the field of view of the project the window opened
(not the project file found above the CSV); a figure builder that fails shows its error;
the Depth Profile Grid names its metric, unit and position axis; its export leaves the SD
of a one-image bin empty and counts the images; a non-finite value is an unreadable row;
the +/- 1 SD band is not called a variance.
"""

from __future__ import annotations

import io
import json
import math
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.viz_dash_helpers import (
    AREA_COLUMN,
    LENGTH_COLUMN,
    FakeImageManager,
    build_area_app,
    build_length_app,
    call_callback,
    write_metric_csv,
)

BUILDERS = [build_length_app, build_area_app]
METRICS = ["length", "area"]


def _project_file(folder, width=20.0, height=15.0) -> None:
    os.makedirs(folder, exist_ok=True)
    record = {"schema": 1, "calibration": {"fov_width_mm": width, "fov_height_mm": height}}
    with open(os.path.join(folder, "sprouts_project.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh)


def _length_app(csv: str, **kwargs):
    from app.data_processing.data_processor import DataProcessor
    from app.visualization.dash_app import DashApp

    return DashApp(DataProcessor(csv), os.path.dirname(csv), FakeImageManager(), **kwargs)


def _export(dash, values: dict) -> dict:
    status, out = call_callback(
        dash.app,
        "viz-export-download.data",
        {"viz-export-btn.n_clicks": 1, **values},
        changed=["viz-export-btn.n_clicks"],
    )
    assert status == 200
    return out["viz-export-download"]["data"]


def _calibration(content: str) -> str:
    (line,) = [ln for ln in content.splitlines() if ln.startswith("# calibration: ")]
    return line.removeprefix("# calibration: ")


def _read(content: str) -> pd.DataFrame:
    """The table as written: every cell as its text (an empty cell is "")."""
    return pd.read_csv(io.StringIO(content), comment="#", dtype=str, keep_default_na=False)


# ---- A: the export names the open project's field of view -------------------------------


def test_a_csv_in_the_project_folder_is_exported_with_the_projects_field_of_view(tmp_path):
    # <root>/root_lengths.csv: the lookup above the CSV never finds <root>'s own file.
    root = tmp_path / "trial"
    _project_file(root)
    csv = write_metric_csv(str(root / "root_lengths.csv"), LENGTH_COLUMN)

    without = _calibration(_export(_length_app(csv), {"view-selector.value": "stacked"})["content"])
    with_root = _calibration(
        _export(_length_app(csv, project_root=str(root)), {"view-selector.value": "stacked"})[
            "content"
        ]
    )

    assert "18.00 x 13.50 mm" in without  # the app's default: the regression
    assert with_root.startswith("field of view 20.00 x 15.00 mm")
    assert "18.00" not in with_root


def test_a_project_inside_another_is_exported_with_its_own_field_of_view(tmp_path):
    # KNOWN_ISSUES M7: outer/inner/skeletons/root_lengths.csv found outer's project file.
    outer = tmp_path / "outer"
    _project_file(outer)
    inner = outer / "inner"
    os.makedirs(inner / "skeletons")
    csv = write_metric_csv(str(inner / "skeletons" / "root_lengths.csv"), LENGTH_COLUMN)

    found = _calibration(_export(_length_app(csv), {"view-selector.value": "stacked"})["content"])
    own = _calibration(
        _export(_length_app(csv, project_root=str(inner)), {"view-selector.value": "stacked"})[
            "content"
        ]
    )

    assert "20.00 x 15.00 mm" in found  # the outer project's file
    assert own.startswith("field of view 18.00 x 13.50 mm")
    assert "20.00" not in own


@pytest.mark.parametrize(
    "module, builder_name",
    [
        ("app.visualization.root_length_visulization", "RootLengthVisualization"),
        ("app.visualization.root_area_visualization", "RootAreaVisualization"),
    ],
    ids=METRICS,
)
def test_the_window_hands_the_loaded_projects_root_to_the_build(tmp_path, module, builder_name):
    # The root is read on the GUI thread, where the build is set up; the worker gets a string.
    import importlib

    cls = getattr(importlib.import_module(module), builder_name)
    manager = SimpleNamespace(original_folder=str(tmp_path / "trial"), images={})
    window = SimpleNamespace(
        csv_path=str(tmp_path / "trial" / "mask" / "root_areas.csv"),
        save_directory=str(tmp_path / "trial" / "mask"),
        image_manager=manager,
    )

    build = cls._dashboard_builder(window)

    assert build.keywords == {"project_root": os.path.normpath(str(tmp_path / "trial"))}
    assert build.args[2] is manager


def test_the_window_without_a_project_keeps_the_lookup_above_the_csv(tmp_path):
    from app.visualization.root_length_visulization import RootLengthVisualization

    window = SimpleNamespace(
        csv_path=str(tmp_path / "root_lengths.csv"),
        save_directory=str(tmp_path),
        image_manager=None,
    )

    assert RootLengthVisualization._dashboard_builder(window).keywords == {"project_root": None}


@pytest.mark.parametrize("metric", METRICS)
def test_the_dashboard_builders_pass_the_project_root_to_the_app(tmp_path, metric):
    from app.visualization.root_area_visualization import build_area_dashboard
    from app.visualization.root_length_visulization import build_length_dashboard

    root = tmp_path / "trial"
    _project_file(root)
    if metric == "length":
        csv = write_metric_csv(str(root / "root_lengths.csv"), LENGTH_COLUMN)
        _, dash = build_length_dashboard(csv, str(root), FakeImageManager(), project_root=str(root))
    else:
        csv = write_metric_csv(str(root / "root_areas.csv"), AREA_COLUMN)
        _, dash = build_area_dashboard(csv, str(root), FakeImageManager(), project_root=str(root))

    line = _calibration(_export(dash, {"view-selector.value": "stacked"})["content"])
    assert line.startswith("field of view 20.00 x 15.00 mm")


# ---- B: a failed figure builder shows its error ------------------------------------------


def _break_stacked(monkeypatch, dashboard):
    from app.visualization import dash_visualizations

    def broken(*_args, **_kwargs):
        raise RuntimeError("no totals")

    monkeypatch.setattr(dash_visualizations, "stacked_bar_figure", broken)
    return {"view-selector.value": "stacked"}


def _break_time(monkeypatch, dashboard):
    def broken(*_args, **_kwargs):
        raise RuntimeError("no totals")

    monkeypatch.setattr(dashboard.data_cache, "get_field_growth", broken)
    return {
        "view-selector.value": "time",
        "selected-tubes-store.data": [1],
        "growth-viz-options.value": ["show_field_avg", "show_field_var"],
    }


def _break_lines(monkeypatch, dashboard):
    def broken(*_args, **_kwargs):
        raise RuntimeError("no totals")

    monkeypatch.setattr(dashboard.data_cache, "get_interval_data", broken)
    return {"view-selector.value": "lines", "tube-selector.value": 1}


@pytest.mark.parametrize("breaker", [_break_stacked, _break_time, _break_lines])
@pytest.mark.parametrize("builder", BUILDERS, ids=METRICS)
def test_a_figure_that_cannot_be_built_shows_its_error(tmp_path, monkeypatch, builder, breaker):
    from app.visualization import theme

    dashboard = builder(tmp_path)
    values = breaker(monkeypatch, dashboard)

    _, out = call_callback(dashboard.app, "main-graph.figure", values, ["view-selector.value"])

    (note,) = out["main-graph"]["figure"]["layout"]["annotations"]
    assert note["text"] == "Error: no totals"
    assert note["font"]["color"] == theme.DANGER
    assert out["main-graph"]["figure"]["data"] == []


# ---- C: the Depth Profile Grid names metric, unit and position axis ------------------------


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


@pytest.mark.parametrize("n_tubes", [1, 6])
def test_the_grid_titles_its_value_axis_once_under_the_grid(tmp_path, n_tubes):
    """Six tubes are twelve bottom panels: one title per panel touched its neighbours below
    about 790 px of graph width. One centred title reads with one tube and with six."""
    rows = tuple(
        (f"T{tube:03d}_L{pos:03d}_{date}", tube, pos, date, 1.0 + tube + pos / 10)
        for tube in range(1, 7)
        for pos in (5, 15, 25)
        for date in ("2024.07.18", "2024.08.07")
    )
    dashboard = _length_app(write_metric_csv(str(tmp_path / "l.csv"), LENGTH_COLUMN, rows))
    layout = dashboard.create_faceted_depth_profile(list(range(1, n_tubes + 1))).layout

    titled = [k for k in layout if k.startswith("xaxis") and layout[k].title.text]
    assert titled == []
    (value,) = [a for a in layout.annotations if a.text == "Mean Length (mm)"]
    first, last = layout.xaxis.domain, layout[f"xaxis{n_tubes if n_tubes > 1 else ''}"].domain
    assert value.x == pytest.approx((first[0] + last[1]) / 2)
    assert value.yref == "paper" and value.y == 0 and value.yanchor == "top"


# ---- D: the grid export: no SD for a one-image bin, and its count --------------------------


def test_the_grid_export_leaves_a_one_image_sd_empty_and_counts_the_images(tmp_path):
    rows = (
        ("T001_L005_2024.07.18", 1, 5, "2024.07.18", 1.0),
        ("T002_L006_2024.07.18", 2, 6, "2024.07.18", 3.0),
        ("T001_L015_2024.07.18", 1, 15, "2024.07.18", 2.5),
    )
    csv = write_metric_csv(str(tmp_path / "root_lengths.csv"), LENGTH_COLUMN, rows)
    download = _export(
        _length_app(csv),
        {"view-selector.value": "faceted", "faceted-tubes-store.data": [1],
         "faceted-dates-store.data": []},
    )  # fmt: skip

    table = _read(download["content"])
    assert list(table.columns) == [
        "Tube", "Date", "Interval start", "Interval end", "Mean Length (mm)", "Count",
        "Field Mean Length (mm)", "Field SD (mm)", "Field Count",
    ]  # fmt: skip
    by_end = {int(r[3]): r for r in table.itertuples(index=False)}
    # L1-10 pools tube 1 (1.0) and tube 2 (3.0): SD sqrt(2), two images.
    assert float(by_end[10][7]) == pytest.approx(math.sqrt(2))
    assert by_end[10][8] == "2"
    # L11-20 has one image: no SD (as the Depth Profile export), one image.
    assert by_end[20][7] == ""
    assert by_end[20][8] == "1"


def test_the_grid_figure_still_draws_a_one_image_bin_as_before(tmp_path):
    rows = (("T001_L015_2024.07.18", 1, 15, "2024.07.18", 2.5),)
    csv = write_metric_csv(str(tmp_path / "root_lengths.csv"), LENGTH_COLUMN, rows)
    figure = _length_app(csv).create_faceted_depth_profile([1])

    field = figure.data[0]
    assert [list(row) for row in field.customdata] == [[11, 20, pytest.approx(2.5), 0]]


# ---- E: a non-finite value is an unreadable row ----------------------------------------


@pytest.mark.parametrize("cell", ["inf", "-inf", "Infinity"])
@pytest.mark.parametrize("metric", METRICS)
def test_a_non_finite_value_is_left_out_as_unreadable(tmp_path, metric, cell):
    from app.data_processing.data_processor import DataProcessor
    from app.data_processing.data_processor_area import DataProcessorArea

    column, processor = {
        "length": (LENGTH_COLUMN, DataProcessor),
        "area": (AREA_COLUMN, DataProcessorArea),
    }[metric]
    path = tmp_path / "metric.csv"
    path.write_text(
        f"Image,Tube,Position,Date,Time,{column},Error\n"
        "A_T001_L005_2024.07.18_120000,1,5,2024.07.18,120000,2.5,\n"
        f"A_T002_L007_2024.07.18_120000,2,7,2024.07.18,120000,{cell},\n",
        encoding="utf-8",
    )

    loaded = processor(str(path))

    assert list(loaded.df["Tube"]) == [1]
    assert loaded.exclusions.unreadable == 1
    assert loaded.exclusions.message() == (
        "1 row is not shown: 1 row with a missing or non-numeric tube, position, date or value."
    )


@pytest.mark.parametrize("column", ["Tube", "Position"])
def test_a_non_finite_tube_or_position_is_left_out_as_unreadable(tmp_path, column):
    from app.data_processing.data_processor import DataProcessor

    cells = {"Tube": "2", "Position": "7", column: "inf"}
    path = tmp_path / "root_lengths.csv"
    path.write_text(
        f"Image,Tube,Position,Date,Time,{LENGTH_COLUMN},Error\n"
        "A_T001_L005_2024.07.18_120000,1,5,2024.07.18,120000,2.5,\n"
        f"A_T002_L007_2024.07.18_120000,{cells['Tube']},{cells['Position']},2024.07.18,"
        "120000,1.0,\n",
        encoding="utf-8",
    )

    loaded = DataProcessor(str(path))

    assert list(loaded.df["Tube"]) == [1]  # the file still loads: only the row is left out
    assert loaded.exclusions.unreadable == 1


def test_the_grid_of_a_csv_with_an_infinite_value_is_drawn_and_exported(tmp_path):
    rows = (
        ("T001_L005_2024.07.18", 1, 5, "2024.07.18", 2.5),
        ("T002_L007_2024.07.18", 2, 7, "2024.07.18", float("inf")),
    )
    csv = write_metric_csv(str(tmp_path / "root_lengths.csv"), LENGTH_COLUMN, rows)
    dashboard = _length_app(csv)

    figure = dashboard.create_faceted_depth_profile([1])
    assert not figure.layout.annotations or "Error" not in figure.layout.annotations[0].text
    table = _read(
        _export(
            dashboard,
            {"view-selector.value": "faceted", "faceted-tubes-store.data": [1],
             "faceted-dates-store.data": []},
        )["content"]
    )  # fmt: skip
    # Tube 2's row is left out, so the field mean of L1-10 is tube 1's alone.
    assert list(table["Field Mean Length (mm)"]) == ["2.5"]
    assert list(table["Field Count"]) == ["1"]


# ---- F: the +/- 1 SD band is named as one ----------------------------------------------


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
