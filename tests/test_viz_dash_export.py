"""PV-16: 'Export CSV' writes the data behind the current view, with provenance.

The expected numbers below are derived by hand from ``tests.viz_dash_helpers.ROWS``:

    tube 1: 2024-07-18 L5=1.5 L15=2.5 | 2024-08-07 L5=4.0
    tube 2: 2024-07-18 L5=3.0         | 2024-08-07 L12=5.25
    tube 3:                           | 2024-08-07 L20=0.75
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from datetime import UTC, datetime

import pandas as pd
import pytest
from dash import Dash

import app
from app import provenance
from app.visualization import dash_export
from tests.viz_dash_helpers import (
    AREA_COLUMN,
    LENGTH_COLUMN,
    FakeImageManager,
    build_area_app,
    build_length_app,
    call_callback,
    write_metric_csv,
)

METRICS = ["length", "area"]


def _read(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), comment="#")


def _header(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        if line.startswith("# ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            fields[key] = value
    return fields


def _sha256(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@pytest.fixture
def length_app(tmp_path):
    return build_length_app(tmp_path)


@pytest.fixture
def area_app(tmp_path):
    return build_area_app(tmp_path)


def _export(dash, values) -> dict:
    """Click "Export CSV" of `dash` (a Dash app, or a dashboard holding one as `.app`)."""
    dash = dash if isinstance(dash, Dash) else dash.app
    status, out = call_callback(
        dash, "viz-export-download.data", {"viz-export-btn.n_clicks": 1, **values},
        changed=["viz-export-btn.n_clicks"],
    )  # fmt: skip
    assert status == 200
    return out["viz-export-download"]["data"]


# ---- through the dashboard ------------------------------------------------------


def test_stacked_export_holds_tube_date_totals_and_provenance(length_app, tmp_path):
    download = _export(length_app.app, {"view-selector.value": "stacked"})
    assert download["filename"].startswith("root_length_stacked_")
    assert download["filename"].endswith(".csv")

    table = _read(download["content"])
    assert list(table.columns) == ["Tube", "Date", "Total Length (mm)"]
    rows = {(r.Tube, r.Date): r[2] for r in table.itertuples(index=False)}
    assert rows == {
        (1, "2024-07-18"): pytest.approx(4.0),
        (1, "2024-08-07"): pytest.approx(4.0),
        (2, "2024-07-18"): pytest.approx(3.0),
        (2, "2024-08-07"): pytest.approx(5.25),
        (3, "2024-08-07"): pytest.approx(0.75),
    }

    header = _header(download["content"])
    assert header["source_sha256"] == _sha256(tmp_path / "root_lengths.csv")
    assert header["source_csv"] == "root_lengths.csv"
    assert header["view"] == "stacked"
    assert header["calibration"].startswith("field of view 18.00 x 13.50 mm")
    assert "no record next to root_lengths.csv" in header["calibration"]
    assert app.__version__ in header["software"]
    generated = datetime.fromisoformat(header["generated_at"].replace("Z", "+00:00"))
    assert generated.tzinfo is not None


def test_the_export_takes_the_calibration_from_the_csv_record(length_app, tmp_path):
    provenance.write_sidecar(
        tmp_path / "root_lengths.csv",
        {"calibration": {"fov_width_mm": 18.0, "fov_height_mm": 13.0}},
    )
    line = _header(_export(length_app.app, {"view-selector.value": "stacked"})["content"])[
        "calibration"
    ]
    assert "18.00 x 13.00 mm, as recorded with root_lengths.csv" in line
    assert "; this app now uses 18.00 x 13.50 mm" in line
    assert line.count("(mm per pixel = field of view / image resolution)") == 1


def test_a_record_equal_to_this_apps_calibration_adds_no_note(length_app, tmp_path):
    provenance.write_sidecar(
        tmp_path / "root_lengths.csv",
        {"calibration": provenance.calibration_info()},
    )
    line = _header(_export(length_app.app, {"view-selector.value": "stacked"})["content"])[
        "calibration"
    ]
    assert line == (
        "field of view 18.00 x 13.50 mm, as recorded with root_lengths.csv "
        "(mm per pixel = field of view / image resolution)"
    )


def test_time_export_has_the_field_average_and_the_selected_tubes(length_app):
    download = _export(
        length_app.app,
        {"view-selector.value": "time", "selected-tubes-store.data": [2],
         "growth-viz-options.value": ["show_field_avg", "show_field_var"]},
    )  # fmt: skip
    table = _read(download["content"])
    assert list(table.columns) == ["Series", "Date", "Total Length (mm)", "SD (mm)", "Tubes"]
    field = table[table.Series == "Field average"].set_index("Date")
    assert field.loc["2024-07-18", "Total Length (mm)"] == pytest.approx(3.5)
    assert field.loc["2024-07-18", "SD (mm)"] == pytest.approx(math.sqrt(0.5))
    assert field.loc["2024-08-07", "Total Length (mm)"] == pytest.approx(10.0 / 3)
    assert field.loc["2024-08-07", "Tubes"] == 3
    tube = table[table.Series == "Tube 2"].set_index("Date")["Total Length (mm)"]
    assert tube.to_dict() == {"2024-07-18": pytest.approx(3.0), "2024-08-07": pytest.approx(5.25)}
    assert _header(download["content"])["filters"] == "tubes=2; field average=shown"


def test_time_export_without_field_average_lists_only_tubes(length_app):
    download = _export(
        length_app.app,
        {"view-selector.value": "time", "selected-tubes-store.data": [1],
         "growth-viz-options.value": []},
    )  # fmt: skip
    assert set(_read(download["content"]).Series) == {"Tube 1"}


LINES_EXPORT = [
    (build_length_app, "root_length", "root_lengths.csv",
     ["Mean Length (mm)", "Min Length (mm)", "Max Length (mm)", "SD (mm)"]),
    (build_area_app, "root_area", "root_areas.csv",
     ["Mean Area (mm²)", "Min Area (mm²)", "Max Area (mm²)", "SD (mm²)"]),
]  # fmt: skip


@pytest.mark.parametrize("builder, slug, source, stats", LINES_EXPORT, ids=["length", "area"])
def test_lines_export_lists_depth_intervals_of_the_plotted_tube(
    tmp_path, builder, slug, source, stats
):
    download = _export(
        builder(tmp_path).app, {"view-selector.value": "lines", "tube-selector.value": 1}
    )
    assert download["filename"].startswith(f"{slug}_lines_")
    table = _read(download["content"])
    assert list(table.columns) == [
        "Tube", "Date", "Interval start", "Interval end", *stats, "Count"
    ]  # fmt: skip
    got = {(r.Date, r[3]): (r[4], r.Count) for r in table.itertuples(index=False)}
    assert got == {
        ("2024-07-18", 10): (pytest.approx(1.5), 1),
        ("2024-07-18", 20): (pytest.approx(2.5), 1),
        ("2024-08-07", 10): (pytest.approx(4.0), 1),
    }
    assert set(table.Tube) == {1}
    assert _header(download["content"])["source_sha256"] == _sha256(tmp_path / source)


def test_the_depth_profile_export_lists_measured_intervals_with_their_spread(tmp_path):
    # L5, L7 and L25 on one date: L1-10 holds two images, L21-30 one, L11-20 none.
    rows = (
        ("T001_L005_2024.07.18", 1, 5, "2024.07.18", 2.0),
        ("T001_L007_2024.07.18", 1, 7, "2024.07.18", 4.5),
        ("T001_L025_2024.07.18", 1, 25, "2024.07.18", 10.0),
    )
    csv = write_metric_csv(str(tmp_path / "root_lengths.csv"), LENGTH_COLUMN, rows)

    content = _export(_length_app(csv), {"view-selector.value": "lines", "tube-selector.value": 1})
    table = _read_cells(content["content"])

    assert table["Interval end"].tolist() == ["10", "30"]  # no row for the empty L11-20
    first, last = table.iloc[0], table.iloc[1]
    assert (first["Min Length (mm)"], first["Max Length (mm)"]) == ("2.0", "4.5")
    assert float(first["SD (mm)"]) == pytest.approx(math.sqrt(((2.0 - 3.25) ** 2) * 2))
    assert (first["Count"], last["Count"]) == ("2", "1")
    assert (last["Min Length (mm)"], last["Max Length (mm)"]) == ("10.0", "10.0")
    assert last["SD (mm)"] == ""  # one image: no SD


def test_faceted_export_pairs_tube_and_field_means(length_app):
    download = _export(
        length_app.app,
        {"view-selector.value": "faceted", "faceted-tubes-store.data": [2],
         "faceted-dates-store.data": []},
    )  # fmt: skip
    table = _read(download["content"])
    assert list(table.columns) == [
        "Tube", "Date", "Interval start", "Interval end", "Mean Length (mm)", "Count",
        "Field Mean Length (mm)", "Field SD (mm)", "Field Count",
    ]  # fmt: skip
    got = {(r.Date, r[3]): (r[4], r[6]) for r in table.itertuples(index=False)}
    # Tube 2 has L5 on 07-18 (interval L1-10) and L12 on 08-07 (L11-20). Field
    # means pool every tube in the same interval and date:
    #   07-18 L1-10  = mean(tube 1 L5 1.5, tube 2 L5 3.0)     = 2.25
    #   08-07 L11-20 = mean(tube 2 L12 5.25, tube 3 L20 0.75) = 3.0
    assert got == {
        ("2024-07-18", 10): (pytest.approx(3.0), pytest.approx(2.25)),
        ("2024-08-07", 20): (pytest.approx(5.25), pytest.approx(3.0)),
    }


def test_faceted_export_without_tubes_is_not_offered(length_app):
    status, _ = call_callback(
        length_app.app,
        "viz-export-download.data",
        {"viz-export-btn.n_clicks": 1, "view-selector.value": "faceted",
         "faceted-tubes-store.data": []},
        changed=["viz-export-btn.n_clicks"],
    )  # fmt: skip
    assert status == 204


def test_area_stacked_export_uses_the_area_column(area_app):
    table = _read(_export(area_app.app, {"view-selector.value": "stacked"})["content"])
    assert list(table.columns) == ["Tube", "Date", "Total Area (mm²)"]
    assert table["Total Area (mm²)"].sum() == pytest.approx(17.0)


# ---- the Qt-free writer ------------------------------------------------------------


def test_render_csv_round_trips_the_table(tmp_path):
    source = tmp_path / "root_lengths.csv"
    source.write_text("Image,Tube\n", encoding="utf-8")
    frame = pd.DataFrame({"Tube": [1, 2], "Date": ["2024-07-18", "2024-08-07"], "v": [0.1, 2.5]})
    table = dash_export.ExportTable(frame, description="demo rows", filters={"tubes": "all"})
    when = datetime(2026, 9, 28, 17, 45, 12, tzinfo=UTC)
    text = dash_export.render_csv(
        table,
        source_csv=str(source),
        metric_column="Length (mm)",
        view="stacked",
        generated_at=when,
    )
    pd.testing.assert_frame_equal(_read(text), frame)
    header = _header(text)
    assert header["generated_at"] == "2026-09-28T17:45:12Z"
    assert header["description"] == "demo rows"
    assert header["metric_column"] == "Length (mm)"
    assert header["filters"] == "tubes=all"


def test_missing_source_is_recorded_not_fatal(tmp_path):
    text = dash_export.render_csv(
        dash_export.ExportTable(pd.DataFrame({"a": [1]}), description="d"),
        source_csv=os.path.join(tmp_path, "gone.csv"),
        metric_column="Area (mm²)",
        view="time",
    )
    assert _header(text)["source_sha256"] == "unavailable"


def test_export_filename_is_descriptive_and_sortable():
    when = datetime(2026, 9, 28, 17, 45, 12)
    assert dash_export.export_filename("root_area", "lines", when) == (
        "root_area_lines_20260928_174512.csv"
    )


def test_the_export_compares_with_the_projects_calibration(tmp_path):
    """O3: the export compares the CSV's recorded field of view with its project's."""
    root = tmp_path / "project"
    (root / "skeletons").mkdir(parents=True)
    (root / "sprouts_project.json").write_text(
        '{"schema": 1, "calibration": {"fov_width_mm": 18.0, "fov_height_mm": 13.0}}',
        encoding="utf-8",
    )
    source = root / "skeletons" / "root_lengths.csv"
    source.write_text("Image,Tube\n", encoding="utf-8")
    provenance.write_sidecar(source, {"calibration": {"fov_width_mm": 18.0, "fov_height_mm": 13.0}})
    table = dash_export.ExportTable(pd.DataFrame({"a": [1]}), description="d")
    lines = dash_export.provenance_lines(
        table, source_csv=str(source), metric_column="Length (mm)", view="stacked"
    )
    (line,) = [line for line in lines if line.startswith("# calibration: ")]
    assert "18.00 x 13.00 mm, as recorded with root_lengths.csv" in line
    assert "this app now uses" not in line
    # Recorded at 13.5 (the default) in a 13.0 project: the note names the project's.
    provenance.write_sidecar(source, {"calibration": {"fov_width_mm": 18.0, "fov_height_mm": 13.5}})
    lines = dash_export.provenance_lines(
        table, source_csv=str(source), metric_column="Length (mm)", view="stacked"
    )
    (line,) = [line for line in lines if line.startswith("# calibration: ")]
    assert line.endswith("; this project uses 18.00 x 13.00 mm")


def test_the_export_takes_the_project_root_when_the_caller_knows_it(tmp_path):
    """A CSV outside the project's folders (or deeper than the walk) is compared with the
    project's field of view when the caller passes the project root (review I3)."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "sprouts_project.json").write_text(
        '{"schema": 1, "calibration": {"fov_width_mm": 18.0, "fov_height_mm": 13.0}}',
        encoding="utf-8",
    )
    source = tmp_path / "copied" / "root_lengths.csv"
    source.parent.mkdir()
    source.write_text("Image,Tube\n", encoding="utf-8")
    provenance.write_sidecar(source, {"calibration": {"fov_width_mm": 18.0, "fov_height_mm": 13.5}})
    table = dash_export.ExportTable(pd.DataFrame({"a": [1]}), description="d")
    text = dash_export.render_csv(
        table,
        source_csv=str(source),
        metric_column="Length (mm)",
        view="stacked",
        project_root=str(root),
    )
    assert _header(text)["calibration"].endswith("; this project uses 18.00 x 13.00 mm")
    # Without it, the CSV's own folders hold no project file: the app's default, the same.
    assert "uses" not in dash_export.calibration_line(str(source))


# ---- the time view's table is built in one pass -------------------------------------


def _growth_over_time_per_tube(cache, tubes, *, field_average):
    """The time view's table as it was built before: one frame per tube (a frozen copy)."""
    total = f"Total {cache.value_column}"
    sd = f"SD ({dash_export._unit(cache.value_column)})"
    columns = ["Series", "Date", total, sd, "Tubes"]
    parts = []
    if field_average:
        stats = cache.get_field_growth()
        parts.append(
            pd.DataFrame(
                {
                    "Series": dash_export.FIELD_AVERAGE_SERIES,
                    "Date": stats["Date"].dt.strftime(dash_export.DATE_FORMAT),
                    total: stats["mean"],
                    sd: stats["std"],
                    "Tubes": stats["count"].astype("Int64"),
                }
            )
        )
    for tube in tubes:
        totals = cache.get_tube_totals(tube)
        if totals.empty:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "Series": f"Tube {int(tube)}",
                    "Date": totals.index.strftime(dash_export.DATE_FORMAT),
                    total: totals.to_numpy(),
                    sd: float("nan"),
                    "Tubes": pd.array([pd.NA] * len(totals), dtype="Int64"),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)[columns]


def _gappy_rows():
    """50 tubes on 3 dates; some tubes skip a date, tube 7 is measured once."""
    rows = []
    for tube in range(1, 51):
        for d, date in enumerate(("2024.06.03", "2024.07.16", "2024.08.30")):
            if (tube + d) % 4 == 0 or (tube == 7 and d):
                continue
            for position in (5, 15):
                stem = f"T{tube:03d}_L{position:03d}_{date}"
                rows.append((stem, tube, position, date, round((tube * 3 + d + position) / 7, 3)))
    return rows


@pytest.mark.parametrize("field_average", [True, False])
@pytest.mark.parametrize("rows", ["fixture", "gappy"])
def test_the_time_table_is_the_per_tube_table(tmp_path, rows, field_average):
    from app.data_processing.data_processor import DataProcessor
    from app.visualization.dash_data_cache import DataCache
    from tests.viz_dash_helpers import LENGTH_COLUMN, ROWS, write_metric_csv

    rows = _gappy_rows() if rows == "gappy" else ROWS
    cache = DataCache(DataProcessor(write_metric_csv(tmp_path / "l.csv", LENGTH_COLUMN, rows)))
    tubes = [int(t) for t in cache.tubes]
    selections = [[], [999], [5], [7], [3, 1, 2], [2, 2], [1, 999, 2], [3, 1, 3, 2, 1],
                  tubes, tubes[::-1]]  # fmt: skip

    for selection in selections:
        got = dash_export.growth_over_time(cache, selection, field_average=field_average)
        want = _growth_over_time_per_tube(cache, selection, field_average=field_average)

        pd.testing.assert_frame_equal(got, want, check_exact=True)
        assert got.to_csv(index=False, lineterminator="\n") == want.to_csv(
            index=False, lineterminator="\n"
        )


# ---- the open project's field of view; the grid export -----------------------------------


def _project_file(folder, width=20.0, height=15.0) -> None:
    os.makedirs(folder, exist_ok=True)
    record = {"schema": 1, "calibration": {"fov_width_mm": width, "fov_height_mm": height}}
    with open(os.path.join(folder, "sprouts_project.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh)


def _length_app(csv: str, **kwargs):
    from app.data_processing.data_processor import DataProcessor
    from app.visualization.dash_app import DashApp

    return DashApp(DataProcessor(csv), os.path.dirname(csv), FakeImageManager(), **kwargs)


def _calibration(content: str) -> str:
    (line,) = [ln for ln in content.splitlines() if ln.startswith("# calibration: ")]
    return line.removeprefix("# calibration: ")


def _read_cells(content: str) -> pd.DataFrame:
    """The table as written: every cell as its text (an empty cell is "")."""
    return pd.read_csv(io.StringIO(content), comment="#", dtype=str, keep_default_na=False)


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

    table = _read_cells(download["content"])
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


def test_the_grid_of_a_csv_with_an_infinite_value_is_drawn_and_exported(tmp_path):
    rows = (
        ("T001_L005_2024.07.18", 1, 5, "2024.07.18", 2.5),
        ("T002_L007_2024.07.18", 2, 7, "2024.07.18", float("inf")),
    )
    csv = write_metric_csv(str(tmp_path / "root_lengths.csv"), LENGTH_COLUMN, rows)
    dashboard = _length_app(csv)

    figure = dashboard.create_faceted_depth_profile([1])
    assert not figure.layout.annotations or "Error" not in figure.layout.annotations[0].text
    table = _read_cells(
        _export(
            dashboard,
            {"view-selector.value": "faceted", "faceted-tubes-store.data": [1],
             "faceted-dates-store.data": []},
        )["content"]
    )  # fmt: skip
    # Tube 2's row is left out, so the field mean of L1-10 is tube 1's alone.
    assert list(table["Field Mean Length (mm)"]) == ["2.5"]
    assert list(table["Field Count"]) == ["1"]


@pytest.mark.parametrize("metric", METRICS)
def test_a_csv_of_one_image_draws_every_view_and_exports_one_row_each(tmp_path, metric):
    from app.data_processing.data_processor import DataProcessor
    from app.data_processing.data_processor_area import DataProcessorArea
    from app.visualization.dash_app import DashApp
    from app.visualization.dash_app_area import DashAppArea

    column = LENGTH_COLUMN if metric == "length" else AREA_COLUMN
    processor, cls = (
        (DataProcessor, DashApp) if metric == "length" else (DataProcessorArea, DashAppArea)
    )
    rows = (("T001_L005_2024.07.18", 1, 5, "2024.07.18", 2.5),)
    csv = write_metric_csv(str(tmp_path / "metric.csv"), column, rows)
    dashboard = cls(processor(csv), str(tmp_path), FakeImageManager())
    views = {
        "stacked": {},
        "time": {"selected-tubes-store.data": [1],
                 "growth-viz-options.value": ["show_field_avg", "show_field_var"]},
        "lines": {"tube-selector.value": 1},
        "faceted": {"faceted-tubes-store.data": [1], "faceted-dates-store.data": []},
    }  # fmt: skip

    for view, values in views.items():
        _, out = call_callback(
            dashboard.app, "main-graph.figure", {"view-selector.value": view, **values},
            ["view-selector.value"],
        )  # fmt: skip
        figure = out["main-graph"]["figure"]
        assert figure["data"], view
        assert "Error" not in json.dumps(figure["layout"].get("annotations", [])), view
        table = _read_cells(_export(dashboard, {"view-selector.value": view, **values})["content"])
        sd = [c for c in table.columns if c.startswith(("SD", "Field SD"))]
        if view == "time":  # the field average of one tube, and the tube itself
            assert table["Series"].tolist() == ["Field average", "Tube 1"]
            assert table["Tubes"].tolist() == ["1", ""]
        else:
            assert len(table) == 1, view
        assert all(cell == "" for c in sd for cell in table[c]), view
        assert "2.5" in table.to_numpy().tolist()[-1], view
