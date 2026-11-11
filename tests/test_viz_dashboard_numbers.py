"""Every number a dashboard view draws and exports equals the CSV's, summed and averaged as the
page says (``app/visualization/dash_export.py``: "the same DataCache aggregates the chart is
drawn from").

A differential test: seeded random metric CSVs (``numpy.random.default_rng``; 1-12 tubes, 1-7
dates at random spacing, so some fall within the 3-day merge window and some just outside it;
gaps in depth, so intervals are missing between measured ones), each with rows the dashboards
must leave out (a failed measurement, a name that cannot be read, an infinite value). The oracle
is ``tests/viz_reference.py``: plain Python, no pandas, no numpy, no app code. For each view the
export (through Dash's dispatch) is read as the numbers it prints, and the figure the view draws
is compared with the same reference.

Row sets, dates, tubes and counts are compared exactly. Values are compared with a relative
tolerance of 1e-9 (absolute 1e-9 near zero): the app sums with Kahan compensation in row order,
the reference with ``math.fsum``, which can differ in the last bits; a wrong sum, mean or SD
differs by far more. Every assertion message names the seed. Positions are distinct per tube
and date: whether a re-imaged frame counts once or twice is the owner's open question.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import math

import numpy as np
import pytest

from tests import viz_reference as ref
from tests.viz_dash_helpers import (
    AREA_COLUMN,
    LENGTH_COLUMN,
    FakeImageManager,
    call_callback,
)

SEEDS = range(8)
METRICS = {"length": LENGTH_COLUMN, "area": AREA_COLUMN}
REL = ABS = 1e-9


def _close(got, want) -> bool:
    if want is None:
        return got is None
    return got is not None and math.isclose(got, want, rel_tol=REL, abs_tol=ABS)


def _write_random_csv(path, column: str, seed: int) -> str:
    """A metric CSV of ``seed``: measured rows plus the rows that must not count."""
    rng = np.random.default_rng(seed)
    first = dt.date(2024, 6, 3)
    gaps = [1, 2, 3, 4, 5, 8, 14]
    dates = [first]
    for _ in range(int(rng.integers(0, 7))):
        dates.append(dates[-1] + dt.timedelta(days=int(rng.choice(gaps))))
    lines = [f"Image,Tube,Position,Date,Time,{column},Error"]
    for tube in range(1, int(rng.integers(1, 13)) + 1):
        for date in dates:
            if rng.random() < 0.25:
                continue  # this tube was not imaged that day
            day = date.strftime("%Y.%m.%d")
            count = int(rng.integers(1, 9))
            for position in sorted(rng.choice(np.arange(1, 46), count, replace=False)):
                value = 0.0 if rng.random() < 0.1 else round(float(rng.uniform(0, 60)), 2)
                name = f"CAM_T{tube:03d}_L{int(position):03d}_{day}_120000_001"
                lines.append(f"{name},{tube},{int(position)},{day},120000,{value},")
    day = dates[0].strftime("%Y.%m.%d")
    lines += [
        f"CAM_T001_L050_{day}_120000,,,,,0,cannot identify image file",  # failed
        "IMG_0001,,,,,0,",  # a name the parser cannot read
        f"CAM_T001_L051_{day}_120000_001,1,51,{day},120000,inf,",  # an infinite value
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


def _dashboard(tmp_path, metric: str, seed: int):
    from app.data_processing.data_processor import DataProcessor
    from app.data_processing.data_processor_area import DataProcessorArea
    from app.visualization.dash_app import DashApp
    from app.visualization.dash_app_area import DashAppArea

    folder = tmp_path / f"{metric}-{seed}"
    folder.mkdir()
    column = METRICS[metric]
    path = _write_random_csv(folder / "metric.csv", column, seed)
    processor, app = (
        (DataProcessor, DashApp) if metric == "length" else (DataProcessorArea, DashAppArea)
    )
    dashboard = app(processor(path), str(folder), FakeImageManager())
    return dashboard, ref.charted_rows(path, column)


def _export(dashboard, values: dict) -> list[dict]:
    """The exported table as the numbers it prints: a cell is a float, or None when empty."""
    status, out = call_callback(
        dashboard.app,
        "viz-export-download.data",
        {"viz-export-btn.n_clicks": 1, **values},
        changed=["viz-export-btn.n_clicks"],
    )
    assert status == 200
    content = out["viz-export-download"]["data"]["content"]
    body = "\n".join(line for line in content.splitlines() if not line.startswith("#"))
    rows = []
    for row in csv.DictReader(io.StringIO(body)):
        cells = {}
        for key, cell in row.items():
            if key in ("Date", "Series"):
                cells[key] = cell
            else:
                cells[key] = float(cell) if cell != "" else None
        rows.append(cells)
    return rows


def _day(date: dt.date) -> str:
    return date.strftime("%Y-%m-%d")


def _unit(metric: str) -> str:
    return "mm" if metric == "length" else "mm²"


@pytest.mark.parametrize("metric", METRICS)
def test_totals_by_tube_equal_the_csvs_sums(tmp_path, metric):
    column = METRICS[metric]
    for seed in SEEDS:
        dashboard, rows = _dashboard(tmp_path, metric, seed)
        want = ref.totals(rows)

        table = _export(dashboard, {"view-selector.value": "stacked"})
        got = {(int(r["Tube"]), r["Date"]): r[f"Total {column}"] for r in table}
        assert set(got) == {(t, _day(d)) for t, d in want}, seed
        for (tube, date), total in want.items():
            assert _close(got[tube, _day(date)], total), (seed, tube, date)

        figure = dashboard.create_stacked_bar_chart()
        for trace in figure.data:
            for x, y in zip(trace.x, trace.y, strict=True):
                key = (int(x.removeprefix("Tube ")), dt.date.fromisoformat(trace.name))
                missing = y is None or math.isnan(y)  # NaN reaches the browser as null: no bar
                assert _close(None if missing else y, want.get(key)), (seed, key, y)


@pytest.mark.parametrize("metric", METRICS)
def test_growth_over_time_equals_the_merged_field_average_and_the_tube_totals(tmp_path, metric):
    column, unit = METRICS[metric], _unit(metric)
    for seed in SEEDS:
        dashboard, rows = _dashboard(tmp_path, metric, seed)
        field, totals = ref.field_growth(rows), ref.totals(rows)
        tubes = sorted({tube for tube, _date in totals})

        table = _export(
            dashboard,
            {"view-selector.value": "time", "selected-tubes-store.data": tubes,
             "growth-viz-options.value": ["show_field_avg", "show_field_var"]},
        )  # fmt: skip
        average = [r for r in table if r["Series"] == "Field average"]
        assert [r["Date"] for r in average] == [_day(d) for d in sorted(field)], seed
        for r, date in zip(average, sorted(field), strict=True):
            assert _close(r[f"Total {column}"], field[date]["mean"]), (seed, date)
            assert _close(r[f"SD ({unit})"], field[date]["sd"]), (seed, date)
            assert r["Tubes"] == field[date]["tubes"], (seed, date)
        per_tube = [(r["Series"], r["Date"], r[f"Total {column}"]) for r in table[len(average) :]]
        want = [
            (f"Tube {t}", _day(d), totals[t, d])
            for t in tubes
            for d in sorted(d for tube, d in totals if tube == t)
        ]
        assert [(s, d) for s, d, _ in per_tube] == [(s, d) for s, d, _ in want], seed
        assert all(_close(g, w) for (_, _, g), (_, _, w) in zip(per_tube, want, strict=True)), seed

        figure = dashboard.show_growth_over_time(tubes, True, False, True)
        traces = {trace.name: trace for trace in figure.data}
        means = [field[d]["mean"] for d in sorted(field)]
        assert all(_close(g, w) for g, w in zip(traces["Field Average"].y, means, strict=True))
        for t in tubes:
            want_y = [totals[t, d] for d in sorted(d for tube, d in totals if tube == t)]
            got_y = list(traces[f"Tube {t}"].y)
            assert len(got_y) == len(want_y), (seed, t)
            assert all(_close(g, w) for g, w in zip(got_y, want_y, strict=True)), (seed, t)


@pytest.mark.parametrize("metric", METRICS)
def test_depth_profile_equals_the_interval_statistics(tmp_path, metric):
    column, unit = METRICS[metric], _unit(metric)
    for seed in SEEDS:
        dashboard, rows = _dashboard(tmp_path, metric, seed)
        want = ref.intervals(rows)
        for tube in sorted({t for t, _d, _e in want}):
            mine = {(d, e): s for (t, d, e), s in want.items() if t == tube}

            table = _export(
                dashboard, {"view-selector.value": "lines", "tube-selector.value": tube}
            )
            got = {(r["Date"], int(r["Interval end"])): r for r in table}
            assert set(got) == {(_day(d), e) for d, e in mine}, (seed, tube)  # measured only
            for (date, end), s in mine.items():
                r = got[_day(date), end]
                assert int(r["Tube"]) == tube and int(r["Interval start"]) == end - 9
                assert r["Count"] == s["count"], (seed, tube, date, end)
                for key, name in (("mean", "Mean"), ("min", "Min"), ("max", "Max")):
                    assert _close(r[f"{name} {column}"], s[key]), (seed, tube, date, end, key)
                assert _close(r[f"SD ({unit})"], s["sd"]), (seed, tube, date, end)

            figure = dashboard.show_growth_lines(tube)
            for trace in figure.data:
                date = dt.date.fromisoformat(trace.name)
                drawn = {
                    int(e): x for e, x in zip(trace.y, trace.x, strict=True) if not math.isnan(x)
                }
                measured = {e: s["mean"] for (d, e), s in mine.items() if d == date}
                assert set(drawn) == set(measured), (seed, tube, date)
                assert all(_close(drawn[e], m) for e, m in measured.items()), (seed, tube, date)


@pytest.mark.parametrize("metric", METRICS)
def test_the_depth_profile_grid_equals_tube_and_field_interval_means(tmp_path, metric):
    column, unit = METRICS[metric], _unit(metric)
    for seed in SEEDS:
        dashboard, rows = _dashboard(tmp_path, metric, seed)
        tubes_of_csv = sorted({t for t, _p, _d, _v in rows})
        tubes = tubes_of_csv[:6]
        mine, field = ref.intervals(rows), ref.field_intervals(rows)
        want = {(t, d, e): (s, field[d, e]) for (t, d, e), s in mine.items() if t in tubes}

        table = _export(
            dashboard,
            {"view-selector.value": "faceted", "faceted-tubes-store.data": tubes,
             "faceted-dates-store.data": []},
        )  # fmt: skip
        got = {(int(r["Tube"]), r["Date"], int(r["Interval end"])): r for r in table}
        assert set(got) == {(t, _day(d), e) for t, d, e in want}, seed
        for (tube, date, end), (s, f) in want.items():
            r = got[tube, _day(date), end]
            assert r["Count"] == s["count"] and r["Field Count"] == f["count"], (seed, tube)
            assert _close(r[f"Mean {column}"], s["mean"]), (seed, tube, date, end)
            assert _close(r[f"Field Mean {column}"], f["mean"]), (seed, tube, date, end)
            assert _close(r[f"Field SD ({unit})"], f["sd"]), (seed, tube, date, end)

        figure = dashboard.create_faceted_depth_profile(tubes)
        data = list(figure.data)
        drawn = 0
        for field_bars, tube_bars in zip(data, data[1:], strict=False):
            if field_bars.name != "Field Avg":
                continue
            title = tube_bars.hovertemplate.split("</b>")[0].removeprefix("<b>Tube ")
            tube, day = title.split(" - ")
            date = dt.date.fromisoformat(day)
            for end, x, fx in zip(tube_bars.y, tube_bars.x, field_bars.x, strict=True):
                s, f = want[int(tube), date, int(end)]
                assert _close(x, s["mean"]) and _close(-fx, f["mean"]), (seed, tube, date, end)
                drawn += 1
        assert drawn == len(want), seed
