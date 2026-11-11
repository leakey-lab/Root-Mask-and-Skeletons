"""viz-data: the Growth Over Time figures (R-VIZ X3, X4), length and area.

X4: the time views are assembled as plain dicts and handed to ``go.Figure`` without
validation. Building one ``go.Scatter`` per tube and validating everything again in
``go.Figure(data=traces)`` took 80 ms for 160 tubes (8 ms now; area 57 -> 6 ms), on
the server thread and holding the GIL. The browser must get the same figure: the
Dash JSON of each view equals what a validating ``go.Figure`` makes of the same dicts,
and its traces carry the cache's field mean, +-1 SD band and per-tube totals.

X3: above :data:`AUTO_LEGEND_MAX_TUBES` tubes the length time view starts without its
legend (a 160-entry legend was ~200 ms of a ~670 ms draw, and ten colours cannot
tell 160 tubes apart; the hover names each one). "Toggle Legend" still shows it,
and an explicit choice is kept.
"""

from __future__ import annotations

import json

import plotly.graph_objects as go
import pytest
from dash._utils import to_json

from app.config import AREA_CSV_HEADERS, LENGTH_CSV_HEADERS
from app.data_processing.data_processor import DataProcessor
from app.data_processing.data_processor_area import DataProcessorArea
from app.visualization import theme
from app.visualization.dash_app_area import DashAppArea
from app.visualization.dash_visualizations import AUTO_LEGEND_MAX_TUBES
from tests.viz_dash_helpers import build_length_app, call_callback
from tests.viz_data_support import Charts as _Charts
from tests.viz_data_support import write_metric_csv

DATES = ("2024.06.03", "2024.06.04", "2024.06.24", "2024.07.16")


def _rows(tubes: int):
    rows = []
    for tube in range(1, tubes + 1):
        for d, date in enumerate(DATES):
            if (tube + d) % 5 == 0:
                continue  # some tubes skip a date
            for position in (5, 15, 27):
                rows.append(
                    (tube, position, date, round(((tube * 7 + d * 3 + position) % 23) / 2, 2))
                )
    return rows


@pytest.fixture
def charts(tmp_path):
    csv = write_metric_csv(tmp_path / "l.csv", LENGTH_CSV_HEADERS, _rows(40))
    return _Charts(DataProcessor(csv))


@pytest.fixture
def area(tmp_path):
    csv = write_metric_csv(tmp_path / "a.csv", AREA_CSV_HEADERS, _rows(40))
    return DashAppArea(DataProcessorArea(csv), str(tmp_path))


def _json(fig) -> dict:
    """What the browser receives (Dash's serializer), parsed."""
    return json.loads(to_json(fig))


# ---- X4: the dict-built figure -----------------------------------------------------


def _dashboard(tmp_path, metric):
    """A fresh dashboard of ``metric`` over the 40-tube rows (its own, empty figure memo)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    if metric == "length":
        csv = write_metric_csv(tmp_path / "l.csv", LENGTH_CSV_HEADERS, _rows(40))
        return _Charts(DataProcessor(csv))
    csv = write_metric_csv(tmp_path / "a.csv", AREA_CSV_HEADERS, _rows(40))
    return DashAppArea(DataProcessorArea(csv), str(tmp_path))


TIME_VIEWS = [
    ("length", (1, 2, 3), True, True, True),
    ("length", (1, 2, 3), True, False, False),
    ("length", (), True, True, True),
    ("length", (4, 5), False, False, True),
    ("length", tuple(range(1, 41)), True, True, False),
    ("length", (99,), True, True, True),  # a tube without data draws nothing for it
    ("area", (1, 2, 3), True, True, True),
    ("area", (), True, True, True),
    ("area", (4, 5), False, False, True),
    ("area", tuple(range(1, 41)), True, True, False),
]


@pytest.mark.parametrize("metric, tubes, field, variance, legend", TIME_VIEWS)
def test_the_time_figure_is_what_validation_makes_of_its_dicts(
    tmp_path, monkeypatch, metric, tubes, field, variance, legend
):
    # Built from dicts with validation off (go.Figure(..., _validate=False)): the browser
    # must get what a validating go.Figure makes of the same dicts (a misspelt property
    # raises there), as for the Depth Profile Grid.
    built = _json(
        _dashboard(tmp_path / "a", metric).show_growth_over_time(tubes, field, variance, legend)
    )
    init = go.Figure.__init__

    def validating(self, *args, **kwargs):
        kwargs.pop("_validate", None)
        init(self, *args, **kwargs)

    monkeypatch.setattr(go.Figure, "__init__", validating)
    validated = _json(
        _dashboard(tmp_path / "b", metric).show_growth_over_time(tubes, field, variance, legend)
    )

    assert built == validated


@pytest.mark.parametrize(
    "metric, label, unit", [("length", "Length", "mm"), ("area", "Area", "mm²")]
)
def test_the_time_figure_draws_the_cache_values(tmp_path, metric, label, unit):
    dashboard = _dashboard(tmp_path, metric)
    cache = dashboard.data_cache
    field = cache.get_field_growth()
    mean, sd = field["mean"].to_numpy(), field["std"].to_numpy()

    figure = dashboard.show_growth_over_time([3, 1], True, True, True)
    upper, lower, average, *tubes = figure.data

    assert list(upper.y) == list(mean + sd) and upper.showlegend is False
    assert list(lower.y) == list(mean - sd) and lower.fill == "tonexty"
    assert lower.name == "Field Mean ±1 SD"
    assert list(average.y) == list(mean) and average.name == "Field Average"
    assert [list(row) for row in average.customdata] == [
        [s, c] for s, c in zip(sd, field["count"], strict=True)
    ]
    assert [t.name for t in tubes] == ["Tube 3", "Tube 1"]  # the selection's order
    for trace, tube in zip(tubes, (3, 1), strict=True):
        totals = cache.get_tube_totals(tube)
        assert list(trace.y) == list(totals.to_numpy())
        assert list(trace.x) == list(totals.index.to_numpy())
    assert figure.layout.yaxis.title.text == f"Total {label} ({unit})"
    # Without the band, the field average comes first.
    plain = dashboard.show_growth_over_time([3, 1], True, False, True)
    assert [t.name for t in plain.data] == ["Field Average", "Tube 3", "Tube 1"]


def test_the_time_figures_are_plotly_figures_the_memo_and_callers_can_use(charts, area):
    # Built from dicts without validation (go.Figure(..., _validate=False)): the tests
    # above compare the JSON with the validated object-built figures, so a misspelt
    # property fails them.
    for fig in (
        charts._growth_over_time_figure((1, 2), True, True, True),
        area.show_growth_over_time(),
    ):
        assert isinstance(fig, go.Figure)
        assert {trace.type for trace in fig.data} == {"scatter"}
        assert fig.layout.template.layout.paper_bgcolor == theme.BG_0  # the SPROUTS theme


# ---- X3: the legend of many tubes -------------------------------------------------


def test_the_legend_is_shown_for_a_few_tubes_and_hidden_for_many(charts):
    few = tuple(range(1, AUTO_LEGEND_MAX_TUBES + 1))
    many = tuple(range(1, AUTO_LEGEND_MAX_TUBES + 2))

    assert charts.show_growth_over_time(few, True, True, None).layout.showlegend is True
    assert charts.show_growth_over_time(many, True, True, None).layout.showlegend is False
    assert AUTO_LEGEND_MAX_TUBES == 30


def test_a_hidden_legend_leaves_no_empty_band_beside_the_plot(charts):
    many = tuple(range(1, 41))

    assert charts.show_growth_over_time(many, True, True, None).layout.margin.r == 24
    assert charts.show_growth_over_time(many, True, True, True).layout.margin.r == 180


@pytest.mark.parametrize("choice", [True, False])
def test_an_explicit_legend_choice_is_kept(charts, choice):
    for tubes in ((1, 2), tuple(range(1, 41))):
        assert charts.show_growth_over_time(tubes, True, True, choice).layout.showlegend is choice


def test_the_legend_starts_automatic_on_the_page(tmp_path):
    from tests.viz_dash_helpers import walk_components

    layout = build_length_app(tmp_path).app.layout
    store = next(
        c for c in walk_components(layout) if getattr(c, "id", None) == "legend-visible-store"
    )
    assert store.data is None  # automatic: shown up to AUTO_LEGEND_MAX_TUBES tubes


@pytest.mark.parametrize(
    "selected, current, expected",
    [
        (list(range(1, 41)), None, True),  # hidden automatically: the toggle shows it
        ([1, 2], None, False),  # shown automatically: the toggle hides it
        (list(range(1, 41)), True, False),
        ([1, 2], False, True),
    ],
)
def test_toggle_legend_flips_what_is_shown(tmp_path, selected, current, expected):
    dash = build_length_app(tmp_path).app
    _, out = call_callback(
        dash,
        "legend-visible-store.data",
        {"toggle-legend-btn.n_clicks": 1, "legend-visible-store.data": current,
         "selected-tubes-store.data": selected},
        changed=["toggle-legend-btn.n_clicks"],
    )  # fmt: skip
    assert out["legend-visible-store"]["data"] is expected


def test_the_page_draws_the_time_view_with_the_automatic_legend(tmp_path):
    dash = build_length_app(tmp_path).app
    _, out = call_callback(
        dash,
        "main-graph.figure",
        {"view-selector.value": "time", "selected-tubes-store.data": [1, 2],
         "growth-viz-options.value": ["show_field_avg"], "legend-visible-store.data": None},
        changed=["view-selector.value"],
    )  # fmt: skip
    assert out["main-graph"]["figure"]["layout"]["showlegend"] is True


# ---- the area dashboard shares the legend rule and the toggle ----------------------


def test_the_area_legend_follows_the_same_automatic_rule(area):
    # The area app is the length dashboard with the area metric: the same 30-tube rule
    # and the same explicit choices (these cases stood in the deleted area-only legend test).
    few = list(range(1, AUTO_LEGEND_MAX_TUBES + 1))
    many = list(range(1, AUTO_LEGEND_MAX_TUBES + 2))

    assert area.show_growth_over_time(few, show_legend=None).layout.showlegend is True
    hidden = area.show_growth_over_time(many, show_legend=None).layout
    assert hidden.showlegend is False
    assert hidden.margin.r == theme.MARGIN_RIGHT_NO_LEGEND
    assert area.show_growth_over_time(many, show_legend=True).layout.showlegend is True
    assert area.show_growth_over_time(few, show_legend=False).layout.showlegend is False


def test_the_area_toggle_button_flips_the_stored_legend_choice(area):
    toggle = "toggle-legend-btn.n_clicks"
    store = "legend-visible-store.data"
    for clicks, choice, expected in ((1, None, False), (2, False, True)):
        _, out = call_callback(area.app, store, {toggle: clicks, store: choice}, [toggle])
        assert out["legend-visible-store"]["data"] is expected  # no tube selected: shown at first
