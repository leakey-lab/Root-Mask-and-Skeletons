"""AR-15 / PV-05 / PV-15 / PV-16: the shared dashboard shell.

Both dashboards are built from ``app.visualization.dash_shell``. These tests go
through Dash's real dispatch (Flask test client) and check what the browser gets:
the served layout, the graph config and the shell callbacks' responses.
"""

from __future__ import annotations

import json

import pytest

from tests.viz_dash_helpers import (
    NO_UPDATE,
    build_area_app,
    build_length_app,
    call_callback,
    call_clientside,
    walk_components,
)

# Every id a callback names is on the served page: tests/test_viz_dash_wiring.py.


@pytest.fixture
def length_app(tmp_path):
    return build_length_app(tmp_path)


def _served_layout(dash) -> dict:
    resp = dash.server.test_client().get("/_dash-layout")
    assert resp.status_code == 200
    return json.loads(resp.data)


def _find(node, predicate):
    if isinstance(node, dict):
        if predicate(node):
            return node
        for value in node.values():
            found = _find(value, predicate)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find(value, predicate)
            if found is not None:
                return found
    return None


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_graph_config_offers_no_dead_modebar_buttons(tmp_path, builder):
    """PV-16: 'downloadCsv' is not a Plotly.js button (it rendered nothing) and
    box / lasso select had no callbacks behind them."""
    graph = _find(
        _served_layout(builder(tmp_path).app),
        lambda n: n.get("props", {}).get("id") == "main-graph",
    )
    config = graph["props"]["config"]
    assert "downloadCsv" not in json.dumps(config)
    assert {"select2d", "lasso2d"} <= set(config["modeBarButtonsToRemove"])
    assert config["displaylogo"] is False
    assert config["toImageButtonOptions"]["format"] == "svg"


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_graph_config_keeps_the_plotly_js_3_defaults(tmp_path, builder):
    """plotly.js 4 shows an "Upload to Cloud" button (it posts the chart to
    cloud.plotly.com) and waits 500 ms for a double-click: both are set back."""
    graph = _find(
        _served_layout(builder(tmp_path).app),
        lambda n: n.get("props", {}).get("id") == "main-graph",
    )
    config = graph["props"]["config"]
    assert config["showSendToCloud"] is False
    assert config["doubleClickDelay"] == 300


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_chart_loading_wrapper_is_a_flex_container(tmp_path, builder):
    """PV-05: dcc.Loading's two wrapper divs must carry the class the stylesheet
    stretches (``viz-loading``), otherwise the chart collapses to Plotly's 450 px."""
    layout = builder(tmp_path).app.layout
    loading = next(
        c
        for c in walk_components(layout)
        if type(c).__name__ == "Loading"
        and "main-graph" in {getattr(x, "id", None) for x in walk_components(c.children)}
    )
    assert loading.parent_className == "viz-loading"
    assert loading.target_components == {"main-graph": "figure"}


# ---- PV-15: the control strip follows the selected view ------------------------
# The strip callbacks run in the browser (R-VIZ X2): as server callbacks their
# responses raced the new figure's, which was then drawn at the old size and again
# at the new one. They are run here in Qt's JavaScript engine.


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
@pytest.mark.parametrize("was_open", [True, False])
@pytest.mark.parametrize(
    "view, expected",
    [("time", True), ("lines", True), ("faceted", True), ("stacked", False)],
)
def test_switching_view_opens_the_strip_only_when_the_view_has_controls(
    tmp_path, builder, view, expected, was_open
):
    out, _ = call_clientside(
        builder(tmp_path).app,
        "viz-controls-collapse.is_open",
        {"view-selector.value": view, "viz-controls-collapse.is_open": was_open},
        changed=["view-selector.value"],
    )
    assert out["viz-controls-collapse"]["is_open"] is expected


@pytest.mark.parametrize("was_open", [True, False])
def test_toggle_button_still_flips_the_strip(length_app, was_open):
    out, _ = call_clientside(
        length_app.app,
        "viz-controls-collapse.is_open",
        {"viz-controls-toggle.n_clicks": 1, "view-selector.value": "time",
         "viz-controls-collapse.is_open": was_open},
        changed=["viz-controls-toggle.n_clicks"],
    )  # fmt: skip
    assert out["viz-controls-collapse"]["is_open"] is (not was_open)


@pytest.mark.parametrize("is_open, caret", [(True, "▾"), (False, "▸")])
def test_caret_reflects_the_strip_state(length_app, is_open, caret):
    out, _ = call_clientside(
        length_app.app,
        "viz-controls-caret.children",
        {"viz-controls-collapse.is_open": is_open},
        changed=["viz-controls-collapse.is_open"],
    )
    assert out["viz-controls-caret"]["children"] == caret


# ---- W19: a view without settings shows no strip at all -------------------------


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_views_without_settings_hide_the_whole_strip(tmp_path, builder):
    # Before, the stacked view showed "▸ Tube & date range ... No options for this
    # view", a disclosure that opened onto an empty strip.
    dash = builder(tmp_path).app
    views = {"stacked": False, "time": True, "lines": True, "faceted": True}
    for view, shown in views.items():
        out, _ = call_clientside(
            dash, "viz-controls.style", {"view-selector.value": view}, ["view-selector.value"]
        )
        assert (out["viz-controls"]["style"].get("display") != "none") is shown, view


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_the_default_view_starts_without_an_empty_strip(tmp_path, builder):
    strip = _find(
        _served_layout(builder(tmp_path).app),
        lambda n: n.get("props", {}).get("id") == "viz-controls",
    )
    assert strip["props"]["style"] == {"display": "none"}  # the default view: stacked bars


# ---- X2: the new figure is drawn once, at the size the strip left it --------------

_FAKE_GRAPH = """
window.__resized = 0;
window.Plotly = {Plots: {resize: function (gd) { window.__resized += 1; }}};
window.__handlers = {};
window.__gd = {
    _fullLayout: {_initialAutoSizeIsDone: true},
    on: function (event, handler) {
        (window.__handlers[event] = window.__handlers[event] || []).push(handler);
    }
};
document.querySelector = function (selector) {
    return selector === '#main-graph .js-plotly-plot' ? window.__gd : null;
};
"""


@pytest.mark.parametrize("changed", ["view-selector.value", "viz-controls-collapse.is_open"])
def test_a_strip_or_view_change_makes_the_next_draw_measure_the_chart_again(length_app, changed):
    # Plotly.react keeps the size a plot autosized to once, so a figure arriving after
    # the strip changed height was drawn at the old size, then again at the new one.
    out, engine = call_clientside(
        length_app.app,
        "viz-resize-dummy.data",
        {"view-selector.value": "time", "viz-controls-collapse.is_open": True},
        [changed],
        setup=_FAKE_GRAPH,
    )
    assert out["viz-resize-dummy"]["data"] == NO_UPDATE
    assert engine.evaluate("window.__gd._fullLayout._initialAutoSizeIsDone").toBool() is False
    assert engine.evaluate("window.__resized").toInt() == 1


def test_every_later_draw_leaves_the_next_one_to_measure_the_chart(length_app):
    # E.g. "Select All" grows the chip list (a server response of its own) before the
    # new figure arrives: that figure must also be drawn at the new size, once.
    _, engine = call_clientside(
        length_app.app,
        "viz-resize-dummy.data",
        {"view-selector.value": "time", "viz-controls-collapse.is_open": True},
        ["view-selector.value"],
        setup=_FAKE_GRAPH,
    )
    assert engine.evaluate("window.__handlers.plotly_afterplot.length").toInt() == 1
    engine.evaluate(
        "window.__gd._fullLayout._initialAutoSizeIsDone = true;"
        "window.__handlers.plotly_afterplot[0]();"
    )
    assert engine.evaluate("window.__gd._fullLayout._initialAutoSizeIsDone").toBool() is False


def test_the_draw_listener_is_added_once(length_app):
    _, engine = call_clientside(
        length_app.app,
        "viz-resize-dummy.data",
        {"view-selector.value": "time", "viz-controls-collapse.is_open": True},
        ["view-selector.value"],
        setup=_FAKE_GRAPH + "window.__gd.__sproutsRemeasure = true;",
    )
    assert engine.evaluate("window.__handlers.plotly_afterplot === undefined").toBool()


def test_the_refit_does_nothing_before_the_first_chart(length_app):
    out, engine = call_clientside(
        length_app.app,
        "viz-resize-dummy.data",
        {"view-selector.value": "time", "viz-controls-collapse.is_open": True},
        ["view-selector.value"],
    )
    assert out["viz-resize-dummy"]["data"] == NO_UPDATE


@pytest.mark.parametrize(
    "values, expected",
    [
        ({"view-selector.value": "stacked"}, "No options for this view"),
        ({"view-selector.value": "lines", "tube-selector.value": 2}, "Tube 2"),
        ({"view-selector.value": "time", "selected-tubes-store.data": []}, "No tubes selected"),
        ({"view-selector.value": "time", "selected-tubes-store.data": [1, 3]}, "2 tubes selected"),
        ({"view-selector.value": "time", "selected-tubes-store.data": [3]}, "1 tube selected"),
        (
            {"view-selector.value": "faceted", "faceted-tubes-store.data": [1, 2, 3],
             "faceted-dates-store.data": []},
            "3/6 tubes · all dates",
        ),
        (
            {"view-selector.value": "faceted", "faceted-tubes-store.data": [1],
             "faceted-dates-store.data": ["2024-07-18T00:00:00"]},
            "1/6 tubes · 1 date",
        ),
    ],
)  # fmt: skip
@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_summary_describes_the_active_selection(tmp_path, builder, values, expected):
    changed = [k for k in values]
    dash = builder(tmp_path).app
    _, out = call_callback(dash, "viz-controls-summary.children", values, changed)
    assert out["viz-controls-summary"]["children"] == expected


def _select_plotted_tube(dash, view, current, options=None):
    """The per-tube selection as the browser runs it on a view switch."""
    if options is None:  # what the page holds: the served dropdown's options
        options = _find(
            _served_layout(dash), lambda n: n.get("props", {}).get("id") == "tube-selector"
        )["props"]["options"]
    out, _ = call_clientside(
        dash,
        "tube-selector.value",
        {"view-selector.value": view, "tube-selector.value": current,
         "tube-selector.options": options},
        changed=["view-selector.value"],
    )  # fmt: skip
    return out["tube-selector"]["value"]


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
@pytest.mark.parametrize("current", [None, 99])
def test_entering_a_per_tube_view_selects_the_plotted_tube(tmp_path, builder, current):
    """The chart shows the first tube; the dropdown must say so (it showed its
    placeholder while the chart already plotted Tube 1)."""
    assert _select_plotted_tube(builder(tmp_path).app, "lines", current) == 1


def test_a_valid_tube_choice_is_kept_when_returning_to_lines(length_app):
    # no update: the user's tube stays selected
    assert _select_plotted_tube(length_app.app, "lines", 2) == NO_UPDATE


@pytest.mark.parametrize("view", ["lines", "time", "faceted", "stacked"])
@pytest.mark.parametrize("current", [None, 1, 2, 3, 99, "2", 2.0])
def test_the_dropdown_names_the_tube_the_chart_plots(length_app, view, current):
    # The figure callback plots plotted_tube(tubes, dropdown value): the browser's choice
    # is that tube, sent only when it differs (what the server callback answered).
    from app.visualization.dash_shell import plotted_tube

    tube = plotted_tube(length_app.data_cache.tubes, current)
    expected = tube if view == "lines" and tube != current else NO_UPDATE
    assert _select_plotted_tube(length_app.app, view, current) == expected


@pytest.mark.parametrize("options", [[], None])
def test_without_tubes_the_dropdown_is_left_alone(length_app, options):
    out, _ = call_clientside(
        length_app.app,
        "tube-selector.value",
        {"view-selector.value": "lines", "tube-selector.value": None,
         "tube-selector.options": options},
        changed=["view-selector.value"],
    )  # fmt: skip
    assert out["tube-selector"]["value"] == NO_UPDATE


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_a_view_that_cannot_be_drawn_shows_why_in_either_dashboard(tmp_path, builder):
    from app.visualization import theme

    dashboard = builder(tmp_path)

    def broken():
        raise RuntimeError("no totals")

    dashboard.create_stacked_bar_chart = broken
    _, out = call_callback(
        dashboard.app,
        "main-graph.figure",
        {"view-selector.value": "stacked"},
        changed=["view-selector.value"],
    )

    (note,) = out["main-graph"]["figure"]["layout"]["annotations"]
    assert note["text"] == "Error generating visualization: no totals"
    assert note["font"] == {"color": theme.DANGER, "size": 16}
    assert out["click-data"]["children"] == "An error occurred while rendering this view."
    assert out["tube-selector"] == {"style": {"display": "none"}}  # its options are kept
    assert out["main-graph"]["style"] == {"height": "100%", "width": "100%", "overflow": "hidden"}
    assert out["image-container"]["style"] == {"display": "none"}


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_a_released_dashboard_is_not_kept_alive_by_dash(tmp_path, builder):
    # Dash keeps the newest app in a module global (for dash.get_app and Dash Pages): the
    # last dashboard built stayed in memory with its data (about 20 MB on a 46k-row trial)
    # after its window was released, until another dashboard was built.
    import gc
    import threading
    import weakref

    dashboard = builder(tmp_path)
    answers = []

    def ask(app):
        values = {"view-selector.value": "stacked"}
        answers.append(call_callback(app, "main-graph.figure", values, ["view-selector.value"])[0])

    # werkzeug answers each request on a thread of its own
    request = threading.Thread(target=ask, args=(dashboard.app,))
    request.start()
    request.join(30)
    assert answers == [200]
    alive = weakref.ref(dashboard)

    del dashboard, request
    gc.collect()

    assert alive() is None


def test_only_the_first_dashboard_reads_the_plugin_entry_points(tmp_path, monkeypatch):
    # Dash read every installed package's entry points for plugin hooks on each new app
    # (5-20 ms per dashboard): it never sets its own read-once flag.
    import dash._hooks as dash_hooks

    from app.visualization.dash_shell import create_dash_app

    scans = []
    distributions = dash_hooks._importlib_metadata.distributions

    def counted():
        scans.append(1)
        return distributions()

    monkeypatch.setattr(dash_hooks._importlib_metadata, "distributions", counted)
    monkeypatch.setattr(dash_hooks.HooksManager, "_registered", False)
    first = create_dash_app(__name__)
    create_dash_app(__name__)
    build_length_app(tmp_path)
    build_area_app(tmp_path)

    assert len(scans) == 1
    assert first.config.name == __name__


# ---- a memoized figure is converted for the browser once --------------------------


def test_browser_figure_is_what_dash_sends_of_the_figure(length_app):
    from dash import Patch, no_update
    from dash._utils import to_json

    from app.visualization.dash_shell import browser_figure, with_uirevision

    figure = with_uirevision(length_app.show_growth_over_time([1, 2], True, True, None), "time")
    kept = browser_figure(figure)

    assert to_json(kept) == to_json(figure)  # byte for byte
    assert browser_figure(figure) is kept  # converted once
    with_uirevision(figure, "time", "again")
    assert browser_figure(figure)["layout"]["uirevision"] == "time:again"
    assert to_json(browser_figure(figure)) == to_json(figure)
    patch = Patch()
    assert browser_figure(patch) is patch
    assert browser_figure(no_update) is no_update


def _raw_response(dash, values, changed) -> bytes:
    """The response bytes of the main-graph callback (call_callback parses them)."""
    from tests.viz_dash_helpers import _split_output_key, find_callback_key

    key = find_callback_key(dash, "main-graph.figure")
    spec = dash.callback_map[key]

    def fill(items):
        return [
            {"id": i["id"], "property": i["property"],
             "value": values.get(f"{i['id']}.{i['property']}")}
            for i in items
        ]  # fmt: skip

    payload = {
        "output": key,
        "outputs": _split_output_key(key),
        "inputs": fill(spec["inputs"]),
        "state": fill(spec["state"]),
        "changedPropIds": list(changed),
    }
    resp = dash.server.test_client().post(
        "/_dash-update-component",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    return resp.data


SHOWN_AGAIN = [
    {"view-selector.value": "time", "selected-tubes-store.data": [1, 2, 3],
     "growth-viz-options.value": ["show_field_avg", "show_field_var"]},
    {"view-selector.value": "faceted", "faceted-tubes-store.data": [1, 2],
     "faceted-dates-store.data": []},
    {"view-selector.value": "lines", "tube-selector.value": 1},
    {"view-selector.value": "stacked"},
]  # fmt: skip


@pytest.mark.parametrize(
    "values", SHOWN_AGAIN, ids=[values["view-selector.value"] for values in SHOWN_AGAIN]
)
@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_a_view_shown_again_sends_the_same_figure(tmp_path, monkeypatch, builder, values):
    # The second request is a FigureMemo hit served from the kept browser JSON: it must
    # send the bytes of the first, and both what Dash makes of the figure itself.
    # Both dashboards answer through the one metric callback module.
    from app.visualization import dash_metric_callbacks

    built = builder(tmp_path)
    view = ["view-selector.value"]

    first = _raw_response(built.app, values, view)
    again = _raw_response(built.app, values, view)
    monkeypatch.setattr(dash_metric_callbacks, "browser_figure", lambda figure: figure)
    (tmp_path / "plain").mkdir()
    plain = _raw_response(builder(tmp_path / "plain").app, values, view)

    assert again == first
    assert plain == first


# ---- FR-15: the notice of the rows left out of the charts -------------------------------


def _find_component(component, component_id):
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            found = _find_component(child, component_id)
            if found is not None:
                return found
    elif children is not None and not isinstance(children, str):
        return _find_component(children, component_id)
    return None


def test_dashboard_page_shows_the_exclusions_notice_only_when_rows_were_left_out():
    from app.data_processing.data_processor import RowExclusions
    from app.visualization.dash_metric_layout import LENGTH_SHELL
    from app.visualization.dash_shell import DashboardShell, exclusions_note

    class Processor:
        exclusions = RowExclusions(failed=2, bad_name=0, unreadable=0)

    note = exclusions_note(Processor())
    assert note and note.startswith("2 rows are not shown")
    assert exclusions_note(object()) is None

    shell = DashboardShell(LENGTH_SHELL)
    with_note = shell.page(controls=[], chart=[], notice=note)
    assert _find_component(with_note, "viz-data-notice").children == note
    assert _find_component(shell.page(controls=[], chart=[]), "viz-data-notice") is None


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_a_dashboard_tells_the_page_which_rows_it_left_out(tmp_path, builder):
    # The served page of a dashboard over a CSV with a failed measurement carries the
    # sentence under its header; a dashboard over a clean CSV carries none.
    from tests.viz_dash_helpers import AREA_COLUMN, LENGTH_COLUMN

    clean = builder(tmp_path)
    assert (
        _find(
            _served_layout(clean.app), lambda n: n.get("props", {}).get("id") == "viz-data-notice"
        )
        is None
    )

    column = LENGTH_COLUMN if builder is build_length_app else AREA_COLUMN
    name = "root_lengths.csv" if builder is build_length_app else "root_areas.csv"
    csv = tmp_path / "failed" / name
    csv.parent.mkdir()
    csv.write_text(
        f"Image,Tube,Position,Date,Time,{column},Error\n"
        "A_T001_L005_2024.07.18_120000_001,1,5,2024.07.18,120000,2.5,\n"
        "A_T001_L006_2024.07.18_120000,,,,,0,cannot identify image file\n",
        encoding="utf-8",
    )
    from app.data_processing.data_processor import DataProcessor
    from app.data_processing.data_processor_area import DataProcessorArea
    from app.visualization.dash_app import DashApp
    from app.visualization.dash_app_area import DashAppArea

    if builder is build_length_app:
        dashboard = DashApp(DataProcessor(str(csv)), str(csv.parent))
    else:
        dashboard = DashAppArea(DataProcessorArea(str(csv)), str(csv.parent))
    notice = _find(
        _served_layout(dashboard.app), lambda n: n.get("props", {}).get("id") == "viz-data-notice"
    )

    assert notice["props"]["children"] == (
        "1 row is not shown: 1 failed measurement (see the CSV's Error column)."
    )
