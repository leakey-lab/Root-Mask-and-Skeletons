"""Dashboard callbacks through Dash's dispatch: PV-07, PV-09, PV-14 (app side) and
the view switch that used to wipe every dropdown selection."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from app.visualization import theme
from tests.viz_dash_helpers import (
    NO_UPDATE,
    FakeImageManager,
    build_area_app,
    build_length_app,
    call_callback,
    call_clientside,
    walk_components,
)

FIGURE = "main-graph.figure"


@pytest.fixture
def length_app(tmp_path):
    return build_length_app(tmp_path)


def _figure(dash, values, changed):
    status, out = call_callback(dash, FIGURE, values, changed)
    assert status == 200, "expected a figure"
    return out["main-graph"]["figure"]


# ---- PV-07: clicks and legend toggles keep the user's zoom ---------------------


def test_a_click_on_the_chart_does_not_rebuild_it(length_app):
    status, _ = call_callback(
        length_app.app,
        FIGURE,
        {"view-selector.value": "stacked", "main-graph.clickData": {"points": [{"x": "Tube 2"}]}},
        changed=["main-graph.clickData"],
    )
    assert status == 204


def test_page_load_draws_the_default_chart(length_app):
    """On load Dash fires the main callback once, triggered by the stores that the
    other initial callbacks just wrote, not by an empty trigger list."""
    figure = _figure(
        length_app.app,
        {"view-selector.value": "stacked", "legend-visible-store.data": True},
        [
            "faceted-tubes-store.data",
            "faceted-dates-store.data",
            "selected-tubes-store.data",
            "legend-visible-store.data",
        ],
    )
    assert figure["data"], "the stacked chart must be drawn on page load"


@pytest.mark.parametrize("view", ["stacked", "time"])
def test_a_load_call_triggered_by_the_legend_store_alone_draws_the_chart(length_app, view):
    """The legend store holds None until the first press, and a press always stores True
    or False: a call that only the store triggered while it holds None is a page load."""
    values = {"view-selector.value": view, "legend-visible-store.data": None,
              "selected-tubes-store.data": [1, 2], "growth-viz-options.value": ["show_field_avg"]}  # fmt: skip
    figure = _figure(length_app.app, values, ["legend-visible-store.data"])

    assert "__dash_patch_update" not in figure  # the whole figure, not a legend Patch
    assert figure["data"]
    if view == "time":
        assert figure["layout"]["showlegend"] is True  # automatic for two tubes


VIEWS = [
    ("stacked", {}),
    ("time", {"selected-tubes-store.data": [1], "growth-viz-options.value": ["show_field_avg"]}),
    ("lines", {"tube-selector.value": 2}),
    ("faceted", {"faceted-tubes-store.data": [1, 2], "faceted-dates-store.data": []}),
]


@pytest.mark.parametrize("view, extra", VIEWS, ids=[v for v, _ in VIEWS])
@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_every_view_keeps_ui_state_across_updates(tmp_path, builder, view, extra):
    figure = _figure(
        builder(tmp_path).app, {"view-selector.value": view, **extra}, ["view-selector.value"]
    )
    assert figure["layout"].get("uirevision")


def test_ui_state_resets_when_the_plotted_tube_changes(length_app):
    def revision(tube):
        figure = _figure(
            length_app.app,
            {"view-selector.value": "lines", "tube-selector.value": tube},
            ["tube-selector.value"],
        )
        return figure["layout"]["uirevision"]

    assert revision(1) != revision(2)


def test_lines_view_plots_the_tube_shown_in_the_dropdown(length_app):
    figure = _figure(
        length_app.app,
        {"view-selector.value": "lines", "tube-selector.value": 2},
        ["view-selector.value", "tube-selector.value"],
    )
    assert "Tube 2" in figure["layout"]["title"]["text"]


def test_legend_toggle_patches_the_figure_instead_of_rebuilding_it(length_app):
    status, out = call_callback(
        length_app.app,
        FIGURE,
        {"view-selector.value": "time", "legend-visible-store.data": False,
         "growth-viz-options.value": ["show_field_avg"], "selected-tubes-store.data": [1, 2]},
        changed=["legend-visible-store.data"],
    )  # fmt: skip
    assert status == 200
    patch = out["main-graph"]["figure"]
    assert patch["__dash_patch_update"] == "__dash_patch_update"
    # the legend and its margin (no empty band beside the plot, R-VIZ X3) only
    assert patch["operations"] == [
        {"operation": "Assign", "location": ["layout", "showlegend"], "params": {"value": False}},
        {"operation": "Assign", "location": ["layout", "margin", "r"], "params": {"value": 24}},
    ]
    assert set(out) == {"main-graph"}  # nothing else is re-sent


# ---- PV-09: hover thumbnails by URL --------------------------------------------------


@pytest.fixture
def app_with_images(tmp_path):
    images = {}
    for position in range(11, 21):
        stem = f"24EFCAM1_T001_L{position:03d}_2024.07.18_111618_001_SMM"
        path = tmp_path / f"{stem}.jpg"
        pixels = np.full((480, 640, 3), 40 + 8 * position, np.uint8)
        Image.fromarray(pixels).save(path, quality=90)
        images[stem] = str(path)
    return build_length_app(tmp_path, FakeImageManager(images))


def _hover(dash, view, y=20):
    return call_callback(
        dash,
        "image-container.children",
        {"main-graph.hoverData": {"points": [{"x": 1.0, "y": y, "customdata": "2024.07.18"}]},
         "tube-selector.value": 1, "view-selector.value": view},
        changed=["main-graph.hoverData"],
    )  # fmt: skip


def test_hover_returns_small_thumbnail_urls_that_serve_jpegs(app_with_images):
    dash = app_with_images.app
    status, out = _hover(dash, "lines")
    assert status == 200
    children = out["image-container"]["children"]
    assert len(json.dumps(children)) < 50 * 1024  # was ~0.75 MB of base64 PNG
    sources = [node["props"]["src"] for node in _json_nodes(children) if node.get("type") == "Img"]
    assert len(sources) == 10
    client = dash.server.test_client()
    for src in sources[:2]:
        assert src.endswith(".jpg") and not src.startswith("data:")
        resp = client.get(src)
        assert resp.status_code == 200 and resp.mimetype == "image/jpeg"


def test_hover_outside_the_lines_view_sends_nothing(app_with_images):
    status, _ = _hover(app_with_images.app, "stacked")
    assert status == 204


@pytest.fixture
def app_with_one_image(tmp_path):
    """Tube 1 on 2024.07.18 is measured at L5 and L15; the one image is L15's, inside
    the interval L11-L20 but not at its end."""
    stem = "24EFCAM1_T001_L015_2024.07.18_111618_001_SMM"
    path = tmp_path / f"{stem}.jpg"
    Image.fromarray(np.full((48, 64, 3), 90, np.uint8)).save(path)
    return build_length_app(tmp_path, FakeImageManager({stem: str(path)}))


def _profile_hover(dashboard, end: int) -> str:
    figure = dashboard.show_growth_lines(1)
    trace = next(t for t in figure.data if t.name == "2024-07-18")
    return trace.text[list(trace.y).index(end)]


def test_depth_profile_hover_finds_an_image_anywhere_in_the_interval(app_with_one_image):
    # S14-05: the hover checked only the interval's end (L20) and said "Not Available"
    # while the side panel showed L15's image for the same point.
    status, out = _hover(app_with_one_image.app, "lines", y=20)
    sources = [n for n in _json_nodes(out["image-container"]["children"]) if n["type"] == "Img"]
    assert status == 200 and len(sources) == 1  # the side panel shows the image

    assert "Image  <b>Available</b>" in _profile_hover(app_with_one_image, 20)
    assert "Image  <b>Not Available</b>" in _profile_hover(app_with_one_image, 10)


def _json_nodes(node):
    if isinstance(node, list):
        for child in node:
            yield from _json_nodes(child)
    elif isinstance(node, dict):
        if "type" in node and "props" in node:
            yield node
            yield from _json_nodes(node["props"].get("children"))


# ---- PV-14 (app side): colours come from the theme ---------------------------------


def _inline_colors(node) -> list:
    found = []
    for item in _json_nodes(node if isinstance(node, list) else [node]):
        style = item["props"].get("style") or {}
        found += [v for k, v in style.items() if "color" in k.lower()]
    return found


@pytest.mark.parametrize("tubes", [[], [1, 2], [1, 2, 3, 4, 5, 6, 7]])
def test_faceted_selection_feedback_has_no_inline_colours(length_app, tubes):
    _, out = call_callback(
        length_app.app,
        "faceted-selection-info.children",
        {"faceted-tube-selector.value": tubes, "faceted-date-selector.value": ["2024-07-18"]},
        changed=["faceted-tube-selector.value"],
    )
    assert _inline_colors(out["faceted-selection-info"]["children"]) == []
    assert _inline_colors(out["tube-date-availability"]["children"]) == []


def test_too_many_faceted_tubes_is_shown_as_an_error_in_theme_colours(length_app):
    figure = _figure(
        length_app.app,
        {"view-selector.value": "faceted", "faceted-tubes-store.data": [1, 2, 3, 4, 5, 6, 7]},
        ["faceted-tubes-store.data"],
    )
    (note,) = figure["layout"]["annotations"]
    assert note["font"]["color"] == theme.DANGER
    assert "7/6" in note["text"]


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_depth_profile_lines_use_the_theme_palette(tmp_path, builder):
    figure = _figure(
        builder(tmp_path).app,
        {"view-selector.value": "lines", "tube-selector.value": 1},
        ["view-selector.value"],
    )
    colors = [trace["line"]["color"] for trace in figure["data"]]
    assert colors == theme.date_colors(2)


# ---- switching views keeps every selection ---------------------------------------


def _blanked(out: dict, component: str) -> bool:
    """True if the response empties ``component``'s options (no_update is fine)."""
    return out.get(component, {}).get("options") == []


def test_view_panels_never_blank_the_dropdown_options(length_app):
    # The panels switch in the browser (R-VIZ X2); run in Qt's JavaScript engine.
    for view in ("stacked", "time", "lines", "faceted"):
        out, _ = call_clientside(
            length_app.app,
            "viz-options-card.style",
            {"view-selector.value": view},
            changed=["view-selector.value"],
        )
        for component in ("tube-multiselect", "faceted-tube-selector", "faceted-date-selector"):
            assert not _blanked(out, component), (view, component)
            assert out[component]["options"] == NO_UPDATE, (view, component)


@pytest.mark.parametrize(
    "view, cards",
    [
        ("stacked", set()),
        ("time", {"tube-selection-card", "viz-options-card"}),
        ("lines", set()),
        ("faceted", {"faceted-options-card"}),
    ],
)
def test_each_view_shows_only_its_own_control_cards(length_app, view, cards):
    out, _ = call_clientside(
        length_app.app,
        "viz-options-card.style",
        {"view-selector.value": view},
        ["view-selector.value"],
    )
    shown = {cid for cid, props in out.items() if props.get("style") == {"display": "block"}}
    hidden = {cid for cid, props in out.items() if props.get("style") == {"display": "none"}}
    assert shown == cards
    assert shown | hidden == {"tube-selection-card", "viz-options-card", "faceted-options-card"}


def test_main_callback_never_blanks_the_tube_options(length_app):
    _, out = call_callback(
        length_app.app, FIGURE, {"view-selector.value": "stacked"}, ["view-selector.value"]
    )
    assert not _blanked(out, "tube-selector")


@pytest.mark.parametrize("builder", [build_length_app, build_area_app], ids=["length", "area"])
def test_tube_dropdowns_are_populated_from_the_start(tmp_path, builder):
    """Options that exist from page load mean Dash 4's Dropdown never has to drop
    a selected value because its options were momentarily empty."""
    layout = builder(tmp_path).app.layout
    for component in walk_components(layout):
        if type(component).__name__ == "Dropdown":
            assert getattr(component, "options", None), f"{component.id} starts without options"


# ---- a figure builder that fails shows its error ----------------------------------------

BUILDERS = [build_length_app, build_area_app]
METRICS = ["length", "area"]


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


# ---- what a view starts with, and what a view change clears -----------------------------


def test_growth_over_time_opens_with_the_field_average_and_its_band(tmp_path):
    dash = build_length_app(tmp_path).app
    layout = json.loads(dash.server.test_client().get("/_dash-layout").data)
    (options,) = [n for n in _json_nodes(layout) if n["props"].get("id") == "growth-viz-options"]
    start = options["props"]["value"]

    figure = _figure(
        dash,
        {"view-selector.value": "time", "growth-viz-options.value": start,
         "selected-tubes-store.data": [1, 2]},
        ["view-selector.value"],
    )  # fmt: skip

    assert start == ["show_field_avg", "show_field_var"]
    names = [trace.get("name") for trace in figure["data"]]
    assert "Field Average" in names and "Field Mean ±1 SD" in names


@pytest.mark.parametrize("changed", ["view-selector.value", "tube-selector.value"])
def test_a_view_or_tube_change_clears_the_hover_thumbnails(length_app, changed):
    out, _ = call_clientside(
        length_app.app,
        "image-container.children",
        {"view-selector.value": "lines", "tube-selector.value": 2},
        [changed],
        duplicate=True,
    )

    assert out["image-container"]["children"] == []
