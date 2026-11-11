"""The callback wiring of both dashboards, checked against the page they serve.

Derived from the app, not frozen: every callback is served once; every id an Input, Output or
State names is on the served page (the dashboards run with ``suppress_callback_exceptions``,
so a misspelt id raises nothing and the callback never fires); every clientside function a
callback names is defined by the page's scripts; "Export CSV" never runs on page load. Until
2026-11-10 this file held SHA-256 hashes of the served layout and the clientside JavaScript and
a frozen table of every callback, taken when ``dash_app.py`` was split (CQ-07); what only they
caught is now a behaviour test (the typed tube entry on Enter, the export on load, the time
view's first options, the thumbnails cleared on a view change).
"""

from __future__ import annotations

import json

import pytest

from tests.viz_dash_helpers import build_area_app, build_length_app


@pytest.fixture(scope="module")
def length_app(tmp_path_factory):
    return build_length_app(tmp_path_factory.mktemp("length_golden"))


def _served(dash, path: str):
    resp = dash.server.test_client().get(path)
    assert resp.status_code == 200
    return json.loads(resp.data)


def test_every_callback_is_served_once(length_app):
    outputs = [callback["output"] for callback in _served(length_app.app, "/_dash-dependencies")]
    assert len(outputs) == len(set(outputs)) == len(length_app.app.callback_map)


BUILDERS = [build_length_app, build_area_app]


def _output_ids(output: str) -> list[str]:
    """The component ids a served callback writes ('a.x' or '..a.x...b.y..', '@hash' kept off)."""
    parts = output[2:-2].split("...") if output.startswith("..") else [output]
    return [part.rsplit(".", 1)[0] for part in parts]


def _served_ids(node) -> set[str]:
    """Every string id in the served layout (JSON)."""
    found = set()
    if isinstance(node, dict):
        props = node.get("props")
        if isinstance(props, dict) and isinstance(props.get("id"), str):
            found.add(props["id"])
        for value in node.values():
            found |= _served_ids(value)
    elif isinstance(node, list):
        for value in node:
            found |= _served_ids(value)
    return found


@pytest.mark.parametrize("builder", BUILDERS, ids=["length", "area"])
def test_every_id_a_callback_names_is_on_the_served_page(tmp_path, builder):
    # The dashboards run with suppress_callback_exceptions=True: a misspelt id raises
    # nothing, and the callback never fires.
    dash = builder(tmp_path).app
    on_page = _served_ids(_served(dash, "/_dash-layout"))
    named = {}
    for callback in _served(dash, "/_dash-dependencies"):
        for cid in _output_ids(callback["output"]):
            named.setdefault(cid, callback["output"])
        for item in callback["inputs"] + callback["state"]:
            named.setdefault(item["id"], callback["output"])
    missing = {cid: output for cid, output in named.items() if cid not in on_page}
    assert missing == {}


@pytest.mark.parametrize("builder", BUILDERS, ids=["length", "area"])
def test_every_clientside_function_a_callback_names_is_defined_by_the_page(qapp, tmp_path, builder):
    from PyQt6.QtQml import QJSEngine

    dash = builder(tmp_path).app
    engine = QJSEngine()
    engine.evaluate("var window = this; window.dash_clientside = {};")
    for script in dash._inline_scripts:
        result = engine.evaluate(script)
        assert not result.isError(), result.toString()
    named = [
        callback["clientside_function"]
        for callback in _served(dash, "/_dash-dependencies")
        if callback["clientside_function"]
    ]
    assert named  # the strip, caret, view cards, plotted tube, refit and thumbnails
    undefined = [
        function
        for function in named
        if not engine.evaluate(
            f"typeof (window.dash_clientside[{json.dumps(function['namespace'])}] || {{}})"
            f"[{json.dumps(function['function_name'])}] === 'function'"
        ).toBool()
    ]
    assert undefined == []


@pytest.mark.parametrize("builder", BUILDERS, ids=["length", "area"])
def test_export_csv_never_runs_on_page_load(tmp_path, builder):
    # Dash fires every callback without prevent_initial_call when the page loads: the
    # export would offer a download each time a dashboard opens.
    (export,) = [
        callback
        for callback in _served(builder(tmp_path).app, "/_dash-dependencies")
        if callback["output"] == "viz-export-download.data"
    ]
    assert export["prevent_initial_call"] is True
