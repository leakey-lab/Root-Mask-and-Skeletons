"""A failed "Export CSV" is reported to the embedding window (C6-06, owner item #23).

The CSV of the current view is built on the Dash server. A build that raises gave the
user nothing: the callback answered PreventUpdate and only the log knew. Now the shell
also hands a one-line message to ``DashboardShell.on_export_failed``; the dashboard
window sets that hook and shows the message as a warning toast. A view with nothing to
export is not a failure and stays silent. The page and its callback wiring are unchanged.
"""

from __future__ import annotations

import pytest
from dash import Dash

from app.visualization import dash_export
from app.visualization.dash_metric_layout import LENGTH_SHELL
from app.visualization.dash_shell import DashboardShell
from tests.viz_dash_helpers import build_area_app, build_length_app, call_callback


def _shell_app(build_table):
    app = Dash(__name__)
    shell = DashboardShell(LENGTH_SHELL)
    app.layout = shell.header()
    shell.register_export(
        app,
        build_table=build_table,
        source_csv="root_lengths.csv",
        metric_column="Length (mm)",
        metric_slug="root_length",
    )
    return app, shell


def _click_export(app, view="stacked", **values):
    status, out = call_callback(
        app, "viz-export-download.data",
        {"viz-export-btn.n_clicks": 1, "view-selector.value": view, **values},
        changed=["viz-export-btn.n_clicks"],
    )  # fmt: skip
    return status, out


def _broken_table(*_args):
    raise KeyError("Tube")


def test_a_table_that_fails_to_build_is_reported_by_view_name():
    app, shell = _shell_app(_broken_table)
    reported = []
    shell.on_export_failed = reported.append

    status, _ = _click_export(app, view="time")

    assert status == 204  # the page keeps working; nothing is downloaded
    assert reported == ["Could not export the Growth Over Time view: KeyError: 'Tube'"]


def test_a_csv_that_fails_to_render_is_reported(monkeypatch):
    app, shell = _shell_app(lambda *_args: object())
    reported = []
    shell.on_export_failed = reported.append

    def broken_render(*_args, **_kwargs):
        raise ValueError("bad table")

    monkeypatch.setattr(dash_export, "render_csv", broken_render)

    status, _ = _click_export(app)

    assert status == 204
    assert reported == ["Could not export the Totals by Tube view: ValueError: bad table"]


def test_a_view_with_nothing_to_export_is_not_a_failure():
    app, shell = _shell_app(lambda *_args: None)
    reported = []
    shell.on_export_failed = reported.append

    status, _ = _click_export(app)

    assert status == 204
    assert reported == []


def test_a_failure_without_a_window_is_only_logged(caplog):
    app, shell = _shell_app(_broken_table)
    assert shell.on_export_failed is None  # a shell outside the app (tests, a browser)

    status, _ = _click_export(app)

    assert status == 204
    assert "Could not export the stacked view" in caplog.text


def test_a_broken_reporter_never_breaks_the_page(caplog):
    app, shell = _shell_app(_broken_table)

    def gone(_message):  # e.g. the window was deleted while the request ran
        raise RuntimeError("wrapped C/C++ object has been deleted")

    shell.on_export_failed = gone

    status, _ = _click_export(app)

    assert status == 204
    assert "Could not report the failed export" in caplog.text


@pytest.mark.parametrize("build", [build_length_app, build_area_app], ids=["length", "area"])
def test_both_dashboards_report_a_failed_export(tmp_path, monkeypatch, build):
    dash_app = build(tmp_path)
    reported = []
    dash_app.shell.on_export_failed = reported.append

    def broken_render(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(dash_export, "render_csv", broken_render)

    status, _ = _click_export(dash_app.app)

    assert status == 204
    assert reported == ["Could not export the Totals by Tube view: OSError: disk full"]
