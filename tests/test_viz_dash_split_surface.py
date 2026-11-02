"""Length dashboard: the surface left after the CQ-07 split, and nothing heavier loads.

``dash_app.py`` was split into ``dash_length_layout``, ``dash_length_views`` and
``dash_length_callbacks`` (now ``dash_metric_layout``, ``dash_metric_views`` and
``dash_metric_callbacks``), and the new modules import nothing the dashboard did not
import before. ``DashApp``'s body moved to ``dash_metric_app.MetricDashApp``, shared
with the area dashboard; ``DashApp`` only names its shell, page title and export file
prefix. The compatibility layer the split kept (15 underscore delegators on
``DashApp`` and 7 module aliases) is gone (S15-07 tier 2): nothing used it, the
callbacks call the modules directly.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.viz_dash_helpers import build_length_app

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MODULES = (
    "app.visualization.dash_metric_layout",
    "app.visualization.dash_metric_views",
    "app.visualization.dash_metric_callbacks",
    "app.visualization.dash_metric_app",
)

# Methods MetricDashApp (the body DashApp had) defines itself: name -> (kind, parameters).
METRIC_DASH_APP_OWN = {
    "_parse_images_after_this_response": ("function", ("self",)),
    "_setup_callbacks": ("function", ("self",)),
    "_setup_layout": ("function", ("self",)),
}
# What DashApp defines itself: the three class attributes that name the length dashboard.
DASH_APP_OWN = {"shell_spec", "page_title", "metric_slug"}
# The compatibility layer removed in S15-07 tier 2: delegators to the layout and view
# modules, and the module constants dash_app.py defined before the split.
REMOVED_DELEGATORS = (
    "_availability_view", "_controls", "_controls_summary", "_export_table", "_faceted_info",
    "_faceted_options_card", "_faceted_outputs", "_faceted_styles", "_hover_date",
    "_legend_patch", "_lines_tube", "_render_view", "_thumbnail", "_tube_selection_card",
    "_viz_options_card",
)  # fmt: skip
REMOVED_MODULE_ALIASES = (
    "_DATE_TOKEN", "_FACETED_GRAPH_STYLE", "_GRAPH_STYLE", "_HIDDEN", "_IMAGES_SHOWN",
    "_SHOWN", "_VIEW_PANELS_JS",
)  # fmt: skip
# Inherited from DashVisualizations (the figures); unchanged by the split.
DASH_APP_INHERITED = {
    "_faceted_figure", "_field_average_traces", "_growth_lines_figure",
    "_growth_over_time_figure", "_interval_hover", "_memoized", "_metric_text",
    "_stacked_bar_figure", "create_faceted_depth_profile", "create_stacked_bar_chart",
    "get_tube_date_availability", "show_growth_lines", "show_growth_over_time",
}  # fmt: skip
DASH_APP_INSTANCE_ATTRS = {
    "_date_options", "_tube_options", "app", "available_images", "data_cache",
    "data_processor", "image_manager", "save_directory", "shell", "thumbnail_server",
}  # fmt: skip


def test_dash_app_keeps_every_method_with_its_kind_and_parameters():
    from app.visualization.dash_app import DashApp
    from app.visualization.dash_metric_app import MetricDashApp

    assert issubclass(DashApp, MetricDashApp)
    own = {name for name in vars(MetricDashApp) if not name.startswith("__")}
    assert own == set(METRIC_DASH_APP_OWN)
    for name, (kind, parameters) in METRIC_DASH_APP_OWN.items():
        attribute = inspect.getattr_static(MetricDashApp, name)
        assert type(attribute).__name__ == kind, name
        assert tuple(inspect.signature(getattr(DashApp, name)).parameters) == parameters, name
    assert {name for name in vars(DashApp) if not name.startswith("__")} == DASH_APP_OWN
    missing = sorted(name for name in DASH_APP_INHERITED if not hasattr(DashApp, name))
    assert not missing, f"DashApp lost {missing}"


def test_dash_app_instance_keeps_its_attributes(tmp_path):
    dashboard = build_length_app(tmp_path)
    missing = sorted(name for name in DASH_APP_INSTANCE_ATTRS if not hasattr(dashboard, name))
    assert not missing, f"DashApp instances lost {missing}"


def test_the_old_compatibility_layer_is_gone():
    import app.visualization.dash_app as dash_app
    from app.visualization import dash_metric_layout

    assert len(REMOVED_DELEGATORS) == 15 and len(REMOVED_MODULE_ALIASES) == 7
    kept = sorted(name for name in REMOVED_DELEGATORS if hasattr(dash_app.DashApp, name))
    assert not kept, f"DashApp still has {kept}"
    kept = sorted(name for name in REMOVED_MODULE_ALIASES if hasattr(dash_app, name))
    assert not kept, f"dash_app still defines {kept}"
    assert dash_app.LENGTH_SHELL is dash_metric_layout.LENGTH_SHELL  # the page's shell


@pytest.mark.parametrize("module", SPLIT_MODULES)
def test_split_module_imports(module):
    __import__(module)


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_split_modules_load_no_package_the_dashboard_did_not_load():
    # First what dash_app imported before the split (what the moved code already paid
    # for), then the split modules and dash_app: no new top-level module may appear,
    # and none of the heavy stacks the dashboard never needed. dash_app imports flask
    # itself; since Dash 4.2 ``import dash`` no longer does.
    result = _run(
        """
        import sys

        import dash, dash.exceptions, dash_bootstrap_components, flask, pandas
        import app.config
        from app.visualization import (
            dash_data_cache, dash_export, dash_image_utils, dash_shell,
            dash_visualizations, faceted_profile, theme,
        )

        def top():
            return {name.split(".")[0] for name in sys.modules}

        before = top()
        import app.visualization.dash_metric_layout
        import app.visualization.dash_metric_views
        import app.visualization.dash_metric_callbacks
        import app.visualization.dash_metric_app
        import app.visualization.dash_app
        new = sorted(top() - before)
        heavy = sorted(top() & {"torch", "torchvision", "cv2", "skimage", "scipy", "PyQt6"})
        print("new=" + ",".join(new))
        print("heavy=" + ",".join(heavy))
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    lines = dict(line.split("=", 1) for line in result.stdout.strip().splitlines())
    assert lines["new"] == "", f"the split modules import new packages: {lines['new']}"
    assert lines["heavy"] == "", f"the dashboard loads {lines['heavy']}"
