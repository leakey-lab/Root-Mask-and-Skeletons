"""
Root-area dashboard (Dash): one app over one root_areas.csv.

:class:`DashAppArea` is the :class:`.dash_metric_app.MetricDashApp` of root area: it
names the shell (:data:`AREA_SHELL`), the page title and the export file prefix. The
page, the callbacks and the figures are the metric dashboard's (:mod:`.dash_metric_app`):
the same four views, controls and toggles, charting area (the data processor's
``metric_label`` and ``unit``, "Area" and "mm²").
"""

from .dash_metric_app import MetricDashApp
from .dash_metric_layout import AREA_SHELL


class DashAppArea(MetricDashApp):
    """The root-area dashboard: one Dash app over one root_areas.csv."""

    shell_spec = AREA_SHELL
    page_title = "SPROUTS — Root Area"
    metric_slug = "root_area"
