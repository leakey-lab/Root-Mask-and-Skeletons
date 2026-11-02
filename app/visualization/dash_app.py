"""
Root-length dashboard (Dash): one app over one root_lengths.csv.

:class:`DashApp` is the :class:`.dash_metric_app.MetricDashApp` of root length: it
names the shell (:data:`.dash_metric_layout.LENGTH_SHELL`), the page title and the
export file prefix. The page, the callbacks and the figures are the metric dashboard's,
shared with the root-area dashboard (:mod:`.dash_app_area`).
"""

from .dash_metric_app import MetricDashApp
from .dash_metric_layout import LENGTH_SHELL


class DashApp(MetricDashApp):
    """The root-length dashboard: one Dash app over one root_lengths.csv."""

    shell_spec = LENGTH_SHELL
    page_title = "SPROUTS — Root Length"
    metric_slug = "root_length"
