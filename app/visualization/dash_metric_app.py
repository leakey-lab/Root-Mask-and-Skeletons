"""
Metric dashboards (Dash, length and area): one app over one metric CSV.

:class:`MetricDashApp` builds the app and holds what its parts share: the data
(:class:`.dash_data_cache.DataCache`), the figures (the
:class:`.dash_visualizations.DashVisualizations` mixin), the hover-image index and
its thumbnail server. The root-length (:class:`.dash_app.DashApp`) and root-area
(:class:`.dash_app_area.DashAppArea`) dashboards are this class; each names only its
shell, page title and export file prefix, and the data processor names the metric
(``metric_label``, ``unit``). The parts live in their own modules:

- :mod:`.dash_metric_layout`: the page (control cards, chart row, stores);
- :mod:`.dash_metric_views`: what each view shows for a selection;
- :mod:`.dash_metric_callbacks`: the callbacks that wire them to the page.

The page chrome comes from :mod:`.dash_shell`, hover thumbnails from
:mod:`.dash_image_utils`. Component ids, callback Outputs and figure customdata
are a contract with the renderer and the embedding window: keep them stable.
"""

from typing import Any, ClassVar

import flask

from . import dash_metric_callbacks as metric_callbacks
from . import dash_metric_layout as metric_layout
from . import theme
from .dash_data_cache import DataCache
from .dash_image_utils import ThumbnailServer, build_available_images_map
from .dash_shell import DashboardShell, ShellSpec, create_dash_app, exclusions_note, tube_options
from .dash_visualizations import DashVisualizations

theme.use("sprouts")


class MetricDashApp(DashVisualizations):
    """A metric dashboard: one Dash app over one metric CSV (root_lengths.csv,
    root_areas.csv).

    The page is built by :mod:`.dash_metric_layout` and the callbacks call
    :mod:`.dash_metric_views` directly; this class holds what they share. A subclass
    sets the three class attributes below and nothing else.
    """

    #: The page chrome: title, views, plot file name (:func:`.dash_metric_layout.metric_shell`).
    shell_spec: ClassVar[ShellSpec]
    #: The browser page's title, e.g. "SPROUTS — Root Length".
    page_title: ClassVar[str]
    #: File-name prefix of "Export CSV", e.g. "root_length".
    metric_slug: ClassVar[str]

    def __init__(self, data_processor: Any, save_directory: str, image_manager=None):
        self.data_processor = data_processor
        self.save_directory = save_directory
        self.image_manager = image_manager

        # (tube, position, date) -> image path, parsed after the first chart was sent
        # (see _parse_images_after_this_response) or on first use (PV-17).
        self.available_images = build_available_images_map(image_manager)
        self._images_parse_started = False
        self.data_cache = DataCache(data_processor)
        self._tube_options = tube_options(self.data_cache.tubes)
        self._date_options = [
            {"label": date.strftime("%Y-%m-%d"), "value": date.isoformat()}
            for date in self.data_cache.dates
        ]

        self.shell = DashboardShell(self.shell_spec)
        # The subclass's module is the Dash import name: the length app stays
        # "app.visualization.dash_app".
        self.app = create_dash_app(type(self).__module__, title=self.page_title)
        # Hover thumbnails are served by URL from this app's own server (PV-09).
        self.thumbnail_server = ThumbnailServer(self.available_images.thumbnails)
        self.thumbnail_server.register(self.app.server)

        self._setup_layout()
        self._setup_callbacks()

    # ---- hover images --------------------------------------------------------------

    def _parse_images_after_this_response(self) -> None:
        """Parse the hover-image names once, after the response being served is sent.

        The figure callback calls this, so the names (~140 ms for 46k) are parsed on a
        background thread while the browser draws the first chart. Parsed while the
        page loaded, they held the GIL and delayed the server's start and the page's
        own requests by about 100 ms. Outside a request the parse starts at once.
        """
        if self._images_parse_started:
            return
        self._images_parse_started = True
        parse = self.available_images.parse_in_background
        if not flask.has_request_context():
            parse()
            return

        @flask.after_this_request
        def parse_once_sent(response):
            response.call_on_close(parse)
            return response

    # ---- layout (dash_metric_layout) ---------------------------------------------

    def _setup_layout(self) -> None:
        """Build the page: shell chrome around the metric controls and chart."""
        self.app.layout = metric_layout.page(
            self.shell,
            self._tube_options,
            self._date_options,
            notice=exclusions_note(self.data_processor),
        )

    # ---- callbacks (dash_metric_callbacks) ---------------------------------------

    def _setup_callbacks(self) -> None:
        """Register every callback of the dashboard."""
        metric_callbacks.register_callbacks(self)
