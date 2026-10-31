"""
Root-length dashboard (Dash): one app over one root_lengths.csv.

:class:`DashApp` builds the app and holds what its parts share: the data
(:class:`.dash_data_cache.DataCache`), the figures (the
:class:`.dash_visualizations.DashVisualizations` mixin), the hover-image index and
its thumbnail server. The parts live in their own modules:

- :mod:`.dash_length_layout`: the page (control cards, chart row, stores);
- :mod:`.dash_length_views`: what each view shows for a selection;
- :mod:`.dash_length_callbacks`: the callbacks that wire them to the page.

The page chrome comes from :mod:`.dash_shell`, hover thumbnails from
:mod:`.dash_image_utils`. Component ids, callback Outputs and figure customdata
are a contract with the renderer and the embedding window: keep them stable.
"""

from typing import Any

import flask

from . import dash_length_callbacks as length_callbacks
from . import dash_length_layout as length_layout
from . import theme
from .dash_data_cache import DataCache
from .dash_image_utils import ThumbnailServer, build_available_images_map
from .dash_length_layout import LENGTH_SHELL
from .dash_shell import DashboardShell, create_dash_app, exclusions_note, tube_options
from .dash_visualizations import DashVisualizations

theme.use("sprouts")


class DashApp(DashVisualizations):
    """The root-length dashboard: one Dash app over one root_lengths.csv.

    The page is built by :mod:`.dash_length_layout` and the callbacks call
    :mod:`.dash_length_views` directly; this class holds what they share.
    """

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

        self.shell = DashboardShell(LENGTH_SHELL)
        self.app = create_dash_app(__name__, title="SPROUTS — Root Length")
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

    # ---- layout (dash_length_layout) ---------------------------------------------

    def _setup_layout(self) -> None:
        """Build the page: shell chrome around the length controls and chart."""
        self.app.layout = length_layout.page(
            self.shell,
            self._tube_options,
            self._date_options,
            notice=exclusions_note(self.data_processor),
        )

    # ---- callbacks (dash_length_callbacks) ---------------------------------------

    def _setup_callbacks(self) -> None:
        """Register every callback of the dashboard."""
        length_callbacks.register_callbacks(self)
