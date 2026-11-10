"""Stand-ins of the dashboard window tests: the QtWebEngine profile, page and view that
``DashVizWindow`` uses, a WSGI stub of a Dash app, and a test dashboard on them.

A COPY of ``test_viz_embed_window.py``'s fakes (that module is frozen in wave 1; wave 2
imports them from here and drops its own), for ``test_qt_environment_linux_quick.py``.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtWidgets import QWidget

from app.visualization._viz_server import _DashServerThreadBase
from app.visualization.viz_window import DashVizWindow


# --------------------------------------------------------------------------- #
#  Fakes for the QtWebEngine parts
# --------------------------------------------------------------------------- #
class FakeProfile(QObject):
    """Stands in for QWebEngineProfile.defaultProfile(), which every view shares."""

    downloadRequested = pyqtSignal(object)


class FakePage(QObject):
    #: QWebEnginePage.renderProcessTerminated(status, exit code)
    renderProcessTerminated = pyqtSignal(object, int)

    def __init__(self, profile: FakeProfile):
        super().__init__()
        self._profile = profile

    def profile(self) -> FakeProfile:
        return self._profile

    def setBackgroundColor(self, _color) -> None:
        pass


class FakeWebView(QWidget):
    """The QWebEngineView calls DashVizWindow makes; records every load()."""

    loadFinished = pyqtSignal(bool)

    def __init__(self, profile: FakeProfile):
        super().__init__()
        self._page = FakePage(profile)
        self._url = QUrl()
        self.loads: list[str] = []

    def page(self) -> FakePage:
        return self._page

    def load(self, url: QUrl) -> None:
        self.loads.append(url.toString())
        self._url = QUrl(url)

    def setUrl(self, url: QUrl) -> None:
        self._url = QUrl(url)

    def url(self) -> QUrl:
        return QUrl(self._url)


class StubDash:
    """What the server thread uses of a DashApp: ``.app.server``, a WSGI callable."""

    class app:  # noqa: N801 - mirrors DashApp.app
        @staticmethod
        def server(environ, start_response):
            start_response("200 OK", [("Content-Type", "text/html")])
            return [b"<html><body>dashboard</body></html>"]


class AnyPortServer(_DashServerThreadBase):
    def __init__(self, dash_app, port: int = 0):
        super().__init__(dash_app, port, object_name="TestDashServer")


class _Dashboard(DashVizWindow):
    window_title = "Test dashboard"
    download_prefix = "test_plot"
    server_thread_class = AnyPortServer

    def __init__(self, csv_path, profile, build=None, **overrides):
        self._build = build or (lambda: (object(), StubDash()))
        self._profile = profile
        for name, value in overrides.items():  # per-test hooks, e.g. server_thread_class
            setattr(self, name, value)
        super().__init__(csv_path)

    def _dashboard_builder(self):
        return self._build

    def _create_web_view(self):
        return FakeWebView(self._profile)
