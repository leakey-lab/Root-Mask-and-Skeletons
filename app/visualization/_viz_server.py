"""
Shared server-thread infrastructure for the Dash visualization windows.

Both RootLengthVisualization and RootAreaVisualization are near-identical
(F-083).  The common QThread lifecycle — create werkzeug server, poll for
stop, join on close — lives here so neither file duplicates it.

Public names re-exported from this module are *not* part of the package API;
all public classes remain in their respective modules.
"""
import logging
import threading

from PyQt6.QtCore import QThread, pyqtSignal

from werkzeug.serving import make_server

logger = logging.getLogger(__name__)


class _DashServerThreadBase(QThread):
    """
    QThread that drives a single-request werkzeug loop.

    Subclasses only need to set a sensible default ``port`` in their
    ``__init__`` signature; everything else is shared.
    """

    error = pyqtSignal(str)
    port_assigned = pyqtSignal(int)

    def __init__(self, dash_app, port: int):
        super().__init__()
        self.dash_app = dash_app
        self.port = port
        self.server = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            self.server = make_server("127.0.0.1", self.port, self.dash_app.app.server)
            self.server.timeout = 0.5
            self.port_assigned.emit(self.port)
            while not self._stop_event.is_set():
                self.server.handle_request()
        except OSError as exc:
            logger.error("DashServerThread port %d unavailable: %s", self.port, exc)
            self.error.emit(f"Port {self.port} unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("DashServerThread unexpected error on port %d", self.port)
            self.error.emit(str(exc))
        finally:
            if self.server:
                try:
                    self.server.server_close()
                except Exception:  # noqa: BLE001
                    pass

    def stop(self) -> None:
        """Signal the loop to exit and close the server socket."""
        self._stop_event.set()
        if self.server:
            try:
                self.server.server_close()
            except Exception:  # noqa: BLE001
                pass
            self.server = None
