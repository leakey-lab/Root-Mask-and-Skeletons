"""Root-area Dash visualization window."""
import logging
import os
from datetime import datetime

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl, QThread, pyqtSignal, Qt, QTimer
import socket
import requests

from .dash_app_area import DashAppArea
from ._viz_server import _DashServerThreadBase
from app.data_processing.data_processor_area import DataProcessorArea
from app.config import DASH_AREA_PORT

logger = logging.getLogger(__name__)


class VisualizationInitWorkerArea(QThread):
    """Worker thread to initialize DataProcessorArea and DashAppArea without blocking the GUI."""

    finished = pyqtSignal(object, object)  # processor, dash_app
    error = pyqtSignal(str)

    def __init__(self, csv_path, save_directory):
        super().__init__()
        self.csv_path = csv_path
        self.save_directory = save_directory

    def run(self) -> None:
        try:
            processor = DataProcessorArea(self.csv_path)

            if processor.df.empty:
                raise ValueError("No data found in CSV file")

            dash_app = DashAppArea(processor, self.save_directory)

            self.finished.emit(processor, dash_app)
        except Exception as exc:  # noqa: BLE001
            logger.exception("VisualizationInitWorkerArea failed")
            self.error.emit(str(exc))


class DashServerThreadArea(_DashServerThreadBase):
    """Werkzeug server thread for the root-area Dash app (port %d)."""

    def __init__(self, dash_app, port: int = DASH_AREA_PORT):
        super().__init__(dash_app, port)


class RootAreaVisualization(QMainWindow):
    server_closed = pyqtSignal()

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.setWindowTitle("Root Area Visualization")
        self.setGeometry(100, 100, 1200, 800)

        self.server_thread = None
        self.check_server_timer = None
        self.server_active = False
        self.port_check_attempts = 0
        self.max_port_check_attempts = 10
        self.save_directory = os.path.dirname(os.path.abspath(csv_path))
        self.port = DASH_AREA_PORT
        self.init_worker = None
        self.processor = None
        self.dash_app = None
        # F-020: guard flag — _show_visualization must only fire once per session.
        self._visualization_shown = False

        self._init_ui()
        QTimer.singleShot(100, self._start_visualization)

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        self.loading_label = QLabel("Initializing visualization...\nPlease wait...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.loading_label)

        try:
            self.web_view = QWebEngineView()
            self.web_view.hide()
            self.web_view.page().profile().downloadRequested.connect(self.handle_download)
            self.layout.addWidget(self.web_view)
        except Exception as exc:  # noqa: BLE001
            logger.error("WebEngine init failed: %s", exc)
            self.loading_label.setText(f"Error initializing WebEngine: {exc}")
            self.web_view = None

    def _is_port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def _start_visualization(self) -> None:
        if not self._is_port_available(self.port):
            self.port_check_attempts += 1
            if self.port_check_attempts < self.max_port_check_attempts:
                QTimer.singleShot(500, self._start_visualization)
                return
            error_msg = f"Error: Port {self.port} is in use.\nPlease try again later."
            self.loading_label.setText(error_msg)
            return

        try:
            self.loading_label.setText("Loading data...\nThis may take a moment...")

            self.init_worker = VisualizationInitWorkerArea(
                self.csv_path, self.save_directory
            )
            self.init_worker.finished.connect(self._on_init_finished)
            self.init_worker.error.connect(self._handle_initialization_error)
            self.init_worker.start()

        except Exception as exc:  # noqa: BLE001
            logger.exception("_start_visualization failed")
            self._handle_initialization_error(str(exc))

    def _on_init_finished(self, processor, dash_app) -> None:
        """Called when worker thread finishes initialization."""
        try:
            self.processor = processor
            self.dash_app = dash_app

            self.loading_label.setText("Starting server...\nPlease wait...")

            self.server_thread = DashServerThreadArea(self.dash_app, port=self.port)
            self.server_thread.error.connect(self._handle_server_error)
            self.server_thread.port_assigned.connect(self._on_port_assigned)
            self.server_thread.start()

            self.check_server_timer = QTimer(self)
            self.check_server_timer.timeout.connect(self._check_server)
            self.check_server_timer.start(100)

            self.server_active = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("_on_init_finished failed")
            self._handle_initialization_error(str(exc))

    def _on_port_assigned(self, port: int) -> None:
        # F-020: port_assigned fires as soon as the socket is bound — the
        # server may not yet be serving HTTP.  Delegate to _check_server
        # (already polling) and rely solely on the first successful HTTP 200
        # to call _show_visualization once.
        logger.debug("Port %d assigned; waiting for HTTP readiness poll", port)

    def _handle_server_error(self, error_msg: str) -> None:
        logger.error("Dash server error: %s", error_msg)
        self.loading_label.setText(f"Server Error:\n{error_msg}")
        self.cleanup_server()

    def _handle_initialization_error(self, error_msg: str) -> None:
        logger.error("Initialization error: %s", error_msg)
        self.loading_label.setText(f"Initialization Error:\n{error_msg}")
        QMessageBox.critical(
            self, "Error", f"Failed to initialize visualization: {error_msg}"
        )
        self.cleanup_server()

    def _check_server(self) -> None:
        """Poll until server is ready, then show visualization exactly once (F-020)."""
        try:
            url = f"http://localhost:{self.port}"
            response = requests.get(url, timeout=0.1)
            if response.status_code == 200:
                self.check_server_timer.stop()
                self._show_visualization()
        except requests.exceptions.RequestException:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("_check_server poll error: %s", exc)

    def _show_visualization(self) -> None:
        """Load the Dash URL into the web view — idempotent (F-020)."""
        if self._visualization_shown:
            logger.debug("_show_visualization: already shown, skipping")
            return
        self._visualization_shown = True

        try:
            if self.web_view:
                url = f"http://localhost:{self.port}"
                self.web_view.load(QUrl(url))
                QTimer.singleShot(500, self.loading_label.hide)
                QTimer.singleShot(500, self.web_view.show)
            else:
                self._handle_initialization_error("Web view not initialized")
        except Exception as exc:  # noqa: BLE001
            logger.exception("_show_visualization failed")
            self._handle_initialization_error(str(exc))

    def cleanup_server(self) -> None:
        # Stop initialization worker if still running
        if self.init_worker and self.init_worker.isRunning():
            self.init_worker.terminate()
            self.init_worker.wait(1000)
            self.init_worker = None

        if self.server_thread:
            if self.check_server_timer:
                self.check_server_timer.stop()
                self.check_server_timer = None

            if self.web_view:
                self.web_view.setUrl(QUrl("about:blank"))
                self.web_view.hide()

            self.server_thread.stop()
            # Memory fix: join the thread so the OS reclaims the port before
            # the caller proceeds (server_close alone does not block).
            self.server_thread.wait(3000)
            self.server_thread = None

            self.server_active = False
            self._visualization_shown = False

            self.server_closed.emit()

    def closeEvent(self, event) -> None:
        try:
            self.cleanup_server()
        except Exception as exc:  # noqa: BLE001
            logger.exception("closeEvent cleanup failed")
        event.accept()

    def handle_download(self, download_item) -> None:
        try:
            filename = download_item.suggestedFileName()
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"root_area_plot_{timestamp}.png"

            filename = os.path.basename(filename)
            save_path = os.path.join(self.save_directory, filename)

            base, ext = os.path.splitext(save_path)
            counter = 1
            while os.path.exists(save_path):
                save_path = f"{base}_{counter}{ext}"
                counter += 1

            download_item.setDownloadDirectory(os.path.dirname(save_path))
            download_item.setDownloadFileName(os.path.basename(save_path))
            download_item.accept()

        except OSError as exc:
            logger.error("handle_download OSError: %s", exc)
            QMessageBox.warning(self, "Download Error", "Failed to save file.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("handle_download unexpected error")
            QMessageBox.warning(self, "Download Error", "Failed to save file.")
