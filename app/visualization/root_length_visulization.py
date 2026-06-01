"""Root-length Dash visualization window (note: filename spelling preserved for
import compatibility — other modules import from this exact module name)."""
import logging
import os
from datetime import datetime

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl, QThread, pyqtSignal, Qt, QTimer
import socket
import requests

from .dash_app import DashApp
from ._viz_server import _DashServerThreadBase, join_qthread
from app.data_processing.data_processor import DataProcessor
from app.config import DASH_LENGTH_PORT

logger = logging.getLogger(__name__)


class VisualizationInitWorker(QThread):
    """Worker thread to initialize DataProcessor and DashApp without blocking the GUI."""

    finished = pyqtSignal(object, object)  # processor, dash_app
    error = pyqtSignal(str)

    def __init__(self, csv_path, save_directory, image_manager):
        super().__init__()
        self.csv_path = csv_path
        self.save_directory = save_directory
        self.image_manager = image_manager

    def run(self) -> None:
        try:
            processor = DataProcessor(self.csv_path)

            if processor.df.empty:
                raise ValueError("No data found in CSV file")

            dash_app = DashApp(processor, self.save_directory, self.image_manager)

            self.finished.emit(processor, dash_app)
        except Exception as exc:  # noqa: BLE001
            logger.exception("VisualizationInitWorker failed")
            self.error.emit(str(exc))


class DashServerThread(_DashServerThreadBase):
    """Werkzeug server thread for the root-length Dash app (port %d)."""

    def __init__(self, dash_app, port: int = DASH_LENGTH_PORT):
        super().__init__(dash_app, port, object_name="DashServerThreadLength")


class RootLengthVisualization(QMainWindow):
    server_closed = pyqtSignal()

    def __init__(self, csv_path, image_manager):
        super().__init__()
        self.csv_path = csv_path
        self.image_manager = image_manager
        self.setWindowTitle("Root Length Visualization")
        self.setGeometry(100, 100, 1200, 800)

        self.server_thread = None
        self.check_server_timer = None
        self.server_active = False
        self.port_check_attempts = 0
        self.max_port_check_attempts = 10
        self.save_directory = os.path.dirname(os.path.abspath(csv_path))
        self.port = DASH_LENGTH_PORT
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
        logger.info("DBG RootLengthViz._start_visualization: port=%d, attempt=%d",
                    self.port, self.port_check_attempts)
        if not self._is_port_available(self.port):
            self.port_check_attempts += 1
            logger.info(
                "DBG RootLengthViz: port %d busy, retry %d/%d",
                self.port, self.port_check_attempts, self.max_port_check_attempts,
            )
            if self.port_check_attempts < self.max_port_check_attempts:
                QTimer.singleShot(500, self._start_visualization)
                return
            logger.error(
                "Port %d still busy after %d attempts, giving up",
                self.port, self.max_port_check_attempts,
            )
            error_msg = f"Error: Port {self.port} is in use.\nPlease try again later."
            self.loading_label.setText(error_msg)
            return

        try:
            self.loading_label.setText("Loading data...\nThis may take a moment...")

            self.init_worker = VisualizationInitWorker(
                self.csv_path, self.save_directory, self.image_manager,
            )
            self.init_worker.setObjectName("VisualizationInitWorker")
            self.init_worker.setParent(self)
            self.init_worker.finished.connect(self._on_init_finished)
            self.init_worker.error.connect(self._handle_initialization_error)
            self.init_worker.start()
            logger.info("DBG RootLengthViz: init worker started")

        except Exception as exc:  # noqa: BLE001
            logger.exception("_start_visualization failed")
            self._handle_initialization_error(str(exc))

    def _on_init_finished(self, processor, dash_app) -> None:
        """Called when worker thread finishes initialization."""
        logger.info("DBG RootLengthViz._on_init_finished: processor=%s, dash_app=%s",
                    processor, dash_app)
        try:
            # Worker is done; release it now so it is not torn down with the widget.
            self._join_init_worker()

            self.processor = processor
            self.dash_app = dash_app

            self.loading_label.setText("Starting server...\nPlease wait...")

            self.server_thread = DashServerThread(self.dash_app, port=self.port)
            self.server_thread.setParent(self)
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
            logger.info("DBG RootLengthViz._check_server: status=%d", response.status_code)
            if response.status_code == 200:
                self.check_server_timer.stop()
                self._show_visualization()
        except requests.exceptions.RequestException as exc:
            logger.debug("Server not ready yet (port %d): %s", self.port, exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("_check_server poll error: %s", exc)

    def _show_visualization(self) -> None:
        """Load the Dash URL into the web view — idempotent (F-020)."""
        logger.info("DBG RootLengthViz._show_visualization: _visualization_shown=%s, web_view=%s",
                    self._visualization_shown, self.web_view)
        if self._visualization_shown:
            logger.info("DBG RootLengthViz._show_visualization: already shown, skipping")
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

    def _join_init_worker(self) -> None:
        """Wait for the init worker to finish and release it safely."""
        if self.init_worker is None:
            return
        worker = self.init_worker
        self.init_worker = None
        for signal in (worker.finished, worker.error):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass
        if join_qthread(worker):
            worker.deleteLater()
        else:
            # Thread still running after timeout; unparent so the window's
            # destruction won't tear down a live QThread.
            worker.setParent(None)
            worker.finished.connect(lambda *_: worker.deleteLater())
            worker.error.connect(lambda *_: worker.deleteLater())

    def _join_server_thread(self) -> None:
        if self.server_thread is None:
            return
        thread = self.server_thread
        self.server_thread = None

        if self.check_server_timer:
            self.check_server_timer.stop()
            self.check_server_timer = None

        if self.web_view:
            self.web_view.setUrl(QUrl("about:blank"))
            self.web_view.hide()

        thread.stop()
        if join_qthread(thread):
            thread.deleteLater()

        self.server_active = False
        self._visualization_shown = False
        self.server_closed.emit()

    def cleanup_server(self) -> None:
        # Join the initialization worker before dropping its reference.
        # run() overrides QThread (no event loop, so quit() is a no-op) and is
        # short-lived, so wait() unconditionally — regardless of isRunning(),
        # which races the fast error path. Never terminate(): force-killing the
        # thread while its C++ object is torn down is exactly what produces
        # "QThread: Destroyed while thread '' is still running".
        self._join_init_worker()
        if self.server_thread is not None:
            self._join_server_thread()

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
                filename = f"root_length_plot_{timestamp}.png"

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
