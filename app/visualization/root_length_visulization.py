import os
import logging
from datetime import datetime
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox, QFileDialog
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl, QThread, pyqtSignal, Qt, QTimer
import socket
from werkzeug.serving import make_server
import requests
from .dash_app import DashApp
from app.data_processing.data_processor import DataProcessor
import threading

logger = logging.getLogger(__name__)


class VisualizationInitWorker(QThread):
    """Worker thread to initialize DataProcessor and DashApp without blocking the GUI."""
    finished = pyqtSignal(object, object)  # processor, dash_app
    error = pyqtSignal(str)
    
    def __init__(self, csv_path, save_directory, image_manager, field_map_path=None, tube_ids_path=None):
        super().__init__()
        self.csv_path = csv_path
        self.save_directory = save_directory
        self.image_manager = image_manager
        self.field_map_path = field_map_path
        self.tube_ids_path = tube_ids_path
    
    def run(self):
        try:
            logger.debug(f"Worker thread: Initializing DataProcessor with csv_path={self.csv_path}")
            if self.field_map_path and os.path.exists(self.field_map_path) and self.tube_ids_path and os.path.exists(self.tube_ids_path):
                logger.debug(f"Worker thread: Using field_map_path={self.field_map_path}, tube_ids_path={self.tube_ids_path}")
                processor = DataProcessor(self.csv_path, self.field_map_path, self.tube_ids_path)
            else:
                logger.debug("Worker thread: Initializing DataProcessor without field map")
                processor = DataProcessor(self.csv_path)
            
            logger.debug(f"Worker thread: DataProcessor initialized, df shape: {processor.df.shape}")
            if processor.df.empty:
                raise ValueError("No data found in CSV file")
            
            logger.debug("Worker thread: Creating DashApp")
            dash_app = DashApp(processor, self.save_directory, self.image_manager)
            logger.info("Worker thread: DashApp created successfully")
            
            self.finished.emit(processor, dash_app)
        except Exception as e:
            logger.error(f"Worker thread error: {e}", exc_info=True)
            self.error.emit(str(e))


class DashServerThread(QThread):
    error = pyqtSignal(str)
    port_assigned = pyqtSignal(int)

    def __init__(self, dash_app, port=8050):
        super().__init__()
        self.dash_app = dash_app
        self.server = None
        self.port = port
        self._stop_event = threading.Event()
        logger.debug(f"DashServerThread initialized with port {port}")

    def run(self):
        logger.debug(f"DashServerThread.run() started for port {self.port}")
        try:
            logger.debug(f"Creating server on 127.0.0.1:{self.port}")
            self.server = make_server("127.0.0.1", self.port, self.dash_app.app.server)
            self.server.timeout = 0.5  # Set timeout to allow checking stop_event
            logger.info(f"Server created successfully on port {self.port}")
            self.port_assigned.emit(self.port)
            logger.debug("Starting server request handling loop")
            while not self._stop_event.is_set():
                self.server.handle_request()
            logger.debug("Server request handling loop ended")
        except OSError as oe:
            error_msg = f"Port {self.port} unavailable: {oe}"
            logger.error(error_msg, exc_info=True)
            self.error.emit(error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error in server thread: {error_msg}", exc_info=True)
            self.error.emit(error_msg)
        finally:
            if self.server:
                logger.debug("Closing server")
                self.server.server_close()

    def stop(self):
        if self.server:
            self._stop_event.set()
            try:
                # Close the server socket to release the port
                self.server.server_close()
            except Exception:
                pass
            self.server = None


class RootLengthVisualization(QMainWindow):
    server_closed = pyqtSignal()

    def __init__(self, csv_path, image_manager):
        logger.info(f"RootLengthVisualization.__init__ called with csv_path={csv_path}")
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
        self.port = 8050
        self.init_worker = None
        self.processor = None
        self.dash_app = None
        logger.debug(f"Initialized with port={self.port}, save_directory={self.save_directory}")

        try:
            logger.debug("Initializing UI")
            self._init_ui()
            logger.debug("Scheduling _start_visualization in 100ms")
            QTimer.singleShot(100, self._start_visualization)
        except Exception as e:
            logger.error(f"Error in __init__: {e}", exc_info=True)
            raise

    def _init_ui(self):
        logger.debug("_init_ui() called")
        try:
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            self.layout = QVBoxLayout(central_widget)

            self.loading_label = QLabel("Initializing visualization...\nPlease wait...")
            self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.layout.addWidget(self.loading_label)
            logger.debug("Loading label created and added")

            try:
                logger.debug("Creating QWebEngineView")
                self.web_view = QWebEngineView()
                self.web_view.hide()
                logger.debug("Connecting download handler")
                self.web_view.page().profile().downloadRequested.connect(self.handle_download)
                self.layout.addWidget(self.web_view)
                logger.info("QWebEngineView initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing WebEngine: {e}", exc_info=True)
                self.loading_label.setText(f"Error initializing WebEngine: {str(e)}")
                self.web_view = None
        except Exception as e:
            logger.error(f"Error in _init_ui: {e}", exc_info=True)
            raise

    def _is_port_available(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except socket.error:
                return False

    def _start_visualization(self):
        logger.info(f"_start_visualization() called, port_check_attempts={self.port_check_attempts}")
        if not self._is_port_available(self.port):
            logger.warning(f"Port {self.port} is not available")
            self.port_check_attempts += 1
            if self.port_check_attempts < self.max_port_check_attempts:
                logger.debug(f"Retrying port check in 500ms (attempt {self.port_check_attempts}/{self.max_port_check_attempts})")
                QTimer.singleShot(500, self._start_visualization)
                return
            else:
                error_msg = f"Error: Port {self.port} is in use.\nPlease try again later."
                logger.error(error_msg)
                self.loading_label.setText(error_msg)
                return

        logger.info(f"Port {self.port} is available, proceeding with initialization")
        try:
            # Locate Experimental Data Files
            base_dir = os.path.dirname(self.csv_path)
            logger.debug(f"Base directory: {base_dir}")
            field_map_path = os.path.join(base_dir, "FieldMap.xlsx")
            tube_ids_path = os.path.join(base_dir, "TubeIDS.csv")
            
            # Fallback to common locations if not in CSV dir
            if not os.path.exists(field_map_path):
                 field_map_path = os.path.abspath("field docs ex/FieldMap.xlsx")
                 logger.debug(f"FieldMap not found in base_dir, trying: {field_map_path}")
            if not os.path.exists(tube_ids_path):
                 tube_ids_path = os.path.abspath("field docs ex/TubeIDS.csv")
                 logger.debug(f"TubeIDS not found in base_dir, trying: {tube_ids_path}")

            # If still not found, ask user (optional)
            if not os.path.exists(field_map_path) or not os.path.exists(tube_ids_path):
                 logger.debug("FieldMap or TubeIDS not found, asking user")
                 reply = QMessageBox.question(self, 'Experimental Data', 
                     "Do you want to load FieldMap.xlsx and TubeIDS.csv for experimental analysis?",
                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                 
                 if reply == QMessageBox.StandardButton.Yes:
                     if not os.path.exists(field_map_path):
                         path, _ = QFileDialog.getOpenFileName(self, "Select FieldMap.xlsx", base_dir, "Excel Files (*.xlsx)")
                         if path:
                             field_map_path = path
                     
                     if not os.path.exists(tube_ids_path):
                         path, _ = QFileDialog.getOpenFileName(self, "Select TubeIDS.csv", base_dir, "CSV Files (*.csv)")
                         if path:
                             tube_ids_path = path
            
            # Update loading message
            self.loading_label.setText("Loading data...\nThis may take a moment...")
            
            # Start worker thread to do heavy initialization
            logger.debug("Starting initialization worker thread")
            self.init_worker = VisualizationInitWorker(
                self.csv_path, self.save_directory, self.image_manager, 
                field_map_path if os.path.exists(field_map_path) else None,
                tube_ids_path if os.path.exists(tube_ids_path) else None
            )
            self.init_worker.finished.connect(self._on_init_finished)
            self.init_worker.error.connect(self._handle_initialization_error)
            self.init_worker.start()

        except Exception as e:
            logger.error(f"Error in _start_visualization: {e}", exc_info=True)
            self._handle_initialization_error(str(e))
    
    def _on_init_finished(self, processor, dash_app):
        """Called when worker thread finishes initialization."""
        logger.info("Initialization worker finished successfully")
        try:
            self.processor = processor
            self.dash_app = dash_app
            
            self.loading_label.setText("Starting server...\nPlease wait...")
            
            logger.debug(f"Creating DashServerThread on port {self.port}")
            self.server_thread = DashServerThread(self.dash_app, port=self.port)
            self.server_thread.error.connect(self._handle_server_error)
            self.server_thread.port_assigned.connect(self._on_port_assigned)
            logger.debug("Starting server thread")
            self.server_thread.start()

            logger.debug("Setting up server check timer")
            self.check_server_timer = QTimer(self)
            self.check_server_timer.timeout.connect(self._check_server)
            self.check_server_timer.start(100)

            self.server_active = True
            logger.info("Visualization startup sequence completed")
        except Exception as e:
            logger.error(f"Error after initialization: {e}", exc_info=True)
            self._handle_initialization_error(str(e))

    def _on_port_assigned(self, port):
        logger.info(f"_on_port_assigned called with port={port}")
        self._show_visualization()

    def _handle_server_error(self, error_msg):
        logger.error(f"_handle_server_error called: {error_msg}")
        self.loading_label.setText(f"Server Error:\n{error_msg}")
        self.cleanup_server()

    def _handle_initialization_error(self, error_msg):
        logger.error(f"_handle_initialization_error called: {error_msg}")
        self.loading_label.setText(f"Initialization Error:\n{error_msg}")
        QMessageBox.critical(
            self, "Error", f"Failed to initialize visualization: {error_msg}"
        )
        self.cleanup_server()

    def _check_server(self):
        try:
            url = f"http://localhost:{self.port}"
            logger.debug(f"Checking server at {url}")
            response = requests.get(url, timeout=0.1)
            if response.status_code == 200:
                logger.info(f"Server is responding with status 200 at {url}")
                self.check_server_timer.stop()
                self._show_visualization()
            else:
                logger.debug(f"Server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.debug(f"Server not ready yet: {e}")
        except Exception as e:
            logger.debug(f"Error checking server: {e}")

    def _show_visualization(self):
        logger.info("_show_visualization() called")
        try:
            if self.web_view:
                url = f"http://localhost:{self.port}"
                logger.info(f"Loading URL in web view: {url}")
                self.web_view.load(QUrl(url))
                QTimer.singleShot(500, self.loading_label.hide)
                QTimer.singleShot(500, self.web_view.show)
                logger.info("Web view load initiated")
            else:
                logger.error("Web view is None, cannot show visualization")
                self._handle_initialization_error("Web view not initialized")
        except Exception as e:
            logger.error(f"Error in _show_visualization: {e}", exc_info=True)
            self._handle_initialization_error(str(e))

    def cleanup_server(self):
        logger.info("cleanup_server() called")
        
        # Stop initialization worker if still running
        if self.init_worker and self.init_worker.isRunning():
            logger.debug("Stopping initialization worker")
            self.init_worker.terminate()
            self.init_worker.wait(1000)
            self.init_worker = None
        
        if self.server_thread:
            if self.check_server_timer:
                logger.debug("Stopping server check timer")
                self.check_server_timer.stop()
                self.check_server_timer = None

            if self.web_view:
                logger.debug("Clearing web view")
                self.web_view.setUrl(QUrl("about:blank"))
                self.web_view.hide()

            logger.debug("Stopping server thread")
            self.server_thread.stop()
            logger.debug("Waiting for server thread to finish")
            self.server_thread.wait(1000)
            self.server_thread = None

            self.server_active = False
            logger.info("Server cleanup completed")

            self.server_closed.emit()

    def closeEvent(self, event):
        logger.info("closeEvent() called")
        try:
            self.cleanup_server()
            event.accept()
        except Exception as e:
            logger.error(f"Error in closeEvent: {e}", exc_info=True)
            event.accept()

    def handle_download(self, download_item):
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

        except Exception:
            QMessageBox.warning(self, "Download Error", "Failed to save file.")
