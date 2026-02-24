import os
from datetime import datetime
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl, QThread, pyqtSignal, Qt, QTimer
import socket
from werkzeug.serving import make_server
import requests
from .dash_app import DashApp
from app.data_processing.data_processor import DataProcessor
import threading


class VisualizationInitWorker(QThread):
    """Worker thread to initialize DataProcessor and DashApp without blocking the GUI."""
    finished = pyqtSignal(object, object)  # processor, dash_app
    error = pyqtSignal(str)
    
    def __init__(self, csv_path, save_directory, image_manager):
        super().__init__()
        self.csv_path = csv_path
        self.save_directory = save_directory
        self.image_manager = image_manager
    
    def run(self):
        try:
            processor = DataProcessor(self.csv_path)
            
            if processor.df.empty:
                raise ValueError("No data found in CSV file")
            
            dash_app = DashApp(processor, self.save_directory, self.image_manager)
            
            self.finished.emit(processor, dash_app)
        except Exception as e:
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

    def run(self):
        try:
            self.server = make_server("127.0.0.1", self.port, self.dash_app.app.server)
            self.server.timeout = 0.5  # Set timeout to allow checking stop_event
            self.port_assigned.emit(self.port)
            while not self._stop_event.is_set():
                self.server.handle_request()
        except OSError as oe:
            error_msg = f"Port {self.port} unavailable: {oe}"
            self.error.emit(error_msg)
        except Exception as e:
            error_msg = str(e)
            self.error.emit(error_msg)
        finally:
            if self.server:
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

        try:
            self._init_ui()
            QTimer.singleShot(100, self._start_visualization)
        except Exception:
            raise

    def _init_ui(self):
        try:
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
            except Exception as e:
                self.loading_label.setText(f"Error initializing WebEngine: {str(e)}")
                self.web_view = None
        except Exception:
            raise

    def _is_port_available(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except socket.error:
                return False

    def _start_visualization(self):
        if not self._is_port_available(self.port):
            self.port_check_attempts += 1
            if self.port_check_attempts < self.max_port_check_attempts:
                QTimer.singleShot(500, self._start_visualization)
                return
            else:
                error_msg = f"Error: Port {self.port} is in use.\nPlease try again later."
                self.loading_label.setText(error_msg)
                return

        try:
           
            # Update loading message
            self.loading_label.setText("Loading data...\nThis may take a moment...")
            
            # Start worker thread to do heavy initialization
            self.init_worker = VisualizationInitWorker(
                self.csv_path, self.save_directory, self.image_manager,
            )
            self.init_worker.finished.connect(self._on_init_finished)
            self.init_worker.error.connect(self._handle_initialization_error)
            self.init_worker.start()

        except Exception as e:
            self._handle_initialization_error(str(e))
    
    def _on_init_finished(self, processor, dash_app):
        """Called when worker thread finishes initialization."""
        try:
            self.processor = processor
            self.dash_app = dash_app
            
            self.loading_label.setText("Starting server...\nPlease wait...")
            
            self.server_thread = DashServerThread(self.dash_app, port=self.port)
            self.server_thread.error.connect(self._handle_server_error)
            self.server_thread.port_assigned.connect(self._on_port_assigned)
            self.server_thread.start()

            self.check_server_timer = QTimer(self)
            self.check_server_timer.timeout.connect(self._check_server)
            self.check_server_timer.start(100)

            self.server_active = True
        except Exception as e:
            self._handle_initialization_error(str(e))

    def _on_port_assigned(self, port):
        self._show_visualization()

    def _handle_server_error(self, error_msg):
        self.loading_label.setText(f"Server Error:\n{error_msg}")
        self.cleanup_server()

    def _handle_initialization_error(self, error_msg):
        self.loading_label.setText(f"Initialization Error:\n{error_msg}")
        QMessageBox.critical(
            self, "Error", f"Failed to initialize visualization: {error_msg}"
        )
        self.cleanup_server()

    def _check_server(self):
        try:
            url = f"http://localhost:{self.port}"
            response = requests.get(url, timeout=0.1)
            if response.status_code == 200:
                self.check_server_timer.stop()
                self._show_visualization()
            else:
                pass
        except requests.exceptions.RequestException:
            pass
        except Exception:
            pass

    def _show_visualization(self):
        try:
            if self.web_view:
                url = f"http://localhost:{self.port}"
                self.web_view.load(QUrl(url))
                QTimer.singleShot(500, self.loading_label.hide)
                QTimer.singleShot(500, self.web_view.show)
            else:
                self._handle_initialization_error("Web view not initialized")
        except Exception as e:
            self._handle_initialization_error(str(e))

    def cleanup_server(self):
        
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
            self.server_thread.wait(1000)
            self.server_thread = None

            self.server_active = False

            self.server_closed.emit()

    def closeEvent(self, event):
        try:
            self.cleanup_server()
            event.accept()
        except Exception:
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
