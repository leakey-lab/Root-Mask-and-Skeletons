import os
from datetime import datetime
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl, QThread, pyqtSignal, Qt
from PyQt6.QtCore import QTimer
import socket
from werkzeug.serving import make_server
import requests
from dash_app import DashApp
from data_processor import DataProcessor


# -----------------------------------
# Dash Server Thread Class
# -----------------------------------
class DashServerThread(QThread):
    """QThread to run Dash server without blocking PyQt."""

    error = pyqtSignal(str)

    def __init__(self, dash_app):
        super().__init__()
        self.dash_app = dash_app
        self.server = None
        self._is_running = False

    def run(self):
        try:
            self._is_running = True
            self.server = make_server("127.0.0.1", 8050, self.dash_app.app.server)
            self.server.serve_forever()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._is_running = False

    def stop(self):
        """Safely stop the server."""
        if self.server and self._is_running:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception as e:
                print(f"Error stopping server: {e}")
            finally:
                self.server = None
                self._is_running = False


# -----------------------------------
# PyQt Main Window Class
# -----------------------------------
class RootLengthVisualization(QMainWindow):
    server_ready = pyqtSignal()
    server_closed = pyqtSignal()

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.setWindowTitle("Root Length Visualization")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize state tracking
        self.server_thread = None
        self.check_server_timer = None
        self.server_active = False
        self.port_check_attempts = 0
        self.max_port_check_attempts = 10
        self.save_directory = os.path.dirname(os.path.abspath(csv_path))

        # Set up UI
        self._init_ui()

        # Start visualization with delay
        QTimer.singleShot(100, self._start_visualization)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Create loading label
        self.loading_label = QLabel("Initializing visualization...\nPlease wait...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet(
            """
            QLabel {
                font-size: 16px;
                color: #666;
                padding: 20px;
                background-color: #f0f0f0;
                border-radius: 10px;
            }
        """
        )
        self.layout.addWidget(self.loading_label)

        # Initialize WebView
        self.web_view = QWebEngineView()
        self.web_view.hide()
        self.web_view.page().profile().downloadRequested.connect(self.handle_download)
        self.layout.addWidget(self.web_view)

    def _is_port_available(self, port):
        """Check if the port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except socket.error:
                return False

    def _start_visualization(self):
        """Initialize and start the visualization server with retry logic."""
        if not self._is_port_available(8050):
            self.port_check_attempts += 1
            if self.port_check_attempts < self.max_port_check_attempts:
                QTimer.singleShot(500, self._start_visualization)
                return
            else:
                self.loading_label.setText(
                    "Error: Port 8050 is in use.\nPlease try again later."
                )
                return

        try:
            # Initialize Data Processor
            self.processor = DataProcessor(self.csv_path)
            if self.processor.df.empty:
                raise ValueError("No data found in CSV file")

            # Initialize Dash App
            self.dash_app = DashApp(self.processor, self.save_directory)

            # Start server
            self.server_thread = DashServerThread(self.dash_app)
            self.server_thread.error.connect(self._handle_server_error)
            self.server_thread.start()

            # Start server check
            self.check_server_timer = QTimer(self)
            self.check_server_timer.timeout.connect(self._check_server)
            self.check_server_timer.start(100)

            self.server_active = True

        except Exception as e:
            self._handle_initialization_error(str(e))

    def _check_server(self):
        """Check if the server is responding."""
        try:
            response = requests.get("http://localhost:8050", timeout=0.1)
            if response.status_code == 200:
                self.check_server_timer.stop()
                self._show_visualization()
        except:
            pass

    def _show_visualization(self):
        """Show the visualization once the server is ready."""
        try:
            self.web_view.load(QUrl("http://localhost:8050"))
            QTimer.singleShot(500, lambda: self.loading_label.hide())
            QTimer.singleShot(500, lambda: self.web_view.show())
        except Exception as e:
            self._handle_initialization_error(str(e))

    def cleanup_server(self):
        """Clean up server resources."""
        if not hasattr(self, "server_thread") or not self.server_thread:
            self.server_closed.emit()
            return

        try:
            # Stop check timer
            if self.check_server_timer:
                self.check_server_timer.stop()
                self.check_server_timer = None

            # Clear web view
            if hasattr(self, "web_view"):
                self.web_view.setUrl(QUrl("about:blank"))
                self.web_view.hide()

            # Stop server
            if self.server_thread:
                self.server_thread.stop()
                self.server_thread.wait(1000)
                if self.server_thread.isRunning():
                    self.server_thread.terminate()
                self.server_thread = None

            self.server_active = False

        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.server_closed.emit()

    def closeEvent(self, event):
        """Handle window close event."""
        try:
            self.cleanup_server()
            event.accept()
        except Exception as e:
            print(f"Error in closeEvent: {e}")
            event.accept()

    def _handle_server_error(self, error_msg):
        """Handle server errors."""
        self.loading_label.setText(f"Server Error:\n{error_msg}")
        self.cleanup_server()

    def _handle_initialization_error(self, error_msg):
        """Handle initialization errors."""
        self.loading_label.setText(f"Initialization Error:\n{error_msg}")
        QMessageBox.critical(
            self, "Error", f"Failed to initialize visualization: {error_msg}"
        )
        self.cleanup_server()

    def handle_download(self, download):
        """Handle download requests from the Dash app."""
        try:
            # Get suggested filename
            filename = download.suggestedFileName()
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"root_length_plot_{timestamp}.png"

            # Ensure filename is safe
            filename = os.path.basename(filename)

            # Create full path
            save_path = os.path.join(self.save_directory, filename)

            # Ensure unique filename
            base, ext = os.path.splitext(save_path)
            counter = 1
            while os.path.exists(save_path):
                save_path = f"{base}_{counter}{ext}"
                counter += 1

            # Set the download path
            download.setDownloadDirectory(os.path.dirname(save_path))
            download.setDownloadFileName(os.path.basename(save_path))
            download.accept()

        except Exception as e:
            QMessageBox.warning(
                self, "Download Error", f"Failed to save file: {str(e)}"
            )
