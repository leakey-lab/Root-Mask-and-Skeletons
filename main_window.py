from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QLabel,
    QSplitter,
    QStatusBar,
    QMessageBox,
    QComboBox,
    QStackedWidget,
    QFileDialog,
    QListWidgetItem,
    QApplication,
)
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtCore import Qt, pyqtSignal
from image_manager import ImageManager
from display_controller import DisplayController
from skeleton_handler import SkeletonHandler
from mask_handler import MaskHandler
from mask_tracing_interface import MaskTracingInterface
import os

# TODO 1: Make The Green signal from mask saved and chnage the dilename to green. Potentila issue might be how the names are displayed in the QListWidget witrh the .jpg,.png whereas while matching names we are matching the name with even that suffix


class MainWindow(QMainWindow):
    mask_saved = pyqtSignal(str)  # Signal to emit when a mask is saved
    mask_cleared = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Viewer")
        self.setGeometry(100, 100, 1200, 700)

        # Initialize components
        self.image_manager = ImageManager()
        self.display_controller = DisplayController(self)
        self.skeleton_handler = SkeletonHandler(self)
        self.mask_handler = MaskHandler(self)
        self.mask_tracing_interface = MaskTracingInterface()

        # Connect the mask_saved and mask_cleared signals
        self.mask_saved.connect(self.highlight_saved_mask)
        self.mask_cleared.connect(self.unhighlight_cleared_mask)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left Panel
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right Panel (Stacked Widget for Display Area and Mask Tracing)
        self.right_panel = QStackedWidget()
        self.right_panel.addWidget(self.create_right_panel())
        self.right_panel.addWidget(self.mask_tracing_interface)
        splitter.addWidget(self.right_panel)

        splitter.setSizes([400, 800])

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def create_left_panel(self):
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)

        # Buttons
        self.generate_button = QPushButton("Generate Skeleton")
        self.generate_button.clicked.connect(self.skeleton_handler.generate_skeleton)
        layout.addWidget(self.generate_button)

        self.load_images_button = QPushButton("Load Images")
        self.load_images_button.clicked.connect(self.load_images)
        layout.addWidget(self.load_images_button)

        self.calculate_length_button = QPushButton("Calculate Root Length")
        self.calculate_length_button.clicked.connect(self.calculate_root_length)
        layout.addWidget(self.calculate_length_button)

        # Toggle Mask Tracing Interface Button
        self.toggle_mask_tracing_button = QPushButton("Toggle Mask Tracing")
        self.toggle_mask_tracing_button.clicked.connect(self.toggle_mask_tracing)
        layout.addWidget(self.toggle_mask_tracing_button)

        # View Mode ComboBox
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(
            ["Single Image", "Overlay", "Side by Side", "Basic View"]
        )
        self.view_mode_combo.currentIndexChanged.connect(
            self.display_controller.update_display_mode
        )
        layout.addWidget(self.view_mode_combo)

        # Image List
        layout.addWidget(QLabel("Images:"))
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_image_selected)
        layout.addWidget(self.file_list)

        return left_widget

    def highlight_saved_mask(self, image_path):
        print(f"DEBUG: highlight_saved_mask called with image_path: {image_path}")
        image_name = os.path.splitext(os.path.basename(os.path.normpath(image_path)))[0]
        items = self.file_list.findItems(image_name, Qt.MatchFlag.MatchExactly)
        if items:
            item = items[0]
            item.setForeground(QColor("green"))
            print(f"DEBUG: Set {image_name} to green (newly saved mask)")
        else:
            print(f"DEBUG: Could not find item for {image_name}")
        self.status_bar.showMessage(f"Mask saved for {image_name}", 3000)

    def unhighlight_cleared_mask(self, image_path):
        print(f"DEBUG: unhighlight_cleared_mask called with image_path: {image_path}")
        image_name = os.path.splitext(os.path.basename(os.path.normpath(image_path)))[0]
        items = self.file_list.findItems(image_name, Qt.MatchFlag.MatchExactly)
        if items:
            item = items[0]
            item.setForeground(QColor("white"))
            print(f"DEBUG: Set {image_name} to white (cleared mask)")
        else:
            print(f"DEBUG: Could not find item for {image_name}")

    def create_right_panel(self):
        right_widget = QWidget()
        self.display_area = self.display_controller.setup_display_area(right_widget)
        return right_widget

    def load_images(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_name:
            self.image_manager.load_images(dir_name)
            self.populate_file_list()

    def populate_file_list(self):
        self.file_list.clear()
        for name in self.image_manager.get_image_names():
            item = QListWidgetItem(os.path.basename(name))
            self.file_list.addItem(item)
            if self.mask_exists(name) or self.image_manager.has_mask(name):
                item.setForeground(QColor("green"))
                print(f"DEBUG: Set {name} to green (existing mask)")
        self.status_bar.showMessage(
            f"Loaded {len(self.image_manager.images)} images", 5000
        )

    def mask_exists(self, image_name):
        image_path = self.image_manager.get_image_path(image_name)
        mask_dir = os.path.join(os.path.dirname(image_path), "mask")
        mask_path = os.path.join(mask_dir, os.path.basename(image_name))
        exists = os.path.exists(mask_path)
        print(
            f"DEBUG: Checking mask for {image_name}: {'exists' if exists else 'does not exist'}"
        )
        return exists

    def on_image_selected(self, item):
        image_name = item.text()
        image_path = self.image_manager.get_image_path(image_name)
        if self.right_panel.currentWidget() == self.mask_tracing_interface:
            self.mask_tracing_interface.load_image(image_path)
        else:
            self.display_controller.display_selected_image(item)

    def calculate_root_length(self):
        self.skeleton_handler.calculate_root_length()

    def update_display(self):
        self.display_controller.update_display()

    def load_results(self, results_dir):
        """
        Load results after skeleton generation.
        This method updates the ImageManager and the UI to reflect the newly generated skeletons.
        """
        images_dir = os.path.join(results_dir, "images")
        if not os.path.isdir(images_dir):
            QMessageBox.critical(
                self, "Error", f"Images directory not found at: {images_dir}"
            )
            return

        self.image_manager.load_images(folder_path=images_dir)
        self.populate_file_list()

        # Store the path to the index.html file if it exists
        self.image_manager.html_path = os.path.join(results_dir, "index.html")
        if os.path.exists(self.image_manager.html_path):
            self.display_controller.set_html_path(self.image_manager.html_path)
            self.status_bar.showMessage("Results loaded successfully.", 5000)
        else:
            self.image_manager.html_path = None
            self.status_bar.showMessage("Results loaded, but no HTML file found.", 5000)

    def toggle_mask_tracing(self):
        if self.right_panel.currentIndex() == 0:
            self.right_panel.setCurrentIndex(1)
            self.toggle_mask_tracing_button.setText("Return to Main View")
            self.mask_tracing_interface.mask_saved.connect(self.on_mask_saved)
            self.mask_tracing_interface.mask_cleared.connect(
                self.on_mask_cleared
            )  # Connect the new signal
        else:
            self.right_panel.setCurrentIndex(0)
            self.toggle_mask_tracing_button.setText("Toggle Mask Tracing")
            self.mask_tracing_interface.mask_saved.disconnect(self.on_mask_saved)
            self.mask_tracing_interface.mask_cleared.disconnect(
                self.on_mask_cleared
            )  # Disconnect the new signal

        # Load the current image into the mask tracing interface
        current_item = self.file_list.currentItem()
        if current_item:
            self.on_image_selected(current_item)

    def on_mask_saved(self, image_path):
        print(f"DEBUG: on_mask_saved called with image_path: {image_path}")
        self.mask_saved.emit(image_path)

    def on_mask_cleared(self, image_path):
        print(f"DEBUG: on_mask_cleared called with image_path: {image_path}")
        self.mask_cleared.emit(image_path)
