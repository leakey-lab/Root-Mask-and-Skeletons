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
    QProgressBar,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, pyqtSignal
from image_manager import ImageManager
from display_controller import DisplayController
from skeleton_handler import SkeletonHandler
from mask_handler import MaskHandler
from mask_tracing_interface import MaskTracingInterface
from root_length_visulization import RootLengthVisualization
from mask_generation_handler import MaskGenerationHandler
import os


class MainWindow(QMainWindow):
    mask_saved = pyqtSignal(str)
    mask_cleared = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Viewer")
        self.setGeometry(100, 100, 1200, 700)

        # Initialize components
        self.image_manager = ImageManager(self)
        self.display_controller = DisplayController(self)
        self.skeleton_handler = SkeletonHandler(self)
        self.mask_handler = MaskHandler(self)
        self.mask_generation_handler = MaskGenerationHandler(self)
        self.mask_tracing_interface = MaskTracingInterface()
        self.root_length_viz = None

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

        # Set splitter properties
        splitter.setHandleWidth(5)
        splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: #ff79c6;
            }
            QSplitter::handle:hover {
                background-color: #bd93f9;
            }
        """
        )

        # Left Panel
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right Panel (Stacked Widget for Display Area and Mask Tracing)
        self.right_panel = QStackedWidget()
        self.right_panel.addWidget(self.create_right_panel())
        self.right_panel.addWidget(self.mask_tracing_interface)
        self.visualization_widget = QWidget()  # Placeholder for visualization
        self.right_panel.addWidget(self.visualization_widget)
        splitter.addWidget(self.right_panel)

        splitter.setSizes([400, 800])

        # Status Bar with Progress Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add loading progress bar
        self.loading_progress_bar = QProgressBar()
        self.loading_progress_bar.setTextVisible(True)
        self.loading_progress_bar.setRange(0, 100)
        self.loading_progress_bar.hide()  # Initially hidden
        self.status_bar.addPermanentWidget(self.loading_progress_bar)

    def create_left_panel(self):
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)

        self.generate_mask_button = QPushButton("Generate ML Masks")
        self.generate_mask_button.clicked.connect(
            self.mask_generation_handler.generate_masks
        )
        layout.addWidget(self.generate_mask_button)

        self.generate_button = QPushButton("Generate Skeleton")
        self.generate_button.clicked.connect(self.skeleton_handler.generate_skeleton)
        layout.addWidget(self.generate_button)

        self.load_images_button = QPushButton("Load Images")
        self.load_images_button.clicked.connect(self.load_images)
        layout.addWidget(self.load_images_button)

        self.calculate_length_button = QPushButton("Calculate Root Length")
        self.calculate_length_button.clicked.connect(self.calculate_root_length)
        layout.addWidget(self.calculate_length_button)

        self.visualize_root_length_button = QPushButton("Visualize Root Length")
        self.visualize_root_length_button.clicked.connect(
            self.toggle_root_length_visualization
        )
        layout.addWidget(self.visualize_root_length_button)

        self.toggle_mask_tracing_button = QPushButton("Toggle Mask Tracing")
        self.toggle_mask_tracing_button.clicked.connect(self.toggle_mask_tracing)
        layout.addWidget(self.toggle_mask_tracing_button)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(
            ["Single Image", "Overlay", "Side by Side", "Basic View"]
        )
        self.view_mode_combo.currentIndexChanged.connect(
            self.display_controller.update_display_mode
        )
        layout.addWidget(self.view_mode_combo)

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
            # Show progress bar before loading
            self.loading_progress_bar.setValue(0)
            self.loading_progress_bar.show()

            # Clear undo/redo stacks if mask tracing interface exists
            if hasattr(self, "mask_tracing_interface"):
                self.mask_tracing_interface.undo_stack.clear()
                self.mask_tracing_interface.redo_stack.clear()
                self.mask_tracing_interface.last_point = None
                self.mask_tracing_interface.drawing = False
                if (
                    hasattr(self.mask_tracing_interface, "mask_pixmap")
                    and self.mask_tracing_interface.mask_pixmap
                ):
                    self.mask_tracing_interface.mask_pixmap.fill(
                        Qt.GlobalColor.transparent
                    )

            # Load images through image manager
            self.image_manager.load_images(dir_name)

    def update_loading_progress(self, value):
        """Update the loading progress bar"""
        if hasattr(self, "loading_progress_bar"):
            self.loading_progress_bar.setValue(value)
            if value == 100:
                self.loading_progress_bar.hide()

    def on_loading_finished(
        self, images, fake_images, masks, html_path, has_fake_real_pairs
    ):
        """Handle completion of image loading"""
        self.loading_progress_bar.hide()
        self.populate_file_list()
        self.status_bar.showMessage(f"Loaded {len(images)} images", 5000)

    def populate_file_list(self):
        """Populate the file list without checking masks"""
        self.file_list.clear()
        for name in self.image_manager.get_image_names():
            item = QListWidgetItem(os.path.basename(name))
            self.file_list.addItem(item)

        self.status_bar.showMessage(
            f"Loaded {len(self.image_manager.images)} images", 5000
        )

    def update_file_list_mask_status(self):
        """Update the file list items to show mask status"""
        if self.right_panel.currentWidget() == self.mask_tracing_interface:
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                name = item.text()
                if self.mask_exists(name) or self.image_manager.has_mask(name):
                    item.setForeground(QColor("green"))
                    print(f"DEBUG: Set {name} to green (existing mask)")
                else:
                    item.setForeground(QColor("white"))

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
        if self.right_panel.currentIndex() != 1:  # If not in mask tracing view
            self.switch_right_panel("mask_tracing")
            self.toggle_mask_tracing_button.setText("Return to Main View")

            # Connect signals for mask tracing interface
            self.mask_tracing_interface.mask_saved.connect(self.on_mask_saved)
            self.mask_tracing_interface.mask_cleared.connect(self.on_mask_cleared)
            self.mask_tracing_interface.b_key_status_changed.connect(
                self.on_b_key_status_changed
            )

            # Update mask status colors when entering mask tracing view
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                name = item.text()
                # Check both the mask directory and image manager for mask status
                image_path = self.image_manager.get_image_path(name)
                if image_path:
                    mask_dir = os.path.join(os.path.dirname(image_path), "mask")
                    mask_path = os.path.join(mask_dir, os.path.basename(image_path))
                    if os.path.exists(mask_path) or self.image_manager.has_mask(name):
                        item.setForeground(QColor("green"))
                        print(f"DEBUG: Set {name} to green (existing mask)")
                    else:
                        item.setForeground(QColor("white"))
        else:  # If in mask tracing view, switch back to main view
            self.switch_right_panel("display")
            self.toggle_mask_tracing_button.setText("Toggle Mask Tracing")

            # Disconnect signals
            self.mask_tracing_interface.mask_saved.disconnect(self.on_mask_saved)
            self.mask_tracing_interface.mask_cleared.disconnect(self.on_mask_cleared)
            self.mask_tracing_interface.b_key_status_changed.disconnect(
                self.on_b_key_status_changed
            )

            # Reset all items to white when leaving mask tracing view
            for i in range(self.file_list.count()):
                self.file_list.item(i).setForeground(QColor("white"))

        # Load the current image into the mask tracing interface
        current_item = self.file_list.currentItem()
        if current_item:
            self.on_image_selected(current_item)

    def on_b_key_status_changed(self, is_pressed):
        """
        Observer method for B key status changes in mask tracing interface.

        Args:
            is_pressed (bool): True if B key is pressed, False if released
        """
        if is_pressed:
            self.status_bar.showMessage(
                "B key pressed - use mouse wheel to adjust brush size", 3000
            )
        else:
            self.status_bar.showMessage("B key released", 3000)

    def on_mask_saved(self, image_path):
        print(f"DEBUG: on_mask_saved called with image_path: {image_path}")
        # Always emit the signal
        self.mask_saved.emit(image_path)

        # Only update the visual status if we're in mask tracing view
        if self.right_panel.currentWidget() == self.mask_tracing_interface:
            image_name = os.path.splitext(
                os.path.basename(os.path.normpath(image_path))
            )[0]
            items = self.file_list.findItems(image_name, Qt.MatchFlag.MatchExactly)
            if items:
                item = items[0]
                item.setForeground(QColor("green"))
                print(f"DEBUG: Set {image_name} to green (newly saved mask)")
            else:
                print(f"DEBUG: Could not find item for {image_name}")
            self.status_bar.showMessage(f"Mask saved for {image_name}", 3000)

    def on_mask_cleared(self, image_path):
        print(f"DEBUG: on_mask_cleared called with image_path: {image_path}")
        # Always emit the signal
        self.mask_cleared.emit(image_path)

        # Only update the visual status if we're in mask tracing view
        if self.right_panel.currentWidget() == self.mask_tracing_interface:
            image_name = os.path.splitext(
                os.path.basename(os.path.normpath(image_path))
            )[0]
            items = self.file_list.findItems(image_name, Qt.MatchFlag.MatchExactly)
            if items:
                item = items[0]
                item.setForeground(QColor("white"))
                print(f"DEBUG: Set {image_name} to white (cleared mask)")
            else:
                print(f"DEBUG: Could not find item for {image_name}")

    def find_test_latest_dir(self, start_path, max_depth=3):
        """
        Search for 'test_latest' directory both up and down from the starting directory,
        up to max_depth levels in either direction.

        Args:
            start_path (str): Starting directory path
            max_depth (int): Maximum levels to search in either direction

        Returns:
            str or None: Path to test_latest directory if found, None otherwise
        """
        start_path = os.path.abspath(start_path)
        print(f"DEBUG: Starting search from {start_path}")

        def search_down(current_path, current_depth):
            """Search downstream directories"""
            if current_depth > max_depth:
                return None

            print(f"DEBUG: Searching down at level {current_depth}: {current_path}")

            # Check if current directory is test_latest
            if os.path.basename(current_path) == "test_latest":
                return current_path

            # Check subdirectories
            try:
                for item in os.listdir(current_path):
                    item_path = os.path.join(current_path, item)
                    if os.path.isdir(item_path):
                        result = search_down(item_path, current_depth + 1)
                        if result:
                            return result
            except PermissionError:
                print(f"DEBUG: Permission denied accessing {current_path}")
            except Exception as e:
                print(f"DEBUG: Error accessing {current_path}: {str(e)}")

            return None

        def search_up(current_path, current_depth):
            """Search upstream directories"""
            if current_depth > max_depth:
                return None

            print(f"DEBUG: Searching up at level {current_depth}: {current_path}")

            # Check if current directory is test_latest
            if os.path.basename(current_path) == "test_latest":
                return current_path

            # Check parent directory if not at root
            parent_dir = os.path.dirname(current_path)
            if parent_dir == current_path:  # Reached root directory
                return None

            try:
                # First check siblings of current directory
                parent = os.path.dirname(current_path)
                for item in os.listdir(parent):
                    item_path = os.path.join(parent, item)
                    if os.path.isdir(item_path) and item_path != current_path:
                        result = search_down(item_path, 0)  # Start new downward search
                        if result:
                            return result

                # Then move up to parent
                return search_up(parent_dir, current_depth + 1)

            except PermissionError:
                print(f"DEBUG: Permission denied accessing parent of {current_path}")
            except Exception as e:
                print(f"DEBUG: Error accessing parent of {current_path}: {str(e)}")

            return None

        # Try searching down first
        result = search_down(start_path, 0)
        if result:
            print(f"DEBUG: Found test_latest directory downstream at: {result}")
            return result

        # If not found, search up
        result = search_up(start_path, 0)
        if result:
            print(f"DEBUG: Found test_latest directory upstream at: {result}")
            return result

        print("DEBUG: test_latest directory not found in either direction")
        return None

    def show_root_length_visualization(self):
        print("DEBUG: Entering show_root_length_visualization method")

        if self.file_list.count() == 0:
            print("DEBUG: file_list is empty")
            QMessageBox.warning(
                self, "Warning", "No images loaded. Please load images first."
            )
            return

        first_image_name = self.file_list.item(0).text()
        first_image_path = self.image_manager.get_image_path(first_image_name)
        print(f"DEBUG: First image path: {first_image_path}")

        if first_image_path is None:
            print("DEBUG: Failed to get image path from image_manager")
            QMessageBox.warning(
                self, "Warning", "Failed to get image path. Please reload images."
            )
            return

        # Get the directory containing the first image
        start_dir = os.path.dirname(first_image_path)

        # Search for test_latest directory
        base_path = self.find_test_latest_dir(
            start_dir
        )  # Now calling as instance method

        if base_path is None:
            print(
                "DEBUG: Could not find test_latest directory, using original directory"
            )
            base_path = start_dir
        else:
            print(f"DEBUG: Found test_latest directory at {base_path}")

        # The CSV should be directly in the test_latest directory
        csv_path = os.path.join(base_path, "root_lengths.csv")
        print(f"DEBUG: Constructed csv_path: {csv_path}")

        if os.path.exists(csv_path):
            print(f"DEBUG: CSV file found at {csv_path}")
            try:
                if self.root_length_viz is None:
                    self.root_length_viz = RootLengthVisualization(csv_path)
                    self.right_panel.addWidget(self.root_length_viz)
                self.right_panel.setCurrentWidget(self.root_length_viz)
                print("DEBUG: RootLengthVisualization added to right panel")
                self.visualize_root_length_button.setText("Close Visualization")
            except Exception as e:
                print(f"DEBUG: Error creating RootLengthVisualization: {str(e)}")
                QMessageBox.critical(
                    self, "Error", f"Failed to create visualization: {str(e)}"
                )
        else:
            print(f"DEBUG: CSV file not found at {csv_path}")
            QMessageBox.warning(
                self,
                "Warning",
                "No root length data found. Please calculate root lengths first.",
            )

    def switch_right_panel(self, panel):
        if panel == "display":
            self.right_panel.setCurrentIndex(0)
        elif panel == "mask_tracing":
            self.right_panel.setCurrentIndex(1)
        elif panel == "visualization":
            self.right_panel.setCurrentIndex(2)

    def close_root_length_visualization(self):
        print("DEBUG: Entering close_root_length_visualization method")
        if self.root_length_viz:
            print("DEBUG: root_length_viz exists, attempting to close")
            self.right_panel.removeWidget(self.root_length_viz)
            self.root_length_viz.deleteLater()
            self.root_length_viz = None
            self.visualize_root_length_button.setText("Visualize Root Length")
            print("DEBUG: Root length visualization closed")

            # Switch back to the main display area
            self.switch_right_panel("display")

            # Update the display to show the current image
            current_item = self.file_list.currentItem()
            if current_item:
                self.on_image_selected(current_item)

            print("DEBUG: Switched back to main display area")
        else:
            print("DEBUG: root_length_viz is None, nothing to close")

    def toggle_root_length_visualization(self):
        if (
            self.root_length_viz is None
            or not self.right_panel.currentWidget() == self.root_length_viz
        ):
            self.show_root_length_visualization()
        else:
            self.close_root_length_visualization()
