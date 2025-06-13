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
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from image_manager import ImageManager
from display_controller import DisplayController
from skeleton_handler import SkeletonHandler
from mask_handler import MaskHandler
from mask_tracing_interface import MaskTracingInterface
from root_length_visulization import RootLengthVisualization
from mask_generation_handler import MaskGenerationHandler
from enhanced_mask_tracing_interface import EnhancedMaskTracingInterface
import os
import logging


class MainWindow(QMainWindow):
    mask_saved = pyqtSignal(str)
    mask_cleared = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Viewer")
        self.setGeometry(100, 100, 1200, 700)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # Initialize components
        self.image_manager = ImageManager(self)
        self.display_controller = DisplayController(self)
        self.skeleton_handler = SkeletonHandler(self)
        self.mask_handler = MaskHandler(self)
        self.mask_generation_handler = MaskGenerationHandler(self)
        self.mask_tracing_interface = EnhancedMaskTracingInterface()
        self.root_length_viz = None

        # Connect the mask_saved and mask_cleared signals
        self.mask_saved.connect(self.highlight_saved_mask)
        self.mask_cleared.connect(self.unhighlight_cleared_mask)

        self.init_ui()
        self.setup_sam2_config()

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

    ### Local Chnage not to be pushed
    def populate_file_list(self):
        """Populate the file list with image names sorted by first 3 numbers"""
        self.file_list.clear()

        # Custom sort key function to extract first 3 numbers from filename
        def sort_by_first_3_numbers(filename):
            import re

            # Extract first 3 consecutive digits from the filename
            match = re.search(r"(\d{1,4})", os.path.basename(filename))
            if match:
                return int(match.group(1))
            return float("inf")  # Files without numbers go to the end

        # Get image names and sort them using the custom key
        sorted_names = sorted(
            self.image_manager.get_image_names(), key=sort_by_first_3_numbers
        )

        for name in sorted_names:
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
        This method updates the skeleton information without reloading all images.
        """
        images_dir = os.path.join(results_dir, "images")
        if not os.path.isdir(images_dir):
            QMessageBox.critical(
                self, "Error", f"Images directory not found at: {images_dir}"
            )
            return

        # Instead of reloading all images, just update the skeleton paths
        if self.image_manager:
            self.image_manager._find_processed_base_path()
            # Update the view mode to trigger skeleton loading
            self.view_mode_combo.setCurrentText(self.view_mode_combo.currentText())

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
        Efficiently search for 'test_latest' directory using a focused search strategy.

        Args:
            start_path (str): Starting directory path
            max_depth (int): Maximum levels to search in either direction

        Returns:
            str or None: Path to test_latest directory if found, None otherwise
        """
        start_path = os.path.abspath(start_path)
        print(f"DEBUG: Starting search from {start_path}")

        def search_directory(root_path, depth=0):
            """Search a directory and its immediate subdirectories for test_latest"""
            if depth > max_depth:
                return None

            # First check if current directory is test_latest
            if os.path.basename(root_path) == "test_latest":
                return root_path

            try:
                # Get immediate subdirectories, prioritizing likely candidates
                subdirs = []
                with os.scandir(root_path) as entries:
                    for entry in entries:
                        if entry.is_dir():
                            # Prioritize directories that are likely to contain our target
                            name = entry.name.lower()
                            if name in (
                                "output",
                                "results",
                                "test",
                                "latest",
                                "test_latest",
                            ):
                                subdirs.insert(0, entry.path)
                            else:
                                subdirs.append(entry.path)

                # Search prioritized subdirectories
                for subdir in subdirs:
                    result = search_directory(subdir, depth + 1)
                    if result:
                        return result

            except PermissionError:
                print(f"DEBUG: Permission denied accessing {root_path}")
            except Exception as e:
                print(f"DEBUG: Error accessing {root_path}: {str(e)}")

            return None

        def search_up(path, depth=0):
            """Search parent directories and their siblings"""
            if depth > max_depth:
                return None

            try:
                # Check current directory
                if os.path.basename(path) == "test_latest":
                    return path

                # Get parent directory
                parent = os.path.dirname(path)
                if parent == path:  # Reached root
                    return None

                # Check siblings of current directory
                sibling_result = None
                with os.scandir(parent) as entries:
                    for entry in entries:
                        if entry.is_dir() and entry.path != path:
                            # Quick check for the target directory name
                            if entry.name == "test_latest":
                                return entry.path
                            # Only search promising sibling directories
                            if entry.name.lower() in (
                                "output",
                                "results",
                                "test",
                                "latest",
                            ):
                                sibling_result = search_directory(entry.path, 0)
                                if sibling_result:
                                    return sibling_result

                # Move up to parent
                return search_up(parent, depth + 1)

            except PermissionError:
                print(f"DEBUG: Permission denied accessing parent of {path}")
            except Exception as e:
                print(f"DEBUG: Error accessing parent of {path}: {str(e)}")

            return None

        # Try searching down first with early termination
        result = search_directory(start_path)
        if result:
            print(f"DEBUG: Found test_latest directory downstream at: {result}")
            return result

        # If not found, search up
        result = search_up(start_path)
        if result:
            print(f"DEBUG: Found test_latest directory upstream at: {result}")
            return result

        print("DEBUG: test_latest directory not found in either direction")
        return None

    def switch_right_panel(self, panel):
        if panel == "display":
            self.right_panel.setCurrentIndex(0)
        elif panel == "mask_tracing":
            self.right_panel.setCurrentIndex(1)
        elif panel == "visualization":
            self.right_panel.setCurrentIndex(2)

    def toggle_root_length_visualization(self):
        """Toggle between showing and hiding the root length visualization."""
        self.logger.debug("Entering toggle_root_length_visualization")

        try:
            # If visualization exists and is currently shown
            if (
                self.root_length_viz is not None
                and self.right_panel.currentWidget() == self.root_length_viz
            ):
                self.logger.debug("Closing existing visualization")
                self.close_root_length_visualization()
            else:
                self.logger.debug("Opening new visualization")
                # Always create a new visualization instance
                self.show_root_length_visualization()
        except Exception as e:
            self.logger.error(f"Error in toggle_root_length_visualization: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Error toggling visualization: {str(e)}"
            )

    def show_root_length_visualization(self):
        """Show the root length visualization with proper loading state management."""
        self.logger.debug("Entering show_root_length_visualization method")

        if self.file_list.count() == 0:
            self.logger.debug("file_list is empty")
            QMessageBox.warning(
                self, "Warning", "No images loaded. Please load images first."
            )
            return

        try:
            # Find the CSV file path
            first_image_name = self.file_list.item(0).text()
            first_image_path = self.image_manager.get_image_path(first_image_name)

            if first_image_path is None:
                raise ValueError("Failed to get image path from image manager")

            # Get the directory containing the first image
            start_dir = os.path.dirname(first_image_path)
            base_path = self.find_test_latest_dir(start_dir)

            if base_path is None:
                self.logger.debug("Using original directory as base path")
                base_path = start_dir

            csv_path = os.path.join(base_path, "root_lengths.csv")
            self.logger.debug(f"Looking for CSV at: {csv_path}")

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"No root length data found at {csv_path}")

            def create_new_visualization():
                # Create new visualization with the current CSV path
                self.logger.debug(
                    f"Creating new RootLengthVisualization with {csv_path}"
                )
                self.root_length_viz = RootLengthVisualization(csv_path)
                self.root_length_viz.server_closed.connect(
                    self.on_visualization_server_closed
                )

                # Add to right panel and show
                self.right_panel.addWidget(self.root_length_viz)
                self.right_panel.setCurrentWidget(self.root_length_viz)

                # Update button text
                self.visualize_root_length_button.setText("Close Visualization")
                self.logger.debug("Visualization setup completed successfully")

            if self.root_length_viz:
                try:
                    self.root_length_viz.server_closed.disconnect()
                except TypeError:
                    pass
                self.root_length_viz.server_closed.connect(
                    lambda: QTimer.singleShot(500, create_new_visualization)
                )
                self.close_root_length_visualization()
            else:
                create_new_visualization()

        except Exception as e:
            self.logger.error(f"Error in show_root_length_visualization: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to create visualization: {str(e)}"
            )
            self.visualize_root_length_button.setText("Visualize Root Length")

    def close_root_length_visualization(self):
        """Safely close and cleanup the root length visualization component."""
        self.logger.debug("Entering close_root_length_visualization method")

        if not self.root_length_viz:
            self.logger.debug("No root length visualization to close")
            return

        try:
            self.logger.debug("Beginning cleanup process")

            # Update UI state first
            self.visualize_root_length_button.setText("Visualize Root Length")
            self.switch_right_panel("display")

            # Clean up visualization
            self.root_length_viz.cleanup_server()

            # Remove the visualization from the right panel
            self.right_panel.removeWidget(self.root_length_viz)
            self.root_length_viz.deleteLater()
            self.root_length_viz = None

            # Update current image display if available
            current_item = self.file_list.currentItem()
            if current_item:
                self.on_image_selected(current_item)

        except Exception as e:
            self.logger.error(f"Error during visualization cleanup: {str(e)}")
            self.root_length_viz = None
            self.switch_right_panel("display")
            QMessageBox.warning(self, "Warning", f"Error during cleanup: {str(e)}")

        self.logger.debug("Root length visualization cleanup completed")

    def on_visualization_server_closed(self):
        """Handle cleanup after visualization server is closed."""
        self.logger.debug("Visualization server closed successfully")

    def setup_sam2_config(self):
        """Setup SAM2 configuration - add this method"""
        # Set your actual model paths here
        self.sam2_model_path = (
            "checkpoints/SAM2_weights/model_best_new_data.pt"  # Your model path
        )
        self.sam2_config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Your config path

        # Apply paths to the interface
        if hasattr(self.mask_tracing_interface, "sam2_controls"):
            self.mask_tracing_interface.sam2_controls.model_path = self.sam2_model_path
            self.mask_tracing_interface.sam2_controls.config_file = (
                self.sam2_config_path
            )
