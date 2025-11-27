"""
Main window for the Root Viewer application.
Orchestrates the GUI components and handles core functionality.
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QSplitter,
    QStatusBar,
    QMessageBox,
    QStackedWidget,
    QFileDialog,
    QProgressBar,
)
from PyQt6.QtGui import QColor, QIcon
from PyQt6.QtCore import Qt, pyqtSignal
from .image_manager import ImageManager
from .display_controller import DisplayController
from app.handlers.skeleton_handler import SkeletonHandler
from app.handlers.mask_handler import MaskHandler
from .mask_tracing_interface import MaskTracingInterface
from app.handlers.mask_generation_handler import MaskGenerationHandler
from . import ui_panels
from . import file_tree_manager
from . import visualization_manager
import os
import logging


class MainWindow(QMainWindow):
    mask_saved = pyqtSignal(str)
    mask_cleared = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Viewer")
        self.setGeometry(100, 100, 1200, 700)
        
        # Set window icon with multiple sizes for better visibility
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources", "app_icon.ico")
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            self.setWindowIcon(icon)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Initialize components
        self.image_manager = ImageManager(self)
        self.display_controller = DisplayController(self)
        self.skeleton_handler = SkeletonHandler(self)
        self.mask_handler = MaskHandler(self)
        self.mask_generation_handler = MaskGenerationHandler(self)
        self.mask_tracing_interface = MaskTracingInterface()
        self.root_length_viz = None
        self.root_area_viz = None

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

        # Left Panel - use extracted module
        left_panel = ui_panels.create_left_panel(self)
        splitter.addWidget(left_panel)

        # Right Panel (Stacked Widget for Display Area and Mask Tracing)
        self.right_panel = QStackedWidget()
        self.right_panel.addWidget(ui_panels.create_right_panel(self))
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
        self.loading_progress_bar.setMinimumWidth(400)
        self.loading_progress_bar.hide()
        self.status_bar.addPermanentWidget(self.loading_progress_bar)

    # ==================== File Tree Methods (delegated) ====================
    
    def find_tree_item_by_image_name(self, image_name):
        """Recursively find a tree item by its stored image name"""
        return file_tree_manager.find_tree_item_by_image_name(self.file_list, image_name)

    def highlight_saved_mask(self, image_path):
        file_tree_manager.highlight_saved_mask(self, image_path)

    def unhighlight_cleared_mask(self, image_path):
        file_tree_manager.unhighlight_cleared_mask(self, image_path)

    def populate_file_list(self):
        """Populate the file tree with hierarchical structure"""
        file_tree_manager.populate_file_list(self)

    def update_file_list_mask_status(self):
        """Update the file tree items to show mask status"""
        file_tree_manager.update_file_list_mask_status(self)

    def natural_sort_key(self, text):
        """Create a sort key that handles numbers naturally"""
        return file_tree_manager.natural_sort_key(text)

    # ==================== Visualization Methods (delegated) ====================
    
    def toggle_root_length_visualization(self):
        """Toggle between showing and hiding the root length visualization."""
        visualization_manager.toggle_root_length_visualization(self)

    def show_root_length_visualization(self):
        """Show the root length visualization."""
        visualization_manager.show_root_length_visualization(self)

    def close_root_length_visualization(self):
        """Safely close and cleanup the root length visualization."""
        visualization_manager.close_root_length_visualization(self)

    def toggle_root_area_visualization(self):
        """Toggle between showing and hiding the root area visualization."""
        visualization_manager.toggle_root_area_visualization(self)

    def show_root_area_visualization(self):
        """Show the root area visualization."""
        visualization_manager.show_root_area_visualization(self)

    def close_root_area_visualization(self):
        """Safely close and cleanup the root area visualization."""
        visualization_manager.close_root_area_visualization(self)

    def find_test_latest_dir(self, start_path, max_depth=3):
        """Efficiently search for 'test_latest' directory."""
        return visualization_manager.find_test_latest_dir(start_path, max_depth)

    def on_visualization_server_closed(self):
        """Handle cleanup after visualization server is closed."""
        self.logger.debug("Visualization server closed successfully")

    def on_area_visualization_server_closed(self):
        """Handle cleanup after area visualization server is closed."""
        self.logger.debug("Area visualization server closed successfully")

    # ==================== Image Loading ====================
    
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

    def on_loading_finished(self, images, fake_images, masks, has_fake_real_pairs):
        """Handle completion of image loading"""
        self.loading_progress_bar.hide()
        self.populate_file_list()
        self.status_bar.showMessage(f"Loaded {len(images)} images", 5000)

    # ==================== Tree Item Selection ====================
    
    def mask_exists(self, image_name):
        image_path = self.image_manager.get_image_path(image_name)
        mask_dir = os.path.join(os.path.dirname(image_path), "mask")
        mask_path = os.path.join(mask_dir, os.path.basename(image_name))
        exists = os.path.exists(mask_path)
        return exists

    def on_tree_item_clicked(self, item, column):
        """Handle tree item clicks - expand/collapse folders, load images"""
        # Get the stored image name from the item's user data
        image_name = item.data(0, Qt.ItemDataRole.UserRole)

        # If item is not an image (it's a folder node), toggle expansion
        if image_name is None:
            item.setExpanded(not item.isExpanded())
            return

        # If it's an image, load it
        self.on_image_selected(item)

    def on_image_selected(self, item):
        """Load and display the selected image"""
        # Get the stored image name from the item's user data
        image_name = item.data(0, Qt.ItemDataRole.UserRole)

        # If item is not an image (it's a folder node), do nothing
        if image_name is None:
            return

        image_path = self.image_manager.get_image_path(image_name)
        if image_path is None:
            return

        if self.right_panel.currentWidget() == self.mask_tracing_interface:
            self.mask_tracing_interface.load_image(image_path)
        else:
            self.display_controller.display_selected_image_by_name(image_name)

    # ==================== Calculation Methods ====================
    
    def calculate_root_length(self):
        self.skeleton_handler.calculate_root_length()

    def calculate_root_area(self):
        self.skeleton_handler.calculate_root_area()

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

        # Update the skeleton paths
        if self.image_manager:
            self.image_manager._find_processed_base_path()
            # Update the view mode to trigger skeleton loading
            self.view_mode_combo.setCurrentText(self.view_mode_combo.currentText())

        self.status_bar.showMessage("Results loaded successfully.", 5000)

    # ==================== Mask Tracing ====================
    
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
            def update_mask_colors(item):
                stored_name = item.data(0, Qt.ItemDataRole.UserRole)
                if stored_name:  # Only update leaf items (images)
                    # Check both the mask directory and image manager for mask status
                    image_path = self.image_manager.get_image_path(stored_name)
                    if image_path:
                        mask_dir = os.path.join(os.path.dirname(image_path), "mask")
                        mask_path = os.path.join(mask_dir, os.path.basename(image_path))
                        if os.path.exists(mask_path) or self.image_manager.has_mask(
                            stored_name
                        ):
                            item.setForeground(0, QColor("green"))
                        else:
                            item.setForeground(0, QColor("white"))

                # Recursively update children
                for i in range(item.childCount()):
                    update_mask_colors(item.child(i))

            # Update all top-level items
            for i in range(self.file_list.topLevelItemCount()):
                update_mask_colors(self.file_list.topLevelItem(i))

        else:  # If in mask tracing view, switch back to main view
            self.switch_right_panel("display")
            self.toggle_mask_tracing_button.setText("✏️ Toggle Mask Tracing")

            # Disconnect signals
            self.mask_tracing_interface.mask_saved.disconnect(self.on_mask_saved)
            self.mask_tracing_interface.mask_cleared.disconnect(self.on_mask_cleared)
            self.mask_tracing_interface.b_key_status_changed.disconnect(
                self.on_b_key_status_changed
            )

            # Reset all items to white when leaving mask tracing view
            def reset_colors(item):
                stored_name = item.data(0, Qt.ItemDataRole.UserRole)
                if stored_name:  # Only reset leaf items (images)
                    item.setForeground(0, QColor("white"))

                # Recursively reset children
                for i in range(item.childCount()):
                    reset_colors(item.child(i))

            # Reset all top-level items
            for i in range(self.file_list.topLevelItemCount()):
                reset_colors(self.file_list.topLevelItem(i))

        # Load the current image into the mask tracing interface
        current_item = self.file_list.currentItem()
        if current_item:
            self.on_image_selected(current_item)

    def on_b_key_status_changed(self, is_pressed):
        """Observer method for B key status changes in mask tracing interface."""
        if is_pressed:
            self.status_bar.showMessage(
                "B key pressed - use mouse wheel to adjust brush size", 3000
            )
        else:
            self.status_bar.showMessage("B key released", 3000)

    def on_mask_saved(self, image_path):
        # Always emit the signal
        self.mask_saved.emit(image_path)

        # Only update the visual status if we're in mask tracing view
        if self.right_panel.currentWidget() == self.mask_tracing_interface:
            image_name = os.path.splitext(
                os.path.basename(os.path.normpath(image_path))
            )[0]
            item = self.find_tree_item_by_image_name(image_name)
            if item:
                item.setForeground(0, QColor("green"))
            self.status_bar.showMessage(f"Mask saved for {image_name}", 3000)

    def on_mask_cleared(self, image_path):
        # Always emit the signal
        self.mask_cleared.emit(image_path)

        # Only update the visual status if we're in mask tracing view
        if self.right_panel.currentWidget() == self.mask_tracing_interface:
            image_name = os.path.splitext(
                os.path.basename(os.path.normpath(image_path))
            )[0]
            item = self.find_tree_item_by_image_name(image_name)
            if item:
                item.setForeground(0, QColor("white"))

    # ==================== Panel Switching ====================
    
    def switch_right_panel(self, panel):
        if panel == "display":
            self.right_panel.setCurrentIndex(0)
        elif panel == "mask_tracing":
            self.right_panel.setCurrentIndex(1)
        elif panel == "visualization":
            self.right_panel.setCurrentIndex(2)
