"""
Visualization management for the main window.
Handles root length and root area visualization toggling and cleanup.
"""

import logging
import os
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from app.config import ROOT_AREAS_CSV, ROOT_LENGTHS_CSV
from app.visualization.root_length_visulization import RootLengthVisualization
from app.visualization.root_area_visualization import RootAreaVisualization

logger = logging.getLogger(__name__)


def find_test_latest_dir(start_path, max_depth=3):
    """
    Efficiently search for 'test_latest' directory using a focused search strategy.

    Args:
        start_path (str): Starting directory path
        max_depth (int): Maximum levels to search in either direction

    Returns:
        str or None: Path to test_latest directory if found, None otherwise
    """
    start_path = os.path.abspath(start_path)

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

        except OSError as e:
            # Unreadable/inaccessible directory (permissions, vanished path);
            # skip it and continue the search elsewhere.
            logger.debug("Skipping directory %s during search: %s", root_path, e)

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

        except OSError as e:
            # Unreadable/inaccessible directory; stop ascending this branch.
            logger.debug("Skipping parent %s during search: %s", path, e)

        return None

    # Try searching down first with early termination
    result = search_directory(start_path)
    if result:
        return result

    # If not found, search up
    result = search_up(start_path)
    if result:
        return result

    return None


def toggle_root_length_visualization(main_window):
    """Toggle between showing and hiding the root length visualization."""
    try:
        # If visualization exists and is currently shown
        if (
            main_window.root_length_viz is not None
            and main_window.right_panel.currentWidget() == main_window.root_length_viz
        ):
            close_root_length_visualization(main_window)
        else:
            # Always create a new visualization instance
            show_root_length_visualization(main_window)
    except Exception as e:
        # Top-level UI guard: log with traceback, then surface to the user.
        logger.exception("Error toggling root length visualization")
        QMessageBox.critical(
            main_window, "Error", f"Error toggling visualization: {str(e)}"
        )


def show_root_length_visualization(main_window):
    """Show the root length visualization with proper loading state management."""

    if main_window.file_list.topLevelItemCount() == 0:
        QMessageBox.warning(
            main_window, "Warning", "No images loaded. Please load images first."
        )
        return

    try:
        # Qt6 limitation: QtWebEngine (QWebEngineView) can't render in a window that
        # contains any QOpenGLWidget. Temporarily disable OpenGL viewports before
        # constructing the visualization (QWebEngineView is created in its __init__).
        if hasattr(main_window, "set_opengl_viewports_enabled"):
            main_window.set_opengl_viewports_enabled(False)
            try:
                QApplication.processEvents()
            except RuntimeError as e:
                # Event loop may be unavailable mid-teardown; non-fatal.
                logger.debug("processEvents failed: %s", e)

        # Find the CSV file path - get the first actual image from the tree
        first_image_name = None

        def find_first_image(item):
            stored_name = item.data(0, Qt.ItemDataRole.UserRole)
            if stored_name:
                return stored_name
            for i in range(item.childCount()):
                result = find_first_image(item.child(i))
                if result:
                    return result
            return None

        for i in range(main_window.file_list.topLevelItemCount()):
            first_image_name = find_first_image(main_window.file_list.topLevelItem(i))
            if first_image_name:
                break

        if not first_image_name:
            raise ValueError("No images found in tree")

        first_image_path = main_window.image_manager.get_image_path(first_image_name)

        if first_image_path is None:

            raise ValueError("Failed to get image path from image manager")
        # Get the directory containing the first image
        start_dir = os.path.dirname(first_image_path)

        # Try to get the base folder (parent of start_dir if images are in a subfolder)
        if main_window.image_manager.original_folder:
            base_folder = main_window.image_manager.original_folder
        else:
            base_folder = start_dir
        
        # Search for CSV in multiple locations (prioritize skeletons folder)
        potential_csv_paths = [
            os.path.join(base_folder, "skeletons", ROOT_LENGTHS_CSV),  # New format
            os.path.join(base_folder, ROOT_LENGTHS_CSV),  # Base folder
        ]

        # Add legacy paths
        base_path = find_test_latest_dir(start_dir)
        if base_path:
            potential_csv_paths.append(os.path.join(base_path, ROOT_LENGTHS_CSV))
        
        # Try each path until we find the CSV
        csv_path = None
        for path in potential_csv_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            error_msg = f"No root length data found. Searched in: {', '.join(potential_csv_paths)}"
            raise FileNotFoundError(error_msg)


        def create_new_visualization():
            # Create new visualization with the current CSV path.
            # Errors propagate to the outer handler in
            # show_root_length_visualization, which logs and notifies the user.
            main_window.root_length_viz = RootLengthVisualization(
                csv_path, main_window.image_manager
            )
            main_window.root_length_viz.server_closed.connect(
                main_window.on_visualization_server_closed
            )

            # Add to right panel and show
            main_window.right_panel.addWidget(main_window.root_length_viz)
            main_window.right_panel.setCurrentWidget(main_window.root_length_viz)

            # Update button text
            main_window.visualize_root_length_button.setText("Close Visualization")

        if main_window.root_length_viz:
            try:
                main_window.root_length_viz.server_closed.disconnect()
            except TypeError:
                pass
            main_window.root_length_viz.server_closed.connect(
                lambda: QTimer.singleShot(500, create_new_visualization)
            )
            close_root_length_visualization(main_window)
        else:
            create_new_visualization()

    except Exception as e:
        logger.exception("Failed to create root length visualization")
        QMessageBox.critical(
            main_window, "Error", f"Failed to create visualization: {str(e)}"
        )
        main_window.visualize_root_length_button.setText("📊 Visualize Root Length")


def close_root_length_visualization(main_window):
    """Safely close and cleanup the root length visualization component."""

    if not main_window.root_length_viz:
        return

    try:

        # Update UI state first
        main_window.visualize_root_length_button.setText("📊 Visualize Root Length")
        main_window.switch_right_panel("display")

        # Clean up visualization
        if main_window.root_length_viz:
            main_window.root_length_viz.cleanup_server()

        # Remove the visualization from the right panel
        if main_window.root_length_viz:
            try:
                main_window.right_panel.removeWidget(main_window.root_length_viz)
            except RuntimeError as e:
                # Widget may already be detached/deleted; safe to ignore.
                logger.debug("removeWidget (length viz) failed: %s", e)
            main_window.root_length_viz.deleteLater()
        main_window.root_length_viz = None

        # Re-enable OpenGL viewports now that WebEngine widget is removed.
        if hasattr(main_window, "set_opengl_viewports_enabled"):
            main_window.set_opengl_viewports_enabled(True)
            try:
                QApplication.processEvents()
            except RuntimeError as e:
                # Event loop may be unavailable mid-teardown; non-fatal.
                logger.debug("processEvents failed: %s", e)

        # Update current image display if available
        current_item = main_window.file_list.currentItem()
        if current_item:
            main_window.on_image_selected(current_item)

    except Exception as e:
        logger.exception("Error during root length visualization cleanup")
        main_window.root_length_viz = None
        main_window.switch_right_panel("display")
        QMessageBox.warning(main_window, "Warning", f"Error during cleanup: {str(e)}")


def toggle_root_area_visualization(main_window):
    """Toggle between showing and hiding the root area visualization."""

    try:
        # If visualization exists and is currently shown
        if (
            main_window.root_area_viz is not None
            and main_window.right_panel.currentWidget() == main_window.root_area_viz
        ):
            close_root_area_visualization(main_window)
        else:
            # Always create a new visualization instance
            show_root_area_visualization(main_window)
    except Exception as e:
        # Top-level UI guard: log with traceback, then surface to the user.
        logger.exception("Error toggling root area visualization")
        QMessageBox.critical(
            main_window, "Error", f"Error toggling visualization: {str(e)}"
        )


def show_root_area_visualization(main_window):
    """Show the root area visualization with proper loading state management."""

    if main_window.file_list.topLevelItemCount() == 0:
        QMessageBox.warning(
            main_window, "Warning", "No images loaded. Please load images first."
        )
        return

    try:
        # Qt6 limitation: QtWebEngine (QWebEngineView) can't render in a window that
        # contains any QOpenGLWidget. Temporarily disable OpenGL viewports before
        # constructing the visualization.
        if hasattr(main_window, "set_opengl_viewports_enabled"):
            main_window.set_opengl_viewports_enabled(False)
            try:
                QApplication.processEvents()
            except RuntimeError as e:
                # Event loop may be unavailable mid-teardown; non-fatal.
                logger.debug("processEvents failed: %s", e)

        # Find the CSV file path - get the first actual image from the tree
        first_image_name = None

        def find_first_image(item):
            stored_name = item.data(0, Qt.ItemDataRole.UserRole)
            if stored_name:
                return stored_name
            for i in range(item.childCount()):
                result = find_first_image(item.child(i))
                if result:
                    return result
            return None

        for i in range(main_window.file_list.topLevelItemCount()):
            first_image_name = find_first_image(main_window.file_list.topLevelItem(i))
            if first_image_name:
                break

        if not first_image_name:
            raise ValueError("No images found in tree")

        first_image_path = main_window.image_manager.get_image_path(first_image_name)

        if first_image_path is None:
            raise ValueError("Failed to get image path from image manager")

        # Get the directory containing the first image
        start_dir = os.path.dirname(first_image_path)

        # For area, look for masks directory
        if main_window.image_manager.original_folder:
            base_folder = main_window.image_manager.original_folder
        else:
            base_folder = start_dir

        # Check for masks directory
        potential_paths = [
            os.path.join(base_folder, "output", "mask"),
            os.path.join(base_folder, "mask"),
        ]

        masks_dir = None
        for path in potential_paths:
            if os.path.exists(path):
                masks_dir = path
                break

        if not masks_dir:
            raise FileNotFoundError("No masks directory found")

        csv_path = os.path.join(masks_dir, ROOT_AREAS_CSV)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No root area data found at {csv_path}")

        def create_new_visualization():
            # Create new visualization with the current CSV path.
            # Errors propagate to the outer handler in
            # show_root_area_visualization, which logs and notifies the user.
            main_window.root_area_viz = RootAreaVisualization(csv_path)
            main_window.root_area_viz.server_closed.connect(
                main_window.on_area_visualization_server_closed
            )

            # Add to right panel and show
            main_window.right_panel.addWidget(main_window.root_area_viz)
            main_window.right_panel.setCurrentWidget(main_window.root_area_viz)

            # Update button text
            main_window.visualize_root_area_button.setText("Close Area Visualization")

        if main_window.root_area_viz:
            try:
                main_window.root_area_viz.server_closed.disconnect()
            except TypeError:
                pass
            main_window.root_area_viz.server_closed.connect(
                lambda: QTimer.singleShot(500, create_new_visualization)
            )
            close_root_area_visualization(main_window)
        else:
            create_new_visualization()

    except Exception as e:
        logger.exception("Failed to create root area visualization")
        QMessageBox.critical(
            main_window, "Error", f"Failed to create visualization: {str(e)}"
        )
        main_window.visualize_root_area_button.setText("📈 Visualize Root Area")


def close_root_area_visualization(main_window):
    """Safely close and cleanup the root area visualization component."""

    if not main_window.root_area_viz:
        return

    try:

        # Update UI state first
        main_window.visualize_root_area_button.setText("📈 Visualize Root Area")
        main_window.switch_right_panel("display")

        # Clean up visualization
        if main_window.root_area_viz:
            main_window.root_area_viz.cleanup_server()

        # Remove the visualization from the right panel
        if main_window.root_area_viz:
            try:
                main_window.right_panel.removeWidget(main_window.root_area_viz)
            except RuntimeError as e:
                # Widget may already be detached/deleted; safe to ignore.
                logger.debug("removeWidget (area viz) failed: %s", e)
            main_window.root_area_viz.deleteLater()
        main_window.root_area_viz = None

        # Re-enable OpenGL viewports now that WebEngine widget is removed.
        if hasattr(main_window, "set_opengl_viewports_enabled"):
            main_window.set_opengl_viewports_enabled(True)
            try:
                QApplication.processEvents()
            except RuntimeError as e:
                # Event loop may be unavailable mid-teardown; non-fatal.
                logger.debug("processEvents failed: %s", e)

        # Update current image display if available
        current_item = main_window.file_list.currentItem()
        if current_item:
            main_window.on_image_selected(current_item)

    except Exception as e:
        logger.exception("Error during root area visualization cleanup")
        main_window.root_area_viz = None
        main_window.switch_right_panel("display")
        QMessageBox.warning(main_window, "Warning", f"Error during cleanup: {str(e)}")

