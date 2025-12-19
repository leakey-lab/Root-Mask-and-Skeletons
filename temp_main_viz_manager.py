"""
Visualization management for the main window.
Handles root length and root area visualization toggling and cleanup.
"""

import os
import logging
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt, QTimer
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

        except PermissionError:
            pass
        except Exception:
            pass

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

        except PermissionError:
            pass
        except Exception:
            pass

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
    main_window.logger.debug("Entering toggle_root_length_visualization")

    try:
        # If visualization exists and is currently shown
        if (
            main_window.root_length_viz is not None
            and main_window.right_panel.currentWidget() == main_window.root_length_viz
        ):
            main_window.logger.debug("Closing existing visualization")
            close_root_length_visualization(main_window)
        else:
            main_window.logger.debug("Opening new visualization")
            # Always create a new visualization instance
            show_root_length_visualization(main_window)
    except Exception as e:
        main_window.logger.error(f"Error in toggle_root_length_visualization: {str(e)}")
        QMessageBox.critical(
            main_window, "Error", f"Error toggling visualization: {str(e)}"
        )


def show_root_length_visualization(main_window):
    """Show the root length visualization with proper loading state management."""
    main_window.logger.debug("Entering show_root_length_visualization method")
    logger.debug("Entering show_root_length_visualization method")

    if main_window.file_list.topLevelItemCount() == 0:
        logger.warning("file_list is empty")
        main_window.logger.debug("file_list is empty")
        QMessageBox.warning(
            main_window, "Warning", "No images loaded. Please load images first."
        )
        return

    try:
        # Find the CSV file path - get the first actual image from the tree
        logger.debug("Searching for first image in file tree")
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
            logger.error("No images found in tree")
            raise ValueError("No images found in tree")

        logger.debug(f"Found first image: {first_image_name}")
        first_image_path = main_window.image_manager.get_image_path(first_image_name)

        if first_image_path is None:
            logger.error(f"Failed to get image path for {first_image_name}")
            raise ValueError("Failed to get image path from image manager")

        logger.debug(f"First image path: {first_image_path}")
        # Get the directory containing the first image
        start_dir = os.path.dirname(first_image_path)
        logger.debug(f"Start directory: {start_dir}")
        
        # Try to get the base folder (parent of start_dir if images are in a subfolder)
        if main_window.image_manager.original_folder:
            base_folder = main_window.image_manager.original_folder
        else:
            base_folder = start_dir
        
        # Search for CSV in multiple locations (prioritize skeletons folder)
        potential_csv_paths = [
            os.path.join(base_folder, "skeletons", "root_lengths.csv"),  # New format
            os.path.join(base_folder, "root_lengths.csv"),  # Base folder
        ]
        
        # Add legacy paths
        base_path = find_test_latest_dir(start_dir)
        if base_path:
            potential_csv_paths.append(os.path.join(base_path, "root_lengths.csv"))
        
        # Try each path until we find the CSV
        csv_path = None
        for path in potential_csv_paths:
            logger.info(f"Looking for CSV at: {path}")
            main_window.logger.debug(f"Looking for CSV at: {path}")
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            error_msg = f"No root length data found. Searched in: {', '.join(potential_csv_paths)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"CSV file found: {csv_path}")

        def create_new_visualization():
            # Create new visualization with the current CSV path
            logger.info(f"Creating new RootLengthVisualization with {csv_path}")
            main_window.logger.debug(
                f"Creating new RootLengthVisualization with {csv_path}"
            )
            try:
                main_window.root_length_viz = RootLengthVisualization(
                    csv_path, main_window.image_manager
                )
                logger.info("RootLengthVisualization instance created")
                main_window.root_length_viz.server_closed.connect(
                    main_window.on_visualization_server_closed
                )
                logger.debug("Connected server_closed signal")

                # Add to right panel and show
                main_window.right_panel.addWidget(main_window.root_length_viz)
                logger.debug("Added visualization to right panel")
                main_window.right_panel.setCurrentWidget(main_window.root_length_viz)
                logger.debug("Set visualization as current widget")

                # Update button text
                main_window.visualize_root_length_button.setText("Close Visualization")
                logger.info("Visualization setup completed successfully")
                main_window.logger.debug("Visualization setup completed successfully")
            except Exception as e:
                logger.error(f"Error creating visualization: {e}", exc_info=True)
                raise

        if main_window.root_length_viz:
            logger.debug("Existing visualization found, closing it first")
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
        logger.error(f"Error in show_root_length_visualization: {str(e)}", exc_info=True)
        main_window.logger.error(f"Error in show_root_length_visualization: {str(e)}")
        QMessageBox.critical(
            main_window, "Error", f"Failed to create visualization: {str(e)}"
        )
        main_window.visualize_root_length_button.setText("📊 Visualize Root Length")


def close_root_length_visualization(main_window):
    """Safely close and cleanup the root length visualization component."""
    main_window.logger.debug("Entering close_root_length_visualization method")

    if not main_window.root_length_viz:
        main_window.logger.debug("No root length visualization to close")
        return

    try:
        main_window.logger.debug("Beginning cleanup process")

        # Update UI state first
        main_window.visualize_root_length_button.setText("📊 Visualize Root Length")
        main_window.switch_right_panel("display")

        # Clean up visualization
        main_window.root_length_viz.cleanup_server()

        # Remove the visualization from the right panel
        main_window.right_panel.removeWidget(main_window.root_length_viz)
        main_window.root_length_viz.deleteLater()
        main_window.root_length_viz = None

        # Update current image display if available
        current_item = main_window.file_list.currentItem()
        if current_item:
            main_window.on_image_selected(current_item)

    except Exception as e:
        main_window.logger.error(f"Error during visualization cleanup: {str(e)}")
        main_window.root_length_viz = None
        main_window.switch_right_panel("display")
        QMessageBox.warning(main_window, "Warning", f"Error during cleanup: {str(e)}")

    main_window.logger.debug("Root length visualization cleanup completed")


def toggle_root_area_visualization(main_window):
    """Toggle between showing and hiding the root area visualization."""
    main_window.logger.debug("Entering toggle_root_area_visualization")

    try:
        # If visualization exists and is currently shown
        if (
            main_window.root_area_viz is not None
            and main_window.right_panel.currentWidget() == main_window.root_area_viz
        ):
            main_window.logger.debug("Closing existing area visualization")
            close_root_area_visualization(main_window)
        else:
            main_window.logger.debug("Opening new area visualization")
            # Always create a new visualization instance
            show_root_area_visualization(main_window)
    except Exception as e:
        main_window.logger.error(f"Error in toggle_root_area_visualization: {str(e)}")
        QMessageBox.critical(
            main_window, "Error", f"Error toggling visualization: {str(e)}"
        )


def show_root_area_visualization(main_window):
    """Show the root area visualization with proper loading state management."""
    main_window.logger.debug("Entering show_root_area_visualization method")
    logger.debug("Entering show_root_area_visualization method")

    if main_window.file_list.topLevelItemCount() == 0:
        logger.warning("file_list is empty")
        main_window.logger.debug("file_list is empty")
        QMessageBox.warning(
            main_window, "Warning", "No images loaded. Please load images first."
        )
        return

    try:
        # Find the CSV file path - get the first actual image from the tree
        logger.debug("Searching for first image in file tree")
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
            logger.error("No images found in tree")
            raise ValueError("No images found in tree")

        logger.debug(f"Found first image: {first_image_name}")
        first_image_path = main_window.image_manager.get_image_path(first_image_name)

        if first_image_path is None:
            logger.error(f"Failed to get image path for {first_image_name}")
            raise ValueError("Failed to get image path from image manager")

        logger.debug(f"First image path: {first_image_path}")
        # Get the directory containing the first image
        start_dir = os.path.dirname(first_image_path)

        # For area, look for masks directory
        if main_window.image_manager.original_folder:
            base_folder = main_window.image_manager.original_folder
            logger.debug(f"Using original_folder: {base_folder}")
        else:
            base_folder = start_dir
            logger.debug(f"Using start_dir as base_folder: {base_folder}")

        # Check for masks directory
        potential_paths = [
            os.path.join(base_folder, "output", "mask"),
            os.path.join(base_folder, "mask"),
        ]
        logger.debug(f"Checking potential mask paths: {potential_paths}")

        masks_dir = None
        for path in potential_paths:
            if os.path.exists(path):
                masks_dir = path
                logger.info(f"Found masks directory: {masks_dir}")
                break

        if not masks_dir:
            logger.error("No masks directory found")
            raise FileNotFoundError("No masks directory found")

        csv_path = os.path.join(masks_dir, "root_areas.csv")
        logger.info(f"Looking for CSV at: {csv_path}")
        main_window.logger.debug(f"Looking for CSV at: {csv_path}")

        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"No root area data found at {csv_path}")

        logger.info(f"CSV file found: {csv_path}")

        def create_new_visualization():
            # Create new visualization with the current CSV path
            logger.info(f"Creating new RootAreaVisualization with {csv_path}")
            main_window.logger.debug(f"Creating new RootAreaVisualization with {csv_path}")
            try:
                main_window.root_area_viz = RootAreaVisualization(csv_path)
                logger.info("RootAreaVisualization instance created")
                main_window.root_area_viz.server_closed.connect(
                    main_window.on_area_visualization_server_closed
                )
                logger.debug("Connected server_closed signal")

                # Add to right panel and show
                main_window.right_panel.addWidget(main_window.root_area_viz)
                logger.debug("Added visualization to right panel")
                main_window.right_panel.setCurrentWidget(main_window.root_area_viz)
                logger.debug("Set visualization as current widget")

                # Update button text
                main_window.visualize_root_area_button.setText("Close Area Visualization")
                logger.info("Area visualization setup completed successfully")
                main_window.logger.debug("Area visualization setup completed successfully")
            except Exception as e:
                logger.error(f"Error creating area visualization: {e}", exc_info=True)
                raise

        if main_window.root_area_viz:
            logger.debug("Existing area visualization found, closing it first")
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
        logger.error(f"Error in show_root_area_visualization: {str(e)}", exc_info=True)
        main_window.logger.error(f"Error in show_root_area_visualization: {str(e)}")
        QMessageBox.critical(
            main_window, "Error", f"Failed to create visualization: {str(e)}"
        )
        main_window.visualize_root_area_button.setText("📈 Visualize Root Area")


def close_root_area_visualization(main_window):
    """Safely close and cleanup the root area visualization component."""
    main_window.logger.debug("Entering close_root_area_visualization method")

    if not main_window.root_area_viz:
        main_window.logger.debug("No root area visualization to close")
        return

    try:
        main_window.logger.debug("Beginning cleanup process")

        # Update UI state first
        main_window.visualize_root_area_button.setText("📈 Visualize Root Area")
        main_window.switch_right_panel("display")

        # Clean up visualization
        main_window.root_area_viz.cleanup_server()

        # Remove the visualization from the right panel
        main_window.right_panel.removeWidget(main_window.root_area_viz)
        main_window.root_area_viz.deleteLater()
        main_window.root_area_viz = None

        # Update current image display if available
        current_item = main_window.file_list.currentItem()
        if current_item:
            main_window.on_image_selected(current_item)

    except Exception as e:
        main_window.logger.error(f"Error during area visualization cleanup: {str(e)}")
        main_window.root_area_viz = None
        main_window.switch_right_panel("display")
        QMessageBox.warning(main_window, "Warning", f"Error during cleanup: {str(e)}")

    main_window.logger.debug("Root area visualization cleanup completed")

