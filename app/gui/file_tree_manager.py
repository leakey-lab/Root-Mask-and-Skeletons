"""
File tree management for the main window.
Handles tree widget population, item finding, and status updates.
"""

import os
import re
from PyQt6.QtWidgets import QTreeWidgetItem
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt

# Global dictionary for fast item lookup
# Maps image_name -> QTreeWidgetItem
_item_cache = {}


def natural_sort_key(text):
    """
    Create a sort key that handles numbers naturally (T1, T2, T10, T20).
    Returns tuple: (is_special, numeric_value, text)
    """
    # Handle special cases like "No Tube"
    if text == "No Tube":
        return (True, float("inf"), text)

    # Try to extract number from text like "T1", "T20", etc.
    match = re.search(r"(\d+)", text)
    if match:
        number = int(match.group(1))
        return (False, number, text)

    # If no number found, just use the text
    return (False, 0, text)


def find_tree_item_by_image_name(file_list, image_name):
    """
    Find a tree item by its stored image name using cached dictionary lookup.
    Falls back to recursive search if not found in cache.
    
    Args:
        file_list: QTreeWidget to search
        image_name: Image name to find
        
    Returns:
        QTreeWidgetItem or None
    """
    # Try cache first (O(1) lookup)
    if image_name in _item_cache:
        return _item_cache[image_name]
    
    # Fallback to recursive search if not in cache
    def search_item(item):
        # Check if this item stores the image name
        stored_name = item.data(0, Qt.ItemDataRole.UserRole)
        if stored_name == image_name:
            # Cache the found item for future lookups
            _item_cache[image_name] = item
            return item

        # Search children
        for i in range(item.childCount()):
            result = search_item(item.child(i))
            if result:
                return result
        return None

    # Search all top-level items
    for i in range(file_list.topLevelItemCount()):
        result = search_item(file_list.topLevelItem(i))
        if result:
            return result
    return None


def populate_file_list(main_window):
    """
    Populate the file tree with hierarchical structure.
    Optimized with caching and batched updates.
    
    Args:
        main_window: MainWindow instance with file_list and image_manager
    """
    # Clear cache when repopulating
    global _item_cache
    _item_cache.clear()
    
    # Disable updates during population for better performance
    main_window.file_list.setUpdatesEnabled(False)
    try:
        main_window.file_list.clear()

        # Get hierarchical structure from image manager
        hierarchy = main_window.image_manager.get_hierarchical_structure()

        # Build the tree
        for field_name in sorted(hierarchy.keys()):
            field_item = QTreeWidgetItem(main_window.file_list, [f"📁 {field_name}"])
            field_item.setData(0, Qt.ItemDataRole.UserRole, None)  # Not an image
            field_item.setExpanded(False)

            # Style field item
            font = field_item.font(0)
            font.setBold(True)
            font.setPointSize(10)
            field_item.setFont(0, font)
            field_item.setForeground(0, QColor("#50fa7b"))  # Green

            for tube_name in sorted(
                hierarchy[field_name].keys(), key=natural_sort_key
            ):
                tube_item = QTreeWidgetItem(field_item, [f"🧪 {tube_name}"])
                tube_item.setData(0, Qt.ItemDataRole.UserRole, None)  # Not an image
                tube_item.setExpanded(False)

                # Style tube item
                font = tube_item.font(0)
                font.setBold(True)
                tube_item.setFont(0, font)
                tube_item.setForeground(0, QColor("#8be9fd"))  # Cyan

                for date_name in sorted(
                    hierarchy[field_name][tube_name].keys(),
                    key=lambda x: (x == "No Date", x),
                ):
                    images_list = hierarchy[field_name][tube_name][date_name]
                    image_count = len(images_list)
                    date_item = QTreeWidgetItem(
                        tube_item, [f"📅 {date_name} ({image_count} images)"]
                    )
                    date_item.setData(0, Qt.ItemDataRole.UserRole, None)  # Not an image
                    date_item.setExpanded(False)

                    # Style date item
                    font = date_item.font(0)
                    font.setBold(False)
                    date_item.setFont(0, font)
                    date_item.setForeground(0, QColor("#ffb86c"))  # Orange

                    # Add images with L prefix directly under date
                    for depth, image_name in images_list:
                        if depth != float("inf"):
                            prefix = f"L{int(depth)}"
                        else:
                            prefix = "L?"

                        display_name = f"  🖼️  {prefix} - {os.path.basename(image_name)}"
                        image_item = QTreeWidgetItem(date_item, [display_name])
                        image_item.setData(
                            0, Qt.ItemDataRole.UserRole, image_name
                        )  # Store full image name

                        # Cache the item for fast lookup
                        _item_cache[image_name] = image_item

                        # Add tooltip with full path
                        image_path = main_window.image_manager.get_image_path(image_name)
                        if image_path:
                            image_item.setToolTip(0, image_path)

        # Expand the first level by default for easy access
        main_window.file_list.expandToDepth(0)
        
    finally:
        # Re-enable updates
        main_window.file_list.setUpdatesEnabled(True)

    main_window.status_bar.showMessage(
        f"Loaded {len(main_window.image_manager.images)} images", 5000
    )


def update_file_list_mask_status(main_window):
    """
    Update the file tree items to show mask status.
    Optimized with batched updates.
    
    Args:
        main_window: MainWindow instance with file_list and mask checking methods
    """
    if main_window.right_panel.currentWidget() == main_window.mask_tracing_interface:
        # Disable updates during batch color changes for better performance
        main_window.file_list.setUpdatesEnabled(False)
        try:
            def update_item_colors(item):
                stored_name = item.data(0, Qt.ItemDataRole.UserRole)
                if stored_name:  # Only update leaf items (images)
                    if main_window.mask_exists(stored_name) or main_window.image_manager.has_mask(
                        stored_name
                    ):
                        item.setForeground(0, QColor("green"))
                    else:
                        item.setForeground(0, QColor("white"))

                # Recursively update children
                for i in range(item.childCount()):
                    update_item_colors(item.child(i))

            # Update all top-level items
            for i in range(main_window.file_list.topLevelItemCount()):
                update_item_colors(main_window.file_list.topLevelItem(i))
        finally:
            main_window.file_list.setUpdatesEnabled(True)


def highlight_saved_mask(main_window, image_path):
    """
    Highlight a tree item when its mask is saved.
    
    Args:
        main_window: MainWindow instance
        image_path: Path to the image whose mask was saved
    """
    image_name = os.path.splitext(os.path.basename(os.path.normpath(image_path)))[0]
    item = find_tree_item_by_image_name(main_window.file_list, image_name)
    if item:
        item.setForeground(0, QColor("green"))
    main_window.status_bar.showMessage(f"Mask saved for {image_name}", 3000)


def unhighlight_cleared_mask(main_window, image_path):
    """
    Remove highlight from a tree item when its mask is cleared.
    
    Args:
        main_window: MainWindow instance
        image_path: Path to the image whose mask was cleared
    """
    image_name = os.path.splitext(os.path.basename(os.path.normpath(image_path)))[0]
    item = find_tree_item_by_image_name(main_window.file_list, image_name)
    if item:
        item.setForeground(0, QColor("white"))

